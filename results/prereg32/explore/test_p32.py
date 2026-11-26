"""Unit tests of the prereg32 pilot (p32_pilot.py): synthetic data only. No test reads the repository's data folder,
starts a real run or judges real run files: the run loop, the refusals and the judge command run in pytest's
temporary folder with stand-in trainers, and the one real (tiny) training uses noise rows made here.

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1
    .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg32/explore/test_p32.py
"""

import copy
import datetime
import json
import math
import os
import re
import sys
import types

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import p32_pilot as P  # noqa: E402

K, T = P.K, P.T
HEAD = "a" * 40
ALL_RUNS = [f"E{i}_seed{s}" for i in range(7) for s in (0, 1)]
SMOKE_RUNS = ["E0_seed0", "E0_seed1", "E1_seed0", "E1_seed1"]


def plan_text():
    with open(os.path.join(HERE, "PLAN.md"), encoding="utf-8") as f:
        return f.read()


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w"):
        pass


def cfg_of(config, seed=0, smoke=False):
    return next(c for c in P.run_plan(smoke) if c["config"] == config and c["seed"] == seed)


def fake_record(cfg, head=HEAD, smoke=False, best=1.0, seconds=10.0, model_sha=None):
    """A valid finished-run record: 30 checkpoints, the best (first minimum) held-out NLL `best` at checkpoint 10."""
    ee = cfg["eval_every"]
    curve = [{"step": ee * (i + 1), "holdout_nll": best + 0.5 + 0.001 * i, "train_batch_loss": 0.7,
              "seconds": float(i)} for i in range(P.CHECKPOINTS)]
    curve[9]["holdout_nll"] = best
    return {"format": P.RUN_FORMAT, "smoke": bool(smoke), "config": cfg["config"], "seed": cfg["seed"],
            "steps": cfg["steps"], "wd": cfg["wd"], "threads": cfg["threads"], "eval_every": ee,
            "checkpoints": P.CHECKPOINTS, "applied": {"wd": cfg["wd"], "threads": cfg["threads"]},
            "torch_threads_after_training": cfg["threads"], "curve": curve, "best_step": curve[9]["step"],
            "best_step_fraction": curve[9]["step"] / cfg["steps"], "best_holdout_nll": best,
            "last_step": curve[-1]["step"], "last_step_holdout_nll": curve[-1]["holdout_nll"],
            "last_step_train_batch_loss": 0.7, "per_n_holdout_nll_best": {"5": {"questions": 1, "nll": best}},
            "seconds": seconds, "seconds_total": seconds, "port_check": {"ok": True}, "model_sha256": model_sha,
            "head": head}


def runs_of(**kw):
    """runs_of(E0=(a, b), E1=(c, None)) -> {(config, seed): record}; None: that seed did not finish."""
    out = {}
    for config, vals in kw.items():
        for seed, v in zip(P.SEEDS, vals):
            if v is not None:
                out[(config, seed)] = fake_record(cfg_of(config, seed), best=v)
    return out


class Trainer:
    """Stands in for train_fn(cfg, mpath): writes a small model file and returns a finished-run record."""

    def __init__(self, head=HEAD, smoke=False, best=None, stop_after=None, stop_path=None, fail_at=None,
                 seconds=10.0, clock=None):
        self.head, self.smoke, self.best, self.seconds = head, smoke, best or {}, seconds
        self.stop_after, self.stop_path, self.fail_at, self.clock = stop_after, stop_path, fail_at, clock
        self.calls = []

    def __call__(self, cfg, mpath):
        name = P.run_name(cfg["config"], cfg["seed"])
        self.calls.append(name)
        if self.fail_at is not None and name == self.fail_at[0]:
            raise self.fail_at[1]
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        with open(mpath, "wb") as f:
            f.write(name.encode())
        if self.stop_after is not None and len(self.calls) == self.stop_after:
            touch(self.stop_path)
        if self.clock is not None:
            self.clock.now += datetime.timedelta(seconds=self.seconds)
        return fake_record(cfg, self.head, self.smoke, self.best.get(name, 1.0), self.seconds, T.sha_file(mpath))


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def loop(tmp, plan, trainer, resume=False, deadline=None, ref=None, smoke=True, head=HEAD, now_fn=None):
    """run_loop in a temporary folder -> (exit code, base folder)."""
    base = os.path.join(str(tmp), "runs")
    kw = {"now_fn": now_fn} if now_fn is not None else {}
    rc = P.run_loop(plan, base, os.path.join(base, "STOP"), os.path.join(base, "state.json"), head, smoke, resume,
                    deadline, trainer, ref=ref, **kw)
    return rc, base


def state_of(base):
    return read(os.path.join(base, "state.json"))


def synthetic(seed=0):
    """10 levels x (2 role-A players + 1 held-out player) x (22 black + 22 white games), 1,540 noise columns."""
    keys, y, col = [], [], []
    for lv in range(K.L):
        for p in range(3):
            for c in K.COLOURS:
                keys += [f"L{lv}:p{p}"] * 22
                y += [lv] * 22
                col += [c] * 22
    keys, y, col = np.array(keys), np.array(y, dtype=np.int64), np.array(col)
    X32 = np.random.default_rng(seed).standard_normal((len(keys), K.D_IN)).astype(np.float32)
    is_h = np.array([k.endswith("p2") for k in keys])
    pool_a = K.PlayerPool(keys, y, col, np.flatnonzero(~is_h))
    pool_h = K.PlayerPool(keys, y, col, np.flatnonzero(is_h))
    hold = K.holdout_set(pool_h, 1, K.HOLD_SEED)[0]
    return X32, pool_a, hold


def fake_train_seed(seen, n_checkpoints=P.CHECKPOINTS, set_threads=True, error=None):
    """A stand-in for p19_train.train_seed that records what p19_lib holds while it is called."""
    def train_seed(seed, X32, pool_a, hold, steps, eval_every, log_prefix):
        import torch
        if set_threads:
            torch.set_num_threads(K.THREADS)
        seen.append({"wd": K.WD, "threads": K.THREADS, "prefix": log_prefix, "steps": steps,
                     "eval_every": eval_every, "seed": seed})
        if error is not None:
            raise error
        curve = [{"step": eval_every * (i + 1), "holdout_nll": 2.0 - 0.01 * i, "train_batch_loss": 1.5,
                  "seconds": float(i)} for i in range(n_checkpoints)]
        best = {"step": curve[-1]["step"], "nll": curve[-1]["holdout_nll"], "params": {"W1": np.zeros(2)}}
        return best, curve, {"questions": steps * K.BATCH}, {"5": steps}, 1.5
    return train_seed


@pytest.fixture(autouse=True)
def keep_torch_threads():
    import torch
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


@pytest.fixture
def stub_training(monkeypatch):
    """train_run with everything around train_seed stubbed -> the list the fake train_seed appends to."""
    monkeypatch.setattr(T, "port_check", lambda params, X32, hold: {"ok": True})
    monkeypatch.setattr(P, "per_n_nll", lambda params, X32, hold: ({"5": {"questions": 1, "nll": 1.0}}, 1.0))
    monkeypatch.setattr(P.P25T, "save_model", lambda path, params, seed, best, smoke: "f" * 64)
    return []


class Box:
    """State of the stubs behind the `box` fixture."""

    def __init__(self):
        self.code = {"head": HEAD, "dirty": "", "sha256_lf": {"x.py": "1" * 64}}
        self.ref = {0: 1.0, 1: 1.0}
        self.loads = 0
        self.load_error = None
        self.tr = Trainer()


@pytest.fixture
def box(tmp_path, monkeypatch):
    """The run / judge commands in a temporary folder: output paths repointed, code state, data loading, the
    seam-R reference and the training replaced by stand-ins (nothing of the repository is read or written)."""
    b = Box()
    data = os.path.join(str(tmp_path), "data32")
    for name, path in (("DATA", data), ("SMOKE_DIR", os.path.join(data, "smoke")),
                       ("STOP", os.path.join(data, "STOP")), ("STARTED", os.path.join(data, "pilot_started.json")),
                       ("STATE", os.path.join(data, "pilot_state.json")),
                       ("PILOT", os.path.join(str(tmp_path), "pilot.json"))):
        monkeypatch.setattr(P, name, path)
    monkeypatch.setattr(T, "STOP", T.STOP)          # cmd_run repoints it: restored after the test
    monkeypatch.setattr(P.r, "blas_env_check", lambda *a, **k: None)
    monkeypatch.setattr(P, "code_state", lambda: copy.deepcopy(b.code))
    monkeypatch.setattr(P, "reference_nll", lambda: dict(b.ref))

    def load_data(smoke):
        b.loads += 1
        if b.load_error is not None:
            raise b.load_error
        return "X32", "pool_a", "hold", {"players": {"A_players": 2}}

    def train_run(cfg, mpath, X32, pool_a, hold, smoke, head):
        assert (X32, pool_a, hold) == ("X32", "pool_a", "hold")
        b.tr.smoke, b.tr.head = smoke, head
        return b.tr(cfg, mpath)

    monkeypatch.setattr(P, "load_data", load_data)
    monkeypatch.setattr(P, "train_run", train_run)
    return b


# ---------------------------------------------------------------- the registered table (PLAN.md 1)
class TestTable:
    def test_configs_equal_the_plan_table_and_order(self):
        rows = re.findall(r"^\| (E\d) \| ([\d,]+) \| ([\d.]+) \|", plan_text(), flags=re.M)
        got = tuple((c, int(s.replace(",", "")), float(w)) for c, s, w in rows)
        assert got == P.CONFIGS[1:]
        assert [c for c, _s, _w in got] == ["E1", "E2", "E3", "E4", "E5", "E6"]

    def test_control_is_the_s25_setting_and_comes_first(self):
        text = plan_text()
        assert "wd 0.01、15,000 步、30 個檢查點" in text and "代號 E0" in text and "最先重跑" in text
        assert P.CONFIGS[0] == ("E0", 15000, 0.01) == (P.CONTROL, K.STEPS, K.WD)
        assert (K.WD, K.THREADS) == (P.REG_WD, P.REG_THREADS) == (0.01, 8)

    def test_threads_seeds_thresholds_as_registered(self):
        text = plan_text()
        assert "4 個 torch 執行緒" in text and P.THREADS == 4
        assert "種子 0 和 1" in text and P.SEEDS == (0, 1)
        assert "**30 個檢查點**" in text and P.CHECKPOINTS == 30
        assert "Δ ≤ **−0.020**" in text and P.GO_DELTA == -0.020
        assert "低至少 0.010" in text and P.GO_SEED == 0.010
        assert "0.02 以內" in text and P.SEAM_R_TOL == 0.02
        assert "300 步" in text and P.SMOKE_STEPS == 300 == K.SMOKE["steps"]

    def test_run_order(self):
        names = [P.run_name(c["config"], c["seed"]) for c in P.run_plan()]
        assert names == ALL_RUNS and len(names) == 14

    def test_30_checkpoints_for_every_config(self):
        plan = P.run_plan()
        assert [c["eval_every"] for c in plan] == [500] * 8 + [2000] * 4 + [5000] * 2
        for c in plan:
            assert c["steps"] % c["eval_every"] == 0 and c["steps"] // c["eval_every"] == 30
            assert len(range(c["eval_every"], c["steps"] + 1, c["eval_every"])) == 30
            assert c["threads"] == 4
        assert {c["config"]: (c["steps"], c["wd"]) for c in plan} == {c: (s, w) for c, s, w in P.CONFIGS}

    def test_control_checkpoints_are_the_s25_ones(self):
        assert cfg_of("E0")["eval_every"] == K.EVAL_EVERY == 500

    def test_smoke_plan(self):
        plan = P.run_plan(smoke=True)
        assert [P.run_name(c["config"], c["seed"]) for c in plan] == SMOKE_RUNS
        assert all(c["steps"] == 300 and c["eval_every"] == 10 and c["threads"] == 4 for c in plan)
        assert [c["wd"] for c in plan] == [0.01, 0.01, 0.1, 0.1]

    def test_steps_not_a_multiple_of_30_are_refused(self, monkeypatch):
        monkeypatch.setattr(P, "SMOKE_STEPS", 310)
        with pytest.raises(ValueError):
            P.run_plan(smoke=True)

    def test_paths(self):
        assert P.run_file("b", "E4", 1) == os.path.join("b", "run_E4_seed1.json")
        assert P.model_file("b", "E4", 1) == os.path.join("b", "models", "p32_E4_seed1.npz")
        assert P.DATA.replace("\\", "/").endswith("/data/prereg32") and P.SMOKE_DIR == os.path.join(P.DATA, "smoke")
        assert T.STOP == P.STOP == os.path.join(P.DATA, "STOP")
        assert P.PILOT == os.path.join(HERE, "pilot.json")
        assert (P.EXIT_DONE, P.EXIT_STOP, P.EXIT_DEADLINE, P.EXIT_SEAM_R) == (0, 3, 4, 5)


# ---------------------------------------------------------------- wd / threads patch
class TestPatch:
    def test_applied_inside_restored_outside(self):
        assert (K.WD, K.THREADS) == (0.01, 8)
        with P.patched(K, 0.3, 4) as applied:
            assert (K.WD, K.THREADS) == (0.3, 4) and applied == {"wd": 0.3, "threads": 4}
        assert (K.WD, K.THREADS) == (0.01, 8)

    def test_restored_after_an_exception(self):
        with pytest.raises(RuntimeError):
            with P.patched(K, 1.0, 4):
                assert (K.WD, K.THREADS) == (1.0, 4)
                raise RuntimeError("boom")
        assert (K.WD, K.THREADS) == (0.01, 8)

    def test_source_guard(self):
        assert P.train_seed_reads_at_call_time() is True

        def other(seed):
            return seed
        assert P.train_seed_reads_at_call_time(other) is False
        clone = types.FunctionType(T.train_seed.__code__, {"K": object()})      # same source, another K
        assert P.train_seed_reads_at_call_time(clone) is False

    def test_train_run_applies_wd_and_threads_only_inside(self, monkeypatch, tmp_path, stub_training):
        monkeypatch.setattr(T, "train_seed", fake_train_seed(stub_training))
        cfg = cfg_of("E3", 1)
        rec = P.train_run(cfg, str(tmp_path / "m.npz"), None, None, None, False, HEAD)
        assert stub_training == [{"wd": 1.0, "threads": 4, "prefix": "pilot E3", "steps": 15000, "eval_every": 500,
                                  "seed": 1}]
        assert (K.WD, K.THREADS) == (0.01, 8)
        assert rec["applied"] == {"wd": 1.0, "threads": 4} and rec["torch_threads_after_training"] == 4
        assert rec["checkpoints"] == 30 and rec["best_step"] == 15000 == rec["last_step"]
        assert rec["last_step_train_batch_loss"] == 1.5 and rec["seconds"] == 1.5 and rec["smoke"] is False
        assert rec["model_sha256"] == "f" * 64 and rec["head"] == HEAD
        assert P.run_file_problems(rec, cfg, HEAD, False) == []
        assert json.loads(json.dumps(rec)) == rec

    def test_train_run_smoke_prefix(self, monkeypatch, tmp_path, stub_training):
        monkeypatch.setattr(T, "train_seed", fake_train_seed(stub_training))
        cfg = cfg_of("E1", 0, smoke=True)
        rec = P.train_run(cfg, str(tmp_path / "m.npz"), None, None, None, True, HEAD)
        assert stub_training[0] == {"wd": 0.1, "threads": 4, "prefix": "smoke E1", "steps": 300, "eval_every": 10,
                                    "seed": 0}
        assert rec["smoke"] is True and P.run_file_problems(rec, cfg, HEAD, True) == []

    def test_restored_after_an_error_in_training(self, monkeypatch, tmp_path, stub_training):
        monkeypatch.setattr(T, "train_seed", fake_train_seed(stub_training, error=RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            P.train_run(cfg_of("E2"), str(tmp_path / "m.npz"), None, None, None, False, HEAD)
        assert stub_training[0]["wd"] == 0.3 and stub_training[0]["threads"] == 4
        assert (K.WD, K.THREADS) == (0.01, 8)

    def test_threads_not_applied_is_an_error(self, monkeypatch, tmp_path, stub_training):
        import torch
        torch.set_num_threads(2)
        monkeypatch.setattr(T, "train_seed", fake_train_seed(stub_training, set_threads=False))
        with pytest.raises(SystemExit, match="not applied"):
            P.train_run(cfg_of("E1"), str(tmp_path / "m.npz"), None, None, None, False, HEAD)
        assert (K.WD, K.THREADS) == (0.01, 8)

    def test_29_checkpoints_are_an_error(self, monkeypatch, tmp_path, stub_training):
        monkeypatch.setattr(T, "train_seed", fake_train_seed(stub_training, n_checkpoints=29))
        with pytest.raises(SystemExit, match="registered checkpoints"):
            P.train_run(cfg_of("E1"), str(tmp_path / "m.npz"), None, None, None, False, HEAD)

    def test_failed_port_check_is_an_error(self, monkeypatch, tmp_path, stub_training):
        monkeypatch.setattr(T, "train_seed", fake_train_seed(stub_training))
        monkeypatch.setattr(T, "port_check", lambda params, X32, hold: {"ok": False})
        with pytest.raises(SystemExit, match="port check failed"):
            P.train_run(cfg_of("E1"), str(tmp_path / "m.npz"), None, None, None, False, HEAD)

    def test_real_training_tiny(self, monkeypatch, tmp_path):
        """The imported train_seed itself on synthetic rows: AdamW gets the config's weight decay, torch runs on
        4 threads, 30 checkpoints, a model file, and p19_lib is back at the registered values afterwards."""
        import torch
        seen, real = {}, torch.optim.AdamW

        def adamw(params, **kw):
            seen.update(kw)
            return real(params, **kw)
        monkeypatch.setattr(torch.optim, "AdamW", adamw)
        monkeypatch.setattr(T, "STOP", str(tmp_path / "STOP"))
        X32, pool_a, hold = synthetic()
        cfg = {"config": "E3", "seed": 0, "steps": 30, "wd": 1.0, "eval_every": 1, "threads": 4}
        mpath = str(tmp_path / "models" / "m.npz")
        rec = P.train_run(cfg, mpath, X32, pool_a, hold, True, HEAD)
        assert seen["weight_decay"] == 1.0 and seen["lr"] == K.LR
        assert rec["torch_threads_after_training"] == 4 == torch.get_num_threads()
        assert (K.WD, K.THREADS) == (0.01, 8)
        assert rec["checkpoints"] == 30 and [c["step"] for c in rec["curve"]] == list(range(1, 31))
        assert os.path.exists(mpath) and rec["model_sha256"] == T.sha_file(mpath)
        assert rec["port_check"]["ok"] is True
        assert abs(rec["best_holdout_nll_numpy64"] - rec["best_holdout_nll"]) < 1e-3
        assert set(rec["per_n_holdout_nll_best"]) == {"5", "10", "15", "20"}
        assert sum(v["questions"] for v in rec["per_n_holdout_nll_best"].values()) == 40
        assert P.run_file_problems(rec, cfg, HEAD, True) == []

    def test_low_priority(self):
        if os.name != "nt":
            assert P.low_priority() == {"requested": True, "set": False, "note": "not Windows"}
            return
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        k32.GetPriorityClass.restype = ctypes.c_uint32
        h = k32.GetCurrentProcess()
        before = int(k32.GetPriorityClass(h))
        try:
            out = P.low_priority()
            assert out["set"] is True and out["priority_class"] == P.BELOW_NORMAL == 0x4000
        finally:
            k32.SetPriorityClass(h, before)


# ---------------------------------------------------------------- judge (PLAN.md 2), pure function
REF1 = {0: 1.0, 1: 1.0}
UP = math.nextafter


class TestJudge:
    def test_delta_exactly_at_the_threshold_is_go(self):
        res = P.judge(runs_of(E0=(0.04, 0.04), E1=(0.02, 0.02)), {0: 0.04, 1: 0.04})
        assert res["configs"]["E1"]["delta"] == -0.020          # exact in binary: 0.04 is 2 x 0.02
        assert res["verdict"] == "GO" and res["candidate"] == "E1"

    def test_delta_one_ulp_short_is_no_go(self):
        v = UP(0.02, 1.0)
        res = P.judge(runs_of(E0=(0.04, 0.04), E1=(v, v)), {0: 0.04, 1: 0.04})
        c = res["configs"]["E1"]
        assert -0.020 < c["delta"] < -0.0199999 and c["delta_ok"] is False and c["seeds_ok"] is True
        assert res["verdict"] == "NO-GO" and res["candidate"] is None and res["passing"] == []

    def test_seed_exactly_at_the_threshold_is_go(self):
        res = P.judge(runs_of(E0=(0.02, 1.0), E1=(0.01, 0.9)), {0: 0.02, 1: 1.0})
        assert res["configs"]["E1"]["seed_minus_control"]["0"] == -0.010
        assert res["verdict"] == "GO" and res["candidate"] == "E1"

    def test_seed_one_ulp_short_is_no_go(self):
        res = P.judge(runs_of(E0=(0.02, 1.0), E1=(UP(0.01, 1.0), 0.9)), {0: 0.02, 1: 1.0})
        c = res["configs"]["E1"]
        assert c["delta_ok"] is True and c["seeds_ok"] is False and c["passes_thresholds"] is False
        assert res["verdict"] == "NO-GO"

    def test_one_seed_only_0_0099_lower_is_no_go(self):
        res = P.judge(runs_of(E0=(1.0, 1.0), E1=(1.0 - 0.0099, 1.0 - 0.0501)), REF1)
        c = res["configs"]["E1"]
        assert c["delta"] < -0.029 and c["delta_ok"] is True and c["seeds_ok"] is False
        assert res["verdict"] == "NO-GO" and res["candidate"] is None
        res = P.judge(runs_of(E0=(1.0, 1.0), E1=(1.0 - 0.0101, 1.0 - 0.0501)), REF1)
        assert res["verdict"] == "GO" and res["candidate"] == "E1"

    def test_the_smallest_delta_is_the_candidate(self):
        res = P.judge(runs_of(E0=(1.0, 1.0), E1=(0.97, 0.97), E4=(0.95, 0.95), E5=(0.96, 0.96), E6=(0.99, 0.99)),
                      REF1)
        assert res["verdict"] == "GO" and res["candidate"] == "E4" and res["passing"] == ["E4", "E5", "E1"]
        assert res["configs"]["E6"]["passes_thresholds"] is False and res["not_run"] == ["E2", "E3"]

    def test_a_tie_goes_to_the_earlier_config(self):
        res = P.judge(runs_of(E0=(1.0, 1.0), E4=(0.95, 0.95), E2=(0.95, 0.95)), REF1)
        assert res["candidate"] == "E2" and res["passing"] == ["E2", "E4"]

    def test_a_better_delta_that_fails_per_seed_is_not_the_candidate(self):
        res = P.judge(runs_of(E0=(1.0, 1.0), E1=(0.97, 0.97), E2=(0.80, 0.995)), REF1)
        assert res["configs"]["E2"]["delta"] < res["configs"]["E1"]["delta"]
        assert res["configs"]["E2"]["delta_ok"] is True and res["configs"]["E2"]["seeds_ok"] is False
        assert res["candidate"] == "E1" and res["passing"] == ["E1"]

    def test_a_config_with_one_finished_seed_is_not_run(self):
        res = P.judge(runs_of(E0=(1.0, 1.0), E1=(0.5, None), E2=(None, 0.5)), REF1)
        for config, fin in (("E1", [0]), ("E2", [1]), ("E3", [])):
            c = res["configs"][config]
            assert c["status"] == "not run" and c["seeds_finished"] == fin and "delta" not in c
        assert res["not_run"] == ["E1", "E2", "E3", "E4", "E5", "E6"]
        assert res["verdict"] == "NO-GO" and res["candidate"] is None

    def test_missing_control_is_an_error(self):
        for runs in (runs_of(E1=(0.9, 0.9)), runs_of(E0=(1.0, None), E1=(0.9, 0.9)),
                     runs_of(E0=(None, 1.0), E1=(0.9, 0.9)), {}):
            with pytest.raises(P.JudgeError):
                P.judge(runs, REF1)
        with pytest.raises(P.JudgeError):
            P.judge(runs_of(E0=(1.0, 1.0)), {0: 1.0})

    def test_seam_r_failure_is_stop(self):
        res = P.judge(runs_of(E0=(1.0, 1.0), E1=(0.9, 0.9)), {0: 0.97, 1: 1.0})
        assert res["verdict"] == "STOP" and res["candidate"] is None and res["passing"] == ["E1"]
        sr = res["seam_r"]
        assert sr["ok"] is False and sr["seeds"]["0"]["ok"] is False and sr["seeds"]["1"]["ok"] is True
        assert abs(sr["seeds"]["0"]["diff"] - 0.03) < 1e-12

    def test_seam_r_boundary(self):
        assert P.seam_r({0: 0.04, 1: 0.02}, {0: 0.02, 1: 0.04})["ok"] is True       # |diff| exactly 0.02
        assert P.seam_r({0: 0.04, 1: 0.04}, {0: 0.02, 1: UP(0.02, 0.0)})["ok"] is False
        assert P.seam_r({0: UP(0.02, 0.0), 1: 0.04}, {0: 0.04, 1: 0.04})["ok"] is False
        out = P.seam_r({0: 1.012, 1: 1.0}, {0: 1.0, 1: 1.019})
        assert out["ok"] is True and out["tol"] == 0.02 and out["seeds"]["1"]["diff"] < 0

    def test_record_only_items(self):
        res = P.judge(runs_of(E0=(1.0, 1.0), E1=(0.9, 0.9)), REF1)
        for item in (res["control"]["record_only"]["0"], res["configs"]["E1"]["record_only"]["1"]):
            assert set(item) >= {"best_step", "best_step_fraction", "last_step_holdout_nll",
                                 "last_step_train_batch_loss", "overfit_gap_last_step",
                                 "per_n_holdout_nll_best"}
        ro = res["configs"]["E1"]["record_only"]["0"]
        assert ro["best_step"] == 5000 and ro["last_step_train_batch_loss"] == 0.7
        assert ro["overfit_gap_last_step"] == ro["last_step_holdout_nll"] - 0.7
        assert res["thresholds"] == {"delta": -0.020, "per_seed": 0.010, "seam_r_tol": 0.02}
        assert res["control"]["mean"] == 1.0 and res["configs"]["E1"]["mean"] == 0.9

    def test_pure(self):
        runs = runs_of(E0=(1.0, 1.01), E1=(0.9, 0.95), E5=(0.97, None))
        before = copy.deepcopy(runs)
        a, b = P.judge(runs, REF1), P.judge(runs, REF1)
        assert a == b and runs == before and json.loads(json.dumps(a)) == a


# ---------------------------------------------------------------- deadline
D = datetime.datetime(2026, 11, 11, 8, 0)
FIN = [{"steps": 16384, "seconds_total": 1024.0, "threads": 4}]         # 0.0625 s per step exactly


def sec(n):
    return datetime.timedelta(seconds=n)


class TestDeadline:
    def test_parse(self):
        assert P.parse_deadline("2026-11-11T08:00") == D and P.parse_deadline(None) is None
        aware = P.parse_deadline("2026-11-11T08:00+08:00")
        want = datetime.datetime.fromisoformat("2026-11-11T08:00+08:00").astimezone().replace(tzinfo=None)
        assert aware.tzinfo is None and aware == want

    def test_no_deadline_always_starts(self):
        d = P.deadline_decision(D, None, 150000, FIN)
        assert d["start"] is True and d["reason"] == "no deadline"

    def test_passed_or_reached_does_not_start(self):
        for now in (D, D + sec(1)):
            for fin in ([], FIN):
                d = P.deadline_decision(now, D, 15000, fin)
                assert d["start"] is False and d["reason"] == "the deadline has passed"

    def test_without_a_measurement_only_the_deadline_counts(self):
        d = P.deadline_decision(D - sec(1), D, 150000, [])
        assert d["start"] is True and d["estimate_seconds"] is None and "no estimate" in d["reason"]

    def test_estimate_boundary(self):
        assert P.seconds_per_step(FIN) == 0.0625
        for back, start in ((3751, True), (3750, True), (3749, False)):       # 60,000 steps = 3750 s
            d = P.deadline_decision(D - sec(back), D, 60000, FIN)
            assert d["estimate_seconds"] == 3750.0 and d["seconds_per_step"] == 0.0625
            assert d["start"] is start, back
        assert P.deadline_decision(D - sec(3749), D, 60000, FIN)["reason"] == "the estimated end is past the deadline"
        assert P.deadline_decision(D - sec(3750), D, 60000, FIN)["projected_end"] == "2026-11-11T08:00:00"

    def test_rate_is_the_aggregate_of_the_same_thread_count(self):
        fin = [{"steps": 15000, "seconds_total": 900.0, "threads": 4},
               {"steps": 60000, "seconds_total": 4800.0, "threads": 4},
               {"steps": 15000, "seconds_total": 1.0, "threads": 8}]
        assert P.seconds_per_step(fin) == 5700.0 / 75000
        assert P.seconds_per_step(fin, threads=8) == 1.0 / 15000
        assert P.seconds_per_step([]) is None and P.seconds_per_step(fin, threads=2) is None

    def test_loop_skips_the_long_run_and_everything_after_it(self, tmp_path):
        plan = [cfg_of("E0"), cfg_of("E6"), cfg_of("E1")]           # 15,000 / 150,000 / 15,000 steps
        clock = Clock(D - sec(3600))
        tr = Trainer(seconds=900.0, clock=clock)                    # 0.06 s per step
        rc, base = loop(tmp_path, plan, tr, deadline=D, smoke=False, now_fn=clock)
        assert rc == P.EXIT_DEADLINE == 4 and tr.calls == ["E0_seed0"]
        st = state_of(base)
        assert st["status"] == "stopped: deadline" and st["finished"] == ["E0_seed0"]
        assert st["not_started"] == [
            {"run": "E6_seed0", "reason": "the estimated end is past the deadline"},
            {"run": "E1_seed0", "reason": "an earlier run was not started for time: nothing later is started"}]
        assert [(x["run"], x["start"]) for x in st["decisions"]] == [("E0_seed0", True), ("E6_seed0", False)]
        assert st["decisions"][1]["estimate_seconds"] == 9000.0
        assert not os.path.exists(P.run_file(base, "E6", 0)) and not os.path.exists(P.run_file(base, "E1", 0))

    def test_loop_runs_what_fits(self, tmp_path):
        plan = [cfg_of("E0"), cfg_of("E1"), cfg_of("E2"), cfg_of("E3")]
        clock = Clock(D - sec(2700))                                # room for exactly three 900 s runs
        tr = Trainer(seconds=900.0, clock=clock)
        rc, base = loop(tmp_path, plan, tr, deadline=D, smoke=False, now_fn=clock)
        assert rc == 4 and tr.calls == ["E0_seed0", "E1_seed0", "E2_seed0"]
        assert state_of(base)["not_started"] == [{"run": "E3_seed0", "reason": "the deadline has passed"}]

    def test_loop_after_the_deadline_starts_nothing(self, tmp_path):
        tr = Trainer()
        rc, base = loop(tmp_path, P.run_plan(), tr, deadline=D, smoke=False, now_fn=Clock(D + sec(1)))
        st = state_of(base)
        assert rc == 4 and tr.calls == [] and [x["run"] for x in st["not_started"]] == ALL_RUNS
        assert st["not_started"][0]["reason"] == "the deadline has passed"


# ---------------------------------------------------------------- run loop: resume, STOP, errors, seam R
class TestLoop:
    def test_fresh_plan_runs_everything_in_order(self, tmp_path):
        tr = Trainer(smoke=True)
        rc, base = loop(tmp_path, P.run_plan(True), tr)
        st = state_of(base)
        assert rc == P.EXIT_DONE == 0 and tr.calls == SMOKE_RUNS
        assert st["status"] == "finished" and st["finished"] == SMOKE_RUNS and st["kept"] == []
        assert st["plan"] == SMOKE_RUNS and st["not_started"] == [] and st["format"] == P.STATE_FORMAT
        for c in P.run_plan(True):
            rec = read(P.run_file(base, c["config"], c["seed"]))
            assert P.run_file_problems(rec, c, HEAD, True) == []
            assert rec["model_sha256"] == T.sha_file(P.model_file(base, c["config"], c["seed"]))

    def test_resume_skips_finished_runs(self, tmp_path):
        plan = P.run_plan(True)
        stop = os.path.join(str(tmp_path), "runs", "STOP")
        tr = Trainer(smoke=True, stop_after=2, stop_path=stop)
        rc, base = loop(tmp_path, plan, tr)
        assert rc == P.EXIT_STOP == 3 and tr.calls == SMOKE_RUNS[:2]
        st = state_of(base)
        assert st["status"] == "stopped: STOP file" and [x["run"] for x in st["not_started"]] == SMOKE_RUNS[2:]
        kept = {n: read_bytes(P.run_file(base, n[:2], int(n[-1]))) for n in SMOKE_RUNS[:2]}
        os.remove(stop)
        tr2 = Trainer(smoke=True)
        rc, base = loop(tmp_path, plan, tr2, resume=True)
        st = state_of(base)
        assert rc == 0 and tr2.calls == SMOKE_RUNS[2:]                  # the finished runs were not trained again
        assert st["kept"] == SMOKE_RUNS[:2] and st["finished"] == SMOKE_RUNS[2:] and st["resume"] is True
        assert len(st["earlier_sessions"]) == 1 and st["earlier_sessions"][0]["status"] == "stopped: STOP file"
        assert kept == {n: read_bytes(P.run_file(base, n[:2], int(n[-1]))) for n in SMOKE_RUNS[:2]}

    def test_resume_with_everything_finished_trains_nothing(self, tmp_path):
        loop(tmp_path, P.run_plan(True), Trainer(smoke=True))
        tr = Trainer(smoke=True)
        rc, base = loop(tmp_path, P.run_plan(True), tr, resume=True)
        assert rc == 0 and tr.calls == [] and state_of(base)["kept"] == SMOKE_RUNS

    def test_finished_run_without_resume_is_refused(self, tmp_path):
        loop(tmp_path, P.run_plan(True), Trainer(smoke=True))
        tr = Trainer(smoke=True)
        with pytest.raises(SystemExit, match="never re-run"):
            loop(tmp_path, P.run_plan(True), tr)
        assert tr.calls == [] and state_of(os.path.join(str(tmp_path), "runs"))["status"] == "error"

    @pytest.mark.parametrize("damage", ["record", "model_sha", "model_missing", "head"])
    def test_resume_refuses_a_damaged_kept_run(self, tmp_path, damage):
        _, base = loop(tmp_path, P.run_plan(True), Trainer(smoke=True))
        rpath, mpath = P.run_file(base, "E0", 1), P.model_file(base, "E0", 1)
        rec = read(rpath)
        if damage == "record":
            rec["curve"] = rec["curve"][:-1]
            write(rpath, rec)
        elif damage == "model_sha":
            with open(mpath, "ab") as f:
                f.write(b"x")
        elif damage == "model_missing":
            os.remove(mpath)
        tr = Trainer(smoke=True)
        with pytest.raises(SystemExit, match="not usable"):
            loop(tmp_path, P.run_plan(True), tr, resume=True, head="b" * 40 if damage == "head" else HEAD)
        assert tr.calls == [] and state_of(base)["status"] == "error"

    def test_stop_file_before_the_first_run(self, tmp_path):
        touch(os.path.join(str(tmp_path), "runs", "STOP"))
        tr = Trainer(smoke=True)
        rc, base = loop(tmp_path, P.run_plan(True), tr)
        st = state_of(base)
        assert rc == 3 and tr.calls == [] and [x["run"] for x in st["not_started"]] == SMOKE_RUNS
        assert "before this run" in st["not_started"][0]["reason"] and st["exit_code"] == 3

    def test_stop_file_during_a_run_discards_it(self, tmp_path):
        stop = os.path.join(str(tmp_path), "runs", "STOP")
        calls = []

        def train(cfg, mpath):
            calls.append(P.run_name(cfg["config"], cfg["seed"]))
            if len(calls) == 2:
                touch(stop)
                raise SystemExit(f"STOP file {stop}: stopped at step 100")
            return Trainer(smoke=True)(cfg, mpath)

        rc, base = loop(tmp_path, P.run_plan(True), train)
        st = state_of(base)
        assert rc == 3 and calls == SMOKE_RUNS[:2] and st["finished"] == ["E0_seed0"]
        assert st["status"] == "stopped: STOP file" and not os.path.exists(P.run_file(base, "E0", 1))
        assert st["not_started"][0]["run"] == "E0_seed1" and "discarded" in st["not_started"][0]["reason"]
        assert st["not_started"][1] == {"run": "E1_seed0", "reason": "STOP file during an earlier run"}

    @pytest.mark.parametrize("error", [SystemExit("port check failed"), RuntimeError("boom"),
                                       SystemExit("STOP file x: stopped at step 100")])
    def test_other_errors_propagate(self, tmp_path, error):
        tr = Trainer(smoke=True, fail_at=("E0_seed1", error))       # no STOP file: not a STOP even if it says so
        with pytest.raises(type(error)):
            loop(tmp_path, P.run_plan(True), tr)
        base = os.path.join(str(tmp_path), "runs")
        st = state_of(base)
        assert st["status"] == "error" and st["finished"] == ["E0_seed0"] and st["error"]
        assert tr.calls == SMOKE_RUNS[:2] and not os.path.exists(P.run_file(base, "E0", 1))

    def test_seam_r_failure_stops_after_the_control(self, tmp_path):
        tr = Trainer(best={"E0_seed1": 1.05})
        rc, base = loop(tmp_path, P.run_plan(), tr, ref={0: 1.0, 1: 1.0}, smoke=False)
        st = state_of(base)
        assert rc == P.EXIT_SEAM_R == 5 and tr.calls == ALL_RUNS[:2] and st["status"] == "stopped: seam R"
        assert [x["run"] for x in st["not_started"]] == ALL_RUNS[2:] and st["seam_r"]["ok"] is False
        assert "seam R sanity failed" in st["not_started"][0]["reason"]

    def test_seam_r_within_tolerance_goes_on(self, tmp_path):
        tr = Trainer(best={"E0_seed1": 1.015})
        rc, base = loop(tmp_path, P.run_plan(), tr, ref={0: 1.0, 1: 1.0}, smoke=False)
        assert rc == 0 and tr.calls == ALL_RUNS and state_of(base)["seam_r"]["ok"] is True

    def test_seam_r_is_checked_again_on_resume(self, tmp_path):
        loop(tmp_path, P.run_plan(), Trainer(best={"E0_seed0": 1.03}), smoke=False)        # no reference: all run
        tr = Trainer()
        rc, base = loop(tmp_path, P.run_plan(), tr, resume=True, ref={0: 1.0, 1: 1.0}, smoke=False)
        assert rc == 5 and tr.calls == []


# ---------------------------------------------------------------- the run command
FAR = ["--deadline", "2999-01-01T00:00"]


class TestRunCommand:
    @pytest.mark.parametrize("dirty", [" M results/prereg32/explore/p32_pilot.py", "?? tools/p19_lib.py"])
    def test_real_run_refuses_dirty_code(self, box, dirty):
        box.code["dirty"] = dirty
        with pytest.raises(SystemExit, match="not committed and clean"):
            P.main(["run"])
        with pytest.raises(SystemExit, match="not committed and clean"):
            P.main(["run", "--resume"])
        assert box.loads == 0 and box.tr.calls == [] and not os.path.exists(P.DATA)     # nothing loaded or written

    def test_smoke_is_allowed_on_dirty_code_and_stays_in_its_folder(self, box):
        box.code["dirty"] = "?? results/prereg32/explore/p32_pilot.py"
        assert P.main(["run", "--smoke"]) == 0
        assert box.tr.calls == SMOKE_RUNS and os.listdir(P.DATA) == ["smoke"]
        st = read(os.path.join(P.SMOKE_DIR, "pilot_state.json"))
        assert st["smoke"] is True and st["status"] == "finished" and st["session"]["code_dirty"] is True
        assert st["session"]["seam_r_reference_prereg25_8_threads"] is None and not os.path.exists(P.STARTED)
        assert read(P.run_file(P.SMOKE_DIR, "E1", 1))["smoke"] is True
        assert P.main(["run", "--smoke"]) == 0                      # a smoke starts from nothing again
        assert box.tr.calls == SMOKE_RUNS * 2

    def test_real_run_then_judge_once(self, box):
        box.tr.best = {"E4_seed0": 0.95, "E4_seed1": 0.96}
        assert P.main(["run"] + FAR) == 0
        assert box.tr.calls == ALL_RUNS and box.loads == 1 and T.STOP == P.STOP
        started, st = read(P.STARTED), read(P.STATE)
        assert started["code"] == box.code and started["deadline"] == "2999-01-01T00:00"
        assert st["smoke"] is False and st["status"] == "finished" and st["deadline"] == "2999-01-01T00:00:00"
        assert st["seam_r"]["ok"] is True and st["session"]["priority"] == {"requested": False}
        assert not os.path.exists(P.SMOKE_DIR) and not os.path.exists(P.PILOT)
        assert P.main(["judge"]) == 0
        res = read(P.PILOT)
        assert (res["verdict"], res["candidate"], res["not_run"]) == ("GO", "E4", [])
        assert len(res["run_files_sha256"]) == 14 and res["runs_finished"] == ALL_RUNS
        assert res["code_same_as_run"] is True and res["killed_flag"] is False and res["run_head"] == HEAD
        before = read_bytes(P.PILOT)
        with pytest.raises(SystemExit, match="judged once"):
            P.main(["judge"])
        for args in (["run"], ["run", "--resume"]):
            with pytest.raises(SystemExit, match="was judged"):
                P.main(args)
        assert read_bytes(P.PILOT) == before and box.tr.calls == ALL_RUNS

    def test_run_is_started_once(self, box):
        assert P.main(["run"]) == 0
        with pytest.raises(SystemExit, match="started once"):
            P.main(["run"])
        assert box.tr.calls == ALL_RUNS and box.loads == 1

    def test_resume_needs_a_started_run(self, box):
        with pytest.raises(SystemExit, match="nothing to resume"):
            P.main(["run", "--resume"])
        assert box.loads == 0 and not os.path.exists(P.DATA)

    def test_stop_then_resume_finishes_the_rest(self, box):
        box.tr.stop_after, box.tr.stop_path = 3, P.STOP
        assert P.main(["run"]) == 3 and box.tr.calls == ALL_RUNS[:3]
        started = read_bytes(P.STARTED)
        with pytest.raises(SystemExit, match="delete it by hand"):
            P.main(["run", "--resume"])
        os.remove(P.STOP)
        box.tr.stop_after = None
        assert P.main(["run", "--resume"]) == 0
        st = read(P.STATE)
        assert box.tr.calls == ALL_RUNS and st["kept"] == ALL_RUNS[:3] and st["finished"] == ALL_RUNS[3:]
        assert read_bytes(P.STARTED) == started and len(st["earlier_sessions"]) == 1

    @pytest.mark.parametrize("change", ["head", "sha"])
    def test_resume_needs_the_same_commit_and_code(self, box, change):
        box.tr.stop_after, box.tr.stop_path = 1, P.STOP
        assert P.main(["run"]) == 3
        os.remove(P.STOP)
        if change == "head":
            box.code["head"] = "b" * 40
        else:
            box.code["sha256_lf"] = {"x.py": "2" * 64}
        with pytest.raises(SystemExit, match="same commit and code"):
            P.main(["run", "--resume"])
        assert box.tr.calls == ALL_RUNS[:1]

    def test_deadline_stop_is_final(self, box):
        assert P.main(["run", "--deadline", "2000-01-01T00:00"]) == 4
        st = read(P.STATE)
        assert box.tr.calls == [] and st["status"] == "stopped: deadline" and len(st["not_started"]) == 14
        with pytest.raises(SystemExit, match="time limit"):
            P.main(["run", "--resume"] + FAR)
        assert box.tr.calls == []

    def test_stop_file_refuses_the_start(self, box):
        touch(P.STOP)
        for args in (["run"], ["run", "--smoke"]):
            with pytest.raises(SystemExit, match="delete it by hand"):
                P.main(args)
        assert box.loads == 0 and not os.path.exists(P.STARTED) and not os.path.exists(P.STATE)

    def test_nothing_is_written_when_loading_fails(self, box):
        box.load_error = SystemExit("data differ")
        with pytest.raises(SystemExit, match="data differ"):
            P.main(["run"])
        assert box.loads == 1 and not os.path.exists(P.DATA)
        box.load_error = None
        assert P.main(["run"]) == 0                                 # so a first start is still possible

    def test_registered_values_guard(self, box, monkeypatch):
        monkeypatch.setattr(K, "WD", 0.1)
        with pytest.raises(SystemExit, match="registered S25 values"):
            P.main(["run", "--smoke"])
        monkeypatch.setattr(K, "WD", P.REG_WD)
        monkeypatch.setattr(K, "THREADS", 4)
        with pytest.raises(SystemExit, match="registered S25 values"):
            P.main(["run"])
        assert box.loads == 0

    def test_source_guard(self, box, monkeypatch):
        monkeypatch.setattr(P, "train_seed_reads_at_call_time", lambda fn=None: False)
        with pytest.raises(SystemExit, match="call time"):
            P.main(["run", "--smoke"])
        assert box.loads == 0

    def test_low_priority_flag(self, box, monkeypatch):
        called = []
        monkeypatch.setattr(P, "low_priority", lambda: called.append(1) or {"requested": True, "set": True})
        assert P.main(["run", "--smoke", "--low-priority"]) == 0
        st = read(os.path.join(P.SMOKE_DIR, "pilot_state.json"))
        assert called == [1] and st["session"]["priority"] == {"requested": True, "set": True}

    def test_torch_threads_are_four(self, box):
        import torch
        assert P.main(["run", "--smoke"]) == 0 and torch.get_num_threads() == P.THREADS == 4


# ---------------------------------------------------------------- the judge command
class TestJudgeCommand:
    def test_nothing_to_judge(self, box):
        with pytest.raises(SystemExit, match="no real run to judge"):
            P.main(["judge"])
        assert not os.path.exists(P.PILOT)

    def test_a_smoke_is_never_judged(self, box):
        assert P.main(["run", "--smoke"]) == 0
        with pytest.raises(SystemExit, match="no real run to judge"):
            P.main(["judge"])
        assert not os.path.exists(P.PILOT)

    def test_smoke_state_in_the_real_place_is_refused(self, box):
        assert P.main(["run"]) == 0
        st = read(P.STATE)
        st["smoke"] = True
        write(P.STATE, st)
        with pytest.raises(SystemExit, match="not the state of a real run"):
            P.main(["judge"])

    def test_dirty_code_is_refused(self, box):
        assert P.main(["run"]) == 0
        box.code["dirty"] = " M results/prereg32/explore/p32_pilot.py"
        with pytest.raises(SystemExit, match="not committed and clean"):
            P.main(["judge"])
        assert not os.path.exists(P.PILOT)

    def test_running_state_needs_the_killed_flag(self, box):
        box.tr.stop_after, box.tr.stop_path = 4, P.STOP
        assert P.main(["run"]) == 3
        st = read(P.STATE)
        st["status"] = "running"                                    # as left behind by a killed process
        write(P.STATE, st)
        with pytest.raises(SystemExit, match="still running"):
            P.main(["judge"])
        assert not os.path.exists(P.PILOT)
        assert P.main(["judge", "--killed"]) == 0
        res = read(P.PILOT)
        assert res["killed_flag"] is True and res["verdict"] == "NO-GO" and res["not_run"] == ["E2", "E3", "E4",
                                                                                               "E5", "E6"]

    def test_only_finished_configs_are_judged(self, box):
        box.tr.best = {"E1_seed0": 0.9, "E1_seed1": 0.9, "E2_seed0": 0.5}
        box.tr.stop_after, box.tr.stop_path = 5, P.STOP
        assert P.main(["run"]) == 3
        assert P.main(["judge"]) == 0
        res = read(P.PILOT)
        assert (res["verdict"], res["candidate"]) == ("GO", "E1")
        assert res["not_run"] == ["E2", "E3", "E4", "E5", "E6"]
        assert res["configs"]["E2"] == {"steps": 15000, "wd": 0.3, "seeds_finished": [0], "status": "not run"}
        assert res["configs"]["E6"]["seeds_finished"] == [] and res["runs_finished"] == ALL_RUNS[:5]
        assert res["pilot_state"]["status"] == "stopped: STOP file" and len(res["run_files_sha256"]) == 5

    def test_missing_control_seed_is_an_error(self, box):
        box.tr.stop_after, box.tr.stop_path = 1, P.STOP
        assert P.main(["run"]) == 3
        with pytest.raises(SystemExit, match="judge: .*needs both seeds"):
            P.main(["judge"])
        assert not os.path.exists(P.PILOT)

    def test_run_file_of_another_commit_is_refused(self, box):
        assert P.main(["run"]) == 0
        path = P.run_file(P.DATA, "E3", 1)
        rec = read(path)
        rec["head"] = "b" * 40
        write(path, rec)
        with pytest.raises(SystemExit, match="not a finished run"):
            P.main(["judge"])
        assert not os.path.exists(P.PILOT)

    def test_seam_r_failure_is_a_stop(self, box):
        box.tr.best = {"E0_seed0": 1.03}
        assert P.main(["run"]) == 5 and box.tr.calls == ALL_RUNS[:2]
        assert P.main(["judge"]) == 0
        res = read(P.PILOT)
        assert (res["verdict"], res["candidate"]) == ("STOP", None) and res["seam_r"]["ok"] is False
        assert res["pilot_state"]["status"] == "stopped: seam R" and len(res["not_run"]) == 6

    def test_code_change_after_the_run_is_recorded(self, box):
        assert P.main(["run"]) == 0
        box.code["sha256_lf"] = {"x.py": "2" * 64, "y.py": "3" * 64}
        assert P.main(["judge"]) == 0
        res = read(P.PILOT)
        assert res["code_same_as_run"] is False and res["code_files_changed_since_run"] == ["x.py", "y.py"]


# ---------------------------------------------------------------- run file checks
def damaged(key, value):
    cfg = cfg_of("E5", 1)
    rec = fake_record(cfg, model_sha="f" * 64)
    rec[key] = value
    return P.run_file_problems(rec, cfg, HEAD, False)


class TestRunFile:
    def test_valid_record(self):
        for smoke in (False, True):
            for cfg in P.run_plan(smoke):
                assert P.run_file_problems(fake_record(cfg, smoke=smoke), cfg, HEAD, smoke) == []

    @pytest.mark.parametrize("key,value", [
        ("format", "x"), ("smoke", True), ("config", "E4"), ("seed", 0), ("steps", 15000), ("wd", 0.1),
        ("threads", 8), ("eval_every", 500), ("checkpoints", 29), ("head", "b" * 40),
        ("applied", {"wd": 0.01, "threads": 4}), ("torch_threads_after_training", 8), ("port_check", {"ok": False}),
        ("best_holdout_nll", 0.5), ("best_step", 2000), ("last_step_holdout_nll", 0.0)])
    def test_one_wrong_field_is_a_problem(self, key, value):
        assert damaged(key, value) != []

    def test_truncated_curve_is_a_problem(self):
        cfg = cfg_of("E6", 0)
        rec = fake_record(cfg)
        rec["curve"] = rec["curve"][:-1]
        assert P.run_file_problems(rec, cfg, HEAD, False) != []

    def test_best_is_the_first_minimum(self):
        cfg = cfg_of("E1", 0)
        rec = fake_record(cfg)
        rec["curve"][20]["holdout_nll"] = rec["best_holdout_nll"]          # a tie later: the first one stays best
        assert P.run_file_problems(rec, cfg, HEAD, False) == []
        rec["best_step"] = rec["curve"][20]["step"]
        assert P.run_file_problems(rec, cfg, HEAD, False) != []


# ---------------------------------------------------------------- code state, launcher
class TestCodeState:
    def test_code_state_covers_own_and_imported_files(self, monkeypatch):
        monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "0")               # read-only git: no index refresh
        cs = P.code_state()
        assert re.fullmatch(r"[0-9a-f]{40}", cs["head"]) and isinstance(cs["dirty"], str)
        files = set(cs["sha256_lf"])
        assert {f"results/prereg32/explore/{f}" for f in P.OWN_FILES} <= files and set(P.IMPORTED) <= files
        for f in ("results/prereg19/explore/p19_lib.py", "results/prereg19/explore/p19_train.py",
                  "results/prereg25/explore/p25_train.py", "results/prereg31/explore/ram_watch.py",
                  "results/prereg25/explore/train_record.json"):
            assert f in files, f
        assert not any(f.startswith("data/") for f in files)        # nothing under data/ is hashed or listed
        assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in cs["sha256_lf"].values())


class TestLauncher:
    def lines(self):
        raw = read_bytes(os.path.join(HERE, "run_pilot.sh"))
        assert b"\r" not in raw and raw.startswith(b"#!/usr/bin/env bash\n") and raw.endswith(b"\n")
        return raw.decode("utf-8").split("\n")

    def test_launcher(self):
        lines = self.lines()
        code = [x for x in lines if x and not x.startswith("#")]
        assert code[0] == "cd /q/ || exit 1"
        assert "unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS" in code
        assert "export PYTHONDONTWRITEBYTECODE=1" in code and "LOG=data/prereg32/pilot_log.txt" in code
        assert "PY=/d/AI-Portfolio/competitions/aicup2026-go/.venv/Scripts/python.exe" in code
        runs = [x for x in code if "p32_pilot.py" in x]
        assert runs == ['"$PY" results/prereg32/explore/p32_pilot.py run --deadline 2026-11-11T08:00 '
                        '--low-priority "$@" >> "$LOG" 2>&1']
        text = "\n".join(code)
        assert "PYTHONUTF8" not in text and "data/local" not in text and "judge" not in text
        assert "--smoke" not in text and "git " in text and "git rev-parse HEAD" in text

    def test_ram_watch_writes_the_pilot_stop_file(self):
        code = "\n".join(x for x in self.lines() if x and not x.startswith("#"))
        i = code.index("results/prereg31/explore/ram_watch.py")
        assert i < code.index("p32_pilot.py")                       # the watch is started first
        assert "--stop data/prereg32/STOP" in code and "--out data/prereg32/ram_watch.jsonl" in code
        assert "data/prereg31/" not in code                         # nothing of prereg31 is written
        assert os.path.normpath(P.STOP) == os.path.normpath(os.path.join(P.REPO, "data", "prereg32", "STOP"))

    def test_gitattributes_pins_lf(self):
        with open(os.path.join(HERE, ".gitattributes"), encoding="utf-8") as f:
            assert "*.sh text eol=lf" in f.read().split("\n")
