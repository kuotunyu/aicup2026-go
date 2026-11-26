"""prereg33 exploration: unit tests (synthetic data and committed records only; never the answers folder).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg33/explore/test_p33.py
"""

import inspect
import os
import sys
import types

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p33_train as P  # noqa: E402

T, K, L25, L29 = P.T, P.K, P.L25, P.L29
REPO = P.REPO


# ---------------------------------------------------------------- players (PLAN.md 1)
def test_roles_are_a_union_h_by_prereg25_salt():
    keys = np.array([f"{lv}:p{i}" for i in range(400) for lv in ("1D", "5K")][:600] * 2)
    a, h, t = P.roles(keys)
    assert set(a.tolist()).isdisjoint(h.tolist())
    assert np.array_equal(t, np.arange(len(keys)))
    for i in a:
        assert L25.holdout_bit(keys[i]) == 0
    for i in h:
        assert L25.holdout_bit(keys[i]) == 1
    # a player's rows are all on one side
    side = {}
    for i in range(len(keys)):
        side.setdefault(keys[i], set()).add(int(i in set(h.tolist())))
    assert all(len(v) == 1 for v in side.values())
    assert L25.SALT == "c10-prereg25|holdout|"


def test_committed_counts_and_inner_disjoint():
    tr25 = L29.read_json(P.P25_TRAIN)
    p = tr25["players"]
    assert p["A_and_H"] == 0 and p["A_players"] + p["H_players"] == 7367
    # A u H is exactly prereg29's A / H universe (probe + V), whose seam D found no inner player (committed report)
    e29 = L29.read_json(os.path.join(REPO, "results", "prereg29", "explore", "eval_report.json"))
    sd = e29["seam_d"]
    assert sd["A_players"] == p["A_players"] and sd["H_players"] == p["H_players"]
    assert sd["train_and_I"] == 0 and sd["train_and_I_player_id"] == 0 and sd["ok"]


def test_same_players():
    tr25 = L29.read_json(P.P25_TRAIN)
    rec = {k: tr25["players"][k] for k in ("A_players", "A_games", "H_players", "H_games", "holdout_questions")}
    assert P.same_players(rec, tr25["players"])
    rec["H_games"] += 1
    assert not P.same_players(rec, tr25["players"])


# ---------------------------------------------------------------- steps per seed from the record
def test_stop_steps_from_prereg25_record():
    tr25 = L29.read_json(P.P25_TRAIN)
    st = P.stop_steps(tr25)
    assert st == {0: 11000, 1: 7500, 2: 10500}
    assert st == {s: tr25["seeds"][str(s)]["best_step"] for s in P.SEEDS}


def test_stop_steps_refuses_off_schedule():
    bad = {"seeds": {"0": {"best_step": 11000}, "1": {"best_step": 7501}, "2": {"best_step": 500}}}
    with pytest.raises(SystemExit):
        P.stop_steps(bad)
    bad["seeds"]["1"]["best_step"] = 15500
    with pytest.raises(SystemExit):
        P.stop_steps(bad)


def test_smoke_stop_steps_on_smoke_grid():
    sm = P.smoke_stop_steps(200)
    assert sm["seam"] == 200
    assert all(v % K.SMOKE["eval_every"] == 0 and 1 <= v <= K.SMOKE["steps"] for v in sm["seeds"].values())


# ---------------------------------------------------------------- mu / sigma = S25's
def test_standardisation_is_s25s():
    tr25 = L29.read_json(P.P25_TRAIN)
    assert T.r.sha_raw(T.Q09_MAP) == tr25["q09_map_sha256"]
    src = inspect.getsource(P.cmd_train)
    assert 'rows.standardised(mp["mu_x"], mp["sd_x"], smoke)' in src
    assert "mp = FW.load_map(T.Q09_MAP)" in src
    # S25 used the same call (p25_train.cmd_train)
    assert 'rows.standardised(mp["mu_x"], mp["sd_x"], smoke)' in inspect.getsource(P.P25T.cmd_train)
    mp = P.FW.load_map(T.Q09_MAP)
    assert mp["mu_x"].shape == (K.D_IN,) and mp["sd_x"].shape == (K.D_IN,) and np.all(mp["sd_x"] > 0)


def test_settings_are_prereg25s():
    tr25 = L29.read_json(P.P25_TRAIN)
    want = L29.registered_settings(K)
    assert all(tr25["settings"][k] == v for k, v in want.items())
    assert tr25["settings"]["seeds"] == list(P.SEEDS) and tr25["settings"]["steps"] == K.STEPS == 15000


def test_model_names_never_overwrite_s25():
    names = {P.model_name(s) for s in P.SEEDS} | {P.model_name(0, seam=True)}
    assert len(names) == 4 and all(n.startswith("s33_") for n in names)
    assert os.path.abspath(P.MODELS) != os.path.abspath(L29.S25_DIR)
    assert P.MODEL_FORMAT == L29.S25_FORMAT
    assert T.STOP == P.STOP


# ---------------------------------------------------------------- once guard (eval)
def test_once_guard(tmp_path):
    import p33_eval as E
    out, st = str(tmp_path / "eval_report.json"), str(tmp_path / "eval_started.json")
    a = types.SimpleNamespace(rerun=False, reason=None)
    cs = {"head": "h2"}
    assert E.once_guard(a, cs, out, st) is None
    T.write_json(st, {"code": {"head": "h1"}})
    with pytest.raises(SystemExit):
        E.once_guard(a, cs, out, st)
    with pytest.raises(SystemExit):                      # --rerun needs a reason
        E.once_guard(types.SimpleNamespace(rerun=True, reason=None), cs, out, st)
    with pytest.raises(SystemExit):                      # same commit as the earlier run
        E.once_guard(types.SimpleNamespace(rerun=True, reason="bug"), {"head": "h1"}, out, st)
    assert E.once_guard(types.SimpleNamespace(rerun=True, reason="bug"), cs, out, st) == ["h1"]


def test_eval_expected_settings_smoke_threads():
    import p33_eval as E
    assert E.expected_settings(False)["threads"] == 8
    assert E.expected_settings(True)["threads"] == P.SMOKE_THREADS == 4
    assert E.SEEDS == P.SEEDS


# ---------------------------------------------------------------- seam logic on synthetic data
def _pools():
    g = np.random.default_rng(7)
    keys, lv, col = [], [], []
    for k in range(K.L):
        for p in range(2):
            for c in ("B", "W"):
                for _ in range(21):
                    keys.append(f"{k}:p{p}")
                    lv.append(k)
                    col.append(c)
    keys, lv, col = np.array(keys), np.array(lv), np.array(col)
    X = g.standard_normal((len(keys), K.D_IN)).astype(np.float32)
    rows = np.arange(len(keys))
    pool = K.PlayerPool(keys, lv, col, rows)
    hold, _ = K.holdout_set(pool, 1, 3)
    return X, pool, hold


@pytest.fixture
def small(monkeypatch, tmp_path):
    import torch
    monkeypatch.setattr(K, "THREADS", 2)
    monkeypatch.setattr(K, "BATCH", 8)
    monkeypatch.setattr(T, "STOP", str(tmp_path / "STOP"))
    monkeypatch.setattr(P, "STOP", str(tmp_path / "STOP"))
    torch.set_num_threads(2)
    return _pools()


def _bytes(params, path):
    P.P25T.save_model(str(path), params, 0, {"step": 1}, True)
    with open(path, "rb") as f:
        return f.read()


def test_fixed_loop_equals_train_seed_with_evaluations(small, monkeypatch, tmp_path):
    """train_seed (evaluating every 2 steps; the scripted NLL makes step 4 the best) and train_fixed stopped at 4 of the
    same 6-step schedule give byte-identical parameters: the evaluations do not change the trajectory."""
    X, pool, hold = small
    real = T.holdout_nll
    script = iter([3.0, 1.0, 2.0])

    def fake(model, X32, hold_):
        _nll, lps = real(model, X32, hold_)      # the real evaluation runs (eval mode, no_grad), only the value is set
        return next(script), lps

    monkeypatch.setattr(T, "holdout_nll", fake)
    best, _c, st_ref, n_ref, _s = T.train_seed(0, X, pool, hold, 6, 2, "test")
    assert best["step"] == 4
    params, st, n_c, _secs, _loss, _state = P.train_fixed(0, X, pool, 6, 4, 2, "test")
    assert _bytes(params, tmp_path / "a.npz") == _bytes(best["params"], tmp_path / "b.npz")
    # the schedule length matters: stopping at 4 of a 4-step schedule is a different model
    p4, *_ = P.train_fixed(0, X, pool, 4, 4, 2, "test")
    assert _bytes(p4, tmp_path / "c.npz") != _bytes(params, tmp_path / "a.npz")
    # another seed differs
    p1, *_ = P.train_fixed(1, X, pool, 6, 4, 2, "test")
    assert _bytes(p1, tmp_path / "d.npz") != _bytes(params, tmp_path / "a.npz")


def test_train_one_and_smoke_seam(small, tmp_path):
    X, pool, hold = small
    out = tmp_path / "out"
    ref, step = P.seam_r_smoke(X, pool, hold, str(out), 4, 2)
    assert step in (2, 4)
    mpath = str(out / "models" / "seam.npz")
    sr = P.train_one(0, X, pool, hold, 4, step, 2, True, mpath, str(out / "seam_record.json"), False, "h", "seamR33")
    assert sr["port_check"]["ok"] and sr["stop_step"] == step and sr["players"] == "A"
    s = P.seam_r(mpath, True, ref)
    assert s["ok"] and s["bytes_equal_reference"]
    # a different stop step must fail the seam
    mpath2 = str(out / "models" / "seam2.npz")
    other = 2 if step == 4 else 4
    P.train_one(0, X, pool, hold, 4, other, 2, True, mpath2, str(out / "seam2_record.json"), False, "h", "seamR33")
    assert not P.seam_r(mpath2, True, ref)["ok"]
    # resume keeps the record of the same commit and refuses another commit
    again = P.train_one(0, X, pool, hold, 4, step, 2, True, mpath, str(out / "seam_record.json"), True, "h", "seamR33")
    assert again["model_sha256"] == sr["model_sha256"]
    with pytest.raises(SystemExit):
        P.train_one(0, X, pool, hold, 4, step, 2, True, mpath, str(out / "seam_record.json"), True, "h2", "seamR33")


def test_stop_file_stops_the_loop(small, tmp_path):
    X, pool, _hold = small
    with open(P.STOP, "w") as f:
        f.write("x")
    with pytest.raises(SystemExit):
        P.train_fixed(0, X, pool, 6, 4, 2, "test")


def test_guard_refuses_the_answers_folder():
    with pytest.raises(SystemExit):
        L29.guard(os.path.join(REPO, "data", "lo" + "cal", "x.csv"))
