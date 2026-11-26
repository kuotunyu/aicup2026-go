"""prereg32 pilot, phase 0 (competition 2): weight decay x training length of the S25 set model (PLAN.md).
Held-out (role H) NLL only: reads no inner data, never the answers folder; CPU, 4 torch threads; decides only
whether a phase-1 PLAN is worth writing.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg32/explore
    $S/p32_pilot.py run [--deadline ISO] [--low-priority]
                                  -> data/prereg32/run_{E0..E6}_seed{0,1}.json + models/p32_{E?}_seed{s}.npz, in the
                                     registered order E0 (control) then E1..E6, seeds 0 and 1 each (started once; the
                                     code must be committed and clean)
    $S/p32_pilot.py run --resume  -> after an interruption: keeps every finished run, continues with the next one
                                     (same commit and code as data/prereg32/pilot_started.json)
    $S/p32_pilot.py run --smoke   -> data/prereg32/smoke/ (features replaced by default_rng(12345) normal noise,
                                     300 steps, E0 and E1 only, held-out 5 per cell: pipeline and timing only,
                                     never a result)
    $S/p32_pilot.py judge         -> $S/pilot.json (pure function of the finished real run files; judged once)
    --deadline: local time; before a run starts its duration is estimated from the seconds per step of the finished
      runs of this pilot (same thread count); if it would end after the deadline it is not started and nothing later
      is started either (data/prereg32/pilot_state.json says which runs and why).
    --low-priority sets the process priority class to BELOW_NORMAL (no numeric effect).
    A file data/prereg32/STOP makes the run stop at the next held-out evaluation and before the next run.
    Exit codes of run: 0 all runs finished, 3 STOP file, 4 deadline, 5 seam R sanity failed.

Everything except weight decay, step count and thread count is S25's, imported unchanged: rows and the
standardisation (p25_train.Rows), player roles (p25_lib.holdout_bit), pools / held-out questions / sampler
(p19_lib), model, training loop, step choice and port check (p19_train.train_seed / port_check), model file
(p25_train.save_model). No HD input (plain 1,540 columns). p19_lib.WD and p19_lib.THREADS are set only inside
`patched` around train_seed / port_check and restored afterwards (train_seed reads both when it is called).
"""

import argparse
import contextlib
import datetime
import inspect
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P25 = os.path.join(REPO, "results", "prereg25", "explore")
P29 = os.path.join(REPO, "results", "prereg29", "explore")
for p in (os.path.join(REPO, "tools"), P25, P29, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p25_train as P25T  # noqa: E402  (Rows, save_model; imports p19_train as T, p19_lib as K, p25_lib)
import p29_lib as L29  # noqa: E402  (registered_settings, loaded_repo_modules, guarded read_json, paths)

T, K, FW, L25 = P25T.T, P25T.K, P25T.FW, P25T.L
r = T.r

# ---------------------------------------------------------------- registered constants (PLAN.md 1 / 2)
CONTROL = "E0"
CONFIGS = (            # (code, steps, weight decay); the order is the registered run order
    ("E0", 15000, 0.01),      # control = S25 settings, re-run here at 4 threads with 30 checkpoints
    ("E1", 15000, 0.1),
    ("E2", 15000, 0.3),
    ("E3", 15000, 1.0),
    ("E4", 60000, 0.1),
    ("E5", 60000, 0.3),
    ("E6", 150000, 0.3),
)
SEEDS = (0, 1)
CHECKPOINTS = 30               # eval_every = steps / 30 for every config
THREADS = 4                    # torch threads of this pilot (S25 registered 8)
REG_WD, REG_THREADS = 0.01, 8  # the registered S25 values: what p19_lib must hold outside `patched`
GO_DELTA = -0.020              # Delta(config) <= GO_DELTA (inclusive, no tolerance)
GO_SEED = 0.010                # and each seed at least this much below the same-seed control (inclusive)
SEAM_R_TOL = 0.02              # |E0 best NLL - the prereg25 8-thread best NLL| <= tol per seed, else STOP
SMOKE_STEPS = 300
SMOKE_CONFIGS = ("E0", "E1")

DATA = os.path.join(REPO, "data", "prereg32")
SMOKE_DIR = os.path.join(DATA, "smoke")
STOP = os.path.join(DATA, "STOP")
STARTED = os.path.join(DATA, "pilot_started.json")
STATE = os.path.join(DATA, "pilot_state.json")
PILOT = os.path.join(HERE, "pilot.json")
V_META = os.path.join(L29.V_PREP, "v_meta.npz")
OWN_FILES = ("PLAN.md", "p32_pilot.py", "test_p32.py", "run_pilot.sh", ".gitattributes")
IMPORTED = tuple(T.IMPORTED) + (
    "results/prereg19/explore/p19_train.py", "results/prereg19/explore/p19_lib.py",
    "results/prereg25/explore/p25_train.py", "results/prereg25/explore/p25_lib.py",
    "results/prereg25/explore/train_record.json",      # settings, players record and the seam-R reference values
    "results/prereg29/explore/p29_lib.py",
    "results/prereg31/explore/ram_watch.py")           # called by run_pilot.sh (parameterised; not imported)
RUN_FORMAT = "c10-prereg32 pilot run v1"
STARTED_FORMAT = "c10-prereg32 pilot started v1"
STATE_FORMAT = "c10-prereg32 pilot state v1"
PILOT_FORMAT = "c10-prereg32 pilot judge v1"
EXIT_DONE, EXIT_STOP, EXIT_DEADLINE, EXIT_SEAM_R = 0, 3, 4, 5
BELOW_NORMAL = 0x00004000


def point_stop():
    """The STOP file train_seed looks at is a module global of p19_train (p25_train pointed it to its own folder)."""
    T.STOP = STOP


point_stop()           # must stay after every import above


def log(*x):
    print(*x, flush=True)


def rel(path):
    try:
        return os.path.relpath(path, REPO).replace("\\", "/")
    except ValueError:          # a path on another drive (unit tests)
        return os.path.abspath(path).replace("\\", "/")


def code_state():
    files = [rel(os.path.join(HERE, f)) for f in OWN_FILES] + list(IMPORTED)
    files = sorted(set(files) | L29.loaded_repo_modules())
    missing = [x for x in files if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = T.git("status", "--porcelain", "--", *files)
    return {"head": T.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: T.sha_lf(os.path.join(REPO, *x.split("/"))) for x in files}}


def low_priority():
    """BELOW_NORMAL priority class for this process (scheduler rule; no numeric effect)."""
    if os.name != "nt":
        return {"requested": True, "set": False, "note": "not Windows"}
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.SetPriorityClass.restype = ctypes.c_int
    k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k32.GetPriorityClass.restype = ctypes.c_uint32
    h = k32.GetCurrentProcess()
    ok = bool(k32.SetPriorityClass(h, BELOW_NORMAL))
    return {"requested": True, "set": ok, "priority_class": int(k32.GetPriorityClass(h)),
            "below_normal": BELOW_NORMAL}


# ---------------------------------------------------------------- the plan of runs (PLAN.md 1)
def run_name(config, seed):
    return f"{config}_seed{seed}"


def run_file(base, config, seed):
    return os.path.join(base, f"run_{run_name(config, seed)}.json")


def model_file(base, config, seed):
    return os.path.join(base, "models", f"p32_{run_name(config, seed)}.npz")


def run_plan(smoke=False):
    """The runs in the registered order: E0 seeds 0 and 1, then E1..E6 seeds 0 and 1 each (smoke: E0 and E1 at
    300 steps). eval_every = steps / 30, so every run has exactly 30 checkpoints."""
    out = []
    for config, steps, wd in CONFIGS:
        if smoke and config not in SMOKE_CONFIGS:
            continue
        st = SMOKE_STEPS if smoke else steps
        if st % CHECKPOINTS:
            raise ValueError(f"{config}: {st} steps are not a multiple of {CHECKPOINTS}")
        for seed in SEEDS:
            out.append({"config": config, "seed": seed, "steps": st, "wd": wd, "eval_every": st // CHECKPOINTS,
                        "threads": THREADS})
    return out


# ---------------------------------------------------------------- the only difference from S25: wd, steps, threads
@contextlib.contextmanager
def patched(lib, wd, threads):
    """Temporarily set lib.WD and lib.THREADS (p19_lib); both are restored on exit, also after an exception."""
    old = (lib.WD, lib.THREADS)
    lib.WD, lib.THREADS = wd, threads
    try:
        yield {"wd": lib.WD, "threads": lib.THREADS}
    finally:
        lib.WD, lib.THREADS = old


def train_seed_reads_at_call_time(fn=None):
    """True when train_seed applies p19_lib.THREADS and p19_lib.WD at call time (so `patched` reaches them) and
    the module it reads them from is the one patched here."""
    fn = fn or T.train_seed
    src = inspect.getsource(fn)
    return bool("torch.set_num_threads(K.THREADS)" in src and "weight_decay=K.WD" in src
                and getattr(fn, "__globals__", {}).get("K") is K)


def roles(rows, hold_per):
    """A / H rows by p25_lib.holdout_bit, the pool of role A, the held-out questions and the players record: the
    statements p25_train.cmd_train has inline (not importable), built from the imported parts; a real run requires
    the record to equal the committed one of prereg25."""
    keys = rows.keys
    bit = {k: L25.holdout_bit(k) for k in set(keys.tolist())}
    rows_a = np.array([i for i, k in enumerate(keys) if bit[k] == 0], dtype=np.int64)
    rows_h = np.array([i for i, k in enumerate(keys) if bit[k] == 1], dtype=np.int64)
    pool_a = K.PlayerPool(keys, rows.y, rows.col, rows_a)
    pool_h = K.PlayerPool(keys, rows.y, rows.col, rows_h)
    if set(pool_a.players) & set(pool_h.players):
        raise SystemExit("A and H share players")
    hold, hold_stats = K.holdout_set(pool_h, hold_per, K.HOLD_SEED)
    is_v = np.arange(len(keys)) >= rows.n_probe
    rec = {"A_players": len(pool_a.players), "A_games": int(len(rows_a)), "H_players": len(pool_h.players),
           "H_games": int(len(rows_h)), "A_and_H": 0,
           "A_V_players": len({keys[i] for i in rows_a[is_v[rows_a]]}), "A_V_games": int(is_v[rows_a].sum()),
           "H_V_players": len({keys[i] for i in rows_h[is_v[rows_h]]}), "H_V_games": int(is_v[rows_h].sum()),
           "probe_rows": rows.n_probe, "v_rows": rows.n_v, "probe_V_overlap": rows.overlap,
           "A_players_per_level": np.bincount(pool_a.level, minlength=K.L).tolist(),
           "H_players_per_level": np.bincount(pool_h.level, minlength=K.L).tolist(),
           "A_eligible_n_level_colour": pool_a.counts(), "H_eligible_n_level_colour": pool_h.counts(),
           "holdout_questions": {str(n): int(len(v[1])) for n, v in hold.items()},
           "holdout_sampling": hold_stats}
    return pool_a, hold, json.loads(json.dumps(rec))


def per_n_nll(params, X32, hold):
    """Held-out NLL of one checkpoint per games-per-question n (record-only, PLAN.md 2), by the imported numpy
    float64 forward (p19_lib.set_logp_np, the one port_check compares with torch) -> ({n: ...}, overall NLL)."""
    par = K.p64(params)
    out, tot, cnt = {}, 0.0, 0
    for n, (idx, y) in hold.items():
        if len(y) == 0:
            continue
        lp = K.set_logp_np(par, X32[idx.ravel()].astype(np.float64).reshape(len(y), n, K.D_IN))
        s = float(-lp[np.arange(len(y)), y].sum())
        out[str(n)] = {"questions": int(len(y)), "nll": s / len(y)}
        tot += s
        cnt += len(y)
    return out, tot / cnt


def train_run(cfg, mpath, X32, pool_a, hold, smoke, head):
    """One run: p19_train.train_seed / port_check with p19_lib.WD = cfg wd and p19_lib.THREADS = 4 inside
    `patched`; the best checkpoint is saved with p25_train.save_model. Returns the run record (not written here)."""
    import torch
    started, t0 = r.now_iso(), time.time()
    with patched(K, cfg["wd"], cfg["threads"]) as applied:
        best, curve, stats, n_count, secs = T.train_seed(
            cfg["seed"], X32, pool_a, hold, cfg["steps"], cfg["eval_every"],
            ("smoke " if smoke else "pilot ") + cfg["config"])
        pc = T.port_check(best["params"], X32, hold)
        threads_after = int(torch.get_num_threads())
    if (K.WD, K.THREADS) != (REG_WD, REG_THREADS):
        raise SystemExit("p19_lib.WD / p19_lib.THREADS were not restored")
    name = run_name(cfg["config"], cfg["seed"])
    if applied != {"wd": cfg["wd"], "threads": cfg["threads"]} or threads_after != cfg["threads"]:
        raise SystemExit(f"{name}: wd / threads not applied ({applied}, torch threads {threads_after})")
    if [c["step"] for c in curve] != [cfg["eval_every"] * (i + 1) for i in range(CHECKPOINTS)]:
        raise SystemExit(f"{name}: not the {CHECKPOINTS} registered checkpoints")
    if not pc["ok"]:
        raise SystemExit(f"{name}: torch / numpy port check failed {pc}")
    per_n, nll_np = per_n_nll(best["params"], X32, hold)
    msha = P25T.save_model(mpath, best["params"], cfg["seed"], best, smoke)
    last = curve[-1]
    rec = {"format": RUN_FORMAT, "smoke": bool(smoke), "config": cfg["config"], "seed": cfg["seed"],
           "steps": cfg["steps"], "wd": cfg["wd"], "threads": cfg["threads"], "eval_every": cfg["eval_every"],
           "checkpoints": len(curve), "applied": applied, "torch_threads_after_training": threads_after,
           "curve": curve, "best_step": best["step"], "best_step_fraction": best["step"] / cfg["steps"],
           "best_holdout_nll": best["nll"], "last_step": last["step"],
           "last_step_holdout_nll": last["holdout_nll"], "last_step_train_batch_loss": last["train_batch_loss"],
           "per_n_holdout_nll_best": per_n, "best_holdout_nll_numpy64": nll_np,
           "sampling": stats, "batches_per_n": n_count, "seconds": secs,
           "seconds_total": round(time.time() - t0, 1), "port_check": pc,
           "model_file": rel(mpath), "model_sha256": msha, "head": head, "started": started, "ended": r.now_iso()}
    log(f"run {name}: best step {best['step']} NLL {best['nll']:.5f}; last step NLL {last['holdout_nll']:.5f}; "
        f"port check {pc}; {rec['seconds_total']}s")
    return rec


def run_file_problems(rec, cfg, head, smoke):
    """Why a run file is not the finished run `cfg` of this pilot (empty list = usable)."""
    want = {"format": RUN_FORMAT, "smoke": bool(smoke), "config": cfg["config"], "seed": cfg["seed"],
            "steps": cfg["steps"], "wd": cfg["wd"], "threads": cfg["threads"], "eval_every": cfg["eval_every"],
            "checkpoints": CHECKPOINTS, "head": head, "applied": {"wd": cfg["wd"], "threads": cfg["threads"]},
            "torch_threads_after_training": cfg["threads"]}
    bad = [f"{k}: {rec.get(k)!r} is not {v!r}" for k, v in want.items() if rec.get(k) != v]
    curve = rec.get("curve") or []
    if [c.get("step") for c in curve] != [cfg["eval_every"] * (i + 1) for i in range(CHECKPOINTS)]:
        bad.append(f"curve: not the {CHECKPOINTS} registered checkpoints")
    else:
        nll = [c["holdout_nll"] for c in curve]
        i = nll.index(min(nll))                 # the first minimum, as train_seed (strict <)
        if rec.get("best_holdout_nll") != nll[i] or rec.get("best_step") != curve[i]["step"]:
            bad.append("best step / best NLL are not the first minimum of the curve")
        if rec.get("last_step_holdout_nll") != nll[-1] \
                or rec.get("last_step_train_batch_loss") != curve[-1]["train_batch_loss"]:
            bad.append("last-step values are not the last checkpoint of the curve")
    if not (rec.get("port_check") or {}).get("ok"):
        bad.append("port check not passed")
    return bad


# ---------------------------------------------------------------- deadline (launcher: 2026-11-11T08:00 local time)
def parse_deadline(text):
    """ISO date-time -> naive local datetime (a value with a UTC offset is converted to local time)."""
    if text is None:
        return None
    d = datetime.datetime.fromisoformat(text)
    if d.tzinfo is not None:
        d = d.astimezone().replace(tzinfo=None)
    return d


def seconds_per_step(finished, threads=THREADS):
    """Measured seconds per step over the finished runs of this pilot at the same thread count (None: none yet)."""
    use = [x for x in finished if x.get("threads") == threads]
    steps = sum(int(x["steps"]) for x in use)
    return (sum(float(x["seconds_total"]) for x in use) / steps) if steps else None


def deadline_decision(now, deadline, steps, finished, threads=THREADS):
    """May a run of `steps` steps start now? Not when the deadline has passed, and not when now + the estimated
    duration is past the deadline. Without any finished run there is no estimate: only the deadline itself counts."""
    out = {"now": now.isoformat(timespec="seconds"), "deadline": deadline.isoformat() if deadline else None,
           "steps": steps, "seconds_per_step": None, "estimate_seconds": None, "projected_end": None}
    if deadline is None:
        return {**out, "start": True, "reason": "no deadline"}
    rate = seconds_per_step(finished, threads)
    out["seconds_per_step"] = rate
    if now >= deadline:
        return {**out, "start": False, "reason": "the deadline has passed"}
    if rate is None:
        return {**out, "start": True,
                "reason": f"no finished run at {threads} threads yet, so no estimate; the deadline has not passed"}
    est = rate * steps
    end = now + datetime.timedelta(seconds=est)
    out.update(estimate_seconds=round(est, 1), projected_end=end.isoformat(timespec="seconds"))
    if end > deadline:
        return {**out, "start": False, "reason": "the estimated end is past the deadline"}
    return {**out, "start": True, "reason": "the estimated end is before the deadline"}


# ---------------------------------------------------------------- seam R sanity and the judgement (PLAN.md 1 / 2)
class JudgeError(ValueError):
    pass


def seam_r(e0, ref, tol=SEAM_R_TOL):
    """E0 (4 threads) against the committed 8-thread best held-out NLL of prereg25, per seed: |diff| <= tol."""
    seeds = {}
    for s in sorted(e0):
        d = e0[s] - ref[s]
        seeds[str(s)] = {"e0_4_threads": e0[s], "prereg25_8_threads": ref[s], "diff": d, "abs_diff": abs(d),
                         "ok": bool(abs(d) <= tol)}
    return {"tol": tol, "ok": bool(all(v["ok"] for v in seeds.values())), "seeds": seeds}


def _mean(vals):
    return sum(vals) / len(vals)


def record_only(rec):
    """The record-only items of PLAN.md 2 for one finished run."""
    last_nll, last_loss = rec.get("last_step_holdout_nll"), rec.get("last_step_train_batch_loss")
    return {"best_step": rec.get("best_step"), "best_step_fraction": rec.get("best_step_fraction"),
            "last_step_holdout_nll": last_nll, "last_step_train_batch_loss": last_loss,
            "overfit_gap_last_step": (last_nll - last_loss) if last_nll is not None and last_loss is not None
            else None,
            "last_minus_best_holdout_nll": (last_nll - rec["best_holdout_nll"]) if last_nll is not None else None,
            "per_n_holdout_nll_best": rec.get("per_n_holdout_nll_best"), "seconds": rec.get("seconds")}


def judge(runs, ref):
    """PLAN.md 2, a pure function. runs: {(config, seed): finished real run record}; ref: {seed: the 8-thread best
    held-out NLL of prereg25}. Needs both control seeds (JudgeError otherwise). A config counts only when both of
    its seeds finished, else it is "not run". Delta = mean of the best held-out NLL of the two seeds - the same mean
    of the control. GO iff some config has Delta <= -0.020 and each seed is at least 0.010 below the same-seed
    control (both inclusive, no tolerance); several: the smallest Delta is the candidate (a tie: the earlier in the
    registered order). Seam R sanity failed -> verdict STOP (pipeline error), no candidate."""
    best = {k: float(v["best_holdout_nll"]) for k, v in runs.items()}
    missing = [s for s in SEEDS if (CONTROL, s) not in best]
    if missing:
        raise JudgeError(f"the control {CONTROL} needs both seeds finished; missing seeds {missing}")
    if any(s not in ref for s in SEEDS):
        raise JudgeError("the seam-R reference needs both seeds")
    e0 = {s: best[(CONTROL, s)] for s in SEEDS}
    e0_mean = _mean([e0[s] for s in SEEDS])
    sr = seam_r(e0, {s: float(ref[s]) for s in SEEDS})
    configs, not_run, passing = {}, [], []
    for order, (config, steps, wd) in enumerate(CONFIGS):
        if config == CONTROL:
            continue
        fin = [s for s in SEEDS if (config, s) in best]
        c = {"steps": steps, "wd": wd, "seeds_finished": fin}
        if len(fin) < len(SEEDS):
            c["status"] = "not run"
            not_run.append(config)
        else:
            vals = {s: best[(config, s)] for s in SEEDS}
            mean = _mean([vals[s] for s in SEEDS])
            delta = mean - e0_mean
            diff = {s: vals[s] - e0[s] for s in SEEDS}
            delta_ok = bool(delta <= GO_DELTA)
            seeds_ok = bool(all(diff[s] <= -GO_SEED for s in SEEDS))
            c.update(status="finished", best_holdout_nll={str(s): vals[s] for s in SEEDS}, mean=mean, delta=delta,
                     seed_minus_control={str(s): diff[s] for s in SEEDS}, delta_ok=delta_ok, seeds_ok=seeds_ok,
                     passes_thresholds=bool(delta_ok and seeds_ok),
                     record_only={str(s): record_only(runs[(config, s)]) for s in SEEDS})
            if c["passes_thresholds"]:
                passing.append((delta, order, config))
        configs[config] = c
    if not sr["ok"]:
        verdict, candidate = "STOP", None
    elif passing:
        verdict, candidate = "GO", min(passing)[2]
    else:
        verdict, candidate = "NO-GO", None
    return {"verdict": verdict, "candidate": candidate,
            "thresholds": {"delta": GO_DELTA, "per_seed": GO_SEED, "seam_r_tol": SEAM_R_TOL},
            "seam_r": sr,
            "control": {"config": CONTROL, "steps": CONFIGS[0][1], "wd": CONFIGS[0][2],
                        "best_holdout_nll": {str(s): e0[s] for s in SEEDS}, "mean": e0_mean,
                        "record_only": {str(s): record_only(runs[(CONTROL, s)]) for s in SEEDS}},
            "configs": configs, "passing": [x[2] for x in sorted(passing)], "not_run": not_run}


def reference_nll():
    """The committed 8-thread best held-out NLL of seeds 0 and 1 in prereg25 (train_record.json)."""
    tr = L29.read_json(L29.P25_TRAIN)
    if tr["settings"].get("threads") != REG_THREADS or tr.get("smoke"):
        raise SystemExit("results/prereg25/explore/train_record.json is not the 8-thread S25 training record")
    return {s: float(tr["seeds"][str(s)]["best_holdout_nll"]) for s in SEEDS}


# ---------------------------------------------------------------- the loop over runs
def run_loop(plan, base, stop_path, state_path, head, smoke, resume, deadline, train_fn, ref=None, session=None,
             now_fn=datetime.datetime.now):
    """Runs of `plan` in order. A finished run (its run file exists) is kept on --resume and never re-run. Before
    each new run: STOP file -> stop; deadline_decision says no -> stop, and nothing later is started. Once both
    control seeds are there: seam R sanity (ref given) -> stop when it fails. The state file says what happened."""
    names = [run_name(c["config"], c["seed"]) for c in plan]
    earlier = []
    if resume and os.path.exists(state_path):
        prev = L29.read_json(state_path)
        earlier = list(prev.pop("earlier_sessions", [])) + [prev]
    state = {"format": STATE_FORMAT, "smoke": bool(smoke), "status": "running", "pid": os.getpid(), "head": head,
             "resume": bool(resume), "deadline": deadline.isoformat() if deadline else None, "threads": THREADS,
             "session_started": r.now_iso(), "updated": r.now_iso(), "plan": names, "kept": [], "finished": [],
             "current": None, "not_started": [], "decisions": [], "seam_r": None, "exit_code": None,
             "error": None, "session": session, "earlier_sessions": earlier}

    def save(**kw):
        state.update(kw)
        state["updated"] = r.now_iso()
        T.write_json(state_path, state)

    def stop(status, code, i, reason, later=None):
        rest = [{"run": names[i], "reason": reason}]
        rest += [{"run": n, "reason": later or reason} for n in names[i + 1:]]
        save(status=status, exit_code=code, current=None, not_started=rest)
        log(f"{status}: not started {[x['run'] for x in rest]} ({reason})")
        return code

    save()
    done = {}
    for i, cfg in enumerate(plan):
        name = names[i]
        rpath = run_file(base, cfg["config"], cfg["seed"])
        mpath = model_file(base, cfg["config"], cfg["seed"])
        if os.path.exists(rpath):
            if not resume:
                save(status="error", error=f"{name}: run file exists")
                raise SystemExit(f"{name}: {rpath} exists; a finished run is never re-run (--resume keeps it)")
            rec = L29.read_json(rpath)
            bad = run_file_problems(rec, cfg, head, smoke)
            if not bad and not os.path.exists(mpath):
                bad.append("model file missing")
            if not bad and rec.get("model_sha256") != T.sha_file(mpath):
                bad.append("model file SHA differs from the run file")
            if bad:
                save(status="error", error=f"{name}: kept run file not usable: {bad}")
                raise SystemExit(f"{name}: the kept run file is not usable (move it away by hand first): {bad}")
            state["kept"].append(name)
            log(f"run {name}: kept (resume)")
        else:
            if os.path.exists(stop_path):
                return stop("stopped: STOP file", EXIT_STOP, i, f"STOP file {rel(stop_path)} before this run")
            dec = deadline_decision(now_fn(), deadline, cfg["steps"], list(done.values()))
            state["decisions"].append({"run": name, **dec})
            if not dec["start"]:
                return stop("stopped: deadline", EXIT_DEADLINE, i, dec["reason"],
                            "an earlier run was not started for time: nothing later is started")
            save(current=name)
            log(f"run {name}: start ({cfg['steps']} steps, wd {cfg['wd']}, {cfg['threads']} threads; "
                f"{dec['reason']}; estimate {dec['estimate_seconds']} s)")
            try:
                rec = train_fn(cfg, mpath)
            except SystemExit as e:
                if os.path.exists(stop_path) and str(e).startswith("STOP file"):
                    return stop("stopped: STOP file", EXIT_STOP, i,
                                f"{e} (the unfinished run is discarded)", "STOP file during an earlier run")
                save(status="error", error=str(e))
                raise
            except BaseException as e:
                save(status="error", error=repr(e))
                raise
            T.write_json(rpath, rec)            # last: this file is what marks the run as finished
            state["finished"].append(name)
            save(current=None)
        done[(cfg["config"], cfg["seed"])] = rec
        if ref is not None and cfg["config"] == CONTROL and all((CONTROL, s) in done for s in SEEDS):
            sr = seam_r({s: float(done[(CONTROL, s)]["best_holdout_nll"]) for s in SEEDS}, ref)
            save(seam_r=sr)
            log(f"seam R sanity {json.dumps(sr)}")
            if not sr["ok"]:
                if i + 1 == len(plan):
                    save(status="stopped: seam R", exit_code=EXIT_SEAM_R)
                    return EXIT_SEAM_R
                return stop("stopped: seam R", EXIT_SEAM_R, i + 1,
                            "seam R sanity failed: the control differs from prereg25 by more than the tolerance "
                            "(pipeline error)")
    save(status="finished", exit_code=EXIT_DONE)
    return EXIT_DONE


def load_data(smoke):
    """Rows, seams and roles exactly as the S25 training (prereg25 / prereg29 / prereg31 do the same checks)."""
    tr25 = L29.read_json(L29.P25_TRAIN)
    want = L29.registered_settings(K)
    if any(tr25["settings"].get(k) != v for k, v in want.items()) \
            or tr25["settings"].get("seeds") != list(L29.S25_SEEDS):
        raise SystemExit("the imported settings differ from prereg25 train_record.json")
    if K.D_IN != FW.X_DIM:
        raise SystemExit("input width: expected p19_lib.D_IN = 1540 (no HD input in this pilot)")
    hold_per = K.SMOKE["hold_per"] if smoke else K.HOLD_PER
    mp = FW.load_map(T.Q09_MAP)
    map_sha = r.sha_raw(T.Q09_MAP)
    if map_sha != r.read_json(T.Q09_FROZEN)["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    wantp = (r.read_json(T.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = T.sha_file(os.path.join(T.PROBE_PREP, "probe_X.npy"))
    seam_p = {"want": wantp.get("probe_X.npy"), "got": got, "ok": bool(wantp.get("probe_X.npy") == got)}
    rows = P25T.Rows()
    vsha = T.sha_file(os.path.join(L29.V_PREP, "v_X.npy"))
    seam_p["v_X_sha256"] = vsha
    seam_p["v_X_is_prep_record"] = bool(vsha == rows.record["files_sha256"]["v_X.npy"])
    seam_p["v_X_is_prereg25_train_record"] = bool(vsha == tr25["seam_p"]["v_X_sha256"])
    seam_p["v_meta_sha256"] = T.sha_file(V_META)
    seam_p["v_meta_is_prep_record"] = bool(seam_p["v_meta_sha256"] == rows.record["files_sha256"]["v_meta.npz"])
    seam_p["ok"] = bool(seam_p["ok"] and seam_p["v_X_is_prep_record"] and seam_p["v_X_is_prereg25_train_record"]
                        and seam_p["v_meta_is_prep_record"])
    if not seam_p["ok"]:
        raise SystemExit(f"seam P failed: {seam_p}")
    pool_a, hold, players_rec = roles(rows, hold_per)
    same = bool(players_rec == tr25["players"]) if not smoke else None
    if not smoke and not same:
        raise SystemExit("players / held-out set differ from prereg25 train_record.json")
    log(f"players {json.dumps({k: players_rec[k] for k in ('A_players', 'A_games', 'H_players', 'H_games')})}"
        f" (whole record equal to the prereg25 one: {same})")
    t = time.time()
    X32 = rows.standardised(mp["mu_x"], mp["sd_x"], smoke)
    log(f"standardised X {X32.shape} ({time.time() - t:.0f}s)")
    if X32.shape != (rows.n_probe + rows.n_v, FW.X_DIM):
        raise SystemExit(f"standardised X has shape {X32.shape}")
    info = {"settings_s25": want, "hold_per": hold_per, "q09_map_sha256": map_sha, "seam_p": seam_p,
            "v_prep_record": {k: rows.record[k] for k in ("games_sha256", "rows", "unusable", "files_sha256")},
            "players": players_rec, "players_same_as_prereg25": same,
            "environment_prereg25": tr25.get("environment")}
    return X32, pool_a, hold, info


# ---------------------------------------------------------------- run
def cmd_run(a):
    t_all = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    deadline = parse_deadline(a.deadline)
    base = SMOKE_DIR if smoke else DATA
    state_path = os.path.join(SMOKE_DIR, "pilot_state.json") if smoke else STATE
    point_stop()
    if T.STOP != STOP:
        raise SystemExit("p19_train.STOP does not point to data/prereg32/STOP")
    if not train_seed_reads_at_call_time():
        raise SystemExit("p19_train.train_seed no longer reads p19_lib.THREADS / p19_lib.WD at call time: stop")
    if (K.WD, K.THREADS) != (REG_WD, REG_THREADS) or CONFIGS[0] != (CONTROL, K.STEPS, K.WD) \
            or K.SMOKE["steps"] != SMOKE_STEPS:
        raise SystemExit("p19_lib is not at the registered S25 values wd 0.01 / 8 threads / 15,000 steps")
    plan = run_plan(smoke)
    cs = code_state()
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg32 files or the imported tools are not committed and clean: {cs['dirty']}")
        if os.path.exists(PILOT):
            raise SystemExit(f"{PILOT} exists: the pilot was judged, nothing more is run")
        prior = [p for p in [STARTED, STATE] + [run_file(base, c["config"], c["seed"]) for c in plan]
                 if os.path.exists(p)]
        if a.resume:
            if not os.path.exists(STARTED):
                raise SystemExit(f"--resume: {STARTED} does not exist, there is nothing to resume")
            stc = L29.read_json(STARTED).get("code") or {}
            if stc.get("sha256_lf") != cs["sha256_lf"] or stc.get("head") != cs["head"]:
                raise SystemExit("--resume needs the same commit and code as pilot_started.json")
            if os.path.exists(STATE) and L29.read_json(STATE).get("status") == "stopped: deadline":
                raise SystemExit("--resume refused: the pilot stopped for the time limit (PLAN.md 2: judge the "
                                 "finished configs, no continuation)")
        elif prior:
            raise SystemExit(f"{prior} exist: run is started once (--resume only after an interruption)")
    if os.path.exists(STOP):
        raise SystemExit(f"STOP file {STOP} exists: delete it by hand to start")
    if smoke and not a.resume:          # a smoke starts from nothing: its own earlier run files are not kept
        for c in plan:
            p = run_file(base, c["config"], c["seed"])
            if os.path.exists(p):
                os.remove(p)
    prio = low_priority() if a.low_priority else {"requested": False}
    import torch
    torch.set_num_threads(THREADS)
    X32, pool_a, hold, info = load_data(smoke)
    ref = None if smoke else reference_nll()
    session = {**info, "priority": prio, "deadline_argument": a.deadline, "code_dirty": bool(cs["dirty"]),
               "settings_pilot": {"configs": [list(c) for c in CONFIGS], "seeds": list(SEEDS),
                                  "checkpoints": CHECKPOINTS, "threads": THREADS, "plan": plan,
                                  "smoke_note": "features replaced by default_rng(12345) normal noise, 300 steps, "
                                                "E0 and E1 only: never a result" if smoke else None},
               "seam_r_reference_prereg25_8_threads": {str(s): v for s, v in ref.items()} if ref else None,
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                               "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                               "cpu_count": os.cpu_count()}}
    if not smoke and not a.resume:      # written only when every check passed, just before the first training
        T.write_json(STARTED, {"format": STARTED_FORMAT, "created": r.now_iso(), "code": cs,
                               "deadline": a.deadline, "session": session})

    def train_fn(cfg, mpath):
        return train_run(cfg, mpath, X32, pool_a, hold, smoke, cs["head"])

    rc = run_loop(plan, base, STOP, state_path, cs["head"], smoke, bool(a.resume), deadline, train_fn, ref=ref,
                  session=session)
    for c in plan:
        p = run_file(base, c["config"], c["seed"])
        if os.path.exists(p):
            x = L29.read_json(p)
            log(f"{'smoke ' if smoke else ''}{run_name(c['config'], c['seed'])}: {x['steps']} steps, wd {x['wd']}, "
                f"{x['threads']} threads, train_seed {x['seconds']} s, total {x['seconds_total']} s")
    log(f"-> {state_path} (exit code {rc}; {time.time() - t_all:.0f}s)")
    return rc


# ---------------------------------------------------------------- judge (PLAN.md 2; once)
def load_runs(base, head):
    """Finished real run files of the registered plan -> ({(config, seed): record}, {file: sha256})."""
    runs, files = {}, {}
    for cfg in run_plan(False):
        p = run_file(base, cfg["config"], cfg["seed"])
        if not os.path.exists(p):
            continue
        rec = L29.read_json(p)
        bad = run_file_problems(rec, cfg, head, False)
        if bad:
            raise SystemExit(f"{rel(p)} is not a finished run of this pilot: {bad}")
        runs[(cfg["config"], cfg["seed"])] = rec
        files[rel(p)] = T.sha_file(p)
    return runs, files


def cmd_judge(a):
    if os.path.exists(PILOT):
        raise SystemExit(f"{PILOT} exists: the pilot is judged once")
    cs = code_state()
    if cs["dirty"]:
        raise SystemExit(f"the prereg32 files or the imported tools are not committed and clean: {cs['dirty']}")
    if not os.path.exists(STARTED) or not os.path.exists(STATE):
        raise SystemExit("data/prereg32/pilot_started.json / pilot_state.json missing: no real run to judge")
    started, state = L29.read_json(STARTED), L29.read_json(STATE)
    if state.get("smoke") or state.get("format") != STATE_FORMAT:
        raise SystemExit("pilot_state.json is not the state of a real run")
    if state.get("status") == "running" and not a.killed:
        raise SystemExit("pilot_state.json says the run is still running: wait for it or stop it with the STOP "
                         "file (--killed only when its process is gone)")
    stc = started.get("code") or {}
    runs, files = load_runs(DATA, stc.get("head"))
    try:
        res = judge(runs, reference_nll())
    except JudgeError as e:
        raise SystemExit(f"judge: {e}")
    changed = sorted(k for k in set(stc.get("sha256_lf") or {}) | set(cs["sha256_lf"])
                     if (stc.get("sha256_lf") or {}).get(k) != cs["sha256_lf"].get(k))
    rec = {"format": PILOT_FORMAT, "created": r.now_iso(), "code": cs,
           "what": "PLAN.md 2: Delta(config) = mean over seeds 0 and 1 of the best held-out NLL - the same mean of "
                   "the control E0 (wd 0.01, 15,000 steps, 4 threads, 30 checkpoints); GO iff some config has "
                   "Delta <= -0.020 and each seed at least 0.010 below the same-seed control (the smallest Delta "
                   "is the candidate); a config with a seed missing is 'not run'; seam R sanity |E0 - prereg25 "
                   "8-thread value| <= 0.02 per seed, else STOP; pure function of the finished run files; judged "
                   "once; a GO only means a phase-1 PLAN is worth writing",
           **res,
           "run_files_sha256": files, "runs_finished": [run_name(*k) for k in sorted(runs)],
           "run_head": stc.get("head"), "run_started": started.get("created"),
           "code_same_as_run": not changed, "code_files_changed_since_run": changed,
           "killed_flag": bool(a.killed), "pilot_state": state,
           "seam_r_reference": {"file": rel(L29.P25_TRAIN), "sha256_lf": T.sha_lf(L29.P25_TRAIN)}}
    T.write_json(PILOT, rec)
    log(f"pilot: {res['verdict']} (candidate {res['candidate']}; not run {res['not_run']}; "
        f"seam R ok {res['seam_r']['ok']})")
    log(f"-> {PILOT}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--deadline", default=None, help="local time, ISO (the launcher passes 2026-11-11T08:00)")
    p.add_argument("--low-priority", action="store_true")
    j = sub.add_parser("judge")
    j.add_argument("--killed", action="store_true",
                   help="the process of the run is gone although pilot_state.json still says running")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
