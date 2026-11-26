"""prereg33 exploration (competition 2): train S33 (PLAN.md 1) = S25's code, settings, rows and salt with the H players
(S25's 748 held-out players) also trained, each seed stopped at S25's chosen step of that seed (no step choice here).
CPU only (.venv torch). Reads no inner data, never the answers folder; computes no candidate number.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg33/explore
    $S/p33_train.py train            -> data/prereg33/models/s33_seed0_seamR.npz (seam R33: A only, stop at
                                        best_step(0), must equal S25 seed 0 byte for byte), s33_seed{0,1,2}.npz
                                        + $S/train_record.json (run once; the code must be committed and clean)
    $S/p33_train.py train --smoke    -> data/prereg33/smoke/ (features replaced by default_rng(12345) normal noise,
                                        300-step schedule, torch threads 4; seam R33 in smoke = this program's fixed-step
                                        loop vs prereg19's train_seed run here on the same noise: pipeline only)
    --resume keeps a finished seed (and the seam-R33 model) after an interruption (same commit and code).
    A file data/prereg33/STOP stops the run every 500 steps (100 in smoke) and before each seed.

Rows, standardisation (the frozen Q09 map's mu_x / sd_x, exactly what S25 used), player roles (prereg25 salt),
held-out set (port check only), model, optimiser, OneCycle schedule, sampler and model file writer are imported
unchanged from results/prereg25/explore/p25_train.py / results/prereg19/explore/p19_train.py / p19_lib.py. The only new
code is train_fixed: p19_train.train_seed's loop line for line, stopping at a given step of the 15,000-step schedule
and without the held-out evaluations (which consume no random number and leave the model in train mode, so they do not
change the trajectory; test_p33.py checks this on synthetic data).
"""

import argparse
import hashlib
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
import p29_train as P29T  # noqa: E402  (seam_r: byte comparison with the frozen S25 seed 0)
import p29_lib as L29  # noqa: E402  (S25 paths / format / sha helpers, loaded_repo_modules)

T, K, FW, L25 = P25T.T, P25T.K, P25T.FW, P25T.L
r = T.r
DATA = os.path.join(REPO, "data", "prereg33")
MODELS = os.path.join(DATA, "models")
SMOKE_DIR = os.path.join(DATA, "smoke")
STOP = os.path.join(DATA, "STOP")
STARTED = os.path.join(DATA, "train_started.json")
RECORD = os.path.join(HERE, "train_record.json")
P25_TRAIN = L29.P25_TRAIN
OWN_FILES = ("PLAN.md", "p33_train.py", "run_train.sh")
IMPORTED = tuple(P29T.IMPORTED) + ("results/prereg29/explore/p29_train.py", "results/prereg29/explore/p29_lib.py",
                                   "results/prereg31/explore/ram_watch.py")
FORMAT = "c10-prereg33 explore train v1"
MODEL_FORMAT = L29.S25_FORMAT        # S33 files keep S25's format string (seam R33 compares bytes; eval loader)
SEEDS = (0, 1, 2)
SEAM_R_SEED = 0
SMOKE_THREADS = 4
FORBIDDEN_SUBSTR = "data/lo" + "cal"
T.STOP = STOP          # p19_train's module global (p25_train / p29_train had pointed it elsewhere); after all imports


def log(*x):
    print(*x, flush=True)


def rel(path):
    try:
        return os.path.relpath(path, REPO).replace("\\", "/")
    except ValueError:                 # another drive (unit tests' tmp folder): keep the absolute path
        return os.path.abspath(path).replace("\\", "/")


def code_state():
    files = [rel(os.path.join(HERE, f)) for f in OWN_FILES] + list(IMPORTED)
    files = sorted(set(files) | L29.loaded_repo_modules())
    if any(FORBIDDEN_SUBSTR in f for f in files):
        raise SystemExit("refused path in code_state")
    missing = [x for x in files if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = T.git("status", "--porcelain", "--", *files)
    return {"head": T.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: T.sha_lf(os.path.join(REPO, *x.split("/"))) for x in files}}


# ---------------------------------------------------------------- PLAN.md 1: steps, players
def stop_steps(tr25, total=None, every=None):
    """seed -> S25's chosen step (prereg25 train_record.json seeds.<s>.best_step), checked against the schedule."""
    total = K.STEPS if total is None else total
    every = K.EVAL_EVERY if every is None else every
    out = {}
    for s in SEEDS:
        b = int(tr25["seeds"][str(s)]["best_step"])
        if not (1 <= b <= total and b % every == 0):
            raise SystemExit(f"seed {s}: best_step {b} is not an evaluation step of a {total}-step schedule")
        out[s] = b
    return out


def smoke_stop_steps(ref_best0):
    """Smoke: no S25 smoke step to read for seeds 1, 2 with this schedule; seam seed 0 uses the reference run's step,
    seeds 1 / 2 the registered real steps scaled to the 300-step schedule (pipeline only)."""
    tr25 = L29.read_json(P25_TRAIN)
    real = stop_steps(tr25)
    sm = {s: max(K.SMOKE["eval_every"], int(round(real[s] / K.STEPS * K.SMOKE["steps"] / K.SMOKE["eval_every"]))
                 * K.SMOKE["eval_every"]) for s in SEEDS}
    return {"seam": int(ref_best0), "seeds": sm}


def roles(keys):
    """Rows of A (training in S25), H (S25's held-out players) and A u H (S33's training rows), by prereg25's salt."""
    keys = np.asarray(keys).astype(str)
    bit = {k: L25.holdout_bit(k) for k in set(keys.tolist())}
    b = np.array([bit[k] for k in keys], dtype=np.int64)
    rows_a = np.flatnonzero(b == 0).astype(np.int64)
    rows_h = np.flatnonzero(b == 1).astype(np.int64)
    rows_t = np.sort(np.concatenate([rows_a, rows_h]))
    if len(rows_t) != len(keys) or len(np.unique(rows_t)) != len(keys):
        raise SystemExit("A u H must be every row exactly once")
    return rows_a, rows_h, rows_t


def same_players(rec, p25p):
    return bool(all(rec[k] == p25p[k] for k in ("A_players", "A_games", "H_players", "H_games",
                                                    "holdout_questions")))


def model_name(seed, seam=False):
    return f"s33_seed{seed}_seamR.npz" if seam else f"s33_seed{seed}.npz"


# ---------------------------------------------------------------- the fixed-step loop (= p19_train.train_seed)
def train_fixed(seed, X32, pool, total_steps, stop_at, log_every, log_prefix):
    """p19_train.train_seed line for line (same model init, AdamW, OneCycleLR over total_steps, sampler stream
    default_rng([19, seed]), batch order), stopped after step stop_at; no held-out evaluation, no step choice.
    -> params at step stop_at (export_params), sampling stats, batches per n, seconds, last batch loss."""
    import torch
    if not (1 <= stop_at <= total_steps):
        raise SystemExit(f"stop step {stop_at} outside 1..{total_steps}")
    torch.set_num_threads(K.THREADS)
    model = T.torch_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=K.LR, weight_decay=K.WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=total_steps, pct_start=K.PCT_START)
    rng = np.random.default_rng([K.SAMPLER_BASE, seed])
    stats = {"colour_switched": 0, "level_redrawn": 0, "questions": 0}
    n_count = {str(n): 0 for n in K.KS}
    t0 = time.time()
    loss_v = None
    model.train()
    for step in range(1, stop_at + 1):
        idx, y, n = K.sample_batch(rng, pool, K.BATCH, stats)
        n_count[str(n)] += 1
        xb = torch.from_numpy(X32[idx.ravel()]).view(K.BATCH, n, K.D_IN)
        lp = model(xb)
        loss = -lp[torch.arange(K.BATCH), torch.from_numpy(y)].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if step % log_every == 0 or step == stop_at:
            loss_v = float(loss.item())
            log(f"{log_prefix} seed {seed} step {step}/{stop_at} (schedule {total_steps}): batch loss {loss_v:.5f} "
                f"lr {sch.get_last_lr()[0]:.3e} {time.time() - t0:.0f}s")
            if step < stop_at and os.path.exists(STOP):
                raise SystemExit(f"STOP file {STOP}: stopped at step {step}")
    state = {"sampler_rng_state_sha256": hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True,
                                                                          default=str).encode()).hexdigest(),
             "scheduler_last_epoch": int(sch.last_epoch), "last_lr": float(sch.get_last_lr()[0])}
    return T.export_params(model), stats, n_count, round(time.time() - t0, 1), loss_v, state


def train_one(seed, X32, pool, hold, total_steps, stop_at, log_every, smoke, mpath, spath, resume, head, role):
    """One fixed-step model: train_fixed -> port check (prereg19's, on the S25 held-out questions) -> S25's writer."""
    if resume and os.path.exists(mpath) and os.path.exists(spath):
        sr = r.read_json(spath)
        if sr.get("head") != head:
            raise SystemExit(f"seed {seed} ({role}): kept record of another commit ({sr.get('head')}); move it away")
        if sr.get("model_sha256") != T.sha_file(mpath) or sr.get("stop_step") != stop_at:
            raise SystemExit(f"{mpath}: SHA or stop step differs from its seed record")
        if not (sr.get("port_check") or {}).get("ok"):
            raise SystemExit(f"seed {seed} ({role}): kept record without a passed port check")
        log(f"seed {seed} ({role}): kept (resume)")
        return sr
    params, stats, n_count, secs, loss_v, state = train_fixed(seed, X32, pool, total_steps, stop_at, log_every,
                                                              f"{'smoke' if smoke else 'train'} {role}")
    pc = T.port_check(params, X32, hold)
    nll, _ = T.holdout_nll(_loaded(params), X32, hold)
    msha = P25T.save_model(mpath, params, seed, {"step": stop_at}, smoke)
    sr = {"seed": seed, "role": role, "stop_step": stop_at, "schedule_steps": total_steps,
          "players": "A" if role == "seamR33" else "A+H", "sampling": stats, "batches_per_n": n_count,
          "seconds": secs, "last_batch_loss": loss_v, "state_at_stop": state, "port_check": pc,
          "s25_holdout_nll_record_only": nll,
          "s25_holdout_note": ("the S25 held-out questions; for A+H models these players were trained on, so this is "
                               "not a held-out number and chooses nothing"),
          "model_file": rel(mpath), "model_sha256": msha, "head": head}
    T.write_json(spath, sr)
    log(f"seed {seed} ({role}): stop step {stop_at}; port check {pc}")
    if not pc["ok"]:
        raise SystemExit(f"seed {seed} ({role}): torch / numpy port check failed {pc}")
    return sr


def _loaded(params):
    import torch
    m = T.torch_model(0)
    m.load_state_dict({k: torch.from_numpy(np.asarray(params[v])) for k, v in T.STATE_KEYS.items()})
    return m


# ---------------------------------------------------------------- seam R33
def seam_r_smoke(X32, pool_a, hold, out_dir, steps, eval_every):
    """Smoke analogue: prereg19's train_seed (S25's code, evaluations every eval_every) on the same noise, A only;
    this program's train_fixed stopped at that run's chosen step must give the same bytes through S25's writer."""
    best, _curve, _st, _n, _s = T.train_seed(SEAM_R_SEED, X32, pool_a, hold, steps, eval_every, "smoke seam-ref")
    ref = os.path.join(out_dir, "models", "seamR_reference_train_seed.npz")
    P25T.save_model(ref, best["params"], SEAM_R_SEED, best, True)
    return ref, int(best["step"])


def seam_r(path, smoke, ref=None):
    if not smoke:
        return P29T.seam_r(path, False)
    got, want = T.sha_file(path), T.sha_file(ref)
    return {"retrain_file": rel(path), "retrain_sha256": got, "reference_file": rel(ref), "reference_sha256": want,
            "applicable": True, "smoke": True, "bytes_equal_reference": L29.models_equal_bytes(path, ref),
            "ok": bool(got == want and L29.models_equal_bytes(path, ref))}


# ---------------------------------------------------------------- train
def cmd_train(a):
    t_all = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_dir = SMOKE_DIR if smoke else DATA
    model_dir = os.path.join(out_dir, "models")
    rec_path = os.path.join(SMOKE_DIR, "train_record.json") if smoke else RECORD
    cs = code_state()
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg33 files or the imported tools are not committed and clean: {cs['dirty']}")
        if os.path.exists(rec_path) and a.resume:
            raise SystemExit(f"{rec_path} exists: the training finished; --resume is refused")
        prior = [p for p in (rec_path, STARTED) if os.path.exists(p)]
        if prior and not a.resume and not (a.rerun and a.reason):
            raise SystemExit(f"{prior} exist: run once (--resume after an interruption; --rerun --reason only after "
                             f"a commit that names a concrete program error)")
        if a.rerun and prior:
            heads = {(r.read_json(p).get("code") or {}).get("head") for p in prior}
            if cs["head"] in heads:
                raise SystemExit("--rerun needs a newer commit than the earlier run's")
        if a.resume and os.path.exists(STARTED):
            stc = r.read_json(STARTED).get("code") or {}
            if stc.get("sha256_lf") != cs["sha256_lf"] or stc.get("head") != cs["head"]:
                raise SystemExit("--resume needs the same commit and code as train_started.json")
        else:
            T.write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                   "rerun_reason": a.reason if a.rerun else None})
    if os.path.exists(STOP):
        raise SystemExit(f"STOP file {STOP} exists: delete it by hand to start")
    if T.STOP != STOP:
        raise SystemExit("p19_train.STOP does not point to data/prereg33/STOP")
    import torch
    if smoke:
        K.THREADS = SMOKE_THREADS          # read by train_seed / train_fixed at call time (smoke only)
        torch.set_num_threads(SMOKE_THREADS)
    tr25 = L29.read_json(P25_TRAIN)
    want = L29.registered_settings(K)
    if smoke:
        want = {k: v for k, v in want.items() if k != "threads"}
    if any(tr25["settings"].get(k) != v for k, v in want.items()) or tr25["settings"].get("seeds") != list(SEEDS):
        raise SystemExit("the imported settings differ from prereg25 train_record.json")
    steps = K.SMOKE["steps"] if smoke else K.STEPS
    eval_every = K.SMOKE["eval_every"] if smoke else K.EVAL_EVERY
    hold_per = K.SMOKE["hold_per"] if smoke else K.HOLD_PER
    stops_real = stop_steps(tr25)
    # ---- inputs (= p29_train: seam P)
    mp = FW.load_map(T.Q09_MAP)
    map_sha = r.sha_raw(T.Q09_MAP)
    if map_sha != r.read_json(T.Q09_FROZEN)["map"]["sha256"] or map_sha != tr25["q09_map_sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json / prereg25's record")
    wantp = (r.read_json(T.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = T.sha_file(os.path.join(T.PROBE_PREP, "probe_X.npy"))
    seam_p = {"want": wantp.get("probe_X.npy"), "got": got, "ok": bool(wantp.get("probe_X.npy") == got)}
    rows = P25T.Rows()
    vsha = T.sha_file(os.path.join(L29.V_PREP, "v_X.npy"))
    seam_p["v_X_sha256"] = vsha
    seam_p["v_X_is_prep_record"] = bool(vsha == rows.record["files_sha256"]["v_X.npy"])
    seam_p["v_X_is_prereg25_train_record"] = bool(vsha == tr25["seam_p"]["v_X_sha256"])
    seam_p["v_meta_sha256"] = T.sha_file(os.path.join(L29.V_PREP, "v_meta.npz"))
    seam_p["v_meta_is_prep_record"] = bool(seam_p["v_meta_sha256"] == rows.record["files_sha256"]["v_meta.npz"])
    seam_p["ok"] = bool(seam_p["ok"] and seam_p["v_X_is_prep_record"] and seam_p["v_X_is_prereg25_train_record"]
                        and seam_p["v_meta_is_prep_record"])
    if not seam_p["ok"]:
        raise SystemExit(f"seam P failed: {seam_p}")
    keys = rows.keys
    rows_a, rows_h, rows_t = roles(keys)
    pool_a = K.PlayerPool(keys, rows.y, rows.col, rows_a)
    pool_h = K.PlayerPool(keys, rows.y, rows.col, rows_h)
    pool_t = K.PlayerPool(keys, rows.y, rows.col, rows_t)
    if set(pool_a.players) & set(pool_h.players):
        raise SystemExit("A and H share players")
    if set(pool_t.players) != set(pool_a.players) | set(pool_h.players):
        raise SystemExit("training players are not A u H")
    hold, hold_stats = K.holdout_set(pool_h, hold_per, K.HOLD_SEED)
    players_rec = {"A_players": len(pool_a.players), "A_games": int(len(rows_a)), "H_players": len(pool_h.players),
                   "H_games": int(len(rows_h)), "A_and_H": 0, "train_players": len(pool_t.players),
                   "train_games": int(len(rows_t)), "probe_rows": rows.n_probe, "v_rows": rows.n_v,
                   "probe_V_overlap": rows.overlap,
                   "train_players_per_level": np.bincount(pool_t.level, minlength=K.L).tolist(),
                   "train_eligible_n_level_colour": pool_t.counts(),
                   "holdout_questions": {str(n): int(len(v[1])) for n, v in hold.items()},
                   "holdout_sampling": hold_stats,
                   "holdout_use": "port check and a record-only NLL; chooses nothing (PLAN.md 1)"}
    players_rec["same_as_prereg25"] = same_players(players_rec, tr25["players"]) if not smoke else None
    if not smoke and not players_rec["same_as_prereg25"]:
        raise SystemExit(f"players / held-out set differ from prereg25: {players_rec}")
    log(f"players {json.dumps({k: players_rec[k] for k in ('A_players', 'H_players', 'train_players', 'train_games')})}")
    t = time.time()
    X32 = rows.standardised(mp["mu_x"], mp["sd_x"], smoke)     # S25's standardisation (the frozen Q09 map)
    log(f"standardised X {X32.shape} ({time.time() - t:.0f}s)")
    # ---- seam R33 first: A only, stop at best_step(0) -> S25 seed 0 byte for byte; else no S33 seed is trained
    ref = None
    if smoke:
        ref, ref_step = seam_r_smoke(X32, pool_a, hold, out_dir, steps, eval_every)
        sm = smoke_stop_steps(ref_step)
        seam_stop, stops = sm["seam"], sm["seeds"]
    else:
        seam_stop, stops = stops_real[SEAM_R_SEED], stops_real
    mpath0 = os.path.join(model_dir, model_name(SEAM_R_SEED, seam=True))
    spath0 = os.path.join(out_dir, f"seed{SEAM_R_SEED}_seamR_record.json")
    sr0 = train_one(SEAM_R_SEED, X32, pool_a, hold, steps, seam_stop, eval_every, smoke, mpath0, spath0, a.resume,
                    cs["head"], "seamR33")
    sr_seam = seam_r(mpath0, smoke, ref)
    sr_seam["stop_step"] = seam_stop
    sr_seam["state_at_stop"] = sr0.get("state_at_stop")
    log(f"seam R33 {json.dumps(sr_seam)}")
    base_rec = {"format": FORMAT, "smoke": smoke, "code": cs, "seam_r": sr_seam, "seed0_seamR": sr0}
    if not sr_seam["ok"]:
        T.write_json(os.path.join(out_dir, "train_seam_r_fail.json"),
                     {**base_rec, "created": r.now_iso(), "status": "stop: seam R33 failed",
                      "note": "the held-out evaluations of train_seed draw no random number and restore train mode, "
                              "so a different evaluation count cannot explain a difference (PLAN.md 1 fallback not "
                              "applicable): find the cause before any S33 seed is trained"})
        raise SystemExit("seam R33 failed: A-only seed 0 at best_step(0) differs from S25 seed 0; no S33 seed trained")
    seeds_rec = {}
    for seed in SEEDS:
        if os.path.exists(STOP):
            raise SystemExit(f"STOP file {STOP}: stopped before seed {seed}")
        seeds_rec[str(seed)] = train_one(seed, X32, pool_t, hold, steps, stops[seed], eval_every, smoke,
                                         os.path.join(model_dir, model_name(seed)),
                                         os.path.join(out_dir, f"seed{seed}_record.json"), a.resume, cs["head"], "S33")
    rec = {**base_rec, "created": r.now_iso(),
           "smoke_note": ("features replaced by default_rng(12345) normal noise, 300-step schedule, torch threads 4, "
                          "stop steps scaled: never a result") if smoke else None,
           "seconds": round(time.time() - t_all, 1),
           "what": "PLAN.md 1: S33 = S25's code, settings, rows and salt (imported unchanged) trained on A u H (7,367 "
                   "players), seed s stopped at prereg25's best_step(s) of the same 15,000-step OneCycle schedule; "
                   "seam R33 = A only, seed 0 at best_step(0), byte-identical to S25 seed 0; no inner data read",
           "settings": {**L29.registered_settings(K), "steps": steps, "eval_every": eval_every, "hold_per": hold_per,
                        "seeds": list(SEEDS), "seam_r_seed": SEAM_R_SEED, "model_format": MODEL_FORMAT,
                        "stop_steps": {str(s): stops[s] for s in SEEDS}, "seam_r_stop_step": seam_stop,
                        "stop_steps_prereg25": {str(s): stops_real[s] for s in SEEDS},
                        "standardisation": "frozen Q09 map mu_x / sd_x (= S25)"},
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                           "torch_threads": torch.get_num_threads()},
           "environment_prereg25": tr25.get("environment"),
           "q09_map_sha256": map_sha, "seam_p": seam_p,
           "v_prep_record": {k: rows.record[k] for k in ("games_sha256", "rows", "unusable", "files_sha256")},
           "players": players_rec, "seeds": seeds_rec,
           "models_sha256": {str(s): seeds_rec[str(s)]["model_sha256"] for s in SEEDS},
           "s25_models_sha256": {str(s): v for s, v in L29.s25_frozen_sha().items()}}
    T.write_json(rec_path, rec)
    log(f"-> {rec_path} ({time.time() - t_all:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
