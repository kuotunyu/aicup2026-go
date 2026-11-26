"""prereg40 exploration (competition 2): train S40 (PLAN.md 2.1-2.2) = prereg23's SET2 on prereg25's rows (probe + V)
with prereg25's A / H split and S25's held-out questions (CPU, .venv torch). Reads no inner data, never the validation
answers; computes no candidate number (stage 0b and the judgement are p40_eval.py).

    Git Bash, repository root on a subst drive (as prereg23 / prereg25 ran their worktrees under AppData\\Local;
    p23_eval asks for it); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg40/explore
    $S/p40_train.py train            -> data/prereg40/models/s40_seed{0..4}.npz + $S/train_record.json (run once; the
                                        code must be committed and clean)
    $S/p40_train.py train --smoke    -> data/prereg40/smoke/ (features replaced by default_rng(12345) normal noise,
                                        300 steps per seed, held-out 5 per cell: pipeline and timing only)
    --resume keeps a finished seed after an interruption (same commit and code as train_started.json).
    A file data/prereg40/STOP makes the run stop at the next held-out evaluation.

Imported unchanged: p23_train.torch_model / train_seed / port_check / holdout_nll / export_params / probe_colours
(prereg23's SET2), p23_lib.features and settings, p19_lib.PlayerPool / holdout_set, p25_lib.holdout_bit.
"""

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P23 = os.path.join(REPO, "results", "prereg23", "explore")
for p in (os.path.join(REPO, "tools"), P23, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p23_train as T  # noqa: E402  (SET2 model and training loop; its K is p23_lib)
import p23_lib as K  # noqa: E402
import p19_lib as K19  # noqa: E402
import p09_forward as FW  # noqa: E402
import p40_lib as L  # noqa: E402

r = T.r
guard_path = T.guard_path
DATA = L.DATA
SMOKE_DIR = os.path.join(DATA, "smoke")
STOP = os.path.join(DATA, "STOP")
STARTED = os.path.join(DATA, "train_started.json")
RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p40_lib.py", "p40_train.py")
IMPORTED = tuple(T.IMPORTED) + ("results/prereg23/explore/p23_train.py", "results/prereg23/explore/p23_lib.py",
                                "results/prereg25/explore/p25_lib.py", "results/prereg25/explore/train_record.json",
                                "results/prereg26/explore/v_ex_record.json")
FORMAT = "c10-prereg40 explore train v1"
NOISE_SEED = 12345
BLOCK = 8192
T.STOP = STOP                       # train_seed's STOP file (module global of p23_train)


def log(*x):
    print(*x, flush=True)


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES] + list(IMPORTED)
    rel = sorted(set(rel) | T.loaded_repo_modules() | L.loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = T.git("status", "--porcelain", "--", *rel)
    return {"head": T.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: T.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def standardised_all(rows, mp, smoke):
    """float32 [N, 1584] of every row (probe then V), block by block; smoke = noise."""
    n = len(rows)
    X = np.empty((n, K.D_IN), dtype=np.float32)
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        if smoke:
            X[s:e] = np.random.default_rng([NOISE_SEED, 4, s]).standard_normal((e - s, K.D_IN)).astype(np.float32)
        else:
            Xr, EXr = rows.raw(np.arange(s, e))
            X[s:e] = L.features32(Xr, EXr, mp)
    return X


def load_s25_models():
    tr = r.read_json(L.S25_TRAIN_RECORD)
    pars, seam = [], {"models": {}, "ok": True}
    for s in L.S25_SEEDS:
        sr = tr["seeds"][str(s)]
        path = os.path.join(REPO, *sr["model_file"].split("/"))
        sha = T.sha_file(path)
        with np.load(guard_path(path), allow_pickle=False) as z:
            ok_fmt = str(z["format"]) == L.S25_MODEL_FORMAT
            par = {k: np.asarray(z[f"p_{k}"]) for k in K19.PARAM_NAMES}
            seed, best_step, msmoke = int(z["seed"]), int(z["best_step"]), bool(z["smoke"])
        ok = bool(ok_fmt and sha == tr["models_sha256"][str(s)] and seed == s and not msmoke
                  and best_step == sr["best_step"])
        seam["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": ok,
                                  "best_holdout_nll_record": sr["best_holdout_nll"]}
        seam["ok"] = bool(seam["ok"] and ok)
        pars.append(K19.p64(par))
    return tr, pars, seam


def seam_h(tr25, pars25, X32, hold, hold_stats):
    """S25's three frozen models on this held-out set (numpy float64, X part only) vs their recorded held-out NLL, and
    the held-out sampling record equal to S25's (PLAN.md 3, rev 1)."""
    lp = L.holdout_lp(K19.set_logp_np, pars25, hold, lambda idx: X32[idx][:, :FW.X_DIM])
    out = {"per_seed": {}, "tol": L.SEAM_H_TOL}
    ok = True
    for i, s in enumerate(L.S25_SEEDS):
        nll, _ = L.mean_nll(lp, hold, members=[i])
        rec = tr25["seeds"][str(s)]["best_holdout_nll"]
        out["per_seed"][str(s)] = {"numpy": nll, "record": rec, "abs_diff": abs(nll - rec)}
        ok = ok and abs(nll - rec) <= L.SEAM_H_TOL
    p25 = tr25["players"]
    out["holdout_questions_equal"] = bool({str(n): int(len(v[1])) for n, v in hold.items()}
                                          == p25["holdout_questions"])
    out["holdout_sampling_equal"] = bool(hold_stats == p25["holdout_sampling"])
    out["ok"] = bool(ok and out["holdout_questions_equal"] and out["holdout_sampling_equal"])
    return out


def save_model(path, params, seed, best, smoke):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, format=np.array(L.MODEL_FORMAT), seed=np.array(seed), best_step=np.array(best["step"]),
             smoke=np.array(bool(smoke)), **{f"p_{k}": params[k] for k in K.PARAM_NAMES})
    os.replace(tmp, path)
    return T.sha_file(path)


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
            raise SystemExit(f"the prereg40 files or the imported tools are not committed and clean: {cs['dirty']}")
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
    steps = K.SMOKE["steps"] if smoke else K.STEPS
    eval_every = K.SMOKE["eval_every"] if smoke else K.EVAL_EVERY
    hold_per = K.SMOKE["hold_per"] if smoke else L.HOLD_PER
    # ---- inputs and seam P
    mp = FW.load_map(T.Q09_MAP)
    map_sha = r.sha_raw(T.Q09_MAP)
    if map_sha != r.read_json(T.Q09_FROZEN)["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    want = (r.read_json(T.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {k: T.sha_file(os.path.join(L.PROBE_PREP, k)) for k in ("probe_X.npy", "probe_ex.npy")}
    vrec = r.read_json(os.path.join(L.V_PREP, "prep_record.json"))
    exrec = r.read_json(L.V_EX_RECORD)
    tr25, pars25, seam_s25 = load_s25_models()
    seam_p = {"want": {k: want.get(k) for k in got}, "got": got,
              "v_X.npy": T.sha_file(os.path.join(L.V_PREP, "v_X.npy")),
              "v_ex.npy": T.sha_file(L.V_EX), "v_meta.npz": T.sha_file(os.path.join(L.V_PREP, "v_meta.npz"))}
    seam_p["v_X_is_prep_record"] = bool(seam_p["v_X.npy"] == vrec["files_sha256"]["v_X.npy"])
    seam_p["v_X_is_s25_train_record"] = bool(seam_p["v_X.npy"] == (tr25.get("seam_p") or {}).get("v_X_sha256"))
    seam_p["v_ex_is_record"] = bool(seam_p["v_ex.npy"] == exrec["v_ex_sha256"] and exrec["seam_v"]["passed"])
    seam_p["v_meta_is_record"] = bool(seam_p["v_meta.npz"] == exrec["v_meta_sha256"])
    seam_p["ok"] = bool(all(want.get(k) == v for k, v in got.items()) and seam_p["v_X_is_prep_record"]
                        and seam_p["v_X_is_s25_train_record"] and seam_p["v_ex_is_record"]
                        and seam_p["v_meta_is_record"])
    if not (seam_p["ok"] and seam_s25["ok"]):
        raise SystemExit(f"seam P / S25 failed: {seam_p} {seam_s25}")
    # ---- rows, A / H, held-out questions (= S25's)
    prep23 = T.ProbeOnly(L.PROBE_PREP)
    if set(np.unique(prep23.src)) != set(r.PROBE_SETS):
        raise SystemExit("probe rows of another set")
    rows = L.Rows(T.probe_colours(prep23))
    del prep23
    rows_a, rows_h, pool_a, pool_h = L.split_pools(rows)
    hold, hold_stats = K19.holdout_set(pool_h, hold_per, L.HOLD_SEED)
    is_v = np.arange(len(rows)) >= rows.n_probe
    players_rec = {"A_players": len(pool_a.players), "A_games": int(len(rows_a)), "H_players": len(pool_h.players),
                   "H_games": int(len(rows_h)), "A_and_H": 0,
                   "A_V_players": len({rows.keys[i] for i in rows_a[is_v[rows_a]]}),
                   "A_V_games": int(is_v[rows_a].sum()),
                   "H_V_players": len({rows.keys[i] for i in rows_h[is_v[rows_h]]}),
                   "H_V_games": int(is_v[rows_h].sum()),
                   "probe_rows": rows.n_probe, "v_rows": rows.n_v, "probe_V_overlap": rows.overlap,
                   "A_players_per_level": np.bincount(pool_a.level, minlength=K.L).tolist(),
                   "H_players_per_level": np.bincount(pool_h.level, minlength=K.L).tolist(),
                   "A_eligible_n_level_colour": pool_a.counts(), "H_eligible_n_level_colour": pool_h.counts(),
                   "holdout_questions": {str(n): int(len(v[1])) for n, v in hold.items()},
                   "holdout_sampling": hold_stats}
    p25 = tr25["players"]
    seam_d = {k: bool(players_rec[k] == p25.get(k)) for k in
              ("A_players", "A_games", "H_players", "H_games", "A_V_players", "H_V_players", "A_players_per_level",
               "H_players_per_level")}
    seam_d["ok"] = bool(all(seam_d.values()))
    log(f"players {json.dumps({k: players_rec[k] for k in ('A_players', 'A_games', 'H_players', 'H_games')})}; "
        f"seam D {seam_d['ok']}")
    if not smoke and not seam_d["ok"]:
        raise SystemExit(f"seam D failed (A / H differ from S25): {seam_d}")
    t = time.time()
    X32 = standardised_all(rows, mp, smoke)
    log(f"standardised [X, EX] {X32.shape} ({time.time() - t:.0f}s)")
    sh = seam_h(tr25, pars25, X32, hold, hold_stats) if not smoke else {"ok": None, "note": "smoke: not run"}
    log(f"seam H {sh}")
    if not smoke and not sh["ok"]:
        raise SystemExit(f"seam H failed (held-out questions differ from S25's): {sh}")
    import torch
    seeds_rec = {}
    for seed in K.SEEDS:
        mpath = os.path.join(model_dir, f"s40_seed{seed}.npz")
        spath = os.path.join(out_dir, f"seed{seed}_record.json")
        if a.resume and os.path.exists(mpath) and os.path.exists(spath):
            seeds_rec[str(seed)] = r.read_json(spath)
            if seeds_rec[str(seed)].get("model_sha256") != T.sha_file(mpath):
                raise SystemExit(f"{mpath}: SHA differs from its seed record")
            if not (seeds_rec[str(seed)].get("port_check") or {}).get("ok"):
                raise SystemExit(f"seed {seed}: kept record without a passed port check")
            log(f"seed {seed}: kept (resume)")
            continue
        best, curve, stats, n_count, secs = T.train_seed(seed, X32, pool_a, hold, steps, eval_every,
                                                         "smoke" if smoke else "train")
        pc = T.port_check(best["params"], X32, hold)
        msha = save_model(mpath, best["params"], seed, best, smoke)
        sr = {"seed": seed, "best_step": best["step"], "best_holdout_nll": best["nll"], "curve": curve,
              "sampling": stats, "batches_per_n": n_count, "seconds": secs, "port_check": pc,
              "model_file": os.path.relpath(mpath, REPO).replace("\\", "/"), "model_sha256": msha}
        T.write_json(spath, sr)
        seeds_rec[str(seed)] = sr
        log(f"seed {seed}: best step {best['step']} NLL {best['nll']:.5f}; port check {pc}")
        if not pc["ok"]:
            raise SystemExit(f"seed {seed}: torch / numpy port check failed {pc}")
    rec = {"format": FORMAT, "created": r.now_iso(), "smoke": smoke,
           "smoke_note": "features replaced by default_rng(12345) normal noise, 300 steps: never a result"
           if smoke else None, "seconds": round(time.time() - t_all, 1), "code": cs,
           "what": "PLAN.md 2.1-2.2: S40 = prereg23's SET2 (imported unchanged) on probe + V rows, A / H by "
                   "prereg25's salt, S25's held-out questions; no inner data read",
           "settings": {"steps": steps, "eval_every": eval_every, "hold_per": hold_per, "batch": K.BATCH,
                        "ks": list(K.KS), "seeds": list(K.SEEDS), "lr": K.LR, "wd": K.WD, "pct_start": K.PCT_START,
                        "d_in": K.D_IN, "hidden": K.HIDDEN, "h_dim": K.H_DIM, "pool_dim": K.POOL_DIM, "rho": K.RHO,
                        "drop_in": K.DROP_IN, "drop": K.DROP, "threads": K.THREADS, "hold_seed": L.HOLD_SEED,
                        "sampler_base": K.SAMPLER_BASE, "salt": L.SALT},
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                           "torch_threads": torch.get_num_threads()},
           "q09_map_sha256": map_sha, "seam_p": seam_p, "seam_s25": seam_s25, "seam_d": seam_d, "seam_h": sh,
           "players": players_rec, "seeds": seeds_rec,
           "models_sha256": {str(s): seeds_rec[str(s)]["model_sha256"] for s in K.SEEDS},
           "peak_memory_record": peak_memory()}
    T.write_json(rec_path, rec)
    log(f"-> {rec_path} ({time.time() - t_all:.0f}s)")
    return 0


def peak_memory():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"peak_working_set_gb": round(getattr(mi, "peak_wset", 0) / 2 ** 30, 2),
                "peak_pagefile_gb": round(getattr(mi, "peak_pagefile", 0) / 2 ** 30, 2)}
    except Exception as e:  # record only
        return {"error": repr(e)}


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
