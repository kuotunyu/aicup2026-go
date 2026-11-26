"""prereg29 exploration (competition 2): train the 7 new S29 seeds (PLAN.md 1) with S25's code, after seam R (seed 0
retrained here must equal S25 seed 0 byte for byte). CPU only (.venv torch). Reads no inner data, never the answers
folder; computes no candidate number.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg29/explore
    $S/p29_train.py train            -> data/prereg29/models/s29_seed0_retrain.npz (seam R), s29_seed{3..9}.npz
                                        + $S/train_record.json (run once; the code must be committed and clean)
    $S/p29_train.py train --smoke    -> data/prereg29/smoke/ (features replaced by default_rng(12345) normal noise,
                                        300 steps per seed, held-out 5 per cell: pipeline and timing only)
    --resume keeps a finished seed after an interruption (same commit and code as train_started.json).
    A file data/prereg29/STOP makes the run stop at the next held-out evaluation.

Rows, standardisation, player roles (salt), held-out set, model, training loop, sampler, step choice, port check and
the model file writer are imported unchanged from results/prereg25/explore/p25_train.py (which imports
results/prereg19/explore/p19_train.py / p19_lib.py). Only the seeds differ.
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
P25 = os.path.join(REPO, "results", "prereg25", "explore")
for p in (os.path.join(REPO, "tools"), P25, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p25_train as P25T  # noqa: E402  (Rows, save_model; imports p19_train as T, p19_lib as K, p25_lib)
import p29_lib as L  # noqa: E402

T, K, FW, L25 = P25T.T, P25T.K, P25T.FW, P25T.L
r = T.r
STOP = os.path.join(L.DATA, "STOP")
STARTED = os.path.join(L.DATA, "train_started.json")
RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p29_lib.py", "p29_train.py", "run_train.sh")
IMPORTED = tuple(T.IMPORTED) + ("results/prereg25/explore/p25_train.py", "results/prereg25/explore/p25_lib.py",
                                "results/prereg25/explore/train_record.json", "results/prereg25/rank/s25_seed0.npz",
                                "results/prereg25/rank/s25_seed1.npz", "results/prereg25/rank/s25_seed2.npz")
FORMAT = "c10-prereg29 explore train v1"
T.STOP = STOP          # train_seed's STOP file (module global of p19_train; p25_train had pointed it to prereg25)


def log(*x):
    print(*x, flush=True)


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES] + list(IMPORTED)
    rel = sorted(set(rel) | L.loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = T.git("status", "--porcelain", "--", *rel)
    return {"head": T.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: T.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def seam_r(path, smoke):
    """Seed 0 retrained by this program vs S25 seed 0 (frozen file and prereg25 train_record.json), byte for byte."""
    frozen = os.path.join(L.S25_DIR, "s25_seed0.npz")
    got = L.sha_file(path)
    out = {"retrain_file": os.path.relpath(path, REPO).replace("\\", "/"), "retrain_sha256": got,
           "s25_seed0_frozen_sha256": L.sha_file(frozen), "s25_seed0_train_record_sha256": L.s25_frozen_sha()[0]}
    if smoke:
        out.update(applicable=False, ok=True, note="smoke: noise features, not comparable")
        return out
    out["bytes_equal_frozen_file"] = L.models_equal_bytes(path, frozen)
    out["applicable"] = True
    out["ok"] = bool(out["bytes_equal_frozen_file"] and got == out["s25_seed0_frozen_sha256"]
                     == out["s25_seed0_train_record_sha256"])
    return out


def train_one(seed, X32, pool_a, hold, steps, eval_every, smoke, mpath, spath, resume, head):
    if resume and os.path.exists(mpath) and os.path.exists(spath):
        sr = r.read_json(spath)
        if sr.get("head") != head:
            raise SystemExit(f"seed {seed}: kept record of another commit ({sr.get('head')}); move it away first")
        if sr.get("model_sha256") != T.sha_file(mpath):
            raise SystemExit(f"{mpath}: SHA differs from its seed record")
        if not (sr.get("port_check") or {}).get("ok"):
            raise SystemExit(f"seed {seed}: kept record without a passed port check")
        log(f"seed {seed}: kept (resume)")
        return sr
    best, curve, stats, n_count, secs = T.train_seed(seed, X32, pool_a, hold, steps, eval_every,
                                                     "smoke" if smoke else "train")
    pc = T.port_check(best["params"], X32, hold)
    msha = P25T.save_model(mpath, best["params"], seed, best, smoke)
    sr = {"seed": seed, "best_step": best["step"], "best_holdout_nll": best["nll"], "curve": curve,
          "sampling": stats, "batches_per_n": n_count, "seconds": secs, "port_check": pc,
          "model_file": os.path.relpath(mpath, REPO).replace("\\", "/"), "model_sha256": msha, "head": head}
    T.write_json(spath, sr)
    log(f"seed {seed}: best step {best['step']} NLL {best['nll']:.5f}; port check {pc}")
    if not pc["ok"]:
        raise SystemExit(f"seed {seed}: torch / numpy port check failed {pc}")
    return sr


def cmd_train(a):
    t_all = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_dir = L.SMOKE_DIR if smoke else L.DATA
    model_dir = os.path.join(out_dir, "models")
    rec_path = os.path.join(L.SMOKE_DIR, "train_record.json") if smoke else RECORD
    cs = code_state()
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg29 files or the imported tools are not committed and clean: {cs['dirty']}")
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
    # the settings must be prereg25's (only the seeds differ)
    tr25 = L.read_json(L.P25_TRAIN)
    want = L.registered_settings(K)
    if any(tr25["settings"].get(k) != v for k, v in want.items()) or tr25["settings"].get("seeds") != list(L.S25_SEEDS):
        raise SystemExit("the imported settings differ from prereg25 train_record.json")
    steps = K.SMOKE["steps"] if smoke else K.STEPS
    eval_every = K.SMOKE["eval_every"] if smoke else K.EVAL_EVERY
    hold_per = K.SMOKE["hold_per"] if smoke else K.HOLD_PER
    mp = FW.load_map(T.Q09_MAP)
    map_sha = r.sha_raw(T.Q09_MAP)
    if map_sha != r.read_json(T.Q09_FROZEN)["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    wantp = (r.read_json(T.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = T.sha_file(os.path.join(T.PROBE_PREP, "probe_X.npy"))
    seam_p = {"want": wantp.get("probe_X.npy"), "got": got, "ok": bool(wantp.get("probe_X.npy") == got)}
    rows = P25T.Rows()
    vsha = T.sha_file(os.path.join(L.V_PREP, "v_X.npy"))
    seam_p["v_X_sha256"] = vsha
    seam_p["v_X_is_prep_record"] = bool(vsha == rows.record["files_sha256"]["v_X.npy"])
    seam_p["v_X_is_prereg25_train_record"] = bool(vsha == tr25["seam_p"]["v_X_sha256"])
    seam_p["v_meta_sha256"] = T.sha_file(os.path.join(L.V_PREP, "v_meta.npz"))
    seam_p["v_meta_is_prep_record"] = bool(seam_p["v_meta_sha256"] == rows.record["files_sha256"]["v_meta.npz"])
    seam_p["ok"] = bool(seam_p["ok"] and seam_p["v_X_is_prep_record"] and seam_p["v_X_is_prereg25_train_record"]
                        and seam_p["v_meta_is_prep_record"])
    if not seam_p["ok"]:
        raise SystemExit(f"seam P failed: {seam_p}")
    keys = rows.keys
    bit = {k: L25.holdout_bit(k) for k in set(keys.tolist())}
    rows_a = np.array([i for i, k in enumerate(keys) if bit[k] == 0], dtype=np.int64)
    rows_h = np.array([i for i, k in enumerate(keys) if bit[k] == 1], dtype=np.int64)
    pool_a = K.PlayerPool(keys, rows.y, rows.col, rows_a)
    pool_h = K.PlayerPool(keys, rows.y, rows.col, rows_h)
    if set(pool_a.players) & set(pool_h.players):
        raise SystemExit("A and H share players")
    hold, hold_stats = K.holdout_set(pool_h, hold_per, K.HOLD_SEED)
    players_rec = {"A_players": len(pool_a.players), "A_games": int(len(rows_a)), "H_players": len(pool_h.players),
                   "H_games": int(len(rows_h)), "A_and_H": 0, "probe_rows": rows.n_probe, "v_rows": rows.n_v,
                   "probe_V_overlap": rows.overlap,
                   "holdout_questions": {str(n): int(len(v[1])) for n, v in hold.items()},
                   "holdout_sampling": hold_stats}
    p25p = tr25["players"]
    players_rec["same_as_prereg25"] = bool(all(players_rec[k] == p25p[k] for k in
                                               ("A_players", "A_games", "H_players", "H_games", "holdout_questions"))
                                           ) if not smoke else None
    if not smoke and not players_rec["same_as_prereg25"]:
        raise SystemExit(f"players / held-out set differ from prereg25: {players_rec}")
    log(f"players {json.dumps({k: players_rec[k] for k in ('A_players', 'A_games', 'H_players', 'H_games')})}")
    t = time.time()
    X32 = rows.standardised(mp["mu_x"], mp["sd_x"], smoke)
    log(f"standardised X {X32.shape} ({time.time() - t:.0f}s)")
    import torch
    # ---- seam R first: seed 0 with this program must be S25 seed 0, else no new seed is trained
    mpath0 = os.path.join(model_dir, L.model_name(L.SEAM_R_SEED, retrain=True))
    spath0 = os.path.join(out_dir, f"seed{L.SEAM_R_SEED}_retrain_record.json")
    sr0 = train_one(L.SEAM_R_SEED, X32, pool_a, hold, steps, eval_every, smoke, mpath0, spath0, a.resume, cs["head"])
    sr0["best_step_prereg25"] = tr25["seeds"][str(L.SEAM_R_SEED)]["best_step"]
    sr0["best_holdout_nll_prereg25"] = tr25["seeds"][str(L.SEAM_R_SEED)]["best_holdout_nll"]
    sr_seam = seam_r(mpath0, smoke)
    log(f"seam R {json.dumps(sr_seam)}")
    base_rec = {"format": FORMAT, "smoke": smoke, "code": cs, "seam_r": sr_seam, "seed0_retrain": sr0}
    if not sr_seam["ok"]:
        T.write_json(os.path.join(out_dir, "train_seam_r_fail.json"), {**base_rec, "created": r.now_iso(),
                                                                       "status": "stop: seam R failed"})
        raise SystemExit("seam R failed: seed 0 retrained here differs from S25 seed 0; no new seed trained")
    seeds_rec = {}
    for seed in L.NEW_SEEDS:
        seeds_rec[str(seed)] = train_one(seed, X32, pool_a, hold, steps, eval_every, smoke,
                                         os.path.join(model_dir, L.model_name(seed)),
                                         os.path.join(out_dir, f"seed{seed}_record.json"), a.resume, cs["head"])
    rec = {**base_rec, "created": r.now_iso(),
           "smoke_note": "features replaced by default_rng(12345) normal noise, 300 steps: never a result"
           if smoke else None, "seconds": round(time.time() - t_all, 1),
           "what": "PLAN.md 1: S29 = S25's 3 frozen seeds + 7 new seeds (3..9) of S25's code, settings, rows and salt "
                   "(imported unchanged); seam R = seed 0 retrained byte-identical to S25 seed 0; no inner data read",
           "settings": {**L.registered_settings(K), "steps": steps, "eval_every": eval_every, "hold_per": hold_per,
                        "new_seeds": list(L.NEW_SEEDS), "s25_seeds_reused": list(L.S25_SEEDS),
                        "seam_r_seed": L.SEAM_R_SEED, "model_format": L.S25_FORMAT},
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                           "torch_threads": torch.get_num_threads()},
           "environment_prereg25": tr25.get("environment"),
           "q09_map_sha256": map_sha, "seam_p": seam_p,
           "v_prep_record": {k: rows.record[k] for k in ("games_sha256", "rows", "unusable", "files_sha256")},
           "players": players_rec, "seeds": seeds_rec,
           "models_sha256": {str(s): seeds_rec[str(s)]["model_sha256"] for s in L.NEW_SEEDS},
           "s25_models_sha256": {str(s): v for s, v in L.s25_frozen_sha().items()}}
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
