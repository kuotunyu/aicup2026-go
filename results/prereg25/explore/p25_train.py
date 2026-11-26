"""prereg25 exploration (competition 2): train S25 (PLAN.md 2.5) = S19 with the validation players added (CPU, .venv
torch). Reads no inner data, never data/local; computes no candidate number.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg25/explore
    $S/p25_train.py train            -> data/prereg25/models/s25_seed{0,1,2}.npz + $S/train_record.json (run once; the
                                        code must be committed and clean)
    $S/p25_train.py train --smoke    -> data/prereg25/smoke/ (features replaced by default_rng(12345) normal noise,
                                        300 steps per seed, held-out 5 per cell: pipeline and timing only)
    --resume keeps a finished seed after an interruption (same commit and code as train_started.json).
    A file data/prereg25/STOP makes the run stop at the next held-out evaluation.

The model, the training loop, the sampler, the step choice, the port check and the numpy forward are imported unchanged
from results/prereg19/explore/p19_train.py and p19_lib.py. Only the rows differ: probe (data/prereg17/prep) + V
(data/prereg25/prep), with the fresh A / H salt of p25_lib.
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
P19 = os.path.join(REPO, "results", "prereg19", "explore")
for p in (os.path.join(REPO, "tools"), P19, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p19_train as T  # noqa: E402  (also puts the prereg12 finetune folder on sys.path)
import p19_lib as K  # noqa: E402
import p09_forward as FW  # noqa: E402
import p25_lib as L  # noqa: E402

r = T.r
guard_path = T.guard_path
DATA = os.path.join(REPO, "data", "prereg25")
V_PREP = os.path.join(DATA, "prep")
SMOKE_DIR = os.path.join(DATA, "smoke")
STOP = os.path.join(DATA, "STOP")
STARTED = os.path.join(DATA, "train_started.json")
RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p25_lib.py", "p25_train.py")
FORMAT = "c10-prereg25 explore train v1"
MODEL_FORMAT = "c10-prereg25 s25 model v1"
T.STOP = STOP                       # train_seed's STOP file (module global of p19_train)


def log(*x):
    print(*x, flush=True)


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES] + list(T.IMPORTED)
    rel = sorted(set(rel) | T.loaded_repo_modules() | L.loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = T.git("status", "--porcelain", "--", *rel)
    return {"head": T.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: T.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


class Rows:
    """probe rows (09's prep, as S19) followed by the V rows; keys '<rank>:<player_id>', levels, colours."""

    def __init__(self):
        self.prep = T.ProbeOnly(T.PROBE_PREP)
        if set(np.unique(self.prep.src)) != set(r.PROBE_SETS):
            raise SystemExit("probe rows of another set")
        pc = T.probe_colours(self.prep)
        self.vX = np.load(guard_path(os.path.join(V_PREP, "v_X.npy")), mmap_mode="r")
        with np.load(guard_path(os.path.join(V_PREP, "v_meta.npz")), allow_pickle=False) as z:
            vy = np.asarray(z["y"]).astype(np.int64)
            vk = np.asarray(z["players"]).astype(str)
            vc = np.asarray(z["color"]).astype(str)
        if self.vX.shape != (len(vy), FW.X_DIM):
            raise SystemExit("v_X.npy and v_meta.npz disagree")
        self.n_probe, self.n_v = len(self.prep.y), len(vy)
        self.keys = np.concatenate([np.asarray(self.prep.players).astype(str), vk])
        self.y = np.concatenate([self.prep.y, vy])
        self.col = np.concatenate([pc.astype(str), vc])
        pk, vks = set(self.keys[:self.n_probe].tolist()), set(vk.tolist())
        self.overlap = {"probe_and_V_keys": len(pk & vks),
                        "probe_and_V_player_id": len({k.split(":", 1)[1] for k in pk} & {k.split(":", 1)[1] for k in vks})}
        if self.overlap["probe_and_V_keys"] or self.overlap["probe_and_V_player_id"]:
            raise SystemExit(f"probe and V share players {self.overlap}")
        self.record = r.read_json(os.path.join(V_PREP, "prep_record.json"))

    def standardised(self, mu, sd, smoke):
        X = np.empty((self.n_probe + self.n_v, K.D_IN), dtype=np.float32)
        X[:self.n_probe] = T.standardised_probe(self.prep, mu, sd, smoke)
        for s in range(0, self.n_v, T.BLOCK):
            e = min(s + T.BLOCK, self.n_v)
            if smoke:
                X[self.n_probe + s:self.n_probe + e] = np.random.default_rng([T.NOISE_SEED, 2, s]).standard_normal(
                    (e - s, K.D_IN)).astype(np.float32)
            else:
                X[self.n_probe + s:self.n_probe + e] = FW.standardise_x(
                    np.asarray(self.vX[s:e], dtype=np.float64), mu, sd).astype(np.float32)
        return X


def save_model(path, params, seed, best, smoke):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, format=np.array(MODEL_FORMAT), seed=np.array(seed), best_step=np.array(best["step"]),
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
            raise SystemExit(f"the prereg25 files or the imported tools are not committed and clean: {cs['dirty']}")
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
    hold_per = K.SMOKE["hold_per"] if smoke else K.HOLD_PER
    mp = FW.load_map(T.Q09_MAP)
    map_sha = r.sha_raw(T.Q09_MAP)
    if map_sha != r.read_json(T.Q09_FROZEN)["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    want = (r.read_json(T.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = T.sha_file(os.path.join(T.PROBE_PREP, "probe_X.npy"))
    seam_p = {"want": want.get("probe_X.npy"), "got": got, "ok": bool(want.get("probe_X.npy") == got)}
    rows = Rows()
    vsha = T.sha_file(os.path.join(V_PREP, "v_X.npy"))
    seam_p["v_X_sha256"] = vsha
    seam_p["v_X_is_prep_record"] = bool(vsha == rows.record["files_sha256"]["v_X.npy"])
    seam_p["ok"] = bool(seam_p["ok"] and seam_p["v_X_is_prep_record"])
    if not seam_p["ok"]:
        raise SystemExit(f"seam P failed: {seam_p}")
    keys = rows.keys
    bit = {k: L.holdout_bit(k) for k in set(keys.tolist())}
    rows_a = np.array([i for i, k in enumerate(keys) if bit[k] == 0], dtype=np.int64)
    rows_h = np.array([i for i, k in enumerate(keys) if bit[k] == 1], dtype=np.int64)
    pool_a = K.PlayerPool(keys, rows.y, rows.col, rows_a)
    pool_h = K.PlayerPool(keys, rows.y, rows.col, rows_h)
    if set(pool_a.players) & set(pool_h.players):
        raise SystemExit("A and H share players")
    hold, hold_stats = K.holdout_set(pool_h, hold_per, K.HOLD_SEED)
    is_v = np.arange(len(keys)) >= rows.n_probe
    players_rec = {"A_players": len(pool_a.players), "A_games": int(len(rows_a)), "H_players": len(pool_h.players),
                   "H_games": int(len(rows_h)), "A_and_H": 0,
                   "A_V_players": len({keys[i] for i in rows_a[is_v[rows_a]]}),
                   "A_V_games": int(is_v[rows_a].sum()),
                   "H_V_players": len({keys[i] for i in rows_h[is_v[rows_h]]}),
                   "H_V_games": int(is_v[rows_h].sum()),
                   "probe_rows": rows.n_probe, "v_rows": rows.n_v, "probe_V_overlap": rows.overlap,
                   "A_players_per_level": np.bincount(pool_a.level, minlength=K.L).tolist(),
                   "H_players_per_level": np.bincount(pool_h.level, minlength=K.L).tolist(),
                   "A_eligible_n_level_colour": pool_a.counts(), "H_eligible_n_level_colour": pool_h.counts(),
                   "holdout_questions": {str(n): int(len(v[1])) for n, v in hold.items()},
                   "holdout_sampling": hold_stats}
    log(f"players {json.dumps({k: players_rec[k] for k in ('A_players', 'A_games', 'H_players', 'H_games', 'A_V_players', 'H_V_players')})}")
    t = time.time()
    X32 = rows.standardised(mp["mu_x"], mp["sd_x"], smoke)
    log(f"standardised X {X32.shape} ({time.time() - t:.0f}s)")
    import torch
    seeds_rec = {}
    for seed in K.SEEDS:
        mpath = os.path.join(model_dir, f"s25_seed{seed}.npz")
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
           "what": "PLAN.md 2.5: S25 = S19's model and training (imported unchanged) on probe + V rows, A / H by the "
                   "fresh salt c10-prereg25|holdout|; no inner data read",
           "settings": {"steps": steps, "eval_every": eval_every, "hold_per": hold_per, "batch": K.BATCH,
                        "ks": list(K.KS), "seeds": list(K.SEEDS), "lr": K.LR, "wd": K.WD, "pct_start": K.PCT_START,
                        "hidden": K.HIDDEN, "h_dim": K.H_DIM, "rho": K.RHO, "drop_in": K.DROP_IN, "drop": K.DROP,
                        "threads": K.THREADS, "hold_seed": K.HOLD_SEED, "sampler_base": K.SAMPLER_BASE,
                        "salt": L.SALT},
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                           "torch_threads": torch.get_num_threads()},
           "q09_map_sha256": map_sha, "seam_p": seam_p, "v_prep_record": {k: rows.record[k] for k in
                                                                         ("games_sha256", "rows", "unusable",
                                                                          "files_sha256")},
           "players": players_rec, "seeds": seeds_rec,
           "models_sha256": {str(s): seeds_rec[str(s)]["model_sha256"] for s in K.SEEDS}}
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
