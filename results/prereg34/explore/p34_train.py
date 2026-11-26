"""prereg34 exploration (competition 2): train S34 (PLAN.md 4) and the stage 0b decision (PLAN.md 5). CPU only (.venv
torch). Reads no inner data, never the answers folder; computes no candidate number.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg34/explore
    $S/p34_train.py train            -> data/prereg34/models/s34_seed0_seamR.npz (seam R34: no OGS row, seed 0, must
                                        equal the frozen S25 seed 0 byte for byte), s34_seed{0,1,2}.npz
                                        + $S/train_record.json (run once; code and align_record.json committed, clean)
    $S/p34_train.py train --smoke    -> data/prereg34/smoke/ (synthetic OGS prep + smoke align record; every feature
                                        replaced by noise, 300-step schedule, torch threads 4: pipeline only)
    $S/p34_train.py stage0b          -> $S/stage0b.json (pure function of the committed train_record.json and
                                        prereg25's committed train record; judged once)
    $S/p34_train.py stage0b --smoke  -> data/prereg34/smoke/stage0b.json (from the smoke train record)
    --resume keeps a finished seed (and the seam-R34 model) after an interruption (same commit and code).
    A file data/prereg34/STOP stops the run at the next held-out evaluation (every 500 steps) and before each seed.

Everything is S25 (imported unchanged): rows (probe + V, results/prereg25/explore/p25_train.Rows), standardisation (the
frozen Q09 map's mu_x / sd_x; OGS rows standardised with the same mu / sd and never used to compute them), roles by
prereg25's salt (A trains, H = S25's 748 held-out players chooses the checkpoint), held-out questions, model, AdamW,
OneCycle, sampler, step choice, port check (prereg19's train_seed / port_check through prereg29's train_one) and S25's
model writer. The only difference (PLAN.md 4): the training pool is A u OGS, where OGS = the aligned OGS players
(labels from the committed align_record.json) that are eligible for at least one question (>= 5 usable rows of one
colour), capped per official level at that level's S25 A player count (cap_players: default_rng(CAP_SEED = 34), levels
ascending, players sorted by numeric OGS id, one permutation per level, the first `cap` kept). OGS players are never in
H. Every OGS player's usable rows are used (<= 25 per player, the stage 1a sample).
Seam R34 (PLAN.md 4): the pool without any OGS row (= S25's A pool; the OGS rows sit unused at the end of X), seed 0 ->
byte-identical to results/prereg25/rank/s25_seed0.npz (prereg29's seam_r).
"""

import argparse
import json
import math
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
import p29_train as P29T  # noqa: E402  (train_one, seam_r)
import p29_lib as L29  # noqa: E402
import p34_prep as PR  # noqa: E402
import p34_align as A  # noqa: E402

T, K, FW, L25 = P25T.T, P25T.K, P25T.FW, P25T.L
r = T.r
DATA = PR.DATA
SMOKE_DIR = PR.SMOKE_DIR
STOP = os.path.join(DATA, "STOP")
STARTED = os.path.join(DATA, "train_started.json")
RECORD = os.path.join(HERE, "train_record.json")
STAGE0B = os.path.join(HERE, "stage0b.json")
P25_TRAIN = L29.P25_TRAIN
OWN_FILES = ("PLAN.md", "p34_ogs.py", "p34_prep.py", "p34_align.py", "p34_train.py", "run_train.sh")
IMPORTED = tuple(P29T.IMPORTED) + ("results/prereg29/explore/p29_train.py", "results/prereg29/explore/p29_lib.py",
                                   "results/prereg31/explore/ram_watch.py")
FORMAT = "c10-prereg34 explore train v1"
S0B_FORMAT = "c10-prereg34 explore stage0b v1"
MODEL_FORMAT = L29.S25_FORMAT        # S34 files keep S25's format string (seam R34 compares bytes; eval loader)
SEEDS = (0, 1, 2)
SEAM_R_SEED = 0
SMOKE_THREADS = 4
CAP_SEED = 34
MIN_ROWS = min(K.KS)                 # a player is sampled only with >= 5 rows of the question's colour
GO_MEAN = -0.010
# PLAN.md 5: the frozen S25 best held-out NLL per seed (prereg25 train_record.json)
S25_NLL = {0: 1.0073546251903926, 1: 1.00711480947954, 2: 1.004849481311173}
T.STOP = STOP          # p19_train's module global (p25_train / p29_train had pointed it elsewhere); after all imports


def log(*x):
    print(*x, flush=True)


rel = PR.rel


def code_state(smoke):
    own = OWN_FILES + (() if smoke else ("align_record.json",))
    return A.code_state(own, IMPORTED)


# ---------------------------------------------------------------- pure parts (unit-tested)
def eligible_players(players, colours, min_rows=MIN_ROWS):
    """OGS players with >= min_rows usable rows of at least one colour."""
    cnt = {}
    for p, c in zip(np.asarray(players).astype(str).tolist(), np.asarray(colours).astype(str).tolist()):
        cnt.setdefault(p, {"B": 0, "W": 0})[c] += 1
    return {p for p, d in cnt.items() if max(d.values()) >= min_rows}


def cap_players(labels, eligible, caps, seed=CAP_SEED):
    """labels {player: level}; eligible set; caps [10] -> (chosen players sorted, record). One generator
    default_rng(seed); levels 0..9 ascending; the level's eligible players sorted by numeric OGS id; one permutation
    per level (also when the level is under its cap); the first caps[k] kept."""
    rng = np.random.default_rng(seed)
    chosen, rec = [], []
    for k in range(len(caps)):
        pl = sorted((p for p, lv in labels.items() if lv == k and p in eligible), key=A.game_number)
        perm = rng.permutation(len(pl))
        take = sorted((pl[i] for i in perm[:int(caps[k])].tolist()), key=A.game_number)
        chosen += take
        rec.append({"level": k, "eligible": len(pl), "cap": int(caps[k]), "kept": len(take)})
    return sorted(chosen), rec


def ogs_keys(players, labels):
    """'<LEVEL>:<player_id>' (the S25 key format) for OGS rows."""
    return np.array([f"{A.LEVELS[labels[p]]}:{p}" for p in np.asarray(players).astype(str).tolist()])


def stage0b_decision(s34, control, seeds=SEEDS, go_mean=GO_MEAN):
    """PLAN.md 5: GO iff mean over the seeds of (best held-out NLL S34 - S25, same seed) <= -0.010 and S34 is lower
    (strictly) in every seed. No tolerance."""
    if tuple(sorted(s34)) != tuple(seeds) or tuple(sorted(control)) != tuple(seeds):
        raise SystemExit("stage 0b needs seeds 0-2 on both sides")
    a = [float(s34[s]) for s in seeds]
    b = [float(control[s]) for s in seeds]
    if not all(math.isfinite(x) for x in a + b):
        raise SystemExit("stage 0b: a non-finite NLL")
    d = [x - y for x, y in zip(a, b)]
    mean = math.fsum(d) / len(d)
    lower = int(sum(x < y for x, y in zip(a, b)))
    c1, c2 = bool(mean <= go_mean), bool(lower == len(seeds))
    return {"seeds": list(seeds), "s34_best_holdout_nll": a, "s25_best_holdout_nll": b, "pair_diff_s34_minus_s25": d,
            "mean_diff_s34_minus_s25": mean, "threshold_mean_diff": go_mean, "condition_mean": c1,
            "seeds_s34_lower": lower, "condition_all_lower": c2, "decision": "GO" if (c1 and c2) else "NO-GO"}


def fill_s25_rows(X, rows, mu, sd, smoke):
    """X[:n_probe + n_v] = p25_train.Rows.standardised(mu, sd, smoke) without its temporary array (same blocks, same
    noise streams; unit-tested equal on synthetic data, and seam R34 checks it on the real run)."""
    n_p = rows.n_probe
    for s in range(0, n_p, T.BLOCK):
        e = min(s + T.BLOCK, n_p)
        if smoke:
            X[s:e] = np.random.default_rng([T.NOISE_SEED, 0, s]).standard_normal((e - s, K.D_IN)).astype(np.float32)
        else:
            X[s:e] = FW.standardise_x(np.asarray(rows.prep.probe_X[s:e], dtype=np.float64), mu, sd).astype(np.float32)
    for s in range(0, rows.n_v, T.BLOCK):
        e = min(s + T.BLOCK, rows.n_v)
        if smoke:
            X[n_p + s:n_p + e] = np.random.default_rng([T.NOISE_SEED, 2, s]).standard_normal(
                (e - s, K.D_IN)).astype(np.float32)
        else:
            X[n_p + s:n_p + e] = FW.standardise_x(np.asarray(rows.vX[s:e], dtype=np.float64), mu, sd).astype(
                np.float32)
    return X


def same_pool(p, q):
    return bool(p.players == q.players and np.array_equal(p.level, q.level)
                and all(np.array_equal(x[c], y[c]) for x, y in zip(p.games, q.games) for c in K.COLOURS))


def seam_r_smoke(rows, pool_a_ref, keys25, mp, hold_per, steps, eval_every, mpath0, model_dir):
    """Smoke analogue of seam R34 (noise features: the frozen S25 is not comparable): prereg25's own construction
    (p25_train.Rows.standardised, A / H pools on the official keys only, its held-out set), prereg19's train_seed and
    S25's writer must give the same bytes as this program's seed-0 seam model (X with the OGS rows appended, filled by
    fill_s25_rows, pools built on the extended keys)."""
    Xref = rows.standardised(mp["mu_x"], mp["sd_x"], True)
    bit = {k: L25.holdout_bit(k) for k in set(keys25.tolist())}
    rows_h = np.array([i for i, k in enumerate(keys25) if bit[k] == 1], dtype=np.int64)
    hold_ref, _st = K.holdout_set(K.PlayerPool(keys25, rows.y, rows.col, rows_h), hold_per, K.HOLD_SEED)
    best, _c, _s, _n, _t = T.train_seed(SEAM_R_SEED, Xref, pool_a_ref, hold_ref, steps, eval_every, "smoke seam-ref")
    del Xref
    ref = os.path.join(model_dir, "seamR_reference_p25_construction.npz")
    P25T.save_model(ref, best["params"], SEAM_R_SEED, best, True)
    eq = L29.models_equal_bytes(mpath0, ref)
    return {"reference_file": rel(ref), "reference_sha256": T.sha_file(ref), "bytes_equal": eq, "ok": bool(eq)}


def model_name(seed, seam=False):
    return f"s34_seed{seed}_seamR.npz" if seam else f"s34_seed{seed}.npz"


def read_align(smoke):
    path = A.SMOKE_RECORD if smoke else A.RECORD
    if not smoke and not A.committed_clean(path):
        raise SystemExit(f"{rel(path)} must be committed as written before the training")
    rec = r.read_json(path)
    if rec.get("format") != A.ALIGN_FORMAT or bool(rec.get("smoke")) != smoke:
        raise SystemExit(f"{rel(path)}: not a {'smoke ' if smoke else ''}align record")
    return rec, path


# ---------------------------------------------------------------- train
def cmd_train(a):
    t_all = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_dir = SMOKE_DIR if smoke else DATA
    model_dir = os.path.join(out_dir, "models")
    rec_path = os.path.join(SMOKE_DIR, "train_record.json") if smoke else RECORD
    cs = code_state(smoke)
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg34 files or the imported tools are not committed and clean: {cs['dirty']}")
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
    elif a.resume or a.rerun:
        raise SystemExit("--resume / --rerun are for the real run only")
    if os.path.exists(STOP):
        raise SystemExit(f"STOP file {STOP} exists: delete it by hand to start")
    if T.STOP != STOP:
        raise SystemExit("p19_train.STOP does not point to data/prereg34/STOP")
    import torch
    if smoke:
        K.THREADS = SMOKE_THREADS          # read by train_seed at call time (smoke only)
        torch.set_num_threads(SMOKE_THREADS)
    tr25 = L29.read_json(P25_TRAIN)
    want = L29.registered_settings(K)
    if smoke:
        want = {k: v for k, v in want.items() if k != "threads"}
    if any(tr25["settings"].get(k) != v for k, v in want.items()) or tr25["settings"].get("seeds") != list(SEEDS):
        raise SystemExit("the imported settings differ from prereg25 train_record.json")
    if {int(s): v["best_holdout_nll"] for s, v in tr25["seeds"].items()} != S25_NLL:
        raise SystemExit("prereg25 train_record.json best held-out NLLs differ from PLAN.md 5")
    steps = K.SMOKE["steps"] if smoke else K.STEPS
    eval_every = K.SMOKE["eval_every"] if smoke else K.EVAL_EVERY
    hold_per = K.SMOKE["hold_per"] if smoke else K.HOLD_PER
    # ---- inputs: seam P (= p29_train / p33_train)
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
    # ---- OGS rows and labels (PLAN.md 3 / 4)
    al, al_path = read_align(smoke)
    ogs = PR.OgsRows(PR.SMOKE_PREP if smoke else PR.PREP)
    ogs_sha = ogs.files_sha256()
    if ogs_sha != al["inputs"]["prep_files_sha256"] or ogs_sha != ogs.record["files_sha256"]:
        raise SystemExit("the OGS prep files differ from the align record / prep record")
    labels = {str(p): int(v) for p, v in al["labels"].items()}
    n25 = rows.n_probe + rows.n_v
    keys25 = rows.keys
    bit = {k: L25.holdout_bit(k) for k in set(keys25.tolist())}
    rows_a = np.array([i for i, k in enumerate(keys25) if bit[k] == 0], dtype=np.int64)
    rows_h = np.array([i for i, k in enumerate(keys25) if bit[k] == 1], dtype=np.int64)
    pool_a25_ref = K.PlayerPool(keys25, rows.y, rows.col, rows_a)          # p25_train's own construction
    caps = np.bincount(pool_a25_ref.level, minlength=K.L).tolist()
    om = ogs.meta
    labelled = np.array([p in labels for p in om["players"].tolist()], dtype=bool)
    elig = eligible_players(om["players"][labelled], om["color"][labelled])
    chosen, cap_rec = cap_players(labels, elig, caps)
    chosen_set = set(chosen)
    ogs_rows = np.array([i for i, p in enumerate(om["players"].tolist()) if p in chosen_set], dtype=np.int64)
    n_ogs = len(ogs_rows)
    keys = np.concatenate([keys25, ogs_keys(om["players"][ogs_rows], labels)])
    y = np.concatenate([rows.y, np.array([labels[p] for p in om["players"][ogs_rows].tolist()], dtype=np.int64)])
    col = np.concatenate([rows.col, om["color"][ogs_rows].astype(str)])
    rows_ogs = np.arange(n25, n25 + n_ogs, dtype=np.int64)
    pool_a25 = K.PlayerPool(keys, y, col, rows_a)
    pool_h = K.PlayerPool(keys, y, col, rows_h)
    pool34 = K.PlayerPool(keys, y, col, np.concatenate([rows_a, rows_ogs]))
    if not same_pool(pool_a25, pool_a25_ref):
        raise SystemExit("the A pool with the OGS rows appended differs from p25_train's A pool")
    if set(pool_a25.players) & set(pool_h.players) or set(pool34.players) & set(pool_h.players):
        raise SystemExit("training and held-out players overlap")
    ogs_in_pool = sorted(set(pool34.players) - set(pool_a25.players))
    if len(ogs_in_pool) != len(chosen) or len(pool34.players) != len(pool_a25.players) + len(chosen):
        raise SystemExit("OGS keys collide with official keys")
    hold, hold_stats = K.holdout_set(pool_h, hold_per, K.HOLD_SEED)
    per_level_ogs = np.bincount(np.array([labels[p] for p in chosen], dtype=np.int64), minlength=K.L).tolist()
    players_rec = {"A_players": len(pool_a25.players), "A_games": int(len(rows_a)), "H_players": len(pool_h.players),
                   "H_games": int(len(rows_h)), "A_and_H": 0, "probe_rows": rows.n_probe, "v_rows": rows.n_v,
                   "probe_V_overlap": rows.overlap,
                   "OGS_labelled_players": len(labels), "OGS_eligible_players": len(elig),
                   "OGS_players": len(chosen), "OGS_games": int(n_ogs), "OGS_players_per_level": per_level_ogs,
                   "OGS_cap": cap_rec, "OGS_cap_seed": CAP_SEED,
                   "S25_A_players_per_level": caps,
                   "train_players": len(pool34.players), "train_games": int(len(rows_a) + n_ogs),
                   "train_players_per_level": np.bincount(pool34.level, minlength=K.L).tolist(),
                   "train_eligible_n_level_colour": pool34.counts(),
                   "holdout_questions": {str(n): int(len(v[1])) for n, v in hold.items()},
                   "holdout_sampling": hold_stats,
                   "holdout_use": "checkpoint choice (S25's rule, PLAN.md 4) and stage 0b (PLAN.md 5)"}
    players_rec["same_as_prereg25"] = bool(all(players_rec[k] == tr25["players"][k] for k in
                                               ("A_players", "A_games", "H_players", "H_games", "holdout_questions"))
                                           ) if not smoke else None
    if not smoke and not players_rec["same_as_prereg25"]:
        raise SystemExit(f"players / held-out set differ from prereg25: {players_rec}")
    if any(per_level_ogs[k] > caps[k] for k in range(K.L)):
        raise SystemExit("OGS players over the cap")
    log(f"players {json.dumps({k: players_rec[k] for k in ('A_players', 'H_players', 'OGS_players', 'OGS_games')})}")
    t = time.time()
    X32 = np.empty((n25 + n_ogs, K.D_IN), dtype=np.float32)
    fill_s25_rows(X32, rows, mp["mu_x"], mp["sd_x"], smoke)              # S25's standardisation (frozen Q09 map)
    ogs.standardised_into(X32[n25:], ogs_rows, mp["mu_x"], mp["sd_x"], smoke)   # same mu / sd
    log(f"standardised X {X32.shape} ({time.time() - t:.0f}s)")
    # ---- seam R34 first: no OGS row in the pool, seed 0 -> S25 seed 0 byte for byte; else no S34 seed is trained
    mpath0 = os.path.join(model_dir, model_name(SEAM_R_SEED, seam=True))
    spath0 = os.path.join(out_dir, f"seed{SEAM_R_SEED}_seamR_record.json")
    sr0 = P29T.train_one(SEAM_R_SEED, X32, pool_a25, hold, steps, eval_every, smoke, mpath0, spath0, a.resume,
                         cs["head"])
    sr_seam = P29T.seam_r(mpath0, smoke)
    sr_seam["pool"] = "S25 A only (no OGS row); the OGS rows are at the end of X, never sampled"
    sr_seam["a_pool_same_as_p25_construction"] = True
    if smoke:
        sr_seam["smoke_reference"] = seam_r_smoke(rows, pool_a25_ref, keys25, mp, hold_per, steps, eval_every,
                                                  mpath0, model_dir)
        sr_seam["ok"] = bool(sr_seam["ok"] and sr_seam["smoke_reference"]["ok"])
    log(f"seam R34 {json.dumps(sr_seam)}")
    base_rec = {"format": FORMAT, "smoke": smoke, "code": cs, "seam_r": sr_seam, "seed0_seamR": sr0}
    if not sr_seam["ok"]:
        T.write_json(os.path.join(out_dir, "train_seam_r_fail.json"),
                     {**base_rec, "created": r.now_iso(), "status": "stop: seam R34 failed"})
        raise SystemExit("seam R34 failed: seed 0 without OGS rows differs from S25 seed 0; no S34 seed trained")
    seeds_rec = {}
    for seed in SEEDS:
        if os.path.exists(STOP):
            raise SystemExit(f"STOP file {STOP}: stopped before seed {seed}")
        sr = P29T.train_one(seed, X32, pool34, hold, steps, eval_every, smoke,
                            os.path.join(model_dir, model_name(seed)), os.path.join(out_dir, f"seed{seed}_record.json"),
                            a.resume, cs["head"])
        sr["players"] = "A+OGS"
        seeds_rec[str(seed)] = sr
    import torch as _t
    rec = {**base_rec, "created": r.now_iso(),
           "smoke_note": ("synthetic OGS players, every feature replaced by noise, 300-step schedule, torch threads "
                          "4: never a result") if smoke else None,
           "seconds": round(time.time() - t_all, 1),
           "what": "PLAN.md 4: S34 = S25's code, settings, rows, salt and held-out set (imported unchanged) with the "
                   "aligned OGS players added to the training role A, capped per level at S25's A player count; "
                   "seam R34 = no OGS row, seed 0, byte-identical to S25 seed 0; no inner data read",
           "settings": {**L29.registered_settings(K), "steps": steps, "eval_every": eval_every, "hold_per": hold_per,
                        "seeds": list(SEEDS), "seam_r_seed": SEAM_R_SEED, "model_format": MODEL_FORMAT,
                        "cap_seed": CAP_SEED, "min_rows_eligible": MIN_ROWS,
                        "standardisation": "frozen Q09 map mu_x / sd_x (= S25) for official and OGS rows"},
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": _t.__version__,
                           "torch_threads": _t.get_num_threads()},
           "environment_prereg25": tr25.get("environment"),
           "q09_map_sha256": map_sha, "seam_p": seam_p,
           "v_prep_record": {k: rows.record[k] for k in ("games_sha256", "rows", "unusable", "files_sha256")},
           "ogs": {"align_record": rel(al_path), "align_record_sha256_lf": A.sha_lf(al_path),
                   "prep_files_sha256": ogs_sha, "prep_rows": int(ogs.n), "players": chosen},
           "players": players_rec, "seeds": seeds_rec,
           "models_sha256": {str(s): seeds_rec[str(s)]["model_sha256"] for s in SEEDS},
           "s25_models_sha256": {str(s): v for s, v in L29.s25_frozen_sha().items()}}
    T.write_json(rec_path, rec)
    log(f"-> {rec_path} ({time.time() - t_all:.0f}s)")
    return 0


# ---------------------------------------------------------------- stage 0b (PLAN.md 5)
def train_record_checks(tr, smoke):
    s = tr.get("settings") or {}
    want = L29.registered_settings(K)
    if smoke:
        want = {k: v for k, v in want.items() if k not in ("threads", "steps", "eval_every", "hold_per")}
    return {"format": tr.get("format") == FORMAT, "smoke_flag": bool(tr.get("smoke")) == smoke,
            "code_clean": smoke or (tr.get("code") or {}).get("dirty") == "",
            "seam_r_ok": bool((tr.get("seam_r") or {}).get("ok")),
            "seam_r_applicable": smoke or bool((tr.get("seam_r") or {}).get("applicable")),
            "seam_p_ok": bool((tr.get("seam_p") or {}).get("ok")),
            "settings": all(s.get(k) == v for k, v in want.items()) and s.get("seeds") == list(SEEDS),
            "seeds_present": sorted(int(x) for x in (tr.get("seeds") or {})) == list(SEEDS),
            "port_checks": all((v.get("port_check") or {}).get("ok") for v in (tr.get("seeds") or {}).values()),
            "same_players_as_prereg25": smoke or bool((tr.get("players") or {}).get("same_as_prereg25"))}


def cmd_stage0b(a):
    smoke = bool(a.smoke)
    out = os.path.join(SMOKE_DIR, "stage0b.json") if smoke else STAGE0B
    tpath = os.path.join(SMOKE_DIR, "train_record.json") if smoke else RECORD
    if os.path.exists(out):
        raise SystemExit(f"{rel(out)} exists: stage 0b is judged once")
    cs = code_state(smoke)
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg34 files or the imported tools are not committed and clean: {cs['dirty']}")
        if not A.committed_clean(tpath):
            raise SystemExit(f"{rel(tpath)} must be committed as written before stage 0b")
    tr = r.read_json(tpath)
    checks = train_record_checks(tr, smoke)
    if not all(checks.values()):
        raise SystemExit(f"train_record.json is not a finished, clean S34 training: {checks}")
    tr25 = L29.read_json(P25_TRAIN)
    control = {int(s): float(v["best_holdout_nll"]) for s, v in tr25["seeds"].items()}
    if control != S25_NLL:
        raise SystemExit("prereg25 train_record.json best held-out NLLs differ from PLAN.md 5")
    if not smoke and tr["players"]["holdout_questions"] != tr25["players"]["holdout_questions"]:
        raise SystemExit("the held-out questions differ from S25's")
    s34 = {int(s): float(v["best_holdout_nll"]) for s, v in tr["seeds"].items()}
    dec = stage0b_decision(s34, control)
    rec = {"format": S0B_FORMAT, "created": r.now_iso(), "smoke": smoke, "code": cs,
           "smoke_note": "smoke: noise-trained S34 against the real S25 NLLs; meaningless" if smoke else None,
           "what": "PLAN.md 5: GO iff mean over seeds 0-2 of the best held-out NLL, S34 - S25 (same seed) <= -0.010 "
                   "and S34 lower in all 3 seeds; S25 = prereg25 train_record.json; pure function of the committed "
                   "records; judged once",
           "train_record": rel(tpath), "train_record_sha256_lf": A.sha_lf(tpath), "train_record_checks": checks,
           "control_source": "results/prereg25/explore/train_record.json seeds.<s>.best_holdout_nll",
           "holdout_questions": tr["players"]["holdout_questions"], **dec}
    T.write_json(out, rec)
    log(f"stage 0b: {dec['decision']} (mean diff {dec['mean_diff_s34_minus_s25']:+.5f}, S34 lower in "
        f"{dec['seeds_s34_lower']} of 3)")
    log(f"-> {rel(out)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    p = sub.add_parser("stage0b")
    p.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    return {"train": cmd_train, "stage0b": cmd_stage0b}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
