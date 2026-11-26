"""prereg31 exploration (competition 2): the HD-only naive Bayes (record-only), the S31 training (PLAN.md 1.5 / 3)
and the stage 0b decision. CPU only (.venv torch). Reads no inner data, never the answers folder; computes no
candidate number.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg31/explore
    $S/p31_train.py hdnb             -> $S/hd_nb.json (PLAN.md revision 1 item 1; record-only; run once)
    $S/p31_train.py train            -> data/prereg31/models/s31_seed0_hdoff_retrain.npz (seam R, HD off),
                                        s31_seed{0..9}.npz + $S/train_record.json (run once; the code must be
                                        committed and clean)
    $S/p31_train.py train --smoke    -> data/prereg31/smoke/ (features replaced by default_rng(12345) normal noise,
                                        the 2 HD columns permuted across rows with default_rng(12345), 300 steps per
                                        seed, held-out 5 per cell: pipeline and timing only, never a result)
    $S/p31_train.py stage0b          -> $S/stage0b.json (pure function of the committed train_record.json and the
                                        committed prereg25 / prereg29 train records; run once)
    --resume keeps a finished seed after an interruption (same commit and code as train_started.json).
    --low-priority sets the process priority class to BELOW_NORMAL (no numeric effect; torch threads stay 8).
    A file data/prereg31/STOP makes the run stop at the next held-out evaluation and before the next seed.

Rows, the standardisation of the 1,540 features, player roles (salt), held-out set, model, training loop, sampler,
step choice and port check are imported unchanged from results/prereg25/explore/p25_train.py (which imports
results/prereg19/explore/p19_train.py / p19_lib.py). The seam-R retrain of seed 0 (HD off) is prereg29's train_one
and seam_r, unchanged. The only difference of S31: 2 standardised HD columns appended after the 1,540 and
p19_lib.D_IN = 1542 inside p31_lib.input_width around train_seed / port_check.
"""

import argparse
import gc
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

import p25_train as P25T  # noqa: E402  (Rows; imports p19_train as T, p19_lib as K, p25_lib)
import p29_train as P29T  # noqa: E402  (train_one / seam_r of the HD-off retrain of seed 0)
import p29_lib as L29  # noqa: E402
import p31_lib as L  # noqa: E402

T, K, FW, L25 = P25T.T, P25T.K, P25T.FW, P25T.L
r = T.r
STOP = L.STOP
STARTED = os.path.join(L.DATA, "train_started.json")
RECORD = L.TRAIN_RECORD
OWN_FILES = ("PLAN.md", "p31_hd.py", "p31_lib.py", "p31_train.py", "hd_stats.json", "run_train.sh", "ram_watch.py")
IMPORTED = tuple(P29T.IMPORTED) + ("results/prereg29/explore/p29_train.py", "results/prereg29/explore/p29_lib.py",
                                   "results/prereg29/explore/train_record.json")
FORMAT = L.TRAIN_FORMAT
NB_FORMAT = "c10-prereg31 explore hd naive bayes v1"
S0B_FORMAT = "c10-prereg31 explore stage0b v1"
BELOW_NORMAL = 0x00004000
T.STOP = STOP          # train_seed's STOP file (module global of p19_train; p25_train / p29_train had pointed it
#                        to their own folders). Must stay after every import above.


def log(*x):
    print(*x, flush=True)


def rel(path):
    return os.path.relpath(path, REPO).replace("\\", "/")


def code_state():
    files = [rel(os.path.join(HERE, f)) for f in OWN_FILES] + list(IMPORTED)
    files = sorted(set(files) | L.loaded_repo_modules())
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


# ---------------------------------------------------------------- rows, roles, held-out set, HD join
def roles(rows, hold_per):
    """A / H rows by p25_lib.holdout_bit, the two pools and the held-out questions (as p25_train / p29_train)."""
    keys = rows.keys
    bit = {k: L25.holdout_bit(k) for k in set(keys.tolist())}
    rows_a = np.array([i for i, k in enumerate(keys) if bit[k] == 0], dtype=np.int64)
    rows_h = np.array([i for i, k in enumerate(keys) if bit[k] == 1], dtype=np.int64)
    pool_a = K.PlayerPool(keys, rows.y, rows.col, rows_a)
    pool_h = K.PlayerPool(keys, rows.y, rows.col, rows_h)
    if set(pool_a.players) & set(pool_h.players):
        raise SystemExit("A and H share players")
    hold, hold_stats = K.holdout_set(pool_h, hold_per, K.HOLD_SEED)
    rec = {"A_players": len(pool_a.players), "A_games": int(len(rows_a)), "H_players": len(pool_h.players),
           "H_games": int(len(rows_h)), "A_and_H": 0, "probe_rows": rows.n_probe, "v_rows": rows.n_v,
           "probe_V_overlap": rows.overlap,
           "holdout_questions": {str(n): int(len(v[1])) for n, v in hold.items()},
           "holdout_sampling": hold_stats}
    return rows_a, rows_h, pool_a, pool_h, hold, rec


def same_players(players_rec, p25p):
    return bool(all(players_rec[k] == p25p[k] for k in
                    ("A_players", "A_games", "H_players", "H_games", "holdout_questions")))


def hd_join(rows):
    """HD of every training row by (rank, game_id) (PLAN.md 1.5) -> (hd int8 [N], stage 0a record, seam A record)."""
    s0a = L.stage0a()
    if not s0a["ok"]:
        raise SystemExit(f"stage 0a is not a PASS with the table on disk: {s0a}")
    tab = L.HdTable()
    games = L.concat_games(L.probe_games(rows.prep.src, rows.prep.ref, rows.prep.players, T.SETS_JSON),
                           L.v_games(L.V_META, L.V_SETS, rows.record["sets_sha256"]))
    agree = L.rows_agree(games, rows.keys, rows.y, rows.col)
    hd, jrec = tab.join(games["rank"], games["game_id"], games["player_id"], games["color"])
    seam = {"table_sha256": tab.sha256, "table_sha256_hd_stats": s0a["table_sha256_hd_stats"],
            "table_is_hd_stats": bool(tab.sha256 == s0a["table_sha256_hd_stats"]), "table_rows": tab.n,
            "probe_rows": rows.n_probe, "probe_rows_expected": L.N_PROBE,
            "v_rows": rows.n_v, "v_rows_expected": L.N_V, "rows_agree": agree, "join": jrec}
    seam["ok"] = bool(seam["table_is_hd_stats"] and rows.n_probe == L.N_PROBE and rows.n_v == L.N_V
                      and agree["ok"] and jrec["ok"] and jrec["rows"] == L.N_PROBE + L.N_V)
    return hd, s0a, seam


# ---------------------------------------------------------------- hdnb (PLAN.md revision 1 item 1; record-only)
def cmd_hdnb(a):
    r.blas_env_check(False)
    if os.path.exists(L.HD_NB):
        raise SystemExit(f"{L.HD_NB} exists: hdnb runs once")
    cs = code_state()
    if cs["dirty"]:
        raise SystemExit(f"the prereg31 files or the imported tools are not committed and clean: {cs['dirty']}")
    rows = P25T.Rows()
    rows_a, _rows_h, _pool_a, _pool_h, hold, players_rec = roles(rows, K.HOLD_PER)
    players_rec["same_as_prereg25"] = same_players(players_rec, L.read_json(L29.P25_TRAIN)["players"])
    if not players_rec["same_as_prereg25"]:
        raise SystemExit(f"players / held-out set differ from prereg25: {players_rec}")
    hd, s0a, seam_a = hd_join(rows)
    if not seam_a["ok"]:
        raise SystemExit(f"seam A failed: {seam_a}")
    enc = s0a["encoding"]
    flag = L.flag_array(hd, enc)
    ci = L.colour_index(rows.col)
    P, n, n1 = L.nb_table(flag, rows.y, ci, rows_a)
    res = L.nb_nll(P, hold, flag, ci)
    rec = {"format": NB_FORMAT, "created": r.now_iso(), "code": cs,
           "what": "PLAN.md revision 1 item 1 (record-only, no threshold, decides nothing): P(flag | level r, "
                   "colour c) from the role-A training rows with add-1 smoothing; posterior over levels on the "
                   "held-out questions of stage 0b ~ prod_k P(z_k | r, c_k) with a uniform prior; mean NLL of the "
                   "true level against ln 10",
           "encoding": enc, "table_sha256": seam_a["table_sha256"], "seam_a": seam_a,
           "players": players_rec, "ranks": list(L.HD.RANKS), "colours": list(L.HD.COLOURS),
           "n_role_a": n.tolist(), "n_flag_role_a": n1.tolist(), "p_flag": P.tolist(), "holdout": res}
    T.write_json(L.HD_NB, rec)
    log(f"hdnb: {json.dumps(res)}")
    log(f"-> {L.HD_NB}")
    return 0


# ---------------------------------------------------------------- train
def hd_record(mu, sd, encoding):
    return {"encoding": encoding, "hd_mu": [float(x) for x in mu], "hd_sd": [float(x) for x in sd]}


def train_one(seed, X34, pool_a, hold, steps, eval_every, smoke, mpath, spath, resume, head, mu, sd, encoding):
    """One S31 seed (HD on): p19_train.train_seed and port_check with p19_lib.D_IN = 1542 inside input_width."""
    hdrec = hd_record(mu, sd, encoding)
    if resume and os.path.exists(mpath) and os.path.exists(spath):
        sr = r.read_json(spath)
        if sr.get("head") != head:
            raise SystemExit(f"seed {seed}: kept record of another commit ({sr.get('head')}); move it away first")
        if sr.get("model_sha256") != T.sha_file(mpath):
            raise SystemExit(f"{mpath}: SHA differs from its seed record")
        if not (sr.get("port_check") or {}).get("ok"):
            raise SystemExit(f"seed {seed}: kept record without a passed port check")
        _par, meta = L.load_model(mpath, K.PARAM_NAMES)
        if sr.get("hd") != hdrec or hd_record(meta["hd_mu"], meta["hd_sd"], meta["encoding"]) != hdrec \
                or meta["seed"] != seed or meta["smoke"] != bool(smoke):
            raise SystemExit(f"seed {seed}: kept model of another HD standardisation / encoding / seed")
        log(f"seed {seed}: kept (resume)")
        return sr
    with L.input_width(K, L.D_IN_HD):
        best, curve, stats, n_count, secs = T.train_seed(seed, X34, pool_a, hold, steps, eval_every,
                                                         "smoke" if smoke else "train")
        pc = T.port_check(best["params"], X34, hold)
    if K.D_IN != FW.X_DIM:
        raise SystemExit("p19_lib.D_IN was not restored")
    msha = L.save_model(mpath, best["params"], seed, best["step"], smoke, mu, sd, encoding, K.PARAM_NAMES)
    sr = {"seed": seed, "best_step": best["step"], "best_holdout_nll": best["nll"], "curve": curve,
          "sampling": stats, "batches_per_n": n_count, "seconds": secs, "port_check": pc,
          "model_file": rel(mpath), "model_sha256": msha, "head": head, "input_width": L.D_IN_HD, "hd": hdrec}
    T.write_json(spath, sr)
    log(f"seed {seed}: best step {best['step']} NLL {best['nll']:.5f}; port check {pc}")
    if not pc["ok"]:
        raise SystemExit(f"seed {seed}: torch / numpy port check failed {pc}")
    return sr


def cmd_train(a):
    t_all = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    prio = low_priority() if a.low_priority else {"requested": False}
    out_dir = L.SMOKE_DIR if smoke else L.DATA
    model_dir = os.path.join(out_dir, "models")
    rec_path = os.path.join(L.SMOKE_DIR, "train_record.json") if smoke else RECORD
    cs = code_state()
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg31 files or the imported tools are not committed and clean: {cs['dirty']}")
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
        raise SystemExit("p19_train.STOP does not point to data/prereg31/STOP")
    # the settings must be prereg25's (only the seeds and the input width differ)
    tr25 = L.read_json(L29.P25_TRAIN)
    want = L.registered_settings(K)
    if any(tr25["settings"].get(k) != v for k, v in want.items()) \
            or tr25["settings"].get("seeds") != list(L29.S25_SEEDS):
        raise SystemExit("the imported settings differ from prereg25 train_record.json")
    if K.D_IN != FW.X_DIM or FW.X_DIM + L.N_HD != L.D_IN_HD:
        raise SystemExit("input widths: expected 1540 and 1542")
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
    vsha = T.sha_file(os.path.join(L29.V_PREP, "v_X.npy"))
    seam_p["v_X_sha256"] = vsha
    seam_p["v_X_is_prep_record"] = bool(vsha == rows.record["files_sha256"]["v_X.npy"])
    seam_p["v_X_is_prereg25_train_record"] = bool(vsha == tr25["seam_p"]["v_X_sha256"])
    seam_p["v_meta_sha256"] = T.sha_file(L.V_META)
    seam_p["v_meta_is_prep_record"] = bool(seam_p["v_meta_sha256"] == rows.record["files_sha256"]["v_meta.npz"])
    seam_p["ok"] = bool(seam_p["ok"] and seam_p["v_X_is_prep_record"] and seam_p["v_X_is_prereg25_train_record"]
                        and seam_p["v_meta_is_prep_record"])
    if not seam_p["ok"]:
        raise SystemExit(f"seam P failed: {seam_p}")
    rows_a, _rows_h, pool_a, _pool_h, hold, players_rec = roles(rows, hold_per)
    players_rec["same_as_prereg25"] = same_players(players_rec, tr25["players"]) if not smoke else None
    if not smoke and not players_rec["same_as_prereg25"]:
        raise SystemExit(f"players / held-out set differ from prereg25: {players_rec}")
    log(f"players {json.dumps({k: players_rec[k] for k in ('A_players', 'A_games', 'H_players', 'H_games')})}")
    # ---- seam A: HD of every row, before any training
    hd, s0a, seam_a = hd_join(rows)
    log(f"seam A {json.dumps(seam_a)}")
    base_rec = {"format": FORMAT, "smoke": smoke, "code": cs, "stage0a": s0a, "seam_a": seam_a}
    if not seam_a["ok"]:
        T.write_json(os.path.join(out_dir, "train_seam_a_fail.json"), {**base_rec, "created": r.now_iso(),
                                                                       "status": "stop: seam A failed"})
        raise SystemExit("seam A failed: a training row without HD, or player / colour differ; nothing trained")
    enc = s0a["encoding"]
    cols = L.hd_columns(hd, L.colour_index(rows.col), enc)
    if smoke:          # no real HD signal in a smoke run: the 2 columns are permuted across rows
        cols = cols[np.random.default_rng(T.NOISE_SEED).permutation(len(cols))]
    mu, sd = L.hd_moments(cols, rows_a)
    hd_rec = {**hd_record(mu, sd, enc), "table_sha256": seam_a["table_sha256"],
              "role_a_rows": int(len(rows_a)), "rows": int(len(hd)),
              "hd_groups_all_rows": seam_a["join"]["hd_groups"],
              "hd_groups_role_a": [int(np.sum(hd[rows_a] == 0)), int(np.sum(hd[rows_a] == 1)),
                                   int(np.sum(hd[rows_a] >= 2))],
              "columns": ["x_B = flag * [target plays Black]", "x_W = flag * [target plays White]"],
              "smoke_columns_permuted": smoke}
    log(f"HD inputs {json.dumps(hd_rec)}")
    t = time.time()
    X32 = rows.standardised(mp["mu_x"], mp["sd_x"], smoke)
    log(f"standardised X {X32.shape} ({time.time() - t:.0f}s)")
    if X32.shape != (len(hd), FW.X_DIM):
        raise SystemExit(f"standardised X has shape {X32.shape}")
    import torch
    # ---- seam R first (HD off, nothing patched): seed 0 must be S25 seed 0, else no S31 seed is trained
    mpath0 = os.path.join(model_dir, L.model_name(L.SEAM_R_SEED, hd_off_retrain=True))
    spath0 = os.path.join(out_dir, f"seed{L.SEAM_R_SEED}_hdoff_retrain_record.json")
    sr0 = P29T.train_one(L.SEAM_R_SEED, X32, pool_a, hold, steps, eval_every, smoke, mpath0, spath0, a.resume,
                         cs["head"])
    sr0["best_step_prereg25"] = tr25["seeds"][str(L.SEAM_R_SEED)]["best_step"]
    sr0["best_holdout_nll_prereg25"] = tr25["seeds"][str(L.SEAM_R_SEED)]["best_holdout_nll"]
    sr_seam = P29T.seam_r(mpath0, smoke)
    log(f"seam R {json.dumps(sr_seam)}")
    base_rec.update(seam_r=sr_seam, seed0_hdoff_retrain=sr0)
    if not sr_seam["ok"]:
        T.write_json(os.path.join(out_dir, "train_seam_r_fail.json"), {**base_rec, "created": r.now_iso(),
                                                                       "status": "stop: seam R failed"})
        raise SystemExit("seam R failed: seed 0 retrained here with HD off differs from S25 seed 0; no S31 seed "
                         "trained")
    # ---- the S31 inputs: the 1,540 standardised features, then the 2 standardised HD columns
    X34 = np.empty((len(hd), L.D_IN_HD), dtype=np.float32)
    X34[:, :FW.X_DIM] = X32
    X34[:, FW.X_DIM:] = L.hd_standardise(cols, mu, sd).astype(np.float32)
    del X32
    gc.collect()
    seeds_rec = {}
    for seed in L.SEEDS:
        if os.path.exists(STOP):
            raise SystemExit(f"STOP file {STOP}: stopped before seed {seed}")
        seeds_rec[str(seed)] = train_one(seed, X34, pool_a, hold, steps, eval_every, smoke,
                                         os.path.join(model_dir, L.model_name(seed)),
                                         os.path.join(out_dir, f"seed{seed}_record.json"), a.resume, cs["head"],
                                         mu, sd, enc)
    rec = {**base_rec, "created": r.now_iso(),
           "smoke_note": "features replaced by default_rng(12345) normal noise, HD columns permuted across rows, "
                         "300 steps: never a result" if smoke else None,
           "seconds": round(time.time() - t_all, 1),
           "what": "PLAN.md 1.5 / 3: S31 = S25's code, settings, rows and salt (imported unchanged) with 2 "
                   "standardised HD inputs appended (input width 1542), seeds 0-9; seam R = seed 0 retrained with "
                   "HD off byte-identical to S25 seed 0; no inner data read",
           "settings": {**L.registered_settings(K), "steps": steps, "eval_every": eval_every, "hold_per": hold_per,
                        "seeds": list(L.SEEDS), "seam_r_seed": L.SEAM_R_SEED, "model_format": L.MODEL_FORMAT,
                        "model_format_hd_off": L29.S25_FORMAT, "input_width": L.D_IN_HD,
                        "input_width_hd_off": FW.X_DIM},
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                           "torch_threads": torch.get_num_threads()},
           "environment_prereg25": tr25.get("environment"), "priority": prio,
           "q09_map_sha256": map_sha, "seam_p": seam_p,
           "v_prep_record": {k: rows.record[k] for k in ("games_sha256", "rows", "unusable", "files_sha256")},
           "players": players_rec, "hd": hd_rec, "seeds": seeds_rec,
           "models_sha256": {str(s): seeds_rec[str(s)]["model_sha256"] for s in L.SEEDS},
           "s25_models_sha256": {str(s): v for s, v in L29.s25_frozen_sha().items()}}
    T.write_json(rec_path, rec)
    log(f"-> {rec_path} ({time.time() - t_all:.0f}s)")
    return 0


# ---------------------------------------------------------------- stage 0b (PLAN.md 3)
train_record_checks = L.train_record_checks


def cmd_stage0b(a):
    if os.path.exists(L.STAGE0B):
        raise SystemExit(f"{L.STAGE0B} exists: stage 0b is judged once")
    cs = code_state()
    if cs["dirty"]:
        raise SystemExit(f"the prereg31 files or the imported tools are not committed and clean: {cs['dirty']}")
    rrel = rel(RECORD)
    if not os.path.exists(RECORD) or T.git("ls-files", "--", rrel) != rrel \
            or T.git("status", "--porcelain", "--", rrel):
        raise SystemExit(f"{rrel} must be committed as written before stage 0b")
    tr = L.read_json(RECORD)
    checks = train_record_checks(tr)
    if not all(checks.values()):
        raise SystemExit(f"train_record.json is not a finished, clean S31 training: {checks}")
    control, src = L.control_nll()
    hq = tr["players"]["holdout_questions"]
    if not (hq == src["holdout_questions_prereg25"] == src["holdout_questions_prereg29"]):
        raise SystemExit("the held-out questions differ from the control records")
    s31 = {int(s): float(v["best_holdout_nll"]) for s, v in tr["seeds"].items()}
    dec = L.stage0b_decision(s31, control)
    rec = {"format": S0B_FORMAT, "created": r.now_iso(), "code": cs,
           "what": "PLAN.md 3: GO iff mean over seeds 0-9 of the best held-out NLL, S31 - control <= -0.010 and S31 "
                   "is lower in at least 8 of the 10 same-seed pairs; control = no-HD models of the same seeds "
                   "(prereg25 seeds 0-2, prereg29 seeds 3-9); pure function of the committed records; judged once",
           "train_record_sha256_lf": L.HD.sha_lf(RECORD), "train_record_checks": checks, "control_sources": src,
           "holdout_questions": hq, "encoding": tr["hd"]["encoding"], **dec}
    T.write_json(L.STAGE0B, rec)
    log(f"stage 0b: {dec['decision']} (mean diff {dec['mean_diff_s31_minus_control']:+.5f}, "
        f"S31 lower in {dec['pairs_s31_lower']} of 10)")
    log(f"-> {L.STAGE0B}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hdnb")
    p = sub.add_parser("train")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    p.add_argument("--low-priority", action="store_true")
    sub.add_parser("stage0b")
    a = ap.parse_args(argv)
    return {"hdnb": cmd_hdnb, "train": cmd_train, "stage0b": cmd_stage0b}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
