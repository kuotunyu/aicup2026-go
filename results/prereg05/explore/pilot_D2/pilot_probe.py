"""prereg05 PILOT of D2 (train-player lookup by opening, conditional switch) in the official question shape.

    .venv/Scripts/python.exe results/prereg05/explore/pilot_D2/pilot_probe.py
    -> results/prereg05/explore/pilot_D2/pilot_report.json, pilot_arrays.npz

Every fixed choice is in pilot_variants.json (written before any similarity or fusion was computed; its SHA-256 is
checked at start and end and recorded). Design-stage simulation on inner pseudo-questions of training players only,
NOT a validation result: nothing is scored, results/score_ledger.csv is not touched, data/local is never opened
(u1_lib guard, installed before anything else), no existing file is changed, no p02_player_* module is imported.
CPU only, BLAS / OpenMP threads 1 in every process, at most 4 worker processes (spawn).

Stages (stage 1 / 1b outputs are cached in _tmp/, which is deleted after the report is written):
  0   seams s1: u1 mono_P_Q02 rank-balanced 0.6106683012815998; run_gen reproduces mono_P_Q02; mono set equal
  1   tokens (d2_probe code), IDF, fit profiles, inner game counts, inner full profiles; seam s3
  1b  per inner player (4 workers): question vectors, own profile = all inner games minus the question's games,
      cosines vs 6,403 reference profiles, top-10 per scenario and orientation; seam s4
  2   S2, theta, V3, w per fold, fusion (seam s2), metrics, PlayerBootstrap, pools, criteria
"""

import os
import sys

sys.dont_write_bytecode = True
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"

import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPLORE = os.path.dirname(HERE)
for _p in (os.path.join(EXPLORE, "u1"), os.path.join(EXPLORE, "D2_train_player_lookup")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from scipy import optimize, sparse  # noqa: E402

import u1_lib as U  # noqa: E402  (installs the data/local guard first; imports the frozen tools read-only)
import d2_probe as D2  # noqa: E402  (token / similarity / vote code, imported unchanged; it sets threads to 4 ...)

for _k in THREAD_VARS:  # ... after numpy is loaded, which has no effect here; reset so spawned workers inherit 1
    os.environ[_k] = "1"

import p02_moves as PM  # noqa: E402

rf, p4, p3 = U.rf, U.p4, U.p3
T0 = time.time()
TMP = os.path.join(HERE, "_tmp")
SPEC = os.path.join(HERE, "pilot_variants.json")
SPEC_SHA = "d887c30a77a85a67f3834011869c30e773bc849b8426724584141ea168d3bc6d"
DS = os.path.join(U.REPO, "data", "prereg04", "design_scratch")
MANIFEST = os.path.join(U.REPO, "results", "prereg04", "design", "scratch_manifest.json")
U1_REPRO = os.path.join(EXPLORE, "u1", "u1_repro.json")
U1_OOF = os.path.join(EXPLORE, "u1", "u1_oof.npz")
D2_REPORT = os.path.join(EXPLORE, "D2_train_player_lookup", "d2_report.json")
D2_PROBE = os.path.join(EXPLORE, "D2_train_player_lookup", "d2_probe.py")
D2_VARIANTS = os.path.join(EXPLORE, "D2_train_player_lookup", "d2_variants.json")
LEVELS = U.LEVELS
L = 10
K = 10
NS = (5, 10, 15, 20)
SCEN = ("pi1", "pi0", "pi05")
DIRS = ("raw", "canon")
N_FIT = 5767
IDF_N = 5768
MASKED = np.float32(-9.0)
Q02_MONO_RB = 0.6106683012815998
TOL = 1e-12
W_BOUNDS = (0.0, 4.0)
N_WORKERS = 4
Q_CHUNK = 64
N_POOLS = 1000
POOL_PER_K = 100
ON_SHARE = 0.10
TERMS = ["OP", "O4", "SWA", "SWB"]
LOGLINES = []


def log(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    LOGLINES.append(line)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def no_player_modules():
    bad = sorted(m for m in sys.modules if m.startswith("p02_player"))
    if bad:
        raise SystemExit(f"competition-1 modules imported: {bad}")


def check_spec():
    sh = sha256_file(SPEC)
    if sh != SPEC_SHA:
        raise SystemExit(f"pilot_variants.json SHA-256 {sh} differs from the registered {SPEC_SHA}")
    return sh


# ================================================================ stage 0: seams s1
def stage0():
    z = U.load_saved(U1_OOF)
    P02 = z["mono_P_Q02"]
    m, _ = rf.metrics(rf.rule_b_np(P02), z["mono_y"], z["mono_n"], z["mono_player"])
    s1a = {"rank_balanced": m["rank_balanced"], "recorded": Q02_MONO_RB,
           "abs_diff": abs(m["rank_balanced"] - Q02_MONO_RB), "exactly_equal": m["rank_balanced"] == Q02_MONO_RB}
    log(f"s1: u1 mono_P_Q02 rank-balanced {m['rank_balanced']!r} (diff {s1a['abs_diff']:.3g})")
    if s1a["abs_diff"] > TOL:
        raise SystemExit(f"seam s1 failed: {s1a}")
    d = U.load_inputs(verbose=False)
    inner = d["inner"]
    ms, Xm = inner["ms"], inner["X_mono"]
    eq = {"player": np.array_equal(ms["player"], z["mono_player"]), "y": np.array_equal(ms["y"], z["mono_y"]),
          "fold": np.array_equal(ms["fold"], z["mono_fold"]), "k": np.array_equal(ms["k"], z["mono_n"]),
          "n": np.array_equal(Xm.n.astype(np.int64), z["mono_n"]),
          "colour": np.array_equal(ms["colour"], z["mono_colour"]),
          "round": np.array_equal(np.array([q[0] for q in ms["questions"]]), z["mono_round"])}
    mg = z["mono_games"]
    eq["games"] = all(np.array_equal(np.array(g, dtype=np.int64), mg[q][mg[q] >= 0].astype(np.int64))
                      for q, g in enumerate(ms["groups"]))
    eq["player_id"] = [p["player_id"] for p in inner["players"]] == z["player_id"].tolist()
    eq["player_fold"] = [p["fold"] for p in inner["players"]] == z["player_fold"].tolist()
    eq["player_level"] = [LEVELS.index(p["rank"]) for p in inner["players"]] == z["player_level"].tolist()
    t = time.time()
    _P, Pm, _th = U.run_gen(d, TERMS)
    s1b = {"run_gen_mono_P_max_abs_diff": float(np.max(np.abs(Pm - P02))), "seconds": round(time.time() - t, 1)}
    s1b["passed"] = s1b["run_gen_mono_P_max_abs_diff"] <= TOL
    log(f"s1: run_gen reproduces mono_P_Q02: max abs diff {s1b['run_gen_mono_P_max_abs_diff']:.3g}; mono set equal "
        f"{all(eq.values())}")
    if not s1b["passed"] or not all(eq.values()):
        raise SystemExit(f"seam s1 failed: {s1b} {eq}")
    seam = {"s1_u1_mono_rank_balanced": s1a, "s1_run_gen": s1b, "s1_mono_set_equal": eq,
            "mono_set_sha256": ms["sha256"], "passed": True}
    return {"z": z, "inner": inner, "seam": seam,
            "inputs_sha256": dict(inner["inputs"], **d["sha"])}


# ================================================================ stage 1: tokens, profiles
def py_tokens(pts, cols, tc):
    """Independent tokeniser written from d2_variants.json 'tokens' (seam s3b)."""
    tp = [p for p, c in zip(pts, cols) if c == tc][:10]
    gp = list(pts[:10])

    def code(prefix, kind, i, a, b=361):
        return (((prefix * 5 + kind) * 10 + i) * 361 + a) * 362 + b
    out = []
    for prefix in (0, 1 + tc):
        out += [code(prefix, 0, i, p) for i, p in enumerate(tp)]
        out += [code(prefix, 1, 0, p) for p in set(tp)]
        out += [code(prefix, 2, i, p) for i, p in enumerate(gp)]
        out += [code(prefix, 3, 0, p) for p in set(gp)]
        out += [code(prefix, 4, 0, gp[i], gp[i + 1]) for i in range(len(gp) - 1)]
    return sorted(out)


def py_moves(mv, off, r, direc):
    s, e = int(off[r]), int(off[r + 1])
    codes = [int(x) for x in mv[s:min(e, s + PM.MAX_MOVES)]]
    pts = [c & 511 for c in codes]
    cols = [c >> 9 for c in codes]
    if direc == "canon":
        smv = [("W" if c else "B", PM.from_xy(p % 19, p // 19)) for p, c in zip(pts, cols)]
        t = PM.canon(smv, 10)
        pts = []
        for _c, point in t:
            x, y = PM.to_xy(point)
            pts.append(y * 19 + x)
    return pts, cols


def check_py_tokens(mv, off, rows, tc, direc, tk_rows):
    bad = 0
    for j, r in enumerate(rows):
        pts, cols = py_moves(mv, off, r, direc)
        ref = py_tokens(pts, cols, int(tc[j]))
        mine = sorted(int(x) for x in tk_rows[j] if x >= 0)
        bad += ref != mine
    return {"checked": int(len(rows)), "mismatches": int(bad)}


def stage1(s0):
    os.makedirs(TMP, exist_ok=True)
    meta_path = os.path.join(TMP, "stage1.json")
    if os.path.exists(meta_path) and all(os.path.exists(os.path.join(TMP, f"work_{d}.npz")) for d in DIRS) \
            and os.path.exists(os.path.join(TMP, "common.npz")):
        log("stage 1: cached")
        return json.load(open(meta_path, encoding="utf-8"))
    meta = {}
    z = s0["z"]
    inner = s0["inner"]
    keys, players = inner["keys"], inner["players"]
    NP = len(players)

    man = {e["file"]: e["sha256"] for e in json.load(open(MANIFEST, encoding="utf-8"))["large_files"]}
    shas = {}
    for name in ("moves_fit.npz", "moves_fit_moves.npy", "moves_inner.npz", "moves_inner_moves.npy"):
        sh = sha256_file(os.path.join(DS, name))
        want = man[f"data/prereg04/design_scratch/{name}"]
        shas[name] = {"sha256": sh, "manifest": want, "equal": sh == want}
        if sh != want:
            raise SystemExit(f"{name}: SHA-256 differs from the scratch manifest")
    meta["inputs_sha256"] = shas
    log("design_scratch moves files: SHA-256 equal to the PREREG_04 scratch manifest")

    zf = np.load(os.path.join(DS, "moves_fit.npz"))
    off_f = zf["off"]
    pid_f = zf["player_id"]
    col_f = zf["color"]
    lab_f = zf["label"].astype(np.int64)
    zf.close()
    mv_f = np.load(os.path.join(DS, "moves_fit_moves.npy"))
    if len(mv_f) != off_f[-1]:
        raise SystemExit("moves_fit_moves.npy length differs from off")
    fit_ids, fit_pidx = np.unique(pid_f, return_inverse=True)
    if len(fit_ids) != N_FIT:
        raise SystemExit(f"{len(fit_ids)} fit players, expected {N_FIT}")
    fit_level = np.full(N_FIT, -1, np.int64)
    fit_level[fit_pidx] = lab_f
    if not (fit_level[fit_pidx] == lab_f).all():
        raise SystemExit("a fit player has two levels")
    if set(fit_ids.tolist()) & set(p["player_id"] for p in players):
        raise SystemExit("fit and inner players overlap")
    tc_f = (col_f == "W").astype(np.int64)
    order = np.argsort(fit_pidx, kind="stable")
    meta["fit_games_used"] = int(len(order))
    meta["fit_cap"] = 0
    del pid_f, col_f

    zi = np.load(os.path.join(DS, "moves_inner.npz"))
    off_i = zi["off"]
    row_of = {(r, g): i for i, (r, g) in enumerate(zip(zi["rank"].tolist(), zi["game_id"].tolist()))}
    perm = np.array([row_of[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])], dtype=np.int64)
    if len(perm) != keys["n"] or len(set(perm.tolist())) != keys["n"] or len(row_of) != keys["n"]:
        raise SystemExit("moves_inner does not hold exactly the inner_keys games")
    if not (np.array_equal(zi["player_id"][perm], np.array(keys["player_id"]))
            and np.array_equal(zi["color"][perm], np.array(keys["color"]))
            and np.array_equal(zi["label"][perm].astype(np.int64), keys["label"])):
        raise SystemExit("moves_inner player / colour / label differ from inner_keys")
    zi.close()
    mv_i = np.load(os.path.join(DS, "moves_inner_moves.npy"))
    if len(mv_i) != off_i[-1]:
        raise SystemExit("moves_inner_moves.npy length differs from off")
    tc_i = (np.array(keys["color"]) == "W").astype(np.int64)
    if not np.array_equal(np.array(keys["color"]), z["game_colour"]):
        raise SystemExit("inner_keys colours differ from u1 game_colour")

    pid_of = {p["player_id"]: i for i, p in enumerate(players)}
    inner_pl = np.array([pid_of.get(p, -1) for p in keys["player_id"]], dtype=np.int64)
    n_inner = np.bincount(inner_pl[inner_pl >= 0], minlength=NP)
    if not np.array_equal(n_inner, z["player_n_inner_games"]):
        raise SystemExit("inner games per player differ from u1 player_n_inner_games")
    mg = z["mono_games"]
    mp_ = z["mono_player"]
    if not (inner_pl[mg[mg >= 0]] == np.repeat(mp_, (mg >= 0).sum(1))).all():
        raise SystemExit("a mono question game is not a game of its player")
    rows_by_player = [np.nonzero(inner_pl == pi)[0] for pi in range(NP)]
    pg_ptr = np.concatenate([[0], np.cumsum([len(r) for r in rows_by_player])]).astype(np.int64)
    pg_rows = np.concatenate(rows_by_player).astype(np.int64)
    meta["inner"] = {"players": NP, "games_of_players": int(pg_ptr[-1]), "games_not_of_players": int((inner_pl < 0).sum()),
                     "games_per_player_median": float(np.median(n_inner)), "min": int(n_inner.min()),
                     "max": int(n_inner.max())}
    keep05 = np.array([hashlib.sha256(("c10-p05-d2|" + p["player_id"]).encode()).digest()[0] < 128 for p in players])
    meta["pi05_kept_inner_players"] = int(keep05.sum())

    rng_seam = np.random.default_rng(2026)
    seam_q = np.sort(rng_seam.choice(len(mp_), size=300, replace=False))
    np.savez(os.path.join(TMP, "common.npz"), mono_games=mg, mono_player=mp_, pg_ptr=pg_ptr, pg_rows=pg_rows,
             keep05=keep05, seam_q=seam_q, fit_level=fit_level, n_inner_games=keys["n"])

    pt_map, key_map = D2.sym_tables()
    rng = np.random.default_rng(12345)
    canon_checks, py_checks = {}, {}
    d2_stage1 = json.load(open(D2_REPORT, encoding="utf-8"))["stage1"]
    seam_vocab = {}
    for direc in DIRS:
        td = time.time()
        acc_k, acc_c = [], []
        py_fit_rows = np.sort(rng.choice(len(order), size=1500, replace=False))
        py_fit_tok = []
        for c0 in range(0, len(order), D2.CHUNK):
            rows = order[c0:c0 + D2.CHUNK]
            pt, col = D2.first_moves(mv_f, off_f, rows)
            if direc == "canon":
                s = D2.canon_index(pt, key_map)
                ptc = D2.canonise(pt, s, pt_map)
                if c0 == 0:
                    canon_checks["fit"] = D2.check_canon(pt, col, s, ptc, 20000, rng)
                pt = ptc
            tk = D2.tokens(pt, col, tc_f[rows])
            sel = py_fit_rows[(py_fit_rows >= c0) & (py_fit_rows < c0 + len(rows))]
            for j in sel:
                py_fit_tok.append((int(order[j]), int(tc_f[order[j]]), tk[j - c0].copy()))
            kk = (fit_pidx[rows][:, None].astype(np.int64) << 25) | tk.astype(np.int64)
            u, cnt = np.unique(kk[tk >= 0], return_counts=True)
            acc_k.append(u)
            acc_c.append(cnt)
        allk = np.concatenate(acc_k)
        allc = np.concatenate(acc_c)
        del acc_k, acc_c
        uk, inv = np.unique(allk, return_inverse=True)
        cnt_f = np.bincount(inv, weights=allc).astype(np.int64)
        del allk, allc, inv
        fp_player = (uk >> 25).astype(np.int64)
        fp_code = (uk & ((1 << 25) - 1)).astype(np.int32)
        del uk
        pt, col = D2.first_moves(mv_i, off_i, perm)
        if direc == "canon":
            s = D2.canon_index(pt, key_map)
            ptc = D2.canonise(pt, s, pt_map)
            canon_checks["inner"] = D2.check_canon(pt, col, s, ptc, 20000, rng)
            pt = ptc
        tk_i = D2.tokens(pt, col, tc_i)
        # s3b: independent tokeniser on 1,500 fit + 1,500 inner games
        fr = [r for r, _t, _k in py_fit_tok]
        a = check_py_tokens(mv_f, off_f, fr, [t for _r, t, _k in py_fit_tok], direc, [k for _r, _t, k in py_fit_tok])
        ir = np.sort(rng.choice(keys["n"], size=1500, replace=False))
        b = check_py_tokens(mv_i, off_i, perm[ir], tc_i[ir], direc, tk_i[ir])
        py_checks[direc] = {"fit": a, "inner": b}
        vocab = np.unique(np.concatenate([fp_code, tk_i[tk_i >= 0]]))
        V = len(vocab)
        fp_tok = np.searchsorted(vocab, fp_code)
        df = np.bincount(fp_tok, minlength=V)
        idf = np.log(IDF_N / (1.0 + df)) + 1.0
        Pf = sparse.csr_matrix((D2.weights(cnt_f.astype(np.float64), idf[fp_tok]), (fp_player, fp_tok)),
                               shape=(N_FIT, V))
        del fp_player, fp_code, fp_tok, cnt_f
        gi, gj = np.nonzero(tk_i >= 0)
        Gin = sparse.csr_matrix((np.ones(len(gi)), (gi, np.searchsorted(vocab, tk_i[gi, gj]))), shape=(keys["n"], V))
        Gin.sum_duplicates()
        Gin.sort_indices()
        A = sparse.csr_matrix((np.ones(len(pg_rows)), (inner_pl[pg_rows], pg_rows)), shape=(NP, keys["n"]))
        Ci = (A @ Gin).tocsr()
        Ci.sum_duplicates()
        Ci.data = D2.weights(Ci.data, idf[Ci.indices])
        P = sparse.vstack([Pf, Ci]).tocsr()
        P, nrm = D2.row_normalise(P)
        if not (nrm > 0).all():
            raise SystemExit("an empty reference profile")
        sv = {"size": int(V), "sha256": hashlib.sha256(vocab.astype(np.int32).tobytes()).hexdigest(),
              "fit_profile_nnz": int(Pf.nnz), "inner_full_profile_nnz": int(Ci.nnz), "tokens_in_fit": int((df > 0).sum())}
        dv = d2_stage1[f"{direc}_vocab"]
        seam_vocab[direc] = {"pilot": sv, "d2": dv,
                             "equal": sv["size"] == dv["size"] and sv["sha256"] == dv["sha256"]
                             and sv["fit_profile_nnz"] == dv["fit_profile_nnz"] and sv["tokens_in_fit"] == dv["tokens_in_fit"]}
        P32 = P.astype(np.float32).tocsc()
        P32.sort_indices()
        np.savez(os.path.join(TMP, f"work_{direc}.npz"), P_data=P32.data, P_indices=P32.indices, P_indptr=P32.indptr,
                 P_shape=np.array(P32.shape), G_data=Gin.data, G_indices=Gin.indices, G_indptr=Gin.indptr,
                 G_shape=np.array(Gin.shape), idf=idf)
        meta[f"{direc}_vocab"] = sv
        meta[f"{direc}_seconds"] = round(time.time() - td, 1)
        log(f"{direc}: vocabulary {V} (D2 equal {seam_vocab[direc]['equal']}), reference {P32.shape} nnz {P32.nnz}, "
            f"py tokens {py_checks[direc]} ({time.time() - td:.0f}s)")
        del P, P32, Pf, Ci, A, Gin, tk_i
    meta["seam_s3"] = {"a_canon_check": canon_checks, "b_py_tokens": py_checks, "c_vocab_vs_d2": seam_vocab}
    ok = (all(v["mismatches"] == 0 for v in canon_checks.values())
          and all(v["mismatches"] == 0 for d in py_checks.values() for v in d.values())
          and all(v["equal"] for v in seam_vocab.values()))
    meta["seam_s3"]["passed"] = bool(ok)
    if not ok:
        raise SystemExit(f"seam s3 failed: {meta['seam_s3']}")
    meta["stage1_seconds"] = round(time.time() - T0, 1)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


# ================================================================ stage 1b: workers
_W = {}


def _init(direc):
    for k in THREAD_VARS:
        os.environ[k] = "1"
    with np.load(os.path.join(TMP, f"work_{direc}.npz"), allow_pickle=False) as zz:
        _W["P"] = sparse.csc_matrix((zz["P_data"], zz["P_indices"], zz["P_indptr"]), shape=tuple(zz["P_shape"]))
        _W["G"] = sparse.csr_matrix((zz["G_data"], zz["G_indices"], zz["G_indptr"]), shape=tuple(zz["G_shape"]))
        _W["idf"] = zz["idf"]
    with np.load(os.path.join(TMP, "common.npz"), allow_pickle=False) as zz:
        for k in zz.files:
            _W[k] = zz[k]
    mp_ = _W["mono_player"]
    order = np.argsort(mp_, kind="stable")
    bounds = np.searchsorted(mp_[order], np.arange(len(_W["pg_ptr"])))
    _W["q_by_player"] = [order[bounds[i]:bounds[i + 1]] for i in range(len(_W["pg_ptr"]) - 1)]
    _W["seam_set"] = set(_W["seam_q"].tolist())
    NREF = _W["P"].shape[0]
    removed05 = np.zeros(NREF, bool)
    removed05[N_FIT:] = ~_W["keep05"]
    _W["removed05"] = removed05
    _W["direc"] = direc


def _topk(X):
    ix = np.argpartition(-X, K, axis=1)[:, :K]
    vx = np.take_along_axis(X, ix, 1)
    o = np.argsort(-vx, axis=1, kind="stable")
    return np.take_along_axis(ix, o, 1).astype(np.int32), np.take_along_axis(vx, o, 1).astype(np.float32)


def _weights_dense(C, idc):
    return np.where(C > 0, 1.0 + np.log(np.maximum(C, 1.0)), 0.0) * idc


def work_player(pi):
    P, G, idf = _W["P"], _W["G"], _W["idf"]
    NREF = P.shape[0]
    games_all = _W["pg_rows"][_W["pg_ptr"][pi]:_W["pg_ptr"][pi + 1]]
    qs_all = _W["q_by_player"][pi]
    Gp = G[games_all]
    cols = np.unique(Gp.indices)
    Gs = Gp[:, cols].tocsr()
    idc = idf[cols]
    Cfull = np.asarray(Gs.sum(0)).ravel()
    loc = np.full(int(_W["n_inner_games"]), -1, np.int64)
    loc[games_all] = np.arange(len(games_all))
    own = N_FIT + pi
    nq = len(qs_all)
    out = {"pi": pi, "qs": qs_all, "idx": np.zeros((3, nq, K), np.int32), "cos": np.zeros((3, nq, K), np.float32),
           "own_cos": np.full(nq, np.nan, np.float32), "own_rank": np.full(nq, -1, np.int32), "seam": []}
    for c0 in range(0, nq, Q_CHUNK):
        qs = qs_all[c0:c0 + Q_CHUNK]
        gl = _W["mono_games"][qs]
        valid = gl >= 0
        rr = np.nonzero(valid)[0]
        lg = loc[gl[valid]]
        if (lg < 0).any():
            raise RuntimeError(f"player {pi}: a question game is not one of his inner games")
        Aq = sparse.csr_matrix((np.ones(len(rr)), (rr, lg)), shape=(len(qs), len(games_all)))
        Cq = np.asarray((Aq @ Gs).todense())
        Wq = _weights_dense(Cq, idc)
        Wq /= np.sqrt((Wq * Wq).sum(1))[:, None]
        Own = Cfull[None, :] - Cq
        if (Own < -1e-9).any():
            raise RuntimeError("negative own counts")
        Wo = _weights_dense(Own, idc)
        no = np.sqrt((Wo * Wo).sum(1))
        oc = np.where(no > 0, (Wq * Wo).sum(1) / np.where(no > 0, no, 1.0), np.nan)
        qmask = (Cq > 0).any(0)
        Psub = P[:, cols[qmask]]
        Wq32 = np.ascontiguousarray(Wq[:, qmask].astype(np.float32).T)
        Sm = np.ascontiguousarray(np.asarray(Psub @ Wq32).T)
        absent = ~(no > 0)
        Sm[:, own] = np.where(absent, MASKED, oc.astype(np.float32))
        sl = slice(c0, c0 + len(qs))
        out["own_cos"][sl] = np.where(absent, np.nan, oc).astype(np.float32)
        ocs = Sm[:, own].copy()
        out["own_rank"][sl] = np.where(absent, -1, (Sm > ocs[:, None]).sum(1))
        for j, sc in enumerate(SCEN):
            X = Sm
            if sc == "pi0":
                X = Sm.copy()
                X[:, own] = MASKED
            elif sc == "pi05":
                X = Sm.copy()
                X[:, _W["removed05"]] = MASKED
            out["idx"][j, sl], out["cos"][j, sl] = _topk(X)
        # seam s4 on the sampled questions of this chunk
        for r, q in enumerate(qs.tolist()):
            if q not in _W["seam_set"]:
                continue
            qg = gl[r][gl[r] >= 0]
            rest = np.setdiff1d(games_all, qg)
            cq = np.asarray(G[qg].sum(0)).ravel()
            wq = np.where(cq > 0, 1.0 + np.log(np.maximum(cq, 1.0)), 0.0) * idf
            wq /= np.sqrt((wq * wq).sum())
            if len(rest):
                co = np.asarray(G[rest].sum(0)).ravel()
                wo = np.where(co > 0, 1.0 + np.log(np.maximum(co, 1.0)), 0.0) * idf
                nn = np.sqrt((wo * wo).sum())
                direct = float(wq @ wo / nn) if nn > 0 else float("nan")
            else:
                direct = float("nan")
            mine = float(oc[r])
            d_own = 0.0 if (np.isnan(direct) and np.isnan(mine)) else abs(direct - mine)
            ref64 = np.asarray(Psub.astype(np.float64) @ Wq[r, qmask]).ravel()
            ref64[own] = Sm[r, own]
            d_ref = float(np.max(np.abs(ref64 - Sm[r].astype(np.float64))))
            out["seam"].append({"q": int(q), "own_abs_diff": float(d_own), "ref_max_abs_diff": d_ref,
                                "own_absent": bool(np.isnan(direct))})
    return out


def stage1b(s0, meta1):
    path = os.path.join(TMP, "cos.npz")
    seam_path = os.path.join(TMP, "cos_seam.json")
    if os.path.exists(path) and os.path.exists(seam_path):
        log("stage 1b: cached")
        with np.load(path, allow_pickle=False) as zz:
            return {k: zz[k] for k in zz.files}, json.load(open(seam_path, encoding="utf-8"))
    z = s0["z"]
    Q = len(z["mono_player"])
    NP = len(z["player_id"])
    cnt_q = np.bincount(z["mono_player"], minlength=NP)
    cost = cnt_q * (z["player_n_inner_games"] + 50)
    task_order = np.argsort(-cost, kind="stable").tolist()
    arr = {}
    seams = {}
    timing = {}
    ctx = mp.get_context("spawn")
    for direc in DIRS:
        td = time.time()
        res = {f"{direc}_{sc}_{w}": (np.full((Q, K), -1, np.int32) if w == "idx" else np.full((Q, K), np.nan, np.float32))
               for sc in SCEN for w in ("idx", "cos")}
        own_cos = np.full(Q, np.nan, np.float32)
        own_rank = np.full(Q, -1, np.int32)
        seen = np.zeros(Q, bool)
        sm = []
        with ctx.Pool(N_WORKERS, initializer=_init, initargs=(direc,)) as pool:
            for i, o in enumerate(pool.imap_unordered(work_player, task_order, chunksize=1)):
                qs = o["qs"]
                if seen[qs].any():
                    raise SystemExit("a question computed twice")
                seen[qs] = True
                for j, sc in enumerate(SCEN):
                    res[f"{direc}_{sc}_idx"][qs] = o["idx"][j]
                    res[f"{direc}_{sc}_cos"][qs] = o["cos"][j]
                own_cos[qs] = o["own_cos"]
                own_rank[qs] = o["own_rank"]
                sm.extend(o["seam"])
                if (i + 1) % 100 == 0:
                    log(f"{direc}: {i + 1}/{NP} players ({time.time() - td:.0f}s)")
        if not seen.all():
            raise SystemExit("some questions were not computed")
        arr.update(res)
        arr[f"{direc}_own_cos"] = own_cos
        arr[f"{direc}_own_rank"] = own_rank
        seams[direc] = {"questions": len(sm), "own_max_abs_diff": max(s["own_abs_diff"] for s in sm),
                        "ref_max_abs_diff": max(s["ref_max_abs_diff"] for s in sm),
                        "own_absent_in_sample": int(sum(s["own_absent"] for s in sm))}
        timing[direc] = round(time.time() - td, 1)
        log(f"{direc}: cosines done ({timing[direc]}s); seam s4 {seams[direc]}")
    ok = all(v["questions"] == 300 and v["own_max_abs_diff"] <= 1e-9 and v["ref_max_abs_diff"] <= 1e-5
             for v in seams.values())
    seam = {"s4": seams, "passed": bool(ok), "seconds": timing, "workers": N_WORKERS, "q_chunk": Q_CHUNK}
    if not ok:
        raise SystemExit(f"seam s4 failed: {seam}")
    np.savez(path, **arr)
    with open(seam_path, "w", encoding="utf-8") as f:
        json.dump(seam, f, indent=1)
    return arr, seam


# ================================================================ stage 2: thresholds, V3, fusion, report
class OffBoot:
    """Official-type statistic (mean over levels of the mean over k of the cell means) on PlayerBootstrap's draws."""

    def __init__(self, boot, player, kidx, n_players):
        self.boot = boot
        self.cell = player * 4 + kidx
        self.np = n_players
        self.cnt = np.bincount(self.cell, minlength=n_players * 4).reshape(n_players, 4).astype(np.float64)

    def stats(self, sc, draws=None):
        s = np.bincount(self.cell, weights=sc, minlength=self.np * 4).reshape(self.np, 4)
        per = []
        for d in (self.boot.draws if draws is None else draws):
            S = s[d].sum(1)
            C = self.cnt[d].sum(1)
            if (C == 0).any():
                raise SystemExit("a bootstrap draw has an empty (level, k) cell")
            per.append((S / C).mean(1))
        return np.mean(per, axis=0)


def official_type(sc, y, kidx, mask=None):
    if mask is None:
        mask = np.ones(len(sc), bool)
    per = []
    for i in range(L):
        cells = [sc[mask & (y == i) & (kidx == j)].mean() for j in range(4) if (mask & (y == i) & (kidx == j)).any()]
        if cells:
            per.append(np.mean(cells))
    return float(np.mean(per))


def fuse(P02, logP, E, w):
    """P_new = P_Q02 bit for bit where E is zero or w == 0, else softmax(log P_Q02 + w E)."""
    P = P02.copy()
    if w == 0:
        return P
    nz = (E != 0).any(1)
    if nz.any():
        P[nz] = rf.softmax(logP[nz] + w * E[nz])
    return P


def fit_w(logP, E, y, tr):
    """w in [0, 4] minimising the level-weighted mean log-loss over the questions tr."""
    wq = rf.level_weights(y[tr])
    yt = y[tr]
    lp = logP[tr]
    Et = E[tr]
    nz = (Et != 0).any(1)
    const = -float(np.sum(wq[~nz] * lp[~nz][np.arange((~nz).sum()), yt[~nz]]))
    ar = np.arange(nz.sum())
    lpn, En, ywn, wqn = lp[nz], Et[nz], yt[nz], wq[nz]
    base0 = -float(np.sum(wqn * lpn[ar, ywn]))

    def loss(w):
        if w == 0:
            return const + base0
        ls = rf.log_softmax(lpn + w * En)
        return const - float(np.sum(wqn * ls[ar, ywn]))
    r = optimize.minimize_scalar(loss, bounds=W_BOUNDS, method="bounded", options={"xatol": 1e-10})
    cands = [(float(r.fun), float(r.x), "brent"), (loss(W_BOUNDS[0]), W_BOUNDS[0], "lower"),
             (loss(W_BOUNDS[1]), W_BOUNDS[1], "upper")]
    best = min(cands, key=lambda c: (c[0], 0 if c[2] == "brent" else 1))
    return best[1], {"loss": best[0], "which": best[2], "brent_x": float(r.x), "brent_loss": float(r.fun),
                     "loss_at_0": cands[1][0], "loss_at_4": cands[2][0], "triggered_fit_questions": int(nz.sum()),
                     "fit_questions": int(tr.sum()),
                     "at_bound": bool(min(abs(best[1] - W_BOUNDS[0]), abs(best[1] - W_BOUNDS[1])) < 1e-6)}


def compare(s_new, s_ref, y, kidx, colour, boot, oboot):
    out = {"official_type": {}, "rank_balanced": {}}
    a, b = official_type(s_new, y, kidx), official_type(s_ref, y, kidx)
    d = oboot.stats(s_new) - oboot.stats(s_ref)
    out["official_type"] = {"pilot": a, "Q02": b, "delta": a - b,
                            "pboot_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                            "pboot_se": float(np.std(d, ddof=1))}
    a, b = U.rank_balanced(s_new, y), U.rank_balanced(s_ref, y)
    d = boot.stats(s_new) - boot.stats(s_ref)
    out["rank_balanced"] = {"pilot": a, "Q02": b, "delta": a - b,
                            "pboot_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                            "pboot_se": float(np.std(d, ddof=1))}
    out["question_mean_delta"] = float(s_new.mean() - s_ref.mean())
    out["exact_rate"] = float((s_new == 1.0).mean())
    out["exact_rate_Q02"] = float((s_ref == 1.0).mean())
    out["per_k"] = {}
    for j, k in enumerate(NS):
        mk = kidx == j
        out["per_k"][str(k)] = {"Q02": U.rank_balanced(s_ref, y, mk), "pilot": U.rank_balanced(s_new, y, mk),
                                "delta": U.rank_balanced(s_new, y, mk) - U.rank_balanced(s_ref, y, mk),
                                "questions": int(mk.sum())}
    out["per_level"] = {}
    for i, lv in enumerate(LEVELS):
        mk = y == i
        cq = [s_ref[mk & (kidx == j)].mean() for j in range(4) if (mk & (kidx == j)).any()]
        cn = [s_new[mk & (kidx == j)].mean() for j in range(4) if (mk & (kidx == j)).any()]
        out["per_level"][lv] = {"Q02_official_type": float(np.mean(cq)), "pilot_official_type": float(np.mean(cn)),
                                "delta_official_type": float(np.mean(cn) - np.mean(cq)),
                                "delta_question_mean": float(s_new[mk].mean() - s_ref[mk].mean()),
                                "questions": int(mk.sum())}
    out["by_colour"] = {c: {"delta_official_type": official_type(s_new, y, kidx, colour == c)
                            - official_type(s_ref, y, kidx, colour == c), "questions": int((colour == c).sum())}
                        for c in ("B", "W")}
    return out


def trigger_stats(trig, ch, own_col, y, kidx):
    out = {"rate": float(trig.mean()), "triggered": int(trig.sum()),
           "by_k": {str(k): float(trig[kidx == j].mean()) for j, k in enumerate(NS)}}
    if trig.any():
        out["top1_is_self"] = float((ch["top1"][trig] == own_col[trig]).mean())
        out["borrowed_exact"] = float((ch["lev1"][trig] == y[trig]).mean())
        out["borrowed_within_1"] = float((np.abs(ch["lev1"][trig] - y[trig]) <= 1).mean())
    out["no_threshold"] = {"top1_is_self": float((ch["top1"] == own_col).mean()),
                           "top1_level_exact": float((ch["lev1"] == y).mean()),
                           "canon_orientation_share": float(ch["use_canon"].mean()),
                           "S2_p50_p90_p99": [float(x) for x in np.percentile(ch["S2"], [50, 90, 99])],
                           "top1_is_self_by_k": {str(k): float((ch["top1"][kidx == j] == own_col[kidx == j]).mean())
                                                 for j, k in enumerate(NS)}}
    return out


def make_pools(player, kidx, n_players):
    qlist = {}
    for q in np.lexsort((np.arange(len(player)), kidx, player)):
        qlist.setdefault((int(player[q]), int(kidx[q])), []).append(int(q))
    has = np.zeros((n_players, 4), bool)
    for (p, j) in qlist:
        has[p, j] = True
    rng = np.random.default_rng(0)
    pools = np.zeros((N_POOLS, 4 * POOL_PER_K), np.int64)
    reused = np.zeros(N_POOLS, np.int64)
    for t in range(N_POOLS):
        order = rng.permutation(n_players)
        used = np.zeros(n_players, bool)
        slots = []
        for j in (3, 2, 1, 0):
            cand = [int(p) for p in order if not used[p] and has[p, j]][:POOL_PER_K]
            if len(cand) < POOL_PER_K:
                extra = [int(p) for p in order if used[p] and has[p, j] and p not in cand][:POOL_PER_K - len(cand)]
                reused[t] += len(extra)
                cand += extra
            used[cand] = True
            for p in cand:
                ql = qlist[(p, j)]
                slots.append(ql[int(rng.integers(0, len(ql)))])
        pools[t] = slots
    return pools, reused, {str(k): int(has[:, j].sum()) for j, k in enumerate(NS)}


def stage2(s0, meta1, arr):
    z = s0["z"]
    P02 = z["mono_P_Q02"]
    if not (P02 > 0).all():
        raise SystemExit("mono_P_Q02 has a zero probability")
    logP = np.log(P02)
    y, n, fold, player, colour = z["mono_y"], z["mono_n"], z["mono_fold"], z["mono_player"], z["mono_colour"]
    kidx = np.searchsorted(np.array(NS), n)
    if not np.array_equal(np.array(NS)[kidx], n):
        raise SystemExit("k outside 5/10/15/20")
    Q, NP = len(y), len(z["player_id"])
    with np.load(os.path.join(TMP, "common.npz"), allow_pickle=False) as zz:
        fit_level = zz["fit_level"]
        keep05 = zz["keep05"]
    ref_level = np.concatenate([fit_level, z["player_level"]])
    own_col = N_FIT + player
    ch = {sc: D2.choose(arr, sc, ref_level) for sc in SCEN}

    theta = np.zeros((5, 4))
    for f in range(5):
        for j, k in enumerate(NS):
            mk = (fold != f) & (kidx == j)
            theta[f, j] = np.percentile(ch["pi0"]["S2"][mk], 99)
    log(f"theta (fold x k): {np.round(theta, 4).tolist()}")
    v2 = {sc: D2.soft_vote(ch[sc], ref_level) for sc in SCEN}
    trig_f = {sc: [ch[sc]["S2"] > theta[f][kidx] for f in range(5)] for sc in SCEN}
    trig_own = {sc: ch[sc]["S2"] > theta[fold, kidx] for sc in SCEN}
    for sc in SCEN:
        for f in range(5):
            if not np.array_equal(trig_f[sc][f][fold == f], trig_own[sc][fold == f]):
                raise SystemExit("trigger bookkeeping")

    def E_of(sc, f):
        return v2[sc] * trig_f[sc][f][:, None]

    # w per fold on pi1 (and pi0 refit, record only)
    w_fit, w_rec = {}, {}
    for basis in ("pi1", "pi0"):
        ws, recs = [], []
        for f in range(5):
            w, rec = fit_w(logP, E_of(basis, f), y, fold != f)
            ws.append(w)
            recs.append(rec)
        w_fit[basis], w_rec[basis] = ws, recs
        log(f"w fitted on {basis}: {np.round(ws, 4).tolist()}")

    # seam s2
    s2 = {"all_zero_E_bitwise": [], "non_triggered_rows_bitwise": {}}
    for w in [1.0] + w_fit["pi1"]:
        s2["all_zero_E_bitwise"].append(bool(np.array_equal(fuse(P02, logP, np.zeros_like(P02), w), P02)))

    def predict(sc, ws):
        P = np.empty_like(P02)
        for f in range(5):
            ev = fold == f
            P[ev] = fuse(P02[ev], logP[ev], E_of(sc, f)[ev], ws[f])
        return P

    preds = {}
    for sc in SCEN:
        P = predict(sc, w_fit["pi1"])
        preds[sc] = P
        s2["non_triggered_rows_bitwise"][sc] = bool(np.array_equal(P[~trig_own[sc]], P02[~trig_own[sc]]))
        if not np.allclose(P.sum(1), 1.0) or np.isnan(P).any():
            raise SystemExit("fused probabilities do not sum to 1")
    preds["pi0_refit"] = predict("pi0", w_fit["pi0"])
    s2["non_triggered_rows_bitwise"]["pi0_refit"] = bool(np.array_equal(preds["pi0_refit"][~trig_own["pi0"]],
                                                                         P02[~trig_own["pi0"]]))
    s2["passed"] = bool(all(s2["all_zero_E_bitwise"]) and all(s2["non_triggered_rows_bitwise"].values()))
    log(f"seam s2: {s2}")
    if not s2["passed"]:
        raise SystemExit("seam s2 failed")

    sc_q02 = rf.official_score(rf.rule_b_np(P02), y)
    boot = rf.PlayerBootstrap(player, y, NP)
    oboot = OffBoot(boot, player, kidx, NP)
    ident = [np.nonzero(z["player_level"] == i)[0][None, :] for i in range(L)]
    oboot_self = float(abs(oboot.stats(sc_q02, ident)[0] - official_type(sc_q02, y, kidx)))
    if oboot_self > 1e-12:
        raise SystemExit(f"official-type bootstrap statistic differs from the point estimate: {oboot_self}")
    q02_metrics = {"rank_balanced": U.rank_balanced(sc_q02, y), "official_type": official_type(sc_q02, y, kidx),
                   "question_mean": float(sc_q02.mean())}
    log(f"Q02 on mono: rank-balanced {q02_metrics['rank_balanced']:.6f}, official-type {q02_metrics['official_type']:.6f}")

    scores, results = {}, {}
    for key, P in preds.items():
        s = rf.official_score(rf.rule_b_np(P), y)
        scores[key] = s
        results[key] = compare(s, sc_q02, y, kidx, colour, boot, oboot)
        r = results[key]["official_type"]
        log(f"{key}: official-type delta {r['delta']:+.5f} CI [{r['pboot_ci95'][0]:+.5f}, {r['pboot_ci95'][1]:+.5f}]; "
            f"rank-balanced {results[key]['rank_balanced']['delta']:+.5f}")

    trig = {}
    for sc in SCEN:
        trig[sc] = trigger_stats(trig_own[sc], ch[sc], own_col, y, kidx)
    own_diag = {}
    for direc in DIRS:
        rk = arr[f"{direc}_own_rank"]
        has = rk >= 0
        own_diag[direc] = {"questions_with_own_profile": int(has.sum()), "own_absent": int((~has).sum()),
                           "own_is_top1": float((rk[has] == 0).mean()), "own_in_top10": float((rk[has] < 10).mean()),
                           "own_is_top1_by_k": {str(k): float((rk[has & (kidx == j)] == 0).mean())
                                                for j, k in enumerate(NS)}}

    pools, reused, players_with_k = make_pools(player, kidx, NP)
    pool_rep = {"pools": N_POOLS, "questions_per_pool": int(pools.shape[1]), "players_with_k": players_with_k,
                "reused_player_slots_total": int(reused.sum()),
                "distinct_players_per_pool_min": int(min(len(np.unique(player[p])) for p in pools))}
    dq = {sc: scores[sc] - sc_q02 for sc in SCEN}
    for sc in SCEN:
        frac = trig_own[sc][pools].mean(1)
        on = frac >= ON_SHARE
        gain = np.where(on, dq[sc][pools].mean(1), 0.0)
        pool_rep[sc] = {"P_ON": float(on.mean()), "trigger_share_mean": float(frac.mean()),
                        "trigger_share_p1_p5_p50_p95_p99": [float(x) for x in np.percentile(frac, [1, 5, 50, 95, 99])],
                        "extra_record_only_pool_question_mean_delta_with_switch": {
                            "mean": float(gain.mean()), "p5_p50_p95": [float(x) for x in np.percentile(gain, [5, 50, 95])]}}
    log(f"pools: P(ON) pi1 {pool_rep['pi1']['P_ON']}, pi0 {pool_rep['pi0']['P_ON']}, pi05 {pool_rep['pi05']['P_ON']}")

    r1, r0 = results["pi1"]["official_type"], results["pi0"]["official_type"]
    crit = {
        "c1": {"pi1_official_type_delta": r1["delta"], "pboot_ci95": r1["pboot_ci95"],
               "pass": bool(r1["delta"] >= 0.03 and r1["pboot_ci95"][0] > 0)},
        "c2": {"pi0_official_type_delta_with_pi1_w": r0["delta"], "pboot_ci95": r0["pboot_ci95"],
               "pi0_trigger_rate": trig["pi0"]["rate"],
               "pass": bool(r0["delta"] >= -0.005 and trig["pi0"]["rate"] <= 0.02)},
        "c3": {"P_ON_pi1": pool_rep["pi1"]["P_ON"], "P_ON_pi0": pool_rep["pi0"]["P_ON"],
               "pass": bool(pool_rep["pi1"]["P_ON"] >= 0.9 and pool_rep["pi0"]["P_ON"] <= 0.01)}}
    crit["all_pass"] = bool(crit["c1"]["pass"] and crit["c2"]["pass"] and crit["c3"]["pass"])
    log(f"criteria: c1 {crit['c1']['pass']}, c2 {crit['c2']['pass']}, c3 {crit['c3']['pass']} -> all {crit['all_pass']}")

    d2 = json.load(open(D2_REPORT, encoding="utf-8"))
    d2cmp = {"note": "D2 (mixed-colour eval questions, main subset 464 players / 92,800 questions, own profile = "
                     "non-bank games only, theta per n bin, w inside Q02's generalised calibration); rank-balanced",
             "V3_pi1": [d2["results"]["V3|pi1"]["main"]["delta"], d2["results"]["V3|pi1"]["main"]["pboot_ci95"]],
             "V3_pi0_own_w": [d2["results"]["V3|pi0"]["main"]["delta"], d2["results"]["V3|pi0"]["main"]["pboot_ci95"]],
             "V3_pi05": [d2["results"]["V3|pi05"]["main"]["delta"], d2["results"]["V3|pi05"]["main"]["pboot_ci95"]],
             "V3_pi1_w_on_pi0_post_hoc": [
                 d2["post_hoc_extra_not_preregistered"]["results"]["V3|pi1_weights_on_pi0"]["main"]["delta"],
                 d2["post_hoc_extra_not_preregistered"]["results"]["V3|pi1_weights_on_pi0"]["main"]["pboot_ci95"]],
             "pi0_trigger_rate": d2["triggers"]["pi0"]["main"]["rate"],
             "pi1_trigger_rate": d2["triggers"]["pi1"]["main"]["rate"],
             "pi1_top1_is_self_triggered": d2["triggers"]["pi1"]["main"]["top1_is_self"],
             "pools_P_ON": {sc: d2["pools_400"]["main"][sc]["P_at_least_10pct_triggered"] for sc in SCEN},
             "w_V3_pi1_per_fold": d2["results"]["V3|pi1"]["w_extra_per_fold"]}

    small = {"theta_fold_by_k": theta, "w_pi1": np.array(w_fit["pi1"]), "w_pi0_refit": np.array(w_fit["pi0"]),
             "score_Q02": sc_q02, "pools": pools.astype(np.int32),
             "own_cos_raw": arr["raw_own_cos"], "own_cos_canon": arr["canon_own_cos"],
             "own_rank_raw": arr["raw_own_rank"], "own_rank_canon": arr["canon_own_rank"]}
    for key in preds:
        small[f"score_{key}"] = scores[key]
    for sc in SCEN:
        small[f"S2_{sc}"] = ch[sc]["S2"]
        small[f"use_canon_{sc}"] = ch[sc]["use_canon"]
        small[f"top1_{sc}"] = ch[sc]["top1"].astype(np.int32)
        small[f"trig_{sc}"] = trig_own[sc]
        rows = np.nonzero(trig_own[sc])[0]
        small[f"E_rows_{sc}"] = rows.astype(np.int32)
        small[f"E_{sc}"] = v2[sc][rows]
    np.savez_compressed(os.path.join(HERE, "pilot_arrays.npz"), **small)

    return {"theta_fold_by_k": theta.tolist(),
            "w": {"pi1_per_fold": w_fit["pi1"], "pi1_fit_records": w_rec["pi1"],
                  "pi0_refit_per_fold_record_only": w_fit["pi0"], "pi0_fit_records": w_rec["pi0"]},
            "Q02_mono": q02_metrics, "results": results, "triggers": trig, "own_profile_rank_pi1": own_diag,
            "pools_400": pool_rep, "criteria": crit, "d2_comparison": d2cmp,
            "seam_s2": s2, "official_type_bootstrap_self_check_abs_diff": oboot_self,
            "pi05_kept_inner_players": int(keep05.sum()), "questions": int(Q), "players": int(NP)}


def main():
    spec_sha = check_spec()
    no_player_modules()
    log(f"pilot_variants.json SHA-256 {spec_sha}")
    s0 = stage0()
    meta1 = stage1(s0)
    arr, seam1b = stage1b(s0, meta1)
    no_player_modules()
    rep2 = stage2(s0, meta1, arr)
    no_player_modules()
    spec_sha_end = check_spec()
    rep = {
        "what": "prereg05 PILOT of D2 (train-player lookup by opening, conditional switch) on the official question "
                "shape (single-colour inner pseudo-questions, k = 5/10/15/20). Training players only; design-stage "
                "simulation, NOT a validation result. No scoring, no score_ledger, data/local never opened, no "
                "existing file changed, no p02_player_* module imported, CPU only.",
        "created": p4.now_iso(), "minutes": round((time.time() - T0) / 60, 1),
        "spec": {"file": "results/prereg05/explore/pilot_D2/pilot_variants.json", "sha256_start": spec_sha,
                 "sha256_end": spec_sha_end, "unchanged": spec_sha == spec_sha_end == SPEC_SHA},
        "seams": {"s1": s0["seam"], "s2": rep2["seam_s2"], "s3": meta1["seam_s3"], "s4": seam1b["s4"],
                  "all_passed": bool(s0["seam"]["passed"] and rep2["seam_s2"]["passed"]
                                     and meta1["seam_s3"]["passed"] and seam1b["passed"])},
        "stage1": {k: v for k, v in meta1.items() if k != "seam_s3"},
        "stage1b": {k: v for k, v in seam1b.items() if k != "s4"},
        "inputs_sha256": dict(s0["inputs_sha256"], u1_oof_npz=sha256_file(U1_OOF), d2_probe_py=sha256_file(D2_PROBE),
                              d2_variants_json=sha256_file(D2_VARIANTS), d2_report_json=sha256_file(D2_REPORT)),
        "u1_oof_sha256_recorded": json.load(open(U1_REPRO, encoding="utf-8"))["output"]["sha256"],
        "compute": {"workers": N_WORKERS, "start_method": "spawn", "threads_env": {k: os.environ.get(k) for k in THREAD_VARS},
                    "cpu_only": True, "q_chunk": Q_CHUNK},
        "git_head": p3.git_head(), "environment": p3.environment(),
        "log": LOGLINES,
    }
    rep["inputs_sha256_u1_equal_recorded"] = rep["inputs_sha256"]["u1_oof_npz"] == rep["u1_oof_sha256_recorded"]
    rep.update({k: v for k, v in rep2.items() if k != "seam_s2"})
    with open(os.path.join(HERE, "pilot_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    log(f"wrote pilot_report.json; criteria all pass: {rep2['criteria']['all_pass']}")


if __name__ == "__main__":
    main()
