"""prereg05 exploration D2: "if the test players are training players, find the most similar training player by the
opening and borrow his level" (conditional switch). CPU probe on inner pseudo-questions only.

    .venv/Scripts/python.exe results/prereg05/explore/D2_train_player_lookup/d2_probe.py
    -> results/prereg05/explore/D2_train_player_lookup/d2_report.json

Every fixed choice is in d2_variants.json (written before this was run). Design-stage simulation, not a validation
result: nothing is scored, results/score_ledger.csv is not touched, data/local is never opened (u1_lib guard), no
frozen file is changed, no p02_player_* module is imported (asserted), BLAS threads 4, one process.

Stages (the first is cached in _cache/, deleted by hand after the report is written):
  0  seam: u1_oof.npz Q02 = 0.6101211982700248; the per-fold loop without an extra term reproduces eval_P_Q02 exactly
  1  opening tokens (numpy, from the design_scratch moves) -> IDF, profiles, question x profile cosines, top-10 per
     scenario and orientation
  2  S2, thresholds, E terms V1 / V2 / V3, 9 fusions + sensitivity, metrics, triggers, 400-question pools, criteria
"""

import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "4"

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "u1"))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

import u1_lib as U  # noqa: E402  (installs the data/local guard first; imports the frozen tools read-only)
import p02_moves as PM  # noqa: E402  (shared preprocessing: MOVE_RE / MAX_MOVES / SYM_TABLE / canon_sym)

p4, rf = U.p4, U.rf
T0 = time.time()
CACHE = os.path.join(HERE, "_cache")
DS = os.path.join(U.REPO, "data", "prereg04", "design_scratch")
MANIFEST = os.path.join(U.REPO, "results", "prereg04", "design", "scratch_manifest.json")
U1_REPRO = os.path.join(U.REPO, "results", "prereg05", "explore", "u1", "u1_repro.json")
LEVELS = U.LEVELS
L = 10
K = 10
TERMS = ["OP", "O4", "SWA", "SWB"]
Q02_RB = 0.6101211982700248
TOL = 1e-12
N_FIT_PLAYERS = 5767
IDF_N = 5768
BASE = 5 * 10 * 361 * 362
SCEN = ("pi1", "pi0", "pi05")
DIRS = ("raw", "canon")
N_BINS = ((5, 7), (8, 12), (13, 20))
NS = (5, 10, 15, 20)
TEMP_V2 = 0.02
EPS_V2 = 0.01
FIT_CAP = int(os.environ.get("D2_FIT_CAP", "0"))       # 0 = every fit game (registered); 100 only if over 55 minutes
CHUNK = 60000
MASKED = np.float32(-9.0)
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


# ================================================================ stage 0: seam
def fold_loop(Xf, Xe, mf, me, Ef, Ee, extra=None):
    """p04_rank_fuse.gen_oof_record's loop; extra(f) -> (E_fit, E_eval) appended as one more term for fold f."""
    P = np.zeros((len(Xe), L))
    folds = []
    for f in range(5):
        ev = Xe.fold == f
        tr = Xf.fold != f
        ef = [E[tr] for E in Ef]
        ee = [E[ev] for E in Ee]
        if extra is not None:
            xf, xe = extra(f)
            ef.append(xf[tr])
            ee.append(xe[ev])
        th, rec = p4.gen_fit_record(mf[tr], ef, Xf.n[tr], Xf.y[tr])
        z, _ = p4.gen_logits(th, me[ev], ee, Xe.n[ev])
        P[ev] = rf.softmax(z)
        folds.append({"fold": f, "theta": [float(x) for x in th], "loss": rec["loss"],
                      "w_terms": [float(x) for x in th[2:2 + len(ef)]]})
    return P, folds


def predict_with(folds, Xe, me, Ee, extra_eval):
    """Probabilities with recorded fold parameters; extra_eval(f) -> E_eval for fold f (sensitivity run)."""
    P = np.full((len(Xe), L), np.nan)
    for fr in folds:
        f = fr["fold"]
        ev = Xe.fold == f
        ee = [E[ev] for E in Ee] + [extra_eval(f)[ev]]
        z, _ = p4.gen_logits(np.asarray(fr["theta"], dtype=np.float64), me[ev], ee, Xe.n[ev])
        P[ev] = rf.softmax(z)
    return P


def stage0():
    z = U.load_saved()
    P02 = z["eval_P_Q02"]
    m, _ = rf.metrics(rf.rule_b_np(P02), z["eval_y"], z["eval_n"], z["eval_player"])
    seam_u1 = {"rank_balanced": m["rank_balanced"], "recorded": Q02_RB, "abs_diff": abs(m["rank_balanced"] - Q02_RB)}
    if seam_u1["abs_diff"] > TOL:
        raise SystemExit(f"u1 seam failed: {seam_u1}")
    log(f"seam u1_oof Q02 rank-balanced {m['rank_balanced']!r} (diff {seam_u1['abs_diff']:.3g})")
    d = U.load_inputs(verbose=False)
    inner = d["inner"]
    Xf, Xe, Sf, Se = inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"]
    for k, a in (("eval_y", Xe.y), ("eval_player", Xe.player), ("eval_fold", Xe.fold)):
        if not np.array_equal(z[k], a):
            raise SystemExit(f"{k} differs between u1_oof.npz and load_inputs")
    if not np.array_equal(z["eval_n"], Xe.n.astype(np.int64)):
        raise SystemExit("eval_n differs")
    eg = z["eval_games"]
    for q in range(len(Xe)):
        a = np.sort(Se.indices[Se.indptr[q]:Se.indptr[q + 1]])
        b = np.sort(eg[q][eg[q] >= 0])
        if not np.array_equal(a, b):
            raise SystemExit(f"eval question {q}: games differ between S_eval and u1 eval_games")
    lp = d["lp"]
    mf, me = rf.pooled(Sf, lp["C5"], "log"), rf.pooled(Se, lp["C5"], "log")
    Ef, Ee = p4.question_terms(Xf, Sf, lp, TERMS), p4.question_terms(Xe, Se, lp, TERMS)
    P0, folds0 = fold_loop(Xf, Xe, mf, me, Ef, Ee)
    seam_loop = bool(np.array_equal(P0, P02))
    log(f"seam: per-fold loop without an extra term equals u1 eval_P_Q02 exactly: {seam_loop}")
    if not seam_loop:
        raise SystemExit("per-fold loop does not reproduce eval_P_Q02")
    return {"z": z, "d": d, "inner": inner, "mf": mf, "me": me, "Ef": Ef, "Ee": Ee, "P02": P02,
            "seam": {"u1_oof_Q02": seam_u1, "fold_loop_equals_eval_P_Q02": seam_loop,
                     "q02_fold_w": [fr["w_terms"] for fr in folds0]}}


# ================================================================ stage 1: tokens, profiles, cosines
def sym_tables():
    """pt (= y*19 + x, the design_scratch move code) -> transformed pt, and -> comparison key x*19 + y (the order of
    the two-letter SGF strings), for the 8 symmetries of p02_moves.SYM_TABLE."""
    pt_map = np.zeros((8, 361), np.int32)
    key_map = np.zeros((8, 361), np.int32)
    for s in range(PM.N_SYMS):
        for x in range(19):
            for y in range(19):
                tx, ty = PM.to_xy(PM.SYM_TABLE[s][PM.from_xy(x, y)])
                pt_map[s, y * 19 + x] = ty * 19 + tx
                key_map[s, y * 19 + x] = tx * 19 + ty
    return pt_map, key_map


def first_moves(mv, off, rows):
    """[G, 60] point (-1 = none) and colour bit (0 B, 1 W) of the first MAX_MOVES moves (= parse_moves)."""
    starts = off[rows]
    lens = np.minimum(off[rows + 1] - starts, PM.MAX_MOVES)
    ar = np.arange(PM.MAX_MOVES)
    valid = ar[None, :] < lens[:, None]
    M = mv[np.where(valid, starts[:, None] + ar[None, :], 0)]
    pt = np.where(valid, (M & 511).astype(np.int32), -1)
    col = np.where(valid, (M >> 9).astype(np.int32), -1)
    if (pt >= 361).any() or ((col > 1) & valid).any():
        raise SystemExit("unexpected move code")
    return pt, col


def canon_index(pt, key_map):
    """p02_moves.canon_sym(mv, 10) vectorised: smallest tuple of the first 10 keys, ties to the smallest s."""
    G = len(pt)
    head = pt[:, :10]
    hv = head >= 0
    hs = np.where(hv, head, 0)
    best = np.zeros(G, np.int64)
    bestkey = np.where(hv, key_map[0][hs], -1)
    ar = np.arange(G)
    for s in range(1, 8):
        ks = np.where(hv, key_map[s][hs], -1)
        dlt = ks - bestkey
        nz = dlt != 0
        first = nz.argmax(1)
        better = nz.any(1) & (dlt[ar, first] < 0)
        best[better] = s
        bestkey[better] = ks[better]
    return best


def canonise(pt, s, pt_map):
    return np.where(pt >= 0, pt_map[s[:, None], np.maximum(pt, 0)], -1).astype(np.int32)


def check_canon(pt, col, s, pt_c, n_check, rng):
    """Compare with p02_moves.canon_sym / canon on a sample (string moves rebuilt from the codes)."""
    pick = rng.choice(len(pt), size=min(n_check, len(pt)), replace=False)
    bad = 0
    for g in pick:
        v = pt[g] >= 0
        mv = [("W" if c else "B", PM.from_xy(p % 19, p // 19)) for p, c in zip(pt[g][v], col[g][v])]
        sref = PM.canon_sym(mv, 10)
        cref = PM.canon(mv, 10)
        mine = [PM.from_xy(p % 19, p // 19) for p in pt_c[g][v]]
        if sref != s[g] and [p for _, p in PM.transform(mv, s[g])] != [p for _, p in PM.transform(mv, sref)]:
            bad += 1
        elif [p for _, p in cref] != mine:
            bad += 1
    return {"checked": int(len(pick)), "mismatches": bad}


def tokens(pt, col, tc):
    """[G, 98] int32 token codes (-1 = none): 49 no-colour tokens, then the same 49 with the target-colour prefix."""
    G = len(pt)
    gp = pt[:, :10].astype(np.int64)
    tm = (pt >= 0) & (col == tc[:, None])
    rk = np.cumsum(tm, 1)
    sel = tm & (rk <= 10)
    tp = np.full((G, 10), -1, np.int64)
    r, c = np.nonzero(sel)
    tp[r, rk[r, c] - 1] = pt[r, c]

    def code(kind, i, a, b=None):
        bb = 361 if b is None else b
        return ((kind * 10 + i) * 361 + a) * 362 + bb

    def sorted_set(x):
        xs = np.sort(x, 1)
        dup = np.zeros_like(xs, dtype=bool)
        dup[:, 1:] = xs[:, 1:] == xs[:, :-1]
        return xs, (xs >= 0) & ~dup

    cols = []
    for i in range(10):
        cols.append(np.where(tp[:, i] >= 0, code(0, i, tp[:, i]), -1))
    ts, tv = sorted_set(tp)
    for i in range(10):
        cols.append(np.where(tv[:, i], code(1, 0, ts[:, i]), -1))
    for i in range(10):
        cols.append(np.where(gp[:, i] >= 0, code(2, i, gp[:, i]), -1))
    gs, gv = sorted_set(gp)
    for i in range(10):
        cols.append(np.where(gv[:, i], code(3, 0, gs[:, i]), -1))
    for i in range(9):
        ok = (gp[:, i] >= 0) & (gp[:, i + 1] >= 0)
        cols.append(np.where(ok, code(4, 0, gp[:, i], gp[:, i + 1]), -1))
    base = np.stack(cols, 1)
    pref = (1 + tc.astype(np.int64))[:, None] * BASE
    colr = np.where(base >= 0, base + pref, -1)
    out = np.concatenate([base, colr], 1)
    assert out.max() < 3 * BASE
    return out.astype(np.int32)


def weights(cnt, idf_cols):
    return (1.0 + np.log(cnt)) * idf_cols


def row_normalise(M):
    M = M.tocsr()
    nrm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
    inv = np.where(nrm > 0, 1.0 / np.where(nrm > 0, nrm, 1.0), 0.0)
    return sparse.diags(inv) @ M, nrm


def stage1(s0):
    os.makedirs(CACHE, exist_ok=True)
    out_path = os.path.join(CACHE, "stage1.npz")
    meta_path = os.path.join(CACHE, "stage1.json")
    if os.path.exists(out_path) and os.path.exists(meta_path):
        log("stage 1: cached")
        with np.load(out_path, allow_pickle=False) as zz:
            arr = {k: zz[k] for k in zz.files}
        return arr, json.load(open(meta_path, encoding="utf-8"))
    meta = {}
    inner = s0["inner"]
    keys = inner["keys"]
    players = inner["players"]
    Xf, Xe, Sf, Se = inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"]

    # ---- input files: SHA-256 vs the PREREG_04 scratch manifest
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

    # ---- fit games
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
    if len(fit_ids) != N_FIT_PLAYERS:
        raise SystemExit(f"{len(fit_ids)} fit players, expected {N_FIT_PLAYERS}")
    fit_level = np.full(N_FIT_PLAYERS, -1, np.int64)
    fit_level[fit_pidx] = lab_f
    if not (fit_level[fit_pidx] == lab_f).all():
        raise SystemExit("a fit player has two levels")
    tc_f = (col_f == "W").astype(np.int64)
    order = np.argsort(fit_pidx, kind="stable")
    if FIT_CAP:
        sp = fit_pidx[order]
        first = np.searchsorted(sp, sp, side="left")
        order = order[(np.arange(len(order)) - first) < FIT_CAP]
    meta["fit_games_used"] = int(len(order))
    meta["fit_cap"] = FIT_CAP
    del pid_f, col_f

    # ---- inner games, in inner_keys order
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

    # ---- bank / non-bank
    pid_of = {p["player_id"]: i for i, p in enumerate(players)}
    NP = len(players)
    dump_row = {(r, g): i for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"]))}
    in_bank = np.zeros(keys["n"], bool)
    for p in players:
        for gid in p["bank"]:
            in_bank[dump_row[(p["rank"], gid)]] = True
    used = np.zeros(keys["n"], bool)
    used[Sf.indices] = True
    used[Se.indices] = True
    if (used & ~in_bank).any():
        raise SystemExit("a question game is not a bank game")
    inner_pl = np.array([pid_of.get(p, -1) for p in keys["player_id"]], dtype=np.int64)
    nonbank = (inner_pl >= 0) & ~used & ~in_bank
    if (nonbank & used).any():
        raise SystemExit("non-bank games intersect question games")
    n_nonbank = np.bincount(inner_pl[nonbank], minlength=NP)
    meta["nonbank"] = {"players": NP, "zero": int((n_nonbank == 0).sum()), "below5": int((n_nonbank < 5).sum()),
                       "at_least5": int((n_nonbank >= 5).sum()), "games": int(nonbank.sum()),
                       "median": float(np.median(n_nonbank)), "max": int(n_nonbank.max()),
                       "question_games": int(used.sum()), "bank_games": int(in_bank.sum()),
                       "disjoint_from_question_games": True}
    log(f"inner players {NP}: non-bank games 0 for {meta['nonbank']['zero']}, <5 for {meta['nonbank']['below5']}, "
        f">=5 for {meta['nonbank']['at_least5']}")

    pt_map, key_map = sym_tables()
    rng = np.random.default_rng(12345)

    # ---- questions (fit then eval), grouped by player
    def groups(S):
        return [S.indices[S.indptr[q]:S.indptr[q + 1]] for q in range(S.shape[0])]
    q_groups = groups(Sf) + groups(Se)
    q_player = np.concatenate([Xf.player, Xe.player])
    QT = len(q_groups)
    q_by_player = [np.nonzero(q_player == pi)[0] for pi in range(NP)]

    NREF = N_FIT_PLAYERS + NP
    keep05 = np.array([hashlib.sha256(("c10-p05-d2|" + p["player_id"]).encode()).digest()[0] < 128
                       for p in players])
    meta["pi05_kept_inner_players"] = int(keep05.sum())

    arrays = {"q_player": q_player, "n_nonbank": n_nonbank, "fit_level": fit_level, "keep05": keep05}
    canon_checks = {}
    for direc in DIRS:
        td = time.time()
        # fit (player, token) counts, chunks in player order
        acc_k, acc_c = [], []
        for c0 in range(0, len(order), CHUNK):
            rows = order[c0:c0 + CHUNK]
            pt, col = first_moves(mv_f, off_f, rows)
            if direc == "canon":
                s = canon_index(pt, key_map)
                ptc = canonise(pt, s, pt_map)
                if c0 == 0:
                    canon_checks["fit"] = check_canon(pt, col, s, ptc, 20000, rng)
                pt = ptc
            tk = tokens(pt, col, tc_f[rows])
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
        # inner game tokens
        pt, col = first_moves(mv_i, off_i, perm)
        if direc == "canon":
            s = canon_index(pt, key_map)
            ptc = canonise(pt, s, pt_map)
            canon_checks["inner"] = check_canon(pt, col, s, ptc, 20000, rng)
            pt = ptc
        tk_i = tokens(pt, col, tc_i)
        vocab = np.unique(np.concatenate([fp_code, tk_i[tk_i >= 0]]))
        V = len(vocab)
        fp_tok = np.searchsorted(vocab, fp_code)
        df = np.bincount(fp_tok, minlength=V)
        idf = np.log(IDF_N / (1.0 + df)) + 1.0
        # fit profiles
        Pf = sparse.csr_matrix((weights(cnt_f.astype(np.float64), idf[fp_tok]), (fp_player, fp_tok)),
                               shape=(N_FIT_PLAYERS, V))
        del fp_player, fp_code, fp_tok, cnt_f
        # inner game count matrix
        gi, gj = np.nonzero(tk_i >= 0)
        Gin = sparse.csr_matrix((np.ones(len(gi)), (gi, np.searchsorted(vocab, tk_i[gi, gj]))), shape=(keys["n"], V))
        Gin.sum_duplicates()
        # inner profiles from non-bank games
        nb_rows = np.nonzero(nonbank)[0]
        A = sparse.csr_matrix((np.ones(len(nb_rows)), (inner_pl[nb_rows], nb_rows)), shape=(NP, keys["n"]))
        Ci = (A @ Gin).tocsr()
        Ci.sum_duplicates()
        Ci.data = weights(Ci.data, idf[Ci.indices])
        P = sparse.vstack([Pf, Ci]).tocsr()
        P, nrm = row_normalise(P)
        P = P.astype(np.float32).tocsc()
        has_profile = nrm > 0
        if not has_profile[:N_FIT_PLAYERS].all():
            raise SystemExit("a fit player has an empty profile")
        if not np.array_equal(has_profile[N_FIT_PLAYERS:], n_nonbank > 0):
            raise SystemExit("inner profile presence differs from the non-bank count")
        meta[f"{direc}_vocab"] = {"size": int(V), "sha256": hashlib.sha256(vocab.astype(np.int32).tobytes()).hexdigest(),
                                  "fit_profile_nnz": int(Pf.nnz), "inner_profile_nnz": int(Ci.nnz),
                                  "tokens_in_fit": int((df > 0).sum())}
        del Pf, Ci, A
        log(f"{direc}: vocabulary {V}, profiles {P.shape} nnz {P.nnz} ({time.time() - td:.0f}s)")

        # cosines per bank player
        top = {sc: (np.full((QT, K), -1, np.int32), np.full((QT, K), np.nan, np.float32)) for sc in SCEN}
        own_cos = np.full(QT, np.nan, np.float32)
        own_rank = np.full(QT, -1, np.int32)
        invalid = ~has_profile
        removed05 = np.zeros(NREF, bool)
        removed05[N_FIT_PLAYERS:] = ~keep05
        for pi in range(NP):
            qs = q_by_player[pi]
            gl = [q_groups[q] for q in qs]
            games = np.unique(np.concatenate(gl))
            loc = {g: j for j, g in enumerate(games.tolist())}
            sub = Gin[games]
            cols = np.unique(sub.indices)
            Bd = sub[:, cols].toarray()
            Aq = np.zeros((len(qs), len(games)))
            for r, g in enumerate(gl):
                Aq[r, [loc[x] for x in g.tolist()]] = 1.0
            C = Aq @ Bd
            W = np.where(C > 0, 1.0 + np.log(np.maximum(C, 1.0)), 0.0) * idf[cols]
            W /= np.sqrt((W * W).sum(1))[:, None]
            Pd = P[:, cols].toarray()
            Sm = W.astype(np.float32) @ Pd.T
            Sm[:, invalid] = MASKED
            own = N_FIT_PLAYERS + pi
            if has_profile[own]:
                oc = Sm[:, own].copy()
                own_cos[qs] = oc
                own_rank[qs] = (Sm > oc[:, None]).sum(1)
            for sc in SCEN:
                X = Sm
                if sc == "pi0":
                    X = Sm.copy()
                    X[:, own] = MASKED
                elif sc == "pi05":
                    X = Sm.copy()
                    X[:, removed05] = MASKED
                ix = np.argpartition(-X, K, axis=1)[:, :K]
                vx = np.take_along_axis(X, ix, 1)
                o = np.argsort(-vx, axis=1, kind="stable")
                top[sc][0][qs] = np.take_along_axis(ix, o, 1)
                top[sc][1][qs] = np.take_along_axis(vx, o, 1)
            if pi % 100 == 0:
                log(f"{direc}: player {pi}/{NP} ({time.time() - td:.0f}s)")
        for sc in SCEN:
            arrays[f"{direc}_{sc}_idx"], arrays[f"{direc}_{sc}_cos"] = top[sc]
        arrays[f"{direc}_own_cos"] = own_cos
        arrays[f"{direc}_own_rank"] = own_rank
        arrays[f"{direc}_has_profile"] = has_profile
        meta[f"{direc}_seconds"] = round(time.time() - td, 1)
        del P, Gin, tk_i
        log(f"{direc}: cosines done ({time.time() - td:.0f}s)")
    meta["canon_check"] = canon_checks
    if any(v["mismatches"] for v in canon_checks.values()):
        raise SystemExit(f"numpy canon differs from p02_moves.canon: {canon_checks}")
    meta["stage1_seconds"] = round(time.time() - T0, 1)
    np.savez(out_path, **arrays)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return arrays, meta


# ================================================================ stage 2: thresholds, variants, fusions, report
def n_bin(n):
    b = np.full(len(n), -1)
    for j, (lo, hi) in enumerate(N_BINS):
        b[(n >= lo) & (n <= hi)] = j
    return b


def choose(arr, sc, ref_level):
    """S2 per orientation, the larger one used (tie -> raw); top-1 and top-10 of that orientation."""
    s2 = {}
    for direc in DIRS:
        c = arr[f"{direc}_{sc}_cos"].astype(np.float64)
        s2[direc] = c[:, 0] - c[:, 1]
    use_c = s2["canon"] > s2["raw"]
    S2 = np.where(use_c, s2["canon"], s2["raw"])
    idx = np.where(use_c[:, None], arr[f"canon_{sc}_idx"], arr[f"raw_{sc}_idx"]).astype(np.int64)
    cos = np.where(use_c[:, None], arr[f"canon_{sc}_cos"], arr[f"raw_{sc}_cos"]).astype(np.float64)
    if (cos[:, :K] <= MASKED + 1).any():
        raise SystemExit("fewer than 10 reference profiles for some question")
    return {"S2": S2, "use_canon": use_c, "idx": idx, "cos": cos, "top1": idx[:, 0], "lev1": ref_level[idx[:, 0]],
            "s2_raw": s2["raw"], "s2_canon": s2["canon"]}


def soft_vote(ch, ref_level):
    c = ch["cos"]
    w = np.exp((c - c[:, :1]) / TEMP_V2)
    w /= w.sum(1, keepdims=True)
    lev = ref_level[ch["idx"]]
    dist = np.zeros((len(c), L))
    ar = np.arange(len(c))
    for j in range(K):
        dist[ar, lev[:, j]] += w[:, j]
    return np.log(dist + EPS_V2)


def rb_mask(sc, y, mask):
    return U.rank_balanced(sc, y, mask)


def compare(P, Pref, Xe, mask, boot):
    y, n = Xe.y, Xe.n
    pred, predr = rf.rule_b_np(P), rf.rule_b_np(Pref)
    s, sr = rf.official_score(pred, y), rf.official_score(predr, y)
    rb, rbr = rb_mask(s, y, mask), rb_mask(sr, y, mask)
    st, str_ = boot.stats(s[mask]), boot.stats(sr[mask])
    dd = st - str_
    out = {"questions": int(mask.sum()), "players": int(len(np.unique(Xe.player[mask]))),
           "rank_balanced": rb, "rank_balanced_Q02": rbr, "delta": rb - rbr,
           "pboot_ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
           "pboot_se": float(np.std(dd, ddof=1)),
           "question_mean_delta": float(s[mask].mean() - sr[mask].mean()),
           "exact_rate": float((pred[mask] == y[mask]).mean()), "exact_rate_Q02": float((predr[mask] == y[mask]).mean()),
           "per_level": {}, "per_n_bin": {}, "per_n": {}}
    for i, lv in enumerate(LEVELS):
        mk = mask & (y == i)
        if mk.any():
            out["per_level"][lv] = {"Q02": float(sr[mk].mean()), "variant": float(s[mk].mean()),
                                    "delta": float(s[mk].mean() - sr[mk].mean()), "questions": int(mk.sum())}
    for lo, hi in N_BINS:
        mk = mask & (n >= lo) & (n <= hi)
        out["per_n_bin"][f"{lo}-{hi}"] = {"Q02": rb_mask(sr, y, mk), "variant": rb_mask(s, y, mk),
                                          "delta": rb_mask(s, y, mk) - rb_mask(sr, y, mk), "questions": int(mk.sum())}
    for k in NS:
        mk = mask & (n == k)
        out["per_n"][str(k)] = {"Q02": rb_mask(sr, y, mk), "variant": rb_mask(s, y, mk),
                                "delta": rb_mask(s, y, mk) - rb_mask(sr, y, mk), "questions": int(mk.sum())}
    return out


def trigger_stats(trig, ch, own_col, y, n, mask):
    out = {"rate": float(trig[mask].mean()), "by_n_bin": {}, "by_n": {}}
    for lo, hi in N_BINS:
        mk = mask & (n >= lo) & (n <= hi)
        out["by_n_bin"][f"{lo}-{hi}"] = float(trig[mk].mean())
    for k in NS:
        out["by_n"][str(k)] = float(trig[mask & (n == k)].mean())
    t = trig & mask
    out["triggered"] = int(t.sum())
    if t.any():
        out["top1_is_self"] = float((ch["top1"][t] == own_col[t]).mean())
        out["borrowed_exact"] = float((ch["lev1"][t] == y[t]).mean())
        out["borrowed_within_1"] = float((np.abs(ch["lev1"][t] - y[t]) <= 1).mean())
    return out


def stage2(s0, arr, meta1):
    inner = s0["inner"]
    players = inner["players"]
    Xf, Xe = inner["X_fit"], inner["X_eval"]
    QF = len(Xf)
    NP = len(players)
    ref_level = np.concatenate([arr["fit_level"], np.array([LEVELS.index(p["rank"]) for p in players])])
    n_nonbank = arr["n_nonbank"]
    n_all = np.concatenate([Xf.n, Xe.n]).astype(np.int64)
    fold_all = np.concatenate([Xf.fold, Xe.fold])
    pl_all = np.concatenate([Xf.player, Xe.player])
    if not np.array_equal(pl_all, arr["q_player"]):
        raise SystemExit("question order differs from stage 1")
    y_all = np.concatenate([Xf.y, Xe.y])
    own_col = N_FIT_PLAYERS + pl_all
    nb_all = n_bin(n_all)
    ch = {sc: choose(arr, sc, ref_level) for sc in SCEN}

    # thresholds: pi0 S2 of the fit questions of the other 4 folds, per n bin
    theta = np.zeros((5, len(N_BINS)))
    for f in range(5):
        for j in range(len(N_BINS)):
            mk = (np.arange(len(n_all)) < QF) & (fold_all != f) & (nb_all == j)
            theta[f, j] = np.percentile(ch["pi0"]["S2"][mk], 99)
    th_q = {f: theta[f][nb_all] for f in range(5)}           # theta of fold f applied to every question
    log(f"thresholds (fold x n bin): {np.round(theta, 4).tolist()}")

    v2 = {sc: soft_vote(ch[sc], ref_level) for sc in SCEN}

    def E_for(variant, sc, f):
        trig = ch[sc]["S2"] > th_q[f]
        if variant == "V1":
            E = np.zeros((len(n_all), L))
            E[np.arange(len(n_all)), ch[sc]["lev1"]] = 1.0
            return E * trig[:, None]
        if variant == "V2":
            return v2[sc]
        if variant == "V3":
            return v2[sc] * trig[:, None]
        raise ValueError(variant)

    # the trigger of each question uses its own fold's theta
    trig_own = {sc: ch[sc]["S2"] > theta[fold_all, nb_all] for sc in SCEN}

    mf, me, Ef, Ee, P02 = s0["mf"], s0["me"], s0["Ef"], s0["Ee"], s0["P02"]
    main_mask = n_nonbank[Xe.player] >= 5
    all_mask = np.ones(len(Xe), bool)
    boots = {"main": rf.PlayerBootstrap(Xe.player[main_mask], Xe.y[main_mask], NP),
             "all": rf.PlayerBootstrap(Xe.player, Xe.y, NP)}
    masks = {"main": main_mask, "all": all_mask}
    ev_sl = slice(QF, None)
    fit_sl = slice(0, QF)

    results = {}
    fold_records = {}
    for variant in ("V1", "V2", "V3"):
        for sc in SCEN:
            tf = time.time()
            P, folds = fold_loop(Xf, Xe, mf, me, Ef, Ee,
                                 extra=lambda f, v=variant, s=sc: (E_for(v, s, f)[fit_sl], E_for(v, s, f)[ev_sl]))
            fold_records[(variant, sc)] = folds
            r = {m: compare(P, P02, Xe, masks[m], boots[m]) for m in masks}
            r["w_extra_per_fold"] = [fr["w_terms"][-1] for fr in folds]
            r["seconds"] = round(time.time() - tf, 1)
            results[f"{variant}|{sc}"] = r
            log(f"{variant} {sc}: main delta {r['main']['delta']:+.4f} CI [{r['main']['pboot_ci95'][0]:+.4f}, "
                f"{r['main']['pboot_ci95'][1]:+.4f}]; all {r['all']['delta']:+.4f}; w {np.round(r['w_extra_per_fold'], 3).tolist()}")
    sens = {}
    for variant in ("V1", "V2", "V3"):
        P = predict_with(fold_records[(variant, "pi0")], Xe, me, Ee, lambda f, v=variant: E_for(v, "pi1", f)[ev_sl])
        sens[variant] = {m: compare(P, P02, Xe, masks[m], boots[m]) for m in masks}
        log(f"sensitivity {variant}: pi0 weights on pi1 E: main delta {sens[variant]['main']['delta']:+.4f}")
    # POST-HOC EXTRA (added after the first run; not pre-registered, not part of the criteria): the switch turned on
    # with pi1-fitted weights while the real overlap is none (pi0) or partial (pi05); fixed rules, nothing selected
    post = {}
    for variant in ("V1", "V2", "V3"):
        for target in ("pi0", "pi05"):
            P = predict_with(fold_records[(variant, "pi1")], Xe, me, Ee,
                             lambda f, v=variant, t=target: E_for(v, t, f)[ev_sl])
            post[f"{variant}|pi1_weights_on_{target}"] = {m: compare(P, P02, Xe, masks[m], boots[m]) for m in masks}
            log(f"post-hoc {variant}: pi1 weights on {target} E: main delta "
                f"{post[f'{variant}|pi1_weights_on_{target}']['main']['delta']:+.4f}")
    for key, folds in fold_records.items():
        results[f"{key[0]}|{key[1]}"]["fold_theta"] = [fr["theta"] for fr in folds]

    # triggers and lookup diagnostics (eval questions)
    trig_rep, diag = {}, {}
    ye, ne = Xe.y, Xe.n.astype(np.int64)
    for sc in SCEN:
        che = {k: (v[ev_sl] if isinstance(v, np.ndarray) and v.shape[0] == len(n_all) else v) for k, v in ch[sc].items()}
        trig_rep[sc] = {m: trigger_stats(trig_own[sc][ev_sl], che, own_col[ev_sl], ye, ne, masks[m]) for m in masks}
        trig_rep[sc]["fit_questions_rate"] = float(trig_own[sc][fit_sl].mean())
        d = {}
        for m in masks:
            mk = masks[m]
            d[m] = {"top1_is_self_no_threshold": float((che["top1"][mk] == own_col[ev_sl][mk]).mean()),
                    "top1_level_exact_no_threshold": float((che["lev1"][mk] == ye[mk]).mean()),
                    "canon_orientation_share": float(che["use_canon"][mk].mean()),
                    "S2_quantiles": [float(x) for x in np.percentile(che["S2"][mk], [50, 90, 99])],
                    "top1_is_self_by_n": {str(k): float((che["top1"][mk & (ne == k)] == own_col[ev_sl][mk & (ne == k)]).mean())
                                          for k in NS}}
        diag[sc] = d
    # own-profile rank under pi1 (per orientation, eval questions of players with a profile)
    own_diag = {}
    for direc in DIRS:
        rk = arr[f"{direc}_own_rank"][ev_sl]
        has = rk >= 0
        own_diag[direc] = {"questions_with_own_profile": int(has.sum()),
                           "own_is_top1": float((rk[has] == 0).mean()),
                           "own_in_top10": float((rk[has] < 10).mean()),
                           "own_is_top1_main": float((rk[has & main_mask] == 0).mean())}

    # 400-question pools (same draws for every scenario)
    pools = {}
    for m in masks:
        rng = np.random.default_rng(0)
        cand = {k: np.nonzero(masks[m] & (ne == k))[0] for k in NS}
        draws = np.stack([np.concatenate([rng.choice(cand[k], 100, replace=False) for k in NS]) for _ in range(1000)])
        pools[m] = {"candidates_per_n": {str(k): int(len(cand[k])) for k in NS}}
        for sc in SCEN:
            frac = trig_own[sc][ev_sl][draws].mean(1)
            pools[m][sc] = {"P_at_least_10pct_triggered": float((frac >= 0.10).mean()),
                            "mean_trigger_fraction": float(frac.mean()),
                            "trigger_fraction_p1_p50_p99": [float(x) for x in np.percentile(frac, [1, 50, 99])]}

    # success criteria (main subset)
    crit = {}
    for variant in ("V1", "V3"):
        r1, r0 = results[f"{variant}|pi1"]["main"], results[f"{variant}|pi0"]["main"]
        c1 = bool(r1["delta"] >= 0.03 and r1["pboot_ci95"][0] > 0)
        c2 = bool(r0["delta"] >= -0.001 and trig_rep["pi0"]["main"]["rate"] <= 0.02)
        c3 = bool(pools["main"]["pi1"]["P_at_least_10pct_triggered"] >= 0.9
                  and pools["main"]["pi0"]["P_at_least_10pct_triggered"] <= 0.01)
        crit[variant] = {"c1_pi1_delta_ge_0.03_and_lower_gt_0": c1, "c2_pi0_delta_ge_-0.001_and_trigger_le_2pct": c2,
                         "c3_pools": c3, "all_three": bool(c1 and c2 and c3),
                         "pi1_delta": r1["delta"], "pi1_ci95": r1["pboot_ci95"], "pi0_delta": r0["delta"],
                         "pi0_trigger_rate": trig_rep["pi0"]["main"]["rate"],
                         "pool_pi1": pools["main"]["pi1"]["P_at_least_10pct_triggered"],
                         "pool_pi0": pools["main"]["pi0"]["P_at_least_10pct_triggered"]}
    success = any(v["all_three"] for v in crit.values())
    return {"thresholds_fold_by_nbin": theta.tolist(), "results": results, "sensitivity_pi0_weights_on_pi1": sens,
            "post_hoc_extra_not_preregistered": {
                "what": "added after the first run, not part of the criteria: pi1-fitted fold parameters applied to the "
                        "pi0 / pi05 E terms (= the switch is on but the real overlap is none / half)",
                "results": post},
            "triggers": trig_rep, "lookup_diagnostics": diag, "own_profile_rank_pi1": own_diag, "pools_400": pools,
            "criteria": crit, "meets_success_signal": success,
            "main_subset": {"players": int((n_nonbank >= 5).sum()), "eval_questions": int(main_mask.sum())}}


def main():
    no_player_modules()
    s0 = stage0()
    arr, meta1 = stage1(s0)
    no_player_modules()
    rep2 = stage2(s0, arr, meta1)
    no_player_modules()
    rep = {
        "what": "prereg05 exploration D2 (train-player lookup by opening, conditional switch). Inner pseudo-questions "
                "only; design-stage simulation, NOT a validation result. No scoring, no score_ledger, data/local never "
                "opened, no frozen file changed, no p02_player_* module imported, CPU (BLAS threads 4).",
        "created": U.p4.now_iso(), "minutes": round((time.time() - T0) / 60, 1),
        "runs": "first full run 2026-10-01: 9.8 minutes (stage 1 computed, 422.8 s); the post-hoc extra block was added "
                "afterwards and the script re-run with stage 1 read from _cache; every pre-registered number is "
                "compared with the first run's report (identical_to_first_run)",
        "variants_file": "results/prereg05/explore/D2_train_player_lookup/d2_variants.json",
        "variants_sha256": sha256_file(os.path.join(HERE, "d2_variants.json")),
        "seam": s0["seam"], "stage1": meta1,
        "inputs_sha256": dict(s0["inner"]["inputs"], **s0["d"]["sha"]),
        "u1_oof_sha256": json.load(open(U1_REPRO, encoding="utf-8"))["output"]["sha256"],
        "git_head": U.p3.git_head(), "environment": U.p3.environment(),
        "log": LOGLINES,
    }
    rep.update(rep2)
    first = os.path.join(CACHE, "d2_report_run1.json")
    if os.path.exists(first):
        r1 = json.load(open(first, encoding="utf-8"))
        rep2j = json.loads(json.dumps(rep2))
        same = {}
        for k in ("thresholds_fold_by_nbin", "sensitivity_pi0_weights_on_pi1", "triggers", "lookup_diagnostics",
                  "own_profile_rank_pi1", "pools_400", "criteria", "meets_success_signal", "main_subset"):
            same[k] = r1.get(k) == rep2j.get(k)
        same["results"] = all({kk: vv for kk, vv in rep2j["results"][key].items() if kk not in ("seconds", "fold_theta")}
                              == {kk: vv for kk, vv in r1["results"][key].items() if kk != "seconds"}
                              for key in r1["results"])
        same["stage1"] = r1.get("stage1") == json.loads(json.dumps(meta1))
        rep["identical_to_first_run"] = same
        log(f"identical to the first run: {same}")
    with open(os.path.join(HERE, "d2_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    log(f"wrote d2_report.json; meets success signal: {rep2['meets_success_signal']}")


if __name__ == "__main__":
    main()
