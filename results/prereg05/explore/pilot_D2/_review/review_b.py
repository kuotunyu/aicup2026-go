"""Independent review of the D2 PILOT, part B: recompute the whole similarity / threshold / fusion pipeline with
separately written code (pure-Python tokeniser from the d2_variants.json text, own profile from the COMPLEMENT of the
question's games instead of subtraction, float64 reference cosines, full stable argsort for the top-10, own w fit),
then compare with pilot_arrays.npz / pilot_report.json.

Training players only (fit + inner); no validation data; nothing scored; data/local guarded by u1_lib (installed in
every spawned worker because this module imports u1_lib at top level). Writes only into this _review folder
(_tmp deleted at the end). CPU, threads 1, 4 workers.
"""
import os
import sys

sys.dont_write_bytecode = True
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _k in THREADS:
    os.environ[_k] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
EXPLORE = os.path.dirname(PILOT)
sys.path.insert(0, os.path.join(EXPLORE, "u1"))
import u1_lib as U  # noqa: E402  (guard first)

import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import optimize, sparse  # noqa: E402

import p02_moves as PM  # noqa: E402

REPO = U.REPO
DS = os.path.join(REPO, "data", "prereg04", "design_scratch")
KEYS = os.path.join(REPO, "results", "prereg02", "rank", "dumps", "inner_keys.npz")
TMP = os.path.join(HERE, "_tmp")
N_FIT = 5767
MASK = -9.0
NS = np.array([5, 10, 15, 20])
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------------ pure-Python tokeniser (from d2_variants.json)
def _sym_tables():
    sp = [[0] * 361 for _ in range(8)]
    ky = [[0] * 361 for _ in range(8)]
    for s in range(8):
        t = PM.SYM_TABLE[s]
        for x in range(19):
            for yy in range(19):
                tx, ty = PM.to_xy(t[PM.from_xy(x, yy)])
                sp[s][yy * 19 + x] = ty * 19 + tx
                ky[s][yy * 19 + x] = tx * 19 + ty      # SGF string order: column letter, then row letter
    return sp, ky


SP, KY = _sym_tables()


def canon_pts(pts):
    head = pts[:10]
    best_s, best = 0, None
    for s in range(8):
        k = tuple(KY[s][p] for p in head)
        if best is None or k < best:
            best_s, best = s, k
    m = SP[best_s]
    return [m[p] for p in pts]


def game_tokens(pts, cols, tc):
    tp = [p for p, c in zip(pts, cols) if c == tc][:10]
    gp = pts[:10]
    out = []
    for pref in (0, 1 + tc):
        for kind, i, a, b in ([(0, i, p, 361) for i, p in enumerate(tp)]
                              + [(1, 0, p, 361) for p in set(tp)]
                              + [(2, i, p, 361) for i, p in enumerate(gp)]
                              + [(3, 0, p, 361) for p in set(gp)]
                              + [(4, 0, gp[i], gp[i + 1]) for i in range(len(gp) - 1)]):
            out.append(((((pref * 5 + kind) * 10 + i) * 361 + a) * 362 + b))
    return out


_T = {}


def tok_init():
    for k in THREADS:
        os.environ[k] = "1"
    with np.load(os.path.join(TMP, "tok_meta.npz"), allow_pickle=False) as z:
        for k in z.files:
            _T[k] = z[k]
    _T["mv_fit"] = np.load(os.path.join(DS, "moves_fit_moves.npy"), mmap_mode="r")
    _T["mv_inner"] = np.load(os.path.join(DS, "moves_inner_moves.npy"), mmap_mode="r")


def tok_work(task):
    which, lo, hi = task
    rows = _T[f"rows_{which}"][lo:hi]
    off = _T[f"off_{which}"]
    tcs = _T[f"tc_{which}"]
    mv = _T[f"mv_{which}"]
    res = {}
    codes = {"raw": [], "canon": []}
    lens = {"raw": [], "canon": []}
    for r in rows.tolist():
        s, e = int(off[r]), int(off[r + 1])
        cc = np.asarray(mv[s:min(e, s + 60)]).tolist()
        pts = [c & 511 for c in cc]
        cols = [c >> 9 for c in cc]
        if any(p >= 361 for p in pts) or any(c > 1 for c in cols):
            raise RuntimeError("unexpected move code")
        tc = int(tcs[r])
        a = game_tokens(pts, cols, tc)
        b = game_tokens(canon_pts(pts), cols, tc)
        codes["raw"].extend(a)
        codes["canon"].extend(b)
        lens["raw"].append(len(a))
        lens["canon"].append(len(b))
    for d in ("raw", "canon"):
        c = np.array(codes[d], dtype=np.int64)
        ln = np.array(lens[d], dtype=np.int64)
        if which == "fit":
            pl = np.repeat(_T["pidx_fit"][rows], ln)
            u, cnt = np.unique((pl << 25) | c, return_counts=True)
            res[d] = (u, cnt)
        else:
            res[d] = (c.astype(np.int32), ln)
    return which, lo, res


# ------------------------------------------------------------------ similarity workers
_W = {}


def sim_init(direc):
    for k in THREADS:
        os.environ[k] = "1"
    with np.load(os.path.join(TMP, f"R_{direc}.npz"), allow_pickle=False) as z:
        _W["R"] = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    with np.load(os.path.join(TMP, f"G_{direc}.npz"), allow_pickle=False) as z:
        _W["G"] = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        _W["idf"] = z["idf"]
    with np.load(os.path.join(TMP, "common.npz"), allow_pickle=False) as z:
        for k in z.files:
            _W[k] = z[k]
    NREF = _W["R"].shape[0]
    rm = np.zeros(NREF, bool)
    rm[N_FIT:] = ~_W["keep05"]
    _W["rm05"] = rm


def _w(C, idc):
    return np.where(C > 0, 1.0 + np.log(np.where(C > 0, C, 1.0)), 0.0) * idc


def sim_work(pi):
    R, G, idf = _W["R"], _W["G"], _W["idf"]
    games = _W["prow"][_W["pptr"][pi]:_W["pptr"][pi + 1]]
    qs = np.nonzero(_W["mono_player"] == pi)[0]
    pos = {int(g): j for j, g in enumerate(games.tolist())}
    Gp = G[games]
    cols = np.unique(Gp.indices)
    GpcT = Gp[:, cols].T.tocsr()          # c x n_p
    idc = idf[cols]
    own = N_FIT + pi
    nq = len(qs)
    out = {"pi": pi, "qs": qs, "idx": np.zeros((3, nq, 10), np.int32), "cos": np.zeros((3, nq, 10)),
           "own_cos": np.full(nq, np.nan), "own_rank": np.full(nq, -1, np.int32)}
    for c0 in range(0, nq, 64):
        qq = qs[c0:c0 + 64]
        m = len(qq)
        A = np.zeros((len(games), m))
        for r, q in enumerate(qq.tolist()):
            g = _W["mono_games"][q]
            for x in g[g >= 0].tolist():
                A[pos[x], r] = 1.0
        if not (A.sum(0) == _W["mono_n"][qq]).all():
            raise RuntimeError("question game count")
        B = 1.0 - A                                # complement: every other game of this player
        Cq = np.asarray(GpcT @ A).T
        Co = np.asarray(GpcT @ B).T
        Wq = _w(Cq, idc)
        Wq /= np.sqrt((Wq * Wq).sum(1))[:, None]
        Wo = _w(Co, idc)
        no = np.sqrt((Wo * Wo).sum(1))
        absent = no == 0
        oc = np.where(absent, np.nan, (Wq * Wo).sum(1) / np.where(absent, 1.0, no))
        qm = (Cq > 0).any(0)
        S = np.asarray(R[:, cols[qm]] @ Wq[:, qm].T).T.copy()
        S[:, own] = np.where(absent, MASK, oc)
        sl = slice(c0, c0 + m)
        out["own_cos"][sl] = oc
        out["own_rank"][sl] = np.where(absent, -1, (S > S[:, own][:, None]).sum(1))
        for j, sc in enumerate(("pi1", "pi0", "pi05")):
            X = S.copy()
            if sc == "pi0":
                X[:, own] = MASK
            elif sc == "pi05":
                X[:, _W["rm05"]] = MASK
            o = np.argsort(-X, axis=1, kind="stable")[:, :10]
            out["idx"][j, sl] = o
            out["cos"][j, sl] = np.take_along_axis(X, o, 1)
    return out


# ------------------------------------------------------------------ main
def stage_tokens(ctx):
    zf = np.load(os.path.join(DS, "moves_fit.npz"), allow_pickle=False)
    off_f, pid_f, col_f, lab_f = zf["off"], zf["player_id"], zf["color"], zf["label"].astype(np.int64)
    zf.close()
    fit_ids, pidx = np.unique(pid_f, return_inverse=True)
    assert len(fit_ids) == N_FIT
    fit_level = np.zeros(N_FIT, np.int64)
    fit_level[pidx] = lab_f
    assert (fit_level[pidx] == lab_f).all()
    zk = np.load(KEYS, allow_pickle=False)
    keys = {k: zk[k] for k in zk.files}
    zi = np.load(os.path.join(DS, "moves_inner.npz"), allow_pickle=False)
    off_i = zi["off"]
    where = {(r, g): i for i, (r, g) in enumerate(zip(zi["rank"].tolist(), zi["game_id"].tolist()))}
    rows_i = np.array([where[(r, g)] for r, g in zip(keys["rank"].tolist(), keys["game_id"].tolist())], np.int64)
    assert len(set(rows_i.tolist())) == len(rows_i) == len(where)
    assert (zi["player_id"][rows_i] == keys["player_id"]).all() and (zi["color"][rows_i] == keys["color"]).all()
    zi.close()
    tc_i_file = np.zeros(len(off_i) - 1, np.int64)
    tc_i_file[rows_i] = (keys["color"] == "W").astype(np.int64)
    np.savez(os.path.join(TMP, "tok_meta.npz"), rows_fit=np.argsort(pidx, kind="stable"), off_fit=off_f,
             tc_fit=(col_f == "W").astype(np.int64), pidx_fit=pidx.astype(np.int64), rows_inner=rows_i,
             off_inner=off_i, tc_inner=tc_i_file)
    tasks = [("fit", lo, min(lo + 15000, len(pidx))) for lo in range(0, len(pidx), 15000)]
    tasks += [("inner", lo, min(lo + 5000, len(rows_i))) for lo in range(0, len(rows_i), 5000)]
    fit_parts = {"raw": [], "canon": []}
    inner_parts = {}
    with ctx.Pool(4, initializer=tok_init) as pool:
        for which, lo, res in pool.imap_unordered(tok_work, tasks, chunksize=1):
            if which == "fit":
                for d in ("raw", "canon"):
                    fit_parts[d].append(res[d])
            else:
                inner_parts[lo] = res
    log("tokens done")
    return {"fit_parts": fit_parts, "inner_parts": inner_parts, "keys": keys, "fit_level": fit_level}


def build(direc, T, rep):
    pairs = np.concatenate([p[0] for p in T["fit_parts"][direc]])
    cnts = np.concatenate([p[1] for p in T["fit_parts"][direc]])
    fpl = (pairs >> 25)
    fcode = (pairs & ((1 << 25) - 1))
    los = sorted(T["inner_parts"])
    icode = np.concatenate([T["inner_parts"][lo][direc][0] for lo in los]).astype(np.int64)
    ilen = np.concatenate([T["inner_parts"][lo][direc][1] for lo in los])
    vocab = np.unique(np.concatenate([fcode, icode]))
    V = len(vocab)
    F = sparse.coo_matrix((cnts.astype(np.float64), (fpl, np.searchsorted(vocab, fcode))), shape=(N_FIT, V)).tocsr()
    F.sum_duplicates()
    df = np.diff(F.tocsc().indptr)               # fit players having each token
    idf = np.log(5768.0 / (1.0 + df)) + 1.0
    nG = len(ilen)
    G = sparse.coo_matrix((np.ones(len(icode)), (np.repeat(np.arange(nG), ilen), np.searchsorted(vocab, icode))),
                          shape=(nG, V)).tocsr()
    G.sum_duplicates()
    keys = T["keys"]
    pidmap = {p: i for i, p in enumerate(T["player_id"])}
    gpl = np.array([pidmap.get(p, -1) for p in keys["player_id"].tolist()])
    Ain = sparse.coo_matrix((np.ones((gpl >= 0).sum()), (gpl[gpl >= 0], np.nonzero(gpl >= 0)[0])),
                            shape=(len(T["player_id"]), nG)).tocsr()
    Cin = (Ain @ G).tocsr()
    Fw = F.copy()
    Fw.data = (1.0 + np.log(Fw.data)) * idf[Fw.indices]
    Cw = Cin.copy()
    Cw.data = (1.0 + np.log(Cw.data)) * idf[Cw.indices]
    R = sparse.vstack([Fw, Cw]).tocsr()
    nr = np.sqrt(np.asarray(R.multiply(R).sum(1)).ravel())
    assert (nr > 0).all()
    R = sparse.diags(1.0 / nr) @ R
    R = R.tocsc()
    R.sort_indices()
    info = {"size": int(V), "sha256": hashlib.sha256(vocab.astype(np.int32).tobytes()).hexdigest(),
            "fit_profile_nnz": int(F.nnz), "inner_full_profile_nnz": int(Cin.nnz), "tokens_in_fit": int((df > 0).sum())}
    pv = rep["stage1"][f"{direc}_vocab"]
    info["equal_to_pilot"] = all(info[k] == pv[k] for k in pv)
    np.savez(os.path.join(TMP, f"R_{direc}.npz"), data=R.data, indices=R.indices, indptr=R.indptr,
             shape=np.array(R.shape))
    np.savez(os.path.join(TMP, f"G_{direc}.npz"), data=G.data, indices=G.indices, indptr=G.indptr,
             shape=np.array(G.shape), idf=idf)
    return info, gpl


def choose(arr, sc, ref_level):
    s2 = {d: arr[d][sc]["cos"][:, 0] - arr[d][sc]["cos"][:, 1] for d in ("raw", "canon")}
    uc = s2["canon"] > s2["raw"]
    S2 = np.where(uc, s2["canon"], s2["raw"])
    idx = np.where(uc[:, None], arr["canon"][sc]["idx"], arr["raw"][sc]["idx"])
    cos = np.where(uc[:, None], arr["canon"][sc]["cos"], arr["raw"][sc]["cos"])
    assert (cos > MASK + 1).all()
    w = np.exp((cos - cos[:, :1]) / 0.02)
    w /= w.sum(1, keepdims=True)
    lev = ref_level[idx]
    dist = np.zeros((len(cos), 10))
    for j in range(10):
        np.add.at(dist, (np.arange(len(cos)), lev[:, j]), w[:, j])
    return {"S2": S2, "uc": uc, "top1": idx[:, 0], "E": np.log(dist + 0.01), "lev1": lev[:, 0]}


def fit_w(logP, E, y, tr):
    yt = y[tr]
    cnt = np.bincount(yt, minlength=10).astype(float)
    wq = 1.0 / ((cnt > 0).sum() * cnt[yt])
    lp, Et = logP[tr], E[tr]

    def loss(w):
        z = lp + w * Et
        z = z - z.max(1, keepdims=True)
        ls = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -float((wq * ls[np.arange(len(yt)), yt]).sum())
    r = optimize.minimize_scalar(loss, bounds=(0.0, 4.0), method="bounded", options={"xatol": 1e-10})
    grid = np.linspace(0, 4, 401)
    gl = np.array([loss(g) for g in grid])
    cands = [(r.fun, r.x), (loss(0.0), 0.0), (loss(4.0), 4.0)]
    best = min(cands, key=lambda c: c[0])
    return float(best[1]), {"brent": float(r.x), "grid_best": float(grid[gl.argmin()]),
                            "grid_min_loss_minus_best": float(gl.min() - best[0])}


def ruleb_score(P, y):
    ex = P.copy()
    ex[:, 1:] += np.exp(-1) * P[:, :-1]
    ex[:, :-1] += np.exp(-1) * P[:, 1:]
    d = np.abs(np.argmax(ex, 1) - y)
    return np.where(d == 0, 1.0, np.where(d == 1, np.exp(-1), 0.0))


def off_type(sc, y, kidx):
    return float(np.mean([np.mean([sc[(y == i) & (kidx == j)].mean() for j in range(4)
                                   if ((y == i) & (kidx == j)).any()]) for i in range(10)]))


def main():
    os.makedirs(TMP, exist_ok=True)
    ctx = mp.get_context("spawn")
    rep = json.load(open(os.path.join(PILOT, "pilot_report.json"), encoding="utf-8"))
    pa = np.load(os.path.join(PILOT, "pilot_arrays.npz"))
    z = U.load_saved()
    out = {}
    T = stage_tokens(ctx)
    T["player_id"] = z["player_id"].tolist()
    assert (T["keys"]["color"] == z["game_colour"]).all()
    gpl = None
    for d in ("raw", "canon"):
        info, gpl = build(d, T, rep)
        out[f"vocab_{d}"] = info
        log(f"{d}: {info}")
    NP = len(z["player_id"])
    prow = [np.nonzero(gpl == p)[0] for p in range(NP)]
    pptr = np.concatenate([[0], np.cumsum([len(r) for r in prow])])
    keep05 = np.array([hashlib.sha256(("c10-p05-d2|" + p).encode()).digest()[0] < 128 for p in T["player_id"]])
    out["keep05"] = int(keep05.sum())
    np.savez(os.path.join(TMP, "common.npz"), prow=np.concatenate(prow), pptr=pptr, keep05=keep05,
             mono_games=z["mono_games"], mono_player=z["mono_player"], mono_n=z["mono_n"])
    del T["fit_parts"], T["inner_parts"]
    Q = len(z["mono_y"])
    arr = {}
    for d in ("raw", "canon"):
        A = {sc: {"idx": np.full((Q, 10), -1, np.int32), "cos": np.full((Q, 10), np.nan)} for sc in ("pi1", "pi0", "pi05")}
        oc = np.full(Q, np.nan)
        orank = np.full(Q, -2, np.int32)
        seen = np.zeros(Q, bool)
        with ctx.Pool(4, initializer=sim_init, initargs=(d,)) as pool:
            order = np.argsort(-np.bincount(z["mono_player"], minlength=NP) * (pptr[1:] - pptr[:-1] + 50), kind="stable")
            for o in pool.imap_unordered(sim_work, order.tolist(), chunksize=1):
                qs = o["qs"]
                assert not seen[qs].any()
                seen[qs] = True
                for j, sc in enumerate(("pi1", "pi0", "pi05")):
                    A[sc]["idx"][qs] = o["idx"][j]
                    A[sc]["cos"][qs] = o["cos"][j]
                oc[qs] = o["own_cos"]
                orank[qs] = o["own_rank"]
        assert seen.all()
        A["own_cos"], A["own_rank"] = oc, orank
        arr[d] = A
        has = ~np.isnan(oc)
        out[f"own_{d}"] = {"absent": int((~has).sum()),
                           "own_cos_max_abs_diff_vs_pilot_float32": float(np.nanmax(np.abs(oc - pa[f"own_cos_{d}"].astype(np.float64)))),
                           "own_absent_same_as_pilot": bool(np.array_equal(~has, np.isnan(pa[f"own_cos_{d}"]))),
                           "own_rank_mismatch_vs_pilot": int((orank != pa[f"own_rank_{d}"]).sum()),
                           "own_is_top1": float((orank[has] == 0).mean())}
        log(f"{d}: similarities done; {out[f'own_{d}']}")

    # ---------------- stage 2 recomputation
    y, n, fold, player = z["mono_y"], z["mono_n"], z["mono_fold"], z["mono_player"]
    kidx = np.searchsorted(NS, n)
    P02 = z["mono_P_Q02"]
    logP = np.log(P02)
    fit_level = T["fit_level"]
    ref_level = np.concatenate([fit_level, z["player_level"]])
    ch = {sc: choose(arr, sc, ref_level) for sc in ("pi1", "pi0", "pi05")}
    for sc in ch:
        out[f"cmp_{sc}"] = {"S2_max_abs_diff": float(np.max(np.abs(ch[sc]["S2"] - pa[f"S2_{sc}"]))),
                            "use_canon_mismatch": int((ch[sc]["uc"] != pa[f"use_canon_{sc}"]).sum()),
                            "top1_mismatch": int((ch[sc]["top1"] != pa[f"top1_{sc}"]).sum())}
    th = np.zeros((5, 4))
    for f in range(5):
        for j in range(4):
            th[f, j] = np.percentile(ch["pi0"]["S2"][(fold != f) & (kidx == j)], 99)
    out["theta_max_abs_diff_vs_pilot"] = float(np.max(np.abs(th - pa["theta_fold_by_k"])))
    trig = {sc: ch[sc]["S2"] > th[fold, kidx] for sc in ch}
    for sc in ch:
        out[f"cmp_{sc}"]["trigger_mismatch"] = int((trig[sc] != pa[f"trig_{sc}"]).sum())
        out[f"cmp_{sc}"]["trigger_rate"] = float(trig[sc].mean())
        rows = np.nonzero(trig[sc] & pa[f"trig_{sc}"])[0]
        pos = {int(r): i for i, r in enumerate(pa[f"E_rows_{sc}"].tolist())}
        Ep = pa[f"E_{sc}"][[pos[int(r)] for r in rows]]
        out[f"cmp_{sc}"]["E_max_abs_diff_on_common_triggered"] = float(np.max(np.abs(ch[sc]["E"][rows] - Ep)))
        tt = trig[sc]
        out[f"cmp_{sc}"]["top1_is_self_triggered"] = float((ch[sc]["top1"][tt] == N_FIT + player[tt]).mean())
        out[f"cmp_{sc}"]["borrowed_exact_triggered"] = float((ch[sc]["lev1"][tt] == y[tt]).mean())
    ws = {}
    for basis in ("pi1", "pi0"):
        ws[basis] = []
        recs = []
        for f in range(5):
            E = ch[basis]["E"] * (ch[basis]["S2"] > th[f][kidx])[:, None]
            w, rec = fit_w(logP, E, y, fold != f)
            ws[basis].append(w)
            recs.append(rec)
        out[f"w_{basis}"] = {"mine": ws[basis], "recs": recs}
    out["w_pi1_max_abs_diff_vs_pilot"] = float(np.max(np.abs(np.array(ws["pi1"]) - pa["w_pi1"])))
    out["w_pi0_max_abs_diff_vs_pilot"] = float(np.max(np.abs(np.array(ws["pi0"]) - pa["w_pi0_refit"])))
    sq = ruleb_score(P02, y)
    base = off_type(sq, y, kidx)
    for key, sc, wv in (("pi1", "pi1", ws["pi1"]), ("pi0", "pi0", ws["pi1"]), ("pi05", "pi05", ws["pi1"]),
                        ("pi0_refit", "pi0", ws["pi0"])):
        s = sq.copy()
        tt = trig[sc]
        lg = logP[tt] + np.array(wv)[fold[tt]][:, None] * ch[sc]["E"][tt]
        lg -= lg.max(1, keepdims=True)
        P = np.exp(lg)
        P /= P.sum(1, keepdims=True)
        s[tt] = ruleb_score(P, y[tt])
        out[f"result_{key}"] = {"official_type_delta": off_type(s, y, kidx) - base,
                                "report_delta": rep["results"][key]["official_type"]["delta"],
                                "score_mismatch_vs_pilot": int((s != pa[f"score_{key}"]).sum())}
    pools = pa["pools"].astype(np.int64)
    out["pools"] = {sc: {"P_ON": float((trig[sc][pools].mean(1) >= 0.10).mean()),
                         "share_mean": float(trig[sc][pools].mean())} for sc in ch}
    with open(os.path.join(HERE, "review_b.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    shutil.rmtree(TMP, ignore_errors=True)
    log("done")


if __name__ == "__main__":
    main()
