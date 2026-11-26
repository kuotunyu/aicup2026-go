"""prereg63 explore (PLAN.md, NOTES.md, frozen.json): competition 1 new questions, v2-7 + A (colour-fill fix) +
B (P1 multi-subspace from L+M1) + C (S25 rank compatibility), B and C ported from Gemini's 11/21 script. CPU only.

    PY results/prereg63/explore/p63_eval.py fit                  # Fisher LDAs on L+M1 -> data/prereg63/p1multi.npz
    PY results/prereg63/explore/p63_eval.py seam                 # D: Gemini's v2-6 / U1_combined; E: v2-7 bit for bit
    PY results/prereg63/explore/p63_eval.py prep-e --part terms|zbc   # E: fixed S2/PA/L terms, then B/C z terms (no scores)
    PY results/prereg63/explore/p63_eval.py e-use                # prints the E use number this judgement gets
    PY results/prereg63/explore/p63_eval.py judge --e-use N      # once -> eval.json
    PY results/prereg63/explore/p63_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg63/explore/p63_eval.py assemble --src <official tests dir>

A63 = CSLS_k( fuse_cfg(pool with A-fixed terms, e_term(pool, c49, cfg), cfg) + 9.0 z_multi + 2.5 z_rank ),
cfg / k = v2-6's (= v2-7's). Base v2-7 = CSLS_k( fuse_cfg(pool, e_term(pool, c49, cfg), cfg) + wP z_P1 ).
Nothing here opens the validation-answer folder or runs tools/score.py.
"""
import argparse
import csv
import glob
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P49_DIR = os.path.join(REPO, "results", "prereg49", "explore")
P45_DIR = os.path.join(REPO, "results", "prereg45", "explore")
for p in (P45_DIR, P49_DIR, os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import p49_eval as P49  # noqa: E402   (v2-7's judgement path; unchanged)

E43, E45, P39E = P49.E43, P49.E45, P49.P39E
BK, EV, st, L, sim, F10 = E43.BK, E43.EV, E43.st, E43.L, E43.sim, P39E.F10
Q42 = E43.Q42
_rel, read_json = E43._rel, E43.read_json
import p09_forward as FW09  # noqa: E402
import p25_forward as FW25  # noqa: E402

DATA = os.path.join(REPO, "data", "prereg63")
P1MULTI = os.path.join(DATA, "p1multi.npz")
TERMS_FIX = os.path.join(DATA, "terms_E_fix")
ZBC_E = os.path.join(DATA, "zBC_E.npz")
PLAN = os.path.join(HERE, "PLAN.md")
FROZEN = os.path.join(HERE, "frozen.json")
SEAM_JSON = os.path.join(HERE, "seam.json")
PREP_JSON = os.path.join(HERE, "prep_e.json")
EVAL_JSON = os.path.join(HERE, "eval.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg63", "apply")
V27_APPLY = os.path.join(REPO, "results", "prereg49", "apply")
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
S25_MODELS = os.path.join(REPO, "data", "prereg25", "models")
S0_D = os.path.join(REPO, "data", "prereg43", "emb_S0", "V2_step_320000_raw_tta1", "store_D.npz")
M1_DIR = os.path.join(REPO, "data", "prereg45", "p1", "M1")

# ---- frozen.json (Gemini's 02_hypotheses_and_comp1_verify.py)
LDA_SPECS = (("full", 0, 770, 128), ("move", 0, 384, 64), ("board", 384, 768, 64))   # lines 638-640
LDA_REG = 1e-2
LDA_BLOCK = 20000
B4_LAM = 0.5                                     # lines 791-793
MULTI_W = {"full": 0.5, "move": 0.3, "board": 0.2}  # line 794
SIGMA_RK = 0.85                                  # line 797
W_MULTI, W_RANK = 9.0, 2.5                       # line 813
D_V26_GEMINI, D_U1_GEMINI, TOL = 0.746777, 0.756796, 1e-6
D_U1_BYK = {"1": 0.5227, "2": 0.7714, "3": 0.8171, "5": 0.9159}
E_V27 = 0.7035676497986656
ADOPT_MEAN = 0.01
RAM_MIN_MB = 7000
RSS_LIMIT_GB = 3.0
RECOMP_TOL = 1e-6   # NOTES.md 10: the LDA refit reproduces the stored float32 terms to ~1 ulp, not bit for bit


# ================================================================ resources
def rss_watchdog(limit_gb=RSS_LIMIT_GB):
    import psutil
    proc = psutil.Process()
    state = {"peak": 0}

    def watch():
        while True:
            r = proc.memory_info().rss
            state["peak"] = max(state["peak"], r)
            if r > limit_gb * (1024 ** 3):
                print("WATCHDOG: RSS %.2f GB > %.1f GB, exiting" % (r / 1024 ** 3, limit_gb), flush=True)
                os._exit(97)
            time.sleep(0.2)

    threading.Thread(target=watch, daemon=True).start()
    return state


_WD = {}


def peak_mb():
    return round(_WD["s"]["peak"] / 1048576.0, 1) if "s" in _WD else None


# ================================================================ A: colour fill (PLAN.md A)
def colour_terms_fixed(per_colour, q_counts):
    """= tools/p06_style.colour_terms except the fill: per colour the row z is taken over the candidates that have
    games of that colour; the others get 0 (no candidate with the colour -> 0, as before)."""
    cols = sorted(per_colour)
    nq = len(next(iter(q_counts.values())))
    total = np.zeros(nq)
    for c in cols:
        total += np.asarray(q_counts[c], np.float64)
    nc = per_colour[cols[0]][0].shape[1]
    out = np.zeros((nq, nc))
    for c in cols:
        T, ch = per_colour[c]
        ch = np.asarray(ch, bool)
        k = np.asarray(q_counts[c], np.float64)
        rows = np.flatnonzero(k > 0)
        if not len(rows):
            continue
        T = np.array(T[rows], np.float64, copy=True)
        ones = np.ones(len(rows), bool)
        if ch.all():
            Z = st.zscore_rows(T, ones)
        else:
            Z = np.zeros(T.shape)
            if ch.any():
                Z[:, ch] = st.zscore_rows(T[:, ch], ones)
        out[rows] += (k[rows] / total[rows])[:, None] * Z
    return out, total > 0


class fixed_fill:
    """Context: tools/p06_sim's colour_terms -> colour_terms_fixed (tools/p06_style.py is not edited)."""

    def __enter__(self):
        if sim.st is not st:
            raise SystemExit("p06_sim and the judgement path import different p06_style modules")
        self.old = sim.st.colour_terms
        sim.st.colour_terms = colour_terms_fixed
        return self

    def __exit__(self, *exc):
        sim.st.colour_terms = self.old
        return False


# ================================================================ B, C: ported from Gemini's script (frozen.json)
def l2_rows(X):
    X = np.asarray(X, np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(nrm > 0, nrm, 1.0)


_MOCK = {}


def _mock():
    """lines 535-549: the PREREG_10 mock chunks, both sides."""
    if not _MOCK:
        n = len(read_json(P39E.MOCK_SETS)["games"])
        means = np.zeros((n, 2, 770), np.float32)
        counts = np.zeros((n, 2), np.int64)
        usable = np.zeros(n, bool)
        for mp in F10.chunk_files(P39E.MOCK_DIR):
            with np.load(mp, allow_pickle=False) as z:
                r = np.asarray(z["ref"], np.int64)
                f = np.asarray(z["failed"], bool)
                c = np.asarray(z["counts"], np.int64)
                means[r] = np.asarray(z["means"], np.float32)
                counts[r] = c
                usable[r] = (~f) & (c[:, 0] > 0)
        _MOCK.update(means=means, counts=counts, usable=usable)
    return _MOCK


def load_p1_full(name):
    """lines 551-577 (Gemini's load_p1_full): -> games, means [n, 2, 770] float32, counts [n, 2], usable [n]."""
    if name == "M1":
        games = E45.read_list45("M1")
        d = M1_DIR
    else:
        games = P39E.read_list(name)
        d = os.path.join(P39E.P1, name)
    n = len(games)
    means = np.zeros((n, 2, 770), np.float32)
    counts = np.zeros((n, 2), np.int64)
    usable = np.zeros(n, bool)
    if os.path.isdir(d):
        for ch in F10.chunk_files(d):
            with np.load(ch, allow_pickle=False) as z:
                r = np.asarray(z["ref"], np.int64)
                f = np.asarray(z["failed"], bool)
                c = np.asarray(z["counts"], np.int64)
                means[r] = np.asarray(z["means"], np.float32)
                counts[r] = c
                usable[r] = (~f) & (c[:, 0] > 0)
    reuse = np.array([g.get("mock_ref", -1) for g in games], np.int64)
    idx = np.flatnonzero(reuse >= 0)
    if len(idx):
        m = _mock()
        means[idx] = m["means"][reuse[idx]]
        counts[idx] = m["counts"][reuse[idx]]
        usable[idx] = m["usable"][reuse[idx]]
    return games, means, counts, usable


def fit_fisher_lda(X, usable, pids, out_dim=96, reg=1e-2, block=LDA_BLOCK):
    """lines 595-636, unchanged."""
    import scipy.linalg as sla
    ok = np.flatnonzero(usable)
    n_ok = len(ok)
    d = X.shape[1]
    s1 = np.zeros(d, np.float64)
    s2 = np.zeros(d, np.float64)
    for s in range(0, n_ok, block):
        xb = np.asarray(X[ok[s:s + block]], np.float64)
        s1 += xb.sum(axis=0)
        s2 += (xb * xb).sum(axis=0)
    mu = s1 / n_ok
    var = np.maximum(s2 / n_ok - mu * mu, 0.0)
    sd = np.sqrt(var)
    sd = np.where(sd > 1e-6, sd, 1.0)
    _, inv = np.unique(pids[ok], return_inverse=True)
    nc = int(inv.max()) + 1
    sums = np.zeros((nc, d), np.float64)
    cnt = np.bincount(inv, minlength=nc).astype(np.float64)
    for s in range(0, n_ok, block):
        xb = (np.asarray(X[ok[s:s + block]], np.float64) - mu) / sd
        np.add.at(sums, inv[s:s + block], xb)
    keep = cnt >= 2
    means_c = np.zeros_like(sums)
    means_c[keep] = sums[keep] / cnt[keep, None]
    Sw = np.zeros((d, d), np.float64)
    for s in range(0, n_ok, block):
        xb = (np.asarray(X[ok[s:s + block]], np.float64) - mu) / sd
        rb = xb - means_c[inv[s:s + block]]
        Sw += rb.T @ rb
    Sw /= max(1.0, n_ok - keep.sum())
    Sw_reg = (1.0 - reg) * Sw + reg * (np.trace(Sw) / d) * np.eye(d)
    diff_c = means_c[keep]
    Sb = ((diff_c * cnt[keep, None]).T @ diff_c) / max(1.0, n_ok)
    evals, evecs = sla.eigh(Sb, Sw_reg)
    order = np.argsort(evals)[::-1][:out_dim]
    return {"mu": mu, "sd": sd, "W": evecs[:, order]}


def project_subspace(X_side0, mod):
    """lines 648-650."""
    Xs = (np.asarray(X_side0, np.float64) - mod["mu"]) / mod["sd"]
    return Xs @ mod["W"]


def b4_proto_term(Z_all, usable, cand_groups, q_groups, lam=B4_LAM, lda_dim=None):
    """lines 652-687, unchanged."""
    Xc_l, gc_l, Xq_l, gq_l = [], [], [], []
    has_q = np.zeros(len(q_groups), bool)
    has_c = np.zeros(len(cand_groups), bool)
    for ci, grp in enumerate(cand_groups):
        u_g = [g for g in grp if usable[g]]
        if u_g:
            has_c[ci] = True
            for g in u_g:
                Xc_l.append(Z_all[g])
                gc_l.append(ci)
    for qi, grp in enumerate(q_groups):
        u_g = [g for g in grp if usable[g]]
        if u_g:
            has_q[qi] = True
            for g in u_g:
                Xq_l.append(Z_all[g])
                gq_l.append(qi)
    Xc = l2_rows(np.array(Xc_l))
    Xq = l2_rows(np.array(Xq_l))
    gc_a = np.array(gc_l, np.int64)
    gq_a = np.array(gq_l, np.int64)
    mu, P = BK.whitening(Xc, gc_a, lam, lda_dim)
    Yc = (Xc - mu) @ P
    Yq = (Xq - mu) @ P
    Pc = np.zeros((len(cand_groups), Yc.shape[1]))
    np.add.at(Pc, gc_a, Yc)
    Pc = l2_rows(Pc)
    Pq = np.zeros((len(q_groups), Yq.shape[1]))
    np.add.at(Pq, gq_a, Yq)
    Pq = l2_rows(Pq)
    T = Pq @ Pc.T
    if not has_c.all() and has_c.any():
        T[:, ~has_c] = T[:, has_c].min(axis=1, keepdims=True)
    return st.zscore_rows(T, has_q)


def rank_kernel(sigma=SIGMA_RK):
    ii = np.arange(10)
    return np.exp(-((ii[:, None] - ii[None, :]) ** 2) / (2.0 * (sigma ** 2)))


def s25_rank_compat_term(s25, q09, means, counts, usable, cand_groups, q_groups, sigma_rk=SIGMA_RK):
    """lines 726-744, unchanged."""
    from scipy import sparse
    all_groups = list(cand_groups) + list(q_groups)
    rows, cols = [], []
    for gi, grp in enumerate(all_groups):
        rows.extend([gi] * len(grp))
        cols.extend(grp)
    S_mat = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(all_groups), len(usable)))
    _E, lp, has = FW25.question_set_term(s25, means, counts, usable, S_mat, q09["mu_x"], q09["sd_x"])
    prob = np.exp(lp - lp.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    nc = len(cand_groups)
    pc = prob[:nc]
    pq = prob[nc:]
    compat = np.log(np.maximum(pq @ rank_kernel(sigma_rk) @ pc.T, 1e-6))
    return st.zscore_rows(compat, has[nc:]), compat


def s25_rank_compat_subset(s25, q09, means, counts, usable, cand_groups, q_groups):
    """NOTES.md 11 (memory): s25_rank_compat_term on only the rows the pool uses (indices remapped); the per-game
    network is row-wise, check-subset compares it with the full-set call on the D pools."""
    rows = np.array(sorted({int(x) for g in list(cand_groups) + list(q_groups) for x in g}), np.int64)
    loc = {int(r): i for i, r in enumerate(rows.tolist())}
    cg = [[loc[int(x)] for x in g] for g in cand_groups]
    qg = [[loc[int(x)] for x in g] for g in q_groups]
    return s25_rank_compat_term(s25, q09, np.asarray(means[rows]), np.asarray(counts[rows]), np.asarray(usable[rows]),
                                cg, qg)


def z_multi_of(Zp, usable, cg, qg):
    """line 794."""
    z = {n: b4_proto_term(Zp[n], usable, cg, qg) for n in MULTI_W}
    return st.zrows(MULTI_W["full"] * z["full"] + MULTI_W["move"] * z["move"] + MULTI_W["board"] * z["board"])


def project_multi(means, mods):
    X0 = means[:, 0, :]
    return {n: project_subspace(X0[:, a:b], mods[n]) for n, a, b, _d in LDA_SPECS}


def load_models():
    q09 = FW09.load_map(Q09_MAP)
    s25 = FW25.load_models(S25_MODELS)
    return q09, s25


def load_multi(rec=None):
    if not os.path.exists(P1MULTI):
        raise SystemExit("missing %s (run fit)" % _rel(P1MULTI))
    if rec is not None and L.sha256_file(P1MULTI) != rec:
        raise SystemExit("%s differs from the recorded sha256" % _rel(P1MULTI))
    with np.load(P1MULTI, allow_pickle=False) as z:
        return {n: {k: np.asarray(z["%s_%s" % (n, k)], np.float64) for k in ("mu", "sd", "W")}
                for n, _a, _b, _d in LDA_SPECS}


# ================================================================ fit
def cmd_fit(a):
    t0 = time.time()
    gL, mL, _cL, uL = load_p1_full("L")
    X_L0 = np.array(mL[:, 0, :], np.float32)
    del mL, _cL
    gM, mM, _cM, uM = load_p1_full("M1")
    X_M0 = np.array(mM[:, 0, :], np.float32)
    del mM, _cM
    X = np.concatenate([X_L0, X_M0], axis=0)
    u = np.concatenate([uL, uM], axis=0)
    pid = np.array([g["player_id"] for g in gL] + [g["player_id"] for g in gM])
    del X_L0, X_M0
    arrs, info = {}, {}
    for n, lo, hi, d in LDA_SPECS:
        Xs = X if (lo, hi) == (0, 770) else X[:, lo:hi]
        m = fit_fisher_lda(Xs, u, pid, out_dim=d, reg=LDA_REG)
        for k in ("mu", "sd", "W"):
            arrs["%s_%s" % (n, k)] = m[k]
        info[n] = {"cols": [lo, hi], "out_dim": d, "W_shape": list(m["W"].shape)}
    os.makedirs(DATA, exist_ok=True)
    tmp = P1MULTI + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, P1MULTI)
    rec = {"format": "c10-prereg63 fit v1", "file": _rel(P1MULTI), "sha256": L.sha256_file(P1MULTI),
           "games": {"L": len(gL), "M1": len(gM)}, "usable": int(u.sum()), "players": int(len(set(pid[u].tolist()))),
           "lda": info, "git_head": L.git_head(), "peak_rss_mb": peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(HERE, "fit.json"), rec)
    print(json.dumps(rec, indent=1))
    return 0


# ================================================================ seam
def seam_d(mods):
    """Gemini's PART A (lines 749-825) for v26_base and U1_combined only."""
    with np.load(os.path.join(L.FEATS, "meta.npz"), allow_pickle=False) as zm:
        store_col = np.asarray(zm["color"]).astype(str)
    base = E45.v26()
    cfg, wP, k = base["cfg"], base["wP"], base["k"]
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    emb = EV.Emb.load(S0_D, colour=store_col)
    gL, mL, _c, uL = load_p1_full("L")
    models_L = P39E.fit_models({"games": gL, "X": np.array(mL[:, 0, :], np.float32), "usable": uL})
    del mL, _c
    gD, mD, cD, uD = load_p1_full("D")
    pos = {int(g["row"]): i for i, g in enumerate(gD)}
    Z_orig = P39E.project_all(mD[:, 0, :], models_L)["joint"]
    Zp = project_multi(mD, mods)
    q09, s25 = load_models()
    per = {"v26": [], "U1": []}
    byk = {"v26": [], "U1": []}
    for p in pools:
        F = BK.fuse_cfg(p, BK.e_term(p, emb, cfg), cfg)
        cg = [[pos[int(r)] for r in g] for g in p["cand_rows"]]
        qg = [[pos[int(r)] for r in g] for g in p["q_rows"]]
        cgu = [[x for x in g if uD[x]] for g in cg]
        qgu = [[x for x in g if uD[x]] for g in qg]
        T, has = P39E._term(Z_orig, cgu, qgu, "b4")
        S_v = st.csls(F + wP * st.zscore_rows(T, has), k)
        zm = z_multi_of(Zp, uD, cg, qg)
        zr, _raw = s25_rank_compat_term(s25, q09, mD, cD, uD, cg, qg)
        S_u = st.csls(F + W_MULTI * zm + W_RANK * zr, k)
        for name, S in (("v26", S_v), ("U1", S_u)):
            per[name].append(float(st.score(S, p["owner"])))
            byk[name].append(EV.score_by_k(S, p))
    out = {n: {"mean": float(np.mean(per[n])), "per_pool": per[n], "by_k": EV.mean_by_k(byk[n])} for n in per}
    out["v26"]["want"], out["U1"]["want"] = D_V26_GEMINI, D_U1_GEMINI
    out["v26"]["equals_prereg43_d_per_pool"] = per["v26"] == base["d_per_pool"]
    for n in ("v26", "U1"):
        out[n]["abs_diff"] = abs(out[n]["mean"] - out[n]["want"])
    out["U1"]["by_k_want"] = D_U1_BYK
    out["U1"]["by_k_match_4dp"] = all(round(out["U1"]["by_k"][x], 4) == D_U1_BYK[x] for x in D_U1_BYK)
    out["ok"] = bool(out["v26"]["abs_diff"] <= TOL and out["U1"]["abs_diff"] <= TOL and out["U1"]["by_k_match_4dp"])
    return out


def seam_e():
    base = E45.v26()
    pools, zs, bper, vec, sets = P49.e_context(base)
    path = P49.c49_file("E")
    per, byk = P49.system(path, pools, zs, base)
    want = read_json(P49.EVAL_JSON)["per_pool"]["c49"]
    return {"per_pool": per, "mean": float(np.mean(per)), "want_mean": E_V27,
            "identical": bool(per == want and float(np.mean(per)) == E_V27), "by_k": EV.mean_by_k(byk),
            "vectors": {"file": _rel(path), "sha256": L.sha256_file(path)}, "base_vectors_E": vec}


def cmd_seam(a):
    t0 = time.time()
    fit = read_json(os.path.join(HERE, "fit.json"))
    mods = load_multi(fit["sha256"])
    d = seam_d(mods)
    print(json.dumps({"D_v26": d["v26"]["mean"], "D_U1": d["U1"]["mean"], "D_ok": d["ok"]}), flush=True)
    rec = {"format": "c10-prereg63 seam v1", "p1multi": {"file": _rel(P1MULTI), "sha256": fit["sha256"]}, "D": d}
    if d["ok"]:
        del mods
        P39E._MOCK.clear()
        _MOCK.clear()
        rec["E_v27"] = seam_e()
    rec["ok"] = bool(d["ok"] and rec.get("E_v27", {}).get("identical", False))
    rec.update(git_head=L.git_head(), peak_rss_mb=peak_mb(), seconds=round(time.time() - t0, 1))
    L.write_json(a.out, rec)
    print(json.dumps({"ok": rec["ok"], "E_v27": rec.get("E_v27", {}).get("mean"), "seconds": rec["seconds"],
                      "peak_rss_mb": rec["peak_rss_mb"]}))
    if not rec["ok"]:
        raise SystemExit("seam failed: stop (do not judge)")
    return 0


# ================================================================ E preparation (no scores)
def e_store_index():
    """Gemini's loader on E and E2 concatenated, indexed by store row."""
    gE, mE, cE, uE = load_p1_full("E")
    gE2, mE2, cE2, uE2 = load_p1_full("E2")
    games = gE + gE2
    means = np.concatenate([mE, mE2])
    del mE, mE2
    counts = np.concatenate([cE, cE2])
    usable = np.concatenate([uE, uE2])
    pos = {int(g["row"]): i for i, g in enumerate(games)}
    if len(pos) != len(games):
        raise SystemExit("a store row is in E and E2")
    return means, counts, usable, pos


def recompute_terms_e():
    """NOTES.md 3: S2, PA, L of every E pool, frozen fill (must equal the stored terms) and fixed fill."""
    E, D, _T, _per = L.splits()
    seeds, players, per_level, k_list, short = EV.pool_spec("E", E, D)
    if list(seeds) != list(EV.E_SEEDS):
        raise SystemExit("E seeds")
    store = sim.Store(L.FEATS)
    ldas = {c: sim.LdaData(L.FEATS, c, store) for c in ("B", "W")}
    os.makedirs(TERMS_FIX, exist_ok=True)
    stored = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    recs = []
    for p in stored:
        s = p["seed"]
        rec = EV.make_pool(store, s, players, per_level, k_list, short)
        if (rec["cand_rows"] != p["cand_rows"] or rec["q_rows"] != p["q_rows"]
                or not np.array_equal(rec["owner"], p["owner"]) or not np.array_equal(rec["k"], p["k"])):
            raise SystemExit("pool %d rebuilt differs from the stored pool" % s)
        models = {c: ldas[c].fit_excluding(store, set(rec["pool"])) for c in ("B", "W")}
        want = ("S2", "PA", "L")
        T_old = sim.pool_terms(store, rec, ldas, models, None, want=want)
        with fixed_fill():
            T_new = sim.pool_terms(store, rec, ldas, models, None, want=want)
        arrs, n_changed, maxdiff = {}, {}, {}
        for n in want:
            o32 = np.asarray(T_old[n][0], np.float32)
            n32 = np.asarray(T_new[n][0], np.float32)
            stored32 = np.asarray(p["T"][n][0], np.float32)
            md = float(np.abs(o32.astype(np.float64) - p["T"][n][0]).max())
            maxdiff[n] = md
            if not (md <= RECOMP_TOL
                    and np.array_equal(np.asarray(T_old[n][1], bool), np.asarray(p["T"][n][1], bool))):
                raise SystemExit("pool %d term %s recomputed with the frozen fill differs from the stored term (max "
                                 "abs %g)" % (s, n, md))
            # NOTES.md 10: rows the fill fix does not touch keep the stored values (bit for bit = v2-7)
            touched = (n32 != o32).any(axis=1)
            arrs[n] = np.where(touched[:, None], n32, stored32)
            arrs[n + "_ok"] = np.asarray(T_new[n][1], bool)
            n_changed[n] = int(touched.sum())
        f = os.path.join(TERMS_FIX, "pool_%03d.npz" % s)
        np.savez(f + ".tmp.npz", **arrs)
        os.replace(f + ".tmp.npz", f)
        recs.append({"seed": s, "file": _rel(f), "sha256": L.sha256_file(f), "rows_changed": n_changed,
                     "frozen_fill_recompute_max_abs_diff_vs_stored": maxdiff})
        print(json.dumps(recs[-1]), flush=True)
    return recs


def zbc_e(mods):
    """B and C z terms of every E pool (pool order = E_SEEDS)."""
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    means, counts, usable, pos = e_store_index()
    Zp = project_multi(means, mods)
    groups = [([[pos[int(r)] for r in g] for g in p["cand_rows"]], [[pos[int(r)] for r in g] for g in p["q_rows"]])
              for p in pools]
    arrs = {}
    for p, (cg, qg) in zip(pools, groups):
        arrs["multi_%03d" % p["seed"]] = z_multi_of(Zp, usable, cg, qg)
    del Zp          # memory: the rank term's float64 [N, 1540] inputs come next (same values, other order)
    q09, s25 = load_models()
    for p, (cg, qg) in zip(pools, groups):
        arrs["rank_%03d" % p["seed"]] = s25_rank_compat_subset(s25, q09, means, counts, usable, cg, qg)[0]
    np.savez(ZBC_E + ".tmp.npz", **arrs)
    os.replace(ZBC_E + ".tmp.npz", ZBC_E)
    return {"file": _rel(ZBC_E), "sha256": L.sha256_file(ZBC_E), "games": len(usable), "usable": int(usable.sum())}


def cmd_prep_e(a):
    """Two processes (memory): --part terms, then --part zbc (adds to the same prep_e.json)."""
    t0 = time.time()
    seam = read_json(SEAM_JSON)
    if not seam.get("ok") or not L.git_committed(SEAM_JSON):
        raise SystemExit("seam.json is not ok / not committed")
    if a.part == "terms":
        rec = {"format": "c10-prereg63 prep-e v1", "terms_fixed": recompute_terms_e(),
               "note": "features only: no owner is read, no score is computed",
               "terms_run": {"git_head": L.git_head(), "peak_rss_mb": peak_mb(), "seconds": round(time.time() - t0, 1)}}
    else:
        rec = read_json(a.out)
        if "terms_fixed" not in rec:
            raise SystemExit("run --part terms first")
        rec["zBC"] = zbc_e(load_multi(seam["p1multi"]["sha256"]))
        rec["zbc_run"] = {"git_head": L.git_head(), "peak_rss_mb": peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("terms_run", "zBC", "zbc_run") if k in rec}))
    return 0


# ================================================================ E use number (NOTES.md 7)
def e_uses_so_far(repo=REPO):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p61_eval_for_p63", os.path.join(repo, "results", "prereg61", "explore", "p61_eval.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    u = m.e_uses_so_far(repo)
    later = [x for x in u["later"] if not x.startswith("prereg63/")]
    for f in sorted(glob.glob(os.path.join(repo, "results", "prereg61", "explore", "eval*.json"))):
        with open(f, encoding="utf-8") as fh:
            if "e_pools_use_number" in json.load(fh):
                later.append("prereg61/%s" % os.path.basename(f))
    later = sorted(set(later))
    return {"through_prereg49": u["through_prereg49"], "later": later,
            "count": len(u["through_prereg49"]) + len(later)}


# ================================================================ judge
def fixed_pools(pools, prep):
    out = []
    want = {r["seed"]: r for r in prep["terms_fixed"]}
    for p in pools:
        r = want[p["seed"]]
        f = os.path.join(REPO, r["file"])
        if L.sha256_file(f) != r["sha256"]:
            raise SystemExit("%s differs from prep_e.json" % r["file"])
        T = dict(p["T"])
        with np.load(f) as z:
            for n in ("S2", "PA", "L"):
                T[n] = (np.asarray(z[n], np.float64), np.asarray(z[n + "_ok"]))
        q = dict(p)
        q["T"] = T
        out.append(q)
    return out


def lacking_colour(pools, colour):
    """Per pool: bool [nq], the owner's candidate games lack a colour of the question's games."""
    out = []
    for p in pools:
        m = np.zeros(len(p["q_rows"]), bool)
        for i, (g, o) in enumerate(zip(p["q_rows"], p["owner"])):
            qc = {str(colour[int(r)]) for r in g}
            cc = {str(colour[int(r)]) for r in p["cand_rows"][int(o)]}
            m[i] = not qc <= cc
        out.append(m)
    return out


def judge_rec(new, old):
    d = np.asarray(new, np.float64) - np.asarray(old, np.float64)
    lo, hi = Q42.bootstrap(d, E43.BOOT, E43.BOOT_SEED)
    need = int(np.ceil(0.9 * len(d)))
    r = {"mean": float(d.mean()), "ci95": [lo, hi], "pools_higher": int((d > 0).sum())}
    r["adopt"] = bool(r["mean"] >= ADOPT_MEAN and lo > 0 and r["pools_higher"] >= need)
    r["rule"] = ("adopt A63 if mean(A63 - v2-7) >= %.2f and ci95 low > 0 and higher in >= %d of %d pools (pool "
                 "bootstrap %d, seed %d)" % (ADOPT_MEAN, need, len(d), E43.BOOT, E43.BOOT_SEED))
    return r


def cmd_judge(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    for f in (PLAN, FROZEN, os.path.join(HERE, "NOTES.md"), os.path.abspath(__file__), SEAM_JSON, PREP_JSON):
        if not L.git_committed(f):
            raise SystemExit("%s is not committed" % _rel(f))
    seam, prep = read_json(SEAM_JSON), read_json(PREP_JSON)
    if "zBC" not in prep:
        raise SystemExit("prep_e.json has no zBC")
    if not seam.get("ok"):
        raise SystemExit("seam not ok")
    uses = e_uses_so_far()
    if a.e_use != uses["count"] + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (a.e_use, uses["count"], uses))
    if L.sha256_file(ZBC_E) != prep["zBC"]["sha256"]:
        raise SystemExit("zBC_E.npz differs from prep_e.json")
    base = E45.v26()
    cfg, wP, k = base["cfg"], base["wP"], base["k"]
    pools, zs, bper, vec, sets = P49.e_context(base)
    path = P49.c49_file("E")
    vec49 = P49.check_c49_file(path, "E", base["tune"]["c_star"])
    e = E43.load_emb(path)
    Fs = E43.fused(pools, e, cfg)
    fpools = fixed_pools(pools, prep)
    Ffix = E43.fused(fpools, e, cfg)
    del e
    with np.load(ZBC_E) as z:
        zm = [np.asarray(z["multi_%03d" % p["seed"]]) for p in pools]
        zr = [np.asarray(z["rank_%03d" % p["seed"]]) for p in pools]
    systems = {
        "v27": [st.csls(F + wP * zp, k) for F, zp in zip(Fs, zs)],
        "A63": [st.csls(F + W_MULTI * m + W_RANK * r, k) for F, m, r in zip(Ffix, zm, zr)],
        "A_only": [st.csls(F + wP * zp, k) for F, zp in zip(Ffix, zs)],
        "BC_only": [st.csls(F + W_MULTI * m + W_RANK * r, k) for F, m, r in zip(Fs, zm, zr)],
    }
    per = {n: [float(st.score(S, p["owner"])) for S, p in zip(Ss, pools)] for n, Ss in systems.items()}
    if per["v27"] != read_json(P49.EVAL_JSON)["per_pool"]["c49"] or float(np.mean(per["v27"])) != E_V27:
        raise SystemExit("seam: v2-7 rebuilt in the judgement differs from prereg49 eval.json")
    byk = {n: EV.mean_by_k([EV.score_by_k(S, p) for S, p in zip(Ss, pools)]) for n, Ss in systems.items()}
    lack = lacking_colour(pools, E43.store().color)
    aff = {"questions": int(sum(m.sum() for m in lack)), "of": int(sum(len(m) for m in lack)),
           "per_pool": [int(m.sum()) for m in lack]}
    for n, Ss in systems.items():
        pts = np.concatenate([st.points(st.ranks(S, p["owner"]))[m] for S, p, m in zip(Ss, pools, lack)])
        aff["mean_points_" + n] = float(pts.mean()) if len(pts) else None
    j = judge_rec(per["A63"], per["v27"])
    rec = {"format": "c10-prereg63 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": uses, "e_vectors": vec49, "base_vectors_E_v26_seam": vec, "p1_sets": sets,
           "cfg": cfg, "wP": wP, "k": k, "w_multi": W_MULTI, "w_rank": W_RANK,
           "seam_v27_E_identical": True, "seam_json_sha256": st.sha256_lf(SEAM_JSON),
           "prep_json_sha256": st.sha256_lf(PREP_JSON), "git_head": L.git_head(),
           "means": {n: float(np.mean(v)) for n, v in per.items()}, "per_pool": per, "by_k": byk,
           "A63_minus_v27": {x: j[x] for x in ("mean", "ci95", "pools_higher")},
           "recorded": {"A_only_minus_v27": judge_rec(per["A_only"], per["v27"]),
                        "BC_only_minus_v27": judge_rec(per["BC_only"], per["v27"]),
                        "true_candidate_lacks_query_colour": aff},
           "rule": j["rule"], "adopt": j["adopt"], "peak_rss_mb": peak_mb(), "seconds": round(time.time() - t0, 1)}
    for x in ("A_only_minus_v27", "BC_only_minus_v27"):
        rec["recorded"][x].pop("adopt")
        rec["recorded"][x].pop("rule")
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "A63_minus_v27", "by_k", "recorded", "adopt")}, indent=1))
    return 0


# ================================================================ apply / assemble
def official_terms(src, pool):
    """p43_eval.official_rec, with the terms computed twice: frozen fill (v2-7) and fixed fill (A63)."""
    import p35_apply as pa
    cands, qs = pa.read_pool(src, pool)
    items, cand_ids, cand_rows, q_rows, qids = L.pool_items(cands, qs)
    if not all(c.startswith(pa.PREFIX[pool] + "p_") for c in cand_ids):
        raise SystemExit("candidate id prefix")
    fit = sim.Store(L.FEATS, load=())
    pst = pa.PoolStore(items, fit.csr("resp").vocab)
    ldas, lm = {}, {}
    for c in ("B", "W"):
        ld = sim.LdaData(L.FEATS, c, fit)
        lm[c] = ld.fit_excluding(fit, set())
        ldas[c] = pa.PoolLda(pst, ld.cols)
    want = ("O", "JO", "S2", "L", "PA")
    rec = {"cand_rows": cand_rows, "q_rows": q_rows}
    rec["T"] = sim.pool_terms(pst, rec, ldas, lm, None, want=want)
    fix = {"cand_rows": cand_rows, "q_rows": q_rows}
    with fixed_fill():
        fix["T"] = sim.pool_terms(pst, fix, ldas, lm, None, want=want)
    return rec, fix, items, cand_ids, qids


def lf(b):
    return b.replace(b"\r\n", b"\n")


def cmd_apply(a):
    t0 = time.time()
    ev = read_json(a.eval)
    if not ev["adopt"]:
        raise SystemExit("eval.json does not adopt A63: keep v2-7")
    seam = read_json(SEAM_JSON)
    mods = load_multi(seam["p1multi"]["sha256"])
    base = E45.v26()
    cfg, wP, k = base["cfg"], base["wP"], base["k"]
    cs = base["tune"]["c_star"]
    rec, fix, items, cand_ids, qids = official_terms(a.src, a.pool)
    name = {"public": "pub2", "private": "priv2"}[a.pool]
    colour = np.array([c for _s, c in items])
    # v2-7 exactly as prereg49's apply
    Pset, Lset = P39E.load_set(name), P39E.load_set("L")
    if len(Pset["games"]) != len(items) or any(g["color"] != c for g, (_s, c) in zip(Pset["games"], items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    models = P39E.fit_models(Lset)
    del Lset
    if P39E.models_sha(models) != base["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    z, has = P39E.p1_terms(Pset["X"], Pset["usable"], colour, rec["cand_rows"], rec["q_rows"], models,
                           E43.P1_PW)[E43.P1_VARIANT]
    del models, Pset
    path = P49.c49_file(a.pool)
    vec49 = P49.check_c49_file(path, a.pool, cs)
    e = E43.pool_emb(path, items, vec49["ckpt_sha256"])
    F27 = BK.fuse_cfg(rec, BK.e_term(rec, e, cfg), cfg)
    data27 = P39E.top5_bytes(qids, st.top5(st.csls(F27 + wP * z, k), cand_ids))
    with open(os.path.join(V27_APPLY, a.pool, "top5.csv"), "rb") as f:
        if lf(f.read()) != lf(data27):
            raise SystemExit("seam: v2-7 rebuilt on %s differs from results/prereg49/apply/%s/top5.csv"
                             % (a.pool, a.pool))
    # A63
    games, means, counts, usable = load_p1_full(name)
    if len(games) != len(items) or any(g["color"] != c for g, (_s, c) in zip(games, items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    Zp = project_multi(means, mods)
    q09, s25 = load_models()
    zm = z_multi_of(Zp, usable, rec["cand_rows"], rec["q_rows"])
    zr = s25_rank_compat_subset(s25, q09, means, counts, usable, rec["cand_rows"], rec["q_rows"])[0]
    Ffix = BK.fuse_cfg(fix, BK.e_term(fix, e, cfg), cfg)
    data = P39E.top5_bytes(qids, st.top5(st.csls(Ffix + W_MULTI * zm + W_RANK * zr, k), cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    r27 = [r.split(b",") for r in data27.split(b"\n")[1:] if r]
    r63 = [r.split(b",") for r in data.split(b"\n")[1:] if r]
    rep = {"format": "c10-prereg63 apply v1", "pool": a.pool, "vectors": vec49, "cfg": cfg, "k": k,
           "w_multi": W_MULTI, "w_rank": W_RANK, "v27_top5_equals_prereg49_apply": True, "questions": len(qids),
           "candidates": len(cand_ids), "games": len(items), "games_without_P1_gemini_loader": int((~usable).sum()),
           "questions_without_P1": int((~np.asarray(has)).sum()), "eval_sha256": st.sha256_lf(a.eval),
           "top5_sha256": L.sha256_file(out), "rows_differ_from_v27": int(sum(x != y for x, y in zip(r63, r27))),
           "top1_differs_from_v27": int(sum(x[1] != y[1] for x, y in zip(r63, r27))),
           "peak_rss_mb": peak_mb(), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    rc = P49.cmd_assemble(a)
    path = os.path.join(a.out, "submission_player.csv")
    with open(path, "rb") as f:
        mine = lf(f.read()).split(b"\n")
    with open(os.path.join(V27_APPLY, "submission_player.csv"), "rb") as f:
        v27 = lf(f.read()).split(b"\n")
    orig = lambda rows: [r for r in rows if r.startswith((b"pub_", b"priv_"))]  # noqa: E731
    new = lambda rows: [r for r in rows if r.startswith((b"pub2_", b"priv2_"))]  # noqa: E731
    rep = read_json(os.path.join(a.out, "assemble_report.json"))
    rep["original_800_identical_to_v27"] = orig(mine) == orig(v27) and len(orig(mine)) == 800
    rep["header_identical_to_v27"] = mine[0] == v27[0]
    t1 = {r.split(b",")[0]: r.split(b",")[1] for r in new(v27)}
    rep["new_top1_changed_vs_v27"] = {
        p: sum(1 for r in new(mine) if r.startswith(p.encode()) and t1[r.split(b",")[0]] != r.split(b",")[1])
        for p in ("pub2_", "priv2_")}
    L.write_json(os.path.join(a.out, "assemble_report.json"), rep)
    print(json.dumps({x: rep[x] for x in ("ok", "rows", "sha256", "original_800_identical_to_v27",
                                          "new_top1_changed_vs_v27")}, indent=1))
    return 0 if (rc == 0 and rep["original_800_identical_to_v27"]) else 1


def cmd_check_subset(a):
    """NOTES.md 11: on the D pools, the rank z of the subset call vs the full-set call (Gemini's)."""
    t0 = time.time()
    gD, mD, cD, uD = load_p1_full("D")
    pos = {int(g["row"]): i for i, g in enumerate(gD)}
    q09, s25 = load_models()
    out = []
    for p in EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS):
        cg = [[pos[int(r)] for r in g] for g in p["cand_rows"]]
        qg = [[pos[int(r)] for r in g] for g in p["q_rows"]]
        zf, rf = s25_rank_compat_term(s25, q09, mD, cD, uD, cg, qg)
        zs, rs = s25_rank_compat_subset(s25, q09, mD, cD, uD, cg, qg)
        out.append({"seed": p["seed"], "z_identical": bool(np.array_equal(zf, zs)),
                    "z_max_abs": float(np.abs(zf - zs).max()), "raw_max_abs": float(np.abs(rf - rs).max())})
        print(json.dumps(out[-1]), flush=True)
    rec = {"format": "c10-prereg63 subset check v1", "pools": out, "all_identical": all(x["z_identical"] for x in out),
           "max_abs": max(x["z_max_abs"] for x in out), "git_head": L.git_head(), "peak_rss_mb": peak_mb(),
           "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(HERE, "subset_check.json"), rec)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    s = sub.add_parser("seam")
    s.add_argument("--out", default=SEAM_JSON)
    pe = sub.add_parser("prep-e")
    pe.add_argument("--out", default=PREP_JSON)
    pe.add_argument("--part", required=True, choices=["terms", "zbc"])
    sub.add_parser("e-use")
    sub.add_parser("check-subset")
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    j.add_argument("--out", default=EVAL_JSON)
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--eval", default=EVAL_JSON)
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=P49.V1_SUBMISSION)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd == "e-use":
        u = e_uses_so_far()
        print(json.dumps(u))
        print(u["count"] + 1)
        return 0
    if a.cmd != "assemble":
        BK.polite()
        BK.wait_for_ram(need_mb=RAM_MIN_MB)
        _WD["s"] = rss_watchdog()
    return {"fit": cmd_fit, "seam": cmd_seam, "prep-e": cmd_prep_e, "check-subset": cmd_check_subset, "judge": cmd_judge, "apply": cmd_apply,
            "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
