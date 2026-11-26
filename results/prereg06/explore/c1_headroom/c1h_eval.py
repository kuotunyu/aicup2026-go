"""Score matrices, fusion, hubness corrections, one-to-one assignment and the one-to-one detector
(c1_headroom exploration). All matrices are questions x candidates in pool order."""
import sys
sys.dont_write_bytecode = True

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

import c1h_lib as L

po = L.po


# ---------------------------------------------------------------- similarity terms
def cos_term(G, rec):
    return po.open_cos(G, rec["cand_rows"], G, rec["q_rows"])[0]


def _weigh_rows(M, idf):
    M = M.copy().tocsr()
    M.data = 1.0 + np.log(M.data)
    M = M.multiply(idf).tocsr()
    norm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) + 1e-12
    return sp.diags(1.0 / norm) @ M


def gamewise_terms(G, rec):
    """Per question game cosine to each candidate profile (p02 weighting, IDF over candidates).
    Returns (cos_games [n_qgames x n_cand], owner question index per game row)."""
    C = (po.indicator(rec["cand_rows"], G.shape[0]) @ G).tocsr()
    C.eliminate_zeros()
    P = C.shape[0]
    df = np.diff(C.tocsc().indptr)
    idf = np.log((1.0 + P) / (1.0 + df)) + 1.0
    rows = [g for q in rec["q_rows"] for g in q]
    qi = np.repeat(np.arange(len(rec["q_rows"])), [len(q) for q in rec["q_rows"]])
    Qg = G[rows]
    cg = np.asarray((_weigh_rows(Qg, idf) @ _weigh_rows(C, idf).T).toarray())
    return cg, qi


def gamewise_logp(G, rec, beta):
    """sum over a question's games of log softmax_c(beta * cos(game, c))."""
    cg, qi = gamewise_terms(G, rec)
    lp = beta * cg - logsumexp(beta * cg, axis=1, keepdims=True)
    out = np.zeros((len(rec["q_rows"]), cg.shape[1]))
    np.add.at(out, qi, lp)
    return out


def knn_term(G, rec, top=1, chunk=250):
    """Game-to-game: for every question game, the mean of the `top` best cosines to each candidate's
    individual games (IDF over all candidate games); averaged over the question's games."""
    crow = [g for c in rec["cand_rows"] for g in c]
    ncand = len(rec["cand_rows"])
    per = len(rec["cand_rows"][0])
    assert all(len(c) == per for c in rec["cand_rows"])
    Cg = G[crow].tocsr()
    df = np.diff(Cg.tocsc().indptr)
    idf = np.log((1.0 + Cg.shape[0]) / (1.0 + df)) + 1.0
    Cw = _weigh_rows(Cg, idf).T.tocsc()
    rows = [g for q in rec["q_rows"] for g in q]
    qi = np.repeat(np.arange(len(rec["q_rows"])), [len(q) for q in rec["q_rows"]])
    Qw = _weigh_rows(G[rows], idf)
    best = np.zeros((len(rows), ncand))
    for s in range(0, len(rows), chunk):
        S = np.asarray((Qw[s:s + chunk] @ Cw).toarray()).reshape(-1, ncand, per)
        if top == 1:
            best[s:s + chunk] = S.max(2)
        else:
            best[s:s + chunk] = -np.sort(-S, axis=2)[:, :, :top].mean(2)
    out = np.zeros((len(rec["q_rows"]), ncand))
    np.add.at(out, qi, best)
    cnt = np.bincount(qi, minlength=len(rec["q_rows"]))[:, None]
    return out / cnt


def style_ll(X, rec, shrink=1.0):
    """Gaussian naive-Bayes log-likelihood of the question's mean style vector under each candidate:
    mean of k games ~ N(mu_c, s2_w / k + s2_w / n_c) per feature, s2_w = pooled within-candidate
    per-game variance (candidate games of the pool only)."""
    cand = [X[c] for c in rec["cand_rows"]]
    mu = np.vstack([a.mean(0) for a in cand])
    s2w = np.mean(np.vstack([a.var(0) for a in cand]), 0) * shrink + 1e-6
    out = np.zeros((len(rec["q_rows"]), len(cand)))
    for i, q in enumerate(rec["q_rows"]):
        xq = X[q].mean(0)
        var = s2w / len(q) + s2w / len(cand[0])
        out[i] = -0.5 * (((xq[None, :] - mu) ** 2) / var[None, :]).sum(1)
    return out


# ---------------------------------------------------------------- fusion and corrections
def fuse(terms, weights):
    s = None
    for t, w in zip(terms, weights):
        if w == 0:
            continue
        z = L.zrows(t) * w
        s = z if s is None else s + z
    return s


def csls(S, K=10):
    """CSLS (Lample et al. 2018): 2 S - r_c - r_q; r_c = mean of the K largest S over questions."""
    rc = -np.sort(-S, axis=0)[:K].mean(0)
    rq = -np.sort(-S, axis=1)[:, :K].mean(1)
    return 2 * S - rc[None, :] - rq[:, None]


def col_center(S, a=1.0):
    return S - a * S.mean(0, keepdims=True)


def logsoft_rows(S, tau):
    Z = S / tau
    return Z - logsumexp(Z, axis=1, keepdims=True)


def sinkhorn(S, tau, iters=50, col_mass=None, lam=None):
    """Log-domain Sinkhorn on exp(S / tau). Rows sum to 1. Columns sum to col_mass (default
    n_rows / n_cols = 1 here) exactly (lam None, balanced) or softly (unbalanced, KL weight lam on the
    column marginal: lam -> 0 keeps the row softmax, lam -> inf is balanced). Returns log plan."""
    n, m = S.shape
    logK = S / tau
    a = np.zeros(n)
    b = np.zeros(m)
    logc = np.log(n / m) if col_mass is None else np.log(col_mass)
    rho = 1.0 if lam is None else lam / (lam + 1.0)
    for _ in range(iters):
        a = -logsumexp(logK + b[None, :], axis=1)
        b = rho * (logc - logsumexp(logK + a[:, None], axis=0))
    return logK + a[:, None] + b[None, :]


def hungarian_top1(W):
    """Rows assigned to distinct columns maximising sum W."""
    r, c = linear_sum_assignment(-W)
    a = np.empty(W.shape[0], np.int64)
    a[r] = c
    return a


def with_top1(S_rest, a):
    """Final ranking: assigned candidate first, then the others by S_rest."""
    F = np.array(S_rest, np.float64, copy=True)
    F[np.arange(len(a)), a] = np.inf
    return F


# ---------------------------------------------------------------- one-to-one detector statistics
def detector_stats(S, tau=None):
    """Label-free statistics of a score matrix (questions x candidates)."""
    n, m = S.shape
    top = S.argmax(1)
    cnt = np.bincount(top, minlength=m)
    colbest = S.argmax(0)
    mutual = float(np.mean(colbest[top] == np.arange(n)))
    Wl = logsoft_rows(L.zrows(S), 1.0 if tau is None else tau)
    a = hungarian_top1(Wl)
    agree = float(np.mean(a == top))
    return {"n0": int((cnt == 0).sum()), "n1": int((cnt == 1).sum()), "n2plus": int((cnt >= 2).sum()),
            "max_count": int(cnt.max()), "mutual_nn": mutual, "hungarian_agree": agree}
