"""prereg41 exploration (competition 2): the per-term (n/12)^gamma_t fusion of PLAN.md 1 (pure numpy / scipy).

    z = e^a [ s^g0 (m_C5 + B beta) + sum_j w_j s^g_j E_j ],  s = n / 12
    theta = [a, g0, w_1..w_J, g_1..g_J, beta (L-1)]; a (-4, 4), every g (-1, 2), w (0, 4); penalty 0.01 ||beta||^2;
    level-weighted NLL; L-BFGS-B, maxiter 500 (all as tools/p04_rank_fuse). A term j that is not 'free' uses g0
    (its own g_j is pinned to 0 and unused): free all = Q41, free only SET = Q41S (record only).
Starts: p04's two (a = 0 / 1, the rest 0) and the fold's own Q25 parameters (g_j = g0); the lowest loss wins.
"""

import math
import os
import sys

import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if os.path.join(REPO, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "tools"))

import p02_rank_fuse as rf  # noqa: E402
import p04_rank_fuse as p4  # noqa: E402

L = p4.L
B = p4.B
FORMAT = "c10-prereg41 per-term gamma v1"


def unpack(th, J):
    th = np.asarray(th, dtype=np.float64)
    return th[0], th[1], th[2:2 + J], th[2 + J:2 + 2 * J], th[2 + 2 * J:]


def eff_gammas(g0, g, free):
    return np.where(np.asarray(free, dtype=bool), g, g0)


def logits(th, m, Es, n, free):
    J = len(Es)
    a, g0, w, g, beta = unpack(th, J)
    ge = eff_gammas(g0, g, free)
    ls = np.log(n / 12.0)
    ea = math.exp(a)
    T0 = ea * np.exp(g0 * ls)
    z = T0[:, None] * (m + B @ beta)
    for j in range(J):
        z = z + (w[j] * ea * np.exp(ge[j] * ls))[:, None] * Es[j]
    return z


def objective(th, m, Es, n, y, wq, free):
    J = len(Es)
    a, g0, w, g, beta = unpack(th, J)
    free = np.asarray(free, dtype=bool)
    ge = eff_gammas(g0, g, free)
    lsn = np.log(n / 12.0)
    ea = math.exp(a)
    T0 = ea * np.exp(g0 * lsn)
    u0 = m + B @ beta
    Tj = [ea * np.exp(ge[j] * lsn) for j in range(J)]
    z = T0[:, None] * u0
    for j in range(J):
        z = z + (w[j] * Tj[j])[:, None] * Es[j]
    lsm = rf.log_softmax(z)
    Q = len(y)
    loss = -np.sum(wq * lsm[np.arange(Q), y]) + rf.PENALTY * float(beta @ beta)
    G = np.exp(lsm)
    G[np.arange(Q), y] -= 1.0
    G *= wq[:, None]
    grad = np.zeros_like(np.asarray(th, dtype=np.float64))
    grad[0] = np.sum(G * z)
    gu0 = np.sum(G * u0, axis=1)
    grad[1] = np.sum(lsn * T0 * gu0)
    for j in range(J):
        ge_j = np.sum(G * Es[j], axis=1)
        grad[2 + j] = np.sum(Tj[j] * ge_j)
        dg = w[j] * np.sum(lsn * Tj[j] * ge_j)
        if free[j]:
            grad[2 + J + j] = dg
        else:
            grad[1] += dg
    grad[2 + 2 * J:] = B.T @ (T0[:, None] * G).sum(0) + 2 * rf.PENALTY * beta
    return loss, grad


def bounds(J, free):
    gb = rf.BOUNDS["gamma"]
    return ([rf.BOUNDS["a"], gb] + [rf.BOUNDS["w"]] * J
            + [gb if f else (0.0, 0.0) for f in free] + [(None, None)] * (L - 1))


def from_base(th_base, J, free):
    """Q25's (p04) theta [a, g, w, beta] -> this layout with g_j = g0 for free terms (0 for pinned ones)."""
    a, g, w, beta = p4._unpack(np.asarray(th_base, dtype=np.float64), J)
    gj = np.where(np.asarray(free, dtype=bool), g, 0.0)
    return np.concatenate([[a, g], w, gj, beta])


def fit_record(m, Es, n, y, free, warm):
    J = len(Es)
    wq = rf.level_weights(y)
    bnds = bounds(J, free)
    starts = []
    for a0 in rf.STARTS:
        x0 = np.zeros(2 + 2 * J + L - 1)
        x0[0] = a0
        starts.append(("a=%g" % a0, x0))
    starts.append(("q25_fold", np.asarray(warm, dtype=np.float64)))
    best, runs = None, []
    for name, x0 in starts:
        r = optimize.minimize(objective, x0, args=(m, Es, n, y, wq, free), jac=True, method="L-BFGS-B",
                              bounds=bnds, options={"maxiter": rf.MAXITER})
        runs.append({"start": name, "loss": float(r.fun), "nit": int(r.nit), "success": bool(r.success),
                     "message": str(r.message)})
        if best is None or r.fun < best[1].fun:
            best = (name, r)
    return best[1].x, {"loss": float(best[1].fun), "winner": best[0], "starts": runs}


def base_loss(th_base, m, Es, n, y):
    """Q25's fitted objective value in the same fold (record only)."""
    return float(p4.gen_obj(np.asarray(th_base, dtype=np.float64), m, Es, n, y, rf.level_weights(y))[0])


def oof_record(Xf, Xe, mf, me, Ef, Ee, base_folds, free):
    """5-fold player OOF: fold f's parameters from the other folds' fit questions only (PLAN.md 1)."""
    J = len(Ef)
    P = np.zeros((len(Xe), L))
    folds = []
    for f in range(5):
        ev = Xe.fold == f
        tr = Xf.fold != f
        bf = [b for b in base_folds if b["fold"] == f][0]
        warm = from_base(bf["theta"], J, free)
        sub = (mf[tr], [E[tr] for E in Ef], Xf.n[tr], Xf.y[tr])
        th, rec = fit_record(*sub, free, warm)
        rec["q25_loss_same_fold"] = base_loss(bf["theta"], *sub)
        P[ev] = rf.softmax(logits(th, me[ev], [E[ev] for E in Ee], Xe.n[ev], free))
        rec.update({"fold": f, "theta": [float(x) for x in th]})
        folds.append(rec)
    return P, folds


def fold_predict(folds, m, Es, n, fold, free):
    P = np.full((len(n), L), np.nan)
    for rec in folds:
        ev = fold == rec["fold"]
        if ev.any():
            P[ev] = rf.softmax(logits(rec["theta"], m[ev], [E[ev] for E in Es], n[ev], free))
    return P


def params(th, terms, free):
    J = len(terms)
    a, g0, w, g, beta = unpack(th, J)
    ge = eff_gammas(g0, g, free)
    return {"a": float(a), "gamma0": float(g0), "w": {t: float(x) for t, x in zip(terms, w)},
            "gamma": {t: float(x) for t, x in zip(terms, ge)}, "free": {t: bool(f) for t, f in zip(terms, free)},
            "w_at_bound": {t: bool(x <= 1e-8 or x >= 4.0 - 1e-8) for t, x in zip(terms, w)},
            "gamma_at_bound": {t: bool(x <= -1.0 + 1e-8 or x >= 2.0 - 1e-8) for t, x in zip(terms, ge)}}
