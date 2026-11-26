"""prereg53 exploration (competition 2): the decision-level calibration of PLAN.md 1 and the stage-0b stack of
PLAN.md 2 (pure numpy / scipy; no repository data is read here).

Calibration (11 parameters theta = [delta_0..delta_9, t]):
    l = log(max(p, 1e-300));  p' = softmax(exp(t) * l + delta);  decision = rule B on p'
    PLAN.md rev 1: maximise H(theta) = sum_q w_q S(ruleB(p'_q), y_q) - LAMBDA ||delta||^2 (w = level_weights, i.e.
    the rank-balanced official score) by a deterministic pattern search from theta = 0: step sizes STEPS, coordinates
    in the order delta_0..delta_9, t, each coordinate tries +step then -step and takes the better strict improvement
    (> IMPROVE_EPS), sweeps until no move (at most MAX_SWEEPS per step). delta in [-3, 3], t in [ln 0.25, ln 4].
Stage-0b stack (PLAN.md 2): z = a * lp + b, b_0 = 0; level-weighted NLL; L-BFGS-B from a = 1, b = 0, maxiter 1000.
"""

import hashlib
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

L = rf.N_LEVELS
E1 = rf.E1
STEPS = (0.2, 0.1, 0.05, 0.02)
MAX_SWEEPS = 10
IMPROVE_EPS = 1e-12
LAMBDA = 0.001
DELTA_BOUND = 3.0
T_BOUNDS = (math.log(0.25), math.log(4.0))
STACK_MAXITER = 1000
P_FLOOR = 1e-300
GATE_0B = 0.003
SALT_0B = "c10-prereg53|0b|"
SALT_0B_SUB = "c10-prereg53|0b-sub|"
N_PARAMS = L + 1


def score_matrix():
    """R[k, j] = official score of guessing k when the truth is j (1, e^-1 for neighbours, else 0)."""
    R = np.zeros((L, L))
    for k in range(L):
        R[k, k] = 1.0
        if k > 0:
            R[k, k - 1] = E1
        if k < L - 1:
            R[k, k + 1] = E1
    return R


R = score_matrix()


def log_p(P):
    return np.log(np.maximum(np.asarray(P, dtype=np.float64), P_FLOOR))


# ---------------------------------------------------------------- calibration (PLAN.md 1)
def calibrate(theta, ell):
    """Calibrated probabilities p' [Q, 10] from l = log p."""
    theta = np.asarray(theta, dtype=np.float64)
    return rf.softmax(math.exp(theta[L]) * ell + theta[:L])


def bounds(variant="full"):
    db = (-DELTA_BOUND, DELTA_BOUND) if variant in ("full", "delta") else (0.0, 0.0)
    tb = T_BOUNDS if variant in ("full", "temp") else (0.0, 0.0)
    return [db] * L + [tb]


def hard_objective(theta, ell, y, wq, lam=LAMBDA):
    """PLAN.md rev 1: the rank-balanced official score of rule B on p'(theta) minus LAMBDA ||delta||^2."""
    pc = calibrate(theta, ell)
    sc = rf.official_score(rf.rule_b_np(pc), y)
    d = np.asarray(theta[:L], dtype=np.float64)
    return float(np.sum(wq * sc)) - lam * float(d @ d)


def fit(P, y, variant="full"):
    """theta from probabilities P [Q, 10] and levels y (PLAN.md rev 1: deterministic pattern search on the hard
    objective). variant: full / delta (t = 0) / temp (delta = 0)."""
    ell = log_p(P)
    y = np.asarray(y, dtype=np.int64)
    wq = rf.level_weights(y)
    bnds = bounds(variant)
    free = [i for i, (lo, hi) in enumerate(bnds) if hi > lo]
    th = np.zeros(N_PARAMS)
    best = hard_objective(th, ell, y, wq)
    at0 = best
    evals, moves, sweeps = 1, 0, 0
    for step in STEPS:
        for _ in range(MAX_SWEEPS):
            sweeps += 1
            improved = False
            for i in free:
                cand_best, cand_th = best, None
                for sgn in (1.0, -1.0):
                    x = th.copy()
                    x[i] = min(max(x[i] + sgn * step, bnds[i][0]), bnds[i][1])
                    if x[i] == th[i]:
                        continue
                    v = hard_objective(x, ell, y, wq)
                    evals += 1
                    if v > cand_best + IMPROVE_EPS:
                        cand_best, cand_th = v, x
                if cand_th is not None:
                    th, best = cand_th, cand_best
                    moves += 1
                    improved = True
            if not improved:
                break
    rec = {"variant": variant, "objective_H": best, "objective_H_at_0": at0, "evals": evals, "moves": moves,
           "sweeps": sweeps, "delta": [float(x) for x in th[:L]], "tau": float(math.exp(th[L])), "t": float(th[L]),
           "delta_at_bound": [bool(abs(x) >= DELTA_BOUND - 1e-8) for x in th[:L]],
           "t_at_bound": bool(th[L] <= T_BOUNDS[0] + 1e-8 or th[L] >= T_BOUNDS[1] - 1e-8),
           "fit_hard_rank_balanced_before": rank_balanced_score(rf.rule_b_np(P), y),
           "fit_hard_rank_balanced_after": rank_balanced_score(rf.rule_b_np(calibrate(th, ell)), y),
           "fit_questions": int(len(y))}
    return th, rec


def apply(theta, P):
    return calibrate(theta, log_p(P))


def rank_balanced_score(pred, y):
    sc = rf.official_score(pred, y)
    return float(np.mean([sc[y == k].mean() for k in range(L) if (y == k).any()]))


# ---------------------------------------------------------------- stage 0b (PLAN.md 2)
def player_bit(key, salt):
    return int(hashlib.sha256((salt + key).encode("utf-8")).hexdigest()[:8], 16) % 2


def stack_obj(th, lp, y, wq):
    a, b = th[0], np.concatenate([[0.0], th[1:]])
    z = a * lp + b
    ls = rf.log_softmax(z)
    Q = len(y)
    loss = -float(np.sum(wq * ls[np.arange(Q), y]))
    G = np.exp(ls)
    G[np.arange(Q), y] -= 1.0
    G *= wq[:, None]
    return loss, np.concatenate([[float(np.sum(G * lp))], G.sum(0)[1:]])


def stack_fit(lp, y):
    y = np.asarray(y, dtype=np.int64)
    wq = rf.level_weights(y)
    x0 = np.zeros(L)
    x0[0] = 1.0
    res = optimize.minimize(stack_obj, x0, args=(lp, y, wq), jac=True, method="L-BFGS-B",
                            options={"maxiter": STACK_MAXITER})
    return np.asarray(res.x, dtype=np.float64), {"loss": float(res.fun), "nit": int(res.nit),
                                                  "success": bool(res.success)}


def stack_logp(th, lp):
    return rf.log_softmax(th[0] * lp + np.concatenate([[0.0], th[1:]]))


def cross_fit_stack(lp, y, part):
    """Two-way cross-fit of the stack over part (0 / 1) -> log-probabilities [Q, 10] (each half from the other)."""
    out = np.full(lp.shape, np.nan)
    recs = {}
    for f in (0, 1):
        tr, ev = part != f, part == f
        th, rec = stack_fit(lp[tr], y[tr])
        out[ev] = stack_logp(th, lp[ev])
        recs[str(f)] = {**rec, "a": float(th[0]), "b": [0.0] + [float(x) for x in th[1:]]}
    return out, recs


def stage0b(lp, y, keys, variants=("full", "delta", "temp")):
    """PLAN.md 2: base B0 = 2-fold player cross-fit stack; candidate = B0 + calibration fitted, for each outer fold,
    on the other fold's nested (sub-half cross-fit) base probabilities. -> dict of probabilities and records."""
    y = np.asarray(y, dtype=np.int64)
    fold = np.array([player_bit(k, SALT_0B) for k in keys])
    sub = np.array([player_bit(k, SALT_0B_SUB) for k in keys])
    base_lp, base_rec = cross_fit_stack(lp, y, fold)
    P0 = np.exp(base_lp)
    cand = {v: np.full(lp.shape, np.nan) for v in variants}
    fold_recs = {}
    for f in (0, 1):
        g = fold != f
        nested_lp, nested_rec = cross_fit_stack(lp[g], y[g], sub[g])
        fr = {"nested_stack": nested_rec}
        for v in variants:
            th, rec = fit(np.exp(nested_lp), y[g], v)
            cand[v][fold == f] = apply(th, P0[fold == f])
            fr[v] = rec
        fold_recs[str(f)] = fr
    return {"fold": fold, "sub": sub, "P0": P0, "cand": cand, "base_stack": base_rec, "folds": fold_recs}


def nll(P, y):
    y = np.asarray(y, dtype=np.int64)
    return float(-np.mean(np.log(np.maximum(P[np.arange(len(y)), y], P_FLOOR))))


def verdict_0b(delta):
    return {"delta": delta, "threshold": GATE_0B, "passed": bool(delta >= GATE_0B - 1e-12)}
