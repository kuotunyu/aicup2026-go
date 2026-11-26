"""prereg15 exploration (competition 2): shared numerics for PLAN.md (EM prior shift, quota assignment, fake test pools,
player accounting and bootstrap, per-colour refit of the fusion's a / gamma / beta). numpy / scipy only; the fusion
objective and logits come from tools/p04_rank_fuse.py (imported, never edited). No file is read or written here.
"""

import math
import os
import sys

import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import p02_rank_fuse as rf  # noqa: E402
import p04_rank_fuse as p4  # noqa: E402

L = rf.N_LEVELS
E1 = math.exp(-1)
KS = (5, 10, 15, 20)
PER_K = 100
POOL = PER_K * len(KS)
SCENARIOS = {  # PLAN.md 2.1 (level order D, C, B, A, 1D ... 6D)
    "S0": [1] * 10,
    "S1": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    "S2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "S3": [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
    "S4": [5, 4, 3, 2, 1, 1, 2, 3, 4, 5],
}
SKEWED = ("S1", "S2", "S3", "S4")
N_POOLS = 2000
SEED_BASE = 20261015          # scenario j (S0 = 0 ... S4 = 4) -> default_rng(SEED_BASE + j)
SEED_TAU = 20261014           # the A2 threshold's calibration pools (S0)
TAU_PERCENTILE = 95.0
EM_TOL = 1e-10
EM_MAX_ITER = 1000
CI_ADJ = (0.625, 99.375)      # Bonferroni over 4 arms: two-sided 98.75 %
CI_95 = (2.5, 97.5)


def proportions(name):
    w = np.asarray(SCENARIOS[name], dtype=np.float64)
    return w / w.sum()


def largest_remainder(total, props):
    """Integer counts summing to total; ties in the remainder go to the smaller level index first."""
    props = np.asarray(props, dtype=np.float64)
    raw = total * props / props.sum()
    base = np.floor(raw).astype(np.int64)
    rem = raw - base
    left = int(total - base.sum())
    order = sorted(range(len(props)), key=lambda i: (-rem[i], i))
    for i in order[:left]:
        base[i] += 1
    return base


def strata(y, ks):
    """{(level, k): question indices} of the single-colour questions."""
    y, ks = np.asarray(y), np.asarray(ks)
    return {(lv, k): np.nonzero((y == lv) & (ks == k))[0] for lv in range(L) for k in KS}


def pool_counts(props):
    """[len(KS), L] questions per (k, level) of one pool (largest remainder of 100 x proportions per k)."""
    c = largest_remainder(PER_K, props)
    return np.tile(c, (len(KS), 1))


def sample_pool(rng, st, counts):
    """One fake pool: k outer (KS order), level inner; without replacement within a (level, k) stratum."""
    out = []
    for ki, k in enumerate(KS):
        for lv in range(L):
            c = int(counts[ki, lv])
            if c:
                idx = st[(lv, k)]
                if len(idx) < c:
                    raise ValueError(f"stratum (level {lv}, k {k}) has {len(idx)} questions, {c} needed")
                out.append(rng.choice(idx, c, replace=False))
    return np.concatenate(out)


def sample_pools(seed, st, props, n_pools=N_POOLS):
    rng = np.random.default_rng(seed)
    counts = pool_counts(props)
    return np.stack([sample_pool(rng, st, counts) for _ in range(n_pools)])


def em_prior(P, pi0, tol=EM_TOL, max_iter=EM_MAX_ITER):
    """Saerens et al. (2002) EM on one pool's posteriors P [n, L] trained under pi0 -> (pi_hat, Q adjusted with
    pi_hat, iterations, converged)."""
    P = np.asarray(P, dtype=np.float64)
    pi0 = np.asarray(pi0, dtype=np.float64)
    pi = pi0.copy()
    it, conv = 0, False
    while it < max_iter:
        it += 1
        Q = P * (pi / pi0)
        Q /= Q.sum(1, keepdims=True)
        new = Q.mean(0)
        delta = float(np.max(np.abs(new - pi)))
        pi = new
        if delta < tol:
            conv = True
            break
    return pi, adjust(P, pi, pi0), it, conv


def adjust(P, pi, pi0):
    Q = np.asarray(P, dtype=np.float64) * (np.asarray(pi, dtype=np.float64) / np.asarray(pi0, dtype=np.float64))
    return Q / Q.sum(1, keepdims=True)


def tv(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def expected_scores(P):
    """[n, L]: expected official score of answering level k (rule B maximises it; rf.rule_b_np is its argmax)."""
    P = np.asarray(P, dtype=np.float64)
    left = np.zeros_like(P)
    right = np.zeros_like(P)
    left[:, 1:] = P[:, :-1]
    right[:, :-1] = P[:, 1:]
    return P + E1 * (left + right)


def quota_assign(Q, counts):
    """Answers maximising the summed expected score under exact per-level counts (linear_sum_assignment)."""
    from scipy.optimize import linear_sum_assignment
    counts = np.asarray(counts, dtype=np.int64)
    n = len(Q)
    if int(counts.sum()) != n:
        raise ValueError(f"quota {counts.sum()} != {n} questions")
    cols = np.repeat(np.arange(L), counts)
    cost = -expected_scores(Q)[:, cols]
    rows, c = linear_sum_assignment(cost)
    pred = np.empty(n, dtype=np.int64)
    pred[rows] = cols[c]
    return pred


def level_balanced_mean(P, y):
    """Mean of P within each level, then over the levels present."""
    P, y = np.asarray(P, dtype=np.float64), np.asarray(y)
    per = [P[y == lv].mean(0) for lv in range(L) if (y == lv).any()]
    return np.mean(per, axis=0)


# ---------------------------------------------------------------- player accounting and bootstrap (PLAN.md 2.3)
def boot_levels(boot, level_of):
    """[(level, draws [N_BOOT, members])] of a p02 PlayerBootstrap (players resampled within each level)."""
    out = []
    for d in boot.draws:
        lv = int(level_of[d[0, 0]])
        if not np.all(level_of[d] == lv):
            raise ValueError("a bootstrap draw mixes levels")
        out.append((lv, d))
    return out


def accumulate(player, diff, n_players):
    """(D_p, N_p): summed difference and appearances per player (diff and player are flat over every pool)."""
    D = np.bincount(player, weights=diff, minlength=n_players).astype(np.float64)
    N = np.bincount(player, minlength=n_players).astype(np.float64)
    return D, N


def delta_point(D, N, level_of, props):
    """sum_l w_l (sum_{p in l} D_p / sum_{p in l} N_p) over the levels with weight > 0."""
    out = 0.0
    for lv in range(L):
        if props[lv] > 0:
            m = level_of == lv
            out += props[lv] * D[m].sum() / N[m].sum()
    return float(out)


def delta_boot(D, N, bl, props):
    """The same estimate under every bootstrap draw (paired across arms / scenarios: same draws)."""
    out = 0.0
    for lv, d in bl:
        if props[lv] > 0:
            out = out + props[lv] * D[d].sum(1) / N[d].sum(1)
    return np.asarray(out, dtype=np.float64)


def pct(draws, q):
    return [float(np.percentile(draws, q[0])), float(np.percentile(draws, q[1]))]


# ---------------------------------------------------------------- per-colour refit of a, gamma, beta (PLAN.md 3)
def _full(th3, w):
    return np.concatenate([th3[:2], w, th3[2:]])


def fixed_w_obj(th3, w, m, Es, n, y, wq):
    J = len(Es)
    loss, g = p4.gen_obj(_full(th3, w), m, Es, n, y, wq)
    return loss, np.concatenate([g[:2], g[2 + J:]])


def fit_fixed_w(m, Es, n, y, w):
    """a, gamma, beta with the term weights w fixed: p4.gen_obj (level-balanced NLL + PENALTY ||beta||^2),
    bounds a / gamma as rf.BOUNDS, starts a in rf.STARTS (gamma, beta 0), L-BFGS-B maxiter rf.MAXITER, the start with
    the smaller loss -> (full theta [a, gamma, w, beta], record)."""
    w = np.asarray(w, dtype=np.float64)
    wq = rf.level_weights(y)
    bnds = [rf.BOUNDS["a"], rf.BOUNDS["gamma"]] + [(None, None)] * (L - 1)
    best, runs = None, []
    for a0 in rf.STARTS:
        x0 = np.zeros(2 + L - 1)
        x0[0] = a0
        r = optimize.minimize(fixed_w_obj, x0, args=(w, m, Es, n, y, wq), jac=True, method="L-BFGS-B",
                              bounds=bnds, options={"maxiter": rf.MAXITER})
        runs.append({"start_a": a0, "loss": float(r.fun), "nit": int(r.nit), "success": bool(r.success),
                     "message": str(r.message)})
        if best is None or r.fun < best.fun:
            best = r
    return _full(best.x, w), {"loss": float(best.fun), "starts": runs}


def predict(theta, m, Es, n):
    z, _ = p4.gen_logits(np.asarray(theta, dtype=np.float64), m, Es, n)
    return rf.softmax(z)
