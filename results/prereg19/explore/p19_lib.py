"""prereg19 exploration (competition 2): pure numpy / scipy parts of PLAN.md (question-level set model S19).

Player split (A / H), the question sampler (3.2), the held-out question set (3.4), the numpy float64 forward of S19
(3.6), question pooling over usable games, rank-balanced scores and the gates (6). No torch here.
"""

import hashlib

import numpy as np
from scipy import sparse

import p09_forward as FW

L = 10
KS = (5, 10, 15, 20)
COLOURS = ("B", "W")
BATCH = 128
STEPS = 15000
EVAL_EVERY = 500
HOLD_PER = 100
HOLD_SEED = 19019
SEEDS = (0, 1, 2)
SAMPLER_BASE = 19
D_IN = FW.X_DIM
HIDDEN = 256
H_DIM = 64
RHO = 32
DROP_IN = 0.1
DROP = 0.3
LR = 1e-3
WD = 1e-2
PCT_START = 0.1
THREADS = 8
SCALE = 4.0
GATE = 0.010
TOL = 1e-12
TORCH_TOL = 1e-4
SMOKE = {"steps": 300, "eval_every": 100, "hold_per": 5}
PARAM_NAMES = ("W1", "b1", "W2", "b2", "R1", "c1", "R2", "c2", "t0", "dt")
E1 = float(np.exp(-1.0))


# ---------------------------------------------------------------- players (PLAN.md 2)
def holdout_bit(key):
    """1 = H (held out), 0 = A (training); key = '<rank>:<player_id>'."""
    h = hashlib.sha256(f"c10-prereg19|holdout|{key}".encode("utf-8")).hexdigest()
    return int(int(h[:8], 16) % 10 == 0)


class PlayerPool:
    """Rows of one player role grouped by player and colour, with the eligible players per (n, level, colour)."""

    def __init__(self, keys, levels, colours, rows):
        rows = np.asarray(rows, dtype=np.int64)
        keys = np.asarray(keys).astype(str)
        levels = np.asarray(levels, dtype=np.int64)
        colours = np.asarray(colours).astype(str)
        players = sorted(set(keys[rows].tolist()))
        idx = {k: i for i, k in enumerate(players)}
        self.players = players
        self.level = np.full(len(players), -1, dtype=np.int64)
        lists = [{c: [] for c in COLOURS} for _ in players]
        for g in rows:
            i = idx[keys[g]]
            lv = int(levels[g])
            if self.level[i] not in (-1, lv):
                raise ValueError(f"player {keys[g]} has two levels")
            self.level[i] = lv
            c = str(colours[g])
            if c not in COLOURS:
                raise ValueError(f"row {g}: colour {c!r}")
            lists[i][c].append(int(g))
        self.games = [{c: np.array(sorted(d[c]), dtype=np.int64) for c in COLOURS} for d in lists]
        self.eligible = {}
        for n in KS:
            self.eligible[n] = [[np.array([i for i in range(len(players)) if self.level[i] == k
                                           and len(self.games[i][c]) >= n], dtype=np.int64)
                                 for c in COLOURS] for k in range(L)]

    def counts(self):
        """Eligible players per n, level, colour (record)."""
        return {str(n): [[int(len(self.eligible[n][k][ci])) for ci in range(2)] for k in range(L)] for n in KS}


def sample_question(rng, pool, k, n, stats):
    """One question of level k with n same-colour games (PLAN.md 3.2) -> (rows, colour index) or None."""
    ci = int(rng.integers(2))
    el = pool.eligible[n][k][ci]
    if len(el) == 0:
        stats["colour_switched"] += 1
        ci = 1 - ci
        el = pool.eligible[n][k][ci]
        if len(el) == 0:
            return None
    p = int(el[rng.integers(len(el))])
    rows = rng.choice(pool.games[p][COLOURS[ci]], size=n, replace=False)
    return np.asarray(rows, dtype=np.int64), ci


def sample_batch(rng, pool, batch, stats):
    """A training batch: one n for the batch, levels uniform -> (idx [batch, n] rows, y [batch], n)."""
    n = KS[int(rng.integers(len(KS)))]
    idx = np.zeros((batch, n), dtype=np.int64)
    y = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        while True:
            k = int(rng.integers(L))
            q = sample_question(rng, pool, k, n, stats)
            if q is not None:
                break
            stats["level_redrawn"] += 1
        idx[b] = q[0]
        y[b] = k
    stats["questions"] += batch
    return idx, y, n


def holdout_set(pool, per, seed):
    """Fixed held-out questions: for each n, for each level, `per` questions (PLAN.md 3.4) -> {n: (idx, y)}."""
    rng = np.random.default_rng(seed)
    stats = {"colour_switched": 0, "level_missing": 0}
    out = {}
    for n in KS:
        idx, ys = [], []
        for k in range(L):
            for _ in range(per):
                q = sample_question(rng, pool, k, n, stats)
                if q is None:
                    stats["level_missing"] += 1
                    continue
                idx.append(q[0])
                ys.append(k)
        out[n] = (np.array(idx, dtype=np.int64).reshape(-1, n), np.array(ys, dtype=np.int64))
    return out, stats


# ---------------------------------------------------------------- numpy forward (PLAN.md 3.6)
def p64(params):
    return {k: np.asarray(params[k]).astype(np.float64) for k in PARAM_NAMES}


def h_np(par, X):
    """Per-game 64-dim representation; X float64 [N, 1540] standardised."""
    a = FW.gelu(X @ par["W1"].T + par["b1"])
    return FW.gelu(a @ par["W2"].T + par["b2"])


def head_np(par, V):
    """Pooled representation [Q, 64] -> log P(y = k) [Q, 10]."""
    f = FW.gelu(V @ par["R1"].T + par["c1"]) @ par["R2"].T + par["c2"]
    return FW.clm_logp(f.reshape(-1, 1), par["t0"], par["dt"])


def set_logp_np(par, X_games):
    """Questions given as [Q, n, 1540] float64 -> lp [Q, 10] (mean of h over the question's games)."""
    Q, n, d = X_games.shape
    H = h_np(par, X_games.reshape(Q * n, d)).reshape(Q, n, -1)
    return head_np(par, H.mean(axis=1))


def lse_mean(lps):
    return FW.lse_mean(lps)


# ---------------------------------------------------------------- question pooling over usable games
def usable_groups(S, usable):
    """Rows of a pooling matrix S (csr [Q, N]) -> per question the usable column indices (order kept)."""
    S = S.tocsr()
    usable = np.asarray(usable, dtype=bool)
    return [[int(c) for c in S.indices[S.indptr[q]:S.indptr[q + 1]] if usable[c]] for q in range(S.shape[0])]


def mean_matrix(groups, n_cols):
    """= p02_rank_fuse.pool_matrix (sparse averaging matrix; empty row for an empty group)."""
    rows, cols, vals = [], [], []
    for q, g in enumerate(groups):
        if len(g):
            rows.extend([q] * len(g))
            cols.extend(g)
            vals.extend([1.0 / len(g)] * len(g))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), n_cols))


def set_term(par_list, H_list, groups, n_cols, scale=SCALE):
    """E_SET = scale * lse_mean over seeds of head(mean of usable h) per question; 0 without a usable game.
    H_list[s]: [n_cols, 64] per-game h of seed s (rows of unusable games are ignored by the groups)."""
    M = mean_matrix(groups, n_cols)
    has = np.array([len(g) > 0 for g in groups])
    lps = [head_np(par, np.asarray(M @ H)) for par, H in zip(par_list, H_list)]
    lp = lse_mean(lps)
    E = np.zeros((len(groups), L))
    E[has] = scale * lp[has]
    per_seed = []
    for x in lps:
        e = np.zeros((len(groups), L))
        e[has] = scale * x[has]
        per_seed.append(e)
    return E, per_seed, lp, has


# ---------------------------------------------------------------- scores and gates (PLAN.md 6)
def official_score(pred, y):
    d = np.abs(np.asarray(pred) - np.asarray(y))
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def rank_balanced(sc, y):
    sc, y = np.asarray(sc, dtype=np.float64), np.asarray(y)
    lv = [float(sc[y == k].mean()) for k in range(L) if (y == k).any()]
    return float(np.mean(lv)) if lv else None


def per_level(sc, y):
    sc, y = np.asarray(sc, dtype=np.float64), np.asarray(y)
    return [float(sc[y == k].mean()) if (y == k).any() else None for k in range(L)]


def gates(inner_diff, inner_ci, mono_diff, mono_ci, by_colour):
    g1 = inner_diff >= GATE
    g2 = inner_ci[0] > 0
    g4 = (mono_ci[0] is not None and mono_ci[0] > 0
          and all(by_colour.get(c) is not None and by_colour[c] >= -TOL for c in COLOURS))
    return {"G1": {"threshold": GATE, "delta": inner_diff, "passed": bool(g1)},
            "G2": {"ci95": inner_ci, "passed": bool(g2)},
            "G4": {"mono_diff": mono_diff, "mono_ci95": mono_ci, "by_colour": by_colour, "passed": bool(g4)},
            "passed": bool(g1 and g2 and g4)}
