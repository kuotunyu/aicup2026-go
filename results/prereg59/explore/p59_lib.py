"""prereg59 (competition 2): opponent-identity term, second version (PLAN.md A, B, C). numpy only; prereg58's
p58_lib is imported (frozen), never edited.

  A  classes = S0's head rows (logit tau1 * cos) + one prototype per unseen labelled player (logit tau2 * cos + b);
     q_g[s] = soft-max over the union, summed by the class's rank; a kernel player's own prototype is left out
     class_parts() once per (tau1 | tau2), combine() per (tau1, tau2, b) - exact, no approximation
  B  M[h, c][r][s] = P(opponent's class rank s | player's rank r, HD group h, player's colour c)   table_counts() / table()
  C  l_g(r) = log(sum_s q_g[s] M[h_g, c_g][r][s] + 1e-9);  question term = SCALE * mean over its games of l_g(r)
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P58 = os.path.join(REPO, "results", "prereg58", "explore")
if P58 not in sys.path:
    sys.path.insert(0, P58)
import p58_lib as K58  # noqa: E402

DATA = os.path.join(REPO, "data", "prereg59")
TAG = K58.TAG
OWN_D = os.path.join(REPO, "data", "prereg43", "emb_S0", TAG, "store_D.npz")      # S0 player-side vectors (existing)
OWN_E = os.path.join(REPO, "data", "prereg43", "emb_S0", TAG, "store_E.npz")
OWN_VAL = os.path.join(DATA, "emb_own_val", TAG, "rows_val.npz")                 # embed_val.sh (GPU, later)
VAL_ROWS = os.path.join(DATA, "val_rows.npy")                                    # rows of the prereg49 move cache
CACHE49 = os.path.join(REPO, "data", "prereg49", "cache")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
PROTO = os.path.join(DATA, "prototypes.npz")
KERNEL = os.path.join(HERE, "kernel.json")
SEAM = os.path.join(HERE, "seam.json")
EVAL = os.path.join(HERE, "eval.json")
RESULTS = os.path.join(HERE, "RESULTS.md")
STARTED = os.path.join(DATA, "eval_started.json")

L, N_HD, N_COL, EPS, FOLDS = K58.L, K58.N_HD, K58.N_COL, K58.EPS, K58.FOLDS
RANKS = K58.RANKS
TAU1 = (30, 50, 80)                  # PLAN A
TAU2 = (30, 50, 80, 120)
BIAS = (-2, -1, 0, 1, 2)
GRID = tuple(itertools.product(TAU1, TAU2, BIAS))      # selection order = tie order (first wins)
SMOOTH = 0.5                         # add-0.5 on each of the 10 cells of a row of M
SCALE = 10.0                         # PLAN C
W_BOUND = 4.0                        # the fusion's weight limit (tools/p02_rank_fuse.BOUNDS["w"])
guard = K58.guard


def l2(x):
    x = np.asarray(x, np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def prototypes(emb, count, player):
    """One prototype per player = L2(mean of his games' L2 vectors), games with a sample only.
    -> (players sorted [P], prototypes [P, d] float64 unit rows, games used [P]). A player without a usable game is
    left out."""
    emb, count, player = np.asarray(emb), np.asarray(count), np.asarray(player).astype(str)
    ok = count > 0
    u, inv = np.unique(player[ok], return_inverse=True)
    e = l2(emb[ok])
    S = np.zeros((len(u), e.shape[1]))
    np.add.at(S, inv, e)
    n = np.bincount(inv, minlength=len(u))
    return u, l2(S / n[:, None]), n


def _rank_sums(e, Cn, cr, taus, own, block):
    """{tau: (S [G, 10], m [G])} with S[g, s] = sum over classes j of rank s (j != own[g]) of exp(tau * cos_gj - m[g]),
    m[g] = tau * the largest remaining cosine. e: game vectors (normalised here, block by block); Cn: unit class rows;
    own: class index to leave out or -1."""
    G = len(e)
    onehot = np.zeros((len(cr), L))
    onehot[np.arange(len(cr)), cr] = 1.0
    out = {t: (np.zeros((G, L)), np.zeros(G)) for t in taus}
    for s in range(0, G, block):
        C = l2(e[s:s + block]) @ Cn.T
        if own is not None:
            o = own[s:s + block]
            has = np.flatnonzero(o >= 0)
            C[has, o[has]] = -np.inf
        mx = C.max(axis=1)
        if not np.isfinite(mx).all():
            raise ValueError("a game without any class left")
        for t in taus:
            out[t][0][s:s + block] = np.exp(t * (C - mx[:, None])) @ onehot
            out[t][1][s:s + block] = t * mx
    return out


def class_parts(emb, Wn, wr, Pn, pr, own=None, tau1=TAU1, tau2=TAU2, block=4000):
    """The head part for every tau1 and the prototype part for every tau2 (own: [G] prototype index of the game's
    player, -1 for none). -> {"A": {tau1: (S, m)}, "B": {tau2: (S, m)}}"""
    return {"A": _rank_sums(emb, Wn, np.asarray(wr), tau1, None, block),
            "B": _rank_sums(emb, Pn, np.asarray(pr), tau2, own, block)}


def combine(parts, t1, t2, b):
    """q [G, 10] of (tau1, tau2, b): soft-max over head logits tau1 * cos and prototype logits tau2 * cos + b."""
    SA, mA = parts["A"][t1]
    SB, mB = parts["B"][t2]
    m = np.maximum(mA, mB + b)
    q = SA * np.exp(mA - m)[:, None] + SB * np.exp(mB + b - m)[:, None]
    return q / q.sum(axis=1, keepdims=True)


def sub_parts(parts, idx):
    return {k: {t: (S[idx], m[idx]) for t, (S, m) in v.items()} for k, v in parts.items()}


# ---------------------------------------------------------------- the conditional table
def cell(r, h, c):
    return (np.asarray(h) * N_COL + np.asarray(c)) * L + np.asarray(r)


def table_counts(q, r, h, c, mask=None):
    """Soft counts [3, 2, 10, 10]: cell (h, c, r_g, s) += q_g[s] (r_g = the player's rank)."""
    q = np.asarray(q, np.float64)
    idx = cell(r, h, c)
    if mask is not None:
        q, idx = q[mask], idx[mask]
    n = N_HD * N_COL * L
    C = np.stack([np.bincount(idx, weights=q[:, s], minlength=n) for s in range(L)], axis=1)
    return C.reshape(N_HD, N_COL, L, L)


def table(counts):
    M = np.asarray(counts, np.float64) + SMOOTH
    return M / M.sum(axis=3, keepdims=True)


def game_ll(q, M, h, c):
    """l_g(r) = log(sum_s q_g[s] M[h_g, c_g][r][s] + 1e-9) -> [G, 10]."""
    q, h, c = np.asarray(q, np.float64), np.asarray(h), np.asarray(c)
    M = np.asarray(M)
    out = np.zeros((len(q), L))
    for hh in range(N_HD):
        for cc in range(N_COL):
            m = (h == hh) & (c == cc)
            if m.any():
                out[m] = q[m] @ M[hh, cc].T
    return np.log(out + EPS)


def fold_tables(q, r, h, c, fold, ok):
    """[FOLDS, 3, 2, 10, 10]: table f is estimated on the games with a sample of the OTHER folds' players."""
    tot = table_counts(q, r, h, c, ok)
    return np.stack([table(tot - table_counts(q, r, h, c, ok & (fold == f))) for f in range(FOLDS)])


def heldout_ll(q, r, h, c, fold, ok):
    """(mean over the held-out games of l_g(true rank), the same normalised over r, per-fold means of the first)."""
    Mf = fold_tables(q, r, h, c, fold, ok)
    ll, lln = np.zeros(len(r)), np.zeros(len(r))
    for f in range(FOLDS):
        te = np.flatnonzero(ok & (fold == f))
        lg = game_ll(q[te], Mf[f], h[te], c[te])
        ll[te] = lg[np.arange(len(te)), r[te]]
        mx = lg.max(axis=1, keepdims=True)
        lgn = lg - mx - np.log(np.exp(lg - mx).sum(axis=1, keepdims=True))
        lln[te] = lgn[np.arange(len(te)), r[te]]
    return (float(ll[ok].mean()), float(lln[ok].mean()),
            [float(ll[ok & (fold == f)].mean()) for f in range(FOLDS)])


def question_term(S, lg):
    """SCALE * the pooled mean of the per-game l_g (S: the harness's question x game mean-pooling matrix)."""
    return SCALE * np.asarray(S @ lg)
