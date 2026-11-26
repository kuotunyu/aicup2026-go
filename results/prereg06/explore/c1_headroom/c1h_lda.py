"""CPU 'learned stylometry' term for the c1_headroom exploration (continuation run).

A linear discriminant (LDA) between PLAYERS, learned on per-game style features of fit players who
are NOT in the simulated pool, then used on the pool: candidate profile = mean of its games in the
LDA space, question = mean of its games, score = - squared Euclidean distance (in the whitened LDA
space this is the Gaussian log-likelihood up to a per-question constant, so the per-question ranking
and the row z-score used by the fusion are the same).

Per-game features (dense_ldain.npy, built once by build_ldain()):
  style2 (66 values, c1h_feats.style2_vec: 5 move-number phases x line / distance / zone shares)
  + sqrt of the row-normalised 'resp' token counts (220 values, c1h_feats.f_resp: line, relation to
    the previous opponent move and to the own previous move, by phase).
The opening tokens are NOT in it (they are the base / n30 / joseki terms).

Training players: every cached fit player (>= 105 black games) outside the pool's 400 (about 680
players, about 145,000 black games). Nothing of the pool's players is used to fit W.
"""
import sys
sys.dont_write_bytecode = True
import os

import numpy as np
import scipy.linalg as sla

import c1h_lib as L

LDAIN = os.path.join(L.SCRATCH, "dense_ldain.npy")
_MEMO = {}


def build_ldain():
    X1 = np.load(os.path.join(L.SCRATCH, "dense_style2.npy"))
    R = L.load_family("resp").astype(np.float64)
    tot = np.asarray(R.sum(1)).ravel()
    tot[tot <= 0] = 1.0
    Rn = np.sqrt(np.asarray((R.multiply(1.0 / tot[:, None])).todense()))
    X = np.hstack([X1, Rn]).astype(np.float32)
    np.save(LDAIN, X)
    return X.shape


def _X():
    return np.load(LDAIN, mmap_mode="r")


def fit(train_rows, labels, reg=1e-3):
    """W (D x D, columns by decreasing discriminant eigenvalue, W^T Sw W = I), feature mean / sd."""
    X = np.asarray(_X()[train_rows], np.float64)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    X = (X - mu) / sd
    lab = np.asarray(labels)
    uniq, inv = np.unique(lab, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    M = np.zeros((len(uniq), X.shape[1]))
    np.add.at(M, inv, X)
    M /= n[:, None]
    Xw = X - M[inv]
    Sw = Xw.T @ Xw / len(X)
    Sb = (M * n[:, None]).T @ M / len(X)
    D = Sw.shape[0]
    Sw_r = Sw + reg * np.trace(Sw) / D * np.eye(D)
    ev, W = sla.eigh(Sb, Sw_r)
    order = np.argsort(-ev)
    return W[:, order], ev[order], mu, sd


_FIT = {}


def fit_for_pool(pool_players, cache):
    """W fitted on every cached player outside pool_players (memoised on the player set)."""
    key = frozenset(pool_players)
    if key not in _FIT:
        train_rows, labels = [], []
        for p, rows in cache["by_player"].items():
            if p in key:
                continue
            train_rows += rows
            labels += [p] * len(rows)
        W, ev, mu, sd = fit(np.asarray(train_rows), labels)
        _FIT.clear()
        _FIT[key] = (W, ev, mu, sd, len(set(labels)), len(train_rows))
    return _FIT[key]


def project(rec, cache):
    key = (id(rec), rec.get("seed"), rec.get("scenario"), len(rec["q_rows"]), len(rec["cand_rows"][0]))
    if key in _MEMO:
        return _MEMO[key]
    pool_players = set(cache["gpid"][rec["cand_rows"][i][0]] for i in range(len(rec["cand_rows"])))
    W, ev, mu, sd, n_pl, n_g = fit_for_pool(pool_players, cache)
    X = _X()

    def emb(rows):
        Z = ((np.asarray(X[rows], np.float64) - mu) / sd) @ W
        return Z.mean(0)
    Cm = np.vstack([emb(r) for r in rec["cand_rows"]])
    Qm = np.vstack([emb(q) for q in rec["q_rows"]])
    _MEMO.clear()
    _MEMO[key] = (Cm, Qm, ev, n_pl, n_g)
    return _MEMO[key]


def lda_term(rec, cache, d):
    Cm, Qm, _ev, _np, _ng = project(rec, cache)
    d = min(d, Cm.shape[1])
    C, Q = Cm[:, :d], Qm[:, :d]
    return -(np.sum(Q * Q, 1)[:, None] - 2 * Q @ C.T + np.sum(C * C, 1)[None, :])
