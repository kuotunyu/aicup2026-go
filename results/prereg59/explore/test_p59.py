"""prereg59: synthetic checks of p59_lib (no data file is read).  .venv/Scripts/python.exe -m pytest -q <this file>"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p59_lib as K  # noqa: E402


def setup(rng, G=90, H=40, P=25, d=12):
    Wn = K.l2(rng.standard_normal((H, d)))
    wr = rng.integers(0, K.L, H)
    Pn = K.l2(rng.standard_normal((P, d)))
    pr = rng.integers(0, K.L, P)
    emb = (rng.standard_normal((G, d)) * rng.uniform(0.5, 3.0, (G, 1))).astype(np.float32)
    own = rng.integers(-1, P, G)
    return emb, Wn, wr, Pn, pr, own


def brute_q(emb, Wn, wr, Pn, pr, own, t1, t2, b):
    out = np.zeros((len(emb), K.L))
    for g in range(len(emb)):
        e = emb[g].astype(np.float64)
        e = e / np.linalg.norm(e)
        z = np.concatenate([t1 * (Wn @ e), t2 * (Pn @ e) + b])
        rk = np.concatenate([wr, pr])
        keep = np.ones(len(z), bool)
        if own is not None and own[g] >= 0:
            keep[len(wr) + own[g]] = False
        p = np.exp(z[keep] - z[keep].max())
        p /= p.sum()
        out[g] = np.bincount(rk[keep], weights=p, minlength=K.L)
    return out


def test_combine_equals_brute_force_softmax_over_the_union():
    rng = np.random.default_rng(0)
    emb, Wn, wr, Pn, pr, own = setup(rng)
    parts = K.class_parts(emb, Wn, wr, Pn, pr, own, block=17)
    for t1, t2, b in ((30, 30, -2), (50, 120, 0), (80, 50, 2), (30, 80, 1)):
        q = K.combine(parts, t1, t2, b)
        assert np.allclose(q.sum(1), 1.0)
        assert np.allclose(q, brute_q(emb, Wn, wr, Pn, pr, own, t1, t2, b), atol=1e-12)
    # no prototype left out (the inner / test use) and a single (tau1, tau2)
    p1 = K.class_parts(emb, Wn, wr, Pn, pr, None, tau1=(50,), tau2=(80,))
    assert np.allclose(K.combine(p1, 50, 80, -1), brute_q(emb, Wn, wr, Pn, pr, None, 50, 80, -1), atol=1e-12)
    # the vector's length does not matter; sub_parts selects rows
    p3 = K.class_parts(3.0 * emb, Wn, wr, Pn, pr, own)
    assert np.allclose(K.combine(p3, 50, 50, 0), K.combine(parts, 50, 50, 0))
    idx = np.array([5, 1, 7])
    assert np.allclose(K.combine(K.sub_parts(parts, idx), 80, 30, 2), K.combine(parts, 80, 30, 2)[idx])


def test_own_prototype_is_left_out():
    rng = np.random.default_rng(1)
    emb, Wn, wr, Pn, pr, _own = setup(rng, G=25, P=25)
    emb = (5.0 * Pn).astype(np.float32)                 # game g looks exactly like prototype g
    pr = np.arange(25) % K.L
    with_own = K.combine(K.class_parts(emb, Wn, wr, Pn, pr, None), 30, 120, 2)
    without = K.combine(K.class_parts(emb, Wn, wr, Pn, pr, np.arange(25)), 30, 120, 2)
    assert (with_own[np.arange(25), pr] > 0.99).all()
    assert (without[np.arange(25), pr] < with_own[np.arange(25), pr]).all()
    assert np.allclose(without, brute_q(emb, Wn, wr, Pn, pr, np.arange(25), 30, 120, 2), atol=1e-12)


def test_prototypes():
    rng = np.random.default_rng(2)
    emb = rng.standard_normal((40, 6)).astype(np.float32)
    count = np.ones(40, np.int32)
    count[[3, 4]] = 0
    player = np.array(["p%d" % (i % 7) for i in range(40)])
    player[35:] = "lonely"
    count[35:] = 0                                       # a player without any usable game gets no prototype
    u, P, n = K.prototypes(emb, count, player)
    assert "lonely" not in u.tolist() and list(u) == sorted(u.tolist())
    assert np.allclose(np.linalg.norm(P, axis=1), 1.0)
    for i, p in enumerate(u.tolist()):
        m = (player == p) & (count > 0)
        ref = K.l2(K.l2(emb[m]).mean(0))
        assert n[i] == m.sum() and np.allclose(P[i], ref)


def synth(rng, G=500):
    r = rng.integers(0, K.L, G)
    h = rng.integers(0, K.N_HD, G)
    c = rng.integers(0, K.N_COL, G)
    q = rng.dirichlet(np.ones(K.L) * 0.3, G)
    return q, r, h, c


def test_table_counts_table_and_game_ll():
    rng = np.random.default_rng(3)
    q, r, h, c = synth(rng)
    mask = rng.random(len(r)) < 0.8
    C = K.table_counts(q, r, h, c, mask)
    ref = np.zeros((K.N_HD, K.N_COL, K.L, K.L))
    for g in np.flatnonzero(mask):
        ref[h[g], c[g], r[g]] += q[g]
    assert C.shape == ref.shape and np.allclose(C, ref) and abs(C.sum() - mask.sum()) < 1e-9
    M = K.table(C)
    assert np.allclose(M.sum(axis=3), 1.0) and (M > 0).all()
    assert np.allclose(K.table(np.zeros_like(C)), 1.0 / K.L)        # add-0.5 only -> uniform rows
    lg = K.game_ll(q, M, h, c)
    for g in range(0, len(q), 37):
        for rr in range(K.L):
            assert abs(lg[g, rr] - np.log(q[g] @ M[h[g], c[g], rr] + K.EPS)) < 1e-12


def test_fold_tables_and_heldout_ll():
    rng = np.random.default_rng(4)
    q, r, h, c = synth(rng, 800)
    fold = rng.integers(0, K.FOLDS, len(r))
    ok = rng.random(len(r)) < 0.9
    Mf = K.fold_tables(q, r, h, c, fold, ok)
    for f in range(K.FOLDS):
        assert np.allclose(Mf[f], K.table(K.table_counts(q, r, h, c, ok & (fold != f))))
    ll, lln, by_fold = K.heldout_ll(q, r, h, c, fold, ok)
    ref = []
    for g in np.flatnonzero(ok):
        ref.append(np.log(q[g] @ Mf[fold[g], h[g], c[g], r[g]] + K.EPS))
    assert abs(ll - np.mean(ref)) < 1e-12 and lln <= 0.0 and len(by_fold) == K.FOLDS


def test_full_table_handles_the_end_classes():
    """Opponents are of the player's rank or one above, except that a 6D player's opponents are all 6D and a D
    player's opponents are D or C: the table learns the end rows, and the term recovers the rank."""
    rng = np.random.default_rng(5)
    G = 8000
    r = rng.integers(0, K.L, G)
    s = np.minimum(r + rng.integers(0, 2, G), K.L - 1)
    h = np.zeros(G, np.int64)
    c = rng.integers(0, 2, G)
    q = np.full((G, K.L), 0.01)
    q[np.arange(G), s] = 1.0
    q /= q.sum(1, keepdims=True)
    M = K.table(K.table_counts(q, r, h, c))
    assert M[0, 0, K.L - 1].argmax() == K.L - 1 and M[0, 0, K.L - 1, K.L - 1] > 0.85
    assert M[0, 0, 0, :2].sum() > 0.85
    idx = np.array([rng.choice(np.flatnonzero(r == k), 10, replace=False) for k in range(K.L)])
    lg = K.game_ll(q, M, h, c)
    assert (lg[idx].sum(1).argmax(1) == np.arange(K.L)).all()


def test_question_term_and_grid():
    rng = np.random.default_rng(6)
    lg = rng.standard_normal((6, K.L))
    S = sp.csr_matrix(np.array([[0.5, 0.5, 0, 0, 0, 0], [0, 0, 0.25, 0.25, 0.25, 0.25]]))
    E = K.question_term(S, lg)
    assert np.allclose(E[0], 10.0 * lg[:2].mean(0)) and np.allclose(E[1], 10.0 * lg[2:].mean(0))
    assert K.SCALE == 10.0 and K.SMOOTH == 0.5
    assert len(K.GRID) == 60 and K.GRID[0] == (30, 30, -2) and K.GRID[-1] == (80, 120, 2)
    assert K.cell(np.array([3]), np.array([2]), np.array([1]))[0] == (2 * 2 + 1) * 10 + 3
