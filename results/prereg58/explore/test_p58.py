"""prereg58: synthetic checks of p58_lib (no data file is read).  .venv/Scripts/python.exe -m pytest -q <this file>"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p58_lib as K  # noqa: E402


def brute_ll(q, Km, h, c):
    out = np.zeros((len(q), K.L))
    for g in range(len(q)):
        for r in range(K.L):
            out[g, r] = np.log(sum(q[g, s] * Km[h[g], c[g], s - r + K.L - 1] for s in range(K.L)) + K.EPS)
    return out


def synth(rng, G=400):
    r = rng.integers(0, K.L, G)
    h = rng.integers(0, K.N_HD, G)
    c = rng.integers(0, K.N_COL, G)
    q = rng.dirichlet(np.ones(K.L) * 0.3, G)
    return q, r, h, c


def test_kernel_counts_and_normalisation():
    rng = np.random.default_rng(0)
    q, r, h, c = synth(rng)
    mask = rng.random(len(r)) < 0.8
    C = K.kernel_counts(q, r, h, c, mask)
    assert C.shape == (K.N_HD, K.N_COL, K.ND)
    assert abs(C.sum() - mask.sum()) < 1e-9                      # every game adds exactly 1
    ref = np.zeros_like(C)
    for g in np.flatnonzero(mask):
        for s in range(K.L):
            ref[h[g], c[g], s - r[g] + K.L - 1] += q[g, s]
    assert np.allclose(C, ref)
    Km = K.kernel(C)
    assert np.allclose(Km.sum(axis=2), 1.0) and (Km > 0).all()
    empty = K.kernel(np.zeros_like(C))
    assert np.allclose(empty, 1.0 / K.ND)                        # add-0.5 only -> uniform


def test_game_ll_equals_brute_force():
    rng = np.random.default_rng(1)
    q, r, h, c = synth(rng, 120)
    Km = K.kernel(K.kernel_counts(q, r, h, c))
    assert np.allclose(K.game_ll(q, Km, h, c), brute_ll(q, Km, h, c), atol=1e-12)


def test_class_q():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((50, 16))
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    cr = rng.integers(0, K.L, 50)
    e = rng.standard_normal((30, 16)).astype(np.float32)
    e[3] = 0.0
    Q = K.class_q(e, Wn, cr, (20, 50), block=7)
    for t in (20, 50):
        assert np.allclose(Q[t].sum(1), 1.0)
        g = 5
        cos = (e[g].astype(np.float64) / np.linalg.norm(e[g].astype(np.float64))) @ Wn.T
        p = np.exp(t * cos - (t * cos).max())
        p /= p.sum()
        assert np.allclose(Q[t][g], np.bincount(cr, weights=p, minlength=K.L))
        assert np.allclose(Q[t][3], np.bincount(cr, minlength=K.L) / 50.0)     # zero vector: class shares
    # the vector's length does not matter
    assert np.allclose(K.class_q(3.0 * e, Wn, cr, (20,))[20], Q[20])


def test_planted_signal_is_recovered():
    """Opponents one rank above the player when the player is black, same rank when white: the summed term finds
    the player's rank."""
    rng = np.random.default_rng(3)
    G = 6000
    r = rng.integers(1, K.L - 1, G)
    c = rng.integers(0, 2, G)
    h = np.zeros(G, np.int64)
    s = r + (c == 0)
    q = np.full((G, K.L), 0.02)
    q[np.arange(G), s] = 1.0
    q /= q.sum(1, keepdims=True)
    Km = K.kernel(K.kernel_counts(q, r, h, c))
    assert Km[0, 0].argmax() == K.L - 1 + 1 and Km[0, 1].argmax() == K.L - 1
    lg = K.game_ll(q, Km, h, c)
    assert (lg.argmax(1) == r).mean() > 0.99
    assert (K.decode(lg[:50]) == r[:50]).mean() > 0.9


def test_score_and_folds():
    SM = K.score_matrix()
    assert SM[3, 3] == 1.0 and abs(SM[3, 4] - np.exp(-1.0)) < 1e-15 and SM[3, 5] == 0.0
    assert K.rank_balanced([1.0, 0.0, 1.0], [0, 0, 1]) == 0.75
    f = [K.player_fold("p%04d" % i) for i in range(2000)]
    assert set(f) == set(range(K.FOLDS)) and f == [K.player_fold("p%04d" % i) for i in range(2000)]
    assert list(K.hd_group([0, 1, 2, 9])) == [0, 1, 2, 2]
    assert list(K.colour_index(["B", "W"])) == [0, 1]
    assert list(K.rank_index(["D", "6D", "1D"])) == [0, 9, 4]
