"""prereg65 unit checks (numpy only, synthetic data; no repository data is read)."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p65_lib as K  # noqa: E402


def counts_of(rank, src=None):
    return np.bincount(rank, minlength=10).astype(float)


class QC49(unittest.TestCase):
    def test_direct_prior_corrected_softmax(self):
        rng = np.random.default_rng(1)
        e = rng.standard_normal((6, 5))
        Wn = K.K59.l2(rng.standard_normal((30, 5)))
        Pn = K.K59.l2(rng.standard_normal((40, 5)))
        wr = np.arange(30) % 10
        pr = np.arange(40) % 10
        src = np.array(["E"] * 20 + ["X"] * 20)
        own = np.array([-1, 0, 3, 8, -1, 2])
        q = K.q_c49(e, Wn, wr, Pn, pr, src, own)
        en = K.K59.l2(e)
        N = K.class_counts(wr, pr, src)
        for g in range(6):
            w = np.zeros(10)
            parts = [(K.TAU1 * Wn @ en[g], wr, N["head"], 0.0, own[g]),
                     (K.TAU2 * Pn[:20] @ en[g], pr[:20], N["E"], K.B_E, -1),
                     (K.TAU2 * Pn[20:] @ en[g], pr[20:], N["X"], K.B_X, -1)]
            zs = []
            for z, rk, n, b, o in parts:
                z = z + b
                if o >= 0:
                    z = z.copy()
                    z[o] = -np.inf
                zs.append((z, rk, n))
            m = max(z.max() for z, _r, _n in zs)
            for z, rk, n in zs:
                w += np.bincount(rk, weights=np.exp(z - m), minlength=10) / n * n.mean()
            np.testing.assert_allclose(q[g], w / w.sum(), rtol=1e-9, atol=1e-14)


class Tables(unittest.TestCase):
    def test_bayes_mean_over_r_is_one(self):
        rng = np.random.default_rng(2)
        cnt = rng.random((3, 2, 10, 10)) * 5
        M = K.table_bayes(cnt)
        np.testing.assert_allclose(M.mean(axis=2), 1.0, rtol=1e-12)

    def test_shrink_only_hd1(self):
        rng = np.random.default_rng(3)
        cnt = rng.random((3, 2, 10, 10)) * 5
        Kh = rng.random((3, 2, 10, 10))
        a, b = K.table_bayes(cnt), K.table_bayes(cnt, Kh, 0.25)
        np.testing.assert_array_equal(a[[0, 2]], b[[0, 2]])
        self.assertFalse(np.allclose(a[1], b[1]))


class Term(unittest.TestCase):
    def test_alpha_limits(self):
        Rc = np.full((2, 10), 2.0)
        Rp = np.full((2, 10), 0.5)
        q_flat = np.full((1, 10), 0.1)
        q_peak = np.eye(10)[:1]
        np.testing.assert_allclose(K.lr_lg(q_flat, Rc[:1], Rp[:1]), np.log(0.5 + K.EPS))
        np.testing.assert_allclose(K.lr_lg(q_peak, Rc[:1], Rp[:1]), np.log(2.0 + K.EPS))

    def test_game_values_masks(self):
        rng = np.random.default_rng(4)
        q = rng.dirichlet(np.ones(10), 5)
        M = np.ones((3, 2, 10, 10))
        Mp = rng.random((3, 2, 10, 10)) + 0.5
        lg = K.game_values(q, M, np.zeros(5, int), np.zeros(5, int), np.array([1, 0, 1, 1, 1]), q, Mp,
                           np.array([True, True, False, True, True]))
        self.assertTrue(np.all(lg[1] == 0.0))
        np.testing.assert_allclose(lg[2], np.log(1.0 + K.EPS))


class Merge(unittest.TestCase):
    def test_no_edge_no_change(self):
        rng = np.random.default_rng(5)
        lp = rng.standard_normal((8, 10))
        S = np.full((8, 8), 0.5)
        pred, _lp, ned, p0 = K.m8(lp, S, ["B"] * 8)
        self.assertEqual(ned, 0)
        np.testing.assert_array_equal(pred, p0)

    def test_colour_blocks_edges(self):
        lp = np.log(np.full((2, 10), 0.1))
        S = np.array([[0, 0.95], [0.95, 0]])
        _p, _l, ned, _p0 = K.m8(lp, S, ["B", "W"])
        self.assertEqual(ned, 0)

    def test_pair_agrees(self):
        a = np.log(np.array([0.02] * 3 + [0.4, 0.42] + [0.02] * 5) / 0.96)
        b = np.log(np.array([0.02] * 3 + [0.6, 0.22] + [0.02] * 5) / 0.96)
        S = np.array([[0, 0.9], [0.9, 0]])
        pred, _l, ned, p0 = K.m8(np.vstack([a, b]), S, ["B", "B"])
        self.assertEqual(ned, 1)
        self.assertTrue(pred[0] == pred[1] and pred[0] in set(p0.tolist()))


if __name__ == "__main__":
    unittest.main()
