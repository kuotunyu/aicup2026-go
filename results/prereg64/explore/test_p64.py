"""prereg64 unit checks (numpy only, synthetic data; no repository data is read)."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p64_lib as K  # noqa: E402


class QOpp(unittest.TestCase):
    def test_direct_softmax_with_own_head_row_left_out(self):
        rng = np.random.default_rng(0)
        e = rng.standard_normal((7, 5))
        Wn, Pn = K.l2(rng.standard_normal((9, 5))), K.l2(rng.standard_normal((4, 5)))
        wr, pr = rng.integers(0, 10, 9), rng.integers(0, 10, 4)
        own = np.array([-1, 0, 3, 8, -1, 2, 5])
        q = K.q_opp(e, Wn, wr, Pn, pr, own)
        en = K.l2(e)
        for g in range(7):
            z = np.concatenate([K.TAU1 * Wn @ en[g], K.TAU2 * Pn @ en[g] + K.BIAS])
            if own[g] >= 0:
                z[own[g]] = -np.inf
            p = np.exp(z - z.max())
            p /= p.sum()
            want = np.bincount(np.concatenate([wr, pr]), weights=p, minlength=10)
            np.testing.assert_allclose(q[g], want, rtol=1e-10, atol=1e-14)


class Merge(unittest.TestCase):
    def test_no_edge_no_change(self):
        lp = np.log(np.full((4, 10), 0.1))
        S = np.full((4, 4), 0.5)
        out, W = K.merge(lp, S, [5, 10, 15, 20])
        self.assertTrue(np.array_equal(out, lp))
        self.assertEqual(K.groups(W), [])

    def test_update_formula(self):
        rng = np.random.default_rng(1)
        lp = rng.standard_normal((3, 10))
        S = np.array([[-1, 0.8, 0.1], [0.8, -1, 0.66], [0.1, 0.66, -1]])
        n = [5, 20, 10]
        out, W = K.merge(lp, S, n)
        w01, w12 = (0.8 - 0.65) / 0.35, (0.66 - 0.65) / 0.35
        np.testing.assert_allclose(out[0], lp[0] + w01 * 20 / 12 * lp[1])
        np.testing.assert_allclose(out[1], lp[1] + w01 * 5 / 12 * lp[0] + w12 * 10 / 12 * lp[2])
        self.assertEqual(K.groups(W), [[0, 1, 2]])


class Fit(unittest.TestCase):
    def test_weights_bounded_and_informative_term_gets_weight(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 10, 500)
        good = np.eye(10)[y] * 3.0 + rng.standard_normal((500, 10)) * 0.5
        noise = rng.standard_normal((500, 10))
        ws, rec = K.fit_weights([good, noise], y)
        self.assertTrue(((ws >= 0) & (ws <= 4)).all())
        self.assertGreater(ws[0], ws[1])

    def test_log_softmax(self):
        z = np.array([[1.0, 2.0, 3.0] + [0.0] * 7])
        np.testing.assert_allclose(np.exp(K.log_softmax(z)).sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
