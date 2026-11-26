"""Small tests of p66_eval.py (no pool, no score). Run: .venv/Scripts/python.exe -m pytest -q results/prereg66/explore/test_p66.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p66_eval as M  # noqa: E402


class Combine(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.F = r.normal(size=(40, 60))
        self.z1, self.zm, self.zs, self.zr = (r.normal(size=(40, 60)) for _ in range(4))
        self.ks = np.array([1, 2, 3, 5] * 10)

    def test_off_is_v27(self):
        got = M.combine(self.F, self.z1, None, None, None, self.ks, M.OFF)
        want = M.st.csls(self.F + 8.0 * self.z1, 20)
        self.assertTrue(np.array_equal(got, want))

    def test_off_ignores_nan_terms(self):
        nan = np.full_like(self.F, np.nan)
        got = M.combine(self.F, self.z1, nan, nan, nan, self.ks, M.OFF)
        self.assertTrue(np.array_equal(got, M.st.csls(self.F + 8.0 * self.z1, 20)))

    def test_primary_order(self):
        P1 = 5.5 * self.z1 + 3.0 * self.zm + 3.0 * self.zs
        want = M.st.csls(self.F + P1 + 1.8 * self.zr, 20) + M.w_post_vec(self.ks) * self.zr
        got = M.combine(self.F, self.z1, self.zm, self.zs, self.zr, self.ks, M.PRIMARY)
        self.assertTrue(np.array_equal(got, want))

    def test_w_post(self):
        self.assertEqual(M.w_post_vec(np.array([1, 2, 3, 5]))[:, 0].tolist(), [2.8, 2.2, 1.5, 0.8])


class Parts(unittest.TestCase):
    def test_c13(self):
        m = np.zeros((2, 2, 770), np.float32)
        m[:, 0, :384] = 3.0
        m[:, 0, 384:768] = 1.0
        m[:, 0, 768] = -2.0
        m[:, 0, 769] = 0.5
        m[:, 1] = 99.0
        X = M.c13_feature_385(m)
        self.assertEqual(X.shape, (2, 385))
        self.assertTrue(np.all(X[:, :384] == 2.0) and np.all(X[:, 384] == -1.5))

    def test_cfg_lam(self):
        cfg = {"whiten": {"lam": 0.7, "lda": None}, "wE": 8}
        self.assertIs(M.cfg_lam(cfg, 0.70), cfg)
        c = M.cfg_lam(cfg, 0.78)
        self.assertEqual(c["whiten"], {"lam": 0.78, "lda": None})
        self.assertEqual(cfg["whiten"]["lam"], 0.7)

    def test_systems(self):
        self.assertEqual(M.SYSTEMS["A66"], M.PRIMARY)
        self.assertEqual(M.R["csls_k"], 20)

    def test_e_uses(self):
        u = M.e_uses_so_far()
        self.assertTrue(u["consistent"], u)
        self.assertGreaterEqual(u["count"], 9)


if __name__ == "__main__":
    unittest.main()
