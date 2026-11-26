"""prereg69 unit tests (numpy only, no data): m8_rows keeps unchanged rows, rewrites only M8's changes, refuses
inconsistencies. Run: .venv/Scripts/python.exe -m pytest -q results/prereg69/explore/test_p69.py"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p69_lib as K69  # noqa: E402

K65 = K69.K65


def rule_b(P):
    return np.argmax(np.asarray(P) @ K65.SM, axis=1)


def softmax(Z):
    return K65.softmax(Z)


class M8Rows(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.L0 = K65.log_softmax(rng.normal(size=(6, 10)) * 2.0)
        self.P0 = softmax(self.L0)
        self.cols = ["B", "B", "W", "W", "B", "W"]

    def test_no_edges_keeps_every_row(self):
        S = np.full((6, 6), -1.0)
        P, pred, info = K69.m8_rows(self.L0, self.P0, S, self.cols, rule_b, softmax)
        self.assertEqual(info["changed"], [])
        self.assertEqual(info["edges"], 0)
        self.assertTrue(np.array_equal(P, self.P0))
        self.assertTrue(np.array_equal(pred, rule_b(self.P0)))

    def test_group_change_and_rows(self):
        # q0 and q1 (same colour, near ranks): q1 strongly at class 6, q0 slightly at class 5 with class 6 close
        L0 = np.full((3, 10), -20.0)
        L0[0, 5], L0[0, 6] = -0.5, -1.0
        L0[1, 6] = 0.0
        L0[2, 1] = 0.0
        L0 = K65.log_softmax(L0)
        P0 = softmax(L0)
        S = np.array([[-1, 0.95, 0.0], [0.95, -1, 0.0], [0.0, 0.0, -1]])
        cols = ["B", "B", "B"]
        P, pred, info = K69.m8_rows(L0, P0, S, cols, rule_b, softmax)
        p_out, lp_new, _e, p0 = K65.m8(L0, S, cols)
        self.assertTrue(np.array_equal(pred, p_out))
        self.assertEqual(info["changed"], [0])
        for i in range(3):
            if i in info["changed"]:
                self.assertTrue(np.array_equal(P[i], softmax(lp_new[i:i + 1])[0]))
            else:
                self.assertTrue(np.array_equal(P[i], P0[i]))
        self.assertTrue(np.array_equal(rule_b(P), pred))

    def test_refuses_pred0_mismatch(self):
        S = np.full((6, 6), -1.0)
        bad = lambda P: rule_b(P) * 0 + 99  # noqa: E731
        with self.assertRaises(ValueError):
            K69.m8_rows(self.L0, self.P0, S, self.cols, bad, softmax)

    def test_adjacency_matches_edges(self):
        rng = np.random.default_rng(1)
        S = rng.uniform(0.5, 1.0, size=(6, 6))
        S = (S + S.T) / 2
        _P, _pred, info = K69.m8_rows(self.L0, self.P0, S, self.cols, rule_b, softmax)
        adj = K69.m8_adjacency(self.L0, S, self.cols)
        self.assertEqual(int(np.triu(adj, 1).sum()), info["edges"])


if __name__ == "__main__":
    unittest.main()
