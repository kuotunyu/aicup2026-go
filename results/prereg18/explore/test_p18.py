"""Unit tests of p18_lib on synthetic data (no repository data). Run: .venv/Scripts/python.exe -m pytest -q
results/prereg18/explore/test_p18.py"""

import os
import sys
import unittest

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p18_lib as K  # noqa: E402


def synth_players(rng, n=900, lo=5, hi=120):
    games, nxt = [], 0
    for _ in range(n):
        g = {}
        for c in K.COLOURS:
            k = int(rng.integers(lo, hi))
            g[c] = np.arange(nxt, nxt + k)
            nxt += k
        games.append(g)
    return K.PlayerGames([f"X:{i}" for i in range(n)], games), nxt


class PLDA(unittest.TestCase):
    def test_llr_matches_direct_density(self):
        rng = np.random.default_rng(1)
        b = rng.uniform(0.05, 2.0, size=4)
        M = rng.normal(size=(5, 4))
        n = np.array([5, 10, 15, 20, 7], dtype=float)
        A = K.plda_llr(M, n, b)
        for i in range(5):
            for j in range(5):
                if i != j:
                    self.assertAlmostEqual(A[i, j], K.plda_llr_direct(M[i], n[i], M[j], n[j], b), places=9)
        self.assertTrue(np.allclose(A, A.T))
        self.assertTrue(np.all(np.diag(A) == 0))

    def test_llr_separates_same_player(self):
        rng = np.random.default_rng(2)
        d, b = 16, np.full(16, 1.0)
        mu = rng.normal(size=(50, d))
        M = np.concatenate([mu + rng.normal(size=(50, d)) / np.sqrt(20), mu + rng.normal(size=(50, d)) / np.sqrt(20)])
        A = K.plda_llr(M, np.full(100, 20.0), b)
        same = np.array([A[i, i + 50] for i in range(50)])
        other = A[:50, 50:][~np.eye(50, dtype=bool)]
        self.assertGreater(np.median(same), np.percentile(other, 99))

    def test_accum_scatter_and_lda(self):
        rng = np.random.default_rng(3)
        d, P = 6, 40
        pid = rng.integers(0, P, size=2000)
        Z = rng.normal(size=(P, d))[pid] + rng.normal(size=(2000, d)) * 0.5
        acc = K.Accum(P, d)
        for s in range(0, 2000, 300):
            acc.add(Z[s:s + 300], pid[s:s + 300])
        Sw, Sb, c, means, cnt = acc.scatter()
        mean_ref = np.stack([Z[pid == p].mean(0) for p in range(P)])
        dev = Z - mean_ref[pid]
        self.assertTrue(np.allclose(Sw, dev.T @ dev / 2000))
        D = mean_ref - mean_ref.mean(0)
        self.assertTrue(np.allclose(Sb, D.T @ D / P))
        V, w, eps = K.lda_fit(Sw, Sb, dim=4)
        self.assertTrue(np.allclose(V.T @ (Sw + eps * np.eye(d)) @ V, np.eye(4), atol=1e-8))
        self.assertTrue(np.all(np.diff(w) <= 0))
        b = K.plda_prior((means - c) @ V, cnt)
        self.assertTrue(np.all(b >= K.B_FLOOR))


class Pools(unittest.TestCase):
    def test_pool_invariants_all_scenarios(self):
        rng = np.random.default_rng(4)
        pg, _ = synth_players(rng)
        for s in K.SCENARIOS:
            sc = K.scenario(s[0])
            for i in range(3):
                qs, att = K.make_pool(7, sc["id"], i, pg, sc["p_multi"], sc["sizes"])
                self.assertEqual(K.check_pool(qs, sc["p_multi"], sc["sizes"], pg), [])
                sizes = {}
                for q in qs:
                    sizes[q["player"]] = sizes.get(q["player"], 0) + 1
                if sc["p_multi"] == 0:
                    self.assertEqual(max(sizes.values()), 1)
                else:
                    self.assertLessEqual(max(sizes.values()), max(sc["sizes"]))
                for q in qs:
                    self.assertTrue(np.all(np.isin(q["games"], pg.games[q["player"]][q["colour"]])))

    def test_pool_deterministic(self):
        rng = np.random.default_rng(5)
        pg, _ = synth_players(rng)
        a, _ = K.make_pool(18, 1, 3, pg, 0.3, (2, 3))
        b, _ = K.make_pool(18, 1, 3, pg, 0.3, (2, 3))
        self.assertEqual([(q["player"], q["colour"], q["k"], q["games"].tolist()) for q in a],
                         [(q["player"], q["colour"], q["k"], q["games"].tolist()) for q in b])

    def test_infeasible_raises(self):
        rng = np.random.default_rng(6)
        pg, _ = synth_players(rng, n=50, lo=5, hi=6)
        with self.assertRaises(RuntimeError):
            K.make_pool(1, 0, 0, pg, 0.0, ())

    def test_multi_fraction(self):
        rng = np.random.default_rng(7)
        pg, _ = synth_players(rng, n=1500)
        multi, groups = 0, 0
        for i in range(10):
            qs, _ = K.make_pool(9, 1, i, pg, 0.3, (2, 3))
            g = {}
            for q in qs:
                g.setdefault(q["group"], []).append(q)
            groups += len(g)
            multi += sum(1 for v in g.values() if len(v) > 1)
        self.assertAlmostEqual(multi / groups, 0.3, delta=0.05)


class Thresholds(unittest.TestCase):
    def test_choose_tau(self):
        llr = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
        same = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
        tau, rec = K.choose_tau(llr, same, target=0.8)
        self.assertEqual(tau, 5.0)            # top 5: 4/5 = 0.8
        self.assertEqual(rec["tp"], 4)
        tau, rec = K.choose_tau(llr, same, target=1.0)
        self.assertEqual(tau, 7.0)
        tau, rec = K.choose_tau(llr, ~same, target=0.99)
        self.assertEqual(tau, float("inf"))

    def test_choose_c(self):
        counts = [0] * 990 + [1] * 6 + [3] * 4
        self.assertEqual(K.choose_c(counts, 0.005), 2)   # >=1: 1.0%, >=2: 0.4%
        self.assertEqual(K.choose_c([0] * 1000, 0.005), 1)

    def test_components(self):
        link = np.zeros((8, 8), dtype=bool)
        for a, b in ((0, 1), (2, 3), (3, 4), (5, 6), (6, 7), (7, 5)):
            link[a, b] = link[b, a] = True
        comps, big = K.pooled_groups(link, max_size=2)
        self.assertEqual(comps, [[0, 1]])
        self.assertEqual(sorted(big), [3, 3])
        comps, big = K.pooled_groups(link, max_size=4)
        self.assertEqual(sorted(map(tuple, comps)), [(0, 1), (2, 3, 4), (5, 6, 7)])


class Misc(unittest.TestCase):
    def test_split_and_scores(self):
        self.assertIn(K.split_bit("D:abc"), (0, 1))
        self.assertEqual(K.split_bit("D:abc"), K.split_bit("D:abc"))
        sc = K.official_score(np.array([0, 1, 3]), np.array([0, 0, 0]))
        self.assertTrue(np.allclose(sc, [1, K.E1, 0]))
        self.assertAlmostEqual(K.rank_balanced([1, 0, 1, 1], [0, 0, 1, 1]), 0.75)

    def test_gates(self):
        g = K.gates(-0.0005, 0.004, 0.012, 0.001)
        self.assertTrue(g["passed"])
        self.assertFalse(K.gates(-0.002, 0.0, 0.02, 0.01)["passed"])
        self.assertFalse(K.gates(0.0, 0.012, 0.02, 0.01)["passed"])
        self.assertFalse(K.gates(0.0, 0.0, 0.009, 0.01)["passed"])
        self.assertFalse(K.gates(0.0, 0.0, 0.02, 0.0)["passed"])


if __name__ == "__main__":
    unittest.main()
