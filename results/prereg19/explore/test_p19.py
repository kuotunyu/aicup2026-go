"""Unit tests of the prereg19 exploration on synthetic data (no competition data).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg19/explore/test_p19.py
"""

import os
import sys
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p19_lib as K  # noqa: E402


def synthetic_pool(seed=0, players_per_level=6):
    rng = np.random.default_rng(seed)
    keys, levels, cols = [], [], []
    for k in range(K.L):
        for p in range(players_per_level):
            nb = int(rng.integers(0, 30))
            nw = int(rng.integers(0, 30))
            if p == 0:
                nb, nw = 25, 0          # only black games
            for c, m in (("B", nb), ("W", nw)):
                keys += [f"{k}:p{p}"] * m
                levels += [k] * m
                cols += [c] * m
    keys, levels, cols = np.array(keys), np.array(levels), np.array(cols)
    perm = rng.permutation(len(keys))
    return keys[perm], levels[perm], cols[perm]


def random_params(rng, d=K.D_IN):
    return {"W1": rng.normal(0, 0.05, (K.HIDDEN, d)), "b1": rng.normal(0, 0.1, K.HIDDEN),
            "W2": rng.normal(0, 0.1, (K.H_DIM, K.HIDDEN)), "b2": rng.normal(0, 0.1, K.H_DIM),
            "R1": rng.normal(0, 0.2, (K.RHO, K.H_DIM)), "c1": rng.normal(0, 0.1, K.RHO),
            "R2": rng.normal(0, 0.3, (1, K.RHO)), "c2": rng.normal(0, 0.1, 1),
            "t0": np.array(-4.0), "dt": rng.normal(0.5, 0.2, K.L - 2)}


class Split(unittest.TestCase):
    def test_holdout_bit(self):
        keys = [f"{k}:{i:06x}" for k in "ABCDE" for i in range(2000)]
        bits = [K.holdout_bit(x) for x in keys]
        self.assertEqual(bits, [K.holdout_bit(x) for x in keys])
        self.assertTrue(set(bits) <= {0, 1})
        self.assertTrue(0.08 < np.mean(bits) < 0.12)


class Sampler(unittest.TestCase):
    def setUp(self):
        self.keys, self.levels, self.cols = synthetic_pool()
        self.pool = K.PlayerPool(self.keys, self.levels, self.cols, np.arange(len(self.keys)))

    def check_q(self, rows, k, n):
        self.assertEqual(len(rows), n)
        self.assertEqual(len(set(rows.tolist())), n)
        self.assertEqual(len(set(self.keys[rows].tolist())), 1)
        self.assertEqual(len(set(self.cols[rows].tolist())), 1)
        self.assertTrue(np.all(self.levels[rows] == k))

    def test_eligibility(self):
        for n in K.KS:
            for k in range(K.L):
                for ci, c in enumerate(K.COLOURS):
                    for i in self.pool.eligible[n][k][ci]:
                        self.assertEqual(self.pool.level[i], k)
                        self.assertGreaterEqual(len(self.pool.games[i][c]), n)

    def test_questions(self):
        rng = np.random.default_rng(1)
        st = {"colour_switched": 0, "level_redrawn": 0, "questions": 0}
        for _ in range(200):
            k = int(rng.integers(K.L))
            n = K.KS[int(rng.integers(4))]
            q = K.sample_question(rng, self.pool, k, n, st)
            if q is not None:
                self.check_q(q[0], k, n)
                self.assertEqual(self.cols[q[0][0]], K.COLOURS[q[1]])

    def test_colour_switch(self):
        # level with only the black-only player eligible at n = 25 games is not possible (max 29), so build one
        keys = np.array(["0:a"] * 22 + ["1:b"] * 22)
        lv = np.array([0] * 22 + [1] * 22)
        cols = np.array(["B"] * 22 + ["W"] * 22)
        pool = K.PlayerPool(keys, lv, cols, np.arange(44))
        st = {"colour_switched": 0, "level_redrawn": 0, "questions": 0}
        rng = np.random.default_rng(3)
        for _ in range(20):
            rows, ci = K.sample_question(rng, pool, 0, 20, st)
            self.assertEqual(K.COLOURS[ci], "B")
            self.assertTrue(np.all(cols[rows] == "B"))
        self.assertGreater(st["colour_switched"], 0)
        self.assertIsNone(K.sample_question(rng, pool, 2, 5, st))

    def test_batch(self):
        rng = np.random.default_rng(2)
        st = {"colour_switched": 0, "level_redrawn": 0, "questions": 0}
        for _ in range(10):
            idx, y, n = K.sample_batch(rng, self.pool, 16, st)
            self.assertEqual(idx.shape, (16, n))
            for b in range(16):
                self.check_q(idx[b], y[b], n)
        self.assertEqual(st["questions"], 160)

    def test_batch_deterministic(self):
        a = K.sample_batch(np.random.default_rng([19, 0]), self.pool, 8, {"colour_switched": 0, "level_redrawn": 0,
                                                                           "questions": 0})
        b = K.sample_batch(np.random.default_rng([19, 0]), self.pool, 8, {"colour_switched": 0, "level_redrawn": 0,
                                                                           "questions": 0})
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2])

    def test_holdout_set(self):
        hold, st = K.holdout_set(self.pool, 3, K.HOLD_SEED)
        for n, (idx, y) in hold.items():
            self.assertEqual(idx.shape[1] if len(y) else n, n)
            for row, k in zip(idx, y):
                self.check_q(row, k, n)
        self.assertEqual(sum(len(v[1]) for v in hold.values()) + st["level_missing"], 3 * K.L * len(K.KS))
        hold2, _ = K.holdout_set(self.pool, 3, K.HOLD_SEED)
        self.assertTrue(all(np.array_equal(hold[n][0], hold2[n][0]) for n in K.KS))


class Forward(unittest.TestCase):
    def test_probabilities(self):
        rng = np.random.default_rng(0)
        par = K.p64(random_params(rng, 20))
        X = rng.normal(size=(7, 5, 20))
        lp = K.set_logp_np(par, X)
        self.assertEqual(lp.shape, (7, K.L))
        self.assertTrue(np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-9))
        # permutation invariance within a question
        lp2 = K.set_logp_np(par, X[:, ::-1, :])
        self.assertTrue(np.allclose(lp, lp2, atol=1e-12))

    def test_torch_equals_numpy(self):
        try:
            import torch
            import p19_train as T
        except Exception as e:  # pragma: no cover
            self.skipTest(f"torch unavailable: {e}")
        rng = np.random.default_rng(1)
        params = {k: np.asarray(v, dtype=np.float32) for k, v in random_params(rng).items()}
        model = T.torch_model(0)
        model.load_state_dict({k: torch.from_numpy(params[v].reshape(model.state_dict()[k].shape))
                               for k, v in T.STATE_KEYS.items()})
        model.eval()
        X = rng.normal(size=(6, 10, K.D_IN)).astype(np.float32)
        with torch.no_grad():
            lp_t = model(torch.from_numpy(X)).double().numpy()
        lp_n = K.set_logp_np(K.p64(params), X.astype(np.float64))
        self.assertLess(float(np.max(np.abs(np.exp(lp_t) - np.exp(lp_n)))), 1e-5)
        exp = T.export_params(model)
        self.assertEqual(set(exp), set(K.PARAM_NAMES))
        for k in K.PARAM_NAMES:
            self.assertTrue(np.array_equal(exp[k].reshape(-1), params[k].reshape(-1)))


class Pooling(unittest.TestCase):
    def test_mean_matrix_equals_pool_matrix(self):
        import p02_rank_fuse as rf
        groups = [[0, 3, 5], [], [2], [1, 4, 6, 7]]
        A = K.mean_matrix(groups, 9).toarray()
        B = rf.pool_matrix(groups, 9).toarray()
        self.assertTrue(np.array_equal(A, B))

    def test_usable_groups_and_set_term(self):
        from scipy import sparse
        rng = np.random.default_rng(4)
        S = K.mean_matrix([[0, 1, 2], [3, 4], [5]], 6)
        usable = np.array([True, False, True, False, False, True])
        g = K.usable_groups(sparse.csr_matrix(S), usable)
        self.assertEqual(g, [[0, 2], [], [5]])
        pars = [K.p64(random_params(rng, 20)) for _ in range(2)]
        Xg = rng.normal(size=(6, 20))
        H = [K.h_np(p, Xg) for p in pars]
        E, per, lp, has = K.set_term(pars, H, g, 6)
        self.assertEqual(has.tolist(), [True, False, True])
        self.assertTrue(np.all(E[1] == 0) and all(np.all(x[1] == 0) for x in per))
        want0 = [K.head_np(p, h[[0, 2]].mean(0, keepdims=True))[0] for p, h in zip(pars, H)]
        self.assertTrue(np.allclose(per[0][0], K.SCALE * want0[0], atol=1e-12))
        self.assertTrue(np.allclose(E[0], K.SCALE * K.lse_mean([w[None] for w in want0])[0], atol=1e-12))


class Scores(unittest.TestCase):
    def test_rank_balanced(self):
        import p03_rank_fuse as p3
        rng = np.random.default_rng(5)
        y = rng.integers(0, K.L, 500)
        pred = np.clip(y + rng.integers(-2, 3, 500), 0, 9)
        sc = K.official_score(pred, y)
        self.assertAlmostEqual(K.rank_balanced(sc, y), p3.score_summary(sc, y)["rank_balanced"], places=14)

    def test_gates(self):
        g = K.gates(0.0100, [0.001, 0.02], 0.01, [0.001, 0.02], {"B": 0.0, "W": 0.01})
        self.assertTrue(g["passed"])
        self.assertFalse(K.gates(0.0099, [0.001, 0.02], 0.01, [0.001, 0.02], {"B": 0.0, "W": 0.01})["passed"])
        self.assertFalse(K.gates(0.02, [0.0, 0.02], 0.01, [0.001, 0.02], {"B": 0.0, "W": 0.01})["passed"])
        self.assertFalse(K.gates(0.02, [0.01, 0.02], 0.01, [0.0, 0.02], {"B": 0.0, "W": 0.01})["passed"])
        self.assertFalse(K.gates(0.02, [0.01, 0.02], 0.01, [0.001, 0.02], {"B": -0.001, "W": 0.01})["passed"])


if __name__ == "__main__":
    unittest.main()
