"""Unit tests of the prereg23 exploration on synthetic data (no competition data).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg23/explore/test_p23.py
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

import p23_lib as K  # noqa: E402


def random_params(rng, d=K.D_IN):
    return {"W1": rng.normal(0, 0.05, (K.HIDDEN, d)), "b1": rng.normal(0, 0.1, K.HIDDEN),
            "W2": rng.normal(0, 0.1, (K.H_DIM, K.HIDDEN)), "b2": rng.normal(0, 0.1, K.H_DIM),
            "R1": rng.normal(0, 0.2, (K.RHO, K.POOL_DIM)), "c1": rng.normal(0, 0.1, K.RHO),
            "R2": rng.normal(0, 0.3, (1, K.RHO)), "c2": rng.normal(0, 0.1, 1),
            "t0": np.array(-4.0), "dt": rng.normal(0.5, 0.2, K.L - 2)}


class Constants(unittest.TestCase):
    def test_registered(self):
        self.assertEqual(K.D_IN, 1584)
        self.assertEqual(K.NEX, 44)
        self.assertEqual(K.POOL_DIM, 128)
        self.assertEqual(K.SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual((K.SAMPLER_BASE, K.HOLD_SEED, K.GATE), (23, 23023, 0.005))
        self.assertEqual((K.STEPS, K.EVAL_EVERY, K.HOLD_PER, K.BATCH, K.SCALE), (15000, 500, 100, 128, 4.0))

    def test_holdout_bit(self):
        keys = [f"{k}:{i:06x}" for k in "ABCDE" for i in range(2000)]
        bits = [K.holdout_bit(x) for x in keys]
        self.assertTrue(set(bits) <= {0, 1})
        self.assertTrue(0.08 < np.mean(bits) < 0.12)
        import p19_lib as K19
        self.assertNotEqual(bits, [K19.holdout_bit(x) for x in keys])  # a different salt


class Features(unittest.TestCase):
    def test_features(self):
        import p09_forward as FW
        rng = np.random.default_rng(0)
        mp = {"mu_x": rng.normal(size=K.X_DIM), "sd_x": rng.uniform(0.5, 2, K.X_DIM),
              "mu_ex": rng.normal(size=K.NEX), "sd_ex": rng.uniform(0.5, 2, K.NEX)}
        X = rng.normal(size=(9, K.X_DIM)).astype(np.float32)
        X[0, 800:] = np.nan
        EX = rng.normal(size=(9, K.NEX)).astype(np.float32)
        EX[1, :22] = np.nan
        F = K.features(X, EX, mp)
        self.assertEqual(F.shape, (9, K.D_IN))
        self.assertTrue(np.isfinite(F).all())
        self.assertTrue(np.array_equal(F[:, :K.X_DIM], FW.standardise_x(X.astype(np.float64), mp["mu_x"], mp["sd_x"])))
        self.assertTrue(np.array_equal(F[:, K.X_DIM:], FW.standardise_ex(EX, mp["mu_ex"], mp["sd_ex"])))
        self.assertTrue(np.all(F[1, K.X_DIM:K.X_DIM + 22] == 0))


class Forward(unittest.TestCase):
    def test_probabilities_and_permutation(self):
        rng = np.random.default_rng(0)
        par = K.p64(random_params(rng, 20))
        X = rng.normal(size=(7, 5, 20))
        lp = K.set_logp_np(par, X)
        self.assertEqual(lp.shape, (7, K.L))
        self.assertTrue(np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-9))
        self.assertTrue(np.allclose(lp, K.set_logp_np(par, X[:, ::-1, :]), atol=1e-12))

    def test_mean_std(self):
        rng = np.random.default_rng(1)
        H = rng.normal(size=(4, 6, 3))
        V = K.mean_std_dense(H)
        self.assertTrue(np.allclose(V[:, :3], H.mean(1), atol=1e-14))
        self.assertTrue(np.allclose(V[:, 3:], H.std(1, ddof=0), atol=1e-14))
        one = K.mean_std_dense(H[:, :1])
        self.assertTrue(np.all(one[:, 3:] == 0))

    def test_torch_equals_numpy(self):
        try:
            import torch
            import p23_train as T
        except Exception as e:  # pragma: no cover
            self.skipTest(f"torch unavailable: {e}")
        rng = np.random.default_rng(1)
        params = {k: np.asarray(v, dtype=np.float32) for k, v in random_params(rng).items()}
        model = T.torch_model(0)
        model.load_state_dict({k: torch.from_numpy(params[v].reshape(model.state_dict()[k].shape))
                               for k, v in T.STATE_KEYS.items()})
        model.eval()
        for n in (1, 5, 20):
            X = rng.normal(size=(6, n, K.D_IN)).astype(np.float32)
            with torch.no_grad():
                lp_t = model(torch.from_numpy(X)).double().numpy()
            lp_n = K.set_logp_np(K.p64(params), X.astype(np.float64))
            self.assertLess(float(np.max(np.abs(np.exp(lp_t) - np.exp(lp_n)))), 1e-5)
        exp = T.export_params(model)
        self.assertEqual(set(exp), set(K.PARAM_NAMES))
        for k in K.PARAM_NAMES:
            self.assertTrue(np.array_equal(exp[k].reshape(-1), params[k].reshape(-1)))

    def test_torch_gradient_finite_with_identical_games(self):
        try:
            import torch
            import p23_train as T
        except Exception as e:  # pragma: no cover
            self.skipTest(f"torch unavailable: {e}")
        model = T.torch_model(0)
        model.eval()
        x = torch.zeros(3, 5, K.D_IN)
        x[:, :, :] = torch.randn(3, 1, K.D_IN)   # identical games -> zero variance
        loss = -model(x)[:, 2].mean()
        loss.backward()
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_torch_training_mode_shapes(self):
        try:
            import torch
            import p23_train as T
        except Exception as e:  # pragma: no cover
            self.skipTest(f"torch unavailable: {e}")
        model = T.torch_model(3)
        lp = model(torch.randn(4, 10, K.D_IN))
        self.assertEqual(tuple(lp.shape), (4, K.L))
        self.assertTrue(torch.allclose(lp.exp().sum(1), torch.ones(4, dtype=lp.dtype), atol=1e-5))


class Pooling(unittest.TestCase):
    def test_pool_mean_std(self):
        rng = np.random.default_rng(2)
        H = rng.normal(size=(12, 5))
        groups = [[0, 3, 5], [], [2], [1, 4, 6, 7, 11], [9, 10]]
        V = K.pool_mean_std(groups, H)
        for q, g in enumerate(groups):
            if not g:
                self.assertTrue(np.all(V[q] == 0))
                continue
            self.assertTrue(np.allclose(V[q, :5], H[g].mean(0), atol=1e-14))
            self.assertTrue(np.allclose(V[q, 5:], H[g].std(0, ddof=0), atol=1e-14))
        self.assertTrue(np.all(V[2, 5:] == 0))
        import p02_rank_fuse as rf
        M = rf.pool_matrix(groups, 12)
        self.assertTrue(np.allclose(V[:, :5], np.asarray(M @ H), atol=1e-15))

    def test_set_term(self):
        from scipy import sparse
        rng = np.random.default_rng(4)
        S = K.mean_matrix([[0, 1, 2], [3, 4], [5]], 6)
        usable = np.array([True, False, True, False, False, True])
        g = K.usable_groups(sparse.csr_matrix(S), usable)
        self.assertEqual(g, [[0, 2], [], [5]])
        pars = [K.p64(random_params(rng, 20)) for _ in range(3)]
        Xg = rng.normal(size=(6, 20))
        H = [K.h_np(p, Xg) for p in pars]
        E, per, lp, has = K.set_term(pars, H, g)
        self.assertEqual(has.tolist(), [True, False, True])
        self.assertTrue(np.all(E[1] == 0) and all(np.all(x[1] == 0) for x in per))
        want0 = [K.set_logp_np(p, Xg[[0, 2]][None])[0] for p in pars]
        self.assertTrue(np.allclose(per[0][0], K.SCALE * want0[0], atol=1e-12))
        self.assertTrue(np.allclose(E[0], K.SCALE * K.lse_mean([w[None] for w in want0])[0], atol=1e-12))


class Gates(unittest.TestCase):
    def test_gates(self):
        ok = {"B": 0.0, "W": 0.01}
        self.assertTrue(K.gates(0.0050, [0.001, 0.02], 0.01, [0.001, 0.02], ok)["passed"])
        self.assertFalse(K.gates(0.0049, [0.001, 0.02], 0.01, [0.001, 0.02], ok)["passed"])
        self.assertFalse(K.gates(0.02, [0.0, 0.02], 0.01, [0.001, 0.02], ok)["passed"])
        self.assertFalse(K.gates(0.02, [0.01, 0.02], 0.01, [0.0, 0.02], ok)["passed"])
        self.assertFalse(K.gates(0.02, [0.01, 0.02], 0.01, [0.001, 0.02], {"B": -0.001, "W": 0.01})["passed"])


if __name__ == "__main__":
    unittest.main()
