"""prereg15 exploration: unit tests of p15_lib on synthetic data (no inner data, no data/local, no score.py).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg15/explore/test_p15.py
"""

import itertools
import os
import sys
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import p15_lib as K  # noqa: E402

rf, p4 = K.rf, K.p4
L = K.L


def synthetic_posteriors(rng, n, prior, sep=2.0):
    """1-d Gaussian classes (mean sep * k, sd 1) drawn with `prior`; exact posteriors under a uniform prior."""
    y = rng.choice(L, size=n, p=prior)
    x = rng.normal(sep * y, 1.0)
    ll = -0.5 * (x[:, None] - sep * np.arange(L)[None, :]) ** 2
    P = np.exp(ll - ll.max(1, keepdims=True))
    return P / P.sum(1, keepdims=True), y


class Counts(unittest.TestCase):
    def test_largest_remainder(self):
        for s in K.SCENARIOS:
            c = K.largest_remainder(100, K.proportions(s))
            self.assertEqual(int(c.sum()), 100)
            self.assertTrue(np.all(np.abs(c - 100 * K.proportions(s)) < 1))
        self.assertEqual(K.largest_remainder(100, K.proportions("S0")).tolist(), [10] * 10)
        # three equal thirds of 10: the remainder goes to the smaller index
        self.assertEqual(K.largest_remainder(10, [1, 1, 1]).tolist(), [4, 3, 3])

    def test_sample_pool(self):
        rng = np.random.default_rng(1)
        n = 20000
        y = rng.integers(0, L, n)
        ks = rng.choice(K.KS, n)
        st = K.strata(y, ks)
        for s in K.SCENARIOS:
            props = K.proportions(s)
            pools = K.sample_pools(7, st, props, n_pools=5)
            again = K.sample_pools(7, st, props, n_pools=5)
            self.assertTrue(np.array_equal(pools, again))
            per = K.largest_remainder(100, props)
            for ix in pools:
                self.assertEqual(len(ix), 400)
                self.assertEqual(len(np.unique(ix)), 400)
                for ki, k in enumerate(K.KS):
                    blk = ix[ki * 100:(ki + 1) * 100]
                    self.assertTrue(np.all(ks[blk] == k))
                    self.assertEqual(np.bincount(y[blk], minlength=L).tolist(), per.tolist())


class EM(unittest.TestCase):
    def test_recovers_prior(self):
        rng = np.random.default_rng(2)
        uni = np.full(L, 0.1)
        for s in ("S1", "S2", "S3", "S4"):
            truth = K.proportions(s)
            P, _y = synthetic_posteriors(rng, 20000, truth)
            pi, Q, it, conv = K.em_prior(P, uni)
            self.assertTrue(conv)
            self.assertLess(K.tv(pi, truth), 0.03, s)
            self.assertTrue(np.allclose(Q.sum(1), 1.0))
            self.assertTrue(np.allclose(Q.mean(0), pi, atol=1e-8))

    def test_balanced_stays(self):
        rng = np.random.default_rng(3)
        uni = np.full(L, 0.1)
        P, _y = synthetic_posteriors(rng, 20000, uni)
        pi, _Q, _it, _c = K.em_prior(P, uni)
        self.assertLess(K.tv(pi, uni), 0.03)

    def test_identity_when_pi_equals_pi0(self):
        rng = np.random.default_rng(4)
        P = rng.dirichlet(np.ones(L), 50)
        pi0 = rng.dirichlet(np.ones(L))
        self.assertTrue(np.allclose(K.adjust(P, pi0, pi0), P))

    def test_adjust_helps_on_skew(self):
        rng = np.random.default_rng(5)
        truth = K.proportions("S1")
        P, y = synthetic_posteriors(rng, 20000, truth)
        base = rf.official_score(rf.rule_b_np(P), y).mean()
        pi, Q, _it, _c = K.em_prior(P, np.full(L, 0.1))
        self.assertGreater(rf.official_score(rf.rule_b_np(Q), y).mean(), base)


class Decisions(unittest.TestCase):
    def test_rule_b_is_argmax_expected(self):
        rng = np.random.default_rng(6)
        P = rng.dirichlet(np.ones(L), 500)
        self.assertTrue(np.array_equal(np.argmax(K.expected_scores(P), 1), rf.rule_b_np(P)))

    def test_quota_counts_and_optimal(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            n = 5
            Q = rng.dirichlet(np.ones(L) * 0.3, n)
            counts = np.zeros(L, dtype=np.int64)
            for lv in rng.choice(L, n):
                counts[lv] += 1
            pred = K.quota_assign(Q, counts)
            self.assertEqual(np.bincount(pred, minlength=L).tolist(), counts.tolist())
            ES = K.expected_scores(Q)
            got = ES[np.arange(n), pred].sum()
            labels = np.repeat(np.arange(L), counts)
            best = max(ES[np.arange(n), list(p)].sum() for p in set(itertools.permutations(labels)))
            self.assertAlmostEqual(got, best, places=12)

    def test_quota_rejects_wrong_total(self):
        with self.assertRaises(ValueError):
            K.quota_assign(np.full((3, L), 0.1), np.array([1] + [0] * 9))


class Accounting(unittest.TestCase):
    def test_point_and_identity_bootstrap(self):
        rng = np.random.default_rng(8)
        n_players = 60
        level_of = np.repeat(np.arange(L), 6)
        player = rng.integers(0, n_players, 5000)
        diff = rng.normal(0, 1, 5000)
        D, N = K.accumulate(player, diff, n_players)
        props = K.proportions("S1")
        dp = K.delta_point(D, N, level_of, props)
        manual = sum(props[lv] * diff[level_of[player] == lv].mean() for lv in range(L))
        self.assertAlmostEqual(dp, manual, places=12)
        bl = [(lv, np.tile(np.nonzero(level_of == lv)[0], (3, 1))) for lv in range(L)]
        self.assertTrue(np.allclose(K.delta_boot(D, N, bl, props), dp))

    def test_boot_levels_from_player_bootstrap(self):
        y_player = np.repeat(np.arange(L), 4)
        player = np.arange(40).repeat(3)
        boot = rf.PlayerBootstrap(player, y_player[player], 40)
        bl = K.boot_levels(boot, y_player)
        self.assertEqual([lv for lv, _d in bl], list(range(L)))
        # S0 weights reproduce PlayerBootstrap.stats (rank-balanced mean over the questions)
        rng = np.random.default_rng(9)
        sc = rng.random(len(player))
        D, N = K.accumulate(player, sc, 40)
        self.assertTrue(np.allclose(K.delta_boot(D, N, bl, K.proportions("S0")), boot.stats(sc)))


class FixedW(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.n_q = 4000
        self.m = rng.normal(0, 1, (self.n_q, L))
        self.Es = [rng.normal(0, 1, (self.n_q, L)) for _ in range(3)]
        self.n = rng.choice([5.0, 10.0, 15.0, 20.0], self.n_q)
        self.w = np.array([0.5, 1.5, 2.0])
        beta = rng.normal(0, 0.3, L - 1)
        self.theta = np.concatenate([[0.4, 0.6], self.w, beta])
        P = K.predict(self.theta, self.m, self.Es, self.n)
        self.y = np.array([rng.choice(L, p=p) for p in P])

    def test_gradient(self):
        wq = rf.level_weights(self.y)
        th3 = np.concatenate([[0.1, -0.2], np.linspace(-0.2, 0.2, L - 1)])
        f0, g = K.fixed_w_obj(th3, self.w, self.m, self.Es, self.n, self.y, wq)
        eps = 1e-6
        for i in range(len(th3)):
            e = np.zeros_like(th3)
            e[i] = eps
            num = (K.fixed_w_obj(th3 + e, self.w, self.m, self.Es, self.n, self.y, wq)[0]
                   - K.fixed_w_obj(th3 - e, self.w, self.m, self.Es, self.n, self.y, wq)[0]) / (2 * eps)
            self.assertAlmostEqual(num, g[i], delta=1e-5 * max(1.0, abs(num)))

    def test_recovers_and_keeps_w(self):
        th, rec = K.fit_fixed_w(self.m, self.Es, self.n, self.y, self.w)
        self.assertTrue(np.array_equal(th[2:5], self.w))
        self.assertLess(abs(th[0] - 0.4), 0.15)
        self.assertLess(abs(th[1] - 0.6), 0.25)
        self.assertEqual(len(rec["starts"]), len(rf.STARTS))

    def test_full_layout_matches_p04(self):
        th, _ = K.fit_fixed_w(self.m, self.Es, self.n, self.y, self.w)
        par = p4.gen_params(th, ["X", "Y", "Z"])
        self.assertEqual(par["w"], {"X": 0.5, "Y": 1.5, "Z": 2.0})
        self.assertAlmostEqual(par["a"], th[0])


if __name__ == "__main__":
    unittest.main()
