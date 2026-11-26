"""Unit tests of p16_lib on synthetic data (no inner data, no data/local).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg16/explore/test_p16.py
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
from scipy import sparse  # noqa: E402

import p16_lib as K  # noqa: E402


def synth_game(rng, n):
    lp14 = np.log(rng.dirichlet(np.ones(14) * 3, size=n) * 0.5).astype(np.float32)
    th = np.sort(rng.choice(np.arange(0, 300, 2), size=n, replace=False))
    lpp = np.log(rng.uniform(1e-4, 1.0, size=n)).astype(np.float32)
    ent = rng.uniform(0.1, 4.0, size=n).astype(np.float32)
    return (lp14, th), (lpp, ent, th + 0)


class Series(unittest.TestCase):
    def test_hs_series(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(7, 14))
        hr, hs, ha = K.hs_series(x)
        p = np.exp(x) / np.exp(x).sum(1, keepdims=True)
        np.testing.assert_allclose(hr, (p * np.arange(14)).sum(1), rtol=1e-12)
        np.testing.assert_array_equal(hs, x[:, 13] - x[:, 0])
        np.testing.assert_array_equal(ha, x[:, 13])

    def test_game_stats_naive(self):
        rng = np.random.default_rng(2)
        (lp14, th), (lpp, ent, tp) = synth_game(rng, 40)
        hr, hs, ha = K.hs_series(lp14)
        pl, ps = K.p1_series(lpp, ent)
        ser = {"h_rank": hr, "h_slope": hs, "h_acc": ha, "p_logp": pl, "p_surp": ps}
        st, mu, s1, s2, c, w = K.game_stats(ser, {"h": th, "p": tp})
        j = 0
        for i, s in enumerate(K.SERIES):
            x = ser[s]
            t = th
            self.assertAlmostEqual(st[j], np.std(x), places=12)
            self.assertAlmostEqual(st[j + 1], np.percentile(x, 10), places=12)
            self.assertAlmostEqual(st[j + 2], np.percentile(x, 90), places=12)
            exp_w = np.sort(x)[-5:].mean() if s == "p_surp" else np.sort(x)[:5].mean()
            self.assertAlmostEqual(st[j + 3], exp_w, places=12)
            for ph, (lo, hi) in enumerate(K.PHASES):
                m = (t >= lo) & (t < (hi if hi is not None else 10 ** 9))
                if m.any():
                    self.assertAlmostEqual(st[j + 4 + ph], x[m].mean(), places=12)
                else:
                    self.assertTrue(np.isnan(st[j + 4 + ph]))
            self.assertAlmostEqual(mu[i], x.mean(), places=12)
            self.assertEqual(c[i], len(x))
            j += 7
        self.assertAlmostEqual(st[35], np.mean(ha < np.log(0.01)))
        self.assertAlmostEqual(st[36], np.mean(pl < np.log(0.01)))
        # worst list of p_surp kept in the bad view (negated)
        np.testing.assert_allclose(w[4], np.sort(-ps)[:10])

    def test_short_game_and_empty_phase(self):
        rng = np.random.default_rng(3)
        (lp14, _th), (lpp, ent, _tp) = synth_game(rng, 3)
        th = np.array([0, 2, 4])
        hr, hs, ha = K.hs_series(lp14)
        pl, ps = K.p1_series(lpp, ent)
        st, *_rest, w = K.game_stats({"h_rank": hr, "h_slope": hs, "h_acc": ha, "p_logp": pl, "p_surp": ps},
                                     {"h": th, "p": th})
        self.assertAlmostEqual(st[3], hr.mean())          # worst 5 of 3 moves = all
        self.assertTrue(np.isnan(st[5]) and np.isnan(st[6]))   # no middle / late moves
        self.assertEqual(int(np.sum(~np.isnan(w[0]))), 3)


class Questions(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.n = 30
        self.T = K.GameTable(self.n)
        self.raw = {}
        self.usable = np.ones(self.n, dtype=bool)
        self.usable[[3, 17]] = False
        h_moves, p_moves = {}, {}
        for g in range(self.n):
            (lp14, th), (lpp, ent, tp) = synth_game(rng, int(rng.integers(8, 60)))
            h_moves[g], p_moves[g] = (lp14, th), (lpp, ent, tp)
            hr, hs, ha = K.hs_series(lp14)
            pl, ps = K.p1_series(lpp, ent)
            self.raw[g] = {"h_rank": hr, "h_slope": hs, "h_acc": ha, "p_logp": pl, "p_surp": ps}
        K.per_game_from_moves(h_moves, p_moves, np.nonzero(self.usable)[0], self.T)
        self.T.e_hs = rng.uniform(0, 9, self.n)
        self.T.e_p1e = rng.uniform(0, 9, self.n)
        self.groups = [np.array([0, 1, 2, 3]), np.array([5, 9]), np.array([3, 17]), np.array([11]),
                       np.arange(10, 30)]
        rows, cols, vals = [], [], []
        for i, g in enumerate(self.groups):
            rows += [i] * len(g)
            cols += list(g)
            vals += [1.0 / len(g)] * len(g)
        self.S = sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.groups), self.n))

    def test_groups_of(self):
        gs = K.groups_of(self.S)
        for a, b in zip(gs, self.groups):
            np.testing.assert_array_equal(np.sort(a), np.sort(b))

    def test_features_naive(self):
        F, nu = K.question_features(K.groups_of(self.S), self.usable, self.T, block=2)
        self.assertEqual(F.shape, (5, 62))
        np.testing.assert_array_equal(nu, [3, 2, 0, 1, 19])
        self.assertTrue(np.all(np.isnan(F[2])))
        for qi in (0, 1, 3, 4):
            g = [x for x in self.groups[qi] if self.usable[x]]
            np.testing.assert_allclose(F[qi, :37], np.nanmean(self.T.stats[g], axis=0), rtol=1e-12, atol=1e-12)
            for i, s in enumerate(K.SERIES):
                allx = np.concatenate([self.raw[x][s] for x in g])
                worst = (np.sort(allx)[-10:] if s == "p_surp" else np.sort(allx)[:10]).mean()
                self.assertAlmostEqual(F[qi, 37 + i], worst, places=10)
                self.assertAlmostEqual(F[qi, 42 + i], np.std(allx), places=8)
                self.assertAlmostEqual(F[qi, 47 + i], np.std([self.raw[x][s].mean() for x in g]), places=10)
            eh, ep = self.T.e_hs[g], self.T.e_p1e[g]
            np.testing.assert_allclose(F[qi, 52:58], [np.std(eh), np.std(ep), np.ptp(eh), np.ptp(ep), ep.min(),
                                                      ep.max()], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(F[qi, 58:62], [eh.mean(), ep.mean(), np.log(len(g)),
                                                      np.mean(np.log([len(self.raw[x]["h_rank"]) for x in g]))],
                                       rtol=1e-12)

    def test_design(self):
        rng = np.random.default_rng(5)
        F = rng.normal(size=(50, 62))
        F[3, 7] = np.nan
        F[:, 10] = 2.0
        mu, sd = K.std_fit(F)
        Z = K.design(F, mu, sd, True)
        self.assertEqual(Z.shape, (50, 73))
        self.assertEqual(Z[3, 7], 0.0)
        self.assertTrue(np.all(Z[:, 10] == 0.0))
        np.testing.assert_allclose(Z[:, 62:], Z[:, K.C_IDX] * Z[:, [K.ANCHOR_IDX]])
        self.assertEqual(K.design(F, mu, sd, False).shape, (50, 62))


class Model(unittest.TestCase):
    def test_gradient(self):
        rng = np.random.default_rng(6)
        Z = rng.normal(size=(40, 5))
        y = rng.integers(0, 10, 40)
        wq = K.level_weights(y)
        th = rng.normal(size=5 * 10 + 10) * 0.1
        f0, g = K.lr_obj(th, Z, y, wq, 0.01)
        num = np.zeros_like(th)
        for i in range(len(th)):
            e = np.zeros_like(th)
            e[i] = 1e-6
            num[i] = (K.lr_obj(th + e, Z, y, wq, 0.01)[0] - K.lr_obj(th - e, Z, y, wq, 0.01)[0]) / 2e-6
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-7)

    def test_level_weights_equal_p02(self):
        import p02_rank_fuse as rf
        y = np.random.default_rng(11).integers(0, 10, 500)
        self.assertTrue(np.array_equal(K.level_weights(y), rf.level_weights(y)))

    def test_learns_signal(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 10, 2000)
        F = np.column_stack([y + rng.normal(0, 1, 2000), rng.normal(size=(2000, 3))])
        mu, sd = K.std_fit(F)
        m = K.lr_fit(K.design(F, mu, sd, False), y, 1e-3)
        lp = K.lr_logp(m, K.design(F, mu, sd, False))
        self.assertLess(K.balanced_nll(lp, y), np.log(10) - 0.5)


class CrossFit(unittest.TestCase):
    def make(self, rng, nq, nplayers):
        player = rng.integers(0, nplayers, nq)
        fold = player % 5
        lv = rng.integers(0, 10, nplayers)
        y = lv[player]
        F = np.column_stack([y + rng.normal(0, 2, nq), rng.normal(size=(nq, 61))])
        return F, y, fold, player

    def test_nested_disjoint_and_reproducible(self):
        rng = np.random.default_rng(8)
        Ff, yf, ff, pf = self.make(rng, 600, 60)
        Fe, _ye, fe, pe = self.make(rng, 300, 60)
        Fm, _ym, fm, pm = self.make(rng, 200, 60)
        ct = K.nested_crossfit(Ff, yf, ff, pf, Fe, fe, pe, Fm, fm, pm, lambdas=(1e-2, 1e-1))
        for f in range(5):
            self.assertTrue(np.isnan(ct["ct_fit"][f][ff == f]).all())
            self.assertFalse(np.isnan(ct["ct_fit"][f][ff != f]).any())
            lam = ct["folds"][f]["lambda"]
            _m, (lpe,) = K.fit_predict(Ff[ff != f], yf[ff != f], lam, [Fe[fe == f]])
            np.testing.assert_array_equal(lpe, ct["ct_eval"][fe == f])
            g = (f + 1) % 5
            tr = (ff != f) & (ff != g)
            _m, (lpg,) = K.fit_predict(Ff[tr], yf[tr], lam, [Ff[ff == g]])
            np.testing.assert_array_equal(lpg, ct["ct_fit"][f][ff == g])

    def test_overlap_refused(self):
        rng = np.random.default_rng(9)
        Ff, yf, ff, pf = self.make(rng, 300, 30)
        pf = pf.copy()
        pf[ff == 1] = pf[ff == 2][0]        # a player in two folds
        Fe, _y, fe, pe = self.make(rng, 100, 30)
        with self.assertRaises(AssertionError):
            K.nested_crossfit(Ff, yf, ff, pf, Fe, fe, pe, Fe, fe, pe, lambdas=(1e-1,))


class FoldLoop(unittest.TestCase):
    def test_reproduces_gen_oof(self):
        import p02_rank_fuse as rf
        import p04_rank_fuse as p4

        class X:
            def __init__(s, n, y, fold):
                s.n, s.y, s.fold = n, y, fold

            def __len__(s):
                return len(s.n)
        rng = np.random.default_rng(10)

        def qs(nq):
            y = rng.integers(0, 10, nq)
            m = rng.normal(size=(nq, 10)) + 2.0 * np.eye(10)[y]
            E = [rng.normal(size=(nq, 10)) + np.eye(10)[y] for _ in range(2)]
            return X(rng.choice([5.0, 10.0, 15.0, 20.0], nq), y, rng.integers(0, 5, nq)), m, E
        Xf, mf, Ef = qs(400)
        Xe, me, Ee = qs(200)
        Xm, mm, Em = qs(150)
        P_ref, folds_ref = p4.gen_oof_record(Xf, Xe, mf, me, Ef, Ee)
        Pm_ref = p4.gen_fold_predict(folds_ref, mm, Em, Xm.n, Xm.fold)
        P, Pm, folds = K.fold_loop(p4, rf, Xf, Xe, Xm, mf, me, mm, Ef, Ee, Em, ct=None)
        self.assertTrue(np.array_equal(P, P_ref))
        self.assertTrue(np.array_equal(Pm, Pm_ref))
        self.assertEqual([f["theta"] for f in folds], [f["theta"] for f in folds_ref])
        # with a CT term the fold-f fit sees only the nested CT of folds != f
        ct = {"ct_fit": {f: np.where((Xf.fold == f)[:, None], np.nan, rng.normal(size=(400, 10))) for f in range(5)},
              "ct_eval": rng.normal(size=(200, 10)), "ct_mono": rng.normal(size=(150, 10))}
        P2, Pm2, folds2 = K.fold_loop(p4, rf, Xf, Xe, Xm, mf, me, mm, Ef, Ee, Em, ct=ct)
        self.assertFalse(np.isnan(P2).any() or np.isnan(Pm2).any())
        self.assertEqual(len(folds2[0]["theta"]), 2 + 3 + 9)


if __name__ == "__main__":
    unittest.main()
