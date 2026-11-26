"""prereg45 explore: small CPU tests of p45_eval.py (synthetic data only; no score of a prereg45 variant).

    PY -m pytest -q results/prereg45/explore/test_p45.py
"""
import datetime
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p45_eval as PE  # noqa: E402

F10, P39E, st = PE.F10, PE.P39E, PE.st


def synth(n_players=40, per=12, dim=150, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1, (n_players, dim))
    lab = np.repeat(np.array(["p%03d" % i for i in range(n_players)]), per)
    X = (centres[np.repeat(np.arange(n_players), per)] + rng.normal(0, 2, (n_players * per, dim))).astype(np.float32)
    X[:, 5] = 3.0                                   # a constant column (sd < 1e-9 branch)
    usable = rng.random(len(X)) > 0.05
    usable[lab == "p000"] = False
    usable[np.flatnonzero(lab == "p001")[1:]] = False   # p001 keeps one usable game -> dropped (min 2)
    return X, lab, usable


def aligned_diff(A, B):
    sign = np.sign(np.sum(A * B, 0))
    return float(np.abs(A * sign - B).max())


class BlockedLda(unittest.TestCase):
    def test_equals_fit_p1_lda(self):
        X, lab, usable = synth(n_players=200, per=8, dim=90)   # players > dims: no degenerate eigenspace
        ref = F10.fit_p1_lda(X, lab, usable)
        got = PE.fit_lda_blocked(X, lab, usable, 64, block=37)
        self.assertEqual((got["n_games"], got["n_players"]), (ref["n_games"], ref["n_players"]))
        for k in ("m", "s", "mu", "sd"):
            np.testing.assert_allclose(got[k], ref[k], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(got["ev"], ref["ev"], rtol=1e-8, atol=1e-10)
        Zr = F10.project_p1(X, ref)
        Zg = PE.project_lda(X, got, block=53)
        self.assertEqual(Zg.shape, Zr.shape)
        self.assertLess(aligned_diff(Zg, Zr), 1e-7 * max(1.0, np.abs(Zr).max()))

    def test_128_extends_64(self):
        X, lab, usable = synth(dim=200)
        a = PE.fit_lda_blocked(X, lab, usable, 64)
        b = PE.fit_lda_blocked(X, lab, usable, 128)
        self.assertEqual(b["W"].shape, (200, 128))
        np.testing.assert_array_equal(a["W"], b["W"][:, :64])
        self.assertEqual(PE.project_lda(X, b).shape, (len(X), 128))

    def test_keep_rows_min_games(self):
        _X, lab, usable = synth()
        idx, kept = PE._keep_rows(lab, usable)
        self.assertNotIn("p000", set(kept))
        self.assertNotIn("p001", set(kept))
        self.assertTrue(usable[idx].all())


class Mlp(unittest.TestCase):
    CFG = dict(PE.MLP, epochs=2, batch=64, hidden=32, out=16)

    def test_deterministic_and_l2(self):
        X, lab, usable = synth(dim=60)
        X0 = X.copy()
        m1, i1 = PE.train_mlp(X, lab, usable, self.CFG, log=lambda s: None)
        np.testing.assert_array_equal(X, X0)        # consume=False leaves X alone
        m2, _i2 = PE.train_mlp(X, lab, usable, self.CFG, log=lambda s: None)
        self.assertEqual(PE.model_sha(m1), PE.model_sha(m2))
        self.assertTrue(i1["deterministic"])
        self.assertEqual(len(i1["history"]), 2)
        Z = PE.project_mlp(X, m1, block=50)
        self.assertEqual(Z.shape, (len(X), 16))
        np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)
        m3, _i3 = PE.train_mlp(X, lab, usable, self.CFG, log=lambda s: None, consume=True)
        self.assertEqual(PE.model_sha(m1), PE.model_sha(m3))
        self.assertFalse(np.array_equal(X, X0))     # consumed (standardised in place)


class Models(unittest.TestCase):
    def test_save_load_roundtrip_and_tamper(self):
        X, lab, usable = synth()
        m = PE.fit_lda_blocked(X, lab, usable, 64)
        with tempfile.TemporaryDirectory() as d:
            rec = PE.save_model("A_LDA64", m, d)
            rec["file"] = os.path.join(d, "A_LDA64.npz")
            got = PE.load_model(rec)
            np.testing.assert_array_equal(PE.project(X, got), PE.project(X, m))
            bad = dict(rec, arrays_sha256="0" * 64)
            with self.assertRaises(SystemExit):
                PE.load_model(bad)


class Terms(unittest.TestCase):
    def test_z_official_equals_p1_terms_joint(self):
        rng = np.random.default_rng(3)
        X, lab, usable = synth(n_players=30, per=10, dim=40)
        models = {k: F10.fit_p1_lda(X, lab, usable) for k in ("joint", "B", "W")}
        colour = np.where(rng.random(len(X)) < 0.5, "B", "W")
        order = rng.permutation(len(X))
        cand = [order[i * 6:(i + 1) * 6].tolist() for i in range(30)]
        qs = [order[180 + i * 3:180 + (i + 1) * 3].tolist() for i in range(30)]
        want, hw = P39E.p1_terms(X, usable, colour, cand, qs, models, "b4")["joint"]
        got, hg = PE.z_official(F10.project_p1(X, models["joint"]), usable, cand, qs)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(hg, hw)


class Rules(unittest.TestCase):
    AFTER = datetime.datetime(2026, 11, 15, 22, 0)
    BEFORE = datetime.datetime(2026, 11, 15, 21, 59)

    def test_sets_choice(self):
        both = {"M1": True, "M2": True}
        self.assertEqual(PE.check_sets_choice("L,M1,M2", both, self.BEFORE), ("L", "M1", "M2"))
        with self.assertRaises(SystemExit):
            PE.check_sets_choice("L,M1", both, self.AFTER)          # M2 complete -> must use it
        m1 = {"M1": True, "M2": False}
        with self.assertRaises(SystemExit):
            PE.check_sets_choice("L,M1,M2", m1, self.AFTER)         # unfinished list never used
        with self.assertRaises(SystemExit):
            PE.check_sets_choice("L,M1", m1, self.BEFORE)           # fall-back only from 22:00
        self.assertEqual(PE.check_sets_choice("L,M1", m1, self.AFTER), ("L", "M1"))
        with self.assertRaises(SystemExit):
            PE.check_sets_choice("L,M1", {"M1": False, "M2": False}, self.AFTER)
        with self.assertRaises(SystemExit):
            PE.check_sets_choice("L", both, self.AFTER)

    def test_chunks_complete(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "M9"))
            games = [{"mock_ref": -1}] * 401
            self.assertEqual(PE.chunks_complete("M9", games, d), (False, 0, 3))
            for i in range(3):
                open(os.path.join(d, "M9", "chunk_%05d.npz" % i), "wb").close()
            open(os.path.join(d, "M9", "chunk_00003.tmp.npz"), "wb").close()
            self.assertEqual(PE.chunks_complete("M9", games, d), (True, 3, 3))

    def test_variant_order_and_grid(self):
        self.assertEqual(PE.VARIANTS, ("A_LDA64", "A_LDA128", "B_MLP"))
        self.assertEqual(PE.WP_GRID, (0.25, 0.5, 1, 2, 4, 8, 16))
        self.assertEqual(PE.K_GRID, (10, 20, 40))
        q = {"A_LDA64": 0.7, "A_LDA128": 0.7, "B_MLP": 0.69}
        self.assertEqual(PE.first_max(list(PE.VARIANTS), key=lambda v: q[v]), "A_LDA64")

    def test_mlp_cfg_is_plan(self):
        self.assertEqual({k: PE.MLP[k] for k in ("hidden", "out", "scale", "lr", "wd", "batch", "epochs", "seed")},
                         {"hidden": 512, "out": 128, "scale": 30.0, "lr": 1e-3, "wd": 1e-4, "batch": 1024,
                          "epochs": 20, "seed": 0})


class Judge(unittest.TestCase):
    def test_e_use_six(self):
        prior = PE.e_judgements_so_far()
        self.assertEqual(prior, ["prereg36", "prereg37", "prereg38", "prereg39", "prereg43"])
        PE.judge_ready({"gate": {"pass": True}}, 6, prior)
        with self.assertRaises(SystemExit):
            PE.judge_ready({"gate": {"pass": True}}, 5, prior)
        with self.assertRaises(SystemExit):
            PE.judge_ready({"gate": {"pass": False}}, 6, prior)

    def test_rule(self):
        rng = np.random.default_rng(0)
        old = rng.uniform(0.6, 0.7, 20)
        self.assertTrue(PE.E43.judge_rule(old + 0.02, old)["adopt"])
        self.assertFalse(PE.E43.judge_rule(old + 0.009, old)["adopt"])
        new = old + 0.03
        new[:3] -= 0.04                              # higher in 17 of 20
        self.assertFalse(PE.E43.judge_rule(new, old)["adopt"])


class Records(unittest.TestCase):
    def test_v26_records(self):
        b = PE.v26()
        self.assertEqual((b["wP"], b["k"]), (8, 20))
        self.assertEqual(float(np.mean(b["d_per_pool"])), PE.D_V26)
        self.assertEqual(float(np.mean(b["e_per_pool"])), PE.E_V26)
        self.assertEqual(len(b["e_per_pool"]), 20)

    def test_no_prereg45_results_yet(self):
        PE.no_e_results_yet()


if __name__ == "__main__":
    unittest.main()
