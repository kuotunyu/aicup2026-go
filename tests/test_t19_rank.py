"""Unit tests of tools/t19_rank.py (PREREG_19 4.2; rev 3) on synthetic data and the committed prereg19 records (read
only): the registered constants, the pins, the Q19 core, the SET seam record, the write guards, the modes and the
bug-revision rule. Never reads data/local, never runs apply on real data.
Run: python -m pytest tests/test_t19_rank.py
"""

import argparse
import json
import os
import sys
import unittest

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t19_rank as T  # noqa: E402
import p19_forward as SF  # noqa: E402

P19 = os.path.join(REPO, "results", "prereg19", "rank")


def frozen():
    with open(os.path.join(P19, "frozen_config.json"), encoding="utf-8") as f:
        return json.load(f)


class Registered(unittest.TestCase):
    def test_constants(self):
        fc = frozen()
        self.assertEqual(T.TERMS, fc["combo"]["terms"])
        self.assertEqual(T.RHO_MONO, fc["direction_ref"]["rho_mono"])
        self.assertEqual(T.RHO_EVAL, fc["direction_ref"]["rho_eval"])
        self.assertEqual(T.RHO_FLOOR, 0.155)
        self.assertEqual(T.K6_19.isoformat(), "2026-11-19T23:59:59+08:00")
        self.assertEqual(T.P19_MIN_REV, 4)
        self.assertEqual(T.TOL_Q09, 1e-12)
        self.assertAlmostEqual(max(0.5 * T.RHO_MONO, T.RHO_FLOOR), 0.4792752582816624, places=15)

    def test_pins_equal_the_committed_files(self):
        for rel, want in T.P19_PINS.items():
            with self.subTest(rel):
                self.assertEqual(T.pin_sha(T.path_of(rel)), want)
        fc, models, rec = T.check_p19_pins()
        self.assertEqual(len(models), 3)
        self.assertTrue(rec["adopted"]["apply_usable"])

    def test_official_log_agrees(self):
        import official_log as ol
        self.assertEqual(ol.T19_RHO_MONO, T.RHO_MONO)
        self.assertEqual(ol.T19_FORMAT, T.FORMATS["apply"])
        self.assertEqual(ol.T19_METHOD, T.METHOD)
        self.assertEqual(list(ol.T19_COMBO_SOURCES), T.COMBO["sources"])
        self.assertEqual(ol.K6_19, T.K6_19)
        self.assertEqual(ol.P19_MIN_REV, T.P19_MIN_REV)
        self.assertEqual(ol.T19_ADOPTION_SHA, T.P19_PINS["results/prereg19/rank/apply_report.json"])
        self.assertIn("results/test19/rank", ol.T14_BASE_DIRS)
        import t14_copy
        self.assertIn("results/test19/rank", t14_copy.BASE_DIRS)


class Core(unittest.TestCase):
    def test_q19_core_matches_the_fusion(self):
        rng = np.random.default_rng(0)
        Q, L = 6, 10
        fc = frozen()
        fz = {"S": sparse.identity(Q, format="csr"), "ok": np.array([True] * 5 + [False]), "n": np.full(Q, 10.0),
              "m": rng.standard_normal((Q, L)), "E": {t: rng.standard_normal((Q, L)) for t in ("OP", "O4", "SWA",
                                                                                          "SWB")},
              "P": rng.dirichlet(np.ones(L), Q)}
        vk = {"groups": [(f"q{i}", [i]) for i in range(Q)]}
        hq = {"E_hs": rng.standard_normal((Q, L))}
        q9 = {"E_p1": rng.standard_normal((Q, L)), "P": rng.dirichlet(np.ones(L), Q)}
        E_set = rng.standard_normal((Q, L))
        has = np.array([True, True, True, True, False, True])
        r = T.q19_core(fz, vk, hq, q9, E_set, has, fc)
        Es = [fz["E"][t] for t in ("OP", "O4", "SWA", "SWB")] + [hq["E_hs"], q9["E_p1"], E_set]
        want = T.r8.fuse_probs(fc["params"]["theta"], fz["m"], Es, fz["n"], fz["ok"])
        np.testing.assert_array_equal(r["P"][:5], want[:5])
        self.assertTrue(np.isnan(r["P"][5]).all())
        self.assertEqual(r["rows"][5][3:], [""] * L)
        self.assertEqual(r["info"]["questions_without_set_game"], 1)
        self.assertEqual(int(r["has"].sum()), 4)

    def test_set_seam_record(self):
        models = SF.load_models(P19)
        mp = {"mu_x": np.zeros(SF.FW.X_DIM), "sd_x": np.ones(SF.FW.X_DIM)}
        rng = np.random.default_rng(1)
        d = {"means": rng.standard_normal((5, 2, SF.FW.F_DIM)).astype(np.float32),
             "counts": np.array([[3, 3]] * 5), "failed": np.zeros(5, dtype=bool)}
        rec = T.set_seam_record(d, dict(d), models, mp)
        self.assertEqual((rec["max_abs_lp_set"], rec["usable_games"], rec["judged"]), (0.0, 5, False))
        e = dict(d, means=d["means"] + np.float32(0.5))
        self.assertGreater(T.set_seam_record(d, e, models, mp)["max_abs_lp_set"], 0.0)


class Guards(unittest.TestCase):
    def test_forbidden_areas(self):
        for rel in ("results/prereg19/rank/x.json", "results/test09/rank/x.csv", "results/test14/x",
                    "data/local/x", "data/test08/p1/x", "results/score_ledger.csv"):
            with self.subTest(rel):
                self.assertIsNotNone(T.forbidden_area(T.path_of(rel)))
        self.assertIsNone(T.forbidden_area(T.path_of("results/test19/rank/x.csv")))
        self.assertIsNone(T.forbidden_area(T.path_of("results/test19_rehearsal/x")))

    def test_label_path_refused(self):
        with self.assertRaises(T.T19Error):
            T.refuse_label_path(os.path.join("x", "rank_val_" + "ans" + "wers.csv"))

    def test_bug_revision_rule(self):
        with self.assertRaises(T.T19Error) as c:
            T.check_bug_revision(3, [])
        self.assertIn("rev 4 or later", str(c.exception))
        self.assertIsNone(T.check_bug_revision(None, []))

    def ns(self, **kw):
        base = dict(dry_run=False, rehearsal=False, results_dir=None, t09_results_dir=None, t08_results_dir=None,
                    p1_data_dir=None, t07_results_dir=None, t04_results_dir=None, t02_results_dir=None,
                    t02_out_dir=None, tests_dir=None, checks_dir=None, r20_uploaded_at=None, now=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_modes(self):
        ctx = T.resolve(self.ns(), "status")
        self.assertEqual((ctx["mode"], ctx["R19"], ctx["R9"]), ("real", T.REAL_R19, T.REAL_R9))
        with self.assertRaises(T.T19Error):
            T.resolve(self.ns(results_dir=os.path.join(REPO, "x")), "apply")
        with self.assertRaises(T.T19Error):
            T.resolve(self.ns(dry_run=True, results_dir=os.path.join(REPO, "results", "test19")), "apply")
        with self.assertRaises(T.T19Error):
            T.resolve(self.ns(dry_run=True, rehearsal=True), "apply")
        with self.assertRaises(T.T19Error):
            T.resolve(self.ns(now="2026-11-20T00:00"), "apply")


class Sources(unittest.TestCase):
    def test_imports(self):
        import ast
        with open(os.path.join(REPO, "tools", "t19_rank.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        got = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                got |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                got.add(n.module.split(".")[0])
        for bad in ("torch", "u1_lib", "p09_map", "p19_set", "p06_player", "t06_player", "t10_player", "score"):
            self.assertNotIn(bad, got)


if __name__ == "__main__":
    unittest.main()
