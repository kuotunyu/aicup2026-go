"""Unit tests of tools/t25_rank.py (PREREG_25 4.2) without test data: modes and directories, the write guards, the
data/local refusal, the registered values against the committed prereg25 files, the pure comparison helpers, the
bound-path mapping and the bug-revision rule. Never runs apply on data; never opens data/local or a label file.
Run: python -m pytest tests/test_t25_rank.py
"""

import argparse
import ast
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t25_rank as t25  # noqa: E402


def ns(**k):
    base = {"cmd": "apply", "dry_run": False, "results_dir": None, "now": None, "rerun": False, "reason": None,
            "bug_revision": None}
    base.update({f.lstrip("-").replace("-", "_"): None for f in t25.DIR_FLAGS.values()})
    base.update(k)
    return argparse.Namespace(**base)


class Modes(unittest.TestCase):
    def test_real_paths(self):
        ctx = t25.resolve(ns(), "apply")
        self.assertTrue(ctx["real"])
        self.assertTrue(t25.t2.same_path(ctx["R25"], os.path.join(REPO, "results", "test25")))
        self.assertTrue(t25.t2.same_path(ctx["R19"], os.path.join(REPO, "results", "test19")))
        self.assertEqual(ctx["write_roots25"], [ctx["R25"]])

    def test_real_refuses_other_dirs(self):
        with self.assertRaises(t25.T25Error) as c:
            t25.resolve(ns(results_dir=tempfile.gettempdir()), "apply")
        self.assertEqual(c.exception.code, 2)
        with self.assertRaises(t25.T25Error):
            t25.resolve(ns(t19_results_dir=tempfile.gettempdir()), "apply")
        with self.assertRaises(t25.T25Error):
            t25.resolve(ns(now="2026-11-30T00:00"), "apply")

    def test_dry_run_needs_every_dir_and_outside_repo(self):
        with self.assertRaises(t25.T25Error):
            t25.resolve(ns(dry_run=True, results_dir=os.path.join(tempfile.gettempdir(), "x25")), "apply")
        dirs = {f.lstrip("-").replace("-", "_"): os.path.join(REPO, "results") for f in t25.DIR_FLAGS.values()}
        with self.assertRaises(t25.T25Error):              # R25 inside the repository
            t25.resolve(ns(dry_run=True, results_dir=os.path.join(REPO, "results", "test25"), **dirs), "apply")

    def test_data_local_refused(self):
        for p in (os.path.join(REPO, "data", "local"), os.path.join(REPO, "data", "local", "x.csv")):
            with self.assertRaises(t25.T25Error):
                t25.refuse_label_path(p)
        with self.assertRaises(t25.T25Error):
            t25.refuse_label_path(os.path.join(tempfile.gettempdir(), "rank_val_" + "ans" + "wers.csv"))

    def test_forbidden_area(self):
        R = os.path.join(REPO, "results")
        for d in ("test19", "test09", "test02", "prereg25", "test14"):
            self.assertIsNotNone(t25.forbidden_area(os.path.join(R, d, "rank", "x.csv")), d)
        self.assertIsNone(t25.forbidden_area(os.path.join(R, "test25", "rank", "x.csv")))
        self.assertIsNotNone(t25.forbidden_area(os.path.join(REPO, "data", "test08", "x")))
        self.assertIsNotNone(t25.forbidden_area(os.path.join(R, "score_ledger.csv")))

    def test_guard_write_outside_roots(self):
        ctx = t25.resolve(ns(), "apply")
        with self.assertRaises(t25.T25Error):
            t25.guard_write(ctx, os.path.join(REPO, "results", "test19", "rank", "x.json"))
        with self.assertRaises(t25.T25Error):
            t25.guard_write(ctx, os.path.join(tempfile.gettempdir(), "x.json"))


class Registered(unittest.TestCase):
    def test_values_equal_frozen_files(self):
        with open(os.path.join(REPO, "results", "prereg25", "rank", "frozen_config.json"), encoding="utf-8") as f:
            fc = json.load(f)
        self.assertEqual(t25.RHO_MONO, fc["direction_ref"]["rho_mono"])
        self.assertEqual(t25.RHO_EVAL, fc["direction_ref"]["rho_eval"])
        self.assertEqual(fc["combo"]["terms"], t25.TERMS)
        self.assertEqual(t25.RHO_FLOOR, 0.155)
        self.assertEqual(t25.K6_25.isoformat(), "2026-11-18T23:59:59+08:00")
        self.assertEqual(t25.P25_MIN_REV, 4)
        for rel, want in t25.P25_PINS.items():
            with self.subTest(rel):
                self.assertEqual(t25.pin_sha(t25.path_of(rel)), want)

    def test_pins_and_adoption(self):
        fc, models, rec = t25.check_p25_pins()
        self.assertEqual(len(models), 3)
        self.assertTrue(rec["adopted"]["usable"])
        fc19, models19, _ = t25.check_p19_pins()
        self.assertEqual(fc19["combo"]["id"], "Q19")
        self.assertEqual(len(models19), 3)

    def test_source_discipline(self):
        with open(os.path.join(REPO, "tools", "t25_rank.py"), encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        self.assertEqual(mods & {"torch", "u1_lib", "p09_map", "p19_set", "p25_set", "score"}, set())
        low = src.lower()
        self.assertNotIn("rank_val_questions", low)
        self.assertNotIn("seam_questions", low)
        self.assertNotIn('"seam_val", "public"', low)
        self.assertNotIn("t9.prereqs", src)
        self.assertNotIn("check_test02", src)


class Pure(unittest.TestCase):
    def test_compare_lp(self):
        a = {"question_id": ["q1", "q2"], "lp": np.zeros((2, 10)), "has": np.array([True, False]),
             "usable": np.array([True, True, False])}
        b = {k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in a.items()}
        self.assertTrue(t25.compare_lp(a, b)["equal"])
        b["lp"][0, 3] = 2e-12
        self.assertFalse(t25.compare_lp(a, b)["equal"])
        b["lp"][0, 3] = 0.0
        b["lp"][1, 3] = 5.0                                # a question without a usable game is not compared
        self.assertTrue(t25.compare_lp(a, b)["equal"])
        b["usable"] = np.array([True, False, False])
        self.assertFalse(t25.compare_lp(a, b)["equal"])

    def test_bound_path(self):
        ctx = {"real": False, "R19": "/a19", "R9": "/a9", "R8": "/a8", "R7": "/a7", "R4": "/a4", "C": "/c",
               "D2": "/d2", "O2": "/o2"}
        j = lambda *p: os.path.join(*p)   # noqa: E731
        self.assertEqual(t25.bound_path(ctx, "results/test19/rank/x.csv"), j("/a19", "rank", "x.csv"))
        self.assertEqual(t25.bound_path(ctx, "results/test02/checks/rank/t1.json"), j("/c", "rank", "t1.json"))
        self.assertEqual(t25.bound_path(ctx, "results/test02/rank/apply_test_report.json"),
                         j("/o2", "apply_test_report.json"))
        self.assertEqual(t25.bound_path(ctx, "results/test02/rank/public/dumps/v.npz"),
                         j("/d2", "rank", "public", "dumps", "v.npz"))
        self.assertEqual(t25.bound_path(ctx, "results/prereg19/rank/frozen_config.json"),
                         t25.path_of("results/prereg19/rank/frozen_config.json"))
        self.assertIsNone(t25.bound_path(ctx, "results/test99/x"))

    def test_bug_revision_needs_rev4(self):
        with self.assertRaises(t25.T25Error):
            t25.check_bug_revision(3, [])
        with self.assertRaises(t25.T25Error):                # no rev 4 row at HEAD
            t25.check_bug_revision(4, [])
        self.assertIsNone(t25.check_bug_revision(None, []))


if __name__ == "__main__":
    unittest.main()
