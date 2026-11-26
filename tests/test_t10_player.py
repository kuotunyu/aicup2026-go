"""Tests for tools/t10_player.py and tools/t10_fake_tests.py (PREREG_10 1.1-1.3, 4.2; fast part).

* ARule: the PREREG_10 1.1 conditions (t10_player.a_conditions) on synthetic test02 apply reports built with the
  frozen PREREG_02 guard and p02_player_fuse.guard_decision: all-black level 3 by rho -> A; b_q != 1, level 1 / 2,
  level 3 by colour, a failed test02 gate (Q00, level 0), a CNN-only selection (level 0), T1 not usable, 06 not usable
  -> no A; inconsistencies are errors (exit 2): T4 missing or disagreeing, the guard recomputed from the report's rho
  differing from guard_level / guard_reason / applied, other thresholds in the report.
* Modes: the directory rules (real mode takes the defaults only; a dry run needs --out-dir outside the repository),
  the write guard (never results/test02, results/test06, data/test06, results/prereg*, data/local, the ledger, the
  upload log), the repository's B adopted through PREREG_13 (PREREG_10 rev 5; read with status), the registered
  constants shared with tools/official_log.py.
* FakeKeys: tools/t10_fake_tests.py's opening key is the 8-symmetry canonical first-4-ply string (x0_open_keys.py).
The end-to-end drills are tests/test_t10_dryrun.py (C10_SLOW=1). Nothing here opens a label file or data/local, or
runs tools/score.py.
Run: python -m pytest tests/test_t10_player.py
"""

import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t10_player as t10  # noqa: E402
import t10_fake_tests as f10  # noqa: E402
import official_log as ol  # noqa: E402
import p02_player_fuse as pf  # noqa: E402
import p02_moves as pm  # noqa: E402

with open(os.path.join(REPO, "results", "prereg02", "player", "frozen_config.json"), encoding="utf-8") as _f:
    GUARD = json.load(_f)["guard"]
USABLE = (True, {"drill": "06 usable"})


def report(rho_raw=0.70, rho_canon=0.69, b_c=1.0, b_q=1.0, eff="Q04", gate=True):
    """test02's apply_report.json fields as tools/t02_player.py writes them (applied / level / reason from
    p02_player_fuse.guard_decision with the frozen guard)."""
    rho = {"raw": rho_raw, "canon": rho_canon}
    combo = pf.COMBO_BY_ID[eff]
    applied, level, why = pf.guard_decision(combo, rho, GUARD, b_c, b_q, True)
    return {"b_c_candidate_black_share": b_c, "b_q_question_black_share": b_q, "guard_level": level,
            "guard_reason": why, "rho": rho, "effective": dict(combo), "applied": dict(applied),
            "gate_passed": gate, "guard": copy.deepcopy(GUARD)}


def t4(b_c=1.0, b_q=1.0):
    return {"b_c": b_c, "b_q": b_q, "black_only_pool": bool(b_c == 1.0 and b_q == 1.0)}


class ARule(unittest.TestCase):
    def check(self, ar, t4p=None, verdict="PASS", usable=USABLE):
        return t10.a_conditions(ar, t4(ar["b_c_candidate_black_share"], ar["b_q_question_black_share"])
                                if t4p is None else t4p, GUARD, verdict, usable)

    def test_all_black_level3_by_rho_holds(self):
        ar = report()
        self.assertEqual(ar["guard_level"], 3)
        self.assertIn("< theta2", ar["guard_reason"])
        r = self.check(ar)
        self.assertEqual(r["errors"], [])
        self.assertTrue(r["holds"], r)
        self.assertEqual(set(r["conditions"]), set(t10.A_CONDITIONS))
        self.assertEqual(r["conditions"]["level3_by_rho"]["orientation"], "raw")
        self.assertAlmostEqual(r["conditions"]["level3_by_rho"]["theta2"], GUARD["thresholds"]["raw"]["theta2"])
        for v in ("NORMALIZED",):
            self.assertTrue(self.check(ar, verdict=v)["holds"])

    def test_canon_orientation(self):
        eff = next(c["id"] for c in pf.COMBOS if c["e"] == "E0" and c["o"] == "canon")
        r = self.check(report(eff=eff, rho_raw=1.0, rho_canon=0.70))
        self.assertTrue(r["holds"], r)
        self.assertEqual(r["conditions"]["level3_by_rho"]["orientation"], "canon")

    def test_b_q_not_one(self):
        ar = report(b_q=0.9997647058823529)
        self.assertEqual(ar["guard_level"], 3)
        r = self.check(ar)
        self.assertEqual(r["errors"], [])
        self.assertFalse(r["holds"])
        self.assertFalse(r["conditions"]["all_black"]["ok"])
        self.assertTrue(r["conditions"]["level3_by_rho"]["ok"])

    def test_level3_by_colour_is_not_a(self):
        ar = report(rho_raw=1.0, rho_canon=1.0, b_c=1.0, b_q=0.3)
        self.assertEqual(ar["guard_level"], 3)
        self.assertIn("|b_c - b_q|", ar["guard_reason"])
        r = self.check(ar)
        self.assertFalse(r["holds"])
        self.assertFalse(r["conditions"]["all_black"]["ok"])
        self.assertFalse(r["conditions"]["level3_by_rho"]["ok"])

    def test_levels_1_and_2_are_not_a(self):
        th = GUARD["thresholds"]["raw"]
        for rho, level in ((1.01, 1), ((th["theta1"] + th["theta2"]) / 2, 2)):
            ar = report(rho_raw=rho)
            self.assertEqual(ar["guard_level"], level)
            r = self.check(ar)
            self.assertEqual(r["errors"], [])
            self.assertFalse(r["holds"], level)
            self.assertFalse(r["conditions"]["level3_by_rho"]["ok"])

    def test_level0_is_not_a(self):
        ar = report(eff="Q00")                                     # CNN-only selection: exempt
        self.assertEqual(ar["guard_level"], 0)
        r = self.check(ar)
        self.assertEqual(r["errors"], [])
        self.assertFalse(r["holds"])
        ar = report(eff="Q00", gate=False)                         # test02's gate failed: Q00
        self.assertFalse(self.check(ar)["holds"])

    def test_t1_and_06_usable(self):
        self.assertFalse(self.check(report(), verdict="FAIL_OPENING")["holds"])
        r = self.check(report(), usable=(False, {"t06_seam_passed": False}))
        self.assertFalse(r["holds"])
        self.assertFalse(r["conditions"]["06_usable"]["ok"])

    def test_t4_missing_or_disagreeing_is_an_error(self):
        ar = report()
        r = t10.a_conditions(ar, None, GUARD, "PASS", USABLE)
        self.assertFalse(r["holds"])
        self.assertTrue(any("T4" in e for e in r["errors"]))
        r = self.check(ar, t4p=t4(1.0, 0.99))
        self.assertTrue(any("T4 records" in e for e in r["errors"]))
        r = self.check(report(b_q=0.99), t4p=t4(1.0, 1.0))
        self.assertTrue(any("T4 records" in e for e in r["errors"]))
        r = t10.a_conditions(report(b_q=0.99), None, GUARD, "PASS", USABLE)    # not black, no T4: no error
        self.assertEqual(r["errors"], [])

    def test_guard_recomputed_must_match(self):
        ar = report()
        ar["guard_reason"] = "rho 0.7000 < theta2 0.9000: CNN only (E part)"
        r = self.check(ar)
        self.assertFalse(r["holds"])
        self.assertTrue(any("recomputed" in e for e in r["errors"]))
        ar = report()
        ar["rho"]["raw"] = 0.95                                    # level 3 claimed, the rho says level 1
        self.assertTrue(any("recomputed" in e for e in self.check(ar)["errors"]))
        ar = report()
        ar["applied"] = dict(pf.COMBO_BY_ID["Q04"])
        self.assertTrue(any("recomputed" in e for e in self.check(ar)["errors"]))
        ar = report()
        ar["guard"]["thresholds"]["raw"]["theta2"] = 0.5
        self.assertTrue(any("thresholds" in e for e in self.check(ar)["errors"]))
        ar = report()
        ar["effective"] = {"id": "Q04", "e": "E1", "o": "raw", "w": 1.0}
        self.assertTrue(any("not a p02 combination" in e for e in self.check(ar)["errors"]))

    def test_rho_missing(self):
        ar = report(rho_raw=None, rho_canon=None)
        self.assertEqual(ar["guard_level"], 1)
        r = self.check(ar)
        self.assertEqual(r["errors"], [])
        self.assertFalse(r["holds"])


class Modes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t10_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = t10.main(argv)
        return code, out.getvalue()

    def test_b_adopted_via_prereg13(self):
        """PREREG_10 rev 5 (was the rev 4 tripwire test_p1_is_b_not_adopted): H10P never ran, H13P adopted B', so the
        repository's B is adopted through PREREG_13. Read with status (writes nothing); the real p1 is not run here (it
        would go on to the test-day checks and the 4090)."""
        tb = t10.tb
        self.assertFalse(tb.results_md_adopted(tb.p10_paths(tb.REAL_P10)["results_md"]))
        self.assertTrue(tb.results_md_adopted(tb.p13_paths(tb.REAL_P13)["results_md13"], tb.MARK13_RE))
        code, out = self.run_main(["status"])
        self.assertEqual(code, 0, out)
        self.assertIn("(real; B adopted)", out)
        self.assertIn("B: adopted via PREREG_13", out)
        self.assertFalse(os.path.exists(tb.REAL_P1_OUT))

    def test_real_mode_takes_the_defaults_only(self):
        code, out = self.run_main(["status", "--out-dir", self.tmp])
        self.assertEqual(code, 2)
        self.assertIn("real mode", out)
        code, out = self.run_main(["status", "--t06-out-dir", self.tmp])
        self.assertEqual(code, 2)
        self.assertIn("real mode", out)

    def test_dry_run_dirs(self):
        base = ["--dry-run", "--t06-out-dir", os.path.join(self.tmp, "o6"), "--t06-data-dir",
                os.path.join(self.tmp, "a6")]
        code, out = self.run_main(["status"] + base)
        self.assertEqual(code, 2)
        self.assertIn("--out-dir", out)
        code, out = self.run_main(["status"] + base + ["--out-dir", os.path.join(REPO, "results", "test10", "x")])
        self.assertEqual(code, 2)
        self.assertIn("inside the repository", out)
        code, out = self.run_main(["status"] + base + ["--out-dir", os.path.join(self.tmp, "o10"),
                                                       "--now", "2026-11-22T00:00:00+08:00"])
        self.assertEqual(code, 0, out)
        self.assertIn("K6-10", out)
        self.assertIn("已過", out)

    def test_write_guard(self):
        ctx = {"write_roots": [os.path.join(self.tmp, "o10")]}
        for p in (os.path.join(REPO, "results", "test02", "player", "x.json"),
                  os.path.join(REPO, "results", "test06", "player", "x.json"),
                  os.path.join(REPO, "data", "test06", "player", "x.npz"),
                  os.path.join(REPO, "results", "prereg10", "x.json"), os.path.join(REPO, "results", "prereg06", "x"),
                  os.path.join(REPO, "results", "official_scores.json"),
                  os.path.join(REPO, "results", "score_ledger.csv"),
                  os.path.join(self.tmp, "elsewhere", "x.json"),
                  os.path.join(self.tmp, "o10", "player_val_" + "ans" + "wers.csv")):
            with self.subTest(p), self.assertRaises(t10.T10Error) as cm:
                t10.guard_write(ctx, p)
            self.assertEqual(cm.exception.code, 2)
        self.assertTrue(t10.guard_write(ctx, os.path.join(self.tmp, "o10", "public", "top5.csv")))

    def test_registered_constants(self):
        self.assertEqual(t10.METHOD, ol.T10_METHOD)
        self.assertEqual(t10.METHOD, "prereg10 10: 06-OFF + all-black level-3 lift (A) [+ P1 LDA (B)] (甲)")
        self.assertEqual(t10.FORMATS["submit"], ol.T10_FORMAT)
        self.assertEqual(t10.FORMATS["seam"], ol.T10_SEAM_FORMAT)
        self.assertEqual(t10.FORMATS["pool"], ol.T10_POOL_FORMAT)
        self.assertEqual(t10.K6_10, ol.K6_10)
        self.assertEqual(t10.K5_10, ol.K5_10)
        self.assertEqual(t10.TESTDAY_REV_ROW, ol.P10_TESTDAY_ROW)
        self.assertEqual(t10.K6_10.isoformat(), "2026-11-21T23:59:59+08:00")
        self.assertEqual(t10.K6B_10.isoformat(), "2026-11-14T23:59:59+08:00")
        self.assertEqual(t10.K5_10.isoformat(), "2026-10-31T23:59:59+08:00")
        self.assertEqual(t10.REAL_O10, os.path.join(REPO, "results", "test10", "player"))
        self.assertEqual(ol.P10_MIN_REV, 2)
        self.assertEqual(set(ol.T10_SOURCES) | {"10B"}, set(t10.SOURCES))


class FakeKeys(unittest.TestCase):
    def test_open_key(self):
        a = "(;GM[1];B[pd];W[dp];B[pp];W[dd];B[qf])"
        b = "(;GM[1];B[dd];W[pp];B[dp];W[pd];B[cf])"     # a rotated / mirrored version of the same 4 plies
        self.assertEqual(f10.open_key(a), pm.points_string(pm.canon(pm.parse_moves(a, 12)[:4], 4)))
        self.assertEqual(len(f10.open_key(a)), 8)
        self.assertEqual(f10.open_key(a), f10.open_key(b))
        self.assertNotEqual(f10.open_key(a), f10.open_key("(;GM[1];B[qd];W[dp];B[pp];W[dd])"))

    def test_drill_kinds(self):
        self.assertEqual(f10.DRILLS, ("level0", "fail_opening", "t06_seam_failed"))


class T06OffWithoutReport(unittest.TestCase):
    def test_used_06_pool_without_off_report_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="c10-t10-off-") as root:
            pool_path = os.path.join(root, "public", "t06_pool.json")
            ctx = {"O6": root}
            with mock.patch.object(t10, "TEST_POOLS", ("public",)), \
                    mock.patch.object(t10, "t06_paths", return_value={"pool": pool_path}), \
                    mock.patch.object(t10, "maybe_json", return_value={"used06": True}):
                with self.assertRaises(t10.T10Error) as caught:
                    t10.t06_off_state(ctx, {})
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("there is no 06-OFF submit report", str(caught.exception))

    def test_unused_06_pool_without_off_report_keeps_q04_fallback(self):
        with tempfile.TemporaryDirectory(prefix="c10-t10-off-") as root:
            pool_path = os.path.join(root, "public", "t06_pool.json")
            ctx = {"O6": root}
            with mock.patch.object(t10, "TEST_POOLS", ("public",)), \
                    mock.patch.object(t10, "t06_paths", return_value={"pool": pool_path}), \
                    mock.patch.object(t10, "maybe_json", return_value={"used06": False}):
                self.assertEqual(t10.t06_off_state(ctx, {}), (None, {"exists": False}))


if __name__ == "__main__":
    unittest.main()
