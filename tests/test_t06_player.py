"""Tests for tools/t06_player.py and tools/t06_fake_tests.py (results/prereg06/design/spec.md 7 and 9.3; fast part).

* Rules: the PREREG_06 2.7 table (pool_precheck), the 5.3 conditions (on_conditions), T2's summary, the data/local and
  label-file guards, the directory and mode rules, the exit codes of main().
* Veto: tools/t06_player.py veto in --dry-run on synthetic logs and submit reports (PREREG_06 5.3: the same-file
  warning pauses with exit 2 until --confirmed-upload-check, d < 0 vetoes, d = 0 does not, K7' passed -> 06-OFF, the
  upload order and the one-per-season rule).
* RealMode: the real-mode prerequisites with the module's path constants pointed at temporary trees and git mocked
  (setattr): test02 not committed -> 2, Q04 not uploaded -> 2, git not clean -> 2, .venv versions differ -> 2,
  K6' passed -> 3 (and not_used.json), FAIL_OPENING -> 3.
* FakeT2: tools/t06_fake_tests.py fake-t2 counts (b), (c), (d) like tools/t02_checks.py on a tiny tree.
The end-to-end drills (from-val and official-shape trees) are in tests/test_t06_dryrun.py (C10_SLOW=1).
Nothing here opens a label file or runs tools/score.py; data/local is never opened.
Run: python -m pytest tests/test_t06_player.py
"""

import contextlib
import copy
import csv
import datetime as dt
import hashlib
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
sys.path.insert(0, HERE)
import t06_player as t6  # noqa: E402
import t06_fake_tests as f6  # noqa: E402
import test_t02_infra as ti  # noqa: E402

TW = dt.timezone(dt.timedelta(hours=8))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = t6.main(argv)
    return code, out.getvalue() + err.getvalue()


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t06_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)


# ================================================================ rules
class Rules(unittest.TestCase):
    def test_pool_precheck_table(self):
        cases = [({"gate_passed": False, "guard_level": 0}, (False, False), "gate failed"),
                 ({"gate_passed": True, "guard_level": 3}, (False, False), "level 3"),
                 ({"gate_passed": True, "guard_level": 0}, (False, False), "level 0"),
                 ({"gate_passed": True, "guard_level": 2}, (True, False), "level 2"),
                 ({"gate_passed": True, "guard_level": 1}, (True, True), "guard level 1")]
        for ar, (use, sw), msg in cases:
            with self.subTest(ar):
                u, why, s = t6.pool_precheck(ar)
                self.assertEqual((u, s), (use, sw))
                self.assertIn(msg, why)
        with self.assertRaises(t6.T06Error) as cm:
            t6.pool_precheck({"gate_passed": True, "guard_level": 7})
        self.assertEqual(cm.exception.code, 2)

    def pools(self):
        rep = {"used06": True, "selected": "V2", "switch_allowed_at_this_level": True, "n_questions": 400,
               "n_candidates": 400, "test02": {"n_questions": 400, "n_candidates": 400, "guard_level": 1},
               "detector": {"usable": True, "opens": True, "r_forced_cost": 1.2, "r_star": 2.25}}
        return copy.deepcopy(rep), copy.deepcopy(rep)

    def t2(self, c=(0, 0), failed=()):
        return {"c_copies": {"player_identification_test_public.csv": c[0],
                             "player_identification_test_private.csv": c[1]},
                "abd_failed": list(failed), "units_present": ["a", "b", "c", "d"], "verdict": "PASS", "fake": False}

    def test_on_conditions_all_true(self):
        pub, priv = self.pools()
        now = dt.datetime(2026, 11, 6, 12, 0, tzinfo=TW)
        c = t6.on_conditions(pub, priv, {"secondary_not_negative": True}, self.t2(), "page read", False, now)
        self.assertEqual(sorted(c), sorted(t6.ON_CONDITIONS))
        self.assertTrue(all(v["ok"] for v in c.values()), c)

    def test_on_conditions_each_one_false(self):
        now = dt.datetime(2026, 11, 6, 12, 0, tzinfo=TW)
        h6p = {"secondary_not_negative": True}

        def check(name, pub=None, priv=None, h=None, t2=None, page="page read", not_o2o=False, when=now):
            p0, p1 = self.pools()
            if pub:
                pub(p0)
            if priv:
                priv(p1)
            c = t6.on_conditions(p0, p1, h or h6p, self.t2() if t2 is None else t2, page, not_o2o, when)
            bad = [k for k, v in c.items() if not v["ok"]]
            self.assertIn(name, bad)
        check("private_switch_usable_and_detector_open", priv=lambda r: r["detector"].update(opens=False))
        check("private_switch_usable_and_detector_open", priv=lambda r: r["detector"].update(usable=False))
        check("private_switch_usable_and_detector_open", priv=lambda r: r.update(switch_allowed_at_this_level=False))
        check("private_switch_usable_and_detector_open", priv=lambda r: r.update(used06=False))
        check("h6p_secondary_not_negative", h={"secondary_not_negative": False})
        check("pools_400_by_400", pub=lambda r: r.update(n_candidates=399))
        check("pools_400_by_400", priv=lambda r: r["test02"].update(n_questions=401))
        check("t2_c_zero_and_abd_not_failed", t2=self.t2(c=(0, 2)))
        check("t2_c_zero_and_abd_not_failed", t2=self.t2(failed=["b:public"]))
        check("t2_c_zero_and_abd_not_failed", t2=dict(self.t2(), units_present=["a", "b", "d"]))
        p0, p1 = self.pools()
        c = t6.on_conditions(p0, p1, h6p, None, "page read", False, now)
        self.assertFalse(c["t2_c_zero_and_abd_not_failed"]["ok"])
        check("both_pools_06_at_guard_level_1", pub=lambda r: r["test02"].update(guard_level=2))
        check("both_pools_06_at_guard_level_1", pub=lambda r: r.update(used06=False))
        check("aidea_page_not_against_one_to_one", page="")
        check("aidea_page_not_against_one_to_one", not_o2o=True)
        check("before_k7", when=dt.datetime(2026, 11, 23, 0, 0, tzinfo=TW))

    def test_t2_summary(self):
        t2 = {"verdict": "FAIL", "units": [
            {"unit": "a", "file": "x.csv", "verdict": "FAIL", "copies": 99},
            {"unit": "b", "pool": "public", "verdict": "PASS", "copies": 0},
            {"unit": "c", "file": "player_identification_test_public.csv", "verdict": "PASS", "copies": 3},
            {"unit": "d", "verdict": "PASS", "copies": 0}]}
        s = t6.t2_summary(t2)
        self.assertEqual(s["abd_failed"], ["a:x.csv"])
        self.assertEqual(s["c_copies"], {"player_identification_test_public.csv": 3})
        self.assertEqual(s["units_present"], ["a", "b", "c", "d"])
        self.assertIsNone(t6.t2_summary(None))

    def test_guards(self):
        with self.assertRaises(t6.T06Error):
            t6.refuse_label(os.path.join("x", "player_val_" + "ans" + "wers.csv"))
        with self.assertRaises(t6.T06Error):
            t6.check_input(os.path.join(t6.LOCAL, "rank_val_questions.csv"))
        with self.assertRaises(t6.T06Error):           # the two game files only where seam_val allows them
            t6.check_input(t6.VAL_Q)
        self.assertEqual(t6.check_input(t6.VAL_Q, allow_local=True), t6.VAL_Q)
        real_local = os.path.realpath(t6.LOCAL)         # through the data junction of a worktree, too
        with self.assertRaises(t6.T06Error):
            t6.check_input(os.path.join(real_local, "player_val_candidates.csv"))
        self.assertTrue(t6.inside(os.path.join(real_local, "x"), t6.LOCAL))
        self.assertFalse(t6.inside(os.path.join(REPO, "data", "localx"), t6.LOCAL))

    def test_constants(self):
        self.assertEqual(t6.K6P.isoformat(), "2026-11-21T23:59:59+08:00")
        self.assertEqual(t6.K7P.isoformat(), "2026-11-22T23:59:59+08:00")
        self.assertEqual(t6.METHOD["off"], "prereg06 06-OFF: Q04 + v2 style + CSLS (甲)")
        self.assertEqual(t6.METHOD["on"], "prereg06 06-ON: 06-OFF + one-to-one switch (甲)")
        import official_log as ol
        self.assertEqual(ol.T06_METHOD, t6.METHOD)
        self.assertEqual(ol.T06_ON_CONDITIONS, t6.ON_CONDITIONS)
        self.assertEqual((ol.K6P, ol.K7P), (t6.K6P, t6.K7P))
        self.assertEqual(ol.T06_REPORT, t6.REPORT)
        self.assertEqual((ol.T06_FORMAT, ol.T06_SEAM_FORMAT, ol.T06_POOL_FORMAT, ol.T06_VETO_FORMAT),
                         (t6.FORMATS["submit"], t6.FORMATS["seam"], t6.FORMATS["pool"], t6.FORMATS["veto"]))
        self.assertEqual(t6.SEAM_MIN_IDENTICAL, 398)
        self.assertEqual(t6.GATE06_MIN, 0.0)

    def test_frozen_pins_match_the_committed_files(self):
        p6 = t6.REAL_P6
        self.assertEqual(t6.sha_lf(os.path.join(p6, "frozen_config.json")), t6.FROZEN_CONFIG_SHA_LF)
        for n, want in t6.P06_PINS_LF.items():
            got = t6.sha_bound(os.path.join(p6, n)) if n.endswith((".csv", ".json")) else t6.sha_raw(os.path.join(p6, n))
            self.assertEqual(got, want, n)
        with open(os.path.join(p6, "apply_report.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["frozen_config_sha256"], t6.FROZEN_CONFIG_SHA_LF)


class DirsAndModes(Tmp):
    def test_dry_run_needs_dirs_outside_the_repo(self):
        code, out = run_main(["feats", "--pool", "public", "--dry-run", "--out-dir", self.p("o6")])
        self.assertEqual(code, 2)
        self.assertIn("--dry-run needs", out)
        code, out = run_main(["status", "--dry-run", "--out-dir", os.path.join(REPO, "results", "test06", "x"),
                              "--data-dir", self.p("a6")])
        self.assertEqual(code, 2)
        self.assertIn("inside the repository", out)

    def test_real_mode_takes_the_defaults_only(self):
        code, out = run_main(["status", "--out-dir", self.p("o6")])
        self.assertEqual(code, 2)
        self.assertIn("real mode: --out-dir must be", out)
        code, out = run_main(["feats", "--pool", "public", "--now", "2026-11-05T00:00"])
        self.assertEqual(code, 2)
        self.assertIn("--now is for --dry-run and status only", out)

    def test_tests_dir_never_local_or_raw(self):
        for bad in (t6.LOCAL, os.path.join(self.tmp, "data", "test", "raw")):
            code, out = run_main(["status", "--dry-run", "--out-dir", self.p("o6"), "--data-dir", self.p("a6"),
                                  "--tests-dir", bad])
            self.assertEqual(code, 2, out)

    def test_status_dry_run_empty(self):
        code, out = run_main(["status", "--dry-run", "--out-dir", self.p("o6"), "--data-dir", self.p("a6"),
                              "--now", "2026-11-21T23:00:00+08:00"])
        self.assertEqual(code, 0)
        self.assertIn("feats seam_val: missing", out)
        self.assertIn("距 K6' 11/21 23:59（06 的正式執行要在這之前 commit）：0 天 0 小時", out)

    def test_guard_write(self):
        ctx = {"write_roots": [self.p("o6")]}
        with self.assertRaises(t6.T06Error):
            t6.guard_write(ctx, os.path.join(t6.RESULTS, "test02", "player", "x.json"))
        with self.assertRaises(t6.T06Error):
            t6.guard_write(ctx, os.path.join(t6.RESULTS, "prereg06", "player", "x.json"))
        with self.assertRaises(t6.T06Error):
            t6.guard_write(ctx, self.p("elsewhere", "x.json"))
        self.assertTrue(t6.guard_write(ctx, self.p("o6", "x.json")))


# ================================================================ veto
class Veto(Tmp):
    """PREREG_06 5.3 veto in --dry-run on a synthetic log (official_log.py's entry shape)."""

    def setUp(self):
        super().setUp()
        self.o6 = self.p("o6")
        self.log = self.p("official_scores.json")
        self.reports = {}
        for v, diff_q04 in (("off", {"public": 37, "private": 41}), ("on", {"public": 52, "private": 55})):
            self.reports[v] = write_json(os.path.join(self.o6, v, t6.REPORT), {
                "format": t6.FORMATS["submit"], "variant": v, "run_id": "r1", "top1_differences_vs_q04": diff_q04,
                "top1_differences_off_vs_on": {"public": 17, "private": 12}})

    def entry(self, eid, variant, at, score, run_id="r1", file=None):
        e = {"id": eid, "comp": "player", "status": "uploaded", "uploaded_at": at, "public_score": score,
             "file": file or f"results/test06/player/{variant}/submission_player.csv"}
        if variant:
            e["test06"] = {"variant": variant, "run_id": run_id, "report_sha256": sha(self.reports[variant])}
        return e

    def write_log(self, entries):
        write_json(self.log, {"format": "c10 official_scores v1", "uploads": entries,
                              "final_choice": {"rank": None, "player": None}})

    def veto(self, *extra, now="2026-11-07T12:00:00+08:00"):
        code, out = run_main(["veto", "--dry-run", "--out-dir", self.o6, "--data-dir", self.p("a6"), "--log",
                              self.log, "--now", now] + list(extra))
        rec = json.load(open(os.path.join(self.o6, "veto.json"), encoding="utf-8")) \
            if os.path.exists(os.path.join(self.o6, "veto.json")) else None
        return code, out, rec

    def base(self, off=0.90, on=0.91, q04=0.88):
        return [self.entry(1, None, "2026-11-04T16:05:00+08:00", q04, file=t6.Q04_FILE_REL),
                self.entry(2, "off", "2026-11-05T10:00:00+08:00", off),
                self.entry(3, "on", "2026-11-05T11:00:00+08:00", on)]

    def test_d_negative_vetoes(self):
        self.write_log(self.base(off=0.90, on=0.89))
        code, out, rec = self.veto()
        self.assertEqual(code, 0, out)
        self.assertTrue(rec["vetoed"])
        self.assertEqual(rec["final_variant"], "off")
        self.assertTrue(rec["reupload_off_needed"])
        self.assertAlmostEqual(rec["d"], 0.89 - 0.90, places=12)
        self.assertEqual(rec["entries"], {"q04": 1, "off": 2, "on": 3})

    def test_records_tools_and_env(self):
        """PREREG_06 rev 3 item 2: veto.json records the tool SHAs, git HEAD and versions (round-2 review, minor 6)."""
        for name, scores, extra in (("kept", dict(off=0.90, on=0.91), ()), ("paused", dict(off=0.90, on=0.90), ()),
                                    ("k7", dict(off=0.90, on=0.91), ("now",))):
            with self.subTest(name):
                if os.path.exists(os.path.join(self.o6, "veto.json")):
                    os.remove(os.path.join(self.o6, "veto.json"))
                self.write_log(self.base(**scores))
                _code, out, rec = self.veto(**({"now": "2026-11-23T00:00:01+08:00"} if extra else {}))
                self.assertIsNotNone(rec, out)
                self.assertEqual(rec["env"], t6.env_record())
                self.assertEqual(sorted(rec["tools"]["files_sha256_lf"]), sorted(t6.TOOLS_COMMITTED))
                self.assertEqual(rec["tools"]["files_sha256_lf"]["tools/t06_player.py"],
                                 t6.st.sha256_lf(os.path.join(REPO, "tools", "t06_player.py")))
                self.assertIn("git_head", rec["tools"])

    def test_d_zero_and_positive_keep_on(self):
        for on in (0.90, 0.95):
            with self.subTest(on=on):
                shutil.rmtree(os.path.join(self.o6, "veto.json"), ignore_errors=True)
                if os.path.exists(os.path.join(self.o6, "veto.json")):
                    os.remove(os.path.join(self.o6, "veto.json"))
                self.write_log(self.base(off=0.90, on=on))
                code, out, rec = self.veto(*(() if on == 0.95 else ("--confirmed-upload-check", "screens checked")))
                self.assertEqual(code, 0, out)
                self.assertFalse(rec["vetoed"])
                self.assertEqual(rec["final_variant"], "on")
                self.assertFalse(rec["reupload_off_needed"])

    def test_same_file_warning_pauses(self):
        for name, scores in (("off-on", dict(off=0.90, on=0.90)), ("q04-on", dict(q04=0.91, off=0.90, on=0.91)),
                             ("q04-off", dict(q04=0.90, off=0.90, on=0.93))):
            with self.subTest(name):
                if os.path.exists(os.path.join(self.o6, "veto.json")):
                    os.remove(os.path.join(self.o6, "veto.json"))
                self.write_log(self.base(**scores))
                code, out, rec = self.veto()
                self.assertEqual(code, 2, out)
                self.assertTrue(rec["paused"])
                self.assertFalse(rec["decided"])
                self.assertIn("possible wrong file", out)
                self.assertTrue(rec["check_items"])
                code, out, rec = self.veto("--rerun", "--reason", "user checked AIdea's upload record",
                                           "--confirmed-upload-check", "AIdea record: #2 06-OFF, #3 06-ON (screens)")
                self.assertEqual(code, 0, out)
                self.assertTrue(rec["decided"])
                self.assertEqual(rec["confirmed_upload_check"], "AIdea record: #2 06-OFF, #3 06-ON (screens)")

    def test_equal_scores_identical_top1_no_warning(self):
        json_on = json.load(open(self.reports["on"], encoding="utf-8"))
        json_on["top1_differences_off_vs_on"] = {"public": 0, "private": 4}
        write_json(self.reports["on"], json_on)
        self.write_log(self.base(off=0.90, on=0.90))
        code, out, rec = self.veto()
        self.assertEqual(code, 0, out)
        self.assertEqual(rec["same_file_warning"], [])
        self.assertFalse(rec["vetoed"])

    def test_k7_passed(self):
        self.write_log(self.base(off=0.90, on=0.95))
        code, out, rec = self.veto(now="2026-11-23T00:00:01+08:00")
        self.assertEqual(code, 0, out)
        self.assertTrue(rec["k7_passed"])
        self.assertEqual(rec["final_variant"], "off")
        self.assertTrue(rec["reupload_off_needed"])

    def test_order_and_counts(self):
        cases = [
            ("no on", lambda e: e[:2], "nothing to veto"),
            ("on before off", lambda e: [e[0], dict(e[1], uploaded_at="2026-11-05T12:00:00+08:00"), e[2]],
             "before any 06-OFF"),
            ("two on", lambda e: e + [dict(e[2], id=4, uploaded_at="2026-11-06T10:00:00+08:00")], "allows one"),
            ("other run", lambda e: [e[0], e[1], dict(e[2], test06=dict(e[2]["test06"], run_id="r2"))],
             "different 06 runs"),
            ("no score", lambda e: [e[0], e[1], dict(e[2], public_score=None)], "public scores"),
            ("report changed", lambda e: [e[0], e[1], dict(e[2], test06=dict(e[2]["test06"], report_sha256="0" * 64))],
             "not the report"),
        ]
        for name, fn, msg in cases:
            with self.subTest(name):
                if os.path.exists(os.path.join(self.o6, "veto.json")):
                    os.remove(os.path.join(self.o6, "veto.json"))
                self.write_log(fn(self.base()))
                code, out, _rec = self.veto()
                self.assertEqual(code, 2, out)
                self.assertIn(msg, out)

    def test_two_runs_pair_the_same_run(self):
        """rev 3 item 22 (test02 re-run, 06 redone): 06-ON pairs with its own run's 06-OFF (review 10/3, veto_drill)."""
        e = self.base(off=0.90, on=0.91)
        old = self.entry(2, "off", "2026-11-05T10:00:00+08:00", 0.89, run_id="r0")
        new = self.entry(3, "off", "2026-11-06T10:00:00+08:00", 0.90, run_id="r1")
        on = dict(e[2], id=4, uploaded_at="2026-11-06T11:00:00+08:00")
        self.write_log([e[0], old, new, on])
        code, out, rec = self.veto()
        self.assertEqual(code, 0, out)
        self.assertEqual(rec["entries"], {"q04": 1, "off": 3, "on": 4})
        self.assertAlmostEqual(rec["d"], 0.91 - 0.90, places=12)
        self.assertEqual(rec["outcome"], "kept")

    def test_off_uploaded_again_before_on_is_paired(self):
        e = self.base(off=0.88, on=0.91, q04=0.88)                    # #2 was Q04's file
        again = self.entry(4, "off", "2026-11-05T10:30:00+08:00", 0.90)
        self.write_log([e[0], e[1], again, e[2]])
        code, out, rec = self.veto()
        self.assertEqual(code, 0, out)
        self.assertEqual(rec["entries"]["off"], 4)
        self.assertEqual(rec["same_file_warning"], [])
        self.assertAlmostEqual(rec["d"], 0.91 - 0.90, places=12)

    def test_wrong_upload_or_inconclusive_gives_off(self):
        for flag, outcome in (("--wrong-upload", "wrong_upload"),
                              ("--upload-check-inconclusive", "upload_check_inconclusive")):
            with self.subTest(flag):
                if os.path.exists(os.path.join(self.o6, "veto.json")):
                    os.remove(os.path.join(self.o6, "veto.json"))
                self.write_log(self.base(off=0.90, on=0.90))          # same-file warning: paused first
                code, out, rec = self.veto()
                self.assertEqual(code, 2, out)
                self.assertTrue(rec["paused"])
                self.assertTrue(any("--wrong-upload" in x for x in rec["check_items"]))
                code, out, rec = self.veto("--rerun", "--reason", "AIdea upload record checked", flag,
                                           "#3 is the 06-OFF file (AIdea record, 11/05 11:00)")
                self.assertEqual(code, 0, out)
                self.assertTrue(rec["decided"])
                self.assertFalse(rec["paused"])
                self.assertEqual(rec["outcome"], outcome)
                self.assertEqual(rec["final_variant"], "off")
                self.assertTrue(rec["reupload_off_needed"])
                self.assertIsNone(rec["d"])
                self.assertEqual(rec["upload_check"]["kind"], outcome)

    def test_one_check_option_only(self):
        self.write_log(self.base(off=0.90, on=0.90))
        code, out, rec = self.veto("--confirmed-upload-check", "x", "--wrong-upload", "y")
        self.assertEqual(code, 2, out)
        self.assertIn("at most one", out)
        self.assertIsNone(rec)

    def test_existing_veto_needs_rerun(self):
        self.write_log(self.base())
        self.assertEqual(self.veto()[0], 0)
        code, out, _ = self.veto()
        self.assertEqual(code, 2)
        self.assertIn("--rerun --reason", out)

    def test_status_k7_line(self):
        self.write_log(self.base())
        code, out = run_main(["status", "--dry-run", "--out-dir", self.o6, "--data-dir", self.p("a6"), "--log",
                              self.log, "--now", "2026-11-23T00:00:01+08:00"])
        self.assertEqual(code, 0)
        self.assertIn("K7' 已過、06-ON 已上傳、否決沒判定", out)

    def test_status_paused_line(self):
        self.write_log(self.base(off=0.90, on=0.90))
        self.assertEqual(self.veto()[0], 2)
        code, out = run_main(["status", "--dry-run", "--out-dir", self.o6, "--data-dir", self.p("a6"), "--log",
                              self.log, "--now", "2026-11-07T13:00:00+08:00"])
        self.assertEqual(code, 0)
        self.assertIn("核對前不重傳", out)


# ================================================================ real mode
class RealMode(Tmp):
    """Real-mode prerequisites (spec 7.2) with the path constants pointed at temporary trees and git mocked."""

    def setUp(self):
        super().setUp()
        self.d2 = self.p("results", "test02")
        self.o2 = os.path.join(self.d2, "player")
        self.c = os.path.join(self.d2, "checks")
        self.uncommitted = set()
        self.build_test02()
        self.patches = [mock.patch.object(t6, "REAL_O6", self.p("results", "test06", "player")),
                        mock.patch.object(t6, "REAL_A6", self.p("data", "test06", "player")),
                        mock.patch.object(t6, "REAL_D2", self.d2), mock.patch.object(t6, "REAL_O2", self.o2),
                        mock.patch.object(t6, "REAL_C", self.c), mock.patch.object(t6, "REAL_T", self.p("data", "test")),
                        mock.patch.object(t6, "REAL_LOG", self.p("results", "official_scores.json")),
                        mock.patch.object(t6, "committed_clean", side_effect=lambda p: not any(
                            t6.abs_inside(p, u) for u in self.uncommitted)),
                        mock.patch.object(t6, "git_clean_tracked", return_value=(True, "")),
                        mock.patch.object(t6, "now_real", return_value=dt.datetime(2026, 11, 5, 12, 0, tzinfo=TW))]
        for pt in self.patches:
            pt.start()

    def tearDown(self):
        for pt in reversed(self.patches):
            pt.stop()
        super().tearDown()

    def build_test02(self):
        sub = ti.write_csv(os.path.join(self.o2, "submission_player.csv"), ["question_id"] + [f"top_{i}" for i in
                                                                                            range(1, 6)], [])
        pools = {}
        for pool in ("public", "private"):
            top = ti.write_csv(os.path.join(self.o2, pool, "top5.csv"), ["question_id"] + [f"top_{i}" for i in
                                                                                        range(1, 6)], [])
            write_json(os.path.join(self.o2, pool, "apply_report.json"), {"mode": "real", "cnn": True,
                                                                          "output": {"sha256": sha(top)}})
            write_json(os.path.join(self.o2, pool, "pool_select.json"), {"mode": "real"})
            pools[pool] = {"top5": {"sha256": sha(top)}}
        write_json(os.path.join(self.o2, "submit_report.json"), {
            "mode": "real", "check_submission": {"official": True, "passed": True}, "output": {"sha256": sha(sub)},
            "pools": pools})
        write_json(os.path.join(self.o2, "seam_val", "seam_local.json"), {"passed": True, "mode": "real", "cnn": True})
        self.write_t1("PASS")

    def write_t1(self, verdict):
        write_json(os.path.join(self.c, "player", "t1.json"), {"verdict": verdict, "complete": True, "stage": "raw",
                                                               "provenance": {"mode": "real"}})

    def test_test02_not_committed(self):
        self.uncommitted.add(self.o2)
        code, out = run_main(["feats", "--pool", "public"])
        self.assertEqual(code, 2, out)
        self.assertIn("test02 competition 1 (reports and dumps): not committed", out)

    def test_q04_not_uploaded(self):
        code, out = run_main(["feats", "--pool", "public"])
        self.assertEqual(code, 2, out)
        self.assertIn("results\\official_scores.json missing" if os.sep == "\\" else "official_scores.json missing",
                      out)
        write_json(t6.REAL_LOG, {"format": "c10 official_scores v1", "uploads": [
            {"id": 1, "comp": "player", "status": "prepared", "file": t6.Q04_FILE_REL, "uploaded_at": None}]})
        code, out = run_main(["feats", "--pool", "public"])
        self.assertEqual(code, 2, out)
        self.assertIn("Q04's first upload", out)

    def test_git_not_clean(self):
        with mock.patch.object(t6, "git_clean_tracked", return_value=(False, " M tools/x.py")):
            code, out = run_main(["seam"])
        self.assertEqual(code, 2, out)
        self.assertIn("git is not clean", out)

    def test_venv_differs(self):
        with mock.patch.object(t6, "VENV", dict(t6.VENV, numpy="0.0")):
            code, out = run_main(["apply", "--pool", "private"])
        self.assertEqual(code, 2, out)
        self.assertIn(".venv versions", out)

    def test_k6p_passed_is_3(self):
        with mock.patch.object(t6, "now_real", return_value=dt.datetime(2026, 11, 22, 0, 0, tzinfo=TW)):
            code, out = run_main(["feats", "--pool", "public"])
        self.assertEqual(code, 3, out)
        self.assertIn("K6' 已過", out)
        nu = json.load(open(os.path.join(t6.REAL_O6, "not_used.json"), encoding="utf-8"))
        self.assertEqual([e["reason"] for e in nu["events"]], ["k6p"])

    def test_k6p_passed_after_a_committed_run_is_2(self):
        """K6' passed but the 06 run was committed by K6': refused (2), no not_used.json (review 10/3)."""
        write_json(os.path.join(t6.REAL_O6, "off", t6.REPORT), {"format": t6.FORMATS["submit"]})
        late = dt.datetime(2026, 11, 22, 0, 0, tzinfo=TW)
        with mock.patch.object(t6, "now_real", return_value=late), \
                mock.patch.object(t6, "git_commit_time", return_value="2026-11-10T10:00:00+08:00"):
            code, out = run_main(["apply", "--pool", "public"])
        self.assertEqual(code, 2, out)
        self.assertIn("K6' 前已 commit 的 06 照常有效", out)
        self.assertFalse(os.path.exists(os.path.join(t6.REAL_O6, "not_used.json")))
        with mock.patch.object(t6, "now_real", return_value=late), \
                mock.patch.object(t6, "git_commit_time", return_value="2026-11-22T00:00:00+08:00"):
            code, out = run_main(["apply", "--pool", "public"])
        self.assertEqual(code, 3, out)                                  # committed after K6': no 06 this season

    def test_fail_opening_is_3(self):
        self.write_t1("FAIL_OPENING")
        code, out = run_main(["feats", "--pool", "private"])
        self.assertEqual(code, 3, out)
        self.assertIn("FAIL_OPENING", out)

    def test_registered_3_before_outputs_exist(self):
        """feats already ran: FAIL_OPENING is still 3, never hidden behind 'outputs exist' (review 10/3)."""
        write_json(os.path.join(t6.REAL_O6, "private", "feats_report.json"), {"format": t6.FORMATS["feats"]})
        os.makedirs(os.path.join(t6.REAL_A6, "private"), exist_ok=True)
        with open(os.path.join(t6.REAL_A6, "private", "feats.npz"), "wb") as f:
            f.write(b"x")
        self.write_t1("FAIL_OPENING")
        code, out = run_main(["feats", "--pool", "private"])
        self.assertEqual(code, 3, out)
        self.assertNotIn("outputs exist", out)

    def test_fail_struct_is_2(self):
        self.write_t1("FAIL_STRUCT")
        code, out = run_main(["feats", "--pool", "private"])
        self.assertEqual(code, 2, out)

    def test_t1_must_be_full_raw(self):
        write_json(os.path.join(self.c, "player", "t1.json"), {"verdict": "PASS", "complete": False,
                                                               "stage": "raw", "provenance": {"mode": "real"}})
        code, out = run_main(["feats", "--pool", "private"])
        self.assertEqual(code, 2, out)
        self.assertIn("full T1", out)

    def write_uploaded_run(self, run_id=None):
        """A complete 06 run on disk (seam + both pool reports bound to the present test02) and an upload log with
        Q04's first upload and a prepared 06-OFF of that run."""
        o6 = t6.REAL_O6
        seam = write_json(os.path.join(o6, "seam_val", "t06_seam.json"), {"inputs": {
            "test02_seam_local_sha256_lf": t6.sha_lf(os.path.join(self.o2, "seam_val", "seam_local.json"))}})
        pools = []
        for pool in t6.TEST_POOLS:
            pools.append(write_json(os.path.join(o6, pool, "t06_pool.json"), {"test02": {
                "apply_report_sha256_lf": t6.sha_lf(os.path.join(self.o2, pool, "apply_report.json")),
                "top5_sha256": sha(os.path.join(self.o2, pool, "top5.csv"))}}))
        rid = run_id or hashlib.sha256("|".join([t6.sha_lf(seam)] + [t6.sha_lf(p) for p in pools]).encode()).hexdigest()
        write_json(t6.REAL_LOG, {"format": "c10 official_scores v1", "uploads": [
            {"id": 1, "comp": "player", "status": "uploaded", "file": t6.Q04_FILE_REL,
             "uploaded_at": "2026-11-04T16:05:00+08:00"},
            {"id": 2, "comp": "player", "status": "prepared", "file": "results/test06/player/off/submission_player.csv",
             "uploaded_at": None, "test06": {"variant": "off", "run_id": rid}}]})

    def test_rerun_refused_once_uploaded(self):
        """feats / seam / apply --rerun once this 06 run is in the upload log and test02 is unchanged: 2, as submit
        --rerun (round-2 review 10/3, minor 7)."""
        self.write_uploaded_run()
        for argv in (["feats", "--pool", "public"], ["feats", "--pool", "seam_val"], ["seam"],
                     ["apply", "--pool", "private"]):
            with self.subTest(argv[0]):
                code, out = run_main(argv + ["--rerun", "--reason", "try again"])
                self.assertEqual(code, 2, out)
                self.assertIn("test02 has not changed since", out)
                self.assertIn("entries [2]", out)

    def test_rerun_allowed_after_test02_rerun_or_other_run(self):
        """The guard does not block rev 3 item 26 (test02 re-run: 06 redone from feats) nor another run's log entry;
        those commands then go on to their own checks (here: the test files are missing)."""
        self.write_uploaded_run(run_id="another run")
        code, out = run_main(["feats", "--pool", "public", "--rerun", "--reason", "x"])
        self.assertEqual(code, 2, out)
        self.assertNotIn("test02 has not changed since", out)
        self.write_uploaded_run()
        write_json(os.path.join(self.o2, "seam_val", "seam_local.json"), {"passed": True, "mode": "real", "cnn": True,
                                                                           "rerun": "test02 re-run"})
        code, out = run_main(["feats", "--pool", "public", "--rerun", "--reason", "test02 rerun x"])
        self.assertEqual(code, 2, out)
        self.assertNotIn("test02 has not changed since", out)
        self.assertIn("input file missing", out)

    def test_h6p_not_adopted_is_3(self):
        with mock.patch.object(t6, "H6P_RESULTS_MARK", "**not this text**"):
            code, out = run_main(["feats", "--pool", "public"])
        self.assertEqual(code, 3, out)
        self.assertIn("H6P 沒有採用", out)


# ================================================================ fake-t2
class FakeT2(Tmp):
    def test_counts(self):
        root = self.p("S")
        tests = ti.write_official(os.path.join(root, "data", "test"))
        a = mock.Mock(root=root, fail=None, rerun=False, reason=None)
        with contextlib.redirect_stdout(io.StringIO()):
            rec = f6.cmd_fake_t2(a)
        self.assertTrue(rec["fake"])
        self.assertEqual(rec["format"], t6.T2_FORMAT)
        units = {(u["unit"], u.get("file") or u.get("pool") or ""): u for u in rec["units"]}
        self.assertEqual(sorted({u for u, _ in units}), ["a", "b", "c", "d"])
        # ti.write_official uses one SGF everywhere: every question game is in its pool's candidates (b), shared by all
        # questions (c), and every private game is among the public ones (d)
        self.assertEqual(units[("b", "public")]["verdict"], "FAIL")
        self.assertEqual(units[("c", "player_identification_test_private.csv")]["verdict"], "FAIL")
        self.assertEqual(units[("d", "")]["copies"], units[("d", "")]["games_in_file"])
        s = t6.t2_summary(rec)
        self.assertIn("b:public", s["abd_failed"])
        with self.assertRaises(SystemExit):
            f6.cmd_fake_t2(a)                                      # exists: --rerun --reason
        a2 = mock.Mock(root=root, fail=["a"], rerun=True, reason="drill")
        with contextlib.redirect_stdout(io.StringIO()):
            rec = f6.cmd_fake_t2(a2)
        self.assertTrue(all(u["verdict"] == "FAIL" for u in rec["units"] if u["unit"] == "a"))
        self.assertIn("forced_fail", [k for u in rec["units"] if u["unit"] == "a" for k in u])

    def test_root_inside_repo_refused(self):
        with self.assertRaises(SystemExit):
            f6.check_root(os.path.join(REPO, "results", "x"))


if __name__ == "__main__":
    unittest.main()
