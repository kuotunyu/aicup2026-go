"""Tests for the PREREG_25 rev 3 additions to tools/official_log.py (PREREG_25 4.2, 4.3).

* TagVersusNew*: the version of tag p19-testday-pre25 (everything up to 19, no 25) and the working-tree version run the
  same flows with the same mocks (the R20 / R10 / 04 / 06 flows, the 07, 08, 09 and 14 flows of
  test_t19_official_log), now with a results/test25 tree on disk; the log bytes, returned records, stdout and refusals
  are identical (a log without 25 entries).
* T25*: the 25 path (prepare and its order rules, the refusals recomputed from the report's numbers, --method, the
  adoption record, final under PREREG_25 4.3 with --not-25-reason, the 19 / 09 / 08 / 07 / 04 rules skipping 25,
  status, the same-file warning).
Synthetic files in a temporary directory only; git and the clock are mocked. Nothing here opens a label file, a path
under data/local or runs tools/score.py. Stdlib only besides the module under test.
Run: python -m pytest tests/test_t25_official_log.py
"""

import copy
import json
import os
import shutil
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import official_log as ol  # noqa: E402
import test_t02_infra as ti  # noqa: E402
import test_t04_official_log as t4o  # noqa: E402
import test_t07_official_log as t7o  # noqa: E402
import test_t09_official_log as t9o  # noqa: E402
import test_t19_official_log as t19o  # noqa: E402

TAG = "p19-testday-pre25"                          # main before the 25 kit
ADOPTION_FILE = os.path.join(REPO, "results", "prereg25", "rank", "adoption.json")
P1_TIME = t9o.P1_TIME
PREREG25_HEAD = ("# PREREG_25\n\n| rev 1 | `535a5ae` | first |\n| rev 2 | `ff41c37` | freeze |\n"
                 "| rev 3 | (commit) | test day |\n")
DIFF25 = {"19": {"public": 42, "private": 47, "all": 89}, "09": {"public": 70, "private": 80, "all": 150},
          "08": {"public": 90, "private": 95, "all": 185}, "07": {"public": 140, "private": 130, "all": 270},
          "04": {"public": 155, "private": 150, "all": 305}, "R20": {"public": 170, "private": 165, "all": 335}}
sha, write_json = t4o.sha, t4o.write_json


def write_t25_tree(repo, rank_rows, report_over=None):
    """results/test25 as tools/t25_rank.py apply leaves it (real mode) plus the committed inputs it binds; the 19
    report it binds is the one in results/test19/rank (write the 19 tree first). -> 25 file."""
    r = lambda rel: os.path.join(repo, *rel.split("/"))   # noqa: E731
    os.makedirs(os.path.dirname(r(ol.T25_ADOPTION)), exist_ok=True)
    shutil.copyfile(ADOPTION_FILE, r(ol.T25_ADOPTION))
    for rel in ol.T25_INPUTS_REQUIRED:
        if not os.path.exists(r(rel)):
            if rel.endswith(".json"):
                write_json(r(rel), {"passed": True, "mode": "real", "fake_input": rel})
            elif rel.endswith(".csv"):
                ti.write_csv(r(rel), ["question_id", "pred_rank"], [[q, "D"] for q, _ in rank_rows])
            else:
                os.makedirs(os.path.dirname(r(rel)), exist_ok=True)
                with open(r(rel), "wb") as f:
                    f.write(b"PK-fake-npz " + rel.encode() + b"\r\n")
    rows = [[q, "3D" if i % 7 == 0 else lab] for i, (q, lab) in enumerate(rank_rows)]
    f25 = ti.write_csv(r("results/test25/rank/submission_rank.csv"), ["question_id", "pred_rank"], rows)
    rep = report25(repo, f25)
    rep.update(report_over or {})
    write_json(r(f"results/test25/rank/{ol.T25_REPORT}"), rep)
    return f25


def report25(repo, f25):
    bound = {rel: ol.sha256_bound(os.path.join(repo, *rel.split("/"))) for rel in ol.T25_INPUTS_REQUIRED}
    rep = lambda ref: {"equal": True, "byte_equal": True, "reference_sha256_lf": ref, "computed_sha256": ref,  # noqa
                       "max_abs_diff": 0.0, "rule_a_differences": 0, "rule_b_differences": 0, "tol": 1e-12}
    q9 = dict(rep(bound[ol.T25_09_PROBS]), p1_ok_equal_t09=True, usable_equal_t09=True, lp_p1e_max_abs_diff_vs_t09=0.0,
              lp_p1e_equal=True)
    q19 = dict(rep(bound[ol.T25_19_PROBS]), lp_set19={"question_id_equal": True, "has_equal": True,
                                                       "usable_equal": True, "lp_max_abs_diff": 0.0, "equal": True})
    return {
        "format": ol.T25_FORMAT, "mode": "real", "created": "2026-11-10T12:00:00+08:00", "exit_code": 0,
        "usable": True, "unusable_reasons": [],
        "combo": {"id": "Q25", "sources": list(ol.T25_COMBO_SOURCES),
                  "terms": ["OP", "O4", "SWA", "SWB", "HS", "P1TO", "SET"]},
        "p1_sets": {s: {"fake_from_val": False, "earliest_chunk_created_utc": P1_TIME} for s in ol.T25_P1_SETS},
        "earliest_p1_chunk_created_utc": P1_TIME, "q09_vs_09": q9, "q19_vs_19": q19,
        "set_seam_record_only": {"max_abs_lp_set": 0.0084, "usable_games": 400, "bitwise_equal_means": False},
        "set_direction": {s: {"rho": 0.96, "threshold": 0.47880806681040905, "passed": True} for s in ol.T25_SETS},
        "rule_b_differences": copy.deepcopy(DIFF25),
        "check_submission": {"mode": "official", "questions": [], "errors": []},
        "outputs_sha256": {"submission_rank.csv": sha(f25)}, "inputs_bound": bound}


# ---------------------------------------------------------------- the tag version against the new one
class TagVersusNew(t19o.TagVersusNew):
    """A log without 25 entries gives the tag p19-testday-pre25 version's log bytes, records, stdout and refusals, with
    a results/test25 tree on disk (the R20 / R10 / 04 / 06 flows)."""

    @classmethod
    def setUpClass(cls):
        cls.tag = t7o.load_version(TAG)
        assert hasattr(cls.tag, "check_reports_t19") and not hasattr(cls.tag, "check_reports_t25"), "not the tag"

    def build(self, kind, with_t06):
        out = super().build(kind, with_t06)
        write_t25_tree(out[0], [[q, "D"] for q in t9o.t6o.IDS])
        return out


class _Flow25:
    def setUp(self):
        self.tag = t7o.load_version(TAG)

    def run_with(self, mod):
        real = t9o.write_t09_tree

        def both(repo, rows, *a, **k):
            out = real(repo, rows, *a, **k)
            t19o.write_t19_tree(repo, rows)
            write_t25_tree(repo, rows)
            return out
        with mock.patch.object(t9o, "write_t09_tree", side_effect=both):
            base = t19o._Flow19 if isinstance(self, t19o._Flow19) else _Flow25   # skip 19's tree writer
            return super(base, self).run_with(mod)

    def check_identical(self):
        old = super().check_identical()
        log = json.loads(old[-1][3])
        self.assertFalse(any("test25" in e or "prereg25_4_3" in e for e in log["uploads"]))
        self.assertNotIn("prereg25_4_3", json.dumps(log["final_choice"]))
        return old


class TagVersusNew07Flow(_Flow25, t19o.TagVersusNew07Flow):
    pass


class TagVersusNew08Flow(_Flow25, t19o.TagVersusNew08Flow):
    pass


class TagVersusNew09Flow(_Flow25, t19o.TagVersusNew09Flow):
    pass


class TagVersusNew14Flow(_Flow25, t19o.TagVersusNew14Flow):
    pass


class TagVersusNew19Flow(_Flow25, t9o.FlowBase):
    """R20 -> 04 -> 07 -> 08 -> 09 -> 19 uploaded, final 19, a refused final 09, public, status: the 19 rules behave as
    before when 25 is not in the log."""

    def steps(self, step):
        mod = t9o.t8o.ol
        prev = mod.git_head_text
        head = mock.patch.object(mod, "git_head_text", side_effect=lambda rel: t19o.PREREG19_HEAD
                                 if rel == mod.P19_DOC else prev(rel))
        head.start()
        try:
            step("prepare R20", self.prepare)
            step("uploaded R20", self.run_ol, "uploaded", "--id", "1", "--ok-at", "2026-11-05T14:10",
                 "--uploaded-at", "2026-11-05T14:15")
            step("prepare 04", self.prepare04)
            step("uploaded 04", self.uploaded, 2, "2026-11-05T15:10")
            step("prepare 07", self.prepare07)
            step("uploaded 07", self.uploaded, 3, "2026-11-06T13:00")
            step("prepare 08", self.prepare08)
            step("uploaded 08", self.uploaded, 4, "2026-11-07T13:00")
            step("prepare 09", lambda: self.prepare("rank", self.r("results/test09/rank/submission_rank.csv"),
                                                    "prereg09 Q09 (PREREG_09 5.3)"))
            step("uploaded 09", self.uploaded, 5, "2026-11-08T13:00")
            step("prepare 19", lambda: self.prepare("rank", self.r("results/test19/rank/submission_rank.csv"),
                                                    "prereg19 Q19 (PREREG_19 4.3)"))
            step("uploaded 19", self.uploaded, 6, "2026-11-09T13:00")
            step("public 19", self.run_ol, "public", "--id", "6", "--score", "0.71")
            step("final 09 refused", self.run_ol, "final", "--comp", "rank", "--id", "5", "--basis", "x")
            step("final 19", self.run_ol, "final", "--comp", "rank", "--id", "6", "--basis", "PREREG_19 4.3")
            step("status", self.run_ol, "status", "--now", "2026-11-12T00:00")
        finally:
            head.stop()

    def check_identical(self):
        old, new = self.run_with(self.tag), self.run_with(ol)
        self.assertEqual([s[0] for s in old], [s[0] for s in new])
        for (name, o_out, o_res, o_log), (_n, n_out, n_res, n_log) in zip(old, new):
            with self.subTest(step=name):
                self.assertEqual(n_log, o_log)
                self.assertEqual(n_out, o_out)
                self.assertEqual(n_res, o_res)
        return old

    def test_19_flow_identical(self):
        old = self.check_identical()
        log = json.loads(old[-1][3])
        self.assertTrue(any("test19" in e for e in log["uploads"]), "the flow must hold a 19 entry")
        self.assertFalse(any("test25" in e for e in log["uploads"]))
        self.assertEqual(json.loads(old[14][2])["prereg19_4_3"]["chosen"], "19")
        self.assertIn("refused", old[13][2])


# ---------------------------------------------------------------- the 25 path
class T25Base(t19o.T19Base):
    def setUp(self):
        super().setUp()
        self.p25_head = PREREG25_HEAD
        prev = ol.git_head_text                       # the 04-19 mock
        self.patches25 = [mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p25_head
                                            if rel == ol.P25_DOC else prev(rel))]
        for pt in self.patches25:
            pt.start()
        self.f25 = write_t25_tree(self.repo, self.rank_rows)

    def tearDown(self):
        for pt in reversed(self.patches25):
            pt.stop()
        super().tearDown()

    def edit25(self, fn):
        p = self.r(f"results/test25/rank/{ol.T25_REPORT}")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def prepare25(self, *extra):
        return self.prepare("rank", self.f25, "prereg25 Q25 (PREREG_25 4.3)", *extra)

    def ready25(self):
        """R20 #1, 04 #2, 07 #3, 08 #4, 09 #5 and 19 #6 uploaded."""
        self.ready19()
        e19, _ = self.up_19()
        return e19

    def up_25(self, at="2026-11-10T13:00"):
        e25, out = self.prepare25()
        self.uploaded(e25["id"], at)
        return e25, out

    def rev4_25(self, at="2026-11-11T10:00:00+08:00"):
        self.p25_head = PREREG25_HEAD + "| rev 4 | `abc` | bug |\n"
        self.rev_added["| rev 4 |"] = ("1" * 40, at)


class T25Prepare(T25Base):
    def test_order_r20_to_19_then_25(self):
        self.assertIn("R20's first upload", self.refused(self.prepare25))
        self.ready19()                                     # R20 ... 09 uploaded; 19 not
        self.assertIn("not logged as uploaded", self.refused(self.prepare25))
        e19, _ = self.prepare19()
        self.assertIn("not logged as uploaded", self.refused(self.prepare25))   # 19 prepared, not uploaded
        self.uploaded(e19["id"], "2026-11-09T13:00")
        e25, out = self.prepare25()
        self.assertEqual((e25["id"], e25["file"]), (7, "results/test25/rank/submission_rank.csv"))
        self.assertEqual(e25["method"], ol.T25_METHOD)
        t25 = e25["test25"]
        self.assertEqual((t25["r20_first_upload_id"], t25["t19_entry_id"], t25["t09_entry_id"], t25["combo"]),
                         (1, 6, 5, "Q25"))
        self.assertEqual(t25["public_rule_b_differences"], {k: v["public"] for k, v in DIFF25.items()})
        self.assertEqual(t25["private_rule_b_differences"], {k: v["private"] for k, v in DIFF25.items()})
        self.assertEqual(t25["k6_25"]["deadline"], "2026-11-18T23:59:59+08:00")
        for k in ("test19", "test09", "prereg04_5_3", "prereg07_5_4", "prereg08_5_4", "prereg09_5_3", "prereg19_4_3",
                  "prereg25_4_3"):
            self.assertNotIn(k, e25)
        for s in ("這一份是 25", "Q25", "#6", "K6-25", "驗證玩家"):
            self.assertIn(s, out)

    def test_conditions_recomputed(self):
        self.ready25()
        cases = [
            ("q09 sha", lambda r: r["q09_vs_09"].update(reference_sha256_lf="0" * 64), "Q09 recomputed"),
            ("q09 diff", lambda r: r["q09_vs_09"].update(max_abs_diff=2e-12), "Q09 recomputed"),
            ("q09 lp", lambda r: r["q09_vs_09"].update(lp_p1e_max_abs_diff_vs_t09=2e-12), "lp_P1TO_E"),
            ("q09 usable", lambda r: r["q09_vs_09"].update(usable_equal_t09=False), "lp_P1TO_E"),
            ("q19 sha", lambda r: r["q19_vs_19"].update(reference_sha256_lf="0" * 64), "Q19 recomputed"),
            ("q19 rank", lambda r: r["q19_vs_19"].update(rule_b_differences=1), "Q19 recomputed"),
            ("q19 equal", lambda r: r["q19_vs_19"].update(equal=False), "Q19 recomputed"),
            ("lp19", lambda r: r["q19_vs_19"]["lp_set19"].update(lp_max_abs_diff=2e-12), "lp_SET19"),
            ("lp19 has", lambda r: r["q19_vs_19"]["lp_set19"].update(has_equal=False), "lp_SET19"),
            ("rho", lambda r: r["set_direction"]["private"].update(rho=0.478), "SET25 direction"),
            ("fake", lambda r: r["p1_sets"]["public"].update(fake_from_val=True), "drill records"),
            ("mode", lambda r: r.update(mode="dry-run"), "real-mode"),
            ("exit", lambda r: r.update(exit_code=3), "exit_code"),
            ("combo", lambda r: r["combo"].update(id="Q19"), "not Q25"),
            ("check_submission", lambda r: r["check_submission"].update(errors=["x"]), "check_submission")]
        for name, fn, msg in cases:
            with self.subTest(name):
                self.f25 = write_t25_tree(self.repo, self.rank_rows)
                self.edit25(fn)
                self.assertIn(msg, self.refused(self.prepare25))
        self.f25 = write_t25_tree(self.repo, self.rank_rows)
        self.edit25(lambda r: r["set_direction"]["private"].update(rho=0.4789))
        e, _ = self.prepare25()                            # rho at the threshold passes
        self.assertAlmostEqual(e["test25"]["rho_threshold"], 0.47880806681040905, places=15)

    def test_method_and_reasons(self):
        self.ready25()
        self.assertIn("does not name Q25", self.refused(self.prepare25, "--method", "prereg19 Q19"))
        for flag, rev in (("--not-04-reason", "PREREG_04 rev 4: x"), ("--not-07-reason", "PREREG_07 rev 5: x"),
                          ("--not-08-reason", "PREREG_08 rev 4: x"), ("--not-09-reason", "PREREG_09 rev 5: x"),
                          ("--not-19-reason", "PREREG_19 rev 4: x")):
            with self.subTest(flag):
                self.assertIn("not apply to a 25 file", self.refused(self.prepare25, flag, rev))
        self.assertIn("19 / 09 / 08 / 07 / 04 / R20 / R10 file only", self.refused(self.prepare25, "--not-25-reason",
                                                                                   "PREREG_25 rev 4: x"))
        e, _ = self.prepare25("--method", "Q25 rule B")
        self.assertEqual(e["method"], "Q25 rule B")

    def test_adoption_record(self):
        self.ready25()
        self.assertEqual(ol.t25_adoption_problems(), [])
        p = self.r(ol.T25_ADOPTION)
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        write_json(p, dict(s, usable=False))
        self.assertIn("is not the committed adoption record", self.refused(self.prepare25))
        os.remove(p)
        self.assertIn("25 was not adopted", self.refused(self.prepare25))

    def test_bound_input_changed_and_k6(self):
        self.ready25()
        with open(self.r("results/prereg25/rank/s25_seed1.npz"), "ab") as f:
            f.write(b"x")
        self.assertIn("results/prereg25/rank/s25_seed1.npz is", self.refused(self.prepare25))
        self.f25 = write_t25_tree(self.repo, self.rank_rows)
        self.commit = ("f" * 40, "2026-11-19T00:00:00+08:00")
        self.assertIn("not by K6-25", self.refused(self.prepare25))

    def test_bound_19_not_eligible(self):
        self.ready25()
        self.edit19(lambda r: r["set_direction"]["public"].update(rho=0.1))
        self.assertIn("results/test19/rank/t19_apply_report.json is", self.refused(self.prepare25))
        self.f25 = write_t25_tree(self.repo, self.rank_rows)   # a 25 run bound to the changed 19 report
        self.assertIn("no longer meets PREREG_19 4.3", self.refused(self.prepare25))

    def test_p1_before_r20_upload_refused(self):
        self.ready25()
        self.edit25(lambda r: r.update(earliest_p1_chunk_created_utc="2026-11-05T06:00:00+00:00"))
        self.assertIn("not after R20's first upload", self.refused(self.prepare25))


class T25Final(T25Base):
    def test_final_25_and_status(self):
        e19 = self.ready25()
        e25, _ = self.up_25()
        self.assertIn("needs --not-25-reason", self.refused(self.run_ol, "final", "--comp", "rank", "--id",
                                                            str(e19["id"]), "--basis", "x"))
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e25["id"]), "--basis", "PREREG_25 4.3")
        self.assertEqual(fc["prereg25_4_3"]["chosen"], "25")
        for k in ("prereg04_5_3", "prereg07_5_4", "prereg08_5_4", "prereg09_5_3", "prereg19_4_3"):
            self.assertNotIn(k, fc)
        self.assertIn("PREREG_25 4.3：最終選擇是 25", out)
        st, out = self.run_ol("status", "--now", "2026-11-12T00:00")
        for k in ("prereg04", "prereg07", "prereg08", "prereg09", "prereg19", "prereg25"):
            self.assertFalse(st["rank"][k]["violation"], k)
        self.assertEqual(st["rank"]["prereg04"]["t25_eligible"], [e25["id"]])
        self.assertNotIn("違反", out)
        self.assertIn("K6-25", out)

    def test_final_19_needs_not_25_reason(self):
        e19 = self.ready25()
        self.up_25()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e19["id"]), "--basis", "x",
                           "--not-25-reason", "PREREG_25 rev 3: bug")
        self.assertIn("rev 4 or later", msg)
        self.assertIn("has no row", self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e19["id"]),
                                                 "--basis", "x", "--not-25-reason", "PREREG_25 rev 4: bug"))
        self.rev4_25(at="2026-11-04T10:00:00+08:00")
        self.assertIn("not after the records", self.refused(self.run_ol, "final", "--comp", "rank", "--id",
                                                            str(e19["id"]), "--basis", "x", "--not-25-reason",
                                                            "PREREG_25 rev 4: bug"))
        self.rev4_25()
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", str(e19["id"]), "--basis", "x", "--not-25-reason",
                            "PREREG_25 rev 4: bug in t25_rank apply")
        self.assertEqual((fc["prereg25_4_3"]["chosen"], fc["prereg25_4_3"]["revision"]["rev"]), ("19", 4))
        self.assertEqual(fc["prereg19_4_3"]["chosen"], "19")
        st, _ = self.run_ol("status", "--now", "2026-11-12T00:00")
        self.assertFalse(st["rank"]["prereg25"]["violation"])
        self.assertFalse(st["rank"]["prereg19"]["violation"])

    def test_25_no_longer_eligible(self):
        e19 = self.ready25()
        e25, _ = self.up_25()
        with open(self.r(ol.T25_SET_SEAM), "ab") as f:
            f.write(b" ")
        self.assertIn("does not meet PREREG_25 4.3 now", self.refused(self.run_ol, "final", "--comp", "rank",
                                                                      "--id", str(e25["id"]), "--basis", "x"))
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", str(e19["id"]), "--basis", "PREREG_19 4.3")
        self.assertEqual((fc["prereg25_4_3"]["eligible_25_entries"], fc["prereg25_4_3"]["chosen"]), ([], "19"))
        st, out = self.run_ol("status", "--now", "2026-11-12T00:00")
        self.assertTrue(st["rank"]["prereg25"]["redo_19"])
        self.assertIn("現在沒有符合的 25", out)

    def test_status_hint_and_violation(self):
        e19 = self.ready25()
        self.run_ol("final", "--comp", "rank", "--id", str(e19["id"]), "--basis", "PREREG_19 4.3")
        self.up_25()
        st, out = self.run_ol("status", "--now", "2026-11-12T00:00")
        self.assertTrue(st["rank"]["prereg25"]["violation"])
        self.assertIn("違反 PREREG_25 4.3", out)
        log = self.log_obj()
        log["final_choice"]["rank"] = None
        write_json(self.log, log)
        st, out = self.run_ol("status", "--now", "2026-11-12T00:00")
        self.assertIn("最終選擇要設成 25", out)
        for k in ("19", "09", "08", "07", "04"):
            self.assertNotIn(f"最終選擇要設成 {k}", out)

    def test_prepare_19_after_25(self):
        self.ready25()
        self.up_25()
        self.f19 = t19o.write_t19_tree(self.repo, self.rank_rows)     # the same 19 run, prepared again
        self.f25 = write_t25_tree(self.repo, self.rank_rows)
        self.assertIn("needs --not-25-reason", self.refused(self.prepare19))
        self.rev4_25()
        e, out = self.prepare19("--not-25-reason", "PREREG_25 rev 4: bug")
        self.assertEqual(e["prereg25_4_3"]["revision"]["rev"], 4)
        self.assertIn("這一份不是 25", out)


class T25Public(T25Base):
    def test_same_score_warning(self):
        e19 = self.ready25()
        self.run_ol("public", "--id", str(e19["id"]), "--score", "0.71")
        e25, _ = self.up_25()
        _e, out = self.run_ol("public", "--id", str(e25["id"]), "--score", "0.71")
        self.assertIn("WARNING (PREREG_25 4.3)", out)
        self.assertIn("（19）", out)
        for w in ("WARNING (PREREG_19 4.3)", "WARNING (PREREG_09 5.3)", "WARNING (PREREG_04 rev 3)"):
            self.assertNotIn(w, out)


if __name__ == "__main__":
    unittest.main()
