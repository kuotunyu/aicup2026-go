"""Tests for the PREREG_19 rev 3 additions to tools/official_log.py (PREREG_19 4.2, 4.3).

* TagVersusNew*: the version of tag p14-testday-pre19 (R20, 04, 06, 07, 08, 09, 10 and 14 additions, no 19) and the
  working-tree version run the same flows with the same mocks (the R20 / R10 / 04 / 06 flows, the 07, 08 and 09 flows
  of test_t14_official_log, and a 14 flow on a 09 base: final 09, 14 prepared, uploaded, final 14, status), now with a
  results/test19 tree on disk; the log bytes, returned records, stdout and refusals are identical (a log without 19
  entries).
* T19*: the 19 path (prepare and its order rules, the refusals recomputed from the report's numbers, --method, the
  adoption record, final under PREREG_19 4.3 with --not-19-reason, the 09 / 08 / 07 / 04 rules skipping 19, status, the
  same-file warning) and 14 on a 19 base.
Synthetic files in a temporary directory only; git and the clock are mocked. Nothing here opens a label file or runs
tools/score.py. Stdlib only besides the module under test.
Run: python -m pytest tests/test_t19_official_log.py
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
import test_t14_official_log as t14o  # noqa: E402

TAG = "p14-testday-pre19"                          # main before the 19 kit
ADOPTION_FILE = os.path.join(REPO, "results", "prereg19", "rank", "apply_report.json")
P1_TIME = t9o.P1_TIME
PREREG19_HEAD = ("# PREREG_19\n\n| rev 1 | `faac6e8` | first |\n| rev 2 | `768f7d2` | freeze |\n"
                 "| rev 3 | (commit) | test day |\n")
DIFF19 = {"09": {"public": 61, "all": 120}, "08": {"public": 90, "all": 180}, "07": {"public": 140, "all": 270},
          "04": {"public": 155, "all": 310}, "R20": {"public": 170, "all": 335}}
sha, write_json = t4o.sha, t4o.write_json


def write_t19_tree(repo, rank_rows, report_over=None):
    """results/test19 as tools/t19_rank.py apply leaves it (real mode) plus the committed inputs it binds; the 09
    report it binds is the one in results/test09/rank. -> 19 file."""
    r = lambda rel: os.path.join(repo, *rel.split("/"))   # noqa: E731
    os.makedirs(os.path.dirname(r(ol.T19_ADOPTION)), exist_ok=True)
    shutil.copyfile(ADOPTION_FILE, r(ol.T19_ADOPTION))
    for rel in ol.T19_INPUTS_REQUIRED:
        if not os.path.exists(r(rel)):
            if rel.endswith(".json"):
                write_json(r(rel), {"passed": True, "mode": "real", "fake_input": rel})
            elif rel.endswith(".csv"):
                ti.write_csv(r(rel), ["question_id", "pred_rank"], [[q, "D"] for q, _ in rank_rows])
            else:
                os.makedirs(os.path.dirname(r(rel)), exist_ok=True)
                with open(r(rel), "wb") as f:
                    f.write(b"PK-fake-npz " + rel.encode() + b"\r\n")
    rows = [[q, "2D" if i % 5 == 0 else lab] for i, (q, lab) in enumerate(rank_rows)]
    f19 = ti.write_csv(r("results/test19/rank/submission_rank.csv"), ["question_id", "pred_rank"], rows)
    rep = report19(repo, f19)
    rep.update(report_over or {})
    write_json(r(f"results/test19/rank/{ol.T19_REPORT}"), rep)
    return f19


def report19(repo, f19):
    bound = {rel: ol.sha256_bound(os.path.join(repo, *rel.split("/"))) for rel in ol.T19_INPUTS_REQUIRED}
    return {
        "format": ol.T19_FORMAT, "mode": "real", "created": "2026-11-09T12:00:00+08:00", "exit_code": 0,
        "usable": True, "unusable_reasons": [],
        "combo": {"id": "Q19", "sources": list(ol.T19_COMBO_SOURCES),
                  "terms": ["OP", "O4", "SWA", "SWB", "HS", "P1TO", "SET"]},
        "p1_sets": {s: {"fake_from_val": False, "earliest_chunk_created_utc": P1_TIME} for s in ol.T19_P1_SETS},
        "earliest_p1_chunk_created_utc": P1_TIME,
        "q09_vs_09": {"equal": True, "byte_equal": True, "t09_sha256": bound[ol.T19_09_PROBS],
                      "computed_sha256": bound[ol.T19_09_PROBS], "max_abs_diff": 0.0, "rule_a_differences": 0,
                      "rule_b_differences": 0, "p1_ok_equal_t09": True, "usable_equal_t09": True,
                      "lp_p1e_max_abs_diff_vs_t09": 0.0, "lp_p1e_equal": True},
        "set_seam_record_only": {"max_abs_lp_set": 0.0, "usable_games": 400, "bitwise_equal_means": True},
        "seam_local": {"verdict": "passed", "passed": True, "max_abs_diff": 0.0, "tol": 1e-4, "rule_b_differences": 0,
                       "byte_identical_probs": True, "byte_identical_pred_B": True, "required_byte_identical": True,
                       "usable_equal_to_prereg19": True},
        "set_direction": {s: {"rho": 0.96, "threshold": 0.4792752582816624, "passed": True} for s in ol.T19_SETS},
        "rule_b_differences": copy.deepcopy(DIFF19),
        "check_submission": {"mode": "official", "questions": [], "errors": []},
        "outputs_sha256": {"submission_rank.csv": sha(f19)}, "inputs_bound": bound}


# ---------------------------------------------------------------- the tag version against the new one
class TagVersusNew(t14o.TagVersusNew):
    """A log without 19 entries gives the tag p14-testday-pre19 version's log bytes, records, stdout and refusals, with
    a results/test19 tree on disk (the R20 / R10 / 04 / 06 flows)."""

    @classmethod
    def setUpClass(cls):
        cls.tag = t7o.load_version(TAG)
        assert hasattr(cls.tag, "check_reports_t14") and not hasattr(cls.tag, "check_reports_t19"), "not the tag"

    def build(self, kind, with_t06):
        out = super().build(kind, with_t06)
        write_t19_tree(out[0], [[q, "D"] for q in t9o.t6o.IDS])
        return out


class _Flow19:
    def setUp(self):
        self.tag = t7o.load_version(TAG)

    def run_with(self, mod):
        real = t9o.write_t09_tree

        def both(repo, rows, *a, **k):
            out = real(repo, rows, *a, **k)
            write_t19_tree(repo, rows)
            return out
        with mock.patch.object(t9o, "write_t09_tree", side_effect=both):
            return super().run_with(mod)

    def check_identical(self):
        old = super().check_identical()
        log = json.loads(old[-1][3])
        self.assertFalse(any("test19" in e or "prereg19_4_3" in e for e in log["uploads"]))
        self.assertNotIn("prereg19_4_3", json.dumps(log["final_choice"]))
        return old


class TagVersusNew07Flow(_Flow19, t14o.TagVersusNew07Flow):
    pass


class TagVersusNew08Flow(_Flow19, t14o.TagVersusNew08Flow):
    pass


class TagVersusNew09Flow(_Flow19, t14o.TagVersusNew09Flow):
    pass


class TagVersusNew14Flow(_Flow19, t9o.FlowBase):
    """R20 -> 04 -> 07 -> 08 -> 09 uploaded, final 09, the 14 tree on the 09 file, 14 prepared and uploaded, final 14,
    a refused final 09 (needs --not-14-reason), status: the 14 layer behaves as before when 19 is not in the log."""

    def steps(self, step):
        mod = t9o.t8o.ol                              # the version under test (FlowBase patched the helpers' module)
        prev = mod.git_head_text
        self.rev_added["| rev 2 |"] = ("2" * 40, "2026-10-20T10:00:00+08:00")
        head = mock.patch.object(mod, "git_head_text", side_effect=lambda rel: t14o.PREREG14_HEAD
                                 if rel == mod.P14_DOC else prev(rel))
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
            step("final 09", self.run_ol, "final", "--comp", "rank", "--id", "5", "--basis", "PREREG_09 5.3")
            t14o.write_t14_tree(self.repo, "results/test09/rank/submission_rank.csv", 5)
            f14 = self.r("results/test14/rank/submission_rank.csv")
            step("prepare 14", lambda: self.prepare("rank", f14, "prereg14 exact-copy layer (PREREG_14 7)"))
            step("uploaded 14", self.uploaded, 6, "2026-11-21T10:00")
            step("final 14", self.run_ol, "final", "--comp", "rank", "--id", "6", "--basis", "PREREG_14 7")
            step("final 09 refused", self.run_ol, "final", "--comp", "rank", "--id", "5", "--basis", "x")
            step("status", self.run_ol, "status", "--now", "2026-11-22T00:00")
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
        log = json.loads(old[-1][3])
        self.assertFalse(any("test19" in e or "prereg19_4_3" in e for e in log["uploads"]))
        return old

    def test_14_flow_identical(self):
        old = self.check_identical()
        log = json.loads(old[-1][3])
        self.assertTrue(any("test14" in e for e in log["uploads"]), "the flow must hold a 14 entry")
        self.assertEqual(json.loads(old[13][2])["prereg14_7"]["chosen"], "14")
        self.assertIn("refused", old[14][2])


# ---------------------------------------------------------------- the 19 path
class T19Base(t14o.T14Base):
    def setUp(self):
        super().setUp()
        self.p19_head = PREREG19_HEAD
        prev = ol.git_head_text                       # the 04-14 mock
        self.patches19 = [mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p19_head
                                            if rel == ol.P19_DOC else prev(rel))]
        for pt in self.patches19:
            pt.start()
        self.f19 = write_t19_tree(self.repo, self.rank_rows)

    def tearDown(self):
        for pt in reversed(self.patches19):
            pt.stop()
        super().tearDown()

    def edit19(self, fn):
        p = self.r(f"results/test19/rank/{ol.T19_REPORT}")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def prepare19(self, *extra):
        return self.prepare("rank", self.f19, "prereg19 Q19 (PREREG_19 4.3)", *extra)

    def ready19(self):
        """R20 #1, 04 #2, 07 #3, 08 #4 and 09 #5 uploaded."""
        self.ready9()
        e9, _ = self.up_09()
        return e9

    def up_19(self, at="2026-11-09T13:00"):
        e19, out = self.prepare19()
        self.uploaded(e19["id"], at)
        return e19, out

    def rev4_19(self, at="2026-11-10T10:00:00+08:00"):
        self.p19_head = PREREG19_HEAD + "| rev 4 | `abc` | bug |\n"
        self.rev_added["| rev 4 |"] = ("1" * 40, at)


class T19Prepare(T19Base):
    def test_order_r20_to_09_then_19(self):
        msg = self.refused(self.prepare19)
        self.assertIn("R20's first upload", msg)
        self.ready9()                                      # R20, 04, 07, 08 uploaded; 09 not
        msg = self.refused(self.prepare19)
        self.assertIn("not logged as uploaded", msg)
        e9, _ = self.prepare09()
        self.assertIn("not logged as uploaded", self.refused(self.prepare19))   # 09 prepared, not uploaded
        self.uploaded(e9["id"], "2026-11-08T13:00")
        e19, out = self.prepare19()
        self.assertEqual((e19["id"], e19["file"]), (6, "results/test19/rank/submission_rank.csv"))
        self.assertEqual(e19["method"], ol.T19_METHOD)
        t19 = e19["test19"]
        self.assertEqual((t19["r20_first_upload_id"], t19["t09_entry_id"], t19["t08_entry_id"], t19["t07_entry_id"],
                          t19["t04_entry_id"], t19["combo"]), (1, 5, 4, 3, 2, "Q19"))
        self.assertEqual(t19["public_rule_b_differences"], {k: v["public"] for k, v in DIFF19.items()})
        self.assertEqual(t19["k6_19"]["deadline"], "2026-11-19T23:59:59+08:00")
        for k in ("test09", "test08", "prereg04_5_3", "prereg07_5_4", "prereg08_5_4", "prereg09_5_3", "prereg19_4_3"):
            self.assertNotIn(k, e19)
        for s in ("這一份是 19", "Q19", "#5", "K6-19", "沒在驗證集打過分"):
            self.assertIn(s, out)

    def test_conditions_recomputed(self):
        self.ready19()
        cases = [
            ("q09 sha", lambda r: r["q09_vs_09"].update(t09_sha256="0" * 64), "Q09 recomputed"),
            ("q09 diff", lambda r: r["q09_vs_09"].update(max_abs_diff=2e-12), "Q09 recomputed"),
            ("q09 lp", lambda r: r["q09_vs_09"].update(lp_p1e_max_abs_diff_vs_t09=2e-12), "Q09 recomputed"),
            ("q09 rank", lambda r: r["q09_vs_09"].update(rule_b_differences=1), "Q09 recomputed"),
            ("q09 usable", lambda r: r["q09_vs_09"].update(usable_equal_t09=False), "Q09 recomputed"),
            ("seam bytes", lambda r: r["seam_local"].update(byte_identical_pred_B=False), "local Q19 seam"),
            ("seam tol", lambda r: r["seam_local"].update(required_byte_identical=False, max_abs_diff=2e-4),
             "local Q19 seam"),
            ("seam b", lambda r: r["seam_local"].update(required_byte_identical=False, rule_b_differences=5),
             "local Q19 seam"),
            ("seam usable", lambda r: r["seam_local"].update(usable_equal_to_prereg19=False), "local Q19 seam"),
            ("rho", lambda r: r["set_direction"]["private"].update(rho=0.479), "SET direction"),
            ("fake", lambda r: r["p1_sets"]["public"].update(fake_from_val=True), "drill records"),
            ("mode", lambda r: r.update(mode="dry-run"), "real-mode"),
            ("exit", lambda r: r.update(exit_code=3), "exit_code"),
            ("combo", lambda r: r["combo"].update(id="Q09"), "not Q19"),
            ("sources", lambda r: r["combo"].update(sources=list(ol.T09_COMBO_SOURCES)), "not Q19"),
            ("check_submission", lambda r: r["check_submission"].update(errors=["x"]), "check_submission")]
        for name, fn, msg in cases:
            with self.subTest(name):
                self.f19 = write_t19_tree(self.repo, self.rank_rows)
                self.edit19(fn)
                self.assertIn(msg, self.refused(self.prepare19))
        self.f19 = write_t19_tree(self.repo, self.rank_rows)
        self.edit19(lambda r: r["set_direction"]["private"].update(rho=0.4793))
        e, _ = self.prepare19()                            # rho at the threshold passes
        self.assertAlmostEqual(e["test19"]["rho_threshold"], 0.4792752582816624, places=15)

    def test_method_and_reasons(self):
        self.ready19()
        self.assertIn("does not name Q19", self.refused(self.prepare19, "--method", "prereg09 Q09"))
        self.assertIn("not to a 19 file", self.refused(self.prepare19, "--not-04-reason", "PREREG_04 rev 4: x"))
        self.assertIn("04 / R20 / R10 file only", self.refused(self.prepare19, "--not-07-reason", "PREREG_07 rev 5: x"))
        self.assertIn("07 / 04 / R20 / R10 file only", self.refused(self.prepare19, "--not-08-reason",
                                                                    "PREREG_08 rev 4: x"))
        self.assertIn("08 / 07 / 04 / R20 / R10 file only", self.refused(self.prepare19, "--not-09-reason",
                                                                         "PREREG_09 rev 5: x"))
        self.assertIn("09 / 08 / 07 / 04 / R20 / R10 file only", self.refused(self.prepare19, "--not-19-reason",
                                                                              "PREREG_19 rev 4: x"))
        e, _ = self.prepare19("--method", "Q19 rule B")
        self.assertEqual(e["method"], "Q19 rule B")

    def test_adoption_record(self):
        self.ready19()
        self.assertEqual(ol.t19_adoption_problems(), [])
        p = self.r(ol.T19_ADOPTION)
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        write_json(p, dict(s, usable="pending_audit"))
        self.assertIn("is not the committed adoption record", self.refused(self.prepare19))
        os.remove(p)
        self.assertIn("19 was not adopted", self.refused(self.prepare19))

    def test_bound_input_changed_and_k6(self):
        self.ready19()
        with open(self.r("results/prereg19/rank/s19_seed1.npz"), "ab") as f:
            f.write(b"x")
        self.assertIn("results/prereg19/rank/s19_seed1.npz is", self.refused(self.prepare19))
        self.f19 = write_t19_tree(self.repo, self.rank_rows)
        self.commit = ("f" * 40, "2026-11-20T00:00:00+08:00")
        self.assertIn("not by K6-19", self.refused(self.prepare19))

    def test_bound_09_not_eligible(self):
        self.ready19()
        self.edit09(lambda r: r["p1_direction"]["public"].update(rho=0.1))
        self.assertIn("results/test09/rank/t09_apply_report.json is", self.refused(self.prepare19))
        self.f19 = write_t19_tree(self.repo, self.rank_rows)   # a 19 run bound to the changed 09 report
        self.assertIn("no longer meets PREREG_09 5.3", self.refused(self.prepare19))

    def test_p1_before_r20_upload_refused(self):
        self.ready19()
        self.edit19(lambda r: r.update(earliest_p1_chunk_created_utc="2026-11-05T06:00:00+00:00"))
        self.assertIn("not after R20's first upload", self.refused(self.prepare19))


class T19Final(T19Base):
    def test_final_19_and_status(self):
        e9 = self.ready19()
        e19, _ = self.up_19()
        self.assertIn("needs --not-19-reason", self.refused(self.run_ol, "final", "--comp", "rank", "--id",
                                                            str(e9["id"]), "--basis", "x"))
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e19["id"]), "--basis", "PREREG_19 4.3")
        self.assertEqual(fc["prereg19_4_3"]["chosen"], "19")
        for k in ("prereg04_5_3", "prereg07_5_4", "prereg08_5_4", "prereg09_5_3"):
            self.assertNotIn(k, fc)
        self.assertIn("PREREG_19 4.3：最終選擇是 19", out)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        for k in ("prereg04", "prereg07", "prereg08", "prereg09", "prereg19"):
            self.assertFalse(st["rank"][k]["violation"], k)
        self.assertEqual(st["rank"]["prereg04"]["t19_eligible"], [e19["id"]])
        self.assertNotIn("違反", out)
        self.assertIn("K6-19", out)

    def test_final_09_needs_not_19_reason(self):
        e9 = self.ready19()
        self.up_19()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "x",
                           "--not-19-reason", "PREREG_19 rev 3: bug")
        self.assertIn("rev 4 or later", msg)
        self.assertIn("has no row", self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e9["id"]),
                                                 "--basis", "x", "--not-19-reason", "PREREG_19 rev 4: bug"))
        self.rev4_19(at="2026-11-04T10:00:00+08:00")
        self.assertIn("not after the records", self.refused(self.run_ol, "final", "--comp", "rank", "--id",
                                                            str(e9["id"]), "--basis", "x", "--not-19-reason",
                                                            "PREREG_19 rev 4: bug"))
        self.rev4_19()
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "x", "--not-19-reason",
                            "PREREG_19 rev 4: bug in t19_rank apply")
        self.assertEqual((fc["prereg19_4_3"]["chosen"], fc["prereg19_4_3"]["revision"]["rev"]), ("09", 4))
        self.assertEqual(fc["prereg09_5_3"]["chosen"], "09")
        st, _ = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertFalse(st["rank"]["prereg19"]["violation"])
        self.assertFalse(st["rank"]["prereg09"]["violation"])

    def test_19_no_longer_eligible(self):
        self.ready19()
        e19, _ = self.up_19()
        with open(self.r(ol.T19_SET_SEAM), "ab") as f:
            f.write(b" ")
        self.assertIn("does not meet PREREG_19 4.3 now", self.refused(self.run_ol, "final", "--comp", "rank",
                                                                      "--id", str(e19["id"]), "--basis", "x"))
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", "5", "--basis", "PREREG_09 5.3")
        self.assertEqual((fc["prereg19_4_3"]["eligible_19_entries"], fc["prereg19_4_3"]["chosen"]), ([], "09"))
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertTrue(st["rank"]["prereg19"]["redo_09"])
        self.assertIn("現在沒有符合的 19", out)

    def test_status_hint_and_violation(self):
        e9 = self.ready19()
        self.run_ol("final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "PREREG_09 5.3")
        e19, _ = self.up_19()
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertTrue(st["rank"]["prereg19"]["violation"])
        self.assertIn("違反 PREREG_19 4.3", out)
        log = self.log_obj()
        log["final_choice"]["rank"] = None
        write_json(self.log, log)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertIn("最終選擇要設成 19", out)
        for k in ("09", "08", "07", "04"):
            self.assertNotIn(f"最終選擇要設成 {k}", out)

    def test_prepare_09_after_19(self):
        self.ready19()
        self.up_19()
        self.f09 = t9o.write_t09_tree(self.repo, self.rank_rows)     # the same 09 run, prepared again
        self.f19 = write_t19_tree(self.repo, self.rank_rows)
        self.assertIn("needs --not-19-reason", self.refused(self.prepare09))
        self.rev4_19()
        e, out = self.prepare09("--not-19-reason", "PREREG_19 rev 4: bug")
        self.assertEqual(e["prereg19_4_3"]["revision"]["rev"], 4)
        self.assertIn("這一份不是 19", out)


class T19With14(T19Base):
    def test_14_on_a_19_base(self):
        self.ready19()
        e19, _ = self.up_19()
        self.run_ol("final", "--comp", "rank", "--id", str(e19["id"]), "--basis", "PREREG_19 4.3")
        self.f14 = t14o.write_t14_tree(self.repo, "results/test19/rank/submission_rank.csv", e19["id"])
        e14, out = self.prepare14()
        self.assertEqual(e14["test14"]["base_rules"].get("prereg19_4_3"), "19")
        self.uploaded(e14["id"], "2026-11-21T10:00")
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e14["id"]), "--basis", "PREREG_14 7")
        self.assertEqual((fc["prereg14_7"]["chosen"], fc["prereg14_7"]["base_entry_id"]), ("14", e19["id"]))
        self.assertEqual(fc["prereg19_4_3"]["chosen"], "19")             # 4.3 judged on the base
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        for k in ("prereg04", "prereg07", "prereg08", "prereg09", "prereg19", "prereg14"):
            self.assertFalse(st["rank"][k]["violation"], k)
        self.assertTrue(st["rank"]["prereg14"]["final_is_14"])
        self.assertNotIn("違反", out)
        self.assertIn("results/test19/rank", ol.T14_BASE_DIRS)

    def test_14_on_09_while_19_holds_is_refused(self):
        e9 = self.ready19()
        self.up_19()
        self.f14 = t14o.write_t14_tree(self.repo, "results/test09/rank/submission_rank.csv", e9["id"])
        self.assertIn("final_choice.rank", self.refused(self.prepare14))      # no final yet
        self.assertIn("needs --not-19-reason", self.refused(self.run_ol, "final", "--comp", "rank", "--id",
                                                            str(e9["id"]), "--basis", "x"))


class T19Public(T19Base):
    def test_same_score_warning(self):
        e9 = self.ready19()
        self.run_ol("public", "--id", str(e9["id"]), "--score", "0.71")
        e19, _ = self.up_19()
        _e, out = self.run_ol("public", "--id", str(e19["id"]), "--score", "0.71")
        self.assertIn("WARNING (PREREG_19 4.3)", out)
        self.assertIn("（09）", out)
        for w in ("WARNING (PREREG_09 5.3)", "WARNING (PREREG_08 5.4)", "WARNING (PREREG_07 5.4)",
                  "WARNING (PREREG_04 rev 3)"):
            self.assertNotIn(w, out)


if __name__ == "__main__":
    unittest.main()
