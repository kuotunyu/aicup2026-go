"""Tests for the PREREG_09 rev 3 additions to tools/official_log.py (PREREG_09 5.2, 5.3).

* TagVersusNew: the version of tag p08-testday-pre09 (R20, 04, 06, 07 and 08 additions, no 09) and the working-tree
  version run the same flows with the same mocks (test_t08_official_log.TagVersusNew's R20, R10, 04 and 06 flows, the
  07 flow, and an 08 flow: R20 -> 04 -> 07 -> 08 uploaded, public scores, the refused 07 final, final 08, the refused
  07 prepare, status), now with a results/test09 tree on disk; the log bytes, returned records, stdout and refusals
  are identical (a log without 09 entries).
* T09*: the 09 path (prepare and its order rules, the refusals recomputed from the report's numbers, --method, the
  adoption record, final under PREREG_09 5.3 with --not-09-reason, the 08 / 07 / 04 rules skipping 09, status, the
  same-file warning).
Synthetic files in a temporary directory only; git and the clock are mocked. Nothing here opens a label file or runs
tools/score.py. Stdlib only besides the module under test.
Run: python -m pytest tests/test_t09_official_log.py
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
import test_t06_official_log as t6o  # noqa: E402
import test_t07_official_log as t7o  # noqa: E402
import test_t08_official_log as t8o  # noqa: E402

TAG = "p08-testday-pre09"                          # the 08 test-day path before the 09 kit
ADOPTION_FILE = os.path.join(REPO, "results", "prereg09", "rank", "apply_report.json")
P1_TIME = t8o.P1_TIME
PREREG09_HEAD = ("# PREREG_09\n\n| rev 1 | `73b4deb` | first |\n| rev 2 | `e0ea7fe` | second |\n"
                 "| rev 3 | (commit) | test day |\n| rev 4 | (commit) | EX rule |\n")
DIFF = {"08": {"public": 74, "all": 150}, "07": {"public": 134, "all": 260}, "04": {"public": 151, "all": 300},
        "R20": {"public": 169, "all": 330}}
sha, write_json, quiet = t4o.sha, t4o.write_json, t4o.quiet
NEW = ol


# ---------------------------------------------------------------- the results/test09 tree of a usable 09 run
def write_t09_tree(repo, rank_rows, report_over=None):
    """results/test09 as tools/t09_rank.py apply leaves it (real mode) plus the committed inputs it binds. The 08
    report it binds is the one in results/test08/rank (a placeholder is written when there is none). -> 09 file."""
    r = lambda rel: os.path.join(repo, *rel.split("/"))   # noqa: E731
    os.makedirs(os.path.dirname(r(NEW.T09_ADOPTION)), exist_ok=True)
    shutil.copyfile(ADOPTION_FILE, r(NEW.T09_ADOPTION))
    for rel in NEW.T09_INPUTS_REQUIRED:
        if not os.path.exists(r(rel)):
            if rel.endswith(".json"):
                write_json(r(rel), {"passed": True, "mode": "real", "fake_input": rel})
            elif rel.endswith(".csv"):
                ti.write_csv(r(rel), ["question_id", "pred_rank"], [[q, "D"] for q, _ in rank_rows])
            else:
                os.makedirs(os.path.dirname(r(rel)), exist_ok=True)
                with open(r(rel), "wb") as f:
                    f.write(b"PK-fake-npz " + rel.encode() + b"\r\n")
    rows = [[q, "1D" if i % 4 == 0 else lab] for i, (q, lab) in enumerate(rank_rows)]
    f09 = ti.write_csv(r("results/test09/rank/submission_rank.csv"), ["question_id", "pred_rank"], rows)
    rep = report09(repo, f09)
    rep.update(report_over or {})
    write_json(r(f"results/test09/rank/{NEW.T09_REPORT}"), rep)
    return f09


def report09(repo, f09):
    bound = {rel: NEW.sha256_bound(os.path.join(repo, *rel.split("/"))) for rel in NEW.T09_INPUTS_REQUIRED}
    return {
        "format": NEW.T09_FORMAT, "mode": "real", "created": "2026-11-08T12:00:00+08:00", "exit_code": 0,
        "usable": True, "unusable_reasons": [],
        "combo": {"id": "Q09", "sources": list(NEW.T09_COMBO_SOURCES),
                  "terms": ["OP", "O4", "SWA", "SWB", "HS", "P1TO"]},
        "p1_sets": {s: {"fake_from_val": False, "earliest_chunk_created_utc": P1_TIME} for s in NEW.T09_P1_SETS},
        "earliest_p1_chunk_created_utc": P1_TIME,
        "gpu_seam": {"passed": True, "counts_equal": True, "max_abs_std_ex": 0.0, "max_abs_lp_p1e": 0.0},
        "q08_vs_08": {"equal": True, "byte_equal": True, "t08_sha256": bound[NEW.T09_08_PROBS],
                      "computed_sha256": bound[NEW.T09_08_PROBS], "max_abs_diff": 0.0, "rule_a_differences": 0,
                      "rule_b_differences": 0, "p1_ok_equal_t08": True, "usable_equal_t08": True},
        "seam_local": {"verdict": "passed", "passed": True, "max_abs_diff": 0.0, "tol": 1e-4, "rule_b_differences": 0,
                       "byte_identical_probs": True, "byte_identical_pred_B": True, "required_byte_identical": True},
        "p1_failures": {s: {"failed_or_no_move": 0, "games": 5000, "ok": True} for s in NEW.T09_SETS},
        "p1_direction": {s: {"rho": 0.97, "threshold": 0.48425802602327594, "passed": True} for s in NEW.T09_SETS},
        "rule_b_differences": copy.deepcopy(DIFF),
        "check_submission": {"mode": "official", "questions": [], "errors": []},
        "outputs_sha256": {"submission_rank.csv": sha(f09)}, "inputs_bound": bound}


# ---------------------------------------------------------------- the tag version against the new one
class TagVersusNew(t8o.TagVersusNew):
    """A log without 09 entries gives the tag p08-testday-pre09 version's log bytes, records, stdout and refusals,
    with a results/test09 tree on disk (the R20 / R10 / 04 / 06 flows of test_t07_official_log)."""

    @classmethod
    def setUpClass(cls):
        cls.tag = t7o.load_version(TAG)
        assert hasattr(cls.tag, "check_reports_t08") and not hasattr(cls.tag, "check_reports_t09"), "not the tag"

    def build(self, kind, with_t06):
        out = super().build(kind, with_t06)
        write_t09_tree(out[0], [[q, "D"] for q in t6o.IDS])
        return out


class FlowBase(t8o.T08Base):
    """Runs a flow with the tag version and with the new one (the test helpers' module references patched)."""

    def setUp(self):
        self.tag = t7o.load_version(TAG)

    def tearDown(self):
        pass

    def steps(self, step):
        raise NotImplementedError

    def run_with(self, mod):
        patches = [mock.patch.object(m, "ol", mod) for m in (ti, t4o, t7o, t8o)]
        for pt in patches:
            pt.start()
        try:
            t8o.T08Base.setUp(self)
            write_t09_tree(self.repo, self.rank_rows)
            out = []

            def step(name, fn, *a):
                try:
                    res, text = fn(*a)
                except SystemExit as e:
                    res, text = ("refused", str(e)), ""
                js = json.dumps(res, sort_keys=True, ensure_ascii=False)
                for root in (self.repo, json.dumps(self.repo)[1:-1]):    # the temporary folders differ per run
                    text, js = text.replace(root, "<REPO>"), js.replace(root, "<REPO>")
                with open(self.log, "rb") as f:
                    out.append((name, text, js, f.read()))
            self.steps(step)
            return out
        finally:
            t8o.T08Base.tearDown(self)
            for pt in reversed(patches):
                pt.stop()

    def check_identical(self):
        old, new = self.run_with(self.tag), self.run_with(ol)
        self.assertEqual([s[0] for s in old], [s[0] for s in new])
        for (name, o_out, o_res, o_log), (_n, n_out, n_res, n_log) in zip(old, new):
            with self.subTest(step=name):
                self.assertEqual(n_log, o_log)
                self.assertEqual(n_out, o_out)
                self.assertEqual(n_res, o_res)
        log = json.loads(old[-1][3])
        self.assertFalse(any("test09" in e or "prereg09_5_3" in e for e in log["uploads"]))
        self.assertNotIn("prereg09_5_3", json.dumps(log["final_choice"]))
        return old


class TagVersusNew07Flow(FlowBase):
    """The 07 flow of test_t08_official_log (R20 -> 04 -> 07, public, final 07, a refused 04 prepare, status) with a
    results/test08 and a results/test09 tree on disk."""

    def steps(self, step):
        step("prepare R20", self.prepare)
        step("uploaded R20", self.run_ol, "uploaded", "--id", "1", "--ok-at", "2026-11-05T14:10", "--uploaded-at",
             "2026-11-05T14:15")
        step("prepare 04", self.prepare04)
        step("uploaded 04", self.uploaded, 2, "2026-11-05T15:10")
        step("public 04", self.run_ol, "public", "--id", "2", "--score", "0.62")
        step("prepare 07", self.prepare07)
        step("uploaded 07", self.uploaded, 3, "2026-11-06T13:00")
        step("public 07", self.run_ol, "public", "--id", "3", "--score", "0.62")
        step("final R20 refused", self.run_ol, "final", "--comp", "rank", "--id", "1", "--basis", "x")
        step("final 07", self.run_ol, "final", "--comp", "rank", "--id", "3", "--basis", "PREREG_07 5.4")
        step("prepare 04 again refused", self.prepare04)
        step("status", self.run_ol, "status", "--now", "2026-11-10T00:00")

    def test_07_flow_identical(self):
        old = self.check_identical()
        log = json.loads(old[-1][3])
        self.assertTrue(any("test07" in e for e in log["uploads"]), "the flow must hold a 07 entry")


class TagVersusNew08Flow(FlowBase):
    """An 08 flow (R20 -> 04 -> 07 -> 08 uploaded, public scores, the refused 07 final, final 08, the refused 07
    prepare, status) with a results/test09 tree on disk."""

    def steps(self, step):
        step("prepare R20", self.prepare)
        step("uploaded R20", self.run_ol, "uploaded", "--id", "1", "--ok-at", "2026-11-05T14:10", "--uploaded-at",
             "2026-11-05T14:15")
        step("prepare 04", self.prepare04)
        step("uploaded 04", self.uploaded, 2, "2026-11-05T15:10")
        step("prepare 07", self.prepare07)
        step("uploaded 07", self.uploaded, 3, "2026-11-06T13:00")
        step("public 07", self.run_ol, "public", "--id", "3", "--score", "0.66")
        step("prepare 08", self.prepare08)
        step("uploaded 08", self.uploaded, 4, "2026-11-07T13:00")
        step("public 08", self.run_ol, "public", "--id", "4", "--score", "0.66")
        step("final 07 refused", self.run_ol, "final", "--comp", "rank", "--id", "3", "--basis", "x")
        step("final 08", self.run_ol, "final", "--comp", "rank", "--id", "4", "--basis", "PREREG_08 5.4")
        step("prepare 07 again refused", self.prepare07)
        step("status", self.run_ol, "status", "--now", "2026-11-10T00:00")

    def test_08_flow_identical(self):
        old = self.check_identical()
        log = json.loads(old[-1][3])
        self.assertTrue(any("test08" in e for e in log["uploads"]), "the flow must hold an 08 entry")
        self.assertEqual(json.loads(old[11][2])["prereg08_5_4"]["chosen"], "08")
        self.assertIn("refused", old[10][2])
        self.assertIn("WARNING (PREREG_08 5.4)", old[9][1])


# ---------------------------------------------------------------- the 09 path
class T09Base(t8o.T08Base):
    def setUp(self):
        super().setUp()
        self.p09_head = PREREG09_HEAD
        self.patches09 = [mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p09_head
                                            if rel == ol.P09_DOC else self.p08_head if rel == ol.P08_DOC
                                            else self.p07_head if rel == ol.P07_DOC else self.p04_head)]
        for pt in self.patches09:
            pt.start()
        self.f09 = write_t09_tree(self.repo, self.rank_rows)

    def tearDown(self):
        for pt in reversed(self.patches09):
            pt.stop()
        super().tearDown()

    def edit09(self, fn):
        p = self.r(f"results/test09/rank/{ol.T09_REPORT}")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def prepare09(self, *extra):
        return self.prepare("rank", self.f09, "prereg09 Q09 (PREREG_09 5.3)", *extra)

    def ready9(self):
        """R20 #1, 04 #2, 07 #3 and 08 #4 uploaded."""
        self.ready8()
        e8, _ = self.up_08()
        return e8

    def up_09(self, at="2026-11-08T13:00"):
        e9, out = self.prepare09()
        self.uploaded(e9["id"], at)
        return e9, out

    def rev5_09(self, at="2026-11-09T10:00:00+08:00"):
        self.p09_head = PREREG09_HEAD + "| rev 5 | `abc` | bug |\n"
        self.rev_added["| rev 5 |"] = ("9" * 40, at)


class T09Prepare(T09Base):
    def test_order_r20_04_07_08_then_09(self):
        msg = self.refused(self.prepare09)
        self.assertIn("R20's first upload", msg)
        self.ready8()                                      # R20, 04, 07 uploaded; 08 not
        msg = self.refused(self.prepare09)
        self.assertIn("not logged as uploaded", msg)
        e8, _ = self.prepare08()
        msg = self.refused(self.prepare09)                 # 08 prepared, not uploaded
        self.assertIn("not logged as uploaded", msg)
        self.uploaded(e8["id"], "2026-11-07T13:00")
        e9, out = self.prepare09()
        self.assertEqual((e9["id"], e9["file"]), (5, "results/test09/rank/submission_rank.csv"))
        self.assertEqual(e9["method"], ol.T09_METHOD)
        t9 = e9["test09"]
        self.assertEqual((t9["r20_first_upload_id"], t9["t08_entry_id"], t9["t07_entry_id"], t9["t04_entry_id"],
                          t9["combo"]), (1, 4, 3, 2, "Q09"))
        self.assertEqual(t9["public_rule_b_differences"], {"08": 74, "07": 134, "04": 151, "R20": 169})
        self.assertEqual(t9["k6_09"]["deadline"], "2026-11-19T23:59:59+08:00")
        for k in ("test08", "test07", "test04", "prereg04_5_3", "prereg07_5_4", "prereg08_5_4", "prereg09_5_3"):
            self.assertNotIn(k, e9)
        for s in ("這一份是 09", "Q09", "#4", "K6-09", "沒在驗證集打過分"):
            self.assertIn(s, out)

    def test_p1_before_r20_upload_refused(self):
        self.ready9()
        self.edit09(lambda r: r.update(earliest_p1_chunk_created_utc="2026-11-05T06:00:00+00:00"))
        self.assertIn("not after R20's first upload", self.refused(self.prepare09))

    def test_method(self):
        self.ready9()
        self.assertIn("does not name Q09", self.refused(self.prepare09, "--method", "prereg08 Q08"))
        e, _ = self.prepare09("--method", "Q09 rule B")
        self.assertEqual(e["method"], "Q09 rule B")

    def test_other_reasons_on_a_09_file_refused(self):
        self.ready9()
        self.assertIn("not to a 09 file", self.refused(self.prepare09, "--not-04-reason", "PREREG_04 rev 4: x"))
        self.assertIn("04 / R20 / R10 file only", self.refused(self.prepare09, "--not-07-reason", "PREREG_07 rev 5: x"))
        self.assertIn("07 / 04 / R20 / R10 file only", self.refused(self.prepare09, "--not-08-reason",
                                                                    "PREREG_08 rev 4: x"))
        self.assertIn("08 / 07 / 04 / R20 / R10 file only", self.refused(self.prepare09, "--not-09-reason",
                                                                         "PREREG_09 rev 4: x"))

    def test_conditions_recomputed(self):
        self.ready9()
        cases = [
            ("gpu ex", lambda r: r["gpu_seam"].update(max_abs_std_ex=0.051), "largest std_ex"),
            ("gpu lp", lambda r: r["gpu_seam"].update(max_abs_lp_p1e=0.041), "largest lp_p1e"),
            ("gpu counts", lambda r: r["gpu_seam"].update(counts_equal=False), "09 GPU seam: failed"),
            ("q08 sha", lambda r: r["q08_vs_08"].update(t08_sha256="0" * 64), "Q08 recomputed"),
            ("q08 diff", lambda r: r["q08_vs_08"].update(max_abs_diff=2e-12), "Q08 recomputed"),
            ("q08 rank", lambda r: r["q08_vs_08"].update(rule_b_differences=1), "Q08 recomputed"),
            ("q08 usable", lambda r: r["q08_vs_08"].update(usable_equal_t08=False), "Q08 recomputed"),
            ("seam bytes", lambda r: r["seam_local"].update(byte_identical_pred_B=False), "local Q09 seam"),
            ("seam tol", lambda r: r["seam_local"].update(required_byte_identical=False, max_abs_diff=2e-4),
             "local Q09 seam"),
            ("seam b", lambda r: r["seam_local"].update(required_byte_identical=False, rule_b_differences=5),
             "local Q09 seam"),
            ("failures 26", lambda r: r["p1_failures"]["public"].update(failed_or_no_move=26), "P1 failures"),
            ("rho", lambda r: r["p1_direction"]["private"].update(rho=0.484), "P1TO_E direction"),
            ("fake", lambda r: r["p1_sets"]["public"].update(fake_from_val=True), "drill records"),
            ("mode", lambda r: r.update(mode="dry-run"), "real-mode"),
            ("exit", lambda r: r.update(exit_code=3), "exit_code"),
            ("combo", lambda r: r["combo"].update(id="Q08"), "not Q09"),
            ("sources", lambda r: r["combo"].update(sources=list(ol.T08_COMBO_SOURCES)), "not Q09"),
            ("check_submission", lambda r: r["check_submission"].update(errors=["x"]), "check_submission")]
        for name, fn, msg in cases:
            with self.subTest(name):
                self.f09 = write_t09_tree(self.repo, self.rank_rows)
                self.edit09(fn)
                self.assertIn(msg, self.refused(self.prepare09))
        self.f09 = write_t09_tree(self.repo, self.rank_rows)
        self.edit09(lambda r: r["p1_failures"]["public"].update(failed_or_no_move=25))
        self.edit09(lambda r: r["p1_direction"]["private"].update(rho=0.4843))
        e, _ = self.prepare09()                            # 25 of 5,000 passes; rho at the threshold passes
        self.assertEqual(e["test09"]["p1_failure_rate"]["public"], 0.005)

    def test_adoption_record(self):
        self.ready9()
        p = self.r(ol.T09_ADOPTION)
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        self.assertEqual(ol.t09_adoption_problems(), [])
        write_json(p, dict(s, usable="pending_audit"))
        self.assertIn("is not the committed adoption record", self.refused(self.prepare09))
        with mock.patch.object(ol, "T09_ADOPTION_SHA", None), mock.patch.object(ol, "sha256_bound",
                                                                                lambda path: None):
            self.assertTrue(any("09 is not adopted" in x for x in ol.t09_adoption_problems()))
        os.remove(p)
        self.assertIn("09 was not adopted", self.refused(self.prepare09))

    def test_h8r_needed(self):
        self.ready9()
        os.remove(self.r(ol.T08_H8R))
        self.assertIn("H8R was not scored", self.refused(self.prepare09))

    def test_bound_input_changed_and_uncommitted(self):
        self.ready9()
        with open(self.r("results/prereg09/rank/p1e_map.npz"), "ab") as f:
            f.write(b"x")
        self.assertIn("results/prereg09/rank/p1e_map.npz is", self.refused(self.prepare09))
        self.f09 = write_t09_tree(self.repo, self.rank_rows)
        self.dirty.add("results/test09/gpu_seam.json")
        self.assertIn("gpu_seam.json must be committed", self.refused(self.prepare09))

    def test_k6_09(self):
        self.ready9()
        self.commit = ("f" * 40, "2026-11-20T00:00:00+08:00")
        self.assertIn("not by K6-09", self.refused(self.prepare09))

    def test_t1_fail_opening(self):
        self.ready9()
        self.write_reports(t1=("FAIL_OPENING", "PASS"))
        self.assertIn("PASS or NORMALIZED", self.refused(self.prepare09))

    def test_bound_08_not_eligible(self):
        self.ready9()
        self.edit08(lambda r: r["p1_direction"]["public"].update(rho=0.1))
        msg = self.refused(self.prepare09)                 # the 09 report bound the earlier 08 report
        self.assertIn("results/test08/rank/t08_apply_report.json is", msg)
        self.f09 = write_t09_tree(self.repo, self.rank_rows)  # a 09 run on the changed 08
        msg = self.refused(self.prepare09)
        self.assertIn("no longer meets PREREG_08 5.4", msg)


class T09Final(T09Base):
    def test_final_09(self):
        e8 = self.ready9()
        e9, _ = self.up_09()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e8["id"]), "--basis", "x")
        self.assertIn("needs --not-09-reason", msg)
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "PREREG_09 5.3")
        self.assertEqual(fc["prereg09_5_3"]["chosen"], "09")
        for k in ("prereg04_5_3", "prereg07_5_4", "prereg08_5_4"):
            self.assertNotIn(k, fc)
        self.assertIn("PREREG_09 5.3：最終選擇是 09", out)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        for k in ("prereg04", "prereg07", "prereg08", "prereg09"):
            self.assertFalse(st["rank"][k]["violation"], k)
        self.assertEqual(st["rank"]["prereg04"]["t09_eligible"], [e9["id"]])
        self.assertNotIn("違反", out)
        self.assertIn("K6-09", out)

    def test_final_08_needs_not_09_reason(self):
        e8 = self.ready9()
        self.up_09()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e8["id"]), "--basis", "x",
                           "--not-09-reason", "PREREG_09 rev 4: bug")
        self.assertIn("rev 5 or later", msg)
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e8["id"]), "--basis", "x",
                           "--not-09-reason", "PREREG_09 rev 5: bug")
        self.assertIn("has no row", msg)
        self.rev5_09(at="2026-11-04T10:00:00+08:00")       # before 09 was prepared
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e8["id"]), "--basis", "x",
                           "--not-09-reason", "PREREG_09 rev 5: bug")
        self.assertIn("not after the records", msg)
        self.rev5_09()
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e8["id"]), "--basis", "x",
                              "--not-09-reason", "PREREG_09 rev 5: bug in t09_rank apply")
        self.assertEqual(fc["prereg09_5_3"]["chosen"], "08")
        self.assertEqual(fc["prereg09_5_3"]["revision"]["rev"], 5)
        self.assertEqual(fc["prereg08_5_4"]["chosen"], "08")
        st, _ = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertFalse(st["rank"]["prereg09"]["violation"])
        self.assertFalse(st["rank"]["prereg08"]["violation"])

    def test_final_07_needs_08_and_09_reasons(self):
        self.ready9()
        self.up_09()
        self.rev5_09()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", "3", "--basis", "x",
                           "--not-09-reason", "PREREG_09 rev 5: bug")
        self.assertIn("--not-08-reason", msg)

    def test_09_no_longer_eligible(self):
        self.ready9()
        e9, _ = self.up_09()
        with open(self.r("results/test09/gpu_seam.json"), "ab") as f:
            f.write(b" ")
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "x")
        self.assertIn("does not meet PREREG_09 5.3 now", msg)
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", "4", "--basis", "PREREG_08 5.4")
        self.assertEqual(fc["prereg09_5_3"]["eligible_09_entries"], [])
        self.assertEqual(fc["prereg09_5_3"]["chosen"], "08")
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertTrue(st["rank"]["prereg09"]["redo_08"])
        self.assertIn("現在沒有符合的 09", out)

    def test_status_hint_and_violation(self):
        e8 = self.ready9()
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", str(e8["id"]), "--basis", "PREREG_08 5.4")
        self.assertNotIn("prereg09_5_3", fc)
        e9, _ = self.up_09()
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertTrue(st["rank"]["prereg09"]["violation"])
        self.assertIn("違反 PREREG_09 5.3", out)
        self.run_ol("final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "PREREG_09 5.3")
        log = self.log_obj()
        log["final_choice"]["rank"] = None
        write_json(self.log, log)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertIn("最終選擇要設成 09", out)
        for k in ("08", "07", "04"):
            self.assertNotIn(f"最終選擇要設成 {k}", out)

    def test_prepare_08_after_09(self):
        self.ready9()
        self.up_09()
        self.f08 = t8o.write_t08_tree(self.repo, self.rank_rows)   # the same 08 run, prepared again
        msg = self.refused(self.prepare08)
        self.assertIn("needs --not-09-reason", msg)
        self.rev5_09()
        e, out = self.prepare08("--not-09-reason", "PREREG_09 rev 5: bug")
        self.assertEqual(e["prereg09_5_3"]["revision"]["rev"], 5)
        self.assertIn("這一份不是 09", out)


class T09Public(T09Base):
    def test_same_score_warning(self):
        e8 = self.ready9()
        self.run_ol("public", "--id", str(e8["id"]), "--score", "0.70")
        e9, _ = self.up_09()
        _e, out = self.run_ol("public", "--id", str(e9["id"]), "--score", "0.70")
        self.assertIn("WARNING (PREREG_09 5.3)", out)
        self.assertIn("（08）", out)
        for w in ("WARNING (PREREG_08 5.4)", "WARNING (PREREG_07 5.4)", "WARNING (PREREG_04 rev 3)"):
            self.assertNotIn(w, out)

    def test_08_07_04_warnings_ignore_09(self):
        self.ready9()
        e9, _ = self.up_09()
        self.run_ol("public", "--id", str(e9["id"]), "--score", "0.62")
        _e, out = self.run_ol("public", "--id", "2", "--score", "0.62")           # 04
        self.assertNotIn("WARNING (PREREG_04 rev 3)", out)
        self.assertNotIn("WARNING (PREREG_07 5.4)", out)
        self.assertNotIn("WARNING (PREREG_08 5.4)", out)
        self.assertIn("WARNING (PREREG_09 5.3)", out)
        _e, out = self.run_ol("public", "--id", "4", "--score", "0.61")           # 08, another score
        self.assertNotIn("WARNING", out)


if __name__ == "__main__":
    unittest.main()
