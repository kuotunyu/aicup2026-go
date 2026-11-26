"""Tests for the PREREG_08 rev 3 additions to tools/official_log.py (PREREG_08 5.2, 5.4).

* TagVersusNew: the version of tag p07-testday-pre08 (R20, 04, 06 and 07 additions, no 08) and the working-tree
  version run the same flows with the same mocks (test_t07_official_log.TagVersusNew's R20, R10, 04 and 06 flows, now
  with a results/test08 tree on disk) and a 07 flow (R20 -> 04 -> 07 uploaded, public scores, final 07, the refused
  04 prepare, status); the log bytes, returned records, stdout and refusals are identical (a log without 08 entries).
* T08*: the 08 path (prepare and its order rules, the refusals recomputed from the report's numbers, --method, final
  under PREREG_08 5.4 with --not-08-reason, the 07 and 04 rules skipping 08, status, the same-file warning).
Synthetic files in a temporary directory only; git and the clock are mocked. Nothing here opens a label file or runs
tools/score.py. Stdlib only besides the module under test.
Run: python -m pytest tests/test_t08_official_log.py
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

TAG = "p07-testday-pre08"                          # the 07 test-day path before the 08 kit
H8R_FILE = os.path.join(REPO, "results", "prereg08", "rank", "score_H8R.json")
P1_TIME = "2026-11-07T08:00:00+00:00"              # after R20's first upload (2026-11-05 14:15 Taiwan time)
PREREG08_HEAD = ("# PREREG_08\n\n| rev 1 | `543315a` | first |\n| rev 2 | `13a6db9` | second |\n"
                 "| rev 3 | (commit) | test day |\n")
DIFF = {"07": {"public": 41, "all": 90}, "04": {"public": 70, "all": 140}, "R20": {"public": 120, "all": 230}}
sha, write_json, quiet = t4o.sha, t4o.write_json, t4o.quiet
NEW = ol


# ---------------------------------------------------------------- the results/test08 tree of a usable 08 run
def write_t08_tree(repo, rank_rows, report_over=None):
    """results/test08 as tools/t08_rank.py apply leaves it (real mode) plus the committed inputs it binds. The 07
    report it binds is the one in results/test07/rank (a placeholder is written when there is none). -> 08 file."""
    r = lambda rel: os.path.join(repo, *rel.split("/"))   # noqa: E731
    os.makedirs(os.path.dirname(r(NEW.T08_H8R)), exist_ok=True)
    shutil.copyfile(H8R_FILE, r(NEW.T08_H8R))
    for rel in NEW.T08_INPUTS_REQUIRED:
        if not os.path.exists(r(rel)):
            if rel.endswith(".json"):
                write_json(r(rel), {"passed": True, "mode": "real", "fake_input": rel})
            elif rel.endswith(".csv"):
                ti.write_csv(r(rel), ["question_id", "pred_rank"], [[q, "D"] for q, _ in rank_rows])
            else:
                os.makedirs(os.path.dirname(r(rel)), exist_ok=True)
                with open(r(rel), "wb") as f:
                    f.write(b"PK-fake-npz " + rel.encode() + b"\r\n")
    rows = [[q, "A" if i % 3 == 0 else lab] for i, (q, lab) in enumerate(rank_rows)]
    f08 = ti.write_csv(r("results/test08/rank/submission_rank.csv"), ["question_id", "pred_rank"], rows)
    rep = report08(repo, f08)
    rep.update(report_over or {})
    write_json(r(f"results/test08/rank/{NEW.T08_REPORT}"), rep)
    return f08


def report08(repo, f08):
    bound = {rel: NEW.sha256_bound(os.path.join(repo, *rel.split("/"))) for rel in NEW.T08_INPUTS_REQUIRED}
    return {
        "format": NEW.T08_FORMAT, "mode": "real", "created": "2026-11-07T12:00:00+08:00", "exit_code": 0,
        "usable": True, "unusable_reasons": [],
        "combo": {"id": "Q08", "sources": list(NEW.T08_COMBO_SOURCES),
                  "terms": ["OP", "O4", "SWA", "SWB", "HS", "P1TO"]},
        "p1_sets": {s: {"fake_from_val": False, "earliest_chunk_created_utc": P1_TIME} for s in NEW.T08_P1_SETS},
        "earliest_p1_chunk_created_utc": P1_TIME,
        "gpu_seam": {"passed": True, "counts_equal": True, "max_abs_std_feature": 0.011, "max_abs_lp_p1": 0.0021},
        "q02h_vs_07": {"equal": True, "computed_sha256": bound[NEW.T08_07_PROBS]},
        "move_cross_check": {s: {"mismatches": 0, "vs_t07_hs": {"compared": 4800, "mismatches": 0},
                                 "vs_t04_o4": {"compared": 4800, "mismatches": 0}} for s in NEW.T08_SETS},
        "seam_local": {"verdict": "passed", "passed": True, "max_abs_diff": 0.0, "tol": 1e-4, "rule_b_differences": 0,
                       "byte_identical_probs": True, "byte_identical_pred_B": True, "required_byte_identical": True},
        "p1_failures": {s: {"failed_or_no_move": 0, "games": 5000, "ok": True} for s in NEW.T08_SETS},
        "p1_direction": {s: {"rho": 0.96, "threshold": 0.4836508533888232, "passed": True} for s in NEW.T08_SETS},
        "rule_b_differences": copy.deepcopy(DIFF),
        "check_submission": {"mode": "official", "questions": [], "errors": []},
        "outputs_sha256": {"submission_rank.csv": sha(f08)}, "inputs_bound": bound}


# ---------------------------------------------------------------- the tag version against the new one
class TagVersusNew(t7o.TagVersusNew):
    """A log without 08 entries gives the tag p07-testday-pre08 version's log bytes, records, stdout and refusals,
    with a results/test08 tree on disk (the R20 / R10 / 04 / 06 flows of test_t07_official_log)."""

    @classmethod
    def setUpClass(cls):
        cls.tag = t7o.load_version(TAG)
        assert hasattr(cls.tag, "check_reports_t07") and not hasattr(cls.tag, "check_reports_t08"), "not the tag"

    def build(self, kind, with_t06):
        out = super().build(kind, with_t06)
        write_t08_tree(out[0], [[q, "D"] for q in t6o.IDS])
        return out


class TagVersusNew07Flow(t7o.T07Base):
    """The 07 flow of test_t07_official_log (R20 -> 04 -> 07, public, final 07, a refused 04 prepare, status) gives
    the same results in the tag version and the new one, with a results/test08 tree on disk."""

    def setUp(self):
        self.tag = t7o.load_version(TAG)

    def tearDown(self):
        pass

    def run_with(self, mod):
        patches = [mock.patch.object(m, "ol", mod) for m in (ti, t4o, t7o)]
        for pt in patches:
            pt.start()
        try:
            t7o.T07Base.setUp(self)
            write_t08_tree(self.repo, self.rank_rows)
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
            return out
        finally:
            t7o.T07Base.tearDown(self)
            for pt in reversed(patches):
                pt.stop()

    def test_07_flow_identical(self):
        old, new = self.run_with(self.tag), self.run_with(ol)
        self.assertEqual([s[0] for s in old], [s[0] for s in new])
        for (name, o_out, o_res, o_log), (_n, n_out, n_res, n_log) in zip(old, new):
            with self.subTest(step=name):
                self.assertEqual(n_log, o_log)
                self.assertEqual(n_out, o_out)
                self.assertEqual(n_res, o_res)
        log = json.loads(old[-1][3])
        self.assertTrue(any("test07" in e for e in log["uploads"]), "the flow must hold a 07 entry")
        self.assertFalse(any("test08" in e or "prereg08_5_4" in e for e in log["uploads"]))
        self.assertEqual(json.loads(old[9][2])["prereg07_5_4"]["chosen"], "07")


# ---------------------------------------------------------------- the 08 path
class T08Base(t7o.T07Base):
    def setUp(self):
        super().setUp()
        self.p08_head = PREREG08_HEAD
        self.patches08 = [mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p08_head
                                            if rel == ol.P08_DOC else self.p07_head if rel == ol.P07_DOC
                                            else self.p04_head)]
        for pt in self.patches08:
            pt.start()
        self.f08 = write_t08_tree(self.repo, self.rank_rows)

    def tearDown(self):
        for pt in reversed(self.patches08):
            pt.stop()
        super().tearDown()

    def edit08(self, fn):
        p = self.r(f"results/test08/rank/{ol.T08_REPORT}")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def prepare08(self, *extra):
        return self.prepare("rank", self.f08, "prereg08 Q08 (PREREG_08 5.4)", *extra)

    def ready8(self):
        """R20 #1, 04 #2 and 07 #3 uploaded."""
        self.ready()
        e7, _ = self.up_07()
        return e7

    def up_08(self, at="2026-11-07T13:00"):
        e8, out = self.prepare08()
        self.uploaded(e8["id"], at)
        return e8, out

    def rev4(self, at="2026-11-08T10:00:00+08:00"):
        self.p08_head = PREREG08_HEAD + "| rev 4 | `abc` | bug |\n"
        self.rev_added["| rev 4 |"] = ("8" * 40, at)


class T08Prepare(T08Base):
    def test_order_r20_04_07_then_08(self):
        msg = self.refused(self.prepare08)
        self.assertIn("R20's first upload", msg)
        self.ready()
        msg = self.refused(self.prepare08)                 # 04 uploaded, 07 not
        self.assertIn("not logged as uploaded", msg)
        e7, _ = self.prepare07()
        msg = self.refused(self.prepare08)                 # 07 prepared, not uploaded
        self.assertIn("not logged as uploaded", msg)
        self.uploaded(e7["id"], "2026-11-06T13:00")
        e8, out = self.prepare08()
        self.assertEqual((e8["id"], e8["file"]), (4, "results/test08/rank/submission_rank.csv"))
        self.assertEqual(e8["method"], ol.T08_METHOD)
        t8 = e8["test08"]
        self.assertEqual((t8["r20_first_upload_id"], t8["t07_entry_id"], t8["t04_entry_id"], t8["combo"]),
                         (1, 3, 2, "Q08"))
        self.assertEqual(t8["public_rule_b_differences"], {"07": 41, "04": 70, "R20": 120})
        self.assertEqual(t8["k6_08"]["deadline"], "2026-11-19T23:59:59+08:00")
        for k in ("test07", "test04", "prereg04_5_3", "prereg07_5_4", "prereg08_5_4"):
            self.assertNotIn(k, e8)
        for s in ("這一份是 08", "Q08", "#3", "K6-08"):
            self.assertIn(s, out)

    def test_p1_before_r20_upload_refused(self):
        self.ready8()
        self.edit08(lambda r: r.update(earliest_p1_chunk_created_utc="2026-11-05T06:00:00+00:00"))
        self.assertIn("not after R20's first upload", self.refused(self.prepare08))

    def test_method(self):
        self.ready8()
        self.assertIn("does not name Q08", self.refused(self.prepare08, "--method", "prereg07 Q02H"))
        e, _ = self.prepare08("--method", "Q08 rule B")
        self.assertEqual(e["method"], "Q08 rule B")

    def test_other_reasons_on_an_08_file_refused(self):
        self.ready8()
        self.assertIn("not to an 08 file", self.refused(self.prepare08, "--not-04-reason", "PREREG_04 rev 4: x"))
        self.assertIn("04 / R20 / R10 file only", self.refused(self.prepare08, "--not-07-reason", "PREREG_07 rev 5: x"))
        self.assertIn("07 / 04 / R20 / R10 file only", self.refused(self.prepare08, "--not-08-reason",
                                                                    "PREREG_08 rev 4: x"))

    def test_conditions_recomputed(self):
        self.ready8()
        cases = [
            ("gpu feature", lambda r: r["gpu_seam"].update(max_abs_std_feature=0.051), "largest std_feature"),
            ("gpu lp", lambda r: r["gpu_seam"].update(max_abs_lp_p1=0.041), "largest lp_p1"),
            ("gpu counts", lambda r: r["gpu_seam"].update(counts_equal=False), "GPU seam: failed"),
            ("q02h", lambda r: r["q02h_vs_07"].update(computed_sha256="0" * 64), "Q02H recomputed"),
            ("cross", lambda r: r["move_cross_check"]["private"].update(mismatches=1), "move cross-check"),
            ("seam bytes", lambda r: r["seam_local"].update(byte_identical_pred_B=False), "local Q08 seam"),
            ("seam tol", lambda r: r["seam_local"].update(required_byte_identical=False, max_abs_diff=2e-4),
             "local Q08 seam"),
            ("seam b", lambda r: r["seam_local"].update(required_byte_identical=False, rule_b_differences=5),
             "local Q08 seam"),
            ("failures 26", lambda r: r["p1_failures"]["public"].update(failed_or_no_move=26), "P1 failures"),
            ("rho", lambda r: r["p1_direction"]["private"].update(rho=0.48), "P1 direction"),
            ("fake", lambda r: r["p1_sets"]["public"].update(fake_from_val=True), "drill records"),
            ("mode", lambda r: r.update(mode="dry-run"), "real-mode"),
            ("exit", lambda r: r.update(exit_code=3), "exit_code"),
            ("combo", lambda r: r["combo"].update(id="Q02H"), "not Q08"),
            ("check_submission", lambda r: r["check_submission"].update(errors=["x"]), "check_submission")]
        for name, fn, msg in cases:
            with self.subTest(name):
                self.f08 = write_t08_tree(self.repo, self.rank_rows)
                self.edit08(fn)
                self.assertIn(msg, self.refused(self.prepare08))
        self.f08 = write_t08_tree(self.repo, self.rank_rows)
        self.edit08(lambda r: r["p1_failures"]["public"].update(failed_or_no_move=25))
        e, _ = self.prepare08()                            # 25 of 5,000 passes
        self.assertEqual(e["test08"]["p1_failure_rate"]["public"], 0.005)

    def test_h8r_line(self):
        self.ready8()
        p = self.r(ol.T08_H8R)
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        s["B_minus_A_rank_balanced"]["ci95"][0] = -0.001
        write_json(p, s)
        self.assertIn("is not the committed H8R record", self.refused(self.prepare08))
        os.remove(p)
        self.assertIn("H8R was not scored", self.refused(self.prepare08))

    def test_h8r_problems_numbers(self):
        p = self.r(ol.T08_H8R)
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        self.assertEqual(ol.t08_h8r_problems(), [])
        with mock.patch.object(ol, "T08_H8R_SHA", None), mock.patch.object(ol, "sha256_bound", lambda path: None):
            for k, v, msg in (("A_rank_balanced", s["A_rank_balanced"] + 2e-12, "reference re-read"),
                              ("B_rank_balanced", 0.5699, "< 0.57")):
                with self.subTest(k):
                    write_json(p, dict(s, **{k: v}))
                    self.assertTrue(any(msg in x for x in ol.t08_h8r_problems()))
            s2 = copy.deepcopy(s)
            s2["B_minus_A_rank_balanced"]["ci95"][0] = 0.0
            write_json(p, s2)
            self.assertTrue(any("lower bound" in x for x in ol.t08_h8r_problems()))

    def test_bound_input_changed_and_uncommitted(self):
        self.ready8()
        with open(self.r("results/test08/p1/public_p1.npz"), "ab") as f:
            f.write(b"x")
        self.assertIn("results/test08/p1/public_p1.npz is", self.refused(self.prepare08))
        self.f08 = write_t08_tree(self.repo, self.rank_rows)
        self.dirty.add("results/test08/p1/public_p1.npz")
        self.assertIn("not committed and clean", self.refused(self.prepare08))
        self.dirty.discard("results/test08/p1/public_p1.npz")
        self.dirty.add("results/test08/p1/gpu_seam.json")
        self.assertIn("gpu_seam.json must be committed", self.refused(self.prepare08))

    def test_k6_08(self):
        self.ready8()
        self.commit = ("f" * 40, "2026-11-20T00:00:00+08:00")
        self.assertIn("not by K6-08", self.refused(self.prepare08))

    def test_t1_fail_opening(self):
        self.ready8()
        self.write_reports(t1=("FAIL_OPENING", "PASS"))
        self.assertIn("PASS or NORMALIZED", self.refused(self.prepare08))

    def test_bound_07_not_eligible(self):
        self.ready8()
        self.edit07(lambda r: r["hs_direction"]["public"].update(rho=0.1))
        msg = self.refused(self.prepare08)                 # the 08 report bound the earlier 07 report
        self.assertIn("results/test07/rank/t07_apply_report.json is", msg)
        self.f08 = write_t08_tree(self.repo, self.rank_rows)  # an 08 run on the changed 07
        msg = self.refused(self.prepare08)
        self.assertIn("no longer meets PREREG_07 5.4", msg)


class T08Final(T08Base):
    def test_final_08(self):
        e7 = self.ready8()
        e8, _ = self.up_08()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e7["id"]), "--basis", "x")
        self.assertIn("needs --not-08-reason", msg)
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e8["id"]), "--basis", "PREREG_08 5.4")
        self.assertEqual(fc["prereg08_5_4"]["chosen"], "08")
        self.assertNotIn("prereg04_5_3", fc)
        self.assertNotIn("prereg07_5_4", fc)
        self.assertIn("PREREG_08 5.4：最終選擇是 08", out)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertFalse(st["rank"]["prereg04"]["violation"])
        self.assertFalse(st["rank"]["prereg07"]["violation"])
        self.assertFalse(st["rank"]["prereg08"]["violation"])
        self.assertEqual(st["rank"]["prereg04"]["t08_eligible"], [e8["id"]])
        self.assertNotIn("違反", out)
        self.assertIn("K6-08", out)

    def test_final_07_needs_not_08_reason(self):
        e7 = self.ready8()
        e8, _ = self.up_08()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e7["id"]), "--basis", "x",
                           "--not-08-reason", "PREREG_08 rev 3: bug")
        self.assertIn("rev 4 or later", msg)
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e7["id"]), "--basis", "x",
                           "--not-08-reason", "PREREG_08 rev 4: bug")
        self.assertIn("has no row", msg)
        self.rev4(at="2026-11-04T10:00:00+08:00")          # before 08 was prepared
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e7["id"]), "--basis", "x",
                           "--not-08-reason", "PREREG_08 rev 4: bug")
        self.assertIn("not after the records", msg)
        self.rev4()
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e7["id"]), "--basis", "x",
                              "--not-08-reason", "PREREG_08 rev 4: bug in t08_rank apply")
        self.assertEqual(fc["prereg08_5_4"]["chosen"], "07")
        self.assertEqual(fc["prereg08_5_4"]["revision"]["rev"], 4)
        self.assertEqual(fc["prereg07_5_4"]["chosen"], "07")
        st, _ = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertFalse(st["rank"]["prereg08"]["violation"])
        self.assertFalse(st["rank"]["prereg07"]["violation"])

    def test_final_04_needs_07_and_08_reasons(self):
        self.ready8()
        self.up_08()
        self.rev4()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", "2", "--basis", "x",
                           "--not-08-reason", "PREREG_08 rev 4: bug")
        self.assertIn("--not-07-reason", msg)

    def test_08_no_longer_eligible(self):
        self.ready8()
        e8, _ = self.up_08()
        with open(self.r("results/test08/p1/seam_p1.npz"), "ab") as f:
            f.write(b"x")
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e8["id"]), "--basis", "x")
        self.assertIn("does not meet PREREG_08 5.4 now", msg)
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", "3", "--basis", "PREREG_07 5.4")
        self.assertEqual(fc["prereg08_5_4"]["eligible_08_entries"], [])
        self.assertEqual(fc["prereg08_5_4"]["chosen"], "07")
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertTrue(st["rank"]["prereg08"]["redo_07"])
        self.assertIn("現在沒有符合的 08", out)

    def test_status_hint_and_violation(self):
        e7 = self.ready8()
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", str(e7["id"]), "--basis", "PREREG_07 5.4")
        self.assertNotIn("prereg08_5_4", fc)
        e8, _ = self.up_08()
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertTrue(st["rank"]["prereg08"]["violation"])
        self.assertIn("違反 PREREG_08 5.4", out)
        self.run_ol("final", "--comp", "rank", "--id", str(e8["id"]), "--basis", "PREREG_08 5.4")
        log = self.log_obj()
        log["final_choice"]["rank"] = None
        write_json(self.log, log)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertIn("最終選擇要設成 08", out)
        self.assertNotIn("最終選擇要設成 07", out)
        self.assertNotIn("最終選擇要設成 04", out)

    def test_prepare_07_after_08(self):
        self.ready8()
        self.up_08()
        self.assertIn("needs --not-08-reason", self.refused(self.prepare07))
        self.rev4()
        e, out = self.prepare07("--not-08-reason", "PREREG_08 rev 4: bug")
        self.assertEqual(e["prereg08_5_4"]["revision"]["rev"], 4)
        self.assertIn("這一份不是 08", out)


class T08Public(T08Base):
    def test_same_score_warning(self):
        e7 = self.ready8()
        self.run_ol("public", "--id", str(e7["id"]), "--score", "0.66")
        e8, _ = self.up_08()
        _e, out = self.run_ol("public", "--id", str(e8["id"]), "--score", "0.66")
        self.assertIn("WARNING (PREREG_08 5.4)", out)
        self.assertIn("（07）", out)
        self.assertNotIn("WARNING (PREREG_07 5.4)", out)
        self.assertNotIn("WARNING (PREREG_04 rev 3)", out)

    def test_07_and_04_warnings_ignore_08(self):
        self.ready8()
        e8, _ = self.up_08()
        self.run_ol("public", "--id", str(e8["id"]), "--score", "0.62")
        _e, out = self.run_ol("public", "--id", "2", "--score", "0.62")           # 04
        self.assertNotIn("WARNING (PREREG_04 rev 3)", out)
        self.assertIn("WARNING (PREREG_08 5.4)", out)
        _e, out = self.run_ol("public", "--id", "3", "--score", "0.61")           # 07, another score
        self.assertNotIn("WARNING", out)


if __name__ == "__main__":
    unittest.main()
