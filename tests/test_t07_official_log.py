"""Tests for the PREREG_07 rev 4 additions to tools/official_log.py (results/prereg07/design/spec.md 8, 9).

* TagVersusNew (spec 8, "不變的保證"): the version of the branch point (BASE, main at 2d81c80: the R20, 04 and 06
  additions, no 07) and the working-tree version run the same flows with the same mocks: competition 2's R20, R10 and
  04 flows, each with competition 1's Q04 flow, with and without a results/test06 tree and with and without a
  results/test07 tree on disk, plus a flow with 06-OFF entries; the log bytes, returned records, stdout and refusals
  are identical (a log without 07 entries). TagVersusNewP04: test_t06_official_log's tag p04-testday-pre06 comparison
  run again with a results/test07 tree on disk.
* T07*: the 07 path (prepare and its order rules, the refusals recomputed from the report's numbers, --method, final
  under PREREG_07 5.4 with --not-07-reason, the 04 rule skipping 07, status, the same-file warning).
Synthetic files in a temporary directory only; git and the clock are mocked. Nothing here opens a label file or runs
tools/score.py. Stdlib only besides the module under test.
Run: python -m pytest tests/test_t07_official_log.py
"""

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import official_log as ol  # noqa: E402
import t02_test_bundle as tb  # noqa: E402
import test_t02_infra as ti  # noqa: E402
import test_t04_official_log as t4o  # noqa: E402
import test_t06_official_log as t6o  # noqa: E402

BASE = "2d81c80"                                   # main at the branch point of p07-testday (06 merged, no 07)
H7R_FILE = os.path.join(REPO, "results", "prereg07", "rank", "score_H7R.json")
NOW = t4o.NOW
HS_TIME = "2026-11-05T08:00:00+00:00"              # 16:00 Taiwan time: after R20's first upload (14:15)
PREREG07_HEAD = ("# PREREG_07\n\n| rev 1 | `5957a01` | first |\n| rev 2 | `9a0f6b8` | second |\n"
                 "| rev 3 | `a7fb0ca` | adoption line |\n| rev 4 | (commit) | test day |\n")
DIFF = {"04": {"public": 57, "all": 103}, "R20": {"public": 108, "all": 214}}
sha, write_json, quiet = t4o.sha, t4o.write_json, t4o.quiet


def load_version(ref):
    r = subprocess.run(["git", "-C", REPO, "show", f"{ref}:tools/official_log.py"], capture_output=True, timeout=60)
    if r.returncode != 0:
        raise AssertionError(f"{ref} without tools/official_log.py")
    spec = importlib.util.spec_from_loader(f"official_log_{ref}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.join(REPO, "tools", "official_log.py")
    exec(compile(r.stdout.decode("utf-8"), f"<{ref}:tools/official_log.py>", "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------- the results/test07 tree of a usable 07 run
def write_t07_tree(repo, rank_rows, report_over=None):
    """results/test07 as tools/t07_rank.py apply leaves it (real mode) plus the committed inputs it binds. The 04
    report it binds is the one in results/test04/rank (a placeholder is written when there is none). -> 07 file."""
    r = lambda rel: os.path.join(repo, *rel.split("/"))   # noqa: E731
    os.makedirs(os.path.dirname(r(ol.T07_H7R)), exist_ok=True)
    shutil.copyfile(H7R_FILE, r(ol.T07_H7R))
    if not os.path.exists(r(ol.T07_04_REPORT)):
        write_json(r(ol.T07_04_REPORT), {"format": "placeholder"})
    if not os.path.exists(r(ol.T07_04_PROBS)):
        ti.write_csv(r(ol.T07_04_PROBS), ["question_id", "n_games", "n_parsed"], [[q, 5, 5] for q, _ in rank_rows])
    for rel in ol.T07_INPUTS_REQUIRED:
        if not os.path.exists(r(rel)):
            if rel.endswith(".json"):
                write_json(r(rel), {"passed": True, "mode": "real", "fake_input": rel})
            else:
                os.makedirs(os.path.dirname(r(rel)), exist_ok=True)
                with open(r(rel), "wb") as f:
                    f.write(b"PK-fake-npz " + rel.encode() + b"\r\n")
    rows = [[q, "B" if i % 5 == 0 else lab] for i, (q, lab) in enumerate(rank_rows)]
    f07 = ti.write_csv(r("results/test07/rank/submission_rank.csv"), ["question_id", "pred_rank"], rows)
    rep = report07(repo, f07)
    rep.update(report_over or {})
    write_json(r(f"results/test07/rank/{ol.T07_REPORT}"), rep)
    return f07


def report07(repo, f07):
    bound = {rel: ol.sha256_bound(os.path.join(repo, *rel.split("/"))) for rel in ol.T07_INPUTS_REQUIRED}
    return {
        "format": ol.T07_FORMAT, "mode": "real", "created": "2026-11-06T12:00:00+08:00", "exit_code": 0,
        "usable": True, "unusable_reasons": [],
        "combo": {"id": "Q02H", "sources": list(ol.T07_COMBO_SOURCES), "terms": ["OP", "O4", "SWA", "SWB", "HS"]},
        "hs_sets": {s: {"fake_from_val": False, "earliest_chunk_created_utc": HS_TIME} for s in ol.T07_HS_SETS},
        "earliest_hs_chunk_created_utc": HS_TIME,
        "gpu_seam": {"passed": True, "counts_equal": True, "max_abs_move": 0.0071, "max_abs_profile_mean": 0.00019,
                     "max_abs_lp_hs": 0.0012},
        "q02_vs_04": {"equal": True, "computed_sha256": bound[ol.T07_04_PROBS]},
        "move_cross_check": {s: {"compared": 4800, "mismatches": 0} for s in ol.T07_SETS},
        "seam_local": {"verdict": "passed", "passed": True, "max_abs_diff": 0.0, "tol": 1e-4, "rule_b_differences": 0,
                       "byte_identical_probs": True, "byte_identical_pred_B": True, "required_byte_identical": True},
        "hs_failures": {s: {"failed_or_no_move": 0, "games": 5000, "ok": True} for s in ol.T07_SETS},
        "hs_direction": {s: {"rho": 0.94, "threshold": 0.4751284462649558, "passed": True} for s in ol.T07_SETS},
        "rule_b_differences": copy.deepcopy(DIFF),
        "check_submission": {"mode": "official", "questions": [], "errors": []},
        "outputs_sha256": {"submission_rank.csv": sha(f07)}, "inputs_bound": bound}


# ---------------------------------------------------------------- the base version against the new one
class TagVersusNew(t6o.TagVersusNew):
    """Spec 8: a log without 07 entries gives the base version's (2d81c80) log bytes, records, stdout and refusals,
    with and without results/test06 and results/test07 trees on disk."""
    with_t07 = False

    @classmethod
    def setUpClass(cls):
        cls.tag = load_version(BASE)
        assert hasattr(cls.tag, "check_reports_t06") and not hasattr(cls.tag, "check_reports_t07"), "not the base"

    def build(self, kind, with_t06):
        out = super().build(kind, with_t06)
        if self.with_t07:
            repo = out[0]
            write_t07_tree(repo, [[q, "D"] for q in t6o.IDS])
        return out

    def test_flows_identical_with_a_t07_tree(self):
        self.with_t07 = True
        try:
            for kind in ("r20", "r10", "04"):
                for with_t06 in (False, True):
                    old = self.compare(kind, with_t06)
                    log = json.loads(old[-1][3])
                    self.assertFalse(any("test07" in e or "prereg07_5_4" in e for e in log["uploads"]))
                    self.assertNotIn("prereg07_5_4", json.dumps(log["final_choice"]))
        finally:
            self.with_t07 = False

    def flow06(self, mod):
        """R20 uploaded, Q04 uploaded with a public score, 06-OFF prepared / uploaded / public, final player, status:
        a log with 06 entries and no 07 entry."""
        repo, tests, rank_file, player_file, _f04 = self.build("r20", True)
        write_t07_tree(repo, [[q, "D"] for q in t6o.IDS])
        f06 = os.path.join(repo, "results", "test06", "player", "off", "submission_player.csv")
        log = os.path.join(repo, "results", "official_scores.json")
        state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
        patches = [mock.patch.object(mod, "REPO", repo),
                   mock.patch.object(mod, "git_file_state", side_effect=lambda rel: dict(state)),
                   mock.patch.object(mod, "now_iso", return_value=t6o.NOW),
                   mock.patch.object(mod, "git_commit_time", side_effect=lambda rel: t6o.REPORT06_COMMIT),
                   mock.patch.object(mod, "git_head_text", side_effect=lambda rel: t6o.PREREG06_HEAD),
                   mock.patch.object(mod, "git_row_added", side_effect=lambda rel, row: (None, None)),
                   mock.patch.object(mod, "t06_now", side_effect=lambda: t6o.CLOCK)]
        steps = [("prepare R20", ["prepare", "--comp", "rank", "--file", rank_file, "--reason", "U1", "--tests-dir",
                                  tests]),
                 ("uploaded R20", ["uploaded", "--id", "1", "--ok-at", "2026-11-04T15:00", "--uploaded-at",
                                   "2026-11-04T15:05"]),
                 ("prepare Q04", ["prepare", "--comp", "player", "--file", player_file, "--reason", "U1", "--tests-dir",
                                  tests]),
                 ("uploaded Q04", ["uploaded", "--id", "2", "--ok-at", "2026-11-04T16:00", "--uploaded-at",
                                   "2026-11-04T16:05"]),
                 ("public Q04", ["public", "--id", "2", "--score", "0.88"]),
                 ("prepare 06-OFF", ["prepare", "--comp", "player", "--file", f06, "--reason", "prereg06 06-OFF",
                                     "--tests-dir", tests]),
                 ("uploaded 06-OFF", ["uploaded", "--id", "3", "--ok-at", "2026-11-05T10:00", "--uploaded-at",
                                      "2026-11-05T10:00"]),
                 ("public 06-OFF", ["public", "--id", "3", "--score", "0.90"]),
                 ("final rank", ["final", "--comp", "rank", "--id", "1", "--basis", "prereg02"]),
                 ("final player Q04 refused", ["final", "--comp", "player", "--id", "2", "--basis", "U5"]),
                 ("final player 06-OFF", ["final", "--comp", "player", "--id", "3", "--basis", "PREREG_06 5.4"]),
                 ("status", ["status", "--now", "2026-11-10T00:00"])]
        out = []
        for pt in patches:
            pt.start()
        try:
            for name, argv in steps:
                with mock.patch("sys.stdout", new_callable=__import__("io").StringIO) as buf:
                    try:
                        res = mod.main(argv + ["--log", log])
                    except SystemExit as e:
                        res = ("refused", str(e))
                    text = buf.getvalue()
                with open(log, "rb") as f:
                    out.append((name, text, json.dumps(res, sort_keys=True, ensure_ascii=False), f.read()))
        finally:
            for pt in reversed(patches):
                pt.stop()
        return out

    def test_06_flow_identical(self):
        old, new = self.flow06(self.tag), self.flow06(ol)
        self.assertEqual([s[0] for s in old], [s[0] for s in new])
        for (name, o_out, o_res, o_log), (_n, n_out, n_res, n_log) in zip(old, new):
            with self.subTest(step=name):
                self.assertEqual(n_log, o_log)
                self.assertEqual(n_out, o_out)
                self.assertEqual(n_res, o_res)
        log = json.loads(old[-1][3])
        self.assertTrue(any("test06" in e for e in log["uploads"]), "the flow must hold a 06 entry")
        self.assertNotIn("prepare 06-OFF", [s[0] for s in old if s[2].startswith('["refused"')])


class TagVersusNewP04(t6o.TagVersusNew):
    """test_t06_official_log.TagVersusNew (tag p04-testday-pre06) again, with a results/test07 tree on disk."""

    def build(self, kind, with_t06):
        out = super().build(kind, with_t06)
        write_t07_tree(out[0], [[q, "D"] for q in t6o.IDS])
        return out


# ---------------------------------------------------------------- the 07 path
class T07Base(t4o.T04Tree, ti.Tmp):
    jw = ti.OfficialLog.jw
    write_reports = ti.OfficialLog.write_reports
    run_ol = ti.OfficialLog.run_ol
    prepare = ti.OfficialLog.prepare

    def setUp(self):
        ti.Tmp.setUp(self)
        self.repo = self.p("repo")
        self.tests = ti.write_official(os.path.join(self.repo, "data", "test"))
        files = {n: tb.file_record(os.path.join(self.tests, n)) for n in tb.ALL_NAMES}
        tb.write_manifest(self.tests, "official", files, {"rank": "PASS", "player": "PASS"})
        ids = [f"pub_q_{i:04d}" for i in range(1, 401)] + [f"priv_q_{i:04d}" for i in range(1, 401)]
        self.rank_rows = [[q, "D"] for q in ids]
        self.rank_file = ti.write_csv(os.path.join(self.repo, "results", "test02", "rank", "submission_rank.csv"),
                                      ["question_id", "pred_rank"], self.rank_rows)
        self.player_file = ti.write_csv(
            os.path.join(self.repo, "results", "test02", "player", "submission_player.csv"),
            ["question_id"] + [f"top_{i}" for i in range(1, 6)],
            [[q] + [f"{q.split('_')[0]}_p_{j:04d}" for j in range(5)] for q in ids])
        self.log = self.p("repo", "results", "official_scores.json")
        self.state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
        self.write_reports()
        self.patches = [mock.patch.object(ol, "REPO", self.repo)]
        for pt in self.patches:
            pt.start()
        self.t04_setup()
        self.p07_head = PREREG07_HEAD
        self.patches07 = [mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p07_head
                                            if rel == ol.P07_DOC else self.p04_head)]
        for pt in self.patches07:
            pt.start()
        self.f07 = write_t07_tree(self.repo, self.rank_rows)

    def tearDown(self):
        for pt in reversed(self.patches07):
            pt.stop()
        self.t04_teardown()
        for pt in reversed(self.patches):
            pt.stop()
        ti.Tmp.tearDown(self)

    def refused(self, fn, *a):
        with self.assertRaises(SystemExit) as cm:
            quiet(fn, *a)
        return str(cm.exception)

    def edit07(self, fn):
        p = self.r(f"results/test07/rank/{ol.T07_REPORT}")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def prepare07(self, *extra):
        return self.prepare("rank", self.f07, "prereg07 Q02H (PREREG_07 5.4)", *extra)

    def up_04(self):
        e4, _ = self.prepare04()
        self.uploaded(e4["id"], "2026-11-05T15:10")
        return e4

    def ready(self):
        """R20 #1 and 04 #2 uploaded."""
        r20 = self.r20_uploaded()
        e4 = self.up_04()
        return r20, e4

    def up_07(self, at="2026-11-06T13:00"):
        e7, out = self.prepare07()
        self.uploaded(e7["id"], at)
        return e7, out


class T07Prepare(T07Base):
    def test_order_r20_then_04_then_07(self):
        msg = self.refused(self.prepare07)
        self.assertIn("R20's first upload", msg)
        self.r20_uploaded()
        msg = self.refused(self.prepare07)
        self.assertIn("not logged as uploaded", msg)
        e4, _ = self.prepare04()
        msg = self.refused(self.prepare07)                 # 04 prepared, not uploaded
        self.assertIn("not logged as uploaded", msg)
        self.uploaded(e4["id"])
        e7, out = self.prepare07()
        self.assertEqual((e7["id"], e7["file"]), (3, "results/test07/rank/submission_rank.csv"))
        self.assertEqual(e7["method"], ol.T07_METHOD)
        t7 = e7["test07"]
        self.assertEqual((t7["r20_first_upload_id"], t7["t04_entry_id"], t7["combo"]), (1, 2, "Q02H"))
        self.assertEqual(t7["public_rule_b_differences"], {"04": 57, "R20": 108})
        self.assertEqual(t7["k6_07"]["deadline"], "2026-11-19T23:59:59+08:00")
        self.assertNotIn("test04", e7)
        self.assertNotIn("prereg04_5_3", e7)
        for s in ("這一份是 07", "Q02H", "#2", "K6-07"):
            self.assertIn(s, out)

    def test_hs_before_r20_upload_refused(self):
        self.ready()
        self.edit07(lambda r: r.update(earliest_hs_chunk_created_utc="2026-11-05T06:00:00+00:00"))
        self.assertIn("not after R20's first upload", self.refused(self.prepare07))

    def test_method(self):
        self.ready()
        self.assertIn("does not name Q02H", self.refused(self.prepare07, "--method", "prereg04 Q02"))
        self.assertIn("names R10", self.refused(self.prepare07, "--method", "prereg02 R10 Q02H"))
        e, _ = self.prepare07("--method", "Q02H rule B")
        self.assertEqual(e["method"], "Q02H rule B")

    def test_not_04_reason_on_a_07_file_refused(self):
        self.ready()
        self.assertIn("not to a 07 file", self.refused(self.prepare07, "--not-04-reason", "PREREG_04 rev 4: x"))
        self.assertIn("04 / R20 / R10 file only", self.refused(self.prepare07, "--not-07-reason", "PREREG_07 rev 5: x"))

    def test_conditions_recomputed(self):
        self.ready()
        cases = [
            ("gpu move", lambda r: r["gpu_seam"].update(max_abs_move=0.051), "largest move"),
            ("gpu mean", lambda r: r["gpu_seam"].update(max_abs_profile_mean=0.0021), "largest profile_mean"),
            ("gpu lp", lambda r: r["gpu_seam"].update(max_abs_lp_hs=0.021), "largest lp_hs"),
            ("gpu counts", lambda r: r["gpu_seam"].update(counts_equal=False), "GPU seam: failed"),
            ("q02", lambda r: r["q02_vs_04"].update(computed_sha256="0" * 64), "Q02 recomputed"),
            ("cross", lambda r: r["move_cross_check"]["private"].update(mismatches=1), "move cross-check"),
            ("seam bytes", lambda r: r["seam_local"].update(byte_identical_pred_B=False), "local Q02H seam"),
            ("seam tol", lambda r: r["seam_local"].update(required_byte_identical=False, max_abs_diff=2e-4),
             "local Q02H seam"),
            ("seam b", lambda r: r["seam_local"].update(required_byte_identical=False, rule_b_differences=5),
             "local Q02H seam"),
            ("failures 26", lambda r: r["hs_failures"]["public"].update(failed_or_no_move=26), "HS failures"),
            ("rho", lambda r: r["hs_direction"]["private"].update(rho=0.47), "HS direction"),
            ("fake", lambda r: r["hs_sets"]["public"].update(fake_from_val=True), "drill records"),
            ("mode", lambda r: r.update(mode="dry-run"), "real-mode"),
            ("exit", lambda r: r.update(exit_code=3), "exit_code"),
            ("combo", lambda r: r["combo"].update(id="Q02"), "not Q02H"),
            ("check_submission", lambda r: r["check_submission"].update(errors=["x"]), "check_submission")]
        for name, fn, msg in cases:
            with self.subTest(name):
                self.f07 = write_t07_tree(self.repo, self.rank_rows)
                self.edit07(fn)
                self.assertIn(msg, self.refused(self.prepare07))
        self.f07 = write_t07_tree(self.repo, self.rank_rows)
        self.edit07(lambda r: r["hs_failures"]["public"].update(failed_or_no_move=25))
        e, _ = self.prepare07()                            # 25 of 5,000 passes
        self.assertEqual(e["test07"]["hs_failure_rate"]["public"], 0.005)

    def test_h7r_line(self):
        self.ready()
        p = self.r(ol.T07_H7R)
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        s["B_rank_balanced"] = 0.5699
        write_json(p, s)
        self.assertIn("is not the committed H7R record", self.refused(self.prepare07))
        os.remove(p)
        self.assertIn("H7R was not scored", self.refused(self.prepare07))

    def test_bound_input_changed_and_uncommitted(self):
        self.ready()
        with open(self.r("results/test07/hs/public_hs.npz"), "ab") as f:
            f.write(b"x")
        self.assertIn("results/test07/hs/public_hs.npz is", self.refused(self.prepare07))
        self.f07 = write_t07_tree(self.repo, self.rank_rows)
        self.dirty.add("results/test07/hs/public_hs.npz")
        self.assertIn("not committed and clean", self.refused(self.prepare07))
        self.dirty.discard("results/test07/hs/public_hs.npz")
        self.dirty.add("results/test07/hs/gpu_seam.json")
        self.assertIn("gpu_seam.json must be committed", self.refused(self.prepare07))

    def test_k6_07(self):
        self.ready()
        self.commit = ("f" * 40, "2026-11-20T00:00:00+08:00")
        msg = self.refused(self.prepare07)
        self.assertIn("not by K6-07", msg)

    def test_t1_fail_opening(self):
        self.ready()
        self.write_reports(t1=("FAIL_OPENING", "PASS"))
        msg = self.refused(self.prepare07)
        self.assertIn("PASS or NORMALIZED", msg)

    def test_bound_04_not_eligible(self):
        self.ready()
        self.edit04(lambda r: r["colour_check"]["public"]["O4"].update(dkappa=-0.01))
        msg = self.refused(self.prepare07)                 # the 07 report bound the earlier 04 report
        self.assertIn("results/test04/rank/t04_apply_report.json is", msg)
        self.f07 = write_t07_tree(self.repo, self.rank_rows)  # a 07 run on the changed 04
        msg = self.refused(self.prepare07)
        self.assertIn("no longer meets PREREG_04 5.3", msg)


class T07Final(T07Base):
    def test_final_07(self):
        self.ready()
        e7, _ = self.up_07()
        self.assertIn("needs --not-04-reason",            # R20 while 04 and 07 hold: the 04 rule speaks first
                      self.refused(self.run_ol, "final", "--comp", "rank", "--id", "1", "--basis", "x"))
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e7["id"]), "--basis", "PREREG_07 5.4")
        self.assertEqual(fc["prereg07_5_4"]["chosen"], "07")
        self.assertNotIn("prereg04_5_3", fc)
        self.assertIn("PREREG_07 5.4：最終選擇是 07", out)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertFalse(st["rank"]["prereg04"]["violation"])
        self.assertEqual(st["rank"]["prereg04"]["t07_eligible"], [e7["id"]])
        self.assertFalse(st["rank"]["prereg07"]["violation"])
        self.assertNotIn("違反", out)
        self.assertIn("K6-07", out)

    def test_final_04_needs_not_07_reason(self):
        _r20, e4 = self.ready()
        e7, _ = self.up_07()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e4["id"]), "--basis", "x")
        self.assertIn("needs --not-07-reason", msg)
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e4["id"]), "--basis", "x",
                           "--not-07-reason", "PREREG_07 rev 4: bug")
        self.assertIn("rev 5 or later", msg)
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e4["id"]), "--basis", "x",
                           "--not-07-reason", "PREREG_07 rev 5: bug")
        self.assertIn("has no row", msg)
        self.p07_head = PREREG07_HEAD + "| rev 5 | `abc` | bug |\n"
        self.rev_added["| rev 5 |"] = ("9" * 40, "2026-11-04T10:00:00+08:00")    # before 07 was prepared
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e4["id"]), "--basis", "x",
                           "--not-07-reason", "PREREG_07 rev 5: bug")
        self.assertIn("not after the records", msg)
        self.rev_added["| rev 5 |"] = ("9" * 40, "2026-11-07T10:00:00+08:00")
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e4["id"]), "--basis", "x",
                              "--not-07-reason", "PREREG_07 rev 5: bug in t07_rank apply")
        self.assertEqual(fc["prereg07_5_4"]["chosen"], "04")
        self.assertEqual(fc["prereg07_5_4"]["revision"]["rev"], 5)
        self.assertEqual(fc["prereg04_5_3"]["chosen"], "04")
        st, _ = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertFalse(st["rank"]["prereg07"]["violation"])

    def test_final_r20_needs_both_reasons(self):
        self.ready()
        self.up_07()
        self.p07_head = PREREG07_HEAD + "| rev 5 | `abc` | bug |\n"
        self.rev_added["| rev 5 |"] = ("9" * 40, "2026-11-07T10:00:00+08:00")
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", "1", "--basis", "x",
                           "--not-07-reason", "PREREG_07 rev 5: bug")
        self.assertIn("--not-04-reason", msg)

    def test_07_no_longer_eligible(self):
        self.ready()
        e7, _ = self.up_07()
        with open(self.r("results/test07/hs/seam_hs.npz"), "ab") as f:
            f.write(b"x")
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e7["id"]), "--basis", "x")
        self.assertIn("does not meet PREREG_07 5.4 now", msg)
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", "2", "--basis", "PREREG_04 5.3")
        self.assertEqual(fc["prereg07_5_4"]["eligible_07_entries"], [])
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertTrue(st["rank"]["prereg07"]["redo_53"])
        self.assertIn("現在沒有符合的 07", out)

    def test_status_hint_and_violation(self):
        _r20, e4 = self.ready()
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", str(e4["id"]), "--basis", "PREREG_04 5.3")
        self.assertNotIn("prereg07_5_4", fc)
        e7, _ = self.up_07()
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertTrue(st["rank"]["prereg07"]["violation"])
        self.assertIn("違反 PREREG_07 5.4", out)
        self.run_ol("final", "--comp", "rank", "--id", str(e7["id"]), "--basis", "PREREG_07 5.4")
        log = self.log_obj()
        log["final_choice"]["rank"] = None
        write_json(self.log, log)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertIn("最終選擇要設成 07", out)
        self.assertNotIn("最終選擇要設成 04", out)

    def test_prepare_r20_or_04_after_07(self):
        self.ready()
        self.up_07()
        self.assertIn("needs --not-07-reason", self.refused(self.prepare04))
        self.p07_head = PREREG07_HEAD + "| rev 5 | `abc` | bug |\n"
        self.rev_added["| rev 5 |"] = ("9" * 40, "2026-11-07T10:00:00+08:00")
        e, out = self.prepare04("--not-07-reason", "PREREG_07 rev 5: bug")
        self.assertEqual(e["prereg07_5_4"]["revision"]["rev"], 5)
        self.assertIn("這一份不是 07", out)


class T07Public(T07Base):
    def test_same_score_warning(self):
        _r20, e4 = self.ready()
        self.run_ol("public", "--id", str(e4["id"]), "--score", "0.62")
        e7, _ = self.up_07()
        _e, out = self.run_ol("public", "--id", str(e7["id"]), "--score", "0.62")
        self.assertIn("WARNING (PREREG_07 5.4)", out)
        self.assertNotIn("WARNING (PREREG_04 rev 3)", out)

    def test_no_warning_when_public_rows_equal(self):
        _r20, e4 = self.ready()
        self.run_ol("public", "--id", str(e4["id"]), "--score", "0.62")
        DIFF["04"]["public"] = 0
        try:
            self.f07 = write_t07_tree(self.repo, self.rank_rows)
            e7, _ = self.up_07()
        finally:
            DIFF["04"]["public"] = 57
        _e, out = self.run_ol("public", "--id", str(e7["id"]), "--score", "0.62")
        self.assertNotIn("WARNING", out)

    def test_04_warning_ignores_07(self):
        self.ready()
        e7, _ = self.up_07()
        self.run_ol("public", "--id", str(e7["id"]), "--score", "0.62")
        _e, out = self.run_ol("public", "--id", "2", "--score", "0.62")
        self.assertNotIn("WARNING (PREREG_04 rev 3)", out)
        self.assertIn("WARNING (PREREG_07 5.4)", out)


if __name__ == "__main__":
    unittest.main()
