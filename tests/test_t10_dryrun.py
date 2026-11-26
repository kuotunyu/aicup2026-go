"""End-to-end dry runs of the 10 test-day kit (PREREG_10 4.1 step 2, 4.2; competition 1, part A).

    C10_SLOW=1  OfficialBaseline (about 6 min): the official-shape tree of tests/test_t06_dryrun.py OfficialShapeDrills
                (tools/t06_fake_tests.py official -> T1 structure -> test02 -> fake-t2 -> 06 feats / seam / apply /
                submit) built again with the same commands: Q04, 06-OFF and 06-ON must equal
                results/test10/design/baseline_pre10.json (competition1.official, raw SHA-256) byte for byte; then
                10: the seam reproduces 06 (and H6P), both pools are at guard level 1 so A never holds, each pool's
                10 rows are 06-OFF's byte for byte and submit is exit 3 ("10 not needed").
    C10_SLOW=1  Nov4Tree (about 5 min): tools/t10_fake_tests.py nov4 --white-question public (both pools nov4, all
                black but one white question game in public) -> T1 structure -> test02 (both pools at guard level 3
                by rho) -> submit -> T4 -> fake-t2 -> 06 (both pools on test02's rows: apply private and submit exit
                3) -> 10: public keeps test02's rows (b_q != 1: the drill "b_q != 1"), private is lifted (10A, the 06
                gate against test02's effective Q04 passed), submit writes the 10 file (reference Q04) whose report
                passes tools/official_log.py's report and input checks. Drills on copies: T1 FAIL_OPENING (3), the
                06 seam failed with other dumps (3), level 0 in private (06 applied again: test02's rows), the 06
                gate failed against R02' (in process: test02's rows), K6-10 passed (3 without a run, 2 with one),
                p1 = B not adopted (3, the B GPU seam drill while B is not adopted), reruns refused (2) once logged.
    C10_SLOW=1  MixTree (about 4 min): nov4 in private only: public at level 1 (06 used), private at level 3 (test02's
                rows in 06) -> 06-OFF is produced without 06-ON; 10 = public 06-OFF rows + private 10A, compared with
                06-OFF; its report passes tools/official_log.py's checks (06-OFF bound).
The trees live outside the repository (C10_DRYRUN_DIR, or a temporary directory). Before merging p10-testday the
earlier dry runs are run again as baseline_pre10.json says (tests/test_t06_dryrun.py, tests/test_t09_dryrun.py).
Nothing here opens a label file or runs tools/score.py; data/local is read only by the 06 / 10 seam (the two
validation game files, under the p06 guard).
"""

import copy
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
import test_t06_dryrun as t6d  # noqa: E402  (run_tool, workdir, t06_args, the 06 runner; its classes are not collected)

SLOW = os.environ.get("C10_SLOW", "")
BASELINE = os.path.join(REPO, "results", "test10", "design", "baseline_pre10.json")
run_tool, read_json, sha, write_obj, rows_of, AIDEA = (t6d.run_tool, t6d.read_json, t6d.sha, t6d.write_obj,
                                                       t6d.rows_of, t6d.AIDEA)


def t10_args(S, o10=None, o6=None, o2=None, checks=None, log=None, now=None):
    a = ["--dry-run", "--out-dir", o10 or os.path.join(S, "o10"), "--t06-out-dir", o6 or os.path.join(S, "o6"),
         "--t06-data-dir", os.path.join(S, "a6"), "--t02-results-dir", os.path.join(S, "results", "test02"),
         "--t02-out-dir", o2 or os.path.join(S, "out", "player"), "--tests-dir", os.path.join(S, "data", "test")]
    for k, v in (("--checks-dir", checks), ("--log", log), ("--now", now)):
        if v:
            a += [k, v]
    return a


def t10(S, cmd, *extra, expect=0, runlog=None, **kw):
    return run_tool(["tools/t10_player.py"] + cmd.split() + t10_args(S, **kw) + list(extra), expect=expect,
                    log=runlog)


def run_test02(S, log, t4=True):
    run_tool(["tools/t02_checks.py", "t1", "--comp", "player", "--dry-run", "--root", S, "--only", "T1.1,T1.2,T1.3"],
             log=log)
    mode = ["--dry-run", "--results-dir", os.path.join(S, "results", "test02"), "--tests-dir",
            os.path.join(S, "data", "test"), "--out-dir", os.path.join(S, "out", "player")]
    for pool in ("seam_val", "public", "private"):
        run_tool(["tools/t02_player.py", "run", "--pool", pool] + mode, log=log)
    run_tool(["tools/t02_player.py", "submit"] + mode, log=log)
    if t4:
        run_tool(["tools/t02_checks.py", "run", "--comp", "player", "--dry-run", "--root", S, "--only", "T4"], log=log)
    run_tool(["tools/t06_fake_tests.py", "fake-t2", "--root", S], log=log)


def run_06(S, log, expect_private=0, expect_submit=0):
    for pool in ("seam_val", "public", "private"):
        t6d.t06(S, f"feats --pool {pool}", runlog=log)
    t6d.t06(S, "seam", runlog=log)
    t6d.t06(S, "apply --pool public", runlog=log)
    t6d.t06(S, "apply --pool private", expect=expect_private, runlog=log)
    t6d.t06(S, "submit", "--aidea-page-checked", AIDEA, expect=expect_submit, runlog=log)


def run_10(S, log, expect_submit=0):
    t10(S, "seam", runlog=log)
    for pool in ("public", "private"):
        t10(S, f"apply --pool {pool}", runlog=log)
    return t10(S, "submit", expect=expect_submit, runlog=log)


def drill_o10(S, name):
    """A fresh 10 output folder holding the base run's 10 seam (apply runs again there)."""
    d = os.path.join(S, "drills", name, "o10")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(S, "o10", "seam_val"), os.path.join(d, "seam_val"))
    return d


def in_process(argv, patches):
    """tools/t10_player.py main() in this process with attributes replaced ((object, name, value) triples)."""
    import contextlib
    import io
    import t10_player as m
    out = io.StringIO()
    with contextlib.ExitStack() as st:
        for obj, name, value in patches(m):
            st.enter_context(mock.patch.object(obj, name, value))
        st.enter_context(contextlib.redirect_stdout(out))
        st.enter_context(contextlib.redirect_stderr(out))
        code = m.main(argv)
    return code, out.getvalue()


def official_log_problems(case, S, need_t4):
    """The 10 submit report checked by tools/official_log.py (t10_report_problems, t10_inputs_problems) after copying
    it with its inputs into a temporary repository under the real-mode paths (mode set to real, git mocked)."""
    import official_log as ol
    root = tempfile.mkdtemp(prefix="c10_t10_ol_")
    maps = [(os.path.join(S, "out", "player"), "results/test02/player"),
            (os.path.join(S, "results", "test02"), "results/test02"),
            (os.path.join(S, "o6"), "results/test06/player"), (os.path.join(S, "o10"), "results/test10/player")]

    def to_rel(k):
        if not os.path.isabs(k):
            return k
        for src, dst in maps:
            if t6d.inside(k, src):
                return dst + "/" + os.path.relpath(k, src).replace(os.sep, "/")
        raise AssertionError(f"inputs_bound key outside the tree: {k}")
    try:
        rep = read_json(os.path.join(S, "o10", "t10_submit_report.json"))
        bound = {}
        for k, want in rep["inputs_bound"].items():
            rel = to_rel(k)
            src = k if os.path.isabs(k) else os.path.join(REPO, *k.split("/"))
            dst = os.path.join(root, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            bound[rel] = want
        sub = os.path.join(root, "results", "test10", "player", "submission_player.csv")
        os.makedirs(os.path.dirname(sub), exist_ok=True)
        shutil.copyfile(os.path.join(S, "o10", "submission_player.csv"), sub)
        state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
        with mock.patch.object(ol, "REPO", root), mock.patch.object(ol, "git_file_state", return_value=state):
            probs = ol.t10_report_problems(dict(rep, mode="real", inputs_bound=bound),
                                           f"results/test10/player/{ol.T10_REPORT}", sub, sha(sub))
            probs += ol.t10_inputs_problems(bound, need_t4=need_t4)
        case.assertIn("results/test02/checks/player/t4.json", bound)
        return probs, bound
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ================================================================ the baseline tree: 10 = 06-OFF
@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (10 dry runs)")
class OfficialBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S, cls.cleanup = t6d.workdir("t10_official")
        cls.log = []
        S = cls.S
        run_tool(["tools/t06_fake_tests.py", "official", "--root", S], log=cls.log)
        run_test02(S, cls.log, t4=False)               # as OfficialShapeDrills.build (no T4 there)
        t6d.run_t06_all(S, cls.log, ("--aidea-page-checked", AIDEA))
        run_tool(["tools/t02_checks.py", "run", "--comp", "player", "--dry-run", "--root", S, "--only", "T4"],
                 log=cls.log)
        cls.code, cls.out = run_10(S, cls.log, expect_submit=3)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "cleanup", False) and not os.environ.get("C10_KEEP"):
            shutil.rmtree(cls.S, ignore_errors=True)

    def test_earlier_outputs_equal_baseline_pre10(self):
        want = read_json(BASELINE)["competition1"]["official"]
        got = {part: {rel: sha(os.path.join(self.S, *rel.split("/"))) for rel in files}
               for part, files in want.items() if part != "tree"}
        self.assertEqual(got, {k: v for k, v in want.items() if k != "tree"})

    def test_10_seam(self):
        seam = read_json(os.path.join(self.S, "o10", "seam_val", "t10_seam.json"))
        self.assertTrue(seam["passed"], seam["criteria"])
        self.assertTrue(seam["criteria"]["top5_byte_identical_to_h6p"])
        self.assertFalse(seam["a_rule"]["holds"])
        self.assertEqual(sha(os.path.join(self.S, "o10", "seam_val", "top5.csv")),
                         read_json(BASELINE)["golden"]["h6p_top5"]["sha256"])

    def test_level1_pools_keep_06_off(self):
        for pool in ("public", "private"):
            rep = read_json(os.path.join(self.S, "o10", pool, "t10_pool.json"))
            self.assertEqual(rep["source"], "06-OFF", pool)
            self.assertFalse(rep["a_holds"])
            self.assertFalse(rep["a_rule"]["conditions"]["level3_by_rho"]["ok"])
            self.assertTrue(rep["a_rule"]["conditions"]["all_black"]["ok"])
            self.assertEqual(sha(os.path.join(self.S, "o10", pool, "top5.csv")),
                             sha(os.path.join(self.S, "o6", pool, "top5_off.csv")))
        self.assertIn("10 不需要", self.out)
        self.assertFalse(os.path.exists(os.path.join(self.S, "o10", "submission_player.csv")))
        nu = read_json(os.path.join(self.S, "o10", "not_used.json"))
        self.assertEqual([e["reason"] for e in nu["events"]], ["10_not_needed"])


# ================================================================ both pools nov4 (public with one white game)
@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (10 dry runs)")
class Nov4Tree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S, cls.cleanup = t6d.workdir("t10_nov4")
        cls.log = []
        S = cls.S
        run_tool(["tools/t10_fake_tests.py", "nov4", "--root", S, "--white-question", "public"], log=cls.log)
        run_test02(S, cls.log)
        run_06(S, cls.log, expect_private=3, expect_submit=3)
        run_10(S, cls.log)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "cleanup", False) and not os.environ.get("C10_KEEP"):
            shutil.rmtree(cls.S, ignore_errors=True)

    def p(self, *parts):
        return os.path.join(self.S, *parts)

    def test_test02_level3_by_rho(self):
        for pool in ("public", "private"):
            ar = read_json(self.p("out", "player", pool, "apply_report.json"))
            self.assertEqual(ar["guard_level"], 3, pool)
            self.assertIn("< theta2", ar["guard_reason"])
            self.assertEqual(ar["applied"]["o"], None)
            self.assertEqual(ar["b_c_candidate_black_share"], 1.0)
        self.assertLess(read_json(self.p("out", "player", "public", "apply_report.json"))["b_q_question_black_share"], 1)
        self.assertEqual(read_json(self.p("out", "player", "private", "apply_report.json"))["b_q_question_black_share"], 1)
        t4 = read_json(self.p("results", "test02", "checks", "player", "t4.json"))["pools"]
        self.assertEqual((t4["public"]["black_only_pool"], t4["private"]["black_only_pool"]), (False, True))

    def test_06_falls_back_in_both_pools(self):
        for pool in ("public", "private"):
            self.assertFalse(read_json(self.p("o6", pool, "t06_pool.json"))["used06"])
        self.assertFalse(os.path.exists(self.p("o6", "off", "submission_player.csv")))
        reasons = [e["reason"] for e in read_json(self.p("o6", "not_used.json"))["events"]]
        self.assertEqual(reasons, ["both_pools_test02", "both_pools_test02"])

    def test_10(self):
        pub = read_json(self.p("o10", "public", "t10_pool.json"))
        priv = read_json(self.p("o10", "private", "t10_pool.json"))
        self.assertEqual((pub["source"], priv["source"]), ("test02", "10A"))
        self.assertFalse(pub["a_rule"]["conditions"]["all_black"]["ok"])            # b_q != 1
        self.assertTrue(pub["a_rule"]["conditions"]["level3_by_rho"]["ok"])
        self.assertTrue(priv["a_holds"])
        self.assertTrue(priv["gate06_vs_r02prime"]["passed"])
        self.assertEqual(priv["gate06_vs_r02prime"]["r02prime"], "Q04")
        self.assertEqual(sha(self.p("o10", "public", "top5.csv")), sha(self.p("out", "player", "public", "top5.csv")))
        self.assertGreater(priv["top1_differences_vs_test02"], 0)
        rep = read_json(self.p("o10", "t10_submit_report.json"))
        self.assertTrue(rep["check_submission"]["passed"])
        self.assertEqual(rep["output"]["rows"], 800)
        self.assertEqual(rep["reference"], "Q04")
        self.assertIsNone(rep["top1_differences_vs_06_off"])
        self.assertEqual(rep["top1_differences_vs_q04"]["public"], 0)
        self.assertEqual(rep["top1_differences_vs_q04"]["private"], priv["top1_differences_vs_test02"])
        self.assertEqual(rep["sources"], {"public": "test02", "private": "10A"})
        self.assertFalse(rep["b"]["adopted"])
        sub = rows_of(self.p("o10", "submission_player.csv"))
        self.assertEqual(sub[1:401], rows_of(self.p("out", "player", "public", "top5.csv"))[1:])
        self.assertEqual(sub[401:], rows_of(self.p("o10", "private", "top5.csv"))[1:])

    def test_report_passes_official_log(self):
        probs, bound = official_log_problems(self, self.S, need_t4=True)
        self.assertEqual(probs, [])
        self.assertNotIn("results/test06/player/off/submission_player.csv", bound)

    # ------------------------------------------------------------ drills
    def test_fail_opening_is_3(self):
        run_tool(["tools/t10_fake_tests.py", "drill", "--root", self.S, "--kind", "fail_opening", "--name", "fo"],
                 log=self.log)
        o10 = drill_o10(self.S, "fo")
        _c, out = t10(self.S, "apply --pool private", expect=3, o10=o10, checks=self.p("drills", "fo", "checks"),
                      runlog=self.log)
        self.assertIn("FAIL_OPENING", out)
        self.assertEqual([e["reason"] for e in read_json(os.path.join(o10, "not_used.json"))["events"]],
                         ["t1_fail_opening"])

    def test_t06_seam_failed_is_3(self):
        run_tool(["tools/t10_fake_tests.py", "drill", "--root", self.S, "--kind", "t06_seam_failed", "--name", "sf"],
                 log=self.log)
        o10 = drill_o10(self.S, "sf")
        _c, out = t10(self.S, "apply --pool private", expect=3, o10=o10, o6=self.p("drills", "sf", "o6"),
                      runlog=self.log)
        self.assertIn("06 不能用", out)
        self.assertTrue(read_json(os.path.join(o10, "not_used.json"))["events"][0]["reason"].startswith("t06_"))

    def test_level0_keeps_test02_rows(self):
        run_tool(["tools/t10_fake_tests.py", "drill", "--root", self.S, "--kind", "level0", "--name", "l0",
                  "--pool", "private"], log=self.log)
        d = self.p("drills", "l0")
        o2, o6 = os.path.join(d, "o2"), os.path.join(d, "o6")
        t6d.t06(self.S, "apply --pool public", o6=o6, o2=o2, runlog=self.log)
        t6d.t06(self.S, "apply --pool private", expect=3, o6=o6, o2=o2, runlog=self.log)
        o10 = drill_o10(self.S, "l0")
        t10(self.S, "apply --pool private", o10=o10, o6=o6, o2=o2, runlog=self.log)
        rep = read_json(os.path.join(o10, "private", "t10_pool.json"))
        self.assertEqual(rep["source"], "test02")
        self.assertEqual(rep["test02"]["guard_level"], 0)
        self.assertFalse(rep["a_rule"]["conditions"]["level3_by_rho"]["ok"])
        self.assertEqual(sha(os.path.join(o10, "private", "top5.csv")), sha(os.path.join(o2, "private", "top5.csv")))

    def test_gate06_failed_keeps_test02_rows(self):
        o10 = drill_o10(self.S, "gate")
        code, out = in_process(["apply", "--pool", "private"] + t10_args(self.S, o10=o10),
                               lambda m: [(m.t6, "GATE06_MIN", 1.0)])
        self.assertEqual(code, 0, out)
        rep = read_json(os.path.join(o10, "private", "t10_pool.json"))
        self.assertTrue(rep["a_holds"])
        self.assertFalse(rep["gate06_vs_r02prime"]["passed"])
        self.assertEqual(rep["source"], "test02")
        self.assertEqual(sha(os.path.join(o10, "private", "top5.csv")),
                         sha(self.p("out", "player", "private", "top5.csv")))

    def test_guard_mismatch_is_2(self):
        o2 = os.path.join(self.S, "drills", "mismatch", "o2")
        shutil.rmtree(o2, ignore_errors=True)
        shutil.copytree(self.p("out", "player"), o2)
        p = os.path.join(o2, "private", "apply_report.json")
        ar = read_json(p)
        ar["guard_reason"] = ar["guard_reason"].replace("< theta2", "< theta2 (edited)")
        write_obj(p, ar)
        code, out = in_process(["apply", "--pool", "private"] + t10_args(self.S, o10=drill_o10(self.S, "mismatch"),
                                                                         o2=o2),
                               lambda m: [(m, "t06_pool_state", lambda ctx, pool, pre: (
                                   True, {}, read_json(self.p("o6", pool, "t06_pool.json")), m.t6.out_paths(ctx, pool)))])
        self.assertEqual(code, 2, out)
        self.assertIn("recomputed", out)

    def test_k6_10(self):
        o10 = drill_o10(self.S, "k6")
        _c, out = t10(self.S, "apply --pool private", expect=3, o10=o10, now="2026-11-22T00:00:00+08:00",
                      runlog=self.log)
        self.assertIn("K6-10 已過", out)
        _c, out = t10(self.S, "apply --pool private", expect=2, now="2026-11-22T00:00:00+08:00", runlog=self.log)
        self.assertIn("K6-10 前已 commit 的 10 照常有效", out)

    def test_b_gpu_seam_is_b_not_adopted(self):
        """PREREG_10 rev 5 (was the rev 4 tripwire 'real p1 = B not adopted, exit 3'): the repository's B is adopted
        through PREREG_13. Real status reads it and writes nothing; the real p1 is not run (it would go on to the
        test-day checks and the 4090). This A tree still never sees B (no --prereg10-dir: an absent folder)."""
        _c, out = run_tool(["tools/t10_player.py", "status"], log=self.log)
        self.assertIn("B: adopted via PREREG_13", out)
        # the committed K4-13 rehearsal record (60d0afe) is the only file allowed there before test day
        p1_dir = os.path.join(REPO, "results", "test10", "p1")
        present = sorted(os.listdir(p1_dir)) if os.path.isdir(p1_dir) else []
        self.assertTrue(set(present) <= {"p1_rehearsal.json"}, present)
        _c, out = run_tool(["tools/t10_player.py", "status"] + t10_args(self.S), log=self.log)
        self.assertIn("B not adopted", out)

    def test_reruns(self):
        _c, out = t10(self.S, "seam", expect=2, runlog=self.log)
        self.assertIn("--rerun --reason", out)
        rep = read_json(self.p("o10", "t10_submit_report.json"))
        log = os.path.join(self.S, "drills", "logged", "official_scores.json")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        write_obj(log, {"format": "c10 official_scores v1", "final_choice": {"rank": None, "player": None},
                        "uploads": [{"id": 1, "comp": "player", "status": "uploaded",
                                     "file": "results/test02/player/submission_player.csv",
                                     "sha256": sha(self.p("out", "player", "submission_player.csv")),
                                     "uploaded_at": "2026-11-04T16:05:00+08:00", "public_score": None},
                                    {"id": 2, "comp": "player", "status": "prepared",
                                     "file": "results/test10/player/submission_player.csv",
                                     "sha256": rep["output"]["sha256"], "uploaded_at": None, "public_score": None,
                                     "test10": {"run_id": rep["run_id"]}}]})
        before = sha(self.p("o10", "t10_submit_report.json"))
        _c, out = t10(self.S, "submit", "--rerun", "--reason", "drill", expect=2, log=log, runlog=self.log)
        self.assertIn("already holds 10 entries [2]", out)
        _c, out = t10(self.S, "apply --pool private", "--rerun", "--reason", "drill", expect=2, log=log,
                      runlog=self.log)
        self.assertIn("already holds 10 entries [2]", out)
        self.assertEqual(sha(self.p("o10", "t10_submit_report.json")), before)


# ================================================================ public level 1 (06), private nov4 (10A)
@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (10 dry runs)")
class MixTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S, cls.cleanup = t6d.workdir("t10_mix")
        cls.log = []
        S = cls.S
        run_tool(["tools/t10_fake_tests.py", "nov4", "--root", S, "--nov4-pools", "private"], log=cls.log)
        run_test02(S, cls.log)
        run_06(S, cls.log)
        run_10(S, cls.log)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "cleanup", False) and not os.environ.get("C10_KEEP"):
            shutil.rmtree(cls.S, ignore_errors=True)

    def p(self, *parts):
        return os.path.join(self.S, *parts)

    def test_levels_and_06(self):
        self.assertEqual(read_json(self.p("out", "player", "public", "apply_report.json"))["guard_level"], 1)
        self.assertEqual(read_json(self.p("out", "player", "private", "apply_report.json"))["guard_level"], 3)
        off = read_json(self.p("o6", "off", "t06_submit_report.json"))
        self.assertFalse(off["on_produced"])
        self.assertEqual({p: off["pools"][p]["used06"] for p in ("public", "private")},
                         {"public": True, "private": False})

    def test_10_against_06_off(self):
        rep = read_json(self.p("o10", "t10_submit_report.json"))
        self.assertEqual(rep["sources"], {"public": "06-OFF", "private": "10A"})
        self.assertEqual(rep["reference"], "06-OFF")
        self.assertEqual(rep["top1_differences_vs_06_off"]["public"], 0)
        self.assertGreater(rep["top1_differences_vs_06_off"]["private"], 0)
        self.assertTrue(rep["t06_off"]["exists"])
        self.assertFalse(rep["t06_off"]["on_produced"])
        self.assertEqual(rep["t06_off"]["sha256"], sha(self.p("o6", "off", "submission_player.csv")))
        sub = rows_of(self.p("o10", "submission_player.csv"))
        self.assertEqual(sub[1:401], rows_of(self.p("o6", "off", "submission_player.csv"))[1:401])
        probs, bound = official_log_problems(self, self.S, need_t4=True)
        self.assertEqual(probs, [])
        self.assertIn("results/test06/player/off/submission_player.csv", bound)


# ================================================================ part B (PREREG_10 rev 4): the mix tree with a fake adopted B
B_NOW = "2026-11-06T00:00:00+08:00"          # a test day after 06 and before K6B-10 (the dry runs do not use the clock)
B_REHEARSAL_NOW = "2026-10-20T00:00:00+08:00"  # before the test set opens


def b_args(S, o10=None, p10=None, p1o=None, p1d=None, now=B_NOW, **kw):
    o10 = o10 or os.path.join(S, "o10")
    return t10_args(S, o10=o10, now=now, **kw) + ["--prereg10-dir", p10 or os.path.join(S, "p10"),
                                         "--p1-out-dir", p1o or o10 + "_p1", "--p1-data-dir", p1d or o10 + "_p1data"]


def tb_run(S, cmd, *extra, expect=0, runlog=None, **kw):
    return run_tool(["tools/t10_player.py"] + cmd.split() + b_args(S, **kw) + list(extra), expect=expect, log=runlog)


def official_log_problems_b(case, S, o10, p1o, p10, p13=None, head="| rev 3 | x |\n| rev 4 | y |\n"):
    """As official_log_problems, for a 10 report with B: the fake PREREG_10 tree maps to results/prereg10 (and, PREREG_10
    rev 5, the fake PREREG_13 folder to results/prereg13) and the P1 reports to results/test10/p1; git (commit times,
    the rev 4 / rev 5 rows) is mocked to dates before K6B-10 / K5-10 / K5-13."""
    import official_log as ol
    root = tempfile.mkdtemp(prefix="c10_t10b_ol_")
    maps = [(os.path.join(S, "out", "player"), "results/test02/player"),
            (os.path.join(S, "results", "test02"), "results/test02"), (os.path.join(S, "o6"), "results/test06/player"),
            (o10, "results/test10/player"), (p1o, "results/test10/p1"), (p10, "results/prereg10")]
    if p13:
        maps.insert(0, (p13, "results/prereg13"))

    def to_rel(k):
        if not os.path.isabs(k):
            return k
        for src, dst in maps:
            if t6d.inside(k, src):
                return dst + "/" + os.path.relpath(k, src).replace(os.sep, "/")
        raise AssertionError(f"inputs_bound key outside the tree: {k}")
    try:
        rep = read_json(os.path.join(o10, "t10_submit_report.json"))
        bound = {}
        for k, want in rep["inputs_bound"].items():
            rel = to_rel(k)
            src = k if os.path.isabs(k) else os.path.join(REPO, *k.split("/"))
            dst = os.path.join(root, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            bound[rel] = want
        sub = os.path.join(root, "results", "test10", "player", "submission_player.csv")
        os.makedirs(os.path.dirname(sub), exist_ok=True)
        shutil.copyfile(os.path.join(o10, "submission_player.csv"), sub)
        state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
        with mock.patch.object(ol, "REPO", root), mock.patch.object(ol, "git_file_state", return_value=state), \
                mock.patch.object(ol, "git_commit_time", return_value=("e" * 40, "2026-11-10T00:00:00+08:00")), \
                mock.patch.object(ol, "git_row_added_on_main", return_value=("f" * 40, "2026-10-20T00:00:00+08:00")), \
                mock.patch.object(ol, "git_head_text", return_value=head):
            probs = ol.t10_report_problems(dict(rep, mode="real", inputs_bound=bound),
                                           f"results/test10/player/{ol.T10_REPORT}", sub, sha(sub))
            probs += ol.t10_inputs_problems(bound, need_t4=True)
        return probs, bound
    finally:
        shutil.rmtree(root, ignore_errors=True)


@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (10 dry runs)")
class BTree(unittest.TestCase):
    """PREREG_10 rev 4 on the mix tree (public at level 1: 06-OFF; private nov4: 10A) with a fake adopted B
    (tools/t10_b_fake.py: the real freeze, synthetic validation P1, a fake score_H10P.json): seam with the validation
    10B, the rehearsal (m, tol_P), the GPU seam, P1 of both pools (--synthetic-p1), apply, submit. With the synthetic
    P1 the pool gate fails in both pools, so 10 equals the A-only 10 byte for byte; in process with the gate minimum
    lowered both pools become 10B and the report passes tools/official_log.py's B rules. Drills: B not adopted (3, A
    only), GPU seam failed (3 for B, apply keeps A), no rehearsal (2), rehearsal after the test set opened (2), K6B-10
    passed (p1 3; apply without P1: B not added), apply before p1 (2), STOP (5) and resume, the fake-scorer path of
    p1_extract on 4 games."""

    @classmethod
    def setUpClass(cls):
        cls.S, cls.cleanup = t6d.workdir("t10_b")
        cls.log = []
        S = cls.S
        run_tool(["tools/t10_fake_tests.py", "nov4", "--root", S, "--nov4-pools", "private"], log=cls.log)
        run_test02(S, cls.log)
        run_06(S, cls.log)
        run_tool(["tools/t10_b_fake.py", "fake-p10", "--root", S], log=cls.log)
        cls.P10 = os.path.join(S, "p10")
        run_tool(["tools/t10_b_fake.py", "drill", "--p10", cls.P10, "--out", os.path.join(S, "p10_na"), "--kind",
                  "not_adopted"], log=cls.log)
        run_tool(["tools/t10_b_fake.py", "drill", "--p10", cls.P10, "--out", os.path.join(S, "p10_sp"), "--kind",
                  "seam_perturbed"], log=cls.log)
        run_tool(["tools/t10_b_fake.py", "drill", "--p10", cls.P10, "--out", os.path.join(S, "p10_inc"), "--kind",
                  "inconsistent"], log=cls.log)
        tb_run(S, "seam", runlog=cls.log)
        tb_run(S, "p1-rehearse --synthetic-p1", now=B_REHEARSAL_NOW, runlog=cls.log)
        for pool in ("seam_val", "public", "private"):
            tb_run(S, f"p1 --pool {pool} --synthetic-p1", runlog=cls.log)
        for pool in ("public", "private"):
            tb_run(S, f"apply --pool {pool}", runlog=cls.log)
        tb_run(S, "submit", runlog=cls.log)
        # the same 10 with B not adopted (rev 3 behaviour), in its own folder
        oa = os.path.join(S, "o10_na")
        for cmd in ("seam", "apply --pool public", "apply --pool private", "submit"):
            tb_run(S, cmd, o10=oa, p10=os.path.join(S, "p10_na"), runlog=cls.log)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "cleanup", False) and not os.environ.get("C10_KEEP"):
            shutil.rmtree(cls.S, ignore_errors=True)

    def p(self, *parts):
        return os.path.join(self.S, *parts)

    def fresh(self, name):
        """A drill output folder holding the base run's 10 seam (made with B)."""
        d = os.path.join(self.S, "drills", name, "o10")
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)
        shutil.copytree(self.p("o10", "seam_val"), os.path.join(d, "seam_val"))
        return d

    def test_seam_reproduces_the_validation_10b(self):
        seam = read_json(self.p("o10", "seam_val", "t10_seam.json"))
        self.assertTrue(seam["passed"], seam["criteria"])
        self.assertNotIn("b_10b_equal_to_prereg10_val", seam["criteria"])     # rev 4 section 14: B-only
        self.assertTrue(seam["b_seam"]["passed"])
        self.assertTrue(seam["b_seam"]["byte_identical"])
        self.assertTrue(seam["b"]["adopted"])
        self.assertEqual(seam["b_seam"]["sha256"], sha(self.p("p10", "player", "player_val_top5.csv")))

    def test_rehearsal_and_gpu_seam(self):
        r = read_json(self.p("o10_p1", "p1_rehearsal.json"))
        self.assertTrue(r["passed"])
        self.assertEqual((r["m"], r["tol_P"], r["scorer"]), (0.0, 0.05, "synthetic"))
        g = read_json(self.p("o10_p1", "gpu_seam.json"))
        self.assertTrue(g["passed"])
        self.assertEqual((g["games"], g["max_abs_projected_diff"], g["tol_P"]), (400, 0.0, 0.05))
        for pool in ("public", "private"):
            rep = read_json(self.p("o10_p1", f"{pool}_p1_report.json"))
            self.assertEqual(rep["failed"], 0)
            self.assertEqual(rep["games"], len(rep["chunks"]) * 200 - (200 - rep["chunks"][-1]["games"]))

    def test_pool_gate_failed_keeps_a(self):
        rep = read_json(self.p("o10", "t10_submit_report.json"))
        self.assertEqual(rep["sources"], {"public": "06-OFF", "private": "10A"})
        for pool in ("public", "private"):
            b = rep["pools"][pool]["b"]
            self.assertFalse(b["applied"])
            self.assertFalse(b["pool_gate"]["passed"])
            self.assertLess(b["pool_gate"]["sr_10B_minus_selected"], 0)
        self.assertTrue(rep["b"]["adopted"])
        self.assertEqual(rep["b"]["pools_10B"], [])
        self.assertEqual(sha(self.p("o10", "submission_player.csv")), sha(self.p("o10_na", "submission_player.csv")))
        na = read_json(self.p("o10_na", "t10_submit_report.json"))
        self.assertFalse(na["b"]["adopted"])
        self.assertNotIn("results/prereg10/RESULTS.md", na["inputs_bound"])

    def test_b_applied_in_process(self):
        o10 = self.fresh("applied")
        argv = b_args(self.S, o10=o10, p1o=self.p("o10_p1"), p1d=self.p("o10_p1data"))
        for pool in ("public", "private"):
            code, out = in_process(["apply", "--pool", pool] + argv, lambda m: [(m.tb, "GATE_B_MIN", -1.0)])
            self.assertEqual(code, 0, out)
        code, out = in_process(["submit"] + argv, lambda m: [])
        self.assertEqual(code, 0, out)
        rep = read_json(os.path.join(o10, "t10_submit_report.json"))
        self.assertEqual(rep["sources"], {"public": "10B", "private": "10B"})
        self.assertEqual(rep["b"]["pools_10B"], ["public", "private"])
        self.assertGreater(rep["top1_differences_vs_06_off"]["public"], 0)
        self.assertTrue(rep["check_submission"]["passed"])
        probs, bound = official_log_problems_b(self, self.S, o10, self.p("o10_p1"), self.P10)
        self.assertEqual(probs, [])
        for r in ("results/prereg10/RESULTS.md", "results/test10/p1/gpu_seam.json",
                  "results/test10/p1/private_p1_report.json", "results/test10/p1/p1_rehearsal.json"):
            self.assertIn(r, bound)

    def test_b_not_adopted_is_3(self):
        for cmd in ("p1 --pool seam_val --synthetic-p1", "p1-rehearse --synthetic-p1"):
            _c, out = tb_run(self.S, cmd, expect=3, o10=self.fresh("na"), p10=self.p("p10_na"), runlog=self.log)
            self.assertIn("B 沒有採用", out)
        self.assertFalse(os.path.exists(self.p("drills", "na", "o10_p1")))

    def test_gpu_seam_failed(self):
        o10 = self.fresh("sp")
        shutil.copytree(self.p("o10_p1"), o10 + "_p1", ignore=shutil.ignore_patterns("gpu_seam.json", "*_p1_report.json",
                                                                                       "b_not_used.json"))
        _c, out = tb_run(self.S, "p1 --pool seam_val --synthetic-p1", expect=2, o10=o10, p10=self.p("p10_sp"),
                         now=B_REHEARSAL_NOW, runlog=self.log)
        self.assertIn("made under another", out)                  # the rehearsal is bound to the val seam it measured
        _c, out = tb_run(self.S, "p1 --pool seam_val --synthetic-p1", expect=3, o10=o10, p10=self.p("p10_sp"),
                         runlog=self.log)                          # on the test day: B not used (review B1)
        self.assertIn("沒有可用的演練紀錄", out)

        def drifting(m):                                          # the test-day extraction drifts on 5 seam games
            orig = m.tb.synthetic_chunk

            def chunk(games):
                res, t = orig(games)
                if games and games[0]["ref"] == 0:
                    res["means"][:5, 0, :] += 0.5
                return res, t
            return [(m.tb, "synthetic_chunk", chunk)]
        code, out = in_process(["p1", "--pool", "seam_val", "--synthetic-p1"] + b_args(self.S, o10=o10), drifting)
        self.assertEqual(code, 3, out)
        self.assertIn("GPU 接縫沒過", out)
        g = read_json(o10 + "_p1/gpu_seam.json")
        self.assertFalse(g["passed"])
        self.assertTrue(g["counts_equal"])
        self.assertGreater(g["max_abs_projected_diff"], 0.05)
        self.assertEqual(g["games_above_tol"], 5)
        self.assertEqual([e["reason"] for e in read_json(o10 + "_p1/b_not_used.json")["events"]],
                         ["no_rehearsal", "gpu_seam_failed"])
        _c, out = tb_run(self.S, "p1 --pool public --synthetic-p1", expect=3, o10=o10, runlog=self.log)
        self.assertIn("GPU 接縫沒過", out)
        tb_run(self.S, "apply --pool private", o10=o10, runlog=self.log)
        rep = read_json(os.path.join(o10, "private", "t10_pool.json"))
        self.assertEqual(rep["source"], "10A")
        self.assertEqual(rep["b"]["pool"]["reason"], "gpu_seam_failed")
        self.assertFalse(os.path.exists(os.path.join(o10, "not_used.json")))     # B's "not used" is not 10's

    def test_rehearsal_rules(self):
        o10 = self.fresh("reh")
        _c, out = tb_run(self.S, "p1 --pool seam_val --synthetic-p1", expect=2, o10=o10, now=B_REHEARSAL_NOW,
                         runlog=self.log)
        self.assertIn("待演練", out)                               # before 11/04 11:00: rehearse first
        _c, out = tb_run(self.S, "p1-rehearse --synthetic-p1", expect=2, o10=o10, now="2026-11-04T11:00:00+08:00",
                         runlog=self.log)
        self.assertIn("before the test set opens", out)
        _c, out = tb_run(self.S, "p1 --pool seam_val --synthetic-p1", expect=3, o10=o10, runlog=self.log)
        self.assertIn("沒有可用的演練紀錄", out)                     # from 11/04 11:00: B not used, recorded
        self.assertEqual([e["reason"] for e in read_json(o10 + "_p1/b_not_used.json")["events"]], ["no_rehearsal"])
        tb_run(self.S, "apply --pool private", o10=o10, runlog=self.log)   # A continues (no GPU seam: p1_not_done)
        self.assertEqual(read_json(os.path.join(o10, "private", "t10_pool.json"))["source"], "10A")

    def test_k6b_and_apply_before_p1(self):
        o10 = self.fresh("k6b")
        shutil.copytree(self.p("o10_p1"), o10 + "_p1", ignore=shutil.ignore_patterns("gpu_seam.json", "*_p1_report.json",
                                                                                       "b_not_used.json"))
        _c, out = tb_run(self.S, "apply --pool private", o10=o10, runlog=self.log)      # review: never blocks A
        self.assertIn("this pool keeps A for now", out)
        rep = read_json(os.path.join(o10, "private", "t10_pool.json"))
        self.assertEqual((rep["source"], rep["b"]["pool"]["reason"]), ("10A", "p1_not_done"))
        _c, out = tb_run(self.S, "p1 --pool seam_val --synthetic-p1", expect=3, o10=o10,
                         now="2026-11-15T00:00:00+08:00", runlog=self.log)
        self.assertIn("K6B-10 已過", out)
        tb_run(self.S, "apply --pool private", "--rerun", "--reason", "drill", o10=o10,
               now="2026-11-15T00:00:00+08:00", runlog=self.log)
        rep = read_json(os.path.join(o10, "private", "t10_pool.json"))
        self.assertEqual((rep["source"], rep["b"]["pool"]["reason"]), ("10A", "k6b_10"))

    def test_val_10b_seam_failed_is_b_only(self):
        """Review M4: the validation-10B seam failing never stops A: seam passes for A, B is not used (recorded)."""
        o10 = os.path.join(self.S, "drills", "v10b", "o10")
        shutil.rmtree(os.path.dirname(o10), ignore_errors=True)
        argv = b_args(self.S, o10=o10, p1o=self.p("o10_p1"), p1d=self.p("o10_p1data"))
        bad = {"passed": False, "byte_identical": False, "rows_identical": 390, "rows": 400}
        code, out = in_process(["seam"] + argv, lambda m: [(m.tb, "seam_10b", lambda *a, **k: dict(bad))])
        self.assertEqual(code, 0, out)
        seam = read_json(os.path.join(o10, "seam_val", "t10_seam.json"))
        self.assertTrue(seam["passed"])
        self.assertNotIn("b_10b_equal_to_prereg10_val", seam["criteria"])
        self.assertFalse(seam["b_seam"]["passed"])
        code, out = in_process(["apply", "--pool", "private"] + argv, lambda m: [(m.tb, "GATE_B_MIN", -1.0)])
        self.assertEqual(code, 0, out)
        rep = read_json(os.path.join(o10, "private", "t10_pool.json"))
        self.assertEqual((rep["source"], rep["b"]["pool"]["reason"]), ("10A", "val_10b_seam_failed"))

    def test_inconsistent_b_files_degrade(self):
        """Review M2 / M4: B files that disagree stop the rehearsal (before the test day, 2); on the test day p1 is exit
        3 (recorded) and seam / apply / submit run as A."""
        o10 = os.path.join(self.S, "drills", "inc", "o10")
        shutil.rmtree(os.path.dirname(o10), ignore_errors=True)
        p10 = self.p("p10_inc")
        _c, out = tb_run(self.S, "p1-rehearse --synthetic-p1", expect=2, o10=o10, p10=p10, now=B_REHEARSAL_NOW,
                         runlog=self.log)
        self.assertIn("disagree", out)
        _c, out = tb_run(self.S, "p1 --pool seam_val --synthetic-p1", expect=3, o10=o10, p10=p10, runlog=self.log)
        self.assertEqual([e["reason"] for e in read_json(o10 + "_p1/b_not_used.json")["events"]], ["b_inconsistent"])
        for cmd in ("seam", "apply --pool public", "apply --pool private", "submit"):
            tb_run(self.S, cmd, o10=o10, p10=p10, runlog=self.log)
        rep = read_json(os.path.join(o10, "t10_submit_report.json"))
        self.assertEqual(rep["sources"], {"public": "06-OFF", "private": "10A"})
        self.assertEqual((rep["b"]["adopted"], rep["b"]["reason"]), (False, "inconsistent"))
        self.assertEqual(sha(os.path.join(o10, "submission_player.csv")), sha(self.p("o10_na", "submission_player.csv")))

    def test_a_dry_runs_never_see_b(self):
        """Review M1: without --prereg10-dir a dry run reads an absent folder, so the A trees stay A-only even after B is
        adopted in the repository."""
        o10 = os.path.join(self.S, "drills", "m1", "o10")
        _c, out = run_tool(["tools/t10_player.py", "status"] + t10_args(self.S, o10=o10), log=self.log)
        self.assertIn("B not adopted", out)
        self.assertIn("not_marked", out)

    def test_stop_and_resume(self):
        o10 = self.fresh("stop")
        shutil.copytree(self.p("o10_p1"), o10 + "_p1", ignore=shutil.ignore_patterns("*_p1_report.json",
                                                                                       "b_not_used.json"))
        d = o10 + "_p1data"
        os.makedirs(os.path.join(d, "public"))
        open(os.path.join(d, "public", "STOP"), "w").close()
        _c, out = tb_run(self.S, "p1 --pool public --synthetic-p1", expect=5, o10=o10, runlog=self.log)
        self.assertIn("STOPPED", out)
        self.assertEqual([x for x in os.listdir(os.path.join(d, "public")) if x.endswith(".npz")], [])
        os.remove(os.path.join(d, "public", "STOP"))
        tb_run(self.S, "p1 --pool public --synthetic-p1", o10=o10, runlog=self.log)
        self.assertEqual(read_json(o10 + "_p1/public_p1_report.json")["keys"]["keys_sha256"],
                         read_json(self.p("o10_p1", "public_p1_report.json"))["keys"]["keys_sha256"])
        _c, out = tb_run(self.S, "p1 --pool public --fake-scorer", expect=2, o10=o10, runlog=self.log)
        self.assertIn("exist", out)

    def test_b_via_prereg13(self):
        """PREREG_10 rev 5 (section 16): the same B tree with H10P never run and B adopted through a fake PREREG_13
        folder (tools/t10_b_fake.py fake-p13): status, seam (the validation 10B), the rehearsal, P1 of the GPU seam and
        both pools, apply (gate lowered in process: both pools 10B), submit; the report records source PREREG_13 and
        H13P, binds results/prereg13 instead of score_H10P.json and passes tools/official_log.py's rev 5 rules (K5-13
        mocked before / after). Drill: PREREG_13 not adopted -> p1 exit 3, nothing written."""
        S, q10 = self.S, self.p("q10")
        run_tool(["tools/t10_b_fake.py", "fake-p13", "--p10", self.P10, "--out", q10], log=self.log)
        o10 = self.p("drills", "p13", "o10")
        shutil.rmtree(os.path.dirname(o10), ignore_errors=True)
        _c, out = tb_run(S, "status", o10=o10, p10=q10, runlog=self.log)
        self.assertIn("B: adopted via PREREG_13", out)
        tb_run(S, "seam", o10=o10, p10=q10, runlog=self.log)
        seam = read_json(os.path.join(o10, "seam_val", "t10_seam.json"))
        self.assertTrue(seam["passed"])
        self.assertTrue(seam["b_seam"]["passed"])
        self.assertEqual(seam["b"]["source"], "PREREG_13")
        tb_run(S, "p1-rehearse --synthetic-p1", o10=o10, p10=q10, now=B_REHEARSAL_NOW, runlog=self.log)
        for pool in ("seam_val", "public", "private"):
            tb_run(S, f"p1 --pool {pool} --synthetic-p1", o10=o10, p10=q10, runlog=self.log)
        argv = b_args(S, o10=o10, p10=q10)
        for pool in ("public", "private"):
            code, out = in_process(["apply", "--pool", pool] + argv, lambda m: [(m.tb, "GATE_B_MIN", -1.0)])
            self.assertEqual(code, 0, out)
        code, out = in_process(["submit"] + argv, lambda m: [])
        self.assertEqual(code, 0, out)
        rep = read_json(os.path.join(o10, "t10_submit_report.json"))
        self.assertEqual(rep["sources"], {"public": "10B", "private": "10B"})
        self.assertEqual(rep["b"]["source"], "PREREG_13")
        self.assertEqual(rep["b"]["h13p"]["n_view"], 263)
        self.assertNotIn("h10p", rep["b"])
        head5 = "| rev 3 | x |\n| rev 4 | y |\n| rev 5 | z |\n"
        probs, bound = official_log_problems_b(self, S, o10, o10 + "_p1", q10, p13=q10 + "_p13", head=head5)
        self.assertEqual(probs, [])
        for r in ("results/prereg13/RESULTS.md", "results/prereg13/player/score_H13P.json",
                  "results/prereg13/player/view_report.json", "results/prereg13/player/prescore_audit.json",
                  "results/prereg10/player/frozen_config.json", "results/test10/p1/gpu_seam.json"):
            self.assertIn(r, bound)
        self.assertNotIn("results/prereg10/player/score_H10P.json", bound)
        probs, _b = official_log_problems_b(self, S, o10, o10 + "_p1", q10, p13=q10 + "_p13")
        self.assertTrue(any("K5-13" in x for x in probs), probs)          # rev 5 not on main: no B through PREREG_13
        # the same rows as the H10P tree's 10B run (B itself is the same frozen B)
        ref = self.fresh("p13ref")
        argv_ref = b_args(S, o10=ref, p1o=self.p("o10_p1"), p1d=self.p("o10_p1data"))
        for pool in ("public", "private"):
            code, out = in_process(["apply", "--pool", pool] + argv_ref, lambda m: [(m.tb, "GATE_B_MIN", -1.0)])
            self.assertEqual(code, 0, out)
        for pool in ("public", "private"):
            self.assertEqual(sha(os.path.join(o10, pool, "top5.csv")), sha(os.path.join(ref, pool, "top5.csv")))
        # drill: PREREG_13 not adopted -> the registered 'B not adopted' (3, nothing written)
        qna = self.p("q10_na")
        run_tool(["tools/t10_b_fake.py", "fake-p13", "--p10", self.P10, "--out", qna, "--kind", "not_adopted"],
                 log=self.log)
        ona = self.p("drills", "p13na", "o10")
        shutil.rmtree(os.path.dirname(ona), ignore_errors=True)
        _c, out = tb_run(S, "p1 --pool seam_val --synthetic-p1", expect=3, o10=ona, p10=qna, runlog=self.log)
        self.assertIn("B 沒有採用", out)
        self.assertFalse(os.path.exists(ona + "_p1"))
        self.assertTrue(read_json(os.path.join(self.P10, "player", "apply_report.json"))["score_allowed"])  # untouched

    def test_fake_scorer_extraction_drill(self):
        o10 = self.fresh("fake")
        shutil.copytree(self.p("o10_p1"), o10 + "_p1", ignore=shutil.ignore_patterns("gpu_seam.json", "*_p1_report.json",
                                                                                       "b_not_used.json"))
        _c, out = tb_run(self.S, "p1 --pool seam_val --fake-scorer --max-games 4 --workers 2", o10=o10,
                         runlog=self.log)
        self.assertIn("drill of 4 games", out)
        files = [x for x in os.listdir(os.path.join(o10 + "_p1data", "seam_val")) if x.endswith(".npz")]
        self.assertEqual(files, ["chunk_00000.npz"])
        import numpy as np
        with np.load(os.path.join(o10 + "_p1data", "seam_val", files[0]), allow_pickle=False) as z:
            self.assertEqual((str(z["scorer"]), z["means"].shape), ("fake", (4, 2, 770)))
            self.assertFalse(np.asarray(z["failed"]).any())
            self.assertTrue((np.asarray(z["counts"])[:, 0] > 0).all())


if __name__ == "__main__":
    unittest.main()
