"""End-to-end dry run of the 19 test-day kit on the from-val fake tree (PREREG_19 4.2; rev 3). Every tool is called
through its command line, as on test day.

    C10_SLOW=1 or 2   the tree of tests/test_t09_dryrun.py (R20, 04, 07, 08 and 09 dry runs in a temporary folder
        outside the repository; about 15 min; use C10_DRYRUN_DIR=<a subst drive> so the path has no 'AppData\\Local'),
        or C10_T19_TREE=<S> to reuse a tree whose S/results/test09/rank/t09_apply_report.json exists (S/results/test19
        is rebuilt). Then 19: t19_rank.py apply / status --dry-run.
        The R20 / 04 / 07 / 08 outputs equal results/test09/design/baseline_pre09.json and the 09 outputs equal
        t09_fake_golden.json byte for byte (the 19 kit never changes them); Q09 recomputed equals 09's
        rank_test_probs.csv byte for byte; the SET seam record is 0; the local seam is byte-identical to
        results/prereg19/rank (rank_val_probs.csv, pred_B.csv, pred_A.csv); the 800 test rows equal prereg19's first 800
        validation rows under the fake prefixes within 1e-12 with every rule-A and rule-B level equal; lp_SET equals
        val_set.npz rows within 1e-12; the submission passes check_submission --official; the 19 outputs equal
        results/test19/design/t19_fake_golden.json; t19 never changes a file of test02 / 04 / 07 / 08 / 09.
        Negative drills (on copies): 09 not used, 09 not done, 09's output changed, Q09 one value off 09's file, K6-19,
        FAIL_OPENING, outputs present (and a rerun without a PREREG_19 revision), BLAS thread variables, a raw tests
        folder, drill flags in the real mode; in a child process with one module value patched: the local seam on the
        tolerance path (3), a failed SET direction check (3), and a --bug-revision rerun over the recorded not_used.json.
The fake tree holds validation games: only the seams and apply run on it. Nothing here opens a label file or runs
tools/score.py; nothing is written inside the repository.
Run: C10_SLOW=1 python -m pytest tests/test_t19_dryrun.py   (C10_KEEP=1 keeps the tree)
"""

import csv
import json
import os
import shutil
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import check_submission  # noqa: E402
import test_t09_dryrun as t9d  # noqa: E402

SLOW = os.environ.get("C10_SLOW", "")
BASELINE = t9d.BASELINE
GOLDEN09 = t9d.GOLDEN09
GOLDEN19 = os.path.join(REPO, "results", "test19", "design", "t19_fake_golden.json")
P19 = os.path.join(REPO, "results", "prereg19", "rank")
PREFIX, RANK_ROWS, RANK_FILES = t9d.PREFIX, t9d.RANK_ROWS, t9d.RANK_FILES
GOLDEN19_FILES = ("results/test19/rank/rank_test_probs.csv", "results/test19/rank/pred_A.csv",
                  "results/test19/rank/pred_B.csv", "results/test19/rank/submission_rank.csv",
                  "results/test19/rank/test_lp_set.npz", "results/test19/rank/seam_val/rank_val_probs.csv",
                  "results/test19/rank/seam_val/pred_A.csv", "results/test19/rank/seam_val/pred_B.csv")
sha, read_json, write_json, lf_lines, snapshot, run_tool = (t9d.sha, t9d.read_json, t9d.write_json, t9d.lf_lines,
                                                            t9d.snapshot, t9d.run_tool)


class Tree19:
    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            cls.built = cls()
        return cls.built

    def __init__(self):
        given = os.environ.get("C10_T19_TREE")
        self.t9 = None
        if given:
            self.S = os.path.abspath(given)
            if not os.path.exists(os.path.join(self.S, "results", "test09", "rank", "t09_apply_report.json")):
                raise unittest.SkipTest(f"C10_T19_TREE={given} holds no test09 dry run")
        else:
            self.t9 = t9d.Tree09.get()
            if self.t9.apply_code != 0:
                raise unittest.SkipTest("the 09 dry run did not finish")
            self.S = self.t9.S
        S = self.S
        self.tests = os.path.join(S, "data", "test")
        self.D2 = os.path.join(S, "results", "test02")
        self.O2 = os.path.join(S, "out", "rank")
        self.C = os.path.join(self.D2, "checks")
        self.R4, self.R7 = os.path.join(S, "results", "test04"), os.path.join(S, "results", "test07")
        self.R8, self.R9 = os.path.join(S, "results", "test08"), os.path.join(S, "results", "test09")
        self.R19 = os.path.join(S, "results", "test19")
        self.drills = os.path.join(S, "t19_drills")
        for d in (self.R19, self.drills):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(self.drills)
        self.snap0 = self.others()
        t0 = time.time()
        self.apply_code, self.apply_out = run_tool(["tools/t19_rank.py", "apply", "--dry-run"] + self.dirs())
        self.status_code, self.status_out = run_tool(["tools/t19_rank.py", "status", "--dry-run"] + self.dirs())
        self.seconds = round(time.time() - t0, 1)
        self.snap1 = self.others()

    def others(self):
        return snapshot(self.D2, self.O2, self.tests, self.R4, self.R7, self.R8, self.R9)

    def dirs(self, R19=None, R9=None, C=None, T=None):
        return ["--results-dir", R19 or self.R19, "--t09-results-dir", R9 or self.R9, "--t08-results-dir", self.R8,
                "--t07-results-dir", self.R7, "--t04-results-dir", self.R4, "--t02-results-dir", self.D2,
                "--t02-out-dir", self.O2, "--tests-dir", T or self.tests, "--checks-dir", C or self.C]

    def copy_dir(self, src, name):
        d = os.path.join(self.drills, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(src, d)
        return d

    def r19(self, name):
        d = os.path.join(self.drills, name, "test19")
        shutil.rmtree(d, ignore_errors=True)
        return d


def tearDownModule():
    t = Tree19.built
    if t is not None and not os.environ.get("C10_KEEP"):
        shutil.rmtree(t.drills, ignore_errors=True)
        shutil.rmtree(t.R19, ignore_errors=True)
    Tree19.built = None
    if t is not None and t.t9 is not None:
        t9d.tearDownModule()


SKIP = unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (R20, 04, 07, 08, 09 and 19 dry runs, about 15-20 min)")


def golden_record(S):
    """The 19 outputs of a passing dry run (written once into results/test19/design/t19_fake_golden.json)."""
    return {"what": "PREREG_19 rev 3: the 19 outputs of the first passing dry run (tests/test_t19_dryrun.py) on the "
                    "from-val fake tree; later runs must equal them byte for byte",
            "files_sha256": {"t19": {rel: sha(os.path.join(S, *rel.split("/"))) for rel in GOLDEN19_FILES}}}


@SKIP
class T19DryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree19.get()
        cls.rank = os.path.join(cls.t.R19, "rank")
        cls.rep = read_json(os.path.join(cls.rank, "t19_apply_report.json"))

    def test_apply_exit_0(self):
        self.assertEqual(self.t.apply_code, 0, self.t.apply_out[-2000:])
        self.assertEqual((self.rep["mode"], self.rep["exit_code"], self.rep["usable"]), ("dry-run", 0, True))
        self.assertIn("apply: exit 0", self.t.status_out)

    def test_earlier_outputs_untouched(self):
        self.assertEqual(self.t.snap1, self.t.snap0)

    def test_r20_04_07_08_09_equal_their_baselines(self):
        g = read_json(BASELINE)["files_sha256"]
        for part in ("r20", "t04", "t07", "t08"):
            for rel, want in g[part].items():
                with self.subTest(rel):
                    self.assertEqual(sha(os.path.join(self.t.S, *rel.split("/"))), want)
        for rel, want in read_json(GOLDEN09)["files_sha256"]["t09"].items():
            with self.subTest(rel):
                self.assertEqual(sha(os.path.join(self.t.S, *rel.split("/"))), want)

    def test_q09_and_set_seam(self):
        q = self.rep["q09_vs_09"]
        self.assertTrue(q["byte_equal"] and q["equal"] and q["p1_ok_equal_t09"] and q["usable_equal_t09"])
        self.assertEqual(q["lp_p1e_max_abs_diff_vs_t09"], 0.0)
        s = self.rep["set_seam_record_only"]
        self.assertEqual((s["max_abs_lp_set"], s["usable_games"], s["bitwise_equal_means"]), (0.0, 400, True))

    def test_local_seam_byte_identical_to_prereg19(self):
        seam = os.path.join(self.rank, "seam_val")
        for n in ("rank_val_probs.csv", "pred_B.csv", "pred_A.csv"):
            self.assertEqual(lf_lines(os.path.join(seam, n)), lf_lines(os.path.join(P19, n)), n)
        sl = read_json(os.path.join(seam, "seam_local.json"))
        self.assertEqual((sl["verdict"], sl["required_byte_identical"], sl["max_abs_diff"]), ("passed", True, 0.0))
        self.assertTrue(sl["usable_equal_to_prereg19"])
        self.assertTrue(self.rep["set_direction"]["seam_val_record_only"]["equal_to_prereg19_apply"])

    def test_test_rows_equal_prereg19_validation_rows(self):
        ref, got = lf_lines(os.path.join(P19, "rank_val_probs.csv")), lf_lines(os.path.join(self.rank,
                                                                                          "rank_test_probs.csv"))
        self.assertEqual(got[0], ref[0])
        self.assertEqual(len(got), 802)
        worst = 0.0
        for s, (lo, hi) in RANK_ROWS.items():
            for i in range(lo, hi):
                a, b = got[1 + i].decode().split(","), ref[1 + i].decode().split(",")
                self.assertEqual(a[0], PREFIX[s] + b[0][len("val_"):])
                self.assertEqual(a[1:3], b[1:3])
                worst = max([worst] + [abs(float(x) - float(y)) for x, y in zip(a[3:], b[3:]) if x and y])
        self.assertLessEqual(worst, 1e-12)
        for name, ref_name in (("submission_rank.csv", "pred_B.csv"), ("pred_A.csv", "pred_A.csv")):
            ref, got = lf_lines(os.path.join(P19, ref_name)), lf_lines(os.path.join(self.rank, name))
            self.assertEqual(len(got), 802)
            for s, (lo, hi) in RANK_ROWS.items():
                for i in range(lo, hi):
                    self.assertEqual(got[1 + i], PREFIX[s].encode() + ref[1 + i][len(b"val_"):], (name, s, i))

    def test_lp_set_equal_val_set(self):
        z = dict(np.load(os.path.join(self.rank, "test_lp_set.npz"), allow_pickle=False))
        v = dict(np.load(os.path.join(P19, "val_set.npz"), allow_pickle=False))
        self.assertEqual(len(z["lp"]), 800)
        self.assertLessEqual(float(np.max(np.abs(z["lp"] - v["lp"][:800]))), 1e-12)
        self.assertTrue(np.array_equal(z["has"], v["has"][:800]))

    def test_submission_and_checks(self):
        sub = os.path.join(self.rank, "submission_rank.csv")
        self.assertEqual(sha(sub), sha(os.path.join(self.rank, "pred_B.csv")))
        qf = [os.path.join(self.t.tests, RANK_FILES[s]) for s in ("public", "private")]
        self.assertEqual(check_submission.check("rank", sub, qf, official=True), [])
        self.assertEqual(self.rep["outputs_sha256"]["submission_rank.csv"], sha(sub))
        for s in ("public", "private"):
            self.assertTrue(self.rep["set_direction"][s]["passed"])
            self.assertAlmostEqual(self.rep["set_direction"][s]["threshold"], 0.4792752582816624, places=15)
        self.assertEqual(set(self.rep["rule_b_differences"]), {"09", "08", "07", "04", "R20"})
        self.assertEqual(self.rep["combo"]["id"], "Q19")

    def test_inputs_bound_cover_official_log(self):
        import official_log as ol
        self.assertEqual([r for r in ol.T19_INPUTS_REQUIRED if r not in self.rep["inputs_bound"]], [])
        self.assertEqual(tuple(self.rep["combo"]["sources"]), ol.T19_COMBO_SOURCES)

    def test_golden(self):
        if not os.path.exists(GOLDEN19):
            self.skipTest(f"{GOLDEN19} not written yet (first dry run)")
        g = read_json(GOLDEN19)["files_sha256"]
        self.assertEqual(sorted(g["t19"]), sorted(GOLDEN19_FILES))
        for rel, want in g["t19"].items():
            with self.subTest(rel):
                self.assertEqual(sha(os.path.join(self.t.S, *rel.split("/"))), want)


@SKIP
class T19Drills(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree19.get()

    def apply(self, R19, expect, R9=None, C=None, T=None, extra=(), env=None):
        _code, out = run_tool(["tools/t19_rank.py", "apply", "--dry-run"] + self.t.dirs(R19, R9, C, T) + list(extra),
                              expect=expect, env_extra=env)
        return out

    def verdict(self, R19):
        p = os.path.join(R19, "rank", "attempt_verdict.json")
        return read_json(p) if os.path.exists(p) else read_json(os.path.join(R19, "not_used.json"))

    def copy_r9(self, name):
        return self.t.copy_dir(self.t.R9, os.path.join(name, "test09"))

    def test_09_not_used_and_not_done(self):
        r9 = self.copy_r9("t09_not_done")
        shutil.rmtree(os.path.join(r9, "rank"))
        self.assertIn("test09 apply has not finished", self.apply(self.t.r19("t09_not_done"), 2, R9=r9))
        write_json(os.path.join(r9, "not_used.json"), {"format": "c10-test09 not used v1", "mode": "dry-run",
                                                       "exit_code": 3, "stage": "p1_direction", "reason": "drill"})
        r19 = self.t.r19("t09_not_used")
        self.apply(r19, 3, R9=r9)
        self.assertEqual(self.verdict(r19)["reason"], "t09_not_used")

    def test_09_output_changed(self):
        r9 = self.copy_r9("out_changed")
        with open(os.path.join(r9, "rank", "submission_rank.csv"), "ab") as f:
            f.write(b"\n")
        self.assertIn("test09 outputs differ from its report", self.apply(self.t.r19("out_changed"), 2, R9=r9))

    def test_q09_one_value_off(self):
        r9 = self.copy_r9("q09")
        p = os.path.join(r9, "rank", "rank_test_probs.csv")
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        i = next(k for k, r in enumerate(rows) if k and r[3])
        rows[i][3] = repr(float(rows[i][3]) + 1e-9)
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        rp = os.path.join(r9, "rank", "t09_apply_report.json")
        rep = read_json(rp)
        rep["outputs_sha256"]["rank_test_probs.csv"] = sha(p)
        write_json(rp, rep)
        r19 = self.t.r19("q09")
        self.apply(r19, 4, R9=r9)
        v = self.verdict(r19)
        self.assertEqual(v["stage"], "q09_vs_09")
        self.assertFalse(v["details"]["equal"])

    def test_k6_19(self):
        r19 = self.t.r19("k6")
        self.apply(r19, 3, extra=["--now", "2026-11-20T00:00:00+08:00"])
        self.assertEqual(self.verdict(r19)["reason"], "k6_19")
        out = self.apply(self.t.R19, 2, extra=["--now", "2026-11-20T00:00:00+08:00", "--rerun", "--reason", "drill"])
        self.assertIn("stays valid", out)

    def test_fail_opening(self):
        c = self.t.copy_dir(self.t.C, "checks_fo")
        p = os.path.join(c, "rank", "t1.json")
        t1 = read_json(p)
        t1["verdict"] = "FAIL_OPENING"
        write_json(p, t1)
        r19 = self.t.r19("fail_opening")
        self.apply(r19, 3, C=c)
        self.assertTrue(os.path.exists(os.path.join(r19, "not_used.json")))

    def test_outputs_present(self):
        self.assertIn("never overwritten", self.apply(self.t.R19, 2))
        self.assertIn("registered decision", self.apply(self.t.R19, 2, extra=["--rerun", "--reason", "drill"]))
        self.assertIn("rev 4 or later", self.apply(self.t.R19, 2, extra=["--rerun", "--reason", "drill",
                                                                         "--bug-revision", "3"]))

    def test_blas_env(self):
        self.assertIn("BLAS thread variables", self.apply(self.t.r19("blas"), 2, env={"MKL_NUM_THREADS": "2"}))

    def test_raw_tests_dir(self):
        self.assertIn("never data/test/raw", self.apply(self.t.r19("raw"), 2, T=os.path.join(self.t.tests, "raw")))

    def patched(self, R19, expect, patch, extra=()):
        """t19_rank.py apply --dry-run in a child process with one module attribute patched (drills of decision paths
        the fake tree cannot reach by editing files)."""
        code = "\n".join(["import json, sys", "sys.path.insert(0, 'tools')", "import t19_rank as T", patch,
                          "sys.exit(T.main(json.loads(sys.argv[1])))"])
        argv = ["apply", "--dry-run"] + self.t.dirs(R19) + list(extra)
        _code, out = run_tool(["-c", code, json.dumps(argv)], expect=expect)
        return out

    def test_local_seam_tolerance_path_not_used(self):
        r19 = self.t.r19("seam_tol")
        self.patched(r19, 3, "T.t9.seam_verdict = lambda sc, identical: 'not_used'")
        v = self.verdict(r19)
        self.assertEqual((v["exit_code"], v["stage"], v["reason"]), (3, "local_seam", "seam_tol"))

    def test_set_direction_not_used_and_bug_revision_rerun(self):
        r19 = self.t.r19("direction")
        self.patched(r19, 3, "T.RHO_FLOOR = 1.1")
        v = self.verdict(r19)
        self.assertEqual((v["stage"], v["reason"]), ("set_direction", "set_direction"))
        nu = os.path.join(r19, "not_used.json")
        before = sha(nu)
        out = self.apply(r19, 2, extra=["--rerun", "--reason", "drill"])
        self.assertIn("registered decision", out)
        # a rerun that stops at a prerequisite (K6-19 -> 3) never overwrites the recorded decision
        out = self.apply(r19, 2, extra=["--rerun", "--reason", "drill", "--now", "2026-11-20T00:00:00+08:00"])
        self.assertIn("not recorded", out)
        self.assertEqual(sha(nu), before)
        self.assertFalse(os.path.exists(os.path.join(r19, "rank", "attempt_verdict.json")))
        fake_rev = ("T.check_bug_revision = lambda n, ov: {'rev': n, 'commit': 'f' * 40, "
                    "'committed_at': '2026-11-10T00:00:00+08:00', 'overturns': [{'what': w, 'time': x} for w, x in ov]}")
        self.patched(r19, 0, fake_rev, extra=["--rerun", "--reason", "drill: rev 4", "--bug-revision", "4"])
        rep = read_json(os.path.join(r19, "rank", "t19_apply_report.json"))
        self.assertEqual(rep["rerun"]["bug_revision"]["rev"], 4)
        self.assertTrue(any(k.endswith("not_used.json") for k in rep["rerun"]["replaced"]))
        self.assertFalse(os.path.exists(nu))
        self.assertEqual(sha(os.path.join(r19, "rank", "submission_rank.csv")),
                         sha(os.path.join(self.t.R19, "rank", "submission_rank.csv")))

    def test_real_mode_drill_flags_refused(self):
        _code, out = run_tool(["tools/t19_rank.py", "apply", "--now", "2026-11-20T00:00"], expect=2)
        self.assertIn("drills only", out)


if __name__ == "__main__":
    unittest.main()
