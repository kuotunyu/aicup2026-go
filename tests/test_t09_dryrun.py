"""End-to-end dry run of the 09 test-day kit on the from-val fake tree (PREREG_09 5.1 step 3, 5.2; rev 3 section 13).
Every tool is called through its command line, as on test day.

    C10_SLOW=1 or 2   the tree of tests/test_t07_dryrun.py (R20, 04 and 07 dry runs in a temporary folder outside
        the repository; about 8-10 min), or C10_T09_TREE=<S> to reuse a tree whose S/results/test07/rank/
        t07_apply_report.json exists (test02 / test04 / test07 dry runs by the same commands; S/results/test08 and
        S/results/test09 are rebuilt). Then:
        08: t09_fake_tests.py from-val (= t08_fake_tests.py from-val, unchanged, plus the per-position arrays of the
            same validation games copied into the drill chunks) -> t08_rank.py apply --dry-run.
        09: t09_rank.py apply / status --dry-run.
        The R20 / 04 / 07 / 08 outputs equal results/test09/design/baseline_pre09.json byte for byte (PREREG_09 5.1
        step 3; the 09 kit never changes them); the 09 GPU seam is 0 / 0; Q08 recomputed equals 08's
        rank_test_probs.csv byte for byte; the local seam is byte-identical to results/prereg09/rank
        (rank_val_probs.csv, pred_B.csv, pred_A.csv); the 800 test rows equal prereg09's first 800 validation rows
        under the fake prefixes within 1e-12 with every rule-A and rule-B level equal (the numpy forward runs on one
        set here and on 19,156 rows in prereg09: last-bit differences only); lp_P1TO_E equals val_lp_p1e.npz rows
        within 1e-12; the submission passes check_submission --official; the 09 outputs equal
        results/test09/design/t09_fake_golden.json; t09 never changes a file of test02 / test04 / test07 / test08.
        Negative drills (each on copies, never on the tree itself): the 09 GPU seam (EX), an 08 chunk changed after
        08's apply, 08's P1 report changed, Q08 one value off 08's file, 08's output changed, 08 not used, 08 not done,
        K6-09, FAIL_OPENING, outputs present (and a rerun without a PREREG_09 revision), BLAS thread variables, a raw
        tests folder.
The fake tree holds validation games: only the seams and apply run on it. Nothing here opens a label file or runs
tools/score.py; nothing is written inside the repository.
Run: C10_SLOW=1 python -m pytest tests/test_t09_dryrun.py   (C10_KEEP=1 keeps the tree)
"""

import builtins
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import unittest

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import check_submission  # noqa: E402

SLOW = os.environ.get("C10_SLOW", "")
BASELINE = os.path.join(REPO, "results", "test09", "design", "baseline_pre09.json")
GOLDEN09 = os.path.join(REPO, "results", "test09", "design", "t09_fake_golden.json")
P09 = os.path.join(REPO, "results", "prereg09", "rank")
PREFIX = {"public": "pub_", "private": "priv_"}
RANK_ROWS = {"public": (0, 400), "private": (400, 800)}
RANK_FILES = {"public": "rank_prediction_test_public.csv", "private": "rank_prediction_test_private.csv"}
GOLDEN09_FILES = ("results/test09/rank/rank_test_probs.csv", "results/test09/rank/pred_A.csv",
                  "results/test09/rank/pred_B.csv", "results/test09/rank/submission_rank.csv",
                  "results/test09/rank/test_lp_p1e.npz", "results/test09/rank/seam_val/rank_val_probs.csv",
                  "results/test09/rank/seam_val/pred_A.csv", "results/test09/rank/seam_val/pred_B.csv")


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def lf_lines(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n").split(b"\n")


def snapshot(*roots):
    out = {}
    for root in roots:
        for d, _dirs, files in os.walk(root):
            for f in files:
                p = os.path.join(d, f)
                out[os.path.relpath(p, os.path.dirname(root))] = sha(p)
    return out


def run_tool(args, expect=0, env_extra=None):
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONUTF8", None)
    env.update(env_extra or {})
    r = subprocess.run([sys.executable] + list(args), cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    out = r.stdout + r.stderr
    if expect is not None and r.returncode != expect:
        raise AssertionError(f"{' '.join(args[:3])} exited {r.returncode} (expected {expect}):\n{out[-4000:]}")
    return r.returncode, out


def edit_npz(path, fn):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    fn(arrays)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ================================================================ the tree (built once)
class Tree09:
    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            cls.built = cls()
        return cls.built

    def __init__(self):
        given = os.environ.get("C10_T09_TREE")
        self.t7 = None
        if given:
            self.S = os.path.abspath(given)
            if not os.path.exists(os.path.join(self.S, "results", "test07", "rank", "t07_apply_report.json")):
                raise unittest.SkipTest(f"C10_T09_TREE={given} holds no test07 dry run")
        else:
            import test_t07_dryrun as t7d
            self.t7 = t7d.Tree07.get()
            if self.t7.apply_code != 0:
                raise unittest.SkipTest("the 07 dry run did not finish")
            self.S = self.t7.S
        S = self.S
        self.tests = os.path.join(S, "data", "test")
        self.D2 = os.path.join(S, "results", "test02")
        self.O2 = os.path.join(S, "out", "rank")
        self.C = os.path.join(self.D2, "checks")
        self.R4 = os.path.join(S, "results", "test04")
        self.R7 = os.path.join(S, "results", "test07")
        self.R8 = os.path.join(S, "results", "test08")
        self.R9 = os.path.join(S, "results", "test09")
        self.drills = os.path.join(S, "t09_drills")
        for d in (self.R8, self.R9, self.drills):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(self.drills)
        self.snap0 = self.others()
        t0 = time.time()
        run_tool(["tools/t09_fake_tests.py", "from-val", "--root", S, "--t08-results-dir", self.R8])
        self.apply8_code, self.apply8_out = run_tool(["tools/t08_rank.py", "apply", "--dry-run"] + self.dirs8())
        self.snap8 = self.others() | snapshot(self.R8)
        self.apply_code, self.apply_out = run_tool(["tools/t09_rank.py", "apply", "--dry-run"] + self.dirs())
        self.status_code, self.status_out = run_tool(["tools/t09_rank.py", "status", "--dry-run"] + self.dirs())
        self.seconds = round(time.time() - t0, 1)
        self.snap1 = self.others()
        self.snap9 = self.others() | snapshot(self.R8)

    def others(self):
        return snapshot(self.D2, self.O2, self.tests, self.R4, self.R7)

    def dirs8(self, R8=None):
        return ["--results-dir", R8 or self.R8, "--t07-results-dir", self.R7, "--t04-results-dir", self.R4,
                "--t02-results-dir", self.D2, "--t02-out-dir", self.O2, "--tests-dir", self.tests,
                "--checks-dir", self.C]

    def dirs(self, R9=None, R8=None, C=None, T=None):
        return ["--results-dir", R9 or self.R9, "--t08-results-dir", R8 or self.R8, "--t07-results-dir", self.R7,
                "--t04-results-dir", self.R4, "--t02-results-dir", self.D2, "--t02-out-dir", self.O2,
                "--tests-dir", T or self.tests, "--checks-dir", C or self.C]

    def copy_dir(self, src, name):
        d = os.path.join(self.drills, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(src, d)
        return d

    def copy_r8(self, name, with_apply=True):
        """A copy of test08 for one drill (with or without 08's apply outputs)."""
        d = self.copy_dir(self.R8, os.path.join(name, "test08"))
        if not with_apply:
            for p in (os.path.join(d, "rank"), os.path.join(d, "not_used.json"), os.path.join(d, "p1", "gpu_seam.json")):
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.exists(p):
                    os.remove(p)
        return d

    def r9(self, name):
        d = os.path.join(self.drills, name, "test09")
        shutil.rmtree(d, ignore_errors=True)
        return d


def tearDownModule():
    t = Tree09.built
    if t is not None and not os.environ.get("C10_KEEP"):
        shutil.rmtree(t.drills, ignore_errors=True)
    Tree09.built = None
    if t is not None and t.t7 is not None:
        import test_t07_dryrun as t7d
        t7d.tearDownModule()


SKIP = unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (R20, 04, 07, 08 and 09 dry runs, about 12-15 min) or 2")


# ================================================================ 09 against prereg09, R20 / 04 / 07 / 08 against the baseline
@SKIP
class T09DryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree09.get()
        cls.rank = os.path.join(cls.t.R9, "rank")
        cls.rep = read_json(os.path.join(cls.rank, "t09_apply_report.json"))

    def test_apply_exit_0(self):
        self.assertEqual(self.t.apply8_code, 0, self.t.apply8_out[-2000:])
        self.assertEqual(self.t.apply_code, 0, self.t.apply_out[-2000:])
        self.assertEqual((self.rep["mode"], self.rep["exit_code"], self.rep["usable"]), ("dry-run", 0, True))
        self.assertIn("apply: exit 0", self.t.status_out)

    def test_earlier_outputs_untouched(self):
        self.assertEqual(self.t.snap1, self.t.snap0)                   # 08 and 09 never touch test02 / 04 / 07
        self.assertEqual(self.t.snap9, self.t.snap8)                   # 09 never touches test08

    def test_r20_04_07_08_equal_the_baseline(self):
        g = read_json(BASELINE)["files_sha256"]
        self.assertEqual(sum(len(g[p]) for p in ("r20", "t04", "t07", "t08")), 38)
        for part in ("r20", "t04", "t07", "t08"):
            for rel, want in g[part].items():
                with self.subTest(rel):
                    self.assertEqual(sha(os.path.join(self.t.S, *rel.split("/"))), want)

    def test_gpu_seam_and_q08(self):
        g = self.rep["gpu_seam"]
        self.assertTrue(g["passed"] and g["counts_equal"] and g["bitwise_equal_ex"] and g["bitwise_equal_means"])
        self.assertEqual((g["max_abs_std_ex"], g["max_abs_lp_p1e"], g["usable_games"]), (0.0, 0.0, 400))
        self.assertEqual((g["max_abs_std_ex_all_44"], g["frac_lowp_record_only"]["games_jumped"]), (0.0, 0))
        q = self.rep["q08_vs_08"]
        self.assertTrue(q["byte_equal"] and q["equal"] and q["p1_ok_equal_t08"] and q["usable_equal_t08"])
        self.assertEqual(q["lp_p1_max_abs_diff_vs_t08_record_only"], 0.0)

    def test_local_seam_byte_identical_to_prereg09(self):
        seam = os.path.join(self.rank, "seam_val")
        for n in ("rank_val_probs.csv", "pred_B.csv", "pred_A.csv"):
            self.assertEqual(lf_lines(os.path.join(seam, n)), lf_lines(os.path.join(P09, n)), n)
        sl = read_json(os.path.join(seam, "seam_local.json"))
        self.assertEqual((sl["verdict"], sl["required_byte_identical"], sl["max_abs_diff"]), ("passed", True, 0.0))
        self.assertTrue(sl["usable_equal_to_prereg09"])
        self.assertTrue(self.rep["p1_direction"]["seam_val_record_only"]["equal_to_prereg09_apply"])

    def test_test_rows_equal_prereg09_validation_rows(self):
        ref, got = lf_lines(os.path.join(P09, "rank_val_probs.csv")), lf_lines(os.path.join(self.rank,
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
            ref, got = lf_lines(os.path.join(P09, ref_name)), lf_lines(os.path.join(self.rank, name))
            self.assertEqual(len(got), 802)
            for s, (lo, hi) in RANK_ROWS.items():
                for i in range(lo, hi):
                    self.assertEqual(got[1 + i], PREFIX[s].encode() + ref[1 + i][len(b"val_"):], (name, s, i))

    def test_lp_equal_val_lp(self):
        z = dict(np.load(os.path.join(self.rank, "test_lp_p1e.npz"), allow_pickle=False))
        v = dict(np.load(os.path.join(P09, "val_lp_p1e.npz"), allow_pickle=False))
        n = len(z["lp"])
        self.assertEqual(n, 9602)
        self.assertLessEqual(float(np.max(np.abs(z["lp"] - v["lp"][:n]))), 1e-12)
        self.assertTrue(np.array_equal(z["p1_ok"], v["p1_ok"][:n]))
        self.assertEqual(str(z["map_sha256"]), str(v["map_sha256"]))

    def test_submission_and_checks(self):
        sub = os.path.join(self.rank, "submission_rank.csv")
        self.assertEqual(sha(sub), sha(os.path.join(self.rank, "pred_B.csv")))
        qf = [os.path.join(self.t.tests, RANK_FILES[s]) for s in ("public", "private")]
        self.assertEqual(check_submission.check("rank", sub, qf, official=True), [])
        self.assertEqual(self.rep["outputs_sha256"]["submission_rank.csv"], sha(sub))
        for s in ("public", "private"):
            self.assertEqual(self.rep["p1_failures"][s]["failed_or_no_move"], 0)
            self.assertTrue(self.rep["p1_direction"][s]["passed"])
            self.assertAlmostEqual(self.rep["p1_direction"][s]["threshold"], 0.48425802602327594, places=15)
        self.assertEqual(set(self.rep["rule_b_differences"]), {"08", "07", "04", "R20"})
        self.assertEqual(self.rep["combo"]["id"], "Q09")

    def test_inputs_bound_cover_official_log(self):
        import official_log as ol
        self.assertEqual([r for r in ol.T09_INPUTS_REQUIRED if r not in self.rep["inputs_bound"]], [])

    def test_golden(self):
        if not os.path.exists(GOLDEN09):
            self.skipTest(f"{GOLDEN09} not written yet (first dry run)")
        g = read_json(GOLDEN09)["files_sha256"]
        self.assertEqual(sorted(g["t09"]), sorted(GOLDEN09_FILES))
        for rel, want in g["t09"].items():
            with self.subTest(rel):
                self.assertEqual(sha(os.path.join(self.t.S, *rel.split("/"))), want)


# ================================================================ negative drills
@SKIP
class T09Drills(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree09.get()

    def apply(self, R9, expect, R8=None, C=None, T=None, extra=(), env=None):
        _code, out = run_tool(["tools/t09_rank.py", "apply", "--dry-run"] + self.t.dirs(R9, R8, C, T) + list(extra),
                              expect=expect, env_extra=env)
        return out

    def verdict(self, R9):
        p = os.path.join(R9, "rank", "attempt_verdict.json")
        return read_json(p) if os.path.exists(p) else read_json(os.path.join(R9, "not_used.json"))

    def refresh_p1_report(self, r8, s):
        p = os.path.join(r8, "p1", f"p1_{s}_report.json")
        rep = read_json(p)
        for ch in rep["chunks"]:
            ch["sha256"] = sha(os.path.join(r8, "p1_data", s, ch["file"]))
        write_json(p, rep)

    def test_gpu_seam_ex(self):
        r8 = self.t.copy_r8("seam_ex", with_apply=False)
        c = os.path.join(r8, "p1_data", "seam", "chunk_00000.npz")

        def f(a):
            sel = a["pos_game"] == 7
            a["pos_logp"][sel] = (a["pos_logp"][sel] - 1.0).astype(np.float32)   # EX of game 7 moves, means not
        edit_npz(c, f)
        self.refresh_p1_report(r8, "seam")
        run_tool(["tools/t08_rank.py", "apply", "--dry-run"] + self.t.dirs8(r8))          # 08 itself still passes
        r9 = self.t.r9("seam_ex")
        self.apply(r9, 4, R8=r8)
        v = self.verdict(r9)
        self.assertEqual((v["stage"], v["reason"]), ("gpu_seam", "gpu_seam"))
        g = read_json(os.path.join(r9, "gpu_seam.json"))
        self.assertGreater(g["max_abs_std_ex"], 0.05)                     # the judged 42 features (rev 4)
        self.assertEqual(g["frac_lowp_record_only"]["features"], ["t_frac_lowp", "o_frac_lowp"])

    def test_chunk_changed_after_08(self):
        r8 = self.t.copy_r8("chunk_changed")
        edit_npz(os.path.join(r8, "p1_data", "private", "chunk_00003.npz"),
                 lambda a: a.__setitem__("note", np.array("x")))
        self.assertIn("differs from the SHA-256 its report lists", self.apply(self.t.r9("chunk_changed"), 2, R8=r8))

    def test_p1_report_changed_after_08(self):
        r8 = self.t.copy_r8("report_changed")
        p = os.path.join(r8, "p1", "p1_public_report.json")
        rep = read_json(p)
        rep["note"] = "changed"
        write_json(p, rep)
        out = self.apply(self.t.r9("report_changed"), 2, R8=r8)
        self.assertIn("test08's bound inputs changed", out)

    def test_q08_one_value_off(self):
        r8 = self.t.copy_r8("q08")
        p = os.path.join(r8, "rank", "rank_test_probs.csv")
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        i = next(k for k, r in enumerate(rows) if k and r[3])
        rows[i][3] = repr(float(rows[i][3]) + 1e-9)
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        rp = os.path.join(r8, "rank", "t08_apply_report.json")
        rep = read_json(rp)
        rep["outputs_sha256"]["rank_test_probs.csv"] = sha(p)
        write_json(rp, rep)
        r9 = self.t.r9("q08")
        self.apply(r9, 4, R8=r8)
        v = self.verdict(r9)
        self.assertEqual(v["stage"], "q08_vs_08")
        self.assertFalse(v["details"]["equal"])

    def test_08_output_changed(self):
        r8 = self.t.copy_r8("out_changed")
        with open(os.path.join(r8, "rank", "submission_rank.csv"), "ab") as f:
            f.write(b"\n")
        self.assertIn("test08 outputs differ from its report", self.apply(self.t.r9("out_changed"), 2, R8=r8))

    def test_08_not_used_and_not_done(self):
        r8 = self.t.copy_r8("t08_not_used", with_apply=False)
        r9 = self.t.r9("t08_not_done")
        self.assertIn("test08 apply has not finished", self.apply(r9, 2, R8=r8))
        write_json(os.path.join(r8, "not_used.json"), {"format": "c10-test08 not used v1", "mode": "dry-run",
                                                       "exit_code": 3, "stage": "p1_direction", "reason": "drill"})
        r9 = self.t.r9("t08_not_used")
        self.apply(r9, 3, R8=r8)
        self.assertEqual(self.verdict(r9)["reason"], "t08_not_used")
        r8 = self.t.copy_r8("t08_unusable")
        rp = os.path.join(r8, "rank", "t08_apply_report.json")
        rep = read_json(rp)
        rep.update(usable=False, exit_code=3)
        write_json(rp, rep)
        self.apply(self.t.r9("t08_unusable"), 3, R8=r8)

    def test_k6_09(self):
        r9 = self.t.r9("k6")
        self.apply(r9, 3, extra=["--now", "2026-11-20T00:00:00+08:00"])
        self.assertEqual(self.verdict(r9)["reason"], "k6_09")
        out = self.apply(self.t.R9, 2, extra=["--now", "2026-11-20T00:00:00+08:00", "--rerun", "--reason", "drill"])
        self.assertIn("stays valid", out)

    def test_fail_opening(self):
        c = self.t.copy_dir(self.t.C, "checks_fo")
        p = os.path.join(c, "rank", "t1.json")
        t1 = read_json(p)
        t1["verdict"] = "FAIL_OPENING"
        write_json(p, t1)
        r9 = self.t.r9("fail_opening")
        self.apply(r9, 3, C=c)
        self.assertTrue(os.path.exists(os.path.join(r9, "not_used.json")))

    def test_outputs_present(self):
        self.assertIn("never overwritten", self.apply(self.t.R9, 2))
        self.assertIn("registered decision", self.apply(self.t.R9, 2, extra=["--rerun", "--reason", "drill"]))
        self.assertIn("rev 5 or later", self.apply(self.t.R9, 2, extra=["--rerun", "--reason", "drill",
                                                                        "--bug-revision", "4"]))

    def test_blas_env(self):
        self.assertIn("BLAS thread variables", self.apply(self.t.r9("blas"), 2, env={"MKL_NUM_THREADS": "2"}))

    def test_raw_tests_dir(self):
        self.assertIn("never data/test/raw", self.apply(self.t.r9("raw"), 2, T=os.path.join(self.t.tests, "raw")))

    def test_real_mode_drill_flags_refused(self):
        _code, out = run_tool(["tools/t09_rank.py", "apply", "--now", "2026-11-20T00:00"], expect=2)
        self.assertIn("drills only", out)


if __name__ == "__main__":
    unittest.main()
