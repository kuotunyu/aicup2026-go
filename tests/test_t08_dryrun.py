"""End-to-end dry run of the 08 test-day kit on the from-val fake tree (PREREG_08 5.1 step 3, 5.2; rev 3 section 13).
Every tool is called through its command line, as on test day.

    C10_SLOW=1 or 2   the tree of tests/test_t07_dryrun.py (R20, 04 and 07 dry runs in a temporary folder outside
        the repository; about 8-10 min), or C10_T08_TREE=<S> to reuse a tree whose S/results/test07/rank/
        t07_apply_report.json exists (test02 / test04 / test07 dry runs by the same commands). Then:
        08: t08_fake_tests.py from-val (seam = validation games 0-399, public / private = the validation P1 rows of
            questions 1-400 / 401-800, copied from data/prereg08/val chunks) -> t08_rank.py apply / status --dry-run.
            The R20 / 04 / 07 outputs equal results/test07/design/t07_fake_golden.json byte for byte (PREREG_08 5.1
            step 3; the 08 kit never changes them); the local seam is byte-identical to results/prereg08/rank
            (rank_val_probs.csv, pred_B.csv, pred_A.csv); the 800 test rows equal prereg08's first 800 validation rows
            under the fake prefixes within 1e-12 with every rule-B and rule-A level equal (lp_P1TO is one matrix
            product per set here and one of 19,156 rows in prereg08: 48 of 9,602 games differ by <= 4e-15); the
            submission passes check_submission --official; the move cross-checks are 0; the 08 outputs equal
            results/test08/design/t08_fake_golden.json; t08 never changes a file of test02 / test04 / test07.
        Negative drills (each on copies, never on the tree itself): GPU seam (features), a P1 file that differs from
            its report, another environment, P1 before R20's upload, Q02H one byte off 07, test07's bound inputs
            changed, the move cross-check, P1 failures 23 / 24 of 4,787, the direction check, 07 not used, 07 not
            done, K6-08, FAIL_OPENING, outputs present, a raw tests folder, and for p1 (fake scorer, CPU): a stale
            chunk, keys that differ from test02's / test07's, public before seam, --max-games in the real mode.
The fake tree holds validation games: only P1 seams and apply run on it. Nothing here opens a label file or runs
tools/score.py; nothing is written inside the repository.
Run: C10_SLOW=1 python -m pytest tests/test_t08_dryrun.py   (C10_KEEP=1 keeps the tree)
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
GOLDEN07 = os.path.join(REPO, "results", "test07", "design", "t07_fake_golden.json")
GOLDEN08 = os.path.join(REPO, "results", "test08", "design", "t08_fake_golden.json")
P08 = os.path.join(REPO, "results", "prereg08", "rank")
PREFIX = {"public": "pub_", "private": "priv_"}
RANK_ROWS = {"public": (0, 400), "private": (400, 800)}
RANK_FILES = {"public": "rank_prediction_test_public.csv", "private": "rank_prediction_test_private.csv"}
GOLDEN08_FILES = ("results/test08/p1/seam_p1.npz", "results/test08/p1/public_p1.npz",
                  "results/test08/p1/private_p1.npz", "results/test08/rank/rank_test_probs.csv",
                  "results/test08/rank/pred_A.csv", "results/test08/rank/pred_B.csv",
                  "results/test08/rank/submission_rank.csv", "results/test08/rank/test_lp_p1.npz",
                  "results/test08/rank/seam_val/rank_val_probs.csv", "results/test08/rank/seam_val/pred_A.csv",
                  "results/test08/rank/seam_val/pred_B.csv")


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


def run_tool(args, expect=0):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONUTF8", None)
    r = subprocess_run([sys.executable] + list(args), env)
    out = r.stdout + r.stderr
    if expect is not None and r.returncode != expect:
        raise AssertionError(f"{' '.join(args[:3])} exited {r.returncode} (expected {expect}):\n{out[-4000:]}")
    return r.returncode, out


def subprocess_run(cmd, env):
    import subprocess
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def edit_npz(path, fn):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    fn(arrays)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def refresh_report(r8, s):
    """p1_<set>_report.json follows the edited npz and chunks (so only the intended check can fail)."""
    p = os.path.join(r8, "p1", f"p1_{s}_report.json")
    rep = read_json(p)
    rep["output"]["sha256"] = sha(os.path.join(r8, "p1", f"{s}_p1.npz"))
    for ch in rep["chunks"]:
        ch["sha256"] = sha(os.path.join(r8, "p1_data", s, ch["file"]))
    write_json(p, rep)


def edit_set(r8, s, fn, chunks=(0,)):
    """fn(arrays) on the given chunks of a set and (rows of chunk 0 only) on <set>_p1.npz; then the report."""
    for ci in chunks:
        edit_npz(os.path.join(r8, "p1_data", s, f"chunk_{ci:05d}.npz"), fn)
    if chunks == (0,):
        edit_npz(os.path.join(r8, "p1", f"{s}_p1.npz"), fn)
    refresh_report(r8, s)


# ================================================================ the tree (built once)
class Tree08:
    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            cls.built = cls()
        return cls.built

    def __init__(self):
        given = os.environ.get("C10_T08_TREE")
        self.t7 = None
        if given:
            self.S = os.path.abspath(given)
            if not os.path.exists(os.path.join(self.S, "results", "test07", "rank", "t07_apply_report.json")):
                raise unittest.SkipTest(f"C10_T08_TREE={given} holds no test07 dry run")
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
        self.drills = os.path.join(S, "t08_drills")
        shutil.rmtree(self.R8, ignore_errors=True)
        shutil.rmtree(self.drills, ignore_errors=True)
        os.makedirs(self.drills)
        self.snap0 = self.others()
        t0 = time.time()
        run_tool(["tools/t08_fake_tests.py", "from-val", "--root", S, "--results-dir", self.R8])
        self.apply_code, self.apply_out = run_tool(["tools/t08_rank.py", "apply", "--dry-run"] + self.dirs())
        self.status_code, self.status_out = run_tool(["tools/t08_rank.py", "status", "--dry-run"] + self.dirs())
        self.seconds = round(time.time() - t0, 1)
        self.snap1 = self.others()

    def others(self):
        return snapshot(self.D2, self.O2, self.tests, self.R4, self.R7)

    def dirs(self, R8=None, R7=None, R4=None, C=None, T=None):
        return ["--results-dir", R8 or self.R8, "--t07-results-dir", R7 or self.R7, "--t04-results-dir", R4 or self.R4,
                "--t02-results-dir", self.D2, "--t02-out-dir", self.O2, "--tests-dir", T or self.tests,
                "--checks-dir", C or self.C]

    def copy_r8(self, name):
        """A copy of the fake P1 sets and chunks only (no apply outputs) for one drill."""
        d = os.path.join(self.drills, name, "test08")
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)
        os.makedirs(os.path.join(d, "p1"))
        for f in os.listdir(os.path.join(self.R8, "p1")):
            if f.endswith("_p1.npz") or (f.startswith("p1_") and f.endswith("_report.json")):
                shutil.copyfile(os.path.join(self.R8, "p1", f), os.path.join(d, "p1", f))
        shutil.copytree(os.path.join(self.R8, "p1_data"), os.path.join(d, "p1_data"))
        return d

    def copy_dir(self, src, name):
        d = os.path.join(self.drills, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(src, d)
        return d


def tearDownModule():
    t = Tree08.built
    if t is not None and not os.environ.get("C10_KEEP"):
        shutil.rmtree(t.drills, ignore_errors=True)
    Tree08.built = None
    if t is not None and t.t7 is not None:
        import test_t07_dryrun as t7d
        t7d.tearDownModule()


SKIP = unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (R20, 04, 07 and 08 dry runs, about 12-15 min) or 2")


# ================================================================ 08 against prereg08, R20 / 04 / 07 against the golden
@SKIP
class T08DryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree08.get()
        cls.rank = os.path.join(cls.t.R8, "rank")
        cls.rep = read_json(os.path.join(cls.rank, "t08_apply_report.json"))

    def test_apply_exit_0(self):
        self.assertEqual(self.t.apply_code, 0, self.t.apply_out[-2000:])
        self.assertEqual((self.rep["mode"], self.rep["exit_code"], self.rep["usable"]), ("dry-run", 0, True))
        self.assertIn("apply: exit 0", self.t.status_out)

    def test_test02_test04_test07_untouched(self):
        self.assertEqual(self.t.snap1, self.t.snap0)

    def test_r20_04_07_equal_the_07_golden(self):
        g = read_json(GOLDEN07)["files_sha256"]
        for part in ("r20", "t04", "t07"):
            for rel, want in g[part].items():
                with self.subTest(rel):
                    self.assertEqual(sha(os.path.join(self.t.S, *rel.split("/"))), want)

    def test_local_seam_byte_identical_to_prereg08(self):
        seam = os.path.join(self.rank, "seam_val")
        for n in ("rank_val_probs.csv", "pred_B.csv", "pred_A.csv"):
            self.assertEqual(lf_lines(os.path.join(seam, n)), lf_lines(os.path.join(P08, n)), n)
        sl = read_json(os.path.join(seam, "seam_local.json"))
        self.assertEqual((sl["verdict"], sl["required_byte_identical"], sl["max_abs_diff"]), ("passed", True, 0.0))
        self.assertTrue(sl["usable_equal_to_prereg08"])
        self.assertTrue(sl["q02h_equal_to_t07_seam_val_record_only"])
        self.assertTrue(self.rep["p1_direction"]["seam_val_record_only"]["equal_to_prereg08_apply"])

    def test_test_rows_equal_prereg08_validation_rows(self):
        ref, got = lf_lines(os.path.join(P08, "rank_val_probs.csv")), lf_lines(os.path.join(self.rank,
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
            ref, got = lf_lines(os.path.join(P08, ref_name)), lf_lines(os.path.join(self.rank, name))
            self.assertEqual(len(got), 802)
            for s, (lo, hi) in RANK_ROWS.items():
                for i in range(lo, hi):
                    self.assertEqual(got[1 + i], PREFIX[s].encode() + ref[1 + i][len(b"val_"):], (name, s, i))

    def test_submission_and_checks(self):
        sub = os.path.join(self.rank, "submission_rank.csv")
        self.assertEqual(sha(sub), sha(os.path.join(self.rank, "pred_B.csv")))
        qf = [os.path.join(self.t.tests, RANK_FILES[s]) for s in ("public", "private")]
        self.assertEqual(check_submission.check("rank", sub, qf, official=True), [])
        self.assertEqual(self.rep["outputs_sha256"]["submission_rank.csv"], sha(sub))
        for s in ("public", "private"):
            self.assertEqual(self.rep["move_cross_check"][s]["mismatches"], 0)
            self.assertGreater(self.rep["move_cross_check"][s]["vs_t07_hs"]["compared"], 4700)
            self.assertGreater(self.rep["move_cross_check"][s]["vs_t04_o4"]["compared"], 4700)
            self.assertEqual(self.rep["p1_failures"][s]["failed_or_no_move"], 0)
            self.assertTrue(self.rep["p1_direction"][s]["passed"])
        self.assertTrue(self.rep["q02h_vs_07"]["equal"])
        self.assertTrue(self.rep["gpu_seam"]["bitwise_equal_means"])
        self.assertEqual(set(self.rep["rule_b_differences"]), {"07", "04", "R20"})

    def test_golden(self):
        if not os.path.exists(GOLDEN08):
            self.skipTest(f"{GOLDEN08} not written yet (first dry run)")
        g = read_json(GOLDEN08)["files_sha256"]["t08"]
        self.assertEqual(sorted(g), sorted(GOLDEN08_FILES))
        for rel, want in g.items():
            with self.subTest(rel):
                self.assertEqual(sha(os.path.join(self.t.S, *rel.split("/"))), want)


# ================================================================ negative drills
@SKIP
class T08Drills(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree08.get()

    def apply(self, R8, expect, R7=None, R4=None, C=None, T=None, extra=()):
        _code, out = run_tool(["tools/t08_rank.py", "apply", "--dry-run"] + self.t.dirs(R8, R7, R4, C, T)
                              + list(extra), expect=expect)
        return out

    def verdict(self, R8):
        p = os.path.join(R8, "rank", "attempt_verdict.json")
        return read_json(p) if os.path.exists(p) else read_json(os.path.join(R8, "not_used.json"))

    def test_gpu_seam_features(self):
        r8 = self.t.copy_r8("seam_feat")

        def f(a):
            if "means" in a:
                a["means"][0, 0, 5] += 1.0          # standardised: 1 / sd >> 0.05
        edit_set(r8, "seam", f)
        self.apply(r8, 4)
        v = self.verdict(r8)
        self.assertEqual((v["stage"], v["reason"]), ("gpu_seam", "gpu_seam"))

    def test_p1_file_differs_from_report(self):
        r8 = self.t.copy_r8("p1_changed")
        with open(os.path.join(r8, "p1", "public_p1.npz"), "ab") as f:
            f.write(b"\0")
        self.assertIn("differs from its report", self.apply(r8, 2))

    def test_chunk_changed(self):
        r8 = self.t.copy_r8("chunk_changed")
        with open(os.path.join(r8, "p1_data", "private", "chunk_00003.npz"), "ab") as f:
            f.write(b"\0")
        self.assertIn("P1 chunks missing or changed", self.apply(r8, 2))

    def test_chunk_run_settings_differ(self):
        """rev 3 section 13 addendum: apply checks every chunk's scorer settings / key and question file / code."""
        r8 = self.t.copy_r8("chunk_meta")
        edit_set(r8, "public", lambda a: a.__setitem__("precision", np.array("fp32")), chunks=(3,))
        out = self.apply(r8, 2)
        self.assertIn("P1 chunks of another run", out)
        self.assertIn("precision", out)
        r8 = self.t.copy_r8("chunk_questions")
        edit_set(r8, "private", lambda a: a.__setitem__("questions_sha256", np.array("q" * 64)), chunks=(1,))
        self.assertIn("questions_sha256", self.apply(r8, 2))

    def test_p1_code_differs_between_sets(self):
        r8 = self.t.copy_r8("p1_code")
        n = len(read_json(os.path.join(r8, "p1", "p1_private_report.json"))["chunks"])

        def f(a):
            c = json.loads(str(a["code_sha256"]))
            c["results/prereg08/p1/p1_lib.py"] = "0" * 64
            a["code_sha256"] = np.array(json.dumps(c, sort_keys=True))
        edit_set(r8, "private", f, chunks=tuple(range(n)))
        self.assertIn("different P1 code", self.apply(r8, 2))

    def test_environment_differs(self):
        r8 = self.t.copy_r8("env")
        p = os.path.join(r8, "p1", "p1_private_report.json")
        rep = read_json(p)
        rep["environment"]["cudnn"] = 92000
        write_json(p, rep)
        self.apply(r8, 4)
        self.assertEqual(self.verdict(r8)["reason"], "p1_environment")

    def test_p1_before_r20_upload(self):
        r8 = self.t.copy_r8("before_r20")
        self.assertIn("before R20's first upload", self.apply(r8, 2, extra=["--r20-uploaded-at",
                                                                            "2099-01-01T00:00:00+08:00"]))

    def test_q02h_one_byte_off_07(self):
        r8 = self.t.copy_r8("q02h")
        r7 = self.t.copy_dir(self.t.R7, "r7_q02h")
        p = os.path.join(r7, "rank", "rank_test_probs.csv")
        with open(p, "rb") as f:
            b = bytearray(f.read())
        i = b.index(b"0.", 200)
        b[i + 5] = ord("1") if b[i + 5] != ord("1") else ord("2")
        with open(p, "wb") as f:
            f.write(bytes(b))
        self.apply(r8, 4, R7=r7)
        self.assertEqual(self.verdict(r8)["stage"], "q02h_vs_07")

    def test_t07_inputs_changed(self):
        r8 = self.t.copy_r8("t07_inputs")
        r7 = self.t.copy_dir(self.t.R7, "r7_inputs")
        p = os.path.join(r7, "hs", "gpu_seam.json")
        rep = read_json(p)
        rep["note"] = "changed"
        write_json(p, rep)
        self.assertIn("test07's bound inputs changed", self.apply(r8, 2, R7=r7))

    def test_move_cross_check(self):
        r8 = self.t.copy_r8("cross")
        edit_set(r8, "public", lambda a: a["n_moves"].__setitem__(5, a["n_moves"][5] + 1))
        self.apply(r8, 4)
        self.assertEqual(self.verdict(r8)["stage"], "move_cross_check")
        out = self.apply(r8, 2, extra=["--rerun", "--reason", "drill"])     # only after a PREREG_08 revision
        self.assertIn("--bug-revision", out)

    def test_failures_23_and_24_of_4787(self):
        for n, code in ((23, 0), (24, 3)):
            with self.subTest(n):
                r8 = self.t.copy_r8(f"fail_{n}")

                def f(a, n=n):
                    a["failed"][:n] = True
                edit_set(r8, "public", f)
                self.apply(r8, code)
                if code == 3:
                    v = self.verdict(r8)
                    self.assertEqual((v["stage"], v["exit_code"]), ("p1_failures", 3))
                    self.assertFalse(os.path.exists(os.path.join(r8, "rank", "submission_rank.csv")))
                else:
                    rep = read_json(os.path.join(r8, "rank", "t08_apply_report.json"))
                    self.assertEqual(rep["p1_failures"]["public"]["failed_or_no_move"], 23)

    def test_direction_check(self):
        r8 = self.t.copy_r8("direction")
        n_chunks = len(read_json(os.path.join(r8, "p1", "p1_private_report.json"))["chunks"])

        def f(a):
            import numpy as np
            a["means"] = a["means"][np.random.default_rng(0).permutation(len(a["means"]))]   # games shuffled
        edit_set(r8, "private", f, chunks=tuple(range(n_chunks)))
        self.apply(r8, 3)
        self.assertEqual(self.verdict(r8)["stage"], "p1_direction")

    def test_07_not_used_and_not_done(self):
        r8 = self.t.copy_r8("t07_not_used")
        r7 = self.t.copy_dir(self.t.R7, "r7_not_used")
        p = os.path.join(r7, "rank", "t07_apply_report.json")
        rep = read_json(p)
        rep.update(usable=False, exit_code=3)
        write_json(p, rep)
        self.apply(r8, 3, R7=r7)
        self.assertEqual(self.verdict(r8)["reason"], "t07_not_used")
        r8 = self.t.copy_r8("t07_not_done")
        os.remove(p)
        self.assertIn("test07 apply has not finished", self.apply(r8, 2, R7=r7))

    def test_k6_08(self):
        r8 = self.t.copy_r8("k6")
        self.apply(r8, 3, extra=["--now", "2026-11-20T00:00:00+08:00"])
        self.assertEqual(self.verdict(r8)["reason"], "k6_08")
        out = self.apply(self.t.R8, 2, extra=["--now", "2026-11-20T00:00:00+08:00", "--rerun", "--reason", "drill"])
        self.assertIn("stays valid", out)

    def test_fail_opening(self):
        r8 = self.t.copy_r8("fail_opening")
        c = self.t.copy_dir(self.t.C, "checks_fo")
        p = os.path.join(c, "rank", "t1.json")
        t1 = read_json(p)
        t1["verdict"] = "FAIL_OPENING"
        write_json(p, t1)
        self.apply(r8, 3, C=c)
        self.assertTrue(os.path.exists(os.path.join(r8, "not_used.json")))

    def test_outputs_present(self):
        self.assertIn("never overwritten", self.apply(self.t.R8, 2))
        self.assertIn("registered decision", self.apply(self.t.R8, 2, extra=["--rerun", "--reason", "drill"]))

    def test_raw_tests_dir(self):
        r8 = self.t.copy_r8("raw")
        self.assertIn("never data/test/raw", self.apply(r8, 2, T=os.path.join(self.t.tests, "raw")))


@SKIP
class T08P1Drills(unittest.TestCase):
    """p1 with the fake scorer (CPU) on a few games: the refusals before and around the GPU."""

    @classmethod
    def setUpClass(cls):
        cls.t = Tree08.get()

    def p1(self, R8, s, expect, T=None, extra=()):
        return run_tool(["tools/t08_rank.py", "p1", "--set", s, "--dry-run", "--fake-scorer", "--workers", "2",
                         "--max-games", "6"] + self.t.dirs(R8, T=T) + list(extra), expect=expect)[1]

    def fresh(self, name):
        r8 = os.path.join(self.t.drills, name, "test08")
        shutil.rmtree(os.path.dirname(r8), ignore_errors=True)
        return r8

    def test_seam_public_and_stale_chunk(self):
        r8 = self.fresh("p1_stale")
        self.p1(r8, "seam", 0)
        out = self.p1(r8, "public", 0)
        self.assertIn('"todo": 1', out)
        rep = read_json(os.path.join(r8, "p1", "p1_public_report.json"))
        self.assertEqual((rep["games"], rep["failed_or_no_move"], rep["fake_scorer"]), (6, 0, True))
        self.assertTrue(rep["keys"]["t07_hs_keys"]["equal"])
        os.remove(os.path.join(r8, "p1", "seam_p1.npz"))
        os.remove(os.path.join(r8, "p1", "p1_seam_report.json"))
        out = self.p1(r8, "seam", 2, extra=["--r20-uploaded-at", "2099-01-01T00:00:00+08:00"])
        self.assertIn("before R20's first upload", out)
        self.assertTrue(os.path.exists(os.path.join(r8, "p1_data", "seam", "chunk_00000.npz")), "never deleted")
        out = self.p1(r8, "seam", 0)                        # resuming in a valid order reuses the chunk
        self.assertIn('"todo": 0', out)
        self.assertIn("exist; rerun only", self.p1(r8, "seam", 2))

    def test_stop_file_exits_5(self):
        """rev 3 section 13 addendum: a STOP file ends p1 with exit 5 (a chained launcher stops; the GPU is free);
        deleting STOP and running the same line resumes."""
        r8 = self.fresh("p1_stop")
        d = os.path.join(r8, "p1_data", "seam")
        os.makedirs(d)
        open(os.path.join(d, "STOP"), "w").close()
        out = self.p1(r8, "seam", 5)
        self.assertIn("STOPPED (exit 5)", out)
        self.assertFalse(os.path.exists(os.path.join(r8, "p1", "seam_p1.npz")))
        self.assertEqual([f for f in os.listdir(d) if f.startswith("chunk_")], [])
        os.remove(os.path.join(d, "STOP"))
        self.assertIn('"todo": 1', self.p1(r8, "seam", 0))

    def test_chunk_of_other_question_file_or_code_not_reused(self):
        r8 = self.fresh("p1_reuse")
        self.p1(r8, "seam", 0)
        c = os.path.join(r8, "p1_data", "seam", "chunk_00000.npz")
        keep = c + ".keep"
        shutil.copyfile(c, keep)
        for k, v in (("questions_sha256", "q" * 64), ("code_sha256", json.dumps({"x": "0" * 64}))):
            for f in ("seam_p1.npz", "p1_seam_report.json"):
                p = os.path.join(r8, "p1", f)
                if os.path.exists(p):
                    os.remove(p)
            shutil.copyfile(keep, c)
            edit_npz(c, lambda a: a.__setitem__(k, np.array(v)))
            out = self.p1(r8, "seam", 2)
            self.assertIn("differs from this run", out)
            self.assertIn(k, out)
        os.remove(keep)

    def test_keys_differ_from_test02(self):
        r8 = self.fresh("p1_keys")
        self.p1(r8, "seam", 0)
        t = self.t.copy_dir(self.t.tests, "tests_keys")
        p = os.path.join(t, RANK_FILES["public"])
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        c1 = rows[0].index("color_1")
        rows[1][c1] = "W" if rows[1][c1] == "B" else "B"
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\r\n").writerows(rows)
        self.assertIn("differ", self.p1(r8, "public", 2, T=t))

    def test_keys_differ_from_test07(self):
        r8 = self.fresh("p1_keys07")
        r7 = self.t.copy_dir(self.t.R7, "r7_keys")
        edit_npz(os.path.join(r7, "hs", "seam_hs.npz"), lambda a: a["slot"].__setitem__(2, a["slot"][2] + 1))
        out = run_tool(["tools/t08_rank.py", "p1", "--set", "seam", "--dry-run", "--fake-scorer", "--workers", "2",
                        "--max-games", "6"] + self.t.dirs(r8, R7=r7), expect=2)[1]
        self.assertIn("slots differ from test07", out)

    def test_public_before_seam_refused(self):
        r8 = self.fresh("p1_order")
        self.assertIn("run p1 --set seam first", self.p1(r8, "public", 2))

    def test_real_mode_drill_flags_refused(self):
        _code, out = run_tool(["tools/t08_rank.py", "p1", "--set", "seam", "--max-games", "5"], expect=2)
        self.assertIn("drills only", out)


if __name__ == "__main__":
    unittest.main()
