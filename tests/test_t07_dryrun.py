"""End-to-end dry run of the 07 test-day kit on the from-val fake tree (results/prereg07/design/spec.md 9). Every tool
is called through its command line, as on test day.

    C10_SLOW=1 or 2   the tree of tests/test_t04_dryrun.py (R20 and 04 dry runs in a temporary folder outside the
        repository; about 5-8 min), or C10_T07_TREE=<S> to reuse a tree built by the same commands (its
        S/results/test04/rank/t04_apply_report.json must exist). Then:
        07 (spec 9 "07 對得回 prereg07"): t07_fake_tests.py from-val (seam = val_hs.npz rows 0-399 + val_seam_moves.npz,
            public / private = the val_hs rows of validation questions 1-400 / 401-800) -> t07_rank.py apply / status
            --dry-run. The local seam is byte-identical to results/prereg07/rank (rank_val_probs.csv, pred_B.csv,
            pred_A.csv); the 800 test rows equal prereg07's first 800 validation rows under the fake prefixes; the
            submission passes check_submission --official; the move cross-check against the dry-run o4_<set>.npz is 0;
            every file equals results/test07/design/t07_fake_golden.json; t07 never changes a file of test02 / test04.
        Negative drills (spec 9 "負面演練"; each on copies, never on the tree itself): GPU seam per move / per profile,
            an HS file that differs from its report, another environment, HS before R20's upload, Q02 one byte off 04,
            test04's bound inputs changed, the move cross-check, HS failures 23 / 24 of 4,787, the direction check, 04
            not used, 04 not done, K6-07, FAIL_OPENING, outputs present, a raw tests folder, and for hs (fake scorer,
            CPU): a stale chunk, keys that differ from test02's, --max-games in the real mode.
The fake tree holds validation games: only HS seams and apply run on it. Nothing here opens a label file or runs
tools/score.py; nothing is written inside the repository.
Run: C10_SLOW=1 python -m pytest tests/test_t07_dryrun.py   (C10_KEEP=1 keeps the tree)
"""

import builtins
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import check_submission  # noqa: E402

SLOW = os.environ.get("C10_SLOW", "")
GOLDEN = os.path.join(REPO, "results", "test07", "design", "t07_fake_golden.json")
P07 = os.path.join(REPO, "results", "prereg07", "rank")
PREFIX = {"public": "pub_", "private": "priv_"}
RANK_ROWS = {"public": (0, 400), "private": (400, 800)}
RANK_FILES = {"public": "rank_prediction_test_public.csv", "private": "rank_prediction_test_private.csv"}


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process (spec 0)"


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
    r = subprocess.run([sys.executable] + list(args), cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    out = r.stdout + r.stderr
    if expect is not None and r.returncode != expect:
        raise AssertionError(f"{' '.join(args[:3])} exited {r.returncode} (expected {expect}):\n{out[-4000:]}")
    return r.returncode, out


def edit_npz(path, fn):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    fn(arrays)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def refresh_report(r7, s):
    """hs_<set>_report.json follows the edited npz (so only the intended check can fail)."""
    p = os.path.join(r7, "hs", f"hs_{s}_report.json")
    rep = read_json(p)
    rep["output"]["sha256"] = sha(os.path.join(r7, "hs", f"{s}_hs.npz"))
    write_json(p, rep)


# ================================================================ the tree (built once)
class Tree07:
    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            cls.built = cls()
        return cls.built

    def __init__(self):
        given = os.environ.get("C10_T07_TREE")
        self.t4 = None
        if given:
            self.S = os.path.abspath(given)
            if not os.path.exists(os.path.join(self.S, "results", "test04", "rank", "t04_apply_report.json")):
                raise unittest.SkipTest(f"C10_T07_TREE={given} holds no test04 dry run")
        else:
            import test_t04_dryrun as t4d
            self.t4 = t4d.Tree.get()
            if self.t4.t04_missing:
                raise unittest.SkipTest("tools/t04_*.py missing")
            self.S = self.t4.S
        S = self.S
        self.tests = os.path.join(S, "data", "test")
        self.D2 = os.path.join(S, "results", "test02")
        self.O2 = os.path.join(S, "out", "rank")
        self.C = os.path.join(self.D2, "checks")
        self.R4 = os.path.join(S, "results", "test04")
        self.R7 = os.path.join(S, "results", "test07")
        self.drills = os.path.join(S, "t07_drills")
        shutil.rmtree(self.R7, ignore_errors=True)
        shutil.rmtree(self.drills, ignore_errors=True)
        os.makedirs(self.drills)
        self.snap0 = self.others()
        t0 = time.time()
        run_tool(["tools/t07_fake_tests.py", "from-val", "--root", S, "--results-dir", self.R7])
        self.apply_code, self.apply_out = run_tool(["tools/t07_rank.py", "apply", "--dry-run"] + self.dirs())
        self.status_code, self.status_out = run_tool(["tools/t07_rank.py", "status", "--dry-run"] + self.dirs())
        self.seconds = round(time.time() - t0, 1)
        self.snap1 = self.others()

    def others(self):
        return snapshot(self.D2, self.O2, self.tests, self.R4)

    def dirs(self, R7=None, R4=None, C=None, T=None):
        return ["--results-dir", R7 or self.R7, "--t04-results-dir", R4 or self.R4, "--t02-results-dir", self.D2,
                "--t02-out-dir", self.O2, "--tests-dir", T or self.tests, "--checks-dir", C or self.C]

    def copy_r7(self, name):
        """A copy of the fake HS sets only (no apply outputs) for one drill."""
        d = os.path.join(self.drills, name, "test07")
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)
        os.makedirs(os.path.join(d, "hs"))
        for f in os.listdir(os.path.join(self.R7, "hs")):
            if f.endswith("_hs.npz") or (f.startswith("hs_") and f.endswith("_report.json")):
                shutil.copyfile(os.path.join(self.R7, "hs", f), os.path.join(d, "hs", f))
        return d

    def copy_dir(self, src, name):
        d = os.path.join(self.drills, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(src, d)
        return d


def tearDownModule():
    t = Tree07.built
    if t is not None and t.t4 is not None:
        t.t4.finish()
        import test_t04_dryrun as t4d
        t4d.Tree.built = None
    elif t is not None and not os.environ.get("C10_KEEP"):
        shutil.rmtree(t.drills, ignore_errors=True)
    Tree07.built = None


SKIP = unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (R20, 04 and 07 dry runs, about 8-10 min) or 2")


# ================================================================ spec 9: 07 against prereg07
@SKIP
class T07DryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree07.get()
        cls.rank = os.path.join(cls.t.R7, "rank")
        cls.rep = read_json(os.path.join(cls.rank, "t07_apply_report.json"))

    def test_apply_exit_0(self):
        self.assertEqual(self.t.apply_code, 0, self.t.apply_out[-2000:])
        self.assertEqual((self.rep["mode"], self.rep["exit_code"], self.rep["usable"]), ("dry-run", 0, True))
        self.assertIn("apply: exit 0", self.t.status_out)

    def test_test02_test04_untouched(self):
        self.assertEqual(self.t.snap1, self.t.snap0)

    def test_local_seam_byte_identical_to_prereg07(self):
        seam = os.path.join(self.rank, "seam_val")
        for n in ("rank_val_probs.csv", "pred_B.csv", "pred_A.csv"):
            self.assertEqual(lf_lines(os.path.join(seam, n)), lf_lines(os.path.join(P07, n)), n)
        sl = read_json(os.path.join(seam, "seam_local.json"))
        self.assertEqual((sl["verdict"], sl["required_byte_identical"], sl["max_abs_diff"]), ("passed", True, 0.0))
        self.assertTrue(self.rep["hs_direction"]["seam_val_record_only"]["equal_to_prereg07_apply"])

    def test_test_rows_equal_prereg07_validation_rows(self):
        for name, ref_name in (("rank_test_probs.csv", "rank_val_probs.csv"), ("submission_rank.csv", "pred_B.csv")):
            ref, got = lf_lines(os.path.join(P07, ref_name)), lf_lines(os.path.join(self.rank, name))
            self.assertEqual(got[0], ref[0])
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
            self.assertGreater(self.rep["move_cross_check"][s]["compared"], 4700)
            self.assertEqual(self.rep["hs_failures"][s]["failed_or_no_move"], 0)
            self.assertTrue(self.rep["hs_direction"][s]["passed"])
        self.assertTrue(self.rep["q02_vs_04"]["equal"])
        self.assertTrue(all(self.rep["q02_vs_04"]["port_c5_equal"].values()))
        self.assertTrue(self.rep["gpu_seam"]["bitwise_equal_sums"])

    def test_golden(self):
        g = read_json(GOLDEN)["files_sha256"]
        for part in ("r20", "t04", "t07"):
            for rel, want in g[part].items():
                with self.subTest(rel):
                    self.assertEqual(sha(os.path.join(self.t.S, *rel.split("/"))), want)


# ================================================================ spec 9: negative drills
@SKIP
class T07Drills(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree07.get()

    def apply(self, R7, expect, R4=None, C=None, T=None, extra=()):
        code, out = run_tool(["tools/t07_rank.py", "apply", "--dry-run"] + self.t.dirs(R7, R4, C, T) + list(extra),
                             expect=expect)
        return out

    def verdict(self, R7):
        p = os.path.join(R7, "rank", "attempt_verdict.json")
        return read_json(p) if os.path.exists(p) else read_json(os.path.join(R7, "not_used.json"))

    def test_gpu_seam_per_move(self):
        r7 = self.t.copy_r7("seam_move")
        edit_npz(os.path.join(r7, "hs", "seam_hs.npz"), lambda a: a["pos_logp"].__setitem__((10, 2),
                                                                                          a["pos_logp"][10, 2] + 0.051))
        refresh_report(r7, "seam")
        self.apply(r7, 4)
        self.assertEqual(self.verdict(r7)["stage"], "gpu_seam")

    def test_gpu_seam_profile_mean(self):
        r7 = self.t.copy_r7("seam_mean")

        def f(a):
            a["sums"][0, 1] += 0.0021 * a["counts"][0]
        edit_npz(os.path.join(r7, "hs", "seam_hs.npz"), f)
        refresh_report(r7, "seam")
        self.apply(r7, 4)
        self.assertEqual(self.verdict(r7)["reason"], "gpu_seam")

    def test_hs_file_differs_from_report(self):
        r7 = self.t.copy_r7("hs_changed")
        with open(os.path.join(r7, "hs", "public_hs.npz"), "ab") as f:
            f.write(b"\0")
        self.assertIn("differs from its report", self.apply(r7, 2))

    def test_environment_differs(self):
        r7 = self.t.copy_r7("env")
        p = os.path.join(r7, "hs", "hs_private_report.json")
        rep = read_json(p)
        rep["environment"]["cudnn"] = 92000
        write_json(p, rep)
        self.apply(r7, 4)
        self.assertEqual(self.verdict(r7)["reason"], "hs_environment")

    def test_hs_before_r20_upload(self):
        r7 = self.t.copy_r7("before_r20")
        self.assertIn("before R20's first upload", self.apply(r7, 2, extra=["--r20-uploaded-at",
                                                                            "2099-01-01T00:00:00+08:00"]))

    def test_q02_one_byte_off_04(self):
        r7 = self.t.copy_r7("q02")
        r4 = self.t.copy_dir(self.t.R4, "r4_q02")
        p = os.path.join(r4, "rank", "rank_test_probs.csv")
        with open(p, "rb") as f:
            b = bytearray(f.read())
        i = b.index(b"0.", 200)
        b[i + 5] = ord("1") if b[i + 5] != ord("1") else ord("2")
        with open(p, "wb") as f:
            f.write(bytes(b))
        self.apply(r7, 4, R4=r4)
        self.assertEqual(self.verdict(r7)["stage"], "q02_vs_04")

    def test_t04_inputs_changed(self):
        r7 = self.t.copy_r7("t04_inputs")
        r4 = self.t.copy_dir(self.t.R4, "r4_inputs")
        p = os.path.join(r4, "sw", "seam_check.json")
        rep = read_json(p)
        rep["note"] = "changed"
        write_json(p, rep)
        self.assertIn("test04's bound inputs changed", self.apply(r7, 2, R4=r4))

    def test_move_cross_check(self):
        r7 = self.t.copy_r7("cross")
        edit_npz(os.path.join(r7, "hs", "public_hs.npz"), lambda a: a["n_moves"].__setitem__(5, a["n_moves"][5] + 1))
        refresh_report(r7, "public")
        self.apply(r7, 4)
        self.assertEqual(self.verdict(r7)["stage"], "move_cross_check")
        out = self.apply(r7, 2, extra=["--rerun", "--reason", "drill"])     # only after a PREREG_07 revision
        self.assertIn("--bug-revision", out)

    def test_failures_23_and_24_of_4787(self):
        for n, code in ((23, 0), (24, 3)):
            with self.subTest(n):
                r7 = self.t.copy_r7(f"fail_{n}")

                def f(a, n=n):
                    a["failed"][:n] = True
                edit_npz(os.path.join(r7, "hs", "public_hs.npz"), f)
                refresh_report(r7, "public")
                self.apply(r7, code)
                if code == 3:
                    v = self.verdict(r7)
                    self.assertEqual((v["stage"], v["exit_code"]), ("hs_failures", 3))
                    self.assertFalse(os.path.exists(os.path.join(r7, "rank", "submission_rank.csv")))
                else:
                    rep = read_json(os.path.join(r7, "rank", "t07_apply_report.json"))
                    self.assertEqual(rep["hs_failures"]["public"]["failed_or_no_move"], 23)

    def test_direction_check(self):
        r7 = self.t.copy_r7("direction")

        def f(a):
            import numpy as np
            rng = np.random.default_rng(0)
            a["sums"] = a["sums"][:, ::-1].copy()       # reverse the profile order: the curve points the other way
            a["sums"] = a["sums"][rng.permutation(len(a["sums"]))]
            a["counts"] = a["counts"]
        edit_npz(os.path.join(r7, "hs", "private_hs.npz"), f)
        refresh_report(r7, "private")
        self.apply(r7, 3)
        v = self.verdict(r7)
        self.assertEqual(v["stage"], "hs_direction")

    def test_04_not_used_and_not_done(self):
        r7 = self.t.copy_r7("t04_not_used")
        r4 = self.t.copy_dir(self.t.R4, "r4_not_used")
        p = os.path.join(r4, "rank", "t04_apply_report.json")
        rep = read_json(p)
        rep.update(usable=False, exit_code=3, unusable_reasons=["drill"])
        write_json(p, rep)
        self.apply(r7, 3, R4=r4)
        self.assertEqual(self.verdict(r7)["reason"], "t04_not_used")
        r7 = self.t.copy_r7("t04_not_done")
        os.remove(p)
        self.assertIn("test04 apply has not finished", self.apply(r7, 2, R4=r4))

    def test_k6_07(self):
        r7 = self.t.copy_r7("k6")
        self.apply(r7, 3, extra=["--now", "2026-11-20T00:00:00+08:00"])
        self.assertEqual(self.verdict(r7)["reason"], "k6_07")
        out = self.apply(self.t.R7, 2, extra=["--now", "2026-11-20T00:00:00+08:00", "--rerun", "--reason", "drill"])
        self.assertIn("stays valid", out)                   # a success before K6-07 stays; nothing is rerun

    def test_fail_opening(self):
        r7 = self.t.copy_r7("fail_opening")
        c = self.t.copy_dir(self.t.C, "checks_fo")
        p = os.path.join(c, "rank", "t1.json")
        t1 = read_json(p)
        t1["verdict"] = "FAIL_OPENING"
        write_json(p, t1)
        self.apply(r7, 3, C=c)
        self.assertTrue(os.path.exists(os.path.join(r7, "not_used.json")))

    def test_outputs_present(self):
        self.assertIn("never overwritten", self.apply(self.t.R7, 2))
        self.assertIn("registered decision", self.apply(self.t.R7, 2, extra=["--rerun", "--reason", "drill"]))

    def test_raw_tests_dir(self):
        r7 = self.t.copy_r7("raw")
        self.assertIn("never data/test/raw", self.apply(r7, 2, T=os.path.join(self.t.tests, "raw")))


@SKIP
class T07HsDrills(unittest.TestCase):
    """hs with the fake scorer (CPU) on a few games: the refusals before and around the GPU."""

    @classmethod
    def setUpClass(cls):
        cls.t = Tree07.get()

    def hs(self, R7, s, expect, T=None, extra=()):
        return run_tool(["tools/t07_rank.py", "hs", "--set", s, "--dry-run", "--fake-scorer", "--workers", "2",
                         "--max-games", "6"] + self.t.dirs(R7, T=T) + list(extra), expect=expect)[1]

    def test_stale_chunk_refused(self):
        r7 = os.path.join(self.t.drills, "hs_stale", "test07")
        shutil.rmtree(os.path.dirname(r7), ignore_errors=True)
        self.hs(r7, "seam", 0)
        os.remove(os.path.join(r7, "hs", "seam_hs.npz"))
        os.remove(os.path.join(r7, "hs", "hs_seam_report.json"))
        out = self.hs(r7, "seam", 2, extra=["--r20-uploaded-at", "2099-01-01T00:00:00+08:00"])
        self.assertIn("before R20's first upload", out)
        self.assertTrue(os.path.exists(os.path.join(r7, "hs_data", "seam", "chunk_00000.npz")), "never deleted")
        out = self.hs(r7, "seam", 0)                        # resuming in a valid order reuses the chunk
        self.assertIn('"todo": 0', out)

    def test_keys_differ_from_test02(self):
        r7 = os.path.join(self.t.drills, "hs_keys", "test07")
        shutil.rmtree(os.path.dirname(r7), ignore_errors=True)
        self.hs(r7, "seam", 0)
        t = self.t.copy_dir(self.t.tests, "tests_keys")
        p = os.path.join(t, RANK_FILES["public"])
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        h = rows[0]
        c1 = h.index("color_1")
        rows[1][c1] = "W" if rows[1][c1] == "B" else "B"
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\r\n").writerows(rows)
        self.assertIn("differ", self.hs(r7, "public", 2, T=t))

    def test_public_before_seam_refused(self):
        r7 = os.path.join(self.t.drills, "hs_order", "test07")
        shutil.rmtree(os.path.dirname(r7), ignore_errors=True)
        self.assertIn("run hs --set seam first", self.hs(r7, "public", 2))

    def test_real_mode_drill_flags_refused(self):
        code, out = run_tool(["tools/t07_rank.py", "hs", "--set", "seam", "--max-games", "5"], expect=2)
        self.assertIn("drills only", out)


if __name__ == "__main__":
    unittest.main()
