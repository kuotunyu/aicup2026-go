"""End-to-end dry run of the PREREG_11 Colab GPU fallback on the from-val fake tree. Every tool is called through its
command line, as on test day.

    C10_SLOW=1 or 2   the R20 / 04 tree of tests/test_t04_dryrun.py (about 8-10 min), or C10_G11_TREE=<S> to reuse a
        tree whose S/results/test04/rank/t04_apply_report.json exists (S/results/test07|08|09 and S/g11 are rebuilt).
        Then:
        07: g11 kit (tree only) -> g11 input --mode dry-run --stage hs (t07_rank prereqs + set_slots) -> g11
            fake-from-val --stage hs --relabel-gpu "<another GPU>" (the 4090 validation chunks of the same games, written
            by the Colab core's chunk writer, recorded under another GPU name) -> g11 import seam / public / private ->
            the unchanged t07_rank.py apply --dry-run.
        08: the same with --stage p1 -> the unchanged t08_rank.py apply --dry-run; 09: t09_rank.py apply --dry-run.
        The 29 outputs equal results/test09/design/t09_fake_golden.json (its t07 / t08 / t09 parts) byte for byte, the
        apply reports record the relabelled GPU, test02 / test04 are untouched.
        FakeScorer: the Colab core and t07_rank.py hs / t08_rank.py p1 (--fake-scorer --max-games 6, seam) write the
        same chunk field for field (except created_utc, seconds, gpu_seconds, tool_sha256); the core's fake chunks are
        imported (--max-games 6) into a separate test07 copy as a drill record.
        Negative drills on copies: outputs present, public before seam, a chunk changed after the Colab run summary,
        another environment than the seam's (4), a second Colab attempt, another input than the recorded one.
The fake tree holds validation games: only the seams and apply run on it. Nothing here opens a label file or runs
tools/score.py; nothing is written inside the repository.
Run: C10_SLOW=1 python -m pytest tests/test_g11_dryrun.py   (C10_KEEP=1 keeps the tree)
"""

import builtins
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import unittest

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)

SLOW = os.environ.get("C10_SLOW", "")
GOLDEN09 = os.path.join(REPO, "results", "test09", "design", "t09_fake_golden.json")
RELABEL = "NVIDIA RTX PRO 6000 Blackwell Server Edition (drill label)"
G11 = "tools/g11_gpu_fallback.py"


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
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(k, None)
    r = subprocess.run([sys.executable] + list(args), cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    out = r.stdout + r.stderr
    if expect is not None and r.returncode != expect:
        raise AssertionError(f"{' '.join(str(x) for x in args[:4])} exited {r.returncode} (expected {expect}):\n"
                             f"{out[-4000:]}")
    return r.returncode, out


class TreeG11:
    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            cls.built = cls()
        return cls.built

    def __init__(self):
        given = os.environ.get("C10_G11_TREE")
        self.t4 = None
        if given:
            self.S = os.path.abspath(given)
            if not os.path.exists(os.path.join(self.S, "results", "test04", "rank", "t04_apply_report.json")):
                raise unittest.SkipTest(f"C10_G11_TREE={given} holds no test04 dry run")
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
        self.R8 = os.path.join(S, "results", "test08")
        self.R9 = os.path.join(S, "results", "test09")
        self.G = os.path.join(S, "g11")
        for d in (self.R7, self.R8, self.R9, self.G):
            shutil.rmtree(d, ignore_errors=True)
        G = self.G
        self.rec, self.kit, self.inp = (os.path.join(G, x) for x in ("rec", "kit", "in"))
        self.tree = os.path.join(self.kit, "tree")
        self.snap0 = self.others()
        t0 = time.time()
        run_tool([G11, "kit", "--no-zip", "--out-dir", self.kit, "--record-dir", self.rec])
        # ---- 07
        run_tool([G11, "input", "--mode", "dry-run", "--stage", "hs"] + self.dirs7() + ["--out-dir", self.inp,
                                                                                        "--record-dir", self.rec])
        self.out_hs = os.path.join(G, "out_hs")
        run_tool([G11, "fake-from-val", "--stage", "hs", "--input", os.path.join(self.inp, "hs_dryrun"), "--out",
                  self.out_hs, "--kit", self.tree, "--relabel-gpu", RELABEL])
        self.imports = {}
        for s in ("seam", "public", "private"):
            self.imports[("hs", s)] = run_tool([G11, "import", "--stage", "hs", "--set", s, "--mode", "dry-run",
                                                "--from", self.out_hs, "--record-dir", self.rec] + self.dirs7())
        self.apply7 = run_tool(["tools/t07_rank.py", "apply", "--dry-run"] + self.dirs7(), expect=None)
        # ---- 08
        run_tool([G11, "input", "--mode", "dry-run", "--stage", "p1"] + self.dirs8() + ["--out-dir", self.inp,
                                                                                        "--record-dir", self.rec])
        self.out_p1 = os.path.join(G, "out_p1")
        run_tool([G11, "fake-from-val", "--stage", "p1", "--input", os.path.join(self.inp, "p1_dryrun"), "--out",
                  self.out_p1, "--kit", self.tree, "--relabel-gpu", RELABEL])
        for s in ("seam", "public", "private"):
            self.imports[("p1", s)] = run_tool([G11, "import", "--stage", "p1", "--set", s, "--mode", "dry-run",
                                                "--from", self.out_p1, "--record-dir", self.rec] + self.dirs8())
        self.apply8 = run_tool(["tools/t08_rank.py", "apply", "--dry-run"] + self.dirs8(), expect=None)
        self.apply9 = run_tool(["tools/t09_rank.py", "apply", "--dry-run"] + self.dirs9(), expect=None)
        self.seconds = round(time.time() - t0, 1)
        self.snap1 = self.others()

    def others(self):
        return snapshot(self.D2, self.O2, self.tests, self.R4)

    def trees(self, R4=True):
        return ["--t04-results-dir", self.R4, "--t02-results-dir", self.D2, "--t02-out-dir", self.O2,
                "--tests-dir", self.tests, "--checks-dir", self.C]

    def dirs7(self, R7=None):
        return ["--results-dir", R7 or self.R7] + self.trees()

    def dirs8(self, R8=None, R7=None):
        return ["--results-dir", R8 or self.R8, "--t07-results-dir", R7 or self.R7] + self.trees()

    def dirs9(self):
        return ["--results-dir", self.R9, "--t08-results-dir", self.R8, "--t07-results-dir", self.R7] + self.trees()

    def drill(self, name):
        d = os.path.join(self.G, "drills", name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        return d

    def rec_copy(self, d, with_imports=False):
        """A record folder with the kit record and the input records (and, optionally, the import records)."""
        r = os.path.join(d, "rec")
        shutil.copytree(self.rec, r, ignore=None if with_imports else shutil.ignore_patterns("import_*.json",
                                                                                              "attempt_*.json"))
        return r

    def r7_copy(self, d, sets=("seam",)):
        """test07 with only the HS outputs of the given sets."""
        r7 = os.path.join(d, "test07", "hs")
        os.makedirs(r7)
        for s in sets:
            for f in (f"{s}_hs.npz", f"hs_{s}_report.json"):
                shutil.copyfile(os.path.join(self.R7, "hs", f), os.path.join(r7, f))
        return os.path.dirname(r7)


def tearDownModule():
    t = TreeG11.built
    if t is not None and t.t4 is not None:
        t.t4.finish()
        import test_t04_dryrun as t4d
        t4d.Tree.built = None
    elif t is not None and not os.environ.get("C10_KEEP"):
        shutil.rmtree(t.G, ignore_errors=True)
    TreeG11.built = None


SKIP = unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (R20, 04, then the fallback with 07 / 08 / 09; about "
                                               "12-15 min) or 2")


@SKIP
class G11DryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = TreeG11.get()

    def test_imports_and_applies_exit_0(self):
        for k, (code, out) in self.t.imports.items():
            self.assertEqual(code, 0, f"{k}: {out[-1500:]}")
        for name in ("apply7", "apply8", "apply9"):
            code, out = getattr(self.t, name)
            self.assertEqual(code, 0, f"{name}: {out[-3000:]}")

    def test_outputs_equal_the_golden(self):
        g = read_json(GOLDEN09)["files_sha256"]
        bad = []
        for part in ("t07", "t08", "t09"):
            for rel, want in g[part].items():
                p = os.path.join(self.t.S, *rel.split("/"))
                if not os.path.exists(p) or sha(p) != want:
                    bad.append(rel)
        self.assertEqual(bad, [], "outputs differ from results/test09/design/t09_fake_golden.json")
        self.assertEqual(sum(len(g[p]) for p in ("t07", "t08", "t09")), 29)

    def test_apply_accepts_another_gpu(self):
        r7 = read_json(os.path.join(self.t.R7, "rank", "t07_apply_report.json"))
        self.assertEqual(r7["hs_environment"]["gpu"], RELABEL)
        self.assertEqual((r7["mode"], r7["exit_code"], r7["usable"]), ("dry-run", 0, True))
        r8 = read_json(os.path.join(self.t.R8, "rank", "t08_apply_report.json"))
        self.assertEqual(r8["p1_environment"]["gpu"], RELABEL)
        r9 = read_json(os.path.join(self.t.R9, "rank", "t09_apply_report.json"))
        self.assertEqual((r9["exit_code"], r9["usable"]), (0, True))

    def test_reports_carry_the_fallback_record(self):
        for st, R, name in (("hs", self.t.R7, "hs"), ("p1", self.t.R8, "p1")):
            for s in ("seam", "public", "private"):
                rep = read_json(os.path.join(R, name, f"{name}_{s}_report.json"))
                self.assertTrue(rep["prerequisites"]["gpu_env"]["colab_fallback"])
                self.assertEqual(rep["environment"]["gpu"], RELABEL)
                self.assertFalse(rep["fake_scorer"] or rep["fake_from_val"] or rep["max_games_drill"])
                imp = read_json(os.path.join(self.t.rec, f"import_{st}_{s}.json"))
                self.assertEqual(imp["outputs"][f"{s}_{name}.npz"], sha(os.path.join(R, name, f"{s}_{name}.npz")))

    def test_test02_test04_untouched(self):
        self.assertEqual(self.t.snap1, self.t.snap0)


@SKIP
class G11FakeScorer(unittest.TestCase):
    """The Colab core and the test-day tools write the same chunk for the same games (FakeScorer, CPU)."""

    @classmethod
    def setUpClass(cls):
        t = cls.t = TreeG11.get()
        d = cls.d = t.drill("fake")
        cls.fk7, cls.fk8, cls.core = (os.path.join(d, x) for x in ("fk7", "fk8", "core"))
        run_tool(["tools/t07_rank.py", "hs", "--set", "seam", "--dry-run", "--fake-scorer", "--max-games", "6",
                  "--workers", "2", "--results-dir", cls.fk7] + t.trees())
        run_tool(["tools/t08_rank.py", "p1", "--set", "seam", "--dry-run", "--fake-scorer", "--max-games", "6",
                  "--workers", "2", "--results-dir", cls.fk8, "--t07-results-dir", t.R7] + t.trees())
        for st in ("hs", "p1"):
            run_tool([os.path.join(t.tree, "colab_gpu", "gpu_fallback_core.py"), "run", "--kit", t.tree, "--input",
                      os.path.join(t.inp, f"{st}_dryrun"), "--out", os.path.join(cls.core, st), "--stage", st,
                      "--set", "seam", "--max-games", "6", "--fake-scorer", "--tools", os.path.join(REPO, "tools"),
                      "--workers", "2"])

    def compare(self, a, b, differ):
        with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
            self.assertEqual(list(za.files), list(zb.files))
            diff = sorted(k for k in za.files if not np.array_equal(za[k], zb[k]) or za[k].dtype != zb[k].dtype)
        self.assertEqual(diff, sorted(differ))

    def test_hs_chunk_equals_t07(self):
        self.compare(os.path.join(self.fk7, "hs_data", "seam", "chunk_00000.npz"),
                     os.path.join(self.core, "hs", "hs", "seam", "chunk_00000.npz"),
                     {"created_utc", "seconds", "tool_sha256"})

    def test_p1_chunk_equals_t08(self):
        self.compare(os.path.join(self.fk8, "p1_data", "seam", "chunk_00000.npz"),
                     os.path.join(self.core, "p1", "p1", "seam", "chunk_00000.npz"),
                     {"created_utc", "seconds", "gpu_seconds", "tool_sha256"})

    def test_fake_chunks_import_as_a_drill(self):
        t = self.t
        d = t.drill("fake_import")
        rec = t.rec_copy(d)
        r7 = os.path.join(d, "test07")
        code, out = run_tool([G11, "import", "--stage", "hs", "--set", "seam", "--mode", "dry-run", "--from",
                              os.path.join(self.core, "hs"), "--max-games", "6", "--record-dir", rec] + t.dirs7(r7))
        rep = read_json(os.path.join(r7, "hs", "hs_seam_report.json"))
        self.assertTrue(rep["fake_scorer"])
        self.assertTrue(rep["max_games_drill"])
        self.assertEqual(rep["games"], 6)


@SKIP
class G11Negative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = TreeG11.get()

    def imp(self, st, s, src, rec, dirs, expect):
        return run_tool([G11, "import", "--stage", st, "--set", s, "--mode", "dry-run", "--from", src,
                         "--record-dir", rec] + dirs, expect=expect)[1]

    def test_outputs_present(self):
        out = self.imp("hs", "seam", self.t.out_hs, self.t.rec, self.t.dirs7(), 2)
        self.assertIn("never overwritten", out)

    def test_public_before_seam(self):
        d = self.t.drill("order")
        out = self.imp("hs", "public", self.t.out_hs, self.t.rec_copy(d), self.t.dirs7(os.path.join(d, "test07")), 2)
        self.assertIn("seam first", out)

    def test_chunk_changed_after_the_run(self):
        d = self.t.drill("changed")
        src = os.path.join(d, "out")
        shutil.copytree(self.t.out_hs, src)
        with open(os.path.join(src, "hs", "seam", "chunk_00001.npz"), "ab") as f:
            f.write(b"\0")
        out = self.imp("hs", "seam", src, self.t.rec_copy(d), self.t.dirs7(os.path.join(d, "test07")), 2)
        self.assertIn("changed since the Colab run summary", out)

    def test_other_environment_than_the_seam(self):
        d = self.t.drill("env")
        src = os.path.join(d, "out")
        run_tool([G11, "fake-from-val", "--stage", "hs", "--input", os.path.join(self.t.inp, "hs_dryrun"), "--out",
                  src, "--kit", self.t.tree, "--relabel-gpu", "NVIDIA L4 (drill label)"])
        r7 = self.t.r7_copy(d, ("seam",))
        rec = self.t.rec_copy(d)
        out = self.imp("hs", "public", src, rec, self.t.dirs7(r7), 4)
        self.assertIn("one environment per stage", out)
        att = read_json(os.path.join(rec, "attempt_hs.json"))              # review minor d: written although refused
        self.assertEqual(att["run_id"], read_json(os.path.join(src, "hs", "ENV.json"))["run_id"])

    def test_second_attempt_refused(self):
        d = self.t.drill("second")
        src = os.path.join(d, "out")
        run_tool([G11, "fake-from-val", "--stage", "hs", "--input", os.path.join(self.t.inp, "hs_dryrun"), "--out",
                  src, "--kit", self.t.tree, "--relabel-gpu", RELABEL])
        r7 = self.t.r7_copy(d, ("seam",))
        out = self.imp("hs", "public", src, self.t.rec_copy(d, with_imports=True), self.t.dirs7(r7), 2)
        self.assertIn("one Colab attempt per stage", out)

    def test_attempt_record_of_the_main_run(self):
        att = read_json(os.path.join(self.t.rec, "attempt_hs.json"))
        self.assertEqual(att["run_id"], read_json(os.path.join(self.t.out_hs, "hs", "ENV.json"))["run_id"])
        self.assertTrue(os.path.exists(os.path.join(self.t.rec, "attempt_p1.json")))

    def test_another_input_refused(self):
        d = self.t.drill("input")
        src = os.path.join(d, "out")
        shutil.copytree(self.t.out_hs, src)
        p = os.path.join(src, "INPUT_MANIFEST.json")
        m = read_json(p)
        m["created"] = "2026-11-05T00:00:00+00:00"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(m, f)
        out = self.imp("hs", "seam", src, self.t.rec_copy(d), self.t.dirs7(os.path.join(d, "test07")), 2)
        self.assertIn("not the recorded input", out)


if __name__ == "__main__":
    unittest.main()
