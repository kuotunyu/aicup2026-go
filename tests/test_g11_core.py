"""Tests for colab_gpu/gpu_fallback_core.py (PREREG_11 1): the chunk format equals tools/t07_rank.py hs and
tools/t08_rank.py p1 field for field, chunks written by the core pass the unchanged t07 / t08 chunk_check, the resume
rules, and a CPU smoke of the whole Colab program (kit tree, fit-player fixture games, fake scorers) in subprocesses.

The pytest process never imports hs_lib, u1_lib, pilot_*, p1_* or torch (the smoke runs them in subprocesses). The
validation chunks data/prereg07/val and data/prereg08/val are read for their field layout and values (4090 output,
no labels). Run: python -m pytest tests/test_g11_core.py
"""

import ast
import builtins
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "colab_gpu"))
import gpu_fallback_core as core  # noqa: E402
import g11_gpu_fallback as g11  # noqa: E402

t7, t8 = g11.t7, g11.t8
VAL7 = os.path.join(REPO, "data", "prereg07", "val", "chunk_00000.npz")
VAL8 = os.path.join(REPO, "data", "prereg08", "val", "chunk_00000.npz")
HAVE_VAL = os.path.exists(VAL7) and os.path.exists(VAL8)


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process"


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def manifest():
    return {"constants": g11.kit_constants()}


def savez_keywords(src_path, func):
    """The keyword names (in order, without **) of the np.savez / save_npz call inside func of a tool."""
    with open(src_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) in ("savez", "save_npz"):
            kws = [k.arg for k in n.keywords if k.arg]
            if "format" in kws:
                return kws
    raise AssertionError(f"no savez call in {func}")


def res_from(z, stage):
    keys = ("sums", "counts", "n_moves", "failed", "errors", "pos_game", "pos_turn", "pos_side", "pos_logp") \
        if stage == "hs" else ("means", "counts", "n_moves", "failed", "errors", "pos_game", "pos_side", "pos_turn",
                               "pos_logp", "pos_entropy")
    return {k: z[k] for k in keys}


VAL_ENV = {"python": "3.12.13", "torch": "2.11.0+cu128", "cudnn": 91900, "gpu": "NVIDIA GeForce RTX 4090",
           "driver": "616.92"}


def hs_chunk(z, M, mode="dry-run", env=VAL_ENV):
    return core.hs_fields(M=M, refs=np.asarray(z["ref"]), set_label="test07_seam", env=env,
                          ck={"bytes": int(z["ckpt_bytes"]), "sha256": str(z["ckpt_sha256"])},
                          keys_sha=str(z["keys_sha256"]), questions_sha=str(z["questions_sha256"]),
                          tool_sha="t", mode=mode, created_utc="2026-11-05T00:00:00+00:00", seconds=1.0,
                          res=res_from(z, "hs"))


def p1_chunk(z, M, mode="dry-run", env=VAL_ENV):
    return core.p1_fields(M=M, refs=np.asarray(z["ref"]), set_label="test08_seam", env=env,
                          ck={"bytes": int(z["ckpt_bytes"]), "sha256": str(z["ckpt_sha256"])},
                          torch_v=str(z["torch"]), settings=core.p1_settings(M, False), meta_sha=str(z["meta_sha256"]),
                          code_json=str(z["code_sha256"]), keys_sha=str(z["keys_sha256"]),
                          questions_sha=str(z["questions_sha256"]), tool_sha="t", mode=mode,
                          created_utc="2026-11-05T00:00:00+00:00", seconds=1.0, gpu_seconds=0.5, res=res_from(z, "p1"))


class SourceOrder(unittest.TestCase):
    """The core's field order is the np.savez keyword order of the test-day tools (read with ast, not imported)."""

    def test_hs_fields_follow_t07(self):
        want = savez_keywords(os.path.join(REPO, "tools", "t07_rank.py"), "hs_run")
        got = list(core.hs_fields(M=manifest(), refs=np.arange(2), set_label="x", env=VAL_ENV,
                                  ck={"bytes": 1, "sha256": "s"}, keys_sha="k", questions_sha="q", tool_sha="t",
                                  mode="real", created_utc="c", seconds=0.0, res={}))
        self.assertEqual(got, want)

    def test_p1_fields_follow_t08(self):
        want = savez_keywords(os.path.join(REPO, "tools", "t08_rank.py"), "p1_run")
        M = manifest()
        got = list(core.p1_fields(M=M, refs=np.arange(2), set_label="x", env=VAL_ENV, ck={"bytes": 1, "sha256": "s"},
                                  torch_v="t", settings=core.p1_settings(M, False), meta_sha="m", code_json="{}",
                                  keys_sha="k", questions_sha="q", tool_sha="t", mode="real", created_utc="c",
                                  seconds=0.0, gpu_seconds=0.0, res={}))
        self.assertEqual(got, want)

    def test_settings_equal_t08_run_settings(self):
        M = manifest()
        for fake in (False, True):
            self.assertEqual(core.p1_settings(M, fake), t8.run_settings(fake))

    def test_chunk_plan(self):
        for n in (0, 1, 199, 200, 201, 400, 4787, 5000):
            want = [(ci, b, min(n, b + 200)) for ci, b in enumerate(range(0, n, 200))]
            self.assertEqual(core.chunk_plan(n, 200), want)


@unittest.skipUnless(HAVE_VAL, "the 4090 validation chunks (data/prereg07/val, data/prereg08/val) are not here")
class AgainstValidationChunks(unittest.TestCase):
    """A chunk the core writes from a 4090 validation chunk's arrays has that chunk's fields, dtypes and values, and the
    unchanged t07 / t08 chunk_check accept it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="c10_g11core_")
        cls.M = manifest()
        cls.z7, cls.z8 = load(VAL7), load(VAL8)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def check_same(self, z, fields, differ):
        self.assertEqual(list(fields), list(z), "field order")
        for k in z:
            a, b = np.asarray(fields[k]), np.asarray(z[k])
            self.assertEqual(a.dtype.kind, b.dtype.kind, k)
            if k not in differ:
                self.assertTrue(np.array_equal(a, b), k)

    def test_hs_chunk_equals_the_validation_chunk(self):
        f = hs_chunk(self.z7, self.M)
        self.check_same(self.z7, f, {"set", "created_utc", "seconds", "tool_sha256", "mode"})

    def test_p1_chunk_equals_the_validation_chunk(self):
        f = p1_chunk(self.z8, self.M)
        self.check_same(self.z8, f, {"format", "set", "created_utc", "seconds", "gpu_seconds", "tool_sha256", "mode"})
        self.assertEqual(str(f["format"]), t8.FORMATS["chunk"])

    def test_t07_chunk_check_accepts_and_refuses(self):
        p = core.save_chunk(os.path.join(self.tmp, "hs", "chunk_00000.npz"), hs_chunk(self.z7, self.M, mode="real"))
        refs, ksha = np.asarray(self.z7["ref"]), str(self.z7["keys_sha256"])
        t7.chunk_check(p, refs, ksha, VAL_ENV, real=True)
        with self.assertRaises(t7.T07Error) as e:
            t7.chunk_check(p, refs, ksha, dict(VAL_ENV, gpu="NVIDIA A100-SXM4-40GB"), real=True)
        self.assertEqual(e.exception.code, 4)
        with self.assertRaises(t7.T07Error) as e:
            t7.chunk_check(p, refs + 1, ksha, VAL_ENV, real=True)
        self.assertEqual(e.exception.code, 2)

    def test_t08_chunk_check_accepts_and_refuses(self):
        p = core.save_chunk(os.path.join(self.tmp, "p1", "chunk_00000.npz"), p1_chunk(self.z8, self.M, mode="real"))
        refs, ksha = np.asarray(self.z8["ref"]), str(self.z8["keys_sha256"])
        kw = dict(questions_sha=str(self.z8["questions_sha256"]), code_sha=str(self.z8["code_sha256"]), fake=False)
        t8.chunk_check(p, refs, ksha, VAL_ENV, "real", True, **kw)
        with self.assertRaises(t8.T08Error) as e:
            t8.chunk_check(p, refs, ksha, dict(VAL_ENV, cudnn=91002), "real", True, **kw)
        self.assertEqual(e.exception.code, 4)
        with self.assertRaises(t8.T08Error) as e:
            t8.chunk_check(p, refs, ksha, VAL_ENV, "dry-run", False, **kw)
        self.assertEqual(e.exception.code, 2)

    def test_core_resume_check(self):
        p = core.save_chunk(os.path.join(self.tmp, "hs2", "chunk_00000.npz"), hs_chunk(self.z7, self.M, mode="real"))
        rec = {"set": "seam", "keys_sha256": str(self.z7["keys_sha256"]),
               "questions_sha256": str(self.z7["questions_sha256"])}
        refs = np.asarray(self.z7["ref"])
        ck = str(self.z7["ckpt_sha256"])
        core.check_existing(p, "hs", self.M, refs, rec, "real", VAL_ENV, False, ck)
        with self.assertRaises(core.Drift):
            core.check_existing(p, "hs", self.M, refs, rec, "real", dict(VAL_ENV, torch="2.10.0+cu126"), False, ck)
        for bad in ({"mode": "rehearsal"}, {"refs": refs + 1}, {"rec": dict(rec, keys_sha256="x")},
                    {"rec": dict(rec, set="public")}):
            with self.subTest(bad=list(bad)), self.assertRaises(core.Refused):
                core.check_existing(p, "hs", self.M, bad.get("refs", refs), bad.get("rec", rec),
                                    bad.get("mode", "real"), VAL_ENV, False, ck)


class StageEnvironment(unittest.TestCase):
    def test_resume_only_in_the_same_environment(self):
        d = tempfile.mkdtemp(prefix="c10_g11env_")
        try:
            r = core.stage_environment(d, "hs", VAL_ENV, "x", "real")
            self.assertEqual(core.stage_environment(d, "hs", dict(VAL_ENV, driver="999"), "x", "real")["run_id"],
                             r["run_id"])                       # the driver is recorded only
            with self.assertRaises(core.Drift):
                core.stage_environment(d, "hs", dict(VAL_ENV, gpu="NVIDIA L4"), "x", "real")
            with self.assertRaises(core.Refused):
                core.stage_environment(d, "hs", VAL_ENV, "another input", "real")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class AllowedEnvironment(unittest.TestCase):
    """Review major 1: a real input carries the allowed environments; the core refuses any other runtime before it
    writes anything (rehearsal / dry-run / smoke inputs are exempt); fake scorers never run on rehearsal inputs."""

    def test_env_key_fields(self):
        k = core.env_key(dict(VAL_ENV, numpy="2.5.3", driver="1"))
        self.assertEqual(sorted(k), sorted(("torch", "cudnn", "gpu", "python", "numpy")))
        self.assertNotIn("driver", k)

    def test_allowed_problem(self):
        env = dict(VAL_ENV, numpy="2.5.3")
        key = core.env_key(env)
        real = {"mode": "real", "allowed_environments": {"hs": [key]}}
        self.assertIsNone(core.allowed_problem(real, "hs", env))
        self.assertIn("not an environment", core.allowed_problem(real, "hs", dict(env, gpu="NVIDIA L4")))
        self.assertIn("not an environment", core.allowed_problem(real, "hs", dict(env, numpy="2.4.0")))
        self.assertIn("no allowed environment", core.allowed_problem(real, "p1", env))
        self.assertIn("no allowed environment", core.allowed_problem({"mode": "real"}, "hs", env))
        for mode in ("rehearsal", "dry-run", "smoke"):
            self.assertIsNone(core.allowed_problem({"mode": mode}, "hs", dict(env, gpu="x")))

    def test_refused_before_writing(self):
        d = tempfile.mkdtemp(prefix="c10_g11allow_")
        try:
            I = {"mode": "rehearsal", "sets": {"seam": {}}, "input_id": "x", "sets_data": {"seam": {"games": []}}}
            for mode in ("rehearsal", "real"):
                with self.assertRaises(core.Refused):
                    core.run_set(d, manifest(), dict(I, mode=mode), d, "hs", "seam", workers=1, fake=True)
            self.assertEqual(os.listdir(d), [], "nothing written")
            src = open(core.__file__, encoding="utf-8").read()          # the real-mode order (torch needs a GPU box)
            body = src[src.index("def run_set("):]
            self.assertLess(body.index("allowed_problem(I, stage, env)"), body.index("os.makedirs(chunk_dir"))
            self.assertLess(body.index("allowed_problem(I, stage, env)"), body.index("stage_environment("))
            run = src[src.index("def cmd_run("):]
            self.assertLess(run.index("allowed_problem(I, st, env0)"), run.index("os.makedirs(out"))
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ================================================================ the whole program on the CPU (subprocesses)
PY = sys.executable


def run(args, expect=0):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    env.pop("PYTHONUTF8", None)
    r = subprocess.run([PY] + args, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env)
    out = r.stdout + r.stderr
    if expect is not None and r.returncode != expect:
        raise AssertionError(f"{args[:3]} exited {r.returncode} (expected {expect}):\n{out[-3000:]}")
    return r.returncode, out


FIXTURE = os.path.join(REPO, "data", "prereg07", "fixtures", "fit_sample.json")
CKPT = os.path.join(REPO, "data", "katago", "b18c384nbt-humanv0.ckpt")


class Smoke:
    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            cls.built = cls()
        return cls.built

    def __init__(self):
        if not (os.path.exists(FIXTURE) and os.path.exists(CKPT)):
            raise unittest.SkipTest("fit_sample.json fixture or the checkpoint missing")
        self.d = tempfile.mkdtemp(prefix="c10_g11smoke_")
        d = self.d
        self.rec, self.kit, self.inp, self.out = (os.path.join(d, x) for x in ("rec", "kit", "in", "out"))
        run(["tools/g11_gpu_fallback.py", "kit", "--no-zip", "--out-dir", self.kit, "--record-dir", self.rec])
        run(["tools/g11_gpu_fallback.py", "input", "--mode", "smoke", "--out-dir", self.inp, "--record-dir", self.rec])
        self.tree = os.path.join(self.kit, "tree")
        self.input = os.path.join(self.inp, "smoke")
        self.code, self.log = run(self.cmd(self.out))

    def cmd(self, out, *extra):
        return [os.path.join(self.tree, "colab_gpu", "gpu_fallback_core.py"), "run", "--kit", self.tree, "--input",
                self.input, "--out", out, "--fake-scorer", "--tools", os.path.join(REPO, "tools"), "--workers", "2",
                *extra]


def tearDownModule():
    if Smoke.built is not None:
        shutil.rmtree(Smoke.built.d, ignore_errors=True)
        Smoke.built = None


class CpuSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Smoke.get()

    def test_run_complete(self):
        self.assertEqual(self.s.code, 0, self.s.log[-2000:])
        summ = json.load(open(os.path.join(self.s.out, "RUN_SUMMARY.json"), encoding="utf-8"))
        self.assertEqual([(r["stage"], r["set"], r["done"]) for r in summ["runs"]], [("hs", "smoke", True),
                                                                                    ("p1", "smoke", True)])
        for st in ("hs", "p1"):
            z = load(os.path.join(self.s.out, st, "smoke", "chunk_00000.npz"))
            self.assertEqual(str(z["mode"]), "smoke")
            self.assertEqual(str(z["ckpt_sha256"]), "fake")
            self.assertEqual(len(z["ref"]), 6)
            self.assertFalse(np.asarray(z["failed"]).any())
            self.assertGreater(len(z["pos_side"]), 0)
        z = load(os.path.join(self.s.out, "p1", "smoke", "chunk_00000.npz"))
        self.assertEqual(json.loads(str(z["code_sha256"])), g11.kit_constants()["p1_code_record"])
        self.assertEqual(str(z["meta_sha256"]), g11.r8.NEUTRAL_META_SHA)
        self.assertEqual(str(z["precision"]), "fake-fp32")

    def test_kit_tree_holds_no_bytecode(self):
        hits = [d for d, _ds, _fs in os.walk(self.s.tree) if os.path.basename(d) == "__pycache__"]
        self.assertEqual(hits, [])

    def test_rerun_resumes(self):
        out2 = os.path.join(self.s.d, "out_resume")
        shutil.copytree(self.s.out, out2)
        before = {f: core.sha_raw(os.path.join(out2, "hs", "smoke", f)) for f in ("chunk_00000.npz",)}
        code, log = run(self.s.cmd(out2))
        self.assertIn('"todo": 0', log)
        self.assertEqual(before["chunk_00000.npz"], core.sha_raw(os.path.join(out2, "hs", "smoke", "chunk_00000.npz")))

    def test_other_environment_stops(self):
        out3 = os.path.join(self.s.d, "out_env")
        shutil.copytree(self.s.out, out3)
        p = os.path.join(out3, "hs", "ENV.json")
        rec = json.load(open(p, encoding="utf-8"))
        rec["key"]["gpu"] = "NVIDIA A100-SXM4-40GB"
        json.dump(rec, open(p, "w", encoding="utf-8"))
        code, log = run(self.s.cmd(out3), expect=4)
        self.assertIn("ENVIRONMENT DIFFERS", log)

    def test_changed_chunk_refused(self):
        out4 = os.path.join(self.s.d, "out_bad")
        shutil.copytree(self.s.out, out4)
        p = os.path.join(out4, "p1", "smoke", "chunk_00000.npz")
        z = load(p)
        z["keys_sha256"] = np.array("0" * 64)
        np.savez(p + ".tmp.npz", **z)
        os.replace(p + ".tmp.npz", p)
        code, log = run(self.s.cmd(out4), expect=2)
        self.assertIn("differs from this run", log)

    def test_fake_refused_for_real_inputs_and_without_tools(self):
        code, log = run([os.path.join(self.s.tree, "colab_gpu", "gpu_fallback_core.py"), "run", "--kit", self.s.tree,
                         "--input", self.s.input, "--out", os.path.join(self.s.d, "out_notools"), "--fake-scorer"],
                        expect=2)
        self.assertIn("--tools", log)

    def test_kit_check_catches_a_changed_file(self):
        tree2 = os.path.join(self.s.d, "tree_bad")
        shutil.copytree(self.s.tree, tree2, ignore=shutil.ignore_patterns("*.ckpt"))
        os.makedirs(os.path.join(tree2, "data", "katago"), exist_ok=True)
        with open(os.path.join(tree2, "results", "prereg08", "p1", "p1_lib.py"), "a", encoding="utf-8") as f:
            f.write("\n# changed\n")
        code, log = run([os.path.join(tree2, "colab_gpu", "gpu_fallback_core.py"), "verify", "--kit", tree2,
                         "--no-ckpt-hash"], expect=2)
        self.assertIn("kit check failed", log)


if __name__ == "__main__":
    unittest.main()
