"""Tests for tools/g11_gpu_fallback.py (PREREG_11) that need no dry-run tree: the trigger and verdict rules, the kit
constants, the set files, the Colab results checks, the chunk fields the test-day chunk_check does not read, and the
command-line refusals of the real mode (no trigger, no record folder, outputs inside the repository).

Nothing here writes inside the repository (every refused command stops before writing). The pytest process never
imports hs_lib, u1_lib, pilot_*, p1_* or torch. Run: python -m pytest tests/test_g11_tool.py
"""

import builtins
import datetime as dt
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

TW = g11.t7.TW


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process"


def run(args, expect=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    env.pop("PYTHONUTF8", None)
    r = subprocess.run([sys.executable, "tools/g11_gpu_fallback.py"] + args, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    out = r.stdout + r.stderr
    if expect is not None:
        assert r.returncode == expect, f"{args[:2]} exited {r.returncode} (expected {expect}):\n{out[-2000:]}"
    return r.returncode, out


class Rules(unittest.TestCase):
    def test_trigger_deadline(self):
        before = dt.datetime(2026, 11, 14, 23, 59, 59, tzinfo=TW)
        after = dt.datetime(2026, 11, 15, 0, 0, 0, tzinfo=TW)
        self.assertIsNotNone(g11.trigger_problem("deadline_11_14", before))
        self.assertIsNone(g11.trigger_problem("deadline_11_14", after))
        for r in ("hardware_failure", "gpu_exit2", "venv_gpu_unrestorable"):
            self.assertIsNone(g11.trigger_problem(r, before))
        self.assertIsNotNone(g11.trigger_problem("slow_gpu", after))

    def test_stage_verdict(self):
        tol = {"lp": 0.02, "mean": 0.002}
        ok = {"seam_passed": True, "counts_equal": True}
        self.assertTrue(g11.stage_verdict(ok, {"lp": 0.02, "mean": 0.002}, tol)["allowed"])        # rev 2: x 1.0 = tol
        self.assertTrue(g11.stage_verdict(ok, {"lp": 0.0086, "mean": 0.0007}, tol)["allowed"])     # A100: 2.5x > tol
        self.assertFalse(g11.stage_verdict(ok, {"lp": 0.0201, "mean": 0.0}, tol)["allowed"])
        self.assertFalse(g11.stage_verdict(ok, {"lp": None, "mean": 0.0}, tol)["allowed"])
        self.assertFalse(g11.stage_verdict(dict(ok, seam_passed=False), {"lp": 0, "mean": 0}, tol)["allowed"])
        self.assertFalse(g11.stage_verdict(dict(ok, counts_equal=False), {"lp": 0, "mean": 0}, tol)["allowed"])
        v = g11.stage_verdict(ok, {"lp": 0.0086, "mean": 0.0}, tol)
        self.assertEqual(v["tolerances_now"], tol)                 # never widened
        self.assertEqual(v["rule_version"], 2)
        self.assertAlmostEqual(v["record_only_offgrid_times_2_5"]["lp"], 0.0215)       # recorded only
        self.assertFalse(v["record_only_times_2_5_within"]["lp"])

    def test_rehearsal_rows(self):
        self.assertEqual(g11.REHEARSAL_ROWS, {"seam": (0, 400), "offgrid": (4787, 5787)})
        self.assertEqual(g11.FACTOR, 2.5)


class Kit(unittest.TestCase):
    def test_constants_are_the_registered_ones(self):
        c = g11.kit_constants()
        pf7, r8, t7, t8 = g11.pf7, g11.r8, g11.t7, g11.t8
        self.assertEqual(c["profiles"], list(pf7.PROFILES))
        self.assertEqual(c["hs_run"], dict(pf7.RUN))
        self.assertEqual(c["p1_run"], dict(r8.RUN))
        self.assertEqual((c["hs_chunk_format"], c["p1_chunk_format"]), (t7.CHUNK_FORMAT, t8.FORMATS["chunk"]))
        self.assertEqual((c["ckpt_sha256"], c["ckpt_bytes"]), (pf7.CKPT_SHA, pf7.CKPT_BYTES))
        self.assertEqual(c["neutral_meta_sha256"], r8.NEUTRAL_META_SHA)
        self.assertEqual(set(c["p1_code_record"]), {
            "results/prereg08/p1/p1_lib.py", "results/prereg08/p1/p1_extract.py", "results/prereg08/p1/p1_fuse.py",
            "results/prereg07/explore/pilot_humansl/hs_lib.py", "results/prereg07/explore/pilot_humansl/pilot_fuse.py",
            "results/prereg05/explore/u1/u1_lib.py"})

    def test_kit_files(self):
        rels = [r for r, _ in g11.kit_files()]
        self.assertEqual(rels, sorted(rels))
        for need in ("colab_gpu/gpu_fallback_core.py", "data/katago/b18c384nbt-humanv0.ckpt",
                     "results/prereg07/explore/pilot_humansl/hs_lib.py", "results/prereg07/explore/pilot_humansl/pilot_run.py",
                     "results/prereg08/p1/p1_extract.py", "third_party/katago/SOURCE.txt",
                     "third_party/katago/python/katago/train/model_pytorch.py"):
            self.assertIn(need, rels)
        self.assertFalse([r for r in rels if r.startswith(("colab/", "data/local")) or "__pycache__" in r])

    def test_manifest_pins_and_id(self):
        m = g11.build_kit_manifest()
        self.assertTrue(m["kit_id"].startswith("g11-kit-v1-"))
        self.assertEqual(m["pins_lf"]["results/prereg07/explore/pilot_humansl/hs_lib.py"],
                         g11.pf7.PINNED["results/prereg07/explore/pilot_humansl/hs_lib.py"])
        self.assertEqual(m["files"]["data/katago/b18c384nbt-humanv0.ckpt"]["sha256"], g11.pf7.CKPT_SHA)
        self.assertEqual(g11.build_kit_manifest()["kit_id"], m["kit_id"])

    def test_zip_members_deterministic(self):
        d = tempfile.mkdtemp(prefix="c10_g11zip_")
        try:
            shas = []
            for i in range(2):
                p = os.path.join(d, f"z{i}.zip")
                import zipfile
                with zipfile.ZipFile(p, "w") as zf:
                    g11.zip_write(zf, "a/b.json", data=b"{}\n")
                    g11.zip_write(zf, "a/c.py", src=os.path.join(REPO, "colab_gpu", "gpu_fallback_core.py"))
                shas.append(core.sha_raw(p))
            self.assertEqual(shas[0], shas[1])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class SetFiles(unittest.TestCase):
    def test_deterministic_and_sensitive(self):
        games = [{"sgf_content": "(;SZ[19];B[pd])", "color": "B", "ref": 0}]
        keys = {"question_id": ["q1"], "slot": [1], "color": ["B"]}
        a = g11.set_file_bytes("seam", games, keys, "k", "q", {"x": 1})
        self.assertEqual(a, g11.set_file_bytes("seam", games, keys, "k", "q", {"x": 1}))
        self.assertNotEqual(a, g11.set_file_bytes("seam", [dict(games[0], color="W")], keys, "k", "q", {"x": 1}))
        self.assertNotEqual(a, g11.set_file_bytes("seam", games, keys, "k2", "q", {"x": 1}))


# ================================================================ Colab results checks on a synthetic results folder
def fake_results(root, n=3, stage="hs", s="seam", mode="dry-run"):
    """A tiny results folder in the Colab layout (core writer, fake chunk values)."""
    M = {"constants": g11.kit_constants(), "kit_id": "g11-kit-v1-test"}
    env = {"python": "3.12", "torch": "2.11.0+cu128", "cudnn": 91900, "gpu": "NVIDIA L4", "driver": "580"}
    st = core.stage_environment(root, stage, env, "x_dryrun", mode)
    res = {"sums": np.zeros((n, 2, 14)), "counts": np.ones((n, 2), np.int32), "n_moves": np.ones(n, np.int16),
           "failed": np.zeros(n, bool), "errors": "[]", "pos_game": np.arange(n, dtype=np.int32),
           "pos_turn": np.zeros(n, np.int16), "pos_side": np.zeros(n, np.int8),
           "pos_logp": np.zeros((n, 14), np.float32)}
    f = core.hs_fields(M=M, refs=np.arange(n), set_label=f"test07_{s}", env=env,
                       ck={"bytes": g11.pf7.CKPT_BYTES, "sha256": g11.pf7.CKPT_SHA}, keys_sha="k", questions_sha="q",
                       tool_sha="t", mode=mode, created_utc="2026-11-05T01:00:00+00:00", seconds=1.0, res=res)
    p = core.save_chunk(os.path.join(root, stage, s, "chunk_00000.npz"), f)
    summ = {"format": core.SUMMARY_FORMAT, "input_id": "x_dryrun", "mode": mode,
            "runs": [{"stage": stage, "set": s, "done": True, "run_id": st["run_id"], "environment": st["key"],
                      "chunks": 1, "chunk_sha256": {"chunk_00000.npz": core.sha_raw(p)}}]}
    for name, obj in (("RUN_SUMMARY.json", summ), ("INPUT_MANIFEST.json", {"input_id": "x_dryrun"}),
                      ("KIT_MANIFEST.json", M)):
        core.write_json_atomic(os.path.join(root, name), obj)
    return M, p


class ColabResults(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="c10_g11res_")
        self.M, self.chunk = fake_results(self.d)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_accepts_a_complete_set(self):
        col = g11.colab_results(self.d, "hs", "seam")
        self.assertEqual(col["files"], ["chunk_00000.npz"])
        self.assertEqual(g11.chunk_extra_problems("hs", self.chunk, "dry-run", "test07_seam", self.M)[0], [])

    def test_changed_chunk(self):
        with open(self.chunk, "ab") as f:
            f.write(b"x")
        with self.assertRaises(g11.G11Error) as e:
            g11.colab_results(self.d, "hs", "seam")
        self.assertEqual(e.exception.code, 2)

    def test_missing_summary_and_incomplete_run(self):
        p = os.path.join(self.d, "RUN_SUMMARY.json")
        summ = json.load(open(p, encoding="utf-8"))
        summ["runs"][0]["done"] = False
        core.write_json_atomic(p, summ)
        with self.assertRaises(g11.G11Error):
            g11.colab_results(self.d, "hs", "seam")
        os.remove(p)
        with self.assertRaises(g11.G11Error):
            g11.colab_results(self.d, "hs", "seam")

    def test_run_id_mismatch_is_drift(self):
        p = os.path.join(self.d, "hs", "ENV.json")
        rec = json.load(open(p, encoding="utf-8"))
        rec["run_id"] = "another"
        core.write_json_atomic(p, rec)
        with self.assertRaises(g11.G11Error) as e:
            g11.colab_results(self.d, "hs", "seam")
        self.assertEqual(e.exception.code, 4)

    def test_extra_fields(self):
        for args, bad in ((("hs", self.chunk, "real", "test07_seam", self.M), "mode"),
                          (("hs", self.chunk, "dry-run", "test07_public", self.M), "set")):
            self.assertIn(bad, g11.chunk_extra_problems(*args)[0])
        M2 = json.loads(json.dumps(self.M))
        M2["constants"]["hs_lib_sha256"] = "0" * 64
        self.assertIn("hs_lib_sha256", g11.chunk_extra_problems("hs", self.chunk, "dry-run", "test07_seam", M2)[0])


class CommandLine(unittest.TestCase):
    """Refusals that stop before anything is written (the real mode has no trigger on this branch)."""

    def test_import_real_needs_a_committed_trigger(self):
        if os.path.exists(os.path.join(REPO, "results", "test11", "trigger_hs.json")):
            self.skipTest("a trigger record exists")
        code, out = run(["import", "--stage", "hs", "--set", "seam", "--from", tempfile.gettempdir()], expect=2)
        self.assertIn("no trigger", out)

    def test_input_real_needs_a_committed_trigger(self):
        if not os.path.exists(os.path.join(REPO, "results", "test11", "kit_manifest.json")) or \
                os.path.exists(os.path.join(REPO, "results", "test11", "trigger_p1.json")):
            self.skipTest("no kit record yet, or a trigger record exists")
        code, out = run(["input", "--mode", "real", "--stage", "p1"], expect=2)
        self.assertIn("no trigger", out)

    def test_drills_need_folders_outside_the_repository(self):
        code, out = run(["import", "--stage", "hs", "--set", "seam", "--mode", "dry-run", "--from", "x"], expect=2)
        self.assertIn("--record-dir", out)
        code, out = run(["input", "--mode", "dry-run", "--stage", "hs", "--record-dir",
                         os.path.join(REPO, "results", "test11")], expect=2)
        self.assertIn("outside the repository", out)
        code, out = run(["rehearsal-compare", "--from", "x", "--work-dir", os.path.join(REPO, "data", "test11", "w")],
                        expect=2)
        self.assertIn("outside the repository", out)

    def test_trigger_refusals(self):
        code, out = run(["trigger", "--stage", "hs", "--reason", "slow_gpu", "--evidence", "x" * 30], expect=2)
        code, out = run(["trigger", "--stage", "hs", "--reason", "hardware_failure", "--evidence", "short"], expect=2)
        self.assertIn("--evidence", out)
        if dt.datetime.now(TW) <= g11.DEADLINE:
            code, out = run(["trigger", "--stage", "p1", "--reason", "deadline_11_14", "--evidence", "x" * 30],
                            expect=2)
            self.assertIn("holds only after", out)
        for st in ("hs", "p1"):
            self.assertFalse(os.path.exists(os.path.join(REPO, "results", "test11", f"trigger_{st}.json")))

    def test_status(self):
        code, out = run(["status"], expect=0)
        self.assertIn("trigger hs", out)


# ================================================================ review fixes (blocker, majors 1-3, minors c-f)
def resave(p, z):
    np.savez(p + ".x.npz", **z)
    os.replace(p + ".x.npz", p)


class ReviewRules(unittest.TestCase):
    def test_rehearsal_cutoff(self):                                       # major 2
        self.assertTrue(g11.rehearsal_in_time("2026-11-04T10:59:59+08:00"))
        self.assertTrue(g11.rehearsal_in_time("2026-11-04T02:59:59+00:00"))
        self.assertFalse(g11.rehearsal_in_time("2026-11-04T11:00:00+08:00"))
        self.assertFalse(g11.rehearsal_in_time("2026-11-05T00:00:00+08:00"))
        self.assertFalse(g11.rehearsal_in_time(None))

    def test_finished_on_4090_counts_committed_files_only(self):           # minor f
        root = os.path.join(g11.RESULTS, "test07", "hs")
        committed = {os.path.join(root, f) for f in ("seam_hs.npz", "hs_seam_report.json", "public_hs.npz")}
        self.assertEqual(g11.finished_on_4090("hs", lambda p: p in committed), ["seam"])
        self.assertEqual(g11.finished_on_4090("hs", lambda p: False), [])

    def test_chunk_origin(self):                                           # blocker / minor c
        d = tempfile.mkdtemp(prefix="c10_g11orig_")
        try:
            _M, p = fake_results(d)                                        # tool_sha 't', seconds 1.0, real ckpt
            core_sha = "c" * 64
            self.assertIn("tool_sha256 (not the kit's core)", g11.chunk_origin_problems(p, core_sha, True))
            self.assertEqual(g11.chunk_origin_problems(p, core_sha, False), ["tool_sha256"])
            z = dict(np.load(p))
            z.update(tool_sha256=np.array(core_sha), seconds=np.array(0.0))
            resave(p, z)
            self.assertEqual(g11.chunk_origin_problems(p, core_sha, True), ["seconds"])
            z.update(seconds=np.array(2.0), ckpt_sha256=np.array("fake"))
            resave(p, z)
            self.assertEqual(g11.chunk_origin_problems(p, core_sha, True), ["fake scorer"])
            z.update(ckpt_sha256=np.array(g11.pf7.CKPT_SHA))
            resave(p, z)
            self.assertEqual(g11.chunk_origin_problems(p, core_sha, True), [])
            z.update(tool_sha256=np.array(g11.FAKE_FROM_VAL_TOOL))
            resave(p, z)
            self.assertEqual(g11.chunk_origin_problems(p, core_sha, False), [])
            self.assertTrue(g11.chunk_origin_problems(p, core_sha, True))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_attempt_record(self):                                         # minor d
        d = tempfile.mkdtemp(prefix="c10_g11att_")
        try:
            col = {"env_rec": {"run_id": "r1", "key": {"gpu": "x"}}, "summary": {"input_id": "hs_test"}, "src": d}
            g11.attempt_record(d, "hs", col, "real")
            self.assertTrue(os.path.exists(os.path.join(d, "attempt_hs.json")))
            g11.attempt_record(d, "hs", col, "real")                       # the same run: fine
            col2 = {"env_rec": {"run_id": "r2", "key": {"gpu": "y"}}, "summary": {"input_id": "hs_test"}, "src": d}
            with self.assertRaises(g11.G11Error) as e:
                g11.attempt_record(d, "hs", col2, "real")
            self.assertIn("one Colab attempt per stage", str(e.exception))
            g11.attempt_record(d, "p1", col2, "real")                      # another stage: its own attempt
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_late_4090_chunks(self):                                       # minor e
        d = tempfile.mkdtemp(prefix="c10_g11late_")
        try:
            t = dt.datetime(2026, 11, 5, 12, 0, 0, tzinfo=TW)
            for name, c in (("seam_4090_a", "2026-11-05T03:00:00+00:00"), ("seam_4090_b", "2026-11-05T05:00:00+00:00")):
                os.makedirs(os.path.join(d, name))
                np.savez(os.path.join(d, name, "chunk_00000.npz"), created_utc=np.array(c))
            late = g11.late_4090_chunks(d, "seam", t)
            self.assertEqual(len(late), 1)
            self.assertIn("seam_4090_b", late[0])
            self.assertEqual(g11.late_4090_chunks(d, "public", t), [])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_notebook_checks_the_gpu_before_running(self):                 # majors 1 and 3
        with open(os.path.join(REPO, "colab_gpu", "build_gpu_fallback_nb.py"), encoding="utf-8") as f:
            src = f.read()
        install = src[src.index("CELL_INSTALL ="):src.index("CELL_UNPACK =")]
        self.assertIn("f'torch=={TORCH_WANT}'", install)
        self.assertIn("ENV['torch'] != TORCH_WANT", install)
        self.assertIn("allowed_environments", install)
        self.assertIn('TORCH_WANT = "2.11.0+cu128"', src)
        self.assertIn("'_{GPU_SLUG}' if INPUT['mode'] == 'rehearsal'", src)


class RoundTwo(unittest.TestCase):
    def rec(self, kit_id="k1", allowed=True):
        return {"format": g11.FORMATS["rehearsal"], "rule_version": 2, "kit_id": kit_id,
                "verdict": {"hs": {"allowed": allowed, "rule_version": 2}}, "environment": {"hs": {"gpu": "g"}}}

    def test_record_needs_the_current_kit(self):                            # item 1
        ok_time = "2026-10-20T10:00:00+08:00"
        self.assertTrue(g11.record_allows(self.rec(), "hs", "k1", True, ok_time))
        self.assertFalse(g11.record_allows(self.rec(kit_id="k0"), "hs", "k1", True, ok_time))
        self.assertFalse(g11.record_allows(self.rec(allowed=False), "hs", "k1", True, ok_time))
        self.assertFalse(g11.record_allows(self.rec(), "p1", "k1", True, ok_time))
        self.assertFalse(g11.record_allows(self.rec(), "hs", "k1", False, ok_time))
        self.assertFalse(g11.record_allows(self.rec(), "hs", "k1", True, "2026-11-04T11:00:00+08:00"))

    def test_rule_1_records_never_count(self):                              # PREREG_11 rev 2
        ok_time = "2026-10-20T10:00:00+08:00"
        r1 = self.rec()
        r1.pop("rule_version")
        r1["verdict"]["hs"].pop("rule_version")
        self.assertFalse(g11.record_allows(r1, "hs", "k1", True, ok_time))
        r1v = self.rec()
        r1v["rule_version"] = 1
        self.assertFalse(g11.record_allows(r1v, "hs", "k1", True, ok_time))
        mixed = self.rec()
        mixed["verdict"]["hs"]["rule_version"] = 1
        self.assertFalse(g11.record_allows(mixed, "hs", "k1", True, ok_time))
        self.assertEqual(g11.RULE_VERSION, 2)

    def test_attempt_recorded_from_env_before_completion(self):             # item 3
        d = tempfile.mkdtemp(prefix="c10_g11att2_")
        try:
            src, rec = os.path.join(d, "src"), os.path.join(d, "rec")
            os.makedirs(rec)
            core.stage_environment(src, "hs", {"torch": "t", "cudnn": 1, "gpu": "NVIDIA H100 80GB HBM3",
                                               "python": "3.12", "numpy": "2.5.3"}, "hs_test", "real")
            r = g11.attempt_from_results(rec, "hs", src, "real")            # no RUN_SUMMARY, no chunk
            self.assertTrue(os.path.exists(os.path.join(rec, "attempt_hs.json")))
            self.assertEqual(r["input_id"], "hs_test")
            with self.assertRaises(g11.G11Error):
                g11.colab_results(src, "hs", "seam")                       # the import itself is refused later
            with self.assertRaises(g11.G11Error):
                g11.attempt_from_results(rec, "hs", os.path.join(d, "nothing"), "real")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_zero_tripwire(self):                                           # item 9
        def out(gpu, val):
            return {"environment": {"hs": {"gpu": gpu}, "p1": {"gpu": gpu}},
                    "verdict": {"hs": {"offgrid_measured": {"a": val}}, "p1": {"offgrid_measured": {"b": val}}}}
        self.assertIn("tripwire", g11.zero_tripwire(out("NVIDIA A100-SXM4-80GB", 0.0)))
        self.assertIsNone(g11.zero_tripwire(out("NVIDIA A100-SXM4-80GB", 1e-4)))
        self.assertIsNone(g11.zero_tripwire(out("NVIDIA GeForce RTX 4090", 0.0)))


REH_INPUT = os.path.join(REPO, "data", "test11", "inputs", "rehearsal")
KIT_TREE = os.path.join(REPO, "data", "test11", "kit", "tree")
HAVE_REH = os.path.isdir(REH_INPUT) and os.path.isdir(KIT_TREE) and os.path.isdir(
    os.path.join(REPO, "data", "prereg07", "val"))


@unittest.skipUnless(HAVE_REH, "the rehearsal input / kit tree (data/test11) or the validation chunks are not here")
class RehearsalRejectsDrillData(unittest.TestCase):
    """Review blocker, end to end: drill data never becomes an allowed rehearsal record."""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(prefix="c10_g11rehdrill_")
        cls.I = core.load_input(REH_INPUT)
        with open(os.path.join(KIT_TREE, "KIT_MANIFEST.json"), encoding="utf-8") as f:
            cls.kit = json.load(f)
        cls.core_sha = cls.kit["files"][core.SELF_REL]["sha256_lf"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def build(self, name, **kw):
        out = os.path.join(self.d, name)
        for st in ("hs", "p1"):
            g11.write_val_chunks(self.I, REH_INPUT, st, out, self.kit, KIT_TREE,
                                 relabel_gpu="NVIDIA A100-SXM4-40GB (drill label)", **kw)
        return out

    def compare(self, out, name):
        return run(["rehearsal-compare", "--from", out, "--work-dir", os.path.join(self.d, "w_" + name), "--json",
                    os.path.join(self.d, name + ".json")])

    def test_cli_refuses_rehearsal_inputs(self):
        code, out = run(["fake-from-val", "--stage", "hs", "--input", REH_INPUT, "--out", os.path.join(self.d, "cli"),
                         "--kit", KIT_TREE, "--relabel-gpu", "NVIDIA A100-SXM4-40GB"], expect=2)
        self.assertIn("dry-run inputs only", out)

    def test_fake_from_val_output_rejected(self):
        code, out = self.compare(self.build("ffv"), "ffv")
        self.assertEqual(code, 2, out[-1500:])
        self.assertIn("not a complete GPU rehearsal run", out)
        self.assertFalse(os.path.exists(os.path.join(self.d, "ffv.json")))

    def test_foreign_tool_or_zero_seconds_rejected(self):
        for name, kw in (("tool", {"tool_sha": "t", "seconds": 1.0, "summary_extra": {}}),
                         ("secs", {"tool_sha": self.core_sha, "seconds": 0.0, "summary_extra": {}})):
            with self.subTest(name):
                code, out = self.compare(self.build(name, **kw), name)
                self.assertEqual(code, 2, out[-1500:])
                self.assertIn("tool_sha256" if name == "tool" else "seconds", out)
                self.assertFalse(os.path.exists(os.path.join(self.d, name + ".json")))

    def test_zero_tripwire_on_another_gpu(self):
        """Chunks that pass every origin check but carry the 4090's own numbers under another GPU name: refused by the
        tripwire (all off-grid differences exactly 0.0), no record written."""
        code, out = self.compare(self.build("trip", tool_sha=self.core_sha, seconds=1.0, summary_extra={}), "trip")
        self.assertEqual(code, 2, out[-1500:])
        self.assertIn("tripwire", out)
        self.assertFalse(os.path.exists(os.path.join(self.d, "trip.json")))

    def test_mechanics_on_the_4090_name(self):
        """The same data under the 4090's own name runs through (every difference 0); written outside the
        repository, never committed. The work folder is never emptied: a fresh subfolder per call."""
        out_dir = os.path.join(self.d, "m")
        for st in ("hs", "p1"):
            g11.write_val_chunks(self.I, REH_INPUT, st, out_dir, self.kit, KIT_TREE, tool_sha=self.core_sha,
                                 seconds=1.0, summary_extra={})
        work = os.path.join(self.d, "w_keep")
        os.makedirs(work)
        with open(os.path.join(work, "keep.txt"), "w") as f:
            f.write("x")
        code, out = run(["rehearsal-compare", "--from", out_dir, "--work-dir", work, "--json",
                         os.path.join(self.d, "m.json")])
        self.assertEqual(code, 0, out[-1500:])
        self.assertTrue(os.path.exists(os.path.join(work, "keep.txt")))
        self.assertEqual(len([x for x in os.listdir(work) if x.startswith("g11_compare_")]), 1)
        with open(os.path.join(self.d, "m.json"), encoding="utf-8") as f:
            rec = json.load(f)
        self.assertTrue(rec["verdict"]["hs"]["allowed"] and rec["verdict"]["p1"]["allowed"])
        self.assertEqual(rec["rule_version"], 2)
        self.assertEqual(rec["kit_id"], self.kit["kit_id"])
        self.assertEqual(sorted(rec["environment"]["hs"]), sorted(core.ENV_KEY_FIELDS))

if __name__ == "__main__":
    unittest.main()
