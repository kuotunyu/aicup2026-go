"""PREREG_08 tools/p08_p1.py: the validation-P1 job in smoke mode with the fake scorer (CPU, synthetic SGFs, output
under data/prereg08/tmp; never data/local, never the GPU), plus the pure pieces of manifest (P2 rule, move hashes).
Every run of val is a subprocess, so p1_lib / hs_lib are never imported in the pytest process (spec E).
Run: .venv/Scripts/python.exe -m pytest -q tests/test_p08_p1.py
"""

import builtins
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import unittest

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import p08_p1 as V  # noqa: E402
import p08_rank_fuse as R  # noqa: E402

TOOL = os.path.join(REPO, "tools", "p08_p1.py")
TMP = os.path.join(REPO, "data", "prereg08", "tmp")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process (spec E)"
    for mod in ("p1_lib", "p1_extract", "p1_fuse", "hs_lib", "u1_lib", "pilot_fuse", "pilot_run"):
        assert mod not in sys.modules, f"{mod} imported inside the pytest process (spec E)"


MOVES = ["pd", "dp", "pp", "dd", "qf", "nc", "fq", "cn", "jj", "kd", "jp", "cj", "qj", "jq"]


def sgf(n_moves, offset=0):
    ms = [MOVES[(i + offset) % len(MOVES)] for i in range(n_moves)]
    if len(set(ms)) != len(ms):
        ms = MOVES[:n_moves]
    return "(;SZ[19]" + "".join(f";{'B' if i % 2 == 0 else 'W'}[{m}]" for i, m in enumerate(ms)) + ")"


def write_questions(path, games, per_q=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "num_games"] + [f"color_{i}" for i in range(1, per_q + 1)]
                   + [f"sgf_{i}" for i in range(1, per_q + 1)])
        for q in range(0, len(games), per_q):
            part = games[q:q + per_q]
            w.writerow([f"T{q // per_q:04d}", len(part)] + [c for c, _ in part] + [""] * (per_q - len(part))
                       + [s for _, s in part] + [""] * (per_q - len(part)))


def run(*args, timeout=900):
    return subprocess.run([sys.executable, TOOL] + list(args), capture_output=True, text=True, cwd=REPO, env=ENV,
                          timeout=timeout)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class SmokeFake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(TMP, f"pytest_p08_{os.getpid()}")
        shutil.rmtree(cls.root, ignore_errors=True)
        # 430 games -> 3 chunks (200 + 200 + 30); colours alternate; lengths 6..14 moves
        cls.N = 430
        cls.games = [("B" if i % 2 == 0 else "W", sgf(6 + i % 9, i)) for i in range(cls.N)]
        cls.q = os.path.join(cls.root, "questions.csv")
        write_questions(cls.q, cls.games)
        cls.out = os.path.join(cls.root, "run1")
        cls.r = run("val", "--smoke", "--fake-scorer", "--workers", "2", "--questions", cls.q, "--out-dir", cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_run_and_outputs(self):
        self.assertEqual(self.r.returncode, 0, self.r.stdout[-3000:] + self.r.stderr[-3000:])
        vp = load(os.path.join(self.out, "feat", "val_p1.npz"))
        N = self.N
        self.assertEqual(vp["counts"].shape, (N, 2))
        self.assertEqual(vp["scal"].shape, (N, 2, 2))
        self.assertEqual(vp["slot"][:6].tolist(), [1, 2, 3, 4, 5, 1])
        self.assertEqual([str(x) for x in vp["color"][:3]], ["B", "W", "B"])
        self.assertEqual(vp["ref"].tolist(), list(range(N)))
        n = np.array([6 + i % 9 for i in range(N)])
        self.assertEqual(vp["n_moves"].tolist(), n.tolist())
        tgt = np.where(np.arange(N) % 2 == 0, (n + 1) // 2, n // 2)       # B moves first
        self.assertEqual(vp["counts"][:, 0].tolist(), tgt.tolist())
        self.assertEqual(vp["counts"][:, 1].tolist(), (n - tgt).tolist())
        self.assertFalse(vp["failed"].any())
        # chunks: every field of run_chunk plus the val fields; scal = the means' log p / entropy columns
        chunks = sorted(os.listdir(os.path.join(self.out, "val")))
        self.assertEqual(chunks, ["chunk_00000.npz", "chunk_00001.npz", "chunk_00002.npz"])
        c0 = load(os.path.join(self.out, "val", "chunk_00000.npz"))
        for k in ("means", "counts", "n_moves", "failed", "errors", "pos_game", "pos_side", "pos_turn", "pos_logp",
                  "pos_entropy", "format", "ref", "keys_sha256", "questions_sha256", "ckpt_sha256", "torch", "cudnn",
                  "gpu_name", "driver", "meta_sha256", "tool_sha256", "mode", "created_utc", "seconds"):
            self.assertIn(k, c0, k)
        self.assertEqual(str(c0["format"]), R.FORMATS["val_chunk"])
        self.assertEqual(str(c0["meta_sha256"]), R.NEUTRAL_META_SHA)
        self.assertEqual(c0["means"].shape, (200, 2, 770))
        np.testing.assert_array_equal(vp["scal"][:200], c0["means"][:, :, 768:770].astype(np.float64))
        # per-game target log p mean = mean of the per-position log p of the target side
        g = c0["pos_game"]
        tgt_rows = c0["pos_side"] == 0
        s = np.zeros(200)
        np.add.at(s, g[tgt_rows], c0["pos_logp"][tgt_rows].astype(np.float64))
        np.testing.assert_allclose(s / c0["counts"][:, 0], vp["scal"][:200, 0, 0], rtol=0, atol=1e-5)
        seam = load(os.path.join(self.out, "feat", "val_seam_feats.npz"))
        self.assertEqual(seam["means"].shape, (400, 2, 770))
        self.assertEqual(seam["means"].dtype, np.float32)
        self.assertEqual(int(seam["games"]), 400)
        self.assertEqual(len(seam["chunk_sha256"]), 2)
        np.testing.assert_array_equal(seam["means"][:200], c0["means"])
        with open(os.path.join(self.out, "feat", "val_report.json"), encoding="utf-8") as f:
            rep = json.load(f)
        self.assertEqual((rep["status"], rep["mode"], rep["games"], len(rep["chunks"])), ("ok", "smoke", N, 3))
        self.assertEqual(rep["val_p1_sha256"], R.sha_raw(os.path.join(self.out, "feat", "val_p1.npz")))
        self.assertEqual(rep["chunks"][0]["sha256"], R.sha_raw(os.path.join(self.out, "val", "chunk_00000.npz")))
        self.assertTrue(rep["first_chunk_started_utc"])

    def test_resume_skips_chunks_and_reassembles_identically(self):
        self.assertEqual(self.r.returncode, 0)
        out2 = os.path.join(self.root, "resume")
        shutil.copytree(os.path.join(self.out, "val"), os.path.join(out2, "val"))
        os.remove(os.path.join(out2, "val", "chunk_00002.npz"))
        r = run("val", "--smoke", "--fake-scorer", "--workers", "2", "--questions", self.q, "--out-dir", out2)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn('"todo": 1', r.stdout)
        a = load(os.path.join(self.out, "feat", "val_p1.npz"))
        b = load(os.path.join(out2, "feat", "val_p1.npz"))
        for k in ("scal", "counts", "n_moves", "failed"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)

    def test_report_exists_is_refused(self):
        self.assertEqual(self.r.returncode, 0)
        r = run("val", "--smoke", "--fake-scorer", "--questions", self.q, "--out-dir", self.out)
        self.assertEqual(r.returncode, 2)

    def test_chunk_with_other_fields_or_environment_is_refused(self):
        self.assertEqual(self.r.returncode, 0)
        src = load(os.path.join(self.out, "val", "chunk_00000.npz"))
        for field, value, msg in (("cudnn", "12345", "another environment"),
                                  ("keys_sha256", "0" * 64, "differs from this run"),
                                  ("meta_sha256", "1" * 64, "differs from this run")):
            out3 = os.path.join(self.root, f"diff_{field}")
            os.makedirs(os.path.join(out3, "val"))
            d = dict(src)
            d[field] = np.array(value)
            np.savez(os.path.join(out3, "val", "chunk_00000.npz"), **d)
            before = R.sha_raw(os.path.join(out3, "val", "chunk_00000.npz"))
            r = run("val", "--smoke", "--fake-scorer", "--questions", self.q, "--out-dir", out3)
            self.assertEqual(r.returncode, 2, field)
            self.assertIn(msg, r.stderr, field)
            self.assertEqual(R.sha_raw(os.path.join(out3, "val", "chunk_00000.npz")), before, "never overwritten")

    def test_stop_file(self):
        out4 = os.path.join(self.root, "stop")
        os.makedirs(os.path.join(out4, "val"))
        open(os.path.join(out4, "val", "STOP"), "w").close()
        r = run("val", "--smoke", "--fake-scorer", "--questions", self.q, "--out-dir", out4)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertFalse([f for f in os.listdir(os.path.join(out4, "val")) if f.startswith("chunk_")])
        self.assertFalse(os.path.exists(os.path.join(out4, "feat")))

    def test_status(self):
        r = run("status", "--out-dir", self.out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["chunks_present"], 3)


class FailureStop(unittest.TestCase):
    def test_failed_and_moveless_games_stop(self):
        root = os.path.join(TMP, f"pytest_p08_fail_{os.getpid()}")
        shutil.rmtree(root, ignore_errors=True)
        try:
            games = [("B", sgf(8)), ("W", "(;SZ[19];B[pd];B[dp])"), ("B", "(;SZ[19])"), ("W", sgf(9))]
            q = os.path.join(root, "q.csv")
            write_questions(q, games, per_q=4)
            r = run("val", "--smoke", "--fake-scorer", "--workers", "1", "--questions", q, "--out-dir",
                    os.path.join(root, "o"))
            self.assertEqual(r.returncode, 1, r.stdout[-2000:] + r.stderr[-2000:])
            with open(os.path.join(root, "o", "feat", "val_report.json"), encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["status"], "stop")
            self.assertEqual((rep["failed"], rep["failed_or_no_move"]), (1, 2))
        finally:
            shutil.rmtree(root, ignore_errors=True)


TRACE = r'''
import builtins, io, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "tools")
LOG = []
_open, _stat, _listdir, _scandir = builtins.open, os.stat, os.listdir, os.scandir
def bad(p):
    try:
        s = os.path.normcase(os.path.abspath(os.fspath(p)))
    except TypeError:
        return False
    return ("data" + os.sep + "local") in s
def t_open(f, *a, **k):
    if not isinstance(f, int):
        LOG.append(("open", os.fspath(f), "p1_lib" in sys.modules, bad(f)))
    return _open(f, *a, **k)
def t_stat(p, *a, **k):
    if bad(p):
        LOG.append(("stat", os.fspath(p), "p1_lib" in sys.modules, True))
    return _stat(p, *a, **k)
def t_listdir(p="."):
    if bad(p):
        LOG.append(("listdir", os.fspath(p), "p1_lib" in sys.modules, True))
    return _listdir(p)
def t_scandir(p="."):
    if bad(p):
        LOG.append(("scandir", os.fspath(p), "p1_lib" in sys.modules, True))
    return _scandir(p)
builtins.open = t_open; io.open = t_open; os.stat = t_stat; os.listdir = t_listdir; os.scandir = t_scandir
import p08_rank_fuse as R
import p08_p1 as V
MODE = sys.argv[1]
if MODE == "real":
    R.OUT = os.path.join(R.TMP, sys.argv[2])     # an empty folder: frozen_config.json is "not committed"
    rc = V.main(["val"])
else:
    rc = V.main(["val", "--smoke", "--fake-scorer", "--workers", "1", "--questions", sys.argv[2], "--out-dir",
                 sys.argv[3]])
print(json.dumps({"rc": rc, "log": LOG}))
'''


class Ordering(unittest.TestCase):
    def trace(self, *args):
        r = subprocess.run([sys.executable, "-c", TRACE] + list(args), capture_output=True, text=True, cwd=REPO,
                           env=ENV, timeout=900)
        self.assertTrue(r.stdout.strip(), r.stderr[-3000:])
        return json.loads(r.stdout.strip().splitlines()[-1]), r

    def test_real_mode_refused_before_any_data_local_access(self):
        res, r = self.trace("real", f"pytest_no_frozen_{os.getpid()}")
        self.assertEqual(res["rc"], 2, r.stderr[-2000:])
        self.assertIn("frozen_config.json", r.stderr)
        self.assertEqual([x for x in res["log"] if x[3]], [], "data/local touched before the refusal")

    def test_question_file_read_before_p1_lib(self):
        root = os.path.join(TMP, f"pytest_p08_order_{os.getpid()}")
        shutil.rmtree(root, ignore_errors=True)
        try:
            q = os.path.join(root, "q.csv")
            write_questions(q, [("B", sgf(8)), ("W", sgf(9))], per_q=2)
            res, r = self.trace("smoke", q, os.path.join(root, "o"))
            self.assertEqual(res["rc"], 0, r.stderr[-2000:])
            qopens = [x for x in res["log"] if x[0] == "open" and os.path.normcase(x[1]) == os.path.normcase(q)]
            self.assertTrue(qopens, "the question file was not opened")
            self.assertFalse(any(x[2] for x in qopens), "p1_lib was imported before the question file was read")
            self.assertTrue(any(x[2] for x in res["log"]), "p1_lib was never imported")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class Flags(unittest.TestCase):
    def test_smoke_never_reads_data_local(self):
        r = run("val", "--smoke", "--fake-scorer", "--questions", os.path.join(REPO, "data", "local", "x.csv"),
                "--out-dir", os.path.join(TMP, "never"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("never read data/local", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(TMP, "never")))

    def test_smoke_output_only_under_tmp(self):
        r = run("val", "--smoke", "--fake-scorer", "--questions", os.path.join(TMP, "x.csv"), "--out-dir",
                os.path.join(REPO, "results", "prereg08", "feat"))
        self.assertEqual(r.returncode, 2)

    def test_real_mode_flags_refused_before_anything(self):
        self.assertEqual(run("val", "--max-games", "5").returncode, 2)
        self.assertEqual(run("val", "--fake-scorer").returncode, 2)


class P2Rule(unittest.TestCase):
    before = datetime.datetime(2026, 10, 7, 12, 0, tzinfo=datetime.timezone.utc)
    after = datetime.datetime(2026, 10, 9, 0, 0, tzinfo=datetime.timezone.utc)

    def test_incomplete_before_cutoff_is_refused(self):
        with self.assertRaises(R.P08Error):
            R.p2_decision(599, 600, True, 0, self.before)
        with self.assertRaises(R.P08Error):
            R.p2_decision(600, 600, False, 0, self.before)          # present but a chunk fails the check

    def test_complete_is_used(self):
        rec = R.p2_decision(600, 600, True, 0, self.before)
        self.assertTrue(rec["used"] and rec["complete"])
        self.assertTrue(R.p2_decision(600, 600, True, 600, self.before)["used"])

    def test_too_many_failures_not_used(self):
        rec = R.p2_decision(600, 600, True, 601, self.before)
        self.assertTrue(rec["complete"])
        self.assertFalse(rec["used"])

    def test_after_cutoff_incomplete_not_used(self):
        rec = R.p2_decision(420, 600, True, 0, self.after)
        self.assertFalse(rec["used"] or rec["complete"])
        self.assertIn("incomplete", rec["reason"])

    def test_cutoff_is_taipei_time(self):
        self.assertEqual(R.P2_CUTOFF.isoformat(), "2026-10-08T23:59:59+08:00")
        just_before = datetime.datetime(2026, 10, 8, 15, 59, 58, tzinfo=datetime.timezone.utc)
        with self.assertRaises(R.P08Error):
            R.p2_decision(10, 600, True, 0, just_before)
        self.assertFalse(R.p2_decision(10, 600, True, 0, just_before + datetime.timedelta(seconds=2))["used"])


class MoveHashes(unittest.TestCase):
    def test_same_moves_same_hash_and_orientation_matters(self):
        a = "(;SZ[19]PB[x];B[pd];W[dp];B[pp])"
        b = "(;SZ[19]PW[y]RE[B+R];B[pd];W[dp];B[pp])"
        c = "(;SZ[19];B[dp];W[pd];B[dd])"                     # a mirror image is a different sequence
        self.assertEqual(V.move_hash(a), V.move_hash(b))
        self.assertNotEqual(V.move_hash(a), V.move_hash(c))
        self.assertIsNone(V.move_hash("(;SZ[19])"))

    def test_duplicate_refs(self):
        self.assertEqual(V.duplicate_refs([5, 7, 9, 11], ["a", "b", None, "a"], {"a", None}), [5, 11])
        self.assertEqual(V.duplicate_refs([1], [None], {None}), [])

    def test_chunk_plan(self):
        self.assertEqual(V.chunk_plan(430, 200), [(0, 0, 200), (1, 200, 400), (2, 400, 430)])
        self.assertEqual(len(V.chunk_plan(R.VAL_GAMES, 200)), 96)


if __name__ == "__main__":
    unittest.main()
