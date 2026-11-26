"""PREREG_07 tools/p07_hs.py: the validation-HS job in smoke mode with the fake scorer (CPU, synthetic SGFs, output
under data/prereg07/tmp; never data/local, never the GPU). Every run is a subprocess, so hs_lib / pilot_run are never
imported in the pytest process (spec 0). Covered: chunking, assembly (val_hs.npz, val_seam_moves.npz, val_report.json),
resume (existing chunks skipped), refusal of a chunk written with another environment, STOP, the failure stop, and the
smoke / real flag rules.
Run: .venv/Scripts/python.exe -m pytest -q tests/test_p07_hs.py
"""

import builtins
import csv
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
TOOL = os.path.join(REPO, "tools", "p07_hs.py")
TMP = os.path.join(REPO, "data", "prereg07", "tmp")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process (spec 0)"


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


class SmokeFake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(TMP, f"pytest_hs_{os.getpid()}")
        shutil.rmtree(cls.root, ignore_errors=True)
        # 230 games -> 2 chunks (200 + 30); colours alternate; lengths 6..14 moves
        cls.games = [("B" if i % 2 == 0 else "W", sgf(6 + i % 9, i)) for i in range(230)]
        cls.q = os.path.join(cls.root, "questions.csv")
        write_questions(cls.q, cls.games)
        cls.out = os.path.join(cls.root, "run1")
        cls.r = run("val", "--smoke", "--fake-scorer", "--workers", "2", "--questions", cls.q, "--out-dir", cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_run_and_outputs(self):
        self.assertEqual(self.r.returncode, 0, self.r.stdout[-3000:] + self.r.stderr[-3000:])
        with np.load(os.path.join(self.out, "hs", "val_hs.npz"), allow_pickle=False) as z:
            vh = {k: z[k] for k in z.files}
        self.assertEqual(len(vh["counts"]), 230)
        self.assertEqual(vh["sums"].shape, (230, 14))
        self.assertEqual(vh["slot"][:6].tolist(), [1, 2, 3, 4, 5, 1])
        self.assertEqual([str(x) for x in vh["color"][:3]], ["B", "W", "B"])
        n = np.array([6 + i % 9 for i in range(230)])
        self.assertEqual(vh["n_moves"].tolist(), n.tolist())
        tgt = np.where(np.arange(230) % 2 == 0, (n + 1) // 2, n // 2)       # B moves first
        self.assertEqual(vh["counts"].tolist(), tgt.tolist())
        self.assertFalse(vh["failed"].any())
        with open(os.path.join(self.out, "hs", "val_report.json"), encoding="utf-8") as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "ok")
        self.assertEqual(rep["mode"], "smoke")
        self.assertEqual(len(rep["chunks"]), 2)
        self.assertEqual(rep["games"], 230)
        with np.load(os.path.join(self.out, "hs", "val_seam_moves.npz"), allow_pickle=False) as z:
            self.assertEqual(int(z["games"]), 230)               # first two chunks = all 230 games here
            self.assertEqual(len(z["pos_logp"]), int(vh["counts"].sum()))
            self.assertEqual(z["pos_logp"].dtype, np.float32)
            g = z["pos_game"]
            self.assertEqual(int(g.max()), 229)
            # per-game sums of the per-move log p equal val_hs sums (float32 -> float64 summation as hs_lib)
            s = np.zeros((230, 14))
            np.add.at(s, g, z["pos_logp"].astype(np.float64))
            np.testing.assert_allclose(s, vh["sums"], rtol=0, atol=1e-9)

    def test_resume_skips_chunks_and_reassembles_identically(self):
        self.assertEqual(self.r.returncode, 0)
        out2 = os.path.join(self.root, "resume")
        shutil.copytree(os.path.join(self.out, "val"), os.path.join(out2, "val"))
        r = run("val", "--smoke", "--fake-scorer", "--workers", "2", "--questions", self.q, "--out-dir", out2)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn('"todo": 0', r.stdout)
        a = np.load(os.path.join(self.out, "hs", "val_hs.npz"))
        b = np.load(os.path.join(out2, "hs", "val_hs.npz"))
        for k in ("sums", "counts", "n_moves", "failed"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)

    def test_report_exists_is_refused(self):
        self.assertEqual(self.r.returncode, 0)
        r = run("val", "--smoke", "--fake-scorer", "--questions", self.q, "--out-dir", self.out)
        self.assertEqual(r.returncode, 2)

    def test_chunk_from_another_environment_is_refused(self):
        self.assertEqual(self.r.returncode, 0)
        out3 = os.path.join(self.root, "envdiff")
        os.makedirs(os.path.join(out3, "val"))
        src = os.path.join(self.out, "val", "chunk_00000.npz")
        with np.load(src, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["cudnn"] = np.array("12345")
        np.savez(os.path.join(out3, "val", "chunk_00000.npz"), **d)
        r = run("val", "--smoke", "--fake-scorer", "--questions", self.q, "--out-dir", out3)
        self.assertEqual(r.returncode, 2)
        self.assertIn("another environment", r.stderr)

    def test_stop_file(self):
        out4 = os.path.join(self.root, "stop")
        os.makedirs(os.path.join(out4, "val"))
        open(os.path.join(out4, "val", "STOP"), "w").close()
        r = run("val", "--smoke", "--fake-scorer", "--questions", self.q, "--out-dir", out4)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertFalse([f for f in os.listdir(os.path.join(out4, "val")) if f.startswith("chunk_")])
        self.assertFalse(os.path.exists(os.path.join(out4, "hs")))


class FailureStop(unittest.TestCase):
    def test_failed_and_moveless_games_stop(self):
        root = os.path.join(TMP, f"pytest_hs_fail_{os.getpid()}")
        shutil.rmtree(root, ignore_errors=True)
        try:
            games = [("B", sgf(8)), ("W", "(;SZ[19];B[pd];B[dp])"), ("B", "(;SZ[19])"), ("W", sgf(9))]
            q = os.path.join(root, "q.csv")
            write_questions(q, games, per_q=4)
            r = run("val", "--smoke", "--fake-scorer", "--workers", "1", "--questions", q, "--out-dir",
                    os.path.join(root, "o"))
            self.assertEqual(r.returncode, 1, r.stdout[-2000:] + r.stderr[-2000:])
            with open(os.path.join(root, "o", "hs", "val_report.json"), encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["status"], "stop")
            self.assertEqual((rep["failed"], rep["failed_or_no_move"]), (1, 2))
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
                os.path.join(REPO, "results", "prereg07", "hs"))
        self.assertEqual(r.returncode, 2)

    def test_real_mode_flags_refused_before_anything(self):
        # argument-level refusals only: a real-mode run is never started from a test
        self.assertEqual(run("val", "--max-games", "5").returncode, 2)
        self.assertEqual(run("val", "--fake-scorer").returncode, 2)


if __name__ == "__main__":
    unittest.main()
