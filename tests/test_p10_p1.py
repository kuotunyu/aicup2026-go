"""tools/p10_p1.py additions after 15b3e98 (PREREG_10 rev 1 part B): manifest on a synthetic folder, and the refusals
of extract --set val that happen before anything is read. Never data/local, never p1_lib / torch, never the GPU.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q tests/test_p10_p1.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import p10_p1 as T  # noqa: E402


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Manifest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="p10p1_")
        self.saved = (T.OUT, T.SETS, T.MANIFEST, T.CHUNK)
        T.OUT = self.d
        T.SETS = os.path.join(self.d, "sets.json")
        T.MANIFEST = os.path.join(self.d, "feat", "mock_manifest.json")
        T.CHUNK = 3
        with open(T.SETS, "w", encoding="utf-8") as f:
            json.dump({"format": "c10-prereg10 mock sets v1", "games": [{"k": i} for i in range(5)]}, f)

    def tearDown(self):
        T.OUT, T.SETS, T.MANIFEST, T.CHUNK = self.saved
        shutil.rmtree(self.d, ignore_errors=True)

    def chunk(self, ci, refs, set_="p10mock", sets_sha=None):
        n = len(refs)
        np.savez(os.path.join(self.d, "chunk_%05d.npz" % ci), format=np.array("c10-prereg08 p1 chunk v1"),
                 set=np.array(set_), sides=np.array(["target", "opponent"]), ref=np.asarray(refs, np.int64),
                 precision=np.array("fp16"), meta_sha256=np.array("m"), ckpt_bytes=np.array(1),
                 code_sha256=np.array("c"), sets_sha256=np.array(sets_sha or sha(T.SETS)),
                 means=np.zeros((n, 2, 770), np.float32), failed=np.zeros(n, bool), counts=np.ones((n, 2), np.int32))

    def test_incomplete_writes_nothing(self):
        self.chunk(0, [0, 1, 2])
        self.assertEqual(T.cmd_manifest(types.SimpleNamespace(set="mock")), 1)
        self.assertFalse(os.path.exists(T.MANIFEST))

    def test_wrong_refs_or_sets(self):
        self.chunk(0, [0, 1, 2])
        self.chunk(1, [3, 5])
        self.assertEqual(T.cmd_manifest(types.SimpleNamespace(set="mock")), 1)
        self.chunk(1, [3, 4], sets_sha="0" * 64)
        self.assertEqual(T.cmd_manifest(types.SimpleNamespace(set="mock")), 1)
        self.assertFalse(os.path.exists(T.MANIFEST))

    def test_complete(self):
        self.chunk(0, [0, 1, 2])
        self.chunk(1, [3, 4])
        self.assertEqual(T.cmd_manifest(types.SimpleNamespace(set="mock")), 0)
        with open(T.MANIFEST, encoding="utf-8") as f:
            m = json.load(f)
        self.assertTrue(m["complete"])
        self.assertEqual([c["sha256"] for c in m["chunks"]],
                         [sha(os.path.join(self.d, "chunk_%05d.npz" % i)) for i in (0, 1)])
        with self.assertRaises(SystemExit):
            T.cmd_manifest(types.SimpleNamespace(set="mock"))          # written once

    def test_val_set_refused(self):
        with self.assertRaises(SystemExit):
            T.cmd_manifest(types.SimpleNamespace(set="val"))


def ns(**kw):
    base = {"set": "val", "smoke": False, "candidates": None, "questions": None, "out_dir": None, "max_games": 0,
            "fake_scorer": False, "workers": 1}
    base.update(kw)
    return types.SimpleNamespace(**base)


class ValRefusals(unittest.TestCase):
    def test_real_needs_the_freeze(self):
        if os.path.exists(os.path.join(REPO, "results", "prereg10", "player", "frozen_config.json")):
            self.skipTest("the freeze exists on this machine")
        with self.assertRaises(SystemExit) as cm:
            T.cmd_extract(ns())
        self.assertIn("after the freeze", str(cm.exception))

    def test_real_takes_no_overrides(self):
        for k, v in (("candidates", "x.csv"), ("out_dir", "x"), ("max_games", 3), ("fake_scorer", True)):
            with self.assertRaises(SystemExit) as cm:
                T.cmd_extract(ns(**{k: v}))
            self.assertIn("real mode takes no", str(cm.exception))

    def test_smoke_never_reads_data_local(self):
        loc = os.path.join(REPO, "data", "local", "player_val_candidates.csv")
        with self.assertRaises(SystemExit) as cm:
            T.cmd_extract(ns(smoke=True, candidates=loc, questions=loc,
                             out_dir=os.path.join(REPO, "data", "prereg10", "tmp", "x")))
        self.assertIn("never read data/local", str(cm.exception))

    def test_smoke_out_dir_under_tmp(self):
        for od in (tempfile.gettempdir(), os.path.join(REPO, "data", "prereg10", "tmp"),
                   os.path.join(REPO, "results", "prereg10", "feat")):
            with self.assertRaises(SystemExit) as cm:
                T.cmd_extract(ns(smoke=True, candidates="a.csv", questions="b.csv", out_dir=od))
            self.assertIn("under data/prereg10/tmp", str(cm.exception))

    def test_mock_refuses_val_flags(self):
        import subprocess
        for extra in (["--smoke"], ["--out-dir", "x"], ["--candidates", "c.csv"], ["--fake-scorer"]):
            r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "p10_p1.py"), "extract", "--set", "mock"]
                               + extra, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            self.assertNotEqual(r.returncode, 0, extra)
            self.assertIn("for --set val only", r.stderr, extra)

    def test_mock_path_unchanged_for_other_sets(self):
        with self.assertRaises(SystemExit) as cm:
            T.cmd_extract(ns(set="probe"))
        self.assertIn("only --set mock", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
