"""prereg51: small CPU tests of p51_eval.py (synthetic data only; no score of a prereg51 arm).

    PY -m pytest -q results/prereg51/explore/test_p51.py
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p51_eval as PE  # noqa: E402


def write_npz(path, sha="a" * 64, n=6, count=None, rows=None, cfg=None):
    cfg = {"tta": 1, "weights": "raw", "games": n, "moves": 10 * n, "ckpt_sha256": sha} if cfg is None else cfg
    np.savez(path, emb=np.zeros((n, 4), np.float32), count=np.arange(n) if count is None else count,
             store_row=np.arange(n) if rows is None else rows, players_sha256_sorted_ids=np.array("p"),
             ckpt_sha256=np.array(sha), ckpt=np.array("x"), embed_cfg=np.array(json.dumps(cfg)))


class Numbering(unittest.TestCase):
    def test_e_use(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(PE.expected_e_use(d), 7)
            open(os.path.join(d, "eval_L.json"), "w").write("{}")
            self.assertEqual(PE.expected_e_use(d), 8)
            open(os.path.join(d, "seam.json"), "w").write("{}")          # not a judgement
            self.assertEqual(PE.expected_e_use(d), 8)

    def test_other(self):
        self.assertEqual(PE.other("L"), "B")
        self.assertEqual(PE.other("B"), "L")


class Base(unittest.TestCase):
    def test_no_other_eval_is_v27(self):
        with tempfile.TemporaryDirectory() as d:
            b = PE.resolve_base("B", here=d)
            self.assertEqual((b["name"], b["key"], b["e_mean"]), ("c49", "c49", PE.V27_E))
            self.assertEqual(len(b["e_per_pool"]), 20)

    def test_other_rejected_is_v27(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"adopt": False, "means": {"L": 0.8}, "per_pool": {"L": [0.8] * 20}},
                      open(os.path.join(d, "eval_L.json"), "w"))
            self.assertEqual(PE.resolve_base("B", here=d)["name"], "c49")

    def test_other_adopted_is_base(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"adopt": True, "means": {"L": 0.8}, "per_pool": {"L": [0.8] * 20}},
                      open(os.path.join(d, "eval_L.json"), "w"))
            b = PE.resolve_base("B", here=d)
            self.assertEqual((b["name"], b["members"], b["key"], b["e_mean"]), ("L", ["L"], "L", 0.8))
            self.assertTrue(b["apply"].endswith("results/prereg51/apply/L"))


class ArmFile(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.ref = os.path.join(self.d.name, "ref.npz")
        write_npz(self.ref, sha="c" * 64)

    def tearDown(self):
        self.d.cleanup()

    def test_good(self):
        p = os.path.join(self.d.name, "a.npz")
        write_npz(p)
        PE.check_arm_file(p, self.ref, "a" * 64)

    def test_bad(self):
        cases = {"sha": dict(sha="b" * 64), "count": dict(count=np.arange(6) + 1),
                 "rows": dict(rows=np.arange(6) + 1),
                 "tta": dict(cfg={"tta": 8, "weights": "raw", "games": 6, "moves": 60}),
                 "ema": dict(cfg={"tta": 1, "weights": "ema", "games": 6, "moves": 60}),
                 "games": dict(cfg={"tta": 1, "weights": "raw", "games": 5, "moves": 60})}
        for name, kw in cases.items():
            p = os.path.join(self.d.name, name + ".npz")
            write_npz(p, **kw)
            with self.assertRaises(SystemExit, msg=name):
                PE.check_arm_file(p, self.ref, "a" * 64)

    def test_missing(self):
        with self.assertRaises(SystemExit):
            PE.check_arm_file(None, self.ref, "a" * 64)


class Paths(unittest.TestCase):
    def test_arm_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(PE.arm_dir("L", repo=d))
            os.makedirs(os.path.join(d, "data", "prereg51", "emb_L", "V2_step_480000_raw_tta1"))
            self.assertTrue(PE.arm_file("L", "E", repo=d).endswith("store_E.npz"))
            os.makedirs(os.path.join(d, "data", "prereg51", "emb_L", "X_step_480000_raw_tta1"))
            with self.assertRaises(SystemExit):
                PE.arm_dir("L", repo=d)


if __name__ == "__main__":
    unittest.main()
