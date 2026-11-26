"""Unit tests of the PREREG_25 freeze / adoption tools (tools/p25_forward.py, tools/p25_set.py) on synthetic data and on
the registered model files (read only). Never reads data/local, never runs a step, never opens a label file.
Run: python -m pytest tests/test_p25_set.py
"""

import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import p19_forward as SF  # noqa: E402
import p25_forward as SF25  # noqa: E402

MODELS = os.path.join(REPO, "results", "prereg25", "rank")
SRC = os.path.join(REPO, "data", "prereg25", "models")


class Forward(unittest.TestCase):
    def test_computation_is_19s(self):
        """Only the loader is new: every computation is p19_forward's function object."""
        for n in ("game_h", "set_term", "question_set_term", "single_game_logp", "usable_groups", "mean_matrix"):
            self.assertIs(getattr(SF25, n), getattr(SF, n))
        self.assertEqual(SF25.SCALE, 4.0)
        self.assertEqual(SF25.SHAPES, SF.SHAPES)
        self.assertEqual(SF25.PARAM_NAMES, SF.PARAM_NAMES)

    def test_registered_constants(self):
        self.assertEqual(SF25.MODEL_FORMAT, "c10-prereg25 s25 model v1")
        self.assertEqual(SF25.BEST_STEP, {0: 11000, 1: 7500, 2: 10500})
        self.assertEqual(SF25.MODEL_SHA256[0], "4a01d6a60355cd514e9cc4988c08424a7952fc33302b47a63542a8caf67163bb")
        self.assertEqual(SF25.MODEL_SHA256[1], "ab53d69b2762a7c01bed354ac6c218cf167fda44bbd3d88eed06a6f4adf3cba4")
        self.assertEqual(SF25.MODEL_SHA256[2], "20e190f56866c7a28b4eeef0c0d4d656867b79a908e066ba91a47570601c4fbb")
        self.assertEqual(SF25.model_name(1), "s25_seed1.npz")

    def test_wrong_file_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s25_seed0.npz")
            rng = np.random.default_rng(0)
            np.savez(p, format=np.array(SF25.MODEL_FORMAT), seed=np.array(0), best_step=np.array(11000),
                     smoke=np.array(False), **{f"p_{k}": rng.standard_normal(SF25.SHAPES[k]).astype(np.float32)
                                               for k in SF25.PARAM_NAMES})
            with self.assertRaises(SF25.SetForwardError):
                SF25.load_model(p, 0)
            with self.assertRaises(SF25.SetForwardError):     # an S19 file is never an S25 file
                SF25.load_model(os.path.join(REPO, "results", "prereg19", "rank", "s19_seed0.npz"), 0)

    @unittest.skipUnless(all(os.path.exists(os.path.join(MODELS, SF25.model_name(s))) for s in SF25.SEEDS),
                         "results/prereg25/rank models not frozen yet")
    def test_registered_files(self):
        pars = SF25.load_models(MODELS)
        self.assertEqual(len(pars), 3)
        for p in pars:
            for k in SF25.PARAM_NAMES:
                self.assertEqual(p[k].dtype, np.float64)
                self.assertEqual(p[k].shape, SF25.SHAPES[k])


class SetTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import p25_set
        cls.P = p25_set

    def test_terms_and_paths(self):
        P = self.P
        self.assertEqual(P.TERMS, ["OP", "O4", "SWA", "SWB", "HS", "P1TO", "SET"])
        self.assertEqual(P.REF_Q25, {"inner": 0.7456519871546352, "mono": 0.7482450052195216})
        self.assertEqual(P.REF_Q19, {"inner": 0.7345843051712626, "mono": 0.7430172362615272})
        self.assertEqual(P.EXPLORE_COMMIT, "1965276")
        self.assertTrue(P.OUT.replace("\\", "/").endswith("results/prereg25/rank"))
        self.assertIn("tools/p25_set.py", P.TOOL_FILES)
        self.assertIn("tools/p19_forward.py", P.TOOL_FILES)
        self.assertEqual(len(set(P.TOOL_FILES)), len(P.TOOL_FILES))

    def test_fold_weight_diff(self):
        P = self.P
        w = {k: 0.5 for k in P.TERMS}
        cand = {"folds": [{"params": {"w": dict(w)}} for _ in range(5)]}
        self.assertEqual(P.fold_weight_diff(cand, [dict(w) for _ in range(5)]), 0.0)
        w2 = dict(w, SET=0.75)
        self.assertEqual(P.fold_weight_diff(cand, [dict(w)] * 4 + [w2]), 0.25)
        self.assertIsNone(P.fold_weight_diff(cand, [dict(w)] * 4))
        self.assertIsNone(P.fold_weight_diff(cand, [{"OP": 1.0}] * 5))

    def test_audit_binding(self):
        P = self.P
        bad = P.audit_binding({"format": "x"}, "2026-11-06T10:00:00+08:00", "2026-11-06T09:00:00+08:00")
        self.assertTrue(any("format" in b for b in bad))
        self.assertTrue(any("program_errors" in b for b in bad))
        self.assertTrue(any("not later" in b for b in bad))

    def test_no_data_local_or_label_in_source(self):
        for f in ("p25_set.py", "p25_forward.py"):
            with open(os.path.join(REPO, "tools", f), encoding="utf-8") as fh:
                src = fh.read().lower()
            self.assertNotIn("rank_val_questions", src)
            self.assertNotIn("rank_val_answers", src)
            self.assertNotIn('"local"', src)
            self.assertNotIn("import torch", src)


if __name__ == "__main__":
    unittest.main()
