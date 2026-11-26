"""Unit tests of the PREREG_19 freeze / validation tools (tools/p19_forward.py, tools/p19_set.py) on synthetic data
and on the registered model files (read only). Never reads data/local, never runs a step, never opens a label file.
Run: python -m pytest tests/test_p19_set.py
"""

import os
import sys
import unittest

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "results", "prereg19", "explore"))
import p19_forward as SF  # noqa: E402
import p09_forward as FW  # noqa: E402
import p19_lib as K  # noqa: E402  (the exploration's specification)

MODELS = os.path.join(REPO, "results", "prereg19", "rank")
SRC = os.path.join(REPO, "data", "prereg19", "models")


def rand_params(rng):
    return {k: rng.standard_normal(SF.SHAPES[k]) * (0.05 if k.startswith(("W", "R")) else 0.1)
            for k in SF.PARAM_NAMES}


class Forward(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(19)
        self.pars = [rand_params(self.rng) for _ in range(3)]
        for p in self.pars:
            p["t0"] = np.array(-4.0)
            p["dt"] = np.full(8, np.log(np.e - 1))

    def test_equals_exploration(self):
        X = self.rng.standard_normal((50, FW.X_DIM))
        for p in self.pars:
            np.testing.assert_array_equal(SF.h_np(p, X), K.h_np(p, X))
            V = self.rng.standard_normal((7, SF.H_DIM))
            np.testing.assert_array_equal(SF.head_np(p, V), K.head_np(p, V))
        S = sparse.csr_matrix(np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 1]], dtype=float))
        usable = np.array([True, False, True, True, False])
        self.assertEqual(SF.usable_groups(S, usable), K.usable_groups(S, usable))
        self.assertEqual(SF.usable_groups(S, usable), [[0], [2, 3], []])
        H = [self.rng.standard_normal((5, SF.H_DIM)) for _ in self.pars]
        a = SF.set_term(self.pars, H, SF.usable_groups(S, usable), 5)
        b = K.set_term(self.pars, H, K.usable_groups(S, usable), 5)
        for x, y in zip((a[0], a[2], a[3]), (b[0], b[2], b[3])):
            np.testing.assert_array_equal(x, y)
        self.assertTrue(np.all(a[0][2] == 0) and not a[3][2])

    def test_question_set_term(self):
        G = 9
        means = self.rng.standard_normal((G, 2, FW.F_DIM)).astype(np.float32)
        counts = np.array([[3, 2]] * G)
        counts[4, 1] = 0                                    # no opponent move: NaN half -> 0 after standardising
        usable = np.ones(G, dtype=bool)
        usable[[1, 7]] = False
        S = sparse.csr_matrix(np.kron(np.eye(3), np.ones((1, 3))))
        mu, sd = np.zeros(FW.X_DIM), np.ones(FW.X_DIM)
        E, lp, has = SF.question_set_term(self.pars, means, counts, usable, S, mu, sd)
        Xs = FW.standardise_x(FW.x_raw(means, counts), mu, sd)
        self.assertEqual(float(Xs[4, FW.F_DIM:].sum()), 0.0)
        for q in range(3):
            g = [i for i in range(3 * q, 3 * q + 3) if usable[i]]
            lps = [SF.head_np(p, SF.h_np(p, Xs[g]).mean(0, keepdims=True)) for p in self.pars]
            want = FW.lse_mean(lps)[0]
            np.testing.assert_allclose(lp[q], want, rtol=0, atol=1e-12)
            np.testing.assert_allclose(E[q], 4 * want, rtol=0, atol=1e-12)
        self.assertTrue(has.all())
        E2, _lp2, has2 = SF.question_set_term(self.pars, means, counts, np.zeros(G, dtype=bool), S, mu, sd)
        self.assertFalse(has2.any())
        self.assertEqual(float(np.abs(E2).sum()), 0.0)

    def test_blocks_do_not_matter_beyond_last_bits(self):
        X = self.rng.standard_normal((40, FW.X_DIM))
        a = SF.game_h(self.pars, X, np.zeros(FW.X_DIM), np.ones(FW.X_DIM), block=8192)
        b = SF.game_h(self.pars, X, np.zeros(FW.X_DIM), np.ones(FW.X_DIM), block=7)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)


class Models(unittest.TestCase):
    def test_registered_files(self):
        folder = MODELS if all(os.path.exists(os.path.join(MODELS, SF.model_name(s))) for s in SF.SEEDS) else SRC
        if not os.path.exists(os.path.join(folder, SF.model_name(0))):
            self.skipTest("no model files")
        pars = SF.load_models(folder)
        self.assertEqual(len(pars), 3)
        self.assertEqual(pars[0]["W1"].dtype, np.float64)
        with self.assertRaises(SF.SetForwardError):
            SF.load_model(os.path.join(folder, SF.model_name(0)), 1)

    def test_registered_constants(self):
        self.assertEqual(SF.SEEDS, (0, 1, 2))
        self.assertEqual(SF.SCALE, 4.0)
        self.assertEqual(SF.BEST_STEP, {0: 10000, 1: 8500, 2: 7000})
        self.assertEqual(SF.PARAM_NAMES, K.PARAM_NAMES)
        self.assertEqual((SF.H_DIM, SF.BLOCK), (K.H_DIM, 8192))


class SetTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import p19_set
        cls.T = p19_set

    def test_terms_and_theta_layout(self):
        T = self.T
        self.assertEqual(T.TERMS, ["OP", "O4", "SWA", "SWB", "HS", "P1TO", "SET"])
        fc = {"format": T.FORMATS["frozen"], "combo": {"id": "Q19", "terms": T.TERMS},
              "params": {"theta": [0.0] * (2 + 7 + 9)}}
        self.assertTrue(T.frozen_usable(fc))
        fc["params"]["theta"] = [0.0] * 17
        with self.assertRaises(T.P19Error):
            T.frozen_usable(fc)

    def test_apply_decision(self):
        T = self.T
        ok = dict(same={"passed": True}, fail_count=0, repro={"Q02": True, "Q02H": True, "Q08": True, "Q09": True},
                  direction={"passed": True}, formats_ok=True, audit="missing")
        self.assertEqual(T.apply_decision(**ok)[0], "pending_audit")
        for k, v in (("same", {"passed": False}), ("fail_count", 1), ("repro", {"Q09": False}),
                     ("direction", {"passed": False, "rho": 0.0, "threshold": 0.1}), ("formats_ok", False),
                     ("audit", "bug")):
            with self.subTest(k):
                self.assertIs(T.apply_decision(**dict(ok, **{k: v}))[0], False)

    def test_audit_binding(self):
        T = self.T
        rep = {"outputs_sha256": {n: n for n in T.AUDIT_BOUND_OUTPUTS},
               "inputs_sha256": {n: n for n in T.AUDIT_BOUND_INPUTS}}
        aud = {"format": T.FORMATS["audit"], "program_errors_affecting_predictions": [],
               "outputs_sha256": dict(rep["outputs_sha256"]), "inputs_sha256": dict(rep["inputs_sha256"])}
        self.assertEqual(T.audit_binding(aud, rep, "2026-10-21T10:00:00+08:00", "2026-10-21T09:00:00+08:00"), [])
        self.assertTrue(T.audit_binding(aud, rep, "2026-10-21T08:00:00+08:00", "2026-10-21T09:00:00+08:00"))
        aud["outputs_sha256"]["val_set.npz"] = "x"
        self.assertTrue(T.audit_binding(aud, rep, "2026-10-21T10:00:00+08:00", "2026-10-21T09:00:00+08:00"))

    def test_label_path_refused(self):
        with self.assertRaises(self.T.P19Error):
            self.T.refuse_label_path(os.path.join("data", "local", "rank_val_" + "ans" + "wers.csv"))


if __name__ == "__main__":
    unittest.main()
