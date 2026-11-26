"""Unit tests of the prereg17 exploration helpers on synthetic data (no competition data, no GPU).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg17/explore/test_p17.py
"""

import os
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import p17_lib as K  # noqa: E402
import p17_tabicl as TT  # noqa: E402


class Pca(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(6, 6))
        self.X = rng.normal(size=(500, 6)) @ A + 3.0

    def test_streaming_moments_equal_numpy(self):
        n, mu, C = K.pca_moments(self.X[s:s + 77] for s in range(0, 500, 77))
        self.assertEqual(n, 500)
        np.testing.assert_allclose(mu, self.X.mean(0), atol=1e-12)
        np.testing.assert_allclose(C, np.cov(self.X.T, bias=True), atol=1e-9)

    def test_components_order_sign_and_projection(self):
        _n, mu, C = K.pca_moments([self.X])
        V, w, total = K.pca_from_cov(C, 3)
        self.assertTrue(np.all(np.diff(w) <= 0))
        np.testing.assert_allclose(V.T @ V, np.eye(3), atol=1e-10)
        for j in range(3):
            self.assertGreater(V[np.argmax(np.abs(V[:, j])), j], 0)
        Z = K.pca_project(self.X, mu, V)
        np.testing.assert_allclose(Z.var(0), w, rtol=1e-8)
        self.assertAlmostEqual(total, float(np.trace(C)))
        # sign convention is deterministic under a flipped input
        V2, _w, _t = K.pca_from_cov(C.copy(), 3)
        np.testing.assert_array_equal(V, V2)


class Context(unittest.TestCase):
    def test_stratified_counts_sorted_deterministic(self):
        y = np.repeat(np.arange(10), 50)[np.random.default_rng(3).permutation(500)]
        a = K.stratified_context(y, 20, 7)
        b = K.stratified_context(y, 20, 7)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(np.diff(a) > 0))
        self.assertEqual([int((y[a] == k).sum()) for k in range(10)], [20] * 10)
        self.assertFalse(np.array_equal(a, K.stratified_context(y, 20, 8)))

    def test_too_few_rows_refused(self):
        with self.assertRaises(ValueError):
            K.stratified_context(np.zeros(5, dtype=int), 3, 0)

    def test_smoke_context_first_per_level(self):
        y = np.repeat(np.arange(10), 50)
        ctx = K.stratified_context(y, 20, 7)
        sm = K.smoke_context(ctx, y, 5)
        self.assertEqual([int((y[sm] == k).sum()) for k in range(10)], [5] * 10)
        self.assertTrue(set(sm.tolist()) <= set(ctx.tolist()))


class Chunks(unittest.TestCase):
    def test_bounds(self):
        b = K.chunk_bounds(73169)
        self.assertEqual(len(b), 37)
        self.assertEqual(b[-1], (72000, 73169))
        self.assertEqual(b, TT.chunk_bounds(73169))

    def test_assemble_any_order_and_gaps(self):
        p = np.random.default_rng(0).random((5, 10))
        recs = [(3, 5, p[3:5]), (0, 3, p[0:3])]
        np.testing.assert_array_equal(K.assemble_chunks(recs, 5), p)
        with self.assertRaises(ValueError):
            K.assemble_chunks([(0, 3, p[0:3])], 5)
        with self.assertRaises(ValueError):
            K.assemble_chunks([(0, 3, p[0:3]), (2, 5, p[2:5])], 5)

    def test_normalise(self):
        P = np.array([[0.2, 0.2] + [0.0] * 8, [0.1] * 10], dtype=np.float32)
        Q, dev = K.normalise_rows(P)
        np.testing.assert_allclose(Q.sum(1), 1.0)
        self.assertAlmostEqual(dev, 0.6, places=6)

    def test_chunk_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = TT.chunk_path(d, 2000)
            self.assertTrue(p.endswith("chunk_00001.npz"))
            sha = {"a": "x"}
            np.savez(p, format=np.array(TT.CHUNK_FORMAT), start=np.array(2000), end=np.array(4000),
                     prob=np.zeros((2000, 10), np.float32), inputs_sha256=np.array('{"a": "x"}'))
            self.assertTrue(TT.chunk_ok(p, 2000, 4000, sha))
            self.assertFalse(TT.chunk_ok(p, 2000, 4000, {"a": "y"}))
            self.assertFalse(TT.chunk_ok(p, 0, 2000, sha))


class Mean7(unittest.TestCase):
    def test_equals_manual(self):
        rng = np.random.default_rng(2)
        lps = [np.log(rng.dirichlet(np.ones(10), size=4)) for _ in range(6)]
        pt = rng.dirichlet(np.ones(10), size=4)
        want = np.log((sum(np.exp(x) for x in lps) + pt) / 7)
        np.testing.assert_allclose(K.mean7_lp(lps, pt), want, atol=1e-14)
        with self.assertRaises(ValueError):
            K.mean7_lp(lps[:5], pt)

    def test_lse_mean_is_p09_forward(self):
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "tools")))
        import p09_forward as FW
        rng = np.random.default_rng(4)
        lps = [np.log(rng.dirichlet(np.ones(10), size=3)) for _ in range(6)]
        np.testing.assert_array_equal(K.lse_mean(lps), FW.lse_mean(lps))


class Overlap(unittest.TestCase):
    def test_overlap(self):
        full, ids = K.player_overlap(["3D:c", "2D:b"], ["3D", "2D"], ["a", "c"])
        self.assertEqual((full, ids), (0, 1))
        with self.assertRaises(ValueError):
            K.player_overlap(["1D:a"], ["3D", "2D"], ["a", "c"])
        with self.assertRaises(ValueError):
            K.player_overlap([], ["3D"], ["a"])
        full, ids = K.player_overlap(["2D:c"], ["3D", "2D"], ["a", "c"])
        self.assertEqual((full, ids), (1, 1))


class Gates(unittest.TestCase):
    def cmp(self, d=0.02, lo=0.001, mlo=0.001, b=0.01, w=0.0):
        return {"inner_diff": d, "inner_ci95": [lo, 0.03], "mono_diff": 0.01, "mono_ci95": [mlo, 0.02],
                "mono_by_colour": {"B": b, "W": w}}

    def test_pass_and_each_failure(self):
        self.assertTrue(K.gates(self.cmp())["passed"])
        self.assertFalse(K.gates(self.cmp(d=0.0099999))["passed"])
        self.assertTrue(K.gates(self.cmp(d=0.010))["G1"]["passed"])
        self.assertFalse(K.gates(self.cmp(lo=0.0))["passed"])
        self.assertFalse(K.gates(self.cmp(mlo=0.0))["passed"])
        self.assertFalse(K.gates(self.cmp(w=-1e-9))["passed"])
        self.assertTrue(K.gates(self.cmp(w=-1e-13))["passed"])


class Constants(unittest.TestCase):
    def test_tabicl_constants_agree(self):
        self.assertEqual(TT.TABICL_PARAMS, K.TABICL_PARAMS)
        self.assertEqual(TT.CKPT_SHA256, K.CKPT_SHA256)
        self.assertEqual(TT.CHUNK, K.CHUNK)
        self.assertEqual((TT.TABICL_VERSION, TT.TORCH_VERSION), (K.TABICL_VERSION, K.TORCH_VERSION))
        self.assertEqual(TT.TABICL_PARAMS["checkpoint_version"], K.CKPT_NAME)
        self.assertEqual((K.PCA_DIM, K.CONTEXT_PER_LEVEL, K.CONTEXT_SEED), (128, 3000, 20261017))

    def test_tabicl_module_imports_no_heavy_packages(self):
        src = open(os.path.join(HERE, "p17_tabicl.py"), encoding="utf-8").read()
        top = src.split("def ")[0]
        for mod in ("import numpy", "import torch", "tabicl import"):
            self.assertNotIn(mod, top)


if __name__ == "__main__":
    unittest.main()
