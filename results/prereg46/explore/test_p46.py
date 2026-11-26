"""prereg46 explore: small CPU tests of p46_eval.py (synthetic data only; no score of a prereg46 variant).

    PY -m pytest -q results/prereg46/explore/test_p46.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p46_eval as PE  # noqa: E402

F10, st = PE.F10, PE.st


def write_chunk(path, ref, means, counts, failed, set_name="p39X"):
    np.savez(path, format=np.array(F10.CHUNK_FORMAT), set=np.array(set_name),
             sides=np.array(["target", "opponent"]), ref=np.asarray(ref, np.int64),
             means=np.asarray(means, np.float32), counts=np.asarray(counts, np.int32),
             failed=np.asarray(failed, bool))


def brute(Z, cg, qg, agg):
    """Reference T_O with pw none: per-game L2, pair cosines, then max / mean-of-max."""
    def l2(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v * 0
    nq, nc = len(qg), len(cg)
    T = np.zeros((nq, nc))
    hasq = np.array([len(g) > 0 for g in qg])
    hasc = np.array([len(g) > 0 for g in cg])
    for i, q in enumerate(qg):
        if not q:
            continue
        for j, c in enumerate(cg):
            if not c:
                continue
            S = np.array([[l2(Z[a]) @ l2(Z[b]) for b in c] for a in q])
            T[i, j] = S.max() if agg == "max" else S.max(1).mean()
    for i in range(nq):
        if hasq[i] and not hasc.all():
            T[i, ~hasc] = T[i, hasc].min()
    return T, hasq


def synth_pool(seed=0, nc=12, nq=9, dim=16):
    rng = np.random.default_rng(seed)
    cg, qg, n = [], [], 0
    for j in range(nc):
        k = int(rng.integers(0, 5)) if j != 0 else 0      # candidate 0: no game
        cg.append(list(range(n, n + k)))
        n += k
    for i in range(nq):
        k = int(rng.integers(0, 4)) if i != 1 else 0      # question 1: no game
        qg.append(list(range(n, n + k)))
        n += k
    Z = rng.normal(0, 1, (n, dim))
    return Z, cg, qg


class Loader(unittest.TestCase):
    def test_side_and_usable(self):
        with tempfile.TemporaryDirectory() as d:
            n = 6
            means = np.zeros((3, 2, F10.P1_DIM), np.float32)
            means[:, 0, 0] = [1, 2, 3]
            means[:, 1, 0] = [10, 20, 30]
            write_chunk(os.path.join(d, "chunk_00000.npz"), [0, 2, 4], means, [[5, 4], [5, 0], [5, 3]],
                        [False, False, True])
            means2 = np.zeros((3, 2, F10.P1_DIM), np.float32)
            means2[:, 1, 0] = [40, 50, 60]
            write_chunk(os.path.join(d, "chunk_00001.npz"), [1, 3, 5], means2, [[1, 1], [0, 2], [1, 1]],
                        [False, False, False])
            paths = F10.chunk_files(d)
            X, u, f, have, cnt = PE.load_side(paths, n, "p39X", 1)
            np.testing.assert_array_equal(X[:, 0], [10, 40, 20, 50, 30, 60])
            np.testing.assert_array_equal(u, [True, True, False, True, False, True])   # counts 0 / failed out
            self.assertTrue(have.all())
            X0, u0, _f, _h, _c = PE.load_side(paths, n, "p39X", 0)
            Xr, ur, _f, _h, _c = F10.load_side0(paths, n, "p39X")
            np.testing.assert_array_equal(X0, Xr)
            np.testing.assert_array_equal(u0, ur)
            with self.assertRaises(SystemExit):
                PE.load_side(paths, n, "p39Y", 1)


class Term(unittest.TestCase):
    def test_none_equals_brute(self):
        for seed in range(4):
            Z, cg, qg = synth_pool(seed)
            usable = np.ones(len(Z), bool)
            got = PE.opp_z(Z, usable, cg, qg, "none")
            for agg in PE.AGG_GRID:
                Vc, gc, Vq, gq = PE.opp_vectors(Z, cg, qg, "none")
                T, h = PE.agg_terms(Vc, gc, Vq, gq, len(cg), len(qg), (agg,))[agg]
                Tr, hr = brute(Z, cg, qg, agg)
                np.testing.assert_array_equal(h, hr)
                np.testing.assert_allclose(T[h], Tr[h], atol=1e-12)
                z, hz = got[agg]
                np.testing.assert_allclose(z, st.zscore_rows(Tr, hr), atol=1e-9)
                self.assertTrue((z[~hz] == 0).all())

    def test_usable_mask(self):
        Z, cg, qg = synth_pool(1)
        usable = np.ones(len(Z), bool)
        usable[::3] = False
        got = PE.opp_z(Z, usable, cg, qg, "none")
        cu = [[g for g in grp if usable[g]] for grp in cg]
        qu = [[g for g in grp if usable[g]] for grp in qg]
        Tr, hr = brute(Z, cu, qu, "mmax")
        np.testing.assert_allclose(got["mmax"][0], st.zscore_rows(Tr, hr), atol=1e-9)

    def test_b4_runs_and_uses_candidate_classes(self):
        rng = np.random.default_rng(3)
        nc, per, dim = 40, 6, 64
        cg = [list(range(j * per, (j + 1) * per)) for j in range(nc)]
        qg = [[nc * per + i, nc * per + nc + i] for i in range(nc)]
        cent = rng.normal(0, 1, (nc, dim))
        Z = np.concatenate([cent[np.repeat(np.arange(nc), per)] + rng.normal(0, 1, (nc * per, dim)),
                            cent + rng.normal(0, 1, (nc, dim)), cent + rng.normal(0, 1, (nc, dim))])
        r = PE.opp_z(Z, np.ones(len(Z), bool), cg, qg, "b4")
        for agg in PE.AGG_GRID:
            self.assertTrue(np.isfinite(r[agg][0]).all())
            self.assertEqual(r[agg][0].shape, (nc, nc))
        Vc, _gc, Vq, _gq = PE.opp_vectors(Z, cg, qg, "b4")
        self.assertEqual(Vc.shape[1], PE.P39E.B4_P1["whiten"]["lda"])
        np.testing.assert_allclose(np.linalg.norm(Vc, axis=1), 1.0)
        # the whitening sees only candidate games: changing the question games does not change Vc
        Z2 = Z.copy()
        Z2[nc * per:] += 5.0
        Vc2, _a, _b, _c = PE.opp_vectors(Z2, cg, qg, "b4")
        np.testing.assert_array_equal(Vc, Vc2)

    def test_too_few_candidates(self):
        Z = np.ones((3, 4))
        r = PE.opp_z(Z, np.ones(3, bool), [[0], [], []], [[1, 2]], "none")
        self.assertTrue((r["max"][0] == 0).all())
        self.assertFalse(r["max"][1].any())


class Fusion(unittest.TestCase):
    def test_wo_zero_is_base_bit_for_bit(self):
        rng = np.random.default_rng(0)
        pools, FPs, zOs, base = [], [], [], []
        for s in range(3):
            nq, nc = 30, 40
            FP = rng.normal(0, 1, (nq, nc))
            owner = rng.integers(0, nc, nq)
            pools.append({"owner": owner})
            FPs.append(FP)
            zOs.append(rng.normal(0, 1, (nq, nc)))
            base.append(float(st.score(st.csls(FP, 20), owner)))
        self.assertEqual(PE.m46_scores(FPs, zOs, 0.0, 20, pools), base)

    def test_grid_order(self):
        rng = np.random.default_rng(1)
        pools = [{"owner": rng.integers(0, 10, 8)}]
        FPs = [rng.normal(0, 1, (8, 10))]
        zO_by = {(pw, agg): [rng.normal(0, 1, (8, 10))] for pw in PE.PW_GRID for agg in PE.AGG_GRID}
        rows = PE.grid_rows(FPs, zO_by, pools, list(PE.PW_GRID))
        self.assertEqual(len(rows), 2 * 2 * 6 * 3)
        self.assertEqual([(r["pw"], r["agg"], r["wO"], r["k"]) for r in rows[:4]],
                         [("b4", "max", 0.25, 10), ("b4", "max", 0.25, 20), ("b4", "max", 0.25, 40),
                          ("b4", "max", 0.5, 10)])
        self.assertEqual(rows[-1]["pw"], "none")
        self.assertEqual(PE.WO_GRID, (0.25, 0.5, 1, 2, 4, 8))
        self.assertEqual(PE.K_GRID, (10, 20, 40))
        rows_none = PE.grid_rows(FPs, zO_by, pools, ["none"])
        self.assertEqual(len(rows_none), 36)


class Rules(unittest.TestCase):
    def test_no_e_results_yet(self):
        with tempfile.TemporaryDirectory() as d:
            PE.no_e_results_yet(d, os.path.join(d, "apply"))
            open(os.path.join(d, "eval.json"), "w").close()
            with self.assertRaises(SystemExit):
                PE.no_e_results_yet(d, os.path.join(d, "apply"))
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "apply"))
            with self.assertRaises(SystemExit):
                PE.no_e_results_yet(d, os.path.join(d, "apply"))

    def test_judge_ready(self):
        ok = {"gate": {"pass": True}}
        prior = ["prereg36", "prereg37", "prereg38", "prereg39", "prereg43"]
        PE.judge_ready(ok, 6, prior)
        with self.assertRaises(SystemExit):
            PE.judge_ready(ok, 5, prior)
        with self.assertRaises(SystemExit):
            PE.judge_ready({"gate": {"pass": False}}, 6, prior)
        with self.assertRaises(SystemExit):
            PE.judge_ready(ok, 7, prior + ["prereg46"])

    def test_e_judgements_counted(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("prereg36", "prereg43", "prereg45"):
                os.makedirs(os.path.join(d, "results", name, "explore"))
            open(os.path.join(d, "results", "prereg36", "explore", "eval.json"), "w").close()
            open(os.path.join(d, "results", "prereg43", "explore", "eval.json"), "w").close()
            self.assertEqual(PE.e_judgements_so_far(d), ["prereg36", "prereg43"])

    def test_constants(self):
        self.assertEqual(PE.D_V26, 0.7467766213119217)
        self.assertEqual(PE.E_V26, 0.681614485210018)
        self.assertEqual(PE.GATE, 0.01)
        self.assertEqual(PE.SEAM_K, 20)
        self.assertEqual(PE.OPP_SIDE, 1)
        self.assertEqual(PE.E43.BOOT_SEED, 43)


if __name__ == "__main__":
    unittest.main()
