"""Small tests of prereg63's own code (no data, no scores):
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg63/explore/test_p63.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p63_eval as M  # noqa: E402


def per_colour(rng, nq, nc, ch_b, ch_w):
    return {"B": (rng.normal(size=(nq, nc)), np.asarray(ch_b, bool)),
            "W": (rng.normal(size=(nq, nc)), np.asarray(ch_w, bool))}


class ColourFill(unittest.TestCase):
    def test_same_as_frozen_when_every_candidate_has_both_colours(self):
        rng = np.random.default_rng(0)
        pc = per_colour(rng, 7, 9, [True] * 9, [True] * 9)
        qc = {"B": [1, 0, 2, 3, 1, 0, 5], "W": [0, 1, 1, 2, 0, 0, 0]}
        a, ha = M.st.colour_terms(pc, qc)
        b, hb = M.colour_terms_fixed(pc, qc)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(ha, hb))

    def test_missing_colour_is_neutral_and_z_over_present(self):
        rng = np.random.default_rng(1)
        ch_b = [True, True, False, True, True, False]
        pc = per_colour(rng, 4, 6, ch_b, [True] * 6)
        qc = {"B": [2, 2, 2, 2], "W": [0, 0, 0, 0]}
        out, has = M.colour_terms_fixed(pc, qc)
        self.assertTrue(has.all())
        self.assertTrue(np.all(out[:, [2, 5]] == 0.0))
        present = out[:, [0, 1, 3, 4]]
        self.assertTrue(np.allclose(present.mean(1), 0.0))
        self.assertTrue(np.allclose(present.std(1), 1.0))
        old, _ = M.st.colour_terms(pc, qc)
        self.assertTrue(np.all(old[:, [2, 5]] < 0))          # the frozen fill puts them at the row minimum

    def test_no_candidate_with_the_colour_gives_zero(self):
        rng = np.random.default_rng(2)
        pc = per_colour(rng, 3, 5, [False] * 5, [True] * 5)
        qc = {"B": [1, 1, 1], "W": [0, 0, 0]}
        out, _ = M.colour_terms_fixed(pc, qc)
        self.assertTrue(np.all(out == 0.0))

    def test_injection_is_scoped(self):
        before = M.sim.st.colour_terms
        with M.fixed_fill():
            self.assertIs(M.sim.st.colour_terms, M.colour_terms_fixed)
        self.assertIs(M.sim.st.colour_terms, before)


class Ported(unittest.TestCase):
    def test_kernel(self):
        K = M.rank_kernel()
        self.assertEqual(K.shape, (10, 10))
        self.assertAlmostEqual(K[0, 1], np.exp(-1 / (2 * 0.85 ** 2)))
        self.assertTrue(np.allclose(K, K.T))

    def test_frozen_constants_match_frozen_json(self):
        fz = M.read_json(M.FROZEN)
        self.assertEqual(fz["U1_combined"]["w_p1_multi"], M.W_MULTI)
        self.assertEqual(fz["U1_combined"]["w_rank"], M.W_RANK)
        self.assertEqual(fz["C_rank_compat"]["kernel"]["sigma"], M.SIGMA_RK)
        self.assertEqual(fz["B_p1_multi"]["pool_term"]["whitening_lam"], M.B4_LAM)
        self.assertEqual(fz["B_p1_multi"]["combine"]["weights"], M.MULTI_W)
        f = fz["B_p1_multi"]["fisher_lda"]
        self.assertEqual(tuple((n, f[n]["cols"][0], f[n]["cols"][1], f[n]["out_dim"]) for n in ("full", "move", "board")),
                         M.LDA_SPECS)
        self.assertEqual(fz["seam_targets_D"]["U1_combined"], M.D_U1_GEMINI)

    def test_b4_proto_term_shapes_and_min_fill(self):
        rng = np.random.default_rng(3)
        Z = rng.normal(size=(40, 6))
        usable = np.ones(40, bool)
        usable[[8, 9]] = False
        cg = [list(range(i * 2, i * 2 + 2)) for i in range(5)]   # candidate 4 = rows 8, 9: no usable game
        qg = [[10, 11], [12], [13, 14, 15]]
        z = M.b4_proto_term(Z, usable, cg, qg)
        self.assertEqual(z.shape, (3, 5))
        self.assertTrue(np.allclose(z.mean(1), 0.0))

    def test_lacking_colour(self):
        colour = np.array(["B", "W", "B", "B", "W", "W"])
        pools = [{"q_rows": [[1], [2], [4, 0]], "owner": np.array([0, 1, 1]), "cand_rows": [[0, 2], [4, 5]]}]
        self.assertEqual(M.lacking_colour(pools, colour)[0].tolist(), [True, True, True])
        pools[0]["owner"] = np.array([1, 0, 0])
        self.assertEqual(M.lacking_colour(pools, colour)[0].tolist(), [False, False, True])


if __name__ == "__main__":
    unittest.main()
