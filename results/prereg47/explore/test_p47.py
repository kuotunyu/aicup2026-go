"""prereg47 explore: small CPU tests of p47_eval.py (synthetic data only; no score of a prereg47 variant).

    PY -m pytest -q results/prereg47/explore/test_p47.py
"""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p47_eval as PE  # noqa: E402

F10, st = PE.F10, PE.st


def brute_features(turn, logp, ent):
    """Reference 52 values of one game from its target-side positions (plain numpy, one segment at a time)."""
    turn, logp, ent = np.asarray(turn), np.asarray(logp, np.float64), np.asarray(ent, np.float64)
    segs = [turn < 60, (turn >= 60) & (turn < 150), turn >= 150, np.ones(len(turn), bool)]
    out, flags = [], []
    for m in segs:
        if not m.any():
            out += [0.0] * 12
            flags.append(1.0)
            continue
        row = []
        for v in (logp[m], ent[m]):
            row += [v.mean(), v.std()] + list(np.percentile(v, [10, 50, 90]))
        row += [float((logp[m] < -4).mean()), m.sum() / len(turn)]
        out += row
        flags.append(0.0)
    return np.array(out + flags)


def synth_positions(rng, n_games, lens):
    g, side, turn, lp, en = [], [], [], [], []
    for i in range(n_games):
        n = lens[i]
        t = np.arange(n)
        s = (t + i) % 2
        g.append(np.full(n, i))
        side.append(s)
        turn.append(t)
        lp.append(-rng.exponential(2.0, n))
        en.append(rng.uniform(0, 5, n))
    return [np.concatenate(x) for x in (g, side, turn, lp, en)]


def write_chunk(path, ref, failed, pos=None, counts=None, set_name="p39X"):
    kw = {}
    if pos is not None:
        g, side, turn, lp, en = pos
        kw = {"pos_game": np.asarray(g, np.int32), "pos_side": np.asarray(side, np.int8),
              "pos_turn": np.asarray(turn, np.int16), "pos_logp": np.asarray(lp, np.float32),
              "pos_entropy": np.asarray(en, np.float32)}
    n = len(ref)
    np.savez(path, format=np.array(F10.CHUNK_FORMAT), set=np.array(set_name),
             sides=np.array(["target", "opponent"]), ref=np.asarray(ref, np.int64),
             means=np.zeros((n, 2, 4), np.float32), counts=np.asarray(counts, np.int32),
             failed=np.asarray(failed, bool), **kw)


class Features(unittest.TestCase):
    def test_chunk_features_equal_brute(self):
        rng = np.random.default_rng(0)
        lens = [300, 40, 120, 1, 0, 155, 61]          # games ending in every segment, one with no move
        g, side, turn, lp, en = synth_positions(rng, len(lens), lens)
        sel = side == 0
        F, nt = PE.chunk_features(g[sel], turn[sel], lp[sel].astype(np.float32), en[sel].astype(np.float32),
                                  len(lens))
        self.assertEqual(F.shape, (len(lens), 52))
        for i in range(len(lens)):
            m = sel & (g == i)
            self.assertEqual(nt[i], m.sum())
            if not m.any():
                self.assertTrue((F[i, :48] == 0).all())
                continue
            want = brute_features(turn[m], lp[m].astype(np.float32), en[m].astype(np.float32))
            np.testing.assert_allclose(F[i], want, rtol=1e-12, atol=1e-12)
        # game 1 (40 moves): only the opening -> mid / end flagged; game 0 (300): no flag
        np.testing.assert_array_equal(F[1, 48:], [0, 1, 1, 0])
        np.testing.assert_array_equal(F[0, 48:], [0, 0, 0, 0])
        self.assertEqual(F[0, 3 * 12 + 11], 1.0)          # "all" share

    def test_feature_names(self):
        self.assertEqual(len(PE.feature_names()), PE.N_FEAT)
        self.assertEqual(PE.N_FEAT, 52)


class Loader(unittest.TestCase):
    def test_usable_rules(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            lens = [50, 19, 30, 200]                       # game 1: 9 or 10 target moves
            pos = synth_positions(rng, 4, lens)
            g, side = pos[0], pos[1]
            counts = np.stack([np.bincount(g[side == 0], minlength=4), np.bincount(g[side == 1], minlength=4)], 1)
            write_chunk(os.path.join(d, "chunk_00000.npz"), [0, 2, 4, 6], [False, False, True, False], pos, counts)
            write_chunk(os.path.join(d, "chunk_00001.npz"), [1, 3, 5, 7], [False] * 4, None, np.ones((4, 2)))
            r = PE.load_moves(F10.chunk_files(d), 8, "p39X")
            self.assertTrue(r["have"].all())
            np.testing.assert_array_equal(r["no_pos"], [False, True] * 4)
            want_u = [True, False, counts[1, 0] >= 10, False, False, False, True, False]
            np.testing.assert_array_equal(r["usable"], want_u)
            self.assertTrue((r["F"][~r["usable"]] == 0).all())
            with self.assertRaises(SystemExit):
                PE.load_moves(F10.chunk_files(d), 8, "p39Y")

    def test_counts_must_match(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as d:
            pos = synth_positions(rng, 2, [30, 30])
            write_chunk(os.path.join(d, "chunk_00000.npz"), [0, 1], [False, False], pos, [[14, 15], [15, 15]])
            with self.assertRaises(SystemExit):
                PE.load_moves(F10.chunk_files(d), 2, "p39X")

    def test_need_skips_games(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as d:
            for c in range(2):
                pos = synth_positions(rng, 2, [40, 40])
                write_chunk(os.path.join(d, "chunk_%05d.npz" % c), [2 * c, 2 * c + 1], [False, False], pos,
                            [[20, 20], [20, 20]])
            need = np.array([False, False, True, False])
            r = PE.load_moves(F10.chunk_files(d), 4, "p39X", need)
            self.assertTrue(r["have"].all())
            np.testing.assert_array_equal(r["usable"], [False, False, True, True])


class Term(unittest.TestCase):
    def synth(self, seed, nc=12, nq=9, dim=32):
        rng = np.random.default_rng(seed)
        cg, qg, n = [], [], 0
        for j in range(nc):
            k = int(rng.integers(0, 5)) if j != 0 else 0
            cg.append(list(range(n, n + k)))
            n += k
        for i in range(nq):
            k = int(rng.integers(0, 4)) if i != 1 else 0
            qg.append(list(range(n, n + k)))
            n += k
        return rng.normal(0, 1, (n, dim)), cg, qg

    def test_none_is_neg_sq_distance_of_means(self):
        for seed in range(3):
            Z, cg, qg = self.synth(seed)
            usable = np.ones(len(Z), bool)
            usable[::4] = False
            z, has = PE.move_z(Z, usable, cg, qg, "none")
            cu = [[g for g in grp if usable[g]] for grp in cg]
            qu = [[g for g in grp if usable[g]] for grp in qg]
            hc = np.array([len(g) > 0 for g in cu])
            hq = np.array([len(g) > 0 for g in qu])
            T = np.zeros((len(qu), len(cu)))
            for i, q in enumerate(qu):
                if not q:
                    continue
                mq = Z[q].mean(0)
                for j, c in enumerate(cu):
                    if c:
                        T[i, j] = -np.sum((mq - Z[c].mean(0)) ** 2)
                if not hc.all():
                    T[i, ~hc] = T[i, hc].min()
            np.testing.assert_array_equal(has, hq)
            np.testing.assert_allclose(z, st.zscore_rows(T, hq), atol=1e-9)
            self.assertTrue((z[~has] == 0).all())

    def test_b4_runs_on_16_dims(self):
        rng = np.random.default_rng(4)
        nc, per = 40, 6
        cg = [list(range(j * per, (j + 1) * per)) for j in range(nc)]
        qg = [[nc * per + i] for i in range(nc)]
        cent = rng.normal(0, 1, (nc, 16))
        Z = np.concatenate([cent[np.repeat(np.arange(nc), per)] + rng.normal(0, 1, (nc * per, 16)),
                            cent + rng.normal(0, 1, (nc, 16))])
        z, has = PE.move_z(Z, np.ones(len(Z), bool), cg, qg, "b4")
        self.assertEqual(z.shape, (nc, nc))
        self.assertTrue(has.all() and np.isfinite(z).all())

    def test_lda_dims_are_prefixes(self):
        rng = np.random.default_rng(5)
        n_pl, per = 60, 5
        lab = np.repeat(np.arange(n_pl), per).astype(str)
        X = rng.normal(0, 1, (n_pl, 52))[np.repeat(np.arange(n_pl), per)] + rng.normal(0, 1, (n_pl * per, 52))
        X[:, -1] = 0.0                                     # a constant column (the "all" flag) is harmless
        model = F10.fit_p1_lda(X, lab, np.ones(len(X), bool))
        self.assertGreaterEqual(model["W"].shape[1], 32)
        Z16, Z32 = PE.project(X, model, 16), PE.project(X, model, 32)
        np.testing.assert_array_equal(Z16, Z32[:, :16])
        self.assertTrue(np.isfinite(Z32).all())
        with tempfile.TemporaryDirectory() as d:
            rec = PE.save_lda(model, os.path.join(d, "lda.npz"))
            rec["file"] = os.path.join(d, "lda.npz")
            m2 = PE.load_lda(rec)
            np.testing.assert_array_equal(PE.project(X, m2, 32), Z32)
            rec["arrays_sha256"] = "0" * 64
            with self.assertRaises(SystemExit):
                PE.load_lda(rec)


class Fusion(unittest.TestCase):
    def test_wm_zero_is_base_bit_for_bit(self):
        rng = np.random.default_rng(0)
        pools, FPs, zs, base = [], [], [], []
        for _s in range(3):
            FP = rng.normal(0, 1, (30, 40))
            owner = rng.integers(0, 40, 30)
            pools.append({"owner": owner})
            FPs.append(FP)
            zs.append(rng.normal(0, 1, (30, 40)))
            base.append(float(st.score(st.csls(FP, 20), owner)))
        self.assertEqual(PE.m47_scores(FPs, zs, 0.0, 20, pools), base)

    def test_grid_order(self):
        rng = np.random.default_rng(1)
        pools = [{"owner": rng.integers(0, 10, 8)}]
        FPs = [rng.normal(0, 1, (8, 10))]
        terms = [(d, pw) for d in PE.DIM_GRID for pw in PE.PW_GRID]
        zb = {t: [rng.normal(0, 1, (8, 10))] for t in terms}
        rows = PE.grid_rows(FPs, zb, pools, terms)
        self.assertEqual(len(rows), 2 * 2 * 6 * 3)
        self.assertEqual([(r["dim"], r["pw"], r["wM"], r["k"]) for r in rows[:4]],
                         [(16, "b4", 0.25, 10), (16, "b4", 0.25, 20), (16, "b4", 0.25, 40), (16, "b4", 0.5, 10)])
        self.assertEqual((rows[-1]["dim"], rows[-1]["pw"]), (32, "none"))
        rows2 = PE.grid_rows(FPs, zb, pools, [(16, "none"), (32, "none")])
        self.assertEqual(len(rows2), 36)
        self.assertEqual(PE.WM_GRID, (0.25, 0.5, 1, 2, 4, 8))
        self.assertEqual(PE.K_GRID, (10, 20, 40))


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
            PE.judge_ready(ok, 7, prior + ["prereg47"])

    def test_e_judgements_counted(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("prereg36", "prereg43", "prereg46", "prereg47"):
                os.makedirs(os.path.join(d, "results", name, "explore"))
            open(os.path.join(d, "results", "prereg36", "explore", "eval.json"), "w").close()
            open(os.path.join(d, "results", "prereg47", "explore", "eval.json"), "w").close()
            self.assertEqual(PE.e_judgements_so_far(d), ["prereg36", "prereg47"])

    def test_constants(self):
        self.assertEqual(PE.D_V26, 0.7467766213119217)
        self.assertEqual(PE.E_V26, 0.681614485210018)
        self.assertEqual(PE.GATE, 0.01)
        self.assertEqual(PE.SEAM_K, 20)
        self.assertEqual(PE.SEG_EDGES, (60, 150))
        self.assertEqual(PE.MIN_MOVES, 10)
        self.assertEqual(PE.BLUNDER, -4.0)
        self.assertEqual(PE.S_SETS, ("L", "M1"))
        self.assertEqual(PE.E43.BOOT_SEED, 43)
        self.assertEqual(F10.MIN_GAMES, 2)


if __name__ == "__main__":
    unittest.main()
