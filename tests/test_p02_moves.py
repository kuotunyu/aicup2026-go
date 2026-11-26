"""Tests for tools/p02_moves.py (shared PREREG_02 move preprocessing).
Run: python -m unittest discover -s tests   (or python -m pytest tests)

The symmetry convention is checked against the notebook's d4 (flip dims=(3,), then
rot90 k = s % 4 on dims (2, 3)) with numpy, with torch when it is installed, and against
the tutorial's own board layout (utils.py) when sgfmill is installed.
"""

import os
import random
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import p02_moves as mv  # noqa: E402


def random_game(rng, n=60):
    pts = rng.sample([mv.from_xy(x, y) for x in range(19) for y in range(19)], n)
    return [("B" if i % 2 == 0 else "W", p) for i, p in enumerate(pts)]


def notebook_d4(plane, s):
    """numpy version of the notebook's d4 on one [19, 19] plane (dims 2, 3 -> axes 0, 1)."""
    y = np.flip(plane, axis=1) if s >= 4 else plane
    return np.rot90(y, k=s % 4, axes=(0, 1))


class Parse(unittest.TestCase):
    def test_regex_passes_and_properties(self):
        sgf = "(;SZ[19];B[qd];W[dp];B[];W[tt];B[pq]C[note];W[cd]MN[3];B[aa])"
        self.assertEqual(mv.parse_moves(sgf),
                         [("B", "qd"), ("W", "dp"), ("B", "pq"), ("W", "cd"), ("B", "aa")])

    def test_limit_60(self):
        body = "".join(";%s[%s]" % (("B", "W")[i % 2], mv.from_xy(i % 19, (i // 19) % 19))
                       for i in range(100))
        got = mv.parse_moves("(;SZ[19]" + body + ")")
        self.assertEqual(len(got), 60)
        self.assertEqual(got[59], ("W", mv.from_xy(59 % 19, 59 // 19)))
        self.assertEqual(len(mv.parse_moves("(;SZ[19]" + body + ")", limit=12)), 12)

    def test_regex_is_the_registered_one(self):
        self.assertEqual(mv.MOVE_RE.pattern, r";([BW])\[([a-s]{2})\]")
        self.assertEqual(mv.MAX_MOVES, 60)


class Symmetries(unittest.TestCase):
    def test_eight_distinct_maps_identity_first(self):
        tables = [tuple(sorted(t.items())) for t in mv.SYM_TABLE]
        self.assertEqual(len(set(tables)), 8)
        self.assertTrue(all(k == v for k, v in mv.SYM_TABLE[0].items()))
        for t in mv.SYM_TABLE:  # each map is a permutation of the 361 points
            self.assertEqual(sorted(t.values()), sorted(t.keys()))

    def test_group_closure_and_inverses(self):
        for s1 in range(8):
            for s2 in range(8):
                s3 = mv.compose(s1, s2)  # raises if the composition is not one of the 8
                g = random_game(random.Random(s1 * 8 + s2), 20)
                self.assertEqual(mv.transform(mv.transform(g, s1), s2), mv.transform(g, s3))
            self.assertEqual(mv.compose(s1, mv.inverse(s1)), 0)
            self.assertEqual(mv.compose(0, s1), s1)

    def test_same_convention_as_notebook_d4_numpy(self):
        for x in range(19):
            for y in range(19):
                plane = np.zeros((19, 19), dtype=np.int8)
                plane[mv.to_plane(x, y)] = 1
                for s in range(8):
                    out = notebook_d4(plane, s)
                    got = tuple(int(v[0]) for v in np.nonzero(out))
                    self.assertEqual(got, mv.to_plane(*mv.sym_xy(x, y, s)), (x, y, s))

    def test_same_convention_as_notebook_d4_torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        g = random_game(random.Random(7), 30)
        x = torch.zeros((1, 2, 19, 19))
        for c, p in g:
            x[0, 0 if c == "B" else 1][mv.to_plane(*mv.to_xy(p))] = 1
        for s in range(8):
            y = torch.flip(x, dims=(3,)) if s >= 4 else x
            y = torch.rot90(y, k=s % 4, dims=(2, 3))
            want = torch.zeros_like(x)
            for c, p in mv.transform(g, s):
                want[0, 0 if c == "B" else 1][mv.to_plane(*mv.to_xy(p))] = 1
            self.assertTrue(torch.equal(y, want), s)

    def test_plane_layout_matches_tutorial_utils(self):
        try:
            from sgfmill import boards, sgf  # noqa: F401
        except ImportError:
            self.skipTest("sgfmill not installed")
        sys.path.insert(0, os.path.join(REPO, "third_party", "aicup2026_tutorial"))
        try:
            from utils import SGFParseRankPrediction
        except ImportError as e:
            self.skipTest(f"tutorial utils not importable: {e}")
        game = "(;SZ[19];B[pd];W[dp];B[qq];W[ab];B[sa])"
        parser = SGFParseRankPrediction(history_length=6)
        feats, err = parser.features_from_sgf_content(game)
        self.assertIsNone(err)
        last_black, last_white = feats[-2], feats[-1]  # planes after the last move
        want_b = np.zeros((19, 19), dtype=np.uint8)
        want_w = np.zeros((19, 19), dtype=np.uint8)
        for c, p in mv.parse_moves(game):
            (want_b if c == "B" else want_w)[mv.to_plane(*mv.to_xy(p))] = 1
        np.testing.assert_array_equal(last_black, want_b)
        np.testing.assert_array_equal(last_white, want_w)


class Canon(unittest.TestCase):
    def test_first_m_invariant_under_all_symmetries(self):
        rng = random.Random(3)
        for _ in range(200):
            g = random_game(rng, 60)
            for M in (1, 6, 10, 12):
                base = mv.canon(g, M)
                for s in range(8):
                    other = mv.canon(mv.transform(g, s), M)
                    self.assertEqual(other[:M], base[:M])
                    # random points: the minimiser is unique, so the whole game agrees too
                    if M >= 2:
                        self.assertEqual(other, base)

    def test_is_the_minimum_and_applied_to_whole_game(self):
        rng = random.Random(4)
        for _ in range(100):
            g = random_game(rng, 40)
            s = mv.canon_sym(g, 12)
            c = mv.canon(g, 12)
            self.assertEqual(c, mv.transform(g, s))
            self.assertEqual(tuple(p for _, p in c[:12]),
                             min(tuple(p for _, p in mv.transform(g, t)[:12]) for t in range(8)))
            self.assertEqual([col for col, _ in c], [col for col, _ in g])

    def test_ties_go_to_smallest_index(self):
        g = [("B", "jj"), ("W", "dd"), ("B", "pp")]   # tengen is fixed by every symmetry
        self.assertEqual(mv.canon_sym(g, 1), 0)
        self.assertEqual(mv.canon(g, 1), g)
        # 'dd' is on a diagonal: only s with the smallest image of 'dd' tie; the smallest index wins
        g2 = [("B", "dd"), ("W", "qc")]
        imgs = [mv.sym_point("dd", s) for s in range(8)]
        best = min(imgs)
        self.assertEqual(mv.canon_sym(g2, 1), imgs.index(best))

    def test_matches_explore_canonical_tuple(self):
        """results/prereg02/explore/opening_rank_ngram.py: canon = min over D4 of the first M points."""
        def tf(p, s):
            x, y = p
            rot, flip = s
            if flip:
                x = 18 - x
            for _ in range(rot):
                x, y = y, 18 - x
            return (x, y)
        syms = [(r, f) for r in range(4) for f in (False, True)]
        rng = random.Random(5)
        for _ in range(300):
            g = random_game(rng, 20)
            M = rng.choice((1, 3, 12))
            pts = [mv.to_xy(p) for _, p in g[:M]]
            explore = min(tuple(tf(p, s) for p in pts) for s in syms)
            ours = tuple(mv.to_xy(p) for _, p in mv.canon(g, M)[:M])
            self.assertEqual(ours, explore)


if __name__ == "__main__":
    unittest.main()
