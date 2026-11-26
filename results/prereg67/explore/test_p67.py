"""Small tests of p67_eval.py (no pool, no score). Run: .venv/Scripts/python.exe -m pytest -q results/prereg67/explore/test_p67.py"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p67_eval as M  # noqa: E402


class Ensemble(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.Z = [r.normal(size=(30, 50)) for _ in range(7)]

    def test_single_member_is_identity(self):
        self.assertTrue(np.array_equal(M.ensemble([self.Z[0]], [1.0]), self.Z[0]))

    def test_zero_weights_skipped(self):
        got = M.ensemble(self.Z[:3], [1.0, 0.0, 0.0])
        self.assertTrue(np.array_equal(got, self.Z[0]))

    def test_weighted_mean(self):
        w = [1, 1, 1, 2, 2, 2, 2]
        want = sum(wi * z for wi, z in zip(w, self.Z)) / sum(w)
        self.assertTrue(np.allclose(M.ensemble(self.Z, w), want, rtol=0, atol=1e-12))

    def test_c49_only_matches_fuse(self):
        """F = G + E_ens * 8 equals fuse_cfg's st.fuse(T + [E], w + [8]) when E_ens = 1.0 * Z (NOTES.md 4)."""
        r = np.random.default_rng(1)
        T = [(r.normal(size=(30, 50)), np.ones(30, bool)) for _ in range(3)]
        E = (r.normal(size=(30, 50)), np.ones(30, bool))
        w = [1.0, 0.5, 0.0]
        want = M.st.fuse(T + [E], w + [8])
        G = M.st.fuse(T, w)
        got = G + M.ensemble([M.st.zscore_rows(*E)], [1.0]) * 8.0
        self.assertTrue(np.array_equal(got, want))

    def test_cfg_ens(self):
        c = M.cfg_ens({"gnorm": True, "pool_w": "mean", "centre_colour": False, "whiten": {"lam": 0.7, "lda": None},
                       "sim": "proto", "wE": 8})
        self.assertEqual(c["whiten"], {"lam": 0.85, "lda": None})
        self.assertTrue(c["centre_colour"] and c["gnorm"])
        self.assertEqual(M.WE_ENS, 14.0)
        self.assertEqual([M.WEIGHTS[m] for m in M.ORDER], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])


class Version(unittest.TestCase):
    def test_none(self):
        m, p = M.pick_version({}, {}, {})
        self.assertEqual((m, p), (["c49", "S", "B", "S1", "S2"], []))
        self.assertEqual(M.version_name(m), "B")

    def test_seven(self):
        e = {"p61": {"pass": True}, "X": {"pass": True}}
        m, _ = M.pick_version(e, {"p61": True, "X": True}, {"p61": True, "X": True})
        self.assertEqual(m, ["c49", "S", "B", "p61", "S1", "S2", "X"])
        self.assertEqual(M.version_name(m), "7")

    def test_x_missing(self):
        m, p = M.pick_version({"p61": {"pass": True}}, {"p61": True}, {"p61": True})
        self.assertEqual(M.version_name(m), "A")
        self.assertEqual(p, [])

    def test_failing_dropped(self):
        e = {"p61": {"pass": False}, "X": {"pass": True}}
        m, _ = M.pick_version(e, {"p61": True, "X": True}, {"p61": True, "X": True})
        self.assertNotIn("p61", m)
        self.assertEqual(M.version_name(m), "6-X (p61 dropped)")

    def test_after_cutoff_dropped(self):
        e = {"p61": {"pass": True}, "X": {"pass": True}}
        m, _ = M.pick_version(e, {"p61": True, "X": True}, {"p61": True, "X": False})
        self.assertEqual(M.version_name(m), "A")

    def test_changed_files_dropped(self):
        e = {"p61": {"pass": True, "files_unchanged": False}}
        m, _ = M.pick_version(e, {"p61": True}, {"p61": True})
        self.assertEqual(M.version_name(m), "B")

    def test_unchecked_present_refused(self):
        _m, p = M.pick_version({}, {"p61": True}, {"p61": True})
        self.assertTrue(p)

    def test_unchecked_after_cutoff_ignored(self):
        m, p = M.pick_version({}, {"X": True}, {"X": False})
        self.assertEqual((M.version_name(m), p), ("B", []))


class EUse(unittest.TestCase):
    def test_count(self):
        with tempfile.TemporaryDirectory() as d:
            for i, n in enumerate((4, 5, 6)):
                os.makedirs(os.path.join(d, "results", "prereg%02d" % i, "explore"))
                with open(os.path.join(d, "results", "prereg%02d" % i, "explore", "eval.json"), "w") as f:
                    json.dump({"e_pools_use_number": n}, f)
            u = M.e_uses_so_far(d)
            self.assertEqual((u["count"], u["consistent"]), (6, True))


if __name__ == "__main__":
    unittest.main()
