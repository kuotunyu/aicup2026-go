"""Checks of the PREREG_07 HumanSL scorer (hs_lib.py) on CPU with the real checkpoint (fit players only).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -rs results/prereg07/explore/pilot_humansl/test_hs_lib.py

Needs data/katago/b18c384nbt-humanv0.ckpt (323,752,318 bytes) and data/prereg07/fixtures/fit_sample.json
(hs_fixtures.py). Missing inputs FAIL (they are not skipped). Lives here, not in tests/, on purpose: it is an
exploration check and must not join the repository's frozen test suites.
Tolerances (fixed before the first run with the real weights):
  (a) issue #1154: every value within relative 1e-4 of the python numbers AND of the engine numbers the issue printed;
      hs_lib's direct forward pass within 1e-6 (absolute) of KataGo's GameState path.
  (b) >= 20 positions x 4 profiles: probabilities within 1e-5 (absolute, unmasked and masked) of
      GameState.get_model_outputs and of humanslnet_server.py; legal masks identical.
  (c) 2 fit games each of D, A, 1D, 4D, 6D: log-likelihood peaks non-decreasing with the level, peak(6D) > peak(D).
"""

import datetime
import hashlib
import math
import os
import re
import sys
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hs_lib as H  # noqa: E402
import hs_parity as PAR  # noqa: E402

VENDOR = os.path.join(H.REPO, "third_party", "katago")
_MODEL = {}


def model():
    if "m" not in _MODEL:
        if not os.path.exists(H.CKPT_PATH):
            raise AssertionError(f"{H.CKPT_PATH} missing")
        _MODEL["m"], _MODEL["state"] = H.load_model(H.CKPT_PATH, device="cpu")
    return _MODEL["m"]


def fixture():
    if "f" not in _MODEL:
        _MODEL["f"] = PAR.load_fixture()
    return _MODEL["f"]


def git_blob_id(path):
    with open(path, "rb") as f:
        b = f.read().replace(b"\r\n", b"\n")            # what git stores under core.autocrlf
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


class Vendor(unittest.TestCase):
    def listed(self):
        rows = []
        with open(os.path.join(VENDOR, "SOURCE.txt"), encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s+(\S+) \| ([0-9a-f]{40}) \| ([0-9a-f]{64}) \| (\d+)\s*$", line)
                if m:
                    rows.append(m.groups())
        return rows

    def test_files_equal_upstream_blobs(self):
        rows = self.listed()
        self.assertEqual(len(rows), 15)
        for path, blob, _sha, _n in rows:
            with self.subTest(path):
                self.assertEqual(git_blob_id(os.path.join(VENDOR, *path.split("/"))), blob)

    def test_no_unlisted_code(self):
        listed = {r[0] for r in self.listed()}
        found = set()
        for root, dirs, files in os.walk(os.path.join(VENDOR, "python")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for x in files:
                found.add(os.path.relpath(os.path.join(root, x), VENDOR).replace("\\", "/"))
        self.assertEqual(sorted(found - listed), [])

    def test_commit_pinned(self):
        with open(os.path.join(VENDOR, "SOURCE.txt"), encoding="utf-8") as f:
            self.assertIn(H.KATAGO_COMMIT, f.read())


class Guard(unittest.TestCase):
    def test_data_local_refused(self):
        p = os.path.join(H.REPO, "data", "local", "never_read.csv")
        for fn in (lambda: open(p), lambda: os.listdir(os.path.dirname(p)), lambda: os.stat(p),
                   lambda: np.load(p)):
            with self.assertRaises(H.LocalDataRefused):
                fn()

    def test_guard_is_component_wise(self):
        self.assertTrue(H._forbidden(os.path.join("D:" + os.sep, "elsewhere", "data", "local", "x.csv")))
        self.assertFalse(H._forbidden(os.path.join("C:" + os.sep, "Users", "u", "AppData", "Local", "Temp", "x")))
        self.assertFalse(H._forbidden(os.path.join(H.REPO, "data", "localx", "x")))


class Metadata(unittest.TestCase):
    def test_inverse_rank_scale(self):
        self.assertEqual([H.inverse_rank(r) for r in ("9d", "1d", "1k", "20k")], [1, 9, 10, 29])
        self.assertEqual(len(H.RANKS), 29)
        self.assertEqual([H.inverse_rank(r) for r in H.RANKS], list(range(29, 0, -1)))

    def test_cpp_preset_row(self):
        """rank_5k must equal cpp makeBasicRankProfile(14, 14, preAZ=false) filled by fillMetadataRow."""
        r = H.Profile.parse("rank_5k").row(H.Board.BLACK)
        exp = np.zeros(192, dtype=np.float64)
        exp[0] = exp[1] = 1.0
        exp[6:6 + 14] = 1.0
        exp[40:40 + 14] = 1.0
        exp[74] = 0.5
        exp[79] = 1.0
        exp[82] = 0.4 * (math.log(1200 + 60.0) - 6.5)
        exp[83] = 0.3 * (math.log(30 + 1.0) - 3.0)
        exp[84] = 0.5 * (math.log(5 + 2.0) - 1.5)
        exp[85] = 0.25 * (math.log(0 + 2.0) - 1.5)
        days = (datetime.date(2020, 3, 1) - datetime.date(1970, 1, 1)).days
        period = 7.0
        for i in range(32):
            exp[87 + 2 * i] = math.cos(days / period * 2 * math.pi)
            exp[88 + 2 * i] = math.sin(days / period * 2 * math.pi)
            period *= 80000.0 ** (1.0 / 31)
        exp[151 + 2] = 1.0
        np.testing.assert_allclose(r, exp, atol=1e-6)

    def test_options(self):
        self.assertEqual(H.Profile.parse("preaz_2d").date, datetime.date(2016, 9, 1))
        fox = H.Profile.parse("fox20220601_3d").row(H.Board.WHITE)
        self.assertEqual(fox[151 + 3], 1.0)
        self.assertEqual(fox[151 + 2], 0.0)
        unk = H.Profile.parse("kgs20200301_3d_oppunknown")
        rb, rw = unk.row(H.Board.BLACK), unk.row(H.Board.WHITE)
        for r in (rb, rw):
            self.assertEqual(r[5], 1.0)                 # opponent rank unknown
            self.assertEqual(r[4], 0.0)
            self.assertEqual(r[40:74].sum(), 0.0)
            self.assertEqual(r[6:40].sum(), 7.0)        # 3d -> inverse rank 7
        same = H.Profile.parse("rank_3d")
        np.testing.assert_array_equal(same.row(H.Board.BLACK), same.row(H.Board.WHITE))
        for n in ("rank_20k", "preaz_9d", "fox20220601_1k", "kgs20200301_5d_opp3d"):
            self.assertEqual(H.Profile.parse(n).name, n)


class Games(unittest.TestCase):
    def test_fixture_games_replay(self):
        f = fixture()
        self.assertEqual(len(f["games"]), 120)
        for g in f["games"]:
            with self.subTest(g["game_id"]):
                ft = H.featurize_game(g["sgf_content"], g["color"], sides=("target", "opponent"))
                n = ft["n_moves"]
                tgt = (n + 1) // 2 if g["color"] == "B" else n // 2
                self.assertEqual(int((ft["side"] == 0).sum()), tgt)
                self.assertEqual(len(ft["move"]), n)
                self.assertTrue(set(np.unique(ft["bin"]).tolist()) <= {0, 1})
                self.assertTrue(ft["legal"][np.arange(n), ft["move"]].all())
                pla_from_turn = np.where(ft["turn"] % 2 == 0, H.Board.BLACK, H.Board.WHITE)
                np.testing.assert_array_equal(ft["pla"], pla_from_turn)

    def test_rejects_other_sgf_forms(self):
        for bad in ("(;SZ[19];B[pd];B[dp])", "(;SZ[19];B[pd];W[])", "(;SZ[19]AB[dd];W[pd])", "(;SZ[13];B[aa])"):
            with self.assertRaises(H.HSError):
                H.parse_moves(bad)


class Checkpoint(unittest.TestCase):
    def test_size_and_sha256(self):
        rec = H.check_checkpoint(H.CKPT_PATH, full_hash=True)
        self.assertEqual(rec["bytes"], 323752318)
        self.assertNotEqual(H.CKPT_SHA256, "TO_BE_FILLED", "pin CKPT_SHA256 in hs_lib.py")
        self.assertEqual(rec["sha256"], H.CKPT_SHA256)

    def test_weights_only_load(self):
        model()
        st = _MODEL["state"]
        self.assertIn("model", st)
        self.assertIn("config", st)
        self.assertIsNotNone(st["config"].get("metadata_encoder"))


class ParityA(unittest.TestCase):
    def test_issue_1154(self):
        r = PAR.check_issue_1154(model())
        for got, py, eng in zip(r["gamestate"], PAR.ISSUE_1154["python"], PAR.ISSUE_1154["engine"]):
            self.assertLessEqual(abs(got - py) / py, 1e-4, r)
            self.assertLessEqual(abs(got - eng) / eng, 1e-4, r)
        self.assertLessEqual(r["max_abs_diff_hs_vs_gamestate"], 1e-6, r)


class ParityB(unittest.TestCase):
    def test_gamestate(self):
        r = PAR.check_gamestate(model(), fixture())
        self.assertGreaterEqual(r["positions"], 20)
        self.assertEqual(r["legal_mask_mismatches"], 0)
        self.assertLessEqual(r["max_abs_diff_unmasked"], 1e-5)
        self.assertLessEqual(r["max_abs_diff_masked"], 1e-5)

    def test_humanslnet_server(self):
        r = PAR.check_server(fixture())
        self.assertGreaterEqual(r["positions"], 20)
        self.assertLessEqual(r["max_abs_diff_unmasked"], 1e-5)


class Padding(unittest.TestCase):
    def test_padded_passes_equal_unpadded(self):
        """Scorer(pad=True) (the CUDA default) must give the same log-probabilities as pad=False; the last pass here
        is partial (13 positions, 4 per pass)."""
        g = next(x for x in fixture()["games"] if x["rank"] == "2D")
        f = H.featurize_game(g["sgf_content"], g["color"])
        sub = {k: (v[:13] if isinstance(v, np.ndarray) else v) for k, v in f.items()}
        prof = H.profiles_from_names(["rank_10k", "rank_3d"])
        a = H.Scorer(model(), prof, device="cpu", batch=8, pad=False).logp_moves(sub)
        b = H.Scorer(model(), prof, device="cpu", batch=8, pad=True).logp_moves(sub)
        self.assertEqual(a.shape, (13, 2))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)


class CurveC(unittest.TestCase):
    def test_peaks_follow_levels(self):
        r = PAR.check_curves(model(), fixture())
        self.assertTrue(r["passed"], {lv: v["peak_profile"] for lv, v in r["per_level"].items()})


if __name__ == "__main__":
    unittest.main()
