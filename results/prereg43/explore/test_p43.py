"""prereg43 explore: small CPU tests of p43_eval.py (synthetic data only; no score of a prereg43 candidate).

    PY -m pytest -q results/prereg43/explore/test_p43.py
"""
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p43_eval as PE  # noqa: E402


class Rules(unittest.TestCase):
    def test_candidates_fixed_order(self):
        self.assertEqual([c[0] for c in PE.CANDIDATES], ["S0", "S1", "S01", "S0M", "S01M"])
        self.assertEqual(PE.cand_label_members("S01M"), ("S0", "S1", "M37"))

    def test_first_max_tie_keeps_earlier(self):
        q = {"S0": 0.5, "S1": 0.6, "S01": 0.6, "S0M": 0.1, "S01M": 0.6}
        self.assertEqual(PE.first_max(list(q), key=lambda x: q[x]), "S1")

    def test_weights_tie_raw(self):
        self.assertEqual(PE.choose_weights(0.5, 0.5), "raw")
        self.assertEqual(PE.choose_weights(0.5, 0.50001), "ema")
        self.assertEqual(PE.choose_weights(0.6, 0.5), "raw")

    def test_grid_contains_v25(self):
        self.assertEqual(PE.WP_GRID, (0.25, 0.5, 1, 2, 4, 8, 16))
        self.assertIn(8, PE.WP_GRID)
        self.assertIn(10, PE.K_GRID)
        self.assertEqual(PE.K_GRID, (10, 20, 40))

    def test_member_paths(self):
        p = PE.member_file("S1", "ema", "E").replace("\\", "/")
        self.assertTrue(p.endswith("data/prereg43/emb_S1/V2_step_320000_ema_tta1/store_E.npz"))
        p = PE.member_file("M37", None, "public").replace("\\", "/")
        self.assertTrue(p.endswith("data/prereg37/emb/step_160000/pool_public.npz"))

    def test_cfg_key_drops_game_fields(self):
        a = {"weights": "raw", "games": 5, "moves": 9, "members": [{"ckpt": "x", "embed_cfg": {"tta": 1, "games": 3}},
                                                                   {"ckpt": "m", "embed_cfg": None}]}
        b = {"weights": "raw", "games": 7, "compare": 1,
             "members": [{"ckpt": "x", "embed_cfg": {"tta": 1, "games": 9}}, {"ckpt": "m", "embed_cfg": None}]}
        self.assertEqual(PE.cfg_key(a), PE.cfg_key(b))
        b["members"][0]["embed_cfg"]["tta"] = 8
        self.assertNotEqual(PE.cfg_key(a), PE.cfg_key(b))


class Judge(unittest.TestCase):
    def test_adopt_and_reject(self):
        rng = np.random.default_rng(0)
        old = rng.uniform(0.6, 0.7, 20)
        self.assertTrue(PE.judge_rule(old + 0.02, old)["adopt"])
        self.assertFalse(PE.judge_rule(old + 0.005, old)["adopt"])          # mean below 0.01
        new = old + 0.03
        new[:3] = old[:3] - 0.001                                          # 17 of 20 higher
        r = PE.judge_rule(new, old)
        self.assertEqual(r["pools_higher"], 17)
        self.assertFalse(r["adopt"])
        new[2] = old[2] + 0.03                                             # 18 of 20
        self.assertTrue(PE.judge_rule(new, old)["adopt"])

    def test_bootstrap_seed_43(self):
        d = np.linspace(-0.01, 0.03, 20)
        a = PE.diff_rec(d, np.zeros(20))
        self.assertEqual(a, PE.diff_rec(d, np.zeros(20)))
        self.assertIn("seed 43", PE.judge_rule(d, np.zeros(20))["rule"])

    def test_e_use_count(self):
        with tempfile.TemporaryDirectory() as t:
            for d in ("prereg36", "prereg37", "prereg38", "prereg39", "prereg40", "prereg42"):
                os.makedirs(os.path.join(t, "results", d, "explore"))
            for d, f in (("prereg36", "eval.json"), ("prereg37", "eval.json"), ("prereg37", "eval_backend.json"),
                         ("prereg38", "eval.json"), ("prereg39", "eval.json"), ("prereg40", "eval.json")):
                open(os.path.join(t, "results", d, "explore", f), "w").close()
            self.assertEqual(PE.e_judgements_so_far(t), ["prereg36", "prereg37", "prereg38", "prereg39"])

    def test_made_after(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "store_E.npz")
            open(p, "w").close()
            now = os.path.getmtime(p)
            PE.made_after([p], now - 10)
            with self.assertRaises(SystemExit):
                PE.made_after([p], now + 10)


class Ensemble(unittest.TestCase):
    def _npz(self, d, name, sha, cfg, n=6, dim=4, seed=0):
        rng = np.random.default_rng(seed)
        p = os.path.join(d, name)
        np.savez(p, emb=rng.normal(size=(n, dim)).astype(np.float32), count=np.arange(n, dtype=np.int32) % 3,
                 ckpt_sha256=np.array(sha), ckpt=np.array(name), ckpt_step=np.array(1, np.int64),
                 embed_cfg=np.array(json.dumps(cfg)), store_row=np.arange(n) + 100,
                 players_sha256_sorted_ids=np.array("h"))
        return p

    def test_member_order_and_l2(self):
        with tempfile.TemporaryDirectory() as t:
            a = self._npz(t, "a.npz", "aa", {"weights": "raw", "games": 6}, seed=1)
            b = self._npz(t, "b.npz", "bb", {"weights": "ema", "games": 6}, seed=2)
            out = os.path.join(t, "ens", "store_D.npz")
            PE.ENS.main(["--in", a, "--in", b, "--label", "S01", "--out", out])
            with np.load(out) as z:
                e = z["emb"].astype(np.float64)
                cfg = json.loads(str(z["embed_cfg"]))
                self.assertEqual(e.shape, (6, 8))
                self.assertTrue(np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-6))
                self.assertEqual([m["embed_cfg"]["weights"] for m in cfg["members"]], ["raw", "ema"])
                self.assertEqual(z["count"].tolist(), (np.arange(6) % 3).tolist())


class Constants(unittest.TestCase):
    def test_v25_numbers(self):
        self.assertEqual(PE.D_V25, 0.7244595631236577)
        self.assertEqual(PE.E_V25, 0.6592585013564308)
        self.assertEqual((PE.GATE, PE.ADOPT_MEAN, PE.BOOT, PE.BOOT_SEED), (0.01, 0.01, 10000, 43))


if __name__ == "__main__":
    unittest.main()
