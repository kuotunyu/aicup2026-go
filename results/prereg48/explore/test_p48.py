"""prereg48 explore: small CPU tests of p48_eval.py (synthetic data only; no score of a prereg48 candidate).

    PY -m pytest -q results/prereg48/explore/test_p48.py
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p48_eval as PE  # noqa: E402

CSTAR_CFG = {"chunk": 16, "ckpt_sha256": "aa", "games": 6, "mode": None, "moves": 60, "part_items": 128,
             "spec": {"name": "se_gpool_trunk"}, "tta": 1, "weights": "raw"}


def npz(path, cfg, n=6, dim=4, seed=0, sha="aa", count=None, rows=None, scale=100.0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, emb=(scale * rng.normal(size=(n, dim))).astype(np.float32),
             count=(np.arange(n, dtype=np.int32) % 3 + 1) if count is None else count,
             ckpt_sha256=np.array(sha), ckpt=np.array("t"), ckpt_step=np.array(1, np.int64),
             embed_cfg=np.array(json.dumps(cfg)), store_row=np.arange(n) + 100 if rows is None else rows,
             players_sha256_sorted_ids=np.array("h"))
    return path


class Constants(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(PE.D_V26, 0.7467766213119217)
        self.assertEqual(PE.E_V26, 0.681614485210018)
        self.assertEqual((PE.GATE, PE.COS_MIN, PE.REL_MAX, PE.SEAM_N), (0.01, 0.99999, 1e-3, 200))
        self.assertEqual(PE.E43.WP_GRID, (0.25, 0.5, 1, 2, 4, 8, 16))
        self.assertEqual(PE.E43.K_GRID, (10, 20, 40))

    def test_paths(self):
        p = PE.c48_file("D").replace("\\", "/")
        self.assertTrue(p.endswith("data/prereg48/emb_S0/V2_step_320000_raw_tta8/store_D.npz"))
        self.assertTrue(PE.c48_file("public").replace("\\", "/").endswith("V2_step_320000_raw_tta8/pool_public.npz"))
        self.assertTrue(PE.tta1_file("D").replace("\\", "/").endswith(
            "data/prereg43/emb_S0/V2_step_320000_raw_tta1/store_D.npz"))
        self.assertTrue(PE.SEAM_VEC.replace("\\", "/").endswith(
            "data/prereg48/seam_tta1/V2_step_320000_raw_tta1/store_D_limit200.npz"))

    def test_cstar_matches_v26_record(self):
        t = PE.read_json(PE.E45.V26_TUNE)
        self.assertEqual(t["c_star"]["file"], "data/prereg43/emb_S0/V2_step_320000_raw_tta1/store_D.npz")
        self.assertEqual(t["c_star"]["embed_cfg"]["tta"], 1)


class VectorSeam(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.b = 113.0 * rng.normal(size=(200, 256)) / 16.0
        self.c = np.arange(200) % 50 + 1
        self.r = np.arange(200)

    def test_scale_relative_pass(self):
        a = self.b + 0.002 * np.sign(np.random.default_rng(2).normal(size=self.b.shape))  # like the measured seam
        a[7, 9] += 0.03                                                                # max abs ~0.03, norm ~113
        s = PE.vector_seam_stats(a, self.c, self.r, self.b, self.c, self.r)
        self.assertGreater(s["max_abs_diff"], 1e-3)
        self.assertLess(s["max_abs_over_median_norm"], 1e-3)
        self.assertTrue(s["pass"])

    def test_fails(self):
        a = self.b.copy()
        a[0, 0] += 0.5                                                                # rel ~4e-3
        self.assertFalse(PE.vector_seam_stats(a, self.c, self.r, self.b, self.c, self.r)["pass"])
        c2 = self.c.copy()
        c2[3] += 1
        self.assertFalse(PE.vector_seam_stats(self.b, c2, self.r, self.b, self.c, self.r)["pass"])
        self.assertFalse(PE.vector_seam_stats(self.b, self.c, self.r[::-1], self.b, self.c, self.r)["pass"])
        a = self.b.copy()
        a[5] = -a[5]                                                                   # cosine -1
        self.assertFalse(PE.vector_seam_stats(a, self.c, self.r, self.b, self.c, self.r)["pass"])


class C48File(unittest.TestCase):
    def test_checks(self):
        cstar = {"embed_cfg": CSTAR_CFG, "ckpt_sha256": "aa"}
        with tempfile.TemporaryDirectory() as t:
            ref = npz(os.path.join(t, "ref", "store_D.npz"), CSTAR_CFG)
            old = PE.tta1_file
            PE.tta1_file = lambda which: ref
            try:
                good = npz(os.path.join(t, "a", "store_D.npz"), dict(CSTAR_CFG, tta=8), seed=3)
                self.assertEqual(PE.check_c48_file(good, "D", cstar)["embed_cfg"]["tta"], 8)
                for name, kw in (("tta1", {"cfg": CSTAR_CFG}), ("ema", {"cfg": dict(CSTAR_CFG, tta=8, weights="ema")}),
                                 ("sha", {"cfg": dict(CSTAR_CFG, tta=8), "sha": "bb"}),
                                 ("moves", {"cfg": dict(CSTAR_CFG, tta=8, moves=61)}),
                                 ("count", {"cfg": dict(CSTAR_CFG, tta=8), "count": np.ones(6, np.int32)}),
                                 ("rows", {"cfg": dict(CSTAR_CFG, tta=8), "rows": np.arange(6)})):
                    p = npz(os.path.join(t, name, "store_D.npz"), kw.pop("cfg"), **kw)
                    with self.assertRaises(SystemExit, msg=name):
                        PE.check_c48_file(p, "D", cstar)
                with self.assertRaises(SystemExit):
                    PE.check_c48_file(os.path.join(t, "none", "store_D.npz"), "D", cstar)
            finally:
                PE.tta1_file = old


class LaterOutputs(unittest.TestCase):
    def test_detects(self):
        saved = (PE.DATA, PE.C48_DIR, PE.EVAL_JSON, PE.OUT_APPLY)
        with tempfile.TemporaryDirectory() as t:
            PE.DATA = t
            PE.C48_DIR = os.path.join(t, "emb_S0", PE.TAG8)
            PE.EVAL_JSON = os.path.join(t, "eval.json")
            PE.OUT_APPLY = os.path.join(t, "apply")
            try:
                os.makedirs(PE.C48_DIR)
                open(os.path.join(PE.C48_DIR, "store_D.npz"), "w").close()
                self.assertEqual(PE.later_outputs(), [])
                PE.no_later_outputs()
                for p in (PE.c48_file("E") + ".parts", os.path.join(t, "seam_tta1", "x", "pool_public.npz"),
                          PE.EVAL_JSON, PE.OUT_APPLY):
                    if p.endswith((".parts", "apply")):
                        os.makedirs(p)
                    else:
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        open(p, "w").close()
                    with self.assertRaises(SystemExit, msg=p):
                        PE.no_later_outputs()
                    if os.path.isdir(p):
                        os.rmdir(p)
                    else:
                        os.remove(p)
                PE.no_later_outputs()
            finally:
                PE.DATA, PE.C48_DIR, PE.EVAL_JSON, PE.OUT_APPLY = saved


class Judge(unittest.TestCase):
    def test_e_use_count(self):
        with tempfile.TemporaryDirectory() as t:
            for d in ("prereg36", "prereg37", "prereg38", "prereg39", "prereg40", "prereg43", "prereg45", "prereg47"):
                os.makedirs(os.path.join(t, "results", d, "explore"))
            for d, f in (("prereg36", "eval.json"), ("prereg37", "eval.json"), ("prereg37", "eval_backend.json"),
                         ("prereg38", "eval.json"), ("prereg39", "eval.json"), ("prereg40", "eval_report.json"),
                         ("prereg43", "eval.json")):
                open(os.path.join(t, "results", d, "explore", f), "w").close()
            prior = PE.e_judgements_so_far(t)
            self.assertEqual(prior, ["prereg36", "prereg37", "prereg38", "prereg39", "prereg43"])
            PE.judge_ready({"gate": {"pass": True}}, 6, prior)
            with self.assertRaises(SystemExit):
                PE.judge_ready({"gate": {"pass": True}}, 5, prior)
            with self.assertRaises(SystemExit):
                PE.judge_ready({"gate": {"pass": False}}, 6, prior)
            with self.assertRaises(SystemExit):
                PE.judge_ready({"gate": {"pass": True}}, 7, prior + ["prereg48"])

    def test_repo_e_use_is_6(self):
        self.assertEqual(len(PE.e_judgements_so_far()), 5)


class SeamD(unittest.TestCase):
    def test_mismatch_stops(self):
        t = PE.read_json(PE.E45.V26_TUNE)
        cs = t["c_star"]
        base = {"tune": t}
        r = {"per_pool": cs["backend"]["per_pool"], "mean": cs["backend"]["mean"]}
        top = {"wP": cs["p1"]["wP"], "k": cs["p1"]["csls_k"], "per_pool": cs["p1"]["per_pool"],
               "mean": cs["p1"]["mean"]}
        PE.check_seam_D(base, cs["backend"]["cfg"], r, top)
        with self.assertRaises(SystemExit):
            PE.check_seam_D(base, dict(cs["backend"]["cfg"], wE=4), r, top)
        with self.assertRaises(SystemExit):
            PE.check_seam_D(base, cs["backend"]["cfg"], r, dict(top, k=10))
        bad = list(top["per_pool"])
        bad[0] += 1e-12
        with self.assertRaises(SystemExit):
            PE.check_seam_D(base, cs["backend"]["cfg"], r, dict(top, per_pool=bad))


if __name__ == "__main__":
    unittest.main()
