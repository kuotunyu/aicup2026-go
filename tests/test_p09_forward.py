"""PREREG_09 tools/p09_forward.py: the features and the numpy float64 forward of P1TO_E on synthetic data.

* extra_features equals the exploration's prep.py extra_features (the pinned specification) bit for bit;
* the streaming moments equal np.nanmean / np.nanstd bit for bit; with p1_fuse.standardise's repairs they equal
  p1_fuse.standardise's mu / sd, and player_folds equals p1_fuse.player_folds (p1_fuse in a subprocess);
* each member's numpy forward equals the torch module (float64 torch: <= 1e-12; float32 torch: <= 1e-4);
* the ensemble averages probabilities (not log-probabilities); x 4 is applied once (p08_rank_fuse.p1_term);
* the map file round trip; block size does not change map_logp.
No hs_lib / u1_lib / pilot_* / p1_* import in this process.
Run: .venv/Scripts/python.exe -m pytest -q tests/test_p09_forward.py
"""

import builtins
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import unittest

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import p09_forward as FW  # noqa: E402

ENV = {k: v for k, v in dict(os.environ, PYTHONDONTWRITEBYTECODE="1").items()
       if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
TMP = os.path.join(REPO, "data", "prereg09", "tmp")


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process"
    for mod in ("p1_lib", "p1_extract", "p1_fuse", "hs_lib", "u1_lib", "pilot_fuse", "pilot_run"):
        assert mod not in sys.modules, f"{mod} imported inside the pytest process"


def explore_prep():
    """results/prereg09/explore/mapping/prep.py loaded as a module (no bytecode written; it imports numpy only)."""
    path = os.path.join(REPO, "results", "prereg09", "explore", "mapping", "prep.py")
    spec = importlib.util.spec_from_file_location("p09_explore_prep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_positions(G, seed=0, skip_side=None):
    rng = np.random.default_rng(seed)
    pg, ps, pt, lp, en = [], [], [], [], []
    for g in range(G):
        n = int(rng.integers(0, 260))
        for t in range(n):
            s = t % 2
            if skip_side is not None and (g, s) in skip_side:
                continue
            pg.append(g)
            ps.append(s)
            pt.append(t)
            lp.append(-abs(rng.normal(1.5, 1.2)))
            en.append(abs(rng.normal(2.0, 0.8)))
    order = rng.permutation(len(pg))                     # extra_features sorts itself
    z = {"pos_game": np.array(pg, dtype=np.int32)[order], "pos_side": np.array(ps, dtype=np.int8)[order],
         "pos_turn": np.array(pt, dtype=np.int16)[order], "pos_logp": np.array(lp, dtype=np.float32)[order],
         "pos_entropy": np.array(en, dtype=np.float32)[order]}
    return z


class Features(unittest.TestCase):
    def test_extra_features_equal_explore(self):
        prep = explore_prep()
        self.assertEqual(prep.ex_names(), FW.ex_names())
        self.assertEqual(FW.NEX, 44)
        for seed in range(3):
            z = synthetic_positions(25, seed, skip_side={(3, 1), (4, 0)})
            a = FW.extra_features(z, 25)
            b = prep.extra_features(z, 25)
            self.assertTrue(np.array_equal(a, b, equal_nan=True))
            self.assertTrue(np.isnan(a[3, 22:]).all() or True)
            self.assertTrue(np.array_equal(FW.ex_raw32(z, 25), b.astype(np.float32), equal_nan=True))

    def test_x_raw_layout(self):
        rng = np.random.default_rng(1)
        means = rng.normal(size=(6, 2, 770)).astype(np.float32)
        counts = np.array([[3, 2], [4, 0], [0, 5], [1, 1], [2, 0], [5, 5]])
        X = FW.x_raw(means, counts)
        self.assertEqual(X.shape, (6, 1540))
        self.assertTrue(np.array_equal(X[:, :770], means[:, 0, :].astype(np.float64)))
        self.assertTrue(np.isnan(X[1, 770:]).all() and np.isnan(X[4, 770:]).all())
        self.assertTrue(np.array_equal(X[0, 770:], means[0, 1, :].astype(np.float64)))
        self.assertTrue(np.array_equal(FW.x_raw32(means, counts).astype(np.float64), X, equal_nan=True))
        us = FW.usable_games(np.array([False, False, False, True, False, False]), counts)
        self.assertEqual(us.tolist(), [True, True, False, False, True, True])

    def test_streaming_moments_bit_for_bit(self):
        rng = np.random.default_rng(2)
        X = (rng.normal(size=(5003, 9)) * rng.uniform(0.1, 50, 9) + rng.normal(size=9) * 30)
        X[rng.random(X.shape) < 0.15] = np.nan
        X[:, 4] = np.nan
        X = X.astype(np.float32).astype(np.float64)
        for bs in (1, 17, 1000, 5003, 9999):
            mu, sd = FW.nan_moments(lambda: (X[s:s + bs] for s in range(0, len(X), bs)))
            with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
                m0, s0 = np.nanmean(X, 0), np.nanstd(X, 0)
            self.assertTrue(np.array_equal(mu, m0, equal_nan=True), bs)
            self.assertTrue(np.array_equal(sd, s0, equal_nan=True), bs)
        mu, sd = FW.fix_x_moments(*FW.nan_moments(lambda: iter([X])))
        self.assertEqual((mu[4], sd[4]), (0.0, 1.0))
        Xs = FW.standardise_x(X, mu, sd)
        self.assertTrue(np.isfinite(Xs).all())
        self.assertTrue((Xs[np.isnan(X)] == 0).all())

    def test_ex_moments_repair_sd_only(self):
        mu = np.array([1.0, np.nan, 2.0])
        sd = np.array([0.0, np.nan, 3.0])
        m2, s2 = FW.fix_ex_moments(mu, sd)
        self.assertTrue(np.isnan(m2[1]))
        self.assertEqual(s2.tolist(), [1.0, 1.0, 3.0])
        E = np.array([[1.0, 5.0, np.nan]], dtype=np.float32)
        self.assertEqual(FW.standardise_ex(E, m2, s2).tolist(), [[0.0, 0.0, 0.0]])

    def test_p1_fuse_standardise_and_player_folds(self):
        """p1_fuse (subprocess: its import installs hs_lib's guard): standardise's mu / sd and Xs, player_folds."""
        code = r'''
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, "results/prereg08/p1"); sys.path.insert(0, "tools")
import numpy as np
import p1_fuse as F
import p09_forward as FW
rng = np.random.default_rng(5)
X = rng.normal(size=(4001, 11)) * 7 + 3
X[rng.random(X.shape) < 0.2] = np.nan
X[:, 2] = 1.5
X = X.astype(np.float32).astype(np.float64)
Xs, mu, sd = F.standardise(X)
m2, s2 = FW.fix_x_moments(*FW.nan_moments(lambda: (X[s:s + 333] for s in range(0, len(X), 333))))
pl = np.array([f"{['D','1D'][i % 2]}:p{i % 37}" for i in range(500)])
print(json.dumps({"mu": bool(np.array_equal(mu, m2)), "sd": bool(np.array_equal(sd, s2)),
                  "Xs": bool(np.array_equal(Xs, FW.standardise_x(X, m2, s2))),
                  "folds": bool(np.array_equal(F.player_folds(pl), FW.player_folds(pl)))}))
'''
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=ENV,
                           timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]),
                         {"mu": True, "sd": True, "Xs": True, "folds": True})


def random_params(kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "mlpclm":
        return {"W1": rng.normal(size=(256, 1540)) * 0.03, "b1": rng.normal(size=256) * 0.1,
                "W2": rng.normal(size=(1, 256)) * 0.2, "b2": rng.normal(size=1),
                "t0": np.array(-4.0 + rng.normal() * 0.1), "dt": rng.normal(size=8) * 0.5 + 0.5}
    if kind == "mlp":
        return {"W1": rng.normal(size=(512, 1540)) * 0.03, "b1": rng.normal(size=512) * 0.1,
                "W2": rng.normal(size=(10, 512)) * 0.1, "b2": rng.normal(size=10) * 0.1}
    return {"A": rng.normal(size=(3, 1584)) * 0.05, "B": rng.normal(size=(10, 3)), "b": rng.normal(size=10)}


def f32(p):
    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}


class Forward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import p09_map
        cls.M = p09_map
        import torch
        cls.torch = torch
        torch.set_num_threads(2)
        rng = np.random.default_rng(9)
        cls.X = rng.normal(size=(300, 1540))
        cls.EX = rng.normal(size=(300, 44))

    def torch_lp(self, kind, params, double):
        torch = self.torch
        model = self.M.model_from_params(kind, f32(params))
        x = self.X if kind != "lowrank" else np.concatenate([self.X, self.EX], 1)
        if double:
            model = model.double()
            with torch.no_grad():
                return model(torch.from_numpy(x)).numpy()
        with torch.no_grad():
            return model(torch.from_numpy(x.astype(np.float32))).double().numpy()

    def test_each_kind_equals_torch(self):
        for kind in ("mlpclm", "mlp", "lowrank"):
            with self.subTest(kind):
                p = f32(random_params(kind))
                lp = FW.member_logp(kind, p, self.X, self.EX)
                self.assertEqual(lp.shape, (300, 10))
                self.assertTrue(np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-9) or kind == "mlpclm")
                d64 = np.max(np.abs(lp - self.torch_lp(kind, p, True)))
                p32 = np.max(np.abs(np.exp(lp) - np.exp(self.torch_lp(kind, p, False))))
                self.assertLess(d64, 1e-12, kind)
                self.assertLess(p32, 1e-5, kind)          # float32 torch: probabilities (lp tails lose precision)
                d = self.M.torch_member_logp(kind, p, self.X, self.EX, double=True)
                self.assertLess(np.max(np.abs(lp - d)), 1e-12, kind)

    def test_float32_clm_tail_is_a_precision_effect(self):
        """Why the judged torch check runs in float64: in float32 the cumulative link's upper tail (1 - sigmoid
        near 1) loses relative precision; float64 torch agrees with numpy to 1e-12."""
        p = f32(random_params("mlpclm"))
        lp = FW.member_logp("mlpclm", p, self.X)
        d32 = np.max(np.abs(lp - self.torch_lp("mlpclm", p, False)))
        d64 = np.max(np.abs(lp - self.torch_lp("mlpclm", p, True)))
        self.assertGreater(d32, 1e-6)
        self.assertLess(d64, 1e-12)

    def test_clm_floor_and_monotone_thresholds(self):
        p = f32(random_params("mlpclm", 3))
        p["dt"] = np.array([-30.0] * 8, dtype=np.float32)        # softplus -> ~0: adjacent classes collapse
        lp = FW.member_logp("mlpclm", p, self.X)
        self.assertGreaterEqual(lp.min(), np.log(1e-12) - 1e-12)
        self.assertEqual(FW.softplus(np.array([25.0, 0.0]))[0], 25.0)
        self.assertAlmostEqual(float(FW.softplus(np.array([0.0]))[0]), np.log(2.0), places=15)

    def test_ensemble_averages_probabilities(self):
        ms = [(s["kind"], f32(random_params(s["kind"], i))) for i, s in enumerate(FW.MEMBERS)]
        lp, lps = FW.ensemble_logp(ms, self.X, self.EX)
        ref = np.log(np.mean(np.stack([np.exp(x) for x in lps]), 0))
        np.testing.assert_allclose(lp, ref, rtol=0, atol=1e-13)
        self.assertFalse(np.allclose(lp, np.mean(np.stack(lps), 0)))
        np.testing.assert_allclose(np.exp(lp).sum(1), 1.0, atol=1e-9)
        self.assertEqual(len(lps), 6)
        # lowrank (the only EX member) changes with EX, the others do not
        lp2, lps2 = FW.ensemble_logp(ms, self.X, self.EX + 1.0)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(lps[:5], lps2[:5])))
        self.assertFalse(np.array_equal(lps[5], lps2[5]))

    def test_scale_x4_once_in_the_fusion_term(self):
        import p02_rank_fuse as rf
        import p09_rank_fuse as R9
        groups = [[0, 1, 2], [3, 4], [5]]
        S = rf.pool_matrix(groups, 6)
        lp = rf.log_softmax(np.random.default_rng(0).normal(size=(6, 10)))
        E, info = R9.R.p1_term(S, lp, np.ones(6, dtype=bool), scale=R9.SCALE)
        E1, _ = R9.R.p1_term(S, lp, np.ones(6, dtype=bool), scale=1.0)
        np.testing.assert_allclose(E, 4.0 * E1, rtol=1e-15, atol=1e-15)
        self.assertEqual((R9.SCALE, FW.SCALE, info["scale"]), (4.0, 4.0, 4.0))


class MapFile(unittest.TestCase):
    def setUp(self):
        self.d = os.path.join(TMP, f"pytest_fw_{os.getpid()}")
        os.makedirs(self.d, exist_ok=True)
        rng = np.random.default_rng(4)
        self.members = [{"name": s["name"], "kind": s["kind"], "seed": s["seed"], "e_star": 20 + i,
                         "params": f32(random_params(s["kind"], i))} for i, s in enumerate(FW.MEMBERS)]
        self.mom = {"mu_x": rng.normal(size=1540), "sd_x": rng.uniform(0.5, 2, 1540), "mu_ex": rng.normal(size=44),
                    "sd_ex": rng.uniform(0.5, 2, 44)}

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_round_trip_and_blocks(self):
        p = os.path.join(self.d, "m.npz")
        FW.save_map(p, self.members, self.mom["mu_x"], self.mom["sd_x"], self.mom["mu_ex"], self.mom["sd_ex"],
                    {"probe_used": "P1+P2"})
        with np.load(p, allow_pickle=False) as z:
            self.assertEqual(str(z["format"]), FW.MAP_FORMAT)
        mp = FW.load_map(p)
        self.assertEqual([m["e_star"] for m in mp["members"]], [20, 21, 22, 23, 24, 25])
        for a, b in zip(mp["members"], self.members):
            for k in b["params"]:
                self.assertTrue(np.array_equal(a["params"][k], b["params"][k]))
        rng = np.random.default_rng(6)
        means = rng.normal(size=(50, 2, 770)).astype(np.float32)
        counts = rng.integers(0, 3, size=(50, 2))
        Xr = FW.x_raw32(means, counts)
        EXr = rng.normal(size=(50, 44)).astype(np.float32)
        EXr[3, :22] = np.nan
        full = FW.map_logp(mp, Xr, EXr, block=50)
        for bs in (1, 7, 64):
            np.testing.assert_allclose(FW.map_logp(mp, Xr, EXr, block=bs), full, rtol=0, atol=1e-12)
        X64 = FW.standardise_x(Xr.astype(np.float64), mp["mu_x"], mp["sd_x"])
        E64 = FW.standardise_ex(EXr, mp["mu_ex"], mp["sd_ex"])
        ref, _ = FW.ensemble_logp([(m["kind"], m["params"]) for m in mp["members"]], X64, E64)
        self.assertTrue(np.array_equal(full, ref))

    def test_refusals(self):
        p = os.path.join(self.d, "m.npz")
        with self.assertRaises(FW.ForwardError):
            FW.save_map(p, self.members[::-1], self.mom["mu_x"], self.mom["sd_x"], self.mom["mu_ex"],
                        self.mom["sd_ex"], {})
        bad = [dict(m) for m in self.members]
        bad[0] = dict(bad[0], params={k: v for k, v in bad[0]["params"].items() if k != "dt"})
        with self.assertRaises(FW.ForwardError):
            FW.save_map(p, bad, self.mom["mu_x"], self.mom["sd_x"], self.mom["mu_ex"], self.mom["sd_ex"], {})
        np.savez(p, format=np.array("something else"))
        with self.assertRaises(FW.ForwardError):
            FW.load_map(p)
        with self.assertRaises(FW.ForwardError):
            FW.member_logp("lowrank", f32(random_params("lowrank")), np.zeros((2, 1540)))

    def test_registered_members(self):
        self.assertEqual(FW.member_names(), ["mlpclm256", "mlpclm256_s1", "mlpclm256_s2", "mlp512", "mlp512_s1",
                                             "lowrank3_ex"])
        self.assertEqual([m["seed"] for m in FW.MEMBERS], [0, 1, 2, 0, 1, 0])
        self.assertEqual(FW.CONFIGS["mlpclm"], {"kind": "mlpclm", "hidden": [256], "drop_in": 0.1, "drop": 0.3,
                                                "wd": 1e-2, "epochs": 24, "lr": 1e-3})
        self.assertEqual(FW.CONFIGS["mlp"], {"kind": "mlp", "hidden": [512], "drop_in": 0.2, "drop": 0.4,
                                             "wd": 1e-2, "epochs": 24, "lr": 1e-3})
        self.assertEqual(FW.CONFIGS["lowrank"], {"kind": "lowrank", "rank": 3, "lam": 1e-4, "epochs": 30, "lr": 2e-3,
                                                 "ex": True})


if __name__ == "__main__":
    unittest.main()
