"""Unit tests of results/prereg28/explore/p28_sim.py (pools, scores, judging). Real fit store (data/prereg06/feats);
no score of any simulated pool is computed here.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg28/explore/test_p28.py
"""
import os
import sys
import unittest
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p28_sim as ps  # noqa: E402

sim, st = ps.sim, ps.st

_CACHE = {}


def cache():
    if "c" not in _CACHE:
        _CACHE["c"] = sim.Store(ps.pp.FEATS).black_cache()
    return _CACHE["c"]


class Pools(unittest.TestCase):
    def test_choose_pool_400_equals_p06(self):
        for seed in (600, 607, 619, 500):
            self.assertEqual(ps.choose_pool_n(cache(), seed, 400), sim.choose_pool(cache(), seed))

    def test_make_pool_400_equals_p06(self):
        for seed in (600, 613):
            for scen in sim.STRUCTURES:
                a = ps.make_pool_n(cache(), seed, scen, 400)
                b = sim.make_pool(cache(), seed, scen)
                self.assertEqual(a["pool"], b["pool"])
                self.assertEqual(a["cand_rows"], b["cand_rows"])
                self.assertEqual(a["q_rows"], b["q_rows"])
                self.assertEqual(a["owner"].tolist(), b["owner"].tolist())
                self.assertEqual(a["k"].tolist(), b["k"].tolist())
                self.assertEqual(a["n_repairs"], b["n_repairs"])

    def test_scaled_structures(self):
        self.assertEqual(ps.scen_m("u390", 400), 390)
        self.assertEqual(ps.scen_m("u390", 800), 780)
        self.assertEqual(ps.scen_m("u375", 600), 562)  # round(562.5) -> 562 (banker's rounding, recorded)
        self.assertEqual(ps.scen_m("d80", 700), 140)
        self.assertEqual(ps.scen_m("d3", 800), 3)
        for n in (500, 800):
            a = ps.make_pool_n(cache(), 601, "d80", n)
            self.assertEqual(len(a["pool"]), n)
            self.assertEqual(len(set(a["pool"])), n)
            self.assertEqual(len(a["q_rows"]), n)
            self.assertEqual(sorted(Counter(a["k"].tolist()).values()), [n // 4] * 4)
            c = Counter(a["owner"].tolist())
            self.assertEqual(len(c), n - n // 5)
            o = ps.make_pool_n(cache(), 601, "o2o", n)
            self.assertEqual(sorted(o["owner"].tolist()), list(range(n)))
            for i, g in enumerate(o["q_rows"]):
                self.assertFalse(set(g) & set(o["cand_rows"][o["owner"][i]]))
                self.assertEqual(len(g), o["k"][i])

    def test_pool_too_large(self):
        with self.assertRaises(ValueError):
            ps.choose_pool_n(cache(), 600, 1000)

    def test_v2_pools(self):
        store_pid = None
        for kind in ps.V2_KINDS:
            for m in (40, 200):
                r = ps.make_v2_pool(cache(), 603, kind, m)
                n = 400 + m
                own = r["owner"].tolist()
                self.assertEqual(len(r["pool"]), n)
                self.assertEqual(len(own), n)
                c = Counter(own)
                n_out = m if kind == "out" else (m // 2 if kind == "mix" else 0)
                self.assertEqual(sum(1 for o in own if o < 0), n_out)
                self.assertEqual(len(r["outsiders"]), n_out)
                self.assertFalse(set(r["outsiders"]) & set(r["pool"]))
                inpool = [o for o in own if o >= 0]
                self.assertEqual(len(set(inpool)), 400)                 # the originals
                self.assertEqual(sum(1 for o, v in c.items() if o >= 0 and v == 2), m - n_out)
                self.assertTrue(all(v <= 2 for o, v in c.items() if o >= 0))
                self.assertEqual(n - len(set(inpool)), m)                # m candidates without a question
                self.assertEqual(sorted(Counter(r["k"].tolist()).values()), [n // 4] * 4)
                for q, g in enumerate(r["q_rows"]):
                    self.assertEqual(len(g), r["k"][q])
                    if own[q] >= 0:
                        self.assertFalse(set(g) & set(r["cand_rows"][own[q]]))
                # the questions of one in-pool owner never share games
                seen = {}
                for q, g in enumerate(r["q_rows"]):
                    if own[q] >= 0:
                        for x in g:
                            self.assertNotIn(x, seen)
                            seen[x] = q
        del store_pid


class Scores(unittest.TestCase):
    def test_q_points_outsiders_zero(self):
        S = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [5.0, 4.0, 3.0]])
        p = ps.q_points(S, [0, -1, 2])
        self.assertAlmostEqual(p[0], 1.0)
        self.assertEqual(p[1], 0.0)
        self.assertAlmostEqual(p[2], np.exp(-2))

    def test_pool_scores_q_matches_p06(self):
        rng = np.random.default_rng(5)
        F = rng.normal(size=(40, 40))
        own = rng.permutation(40)
        a, per = ps.pool_scores_q(F, own)
        b = sim.pool_scores(F, own)
        for k, v in b.items():
            self.assertEqual(a[k], v, k)
        self.assertAlmostEqual(per["on"].mean(), a["on"], places=15)


def _row(scen, r, on=1.0, off=0.9, seed=0, n=10):
    res = {}
    for name in ("V2", "V3", "V0"):
        for s in ("E0", "PA"):
            res["%s|%s" % (name, s)] = {"r_forced_cost": r, "on": on, ps.OFF: off}
        res["%s|open" % name] = {"r_forced_cost": r}
    return {"seed": seed, "scen": scen, "res": res, "n_repairs": 0, "q_owner_pid": ["p%d" % i for i in range(n)]}


def _std(r_o2o=1.0, r_non=50.0, r_near=50.0, on=1.0, off=0.9, seeds=len(ps.SEEDS)):
    out = []
    for s in range(seeds):
        rows = [_row("o2o", r_o2o, on, off, s)]
        rows += [_row(sc, r_non, seed=s) for sc in sim.NON_O2O]
        rows += [_row(sc, r_near, seed=s) for sc in sim.NEAR_O2O]
        q = {"on": [on] * 10, ps.OFF: [off] * 10}
        out.append({"rows": rows, "q": {k: q for k in ("V2|E0", "V2|PA", "V3|E0", "V3|PA")}, "lda_players": 1})
    return out


def _v2(r=50.0, seeds=ps.SEEDS):
    out = []
    for k in ps.V2_KINDS:
        for m in ps.V2_M:
            for s in seeds:
                out.append({"rows": [_row("v2%s%d" % (k, m), r, seed=s)], "lda_players": 1})
    return out


class Judge(unittest.TestCase):
    def std(self, **kw):
        return {n: _std(**kw) for n in (400,) + ps.N_GRID}

    def test_pass(self):
        j = ps.judge(self.std(), _v2())
        self.assertTrue(j["passed"], j["verdict"])
        self.assertTrue(j["V3_usable_at_every_N"])

    def test_false_open_fails_g1(self):
        s = self.std()
        s[700][0]["rows"][1]["res"]["V2|PA"]["r_forced_cost"] = 2.0
        j = ps.judge(s, _v2())
        self.assertFalse(j["G1_passed"])
        self.assertFalse(j["passed"])

    def test_open_only_false_open_fails_g1(self):
        s = self.std()
        s[500][1]["rows"][3]["res"]["V2|open"]["r_forced_cost"] = 2.25
        self.assertFalse(ps.judge(s, _v2())["G1_passed"])

    def test_seam_size_does_not_count(self):
        s = self.std()
        s[400][0]["rows"][1]["res"]["V2|PA"]["r_forced_cost"] = 2.0
        self.assertTrue(ps.judge(s, _v2())["passed"])

    def test_o2o_share(self):
        s = self.std(r_o2o=3.0)
        j = ps.judge(s, _v2())
        self.assertFalse(j["G1_passed"])
        self.assertFalse(j["G2_passed"])           # never opens -> mean gain 0

    def test_g2_negative(self):
        j = ps.judge(self.std(on=0.8, off=0.9), _v2())
        self.assertFalse(j["G2_passed"])

    def test_v3_only_fails_is_rule_not_gate(self):
        s = self.std()
        s[800][2]["rows"][2]["res"]["V3|E0"]["r_forced_cost"] = 2.26
        j = ps.judge(s, _v2())
        self.assertTrue(j["passed"])
        self.assertFalse(j["V3_usable_at_every_N"])

    def test_g3_any_open_fails(self):
        v = _v2()
        v[37]["rows"][0]["res"]["V3|open"]["r_forced_cost"] = 2.0
        j = ps.judge(self.std(), v)
        self.assertFalse(j["G3"]["passed"])
        self.assertFalse(j["passed"])

    def test_std_missing_pools_refused(self):
        s = self.std()
        s[600] = s[600][:-1]
        with self.assertRaises(SystemExit):
            ps.judge(s, _v2())
        s = self.std()
        s[500][0]["rows"] = s[500][0]["rows"][:-1]
        with self.assertRaises(SystemExit):
            ps.judge(s, _v2())

    def test_g3_missing_pools_fails(self):
        self.assertFalse(ps.judge(self.std(), _v2(seeds=ps.SEEDS[:-1]))["G3"]["passed"])

    def test_v0_never_opens(self):
        self.assertFalse(ps.opens(_row("o2o", 0.0), "V0|E0"))
        self.assertTrue(ps.opens(_row("o2o", ps.R_STAR["V2"]), "V2|E0"))
        self.assertFalse(ps.opens(_row("o2o", ps.R_STAR["V2"] + 1e-12), "V2|E0"))

    def test_constants_match_frozen(self):
        import json
        ref = json.load(open(ps.SWITCH_REPORT, encoding="utf-8"))
        for name in ("V2", "V3"):
            self.assertEqual(ref["per_combination"][name]["r_star"], ps.R_STAR[name])
            self.assertTrue(ref["per_combination"][name]["usable"])
        self.assertFalse(ref["per_combination"]["V0"]["usable"])
        self.assertEqual(ref["k_star"], 10)
        self.assertEqual(ref["off_rule"], ps.OFF)
        self.assertEqual(ref["seeds"]["evaluation"], list(ps.SEEDS))
        self.assertEqual(ref["thread_env_per_worker"], ps.THREADS)


if __name__ == "__main__":
    unittest.main()
