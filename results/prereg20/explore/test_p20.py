"""Unit tests of results/prereg20/explore/p20_sim.py (pool rules, strategy composition, ranking matrices).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg20/explore/test_p20.py
"""
import os
import sys
import unittest
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p20_sim as P  # noqa: E402


class Owners(unittest.TestCase):
    def test_d_structures(self):
        for m in (1, 5, 40, 120):
            o = P.owners_for("d%d" % m, 7, 400)
            c = Counter(o)
            self.assertEqual(len(o), 400)
            self.assertEqual(sum(1 for v in c.values() if v == 2), m)
            self.assertEqual(len(c), 400 - m)

    def test_u_and_o2o(self):
        self.assertEqual(P.owners_for("o2o", 1, 400), list(range(400)))
        self.assertEqual(len(set(P.owners_for("u300", 3, 400))), 300)

    def test_make_pool_rule_verbatim(self):
        """Same owners as p06_sim.make_pool's rule for the same scenario / seed (rng strings identical)."""
        import random
        r2 = random.Random("scen|d80|11")
        perm = list(range(400))
        r2.shuffle(perm)
        owner = perm[:320] + perm[240:320]
        r2.shuffle(owner)
        self.assertEqual(P.owners_for("d80", 11, 400), owner)


class Repair(unittest.TestCase):
    def test_capacity(self):
        ks = [k for k in (5, 10, 15, 20) for _ in range(100)]
        for struct in ("d40", "d120", "u300", "wr"):
            o = P.owners_for(struct, 5, 400)
            o2, k2, _n, _fb = P.repair(struct, 5, o, ks, [20] * 400)
            need = Counter()
            for q, ow in enumerate(o2):
                need[ow] += k2[q]
            self.assertLessEqual(max(need.values()), 20)
            self.assertEqual(sorted(k2), sorted(ks))


class Matrices(unittest.TestCase):
    def test_on_matrix(self):
        rng = np.random.default_rng(0)
        F = rng.normal(size=(6, 6))
        rest = rng.normal(size=(6, 6))
        M = P.on_matrix(F, rest)
        a = P.st.hungarian_top1(P.st.zrows(F))
        self.assertTrue(np.all(M.argmax(1) == a))
        for i in range(6):
            others = [j for j in range(6) if j != a[i]]
            self.assertTrue(np.array_equal(M[i, others], rest[i, others]))


def fake(level=1, gate06=True, gateB=True, r06=1.0, rB=1.0, pts=None):
    base = {"off06": [0.5], "on06": [0.6], "off10": [0.7], "onB": [0.8], "Q04": [0.1], "E0": [0.05]}
    base.update(pts or {})
    return {"level": level, "gate06": gate06, "gateB": gateB, "r06": r06, "r_star06": 2.0, "usable06": True,
            "rB": rB, "chosen": "V2", "pts": base}


class Strategies(unittest.TestCase):
    def test_current_keeps_06on_when_public_agrees(self):
        S, d = P.strategies(fake(), fake(), 2.0)
        self.assertTrue(d["on06_final"])
        self.assertEqual(S["a"].tolist(), [0.6])
        self.assertEqual(S["b"].tolist(), [0.7])
        self.assertEqual(S["c"].tolist(), [0.8])

    def test_vetoes(self):
        pub = fake(pts={"on06": [0.4], "onB": [0.6]})
        S, d = P.strategies(fake(), pub, 2.0)
        self.assertFalse(d["on06_final"])
        self.assertFalse(d["onB_final"])
        self.assertEqual(S["a"].tolist(), [0.7])
        self.assertEqual(S["c"].tolist(), [0.7])
        self.assertEqual(S["c0"].tolist(), [0.8])

    def test_detector_closed(self):
        S, d = P.strategies(fake(r06=3.0, rB=3.0), fake(), 2.0)
        self.assertEqual(S["a"].tolist(), [0.7])
        self.assertEqual(S["c"].tolist(), [0.7])

    def test_no_threshold(self):
        S, d = P.strategies(fake(), fake(), None)
        self.assertFalse(d["openB"])
        self.assertEqual(S["c"].tolist(), [0.7])

    def test_b_missing_falls_back_to_current(self):
        S, d = P.strategies(fake(gateB=False), fake(), 2.0)
        self.assertFalse(d["condB"])
        self.assertEqual(S["c"].tolist(), S["a"].tolist())
        self.assertEqual(S["a"].tolist(), [0.6])
        self.assertEqual(S["b"].tolist(), [0.5])

    def test_guard_level3(self):
        S, d = P.strategies(fake(level=3), fake(level=3), 2.0)
        self.assertFalse(d["cond06"])
        self.assertFalse(d["condB"])
        self.assertEqual(S["a"].tolist(), [0.7])
        self.assertEqual(S["c"].tolist(), [0.7])
        S2, d2 = P.strategies(fake(level=3), fake(level=3), 2.0, ignore_guard=True)
        self.assertEqual(S2["a"].tolist(), [0.6])
        self.assertEqual(S2["c"].tolist(), [0.8])

    def test_06_gate_fail_level3_uses_cnn_rows(self):
        S, _d = P.strategies(fake(level=3, gate06=False), fake(level=3), 2.0)
        self.assertEqual(S["a"].tolist(), [0.05])
        S, _d = P.strategies(fake(level=1, gate06=False), fake(), 2.0)
        self.assertEqual(S["a"].tolist(), [0.1])


class MoreStrategies(unittest.TestCase):
    def test_level2_no_switch_but_06_used(self):
        S, d = P.strategies(fake(level=2), fake(), 2.0)
        self.assertFalse(d["cond06"])
        self.assertFalse(d["condB"])
        self.assertEqual(S["a"].tolist(), [0.7])
        self.assertEqual(S["c"].tolist(), [0.7])

    def test_bon_needs_v2(self):
        pv = fake()
        pv["chosen"] = "V3"
        S, d = P.strategies(pv, fake(), 2.0)
        self.assertFalse(d["openB"])
        self.assertEqual(S["c"].tolist(), [0.7])
        self.assertTrue(d["a_06on_but_c_not_BON"])


class RepairEqualsP10(unittest.TestCase):
    def test_wr(self):
        ks = [k for k in (5, 10, 15, 20) for _ in range(100)]
        import random
        random.Random("k|3").shuffle(ks)
        o1, k1, n1, f1 = P.pf10._wr_questions(3, 400, list(ks), [20] * 400)
        o2, k2, n2, f2 = P.repair("wr", 3, P.owners_for("wr", 3, 400), list(ks), [20] * 400)
        self.assertEqual((o1, k1, n1, f1), (o2, k2, n2, f2))


def fake_row(seed, role, scen, level=1, r06=1.0, rB=1.0, pts=None, n=2):
    base = {"off06": [0.5] * n, "on06": [0.6] * n, "off10": [0.7] * n, "onB": [0.8] * n, "Q04": [0.1] * n,
            "E0": [0.05] * n}
    base.update(pts or {})
    return {"seed": seed, "role": role, "scen": scen, "level": level, "gate06": True, "gateB": True, "r06": r06,
            "r_star06": 2.0, "usable06": True, "rB": rB, "chosen": "V2", "pts": base,
            "owner_pid": ["p%d" % i for i in range(n)]}


class Judge(unittest.TestCase):
    def setUp(self):
        self.saved = P.EVAL_TRIALS
        P.EVAL_TRIALS = 2

    def tearDown(self):
        P.EVAL_TRIALS = self.saved

    def results(self, priv_rb=1.0, nov_level=3):
        rows = []
        for i in range(2):
            for scen in P.EVAL_SCEN:
                lv = nov_level if scen == "nov4" else 1
                rows.append(fake_row(P.PRIV_SEED0 + i, "private", scen, level=lv, rB=priv_rb))
                rows.append(fake_row(P.PUB_SEED0 + i, "public", scen, level=lv))
        return [{"rows": rows}]

    def test_gates_pass(self):
        rep = P.judge(self.results(), 2.0)
        g = rep["gates"]
        self.assertAlmostEqual(g["G1_o2o"]["c_minus_a"][0], 0.2)
        self.assertTrue(g["G1_o2o"]["passed"])
        # the B' detector opens in every d40 / d120 pool of this fake: false-on 100 % -> G2 fails
        self.assertEqual(g["G2_non_o2o"]["B_detector_false_on_pools"], 4)
        self.assertFalse(g["G2_non_o2o"]["passed"])
        self.assertIn("nov4_registered", rep["by_scenario"])
        self.assertIn("nov4_guard_ignored", rep["by_scenario"])
        self.assertAlmostEqual(rep["by_scenario"]["nov4_registered"]["c_minus_a"][0], 0.0)
        self.assertAlmostEqual(g["G3_nov4_guard_ignored"]["c_minus_a"][0], 0.2)

    def test_closed_detector(self):
        rep = P.judge(self.results(priv_rb=3.0), 2.0)
        g = rep["gates"]
        self.assertAlmostEqual(g["G1_o2o"]["c_minus_a"][0], 0.1)    # (a) 06-ON 0.6, (c) B'-OFF 0.7
        self.assertEqual(g["G2_non_o2o"]["B_detector_false_on_pools"], 0)
        self.assertTrue(g["G2_non_o2o"]["passed"])


@unittest.skipUnless(os.path.exists(P.SETS), "mock sets.json not present")
class Split(unittest.TestCase):
    def test_split(self):
        ix = P.pf10.SetsIndex(P.read_json(P.SETS))
        dev, ev = P.split_players(ix)
        self.assertFalse(set(dev) & set(ev))
        self.assertEqual(len(dev) + len(ev), 926)
        pool = P.choose_eval(ix, ev, 2000)
        self.assertEqual(len(pool), 400)
        self.assertTrue(set(pool) <= set(ev))
        dp = P.choose_dev(ix, dev, 1000)
        self.assertEqual(len(set(dp)), 400)
        self.assertTrue(set(dp) <= set(dev))


if __name__ == "__main__":
    unittest.main()
