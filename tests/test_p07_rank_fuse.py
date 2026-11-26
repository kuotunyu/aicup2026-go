"""PREREG_07 tools/p07_rank_fuse.py: unit tests of the pure pieces (masked E_HS, lp_HS from the frozen map, direction
check, move cross-check, gates, apply's score_allowed, output layout, modes). No hs_lib / u1_lib / pilot_* import in
this process (spec 0); the slow end-to-end preview (check -> selfcheck -> tune into data/prereg07/tmp) runs in a
subprocess only with C10_SLOW set.
Run: .venv/Scripts/python.exe -m pytest -q tests/test_p07_rank_fuse.py
"""

import builtins
import json
import math
import os
import shutil
import subprocess
import sys
import unittest

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import p02_rank_fuse as rf  # noqa: E402
import p07_rank_fuse as P  # noqa: E402


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process (spec 0)"


def lp_rows(n, seed=0):
    rng = np.random.default_rng(seed)
    return rf.log_softmax(rng.normal(size=(n, 10)))


class MaskedE(unittest.TestCase):
    def setUp(self):
        self.groups = [[0, 1, 2], [3, 4], [5], []]
        self.S = rf.pool_matrix(self.groups, 7)
        self.lp = lp_rows(7)

    def test_all_usable_is_pooled_bit_for_bit(self):
        usable = np.array([True] * 6 + [False])          # game 6 is in no question
        E, info = P.masked_E(self.S, self.lp, usable)
        self.assertTrue(np.array_equal(E, rf.pooled(self.S, self.lp, "log")))
        self.assertEqual(info["path"], "pooled")
        self.assertEqual(info["unusable_slots_in_questions"], 0)

    def test_unusable_games_are_dropped_and_renormalised(self):
        usable = np.array([True, False, True, False, False, True, True])
        E, info = P.masked_E(self.S, self.lp, usable)
        self.assertEqual(info["path"], "masked")
        np.testing.assert_allclose(E[0], (self.lp[0] + self.lp[2]) / 2, rtol=0, atol=1e-15)
        self.assertTrue(np.array_equal(E[1], np.zeros(10)))      # no usable game -> 0 vector
        np.testing.assert_allclose(E[2], self.lp[5], rtol=0, atol=0)
        self.assertTrue(np.array_equal(E[3], np.zeros(10)))
        self.assertEqual(info["questions_without_usable_game"], 2)

    def test_nan_on_unusable_rows_does_not_leak(self):
        lp = self.lp.copy()
        lp[1] = np.nan
        usable = np.array([True, False, True, True, True, True, True])
        E, _ = P.masked_E(self.S, lp, usable)
        self.assertTrue(np.isfinite(E).all())

    def test_usable_definition(self):
        u = P.usable_games([True, True, False, True], [False, True, False, False], [3, 3, 3, 0])
        self.assertEqual(u.tolist(), [True, False, False, False])

    def test_failures_count_no_move_games(self):
        self.assertEqual(P.hs_failures([False, True, False], [10, 10, 0]).tolist(), [False, True, True])


class HSMap(unittest.TestCase):
    def test_hs_lp_matches_the_pilot_formula(self):
        rng = np.random.default_rng(1)
        model = {"mu": rng.normal(size=14), "sd": rng.uniform(0.5, 2, size=14), "W": rng.normal(size=(14, 10)),
                 "b": rng.normal(size=10)}
        sums = rng.normal(-300, 50, size=(6, 14))
        counts = np.array([100, 0, 80, 90, 70, 60])
        failed = np.array([False, False, True, False, False, False])
        lp, ok = P.hs_lp(model, sums, counts, failed)
        self.assertEqual(ok.tolist(), [True, False, False, True, True, True])
        x = sums[ok] / counts[ok, None]
        Z = ((x - model["mu"]) / model["sd"]) @ model["W"] + model["b"]
        ref = Z - Z.max(1, keepdims=True) - np.log(np.exp(Z - Z.max(1, keepdims=True)).sum(1, keepdims=True))
        self.assertTrue(np.array_equal(lp[ok], ref))
        self.assertTrue(np.array_equal(lp[~ok], np.zeros((2, 10))))

    def test_load_hs_map_refuses_other_profiles(self):
        d = os.path.join(P.TMP, f"pytest_map_{os.getpid()}")
        os.makedirs(d, exist_ok=True)
        try:
            p = os.path.join(d, "m.npz")
            np.savez(p, format=np.array(P.FORMATS["hs_map"]), profiles=np.array(P.PROFILES[::-1]), mu=np.zeros(14),
                     sd=np.ones(14), W=np.zeros((14, 10)), b=np.zeros(10), **{"lambda": np.array(1e-4)})
            with self.assertRaises(P.P07Error):
                P.load_hs_map(p)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Direction(unittest.TestCase):
    def test_threshold_is_the_stricter(self):
        self.assertAlmostEqual(P.direction_record(0.5, 100, 0.9, 0.078)["threshold"], 0.45)
        self.assertAlmostEqual(P.direction_record(0.5, 100, 0.1, 0.078)["threshold"], 0.078)
        self.assertFalse(P.direction_record(0.077, 100, 0.1, 0.078)["passed"])
        self.assertTrue(P.direction_record(0.078, 100, 0.1, 0.078)["passed"])
        self.assertFalse(P.direction_record(float("nan"), 100, 0.1, 0.078)["passed"])

    def test_rho_on_aligned_and_scrambled_questions(self):
        rng = np.random.default_rng(2)
        lvl = rng.integers(0, 10, size=400)
        P_q = np.full((400, 10), 0.01)
        P_q[np.arange(400), lvl] = 0.91
        E = np.full((400, 10), -5.0)
        E[np.arange(400), lvl] = 0.0
        rho, n = P.hs_rho(E, P_q, np.ones(400, dtype=bool))
        self.assertGreater(rho, 0.95)
        self.assertEqual(n, 400)
        rho2, _ = P.hs_rho(E[rng.permutation(400)], P_q, np.ones(400, dtype=bool))
        self.assertLess(abs(rho2), 0.2)


class CrossCheck(unittest.TestCase):
    def test_counts_mismatches_only_where_neither_failed(self):
        r = P.move_cross_check([100, 101, 50, 7], [50, 51, 25, 4], [False, False, True, False],
                               [100, 102, 49, 7], [50, 51, 25, 3], [False, False, False, True], ["a", "b", "c", "d"])
        self.assertEqual(r["compared"], 2)
        self.assertEqual(r["mismatches"], 1)
        self.assertEqual(r["first_mismatches"][0]["id"], "b")
        self.assertEqual((r["only_hs_failed"], r["only_o4_failed"]), (1, 1))

    def test_colour_flip_is_caught(self):
        # a colour flip changes the target move count by one in about half of the games
        tot = np.array([101, 100, 151, 150])
        hs_t = np.array([51, 50, 76, 75])
        o4_t = tot - hs_t
        r = P.move_cross_check(tot, hs_t, np.zeros(4, bool), tot, o4_t, np.zeros(4, bool))
        self.assertEqual(r["mismatches"], 2)


class Gates(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.n = 600
        self.y = np.repeat(np.arange(10), 60)
        self.player = np.arange(self.n) // 2
        self.boot = rf.PlayerBootstrap(self.player, self.y, self.n // 2)
        self.sc_q = rng.uniform(0, 1, self.n)

    def mono(self, rb, lo, b, w, n=P.MONO_QUESTIONS):
        return {"rank_balanced": rb, "ci95": [lo, lo + 0.02], "n_questions": n,
                "by_colour": {"B": {"rank_balanced": b}, "W": {"rank_balanced": w}}}

    def test_pass_and_each_failure(self):
        sc_h = self.sc_q + 0.05
        rb_q = 0.6
        g = P.gate_record(rb_q + 0.0142, rb_q, sc_h, self.sc_q, self.boot, self.mono(0.015, 0.009, 0.01, 0.01), 0)
        self.assertTrue(g["passed"])
        self.assertFalse(P.gate_record(rb_q + 0.0099, rb_q, sc_h, self.sc_q, self.boot,
                                       self.mono(0.015, 0.009, 0.01, 0.01), 0)["G1"]["passed"])
        self.assertTrue(P.gate_record(rb_q + 0.010 - 1e-13, rb_q, sc_h, self.sc_q, self.boot,
                                      self.mono(0.015, 0.009, 0.01, 0.01), 0)["G1"]["passed"])
        self.assertFalse(P.gate_record(rb_q + 0.02, rb_q, self.sc_q - 0.01, self.sc_q, self.boot,
                                       self.mono(0.015, 0.009, 0.01, 0.01), 0)["G3"]["passed"])
        for m, failed in ((self.mono(0.015, -0.001, 0.01, 0.01), 0), (self.mono(0.015, 0.009, 0.01, -0.001), 0),
                          (self.mono(0.015, 0.009, 0.01, 0.01), 307), (self.mono(0.015, 0.009, 0.01, 0.01, 1000), 0)):
            self.assertFalse(P.gate_record(rb_q + 0.02, rb_q, sc_h, self.sc_q, self.boot, m, failed)["G4"]["passed"])

    def test_bound_hits(self):
        self.assertEqual(P.bound_hits({"OP": 0.4, "SWB": 0.0, "HS": 4.0, "O4": 3.9999999999}),
                         {"SWB": "lower", "HS": "upper", "O4": "upper"})


class ApplyDecision(unittest.TestCase):
    ok_cross = {"mismatches": 0}
    ok_dir = {"passed": True, "rho": 0.9, "threshold": 0.47}

    def test_all_good_needs_the_audit(self):
        # apply never writes True: only apply --finalize does, with an audit bound to these outputs
        for audit in ("missing", "ok", "unbound"):
            self.assertEqual(P.apply_decision(0, None, True, self.ok_cross, self.ok_dir, audit, True)[0],
                             "pending_audit")
        self.assertIs(P.apply_decision(0, None, True, self.ok_cross, self.ok_dir, "bug", True)[0], False)

    def test_each_blocker(self):
        cases = [(1, None, True, self.ok_cross, self.ok_dir, True), (96, "pass moves", True, self.ok_cross, self.ok_dir,
                                                                      True),
                 (0, None, False, self.ok_cross, self.ok_dir, True), (0, None, True, {"mismatches": 1}, self.ok_dir, True),
                 (0, None, True, self.ok_cross, {"passed": False}, True), (0, None, True, self.ok_cross, self.ok_dir,
                                                                             False)]
        for fc, acc, q, cr, di, fo in cases:
            allowed, reasons = P.apply_decision(fc, acc, q, cr, di, "ok", fo)
            self.assertIs(allowed, False, (fc, acc, q, cr, di, fo))
            self.assertTrue(reasons)

    def test_data_failures_up_to_the_limit(self):
        self.assertEqual(P.apply_decision(95, "pass moves", True, self.ok_cross, self.ok_dir, "ok", True)[0],
                         "pending_audit")


class Layout(unittest.TestCase):
    def test_probs_csv_matches_the_p04_writer(self):
        groups = [("Q1", [0, 1]), ("Q2", [2])]
        Pm = np.array([[0.1] * 10, [np.nan] * 10])
        ok = np.array([True, False])
        b = P.probs_csv_bytes(groups, Pm, ok, np.array([2.0, 0.0]))
        lines = b.decode("utf-8").split("\n")
        self.assertEqual(lines[0], ",".join(rf.PROB_COLUMNS))
        self.assertEqual(lines[1], "Q1,2,2," + ",".join([repr(0.1)] * 10))
        self.assertEqual(lines[2], "Q2,1,0" + "," * 10)
        self.assertNotIn(b"\r", b)

    def test_fuse_probs_nan_rows_and_softmax(self):
        theta = np.zeros(2 + 5 + 9)
        m = lp_rows(3)
        Es = [np.zeros((3, 10))] * 5
        out = P.fuse_probs(theta, m, Es, np.array([12.0, 12.0, 0.0]), np.array([True, True, False]))
        np.testing.assert_allclose(out[0], rf.softmax(m[0:1])[0], rtol=0, atol=1e-15)
        self.assertTrue(np.isnan(out[2]).all())

    def test_leaves_close(self):
        self.assertEqual(P.leaves_close({"a": [1.0, 2.0], "b": "x"}, {"a": [1.0, 2.0 + 1e-13], "b": "x"}), [])
        self.assertTrue(P.leaves_close({"a": 1.0}, {"a": 1.0 + 1e-11}))
        self.assertTrue(P.leaves_close({"a": 1.0}, {"b": 1.0}))


class Modes(unittest.TestCase):
    def ns(self, **kw):
        import argparse
        return argparse.Namespace(**{"precommit": False, "out_dir": None, **kw})

    def test_preview_only_under_tmp(self):
        self.assertTrue(P.resolve(self.ns(out_dir=os.path.join(P.TMP, "x")), "check").preview)
        for bad in (os.path.join(REPO, "results", "prereg07", "rank"), os.path.join(REPO, "data", "elsewhere"),
                    os.environ.get("TEMP", "C:\\")):
            with self.assertRaises(P.P07Error):
                P.resolve(self.ns(out_dir=bad), "check")
        with self.assertRaises(P.P07Error):
            P.resolve(self.ns(out_dir=os.path.join(P.TMP, "x"), precommit=True), "check")
        c = P.resolve(self.ns(), "tune")
        self.assertFalse(c.preview)
        self.assertEqual(c.out_dir, P.OUT)

    def test_paths(self):
        self.assertTrue(P.under_data_local(os.path.join(REPO, "data", "local", "x.csv")))
        self.assertTrue(P.under_data_local(os.path.join("D:" + os.sep, "other", "data", "local")))
        self.assertFalse(P.under_data_local(os.path.join("C:" + os.sep, "Users", "u", "AppData", "Local", "Temp")))
        with self.assertRaises(P.P07Error):
            P.refuse_label_path(os.path.join(REPO, "data", "local", "rank_val_" + "ans" + "wers.csv"))

    def test_constants(self):
        self.assertEqual(len(P.PROFILES), 14)
        self.assertEqual(P.Q02H_TERMS, ["OP", "O4", "SWA", "SWB", "HS"])
        self.assertEqual(P.MONO_FAIL_MAX, 61356 // 200)
        self.assertEqual(P.VAL_FAIL_MAX, P.VAL_GAMES // 200)
        self.assertAlmostEqual(P.RHO_FLOOR_VAL, 3.09 / math.sqrt(1589 - 1), places=3)
        self.assertAlmostEqual(P.RHO_FLOOR_TEST, 3.09 / math.sqrt(400 - 1), places=3)
        self.assertTrue(set(P.INNER_ONLY_FUNCS) >= {"_import_u1", "cmd_check", "cmd_selfcheck", "cmd_tune"})


class Q02SeamUnderGuard(unittest.TestCase):
    """apply's Q02 reproduction needs no data/local file: run it in a subprocess after hs_lib's guard is installed
    (any data/local access would raise) and require byte equality with results/prereg04/rank/rank_val_probs.csv."""

    def test_q02_reproduces_prereg04_bytes(self):
        code = ("import sys, json, builtins; sys.dont_write_bytecode = True; sys.path.insert(0, 'tools'); "
                "import p07_rank_fuse as P; P.import_pilot_fuse(); assert builtins._c10_hs_guard; "
                "q = P.q02_validation(); print(json.dumps(q['seam']))")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        seam = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(seam["passed"], seam)
        self.assertEqual(seam["computed_sha256"], P.P04_PROBS_SHA)

    def test_apply_core_on_synthetic_hs(self):
        """apply_core end to end on synthetic validation HS whose move counts copy val_o4.npz (cross-check 0
        mismatches) and whose sums are random; a zero HS weight must give Q02 back bit for bit."""
        code = r'''
import sys, json, builtins
sys.dont_write_bytecode = True
sys.path.insert(0, "tools")
import numpy as np
import p07_rank_fuse as P
PF = P.import_pilot_fuse()
assert builtins._c10_hs_guard
q = P.q02_validation()
o4 = np.load(P.VAL_O4)
vk = q["vk"]
cnt = o4["n_moves"][:, 0, :].sum(1)
rng = np.random.default_rng(0)
vh = {"question_id": np.array(vk["question_id"]), "slot": np.array(vk["slot"]), "color": np.array(vk["color"]),
      "sums": -rng.uniform(1, 3, size=(vk["n"], 14)) * cnt[:, None], "counts": cnt, "n_moves": o4["moves_parsed"],
      "failed": o4["replay_failed"].astype(bool)}
model = {"mu": rng.normal(-2, 0.2, 14), "sd": rng.uniform(0.1, 0.3, 14), "W": rng.normal(size=(14, 10)),
         "b": rng.normal(size=10)}
fc4 = json.load(open(P.P04_FROZEN, encoding="utf-8"))
th = fc4["params"]["theta"]
theta0 = th[:6] + [0.0] + th[6:]                     # w_HS = 0
out = P.apply_core(PF, q, vh, model, {"params": {"theta": theta0}, "hs_direction_ref": {"rho_eval": 0.9}})
same = bool(np.array_equal(np.nan_to_num(out["P"], nan=-1), np.nan_to_num(q["P"], nan=-1)))
theta1 = th[:6] + [1.0] + th[6:]
out1 = P.apply_core(PF, q, vh, model, {"params": {"theta": theta1}, "hs_direction_ref": {"rho_eval": 0.9}})
print(json.dumps({"same_as_q02_at_w0": same, "csv_equal_at_w0": out["csv_bytes"] == open(P.P04_PROBS, "rb").read(),
                  "mismatches": out["cross"]["mismatches"], "compared": out["cross"]["compared"],
                  "fail_count": out["fail_count"], "differs_at_w1": bool(not np.allclose(
                      np.nan_to_num(out1["P"]), np.nan_to_num(q["P"]))), "path": out["e_info"]["path"]}))
'''
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        res = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(res["same_as_q02_at_w0"], res)
        self.assertEqual(res["mismatches"], 0)
        self.assertGreater(res["compared"], 19000)
        self.assertTrue(res["differs_at_w1"])


@unittest.skipUnless(os.environ.get("C10_SLOW"), "C10_SLOW: end-to-end preview of check -> selfcheck -> tune (~2 min)")
class PreviewEndToEnd(unittest.TestCase):
    def test_preview_reproduces_the_pilot_and_the_gates(self):
        out = os.path.join(P.TMP, f"pytest_preview_{os.getpid()}")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        try:
            for cmd in ("check", "selfcheck", "tune"):
                r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "p07_rank_fuse.py"), cmd, "--out-dir",
                                    out], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
                self.assertEqual(r.returncode, 0, f"{cmd}: {r.stdout[-2000:]} {r.stderr[-2000:]}")
            with open(os.path.join(out, "tune_report.json"), encoding="utf-8") as f:
                rep = json.load(f)
            self.assertTrue(rep["preview"])
            self.assertAlmostEqual(rep["gates"]["G1"]["delta"], P.PILOT["diff"], delta=1e-12)
            self.assertTrue(rep["gates"]["passed"])
            self.assertEqual(rep["record_only"]["new_subset_G4"]["questions"], P.NEW_SUBSET_QUESTIONS)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
