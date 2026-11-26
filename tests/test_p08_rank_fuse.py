"""PREREG_08 tools/p08_rank_fuse.py: unit tests of the pure pieces (masked E_P1 and the x4 scale, the six-term fusion,
probe union / player folds / duplicate exclusion, gates, apply's score_allowed, the audit binding, finalize, tune
rerun, modes). No hs_lib / u1_lib / pilot_* / p1_* import in this process (spec E); the player-fold check runs
p1_fuse in a subprocess.
Run: .venv/Scripts/python.exe -m pytest -q tests/test_p08_rank_fuse.py
"""

import argparse
import builtins
import json
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
import p04_rank_fuse as p4  # noqa: E402
import p07_rank_fuse as P7  # noqa: E402
import p08_rank_fuse as P  # noqa: E402

ENV = {k: v for k, v in dict(os.environ, PYTHONDONTWRITEBYTECODE="1").items() if k not in P.BLAS_ENV}


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process (spec E)"
    for mod in ("p1_lib", "p1_extract", "p1_fuse", "hs_lib", "u1_lib", "pilot_fuse", "pilot_run"):
        assert mod not in sys.modules, f"{mod} imported inside the pytest process (spec E)"


def lp_rows(n, seed=0):
    rng = np.random.default_rng(seed)
    return rf.log_softmax(rng.normal(size=(n, 10)))


class MaskedP1(unittest.TestCase):
    def setUp(self):
        self.groups = [[0, 1, 2], [3, 4], [5], []]
        self.S = rf.pool_matrix(self.groups, 7)
        self.lp = lp_rows(7)

    def test_all_usable_is_pooled_of_4lp_bit_for_bit(self):
        usable = np.array([True] * 6 + [False])          # game 6 is in no question
        E, info = P.p1_term(self.S, self.lp, usable)
        self.assertTrue(np.array_equal(E, rf.pooled(self.S, self.lp * 4.0, "log")))
        self.assertEqual((info["path"], info["scale"]), ("pooled", 4.0))

    def test_scale_applied_once(self):
        usable = np.ones(7, dtype=bool)
        E, _ = P.p1_term(self.S, self.lp, usable)
        E1, _ = P.p1_term(self.S, self.lp, usable, scale=1.0)
        np.testing.assert_allclose(E, 4.0 * E1, rtol=1e-15, atol=1e-15)
        self.assertFalse(np.allclose(E, 16.0 * E1))
        self.assertEqual(P.SCALE, 4.0)

    def test_unusable_games_are_dropped(self):
        usable = np.array([True, False, True, False, False, True, True])
        E, info = P.p1_term(self.S, self.lp, usable)
        self.assertEqual(info["path"], "masked")
        np.testing.assert_allclose(E[0], 4.0 * (self.lp[0] + self.lp[2]) / 2, rtol=0, atol=1e-14)
        self.assertTrue(np.array_equal(E[1], np.zeros(10)))
        self.assertTrue(np.array_equal(E[3], np.zeros(10)))

    def test_forced_mask_equals_pooled_numerically(self):
        usable = np.ones(7, dtype=bool)
        E, _ = P.p1_term(self.S, self.lp, usable)
        Ef, info = P.p1_term(self.S, self.lp, usable, force_mask=True)
        self.assertEqual(info["path"], "masked")
        np.testing.assert_allclose(Ef, E, rtol=0, atol=1e-13)


class Fusion(unittest.TestCase):
    def test_six_terms_equal_the_p04_gen_formula(self):
        rng = np.random.default_rng(4)
        Q = 9
        m = lp_rows(Q, 5)
        Es = [rng.normal(size=(Q, 10)) for _ in P.Q08_TERMS]
        n = rng.integers(1, 20, size=Q).astype(np.float64)
        theta = np.concatenate([[0.3, 0.4], rng.uniform(0, 4, size=6), rng.normal(size=9)])
        ok = np.ones(Q, dtype=bool)
        ok[2] = False
        got = P.fuse_probs(theta, m, Es, n, ok)
        a, g, w, beta = theta[0], theta[1], theta[2:8], theta[8:]
        u = m + p4.B @ beta
        for j in range(6):
            u = u + w[j] * Es[j]
        ref = rf.softmax(np.exp(a) * np.power(n / 12.0, g)[:, None] * u)
        np.testing.assert_allclose(got[ok], ref[ok], rtol=0, atol=1e-14)
        self.assertTrue(np.isnan(got[2]).all())
        self.assertTrue(np.array_equal(np.nan_to_num(got), np.nan_to_num(P7.fuse_probs(theta, m, Es, n, ok))))
        self.assertEqual(p4.gen_params(theta, P.Q08_TERMS)["w"]["P1TO"], float(theta[7]))

    def test_term_order(self):
        self.assertEqual(P.Q08_TERMS, ["OP", "O4", "SWA", "SWB", "HS", "P1TO"])


class ProbeUnion(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.g1 = [{"rank": ["D", "1D"][j % 2], "player_id": f"p{j % 4}", "game_id": f"a{j}"} for j in range(8)]
        self.g2 = [{"rank": ["D", "1D"][j % 2], "player_id": f"p{j % 3}", "game_id": f"b{j}"} for j in range(6)]
        self.X1, self.X2 = rng.normal(size=(8, 3)), rng.normal(size=(6, 3))
        self.u1 = np.array([True, True, False, True, True, True, True, True])
        self.u2 = np.ones(6, dtype=bool)

    def rows(self, dups1=(), dups2=(), with_p2=True):
        parts = [("p1_probe", self.X1, self.u1, np.arange(8), self.g1, list(dups1))]
        if with_p2:
            parts.append(("p2_probe", self.X2, self.u2, np.arange(6), self.g2, list(dups2)))
        return P.probe_rows(parts)

    def test_union_order_labels_players(self):
        r = self.rows(dups1=[4], dups2=[0])
        keep1 = [0, 1, 3, 5, 6, 7]
        keep2 = [1, 2, 3, 4, 5]
        self.assertTrue(np.array_equal(r["X"], np.concatenate([self.X1[keep1], self.X2[keep2]])))
        self.assertEqual(r["source"].tolist(), ["p1_probe"] * 6 + ["p2_probe"] * 5)
        self.assertEqual(r["y"].tolist(), [0 if j % 2 == 0 else 4 for j in keep1] + [0 if j % 2 == 0 else 4
                                                                                    for j in keep2])
        self.assertEqual(r["players"][0], "D:p0")
        self.assertEqual(r["record"]["p1_probe"], {"games_in_chunks": 8, "unusable": 1, "duplicates_excluded": 1,
                                                    "used": 6})
        self.assertEqual(r["record"]["p2_probe"]["duplicates_excluded"], 1)
        # the same player key in both sets.json (a fit player can be in P1 and in P2)
        both = set(r["players"][r["source"] == "p1_probe"].tolist()) & set(r["players"][r["source"] == "p2_probe"]
                                                                           .tolist())
        self.assertIn("1D:p1", both)

    def test_deterministic(self):
        a, b = self.rows(dups1=[4]), self.rows(dups1=[4])
        for k in ("X", "y", "players", "source", "ref"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)
        self.assertEqual(json.dumps(a["record"], sort_keys=True), json.dumps(b["record"], sort_keys=True))

    def test_unusable_duplicate_counted_once(self):
        r = self.rows(dups1=[2], with_p2=False)          # game 2 is unusable anyway
        self.assertEqual(r["record"]["p1_probe"]["duplicates_excluded"], 0)
        self.assertEqual(r["record"]["p1_probe"]["used"], 7)

    def test_player_folds_deterministic_and_shared(self):
        """p1_fuse.player_folds (5 folds by player key, default_rng(0)) in a subprocess: the fold of a player does not
        depend on row order and a key shared by P1 and P2 rows gets one fold."""
        code = r'''
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, "results/prereg08/p1")
import numpy as np
import p1_fuse as F
pl = np.array(["D:p0", "1D:p1", "D:p0", "D:p2", "1D:p1", "D:p3", "D:p0"])
f1 = F.player_folds(pl)
perm = np.random.default_rng(3).permutation(len(pl))
f2 = F.player_folds(pl[perm])
same = all(f2[i] == f1[perm[i]] for i in range(len(pl)))
shared = len({int(f1[i]) for i in range(len(pl)) if pl[i] == "D:p0"}) == 1
print(json.dumps({"same": bool(same), "shared": bool(shared), "again": bool(np.array_equal(f1, F.player_folds(pl)))}))
'''
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=ENV,
                           timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        res = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(res, {"same": True, "shared": True, "again": True})


class Gates(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.n = 600
        self.y = np.repeat(np.arange(10), 60)
        self.player = np.arange(self.n) // 2
        self.boot = rf.PlayerBootstrap(self.player, self.y, self.n // 2)
        self.sc = rng.uniform(0, 1, self.n)

    def mono(self, lo, b=0.01, w=0.01, n=P.MONO_QUESTIONS):
        return {"rank_balanced": 0.02, "ci95": [lo, lo + 0.02], "n_questions": n,
                "by_colour": {"B": {"rank_balanced": b}, "W": {"rank_balanced": w}}}

    def test_g1_boundary(self):
        up = self.sc + 0.05
        self.assertTrue(P.gate_record(0.010 - 1e-12, 0.0, up, self.sc, self.boot, self.mono(0.01))["G1"]["passed"])
        self.assertFalse(P.gate_record(0.010 - 2e-12, 0.0, up, self.sc, self.boot, self.mono(0.01))["G1"]["passed"])
        self.assertTrue(P.gate_record(0.02, 0.0, up, self.sc, self.boot, self.mono(0.01))["passed"])

    def test_g2_lower_bound_zero_fails(self):
        g = P.gate_record(0.02, 0.0, self.sc, self.sc, self.boot, self.mono(0.01))
        self.assertEqual(g["G2"]["ci95"][0], 0.0)
        self.assertFalse(g["G2"]["passed"])
        self.assertFalse(g["passed"])

    def test_g4(self):
        up = self.sc + 0.05
        for m in (self.mono(0.0), self.mono(-0.001), self.mono(0.01, b=0.0), self.mono(0.01, w=-0.01),
                  self.mono(0.01, n=P.MONO_QUESTIONS - 1)):
            g = P.gate_record(0.02, 0.0, up, self.sc, self.boot, m)
            self.assertFalse(g["G4"]["passed"], m)
            self.assertFalse(g["passed"])
        self.assertTrue(P.gate_record(0.02, 0.0, up, self.sc, self.boot, self.mono(1e-9))["G4"]["passed"])

    def test_question_mean_is_record_only(self):
        down = self.sc.copy()
        down[:300] -= 0.2
        down[300:] += 0.3                                       # mixed per-question sign
        g = P.gate_record(0.02, 0.0, self.sc + 0.05, self.sc, self.boot, self.mono(0.01))
        self.assertNotIn("G3", g)
        self.assertIn("question_mean", g)


class ApplyDecision(unittest.TestCase):
    ok_cross = {"mismatches": 0}
    ok_dir = {"passed": True, "rho": 0.9, "threshold": 0.47}

    def test_all_good_needs_the_audit(self):
        for audit in ("missing", "ok", "unbound"):
            self.assertEqual(P.apply_decision(0, None, True, True, self.ok_cross, self.ok_dir, audit, True)[0],
                             "pending_audit")
        self.assertIs(P.apply_decision(0, None, True, True, self.ok_cross, self.ok_dir, "bug", True)[0], False)

    def test_each_blocker(self):
        cases = [(1, None, True, True, self.ok_cross, self.ok_dir, True),
                 (96, "pass moves", True, True, self.ok_cross, self.ok_dir, True),
                 (0, None, False, True, self.ok_cross, self.ok_dir, True),
                 (0, None, True, False, self.ok_cross, self.ok_dir, True),
                 (0, None, True, True, {"mismatches": 1}, self.ok_dir, True),
                 (0, None, True, True, {"mismatches": None}, self.ok_dir, True),
                 (0, None, True, True, self.ok_cross, {"passed": False}, True),
                 (0, None, True, True, self.ok_cross, self.ok_dir, False)]
        for fc, acc, q, h, cr, di, fo in cases:
            allowed, reasons = P.apply_decision(fc, acc, q, h, cr, di, "ok", fo)
            self.assertIs(allowed, False, (fc, acc, q, h, cr, di, fo))
            self.assertTrue(reasons)

    def test_data_failures_up_to_the_limit(self):
        self.assertEqual(P.apply_decision(95, "pass moves", True, True, self.ok_cross, self.ok_dir, "ok", True)[0],
                         "pending_audit")
        self.assertEqual(P.VAL_FAIL_MAX, P.VAL_GAMES // 200)


class Audit(unittest.TestCase):
    rep = {"outputs_sha256": {n: n * 2 for n in P.AUDIT_BOUND_OUTPUTS},
           "inputs_sha256": {n: n * 3 for n in P.AUDIT_BOUND_INPUTS}}

    def audit(self, **kw):
        a = {"format": P.FORMATS["audit"], "program_errors_affecting_Q08": [],
             "outputs_sha256": dict(self.rep["outputs_sha256"]), "inputs_sha256": dict(self.rep["inputs_sha256"])}
        a.update(kw)
        return a

    def test_bound(self):
        self.assertEqual(P.audit_binding(self.audit(), self.rep, "2026-10-08T10:00:00+08:00",
                                         "2026-10-08T09:00:00+08:00"), [])

    def test_unbound_cases(self):
        t_ok = ("2026-10-08T10:00:00+08:00", "2026-10-08T09:00:00+08:00")
        bad_out = dict(self.rep["outputs_sha256"], **{"pred_B.csv": "x"})
        bad_in = dict(self.rep["inputs_sha256"], **{"p1_map.npz": "y"})
        cases = [(self.audit(format="c10-prereg07 prescore_audit v1"), t_ok),
                 (self.audit(outputs_sha256=bad_out), t_ok), (self.audit(inputs_sha256=bad_in), t_ok),
                 (self.audit(program_errors_affecting_Q08=None), t_ok),
                 (self.audit(), ("2026-10-08T09:00:00+08:00", "2026-10-08T09:00:00+08:00")),   # not later
                 (self.audit(), ("2026-10-08T08:00:00+08:00", "2026-10-08T09:00:00+08:00")),   # earlier
                 (self.audit(), (None, "2026-10-08T09:00:00+08:00"))]                          # not committed
        for a, (ta, tp) in cases:
            self.assertTrue(P.audit_binding(a, self.rep, ta, tp), (a.get("format"), ta, tp))

    def test_finalize_only_after_the_audit(self):
        rep = {"score_allowed": "pending_audit", "score_blocked_reasons": ["prescore_audit.json of these outputs not "
                                                                          "committed yet"]}
        with self.assertRaises(P.P08Error):
            P.finalize_decision(rep, "missing", None)
        with self.assertRaises(P.P08Error):
            P.finalize_decision(rep, "unbound", {"binding_problems": ["x"]})
        self.assertEqual(P.finalize_decision(rep, "ok", {}), (True, []))
        self.assertIs(P.finalize_decision(rep, "bug", {})[0], False)
        with self.assertRaises(P.P08Error):
            P.finalize_decision({"score_allowed": True, "score_blocked_reasons": []}, "ok", {})
        with self.assertRaises(P.P08Error):
            P.finalize_decision({"score_allowed": False, "score_blocked_reasons": ["Q02H does not reproduce"]}, "ok",
                                {})


class TuneRerun(unittest.TestCase):
    def lookup(self, t):
        return lambda n: ("abc123", t)

    def test_first_run_and_preview(self):
        self.assertIsNone(P.tune_rerun_check(None, False, None, None, self.lookup("x")))
        self.assertEqual(P.tune_rerun_check({"reason": "r"}, True, None, None, self.lookup("x")), {"reason": "r"})

    def test_real_rerun_needs_a_later_revision(self):
        for rev in (None, 1):
            with self.assertRaises(P.P08Error):
                P.tune_rerun_check({"reason": "r"}, False, rev, "2026-10-07T10:00:00+00:00",
                                   self.lookup("2026-10-07T12:00:00+00:00"))
        with self.assertRaises(P.P08Error):
            P.tune_rerun_check({"reason": "r"}, False, 2, "2026-10-07T10:00:00+00:00",
                               self.lookup("2026-10-07T09:00:00+00:00"))
        r = P.tune_rerun_check({"reason": "r"}, False, 2, "2026-10-07T10:00:00+00:00",
                               self.lookup("2026-10-07T12:00:00+00:00"))
        self.assertEqual(r["revision"]["rev"], 2)

    def test_unknown_revision(self):
        with self.assertRaises(P.P08Error):
            P.revision_committed(97)


class Modes(unittest.TestCase):
    def ns(self, **kw):
        return argparse.Namespace(**{"precommit": False, "out_dir": None, **kw})

    def test_preview_only_under_tmp(self):
        self.assertTrue(P.resolve(self.ns(out_dir=os.path.join(P.TMP, "x")), "check").preview)
        for bad in (os.path.join(REPO, "results", "prereg08", "rank"), os.path.join(REPO, "data", "prereg07", "tmp"),
                    os.environ.get("TEMP", "C:\\")):
            with self.assertRaises(P.P08Error):
                P.resolve(self.ns(out_dir=bad), "check")
        with self.assertRaises(P.P08Error):
            P.resolve(self.ns(out_dir=os.path.join(P.TMP, "x"), precommit=True), "check")
        c = P.resolve(self.ns(), "tune")
        self.assertFalse(c.preview)
        self.assertEqual(c.out_dir, P.OUT)

    def test_preview_record_not_accepted_by_a_later_real_step(self):
        d = os.path.join(P.TMP, f"pytest_modes_{os.getpid()}")
        os.makedirs(d, exist_ok=True)
        try:
            P.write_json(os.path.join(d, "check_report.json"), {"format": P.FORMATS["check"], "preview": True})
            c = P.Ctx()
            c.out_dir, c.preview, c.precommit = d, False, False
            with self.assertRaises(P.P08Error) as cm:
                P.prior_report(c, "check_report.json", P.FORMATS["check"], False)
            self.assertIn("preview", str(cm.exception))
            c.preview = True
            P.write_json(os.path.join(d, "selfcheck.json"), {"format": P.FORMATS["selfcheck"], "preview": False})
            with self.assertRaises(P.P08Error):
                P.prior_report(c, "selfcheck.json", P.FORMATS["selfcheck"], False)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_frozen_usable(self):
        good = {"format": P.FORMATS["frozen"], "preview": False, "score_pending_val": True}
        self.assertTrue(P.frozen_usable(good))
        for bad in (dict(good, preview=True), dict(good, score_pending_val=False),
                    dict(good, format="c10-prereg07 rank frozen_config v1")):
            with self.assertRaises(P.P08Error):
                P.frozen_usable(bad)

    def test_label_path_refused(self):
        with self.assertRaises(P.P08Error):
            P.refuse_label_path(os.path.join(REPO, "data", "local", "rank_val_" + "ans" + "wers.csv"))
        with self.assertRaises(P.P08Error):
            P.sha_raw(os.path.join(REPO, "data", "local", "rank_val_" + "ans" + "wers.csv"))
        self.assertTrue(P.under_data_local(os.path.join(REPO, "data", "local", "x.csv")))

    def test_blas_thread_variables_refused(self):
        old = {k: os.environ.get(k) for k in P.BLAS_ENV + (P.PREVIEW_THREADS_OK,)}
        try:
            for k in old:
                os.environ.pop(k, None)
            self.assertEqual(P.blas_env_check(False), {})
            os.environ["OMP_NUM_THREADS"] = "4"
            with self.assertRaises(P.P08Error):
                P.blas_env_check(False)
            with self.assertRaises(P.P08Error):
                P.blas_env_check(True)
            os.environ[P.PREVIEW_THREADS_OK] = "1"
            self.assertEqual(P.blas_env_check(True), {"OMP_NUM_THREADS": "4"})
            with self.assertRaises(P.P08Error):
                P.blas_env_check(False)                     # real runs never
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_constants(self):
        self.assertEqual(P.LAMBDAS, [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001])
        self.assertEqual((P.GATE, P.TOL, P.P2_FAIL_MAX), (0.010, 1e-12, 600))
        self.assertEqual((P.RHO_FLOOR_VAL, P.RHO_FLOOR_TEST), (0.078, 0.155))
        self.assertEqual({k: v["chunks"] for k, v in P.SETS.items()},
                         {"p1_probe": 150, "bank": 60, "mono": 307, "p2_probe": 600})
        self.assertEqual({k: v["games"] for k, v in P.SETS.items()},
                         {"p1_probe": 30000, "bank": 11813, "mono": 61356, "p2_probe": 120000})
        self.assertTrue(set(P.INNER_ONLY_FUNCS) >= {"_import_u1", "cmd_check", "cmd_selfcheck", "cmd_tune"})

    def test_pins_hold_now(self):
        got = P.check_pins()
        self.assertEqual({k: got[k] for k in P.PINNED}, P.PINNED)
        self.assertEqual(P.check_p07_pins(), P.P07_PINS)


class Q08CoreUnderGuard(unittest.TestCase):
    """apply's Q08 step (q08_core) end to end in a subprocess after hs_lib's guard is installed (no data/local file):
    synthetic validation P1 features whose move counts copy val_hs.npz; with w_P1 = 0 the Q08 csv must equal 07's
    committed rank_val_probs.csv byte for byte, the move cross-check must have 0 mismatches, and w_P1 = 1 must
    change the probabilities."""

    def test_q08_core_on_synthetic_p1(self):
        code = r'''
import sys, json, builtins, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, "tools")
import numpy as np
import p08_rank_fuse as R
P7 = R.P7
PF = R.import_pilot_fuse()
assert builtins._c10_hs_guard
_L1, F = R.import_p1()
q = P7.q02_validation()
vh = dict(np.load(R.P07_VAL_HS))
core = P7.apply_core(PF, q, vh, P7.load_hs_map(R.P07_HS_MAP), json.load(open(R.P07_FROZEN, encoding="utf-8")))
o4z = dict(np.load(P7.VAL_O4))
n = q["vk"]["n"]
rng = np.random.default_rng(0)
tgt = np.asarray(vh["counts"]).astype(np.int64)
tot = np.asarray(vh["n_moves"]).astype(np.int64)
vp = {"means": rng.normal(size=(n, 2, 770)), "counts": np.stack([tgt, tot - tgt], 1), "n_moves": tot,
      "failed": np.asarray(vh["failed"]).astype(bool)}
model = {"mu": rng.normal(size=1540), "sd": rng.uniform(0.5, 2, 1540), "W": rng.normal(size=(1540, 10)) * 0.02,
         "b": rng.normal(size=10) * 0.1}
th = json.load(open(R.P07_FROZEN, encoding="utf-8"))["params"]["theta"]
fc0 = {"params": {"theta": th[:7] + [0.0] + th[7:]}, "p1_direction_ref": {"rho_eval": 0.1}}
fc1 = {"params": {"theta": th[:7] + [1.0] + th[7:]}, "p1_direction_ref": {"rho_eval": 0.1}}
a = R.q08_core(F, q, core, vp, vh, o4z, model, fc0)
b = R.q08_core(F, q, core, vp, vh, o4z, model, fc1)
ref = open(R.P07_PROBS, "rb").read().replace(b"\r\n", b"\n")
print(json.dumps({"csv_equal_07_at_w0": a["csv_bytes"] == ref,
                  "sha_w0": hashlib.sha256(a["csv_bytes"]).hexdigest(), "mismatches": a["cross"]["mismatches"],
                  "compared": a["cross"]["vs_val_o4"]["compared"], "fail_count": a["fail_count"],
                  "usable": int(a["usable"].sum()), "path": a["e_info"]["path"], "scale": a["e_info"]["scale"],
                  "differs_at_w1": bool(not np.allclose(np.nan_to_num(b["P"]), np.nan_to_num(a["P"]))),
                  "rho_finite": bool(np.isfinite(a["direction"]["rho"]))}))
'''
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=ENV,
                           timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        res = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(res["csv_equal_07_at_w0"], res)
        self.assertEqual(res["sha_w0"], P.P07_PROBS_SHA)
        self.assertEqual(res["mismatches"], 0)
        self.assertGreater(res["compared"], 19000)
        self.assertEqual(res["scale"], 4.0)
        self.assertTrue(res["differs_at_w1"])
        self.assertTrue(res["rho_finite"])


class ChunkMeta(unittest.TestCase):
    def test_meta_problems_on_a_synthetic_chunk(self):
        d = os.path.join(P.TMP, f"pytest_chunk_{os.getpid()}")
        os.makedirs(d, exist_ok=True)
        try:
            p = os.path.join(d, "chunk_00000.npz")
            code = json.dumps({"results/prereg08/p1/p1_lib.py": P.PINNED["results/prereg08/p1/p1_lib.py"],
                               "results/prereg08/p1/p1_extract.py": P.PINNED["results/prereg08/p1/p1_extract.py"]})
            fields = dict(format=np.array(P.CHUNK_FORMAT), set=np.array("probe"), sides=np.array(["target", "opponent"]),
                          ref=np.arange(3), precision=np.array("fp16"), batch=np.array(1024),
                          channels_last=np.array(True), device=np.array("cuda"), torch=np.array("2.11.0+cu128"),
                          ckpt_sha256=np.array("not hashed in this run"), katago_commit=np.array(P.KATAGO_COMMIT),
                          meta_sha256=np.array(P.NEUTRAL_META_SHA), code_sha256=np.array(code),
                          created_utc=np.array("2026-10-06T00:00:00+00:00"), means=np.zeros((3, 2, 770), np.float32),
                          counts=np.array([[3, 2], [0, 4], [5, 5]]), n_moves=np.array([5, 4, 10]),
                          failed=np.array([False, False, True]))
            np.savez(p, **fields)
            rec, ref, failed, cnt, nm = P.chunk_meta(p)
            self.assertEqual(P.chunk_meta_problems(rec, "probe"), [])
            self.assertEqual((rec["failed"], rec["no_target_moves"], rec["games"]), (1, 1, 3))
            self.assertEqual(P.chunk_meta_problems(rec, "bank"), ["set"])
            np.savez(p, **dict(fields, precision=np.array("fp32"), meta_sha256=np.array("0" * 64)))
            self.assertEqual(P.chunk_meta_problems(P.chunk_meta(p)[0], "probe"), ["meta_sha256", "precision"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
