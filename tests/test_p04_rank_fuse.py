"""Tests for tools/p04_rank_fuse.py (PREREG_04 rev 1 sections 2.1-2.6, competition 2).
Run: .venv/Scripts/python.exe -m pytest tests/test_p04_rank_fuse.py

Unit tests of the generalised calibration (ported verbatim from the design's fuse.py; finite-difference gradient),
selection, gates G1-G4, the colour rules and the path rules; end-to-end tests on the synthetic prereg02 / prereg03
world of tests/test_p03_rank_fuse.py plus synthetic p04_opp.py outputs, in a temporary directory outside the
repository. No real data, no label file, no GPU. The real-data reproduction (Q00 = R20, Q01 / Q02 = the design
values) is 'selfcheck' itself.
"""

import builtins
import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import decide  # noqa: E402
import p02_rank_fuse as rf  # noqa: E402
import p02_rank_pseudo as pseudo  # noqa: E402
import p03_rank_fuse as p03  # noqa: E402
import p04_rank_fuse as p04  # noqa: E402
from test_p03_rank_fuse import P03World, quiet  # noqa: E402

DESIGN_FUSE = os.path.join(REPO, "results", "prereg04", "design", "evidence", "B_proxy", "fuse.py")


def design_fuse():
    """The design-phase fuse.py as a module (its main() is not run), without writing bytecode into the committed
    design folder (fuse.py sets sys.dont_write_bytecode itself, but only after the loader cached it)."""
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("p04_design_fuse", DESIGN_FUSE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def sha(path):
    return pseudo.sha256_file(path)


# ================================================================ generalised calibration
def random_problem(seed, Q=300, J=2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, size=Q)
    k = np.arange(10)
    m = -0.4 * np.abs(k[None] - y[:, None]) + rng.normal(0, 1, size=(Q, 10))
    m = rf.log_softmax(m)
    Es = [rf.log_softmax(-0.3 * np.abs(k[None] - y[:, None]) + rng.normal(0, 1, size=(Q, 10))) for _ in range(J)]
    n = rng.integers(5, 21, size=Q).astype(np.float64)
    return m, Es, n, y


class GenCalibration(unittest.TestCase):
    def test_port_is_verbatim(self):
        F = design_fuse()
        for name in ("_unpack", "gen_logits", "gen_obj", "gen_oof"):
            self.assertEqual(inspect.getsource(getattr(p04, name)), inspect.getsource(getattr(F, name)), name)
        # gen_fit only adds the optimiser record: the same fit, bit for bit
        for J in (0, 1, 3):
            m, Es, n, y = random_problem(J, J=J)
            self.assertTrue(np.array_equal(p04.gen_fit(m, Es, n, y), F.gen_fit(m, Es, n, y)), J)
            th, rec = p04.gen_fit_record(m, Es, n, y)
            self.assertTrue(np.array_equal(th, F.gen_fit(m, Es, n, y)))
            wq = rf.level_weights(y)
            self.assertEqual(rec["loss"], float(p04.gen_obj(th, m, Es, n, y, wq)[0]))
            self.assertEqual(len(rec["starts"]), len(rf.STARTS))
        self.assertTrue(np.array_equal(p04.B, F.B))
        self.assertEqual((rf.PENALTY, rf.MAXITER, rf.STARTS, rf.BOUNDS["w"]), (0.01, 500, (0.0, 1.0), (0.0, 4.0)))

    def test_gradient_finite_difference(self):
        for J in (0, 1, 3):
            m, Es, n, y = random_problem(10 + J, Q=120, J=J)
            wq = rf.level_weights(y)
            rng = np.random.default_rng(J)
            th = np.concatenate([[0.7, 0.3], rng.uniform(0.1, 1.5, size=J), rng.normal(0, 0.3, size=9)])
            _, g = p04.gen_obj(th, m, Es, n, y, wq)
            num = np.zeros_like(th)
            h = 1e-6
            for i in range(len(th)):
                e = np.zeros_like(th)
                e[i] = h
                num[i] = (p04.gen_obj(th + e, m, Es, n, y, wq)[0] - p04.gen_obj(th - e, m, Es, n, y, wq)[0]) / (2 * h)
            np.testing.assert_allclose(g, num, rtol=1e-6, atol=1e-8, err_msg=f"J={J}")

    def test_one_term_is_p02s_objective(self):
        m, Es, n, y = random_problem(3, J=1)
        wq = rf.level_weights(y)
        th = np.array([0.9, 0.2, 0.4] + [0.05 * i for i in range(9)])
        l1, g1 = p04.gen_obj(th, m, Es, n, y, wq)
        l2, g2 = rf.objective(th, m, Es[0], n, y, wq, True, rf.basis())
        self.assertLess(abs(l1 - l2), 1e-12 * max(1.0, abs(l2)))
        np.testing.assert_allclose(g1, g2, rtol=1e-10, atol=1e-12)
        z1, _ = p04.gen_logits(th, m, Es, n)
        z2, _ = rf.question_logits(th, m, Es[0], n, True, rf.basis())
        np.testing.assert_allclose(z1, z2, rtol=0, atol=1e-13)

    def qsets(self, seed=0, J=2):
        rng = np.random.default_rng(seed)
        Xf, Xe = [], []
        for Q in (400, 500):
            y = rng.integers(0, 10, size=Q)
            X = rf.QSet(rng.integers(5, 21, size=Q), y, rng.integers(0, 50, size=Q), rng.integers(0, 5, size=Q),
                        rf.log_softmax(rng.normal(size=(Q, 10))))
            (Xf if not Xf else Xe).append(X)
        m, Es, _n, _y = random_problem(seed + 1, Q=400, J=J)
        me, Ee, _n, _y = random_problem(seed + 2, Q=500, J=J)
        return Xf[0], Xe[0], m, me, Es, Ee

    def test_oof_record_equals_oof(self):
        Xf, Xe, mf, me, Ef, Ee = self.qsets()
        P, ws = p04.gen_oof(Xf, Xe, mf, me, Ef, Ee)
        P2, folds = p04.gen_oof_record(Xf, Xe, mf, me, Ef, Ee)
        self.assertTrue(np.array_equal(P, P2))
        self.assertEqual(ws, [f["theta"][2:4] for f in folds])
        self.assertTrue(np.array_equal(p04.gen_fold_predict(folds, me, Ee, Xe.n, Xe.fold), P))
        self.assertEqual([f["fold"] for f in folds], list(range(5)))

    def test_params_layout(self):
        th = np.array([0.5, -0.2, 0.1, 0.7] + [0.01 * i for i in range(9)])
        par = p04.gen_params(th, ["OP", "O4"])
        self.assertEqual(par["w"], {"OP": 0.1, "O4": 0.7})
        self.assertEqual(par["theta"], [float(x) for x in th])
        np.testing.assert_allclose(par["b"], p04.B @ th[4:], atol=0)
        th1 = th[[0, 1, 2] + list(range(4, 13))]
        q = p04.p02_params(th1)
        self.assertEqual(p04.as_p02_dict(q), rf.params_dict(th1, True, rf.basis()))
        self.assertEqual(p03.theta_of(p04.as_p02_dict(q), True).tolist(), q["theta"])


# ================================================================ selection and gates
class Selection(unittest.TestCase):
    def res(self, vals):
        return [{"id": cid, "status": "dropped" if v is None else "ok", "oof": None if v is None else
                 {"rank_balanced": v}} for (cid, _), v in zip(p04.COMBOS, vals)]

    def test_order_and_tolerance(self):
        self.assertEqual([c for c, _ in p04.COMBOS], ["Q00", "Q01", "Q02"])
        self.assertEqual(dict(p04.COMBOS)["Q02"], ("C5", "OP", "O4", "SWA", "SWB"))
        chosen, rec = p04.select(self.res([0.5855, 0.6042, 0.6101]))
        self.assertEqual(chosen["id"], "Q02")
        chosen, rec = p04.select(self.res([0.5855, 0.6042, 0.6071]))
        self.assertEqual(chosen["id"], "Q01")               # within 0.003 -> the earlier one
        self.assertEqual(rec["best_id"], "Q02")
        chosen, _ = p04.select(self.res([0.600, 0.603, None]))
        self.assertEqual(chosen["id"], "Q00")               # exactly 0.003 (float slack) -> Q00
        chosen, _ = p04.select(self.res([0.600, None, None]))
        self.assertEqual(chosen["id"], "Q00")

    def world(self, seed, shift, colour_bias=0.0):
        rng = np.random.default_rng(seed)
        level_of = np.repeat(np.arange(10), 6)
        player = np.repeat(np.arange(60), 20)
        y = level_of[player]
        q00 = rng.choice([0.0, math.exp(-1), 1.0], size=len(player), p=[0.3, 0.3, 0.4])
        new = q00.copy()
        new[rng.random(len(player)) < shift] = 1.0
        return player, y, q00, new

    def rb(self, sc, y):
        return float(np.mean([sc[y == i].mean() for i in range(10)]))

    def mono(self, player, y, new, q00, colour):
        ms = {"player": player, "y": y, "n_players": 60, "colour": colour, "k": np.full(len(y), 5)}
        return p03.mono_diff_record(ms, rf.PlayerBootstrap(player, y, 60), new, q00)

    def test_gates(self):
        player, y, q00, new = self.world(1, 0.15)
        colour = np.where(np.arange(len(y)) % 2 == 0, "B", "W")
        boot = rf.PlayerBootstrap(player, y, 60)
        md = self.mono(player, y, new, q00, colour)
        g = p04.gate_record("Q02", self.rb(new, y), self.rb(q00, y), new, q00, boot, md)
        self.assertTrue(all(g[k]["passed"] for k in ("G1", "G2", "G3", "G4")) and g["passed"])
        self.assertEqual(g["G4"]["ci95"], md["ci95"])
        # G4: the white point estimate not > 0 fails the gate even when the pooled CI is above 0
        md2 = json.loads(json.dumps(md))
        md2["by_colour"]["W"]["rank_balanced"] = 0.0
        g = p04.gate_record("Q02", self.rb(new, y), self.rb(q00, y), new, q00, boot, md2)
        self.assertFalse(g["G4"]["passed"])
        self.assertFalse(g["passed"])
        md3 = json.loads(json.dumps(md))
        md3["ci95"][0] = 0.0
        self.assertFalse(p04.gate_record("Q02", self.rb(new, y), self.rb(q00, y), new, q00, boot, md3)["passed"])
        # G1 below +0.010
        player, y, q00, small = self.world(1, 0.004)
        g = p04.gate_record("Q01", self.rb(small, y), self.rb(q00, y), small, q00, rf.PlayerBootstrap(player, y, 60),
                            self.mono(player, y, small, q00, colour))
        self.assertFalse(g["G1"]["passed"])
        self.assertFalse(g["passed"])
        # G1 float slack as p03
        g = p04.gate_record("Q01", 0.5 + 0.010 - 5e-13, 0.5, new, q00, boot, md)
        self.assertTrue(g["G1"]["passed"])
        g = p04.gate_record("Q01", 0.5 + 0.010 - 5e-12, 0.5, new, q00, boot, md)
        self.assertFalse(g["G1"]["passed"])
        g = p04.gate_record("Q00", 0.6, 0.6, q00, q00, boot, None)
        self.assertFalse(g["passed"])
        self.assertIsNone(g["G4"])


# ================================================================ O4 files and colour rules
class SideVectors(unittest.TestCase):
    def test_layouts(self):
        rng = np.random.default_rng(0)
        ll = rng.normal(size=(7, 3, 3, 10))
        z = {"ll": ll, "variants": np.array(["t_choice_a20", "x", "o_choice_a20"])}
        o, t, lay = p04.side_vectors(z, "f")
        self.assertTrue(np.array_equal(o, ll[:, 2].sum(1)) and np.array_equal(t, ll[:, 0].sum(1)))
        self.assertIn("design", lay)
        o2, t2, lay2 = p04.side_vectors({"o4": o, "t4": t}, "f")
        self.assertTrue(np.array_equal(o2, o) and np.array_equal(t2, t))
        with self.assertRaises(p04.P04Error):
            p04.side_vectors({"o4": o.astype(np.float32), "t4": t}, "f")
        with self.assertRaises(p04.P04Error):
            p04.side_vectors({"o4": o}, "f")
        with self.assertRaises(p04.P04Error):
            p04.side_vectors({"ll": ll, "variants": np.array(["a", "b", "c"])}, "f")


class ColourRules(unittest.TestCase):
    def test_usable(self):
        rec = {"inner": {"ci95": [0.01, 0.03]}, "val_sub_questions": {"dkappa": 0.02}}
        self.assertEqual(p04.colour_usable("SWA", rec), [])
        self.assertEqual(p04.colour_usable("O4", {"inner": {"ci95": [0.01, 0.03]}, "val_sub_questions": "deferred"}),
                         [])
        self.assertEqual(len(p04.colour_usable("SWB", dict(rec, val_sub_questions={"dkappa": 0.0}))), 1)
        self.assertEqual(len(p04.colour_usable("SWB", dict(rec, val_sub_questions=None))), 1)
        self.assertEqual(len(p04.colour_usable("O4", {"inner": {"ci95": [0.0, 0.03]}})), 1)
        self.assertEqual(len(p04.colour_usable("SWA", {"inner": {"ci95": [-0.01, 0.03]},
                                                       "val_sub_questions": {"dkappa": -1}})), 2)

    def test_pairs(self):
        self.assertEqual(p04.PAIRS, (("O4", "T4"), ("SWA", "A0"), ("SWB", "B0")))
        self.assertEqual(p04.TERMS, ("OP", "O4", "SWA", "SWB"))
        self.assertEqual(p04.SW_SHA256["a"]["inner"][:8], "74382520")
        self.assertEqual(p04.MONO_SHA256[:8], "45d0f14a")
        self.assertEqual(p04.DESIGN_OOF, {"Q01": 0.6042484782462081, "Q02": 0.6101211982700248})
        self.assertEqual(p04.R20_INNER_OOF, 0.5854702521055817)
        self.assertEqual(p04.DUP_STOP_FRACTION, 0.01)
        self.assertEqual(p04.FAIL_OPENING["policy"][:18], "04 does not play; ")


class PinnedNumbers(unittest.TestCase):
    """Real-mode O4 pinned numbers and dupcheck (PREREG_04 1.2, 2.2 items 1 and 3), p04_opp.py's report layout."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_p04_pins_")
        self.c = types.SimpleNamespace(smoke=False, opp_dir=self.tmp)
        self.saved = (p04.COUNTS_NPZ, p04.DESIGN_COUNTS, p04.DESIGN_PROXY)
        # the large files are "not on this machine": the reports' own records are used
        p04.COUNTS_NPZ = p04.DESIGN_COUNTS = p04.DESIGN_PROXY = os.path.join(self.tmp, "absent.npz")

    def tearDown(self):
        p04.COUNTS_NPZ, p04.DESIGN_COUNTS, p04.DESIGN_PROXY = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, obj):
        with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def count_report(self, **kw):
        rep = dict(p04.O4_PINNED, status="ok", preview=False, counts_sha256="ab" * 32,
                   design_counts={"arrays_equal": {k: True for k in ("C_t", "A_t", "C_o", "A_o")}})
        rep.update(kw)
        self.write(p04.COUNT_REPORT, rep)

    def inner_report(self, diff=0.0):
        self.write(p04.O4_INNER_REPORT, {"status": "ok", "preview": False, "design_proxy": {"compare": {
            "o_choice_a20": {"max_abs_diff": diff}, "t_choice_a20": {"max_abs_diff": 0.0}}}})

    def test_count_report(self):
        self.count_report()
        self.inner_report()
        probs, rec = p04.o4_pinned(self.c, {"O4": None, "T4": None, "keys": None})
        self.assertEqual(probs, [])
        self.assertIn("count_report.json", rec["counts_npz"]["via"])
        self.count_report(opponent_moves=79985353)
        self.assertEqual(len(p04.o4_pinned(self.c, None)[0]), 1)
        self.count_report(preview=True)
        self.assertEqual(len(p04.o4_pinned(self.c, None)[0]), 1)
        self.count_report(status="stop")
        self.assertEqual(len(p04.o4_pinned(self.c, None)[0]), 1)
        self.count_report(design_counts={"arrays_equal": {"C_t": True, "A_t": True, "C_o": False, "A_o": True}})
        self.assertEqual(len(p04.o4_pinned(self.c, None)[0]), 1)
        self.count_report(design_counts={"state": "absent"})
        self.assertEqual(len(p04.o4_pinned(self.c, None)[0]), 1)
        self.count_report()
        self.inner_report(diff=2e-12)
        self.assertEqual(len(p04.o4_pinned(self.c, {"O4": None, "T4": None, "keys": None})[0]), 1)
        os.remove(os.path.join(self.tmp, p04.COUNT_REPORT))
        self.assertEqual(len(p04.o4_pinned(self.c, None)[0]), 1)

    def test_dupcheck(self):
        def dup(hits, slots=p04.VAL_GAMES, preview=False, stop=None):
            share = hits / slots
            self.write(p04.DUPCHECK_REPORT, {"status": "stop" if share >= 0.01 else "ok", "preview": preview,
                                             "validation_vs_fit": {"slots": slots, "slots_in_fit_any": hits,
                                                                   "share_of_slots": share,
                                                                   "stop": share >= 0.01 if stop is None else stop}})
            return p04.dupcheck_status(self.c)
        probs, rec = dup(0)
        self.assertEqual(probs, [])
        probs, rec = dup(191)                    # 0.997 %: recorded and disclosed, not a stop
        self.assertEqual(probs, [])
        self.assertIn("disclose", rec)
        self.assertGreaterEqual(len(dup(192)[0]), 1)          # 1.002 %: stop
        self.assertEqual(len(dup(0, slots=19000)[0]), 1)      # not the validation slot count
        self.assertEqual(len(dup(0, preview=True)[0]), 1)
        self.assertEqual(len(dup(0, stop=True)[0]), 1)
        os.remove(os.path.join(self.tmp, p04.DUPCHECK_REPORT))
        self.assertEqual(len(p04.dupcheck_status(self.c)[0]), 1)
        # the colour split of p04_opp.py's report is carried into the check record
        self.write(p04.DUPCHECK_REPORT, {"status": "ok", "preview": False, "validation_vs_fit": {
            "slots": p04.VAL_GAMES, "slots_in_fit_any": 3, "slots_in_fit_same_target_colour": 1,
            "slots_in_fit_other_target_colour": 2, "share_of_slots": 3 / p04.VAL_GAMES, "stop": False}})
        probs, rec = p04.dupcheck_status(self.c)
        self.assertEqual(probs, [])
        self.assertEqual((rec["slots_in_fit_same_target_colour"], rec["slots_in_fit_other_target_colour"]), (1, 2))

    def test_selftest(self):
        good = {"format": p04.SELFTEST_FORMAT, "passed": True, "problems": [], "preview": False,
                "capture": {"games": 150, "games_with_mismatch": 0, "anomalies": 0},
                "features": {"positions": 379, "mismatches": 0}}
        self.write(p04.SELFTEST_REPORT, good)
        probs, rec = p04.selftest_status(self.c)
        self.assertEqual(probs, [])
        self.assertEqual((rec["passed"], rec["feature_positions"]), (True, 379))
        for bad in (dict(good, passed=False), dict(good, problems=["features: 1 mismatches on 379 positions"]),
                    dict(good, preview=True), {k: v for k, v in good.items() if k != "preview"},
                    dict(good, format="c10-prereg04 opp count v1"),
                    dict(good, capture={"games_with_mismatch": 1}), dict(good, features={"mismatches": 2})):
            self.write(p04.SELFTEST_REPORT, bad)
            self.assertGreaterEqual(len(p04.selftest_status(self.c)[0]), 1, bad)
        os.remove(os.path.join(self.tmp, p04.SELFTEST_REPORT))
        probs, rec = p04.selftest_status(self.c)
        self.assertEqual(len(probs), 1)
        self.assertFalse(rec["present"])

    def test_committed_inputs_include_selftest(self):
        seen = []

        def fake(paths, what):
            seen.extend(paths)
            return {"files": len(paths)}
        old = p03.require_tracked_clean
        p03.require_tracked_clean = fake
        try:
            c = types.SimpleNamespace(smoke=False, preview=False, opp_dir=self.tmp,
                                      arm_dir={x: os.path.join(self.tmp, x) for x in p04.ARMS})
            p04.committed_inputs(c)
        finally:
            p03.require_tracked_clean = old
        names = {os.path.basename(p) for p in seen}
        self.assertTrue({p04.O4_INNER, p04.O4_INNER_REPORT, p04.COUNT_REPORT, p04.DUPCHECK_REPORT,
                         p04.SELFTEST_REPORT} <= names, names)


class FakeGit:
    """Stands in for p03.git: ls-files -> ls_rc; status -> (status_rc, status_text); anything else -> (0, '')."""

    def __init__(self, ls_rc=0, status_rc=0, status_text=""):
        self.ls_rc, self.status_rc, self.status_text = ls_rc, status_rc, status_text
        self.calls = []

    def __call__(self, args, binary=False, repo=None):
        self.calls.append(list(args))
        if args[0] == "ls-files":
            return self.ls_rc, (b"" if binary else "")
        if args[0] == "status":
            return self.status_rc, (self.status_text.encode() if binary else self.status_text)
        return 0, (b"" if binary else "")


class OrderGuard(unittest.TestCase):
    """PREREG_04 1.2 second line of defence in the fusion tool: a validation O4 next to an uncommitted frozen config."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_p04_guard_")
        self.old_git = p03.git
        self.c = types.SimpleNamespace(smoke=False, opp_dir=self.tmp)

    def tearDown(self):
        p03.git = self.old_git
        shutil.rmtree(self.tmp, ignore_errors=True)

    def guard(self, **kw):
        p03.git = FakeGit(**kw)
        return p04.val_o4_order_guard(self.c)

    def test_guard(self):
        self.assertIsNone(self.guard(ls_rc=1))                       # no validation O4 file: nothing to guard
        open(os.path.join(self.tmp, p04.O4_VAL), "wb").close()
        msg = self.guard(ls_rc=1)
        self.assertIn("not committed", msg)
        self.assertIn("PREREG_04 1.2", msg)
        self.assertIn("changed in the working tree or only staged", self.guard(ls_rc=0, status_text="A  x"))
        self.assertIn("changed in the working tree or only staged", self.guard(ls_rc=0, status_text=" M x"))
        self.assertIn("not verifiable", self.guard(ls_rc=0, status_rc=128))
        self.assertIsNone(self.guard(ls_rc=0, status_text=""))       # committed and unchanged
        fg = p03.git
        self.assertTrue(any(c[0] == "ls-files" and c[-1] == "results/prereg04/rank/frozen_config.json"
                            for c in fg.calls), fg.calls)
        self.c.smoke = True
        self.assertIsNone(self.guard(ls_rc=1))                       # smoke worlds are exempt

    def test_tune_stops_on_the_guard(self):
        saved = {n: getattr(p04, n) for n in ("start", "tool_state", "require_tools_committed", "val_o4_order_guard",
                                              "require_order")}
        reached = []
        try:
            p04.start = lambda c: {}
            p04.tool_state = lambda: {"git_head": "x", "dirty_or_untracked": [], "files_sha256": {}}
            p04.require_tools_committed = lambda t: None
            p04.val_o4_order_guard = lambda c: "GUARD: validation O4 before the frozen config"
            p04.require_order = lambda c, names: reached.append("order")
            p03.git = FakeGit(ls_rc=1)                               # frozen_config.json not tracked yet
            with self.assertRaisesRegex(p04.P04Error, "GUARD"):
                p04.cmd_tune(args("tune"))
            self.assertEqual(reached, [])                            # stopped before reading check / selfcheck
            self.assertEqual(quiet(p04.main, ["tune"]), 1)
        finally:
            for n, f in saved.items():
                setattr(p04, n, f)

    def test_check_records_the_guard_as_a_stop(self):
        src = inspect.getsource(p04.cmd_check)
        self.assertIn("guard = val_o4_order_guard(c)", src)
        self.assertIn("stops.append(guard)", src)


class ApplyBindings(unittest.TestCase):
    """apply reads the check report and the validation O4 that tune / p04_opp.py val made for this frozen config."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_p04_bind_")
        self.crp = os.path.join(self.tmp, "check_report.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_check(self, **kw):
        pseudo.write_json(self.crp, dict({"format": p04.CHECK_FORMAT, "status": "ok", "smoke": True, "preview": False,
                                          "design_input": False}, **kw))
        return sha(self.crp)

    def test_check_report_binding(self):
        smoke = types.SimpleNamespace(smoke=True)
        s = self.write_check()
        cr, crsha, commit = p04.check_report_binding(smoke, {"check": {"check_report_sha256": s}}, self.crp)
        self.assertEqual((crsha, commit, cr["status"]), (s, None, "ok"))
        with self.assertRaisesRegex(p04.P04Error, "differs from the one tune recorded"):
            p04.check_report_binding(smoke, {"check": {"check_report_sha256": "0" * 64}}, self.crp)
        with self.assertRaisesRegex(p04.P04Error, "differs from the one tune recorded"):
            p04.check_report_binding(smoke, {}, self.crp)
        for bad in ({"status": "stop"}, {"preview": True}, {"design_input": True}, {"smoke": False},
                    {"format": p04.TUNE_FORMAT}):
            s = self.write_check(**bad)
            with self.assertRaisesRegex(p04.P04Error, "not a passed check"):
                p04.check_report_binding(smoke, {"check": {"check_report_sha256": s}}, self.crp)
        # real mode: the file must be committed in the repository (this one is outside it)
        s = self.write_check(smoke=False)
        with self.assertRaises(p03.P03Error):
            p04.check_report_binding(types.SimpleNamespace(smoke=False), {"check": {"check_report_sha256": s}},
                                     self.crp)

    def test_val_report_binding(self):
        real, smoke = types.SimpleNamespace(smoke=False), types.SimpleNamespace(smoke=True)
        fc = {"source_sha256": {"O4": {"tables_sha256": "t" * 64}}, "tool": {"files_sha256": {"p04_opp.py": "p" * 64}}}
        rep = {"frozen_config_sha256": "f" * 64, "p04_opp_sha256": "p" * 64, "tables_sha256": "t" * 64}
        rec = p04.val_report_binding(real, fc, {"report": rep}, "f" * 64)
        self.assertTrue(all(v["equal"] for v in rec.values()))
        for key, val in (("frozen_config_sha256", "e" * 64), ("p04_opp_sha256", "q" * 64), ("tables_sha256", None),
                         ("frozen_config_sha256", None)):
            with self.assertRaisesRegex(p04.P04Error, key):
                p04.val_report_binding(real, fc, {"report": dict(rep, **{key: val})}, "f" * 64)
        with self.assertRaisesRegex(p04.P04Error, "tables_sha256"):          # nothing frozen: refused in real mode
            p04.val_report_binding(real, {"tool": fc["tool"]}, {"report": dict(rep, tables_sha256=None)}, "f" * 64)
        with self.assertRaisesRegex(p04.P04Error, "frozen_config_sha256"):   # another frozen config
            p04.val_report_binding(real, fc, {"report": rep}, "g" * 64)
        # smoke worlds have no frozen tables: nothing to compare; when frozen, they must match
        self.assertIsInstance(p04.val_report_binding(smoke, {}, {"report": {}}, "x"), dict)
        with self.assertRaises(p04.P04Error):
            p04.val_report_binding(smoke, fc, {"report": dict(rep, tables_sha256="u" * 64)}, "x")

    def test_opp_report_needs_tables_in_real_mode(self):
        path = os.path.join(self.tmp, p04.O4_INNER_REPORT)
        pseudo.write_json(path, {"status": "ok", "preview": False, "output_sha256": "o" * 64,
                                 "tool": {"files_sha256": {"p04_opp.py": "p" * 64}}})
        c = types.SimpleNamespace(smoke=False, opp_dir=self.tmp)
        probs, rec = p04.opp_report(c, p04.O4_INNER_REPORT, "o" * 64)
        self.assertEqual(len(probs), 1)
        self.assertIn("tables_sha256", probs[0])
        self.assertEqual(rec["p04_opp_sha256"], "p" * 64)
        pseudo.write_json(path, {"status": "ok", "preview": False, "output_sha256": "o" * 64,
                                 "tables_sha256": "ab" * 32, "frozen_config": {"sha256": "cd" * 32}})
        probs, rec = p04.opp_report(c, p04.O4_INNER_REPORT, "o" * 64)
        self.assertEqual(probs, [])
        self.assertEqual((rec["tables_sha256"], rec["frozen_config_sha256"]), ("ab" * 32, "cd" * 32))
        self.assertEqual(p04.opp_report(types.SimpleNamespace(smoke=True, opp_dir=self.tmp), p04.O4_INNER_REPORT,
                                        "o" * 64)[0], [])


# ================================================================ paths and modes
def args(cmd, **kw):
    base = {"command": cmd, "p02_dir": None, "p02_local_dir": None, "p03_dir": None, "opp_dir": None, "out_dir": None,
            "split": None, "questions": None, "design_o4": False, "smoke": False}
    base.update(kw)
    return types.SimpleNamespace(**base)


class Modes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_p04_modes_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_refusals(self):
        with self.assertRaises(p04.P04Error):
            p04.resolve(args("check", smoke=True, design_o4=True), "check")
        with self.assertRaises(p04.P04Error):      # design input writes a preview only
            p04.resolve(args("check", design_o4=True), "check")
        with self.assertRaises(p04.P04Error):      # apply has no preview
            p04.resolve(args("apply", out_dir=self.tmp), "apply")
        with self.assertRaises(p04.P04Error):      # design + apply
            p04.resolve(args("apply", design_o4=True, out_dir=self.tmp), "apply")
        with self.assertRaises(p04.P04Error):      # real mode takes the repository paths only
            p04.resolve(args("check", opp_dir=self.tmp), "check")
        with self.assertRaises(p04.P04Error):      # inside the repository only results/prereg04/rank
            p04.resolve(args("check", out_dir=os.path.join(REPO, "results", "prereg04", "opp")), "check")
        with self.assertRaises(p04.P04Error):
            p04.resolve(args("check", out_dir=os.path.join(REPO, "results", "prereg03", "rank")), "check")
        with self.assertRaises(p04.P04Error):      # smoke needs every path
            p04.resolve(args("check", smoke=True, out_dir=self.tmp), "check")
        with self.assertRaises(p04.P04Error):      # smoke output outside the repository
            p04.resolve(args("check", smoke=True, out_dir=os.path.join(REPO, "x"), p02_dir=self.tmp,
                             p02_local_dir=self.tmp, p03_dir=self.tmp, opp_dir=self.tmp, split=self.tmp), "check")
        with self.assertRaises(p04.P04Error):      # smoke inputs outside the repository
            p04.resolve(args("check", smoke=True, out_dir=self.tmp, p02_dir=os.path.join(REPO, "results", "prereg02",
                                                                                         "rank"),
                             p02_local_dir=self.tmp, p03_dir=self.tmp, opp_dir=self.tmp, split=self.tmp), "check")

    def test_real_preview_flags(self):
        c = p04.resolve(args("check", out_dir=self.tmp, design_o4=True), "check")
        self.assertTrue(c.preview and c.design and not c.smoke)
        self.assertEqual(p04.mode_flags(c), {"smoke": False, "preview": True, "design_input": True})
        c = p04.resolve(args("tune"), "tune")
        self.assertFalse(c.preview)
        self.assertTrue(p03.same_path(c.out_dir, p04.P04_OUT))


class FrozenTools(unittest.TestCase):
    def test_files_and_head(self):
        files = p04.frozen_files()
        for f in ("p02_rank_fuse.py", "p02_rank_pseudo.py", "p02_rank_opening.py", "p02_moves.py",
                  "p02_player_fuse.py", "p02_player_open.py", "p03_rank_fuse.py", "c10data.py", "decide.py"):
            self.assertIn(f, files)
        self.assertNotIn("p04_rank_fuse.py", files)
        rec = p04.frozen_tools_check()
        self.assertEqual(set(rec["files_sha256"]), set(files))

        def altered(rel):
            b = p03.head_blob(rel)
            return b + b"\n# edited\n" if rel.endswith("decide.py") else b
        with self.assertRaises(p04.P04Error):
            p04.frozen_tools_check(blob=altered)
        # CRLF in the working copy is not a change
        self.assertEqual(p03.sha256_bytes_text(b"a\r\nb\n"), p03.sha256_bytes_text(b"a\nb\n"))

    def test_imports_are_p02_p03_functions(self):
        for name in ("basis", "log_softmax", "softmax", "rule_b_np", "pool_matrix", "pooled", "metrics",
                     "PlayerBootstrap", "load_dump", "load_val_keys", "read_question_meta", "params_dict"):
            self.assertIs(getattr(p04, name), getattr(rf, name), name)
        self.assertIs(p04.p3, p03)

    def test_interface_constants_equal_p04_opp(self):
        # the two tools meet through file names and format strings: p04_opp.py's val gate reads the frozen config
        # tune writes, check reads p04_opp.py's reports
        import p04_opp as opp
        self.assertEqual(opp.FROZEN_CONFIG_FORMAT, p04.FROZEN_FORMAT)
        self.assertEqual(opp.FORMAT["selftest"], p04.SELFTEST_FORMAT)
        self.assertEqual((opp.INNER_NAME, opp.INNER_REPORT), (p04.O4_INNER, p04.O4_INNER_REPORT))
        self.assertEqual((opp.VAL_NAME, opp.VAL_REPORT), (p04.O4_VAL, p04.O4_VAL_REPORT))
        self.assertEqual((opp.COUNT_REPORT, opp.DUP_NAME, opp.SELFTEST_NAME),
                         (p04.COUNT_REPORT, p04.DUPCHECK_REPORT, p04.SELFTEST_REPORT))
        self.assertEqual((opp.DUP_STOP_SHARE, opp.VAL_SLOTS, opp.INNER_GAMES),
                         (p04.DUP_STOP_FRACTION, p04.VAL_GAMES, p04.INNER_GAMES))
        self.assertEqual(opp.OWN_FILE, p04.OPP_TOOL)
        self.assertTrue(p03.same_path(opp.OPP_OUT, p04.OPP_DIR))
        self.assertTrue(p03.same_path(opp.FROZEN_CONFIG, os.path.join(p04.P04_OUT, "frozen_config.json")))
        self.assertTrue(p03.same_path(opp.COUNTS_REAL, p04.COUNTS_NPZ))


# ================================================================ synthetic end to end
def biased_synth_logits(rng, y, S, strength):
    """test_p02_rank.synth_logits with a level shift shared by blocks of 8 consecutive rows (the rows of one player /
    question are consecutive): pooling many games no longer recovers the level, as with real players, so R20 is not
    perfect on the synthetic questions and an independent source (O4) can add something."""
    y = np.asarray(y)
    k = np.arange(10)
    blocks = (len(y) + 7) // 8
    shift = np.repeat(rng.choice([-1, 0, 1], size=blocks, p=[0.2, 0.6, 0.2]), 8)[:len(y)]
    base = -2.0 * strength * np.abs(k[None, :] - np.clip(y + shift, 0, 9)[:, None]).astype(np.float64)
    z = base[:, None, :] + rng.normal(0, 1.0, size=(len(y), S, 10))
    return z.astype(np.float32)


class P04World:
    """P03World (synthetic prereg02 R20 chain + two prereg03 arms) plus synthetic p04_opp.py outputs."""

    def __init__(self, root, seed=0, o4="ok", val_o4="ok", dup_hits=0):
        import test_p02_rank
        old = test_p02_rank.synth_logits
        test_p02_rank.synth_logits = biased_synth_logits
        try:
            self.W = P03World(os.path.join(root, "p03w"), seed=seed)
        finally:
            test_p02_rank.synth_logits = old
        w = self.W.w
        self.opp = os.path.join(root, "opp")
        self.out = os.path.join(root, "out04")
        os.makedirs(self.opp)
        self.rng = np.random.default_rng(seed + 400)
        with np.load(os.path.join(w.dumps, "inner_keys.npz")) as z:
            self.keys = {k: z[k] for k in z.files}
        y = np.asarray(self.keys["label"], dtype=np.int64)
        consistent, noisy = self.side_pair(y, [str(p) for p in self.keys["player_id"]])
        o4v, t4v = (noisy, consistent) if o4 == "ok" else (consistent, noisy)
        inner = os.path.join(self.opp, p04.O4_INNER)
        np.savez(inner, rank=self.keys["rank"], game_id=self.keys["game_id"], player_id=self.keys["player_id"],
                 color=self.keys["color"], label=self.keys["label"], o4=o4v, t4=t4v, scored=np.ones(len(y), dtype=bool),
                 anomalies=np.zeros(len(y), dtype=np.int64))
        self.report(p04.O4_INNER_REPORT, output_sha256=sha(inner))
        self.report(p04.COUNT_REPORT, games=100, target_moves=10, opponent_moves=10, anomalies=0, n_cells=5,
                    counts_sha256="0" * 64)
        n_slots = len(self.W.val_y)
        self.report(p04.DUPCHECK_REPORT, status="stop" if dup_hits / n_slots >= 0.01 else "ok",
                    validation_vs_fit={"slots": n_slots, "slots_in_fit_any": dup_hits,
                                       "share_of_slots": dup_hits / n_slots, "stop": dup_hits / n_slots >= 0.01})
        # validation O4 (p04_opp.py val would write it after the frozen config; here the smoke world has it ready)
        qkey = [q for q, games, _ in w.questions for _ in games]
        vc, vn = self.side_pair(np.asarray(self.W.val_y), qkey)
        vo, vt = (vn, vc) if val_o4 == "ok" else (vc, vn)
        with np.load(os.path.join(w.dumps, "val_keys.npz")) as z:
            vk = {k: z[k] for k in z.files}
        self.val_path = os.path.join(self.opp, p04.O4_VAL)
        np.savez(self.val_path, question_id=vk["question_id"], slot=vk["slot"].astype(np.int64), color=vk["color"],
                 o4=vo, t4=vt, replay_failed=np.zeros(len(vo), dtype=bool))
        self.report(p04.O4_VAL_REPORT, output_sha256=sha(self.val_path))

    def side_pair(self, y, group):
        """(consistent within a group: level signal + group effect, noisy: level signal + independent noise)."""
        k = np.arange(10)
        base = -1.2 * np.abs(k[None] - y[:, None]).astype(np.float64)
        eff = {g: self.rng.normal(0, 1.5, size=10) for g in sorted(set(group))}
        consistent = base + np.array([eff[g] for g in group]) + self.rng.normal(0, 0.2, size=base.shape)
        noisy = base + self.rng.normal(0, 1.6, size=base.shape)
        return consistent, noisy

    def report(self, name, status="ok", **kw):
        with open(os.path.join(self.opp, name), "w", encoding="utf-8") as f:
            json.dump(dict({"status": status, "preview": True}, **kw), f)

    def args(self, cmd, *extra):
        w = self.W.w
        base = [cmd, "--smoke", "--p02-dir", w.p02, "--p02-local-dir", w.out, "--p03-dir", self.W.p03,
                "--opp-dir", self.opp, "--out-dir", self.out, "--split", w.split_csv]
        if cmd == "apply":
            base += ["--questions", w.q_csv]
        return base + list(extra)

    def run(self, cmd, *extra):
        return quiet(p04.main, self.args(cmd, *extra))

    def read(self, name):
        with open(os.path.join(self.out, name), encoding="utf-8") as f:
            return json.load(f)


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_p04_rank_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pipeline(self):
        P = P04World(self.tmp)
        self.assertEqual(P.run("tune"), 1)            # nothing before check
        self.assertEqual(P.run("apply"), 1)
        self.assertEqual(P.run("check"), 0)
        cr = P.read("check_report.json")
        self.assertEqual(cr["status"], "ok", cr["stops"])
        self.assertEqual(cr["usable_sources"], ["C5", "OP", "O4", "SWA", "SWB"])
        self.assertEqual(cr["smoke"], True)
        for s in ("O4", "SWA", "SWB"):
            col = cr["sources"][s]["colour"]
            self.assertGreater(col["inner"]["ci95"][0], 0, s)
        self.assertIsInstance(cr["sources"]["O4"]["colour"]["val_sub_questions"], str)   # deferred
        self.assertGreater(cr["sources"]["SWA"]["colour"]["val_sub_questions"]["dkappa"], 0)
        self.assertEqual(P.run("tune"), 1)            # selfcheck missing
        self.assertEqual(P.run("selfcheck"), 0)
        sc = P.read("selfcheck.json")
        self.assertTrue(sc["passed"], sc["failures"])
        for k in ("Q00_inner_oof_equals_R20", "Q00_refit_params_equal_frozen", "Q00_val_probs_equal_rank_val_probs",
                  "gen_path_Q00_seam"):
            self.assertIs(sc["checks"][k]["ok"], True, k)
        self.assertEqual(set(sc["design_reproduction"]), {"Q01", "Q02"})
        # tune never opens a validation file
        opened = []
        real_open, real_load = builtins.open, np.load

        def spy_open(file, *a, **kw):
            opened.append(str(file))
            return real_open(file, *a, **kw)

        def spy_load(file, *a, **kw):
            opened.append(str(file))
            return real_load(file, *a, **kw)
        builtins.open, np.load = spy_open, spy_load
        try:
            rc = P.run("tune")
        finally:
            builtins.open, np.load = real_open, real_load
        self.assertEqual(rc, 0)
        bases = [os.path.basename(p) for p in opened]
        self.assertIn(p04.O4_INNER, bases)
        self.assertFalse([b for b in bases if b.startswith("val_") or "rank_val" in b or "opening_val" in b], bases)
        tune = P.read("tune_report.json")
        fc = P.read("frozen_config.json")
        self.assertEqual([c["id"] for c in tune["combinations"]], ["Q00", "Q01", "Q02"])
        ok = {c["id"]: c["oof"]["rank_balanced"] for c in tune["combinations"]}
        self.assertEqual(ok["Q00"], sc["Q00"]["oof"]["rank_balanced"])
        for cid in ("Q01", "Q02"):
            self.assertEqual(ok[cid], sc["design_reproduction"][cid]["rank_balanced"])
        best = max(ok.values())
        first = next(i for i in ok if best - ok[i] <= p04.SELECT_TOL + p04.FLOAT_EPS)
        self.assertEqual(fc["combo"]["id"], first)
        self.assertIn(first, ("Q01", "Q02"))          # the synthetic O4 carries level information C5 lacks
        g = fc["gates"]
        self.assertEqual(g["passed"], all(g[k]["passed"] for k in ("G1", "G2", "G3", "G4")))
        self.assertEqual(fc["score_allowed"], g["passed"])
        self.assertEqual(fc["score_pending_at_apply"][0][:3], "O4:")
        self.assertEqual(fc["fail_opening"], p04.FAIL_OPENING)
        self.assertEqual([r["name"] for r in tune["record_only"]], [r[0] for r in p04.RECORD_ONLY])
        self.assertTrue(all(r["status"] == "ok" for r in tune["record_only"]))
        r10 = next(r for r in tune["record_only"] if r["name"].startswith("Q00 no opening"))
        self.assertEqual(r10["path"], "p02")
        self.assertEqual(set(fc["single_colour"]["kappa_reference"]),
                         set(fc["combo"]["sources"]) & {"O4", "SWA", "SWB"})
        self.assertEqual(fc["params"]["terms"], fc["combo"]["terms"])
        self.assertEqual(len(fc["folds"]), 5)
        # apply
        self.assertEqual(P.run("apply"), 0)
        app = P.read("apply_report.json")
        self.assertGreater(app["val_colour_check"]["O4"]["dkappa"], 0)
        self.assertEqual(app["score_allowed"], fc["score_allowed"])
        recs = decide.read_probs(os.path.join(P.out, "rank_val_probs.csv"))
        w = P.W.w
        self.assertEqual([r["question_id"] for r in recs], [q for q, _, _ in w.questions])
        self.assertTrue(recs[3]["missing"])
        # independent recomputation of question 0 with the frozen parameters
        n0 = len(w.questions[0][1])
        ok0 = w.val_parsed[:n0]
        z = w.val_logits["p02e9_d4"][:n0][ok0].astype(np.float64)
        l5 = rf.log_softmax(np.mean([rf.log_softmax(z[:, s, :]) for s in range(8)], axis=0))
        ov = p03.load_opening(os.path.join(w.out, "opening_val.npz"), "val")
        has = np.asarray(ov["has"])[:n0]
        mop = np.asarray(ov["o"])[:n0][has].mean(0) if has.any() else np.zeros(10)
        with np.load(P.val_path) as zz:
            o4 = rf.log_softmax(zz["o4"][:n0][ok0])
        terms = {"OP": mop, "O4": o4.mean(0)}
        for x in ("a", "b"):
            with np.load(os.path.join(P.W.p03, f"rank_{x}", "dumps", f"val_p03{x}_s0_swap.npz")) as zz:
                terms["SW" + x.upper()] = rf.log_softmax(zz["logits"][:n0][ok0].astype(np.float64)).mean(0)
        par = fc["params"]
        u = l5.mean(0) + p04.B @ np.asarray(par["beta"])
        for t in par["terms"]:
            u = u + par["w"][t] * terms[t]
        want = rf.softmax((math.exp(par["a"]) * (ok0.sum() / 12.0) ** par["gamma"] * u)[None])[0]
        np.testing.assert_allclose(recs[0]["probs"], want, rtol=0, atol=1e-12)
        self.assertEqual(app["check_report"]["sha256"], fc["check"]["check_report_sha256"])
        self.assertIn("bound_to_frozen_config", app["o4_val_file"])
        # a check report other than the one tune recorded is refused
        crp = os.path.join(P.out, "check_report.json")
        with open(crp, "rb") as f:
            kept = f.read()
        with open(crp, "ab") as f:
            f.write(b" ")
        self.assertEqual(P.run("apply"), 1)
        with open(crp, "wb") as f:
            f.write(kept)
        self.assertEqual(P.run("apply"), 0)
        # a tampered frozen configuration is refused
        fpath = os.path.join(P.out, "frozen_config.json")
        fz = json.load(open(fpath, encoding="utf-8"))
        fz["params"]["beta"][0] += 1e-9
        pseudo.write_json(fpath, fz)
        self.assertEqual(P.run("apply"), 1)

    def test_selfcheck_catches_moved_r20_parameter(self):
        P = P04World(self.tmp)
        self.assertEqual(P.run("selfcheck"), 0)
        fpath = os.path.join(P.W.w.out, "frozen_config.json")
        fc = json.load(open(fpath, encoding="utf-8"))
        fc["params"]["a"] += 1e-9
        pseudo.write_json(fpath, fc)
        self.assertEqual(P.run("selfcheck"), 1)
        self.assertEqual(P.read("selfcheck.json")["failures"], ["Q00_refit_params_equal_frozen"])

    def test_colour_failure_drops_o4_and_its_combinations(self):
        P = P04World(self.tmp, o4="bad")
        self.assertEqual(P.run("check"), 0)
        cr = P.read("check_report.json")
        self.assertEqual(cr["sources"]["O4"]["status"], "dropped")
        self.assertLess(cr["sources"]["O4"]["colour"]["inner"]["ci95"][0], 0)
        self.assertEqual([c["status"] for c in cr["combinations"]], ["ok", "dropped", "dropped"])
        self.assertEqual(P.run("selfcheck"), 0)
        self.assertEqual(P.run("tune"), 0)
        fc = P.read("frozen_config.json")
        self.assertEqual(fc["combo"]["id"], "Q00")
        self.assertFalse(fc["score_allowed"])
        tune = P.read("tune_report.json")
        skipped = {r["name"] for r in tune["record_only"] if r["status"] == "skipped"}
        self.assertEqual(skipped, {"O4 alone", "Q02+C6A", "Q01 no opening", "Q02 no opening"})
        self.assertEqual(P.run("apply"), 0)
        app = P.read("apply_report.json")
        self.assertFalse(app["score_allowed"])
        self.assertIsNone(app["o4_val_file"])
        # Q00 = R20's validation probabilities exactly
        a = decide.read_probs(os.path.join(P.out, "rank_val_probs.csv"))
        b = decide.read_probs(os.path.join(P.W.w.out, "rank_val_probs.csv"))
        self.assertEqual([r["probs"] for r in a], [r["probs"] for r in b])

    def test_validation_o4_colour_blocks_scoring(self):
        P = P04World(self.tmp, val_o4="bad")
        for cmd in ("check", "selfcheck", "tune"):
            self.assertEqual(P.run(cmd), 0, cmd)
        fc = P.read("frozen_config.json")
        self.assertIn("O4", fc["combo"]["sources"])
        self.assertEqual(P.run("apply"), 0)
        app = P.read("apply_report.json")
        self.assertLessEqual(app["val_colour_check"]["O4"]["dkappa"], 0)
        self.assertFalse(app["score_allowed"])
        self.assertTrue(any(r.startswith("O4: validation single-colour") for r in app["score_blocked_reasons"]))
        # a replay failure or a missing validation O4 stops apply
        with np.load(P.val_path) as z:
            arr = {k: z[k] for k in z.files}
        arr["replay_failed"][0] = True
        np.savez(P.val_path, **arr)
        P.report(p04.O4_VAL_REPORT, output_sha256=sha(P.val_path))
        self.assertEqual(P.run("apply"), 1)
        os.remove(P.val_path)
        self.assertEqual(P.run("apply"), 1)

    def test_structure_stops(self):
        P = P04World(self.tmp, dup_hits=40)
        self.assertEqual(P.run("check"), 1)
        cr = P.read("check_report.json")
        self.assertEqual(cr["status"], "stop")
        self.assertTrue(any(s.startswith("dupcheck:") for s in cr["stops"]))
        P.report(p04.DUPCHECK_REPORT, validation_vs_fit={"slots": 100, "slots_in_fit_any": 0, "share_of_slots": 0.0,
                                                         "stop": False})
        self.assertEqual(P.run("check"), 0)
        # O4 rows in another order than inner_keys -> stop
        inner = os.path.join(P.opp, p04.O4_INNER)
        with np.load(inner) as z:
            arr = {k: z[k] for k in z.files}
        good = dict(arr)
        order = np.arange(len(arr["o4"]))[::-1]
        np.savez(inner, **{k: v[order] for k, v in arr.items()})
        P.report(p04.O4_INNER_REPORT, output_sha256=sha(inner))
        self.assertEqual(P.run("check"), 1)
        self.assertTrue(any("inner_keys" in s for s in P.read("check_report.json")["stops"]))
        # the report next to it must say ok and name this file
        np.savez(inner, **good)
        P.report(p04.O4_INNER_REPORT, output_sha256="0" * 64)
        self.assertEqual(P.run("check"), 1)
        P.report(p04.O4_INNER_REPORT, status="stop", output_sha256=sha(inner))
        self.assertEqual(P.run("check"), 1)
        P.report(p04.O4_INNER_REPORT, output_sha256=sha(inner))
        P.report(p04.COUNT_REPORT, status="stop")
        self.assertEqual(P.run("check"), 1)
        P.report(p04.COUNT_REPORT)
        self.assertEqual(P.run("check"), 0)
        # a stop blocks tune even when selfcheck passed
        self.assertEqual(P.run("selfcheck"), 0)
        P.report(p04.COUNT_REPORT, status="stop")
        self.assertEqual(P.run("check"), 1)
        self.assertEqual(P.run("tune"), 1)


if __name__ == "__main__":
    unittest.main()
