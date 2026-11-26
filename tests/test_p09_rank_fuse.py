"""PREREG_09 tools/p09_rank_fuse.py: the pure pieces (gates vs Q08, apply's usable decision, the audit binding,
finalize, tune rerun, modes, constants, pins) and apply's Q09 step on synthetic P1 features under the guard.
No hs_lib / u1_lib / pilot_* / p1_* import in this process (the Q09 core test runs in a subprocess).
Run: .venv/Scripts/python.exe -m pytest -q tests/test_p09_rank_fuse.py
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

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import p02_rank_fuse as rf  # noqa: E402
import p09_forward as FW  # noqa: E402
import p09_rank_fuse as P  # noqa: E402

ENV = {k: v for k, v in dict(os.environ, PYTHONDONTWRITEBYTECODE="1").items() if k not in P.BLAS_ENV}


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process"
    for mod in ("p1_lib", "p1_extract", "p1_fuse", "hs_lib", "u1_lib", "pilot_fuse", "pilot_run", "torch"):
        if mod == "torch":
            continue
        assert mod not in sys.modules, f"{mod} imported inside the pytest process"


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

    def gate(self, *a):
        return P.gate_record(*a, compared="Q09 vs Q08")

    def test_g1_boundary(self):
        up = self.sc + 0.05
        self.assertTrue(self.gate(0.010 - 1e-12, 0.0, up, self.sc, self.boot, self.mono(0.01))["G1"]["passed"])
        self.assertFalse(self.gate(0.010 - 2e-12, 0.0, up, self.sc, self.boot, self.mono(0.01))["G1"]["passed"])
        g = self.gate(0.02, 0.0, up, self.sc, self.boot, self.mono(0.01))
        self.assertTrue(g["passed"])
        self.assertEqual(g["compared"], "Q09 vs Q08")

    def test_g2_lower_bound_zero_fails(self):
        g = self.gate(0.02, 0.0, self.sc, self.sc, self.boot, self.mono(0.01))
        self.assertEqual(g["G2"]["ci95"][0], 0.0)
        self.assertFalse(g["G2"]["passed"] or g["passed"])

    def test_g4_needs_lower_bound_and_both_colours(self):
        up = self.sc + 0.05
        for m in (self.mono(0.0), self.mono(-0.001), self.mono(0.01, b=0.0), self.mono(0.01, w=-0.01),
                  self.mono(0.01, n=P.MONO_QUESTIONS - 1), self.mono(0.01, b=None)):
            g = self.gate(0.02, 0.0, up, self.sc, self.boot, m)
            self.assertFalse(g["G4"]["passed"], m)
            self.assertFalse(g["passed"])
        self.assertTrue(self.gate(0.02, 0.0, up, self.sc, self.boot, self.mono(1e-9))["G4"]["passed"])

    def test_question_mean_record_only(self):
        g = self.gate(0.02, 0.0, self.sc + 0.05, self.sc, self.boot, self.mono(0.01))
        self.assertNotIn("G3", g)
        self.assertIn("question_mean", g)


class ApplyDecision(unittest.TestCase):
    same = {"passed": True}
    ok_dir = {"passed": True, "rho": 0.9, "threshold": 0.47}

    def test_all_good_needs_the_audit(self):
        for audit in ("missing", "ok", "unbound"):
            self.assertEqual(P.apply_decision(self.same, 0, True, True, True, self.ok_dir, True, audit)[0],
                             "pending_audit")
        self.assertIs(P.apply_decision(self.same, 0, True, True, True, self.ok_dir, True, "bug")[0], False)

    def test_each_blocker(self):
        cases = [({"passed": False}, 0, True, True, True, self.ok_dir, True),
                 (self.same, 1, True, True, True, self.ok_dir, True),
                 (self.same, 0, False, True, True, self.ok_dir, True),
                 (self.same, 0, True, False, True, self.ok_dir, True),
                 (self.same, 0, True, True, False, self.ok_dir, True),
                 (self.same, 0, True, True, True, {"passed": False}, True),
                 (self.same, 0, True, True, True, self.ok_dir, False)]
        for case in cases:
            usable, reasons = P.apply_decision(*case, "ok")
            self.assertIs(usable, False, case)
            self.assertTrue(reasons)


class Audit(unittest.TestCase):
    rep = {"outputs_sha256": {n: n * 2 for n in P.AUDIT_BOUND_OUTPUTS},
           "inputs_sha256": {n: n * 3 for n in P.AUDIT_BOUND_INPUTS}}

    def audit(self, **kw):
        a = {"format": P.FORMATS["audit"], "program_errors_affecting_predictions": [],
             "outputs_sha256": dict(self.rep["outputs_sha256"]), "inputs_sha256": dict(self.rep["inputs_sha256"])}
        a.update(kw)
        return a

    def test_bound(self):
        self.assertEqual(P.audit_binding(self.audit(), self.rep, "2026-10-13T10:00:00+08:00",
                                         "2026-10-13T09:00:00+08:00"), [])

    def test_unbound_cases(self):
        t_ok = ("2026-10-13T10:00:00+08:00", "2026-10-13T09:00:00+08:00")
        cases = [(self.audit(format="c10-prereg08 prescore_audit v1"), t_ok),
                 (self.audit(outputs_sha256=dict(self.rep["outputs_sha256"], **{"val_lp_p1e.npz": "x"})), t_ok),
                 (self.audit(inputs_sha256=dict(self.rep["inputs_sha256"], map="y")), t_ok),
                 (self.audit(program_errors_affecting_predictions=None), t_ok),
                 (self.audit(), ("2026-10-13T09:00:00+08:00", "2026-10-13T09:00:00+08:00")),
                 (self.audit(), (None, "2026-10-13T09:00:00+08:00"))]
        for a, (ta, tp) in cases:
            self.assertTrue(P.audit_binding(a, self.rep, ta, tp))

    def test_finalize(self):
        rep = {"usable": "pending_audit", "usable_blocked_reasons": ["apply_audit.json of these outputs not "
                                                                     "committed yet"]}
        with self.assertRaises(P.P09Error):
            P.finalize_decision(rep, "missing", None)
        with self.assertRaises(P.P09Error):
            P.finalize_decision(rep, "unbound", {"binding_problems": ["x"]})
        self.assertEqual(P.finalize_decision(rep, "ok", {}), (True, []))
        self.assertIs(P.finalize_decision(rep, "bug", {})[0], False)
        with self.assertRaises(P.P09Error):
            P.finalize_decision({"usable": True}, "ok", {})
        with self.assertRaises(P.P09Error):
            P.finalize_decision({"usable": False, "usable_blocked_reasons": ["Q08 does not reproduce"]}, "ok", {})


class Modes(unittest.TestCase):
    def ns(self, **kw):
        return argparse.Namespace(**{"precommit": False, "out_dir": None, **kw})

    def test_preview_folders(self):
        c = P.resolve(self.ns(out_dir=os.path.join(P.TMP, "x")), "check")
        self.assertTrue(c.preview)
        self.assertEqual(c.map_dir, os.path.join(P.TMP, "x", "map"))
        for bad in (P.OUT, P.TMP, P.EXPLORE_TMP, os.path.join(P.EXPLORE_TMP, "y"),
                    os.path.join(REPO, "data", "prereg08", "tmp", "x"), os.environ.get("TEMP", "C:\\")):
            with self.assertRaises(P.P09Error):
                P.resolve(self.ns(out_dir=bad), "check")
        with self.assertRaises(P.P09Error):
            P.resolve(self.ns(out_dir=os.path.join(P.TMP, "x"), precommit=True), "check")
        c = P.resolve(self.ns(), "tune")
        self.assertEqual((c.preview, c.out_dir, c.map_dir), (False, P.OUT, P.MAP_DIR))

    def test_preview_record_not_accepted_by_a_real_step(self):
        d = os.path.join(P.TMP, f"pytest_modes_{os.getpid()}")
        os.makedirs(d, exist_ok=True)
        try:
            P.write_json(os.path.join(d, "check_report.json"), {"format": P.FORMATS["check"], "preview": True})
            c = P.Ctx()
            c.out_dir, c.preview, c.precommit = d, False, False
            with self.assertRaises(P.P09Error) as cm:
                P.prior_report(c, "check_report.json", P.FORMATS["check"], False)
            self.assertIn("preview", str(cm.exception))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_frozen_usable(self):
        good = {"format": P.FORMATS["frozen"], "preview": False, "gates_passed": True, "combo": {"id": "Q09"}}
        self.assertTrue(P.frozen_usable(good))
        self.assertTrue(P.frozen_usable(dict(good, combo={"id": "Q08P"})))
        for bad in (dict(good, preview=True), dict(good, gates_passed=False), dict(good, combo={"id": "Q08"}),
                    dict(good, format="c10-prereg08 rank frozen_config v1")):
            with self.assertRaises(P.P09Error):
                P.frozen_usable(bad)

    def test_tune_rerun(self):
        self.assertIsNone(P.tune_rerun_check(None, False, None, None))
        self.assertEqual(P.tune_rerun_check({"reason": "r"}, True, None, None), {"reason": "r"})
        for rev in (None, 1):
            with self.assertRaises(P.P09Error):
                P.tune_rerun_check({"reason": "r"}, False, rev, "2026-10-12T10:00:00+00:00")
        with self.assertRaises(P.P09Error):
            P.tune_rerun_check({"reason": "r"}, False, 2, "2026-10-12T10:00:00+00:00")   # rev 2 not committed yet

    def test_label_path_and_blas(self):
        with self.assertRaises(P.P09Error):
            P.refuse_label_path(os.path.join(REPO, "data", "local", "rank_val_" + "ans" + "wers.csv"))
        with self.assertRaises(P.P09Error):
            P.sha_raw(os.path.join(REPO, "data", "local", "rank_val_" + "ans" + "wers.csv"))
        old = {k: os.environ.get(k) for k in P.BLAS_ENV + (P.PREVIEW_THREADS_OK,)}
        try:
            for k in old:
                os.environ.pop(k, None)
            self.assertEqual(P.blas_env_check(False), {})
            os.environ["MKL_NUM_THREADS"] = "2"
            for prev in (False, True):
                with self.assertRaises(P.P09Error):
                    P.blas_env_check(prev)
            os.environ[P.PREVIEW_THREADS_OK] = "1"
            self.assertEqual(P.blas_env_check(True), {"MKL_NUM_THREADS": "2"})
            with self.assertRaises(P.P09Error):
                P.blas_env_check(False)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_cli_refusals(self):
        def run(args, env=ENV):
            return subprocess.run([sys.executable, os.path.join(REPO, "tools", "p09_rank_fuse.py")] + args,
                                  capture_output=True, text=True, cwd=REPO, env=env, timeout=600)
        r = run(["apply"], env=dict(ENV, OPENBLAS_NUM_THREADS="2"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("BLAS", r.stderr)
        r = run(["tune", "--out-dir", P.OUT])
        self.assertEqual(r.returncode, 2)
        if not os.path.exists(os.path.join(P.OUT, "frozen_config.json")):
            r = run(["apply"])
            self.assertEqual(r.returncode, 2)
            self.assertIn("frozen_config.json", r.stderr)


class ReviewRules(unittest.TestCase):
    def ctx(self, preview):
        c = P.Ctx()
        c.preview, c.precommit, c.out_dir = preview, False, P.TMP if preview else P.OUT
        return c

    def test_selfcheck_accept(self):
        ok = {"passed": True}
        ens = {"passed": False, "only_ensemble_tolerance_failed": True, "problems": ["retrained ensemble ..."]}
        other = {"passed": False, "only_ensemble_tolerance_failed": False, "problems": ["seam ENS ..."]}
        for prev in (False, True):
            self.assertIsNone(P.selfcheck_accept(self.ctx(prev), ok, None))
            with self.assertRaises(P.P09Error):
                P.selfcheck_accept(self.ctx(prev), ok, 2)               # --revision only for a failed selfcheck
            for rev in (None, 2):
                with self.assertRaises(P.P09Error):
                    P.selfcheck_accept(self.ctx(prev), other, rev)      # seams / torch failures never pass
            for rev in (None, 1):
                with self.assertRaises(P.P09Error):
                    P.selfcheck_accept(self.ctx(prev), ens, rev)
        self.assertEqual(P.selfcheck_accept(self.ctx(True), ens, 2)["revision"], 2)
        with self.assertRaises(P.P09Error):
            P.selfcheck_accept(self.ctx(False), ens, 97)                # rev 97 is in no commit

    def test_fallback_allowed(self):
        self.assertTrue(P.fallback_allowed({"gates": {"passed": False}, "problems": []}))
        for bad in ({"gates": {"passed": True}}, {"gates": {}}, {},
                    {"gates": {"passed": False}, "problems": ["torch vs numpy ..."]}):
            with self.assertRaises(P.P09Error):
                P.fallback_allowed(bad)

    def test_frozen_rejects_precommit(self):
        good = {"format": P.FORMATS["frozen"], "preview": False, "gates_passed": True, "combo": {"id": "Q09"}}
        with self.assertRaises(P.P09Error):
            P.frozen_usable(dict(good, precommit=True))
        self.assertTrue(P.frozen_usable(dict(good, precommit=False)))

    def test_training_seconds(self):
        mr = {"cv": {"mlp512": {"seconds_cv": 10.0, "seconds_refit": 2.0}}}
        tr = {"members": {"mlp512": {"seconds": 12.5, "reused": False}}, "seconds": 99}
        self.assertEqual(P.training_seconds(mr, tr, False),
                         {"mlp512": {"cv": 10.0, "refit": 2.0, "member_total": 12.5, "reused_file": False}})
        self.assertEqual(P.training_seconds(mr, tr, True), {"lr_fit_seconds": 99})

    def test_code_differs(self):
        ts = {"files_sha256": {"tools/p09_map.py": "a", "tools/p09_forward.py": "b", "tools/p09_rank_fuse.py": "c"}}
        rec = {"files_sha256": {"tools/p09_map.py": "a", "tools/p09_forward.py": "b", "tools/p09_rank_fuse.py": "x"}}
        self.assertEqual(P.code_differs(ts, rec, P.TRAINING_CODE), [])        # rank_fuse changes do not matter
        self.assertEqual(P.code_differs(ts, {"files_sha256": {"tools/p09_map.py": "z"}}, P.TRAINING_CODE),
                         ["tools/p09_map.py", "tools/p09_forward.py"])


class Constants(unittest.TestCase):
    def test_registered_numbers(self):
        self.assertEqual(P.REF, {"Q02H": 0.6242752597994481, "Q08": 0.6986195853659671,
                                 "Q08_mono": 0.6946737091777063, "ENS": 0.7113584160645174,
                                 "ENS_mono": 0.7154116187609454})
        self.assertEqual((P.GATE, P.TOL, P.ENS_TOL, P.TORCH_TOL, P.SCALE), (0.010, 1e-12, 0.002, 1e-4, 4.0))
        self.assertEqual((P.RHO_FLOOR_VAL, P.RHO_FLOOR_TEST, P.MONO_QUESTIONS), (0.078, 0.155, 120544))
        self.assertEqual(P.INNER_USABLE, 73169)
        self.assertEqual(P.VENV, {"python": "3.12.13", "torch": "2.14.0+cpu", "numpy": "2.5.3", "scipy": "1.18.1",
                                  "threads": 8})
        self.assertEqual(P.Q08_TERMS, ["OP", "O4", "SWA", "SWB", "HS", "P1TO"])
        self.assertEqual({k: v["e_star"] for k, v in P.EXPLORE_MEMBERS.items()},
                         {"mlpclm256": 24, "mlpclm256_s1": 24, "mlpclm256_s2": 24, "mlp512": 24, "mlp512_s1": 15,
                          "lowrank3_ex": 30})
        self.assertEqual(list(P.EXPLORE_MEMBERS), FW.member_names())

    def test_pins_hold_now(self):
        got = P.check_pins()
        for r, s in {**P.P08_FROZEN_PINS, **P.EXPLORE_PINS, **P.P08_TOOL_PINS}.items():
            self.assertEqual(got[r], s, r)
        if os.path.exists(P.P3_SETS):
            self.assertEqual(got["data/prereg09/p3/sets.json"], P.DATA_PINS["data/prereg09/p3/sets.json"])


class ValExtraFeatures(unittest.TestCase):
    """read_val_ex on synthetic 08-format validation chunks: SHA-256 against the report, format / mode / keys checks,
    val_keys order, the extra features equal p09_forward.ex_raw32 (no real validation chunk is read)."""

    def setUp(self):
        self.d = os.path.join(P.TMP, f"pytest_valex_{os.getpid()}")
        os.makedirs(self.d, exist_ok=True)
        rng = np.random.default_rng(8)
        self.chunks, self.want = [], []
        for ci, refs in enumerate((np.arange(0, 4), np.arange(4, 7))):
            G = len(refs)
            pg = np.repeat(np.arange(G), 12).astype(np.int32)
            z = {"pos_game": pg, "pos_side": (np.arange(len(pg)) % 2).astype(np.int8),
                 "pos_turn": np.tile(np.arange(12), G).astype(np.int16),
                 "pos_logp": -np.abs(rng.normal(size=len(pg))).astype(np.float32),
                 "pos_entropy": np.abs(rng.normal(size=len(pg))).astype(np.float32)}
            p = os.path.join(self.d, f"chunk_{ci:05d}.npz")
            np.savez(p, format=np.array(P.R.FORMATS["val_chunk"]), meta_sha256=np.array(P.R.NEUTRAL_META_SHA),
                     mode=np.array("real"), keys_sha256=np.array(P.P7.VAL_KEYS_SHA), ref=refs, **z)
            self.chunks.append({"file": P.rel(p), "sha256": P.sha_raw(p)})
            self.want.append(FW.ex_raw32(z, G))

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_read(self):
        ex, shas = P.read_val_ex({"chunks": self.chunks}, {"n": 7})
        self.assertTrue(np.array_equal(ex, np.concatenate(self.want), equal_nan=True))
        self.assertEqual(shas, [c["sha256"] for c in self.chunks])
        with self.assertRaises(P.P09Error):
            P.read_val_ex({"chunks": self.chunks}, {"n": 8})
        bad = [dict(self.chunks[0], sha256="0" * 64)] + self.chunks[1:]
        with self.assertRaises(P.P09Error):
            P.read_val_ex({"chunks": bad}, {"n": 7})
        p = P.path_of(self.chunks[1]["file"])
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        np.savez(p, **dict(d, mode=np.array("smoke")))
        moved = [self.chunks[0], dict(self.chunks[1], sha256=P.sha_raw(p))]
        with self.assertRaises(P.P09Error):
            P.read_val_ex({"chunks": moved}, {"n": 7})


class Q09CoreUnderGuard(unittest.TestCase):
    """apply's Q09 step (q09_core) end to end in a subprocess after hs_lib's guard is installed (no data/local file):
    synthetic validation P1 features (move counts from val_hs.npz) and a synthetic map; with w_P1 = 0 the csv equals
    07's committed rank_val_probs.csv byte for byte; E is the masked mean of 4 lp (x 4 once); w_P1 = 1 changes it;
    the Q08P (LR) path runs too."""

    def test_q09_core_on_synthetic_p1(self):
        code = r'''
import sys, json, builtins, hashlib, os
sys.dont_write_bytecode = True
sys.path.insert(0, "tools")
import numpy as np
import p09_rank_fuse as P
import p09_forward as FW
R, P7, rf = P.R, P.P7, P.rf
PF = R.import_pilot_fuse()
assert builtins._c10_hs_guard
q = P7.q02_validation()
vh = dict(np.load(R.P07_VAL_HS))
core = P7.apply_core(PF, q, vh, P7.load_hs_map(R.P07_HS_MAP), json.load(open(R.P07_FROZEN, encoding="utf-8")))
n = q["vk"]["n"]
rng = np.random.default_rng(0)
tgt = np.asarray(vh["counts"]).astype(np.int64)
tot = np.asarray(vh["n_moves"]).astype(np.int64)
vp = {"means": rng.normal(size=(n, 2, 770)).astype(np.float32), "counts": np.stack([tgt, tot - tgt], 1),
      "n_moves": tot, "failed": np.asarray(vh["failed"]).astype(bool)}
ex32 = rng.normal(size=(n, 44)).astype(np.float32)
ex32[::7, :22] = np.nan
def params(kind, s):
    g = np.random.default_rng(s)
    if kind == "mlpclm":
        return {"W1": g.normal(size=(256, 1540)) * 0.02, "b1": g.normal(size=256) * 0.1, "W2": g.normal(size=(1, 256)) * 0.2,
                "b2": g.normal(size=1), "t0": np.array(-4.0), "dt": np.ones(8) * 0.5}
    if kind == "mlp":
        return {"W1": g.normal(size=(512, 1540)) * 0.02, "b1": g.normal(size=512) * 0.1, "W2": g.normal(size=(10, 512)) * 0.1,
                "b2": g.normal(size=10) * 0.1}
    return {"A": g.normal(size=(3, 1584)) * 0.05, "B": g.normal(size=(10, 3)), "b": g.normal(size=10)}
members = [{"name": s["name"], "kind": s["kind"], "seed": s["seed"], "e_star": 24, "params": params(s["kind"], i)}
           for i, s in enumerate(FW.MEMBERS)]
os.makedirs(P.TMP, exist_ok=True)
mpath = os.path.join(P.TMP, f"pytest_q09core_{os.getpid()}.npz")
FW.save_map(mpath, members, rng.normal(size=1540), rng.uniform(0.5, 2, 1540), rng.normal(size=44),
            rng.uniform(0.5, 2, 44), {"probe_used": "synthetic"})
mp = FW.load_map(mpath)
os.remove(mpath)
th = json.load(open(R.P07_FROZEN, encoding="utf-8"))["params"]["theta"]
def fc(w, cid="Q09"):
    return {"combo": {"id": cid}, "params": {"theta": th[:7] + [w] + th[7:]}, "direction_ref": {"rho_eval": 0.1}}
a = P.q09_core(q, core, vp, ex32, fc(0.0), mp)
b = P.q09_core(q, core, vp, ex32, fc(1.0), mp)
ref = open(R.P07_PROBS, "rb").read().replace(b"\r\n", b"\n")
us = FW.usable_games(vp["failed"], vp["counts"])
X = FW.x_raw(vp["means"], vp["counts"])
lp = np.zeros((n, 10)); lp[us] = FW.map_logp(mp, X[us], ex32[us])
usable = np.asarray(q["vk"]["parsed"], dtype=bool) & us
E4, _ = P7.masked_E(q["S"], 4.0 * lp, usable)
lr = {"W": rng.normal(size=(1540, 10)) * 0.01, "b": rng.normal(size=10) * 0.1, "mu": rng.normal(size=1540),
      "sd": rng.uniform(0.5, 2, 1540)}
c = P.q09_core(q, core, vp, ex32, fc(1.0, "Q08P"), lr)
print(json.dumps({"csv_equal_07_at_w0": a["csv_bytes"] == ref, "sha_w0": hashlib.sha256(a["csv_bytes"]).hexdigest(),
                  "lp_equal": bool(np.array_equal(a["lp"], lp)), "E_is_4lp": bool(np.allclose(a["E"], E4, rtol=0, atol=1e-12)),
                  "usable": int(a["usable"].sum()), "scale": a["e_info"]["scale"],
                  "differs_at_w1": bool(not np.allclose(np.nan_to_num(b["P"]), np.nan_to_num(a["P"]))),
                  "lr_differs": bool(not np.array_equal(c["lp"], b["lp"])),
                  "rho_finite": bool(np.isfinite(a["direction"]["rho"]))}))
'''
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=ENV,
                           timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        res = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(res["csv_equal_07_at_w0"], res)
        self.assertEqual(res["sha_w0"], P.R.P07_PROBS_SHA)
        self.assertTrue(res["lp_equal"] and res["E_is_4lp"], res)
        self.assertEqual(res["scale"], 4.0)
        self.assertGreater(res["usable"], 19000)
        self.assertTrue(res["differs_at_w1"] and res["lr_differs"] and res["rho_finite"], res)


if __name__ == "__main__":
    unittest.main()
