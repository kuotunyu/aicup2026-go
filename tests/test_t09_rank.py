"""Unit tests of tools/t09_rank.py and tools/t09_fake_tests.py (PREREG_09 5.2, rev 3 section 13; negative drills at
function level). CPU, fast.

Never imports hs_lib, u1_lib, pilot_*, p1_* or torch in the pytest process (the autouse fixture checks the guards
stayed out); the paths that need them run as subprocesses in tests/test_t09_dryrun.py. Reads only committed results
files (results/prereg09/rank/*, results/prereg08/*) and the validation P1 chunks data/prereg08/val/chunk_00000-00001.npz
(SHA-256 against results/prereg08/feat/val_report.json); never data/local, never a label file, never tools/score.py;
temporary files under data/test09/tmp only (removed).
Run: .venv/Scripts/python.exe -m pytest -q tests/test_t09_rank.py
"""

import argparse
import builtins
import csv
import json
import os
import shutil
import sys
import unittest
import uuid

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t09_rank as t9  # noqa: E402
import p09_forward as FW  # noqa: E402
import official_log as ol  # noqa: E402

TMP_ROOT = os.path.join(REPO, "data", "test09", "tmp")
GUARDED = ("hs_lib", "u1_lib", "pilot_run", "pilot_fuse", "p1_lib", "p1_fuse", "p1_extract", "torch", "p09_map")
TOL_JSON = os.path.join(REPO, "results", "test09", "design", "tol_e_measurement.json")


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process"


def ns(**kw):
    base = dict(dry_run=False, rehearsal=False, results_dir=None, t08_results_dir=None, p1_data_dir=None,
                t07_results_dir=None, t04_results_dir=None, t02_results_dir=None, t02_out_dir=None, tests_dir=None,
                checks_dir=None, r20_uploaded_at=None, now=None)
    base.update(kw)
    return argparse.Namespace(**base)


def code_of(fn, *a, **k):
    with pytest.raises(t9.T09Error) as ei:
        fn(*a, **k)
    return ei.value.code


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(TMP_ROOT, "unit_" + uuid.uuid4().hex[:8])
        os.makedirs(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for d in (TMP_ROOT, os.path.dirname(TMP_ROOT)):     # leave no empty data/test09 behind
            try:
                os.rmdir(d)
            except OSError:
                pass


# ---------------------------------------------------------------- modes
class Resolve(Tmp):
    def outside(self, name):
        return os.path.join(os.path.dirname(REPO), "c10_t09_unit_outside", name)

    def trees(self, **kw):
        d = dict(t08_results_dir=self.outside("r8"), t07_results_dir=self.outside("r7"),
                 t04_results_dir=self.outside("r4"), t02_results_dir=self.outside("d2"),
                 t02_out_dir=self.outside("o2"), tests_dir=self.outside("t"))
        d.update(kw)
        return d

    def test_real_mode_paths(self):
        c = t9.resolve(ns(), "apply")
        self.assertEqual((c["mode"], c["R9"], c["R8"], c["H"], c["R7"], c["R4"]),
                         ("real", t9.REAL_R9, t9.REAL_R8, t9.REAL_H8, t9.REAL_R7, t9.REAL_R4))
        self.assertEqual(c["write_roots"], [t9.REAL_R9, t9.REAL_TMP])
        self.assertEqual(c["rank"], os.path.join(t9.REAL_R8, "rank"))

    def test_real_mode_refuses_drill_flags_and_other_dirs(self):
        for kw in ({"now": "2026-11-20T00:00"}, {"r20_uploaded_at": "2026-11-04T15:00"},
                   {"results_dir": self.outside("r9")}, {"t08_results_dir": self.outside("r8")},
                   {"p1_data_dir": self.outside("h")}, {"checks_dir": self.outside("c")}):
            with self.subTest(kw):
                self.assertEqual(code_of(t9.resolve, ns(**kw), "apply"), 2)

    def test_dry_run_needs_trees_outside_the_repo(self):
        self.assertEqual(code_of(t9.resolve, ns(dry_run=True, **self.trees()), "apply"), 2)      # no --results-dir
        self.assertEqual(code_of(t9.resolve, ns(dry_run=True, results_dir=self.outside("r9"),
                                                **self.trees(t08_results_dir=None)), "apply"), 2)
        self.assertEqual(code_of(t9.resolve, ns(dry_run=True, results_dir=os.path.join(REPO, "results", "x"),
                                                **self.trees()), "apply"), 2)
        self.assertEqual(code_of(t9.resolve, ns(dry_run=True, results_dir=self.outside("r9"),
                                                **self.trees(t08_results_dir=os.path.join(REPO, "results", "test08"))),
                                 "apply"), 2)
        c = t9.resolve(ns(dry_run=True, results_dir=self.outside("r9"), **self.trees()), "apply")
        self.assertEqual(c["H"], os.path.join(self.outside("r8"), "p1_data"))
        self.assertEqual((c["mode"], c["TMP"]), ("dry-run", os.path.join(self.outside("r9"), "tmp")))
        self.assertEqual(code_of(t9.resolve, ns(dry_run=True, rehearsal=True, results_dir=self.outside("r9"),
                                                **self.trees()), "apply"), 2)

    def test_rehearsal_default_folder(self):
        c = t9.resolve(ns(rehearsal=True, **self.trees(t08_results_dir=t9.REHEARSAL_R8)), "apply")
        self.assertEqual((c["R9"], c["H"], c["TMP"]), (t9.REHEARSAL_R9, t9.REHEARSAL_H8, t9.REHEARSAL_TMP))

    def test_tests_dir_raw_and_local_refused(self):
        for t in (os.path.join(self.outside("t"), "raw"), os.path.join(REPO, "data", "local")):
            with self.subTest(t):
                self.assertEqual(code_of(t9.resolve, ns(dry_run=True, results_dir=self.outside("r9"),
                                                        **self.trees(tests_dir=t)), "apply"), 2)

    def test_forbidden_writes(self):
        for rel in ("results/prereg09/rank/x.json", "results/prereg08/rank/x.json", "results/test08/rank/x.csv",
                    "results/test07/rank/x.csv", "results/test04/rank/x.csv", "results/test02/x", "data/local/x",
                    "results/score_ledger.csv", "results/test06/x", "data/test08/p1/x", "data/prereg08/val/x"):
            with self.subTest(rel):
                self.assertIsNotNone(t9.forbidden_area(os.path.join(REPO, *rel.split("/"))))
        self.assertIsNone(t9.forbidden_area(os.path.join(REPO, "results", "test09", "rank", "x.csv")))
        self.assertIsNone(t9.forbidden_area(os.path.join(REPO, "data", "test09", "tmp", "x")))
        ctx = {"write_roots": [self.tmp]}
        self.assertEqual(code_of(t9.guard_write, ctx, os.path.join(REPO, "results", "test09", "x")), 2)
        self.assertEqual(t9.guard_write(ctx, os.path.join(self.tmp, "a")), os.path.join(self.tmp, "a"))


# ---------------------------------------------------------------- registered values and prerequisites
class Rules(Tmp):
    def test_constants(self):
        self.assertEqual((t9.TOL_EX, t9.RHO_FLOOR, t9.P09_MIN_REV, t9.SEAM_GAMES, t9.TOL_Q08),
                         (0.05, 0.155, 5, 400, 1e-12))
        self.assertEqual(t9.EX_RECORD_ONLY, ("t_frac_lowp", "o_frac_lowp"))
        self.assertEqual((len(t9.EX_JUDGED), len(t9.EX_RECORDED)), (42, 2))
        self.assertEqual(t9.K6_09.isoformat(), "2026-11-19T23:59:59+08:00")
        self.assertEqual(t9.METHOD, "prereg09 Q09 = Q02H + 4 x P1TO_E (ens3fam), rule B (甲)")
        self.assertAlmostEqual(max(0.5 * t9.RHO_MONO, t9.RHO_FLOOR), 0.48425802602327594, places=15)

    def test_tol_e_is_the_measured_one(self):
        if not os.path.exists(TOL_JSON):
            self.skipTest("results/test09/design/tol_e_measurement.json not written yet")
        with open(TOL_JSON, encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["tol_E"], max(0.04, 2.5 * m["m"]))
        self.assertEqual(t9.TOL_E, m["tol_E"])

    def test_official_log_agrees(self):
        self.assertEqual(ol.T09_TOL, {"std_ex": t9.TOL_EX, "lp_p1e": t9.TOL_E})
        self.assertEqual((ol.T09_RHO_MONO, ol.T09_RHO_FLOOR, ol.T09_TOL_Q08), (t9.RHO_MONO, t9.RHO_FLOOR, t9.TOL_Q08))
        self.assertEqual(list(ol.T09_COMBO_SOURCES), t9.COMBO["sources"])
        self.assertEqual((ol.T09_COMBO_ID, ol.T09_METHOD, ol.T09_FORMAT), (t9.COMBO["id"], t9.METHOD,
                                                                           t9.FORMATS["apply"]))
        self.assertEqual((ol.K6_09, ol.P09_MIN_REV), (t9.K6_09, t9.P09_MIN_REV))
        self.assertEqual(ol.T09_ADOPTION_SHA, t9.P09_PINS[ol.T09_ADOPTION])
        self.assertTrue(set(ol.T09_INPUTS_REQUIRED) <= set(t9.P09_PINS) | {
            ol.T09_GPU_SEAM, ol.T09_SEAM_LOCAL, ol.T09_08_REPORT, ol.T09_08_PROBS, "results/test08/rank/submission_rank.csv",
            "results/test08/rank/test_lp_p1.npz", "results/test07/rank/t07_apply_report.json",
            "results/test04/rank/t04_apply_report.json", "results/test02/rank/apply_test_report.json",
            "results/test02/checks/rank/t1.json"} | {f"results/test08/p1/p1_{s}_report.json" for s in t9.SETS})

    def test_pins_hold(self):
        fc, mp, rec = t9.check_p09_pins()
        self.assertEqual(fc["combo"]["id"], "Q09")
        self.assertEqual(rec["prereg09_sha256"], t9.P09_PINS)
        self.assertEqual([m["name"] for m in mp["members"]], FW.member_names())
        self.assertEqual(float(fc["direction_ref"]["rho_mono"]), t9.RHO_MONO)
        self.assertEqual(fc["map"]["sha256"], t9.MAP_SHA)

    def test_pin_change_refused(self):
        orig = dict(t9.P09_PINS)
        try:
            t9.P09_PINS["results/prereg09/rank/p1e_map.npz"] = "0" * 64
            self.assertEqual(code_of(t9.check_p09_pins), 2)
        finally:
            t9.P09_PINS.clear()
            t9.P09_PINS.update(orig)

    def test_venv(self):
        fc, _mp, _rec = t9.check_p09_pins()
        self.assertTrue(t9.check_venv(fc)["equal"])
        bad = json.loads(json.dumps(fc))
        bad["environment"]["versions"]["numpy"] = "2.5.4"
        self.assertEqual(code_of(t9.check_venv, bad), 4)

    def test_k6_09(self):
        ctx = {"real": False, "mode": "dry-run", "rank9": self.tmp}
        self.assertFalse(t9.check_k6(ctx, ns(now="2026-11-19T23:59:59+08:00"))["passed"])
        self.assertEqual(code_of(t9.check_k6, ctx, ns(now="2026-11-20T00:00:00+08:00")), 3)
        with open(os.path.join(self.tmp, "t09_apply_report.json"), "w", encoding="utf-8") as f:
            json.dump({"exit_code": 0}, f)
        self.assertEqual(code_of(t9.check_k6, ctx, ns(now="2026-11-20T00:00:00+08:00")), 2)

    def test_bug_revision_below_5(self):
        self.assertEqual(code_of(t9.check_bug_revision, 3, []), 2)
        self.assertEqual(code_of(t9.check_bug_revision, 4, []), 2)          # rev 4 is the EX rule (test day)
        self.assertIsNone(t9.check_bug_revision(None, []))

    def test_rate_rule(self):
        self.assertTrue(t9.rate_ok(25, 5000))
        self.assertFalse(t9.rate_ok(26, 5000))

    def test_blas_env(self):
        old = os.environ.get("OPENBLAS_NUM_THREADS")
        os.environ["OPENBLAS_NUM_THREADS"] = "4"
        try:
            self.assertEqual(code_of(t9.blas_env_check), 2)
        finally:
            if old is None:
                os.environ.pop("OPENBLAS_NUM_THREADS")
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = old


class T08State(Tmp):
    def ctx(self):
        r8 = os.path.join(self.tmp, "test08")
        os.makedirs(os.path.join(r8, "rank"), exist_ok=True)
        return {"real": False, "mode": "dry-run", "R8": r8, "rank": os.path.join(r8, "rank"),
                "p1": os.path.join(r8, "p1"), "R7": None, "R4": None, "C": None, "D2": None, "O2": None}

    def write(self, path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def test_not_used_not_done_and_format(self):
        c = self.ctx()
        self.assertEqual(code_of(t9.t08_state, c), 2)                            # 08 not done
        self.write(os.path.join(c["R8"], "not_used.json"), {"exit_code": 3, "mode": "dry-run"})
        self.assertEqual(code_of(t9.t08_state, c), 3)                            # 08 not used -> 09 not used
        os.remove(os.path.join(c["R8"], "not_used.json"))
        rp = os.path.join(c["rank"], "t08_apply_report.json")
        self.write(rp, {"format": "x", "mode": "dry-run", "exit_code": 0, "usable": True})
        self.assertEqual(code_of(t9.t08_state, c), 2)
        self.write(rp, {"format": t9.t8.FORMATS["apply"], "mode": "real", "exit_code": 0, "usable": True})
        self.assertEqual(code_of(t9.t08_state, c), 2)                            # another mode
        self.write(rp, {"format": t9.t8.FORMATS["apply"], "mode": "dry-run", "exit_code": 3, "usable": False})
        self.assertEqual(code_of(t9.t08_state, c), 3)
        self.write(rp, {"format": t9.t8.FORMATS["apply"], "mode": "dry-run", "exit_code": 0, "usable": True,
                        "combo": {"id": "Q08"}})
        self.assertEqual(code_of(t9.t08_state, c), 2)                            # outputs missing


# ---------------------------------------------------------------- chunks, features and the GPU seam
class Seam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.val = t9.val_seam_chunks()
        cls.mp = FW.load_map(t9.path_of("results/prereg09/rank/p1e_map.npz"))
        cls.lpz = dict(np.load(t9.path_of("results/prereg09/rank/val_lp_p1e.npz"), allow_pickle=False))

    def seam(self):
        return {k: np.array(v) for k, v in self.val.items()}

    def test_forward_reproduces_the_committed_validation_lp(self):
        lp, ok = t9.set_features(self.val, self.mp)
        self.assertTrue(np.array_equal(ok, self.lpz["p1_ok"][:400]))
        self.assertLess(float(np.max(np.abs(lp - self.lpz["lp"][:400]))), 1e-12)

    def test_identical(self):
        r = t9.gpu_seam_record(self.seam(), self.val, self.mp)
        self.assertTrue(r["passed"])
        self.assertEqual((r["max_abs_std_ex"], r["max_abs_lp_p1e"]), (0.0, 0.0))
        self.assertTrue(r["bitwise_equal_means"] and r["bitwise_equal_ex"])
        self.assertEqual(r["games"], 400)

    def test_ex_tolerance(self):
        g = int(np.flatnonzero(FW.usable_games(self.val["failed"], self.val["counts"]))[0])
        A = np.asarray(self.mp["members"][5]["params"]["A"], dtype=np.float64)
        w = np.abs(A[:, FW.X_DIM:]).sum(0)
        j = min(t9.EX_JUDGED, key=lambda i: w[i])                          # the judged EX feature with the least weight
        for d, ok in ((0.049, True), (0.051, False)):
            z = self.seam()
            z["ex32"][g, j] += np.float32(d * self.mp["sd_ex"][j])
            r = t9.gpu_seam_record(z, self.val, self.mp)
            self.assertLess(abs(r["max_abs_std_ex"] - d), 1e-3)
            self.assertLessEqual(r["max_abs_lp_p1e"], t9.TOL_E)
            self.assertEqual(r["passed"], ok, d)

    def test_frac_lowp_jump_recorded_only(self):
        """PREREG_09 rev 4: a jump in only the two frac_lowp features passes (recorded); a 0.06 shift in any other
        EX feature fails."""
        ok_games = np.flatnonzero(FW.usable_games(self.val["failed"], self.val["counts"]))
        names = FW.ex_names()
        z = self.seam()
        for k, g in zip(t9.EX_RECORDED, ok_games[:3]):
            z["ex32"][g, k] += np.float32(0.25 * self.mp["sd_ex"][k])      # the measured off-grid size
        r = t9.gpu_seam_record(z, self.val, self.mp)
        self.assertEqual(r["max_abs_std_ex"], 0.0)
        self.assertGreater(r["max_abs_std_ex_all_44"], 0.2)
        self.assertEqual(r["frac_lowp_record_only"]["games_jumped"], 2)
        self.assertGreater(r["frac_lowp_record_only"]["max_abs_std"], 0.2)
        self.assertLessEqual(r["max_abs_lp_p1e"], t9.TOL_E)
        self.assertTrue(r["passed"])
        g = int(ok_games[5])
        for j in (names.index("t_logp_q50"), names.index("o_ent_std"), names.index("t_log1p_count")):
            with self.subTest(names[j]):
                z = self.seam()
                z["ex32"][g, j] += np.float32(0.06 * self.mp["sd_ex"][j])
                r = t9.gpu_seam_record(z, self.val, self.mp)
                self.assertAlmostEqual(r["max_abs_std_ex"], 0.06, delta=1e-3)
                self.assertFalse(r["passed"])

    def test_lp_tolerance(self):
        g = int(np.flatnonzero(FW.usable_games(self.val["failed"], self.val["counts"]))[0])
        z = self.seam()
        z["means"][g, 0, :] += (0.5 * self.mp["sd_x"][:FW.F_DIM]).astype(np.float32)
        r = t9.gpu_seam_record(z, self.val, self.mp)
        self.assertEqual(r["max_abs_std_ex"], 0.0)
        self.assertGreater(r["max_abs_lp_p1e"], t9.TOL_E)
        self.assertFalse(r["passed"])
        self.assertTrue(r["lp_above_review_line"])

    def test_counts(self):
        for k, fn in (("n_moves", lambda z: z["n_moves"].__setitem__(3, z["n_moves"][3] + 1)),
                      ("counts", lambda z: z["counts"].__setitem__((3, 1), z["counts"][3, 1] + 1)),
                      ("failed", lambda z: z["failed"].__setitem__(3, True))):
            with self.subTest(k):
                z = self.seam()
                fn(z)
                r = t9.gpu_seam_record(z, self.val, self.mp)
                self.assertFalse(r["counts_equal"])
                self.assertFalse(r["passed"])
        z = {k: v[:399] for k, v in self.seam().items()}
        self.assertFalse(t9.gpu_seam_record(z, self.val, self.mp)["passed"])


class Chunks(Tmp):
    def test_sha_and_positions(self):
        vr = t9.read_json(t9.VAL_REPORT)
        src = t9.path_of(vr["chunks"][0]["file"])
        dst = os.path.join(self.tmp, "chunk_00000.npz")
        shutil.copyfile(src, dst)
        rows, d = t9.read_set_chunks(self.tmp, ["chunk_00000.npz"], [vr["chunks"][0]["sha256"]])
        self.assertEqual((rows[0]["games"], d["ex32"].shape), (200, (200, FW.NEX)))
        self.assertEqual(code_of(t9.read_set_chunks, self.tmp, ["chunk_00000.npz"], ["0" * 64]), 2)
        with np.load(dst, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "pos_turn"}
        np.savez(dst, **arrays)
        self.assertEqual(code_of(t9.read_set_chunks, self.tmp, ["chunk_00000.npz"]), 2)
        self.assertEqual(code_of(t9.read_set_chunks, self.tmp, ["chunk_00009.npz"]), 2)

    def test_incomplete_positions_refused(self):
        """rev 4 addendum: positions per game and side must equal counts exactly; empty / truncated / shifted
        per-position arrays are refused (2)."""
        vr = t9.read_json(t9.VAL_REPORT)
        src = t9.path_of(vr["chunks"][0]["file"])
        with np.load(src, allow_pickle=False) as z:
            base = {k: z[k] for k in z.files}
        self.assertEqual(t9.pos_problems(base, len(base["ref"])), [])
        dst = os.path.join(self.tmp, "chunk_00000.npz")
        n = len(base["pos_side"])
        cases = {"empty": {k: base[k][:0] for k in t9.POS_KEYS},
                 "truncated": {k: base[k][:n - 1] for k in t9.POS_KEYS},
                 "one array short": {"pos_turn": base["pos_turn"][:n - 1]},
                 "side flipped": {"pos_side": np.where(np.arange(n) == 0, 1 - base["pos_side"], base["pos_side"])
                                  .astype(base["pos_side"].dtype)},
                 "game out of range": {"pos_game": np.where(np.arange(n) == n - 1, 200, base["pos_game"])
                                       .astype(base["pos_game"].dtype)}}
        for name, over in cases.items():
            with self.subTest(name):
                np.savez(dst, **dict(base, **over))
                self.assertTrue(t9.pos_problems(dict(base, **over), len(base["ref"])))
                self.assertEqual(code_of(t9.read_set_chunks, self.tmp, ["chunk_00000.npz"]), 2)


class Q08Compare(Tmp):
    REF = os.path.join(REPO, "results", "prereg08", "rank", "rank_val_probs.csv")

    def rows(self):
        with open(self.REF, newline="", encoding="utf-8") as f:
            return [r for r in csv.reader(f)][1:]

    def test_equal_and_tolerance(self):
        rows = self.rows()
        r = t9.probs_rows_compare(rows, self.REF)
        self.assertTrue(r["byte_equal"] and r["equal"])
        i = next(k for k, x in enumerate(rows) if x[3])
        tiny = [list(x) for x in rows]
        tiny[i][3] = repr(float(tiny[i][3]) + 1e-13)
        r = t9.probs_rows_compare(tiny, self.REF)
        self.assertFalse(r["byte_equal"])
        self.assertTrue(r["equal"])
        big = [list(x) for x in rows]
        big[i][3] = repr(float(big[i][3]) + 1e-11)
        self.assertFalse(t9.probs_rows_compare(big, self.REF)["equal"])
        swap = [list(x) for x in rows]
        p = np.array(swap[i][3:], dtype=float)
        k = int(p.argmax())
        p[[k, (k + 5) % 10]] = p[[(k + 5) % 10, k]]
        swap[i][3:] = [repr(float(v)) for v in p]
        r = t9.probs_rows_compare(swap, self.REF)
        self.assertFalse(r["equal"])
        self.assertGreater(r["rule_a_differences"], 0)
        self.assertFalse(t9.probs_rows_compare(rows[:-1], self.REF)["equal"])
        ids = [list(x) for x in rows]
        ids[0][0] = "val_x"
        self.assertFalse(t9.probs_rows_compare(ids, self.REF)["equal"])


class Helpers(unittest.TestCase):
    def test_seam_verdict_is_07s(self):
        base = {"problems": [], "count_mismatch_questions": [], "byte_identical_probs": True,
                "byte_identical_pred_B": True, "passed": True}
        self.assertEqual(t9.seam_verdict(base, True), "passed")
        self.assertEqual(t9.seam_verdict(dict(base, byte_identical_pred_B=False), True), "drift")
        self.assertEqual(t9.seam_verdict(dict(base, byte_identical_probs=False, passed=False), False), "not_used")
        self.assertEqual(t9.seam_verdict(dict(base, problems=["x"]), True), "structure")

    def test_fake_tests_measurement_rule(self):
        import t09_fake_tests as t9f
        self.assertEqual((t9f.FLOOR_E, t9f.FACTOR_E), (0.04, 2.5))
        self.assertEqual(code_of_fake(t9f.write_json, os.path.join(REPO, "results", "test08", "x.json"), {}),
                         "refused")

    def test_no_guarded_module_loaded(self):
        """A fresh process that imports the two 09 tools loads none of the guarded modules (torch, p09_map, hs_lib,
        u1_lib, pilot_*, p1_*); checked in a subprocess because other test files of a full run load torch."""
        import subprocess
        code = ("import sys; sys.path.insert(0, 'tools'); import t09_rank, t09_fake_tests; "
                f"print(sorted(m for m in {list(GUARDED)!r} if m in sys.modules))")
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(r.stdout.strip(), "[]")


def code_of_fake(fn, *a):
    try:
        fn(*a)
    except SystemExit:
        return "refused"
    return "ok"


if __name__ == "__main__":
    unittest.main()
