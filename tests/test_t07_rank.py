"""Unit tests of tools/t07_rank.py (results/prereg07/design/spec.md 7, 9 "負面演練" at function level). CPU, fast.

Never imports hs_lib, u1_lib or pilot_* in the pytest process (spec 0; the autouse fixture checks the guards stayed
out); the paths that need them run as subprocesses in tests/test_t07_dryrun.py. Reads only committed results files
(results/prereg07/hs/val_hs.npz, val_seam_moves.npz, hs_map.npz, score_H7R.json); never data/local, never a label file,
never tools/score.py; temporary files under data/test07/tmp only (removed).
Run: .venv/Scripts/python.exe -m pytest -q tests/test_t07_rank.py
"""

import argparse
import builtins
import datetime as dt
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
import t07_rank as t7  # noqa: E402

TMP_ROOT = os.path.join(REPO, "data", "test07", "tmp")


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process (spec 0)"


def ns(**kw):
    base = dict(dry_run=False, rehearsal=False, results_dir=None, hs_data_dir=None, t04_results_dir=None,
                t02_results_dir=None, t02_out_dir=None, tests_dir=None, checks_dir=None, r20_uploaded_at=None, now=None,
                max_games=0, fake_scorer=False, ignore_other_gpu_jobs=False)
    base.update(kw)
    return argparse.Namespace(**base)


def code_of(fn, *a, **k):
    with pytest.raises(t7.T07Error) as ei:
        fn(*a, **k)
    return ei.value.code


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(TMP_ROOT, "unit_" + uuid.uuid4().hex[:8])
        os.makedirs(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for d in (TMP_ROOT, os.path.dirname(TMP_ROOT)):     # leave no empty data/test07 behind
            try:
                os.rmdir(d)
            except OSError:
                pass


# ---------------------------------------------------------------- modes
class Resolve(Tmp):
    def outside(self, name):
        return os.path.join(os.path.dirname(REPO), "c10_t07_unit_outside", name)

    def test_real_mode_paths(self):
        c = t7.resolve(ns(), "apply")
        self.assertEqual((c["mode"], c["R7"], c["H"], c["R4"]), ("real", t7.REAL_R7, t7.REAL_H, t7.REAL_R4))

    def test_real_mode_refuses_drill_flags(self):
        for kw in ({"max_games": 5}, {"fake_scorer": True}, {"now": "2026-11-20T00:00"},
                   {"r20_uploaded_at": "2026-11-04T15:00"}, {"ignore_other_gpu_jobs": True}):
            with self.subTest(kw):
                self.assertEqual(code_of(t7.resolve, ns(**kw), "hs"), 2)

    def test_real_mode_other_dirs_refused(self):
        self.assertEqual(code_of(t7.resolve, ns(results_dir=self.outside("r7")), "apply"), 2)
        self.assertEqual(code_of(t7.resolve, ns(hs_data_dir=self.outside("h")), "apply"), 2)

    def test_dry_run_needs_trees_outside_the_repo(self):
        trees = dict(t04_results_dir=self.outside("r4"), t02_results_dir=self.outside("d2"),
                     t02_out_dir=self.outside("o2"), tests_dir=self.outside("t"))
        self.assertEqual(code_of(t7.resolve, ns(dry_run=True, **trees), "apply"), 2)          # no --results-dir
        self.assertEqual(code_of(t7.resolve, ns(dry_run=True, results_dir=os.path.join(REPO, "results", "x"),
                                                **trees), "apply"), 2)
        c = t7.resolve(ns(dry_run=True, results_dir=self.outside("r7"), **trees), "apply")
        self.assertEqual(c["H"], os.path.join(self.outside("r7"), "hs_data"))
        self.assertEqual(c["mode"], "dry-run")

    def test_rehearsal_default_folder(self):
        trees = dict(t04_results_dir=self.outside("r4"), t02_results_dir=self.outside("d2"),
                     t02_out_dir=self.outside("o2"), tests_dir=self.outside("t"))
        c = t7.resolve(ns(rehearsal=True, **trees), "apply")
        self.assertEqual((c["R7"], c["H"], c["TMP"]), (t7.REHEARSAL_R7, t7.REHEARSAL_H, t7.REHEARSAL_TMP))

    def test_tests_dir_raw_and_local_refused(self):
        trees = dict(t04_results_dir=self.outside("r4"), t02_results_dir=self.outside("d2"),
                     t02_out_dir=self.outside("o2"), results_dir=self.outside("r7"))
        for t in (os.path.join(self.outside("t"), "raw"), os.path.join(REPO, "data", "local")):
            with self.subTest(t):
                self.assertEqual(code_of(t7.resolve, ns(dry_run=True, tests_dir=t, **trees), "apply"), 2)

    def test_forbidden_writes(self):
        for rel in ("results/prereg07/rank/x.json", "results/test04/rank/x.csv", "results/test02/x", "data/local/x",
                    "results/score_ledger.csv", "results/test06/x"):
            with self.subTest(rel):
                self.assertIsNotNone(t7.forbidden_area(os.path.join(REPO, *rel.split("/"))))
        self.assertIsNone(t7.forbidden_area(os.path.join(REPO, "results", "test07", "rank", "x.csv")))
        ctx = {"write_roots": [self.tmp]}
        self.assertEqual(code_of(t7.guard_write, ctx, os.path.join(REPO, "results", "test07", "x")), 2)
        self.assertEqual(t7.guard_write(ctx, os.path.join(self.tmp, "a")), os.path.join(self.tmp, "a"))


# ---------------------------------------------------------------- registered decisions
class Rules(Tmp):
    def test_rate_rule(self):
        self.assertTrue(t7.rate_ok(25, 5000))
        self.assertFalse(t7.rate_ok(26, 5000))
        self.assertTrue(t7.rate_ok(23, 4787))
        self.assertFalse(t7.rate_ok(24, 4787))
        self.assertFalse(t7.rate_ok(0, 0))

    def test_seam_verdict(self):
        base = {"problems": [], "count_mismatch_questions": [], "byte_identical_probs": True,
                "byte_identical_pred_B": True, "passed": True}
        self.assertEqual(t7.seam_verdict(base, True), "passed")
        self.assertEqual(t7.seam_verdict(dict(base, byte_identical_pred_B=False), True), "drift")
        self.assertEqual(t7.seam_verdict(dict(base, byte_identical_probs=False, passed=True), False), "passed")
        self.assertEqual(t7.seam_verdict(dict(base, byte_identical_probs=False, passed=False), False), "not_used")
        self.assertEqual(t7.seam_verdict(dict(base, count_mismatch_questions=["q"]), False), "structure")
        self.assertEqual(t7.seam_verdict(dict(base, problems=["order"]), True), "structure")

    def test_k6_07(self):
        ctx = {"real": False, "mode": "dry-run", "rank": self.tmp}
        rec = t7.check_k6(ctx, ns(now="2026-11-19T23:59:59+08:00"))
        self.assertFalse(rec["passed"])
        self.assertEqual(code_of(t7.check_k6, ctx, ns(now="2026-11-20T00:00:00+08:00")), 3)
        with open(os.path.join(self.tmp, "t07_apply_report.json"), "w", encoding="utf-8") as f:
            json.dump({"exit_code": 0}, f)
        self.assertEqual(code_of(t7.check_k6, ctx, ns(now="2026-11-20T00:00:00+08:00")), 2)

    def test_h7r_condition(self):
        rec = t7.h7r_condition()
        self.assertTrue(rec["holds"])
        self.assertEqual(rec["questions"], 1589)
        real = t7.path_of("results/prereg07/rank/score_H7R.json")
        with open(real, encoding="utf-8") as f:
            s = json.load(f)
        fake = os.path.join(self.tmp, "score_H7R.json")
        orig = t7.path_of
        for k, v in (("A_rank_balanced", s["A_rank_balanced"] + 2e-12), ("B_rank_balanced", 0.5699),
                     ("A_mean", s["A_mean"] - 2e-12)):
            with self.subTest(k):
                with open(fake, "w", encoding="utf-8") as f:
                    json.dump(dict(s, **{k: v}), f)
                t7.path_of = lambda rel: fake if rel.endswith("score_H7R.json") else orig(rel)
                try:
                    self.assertEqual(code_of(t7.h7r_condition), 3)
                finally:
                    t7.path_of = orig
        t7.path_of = lambda rel: os.path.join(self.tmp, "missing.json") if rel.endswith("score_H7R.json") else orig(rel)
        try:
            self.assertEqual(code_of(t7.h7r_condition), 3)
        finally:
            t7.path_of = orig

    def test_pins_hold(self):
        fc, rec = t7.check_p07_pins()
        self.assertEqual(fc["combo"]["id"], "Q02H")
        self.assertEqual(rec["prereg07_sha256"], t7.P07_PINS)

    def test_gpu_env(self):
        fc, _ = t7.check_p07_pins()
        ref = fc["environment"]["venv_gpu_reference"]
        env = {k: ref[k] for k in ("python", "torch", "cudnn", "gpu")}
        env["driver"] = "999.99"
        self.assertTrue(t7.check_gpu_env(fc, env, False)["equal"])           # the driver is recorded only
        self.assertEqual(code_of(t7.check_gpu_env, fc, dict(env, torch="2.12.0+cu128"), False), 4)
        self.assertEqual(code_of(t7.check_gpu_env, fc, dict(env, cudnn=92000), False), 4)
        self.assertEqual(code_of(t7.check_gpu_env, fc, dict(env, gpu="NVIDIA GeForce RTX 3090"), False), 2)
        self.assertTrue(t7.check_gpu_env(fc, {}, True)["fake_scorer"])

    def test_bug_revision_below_5(self):
        self.assertEqual(code_of(t7.check_bug_revision, 4, []), 2)
        self.assertIsNone(t7.check_bug_revision(None, []))


# ---------------------------------------------------------------- chunks
class Chunks(Tmp):
    def chunk(self, name, created, env=None, keys="k" * 64, refs=(0, 1)):
        env = env or {"torch": "2.11.0+cu128", "cudnn": "91900", "gpu": t7.GPU_NAME}
        p = os.path.join(self.tmp, name)
        np.savez(p, format=np.array(t7.CHUNK_FORMAT), profiles=np.array(t7.pf7.PROFILES),
                 sides=np.array(t7.pf7.RUN["sides"]), ref=np.array(refs, dtype=np.int64), keys_sha256=np.array(keys),
                 torch=np.array(env["torch"]), cudnn=np.array(env["cudnn"]), gpu_name=np.array(env["gpu"]),
                 ckpt_sha256=np.array(t7.pf7.CKPT_SHA), created_utc=np.array(created))
        return p

    def test_stale_chunks(self):
        self.chunk("chunk_00000.npz", "2026-11-04T06:00:00+00:00")
        self.chunk("chunk_00001.npz", "2026-11-04T08:00:00+00:00")
        st = t7.stale_chunks(self.tmp, "2026-11-04T15:05:00+08:00")             # 07:05 UTC
        self.assertEqual([os.path.basename(p) for p, _ in st], ["chunk_00000.npz"])
        self.assertEqual(t7.stale_chunks(self.tmp, None), [])

    def test_chunk_check(self):
        env = {"torch": "2.11.0+cu128", "cudnn": "91900", "gpu": t7.GPU_NAME}
        p = self.chunk("chunk_00000.npz", "2026-11-05T00:00:00+00:00")
        t7.chunk_check(p, np.array([0, 1]), "k" * 64, env)
        self.assertEqual(code_of(t7.chunk_check, p, np.array([0, 2]), "k" * 64, env), 2)
        self.assertEqual(code_of(t7.chunk_check, p, np.array([0, 1]), "x" * 64, env), 2)
        self.assertEqual(code_of(t7.chunk_check, p, np.array([0, 1]), "k" * 64, dict(env, torch="2.12")), 4)
        self.assertEqual(code_of(t7.chunk_check, p, np.array([0, 1]), "k" * 64, dict(env, cudnn="92000")), 4)


# ---------------------------------------------------------------- the GPU seam
class GpuSeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load = lambda r: dict(np.load(t7.path_of(r), allow_pickle=False))   # noqa: E731
        cls.vh = load("results/prereg07/hs/val_hs.npz")
        cls.vm = load("results/prereg07/hs/val_seam_moves.npz")
        cls.model = t7.pf7.load_hs_map(t7.path_of("results/prereg07/rank/hs_map.npz"))

    def seam(self):
        z = {k: np.array(self.vh[k][:400]) for k in ("question_id", "slot", "color", "sums", "counts", "n_moves",
                                                     "failed")}
        z.update(pos_game=self.vm["pos_game"].copy(), pos_turn=self.vm["pos_turn"].copy(),
                 pos_logp=self.vm["pos_logp"].copy())
        return z

    def test_identical(self):
        r = t7.gpu_seam_record(self.seam(), self.vh, self.vm, self.model)
        self.assertTrue(r["passed"])
        self.assertEqual((r["max_abs_move"], r["max_abs_profile_mean"], r["max_abs_lp_hs"]), (0.0, 0.0, 0.0))
        self.assertTrue(r["bitwise_equal_sums"] and r["bitwise_equal_moves"])
        self.assertEqual(r["moves_compared"], len(self.vm["pos_game"]))

    def test_move_tolerance(self):
        for d, ok in ((0.049, True), (0.051, False)):
            z = self.seam()
            z["pos_logp"][7, 3] += np.float32(d)
            r = t7.gpu_seam_record(z, self.vh, self.vm, self.model)
            self.assertEqual(r["passed"], ok, d)

    def test_profile_mean_tolerance(self):
        g = int(np.flatnonzero(self.vh["counts"][:400] > 0)[0])
        for d, ok in ((0.0019, True), (0.0021, False)):
            z = self.seam()
            z["sums"][g, 1] += d * z["counts"][g]           # profile 1: lp_HS moves 0.0037 for 0.0019 (measured)
            r = t7.gpu_seam_record(z, self.vh, self.vm, self.model)
            self.assertEqual(r["passed"], ok, d)
        z = self.seam()                                      # profile 5: 0.0019 moves lp_HS 0.0214 > 0.02 (measured)
        z["sums"][g, 5] += 0.0019 * z["counts"][g]
        r = t7.gpu_seam_record(z, self.vh, self.vm, self.model)
        self.assertLessEqual(r["max_abs_profile_mean"], 0.002)
        self.assertGreater(r["max_abs_lp_hs"], t7.TOL_LP)
        self.assertFalse(r["passed"])

    def test_counts_and_order(self):
        z = self.seam()
        z["n_moves"][3] += 1
        self.assertFalse(t7.gpu_seam_record(z, self.vh, self.vm, self.model)["counts_equal"])
        z = self.seam()
        z["pos_turn"][0] += 1
        r = t7.gpu_seam_record(z, self.vh, self.vm, self.model)
        self.assertFalse(r["passed"])
        z = {k: v[:399] for k, v in self.seam().items() if k not in ("pos_game", "pos_turn", "pos_logp")}
        z.update(pos_game=self.vm["pos_game"], pos_turn=self.vm["pos_turn"], pos_logp=self.vm["pos_logp"])
        self.assertFalse(t7.gpu_seam_record(z, self.vh, self.vm, self.model)["passed"])

    def test_lp_hs_tolerance(self):
        g = int(np.flatnonzero(self.vh["counts"][:400] > 0)[0])
        z = self.seam()
        z["sums"][g, :] += 0.0019 * z["counts"][g] * np.sign(self.model["W"].sum(1))
        r = t7.gpu_seam_record(z, self.vh, self.vm, self.model)
        self.assertLessEqual(r["max_abs_profile_mean"], 0.002)
        self.assertEqual(r["passed"], r["max_abs_lp_hs"] <= t7.TOL_LP)
        self.assertEqual(r["lp_hs_above_review_line"], r["max_abs_lp_hs"] > t7.LP_REVIEW)


# ---------------------------------------------------------------- helpers
class Helpers(Tmp):
    def test_probs_bytes_equal_t02_writer(self):
        rows = [["pub_q_0001", 5, 5] + [repr(x) for x in (0.5, 0.25, 0.125, 0.0625, 0.0625, 0, 0, 0, 0, 0)],
                ["pub_q_0002", 7, 0] + [""] * 10]
        p = os.path.join(self.tmp, "p.csv")
        t7.t2.write_probs(p, rows)
        with open(p, "rb") as f:
            self.assertEqual(f.read(), t7.probs_bytes(rows))

    def test_t04_bound_paths_outside_real(self):
        S = self.tmp
        ctx = {"real": False, "R4": os.path.join(S, "r4"), "C": os.path.join(S, "d2", "checks"),
               "D2": os.path.join(S, "d2"), "O2": os.path.join(S, "o2")}
        files = {"results/test04/opp/opp_report.json": os.path.join(S, "r4", "opp", "opp_report.json"),
                 "results/test02/rank/apply_test_report.json": os.path.join(S, "o2", "apply_test_report.json"),
                 "results/test02/rank/seam_val/dumps/val_keys.npz": os.path.join(S, "d2", "rank", "seam_val", "dumps",
                                                                                 "val_keys.npz"),
                 "results/test02/checks/rank/t1.json": os.path.join(S, "d2", "checks", "rank", "t1.json")}
        bound = {}
        for k, p in files.items():
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(k.encode())
            bound[k] = t7.pin_sha(p)
        bound["results/prereg04/rank/frozen_config.json"] = t7.pin_sha(t7.t4.FROZEN_CONFIG)
        self.assertEqual(t7.t04_bound_problems(ctx, bound), [])
        with open(files["results/test02/checks/rank/t1.json"], "ab") as f:
            f.write(b"x")
        self.assertEqual(t7.t04_bound_problems(ctx, bound), ["results/test02/checks/rank/t1.json changed"])
        self.assertEqual(t7.t04_bound_problems(ctx, {}), ["t04_apply_report.json lists no inputs_bound"])

    def test_no_guarded_module_loaded(self):
        for m in ("hs_lib", "u1_lib", "pilot_run", "pilot_fuse"):
            self.assertNotIn(m, sys.modules)


if __name__ == "__main__":
    unittest.main()
