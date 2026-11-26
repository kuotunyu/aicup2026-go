"""Unit tests of tools/t08_rank.py (PREREG_08 5.2, rev 3 section 13; negative drills at function level). CPU, fast.

Never imports hs_lib, u1_lib, pilot_* or p1_* in the pytest process (the autouse fixture checks the guards stayed out);
the paths that need them run as subprocesses in tests/test_t08_dryrun.py. Reads only committed results files
(results/prereg08/feat/val_seam_feats.npz, results/prereg08/rank/p1_map.npz, val_lp_p1.npz, score_H8R.json); never
data/local, never a label file, never tools/score.py; temporary files under data/test08/tmp only (removed).
Run: .venv/Scripts/python.exe -m pytest -q tests/test_t08_rank.py
"""

import argparse
import builtins
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
import t08_rank as t8  # noqa: E402

TMP_ROOT = os.path.join(REPO, "data", "test08", "tmp")
GUARDED = ("hs_lib", "u1_lib", "pilot_run", "pilot_fuse", "p1_lib", "p1_fuse", "p1_extract")


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process"


def ns(**kw):
    base = dict(dry_run=False, rehearsal=False, results_dir=None, p1_data_dir=None, t07_results_dir=None,
                t04_results_dir=None, t02_results_dir=None, t02_out_dir=None, tests_dir=None, checks_dir=None,
                r20_uploaded_at=None, now=None, max_games=0, fake_scorer=False, ignore_other_gpu_jobs=False)
    base.update(kw)
    return argparse.Namespace(**base)


def code_of(fn, *a, **k):
    with pytest.raises(t8.T08Error) as ei:
        fn(*a, **k)
    return ei.value.code


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(TMP_ROOT, "unit_" + uuid.uuid4().hex[:8])
        os.makedirs(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for d in (TMP_ROOT, os.path.dirname(TMP_ROOT)):     # leave no empty data/test08 behind
            try:
                os.rmdir(d)
            except OSError:
                pass


# ---------------------------------------------------------------- modes
class Resolve(Tmp):
    def outside(self, name):
        return os.path.join(os.path.dirname(REPO), "c10_t08_unit_outside", name)

    def trees(self, **kw):
        d = dict(t07_results_dir=self.outside("r7"), t04_results_dir=self.outside("r4"),
                 t02_results_dir=self.outside("d2"), t02_out_dir=self.outside("o2"), tests_dir=self.outside("t"))
        d.update(kw)
        return d

    def test_real_mode_paths(self):
        c = t8.resolve(ns(), "apply")
        self.assertEqual((c["mode"], c["R8"], c["H"], c["R7"], c["R4"]),
                         ("real", t8.REAL_R8, t8.REAL_H, t8.REAL_R7, t8.REAL_R4))
        self.assertEqual(c["write_roots"], [t8.REAL_R8, t8.REAL_H, t8.REAL_TMP])

    def test_real_mode_refuses_drill_flags(self):
        for kw in ({"max_games": 5}, {"fake_scorer": True}, {"now": "2026-11-20T00:00"},
                   {"r20_uploaded_at": "2026-11-04T15:00"}, {"ignore_other_gpu_jobs": True}):
            with self.subTest(kw):
                self.assertEqual(code_of(t8.resolve, ns(**kw), "p1"), 2)

    def test_real_mode_other_dirs_refused(self):
        self.assertEqual(code_of(t8.resolve, ns(results_dir=self.outside("r8")), "apply"), 2)
        self.assertEqual(code_of(t8.resolve, ns(p1_data_dir=self.outside("h")), "apply"), 2)
        self.assertEqual(code_of(t8.resolve, ns(t07_results_dir=self.outside("r7")), "apply"), 2)

    def test_dry_run_needs_trees_outside_the_repo(self):
        self.assertEqual(code_of(t8.resolve, ns(dry_run=True, **self.trees()), "apply"), 2)      # no --results-dir
        self.assertEqual(code_of(t8.resolve, ns(dry_run=True, **self.trees(t07_results_dir=None),
                                                results_dir=self.outside("r8")), "apply"), 2)
        self.assertEqual(code_of(t8.resolve, ns(dry_run=True, results_dir=os.path.join(REPO, "results", "x"),
                                                **self.trees()), "apply"), 2)
        c = t8.resolve(ns(dry_run=True, results_dir=self.outside("r8"), **self.trees()), "apply")
        self.assertEqual(c["H"], os.path.join(self.outside("r8"), "p1_data"))
        self.assertEqual(c["mode"], "dry-run")

    def test_rehearsal_default_folder(self):
        c = t8.resolve(ns(rehearsal=True, **self.trees()), "apply")
        self.assertEqual((c["R8"], c["H"], c["TMP"]), (t8.REHEARSAL_R8, t8.REHEARSAL_H, t8.REHEARSAL_TMP))

    def test_tests_dir_raw_and_local_refused(self):
        for t in (os.path.join(self.outside("t"), "raw"), os.path.join(REPO, "data", "local")):
            with self.subTest(t):
                self.assertEqual(code_of(t8.resolve, ns(dry_run=True, results_dir=self.outside("r8"),
                                                        **self.trees(tests_dir=t)), "apply"), 2)

    def test_forbidden_writes(self):
        for rel in ("results/prereg08/rank/x.json", "results/test07/rank/x.csv", "results/test04/rank/x.csv",
                    "results/test02/x", "data/local/x", "results/score_ledger.csv", "results/test06/x"):
            with self.subTest(rel):
                self.assertIsNotNone(t8.forbidden_area(os.path.join(REPO, *rel.split("/"))))
        self.assertIsNone(t8.forbidden_area(os.path.join(REPO, "results", "test08", "rank", "x.csv")))
        ctx = {"write_roots": [self.tmp]}
        self.assertEqual(code_of(t8.guard_write, ctx, os.path.join(REPO, "results", "test08", "x")), 2)
        self.assertEqual(t8.guard_write(ctx, os.path.join(self.tmp, "a")), os.path.join(self.tmp, "a"))


# ---------------------------------------------------------------- registered decisions
class Rules(Tmp):
    def test_rate_rule(self):
        self.assertTrue(t8.rate_ok(25, 5000))
        self.assertFalse(t8.rate_ok(26, 5000))
        self.assertTrue(t8.rate_ok(23, 4787))
        self.assertFalse(t8.rate_ok(24, 4787))
        self.assertFalse(t8.rate_ok(0, 0))

    def test_constants(self):
        self.assertEqual((t8.TOL_FEAT, t8.TOL_LP, t8.RHO_FLOOR, t8.P08_MIN_REV), (0.05, 0.04, 0.155, 4))
        self.assertEqual(t8.K6_08.isoformat(), "2026-11-19T23:59:59+08:00")
        self.assertEqual(t8.METHOD, "prereg08 Q08 = Q02H + 4 x P1TO, rule B (甲)")

    def test_k6_08(self):
        ctx = {"real": False, "mode": "dry-run", "rank": self.tmp}
        rec = t8.check_k6(ctx, ns(now="2026-11-19T23:59:59+08:00"))
        self.assertFalse(rec["passed"])
        self.assertEqual(code_of(t8.check_k6, ctx, ns(now="2026-11-20T00:00:00+08:00")), 3)
        with open(os.path.join(self.tmp, "t08_apply_report.json"), "w", encoding="utf-8") as f:
            json.dump({"exit_code": 0}, f)
        self.assertEqual(code_of(t8.check_k6, ctx, ns(now="2026-11-20T00:00:00+08:00")), 2)

    def test_h8r_condition(self):
        rec = t8.h8r_condition()
        self.assertTrue(rec["holds"])
        self.assertEqual(rec["questions"], 1589)
        real = t8.path_of("results/prereg08/rank/score_H8R.json")
        with open(real, encoding="utf-8") as f:
            s = json.load(f)
        fake = os.path.join(self.tmp, "score_H8R.json")
        orig = t8.path_of
        lo0 = json.loads(json.dumps(s))
        lo0["B_minus_A_rank_balanced"]["ci95"][0] = 0.0
        cases = [dict(s, A_rank_balanced=s["A_rank_balanced"] + 2e-12), dict(s, B_rank_balanced=0.5699),
                 dict(s, A_mean=s["A_mean"] - 2e-12), lo0]
        for i, obj in enumerate(cases):
            with self.subTest(i):
                with open(fake, "w", encoding="utf-8") as f:
                    json.dump(obj, f)
                t8.path_of = lambda rel: fake if rel.endswith("score_H8R.json") else orig(rel)
                try:
                    self.assertEqual(code_of(t8.h8r_condition), 3)
                finally:
                    t8.path_of = orig
        t8.path_of = lambda rel: os.path.join(self.tmp, "missing.json") if rel.endswith("score_H8R.json") else orig(rel)
        try:
            self.assertEqual(code_of(t8.h8r_condition), 3)
        finally:
            t8.path_of = orig

    def test_pins_hold(self):
        fc, model, rec = t8.check_p08_pins()
        self.assertEqual(fc["combo"]["id"], "Q08")
        self.assertEqual(rec["prereg08_sha256"], t8.P08_PINS)
        self.assertEqual(model["scale"], 4.0)
        self.assertEqual(float(fc["p1_direction_ref"]["rho_mono"]), t8.RHO_MONO)

    def test_gpu_env(self):
        fc, _m, _r = t8.check_p08_pins()
        ref = fc["environment"]["venv_gpu_reference"]
        env = {k: ref[k] for k in ("python", "torch", "cudnn", "gpu")}
        env["driver"] = "999.99"
        self.assertTrue(t8.check_gpu_env(fc, env, False)["equal"])           # the driver is recorded only
        self.assertEqual(code_of(t8.check_gpu_env, fc, dict(env, torch="2.12.0+cu128"), False), 4)
        self.assertEqual(code_of(t8.check_gpu_env, fc, dict(env, cudnn=92000), False), 4)
        self.assertEqual(code_of(t8.check_gpu_env, fc, dict(env, python="3.12.14"), False), 4)
        self.assertEqual(code_of(t8.check_gpu_env, fc, dict(env, gpu="NVIDIA GeForce RTX 3090"), False), 2)
        self.assertTrue(t8.check_gpu_env(fc, {}, True)["fake_scorer"])

    def test_bug_revision_below_4(self):
        self.assertEqual(code_of(t8.check_bug_revision, 3, []), 2)
        self.assertIsNone(t8.check_bug_revision(None, []))

    def test_blas_env(self):
        old = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "4"
        try:
            self.assertEqual(code_of(t8.blas_env_check), 2)
        finally:
            if old is None:
                os.environ.pop("OMP_NUM_THREADS")
            else:
                os.environ["OMP_NUM_THREADS"] = old


# ---------------------------------------------------------------- chunks
class Chunks(Tmp):
    CODE = json.dumps({"results/prereg08/p1/p1_lib.py": "a" * 64}, sort_keys=True)

    def chunk(self, name, created, env=None, keys="k" * 64, refs=(0, 1), mode="dry-run", fmt=None, **over):
        env = env or {"torch": "2.11.0+cu128", "cudnn": "91900", "gpu": t8.GPU_NAME}
        p = os.path.join(self.tmp, name)
        meta = dict(questions_sha256="q" * 64, code_sha256=self.CODE, ckpt_sha256=t8.r8.CKPT_SHA,
                    precision=t8.r8.RUN["precision"], batch=t8.r8.RUN["batch"],
                    channels_last=t8.r8.RUN["channels_last"], device=t8.r8.RUN["device"])
        meta.update(over)
        np.savez(p, format=np.array(fmt or t8.FORMATS["chunk"]), sides=np.array(t8.r8.RUN["sides"]),
                 ref=np.array(refs, dtype=np.int64), keys_sha256=np.array(keys),
                 meta_sha256=np.array(t8.r8.NEUTRAL_META_SHA), mode=np.array(mode),
                 means=np.zeros((len(refs), 2, t8.F_DIM), dtype=np.float32),
                 torch=np.array(env["torch"]), cudnn=np.array(env["cudnn"]), gpu_name=np.array(env["gpu"]),
                 created_utc=np.array(created), **{k: np.array(v) for k, v in meta.items()})
        return p

    def cc(self, p, refs=(0, 1), keys="k" * 64, env=None, mode="dry-run", **kw):
        env = env or {"torch": "2.11.0+cu128", "cudnn": "91900", "gpu": t8.GPU_NAME}
        kw = dict(dict(questions_sha="q" * 64, code_sha=self.CODE, fake=False), **kw)
        return t8.chunk_check(p, np.array(refs), keys, env, mode, **kw)

    def test_stale_chunks(self):
        self.chunk("chunk_00000.npz", "2026-11-04T06:00:00+00:00")
        self.chunk("chunk_00001.npz", "2026-11-04T08:00:00+00:00")
        st = t8.stale_chunks(self.tmp, "2026-11-04T15:05:00+08:00")             # 07:05 UTC
        self.assertEqual([os.path.basename(p) for p, _ in st], ["chunk_00000.npz"])
        self.assertEqual(t8.stale_chunks(self.tmp, None), [])

    def test_chunk_check(self):
        env = {"torch": "2.11.0+cu128", "cudnn": "91900", "gpu": t8.GPU_NAME}
        p = self.chunk("chunk_00000.npz", "2026-11-05T00:00:00+00:00")
        self.cc(p)
        self.assertEqual(code_of(self.cc, p, refs=(0, 2)), 2)
        self.assertEqual(code_of(self.cc, p, keys="x" * 64), 2)
        self.assertEqual(code_of(self.cc, p, mode="real"), 2)                                         # mode
        self.assertEqual(code_of(self.cc, p, env=dict(env, torch="2.12")), 4)
        self.assertEqual(code_of(self.cc, p, env=dict(env, cudnn="92000")), 4)
        q = self.chunk("chunk_00001.npz", "2026-11-05T00:00:00+00:00", fmt="c10-prereg08 val chunk v1")
        self.assertEqual(code_of(self.cc, q), 2)

    def test_chunk_check_questions_code_and_run_settings(self):
        """rev 3 section 13 addendum: a chunk of another question file, P1 code, checkpoint or scorer settings is
        never reused (2)."""
        p = self.chunk("chunk_00000.npz", "2026-11-05T00:00:00+00:00")
        self.assertEqual(code_of(self.cc, p, questions_sha="z" * 64), 2)
        self.assertEqual(code_of(self.cc, p, code_sha=json.dumps({"results/prereg08/p1/p1_lib.py": "b" * 64})), 2)
        self.assertEqual(code_of(self.cc, p, fake=True), 2)                  # fake-scorer run vs a GPU chunk
        for i, over in enumerate(({"precision": "fp32"}, {"batch": 512}, {"channels_last": False}, {"device": "cpu"},
                                  {"ckpt_sha256": "0" * 64})):
            q = self.chunk(f"chunk_{i + 1:05d}.npz", "2026-11-05T00:00:00+00:00", **over)
            self.assertEqual(code_of(self.cc, q), 2, over)
        f = self.chunk("chunk_00009.npz", "2026-11-05T00:00:00+00:00", ckpt_sha256="fake", precision="fake-fp32",
                       device="cpu", env={"torch": "fake", "cudnn": "91900", "gpu": t8.GPU_NAME})
        self.cc(f, env={"torch": "fake", "cudnn": "91900", "gpu": t8.GPU_NAME}, fake=True)
        self.assertEqual(code_of(self.cc, f, env={"torch": "fake", "cudnn": "91900", "gpu": t8.GPU_NAME}), 2)

    def test_hs_keys_match_is_equality(self):
        os.makedirs(os.path.join(self.tmp, "hs"))
        keys = {"question_id": np.array(["q1", "q1", "q2"]), "slot": np.array([1, 2, 1]),
                "color": np.array(["B", "W", "B"])}
        np.savez(os.path.join(self.tmp, "hs", "seam_hs.npz"), **keys)
        ctx = {"R7": self.tmp}
        self.assertTrue(t8.hs_keys_match(ctx, "seam", keys)["equal"])
        short = {k: v[:2] for k, v in keys.items()}                           # fewer rows than 07's file: refused
        self.assertEqual(code_of(t8.hs_keys_match, ctx, "seam", short), 2)
        longer = {k: np.concatenate([v, v[:1]]) for k, v in keys.items()}
        self.assertEqual(code_of(t8.hs_keys_match, ctx, "seam", longer), 2)
        bad = dict(keys, color=np.array(["B", "B", "B"]))
        self.assertEqual(code_of(t8.hs_keys_match, ctx, "seam", bad), 2)

    def test_stop_exit_code(self):
        self.assertEqual(t8.EXIT_STOPPED, 5)
        self.assertNotIn(t8.EXIT_STOPPED, (0, 2, 3, 4))


# ---------------------------------------------------------------- the GPU seam
class GpuSeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load = lambda r: dict(np.load(t8.path_of(r), allow_pickle=False))   # noqa: E731
        cls.ref = load("results/prereg08/feat/val_seam_feats.npz")
        cls.model = load("results/prereg08/rank/p1_map.npz")
        cls.lpz = load("results/prereg08/rank/val_lp_p1.npz")

    def seam(self):
        return {k: np.array(v) for k, v in self.ref.items()}

    def test_numpy_port_matches_the_committed_lp(self):
        ok = ~self.ref["failed"] & (self.ref["counts"][:, 0] > 0)
        x = t8.std_x(self.ref["means"], self.ref["counts"], self.model["mu"], self.model["sd"])[ok]
        got = t8.lp_np(x, self.model)
        want = self.lpz["lp"][:400][ok]
        self.assertLess(float(np.max(np.abs(got - want))), 1e-12)

    def test_identical(self):
        r = t8.gpu_seam_record(self.seam(), self.ref, self.model)
        self.assertTrue(r["passed"])
        self.assertEqual((r["max_abs_std_feature"], r["max_abs_lp_p1"]), (0.0, 0.0))
        self.assertTrue(r["bitwise_equal_means"])
        self.assertEqual(r["games"], 400)

    def test_feature_tolerance(self):
        g = int(np.flatnonzero(self.ref["counts"][:, 0] > 0)[0])
        j = int(np.argmin(np.abs(self.model["W"][:770]).sum(1)))          # the feature with the least weight
        for d, ok in ((0.049, True), (0.051, False)):
            z = self.seam()
            z["means"][g, 0, j] += np.float32(d * self.model["sd"][j])
            r = t8.gpu_seam_record(z, self.ref, self.model)
            self.assertLess(abs(r["max_abs_std_feature"] - d), 1e-3)
            self.assertLessEqual(r["max_abs_lp_p1"], t8.TOL_LP)
            self.assertEqual(r["passed"], ok, d)

    def test_lp_tolerance(self):
        g = int(np.flatnonzero(self.ref["counts"][:, 0] > 0)[0])
        z = self.seam()
        w = self.model["W"][:770, 0] - self.model["W"][:770, 9]            # pushes level 0 against level 9
        z["means"][g, 0, :] += (0.04 * np.sign(w) * self.model["sd"][:770]).astype(np.float32)
        r = t8.gpu_seam_record(z, self.ref, self.model)
        self.assertLessEqual(r["max_abs_std_feature"], t8.TOL_FEAT)
        self.assertGreater(r["max_abs_lp_p1"], t8.TOL_LP)
        self.assertFalse(r["passed"])
        self.assertTrue(r["lp_above_review_line"])

    def test_counts_and_keys(self):
        for k, fn in (("n_moves", lambda z: z["n_moves"].__setitem__(3, z["n_moves"][3] + 1)),
                      ("counts", lambda z: z["counts"].__setitem__((3, 1), z["counts"][3, 1] + 1)),
                      ("failed", lambda z: z["failed"].__setitem__(3, True)),
                      ("slot", lambda z: z["slot"].__setitem__(3, z["slot"][3] + 1))):
            with self.subTest(k):
                z = self.seam()
                fn(z)
                r = t8.gpu_seam_record(z, self.ref, self.model)
                self.assertFalse(r["counts_equal"])
                self.assertFalse(r["passed"])
        z = {k: v[:399] for k, v in self.seam().items() if k not in ("games", "chunk_sha256")}
        self.assertFalse(t8.gpu_seam_record(z, self.ref, self.model)["passed"])


# ---------------------------------------------------------------- helpers
class Helpers(unittest.TestCase):
    def test_cross_check(self):
        p = {"n_moves": np.array([100, 120, 0]), "counts": np.array([[50, 50], [60, 60], [0, 0]]),
             "failed": np.array([False, False, True])}
        hs = {"n_moves": np.array([100, 120, 5]), "counts": np.array([50, 60, 2]), "failed": np.array([False] * 3)}
        o4 = {"moves_parsed": np.array([100, 120, 5]), "n_moves": np.zeros((3, 2, 4), dtype=np.int64),
              "replay_failed": np.array([False] * 3)}
        o4["n_moves"][:, 0, 0] = [50, 60, 2]
        r = t8.p1_cross_check(p, hs, o4, ["a", "b", "c"])
        self.assertEqual((r["mismatches"], r["vs_t07_hs"]["compared"]), (0, 2))
        hs["n_moves"][1] += 1
        r = t8.p1_cross_check(p, hs, o4, ["a", "b", "c"])
        self.assertEqual(r["mismatches"], 1)
        self.assertEqual(r["vs_t07_hs"]["first_mismatches"][0]["id"], "b")

    def test_seam_verdict_is_07s(self):
        base = {"problems": [], "count_mismatch_questions": [], "byte_identical_probs": True,
                "byte_identical_pred_B": True, "passed": True}
        self.assertEqual(t8.seam_verdict(base, True), "passed")
        self.assertEqual(t8.seam_verdict(dict(base, byte_identical_probs=False), True), "drift")
        self.assertEqual(t8.seam_verdict(dict(base, byte_identical_probs=False, passed=False), False), "not_used")

    def test_no_guarded_module_loaded(self):
        for m in GUARDED:
            self.assertNotIn(m, sys.modules)


if __name__ == "__main__":
    unittest.main()
