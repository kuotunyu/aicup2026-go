"""prereg61: small tests of the wrappers (no score of the method is computed).

    PY    -m pytest -q results/prereg61/explore/test_p61.py      # constants, schedule, E use number
    C10_P61_SMOKE=1 GPUPY -m pytest -q results/prereg61/explore/test_p61.py -k Smoke
        # CPU smoke: 2 steps from c49, then resume to step 3, under data/prereg61/smoke (deleted afterwards)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p61_run as W  # noqa: E402

R, P = W.R, W.P
try:
    import p61_eval as WE
except Exception:                                   # the GPU environment may lack the evaluation stack
    WE = None


class Schedule(unittest.TestCase):
    def test_lr_factor_as_specified(self):
        f = lambda s: P.lr_factor(s, W.WARMUP, W.TOTAL_STEPS)       # noqa: E731
        self.assertEqual((W.TOTAL_STEPS, W.LR, W.WARMUP), (200000, 1e-3, 2000))
        self.assertAlmostEqual(f(0), 1 / 2000.0)                    # step 0: warm-up starts at lr 5e-7
        self.assertAlmostEqual(W.LR * f(0), 5e-7)
        self.assertAlmostEqual(f(1999), 1.0)
        self.assertAlmostEqual(f(2000), 1.0)                        # step 2,000: the peak, lr 1e-3
        self.assertAlmostEqual(W.LR * f(2000), 1e-3)
        self.assertAlmostEqual(f(2000 + 99000), 0.5)
        self.assertGreater(f(199999), 0.0)
        self.assertLess(f(199999), 1e-9)
        self.assertAlmostEqual(f(200000), 0.0)                      # step 200,000: 0
        self.assertTrue(all(f(s) < f(s + 1) for s in range(0, 1999, 37)))
        self.assertTrue(all(f(s) > f(s + 1000) for s in range(2000, 199000, 1000)))

    def test_configure_moves_every_constant(self):
        W.configure()
        d = os.path.join(W.REPO, "data", "prereg61")
        self.assertEqual((R.TOTAL_STEPS, R.LR, R.WARMUP, R.KEEP, R.RESUME_EVERY, R.SEED, R.VARIANT),
                         (200000, 1e-3, 2000, (50000, 100000, 150000, 200000), 1000, 0, "V2"))
        self.assertEqual((R.DATA, R.OUT, R.EMB, R.STOP_FILE),
                         (d, os.path.join(d, "S0"), os.path.join(d, "emb"), os.path.join(d, "STOP")))
        self.assertEqual((R.SEAM, R.SEAM_GPU), (os.path.join(d, "seam"), os.path.join(d, "seam_gpu")))
        self.assertEqual(R.TRAIN_RECORD, os.path.join(HERE, "train_record.json"))
        self.assertEqual(os.path.basename(R.ckpt_file(R.OUT, R.TOTAL_STEPS)), "step_200000.pt")
        self.assertEqual(R.emb_tag(), "p60_step_200000_raw_tta1")
        self.assertIn("p61_run.py", R.code_sha())
        self.assertIn("p60_run.py", R.code_sha())
        for v in (R.DATA, R.OUT, R.EMB, R.SEAM, R.SEAM_GPU, R.STOP_FILE, R.TRAIN_RECORD):
            self.assertNotIn("prereg60", v)

    def test_resume_by_step_and_prune_with_the_new_keep(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("step_050000.pt", "step_099000.pt", "step_100000.pt", "step_101000.pt"):
                open(os.path.join(d, name), "w").close()
            self.assertEqual(P.prune(d, W.KEEP, 101000), ["step_099000.pt"])
            self.assertEqual(os.path.basename(P.last_ckpt(d)), "step_101000.pt")
            self.assertEqual(sorted(os.listdir(d)), ["step_050000.pt", "step_100000.pt", "step_101000.pt"])


@unittest.skipIf(WE is None, "evaluation stack not importable in this environment")
class Eval(unittest.TestCase):
    def test_paths(self):
        E = WE.configure()
        self.assertEqual((WE.STEP, WE.LR, WE.WARMUP), (W.TOTAL_STEPS, W.LR, W.WARMUP))
        self.assertEqual(WE.TAG, W.emb_tag())
        self.assertEqual(os.path.normcase(WE.CKPT), os.path.normcase(R.ckpt_file(os.path.join(W.DATA, "S0"), 200000)))
        self.assertEqual(os.path.normcase(WE.TRAIN_RECORD), os.path.normcase(W.TRAIN_RECORD))
        self.assertEqual((E.STEP, E.TAG, E.CKPT, E.EVAL_JSON, E.TRAIN_RECORD),
                         (200000, WE.TAG, WE.CKPT, WE.EVAL_JSON, WE.TRAIN_RECORD))
        self.assertEqual(E.check_vectors.__defaults__, (WE.TAG,))
        self.assertEqual(E.train_record.__defaults__, (WE.TRAIN_RECORD, WE.CKPT))
        self.assertTrue(E.vec_file("E").replace("\\", "/").endswith(
            "data/prereg61/emb/p60_step_200000_raw_tta1/" + E.P51.vec_name("E")))
        self.assertEqual(str(WE.LAST), "2026-11-23 12:00:00")
        self.assertEqual(WE.E60.V27_E, 0.7035676497986656)
        self.assertEqual((WE.E43.ADOPT_MEAN, WE.E43.BOOT, WE.E43.BOOT_SEED), (0.01, 10000, 43))
        self.assertTrue(WE.EVAL_JSON.replace("\\", "/").endswith("results/prereg61/explore/eval.json"))
        self.assertEqual(len(WE.CODE), 6)

    def test_e_uses_counts_prereg60(self):
        with tempfile.TemporaryDirectory() as d:
            def put(pre, name, obj):
                os.makedirs(os.path.join(d, "results", pre, "explore"), exist_ok=True)
                json.dump(obj, open(os.path.join(d, "results", pre, "explore", name), "w"))
            for pre in ("prereg36", "prereg43", "prereg49"):
                put(pre, "eval.json", {"e_pools_use_number": 1})
            put("prereg51", "eval_B.json", {"e_pools_use_number": 7})
            put("prereg58", "eval.json", {"adopt": False})                 # competition 2: not an E judgement
            self.assertEqual(WE.e_uses_so_far(d)["count"], 4)
            put("prereg60", "eval.json", {"e_pools_use_number": 8})        # prereg60's judgement counts here
            put("prereg60", "seam.json", {"e_pools_use_number": 0})        # not an eval file
            put("prereg61", "eval.json", {"e_pools_use_number": 9})        # its own file never counts
            u = WE.e_uses_so_far(d)
            self.assertEqual(u["later"], ["prereg51/eval_B.json", "prereg60/eval.json"])
            self.assertEqual(u["count"], 5)

    def test_e_uses_on_the_repo(self):
        u = WE.e_uses_so_far()
        self.assertIn("prereg60/eval.json", u["later"])
        self.assertEqual(u["count"], WE.E60.e_uses_so_far()["count"] + 1 -
                         sum(x.startswith("prereg61/") for x in WE.E60.e_uses_so_far()["later"]))

    def test_schedule_ok(self):
        WE.configure()
        cfg = {"total_steps": 200000, "lr": 1e-3, "warmup": 2000, "first_batch": 320000, "n_planes": 18,
               "early_moves": True, "action_plane": True, "fp16": True, "smoke": False}
        good = {"config": {"p60": cfg}, "code_sha256": {"p61_run.py": WE.E60.sha_lf(WE.RUN)}}
        WE.schedule_ok(good)
        for k, v in (("total_steps", 30000), ("lr", 5e-4), ("warmup", 500), ("smoke", True)):
            bad = {"config": {"p60": dict(cfg, **{k: v})}, "code_sha256": good["code_sha256"]}
            with self.assertRaises(SystemExit):
                WE.schedule_ok(bad)
        with self.assertRaises(SystemExit):
            WE.schedule_ok({"config": {"p60": cfg}, "code_sha256": {"p60_run.py": "x"}})
        with self.assertRaises(SystemExit):
            WE.schedule_ok({"config": {"p60": cfg}, "code_sha256": {"p61_run.py": "0" * 64}})


@unittest.skipUnless(os.environ.get("C10_P61_SMOKE") == "1", "set C10_P61_SMOKE=1 (loads c49 on the CPU, ~1 min)")
class Smoke(unittest.TestCase):
    def test_two_steps_then_resume(self):
        smoke = os.path.join(W.DATA, "smoke")
        out = os.path.join(smoke, "S0")
        if os.path.exists(smoke):
            shutil.rmtree(smoke)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, "-u", os.path.join(HERE, "p61_run.py"), "train", "--smoke", "--device", "cpu",
               "--workers", "0", "--min-free-ram-mb", "0"]
        try:
            r = subprocess.run(cmd + ["--steps", "2"], cwd=W.REPO, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(sorted(os.listdir(os.path.join(out, "ckpt"))), ["step_000002.pt"])
            r = subprocess.run(cmd + ["--steps", "3", "--auto-resume"], cwd=W.REPO, env=env, capture_output=True,
                               text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("resumed from", r.stdout)
            self.assertIn("(step 2)", r.stdout)
            self.assertEqual(sorted(os.listdir(os.path.join(out, "ckpt"))), ["step_000003.pt"])
            rows = [json.loads(x) for x in open(os.path.join(out, "train_log.jsonl"), encoding="utf-8")]
            self.assertEqual([x["step"] for x in rows], [2, 3])
            # lr after optimiser step n (0-based steps 0 .. n-1 done) = LR * (n + 1) / WARMUP during warm-up
            self.assertAlmostEqual(rows[0]["lr"], 1e-3 * 3 / 2000, places=12)
            self.assertAlmostEqual(rows[1]["lr"], 1e-3 * 4 / 2000, places=12)
            rec = json.load(open(os.path.join(out, "train_record.json"), encoding="utf-8"))
            c = rec["config"]["p60"]
            self.assertEqual((c["total_steps"], c["lr"], c["warmup"], c["first_batch"], c["n_planes"], c["smoke"]),
                             (200000, 1e-3, 2000, 320000, 18, True))
            self.assertEqual(c["base_ckpt_sha256"], R.C49_SHA)
            self.assertEqual((rec["steps"], rec["finished"], rec["smoke"]), (3, False, True))
            self.assertEqual(rec["model"], "raw weights of step_200000.pt")
            self.assertIn("p61_run.py", rec["code_sha256"])
            self.assertFalse(os.path.exists(W.TRAIN_RECORD) and
                             json.load(open(W.TRAIN_RECORD, encoding="utf-8")).get("smoke"))
            print("\nSMOKE " + json.dumps({"log": rows, "seconds": rec["seconds"],
                                           "positions": rec["counts"]["positions_seen"]}))
        finally:
            shutil.rmtree(smoke, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
