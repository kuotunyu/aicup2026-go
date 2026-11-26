"""C10_SLOW=1 dry run of tools/t25_rank.py (PREREG_25 4.2) on the committed test-day trees (read only): the real
results/test02-test19 outputs and data/test08/p1, data/test; the test25 output goes to a temporary folder outside the
repository. Never opens data/local (t25 never does), never runs tools/score.py, never writes into the repository.

Drills: the normal run (exit 0, under an audit hook: no path under data/local opened, listed or stat-ed; Q09 and Q19 recomputed equal to test09 / test19 byte for byte; check_submission
official; equal to the committed results/test25/rank outputs once they exist); outputs already there (2); K6-25 passed
without a success (3, not_used.json) and the rerun gate on it (2 without / with an unregistered --bug-revision); a
changed 19 report (2); Q19 drift (a copied test19 tree whose rank_test_probs.csv differs by one value; 4,
attempt_verdict.json); the SET25 direction failing (3, through a subprocess that raises the floor); BLAS variables
(2); status.

Run: C10_SLOW=1 [C10_DRYRUN_DIR=<a folder whose path has no 'AppData\\Local', e.g. a subst drive>]
     python -m pytest -q -p no:cacheprovider tests/test_t25_dryrun.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable
TOOL = os.path.join(REPO, "tools", "t25_rank.py")
SLOW = os.environ.get("C10_SLOW") == "1"
R = lambda *p: os.path.join(REPO, *p)   # noqa: E731
DIRS = {"--t19-results-dir": R("results", "test19"), "--t09-results-dir": R("results", "test09"),
        "--t08-results-dir": R("results", "test08"), "--p1-data-dir": R("data", "test08", "p1"),
        "--t07-results-dir": R("results", "test07"), "--t04-results-dir": R("results", "test04"),
        "--t02-results-dir": R("results", "test02"), "--t02-out-dir": R("results", "test02", "rank"),
        "--tests-dir": R("data", "test"), "--checks-dir": R("results", "test02", "checks")}


def env(**extra):
    e = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                          "MKL_NUM_THREADS")}
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e.update(extra)
    return e


@unittest.skipUnless(SLOW, "C10_SLOW=1 runs the test25 dry run (about 2 minutes)")
class DryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = os.environ.get("C10_DRYRUN_DIR") or tempfile.gettempdir()
        if "appdata" in base.lower() and "local" in base.lower():
            raise unittest.SkipTest("the dry-run folder path holds 'AppData\\Local' (the data/local guards refuse it): "
                                    "set C10_DRYRUN_DIR to a subst drive")
        cls.tmp = tempfile.mkdtemp(prefix="t25dry_", dir=base)
        if cls.tmp.lower().startswith(os.path.abspath(REPO).lower()):
            raise unittest.SkipTest("the dry-run folder must be outside the repository")

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("C10_KEEP") != "1":
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_t25(self, cmd, r25, *extra, dirs=None, e=None):
        args = [PY, TOOL, cmd, "--dry-run", "--results-dir", r25]
        for k, v in (dirs or DIRS).items():
            args += [k, v]
        p = subprocess.run(args + list(extra), capture_output=True, text=True, encoding="utf-8", cwd=REPO,
                           env=e or env(), timeout=1800)
        return p.returncode, p.stdout + p.stderr

    def r25(self, name):
        return os.path.join(self.tmp, name)

    def test_1_normal_and_rerun(self):
        out_dir = self.r25("normal")
        # an audit hook records every opened / listed / scanned path: none may be under data/local (PREREG_25 4.2)
        code = ("import os, sys; sys.path.insert(0, r'%s'); hits = []\n"
                "def hook(ev, args):\n"
                "    if ev in ('open', 'os.listdir', 'os.scandir', 'os.stat') and args and "
                "isinstance(args[0], (str, bytes, os.PathLike)):\n"
                "        p = os.fsdecode(args[0]).replace(chr(92), '/').lower()\n"
                "        if '/data/local' in p or p.startswith('data/local'):\n"
                "            hits.append(p)\n"
                "sys.addaudithook(hook)\n"
                "import t25_rank as t\n"
                "rc = t.main(sys.argv[1:])\n"
                "print('DATA_LOCAL_HITS', len(hits), hits[:3])\n"
                "sys.exit(rc)\n") % os.path.join(REPO, "tools")
        args = [PY, "-c", code, "apply", "--dry-run", "--results-dir", out_dir]
        for k, v in DIRS.items():
            args += [k, v]
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", cwd=REPO, env=env(), timeout=1800)
        rc, out = p.returncode, p.stdout + p.stderr
        self.assertEqual(rc, 0, out)
        self.assertIn("DATA_LOCAL_HITS 0 []", out)
        with open(os.path.join(out_dir, "rank", "t25_apply_report.json"), encoding="utf-8") as f:
            rep = json.load(f)
        self.assertEqual((rep["exit_code"], rep["usable"], rep["mode"]), (0, True, "dry-run"))
        self.assertTrue(rep["q09_vs_09"]["equal"] and rep["q09_vs_09"]["byte_equal"])
        self.assertTrue(rep["q19_vs_19"]["equal"] and rep["q19_vs_19"]["byte_equal"])
        self.assertTrue(rep["q19_vs_19"]["lp_set19"]["equal"])
        self.assertEqual(rep["check_submission"]["errors"], [])
        self.assertTrue(all(rep["set_direction"][s]["passed"] for s in ("public", "private")))
        self.assertEqual(rep["decide_summary"]["n_questions"], 800)
        self.assertIn("19", rep["rule_b_differences"])
        real = R("results", "test25", "rank")
        if os.path.exists(os.path.join(real, "t25_apply_report.json")):   # after the official run: the same file
            for n in ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "submission_rank.csv", "test_lp_set.npz"):
                with open(os.path.join(real, n), "rb") as f1, open(os.path.join(out_dir, "rank", n), "rb") as f2:
                    self.assertEqual(f1.read(), f2.read(), n)
        rc, out = self.run_t25("apply", out_dir)
        self.assertEqual(rc, 2, out)
        self.assertIn("outputs already exist", out)
        rc, out = self.run_t25("status", out_dir)
        self.assertEqual(rc, 0, out)
        self.assertIn("apply: exit 0", out)

    def test_2_k6_passed_and_rerun_gate(self):
        out_dir = self.r25("k6")
        rc, out = self.run_t25("apply", out_dir, "--now", "2026-11-19T00:00:00+08:00")
        self.assertEqual(rc, 3, out)
        with open(os.path.join(out_dir, "not_used.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["reason"], "k6_25")
        rc, out = self.run_t25("apply", out_dir, "--rerun", "--reason", "drill")
        self.assertEqual(rc, 2, out)
        self.assertIn("--bug-revision", out)
        rc, out = self.run_t25("apply", out_dir, "--rerun", "--reason", "drill", "--bug-revision", "4")
        self.assertEqual(rc, 2, out)
        self.assertIn("rev 4", out)

    def copy19(self, name):
        dst = os.path.join(self.tmp, name, "test19")
        shutil.copytree(R("results", "test19"), dst)
        return dst

    def test_3_changed_19_report(self):
        r19 = self.copy19("bad19")
        p = os.path.join(r19, "rank", "t19_apply_report.json")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        rep["exit_code"] = 3
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
        rc, out = self.run_t25("apply", self.r25("bad19_out"), dirs=dict(DIRS, **{"--t19-results-dir": r19}))
        self.assertEqual(rc, 2, out)
        self.assertIn("exit_code", out)

    def test_4_q19_drift(self):
        r19 = self.copy19("drift19")
        p = os.path.join(r19, "rank", "rank_test_probs.csv")
        with open(p, "rb") as f:
            lines = f.read().split(b"\n")
        cells = lines[1].split(b",")
        cells[3] = repr(float(cells[3]) + 1e-9).encode()
        lines[1] = b",".join(cells)
        with open(p, "wb") as f:
            f.write(b"\n".join(lines))
        import hashlib
        rp = os.path.join(r19, "rank", "t19_apply_report.json")
        with open(rp, encoding="utf-8") as f:
            rep = json.load(f)
        with open(p, "rb") as f:
            rep["outputs_sha256"]["rank_test_probs.csv"] = hashlib.sha256(f.read()).hexdigest()
        with open(rp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
        out_dir = self.r25("drift_out")
        rc, out = self.run_t25("apply", out_dir, dirs=dict(DIRS, **{"--t19-results-dir": r19}))
        self.assertEqual(rc, 4, out)
        with open(os.path.join(out_dir, "rank", "attempt_verdict.json"), encoding="utf-8") as f:
            v = json.load(f)
        self.assertEqual((v["exit_code"], v["stage"]), (4, "q19_vs_19"))
        self.assertFalse(os.path.exists(os.path.join(out_dir, "rank", "submission_rank.csv")))

    def test_5_direction_fails(self):
        out_dir = self.r25("direction")
        code = ("import sys; sys.path.insert(0, r'%s'); import t25_rank as t; t.RHO_FLOOR = 0.99; "
                "sys.exit(t.main(sys.argv[1:]))" % os.path.join(REPO, "tools"))
        args = [PY, "-c", code, "apply", "--dry-run", "--results-dir", out_dir]
        for k, v in DIRS.items():
            args += [k, v]
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", cwd=REPO, env=env(), timeout=1800)
        self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
        with open(os.path.join(out_dir, "not_used.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["stage"], "set_direction")

    def test_6_blas(self):
        rc, out = self.run_t25("apply", self.r25("blas"), e=env(OMP_NUM_THREADS="2"))
        self.assertEqual(rc, 2, out)
        self.assertIn("BLAS", out)


if __name__ == "__main__":
    unittest.main()
