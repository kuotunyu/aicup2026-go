"""The 21 test-day rehearsal on fake trees (PREREG_21 section 6; docs/TESTDAY_21.md 'rehearsal'). Slow: runs only with
C10_SLOW=1. Uses fit training players only (no validation player, no data/local).

* Copies: tools/t21_copy.py fake builds the 800-question official-shaped tree of section 6; scan and apply in --dry-run
  must reproduce the committed results/test21/design/sim.json (overridable count, overrides, changed rows, output
  SHA-256 of submission_player.csv and overrides.csv, base SHA-256) and its pass conditions; a second apply without
  --rerun is refused (2); status runs.
* NoCopies: fake --no-copies: scan and apply give the registered 'not used' (3), not_used.json, no submission file.
* BaseDrift: the base file changed after it was logged: apply is local drift (4) and writes no submission.
The tree goes to C10_DRYRUN_DIR (outside the repository) when set, else a temporary folder.
Run: C10_SLOW=1 python -m pytest -q tests/test_t21_dryrun.py
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t21_copy as tc  # noqa: E402

SLOW = os.environ.get("C10_SLOW") == "1"


def run(*argv):
    with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
        return tc.main(list(argv))


@unittest.skipUnless(SLOW, "C10_SLOW=1 only (about 2 minutes)")
class DryRun(unittest.TestCase):
    def setUp(self):
        parent = os.environ.get("C10_DRYRUN_DIR") or None
        self.root = tempfile.mkdtemp(prefix="c10_t21_dry_", dir=parent)
        self.addCleanup(shutil.rmtree, self.root, True)
        self.out = lambda *p: os.path.join(self.root, "results", "test21", *p)   # noqa: E731

    def test_copies_reproduce_sim(self):
        sim = tc.read_json(tc.SIM)
        self.assertEqual(run("fake", "--root", self.root), 0)
        self.assertEqual(run("scan", "--dry-run", "--root", self.root), 0)
        scan = tc.read_json(self.out("scan", "scan.json"))
        self.assertEqual(scan["questions_overridable"], sim["outputs_sha256"]["scan_questions_overridable"])
        self.assertEqual(run("apply", "--dry-run", "--root", self.root), 0)
        rep = tc.read_json(self.out("player", "t21_copy_report.json"))
        for k in ("questions_overridden", "rows_changed", "overridden_by_pool", "changed_by_pool"):
            self.assertEqual(rep[k], sim["apply"][k], k)
        self.assertEqual(len(rep["conflicts"]), sim["apply"]["conflicts"])
        self.assertEqual(rep["outputs_sha256"]["submission_player.csv"], sim["outputs_sha256"]["submission_player.csv"])
        self.assertEqual(rep["outputs_sha256"]["overrides.csv"], sim["outputs_sha256"]["overrides.csv"])
        base = os.path.join(self.root, "results", "test10", "player", "submission_player.csv")
        self.assertEqual(tc.sha256_file(base), sim["outputs_sha256"]["base_submission_player_csv"])
        ev = tc.evaluate(tc.read_json(os.path.join(self.root, "t21_expected.json")), rep, base,
                         self.out("player", "submission_player.csv"))
        self.assertTrue(all(ev["conditions"].values()), ev)
        self.assertEqual(ev["by_kind"], sim["evaluation"]["by_kind"])
        self.assertEqual(run("apply", "--dry-run", "--root", self.root), 2)
        self.assertEqual(run("status", "--dry-run", "--root", self.root), 0)

    def test_no_copies_not_used(self):
        self.assertEqual(run("fake", "--root", self.root, "--no-copies"), 0)
        self.assertEqual(run("scan", "--dry-run", "--root", self.root), 3)
        self.assertEqual(tc.read_json(self.out("not_used.json"))["stage"], "scan")
        self.assertEqual(run("apply", "--dry-run", "--root", self.root), 3)
        self.assertFalse(os.path.exists(self.out("player", "submission_player.csv")))

    def test_base_drift(self):
        self.assertEqual(run("fake", "--root", self.root), 0)
        base = os.path.join(self.root, "results", "test10", "player", "submission_player.csv")
        with open(base, "rb") as f:
            data = f.read()
        with open(base, "wb") as f:
            f.write(data.replace(b"\n", b"\r\n"))
        self.assertEqual(run("apply", "--dry-run", "--root", self.root), 4)
        self.assertFalse(os.path.exists(self.out("player", "submission_player.csv")))


if __name__ == "__main__":
    unittest.main()
