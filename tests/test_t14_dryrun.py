"""C10_SLOW=1 rehearsal of the 14 kit (PREREG_14 section 6): rebuild the 800 fake questions from fit players, run
tools/t14_copy.py scan and apply with --dry-run in a temporary tree (index part fit, held-out games excluded), and
compare with the committed results/test14/design/sim.json (files, apply counts, output SHA-256, evaluation).
Reads data/training (fit players only); never data/local. About 5 minutes.
Run: C10_SLOW=1 python -m pytest tests/test_t14_dryrun.py
"""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t14_copy as tc  # noqa: E402

SIM = os.path.join(REPO, "results", "test14", "design", "sim.json")


@unittest.skipUnless(os.environ.get("C10_SLOW"), "set C10_SLOW=1 (about 5 minutes, reads data/training)")
class DryRun(unittest.TestCase):
    def test_sim_rebuilt_equal(self):
        with open(SIM, encoding="utf-8") as f:
            sim = json.load(f)
        self.assertTrue(sim["evaluation"]["passed"])
        files, expected, exclude, base = tc.make_fake(tc.TRAIN_DIR)
        self.assertEqual(len(exclude), sim["held_out_games_excluded_from_index"])
        with tempfile.TemporaryDirectory(prefix="c10_t14_dry_") as root:
            bpath = tc.write_fake_tree(root, files, base)
            ex = os.path.join(root, "t14_exclude.json")
            tc.write_json(ex, exclude)
            common = ["--dry-run", "--root", root, "--index-part", "fit", "--exclude", ex]
            self.assertEqual(tc.main(["scan"] + common), 0)
            scan = tc.read_json(os.path.join(root, "results", "test14", "scan", "scan.json"))
            self.assertEqual(scan["files"], sim["files"])
            self.assertEqual(scan["questions_overridable"], sim["apply"]["questions_overridden"])
            self.assertEqual(tc.main(["apply"] + common), 0)
            rdir = os.path.join(root, "results", "test14", "rank")
            rep = tc.read_json(os.path.join(rdir, "t14_copy_report.json"))
            self.assertEqual(rep["files"], sim["files"])
            for k, v in sim["apply"].items():
                self.assertEqual(len(rep["conflicts"]) if k == "conflicts" else rep[k], v, k)
            self.assertEqual(dict(rep["outputs_sha256"], base_submission_rank_csv=tc.sha256_file(bpath)),
                             sim["outputs_sha256"])
            ev = tc.evaluate(expected, rep, bpath, os.path.join(rdir, "submission_rank.csv"))
            self.assertEqual(json.loads(json.dumps(ev)), sim["evaluation"])
            self.assertEqual(tc.main(["status"] + common), 0)


if __name__ == "__main__":
    unittest.main()
