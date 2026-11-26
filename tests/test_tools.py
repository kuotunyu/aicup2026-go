"""Tests for the audit fixes: bootstrap intervals, the score ledger, the strict
submission check and the leak-checked loader.  Run: python -m unittest discover -s tests"""

import csv
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import c10data  # noqa: E402
import check_submission  # noqa: E402
import score  # noqa: E402


def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


class TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def p(self, name):
        return os.path.join(self.tmp, name)


class BootstrapInterval(unittest.TestCase):
    def test_known_stratified_interval(self):
        # stratum X: diffs {0, 2}; stratum Y: 50 diffs of 5.
        # Balanced statistic = (mean_X + 5) / 2 with mean_X in {0, 1, 2} -> {2.5, 3.0, 3.5}.
        a = {f"x{i}": 0.0 for i in range(2)} | {f"y{i}": 0.0 for i in range(50)}
        b = {"x0": 0.0, "x1": 2.0} | {f"y{i}": 5.0 for i in range(50)}
        strata = {q: q[0] for q in a}
        r = score.paired_bootstrap(a, b, strata, n_boot=10000, seed=0)
        self.assertEqual(r["mean_diff"], 3.0)
        self.assertEqual(r["ci95"], [2.5, 3.5])
        self.assertEqual(r["strata"], 2)

    def test_percentile_positions(self):
        # 100 distinct, irregular diffs so neighbouring order statistics differ;
        # recompute the resampled means independently and check the index choice.
        import math
        a = {f"q{i}": 0.0 for i in range(100)}
        b = {f"q{i}": math.sqrt(i + 0.123) * 1000 / 7 + (i % 7) * 0.0137 for i in range(100)}
        r = score.paired_bootstrap(a, b, n_boot=10000, seed=1)
        import random
        rng = random.Random(1)
        diffs = [b[q] - a[q] for q in sorted(a)]
        means = sorted(sum(diffs[rng.randrange(100)] for _ in range(100)) / 100 for _ in range(10000))
        self.assertEqual(r["ci95"], [means[250], means[9749]])


class Ledger(TmpDir):
    def setUp(self):
        super().setUp()
        self.ans = write(self.p("rank_val_answers.csv"), ["question_id", "rank", "player_id", "num_games"],
                         [["q1", "D", "a", 5], ["q2", "D", "b", 5], ["q3", "6D", "c", 5]])
        self.pa = write(self.p("a.csv"), ["question_id", "pred_rank"], [["q1", "D"], ["q2", "C"], ["q3", "D"]])
        self.pb = write(self.p("b.csv"), ["question_id", "pred_rank"], [["q1", "D"], ["q2", "D"], ["q3", "6D"]])

    def test_val_answers_need_ledger(self):
        with self.assertRaises(SystemExit):
            score.main(["rank", self.pa, self.ans])

    def test_ledger_rows_and_strata(self):
        ledger = self.p("ledger.csv")
        with redirect_stdout(io.StringIO()):
            score.main(["compare-rank", self.pa, self.pb, self.ans, "--ledger", ledger, "--note", "t"])
            score.main(["rank", self.pb, self.ans, "--ledger", ledger, "--note", "t2"])
        with open(ledger, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)                       # A, B, then the single score
        self.assertEqual([r["note"] for r in rows], ["t", "t", "t2"])
        a = score.score_rank(self.pa, self.ans)
        b = score.score_rank(self.pb, self.ans)
        buf = io.StringIO()
        with redirect_stdout(buf):
            score.main(["compare-rank", self.pa, self.pb, self.ans, "--ledger", ledger, "--note", "t3"])
        import json
        res = json.loads(buf.getvalue())
        self.assertAlmostEqual(res["B_minus_A_rank_balanced"]["mean_diff"],
                               b["rank_balanced_mean"] - a["rank_balanced_mean"])
        self.assertEqual(res["B_minus_A_rank_balanced"]["strata"], 2)


class StrictCheck(TmpDir):
    def setUp(self):
        super().setUp()
        self.q_pub = write(self.p("q_pub.csv"), ["question_id", "num_games"], [["pub_q_0001", 5], ["pub_q_0002", 5]])
        self.q_priv = write(self.p("q_priv.csv"), ["question_id", "num_games"], [["priv_q_0001", 5]])
        self.c_pub = write(self.p("c_pub.csv"), ["game_id", "player_id", "sgf_content", "color"],
                           [[f"pub_g{i}", f"pub_p_{i}", "x", "B"] for i in range(6)])
        self.c_priv = write(self.p("c_priv.csv"), ["game_id", "player_id", "sgf_content", "color"],
                            [[f"priv_g{i}", f"priv_p_{i}", "x", "B"] for i in range(6)])

    def raw(self, name, text):
        path = self.p(name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(text)
        return path

    def rank(self, text):
        return check_submission.check("rank", self.raw("s.csv", text), [self.q_pub, self.q_priv])

    def player(self, text):
        return check_submission.check("player", self.raw("s.csv", text), [self.q_pub, self.q_priv],
                                      [self.c_pub, self.c_priv])

    def test_rank_ok(self):
        self.assertEqual(self.rank("question_id,pred_rank\npub_q_0001,D\npub_q_0002,6D\npriv_q_0001,1D\n"), [])

    def test_rank_problems(self):
        self.assertTrue(self.rank("question_id,pred_rank\npub_q_0001,D\npub_q_0002,6D\n"))           # missing priv
        self.assertTrue(self.rank("question_id,pred_rank\npub_q_0001,d\npub_q_0002,6D\npriv_q_0001,1D\n"))
        self.assertTrue(self.rank("question_id,pred_rank\npub_q_0001, D\npub_q_0002,6D\npriv_q_0001,1D\n"))
        self.assertTrue(self.rank("question_id,pred_rank,x\npub_q_0001,D,1\npub_q_0002,6D,1\npriv_q_0001,1D,1\n"))
        self.assertTrue(self.rank("question_id,pred_rank\npub_q_0001,D,XX\npub_q_0002,6D\npriv_q_0001,1D\n"))
        self.assertTrue(self.rank("﻿question_id,pred_rank\npub_q_0001,D\npub_q_0002,6D\npriv_q_0001,1D\n"))
        self.assertTrue(self.rank("question_id,pred_rank\npub_q_0001,D\npub_q_0001,D\npub_q_0002,6D\npriv_q_0001,1D\n"))

    def test_player_pools(self):
        ok = ("question_id,top_1,top_2,top_3,top_4,top_5\n"
              "pub_q_0001,pub_p_0,pub_p_1,pub_p_2,pub_p_3,pub_p_4\n"
              "pub_q_0002,pub_p_5,pub_p_1,pub_p_2,pub_p_3,pub_p_4\n"
              "priv_q_0001,priv_p_0,priv_p_1,priv_p_2,priv_p_3,priv_p_4\n")
        self.assertEqual(self.player(ok), [])
        cross = ok.replace("priv_q_0001,priv_p_0", "priv_q_0001,pub_p_0")
        self.assertTrue(self.player(cross))
        repeat = ok.replace("pub_p_5,pub_p_1", "pub_p_1,pub_p_1")
        self.assertTrue(self.player(repeat))


class Loader(TmpDir):
    """Two players per rank; one val player; the loader must drop it and catch leaks."""

    def setUp(self):
        super().setUp()
        self.train_dir = self.p("training")
        os.makedirs(self.train_dir)
        split_rows = []
        for lv in c10data.LEVELS:
            rows = []
            for j in range(2):
                pid = f"{lv}_p{j}"
                for g in range(3):
                    rows.append([pid, f"g{j}{g}", lv, "B", f"(;SZ[19];B[{lv}{j}{g}])"])
                split_rows.append([pid, lv, 3, "1-4", "val" if (lv == "D" and j == 0) else "train"])
            write(os.path.join(self.train_dir, f"train_{lv}.csv"), c10data.COLUMNS, rows)
        with open(os.path.join(self.train_dir, "._train_D.csv"), "wb") as f:
            f.write(b"\x00\x05\x16\x07 mac debris")
        self.split = write(self.p("split.csv"), ["player_id", "rank", "n_games", "bin", "split"], split_rows)
        self.local = self.p("local")
        os.makedirs(self.local)

    def test_train_part_excludes_val_player(self):
        write(os.path.join(self.local, "rank_val_questions.csv"), ["question_id", "num_games", "sgf_1"],
              [["val_q_0001", 1, "(;SZ[19];B[D00])"]])
        rows = c10data.load_rows(self.train_dir, self.split, "train", self.local, check_counts=False)
        self.assertNotIn("D_p0", {r["player_id"] for r in rows})
        self.assertEqual(len(rows), 19 * 3)

    def test_leak_is_caught(self):
        write(os.path.join(self.local, "rank_val_questions.csv"), ["question_id", "num_games", "sgf_1"],
              [["val_q_0001", 1, "(;SZ[19];B[C10])"]])        # a training player's game
        with self.assertRaises(c10data.LeakError):
            c10data.load_rows(self.train_dir, self.split, "train", self.local, check_counts=False)

    def test_missing_local_files_refused(self):
        with self.assertRaises(c10data.LeakError):
            c10data.load_rows(self.train_dir, self.split, "train", self.local, check_counts=False)

    def test_count_assertion(self):
        with self.assertRaises(c10data.LeakError):
            c10data.load_rows(self.train_dir, self.split, "train", None, check_counts=True)


if __name__ == "__main__":
    unittest.main()
