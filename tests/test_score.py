"""Hand-computed checks for tools/score.py.  Run: python -m unittest discover -s tests"""

import csv
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import score  # noqa: E402

E1 = math.exp(-1)
E2 = math.exp(-2)


def write(tmp, name, header, rows):
    path = os.path.join(tmp, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


class RankScore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # truth: q1 D, q2 A, q3 1D, q4 6D
        self.ans = write(self.tmp, "ans.csv", ["question_id", "rank", "player_id", "num_games"],
                         [["q1", "D", "p1", 5], ["q2", "A", "p2", 7], ["q3", "1D", "p3", 9], ["q4", "6D", "p4", 20]])

    def pred(self, ranks, name="pred.csv"):
        return write(self.tmp, name, ["question_id", "pred_rank"], [[f"q{i}", r] for i, r in enumerate(ranks, 1)])

    def test_points(self):
        self.assertEqual(score.rank_points("3D", "3D"), 1.0)
        self.assertAlmostEqual(score.rank_points("A", "1D"), E1)   # A and 1D are neighbours
        self.assertAlmostEqual(score.rank_points("1D", "A"), E1)
        self.assertEqual(score.rank_points("B", "1D"), 0.0)        # two levels apart
        self.assertEqual(score.rank_points("D", "6D"), 0.0)

    def test_mixed(self):
        # q1 exact 1, q2 A->1D adjacent e^-1, q3 1D->3D 0, q4 6D->5D adjacent e^-1
        r = score.score_rank(self.pred(["D", "1D", "3D", "5D"]), self.ans)
        self.assertAlmostEqual(r["mean"], (1 + E1 + 0 + E1) / 4)
        self.assertAlmostEqual(r["rank_balanced_mean"], (1 + E1 + 0 + E1) / 4)  # one question per rank
        self.assertAlmostEqual(r["exact_rate"], 0.25)
        self.assertAlmostEqual(r["adjacent_rate"], 0.5)
        self.assertEqual(r["confusion_truth_by_pred"]["1D"]["3D"], 1)

    def test_rank_balanced_differs_from_mean(self):
        ans = write(self.tmp, "ans2.csv", ["question_id", "rank", "player_id", "num_games"],
                    [["q1", "D", "a", 5], ["q2", "D", "b", 5], ["q3", "D", "c", 5], ["q4", "6D", "d", 5]])
        pred = self.pred(["D", "D", "D", "D"], "p2.csv")
        r = score.score_rank(pred, ans)
        self.assertAlmostEqual(r["mean"], 3 / 4)
        self.assertAlmostEqual(r["rank_balanced_mean"], (1.0 + 0.0) / 2)

    def test_perfect(self):
        r = score.score_rank(self.pred(["D", "A", "1D", "6D"]), self.ans)
        self.assertEqual(r["mean"], 1.0)

    def test_format_errors(self):
        with self.assertRaises(score.SubmissionError):   # missing question
            score.score_rank(write(self.tmp, "m.csv", ["question_id", "pred_rank"],
                                   [["q1", "D"], ["q2", "A"], ["q3", "1D"]]), self.ans)
        with self.assertRaises(score.SubmissionError):   # duplicate
            score.score_rank(write(self.tmp, "d.csv", ["question_id", "pred_rank"],
                                   [["q1", "D"], ["q1", "D"], ["q2", "A"], ["q3", "1D"], ["q4", "6D"]]), self.ans)
        with self.assertRaises(score.SubmissionError):   # unknown rank label
            score.score_rank(self.pred(["D", "A", "7D", "6D"], "u.csv"), self.ans)
        with self.assertRaises(score.SubmissionError):   # wrong column name
            score.score_rank(write(self.tmp, "c.csv", ["question_id", "rank"],
                                   [["q1", "D"], ["q2", "A"], ["q3", "1D"], ["q4", "6D"]]), self.ans)
        with self.assertRaises(score.SubmissionError):   # extra question
            score.score_rank(self.pred(["D", "A", "1D", "6D", "D"], "x.csv"), self.ans)


class PlayerScore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cand = write(self.tmp, "cand.csv", ["game_id", "player_id", "sgf_content", "color"],
                          [[f"g{i}", f"p{i}", "(;SZ[19];B[pd])", "B"] for i in range(1, 8)])
        self.ans = write(self.tmp, "ans.csv", ["question_id", "player_id", "rank"],
                         [["q1", "p1", "D"], ["q2", "p2", "A"], ["q3", "p3", "6D"]])

    def pred(self, rows, name="pred.csv"):
        return write(self.tmp, name, ["question_id"] + score.TOP, rows)

    def test_positions(self):
        r = score.score_player(self.pred([
            ["q1", "p1", "p2", "p3", "p4", "p5"],   # rank 1 -> 1
            ["q2", "p1", "p3", "p2", "p4", "p5"],   # rank 3 -> e^-2
            ["q3", "p1", "p2", "p4", "p5", "p6"],   # absent -> 0
        ]), self.ans, self.cand)
        self.assertAlmostEqual(r["mean"], (1 + E2 + 0) / 3)
        self.assertAlmostEqual(r["top1_rate"], 1 / 3)
        self.assertAlmostEqual(r["top5_rate"], 2 / 3)

    def test_fifth_place(self):
        r = score.score_player(self.pred([
            ["q1", "p2", "p3", "p4", "p5", "p1"],
            ["q2", "p2", "p1", "p3", "p4", "p5"],
            ["q3", "p1", "p2", "p3", "p4", "p5"],
        ]), self.ans, self.cand)
        self.assertAlmostEqual(r["mean"], (math.exp(-4) + 1 + E2) / 3)

    def test_errors(self):
        with self.assertRaises(score.SubmissionError):   # repeated player
            score.score_player(self.pred([["q1", "p1", "p1", "p3", "p4", "p5"],
                                          ["q2", "p1", "p2", "p3", "p4", "p5"],
                                          ["q3", "p1", "p2", "p3", "p4", "p5"]], "r.csv"), self.ans, self.cand)
        with self.assertRaises(score.SubmissionError):   # player outside the pool
            score.score_player(self.pred([["q1", "p1", "p2", "p3", "p4", "zz"],
                                          ["q2", "p1", "p2", "p3", "p4", "p5"],
                                          ["q3", "p1", "p2", "p3", "p4", "p5"]], "o.csv"), self.ans, self.cand)


class Bootstrap(unittest.TestCase):
    def test_identical_predictions(self):
        a = {"q1": 1.0, "q2": 0.0, "q3": E1}
        r = score.paired_bootstrap(a, dict(a), n_boot=200)
        self.assertEqual(r["mean_diff"], 0.0)
        self.assertEqual(r["ci95"], [0.0, 0.0])

    def test_uniform_gain(self):
        a = {f"q{i}": 0.0 for i in range(50)}
        b = {f"q{i}": 1.0 for i in range(50)}
        r = score.paired_bootstrap(a, b, n_boot=200)
        self.assertEqual(r["mean_diff"], 1.0)
        self.assertEqual(r["share_resamples_leq_0"], 0.0)

    def test_strata_balance(self):
        # D: 3 questions gain 1; 6D: 1 question loses 1 -> balanced diff (1 + -1)/2 = 0
        a = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 1.0}
        b = {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 0.0}
        strata = {"q1": "D", "q2": "D", "q3": "D", "q4": "6D"}
        self.assertAlmostEqual(score.paired_bootstrap(a, b, strata, n_boot=50)["mean_diff"], 0.0)
        self.assertAlmostEqual(score.paired_bootstrap(a, b, n_boot=50)["mean_diff"], 0.5)


if __name__ == "__main__":
    unittest.main()
