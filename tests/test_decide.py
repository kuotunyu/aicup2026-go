"""Hand-computed checks for tools/decide.py (PREREG_01 point 3).
Run: python -m unittest discover -s tests

The expected answers below are worked out by hand in the comments; the
constant e^-1 is recomputed here (and pinned to a literal) instead of being
taken from decide.py, so a wrong constant in decide.py makes the tests fail.
"""

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import decide  # noqa: E402

E = math.exp(-1)
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
HEADER = ["question_id", "n_games", "n_parsed"] + ["p_" + lv for lv in LEVELS]


def vec(**kw):
    """10-vector from level -> probability; levels starting with a digit are passed as _1D etc."""
    p = [0.0] * 10
    for k, v in kw.items():
        p[LEVELS.index(k.lstrip("_"))] = v
    return p


def row(qid, p, n_games=10, n_parsed=10):
    """CSV row; p is a 10-vector (written with repr, no rounding) or None (blank)."""
    probs = [""] * 10 if p is None else [repr(x) for x in p]
    return [qid, str(n_games), str(n_parsed)] + probs


class Constant(unittest.TestCase):
    def test_e_minus_1(self):
        self.assertEqual(E, 0.36787944117144233)


class HandComputedB(unittest.TestCase):
    def test_b_differs_from_a(self):
        # p_A 0.4, p_1D 0.3, p_2D 0.3.  A: argmax -> A.
        # B scores: B = e*0.4 = .147, A = 0.4 + e*0.3 = .510, 1D = 0.3 + e*(0.4+0.3) = .558,
        #           2D = 0.3 + e*0.3 = .410, 3D = e*0.3 = .110, others 0  -> B picks 1D.
        p = vec(A=0.4, _1D=0.3, _2D=0.3)
        want = [0.0, 0.0, E * 0.4, 0.4 + E * 0.3, 0.3 + E * 0.7, 0.3 + E * 0.3, E * 0.3, 0.0, 0.0, 0.0]
        got = decide.b_scores(p)
        self.assertEqual(len(got), 10)
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=12)
        self.assertEqual(decide.rule_a(p), "A")
        self.assertEqual(decide.rule_b(p), "1D")

    def test_depends_on_the_constant(self):
        # p_A .472, p_1D .3, p_2D .228.  1D minus A = (0.3 + c*0.7) - (0.472 + c*0.3) = -0.172 + 0.4c:
        # c = e^-1 -> -0.0249, B stays on A;  c = 0.5 would give +0.028 and pick 1D.
        p = vec(A=0.472, _1D=0.3, _2D=0.228)
        self.assertEqual(decide.rule_a(p), "A")
        self.assertEqual(decide.rule_b(p), "A")
        s = decide.b_scores(p)
        self.assertAlmostEqual(s[4] - s[3], -0.172 + 0.4 * E, places=12)

    def test_low_end_has_no_left_neighbour(self):
        # p_D .5, p_C .2, p_B .3:  D = 0.5 + e*0.2 = .574, C = 0.2 + e*0.8 = .494,
        # B = 0.3 + e*0.2 = .374, A = e*0.3  -> B picks D.
        p = vec(D=0.5, C=0.2, B=0.3)
        s = decide.b_scores(p)
        self.assertAlmostEqual(s[0], 0.5 + E * 0.2, places=12)
        self.assertEqual(decide.rule_b(p), "D")

    def test_ends_do_not_wrap_low(self):
        # p_D .35, p_2D .40, p_6D .25.  No wrap: D = .35, 2D = .40, 6D = .25, 1D = 3D = e*.4 = .147
        # -> A and B both pick 2D.  With wrap-around D would get .35 + e*.25 = .442 and win.
        p = vec(D=0.35, _2D=0.40, _6D=0.25)
        s = decide.b_scores(p)
        self.assertAlmostEqual(s[0], 0.35, places=12)
        self.assertAlmostEqual(s[9], 0.25, places=12)
        self.assertEqual(decide.rule_a(p), "2D")
        self.assertEqual(decide.rule_b(p), "2D")

    def test_ends_do_not_wrap_high(self):
        # p_D .25, p_3D .40, p_6D .35.  No wrap: 3D = .40, 6D = .35, D = .25 -> B picks 3D.
        # With wrap-around 6D would get .35 + e*.25 = .442 and win.
        p = vec(D=0.25, _3D=0.40, _6D=0.35)
        s = decide.b_scores(p)
        self.assertAlmostEqual(s[9], 0.35, places=12)
        self.assertAlmostEqual(s[0], 0.25, places=12)
        self.assertEqual(decide.rule_b(p), "3D")

    def test_high_end_has_no_right_neighbour(self):
        # p_4D .3, p_5D .2, p_6D .5:  6D = 0.5 + e*0.2 = .574, 5D = 0.2 + e*0.8 = .494 -> 6D.
        p = vec(_4D=0.3, _5D=0.2, _6D=0.5)
        s = decide.b_scores(p)
        self.assertAlmostEqual(s[9], 0.5 + E * 0.2, places=12)
        self.assertEqual(decide.rule_b(p), "6D")


class Ties(unittest.TestCase):
    def test_uniform(self):
        # all 0.1:  A ties on all ten -> D.  B: ends 0.1 + e*0.1, middle eight 0.1 + e*0.2 -> first middle = C.
        p = [0.1] * 10
        self.assertEqual(decide.rule_a(p), "D")
        self.assertEqual(decide.rule_b(p), "C")

    def test_a_two_level_tie(self):
        # p_C .3, p_2D .2, p_3D .3, p_4D .2:  A ties C / 3D -> C (weaker).
        # B: 3D = 0.3 + e*0.4 = .447 is the unique max -> 3D.
        p = vec(C=0.3, _2D=0.2, _3D=0.3, _4D=0.2)
        self.assertEqual(decide.rule_a(p), "C")
        self.assertEqual(decide.rule_b(p), "3D")

    def test_b_two_level_tie(self):
        # p = .15 at D, B, 4D, 6D;  .1 at C, 5D;  .2 at 1D  (sum 1).
        # A: 1D (unique max .2).
        # B: C = 5D = 0.1 + e*0.3 = .2104 (same operands -> exactly equal), 1D = .2,
        #    D = B = 4D = 6D = 0.15 + e*0.1 = .187, A = e*.35 -> tie C / 5D -> C.
        p = vec(D=0.15, C=0.1, B=0.15, _1D=0.2, _4D=0.15, _5D=0.1, _6D=0.15)
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        s = decide.b_scores(p)
        self.assertEqual(s[1], s[8])
        self.assertEqual(max(s), s[1])
        self.assertEqual(decide.rule_a(p), "1D")
        self.assertEqual(decide.rule_b(p), "C")

    def test_argmax_first(self):
        self.assertEqual(decide.argmax_first([1, 3, 3, 2]), 1)
        self.assertEqual(decide.argmax_first([5, 5]), 0)
        self.assertEqual(decide.argmax_first([0, 0, 1]), 2)


class TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rows, header=HEADER, name="probs.csv"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return path

    def read_pred(self, out, name):
        with open(os.path.join(out, name), newline="", encoding="utf-8") as f:
            return list(csv.reader(f))


class RunAndOutput(TmpDir):
    def test_order_missing_and_summary(self):
        rows = [
            row("q3", vec(A=0.4, _1D=0.3, _2D=0.3)),          # A: A,   B: 1D
            row("q1", None, n_games=4, n_parsed=0),           # missing (n_parsed 0, blank)  -> D, D
            row("q9", [0.1] * 10),                            # A: D,   B: C
            row("q2", None, n_games=0, n_parsed=0),           # missing (no games at all)    -> D, D
            row("q5", vec(_6D=1.0), n_games=6, n_parsed=0),   # missing, probs ignored       -> D, D (not 6D)
            row("q4", vec(_6D=1.0), n_games=6, n_parsed=6),   # 6D, 6D
        ]
        out = os.path.join(self.tmp, "out")
        s = decide.run(self.write(rows), out)

        a = self.read_pred(out, "pred_A.csv")
        b = self.read_pred(out, "pred_B.csv")
        self.assertEqual(a[0], ["question_id", "pred_rank"])
        self.assertEqual(b[0], ["question_id", "pred_rank"])
        self.assertEqual(a[1:], [["q3", "A"], ["q1", "D"], ["q9", "D"], ["q2", "D"], ["q5", "D"], ["q4", "6D"]])
        self.assertEqual(b[1:], [["q3", "1D"], ["q1", "D"], ["q9", "C"], ["q2", "D"], ["q5", "D"], ["q4", "6D"]])

        with open(os.path.join(out, "decide_summary.json"), encoding="utf-8") as f:
            js = json.load(f)
        self.assertEqual(js, s)
        self.assertEqual(js["n_questions"], 6)
        self.assertEqual(js["n_missing"], 3)
        self.assertEqual(js["missing_question_ids"], ["q1", "q2", "q5"])
        self.assertEqual(js["missing_with_probs_ignored"], 1)
        self.assertEqual(js["missing_fallback"], "D")
        self.assertEqual(js["n_A_ne_B"], 2)
        self.assertEqual(js["A_to_B_changes"], {"A->1D": 1, "D->C": 1})
        self.assertEqual(list(js["distribution_A"]), LEVELS)
        self.assertEqual(js["distribution_A"]["D"], 4)
        self.assertEqual(js["distribution_A"]["A"], 1)
        self.assertEqual(js["distribution_A"]["6D"], 1)
        self.assertEqual(sum(js["distribution_A"].values()), 6)
        self.assertEqual(js["distribution_B"]["D"], 3)
        self.assertEqual(js["distribution_B"]["C"], 1)
        self.assertEqual(js["distribution_B"]["1D"], 1)
        self.assertEqual(js["distribution_B"]["6D"], 1)
        self.assertEqual(sum(js["distribution_B"].values()), 6)

    def test_exact_bytes(self):
        out = os.path.join(self.tmp, "out")
        decide.run(self.write([row("b", vec(C=1.0)), row("a", vec(_5D=1.0))]), out)
        with open(os.path.join(out, "pred_A.csv"), "rb") as f:
            self.assertEqual(f.read(), b"question_id,pred_rank\nb,C\na,5D\n")

    def test_no_rounding(self):
        # p_B .35, p_A .3500000001, p_4D .2999999999.  Read exactly, A wins by 1e-10;
        # rounded to a few decimals it would tie with B and the tie rule would give B.
        path = self.write([["q1", "10", "10", "0", "0", "0.35", "0.3500000001", "0.2999999999",
                            "0", "0", "0", "0", "0"]])
        out = os.path.join(self.tmp, "out")
        decide.run(path, out)
        self.assertEqual(self.read_pred(out, "pred_A.csv")[1], ["q1", "A"])

    def test_sum_within_tolerance_is_accepted(self):
        p = vec(A=0.5, _1D=0.5 + 5e-6)  # sum off by 5e-6 < 1e-5
        out = os.path.join(self.tmp, "out")
        decide.run(self.write([row("q1", p)]), out)
        self.assertEqual(self.read_pred(out, "pred_A.csv")[1], ["q1", "1D"])

    def test_extra_columns_are_ignored(self):
        out = os.path.join(self.tmp, "out")
        header = HEADER + ["note"]
        decide.run(self.write([row("q1", vec(B=1.0)) + ["x"]], header=header), out)
        self.assertEqual(self.read_pred(out, "pred_B.csv")[1], ["q1", "B"])


class Errors(TmpDir):
    def assert_rejected(self, rows, header=HEADER, msg=None):
        out = os.path.join(self.tmp, "out")
        with self.assertRaises(decide.DecideError) as cm:
            decide.run(self.write(rows, header=header), out)
        if msg:
            self.assertIn(msg, str(cm.exception))
        self.assertFalse(os.path.exists(out), "nothing may be written on error")
        return str(cm.exception)

    # --- only n_parsed == 0 may become D; everything else that is incomplete stops the run ---

    def test_n_parsed_positive_all_blank(self):
        # games were parsed but no probabilities came back: a producer bug, not a missing question
        m = self.assert_rejected([row("q1", vec(A=1.0)), row("q7", None, n_games=7, n_parsed=3)],
                                 msg="n_parsed=3 > 0")
        self.assertIn("'q7'", m)
        self.assertIn("line 3", m)
        self.assertIn("blank", m)

    def test_n_parsed_positive_partly_blank(self):
        r = row("q8", vec(A=1.0), n_games=5, n_parsed=1)
        r[HEADER.index("p_6D")] = ""
        m = self.assert_rejected([r], msg="n_parsed=1 > 0")
        self.assertIn("'q8'", m)
        self.assertIn("p_6D", m)

    def test_n_parsed_zero_partly_blank(self):
        r = row("q1", vec(A=1.0), n_games=5, n_parsed=0)
        r[HEADER.index("p_B")] = ""
        self.assert_rejected([r], msg="some probability columns blank")

    def test_truncated_row_without_probabilities(self):
        # only question_id,n_games,n_parsed: csv.DictReader would give None -> "blank" -> D
        m = self.assert_rejected([row("q1", vec(A=1.0)), ["q2", "9", "9"]], msg="truncated")
        self.assertIn("'q2'", m)
        self.assertIn("line 3", m)
        self.assertIn("3 fields", m)

    def test_truncated_row_with_n_parsed_zero(self):
        # even a question that would be filled with D must not be accepted when cut short
        self.assert_rejected([["q2", "9", "0"]], msg="truncated")

    def test_truncated_in_the_middle_of_the_probabilities(self):
        r = row("q3", vec(D=0.5, C=0.5))[:8]  # p_D .. p_1D present, p_2D .. p_6D cut off
        m = self.assert_rejected([r], msg="truncated")
        self.assertIn("p_2D", m)

    def test_truncated_extra_column(self):
        # the extra "note" column is not needed, but a row shorter than its header is still corrupt
        self.assert_rejected([row("q1", vec(B=1.0))], header=HEADER + ["note"], msg="truncated")

    def test_more_fields_than_header(self):
        m = self.assert_rejected([row("q1", vec(A=1.0)), row("q4", vec(A=1.0)) + ["0.1"]],
                                 msg="more than the 13 columns")
        self.assertIn("'q4'", m)
        self.assertIn("14 fields", m)

    def test_more_fields_than_header_with_extra_column(self):
        self.assert_rejected([row("q1", vec(A=1.0)) + ["note", "x"]], header=HEADER + ["note"],
                             msg="more than the 14 columns")

    def test_n_games_negative(self):
        m = self.assert_rejected([row("q6", None, n_games=-1, n_parsed=0)], msg="n_games is negative")
        self.assertIn("'q6'", m)

    def test_n_parsed_negative(self):
        self.assert_rejected([row("q1", vec(A=1.0), n_games=3, n_parsed=-2)], msg="n_parsed is negative")

    def test_n_games_not_an_integer(self):
        self.assert_rejected([row("q1", vec(A=1.0), n_games="1.5")], msg="n_games is not an integer")

    def test_duplicate_column_name(self):
        header = HEADER + ["p_A"]
        self.assert_rejected([row("q1", vec(A=1.0)) + ["0"]], header=header, msg="duplicate column")

    def test_sum_too_high(self):
        self.assert_rejected([row("q1", vec(A=0.5, _1D=0.5 + 2e-5))], msg="sum")

    def test_sum_too_low(self):
        self.assert_rejected([row("q1", vec(A=0.5, _1D=0.5 - 2e-5))], msg="sum")

    def test_sum_far_off(self):
        self.assert_rejected([row("q1", [0.2] * 10)], msg="sum")

    def test_negative(self):
        self.assert_rejected([row("q1", vec(A=0.6, _1D=0.5, _2D=-0.1))], msg="negative")

    def test_nan(self):
        r = row("q1", vec(A=1.0))
        r[3] = "nan"
        self.assert_rejected([r], msg="finite")

    def test_not_a_number(self):
        r = row("q1", vec(A=1.0))
        r[3] = "abc"
        self.assert_rejected([r], msg="not a number")

    def test_partly_blank(self):
        r = row("q1", vec(A=1.0))
        r[3] = ""
        self.assert_rejected([r], msg="blank")

    def test_missing_column(self):
        header = [h for h in HEADER if h != "p_3D"]
        r = row("q1", vec(A=1.0))
        del r[HEADER.index("p_3D")]
        self.assert_rejected([r], header=header, msg="missing columns")

    def test_duplicate_question(self):
        self.assert_rejected([row("q1", vec(A=1.0)), row("q1", vec(B=1.0))], msg="duplicate")

    def test_empty_question_id(self):
        self.assert_rejected([row("", vec(A=1.0))], msg="empty question_id")

    def test_n_parsed_above_n_games(self):
        self.assert_rejected([row("q1", vec(A=1.0), n_games=3, n_parsed=4)], msg="n_parsed")

    def test_no_rows(self):
        self.assert_rejected([], msg="no questions")

    def test_bad_row_after_good_ones_writes_nothing(self):
        self.assert_rejected([row("q1", vec(A=1.0)), row("q2", vec(A=1.0)), row("q3", [0.2] * 10)])


OUT_NAMES = ["pred_A.csv", "pred_B.csv", "decide_summary.json"]


class NoStaleOutputs(TmpDir):
    """A failed run must not leave an older prediction behind that looks current."""

    def stale_out(self):
        out = os.path.join(self.tmp, "out")
        os.makedirs(out)
        for name in OUT_NAMES + ["pred_A.csv.tmp", "keep_me.txt"]:
            with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                f.write("OLD\n")
        return out

    def assert_outputs_gone(self, out):
        left = sorted(os.listdir(out))
        self.assertEqual(left, ["keep_me.txt"], "old outputs or temp files left behind")

    def test_error_in_input_removes_old_outputs(self):
        out = self.stale_out()
        with self.assertRaises(decide.DecideError):
            decide.run(self.write([row("q1", vec(A=1.0)), ["q2", "9", "9"]]), out)
        self.assert_outputs_gone(out)

    def test_missing_input_file_removes_old_outputs(self):
        out = self.stale_out()
        with self.assertRaises(OSError):
            decide.run(os.path.join(self.tmp, "no_such.csv"), out)
        self.assert_outputs_gone(out)

    def test_success_replaces_old_outputs_and_leaves_no_temp(self):
        out = self.stale_out()
        s = decide.run(self.write([row("q1", vec(C=1.0))]), out)
        self.assertEqual(sorted(os.listdir(out)), sorted(OUT_NAMES + ["keep_me.txt"]))
        self.assertEqual(self.read_pred(out, "pred_A.csv"), [["question_id", "pred_rank"], ["q1", "C"]])
        with open(os.path.join(out, "decide_summary.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), s)

    def test_hashes_match_the_final_files(self):
        out = os.path.join(self.tmp, "out")
        s = decide.run(self.write([row("q1", vec(C=1.0)), row("q2", [0.1] * 10)]), out)
        for key, name in (("pred_A_sha256", "pred_A.csv"), ("pred_B_sha256", "pred_B.csv")):
            with open(os.path.join(out, name), "rb") as f:
                self.assertEqual(s[key], hashlib.sha256(f.read()).hexdigest())

    def test_failure_while_renaming_leaves_none_of_the_three(self):
        out = self.stale_out()
        real_replace = os.replace
        calls = []

        def flaky_replace(src, dst):
            calls.append(dst)
            if len(calls) == 2:
                raise OSError("disk full (simulated)")
            return real_replace(src, dst)

        with mock.patch.object(decide.os, "replace", flaky_replace):
            with self.assertRaises(OSError):
                decide.run(self.write([row("q1", vec(C=1.0))]), out)
        self.assertEqual(len(calls), 2)  # pred_A.csv was already in place when it failed
        self.assert_outputs_gone(out)

    def test_failure_while_writing_the_summary_leaves_none_of_the_three(self):
        out = self.stale_out()
        with mock.patch.object(decide.json, "dump", side_effect=OSError("simulated")):
            with self.assertRaises(OSError):
                decide.run(self.write([row("q1", vec(C=1.0))]), out)
        self.assert_outputs_gone(out)

    def test_input_inside_out_dir_under_an_output_name_is_refused(self):
        out = os.path.join(self.tmp, "out")
        os.makedirs(out)
        path = self.write([row("q1", vec(C=1.0))], name=os.path.join("out", "pred_A.csv"))
        with self.assertRaises(decide.DecideError):
            decide.run(path, out)
        self.assertTrue(os.path.exists(path), "the input must not be deleted")


class CommandLine(TmpDir):
    SCRIPT = os.path.join(TOOLS, "decide.py")

    def call(self, *args):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        return subprocess.run([sys.executable, self.SCRIPT, *args], capture_output=True,
                              text=True, encoding="utf-8", env=env)

    def test_ok(self):
        out = os.path.join(self.tmp, "out")
        path = self.write([row("q1", [0.1] * 10), row("q2", None, n_parsed=0)])
        r = self.call(path, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("questions: 2", r.stdout)
        self.assertIn("missing", r.stdout)
        self.assertIn("q2", r.stdout)
        self.assertIn("A != B: 1", r.stdout)
        for name in ("pred_A.csv", "pred_B.csv", "decide_summary.json"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)

    def test_error_exit_code(self):
        out = os.path.join(self.tmp, "out")
        r = self.call(self.write([row("q1", [0.2] * 10)]), out)
        self.assertEqual(r.returncode, 1)
        self.assertIn("ERROR", r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_blank_probs_with_parsed_games_exit_code_and_old_outputs_removed(self):
        out = os.path.join(self.tmp, "out")
        os.makedirs(out)
        for name in ("pred_A.csv", "pred_B.csv", "decide_summary.json"):
            with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                f.write("OLD\n")
        r = self.call(self.write([row("q1", vec(A=1.0)), row("q2", None, n_games=8, n_parsed=8)]), out)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("ERROR", r.stderr)
        self.assertIn("'q2'", r.stderr)
        self.assertIn("n_parsed=8 > 0", r.stderr)
        self.assertIn("nothing written", r.stderr)
        self.assertEqual(os.listdir(out), [])

    def test_truncated_row_exit_code(self):
        out = os.path.join(self.tmp, "out")
        r = self.call(self.write([["q9", "4", "4", "0.5", "0.5"]]), out)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("'q9'", r.stderr)
        self.assertIn("truncated", r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_usage(self):
        r = self.call()
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
