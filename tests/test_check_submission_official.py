"""check_submission.py: the test-day --official mode and the argument handling (spec 3.4).

Synthetic files at official size (2 x 400 questions, 2 pools of 400 players x 100 games); no
competition data is read. tests/test_tools.py::StrictCheck keeps covering the default mode.
Run: python -m pytest tests/test_check_submission_official.py
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOL = os.path.join(REPO, "tools", "check_submission.py")
sys.path.insert(0, os.path.join(REPO, "tools"))
import check_submission as cs  # noqa: E402

LEVELS = cs.LEVELS
QH = ["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, 21)] + [f"color_{i}" for i in range(1, 21)]
CH = ["game_id", "player_id", "sgf_content", "color"]


def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def q_rows(prefix, n=400):
    return [[f"{prefix}_q_{i:04d}", 5] + ["(;SZ[19];B[dd])"] * 5 + [""] * 15 + ["B"] * 5 + [""] * 15
            for i in range(1, n + 1)]


def c_rows(prefix, players=400, games=100):
    return [[f"{prefix}_g_{i:06d}", f"{prefix}_p_{i // games:04d}", "(;SZ[19];B[dd])", "B"]
            for i in range(players * games)]


class Official(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        p = lambda n: os.path.join(cls.tmp, n)  # noqa: E731
        cls.q_pub = write(p("rank_prediction_test_public.csv"), QH, q_rows("pub"))
        cls.q_priv = write(p("rank_prediction_test_private.csv"), QH, q_rows("priv"))
        cls.c_pub = write(p("player_identification_candidates_public.csv"), CH, c_rows("pub"))
        cls.c_priv = write(p("player_identification_candidates_private.csv"), CH, c_rows("priv"))
        cls.ids = [r[0] for r in q_rows("pub")] + [r[0] for r in q_rows("priv")]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def p(self, name):
        return os.path.join(self.tmp, name)

    # ------------------------------------------------------------ helpers
    def rank_sub(self, rows, name="submission_rank.csv"):
        return write(self.p(name), ["question_id", "pred_rank"], rows)

    def rank_ok_rows(self):
        return [[q, LEVELS[i % 10]] for i, q in enumerate(self.ids)]

    def player_ok_rows(self):
        out = []
        for q in self.ids:
            pre = "pub" if q.startswith("pub_") else "priv"
            k = int(q[-4:]) % 395
            out.append([q] + [f"{pre}_p_{k + j:04d}" for j in range(5)])
        return out

    def player_sub(self, rows, name="submission_player.csv"):
        return write(self.p(name), ["question_id"] + cs.TOP, rows)

    def rank(self, path, qs=None):
        return cs.check("rank", path, qs or [self.q_pub, self.q_priv], official=True)

    def player(self, path, qs=None, cands=None):
        return cs.check("player", path, qs or [self.q_pub, self.q_priv], cands or [self.c_pub, self.c_priv],
                        official=True)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", TOOL] + list(args), capture_output=True, text=True,
                              timeout=120)

    # ------------------------------------------------------------ cases
    def test_ok_rank_and_player(self):
        self.assertEqual(self.rank(self.rank_sub(self.rank_ok_rows())), [])
        self.assertEqual(self.player(self.player_sub(self.player_ok_rows())), [])

    def test_ok_cli_message(self):
        sub = self.rank_sub(self.rank_ok_rows())
        r = self.cli("rank", sub, "--questions", self.q_pub, self.q_priv, "--official")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(r.stdout.startswith("OK (official): "), r.stdout)
        self.assertIn("800 rows in official order", r.stdout)

    def test_reversed_order(self):
        sub = self.rank_sub(list(reversed(self.rank_ok_rows())))
        errs = self.rank(sub)
        self.assertEqual(len(errs), 1, errs)
        self.assertIn("line 2: row order differs", errs[0])
        self.assertIn("expected pub_q_0001, got priv_q_0400", errs[0])
        # the default mode keeps accepting any order (backward compatible)
        self.assertEqual(cs.check("rank", sub, [self.q_pub, self.q_priv]), [])

    def test_first_mismatch_reported(self):
        rows = self.rank_ok_rows()
        rows[10], rows[11] = rows[11], rows[10]
        errs = self.rank(self.rank_sub(rows))
        self.assertEqual(errs, [e for e in errs if "line 12: row order differs" in e], errs)

    def test_399_rows(self):
        errs = self.rank(self.rank_sub(self.rank_ok_rows()[:399] + self.rank_ok_rows()[400:]))
        self.assertTrue(any("1 questions missing" in e for e in errs), errs)
        self.assertTrue(any("exactly 800 rows after the header" in e and "got 799" in e for e in errs), errs)

    def test_question_file_with_399_rows(self):
        short = write(self.p("q_pub_399.csv"), QH, q_rows("pub", 399))
        rows = [r for r in self.rank_ok_rows() if r[0] != "pub_q_0400"]
        errs = self.rank(self.rank_sub(rows), qs=[short, self.q_priv])
        self.assertTrue(any("399 questions; official public files have exactly 400" in e for e in errs), errs)

    def test_prefixes_swapped_between_files(self):
        errs = self.rank(self.rank_sub(self.rank_ok_rows()), qs=[self.q_priv, self.q_pub])
        self.assertTrue(any("do not start with 'pub_'" in e and "public file" in e for e in errs), errs)
        self.assertTrue(any("do not start with 'priv_'" in e for e in errs), errs)
        self.assertTrue(any("row order differs" in e for e in errs), errs)

    def test_candidate_with_99_games(self):
        rows = c_rows("pub")
        rows[5][1] = "pub_p_0001"          # player 0000 now has 99 games, player 0001 has 101
        bad = write(self.p("c_pub_99.csv"), CH, rows)
        errs = self.player(self.player_sub(self.player_ok_rows()), cands=[bad, self.c_priv])
        self.assertTrue(any("2 candidates without exactly 100 games" in e and "pub_p_0000 has 99" in e
                            for e in errs), errs)

    def test_pools_share_a_game_id(self):
        rows = c_rows("priv")
        rows[0][0] = "pub_g_000000"
        bad = write(self.p("c_priv_shared.csv"), CH, rows)
        errs = self.player(self.player_sub(self.player_ok_rows()), cands=[self.c_pub, bad])
        self.assertTrue(any("share 1 game_id" in e for e in errs), errs)
        self.assertTrue(any("game_id do not start with 'priv_'" in e for e in errs), errs)

    def test_pools_share_a_player_id(self):
        rows = c_rows("priv")
        for r in rows[:100]:
            r[1] = "pub_p_0000"
        bad = write(self.p("c_priv_shared_p.csv"), CH, rows)
        errs = self.player(self.player_sub(self.player_ok_rows()), cands=[self.c_pub, bad])
        self.assertTrue(any("share 1 player_id" in e for e in errs), errs)

    def test_pick_from_the_wrong_pool(self):
        rows = self.player_ok_rows()
        rows[0][3] = "priv_p_0003"
        errs = self.player(self.player_sub(rows))
        self.assertEqual(len(errs), 1, errs)
        self.assertIn("pub_q_0001 is a public question but names players that are not pub_p_*", errs[0])
        self.assertIn("private pool", errs[0])

    def test_wrong_basename(self):
        errs = self.rank(self.rank_sub(self.rank_ok_rows(), name="pred_B.csv"))
        self.assertEqual(errs, ["file name must be exactly submission_rank.csv for an official rank upload; "
                                "got pred_B.csv"])
        errs = self.player(self.player_sub(self.player_ok_rows(), name="submission_rank.csv"))
        self.assertTrue(errs and "submission_player.csv" in errs[0], errs)

    def test_one_question_file_in_official_mode(self):
        rows = [r for r in self.rank_ok_rows() if r[0].startswith("pub_")]
        errs = self.rank(self.rank_sub(rows), qs=[self.q_pub])
        self.assertTrue(any("exactly 2 question files" in e for e in errs), errs)

    def test_zero_question_files(self):
        sub = self.rank_sub(self.rank_ok_rows())
        self.assertEqual(cs.check("rank", sub, []), ["no question files given"])
        self.assertEqual(cs.check("rank", sub, [], official=True), ["no question files given"])
        r = self.cli("rank", sub)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("no question files given", r.stdout)

    def test_duplicate_within_one_question_file(self):
        rows = q_rows("pub", 3)
        rows[2][0] = rows[1][0]
        qf = write(self.p("q_dup.csv"), QH, rows)
        sub = write(self.p("s_dup.csv"), ["question_id", "pred_rank"], [["pub_q_0001", "D"], ["pub_q_0002", "D"]])
        self.assertEqual(cs.check("rank", sub, [qf]), [f"question pub_q_0002 appears twice in {qf}"])
        other = write(self.p("q_other.csv"), QH, q_rows("pub", 1))
        errs = cs.check("rank", sub, [other, write(self.p("q_two.csv"), QH, q_rows("pub", 2))])
        self.assertIn("question pub_q_0001 appears in two question files", errs)


class Arguments(unittest.TestCase):
    """Bad argv: usage on stderr, exit code 1, never a traceback."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.q = write(os.path.join(self.tmp, "q.csv"), ["question_id", "num_games"], [["pub_q_0001", 5]])
        self.sub = write(os.path.join(self.tmp, "s.csv"), ["question_id", "pred_rank"], [["pub_q_0001", "D"]])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", TOOL] + list(args), capture_output=True, text=True,
                              timeout=60)

    def assert_usage_error(self, *args):
        r = self.run_cli(*args)
        self.assertEqual(r.returncode, 1, (args, r.stdout, r.stderr))
        self.assertNotIn("Traceback", r.stderr + r.stdout, args)
        self.assertIn("usage:", r.stderr, args)
        return r

    def test_positional_before_flag(self):
        r = self.assert_usage_error("rank", self.sub, self.q, "--questions", self.q)
        self.assertIn("unexpected argument", r.stderr)

    def test_other_bad_argv(self):
        self.assert_usage_error()
        self.assert_usage_error("ranks", self.sub, "--questions", self.q)
        self.assert_usage_error("rank")
        self.assert_usage_error("rank", "--questions", self.q)
        self.assert_usage_error("rank", self.sub, "--questoins", self.q)
        self.assert_usage_error("rank", self.sub, "--questions", self.q, "--candidates", self.q)
        self.assert_usage_error("rank", self.sub, "--official", self.q)

    def test_help(self):
        for args in (["--help"], ["-h"], ["rank", self.sub, "--help"]):
            r = self.run_cli(*args)
            self.assertEqual(r.returncode, 0, args)
            self.assertIn("usage:", r.stdout)

    def test_missing_file_is_a_clean_failure(self):
        r = self.run_cli("rank", self.sub, "--questions", os.path.join(self.tmp, "nope.csv"))
        self.assertEqual(r.returncode, 1)
        self.assertNotIn("Traceback", r.stderr + r.stdout)
        self.assertIn("FAIL: cannot read an input file", r.stdout)

    def test_default_mode_ok_message_unchanged(self):
        r = self.run_cli("rank", self.sub, "--questions", self.q)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip(),
                         "OK: rank submission covers all 1 questions and passes every strict check")


if __name__ == "__main__":
    unittest.main()
