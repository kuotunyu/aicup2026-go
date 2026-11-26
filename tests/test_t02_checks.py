"""Tests for tools/t02_checks.py (test-day checks T1-T6, spec results/test02/design/spec.md section 4).

Synthetic data only: nothing here reads a training, validation or test file, except the last class,
which checks the committed null in results/test02/null when it exists. Run:
    python -m pytest -q tests/test_t02_checks.py
"""

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
import p02_moves  # noqa: E402
import t02_checks as tc  # noqa: E402

NEEDLE = "ans" + "wers"
LETTERS = "abcdefghijklmnopqrs"
ALL_POINTS = [a + b for a in LETTERS for b in LETTERS]
OPENINGS = ["pd", "qd", "dp", "pp", "dd", "cd", "qc", "dc"]      # first-move classes in S99


def random_game(rng, n=15, first=None):
    """n moves on distinct points, alternating from B; the first move in an S99 class."""
    f = first or rng.choice(OPENINGS)
    pts = [f] + rng.sample([p for p in ALL_POINTS if p != f], n - 1)
    return [("B" if i % 2 == 0 else "W", p) for i, p in enumerate(pts)]


def sgf_of(mv, head="SZ[19]"):
    return "(;" + head + "".join(";%s[%s]" % (c, p) for c, p in mv) + ")"


def write_csv(path, header, rows, lt="\n", bom=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=lt)
    w.writerow(header)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    with open(path, "wb") as f:
        f.write((b"\xef\xbb\xbf" if bom else b"") + data)


_RNG = random.Random(20260926)
BASE = [sgf_of(random_game(_RNG, 12 + i % 9)) for i in range(3000)]


def question_rows(pre, n=400, games=BASE, offset=0):
    rows = []
    for i in range(n):
        k = tc.K_MIN + i % 16
        sg = [games[(offset + 20 * i + j) % len(games)] for j in range(k)] + [""] * (20 - k)
        cl = [("B" if (i + j) % 2 == 0 else "W") for j in range(k)] + [""] * (20 - k)
        rows.append(["%sq_%04d" % (pre, i + 1), str(k)] + sg + cl)
    return rows


def candidate_rows(pre, n_players=400, per=100, games=BASE, offset=0):
    rows = []
    for i in range(n_players):
        pid = "%sp_%016x" % (pre, 7919 * i + (1 if pre == "pub_" else 2))
        for j in range(per):
            g = i * per + j
            rows.append(["%sg_%06d" % (pre, g), pid, games[(offset + g) % len(games)], "B" if g % 2 == 0 else "W"])
    return rows


def build_tree(root, comp, lt="\n", bom=(), mutate=None):
    """Official-shaped synthetic files in root/data/test; mutate(files) may edit {name: [header, rows]}."""
    files = {}
    for pool, kind, name in tc.FILES[comp]:
        pre = tc.ID_PREFIX[pool]
        off = 0 if pool == "public" else 1000
        if kind == "questions":
            files[name] = [list(tc.Q_COLS), question_rows(pre, offset=off)]
        else:
            files[name] = [list(tc.C_COLS), candidate_rows(pre, offset=off)]
    if mutate:
        mutate(files)
    d = os.path.join(root, "data", "test")
    for name, (hdr, rows) in files.items():
        if rows is None:
            continue
        write_csv(os.path.join(d, name), hdr, rows, lt=lt(name) if callable(lt) else lt, bom=name in bom)
    return d


class TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="t02checks_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------- keys
class GameKeys(unittest.TestCase):
    def games(self):
        rng = random.Random(3)
        out = [random_game(rng, n) for n in (1, 2, 5, 11, 12, 13, 59, 60, 61, 99, 100, 101, 180)]
        # symmetric heads: tengen then mirrored pairs -> several symmetries tie on the first 12 moves
        sym = ["jj", "dd", "pp", "dp", "pd", "jd", "jp", "dj", "pj", "cc", "qq", "cq"]
        rest = [p for p in ALL_POINTS if p not in sym]
        for extra in (0, 3, 40, 120):
            pts = sym + rng.sample(rest, extra)
            out.append([("B" if i % 2 == 0 else "W", p) for i, p in enumerate(pts)])
        out.append([("B" if i % 2 == 0 else "W", p) for i, p in enumerate(["jj", "jk", "kj"])])
        return out

    def test_tie_speedup_equals_bruteforce(self):
        for mv in self.games():
            pts = [p for _, p in mv]
            brute = min("".join(p02_moves.SYM_TABLE[s][p] for p in pts) for s in range(8))
            self.assertEqual(tc.canon_points(pts), brute)
            for n in (1, 5, 12, 60, 100):
                if len(pts) >= n:
                    b = min("".join(p02_moves.SYM_TABLE[s][p] for p in pts[:n]) for s in range(8))
                    self.assertEqual(tc.canon_points(pts, n), b)

    def test_key_invariant_under_d4_and_colour(self):
        for mv in self.games():
            pts = [p for _, p in mv]
            k = tc.game_key(pts)
            for s in range(8):
                self.assertEqual(tc.game_key([p for _, p in p02_moves.transform(mv, s)]), k)
            flipped = [("W" if c == "B" else "B", p) for c, p in mv]      # colour letters are left out
            self.assertEqual(tc.game_key([p for _, p in flipped]), k)
            self.assertEqual(len(k), tc.KEY_BYTES)
        self.assertIsNone(tc.game_key([]))

    def test_prefix_keys_equal_direct_keys(self):
        for mv in self.games():
            pts = [p for _, p in mv]
            kf, k100, k60 = tc.game_keys(pts)
            self.assertEqual(kf, tc.game_key(pts))
            self.assertEqual(k100, tc.game_key(pts, 100) if len(pts) >= 100 else None)
            self.assertEqual(k60, tc.game_key(pts, 60) if len(pts) >= 60 else None)

    def test_different_games_differ(self):
        rng = random.Random(5)
        keys = {tc.game_key([p for _, p in random_game(rng, 30)]) for _ in range(500)}
        self.assertEqual(len(keys), 500)

    def test_first_class(self):
        for p in ("pd", "dp", "pp", "dd"):
            self.assertEqual(tc.first_class(p), "dd")
        self.assertEqual(tc.first_class("qd"), "cd")
        self.assertEqual(tc.first_class("jj"), "jj")
        self.assertIsNone(tc.first_class(None))
        self.assertTrue({tc.first_class(p) for p in OPENINGS} <= set(tc.S99))


# ---------------------------------------------------------------- thresholds and rules
class Thresholds(unittest.TestCase):
    def test_t2_threshold(self):
        for n, want in ((0, 10), (4999, 10), (10000, 10), (10001, 11), (40000, 40), (45123, 46), (5200, 10)):
            self.assertEqual(tc.t2_threshold(n), want, n)
            self.assertEqual(tc.t2_threshold(n), max(10, math.ceil(0.001 * n)), n)

    def test_t3_needed_and_rule(self):
        self.assertEqual(tc.t3_needed(400), 40)
        self.assertEqual(tc.t3_needed(226), 23)
        self.assertEqual(tc.t3_needed(401), 41)
        self.assertEqual(tc.t3_needed(10), 1)
        self.assertTrue(tc.t3_pool_overlap(40, 0, 400))
        self.assertTrue(tc.t3_pool_overlap(0, 40, 400))
        self.assertFalse(tc.t3_pool_overlap(39, 39, 400))
        self.assertTrue(tc.t3_pool_overlap(23, 0, 226))
        self.assertFalse(tc.t3_pool_overlap(22, 22, 226))

    def test_q99_higher_on_a_synthetic_null(self):
        rng = np.random.default_rng(0)
        null = rng.random(1985)
        thr = tc.q99(null)
        self.assertEqual(thr, float(np.sort(null)[math.ceil(0.99 * 1984)]))
        self.assertIn(thr, null)
        self.assertEqual(int((null > thr).sum()), 19)       # same count as the pinned 19 of 1,985
        # a pool of 400 null candidates stays far below the 40 needed
        pool = rng.random(400)
        self.assertFalse(tc.t3_pool_overlap(int((pool > thr).sum()), int((pool > thr).sum()), 400))

    def test_check_pins(self):
        t3 = {"reference": dict(tc.PINS["reference"]),
              "orient": {o: dict(tc.PINS[o]) for o in tc.ORIENTS}}
        self.assertEqual(tc.check_pins(t3), [])
        t3["orient"]["raw"]["S2_q99"] = 0.1004267242871778
        t3["orient"]["canon"]["vocab"] = 1
        self.assertEqual(len(tc.check_pins(t3)), 2)


# ---------------------------------------------------------------- SGF grammar
class SgfGrammar(unittest.TestCase):
    def test_clean_fast_path(self):
        g = tc.sgf_check("(;SZ[19];B[pd];W[dp];B[pp])")
        self.assertFalse(g["a_bad"] or g["b_bad"])
        self.assertEqual(g["n_moves"], 3)
        self.assertEqual(g["first"], "pd")

    def test_setup_size_pass_order(self):
        cases = {
            "(;SZ[19]AB[dd][pp];W[dp];B[pd])": ("a", "setup"),
            "(;SZ[19]HA[2];W[dp];B[pd])": ("a", "setup"),
            "(;SZ[13];B[dd];W[jj])": ("a", "SZ"),
            "(;SZ[19];B[pd];W[];B[dp])": ("b", "pass"),
            "(;SZ[19];B[pd];W[tt];B[dp])": ("b", "pass"),
            "(;SZ[19];W[pd];B[dp])": ("b", "first"),
            "(;SZ[19];B[pd];B[dp];W[pp])": ("b", "not"),
            "(;SZ[19])": ("b", "no"),
        }
        for sgf, (part, word) in cases.items():
            g = tc.sgf_check(sgf)
            self.assertTrue(g["a_bad"] if part == "a" else g["b_bad"], sgf)
            self.assertTrue(any(r.startswith(word) for r in g["reasons"]), (sgf, g["reasons"]))

    def test_record_only_facts(self):
        g = tc.sgf_check("(;GM[1]FF[4]KM[6.5]C[a \\] comment];B[pd];W[dp])")
        self.assertIsNone(g["sz"])                      # missing SZ: recorded, not a violation
        self.assertFalse(g["a_bad"] or g["b_bad"])
        self.assertEqual(set(g["other_props"]), {"GM", "FF", "KM", "C"})
        self.assertIsNone(g["parse_error"])
        g = tc.sgf_check("(;SZ[19]B[pd];W[dp])")          # move in the root node: regex misses it
        self.assertTrue(g["regex_mismatch"])
        self.assertTrue(g["b_bad"])                      # the MOVE_RE sequence starts with W
        g = tc.sgf_check("(;SZ[19];B[pd];W[dp] junk ])")
        self.assertIsNotNone(g["parse_error"])


# ---------------------------------------------------------------- T1
class T1(TmpDir):
    def t1(self, comp, **kw):
        d = build_tree(self.tmp, comp, **kw)
        return tc.t1_check(comp, d)

    def test_rank_pass_lf_and_crlf(self):
        for lt in ("\n", "\r\n"):
            r = tc.t1_check("rank", build_tree(tempfile.mkdtemp(dir=self.tmp), "rank", lt=lt))
            self.assertEqual(r["verdict"], "PASS", r["subchecks"])
            self.assertEqual(r["normalize"], {})
            self.assertEqual(r["normalize_transforms"], [])
            self.assertTrue(r["complete"])
            self.assertEqual(r["subchecks"]["T1.4"]["verdict"], "PASS")
            self.assertEqual(r["subchecks"]["T1.5"]["verdict"], "PASS")

    def test_player_pass(self):
        r = self.t1("player")
        self.assertEqual(r["verdict"], "PASS", r["subchecks"])
        self.assertEqual(r["subchecks"]["T1.2"]["files"]["cross"],
                         {"question_ids_in_both_files": 0, "player_ids_in_both_pools": 0, "game_ids_in_both_pools": 0})
        info = r["files"]["player_identification_candidates_public.csv"]
        self.assertEqual(info["rows"], 40000)
        self.assertEqual(len(info["sha256"]), 64)

    def test_normalizations(self):
        name = "rank_prediction_test_public.csv"

        def lower(files):
            files[name][1][0][22] = "b"

        def reorder(files):
            hdr, rows = files[name]
            order = [1, 0] + list(range(2, 42))
            files[name] = [[hdr[j] for j in order], [[r[j] for j in order] for r in rows]]

        def whitespace(files):
            files[name][1][0][2 + 19] = "  "            # sgf_20 of a 5-game question

        def colour_after(files):                        # PREREG_02 rev 8: 'color_1..color_20 always B'
            for r in files[name][1]:
                k = int(r[1])
                for i in range(k + 1, 21):
                    r[21 + i] = "B"

        def lower_colour_after(files):
            files[name][1][0][21 + 20] = "w"            # color_20 of a 5-game question, lower case

        def whitespace_sgf_and_colour(files):
            files[name][1][0][1 + 20] = " "             # sgf_20 whitespace only
            files[name][1][0][21 + 20] = "B"            # color_20 filled

        cases = [({"bom": (name,)}, ["strip_bom"]), ({"mutate": lower}, ["upper_colour"]),
                 ({"mutate": reorder}, ["reorder_columns_to_readme"]),
                 ({"mutate": whitespace}, ["clear_slots_after_num_games_if_empty_whitespace"]),
                 ({"mutate": colour_after}, ["clear_colour_after_num_games_if_sgf_empty"]),
                 ({"mutate": lower_colour_after}, ["clear_colour_after_num_games_if_sgf_empty"]),
                 ({"mutate": whitespace_sgf_and_colour}, ["clear_slots_after_num_games_if_empty_whitespace",
                                                          "clear_colour_after_num_games_if_sgf_empty"])]
        for kw, want in cases:
            r = tc.t1_check("rank", build_tree(tempfile.mkdtemp(dir=self.tmp), "rank", **kw))
            self.assertEqual(r["verdict"], "NORMALIZED", (kw, r["subchecks"]))
            self.assertEqual(r["normalize"], {name: want})
            self.assertEqual(r["normalize_transforms"], want)
            self.assertTrue(set(r["normalize_transforms"]) <= set(tc.NORMALIZE_WHITELIST))
        r = tc.t1_check("rank", build_tree(tempfile.mkdtemp(dir=self.tmp), "rank", mutate=colour_after))
        counts = r["subchecks"]["T1.3"]["files"][name]["counts"]
        self.assertEqual(counts["colour_after_num_games_with_empty_sgf"], sum(20 - (5 + i % 16) for i in range(400)))
        self.assertNotIn("content_after_num_games", counts)
        self.assertEqual(r["subchecks"]["T1.3"]["verdict"], "NORMALIZED")

    def test_official_colour_design_is_record_only(self):
        """PREREG_02 rev 8: competition 1 black only, competition 2 one colour per question; recorded, never a
        verdict (the synthetic files here alternate colours inside a question)."""
        r = self.t1("rank")
        f = r["subchecks"]["T1.3"]["files"]["rank_prediction_test_public.csv"]["official_colour_design_record_only"]
        self.assertEqual(r["verdict"], "PASS")
        self.assertFalse(f["matches_official_design"])
        self.assertEqual(f["questions_with_both_colours"], 400)

        def single(files):
            for name, (hdr, rows) in files.items():
                for j, row in enumerate(rows):
                    k = int(row[1])
                    for i in range(1, 21):
                        row[21 + i] = ("B" if j % 2 else "W") if name.startswith("rank") else "B"
                        if i > k:
                            row[21 + i] = row[22]            # filled with the question colour
        r = tc.t1_check("rank", build_tree(tempfile.mkdtemp(dir=self.tmp), "rank", mutate=single))
        self.assertEqual(r["verdict"], "NORMALIZED")
        f = r["subchecks"]["T1.3"]["files"]["rank_prediction_test_private.csv"]["official_colour_design_record_only"]
        self.assertTrue(f["matches_official_design"])
        self.assertEqual((f["questions_with_both_colours"], f["colour_after_num_games_not_the_question_colour"]), (0, 0))
        self.assertEqual(f["counts"]["questions_single_colour_B"] + f["counts"]["questions_single_colour_W"], 400)

        def black_only(files):
            for name, (hdr, rows) in files.items():
                for row in rows:
                    if len(row) == 4:
                        row[3] = "B"
                    else:
                        for i in range(1, 21):
                            row[21 + i] = "B"
        r = tc.t1_check("player", build_tree(tempfile.mkdtemp(dir=self.tmp), "player", mutate=black_only))
        self.assertEqual(r["verdict"], "NORMALIZED", r["subchecks"]["T1.3"]["problems"])
        self.assertEqual(r["normalize_transforms"], ["clear_colour_after_num_games_if_sgf_empty"])
        for name, d in r["subchecks"]["T1.3"]["files"].items():
            if name != "cross":
                self.assertTrue(d["official_colour_design_record_only"]["matches_official_design"], name)
        r = self.t1("player")                           # the default synthetic pools are mixed: recorded only
        d = r["subchecks"]["T1.3"]["files"]["player_identification_candidates_public.csv"]
        self.assertEqual(d["official_colour_design_record_only"]["rows_not_B"], 20000)
        self.assertEqual(r["verdict"], "PASS")

    def test_mixed_line_ends(self):
        d = build_tree(self.tmp, "rank")
        p = os.path.join(d, "rank_prediction_test_private.csv")
        with open(p, "rb") as f:
            data = f.read()
        with open(p, "wb") as f:
            f.write(data.replace(b"\n", b"\r\n", 5))
        r = tc.t1_check("rank", d)
        self.assertEqual(r["verdict"], "NORMALIZED")
        self.assertEqual(r["normalize"], {"rank_prediction_test_private.csv": ["crlf_to_lf"]})

    def test_struct_failures(self):
        pub, priv = "rank_prediction_test_public.csv", "rank_prediction_test_private.csv"

        def drop_row(f):
            f[pub][1].pop()

        def prefix(f):
            f[pub][1][3][0] = "priv_q_9999"

        def dup_across(f):
            f[priv][1][0][0] = f[pub][1][0][0].replace("pub_", "priv_")
            f[pub][1][0][0] = "pub_q_0002"

        def num21(f):
            f[pub][1][0][1] = "21"

        def after_n(f):
            f[pub][1][0][2 + 10] = BASE[0]             # sgf_11 of a 5-game question

        def empty_within(f):
            f[pub][1][10][2] = ""

        def bad_colour(f):
            f[pub][1][0][22] = "X"

        def missing(f):
            f[priv][1] = None

        def extra_col(f):
            f[pub][0] = f[pub][0] + ["note"]
            for r in f[pub][1]:
                r.append("")

        def after_n_with_colour(f):                    # rev 8 clears colours only; an sgf after n stays FAIL
            f[pub][1][0][1 + 11] = BASE[0]
            f[pub][1][0][21 + 11] = "B"

        def odd_colour_after_n(f):
            f[pub][1][0][21 + 12] = "X"

        def padded_colour_after_n(f):
            f[pub][1][0][21 + 12] = " B"

        for fn in (drop_row, prefix, dup_across, num21, after_n, empty_within, bad_colour, missing, extra_col,
                   after_n_with_colour, odd_colour_after_n, padded_colour_after_n):
            r = tc.t1_check("rank", build_tree(tempfile.mkdtemp(dir=self.tmp), "rank", mutate=fn))
            self.assertEqual(r["verdict"], "FAIL_STRUCT", fn.__name__)
            self.assertTrue(any(r["subchecks"][s]["problems"] for s in tc.T1_STRUCT), fn.__name__)

    def test_player_struct_failures(self):
        cp, cq = "player_identification_candidates_public.csv", "player_identification_candidates_private.csv"

        def games99(f):
            rows = f[cp][1]
            rows[99][1] = rows[100][1]                  # candidate 0 has 99 games, candidate 1 has 101

        def shared_game(f):
            f[cq][1][5][0] = f[cp][1][5][0]

        def shared_player(f):
            pid = f[cp][1][0][1]
            for r in f[cq][1][:100]:
                r[1] = pid

        def wrong_player_prefix(f):
            for r in f[cp][1][:100]:
                r[1] = r[1].replace("pub_p_", "pub_x_")

        for fn in (games99, shared_game, shared_player, wrong_player_prefix):
            r = tc.t1_check("player", build_tree(tempfile.mkdtemp(dir=self.tmp), "player", mutate=fn))
            self.assertEqual(r["verdict"], "FAIL_STRUCT", fn.__name__)
            self.assertTrue(r["subchecks"]["T1.2"]["problems"], fn.__name__)

    def test_t1_4_threshold(self):
        name = "rank_prediction_test_public.csv"
        rng = random.Random(9)

        def setup(share_games):
            def m(f):
                done = 0
                for r in f[name][1]:
                    for j in range(int(r[1])):
                        if done < share_games:
                            r[2 + j] = sgf_of(random_game(rng, 12), head="SZ[19]AB[dd]")
                            done += 1
            return m

        r = tc.t1_check("rank", build_tree(tempfile.mkdtemp(dir=self.tmp), "rank", mutate=setup(200)))
        self.assertEqual(r["verdict"], "FAIL_OPENING")        # about 200 of 5,000 games = 4%
        self.assertEqual(r["subchecks"]["T1.4"]["verdict"], "FAIL")
        f4 = r["subchecks"]["T1.4"]["files"][name]
        self.assertEqual(f4["violating_games"], 200)
        self.assertLessEqual(len(f4["examples"]), tc.MAX_EXAMPLES)
        r = tc.t1_check("rank", build_tree(tempfile.mkdtemp(dir=self.tmp), "rank", mutate=setup(20)))
        self.assertEqual(r["verdict"], "PASS")                # 0.4%: recorded only
        self.assertEqual(r["subchecks"]["T1.4"]["files"][name]["violating_games"], 20)

    def test_t1_5_not_from_move_one(self):
        name = "player_identification_candidates_private.csv"
        rng = random.Random(11)
        edge = [p for p in ALL_POINTS if tc.first_class(p) not in tc.S99]

        def later_start(f):
            for i, r in enumerate(f[name][1]):
                if i % 5 != 0:        # 80% of games start somewhere off the usual opening points
                    r[2] = sgf_of(random_game(rng, 12, first=rng.choice(edge)))

        r = tc.t1_check("player", build_tree(self.tmp, "player", mutate=later_start))
        self.assertEqual(r["verdict"], "FAIL_OPENING")
        self.assertEqual(r["subchecks"]["T1.5"]["verdict"], "FAIL")
        self.assertLess(r["subchecks"]["T1.5"]["files"][name]["share"], 0.95)
        self.assertEqual(r["subchecks"]["T1.4"]["verdict"], "PASS")

    def test_only_structure(self):
        name = "rank_prediction_test_public.csv"

        def bad_games(f):
            for r in f[name][1]:
                r[2] = "(;SZ[19];W[pd];B[dp])"

        d = build_tree(self.tmp, "rank", mutate=bad_games)
        r = tc.t1_check("rank", d, only=tc.T1_STRUCT)
        self.assertEqual(r["verdict"], "PASS")
        self.assertFalse(r["complete"])
        self.assertEqual(r["subchecks"]["T1.4"]["verdict"], "NOT_RUN")
        self.assertEqual(tc.t1_check("rank", d)["verdict"], "FAIL_OPENING")
        with self.assertRaises(SystemExit):
            tc.t1_check("rank", d, only=("T1.4",))


# ---------------------------------------------------------------- CLI, modes and guards
class Cli(TmpDir):
    def setUp(self):
        super().setUp()
        self._orig = tc._validation_sgf_hashes
        tc._validation_sgf_hashes = lambda: {b"not-a-real-hash-of-any-game"}

    def tearDown(self):
        tc._validation_sgf_hashes = self._orig
        super().tearDown()

    def main(self, *argv):
        with redirect_stdout(io.StringIO()):
            return tc.main(list(argv))

    def test_t1_dry_run_writes_derived_out_and_refuses_overwrite(self):
        d = build_tree(self.tmp, "rank")
        self.assertEqual(self.main("t1", "--comp", "rank", "--tests-dir", d, "--dry-run"), 0)
        p = os.path.join(self.tmp, "results", "test02", "checks", "rank", "t1.json")
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["provenance"]["mode"], "dry-run")
        self.assertEqual(r["provenance"]["validation_hash_guard"]["hits"], 0)
        with open(os.path.join(os.path.dirname(p), "summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        self.assertEqual(s["checks"]["T1"]["verdict"], "PASS")
        self.assertEqual(s["checks"]["T1"]["details_file"], "t1.json")
        with self.assertRaises(SystemExit):
            self.main("t1", "--comp", "rank", "--tests-dir", d, "--dry-run")
        with self.assertRaises(SystemExit):
            self.main("t1", "--comp", "rank", "--tests-dir", d, "--dry-run", "--rerun")
        self.assertEqual(self.main("t1", "--comp", "rank", "--tests-dir", d, "--dry-run", "--rerun",
                                   "--reason", "second look"), 0)
        with open(p, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["provenance"]["rerun_reason"], "second look")

    def test_exit_codes(self):
        name = "rank_prediction_test_public.csv"
        d = build_tree(os.path.join(self.tmp, "a"), "rank", bom=(name,))
        self.assertEqual(self.main("t1", "--comp", "rank", "--root", os.path.join(self.tmp, "a"), "--dry-run"), 0)
        self.assertEqual(self.main("t1", "--comp", "rank", "--tests-dir", d, "--out", os.path.join(self.tmp, "o2"),
                                   "--dry-run", "--require-pass"), 4)

        def drop(f):
            f[name][1].pop()
        d = build_tree(os.path.join(self.tmp, "b"), "rank", mutate=drop)
        self.assertEqual(self.main("t1", "--comp", "rank", "--tests-dir", d, "--dry-run"), 3)

        def w_first(f):
            for r in f[name][1]:
                r[2] = "(;SZ[19];W[pd];B[dp])"
        d = build_tree(os.path.join(self.tmp, "c"), "rank", mutate=w_first)
        self.assertEqual(self.main("t1", "--comp", "rank", "--tests-dir", d, "--dry-run"), 2)

    def test_out_dir_guards(self):
        d = build_tree(self.tmp, "rank")
        for out in (os.path.join(REPO, "results", "prereg02", "x"), os.path.join(REPO, "results", "prereg01"),
                    os.path.join(REPO, "data", "local", "x"), os.path.join(REPO, "results", "test02", "checks_dry")):
            with self.assertRaises(SystemExit):
                self.main("t1", "--comp", "rank", "--tests-dir", d, "--out", out, "--dry-run")
        with self.assertRaises(SystemExit):              # real mode reads data/test only
            self.main("t1", "--comp", "rank", "--tests-dir", d)
        with self.assertRaises(SystemExit):              # --root is not for real mode
            self.main("t1", "--comp", "rank", "--root", self.tmp)
        with self.assertRaises(SystemExit):              # no derivable out dir
            self.main("t1", "--comp", "rank", "--tests-dir", self.tmp, "--dry-run")
        with self.assertRaises(SystemExit):
            self.main("run", "--comp", "rank", "--tests-dir", d, "--out", os.path.join(REPO, "results", "prereg02"),
                      "--dry-run")
        with self.assertRaises(SystemExit):
            self.main("precompute", "--dry-run", "--out", os.path.join(REPO, "results", "test02", "null_x"))
        with self.assertRaises(SystemExit):
            self.main("precompute", "--out", os.path.join(self.tmp, "null"))       # real mode: fixed null dir

    def test_forbidden_basename(self):
        p = os.path.join(self.tmp, "rank_val_%s.csv" % NEEDLE)
        with self.assertRaises(SystemExit):
            tc.refuse_forbidden(p)
        with self.assertRaises(SystemExit):
            tc.Table(p)
        self.assertEqual(tc.refuse_forbidden(os.path.join(self.tmp, "x.csv")), os.path.join(self.tmp, "x.csv"))

    def test_smoke_and_rehearsal_structure_only(self):
        d = build_tree(self.tmp, "rank")
        for m in ("--smoke", "--rehearsal"):
            with self.assertRaises(SystemExit):
                self.main("t1", "--comp", "rank", "--tests-dir", d, m, "--out", os.path.join(self.tmp, "o" + m))
            with self.assertRaises(SystemExit):
                self.main("run", "--comp", "rank", "--tests-dir", d, m, "--out", os.path.join(self.tmp, "r" + m))
            self.assertEqual(self.main("t1", "--comp", "rank", "--tests-dir", d, m, "--only", "T1.1,T1.2,T1.3",
                                       "--out", os.path.join(self.tmp, "s" + m)), 0)

    def test_validation_games_refused(self):
        d = build_tree(self.tmp, "rank")
        tc._validation_sgf_hashes = lambda: {hashlib.sha1(BASE[3].encode()).digest()}
        with self.assertRaises(SystemExit) as cm:
            self.main("t1", "--comp", "rank", "--tests-dir", d, "--dry-run")
        self.assertIn("validation", str(cm.exception.code))
        self.assertEqual(self.main("t1", "--comp", "rank", "--tests-dir", d, "--dry-run", "--only",
                                   "T1.1,T1.2,T1.3"), 0)
        tc._validation_sgf_hashes = lambda: set()           # no guard available -> refuse
        with self.assertRaises(SystemExit):
            self.main("t1", "--comp", "rank", "--tests-dir", d, "--dry-run", "--rerun", "--reason", "x")

    def test_real_mode_path_pairs(self):
        import argparse
        ns = lambda **k: argparse.Namespace(**dict({"tests_dir": None, "out": None, "root": None}, **k))
        self.assertEqual(tc.resolve_paths(ns(), "real", tc.REAL_T1), (os.path.abspath(tc.DATA_TEST_RAW),
                                                                      os.path.abspath(tc.CHECKS_DIR)))
        # the bundle build re-checks the built copies into checks/built (t02_test_bundle.run_t1_recheck)
        self.assertEqual(tc.resolve_paths(ns(tests_dir=tc.DATA_TEST, out=tc.CHECKS_BUILT), "real", tc.REAL_T1),
                         (os.path.abspath(tc.DATA_TEST), os.path.abspath(tc.CHECKS_BUILT)))
        with self.assertRaises(SystemExit):
            tc.resolve_paths(ns(tests_dir=tc.DATA_TEST_RAW, out=tc.CHECKS_BUILT), "real", tc.REAL_T1)
        with self.assertRaises(SystemExit):
            tc.resolve_paths(ns(tests_dir=tc.DATA_TEST_RAW), "real", tc.REAL_RUN)
        self.assertEqual(tc.resolve_paths(ns(), "real", tc.REAL_RUN)[1], os.path.abspath(tc.CHECKS_DIR))
        with self.assertRaises(SystemExit):
            tc.resolve_paths(ns(tests_dir=os.path.join(REPO, "data", "local")), "dry-run", tc.REAL_RUN)

    def test_infer_root(self):
        self.assertEqual(tc._infer_root(os.path.join(self.tmp, "data", "test")), os.path.abspath(self.tmp))
        self.assertEqual(tc._infer_root(os.path.join(self.tmp, "data", "test", "raw")), os.path.abspath(self.tmp))
        self.assertIsNone(tc._infer_root(self.tmp))


# ---------------------------------------------------------------- synthetic training data for T2 / T3
def synthetic_train(n_fit=24, n_inner=6, games=120, seed=1, long_games=False):
    """Rows like c10data.load_rows(part='train'): each player has a fixed personal opening."""
    rng = random.Random(seed)
    rows, inner = [], set()
    for i in range(n_fit + n_inner):
        pid = "%032x" % (0xabc000 + i)
        if i >= n_fit:
            inner.add(pid)
        style = rng.sample(ALL_POINTS, 8)
        lv = tc.c10data.LEVELS[i % 10]
        for j in range(games):
            colour = "B" if j % 2 == 0 else "W"
            rest = [p for p in ALL_POINTS if p not in style]
            n_mv = (110 + (j % 30)) if long_games else 30
            pts = style[:6] + rng.sample(rest, n_mv - 6) if colour == "B" else \
                [rng.choice(OPENINGS)] + style[:5] + rng.sample(rest, n_mv - 6)
            if len(set(pts)) != len(pts):
                pts = list(dict.fromkeys(pts)) + rng.sample([p for p in rest if p not in pts], len(pts) - len(set(pts)))
            mv = [("B" if k % 2 == 0 else "W", p) for k, p in enumerate(pts)]
            rows.append({"player_id": pid, "game_id": "g_%07d" % (1000 * i + j), "rank": lv,
                         "color": colour, "sgf_content": sgf_of(mv)})
    return rows, inner


class FakeFile:
    """Stands in for t02_checks.TestFile (pool, kind, name, games, owners)."""

    def __init__(self, pool, kind, name, games, num_games=None):
        self.pool, self.kind, self.name = pool, kind, name
        self.games = []
        for owner, label, colour, sgf in games:
            full = tc.MOVE_RE.findall(sgf)
            self.games.append({"owner": owner, "label": label, "colour": colour, "sgf": sgf,
                               "pts": [p for _, p in full], "first": full[0][1] if full else None})
        self.owners = sorted({g["owner"] for g in self.games}) if kind == "candidates" else \
            list(dict.fromkeys(g["owner"] for g in self.games))
        self.num_games = num_games or []
        self.sha256 = "0" * 64

    def n_games(self):
        return len(self.games)


class T2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.inner = synthetic_train(n_fit=8, n_inner=4, games=40, seed=2, long_games=True)
        cls.T = tc.Train(cls.rows, cls.inner, keys=True, extra_lengths=[50])

    def index(self, part="train", black=True):
        T = self.T
        mask = np.ones(T.n, bool) if part == "train" else ~T.inner_game
        return tc.TrainIndex(T.keys, T.extra, T.pid_idx, T.pids, mask, part,
                             black=tc.train_black(T) if black else None)

    def fresh(self, rng, n=120):
        return sgf_of(random_game(rng, n))

    def test_units(self):
        rng = random.Random(4)
        tr = [r["sgf_content"] for r in self.rows]
        rot = []
        for s in tr[:12]:                                 # 12 training games, rotated: still copies
            mv = tc.MOVE_RE.findall(s)
            rot.append(sgf_of(p02_moves.transform(mv, 5)))
        short = tc.MOVE_RE.findall(tr[45])[:50]           # 50-move prefix of a training game (K_50)
        q_pub = [("pub_q_%04d" % i, "pub_q_%04d/sgf_%d" % (i, j), "B", self.fresh(rng)) for i in range(20) for j in (1, 2)]
        for k, s in enumerate(rot):
            q_pub[k] = (q_pub[k][0], q_pub[k][1], "B", s)
        q_pub[30] = (q_pub[30][0], q_pub[30][1], "B", sgf_of(short))
        shared = self.fresh(rng)
        q_pub[31] = (q_pub[31][0], q_pub[31][1], "B", shared)          # (c) same colour across questions
        q_pub[33] = (q_pub[33][0], q_pub[33][1], "B", shared)
        opp = self.fresh(rng)
        q_pub[35] = (q_pub[35][0], q_pub[35][1], "B", opp)             # (c) opposite colour
        q_pub[37] = (q_pub[37][0], q_pub[37][1], "W", opp)
        c_pub = [("pub_p_%d" % (i // 5), "pub_g_%d" % i, "W", self.fresh(rng)) for i in range(40)]
        c_pub[0] = (c_pub[0][0], c_pub[0][1], "W", q_pub[20][3])       # (b) question game = candidate game
        q_priv = [("priv_q_%04d" % i, "priv_q_%04d/sgf_1" % i, "B", self.fresh(rng)) for i in range(20)]
        q_priv[0] = (q_priv[0][0], q_priv[0][1], "B", q_pub[25][3])    # (d)
        c_priv = [("priv_p_%d" % (i // 5), "priv_g_%d" % i, "B", self.fresh(rng)) for i in range(40)]
        files = [FakeFile("public", "questions", "player_identification_test_public.csv", q_pub),
                 FakeFile("public", "candidates", "player_identification_candidates_public.csv", c_pub),
                 FakeFile("private", "questions", "player_identification_test_private.csv", q_priv),
                 FakeFile("private", "candidates", "player_identification_candidates_private.csv", c_priv)]
        res = tc.t2_run("player", files, self.index())
        u = {(x["unit"], x.get("file") or x.get("pool")): x for x in res["units"]}
        a = u[("a", "player_identification_test_public.csv")]
        self.assertEqual(a["copies"], 13)                 # 12 rotated + the 50-move prefix
        self.assertEqual(a["games_under_100_moves"], 1)
        self.assertEqual(a["verdict"], "FAIL")            # >= max(10, ceil(0.001 x 40))
        self.assertEqual(a["threshold"], 10)
        self.assertEqual(a["questions_with_copy"], 7)     # 12 games in 6 questions + the short one
        self.assertEqual(a["questions_whose_copies_map_to_one_training_player"], 7)
        self.assertEqual(a["copies_per_question_histogram"], {2: 6, 1: 1})
        # PREREG_02 rev 8 sides (record only): the test colour is B; training rows 0-11 alternate B / W from B and
        # row 45 is W, so 6 copies are the same side and 7 the opponent's side; the FAIL stays on all 13
        self.assertEqual((a["same_side_copies_record_only"], a["opposite_side_copies_record_only"],
                          a["copies_matching_both_sides_record_only"], a["copies_side_unknown_record_only"]),
                         (6, 7, 0, 0))
        self.assertIn("sides", res["definitions"])
        res_nb = tc.t2_run("player", files, self.index(black=False))           # no colour in the index
        a_nb = [x for x in res_nb["units"] if x["unit"] == "a" and x["file"] == a["file"]][0]
        self.assertEqual((a_nb["copies"], a_nb["copies_side_unknown_record_only"], a_nb["verdict"]), (13, 13, "FAIL"))
        self.assertEqual(u[("a", "player_identification_candidates_public.csv")]["copies"], 0)
        self.assertEqual(u[("b", "public")]["copies"], 1)
        self.assertEqual(u[("b", "public")]["verdict"], "PASS")
        c = u[("c", "player_identification_test_public.csv")]
        self.assertEqual(c["copies"], 4)
        self.assertEqual(c["games_shared_with_same_colour"], 2)
        self.assertEqual(c["games_shared_with_opposite_colour"], 2)
        d = [x for x in res["units"] if x["unit"] == "d"][0]
        self.assertEqual(d["copies"], 1)
        self.assertEqual(res["verdict"], "FAIL")
        # the fit-only index (dry-run negative control) misses copies of inner players' games
        inner_games = [i for i in range(self.T.n) if self.T.inner_game[i]]
        q_in = [("pub_q_%04d" % i, "l%d" % i, "B", tr[g]) for i, g in enumerate(inner_games[:15])]
        f_in = [FakeFile("public", "questions", "rank_prediction_test_public.csv", q_in),
                FakeFile("private", "questions", "rank_prediction_test_private.csv", q_priv)]
        self.assertEqual(tc.t2_run("rank", f_in, self.index("fit"))["units"][0]["copies"], 0)
        r = tc.t2_run("rank", f_in, self.index("train"))
        self.assertEqual(r["units"][0]["copies"], 15)
        self.assertEqual(r["units"][0]["verdict"], "FAIL")
        self.assertEqual({x["unit"] for x in r["units"]}, {"a", "c", "d"})     # no (b) for rank

    def test_index_cache_roundtrip(self):
        tmp = tempfile.mkdtemp(prefix="t02idx_")
        try:
            p = os.path.join(tmp, "idx.npz")
            tc.save_index_cache(self.T, p)
            self.assertIsNone(tc.load_index_cache(p))                  # wrong game count -> stale
            keys, pid_idx, pids, inner_game, black = tc.load_index_cache(p, expected_games=self.T.n)
            for kt in ("full", "k100", "k60"):
                self.assertEqual(keys[kt], self.T.keys[kt])
            self.assertEqual(pids, self.T.pids)
            self.assertTrue(np.array_equal(inner_game, self.T.inner_game))
            self.assertEqual(black.tolist(), [c == "B" for c in self.T.colour])
            b1 = open(p, "rb").read()
            tc.save_index_cache(self.T, p)
            self.assertEqual(open(p, "rb").read(), b1)               # deterministic bytes
            # a cache of the rev 5 layout (no training colour) is stale and gets rebuilt
            z = tc.load_npz(p)
            del z["black"]
            meta = json.loads(str(z["meta"]))
            meta.pop("format")
            z["meta"] = np.array(json.dumps(meta, sort_keys=True))
            tc.save_npz(p, **z)
            with redirect_stdout(io.StringIO()):
                self.assertIsNone(tc.load_index_cache(p, expected_games=self.T.n))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_nul_ending_keys_survive_the_index(self):
        k = b"abcdefghij\x00\x00"
        ix = tc.KeyIndex([k, b"zzzzzzzzzzzz", None], [7, 8, 9])
        self.assertEqual(ix.lookup(k).tolist(), [7])
        self.assertEqual(ix.lookup(b"abcdefghij\x00\x01").tolist(), [])
        self.assertEqual(ix.lookup(None).tolist(), [])


class T3(unittest.TestCase):
    """The T3 engine on a small synthetic training set (null players need >= 100 games)."""

    @classmethod
    def setUpClass(cls):
        rows, inner = synthetic_train(n_fit=24, n_inner=4, games=120, seed=7)
        cls.rows = rows
        cls.T = tc.Train(rows, inner, keys=False)
        res, arrays = {}, {}
        for o in tc.ORIENTS:
            res[o], arrays[o] = tc.t3_orient_precompute(cls.T, o, power_sims=20, t3q_ks=range(5, 8))
        cls.null = {"orient": res}
        cls.npz = arrays

    def test_null_structure(self):
        for o in tc.ORIENTS:
            r = self.null["orient"][o]
            self.assertEqual(r["n_null"], 24)
            self.assertEqual(r["n_holdout_inner"], 4)
            self.assertEqual(r["S2_q99"], tc.q99(self.npz[o]["null_S2"]))
            self.assertEqual(r["idf_sha256"], tc.sha256_array(self.npz[o]["idf"].astype("<f8")))
            self.assertEqual(len(self.npz[o]["idf"]), r["vocab"])
            self.assertEqual(sorted(r["t3q"]["by_k"]), [5, 6, 7])
            # the idf formula of the registration: ln((1 + |R|) / (1 + df)) + 1
            self.assertAlmostEqual(float(self.npz[o]["idf"].max()), math.log(25.0) + 1.0)
        e = tc.t3_either(self.null["orient"], self.npz, sims=20)
        self.assertIn("P_declare_raw_or_canon", e)

    def test_rebuild_is_deterministic(self):
        again, _ = tc.t3_orient_precompute(self.T, "raw", power_sims=20, t3q_ks=range(5, 6))
        for k in ("S2_q99", "idf_sha256", "vocab_sha256", "vocab", "null_n_exceed"):
            self.assertEqual(again[k], self.null["orient"]["raw"][k])

    def files(self):
        by = {}
        for r in self.rows:
            by.setdefault(r["player_id"], []).append(r)
        fit = [p for p in sorted(by) if p not in self.T.inner_pids][:10]
        order = self.T.t3_order()
        gi = {(self.T.gid[i]): i for i in range(self.T.n)}
        c_pub = []
        for k, p in enumerate(fit):                       # training players: the positive pool
            for i in order[p][:100]:
                c_pub.append(("pub_p_%d" % k, "pub_g_%d" % i, self.T.colour[i], self.rows[i]["sgf_content"]))
        new_rows, _ = synthetic_train(n_fit=10, n_inner=0, games=100, seed=99)
        c_priv = [("priv_p_%s" % r["player_id"][-4:], "priv_g_%d" % j, r["color"], r["sgf_content"])
                  for j, r in enumerate(new_rows)]
        q_pub = [("pub_q_%d" % k, "x", self.T.colour[i], self.rows[i]["sgf_content"])
                 for k, p in enumerate(fit) for i in order[p][100:106]]
        q_priv = [("priv_q_%d" % k, "y", r["color"], r["sgf_content"]) for k, r in enumerate(new_rows[:60])]
        self.assertTrue(gi)
        return [FakeFile("public", "questions", "player_identification_test_public.csv", q_pub),
                FakeFile("public", "candidates", "player_identification_candidates_public.csv", c_pub),
                FakeFile("private", "questions", "player_identification_test_private.csv", q_priv),
                FakeFile("private", "candidates", "player_identification_candidates_private.csv", c_priv)]

    def test_pool_rule_positive_and_negative(self):
        rec, ok = tc.t3_run("player", self.files(), self.T, self.null, self.npz, "PASS", None, log_=lambda *a: None)
        self.assertTrue(ok)
        self.assertTrue(all(v["passed"] for v in rec["reference_check"].values()))
        pub, priv = rec["pools"]["public"], rec["pools"]["private"]
        self.assertEqual(pub["needed"], 1)
        self.assertEqual(pub["n_candidates"], 10)
        # training players profiled on games that are in R stand out; new players much less
        self.assertGreaterEqual(pub["raw"]["n_above"], 8)
        self.assertGreaterEqual(pub["canon"]["n_above"], 8)
        self.assertLess(priv["raw"]["n_above"], pub["raw"]["n_above"])
        for d in (pub, priv):
            self.assertEqual(d["overlap"], tc.t3_pool_overlap(d["raw"]["n_above"], d["canon"]["n_above"], 10))
            self.assertEqual(d["raw"]["q99"], self.null["orient"]["raw"]["S2_q99"])
        self.assertTrue(pub["overlap"])
        self.assertEqual(rec["verdict"], "OVERLAP")
        self.assertIn("player_identification_test_public.csv", rec["t3q"])
        rec2, _ = tc.t3_run("player", self.files(), self.T, self.null, self.npz, "FAIL_OPENING", None,
                            log_=lambda *a: None)
        self.assertEqual(rec2["verdict"], "NOT_EVALUABLE")

    def test_run_command_end_to_end(self):
        """cmd_run in dry-run on small on-disk files, with the synthetic training set in place of
        load_train and a synthetic null / refs directory; checks outputs, summary and guards."""
        tmp = tempfile.mkdtemp(prefix="t02run_")
        orig = (tc.load_train, tc._validation_sgf_hashes)
        try:
            nd = os.path.join(tmp, "null")
            os.makedirs(nd)
            null = {"orient": {o: dict(self.null["orient"][o]) for o in tc.ORIENTS}}
            tc.write_json(os.path.join(nd, "t3_null.json"), null)
            for o in tc.ORIENTS:
                tc.save_npz(os.path.join(nd, "t3_null_%s.npz" % o), **self.npz[o])
            tc.write_json(os.path.join(nd, "train_refs.json"), T4T5T6.refs(None))
            root = os.path.join(tmp, "S")
            d = os.path.join(root, "data", "test")
            files = self.files()
            for f in files:
                if f.kind == "candidates":
                    rows = [[g["label"], g["owner"], g["sgf"], g["colour"]] for g in f.games]
                    write_csv(os.path.join(d, f.name), tc.C_COLS, rows)
                else:
                    by = {}
                    for g in f.games:
                        by.setdefault(g["owner"], []).append(g)
                    rows = []
                    for q, gs in by.items():
                        gs = gs[:20]
                        rows.append([q, str(len(gs))] + [g["sgf"] for g in gs] + [""] * (20 - len(gs)) +
                                    [g["colour"] for g in gs] + [""] * (20 - len(gs)))
                    write_csv(os.path.join(d, f.name), tc.Q_COLS, rows)
            tc._validation_sgf_hashes = lambda: {b"none"}
            tc.load_train = lambda keys=True, extra_lengths=(), grammar=False: tc.Train(
                self.rows, self.T.inner_pids, keys=keys, extra_lengths=extra_lengths)
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):          # t1 must run first
                    tc.main(["run", "--comp", "player", "--root", root, "--dry-run", "--null-dir", nd])
                self.assertEqual(tc.main(["t1", "--comp", "player", "--root", root, "--dry-run"]), 3)
                rc = tc.main(["run", "--comp", "player", "--root", root, "--dry-run", "--null-dir", nd])
            self.assertEqual(rc, 0)
            out = os.path.join(root, "results", "test02", "checks", "player")
            with open(os.path.join(out, "summary.json"), encoding="utf-8") as fh:
                s = json.load(fh)
            self.assertEqual(sorted(s["checks"]), ["T1", "T2", "T3", "T4", "T5"])
            self.assertEqual(s["checks"]["T1"]["verdict"], "FAIL_STRUCT")      # small synthetic pools
            self.assertEqual(s["checks"]["T3"]["verdict"], "OVERLAP")
            with open(os.path.join(out, "t2.json"), encoding="utf-8") as fh:
                t2 = json.load(fh)
            a_pub = [u for u in t2["units"] if u["unit"] == "a" and u["file"].endswith("candidates_public.csv")][0]
            self.assertEqual(a_pub["copies"], 1000)                            # training games -> copies
            self.assertEqual(t2["verdict"], "FAIL")
            with open(os.path.join(out, "t3.json"), encoding="utf-8") as fh:
                t3 = json.load(fh)
            self.assertTrue(t3["t2a_fired"])
            self.assertTrue(all(v["passed"] for v in t3["reference_check"].values()))
            self.assertEqual(t3["provenance"]["mode"], "dry-run")
            # the negative control: a fit-only index does not know the inner players' games
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):        # outputs exist
                    tc.main(["run", "--comp", "player", "--root", root, "--dry-run", "--null-dir", nd,
                             "--only", "T2"])
                tc.main(["run", "--comp", "player", "--root", root, "--dry-run", "--null-dir", nd, "--only",
                         "T2,T4,T5", "--index-part", "fit", "--rerun", "--reason", "fit index"])
            with open(os.path.join(out, "t2.json"), encoding="utf-8") as fh:
                t2 = json.load(fh)
            self.assertEqual(t2["index"]["part"], "fit")
            self.assertEqual(t2["provenance"]["rerun_reason"], "fit index")
            # a missing pool is skipped in dry runs (the one-pool NEG control tree)
            os.remove(os.path.join(d, "player_identification_candidates_private.csv"))
            os.remove(os.path.join(d, "player_identification_test_private.csv"))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(tc.main(["run", "--comp", "player", "--root", root, "--dry-run", "--null-dir",
                                          nd, "--only", "T3,T4", "--rerun", "--reason", "one pool"]), 0)
            with open(os.path.join(out, "t3.json"), encoding="utf-8") as fh:
                self.assertEqual(sorted(json.load(fh)["pools"]), ["public"])
        finally:
            tc.load_train, tc._validation_sgf_hashes = orig
            shutil.rmtree(tmp, ignore_errors=True)

    def test_reference_mismatch_is_an_error(self):
        null = json.loads(json.dumps(self.null))
        null["orient"]["canon"]["S2_q99"] = 0.5
        rec, ok = tc.t3_run("player", self.files(), self.T, null, self.npz, "PASS", None, log_=lambda *a: None)
        self.assertFalse(ok)
        self.assertEqual(rec["verdict"], "ERROR")
        self.assertFalse(rec["reference_check"]["canon"]["passed"])
        self.assertTrue(rec["reference_check"]["raw"]["passed"])
        null = json.loads(json.dumps(self.null))
        null["reference"] = dict(self.T.reference_record())
        rec, ok = tc.t3_run("player", self.files(), self.T, null, self.npz, "PASS", None, log_=lambda *a: None)
        self.assertTrue(ok)
        self.assertTrue(rec["reference_check"]["fit_reference"]["passed"])
        null["reference"]["sha256_sorted_player_ids"] = "0" * 64
        rec, ok = tc.t3_run("player", self.files(), self.T, null, self.npz, "PASS", None, log_=lambda *a: None)
        self.assertFalse(ok)
        self.assertEqual(rec["verdict"], "ERROR")

    def test_unseen_tokens_only_enter_the_norm(self):
        po = tc._po()
        ref = tc.t3_reference(self.T, "raw")
        i = self.T.games_of[self.T.fit[0]][0]
        seen = (self.T.colour[i], self.T.mv60(i))                # a training game: every token is known
        _t, _s, n_ext = tc.t3_profile_s2(ref, [seen], [[0]], "raw")
        self.assertEqual(n_ext, 0)
        odd = ("B", [("B", "pd"), ("W", "aa"), ("B", "ss"), ("W", "as"), ("B", "sa")])
        games = [seen, odd]
        top1, s2, n_ext = tc.t3_profile_s2(ref, games, [[0, 1]], "raw")
        self.assertGreater(n_ext, 0)
        # by hand: tf-idf over all tokens (unseen ones with idf = ln(1 + |R|) + 1), norm over all of
        # them, dot product only over the reference vocabulary
        cnt = {}
        for colour, mv in games:
            for t in po.game_tokens(colour, mv, "raw"):
                cnt[t] = cnt.get(t, 0) + 1
        V, n_fit = ref["V"], len(ref["fit"])
        w = {t: (1 + math.log(k)) * (ref["idf"][ref["vocab"][t]] if t in ref["vocab"]
                                     else math.log(1 + n_fit) + 1) for t, k in cnt.items()}
        norm = math.sqrt(sum(v * v for v in w.values())) + 1e-12
        q = np.zeros(V)
        for t, v in w.items():
            if t in ref["vocab"]:
                q[ref["vocab"][t]] = v / norm
        cos = ref["R"] @ q
        self.assertAlmostEqual(float(top1[0]), float(np.max(cos)), places=12)
        srt = np.sort(cos)[::-1]
        self.assertAlmostEqual(float(s2[0]), float(srt[0] - srt[1]), places=12)


class T3bBlackOnly(unittest.TestCase):
    """T3b (PREREG_02 rev 8) on a small synthetic training set: a candidate pool whose games are all black is
    compared with the fit players' BLACK games; mixed pools keep the registered T3."""

    @classmethod
    def setUpClass(cls):
        rows, inner = synthetic_train(n_fit=24, n_inner=4, games=230, seed=17)
        cls.rows = rows
        cls.T = tc.Train(rows, inner, keys=False)
        res, arrays, resb, arrb = {}, {}, {}, {}
        for o in tc.ORIENTS:
            ref = tc.t3_reference(cls.T, o)
            res[o], arrays[o] = tc.t3_orient_precompute(cls.T, o, power_sims=20, t3q_ks=range(5, 7), ref=ref)
            resb[o], arrb[o] = tc.t3b_orient_precompute(cls.T, o, power_sims=20, reg=ref)
        cls.null, cls.npz = {"orient": res}, arrays
        cls.null_b = {"orient": resb, "reference": cls.T.reference_record_black()}
        cls.npz_b = arrb
        cls.new_rows, _ = synthetic_train(n_fit=10, n_inner=0, games=220, seed=99)

    def test_null_structure(self):
        blk = self.T.black_order()
        for p in self.T.fit:
            self.assertTrue(all(self.T.colour[i] == "B" for i in blk[p]))
            self.assertEqual(blk[p], [i for i in self.T.t3_order()[p] if self.T.colour[i] == "B"])
        for o in tc.ORIENTS:
            r, z = self.null_b["orient"][o], self.npz_b[o]
            self.assertEqual((r["n_null"], r["n_holdout_inner"]), (24, 4))
            self.assertEqual(r["S2_q99"], tc.q99(z["null_S2"]))
            self.assertEqual(r["vocab"], self.null["orient"][o]["vocab"])            # the T3 vocabulary
            self.assertEqual(r["vocab_sha256"], self.null["orient"][o]["vocab_sha256"])
            self.assertNotEqual(r["idf_sha256"], self.null["orient"][o]["idf_sha256"])  # df over black games only
            self.assertAlmostEqual(float(z["idf"].max()), math.log(25.0) + 1.0)
            self.assertEqual(r["idf_sha256"], tc.sha256_array(z["idf"].astype("<f8")))
            self.assertEqual(sorted(r["pos"]), ["100", "20", "50", "all"])
            self.assertIn("registered_T3_on_black_only_profiles", r)
            self.assertEqual(set(r["power_pool_rule"]), {"0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0"})
        fp = tc.t3b_false_positive(self.null_b["orient"], self.npz_b)
        self.assertEqual(fp["null_fit_leave_own_out"]["n"], 24)
        self.assertEqual(fp["holdout_inner"]["n"], 4)
        # the reference by hand: R_b rows are the fit players' black games only
        ref = tc.t3b_reference(self.T, "raw")
        po = tc._po()
        want = (po.indicator([blk[p] for p in self.T.fit], self.T.n) @ ref["Gm"]).tocsc()
        df = np.round(25.0 / np.exp(ref["idf"] - 1.0) - 1.0).astype(int)      # idf = ln(25 / (1 + df)) + 1
        self.assertEqual(df.tolist(), np.diff(want.indptr).tolist())
        self.assertEqual(ref["R"].shape[0], 24)

    def files(self, pub_colour="B", priv_black=True):
        fit = [p for p in self.T.fit][:10]
        blk = self.T.black_order()
        order = self.T.t3_order()
        c_pub = []
        for k, p in enumerate(fit):                       # training players' black games: the positive pool
            src = blk[p][:100] if pub_colour == "B" else order[p][:100]
            for i in src:
                c_pub.append(("pub_p_%d" % k, "pub_g_%d" % i, self.T.colour[i], self.rows[i]["sgf_content"]))
        new_b = [r for r in self.new_rows if r["color"] == "B" or not priv_black]
        by = {}
        for r in new_b:
            by.setdefault(r["player_id"], []).append(r)
        c_priv = [("priv_p_%s" % p[-4:], "priv_g_%s" % r["game_id"], r["color"], r["sgf_content"])
                  for p, rs in by.items() for r in rs[:100]]
        q_pub = [("pub_q_%d" % k, "x", "B", self.rows[i]["sgf_content"]) for k, p in enumerate(fit)
                 for i in blk[p][100:105]]
        q_priv = [("priv_q_%d" % k, "y", "B", r["sgf_content"]) for k, r in enumerate(new_b[:50])]
        return [FakeFile("public", "questions", "player_identification_test_public.csv", q_pub),
                FakeFile("public", "candidates", "player_identification_candidates_public.csv", c_pub),
                FakeFile("private", "questions", "player_identification_test_private.csv", q_priv),
                FakeFile("private", "candidates", "player_identification_candidates_private.csv", c_priv)]

    def run_t3(self, files, null_b="default", t1="PASS"):
        nb = self.null_b if null_b == "default" else null_b
        return tc.t3_run("player", files, self.T, self.null, self.npz, t1, None, log_=lambda *a: None,
                         null_b=nb, npz_b=self.npz_b if nb is not None else None)

    def test_black_only_pools_use_t3b(self):
        rec, ok = self.run_t3(self.files())
        self.assertTrue(ok)
        self.assertTrue(all(v["passed"] for v in rec["reference_check"].values()), rec["reference_check"])
        self.assertIn("t3b_raw", rec["reference_check"])
        self.assertEqual(rec["variants"], {"public": "T3b", "private": "T3b"})
        pub, priv = rec["pools"]["public"], rec["pools"]["private"]
        for d in (pub, priv):
            self.assertEqual(d["b_c"], 1.0)
            self.assertEqual(d["raw"]["q99"], self.null_b["orient"]["raw"]["S2_q99"])
            self.assertEqual(d["canon"]["q99"], self.null_b["orient"]["canon"]["S2_q99"])
            self.assertEqual(d["registered_T3_record_only"]["raw"]["q99"], self.null["orient"]["raw"]["S2_q99"])
            self.assertEqual(d["overlap"], tc.t3_pool_overlap(d["raw"]["n_above"], d["canon"]["n_above"], 10))
        self.assertGreaterEqual(pub["raw"]["n_above"], 8)
        self.assertLess(priv["raw"]["n_above"], pub["raw"]["n_above"])
        self.assertTrue(pub["overlap"])
        self.assertEqual(rec["verdict"], "OVERLAP")
        rec2, _ = self.run_t3(self.files(), t1="FAIL_OPENING")
        self.assertEqual(rec2["verdict"], "NOT_EVALUABLE")

    def test_mixed_pools_keep_the_registered_t3(self):
        rec, ok = self.run_t3(self.files(pub_colour="mixed", priv_black=False), null_b=None)
        self.assertTrue(ok)                               # no black-only pool: the T3b null is not needed
        self.assertEqual(rec["variants"], {"public": "T3", "private": "T3"})
        self.assertNotIn("t3b_raw", rec["reference_check"])
        pub = rec["pools"]["public"]
        self.assertLess(pub["b_c"], 1.0)
        self.assertEqual(pub["raw"]["q99"], self.null["orient"]["raw"]["S2_q99"])
        self.assertNotIn("registered_T3_record_only", pub)
        rec, ok = self.run_t3(self.files(pub_colour="mixed", priv_black=True))
        self.assertEqual(rec["variants"], {"public": "T3", "private": "T3b"})

    def near_single_colour_files(self, n_minor, major="B"):
        """Private pool only: the 10 new (non-training) players, 100 - n_minor games of colour `major` and n_minor of
        the other colour each (b_c = 1 - n_minor / 100 for major B)."""
        by = {}
        for r in self.new_rows:
            by.setdefault(r["player_id"], []).append(r)
        c = []
        for p, rs in by.items():
            games = [r for r in rs if r["color"] == major][:100 - n_minor] + \
                [r for r in rs if r["color"] != major][:n_minor]
            c += [("priv_p_%s" % p[-4:], "priv_g_%s" % r["game_id"], r["color"], r["sgf_content"]) for r in games]
        q = [("priv_q_%d" % k, "y", r["color"], r["sgf_content"]) for k, r in enumerate(self.new_rows[:50])]
        return [FakeFile("private", "questions", "player_identification_test_private.csv", q),
                FakeFile("private", "candidates", "player_identification_candidates_private.csv", c)]

    def test_near_single_colour_pool_left_on_t3_is_low_power(self):
        """PREREG_02 rev 8 (review 2026-09-28): a pool on the registered T3 with max(b_c, 1 - b_c) >= 0.9 has little
        power; without a declared overlap its verdict is LOW_POWER ('not determinable'), never PASS."""
        lp = tc.t3_low_power
        self.assertEqual(tc.T3_LOW_POWER_SHARE, 0.9)
        self.assertTrue(lp(0.9, "T3") and lp(360 / 400, "T3") and lp(1.0 - 0.1, "T3"))
        self.assertTrue(lp(0.1, "T3") and lp(0.0, "T3") and lp(0.999, "T3"))     # all white: T3b does not apply
        self.assertFalse(lp(0.89, "T3") or lp(0.11, "T3") or lp(0.5, "T3"))
        self.assertFalse(lp(1.0, "T3b") or lp(None, "T3"))
        self.assertIn(("T3", "LOW_POWER"), tc.CONSEQUENCE)
        # 95 black + 5 white per candidate: T3 (not T3b), low power, no overlap declared -> LOW_POWER
        rec, ok = self.run_t3(self.near_single_colour_files(5), null_b=None)
        self.assertTrue(ok)
        d = rec["pools"]["private"]
        self.assertEqual((d["variant"], d["b_c"]), ("T3", 0.95))
        self.assertNotIn("registered_T3_record_only", d)
        self.assertTrue(d["low_power"])
        self.assertFalse(d["overlap"])
        self.assertEqual(d["conclusion"], "not determinable (low power)")
        self.assertEqual(rec["verdict"], "LOW_POWER")
        self.assertEqual(rec["consequence"], tc.CONSEQUENCE[("T3", "LOW_POWER")])
        self.assertEqual(rec["low_power_pools"], ["private"])
        self.assertIn("low_power_rule", rec)
        rec2, _ = self.run_t3(self.near_single_colour_files(5), null_b=None, t1="FAIL_OPENING")
        self.assertEqual(rec2["verdict"], "NOT_EVALUABLE")
        # 85 black: not flagged (this 10-candidate pool needs 1 above q99 and declares overlap)
        rec, _ = self.run_t3(self.near_single_colour_files(15), null_b=None)
        d = rec["pools"]["private"]
        self.assertAlmostEqual(d["b_c"], 0.85, places=12)
        self.assertFalse(d["low_power"])
        self.assertEqual(rec["low_power_pools"], [])
        self.assertEqual((d["overlap"], d["conclusion"], rec["verdict"]), (True, "overlap", "OVERLAP"))
        # all white: T3 (T3b is black only), low power; a declared overlap stays OVERLAP
        rec, _ = self.run_t3(self.near_single_colour_files(0, major="W"), null_b=None)
        d = rec["pools"]["private"]
        self.assertEqual((d["variant"], d["b_c"], d["low_power"]), ("T3", 0.0, True))
        self.assertEqual((d["overlap"], d["conclusion"], rec["verdict"]), (True, "overlap", "OVERLAP"))
        # a black-only pool on T3b is never low_power
        rec, _ = self.run_t3(self.files())
        self.assertEqual([p["low_power"] for p in rec["pools"].values()], [False, False])
        self.assertEqual(rec["low_power_pools"], [])

    def test_missing_or_changed_t3b_null_is_an_error(self):
        rec, ok = self.run_t3(self.files(), null_b=None)
        self.assertFalse(ok)
        self.assertEqual(rec["verdict"], "ERROR")
        self.assertIn("missing", rec["reference_check"]["t3b"]["problem"])
        self.assertIsNone(rec["pools"]["public"]["overlap"])
        nb = json.loads(json.dumps(self.null_b))
        nb["orient"]["canon"]["S2_q99"] = 0.5
        rec, ok = self.run_t3(self.files(), null_b=nb)
        self.assertFalse(ok)
        self.assertEqual(rec["verdict"], "ERROR")
        self.assertFalse(rec["reference_check"]["t3b_canon"]["passed"])
        self.assertTrue(rec["reference_check"]["t3b_raw"]["passed"])
        nb = json.loads(json.dumps(self.null_b))
        nb["reference"]["black_games"] += 1
        rec, ok = self.run_t3(self.files(), null_b=nb)
        self.assertFalse(rec["reference_check"]["t3b_fit_reference"]["passed"])
        self.assertEqual(rec["verdict"], "ERROR")

    def test_check_pins_t3b(self):
        t3b = {"reference": dict(tc.T3B_PINS["reference"]), "orient": {o: dict(tc.T3B_PINS[o]) for o in tc.ORIENTS}}
        unset = sum(v is None for d in [tc.T3B_PINS["reference"]] + [tc.T3B_PINS[o] for o in tc.ORIENTS]
                    for v in d.values())
        self.assertEqual(len(tc.check_pins_t3b(t3b)), unset)
        if not unset:
            t3b["orient"]["raw"]["S2_q99"] += 1e-9
            t3b["reference"]["black_games"] -= 1
            self.assertEqual(len(tc.check_pins_t3b(t3b)), 2)

    def test_precompute_part_t3b_leaves_the_base_files(self):
        tmp = tempfile.mkdtemp(prefix="t02pre_")
        orig = tc.load_train
        try:
            out = os.path.join(tmp, "null")
            os.makedirs(out)
            for f in tc.NULL_FILES:                       # stand-ins for the committed rev 5 files
                with open(os.path.join(out, f), "wb") as fh:
                    fh.write(b"rev 5 " + f.encode())
            before = {f: open(os.path.join(out, f), "rb").read() for f in tc.NULL_FILES}
            tc.load_train = lambda keys=True, extra_lengths=(), grammar=False: tc.Train(
                self.rows, self.T.inner_pids, keys=keys, extra_lengths=extra_lengths)
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):       # synthetic data never matches the pins
                    tc.main(["precompute", "--dry-run", "--part", "t3b", "--out", out])
                self.assertFalse(any(os.path.exists(os.path.join(out, f)) for f in tc.T3B_NULL_FILES))
                with self.assertRaises(SystemExit):       # the index cache needs the T2 keys
                    tc.main(["precompute", "--dry-run", "--part", "t3b", "--out", out, "--allow-pin-mismatch",
                             "--t2-index-cache", os.path.join(tmp, "idx.npz")])
                rc = tc.main(["precompute", "--dry-run", "--part", "t3b", "--out", out, "--allow-pin-mismatch"])
            self.assertEqual(rc, 1)
            self.assertEqual({f: open(os.path.join(out, f), "rb").read() for f in tc.NULL_FILES}, before)
            with open(os.path.join(out, "t3b_null.json"), encoding="utf-8") as fh:
                t3b = json.load(fh)
            self.assertEqual(t3b["format"], "c10-test02 t3b null v1")
            self.assertEqual(t3b["provenance"]["part"], "t3b")
            self.assertFalse(t3b["pins"]["all_match"])
            self.assertEqual(t3b["reference"], self.T.reference_record_black())
            for o in tc.ORIENTS:
                z = tc.load_npz(os.path.join(out, "t3b_null_%s.npz" % o))
                self.assertEqual(tc.sha256_file(os.path.join(out, "t3b_null_%s.npz" % o)),
                                 t3b["npz_sha256"]["t3b_null_%s.npz" % o])
                self.assertEqual(t3b["orient"][o]["S2_q99"], self.null_b["orient"][o]["S2_q99"])
                self.assertTrue(np.array_equal(z["null_S2"], self.npz_b[o]["null_S2"]))
            self.assertIn("P_declare_raw_or_canon", t3b["either"])
            self.assertIn("holdout_inner", t3b["false_positive"])
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):       # outputs exist
                    tc.main(["precompute", "--dry-run", "--part", "t3b", "--out", out, "--allow-pin-mismatch"])
        finally:
            tc.load_train = orig
            shutil.rmtree(tmp, ignore_errors=True)


class T4T5T6(TmpDir):
    def refs(self):
        return {"t4_null_ranges": {"black_share": {"cand": [0.49, 0.521], "ques": [0.472, 0.517]},
                                   "upper_right_share": {"cand": [0.791, 0.839], "ques": [0.749, 0.806]},
                                   "first_move_point_tvd_max": {"cand": 0.048, "ques": 0.073},
                                   "share_in_S99_min": {"cand": 0.983, "ques": 0.985}},
                "first_move_point_dist_train": {"pd": 0.6, "qd": 0.4},
                "first_move_class_dist_train": {"dd": 0.6, "cd": 0.4},
                "black_share": {"fit": 0.5006}, "upper_right_share": {"fit": 0.8166},
                "t5": {"uniform_5_20": {"mean": 12.5, "share_below_10": 0.3125}}}

    def files(self, q_colour="B", c_colour="W"):
        g = sgf_of([("B", "pd"), ("W", "dp")])
        q = [("pub_q_%d" % i, "x", q_colour, g) for i in range(10)]
        c = [("pub_p_%d" % i, "pub_g_%d" % i, c_colour, g) for i in range(10)]
        q2 = [("priv_q_%d" % i, "x", "B" if i % 2 else "W", g) for i in range(10)]
        c2 = [("priv_p_%d" % i, "priv_g_%d" % i, "B" if i % 2 else "W", g) for i in range(10)]
        return [FakeFile("public", "questions", "player_identification_test_public.csv", q, num_games=[5] * 4 + [20] * 6),
                FakeFile("public", "candidates", "player_identification_candidates_public.csv", c),
                FakeFile("private", "questions", "player_identification_test_private.csv", q2, num_games=list(range(5, 15))),
                FakeFile("private", "candidates", "player_identification_candidates_private.csv", c2)]

    def test_t4_colour_guard(self):
        r = tc.t4_run("player", self.files(), self.refs())
        self.assertEqual(r["verdict"], "GUARD_LEVEL3")
        self.assertEqual(r["pools"]["public"]["abs_diff"], 1.0)
        self.assertTrue(r["pools"]["public"]["guard_level3_colour"])
        self.assertFalse(r["pools"]["private"]["guard_level3_colour"])
        f = r["files"]["player_identification_test_public.csv"]
        self.assertEqual(f["upper_right_share"], 1.0)
        self.assertFalse(f["rotation_flag"])
        self.assertAlmostEqual(f["first_move_point_tvd_vs_train"], 0.4)
        r = tc.t4_run("player", self.files(q_colour="B", c_colour="B"), self.refs())
        self.assertEqual(r["verdict"], "PASS")

    def test_t4_black_only_files_mark_the_mixed_null_ranges_not_applicable(self):
        """PREREG_02 rev 8: the null ranges come from mixed-colour pools; a black-only file (official competition 1)
        has them marked 'not applicable' (values kept under mixed_colour_null_ranges_record_only)."""
        r = tc.t4_run("player", self.files(q_colour="B", c_colour="B"), self.refs())
        for name in ("player_identification_test_public.csv", "player_identification_candidates_public.csv"):
            f = r["files"][name]
            self.assertEqual(f["black_share"], 1.0)
            self.assertFalse(f["null_ranges_applicable"])
            for k in ("black_share_null_range", "black_share_inside_null_range", "first_move_point_tvd_null_max",
                      "upper_right_null_range"):
                self.assertEqual(f[k], tc.NOT_APPLICABLE, (name, k))
            kind = "ques" if "test" in name else "cand"
            self.assertEqual(f["mixed_colour_null_ranges_record_only"]["black_share_null_range"],
                             self.refs()["t4_null_ranges"]["black_share"][kind])
            self.assertFalse(f["mixed_colour_null_ranges_record_only"]["black_share_inside_null_range"])
            self.assertEqual(f["upper_right_share"], 1.0)                # the record-only values stay
        self.assertTrue(r["pools"]["public"]["black_only_pool"])
        self.assertFalse(r["pools"]["private"]["black_only_pool"])
        f = r["files"]["player_identification_test_private.csv"]            # mixed: unchanged
        self.assertTrue(f["null_ranges_applicable"])
        self.assertIsInstance(f["black_share_inside_null_range"], bool)
        self.assertEqual(r["verdict"], "PASS")

    def test_t5_official_design(self):
        g = sgf_of([("B", "pd"), ("W", "dp")])
        q = [("pub_q_%d" % i, "x", "B", g) for i in range(400)]
        ng = [5] * 100 + [10] * 100 + [15] * 100 + [20] * 100
        files = [FakeFile("public", "questions", "rank_prediction_test_public.csv", q, num_games=ng),
                 FakeFile("private", "questions", "rank_prediction_test_private.csv", q, num_games=ng[:-1] + [19])]
        r = tc.t5_run(files, {"t5": {}})
        pub, priv = r["files"]["rank_prediction_test_public.csv"], r["files"]["rank_prediction_test_private.csv"]
        self.assertTrue(pub["official_design_match"])
        self.assertFalse(priv["official_design_match"])
        self.assertEqual(pub["official_design"], {"5": 100, "10": 100, "15": 100, "20": 100})
        self.assertAlmostEqual(pub["chi2_vs_uniform"], 1200.0)           # recorded; not a test of anything there
        self.assertIn("not a test", pub["chi2_note"])
        self.assertEqual(r["references"]["official_design"], pub["official_design"])
        self.assertEqual(r["verdict"], "RECORD_ONLY")

    def test_t5(self):
        r = tc.t5_run(self.files(), {"t5": {}})
        f = r["files"]["player_identification_test_public.csv"]
        self.assertEqual(f["questions"], 10)
        self.assertEqual(f["histogram"]["5"], 4)
        self.assertEqual(f["histogram"]["20"], 6)
        self.assertAlmostEqual(f["mean"], (4 * 5 + 6 * 20) / 10)
        self.assertAlmostEqual(f["share_below_10"], 0.4)
        self.assertEqual(f["df"], 15)
        self.assertEqual(r["verdict"], "RECORD_ONLY")
        exp = 10 / 16
        want = sum(((o - exp) ** 2) / exp for o in [4] + [0] * 14 + [6])
        self.assertAlmostEqual(f["chi2_vs_uniform"], want)

    def test_t6(self):
        root = self.tmp
        d = build_tree(root, "player")
        pdir = os.path.join(root, "results", "test02", "player")
        rows = []
        for pool in tc.POOLS:
            qt = tc.Table(os.path.join(d, tc.FILES_BY[("player", pool, "questions")]))
            ct = tc.Table(os.path.join(d, tc.FILES_BY[("player", pool, "candidates")]))
            pids = sorted(set(ct.column("player_id")))
            for k, q in enumerate(qt.column("question_id")):
                rows.append([q] + [pids[(k // 2 + j) % 400] for j in range(5)])     # 200 distinct top-1
            os.makedirs(os.path.join(pdir, pool), exist_ok=True)
            tc.write_json(os.path.join(pdir, pool, "apply_report.json"),
                          {"sr_top1_distinct_per_round": [350 + i for i in range(10)]})
        write_csv(os.path.join(pdir, "submission_player.csv"), ["question_id"] + ["top_%d" % i for i in range(1, 6)], rows)
        orig = tc._validation_sgf_hashes
        with redirect_stdout(io.StringIO()):
            rc = tc.main(["t6", "--comp", "player", "--root", root, "--dry-run",
                          "--null-dir", os.path.join(root, "nonexistent")])
        tc._validation_sgf_hashes = orig
        self.assertEqual(rc, 0)
        with open(os.path.join(root, "results", "test02", "checks", "player", "t6.json"), encoding="utf-8") as f:
            r = json.load(f)
        for pool in tc.POOLS:
            p = r["pools"][pool]
            self.assertEqual(p["questions"], 400)
            self.assertEqual(p["candidates"], 400)
            self.assertTrue(p["questions_equal_candidates_equal_400"])
            self.assertEqual(p["distinct_top1_submission"], 200)
            self.assertEqual(p["top1_outside_pool"], 0)
            self.assertEqual(p["sr_top1_distinct_per_round"], [350 + i for i in range(10)])
        self.assertEqual(r["verdict"], "RECORD_ONLY")
        with self.assertRaises(SystemExit):
            tc.main(["t6", "--comp", "rank", "--root", root, "--dry-run"])


# ---------------------------------------------------------------- real mode (review findings 2026-09-26, fixer)
class RealModeT1(TmpDir):
    """Real mode on a temporary 'repository' layout (the module paths point into the temp dir; git is mocked):
    the raw t1.json is always the full T1, and content checks record local validation games and refuse a
    validation-derived tree."""

    def setUp(self):
        super().setUp()
        self.raw = os.path.join(self.tmp, "data", "test", "raw")
        self.built = os.path.join(self.tmp, "data", "test")
        self.checks = os.path.join(self.tmp, "results", "test02", "checks")
        self.saved = {k: getattr(tc, k) for k in ("DATA_TEST", "DATA_TEST_RAW", "CHECKS_DIR", "CHECKS_BUILT",
                                                  "REAL_T1", "not_committed_clean", "_validation_sgf_hashes")}
        tc.DATA_TEST, tc.DATA_TEST_RAW = self.built, self.raw
        tc.CHECKS_DIR, tc.CHECKS_BUILT = self.checks, os.path.join(self.checks, "built")
        tc.REAL_T1 = ((self.raw, self.checks), (self.built, tc.CHECKS_BUILT))
        tc.not_committed_clean = lambda paths: []
        tc._validation_sgf_hashes = lambda: {b"not-a-real-hash-of-any-game"}
        d = build_tree(os.path.join(self.tmp, "src"), "rank")
        for _pool, _kind, name in tc.FILES["rank"]:
            os.makedirs(self.raw, exist_ok=True)
            os.makedirs(self.built, exist_ok=True)
            shutil.copyfile(os.path.join(d, name), os.path.join(self.raw, name))
            shutil.copyfile(os.path.join(d, name), os.path.join(self.built, name))

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(tc, k, v)
        super().tearDown()

    def main(self, *argv):
        with redirect_stdout(io.StringIO()) as buf:
            rc = tc.main(list(argv))
        return rc, buf.getvalue()

    def test_raw_t1_is_always_complete(self):
        with self.assertRaises(SystemExit) as cm:                        # the rehearsal-style command
            self.main("t1", "--comp", "rank", "--only", "T1.1,T1.2,T1.3")
        self.assertIn("--only is refused", str(cm.exception.code))
        self.assertFalse(os.path.exists(os.path.join(self.checks, "rank", "t1.json")))
        rc, _ = self.main("t1", "--comp", "rank")
        self.assertEqual(rc, 0)
        with open(os.path.join(self.checks, "rank", "t1.json"), encoding="utf-8") as f:
            t1 = json.load(f)
        self.assertEqual((t1["complete"], t1["stage"], t1["provenance"]["mode"]), (True, "raw", "real"))
        guard = t1["provenance"]["validation_hash_guard"]
        self.assertEqual((guard["available"], guard["hits"]), (True, 0))
        # the built copies (t02_test_bundle.py build) keep the structure-only re-check
        rc, _ = self.main("t1", "--comp", "rank", "--tests-dir", self.built, "--out", tc.CHECKS_BUILT, "--only",
                          "T1.1,T1.2,T1.3")
        self.assertEqual(rc, 0)
        with open(os.path.join(tc.CHECKS_BUILT, "rank", "t1.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["stage"], "built")

    def test_validation_derived_tree_refused_in_real_mode(self):
        tc._validation_sgf_hashes = lambda: {hashlib.sha1(s.encode("utf-8")).digest() for s in BASE}
        with self.assertRaises(SystemExit) as cm:
            self.main("t1", "--comp", "rank")
        self.assertIn("validation-derived tree", str(cm.exception.code))
        self.assertFalse(os.path.exists(os.path.join(self.checks, "rank", "t1.json")))
        # a few validation games in an official file are recorded, not refused
        tc._validation_sgf_hashes = lambda: {hashlib.sha1(BASE[3].encode("utf-8")).digest()}
        rc, out = self.main("t1", "--comp", "rank")
        self.assertEqual(rc, 0)
        self.assertIn("local validation games", out)
        with open(os.path.join(self.checks, "rank", "t1.json"), encoding="utf-8") as f:
            guard = json.load(f)["provenance"]["validation_hash_guard"]
        self.assertGreater(guard["hits"], 0)
        self.assertLess(max(v["share"] for v in guard["files"].values()), tc.REAL_VALIDATION_REFUSE_SHARE)
        tc._validation_sgf_hashes = lambda: set()                      # no data/local: recorded as unavailable
        rc, _ = self.main("t1", "--comp", "rank", "--rerun", "--reason", "x")
        self.assertEqual(rc, 0)
        with open(os.path.join(self.checks, "rank", "t1.json"), encoding="utf-8") as f:
            self.assertFalse(json.load(f)["provenance"]["validation_hash_guard"]["available"])

    def test_guard_shares(self):
        tc._validation_sgf_hashes = lambda: {hashlib.sha1(s.encode("utf-8")).digest() for s in ("a", "b")}
        g = tc.real_validation_guard({"x.csv": ["a", "c", "d", "e"], "y.csv": []})
        self.assertEqual((g["hits"], g["files"]["x.csv"]["share"], g["files"]["y.csv"]["games"]), (1, 0.25, 0))
        with self.assertRaises(SystemExit):
            tc.real_validation_guard({"x.csv": ["a", "b", "c", "d"]})           # half of the games


# ---------------------------------------------------------------- source hygiene and determinism
class Hygiene(unittest.TestCase):
    def src(self, path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_no_label_file_word(self):
        for p in (os.path.join(TOOLS, "t02_checks.py"), os.path.abspath(__file__)):
            self.assertNotIn(NEEDLE, self.src(p).lower(), p)

    def test_imports(self):
        mods = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)",
                              self.src(os.path.join(TOOLS, "t02_checks.py")), re.M))
        self.assertFalse(mods & {"p02_rank_fuse", "p02_rank_opening", "p02_rank_pseudo", "decide", "t02_rank",
                                 "t02_player", "p02_player_fuse", "score"}, mods)
        self.assertTrue({"c10data", "p02_moves", "p02_player_open"} <= mods)

    def test_npz_bytes_deterministic(self):
        tmp = tempfile.mkdtemp(prefix="t02npz_")
        try:
            a = {"x": np.arange(5.0), "s": np.array(["a", "bc"]), "i": np.array([1, 2], np.int32)}
            p1, p2 = os.path.join(tmp, "a.npz"), os.path.join(tmp, "b.npz")
            tc.save_npz(p1, **a)
            tc.save_npz(p2, **a)
            self.assertEqual(open(p1, "rb").read(), open(p2, "rb").read())
            z = tc.load_npz(p1)
            for k, v in a.items():
                self.assertTrue(np.array_equal(z[k], v))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(os.path.isfile(os.path.join(REPO, "results", "test02", "null", "t3_null.json")),
                     "committed null not present yet")
class CommittedNull(unittest.TestCase):
    """The committed null reproduces the registered pins (PREREG_02 9.8; spec 4.1)."""

    def test_pins(self):
        nd = os.path.join(REPO, "results", "test02", "null")
        with open(os.path.join(nd, "t3_null.json"), encoding="utf-8") as f:
            t3 = json.load(f)
        self.assertEqual(tc.check_pins(t3), [])
        self.assertTrue(t3["pins"]["all_match"])
        self.assertEqual(t3["orient"]["raw"]["S2_q99"], 0.1004267242871777)
        self.assertEqual(t3["orient"]["canon"]["S2_q99"], 0.08085695648061325)
        for o in tc.ORIENTS:
            z = tc.load_npz(os.path.join(nd, "t3_null_%s.npz" % o))
            thr = t3["orient"][o]["S2_q99"]
            self.assertEqual(tc.q99(z["null_S2"]), thr)
            self.assertEqual(int((z["null_S2"] > thr).sum()), tc.PINS[o]["null_n_exceed"])
            self.assertEqual(int((z["holdout_S2"] > thr).sum()), tc.PINS[o]["holdout_inner_n_exceed"])
            self.assertEqual(len(z["player_id"]), 1985)
            self.assertEqual(len(z["idf"]), tc.PINS[o]["vocab"])
            self.assertEqual(tc.sha256_array(z["idf"].astype("<f8")), t3["orient"][o]["idf_sha256"])
            self.assertEqual(tc.sha256_file(os.path.join(nd, "t3_null_%s.npz" % o)),
                             t3["npz_sha256"]["t3_null_%s.npz" % o])
        with open(os.path.join(nd, "train_refs.json"), encoding="utf-8") as f:
            refs = json.load(f)
        self.assertEqual(refs["S99"], list(tc.S99))
        self.assertEqual(round(refs["black_share"]["fit"], 4), 0.5006)
        self.assertEqual(round(refs["upper_right_share"]["fit"], 4), 0.8166)
        mc = refs["move_count"]["fit"]
        self.assertEqual((mc["min"], mc["p1"], mc["median"], mc["max"]), (100, 107.0, 228.0, 445))
        self.assertEqual(refs["sgf_training"]["t1_4_violations"], 0)

    def test_rev5_files_byte_identical(self):
        """PREREG_02 rev 8 adds the T3b null next to these files; the rev 5 null (commit c3e8ea1) is never rewritten
        (precompute --part t3b leaves it untouched)."""
        want = {"t3_null.json": "1950e9a8af5543dc95f6dd9d28664278ba145fafef153b785905264a6e934dd4",
                "t3_null_raw.npz": "4eb9cf053b3ae296df6de020e7f1ecf4caa3e3bc53ed477e0e749a4f4d4da714",
                "t3_null_canon.npz": "5312a18c34e71d4a254b5239b508612265be0b38a543b2f65dbd458159fcf69e",
                "train_refs.json": "380a0a7fa6e4cb7751e8ee3404a1ccba645ca73dd919a6a5e9a8451636988203"}
        nd = os.path.join(REPO, "results", "test02", "null")
        self.assertEqual({f: tc.sha256_file(os.path.join(nd, f)) for f in want}, want)


@unittest.skipUnless(os.path.isfile(os.path.join(REPO, "results", "test02", "null", "t3b_null.json")),
                     "T3b null not computed yet")
class CommittedT3bNull(unittest.TestCase):
    """The T3b null (PREREG_02 rev 8) reproduces its pins (T3B_PINS)."""

    def test_pins(self):
        nd = os.path.join(REPO, "results", "test02", "null")
        with open(os.path.join(nd, "t3b_null.json"), encoding="utf-8") as f:
            t3b = json.load(f)
        self.assertEqual(tc.check_pins_t3b(t3b), [])
        self.assertTrue(t3b["pins"]["all_match"])
        self.assertTrue(t3b["registered_reference_check"]["all_match"])
        self.assertEqual(t3b["provenance"]["mode"], "real")
        for o in tc.ORIENTS:
            z = tc.load_npz(os.path.join(nd, "t3b_null_%s.npz" % o))
            thr = t3b["orient"][o]["S2_q99"]
            self.assertEqual(thr, tc.T3B_PINS[o]["S2_q99"])
            self.assertEqual(tc.q99(z["null_S2"]), thr)
            self.assertEqual(int((z["null_S2"] > thr).sum()), tc.T3B_PINS[o]["null_n_exceed"])
            self.assertEqual(int((z["holdout_S2"] > thr).sum()), tc.T3B_PINS[o]["holdout_inner_n_exceed"])
            self.assertEqual(len(z["player_id"]), tc.T3B_PINS[o]["n_null"])
            self.assertEqual(len(z["holdout_player_id"]), tc.T3B_PINS[o]["n_holdout_inner"])
            self.assertEqual(len(z["idf"]), tc.T3B_PINS[o]["vocab"])
            self.assertEqual(tc.sha256_array(z["idf"].astype("<f8")), t3b["orient"][o]["idf_sha256"])
            self.assertEqual(tc.sha256_file(os.path.join(nd, "t3b_null_%s.npz" % o)),
                             t3b["npz_sha256"]["t3b_null_%s.npz" % o])
        with open(os.path.join(nd, "t3_null.json"), encoding="utf-8") as f:
            t3 = json.load(f)
        for o in tc.ORIENTS:                      # same vocabulary as the registered T3, a different idf
            self.assertEqual(t3b["orient"][o]["vocab_sha256"], t3["orient"][o]["vocab_sha256"])
            self.assertNotEqual(t3b["orient"][o]["idf_sha256"], t3["orient"][o]["idf_sha256"])


if __name__ == "__main__":
    unittest.main()
