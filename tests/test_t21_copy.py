"""Unit tests of tools/t21_copy.py (PREREG_21 sections 2-4, 7, 9) on small synthetic pools. No real data, no
data/local, no training data; temporary trees only. The end-to-end cases build official-shaped pools (400 candidates x
100 games, 400 questions per pool) from tiny filler games so that check_submission --official runs for real.
Run: python -m pytest tests/test_t21_copy.py"""

import csv
import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t02_checks as t2  # noqa: E402
import t21_copy as tc  # noqa: E402

LETTERS = "abcdefghijklmnopqrs"


def rand_game(seed, n=120):
    r = random.Random(seed)
    pts = r.sample([a + b for a in LETTERS for b in LETTERS], n)
    return "(;SZ[19]" + "".join(f";{'BW'[i % 2]}[{p}]" for i, p in enumerate(pts)) + ")"


def write_candidates(path, prefix, rows):
    """rows: [(player_id, sgf, colour)] -> an official-shaped candidate file (game ids <prefix>g<n>)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(t2.C_COLS)
        for i, (pid, sgf, c) in enumerate(rows):
            w.writerow([f"{prefix}g{i + 1:06d}", pid, sgf, c])


def write_questions(path, prefix, questions):
    """questions: [[sgf, ...]] (all black) -> an official-shaped player question file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(t2.Q_COLS)
        for j, games in enumerate(questions):
            w.writerow([f"{prefix}{j + 1:04d}", len(games)] + games + [""] * (20 - len(games))
                       + ["B"] * len(games) + [""] * (20 - len(games)))


G = {k: rand_game(k) for k in range(1, 60)}
SHORT = rand_game(99, n=99)
FILL = "(;SZ[19];B[aa])"          # a 1-move filler game (never compared: under 100 moves)


class Pools(unittest.TestCase):
    """Two tiny pools: public p1 {G1, G2}, p2 {G4}, p3 {G5}, p4 {SHORT}, p5 {G7 as W}, p6 {G4 again}; private q1 {G9}."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="c10_t21_unit_")
        self.addCleanup(self.tmp.cleanup)
        d = self.tmp.name
        write_candidates(os.path.join(d, "pub.csv"), "pub_", [
            ("pub_p_1", G[1], "B"), ("pub_p_1", G[2], "B"), ("pub_p_2", G[4], "B"), ("pub_p_3", G[5], "B"),
            ("pub_p_4", SHORT, "B"), ("pub_p_5", G[7], "W"), ("pub_p_6", G[11], "B"), ("pub_p_7", G[11], "B")])
        write_candidates(os.path.join(d, "priv.csv"), "priv_", [("priv_p_1", G[9], "B")])
        self.own = tc.PoolIndex(os.path.join(d, "pub.csv"))
        self.other = tc.PoolIndex(os.path.join(d, "priv.csv"))

    def dq(self, games, colour="B"):
        return tc.decide_question(self.own, self.other, [(i + 1, colour, g) for i, g in enumerate(games)])


class Keys(Pools):
    def test_full_key_is_t2s(self):
        for s in (G[1], G[9], SHORT):
            pts = [p for _c, p in t2.MOVE_RE.findall(s)]
            self.assertEqual(tc.full_key(pts), t2.game_keys(pts)[0])

    def test_index_summary(self):
        s = self.own.summary()
        self.assertEqual((s["rows"], s["candidates"]), (8, 7))
        self.assertEqual(s["keys_with_two_or_more_rows"], 1)
        self.assertEqual(s["keys_with_two_or_more_candidates"], 1)
        self.assertEqual(s["rows_under_min_moves"], 1)
        self.assertEqual(s["rows_not_alternating_from_black"], 0)


class Match(Pools):
    def test_one_copy_overrides(self):
        d = self.dq([G[30], G[1], G[31], G[32], G[33]])
        self.assertEqual(d["override"], "pub_p_1")
        self.assertEqual(d["copy_slots"], [2])
        self.assertEqual(d["copy_games"], [["pub_g000001"]])

    def test_two_copies_same_candidate(self):
        self.assertEqual(self.dq([G[1], G[2], G[30]])["override"], "pub_p_1")

    def test_symmetry_counts(self):
        for s in range(1, 8):
            self.assertEqual(self.dq([tc.sym_sgf(G[5], s)])["override"], "pub_p_3", s)

    def test_conflict_two_candidates(self):
        d = self.dq([G[1], G[4]])
        self.assertIsNone(d["override"])
        self.assertTrue(d["conflict"])
        self.assertEqual(d["source_candidates"], ["pub_p_1", "pub_p_2"])

    def test_conflict_one_game_in_two_libraries(self):
        d = self.dq([G[11], G[30]])                          # G11 is in pub_p_6's and pub_p_7's libraries
        self.assertIsNone(d["override"])
        self.assertTrue(d["conflict"])

    def test_truncated_or_extended_is_not_a_copy(self):
        self.assertIsNone(self.dq([tc.drop_last_move(G[1])])["override"])
        longer = G[1][:-1] + (";W[ss])" if ";W[ss]" not in G[1] and ";B[ss]" not in G[1] else ";W[aa])")
        self.assertIsNone(self.dq([longer])["override"])

    def test_short_game_never_matches(self):
        d = self.dq([SHORT])
        self.assertIsNone(d["override"])
        self.assertEqual(d["status_counts"], {"short": 1})

    def test_other_colour_not_used_and_no_conflict(self):
        d = self.dq([G[7], G[30]])                           # G7 is a W row in the pool; the question is black
        self.assertIsNone(d["override"])
        self.assertFalse(d["conflict"])
        self.assertEqual(d["opposite_colour_copies_record_only"], 1)
        d = self.dq([G[7], G[1]])
        self.assertEqual(d["override"], "pub_p_1")

    def test_other_pool_record_only(self):
        d = self.dq([G[9], G[30]])
        self.assertIsNone(d["override"])
        self.assertFalse(d["conflict"])
        self.assertEqual(d["other_pool_copies_record_only"], 1)

    def test_no_colour_and_illegal(self):
        d = tc.decide_question(self.own, self.other, [(1, "", G[1])])
        self.assertIsNone(d["override"])
        self.assertEqual(d["status_counts"], {"no_colour": 1})
        bad = G[1].replace(";B[", ";W[", 1)                   # starts with white
        self.assertEqual(self.dq([bad])["status_counts"], {"illegal": 1})

    def test_question_reader_ignores_slots_after_num_games(self):
        p = os.path.join(self.tmp.name, "q.csv")
        with open(p, "w", encoding="utf-8", newline="") as f:      # num_games 1, but G1 sits in slot 2
            w = csv.writer(f, lineterminator="\n")
            w.writerow(t2.Q_COLS)
            w.writerow(["pub_0001", 1, G[30], G[1]] + [""] * 18 + ["B", "B"] + [""] * 18)
        qs, _sha = tc.read_questions(p)
        self.assertEqual([len(g) for _q, g in qs], [1])
        self.assertIsNone(tc.decide_question(self.own, self.other, qs[0][1])["override"])


class Rows(unittest.TestCase):
    def test_new_top5(self):
        b = ["a", "b", "c", "d", "e"]
        self.assertEqual(tc.new_top5(b, "x"), ["x", "a", "b", "c", "d"])
        self.assertEqual(tc.new_top5(b, "c"), ["c", "a", "b", "d", "e"])
        self.assertEqual(tc.new_top5(b, "a"), b)
        self.assertEqual(tc.new_top5(b, "e"), ["e", "a", "b", "c", "d"])

    def write(self, data):
        d = tempfile.mkdtemp(prefix="c10_t21_rows_")
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        p = os.path.join(d, "submission_player.csv")
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_build_keeps_bytes_and_line_ends(self):
        p = self.write(b"question_id,top_1,top_2,top_3,top_4,top_5\r\npub_0001,a,b,c,d,e\r\npub_0002,f,g,h,i,j\r\n")
        data, changed = tc.build_21(p, {"pub_0001": "c", "pub_0002": "f"})
        self.assertEqual(data, b"question_id,top_1,top_2,top_3,top_4,top_5\r\npub_0001,c,a,b,d,e\r\n"
                               b"pub_0002,f,g,h,i,j\r\n")
        self.assertEqual([c["question_id"] for c in changed], ["pub_0001"])

    def test_base_refusals(self):
        for data in (b"\xef\xbb\xbfquestion_id,top_1,top_2,top_3,top_4,top_5\npub_0001,a,b,c,d,e\n",
                     b"question_id,top_1\npub_0001,a\n",
                     b"question_id,top_1,top_2,top_3,top_4,top_5\npub_0001,a,a,c,d,e\n",
                     b'question_id,top_1,top_2,top_3,top_4,top_5\npub_0001,"a",b,c,d,e\n'):
            with self.subTest(data=data[:20]):
                with self.assertRaises(tc.Drift):
                    tc.read_base_lines(self.write(data))
        p = self.write(b"question_id,top_1,top_2,top_3,top_4,top_5\npub_0001,a,b,c,d,e\n")
        with self.assertRaises(tc.Drift):
            tc.build_21(p, {"pub_0009": "x"})


# ---------------------------------------------------------------- official-shaped end to end (dry-run)
def build_tree(root, inject, base_rows=None, final=True, base_dir="results/test10/player"):
    """An official-shaped tree: per pool 400 candidates x 100 games (game 1 of candidate k is G-like unique, the rest
    fillers), 400 questions of 5 filler-unique games. inject: {(pool, question number): [sgf, ...]} put first in that
    question. -> base file path."""
    tdir = os.path.join(root, "data", "test")
    lib = {}
    for pool, pre in (("public", "pub_"), ("private", "priv_")):
        rows = []
        for k in range(1, 401):
            pid = f"{pre}p_{k:04d}"
            first = rand_game(10000 * (pool == "private") + k)
            lib[(pool, k)] = first
            rows += [(pid, first, "B")] + [(pid, FILL, "B")] * 99
        write_candidates(os.path.join(tdir, tc.C_FILES[pool]), pre, rows)
        qs = []
        for j in range(1, 401):
            own = [rand_game(50000 + 1000 * (pool == "private") + j, n=101 + i) for i in range(4)]
            qs.append(list(inject.get((pool, j), [])) + own)
        write_questions(os.path.join(tdir, tc.Q_FILES[pool]), pre, qs)
    brel = base_dir + "/submission_player.csv"
    bpath = os.path.join(root, *brel.split("/"))
    os.makedirs(os.path.dirname(bpath), exist_ok=True)
    if base_rows is None:
        base_rows = []
        for pool, pre in (("public", "pub_"), ("private", "priv_")):
            for j in range(1, 401):
                base_rows.append([f"{pre}{j:04d}"] + [f"{pre}p_{(j + i) % 400 + 1:04d}" for i in range(5)])
    with open(bpath, "wb") as f:
        f.write(tc.HEADER + b"\n" + b"".join(",".join(r).encode("ascii") + b"\n" for r in base_rows))
    sha = tc.sha256_file(bpath)
    lg = {"format": "c10 official_scores v1", "final_choice": {"rank": None, "player": None},
          "uploads": [{"id": 1, "comp": "player", "status": "uploaded", "file": brel, "sha256": sha,
                       "method": "fake", "uploaded_at": "2026-11-20T10:00:00+08:00"}]}
    if final:
        lg["final_choice"]["player"] = {"id": 1, "sha256": sha, "file": brel, "basis": "x"}
    tc.write_json(os.path.join(root, "results", "official_scores.json"), lg)
    return bpath, lib


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="c10_t21_e2e_")
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def run_tool(self, *argv):
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            return tc.main(list(argv))

    def out(self, *rel):
        return os.path.join(self.root, "results", "test21", *rel)

    def test_used(self):
        _b, lib = build_tree(self.root, {})
        bpath, lib = build_tree(self.root, {("public", 3): [lib[("public", 7)]],          # 3 -> pub_p_0007
                                            ("public", 5): [lib[("public", 6)]],          # already top 1? no: 6..10
                                            ("public", 9): [lib[("public", 9)], lib[("public", 11)]],   # conflict
                                            ("private", 2): [tc.sym_sgf(lib[("private", 400)], 3)],
                                            ("private", 4): [lib[("public", 4)]]})       # other pool
        self.assertEqual(self.run_tool("scan", "--dry-run", "--root", self.root), 0)
        scan = tc.read_json(self.out("scan", "scan.json"))
        self.assertEqual(scan["questions_overridable"], 3)
        self.assertEqual(len(scan["conflicts"]), 1)
        self.assertEqual(scan["files"][tc.Q_FILES["private"]]["other_pool_copies_record_only"], 1)
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root), 0)
        rep = tc.read_json(self.out("player", "t21_copy_report.json"))
        self.assertEqual(rep["verdict"], "used")
        self.assertEqual(rep["check_submission"], {"mode": "official", "errors": []})
        by_q = {o["question_id"]: o for o in rep["overrides"]}
        self.assertEqual(by_q["pub_0003"]["new_top5"], ["pub_p_0007", "pub_p_0004", "pub_p_0005", "pub_p_0006",
                                                         "pub_p_0008"])
        self.assertEqual(by_q["pub_0003"]["base_rank_of_candidate"], 4)
        self.assertFalse(by_q["pub_0005"]["changed"])                    # pub_p_0006 already first
        self.assertEqual(by_q["priv_0002"]["new_top5"][0], "priv_p_0400")
        self.assertIsNone(by_q["priv_0002"]["base_rank_of_candidate"])
        self.assertEqual((rep["questions_overridden"], rep["rows_changed"]), (3, 2))
        self.assertEqual(rep["changed_by_pool"], {"public": 1, "private": 1})
        with open(bpath, "rb") as f:
            b = f.read().splitlines(keepends=True)
        with open(self.out("player", "submission_player.csv"), "rb") as f:
            n = f.read().splitlines(keepends=True)
        diff = [i for i, (x, y) in enumerate(zip(b, n)) if x != y]
        self.assertEqual(len(b), len(n))
        self.assertEqual(diff, [3, 402])                                  # pub_0003, priv_0002
        self.assertEqual(rep["outputs_sha256"]["submission_player.csv"],
                         tc.sha256_file(self.out("player", "submission_player.csv")))
        # rerun without --rerun: refused; status
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root), 2)
        self.assertEqual(self.run_tool("status", "--dry-run", "--root", self.root), 0)
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root, "--rerun", "--reason", "x"), 0)
        moved = sorted(os.listdir(self.out("superseded")))                # the earlier outputs are kept, not overwritten
        self.assertEqual([m.split("_", 1)[1] for m in moved], ["player_overrides.csv", "player_submission_player.csv"])

    def test_no_copy_is_not_used(self):
        build_tree(self.root, {})
        self.assertEqual(self.run_tool("scan", "--dry-run", "--root", self.root), 3)
        nu = tc.read_json(self.out("not_used.json"))
        self.assertEqual(nu["stage"], "scan")
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root), 3)       # not_used.json exists
        self.assertFalse(os.path.exists(self.out("player", "submission_player.csv")))

    def test_apply_equal_to_base_not_used_and_changed_base_rerun(self):
        _b, lib = build_tree(self.root, {})
        bpath, lib = build_tree(self.root, {("public", 5): [lib[("public", 6)]]})            # pub_p_0006 first
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root), 3)
        nu = tc.read_json(self.out("not_used.json"))
        self.assertEqual((nu["stage"], nu["base"]["entry_id"]), ("apply", 1))
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root, "--rerun", "--reason", "x"), 3)
        rows = []
        for pool, pre in (("public", "pub_"), ("private", "priv_")):
            for j in range(1, 401):
                rows.append([f"{pre}{j:04d}"] + [f"{pre}p_{(j + i + 2) % 400 + 1:04d}" for i in range(5)])
        with open(bpath, "wb") as f:                                                          # a new base #2
            f.write(tc.HEADER + b"\n" + b"".join(",".join(r).encode("ascii") + b"\n" for r in rows))
        lg = tc.read_json(os.path.join(self.root, "results", "official_scores.json"))
        sha = tc.sha256_file(bpath)
        lg["uploads"].append(dict(lg["uploads"][0], id=2, sha256=sha))
        lg["final_choice"]["player"] = dict(lg["final_choice"]["player"], id=2, sha256=sha)
        tc.write_json(os.path.join(self.root, "results", "official_scores.json"), lg)
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root, "--rerun", "--reason", "base #2"),
                         0)
        self.assertFalse(os.path.exists(self.out("not_used.json")))
        self.assertTrue(os.listdir(self.out("superseded")))

    def test_base_refusals_and_drift(self):
        _b, lib = build_tree(self.root, {})
        bpath, _ = build_tree(self.root, {("public", 3): [lib[("public", 7)]]}, final=False)
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root), 2)      # no final choice
        build_tree(self.root, {("public", 3): [lib[("public", 7)]]})
        with open(bpath, "ab") as f:
            f.write(b"\n")
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root), 4)      # base bytes differ
        build_tree(self.root, {("public", 3): [lib[("public", 7)]]})
        lg = tc.read_json(os.path.join(self.root, "results", "official_scores.json"))
        lg["uploads"][0]["test21"] = {"base_entry_id": 0}
        tc.write_json(os.path.join(self.root, "results", "official_scores.json"), lg)
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root), 2)      # 21 on 21
        lg["uploads"][0].pop("test21")
        lg["uploads"][0]["status"] = "prepared"
        tc.write_json(os.path.join(self.root, "results", "official_scores.json"), lg)
        self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root), 2)      # not uploaded

    def test_check_submission_failure_is_drift(self):
        _b, lib = build_tree(self.root, {})
        build_tree(self.root, {("public", 3): [lib[("public", 7)]]})
        with mock.patch.object(tc.check_submission, "check", return_value=["bad"]):
            self.assertEqual(self.run_tool("apply", "--dry-run", "--root", self.root), 4)
        rep = tc.read_json(self.out("player", "t21_copy_report.json"))
        self.assertEqual(rep["verdict"], "error")


class Modes(unittest.TestCase):
    def run_tool(self, *argv):
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            return tc.main(list(argv))

    def test_root_inside_repo_refused(self):
        self.assertEqual(self.run_tool("scan", "--dry-run", "--root", os.path.join(REPO, "x")), 2)
        self.assertEqual(self.run_tool("fake", "--root", os.path.join(REPO, "x")), 2)

    def test_real_mode_flags(self):
        self.assertEqual(self.run_tool("scan", "--root", "C:/x"), 2)
        self.assertEqual(self.run_tool("sim", "--dry-run", "--root", "C:/x"), 2)

    def test_real_mode_before_test_day_writes_nothing(self):
        """Before 11/04 there is no T1 of the official files: a refusal (2), and no not_used.json in the repository."""
        nu = os.path.join(tc.OUT, "not_used.json")
        existed = os.path.exists(nu)
        with mock.patch.object(tc, "require_committed", return_value=None), \
                mock.patch.object(tc, "T1_JSON", os.path.join(tempfile.gettempdir(), "c10_t21_no_such_t1.json")):
            self.assertEqual(self.run_tool("scan"), 2)
            self.assertEqual(self.run_tool("apply"), 2)
        self.assertEqual(os.path.exists(nu), existed)

    def test_guard(self):
        with self.assertRaises(tc.Refuse):
            tc.guard_read(os.path.join(REPO, "data", "local", "x.csv"))
        with self.assertRaises(tc.Refuse):
            tc.guard_read(os.path.join(tempfile.gettempdir(), "ans" + "wers.csv"))

    def test_k5_and_sim(self):
        """check_k5_and_sim: the sim must have passed, sim first committed and the rev 2 row on main by K5-21."""
        sim = {"evaluation": {"passed": True}, "provenance": {"tool_sha256_lf": "s"}}
        early, late = tc.K5_21.replace(day=1), tc.K5_21.replace(day=4)
        with mock.patch.object(tc, "require_committed", return_value=None), \
                mock.patch.object(tc, "read_json", return_value=sim), \
                mock.patch.object(tc, "sha256_lf", return_value="s"):
            with mock.patch.object(tc, "git_last_commit_time", return_value=early), \
                    mock.patch.object(tc, "git_row_on_main", return_value=("h", early)):
                self.assertTrue(tc.check_k5_and_sim()["k5_ok"])
            for ft, rt in ((late, early), (early, late), (early, None), (None, early)):
                with self.subTest(ft=ft, rt=rt), mock.patch.object(tc, "git_last_commit_time", return_value=ft), \
                        mock.patch.object(tc, "git_row_on_main", return_value=("h" if rt else None, rt)):
                    with self.assertRaises(tc.NotUsed):
                        tc.check_k5_and_sim()
            with mock.patch.object(tc, "git_last_commit_time", return_value=early),                     mock.patch.object(tc, "git_row_on_main", return_value=("h", early)):
                sim["provenance"]["tool_sha256_lf"] = "other tool"
                with self.assertRaises(tc.Refuse):
                    tc.check_k5_and_sim()
            sim["evaluation"]["passed"] = False
            with self.assertRaises(tc.NotUsed):
                tc.check_k5_and_sim()

    def test_raw_fail_opening_before_build_writes_nothing(self):
        """Review 10/20 I3: a raw FAIL_OPENING committed before docs/TESTDAY_02_PASS.md and the build (no MANIFEST) is a
        refusal (2), not the season's 'not used'."""
        with tempfile.TemporaryDirectory(prefix="c10_t21_pre_") as d:
            out = os.path.join(d, "test21")
            with mock.patch.object(tc, "require_committed", return_value=None),                     mock.patch.object(tc, "check_t1", return_value={"verdict": "FAIL_OPENING", "_t1": {}}),                     mock.patch.object(tc, "DATA_TEST", os.path.join(d, "test")), mock.patch.object(tc, "OUT", out):
                self.assertEqual(self.run_tool("scan"), 2)
                self.assertFalse(os.path.exists(out))
                os.makedirs(os.path.join(d, "test"))
                tc.write_json(os.path.join(d, "test", "MANIFEST.json"), {"kind": "official"})
                self.assertEqual(self.run_tool("scan"), 3)               # after the build: the registered 'not used'
                self.assertEqual(tc.read_json(os.path.join(out, "not_used.json"))["stage"], "preflight")
                self.assertEqual(self.run_tool("scan"), 3)               # again: the earlier record is kept
                self.assertEqual(len(os.listdir(os.path.join(out, "superseded"))), 1)

    def test_apply_after_k6_refused(self):
        with mock.patch.object(tc, "preflight", return_value={}),                 mock.patch.object(tc, "K6_21", tc.K6_21.replace(year=2020)),                 mock.patch.object(tc, "OUT", os.path.join(tempfile.gettempdir(), "c10_t21_no_out")):
            self.assertEqual(self.run_tool("apply"), 2)


if __name__ == "__main__":
    unittest.main()
