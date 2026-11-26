"""Unit tests of tools/t14_copy.py (PREREG_14 sections 2-5, 7, 9) on small synthetic training data. No real data,
no data/local; temporary trees only. Run: python -m pytest tests/test_t14_copy.py"""

import csv
import hashlib
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
import t14_copy as tc  # noqa: E402

LETTERS = "abcdefghijklmnopqrs"


def rand_game(seed, n=120):
    r = random.Random(seed)
    pts = r.sample([a + b for a in LETTERS for b in LETTERS], n)
    return "(;SZ[19]" + "".join(f";{'BW'[i % 2]}[{p}]" for i, p in enumerate(pts)) + ")"


def write_train(d, rows):
    """rows: [(player_id, level, colour, sgf)] -> d/train_<LV>.csv for every level (game ids per file)."""
    os.makedirs(d, exist_ok=True)
    per = {lv: [] for lv in tc.LEVELS}
    for pid, lv, colour, sgf in rows:
        per[lv].append((pid, colour, sgf))
    for lv, rs in per.items():
        with open(os.path.join(d, f"train_{lv}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(tc.TRAIN_COLUMNS)
            for i, (pid, colour, sgf) in enumerate(rs):
                w.writerow([pid, f"g_{i + 1:07d}", lv, colour, sgf])


def write_questions(path, prefix, questions):
    """questions: [(colour, [sgf, ...])] -> an official-shaped rank question file."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(t2.Q_COLS)
        for j, (c, games) in enumerate(questions):
            w.writerow([f"{prefix}{j + 1:04d}", len(games)] + games + [""] * (20 - len(games))
                       + [c] * len(games) + [""] * (20 - len(games)))


G = {k: rand_game(k) for k in range(1, 40)}
SHORT = rand_game(99, n=99)
TRAIN = [("p1", "3D", "B", G[1]), ("p1", "3D", "B", G[2]), ("p1", "3D", "W", G[3]),
         ("p2", "3D", "B", G[4]), ("p3", "D", "B", G[5]), ("p3", "D", "W", G[6]),
         ("p4", "6D", "B", SHORT), ("p5", "A", "W", G[7])]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="c10_t14_unit_")
        self.addCleanup(self.tmp.cleanup)
        self.train = os.path.join(self.tmp.name, "training")
        write_train(self.train, TRAIN)
        self.index = tc.Index(self.train, "all")


class Keys(Base):
    def test_full_key_is_t2s(self):
        for s in (G[1], G[9], SHORT):
            pts = [p for _c, p in t2.MOVE_RE.findall(s)]
            self.assertEqual(tc.full_key(pts), t2.game_keys(pts)[0])

    def test_index_rows_and_pins(self):
        p = self.index.pins()
        self.assertEqual(p["rows"], 8)
        self.assertEqual(p["players"], 5)
        self.assertEqual(p["keys_with_two_or_more_rows"], 0)
        self.assertEqual(p["rows_under_min_moves"], 1)
        self.assertEqual(tc.pin_differences(p, dict(p)), {})
        self.assertIn("rows", tc.pin_differences(p, dict(p, rows=9)))

    def test_two_levels_stop(self):
        d = os.path.join(self.tmp.name, "bad")
        write_train(d, TRAIN + [("p1", "4D", "B", G[20])])
        with self.assertRaises(tc.Drift):
            tc.Index(d, "all")

    def test_exclude(self):
        idx = tc.Index(self.train, "all", exclude=[["p1", "g_0000001"]])
        self.assertEqual(idx.pins()["rows"], 7)


class Match(Base):
    def dq(self, colour, games):
        return tc.decide_question(self.index, [(i + 1, colour, g) for i, g in enumerate(games)])

    def test_one_same_side_copy_overrides(self):
        d = self.dq("B", [G[1], G[30], G[31], G[32], G[33]])
        self.assertEqual(d["override"], "3D")
        self.assertEqual(d["copy_slots"], [1])
        self.assertEqual(d["source_players"], ["p1"])

    def test_two_copies_same_player(self):
        self.assertEqual(self.dq("B", [G[1], G[2], G[30]])["override"], "3D")

    def test_symmetry_counts(self):
        for s in range(1, 8):
            self.assertEqual(self.dq("B", [tc.sym_sgf(G[5], s)])["override"], "D", s)

    def test_opposite_side_not_used(self):
        d = self.dq("W", [G[1], G[30]])
        self.assertIsNone(d["override"])
        self.assertEqual(d["opposite_side_copies_record_only"], 1)
        self.assertFalse(d["conflict"])

    def test_opposite_does_not_block_same(self):
        d = self.dq("B", [G[1], G[3]])          # G[3] is p1's white game: opposite side here
        self.assertEqual(d["override"], "3D")
        self.assertEqual(d["opposite_side_copies_record_only"], 1)

    def test_truncated_or_extended_is_not_a_copy(self):
        self.assertIsNone(self.dq("B", [tc.drop_last_move(G[1])])["override"])
        longer = G[1][:-1] + ";W[ss])" if ";W[ss]" not in G[1] else G[1][:-1] + ";W[aa])"
        self.assertIsNone(self.dq("B", [longer])["override"])

    def test_conflict_two_players_same_level(self):
        d = self.dq("B", [G[1], G[4]])          # p1 and p2 are both 3D: still a conflict (PREREG_14 4 (b))
        self.assertIsNone(d["override"])
        self.assertTrue(d["conflict"])

    def test_conflict_two_levels(self):
        d = self.dq("B", [G[1], G[5]])
        self.assertIsNone(d["override"])
        self.assertTrue(d["conflict"])

    def test_short_game_never_matches(self):
        d = self.dq("B", [SHORT])
        self.assertIsNone(d["override"])
        self.assertEqual(d["status_counts"], {"short": 1})

    def test_not_alternating(self):
        bad = G[1].replace(";W[", ";B[", 1)
        d = self.dq("B", [bad])
        self.assertEqual(d["status_counts"], {"illegal": 1})
        self.assertIsNone(d["override"])

    def test_case_of_colour(self):
        # question colours are upper-cased by the T2 reader (t02_checks.question_games)
        path = os.path.join(self.tmp.name, "q.csv")
        write_questions(path, "pub_", [("b", [G[1]])])
        qs, _sha = tc.read_questions(path)
        self.assertEqual(qs[0][1][0][1], "B")

    def test_duplicate_training_rows_one_player(self):
        d = os.path.join(self.tmp.name, "dup")
        write_train(d, TRAIN + [("p1", "3D", "B", tc.sym_sgf(G[1], 2))])
        idx = tc.Index(d, "all")
        self.assertEqual(idx.pins()["keys_with_two_or_more_rows"], 1)
        self.assertEqual(tc.decide_question(idx, [(1, "B", G[1])])["override"], "3D")
        d2 = os.path.join(self.tmp.name, "dup2")
        write_train(d2, TRAIN + [("p9", "1D", "B", G[1])])
        idx2 = tc.Index(d2, "all")
        self.assertEqual(idx2.pins()["keys_with_two_or_more_levels"], 1)
        r = tc.decide_question(idx2, [(1, "B", G[1])])
        self.assertIsNone(r["override"])
        self.assertTrue(r["conflict"])


class Slots(Base):
    def test_num_games_and_empty_colour(self):
        path = os.path.join(self.tmp.name, "q.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(t2.Q_COLS)
            # q1: num_games 1, the copy sits in slot 2 (after num_games) -> never read
            w.writerow(["pub_0001", 1, G[30], G[1]] + [""] * 18 + ["B", "B"] + [""] * 18)
            # q2: the copy has an empty colour -> 'no_colour', not counted on either side
            w.writerow(["pub_0002", 2, G[31], G[1]] + [""] * 18 + ["B", ""] + [""] * 18)
        qs, _sha = tc.read_questions(path)
        self.assertEqual([len(g) for _q, g in qs], [1, 2])
        d1 = tc.decide_question(self.index, qs[0][1])
        self.assertEqual((d1["copies"], d1["override"]), (0, None))
        d2 = tc.decide_question(self.index, qs[1][1])
        self.assertEqual((d2["copies"], d2["opposite_side_copies_record_only"], d2["override"]), (0, 0, None))
        self.assertEqual(d2["status_counts"].get("no_colour"), 1)


class RealPreflight(unittest.TestCase):
    """Real-mode preconditions on synthetic files (module paths and git patched; nothing in the repository written)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="c10_t14_real_")
        self.addCleanup(self.tmp.cleanup)
        self.d = self.tmp.name

    def test_manifest(self):
        tdir = os.path.join(self.d, "test")
        os.makedirs(tdir)
        for n in tc.RANK_FILES:
            write_questions(os.path.join(tdir, n), "pub_", [("B", [G[30]])])
        files = {n: {"sha256": tc.sha256_file(os.path.join(tdir, n)), "source_sha256": "raw" + n}
                 for n in tc.RANK_FILES}
        mp = os.path.join(tdir, "MANIFEST.json")
        tc.write_json(mp, {"kind": "rehearsal", "files": files})
        with self.assertRaises(tc.Refuse):
            tc.check_manifest(tdir)
        tc.write_json(mp, {"kind": "official", "files": files})
        self.assertEqual(len(tc.check_manifest(tdir)), 64)
        t1 = {"files": {n: {"sha256": "raw" + n} for n in tc.RANK_FILES}}
        tc.check_manifest(tdir, t1)
        t1["files"][tc.RANK_FILES[1]]["sha256"] = "other"
        with self.assertRaises(tc.Refuse):
            tc.check_manifest(tdir, t1)
        with open(os.path.join(tdir, tc.RANK_FILES[0]), "a", encoding="utf-8") as f:
            f.write("\n")
        with self.assertRaises(tc.Refuse):
            tc.check_manifest(tdir)

    def test_t1(self):
        p = os.path.join(self.d, "t1.json")
        good = {"complete": True, "stage": "raw", "provenance": {"mode": "real"}, "verdict": "PASS", "files": {}}
        with mock.patch.object(tc, "T1_JSON", p), mock.patch.object(tc, "committed_clean", return_value=True):
            tc.write_json(p, good)
            self.assertEqual(tc.check_t1()["verdict"], "PASS")
            tc.write_json(p, dict(good, verdict="FAIL_OPENING"))
            with self.assertRaises(tc.NotUsed):
                tc.check_t1()
            tc.write_json(p, dict(good, complete=False))
            with self.assertRaises(tc.Refuse):
                tc.check_t1()
        with mock.patch.object(tc, "T1_JSON", p), mock.patch.object(tc, "committed_clean", return_value=False):
            with self.assertRaises(tc.Refuse):
                tc.check_t1()

    def test_k5_and_sim(self):
        sim = os.path.join(self.d, "sim.json")
        early, late = (tc.datetime(2026, 10, 20, tzinfo=tc.TW), tc.datetime(2026, 11, 4, 11, 0, 1, tzinfo=tc.TW))
        with mock.patch.object(tc, "SIM", sim), mock.patch.object(tc, "require_committed"):
            tc.write_json(sim, {"evaluation": {"passed": True}})
            with mock.patch.object(tc, "git_first_commit_time", return_value=early):
                self.assertTrue(tc.check_k5_and_sim()["k5_ok"])
            with mock.patch.object(tc, "git_first_commit_time", side_effect=lambda r, row=None: late if row else early):
                with self.assertRaises(tc.NotUsed):
                    tc.check_k5_and_sim()
            with mock.patch.object(tc, "git_first_commit_time", return_value=None):
                with self.assertRaises(tc.NotUsed):
                    tc.check_k5_and_sim()
            tc.write_json(sim, {"evaluation": {"passed": False}})
            with mock.patch.object(tc, "git_first_commit_time", return_value=early):
                with self.assertRaises(tc.NotUsed):
                    tc.check_k5_and_sim()

    def test_preflight_writes_not_used_on_t1(self):
        import argparse
        ctx = tc.Ctx(argparse.Namespace(dry_run=False, root=None, index_part=None, train_dir=None, exclude=None,
                                        pins=None, split_csv=None, rerun=False, reason=None))
        ctx.out = os.path.join(self.d, "out")
        with mock.patch.object(tc, "require_committed"), \
                mock.patch.object(tc, "check_k5_and_sim", return_value={"k5_ok": True}), \
                mock.patch.object(tc, "check_t1", side_effect=tc.NotUsed("T1 verdict 'FAIL_STRUCT'")), \
                mock.patch.object(tc, "provenance", return_value={}):
            with self.assertRaises(tc.NotUsed):
                tc.preflight(ctx, "apply", None)
        nu = tc.read_json(os.path.join(ctx.out, "not_used.json"))
        self.assertEqual(nu["stage"], "preflight")
        self.assertIn("FAIL_STRUCT", nu["reason"])

    def test_split_csv_never_brings_back_val_players(self):
        split = tc.c10data.read_split(tc.SPLIT_CSV)
        val = sorted(p for p, info in split.items() if info["split"] != "train")[:3]
        self.assertTrue(val)
        fake = os.path.join(self.d, "split.csv")
        with open(fake, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["player_id", "rank", "n_games", "bin", "split"])
            for pid in val + ["p_unit_1"]:
                w.writerow([pid, "D", 1, "x", "train"])
        self.assertEqual(tc.players_of_part("train", fake), {"p_unit_1"})


class Build(unittest.TestCase):
    def test_only_changed_rows_and_line_ends_kept(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "submission_rank.csv")
            with open(p, "wb") as f:
                f.write(b"question_id,pred_rank\r\npub_0001,D\r\npub_0002,3D\r\npub_0003,A\r\n")
            data, changed = tc.build_14(p, {"pub_0002": "3D", "pub_0003": "6D"})
            self.assertEqual(data, b"question_id,pred_rank\r\npub_0001,D\r\npub_0002,3D\r\npub_0003,6D\r\n")
            self.assertEqual(changed, [{"question_id": "pub_0003", "base": "A", "new": "6D"}])
            with self.assertRaises(tc.Drift):
                tc.build_14(p, {"pub_0009": "D"})

    def test_quoted_base_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.csv")
            with open(p, "wb") as f:
                f.write(b'question_id,pred_rank\n"pub_0001",D\n')
            with self.assertRaises(tc.Drift):
                tc.read_base_lines(p)


class Guards(unittest.TestCase):
    def test_no_local(self):
        with self.assertRaises(tc.Refuse):
            tc.guard_read(os.path.join(REPO, "data", "local", "x.csv"))
        with self.assertRaises(tc.Refuse):
            tc.guard_read(os.path.join(REPO, "x_" + "ans" + "wers.csv"))

    def ns(self, **kw):
        import argparse
        d = dict(dry_run=True, root=None, index_part=None, train_dir=None, exclude=None, pins=None, rerun=False,
                 reason=None)
        d.update(kw)
        return argparse.Namespace(**d)

    def test_dry_run_part_all_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(tc.Refuse):
                tc.Ctx(self.ns(root=d, index_part="all"))

    def test_root_inside_repo_refused(self):
        with self.assertRaises(tc.Refuse):
            tc.Ctx(self.ns(root=os.path.join(REPO, "results")))

    def test_real_mode_takes_no_dry_run_options(self):
        with self.assertRaises(tc.Refuse):
            tc.Ctx(self.ns(dry_run=False, index_part="fit"))

    def test_pins_and_sim_are_real_only(self):
        self.assertEqual(tc.main(["pins", "--dry-run"]), 2)
        self.assertEqual(tc.main(["sim", "--dry-run"]), 2)


class EndToEnd(unittest.TestCase):
    """A 400 + 400 official-shaped tree on synthetic training data, the dry-run apply and its exit codes."""

    def build(self, root, copies):
        train = os.path.join(root, "training")
        write_train(train, TRAIN)
        split = os.path.join(root, "split.csv")
        with open(split, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["player_id", "rank", "n_games", "bin", "split"])
            for pid, lv in sorted({(r[0], r[1]) for r in TRAIN}):
                w.writerow([pid, lv, 1, "x", "train"])
        tdir = os.path.join(root, "data", "test")
        os.makedirs(tdir)
        files = {}
        for fi, (pool, name) in enumerate(zip(tc.POOLS, tc.RANK_FILES)):
            qs = []
            for j in range(400):
                games = [rand_game(10_000 + fi * 10_000 + j * 7 + k) for k in range(5)]
                if (pool, j) in copies:
                    games[2] = copies[(pool, j)]
                qs.append(("B", games))
            p = os.path.join(tdir, name)
            write_questions(p, ("pub_", "priv_")[fi], qs)
            files[name] = {"sha256": tc.sha256_file(p)}
        tc.write_json(os.path.join(tdir, "MANIFEST.json"), {"kind": "t14-unit", "files": files})
        bdir = os.path.join(root, "results", "test09", "rank")
        os.makedirs(bdir)
        bpath = os.path.join(bdir, "submission_rank.csv")
        base = [(f"{('pub_', 'priv_')[fi]}{j + 1:04d}", "D") for fi in range(2) for j in range(400)]
        with open(bpath, "wb") as f:
            f.write(b"question_id,pred_rank\n" + b"".join(f"{q},{lv}\n".encode() for q, lv in base))
        sha = tc.sha256_file(bpath)
        tc.write_json(os.path.join(root, "results", "official_scores.json"), {
            "format": "c10 official_scores v1",
            "final_choice": {"rank": {"id": 3, "sha256": sha, "file": "results/test09/rank/submission_rank.csv"}},
            "uploads": [{"id": 3, "comp": "rank", "status": "uploaded", "sha256": sha,
                         "file": "results/test09/rank/submission_rank.csv"}]})
        self.split = split
        return train, bpath

    def run_tool(self, cmd, root, train, *extra):
        return tc.main([cmd, "--dry-run", "--root", root, "--index-part", "train", "--train-dir", train,
                        "--split-csv", self.split] + list(extra))

    def test_copies_override(self):
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, bpath = self.build(root, {("public", 4): G[1], ("private", 9): tc.sym_sgf(G[5], 3),
                                             ("private", 10): G[3]})
            self.assertEqual(tc.players_of_part("train", self.split), {"p1", "p2", "p3", "p4", "p5"})
            self.assertEqual(self.run_tool("apply", root, train), 0)
            rep = tc.read_json(os.path.join(root, "results", "test14", "rank", "t14_copy_report.json"))
            self.assertEqual(rep["verdict"], "used")
            self.assertEqual(rep["questions_overridden"], 2)          # the white copy (private 11) is opposite side
            self.assertEqual({r["question_id"]: r["new_level"] for r in rep["overrides"]},
                             {"pub_0005": "3D", "priv_0010": "D"})
            self.assertEqual(rep["rows_changed"], 1)                  # priv_0010 is already D in the base
            self.assertEqual(rep["check_submission"]["errors"], [])
            with open(bpath, "rb") as f:
                b = f.read().splitlines()
            with open(os.path.join(root, "results", "test14", "rank", "submission_rank.csv"), "rb") as f:
                o = f.read().splitlines()
            self.assertEqual([i for i, (x, y) in enumerate(zip(b, o)) if x != y], [5])
            self.assertEqual(o[5], b"pub_0005,3D")
            self.assertEqual(self.run_tool("apply", root, train), 2)      # exists: only with --rerun --reason
            self.assertEqual(self.run_tool("apply", root, train, "--rerun", "--reason", "unit"), 0)

    def test_no_copy_is_not_used(self):
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, _b = self.build(root, {("public", 0): G[3]})      # opposite side only
            self.assertEqual(self.run_tool("apply", root, train), 3)
            self.assertTrue(os.path.isfile(os.path.join(root, "results", "test14", "not_used.json")))
            self.assertFalse(os.path.exists(os.path.join(root, "results", "test14", "rank", "submission_rank.csv")))
            self.assertEqual(self.run_tool("apply", root, train, "--rerun", "--reason", "x"), 3)   # not_used is final

    def test_same_level_as_base_is_not_used(self):
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, _b = self.build(root, {("public", 0): G[5]})      # p3 is D, the base says D
            self.assertEqual(self.run_tool("apply", root, train), 3)
            nu = tc.read_json(os.path.join(root, "results", "test14", "not_used.json"))
            self.assertIn("equal the base", nu["reason"])

    def test_base_changed_after_not_used(self):
        """PREREG_14 7: an apply 'not used' because the overrides equal the base may be rerun on a CHANGED base only."""
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, bpath = self.build(root, {("public", 0): G[5]})      # p3 is D, the base says D
            self.assertEqual(self.run_tool("apply", root, train), 3)
            self.assertEqual(self.run_tool("apply", root, train, "--rerun", "--reason", "same base"), 3)
            # a new final: entry 4, another file (08) whose pub_0001 is 3D
            nb = os.path.join(root, "results", "test08", "rank", "submission_rank.csv")
            os.makedirs(os.path.dirname(nb))
            with open(bpath, "rb") as f:
                data = f.read().replace(b"pub_0001,D\n", b"pub_0001,3D\n", 1)
            with open(nb, "wb") as f:
                f.write(data)
            lp = os.path.join(root, "results", "official_scores.json")
            lg = tc.read_json(lp)
            sha = tc.sha256_file(nb)
            lg["uploads"].append({"id": 4, "comp": "rank", "status": "uploaded", "sha256": sha,
                                  "file": "results/test08/rank/submission_rank.csv"})
            lg["final_choice"]["rank"] = {"id": 4, "sha256": sha, "file": "results/test08/rank/submission_rank.csv"}
            tc.write_json(lp, lg)
            self.assertEqual(self.run_tool("apply", root, train), 3)    # without --rerun: not_used stands
            self.assertEqual(self.run_tool("apply", root, train, "--rerun", "--reason", "base changed to #4"), 0)
            rep = tc.read_json(os.path.join(root, "results", "test14", "rank", "t14_copy_report.json"))
            self.assertEqual(rep["base"]["entry_id"], 4)
            self.assertEqual(rep["rows_changed"], 1)
            self.assertIn("superseded_not_used", rep["preflight"])
            self.assertFalse(os.path.exists(os.path.join(root, "results", "test14", "not_used.json")))
            self.assertEqual(len(os.listdir(os.path.join(root, "results", "test14", "superseded"))), 1)

    def test_final_not_set_refused(self):
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, _b = self.build(root, {("public", 0): G[1]})
            lp = os.path.join(root, "results", "official_scores.json")
            lg = tc.read_json(lp)
            lg["final_choice"]["rank"] = None
            tc.write_json(lp, lg)
            self.assertEqual(self.run_tool("apply", root, train), 2)

    def test_final_on_a_14_entry_refused(self):
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, _b = self.build(root, {("public", 0): G[1]})
            lp = os.path.join(root, "results", "official_scores.json")
            lg = tc.read_json(lp)
            lg["uploads"][0]["test14"] = {"x": 1}
            tc.write_json(lp, lg)
            self.assertEqual(self.run_tool("apply", root, train), 2)

    def test_base_bytes_changed_is_drift(self):
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, bpath = self.build(root, {("public", 0): G[1]})
            with open(bpath, "ab") as f:
                f.write(b"\n")
            self.assertEqual(self.run_tool("apply", root, train), 4)

    def test_pins_mismatch_is_drift(self):
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, _b = self.build(root, {("public", 0): G[1]})
            pins = os.path.join(root, "pins.json")
            good = tc.Index(train, "train", split_csv=self.split).pins()
            tc.write_json(pins, {"parts": {"train": good}})
            self.assertEqual(self.run_tool("scan", root, train, "--pins", pins), 0)
            tc.write_json(pins, {"parts": {"train": dict(good, distinct_full_keys=1)}})
            self.assertEqual(self.run_tool("scan", root, train, "--pins", pins, "--rerun", "--reason", "u"), 4)

    def test_scan_records(self):
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, _b = self.build(root, {("public", 0): G[1], ("public", 1): G[3]})
            self.assertEqual(self.run_tool("scan", root, train), 0)
            s = tc.read_json(os.path.join(root, "results", "test14", "scan", "scan.json"))
            pub = s["files"][tc.RANK_FILES[0]]
            self.assertEqual(pub["copies"], 1)
            self.assertEqual(pub["opposite_side_copies_record_only"], 1)
            self.assertEqual(s["questions_overridable"], 1)
            self.assertEqual(s["overridable"][0]["question_id"], "pub_0001")

    def test_scan_without_copies_writes_not_used(self):
        with tempfile.TemporaryDirectory(prefix="c10_t14_e2e_") as root:
            train, _b = self.build(root, {})
            self.assertEqual(self.run_tool("scan", root, train), 3)
            self.assertTrue(os.path.isfile(os.path.join(root, "results", "test14", "not_used.json")))
            self.assertEqual(self.run_tool("apply", root, train), 3)
            self.assertEqual(self.run_tool("apply", root, train, "--rerun", "--reason", "x"), 3)   # scan's not_used


if __name__ == "__main__":
    unittest.main()
