"""tools/trailing_pass.py (PREREG_02 rev 9, docs/PREREG_02_REV9_TRAILING_PASS.md). Stdlib + the tool only.

Unit tests of the game rule, the CSV cell scan and the byte-level edit, and an end-to-end run on a synthetic tree of
the official shape (400 questions per file, 400 x 100 candidate games per pool; synthetic games, no training or
validation data). The T1 reports are written with the frozen t02_checks.t1_check (the same function the t1 command
calls); the ingest step runs the unchanged tools/t02_test_bundle.py in a scratch root.
"""

import csv
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import t02_checks as tc  # noqa: E402
import trailing_pass as tp  # noqa: E402

LETTERS = "abcdefghijklmnopqrs"


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ================================================================ game rule
class GameRule(unittest.TestCase):
    def cls(self, s):
        return tp.classify_game(s)

    def test_no_pass(self):
        self.assertEqual(self.cls("(;SZ[19];B[pd];W[dp])")[0], "no_pass")

    def test_single_trailing_pass(self):
        st, new, d = self.cls("(;SZ[19];B[pd];W[dp];B[])")
        self.assertEqual((st, new, d["pass_nodes"]), ("trailing", "(;SZ[19];B[pd];W[dp])", 1))

    def test_two_trailing_passes_tt_and_empty(self):
        for tail in (";B[tt];W[tt]", ";B[];W[]", ";B[tt];W[]", ";W[];B[]"):
            with self.subTest(tail):
                st, new, d = self.cls("(;SZ[19];B[pd];W[dp]%s)" % tail)
                self.assertEqual((st, new, d["pass_nodes"], d["removed"]), ("trailing", "(;SZ[19];B[pd];W[dp])", 2, tail))

    def test_whitespace_kept_outside_the_tail(self):
        st, new, d = self.cls("(;SZ[19];B[pd];W[dp]\n; B [] \n ;W[tt]\n)")
        self.assertEqual(st, "trailing")
        self.assertEqual(new, "(;SZ[19];B[pd];W[dp]\n\n)")
        self.assertEqual(tc.sgf_check(new)["reasons"], [])

    def test_mid_game_pass_not_edited(self):
        # two passes in the middle keep the colours of the real moves alternating: T1.4 reason is 'pass' only
        s = "(;SZ[19];B[pd];W[];B[];W[dp])"
        self.assertEqual(tc.sgf_check(s)["reasons"], ["pass"])
        self.assertEqual(self.cls(s)[0], "mid_game_pass")
        # mid-game and trailing at once: not edited either
        self.assertEqual(self.cls("(;SZ[19];B[pd];W[];B[];W[dp];B[])")[0], "mid_game_pass")

    def test_pass_node_with_other_properties_not_edited(self):
        for s in ("(;SZ[19];B[pd];W[dp];B[]C[end])", "(;SZ[19];B[pd];W[dp];B[]BL[10])"):
            with self.subTest(s):
                self.assertEqual(self.cls(s)[0], "pass_not_trailing_other")

    def test_variation_or_no_closing_paren_not_edited(self):
        for s in ("(;SZ[19];B[pd];W[dp](;B[]))", "(;SZ[19];B[pd];W[dp];B[]", "(;SZ[19];B[pd];W[dp];B[][])"):
            with self.subTest(s):
                self.assertNotEqual(self.cls(s)[0], "trailing")

    def test_same_colour_tail_is_a_tail(self):
        # the registration does not constrain the colours of the pass nodes (rev 10)
        self.assertEqual(self.cls("(;SZ[19];B[pd];W[dp];W[])")[0], "trailing")

    def test_pass_with_another_reason_not_edited(self):
        for s in ("(;SZ[19]AB[dd];B[pd];W[dp];B[])", "(;SZ[13];B[pd];W[dp];B[])", "(;SZ[19];W[pd];B[dp];W[])",
                  "(;SZ[19];B[pd];B[dp];W[])", "(;SZ[19];B[])"):
            with self.subTest(s):
                self.assertEqual(self.cls(s)[0], "pass_with_other_reason")

    def test_lowercase_or_other_values_are_not_passes(self):
        st, _n, _d = self.cls("(;SZ[19];B[pd];W[dp];B[TT])")
        self.assertNotEqual(st, "trailing")


# ================================================================ CSV scan
class CellScan(unittest.TestCase):
    def test_matches_csv_reader(self):
        texts = ['a,b,c\n1,"x,y",\n\n"q""q",2,3', 'a,b\r\n"1\r\n2",3\r\n', 'a\n', '', 'a,"",b\n,,\n',
                 '"(;SZ[19];B[pd])",B\r\n']
        for t in texts:
            with self.subTest(t):
                got = [[c[0] for c in r] for r in tp.scan_cells(t)]
                self.assertEqual(got, tp._csv_rows(t))
                for r in tp.scan_cells(t):
                    for v, a, b, q in r:
                        if '""' not in t[a:b]:
                            self.assertEqual(t[a:b], v)

    def test_refuses_text_after_quote(self):
        with self.assertRaises(ValueError):
            tp.scan_cells('"a"b,c\n')


def q_header():
    return tc.Q_COLS


def write_csv(path, header, rows, lineterminator="\n", bom=False, quote_all=False):
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator=lineterminator, quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL)
    w.writerow(header)
    w.writerows(rows)
    data = (b"\xef\xbb\xbf" if bom else b"") + buf.getvalue().encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


class FileEdit(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="tp_unit_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def qrow(self, qid, games, colour="B", extra_after=None):
        sg = games + [""] * (20 - len(games))
        if extra_after is not None:
            sg[len(games)] = extra_after
        return [qid, str(len(games))] + sg + [colour] * len(games) + [""] * (20 - len(games))

    def test_only_the_tail_bytes_go_bom_crlf_quotes_kept(self):
        g_clean = "(;SZ[19];B[pd];W[dp])"
        rows = [self.qrow("pub_q_1", [g_clean + "", "(;SZ[19];B[pd];W[dp];B[];W[tt])"] + [g_clean] * 3),
                self.qrow("pub_q_2", ["(;SZ[19];B[pd];W[];B[];W[dp])"] + [g_clean] * 4)]
        p = os.path.join(self.d, "rank_prediction_test_public.csv")
        old = write_csv(p, q_header(), rows, lineterminator="\r\n", bom=True, quote_all=True)
        rec, new = tp.analyse_file(p, "questions")
        self.assertIsNone(rec["scan_problem"])
        self.assertEqual(rec["counts"]["trailing"], 1)
        self.assertEqual(rec["counts"]["mid_game_pass"], 1)
        e = rec["edited"][0]
        self.assertEqual(old[e["file_bytes"][0]:e["file_bytes"][1]], b";B[];W[tt]")
        self.assertEqual(rec["counts"]["pass_nodes_removed"], 2)
        self.assertEqual(new, old.replace(b";B[];W[tt]", b"", 1))
        self.assertTrue(new.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(new.count(b"\r\n"), old.count(b"\r\n"))
        self.assertEqual(rec["bytes_removed"], len(";B[];W[tt]"))

    def test_slot_after_num_games_is_not_a_game(self):
        g = "(;SZ[19];B[pd];W[dp])"
        rows = [self.qrow("pub_q_1", [g] * 5, extra_after="(;SZ[19];B[pd];W[dp];B[])")]
        p = os.path.join(self.d, "rank_prediction_test_public.csv")
        write_csv(p, q_header(), rows)
        rec, new = tp.analyse_file(p, "questions")
        self.assertEqual(rec["games"], 5)
        self.assertIsNone(new)

    def test_escaped_quote_cell_not_edited(self):
        rows = [["pub_g_1", "pub_p_1", '(;SZ[19]GC["x"];B[pd];W[dp];B[])', "B"]]
        p = os.path.join(self.d, "player_identification_candidates_public.csv")
        write_csv(p, tc.C_COLS, rows)
        rec, new = tp.analyse_file(p, "candidates")
        self.assertIsNone(new)
        self.assertEqual(rec["counts"]["trailing_unusual_cell"], 1)

    def test_empty_file_is_a_scan_problem(self):
        p = os.path.join(self.d, "player_identification_candidates_public.csv")
        open(p, "wb").close()
        rec, new = tp.analyse_file(p, "candidates")
        self.assertIsNone(new)
        self.assertTrue(rec["scan_problem"])

    def test_candidates_lf(self):
        rows = [["pub_g_%d" % i, "pub_p_1", "(;SZ[19];B[pd];W[dp]%s)" % (";B[]" if i % 2 else ""), "B"]
                for i in range(6)]
        p = os.path.join(self.d, "player_identification_candidates_public.csv")
        old = write_csv(p, tc.C_COLS, rows)
        rec, new = tp.analyse_file(p, "candidates")
        self.assertEqual(rec["counts"]["trailing"], 3)
        self.assertEqual(new, old.replace(b";B[]", b""))


# ================================================================ synthetic official tree
def synthetic_tree(dst, seed=7):
    """The six official files in dst: official shape, synthetic games (first move pd, 30 alternating moves)."""
    rng = random.Random(seed)

    def game():
        mv = ["pd"] + ["".join(rng.choice(LETTERS) for _ in range(2)) for _ in range(29)]
        return "(;SZ[19]" + "".join(";%s[%s]" % ("BW"[k % 2], p) for k, p in enumerate(mv)) + ")"

    sizes = [5] * 100 + [10] * 100 + [15] * 100 + [20] * 100
    for pool, pre in (("public", "pub_"), ("private", "priv_")):
        for kind, name in (("rank", "rank_prediction_test_%s.csv" % pool),
                           ("player", "player_identification_test_%s.csv" % pool)):
            rows = []
            for k, n in enumerate(sizes):
                colour = "B" if kind == "player" or k % 2 else "W"
                rows.append(["%sq_%s_%04d" % (pre, kind[0], k), str(n)] + [game() for _ in range(n)] + [""] * (20 - n)
                            + [colour] * n + [""] * (20 - n))
            write_csv(os.path.join(dst, name), tc.Q_COLS, rows)
        rows = [["%sg_%06d" % (pre, k * 100 + j), "%sp_%04d" % (pre, k), game(), "B"]
                for k in range(400) for j in range(100)]
        write_csv(os.path.join(dst, "player_identification_candidates_%s.csv" % pool), tc.C_COLS, rows)


def inject(src_dir, dst_dir, share=0.3, mid_share=0.0, comps=("rank", "player"), salt="x", mid_comps=("rank",)):
    """Copies of the six files with trailing passes (share of the T1.4 games of the given competitions) and
    optionally mid-game passes (rank only by default); byte-level insertions at the closing ')' of the selected games."""
    os.makedirs(dst_dir, exist_ok=True)
    stats = {}
    for comp in ("rank", "player"):
        for _pool, kind, name in tc.FILES[comp]:
            raw = open(os.path.join(src_dir, name), "rb").read()
            if comp not in comps:
                open(os.path.join(dst_dir, name), "wb").write(raw)
                continue
            text = raw.decode("utf-8")
            t = tc.Table(os.path.join(src_dir, name))
            want = {g[3] for g in tc.file_games(t, kind)}
            ins, n_t, n_m, k = [], 0, 0, 0
            for rec in tp.scan_cells(text):
                for v, a, b, _q in rec:
                    if v not in want or not v.endswith(")"):
                        continue
                    k += 1
                    u = int(hashlib.sha256(("%s|%s|%d" % (salt, name, k)).encode()).hexdigest()[:8], 16) / 2 ** 32
                    if comp in mid_comps and u < mid_share:
                        i = v.index(";W[") + a          # right after the first black move: two passes, then go on
                        ins.append((i, ";W[];B[]"))
                        n_m += 1
                    elif u < mid_share + share:
                        last = "BW"[len(tp.tc.MOVE_RE.findall(v)) % 2]   # the side to move after the last move
                        other = "W" if last == "B" else "B"
                        tail = [";%s[]" % last, ";%s[tt]" % last, ";%s[];%s[]" % (last, other),
                                ";%s[tt];%s[tt]" % (last, other)][k % 4]
                        ins.append((b - 1, tail))
                        n_t += 1
            parts, pos = [], 0
            for i, s in sorted(ins):
                parts.append(text[pos:i])
                parts.append(s)
                pos = i
            parts.append(text[pos:])
            open(os.path.join(dst_dir, name), "wb").write("".join(parts).encode("utf-8"))
            stats[name] = {"trailing": n_t, "mid": n_m}
    return stats


def write_t1(root, comp, tests_dir, reason=None):
    res = tc.t1_check(comp, tests_dir)
    res["provenance"] = {"mode": "dry-run", "rerun_reason": reason}
    res["stage"] = "raw"
    p = os.path.join(root, "results", "test02", "checks", comp, "t1.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f)
    return res["verdict"]


def ingest(root, src, rerun_reason=None):
    cmd = [sys.executable, "-B", os.path.join(TOOLS, "t02_test_bundle.py"), "ingest", "--src", src, "--root", root]
    if rerun_reason:
        cmd += ["--rerun", "--reason", rerun_reason]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        raise AssertionError(r.stdout + r.stderr)


def run_tp(*argv):
    try:
        return tp.main(list(argv))
    except SystemExit as e:
        return e


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.mkdtemp(prefix="tp_e2e_")
        cls.clean = os.path.join(cls.base, "clean")
        synthetic_tree(cls.clean)

    @classmethod
    def tearDownClass(cls):
        for r, _d, fs in os.walk(cls.base):
            for f in fs:
                os.chmod(os.path.join(r, f), 0o666)
        shutil.rmtree(cls.base, ignore_errors=True)

    def tree(self, name, share=0.3, mid_share=0.0, comps=("rank", "player")):
        root = os.path.join(self.base, name)
        dl = os.path.join(self.base, name + "_download")
        stats = inject(self.clean, dl, share, mid_share, comps, salt=name)
        ingest(root, dl)
        raw = os.path.join(root, "data", "test", "raw")
        v = {c: write_t1(root, c, raw) for c in ("rank", "player")}
        return root, raw, stats, v

    def prep(self, root):
        return tc.read_json(os.path.join(root, "results", "test02", "checks", "trailing_pass", "prepare.json"))

    def test_clean_tree_not_needed(self):
        root, _raw, _s, v = self.tree("clean_tree", share=0.0)
        self.assertEqual(v, {"rank": "PASS", "player": "PASS"})
        self.assertEqual(run_tp("prepare", "--root", root), 2)
        p = self.prep(root)
        self.assertEqual({c: p["comps"][c]["decision"] for c in p["comps"]}, {"rank": "not_needed", "player": "not_needed"})
        self.assertFalse(os.path.exists(os.path.join(root, "data", "test_pass")))

    def test_trailing_passes_applied_and_equal_to_the_clean_files(self):
        root, raw, stats, v = self.tree("trailing")
        self.assertEqual(v, {"rank": "FAIL_OPENING", "player": "FAIL_OPENING"})
        self.assertTrue(all(s["trailing"] > 0 for s in stats.values()))
        before = {n: sha(open(os.path.join(raw, n), "rb").read()) for n in tp.ALL_NAMES + ["SOURCE.json"]}
        self.assertEqual(run_tp("prepare", "--root", root), 0)
        self.assertEqual(before, {n: sha(open(os.path.join(raw, n), "rb").read()) for n in before},
                         "prepare never writes data/test/raw")
        p = self.prep(root)
        self.assertEqual(p["applied_comps"], ["rank", "player"])
        stripped = os.path.join(root, "data", "test_pass", "stripped")
        for n in tp.ALL_NAMES:
            with self.subTest(n):
                a = open(os.path.join(stripped, n), "rb").read()
                b = open(os.path.join(self.clean, n), "rb").read()
                self.assertEqual(sha(a), sha(b))
                self.assertEqual(p["files"][n]["games_stripped"], stats[n]["trailing"])
                self.assertEqual(p["files"][n]["sha256_before"],
                                 sha(open(os.path.join(root, "data", "test_pass", "original", n), "rb").read()))
        # verify before the ingest fails, refuses a second prepare without --rerun
        self.assertEqual(run_tp("verify", "--root", root), 1)
        self.assertIsInstance(run_tp("prepare", "--root", root), SystemExit)
        ingest(root, stripped, rerun_reason="rev 9 test")
        self.assertEqual(run_tp("verify", "--root", root, "--rerun", "--reason", "t"), 1)    # t1 not rerun yet
        for c in ("rank", "player"):
            self.assertEqual(write_t1(root, c, raw, reason="rev 9 test"), "PASS")
        self.assertEqual(run_tp("verify", "--root", root, "--rerun", "--reason", "after t1"), 0)
        # prepare refuses on the copy (SOURCE.json names data/test_pass)
        e = run_tp("prepare", "--root", root, "--rerun", "--reason", "again")
        self.assertIsInstance(e, SystemExit)
        self.assertIn("already holds the rev 9 copy", str(e))

    def test_mid_game_control_rank_not_applicable_player_applied(self):
        root, _raw, stats, v = self.tree("control", share=0.3, mid_share=0.03)
        self.assertEqual(v["rank"], "FAIL_OPENING")
        self.assertEqual(run_tp("prepare", "--root", root), 0)
        p = self.prep(root)
        self.assertEqual(p["comps"]["rank"]["decision"], "not_applicable")
        self.assertEqual(p["comps"]["rank"]["copy_t1_predicted"]["verdict"], "FAIL_OPENING")
        self.assertEqual(p["comps"]["player"]["decision"], "apply")
        for n in tp.ALL_NAMES:
            if p["files"][n]["comp"] == "rank":
                self.assertFalse(p["files"][n]["edited"])
                self.assertEqual(p["files"][n]["sha256_after"], p["files"][n]["sha256_before"])
                self.assertGreater(p["files"][n]["counts"]["mid_game_pass"], 0)

    def test_mid_game_only_rank_not_applicable_exit_2(self):
        root, _raw, _s, v = self.tree("midonly", share=0.0, mid_share=0.03, comps=("rank",))
        self.assertEqual(v, {"rank": "FAIL_OPENING", "player": "PASS"})
        self.assertEqual(run_tp("prepare", "--root", root), 2)
        p = self.prep(root)
        self.assertEqual(p["comps"]["rank"]["decision"], "not_applicable")
        self.assertIn("no trailing-pass game", p["comps"]["rank"]["reason"])

    def test_under_one_percent_not_needed(self):
        root, _raw, _s, v = self.tree("few", share=0.005, comps=("rank",))
        self.assertEqual(v["rank"], "PASS")
        self.assertEqual(run_tp("prepare", "--root", root), 2)

    def test_build_present_refused(self):
        root, _raw, _s, _v = self.tree("built", comps=("rank",))
        os.makedirs(os.path.join(root, "results", "test02", "checks", "built"))
        e = run_tp("prepare", "--root", root)
        self.assertIsInstance(e, SystemExit)
        self.assertIn("STOP", str(e))

    def test_t1_of_other_files_refused(self):
        root, raw, _s, _v = self.tree("stale", comps=("rank",))
        with open(os.path.join(raw, "rank_prediction_test_public.csv"), "ab") as f:
            f.write(b"\n")
        e = run_tp("prepare", "--root", root)
        self.assertIsInstance(e, SystemExit)
        self.assertIn("not of these files", str(e))

    def test_root_inside_repo_refused(self):
        e = run_tp("prepare", "--root", os.path.join(REPO, "results"))
        self.assertIsInstance(e, SystemExit)


if __name__ == "__main__":
    unittest.main()
