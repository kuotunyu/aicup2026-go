"""tools/root_props.py (PREREG_02 rev 11, docs/PREREG_22_ROOT_PROPS.md). Stdlib + the tools only.

Unit tests of the root rule and the byte-level file edit, and end-to-end runs on the synthetic official-shape tree of
tests/test_trailing_pass.py (synthetic games; no training or validation data), including the order with rev 9.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
for d in (TOOLS, HERE):
    if d not in sys.path:
        sys.path.insert(0, d)
import t02_checks as tc  # noqa: E402
import trailing_pass as tp  # noqa: E402
import root_props as rp  # noqa: E402
from test_trailing_pass import inject, ingest, synthetic_tree, write_t1  # noqa: E402

PLAIN = "(;SZ[19];B[pd];W[dp];B[pp])"


def sha(b):
    return hashlib.sha256(b).hexdigest()


class GameRule(unittest.TestCase):
    def test_plain_untouched(self):
        self.assertEqual(rp.classify_game(PLAIN)[:2], ("plain", None))

    def test_v11_placement_and_faq_props(self):
        for s in ("(;SZ[19]HD[2];B[pd];W[dp];B[pp])", "(;SZ[19]HD[0]CA[UTF-8]RE[W+R];B[pd];W[dp];B[pp])",
                  "(;CA[UTF-8]SZ[19]HD[9]RE[B+12.5];B[pd];W[dp];B[pp])", "(;SZ[19]KM[6.5]RU[Japanese]PB[]PW[x];B[pd];W[dp];B[pp])",
                  "(;SZ[19]HD[1]C[a\\]b];B[pd];W[dp];B[pp])"):
            with self.subTest(s):
                st, new, _d = rp.classify_game(s)
                self.assertEqual((st, new), ("root_props", PLAIN))
                self.assertTrue(tc.FAST_SGF.fullmatch(new))

    def test_whitespace_outside_removed_tokens_kept(self):
        st, new, _d = rp.classify_game("(; SZ[19] HD[3] ;B[pd];W[dp];B[pp])")
        self.assertEqual((st, new), ("root_props", "(; SZ[19] ;B[pd];W[dp];B[pp])"))

    def test_setup_props_kept_as_violations(self):
        st, new, _d = rp.classify_game("(;SZ[19]HD[2]AB[dd][pp];B[pd];W[dp])")
        self.assertEqual(new, "(;SZ[19]AB[dd][pp];B[pd];W[dp])")
        self.assertEqual(tc.sgf_check(new)["reasons"], ["setup AB"])
        self.assertEqual(rp.classify_game("(;SZ[19]HA[2]AB[dd];B[pd])")[0], "plain")

    def test_non_root_nodes_never_touched(self):
        st, new, _d = rp.classify_game("(;SZ[19]HD[1];B[pd]C[x];W[dp]RE[W+R])")
        self.assertEqual(new, "(;SZ[19];B[pd]C[x];W[dp]RE[W+R])")
        self.assertEqual(rp.classify_game("(;SZ[19];B[pd]C[x];W[dp])")[0], "plain")

    def test_trailing_pass_kept(self):
        st, new, _d = rp.classify_game("(;SZ[19]HD[0];B[pd];W[dp];B[])")
        self.assertEqual(new, "(;SZ[19];B[pd];W[dp];B[])")
        self.assertEqual(tp.classify_game(new)[1], "(;SZ[19];B[pd];W[dp])")

    def test_unparsed_not_edited(self):
        for s in ("(;SZ[19]HD[2;B[pd])", "SZ[19]HD[2];B[pd]", "(;SZ[19]HD[2]#;B[pd])"):
            with self.subTest(s):
                st, new, _d = rp.classify_game(s)
                self.assertIn(st, ("parse_error", "root_unparsed", "check_mismatch"))
                self.assertIsNone(new)

    def test_lowercase_root_not_edited(self):
        self.assertEqual(rp.classify_game("(;Sz[19]HD[2];B[pd])")[:2], ("root_unparsed", None))

    def test_unicode_and_multi_values(self):
        st, new, _d = rp.classify_game("(;SZ[19]PB[張三]PW[]AP[a][b]HD[4];B[pd];W[dp];B[pp])")
        self.assertEqual((st, new), ("root_props", PLAIN))

    def test_idempotent(self):
        _st, new, _d = rp.classify_game("(;CA[UTF-8]SZ[19]HD[2]RE[W+R];B[pd];W[dp])")
        self.assertEqual(rp.classify_game(new)[0], "plain")


class FileEdit(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="rp_unit_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_candidates_bom_crlf_only_root_bytes(self):
        rows = [["pub_g_%d" % k, "pub_p_1", "(;SZ[19]HD[%d];B[pd];W[dp])" % (k % 3), "B"] for k in range(6)]
        p = os.path.join(self.d, "player_identification_candidates_public.csv")
        old = write_csv(p, tc.C_COLS, rows, bom=True, eol="\r\n")
        rec, new = rp.analyse_file(p, "candidates")
        self.assertEqual(rec["counts"]["root_props"], 6)
        self.assertEqual(rec["hd_values"], {"0": 2, "1": 2, "2": 2})
        self.assertEqual(new, old.replace(b"HD[0]", b"").replace(b"HD[1]", b"").replace(b"HD[2]", b""))
        self.assertTrue(new.startswith(tp.BOM))
        self.assertEqual(rec["plain_form_after"], 6)

    def test_escaped_quote_cell_not_edited(self):
        rows = [["pub_g_1", "pub_p_1", '(;SZ[19]HD[1]C[say "hi"];B[pd];W[dp])', "B"],
                ["pub_g_2", "pub_p_1", "(;SZ[19]HD[1];B[pd];W[dp])", "B"]]
        p = os.path.join(self.d, "player_identification_candidates_public.csv")
        write_csv(p, tc.C_COLS, rows)
        rec, new = rp.analyse_file(p, "candidates")
        self.assertEqual((rec["counts"]["unusual_cell"], rec["counts"]["root_props"]), (1, 1))
        self.assertIn(b'HD[1]C[say ""hi""]', new)


def write_csv(path, cols, rows, bom=False, eol="\n"):
    import csv
    import io
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator=eol)
    w.writerow(cols)
    w.writerows(rows)
    b = (tp.BOM if bom else b"") + buf.getvalue().encode("utf-8")
    with open(path, "wb") as f:
        f.write(b)
    return b


def add_hd(src, dst, salt):
    os.makedirs(dst, exist_ok=True)
    for n in tp.ALL_NAMES:
        b = open(os.path.join(src, n), "rb").read().decode("utf-8")
        parts = b.split("(;SZ[19]")
        out = parts[0]
        for k, rest in enumerate(parts[1:]):
            u = int(hashlib.sha256(("%s|%s|%d" % (salt, n, k)).encode()).hexdigest()[:8], 16) % 10
            out += "(;SZ[19]" + ("HD[%d]" % (u % 3) if u < 7 else "HD[%d]CA[UTF-8]RE[W+R]" % (2 + u)) + rest
        open(os.path.join(dst, n), "wb").write(out.encode("utf-8"))


def run_tool(mod, *argv):
    try:
        return mod.main(list(argv))
    except SystemExit as e:
        return e


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.mkdtemp(prefix="rp_e2e_")
        cls.clean = os.path.join(cls.base, "clean")
        synthetic_tree(cls.clean)
        cls.clean_sha = {n: sha(open(os.path.join(cls.clean, n), "rb").read()) for n in tp.ALL_NAMES}

    @classmethod
    def tearDownClass(cls):
        for r, _d, fs in os.walk(cls.base):
            for f in fs:
                os.chmod(os.path.join(r, f), 0o666)
        shutil.rmtree(cls.base, ignore_errors=True)

    def tree(self, name, src):
        root = os.path.join(self.base, name)
        ingest(root, src)
        raw = os.path.join(root, "data", "test", "raw")
        return root, raw, {c: write_t1(root, c, raw) for c in ("rank", "player")}

    def test_clean_not_needed(self):
        root, _raw, v = self.tree("c", self.clean)
        self.assertEqual(run_tool(rp, "prepare", "--root", root), 2)
        self.assertFalse(os.path.exists(os.path.join(root, "data", "test_rootprops")))

    def test_hd_applied_equals_clean(self):
        dl = os.path.join(self.base, "h_dl")
        add_hd(self.clean, dl, "h")
        root, raw, v = self.tree("h", dl)
        self.assertEqual(v, {"rank": "PASS", "player": "PASS"})
        self.assertEqual(run_tool(rp, "prepare", "--root", root), 0)
        norm = os.path.join(root, "data", "test_rootprops", "normalised")
        for n in tp.ALL_NAMES:
            self.assertEqual(sha(open(os.path.join(norm, n), "rb").read()), self.clean_sha[n], n)
        self.assertEqual(run_tool(rp, "verify", "--root", root), 1)          # not ingested yet
        ingest(root, norm, "rev 11 test")
        for c in ("rank", "player"):
            write_t1(root, c, raw, "rev 11 test")
        self.assertEqual(run_tool(rp, "verify", "--root", root, "--rerun", "--reason", "after"), 0)
        self.assertIsInstance(run_tool(rp, "prepare", "--root", root, "--rerun", "--reason", "again"), SystemExit)

    def test_fail_opening_needs_rev9_first_then_both(self):
        dl_h = os.path.join(self.base, "hp_h")
        add_hd(self.clean, dl_h, "hp")
        dl = os.path.join(self.base, "hp_dl")
        inject(dl_h, dl, 0.3, 0.0, ("rank", "player"), "hp")
        root, raw, v = self.tree("hp", dl)
        self.assertEqual(v, {"rank": "FAIL_OPENING", "player": "FAIL_OPENING"})
        self.assertIsInstance(run_tool(rp, "prepare", "--root", root), SystemExit)
        self.assertEqual(run_tool(tp, "prepare", "--root", root), 0)
        strip = os.path.join(root, "data", "test_pass", "stripped")
        ingest(root, strip, "rev 9 test")
        for c in ("rank", "player"):
            write_t1(root, c, raw, "rev 9 test")
        self.assertIsInstance(run_tool(rp, "prepare", "--root", root), SystemExit)   # rev 9 verify missing
        self.assertEqual(run_tool(tp, "verify", "--root", root), 0)
        self.assertEqual(run_tool(rp, "prepare", "--root", root), 0)
        norm = os.path.join(root, "data", "test_rootprops", "normalised")
        for n in tp.ALL_NAMES:
            self.assertEqual(sha(open(os.path.join(norm, n), "rb").read()), self.clean_sha[n], n)
        ingest(root, norm, "rev 11 test")
        for c in ("rank", "player"):
            write_t1(root, c, raw, "rev 11 test")
        self.assertEqual(run_tool(rp, "verify", "--root", root), 0)

    def test_fail_opening_rev9_not_applicable_then_rev11(self):
        dl_h = os.path.join(self.base, "mo_h")
        add_hd(self.clean, dl_h, "mo")
        dl = os.path.join(self.base, "mo_dl")
        inject(dl_h, dl, 0.0, 0.03, ("rank",), "mo")
        root, raw, v = self.tree("mo", dl)
        self.assertEqual(v["rank"], "FAIL_OPENING")
        self.assertEqual(run_tool(tp, "prepare", "--root", root), 2)
        self.assertEqual(run_tool(rp, "prepare", "--root", root), 0)
        ingest(root, os.path.join(root, "data", "test_rootprops", "normalised"), "rev 11 test")
        for c in ("rank", "player"):
            self.assertEqual(write_t1(root, c, raw, "rev 11 test"), v[c])
        self.assertEqual(run_tool(rp, "verify", "--root", root), 0)
        open(os.path.join(root, "data", "test", "MANIFEST.json"), "w").write("{}")
        self.assertEqual(run_tool(rp, "verify", "--root", root, "--rerun", "--reason", "built"), 1)

    def test_one_comp_fail_struct_exit_3(self):
        import json
        dl = os.path.join(self.base, "fs_dl")
        add_hd(self.clean, dl, "fs")
        root, _raw, _v = self.tree("fs", dl)
        t1p = os.path.join(root, "results", "test02", "checks", "player", "t1.json")
        d = tc.read_json(t1p)
        d["verdict"] = "FAIL_STRUCT"
        with open(t1p, "w", encoding="utf-8") as f:
            json.dump(d, f)
        self.assertEqual(run_tool(rp, "prepare", "--root", root), 3)
        self.assertIsInstance(run_tool(rp, "verify", "--root", root), SystemExit)

    def test_build_present_refused(self):
        dl = os.path.join(self.base, "b_dl")
        add_hd(self.clean, dl, "b")
        root, _raw, _v = self.tree("b", dl)
        open(os.path.join(root, "data", "test", "MANIFEST.json"), "w").write("{}")
        self.assertIsInstance(run_tool(rp, "prepare", "--root", root), SystemExit)

    def test_root_inside_repo_refused(self):
        self.assertIsInstance(run_tool(rp, "prepare", "--root", os.path.join(REPO, "results")), SystemExit)


if __name__ == "__main__":
    unittest.main()
