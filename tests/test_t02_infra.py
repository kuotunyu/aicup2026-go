"""Tests for the test-day infrastructure: tools/t02_test_bundle.py, tools/t02_fake_tests.py and
tools/official_log.py (results/test02/design/spec.md 3.1, 3.5, 5.1, 6).

Synthetic data only, at official size where the tools demand it (tiny SGF strings); nothing here reads a
competition file, and nothing is written inside the repository. The real from-val / from-train runs belong
to the integrator's tests/test_t02_dryrun.py. Run: python -m pytest tests/test_t02_infra.py
"""

import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import official_log as ol  # noqa: E402
import t02_fake_tests as ft  # noqa: E402
import t02_test_bundle as tb  # noqa: E402

QH = tb.QUESTION_COLUMNS
CH = tb.CANDIDATE_COLUMNS
SGF = "(;SZ[19];B[dd];W[pp])"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_csv(path, header, rows, term="\n", bom=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator=term)
    w.writerow(header)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    with open(path, "wb") as f:
        f.write((b"\xef\xbb\xbf" if bom else b"") + data)
    return path


def q_rows(prefix, n=400, colour="B"):
    rows = []
    for i in range(1, n + 1):
        k = 5 + i % 16
        rows.append([f"{prefix}q_{i:04d}", k] + [SGF] * k + [""] * (20 - k) + [colour] * k + [""] * (20 - k))
    return rows


def c_rows(prefix, players=400, games=100):
    return [[f"{prefix}g_{i:06d}", f"{prefix}p_{i // games:04d}", SGF, "BW"[i % 2]] for i in range(players * games)]


def write_official(d):
    """The six official files, clean (LF, no BOM, README column order)."""
    for pool, pre in (("public", "pub_"), ("private", "priv_")):
        write_csv(os.path.join(d, tb.RANK_FILES[pool]), QH, q_rows(pre))
        write_csv(os.path.join(d, tb.PLAYER_Q_FILES[pool]), QH, q_rows(pre))
        write_csv(os.path.join(d, tb.PLAYER_C_FILES[pool]), CH, c_rows(pre))
    return d


def write_t1(checks, comp, tests_dir, verdict="PASS", normalize=None, shas=None):
    d = os.path.join(checks, comp)
    os.makedirs(d, exist_ok=True)
    files = shas if shas is not None else {n: {"sha256": sha(os.path.join(tests_dir, n))}
                                           for n in tb.COMP_FILES[comp]}
    obj = {"verdict": verdict, "normalize": normalize or [], "inputs": files}
    with open(os.path.join(d, "t1.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f)


def fake_recheck(comp, tests_dir, checks_dir, kind, reason=None):
    """Stand-in for tools/t02_checks.py t1 --only T1.1,T1.2,T1.3: structure check, then its json."""
    verdict = "PASS" if not tb.quick_structure(tests_dir, comp) else "FAIL_STRUCT"
    _out, path = tb.recheck_target(comp, checks_dir, kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "kind": kind}, f)
    return verdict, path


def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)


# ================================================================ t02_test_bundle
class Ingest(Tmp):
    def setUp(self):
        super().setUp()
        self.src = write_official(self.p("dl", "nested", "a"))

    def ingest(self, *extra, src=None):
        argv = ["ingest", "--root", self.p("S")] + sum((["--src", s] for s in (src or [self.p("dl")])), [])
        return quiet(tb.main, argv + list(extra))[0]

    def test_folder_and_zip(self):
        rec = self.ingest()
        raw = self.p("S", "data", "test", "raw")
        for n in tb.ALL_NAMES:
            self.assertEqual(sha(os.path.join(raw, n)), sha(os.path.join(self.src, n)))
            self.assertEqual(rec["files"][n]["sha256"], sha(os.path.join(self.src, n)))
        self.assertTrue(os.path.exists(os.path.join(raw, tb.SOURCE_NAME)))
        # a zip holding the same files (inside a folder) and the same zip given directly
        z = self.p("z", "dl.zip")
        os.makedirs(self.p("z"))
        with zipfile.ZipFile(z, "w") as zf:
            for n in tb.ALL_NAMES:
                zf.write(os.path.join(self.src, n), f"tests/{n}")
            zf.writestr("__MACOSX/tests/._" + tb.ALL_NAMES[0], b"debris")
        rec = quiet(tb.main, ["ingest", "--root", self.p("S2"), "--src", z])[0]
        self.assertEqual(rec["files"][tb.ALL_NAMES[0]]["from"], [f"{z}!tests/{tb.ALL_NAMES[0]}"])

    def test_identical_duplicate_accepted_different_refused(self):
        shutil.copytree(self.src, self.p("dl", "copy"))
        rec = self.ingest()
        self.assertEqual(len(rec["files"][tb.ALL_NAMES[0]]["from"]), 2)
        with open(self.p("dl", "copy", tb.ALL_NAMES[0]), "ab") as f:
            f.write(b"x\n")
        with self.assertRaises(SystemExit) as cm:
            self.ingest("--rerun", "--reason", "t")
        self.assertIn("different contents", str(cm.exception))

    def test_missing_name_refused(self):
        os.remove(os.path.join(self.src, tb.RANK_FILES["private"]))
        with open(os.path.join(self.src, "Rank_Prediction_Test_Private.csv"), "w") as f:
            f.write("x\n")
        with self.assertRaises(SystemExit) as cm:
            self.ingest()
        self.assertIn("rank_prediction_test_private.csv", str(cm.exception))
        self.assertIn("similar names seen", str(cm.exception))
        self.assertFalse(os.path.exists(self.p("S", "data", "test", "raw")))

    def test_no_silent_overwrite(self):
        self.ingest()
        with self.assertRaises(SystemExit):
            self.ingest()
        with self.assertRaises(SystemExit):
            self.ingest("--rerun")
        rec = self.ingest("--rerun", "--reason", "second download")
        self.assertEqual(rec["rerun_reason"], "second download")

    def test_own_bundles_refused_by_the_real_ingest(self):
        """Review finding (realmode, minor): a rehearsal / smoke / fake tree (built from validation games) or a
        re-packed bundle is never ingested as the official files; a scratch ingest only notes it."""
        man = {"format": tb.MANIFEST_FORMAT, "kind": "rehearsal"}
        tree = self.p("fake")
        write_official(os.path.join(tree, "data", "test"))
        with open(os.path.join(tree, "data", "test", tb.MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(man, f)
        z = self.p("zips", "C10_test_v1.zip")
        os.makedirs(os.path.dirname(z))
        with zipfile.ZipFile(z, "w") as zf:
            for n in tb.ALL_NAMES:
                zf.write(os.path.join(tree, "data", "test", n), f"tests/{n}")
            zf.writestr("tests/MANIFEST.json", json.dumps(dict(man, kind="smoke")))
        fake_tree = self.p("ft")
        write_official(fake_tree)
        with open(os.path.join(fake_tree, "FAKE_TREE.json"), "w", encoding="utf-8") as f:
            json.dump({"format": "c10-test02 fake tree v1", "kind": "from-val"}, f)
        with mock.patch.object(tb, "REPO", self.p("repo")):
            for src, kind in ((os.path.join(tree, "data", "test"), "kind rehearsal"), (z, "kind smoke"),
                              (fake_tree, "fake tree")):
                with self.subTest(src=src):
                    with self.assertRaises(SystemExit) as cm:
                        quiet(tb.main, ["ingest", "--src", src])
                    self.assertIn(kind, str(cm.exception))
                    self.assertFalse(os.path.exists(self.p("repo", "data", "test", "raw")))
            rec = quiet(tb.main, ["ingest", "--src", self.p("dl")])[0]           # the download itself is fine
            self.assertEqual(rec["own_records_in_sources"], [])
        rec, out = quiet(tb.main, ["ingest", "--root", self.p("S3"), "--src", z])   # scratch: noted only
        self.assertIn("note:", out)
        self.assertEqual(len(rec["own_records_in_sources"]), 1)
        with open(self.p("dl", "MANIFEST.json"), "w", encoding="utf-8") as f:       # someone else's MANIFEST.json
            json.dump({"format": "something else"}, f)
        rec = self.ingest("--rerun", "--reason", "x")
        self.assertEqual(rec["own_records_in_sources"], [])

    def test_dest_inside_repo_outside_data_test_refused(self):
        with self.assertRaises(SystemExit):
            quiet(tb.main, ["ingest", "--src", self.p("dl"), "--dest", os.path.join(REPO, "results", "x")])
        with self.assertRaises(SystemExit):
            quiet(tb.main, ["ingest", "--src", self.p("dl"), "--root", REPO])


class Build(Tmp):
    def setUp(self):
        super().setUp()
        self.S = self.p("S")
        self.raw = write_official(os.path.join(self.S, "data", "test", "raw"))
        self.checks = os.path.join(self.S, "results", "test02", "checks")
        self.patch = mock.patch.object(tb, "run_t1_recheck", side_effect=fake_recheck)
        self.recheck = self.patch.start()

    def tearDown(self):
        self.patch.stop()
        super().tearDown()

    def build(self, *extra, kind="smoke"):
        return quiet(tb.main, ["build", "--kind", kind, "--root", self.S] + list(extra))[0]

    def t1_all(self, **kw):
        for comp in ("rank", "player"):
            write_t1(self.checks, comp, self.raw, **kw)

    def test_pass_builds_byte_copies_manifest_and_zip(self):
        self.t1_all()
        rec = self.build()
        out = os.path.join(self.S, "data", "test")
        for n in tb.ALL_NAMES:
            self.assertEqual(sha(os.path.join(out, n)), sha(os.path.join(self.raw, n)))
        man = json.load(open(os.path.join(out, tb.MANIFEST_NAME), encoding="utf-8"))
        self.assertEqual(man["format"], "c10-test02 manifest v1")
        self.assertEqual(man["kind"], "smoke")
        self.assertEqual(man["t1"], {"rank": "PASS", "player": "PASS"})
        self.assertEqual(set(man["files"]), set(tb.ALL_NAMES))
        for n, r in man["files"].items():
            self.assertEqual(set(r) >= {"sha256", "bytes", "rows", "source_sha256", "normalized"}, True)
            self.assertEqual(r["source_sha256"], r["sha256"])
            self.assertEqual(r["normalized"], [])
        self.assertEqual(man["files"][tb.PLAYER_C_FILES["public"]]["rows"], 40000)
        z = os.path.join(self.S, "data", "colab_bundle", tb.ZIP_NAME)
        self.assertEqual(rec["zip_sha256"], sha(z))
        with zipfile.ZipFile(z) as zf:
            names = sorted(zf.namelist())
            self.assertEqual(names, sorted(f"tests/{n}" for n in list(tb.ALL_NAMES) + [tb.MANIFEST_NAME]))
            self.assertEqual(hashlib.sha256(zf.read("tests/MANIFEST.json")).hexdigest(), rec["manifest_sha256"])
        self.assertEqual(self.recheck.call_count, 2)
        self.assertEqual({c.args[3] for c in self.recheck.call_args_list}, {"smoke"})
        self.assertTrue(os.path.exists(os.path.join(self.checks, "built", "rank", "t1.json")))
        self.assertTrue(os.path.exists(os.path.join(self.checks, "bundle_build.json")))
        # same inputs -> same zip bytes apart from the MANIFEST time; outputs are never replaced silently
        with self.assertRaises(SystemExit):
            self.build()
        rec2 = self.build("--rerun", "--reason", "again")
        self.assertEqual(rec2["rerun_reason"], "again")

    def test_normalized_copies(self):
        # raw rank public: BOM, CRLF, lower-case colours, whitespace-only slots after num_games, columns shuffled
        clean = q_rows("pub_")
        dirty = [list(r) for r in clean]
        for r in dirty[:10]:
            r[2 + 19] = " "                                        # sgf_20 (num_games <= 20; k < 20 here)
            r[22] = r[22].lower()                                  # color_1 'B' -> 'b'
        for r in dirty[10:20]:                                     # rev 8: colour letters after num_games
            k = r[1]
            for i in range(k + 1, 21):
                r[21 + i] = "B" if i % 3 else "b"
            if k < 20:
                r[1 + 20] = "  "                                   # with a whitespace-only sgf_20
        order = ["question_id", "num_games"] + [f"color_{i}" for i in range(1, 21)] + [f"sgf_{i}" for i in range(1, 21)]
        idx = [QH.index(c) for c in order]
        name = tb.RANK_FILES["public"]
        write_csv(os.path.join(self.raw, name), order, [[r[i] for i in idx] for r in dirty], term="\r\n", bom=True)
        cands = c_rows("priv_")
        for r in cands[:5]:
            r[3] = r[3].lower()
        cname = tb.PLAYER_C_FILES["private"]
        write_csv(os.path.join(self.raw, cname), CH, cands, term="\r\n")
        raw_sha = {n: sha(os.path.join(self.raw, n)) for n in (name, cname)}
        write_t1(self.checks, "rank", self.raw, verdict="NORMALIZED", normalize=[
            {"transform": t, "file": name} for t in ("strip_bom", "crlf_to_lf", "upper_colour",
                                                     "reorder_columns_to_readme",
                                                     "clear_slots_after_num_games_if_empty_whitespace",
                                                     "clear_colour_after_num_games_if_sgf_empty")])
        write_t1(self.checks, "player", self.raw, verdict="NORMALIZED", normalize=["crlf_to_lf", "upper_colour"])
        self.build()
        out = os.path.join(self.S, "data", "test")
        want = write_csv(self.p("want", name), QH, clean)
        self.assertEqual(sha(os.path.join(out, name)), sha(want))
        want_c = write_csv(self.p("want", cname), CH, c_rows("priv_"))
        self.assertEqual(sha(os.path.join(out, cname)), sha(want_c))
        man = json.load(open(os.path.join(out, tb.MANIFEST_NAME), encoding="utf-8"))
        self.assertEqual(man["files"][name]["normalized"], list(tb.NORMALIZE_WHITELIST))
        self.assertEqual(man["files"][name]["source_sha256"], raw_sha[name])
        self.assertEqual(man["files"][cname]["normalized"], ["crlf_to_lf", "upper_colour"])
        self.assertEqual(man["files"][tb.PLAYER_C_FILES["public"]]["normalized"], [])   # already clean: no-op
        self.assertEqual(man["t1"], {"rank": "NORMALIZED", "player": "NORMALIZED"})

    def test_refusals(self):
        cases = [
            (dict(verdict="FAIL_STRUCT"), "FAIL_STRUCT"),
            (dict(verdict="FAIL_OPENING"), "--allow-fail-opening"),
            (dict(verdict="PASS", normalize=["strip_bom"]), "inconsistent"),
            (dict(verdict="NORMALIZED"), "no normalize transforms"),
            (dict(verdict="NORMALIZED", normalize=["fix_moves"]), "whitelist"),
            (dict(shas={}), "does not record the SHA-256"),
            (dict(shas={n: {"sha256": "0" * 64} for n in tb.COMP_FILES["rank"]}), "another"),
        ]
        write_t1(self.checks, "player", self.raw)
        for kw, msg in cases:
            write_t1(self.checks, "rank", self.raw, **kw)
            with self.assertRaises(SystemExit) as cm:
                self.build()
            self.assertIn(msg, str(cm.exception), kw)
            self.assertFalse(os.path.exists(os.path.join(self.S, "data", "test", tb.MANIFEST_NAME)), kw)
        os.remove(os.path.join(self.checks, "rank", "t1.json"))
        with self.assertRaises(SystemExit) as cm:
            self.build()
        self.assertIn("is missing", str(cm.exception))

    def test_fail_opening_with_flag(self):
        write_t1(self.checks, "rank", self.raw, verdict="FAIL_OPENING")
        write_t1(self.checks, "player", self.raw)
        with self.assertRaises(SystemExit):
            self.build("--allow-fail-opening")                    # needs a reason
        rec = self.build("--allow-fail-opening", "--reason", "T1.5 share 0.47")
        man = json.load(open(os.path.join(self.S, "data", "test", tb.MANIFEST_NAME), encoding="utf-8"))
        self.assertEqual(man["t1"]["rank"], "FAIL_OPENING")
        self.assertEqual(man["fail_opening_reason"], "T1.5 share 0.47")
        self.assertEqual(rec["t1_raw"]["rank"]["verdict"], "FAIL_OPENING")

    def test_recheck_must_pass(self):
        self.t1_all()
        self.recheck.side_effect = lambda comp, t, c, k, r=None: fake_recheck(comp, t, c, k) if comp == "rank" \
            else ("FAIL_STRUCT", fake_recheck(comp, t, c, k)[1])
        with self.assertRaises(SystemExit) as cm:
            self.build()
        self.assertIn("gave 'FAIL_STRUCT'", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.S, "data", "colab_bundle", tb.ZIP_NAME)))

    def test_mode_guards(self):
        self.t1_all()
        with self.assertRaises(SystemExit) as cm:
            self.build(kind="official")                            # the real run takes no --root
        self.assertIn("no --root", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            quiet(tb.main, ["build", "--kind", "official", "--tests-dir", self.raw])
        self.assertIn("needs the repo path", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            self.build("--out-dir", os.path.join(REPO, "data", "test_should_not_exist"))
        self.assertIn("inside the repository", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(REPO, "data", "test_should_not_exist")))
        with self.assertRaises(SystemExit):
            quiet(tb.main, ["build", "--kind", "smoke", "--root", REPO])

    def test_in_place_build(self):
        tests = write_official(self.p("T", "data", "test"))
        for comp in ("rank", "player"):
            write_t1(self.p("T", "results", "test02", "checks"), comp, tests)
        rec = quiet(tb.main, ["build", "--kind", "rehearsal", "--root", self.p("T"), "--tests-dir", tests])[0]
        self.assertTrue(rec["in_place"])
        tb.verify_manifest(tests)

    def test_normalize_by_file_preferred(self):
        name = tb.RANK_FILES["private"]
        write_csv(os.path.join(self.raw, name), QH, q_rows("priv_"), term="\r\n")
        write_t1(self.checks, "player", self.raw)
        path = os.path.join(self.checks, "rank", "t1.json")
        write_t1(self.checks, "rank", self.raw, verdict="NORMALIZED", normalize=["crlf_to_lf"])
        obj = json.load(open(path, encoding="utf-8"))
        obj["normalize_by_file"] = {tb.RANK_FILES["public"]: [], name: ["crlf_to_lf"]}
        json.dump(obj, open(path, "w", encoding="utf-8"))
        self.build()
        man = json.load(open(os.path.join(self.S, "data", "test", tb.MANIFEST_NAME), encoding="utf-8"))
        self.assertEqual(man["files"][name]["normalized"], ["crlf_to_lf"])
        obj["normalize_by_file"] = {name: ["strip_bom"]}
        json.dump(obj, open(path, "w", encoding="utf-8"))
        with self.assertRaises(SystemExit) as cm:
            self.build("--rerun", "--reason", "x")
        self.assertIn("disagree", str(cm.exception))


@unittest.skipUnless(os.path.exists(os.path.join(REPO, "tools", "t02_checks.py")), "tools/t02_checks.py missing")
class BuildWithRealChecks(Tmp):
    """No mock: raw T1 and the build re-check both run tools/t02_checks.py (--smoke, T1.1-T1.3 only)."""

    def t1(self, comp, *extra):
        import subprocess
        r = subprocess.run([sys.executable, "-B", os.path.join(REPO, "tools", "t02_checks.py"), "t1", "--comp", comp,
                            "--smoke", "--root", self.p("S"), "--tests-dir", self.p("S", "data", "test", "raw"),
                            "--only", "T1.1,T1.2,T1.3"] + list(extra), capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_build(self):
        write_official(self.p("S", "data", "test", "raw"))
        for comp in ("rank", "player"):
            self.t1(comp)
        rec = quiet(tb.main, ["build", "--kind", "smoke", "--root", self.p("S")])[0]
        self.assertEqual({c: rec["t1_built"][c]["verdict"] for c in ("rank", "player")},
                         {"rank": "PASS", "player": "PASS"})
        self.assertTrue(os.path.exists(self.p("S", "results", "test02", "checks", "built", "player", "t1.json")))
        rec2 = quiet(tb.main, ["build", "--kind", "smoke", "--root", self.p("S"), "--rerun", "--reason", "again"])[0]
        self.assertEqual(rec2["files"], rec["files"])

    def test_build_normalized(self):
        raw = write_official(self.p("S", "data", "test", "raw"))
        clean = q_rows("pub_")
        dirty = [list(r) for r in clean]
        for r in dirty[:10]:
            r[21] = "  "                                           # sgf_20 after num_games: whitespace only
            r[22] = "b"                                            # color_1 lower case
        for r in dirty[10:20]:                                     # rev 8: 'color_1..color_20 always B'
            for i in range(r[1] + 1, 21):
                r[21 + i] = "B"
        order = ["question_id", "num_games"] + [f"color_{i}" for i in range(1, 21)] + [f"sgf_{i}" for i in range(1, 21)]
        idx = [QH.index(c) for c in order]
        name = tb.RANK_FILES["public"]
        write_csv(os.path.join(raw, name), order, [[r[i] for i in idx] for r in dirty], term="\r\n", bom=True)
        cands = c_rows("priv_")
        for r in cands[:5]:
            r[3] = r[3].lower()
        cname = tb.PLAYER_C_FILES["private"]
        write_csv(os.path.join(raw, cname), CH, cands)
        import subprocess
        for comp in ("rank", "player"):
            r = subprocess.run([sys.executable, "-B", os.path.join(REPO, "tools", "t02_checks.py"), "t1", "--comp",
                                comp, "--smoke", "--root", self.p("S"), "--tests-dir", raw, "--only",
                                "T1.1,T1.2,T1.3"], capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        t1 = json.load(open(self.p("S", "results", "test02", "checks", "rank", "t1.json"), encoding="utf-8"))
        self.assertEqual(t1["verdict"], "NORMALIZED")
        rec = quiet(tb.main, ["build", "--kind", "smoke", "--root", self.p("S")])[0]
        self.assertEqual(rec["t1_built"]["rank"]["verdict"], "PASS")
        out = self.p("S", "data", "test")
        want = write_csv(self.p("want", name), QH, clean, term="\r\n")      # uniform CRLF is kept
        self.assertEqual(sha(os.path.join(out, name)), sha(want))
        self.assertEqual(sha(os.path.join(out, cname)), sha(write_csv(self.p("want", cname), CH, c_rows("priv_"))))
        self.assertEqual(rec["normalized"][name], ["strip_bom", "upper_colour", "reorder_columns_to_readme",
                                                   "clear_slots_after_num_games_if_empty_whitespace",
                                                   "clear_colour_after_num_games_if_sgf_empty"])
        self.assertEqual(rec["normalized"][cname], ["upper_colour"])
        self.assertEqual(t1["normalize"][name], rec["normalized"][name])       # what T1 asked for is what changed

    def test_official_recheck_uses_the_real_t1_pair(self):
        """build --kind official: the re-check command must pass t02_checks.py's real-mode path rule (REAL_T1:
        data/test -> results/test02/checks/built). No mock of t02_checks; nothing is run or written (review
        finding 2026-09-26: the old target results/test02/checks/<comp>/t1_built.json was always refused)."""
        import t02_checks as tc
        for comp in ("rank", "player"):
            out, target = tb.recheck_target(comp, tc.CHECKS_DIR, "official")
            self.assertTrue(tb.same_path(out, tc.CHECKS_BUILT), out)
            self.assertTrue(tb.same_path(target, os.path.join(tc.CHECKS_BUILT, comp, "t1.json")), target)
            self.assertIn((tc.DATA_TEST, tc.CHECKS_BUILT), tc.REAL_T1)
            seen = []

            def fake_run(cmd, **kw):
                seen.append(cmd)
                raise RuntimeError("stop before running")
            with mock.patch.object(tb.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(RuntimeError):
                    tb.run_t1_recheck(comp, tc.DATA_TEST, tc.CHECKS_DIR, "official")
            cmd = seen[0]
            self.assertEqual(os.path.basename(cmd[2]), "t02_checks.py")
            args = tc.build_parser().parse_args(cmd[3:])
            self.assertEqual(tc.mode_of(args), "real")
            tests, got_out = tc.resolve_paths(args, "real", tc.REAL_T1)       # raised SystemExit before the fix
            self.assertTrue(tb.same_path(tests, tc.DATA_TEST))
            self.assertTrue(tb.same_path(got_out, tc.CHECKS_BUILT))


class OfficialBuildNeedsFullT1(Tmp):
    """Review finding (realmode, major): --kind official accepts only the full T1 (T1.1-T1.5) on the raw files in
    real mode; the other kinds keep accepting the structure-only T1 of the validation-derived trees."""

    def test_load_t1(self):
        raw = write_official(self.p("raw"))
        checks = self.p("checks")
        full = {"complete": True, "stage": "raw", "provenance": {"mode": "real"}}
        for extra, ok in ((full, True), (dict(full, complete=False), False), (dict(full, stage="built"), False),
                          (dict(full, provenance={"mode": "dry-run"}), False), ({}, False)):
            with self.subTest(extra=extra):
                write_t1(checks, "rank", raw)
                p = os.path.join(checks, "rank", "t1.json")
                obj = json.load(open(p, encoding="utf-8"))
                obj.update(extra)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(obj, f)
                self.assertEqual(tb.load_t1(checks, "rank", raw, False)["verdict"], "PASS")      # smoke / rehearsal
                if ok:
                    self.assertEqual(tb.load_t1(checks, "rank", raw, False, require_full=True)["verdict"], "PASS")
                else:
                    with self.assertRaises(SystemExit) as cm:
                        tb.load_t1(checks, "rank", raw, False, require_full=True)
                    self.assertIn("--kind official needs the full T1", str(cm.exception))


class NormalizeUnits(unittest.TestCase):
    def test_parse_shapes(self):
        n0, n1 = tb.COMP_FILES["rank"]
        self.assertEqual(tb.parse_normalize(["strip_bom"], "rank"), {n0: ["strip_bom"], n1: ["strip_bom"]})
        self.assertEqual(tb.parse_normalize({n0: ["crlf_to_lf"]}, "rank"), {n0: ["crlf_to_lf"], n1: []})
        self.assertEqual(tb.parse_normalize({"upper_colour": [f"data/test/raw/{n1}"]}, "rank"),
                         {n0: [], n1: ["upper_colour"]})
        self.assertEqual(tb.parse_normalize([{"name": "strip_bom", "files": [n1]}], "rank"),
                         {n0: [], n1: ["strip_bom"]})
        with self.assertRaises(SystemExit):
            tb.parse_normalize({"player_identification_test_public.csv": ["strip_bom"]}, "rank")

    def test_recorded_shas(self):
        n0 = tb.COMP_FILES["rank"][0]
        h = "a" * 64
        self.assertEqual(dict(tb.recorded_input_shas({"x": [{"path": f"data/test/raw/{n0}", "sha256": h}]},
                                                      tb.COMP_FILES["rank"])), {n0: {h}})
        self.assertEqual(dict(tb.recorded_input_shas({"files": {n0: h}}, tb.COMP_FILES["rank"])), {n0: {h}})

    def test_clear_colour_after_num_games(self):
        """PREREG_02 rev 8: a colour letter after num_games is cleared only when that slot's sgf is empty; an sgf
        after num_games is never touched (the re-check of the built copy rejects it)."""
        name = tb.RANK_FILES["public"]
        rows = q_rows("pub_", n=3)
        rows[0][21 + 20] = "B"                                     # k = 6: color_20 filled, sgf_20 empty
        rows[1][21 + 19] = "w"                                     # k = 7: color_19 lower case
        rows[2][1 + 20], rows[2][21 + 20] = SGF, "B"               # k = 8: an sgf after num_games stays
        raw = io.StringIO(newline="")
        csv.writer(raw, lineterminator="\n").writerows([QH] + rows)
        data, changed = tb.normalize_bytes(name, raw.getvalue().encode(), ["clear_colour_after_num_games_if_sgf_empty"])
        self.assertEqual(changed, ["clear_colour_after_num_games_if_sgf_empty"])
        got = list(csv.reader(io.StringIO(data.decode(), newline="")))
        self.assertEqual((got[1][21 + 20], got[2][21 + 19]), ("", ""))
        self.assertEqual((got[3][1 + 20], got[3][21 + 20]), (SGF, "B"))
        self.assertEqual([g[:21 + 19] for g in got[1:3]], [[str(x) for x in r[:21 + 19]] for r in rows[:2]])
        self.assertIn("clear_colour_after_num_games_if_sgf_empty", tb.NORMALIZE_WHITELIST)
        clean = io.StringIO(newline="")
        csv.writer(clean, lineterminator="\n").writerows([QH] + q_rows("pub_", n=3))
        self.assertEqual(tb.normalize_bytes(name, clean.getvalue().encode(),
                                            ["clear_colour_after_num_games_if_sgf_empty"])[1], [])

    def test_quoting_that_does_not_round_trip_is_refused(self):
        name = tb.RANK_FILES["public"]
        raw = ("question_id,num_games\r\n" + '"pub_q_0001",5\r\n').encode()
        with self.assertRaises(SystemExit) as cm:
            tb.normalize_bytes(name, raw, ["crlf_to_lf", "upper_colour"])
        self.assertIn("does not give back the same bytes", str(cm.exception))
        self.assertEqual(tb.normalize_bytes(name, raw, ["crlf_to_lf"]),
                         (raw.replace(b"\r\n", b"\n"), ["crlf_to_lf"]))


class Unpack(Tmp):
    def setUp(self):
        super().setUp()
        self.tests = write_official(self.p("S", "data", "test"))
        files = {n: tb.file_record(os.path.join(self.tests, n)) for n in tb.ALL_NAMES}
        mpath, _ = tb.write_manifest(self.tests, "rehearsal", files, {"rank": "PASS", "player": "PASS"})
        self.man_sha = sha(mpath)
        self.files = {n: files[n]["sha256"] for n in tb.ALL_NAMES}

    def make_zip(self, comp="rank", status="completed", members=None, config=None, name=None, runs=None):
        cfg = {"smoke": False, "rehearsal": True,
               "test_bundle": {"zip_sha256": "0" * 64, "manifest_sha256": self.man_sha, "files": self.files}}
        cfg.update(config or {})
        path = self.p(name or f"test02_{comp}_results.zip")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{comp}/config.json", json.dumps(cfg))
            zf.writestr(f"{comp}/run_log.json", json.dumps({"runs": runs or [{"run": 1, "status": status}]}))
            zf.writestr(f"{comp}/seam_val/dumps/_INPUT.json", "{}")
            for m, data in (members or {}).items():
                zf.writestr(m, data)
        return path

    def unpack(self, zpath, comp="rank", *extra, results=None):
        return quiet(tb.main, ["unpack", "--zip", zpath, "--comp", comp, "--root", self.p("S"),
                               "--results-dir", results or self.p("R")] + list(extra))[0]

    def test_ok(self):
        rec = self.unpack(self.make_zip())
        self.assertEqual(rec["members"], 3)
        self.assertTrue(os.path.exists(self.p("R", "rank", "seam_val", "dumps", "_INPUT.json")))
        self.assertEqual(json.load(open(self.p("R", "unpack_rank.json"), encoding="utf-8"))["run_status"],
                         "completed")
        with self.assertRaises(SystemExit) as cm:                  # never over existing files
            self.unpack(self.make_zip(name="again.zip"))
        self.assertIn("already exist", str(cm.exception))

    def test_unsafe_members(self):
        # (zipfile turns backslashes into '/' on Windows, so 'rank\..\..' arrives as 'rank/../..')
        for bad in ("../evil.txt", "rank/../../evil.txt", "/abs.txt", "rank\\..\\..\\evil.txt", "player/x.txt",
                    "rank/C:x", "rank/x_" + "ans" + "wers.csv"):
            with self.assertRaises(SystemExit, msg=bad):
                self.unpack(self.make_zip(members={bad: "x"}, name="bad.zip"))
            self.assertFalse(os.path.exists(self.p("R")), bad)
        self.assertFalse(os.path.exists(self.p("evil.txt")))

    def test_failed_run(self):
        z = self.make_zip(status="seam_failed")
        with self.assertRaises(SystemExit) as cm:
            self.unpack(z)
        self.assertIn("seam_failed", str(cm.exception))
        rec = self.unpack(z, "rank", "--allow-failed", results=self.p("inspect"))
        self.assertEqual(rec["run_status"], "seam_failed")
        with self.assertRaises(SystemExit):                        # inspection never inside the repo
            self.unpack(z, "rank", "--allow-failed", results=os.path.join(REPO, "results", "test02"))

    def test_earlier_failed_run_in_the_same_folder(self):
        """Review finding (userdocs, major): a completed run after a failed one in the same run_log is a rerun in the
        same Drive folder, not the registered rerun (PREREG_02 9.2, 10.10). The real results dir refuses it; a
        scratch unpack warns; every unpack records the run history."""
        runs = [{"run": 1, "status": "repro_failed"}, {"run": 2, "status": "completed"}]
        rec, out = quiet(tb.main, ["unpack", "--zip", self.make_zip(runs=runs, name="a.zip"), "--comp", "rank",
                                   "--root", self.p("S"), "--results-dir", self.p("R")])
        self.assertEqual(rec["earlier_failed_runs"], [{"run": 1, "status": "repro_failed"}])
        self.assertEqual([r["status"] for r in rec["runs"]], ["repro_failed", "completed"])
        self.assertIn("WARNING", out)
        self.assertIn("1 repro_failed, 2 completed", out)
        # the real results dir (the repository is a temporary one here)
        real_cfg = {"rehearsal": False}
        with mock.patch.object(tb, "REPO", self.p("S")):
            argv = ["unpack", "--comp", "rank"]
            for bad in ("seam_failed", "readonly_changed", "final_check_failed", "repro_failed"):
                z = self.make_zip(runs=[{"run": 1, "status": bad}, {"run": 2, "status": "completed"}],
                                  config=real_cfg, name=f"{bad}.zip")
                with self.assertRaises(SystemExit) as cm:
                    quiet(tb.main, argv + ["--zip", z])
                self.assertIn("earlier failed runs", str(cm.exception))
                self.assertFalse(os.path.exists(self.p("S", "results", "test02", "rank")), bad)
            ok_runs = [{"run": 1, "status": "preflight_failed"}, {"run": 2, "status": "interrupted"},
                       {"run": 3, "status": "completed"}]
            rec = quiet(tb.main, argv + ["--zip", self.make_zip(runs=ok_runs, config=real_cfg, name="ok.zip")])[0]
        self.assertEqual(rec["earlier_failed_runs"], [])
        self.assertEqual(len(rec["runs"]), 3)
        self.assertEqual(tb.earlier_failures({"runs": [{"status": "seam_failed"}]}), [])      # the last run itself

    def test_manifest_and_mode_checks(self):
        with self.assertRaises(SystemExit) as cm:
            self.unpack(self.make_zip(config={"test_bundle": {"manifest_sha256": "1" * 64, "files": self.files}}))
        self.assertIn("manifest_sha256", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:                  # the real dir needs the real data/test
            self.unpack(self.make_zip(), results=os.path.join(REPO, "results", "test02"))
        self.assertIn("real tests dir", str(cm.exception))
        for ro in (os.path.join(REPO, "results", "prereg02"), os.path.join(REPO, "results", "other")):
            with self.assertRaises(SystemExit):
                self.unpack(self.make_zip(), results=ro)
        self.assertEqual(tb.run_status({"status": "completed", "runs": [{"status": "seam_failed"}]}), "completed")


# ================================================================ t02_fake_tests from-val (synthetic inputs)
def make_fake_inputs(base):
    """A synthetic data/local + results/prereg02 layout with the real file shapes (tiny content)."""
    local, p02 = os.path.join(base, "local"), os.path.join(base, "prereg02")
    rng = np.random.default_rng(0)
    # rank: 810 questions, CRLF like split v1
    rows, keys = [], {"question_id": [], "slot": [], "color": [], "parsed": []}
    for i in range(1, 811):
        k = 5 + i % 16
        cols = ["B" if (i + j) % 2 else "W" for j in range(k)]
        rows.append([f"val_q_{i:04d}", k] + [SGF] * k + [""] * (20 - k) + cols + [""] * (20 - k))
        for j in range(k):
            keys["question_id"].append(f"val_q_{i:04d}")
            keys["slot"].append(j + 1)
            keys["color"].append(cols[j])
            keys["parsed"].append(bool((i + j) % 7))
    write_csv(os.path.join(local, ft.RANK_VAL), QH, rows, term="\r\n")
    n = len(keys["slot"])
    d = os.path.join(p02, "rank", "dumps")
    os.makedirs(d)
    kz = {"question_id": np.array(keys["question_id"]), "slot": np.array(keys["slot"], dtype=np.int32),
          "color": np.array(keys["color"]), "parsed": np.array(keys["parsed"])}
    lg = rng.standard_normal((n, 8, 10)).astype(np.float32)
    lg[~kz["parsed"]] = np.nan
    np.savez(os.path.join(d, "val_keys.npz"), **kz)
    np.savez(os.path.join(d, "val_p02e9_d4.npz"), logits=lg, weights_sha256=np.array("f" * 64), syms=np.array(8))
    for stem in ("val_keys", "val_p02e9_d4"):
        with open(os.path.join(d, f"_DONE_{stem}.json"), "w", encoding="utf-8") as f:
            json.dump({"file": f"dumps/{stem}.npz", "rows": n, "sha256": sha(os.path.join(d, stem + ".npz")),
                       "weights_sha256": "f" * 64 if stem != "val_keys" else None, "gpu_name": "GPU X",
                       "torch": "t", "cudnn": 1}, f)
    # player: 400 questions, 400 candidates x 100 games
    q, qk = [], {"question_id": [], "slot": [], "color": []}
    for i in range(1, 401):
        k = 5 + i % 16
        q.append([f"val_pq_{i:04d}", k] + [SGF] * k + [""] * (20 - k) + ["W"] * k + [""] * (20 - k))
        for j in range(k):
            qk["question_id"].append(f"val_pq_{i:04d}")
            qk["slot"].append(j + 1)
            qk["color"].append("W")
    write_csv(os.path.join(local, ft.PLAYER_VAL_Q), QH, q, term="\r\n")
    pids = [f"val_p_{hashlib.sha256(str(i).encode()).hexdigest()[:16]}" for i in range(400)]
    c = [[f"val_g_{g + 1:06d}", pids[g // 100], SGF, "BW"[g % 2]] for g in range(40000)]
    write_csv(os.path.join(local, ft.PLAYER_VAL_C), CH, c, term="\r\n")
    d = os.path.join(p02, "player", "dumps")
    os.makedirs(d)
    np.savez(os.path.join(d, "cand_games.npz"), game_id=np.array([r[0] for r in c]),
             player_id=np.array([r[1] for r in c]), color=np.array([r[3] for r in c]),
             emb_sum=rng.standard_normal((40000, 4)).astype(np.float32),
             n_windows=np.full(40000, 3, dtype=np.int32), weights_sha256=np.array("e" * 64))
    nq = len(qk["slot"])
    np.savez(os.path.join(d, "query_games.npz"), question_id=np.array(qk["question_id"]),
             slot=np.array(qk["slot"], dtype=np.int32), color=np.array(qk["color"]),
             emb_sum=rng.standard_normal((nq, 4)).astype(np.float32), n_windows=np.full(nq, 2, dtype=np.int32),
             parsed=np.ones(nq, dtype=bool), weights_sha256=np.array("e" * 64))
    with open(os.path.join(p02, "player", "repro_check.json"), "w", encoding="utf-8") as f:
        json.dump({"passed": True, "gpu": "GPU X", "torch": "t"}, f)
    return local, p02


class FromVal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.local, cls.p02 = make_fake_inputs(os.path.join(cls.tmp, "in"))
        cls.S = os.path.join(cls.tmp, "S")
        cls.tree = ft.from_val(cls.S, local_dir=cls.local, p02_dir=cls.p02, log=lambda s: None)
        cls.tests = os.path.join(cls.S, "data", "test")
        cls.res = os.path.join(cls.S, "results", "test02")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_files_are_prefix_replacements(self):
        with open(os.path.join(self.local, ft.RANK_VAL), "rb") as f:
            src = f.read().split(b"\r\n")
        for pool, (lo, hi), pre in (("public", (0, 400), b"pub_"), ("private", (400, 800), b"priv_")):
            with open(os.path.join(self.tests, tb.RANK_FILES[pool]), "rb") as f:
                got = f.read()
            want = b"\r\n".join([src[0]] + [pre + ln[4:] for ln in src[1 + lo:1 + hi]]) + b"\r\n"
            self.assertEqual(got, want, pool)
        with open(os.path.join(self.local, ft.PLAYER_VAL_C), "rb") as f:
            src = f.read().split(b"\r\n")
        with open(os.path.join(self.tests, tb.PLAYER_C_FILES["private"]), "rb") as f:
            got = f.read().split(b"\r\n")
        self.assertEqual(got[1], b"priv_g_000001,priv_p_" + src[1].split(b",")[1][6:] + b"," + src[1].split(b",", 2)[2])
        self.assertEqual(len(got), 40002)
        for c in ("rank", "player"):
            self.assertEqual(tb.quick_structure(self.tests, c), [])

    def test_manifest_zip_and_config_agree(self):
        man, man_sha = tb.verify_manifest(self.tests)
        self.assertEqual(man["kind"], "rehearsal")
        z = os.path.join(self.S, "data", "colab_bundle", tb.ZIP_NAME)
        for comp in ("rank", "player"):
            cfg = json.load(open(os.path.join(self.res, comp, "config.json"), encoding="utf-8"))
            self.assertIs(cfg["smoke"], False)
            self.assertIs(cfg["rehearsal"], True)
            self.assertEqual(cfg["test_bundle"]["manifest_sha256"], man_sha)
            self.assertEqual(cfg["test_bundle"]["zip_sha256"], sha(z))
            for n in tb.ALL_NAMES:
                self.assertEqual(cfg["test_bundle"]["files"][n], sha(os.path.join(self.tests, n)))
            self.assertEqual(tb.run_status(json.load(open(os.path.join(self.res, comp, "run_log.json")))),
                             "completed")
            for f in ("env.json", "timings.json", "readonly_before.json", "readonly_after.json"):
                self.assertTrue(os.path.exists(os.path.join(self.res, comp, f)), f)
        self.assertTrue(json.load(open(os.path.join(self.res, "rank", "seam_check.json")))["passed"])
        self.assertTrue(json.load(open(os.path.join(self.res, "player", "repro_check.json")))["passed"])
        self.assertTrue(os.path.exists(os.path.join(self.res, "player", "seam_p02.json")))
        self.assertIn("never run content statistics", self.tree["rule"])

    def test_rank_dumps(self):
        src = os.path.join(self.p02, "rank", "dumps")
        seam = os.path.join(self.res, "rank", "seam_val", "dumps")
        for f in ("val_keys.npz", "val_p02e9_d4.npz", "_DONE_val_keys.json", "_DONE_val_p02e9_d4.json"):
            self.assertEqual(sha(os.path.join(seam, f)), sha(os.path.join(src, f)), f)
        k0 = np.load(os.path.join(src, "val_keys.npz"))
        d0 = np.load(os.path.join(src, "val_p02e9_d4.npz"))
        qids = k0["question_id"].tolist()
        r1, r2 = qids.index("val_q_0401"), qids.index("val_q_0801")
        for pool, (a, b), pre in (("public", (0, r1), "pub_"), ("private", (r1, r2), "priv_")):
            d = os.path.join(self.res, "rank", pool, "dumps")
            k = np.load(os.path.join(d, "val_keys.npz"))
            self.assertEqual(k["question_id"].tolist(), [pre + q[4:] for q in qids[a:b]])
            for f in ("slot", "color", "parsed"):
                self.assertTrue(np.array_equal(k[f], k0[f][a:b]), f)
            lg = np.load(os.path.join(d, "val_p02e9_d4.npz"))
            self.assertEqual(lg["logits"].tobytes(), d0["logits"][a:b].tobytes())
            self.assertEqual(str(lg["weights_sha256"]), str(d0["weights_sha256"]))
            self.assertEqual(int(lg["syms"]), 8)
            mk = json.load(open(os.path.join(d, "_DONE_val_p02e9_d4.json"), encoding="utf-8"))
            self.assertEqual((mk["rows"], mk["sha256"]), (b - a, sha(os.path.join(d, "val_p02e9_d4.npz"))))
            self.assertEqual(mk["weights_sha256"], "f" * 64)
            inp = json.load(open(os.path.join(d, "_INPUT.json"), encoding="utf-8"))
            self.assertEqual(inp["sha256"], sha(os.path.join(self.tests, tb.RANK_FILES[pool])))

    def test_player_dumps(self):
        src = os.path.join(self.p02, "player", "dumps")
        c0 = np.load(os.path.join(src, "cand_games.npz"))
        for f in ("cand_games.npz", "query_games.npz"):
            self.assertEqual(sha(os.path.join(self.res, "player", "seam_val", "dumps", f)), sha(os.path.join(src, f)))
        c = np.load(os.path.join(self.res, "player", "private", "dumps", "cand_games.npz"))
        self.assertEqual(c["emb_sum"].tobytes(), c0["emb_sum"].tobytes())
        self.assertEqual(c["n_windows"].tobytes(), c0["n_windows"].tobytes())
        self.assertEqual(c["player_id"][0], "priv_p_" + str(c0["player_id"][0])[6:])
        self.assertEqual(c["game_id"][0], "priv_g_000001")
        q = np.load(os.path.join(self.res, "player", "public", "dumps", "query_games.npz"))
        self.assertEqual(q["question_id"][0], "pub_pq_0001")
        inp = json.load(open(os.path.join(self.res, "player", "public", "dumps", "_INPUT.json"), encoding="utf-8"))
        self.assertEqual(inp["candidates_sha256"], sha(os.path.join(self.tests, tb.PLAYER_C_FILES["public"])))

    def test_result_zip_round_trip_through_unpack(self):
        """Pack the fake rank records the way the notebook does and unpack them with t02_test_bundle."""
        z = os.path.join(self.tmp, "test02_rank_results.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(os.path.join(self.res, "rank")):
                for f in files:
                    p = os.path.join(root, f)
                    zf.write(p, os.path.relpath(p, self.res).replace(os.sep, "/"))
        out = os.path.join(self.tmp, "unpacked")
        rec = quiet(tb.main, ["unpack", "--zip", z, "--comp", "rank", "--root", self.S, "--results-dir", out])[0]
        self.assertEqual(rec["run_status"], "completed")
        self.assertEqual(sha(os.path.join(out, "rank", "public", "dumps", "val_p02e9_d4.npz")),
                         sha(os.path.join(self.res, "rank", "public", "dumps", "val_p02e9_d4.npz")))

    def test_guards(self):
        with self.assertRaises(SystemExit):
            ft.from_val(self.S, local_dir=self.local, p02_dir=self.p02, log=lambda s: None)   # exists, no --rerun
        with self.assertRaises(SystemExit):
            ft.from_val(os.path.join(REPO, "data", "fake_should_not_exist"), local_dir=self.local,
                        p02_dir=self.p02, log=lambda s: None)
        self.assertFalse(os.path.exists(os.path.join(REPO, "data", "fake_should_not_exist")))


class FromTrain(Tmp):
    def rows(self):
        fit, inner = [], []
        for lv_i, lv in enumerate(["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]):
            for p in range(90):
                pid = f"fit_{lv}_{p}"
                for g in range(106 + p % 3):
                    fit.append({"player_id": pid, "game_id": f"{pid}_g{g}", "rank": lv, "color": "BW"[g % 2],
                                "sgf_content": f"(;SZ[19];B[dd]C[{pid} {g}])"})
        for p in range(130):
            pid = f"inner_{p}"
            for g in range(90 if p < 20 else 110):
                inner.append({"player_id": pid, "game_id": f"{pid}_g{g}", "rank": "D", "color": "BW"[g % 2],
                              "sgf_content": f"(;SZ[19];B[pp]C[{pid} {g}])"})
        return fit, inner

    def test_tree(self):
        fit, inner = self.rows()
        S2 = self.p("S2")
        tree = ft.from_train(S2, rows_fit=fit, rows_inner=inner, log=lambda s: None)
        tests = os.path.join(S2, "data", "test")
        self.assertEqual(tree["quick_structure"], "PASS")
        for c in ("rank", "player"):
            self.assertEqual(tb.quick_structure(tests, c), [])
        fit_sgf = {r["sgf_content"] for r in fit}
        inner_sgf = {r["sgf_content"] for r in inner}
        pools = {}
        for pool in ("public", "private"):
            with open(os.path.join(tests, tb.PLAYER_C_FILES[pool]), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(r["sgf_content"] in fit_sgf for r in rows))
            pools[pool] = {r["sgf_content"].split("C[")[1].split()[0] for r in rows}
            self.assertEqual(len(pools[pool]), 400)
            self.assertTrue(all(r["player_id"].startswith(tb.P_PREFIX[pool]) and len(r["player_id"]) ==
                                len(tb.P_PREFIX[pool]) + 8 for r in rows))
            with open(os.path.join(tests, tb.RANK_FILES[pool]), newline="", encoding="utf-8") as f:
                rq = list(csv.DictReader(f))
            self.assertTrue(all(r[f"sgf_{i}"] in inner_sgf for r in rq for i in range(1, int(r["num_games"]) + 1)))
        self.assertFalse(pools["public"] & pools["private"])
        neg = os.path.join(S2, "neg", "data", "test", tb.PLAYER_C_FILES["public"])
        with open(neg, newline="", encoding="utf-8") as f:
            n_neg = len({r["player_id"] for r in csv.DictReader(f)})
        self.assertEqual(n_neg, 110)                               # inner players with >= 100 games
        self.assertEqual(tree["pools"]["neg player public (NEG)"]["t3_threshold"], 11)
        self.assertEqual(tree["shape"], "mixed")
        with self.assertRaises(SystemExit):
            ft.from_train(S2, rows_fit=fit, rows_inner=inner, log=lambda s: None)
        with self.assertRaises(SystemExit):                        # filled slot colours only for the official shape
            ft.from_train(self.p("S3"), rows_fit=fit, rows_inner=inner, log=lambda s: None, slot_colours="filled")


class FromTrainOfficial(Tmp):
    """from-train --shape official (PREREG_02 rev 8): black-only 400 x 100 pools and questions of 5/10/15/20 games
    (100 each), single-colour rank questions; slot colours after num_games empty or filled."""
    LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]

    @classmethod
    def setUpClass(cls):
        fit, inner = [], []
        for lv in cls.LEVELS:
            for p in range(90):
                pid = f"fit_{lv}_{p}"
                nb = 120 + p % 5
                for g in range(nb + 10):
                    fit.append({"player_id": pid, "game_id": f"{pid}_g{g}", "rank": lv, "color": "B" if g < nb else "W",
                                "sgf_content": f"(;SZ[19];B[dd]C[{pid} {g}])"})
        for p in range(130):
            pid = f"inner_{p}"
            for g in range(110 + (100 if p < 15 else 0)):
                inner.append({"player_id": pid, "game_id": f"{pid}_g{g}", "rank": "D",
                              "color": "BW"[g % 2] if g < 110 else "B", "sgf_content": f"(;SZ[19];B[pp]C[{pid} {g}])"})
        cls.fit, cls.inner = fit, inner

    def tree(self, slot_colours):
        S = self.p("S_" + slot_colours)
        tree = ft.from_train(S, rows_fit=self.fit, rows_inner=self.inner, log=lambda s: None, shape="official",
                             slot_colours=slot_colours)
        return S, os.path.join(S, "data", "test"), tree

    def rows(self, path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_empty_slot_colours(self):
        import t02_checks as tc
        S, tests, tree = self.tree("empty")
        self.assertEqual((tree["shape"], tree["slot_colours"], tree["quick_structure"]), ("official", "empty", "PASS"))
        fit_sgf = {r["sgf_content"] for r in self.fit if r["color"] == "B"}
        for pool in ("public", "private"):
            cands = self.rows(os.path.join(tests, tb.PLAYER_C_FILES[pool]))
            self.assertEqual(len(cands), 40000)
            self.assertEqual({r["color"] for r in cands}, {"B"})
            self.assertTrue(all(r["sgf_content"] in fit_sgf for r in cands))
            for name in (tb.PLAYER_Q_FILES[pool], tb.RANK_FILES[pool]):
                qs = self.rows(os.path.join(tests, name))
                ks = [int(r["num_games"]) for r in qs]
                self.assertEqual({k: ks.count(k) for k in set(ks)}, {5: 100, 10: 100, 15: 100, 20: 100}, name)
                for r in qs:
                    k = int(r["num_games"])
                    cols = {r[f"color_{i}"] for i in range(1, k + 1)}
                    self.assertEqual(len(cols), 1, name)                          # one colour per question
                    self.assertTrue(all(r[f"color_{i}"] == r[f"sgf_{i}"] == "" for i in range(k + 1, 21)))
                    if name in tb.PLAYER_Q_FILES.values():
                        self.assertEqual(cols, {"B"})
            rq = self.rows(os.path.join(tests, tb.RANK_FILES[pool]))
            self.assertEqual({r["color_1"] for r in rq}, {"B", "W"})
        for comp in ("rank", "player"):
            r = tc.t1_check(comp, tests)
            self.assertEqual(r["verdict"], "PASS", (comp, r["subchecks"]["T1.3"]["problems"]))
            for name, d in r["subchecks"]["T1.3"]["files"].items():
                self.assertTrue(d["official_colour_design_record_only"]["matches_official_design"], name)
        neg = self.rows(os.path.join(S, "neg", "data", "test", tb.PLAYER_C_FILES["public"]))
        self.assertEqual((len({r["player_id"] for r in neg}), {r["color"] for r in neg}), (15, {"B"}))
        self.assertEqual(tree["pools"]["neg player public (NEG)"]["t3_threshold"], 2)

    def test_filled_slot_colours_normalize_and_build(self):
        import subprocess
        import t02_checks as tc
        S, tests, tree = self.tree("filled")
        _S, tests_e, _t = self.tree("empty")
        self.assertTrue(tree["quick_structure"].startswith("FAIL on the raw files as intended"))
        for pool in ("public", "private"):
            for name in (tb.PLAYER_C_FILES[pool],):
                self.assertEqual(sha(os.path.join(tests, name)), sha(os.path.join(tests_e, name)))
            for name in (tb.PLAYER_Q_FILES[pool], tb.RANK_FILES[pool]):
                for r in self.rows(os.path.join(tests, name)):
                    k = int(r["num_games"])
                    self.assertTrue(all(r[f"color_{i}"] == r["color_1"] and r[f"sgf_{i}"] == ""
                                        for i in range(k + 1, 21)))
        qnames = set(tb.PLAYER_Q_FILES.values()) | set(tb.RANK_FILES.values())
        for comp in ("rank", "player"):
            r = tc.t1_check(comp, tests)
            self.assertEqual(r["verdict"], "NORMALIZED", comp)
            self.assertEqual(r["normalize_transforms"], ["clear_colour_after_num_games_if_sgf_empty"])
            self.assertEqual(set(r["normalize"]), set(tb.COMP_FILES[comp]) & qnames)
            proc = subprocess.run([sys.executable, "-B", os.path.join(REPO, "tools", "t02_checks.py"), "t1", "--comp",
                                   comp, "--smoke", "--root", S, "--tests-dir", tests, "--only", "T1.1,T1.2,T1.3"],
                                  capture_output=True, text=True, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = os.path.join(S, "built", "data", "test")
        rec = quiet(tb.main, ["build", "--kind", "smoke", "--root", S, "--tests-dir", tests, "--out-dir", out])[0]
        self.assertEqual({c: rec["t1_built"][c]["verdict"] for c in ("rank", "player")},
                         {"rank": "PASS", "player": "PASS"})
        self.assertEqual({n: v for n, v in rec["normalized"].items()},
                         {n: ["clear_colour_after_num_games_if_sgf_empty"] for n in qnames})
        for n in tb.ALL_NAMES:                     # the normalized copies equal the empty-slot tree byte for byte
            self.assertEqual(sha(os.path.join(out, n)), sha(os.path.join(tests_e, n)), n)


# ================================================================ official_log
class OfficialLog(Tmp):
    def setUp(self):
        super().setUp()
        self.repo = self.p("repo")
        self.tests = write_official(os.path.join(self.repo, "data", "test"))
        files = {n: tb.file_record(os.path.join(self.tests, n)) for n in tb.ALL_NAMES}
        tb.write_manifest(self.tests, "official", files, {"rank": "PASS", "player": "PASS"})
        ids = [f"pub_q_{i:04d}" for i in range(1, 401)] + [f"priv_q_{i:04d}" for i in range(1, 401)]
        self.rank_rows = [[q, "D"] for q in ids]
        self.rank_file = write_csv(os.path.join(self.repo, "results", "test02", "rank", "submission_rank.csv"),
                                   ["question_id", "pred_rank"], self.rank_rows)
        self.player_file = write_csv(
            os.path.join(self.repo, "results", "test02", "player", "submission_player.csv"),
            ["question_id"] + [f"top_{i}" for i in range(1, 6)],
            [[q] + [f"{q.split('_')[0]}_p_{j:04d}" for j in range(5)] for q in ids])
        self.log = self.p("repo", "results", "official_scores.json")
        self.state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
        self.write_reports()
        self.patches = [mock.patch.object(ol, "REPO", self.repo),
                        mock.patch.object(ol, "git_file_state", side_effect=lambda rel: dict(self.state))]
        for pt in self.patches:
            pt.start()

    def tearDown(self):
        for pt in self.patches:
            pt.stop()
        super().tearDown()

    def jw(self, path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def write_reports(self, rank=None, player=None, t1=("PASS", "PASS"), seam=None, t1_extra=None, t2t5=True):
        """The committed tool reports that back an upload (t02_rank apply / t02_player submit, local seams, the full
        raw T1 in real mode, and T2-T5 recorded in summary.json)."""
        R = os.path.join(self.repo, "results", "test02")
        self.jw(os.path.join(R, "rank", "apply_test_report.json"), dict(
            {"mode": "real", "seam_local": {"passed": True}, "fallback_used": False,
             "check_submission": {"mode": "official", "errors": []},
             "outputs_sha256": {"submission_rank.csv": sha(self.rank_file)}}, **(rank or {})))
        self.jw(os.path.join(R, "player", "submit_report.json"), dict(
            {"mode": "real", "check_submission": {"official": True, "passed": True, "errors": []},
             "output": {"sha256": sha(self.player_file)}}, **(player or {})))
        for comp in ("rank", "player"):
            self.jw(os.path.join(R, comp, "seam_val", "seam_local.json"),
                    dict({"passed": True, "mode": "real"}, **(seam or {})))
        for comp, v in zip(("rank", "player"), t1):
            self.jw(os.path.join(R, "checks", comp, "t1.json"), dict(
                {"verdict": v, "comp": comp, "complete": True, "stage": "raw",
                 "subchecks_run": ["T1.1", "T1.2", "T1.3", "T1.4", "T1.5"], "provenance": {"mode": "real"}},
                **(t1_extra or {})))
            if t2t5:
                checks = {}
                for c, verdict in (("T2", "PASS"), ("T3", "RECORD_ONLY" if comp == "rank" else "PASS"),
                                   ("T4", "PASS"), ("T5", "RECORD_ONLY")):
                    dp = os.path.join(R, "checks", comp, c.lower() + ".json")
                    self.jw(dp, {"verdict": verdict})
                    checks[c] = {"verdict": verdict, "details_file": c.lower() + ".json", "sha256": sha(dp)}
                self.jw(os.path.join(R, "checks", comp, "summary.json"), {"checks": checks})

    def run_ol(self, *argv):
        return quiet(ol.main, list(argv) + ["--log", self.log])

    def prepare(self, comp="rank", path=None, reason="first upload (U1)", *extra):
        return self.run_ol("prepare", "--comp", comp, "--file", path or self.rank_file, "--reason", reason,
                           "--tests-dir", self.tests, *extra)

    def test_prepare_schema_and_ok_text(self):
        e, out = self.prepare()
        log = json.load(open(self.log, encoding="utf-8"))
        self.assertEqual(log["format"], "c10 official_scores v1")
        self.assertEqual(log["teams"], {"rank": "TEAM_11248", "player": "TEAM_11249"})
        self.assertEqual(log["baselines_public"], {"rank": 0.4631, "player": 0.2845})
        self.assertEqual(log["final_choice"], {"rank": None, "player": None})
        keys = {"id", "comp", "status", "file", "sha256", "rows", "git_commit", "method", "reason",
                "check_submission", "prepared_at", "user_ok_at", "uploaded_at", "public_score",
                "public_delta_vs_prev", "lb_rank", "investigate", "note"}
        self.assertTrue(keys <= set(log["uploads"][0]))
        self.assertEqual((e["id"], e["status"], e["rows"], e["file"]),
                         (1, "prepared", 800, "results/test02/rank/submission_rank.csv"))
        self.assertEqual(e["method"], "prereg02 R20 rule B (甲)")
        self.assertEqual(e["check_submission"], "OK (official)")
        for s in ("TEAM_11248", "競賽二", sha(self.rank_file)[:12], "800", "#1"):
            self.assertIn(s, out)
        e2, out2 = self.prepare("player", self.player_file)
        self.assertEqual(e2["id"], 2)
        self.assertIn("TEAM_11249", out2)

    def test_prepare_refusals(self):
        with self.assertRaises(SystemExit):                        # name does not match the competition
            self.prepare("player", self.rank_file)
        self.state["clean"] = False
        with self.assertRaises(SystemExit) as cm:
            self.prepare()
        self.assertIn("committed and unchanged", str(cm.exception))
        self.state["clean"] = True
        write_csv(self.rank_file, ["question_id", "pred_rank"], list(reversed(self.rank_rows)))
        with self.assertRaises(SystemExit) as cm:
            self.prepare()
        self.assertIn("row order differs", str(cm.exception))
        write_csv(self.rank_file, ["question_id", "pred_rank"], self.rank_rows)
        with open(os.path.join(self.tests, tb.RANK_FILES["public"]), "ab") as f:
            f.write(b"\n")
        with self.assertRaises(SystemExit) as cm:
            self.prepare()
        self.assertIn("differs from", str(cm.exception))
        files = {n: tb.file_record(os.path.join(self.tests, n)) for n in tb.ALL_NAMES}
        tb.write_manifest(self.tests, "rehearsal", files, {"rank": "PASS", "player": "PASS"})
        with self.assertRaises(SystemExit) as cm:                  # a rehearsal MANIFEST never backs an upload
            self.prepare()
        self.assertIn("kind 'rehearsal'", str(cm.exception))
        with self.assertRaises(SystemExit):                        # outside the repository
            self.run_ol("prepare", "--comp", "rank", "--file", write_csv(self.p("x", "submission_rank.csv"),
                        ["question_id", "pred_rank"], self.rank_rows), "--reason", "r", "--tests-dir", self.tests)
        self.assertFalse(os.path.exists(self.log))

    def test_prepare_is_bound_to_passed_checks_and_seams(self):
        """PREREG_02 rev 5 9.12 (review finding 2026-09-26): the file must be the one the real-mode tool wrote, with
        the local seam passed and T1 PASS / NORMALIZED (FAIL_OPENING: rank only with the R10 fallback)."""
        cases = [
            (dict(rank={"mode": "dry-run"}), "mode 'dry-run'"),
            (dict(rank={"seam_local": {"passed": False}}), "local seam did not pass"),
            (dict(rank={"outputs_sha256": {"submission_rank.csv": "0" * 64}}), "records another"),
            (dict(rank={"check_submission": {"mode": "default", "errors": []}}), "check_submission"),
            (dict(seam={"passed": False}), "seam_local.json"),
            (dict(seam={"mode": "rehearsal"}), "seam_local.json"),
            (dict(t1=("FAIL_STRUCT", "PASS")), "FAIL_STRUCT"),
            (dict(t1=("FAIL_OPENING", "PASS")), "--fallback R10"),
        ]
        for kw, msg in cases:
            with self.subTest(kw=kw):
                self.write_reports(**kw)
                with self.assertRaises(SystemExit) as cm:
                    self.prepare()
                self.assertIn(msg, str(cm.exception))
        self.assertFalse(os.path.exists(self.log))
        os.remove(os.path.join(self.repo, "results", "test02", "rank", "apply_test_report.json"))
        with self.assertRaises(SystemExit) as cm:
            self.prepare()
        self.assertIn("is missing", str(cm.exception))
        # FAIL_OPENING with the R10 fallback (rank) and the per-pool procedure (player) may be uploaded
        self.write_reports(rank={"fallback_used": True}, t1=("FAIL_OPENING", "FAIL_OPENING"))
        e, out = self.prepare()
        self.assertEqual((e["t1_verdict"], e["fallback_used"]), ("FAIL_OPENING", True))
        self.assertIn("R10", out)
        e2, _ = self.prepare("player", self.player_file)
        self.assertEqual(e2["t1_verdict"], "FAIL_OPENING")
        self.assertEqual(set(e2["reports_sha256"]), {"results/test02/player/submit_report.json",
                                                     "results/test02/player/seam_val/seam_local.json",
                                                     "results/test02/checks/player/t1.json",
                                                     "results/test02/checks/player/summary.json"}
                         | {"results/test02/checks/player/t%d.json" % i for i in (2, 3, 4, 5)})
        self.write_reports(player={"output": {"sha256": "1" * 64}})
        with self.assertRaises(SystemExit) as cm:
            self.prepare("player", self.player_file)
        self.assertIn("records another", str(cm.exception))
        self.write_reports(player={"check_submission": {"official": False, "passed": True}})
        with self.assertRaises(SystemExit):
            self.prepare("player", self.player_file)

    def test_prepare_needs_the_full_raw_t1(self):
        """Review finding (realmode, major): a structure-only t1.json (--only T1.1,T1.2,T1.3), a built-copy re-check
        or a t1.json from another mode never backs an upload."""
        for extra, msg in (({"complete": False, "subchecks_run": ["T1.1", "T1.2", "T1.3"]}, "T1 ran only"),
                           ({"stage": "built"}, "stage 'built'"),
                           ({"provenance": {"mode": "rehearsal"}}, "mode 'rehearsal'")):
            with self.subTest(extra=extra):
                self.write_reports(t1_extra=extra)
                with self.assertRaises(SystemExit) as cm:
                    self.prepare()
                self.assertIn(msg, str(cm.exception))
                with self.assertRaises(SystemExit):
                    self.prepare("player", self.player_file)
        self.assertFalse(os.path.exists(self.log))

    def test_prepare_needs_t2_to_t5_recorded(self):
        """docs/TESTDAY.md U1: T2-T5 ran and are recorded (any verdict); --checks-missing-reason overrides."""
        cdir = os.path.join(self.repo, "results", "test02", "checks", "rank")
        os.remove(os.path.join(cdir, "t3.json"))
        with self.assertRaises(SystemExit) as cm:
            self.prepare()
        self.assertIn("T2-T5 are not all recorded", str(cm.exception))
        self.assertIn("T3", str(cm.exception))
        e, out = self.prepare("rank", None, "first upload (U1)", "--checks-missing-reason", "T3 crashed (bug #1)")
        self.assertEqual(e["t2_t5"]["missing"], ["T3"])
        self.assertEqual(e["t2_t5"]["missing_reason"], "T3 crashed (bug #1)")
        self.assertEqual(e["t2_t5"]["recorded"], {"T2": "PASS", "T4": "PASS", "T5": "RECORD_ONLY"})
        self.assertIn("T3 crashed", out)
        self.write_reports()
        with open(os.path.join(cdir, "t4.json"), "w", encoding="utf-8") as f:     # not the file summary records
            f.write('{"verdict": "PASS", "edited": true}')
        with self.assertRaises(SystemExit) as cm:
            self.prepare()
        self.assertIn("not the file summary.json records", str(cm.exception))
        os.remove(os.path.join(cdir, "summary.json"))
        with self.assertRaises(SystemExit):
            self.prepare()
        self.write_reports()
        e, out = self.prepare()
        self.assertEqual(e["t2_t5"], {"recorded": {"T2": "PASS", "T3": "RECORD_ONLY", "T4": "PASS",
                                                   "T5": "RECORD_ONLY"}, "missing": [], "missing_reason": None})
        self.assertIn("results/test02/checks/rank/summary.json", e["reports_sha256"])
        self.assertIn("results/test02/checks/rank/t3.json", e["reports_sha256"])
        self.assertIn("T2 PASS", out)

    def test_rank_method_follows_the_fallback(self):
        """Review finding (realmode, minor): an R10 fallback file is recorded as R10, and --method cannot say R20."""
        self.write_reports(rank={"fallback_used": True}, t1=("FAIL_OPENING", "PASS"))
        e, out = self.prepare()
        self.assertIn("R10", e["method"])
        self.assertNotIn("R20", e["method"])
        with self.assertRaises(SystemExit) as cm:
            self.prepare("rank", None, "again", "--method", "prereg02 R20 rule B (甲)")
        self.assertIn("names R20", str(cm.exception))
        self.write_reports()
        with self.assertRaises(SystemExit):
            self.prepare("rank", None, "again", "--method", "prereg02 R10 fallback")
        e, _ = self.prepare("rank", None, "again", "--method", "prereg02 R20 rule B (甲), re-upload")
        self.assertIn("R20", e["method"])
        self.assertEqual(ol.method_for("rank", False, "R100 variant (not a registered id)"),
                         "R100 variant (not a registered id)")
        self.assertEqual(ol.method_for("player", None, None), ol.DEFAULT_METHOD["player"])

    def test_prepare_needs_its_own_tools_committed(self):
        """Review finding (realmode, minor): prepare prints 'check_submission --official 通過', so the
        check_submission.py it ran and this tool must be the committed ones."""
        for dirty in ol.OWN_TOOLS:
            with self.subTest(dirty=dirty):
                with mock.patch.object(ol, "git_file_state", side_effect=lambda rel, d=dirty: dict(
                        self.state, clean=rel != d)):
                    with self.assertRaises(SystemExit) as cm:
                        self.prepare()
                self.assertIn(dirty, str(cm.exception))
        self.assertFalse(os.path.exists(self.log))

    def test_crlf_checkout_gets_a_restore_hint(self):
        """Review finding (realmode, minor): a git checkout with core.autocrlf=true turns the committed LF file into
        CRLF; prepare and uploaded tell how to get the LF bytes back."""
        lf = open(self.rank_file, "rb").read()
        with open(self.rank_file, "wb") as f:
            f.write(lf.replace(b"\n", b"\r\n"))
        with self.assertRaises(SystemExit) as cm:
            self.prepare()
        self.assertIn("git -c core.autocrlf=false checkout -- results/test02/rank/submission_rank.csv",
                      str(cm.exception))
        with open(self.rank_file, "wb") as f:
            f.write(lf)
        self.prepare()
        with open(self.rank_file, "wb") as f:
            f.write(lf.replace(b"\n", b"\r\n"))
        with self.assertRaises(SystemExit) as cm:
            self.run_ol("uploaded", "--id", "1", "--ok-at", "2099-11-04T15:00", "--uploaded-at", "2099-11-04T15:05")
        self.assertIn("core.autocrlf=false checkout", str(cm.exception))

    def test_upload_public_final_status(self):
        self.prepare()
        with self.assertRaises(SystemExit):                        # OK before the entry existed
            self.run_ol("uploaded", "--id", "1", "--ok-at", "2026-01-01T10:00", "--uploaded-at", "2026-01-01T10:05")
        with self.assertRaises(SystemExit):                        # upload before the OK
            self.run_ol("uploaded", "--id", "1", "--ok-at", "2099-11-04T10:00", "--uploaded-at", "2099-11-04T09:00")
        e, out = self.run_ol("uploaded", "--id", "1", "--ok-at", "2099-11-04T15:00", "--uploaded-at",
                             "2099-11-04T15:05")
        self.assertEqual(e["status"], "uploaded")
        self.assertEqual(e["uploaded_at"], "2099-11-04T15:05:00+08:00")
        with self.assertRaises(SystemExit):                        # only once
            self.run_ol("uploaded", "--id", "1", "--ok-at", "2099-11-04T15:00", "--uploaded-at", "2099-11-04T15:05")
        e, out = self.run_ol("public", "--id", "1", "--score", "0.52", "--lb-rank", "31")
        self.assertEqual((e["public_delta_vs_prev"], e["investigate"], e["lb_rank"]), (None, False, 31))
        # second upload of the same file, 0.06 lower -> investigate
        self.prepare(reason="re-upload")
        self.run_ol("uploaded", "--id", "2", "--ok-at", "2099-11-05T15:00", "--uploaded-at", "2099-11-05T15:01")
        e, out = self.run_ol("public", "--id", "2", "--score", "0.46")
        self.assertAlmostEqual(e["public_delta_vs_prev"], -0.06)
        self.assertTrue(e["investigate"])
        self.assertIn("INVESTIGATE", out)
        with self.assertRaises(SystemExit):
            self.run_ol("public", "--id", "2", "--score", "0.5")   # already set
        # below baseline alone
        self.prepare(reason="third")
        self.run_ol("uploaded", "--id", "3", "--ok-at", "2099-11-06T15:00", "--uploaded-at", "2099-11-06T15:01")
        e, _ = self.run_ol("public", "--id", "3", "--score", "0.45")
        self.assertAlmostEqual(e["public_delta_vs_prev"], -0.01)
        self.assertTrue(e["investigate"])
        # final choice = entry 1 (same bytes as 3): last upload equals it
        with self.assertRaises(SystemExit):
            self.run_ol("final", "--comp", "player", "--id", "1", "--basis", "x")
        self.run_ol("final", "--comp", "rank", "--id", "1", "--basis", "prereg02 local validation")
        st, out = self.run_ol("status", "--now", "2099-11-01T00:00")
        self.assertTrue(st["rank"]["last_is_final"])
        self.assertEqual(st["rank"]["uploads_per_day"], {"2099-11-04": 1, "2099-11-05": 1, "2099-11-06": 1})
        self.assertIn("最終選擇：#1", out)
        self.assertIn("還沒有上傳", out)                             # player
        self.assertIn("已過", out)

    def test_file_changed_after_prepare(self):
        self.prepare()
        write_csv(self.rank_file, ["question_id", "pred_rank"], [[q, "C"] for q, _ in self.rank_rows])
        with self.assertRaises(SystemExit) as cm:
            self.run_ol("uploaded", "--id", "1", "--ok-at", "2099-11-04T15:00", "--uploaded-at", "2099-11-04T15:05")
        self.assertIn("no longer the prepared bytes", str(cm.exception))

    def test_daily_limit_warning(self):
        for i in range(1, 7):
            self.prepare(reason=f"n{i}")
            _e, out = self.run_ol("uploaded", "--id", str(i), "--ok-at", f"2099-11-04T1{i}:00",
                                  "--uploaded-at", f"2099-11-04T1{i}:01")
        self.assertIn("WARNING: 6 uploads on 2099-11-04", out)
        _st, out = self.run_ol("status", "--now", "2099-11-01T00:00")
        self.assertIn("超過每天 5 次", out)


if __name__ == "__main__":
    unittest.main()
