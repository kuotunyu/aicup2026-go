"""Tests for colab/build_test02_player_nb.py and colab/test02_player.ipynb (spec 2.0 / 2.2 / 2.3).

* byte-identical rebuild (builder run into a temp dir, compared with the committed notebook and diff);
* segment identity with colab/prereg02_player.ipynb (ast.get_source_segment), cells kept verbatim;
* the three per-pool dump cells: preamble + identical body, order public, private, seam_val, repro after;
* the label-file refusal line exactly once; competition-2 test files only named, never opened;
* pins consistent with the repo files (base notebook, prereg02 dumps and weights, prereg01 files);
* the notebook source SHA-256 recomputes (the rule tools/t02_player.py enforces in real mode);
* the notebook's own quick preflight functions and setup cell, executed on synthetic files.
Synthetic data only; nothing here reads a validation or test file. Run:
    python -m pytest tests/test_t02_nb_player.py
"""

import ast
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import unittest
import zipfile
from collections import Counter
from contextlib import redirect_stdout
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t02_player as tp  # noqa: E402

NB = os.path.join(REPO, "colab", "test02_player.ipynb")
BASE = os.path.join(REPO, "colab", "prereg02_player.ipynb")
BUILDER = os.path.join(REPO, "colab", "build_test02_player_nb.py")
DIFF = os.path.join(REPO, "results", "test02", "notebook_diff_player.txt")
NEEDLE = "ans" + "wers"
BASE_SHA256 = "738c1dd81aa8a93953e45019cf6563e9b2b44ffa7cf846763c51fcc8190a5c12"
OFFICIAL_NAMES = ["rank_prediction_test_public.csv", "rank_prediction_test_private.csv",
                  "player_identification_test_public.csv", "player_identification_test_private.csv",
                  "player_identification_candidates_public.csv", "player_identification_candidates_private.csv"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def src(cell):
    return "".join(cell["source"])


def sha_raw(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha_folded(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def segments(source):
    tree = ast.parse(source)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("game_embeddings", "save_npz"):
            out[node.name] = ast.get_source_segment(source, node)
        if isinstance(node, ast.With) and ast.get_source_segment(source, node.items[0].context_expr) == "Pool(NPROC)":
            out["with_pool"] = ast.get_source_segment(source, node)
    return out


class Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = load(NB)
        cls.base = load(BASE)
        cls.by_id = {c["id"]: c for c in cls.nb["cells"]}
        cls.base_by_id = {c["id"]: c for c in cls.base["cells"]}

    def cell_with(self, text):
        hits = [c for c in self.nb["cells"] if c["cell_type"] == "code" and text in src(c)]
        self.assertEqual(len(hits), 1, text)
        return hits[0]

    def test_rebuild_is_byte_identical(self):
        tmp = tempfile.mkdtemp()
        try:
            r = subprocess.run([sys.executable, BUILDER, REPO, "--out-dir", tmp], capture_output=True, text=True,
                               encoding="utf-8", timeout=600)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            for mine, committed in (("test02_player.ipynb", NB), ("notebook_diff_player.txt", DIFF)):
                with open(os.path.join(tmp, mine), "rb") as f1, open(committed, "rb") as f2:
                    self.assertEqual(f1.read(), f2.read().replace(b"\r\n", b"\n"), mine)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_base_pin(self):
        self.assertEqual(sha_raw(BASE), BASE_SHA256)
        self.assertEqual(self.nb["metadata"]["c10"]["base_notebook_sha256"], BASE_SHA256)
        with open(BUILDER, encoding="utf-8") as f:
            self.assertIn('BASE_SHA256 = "%s"' % BASE_SHA256, f.read())

    def test_cells_kept_verbatim(self):
        for cid in ("c10-player-03", "c27134cb", "c10-player-04", "7bfd8e59", "bf5fcdb7", "c10-player02-06",
                    "c10-player02-07"):
            self.assertEqual(src(self.by_id[cid]), src(self.base_by_id[cid]), cid)
            self.assertEqual(self.by_id[cid]["metadata"]["c10"]["change"], "identical", cid)

    def test_dump_cells_segment_identity_and_order(self):
        base_dump = src(self.base_by_id["c10-player02-05"])
        want = segments(base_dump)
        self.assertEqual(set(want), {"game_embeddings", "save_npz", "with_pool"})
        ids = [c["id"] for c in self.nb["cells"]]
        dump_ids = ["c10-player02-05-%s" % p for p in ("public", "private", "seam_val")]
        self.assertEqual([i for i in ids if i.startswith("c10-player02-05")], dump_ids)
        bodies = set()
        for pool, cid in zip(("public", "private", "seam_val"), dump_ids):
            s = src(self.by_id[cid])
            first, body = s.split("\n", 1)
            self.assertTrue(first.startswith("POOL = %r; QUESTIONS_CSV, CANDIDATES_CSV = POOL_FILES[POOL]" % pool),
                            first)
            bodies.add(body)
            self.assertEqual(segments(s), want, cid)
        self.assertEqual(len(bodies), 1)
        body = bodies.pop()
        self.assertIn("DUMP_DIR = os.path.join(RESULT_DIR, POOL, 'dumps')", body)
        self.assertIn("tuple(EXPECTED_COUNTS[POOL])", body)
        self.assertIn("DUMP_INFO_BY_POOL[POOL] = DUMP_INFO", body)
        self.assertIn("write_input_json(POOL)", body)
        # seam_val runs last, directly before the (verbatim) reproduction check that reads its globals
        self.assertLess(ids.index(dump_ids[-1]), ids.index("c10-player02-07"))
        self.assertLess(ids.index("c10-player02-07"), ids.index("c10-player-10"))

    def test_refusal_line_exactly_once(self):
        hits = [(c["cell_type"], ln.strip()) for c in self.nb["cells"] for ln in src(c).splitlines()
                if NEEDLE in ln.lower()]
        self.assertEqual(len(hits), 1, hits)
        self.assertEqual(hits[0][0], "code")
        self.assertTrue(hits[0][1].startswith("assert not any(") and "'%s' in" % NEEDLE in hits[0][1])
        with open(NB, encoding="utf-8") as f:
            self.assertEqual(f.read().lower().count(NEEDLE), 1)
        for path in (BUILDER, os.path.join(REPO, "tools", "t02_player.py"), os.path.abspath(__file__)):
            with open(path, encoding="utf-8") as f:
                self.assertNotIn(NEEDLE, f.read().lower(), path)

    def test_competition_2_files_named_only(self):
        lines = [ln for c in self.nb["cells"] if c["cell_type"] == "code" for ln in src(c).splitlines()
                 if "rank_prediction_test_" in ln]
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("TEST_NAMES = "))
        self.assertEqual(ast.literal_eval(lines[0].split("=", 1)[1].strip()), OFFICIAL_NAMES)
        setup = src(self.by_id["c10-player-02"])
        self.assertIn("'public': ('player_identification_test_public.csv', "
                      "'player_identification_candidates_public.csv')", setup)
        self.assertIn("'private': ('player_identification_test_private.csv', "
                      "'player_identification_candidates_private.csv')", setup)
        pre = self.cell_with("TEST_NAMES = [")
        self.assertIn("for _n in ['tests/MANIFEST.json'] + ['tests/' + n for n in PLAYER_NAMES]:", src(pre))

    def test_pins_consistent_with_repo(self):
        pre = src(self.cell_with("WEIGHTS_SHA256_PIN = "))
        w = re.search(r"WEIGHTS_SHA256_PIN = '([0-9a-f]{64})'", pre).group(1)
        dumps = ast.literal_eval(re.search(r"P02_DUMP_SHA256 = (\{.*?\})\n", pre).group(1))
        p02 = os.path.join(REPO, "results", "prereg02", "player")
        fz = load(os.path.join(p02, "frozen_config.json"))
        self.assertEqual(w, fz["inputs"]["cand_games_npz"]["weights_sha256"])
        self.assertEqual(w, tp.WEIGHTS_SHA256)
        for k, v in dumps.items():
            self.assertEqual(sha_raw(os.path.join(p02, "dumps", k)), v, k)
        self.assertEqual(dumps, tp.P02_DUMP_SHA256)
        # prereg01 pins inherited in the preflight cell equal the repo copies
        p01 = ast.literal_eval(re.search(r"P01_EXPECTED_SHA256 = (\{.*?\n\})", pre, re.S).group(1))
        d1 = os.path.join(REPO, "results", "prereg01", "player")
        self.assertEqual(p01["best_model.pth"], w)
        self.assertEqual(sha_folded(os.path.join(d1, "player_val_top5.csv")), p01["player_val_top5.csv"])
        self.assertEqual(sha_raw(os.path.join(d1, "player_val_embeddings.npz")), p01["player_val_embeddings.npz"])
        self.assertEqual(sha_folded(os.path.join(d1, "predict_summary.json")), p01["predict_summary.json"])
        # GPU gate still G4 only
        self.assertIn("GPU_MODEL_REQUIRED = 'RTX PRO 6000'", pre)

    def test_notebook_source_sha256(self):
        got, embedded = tp.notebook_source_sha256(NB)
        self.assertIsNotNone(got)
        self.assertEqual(got, embedded)
        self.assertEqual(self.nb["metadata"]["c10"]["notebook_source_sha256"], got)
        self.assertTrue(tp.notebook_identity({"notebook": tp.NOTEBOOK_REL,
                                              "notebook_sha256": got})["matches_committed_notebook"])
        self.assertFalse(tp.notebook_identity({"notebook": tp.NOTEBOOK_REL,
                                               "notebook_sha256": "0" * 64})["matches_committed_notebook"])
        # any code change changes the SHA
        tmp = tempfile.mkdtemp()
        try:
            nb = json.loads(json.dumps(self.nb))
            nb["cells"][1]["source"] = src(nb["cells"][1]) + "\n# edited\n"
            p = os.path.join(tmp, "x.ipynb")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(nb, f)
            self.assertNotEqual(tp.notebook_source_sha256(p)[0], got)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_changes_table_in_config_and_metadata(self):
        cfg = src(self.cell_with("T02_CHANGES = ["))
        start = cfg.index("T02_CHANGES = [")
        end = cfg.index("\n]\n", start) + 2
        table = ast.literal_eval(cfg[start:end].split("=", 1)[1].strip())
        self.assertEqual(table, self.nb["metadata"]["c10"]["changes_from_prereg02"])
        self.assertIn("'changes_from_prereg02': T02_CHANGES", cfg)
        for key in ("'smoke': SMOKE", "'rehearsal': REHEARSAL", "'notebook_sha256': NB_SOURCE_SHA256",
                    "'gpu': ENV['gpu']", "'torch': ENV['torch']", "'cudnn': ENV['cudnn']", "'weights': None",
                    "'test_bundle': None", "'sets': None"):
            self.assertIn(key, cfg)
        head = src(self.nb["cells"][0])
        for i, t in enumerate(table, 1):
            self.assertIn("%d. **" % i, head)

    def test_code_cells_compile_and_no_training(self):
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            compile(src(c), "<cell %d>" % i, "exec")
            if c["id"] != "c27134cb":
                self.assertNotIn("import Pool", src(c), i)
            self.assertNotIn("torch.save", src(c), i)
            names = {t.string for t in tokenize.generate_tokens(io.StringIO(src(c)).readline)
                     if t.type == tokenize.NAME}          # code names only (comments may mention them)
            self.assertFalse(names & {"load_rows", "c10data", "optimizer", "train_loader", "decide"}, i)
        self.assertEqual(len({c["id"] for c in self.nb["cells"]}), len(self.nb["cells"]))

    def test_outputs_and_status_contract(self):
        fin = src(self.by_id["c10-player-10"])
        self.assertIn("RESULTS_ZIP = os.path.join(RESULT_ROOT, 'test02_player_results.zip')", fin)
        self.assertIn("save_json('readonly_after.json', READONLY_AFTER)", fin)
        self.assertIn("'readonly_changed' if not READONLY_UNCHANGED", fin)
        self.assertIn("'repro_failed' if not REPRO['passed']", fin)
        self.assertIn("set_status(FINAL_STATUS", fin)
        self.assertIn("drive.flush_and_unmount()", fin)
        pre = src(self.cell_with("def set_status("))
        self.assertIn("RUN_LOG['status'] = status", pre)
        self.assertIn("save_json('readonly_before.json', READONLY_BEFORE)", pre)
        seam = src(self.cell_with("SEAM_P02 = {"))
        self.assertIn("save_json('seam_p02.json', SEAM_P02)", seam)
        setup = src(self.by_id["c10-player-02"])
        self.assertIn("'test02_smoke' if SMOKE else 'test02_rehearsal' if REHEARSAL else 'test02'", setup)


def notebook_functions(names, consts=()):
    """Exec the named top-level functions / constant assignments of the notebook's preflight cell."""
    nb = load(NB)
    cell = [src(c) for c in nb["cells"] if c["cell_type"] == "code" and "def quick_questions(" in src(c)]
    assert len(cell) == 1
    s = cell[0]
    tree = ast.parse(s)
    parts = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            parts.append(ast.get_source_segment(s, node))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in consts for t in node.targets):
            parts.append(ast.get_source_segment(s, node))
    ns = {"csv": csv, "io": io, "os": os, "Counter": Counter}
    exec(compile("\n\n".join(parts), "<preflight functions>", "exec"), ns)
    return ns


class QuickPreflight(unittest.TestCase):
    """The notebook's quick format check (a T1 subset) on synthetic official-shaped files."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ns = notebook_functions({"read_test_csv", "quick_questions", "quick_candidates", "smoke_counts"},
                                     {"QCOLS", "CCOLS"})
        self.ns["TESTS_DIR"] = self.tmp
        self.ns.update(SMOKE_N_CAND_PLAYERS=12, SMOKE_CAND_GAMES=3, SMOKE_N_QUESTIONS=6, SMOKE_Q_GAMES=2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def questions(self, name, prefix="pub_", n=400, bom=False, crlf=False, mutate=None):
        rows = []
        for i in range(n):
            k = 5 + i % 16
            r = {"question_id": "%sq_%04d" % (prefix, i + 1), "num_games": str(k)}
            for j in range(1, 21):
                r["sgf_%d" % j] = "(;SZ[19];B[pd])" if j <= k else ""
                r["color_%d" % j] = "BW"[j % 2] if j <= k else ""
            if mutate:
                mutate(i, r)
            rows.append(r)
        cols = ["question_id", "num_games"] + ["sgf_%d" % j for j in range(1, 21)] + ["color_%d" % j for j in range(1, 21)]
        with open(os.path.join(self.tmp, name), "w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\r\n" if crlf else "\n")
            w.writeheader()
            w.writerows(rows)

    def candidates(self, name, pprefix="pub_p_", gprefix="pub_g_", short=False, colour="B"):
        with open(os.path.join(self.tmp, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["game_id", "player_id", "sgf_content", "color"])
            g = 0
            for p in range(400):
                for _ in range(100 - (1 if short and p == 7 else 0)):
                    g += 1
                    w.writerow(["%s%06d" % (gprefix, g), "%s%03d" % (pprefix, p), "(;SZ[19];B[pd])", colour])

    def q(self, name, prefix="pub_"):
        probs = []
        info = self.ns["quick_questions"](name, prefix, probs)
        return probs, info

    def c(self, name, pp="pub_p_", gp="pub_"):
        probs = []
        info = self.ns["quick_candidates"](name, pp, gp, probs)
        return probs, info

    def test_good_files_pass(self):
        self.questions("q.csv", crlf=True)
        probs, info = self.q("q.csv")
        self.assertEqual(probs, [])
        self.assertEqual(info["rows"], 400)
        self.assertEqual(sum(info["num_games"]), sum(5 + i % 16 for i in range(400)))
        self.candidates("c.csv")
        probs, info = self.c("c.csv")
        self.assertEqual(probs, [])
        self.assertEqual((info["rows"], len(info["players"])), (40000, 400))
        # SMOKE counts follow the dump cell's reduction
        self.assertEqual(self.ns["smoke_counts"](self.q("q.csv")[1], info), (36, 12, 6))

    def test_question_file_problems(self):
        cases = {
            "bom": dict(bom=True),
            "399": dict(n=399),
            "prefix": dict(prefix="priv_"),
            "num_games": dict(mutate=lambda i, r: r.update(num_games="21") if i == 3 else None),
            "hole": dict(mutate=lambda i, r: r.update(sgf_2="") if i == 5 else None),
            "extra": dict(mutate=lambda i, r: r.update(sgf_20="(;)") if i == 0 else None),
            "colour": dict(mutate=lambda i, r: r.update(color_1="b") if i == 9 else None),
        }
        for label, kw in cases.items():
            prefix = kw.pop("prefix", "pub_")
            self.questions("q_%s.csv" % label, prefix=prefix, **kw)
            probs, _ = self.q("q_%s.csv" % label)
            self.assertTrue(probs, label)
        # missing columns
        with open(os.path.join(self.tmp, "q_cols.csv"), "w", encoding="utf-8") as f:
            f.write("question_id,num_games\npub_q_0001,5\n")
        probs, info = self.q("q_cols.csv")
        self.assertTrue(probs and info is None)

    def test_candidate_file_problems(self):
        self.candidates("c99.csv", short=True)
        probs, _ = self.c("c99.csv")
        self.assertTrue(any("100" in p for p in probs), probs)
        self.candidates("cpref.csv", pprefix="priv_p_")
        self.assertTrue(self.c("cpref.csv")[0])
        self.candidates("cgame.csv", gprefix="priv_g_")
        self.assertTrue(self.c("cgame.csv")[0])
        self.candidates("ccol.csv", colour="w")
        self.assertTrue(self.c("ccol.csv")[0])


class SetupCell(unittest.TestCase):
    """The setup cell: output root from SMOKE / the test zip's kind, and the write guard over prereg01 and
    prereg02 (a write into prereg02_smoke is refused)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.drive = os.path.join(self.tmp, "drive")
        os.makedirs(self.drive)
        nb = load(NB)
        self.src = [src(c) for c in nb["cells"] if c["id"] == "c10-player-02"][0]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def zip_kind(self, kind):
        with zipfile.ZipFile(os.path.join(self.drive, "C10_test_v1.zip"), "w") as z:
            z.writestr("tests/MANIFEST.json", json.dumps({"format": "c10-test02 manifest v1", "kind": kind}))

    def run_setup(self, smoke):
        env = {"C10_DRIVE_DIR": self.drive, "C10_WORK_DIR": os.path.join(self.tmp, "work")}
        if smoke:
            env["C10_SMOKE"] = "1"
        ns = {"__name__": "__main__"}
        with mock.patch.dict(os.environ, env, clear=False), redirect_stdout(io.StringIO()):
            if not smoke:
                os.environ.pop("C10_SMOKE", None)
            try:
                exec(compile(self.src, "<setup>", "exec"), ns)
            except SystemExit:
                pass        # non-SMOKE on a spawn platform stops after the paths are set (fork check)
        return ns

    def test_result_roots(self):
        self.zip_kind("smoke")
        ns = self.run_setup(True)
        self.assertEqual(os.path.basename(ns["RESULT_ROOT"]), "test02_smoke")
        self.assertEqual(os.path.basename(ns["P02_ROOT"]), "prereg02_smoke")
        self.assertEqual(ns["RESULT_DIR"], os.path.join(ns["RESULT_ROOT"], "player"))
        with self.assertRaises(RuntimeError):
            ns["guard_write"](os.path.join(ns["P02_ROOT"], "player", "x.json"))
        with self.assertRaises(RuntimeError):
            ns["guard_write"](os.path.join(ns["P01_ROOT"], "player"))
        self.assertEqual(ns["guard_write"](os.path.join(ns["RESULT_DIR"], "x")), os.path.join(ns["RESULT_DIR"], "x"))
        self.zip_kind("rehearsal")
        ns = self.run_setup(False)
        self.assertTrue(ns["REHEARSAL"])
        self.assertEqual(os.path.basename(ns["RESULT_ROOT"]), "test02_rehearsal")
        self.zip_kind("official")
        ns = self.run_setup(False)
        self.assertFalse(ns["REHEARSAL"])
        self.assertEqual(os.path.basename(ns["RESULT_ROOT"]), "test02")
        self.assertEqual(os.path.basename(ns["P02_ROOT"]), "prereg02")
        os.remove(os.path.join(self.drive, "C10_test_v1.zip"))      # no zip: official root, preflight stops later
        ns = self.run_setup(False)
        self.assertIsNone(ns["TEST_KIND"])
        self.assertEqual(os.path.basename(ns["RESULT_ROOT"]), "test02")


if __name__ == "__main__":
    unittest.main()
