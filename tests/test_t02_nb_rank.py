"""Tests for colab/build_test02_rank_nb.py and colab/test02_rank.ipynb (test02, competition 2).

* byte-identical rebuild of the committed notebook and diff file;
* segment identity with the base colab/prereg02_rank.ipynb (checked here independently of the builder);
* the validation label token appears exactly once (the inherited prepare-cell refusal line);
* the pins in the notebook agree with the committed prereg02 files;
* the functions defined in the new cells (quick format check, seam comparison, write guard) behave as the
  spec says, on synthetic inputs (executed from the cell sources; CPU torch only).
Run: python -m pytest tests/test_t02_nb_rank.py
"""

import ast
import builtins
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t02_rank  # noqa: E402

NB = os.path.join(REPO, "colab", "test02_rank.ipynb")
BASE = os.path.join(REPO, "colab", "prereg02_rank.ipynb")
DIFF = os.path.join(REPO, "results", "test02", "notebook_diff_rank.txt")
BUILDER = os.path.join(REPO, "colab", "build_test02_rank_nb.py")
P02 = os.path.join(REPO, "results", "prereg02", "rank")
TOKEN = "ans" + "wers"
BASE_SHA256 = "9fb1f9d06ba5ee5fbebfb6dfe022246888c93e7762e9a89e1adc99c9f1c12b0c"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def src(cell):
    return "".join(cell["source"])


def base_key(meta):
    if "new" in meta:
        return meta["new"]
    b = meta["base"]
    return "official_%d" % b["official_cell"] if "official_cell" in b else b["new"]


class Cells:
    nb = load(NB)
    base = load(BASE)

    @classmethod
    def new(cls, tag):
        hits = [c for c in cls.nb["cells"] if c["metadata"]["c10"].get("new") == tag]
        assert len(hits) == 1, tag
        return src(hits[0])

    @classmethod
    def reused(cls, key):
        hits = [c for c in cls.nb["cells"] if "base_cell" in c["metadata"]["c10"]
                and base_key(cls.base["cells"][c["metadata"]["c10"]["base_cell"]]["metadata"]["c10"]) == key]
        assert len(hits) == 1, key
        return src(hits[0])

    @classmethod
    def base_cell(cls, key):
        hits = [c for c in cls.base["cells"] if base_key(c["metadata"]["c10"]) == key]
        assert len(hits) == 1, key
        return src(hits[0])


def defs(source):
    tree = ast.parse(source)
    return tree, {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def exec_defs(source, names, env, extra_assign=()):
    """Execute only the named top-level function definitions (and assignments to extra_assign) of a cell."""
    tree = ast.parse(source)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in extra_assign)]
    assert {n.name for n in keep if isinstance(n, ast.FunctionDef)} == set(names), names
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<cell>", "exec"), env)
    return env


class Build(unittest.TestCase):
    def test_base_notebook_is_the_committed_prereg02_one(self):
        self.assertEqual(sha(BASE), BASE_SHA256)

    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb, diff = os.path.join(tmp, "nb.ipynb"), os.path.join(tmp, "diff.txt")
            r = subprocess.run([sys.executable, BUILDER, REPO, "--out-nb", nb, "--out-diff", diff],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertEqual(sha(nb), sha(NB))
            with open(diff, encoding="utf-8") as f1, open(DIFF, encoding="utf-8") as f2:
                a, b = f1.read().splitlines(), f2.read().splitlines()
            # the diff header names the output path's SHA-256 (same bytes), nothing else may differ
            self.assertEqual(a, b)

    def test_cells_and_ids(self):
        nb = Cells.nb
        ids = [c["id"] for c in nb["cells"]]
        self.assertEqual(ids, ["c10-rank-t02-%02d" % i for i in range(len(ids))])
        tags = [c["metadata"]["c10"].get("new") or base_key(Cells.base["cells"][c["metadata"]["c10"]["base_cell"]]
                                                               ["metadata"]["c10"]) for c in nb["cells"]]
        self.assertEqual(tags, ["header", "setup", "prepare", "official_1", "environment", "official_2",
                                "preflight_t02", "dump_tools", "dump_set_seam_val", "dump_set_public",
                                "dump_set_private", "seam_check", "final_check"])
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile(src(c), "<cell %d>" % i, "exec")
        self.assertEqual(nb["metadata"], Cells.base["metadata"])

    def test_unchanged_reused_cells_are_verbatim(self):
        for c in Cells.nb["cells"]:
            m = c["metadata"]["c10"]
            if "base_cell" in m and not m["changed"]:
                self.assertEqual(src(c), src(Cells.base["cells"][m["base_cell"]]))
        self.assertEqual(Cells.reused("official_1"), Cells.base_cell("official_1"))
        self.assertEqual(Cells.reused("environment"), Cells.base_cell("environment"))


class SegmentIdentity(unittest.TestCase):
    """The spec 2.0 list, checked here without the builder's helpers."""

    def test_functions(self):
        b, n = Cells.base_cell("dump_tools"), Cells.reused("dump_tools")
        (_bt, bd), (_nt, nd) = defs(b), defs(n)
        for name in ("d4", "load_net", "sgf_to_features", "dump_is_done", "write_dump", "write_keys", "pack_results",
                     "flush_drive", "rule_b", "_argmax_first", "_dump_path", "_marker_path"):
            self.assertEqual(ast.get_source_segment(b, bd[name]), ast.get_source_segment(n, nd[name]), name)

    def lines_between(self, source, start, end):
        """Full lines from the first line starting with `start` (stripped) to the first later line starting with
        `end` (inclusive)."""
        lines = source.splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.strip().startswith(start))
        j = next(k for k in range(i, len(lines)) if lines[k].strip().startswith(end))
        return lines[i:j + 1]

    def test_d4_self_check(self):
        b, n = Cells.base_cell("dump_tools"), Cells.reused("dump_tools")
        self.assertEqual(self.lines_between(b, "_t = torch.arange", "assert all(any("),
                         self.lines_between(n, "_t = torch.arange", "assert all(any("))

    def test_parse_loop_and_inference_loop(self):
        b, n = Cells.base_cell("dump_tools"), Cells.reused("dump_tools")
        self.assertEqual(self.lines_between(b, "sgf_cols = ", "parsed = "), self.lines_between(n, "sgf_cols = ", "parsed = "))
        # the first `with torch.no_grad():` of the base cell belongs to dump_inner (removed); compare dump_val's
        bv = b[b.index("def dump_val("):]
        nv = n[n.index("def dump_questions("):]
        self.assertEqual(self.lines_between(bv, "with torch.no_grad():", "logits[rows, s] = "),
                         self.lines_between(nv, "with torch.no_grad():", "logits[rows, s] = "))
        self.assertIn("for _qid, start, k in V['spans']:", "\n".join(self.lines_between(nv, "with torch.no_grad():",
                                                                                     "logits[rows, s] = ")))
        self.assertIn("pd.read_csv(path, dtype={'question_id': str})", n)
        self.assertIn("test = test.head(SMOKE_CFG['questions'])", n)

    def test_three_set_cells_share_one_body(self):
        cells = [Cells.new("dump_set_" + s) for s in ("seam_val", "public", "private")]
        self.assertEqual(len({c.split("\n", 1)[1] for c in cells}), 1)
        self.assertEqual([c.split("\n", 1)[0].split("'")[1] for c in cells], ["seam_val", "public", "private"])
        body = cells[0]
        self.assertIn("write_keys('val_keys'", body)
        self.assertIn("dump_questions(NET, _path, DUMP_NAME, DUMP_SYMS, W_LABEL, WSHA)", body)
        self.assertIn("load_net(W_PATH, expect_sha=WSHA)", body)
        self.assertIn("DUMP_DIR = guard_write(os.path.join(RESULT_DIR, SET, 'dumps'))", body)


class Token(unittest.TestCase):
    def test_exactly_the_inherited_refusal_line(self):
        hits = [(i, ln) for i, c in enumerate(Cells.nb["cells"]) for ln in src(c).splitlines() if TOKEN in ln.lower()]
        self.assertEqual(len(hits), 1, hits)
        line = hits[0][1]
        self.assertTrue(line.startswith("assert not any(") and "'%s' in" % TOKEN in line, line)
        self.assertIn(line, Cells.base_cell("prepare").splitlines())
        self.assertIn(line, Cells.reused("prepare").splitlines())
        with open(NB, encoding="utf-8") as f:
            self.assertEqual(f.read().lower().count(TOKEN), 1)


class Pins(unittest.TestCase):
    def constants(self):
        tree = ast.parse(Cells.new("preflight_t02"))
        out = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.endswith("_PIN"):
                out[n.targets[0].id] = ast.literal_eval(n.value)
        return out

    def test_pins_agree_with_the_committed_prereg02_files(self):
        c = self.constants()
        fc = load(os.path.join(P02, "frozen_config.json"))
        mk = load(os.path.join(P02, "dumps", "_DONE_val_p02e9_d4.json"))
        self.assertEqual(c["W_PIN"], fc["dumps"]["p02e9_d4"]["weights_sha256"])
        self.assertEqual(c["W_PIN"], "423581b4fddd5c2c160ebf691f4e86b9e00c4fe743d5cfb23df7891f4a2ca939")
        self.assertEqual(c["W_LABEL_PIN"], mk["weights"])
        self.assertEqual(c["P02_REF_GPU_PIN"], mk["gpu_name"])
        self.assertEqual(c["SEAM_REF_SHA256_PIN"],
                         {n: sha(os.path.join(P02, "dumps", n)) for n in ("val_keys.npz", "val_p02e9_d4.npz")})
        self.assertEqual(c["SEAM_REF_SHA256_PIN"]["val_keys.npz"],
                         "055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c")
        with np.load(os.path.join(P02, "dumps", "val_p02e9_d4.npz"), allow_pickle=False) as z:
            self.assertEqual(str(z["weights_sha256"]), c["W_PIN"])

    def test_official_names_and_zip_contract(self):
        setup = Cells.reused("setup")
        tree = ast.parse(setup)
        consts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id in ("OFFICIAL_TEST_NAMES", "TEST_FILES", "TEST_MANIFEST_FORMAT",
                                          "TEST_MANIFEST_MEMBER")}
        self.assertEqual(consts["TEST_FILES"], t02_rank.TEST_FILES)
        self.assertEqual(len(consts["OFFICIAL_TEST_NAMES"]), 6)
        self.assertLessEqual(set(t02_rank.TEST_FILES.values()), set(consts["OFFICIAL_TEST_NAMES"]))
        self.assertEqual(consts["TEST_MANIFEST_FORMAT"], "c10-test02 manifest v1")
        self.assertEqual(consts["TEST_MANIFEST_MEMBER"], "tests/MANIFEST.json")
        self.assertIn("os.path.join(DRIVE_DIR, 'C10_test_v1.zip')", setup)
        for root in ("'test02_smoke'", "'test02_rehearsal'", "'test02'", "'prereg01_smoke'", "'prereg02_smoke'"):
            self.assertIn(root, setup)
        final = Cells.reused("final_check")
        self.assertIn("pack_results('test02_rank_results.zip')", final)
        self.assertIn("pack_results('test02_rank_results.zip')", Cells.new("seam_check"))
        for status in ("'completed'", "'readonly_changed'"):
            self.assertIn(status, final)
        self.assertIn("'seam_failed'", Cells.new("seam_check"))

    def test_notebook_sha256_rule(self):
        recorded, recomputed = t02_rank.notebook_source_sha256(NB)
        self.assertIsNotNone(recorded)
        self.assertEqual(recorded, recomputed)
        self.assertIn(recorded, open(DIFF, encoding="utf-8").read())

    def test_gpu_gate_and_no_training(self):
        off2 = Cells.reused("official_2")
        self.assertIn("GPU_MODELS_OK = ('RTX PRO 6000', 'A100')", off2)
        self.assertIn("if not SMOKE and DEVICE.type != 'cuda':", off2)
        whole = "\n".join(src(c) for c in Cells.nb["cells"] if c["cell_type"] == "code")
        names = {n.id for c in Cells.nb["cells"] if c["cell_type"] == "code"
                 for n in ast.walk(ast.parse(src(c))) if isinstance(n, ast.Name)}
        for gone in ("FINISH_WITHOUT_TRAINING", "df_val", "df_train", "CKPT_DIR", "RESULT_CKPT", "load_rows",
                     "EXTRA_EPOCHS", "dump_inner", "parse_val", "dump_val", "P01_BEFORE"):
            self.assertNotIn(gone, names, gone)
        self.assertNotIn("tarfile.open", whole)
        self.assertNotIn("optimizer.step", whole)

    def test_every_loaded_name_is_bound_somewhere(self):
        """Coarse NameError guard: every global name the code cells load is bound in some cell or is a builtin."""
        bound, loaded = set(dir(builtins)) | {"drive"}, set()
        for c in Cells.nb["cells"]:
            if c["cell_type"] != "code":
                continue
            for n in ast.walk(ast.parse(src(c))):
                if isinstance(n, ast.Name):
                    (bound if isinstance(n.ctx, (ast.Store, ast.Del)) else loaded).add(n.id)
                elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    bound.add(n.name)
                elif isinstance(n, ast.arg):
                    bound.add(n.arg)
                elif isinstance(n, ast.alias):
                    bound.add((n.asname or n.name).split(".")[0])
                elif isinstance(n, ast.ExceptHandler) and n.name:
                    bound.add(n.name)
        self.assertEqual(sorted(loaded - bound), [])


class CellFunctions(unittest.TestCase):
    """The new cells' functions on synthetic inputs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t02_nb_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- quick format check (preflight)
    def questions_file(self, rows, bom=False, name="q.csv"):
        header = ["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)] + ["color_%d" % i for i in range(1, 21)]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(header)
        for q, n, cols in rows:
            w.writerow([q, n] + ["(;B[pd];W[dp])" if i < n else "" for i in range(20)] + cols + [""] * (20 - len(cols)))
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write((b"\xef\xbb\xbf" if bom else b"") + buf.getvalue().encode())
        return path

    def quick(self):
        env = {"csv": csv, "io": io, "N_TEST_QUESTIONS": 400}
        return exec_defs(Cells.new("preflight_t02"), ["quick_check_questions"], env)["quick_check_questions"]

    def good_rows(self, prefix, n=400):
        return [("%sq_%04d" % (prefix, i + 1), 5 + i % 16, ["BW"[j % 2] for j in range(5 + i % 16)])
                for i in range(n)]

    def test_quick_check(self):
        qc = self.quick()
        problems, ids = qc(self.questions_file(self.good_rows("pub_")), "pub_")
        self.assertEqual(problems, [])
        self.assertEqual(len(ids), 400)
        cases = {
            "bom": (self.good_rows("pub_"), True),
            "399 rows": (self.good_rows("pub_", 399), False),
            "wrong prefix": (self.good_rows("priv_"), False),
            "num_games 4": ([("pub_q_0001", 4, ["B"] * 4)] + self.good_rows("pub_")[1:], False),
            "lower-case colour": ([("pub_q_0001", 5, ["b"] * 5)] + self.good_rows("pub_")[1:], False),
            "duplicate id": ([self.good_rows("pub_")[1]] + self.good_rows("pub_")[1:], False),
        }
        for label, (rows, bom) in cases.items():
            with self.subTest(label):
                problems, _ = qc(self.questions_file(rows, bom=bom), "pub_")
                self.assertTrue(problems, label)
        # a filled slot after num_games
        path = self.questions_file(self.good_rows("pub_"))
        text = open(path, encoding="utf-8").read().splitlines()
        cells = text[1].split(",")
        cells[2 + 7] = "(;B[aa])"            # sgf_8 of a question with num_games 5
        text[1] = ",".join(cells)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(text) + "\n")
        self.assertTrue(qc(path, "pub_")[0])
        # a missing column
        path = os.path.join(self.tmp, "cols.csv")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(",".join(["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)]
                             + ["color_%d" % i for i in range(1, 20)]) + "\n")
        problems, _ids = qc(path, "pub_")
        self.assertTrue(problems and "color_20" in problems[0], problems)

    # ---- the seam comparison
    def seam_env(self):
        import torch
        dt = Cells.reused("dump_tools")
        env = {"np": np, "torch": torch, "math": math, "LEVELS": ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]}
        exec_defs(dt, ["_argmax_first", "rule_b"], env, extra_assign=("_E1",))
        exec_defs(Cells.new("seam_check"), ["question_spans", "c0_of", "seam_compare"], env, extra_assign=("SEAM_KEYS",))
        return env

    def dumps(self, n_q=30, seed=0):
        rng = np.random.default_rng(seed)
        qid, slot, color, parsed = [], [], [], []
        for q in range(n_q):
            k = 5 + q % 4
            for s in range(1, k + 1):
                qid.append("val_q_%04d" % (q + 1))
                slot.append(s)
                color.append("BW"[s % 2])
                parsed.append(not (q == 2 or (q == 4 and s == 1)))
        keys = {"question_id": np.array(qid), "slot": np.array(slot, dtype=np.int32), "color": np.array(color),
                "parsed": np.array(parsed)}
        L = rng.normal(0, 2, size=(len(qid), 8, 10)).astype(np.float32)
        L[~keys["parsed"]] = np.nan
        return keys, L

    def test_seam_compare(self):
        env = self.seam_env()
        sc = env["seam_compare"]
        keys, L = self.dumps()
        ref = {"keys": keys, "logits": L}
        r = sc({"keys": dict(keys), "logits": L.copy()}, ref, True, False)
        self.assertTrue(r["passed"] and r["bitwise_equal"])
        self.assertEqual((r["tol"], r["max_abs_c0_diff"], r["max_abs_logit_diff"]), (1e-4, 0.0, 0.0))
        self.assertEqual(r["rule_b_diff"]["allowed"], 4)
        # a small drift passes on another GPU (1e-3) but not on the same one (1e-4)
        L2 = L.copy()
        L2[0, 0, :] += np.linspace(0, 0.05, 10, dtype=np.float32)
        same, other = sc({"keys": keys, "logits": L2}, ref, True, False), sc({"keys": keys, "logits": L2}, ref, False, False)
        self.assertFalse(same["bitwise_equal"])
        self.assertTrue(1e-4 < same["max_abs_c0_diff"] < 1e-3, same["max_abs_c0_diff"])
        self.assertFalse(same["passed"])
        self.assertTrue(other["passed"])
        # only the original orientation enters C0; other symmetries are recorded in max_abs_logit_diff
        L3 = L.copy()
        L3[:, 3, :] += 1.0
        r3 = sc({"keys": keys, "logits": L3}, ref, True, False)
        self.assertTrue(r3["passed"])
        self.assertAlmostEqual(r3["max_abs_logit_diff"], 1.0, places=5)
        # different key arrays fail
        k2 = dict(keys, slot=keys["slot"].copy())
        k2["slot"][0] = 9
        self.assertFalse(sc({"keys": k2, "logits": L}, ref, False, False)["passed"])
        # rule-B differences: at most 4 of 1,589 (smoke: scaled, at least 1)
        spans = env["question_spans"](keys["question_id"])
        flipped = 0
        L4 = L.copy()
        for q, start, k in spans:
            ok = [start + j for j in range(k) if keys["parsed"][start + j]]
            if ok and flipped < 5:
                L4[ok, 0, :] = L4[ok, 0, ::-1]
                if env["rule_b"](env["c0_of"](L4, [[q, start, k]], keys["parsed"])[0]) != \
                        env["rule_b"](env["c0_of"](L, [[q, start, k]], keys["parsed"])[0]):
                    flipped += 1
                else:
                    L4[ok, 0, :] = L[ok, 0, :]
        r4 = sc({"keys": keys, "logits": L4}, ref, False, False)
        self.assertEqual(r4["rule_b_diff"]["questions"], 5)
        self.assertGreater(r4["max_abs_c0_diff"], 1e-3)
        smoke = sc({"keys": keys, "logits": L.copy()}, ref, True, True)
        self.assertEqual(smoke["rule_b_diff"]["allowed"], max(1, math.ceil(4 * 30 / 1589)))

    def test_c0_is_the_float32_mean_softmax(self):
        env = self.seam_env()
        keys, L = self.dumps(n_q=3)
        spans = env["question_spans"](keys["question_id"])
        c0 = env["c0_of"](L, spans, keys["parsed"])
        self.assertIsNone(c0[2])                                   # question 3 has no parsed game
        q, start, k = spans[0]
        z = L[start:start + k, 0, :].astype(np.float32)
        e = np.exp(z - z.max(1, keepdims=True))
        want = (e / e.sum(1, keepdims=True)).astype(np.float32).mean(0)
        np.testing.assert_allclose(c0[0], want, atol=1e-6)
        with self.assertRaises(AssertionError):
            env["question_spans"](np.array(["a", "b", "a"]))       # a question's slots must be contiguous

    # ---- the write guard and fingerprints (setup)
    def test_guard_write_refuses_both_readonly_roots(self):
        setup = Cells.reused("setup")
        p01, p02, t02 = (os.path.join(self.tmp, "results", n) for n in ("prereg01", "prereg02", "test02"))
        env = {"os": os, "READONLY_ROOTS": {"prereg01": p01, "prereg02": p02}}
        exec_defs(setup, ["_real", "_is_under", "guard_write", "readonly_fingerprint"], env)
        for p in (p01, os.path.join(p01, "rank", "x.json"), p02, os.path.join(p02, "rank", "dumps", "y.npz")):
            with self.assertRaises(PermissionError):
                env["guard_write"](p)
        self.assertEqual(env["guard_write"](os.path.join(t02, "rank")), os.path.join(t02, "rank"))
        self.assertEqual(env["guard_write"](os.path.join(self.tmp, "results", "prereg02_x")),
                         os.path.join(self.tmp, "results", "prereg02_x"))
        os.makedirs(os.path.join(p02, "rank"))
        with open(os.path.join(p02, "rank", "a.txt"), "w") as f:
            f.write("x")
        fp = env["readonly_fingerprint"](p02)
        self.assertEqual(list(fp), ["rank/a.txt"])
        self.assertEqual(fp["rank/a.txt"][0], 1)


if __name__ == "__main__":
    unittest.main()
