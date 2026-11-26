"""Tests for colab/build_test04_rank_nb.py and colab/test04_rank.ipynb (test04, competition 2; spec
results/test04/design/spec.md 4 and 10.2).

* Build: the two base notebooks are the committed ones; a rebuild is byte-identical (notebook and diff file); the
  notebook JSON is what nbformat writes; cell ids, compile, no outputs; the validation label token only in the
  inherited prepare-cell refusal line; unchanged base cells verbatim; CHANGE_ROWS equals the real changes.
* SegmentIdentity: the spec 4.1 segments, recomputed here with ast (independently of the builder).
* Pins: every pinned value of the notebook equals the value read from the committed repository files; the
  notebook_sha256 rule; the upload copy.
* NoTraining: no training names, no autocast, no tarfile.open, bfloat16 only inside state_dict_sha256; every
  loaded name is bound.
* CellFunctions (CPU torch): dump_val_p03 through the QUESTIONS_PATH adapter equals prereg03's own function byte
  for byte; swap / s0 are the registered views; guard_write; the seam functions on synthetic arrays.
* Preflight: the setup, official-2 and preflight cells executed on a fake Drive tree (SMOKE), every stop of
  spec 4.3 point 6 and the test02 record; the read-only roots (test02/player may change, test02/rank may not);
  the seam and final cells on fake dumps (pass, seam failure, final-check failure).
* Adapter: the set cells' QUESTIONS_PATH assertion, marker rows and 'written' checks.
Run: python -m pytest -q tests/test_t04_nb.py
"""

import ast
import builtins
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
import unittest
import zipfile
from multiprocessing.dummy import Pool as ThreadPool
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import t02_rank as t2  # noqa: E402

NB = os.path.join(REPO, "colab", "test04_rank.ipynb")
T02 = os.path.join(REPO, "colab", "test02_rank.ipynb")
P03 = os.path.join(REPO, "colab", "prereg03_rank_a.ipynb")
P03B = os.path.join(REPO, "colab", "prereg03_rank_b.ipynb")
DIFF = os.path.join(REPO, "results", "test04", "notebook_diff_rank.txt")
BUILDER = os.path.join(REPO, "colab", "build_test04_rank_nb.py")
SMOKE_TREE = os.path.join(REPO, "colab", "make_test04_smoke_tree.py")
UPLOAD = os.path.join(REPO, "data", "colab_upload", "test04", "test04_rank.ipynb")
TUTORIAL = os.path.join(REPO, "third_party", "aicup2026_tutorial")
P03_RESULTS = os.path.join(REPO, "results", "prereg03")
TOKEN = "ans" + "wers"
T02_SHA256 = "53337a7acc396bdc1f94b3cbdc7dd0ce859766511099e8004697ff0d6a487b05"
P03_SHA256 = "85e0c2ae3269c68ad166310efb2525b2d48ed1279c9cf6eb6b428e59e8693557"
P03B_SHA256 = "aa3669accceca23fc3ab2439bcab68204281bcec999cd1e00d43dd16825bd8b3"
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
ARMS = ("a", "b")
DUMP_NAMES = {"a": "p03a", "b": "p03b"}
SETS = ("seam_val", "public", "private")
KEY_ORDER = ["header", "setup", "prepare", "official_1", "environment", "official_2", "preflight_t04", "view_d4",
             "model_tools", "dump_tools", "dump_set_seam_val", "dump_set_public", "dump_set_private", "seam_check",
             "final_check"]


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


def cell_key(cell):
    """Key of a cell of test02_rank.ipynb / prereg03_rank_a.ipynb."""
    m = cell["metadata"]["c10"]
    return base_key(m["base"]) if "base_cell" in m else m["new"]


def t04_key(cell):
    """Key of a cell of test04_rank.ipynb (the base cell's key, or the new tag)."""
    m = cell["metadata"]["c10"]
    return cell_key({"metadata": {"c10": m["base"]}}) if "base_cell" in m else m["new"]


class Cells:
    nb = load(NB)
    t02 = load(T02)
    p03 = load(P03)

    @classmethod
    def out(cls, key):
        hits = [src(c) for c in cls.nb["cells"] if t04_key(c) == key]
        assert len(hits) == 1, key
        return hits[0]

    @classmethod
    def base(cls, which, key):
        nb = cls.t02 if which == "test02" else cls.p03
        hits = [src(c) for c in nb["cells"] if cell_key(c) == key]
        assert len(hits) == 1, (which, key)
        return hits[0]


def defs(source):
    tree = ast.parse(source)
    return {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def code_tokens(segment):
    return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(segment).readline)
            if t.type not in (tokenize.COMMENT, tokenize.NL)]


def exec_defs(source, names, env, assigns=()):
    """Execute only the named top-level function definitions (and single-name assignments) of a cell."""
    tree = ast.parse(source)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in assigns)]
    got = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in keep}
    assert got == set(names) | set(assigns), (sorted(set(names) | set(assigns) - got))
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<cell>", "exec"), env)
    return env


def load_builder():
    spec = importlib.util.spec_from_file_location("build_test04_rank_nb", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)   # import-safe: building and writing happen only in main()
    finally:
        sys.dont_write_bytecode = old
    return mod


def preflight_constants():
    out = {}
    for n in ast.parse(Cells.out("preflight_t04")).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.endswith("_PIN"):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


# ================================================================================================= Build
class Build(unittest.TestCase):
    def test_base_notebooks_are_the_committed_ones(self):
        for path, want in ((T02, T02_SHA256), (P03, P03_SHA256), (P03B, P03B_SHA256)):
            with open(path, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw, path)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), want, path)

    def test_prereg03_arms_differ_in_the_arm_line_only(self):
        la = open(P03, encoding="utf-8").read().split("\n")
        lb = open(P03B, encoding="utf-8").read().split("\n")
        self.assertEqual(len(la), len(lb))
        diff = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
        self.assertEqual(len(diff), 1)
        self.assertEqual(la[diff[0]].replace("ARM = 'A'", "ARM = 'B'"), lb[diff[0]])

    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb, diff = os.path.join(tmp, "nb.ipynb"), os.path.join(tmp, "diff.txt")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run([sys.executable, BUILDER, REPO, "--out-nb", nb, "--out-diff", diff],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertEqual(sha(nb), sha(NB))
            self.assertEqual(sha(diff), sha(DIFF))

    def test_the_writer_is_nbformat_layout(self):
        import nbformat
        with open(NB, "rb") as f:
            raw = f.read()
        nb = nbformat.reads(raw.decode("utf-8"), as_version=4)
        nbformat.validate(nb)
        buf = io.StringIO()
        nbformat.write(nb, buf)
        self.assertEqual(buf.getvalue().encode("utf-8"), raw)
        self.assertNotIn(b"\r\n", raw)

    def test_cells_ids_and_compile(self):
        nb = Cells.nb
        self.assertEqual([c["id"] for c in nb["cells"]], ["c10-rank-t04-%02d" % i for i in range(len(nb["cells"]))])
        self.assertEqual([t04_key(c) for c in nb["cells"]], KEY_ORDER)
        want_base = {"setup": "test02", "prepare": "test02", "official_1": "test02", "environment": "test02",
                     "official_2": "test02", "view_d4": "prereg03", "dump_tools": "test02", "final_check": "test02"}
        for c in nb["cells"]:
            m = c["metadata"]["c10"]
            k = t04_key(c)
            if k in want_base:
                self.assertEqual(m["base_nb"], want_base[k], k)
                base = Cells.t02 if m["base_nb"] == "test02" else Cells.p03
                self.assertEqual(base["cells"][m["base_cell"]]["id"], m["base_id"])
                self.assertEqual(base["cells"][m["base_cell"]]["metadata"]["c10"], m["base"])
            else:
                self.assertEqual(m, {"new": k})
            if c["cell_type"] == "code":
                compile(src(c), c["id"], "exec")
                self.assertEqual(c["outputs"], [])
                self.assertIsNone(c["execution_count"])
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
        self.assertEqual(nb["metadata"], Cells.t02["metadata"])
        self.assertTrue(nb["cells"][-1]["metadata"]["c10"].get("rewritten"))

    def test_unchanged_reused_cells_are_verbatim(self):
        for c in Cells.nb["cells"]:
            m = c["metadata"]["c10"]
            if "base_cell" in m:
                base = Cells.t02 if m["base_nb"] == "test02" else Cells.p03
                self.assertEqual(src(c) != src(base["cells"][m["base_cell"]]), m["changed"], c["id"])
        for key in ("prepare", "official_1", "environment"):
            self.assertEqual(Cells.out(key), Cells.base("test02", key), key)
        self.assertEqual(Cells.out("view_d4"), Cells.base("prereg03", "view_d4"))

    def test_change_rows_are_the_real_changes(self):
        b = load_builder()
        changed = {(c["metadata"]["c10"]["base_nb"], t04_key(c)) for c in Cells.nb["cells"]
                   if "base_cell" in c["metadata"]["c10"] and c["metadata"]["c10"]["changed"]}
        self.assertEqual(changed, {k for k, _l, _t in b.CHANGE_ROWS if k is not None})
        self.assertEqual(changed, {("test02", "setup"), ("test02", "official_2"), ("test02", "dump_tools"),
                                   ("test02", "final_check")})
        setup = Cells.out("setup")
        t04_changes = [ast.literal_eval(n.value) for n in ast.parse(setup).body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "T04_CHANGES"]
        self.assertEqual(t04_changes, [["[%s] %s" % (label, text) for _k, label, text in b.CHANGE_ROWS]])
        header = src(Cells.nb["cells"][0])
        for _k, label, text in b.CHANGE_ROWS:
            self.assertIn("| %s | %s |" % (label, text.replace("|", "\\|")), header)

    def test_token_only_in_the_inherited_refusal_line(self):
        hits = [(t04_key(c), ln) for c in Cells.nb["cells"] for ln in src(c).splitlines() if TOKEN in ln.lower()]
        self.assertEqual(len(hits), 1, hits)
        key, line = hits[0]
        self.assertEqual(key, "prepare")
        self.assertTrue(line.startswith("assert not any(") and "'%s' in" % TOKEN in line, line)
        self.assertIn(line, Cells.base("test02", "prepare").splitlines())
        self.assertEqual(open(NB, encoding="utf-8").read().lower().count(TOKEN), 1)
        self.assertEqual(open(DIFF, encoding="utf-8").read().lower().count(TOKEN), 0)
        for path in (BUILDER, SMOKE_TREE, os.path.abspath(__file__)):
            self.assertNotIn(TOKEN, open(path, encoding="utf-8").read().lower(), path)

    def test_generators_do_not_import_frozen_code(self):
        for path in (BUILDER, SMOKE_TREE):
            tree = ast.parse(open(path, encoding="utf-8").read())
            mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse({m for m in mods if m.startswith(("build_", "p0", "t02_", "t04_"))}, (path, mods))
        tree = ast.parse(open(BUILDER, encoding="utf-8").read())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertEqual(mods - set(sys.stdlib_module_names), set(), "the builder uses the standard library only")

    def test_diff_file_header(self):
        text = open(DIFF, encoding="utf-8").read()
        self.assertIn("colab/test02_rank.ipynb  sha256=%s" % T02_SHA256, text)
        self.assertIn("colab/prereg03_rank_a.ipynb  sha256=%s" % P03_SHA256, text)
        self.assertIn("colab/test04_rank.ipynb  sha256=%s" % sha(NB), text)


# ================================================================================================= SegmentIdentity
class SegmentIdentity(unittest.TestCase):
    """Spec 4.1, recomputed here with ast (not with the builder's helpers)."""

    def assertSegments(self, base_src, new_src, names, exact=True):
        b, n = defs(base_src), defs(new_src)
        for name in names:
            self.assertIn(name, b, name)
            self.assertIn(name, n, name)
            if exact:
                self.assertEqual(b[name], n[name], name)
            else:
                self.assertEqual(code_tokens(b[name]), code_tokens(n[name]), name)

    def test_view_d4_cell_is_prereg03s(self):
        self.assertEqual(Cells.out("view_d4"), Cells.base("prereg03", "view_d4"))
        self.assertIn("d4", defs(Cells.out("view_d4")))

    def test_prereg03_segments(self):
        self.assertSegments(Cells.base("prereg03", "dump_tools"), Cells.out("dump_tools"), ("load_net", "dump_val_p03"))
        self.assertSegments(Cells.base("prereg03", "official_14"), Cells.out("model_tools"),
                            ("state_dict_sha256", "_nchw_output", "channels_last"))
        self.assertEqual(set(defs(Cells.out("model_tools"))), {"state_dict_sha256", "_nchw_output", "channels_last"})

    def test_test02_segments(self):
        self.assertSegments(Cells.base("test02", "dump_tools"), Cells.out("dump_tools"),
                            ("write_dump", "dump_is_done", "write_keys", "require_gpu_for", "pack_results", "flush_drive",
                             "sgf_to_features", "parse_questions", "_argmax_first", "rule_b", "_dump_path",
                             "_marker_path"))
        self.assertSegments(Cells.base("test02", "setup"), Cells.out("setup"),
                            ("readonly_fingerprint", "_is_under", "_real", "sha256_path", "gpu_name_of", "write_json",
                             "read_json", "read_runs"))
        self.assertSegments(Cells.base("test02", "preflight_t02"), Cells.out("preflight_t04"), ("quick_check_questions",))
        self.assertSegments(Cells.base("test02", "seam_check"), Cells.out("seam_check"),
                            ("question_spans", "c0_of", "_load_npz"))

    def test_segments_equal_up_to_comments(self):
        self.assertSegments(Cells.base("test02", "setup"), Cells.out("setup"), ("guard_write",), exact=False)
        self.assertSegments(Cells.base("test02", "dump_tools"), Cells.out("dump_tools"), ("record_readonly_after",),
                            exact=False)

    def test_dropped_and_added_functions(self):
        dt = defs(Cells.out("dump_tools"))
        self.assertNotIn("d4", dt)
        self.assertNotIn("dump_questions", dt)
        self.assertIn("parse_val", dt)
        tree = ast.parse(dt["parse_val"])
        self.assertEqual(ast.dump(tree.body[0].body[-1]),
                         ast.dump(ast.parse("return parse_questions(QUESTIONS_PATH)").body[0]))

    def test_prepare_cell_is_test02s(self):
        self.assertEqual(Cells.out("prepare"), Cells.base("test02", "prepare"))

    def test_three_set_cells_share_one_body(self):
        cells = [Cells.out("dump_set_" + s) for s in SETS]
        self.assertEqual(len({c.split("\n", 1)[1] for c in cells}), 1)
        self.assertEqual([c.split("\n", 1)[0].split("'")[1] for c in cells], list(SETS))


# ================================================================================================= Pins
class Pins(unittest.TestCase):
    def test_weights(self):
        c = preflight_constants()
        fuse = ast.parse(open(os.path.join(REPO, "tools", "p04_rank_fuse.py"), encoding="utf-8").read())
        sw_weights = [ast.literal_eval(n.value) for n in fuse.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "SW_WEIGHTS"][0]
        for x in ARMS:
            d = os.path.join(P03_RESULTS, "rank_" + x)
            cfg = load(os.path.join(d, "config.json"))
            self.assertEqual(c["W_PIN"][x], cfg["final_sha256"])
            self.assertEqual(c["W_PIN"][x], sw_weights[x])
            for v in ("d4", "s0_swap"):
                m = load(os.path.join(d, "dumps", "_DONE_val_%s_%s.json" % (DUMP_NAMES[x], v)))
                self.assertEqual(m["weights_sha256"], c["W_PIN"][x])
                self.assertEqual(m["weights"], c["W_LABEL_PIN"][x])
                with np.load(os.path.join(d, "dumps", "val_%s_%s.npz" % (DUMP_NAMES[x], v)), allow_pickle=False) as z:
                    self.assertEqual(str(z["weights_sha256"]), c["W_PIN"][x])
            self.assertEqual(c["W_LABEL_PIN"][x], "prereg03/rank_%s/ckpt/final.pth" % x)
        self.assertEqual(c["W_PIN"], {"a": "790b767c5425e262130d6c103c71e89238edb17f927fa48e2a2770db788e50ac",
                                      "b": "8e88f6ec6cbaad5372e5d62607b99ece3d77dc7a20af86a9197553f1a6c59a42"})

    def test_seam_reference(self):
        c = preflight_constants()
        fuse = ast.parse(open(os.path.join(REPO, "tools", "p04_rank_fuse.py"), encoding="utf-8").read())
        sw_sha = [ast.literal_eval(n.value) for n in fuse.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "SW_SHA256"][0]
        prereg04 = open(os.path.join(REPO, "docs", "PREREG_04_OPPONENT.md"), encoding="utf-8").read().splitlines()
        for x in ARMS:
            dumps = os.path.join(P03_RESULTS, "rank_" + x, "dumps")
            names = ("val_keys.npz", "val_%s_d4.npz" % DUMP_NAMES[x], "val_%s_s0_swap.npz" % DUMP_NAMES[x])
            self.assertEqual(c["SEAM_REF_SHA256_PIN"][x], {n: sha(os.path.join(dumps, n)) for n in names})
            for n in names:
                self.assertEqual(load(os.path.join(dumps, "_DONE_%s.json" % n[:-4]))["sha256"],
                                 c["SEAM_REF_SHA256_PIN"][x][n])
            swap = c["SEAM_REF_SHA256_PIN"][x][names[2]]
            self.assertEqual(swap, sw_sha[x]["val"])
            self.assertIn("| `rank_%s/dumps/%s` | `%s` |" % (x, names[2], swap), prereg04)
        self.assertEqual(c["SEAM_REF_SHA256_PIN"]["a"]["val_keys.npz"],
                         "055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c")

    def test_reference_environment_and_tf32(self):
        c = preflight_constants()
        for x in ARMS:
            m = load(os.path.join(P03_RESULTS, "rank_" + x, "dumps", "_DONE_val_%s_d4.json" % DUMP_NAMES[x]))
            self.assertEqual(c["P03_REF_ENV_PIN"], {k: m[k] for k in ("gpu_name", "torch", "cudnn")})
            cfg = load(os.path.join(P03_RESULTS, "rank_" + x, "config.json"))
            self.assertEqual(c["TF32_P03_PIN"], cfg["runs"][0]["training"]["tf32"])
        self.assertEqual(c["P03_REF_ENV_PIN"]["gpu_name"], "NVIDIA RTX PRO 6000 Blackwell Server Edition")
        self.assertEqual(c["TF32_P03_PIN"], {"cudnn": True, "matmul": False})

    def test_bundle_manifest(self):
        c = preflight_constants()
        env = load(os.path.join(REPO, "results", "prereg02", "rank", "env.json"))
        for r in env["runs"]:
            self.assertEqual(c["BUNDLE_FILES_PIN"], r["bundle_manifest"]["files"])
        self.assertEqual(c["BUNDLE_FILES_PIN"]["local/rank_val_questions.csv"],
                         "73e51bbfd7b051e73cbb62af4b180c403d2422f122240c5e67e9c92303776db5")

    def test_notebook_sha256_rule(self):
        recorded, recomputed = t2.notebook_source_sha256(NB)
        self.assertIsNotNone(recorded)
        self.assertEqual(recorded, recomputed)
        self.assertIn(recorded, open(DIFF, encoding="utf-8").read())
        self.assertIn("NOTEBOOK_REL = 'colab/test04_rank.ipynb'", Cells.out("setup"))

    def test_base_notebook_shas_recorded(self):
        setup = Cells.out("setup")
        self.assertIn("'colab/test02_rank.ipynb': '%s'" % T02_SHA256, setup)
        self.assertIn("'colab/prereg03_rank_a.ipynb': '%s'" % P03_SHA256, setup)
        self.assertIn("'format': 'c10-test04 rank notebook config v1'", setup)

    def test_upload_copy(self):
        if not os.path.exists(UPLOAD):
            self.skipTest("data/colab_upload/test04/test04_rank.ipynb not written (builder --upload-copy)")
        self.assertEqual(sha(UPLOAD), sha(NB))
        self.assertEqual(os.listdir(os.path.dirname(UPLOAD)), ["test04_rank.ipynb"])


# ================================================================================================= NoTraining
class NoTraining(unittest.TestCase):
    def code(self):
        return [src(c) for c in Cells.nb["cells"] if c["cell_type"] == "code"]

    def test_no_training_names(self):
        names = {n.id for s in self.code() for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Name)}
        for gone in ("optimizer", "df_train", "df_val", "T_TOTAL", "make_train_loader", "load_rows", "CKPT_DIR",
                     "RESULT_CKPT", "NET", "VAL", "dump_questions", "dump_inner_p03", "ARM"):
            self.assertNotIn(gone, names, gone)

    def test_no_autocast_bf16_or_training_archive(self):
        attrs = [(n.value.id if isinstance(n.value, ast.Name) else None, n.attr)
                 for s in self.code() for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Attribute)]
        self.assertFalse([a for a in attrs if a[1] == "autocast"])
        self.assertNotIn(("tarfile", "open"), attrs)
        # bfloat16 only as the dtype test inside the verbatim state_dict_sha256 (hashing, never inference)
        self.assertEqual([a for a in attrs if a[1] == "bfloat16"], [("torch", "bfloat16")])
        self.assertIn("torch.bfloat16", defs(Cells.out("model_tools"))["state_dict_sha256"])

    def test_every_loaded_name_is_bound_somewhere(self):
        bound, loaded = set(dir(builtins)) | {"drive"}, set()
        for s in self.code():
            for n in ast.walk(ast.parse(s)):
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

    def test_gpu_gate(self):
        off2 = Cells.out("official_2")
        self.assertIn("GPU_MODELS_OK = ('RTX PRO 6000', 'A100')", off2)
        self.assertIn("if not SMOKE and DEVICE.type != 'cuda':", off2)
        self.assertIn("GPU_MODEL_SHOWN = GPU_MODEL_NOW or ('查不到 GPU 型號' if DEVICE.type == 'cuda' else None)", off2)
        self.assertIn("IN_CHANNELS = 501", off2)


# ================================================================================================= CellFunctions
def write_questions(path, questions):
    """questions: [(question_id, [(sgf, colour), ...])] -> a rank question file (20 sgf / colour columns)."""
    header = ["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)] + ["color_%d" % i for i in range(1, 21)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for q, games in questions:
            n = len(games)
            w.writerow([q, n] + [g for g, _c in games] + [""] * (20 - n) + [c for _g, c in games] + [""] * (20 - n))


SGF_MOVES = ["pd", "dp", "pp", "dd", "qf", "fq", "cn", "nc", "jj", "kk", "qq", "cc"]


def small_sgf(i):
    moves = SGF_MOVES[i % 5:] + SGF_MOVES[:i % 5]
    body = "".join(";%s[%s]" % ("BW"[j % 2], m) for j, m in enumerate(moves[:4 + i % 7]))
    return "(;GM[1]FF[4]SZ[19]%s)" % body


def common_env(torch, smoke=True):
    env = {"np": np, "torch": torch, "os": os, "io": io, "json": json, "time": time, "math": math,
           "hashlib": hashlib, "datetime": datetime, "Pool": ThreadPool, "NPROC": 2, "SMOKE": smoke,
           "DEVICE": torch.device("cpu"), "IN_CHANNELS": 501, "NUM_CLASSES": 10, "LEVELS": list(LEVELS),
           "RUN": {"run": 1}, "ENV_INFO": {"gpu": None, "cudnn": None}, "READONLY_ROOTS": {},
           "save_config": lambda **kw: None, "mark_time": lambda *a, **k: None, "print": lambda *a, **k: None}
    import pandas as pd
    env["pd"] = pd
    if TUTORIAL not in sys.path:
        sys.path.insert(0, TUTORIAL)
    from utils import SGFParseRankPrediction
    env["SGFParseRankPrediction"] = SGFParseRankPrediction
    return env


class CellFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        if TUTORIAL not in sys.path:
            sys.path.insert(0, TUTORIAL)
        from network import GoRankResNet
        cls.GoRankResNet = GoRankResNet
        cls.tmp = tempfile.mkdtemp(prefix="c10_t04_nb_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def notebook_env(self, which, qpath, dump_dir):
        """prereg03's own dump code (which='p03') or the test04 cells (which='t04') on CPU."""
        torch = self.torch
        env = common_env(torch)
        if which == "p03":
            setup, view, off14, dt = (Cells.base("prereg03", k) for k in ("setup", "view_d4", "official_14", "dump_tools"))
            dt_assigns = ("DUMP_MAX_BYTES", "GPU_BATCH", "VAL", "_infer_parser")
            dt_defs = ("load_net", "_dump_path", "_marker_path", "dump_is_done", "write_dump", "require_gpu_for",
                       "write_keys", "sgf_to_features", "parse_val", "dump_val_p03")
            env["RANK_QUESTIONS"] = qpath
        else:
            setup, view, off14, dt = (Cells.out(k) for k in ("setup", "view_d4", "model_tools", "dump_tools"))
            dt_assigns = ("DUMP_MAX_BYTES", "GPU_BATCH", "QSETS", "QUESTIONS_PATH", "_infer_parser")
            dt_defs = ("load_net", "_dump_path", "_marker_path", "dump_is_done", "write_dump", "require_gpu_for",
                       "write_keys", "sgf_to_features", "parse_questions", "parse_val", "dump_val_p03")
        exec_defs(setup, ("now_iso", "_real", "_is_under", "guard_write", "write_json", "read_json", "sha256_path",
                          "gpu_name_of"), env, assigns=("SMOKE_CFG",))
        exec(compile(view, "<view_d4>", "exec"), env)
        exec_defs(off14, ("state_dict_sha256", "_nchw_output", "channels_last"), env)
        exec_defs(dt, dt_defs, env, assigns=dt_assigns)
        os.makedirs(dump_dir, exist_ok=True)
        env["DUMP_DIR"] = dump_dir
        return env

    def small_net(self, seed):
        torch = self.torch
        torch.manual_seed(seed)
        net = self.GoRankResNet(in_channels=501, num_channels=8, num_blocks=1).eval()
        return net

    def test_dump_val_p03_through_the_adapter_equals_prereg03(self):
        torch = self.torch
        qpath = os.path.join(self.tmp, "questions.csv")
        write_questions(qpath, [("q_0001", [(small_sgf(i), "B") for i in range(5)]),
                                ("q_0002", [(small_sgf(5 + i), "W") for i in range(5)]),
                                ("q_0003", [(small_sgf(10), "B"), ("not an sgf", "W"), (small_sgf(11), "W"),
                                            (small_sgf(12), "B"), (small_sgf(13), "W")])])
        wsha = "f" * 64
        e4 = self.notebook_env("t04", qpath, os.path.join(self.tmp, "t04"))
        e3 = self.notebook_env("p03", qpath, os.path.join(self.tmp, "p03"))
        net = e4["channels_last"](self.small_net(3))
        e4["QUESTIONS_PATH"] = qpath
        V = e4["parse_questions"](qpath)
        self.assertIs(e4["parse_val"](), V)
        self.assertEqual(e4["write_keys"]("val_keys", V["keys"]), "written")
        self.assertEqual(e4["dump_val_p03"](net, "p03a", "prereg03/rank_a/ckpt/final.pth", wsha), "written")
        self.assertEqual(e3["write_keys"]("val_keys", e3["parse_val"]()["keys"]), "written")
        self.assertEqual(e3["dump_val_p03"](net, "p03a", "prereg03/rank_a/ckpt/final.pth", wsha), "written")
        for n in ("val_keys.npz", "val_p03a_d4.npz", "val_p03a_s0_swap.npz"):
            self.assertEqual(sha(os.path.join(self.tmp, "t04", n)), sha(os.path.join(self.tmp, "p03", n)), n)
        # the two views: s0 = net(dump_input(x, is_black, 0)), swap = net(dump_input(x, ~is_black, 0))
        with np.load(os.path.join(self.tmp, "t04", "val_p03a_d4.npz"), allow_pickle=False) as z:
            d4 = z["logits"]
        with np.load(os.path.join(self.tmp, "t04", "val_p03a_s0_swap.npz"), allow_pickle=False) as z:
            sw = z["logits"]
            self.assertEqual(str(z["view"]), "s0_colour_swapped")
        keys = V["keys"]
        self.assertEqual(keys["parsed"].tolist(), [True] * 11 + [False] + [True] * 3)
        self.assertTrue(np.isnan(d4[11]).all() and np.isnan(sw[11]).all())
        with torch.no_grad():
            for _q, start, k in V["spans"]:
                ok = [start + j for j in range(k) if V["feats"][start + j] is not None]
                x = torch.from_numpy(np.stack([V["feats"][g] for g in ok]))
                ib = torch.from_numpy(keys["color"][ok] == "B")
                s0 = net(e4["dump_input"](x, ib, 0)).float().numpy()
                s1 = net(e4["dump_input"](x, ~ib, 0)).float().numpy()
                self.assertEqual(s0.tobytes(), d4[ok, 0].tobytes())
                self.assertEqual(s1.tobytes(), sw[ok].tobytes())
                self.assertFalse(np.array_equal(s0, s1))

    def test_guard_write_blocks_the_four_readonly_roots(self):
        env = {"os": os}
        exec_defs(Cells.out("setup"), ("_real", "_is_under", "guard_write", "readonly_fingerprint"), env)
        results = os.path.join(self.tmp, "results")
        env["READONLY_ROOTS"] = {"prereg01": os.path.join(results, "prereg01"), "prereg02": os.path.join(results, "prereg02"),
                                 "prereg03": os.path.join(results, "prereg03"),
                                 "test02_rank": os.path.join(results, "test02", "rank")}
        for p in (os.path.join(results, "prereg03", "rank_a", "ckpt", "final.pth"),
                  os.path.join(results, "prereg01", "x"), os.path.join(results, "prereg02", "rank", "dumps", "y.npz"),
                  os.path.join(results, "test02", "rank", "config.json"),
                  os.path.join(results, "test04", "..", "prereg03", "z")):
            with self.assertRaises(PermissionError):
                env["guard_write"](p)
        for p in (os.path.join(results, "test02", "player", "x.json"), os.path.join(results, "test04", "sw", "x"),
                  os.path.join(results, "test04", "test04_rank_results.zip"), os.path.join(results, "prereg03_other")):
            self.assertEqual(env["guard_write"](p), p)

    # ---- the seam functions
    def seam_env(self, smoke=False):
        torch = self.torch
        env = {"np": np, "torch": torch, "math": math, "SMOKE": smoke, "LEVELS": list(LEVELS)}
        exec_defs(Cells.out("dump_tools"), ("_argmax_first", "rule_b"), env, assigns=("_E1",))
        exec_defs(Cells.out("seam_check"), ("_load_npz", "question_spans", "c0_of", "e_of", "max_diff", "seam_tol",
                                            "seam_arm", "dump_env", "weights_match"), env,
                  assigns=("SEAM_KEYS", "SEAM_ENV_KEYS"))
        return env

    @staticmethod
    def synthetic(n_q=30, seed=0):
        rng = np.random.default_rng(seed)
        qid, slot, color, parsed = [], [], [], []
        for q in range(n_q):
            k = 5 + q % 4
            for s in range(1, k + 1):
                qid.append("val_q_%04d" % (q + 1))
                slot.append(s)
                color.append("B" if q % 2 else "W")
                parsed.append(not (q == 2 or (q == 4 and s == 1)))
        keys = {"question_id": np.array(qid), "slot": np.array(slot, dtype=np.int32), "color": np.array(color),
                "parsed": np.array(parsed)}
        d4 = rng.normal(0, 2, size=(len(qid), 8, 10)).astype(np.float32)
        swap = rng.normal(0, 2, size=(len(qid), 10)).astype(np.float32)
        d4[~keys["parsed"]] = np.nan
        swap[~keys["parsed"]] = np.nan
        return keys, d4, swap

    def test_seam_tolerance_rule(self):
        env = self.seam_env()
        g4 = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        self.assertEqual(env["seam_tol"](g4, g4), (True, 1e-4))
        self.assertEqual(env["seam_tol"]("NVIDIA A100-SXM4-40GB", g4), (False, 1e-3))
        self.assertEqual(env["seam_tol"](None, None), (False, 1e-3))
        self.assertEqual(self.seam_env(smoke=True)["seam_tol"](None, None), (True, 1e-4))

    def test_seam_arm(self):
        env = self.seam_env()
        sa = env["seam_arm"]
        keys, d4, swap = self.synthetic()
        ref = {"keys": keys, "d4": d4, "swap": swap}
        r = sa({"keys": dict(keys), "d4": d4.copy(), "swap": swap.copy()}, ref, 1e-4)
        self.assertTrue(r["passed"] and r["bitwise_equal_d4"] and r["bitwise_equal_swap"])
        self.assertEqual((r["c0_s0_max_abs_diff"], r["c0_swap_max_abs_diff"], r["max_abs_logit_diff_swap"]), (0.0, 0.0, 0.0))
        self.assertEqual(r["max_abs_logit_diff_per_sym"], [0.0] * 8)
        self.assertEqual(r["rule_b_differences_s0_record_only"], 0)
        # one logit of one parsed game moved: small (5e-5) passes at 1e-4; larger (2e-4) fails at 1e-4, passes at 1e-3
        spans = env["question_spans"](keys["question_id"])
        for view in ("d4", "swap"):
            found = {}
            for delta in np.geomspace(1e-4, 0.2, 40):
                new = {"keys": keys, "d4": d4.copy(), "swap": swap.copy()}
                if view == "d4":
                    new["d4"][0, 0, 3] += np.float32(delta)
                    diff = env["max_diff"](env["c0_of"](new["d4"], spans, keys["parsed"]),
                                           env["c0_of"](d4, spans, keys["parsed"]))
                else:
                    new["swap"][0, 3] += np.float32(delta)
                    diff = env["max_diff"](env["c0_of"](new["swap"][:, None, :], spans, keys["parsed"]),
                                           env["c0_of"](swap[:, None, :], spans, keys["parsed"]))
                r4, r3 = sa(new, ref, 1e-4), sa(new, ref, 1e-3)
                self.assertEqual(r4["c0_%s_max_abs_diff" % ("s0" if view == "d4" else "swap")], diff)
                self.assertEqual(r4["passed"], diff <= 1e-4)
                self.assertEqual(r3["passed"], diff <= 1e-3)
                self.assertFalse(r4["bitwise_equal_%s" % view])
                if 3e-5 <= diff <= 8e-5:
                    found["5e-5"] = r4["passed"]
                if 1.5e-4 <= diff <= 3e-4:
                    found["2e-4"] = (r4["passed"], r3["passed"])
            self.assertEqual(found, {"5e-5": True, "2e-4": (False, True)}, view)
        # the other symmetries enter only the recorded logit difference
        d4b = d4.copy()
        d4b[:, 3, :] += 1.0
        r = sa({"keys": keys, "d4": d4b, "swap": swap}, ref, 1e-4)
        self.assertTrue(r["passed"])
        self.assertAlmostEqual(r["max_abs_logit_diff_per_sym"][3], 1.0, places=5)
        self.assertFalse(r["bitwise_equal_d4"])
        # different key arrays or shapes fail
        k2 = dict(keys, slot=keys["slot"].copy())
        k2["slot"][0] = 9
        self.assertFalse(sa({"keys": k2, "d4": d4, "swap": swap}, ref, 1e-3)["passed"])
        k3 = dict(keys, parsed=keys["parsed"].copy())
        k3["parsed"][0] = False
        self.assertFalse(sa({"keys": k3, "d4": d4, "swap": swap}, ref, 1e-3)["passed"])
        self.assertFalse(sa({"keys": keys, "d4": d4, "swap": swap[:, :9]}, ref, 1e-3)["passed"])

    def test_swap_c0_is_on_the_n_by_10_logits(self):
        env = self.seam_env()
        keys, d4, swap = self.synthetic(n_q=3)
        spans = env["question_spans"](keys["question_id"])
        c0 = env["c0_of"](swap[:, None, :], spans, keys["parsed"])
        self.assertIsNone(c0[2])
        _q, start, k = spans[0]
        z = swap[start:start + k].astype(np.float32)
        e = np.exp(z - z.max(1, keepdims=True))
        np.testing.assert_allclose(c0[0], (e / e.sum(1, keepdims=True)).mean(0), atol=1e-6)
        # E: float64 mean of log_softmax over the parsed games of a question
        E = env["e_of"](swap, spans, keys["parsed"])
        zz = swap[start:start + k].astype(np.float64)
        want = (zz - np.log(np.exp(zz).sum(1, keepdims=True))).mean(0)
        np.testing.assert_allclose(E[0], want, atol=1e-12)
        self.assertIsNone(E[2])

    def test_environment_and_weights(self):
        env = self.seam_env()
        m = {"gpu_name": "G", "torch": "2.11.0+cu128", "cudnn": 91900, "weights_sha256": "w"}
        markers = {"%s/%s" % (s, n): dict(m) for s in SETS for n in ("val_keys", "val_p03a_d4", "val_p03a_s0_swap",
                                                                     "val_p03b_d4", "val_p03b_s0_swap")}
        self.assertEqual(len(markers), 15)
        _e, same = env["dump_env"](markers)
        self.assertTrue(same)
        for key, change in (("gpu_name", "NVIDIA A100-SXM4-40GB"), ("torch", "2.12.0"), ("cudnn", 91000)):
            mixed = {k: dict(v) for k, v in markers.items()}
            mixed["public/val_p03b_d4"][key] = change
            self.assertFalse(env["dump_env"](mixed)[1], key)
        missing = dict(markers)
        missing["private/val_keys"] = None
        self.assertFalse(env["dump_env"](missing)[1])
        wm = {k: v for k, v in markers.items() if "p03a" in k}
        wn = {k: "w" for k in wm}
        self.assertEqual(env["weights_match"](wm, wn, "w"), (True, []))
        self.assertFalse(env["weights_match"](dict(wm, **{"public/val_p03a_d4": dict(m, weights_sha256="x")}), wn, "w")[0])
        self.assertFalse(env["weights_match"](wm, dict(wn, **{"public/val_p03a_d4": "x"}), "w")[0])


# ================================================================================================= fake Drive tree
BUNDLE_FILES_PIN = None


def npz_bytes(arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


class FakeDrive:
    """A fake Drive tree for the notebook's SMOKE path: prereg03_smoke (random weight files, tiny reference dumps),
    a smoke test bundle, optional test02_smoke records."""

    def __init__(self, root, n_q=3, per_q=5):
        self.root, self.drive, self.work = root, os.path.join(root, "drive"), os.path.join(root, "work")
        self.results = os.path.join(self.drive, "results")
        self.n_q, self.per_q = n_q, per_q
        os.makedirs(self.work)
        self.keys = self.make_keys("val_q_", n_q)
        self.rng = np.random.default_rng(7)
        self.wsha, self.ref = {}, {}
        for x in ARMS:
            d = os.path.join(self.results, "prereg03_smoke", "rank_" + x)
            os.makedirs(os.path.join(d, "ckpt"))
            os.makedirs(os.path.join(d, "dumps"))
            with open(os.path.join(d, "ckpt", "final.pth"), "wb") as f:
                f.write(("weights of arm %s" % x).encode() * 100)
            self.wsha[x] = sha(os.path.join(d, "ckpt", "final.pth"))
            self.ref[x] = self.make_logits(n_q * per_q)
            self.write_dump(os.path.join(d, "dumps"), "val_keys", self.keys, None, None, None)
            self.write_dump(os.path.join(d, "dumps"), "val_%s_d4" % DUMP_NAMES[x],
                            {"logits": self.ref[x]["d4"], "weights_sha256": np.array(self.wsha[x]), "syms": np.array(8)},
                            8, "prereg03/rank_%s/ckpt/final.pth" % x, self.wsha[x])
            self.write_dump(os.path.join(d, "dumps"), "val_%s_s0_swap" % DUMP_NAMES[x],
                            {"logits": self.ref[x]["swap"], "weights_sha256": np.array(self.wsha[x]),
                             "view": np.array("s0_colour_swapped")}, 1, "prereg03/rank_%s/ckpt/final.pth" % x,
                            self.wsha[x])
        self.qval = os.path.join(root, "bundle_local", "rank_val_questions.csv")
        os.makedirs(os.path.dirname(self.qval))
        write_questions(self.qval, [("val_q_%04d" % (q + 1), [("(;GM[1])", "B")] * per_q) for q in range(n_q)])
        self.write_test_zip()

    def make_keys(self, prefix, n_q):
        qid = [prefix + "%04d" % (q + 1) for q in range(n_q) for _s in range(self.per_q)]
        return {"question_id": np.array(qid), "slot": np.array([s + 1 for _q in range(n_q) for s in range(self.per_q)],
                                                                 dtype=np.int32),
                "color": np.array(["B"] * len(qid)), "parsed": np.array([True] * len(qid))}

    def make_logits(self, n):
        return {"d4": self.rng.normal(0, 2, size=(n, 8, 10)).astype(np.float32),
                "swap": self.rng.normal(0, 2, size=(n, 10)).astype(np.float32)}

    @staticmethod
    def write_dump(d, stem, arrays, syms, weights, wsha, run=1, env=None):
        data = npz_bytes(arrays)
        with open(os.path.join(d, stem + ".npz"), "wb") as f:
            f.write(data)
        m = {"file": "dumps/%s.npz" % stem, "rows": len(next(iter(arrays.values()))), "syms": syms, "bytes": len(data),
             "sha256": hashlib.sha256(data).hexdigest(), "weights": weights, "weights_sha256": wsha, "seconds": 0.0,
             "time": "2026-11-05T10:00:00+00:00", "run": run, "gpu_name": None, "torch": None, "cudnn": None,
             "device": "cpu"}
        m.update(env or {})
        with open(os.path.join(d, "_DONE_%s.json" % stem), "w", encoding="utf-8") as f:
            json.dump(m, f)

    def question_file(self, prefix, n=400):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)] + ["color_%d" % i for i in range(1, 21)])
        for i in range(n):
            k = 5 + 5 * (i % 4)
            w.writerow(["%sq_%04d" % (prefix, i + 1), k] + ["(;GM[1];B[pd])"] * k + [""] * (20 - k) + ["W"] * k
                       + [""] * (20 - k))
        return buf.getvalue().encode()

    def write_test_zip(self):
        files = {"rank_prediction_test_public.csv": self.question_file("pub_"),
                 "rank_prediction_test_private.csv": self.question_file("priv_"),
                 "player_identification_test_public.csv": b"a\n", "player_identification_test_private.csv": b"b\n",
                 "player_identification_candidates_public.csv": b"c\n",
                 "player_identification_candidates_private.csv": b"d\n"}
        man = {"format": "c10-test02 manifest v1", "kind": "smoke", "t1": {"rank": "PASS", "player": "PASS"},
               "files": {n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)} for n, b in files.items()}}
        self.manifest_bytes = json.dumps(man).encode()
        with zipfile.ZipFile(os.path.join(self.drive, "C10_test_v1.zip"), "w") as z:
            for n, b in files.items():
                z.writestr("tests/" + n, b)
            z.writestr("tests/MANIFEST.json", self.manifest_bytes)

    def write_test02(self, manifest_sha=None, status="completed"):
        d = os.path.join(self.results, "test02_smoke", "rank")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"test_bundle": {"manifest_sha256": manifest_sha or hashlib.sha256(self.manifest_bytes).hexdigest()}}, f)
        with open(os.path.join(d, "run_log.json"), "w", encoding="utf-8") as f:
            json.dump({"status": status, "runs": [{"run": 1, "status": status}]}, f)

    def write_sw_folder(self, name, seam_passed=None, statuses=()):
        d = os.path.join(self.results, "test04_smoke", name)
        os.makedirs(d, exist_ok=True)
        if seam_passed is not None:
            with open(os.path.join(d, "seam_check.json"), "w", encoding="utf-8") as f:
                json.dump({"passed": seam_passed}, f)
        if statuses:
            with open(os.path.join(d, "run_log.json"), "w", encoding="utf-8") as f:
                json.dump({"status": statuses[-1], "resume_count": 0,
                           "runs": [{"run": i + 1, "status": s} for i, s in enumerate(statuses)]}, f)
        return d


class NotebookRun:
    """Executes the notebook's cells (setup, stand-ins for prepare / environment / official 1, official 2,
    preflight, and optionally the later cells) in one namespace on a FakeDrive, like colab/run_local_smoke.py."""

    def __init__(self, fake):
        self.fake = fake
        self.ns = None

    def __enter__(self):
        self.cwd = os.getcwd()
        self.env = mock.patch.dict(os.environ, {"C10_SMOKE": "1", "C10_DRIVE_DIR": self.fake.drive,
                                                "C10_WORK_DIR": self.fake.work})
        self.env.start()
        return self

    def __exit__(self, *exc):
        self.env.stop()
        os.chdir(self.cwd)
        return False

    def run_cell(self, key):
        exec(compile(Cells.out(key), "<%s>" % key, "exec"), self.ns)

    def start(self):
        self.ns = {"__name__": "__main__", "print": lambda *a, **k: None}
        self.run_cell("setup")
        # the prepare cell's results (the bundle is not unpacked here): the bundle manifest, the validation question
        # file (a stand-in), the bundle zip path (environment cell: its size only), hashlib / subprocess imports
        bundle = os.path.join(self.fake.root, "bundle.zip")
        with open(bundle, "wb") as f:
            f.write(b"stand-in")
        self.ns.update({"MANIFEST": {"files": preflight_constants()["BUNDLE_FILES_PIN"]},
                        "RANK_QUESTIONS": self.fake.qval, "BUNDLE_ZIP_SRC": bundle, "hashlib": hashlib,
                        "subprocess": subprocess})
        if TUTORIAL not in sys.path:
            sys.path.insert(0, TUTORIAL)
        self.run_cell("official_1")
        self.run_cell("environment")
        self.ns["Pool"] = ThreadPool    # as colab/run_local_smoke.py does on Windows
        self.run_cell("official_2")
        return self.ns

    def run_log(self):
        return load(os.path.join(self.fake.results, "test04_smoke", "sw", "run_log.json"))


class Preflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t04_pf_")
        self.fake = FakeDrive(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_preflight(self, expect_stop=False):
        with NotebookRun(self.fake) as r:
            ns = r.start()
            if expect_stop:
                with self.assertRaises(SystemExit) as cm:
                    r.run_cell("preflight_t04")
                log = r.run_log()
                self.assertEqual(log["status"], "preflight_failed")
                self.assertEqual(log["runs"][-1]["status"], "preflight_failed")
                return str(cm.exception)
            r.run_cell("preflight_t04")
            return ns

    def test_passes_on_a_clean_tree(self):
        ns = self.run_preflight()
        self.assertEqual(ns["ATTEMPT"], 1)
        self.assertEqual(ns["TEST02_RECORD"], {"config_sha256": None, "manifest_sha256": None, "run_log_status": None})
        self.assertEqual(ns["TF32"]["used"], {"cudnn": True, "matmul": False})
        cfg = load(os.path.join(self.fake.results, "test04_smoke", "sw", "config.json"))
        self.assertEqual(cfg["format"], "c10-test04 rank notebook config v1")
        self.assertEqual(set(cfg["weights"]), {"a", "b"})
        self.assertEqual(cfg["weights"]["a"], {"label": "prereg03/rank_a/ckpt/final.pth", "sha256": self.fake.wsha["a"]})
        self.assertEqual(set(cfg["sets"]), set(SETS))
        self.assertEqual(cfg["sets"]["public"]["questions_count"], 400)
        self.assertEqual(cfg["sets"]["public"]["rows"], 12)
        self.assertEqual(cfg["test_bundle"]["manifest_sha256"], hashlib.sha256(self.fake.manifest_bytes).hexdigest())
        self.assertEqual(set(cfg["seam_reference"]), {"a", "b", "gpu", "env"})
        for k in ("tf32", "test02_record", "notebook_sha256", "base_notebooks", "kind", "smoke", "rehearsal"):
            self.assertIn(k, cfg)
        for s in SETS:
            self.assertTrue(os.path.exists(os.path.join(self.fake.results, "test04_smoke", "sw", s, "dumps", "_INPUT.json")))
        self.assertEqual(load(os.path.join(self.fake.results, "test04_smoke", "sw", "run_log.json"))["status"], "running")

    def test_weights_mismatch_stops(self):
        with open(os.path.join(self.fake.results, "prereg03_smoke", "rank_b", "ckpt", "final.pth"), "ab") as f:
            f.write(b"x")
        self.assertIn("SHA-256", self.run_preflight(expect_stop=True))

    def test_test02_record(self):
        self.fake.write_test02(manifest_sha="0" * 64)
        self.assertIn("test02", self.run_preflight(expect_stop=True))
        self.fake.write_test02(status="seam_failed")
        self.assertIn("completed", self.run_preflight(expect_stop=True))
        self.fake.write_test02()
        ns = self.run_preflight()
        self.assertEqual(ns["TEST02_RECORD"]["run_log_status"], "completed")

    def test_test02_required_outside_smoke(self):
        env = {"os": os, "hashlib": hashlib, "json": json}
        exec_defs(Cells.out("setup"), ("read_json", "sha256_path"), env)
        exec_defs(Cells.out("preflight_t04"), ("test02_state",), env)
        rank_dir = os.path.join(self.tmp, "t02rank")
        rec, problem = env["test02_state"](rank_dir, "m" * 64, True)
        self.assertIn("不在", problem)
        self.assertEqual(env["test02_state"](rank_dir, "m" * 64, False), (rec, None))
        os.makedirs(rank_dir)
        with open(os.path.join(rank_dir, "config.json"), "w") as f:
            json.dump({"test_bundle": {"manifest_sha256": "m" * 64}}, f)
        self.assertIsNotNone(env["test02_state"](rank_dir, "m" * 64, True)[1])     # run_log.json missing
        with open(os.path.join(rank_dir, "run_log.json"), "w") as f:
            json.dump({"status": "completed"}, f)
        self.assertIsNone(env["test02_state"](rank_dir, "m" * 64, True)[1])
        self.assertIn("MANIFEST", env["test02_state"](rank_dir, "n" * 64, True)[1])

    def test_previous_seam_failure_stops(self):
        sw = self.fake.write_sw_folder("sw", seam_passed=False)
        self.assertTrue(os.path.isdir(sw))
        self.assertIn("sw_failed_N", self.run_preflight(expect_stop=True))

    def test_earlier_failed_run_in_the_same_folder_stops(self):
        self.fake.write_sw_folder("sw", statuses=("seam_failed",))
        self.assertIn("接縫類失敗", self.run_preflight(expect_stop=True))

    def test_no_resume_after_a_done_marker(self):
        d = os.path.join(self.fake.results, "test04_smoke", "sw", "public", "dumps")
        os.makedirs(d)
        with open(os.path.join(d, "_DONE_val_keys.json"), "w") as f:
            json.dump({"rows": 1}, f)
        self.assertIn("sw_interrupted_N", self.run_preflight(expect_stop=True))

    def test_interrupted_folder_that_was_a_seam_failure_stops(self):
        self.fake.write_sw_folder("sw_interrupted_1", statuses=("running", "seam_failed"))
        self.assertIn("不是斷線", self.run_preflight(expect_stop=True))

    def test_two_failed_folders_stop(self):
        self.fake.write_sw_folder("sw_failed_1", seam_passed=False)
        self.fake.write_sw_folder("sw_failed_2", statuses=("final_check_failed",))
        self.assertIn("登記只准重跑一次", self.run_preflight(expect_stop=True))

    def test_one_failed_folder_is_the_registered_rerun(self):
        self.fake.write_sw_folder("sw_failed_1", seam_passed=False, statuses=("seam_failed",))
        self.fake.write_sw_folder("sw_interrupted_1", statuses=("running",))   # a plain interruption does not count
        ns = self.run_preflight()
        self.assertEqual(ns["ATTEMPT"], 2)
        self.assertEqual(sorted(ns["PRIOR_SW"]), ["sw_failed_1", "sw_interrupted_1"])

    def test_rerun_only_on_g4(self):
        """Review round 1 (rev 3 items 4, 9): with a folder renamed sw_failed_* the run is the registered rerun (or a
        rerun after a disconnect) and stops in preflight unless the GPU is G4; outside SMOKE only."""
        env = {}
        exec_defs(Cells.out("preflight_t04"), ("rerun_gpu_problem",), env)
        f = env["rerun_gpu_problem"]
        g4, a100 = "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA A100-SXM4-40GB"
        self.assertEqual(preflight_constants()["P03_REF_ENV_PIN"]["gpu_name"], g4)
        self.assertIsNone(f({}, a100, g4))                                        # first attempt: A100 allowed
        self.assertIsNone(f({"sw_interrupted_1": []}, a100, g4))                  # a disconnect of the first one
        self.assertIsNone(f({"sw_failed_1": ["seam_check.json passed = False"]}, g4, g4))
        # review round 2: the full-name rule of t04_rank.py unpack / official_log.py (a G4 name variant stops too)
        for gpu in (a100, None, "Tesla T4", g4 + " MIG 1g.12gb", "NVIDIA RTX PRO 6000", "RTX PRO 6000 Blackwell"):
            with self.subTest(gpu=gpu):
                msg = f({"sw_failed_1": [], "sw_interrupted_1": []}, gpu, g4)      # also an archive of apply exit 1
                self.assertIn("只能用 G4", msg)
                self.assertIn("sw_failed_1", msg)
                self.assertIn(g4, msg)
                self.assertIn("截圖給 Claude" if "RTX PRO 6000" in (gpu or "") else "不要改選 A100", msg)
                # review round 3: docs/COLAB_RUN.md tells the two messages apart by their first words
                self.assertTrue(msg.startswith("G4 的名稱和 03 記的不同" if "RTX PRO 6000" in (gpu or "")
                                               else "這是登記的重跑"), msg[:40])
        # review round 3: an uncounted archive (sw_exempt_*) alone is not a rerun: A100 allowed, as unpack
        self.assertIsNone(f({"sw_exempt_1": ["seam_check.json passed = False"]}, a100, g4))
        self.assertIn("只能用 G4", f({"sw_exempt_1": [], "sw_failed_2": []}, a100, g4))
        pf = Cells.out("preflight_t04")
        self.assertIn("RERUN_GPU_PROBLEM = None if SMOKE else rerun_gpu_problem(PRIOR_SW, GPU_MODEL_NOW, "
                      "P03_REF_ENV_PIN['gpu_name'])", pf)
        self.assertLess(pf.index("RERUN_GPU_PROBLEM ="), pf.index("SET_QUESTIONS ="))   # before any dump input
        self.fake.write_sw_folder("sw_failed_1", seam_passed=False, statuses=("seam_failed",))
        self.assertEqual(self.run_preflight()["RERUN_GPU_PROBLEM"], None)          # SMOKE: not checked
        self.assertIn("只能選 G4", Cells.out("official_2"))

    def test_rerun_gpu_stop_is_taken(self):
        """Review round 2: the preflight lines from RERUN_GPU_PROBLEM up to step 7 run outside SMOKE and really stop
        (preflight_stop) a rerun that is not on G4, before any dump input is written; a G4 rerun, a first attempt and
        SMOKE pass. (test_rerun_only_on_g4 calls the function alone; the Preflight runs are SMOKE.)"""
        pf = Cells.out("preflight_t04")
        seg = pf[pf.index("RERUN_GPU_PROBLEM = "):pf.index("# ⑦")]
        self.assertNotIn("write_json", seg)
        self.assertNotIn("_INPUT", seg)
        env = exec_defs(pf, ("rerun_gpu_problem",), {}, assigns=("P03_REF_ENV_PIN",))
        g4 = env["P03_REF_ENV_PIN"]["gpu_name"]
        stops = []

        def preflight_stop(message):
            stops.append(message)
            raise SystemExit(message)
        cases = [("A100 rerun", False, {"sw_failed_1": []}, "NVIDIA A100-SXM4-40GB", True),
                 ("no GPU name", False, {"sw_failed_1": []}, None, True),
                 ("G4 name variant", False, {"sw_failed_1": [], "sw_interrupted_1": []}, g4 + " MIG", True),
                 ("G4 rerun", False, {"sw_failed_1": []}, g4, False),
                 ("first attempt on A100", False, {"sw_interrupted_1": []}, "NVIDIA A100-SXM4-40GB", False),
                 ("SMOKE", True, {"sw_failed_1": []}, None, False)]
        for label, smoke, prior, gpu, stop in cases:
            with self.subTest(label):
                del stops[:]
                ns = dict(env, SMOKE=smoke, PRIOR_SW=prior, GPU_MODEL_NOW=gpu, preflight_stop=preflight_stop)
                if stop:
                    with self.assertRaises(SystemExit) as cm:
                        exec(compile(seg, "<preflight ⑥-5>", "exec"), ns)
                    self.assertIn("只能用 G4", str(cm.exception))
                    self.assertEqual(len(stops), 1)
                else:
                    exec(compile(seg, "<preflight ⑥-5>", "exec"), ns)
                    self.assertIsNone(ns["RERUN_GPU_PROBLEM"])
                    self.assertEqual(stops, [])

    # ---- review round 3
    def test_exempt_archives_do_not_count_and_failed_ones_always_do(self):
        """The notebook counts archives as tools/t04_rank.py does: sw_exempt_* (an uncounted archive: a run before R20's
        first upload, a bug revision) is not counted toward the one rerun, even with a seam failure inside; every
        sw_failed_* is (also an apply-exit-1 archive whose Drive folder passed its own seam)."""
        self.fake.write_sw_folder("sw_exempt_1", seam_passed=False, statuses=("seam_failed",))
        self.fake.write_sw_folder("sw_failed_2", seam_passed=False, statuses=("seam_failed",))
        ns = self.run_preflight()
        self.assertEqual((ns["ATTEMPT"], ns["SEAM_FAILED_FOLDERS"]), (2, ["sw_failed_2"]))
        self.assertEqual(sorted(ns["PRIOR_SW"]), ["sw_exempt_1", "sw_failed_2"])
        # the ⑥-5 lines outside SMOKE: sw_exempt_1 alone on A100 passes; with sw_failed_2 it stops
        pf = Cells.out("preflight_t04")
        seg = pf[pf.index("RERUN_GPU_PROBLEM = "):pf.index("# ⑦")]
        env = exec_defs(pf, ("rerun_gpu_problem",), {}, assigns=("P03_REF_ENV_PIN",))

        def preflight_stop(message):
            raise SystemExit(message)
        base = dict(env, SMOKE=False, GPU_MODEL_NOW="NVIDIA A100-SXM4-40GB", preflight_stop=preflight_stop)
        ns2 = dict(base, PRIOR_SW={"sw_exempt_1": ["seam_check.json passed = False"]})
        exec(compile(seg, "<preflight ⑥-5>", "exec"), ns2)
        self.assertIsNone(ns2["RERUN_GPU_PROBLEM"])
        with self.assertRaises(SystemExit):
            exec(compile(seg, "<preflight ⑥-5>", "exec"),
                 dict(base, PRIOR_SW={"sw_exempt_1": [], "sw_failed_2": ["x"]}))
        # an apply-exit-1 archive (its Drive seam passed) counts: with one more counted archive the notebook stops
        self.fake.write_sw_folder("sw_failed_3", seam_passed=True, statuses=("completed",))
        self.assertIn("登記只准重跑一次", self.run_preflight(expect_stop=True))

    def test_interrupted_folder_seam_failure_message(self):
        self.fake.write_sw_folder("sw_interrupted_1", seam_passed=False, statuses=("seam_failed",))
        msg = self.run_preflight(expect_stop=True)
        self.assertIn("不是斷線", msg)
        self.assertIn("也不要改名", msg)

    def test_stale_running_runs_become_interrupted(self):
        self.fake.write_sw_folder("sw", statuses=("running",))
        ns = self.run_preflight()
        log = load(os.path.join(self.fake.results, "test04_smoke", "sw", "run_log.json"))
        self.assertEqual([r["status"] for r in log["runs"]], ["interrupted", "running"])
        self.assertEqual(ns["RUN"]["run"], 2)


class ReadonlyRoots(unittest.TestCase):
    def test_roots_and_test02_player_writes(self):
        tmp = tempfile.mkdtemp(prefix="c10_t04_ro_")
        try:
            fake = FakeDrive(tmp)
            fake.write_test02()
            with NotebookRun(fake) as r:
                ns = r.start()
                r.run_cell("preflight_t04")
                roots = ns["READONLY_ROOTS"]
                res = fake.results
                self.assertEqual(roots, {"prereg01": os.path.join(res, "prereg01_smoke"),
                                         "prereg02": os.path.join(res, "prereg02_smoke"),
                                         "prereg03": os.path.join(res, "prereg03_smoke"),
                                         "test02_rank": os.path.join(res, "test02_smoke", "rank")})
                exec_defs(Cells.out("dump_tools"), ("record_readonly_after",), ns)
                # competition 1's test02_player writing into test02/player during the run: still unchanged
                player = os.path.join(res, "test02_smoke", "player")
                os.makedirs(player)
                with open(os.path.join(player, "run_log.json"), "w") as f:
                    f.write("{}")
                _after, changed = ns["record_readonly_after"]()
                self.assertEqual(changed, {})
                with open(os.path.join(res, "test02_smoke", "rank", "config.json"), "a") as f:
                    f.write(" ")
                _after, changed = ns["record_readonly_after"]()
                self.assertEqual(list(changed), ["test02_rank"])
            setup = Cells.out("setup")
            for text in ("'test04_smoke' if SMOKE else ('test04_rehearsal' if REHEARSAL else 'test04')",
                         "'test02_smoke' if SMOKE else ('test02_rehearsal' if REHEARSAL else 'test02'),",
                         "RESULT_DIR = os.path.join(RESULT_ROOT, 'sw')"):
                self.assertIn(text, setup)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SeamAndFinalCells(unittest.TestCase):
    """The seam and final cells on fake dumps written by the notebook's own write_keys / write_dump."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t04_sf_")
        self.fake = FakeDrive(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_until_dumps(self, r, mutate=None):
        ns = r.start()
        for key in ("preflight_t04", "view_d4", "model_tools", "dump_tools"):
            r.run_cell(key)
        sw = os.path.join(self.fake.results, "test04_smoke", "sw")
        for s in SETS:
            ns["DUMP_DIR"] = os.path.join(sw, s, "dumps")
            # SMOKE processes the first 12 questions of each test file (the seam_val stand-in file has 3)
            keys = self.fake.keys if s == "seam_val" else self.fake.make_keys({"public": "pub_q_", "private": "priv_q_"}[s], 12)
            ns["write_keys"]("val_keys", keys)
            for x in ARMS:
                lg = self.fake.ref[x] if s == "seam_val" else self.fake.make_logits(len(keys["question_id"]))
                if mutate and s == "seam_val":
                    lg = mutate(x, {k: v.copy() for k, v in lg.items()})
                n = DUMP_NAMES[x]
                ns["write_dump"]("val_%s_d4" % n, len(keys["question_id"]), 8, ns["W_LABEL"][x], ns["WSHA"][x], 0.0,
                                 {"logits": lg["d4"], "weights_sha256": np.array(ns["WSHA"][x]), "syms": np.array(8)})
                ns["write_dump"]("val_%s_s0_swap" % n, len(keys["question_id"]), 1, ns["W_LABEL"][x], ns["WSHA"][x],
                                 0.0, {"logits": lg["swap"], "weights_sha256": np.array(ns["WSHA"][x]),
                                       "view": np.array("s0_colour_swapped")})
        return ns, sw

    def test_pass_then_completed(self):
        self.fake.write_test02()
        with NotebookRun(self.fake) as r:
            ns, sw = self.run_until_dumps(r)
            # competition 1's test02_player writes into test02/player/ while this runs: not a read-only change
            player = os.path.join(self.fake.results, "test02_smoke", "player")
            os.makedirs(player)
            with open(os.path.join(player, "config.json"), "w") as f:
                f.write("{}")
            r.run_cell("seam_check")
            seam = load(os.path.join(sw, "seam_check.json"))
            self.assertTrue(seam["passed"] and seam["bitwise_equal"] and seam["env_same"])
            self.assertEqual(seam["tol"], 1e-4)
            for x in ARMS:
                self.assertTrue(seam["arms"][x]["file_sha_equal_d4"] and seam["arms"][x]["file_sha_equal_swap"])
            r.run_cell("final_check")
            self.assertEqual(r.run_log()["status"], "completed")
            with zipfile.ZipFile(os.path.join(self.fake.results, "test04_smoke", "test04_rank_results.zip")) as z:
                names = z.namelist()
            self.assertTrue(all(n.startswith("sw/") for n in names))
            self.assertIn("sw/private/dumps/_DONE_val_p03b_s0_swap.json", names)
            self.assertEqual(len(names), 7 + 3 * 11)

    def test_run_all_again_after_completion(self):
        """Review round 3: pressing 'run all' again on a completed folder (seam passed, zip packed) stops with a
        message of its own: download that zip, do not rename (it used to say 'disconnection: rename sw to
        sw_interrupted_N', throwing a good result away). Without the zip it is the disconnection message again."""
        self.fake.write_test02()
        with NotebookRun(self.fake) as r:
            self.run_until_dumps(r)
            r.run_cell("seam_check")
            r.run_cell("final_check")
            self.assertEqual(r.run_log()["status"], "completed")
        zp = os.path.join(self.fake.results, "test04_smoke", "test04_rank_results.zip")
        before = sha(zp)
        with NotebookRun(self.fake) as r:
            r.start()
            with self.assertRaises(SystemExit) as cm:
                r.run_cell("preflight_t04")
            msg = str(cm.exception)
            self.assertEqual([x["status"] for x in r.run_log()["runs"]], ["completed", "preflight_failed"])
        self.assertIn("已經跑完了", msg)
        self.assertIn("不要改名", msg)
        # the Drive path from the folder above C10 (here the fake Drive folder is called 'drive')
        self.assertIn("drive/results/test04_smoke/test04_rank_results.zip", msg.replace(os.sep, "/"))
        self.assertNotIn("sw_interrupted", msg)
        self.assertEqual(sha(zp), before)
        with NotebookRun(self.fake) as r:      # again: still the completed message (the stop is not a result run)
            r.start()
            with self.assertRaises(SystemExit) as cm:
                r.run_cell("preflight_t04")
            self.assertIn("已經跑完了", str(cm.exception))
        os.remove(zp)
        with NotebookRun(self.fake) as r:
            r.start()
            with self.assertRaises(SystemExit) as cm:
                r.run_cell("preflight_t04")
            self.assertIn("斷線後不接續", str(cm.exception))

    def test_seam_failure_packs_and_stops(self):
        def mutate(x, lg):
            if x == "b":
                lg["swap"][0, 2] += np.float32(0.5)
            return lg
        with NotebookRun(self.fake) as r:
            _ns, sw = self.run_until_dumps(r, mutate=mutate)
            with self.assertRaises(SystemExit) as cm:
                r.run_cell("seam_check")
            self.assertIn("接縫核對沒過", str(cm.exception))
            seam = load(os.path.join(sw, "seam_check.json"))
            self.assertFalse(seam["passed"])
            self.assertTrue(seam["arms"]["a"]["passed"])
            self.assertFalse(seam["arms"]["b"]["passed"])
            self.assertTrue(seam["arms"]["b"]["bitwise_equal_d4"])
            self.assertEqual(r.run_log()["status"], "seam_failed")
            self.assertTrue(os.path.exists(os.path.join(self.fake.results, "test04_smoke", "test04_rank_results.zip")))
            self.assertTrue(os.path.exists(os.path.join(sw, "readonly_after.json")))

    def test_mixed_environment_fails_the_seam(self):
        with NotebookRun(self.fake) as r:
            ns, sw = self.run_until_dumps(r)
            p = os.path.join(sw, "private", "dumps", "_DONE_val_p03a_d4.json")
            m = load(p)
            m["gpu_name"] = "NVIDIA A100-SXM4-40GB"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(m, f)
            with self.assertRaises(SystemExit):
                r.run_cell("seam_check")
            seam = load(os.path.join(sw, "seam_check.json"))
            self.assertFalse(seam["env_same"])
            self.assertTrue(seam["arms"]["a"]["passed"] and seam["arms"]["b"]["passed"])

    def test_final_check_failure(self):
        with NotebookRun(self.fake) as r:
            ns, sw = self.run_until_dumps(r)
            r.run_cell("seam_check")
            p = os.path.join(sw, "public", "dumps", "_DONE_val_p03b_d4.json")
            m = load(p)
            m["run"] = 0     # computed by an earlier run: not allowed (no resume)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(m, f)
            with self.assertRaises(SystemExit) as cm:
                r.run_cell("final_check")
            self.assertIn("結尾核對沒過", str(cm.exception))
            self.assertEqual(r.run_log()["status"], "final_check_failed")
            cfg = load(os.path.join(sw, "config.json"))
            self.assertTrue(any("val_p03b_d4" in p for p in cfg["runs"][-1]["outputs"]["final_problems"]))


class VersionPrompt(unittest.TestCase):
    """Review round 3 (user-colab major): in Colab (not SMOKE) the setup cell asks the user to confirm the 12-character
    notebook version BEFORE any record is written or anything computed; anything but y stops (not a run). The wrong
    notebook therefore never writes into sw and never computes."""

    def block(self):
        setup = Cells.out("setup")
        i = setup.index("NOTEBOOK_VERSION = NOTEBOOK_SHA256[:12]")
        j = setup.index("# 和官方 notebook 的全部差異")
        return setup, i, j, setup[i:j]

    def test_before_any_write(self):
        setup, i, j, blk = self.block()
        self.assertIn("input(", blk)
        self.assertLess(setup.index("NOTEBOOK_SHA256 = '"), i)
        # every call the cell makes before the prompt (top-level statements; function bodies only define): none writes
        calls = set()
        for stmt in ast.parse(setup[:i]).body:
            if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
                continue
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call):
                    f = n.func
                    calls.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
        writes = {"write_json", "save_run_log", "save_config", "save_config_summary", "mark_time", "savez", "replace",
                  "copyfile", "copy", "copytree", "rmtree", "move", "write", "writestr", "dump", "open", "remove"}
        self.assertEqual(calls & writes, set(), sorted(calls))
        self.assertIn("makedirs", calls)          # only the (empty) result folder is created before the prompt
        for later in ("def write_json(", "save_run_log()\n", "save_config()\n", "RUN_LOG_PATH = "):
            self.assertLess(setup.index("input("), setup.index(later), later)
        self.assertIn("if IN_COLAB and not SMOKE:", blk)

    def test_prompt_behaviour(self):
        _setup, _i, _j, blk = self.block()
        code = compile(blk, "<setup version prompt>", "exec")
        sha_ = "0e7178be2955" + "a" * 52
        asked = []

        def answer(text):
            def f(prompt):
                asked.append(prompt)
                if isinstance(text, BaseException):
                    raise text
                return text
            return f
        for label, in_colab, smoke, reply, stops in (("y", True, False, "y", False), ("Y with spaces", True, False, " Y ", False),
                                                     ("n", True, False, "n", True), ("empty", True, False, "", True),
                                                     ("EOF", True, False, EOFError(), True),
                                                     ("yes", True, False, "yes", True),
                                                     ("local", False, False, "n", False), ("SMOKE", True, True, "n", False)):
            with self.subTest(label):
                del asked[:]
                ns = {"IN_COLAB": in_colab, "SMOKE": smoke, "NOTEBOOK_SHA256": sha_, "input": answer(reply)}
                if stops:
                    with self.assertRaises(SystemExit) as cm:
                        exec(code, ns)
                    self.assertIn("沒有確認", str(cm.exception))
                    self.assertIn("不算一次執行", str(cm.exception))
                    self.assertIn("中斷連線並刪除執行階段", str(cm.exception))
                else:
                    exec(code, ns)
                self.assertEqual(ns["NOTEBOOK_VERSION"], "0e7178be2955")
                self.assertEqual(len(asked), 1 if in_colab and not smoke else 0)
                if asked:
                    self.assertIn("0e7178be2955", asked[0])


# ================================================================================================= Adapter
class Adapter(unittest.TestCase):
    """The set cells: QUESTIONS_PATH is the set's question file, both arms' markers have the set's rows."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t04_ad_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def env(self, rows_shift=0, status="written"):
        import torch
        base = tempfile.mkdtemp(dir=self.tmp)
        paths = {s: os.path.join(base, "%s.csv" % s) for s in SETS}
        env = {"time": time, "os": os, "np": np, "json": json, "torch": torch, "DEVICE": torch.device("cpu"), "SET_QUESTIONS": paths,
               "RESULT_DIR": os.path.join(base, "sw"), "guard_write": lambda p: p, "ARMS": ARMS,
               "DUMP_NAMES": DUMP_NAMES, "NETS": {}, "NET_STATE_SHA256": {}, "W_PATH": {x: "w_" + x for x in ARMS},
               "WSHA": {x: x * 64 for x in ARMS}, "W_LABEL": {x: "prereg03/rank_%s/ckpt/final.pth" % x for x in ARMS},
               "QSETS": {}, "SET_LOG": {}, "mark_time": lambda *a: None, "print": lambda *a, **k: None,
               "save_config": lambda **kw: None, "save_config_summary": lambda **kw: None,
               "state_dict_sha256": lambda net: "s" * 64, "load_net": lambda path, expect_sha: (object(), expect_sha),
               "write_keys": lambda stem, arrays: "written"}
        exec_defs(Cells.out("setup"), ("read_json", "write_json"), env)
        exec_defs(Cells.out("dump_tools"), ("parse_val",), env, assigns=("QUESTIONS_PATH",))
        env["SET_INPUT"] = {s: {"set": s, "question_file": "x", "sha256": "0"} for s in SETS}
        for s in SETS:
            d = os.path.join(env["RESULT_DIR"], s, "dumps")
            os.makedirs(d)
            env["write_json"](os.path.join(d, "_INPUT.json"), env["SET_INPUT"][s])

        def parse_questions(path):
            if path not in env["QSETS"]:
                env["QSETS"][path] = {"spans": [("q", 0, 5)], "feats": [object()] * 5,
                                      "keys": {"parsed": np.array([True] * 5)}, "path": path}
            return env["QSETS"][path]

        def dump_val_p03(net, name, weights, weights_sha):
            V = env["parse_val"]()
            for v in ("d4", "s0_swap"):
                env["write_json"](os.path.join(env["DUMP_DIR"], "_DONE_val_%s_%s.json" % (name, v)),
                                  {"rows": len(V["feats"]) + rows_shift})
            return status

        env["parse_questions"], env["dump_val_p03"] = parse_questions, dump_val_p03
        return env

    def test_set_cells_record_their_question_file(self):
        env = self.env()
        for s in SETS:
            exec(compile(Cells.out("dump_set_" + s), s, "exec"), env)
            self.assertEqual(env["SET_LOG"][s]["questions_path"], env["SET_QUESTIONS"][s])
            self.assertEqual(env["SET_LOG"][s]["done_rows"], {x: {"d4": 5, "s0_swap": 5} for x in ARMS})
            self.assertEqual(env["SET_LOG"][s]["status"], {"keys": "written", "a": "written", "b": "written"})
        self.assertEqual(set(env["NETS"]), set(ARMS))
        self.assertEqual(env["QSETS"], {})

    def test_wrong_questions_path_stops(self):
        env = self.env()
        cell = Cells.out("dump_set_public")
        self.assertEqual(cell.count("QUESTIONS_PATH = _path"), 1)
        with self.assertRaises(AssertionError):
            exec(compile(cell.replace("QUESTIONS_PATH = _path", "QUESTIONS_PATH = SET_QUESTIONS['private']"),
                         "public", "exec"), env)

    def test_marker_rows_and_skipped_dumps_stop(self):
        with self.assertRaises(SystemExit):
            exec(compile(Cells.out("dump_set_private"), "private", "exec"), self.env(rows_shift=-1))
        with self.assertRaises(SystemExit):
            exec(compile(Cells.out("dump_set_seam_val"), "seam_val", "exec"), self.env(status="skipped"))


if __name__ == "__main__":
    unittest.main()
