"""Tests for colab/build_prereg03_rank_nb.py and colab/prereg03_rank_{a,b}.ipynb (prereg03, competition 2).

* byte-identical rebuild of the two committed notebooks and the two diff files;
* the A and B notebooks differ only in the ARM line; every code cell compiles; the validation label token
  appears only in the inherited prepare-cell refusal; unchanged base cells are verbatim;
* registered values in the notebook agree with docs/PREREG_03_RECIPE_V2.md / results/prereg03/design/plan.md
  (T, warmup, lr, weight decay, the E_L table, the 200-game cap, the time limits, the worker counts);
* pins agree with the committed prereg02 files (key-file digests, bundle manifest), recomputed here
  independently of the builder;
* functions defined in the new cells behave as specified, executed from the cell sources on CPU torch:
  target-perspective view, D4 gather == d4, lr schedule, shard order / worker seeds, arm B acceptance,
  scan_keys' per-shard order and the shard-layout digest (resume check), the Drive-space pause;
* review fixes (build_review.md section 3): pass_N.pth off Drive, the pause before a pass, the lr check as a registered
  stop, the shard layout in the recipe, arm B never restoring arm A's backup, preflight stops when space is unknown;
* second review fixes (build_review.md section 4): ckpt/history.csv checked against the in-memory history (and rewritten)
  before packing, one 3 GB Drive threshold for preflight and the pass pause (5 GB when arm A will write the feature
  backup) with the exact number in each message, the GPU gate text when CUDA exists but the model cannot be read;
* E_L recomputed from the real fit rows equals the registered table, and pass-1 acceptance matches an
  independent implementation (skipped when data/training is absent; fit players only);
* optional: with C10_P03_SMOKE_RESULTS=<.../prereg03_smoke> the SMOKE dumps load with p02_rank_fuse.load_dump.
Run: .venv/Scripts/python.exe -m pytest tests/test_p03_nb.py
"""

import ast
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))

NB = {"A": os.path.join(REPO, "colab", "prereg03_rank_a.ipynb"), "B": os.path.join(REPO, "colab", "prereg03_rank_b.ipynb")}
DIFF = {"A": os.path.join(REPO, "results", "prereg03", "design", "nb_diff_a.txt"),
        "B": os.path.join(REPO, "results", "prereg03", "design", "nb_diff_b.txt")}
BASE = os.path.join(REPO, "colab", "prereg02_rank.ipynb")
BUILDER = os.path.join(REPO, "colab", "build_prereg03_rank_nb.py")
PLAN = os.path.join(REPO, "results", "prereg03", "design", "plan.md")
P02 = os.path.join(REPO, "results", "prereg02", "rank")
TOKEN = "ans" + "wers"
BASE_SHA256 = "9fb1f9d06ba5ee5fbebfb6dfe022246888c93e7762e9a89e1adc99c9f1c12b0c"
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


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
    m = cell["metadata"]["c10"]
    return base_key(m["base"]) if "base_cell" in m else m["new"]


def cell(nb, key):
    found = [c for c in nb["cells"] if cell_key(c) == key]
    assert len(found) == 1, key
    return src(found[0])


def defs(source, names, extra_assigns=()):
    """Source text of the named top-level functions / classes and assignments of a cell (in cell order)."""
    tree = ast.parse(source)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            out.append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in extra_assigns for t in node.targets):
            out.append(ast.get_source_segment(source, node))
    return "\n\n".join(out)


def keys_digest(arrays):
    # independent re-implementation (same definition as the notebook's key-file cell)
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(("%s|%s|%s|" % (k, a.dtype.str, a.shape)).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


class Built(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = {k: load(v) for k, v in NB.items()}
        cls.base = load(BASE)

    def test_base_sha(self):
        self.assertEqual(sha(BASE), BASE_SHA256)

    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, BUILDER, REPO, "--out-dir", d], check=True, capture_output=True)
            for arm in ("A", "B"):
                self.assertEqual(sha(os.path.join(d, os.path.basename(NB[arm]))), sha(NB[arm]), arm)
                self.assertEqual(sha(os.path.join(d, os.path.basename(DIFF[arm]))), sha(DIFF[arm]), arm)

    def test_arms_differ_in_the_arm_line_only(self):
        la = open(NB["A"], encoding="utf-8").read().split("\n")
        lb = open(NB["B"], encoding="utf-8").read().split("\n")
        self.assertEqual(len(la), len(lb))
        diff = [(x, y) for x, y in zip(la, lb) if x != y]
        self.assertEqual(len(diff), 1)
        self.assertIn("ARM = 'A'", diff[0][0])
        self.assertIn("ARM = 'B'", diff[0][1])
        for arm in ("A", "B"):
            setup = cell(self.nb[arm], "setup")
            self.assertEqual(len(re.findall(r"^ARM = '[AB]'$", setup, flags=re.M)), 1)

    def test_cells_compile_ids_and_token(self):
        for arm in ("A", "B"):
            nb = self.nb[arm]
            ids = [c["id"] for c in nb["cells"]]
            self.assertEqual(ids, ["c10-rank03-%02d" % i for i in range(len(ids))])
            for i, c in enumerate(nb["cells"]):
                if c["cell_type"] == "code":
                    compile(src(c), "<cell %d>" % i, "exec")
                    self.assertEqual(c["outputs"], [])
            lines = [ln for c in nb["cells"] for ln in src(c).splitlines() if TOKEN in ln]
            self.assertEqual(len(lines), 1)
            self.assertIn("in p for p in _all_files", lines[0])
            self.assertEqual(cell(nb, "prepare"), src(self.base["cells"][3]))

    def test_unchanged_base_cells_are_verbatim(self):
        base_by_idx = self.base["cells"]
        for c in self.nb["A"]["cells"]:
            m = c["metadata"]["c10"]
            if "base_cell" in m:
                same = src(c) == src(base_by_idx[m["base_cell"]])
                self.assertEqual(not same, m["changed"], cell_key(c))
        for key in ("prepare", "official_1", "environment", "official_4", "official_6", "official_8"):
            self.assertFalse([c for c in self.nb["A"]["cells"] if cell_key(c) == key][0]["metadata"]["c10"]["changed"], key)

    def test_cell_order(self):
        keys = [cell_key(c) for c in self.nb["A"]["cells"]]
        self.assertEqual(keys, ["header", "setup", "prepare", "official_1", "environment", "official_2", "preflight_p03",
                                "official_4", "official_6", "official_8", "feature_backup_tools", "official_9", "view_d4",
                                "colour_table", "arm_b_sampling", "official_11", "official_12", "official_14",
                                "official_15", "dump_tools", "key_files", "training", "final_dumps", "final_check"])


class Registered(unittest.TestCase):
    """Values written into the notebook == the registration / plan (read from the documents where possible)."""

    @classmethod
    def setUpClass(cls):
        cls.nb = load(NB["A"])
        cls.plan = open(PLAN, encoding="utf-8").read()

    def test_recipe_constants(self):
        off2 = cell(self.nb, "official_2")
        for line in ("IN_CHANNELS = 501", "WEIGHT_DECAY = 0.01", "WARMUP_STEPS = 1000", "T_TOTAL = 33768",
                     "TIME_CHECK_STEP = 300", "TIME_LIMIT_MINUTES = {'G4': 150, 'A100': 180}", "SHUFFLE_BUFFER = 1000",
                     "PEAK_LR = LEARNING_RATE", "LEARNING_RATE = 1e-3", "BATCH_SIZE = 256", "SEED = 42"):
            self.assertIn(line, off2)
        self.assertIn("LOADER_WORKERS = {'G4': {'A': 16, 'B': 32}, 'A100': {'A': 10, 'B': 10}, 'cpu-smoke': {'A': 0, 'B': 0}}"
                      "[GPU_CLASS][ARM]", off2)
        self.assertIn("GPU_MODELS_OK = ('RTX PRO 6000', 'A100')", off2)
        self.assertIn("T＝33,768", self.plan)
        self.assertIn("150 分鐘（G4）或 > 180 分鐘（A100）", self.plan)
        self.assertIn("G4：A 16 個、B 32 個。A100：兩臂各 10 個", self.plan)

    def test_e_l_table_matches_plan(self):
        m = re.search(r"E_L（該級加上 200 盤上限後的總盤數）：(.+)", self.plan)
        table = {k: int(v.replace(",", "")) for k, v in re.findall(r"(\w+) ([\d,]+)", m.group(1))}
        ns = {}
        exec(defs(cell(self.nb, "arm_b_sampling"), (), ("E_L_REGISTERED", "CAP_REGISTERED", "TARGET_PER_LEVEL_REGISTERED")), ns)
        self.assertEqual(ns["E_L_REGISTERED"], table)
        self.assertEqual(list(ns["E_L_REGISTERED"]), LEVELS)
        self.assertEqual(ns["CAP_REGISTERED"], 200)
        self.assertEqual(ns["TARGET_PER_LEVEL_REGISTERED"], 35845)
        self.assertIn("q_g ＝ min(1, 200/n_p) × 35,845 / E_L", self.plan)
        self.assertIn("c10-prereg03|acc|{pass}|{rank}|{game_id}", self.plan)

    def test_pins_match_prereg02(self):
        kc = cell(self.nb, "key_files")
        ns = {}
        exec(defs(kc, (), ("KEY_DIGEST_PIN",)), ns)
        for stem in ("inner_keys", "val_keys"):
            with np.load(os.path.join(P02, "dumps", stem + ".npz"), allow_pickle=False) as z:
                self.assertEqual(ns["KEY_DIGEST_PIN"][stem], keys_digest({k: z[k] for k in z.files}), stem)
        pf = cell(self.nb, "preflight_p03")
        ns = {}
        exec(defs(pf, (), ("BUNDLE_FILES_PIN",)), ns)
        env = load(os.path.join(P02, "env.json"))
        self.assertEqual(ns["BUNDLE_FILES_PIN"], env["runs"][-1]["bundle_manifest"]["files"])
        # feature fingerprint pin == the dict the notebook builds on a real run
        fp = {"bundle_files": ns["BUNDLE_FILES_PIN"], "smoke": False, "smoke_cfg": None, "seed": 42, "per_shard": 1000,
              "fit_games": 720307, "inner_games": 73556}
        want = hashlib.sha256(json.dumps(fp, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        self.assertIn("FEAT_FINGERPRINT_SHA256_PIN = '%s'" % want, cell(self.nb, "official_9"))

    def test_no_forbidden_paths_written(self):
        setup = cell(self.nb, "setup")
        self.assertIn("for _n in ('prereg01', 'prereg02', 'test02')}", setup)
        self.assertIn("RESULT_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg03_smoke' if SMOKE else 'prereg03')", setup)


class CellFunctions(unittest.TestCase):
    """The new cells' functions, executed from the notebook source on CPU torch."""

    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.nb = load(NB["B"])
        base_dt = src(load(BASE)["cells"][16])
        cls.view_ns = {"torch": torch, "DEVICE": torch.device("cpu"), "IN_CHANNELS": 501, "save_config": lambda **kw: None,
                       "print": lambda *a, **k: None}
        exec(cell(cls.nb, "view_d4"), cls.view_ns)   # runs the cell's own unit checks too
        cls.base_d4 = {}
        exec(defs(base_dt, ("d4",)), {"torch": torch}, cls.base_d4)

    def test_view(self):
        torch = self.torch
        ns = self.view_ns
        g = torch.Generator().manual_seed(1)
        x = torch.randint(0, 2, (5, 500, 19, 19), generator=g).to(torch.uint8)
        isb = torch.tensor([True, False, True, False, False])
        v = ns["to_target_view"](x, isb)
        for b in range(5):
            for t in (0, 1, 125, 249):
                me, op = (x[b, 2 * t], x[b, 2 * t + 1]) if isb[b] else (x[b, 2 * t + 1], x[b, 2 * t])
                self.assertTrue(torch.equal(v[b, 2 * t], me))
                self.assertTrue(torch.equal(v[b, 2 * t + 1], op))
            self.assertTrue(bool((v[b, 500] == int(isb[b])).all()))
        # the view is an involution on the stone planes (applying it twice with the same colour gives x back)
        vv = ns["to_target_view"](v[:, :500].contiguous(), isb)
        self.assertTrue(torch.equal(vv[:, :500], x))
        # the colour-swapped view of a black-target game equals the correct view of the same game as a white target
        sw = ns["to_target_view"](x, ~isb)
        self.assertTrue(torch.equal(sw[:, :500], ns["to_target_view"](x, torch.zeros(5, dtype=torch.bool))[:, :500] * isb[:, None, None, None]
                                    + ns["to_target_view"](x, torch.ones(5, dtype=torch.bool))[:, :500] * (~isb)[:, None, None, None]))
        self.assertTrue(bool((sw[:, 500] == (~isb).to(torch.uint8)[:, None, None]).all()))

    def test_d4_gather_equals_d4(self):
        torch = self.torch
        ns = self.view_ns
        self.assertEqual(ns["D4_PERM"].shape, (8, 361))
        g = torch.Generator().manual_seed(2)
        x = torch.rand(16, 3, 19, 19, generator=g)
        s = torch.randint(0, 8, (16,), generator=g)
        got = ns["d4_gather"](x, s)
        for b in range(16):
            self.assertTrue(torch.equal(got[b:b + 1], self.base_d4["d4"](x[b:b + 1], int(s[b]))))
        # the moved d4 is the base d4
        for k in range(8):
            self.assertTrue(torch.equal(ns["d4"](x[:2], k), self.base_d4["d4"](x[:2], k)))
        # training input == dump input for every symmetry
        xu = torch.randint(0, 2, (3, 500, 19, 19), generator=g).to(torch.uint8)
        ib = torch.tensor([True, False, True])
        for k in range(8):
            a = ns["model_input"](xu, ib, torch.full((3,), k, dtype=torch.long))
            b = ns["dump_input"](xu, ib, k)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(a.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(a.dtype, torch.float32)

    def test_channels_last_hook_keeps_the_official_function(self):
        torch = self.torch
        sys.path.insert(0, os.path.join(REPO, "third_party", "aicup2026_tutorial"))
        from network import GoRankResNet
        ns = {"torch": torch}
        exec(defs(cell(self.nb, "official_14"), ("_nchw_output", "channels_last")), ns)
        torch.manual_seed(0)
        a = GoRankResNet(in_channels=501).eval()
        b = GoRankResNet(in_channels=501).eval()
        b.load_state_dict(a.state_dict())
        ns["channels_last"](b)
        self.assertTrue(b.conv_in.weight.is_contiguous(memory_format=torch.channels_last))
        x = torch.rand(3, 501, 19, 19, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            ya, yb = a(x), b(x.contiguous(memory_format=torch.channels_last))
        self.assertLess(float((ya - yb).abs().max()), 1e-5)
        GoRankResNet(in_channels=501).load_state_dict(b.state_dict())   # final.pth loads into the official class

    def test_guard_write_blocks_the_three_readonly_roots(self):
        ns = {"os": os}
        exec(defs(cell(self.nb, "setup"), ("_real", "_is_under", "guard_write", "readonly_fingerprint")), ns)
        with tempfile.TemporaryDirectory() as d:
            results = os.path.join(d, "results")
            ns["READONLY_ROOTS"] = {n: os.path.join(results, n) for n in ("prereg01", "prereg02", "test02")}
            for n in ("prereg01", "prereg02", "test02"):
                for p in (os.path.join(results, n), os.path.join(results, n, "rank", "x.npz"),
                          os.path.join(results, "prereg03", "..", n, "y")):
                    with self.assertRaises(PermissionError):
                        ns["guard_write"](p)
            for p in (os.path.join(results, "prereg03", "rank_a", "x"), os.path.join(results, "prereg02_other", "x"),
                      os.path.join(results, "test02_rehearsal", "x")):
                self.assertEqual(ns["guard_write"](p), p)
            self.assertEqual(ns["readonly_fingerprint"](os.path.join(results, "test02")), {})
            os.makedirs(os.path.join(results, "prereg01", "rank"))
            with open(os.path.join(results, "prereg01", "rank", "a.txt"), "w") as f:
                f.write("x")
            fp = ns["readonly_fingerprint"](os.path.join(results, "prereg01"))
            self.assertEqual(list(fp), ["rank/a.txt"])
            self.assertEqual(fp["rank/a.txt"][0], 1)

    def test_lr_schedule(self):
        ns = {"math": math, "WARMUP_STEPS": 1000, "T_TOTAL": 33768}
        exec(defs(cell(self.nb, "training"), ("lr_lambda",)), ns)
        f = ns["lr_lambda"]
        self.assertAlmostEqual(f(0), 1 / 1000)
        self.assertAlmostEqual(f(999), 1.0)
        self.assertAlmostEqual(f(1000), 1.0)
        self.assertAlmostEqual(f(1000 + (33768 - 1000) // 2), 0.5, places=4)
        self.assertEqual(f(33768), 0.0)
        vals = [f(t) for t in range(1000, 33769, 97)]
        self.assertTrue(all(a >= b for a, b in zip(vals, vals[1:])))
        self.assertTrue(all(f(t) < f(t + 1) for t in range(0, 999)))

    def test_shard_order_and_worker_seed(self):
        ns = {"random": random, "SEED": 42}
        exec(defs(cell(self.nb, "official_11"), ("shard_order", "worker_seed")), ns)
        shards = ["./rank-features/train/train-w%d-%06d.tar.gz" % (w, i) for w in range(48) for i in range(16)]
        o1 = ns["shard_order"](list(reversed(shards)), 1)
        self.assertEqual(sorted(o1), sorted(shards))
        ref = sorted(shards)
        random.Random(42 * 1000 + 1).shuffle(ref)
        self.assertEqual(o1, ref)
        self.assertEqual(o1, ns["shard_order"](shards, 1))
        self.assertNotEqual(o1, ns["shard_order"](shards, 2))
        self.assertEqual(ns["worker_seed"](3, 7), (42 * 1000 + 3) * 1000 + 7)

    def test_acceptance_functions(self):
        ns = {"hashlib": hashlib, "LEVELS": LEVELS}
        exec(defs(cell(self.nb, "arm_b_sampling"), ("level_totals", "acceptance_q", "acceptance_u", "accepted_keys")), ns)
        u = ns["acceptance_u"](7, "1D", "g_0012345")
        want = int(hashlib.sha256(b"c10-prereg03|acc|7|1D|g_0012345").hexdigest()[:16], 16) / 2 ** 64
        self.assertEqual(u, want)
        self.assertEqual(ns["acceptance_q"](400, 49949, 35845, 200), 0.5 * 35845 / 49949)
        self.assertEqual(ns["acceptance_q"](10, 35845, 35845, 200), 1.0)
        tot, n = ns["level_totals"](["a", "a", "a", "b", "c"], ["D", "D", "D", "D", "C"], 2)
        self.assertEqual(tot["D"], 3)
        self.assertEqual(tot["C"], 1)
        self.assertEqual(n, {"a": 3, "b": 1, "c": 1})
        with self.assertRaises(ValueError):
            ns["level_totals"](["a", "a"], ["D", "C"], 2)
        rows = [("D_g%d" % i, "D", "g%d" % i, 0.5, i % 2 == 0) for i in range(4000)]
        keep, by_level, black = ns["accepted_keys"](1, rows)
        self.assertTrue(1800 < len(keep) < 2200)
        self.assertEqual(by_level["D"], len(keep))
        self.assertEqual(black, sum(1 for k in keep if int(k[3:]) % 2 == 0))
        self.assertNotEqual(keep, ns["accepted_keys"](2, rows)[0])


class LoaderWorkers(unittest.TestCase):
    """The loader cell's pipelines with several DataLoader workers, simulated in-process (Windows cannot fork):
    each simulated worker w of W iterates the same DataPipeline with torch's worker info patched to (w, W).
    Checks: shard split and per-worker order depend only on (42, pass, worker) and are repeatable; every sample
    is read exactly once per pass; arm B's select keeps exactly the accepted keys; partial last batches."""

    W = 3
    N_SHARDS = 7
    PER = 23

    @classmethod
    def setUpClass(cls):
        import types
        import torch
        import webdataset as wds
        from torch.utils.data import DataLoader
        cls.torch, cls.wds = torch, wds
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cwd = os.getcwd()
        os.chdir(cls.tmp.name)   # webdataset treats 'C:' as a URL scheme: shards are given as relative paths
        os.makedirs(os.path.join("rank-features", "train"))
        cls.keys, colour = [], {}
        g = 0
        for w in range(cls.N_SHARDS):
            with wds.TarWriter("rank-features/train/train-w%d-000000.tar.gz" % w) as sink:
                for _ in range(cls.PER):
                    lv = LEVELS[g % 10]
                    key = "%s_g_%07d" % (lv, g)
                    feat = np.zeros((500, 19, 19), dtype=np.uint8)
                    feat[0, 0, 0] = g % 251
                    sink.write({"__key__": key, "features.npy": feat, "label.txt": lv, "rank_idx.txt": str(g % 10),
                                "game_name.txt": "g_%07d" % g})
                    cls.keys.append(key)
                    colour[key] = g % 3 == 0
                    g += 1
        nb = load(NB["A"])
        cls.ns = {"torch": torch, "wds": wds, "np": np, "hashlib": hashlib, "DataLoader": DataLoader, "COLOR_IS_BLACK": colour,
                  "LEVELS": LEVELS, "SEED": 42, "SHUFFLE_BUFFER": 1000, "BATCH_SIZE": 8, "LOADER_WORKERS": 0,
                  "train_shards": sorted("rank-features/train/train-w%d-000000.tar.gz" % w for w in range(cls.N_SHARDS))}
        exec(cell(nb, "official_11"), cls.ns)
        cls.types = types

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        cls.tmp.cleanup()

    def run_worker(self, pass_no, w, accepted=None):
        """Batches that simulated worker w of W yields in pass pass_no."""
        from unittest import mock
        info = self.types.SimpleNamespace(id=w, num_workers=self.W)
        ds = self.ns["make_train_loader"](pass_no, accepted).dataset
        with mock.patch("torch.utils.data.get_worker_info", return_value=info):
            return [(tuple(b[3]), b[0].dtype, tuple(b[0].shape), b[2].tolist()) for b in ds]

    def test_worker_streams(self):
        for pass_no in (1, 2):
            streams = [self.run_worker(pass_no, w) for w in range(self.W)]
            self.assertEqual(streams, [self.run_worker(pass_no, w) for w in range(self.W)])   # repeatable
            order = self.ns["shard_order"](self.ns["train_shards"], pass_no)
            for w, st in enumerate(streams):
                keys_w = [k for b in st for k in b[0]]
                want = {k for s in order[w::self.W] for k in self.keys[int(s.split("-w")[1][0]) * self.PER:
                                                                      (int(s.split("-w")[1][0]) + 1) * self.PER]}
                self.assertEqual(set(keys_w), want)
                n = len(keys_w)
                self.assertEqual(len(st), -(-n // 8))              # partial last batch is a step
                self.assertTrue(all(b[1] == self.torch.uint8 and b[2][1:] == (500, 19, 19) for b in st))
                for b in st:
                    self.assertEqual(b[3], [self.ns["COLOR_IS_BLACK"][k] for k in b[0]])
            all_keys = [k for st in streams for b in st for k in b[0]]
            self.assertEqual(sorted(all_keys), sorted(self.keys))   # every sample exactly once
        self.assertNotEqual([self.run_worker(1, w) for w in range(self.W)], [self.run_worker(2, w) for w in range(self.W)])

    def test_arm_b_select(self):
        accepted = frozenset(self.keys[::3])
        got = [k for w in range(self.W) for b in self.run_worker(5, w, accepted) for k in b[0]]
        self.assertEqual(sorted(got), sorted(accepted))

    def test_missing_colour_stops(self):
        del_key = self.keys[0]
        saved = self.ns["COLOR_IS_BLACK"].pop(del_key)
        try:
            with self.assertRaises(SystemExit):
                list(self.ns["iterate"](self.ns["make_train_loader"](1, None), "test"))
        finally:
            self.ns["COLOR_IS_BLACK"][del_key] = saved

    def test_scan_keys(self):
        keys, by_shard = self.ns["scan_keys"](self.ns["train_shards"])
        self.assertEqual(sorted(keys), sorted(self.keys))
        self.assertEqual(sorted(by_shard), sorted(os.path.basename(s) for s in self.ns["train_shards"]))
        for w in range(self.N_SHARDS):   # each shard's keys in the order they were written
            self.assertEqual(by_shard["train-w%d-000000.tar.gz" % w], self.keys[w * self.PER:(w + 1) * self.PER])

    def write_shards(self, d, parts, mtime):
        """parts: {shard basename: [key, ...]}; writes them under d (relative path) with a fixed tar mtime."""
        wds = self.wds
        os.makedirs(d, exist_ok=True)
        out = []
        for name, keys in parts.items():
            path = "%s/%s" % (d, name)
            with wds.TarWriter(path, mtime=mtime) as sink:
                for key in keys:
                    g = int(key.split("_")[-1])
                    sink.write({"__key__": key, "features.npy": np.zeros((500, 19, 19), dtype=np.uint8),
                                "label.txt": LEVELS[g % 10], "rank_idx.txt": str(g % 10), "game_name.txt": "g_%07d" % g})
            out.append(path)
        return sorted(out)

    def layout(self, shards):
        return self.ns["shard_layout_sha256"](shards, self.ns["scan_keys"](shards)[1])

    def test_shard_layout_digest(self):
        # PREREG_03 plan 3.6: the resume must see the same shard layout. The digest depends on the shard names and the key
        # order inside each shard only: not on the compressed bytes (tar mtimes), the path prefix or the worker count
        by48 = {"train-w%d-000000.tar.gz" % w: self.keys[w::self.N_SHARDS] for w in range(self.N_SHARDS)}
        s1 = self.write_shards("lay1/train", by48, mtime=1.0)
        s2 = self.write_shards("lay2/train", by48, mtime=2.0)
        self.assertNotEqual(sha(s1[0]), sha(s2[0]))                 # other bytes
        self.assertEqual(self.layout(s1), self.layout(s2))           # same layout
        # the same games cut another way (another NPROC at extraction): another layout
        by3 = {"train-w%d-000000.tar.gz" % w: self.keys[w::3] for w in range(3)}
        self.assertNotEqual(self.layout(self.write_shards("lay3/train", by3, mtime=1.0)), self.layout(s1))
        # same shard names, another order inside one shard: another layout
        swapped = dict(by48)
        swapped["train-w0-000000.tar.gz"] = list(reversed(by48["train-w0-000000.tar.gz"]))
        self.assertNotEqual(self.layout(self.write_shards("lay4/train", swapped, mtime=1.0)), self.layout(s1))
        # a shard without keys still counts (it takes a place in shard_order)
        with_empty = dict(by48, **{"train-w99-000000.tar.gz": []})
        self.assertNotEqual(self.layout(self.write_shards("lay5/train", with_empty, mtime=1.0)), self.layout(s1))
        # the digest does not depend on the DataLoader worker split
        from unittest import mock
        by_w = {}
        for w in range(self.W):
            info = self.types.SimpleNamespace(id=w, num_workers=self.W)
            ds = self.ns["wds"].DataPipeline(self.ns["wds"].SimpleShardList(sorted(s1)), self.ns["wds"].split_by_worker,
                                              self.ns["wds"].tarfile_to_samples(), self.ns["wds"].map(self.ns["_sample_key"]),
                                              self.ns["wds"].batched(5, partial=True))
            with mock.patch("torch.utils.data.get_worker_info", return_value=info):
                for b in ds:
                    for k, u in zip(b[0], b[1]):
                        by_w.setdefault(self.ns["_shard_name"](u), []).append(k)
        self.assertEqual(self.ns["shard_layout_sha256"](s1, by_w), self.layout(s1))


class ReviewFixes(unittest.TestCase):
    """Fixes after the first review (build_review.md section 3): Drive space, shard layout on resume, stop classes."""

    @classmethod
    def setUpClass(cls):
        cls.nb = load(NB["A"])
        cls.train = cell(cls.nb, "training")

    def test_pass_checkpoints_stay_off_drive(self):
        t = self.train
        self.assertIn("PASS_CKPT_DIR = guard_write(os.path.join(WORK, 'prereg03_pass_ckpt_' + ARM_TAG", t)
        self.assertNotIn("os.path.join(CKPT_DIR, 'pass_", t)
        self.assertEqual(t.count("os.path.join(PASS_CKPT_DIR, 'pass_"), 3)

    def test_drive_space_pause(self):
        ns = {"shutil": __import__("shutil"), "now_iso": lambda: "t",
              "RUN": {"run": 3, "status": None}, "run_log": {"runs": [], "status": None}}
        calls = []
        ns["save_run_log"] = lambda: calls.append("save_run_log")
        ns["flush_drive"] = lambda: calls.append("flush_drive")
        exec(defs(cell(self.nb, "preflight_p03"), ("gb_text",), ("MIN_PASS_FREE_BYTES",)), ns)   # the threshold lives there
        self.assertEqual(ns["MIN_PASS_FREE_BYTES"], 3 * 10 ** 9)
        exec(defs(self.train, ("drive_free_bytes", "drive_space_pause")), ns)
        with tempfile.TemporaryDirectory() as d:
            ns["DRIVE_DIR"] = d
            self.assertGreater(ns["drive_free_bytes"](), 0)
            ns["DRIVE_DIR"] = os.path.join(d, "missing")
            self.assertIsNone(ns["drive_free_bytes"]())
        with self.assertRaises(SystemExit) as cm:
            ns["drive_space_pause"](5, 1.5e9)
        self.assertEqual(ns["RUN"]["status"], "drive_space_pause")
        self.assertEqual(ns["RUN"]["paused_before_pass"], 5)
        self.assertEqual(ns["run_log"]["status"], "drive_space_pause")
        self.assertEqual(calls, ["save_run_log", "flush_drive"])
        msg = str(cm.exception)
        self.assertIn("第 5 輪的開頭接續", msg)
        # the exact number to clear to, the same number preflight asks for on the re-run
        self.assertIn("Drive 只剩 1.50 GB，第 5 輪還沒開始就先停下（每一輪開始前要剩 3 GB 以上）", msg)
        self.assertIn("等 Drive 剩 3 GB 以上", msg)
        self.assertIn("垃圾桶清空了還不到 3 GB：C10 裡的東西都不要刪", msg)
        self.assertIn("這時沒有 zip，不用刪執行階段", msg)
        self.assertIsNone(re.search(r"(?<![\d.])2 GB", msg))   # the old 2 GB threshold is gone ("0.32 GB" is the state.pt size)
        body = defs(self.train, ("drive_space_pause",))
        self.assertNotIn("STOPPED_PATH", body)   # not a registered stop: no STOPPED.json, no failed zip
        self.assertNotIn("stop_arm", body)
        self.assertNotIn("pack_results", body)
        # checked at the top of every pass, before the pass seeds or trains
        loop = self.train[self.train.index("while not TRAIN_DONE:"):]
        self.assertLess(loop.index("drive_space_pause(_p, _free)"), loop.index("random.seed(SEED * 1000 + _p)"))

    def test_lr_check_is_a_registered_stop(self):
        t = self.train
        i = t.index("if abs(optimizer.param_groups[0]['lr'] - expected_lr(global_step)) > 1e-12:")
        j = t.index("stop_arm('pass_check_failed', '第 %d 輪的核對不符")
        self.assertLess(i, j)
        self.assertNotIn("SystemExit", t[i:j])

    def test_local_tool_knows_every_registered_stop(self):
        # every status the notebook's stop_arm() can write is refused by tools/p03_rank_fuse.py (PREREG_03 1.3)
        import p03_rank_fuse
        code = "\n".join(src(c) for c in self.nb["cells"] if c["cell_type"] == "code")
        statuses = set(re.findall(r"stop_arm\('([a-z_]+)'", code))
        self.assertEqual(statuses, set(p03_rank_fuse.STOP_STATUSES))
        self.assertNotIn("drive_space_pause", p03_rank_fuse.STOP_STATUSES)

    def test_recipe_holds_the_shard_layout(self):
        self.assertIn("'train_shard_layout_sha256': SHARD_CHECK['train']['layout_sha256'],", self.train)
        self.assertIn("'layout_sha256': shard_layout_sha256(_shards, _by_shard)", cell(self.nb, "official_12"))
        self.assertIn("RECIPE_GPU_KEYS = {'GPU_CLASS', 'WORKERS', 'train_shard_layout_sha256'}", self.train)

    def test_arm_b_never_restores_arm_a_backup(self):
        off9 = cell(self.nb, "official_9")
        self.assertIn("for _src in ((FEAT_BACKUP, P02_FEAT, P01_FEAT) if ARM == 'A' else (P02_FEAT, P01_FEAT)):", off9)
        self.assertIn("    if ARM == 'A':   # 本版：只有 A 臂寫備份", off9)

    def test_preflight_stops_when_drive_space_unknown(self):
        pf = cell(self.nb, "preflight_p03")
        block = pf[pf.index("# ③ Drive 剩餘空間"):pf.index("# ④")]
        self.assertIn("except OSError as e:\n    raise SystemExit(", block)
        self.assertIn("if DRIVE_FREE_BYTES < MIN_DRIVE_FREE_BYTES:", block)
        self.assertIn("MIN_PASS_FREE_BYTES = 3 * 10 ** 9", pf)
        self.assertIn("MIN_DRIVE_FREE_BYTES = MIN_PASS_FREE_BYTES + (FEATURE_BACKUP_ROOM_BYTES if WILL_WRITE_FEATURE_BACKUP else 0)",
                      pf)
        # one threshold: the training cell uses the preflight constant and does not define its own
        self.assertNotIn("MIN_PASS_FREE_BYTES =", self.train)
        self.assertIn("if _free is not None and _free < MIN_PASS_FREE_BYTES:", self.train)


def _stmt(source, startswith):
    """Source of the unique top-level statement of a cell whose text starts with `startswith`."""
    tree = ast.parse(source)
    segs = [ast.get_source_segment(source, n) for n in tree.body]
    found = [s for s in segs if s.startswith(startswith)]
    assert len(found) == 1, (startswith, len(found))
    return found[0]


class SecondReviewFixes(unittest.TestCase):
    """Fixes after the second review (build_review.md section 4): history.csv on a resume after training is done,
    one Drive-space threshold (3 GB; 5 GB when arm A will write the feature backup) with the exact number in every
    message, and the GPU gate when CUDA exists but the model cannot be read."""

    @classmethod
    def setUpClass(cls):
        cls.nb = {k: load(v) for k, v in NB.items()}
        cls.pf = cell(cls.nb["A"], "preflight_p03")
        cls.train = cell(cls.nb["A"], "training")
        cls.final = cell(cls.nb["A"], "final_check")
        cls.setup_defs = defs(cell(cls.nb["A"], "setup"), ("read_json", "gpu_name_of"))

    # ---- history.csv
    def history_ns(self, d, rows):
        import pandas as pd
        ns = {"os": os, "pd": pd, "CKPT_DIR": d, "guard_write": lambda p: p, "T_TOTAL": 70, "history": rows}
        exec(defs(self.train, ("save_history", "history_file_problems"), ("HISTORY_PATH", "HISTORY_COLS_CHECKED")), ns)
        return ns

    @staticmethod
    def rows():
        # the shape of the notebook's history rows (arm A: accepted None; the pass stopped at T has samples_expected None)
        base = {"pass_complete": True, "samples_expected": 2000, "train_loss": 2.3025850929940455, "train_acc": 9.8,
                "inner_loss": 2.302, "inner_acc": 8.5, "lr_start": 1e-4, "lr_end": 0.0, "seconds_train": 12.5,
                "seconds_monitor": 1.0, "ms_per_step": 198.4, "run": 1, "accepted": None, "accepted_by_level": None}
        r1 = dict(base, **{"pass": 1, "steps": 63, "samples": 2000, "black": 960, "global_step": 63,
                           "order_sha256": "a" * 64, "d4_sha256": "b" * 64})
        r2 = dict(base, **{"pass": 2, "pass_complete": False, "samples_expected": None, "steps": 7, "samples": 224,
                           "black": 110, "global_step": 70, "run": 2, "order_sha256": "0123456789" * 6 + "1e10",
                           "d4_sha256": "d" * 64})
        return [{k: r[k] for k in ("pass", "pass_complete", "steps", "samples", "samples_expected", "black", "global_step",
                                   "train_loss", "train_acc", "inner_loss", "inner_acc", "lr_start", "lr_end",
                                   "seconds_train", "seconds_monitor", "ms_per_step", "run", "accepted",
                                   "accepted_by_level", "order_sha256", "d4_sha256")} for r in (r1, r2)]

    def test_history_file_check(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self.rows()
            ns = self.history_ns(d, rows)
            self.assertEqual(ns["history_file_problems"](), ["ckpt/history.csv 不見了"])
            ns["save_history"]()
            self.assertEqual(ns["history_file_problems"](), [])          # a digit-and-e sha still compares as text
            # the stale file of a resume after the last pass: state.pt says done, history.csv stopped a pass earlier
            ns["history"] = rows[:1]
            ns["save_history"]()
            ns["history"] = rows
            probs = ns["history_file_problems"]()
            self.assertEqual(len(probs), 1)
            self.assertIn("是 1 列", probs[0])
            ns["save_history"]()
            self.assertEqual(ns["history_file_problems"](), [])
            # same shape, another value
            path = os.path.join(d, "history.csv")
            txt = open(path, encoding="utf-8").read()
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt.replace("d" * 64, "e" * 64))
            self.assertEqual(ns["history_file_problems"](), ["ckpt/history.csv 的 d4_sha256 欄和記憶體裡的 history 不同"])
            # in memory and on Drive agree but neither reaches T
            ns["T_TOTAL"] = 71
            ns["save_history"]()
            self.assertEqual(ns["history_file_problems"](), ["ckpt/history.csv 沒有記到第 71 步"])

    def test_final_rewrites_then_checks_history_before_packing(self):
        f = self.final
        i_check = f.index("_hist_before = history_file_problems()")
        i_write = f.index("    save_history()", i_check)
        i_again = f.index("_hist_after = history_file_problems()")
        i_add = f.index("_final_problems.extend(_hist_after)")
        i_stop = f.index("stop_arm('final_check_failed', '收尾核對不符")
        i_pack = f.index("RESULTS_ZIP, _skipped_big = pack_results(RESULTS_ZIP_NAME)")
        self.assertTrue(i_check < i_write < i_again < i_add < i_stop < i_pack)
        self.assertIn("'history_file': HISTORY_FILE_CHECK", f)
        self.assertIn("HISTORY_PATH = os.path.join(CKPT_DIR, 'history.csv')", self.train)
        self.assertEqual(self.train.count("to_csv("), 1)   # save_history() is the only writer

    # ---- Drive-space thresholds
    def preflight_ns(self, arm, d, markers=()):
        ns = {"os": os, "json": json, "ARM": arm, "SMOKE": False, "SMOKE_CFG": {"nproc": 2}, "SEED": 42, "PER_SHARD": 1000,
              "MANIFEST": {"files": {"a": "1"}}, "FEAT_BACKUP": os.path.join(d, "prereg03", "features_rank"),
              "P02_FEAT": os.path.join(d, "prereg02", "features_rank"), "P01_FEAT": os.path.join(d, "prereg01", "features_rank")}
        exec(self.setup_defs, ns)
        good = {"bundle_files": {"a": "1"}, "smoke": False, "smoke_cfg": None, "seed": 42, "per_shard": 1000,
                "fit_games": 720307, "inner_games": 73556}
        for key, content in markers:
            os.makedirs(ns[key], exist_ok=True)
            with open(os.path.join(ns[key], "_COMPLETE.json"), "w", encoding="utf-8") as f:
                f.write(content if isinstance(content, str) else json.dumps({"fingerprint": dict(good, **content)}))
        exec(defs(self.pf, ("gb_text", "feature_backup_marker_ok"),
                  ("MIN_PASS_FREE_BYTES", "FEATURE_BACKUP_ROOM_BYTES", "WILL_WRITE_FEATURE_BACKUP", "MIN_DRIVE_FREE_BYTES")), ns)
        return ns

    def preflight_message(self, ns, free):
        ns = dict(ns, DRIVE_FREE_BYTES=free)
        with self.assertRaises(SystemExit) as cm:
            exec(_stmt(self.pf, "if DRIVE_FREE_BYTES < MIN_DRIVE_FREE_BYTES:"), ns)
        return str(cm.exception)

    def test_preflight_threshold(self):
        cases = [("A", (), True), ("B", (), False),                                   # Pro+ first run: A re-extracts
                 ("A", (("P01_FEAT", {}),), False),                                    # AI Pro: prereg01's backup
                 ("A", (("FEAT_BACKUP", {}),), False),                                 # Pro+ resume: A's own backup
                 ("A", (("P02_FEAT", {"per_shard": 500}),), True),                     # another setting: re-extract
                 ("A", (("P01_FEAT", "{not json"),), True),                            # unreadable marker
                 ("A", (("P01_FEAT", "[1, 2]"),), True)]                               # not a dict
        for arm, markers, will in cases:
            with tempfile.TemporaryDirectory() as d:
                ns = self.preflight_ns(arm, d, markers)
                self.assertEqual(ns["WILL_WRITE_FEATURE_BACKUP"], will, (arm, markers))
                self.assertEqual(ns["MIN_DRIVE_FREE_BYTES"], 5 * 10 ** 9 if will else 3 * 10 ** 9, (arm, markers))
                self.assertEqual(ns["MIN_PASS_FREE_BYTES"], 3 * 10 ** 9)
                need = "5 GB" if will else "3 GB"
                msg = self.preflight_message(ns, 2.9e9)
                self.assertTrue(msg.startswith("Drive 只剩 2.90 GB，這份 notebook 開始前要剩 %s 以上" % need), msg[:80])
                self.assertIn("等 Drive 剩 %s 以上" % need, msg)
                self.assertIn("垃圾桶清空了還不到 %s：C10 裡的東西都不要刪" % need, msg)
                self.assertIn("清空垃圾桶", msg)
                if will:
                    self.assertIn("比平常的 3 GB 多留 2 GB", msg)
                    self.preflight_message(ns, 4.99e9)
                ns2 = dict(ns, DRIVE_FREE_BYTES=(5e9 if will else 3e9))
                exec(_stmt(self.pf, "if DRIVE_FREE_BYTES < MIN_DRIVE_FREE_BYTES:"), ns2)   # exactly the threshold passes

    # ---- GPU model that cannot be read
    def gate(self, device_type, smi):
        import types
        off2 = cell(self.nb["A"], "official_2")
        ns = {"SMOKE": False, "DEVICE": types.SimpleNamespace(type=device_type), "ENV_INFO": {"gpu": {"nvidia_smi": smi}}}
        exec(self.setup_defs, ns)
        for start in ("GPU_MODELS_OK = ", "GPU_MODEL_NOW = ", "GPU_MODEL_SHOWN = ",
                      "if not SMOKE and DEVICE.type == 'cuda' and not any("):
            exec(_stmt(off2, start), ns)
        return ns

    def test_gpu_model_unreadable(self):
        with self.assertRaises(SystemExit) as cm:
            self.gate("cuda", None)
        msg = str(cm.exception)
        self.assertTrue(msg.startswith("有 GPU，但查不到 GPU 型號，無法確認是 G4（RTX PRO 6000）或 A100，停在這裡（還沒開始算）"), msg)
        self.assertNotIn("None", msg)
        with self.assertRaises(SystemExit) as cm:
            self.gate("cuda", ["NVIDIA L4, 23034 MiB, 580.82.07"])
        self.assertTrue(str(cm.exception).startswith("這次拿到的 GPU 是 NVIDIA L4，不是 G4（RTX PRO 6000）或 A100，停在這裡"))
        ok = self.gate("cuda", ["NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07"])
        self.assertEqual(ok["GPU_MODEL_SHOWN"], "NVIDIA A100-SXM4-40GB")
        self.assertIsNone(self.gate("cpu", None)["GPU_MODEL_SHOWN"])
        # every user-facing GPU line prints the shown label, never the raw None
        self.assertIn("print('device', DEVICE, '| GPU', GPU_MODEL_SHOWN,", cell(self.nb["A"], "official_2"))
        self.assertIn("% (ARM, GPU_MODEL_SHOWN, GPU_CLASS, LOADER_WORKERS,", self.pf)


@unittest.skipUnless(os.path.exists(os.path.join(REPO, "data", "training", "train_D.csv")), "data/training is absent")
class RealFit(unittest.TestCase):
    """E_L and pass-1 acceptance on the real fit rows (fit players only; no validation player is read)."""

    @classmethod
    def setUpClass(cls):
        from c10data import load_rows
        rows = load_rows(os.path.join(REPO, "data", "training"), os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                         part="fit")
        cls.fit = [(r["player_id"], r["rank"], r["game_id"], r["color"]) for r in rows]
        del rows
        cls.nb = load(NB["B"])
        cls.ns = {"hashlib": hashlib, "LEVELS": LEVELS}
        exec(defs(cell(cls.nb, "arm_b_sampling"), ("level_totals", "acceptance_q", "acceptance_u", "accepted_keys"),
                  ("E_L_REGISTERED", "CAP_REGISTERED", "TARGET_PER_LEVEL_REGISTERED")), cls.ns)

    def test_e_l_equals_registered(self):
        ns = self.ns
        e_l, n_p = ns["level_totals"]([p for p, _r, _g, _c in self.fit], [r for _p, r, _g, _c in self.fit], 200)
        self.assertEqual(e_l, ns["E_L_REGISTERED"])
        self.assertEqual(min(e_l.values()), 35845)
        self.assertEqual(len(self.fit), 720307)
        self.assertEqual(len(n_p), 5767)

    def test_pass1_acceptance_matches_independent_code(self):
        ns = self.ns
        e_l = ns["E_L_REGISTERED"]
        n_p = {}
        for p, _r, _g, _c in self.fit:
            n_p[p] = n_p.get(p, 0) + 1
        rows = [("%s_%s" % (r, g), r, g, ns["acceptance_q"](n_p[p], e_l[r], 35845, 200), c == "B") for p, r, g, c in self.fit]
        keep, by_level, black = ns["accepted_keys"](1, rows)
        # independent: integer comparison u * 2^64 < q * 2^64 is avoided; recompute with float division as registered
        mine = set()
        for p, r, g, c in self.fit:
            q = min(1.0, 200 / n_p[p]) * 35845 / e_l[r]
            if int(hashlib.sha256(("c10-prereg03|acc|1|%s|%s" % (r, g)).encode()).hexdigest()[:16], 16) / 2.0 ** 64 < q:
                mine.add("%s_%s" % (r, g))
        self.assertEqual(set(keep), mine)
        for lv in LEVELS:
            self.assertLess(abs(by_level[lv] - 35845), 700, (lv, by_level[lv]))   # about 3.5 binomial sd
        self.assertLess(abs(len(keep) - 358450), 2000)
        self.assertTrue(0.45 < black / len(keep) < 0.55)
        print("\npass 1 accepted:", len(keep), by_level, "black", black)


@unittest.skipUnless(os.environ.get("C10_P03_SMOKE_RESULTS"), "set C10_P03_SMOKE_RESULTS=<scratch>/drive/results/prereg03_smoke")
class SmokeDumps(unittest.TestCase):
    """The notebook's SMOKE outputs load with the prereg02 local tool's readers (the dump contract for p03_rank_fuse)."""

    def test_load(self):
        import p02_rank_fuse as f
        root = os.environ["C10_P03_SMOKE_RESULTS"]
        for tag in ("a", "b"):
            d = os.path.join(root, "rank_" + tag, "dumps")
            if not os.path.isdir(d):
                continue
            vk = f.load_val_keys(d)
            with np.load(os.path.join(d, "inner_keys.npz"), allow_pickle=False) as z:
                n_in = len(z["game_id"])
            dump = f.load_dump(os.path.join(d, "inner_p03%s_d4.npz" % tag), n_in, 8)
            self.assertEqual(dump["logits"].shape, (n_in, 8, 10))
            val = f.load_dump(os.path.join(d, "val_p03%s_d4.npz" % tag), vk["n"], 8)
            f.check_val_logits(val, vk["parsed"])
            for stem, n in (("inner_p03%s_s0_swap" % tag, n_in), ("val_p03%s_s0_swap" % tag, vk["n"])):
                with np.load(os.path.join(d, stem + ".npz"), allow_pickle=False) as z:
                    self.assertEqual(z["logits"].shape, (n, 10))
                    self.assertEqual(z["logits"].dtype, np.float32)
                    self.assertEqual(str(z["weights_sha256"]), dump["weights_sha256"])
                    self.assertEqual(str(z["view"]), "s0_colour_swapped")
            sw = np.load(os.path.join(d, "val_p03%s_s0_swap.npz" % tag))["logits"]
            self.assertTrue(np.array_equal(np.isnan(sw).all(1), ~vk["parsed"]))


if __name__ == "__main__":
    unittest.main()
