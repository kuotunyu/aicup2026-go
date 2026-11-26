"""Tests for tools/p03_rank_fuse.py (PREREG_03 competition 2; plan.md 4.7).
Run: .venv/Scripts/python.exe -m unittest tests.test_p03_rank_fuse   (or python -m pytest tests/test_p03_rank_fuse.py)

Everything runs on synthetic data in a temporary directory outside the repository: the synthetic prereg02
world of tests/test_p02_rank.py (players, dumps, pseudo-questions, opening scores, the p02 chain with R20)
plus synthetic prereg03 arm results in the notebook's interface (plan.md 3.7). No real data, no label file,
no GPU. The real-data P00 = R20 reproduction is 'selfcheck' itself (run on the committed prereg02 files).
"""

import builtins
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import decide  # noqa: E402
import p02_rank_fuse as rf  # noqa: E402
import p02_rank_pseudo as pseudo  # noqa: E402
import p03_rank_fuse as p03  # noqa: E402
from test_p02_rank import World, save_npz  # noqa: E402


def repo_prereg03_status():
    """`git status` of results/prereg03 in the real repository ('' = nothing changed or added).
    Since the real prereg03 results were unpacked and committed (e55b019), rank_a/ exists, so the refusal tests assert
    "the refused command left results/prereg03 untouched" instead of "rank_a/ does not exist"."""
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "results/prereg03"],
                       cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"git status failed: {r.stderr.strip()}"

LEVELS = p03.LEVELS
T_SMOKE = 40


def quiet(fn, *a, **kw):
    """Run fn with stdout silenced (the tools print progress)."""
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ================================================================ synthetic prereg03 world
class P03World:
    """A p02 World with the R20 chain run (frozen R20, its validation probabilities) and two synthetic arms."""

    def __init__(self, root, seed=0, arms=("a", "b"), strength=None, modes=None, write_arms=True):
        self.root = root
        self.w = World(os.path.join(root, "w"), seed=seed)
        w = self.w
        w.write_pseudo()
        w.write_opening()
        old = rf.COMBOS
        rf.COMBOS = [c for c in old if c["id"] in ("R00", "R20")]
        try:
            for cmd in ("repro", "tune", "apply"):
                rc = quiet(rf.main, w.args(cmd))
                assert rc == 0, cmd
        finally:
            rf.COMBOS = old
        fc = json.load(open(os.path.join(w.out, "frozen_config.json"), encoding="utf-8"))
        assert fc["combo"]["id"] == "R20", fc["combo"]
        quiet(decide.run, os.path.join(w.out, "rank_val_probs.csv"), w.out)
        self.p03 = os.path.join(root, "p03")
        self.out = os.path.join(root, "out")
        self.zips = os.path.join(root, "zips")
        os.makedirs(self.p03)
        os.makedirs(self.zips)
        self.rng = np.random.default_rng(seed + 100)
        self.val_y = np.array([lv for _, games, lv in w.questions for _ in games])
        self.r10 = os.path.join(root, "fallback_R10.json")
        pseudo.write_json(self.r10, {"format": p03.R10_FORMAT,
                                     "combo": {"id": "R10", "source": "C5", "pool": "log", "calibrated": True,
                                               "opening": False},
                                     "inner_oof_rank_balanced": 0.2, "params": None})
        self.strength = dict({"a": 0.9, "b": 0.8}, **(strength or {}))
        self.modes = dict({"a": "ok", "b": "ok"}, **(modes or {}))
        self.arm_src = {}
        for x in arms:
            src = os.path.join(root, "src", f"rank_{x}")
            self.write_arm_tree(src, x)
            self.arm_src[x] = src
            if write_arms:
                shutil.copytree(src, os.path.join(self.p03, f"rank_{x}"))

    # ---- arm result trees in the notebook's layout
    def logits(self, y, S, strength, signal=True):
        k = np.arange(10)
        base = -strength * np.abs(k[None, :] - np.asarray(y)[:, None]).astype(np.float64) if signal else 0.0
        z = base + self.rng.normal(0, 1.0, size=(len(y), 10)) if S is None else \
            (np.asarray(base)[:, None, :] if signal else 0.0) + self.rng.normal(0, 1.0, size=(len(y), S, 10))
        return np.asarray(z, dtype=np.float32)

    def write_arm_tree(self, d, x):
        w = self.w
        mode, st = self.modes[x], self.strength[x]
        dd = os.path.join(d, "dumps")
        os.makedirs(os.path.join(d, "ckpt"))
        os.makedirs(dd)
        shutil.copy(os.path.join(w.dumps, "inner_keys.npz"), dd)
        shutil.copy(os.path.join(w.dumps, "val_keys.npz"), dd)
        wsha = hashlib.sha256(f"final-{x}".encode()).hexdigest()
        ok_sig = mode != "wiring_bug"
        inner = self.logits(w.label, 8, st, signal=ok_sig)
        inner_sw = self.logits(w.label, None, st * 1.5, signal=not ok_sig)
        val_ok = mode not in ("wiring_bug", "val_kappa_bug")
        val = self.logits(self.val_y, 8, st, signal=val_ok)
        val_sw = self.logits(self.val_y, None, st * 1.5, signal=not val_ok)
        val[~w.val_parsed] = np.nan
        val_sw[~w.val_parsed] = np.nan
        stems = {f"inner_p03{x}_d4": (inner, 8), f"inner_p03{x}_s0_swap": (inner_sw, None),
                 f"val_p03{x}_d4": (val, 8), f"val_p03{x}_s0_swap": (val_sw, None)}
        for stem, (z, S) in stems.items():
            arrays = {"logits": z, "weights_sha256": np.array(wsha)}
            if S:
                arrays["syms"] = np.array(S)
            save_npz(os.path.join(dd, stem + ".npz"), **arrays)
        rows = {"inner_keys": len(w.label), "val_keys": len(self.val_y)}
        for stem in p03.arm_stems(x):
            path = os.path.join(dd, stem + ".npz")
            is_d = stem not in rows
            n = rows.get(stem, len(w.label) if stem.startswith("inner") else len(self.val_y))
            mk = {"file": f"dumps/{stem}.npz", "rows": n, "syms": (8 if stem.endswith("_d4") else (1 if is_d else None)),
                  "sha256": pseudo.sha256_file(path),
                  "weights": f"prereg03/rank_{x}/ckpt/final.pth" if is_d else None,
                  "weights_sha256": wsha if is_d else None, "gpu_name": None}
            with open(os.path.join(dd, f"_DONE_{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(mk, f)
        cfg = {"smoke": True, "arm": x.upper(), "recipe": {"ARM": x.upper(), "T": T_SMOKE},
               "training_complete": True, "final_sha256": wsha}
        for name, obj in (("config.json", cfg),
                          ("run_log.json", {"runs": [{"run": 1, "status": "completed"}]}),
                          ("env.json", {}), ("timings.json", {}), ("leak_check.json", {}),
                          ("readonly_before.json", {"runs": [{"files": {}}]}),
                          ("readonly_after.json", {"runs": [{"same_as_before": True}]})):
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                json.dump(obj, f)
        with open(os.path.join(d, "ckpt", "history.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["pass", "global_step", "train_acc", "inner_acc", "accepted"])
            for p in range(1, 5):
                wr.writerow([p, p * T_SMOKE // 4, 30.0 + p, 25.0, 100])

    def make_zip(self, x, name=None, extra=None, drop=None, prefix=None):
        src = self.arm_src[x]
        path = os.path.join(self.zips, name or f"prereg03_rank_{x}_results.zip")
        pre = prefix if prefix is not None else f"rank_{x}/"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for root, _dirs, files in os.walk(src):
                for f in sorted(files):
                    rel = os.path.relpath(os.path.join(root, f), src).replace(os.sep, "/")
                    if drop and rel == drop:
                        continue
                    z.write(os.path.join(root, f), pre + rel)
            for member, data in (extra or {}).items():
                z.writestr(member, data)
        return path

    def args(self, cmd, *extra):
        w = self.w
        base = [cmd, "--smoke", "--p02-dir", w.p02, "--p02-local-dir", w.out, "--p03-dir", self.p03,
                "--out-dir", self.out, "--split", w.split_csv, "--fallback-r10", self.r10]
        if cmd == "apply":
            base += ["--questions", w.q_csv]
        return base + list(extra)

    def run(self, cmd, *extra):
        return quiet(p03.main, self.args(cmd, *extra))

    def read(self, name):
        with open(os.path.join(self.out, name), encoding="utf-8") as f:
            return json.load(f)


class TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_p03_rank_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ================================================================ hygiene and reuse
class Hygiene(unittest.TestCase):
    def test_no_label_file_name_in_p03_tool(self):
        word = "answer" + "s"
        with open(os.path.join(TOOLS, "p03_rank_fuse.py"), encoding="utf-8") as f:
            self.assertNotIn(word, f.read().lower())

    def test_p02_functions_and_constants_are_imported_not_copied(self):
        for name in ("basis", "log_softmax", "softmax", "rule_b_np", "pool_matrix", "pooled", "QSet",
                     "question_logits", "fit", "predict", "metrics", "PlayerBootstrap", "load_dump",
                     "load_val_keys", "check_val_logits", "build_qset", "read_question_meta"):
            self.assertIs(getattr(p03, name), getattr(rf, name), name)
        for name in ("GATE", "SELECT_TOL", "PENALTY", "BOUNDS", "STARTS", "N_BOOT", "BOOT_SEED", "P02E9_MIN_ACC"):
            self.assertEqual(getattr(p03, name), getattr(rf, name), name)
        self.assertEqual((p03.GATE, p03.SELECT_TOL, p03.P02E9_MIN_ACC, p03.N_BOOT, p03.BOOT_SEED),
                         (0.010, 0.003, 24.15, 1000, 0))

    def test_registered_constants(self):
        self.assertEqual(p03.T_STEPS, 12 * 2814)
        self.assertEqual(sum(p03.VAL_LEVEL_QUESTIONS.values()), 1589)
        self.assertEqual([c for c, _ in p03.COMBOS], ["P00", "P01", "P02", "P03", "P04", "P05", "P06"])
        self.assertEqual(dict(p03.COMBOS)["P03"], ("C5", "C6A", "C6B"))
        self.assertEqual(p03.SOURCES["C5"], rf.SOURCES["C5"])


# ================================================================ sources and combinations
class Combination(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.z = {n: rng.normal(size=(50, 8, 10)).astype(np.float32) for n in ("p02e9_d4", "p03a_d4", "p03b_d4")}

    def test_p00_is_p02_c5_exactly(self):
        self.assertTrue(np.array_equal(p03.combo_logprobs(("C5",), self.z), rf.source_logprobs("C5", self.z)))

    def test_concatenation_with_equal_weights(self):
        got = p03.combo_logprobs(("C5", "C6A", "C6B"), self.z)
        parts = [rf.log_softmax(self.z[n][:, s, :].astype(np.float64))
                 for n in ("p02e9_d4", "p03a_d4", "p03b_d4") for s in range(8)]
        want = rf.log_softmax(np.mean(parts, axis=0))
        np.testing.assert_allclose(got, want, atol=1e-12)
        # equal weight: the mean of the three per-source means (8 orientations each) is the same thing
        per_source = [np.mean(parts[i * 8:(i + 1) * 8], axis=0) for i in range(3)]
        np.testing.assert_allclose(got, rf.log_softmax(np.mean(per_source, axis=0)), atol=1e-12)
        # the order of the sources does not matter
        np.testing.assert_allclose(p03.combo_logprobs(("C6B", "C5", "C6A"), self.z), got, atol=1e-12)
        # two sources: each counts half
        two = p03.combo_logprobs(("C5", "C6A"), self.z)
        m5 = np.mean(parts[:8], axis=0)
        ma = np.mean(parts[8:16], axis=0)
        np.testing.assert_allclose(two, rf.log_softmax(0.5 * m5 + 0.5 * ma), atol=1e-12)

    def test_source_without_eight_orientations_refused(self):
        z = dict(self.z, p03a_d4=self.z["p03a_d4"][:, :1, :])
        with self.assertRaises(p03.P03Error):
            p03.combo_logprobs(("C5", "C6A"), z)

    def test_available_combos_keep_order_and_drop(self):
        got = p03.available_combos(["C5", "C6A"])
        self.assertEqual([g[0] for g in got], ["P00", "P01", "P02", "P03", "P04", "P05", "P06"])
        self.assertEqual([g[0] for g in got if g[2] == "ok"], ["P00", "P01", "P04"])
        got = p03.available_combos(["C5"])
        self.assertEqual([g[0] for g in got if g[2] == "ok"], ["P00"])

    def test_selection_tolerance_and_order(self):
        def res(vals):
            return [{"id": cid, "status": "dropped" if v is None else "ok",
                     "oof": None if v is None else {"rank_balanced": v}} for (cid, _), v in zip(p03.COMBOS, vals)]
        chosen, rec = p03.select(res([0.580, 0.583, 0.590, None, 0.5925, 0.5900, 0.5]))
        self.assertEqual(chosen["id"], "P02")   # best P04 0.5925; P02 0.590 is within 0.003 and earlier
        self.assertEqual(rec["best_id"], "P04")
        self.assertEqual(rec["within_tolerance"], ["P02", "P04", "P05"])
        chosen, rec = p03.select(res([0.580, 0.586, 0.590, 0.5801, None, None, None]))
        self.assertEqual(chosen["id"], "P02")   # 0.590 - 0.586 = 0.004 > 0.003: only P02 itself is within
        self.assertEqual(rec["within_tolerance"], ["P02"])
        chosen, _ = p03.select(res([0.600, 0.603, None, None, None, None, None]))
        self.assertEqual(chosen["id"], "P00")   # P00 within 0.003 of the best -> P00 (not submitted)

    def test_selection_exact_tolerance_edge(self):
        vals = [0.580, 0.587, 0.590] + [None] * 4
        res = [{"id": cid, "status": "dropped" if v is None else "ok", "oof": None if v is None else
                {"rank_balanced": v}} for (cid, _), v in zip(p03.COMBOS, vals)]
        chosen, _ = p03.select(res)
        self.assertEqual(chosen["id"], "P01")   # 0.590 - 0.587 = 0.003 (float slack 1e-12)


# ================================================================ gates and recorded estimates
class Gates(unittest.TestCase):
    def world(self, seed, shift):
        rng = np.random.default_rng(seed)
        n_players = 60
        level_of = np.repeat(np.arange(10), 6)
        player = np.repeat(np.arange(n_players), 20)
        y = level_of[player]
        p00 = rng.choice([0.0, math.exp(-1), 1.0], size=len(player), p=[0.3, 0.3, 0.4])
        new = p00.copy()
        flip = rng.random(len(player)) < shift
        new[flip] = 1.0
        return player, y, p00, new, n_players

    def rb(self, sc, y):
        return float(np.mean([sc[y == i].mean() for i in range(10)]))

    def test_gates_pass_and_fail(self):
        player, y, p00, new, n = self.world(1, 0.15)
        boot = rf.PlayerBootstrap(player, y, n)
        g = p03.gate_record("P01", self.rb(new, y), self.rb(p00, y), new, p00, boot)
        self.assertTrue(g["G1"]["passed"] and g["G2"]["passed"] and g["G3"]["passed"] and g["passed"])
        d = boot.stats(new) - boot.stats(p00)
        self.assertEqual(g["G2"]["ci95"], [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        self.assertAlmostEqual(g["G3"]["question_mean_difference"], float(np.mean(new - p00)), places=15)
        # G1 fails below +0.010
        player, y, p00, new2, n = self.world(1, 0.004)
        g = p03.gate_record("P01", self.rb(new2, y), self.rb(p00, y), new2, p00, rf.PlayerBootstrap(player, y, n))
        self.assertFalse(g["G1"]["passed"])
        self.assertFalse(g["passed"])
        # G3 fails when the question mean drops although the rank-balanced mean rises
        sc = p00.copy()
        lv0 = np.nonzero(y == 0)[0]
        others = np.nonzero(y != 0)[0]
        sc[lv0] = 1.0            # +, weighted 1/10 of the rank-balanced mean
        sc[others[: len(lv0) * 3]] = 0.0
        g = p03.gate_record("P02", self.rb(sc, y), self.rb(p00, y), sc, p00, rf.PlayerBootstrap(player, y, n))
        self.assertEqual(g["G3"]["passed"], float(np.mean(sc - p00)) >= -1e-12)
        # P00 selected: not submitted
        g = p03.gate_record("P00", 0.6, 0.6, p00, p00, boot)
        self.assertFalse(g["passed"])
        self.assertIsNone(g["G1"])

    def test_g1_float_slack(self):
        player, y, p00, _new, n = self.world(2, 0.0)
        boot = rf.PlayerBootstrap(player, y, n)
        g = p03.gate_record("P03", 0.5 + 0.010 - 5e-13, 0.5, p00 + 0.0, p00, boot)
        self.assertTrue(g["G1"]["passed"])
        g = p03.gate_record("P03", 0.5 + 0.010 - 5e-12, 0.5, p00 + 0.0, p00, boot)
        self.assertFalse(g["G1"]["passed"])

    def test_se_val_by_hand(self):
        rng = np.random.default_rng(5)
        y = np.repeat(np.arange(10), 30)
        d = rng.normal(0.01, 0.4, size=len(y))
        rec = p03.se_val_record(0.012, d, y)
        want = math.sqrt(sum(np.var(d[y == i], ddof=1) / p03.VAL_LEVEL_QUESTIONS[lv]
                             for i, lv in enumerate(LEVELS))) / 10
        self.assertAlmostEqual(rec["se_val"], want, places=15)
        from scipy.stats import norm
        self.assertAlmostEqual(rec["adoption_probability"], float(norm.cdf(0.012 / want - 1.96)), places=12)


# ================================================================ colour checks
class Colour(unittest.TestCase):
    def test_kappa_by_hand(self):
        rng = np.random.default_rng(0)
        lp = rf.log_softmax(rng.normal(size=(9, 10)))
        groups = [[0, 1, 2], [3, 4], [5], [6, 7, 8]]
        got = p03.kappa_questions(lp, groups)
        for q, g in enumerate(groups):
            if len(g) < 2:
                self.assertTrue(np.isnan(got[q]))
                continue
            vals = []
            for i in g:
                others = np.mean([lp[h] for h in g if h != i], axis=0)
                qd = np.exp(others - others.max())
                qd /= qd.sum()
                vals.append(float(np.sum(qd * lp[i])))
            self.assertAlmostEqual(got[q], float(np.mean(vals)), places=12)

    def test_kappa_chunks(self):
        rng = np.random.default_rng(1)
        lp = rf.log_softmax(rng.normal(size=(400, 10)))
        groups = [list(range(i, i + 4)) for i in range(0, 400, 4)]
        a = p03.kappa_questions(lp, groups)
        old = p03.KAPPA_CHUNK
        p03.KAPPA_CHUNK = 7
        try:
            b = p03.kappa_questions(lp, groups)
        finally:
            p03.KAPPA_CHUNK = old
        np.testing.assert_allclose(a, b, atol=1e-13)

    def test_consistent_player_has_higher_kappa(self):
        rng = np.random.default_rng(2)
        k = np.arange(10)
        y = np.repeat(rng.integers(0, 10, size=50), 8)
        good = rf.log_softmax(-1.0 * np.abs(k[None] - y[:, None]) + rng.normal(size=(400, 10)))
        noise = rf.log_softmax(rng.normal(size=(400, 10)))
        groups = [list(range(i, i + 8)) for i in range(0, 400, 8)]
        self.assertGreater(np.mean(p03.kappa_questions(good, groups) - p03.kappa_questions(noise, groups)), 0)

    def test_colour_decision(self):
        stops, rec = p03.colour_decision(25.0, 24.0, 0.01, 0.02)
        self.assertEqual(stops, [])
        self.assertTrue(rec["kappa_requirement_applies"])
        self.assertAlmostEqual(rec["dacc_points"], 1.0)
        stops, rec = p03.colour_decision(25.0, 24.0, 0.01, -0.001)       # applies, val not > 0
        self.assertEqual(len(stops), 1)
        self.assertIn("semantics", stops[0])
        stops, rec = p03.colour_decision(25.0, 24.0, 0.01, 0.0)          # strictly > 0 required
        self.assertEqual(len(stops), 1)
        stops, rec = p03.colour_decision(25.0, 24.7, 0.01, -0.5)         # dacc 0.3 < 0.5: recorded only
        self.assertEqual(stops, [])
        self.assertFalse(rec["kappa_requirement_applies"])
        stops, rec = p03.colour_decision(25.0, 24.0, -0.001, -0.5)       # inner CI includes 0: recorded only
        self.assertEqual(stops, [])
        stops, rec = p03.colour_decision(24.0, 24.2 + 1e-9, None, None)  # swapped better by > 0.2 points
        self.assertTrue(rec["wiring_stop"])
        self.assertIn("wiring", stops[0])
        stops, rec = p03.colour_decision(24.0, 24.19, None, None)
        self.assertFalse(rec["wiring_stop"])

    def test_bootstrap_plain_mean_by_hand(self):
        rng = np.random.default_rng(3)
        level_of = np.array([0, 0, 1, 1, 1, 4])
        player = np.concatenate([np.full(rng.integers(1, 5), p) for p in range(6)])
        y = level_of[player]
        v = rng.normal(size=len(player))
        boot = rf.PlayerBootstrap(player, y, 6)
        got = p03.boot_plain_mean(boot, v)
        for b in (0, 5, 999):
            rows = np.concatenate([d[b] for d in boot.draws])
            num = sum(v[player == p].sum() for p in rows)
            den = sum((player == p).sum() for p in rows)
            self.assertAlmostEqual(got[b], num / den, places=12)


# ================================================================ single-colour records (PREREG_03 rev 3)
def mono_keys(seed=0, n_players=24):
    """Synthetic inner keys and bank players: player j has 3 + 4j games of random colours (some pools < 5, some
    >= 20), levels cycling, folds 0..4."""
    rng = np.random.default_rng(seed)
    rank, gid, pid, col = [], [], [], []
    players = []
    for j in range(n_players):
        lv = LEVELS[j % 10]
        p = "p_%03d" % j
        players.append({"player_id": p, "rank": lv, "fold": j % 5})
        for _ in range(3 + 4 * j):
            rank.append(lv)
            gid.append("g_%06d" % len(gid))
            pid.append(p)
            col.append("B" if rng.random() < 0.5 else "W")
    keys = {"rank": rank, "game_id": gid, "player_id": pid, "color": col, "n": len(gid),
            "label": np.array([LEVELS.index(r) for r in rank])}
    return keys, players


def kappa_by_hand(lp, g):
    vals = []
    for i in g:
        others = np.mean([lp[h] for h in g if h != i], axis=0)
        qd = np.exp(others - others.max())
        qd /= qd.sum()
        vals.append(float(np.sum(qd * lp[i])))
    return float(np.mean(vals))


class SingleColour(unittest.TestCase):
    def test_questions_are_single_colour_and_complete(self):
        keys, players = mono_keys()
        qs = p03.mono_questions(keys, players)
        pools = {}
        for i, (p, c) in enumerate(zip(keys["player_id"], keys["color"])):
            pools.setdefault((p, c), []).append(i)
        seen = {}
        for r, pi, c, k, rows in qs:
            pid = players[pi]["player_id"]
            self.assertIn(k, p03.MONO_KS)
            self.assertEqual(len(rows), k)
            self.assertEqual(len(set(rows)), k)
            self.assertTrue(all(keys["color"][i] == c and keys["player_id"][i] == pid for i in rows))
            seen[(r, pid, c, k)] = seen.get((r, pid, c, k), 0) + 1
        want = {(r, p, c, k): 1 for r in p03.MONO_ROUNDS for (p, c), rows in pools.items()
                for k in p03.MONO_KS if len(rows) >= k}
        self.assertEqual(seen, want)          # every (round, player, colour, k <= pool) exactly once, nothing else
        self.assertTrue(any(k == 20 for *_, k, _ in qs) and any(len(v) < 5 for v in pools.values()))
        self.assertEqual(p03.mono_questions(keys, players), qs)     # deterministic

    def test_salts_by_hand(self):
        keys, players = mono_keys()
        qs = p03.mono_questions(keys, players)
        r, pi, c, k, rows = next(q for q in qs if q[0] == 7 and q[3] == 10)
        pid = players[pi]["player_id"]
        pool = sorted([i for i in range(keys["n"]) if keys["player_id"][i] == pid and keys["color"][i] == c],
                      key=lambda i: hashlib.sha256(f"c10-prereg03|mono|order|{keys['rank'][i]}|{keys['game_id'][i]}"
                                                   .encode()).hexdigest())
        import random
        seed = int(hashlib.sha256(f"c10-prereg03|mono|q|7|{pid}|{c}|10".encode()).hexdigest()[:16], 16)
        self.assertEqual(rows, [pool[j] for j in random.Random(seed).sample(range(len(pool)), 10)])

    def test_a_question_does_not_depend_on_other_players(self):
        keys, players = mono_keys()
        qs = p03.mono_questions(keys, players)
        drop = players[3]["player_id"]
        keep = [i for i in range(keys["n"]) if keys["player_id"][i] != drop]
        k2 = {n: [keys[n][i] for i in keep] for n in ("rank", "game_id", "player_id", "color")}
        qs2 = p03.mono_questions(k2, players)
        by_gid = lambda kk, q: (q[0], q[1], q[2], q[3], [kk["game_id"][i] for i in q[4]])  # noqa: E731
        self.assertEqual([by_gid(keys, q) for q in qs if q[1] != 3], [by_gid(k2, q) for q in qs2])

    def test_set_counts_and_sha(self):
        keys, players = mono_keys()
        ms = p03.mono_set({"keys": keys, "players": players})
        self.assertEqual(ms["counts"]["questions"], len(ms["groups"]))
        self.assertEqual(sum(ms["counts"]["questions_by_k"].values()), len(ms["groups"]))
        self.assertEqual(ms["counts"]["questions_by_colour"]["B"] + ms["counts"]["questions_by_colour"]["W"],
                         len(ms["groups"]))
        self.assertTrue(np.array_equal(ms["y"], [LEVELS.index(players[p]["rank"]) for p in ms["player"]]))
        self.assertTrue(np.array_equal(ms["fold"], [players[p]["fold"] for p in ms["player"]]))
        keys2 = dict(keys, game_id=list(keys["game_id"]))
        keys2["game_id"][-1] = "g_other"
        self.assertNotEqual(p03.mono_set({"keys": keys2, "players": players})["sha256"], ms["sha256"])
        # a game whose level differs from its player's is refused
        keys3 = dict(keys, rank=list(keys["rank"]))
        keys3["rank"][0] = LEVELS[5]
        with self.assertRaises(p03.P03Error):
            p03.mono_questions(keys3, players)

    def test_val_colour_subquestions(self):
        vk = {"groups": [("q1", [0, 1, 2, 3]), ("q2", [4, 5, 6]), ("q3", [7, 8])],
              "color": ["B", "W", "B", "W", "B", "B", "W", "W", "B"],
              "parsed": np.array([True, True, True, False, True, True, True, False, True])}
        got = p03.val_colour_subquestions(vk)
        # q1: B {0, 2}, W {1} (3 unparsed) -> only B; q2: B {4, 5}, W {6} -> only B; q3: W {} , B {8} -> none
        self.assertEqual(got, [(0, "B", [0, 2]), (1, "B", [4, 5])])

    def test_question_bootstrap_by_hand(self):
        rng = np.random.default_rng(4)
        qidx = np.array([0, 0, 1, 2, 2, 5])
        v = rng.normal(size=6)
        got = p03.question_boot_mean(qidx, v)
        uq = [0, 1, 2, 5]
        d = np.random.default_rng(p03.BOOT_SEED).integers(0, 4, size=(p03.N_BOOT, 4))
        for b in (0, 17, 999):
            rows = [i for j in d[b] for i in range(6) if qidx[i] == uq[j]]
            self.assertAlmostEqual(got[b], float(np.mean(v[rows])), places=12)

    def test_kappa_reference_by_hand_and_condition(self):
        keys, players = mono_keys(seed=1)
        ms = p03.mono_set({"keys": keys, "players": players})
        rng = np.random.default_rng(5)
        k10 = np.arange(10)
        y = keys["label"]
        lp_c = rf.log_softmax(-1.0 * np.abs(k10[None] - y[:, None]) + rng.normal(size=(keys["n"], 10)))
        lp_s = rf.log_softmax(rng.normal(size=(keys["n"], 10)))
        vk = {"groups": [("q1", [0, 1, 2, 3]), ("q2", [4, 5, 6, 7, 8])],
              "color": ["B", "W", "B", "W", "B", "B", "W", "W", "B"], "parsed": np.ones(9, dtype=bool)}
        vc = rf.log_softmax(rng.normal(size=(9, 10)) + 2 * (k10 == 3))
        vs = rf.log_softmax(rng.normal(size=(9, 10)))
        boot = rf.PlayerBootstrap(ms["player"], ms["y"], ms["n_players"])
        rec = p03.kappa_single_colour(lp_c, lp_s, ms, boot, 1.0, vk, vc, vs)
        dk = [kappa_by_hand(lp_c, g) - kappa_by_hand(lp_s, g) for g in ms["groups"]]
        self.assertAlmostEqual(rec["inner"]["dkappa"], float(np.mean(dk)), places=10)
        for col in ("B", "W"):
            mk = ms["colour"] == col
            self.assertAlmostEqual(rec["inner"]["by_colour"][col]["dkappa"], float(np.mean(np.array(dk)[mk])),
                                   places=10)
        np.testing.assert_allclose(rec["inner"]["ci95"], p03.ci95(p03.boot_plain_mean(boot, np.array(dk))),
                                   atol=1e-12)
        subs = [[0, 2], [1, 3], [4, 5, 8], [6, 7]]
        dv = [kappa_by_hand(vc, g) - kappa_by_hand(vs, g) for g in subs]
        self.assertAlmostEqual(rec["val_sub_questions"]["dkappa"], float(np.mean(dv)), places=10)
        self.assertEqual(rec["val_sub_questions"]["sub_questions"], 4)
        self.assertAlmostEqual(rec["val_sub_questions"]["by_colour"]["W"]["dkappa"], (dv[1] + dv[3]) / 2, places=10)
        ref = rec["test_day_reference"]
        self.assertEqual(ref["single_colour_condition"], rec["inner"]["ci95"][0] > 0)
        self.assertTrue(ref["single_colour_condition"])
        self.assertFalse(p03.kappa_single_colour(lp_c, lp_s, ms, boot, 0.49, vk, vc, vs)["test_day_reference"]
                         ["single_colour_condition"])                   # dacc below 0.5 points
        self.assertFalse(p03.kappa_single_colour(lp_s, lp_s, ms, boot, 5.0, vk, vc, vs)["test_day_reference"]
                         ["single_colour_condition"])                   # dkappa 0: CI lower bound not > 0
        neg = p03.kappa_single_colour(lp_c, lp_s, ms, boot, 1.0, vk, vs, vc)["test_day_reference"]
        self.assertFalse(neg["val_sub_question_dkappa_positive"])

    def test_single_colour_score_block(self):
        """PREREG_03 rev 3 11(c): condition true with a validation sub-question dkappa not > 0 withholds H3R."""
        def ref(cond, pos):
            return {"kappa_single_colour": {"test_day_reference": {
                "single_colour_condition": cond, "val_sub_question_dkappa_positive": pos, "dacc_points": 1.0,
                "inner_dkappa_ci95_lower": 0.01 if cond else -0.01,
                "val_sub_question_dkappa": None if pos is None else (0.02 if pos else -0.02)}}}
        block = p03.single_colour_score_block
        ok, bad = ref(True, True), ref(True, False)
        self.assertEqual(block("P01", {"C6A": ok}), [])
        self.assertEqual(block("P01", {"C6A": ref(False, False)}), [])        # condition false: record only
        self.assertEqual(block("P01", {"C6A": ref(False, None)}), [])
        r = block("P01", {"C6A": bad})
        self.assertEqual(len(r), 1)
        self.assertTrue(r[0].startswith("C6A:") and "H3R is not run" in r[0], r)
        self.assertEqual(len(block("P01", {"C6A": ref(True, None)})), 1)      # no validation reference at all
        self.assertEqual(len(block("P01", {})), 1)                            # a C6 source without a reference
        self.assertEqual(len(block("P01", {"C6A": {"kappa_single_colour": None}})), 1)
        self.assertEqual(block("P00", {}), [])                                # P00 has no C6 source
        self.assertEqual(len(block("P03", {"C6A": bad, "C6B": bad})), 2)
        self.assertEqual([x[:4] for x in block("P03", {"C6A": ok, "C6B": bad})], ["C6B:"])
        self.assertEqual(block("P04", {"C6A": ok, "C6B": bad}), [])            # C6B is not in P04
        self.assertEqual(len(block("P05", {"C6A": ok, "C6B": bad})), 1)


class ToolState(unittest.TestCase):
    def test_porcelain_paths_keep_their_first_letter(self):
        """git status --porcelain starts a modified file's line with a blank; the first path must stay whole."""
        old = p03.git

        def fake_git(args, binary=False, repo=None):
            if args[:2] == ["status", "--porcelain"]:
                out = b" M tools/p03_rank_fuse.py\n?? tools/decide.py\n"
                return 0, (out if binary else out.decode().strip())
            return old(args, binary=binary, repo=repo)
        p03.git = fake_git
        try:
            st = p03.tool_state()
        finally:
            p03.git = old
        self.assertEqual(st["dirty_or_untracked"], ["tools/p03_rank_fuse.py", "tools/decide.py"])


# ================================================================ frozen tool hashes
class FrozenTools(TmpDir):
    def copy_tools(self):
        d = os.path.join(self.tmp, "tools")
        os.makedirs(d)
        for f in p03.FROZEN_TOOL_FILES:
            shutil.copy(os.path.join(TOOLS, f), d)
        return d

    def test_real_repo_passes(self):
        rec = p03.frozen_tools_check()
        self.assertEqual(set(rec["files_sha256"]), set(p03.FROZEN_TOOL_FILES))

    def test_edited_p02_file_refused(self):
        d = self.copy_tools()
        self.assertIn("p02_rank_fuse.py", p03.frozen_tools_check(d)["files_sha256"])
        with open(os.path.join(d, "p02_rank_fuse.py"), "a", encoding="utf-8") as f:
            f.write("\nGATE = 0.005\n")
        with self.assertRaises(p03.P03Error) as cm:
            p03.frozen_tools_check(d)
        self.assertIn("p02_rank_fuse.py", str(cm.exception))

    def test_every_frozen_file_is_checked(self):
        for name in p03.FROZEN_TOOL_FILES:
            d = self.copy_tools()
            with open(os.path.join(d, name), "ab") as f:
                f.write(b"# edit\n")
            with self.assertRaises(p03.P03Error, msg=name):
                p03.frozen_tools_check(d)
            shutil.rmtree(d)

    def test_line_ends_ignored_and_missing_refused(self):
        d = self.copy_tools()
        path = os.path.join(d, "decide.py")
        with open(path, "rb") as f:
            raw = f.read().replace(b"\r\n", b"\n")
        with open(path, "wb") as f:
            f.write(raw.replace(b"\n", b"\r\n"))
        p03.frozen_tools_check(d)          # CRLF only: accepted
        os.remove(os.path.join(d, "c10data.py"))
        with self.assertRaises(p03.P03Error):
            p03.frozen_tools_check(d)

    def test_head_blob_is_the_reference(self):
        d = self.copy_tools()
        # a HEAD whose p02_moves.py differs from the working file is refused too
        def blob(rel):
            b = p03.head_blob(rel)
            return b + b"# other\n" if rel.endswith("p02_moves.py") else b
        with self.assertRaises(p03.P03Error):
            p03.frozen_tools_check(d, blob=blob)


class Committed(TmpDir):
    def test_tracked_clean_in_a_scratch_repo(self):
        repo = os.path.join(self.tmp, "repo")
        d = os.path.join(repo, "results", "prereg03", "rank")
        os.makedirs(d)
        path = os.path.join(d, "frozen_config.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}\n")
        g = ["git", "-C", repo]
        subprocess.run(g + ["init", "-q"], check=True)
        old = p03.REPO
        p03.REPO = repo
        try:
            with self.assertRaises(p03.P03Error):          # untracked
                p03.require_tracked_clean([path], "x")
            subprocess.run(g + ["add", "-A"], check=True)
            subprocess.run(g + ["-c", "user.name=kuotunyu", "-c",
                                "user.email=61350295+kuotunyu@users.noreply.github.com",
                                "-c", "commit.gpgsign=false", "commit", "-q", "-m", "t"], check=True)
            self.assertEqual(p03.require_tracked_clean([path], "x")["files"], 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"changed": 1}\n')
            with self.assertRaises(p03.P03Error):          # changed
                p03.require_tracked_clean([path], "x")
            with self.assertRaises(p03.P03Error):          # outside
                p03.require_tracked_clean([os.path.join(self.tmp, "elsewhere.json")], "x")
        finally:
            p03.REPO = old


# ================================================================ unpack
class Unpack(TmpDir):
    def setUp(self):
        super().setUp()
        self.W = P03World(self.tmp, arms=("a",), write_arms=False)

    def unpack(self, zpath, x="a", dest=None):
        dest = dest or os.path.join(self.W.p03, f"rank_{x}")
        return quiet(p03.main, ["unpack", "--arm", x, zpath, "--smoke", "--dest", dest])

    def test_unpack_ok_and_no_overwrite(self):
        z = self.W.make_zip("a")
        self.assertEqual(self.unpack(z), 0)
        rec = json.load(open(os.path.join(self.W.p03, "unpack_rank_a.json"), encoding="utf-8"))
        self.assertEqual(rec["zip_sha256"], pseudo.sha256_file(z))
        self.assertEqual(set(rec["members"]), set(p03.arm_required("a")))
        for rel in p03.arm_required("a"):
            self.assertTrue(os.path.exists(os.path.join(self.W.p03, "rank_a", *rel.split("/"))))
        self.assertEqual(self.unpack(z), 1)        # already unpacked

    def test_pause_and_refusals_before_computing_are_not_stops(self):
        # a Drive-space pause or a run that ended in a refusal (status None), then a completed resume: a result
        rl = os.path.join(self.W.arm_src["a"], "run_log.json")
        json.dump({"runs": [{"run": 1, "status": None}, {"run": 2, "status": "drive_space_pause"},
                            {"run": 3, "status": "completed"}], "status": "completed", "resume_count": 2},
                  open(rl, "w", encoding="utf-8"))
        self.assertEqual(self.unpack(self.W.make_zip("a")), 0)

    def test_refusals(self):
        W = self.W
        cases = {
            "weights": W.make_zip("a", name="w.zip", extra={"rank_a/ckpt/final.pth": b"x"}),
            "state": W.make_zip("a", name="s.zip", extra={"rank_a/ckpt/state.pt": b"x"}),
            "tmp": W.make_zip("a", name="t.zip", extra={"rank_a/dumps/x.npz.tmp": b"x"}),
            "traversal": W.make_zip("a", name="tr.zip", extra={"rank_a/../evil.json": b"{}"}),
            "outside": W.make_zip("a", name="o.zip", extra={"rank_b/x.json": b"{}"}),
            "missing": W.make_zip("a", name="m.zip", drop="dumps/val_p03a_s0_swap.npz"),
            "failed": W.make_zip("a", name="prereg03_rank_a_results_failed.zip"),
            "wrong prefix": W.make_zip("a", name="p.zip", prefix="rank/"),
            "label name": W.make_zip("a", name="l.zip", extra={"rank_a/rank_val_" + "answer" + "s.csv": b""}),
        }
        for what, z in cases.items():
            self.assertEqual(self.unpack(z), 1, what)
            self.assertFalse(os.path.exists(os.path.join(W.p03, "rank_a")), what)
        # zip of arm A given as arm B
        self.assertEqual(self.unpack(W.make_zip("a", name="ab.zip"), x="b"), 1)
        # marker SHA-256 differs from the dump
        src = W.arm_src["a"]
        mk = os.path.join(src, "dumps", "_DONE_val_p03a_d4.json")
        obj = json.load(open(mk, encoding="utf-8"))
        good = obj["sha256"]
        obj["sha256"] = "0" * 64
        json.dump(obj, open(mk, "w", encoding="utf-8"))
        self.assertEqual(self.unpack(W.make_zip("a", name="mk.zip")), 1)
        obj["sha256"] = good
        json.dump(obj, open(mk, "w", encoding="utf-8"))
        # a run that did not complete
        rl = os.path.join(src, "run_log.json")
        json.dump({"runs": [{"run": 1, "status": "nan_stop"}]}, open(rl, "w", encoding="utf-8"))
        self.assertEqual(self.unpack(W.make_zip("a", name="ns.zip")), 1)
        # stopped earlier in the same Drive folder, then run again to completion (PREREG_03 1.3: not a result)
        for i, st in enumerate(p03.STOP_STATUSES):
            json.dump({"runs": [{"run": 1, "status": st}, {"run": 2, "status": "completed"}], "status": "completed"},
                      open(rl, "w", encoding="utf-8"))
            self.assertEqual(self.unpack(W.make_zip("a", name=f"rerun{i}.zip")), 1, st)
        json.dump({"runs": [{"run": 1, "status": "completed"}]}, open(rl, "w", encoding="utf-8"))
        cfg = os.path.join(src, "config.json")
        cfg_obj = json.load(open(cfg, encoding="utf-8"))
        json.dump(dict(cfg_obj, stopped={"status": "time_stop", "run": 1}), open(cfg, "w", encoding="utf-8"))
        self.assertEqual(self.unpack(W.make_zip("a", name="cs.zip")), 1)          # config.json keeps the stop
        json.dump(cfg_obj, open(cfg, "w", encoding="utf-8"))
        self.assertEqual(self.unpack(W.make_zip("a", name="sf.zip", extra={"rank_a/STOPPED.json": b"{}"})), 1)
        self.assertFalse(os.path.exists(os.path.join(W.p03, "rank_a")))
        # --smoke takes SMOKE results only; smoke dest must be outside the repository
        cfg = os.path.join(src, "config.json")
        json.dump({"smoke": False}, open(cfg, "w", encoding="utf-8"))
        self.assertEqual(self.unpack(W.make_zip("a", name="real.zip")), 1)
        json.dump({"smoke": True}, open(cfg, "w", encoding="utf-8"))
        self.assertEqual(self.unpack(W.make_zip("a", name="in.zip"),
                                     dest=os.path.join(REPO, "results", "prereg03", "rank_a")), 1)
        before = repo_prereg03_status()
        self.assertEqual(quiet(p03.main, ["unpack", "--arm", "a", W.make_zip("a", name="nd.zip"), "--smoke"]), 1)
        self.assertEqual(repo_prereg03_status(), before)
        # real mode refuses a SMOKE zip before touching the repository (the real rank_a/ is committed since e55b019)
        self.assertEqual(quiet(p03.main, ["unpack", "--arm", "a", W.make_zip("a", name="r.zip")]), 1)
        self.assertEqual(repo_prereg03_status(), before)


# ================================================================ end to end (--smoke)
class EndToEnd(TmpDir):
    def test_selfcheck_reproduces_p02_r20(self):
        W = P03World(self.tmp)
        self.assertEqual(W.run("selfcheck"), 0)
        rep = W.read("selfcheck.json")
        self.assertTrue(rep["passed"])
        for k in ("per_game_logprobs_equal_p02_source_logprobs_C5", "inner_oof_equals_R20",
                  "refit_params_equal_frozen", "val_probs_equal_rank_val_probs"):
            self.assertIs(rep["checks"][k]["ok"], True, k)
        tune = json.load(open(os.path.join(W.w.out, "tune_report.json"), encoding="utf-8"))
        r20 = next(c for c in tune["combinations"] if c["id"] == "R20")
        self.assertEqual(rep["P00"]["oof"]["rank_balanced"], r20["oof"]["rank_balanced"])
        # a frozen parameter moved by 1e-9 is caught
        fpath = os.path.join(W.w.out, "frozen_config.json")
        with open(fpath, "rb") as f:
            original = f.read()
        fc = json.loads(original)
        fc["params"]["a"] += 1e-9
        pseudo.write_json(fpath, fc)
        self.assertEqual(W.run("selfcheck"), 1)
        rep = W.read("selfcheck.json")
        self.assertEqual(rep["failures"], ["refit_params_equal_frozen"])
        with open(fpath, "wb") as f:
            f.write(original)
        self.assertEqual(W.run("selfcheck"), 0)
        # a validation probability moved by 1e-9 is caught too (probability mass kept, no negative value)
        vpath = os.path.join(W.w.out, "rank_val_probs.csv")
        with open(vpath, encoding="utf-8") as f:
            lines = f.read().splitlines()
        cells = lines[1].split(",")
        p = [float(x) for x in cells[3:]]
        hi, lo = int(np.argmax(p)), int(np.argsort(p)[-2])
        cells[3 + hi] = repr(p[hi] - 1e-9)
        cells[3 + lo] = repr(p[lo] + 1e-9)
        lines[1] = ",".join(cells)
        with open(vpath, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        self.assertEqual(W.run("selfcheck"), 1)
        self.assertIn("val_probs_equal_rank_val_probs", W.read("selfcheck.json")["failures"])

    def test_pipeline_unpack_check_selfcheck_tune_apply(self):
        W = P03World(self.tmp, write_arms=False)
        for x in ("a", "b"):
            self.assertEqual(quiet(p03.main, ["unpack", "--arm", x, W.make_zip(x), "--smoke", "--dest",
                                              os.path.join(W.p03, f"rank_{x}")]), 0)
        # order: nothing before check
        self.assertEqual(W.run("tune"), 1)
        self.assertEqual(W.run("apply"), 1)
        self.assertEqual(W.run("check"), 0)
        cr = W.read("check_report.json")
        self.assertEqual(cr["status"], "ok")
        self.assertEqual(cr["usable_sources"], ["C5", "C6A", "C6B"])
        for s in ("C6A", "C6B"):
            col = cr["sources"][s]["colour"]
            self.assertGreater(col["dacc_points"], 0.5)
            self.assertGreater(col["kappa_inner"]["ci95"][0], 0)
            self.assertGreater(col["kappa_val"]["dkappa"], 0)
            self.assertTrue(col["kappa_requirement_applies"])
        self.assertEqual(W.run("tune"), 1)          # selfcheck missing
        self.assertEqual(W.run("selfcheck"), 0)
        # tune never opens a validation file
        opened = []
        real_open, real_load = builtins.open, np.load

        def spy_open(file, *a, **kw):
            opened.append(str(file))
            return real_open(file, *a, **kw)

        def spy_load(file, *a, **kw):
            opened.append(str(file))
            return real_load(file, *a, **kw)
        builtins.open, np.load = spy_open, spy_load
        try:
            rc = W.run("tune")
        finally:
            builtins.open, np.load = real_open, real_load
        self.assertEqual(rc, 0)
        bases = [os.path.basename(p) for p in opened]
        self.assertTrue(any(b.startswith("inner_p03a") for b in bases))
        self.assertFalse([b for b in bases if b.startswith("val_") or "rank_val" in b or "opening_val" in b],
                         bases)
        tune = W.read("tune_report.json")
        fc = W.read("frozen_config.json")
        self.assertEqual([c["id"] for c in tune["combinations"]], [c for c, _ in p03.COMBOS])
        self.assertTrue(all(c["status"] == "ok" for c in tune["combinations"]))
        ok = {c["id"]: c["oof"]["rank_balanced"] for c in tune["combinations"]}
        best = max(ok.values())
        first = next(i for i in ok if best - ok[i] <= p03.SELECT_TOL + p03.FLOAT_EPS)
        self.assertEqual(fc["combo"]["id"], first)
        self.assertEqual(fc["format"], "c10-prereg03 rank frozen_config v1")
        sc = W.read("selfcheck.json")
        self.assertEqual(ok["P00"], sc["P00"]["oof"]["rank_balanced"])
        g = fc["gates"]
        if first != "P00":
            self.assertAlmostEqual(g["G1"]["delta"], ok[first] - ok["P00"], places=15)
            self.assertEqual(g["passed"], g["G1"]["passed"] and g["G2"]["passed"] and g["G3"]["passed"])
            self.assertIsNotNone(fc["se_val"]["se_val"])
        self.assertEqual(fc["score_allowed"], bool(g["passed"]) and first != "P00")
        self.assertIn(fc["fail_opening"]["use"], ("noopen", "R10"))
        self.assertEqual(fc["fail_opening"]["use"] == "noopen",
                         fc["no_opening"]["oof_rank_balanced"] > 0.2 + p03.FLOAT_EPS)
        self.assertEqual(fc["no_opening"]["params"]["w"], 0.0)
        self.assertEqual(W.run("tune"), 0)          # smoke may rerun; real mode refuses a tracked frozen config
        self.assertEqual(W.run("apply"), 0)
        app = W.read("apply_report.json")
        self.assertEqual(app["score_allowed"], fc["score_allowed"])
        recs = decide.read_probs(os.path.join(W.out, "rank_val_probs.csv"))
        self.assertEqual([r["question_id"] for r in recs], [q for q, _, _ in W.w.questions])
        self.assertTrue(recs[3]["missing"])
        # independent recomputation of question 0 of the frozen combination
        n0 = len(W.w.questions[0][1])
        z = {"p02e9_d4": W.w.val_logits["p02e9_d4"][:n0]}
        for x in ("a", "b"):
            with np.load(os.path.join(W.p03, f"rank_{x}", "dumps", f"val_p03{x}_d4.npz")) as zz:
                z[f"p03{x}_d4"] = zz["logits"][:n0]
        lp = p03.combo_logprobs(tuple(fc["combo"]["sources"]), z)
        ov = p03.load_opening(os.path.join(W.w.out, "opening_val.npz"), "val")
        has = np.asarray(ov["has"])[:n0]
        mop = np.asarray(ov["o"])[:n0][has].mean(0) if has.any() else np.zeros(10)
        par = fc["params"]
        u = lp.mean(0) + par["w"] * mop + np.array(par["b"])
        want = rf.softmax((math.exp(par["a"]) * (n0 / 12) ** par["gamma"] * u)[None])[0]
        np.testing.assert_allclose(recs[0]["probs"], want, atol=1e-12)

    def test_check_stops_on_wiring_and_val_kappa(self):
        W = P03World(self.tmp, modes={"a": "wiring_bug", "b": "val_kappa_bug"})
        self.assertEqual(W.run("check"), 1)
        cr = W.read("check_report.json")
        self.assertEqual(cr["status"], "stop")
        self.assertEqual(cr["sources"]["C6A"]["status"], "stop")
        self.assertTrue(cr["sources"]["C6A"]["colour"]["wiring_stop"])
        self.assertEqual(cr["sources"]["C6B"]["status"], "stop")
        self.assertTrue(cr["sources"]["C6B"]["colour"]["kappa_requirement_applies"])
        self.assertLessEqual(cr["sources"]["C6B"]["colour"]["kappa_val"]["dkappa"], 0)
        self.assertEqual(W.run("selfcheck"), 1)     # a stopped check blocks the later steps
        self.assertEqual(W.run("tune"), 1)
        # SMOKE only: --smoke-record-colour records the same stops without stopping (notebook SMOKE models)
        self.assertEqual(W.run("check", "--smoke-record-colour"), 0)
        cr2 = W.read("check_report.json")
        self.assertEqual(cr2["status"], "ok")
        self.assertEqual(cr2["thresholds"]["colour_stops"], "recorded only (--smoke-record-colour)")
        for s in ("C6A", "C6B"):
            self.assertEqual(cr2["sources"][s]["colour"]["stops_recorded_only_smoke"],
                             cr["sources"][s]["stops"])
        self.assertTrue(cr2["sources"]["C6A"]["colour"]["wiring_stop"])
        # never in real mode (the out-dir is a preview outside the repository, so a regression writes nothing there)
        real_out = os.path.join(self.tmp, "real_preview")
        self.assertEqual(quiet(p03.main, ["check", "--smoke-record-colour", "--out-dir", real_out]), 1)
        self.assertFalse(os.path.exists(real_out))

    def test_floor_and_missing_arm(self):
        W = P03World(self.tmp, arms=("a",))
        self.assertEqual(W.run("check", "--min-acc", "99"), 0)
        cr = W.read("check_report.json")
        self.assertEqual(cr["sources"]["C6A"]["status"], "unusable")
        self.assertEqual(cr["sources"]["C6B"]["status"], "not_run")
        self.assertEqual(cr["usable_sources"], ["C5"])
        self.assertEqual(W.run("selfcheck"), 0)
        self.assertEqual(W.run("tune"), 0)
        tune = W.read("tune_report.json")
        self.assertEqual([c["id"] for c in tune["combinations"] if c["status"] == "ok"], ["P00"])
        fc = W.read("frozen_config.json")
        self.assertEqual(fc["combo"]["id"], "P00")
        self.assertFalse(fc["gates"]["passed"])
        self.assertFalse(fc["score_allowed"])
        self.assertEqual(W.run("apply"), 0)
        app = W.read("apply_report.json")
        self.assertFalse(app["score_allowed"])
        # P00's no-opening version is R10's procedure
        self.assertEqual(fc["no_opening"]["combo"]["sources"], ["C5"])
        # arm A alone (default floor): P00, P01, P04 fitted
        self.assertEqual(W.run("check"), 0)
        self.assertEqual(W.read("check_report.json")["usable_sources"], ["C5", "C6A"])
        self.assertEqual(W.run("selfcheck"), 0)
        self.assertEqual(W.run("tune"), 0)
        tune = W.read("tune_report.json")
        self.assertEqual([c["id"] for c in tune["combinations"] if c["status"] == "ok"], ["P00", "P01", "P04"])

    def test_structure_stops(self):
        W = P03World(self.tmp)
        dd = os.path.join(W.p03, "rank_a", "dumps")
        # inner keys in another row order
        path = os.path.join(dd, "inner_keys.npz")
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        perm = np.arange(len(arrays["label"]))[::-1]
        save_npz(path, **{k: v[perm] for k, v in arrays.items()})
        self.assertEqual(W.run("check"), 1)
        cr = W.read("check_report.json")
        self.assertFalse(cr["sources"]["C6A"]["inner_keys_equal_to_prereg02"])
        save_npz(path, **arrays)
        mk = os.path.join(dd, "_DONE_inner_keys.json")
        obj = json.load(open(mk, encoding="utf-8"))
        obj["sha256"] = pseudo.sha256_file(path)
        json.dump(obj, open(mk, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 0)
        # history does not reach T
        hist = os.path.join(W.p03, "rank_a", "ckpt", "history.csv")
        with open(hist, encoding="utf-8") as f:
            text = f.read()
        with open(hist, "w", encoding="utf-8") as f:
            f.write("\n".join(text.splitlines()[:-1]) + "\n")
        self.assertEqual(W.run("check"), 1)
        self.assertTrue(any("global_step" in s for s in W.read("check_report.json")["stops"]))
        with open(hist, "w", encoding="utf-8") as f:
            f.write(text)
        # validation NaN pattern differs from val_p02e9_d4's
        vpath = os.path.join(dd, "val_p03a_d4.npz")
        with np.load(vpath) as z:
            arrays = {k: z[k] for k in z.files}
        lg = arrays["logits"].copy()
        lg[0] = np.nan
        save_npz(vpath, **dict(arrays, logits=lg))
        mk = os.path.join(dd, "_DONE_val_p03a_d4.json")
        obj = json.load(open(mk, encoding="utf-8"))
        obj["sha256"] = pseudo.sha256_file(vpath)
        json.dump(obj, open(mk, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 1)
        # different weights in one dump
        save_npz(vpath, **dict(arrays, weights_sha256=np.array("other")))
        obj["sha256"] = pseudo.sha256_file(vpath)
        json.dump(obj, open(mk, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 1)
        self.assertTrue(any("different weights" in s or "weights_sha256" in s
                            for s in W.read("check_report.json")["stops"]))
        save_npz(vpath, **arrays)
        obj["sha256"] = pseudo.sha256_file(vpath)
        json.dump(obj, open(mk, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 0)
        # config.json: final.pth hash other than the dumps' weights, or training not complete
        cpath = os.path.join(W.p03, "rank_a", "config.json")
        cfg = json.load(open(cpath, encoding="utf-8"))
        for key, bad in (("final_sha256", "f" * 64), ("training_complete", False)):
            json.dump(dict(cfg, **{key: bad}), open(cpath, "w", encoding="utf-8"))
            self.assertEqual(W.run("check"), 1, key)
        json.dump(dict(cfg, recipe={"ARM": "B", "T": T_SMOKE}), open(cpath, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 1)       # arm B's config in arm A's folder
        json.dump(cfg, open(cpath, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 0)
        # an arm that stopped earlier in the same folder and was run again to completion (PREREG_03 1.3)
        rl = os.path.join(W.p03, "rank_a", "run_log.json")
        good_rl = open(rl, encoding="utf-8").read()
        json.dump({"runs": [{"run": 1, "status": "time_stop"}, {"run": 2, "status": "completed"}], "status": "completed"},
                  open(rl, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 1)
        self.assertTrue(any("stopped runs" in s for s in W.read("check_report.json")["stops"]))
        with open(rl, "w", encoding="utf-8") as f:
            f.write(good_rl)
        json.dump(dict(cfg, stopped={"status": "nan_stop"}), open(cpath, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 1)
        json.dump(cfg, open(cpath, "w", encoding="utf-8"))
        spath = os.path.join(W.p03, "rank_a", "STOPPED.json")
        json.dump({"status": "pass_check_failed"}, open(spath, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 1)
        os.remove(spath)
        self.assertEqual(W.run("check"), 0)
        # a read-only folder changed during the run
        rpath = os.path.join(W.p03, "rank_a", "readonly_after.json")
        json.dump({"runs": [{"same_as_before": True}, {"same_as_before": False}]}, open(rpath, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 1)

    def test_smoke_refusals_and_readonly_outputs(self):
        W = P03World(self.tmp)
        # smoke output inside the repository
        args = W.args("check")
        args[args.index("--out-dir") + 1] = os.path.join(REPO, "results", "prereg03", "rank")
        self.assertEqual(quiet(p03.main, args), 1)
        # smoke on real-sized or non-SMOKE prereg02 dumps
        with open(os.path.join(W.w.p02, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"smoke": False}, f)
        self.assertEqual(W.run("check"), 1)
        with open(os.path.join(W.w.p02, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"smoke": True}, f)
        # an arm that is not a SMOKE result
        cfg = os.path.join(W.p03, "rank_b", "config.json")
        json.dump({"smoke": False, "recipe": {"T": T_SMOKE}}, open(cfg, "w", encoding="utf-8"))
        self.assertEqual(W.run("check"), 1)
        # --min-acc only with --smoke; real mode only on the repository paths
        self.assertEqual(quiet(p03.main, ["check", "--min-acc", "10"]), 1)
        self.assertEqual(quiet(p03.main, ["check", "--p02-dir", W.w.p02]), 1)
        # real mode never writes into prereg02 / test02 / prereg01
        for ro in ("prereg02", "test02", "prereg01"):
            for cmd in ("check", "selfcheck", "tune", "apply"):
                self.assertEqual(quiet(p03.main, [cmd, "--out-dir", os.path.join(REPO, "results", ro, "rank")]), 1)
        with self.assertRaises(p03.P03Error):
            a = type("A", (), dict(smoke=True, min_acc=None, p02_dir=W.w.p02, p02_local_dir=W.w.out,
                                   p03_dir=W.p03, split=W.w.split_csv, fallback_r10=None, questions=None,
                                   out_dir=os.path.join(W.p03, "rank_a")))
            p03.resolve(a, "check")         # never into an arm folder

    # ---- single-colour records (PREREG_03 rev 3)
    def val_subquestions_by_hand(self, vk):
        out = []
        for _q, rows in vk["groups"]:
            for col in ("B", "W"):
                g = [i for i in rows if vk["parsed"][i] and vk["color"][i] == col]
                if len(g) >= 2:
                    out.append((col, g))
        return out

    def test_single_colour_records_end_to_end(self):
        W = P03World(self.tmp)
        self.assertEqual(W.run("check"), 0)
        cr = W.read("check_report.json")
        keys = pseudo.load_inner_keys(W.w.dumps)
        players = pseudo.load_pseudo(os.path.join(W.w.out, "pseudo_inner.json"))["players"]
        qs = p03.mono_questions(keys, players)
        self.assertGreater(len(qs), 0)
        self.assertEqual(cr["single_colour"]["inner_questions"]["counts"]["questions"], len(qs))
        vk = rf.load_val_keys(W.w.dumps)
        subs = self.val_subquestions_by_hand(vk)
        self.assertEqual(cr["single_colour"]["val_sub_questions"]["sub_questions"], len(subs))
        for x, s in (("a", "C6A"), ("b", "C6B")):
            col = cr["sources"][s]["colour"]
            rec = col["kappa_single_colour"]
            dd = os.path.join(W.p03, f"rank_{x}", "dumps")
            with np.load(os.path.join(dd, f"inner_p03{x}_d4.npz")) as z:
                lc = rf.log_softmax(z["logits"][:, 0, :].astype(np.float64))
            with np.load(os.path.join(dd, f"inner_p03{x}_s0_swap.npz")) as z:
                ls = rf.log_softmax(z["logits"].astype(np.float64))
            dk = np.array([kappa_by_hand(lc, q[4]) - kappa_by_hand(ls, q[4]) for q in qs])
            self.assertAlmostEqual(rec["inner"]["dkappa"], float(np.mean(dk)), places=10)
            cols = np.array([q[2] for q in qs])
            for c in ("B", "W"):
                self.assertAlmostEqual(rec["inner"]["by_colour"][c]["dkappa"], float(np.mean(dk[cols == c])),
                                       places=10)
            with np.load(os.path.join(dd, f"val_p03{x}_d4.npz")) as z:
                vz = np.where(vk["parsed"][:, None], z["logits"][:, 0, :], 0.0).astype(np.float64)
            with np.load(os.path.join(dd, f"val_p03{x}_s0_swap.npz")) as z:
                vs = np.where(vk["parsed"][:, None], z["logits"], 0.0).astype(np.float64)
            vlc, vls = rf.log_softmax(vz), rf.log_softmax(vs)
            dv = [kappa_by_hand(vlc, g) - kappa_by_hand(vls, g) for _c, g in subs]
            self.assertAlmostEqual(rec["val_sub_questions"]["dkappa"], float(np.mean(dv)), places=10)
            ref = rec["test_day_reference"]
            self.assertEqual(ref["dacc_points"], col["dacc_points"])
            self.assertTrue(ref["single_colour_condition"] and ref["val_sub_question_dkappa_positive"])
        self.assertEqual(W.run("selfcheck"), 0)
        self.assertEqual(W.run("tune"), 0)
        tr, fc = W.read("tune_report.json"), W.read("frozen_config.json")
        sc = tr["single_colour"]
        self.assertEqual(sc["status"], "ok")
        self.assertEqual(sc["questions_sha256"], cr["single_colour"]["inner_questions"]["questions_sha256"])
        self.assertEqual(list(sc["combinations"]), [c for c, _ in p03.COMBOS])
        # by hand: each question scored with the parameters of its player's fold (fold records of tune_report)
        logits = {}
        with np.load(os.path.join(W.w.dumps, "inner_p02e9_d4.npz")) as z:
            logits["p02e9_d4"] = z["logits"]
        for x in ("a", "b"):
            with np.load(os.path.join(W.p03, f"rank_{x}", "dumps", f"inner_p03{x}_d4.npz")) as z:
                logits[f"p03{x}_d4"] = z["logits"]
        with np.load(os.path.join(W.w.out, "opening_inner.npz")) as z:
            at = {(str(r), str(g)): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
            perm = [at[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])]
            o, has = np.asarray(z["o"], dtype=np.float64)[perm], np.asarray(z["has"], dtype=bool)[perm]
        cid = fc["combo"]["id"]
        for combo in sorted({"P00", cid}):
            rec = next(c for c in tr["combinations"] if c["id"] == combo)
            params = {f["fold"]: f["params"] for f in rec["folds"]}
            lp = p03.combo_logprobs(dict(p03.COMBOS)[combo], logits)
            score, y = [], []
            for _r, pi, _c, _k, rows in qs:
                par = params[players[pi]["fold"]]
                hh = [i for i in rows if has[i]]
                mop = o[hh].mean(0) if hh else np.zeros(10)
                t = math.exp(par["a"]) * (len(rows) / 12.0) ** par["gamma"]
                p = rf.softmax((t * (lp[rows].mean(0) + par["w"] * mop + np.array(par["b"])))[None])
                d = abs(int(rf.rule_b_np(p)[0]) - LEVELS.index(players[pi]["rank"]))
                score.append(1.0 if d == 0 else (math.exp(-1) if d == 1 else 0.0))
                y.append(LEVELS.index(players[pi]["rank"]))
            score, y = np.array(score), np.array(y)
            rb = float(np.mean([score[y == i].mean() for i in range(10) if (y == i).any()]))
            self.assertAlmostEqual(sc["combinations"][combo]["rank_balanced"], rb, places=12)
            mk = cols == "W"
            rbw = float(np.mean([score[mk & (y == i)].mean() for i in range(10) if (mk & (y == i)).any()]))
            self.assertAlmostEqual(sc["combinations"][combo]["by_colour"]["W"]["rank_balanced"], rbw, places=12)
        f1 = fc["single_colour"]
        self.assertEqual(f1["selected"]["id"], cid)
        self.assertEqual(f1["selected"]["rank_balanced"], sc["combinations"][cid]["rank_balanced"])
        self.assertEqual(f1["P00"]["rank_balanced"], sc["combinations"]["P00"]["rank_balanced"])
        want_ref = [s for s in dict(p03.COMBOS)[cid] if s != "C5"]
        self.assertEqual(sorted(f1["kappa_reference"]), sorted(want_ref))
        for s in want_ref:
            self.assertEqual(f1["kappa_reference"][s]["kappa_single_colour"],
                             cr["sources"][s]["colour"]["kappa_single_colour"])
        if cid != "P00":
            d = f1["selected_minus_P00"]
            self.assertAlmostEqual(d["rank_balanced"], f1["selected"]["rank_balanced"] - f1["P00"]["rank_balanced"],
                                   places=15)
        # the fold records' parameters must reproduce the OOF probabilities, else tune stops
        old = p03.theta_of
        p03.theta_of = lambda par, opening: old(par, opening) + np.r_[1e-9, np.zeros(len(par["beta"]) + 1 + opening)]
        try:
            self.assertEqual(W.run("tune"), 1)
        finally:
            p03.theta_of = old
        # tune refuses a check_report.json built from other single-colour questions
        crp = os.path.join(W.out, "check_report.json")
        cr["single_colour"]["inner_questions"]["questions_sha256"] = "0" * 64
        pseudo.write_json(crp, cr)
        self.assertEqual(W.run("tune"), 1)

    def test_single_colour_block_withholds_h3r(self):
        """PREREG_03 rev 3 11(c): a selected C6 source whose single-colour condition holds while its validation
        single-colour sub-question dkappa is not > 0 cannot be used on test day: score_allowed turns false in tune and
        apply (the scoring chance is kept); the selection, gates, parameters and validation probabilities are
        unchanged."""
        # every combination scores 1.0 in the synthetic world, so P00 is always selected: force P01 with passed
        # gates (the block is about score_allowed, not about the selection)
        orig_select, orig_gate = p03.select, p03.gate_record

        def forced_select(results):
            _chosen, rec = orig_select(results)
            return next(r for r in results if r["id"] == "P01"), dict(rec, selected="P01", note="test: forced")

        def forced_gate(sel_id, *a):
            return dict(orig_gate(sel_id, *a), passed=True, reason="test: forced")
        p03.select, p03.gate_record = forced_select, forced_gate
        try:
            self._single_colour_block()
        finally:
            p03.select, p03.gate_record = orig_select, orig_gate

    def _single_colour_block(self):
        W = P03World(self.tmp)
        for cmd in ("check", "selfcheck", "tune", "apply"):
            self.assertEqual(W.run(cmd), 0, cmd)
        fc0, app0 = W.read("frozen_config.json"), W.read("apply_report.json")
        with open(os.path.join(W.out, "rank_val_probs.csv"), "rb") as f:
            probs0 = f.read()
        cid = fc0["combo"]["id"]
        self.assertNotEqual(cid, "P00")
        self.assertTrue(fc0["gates"]["passed"])
        self.assertTrue(fc0["score_allowed"])
        self.assertTrue(app0["score_allowed"])
        self.assertEqual(fc0["score_blocked_single_colour"], [])
        self.assertEqual(app0["score_blocked_single_colour"], [])
        # check_report.json records: condition holds, validation sub-question dkappa <= 0 (every C6 source)
        crp = os.path.join(W.out, "check_report.json")
        cr = W.read("check_report.json")
        for s in ("C6A", "C6B"):
            ref = cr["sources"][s]["colour"]["kappa_single_colour"]["test_day_reference"]
            self.assertTrue(ref["single_colour_condition"])
            ref.update(val_sub_question_dkappa=-0.001, val_sub_question_dkappa_positive=False)
        pseudo.write_json(crp, cr)
        self.assertEqual(W.run("tune"), 0)
        fc1, tr1 = W.read("frozen_config.json"), W.read("tune_report.json")
        for k in ("combo", "params", "selection", "gates", "se_val", "oof", "no_opening", "fail_opening"):
            self.assertEqual(fc1[k], fc0[k], k)
        srcs = [s for s in dict(p03.COMBOS)[cid] if s != "C5"]
        self.assertFalse(fc1["score_allowed"])
        self.assertFalse(tr1["score_allowed"])
        self.assertEqual([r.split(":")[0] for r in fc1["score_blocked_single_colour"]], srcs)
        self.assertEqual(tr1["score_blocked_single_colour"], fc1["score_blocked_single_colour"])
        self.assertEqual(W.run("apply"), 0)
        app1 = W.read("apply_report.json")
        self.assertFalse(app1["score_allowed"])
        self.assertEqual(app1["score_blocked_single_colour"], fc1["score_blocked_single_colour"])
        self.assertTrue(set(app1["score_blocked_single_colour"]) <= set(app1["score_blocked_reasons"]))
        with open(os.path.join(W.out, "rank_val_probs.csv"), "rb") as f:
            self.assertEqual(f.read(), probs0)
        # apply recomputes the block from frozen_config.json's kappa reference (it does not copy score_allowed)
        fpath = os.path.join(W.out, "frozen_config.json")
        for s in srcs:
            fc1["single_colour"]["kappa_reference"][s]["kappa_single_colour"]["test_day_reference"][
                "val_sub_question_dkappa_positive"] = True
        pseudo.write_json(fpath, fc1)
        self.assertEqual(W.run("apply"), 0)
        self.assertTrue(W.read("apply_report.json")["score_allowed"])
        # condition false: the validation sub-question dkappa is recorded only, H3R runs as registered
        for s in ("C6A", "C6B"):
            cr["sources"][s]["colour"]["kappa_single_colour"]["test_day_reference"]["single_colour_condition"] = False
        pseudo.write_json(crp, cr)
        self.assertEqual(W.run("tune"), 0)
        self.assertTrue(W.read("frozen_config.json")["score_allowed"])
        self.assertEqual(W.read("frozen_config.json")["score_blocked_single_colour"], [])

    def test_single_colour_records_are_record_only(self):
        def chain(W, want_check=0):
            self.assertEqual(W.run("check"), want_check)
            out = {"check": W.read("check_report.json")}
            if want_check:
                return out
            for cmd in ("selfcheck", "tune", "apply"):
                self.assertEqual(W.run(cmd), 0, cmd)
            out.update(tune=W.read("tune_report.json"), frozen=W.read("frozen_config.json"))
            with open(os.path.join(W.out, "rank_val_probs.csv"), "rb") as f:
                out["probs"] = f.read()
            return out

        def both(W, want_check=0):
            on = chain(W, want_check)
            old = p03.MONO_ROUNDS
            p03.MONO_ROUNDS = ()          # no single-colour question at all
            try:
                off = chain(W, want_check)
            finally:
                p03.MONO_ROUNDS = old
            return on, off

        def strip_check(cr):
            src = json.loads(json.dumps(cr["sources"]))
            for s in src.values():
                (s.get("colour") or {}).pop("kappa_single_colour", None)
            return {"status": cr["status"], "stops": cr["stops"], "usable": cr["usable_sources"],
                    "combinations": cr["combinations"], "sources": src}

        # (in this world every C6 source has the single-colour condition with a positive validation sub-question
        # dkappa, so the 11(c) block is off and score_allowed must match too; the block itself:
        # test_single_colour_block_withholds_h3r)
        W = P03World(os.path.join(self.tmp, "ok"))
        on, off = both(W)
        self.assertGreater(on["check"]["single_colour"]["inner_questions"]["counts"]["questions"], 0)
        self.assertEqual(off["check"]["single_colour"]["inner_questions"]["counts"]["questions"], 0)
        self.assertEqual(strip_check(on["check"]), strip_check(off["check"]))
        for k in ("combinations", "selection", "gates", "score_allowed", "se_val", "per_level_diff_vs_P00", "refit",
                  "no_opening", "fail_opening", "usable_sources"):
            self.assertEqual(on["tune"][k], off["tune"][k], k)
        for k in ("combo", "params", "dumps", "selection", "gates", "score_allowed", "se_val", "oof", "no_opening",
                  "fail_opening", "P00_oof_rank_balanced", "per_level_diff_vs_P00"):
            self.assertEqual(on["frozen"][k], off["frozen"][k], k)
        self.assertEqual(on["probs"], off["probs"])
        self.assertEqual(on["tune"]["single_colour"]["status"], "ok")
        self.assertTrue(off["tune"]["single_colour"]["status"].startswith("no single-colour"))
        # the check-time colour stops are the registered mixed-colour ones, with or without the new records
        W2 = P03World(os.path.join(self.tmp, "bug"), modes={"b": "val_kappa_bug"})
        on, off = both(W2, want_check=1)
        self.assertEqual(strip_check(on["check"]), strip_check(off["check"]))
        self.assertEqual(on["check"]["sources"]["C6B"]["status"], "stop")
        self.assertFalse(on["check"]["sources"]["C6B"]["colour"]["kappa_single_colour"]["test_day_reference"]
                         ["val_sub_question_dkappa_positive"])

    def test_real_mode_apply_needs_committed_frozen_config(self):
        # the real out-dir has no committed frozen_config.json yet: apply refuses before reading anything.
        # Since 1218ba8 the real frozen_config.json is committed, so running apply here would redo the registered
        # prereg03 apply and rewrite results/prereg03/rank/apply_report.json: skip instead.
        fc = os.path.join(REPO, "results", "prereg03", "rank", "frozen_config.json")
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", fc], cwd=REPO, capture_output=True).returncode == 0
        if tracked:
            self.skipTest("the real prereg03 frozen_config.json is committed (1218ba8); the real apply is not re-run")
        self.assertEqual(quiet(p03.main, ["apply"]), 1)
        self.assertEqual(quiet(p03.main, ["tune"]), 1)


if __name__ == "__main__":
    unittest.main()
