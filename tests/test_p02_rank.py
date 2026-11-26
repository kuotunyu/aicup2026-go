"""Tests for the PREREG_02 competition-2 local tools (tools/p02_rank_*.py).
Run: python -m unittest discover -s tests   (or python -m pytest tests)

Everything runs on synthetic data written to a temporary directory outside the
repository: synthetic players (taken from a synthetic players_split.csv so that
c10data.players_for(..., "inner") holds them), synthetic SGFs, synthetic dumps in
the notebook's interface format. No real data, no label file, no GPU.
"""

import csv
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
import c10data  # noqa: E402
import decide  # noqa: E402
import p02_moves  # noqa: E402
import p02_rank_fuse as fuse  # noqa: E402
import p02_rank_opening as opening  # noqa: E402
import p02_rank_pseudo as pseudo  # noqa: E402

LEVELS = c10data.LEVELS
E1 = math.exp(-1)
POINTS = [p02_moves.from_xy(x, y) for x in range(19) for y in range(19)]


# ================================================================ synthetic world
def write_split(path, per_level=200):
    """All-train synthetic split (one bin) plus one val player per level; returns inner players by level."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "rank", "n_games", "bin", "split"])
        for lv in LEVELS:
            for i in range(per_level):
                w.writerow([f"{lv}_p{i:04d}", lv, 30, "25-104", "train"])
            w.writerow([f"{lv}_v0", lv, 30, "25-104", "val"])
    split = c10data.read_split(path)
    inner = c10data.players_for(split, "inner")
    by_level = defaultdict(list)
    for p in sorted(inner):
        by_level[split[p]["rank"]].append(p)
    return inner, by_level


def synth_sgf(rng, level, n_moves=14):
    """Opening with a level-dependent first move (60 %), later moves from a small pool."""
    star = ["dd", "pd", "dp", "pp", "jj", "dj", "jd", "pj", "jp", "cc"]
    pool = POINTS[::12]
    first = star[level] if rng.random() < 0.6 else rng.choice(pool)
    pts = [first]
    while len(pts) < n_moves:
        p = rng.choice(pool)
        if p not in pts:
            pts.append(p)
    return "(;SZ[19]" + "".join(";%s[%s]" % ("BW"[i % 2], p) for i, p in enumerate(pts)) + ")"


def synth_logits(rng, y, S, strength):
    k = np.arange(10)
    base = -strength * np.abs(k[None, :] - np.asarray(y)[:, None]).astype(np.float64)
    z = base[:, None, :] + rng.normal(0, 1.0, size=(len(y), S, 10))
    return z.astype(np.float32)


def save_npz(path, **arrays):
    np.savez(path, **arrays)


class World:
    """Synthetic inner games, validation questions, dumps and prereg01 probabilities."""

    def __init__(self, root, seed=0, players_per_level=6, big=3, n_questions=14):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        prng = random.Random(seed)
        self.split_csv = os.path.join(root, "players_split.csv")
        self.inner_set, by_level = write_split(self.split_csv)
        self.p01 = os.path.join(root, "p01", "rank")
        self.p02 = os.path.join(root, "p02", "rank")
        self.out = os.path.join(root, "local", "rank")
        self.dumps = os.path.join(self.p02, "dumps")
        for d in (self.p01, self.dumps, self.out):
            os.makedirs(d, exist_ok=True)
        self.set_notebook_smoke(True)   # the notebook's config.json flag: these are SMOKE-like dumps

        # inner games: `big` players per level with 25 games, the rest 5..19, one with 3
        self.games = []  # (rank, game_id, player_id, colour, sgf)
        for li, lv in enumerate(LEVELS):
            for j, pid in enumerate(by_level[lv][:players_per_level]):
                if j < big:
                    n = 25
                elif j < players_per_level - 1:
                    n = prng.randint(5, 19)
                else:
                    n = 3
                for _ in range(n):
                    gid = "g_%07d" % (len(self.games) + 1)
                    self.games.append((lv, gid, pid, prng.choice("BW"), synth_sgf(prng, li)))
        self.label = np.array([LEVELS.index(g[0]) for g in self.games])
        N = len(self.games)
        save_npz(os.path.join(self.dumps, "inner_keys.npz"),
                 rank=np.array([g[0] for g in self.games]), game_id=np.array([g[1] for g in self.games]),
                 player_id=np.array([g[2] for g in self.games]), color=np.array([g[3] for g in self.games]),
                 label=self.label.astype(np.int8))
        strengths = {"p01e1": 0.2, "p01e2": 0.2, "p01e3": 0.3, "p01e4": 0.3, "p01e5_d4": 0.4, "p02e9_d4": 0.6}
        self.inner_logits = {}
        for name, st in strengths.items():
            S = fuse.DUMP_S[name]
            z = synth_logits(self.rng, self.label, S, st)
            self.inner_logits[name] = z
            save_npz(os.path.join(self.dumps, f"inner_{name}.npz"), logits=z,
                     weights_sha256=np.array("w_" + name), syms=np.array(S))

        # validation questions (players outside inner); question 3 has no parsed game
        self.questions = []  # (qid, [(colour, sgf)], true level)
        for q in range(n_questions):
            lv = q % 10
            k = prng.randint(5, 20)
            self.questions.append(("val_q_%04d" % (q + 1), [(prng.choice("BW"), synth_sgf(prng, lv))
                                                             for _ in range(k)], lv))
        qid, slot, col, parsed, yv = [], [], [], [], []
        for qi, (q, games, lv) in enumerate(self.questions):
            for s, (c, _) in enumerate(games, 1):
                qid.append(q)
                slot.append(s)
                col.append(c)
                parsed.append(not (qi == 3 or (qi == 5 and s == 2)))
                yv.append(lv)
        self.val_parsed = np.array(parsed)
        save_npz(os.path.join(self.dumps, "val_keys.npz"), question_id=np.array(qid),
                 slot=np.array(slot, dtype=np.int16), color=np.array(col), parsed=self.val_parsed)
        self.val_logits = {}
        for name, st in strengths.items():
            z = synth_logits(self.rng, yv, fuse.DUMP_S[name], st)
            z[~self.val_parsed] = np.nan
            self.val_logits[name] = z
            save_npz(os.path.join(self.dumps, f"val_{name}.npz"), logits=z,
                     weights_sha256=np.array("w_" + name), syms=np.array(fuse.DUMP_S[name]))
        self.val_qid = qid

        # prereg01 probabilities as the notebook computed them (float32 softmax, float32 mean)
        self.p01_csv = os.path.join(self.p01, "rank_val_probs.csv")
        with open(self.p01_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(fuse.PROB_COLUMNS)
            row = 0
            for q, games, _ in self.questions:
                z = self.val_logits["p01e5_d4"][row:row + len(games), 0, :]
                ok = self.val_parsed[row:row + len(games)]
                row += len(games)
                if not ok.any():
                    w.writerow([q, len(games), 0] + [""] * 10)
                    continue
                zz = z[ok].astype(np.float32)
                e = np.exp(zz - zz.max(1, keepdims=True))
                p = (e / e.sum(1, keepdims=True)).astype(np.float32).mean(0)
                w.writerow([q, len(games), int(ok.sum())] + [repr(float(x)) for x in p])

        # question file in the official layout (no labels)
        self.q_csv = os.path.join(root, "rank_val_questions.csv")
        with open(self.q_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, 21)]
                       + [f"color_{i}" for i in range(1, 21)])
            for q, games, _ in self.questions:
                sg = [g[1] for g in games] + [""] * (20 - len(games))
                cl = [g[0] for g in games] + [""] * (20 - len(games))
                w.writerow([q, len(games)] + sg + cl)

        # fit games for the opening counts
        self.fit = []
        for li, lv in enumerate(LEVELS):
            for _ in range(150):
                self.fit.append((prng.choice("BW"), synth_sgf(prng, li), li))

    # ---- notebook records
    def set_notebook_smoke(self, flag):
        with open(os.path.join(self.p02, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"smoke": flag}, f)

    def write_repro_check(self, **kw):
        with open(os.path.join(self.p02, "repro_check.json"), "w", encoding="utf-8") as f:
            json.dump(dict({"pass": True}, **kw), f)

    def write_gpu_records(self, p01_gpu, dump_gpu="absent"):
        """prereg01 env.json (last run's GPU) and, unless 'absent', the val_p01e5_d4 completion marker."""
        smi = ["%s, 97887 MiB, 580.82.07" % p01_gpu] if p01_gpu else None
        with open(os.path.join(self.p01, "env.json"), "w", encoding="utf-8") as f:
            json.dump({"runs": [{"gpu": {"nvidia_smi": smi, "device_count": 1} if smi else None}]}, f)
        mk = os.path.join(self.dumps, "_DONE_val_p01e5_d4.json")
        if dump_gpu == "absent":
            if os.path.exists(mk):
                os.remove(mk)
        else:
            with open(mk, "w", encoding="utf-8") as f:
                json.dump({"rows": len(self.val_qid), "gpu_name": dump_gpu,
                           "sha256": pseudo.sha256_file(os.path.join(self.dumps, "val_p01e5_d4.npz"))}, f)

    def bump_p01(self, delta):
        """Move the largest probability of the first prereg01 question by delta (rule B unchanged)."""
        with open(self.p01_csv, encoding="utf-8") as f:
            text = f.read().splitlines()
        cells = text[1].split(",")
        p = [float(x) for x in cells[3:]]
        top = int(np.argmax(p))
        p[top] += delta
        p[(top + 5) % 10] -= delta
        text[1] = ",".join(cells[:3] + [repr(x) for x in p])
        with open(self.p01_csv, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(text) + "\n")

    # ---- local steps
    def write_pseudo(self, **kw):
        keys = pseudo.load_inner_keys(self.dumps)
        rows = {(g[0], g[1]): (g[2], g[3]) for g in self.games}
        obj = pseudo.build(keys, rows, self.inner_set, smoke=True, **kw)
        pseudo.write_json(os.path.join(self.out, pseudo.OUT_NAME), obj)
        return obj

    def write_opening(self):
        inner = [(g[0], g[1], g[2], g[3], opening.game_string(g[4])) for g in self.games]
        val = [(q, s, c, opening.game_string(sgf)) for q, games, _ in self.questions
               for s, (c, sgf) in enumerate(games, 1)]
        model = opening.OpeningModel([(g[3], g[4]) for g in inner] + [(v[2], v[3]) for v in val])
        model.count([(c, opening.game_string(s), li) for c, s, li in self.fit])
        si = model.score([(g[3], g[4]) for g in inner])
        sv = model.score([(v[2], v[3]) for v in val])
        return opening.write_outputs(self.out, inner, val, model, si, sv, {}, {"synthetic": True}, 0.0)

    def args(self, cmd, *extra):
        return [cmd, "--p01-dir", self.p01, "--p02-dir", self.p02, "--out-dir", self.out,
                "--split", self.split_csv, "--questions", self.q_csv, "--smoke"] + list(extra)


class TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_p02_rank_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ================================================================ source hygiene
class SourceHygiene(unittest.TestCase):
    def test_no_label_file_name_in_p02_tools(self):
        word = "answer" + "s"
        files = [f for f in os.listdir(TOOLS) if f.startswith("p02_") and f.endswith(".py")]
        self.assertTrue({"p02_moves.py", "p02_rank_pseudo.py", "p02_rank_opening.py",
                         "p02_rank_fuse.py"} <= set(files))
        for f in files:
            with open(os.path.join(TOOLS, f), encoding="utf-8") as fh:
                self.assertNotIn(word, fh.read().lower(), f)

    def test_rank_tools_do_not_import_player_tools(self):
        for f in ("p02_rank_pseudo.py", "p02_rank_opening.py", "p02_rank_fuse.py", "p02_moves.py"):
            with open(os.path.join(TOOLS, f), encoding="utf-8") as fh:
                src = fh.read()
            self.assertIsNone(re.search(r"^\s*(from|import)\s+p02_player", src, re.M), f)
            self.assertNotIn("prereg02/player", src, f)   # no player outputs
            self.assertNotIn('"player")', src, f)          # os.path.join(..., "player")

    def test_label_paths_refused(self):
        with self.assertRaises(pseudo.P02Error):
            pseudo.refuse_label_file(os.path.join("data", "local", "rank_val_" + "answer" + "s.csv"))


# ================================================================ rule B and scoring
class RuleB(unittest.TestCase):
    def test_equals_decide_rule_b(self):
        rng = np.random.default_rng(0)
        P = rng.dirichlet(np.ones(10) * 0.7, size=5000)
        ties = [np.full(10, 0.1), np.array([0.3, 0.3] + [0.05] * 8), np.array([0.0] * 9 + [1.0]),
                np.array([1.0] + [0.0] * 9), np.array([0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0])]
        P = np.vstack([P, np.array(ties)])
        got = fuse.rule_b_np(P)
        want = [LEVELS.index(decide.rule_b([float(x) for x in p])) for p in P]
        self.assertEqual(got.tolist(), want)

    def test_official_score(self):
        self.assertEqual(fuse.official_score([3, 4, 5, 9], [3, 3, 3, 0]).tolist(), [1.0, E1, 0.0, 0.0])


# ================================================================ pseudo-questions
class Pseudo(TmpDir):
    def test_bank_folds_rounds(self):
        w = World(self.tmp)
        obj = w.write_pseudo()
        players = obj["players"]
        n_by = defaultdict(int)
        for g in w.games:
            n_by[g[2]] += 1
        self.assertEqual({p["player_id"] for p in players}, {p for p, n in n_by.items() if n >= 5})
        for p in players:
            self.assertEqual(len(p["bank"]), min(20, p["n_games"]))
            self.assertEqual(p["fit"], p["n_games"] >= 20)
        # player order
        self.assertEqual([(p["rank"], p["player_id"]) for p in players],
                         sorted([(p["rank"], p["player_id"]) for p in players],
                                key=lambda t: (LEVELS.index(t[0]), t[1])))
        # folds balanced within every level
        by_level = defaultdict(lambda: [0] * 5)
        for p in players:
            by_level[p["rank"]][p["fold"]] += 1
        for counts in by_level.values():
            self.assertLessEqual(max(counts) - min(counts), 1)
        loaded = pseudo.load_pseudo(os.path.join(w.out, pseudo.OUT_NAME))
        # fit questions: rounds 0..49, only fit players, bank of exactly 20 games
        fit_players = {i for i, p in enumerate(players) if p["fit"]}
        self.assertEqual({r for r, _, _ in loaded["fit"]}, set(range(50)))
        self.assertEqual({i for _, i, _ in loaded["fit"]}, fit_players)
        for _, i, pos in loaded["fit"]:
            self.assertEqual(len(players[i]["bank"]), 20)
            self.assertTrue(5 <= len(pos) <= 20 and len(set(pos)) == len(pos))
        # eval: rounds 1000..1199, every bank player once per round
        per_round = defaultdict(list)
        for r, i, pos in loaded["eval"]:
            per_round[r].append(i)
            self.assertEqual(len(pos), min(len(pos), len(players[i]["bank"])))
            self.assertGreaterEqual(len(pos), min(5, len(players[i]["bank"])))
        self.assertEqual(sorted(per_round), list(range(1000, 1200)))
        for r, idx in per_round.items():
            self.assertEqual(idx, list(range(len(players))))

    def test_deterministic_and_salt_sensitive(self):
        w = World(self.tmp)
        a = w.write_pseudo()
        b = w.write_pseudo()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        old = pseudo.SALT
        try:
            pseudo.SALT = "c10-prereg02-other"
            c = w.write_pseudo()
        finally:
            pseudo.SALT = old
        self.assertNotEqual([p["bank"] for p in a["players"]], [p["bank"] for p in c["players"]])
        self.assertNotEqual(a["questions"]["eval"]["rounds"][0]["draws"],
                            c["questions"]["eval"]["rounds"][0]["draws"])
        self.assertNotEqual([p["fold"] for p in a["players"]], [p["fold"] for p in c["players"]])

    def test_draw_is_rng_sample_of_bank(self):
        sizes = [20, 7, 5, 20, 13]
        banks = [["g%d_%d" % (i, j) for j in range(n)] for i, n in enumerate(sizes)]
        for r in (0, 49, 1000, 1199):
            rng = random.Random(int(pseudo.sha_hex(f"c10-prereg02|pq|{r}")[:16], 16))
            want = [rng.sample(b, min(rng.randint(5, 20), len(b))) for b in banks]
            got = [[banks[i][k] for k in pos] for i, pos in enumerate(pseudo.draw_round(r, sizes))]
            self.assertEqual(got, want)

    def test_tampering_detected(self):
        w = World(self.tmp)
        w.write_pseudo()
        path = os.path.join(w.out, pseudo.OUT_NAME)
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        d = obj["questions"]["eval"]["rounds"][5]["draws"]
        obj["questions"]["eval"]["rounds"][5]["draws"] = ("1" if d[0] != "1" else "2") + d[1:]
        pseudo.write_json(path, obj)
        with self.assertRaises(pseudo.P02Error):
            pseudo.load_pseudo(path)

    def test_non_inner_players_refused(self):
        w = World(self.tmp)
        keys = pseudo.load_inner_keys(w.dumps)
        rows = {(g[0], g[1]): (g[2], g[3]) for g in w.games}
        some = next(iter(set(keys["player_id"])))
        with self.assertRaises(pseudo.P02Error):
            pseudo.build(keys, rows, w.inner_set - {some}, smoke=True)
        rows.pop((w.games[0][0], w.games[0][1]))
        with self.assertRaises(pseudo.P02Error):
            pseudo.build(keys, rows, w.inner_set, smoke=True)
        with self.assertRaises(pseudo.P02Error):  # real mode needs all 73,556 inner games
            pseudo.build(keys, {(g[0], g[1]): (g[2], g[3]) for g in w.games}, w.inner_set)


# ================================================================ opening n-gram
def naive_opening(fit, scored, M=12, alpha=20.0):
    """Straight transcription of results/prereg02/explore/opening_rank_ngram.py (full tables)."""
    ctx, cnt = defaultdict(lambda: [0] * 10), defaultdict(lambda: [0] * 10)
    for c, seq, lv in fit:
        for L in range(len(seq)):
            ctx[(c, seq[:L])][lv] += 1
            cnt[(c, seq[:L], seq[L])][lv] += 1
    out = []
    for c, seq in scored:
        ll = [0.0] * 10
        for L in range(len(seq)):
            cx = ctx.get((c, seq[:L]))
            if cx is None:
                break
            cm = cnt.get((c, seq[:L], seq[L]))
            tot_cx, tot_cm = sum(cx), (sum(cm) if cm else 0)
            pg = (tot_cm + 1.0) / (tot_cx + 361.0)
            for i in range(10):
                ll[i] += math.log(((cm[i] if cm else 0) + alpha * pg) / (cx[i] + alpha))
        out.append(ll)
    return np.array(out)


class Opening(TmpDir):
    def test_matches_naive_transcription(self):
        prng = random.Random(1)
        fit = [(prng.choice("BW"), synth_sgf(prng, li, prng.randint(3, 16)), li)
               for li in range(10) for _ in range(60)]
        scored_sgf = [(prng.choice("BW"), synth_sgf(prng, prng.randrange(10), prng.randint(1, 16)))
                      for _ in range(300)] + [("B", "(;SZ[19])"), ("W", "(;SZ[19];B[aa];W[ab])")]

        def tup(s):
            return tuple(p02_moves.to_xy(p) for _, p in p02_moves.canon(p02_moves.parse_moves(s, 12), 12))
        want = naive_opening([(c, tup(s), li) for c, s, li in fit], [(c, tup(s)) for c, s in scored_sgf])
        model = opening.OpeningModel([(c, opening.game_string(s)) for c, s in scored_sgf])
        model.count([(c, opening.game_string(s), li) for c, s, li in fit])
        got, nt = model.score([(c, opening.game_string(s)) for c, s in scored_sgf])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)
        self.assertEqual(nt[-2], 0)          # no move -> no term
        self.assertTrue((nt[:-2] > 0).all())  # the empty prefix is always seen in fit

    def test_outputs(self):
        w = World(self.tmp)
        rep = w.write_opening()
        oi = opening.load_opening(os.path.join(w.out, "opening_inner.npz"), "inner")
        ov = opening.load_opening(os.path.join(w.out, "opening_val.npz"), "val")
        self.assertEqual(oi["meta"]["build_id"], ov["meta"]["build_id"])
        self.assertEqual(len(oi["o"]), len(w.games))
        self.assertEqual(len(ov["o"]), sum(len(g) for _, g, _ in w.questions))
        has = np.asarray(oi["has"])
        o = np.asarray(oi["o"])
        np.testing.assert_allclose(np.exp(o[has]).sum(1), 1.0, atol=1e-12)
        self.assertTrue(rep["inner"]["with_score"] > 0)


# ================================================================ fusion numerics
class Numerics(unittest.TestCase):
    def test_basis(self):
        B = fuse.basis()
        self.assertEqual(B.shape, (10, 9))
        np.testing.assert_allclose(B.T @ B, np.eye(9), atol=1e-12)
        np.testing.assert_allclose(B.sum(0), 0.0, atol=1e-12)

    def test_gradient(self):
        rng = np.random.default_rng(1)
        Q = 300
        m = fuse.log_softmax(rng.normal(size=(Q, 10)))
        mop = fuse.log_softmax(rng.normal(size=(Q, 10)))
        n = rng.integers(5, 21, size=Q).astype(float)
        y = rng.integers(0, 10, size=Q)
        wq = fuse.level_weights(y)
        B = fuse.basis()
        for op in (False, True):
            th = rng.normal(scale=0.3, size=fuse.n_params(op))
            f0, g = fuse.objective(th, m, mop, n, y, wq, op, B)
            num = np.zeros_like(th)
            for i in range(len(th)):
                e = np.zeros_like(th)
                e[i] = 1e-6
                num[i] = (fuse.objective(th + e, m, mop, n, y, wq, op, B)[0]
                          - fuse.objective(th - e, m, mop, n, y, wq, op, B)[0]) / 2e-6
            np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-7)

    def test_level_weights_sum_to_one(self):
        y = np.array([0, 0, 1, 2, 2, 2, 9])
        w = fuse.level_weights(y)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertAlmostEqual(w[y == 2].sum(), 0.25)

    def test_r00_is_softmax_mean(self):
        rng = np.random.default_rng(2)
        z = rng.normal(scale=3, size=(200, 8, 10)).astype(np.float32)
        groups = [list(rng.choice(200, size=rng.integers(1, 21), replace=False)) for _ in range(50)]
        lp = fuse.source_logprobs("C1", {"p01e5_d4": z})
        S = fuse.pool_matrix(groups, 200)
        X = fuse.QSet([len(g) for g in groups], mop=np.zeros((50, 10)))
        X.m[("C1", "prob")] = fuse.pooled(S, lp, "prob")
        P = fuse.predict(fuse.uncalibrated_theta(), X, "C1", "prob", False, fuse.basis())
        want = np.array([fuse.softmax(z[g, 0, :].astype(np.float64)).mean(0) for g in groups])
        np.testing.assert_allclose(P, want, atol=1e-12)

    def test_source_definitions(self):
        rng = np.random.default_rng(3)
        d = {n: rng.normal(size=(30, fuse.DUMP_S[n], 10)).astype(np.float32) for n in fuse.DUMP_S}
        ls = lambda x: fuse.log_softmax(x.astype(np.float64))  # noqa: E731
        np.testing.assert_allclose(fuse.source_logprobs("C1", d), ls(d["p01e5_d4"][:, 0]))
        np.testing.assert_allclose(fuse.source_logprobs("C2", d),
                                   ls(np.mean([ls(d["p01e5_d4"][:, s]) for s in range(8)], 0)))
        np.testing.assert_allclose(fuse.source_logprobs("C3", d),
                                   ls(np.mean([ls(d["p01e3"][:, 0]), ls(d["p01e4"][:, 0]),
                                               ls(d["p01e5_d4"][:, 0])], 0)))
        np.testing.assert_allclose(fuse.source_logprobs("C4", d), ls(d["p02e9_d4"][:, 0]))
        np.testing.assert_allclose(fuse.source_logprobs("C5", d),
                                   ls(np.mean([ls(d["p02e9_d4"][:, s]) for s in range(8)], 0)))

    def test_combination_list(self):
        ids = [c["id"] for c in fuse.COMBOS]
        self.assertEqual(ids, ["R%02d" % i for i in range(21)])
        self.assertEqual((fuse.COMBOS[0]["source"], fuse.COMBOS[0]["pool"], fuse.COMBOS[0]["calibrated"]),
                         ("C1", "prob", False))
        want = [(s, p) for s in ("C1", "C4", "C3", "C2", "C5") for p in ("prob", "log")]
        self.assertEqual([(c["source"], c["pool"]) for c in fuse.COMBOS[1:11]], want)
        self.assertEqual([(c["source"], c["pool"]) for c in fuse.COMBOS[11:]], want)
        self.assertTrue(all(not c["opening"] for c in fuse.COMBOS[:11]))
        self.assertTrue(all(c["opening"] for c in fuse.COMBOS[11:]))
        p02 = [c["id"] for c in fuse.COMBOS if c["source"] in fuse.P02_SOURCES]
        self.assertEqual(p02, ["R03", "R04", "R09", "R10", "R13", "R14", "R19", "R20"])

    def test_recovers_temperature_and_opening_weight(self):
        rng = np.random.default_rng(4)
        Q = 30000
        m = fuse.log_softmax(rng.normal(scale=1.0, size=(Q, 10)))
        mop = fuse.log_softmax(rng.normal(scale=1.0, size=(Q, 10)))
        n = rng.integers(5, 21, size=Q).astype(float)
        B = fuse.basis()
        true = {"a": 0.6, "gamma": 0.5, "w": 0.8}
        b_true = np.linspace(-0.2, 0.2, 10)
        b_true -= b_true.mean()
        beta_true = B.T @ b_true
        for op in (False, True):
            th = np.concatenate([[true["a"], true["gamma"]] + ([true["w"]] if op else []), beta_true])
            z, _ = fuse.question_logits(th, m, mop, n, op, B)
            p = fuse.softmax(z)
            y = np.array([rng.choice(10, p=row) for row in p])
            X = fuse.QSet(n, y, np.zeros(Q, int), np.zeros(Q, int), mop)
            X.m[("C1", "log")] = m
            theta, rec = fuse.fit(X, "C1", "log", op, B)
            got = rec["params"]
            self.assertAlmostEqual(got["a"], true["a"], delta=0.06)
            self.assertAlmostEqual(got["gamma"], true["gamma"], delta=0.08)
            if op:
                self.assertAlmostEqual(got["w"], true["w"], delta=0.08)
            self.assertAlmostEqual(sum(got["b"]), 0.0, places=10)
            self.assertEqual(len(rec["starts"]), 2)


# ================================================================ end to end (synthetic, --smoke)
class EndToEnd(TmpDir):
    def run_all(self, w):
        w.write_pseudo()
        w.write_opening()
        self.assertEqual(fuse.main(w.args("repro")), 0)
        self.assertEqual(fuse.main(w.args("tune")), 0)
        self.assertEqual(fuse.main(w.args("apply")), 0)

    def test_pipeline(self):
        w = World(self.tmp)
        self.run_all(w)
        rep = json.load(open(os.path.join(w.out, "repro_local.json"), encoding="utf-8"))
        self.assertTrue(rep["passed"])
        self.assertLess(rep["max_abs_diff"], 1e-6)
        self.assertEqual(rep["rule_b_differences"], 0)
        tune = json.load(open(os.path.join(w.out, "tune_report.json"), encoding="utf-8"))
        self.assertEqual([c["id"] for c in tune["combinations"]], ["R%02d" % i for i in range(21)])
        self.assertTrue(all(c["status"] == "ok" for c in tune["combinations"]))
        sel = tune["selection"]
        ok = {c["id"]: c["oof"]["rank_balanced"] for c in tune["combinations"]}
        self.assertEqual(sel["best_rank_balanced"], max(ok.values()))
        first = next(i for i in ok if max(ok.values()) - ok[i] <= fuse.SELECT_TOL + fuse.FLOAT_EPS)
        self.assertEqual(sel["selected"], first)
        self.assertEqual(tune["gate"]["passed"], ok[first] - ok["R00"] >= fuse.GATE - fuse.FLOAT_EPS)
        # apply repeats the tune gate and says whether scoring is allowed
        app = json.load(open(os.path.join(w.out, "apply_report.json"), encoding="utf-8"))
        self.assertEqual(app["gate"], tune["gate"])
        self.assertIs(app["score_allowed"], tune["gate"]["passed"] is True)
        self.assertEqual(app["score_blocked_reasons"] == [], tune["gate"]["passed"] is True)
        for c in tune["combinations"]:
            self.assertEqual(set(c["oof"]["per_n_bin"]), {"5-7", "8-12", "13-20"})
            self.assertIn("se", c["oof"])
        # rank_val_probs.csv passes decide.py; the unparsed question is blank
        recs = decide.read_probs(os.path.join(w.out, "rank_val_probs.csv"))
        self.assertEqual([r["question_id"] for r in recs], [q for q, _, _ in w.questions])
        self.assertTrue(recs[3]["missing"] and recs[3]["probs"] is None)
        self.assertEqual(recs[5]["n_parsed"], recs[5]["n_games"] - 1)
        dec = os.path.join(self.tmp, "decide")
        s = decide.run(os.path.join(w.out, "rank_val_probs.csv"), dec)
        self.assertEqual(s["n_missing"], 1)

    def test_apply_blocks_scoring_when_the_gate_failed(self):
        w = World(self.tmp)
        self.run_all(w)
        fpath = os.path.join(w.out, "frozen_config.json")
        for gate in ({"threshold": fuse.GATE, "selected_minus_R00": 0.0, "passed": False}, None):
            fc = json.load(open(fpath, encoding="utf-8"))
            if gate is None:
                del fc["gate"]
            else:
                fc["gate"] = gate
            pseudo.write_json(fpath, fc)
            self.assertEqual(fuse.main(w.args("apply")), 0)
            app = json.load(open(os.path.join(w.out, "apply_report.json"), encoding="utf-8"))
            self.assertIs(app["score_allowed"], False)
            self.assertEqual(len(app["score_blocked_reasons"]), 1)
            self.assertEqual(app["gate"], gate)

    def test_apply_formula_and_r00_reproduces_prereg01(self):
        w = World(self.tmp, seed=5)
        self.run_all(w)
        fpath = os.path.join(w.out, "frozen_config.json")
        fc = json.load(open(fpath, encoding="utf-8"))
        # independent recomputation of the selected combination for question 0
        c, par = fc["combo"], fc["params"]
        n0 = len(w.questions[0][1])
        lp = fuse.source_logprobs(c["source"], {k: v[:n0] for k, v in w.val_logits.items()})
        m = (np.log(np.exp(lp).mean(0)) if c["pool"] == "prob" else lp.mean(0))
        if c["opening"]:
            ov = opening.load_opening(os.path.join(w.out, "opening_val.npz"), "val")
            has = np.asarray(ov["has"])[:n0]
            mop = np.asarray(ov["o"])[:n0][has].mean(0)
        else:
            mop = 0.0
        u = m + par["w"] * mop + np.array(par["b"])
        want = fuse.softmax((math.exp(par["a"]) * (n0 / 12) ** par["gamma"] * u)[None])[0]
        got = decide.read_probs(os.path.join(w.out, "rank_val_probs.csv"))[0]["probs"]
        np.testing.assert_allclose(got, want, atol=1e-12)
        # R00 through apply reproduces the prereg01 file
        fc["combo"] = dict(fuse.COMBOS[0])
        fc["dumps"] = {"p01e5_d4": {"weights_sha256": "w_p01e5_d4", "syms": 8}}
        fc["params"] = fuse.params_dict(fuse.uncalibrated_theta(), False, fuse.basis())
        pseudo.write_json(fpath, fc)
        self.assertEqual(fuse.main(w.args("apply")), 0)
        ours = decide.read_probs(os.path.join(w.out, "rank_val_probs.csv"))
        ref = decide.read_probs(w.p01_csv)
        for a, b in zip(ours, ref):
            self.assertEqual((a["question_id"], a["n_games"], a["n_parsed"]),
                             (b["question_id"], b["n_games"], b["n_parsed"]))
            if b["probs"] is not None:
                np.testing.assert_allclose(a["probs"], b["probs"], atol=1e-6)

    def test_refusals(self):
        w = World(self.tmp)
        w.write_pseudo()
        w.write_opening()
        self.assertEqual(fuse.main(w.args("tune")), 1)          # repro has not run
        # a failing repro blocks tune
        with open(w.p01_csv, encoding="utf-8") as f:
            text = f.read().splitlines()
        cells = text[1].split(",")
        p = [float(x) for x in cells[3:]]
        p[0], p[9] = p[9], p[0]
        text[1] = ",".join(cells[:3] + [repr(x) for x in p])
        with open(w.p01_csv, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(text) + "\n")
        self.assertEqual(fuse.main(w.args("repro")), 1)
        self.assertEqual(fuse.main(w.args("tune")), 1)
        # apply without --smoke: the frozen config must be committed inside the repository
        args = [a for a in w.args("apply") if a != "--smoke"]
        self.assertEqual(fuse.main(args), 1)
        # --smoke never writes inside the repository; nothing ever writes into results/prereg01
        with self.assertRaises(pseudo.P02Error):
            pseudo.check_out_dir(os.path.join(REPO, "results", "prereg02_smoke_x"), smoke=True)
        with self.assertRaises(pseudo.P02Error):
            pseudo.check_out_dir(os.path.join(REPO, "results", "prereg01", "rank"), smoke=False)
        with self.assertRaises(pseudo.P02Error):
            pseudo.check_out_dir(os.path.join(w.p01, "x"), smoke=True, extra_readonly=(w.p01,))
        pseudo.check_out_dir(os.path.join(REPO, "results", "prereg02", "rank"), smoke=False)

    def repro(self, w, args=None):
        rc = fuse.main(args or w.args("repro"))
        return rc, json.load(open(os.path.join(w.out, "repro_local.json"), encoding="utf-8"))

    def test_repro_tolerance_follows_the_dump_gpu(self):
        """PREREG_02 revision: the local check uses the GPU rule of 1.1 point 2 (the same computation on
        the same dump): 1e-3 when the dump was made on a GPU model other than prereg01's, else 1e-4;
        the dump's GPU is the one in its completion marker, not the GPU of a later run."""
        w = World(self.tmp)
        w.bump_p01(5e-4)                                  # largest difference ~5e-4, rule B unchanged
        w.write_gpu_records("GPU-A", dump_gpu="GPU-B")
        rc, rep = self.repro(w)
        self.assertEqual(rc, 0)
        self.assertEqual((rep["tolerance"], rep["gpu_differs_from_prereg01"], rep["dump_gpu_name"]),
                         (1e-3, True, "GPU-B"))
        self.assertEqual(rep["rule_b_differences"], 0)
        self.assertGreater(rep["max_abs_diff"], 4e-4)
        w.write_gpu_records("GPU-A", dump_gpu="GPU-A")   # same GPU model -> 1e-4
        rc, rep = self.repro(w)
        self.assertEqual((rc, rep["tolerance"], rep["gpu_differs_from_prereg01"]), (1, 1e-4, False))
        # a later run on another GPU does not change the rule: the notebook's flag must agree with the marker
        w.write_repro_check(same_gpu_as_p01=False)
        rc, rep = self.repro(w)
        self.assertEqual((rc, rep["gpu_consistent_with_notebook"]), (1, False))
        # no marker: repro_check.json same_gpu_as_p01 decides; nothing known: 1e-4
        w.write_gpu_records("GPU-A")
        rc, rep = self.repro(w)
        self.assertEqual((rc, rep["tolerance"], rep["gpu_decided_by"]), (0, 1e-3, "repro_check.json same_gpu_as_p01"))
        os.remove(os.path.join(w.p02, "repro_check.json"))
        os.remove(os.path.join(w.p01, "env.json"))
        rc, rep = self.repro(w)
        self.assertEqual((rc, rep["tolerance"], rep["gpu_differs_from_prereg01"]), (1, 1e-4, None))
        # a marker without a GPU (CPU) counts as unknown in a real run; in a smoke run it is compared
        w.write_gpu_records("GPU-A")
        env = os.path.join(w.p01, "env.json")
        self.assertIsNone(fuse.repro_gpu({"gpu_name": None}, None, env, "", smoke=False)["differs"])
        self.assertTrue(fuse.repro_gpu({"gpu_name": None}, None, env, "", smoke=True)["differs"])

    def test_repro_needs_a_clean_notebook_run_without_smoke(self):
        w = World(self.tmp)
        rc, rep = self.repro(w)                          # --smoke: the local number, for the seam
        self.assertEqual(rc, 0)
        self.assertIsNone(rep["seam"]["ok"])
        w.set_notebook_smoke(False)
        real = [a for a in w.args("repro") if a != "--smoke"]
        self.assertEqual(fuse.main(real), 1)            # no repro_check.json / p01_readonly_after.json
        w.write_repro_check(same_gpu_as_p01=True, val_c0_vs_p01={"max_abs_diff": rep["max_abs_diff"] + 3e-7})
        after = os.path.join(w.p02, "p01_readonly_after.json")
        with open(after, "w", encoding="utf-8") as f:
            json.dump({"runs": [{"run": 1, "same_as_before": False, "changed_files": ["rank/env.json"]}]}, f)
        self.assertEqual(fuse.main(real), 1)            # prereg01 changed during the notebook run
        rep2 = json.load(open(os.path.join(w.out, "repro_local.json"), encoding="utf-8"))
        self.assertEqual(rep2["notebook_run"]["p01_changed_files"], ["rank/env.json"])
        with open(after, "w", encoding="utf-8") as f:
            json.dump({"runs": [{"run": 1, "same_as_before": False},
                                {"run": 2, "same_as_before": True, "changed_files": []}]}, f)
        self.assertEqual(fuse.main(real), 0)            # the last run is the one that counts
        rep2 = json.load(open(os.path.join(w.out, "repro_local.json"), encoding="utf-8"))
        self.assertTrue(rep2["passed"] and rep2["seam"]["ok"] and rep2["seam"]["required"])
        # seam: the notebook's C0 largest difference must equal the local one within 1e-6
        w.write_repro_check(same_gpu_as_p01=True, val_c0_vs_p01={"max_abs_diff": rep["max_abs_diff"] + 5e-6})
        rc, rep3 = self.repro(w, real)
        self.assertEqual((rc, rep3["seam"]["ok"]), (1, False))
        w.write_repro_check(same_gpu_as_p01=True)         # no notebook number: required without --smoke
        rc, rep3 = self.repro(w, real)
        self.assertEqual((rc, rep3["seam"]["ok"]), (1, None))
        w.set_notebook_smoke(True)
        self.assertEqual(fuse.main(w.args("repro")), 0)   # --smoke: recorded only

    def test_smoke_runs_only_on_smoke_dumps(self):
        w = World(self.tmp)
        w.write_pseudo()
        w.write_opening()
        w.set_notebook_smoke(False)                        # the notebook's own flag says: real dumps
        for cmd in ("repro", "tune", "apply"):
            self.assertEqual(fuse.main(w.args(cmd)), 1, cmd)
        pa = ["--p02-dir", w.p02, "--out-dir", w.out, "--split", w.split_csv, "--smoke"]
        self.assertEqual(pseudo.main(pa), 1)
        os.remove(os.path.join(w.p02, "config.json"))      # no flag at all
        self.assertEqual(fuse.main(w.args("repro")), 1)
        with self.assertRaises(pseudo.P02Error):
            pseudo.check_dump_kind(w.p02, smoke=True)
        w.set_notebook_smoke(True)                         # SMOKE dumps are refused by a real run
        with self.assertRaises(pseudo.P02Error):
            pseudo.check_dump_kind(w.p02, smoke=False)
        self.assertTrue(pseudo.check_dump_kind(w.p02, smoke=True))
        # full-size dumps are refused with --smoke
        with self.assertRaises(pseudo.P02Error):
            pseudo.check_smoke_size(73556, c10data.EXPECTED["inner"]["games"], "inner_keys.npz", True)
        with self.assertRaises(pseudo.P02Error):
            pseudo.check_smoke_size(19156, fuse.VAL_GAMES_FULL, "val_keys.npz", True)
        pseudo.check_smoke_size(19156, fuse.VAL_GAMES_FULL, "val_keys.npz", False)
        pseudo.check_smoke_size(142, fuse.VAL_GAMES_FULL, "val_keys.npz", True)

    def test_p02e9_history_disagreement_is_recorded_not_fatal(self):
        w = World(self.tmp)
        os.makedirs(os.path.join(w.p02, "ckpt"))
        with open(os.path.join(w.p02, "ckpt", "history.csv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("epoch,val_acc,steps\n6,30.0,10\n9,10.0,10\n")     # the monitoring says < 24.15 %
        self.run_all(w)
        t = json.load(open(os.path.join(w.out, "tune_report.json"), encoding="utf-8"))["p02e9"]
        self.assertGreaterEqual(t["inner_accuracy_percent"], fuse.P02E9_MIN_ACC)
        self.assertEqual(t["history_val_acc_percent"], 10.0)
        self.assertFalse(t["history_same_side"])
        self.assertTrue(t["usable"])                                      # the dump decides

    def test_oof_folds_fit_only_on_the_other_folds(self):
        """Every OOF fit uses exactly the fit players outside the evaluated fold; the refit uses all."""
        w = World(self.tmp)
        obj = w.write_pseudo()
        w.write_opening()
        self.assertEqual(fuse.main(w.args("repro")), 0)
        fit_players = {i for i, p in enumerate(obj["players"]) if p["fit"]}
        fold = {i: p["fold"] for i, p in enumerate(obj["players"])}
        calls, orig = [], fuse.fit

        def spy(X, source, pool, opening_, B):
            calls.append(set(X.player.tolist()))
            return orig(X, source, pool, opening_, B)
        fuse.fit = spy
        try:
            self.assertEqual(fuse.main(w.args("tune")), 0)
        finally:
            fuse.fit = orig
        tune = json.load(open(os.path.join(w.out, "tune_report.json"), encoding="utf-8"))
        n_cal = sum(1 for c in tune["combinations"] if c["status"] == "ok" and c["calibrated"])
        self.assertEqual(n_cal, 20)
        self.assertEqual(len(calls), 5 * n_cal + (1 if tune["selection"]["selected"] != "R00" else 0))
        for k, got in enumerate(calls[:5 * n_cal]):
            f = k % 5
            self.assertEqual(got, {i for i in fit_players if fold[i] != f}, (k, f))
        for got in calls[5 * n_cal:]:
            self.assertEqual(got, fit_players)
        for c in tune["combinations"]:
            if c["calibrated"]:
                self.assertEqual([r["fold"] for r in c["folds"]], [0, 1, 2, 3, 4])

    def test_committed_results_check(self):
        """tune / apply (real mode) need the unpacked Colab results committed and unchanged."""
        import subprocess
        repo = os.path.join(self.tmp, "repo")
        p02 = os.path.join(repo, "results", "prereg02", "rank")
        os.makedirs(os.path.join(p02, "dumps"))
        for rel in ("dumps/inner_keys.npz", "dumps/_DONE_inner_keys.json", "repro_check.json", "run_log.json",
                    "config.json"):
            with open(os.path.join(p02, *rel.split("/")), "w", encoding="utf-8") as f:
                f.write("{}\n")
        g = ["git", "-C", repo]
        subprocess.run(g + ["init", "-q"], check=True)
        old = fuse.REPO
        fuse.REPO = repo
        try:
            with self.assertRaises(pseudo.P02Error):          # not committed
                fuse.require_committed_results(p02)
            subprocess.run(g + ["add", "-A"], check=True)
            subprocess.run(g + ["-c", "user.name=kuotunyu", "-c",
                                "user.email=61350295+kuotunyu@users.noreply.github.com",
                                "-c", "commit.gpgsign=false", "commit", "-q", "-m", "t"], check=True)
            self.assertEqual(fuse.require_committed_results(p02)["files"], 5)
            with open(os.path.join(p02, "dumps", "val_keys.npz"), "w", encoding="utf-8") as f:
                f.write("x")                                  # an untracked dump
            with self.assertRaises(pseudo.P02Error):
                fuse.require_committed_results(p02)
            os.remove(os.path.join(p02, "dumps", "val_keys.npz"))
            with open(os.path.join(p02, "repro_check.json"), "w", encoding="utf-8") as f:
                f.write('{"pass": true}\n')                   # a changed record
            with self.assertRaises(pseudo.P02Error):
                fuse.require_committed_results(p02)
            with self.assertRaises(pseudo.P02Error):          # outside the repository
                fuse.require_committed_results(os.path.join(self.tmp, "elsewhere"))
        finally:
            fuse.REPO = old


    def test_tool_hashes_ignore_line_ends(self):
        path = os.path.join(self.tmp, "t.py")
        with open(path, "wb") as f:
            f.write(b"a = 1\nb = 2\n")
        lf = fuse.sha256_text(path)
        with open(path, "wb") as f:
            f.write(b"a = 1\r\nb = 2\r\n")
        self.assertEqual(fuse.sha256_text(path), lf)
        self.assertEqual(fuse.tool_state()["files_sha256"]["p02_rank_fuse.py"],
                         fuse.sha256_text(os.path.join(TOOLS, "p02_rank_fuse.py")))

    def test_p02e9_dropped_when_missing_or_weak(self):
        w = World(self.tmp)
        os.remove(os.path.join(w.dumps, "val_p02e9_d4.npz"))
        self.run_all(w)
        tune = json.load(open(os.path.join(w.out, "tune_report.json"), encoding="utf-8"))
        dropped = [c["id"] for c in tune["combinations"] if c["status"] == "dropped"]
        self.assertEqual(dropped, ["R03", "R04", "R09", "R10", "R13", "R14", "R19", "R20"])
        self.assertFalse(tune["p02e9"]["usable"])
        # weak epoch 9: shuffle the inner logits so its accuracy falls to chance
        w2 = World(os.path.join(self.tmp, "b"))
        z = w2.inner_logits["p02e9_d4"][np.random.default_rng(0).permutation(len(w2.games))]
        save_npz(os.path.join(w2.dumps, "inner_p02e9_d4.npz"), logits=z,
                 weights_sha256=np.array("w_p02e9_d4"), syms=np.array(8))
        w2.write_pseudo()
        w2.write_opening()
        self.assertEqual(fuse.main(w2.args("repro")), 0)
        self.assertEqual(fuse.main(w2.args("tune")), 0)
        t2 = json.load(open(os.path.join(w2.out, "tune_report.json"), encoding="utf-8"))
        self.assertLess(t2["p02e9"]["inner_accuracy_percent"], fuse.P02E9_MIN_ACC)
        self.assertEqual([c["id"] for c in t2["combinations"] if c["status"] == "dropped"],
                         ["R03", "R04", "R09", "R10", "R13", "R14", "R19", "R20"])



class Bootstrap(unittest.TestCase):
    def test_player_bootstrap_by_hand(self):
        """Players resampled within each level; per level the question-weighted mean; mean over levels."""
        rng = np.random.default_rng(7)
        n_players = 9
        level_of_player = np.array([0, 0, 0, 3, 3, 5, 5, 5, 5])
        player = np.concatenate([np.full(rng.integers(1, 6), p) for p in range(n_players)])
        y = level_of_player[player]
        sc = rng.random(len(player))
        ref = rng.random(len(player))
        old = fuse.N_BOOT
        fuse.N_BOOT = 11
        try:
            got = fuse.PlayerBootstrap(player, y, n_players).summary(sc, ref)
        finally:
            fuse.N_BOOT = old
        draw_rng = np.random.default_rng(fuse.BOOT_SEED)
        draws = {}
        for lv in range(10):
            members = [p for p in range(n_players) if level_of_player[p] == lv]
            if members:
                draws[lv] = [[members[j] for j in row] for row in draw_rng.integers(0, len(members), size=(11, len(members)))]

        def stat(v, b):
            per = []
            for lv, rows in draws.items():
                num = sum(v[player == p].sum() for p in rows[b])
                den = sum((player == p).sum() for p in rows[b])
                per.append(num / den)
            return float(np.mean(per))
        a = np.array([stat(sc, b) for b in range(11)])
        d = a - np.array([stat(ref, b) for b in range(11)])
        self.assertAlmostEqual(got["se"], float(np.std(a, ddof=1)), places=12)
        self.assertAlmostEqual(got["diff_vs_R00_se"], float(np.std(d, ddof=1)), places=12)
        np.testing.assert_allclose(got["diff_vs_R00_ci95"], [np.percentile(d, 2.5), np.percentile(d, 97.5)],
                                   atol=1e-12)


if __name__ == "__main__":
    unittest.main()
