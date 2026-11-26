"""prereg57: small CPU tests of p57_ft.py / p57_eval.py (synthetic data only; no score of the method is computed).

    PY -m pytest -q results/prereg57/explore/test_p57.py
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p57_ft as FT  # noqa: E402

L, V = FT.L, FT.V


def random_game(rng, n_moves):
    """Alternating colours on distinct points: never an occupied point, whatever is captured."""
    pts = rng.permutation(361)[:n_moves]
    col = np.arange(n_moves) % 2
    return (col * 361 + pts).astype(np.uint16)


class World:
    """A tiny move cache: 3 base classes x 4 games, then 3 candidates with 3 / 2 / 4 games (one of them too short
    to give a sample), then 2 question games."""

    def __init__(self, d, seed=0):
        rng = np.random.default_rng(seed)
        games = [random_game(rng, 40 + int(rng.integers(20))) for _ in range(12 + 9 + 2)]
        self.short = 12 + 4                                  # second game of candidate 1: 6 moves -> no sample
        games[self.short] = random_game(rng, 6)
        self.off = np.concatenate([[0], np.cumsum([len(g) for g in games])]).astype(np.int64)
        self.all = np.concatenate(games)
        np.save(os.path.join(d, "codes.npy"), self.all)
        self.dir = d
        self.is_white = rng.integers(0, 2, len(games)).astype(bool)
        self.t_games = np.arange(12, dtype=np.int64)
        self.t_off = np.array([0, 4, 8, 12], np.int64)
        self.cand_groups = [[12, 13, 14], [15, 16], [17, 18, 19, 20]]
        self.q_games = [21, 22]
        self.codes = self.all                               # in memory: no memory map keeps the temp file open
        self.games = games

    def base(self, n, moves=6):
        gb = V.GameBatches(self.dir, self.off, self.is_white, self.t_games, self.t_off, 0, batch_games=n,
                           moves=moves)
        gb.codes = self.all
        return gb

    def mixed(self, share=0.25, segments=8, moves=6, offset=1000):
        n_pool, n_base = FT.split_segments(share, segments)
        return FT.MixedBatches(self.base(n_base, moves) if n_base else None, offset, 3, self.codes, self.off[:-1],
                               self.off[1:], self.is_white, self.cand_groups, n_pool, 0, moves)


class Head(unittest.TestCase):
    def test_init_rows_are_group_vecs(self):
        import p36_eval as EV
        rng = np.random.default_rng(1)
        emb = rng.normal(size=(9, 8)).astype(np.float32)
        count = np.array([3, 0, 5, 2, 2, 0, 0, 1, 4])
        groups = [[0, 1, 2], [3, 4], [5, 6], [7, 8]]
        rows, has = FT.head_init_rows(emb, count, groups)
        self.assertEqual(rows.shape, (4, 8))
        self.assertEqual(has.tolist(), [True, True, False, True])
        self.assertTrue(np.allclose(np.linalg.norm(rows[has], axis=1), 1.0))
        self.assertTrue((rows[2] == 0).all())
        want, whas = EV.Emb(emb, count).group_vecs(groups)
        self.assertTrue(np.array_equal(rows, want) and np.array_equal(has, whas))
        v = emb[[0, 2]].astype(np.float64).mean(0)             # the game without a sample is left out
        self.assertTrue(np.allclose(rows[0], v / np.linalg.norm(v)))

    def test_extension(self):
        base = np.arange(40, dtype=np.float64).reshape(5, 8)
        rows = np.ones((3, 8))
        w = FT.extended_head_weight(base, rows)
        self.assertEqual(w.shape, (8, 8))
        self.assertEqual(w.dtype, np.float32)
        self.assertTrue(np.array_equal(w[:5], base.astype(np.float32)) and np.array_equal(w[5:], rows))
        with self.assertRaises(ValueError):
            FT.extended_head_weight(base, np.ones((3, 7)))

    def test_torch_head(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("torch")
        torch.manual_seed(0)
        spec = V.spec_of(FT.VARIANT)
        n_base, n_cand = 7, 3
        _net, base_head = V.build(spec, n_base)
        rng = np.random.default_rng(2)
        emb = rng.normal(size=(6, L.EMB_DIM)).astype(np.float32)
        rows, has = FT.head_init_rows(emb, np.ones(6, int), [[0, 1], [2, 3], [4, 5]])
        self.assertTrue(has.all())
        _net2, head = V.build(spec, n_base + n_cand)
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(FT.extended_head_weight(base_head.weight.detach().numpy(), rows)))
        self.assertEqual(tuple(head.weight.shape), (n_base + n_cand, L.EMB_DIM))
        self.assertTrue(torch.equal(head.weight[:n_base], base_head.weight))
        x = F.normalize(torch.from_numpy(rows.astype(np.float32)), dim=1)
        lg = V.head_logits(head, spec, x)
        self.assertEqual(tuple(lg.shape), (n_cand, n_base + n_cand))
        self.assertEqual(lg.argmax(1).tolist(), [n_base, n_base + 1, n_base + 2])
        self.assertTrue(torch.allclose(lg[torch.arange(3), n_base + torch.arange(3)], torch.full((3,), L.SCALE),
                                       atol=1e-3))
        # the base classes' logits are the base head's
        self.assertTrue(torch.allclose(lg[:, :n_base], V.head_logits(base_head, spec, x), atol=1e-5))


class Sampler(unittest.TestCase):
    def test_split(self):
        self.assertEqual(FT.split_segments(0.25), (24, 72))
        self.assertEqual(FT.split_segments(0.5), (48, 48))
        self.assertEqual(FT.split_segments(1.0), (96, 0))
        self.assertEqual(FT.split_segments("0.25", 8), (2, 6))
        for bad in (0.3, 0.0, 1.5):
            with self.assertRaises(SystemExit):
                FT.split_segments(bad)

    def test_share_labels_and_base_part(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            ds = w.mixed(0.25, 8, 6, offset=1000)
            for s in (0, 5):
                b = ds.batch(s)
                self.assertEqual(len(b["y"]), 8)
                self.assertEqual((b["y"] >= 3).sum(), 2)                 # the requested share: 2 of 8 segments
                self.assertTrue((b["y"][:6] < 3).all() and (b["y"][6:] >= 3).all())
                self.assertTrue(b["y"].max() < 3 + len(w.cand_groups))
                self.assertEqual(b["x"].dtype, np.uint8)
                self.assertEqual(len(b["x"]), len(b["seg"]))
                self.assertEqual(len(b["x"]), len(b["t"]))
                self.assertTrue((np.bincount(b["seg"], minlength=8) <= 6).all())
                self.assertTrue((np.bincount(b["seg"], minlength=8) >= 1).all())
                self.assertTrue((np.diff(b["seg"]) >= 0).all())
                ref = w.base(6).batch(1000 + s)                            # the base sampler's own batch
                nb = len(ref["seg"])
                for k in ("x", "seg", "t"):
                    self.assertTrue(np.array_equal(b[k][:nb], ref[k]), k)
                self.assertTrue(np.array_equal(b["y"][:6], ref["y"]))
                # the base part is the first 6 segments of the base's full-size batch of the same number
                full = w.base(8).batch(1000 + s)
                self.assertTrue(np.array_equal(full["y"][:6], ref["y"]))
            self.assertTrue(all(np.array_equal(ds.batch(7)[k], ds.batch(7)[k]) for k in ("x", "seg", "y", "t")))
            self.assertFalse(np.array_equal(ds.batch(7)["x"], ds.batch(8)["x"]))

    def test_pool_part_is_the_candidates_own_moves(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            ds = w.mixed(0.5, 8, 6)
            codes_all = w.all
            seen_c, seen_g = [], set()
            for s in range(120):
                b = ds.batch(s)
                draws = ds.draw_pool(s)
                self.assertEqual(len(draws), 4)
                self.assertEqual(b["y"][4:].tolist(), [3 + c for c, _g, _x, _t in draws])
                pos = 0
                nb = int((b["seg"] < 4).sum())
                for c, g, x, t in draws:
                    self.assertIn(g, w.cand_groups[c])                     # a game of that candidate
                    self.assertNotIn(g, w.q_games)                         # never a question's game
                    self.assertNotEqual(g, w.short)
                    codes = codes_all[w.off[g]:w.off[g + 1]]
                    col = "w" if w.is_white[g] else "b"
                    self.assertTrue(((codes[t] >= 361) == w.is_white[g]).all())    # the candidate's colour only
                    self.assertTrue((t >= L.HIST - 1).all() and len(t) <= 6 and len(set(t.tolist())) == len(t))
                    full, all_t = L.planes_for_codes(codes, col)
                    sel = full[np.searchsorted(all_t, t)]
                    self.assertTrue(any(np.array_equal(x, L.sym(sel, k)) for k in range(8)))
                    self.assertTrue(np.array_equal(b["x"][nb + pos:nb + pos + len(x)], x))
                    self.assertTrue(np.array_equal(b["t"][nb + pos:nb + pos + len(x)], t))
                    pos += len(x)
                    seen_c.append(c)
                    seen_g.add(g)
            frac = np.bincount(seen_c, minlength=3) / float(len(seen_c))    # uniform over candidates, not games
            self.assertTrue((np.abs(frac - 1 / 3.0) < 0.08).all(), frac)
            self.assertEqual(seen_g, {12, 13, 14, 15, 17, 18, 19, 20})      # 16 (too short) is never used

    def test_pool_only_and_torch_item(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            ds = w.mixed(1.0, 8, 6)
            b = ds.batch(3)
            self.assertTrue((b["y"] >= 3).all() and len(b["y"]) == 8)
            try:
                import torch
            except ImportError:
                return
            it = ds[3]
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in it.values()))
            self.assertTrue(np.array_equal(it["y"].numpy(), b["y"]))

    def test_candidate_without_game(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            with self.assertRaises(ValueError):
                FT.MixedBatches(None, 0, 3, w.codes, w.off[:-1], w.off[1:], w.is_white, [[12], []], 8)


class Order(unittest.TestCase):
    POOL = {"cand_rows": [[50, 30], [90, 10]], "q_rows": [[70], [20, 80]], "seed": 0}

    def test_pool_rows(self):
        self.assertEqual(FT.pool_rows(self.POOL["cand_rows"], self.POOL["q_rows"]).tolist(),
                         [50, 30, 90, 10, 70, 20, 80])

    def test_file_round_trip_scores_like_the_full_store(self):
        """A per-pool file (candidates' games, then questions') gives the backend the same games as the full
        store file does: p38_backend.pool_games is identical."""
        import p36_eval as EV
        import p38_backend as BK
        rng = np.random.default_rng(3)
        store_rows = np.arange(0, 100, 10)
        emb = rng.normal(size=(10, 6)).astype(np.float32)
        count = np.array([4, 2, 0, 3, 1, 5, 2, 2, 7, 1])
        colour = np.array(["B", "W"] * 50)
        full = EV.Emb(emb, count, store_rows, colour, "s", "full")
        rows = FT.pool_rows(self.POOL["cand_rows"], self.POOL["q_rows"])
        idx = [int(r) // 10 for r in rows]
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "v.npz")
            FT.write_vectors(f, emb[idx], count[idx], "sha", "tag", 1000, {"weights": "ema", "tta": 1}, "ident", rows)
            v = FT.read_vectors(f)
        self.assertTrue(np.array_equal(v["store_row"], rows))
        self.assertEqual((v["ckpt_sha256"], v["ckpt"], v["ckpt_step"], v["pool_identity"]),
                         ("sha", "tag", 1000, "ident"))
        self.assertEqual(v["embed_cfg"], {"weights": "ema", "tta": 1})
        part = EV.Emb(v["emb"], v["count"], v["store_row"], colour, v["ckpt_sha256"], v["ckpt"])
        a, b = BK.pool_games(full, self.POOL), BK.pool_games(part, self.POOL)
        self.assertEqual(sorted(a), sorted(b))
        for k in a:
            self.assertTrue(np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k)
        ca, _ = full.group_vecs(self.POOL["cand_rows"])
        cb, _ = part.group_vecs(self.POOL["cand_rows"])
        self.assertTrue(np.array_equal(ca, cb))
        # the head's initial rows = the candidates' prototypes the eval code builds
        groups = [[0, 1], [2, 3]]
        rows0, _has = FT.head_init_rows(v["emb"], v["count"], groups)
        self.assertTrue(np.array_equal(rows0, cb))

    def test_official_layout_has_no_store_row(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "v.npz")
            FT.write_vectors(f, np.zeros((3, 4)), np.ones(3), "sha", "tag", 5, {}, "ident")
            v = FT.read_vectors(f)
            self.assertIsNone(v["store_row"])
            self.assertEqual(v["emb"].dtype, np.float32)

    def test_identity(self):
        codes = np.arange(40, dtype=np.uint16)
        start, end, wh = np.array([0, 10, 25]), np.array([10, 25, 40]), np.array([False, True, False])
        g = FT.games_sha(codes, start, end, wh)
        self.assertNotEqual(g, FT.games_sha(codes, start, end, np.array([False, False, False])))
        self.assertNotEqual(g, FT.games_sha(codes, start, np.array([10, 24, 40]), wh))
        a = FT.identity_of("E", 0, [[0], [1]], [[2]], [5, 6, 7], ["a", "b"], g)
        self.assertEqual(a, FT.identity_of("E", 0, [[0], [1]], [[2]], np.array([5, 6, 7]), ["a", "b"], g))
        for other in (FT.identity_of("E", 1, [[0], [1]], [[2]], [5, 6, 7], ["a", "b"], g),
                      FT.identity_of("E", 0, [[1], [0]], [[2]], [5, 6, 7], ["a", "b"], g),
                      FT.identity_of("E", 0, [[0], [1]], [[2]], [5, 6, 8], ["a", "b"], g),
                      FT.identity_of("D", 0, [[0], [1]], [[2]], [5, 6, 7], ["a", "b"], g)):
            self.assertNotEqual(a, other)


class Resume(unittest.TestCase):
    def test_highest_step_number(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(FT.last_ckpt(d))
            for n in (900, 1000, 99000, 100000, 6000):
                open(os.path.join(d, os.path.basename(FT.ckpt_file("x", n))), "w").close()
            self.assertEqual(os.path.basename(FT.last_ckpt(d)), "step_100000.pt")
            open(os.path.join(d, "step_96000.pt"), "w").close()               # 5-digit names sort after 6-digit ones
            open(os.path.join(d, "step_288000.pt"), "w").close()
            self.assertEqual(os.path.basename(FT.last_ckpt(d)), "step_288000.pt")

    def test_names(self):
        self.assertEqual(os.path.basename(FT.ckpt_file("x", 1000)), "step_001000.pt")
        self.assertEqual(os.path.basename(FT.emb_file("x", 1000)), "step_1000_ema.npz")
        self.assertEqual(FT.run_tag("0.25", "2e-4"), "sh0.25_lr2e-4")
        self.assertTrue(FT.dev_dir("t", 1).replace("\\", "/").endswith("data/prereg57/dev/t/D1"))
        self.assertTrue(FT.judge_dir(9).replace("\\", "/").endswith("data/prereg57/judge/E9"))
        self.assertEqual(len(FT.GRID) * len(FT.DEV_STEPS), 12)

    def test_chains_use_the_registered_grid_and_names(self):
        def text(name):
            with open(os.path.join(HERE, name), encoding="utf-8") as f:
                return f.read()
        dev, common = text("run_dev.sh"), text("chain_common.sh")
        self.assertIn("for cfg in %s; do" % " ".join('"%s %s"' % g for g in FT.GRID), dev)
        self.assertIn("for p in %s; do" % " ".join(str(i) for i in FT.DEV_POOLS), dev)
        self.assertEqual(dev.count("for n in %s; do" % " ".join(str(n) for n in FT.DEV_STEPS)), 2)
        self.assertIn("adapt_run S0 D $p $share $lr %d %s $out" % (FT.DEV_STEPS[-1],
                                                                  ",".join(str(n) for n in FT.DEV_STEPS)), dev)
        self.assertIn("out=data/prereg57/dev/%s/D$p" % FT.run_tag("${share}", "${lr}"), dev)
        self.assertIn("for p in %s; do" % " ".join(str(i) for i in FT.JUDGE_POOLS), text("run_judge.sh"))
        self.assertIn("out=data/prereg57/judge/E$p", text("run_judge.sh"))
        self.assertIn("out=data/prereg57/apply/$pool", text("run_apply.sh"))
        # the file names the chains test for = the names p57_ft / p57_eval use
        self.assertIn("printf 'step_%06d.pt'", common)
        self.assertIn("npz=$out/emb/step_${n}_ema.npz", common)
        self.assertEqual(os.path.basename(FT.emb_file("o", 2000)), "step_2000_ema.npz")
        self.assertIn("-ge 7000", common)
        for name in ("run_dev.sh", "run_judge.sh", "run_apply.sh", "chain_common.sh"):
            self.assertNotIn("\r", text(name))

    def test_log_and_stop(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "train_log.jsonl")
            with open(p, "w") as f:
                for step, loss in ((100, 5.0), (200, 4.0), (300, 3.5), (200, 4.1), (300, 3.6), (400, 3.0)):
                    f.write(json.dumps({"step": step, "loss": loss}) + "\n")
            rows = FT.read_log(p, 300)                                        # resumed from step 100: later lines win
            self.assertEqual([(r["step"], r["loss"]) for r in rows], [(100, 5.0), (200, 4.1), (300, 3.6)])
            self.assertEqual(FT.read_log(os.path.join(d, "none"), 10), [])
            s = os.path.join(d, "STOP")
            self.assertIsNone(FT.stop_requested((s, None)))
            open(s, "w").close()
            self.assertEqual(FT.stop_requested((None, s)), s)

    def test_dev_choice_guard(self):
        self.assertIsNone(FT.require_dev_choice("D", 0.5, 2e-4, 1000))
        with tempfile.TemporaryDirectory(dir=HERE) as d:                      # not committed -> refused
            p = os.path.join(d, "dev.json")
            with open(p, "w") as f:
                json.dump({"gate": {"pass": True}, "chosen": {"share": "0.5", "lr": "2e-4", "n": 2000}}, f)
            for kind in ("E", "official"):
                with self.assertRaises(SystemExit):
                    FT.require_dev_choice(kind, 0.5, 2e-4, 2000, dev_json=p)
                with self.assertRaises(SystemExit):
                    FT.require_dev_choice(kind, 0.5, 2e-4, 2000, dev_json=os.path.join(d, "missing.json"))


class Eval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import p57_eval as PE
        cls.PE = PE

    @staticmethod
    def setting(mean, n, share, lr="2e-4", diff=(0.02, 0.02)):
        return {"v26": {"mean": mean}, "n": n, "share": share, "lr": lr, "tag_n": "sh%s_lr%s_n%d" % (share, lr, n),
                "minus_reference": {"per_pool": list(diff)}}

    def test_choose(self):
        s = self.setting
        a = [s(0.70, 4000, "0.5"), s(0.71, 6000, "0.5"), s(0.71, 2000, "0.5"), s(0.71, 2000, "0.25"),
             s(0.71, 2000, "0.5", "5e-4")]
        c = self.PE.choose(a)
        self.assertEqual((c["n"], c["share"]), (2000, "0.25"))                # ties: fewer steps, then smaller share
        self.assertEqual(self.PE.choose([s(0.70, 1000, "0.25"), s(0.72, 6000, "0.5")])["n"], 6000)

    def test_gate(self):
        s = self.setting
        good, one_pool, both_low = s(0.72, 1000, "0.5"), s(0.73, 2000, "0.5", diff=(0.05, 0.009)), \
            s(0.71, 4000, "0.5", diff=(0.004, 0.004))
        g = self.PE.gate_of([good, both_low], good)
        self.assertTrue(g["pass"] and g["any_setting_passes"] and g["note"] is None)
        g = self.PE.gate_of([good, one_pool], one_pool)                       # the chosen one fails on one pool
        self.assertFalse(g["pass"])
        self.assertTrue(g["any_setting_passes"] and g["note"])
        g = self.PE.gate_of([both_low], both_low)
        self.assertFalse(g["pass"] or g["any_setting_passes"])
        edge = s(0.7, 1000, "0.5", diff=(0.01, 0.01))
        self.assertTrue(self.PE.gate_of([edge], edge)["pass"])

    def test_quick_combine(self):
        g0 = [{"wE": 2, "k": 10, "score": 0.5}, {"wE": 4, "k": 10, "score": 0.7}, {"wE": 8, "k": 10, "score": 0.6}]
        g1 = [{"wE": 2, "k": 10, "score": 0.6}, {"wE": 4, "k": 10, "score": 0.5}, {"wE": 8, "k": 10, "score": 0.6}]
        cells, best = self.PE.quick_combine([g0, g1])
        self.assertEqual([c["mean"] for c in cells], [0.55, 0.6, 0.6])
        self.assertEqual((best["wE"], best["per_pool"]), (4, [0.7, 0.5]))    # the first maximum
        self.assertEqual(self.PE.cell_of(cells, 8, 10)["per_pool"], [0.6, 0.6])

    def test_load_emb_refuses_another_order(self):
        pool = dict(Order.POOL)
        rows = FT.pool_rows(pool["cand_rows"], pool["q_rows"])
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "v.npz")
            FT.write_vectors(f, np.zeros((7, 4)), np.ones(7), "sha", "tag", 5, {}, "ident", rows[::-1])
            with self.assertRaises(SystemExit):
                self.PE.load_emb(f, pool)
            g = os.path.join(d, "w.npz")
            FT.write_vectors(g, np.zeros((7, 4)), np.ones(7), "sha", "tag", 5, {}, "ident")
            with self.assertRaises(SystemExit):
                self.PE.load_emb(g, pool)

    def test_check_vectors(self):
        PE = self.PE
        with tempfile.TemporaryDirectory() as d:
            ck = os.path.join(d, "step_001000.pt")
            with open(ck, "wb") as f:
                f.write(b"weights")
            sha = L.sha256_file(ck)
            c57 = dict(PE.RECIPE, base="S0", pool_kind="D", pool=0, share=0.25, lr=2e-4, pool_identity="ident",
                       base_ckpt_sha256=FT.BASES["S0"]["sha256"], pool_segments=24)

            def vec(**kw):
                cfg = {"weights": "ema", "tta": 1, "limit": None, "p57": dict(c57)}
                v = {"embed_cfg": cfg, "ckpt_step": 1000, "ckpt_sha256": sha, "pool_identity": "ident",
                     "count": np.array([3, 4])}
                for k, x in kw.items():
                    if k in v:
                        v[k] = x
                    elif k in cfg:
                        cfg[k] = x
                    else:
                        cfg["p57"][k] = x
                return v

            path = os.path.join(d, "v.npz")
            np.savez(path, x=np.zeros(1))
            ok = dict(path=path, ckpt=ck, base="S0", kind="D", name=0, share="0.25", lr="2e-4", n=1000,
                      ref_count=np.array([3, 4]), after=0)
            self.assertEqual(PE.check_vectors(vec(), **ok)["ckpt_sha256"], sha)
            for kw in ({"weights": "raw"}, {"limit": 16}, {"share": 0.5}, {"lr": 5e-4}, {"pool": 1},
                       {"pool_kind": "E"}, {"smoke": True}, {"ckpt_step": 2000}, {"ckpt_sha256": "0" * 64},
                       {"count": np.array([3, 5])}, {"pool_identity": "other"}, {"warmup": 100}, {"base": "c49"}):
                with self.assertRaises(SystemExit, msg=str(kw)):
                    PE.check_vectors(vec(**kw), **ok)
            with self.assertRaises(SystemExit):                               # older than the commit
                PE.check_vectors(vec(), **dict(ok, after=os.path.getmtime(path) + 10))

    def test_assemble_check(self):
        head = b"question_id,top_1,top_2,top_3,top_4,top_5\n"
        orig = b"".join(b"pub_q_%d,a,b,c,d,e\n" % i for i in range(400)) + \
            b"".join(b"priv_q_%d,a,b,c,d,e\n" % i for i in range(400))
        new2 = b"".join(b"pub2_q_%d,a,b,c,d,e\n" % i for i in range(400)) + \
            b"".join(b"priv2_q_%d,a,b,c,d,e\n" % i for i in range(400))
        old = head + orig + new2.replace(b",a,", b",z,")
        self.assertTrue(self.PE.assemble_check(head + orig + new2, old)["ok"])
        self.assertFalse(self.PE.assemble_check(head + orig.replace(b"pub_q_7,a", b"pub_q_7,b") + new2, old)["ok"])
        self.assertFalse(self.PE.assemble_check(head + orig + new2[:-22], old)["ok"])        # one row short
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.csv")
            with open(p, "wb") as f:
                f.write(old.replace(b"\n", b"\r\n"))
            data, crlf = self.PE.lf_bytes(p)
            self.assertTrue(crlf and data == old)

    def test_e_use_count(self):
        u = self.PE.e_uses_so_far()
        self.assertEqual(u["count"], len(u["through_prereg49"]) + len(u["prereg51"]) + len(u["prereg54"]))
        self.assertEqual(len(u["through_prereg49"]), 6)


if __name__ == "__main__":
    unittest.main()
