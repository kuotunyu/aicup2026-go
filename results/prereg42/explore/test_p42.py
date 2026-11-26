"""prereg42 explore: tests on synthetic games / pools (no move cache, no feature store, no E pools).

    PY -m pytest -q results/prereg42/explore/test_p42.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p42_lib as V  # noqa: E402

L = V.L


def synth_cache(n_players=6, games_each=12, seed=0, moves=70):
    """Random games (distinct points, alternating colours) as the move cache arrays + T grouping."""
    rng = np.random.default_rng(seed)
    parts, off, white = [], [0], []
    for _g in range(n_players * games_each):
        pts = rng.permutation(361)[:moves]
        codes = np.array([p + (361 if i % 2 else 0) for i, p in enumerate(pts)], np.uint16)
        parts.append(codes)
        off.append(off[-1] + len(codes))
        white.append(bool(rng.random() < 0.5))
    codes = np.concatenate(parts)
    order = np.arange(n_players * games_each)
    t_off = np.arange(0, n_players * games_each + 1, games_each).astype(np.int64)
    return codes, np.asarray(off, np.int64), np.asarray(white), order, t_off


class MemCodes:
    """GameBatches with in-memory codes (no cache directory)."""

    @staticmethod
    def patch(ds, codes):
        ds.codes = codes
        return ds


class Samplers(unittest.TestCase):
    def setUp(self):
        self.codes, self.off, self.white, self.order, self.t_off = synth_cache()

    def test_game_batches_equal_p36(self):
        import p36_train
        a = MemCodes.patch(V.GameBatches("x", self.off, self.white, self.order, self.t_off, 0, 8, 10), self.codes)
        b = p36_train.TrainSet("x", self.off, self.white, self.order, self.t_off, 0, 8, 10)
        b.codes = self.codes
        for s in (0, 1, 17, 39999):
            ra = a.batch(s)
            xb, segb, yb = b.batch(s)
            self.assertTrue(np.array_equal(ra["x"], xb))
            self.assertTrue(np.array_equal(ra["seg"], segb))
            self.assertTrue(np.array_equal(ra["y"], yb))
            self.assertEqual(len(ra["t"]), len(ra["seg"]))

    def test_episode_shape(self):
        ds = MemCodes.patch(V.EpisodeBatches("x", self.off, self.white, self.order, self.t_off, 0, 4, 36), self.codes)
        r = ds.batch(5)
        r2 = ds.batch(5)
        for k in r:
            self.assertTrue(np.array_equal(r[k], r2[k]))          # deterministic per step
        self.assertEqual(len(set(r["y"].tolist())), 4)             # different players
        n_games = len(r["gset"])
        self.assertEqual(int(r["seg"].max()) + 1, n_games)
        self.assertEqual(sorted(set(r["gset"].tolist())), list(range(8)))
        for i in range(8):
            k = int((r["gset"] == i).sum())
            self.assertIn(k, V.EP_KS)
        per_game = np.bincount(r["seg"])
        for gi, gs in enumerate(r["gset"].tolist()):
            k = int((r["gset"] == gs).sum())
            self.assertLessEqual(per_game[gi], max(1, 36 // k))


class Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.manual_seed(0)
        torch.set_num_threads(2)
        cls.torch = torch

    def batch(self, n_games=3, moves=5):
        torch = self.torch
        g = torch.Generator().manual_seed(1)
        x = (torch.rand(n_games * moves, L.N_PLANES, 19, 19, generator=g) < 0.2).to(torch.uint8)
        seg = torch.arange(n_games).repeat_interleave(moves)
        t = torch.arange(n_games * moves) % 40 + 7
        return x, seg, t

    def test_forward_all_variants(self):
        for v in sorted(V.VARIANTS):
            spec = V.spec_of(v)
            net, head = V.build(spec, 5)
            net.eval()
            x, seg, t = self.batch()
            out = net(x.float(), seg, 3, t)
            self.assertEqual(tuple(out.shape), (3, L.EMB_DIM))
            emb = self.torch.nn.functional.normalize(out, dim=1)
            lg = V.head_logits(head, spec, emb, self.torch.tensor([0, 1, 2]))
            self.assertEqual(tuple(lg.shape), (3, 5))

    def test_txpool_packing(self):
        """A game's vector does not depend on the other games of the batch or on the order of the segments."""
        net, _h = V.build(V.spec_of("V3"))
        net.eval()
        x, seg, t = self.batch(3, 4)
        with self.torch.no_grad():
            e = net.trunk(x.float())
            allv = net.pool(e, seg, 3, t)
            one = net.pool(e[seg == 1], self.torch.zeros(4, dtype=self.torch.int64), 1, t[seg == 1])
            perm = self.torch.tensor([8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7])
            pv = net.pool(e[perm], seg[perm], 3, t[perm])
        self.assertTrue(self.torch.allclose(allv[1], one[0], atol=1e-5))
        self.assertTrue(self.torch.allclose(allv, pv, atol=1e-5))

    def test_v3b_step0_is_segment_mean(self):
        """PLAN.md revision 3: at initialisation V3b's game vector is exactly prereg36's mean pooling (train mode,
        dropout on), including a game without samples (zeros)."""
        torch = self.torch
        net, _h = V.build(V.spec_of("V3b"), 5)
        net.train()
        x, seg, t = self.batch(3, 5)
        e = net.trunk(x.float())
        g = net.pool(e, seg, 4, t)
        ref = L.segment_mean(e, seg, 4)
        self.assertTrue(torch.equal(g, ref))
        self.assertEqual(float(g[3].detach().abs().sum()), 0.0)
        self.assertLess(float(net.pool.pos.weight.detach().std()), 0.03)
        self.assertFalse(hasattr(net.pool, "cls"))

    def test_v3b_packing_and_empty(self):
        """With a non-zero readout: a game's vector does not depend on the other games or the segment order; an
        empty game stays zero."""
        torch = self.torch
        net, _h = V.build(V.spec_of("V3b"))
        with torch.no_grad():
            net.pool.out.weight.normal_(0, 0.1)
            net.pool.out.bias.normal_(0, 0.1)
        net.eval()
        x, seg, t = self.batch(3, 4)
        with torch.no_grad():
            e = net.trunk(x.float())
            allv = net.pool(e, seg, 4, t)
            one = net.pool(e[seg == 1], torch.zeros(4, dtype=torch.int64), 1, t[seg == 1])
            perm = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7])
            pv = net.pool(e[perm], seg[perm], 4, t[perm])
        self.assertTrue(torch.allclose(allv[1], one[0], atol=1e-5))
        self.assertTrue(torch.allclose(allv, pv, atol=1e-5))
        self.assertFalse(torch.allclose(allv[:3], L.segment_mean(e, seg, 4)[:3], atol=1e-3))
        self.assertEqual(float(allv[3].abs().sum()), 0.0)

    def test_param_groups(self):
        for v in sorted(V.VARIANTS):
            spec = V.spec_of(v)
            net, head = V.build(spec, 5)
            gs = V.param_groups(net, head, spec)
            ids = [id(p) for g in gs for p in g["params"]]
            allp = list(net.parameters()) + list(head.parameters())
            self.assertEqual(sorted(ids), sorted(id(p) for p in allp))
            if v == "V3b":
                self.assertEqual(len(gs), 2)
                self.assertEqual(gs[0]["lr"], V.LR)
                self.assertAlmostEqual(gs[1]["lr"], V.LR * 0.1)
                pool_ids = set(id(p) for p in net.pool.parameters())
                self.assertEqual(set(id(p) for p in gs[1]["params"]), pool_ids)
            else:
                self.assertEqual(len(gs), 1)
                self.assertEqual(gs[0]["lr"], V.LR)

    def test_subcenter_margin(self):
        net, head = V.build(V.spec_of("V4"), 4)
        emb = self.torch.nn.functional.normalize(self.torch.randn(2, L.EMB_DIM), dim=1)
        y = self.torch.tensor([1, 3])
        head.m = 0.0
        a = head(emb, y)
        head.m = 0.1
        b = head(emb, y)
        d = (a - b).detach().numpy()
        self.assertAlmostEqual(float(d[0, 1]), L.SCALE * 0.1, places=4)
        self.assertAlmostEqual(float(d[1, 3]), L.SCALE * 0.1, places=4)
        self.assertEqual(float(np.abs(d[0, [0, 2, 3]]).max()), 0.0)
        self.assertEqual(V.margin_at(0), 0.0)
        self.assertAlmostEqual(V.margin_at(2000), 0.05)
        self.assertEqual(V.margin_at(10 ** 6), V.MARGIN)

    def test_sym_t_equals_sym(self):
        a = np.arange(2 * 3 * 19 * 19).reshape(2, 3, 19, 19)
        for s in range(8):
            self.assertTrue(np.array_equal(V.sym_t(self.torch.from_numpy(a), s).numpy(), L.sym(a, s)))

    def test_ema(self):
        net, _h = V.build(V.spec_of("V1"))
        ema = V.Ema(net, decay=0.5)
        name, p = next(iter(net.named_parameters()))
        before = ema.params[name].clone()
        with self.torch.no_grad():
            p.add_(1.0)
        ema.update(net)
        self.assertTrue(self.torch.allclose(ema.params[name], before + 0.5))
        sd = ema.state_dict()
        net2, _h = V.build(V.spec_of("V1"))
        net2.load_state_dict(sd)

    def test_loss_backward_all_variants(self):
        import p42_train as TR
        codes, off, white, order, t_off = synth_cache()
        for v in sorted(V.VARIANTS):
            spec = V.spec_of(v)
            if spec["sampler"] == "episode":
                ds = V.EpisodeBatches("x", off, white, order, t_off, 0, 3, 12)
            else:
                ds = V.GameBatches("x", off, white, order, t_off, 0, 3, 6)
            ds.codes = codes
            b = {k: self.torch.from_numpy(x) for k, x in ds.batch(0).items()}
            net, head = V.build(spec, 6)
            loss = TR.loss_of(b, net, head, spec, False, 100)
            loss.backward()
            self.assertTrue(np.isfinite(float(loss.detach())))

    def test_embed_tta(self):
        import p42_embed as EM
        torch = self.torch
        net, _h = V.build(V.spec_of("V2"))
        net.eval()
        x, seg, t = self.batch(2, 3)
        with torch.no_grad():
            v1, c = EM.game_vectors(net, V.spec_of("V2"), x, seg, t, 3, 1, None, torch.device("cpu"), False)
            v8, _c = EM.game_vectors(net, V.spec_of("V2"), x, seg, t, 3, 8, None, torch.device("cpu"), False)
            ref = sum(L.segment_mean(net.trunk(V.sym_t(x, s).float()), seg, 3) for s in range(8)) / 8
        self.assertEqual(c.tolist(), [3, 3, 0])
        self.assertTrue(torch.allclose(v8[:2], ref[:2], atol=1e-5))
        self.assertEqual(float(v8[2].abs().sum()), 0.0)
        self.assertTrue(torch.allclose(v1[:2], L.segment_mean(net.trunk(x.float()), seg, 3)[:2], atol=1e-5))
        for var in ("V3", "V3b"):
            net3, _h = V.build(V.spec_of(var))
            net3.eval()
            with torch.no_grad():
                for mode in ("full", "chunk24"):
                    v, _c = EM.game_vectors(net3, V.spec_of(var), x, seg, t, 3, 1, mode, torch.device("cpu"), False)
                    self.assertEqual(float(v[2].abs().sum()), 0.0)
                    self.assertTrue(bool((v[:2].abs().sum(1) > 0).all()))
                    if var == "V3b" and mode == "full":   # zero readout at init: full == mean pooling
                        self.assertTrue(torch.allclose(v, L.segment_mean(net3.trunk(x.float()), seg, 3), atol=1e-6))

    def test_chunks(self):
        seg = np.array([0] * 50 + [1] * 10 + [2] * 24)
        cid, owner, n = V.chunk_segments(seg, 4)
        self.assertEqual(n, 2 + 1 + 1 + 1)
        self.assertEqual(owner.tolist(), [0, 0, 1, 2, 3])
        self.assertEqual(np.bincount(cid[:50]).tolist(), [25, 25])
        self.assertTrue((cid[50:60] == 2).all() and (cid[60:] == 3).all())


class Eval(unittest.TestCase):
    def test_quick_grid_and_rule(self):
        import p42_eval as EV
        sys.path.insert(0, V.P38)
        import test_p38
        pools, embs = [], []
        for s in range(3):
            p, e = test_p38.synth(s, nc=14, games=6, d=40)
            pools.append(p)
            embs.append(e)
        rows = EV.quick_grid(pools[:1], embs[0])
        self.assertEqual(len(rows), len(EV.QUICK_WE) * len(EV.QUICK_K))
        b = EV.best_row(rows)
        self.assertEqual(b["mean"], max(r["mean"] for r in rows))
        self.assertEqual(rows.index(b), [r["mean"] for r in rows].index(b["mean"]))
        r = EV.judge_numbers([0.5] * 20, [0.48] * 18 + [0.6] * 2)
        self.assertFalse(r["adopt"])
        r = EV.judge_numbers([0.5] * 20, [0.48] * 19 + [0.51])
        self.assertTrue(r["adopt"])
        r = EV.judge_numbers([0.5] * 20, [0.495] * 20)
        self.assertFalse(r["adopt"])

    def test_candidate_order(self):
        import p42_eval as EV
        q = {"rows": {k: {} for k in ("C0", "baseabcd1234_step_40000_raw_tta8", "V1_step_40000_raw_tta1",
                                      "V1_step_40000_ema_tta1", "V3_step_40000_raw_tta1_full")}}
        trs = {"V1": {"steps": 40000}, "V3": {"steps": 40000}}
        order, missing = EV.candidate_order(q, trs, "baseabcd1234")
        self.assertEqual(order[:3], ["baseabcd1234_step_40000_raw_tta8", "V1_step_40000_raw_tta1",
                                     "V1_step_40000_ema_tta1"])
        self.assertEqual(sorted(missing), ["V3_step_40000_ema_tta1_chunk24", "V3_step_40000_ema_tta1_full",
                                           "V3_step_40000_raw_tta1_chunk24"])
        q["rows"]["V3b_step_40000_raw_tta1_full"] = {}
        trs["V3b"] = {"steps": 40000}
        order, missing = EV.candidate_order(q, trs, "baseabcd1234")
        self.assertIn("V3b_step_40000_raw_tta1_full", order)
        self.assertIn("V3b_step_40000_ema_tta1_chunk24", missing)
        self.assertLess(order.index("V3_step_40000_raw_tta1_full"), order.index("V3b_step_40000_raw_tta1_full"))


if __name__ == "__main__":
    unittest.main()
