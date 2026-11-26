"""prereg60: small CPU tests (synthetic games and random weights only; no score of the method is computed).

    GPUPY -m pytest -q results/prereg60/explore/test_p60.py      # planes, samplers, widened network (needs torch)
    PY    -m pytest -q results/prereg60/explore/test_p60.py      # planes, samplers, p60_eval helpers (torch tests skip)
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
import p60_lib as P  # noqa: E402

L, V, EM = P.L, P.V, P.EM

try:
    import torch
except Exception:                                   # the CPU environment has no torch
    torch = None
try:
    import p60_eval as PE
except Exception:                                   # the GPU environment may lack the evaluation stack
    PE = None


def random_game(rng, n_moves):
    """Alternating colours on distinct points: never an occupied point, whatever is captured."""
    pts = rng.permutation(361)[:n_moves]
    col = np.arange(n_moves) % 2
    return (col * 361 + pts).astype(np.uint16)


class Planes(unittest.TestCase):
    def test_action_plane_marks_exactly_the_move(self):
        rng = np.random.default_rng(0)
        for colour in ("b", "w"):
            codes = random_game(rng, 120)
            x, t = P.planes_for_codes60(codes, colour)
            self.assertEqual(x.shape, (60, 18, 19, 19))
            self.assertTrue((x[:, P.ACTION].reshape(len(t), -1).sum(1) == 1).all())
            for i, m in enumerate(t.tolist()):
                p = int(codes[m]) % 361
                self.assertEqual(int(x[i, P.ACTION, p // 19, p % 19]), 1)
                self.assertEqual(int(codes[m]) // 361, 0 if colour == "b" else 1)      # it is the target's move
            # the marked point is empty in the position the target plays in (planes 7 and 15 = the current board)
            self.assertFalse((x[:, P.ACTION] & (x[:, 7] | x[:, 15])).any())

    def test_action_plane_sgf_coordinates(self):
        # B[pd]: column p = 15, SGF row d = 3 from the top -> board row 18 - 3 = 15 (row 0 = bottom line)
        codes = L.moves_of("(;SZ[19];B[pd];W[dp];B[pp])")
        x, t = P.planes_for_codes60(codes, "b")
        self.assertEqual(t.tolist(), [0, 2])
        self.assertEqual(np.argwhere(x[0, P.ACTION]).tolist(), [[15, 15]])
        self.assertEqual(np.argwhere(x[1, P.ACTION]).tolist(), [[3, 15]])
        self.assertEqual(int(x[0, :16].sum()), 0)                                    # nothing on the board yet
        self.assertEqual(np.argwhere(x[1, 7]).tolist(), [[15, 15]])                  # own stone, current position
        self.assertEqual(np.argwhere(x[1, 15]).tolist(), [[3, 3]])                   # the opponent's
        self.assertTrue((x[:, 16] == 1).all())
        xw, tw = P.planes_for_codes60(codes, "w")
        self.assertEqual(tw.tolist(), [1])
        self.assertEqual(np.argwhere(xw[0, P.ACTION]).tolist(), [[3, 3]])
        self.assertTrue((xw[:, 16] == 0).all())

    def test_old_planes_identical_where_history_is_full(self):
        rng = np.random.default_rng(1)
        for colour in ("b", "w"):
            codes = random_game(rng, 150)
            old, t_old = L.planes_for_codes(codes, colour)
            new, t_new = P.planes_for_codes60(codes, colour)
            full = t_new >= L.HIST - 1
            self.assertTrue(np.array_equal(t_new[full], t_old))
            self.assertTrue(np.array_equal(new[full][:, :17], old))
            self.assertEqual(int((~full).sum()), 4 if colour == "b" else 3)
            # early=False is the old sample set; action=False leaves plane 17 empty
            x, t = P.planes_for_codes60(codes, colour, early=False, action=False)
            self.assertTrue(np.array_equal(t, t_old) and np.array_equal(x[:, :17], old))
            self.assertEqual(int(x[:, P.ACTION].sum()), 0)

    def test_early_moves_padded_with_empty_boards(self):
        rng = np.random.default_rng(2)
        codes = random_game(rng, 60)
        S = L.replay(codes)
        for colour, v in (("b", 1), ("w", -1)):
            x, t = P.planes_for_codes60(codes, colour)
            for i, m in enumerate(t.tolist()):
                if m >= L.HIST - 1:
                    continue
                missing = L.HIST - 1 - m                 # slots before the start of the game
                self.assertEqual(int(x[i, :missing].sum()) + int(x[i, 8:8 + missing].sum()), 0)
                for j in range(missing, L.HIST):         # the slots that exist show the real positions
                    pos = S[m - (L.HIST - 1) + j].reshape(19, 19)
                    self.assertTrue(np.array_equal(x[i, j], (pos == v).astype(np.uint8)))
                    self.assertTrue(np.array_equal(x[i, 8 + j], (pos == -v).astype(np.uint8)))
        x, t = P.planes_for_codes60(codes, "b")
        self.assertEqual(int(t[0]), 0)
        self.assertEqual(int(x[0, :16].sum()), 0)        # the first move of the game: eight empty boards

    def test_subset_and_count(self):
        rng = np.random.default_rng(3)
        codes = random_game(rng, 100)
        self.assertEqual(len(P.target_moves60(codes, "b")), 50)                       # every move of the target
        self.assertEqual(len(P.target_moves60(codes, "b", early=False)), 46)
        x, t = P.planes_for_codes60(codes, "b", 24, np.random.default_rng(5))
        self.assertEqual(len(t), 24)
        self.assertTrue((np.diff(t) > 0).all())
        full, t_all = P.planes_for_codes60(codes, "b")
        self.assertTrue(np.array_equal(x, full[np.searchsorted(t_all, t)]))
        short = random_game(rng, 5)                       # prereg36: no sample; now 3 (black) / 2 (white)
        self.assertEqual(len(L.planes_for_codes(short, "b")[0]), 0)
        self.assertEqual(len(P.planes_for_codes60(short, "b")[0]), 3)
        self.assertEqual(len(P.planes_for_codes60(short, "w")[0]), 2)
        self.assertEqual(P.planes_for_codes60(np.zeros(0, np.uint16), "b")[0].shape, (0, 18, 19, 19))

    def test_illegal_move_raises_as_before(self):
        codes = np.array([5, 361 + 6, 7, 361 + 5, 9, 361 + 10], np.uint16)            # move 3 on an occupied point
        with self.assertRaises(ValueError):
            P.planes_for_codes60(codes, "b")


class World:
    def __init__(self, d, seed=0):
        rng = np.random.default_rng(seed)
        games = [random_game(rng, 40 + int(rng.integers(40))) for _ in range(12)]
        games[5] = random_game(rng, 5)
        self.games = games
        self.off = np.concatenate([[0], np.cumsum([len(g) for g in games])]).astype(np.int64)
        self.all = np.concatenate(games)
        self.dir = d
        self.is_white = rng.integers(0, 2, len(games)).astype(bool)
        self.t_games = np.arange(12, dtype=np.int64)
        self.t_off = np.array([0, 4, 8, 12], np.int64)


class Samplers(unittest.TestCase):
    def test_batches_without_early_equal_the_old_sampler(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            old = V.GameBatches(w.dir, w.off, w.is_white, w.t_games, w.t_off, 0, batch_games=6, moves=6)
            new = P.GameBatches60(w.dir, w.off, w.is_white, w.t_games, w.t_off, 0, batch_games=6, moves=6,
                                  early=False, action=False)
            old.codes = new.codes = w.all
            for s in (0, 1, 320000):
                a, b = old.batch(s), new.batch(s)
                self.assertTrue(all(np.array_equal(a[k], b[k]) for k in ("seg", "y", "t")))
                self.assertTrue(np.array_equal(a["x"], b["x"][:, :17]))
                self.assertEqual(int(b["x"][:, 17].sum()), 0)

    def test_batches_with_early_and_action(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            ds = P.GameBatches60(w.dir, w.off, w.is_white, w.t_games, w.t_off, 0, batch_games=8, moves=6)
            ds.codes = w.all
            b = ds.batch(7)
            self.assertEqual(b["x"].shape[1:], (18, 19, 19))
            self.assertEqual(len(b["y"]), 8)
            self.assertTrue((b["x"][:, 17].reshape(len(b["seg"]), -1).sum(1) == 1).all())
            self.assertTrue(np.array_equal(b["x"], ds.batch(7)["x"]))                 # deterministic in s
            early = sum(int((ds.batch(s)["t"] < L.HIST - 1).sum()) for s in range(20))
            self.assertGreater(early, 0)                                              # early moves are drawn

    def test_game_set_counts(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            new = P.GameSet60(w.all, w.off[:-1], w.off[1:], w.is_white)
            old = EM.GameSet(w.all, w.off[:-1], w.off[1:], w.is_white)
            same = P.GameSet60(w.all, w.off[:-1], w.off[1:], w.is_white, early=False, action=False)
            if torch is None:
                self.skipTest("torch not installed")
            _i, x, seg, t = new[0]
            _i, xo, sego, to = old[0]
            _i, xs, segs, tso = same[0]
            self.assertTrue(torch.equal(xs[:, :17], xo) and torch.equal(segs, sego) and torch.equal(tso, to))
            cn = np.bincount(seg.numpy(), minlength=12)
            co = np.bincount(sego.numpy(), minlength=12)
            want = [len(P.target_moves60(g, "w" if wh else "b")) for g, wh in zip(w.games, w.is_white)]
            self.assertEqual(cn.tolist(), want)
            self.assertEqual(co[5], 0)                      # the 5-move game had no sample
            self.assertGreater(cn[5], 0)
            d_ = cn - co
            self.assertTrue((d_ >= 0).all() and (d_ <= 7).all())


@unittest.skipIf(torch is None, "torch not installed")
class Net(unittest.TestCase):
    def test_widened_net_equals_the_old_net(self):
        torch.manual_seed(0)
        spec = V.spec_of("V2")
        net, _h = V.build(spec)
        with torch.no_grad():                               # non-trivial BN statistics
            for m in net.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.running_mean.normal_(0, 0.1)
                    m.running_var.uniform_(0.5, 1.5)
        net.eval()
        rng = np.random.default_rng(0)
        codes = random_game(rng, 80)
        old, _t = L.planes_for_codes(codes, "b")
        new, _t2 = P.planes_for_codes60(codes, "b", early=False)                     # same samples + real plane 17
        self.assertGreater(int(new[:, 17].sum()), 0)
        xo, xn = torch.from_numpy(old[:12]).float(), torch.from_numpy(new[:12]).float()
        with torch.no_grad():
            a = net.trunk(xo)
            sd = {k: v.clone() for k, v in net.state_dict().items()}
            P.widen(net)
            self.assertEqual(tuple(net.trunk.conv_in.weight.shape), (L.CHANNELS, 18, 3, 3))
            self.assertEqual(float(net.trunk.conv_in.weight[:, 17].abs().max()), 0.0)
            b = net.trunk(xn)
            self.assertLess(float((a - b).abs().max()), 1e-4 * max(1.0, float(a.abs().max())))
            # a non-zero new channel changes the output (the plane is really fed)
            net.trunk.conv_in.weight[:, 17].fill_(0.5)
            self.assertGreater(float((net.trunk(xn) - a).abs().max()), 1e-3)
        # build_wide + widen_state: the state-dict route gives the same network
        wide, _h2 = P.build_wide(spec)
        wide.load_state_dict(P.widen_state(sd))
        wide.eval()
        with torch.no_grad():
            self.assertLess(float((wide.trunk(xn) - a).abs().max()), 1e-4 * max(1.0, float(a.abs().max())))
        with self.assertRaises(ValueError):
            P.widen(wide)


class Schedule(unittest.TestCase):
    def test_lr_factor(self):
        f = [P.lr_factor(s, 500, 30000) for s in range(30000)]
        self.assertAlmostEqual(f[0], 1 / 500.0)
        self.assertAlmostEqual(f[499], 1.0)
        self.assertAlmostEqual(f[500], 1.0)
        self.assertTrue(all(x < y for x, y in zip(f[:499], f[1:500])))
        self.assertTrue(all(x > y for x, y in zip(f[500:-1], f[501:])))
        self.assertAlmostEqual(f[500 + 14750], 0.5)
        self.assertLess(f[-1], 1e-8)
        self.assertGreater(f[-1], 0.0)
        self.assertAlmostEqual(P.lr_factor(30000, 500, 30000), 0.0)

    def test_resume_by_step_and_prune(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(P.last_ckpt(d))
            for name in ("step_009000.pt", "step_010000.pt", "step_99000.pt", "step_100000.pt", "step_020000.pt"):
                open(os.path.join(d, name), "w").close()
            # file-name order would put step_99000 last; the step number decides
            self.assertEqual(os.path.basename(P.last_ckpt(d)), "step_100000.pt")
            self.assertEqual(sorted(os.listdir(d))[-1], "step_99000.pt")
            gone = P.prune(d, (10000, 20000, 30000), 100000)
            self.assertEqual(gone, ["step_009000.pt", "step_99000.pt"])
            self.assertEqual(sorted(os.listdir(d)), ["step_010000.pt", "step_020000.pt", "step_100000.pt"])
            self.assertEqual(P.step_of(os.path.join(d, "step_020000.pt")), 20000)

    def test_run_constants(self):
        if torch is None:
            self.skipTest("p60_run needs the GPU environment's imports only at run time; constants checked there")
        import p60_run as R
        self.assertEqual((R.TOTAL_STEPS, R.LR, R.WARMUP, R.KEEP, R.SEED, R.VARIANT),
                         (30000, 5e-4, 500, (10000, 20000, 30000), 0, "V2"))
        self.assertEqual((V.WD, V.EMA_DECAY, L.BATCH_GAMES, L.SEG_MOVES), (1e-4, 0.999, 96, 24))
        self.assertEqual(os.path.basename(R.ckpt_file("x", 30000)), "step_030000.pt")
        self.assertEqual(R.emb_tag(), "p60_step_030000_raw_tta1")


@unittest.skipIf(PE is None, "evaluation stack not importable in this environment")
class Eval(unittest.TestCase):
    def test_paths_and_rule(self):
        self.assertEqual(PE.TAG, "p60_step_030000_raw_tta1")
        self.assertTrue(PE.CKPT.replace("\\", "/").endswith("data/prereg60/S0/ckpt/step_030000.pt"))
        self.assertEqual(PE.V27_E, 0.7035676497986656)
        self.assertEqual((PE.E43.ADOPT_MEAN, PE.E43.BOOT, PE.E43.BOOT_SEED), (0.01, 10000, 43))
        j = PE.E43.judge_rule([0.72] * 18 + [0.70] * 2, [0.705] * 20)
        self.assertTrue(j["adopt"])
        j = PE.E43.judge_rule([0.72] * 17 + [0.70] * 3, [0.705] * 20)
        self.assertFalse(j["adopt"])

    def test_e_uses(self):
        with tempfile.TemporaryDirectory() as d:
            def put(pre, name, obj):
                os.makedirs(os.path.join(d, "results", pre, "explore"), exist_ok=True)
                json.dump(obj, open(os.path.join(d, "results", pre, "explore", name), "w"))
            for pre in ("prereg36", "prereg43", "prereg49"):
                put(pre, "eval.json", {"e_pools_use_number": 1})
            self.assertEqual(PE.e_uses_so_far(d)["count"], 3)
            put("prereg51", "eval_B.json", {"e_pools_use_number": 7})
            put("prereg57", "eval.json", {"e_pools_use_number": 8})
            put("prereg58", "eval.json", {"adopt": False})                 # competition 2: not an E judgement
            put("prereg51", "seam.json", {"e_pools_use_number": 0})        # not an eval file
            put("prereg60", "eval.json", {"e_pools_use_number": 9})        # its own file never counts
            u = PE.e_uses_so_far(d)
            self.assertEqual(u["later"], ["prereg51/eval_B.json", "prereg57/eval.json"])
            self.assertEqual(u["count"], 5)

    def test_e_uses_on_the_repo_matches_prereg57s_rule(self):
        u = PE.e_uses_so_far()
        p51 = len(PE.P51.existing_evals())
        p54 = len(PE.glob.glob(os.path.join(PE.REPO, "results", "prereg54", "explore", "eval*.json")))
        p57 = len(PE.glob.glob(os.path.join(PE.REPO, "results", "prereg57", "explore", "eval*.json")))
        self.assertEqual(u["count"], len(PE.P49.e_judgements_so_far()) + p51 + p54 + p57)

    def _npz(self, path, sha, count, rows, tag=None, step=30000, cfg=None):
        n = len(count)
        cfg = {"tta": 1, "weights": "raw", "games": n, "moves": 10 * n, "ckpt_sha256": sha} if cfg is None else cfg
        np.savez(path, emb=np.zeros((n, 4), np.float32), count=np.asarray(count, np.int32),
                 store_row=np.asarray(rows), players_sha256_sorted_ids=np.array("p"), ckpt_sha256=np.array(sha),
                 ckpt=np.array(tag or PE.TAG), ckpt_step=np.array(step), embed_cfg=np.array(json.dumps(cfg)))

    def test_check_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            ref, new = os.path.join(d, "ref.npz"), os.path.join(d, "new.npz")
            self._npz(ref, "c" * 64, [0, 50, 60, 0], [3, 4, 5, 6], tag="V2_step_320000_raw_tta1", step=320000)
            self._npz(new, "a" * 64, [3, 54, 63, 0], [3, 4, 5, 6])
            r = PE.check_vectors(new, ref, "a" * 64)
            self.assertEqual((r["more_samples_total"], r["games_with_vector"], r["games_with_vector_c49"]),
                             (10, 3, 2))
            for kw, sha in (({"count": [3, 54, 59, 0]}, "a" * 64),            # fewer samples than c49
                            ({"count": [8, 54, 63, 0]}, "a" * 64),            # more than 7 early moves
                            ({"rows": [3, 4, 5, 7]}, "a" * 64),               # other games
                            ({"tag": "p60_noearly_step_030000_raw_tta1"}, "a" * 64),
                            ({"step": 20000}, "a" * 64),
                            ({}, "b" * 64)):                                  # another checkpoint
                args = {"count": [3, 54, 63, 0], "rows": [3, 4, 5, 6]}
                args.update(kw)
                self._npz(new, "a" * 64, args["count"], args["rows"], args.get("tag"), args.get("step", 30000))
                with self.assertRaises(SystemExit):
                    PE.check_vectors(new, ref, sha)

    def test_assemble_check(self):
        head = b"question_id,top1,top2,top3,top4,top5"
        old = b"\n".join([head] + [b"pub_%d,a" % i for i in range(400)] + [b"priv_%d,a" % i for i in range(400)] +
                         [b"pub2_%d,x" % i for i in range(400)] + [b"priv2_%d,x" % i for i in range(400)]) + b"\n"
        new = old.replace(b"pub2_7,x", b"pub2_7,y")
        self.assertTrue(PE.assemble_check(new, old)["ok"])
        self.assertFalse(PE.assemble_check(old.replace(b"pub_7,a", b"pub_7,b"), old)["ok"])
        self.assertFalse(PE.assemble_check(old.replace(b"priv2_399,x\n", b""), old)["ok"])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.csv")
            open(p, "wb").write(old.replace(b"\n", b"\r\n"))
            data, crlf = PE.lf_bytes(p)
            self.assertTrue(crlf and data == old)


if __name__ == "__main__":
    unittest.main()
