"""CPU tests of the prereg12 fine-tuning pilot (no GPU, no data/local, no validation features).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg12/explore/finetune/test_ft.py
    C10_FT_REAL=1 ...   also loads the real HumanSL checkpoint on the CPU (about 1-2 minutes)
The pipeline smoke root data/prereg12/smoke/pipe (made by the commands in PLAN.md 6) is used when present.
"""

import os
import sys
import unittest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import ft_lib as F  # noqa: E402

SMOKE = os.path.join(F.DATA, "smoke", "pipe")
REAL = os.environ.get("C10_FT_REAL") == "1"


def bank_games(n):
    import p1_extract as E
    return E.set_games(E.load_sets(F.P1_SETS), "bank")[:n]


class Featurise(unittest.TestCase):
    def test_turns_equal_p1(self):
        for g in bank_games(2):
            full = F.P.featurize_game(g["sgf_content"], g["color"], sides=("target", "opponent"))
            mine = F.featurize_turns(g["sgf_content"], g["color"], list(range(full["n_moves"])))
            for k in ("bin", "glob", "legal", "move", "pla", "side", "turn"):
                self.assertTrue(np.array_equal(full[k], mine[k]), k)
            pick = F.select_turns(g["sgf_content"], g["color"], 16, [1, 2, 3])
            sub = F.featurize_turns(g["sgf_content"], g["color"], pick)
            rows = np.searchsorted(full["turn"], np.array(pick))
            for k in ("bin", "glob", "move", "pla", "side", "turn"):
                self.assertTrue(np.array_equal(full[k][rows], sub[k]), k)
            self.assertEqual(int((sub["side"] == 0).sum()), 16)
            self.assertEqual(int((sub["side"] == 1).sum()), 16)

    def test_pack_roundtrip(self):
        rng = np.random.default_rng(0)
        b = (rng.random((5, F.N_BIN, F.POS_LEN, F.POS_LEN)) < 0.3).astype(np.uint8)
        bits = F.pack_bin(b)
        self.assertEqual(bits.shape, (5, F.N_BYTES))
        self.assertTrue(np.array_equal(F.unpack_bin_np(bits), b))
        torch = F.torch_mod()
        self.assertTrue(np.array_equal(F.unpack_bits_torch(torch.as_tensor(bits)).numpy().astype(np.uint8), b))


class Schedule(unittest.TestCase):
    def test_head_only_phase(self):
        import ft_pilot as FP
        cfg = F.CONFIG
        self.assertEqual(FP.schedule(0, 10000, cfg)[1], 0.0)
        self.assertEqual(FP.schedule(cfg["head_only_steps"] - 1, 10000, cfg)[1], 0.0)
        self.assertGreater(FP.schedule(cfg["head_only_steps"], 10000, cfg)[1], 0.0)
        h, t = FP.schedule(9999, 10000, cfg)
        self.assertAlmostEqual(h, cfg["cosine_floor"], places=3)


@unittest.skipUnless(os.path.isdir(os.path.join(SMOKE, "cache", "bank")), "pipeline smoke root missing")
class SmokeCache(unittest.TestCase):
    def test_bank_counts_equal_p1_chunks(self):
        import ft_pilot as FP
        files = FP.cache_files(FP.Paths(SMOKE), "bank")
        with np.load(os.path.join(F.REPO, "data", "prereg08", "p1", "bank", "chunk_00000.npz")) as z:
            p1 = {int(r): (c.tolist(), int(n)) for r, c, n in zip(z["ref"], z["counts"], z["n_moves"])}
        n = 0
        for p in files:
            z = F.load_cache_chunk(p)
            for i, r in enumerate(z["ref"]):
                if int(r) in p1:
                    self.assertEqual((z["counts"][i].tolist(), int(z["n_moves"][i])), p1[int(r)])
                    n += 1
        self.assertGreater(n, 0)

    def test_ft_chunks_match_cache(self):
        d = os.path.join(SMOKE, "ft", "bank")
        if not os.path.isdir(d):
            self.skipTest("no ft smoke output")
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz") or ".tmp" in f:
                continue
            with np.load(os.path.join(d, f)) as z:
                c = F.load_cache_chunk(os.path.join(SMOKE, "cache", "bank", f))
                self.assertTrue(np.array_equal(z["ref"], c["ref"]))
                self.assertTrue(np.array_equal(z["counts"], c["counts"]))
                self.assertTrue(np.allclose(np.exp(z["lp"]).sum(1), 1.0))


@unittest.skipUnless(REAL, "set C10_FT_REAL=1 to load the real checkpoint on the CPU")
class RealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch = F.torch_mod()
        torch.set_num_threads(8)
        cls.net, cls.info = F.build_net("cpu", tiny=False)
        g = bank_games(1)[0]
        f = F.P.featurize_game(g["sgf_content"], g["color"], sides=("target", "opponent"))
        cls.f = {k: (v[:6] if hasattr(v, "__len__") and not isinstance(v, int) else v) for k, v in f.items()}

    def test_trunk_equals_p1_probe_features(self):
        torch = F.torch_mod()
        f = self.f
        sc = F.P.ProbeScorer(self.net.base, device="cpu", precision="fp32", batch=6)
        try:
            ref = sc.position_features(f)[:, :768]
        finally:
            sc.close()
        self.net.eval()
        with torch.no_grad():
            bits = torch.as_tensor(F.pack_bin(f["bin"]))
            tr = self.net.trunk(F.unpack_bits_torch(bits), torch.as_tensor(f["glob"]),
                                torch.as_tensor(f["pla"].astype(np.int64)))
            flat = tr.float().reshape(6, F.C_TRUNK, -1)
            mv = torch.as_tensor(f["move"].astype(np.int64))
            at = flat.gather(2, mv.view(6, 1, 1).expand(6, F.C_TRUNK, 1)).squeeze(2)
            mine = torch.cat([at, flat.mean(2)], 1).numpy()
        self.assertEqual(self.info["n_blocks"], 18)
        np.testing.assert_allclose(mine, ref, rtol=0, atol=1e-5)

    def test_gradients_only_in_last_blocks_and_head(self):
        torch = F.torch_mod()
        net = self.net
        net.train()
        f = self.f
        d = {"bits": torch.as_tensor(F.pack_bin(f["bin"])), "glob": torch.as_tensor(f["glob"]),
             "move": torch.as_tensor(f["move"].astype(np.int64)), "pla": torch.as_tensor(f["pla"].astype(np.int64)),
             "seg": torch.as_tensor(f["side"].astype(np.int64)), "y": torch.tensor([3]),
             "colour": torch.tensor([0]), "n_games": 1}
        import ft_pilot as FP
        before = {n: p.detach().clone() for n, p in net.named_parameters()}
        opt = FP.make_optimizer(net, F.CONFIG)
        loss = torch.nn.functional.cross_entropy(FP.forward_batch(net, d, "cpu", "fp32"), d["y"])
        opt.zero_grad()
        loss.backward()
        n = len(net.base.blocks)
        k = F.CONFIG["k_trainable_blocks"]
        allowed = tuple(f"base.blocks.{i}." for i in range(n - k, n)) + ("base.norm_trunkfinal.", "head.")
        for name, p in net.named_parameters():
            if name.startswith(allowed):
                self.assertTrue(p.requires_grad, name)
            else:
                self.assertFalse(p.requires_grad, name)
                self.assertIsNone(p.grad, name)
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for nm, p in net.named_parameters() if nm.startswith(f"base.blocks.{n - 1}.")))
        opt.step()
        for name, p in net.named_parameters():
            if not name.startswith(allowed):
                self.assertTrue(torch.equal(p.detach(), before[name]), name)
        self.assertFalse(net.base.training)


if __name__ == "__main__":
    unittest.main()
