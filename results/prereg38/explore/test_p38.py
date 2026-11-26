"""prereg38 explore: tests of p38_backend.py on synthetic pools (no feature store, no pools of the repo).

    PY -m pytest -q results/prereg38/explore/test_p38.py
"""
import copy
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p38_backend as B  # noqa: E402

pe, st, ps = B.pe, B.st, B.ps


class NoAnswers(dict):
    """A pool that refuses to hand out the answers (or the stored k)."""

    def __getitem__(self, k):
        if k in ("owner", "k"):
            raise AssertionError("the backend read pool[%r]" % k)
        return dict.__getitem__(self, k)

    def get(self, k, d=None):
        if k in ("owner", "k"):
            raise AssertionError("the backend read pool[%r]" % k)
        return dict.get(self, k, d)


def synth(seed=0, nc=14, games=6, d=12, ks=(1, 2, 3, 5, 1, 2, 3, 5, 2, 3)):
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=(nc, d))
    shift = rng.normal(size=d) * 0.7                          # a colour offset
    emb, colour, cand_rows, q_rows, owner = [], [], [], [], []

    def game(o, white):
        emb.append(centre[o] + rng.normal(size=d) * 0.8 + (shift if white else 0.0))
        colour.append("W" if white else "B")
        return len(emb) - 1

    for c in range(nc):
        whites = rng.random(games) < 0.5
        if c == 0:
            whites[:] = False                                 # a candidate without white games
        cand_rows.append([game(c, bool(w)) for w in whites])
    for k in ks:
        o = int(rng.integers(nc))
        owner.append(o)
        q_rows.append([game(o, bool(rng.random() < 0.5)) for _ in range(k)])
    emb = np.asarray(emb)
    nq = len(ks)
    T = {n: (rng.normal(size=(nq, nc)), np.ones(nq, bool)) for n in ps.TERMS}
    e = pe.Emb(emb, np.full(len(emb), 30), None, np.asarray(colour), "sha", "name")
    pool = {"T": T, "cand_rows": cand_rows, "q_rows": q_rows, "owner": np.asarray(owner), "k": np.asarray(ks)}
    return pool, e


CFGS = {
    "base": {},
    "centre": {"centre_colour": True},
    "colour": {"sim": "colour", "alpha": 0.5},
    "gnorm_count": {"gnorm": True, "pool_w": "count"},
    "pair_top2": {"sim": "pair", "pair": {"agg": "top", "m": 2}},
    "pair_lse": {"sim": "pair", "pair": {"agg": "lse", "tau": 0.1}},
    "whiten": {"whiten": {"lam": 0.3, "lda": None}},
    "whiten_lda": {"gnorm": True, "whiten": {"lam": 0.5, "lda": 4}},
    "asnorm": {"norm": ["asnorm", 5]},
    "dual": {"norm": ["dual", 1]},
    "none": {"norm": ["none"]},
    "aqe": {"aqe": {"n": 1, "beta": 0.5}},
    "qsmooth": {"qsmooth": {"n": 2, "beta": 0.5}},
    "krecip": {"sim": "krecip", "krecip": {"k1": 4, "k2": 2, "lam": 0.3}},
    "perk": {"wE": {"1": 2, "2": 4, "3": 8, "5": 16}, "rowz": True,
             "gmul": {k: {"O": 0.5, "L": 1, "JO": 2, "PA": 0} for k in ("1", "2", "3", "5")}},
}


def cfg_of(name):
    return B.with_(B.BASE, norm=["csls", 3], **CFGS[name]) if "norm" not in CFGS[name] else B.with_(B.BASE, **CFGS[name])


class Seam(unittest.TestCase):
    def test_default_is_m36_bit_for_bit(self):
        pool, e = synth()
        w, _k = pe.g_combo()
        want = st.csls(pe.fused(pool, pe.e36(e, pool), w, 4), 20)
        self.assertTrue(np.array_equal(B.backend_scores(pool, e, B.BASE), want))
        self.assertTrue(np.array_equal(B.backend_scores(pool, e, B.BASE, cache={}), want))

    def test_general_prototype_path_matches_e36(self):
        pool, e = synth(1)
        G = B.pool_games(e, pool)
        Xc, Xq = B.transform(G, B.BASE)
        Q, hq, C, _hc = B._protos(G, Xc, Xq, B.BASE)
        M, ok = pe.e36(e, pool)
        self.assertTrue(np.allclose(Q @ C.T, M, atol=1e-12))
        self.assertTrue(np.array_equal(hq, ok))

    def test_per_k_fusion_with_equal_weights_is_the_scalar_fusion(self):
        pool, e = synth(2)
        E = pe.e36(e, pool)
        a = B.fuse_cfg(pool, E, B.with_(B.BASE, wE=4))
        b = B.fuse_cfg(pool, E, B.with_(B.BASE, wE={k: 4 for k in ("1", "2", "3", "5")}))
        self.assertTrue(np.allclose(a, b, atol=1e-12))


class Symmetry(unittest.TestCase):
    def test_candidate_and_question_order(self):
        for name in CFGS:
            pool, e = synth(3)
            cfg = cfg_of(name)
            S = B.backend_scores(pool, e, cfg)
            rng = np.random.default_rng(5)
            pc, pq = rng.permutation(len(pool["cand_rows"])), rng.permutation(len(pool["q_rows"]))
            p2 = {"T": {n: (M[pq][:, pc], ok[pq]) for n, (M, ok) in pool["T"].items()},
                  "cand_rows": [pool["cand_rows"][i] for i in pc], "q_rows": [pool["q_rows"][i] for i in pq]}
            S2 = B.backend_scores(p2, e, cfg)
            self.assertEqual(S2.shape, S.shape, name)
            self.assertTrue(np.allclose(S2, S[pq][:, pc], atol=1e-8), name)


class NoLeak(unittest.TestCase):
    def test_backend_never_reads_the_answers(self):
        for name in CFGS:
            pool, e = synth(4)
            cfg = cfg_of(name)
            S = B.backend_scores(pool, e, cfg)
            S2 = B.backend_scores(NoAnswers(pool), e, cfg, cache={})
            self.assertTrue(np.array_equal(S, S2), name)
            p3 = copy.deepcopy(pool)
            p3["owner"] = (pool["owner"] + 1) % len(pool["cand_rows"])
            self.assertTrue(np.array_equal(B.backend_scores(p3, e, cfg), S), name)

    def test_oracle_with_distinct_owners_is_the_plain_backend(self):
        pool, e = synth(6)
        S = B.backend_scores(pool, e, B.BASE)
        So = B.oracle_linked_scores(pool, e, B.BASE, np.arange(len(pool["q_rows"])))
        self.assertTrue(np.allclose(S, So, atol=1e-8))

    def test_oracle_merges_same_owner_rows(self):
        pool, e = synth(7)
        owner = np.zeros(len(pool["q_rows"]), np.int64)
        So = B.oracle_linked_scores(pool, e, B.BASE, owner)
        self.assertTrue(np.allclose(So, So[0][None, :]))


class Pieces(unittest.TestCase):
    def test_whitening_makes_the_within_covariance_identity(self):
        pool, e = synth(8, nc=30, games=10, d=6)
        G = B.pool_games(e, pool)
        mu, P = B.whitening(G["Xc"], G["gc"], 1e-9)
        X = (G["Xc"] - mu) @ P
        means = np.stack([X[G["gc"] == c].mean(0) for c in range(G["nc"])])
        R = X - means[G["gc"]]
        W = R.T @ R / (len(X) - G["nc"])
        self.assertTrue(np.allclose(W, np.eye(6), atol=1e-6))
        _mu, P4 = B.whitening(G["Xc"], G["gc"], 0.3, 4)
        self.assertEqual(P4.shape, (6, 4))

    def test_pair_top_all_is_mean(self):
        pool, e = synth(9)
        G = B.pool_games(e, pool)
        a, _ = B.pair_sim(G, G["Xc"], G["Xq"], {"agg": "mean"})
        b, _ = B.pair_sim(G, G["Xc"], G["Xq"], {"agg": "top", "m": 99})
        self.assertTrue(np.allclose(a, b, atol=1e-12))
        c, _ = B.pair_sim(G, G["Xc"], G["Xq"], {"agg": "lse", "tau": 1e3})
        self.assertTrue(np.allclose(a, c, atol=1e-3))

    def test_games_without_a_sample_are_dropped(self):
        pool, e = synth(10)
        e.count = e.count.copy()
        e.count[pool["q_rows"][0][0]] = 0
        G = B.pool_games(e, pool)
        self.assertEqual(len(G["Xq"]), sum(len(g) for g in pool["q_rows"]) - 1)
        for name in ("colour", "pair_top2", "whiten"):
            self.assertTrue(np.isfinite(B.backend_scores(pool, e, cfg_of(name))).all(), name)

    def test_normalisers_keep_the_shape_and_are_finite(self):
        F = np.random.default_rng(0).normal(size=(9, 13))
        for nm in B.norm_grid(["csls", 20], True):
            S = B.normalise(F, nm)
            self.assertEqual(S.shape, F.shape)
            self.assertTrue(np.isfinite(S).all(), nm)

    def test_per_k_takes_the_nearest_key(self):
        v = B._per_k({"1": 10, "2": 20, "3": 30, "5": 50}, np.array([1, 2, 3, 4, 5, 9]))
        self.assertEqual(v.tolist(), [10, 20, 30, 30, 50, 50])


class MainSessionTools(unittest.TestCase):
    def test_judge_numbers_same_system_does_not_adopt(self):
        pool, e = synth(11)
        r = B.judge_numbers([pool], e, B.BASE, e, B.BASE)
        self.assertEqual(r["means"]["backend"], r["means"]["adopted_system"])
        self.assertEqual(r["backend_minus_adopted"]["mean"], 0.0)
        self.assertFalse(r["adopt"])

    def test_judge_rule(self):
        pool, e = synth(12)
        r = B.judge_numbers([pool], e, cfg_of("whiten"), e, B.base_cfg_of({"M": {"wE": 8, "csls_k": 10}}))
        d = r["backend_minus_adopted"]
        want = d["mean"] >= 0.01 and d["ci95"][0] > 0 and d["pools_higher"] >= 1
        self.assertEqual(r["adopt"], bool(want))
        self.assertIn(">= 1 of 1 pools", r["rule"])

    def test_retune_keeps_the_structure(self):
        pool, e = synth(13, nc=20, games=8)
        bench = B.Bench([pool], e)
        ref = B.with_(B.BASE, gnorm=True, whiten={"lam": 0.02, "lda": 32})
        c, r = B.retune_structure(bench, ["B4", "B2b"], ref)
        self.assertTrue(c["gnorm"])
        self.assertIsNotNone(c["whiten"])
        self.assertIsInstance(c["wE"], dict)
        self.assertIsInstance(c["gmul"], dict)
        self.assertTrue(c["rowz"])
        self.assertGreaterEqual(r["mean"], 0.0)
        c2, _r2 = B.retune_structure(bench, [], ref)
        self.assertIsNone(c2["whiten"])


if __name__ == "__main__":
    unittest.main()
