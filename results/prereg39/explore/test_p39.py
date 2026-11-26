"""prereg39 explore: small CPU tests (no KataGo, no GPU, no score of real data).

    PY -m pytest -q results/prereg39/explore/test_p39.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p39_lists as PL  # noqa: E402
import p39_eval as PE  # noqa: E402

LISTS = PL.LISTS
HAVE_LISTS = os.path.exists(os.path.join(LISTS, "L.json"))


class Rules(unittest.TestCase):
    def test_l_games_deterministic_and_balanced(self):
        games = [("g%03d" % i, "B" if i % 3 else "W") for i in range(90)]
        a = PL.l_games("p1", games)
        self.assertEqual(a, PL.l_games("p1", list(games)))
        self.assertEqual(len(a), 30)
        self.assertEqual(len(set(a)), 30)
        self.assertEqual(sum(1 for i in a if games[i][1] == "B"), 15)
        self.assertNotEqual(a, PL.l_games("p2", games))

    def test_l_games_tops_up_and_short(self):
        games = [("g%03d" % i, "B" if i < 36 else "W") for i in range(40)]     # only 4 white games
        a = PL.l_games("p", games)
        self.assertEqual(len(a), 30)
        self.assertEqual(sum(1 for i in a if games[i][1] == "W"), 4)
        self.assertEqual(sorted(PL.l_games("p", games[:22])), list(range(22)))

    def test_l_players_per_level(self):
        T = ["p%04d" % i for i in range(300)]
        lv = {p: ("A" if i % 2 else "6D") for i, p in enumerate(T)}
        a = PL.l_players(T, lv)
        self.assertEqual(len(a), 240)
        self.assertEqual(a, PL.l_players(list(reversed(T)), lv))


@unittest.skipUnless(HAVE_LISTS, "data/prereg39/lists not built")
class ListsOnDisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lists = {}
        for n in ("L", "D", "E", "E2", "pub2", "priv2"):
            with open(os.path.join(LISTS, n + ".json"), encoding="utf-8") as f:
                cls.lists[n] = json.load(f)["games"]
        cls.E, cls.D, cls.T, _ = PL.L36.splits()

    def test_record_shas(self):
        with open(PL.RECORD, encoding="utf-8") as f:
            rec = json.load(f)
        for n in self.lists:
            self.assertEqual(PL.sha_file(os.path.join(LISTS, n + ".json")), rec["lists_sha256"][n])
            self.assertEqual(len(self.lists[n]), rec["sets"][n]["games"])

    def test_players_disjoint(self):
        pl = {n: {g["player_id"] for g in self.lists[n]} for n in ("L", "D", "E", "E2")}
        held = set(self.E) | set(self.D)
        self.assertFalse(pl["L"] & held)
        self.assertTrue(pl["L"] <= set(self.T))
        self.assertTrue(pl["D"] <= set(self.D))
        self.assertTrue(pl["E"] | pl["E2"] <= set(self.E))

    def test_games_unique_and_l_shape(self):
        keys = [(g["rank"], g["game_id"], g["player_id"]) for n in ("L", "D", "E", "E2") for g in self.lists[n]]
        self.assertEqual(len(keys), len(set(keys)))
        per = {}
        for g in self.lists["L"]:
            per.setdefault(g["player_id"], []).append(g["color"])
        self.assertEqual(len(per), 1200)
        self.assertTrue(all(20 <= len(v) <= 30 for v in per.values()))

    def test_official_lists(self):
        for n in ("pub2", "priv2"):
            self.assertEqual(len(self.lists[n]), 7100)
            self.assertEqual([g["item"] for g in self.lists[n]], list(range(7100)))

    def test_mock_reuse_only_black_store_rows(self):
        for n in ("L", "D", "E", "E2"):
            for g in self.lists[n]:
                if g["mock_ref"] >= 0:
                    self.assertEqual(g["color"], "B")
                    self.assertGreaterEqual(g["row"], 0)


def fake_players(rng, n_players, n_games, dim=770, signal=1.0):
    mu = rng.normal(size=(n_players, dim)) * signal
    X = (np.repeat(mu, n_games, 0) + rng.normal(size=(n_players * n_games, dim))).astype(np.float32)
    lab = np.repeat(np.arange(n_players), n_games)
    col = np.tile(np.array(["B", "W"]), n_players * n_games // 2)
    return X, lab, col


class FakeTerms(unittest.TestCase):
    """Random 770-dim vectors through fit -> terms -> fusion grid: shapes and z rows only (not a real score)."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        X, lab, col = fake_players(rng, 120, 12)
        Lset = {"games": [{"player_id": "t%d" % a, "color": c} for a, c in zip(lab, col)], "X": X,
                "usable": np.ones(len(X), bool)}
        cls.models = PE.fit_models(Lset)
        Xp, labp, colp = fake_players(rng, 30, 14)
        cls.X, cls.col = Xp, colp
        cls.usable = np.ones(len(Xp), bool)
        cls.usable[5] = False
        cls.cand = [list(range(p * 14, p * 14 + 10)) for p in range(30)]
        cls.owner = np.array([3, 7, 7, 20, 29, 0])
        cls.q = [[o * 14 + 10 + j for j in range(k)] for o, k in zip(cls.owner, (1, 2, 3, 4, 1, 2))]

    def test_shapes_and_z(self):
        t = PE.p1_terms(self.X, self.usable, self.col, self.cand, self.q, self.models)
        for v in PE.VARIANTS:
            z, has = t[v]
            self.assertEqual(z.shape, (6, 30))
            self.assertTrue(np.asarray(has).all())
            self.assertTrue(np.isfinite(z).all())
        z = t["joint"][0]
        np.testing.assert_allclose(z.mean(1), 0, atol=1e-9)
        np.testing.assert_allclose(z.std(1), 1, atol=1e-9)

    def test_question_without_usable_game(self):
        usable = self.usable.copy()
        usable[self.q[0]] = False
        t = PE.p1_terms(self.X, usable, self.col, self.cand, self.q, self.models)
        for v in PE.VARIANTS:
            z, has = t[v]
            self.assertFalse(has[0])
            self.assertTrue((z[0] == 0).all())

    def test_models_sha_stable(self):
        self.assertEqual(PE.models_sha(self.models), PE.models_sha(self.models))
        self.assertEqual(self.models["joint"]["W"].shape[0], 770)

    def test_grid_runs(self):
        t = PE.p1_terms(self.X, self.usable, self.col, self.cand, self.q, self.models)
        pools = [{"owner": self.owner}]
        Fb = [np.zeros((6, 30))]
        for v in PE.VARIANTS:
            m = PE.mean_score(Fb, [t[v][0]], 1.0, 10, pools)
            self.assertTrue(0.0 <= m <= 1.0)

    def test_store_index(self):
        s1 = {"X": self.X[:10], "usable": self.usable[:10], "games": [{"row": 100 + i, "color": "B"} for i in range(10)]}
        s2 = {"X": self.X[10:20], "usable": self.usable[10:20],
              "games": [{"row": 500 + i, "color": "W"} for i in range(10)]}
        X, usable, colour, pos = PE.store_index([s1, s2])
        self.assertEqual(X.shape, (20, 770))
        self.assertEqual(pos[505], 15)
        self.assertEqual(colour[15], "W")
        with self.assertRaises(SystemExit):
            PE.store_index([s1, s1])


@unittest.skipUnless(os.path.isdir(os.path.join(PL.DATA, "tmp", "s1", "L")) and HAVE_LISTS, "no smoke chunks")
class SmokeChunks(unittest.TestCase):
    def test_smoke_chunk_layout(self):
        games = PE.read_list("L")
        paths = PE.F10.chunk_files(os.path.join(PL.DATA, "tmp", "s1", "L"))
        X, usable, failed, have, _cnt = PE.F10.load_side0(paths, len(games), "p39L")
        fresh = [j for j, g in enumerate(games) if g["mock_ref"] < 0][:20]
        self.assertEqual(np.flatnonzero(have).tolist(), fresh)
        self.assertEqual(X.shape[1], 770)
        with self.assertRaises(SystemExit):
            PE.load_set("L", p1_dir=os.path.join(PL.DATA, "tmp", "s1"))      # incomplete -> refused


# ================================================================ PLAN.md revision 2: base = M36 + prereg38 backend
REPO = PE.REPO
HAVE_D = os.path.exists(os.path.join(REPO, "data", "prereg36", "terms_D", "pool_100.npz")) and \
    os.path.exists(os.path.join(PE.P38_DEFAULT["emb_dir"], "store_D.npz"))
HAVE_MOCK = os.path.exists(PE.MOCK_SETS)


class FakeB4(unittest.TestCase):
    """pw "b4" and the "none" regression on random vectors (shapes / z rows only, not a real score)."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        X, lab, col = fake_players(rng, 120, 12)
        Lset = {"games": [{"player_id": "t%d" % a, "color": c} for a, c in zip(lab, col)], "X": X,
                "usable": np.ones(len(X), bool)}
        cls.models = PE.fit_models(Lset)
        Xp, _labp, colp = fake_players(rng, 60, 14)
        cls.X, cls.col = Xp, colp
        cls.usable = np.ones(len(Xp), bool)
        cls.usable[3] = False
        cls.cand = [list(range(p * 14, p * 14 + 10)) for p in range(60)]
        cls.owner = np.array([3, 7, 7, 20, 59, 0, 41, 12])
        cls.q = [[o * 14 + 10 + j for j in range(k)] for o, k in zip(cls.owner, (1, 2, 3, 4, 1, 2, 3, 1))]

    def test_b4_shapes_and_z(self):
        t = PE.p1_terms(self.X, self.usable, self.col, self.cand, self.q, self.models, pw="b4")
        for v in PE.VARIANTS:
            z, has = t[v]
            self.assertEqual(z.shape, (8, 60))
            self.assertTrue(np.asarray(has).all())
            self.assertTrue(np.isfinite(z).all())
        z = t["joint"][0]
        np.testing.assert_allclose(z.mean(1), 0, atol=1e-9)
        np.testing.assert_allclose(z.std(1), 1, atol=1e-9)

    def test_b4_question_and_candidate_without_games(self):
        usable = self.usable.copy()
        usable[self.q[0]] = False
        usable[self.cand[5]] = False                      # candidate 5 has no usable game
        t = PE.p1_terms(self.X, usable, self.col, self.cand, self.q, self.models, pw="b4")
        for v in PE.VARIANTS:
            z, has = t[v]
            self.assertFalse(has[0])
            self.assertTrue((z[0] == 0).all())
            self.assertTrue(np.isfinite(z).all())
        Zs = PE.project_all(self.X, self.models)
        cg = [[g for g in grp if usable[g]] for grp in self.cand]
        qg = [[g for g in grp if usable[g]] for grp in self.q]
        T, has = PE.b4_term(Zs["joint"], cg, qg)
        others = np.delete(T, 5, axis=1)
        np.testing.assert_array_equal(T[has, 5], others[has].min(1))

    def test_b4_is_pool_invariant_to_game_order(self):
        """The B4 transform only depends on the pool's games and candidate ids, not on their order in X."""
        perm = np.random.default_rng(5).permutation(len(self.X))
        inv = np.argsort(perm)
        X2, col2, us2 = self.X[perm], self.col[perm], self.usable[perm]
        cand2 = [[int(inv[g]) for g in grp] for grp in self.cand]
        q2 = [[int(inv[g]) for g in grp] for grp in self.q]
        a = PE.p1_terms(self.X, self.usable, self.col, self.cand, self.q, self.models, pw="b4")["joint"][0]
        b = PE.p1_terms(X2, us2, col2, cand2, q2, self.models, pw="b4")["joint"][0]
        np.testing.assert_allclose(a, b, atol=1e-8)

    def test_none_unchanged(self):
        """pw "none" = the pre-revision-2 term (p10_fuse.t_p1 with its own usable filter)."""
        t = PE.p1_terms(self.X, self.usable, self.col, self.cand, self.q, self.models)
        Z = PE.F10.project_p1(self.X, self.models["joint"])
        T, has = PE.F10.t_p1(Z, self.cand, self.q, self.usable)
        np.testing.assert_array_equal(t["joint"][0], PE.st.zscore_rows(T, has))
        Zs = PE.project_all(self.X, self.models)
        t2 = PE.p1_terms(self.X, self.usable, self.col, self.cand, self.q, self.models, Zs=Zs)
        for v in PE.VARIANTS:
            np.testing.assert_array_equal(t[v][0], t2[v][0])

    def test_unknown_pw(self):
        with self.assertRaises(ValueError):
            PE.p1_terms(self.X, self.usable, self.col, self.cand, self.q, self.models, pw="xx")


class Grid(unittest.TestCase):
    def test_grid_p38(self):
        g = PE.grid_for({"name": "p38"})
        self.assertEqual(len(g), 2 * 2 * 6 * 3)
        self.assertEqual(g[0], ("joint", "none", 0.25, 10))
        self.assertEqual(g[18], ("joint", "b4", 0.25, 10))
        self.assertEqual(g[-1], ("split", "b4", 8, 40))

    def test_grid_old_bases(self):
        self.assertEqual(len(PE.grid_for({"name": "p36"})), 36)
        self.assertTrue(all(x[1] == "none" for x in PE.grid_for({"name": "p37"})))

    def test_first_maximum_and_wp0_seam(self):
        rng = np.random.default_rng(2)
        pools = [{"owner": rng.integers(0, 30, 12)} for _ in range(3)]
        Fs = [rng.normal(size=(12, 30)) for _ in pools]
        z0 = [np.zeros((12, 30)) for _ in pools]
        terms = {"none": [{"joint": (z, None), "split": (z, None)} for z in z0],
                 "b4": [{"joint": (z, None), "split": (z, None)} for z in z0]}
        rows, top = PE.select_grid(Fs, terms, pools, PE.grid_for({"name": "p38"}))
        self.assertEqual(top[1:], ("joint", "none", 0.25, top[4]))       # all equal per k -> first in order
        base = {"k": 10}
        self.assertEqual(PE.mean_score(Fs, z0, 0.0, 10, pools), float(np.mean(PE.base_scores(base, Fs, pools))))

    def test_m39_of_old_tune(self):
        self.assertEqual(PE.m39_of({"M39": {"variant": "joint", "wP": 1, "csls_k": 20}}), ("joint", "none", 1, 20))


class JudgeRule(unittest.TestCase):
    def _run(self, gain, n=20):
        rng = np.random.default_rng(3)
        pools, Fs, terms = [], [], []
        for i in range(n):
            owner = np.arange(40) % 30
            F = rng.normal(size=(40, 30))
            z = np.zeros((40, 30))
            z[np.arange(40), owner] = gain                   # P1 pushes the owner up
            pools.append({"owner": owner, "k": np.ones(40, np.int64), "q_rows": [[0]] * 40})
            Fs.append(F)
            terms.append({"joint": (z, np.ones(40, bool))})
        return PE.judge_numbers(Fs, terms, pools, {"k": 10}, ("joint", "none", 1.0, 10))

    def test_adopts_strong_gain(self):
        r = self._run(3.0)
        self.assertTrue(r["adopt"])
        self.assertGreaterEqual(r["M39_minus_base"]["pools_positive"], 18)
        self.assertIn(">= 18 of 20", r["rule"])

    def test_rejects_no_gain(self):
        r = self._run(0.0)
        self.assertFalse(r["adopt"])
        self.assertEqual(r["M39_minus_base"]["mean"], 0.0)

    def test_e10_rule(self):
        self.assertIn(">= 9 of 10", self._run(3.0, n=10)["rule"])

    def test_e_seam(self):
        base = {"e_per_pool": [0.5, 0.25, 0.75], "e_mean": 0.5}
        PE.check_e_seam(base, np.array([0.5, 0.25, 0.75]), 3)
        PE.check_e_seam(base, np.array([0.5, 0.25]), 2)
        with self.assertRaises(SystemExit):
            PE.check_e_seam(base, np.array([0.5, 0.25, 0.7500000000000001]), 3)


class Top5(unittest.TestCase):
    def test_bytes_and_base_check(self):
        import tempfile
        data = PE.top5_bytes(["q1", "q2"], [["a", "b", "c", "d", "e"], ["e", "d", "c", "b", "a"]])
        self.assertEqual(data, b"question_id,top_1,top_2,top_3,top_4,top_5\nq1,a,b,c,d,e\nq2,e,d,c,b,a\n")
        base = {"default": True, "record": {"tune_sha256_lf": "abc"}}
        self.assertIsNone(PE.check_base_top5({"default": False}, "public", data))
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "public"))
            with open(os.path.join(d, "public", "top5.csv"), "wb") as f:
                f.write(data)
            with open(os.path.join(d, "public", "apply_report.json"), "w", encoding="utf-8") as f:
                json.dump({"tune_sha256": "abc"}, f)
            self.assertTrue(PE.check_base_top5(base, "public", data, d))
            with self.assertRaises(SystemExit):
                PE.check_base_top5(base, "public", data.replace(b"q2,e", b"q2,a"), d)
            with self.assertRaises(SystemExit):
                PE.check_base_top5({"default": True, "record": {"tune_sha256_lf": "other"}}, "public", data, d)


class BaseP38Records(unittest.TestCase):
    """Only the committed prereg38 records (no pools, no vectors)."""

    def test_default_base(self):
        b = PE.base_info("p38")
        self.assertTrue(b["default"])
        self.assertEqual(b["k"], 10)
        self.assertEqual(b["e_mean"], 0.4969569920097353)
        self.assertEqual(b["d_seam"]["mean"], 0.5805158972862898)
        self.assertEqual(len(b["e_per_pool"]), 20)
        self.assertEqual(len(b["d_seam"]["per_pool"]), 10)
        r = b["record"]
        self.assertEqual(r["cfg"]["whiten"], {"lam": 0.02, "lda": 32})
        self.assertEqual(r["tune"], "results/prereg38/explore/tune_D_B4.json")
        self.assertEqual(r["emb_dir"], "data/prereg36/emb/step_40000")
        b2 = PE.base_from_record(json.loads(json.dumps(r)))
        self.assertEqual(b2["record"], r)
        self.assertTrue(b2["default"])

    def test_record_tamper_refused(self):
        r = dict(PE.base_info("p38")["record"])
        r["eval_sha256_lf"] = "0" * 64
        with self.assertRaises(SystemExit):
            PE.base_from_record(r)

    def test_eval_must_adopt_the_tune(self):
        with self.assertRaises(SystemExit):        # prereg38's eval judged tune_D_B4.json, not tune_D.json
            PE.base_info("p38", tune=os.path.join(PE.P38, "tune_D.json"))
        with self.assertRaises(SystemExit):
            PE.base_info("p36", tune=PE.P38_DEFAULT["tune"])


@unittest.skipUnless(HAVE_D, "prereg36 D pools / step_40000 vectors not here")
class SeamD(unittest.TestCase):
    """Real D pools: the p38 base reproduces tune_D_B4.json per pool, bit for bit (PLAN.md revision 2)."""

    def test_d_seam(self):
        PE.BK.blas_threads(1)
        b = PE.base_info("p38")
        pools = PE.EV.load_pools(PE.EV.TERMS_DIR["D"], PE.EV.D_SEEDS)
        Fs = b["fused"](pools, "D")
        per = PE.check_d_seam(b, Fs, pools)
        self.assertEqual(per, b["d_seam"]["per_pool"])
        self.assertEqual(float(np.mean(per)), 0.5805158972862898)
        bad = dict(b, d_seam={"per_pool": per[:-1] + [per[-1] + 1e-15], "mean": b["d_seam"]["mean"]})
        with self.assertRaises(SystemExit):
            PE.check_d_seam(bad, Fs, pools)


@unittest.skipUnless(HAVE_D and HAVE_MOCK and HAVE_LISTS, "D pools / PREREG_10 mock / lists not here")
class MockCompat(unittest.TestCase):
    """The PREREG_10 mock chunks (black games of fit players) through the revision-2 path on a real D pool:
    loading, metadata, LDA fit, both pw, shapes and z rows. No score is computed or printed."""

    @classmethod
    def setUpClass(cls):
        PE.BK.blas_threads(1)
        cls.m = PE.mock_side0()
        lg = PE.read_list("L")
        dg = PE.read_list("D")
        li = [g for g in lg if g["mock_ref"] >= 0]
        X = cls.m["X"][[g["mock_ref"] for g in li]]
        us = cls.m["usable"][[g["mock_ref"] for g in li]]
        lab = np.array([g["player_id"] for g in li])
        one = PE.F10.fit_p1_lda(X, lab, us)        # test-only LDA: the mock has black games only, so the same
        cls.models = {"joint": one, "B": one, "W": one}   # model stands in for the white one
        cls.dg = dg
        n = len(dg)
        cls.X = np.zeros((n, 770), np.float32)
        cls.usable = np.zeros(n, bool)
        ref = np.array([g["mock_ref"] for g in dg])
        idx = np.flatnonzero(ref >= 0)
        cls.X[idx] = cls.m["X"][ref[idx]]
        cls.usable[idx] = cls.m["usable"][ref[idx]]
        cls.colour = np.array([g["color"] for g in dg])
        cls.pos = {g["row"]: i for i, g in enumerate(dg)}
        cls.pool = PE.EV.load_pools(PE.EV.TERMS_DIR["D"], [100])[0]

    def test_mock_metadata_single(self):
        self.assertEqual(len(self.m["meta"]), 1)
        fmt = next(iter(self.m["meta"]))[0]
        self.assertTrue(fmt)

    def test_reused_games_black_and_mostly_usable(self):
        ref = [g for g in self.dg if g["mock_ref"] >= 0]
        self.assertEqual(len(ref), 5148)
        self.assertTrue(all(g["color"] == "B" for g in ref))
        self.assertGreater(int(self.usable.sum()), 0.95 * len(ref))

    def test_both_pw_on_a_d_pool(self):
        Zs = PE.project_all(self.X, self.models)
        for pw in PE.PW:
            t = PE.pool_p1(self.pool, self.X, self.usable, self.colour, self.pos, self.models, pw, Zs)
            for v in PE.VARIANTS:
                z, has = t[v]
                self.assertEqual(z.shape, (len(self.pool["q_rows"]), len(self.pool["cand_rows"])))
                self.assertTrue(np.isfinite(z).all())
                has = np.asarray(has)
                self.assertGreater(int(has.sum()), 0)
                self.assertTrue((z[~has] == 0).all())
                np.testing.assert_allclose(z[has].mean(1), 0, atol=1e-9)

    def test_load_set_refuses_missing_fresh_games(self):
        with self.assertRaises(SystemExit):
            PE.load_set("D")                      # the fresh (white / non-mock) D games are not extracted yet


if __name__ == "__main__":
    unittest.main()
