"""tools/p10_fuse.py on synthetic data (PREREG_10 rev 1 part B). No data/local, no repository data, no GPU.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q tests/test_p10_fuse.py
"""
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import p06_sim as sim  # noqa: E402
import p06_style as st  # noqa: E402
import p10_apply as A  # noqa: E402
import p10_fuse as F  # noqa: E402
from c10data import LEVELS  # noqa: E402


def synth_X(n_players=12, per=10, dim=770, seed=0, spread=3.0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, spread, (n_players, dim))
    X, lab = [], []
    for p in range(n_players):
        X.append(centres[p] + rng.normal(0, 1, (per, dim)))
        lab += ["p%02d" % p] * per
    return np.vstack(X).astype(np.float32), np.array(lab)


class P1Term(unittest.TestCase):
    def test_lda_identifies_players(self):
        X, lab = synth_X(dim=40)
        m = F.fit_p1_lda(X, lab, np.ones(len(lab), bool))
        self.assertEqual((m["n_players"], m["n_games"]), (12, 120))
        Z = F.project_p1(X, m)
        cand = [list(range(p * 10, p * 10 + 7)) for p in range(12)]
        q = [list(range(p * 10 + 7, p * 10 + 10)) for p in range(12)]
        T, has = F.t_p1(Z, cand, q)
        self.assertTrue(has.all())
        self.assertTrue((T.argmax(1) == np.arange(12)).all())

    def test_standardisation_and_min_games(self):
        X, lab = synth_X(n_players=4, per=3, dim=8)
        mask = np.ones(len(lab), bool)
        mask[[0, 1]] = False                     # player p00 keeps one game -> dropped
        m = F.fit_p1_lda(X, lab, mask)
        self.assertEqual((m["n_players"], m["n_games"]), (3, 9))
        idx = np.flatnonzero(mask)[1:]
        np.testing.assert_allclose(m["m"], X[idx].astype(np.float64).mean(0))
        np.testing.assert_allclose(m["s"], X[idx].astype(np.float64).std(0) + 1e-6)
        ref = st.lda_fit((X[idx].astype(np.float64) - m["m"]) / m["s"], lab[idx])
        np.testing.assert_array_equal(ref["W"], m["W"])

    def test_projection_blocks_equal_whole(self):
        X, lab = synth_X(dim=20)
        m = F.fit_p1_lda(X, lab, np.ones(len(lab), bool))
        a = F.project_p1(X, m)
        b = F.project_p1(X, m, block=7)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
        np.testing.assert_allclose(F.project_p1(X, m, rows=[3, 5]), a[[3, 5]], rtol=0, atol=1e-12)

    def test_missing_rows_and_columns(self):
        rng = np.random.default_rng(1)
        Z = rng.normal(size=(10, 4))
        usable = np.ones(10, bool)
        usable[[8, 9]] = False
        T, has = F.t_p1(Z, [[0, 1], [2, 3], [8]], [[4], [9], [5, 6]], usable)
        self.assertEqual(has.tolist(), [True, False, True])
        self.assertTrue(np.allclose(T[:, 2], T[:, :2].min(1)))
        z = F.z_p1(Z, [[0, 1], [2, 3], [8]], [[4], [9], [5, 6]], usable)
        self.assertTrue((z[1] == 0).all())
        self.assertAlmostEqual(float(z[0].mean()), 0.0)

    def test_fuse(self):
        rng = np.random.default_rng(2)
        Fm, zT = rng.normal(size=(6, 9)), rng.normal(size=(6, 9))
        np.testing.assert_array_equal(F.fuse_10b(Fm, zT, 4.0), st.csls(Fm + 4.0 * zT, 10))
        np.testing.assert_array_equal(F.fuse_10b(Fm, zT, 0.0, 3), st.csls(Fm, 3))


class Selection(unittest.TestCase):
    def test_choose_w(self):
        g = (0.5, 1.0, 2.0, 4.0)
        self.assertEqual(F.choose_w({0.5: 0.90, 1.0: 0.95, 2.0: 0.9496, 4.0: 0.9494}, g), 2.0)
        self.assertEqual(F.choose_w({0.5: 0.90, 1.0: 0.95, 2.0: 0.9496, 4.0: 0.9495}, g), 4.0)
        self.assertEqual(F.choose_w({0.5: 0.96, 1.0: 0.95, 2.0: 0.94, 4.0: 0.93}, g), 0.5)

    def test_player_boot(self):
        """M2: point = per-question mean pooled over pools; CI = players resampled, pooled mean recomputed."""
        diffs = np.array([0.1, 0.3, -0.2, 0.0, 0.5])
        owners = np.array(["a", "a", "b", "c", "c"])
        u, sums, counts = F.player_sums(diffs, owners)
        self.assertEqual(u.tolist(), ["a", "b", "c"])
        np.testing.assert_allclose(sums, [0.4, -0.2, 0.5])
        np.testing.assert_array_equal(counts, [2, 1, 2])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, 3, (2000, 3))
        m = sums[idx].sum(1) / counts[idx].sum(1)
        got = F.player_boot(sums, counts)
        self.assertAlmostEqual(got[0], diffs.mean(), places=15)        # 0.14, not the per-player mean 0.0833
        self.assertEqual(got[1:], [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))])

    def test_gates(self):
        ok = {"base": [0.003, 0.0001, 0.01], "nov4": [0.02, 0.001, 0.05], "wr": [-0.002, -0.01, 0.0],
              "c20": [0.0, -0.01, 0.01]}
        self.assertTrue(F.gates(ok)["passed"])
        for scen, val, gate in (("base", [0.0029, 0.001, 0.01], "G1_base"), ("base", [0.01, 0.0, 0.02], "G1_base"),
                                ("nov4", [0.0199, 0.001, 0.04], "G2_nov4"), ("nov4", [0.05, -1e-9, 0.1], "G2_nov4"),
                                ("wr", [-0.0021, -0.01, 0.0], "G3_no_harm"), ("c20", [-0.003, -0.01, 0], "G3_no_harm")):
            b = dict(ok)
            b[scen] = val
            g = F.gates(b)
            self.assertFalse(g["passed"], (scen, val))
            self.assertFalse(g[gate]["passed"], (scen, val))

    def test_summarise(self):
        rows = []
        grid = (1.0, 2.0)
        for seed in range(4):
            for scen in F.SCENARIOS:
                v2 = np.full(4, 0.5)
                rows.append({"seed": seed, "scen": scen, "owner_pid": ["a", "b", "c", "d"],
                             "pts": {"E0": v2.tolist(), "Q04": v2.tolist(), "V2": v2.tolist(), "P1": v2.tolist(),
                                     "10B@1": (v2 + 0.01).tolist(), "10B@2": (v2 + 0.0105).tolist()}})
        w, tune, boot, qm, rec = F.summarise(rows, (0, 1), (2, 3), grid)
        self.assertEqual(w, 2.0)
        self.assertAlmostEqual(boot["base"][0], 0.0105)
        self.assertEqual((qm["nov4"]["questions"], qm["nov4"]["players"]), (8, 4))
        self.assertAlmostEqual(rec["wr"]["eval"]["10B@1"], 0.51)


def synth_sets(n_per_level=6, n_cand=8, n_base=4, nov=None, seed=0):
    rng = random.Random(seed)
    players, games, srow = [], [], 0
    for lv in LEVELS:
        for i in range(n_per_level):
            p = "%s_%02d" % (lv, i)
            players.append(p)
            n_nov = rng.randint(0, 4) if nov is None else nov
            for role, n in (("cand", n_cand), ("base", n_base), ("nov4", n_nov)):
                for k in range(n):
                    games.append({"rank": lv, "game_id": "g%d" % srow, "player_id": p, "color": "B", "role": role,
                                  "k": k, "store_row": srow})
                    srow += 1
    return {"format": F.FORMATS["sets"], "players": players, "games": games}


class Pools(unittest.TestCase):
    def setUp(self):
        self.ix = F.SetsIndex(synth_sets())
        self.shape = F.Shape(per_level=4, n_cand=8, n_c20=3, k_values=(1, 2, 3, 4))

    def test_choose_players_matches_sim(self):
        cache = {"n_all": {}, "n_black": {}, "level": {}}
        for p in self.ix.players:
            cache["n_all"][p] = cache["n_black"][p] = 150
            cache["level"][p] = self.ix.level[p]
        for seed in range(5):
            self.assertEqual(F.choose_players(self.ix, seed, 4), sim.choose_pool(cache, seed, per_level=4))

    def test_k_list(self):
        self.assertEqual(F.k_list(400, sim.K_VALUES), sim.k_list(400))
        with self.assertRaises(F.Refused):
            F.k_list(10, (1, 2, 3))

    def test_base_and_c20(self):
        pr = F.build_pool(self.ix, 3, "base", self.shape)
        n = len(pr["pool"])
        self.assertEqual(n, 40)
        self.assertEqual(pr["owner"].tolist(), list(range(n)))
        ks = F.k_list(n, (1, 2, 3, 4))
        random.Random("k|3").shuffle(ks)
        self.assertEqual(pr["k"].tolist(), ks)
        for i, p in enumerate(pr["pool"]):
            self.assertEqual(pr["cand_idx"][i], self.ix.by[p]["cand"])
            self.assertEqual(pr["q_idx"][i], self.ix.by[p]["base"][:ks[i]])
        c20 = F.build_pool(self.ix, 3, "c20", self.shape)
        self.assertTrue(all(len(c) == 3 for c in c20["cand_idx"]))
        self.assertEqual(c20["q_idx"], pr["q_idx"])
        self.assertEqual(F.build_pool(self.ix, 3, "base", self.shape)["q_idx"], pr["q_idx"])   # deterministic

    def test_nov4(self):
        pr = F.build_pool(self.ix, 5, "nov4", self.shape)
        n = len(pr["pool"])
        used = set()
        for i in range(n):
            q = pr["q_idx"][i]
            self.assertGreaterEqual(len(q), 1)
            self.assertEqual(len(q), pr["k"][i])
            self.assertLessEqual(len(q), pr["info"]["k_nominal"][i])
            o = pr["pool"][pr["owner"][i]]
            self.assertTrue(set(q) <= set(self.ix.by[o]["nov4"]))
            self.assertFalse(used & set(q))
            used |= set(q)
        for i in range(n):
            if pr["owner"][i] != i:          # replaced: the player's own reserve was empty or used up
                self.assertTrue(set(self.ix.by[pr["pool"][i]]["nov4"]) <= used)
        self.assertEqual(pr["info"]["n_replaced_owner"], int((pr["owner"] != np.arange(n)).sum()))

    def test_wr(self):
        pr = F.build_pool(self.ix, 7, "wr", self.shape)
        n = len(pr["pool"])
        ks = F.k_list(n, (1, 2, 3, 4))
        self.assertEqual(sorted(pr["k"].tolist()), sorted(ks))
        need = {}
        for i in range(n):
            need[pr["owner"][i]] = need.get(pr["owner"][i], 0) + pr["k"][i]
            self.assertEqual(len(pr["q_idx"][i]), pr["k"][i])
            self.assertTrue(set(pr["q_idx"][i]) <= set(self.ix.by[pr["pool"][pr["owner"][i]]]["base"]))
        self.assertTrue(all(v <= 4 for v in need.values()))
        r2 = random.Random("scen|wr|7")
        self.assertLess(len(set(pr["owner"].tolist())), n)
        self.assertGreaterEqual(len(set(pr["owner"].tolist())), len({r2.randrange(n) for _ in range(n)}))

    def test_wr_equals_make_pool_when_make_pool_needs_no_fallback(self):
        # with a roomy capacity the repair is p06_sim.make_pool's rule exactly
        n = 40
        ks = F.k_list(n, (5, 10, 15, 20))
        random.Random("k|11").shuffle(ks)
        owner, ks2, nrep, nfb = F._wr_questions(11, n, ks, [200] * n)
        r2 = random.Random("scen|wr|11")
        self.assertEqual(owner, [r2.randrange(n) for _ in range(n)])
        self.assertEqual((ks2, nrep, nfb), (ks, 0, 0))

    def test_wr_no_room_is_refused(self):
        with self.assertRaises(F.Refused):
            F._wr_questions(0, 8, [5] * 8, [1] * 8)

    def test_sets_index_checks(self):
        s = synth_sets()
        s["games"][3]["k"] = 9
        with self.assertRaises(F.Refused):
            F.SetsIndex(s)
        s = synth_sets()
        s["games"][0]["color"] = "W"
        with self.assertRaises(F.Refused):
            F.SetsIndex(s)
        s = synth_sets()
        s["format"] = "other"
        with self.assertRaises(F.Refused):
            F.SetsIndex(s)


def write_chunk(path, refs, means, failed=None, counts=None, set_="p10mock"):
    n = len(refs)
    np.savez(path, format=np.array(F.CHUNK_FORMAT), set=np.array(set_), sides=np.array(["target", "opponent"]),
             ref=np.asarray(refs, np.int64), precision=np.array("fp16"), meta_sha256=np.array("m"),
             ckpt_bytes=np.array(1), code_sha256=np.array("c"), means=means.astype(np.float32),
             failed=np.zeros(n, bool) if failed is None else failed,
             counts=np.ones((n, 2), np.int32) * 5 if counts is None else counts)


class Chunks(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="p10fuse_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_load_side0(self):
        rng = np.random.default_rng(0)
        M = rng.normal(size=(5, 2, 770))
        cnt = np.ones((5, 2), np.int32)
        cnt[3, 0] = 0
        write_chunk(os.path.join(self.d, "chunk_00000.npz"), [0, 1, 2], M[:3],
                    failed=np.array([False, True, False]), counts=cnt[:3])
        write_chunk(os.path.join(self.d, "chunk_00001.npz"), [3, 4], M[3:], counts=cnt[3:])
        X, usable, failed, have, c0 = F.load_side0(F.chunk_files(self.d), 6, "p10mock")
        np.testing.assert_array_equal(X[:5], M[:, 0, :].astype(np.float32))
        self.assertEqual(usable.tolist(), [True, False, True, False, True, False])
        self.assertEqual(have.tolist(), [True] * 5 + [False])
        rec = F.chunk_record(os.path.join(self.d, "chunk_00000.npz"))
        self.assertEqual((rec["games"], rec["failed"], rec["no_target_moves"]), (3, 1, 0))
        with self.assertRaises(F.Refused):
            F.load_side0(F.chunk_files(self.d), 6, "p10val")

    def test_duplicate_refs_refused(self):
        M = np.zeros((2, 2, 770))
        write_chunk(os.path.join(self.d, "chunk_00000.npz"), [0, 1], M)
        write_chunk(os.path.join(self.d, "chunk_00001.npz"), [1, 2], M)
        with self.assertRaises(F.Refused):
            F.load_side0(F.chunk_files(self.d), 3)


class Modes(unittest.TestCase):
    def ns(self, **kw):
        base = {"out_dir": None, "sets": None, "chunk_dir": None, "shape": None, "seeds_tune": None,
                "seeds_eval": None, "pilot_dir": None, "x3_json": None, "rerun": False, "reason": None}
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_resolve(self):
        c = F.resolve(self.ns(), "mock")
        self.assertFalse(c.preview)
        self.assertEqual(c.out_dir, F.OUT)
        c = F.resolve(self.ns(out_dir=os.path.join(F.TMP, "x")), "mock")
        self.assertTrue(c.preview)
        for bad in (F.OUT, F.TMP, os.path.join(REPO, "data", "prereg10", "mock"), tempfile.gettempdir()):
            with self.assertRaises(F.Refused):
                F.resolve(self.ns(out_dir=bad), "mock")
        for k in ("sets", "chunk_dir", "shape", "seeds_tune", "seeds_eval", "pilot_dir", "x3_json"):
            with self.assertRaises(F.Refused):
                F.resolve(self.ns(**{k: "x"}), "mock")

    def test_preview_record_never_accepted(self):
        d = tempfile.mkdtemp(dir=F.TMP if os.path.isdir(F.TMP) else None, prefix="p10test_")
        try:
            st.write_json(os.path.join(d, "mock_report.json"), {"format": F.FORMATS["mock"], "preview": False})
            c = types.SimpleNamespace(out_dir=d, preview=True)
            with self.assertRaises(F.Refused):
                F.prior(c, "mock_report.json", F.FORMATS["mock"])
            st.write_json(os.path.join(d, "mock_report.json"), {"format": F.FORMATS["mock"], "preview": True})
            c2 = types.SimpleNamespace(out_dir=d, preview=False)
            with self.assertRaises(F.Refused):
                F.prior(c2, "mock_report.json", F.FORMATS["mock"])
            r, _sha, commit = F.prior(c, "mock_report.json", F.FORMATS["mock"])
            self.assertTrue(r["preview"])
            self.assertIsNone(commit)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_frozen_usable(self):
        ok = {"format": F.FORMATS["frozen"], "preview": False, "b_passed": True, "score_pending_val": True}
        self.assertTrue(A.frozen_usable(ok))
        for k, v in (("preview", True), ("b_passed", False), ("format", "x")):
            with self.assertRaises(F.Refused):
                A.frozen_usable(dict(ok, **{k: v}))

    def test_parse_shape_and_seeds(self):
        s = F.parse_shape("per_level=10,n_cand=24,n_c20=8,k=2/4/6/8")
        self.assertEqual(s.as_dict(), {"per_level": 10, "n_cand": 24, "n_c20": 8, "k_values": [2, 4, 6, 8]})
        self.assertFalse(s.official())
        self.assertTrue(F.Shape().official())
        self.assertEqual(F.parse_seeds("3-5", ()), (3, 4, 5))
        self.assertEqual(F.parse_seeds(None, (1,)), (1,))
        with self.assertRaises(F.Refused):
            F.parse_shape("bogus=1")

    def test_blas_env(self):
        old = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "4"
        try:
            with self.assertRaises(F.Refused):
                F.blas_env_check(False)
            self.assertEqual(F.blas_env_check(True), {"OMP_NUM_THREADS": "4"})
        finally:
            if old is None:
                os.environ.pop("OMP_NUM_THREADS")
            else:
                os.environ["OMP_NUM_THREADS"] = old


class Selfcheck(unittest.TestCase):
    def ref(self, d=0.0):
        u = {s: {"V2": 0.9, "V2+P1ldaBW@1": 0.95} for s in F.PILOT_SCEN}
        f = {s: {k: v + d for k, v in x.items()} for s, x in u.items()}
        c = {s: {k: round(v, 4) for k, v in x.items()} for s, x in u.items()}
        return f, u, c

    def test_compare(self):
        f, u, c = self.ref(0.0)
        self.assertEqual(F.compare_selfcheck(f, u, c)[:2], (True, False))
        f, u, c = self.ref(1e-9)
        self.assertEqual(F.compare_selfcheck(f, u, c)[:2], (True, False))
        f, u, c = self.ref(5e-7)                       # numerical-path size: fails, may go on with a revision
        self.assertEqual(F.compare_selfcheck(f, u, c)[:2], (False, True))
        f, u, c = self.ref(1e-4)
        self.assertEqual(F.compare_selfcheck(f, u, c)[:2], (False, False))
        f, u, c = self.ref(0.0)
        del f["c30"]["V2"]                             # a missing value is never numerical-only
        self.assertEqual(F.compare_selfcheck(f, u, c)[:2], (False, False))

    def test_four_decimals_record_only(self):
        f, u, c = self.ref(0.0)
        c = {s: {k: v + 0.001 for k, v in x.items()} for s, x in c.items()}
        ok, _num, rows = F.compare_selfcheck(f, u, c)
        self.assertTrue(ok)
        self.assertFalse(any(r["formal_4dp_equal_record_only"] for r in rows))

    def test_accept(self):
        c = types.SimpleNamespace(preview=True, out_dir="x")
        self.assertIsNone(F.selfcheck_accept(c, {"passed": True}, None))
        with self.assertRaises(F.Refused):
            F.selfcheck_accept(c, {"passed": True}, 3)
        with self.assertRaises(F.Refused):
            F.selfcheck_accept(c, {"passed": False, "only_numeric_path": False}, 3)
        self.assertEqual(F.selfcheck_accept(c, {"passed": False, "only_numeric_path": True}, 3)["revision"], 3)

    def test_later_revision(self):
        saved = F.revision_committed
        F.revision_committed = lambda n: ("abc", "2026-10-12T10:00:00+08:00")
        try:
            self.assertEqual(F.later_revision(2, "2026-10-11T10:00:00+08:00", "x")["commit"], "abc")
            for n, t in ((1, "2026-10-11T10:00:00+08:00"), (None, "2026-10-11T10:00:00+08:00"),
                         (2, "2026-10-12T10:00:00+08:00"), (2, None)):
                with self.assertRaises(F.Refused):
                    F.later_revision(n, t, "x")
        finally:
            F.revision_committed = saved

    def test_binding_covers_only_the_mock_path(self):
        self.assertNotIn("tools/p10_apply.py", F.MOCK_TOOL_FILES)
        self.assertNotIn("tools/p10_p1.py", F.MOCK_TOOL_FILES)
        self.assertIn("tools/p10_fuse.py", F.MOCK_TOOL_FILES)
        self.assertTrue(set(F.MOCK_TOOL_FILES) < set(A.APPLY_TOOL_FILES))
        ts = {"files_sha256": {f: "x" for f in A.APPLY_TOOL_FILES}}
        rec = {"files_sha256": dict(ts["files_sha256"], **{"tools/p10_apply.py": "y", "tools/p10_p1.py": "y"})}
        self.assertEqual(F.tools_differ(ts, rec), [])
        self.assertEqual(F.tools_differ(ts, rec, A.APPLY_TOOL_FILES), ["tools/p10_apply.py", "tools/p10_p1.py"])


class Apply(unittest.TestCase):
    def test_val_keys(self):
        cand = [("g1", "pA", "B", "x"), ("g2", "pA", "W", "y")]
        qs = [("q1", [(1, "W", "a"), (2, "B", "b")]), ("q2", [(1, "B", "c")])]
        keys, cols = A.val_keys(cand, qs)
        self.assertEqual(keys, ["g1", "g2", "q1#1", "q1#2", "q2#1"])
        self.assertEqual(cols, ["B", "W", "W", "B", "B"])
        self.assertEqual(A.keys_sha(keys), hashlib.sha256("g1\ng2\nq1#1\nq1#2\nq2#1".encode()).hexdigest())

    def test_apply_decision(self):
        self.assertEqual(A.apply_decision(True, True, 0.0, True, True, None)[0], "pending_audit")
        self.assertEqual(A.apply_decision(True, True, 0.005, True, True, None)[0], "pending_audit")
        for args in ((False, True, 0, True, True, None), (True, False, 0, True, True, None),
                     (True, True, 0.0051, True, True, None), (True, True, 0, False, True, None),
                     (True, True, 0, True, False, None), (True, True, 0, True, True, "bug")):
            self.assertIs(A.apply_decision(*args)[0], False, args)

    def test_audit_binding_and_finalize(self):
        rep = {"outputs_sha256": {"player_val_top5.csv": "o"},
               "inputs_sha256": {"p1_lda.npz": "l", "frozen_config.json": "f", "val_report.json": "v"},
               "score_allowed": "pending_audit", "score_blocked_reasons": ["prescore_audit.json of ..."]}
        au = {"format": F.FORMATS["audit"], "program_errors_affecting_10B": [], "independent_recompute_identical": True,
              "outputs_sha256": {"player_val_top5.csv": "o"},
              "inputs_sha256": {"p1_lda.npz": "l", "frozen_config.json": "f", "val_report.json": "v"}}
        t1, t2 = "2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00"
        self.assertEqual(A.audit_binding(au, rep, t2, t1), [])
        self.assertTrue(A.audit_binding(au, rep, t1, t2))
        self.assertTrue(A.audit_binding(dict(au, independent_recompute_identical=False), rep, t2, t1))
        self.assertTrue(A.audit_binding(dict(au, outputs_sha256={"player_val_top5.csv": "x"}), rep, t2, t1))
        self.assertEqual(A.finalize_decision(rep, "ok", {}), (True, []))
        self.assertEqual(A.finalize_decision(rep, "bug", {})[0], False)
        for state in ("missing", "unbound"):
            with self.assertRaises(F.Refused):
                A.finalize_decision(rep, state, {})
        with self.assertRaises(F.Refused):
            A.finalize_decision(dict(rep, score_allowed=False, score_blocked_reasons=["pool gate ..."]), "ok", {})
        with self.assertRaises(F.Refused):
            A.finalize_decision(dict(rep, score_allowed=True), "ok", {})

    def test_sr06_with_p1_w0_equals_06(self):
        """With w* = 0 every 10B mean equals the 06 mean; with w* > 0 the loop is the same pseudo pools."""
        rng = np.random.default_rng(3)
        cand_ids = ["c%02d" % i for i in range(6)]
        gid = []
        c_by_pid = {}
        for p in cand_ids:
            c_by_pid[p] = list(range(len(gid), len(gid) + 100))
            gid += ["%s_g%03d" % (p, j) for j in range(100)]
        ms = types.SimpleNamespace(gid=np.array(gid))
        Z = rng.normal(size=(len(gid), 5))
        usable = np.ones(len(gid), bool)

        def apply_terms(_ms, _models, cgr, qg, _e):
            nq, nc = len(qg), len(cgr)
            return {k: (rng.normal(size=(nq, nc)), np.ones(nq, bool)) for k in ("E0", "O", "L", "JO", "S2")}
        P6 = types.SimpleNamespace(apply_terms=apply_terms)
        combos = sim.combo_list(None)
        srm, sr10 = A.sr06_with_p1(P6, ms, cand_ids, c_by_pid, [5, 10, 15, 20], None, None, combos, 10, Z, usable,
                                   0.0)
        for n, _w in combos:
            self.assertAlmostEqual(srm[n], sr10[n], places=12)
        self.assertIn("R02", srm)


if __name__ == "__main__":
    unittest.main()
