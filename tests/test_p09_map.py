"""PREREG_09 tools/p09_map.py on synthetic data (no P3 chunk, no validation file, no real training).

* the training port equals the exploration's train.py train() bit for bit (same seeds, same threads; subprocess);
* the epoch rule (first minimum of the mean OOF log-loss), the per-member CV + refit on a tiny synthetic probe;
* build_prep on synthetic P1-format chunks: usable / duplicate exclusion, order, labels, raw X (NaN opponent half),
  raw EX, and Prep's moments = np.nanmean / np.nanstd of the subset;
* train_members -> member files (rerun keeps them) -> p1e_map.npz -> numpy forward; torch float64 vs numpy <= 1e-4;
* the P3 rule (p3_decision) and the CLI refusals that never read data (preview-only flags, BLAS variables).
Run: .venv/Scripts/python.exe -m pytest -q tests/test_p09_map.py
"""

import builtins
import json
import os
import shutil
import subprocess
import sys
import unittest
import datetime

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import p09_forward as FW  # noqa: E402
import p09_map as M  # noqa: E402
import p09_rank_fuse as R9  # noqa: E402

ENV = {k: v for k, v in dict(os.environ, PYTHONDONTWRITEBYTECODE="1").items() if k not in R9.BLAS_ENV}
PY = sys.executable
TZ8 = datetime.timezone(datetime.timedelta(hours=8))


@pytest.fixture(autouse=True)
def _no_import_guards():
    yield
    for flag in ("_c10_hs_guard", "_c10_u1_guard"):
        assert not getattr(builtins, flag, False), f"{flag} installed inside the pytest process"
    for mod in ("p1_lib", "p1_extract", "p1_fuse", "hs_lib", "u1_lib", "pilot_fuse", "pilot_run"):
        assert mod not in sys.modules, f"{mod} imported inside the pytest process"


class TrainingPort(unittest.TestCase):
    def test_train_one_equals_explore_train(self):
        code = r'''
import sys, os, json, importlib.util
sys.dont_write_bytecode = True
os.environ["MAP_THREADS"] = "2"
sys.path.insert(0, "tools")
import numpy as np
spec = importlib.util.spec_from_file_location("p09_explore_train", "results/prereg09/explore/mapping/train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
import torch
import p09_map as M
assert torch.get_num_threads() == 2
rng = np.random.default_rng(0)
n = 700
X = rng.normal(size=(n, 1540)).astype(np.float32)
ex = rng.normal(size=(n, 44)).astype(np.float32)
y = rng.integers(0, 10, n).astype(np.int64)
rows, ev = np.arange(600), np.arange(600, 700)
out = {}
for kind, vname in (("mlpclm", "mlpclm256"), ("mlp", "mlp512"), ("lowrank", "lowrank3_ex")):
    cfg = T.V[vname]
    use_ex = kind == "lowrank"
    get = (lambda r: np.concatenate([X[r], ex[r]], 1)) if use_ex else (lambda r: X[r])
    d = 1540 + (44 if use_ex else 0)
    for seed, epochs, stop in ((0, 3, None), (1, 4, 2)):
        a, sa = T.train(cfg, X, ex if use_ex else None, y, rows, seed, epochs, stop_after=stop, eval_rows=ev)
        b, sb = M.train_one(kind, get, d, y, rows, seed, epochs, stop_after=stop, eval_rows=ev, n=n)
        da, db = a.state_dict(), b.state_dict()
        same = sorted(da) == sorted(db) and all(torch.equal(da[k], db[k]) for k in da)
        snaps = len(sa) == len(sb) and all(np.array_equal(x, z) for x, z in zip(sa, sb))
        pa = T.predict(a, X, ex if use_ex else None, ev)
        pb = M.predict(b, get, n, ev)
        out[f"{kind}_{seed}"] = [bool(same), bool(snaps), bool(np.array_equal(pa, pb)), len(sb)]
print(json.dumps(out))
'''
        r = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=REPO, env=ENV, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        res = json.loads(r.stdout.strip().splitlines()[-1])
        for k, v in res.items():
            self.assertEqual(v[:3], [True, True, True], k)
        self.assertEqual(res["mlpclm_0"][3], 3)
        self.assertEqual(res["mlpclm_1"][3], 2)


class EpochRule(unittest.TestCase):
    def test_first_minimum(self):
        y = np.array([0, 1, 2, 3])
        oof = np.full((5, 4, 10), 0.05, dtype=np.float32)
        good = [0.2, 0.5, 0.5, 0.3, 0.2]                       # p of the true class per epoch
        for e, g in enumerate(good):
            oof[e, np.arange(4), y] = g
        best, ll = R9.epoch_rule(oof, y)
        self.assertEqual(best, 2)                              # epochs 2 and 3 tie: the first one
        self.assertAlmostEqual(ll[1], -np.log(np.float32(0.5)), places=12)
        oof[4, np.arange(4), y] = 0.9
        self.assertEqual(R9.epoch_rule(oof, y)[0], 5)

    def test_fit_member_tiny(self):
        import torch
        torch.set_num_threads(2)
        rng = np.random.default_rng(3)
        n = 260
        X = rng.normal(size=(n, 1540)).astype(np.float32)
        y = rng.integers(0, 10, n).astype(np.int64)
        X[:, 0] += y.astype(np.float32)                        # a learnable signal
        players = np.array([f"p{i % 40}" for i in range(n)])
        fold = FW.player_folds(players)
        model, cv = M.fit_member({"name": "mlpclm256", "kind": "mlpclm", "seed": 0}, lambda r: X[r], 1540, y, fold,
                                 log=lambda *a: None)
        self.assertEqual(len(cv["logloss_by_epoch"]), 24)
        self.assertEqual(cv["epochs_chosen"], int(np.argmin(cv["logloss_by_epoch"])) + 1)
        self.assertAlmostEqual(cv["logloss"], min(cv["logloss_by_epoch"]), places=12)
        p = M.model_params(model, "mlpclm")
        self.assertEqual(sorted(p), sorted(FW.PARAMS["mlpclm"]))
        _m, cv2 = M.fit_member({"name": "lowrank3_ex", "kind": "lowrank", "seed": 0},
                               lambda r: np.concatenate([X[r], np.zeros((len(r), 44), np.float32)], 1), 1584, y, fold,
                               log=lambda *a: None)
        self.assertEqual((cv2["epochs_chosen"], cv2["logloss_by_epoch"]), (30, None))


def write_chunk(path, ref, games_rng, failed=None, no_opp=(), no_tgt=()):
    G = len(ref)
    means = games_rng.normal(size=(G, 2, 770)).astype(np.float32)
    counts = games_rng.integers(5, 60, size=(G, 2))
    for i in no_opp:
        counts[i, 1] = 0
    for i in no_tgt:
        counts[i, 0] = 0
    failed = np.zeros(G, dtype=bool) if failed is None else failed
    pg, ps, pt, lp, en = [], [], [], [], []
    for g in range(G):
        if failed[g]:
            continue
        for s in (0, 1):
            for t in range(int(counts[g, s])):
                pg.append(g)
                ps.append(s)
                pt.append(2 * t + s)
                lp.append(-abs(games_rng.normal(1.0, 1.0)))
                en.append(abs(games_rng.normal(2.0, 0.5)))
    np.savez(path, format=np.array(R9.R.CHUNK_FORMAT), set=np.array("probe"), meta_sha256=np.array(R9.R.NEUTRAL_META_SHA),
             ref=np.asarray(ref, dtype=np.int64), means=means, counts=counts, failed=failed,
             n_moves=counts.sum(1), pos_game=np.array(pg, np.int32), pos_side=np.array(ps, np.int8),
             pos_turn=np.array(pt, np.int16), pos_logp=np.array(lp, np.float32), pos_entropy=np.array(en, np.float32))
    return means, counts, failed


class Prep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = os.path.join(R9.TMP, f"pytest_prep_{os.getpid()}")
        shutil.rmtree(cls.d, ignore_errors=True)
        os.makedirs(cls.d)
        rng = np.random.default_rng(11)
        lv = R9.LEVELS
        cls.games = {s: [{"rank": lv[j % 10], "player_id": f"{s}{j % 13}", "game_id": f"{s}g{j}"} for j in range(30)]
                     for s in ("p1_probe", "p2_probe")}
        cls.files, cls.raw = {}, {}
        for s, specs in (("p1_probe", [(range(0, 15), [2], [3], [4]), (range(15, 30), [], [], [])]),
                         ("p2_probe", [(range(0, 15), [], [1], []), (range(15, 30), [0], [], [])]),
                         ("bank", [(range(0, 10), [], [], [])]), ("mono", [(range(10, 25), [5], [], [])])):
            os.makedirs(os.path.join(cls.d, "src", s))
            cls.files[s] = []
            for ci, (refs, fail, no_opp, no_tgt) in enumerate(specs):
                p = os.path.join(cls.d, "src", s, f"chunk_{ci:05d}.npz")
                f = np.zeros(len(refs), dtype=bool)
                f[fail] = True
                cls.raw[(s, ci)] = (list(refs), *write_chunk(p, list(refs), rng, f, no_opp, no_tgt))
                cls.files[s].append((p, None))
        cls.map_dir = os.path.join(cls.d, "map")
        cls.rec = M.build_prep(cls.map_dir,
                               [("p1_probe", cls.files["p1_probe"], cls.games["p1_probe"], [20]),
                                ("p2_probe", cls.files["p2_probe"], cls.games["p2_probe"], [])],
                               [("bank", cls.files["bank"]), ("mono", cls.files["mono"])],
                               {"probe_used": "P1+P2", "probe_sets": ["p1_probe", "p2_probe"], "limited": False,
                                "preview": True}, log=lambda *a: None)
        cls.prep = R9.Prep(cls.map_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_rows_and_labels(self):
        rec = self.rec
        # p1: chunk0 failed game 2, no target move game 4 -> 13; chunk1 dup ref 20 -> 14
        self.assertEqual(rec["sets"]["p1_probe"]["used"], 27)
        self.assertEqual(rec["sets"]["p1_probe"]["duplicates_excluded"], 1)
        self.assertEqual(rec["sets"]["p1_probe"]["unusable"], 2)
        self.assertEqual(rec["sets"]["p2_probe"]["used"], 29)
        self.assertEqual((rec["probe_rows"], rec["inner_rows"]), (56, 24))
        p = self.prep
        self.assertEqual(p.src.tolist(), ["p1_probe"] * 27 + ["p2_probe"] * 29)
        self.assertNotIn(20, p.ref[p.src == "p1_probe"].tolist())
        self.assertEqual(p.ref[:5].tolist(), [0, 1, 3, 5, 6])
        g = self.games["p1_probe"]
        self.assertEqual(p.y[2], R9.LEVELS.index(g[3]["rank"]))
        self.assertEqual(p.players[2], g[3]["rank"] + ":" + g[3]["player_id"])
        refs, means, counts, _f = self.raw[("p1_probe", 0)]
        self.assertTrue(np.array_equal(p.probe_X[2], FW.x_raw32(means, counts)[3], equal_nan=True))
        self.assertTrue(np.isnan(p.probe_X[2, 770:]).all())                   # game 3: no opponent move
        self.assertEqual(p.inner_src.tolist(), ["bank"] * 10 + ["mono"] * 14)
        self.assertEqual(p.inner_ref[10:].tolist(), [10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        self.assertEqual(sorted(rec["files_sha256"]), sorted(R9.PREP_FILES))
        self.assertEqual(self.prep.subset(["p2_probe"]).tolist(), list(range(27, 56)))

    def test_extra_features_stored(self):
        refs, means, counts, failed = self.raw[("mono", 0)]
        with np.load(self.files["mono"][0][0]) as z:
            ex = FW.ex_raw32({k: z[k] for k in ("pos_game", "pos_side", "pos_turn", "pos_logp", "pos_entropy")}, 15)
        keep = ~failed
        self.assertTrue(np.array_equal(self.prep.inner_ex[10:], ex[keep], equal_nan=True))

    def test_moments_equal_numpy(self):
        idx = self.prep.subset(["p1_probe", "p2_probe"])
        mom = self.prep.moments(idx, block=7)
        X = np.asarray(self.prep.probe_X[idx], dtype=np.float64)
        with np.errstate(all="ignore"):
            mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        self.assertTrue(np.array_equal(mom["mu_x"], mu) and np.array_equal(mom["sd_x"], sd))
        E = np.asarray(self.prep.probe_ex[idx], dtype=np.float32).astype(np.float64)
        mu, sd = np.nanmean(E, 0), np.nanstd(E, 0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        self.assertTrue(np.array_equal(mom["mu_ex"], mu, equal_nan=True) and np.array_equal(mom["sd_ex"], sd))

    def test_train_members_map_and_torch_check(self):
        folder = os.path.join(self.d, "train")
        logs = []
        mom, recs, info = M.train_members(self.prep, ["p1_probe", "p2_probe"], folder, 2, log=logs.append)
        self.assertEqual(sorted(recs), sorted(FW.member_names()))
        self.assertEqual(info["rows"], 56)
        self.assertTrue(all(not r["reused"] for r in recs.values()))
        _m2, recs2, _i2 = M.train_members(self.prep, ["p1_probe", "p2_probe"], folder, 2, log=logs.append)
        self.assertTrue(all(r["reused"] for r in recs2.values()))
        # M1: a saved member is reused only with the same training code, torch version and thread count
        spec = FW.MEMBERS[3]
        path = os.path.join(folder, f"member_{spec['name']}.npz")
        with np.load(path) as z:
            extra = json.loads(str(z["extra"]))
        self.assertEqual(sorted(extra["code_sha256"]), ["tools/p09_forward.py", "tools/p09_map.py"])
        self.assertEqual(extra["code_sha256"]["tools/p09_map.py"], R9.sha_lf(R9.path_of("tools/p09_map.py")))
        self.assertEqual(extra["threads"], 2)
        import torch
        self.assertEqual(extra["torch"], torch.__version__)
        self.assertTrue(M.member_reusable(path, spec, "P1+P2", 56, extra))
        for bad in (dict(extra, threads=3), dict(extra, torch="2.0.0+cpu"),
                    dict(extra, code_sha256=dict(extra["code_sha256"], **{"tools/p09_map.py": "0" * 64})),
                    dict(extra, code_sha256=dict(extra["code_sha256"], **{"tools/p09_forward.py": "1" * 64})),
                    dict(extra, prep_record_sha256="x")):
            self.assertFalse(M.member_reusable(path, spec, "P1+P2", 56, bad))
        self.assertFalse(M.member_reusable(path, spec, "P1+P2+P3", 56, extra))
        recs3 = M.train_members(self.prep, ["p1_probe", "p2_probe"], folder, 1, log=logs.append,
                                members=["lowrank3_ex"])[1]
        self.assertFalse(recs3["lowrank3_ex"]["reused"])                # another thread count: retrained
        M.train_members(self.prep, ["p1_probe", "p2_probe"], folder, 2, log=logs.append, members=["lowrank3_ex"])
        mp_path = os.path.join(self.d, "p1e_map.npz")
        rec = R9.assemble_map(mp_path, folder, {"probe_used": "P1+P2"})
        self.assertEqual(rec["e_star"]["lowrank3_ex"], 30)
        mp = FW.load_map(mp_path)
        self.assertTrue(np.array_equal(mp["mu_x"], mom["mu_x"]))
        lp = R9.inner_map_lp(self.prep, mp)
        self.assertEqual(lp.shape, (24, 10))
        np.testing.assert_allclose(np.exp(lp).sum(1), 1.0, atol=1e-9)
        t = M.torch_numpy_max_diff(mp_path, self.prep.inner_X, self.prep.inner_ex)
        self.assertLess(t["max_abs_lp_diff"], 1e-10)
        self.assertLess(t["float32_record_only"]["max_abs_prob_diff"], 1e-5)
        # the frozen LR path (one call per inner set) equals a direct computation
        rng = np.random.default_rng(2)
        W, b = rng.normal(size=(1540, 10)) * 0.01, rng.normal(size=10)
        lr = R9.inner_lr_lp(self.prep, W, b, mom["mu_x"], mom["sd_x"])
        ref = FW.lr_logp(W, b, FW.standardise_x(np.asarray(self.prep.inner_X, dtype=np.float64), mom["mu_x"],
                                                  mom["sd_x"]))
        np.testing.assert_allclose(lr, ref, rtol=0, atol=1e-12)
        with self.assertRaises(R9.P09Error):
            os.replace(os.path.join(folder, "member_mlp512.npz"), os.path.join(folder, "x.npz"))
            try:
                R9.assemble_map(mp_path, folder, {})
            except FileNotFoundError:
                raise R9.P09Error("missing member")
            finally:
                os.replace(os.path.join(folder, "x.npz"), os.path.join(folder, "member_mlp512.npz"))

    def test_chunk_sha_checked(self):
        p, _ = self.files["bank"][0]
        with self.assertRaises(R9.P09Error):
            M.read_chunk_rows(p, "0" * 64, ())


class ScanP3(unittest.TestCase):
    """scan_p3 on synthetic P1-format chunks in a temporary folder (R9.P3_DIR redirected; no P3 chunk is read)."""

    def setUp(self):
        self.d = os.path.join(R9.TMP, f"pytest_p3_{os.getpid()}")
        os.makedirs(self.d, exist_ok=True)
        self.old = (R9.P3_DIR, R9.P3_CHUNKS, R9.P3_GAMES)
        R9.P3_DIR, R9.P3_CHUNKS, R9.P3_GAMES = self.d, 2, 250
        code = json.dumps({"results/prereg08/p1/p1_lib.py": R9.R.PINNED["results/prereg08/p1/p1_lib.py"],
                           "results/prereg08/p1/p1_extract.py": R9.R.PINNED["results/prereg08/p1/p1_extract.py"]})
        for ci, (s, e) in enumerate(((0, 200), (200, 250))):
            G = e - s
            failed = np.zeros(G, dtype=bool)
            failed[:2] = ci == 1
            np.savez(os.path.join(self.d, f"chunk_{ci:05d}.npz"), format=np.array(R9.R.CHUNK_FORMAT),
                     set=np.array("probe"), sides=np.array(["target", "opponent"]), ref=np.arange(s, e),
                     precision=np.array("fp16"), batch=np.array(1024), channels_last=np.array(True),
                     device=np.array("cuda"), torch=np.array("2.11.0+cu128"), ckpt_sha256=np.array("x"),
                     katago_commit=np.array(R9.R.KATAGO_COMMIT), meta_sha256=np.array(R9.R.NEUTRAL_META_SHA),
                     code_sha256=np.array(code), created_utc=np.array("2026-10-09T00:00:00+00:00"),
                     means=np.zeros((G, 2, 770), np.float32), counts=np.ones((G, 2), np.int32),
                     n_moves=np.full(G, 2), failed=failed)
        self.games = [{"ref": j} for j in range(250)]

    def tearDown(self):
        R9.P3_DIR, R9.P3_CHUNKS, R9.P3_GAMES = self.old
        shutil.rmtree(self.d, ignore_errors=True)

    def test_complete_and_refs(self):
        s = M.scan_p3(self.games)
        self.assertTrue(s["complete"], s)
        self.assertEqual((s["chunks_present"], s["totals"]["games"], s["totals"]["failed"]), (2, 250, 2))
        s = M.scan_p3([{"ref": j + (1 if j == 5 else 0)} for j in range(250)])
        self.assertFalse(s["refs_match_sets"] or s["complete"])
        open(os.path.join(self.d, "chunk_00007.npz"), "wb").close()
        s = M.scan_p3(self.games)
        self.assertFalse(s["complete"])
        self.assertTrue(s["extra_files"])


class P3Rule(unittest.TestCase):
    def test_decision(self):
        before = datetime.datetime(2026, 10, 10, 12, 0, tzinfo=TZ8)
        after = datetime.datetime(2026, 10, 11, 0, 0, 1, tzinfo=TZ8)
        with self.assertRaises(R9.P09Error):
            R9.p3_decision(599, 600, True, 0, before)
        with self.assertRaises(R9.P09Error):
            R9.p3_decision(600, 600, False, 0, before)               # a chunk fails the check: wait
        d = R9.p3_decision(600, 600, True, 600, before)
        self.assertTrue(d["used"] and d["complete"])
        self.assertFalse(R9.p3_decision(600, 600, True, 601, before)["used"])
        d = R9.p3_decision(310, 600, True, 0, after)
        self.assertFalse(d["used"])
        self.assertIn("incomplete", d["reason"])
        self.assertFalse(R9.p3_decision(600, 600, False, 0, after)["used"])
        self.assertEqual(R9.P3_CUTOFF.isoformat(), "2026-10-10T23:59:59+08:00")
        self.assertEqual((R9.P3_CHUNKS, R9.P3_GAMES, R9.P3_FAIL_MAX), (600, 120000, 600))

    def test_probe_sets(self):
        self.assertEqual(R9.probe_sets_from_manifest({"p3": {"used": True}}), list(R9.PROBE_SETS))
        self.assertEqual(R9.probe_sets_from_manifest({"p3": {"used": False}}), ["p1_probe", "p2_probe"])
        self.assertEqual(R9.probe_used_label(list(R9.PROBE_SETS)), "P1+P2+P3")


class Cli(unittest.TestCase):
    def run_cli(self, args, env=None):
        return subprocess.run([PY, os.path.join(REPO, "tools", "p09_map.py")] + args, capture_output=True, text=True,
                              cwd=REPO, env=env or ENV, timeout=600)

    def test_preview_only_flags_refused_in_real_mode(self):
        for args in (["train", "--threads", "2"], ["train", "--members", "mlp512"], ["selfcheck", "--threads", "2"],
                     ["prep", "--limit-chunks", "3"], ["prep", "--probe", "P1+P2"]):
            with self.subTest(args):
                r = self.run_cli(args)
                self.assertEqual(r.returncode, 2, r.stderr[-1500:])
                self.assertIn("ERROR", r.stderr)

    def test_blas_variables_refused(self):
        r = self.run_cli(["train"], env=dict(ENV, OMP_NUM_THREADS="4"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("BLAS thread variables", r.stderr)

    def test_out_dir_must_be_under_tmp(self):
        for od in (os.path.join(REPO, "results", "prereg09", "rank"), R9.EXPLORE_TMP, R9.TMP,
                   os.path.join(REPO, "data", "prereg08", "tmp", "x")):
            with self.subTest(od):
                r = self.run_cli(["train", "--out-dir", od])
                self.assertEqual(r.returncode, 2, r.stderr[-1500:])

    def test_status_reads_nothing(self):
        r = self.run_cli(["status"])
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        out = json.loads(r.stdout)
        self.assertEqual(out["p3_chunks_expected"], 600)


if __name__ == "__main__":
    unittest.main()
