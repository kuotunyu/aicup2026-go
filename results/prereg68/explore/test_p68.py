"""Small tests of p68_eval.py / p68_manifest.py / p68_embed.py (synthetic data and metadata only: no pool, no score,
no E number). Run: .venv/Scripts/python.exe -m pytest -q results/prereg68/explore/test_p68.py"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p68_eval as M  # noqa: E402
import p68_manifest as MF  # noqa: E402
import p68_embed as EMB  # noqa: E402

P67 = M.P67
REAL_EVER = M.ever_committed
BASE = list(M.BASE)


class Weights(unittest.TestCase):
    def test_order_and_weights(self):
        self.assertEqual(list(M.ORDER), ["c49", "S", "B", "p61", "S1", "S2", "X", "X2", "X3"])
        self.assertEqual(M.weights_for(M.ORDER), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.assertEqual(M.weights_for(BASE)[-2:], [0.0, 0.0])
        self.assertEqual(M.WE_ENS, 14.0)
        self.assertEqual((M.ADOPT_MEAN, M.NEED_POOLS, M.BOOT, M.BOOT_SEED), (0.005, 16, 10000, 43))

    def test_unknown_member(self):
        with self.assertRaises(SystemExit):
            M.weights_for(["c49", "Y"])

    def test_zero_weight_new_members_equal_v29(self):
        """X2 = X3 = 0 -> the ensemble is prereg67's 7-member ensemble bit for bit."""
        r = np.random.default_rng(0)
        Z = {m: r.normal(size=(30, 50)) for m in M.ORDER}
        want = P67.ensemble([Z[m] for m in BASE], [P67.WEIGHTS[m] for m in BASE])
        w = M.weights_for(BASE)
        on = [(m, wi) for m, wi in zip(M.ORDER, w) if wi]
        got = P67.ensemble([Z[m] for m, _ in on], [wi for _, wi in on])
        self.assertTrue(np.array_equal(got, want))
        self.assertTrue(np.array_equal(P67.ensemble([Z[m] for m in M.ORDER], w), want))

    def test_nine_weighted_mean(self):
        r = np.random.default_rng(1)
        Z = [r.normal(size=(20, 40)) for _ in M.ORDER]
        w = M.weights_for(M.ORDER)
        want = sum(wi * z for wi, z in zip(w, Z)) / 15.0
        self.assertTrue(np.allclose(P67.ensemble(Z, w), want, rtol=0, atol=1e-12))


# ---------------------------------------------------------------- version (audit item 3)
SH = {"a": "1"}
SH2 = {"a": "2"}


def scored(p, shas=SH):
    return {"status": "scored", "pass": p, "files_sha256": shas, "before_cutoff": True}


def disk(present=True, before=True, shas=SH):
    return {"present": present, "before_cutoff": before, "shas": shas if present else None}


class Version(unittest.TestCase):
    def pv(self, ent, dk, past):
        return M.pick_version(ent, dk, past)

    def test_cutoff_is_taipei(self):
        self.assertEqual(M.CUTOFF.utcoffset().total_seconds(), 8 * 3600)
        self.assertEqual(M.CUTOFF.isoformat(), "2026-11-24T12:00:00+08:00")

    def test_nine(self):
        m, p, w, _s = self.pv({"X2": scored(True), "X3": scored(True)}, {"X2": disk(), "X3": disk()}, False)
        self.assertEqual((M.version_name(m), p, w), ("9", [], []))

    def test_passed_then_changed_is_problem(self):
        m, p, w, s = self.pv({"X2": scored(True), "X3": scored(True)}, {"X2": disk(shas=SH2), "X3": disk()}, False)
        self.assertTrue(p and s["X2"] == "problem")

    def test_passed_then_deleted_is_problem(self):
        _m, p, _w, _s = self.pv({"X2": scored(True), "X3": scored(True)},
                                {"X2": disk(), "X3": disk(present=False)}, True)
        self.assertTrue(p)

    def test_below_gate_other_in_at_once(self):
        m, p, w, s = self.pv({"X2": scored(True), "X3": scored(False)}, {"X2": disk(), "X3": disk()}, False)
        self.assertEqual((M.version_name(m), p, w, s["X3"]), ("8-X2", [], [], "out-below-gate"))

    def test_not_ready_wait_then_timeout(self):
        m, p, w, _s = self.pv({"X2": scored(True)}, {"X2": disk(), "X3": disk(present=False)}, False)
        self.assertEqual((M.version_name(m), w), ("8-X2", ["X3"]))
        m, p, w, s = self.pv({"X2": scored(True)}, {"X2": disk(), "X3": disk(present=False)}, True)
        self.assertEqual((M.version_name(m), p, w, s["X3"]), ("8-X2", [], [], "out-timeout"))

    def test_present_unchecked_is_problem(self):
        _m, p, _w, _s = self.pv({"X2": scored(True)}, {"X2": disk(), "X3": disk()}, False)
        self.assertTrue(p)
        _m, p, _w, _s = self.pv({"X2": scored(True)}, {"X2": disk(), "X3": disk()}, True)
        self.assertTrue(p)            # in time but never checked: stop even after the cutoff

    def test_late_files_timeout(self):
        m, p, _w, s = self.pv({"X2": scored(True)}, {"X2": disk(), "X3": disk(before=False)}, True)
        self.assertEqual((M.version_name(m), p, s["X3"]), ("8-X2", [], "out-timeout"))

    def test_invalid_waits_for_repair(self):
        inv = {"status": "invalid", "pass": None, "files_sha256": SH, "problems": ["bad manifest"]}
        m, p, w, _s = self.pv({"X2": scored(True), "X3": inv}, {"X2": disk(), "X3": disk()}, False)
        self.assertEqual((M.version_name(m), p, w), ("8-X2", [], ["X3"]))
        _m, p, _w, _s = self.pv({"X2": scored(True), "X3": inv}, {"X2": disk(), "X3": disk(present=False)}, False)
        self.assertEqual(_w, ["X3"])  # deleted before the cutoff: still repairable
        _m, p, _w, _s = self.pv({"X2": scored(True), "X3": inv}, {"X2": disk(), "X3": disk(shas=SH2)}, False)
        self.assertTrue(p)            # repaired files must be checked again

    def test_known_invalid_stays_problem_after_cutoff(self):
        inv = {"status": "invalid", "pass": None, "files_sha256": SH, "problems": ["bad manifest"]}
        for dk in (disk(), disk(present=False), disk(before=False, shas=SH2)):
            m, p, w, s = self.pv({"X2": scored(True), "X3": inv}, {"X2": disk(), "X3": dk}, True)
            self.assertEqual((s["X3"], w), ("problem", []))
            self.assertTrue(p and "X3" in p[0] and "bad manifest" in p[0])

    def test_none(self):
        m, _p, w, _s = self.pv({}, {"X2": disk(False), "X3": disk(False)}, True)
        self.assertEqual((m, w), (BASE, []))
        self.assertTrue(M.version_name(m).startswith("7"))


class Rule(unittest.TestCase):
    def test_adopt(self):
        old = np.linspace(0.70, 0.75, 20)
        self.assertTrue(M.judge_rule(old + 0.006, old)["adopt"])
        self.assertFalse(M.judge_rule(old + 0.004, old)["adopt"])

    def test_pools(self):
        old = np.zeros(20)
        new = old + 0.02
        new[:5] = -0.001          # 15 / 20 higher
        r = M.judge_rule(new, old)
        self.assertEqual(r["pools_higher"], 15)
        self.assertFalse(r["adopt"])
        new[4] = 0.001            # 16 / 20
        self.assertTrue(M.judge_rule(new, old)["adopt"])


# ---------------------------------------------------------------- E use from committed files (audit item 1)
def git(d, *a):
    return subprocess.run(["git", "-C", d] + list(a), capture_output=True, text=True, check=True)


def put_eval(d, name, n):
    os.makedirs(os.path.join(d, "results", name, "explore"), exist_ok=True)
    with open(os.path.join(d, "results", name, "explore", "eval.json"), "w") as f:
        json.dump({"e_pools_use_number": n}, f)


class EUse(unittest.TestCase):
    def test_committed_only(self):
        with tempfile.TemporaryDirectory() as d:
            git(d, "init", "-q")
            git(d, "config", "user.email", "t@t")
            git(d, "config", "user.name", "t")
            for name, n in (("prereg01", 4), ("prereg02", 5), ("prereg68", 6)):
                put_eval(d, name, n)
            git(d, "add", "-A")
            git(d, "commit", "-qm", "x")
            u = M.e_uses_so_far(d)
            self.assertEqual((u["count"], u["consistent"]), (5, True))
            put_eval(d, "prereg03", 6)                      # uncommitted fake with the next number
            self.assertEqual(M.e_uses_so_far(d)["count"], 5)
            with open(os.path.join(d, "results", "prereg02", "explore", "eval.json"), "w") as f:
                json.dump({"e_pools_use_number": 9}, f)      # an uncommitted edit is not read either
            self.assertEqual(M.e_uses_so_far(d)["count"], 5)


# ---------------------------------------------------------------- one judgement (audit item 1)
class Judge(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.calls = {"new": 0}
        lock = threading.Lock()
        calls = self.calls

        def e_scores(ctx, E, wE):
            if E == "NEW":
                with lock:
                    calls["new"] += 1
                if getattr(self, "crash", False):
                    raise RuntimeError("killed")
            return [0.7] * 20, {}

        ev29 = {"per_pool": {"ens": [0.7] * 20}}
        self.patches = [
            mock.patch.object(M, "EVAL_JSON", os.path.join(self.d, "eval.json")),
            mock.patch.object(M, "JUDGE_STARTED", os.path.join(self.d, "judge_started.json")),
            mock.patch.object(M, "JUDGE_FAILED", os.path.join(self.d, "judge_failed.json")),
            mock.patch.object(M, "MEMBERS_JSON", os.path.join(self.d, "members.json")),
            mock.patch.object(M, "SEAM_JSON", os.path.join(self.d, "members.json")),
            mock.patch.object(M, "ever_committed", lambda p, repo=None: False),
            mock.patch.object(M, "commit_start", lambda n: True),
            mock.patch.object(M, "judge_preflight", lambda a: (ev29, {"count": 11}, BASE + ["X2"], {}, {})),
            mock.patch.object(M, "e_ensembles", lambda *a, **k: (("BASE", "NEW"), [np.ones(3, bool)], {})),
            mock.patch.object(M, "member_shas", lambda m, e, ev: {}),
            mock.patch.object(M, "v29_E_check", lambda per, ev: True),
            mock.patch.object(M.P67, "e_context", lambda: {"cfg": {}}),
            mock.patch.object(M.P67, "cfg_ens", lambda c: {}),
            mock.patch.object(M.P67, "e_scores", e_scores),
            mock.patch.object(M.EV, "mean_by_k", lambda b: {}),
            mock.patch.object(M.L, "git_head", lambda: "h"),
            mock.patch.object(M.P63, "peak_mb", lambda: 0.0),
        ]
        with open(os.path.join(self.d, "members.json"), "w") as f:
            f.write("{}")
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def run_judge(self):
        return M.cmd_judge(argparse.Namespace(e_use=12))

    def test_twice(self):
        self.assertEqual(self.run_judge(), 0)
        with self.assertRaises(SystemExit):
            self.run_judge()
        self.assertEqual(self.calls["new"], 1)

    def test_crash_then_rerun(self):
        self.crash = True
        with self.assertRaises(RuntimeError):
            self.run_judge()
        self.assertTrue(os.path.exists(M.JUDGE_STARTED) and os.path.exists(M.JUDGE_FAILED))
        self.crash = False
        with self.assertRaises(SystemExit):
            self.run_judge()
        self.assertEqual(self.calls["new"], 1)
        self.assertFalse(os.path.exists(M.EVAL_JSON))

    def test_two_processes(self):
        out = []
        barrier = threading.Barrier(2)

        def go():
            barrier.wait()
            try:
                out.append(self.run_judge())
            except SystemExit as e:
                out.append(str(e))
        ts = [threading.Thread(target=go) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(self.calls["new"], 1)
        self.assertEqual(sum(1 for x in out if x == 0), 1)

    def test_commit_failure_stops_before_score(self):
        with mock.patch.object(M, "commit_start", lambda n: False):
            with self.assertRaises(SystemExit):
                self.run_judge()
        self.assertEqual(self.calls["new"], 0)
        self.assertTrue(os.path.exists(M.JUDGE_STARTED))
        with self.assertRaises(SystemExit):            # the same start record blocks a second entry
            self.run_judge()
        self.assertEqual(self.calls["new"], 0)
        self.assertFalse(os.path.exists(M.EVAL_JSON))

    def test_commit_success_scores_once(self):
        self.assertEqual(self.run_judge(), 0)
        self.assertEqual(self.calls["new"], 1)

    def test_git_query_failure_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:          # not a git repository: an error, not 'never committed'
            with self.assertRaises(SystemExit):
                REAL_EVER(os.path.join(d, "judge_started.json"), d)
            with self.assertRaises(SystemExit):
                M.e_uses_so_far(d)

    def test_committed_history_blocks(self):
        with mock.patch.object(M, "ever_committed", lambda p, repo=None: p == M.JUDGE_STARTED):
            with self.assertRaises(SystemExit):
                self.run_judge()
        self.assertEqual(self.calls["new"], 0)

    def test_no_out_option(self):
        with self.assertRaises(SystemExit):
            M.main(["judge", "--e-use", "12", "--out", os.path.join(self.d, "other.json")])
        with self.assertRaises(SystemExit):
            M.main(["apply", "--src", "x", "--pool", "public", "--eval", "x.json"])
        with self.assertRaises(SystemExit):
            M.main(["check-member", "--name", "X2", "--dir", "data/prereg62/X/emb"])


# ---------------------------------------------------------------- member identity (audit item 2)
class Identity(unittest.TestCase):
    def setUp(self):
        self.x = M.read_json(M.X_RECORD)
        self.xsha = self.x["checkpoints"]["step_400000.pt"]

    def fake(self, seed, sha):
        r = json.loads(json.dumps(self.x))
        r["config"]["p60"]["seed"] = seed
        r["checkpoints"] = {"step_400000.pt": sha}
        return r

    def test_good(self):
        self.assertEqual(M.identity_problems("X2", self.fake(1, "a" * 64), self.x, "a" * 64), [])
        self.assertEqual(M.identity_problems("X3", self.fake(2, "b" * 64), self.x, "b" * 64), [])

    def test_old_X_renamed(self):
        self.assertTrue(M.identity_problems("X2", self.x, self.x, self.xsha))        # seed 0
        self.assertTrue(M.distinct_problems(self.xsha, {"X(record)": self.xsha}))

    def test_wrong_ckpt_and_seed(self):
        self.assertTrue(M.identity_problems("X2", self.fake(1, "a" * 64), self.x, "c" * 64))
        self.assertTrue(M.identity_problems("X2", self.fake(2, "a" * 64), self.x, "a" * 64))
        self.assertTrue(M.identity_problems("X2", self.fake(1, "a" * 64), self.x, None))

    def test_recipe_change(self):
        r = self.fake(1, "a" * 64)
        r["config"]["p60"]["T_sha256_sorted_ids"] = "0" * 64
        self.assertTrue(M.identity_problems("X2", r, self.x, "a" * 64))
        s1 = M.read_json(os.path.join(M.REPO, "results", "prereg62", "explore", "train_record_S1.json"))
        self.assertTrue(M.identity_problems("X2", s1, self.x, s1["checkpoints"]["step_200000.pt"]))

    def test_x2_equals_x3(self):
        self.assertEqual(M.distinct_problems("a" * 64, {"X3": "a" * 64}), ["checkpoint sha256 equals X3's"])

    def test_arrays(self):
        self.assertEqual(M.array_problems("f", np.ones((3, 2)), np.array([1, 2, 3])), [])
        self.assertTrue(M.array_problems("f", np.array([[1.0, np.nan]]), np.array([1])))
        self.assertTrue(M.array_problems("f", np.ones((2, 2)), np.array([1, 0])))
        self.assertTrue(M.array_problems("f", np.ones((2, 2)), np.array([1])))


# ---------------------------------------------------------------- per-game manifest (audit item 2)
def sgf(moves):
    return "(;SZ[19]HD[0]" + "".join(";%s[%s]" % m for m in moves) + ")"


G = {k: sgf(v) for k, v in {
    "a": [("B", "pd"), ("W", "dd"), ("B", "pp")],
    "b": [("B", "dp"), ("W", "pp"), ("B", "dd"), ("W", "tt")],
    "c": [("B", "qd"), ("W", "cc"), ("B", "qq"), ("W", "dq")],
    "d": [("B", "jj"), ("W", "kk"), ("B", "ll"), ("W", "mm")],       # same colour / length as c, other moves
    "e": [("B", "cd"), ("W", "dc")]}.items()}


def write_pool(src, pool, cand_games, questions):
    pre = MF.POOL_PREFIX[pool]
    with open(os.path.join(src, "player_identification_candidates2_%s.csv" % pool), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "player_id", "sgf_content", "color"])
        for i, (pid, g, c) in enumerate(cand_games):
            w.writerow(["%sg_%07d" % (pre, i), pre + "p_" + pid, G[g], c])
    with open(os.path.join(src, "player_identification_test2_%s.csv" % pool), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "num_games", "sgf_1", "sgf_2", "color_1", "color_2"])
        for qid, gs in questions:
            row = [pre + "q_" + qid, len(gs)] + [G[g] for g, _ in gs] + [""] * (2 - len(gs)) + \
                  [c for _, c in gs] + [""] * (2 - len(gs))
            w.writerow(row)


class Manifest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.src = os.path.join(self.d, "tests")
        os.makedirs(self.src)
        cand = [("0001", "a", "B"), ("0002", "b", "W"), ("0001", "e", "W")]
        qs = [("0001", [("c", "W"), ("d", "W")]), ("0002", [("a", "B")])]
        for pool in ("public", "private"):
            write_pool(self.src, pool, cand if pool == "public" else cand[::-1], qs)
        self.ckpt = os.path.join(self.d, "step_400000.pt")
        with open(self.ckpt, "wb") as f:
            f.write(b"ckpt X2")
        self.ck = MF.sha_file(self.ckpt)
        self.p = mock.patch.object(EMB, "out_path", lambda arm, name: os.path.join(self.d, "emb", name + ".npz"))
        self.p.start()

    def tearDown(self):
        self.p.stop()

    def stub(self, ck_sha=None, permute=None):
        def embed_main(argv):
            import p36_embed
            src, pool = argv[argv.index("--src") + 1], argv[argv.index("--pool") + 1]
            _codes, start, end, is_white = p36_embed.pool_games(src, pool)
            rows = MF.rebuild(src, pool)
            cnt = np.array([r[6] for r in rows], np.int32)
            emb = np.arange(len(cnt) * 4, dtype=np.float32).reshape(len(cnt), 4)
            if permute is not None:
                emb, cnt = emb[permute], cnt[permute]
            out = EMB.out_path("X2", "pool_" + pool)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out, emb=emb, count=cnt, ckpt_sha256=np.array(ck_sha or self.ck),
                     ckpt_step=np.array(400000), embed_cfg=np.array("{}"))
            return 0
        return embed_main

    def argv(self, pool):
        return ["--arm", "X2", "--ckpt", self.ckpt, "--workers", "2", "--source", "pool", "--src", self.src,
                "--pool", pool]

    def test_counts_are_target_moves(self):
        rows = MF.rebuild(self.src, "public")
        self.assertEqual([r[6] for r in rows], [2, 1, 1, 2, 2, 2])     # pass 'tt' not counted
        self.assertEqual(rows[2][:4], ["public", "cand", "pub2_p_0001", 1])

    def test_good_and_rerun(self):
        self.assertEqual(EMB.run(self.argv("public"), self.stub()), 0)
        npz = EMB.out_path("X2", "pool_public")
        self.assertEqual(MF.verify_pool(npz, self.src, "public", self.ck), [])
        self.assertEqual(EMB.run(self.argv("public"), self.stub()), 0)          # exists + valid sidecar
        self.assertFalse(os.path.exists(npz + ".pending.json"))

    def test_swapped_pool_rejected(self):
        EMB.run(self.argv("public"), self.stub())
        self.assertTrue(MF.verify_pool(EMB.out_path("X2", "pool_public"), self.src, "private", self.ck))

    def test_swapped_same_colour_same_length_games_rejected(self):
        EMB.run(self.argv("public"), self.stub())
        npz = EMB.out_path("X2", "pool_public")
        write_pool(self.src, "public", [("0001", "a", "B"), ("0002", "b", "W"), ("0001", "e", "W")],
                   [("0001", [("d", "W"), ("c", "W")]), ("0002", [("a", "B")])])
        self.assertTrue(MF.verify_pool(npz, self.src, "public", self.ck))

    def test_rows_swapped_in_file_rejected(self):
        EMB.run(self.argv("public"), self.stub())
        npz = EMB.out_path("X2", "pool_public")
        with np.load(npz) as z:
            d = {k: z[k] for k in z.files}
        d["emb"] = d["emb"][[0, 1, 2, 4, 3, 5]]
        np.savez(npz, **d)
        self.assertTrue(MF.verify_pool(npz, self.src, "public", self.ck))

    def test_other_checkpoint_rejected(self):
        with self.assertRaises(SystemExit):
            EMB.run(self.argv("public"), self.stub(ck_sha="f" * 64))
        self.assertFalse(os.path.exists(MF.sidecar_path(EMB.out_path("X2", "pool_public"))))
        EMB.run(self.argv("private"), self.stub())
        self.assertTrue(MF.verify_pool(EMB.out_path("X2", "pool_private"), self.src, "private", "e" * 64))

    def test_wrong_order_counts_rejected(self):
        with self.assertRaises(SystemExit):
            EMB.run(self.argv("public"), self.stub(permute=[1, 0, 2, 3, 4, 5]))

    def test_npz_without_sidecar_refused(self):
        npz = EMB.out_path("X2", "pool_public")
        os.makedirs(os.path.dirname(npz))
        np.savez(npz, emb=np.zeros((6, 4)), count=np.ones(6), ckpt_sha256=np.array(self.ck),
                 ckpt_step=np.array(400000))
        with self.assertRaises(SystemExit):
            EMB.run(self.argv("public"), self.stub())

    def test_planted_foreign_part_refused(self):
        npz = EMB.out_path("X2", "pool_public")
        os.makedirs(npz + ".parts")
        np.savez(os.path.join(npz + ".parts", "part_00000.npz"), emb=np.zeros((6, 4), np.float32),
                 count=np.array([2, 1, 1, 2, 2, 2], np.int32))
        with open(npz + ".pending.json", "w") as f:       # even with a matching-looking binding
            json.dump({"manifest_sha256": MF.manifest_sha(MF.rebuild(self.src, "public")), "ckpt_sha256": self.ck,
                       "pool": "public"}, f)
        called = []
        with self.assertRaises(SystemExit):
            EMB.run(self.argv("public"), lambda av: called.append(1) or 0)
        self.assertEqual(called, [])                        # the embedder never ran on the foreign part
        self.assertFalse(os.path.exists(MF.sidecar_path(npz)))

    def store_stub(self):
        def embed_main(argv):
            out = EMB.out_path("X2", "store_E")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out, emb=np.ones((3, 4), np.float32), count=np.array([1, 2, 3], np.int32),
                     ckpt_sha256=np.array(self.ck), ckpt_step=np.array(400000), embed_cfg=np.array("{}"))
            return 0
        return embed_main

    def store_argv(self):
        return ["--arm", "X2", "--ckpt", self.ckpt, "--workers", "2", "--source", "store", "--players", "E"]

    def edit_side(self, npz, **kw):
        side = MF.sidecar_path(npz)
        with open(side) as f:
            d = json.load(f)
        for k, v in kw.items():
            if v is None:
                d.pop(k, None)
            else:
                d[k] = v
        with open(side, "w") as f:
            json.dump(d, f)

    def check_all(self, npz, kind):
        if kind == "pool":
            return MF.verify_pool(npz, self.src, "public", self.ck)
        return MF.verify_store(npz, self.ck)

    def made(self, kind):
        if kind == "pool":
            EMB.run(self.argv("public"), self.stub())
            return EMB.out_path("X2", "pool_public"), self.argv("public")
        EMB.run(self.store_argv(), self.store_stub())
        return EMB.out_path("X2", "store_E"), self.store_argv()

    def test_cold_start_required(self):
        """audit 1122e: existing vector files are accepted only with a p68_embed.py provenance, cold_start true."""
        for kind in ("pool", "store"):
            with self.subTest(kind=kind):
                for f in os.listdir(self.d):
                    if f == "emb":
                        import shutil
                        shutil.rmtree(os.path.join(self.d, f))
                npz, argv = self.made(kind)
                self.assertEqual(self.check_all(npz, kind), [])
                self.edit_side(npz, cold_start=False)
                self.assertTrue(any("cold_start" in x for x in self.check_all(npz, kind)))
                with self.assertRaises(SystemExit):          # existing-npz branch of the embed entry too
                    EMB.run(argv, lambda av: 0)
                self.edit_side(npz, cold_start=None)         # absent (an old-format record)
                self.assertTrue(any("cold_start" in x for x in self.check_all(npz, kind)))
                with self.assertRaises(SystemExit):
                    EMB.run(argv, lambda av: 0)
                self.edit_side(npz, cold_start=True)
                self.assertEqual(self.check_all(npz, kind), [])
                with np.load(npz) as z:                      # npz replaced after the provenance was written
                    d = {k: z[k] for k in z.files}
                d["emb"] = d["emb"] + 1
                np.savez(npz, **d)
                self.assertTrue(any("sha256 differs" in x for x in self.check_all(npz, kind)))
                os.remove(MF.sidecar_path(npz))              # provenance missing
                self.assertTrue(any("no provenance" in x for x in self.check_all(npz, kind)))
                with self.assertRaises(SystemExit):
                    EMB.run(argv, lambda av: 0)

    def test_count_must_be_integer(self):
        rows = MF.rebuild(self.src, "public")
        good = np.array([r[6] for r in rows], np.int32)
        self.assertTrue(MF.count_check(good, rows)["ok"])
        self.assertFalse(MF.count_check(good.astype(np.float64) + 0.9, rows)["ok"])

    def test_parts_without_binding_refused(self):
        os.makedirs(EMB.out_path("X2", "pool_public") + ".parts")
        with self.assertRaises(SystemExit):
            EMB.run(self.argv("public"), self.stub())

    def test_items_match(self):
        rows = MF.rebuild(self.src, "public")
        import p35_apply as pa
        import p36_lib as L36
        items = L36.pool_items(*pa.read_pool(self.src, "public"))[0]
        self.assertTrue(M.items_match_manifest(items, rows))
        self.assertFalse(M.items_match_manifest(items[::-1], rows))


# ---------------------------------------------------------------- apply / assemble (audit item 4)
HEAD = b"question_id,top_1,top_2,top_3,top_4,top_5\n"


def top5(pre, n, shift=0):
    return HEAD + b"".join(("%sq_%04d,%s\n" % (pre, i, ",".join("%sp_%04d" % (pre, (i + j + shift) % 9 + 1)
                                                                for j in range(5)))).encode() for i in range(n))


class Assemble(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.out = os.path.join(self.d, "apply")
        self.v29 = os.path.join(self.d, "v29")
        os.makedirs(self.v29)
        self.orig = b"".join(b"pub_q_%04d,a,b,c,d,e\n" % i for i in range(400)) + \
            b"".join(b"priv_q_%04d,a,b,c,d,e\n" % i for i in range(400))
        with open(os.path.join(self.v29, "submission_player.csv"), "wb") as f:
            f.write(HEAD + self.orig + top5("pub2_", 400)[len(HEAD):] + top5("priv2_", 400)[len(HEAD):])
        self.ev = {"version": "9", "members": list(M.ORDER), "e_pools_use_number": 12}
        for pool, pre in (("public", "pub2_"), ("private", "priv2_")):
            os.makedirs(os.path.join(self.out, pool))
            self.write_pool(pool, top5(pre, 400, 1))
        self.orig_used = self.orig

        def fake_assemble(a):
            rows = [HEAD, self.orig_used]
            for pool in ("public", "private"):
                with open(os.path.join(a.out, pool, "top5.csv"), "rb") as f:
                    rows.append(f.read()[len(HEAD):])
            data = b"".join(rows)
            with open(os.path.join(a.out, "submission_player.csv"), "wb") as f:
                f.write(data)
            n = data.count(b"\n") - 1
            with open(os.path.join(a.out, "assemble_report.json"), "w") as f:
                json.dump({"rows": n, "ok": True, "sha256": "x"}, f)
            return 0
        self.ps = [mock.patch.object(M, "V29_APPLY", self.v29),
                   mock.patch.object(M, "adopted_eval", lambda: (self.ev, "EVSHA")),
                   mock.patch.object(M.E43, "cmd_assemble", fake_assemble)]
        for p in self.ps:
            p.start()

    def tearDown(self):
        for p in self.ps:
            p.stop()

    def write_pool(self, pool, data, **over):
        top = os.path.join(self.out, pool, "top5.csv")
        with open(top, "wb") as f:
            f.write(data)
        rep = {"format": M.APPLY_FORMAT, "pool": pool, "version": "9", "members": list(M.ORDER),
               "eval_sha256": "EVSHA", "e_pools_use_number": 12, "top5_sha256": M.L.sha256_file(top),
               "v29_top5_equals_prereg67_apply": True}
        rep.update(over)
        with open(os.path.join(self.out, pool, "apply_report.json"), "w") as f:
            json.dump(rep, f)

    def run_asm(self):
        return M.cmd_assemble(argparse.Namespace(v1="v1", src="src", out=self.out))

    def final(self):
        return os.path.exists(os.path.join(self.out, "submission_player.csv"))

    def test_good(self):
        self.assertEqual(self.run_asm(), 0)
        self.assertTrue(self.final())

    def test_one_pool_old_top5(self):
        self.write_pool("private", top5("priv2_", 400, 1))
        with open(os.path.join(self.out, "private", "top5.csv"), "wb") as f:
            f.write(top5("priv2_", 400))            # v2-9's top5 put back after apply
        self.assertEqual(self.run_asm(), 1)
        self.assertFalse(self.final())

    def test_stale_report(self):
        self.write_pool("public", top5("pub2_", 400, 1), eval_sha256="OLD")
        self.assertEqual(self.run_asm(), 1)
        self.assertFalse(self.final())
        self.write_pool("public", top5("pub2_", 400, 1), version="8-X2", members=BASE + ["X2"])
        self.assertEqual(self.run_asm(), 1)

    def test_original_row_changed(self):
        self.assertEqual(self.run_asm(), 0)              # an earlier good final ...
        self.orig_used = self.orig.replace(b"pub_q_0007,a,b,c,d,e", b"pub_q_0007,b,a,c,d,e")
        self.assertEqual(self.run_asm(), 1)
        self.assertFalse(self.final())                   # ... is not left behind as if current


if __name__ == "__main__":
    unittest.main()
