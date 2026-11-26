"""Tests for tools/t02_player.py (competition 1 on the test pools, PREREG_02 rev 5 9.2 / 9.4 / 9.5).
Synthetic pools only; nothing here reads a validation or test file. The --smoke seam is checked
against the frozen tools/p02_player_fuse.py select / apply run on the same synthetic pool, so the
per-pool replication in t02_player.py is shown equal to cmd_select / cmd_apply. check_submission is
mocked (its official mode belongs to another implementer). Run:
    python -m pytest tests/test_t02_player.py
"""

import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import p02_player_fuse as pf  # noqa: E402
import p02_player_open as po  # noqa: E402
import t02_player as tp  # noqa: E402

LETTERS = "abcdefghijklmnopqrs"
NEEDLE = "ans" + "wers"
W_SMOKE = "5" * 64
PREFIXES = {"seam_val": ("val_pq_", "val_p_", "val_g_"), "public": ("pub_pq_", "pub_p_", "pub_g_"),
            "private": ("priv_pq_", "priv_p_", "priv_g_")}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sgf_of(mv):
    return "(;SZ[19]" + "".join(";%s[%s]" % (c, p) for c, p in mv) + ")"


def write_pool(qpath, cpath, ddir, pool, n_players=10, n_games=100, seed=0, weights=W_SMOKE, short=None):
    """Candidate file, question file and the two notebook dumps of one synthetic pool (ids with the
    pool's prefixes). short: a player index that gets 99 games."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    qp, pp, gp = PREFIXES[pool]
    pts = [a + b for a in LETTERS for b in LETTERS]
    players = ["%s%02d" % (pp, i) for i in range(n_players)]
    style = {p: rng.sample(pts, 12) for p in players}
    centre = {p: nrng.normal(size=16) for p in players}

    def game(p):
        colour = rng.choice("BW")
        opening = [x for x in style[p] if rng.random() < 0.7]
        rest = [x for x in rng.sample(pts, 60) if x not in opening]
        seq = (opening + rest)[:50]
        mv = [("B" if i % 2 == 0 else "W", x) for i, x in enumerate(seq)]
        if colour == "W":
            mv = [("B", rest[-1])] + [("W" if c == "B" else "B", x) for c, x in mv]
        return colour, sgf_of(mv)

    def emb(p):
        nw = int(nrng.integers(20, 60))
        return (centre[p] + nrng.normal(scale=6.0, size=16)).astype(np.float32) * nw, nw

    cand, cg = [], {k: [] for k in ("game_id", "player_id", "color", "emb_sum", "n_windows")}
    gi = 0
    for k, p in enumerate(players):
        for _ in range(n_games - (1 if short == k else 0)):
            gi += 1
            c, s = game(p)
            cand.append(["%s%06d" % (gp, gi), p, s, c])
            e, n = emb(p)
            for key, v in zip(cg, ("%s%06d" % (gp, gi), p, c, e, n)):
                cg[key].append(v)
    rng.shuffle(cand)
    os.makedirs(ddir, exist_ok=True)
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "player_id", "sgf_content", "color"])
        w.writerows(cand)
    np.savez(os.path.join(ddir, "cand_games.npz"), weights_sha256=np.array(weights),
             **{k: (np.stack(v) if k == "emb_sum" else np.array(v)) for k, v in cg.items()})
    qs, qg = [], {k: [] for k in ("question_id", "slot", "color", "emb_sum", "n_windows", "parsed")}
    order = list(range(n_players))
    rng.shuffle(order)
    for i, k in enumerate(order):
        p = players[k]
        qid = "%s%04d" % (qp, i + 1)
        n = rng.randint(5, 20)
        games = [game(p) for _ in range(n)]
        qs.append([qid, n] + [s for _c, s in games] + [""] * (20 - n) + [c for c, _s in games] + [""] * (20 - n))
        for slot, (c, _s) in enumerate(games, 1):
            e, nw = emb(p)
            for key, v in zip(qg, (qid, slot, c, e, nw, True)):
                qg[key].append(v)
    with open(qpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)] +
                   ["color_%d" % i for i in range(1, 21)])
        w.writerows(qs)
    np.savez(os.path.join(ddir, "query_games.npz"), weights_sha256=np.array(weights),
             **{k: (np.stack(v) if k == "emb_sum" else np.array(v)) for k, v in qg.items()})


def make_tree(root, smoke=True, rehearsal=False, short_pool=None):
    """The spec 2.3 layout for competition 1 (synthetic): tests, notebook records, dumps, t1.json."""
    weights = W_SMOKE if smoke else tp.WEIGHTS_SHA256
    T = os.path.join(root, "data", "test")
    R = os.path.join(root, "results", "test02")
    P = os.path.join(R, "player")
    L = os.path.join(root, "local")
    for d in (T, P, L):
        os.makedirs(d, exist_ok=True)
    files, sets = {}, {}
    for seed, pool in enumerate(tp.POOLS):
        if pool == "seam_val":
            q, c = os.path.join(L, "player_val_questions.csv"), os.path.join(L, "player_val_candidates.csv")
        else:
            q, c = (os.path.join(T, n) for n in tp.TEST_FILES[pool])
        dd = os.path.join(P, pool, "dumps")
        write_pool(q, c, dd, pool, seed=seed, weights=weights, short=3 if pool == short_pool else None)
        if pool != "seam_val":
            files[os.path.basename(q)], files[os.path.basename(c)] = sha(q), sha(c)
        sets[pool] = {"questions_sha256": sha(q), "candidates_sha256": sha(c), "rows": 10}
        with open(os.path.join(dd, "_INPUT.json"), "w", encoding="utf-8") as f:
            json.dump({"set": pool, "question_file": os.path.basename(q), "sha256": sha(q),
                       "candidates_sha256": sha(c)}, f)
    man = os.path.join(T, "MANIFEST.json")
    with open(man, "w", encoding="utf-8") as f:
        json.dump({"format": "c10-test02 manifest v1", "kind": "smoke" if smoke else "rehearsal",
                   "files": {k: {"sha256": v} for k, v in files.items()},
                   "t1": {"rank": "PASS", "player": "PASS"}}, f)
    cfg = {"smoke": smoke, "rehearsal": rehearsal, "notebook": "colab/test02_player.ipynb",
           "weights": {"path": "prereg01/player/ckpt/best_model.pth", "sha256": weights},
           "test_bundle": {"zip_sha256": "0" * 64, "manifest_sha256": sha(man), "files": files}, "sets": sets}
    jdump(os.path.join(P, "config.json"), cfg)
    jdump(os.path.join(P, "run_log.json"), {"runs": [{"run": 1, "status": "completed"}], "status": "completed"})
    jdump(os.path.join(P, "repro_check.json"), {"passed": True})
    jdump(os.path.join(P, "seam_p02.json"), {"bitwise_equal": True})
    jdump(os.path.join(R, "checks", "player", "t1.json"), {"verdict": "PASS", "normalize": []})
    return SimpleNamespace(root=root, T=T, R=R, P=P, L=L)


def jdump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def frozen_reference(tree, ref):
    """Run the frozen p02_player_fuse select / apply (--smoke) on the synthetic seam_val pool."""
    p02 = os.path.join(tree.root, "ref_p02")
    shutil.copytree(os.path.join(tree.P, "seam_val", "dumps"), os.path.join(p02, "dumps"))
    jdump(os.path.join(p02, "config.json"), {"smoke": True})
    cands = os.path.join(tree.L, "player_val_candidates.csv")
    ques = os.path.join(tree.L, "player_val_questions.csv")
    os.makedirs(ref)
    jdump(os.path.join(ref, "repro_local.json"), {
        "passed": True, "n_top5_identical": 10, "n_questions": 10, "max_abs_centroid_diff": 0.0,
        "inputs": {"candidates_csv": {"sha256": sha(cands)},
                   "cand_games_npz": {"sha256": sha(os.path.join(p02, "dumps", "cand_games.npz"))},
                   "query_games_npz": {"sha256": sha(os.path.join(p02, "dumps", "query_games.npz"))}}})
    th = po.guard_thresholds(1.0, 0.9, 0.7)
    jdump(os.path.join(ref, "guard_calibration.json"), {"smoke": True, "guard": {"raw": th, "canon": th}})
    with redirect_stdout(StringIO()):
        pf.main(["select", "--p02-dir", p02, "--candidates", cands, "--out-dir", ref, "--smoke"])
        pf.main(["apply", "--p02-dir", p02, "--candidates", cands, "--questions", ques, "--out-dir", ref,
                 "--smoke"])


class FakeCheck:
    def __init__(self, errors=()):
        self.calls, self.errors = [], list(errors)

    def check(self, kind, sub, qfiles, cfiles=(), official=False):
        with open(sub, "rb") as f:
            raw = f.read()
        self.calls.append({"kind": kind, "sub": sub, "q": list(qfiles), "c": list(cfiles), "official": official,
                           "raw": raw})
        return list(self.errors)


def run(*argv):
    with redirect_stdout(StringIO()):
        tp.main(list(argv))


class SmokeEndToEnd(unittest.TestCase):
    """seam_val, public, private and submit on a synthetic tree in --smoke mode."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.tree = make_tree(os.path.join(cls.tmp, "S"))
        cls.ref = os.path.join(cls.tmp, "ref")
        frozen_reference(cls.tree, cls.ref)
        cls.out = os.path.join(cls.tmp, "out")
        cls.fake = FakeCheck()
        with mock.patch.object(tp, "check_submission", cls.fake):
            for pool in tp.POOLS:
                run("run", "--pool", pool, *cls.args(cls.out))
            run("submit", *cls.args(cls.out))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def args(cls, out, *extra):
        t = cls.tree
        return ["--smoke", "--results-dir", t.R, "--tests-dir", t.T, "--out-dir", out, "--seam-ref", cls.ref,
                "--seam-candidates", os.path.join(t.L, "player_val_candidates.csv"),
                "--seam-questions", os.path.join(t.L, "player_val_questions.csv")] + list(extra)

    def j(self, *rel):
        with open(os.path.join(self.out, *rel), encoding="utf-8") as f:
            return json.load(f)

    def test_seam_equals_the_frozen_select_and_apply(self):
        seam = self.j("seam_val", "seam_local.json")
        self.assertTrue(seam["passed"], seam["criteria"])
        self.assertTrue(seam["means_exact"])
        self.assertEqual(seam["means_max_abs_diff"], 0.0)
        self.assertTrue(seam["byte_identical_top5"])
        self.assertEqual(seam["n_top5_identical"], 10)
        with open(os.path.join(self.ref, "frozen_config.json"), encoding="utf-8") as f:
            fz = json.load(f)
        ps = self.j("seam_val", "pool_select.json")
        self.assertEqual(ps["self_retrieval_mean"], fz["self_retrieval_mean"])
        self.assertEqual(ps["sr_max_cos_median"], fz["sr_max_cos_median"])
        self.assertEqual(ps["selected"], fz["selected"])
        self.assertEqual(ps["gate"], fz["gate"])
        with open(os.path.join(self.ref, "sr_report.json"), encoding="utf-8") as f:
            sr = json.load(f)
        self.assertEqual(ps["combinations"], sr["combinations"])
        with open(os.path.join(self.ref, "player_val_top5.csv"), "rb") as f1, \
                open(os.path.join(self.out, "seam_val", "top5.csv"), "rb") as f2:
            self.assertEqual(f1.read(), f2.read())
        ar = self.j("seam_val", "apply_report.json")
        with open(os.path.join(self.ref, "apply_report.json"), encoding="utf-8") as f:
            ref_ar = json.load(f)
        for k in ("applied", "guard_level", "guard_reason", "rho", "b_c_candidate_black_share",
                  "b_q_question_black_share", "n_fallback_questions"):
            self.assertEqual(ar[k], ref_ar[k], k)

    def test_pools_rows_in_question_file_order_and_own_ids(self):
        for pool in ("public", "private"):
            qfile = os.path.join(self.tree.T, tp.TEST_FILES[pool][0])
            rows = tp.read_top5(os.path.join(self.out, pool, "top5.csv"))
            self.assertEqual([r[0] for r in rows], tp.question_ids(qfile))
            pp = tp.PREFIX[pool][1]
            self.assertTrue(all(x.startswith(pp) for r in rows for x in r[1:]))
            self.assertTrue(all(len(set(r[1:])) == 5 for r in rows))
            ar = self.j(pool, "apply_report.json")
            self.assertEqual(ar["rho_denominator_sr_max_cos_median"],
                             self.j(pool, "pool_select.json")["sr_max_cos_median"])     # own denominator
            self.assertEqual(len(ar["sr_top1_distinct_per_round"]), po.SR_ROUNDS)
            self.assertTrue(all(1 <= x <= 10 for x in ar["sr_top1_distinct_per_round"]))
            self.assertEqual(ar["seam"]["passed"], True)
            self.assertIn("python", ar["provenance"])
            self.assertEqual(len(ar["provenance"]["tool_sha256"]), 64)

    def test_submission_public_then_private_lf(self):
        call = self.fake.calls[-1]
        self.assertEqual(call["kind"], "player")
        self.assertFalse(call["official"])                    # --smoke: subset-style check
        self.assertEqual([os.path.basename(p) for p in call["q"]], [tp.TEST_FILES["public"][0],
                                                                    tp.TEST_FILES["private"][0]])
        self.assertEqual([os.path.basename(p) for p in call["c"]], [tp.TEST_FILES["public"][1],
                                                                    tp.TEST_FILES["private"][1]])
        raw = call["raw"]
        self.assertNotIn(b"\r", raw)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        lines = raw.decode("utf-8").splitlines()
        self.assertEqual(lines[0], "question_id," + ",".join(pf.TOP_COLS))
        want = []
        for pool in ("public", "private"):
            with open(os.path.join(self.out, pool, "top5.csv"), encoding="utf-8") as f:
                want += f.read().splitlines()[1:]
        self.assertEqual(lines[1:], want)
        rep = self.j("submit_report.json")
        self.assertEqual(rep["output"]["rows"], 20)
        self.assertEqual(rep["output"]["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(set(rep["pools"]), {"public", "private"})

    def test_existing_outputs_need_rerun_with_reason(self):
        with mock.patch.object(tp, "check_submission", FakeCheck()):
            with self.assertRaises(SystemExit):
                run("run", "--pool", "public", *self.args(self.out))
            with self.assertRaises(SystemExit):
                run("run", "--pool", "public", *self.args(self.out, "--rerun"))            # no --reason
            with self.assertRaises(SystemExit):
                run("submit", *self.args(self.out))
        out2 = os.path.join(self.tmp, "out_rerun")
        shutil.copytree(self.out, out2)
        before = sha(os.path.join(out2, "public", "top5.csv"))
        run("run", "--pool", "public", *self.args(out2, "--rerun", "--reason", "test rerun"))
        with open(os.path.join(out2, "public", "apply_report.json"), encoding="utf-8") as f:
            ar = json.load(f)
        self.assertEqual(ar["rerun"]["reason"], "test rerun")
        self.assertEqual(sha(os.path.join(out2, "public", "top5.csv")), before)            # deterministic
        self.assertIn(before, ar["rerun"]["previous_sha256"].values())

    def test_submit_refused_when_check_submission_fails(self):
        out2 = os.path.join(self.tmp, "out_bad")
        shutil.copytree(self.out, out2)
        for n in ("submission_player.csv", "submit_report.json"):
            os.remove(os.path.join(out2, n))
        with mock.patch.object(tp, "check_submission", FakeCheck(["line 5: wrong pool"])):
            with self.assertRaises(SystemExit):
                run("submit", *self.args(out2))
        self.assertFalse(os.path.exists(os.path.join(out2, "submission_player.csv")))
        with open(os.path.join(out2, "submit_report.json"), encoding="utf-8") as f:
            self.assertFalse(json.load(f)["check_submission"]["passed"])

    def test_submit_refused_when_top5_changed(self):
        out2 = os.path.join(self.tmp, "out_tamper")
        shutil.copytree(self.out, out2)
        for n in ("submission_player.csv", "submit_report.json"):
            os.remove(os.path.join(out2, n))
        with open(os.path.join(out2, "private", "top5.csv"), "a", encoding="utf-8") as f:
            f.write("\n")
        with mock.patch.object(tp, "check_submission", FakeCheck()):
            with self.assertRaises(SystemExit):
                run("submit", *self.args(out2))

    def test_submit_refused_when_the_seam_changed_after_the_pools(self):
        out2 = os.path.join(self.tmp, "out_seam")
        shutil.copytree(self.out, out2)
        for n in ("submission_player.csv", "submit_report.json"):
            os.remove(os.path.join(out2, n))
        p = os.path.join(out2, "seam_val", "seam_local.json")
        with open(p, encoding="utf-8") as f:
            seam = json.load(f)
        seam["note"] = "rewritten"
        jdump(p, seam)
        with mock.patch.object(tp, "check_submission", FakeCheck()):
            with self.assertRaises(SystemExit) as cm:
                run("submit", *self.args(out2))
        self.assertIn("another seam_local.json", str(cm.exception))

    def test_no_cnn_seam_is_the_opening_only_part(self):
        out2 = os.path.join(self.tmp, "out_nocnn")
        with self.assertRaises(SystemExit):
            run("run", "--pool", "seam_val", *self.args(out2, "--no-cnn"))              # --reason needed
        run("run", "--pool", "seam_val", *self.args(out2, "--no-cnn", "--reason", "test"))
        with open(os.path.join(out2, "seam_val", "seam_local.json"), encoding="utf-8") as f:
            seam = json.load(f)
        self.assertEqual(set(seam["criteria"]), {"opening_only_means_exact", "sr_max_cos_median_exact"})
        self.assertTrue(seam["passed"])
        with open(os.path.join(out2, "seam_val", "pool_select.json"), encoding="utf-8") as f:
            ps = json.load(f)
        self.assertEqual(set(ps["self_retrieval_mean"]), {"Q22", "Q23"})
        with self.assertRaises(SystemExit):     # the CNN seam of `out` does not cover a --no-cnn test pool
            run("run", "--pool", "public", *self.args(self.out, "--no-cnn", "--reason", "t", "--rerun"))


class Guards(unittest.TestCase):
    """The negative cases of spec 5.2 that belong to competition 1."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tree(self, **kw):
        return make_tree(os.path.join(self.tmp, "S"), **kw)

    def dry(self, t, out, *extra):
        return ["--dry-run", "--results-dir", t.R, "--tests-dir", t.T, "--out-dir", out,
                "--seam-candidates", os.path.join(t.L, "player_val_candidates.csv"),
                "--seam-questions", os.path.join(t.L, "player_val_questions.csv")] + list(extra)

    def fake_seam(self, t, out, passed=True, mode="dry-run"):
        with open(os.path.join(t.P, "config.json"), "rb") as f:
            cfg_sha = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
        jdump(os.path.join(out, "seam_val", "seam_local.json"),
              {"passed": passed, "mode": mode, "cnn": True, "notebook_config_sha256": cfg_sha})

    def test_out_dir_rules(self):
        t = self.tree(smoke=False)
        for out in (os.path.join(tp.REPO, "results", "test02", "x"),              # dry-run inside the repo
                    os.path.join(tp.REPO, "results", "prereg02", "player", "x")):
            with self.assertRaises(SystemExit):
                run("run", "--pool", "public", *self.dry(t, out))
        with self.assertRaises(SystemExit):                                          # real mode, prereg02
            run("run", "--pool", "public", "--out-dir", os.path.join(tp.REPO, "results", "prereg02", "player"))
        with self.assertRaises(SystemExit):                                          # real mode, data/local
            run("run", "--pool", "public", "--out-dir", os.path.join(tp.REPO, "data", "local"))
        with self.assertRaises(SystemExit):                                          # real mode, other results dir
            run("run", "--pool", "public", "--results-dir", t.R)
        with self.assertRaises(SystemExit):                                          # real mode, other tests dir
            run("run", "--pool", "public", "--tests-dir", t.T)
        with self.assertRaises(SystemExit):                                          # needs --out-dir
            run("run", "--pool", "public", "--dry-run", "--results-dir", t.R, "--tests-dir", t.T)
        with self.assertRaises(SystemExit):                                          # seam ref only in smoke
            run("run", "--pool", "seam_val", *self.dry(t, os.path.join(self.tmp, "o"), "--seam-ref", self.tmp))

    def test_real_mode_refuses_without_notebook_outputs(self):
        # nothing is written: it stops at the missing notebook records (results/test02/player)
        if os.path.exists(os.path.join(tp.REPO, "results", "test02", "player", "config.json")):
            self.skipTest("real notebook outputs present")
        with self.assertRaises(SystemExit):
            run("run", "--pool", "seam_val")
        self.assertFalse(os.path.exists(os.path.join(tp.REPO, "results", "test02", "player", "seam_val",
                                                     "pool_select.json")))

    def test_public_needs_a_passed_seam(self):
        t = self.tree(smoke=False)
        out = os.path.join(self.tmp, "o")
        with self.assertRaises(SystemExit) as cm:
            run("run", "--pool", "public", *self.dry(t, out))
        self.assertIn("seam_val first", str(cm.exception))
        self.fake_seam(t, out, passed=False)
        with self.assertRaises(SystemExit) as cm:
            run("run", "--pool", "public", *self.dry(t, out))
        self.assertIn("seam did not pass", str(cm.exception))
        self.fake_seam(t, out, mode="smoke")
        with self.assertRaises(SystemExit):
            run("run", "--pool", "public", *self.dry(t, out))
        self.fake_seam(t, out)
        run("run", "--pool", "public", *self.dry(t, out))       # passes the gates and runs
        with open(os.path.join(out, "public", "apply_report.json"), encoding="utf-8") as f:
            ar = json.load(f)
        self.assertEqual(ar["mode"], "dry-run")
        self.assertEqual(ar["dump_weights_sha256"], {"cand_games_npz": tp.WEIGHTS_SHA256,
                                                     "query_games_npz": tp.WEIGHTS_SHA256})
        # the notebook's config.json changed after the seam run
        cfg = os.path.join(t.P, "config.json")
        with open(cfg, "a", encoding="utf-8") as f:
            f.write("\n")
        with self.assertRaises(SystemExit):
            run("run", "--pool", "private", *self.dry(t, out))

    def test_tampered_weights(self):
        t = self.tree(smoke=False)
        out = os.path.join(self.tmp, "o")
        self.fake_seam(t, out)
        p = os.path.join(t.P, "public", "dumps", "cand_games.npz")
        with np.load(p) as z:
            a = {k: z[k] for k in z.files}
        a["weights_sha256"] = np.array("f" * 64)
        np.savez(p, **a)
        with self.assertRaises(SystemExit) as cm:
            run("run", "--pool", "public", *self.dry(t, out))
        self.assertIn("weights_sha256", str(cm.exception))
        cfg = os.path.join(t.P, "config.json")                   # config.json's weights too
        t2 = make_tree(os.path.join(self.tmp, "S2"), smoke=False)
        with open(os.path.join(t2.P, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
        c["weights"]["sha256"] = "e" * 64
        jdump(os.path.join(t2.P, "config.json"), c)
        with self.assertRaises(SystemExit):
            run("run", "--pool", "seam_val", *self.dry(t2, os.path.join(self.tmp, "o2")))
        self.assertTrue(os.path.exists(cfg))

    def test_test_file_sha_differs_from_notebook_record(self):
        t = self.tree(smoke=False)
        out = os.path.join(self.tmp, "o")
        self.fake_seam(t, out)
        q = os.path.join(t.T, tp.TEST_FILES["private"][0])
        with open(q, "ab") as f:
            f.write(b"\n")
        with self.assertRaises(SystemExit) as cm:
            run("run", "--pool", "private", *self.dry(t, out))
        self.assertIn("SHA-256", str(cm.exception))
        # MANIFEST.json differs from the notebook's record
        t2 = make_tree(os.path.join(self.tmp, "S2"), smoke=False)
        self.fake_seam(t2, out)
        with open(os.path.join(t2.T, "MANIFEST.json"), "a", encoding="utf-8") as f:
            f.write(" ")
        with self.assertRaises(SystemExit) as cm:
            run("run", "--pool", "public", *self.dry(t2, os.path.join(self.tmp, "o3")))

    def test_notebook_run_not_usable(self):
        cases = [("repro_check.json", {"passed": False}),
                 ("run_log.json", {"runs": [{"status": "completed"}, {"status": "repro_failed"}]}),
                 ("run_log.json", {"runs": [{"status": "completed"}], "status": "seam_failed"})]
        for i, (name, obj) in enumerate(cases):
            t = make_tree(os.path.join(self.tmp, "S%d" % i), smoke=False)
            jdump(os.path.join(t.P, name), obj)
            with self.assertRaises(SystemExit):
                run("run", "--pool", "seam_val", *self.dry(t, os.path.join(self.tmp, "o%d" % i)))
        t = make_tree(os.path.join(self.tmp, "Sm"), smoke=True)          # --dry-run refuses SMOKE records
        with self.assertRaises(SystemExit) as cm:
            run("run", "--pool", "seam_val", *self.dry(t, os.path.join(self.tmp, "om")))
        self.assertIn("smoke", str(cm.exception))
        t = make_tree(os.path.join(self.tmp, "Sr"), smoke=False, rehearsal=False)
        reh = ["--rehearsal"] + self.dry(t, os.path.join(self.tmp, "or"))[1:]
        with self.assertRaises(SystemExit) as cm:                          # --rehearsal needs rehearsal = true
            run("run", "--pool", "seam_val", *reh)
        self.assertIn("rehearsal", str(cm.exception))

    def test_t1_must_pass(self):
        t = self.tree(smoke=False)
        out = os.path.join(self.tmp, "o")
        self.fake_seam(t, out)
        for verdict in ("FAIL_STRUCT", None):
            path = os.path.join(t.R, "checks", "player", "t1.json")
            if verdict is None:
                os.remove(path)
            else:
                jdump(path, {"verdict": verdict})
            with self.assertRaises(SystemExit):
                run("run", "--pool", "public", *self.dry(t, out))

    def test_real_mode_t1_must_be_the_full_raw_t1(self):
        """Review finding 2026-09-26 (fixer round): real mode accepts only the full T1 on the raw files."""
        C = os.path.join(self.tmp, "checks")
        path = os.path.join(C, "player", "t1.json")
        full = {"verdict": "PASS", "complete": True, "stage": "raw", "provenance": {"mode": "real"}}
        jdump(path, full)
        a = SimpleNamespace(checks_dir=C, mode="real")
        self.assertEqual(tp.check_t1(a, True)["verdict"], "PASS")
        for extra in ({"complete": False, "subchecks_run": ["T1.1", "T1.2", "T1.3"]}, {"stage": "built"},
                      {"provenance": {"mode": "dry-run"}}):
            with self.subTest(extra=extra):
                jdump(path, dict(full, **extra))
                with self.assertRaises(SystemExit) as cm:
                    tp.check_t1(a, True)
                self.assertIn("full T1", str(cm.exception))
                self.assertEqual(tp.check_t1(SimpleNamespace(checks_dir=C, mode="rehearsal"), True)["verdict"], "PASS")

    def test_real_mode_refuses_an_earlier_failed_run(self):
        """Review finding 2026-09-26 (userdocs, major): a completed run after a failed one in the same run_log is a
        rerun in the same Drive folder, not the registered rerun (PREREG_02 9.2, 10.10)."""
        t = self.tree(smoke=False)
        rl = os.path.join(t.P, "run_log.json")
        jdump(rl, {"status": "completed", "runs": [{"run": 1, "status": "repro_failed"},
                                                    {"run": 2, "status": "completed"}]})
        nb = tp.notebook_state(t.P)
        self.assertEqual(nb["earlier_failed_runs"], [(1, "repro_failed")])
        with self.assertRaises(SystemExit) as cm:
            tp.check_notebook(SimpleNamespace(mode="real", player_dir=t.P), nb, True)
        self.assertIn("earlier failed runs", str(cm.exception))
        jdump(rl, {"status": "completed", "runs": [{"run": 1, "status": "preflight_failed"},
                                                    {"run": 2, "status": "interrupted"},
                                                    {"run": 3, "status": "completed"}]})
        self.assertEqual(tp.notebook_state(t.P)["earlier_failed_runs"], [])

    def test_fail_opening_keeps_the_per_pool_procedure(self):
        """PREREG_02 rev 5 9.6: T1.4 / T1.5 failed -> competition 1 runs as usual (run and submit), and every
        report records the verdict (review finding 2026-09-26: it used to stop)."""
        t = self.tree(smoke=False)
        out = os.path.join(self.tmp, "o")
        self.fake_seam(t, out)
        jdump(os.path.join(t.R, "checks", "player", "t1.json"), {"verdict": "FAIL_OPENING", "normalize": []})
        fake = FakeCheck()
        with mock.patch.object(tp, "check_submission", fake):
            for pool in ("public", "private"):
                run("run", "--pool", pool, *self.dry(t, out))
            run("submit", *self.dry(t, out))
        for pool in ("public", "private"):
            for name in ("pool_select.json", "apply_report.json"):
                with open(os.path.join(out, pool, name), encoding="utf-8") as f:
                    rep = json.load(f)
                self.assertEqual((rep["t1"]["verdict"], rep["t1"]["fail_opening"]), ("FAIL_OPENING", True), name)
        with open(os.path.join(out, "submit_report.json"), encoding="utf-8") as f:
            sr = json.load(f)
        self.assertEqual(sr["t1"]["verdict"], "FAIL_OPENING")
        self.assertTrue(sr["check_submission"]["passed"])
        self.assertTrue(fake.calls[-1]["official"])
        self.assertTrue(os.path.exists(os.path.join(out, tp.SUBMISSION)))
        self.assertIn("FAIL_OPENING", tp.T1_ALLOWED)
        self.assertNotIn("FAIL_STRUCT", tp.T1_ALLOWED)

    def test_frozen_tool_sha_mismatch(self):
        t = self.tree(smoke=False)
        now = po.source_shas()
        bad = dict(now, **{"p02_player_fuse.py": "0" * 64})
        with mock.patch.object(tp.po, "source_shas", return_value=bad):
            with self.assertRaises(SystemExit) as cm:
                run("run", "--pool", "seam_val", *self.dry(t, os.path.join(self.tmp, "o")))
        self.assertIn("frozen player tools changed", str(cm.exception))

    def test_99_games_for_one_candidate_stops(self):
        t = self.tree(smoke=False, short_pool="public")
        out = os.path.join(self.tmp, "o")
        self.fake_seam(t, out)
        with self.assertRaises(SystemExit) as cm:
            run("run", "--pool", "public", *self.dry(t, out))
        self.assertIn("exactly 100 games", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(out, "public", "top5.csv")))

    def test_wrong_pool_prefix_stops(self):
        t = self.tree(smoke=False)
        out = os.path.join(self.tmp, "o")
        self.fake_seam(t, out)
        # swap the two pools' files: the public slot now holds priv_ ids (records updated to match)
        pub_q, pub_c = (os.path.join(t.T, n) for n in tp.TEST_FILES["public"])
        priv_q, priv_c = (os.path.join(t.T, n) for n in tp.TEST_FILES["private"])
        shutil.copyfile(priv_q, pub_q)
        shutil.copyfile(priv_c, pub_c)
        shutil.rmtree(os.path.join(t.P, "public", "dumps"))
        shutil.copytree(os.path.join(t.P, "private", "dumps"), os.path.join(t.P, "public", "dumps"))
        with open(os.path.join(t.T, "MANIFEST.json"), encoding="utf-8") as f:
            man = json.load(f)
        for n in tp.TEST_FILES["public"]:
            man["files"][n] = {"sha256": sha(os.path.join(t.T, n))}
        jdump(os.path.join(t.T, "MANIFEST.json"), man)
        with open(os.path.join(t.P, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        for n in tp.TEST_FILES["public"]:
            cfg["test_bundle"]["files"][n] = sha(os.path.join(t.T, n))
        cfg["test_bundle"]["manifest_sha256"] = sha(os.path.join(t.T, "MANIFEST.json"))
        cfg["sets"]["public"] = dict(cfg["sets"]["private"])
        jdump(os.path.join(t.P, "config.json"), cfg)
        self.fake_seam(t, out)
        with self.assertRaises(SystemExit) as cm:
            run("run", "--pool", "public", *self.dry(t, out))
        self.assertIn("prefix", str(cm.exception))

    def test_input_record_must_match(self):
        t = self.tree(smoke=False)
        out = os.path.join(self.tmp, "o")
        self.fake_seam(t, out)
        jdump(os.path.join(t.P, "public", "dumps", "_INPUT.json"), {"sha256": "0" * 64, "candidates_sha256": "1"})
        with self.assertRaises(SystemExit) as cm:
            run("run", "--pool", "public", *self.dry(t, out))
        self.assertIn("_INPUT.json", str(cm.exception))

    def test_no_cnn_needs_a_basis_outside_smoke(self):
        t = make_tree(os.path.join(self.tmp, "S"), smoke=False, rehearsal=True)
        reh = ["--rehearsal"] + self.dry(t, os.path.join(self.tmp, "o"))[1:]
        with self.assertRaises(SystemExit) as cm:           # dumps there and the reproduction passed
            run("run", "--pool", "seam_val", *reh, "--no-cnn", "--reason", "test")
        self.assertIn("--no-cnn refused", str(cm.exception))
        nb = tp.notebook_state(t.P)
        a = SimpleNamespace(player_dir=t.P, mode="rehearsal")
        jdump(os.path.join(t.P, "repro_check.json"), {"passed": False})
        self.assertTrue(tp.no_cnn_basis(a, "seam_val", tp.notebook_state(t.P))["accepted"])
        shutil.rmtree(os.path.join(t.P, "public", "dumps"))
        self.assertTrue(tp.no_cnn_basis(a, "public", nb)["dumps_missing"])

    def test_submit_needs_both_pools(self):
        t = self.tree(smoke=False)
        out = os.path.join(self.tmp, "o")
        with mock.patch.object(tp, "check_submission", FakeCheck()):
            with self.assertRaises(SystemExit) as cm:
                run("submit", *self.dry(t, out))
            self.assertIn("seam_local.json missing or not passed", str(cm.exception))
            self.fake_seam(t, out)
            with self.assertRaises(SystemExit) as cm:
                run("submit", *self.dry(t, out))
        self.assertIn("run --pool public first", str(cm.exception))


class Rules(unittest.TestCase):
    def test_gate_failure_uses_q00_at_test_time(self):
        gate = {"passed": False}
        eff, why = tp.effective_combo("Q04", gate, True)
        self.assertEqual(eff["id"], "Q00")
        self.assertIn("Q00", why)
        eff, _ = tp.effective_combo("Q04", {"passed": True}, True)
        self.assertEqual(eff["id"], "Q04")
        eff, why = tp.effective_combo("Q22", gate, False)        # fallback list: no Q00 to use
        self.assertEqual(eff["id"], "Q22")
        # Q00 is CNN-only: the guard leaves it alone (level 0)
        th = po.guard_thresholds(1.0, 0.9, 0.7)
        applied, level, _ = pf.guard_decision(pf.COMBO_BY_ID["Q00"], {"raw": 0.1, "canon": 0.1},
                                              {"thresholds": {"raw": th, "canon": th}}, 0.5, 0.5, True)
        self.assertEqual((applied["id"], level), ("Q00", 0))

    def test_pins_match_the_committed_prereg02_files(self):
        d = os.path.join(tp.REPO, "results", "prereg02", "player")
        self.assertEqual(po.sha256_text_file(os.path.join(d, "frozen_config.json")), tp.FROZEN_CONFIG_SHA256)
        self.assertEqual(po.sha256_text_file(os.path.join(d, "guard_calibration.json")), tp.GUARD_CALIBRATION_SHA256)
        self.assertEqual(po.sha256_text_file(os.path.join(d, "sr_report.json")), tp.SR_REPORT_SHA256)
        self.assertEqual(po.sha256_text_file(os.path.join(d, "player_val_top5.csv")), tp.VAL_TOP5_SHA256)
        for n, s in tp.P02_DUMP_SHA256.items():
            self.assertEqual(sha(os.path.join(d, "dumps", n)), s)
        with open(os.path.join(d, "frozen_config.json"), encoding="utf-8") as f:
            fz = json.load(f)
        self.assertEqual(fz["inputs"]["cand_games_npz"]["weights_sha256"], tp.WEIGHTS_SHA256)
        self.assertEqual(fz["guard"]["sha256"], tp.GUARD_CALIBRATION_SHA256)
        self.assertEqual(fz["selected"]["id"], tp.SEAM_COMBO)
        self.assertEqual(fz["tools_sha256"], po.source_shas())

    def test_sources_never_name_the_label_file_and_import_no_rank_module(self):
        for path in (os.path.join(TOOLS, "t02_player.py"), os.path.abspath(__file__)):
            with open(path, encoding="utf-8") as f:
                self.assertNotIn(NEEDLE, f.read().lower(), path)
        with open(os.path.join(TOOLS, "t02_player.py"), encoding="utf-8") as f:
            src = f.read()
        mods = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", src, re.M))
        self.assertFalse({m for m in mods if m.startswith(("p02_rank", "t02_rank", "decide"))}, mods)
        self.assertLessEqual({m for m in mods if m.startswith(("p02_", "t02_"))},
                             {"p02_player_fuse", "p02_player_open"})
        self.assertNotIn("prereg02/rank", src.replace("\\", "/"))
        with self.assertRaises(SystemExit):
            tp.question_ids(os.path.join("x", "player_" + NEEDLE + ".csv"))


if __name__ == "__main__":
    unittest.main()
