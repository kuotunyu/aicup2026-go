"""Integrator's end-to-end dry runs of the test-day kit without any label file (spec results/test02/design/
spec.md 5.2; PREREG_02 rev 5 section 9).

    C10_SLOW=1  FromValPipeline (about 3-4 min): t02_fake_tests.py from-val -> t02_checks.py t1 (structure only)
                -> t02_rank.py opening / apply -> t02_player.py run seam_val / public / private / submit ->
                t02_checks.py t6. The seam outputs must be byte-identical to the committed prereg02 files
                (rank_val_probs.csv, pred_B.csv, player_val_top5.csv), the test outputs must equal the validation
                outputs under the fake id mapping, and both submissions must pass check_submission --official.
    C10_SLOW=2  also FromTrainChecks (about 12 min): t02_fake_tests.py from-train -> full T1 -> T2 / T3 / T4 / T5 on
                the POS pools (training players: T2(a) FAIL, T3 overlap) and the NEG pool (inner players: T3 not
                declared), and the rank questions against the fit-only index (0 copies).
                And FromTrainOfficialShape (PREREG_02 rev 8, about 15 min): from-train --shape official with the
                slot colours after num_games empty (T1 PASS) and filled (T1 NORMALIZED by
                clear_colour_after_num_games_if_sgf_empty; the bundle build's copies re-check PASS and equal the
                empty tree byte for byte); T2(a) sides, T3b as positive (fit) / negative (inner) control on the
                black-only pools, T4 'not applicable' on black-only files, T5 official_design_match.

Every tool is called through its command line, as on test day. The trees live in a temporary directory outside
the repository (or under C10_DRYRUN_DIR when set; it must be outside the repository and is reused with --rerun).
Content checks (T1.4-T5) never run on the from-val tree: its games are validation games (spec 0.11); only
T1.1-T1.3 and the apply pipeline use it. Nothing here opens a label file or runs tools/score.py.

The golden values of the player pools depend on the fake id scheme (self-retrieval hashes player_id|game_id);
they were written from the first green run and equal the values the implementer reported.
"""

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
import check_submission  # noqa: E402

SLOW = os.environ.get("C10_SLOW", "")
P02 = os.path.join(REPO, "results", "prereg02")

# committed prereg02 references (spec 3.2 / 3.3)
RANK_VAL_PROBS_SHA = "a83700432fce5d7d26653fdaee345b0c894729fdff0029a244ed94e06c803b63"
RANK_PRED_B_SHA = "863a62582424adddc9c2e1821fcc056008bd651e103c19ec8ada98a14f87a2fd"
PLAYER_TOP5_SHA = "69ca74a61ce34f8ffa8949df731df433d474347dce9aabb535156ab8197ead6c"
SR_MED_FROZEN = {"raw": 0.22665843779803582, "canon": 0.22122531121305591}

# golden values of the from-val tree (fake ids pub_/priv_; spec 5.2 items 4-6)
PLAYER_GOLDEN = {
    "seam_val": {"Q04": 0.8997649863787647, "gate": 0.4991241719699286,
                 "sr_med": SR_MED_FROZEN},
    "public": {"Q04": 0.9021469023127683, "Q08": 0.8982233054119719, "gate": 0.49217940296670204,
               "sr_med": {"raw": 0.22536904480236913, "canon": 0.2204601908695368}, "rho_raw": 1.0126024400509277},
    "private": {"Q04": 0.8982118617937633, "Q08": 0.8970547760408695, "gate": 0.4934357493963071,
                "sr_med": {"raw": 0.22730749007053563, "canon": 0.22166653530770875},
                "rho_raw": 1.0039671134814359},
}
PLAYER_SUBMISSION_SHA = "b5d7473e1011c299f509a5ac269d60471068fc3f6cfbe1430ddafcd50ee9b8b4"
RANK_SUBMISSION_SHA_PREFIX = "d9b9801b3df8"

RANK_FILES = {"public": "rank_prediction_test_public.csv", "private": "rank_prediction_test_private.csv"}
PLAYER_Q = {"public": "player_identification_test_public.csv", "private": "player_identification_test_private.csv"}
PLAYER_C = {"public": "player_identification_candidates_public.csv",
            "private": "player_identification_candidates_private.csv"}
PREFIX = {"public": "pub_", "private": "priv_"}
RANK_ROWS = {"public": (0, 400), "private": (400, 800)}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lines(path):
    with open(path, "rb") as f:
        return f.read().split(b"\n")


def csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def inside(path, root):
    p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def run_tool(args, expect=0, log=None):
    """Run PY tools/<args> from the repository root; returns (exit code, stdout+stderr)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("PYTHONUTF8", None)
    t0 = time.time()
    r = subprocess.run([sys.executable] + list(args), cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    out = r.stdout + r.stderr
    if log is not None:
        log.append({"argv": [os.path.basename(args[0])] + list(args[1:]), "exit": r.returncode,
                    "seconds": round(time.time() - t0, 1), "tail": out[-600:]})
    if expect is not None and r.returncode != expect:
        raise AssertionError(f"{' '.join(args)} exited {r.returncode} (expected {expect}):\n{out[-4000:]}")
    return r.returncode, out


def workdir(label):
    base = os.environ.get("C10_DRYRUN_DIR")
    if base:
        base = os.path.abspath(base)
        if inside(base, REPO):
            raise unittest.SkipTest("C10_DRYRUN_DIR must be outside the repository")
        d = os.path.join(base, label)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        return d, False
    return tempfile.mkdtemp(prefix="c10_%s_" % label), True


# ================================================================ from-val: the apply pipeline
@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (pipeline dry run, ~3 min) or 2 (+ checks tree, ~12 min)")
class FromValPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S, cls.cleanup = workdir("fromval")
        S = cls.S
        cls.tests = os.path.join(S, "data", "test")
        cls.res = os.path.join(S, "results", "test02")
        cls.rank_out = os.path.join(S, "out", "rank")
        cls.player_out = os.path.join(S, "out", "player")
        cls.log = []
        L = cls.log
        run_tool(["tools/t02_fake_tests.py", "from-val", "--root", S], log=L)
        cls.t1_exit = {}
        for comp in ("rank", "player"):
            cls.t1_exit[comp] = run_tool(["tools/t02_checks.py", "t1", "--comp", comp, "--dry-run", "--root", S,
                                          "--only", "T1.1,T1.2,T1.3"], expect=None, log=L)[0]
        mode = ["--dry-run", "--results-dir", cls.res, "--tests-dir", cls.tests]
        run_tool(["tools/t02_rank.py", "opening"] + mode + ["--out-dir", cls.rank_out], log=L)
        run_tool(["tools/t02_rank.py", "apply"] + mode + ["--out-dir", cls.rank_out], log=L)
        # the registered R10 path (T1 FAIL_OPENING; PREREG_02 rev 5 9.6): a copy of the structure-only t1.json with
        # its verdict set to FAIL_OPENING, in a separate checks dir (dry run only; nothing else is changed)
        t1 = read_json(os.path.join(cls.res, "checks", "rank", "t1.json"))
        t1.update(verdict="FAIL_OPENING", dry_run_note="verdict edited by tests/test_t02_dryrun.py to exercise "
                                                         "the R10 fallback path")
        cls.fo_checks = os.path.join(S, "checks_fail_opening")
        os.makedirs(os.path.join(cls.fo_checks, "rank"))
        with open(os.path.join(cls.fo_checks, "rank", "t1.json"), "w", encoding="utf-8") as f:
            json.dump(t1, f)
        cls.r10_out = os.path.join(S, "out", "rank_r10")
        fo = ["--checks-dir", cls.fo_checks, "--out-dir", cls.r10_out]
        run_tool(["tools/t02_rank.py", "opening"] + mode + fo, log=L)
        run_tool(["tools/t02_rank.py", "apply"] + mode + fo + ["--fallback", "R10", "--reason", "dry run of R10"], log=L)
        cls.r10_refused = run_tool(["tools/t02_rank.py", "apply"] + mode + [
            "--out-dir", os.path.join(S, "out", "rank_r10_refused"), "--fallback", "R10", "--reason", "must be refused"],
            expect=None, log=L)
        # review finding 2026-09-26: public / private dumps written by another environment than seam_val (a resumed
        # notebook run) are refused. A copy of the rank notebook tree with the public marker's GPU edited.
        mixed = os.path.join(S, "results_mixed", "test02")
        shutil.copytree(os.path.join(cls.res, "rank"), os.path.join(mixed, "rank"))
        mk = os.path.join(mixed, "rank", "public", "dumps", "_DONE_val_p02e9_d4.json")
        m = read_json(mk)
        m.update(gpu_name="NVIDIA A100-SXM4-40GB", torch="9.9.9+cu999")
        with open(mk, "w", encoding="utf-8") as f:
            json.dump(m, f)
        cls.mixed_out = os.path.join(S, "out", "rank_mixed")
        os.makedirs(cls.mixed_out)
        shutil.copyfile(os.path.join(cls.rank_out, "opening_test.npz"), os.path.join(cls.mixed_out, "opening_test.npz"))
        cls.mixed_result = run_tool(["tools/t02_rank.py", "apply", "--dry-run", "--results-dir", mixed, "--tests-dir",
                                     cls.tests, "--out-dir", cls.mixed_out, "--checks-dir",
                                     os.path.join(cls.res, "checks")], expect=None, log=L)
        for pool in ("seam_val", "public", "private"):
            run_tool(["tools/t02_player.py", "run", "--pool", pool] + mode + ["--out-dir", cls.player_out], log=L)
        run_tool(["tools/t02_player.py", "submit"] + mode + ["--out-dir", cls.player_out], log=L)
        # the registered FAIL_OPENING path of competition 1 (PREREG_02 rev 5 9.6: the per-pool procedure is kept):
        # a copy of the structure-only player t1.json with its verdict set to FAIL_OPENING (dry run only)
        t1p = read_json(os.path.join(cls.res, "checks", "player", "t1.json"))
        t1p.update(verdict="FAIL_OPENING", dry_run_note="verdict edited by tests/test_t02_dryrun.py to exercise "
                                                          "the player FAIL_OPENING path")
        os.makedirs(os.path.join(cls.fo_checks, "player"))
        with open(os.path.join(cls.fo_checks, "player", "t1.json"), "w", encoding="utf-8") as f:
            json.dump(t1p, f)
        cls.player_fo_out = os.path.join(S, "out", "player_fail_opening")
        pfo = ["--checks-dir", cls.fo_checks, "--out-dir", cls.player_fo_out]
        for pool in ("seam_val", "public", "private"):
            run_tool(["tools/t02_player.py", "run", "--pool", pool] + mode + pfo, log=L)
        run_tool(["tools/t02_player.py", "submit"] + mode + pfo, log=L)
        run_tool(["tools/t02_checks.py", "t6", "--comp", "player", "--dry-run", "--root", S,
                  "--player-dir", cls.player_out,
                  "--submission", os.path.join(cls.player_out, "submission_player.csv")], log=L)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "cleanup", False) and not os.environ.get("C10_KEEP"):
            shutil.rmtree(cls.S, ignore_errors=True)

    # ---- 5.2 item 1
    def test_t1_structure_pass(self):
        for comp in ("rank", "player"):
            self.assertEqual(self.t1_exit[comp], 0, comp)
            t1 = read_json(os.path.join(self.res, "checks", comp, "t1.json"))
            self.assertEqual(t1["verdict"], "PASS", comp)
            self.assertFalse(t1.get("complete"), "content checks must not run on the from-val tree")

    def test_tree_never_used_for_content_statistics(self):
        tree = read_json(os.path.join(self.S, "FAKE_TREE.json"))
        self.assertIn("never run content statistics", tree["rule"])
        code, out = run_tool(["tools/t02_checks.py", "run", "--comp", "rank", "--dry-run", "--root", self.S,
                              "--only", "T4"], expect=None)
        self.assertNotEqual(code, 0)
        self.assertIn("validation games", out)
        self.assertFalse(os.path.exists(os.path.join(self.res, "checks", "rank", "t4.json")))

    # ---- 5.2 item 2
    def test_opening_seam_bitwise(self):
        rep = read_json(os.path.join(self.rank_out, "opening_test_report.json"))
        self.assertTrue(rep["passed"])
        self.assertTrue(all(rep["seam_check"]["equal"].values()), rep["seam_check"]["equal"])
        import numpy as np
        with np.load(os.path.join(self.rank_out, "opening_test.npz"), allow_pickle=False) as z, \
                np.load(os.path.join(P02, "rank", "opening_val.npz"), allow_pickle=False) as ref:
            sel = z["set"] == "seam_val"
            for k in ("ll", "o", "has", "n_terms"):
                self.assertEqual(z[k][sel].tobytes(), ref[k].tobytes(), k)
            self.assertEqual(int((z["set"] == "public").sum()) + int((z["set"] == "private").sum()),
                             int(sum(1 for q in z["question_id"][~sel])))

    # ---- 5.2 item 3
    def test_rank_seam_byte_identical(self):
        seam = os.path.join(self.rank_out, "seam_val")
        self.assertEqual(sha(os.path.join(seam, "rank_val_probs.csv")), RANK_VAL_PROBS_SHA)
        self.assertEqual(sha(os.path.join(seam, "pred_B.csv")), RANK_PRED_B_SHA)
        sl = read_json(os.path.join(seam, "seam_local.json"))
        self.assertTrue(sl["passed"])
        self.assertTrue(sl["byte_identical_probs"] and sl["byte_identical_pred_B"])
        self.assertEqual((sl["max_abs_diff"], sl["rule_b_differences"], sl["warnings"]), (0.0, 0, []))

    def test_rank_test_rows_equal_validation_rows(self):
        ref = lines(os.path.join(P02, "rank", "rank_val_probs.csv"))
        got = lines(os.path.join(self.rank_out, "rank_test_probs.csv"))
        self.assertEqual(got[0], ref[0])
        self.assertEqual(len(got), 1 + 800 + 1)
        self.assertEqual(got[-1], b"")
        for pool, (lo, hi) in RANK_ROWS.items():
            for i in range(lo, hi):
                r = ref[1 + i]
                self.assertTrue(r.startswith(b"val_q_"))
                self.assertEqual(got[1 + i], PREFIX[pool].encode() + r[len(b"val_"):], (pool, i))

    def test_rank_submission(self):
        sub = os.path.join(self.rank_out, "submission_rank.csv")
        self.assertEqual(sha(sub), sha(os.path.join(self.rank_out, "pred_B.csv")))
        ref = csv_rows(os.path.join(P02, "rank", "pred_B.csv"))
        got = csv_rows(sub)
        self.assertEqual(got[0], ["question_id", "pred_rank"])
        self.assertEqual(len(got), 801)
        for pool, (lo, hi) in RANK_ROWS.items():
            for i in range(lo, hi):
                q, lab = ref[1 + i]
                self.assertEqual(got[1 + i], [PREFIX[pool] + q[len("val_"):], lab])
        qfiles = [os.path.join(self.tests, RANK_FILES[p]) for p in ("public", "private")]
        self.assertEqual(check_submission.check("rank", sub, qfiles, official=True), [])
        self.assertTrue(sha(sub).startswith(RANK_SUBMISSION_SHA_PREFIX))
        rep = read_json(os.path.join(self.rank_out, "apply_test_report.json"))
        self.assertEqual(rep["check_submission"]["mode"], "official")
        self.assertFalse(rep["fallback_used"])
        self.assertEqual(rep["combo"]["id"], "R20")
        self.assertEqual(rep["questions_without_parsed_game"], {"public": 0, "private": 0})

    def test_rank_reversed_pools_fail_official(self):
        sub = os.path.join(self.rank_out, "submission_rank.csv")
        qfiles = [os.path.join(self.tests, RANK_FILES[p]) for p in ("private", "public")]
        self.assertNotEqual(check_submission.check("rank", sub, qfiles, official=True), [])

    def test_rank_r10_fallback_path(self):
        rep = read_json(os.path.join(self.r10_out, "apply_test_report.json"))
        fb = read_json(os.path.join(REPO, "results", "test02", "rank", "fallback_R10.json"))
        self.assertTrue(rep["fallback_used"])
        self.assertEqual(rep["combo"]["id"], "R10")
        self.assertFalse(rep["combo"]["opening"])
        self.assertEqual(rep["params"], fb["params"])
        self.assertEqual(rep["seam_combo"]["id"], "R20")
        # the local seam stays the frozen R20 on the validation questions: byte-identical to prereg02
        self.assertEqual(sha(os.path.join(self.r10_out, "seam_val", "rank_val_probs.csv")), RANK_VAL_PROBS_SHA)
        self.assertEqual(sha(os.path.join(self.r10_out, "seam_val", "pred_B.csv")), RANK_PRED_B_SHA)
        sub = os.path.join(self.r10_out, "submission_rank.csv")
        qfiles = [os.path.join(self.tests, RANK_FILES[p]) for p in ("public", "private")]
        self.assertEqual(check_submission.check("rank", sub, qfiles, official=True), [])
        self.assertNotEqual(sha(os.path.join(self.r10_out, "rank_test_probs.csv")),
                            sha(os.path.join(self.rank_out, "rank_test_probs.csv")))
        code, out = self.r10_refused
        self.assertEqual(code, 1)
        self.assertIn("FAIL_OPENING", out)

    def test_rank_dumps_from_another_environment_refused(self):
        code, out = self.mixed_result
        self.assertEqual(code, 1)
        self.assertIn("environment of the seam_val dump", out)
        self.assertEqual(sorted(os.listdir(self.mixed_out)), ["opening_test.npz"])       # nothing written
        # the normal run records the environment and the tolerance derived from the markers
        sl = read_json(os.path.join(self.rank_out, "seam_val", "seam_local.json"))
        self.assertTrue(sl["dump_env"]["same"])
        self.assertEqual((sl["tol"], sl["tol_check"]["same_gpu"]), (1e-4, True))

    # ---- 5.2 items 4-5
    def player_reports(self, pool):
        d = os.path.join(self.player_out, pool)
        return read_json(os.path.join(d, "pool_select.json")), read_json(os.path.join(d, "apply_report.json"))

    def test_player_seam(self):
        sl = read_json(os.path.join(self.player_out, "seam_val", "seam_local.json"))
        self.assertTrue(sl["passed"], sl["criteria"])
        self.assertTrue(sl["dumps_bitwise_equal"])
        self.assertTrue(sl["means_exact"])
        self.assertTrue(sl["byte_identical_top5"])
        self.assertEqual(sl["n_top5_identical"], 400)
        self.assertEqual(sl["sr_max_cos_median"], SR_MED_FROZEN)
        self.assertEqual(sha(os.path.join(self.player_out, "seam_val", "top5.csv")), PLAYER_TOP5_SHA)
        ps, ar = self.player_reports("seam_val")
        g = PLAYER_GOLDEN["seam_val"]
        self.assertEqual(ps["selected"]["id"], "Q04")
        self.assertEqual(ps["self_retrieval_mean"]["Q04"], g["Q04"])
        self.assertEqual(ps["gate"]["selected_minus_q00"], g["gate"])
        self.assertTrue(ps["gate"]["passed"])
        self.assertEqual(ar["guard_level"], 1)
        sr = read_json(os.path.join(P02, "player", "sr_report.json"))
        self.assertEqual(ps["self_retrieval_mean"], {r["id"]: r["mean"] for r in sr["combinations"]})

    def test_player_test_pools(self):
        ref = {r[0]: r[1:] for r in csv_rows(os.path.join(P02, "player", "player_val_top5.csv"))[1:]}
        for pool in ("public", "private"):
            g = PLAYER_GOLDEN[pool]
            ps, ar = self.player_reports(pool)
            self.assertEqual(ps["selected"]["id"], "Q04", pool)
            self.assertEqual(ps["effective"]["id"], "Q04", pool)
            self.assertEqual(ps["self_retrieval_mean"]["Q04"], g["Q04"], pool)
            self.assertEqual(ps["self_retrieval_mean"]["Q08"], g["Q08"], pool)
            self.assertEqual(ps["gate"]["selected_minus_q00"], g["gate"], pool)
            self.assertTrue(ps["gate"]["passed"], pool)
            self.assertEqual(ps["sr_max_cos_median"], g["sr_med"], pool)
            self.assertEqual(ar["guard_level"], 1, pool)
            self.assertEqual(ar["applied"]["id"], "Q04", pool)
            self.assertEqual(ar["rho"]["raw"], g["rho_raw"], pool)
            self.assertEqual(ar["n_fallback_questions"], 0, pool)
            self.assertEqual(len(ar["sr_top1_distinct_per_round"]), 10, pool)
            rows = csv_rows(os.path.join(self.player_out, pool, "top5.csv"))
            self.assertEqual(rows[0], ["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"])
            same = 0
            for r in rows[1:]:
                vq = "val_" + r[0][len(PREFIX[pool]):]
                want = [PREFIX[pool] + t[len("val_"):] for t in ref[vq]]
                same += r[1:] == want
            self.assertEqual((len(rows) - 1, same), (400, 400), pool)

    # ---- 5.2 item 6
    def test_player_submission(self):
        sub = os.path.join(self.player_out, "submission_player.csv")
        rows = csv_rows(sub)
        self.assertEqual(len(rows), 801)
        q = [os.path.join(self.tests, PLAYER_Q[p]) for p in ("public", "private")]
        c = [os.path.join(self.tests, PLAYER_C[p]) for p in ("public", "private")]
        self.assertEqual(check_submission.check("player", sub, q, c, official=True), [])
        self.assertEqual(sha(sub), PLAYER_SUBMISSION_SHA)
        with open(sub, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r", raw)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        rep = read_json(os.path.join(self.player_out, "submit_report.json"))
        self.assertTrue(rep["check_submission"]["official"] and rep["check_submission"]["passed"])

    def test_player_fail_opening_keeps_the_per_pool_procedure(self):
        """PREREG_02 rev 5 9.6 (review finding 2026-09-26): with T1 FAIL_OPENING, competition 1 runs seam_val,
        public, private and submit as usual; the numbers are the same, the reports record the verdict."""
        fo = self.player_fo_out
        self.assertEqual(sha(os.path.join(fo, "submission_player.csv")), PLAYER_SUBMISSION_SHA)
        self.assertTrue(read_json(os.path.join(fo, "seam_val", "seam_local.json"))["passed"])
        for pool in ("seam_val", "public", "private"):
            self.assertEqual(sha(os.path.join(fo, pool, "top5.csv")), sha(os.path.join(self.player_out, pool, "top5.csv")))
            for name in ("pool_select.json", "apply_report.json"):
                t1 = read_json(os.path.join(fo, pool, name))["t1"]
                self.assertEqual((t1["verdict"], t1["fail_opening"]), ("FAIL_OPENING", True), (pool, name))
        rep = read_json(os.path.join(fo, "submit_report.json"))
        self.assertEqual(rep["t1"]["verdict"], "FAIL_OPENING")
        self.assertTrue(rep["check_submission"]["official"] and rep["check_submission"]["passed"])
        seam = read_json(os.path.join(self.player_out, "seam_val", "seam_local.json"))
        self.assertEqual(seam["failed_implementation_additions"], [])
        self.assertIn("scipy", seam["environment"])

    def test_t6_reads_the_player_contract(self):
        t6 = read_json(os.path.join(self.res, "checks", "player", "t6.json"))
        self.assertEqual(t6["verdict"], "RECORD_ONLY")
        for pool in ("public", "private"):
            p = t6["pools"][pool]
            self.assertEqual((p["questions"], p["candidates"], p["questions_missing_in_submission"],
                              p["top1_outside_pool"]), (400, 400, 0, 0), pool)
            _ps, ar = self.player_reports(pool)
            self.assertEqual(p["sr_top1_distinct_per_round"], ar["sr_top1_distinct_per_round"], pool)
            self.assertEqual(p["distinct_top1_submission"], ar["n_distinct_top1"], pool)

    def test_reruns_refused_without_reason(self):
        mode = ["--dry-run", "--results-dir", self.res, "--tests-dir", self.tests]
        code, out = run_tool(["tools/t02_rank.py", "apply"] + mode + ["--out-dir", self.rank_out], expect=None)
        self.assertEqual(code, 1)
        self.assertIn("--rerun", out)
        code, out = run_tool(["tools/t02_player.py", "submit"] + mode + ["--out-dir", self.player_out], expect=None)
        self.assertNotEqual(code, 0)
        self.assertIn("--rerun", out)


# ================================================================ from-train: the checks
@unittest.skipUnless(SLOW == "2", "set C10_SLOW=2 for the checks dry run on training players (~12 min)")
class FromTrainChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S2, cls.cleanup = workdir("fromtrain")
        S2 = cls.S2
        cls.checks = os.path.join(S2, "results", "test02", "checks")
        cls.neg_tests = os.path.join(S2, "neg", "data", "test")
        cls.neg_checks = os.path.join(S2, "neg", "results", "test02", "checks")
        cache = os.path.join(S2, "cache", "t2_train_index.npz")
        os.makedirs(os.path.dirname(cache))
        cls.log = []
        L = cls.log
        run_tool(["tools/t02_fake_tests.py", "from-train", "--root", S2], log=L)
        cls.tree = read_json(os.path.join(S2, "FAKE_TREE.json"))
        cls.t1_exit = {c: run_tool(["tools/t02_checks.py", "t1", "--comp", c, "--dry-run", "--root", S2],
                                   expect=None, log=L)[0] for c in ("rank", "player")}
        # POS pools: T2-T5 against the full training index (the index cache is written for the rank run)
        run_tool(["tools/t02_checks.py", "run", "--comp", "player", "--dry-run", "--root", S2,
                  "--index-cache", cache], log=L)
        # rank questions (inner players) against the fit-only index: the negative control of T2(a)
        run_tool(["tools/t02_checks.py", "run", "--comp", "rank", "--dry-run", "--root", S2, "--only", "T2,T4,T5",
                  "--index-part", "fit", "--index-cache", cache], log=L)
        # NEG pool (inner players, one pool, not official size): T1 FAIL_STRUCT by design, T3 not declared
        cls.neg_t1_exit = run_tool(["tools/t02_checks.py", "t1", "--comp", "player", "--dry-run",
                                    "--tests-dir", cls.neg_tests, "--out", cls.neg_checks], expect=None, log=L)[0]
        run_tool(["tools/t02_checks.py", "run", "--comp", "player", "--dry-run", "--tests-dir", cls.neg_tests,
                  "--out", cls.neg_checks, "--only", "T3,T4"], log=L)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "cleanup", False) and not os.environ.get("C10_KEEP"):
            shutil.rmtree(cls.S2, ignore_errors=True)

    def check(self, comp, name, neg=False):
        return read_json(os.path.join(self.neg_checks if neg else self.checks, comp, name))

    def test_t1_pass_on_training_pools(self):
        for comp in ("rank", "player"):
            self.assertEqual(self.t1_exit[comp], 0, comp)
            t1 = self.check(comp, "t1.json")
            self.assertEqual(t1["verdict"], "PASS", comp)
            self.assertTrue(t1["complete"], comp)

    def test_t2a_fires_on_pos_player_files(self):
        t2 = self.check("player", "t2.json")
        self.assertEqual(t2["verdict"], "FAIL")
        a = [u for u in t2["units"] if u["unit"] == "a"]
        self.assertEqual({u["file"] for u in a}, set(PLAYER_Q.values()) | set(PLAYER_C.values()))
        for u in a:
            if u["file"] in PLAYER_C.values():
                self.assertEqual(u["verdict"], "FAIL", u["file"])

    def test_t2a_zero_copies_for_inner_rank_questions_against_fit(self):
        t2 = self.check("rank", "t2.json")
        self.assertEqual(t2["index"]["part"], "fit")
        self.assertEqual(t2["verdict"], "PASS")
        a = [u for u in t2["units"] if u["unit"] == "a"]
        self.assertEqual(len(a), 2)
        for u in a:
            self.assertEqual(u["copies"], 0, u["file"])
            self.assertEqual(u["verdict"], "PASS", u["file"])

    def test_t3_pos_declared_neg_not(self):
        t3 = self.check("player", "t3.json")
        self.assertEqual(t3["verdict"], "OVERLAP")
        for pool in ("public", "private"):
            d = t3["pools"][pool]
            self.assertTrue(d["overlap"], pool)
            self.assertGreaterEqual(max(d["raw"]["n_above"], d["canon"]["n_above"]), d["needed"], pool)
        self.assertTrue(all(v["passed"] for k, v in t3["reference_check"].items()), t3["reference_check"])
        self.assertNotEqual(self.neg_t1_exit, 0)
        self.assertEqual(self.check("player", "t1.json", neg=True)["verdict"], "FAIL_STRUCT")
        n3 = self.check("player", "t3.json", neg=True)
        self.assertEqual(n3["verdict"], "PASS")
        d = n3["pools"]["public"]
        self.assertEqual((d["n_candidates"], d["needed"]), (226, 23))
        self.assertFalse(d["overlap"])
        self.assertFalse(d["low_power"])                  # mixed colours: the rev 8 low-power flag stays off
        self.assertEqual((d["conclusion"], n3["low_power_pools"]), ("no overlap declared", []))
        self.assertLess(max(d["raw"]["n_above"], d["canon"]["n_above"]), d["needed"])

    def test_t4_inside_null_ranges(self):
        for comp in ("player", "rank"):
            t4 = self.check(comp, "t4.json")
            for name, f in t4["files"].items():
                self.assertTrue(f["black_share_inside_null_range"], (comp, name, f["black_share"]))
                self.assertLessEqual(f["first_move_point_tvd_vs_train"], f["first_move_point_tvd_null_max"],
                                     (comp, name))
                self.assertFalse(f["rotation_flag"], (comp, name))
        self.assertEqual(self.check("player", "t4.json")["verdict"], "PASS")
        t5 = self.check("rank", "t5.json")
        self.assertEqual(t5["verdict"], "RECORD_ONLY")
        for name, f in t5["files"].items():
            self.assertEqual(f["outside_5_20"], 0, name)
            self.assertTrue(math.isfinite(f["chi2_vs_uniform"]), name)
            self.assertFalse(f["official_design_match"], name)           # the mixed tree draws k from 5..20

    def test_t2a_sides_and_t3_variant_on_the_mixed_tree(self):
        t2 = self.check("player", "t2.json")
        for u in t2["units"]:
            if u["unit"] == "a":
                self.assertEqual(u["same_side_copies_record_only"] + u["opposite_side_copies_record_only"] +
                                 u["copies_side_unknown_record_only"], u["copies"], u["file"])
                self.assertEqual(u["opposite_side_copies_record_only"], 0, u["file"])   # own games, own colour
        t3 = self.check("player", "t3.json")
        self.assertEqual(t3["variants"], {"public": "T3", "private": "T3"})             # mixed pools
        self.assertEqual(self.check("player", "t3.json", neg=True)["variants"], {"public": "T3"})


# ================================================================ from-train, official shape (PREREG_02 rev 8)
@unittest.skipUnless(SLOW == "2", "set C10_SLOW=2 for the official-shape checks dry run on training players (~15 min)")
class FromTrainOfficialShape(unittest.TestCase):
    """The official structure (AIdea data page, docs/DATA.md 2026-09-28): competition 1 black only (400 x 100
    candidate games, questions of 5/10/15/20 games x 100), competition 2 one colour per question; the colour cells
    after num_games empty or filled with the question colour."""

    @classmethod
    def setUpClass(cls):
        cls.S, cls.cleanup = workdir("official_empty")
        cls.F, _ = workdir("official_filled")
        S, F = cls.S, cls.F
        cls.checks = os.path.join(S, "results", "test02", "checks")
        cls.fchecks = os.path.join(F, "results", "test02", "checks")
        cls.neg_tests = os.path.join(S, "neg", "data", "test")
        cls.neg_checks = os.path.join(S, "neg", "results", "test02", "checks")
        cache = os.path.join(S, "cache", "t2_train_index.npz")
        os.makedirs(os.path.dirname(cache))
        cls.log = []
        L = cls.log
        run_tool(["tools/t02_fake_tests.py", "from-train", "--root", S, "--shape", "official"], log=L)
        run_tool(["tools/t02_fake_tests.py", "from-train", "--root", F, "--shape", "official", "--slot-colours",
                  "filled"], log=L)
        cls.trees = {"empty": read_json(os.path.join(S, "FAKE_TREE.json")),
                     "filled": read_json(os.path.join(F, "FAKE_TREE.json"))}
        cls.t1_exit = {(v, c): run_tool(["tools/t02_checks.py", "t1", "--comp", c, "--dry-run", "--root", root],
                                        expect=None, log=L)[0]
                       for v, root in (("empty", S), ("filled", F)) for c in ("rank", "player")}
        # the filled tree through the bundle: normalized copies, T1.1-T1.3 re-check of the copies
        cls.built = os.path.join(F, "built", "data", "test")
        cls.build = run_tool(["tools/t02_test_bundle.py", "build", "--kind", "smoke", "--root", F, "--tests-dir",
                              os.path.join(F, "data", "test"), "--out-dir", cls.built], expect=None, log=L)
        # T2-T5 on the empty-slot tree (the game content of the two trees is the same)
        run_tool(["tools/t02_checks.py", "run", "--comp", "player", "--dry-run", "--root", S, "--index-cache", cache],
                 log=L)
        run_tool(["tools/t02_checks.py", "run", "--comp", "rank", "--dry-run", "--root", S, "--only", "T2,T4,T5",
                  "--index-cache", cache], log=L)
        cls.neg_t1_exit = run_tool(["tools/t02_checks.py", "t1", "--comp", "player", "--dry-run", "--tests-dir",
                                    cls.neg_tests, "--out", cls.neg_checks], expect=None, log=L)[0]
        run_tool(["tools/t02_checks.py", "run", "--comp", "player", "--dry-run", "--tests-dir", cls.neg_tests,
                  "--out", cls.neg_checks, "--only", "T3,T4"], log=L)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "cleanup", False) and not os.environ.get("C10_KEEP"):
            shutil.rmtree(cls.S, ignore_errors=True)
            shutil.rmtree(cls.F, ignore_errors=True)

    def check(self, comp, name, neg=False, filled=False):
        base = self.neg_checks if neg else self.fchecks if filled else self.checks
        return read_json(os.path.join(base, comp, name))

    def test_tree_shape(self):
        for v, tree in self.trees.items():
            self.assertEqual((tree["shape"], tree["slot_colours"]), ("official", v))
        self.assertEqual(self.trees["empty"]["quick_structure"], "PASS")
        self.assertTrue(self.trees["filled"]["quick_structure"].startswith("FAIL on the raw files as intended"))
        for pool in ("public", "private"):
            rows = csv_rows(os.path.join(self.S, "data", "test", PLAYER_C[pool]))
            self.assertEqual({r[3] for r in rows[1:]}, {"B"})
            self.assertEqual(len(rows) - 1, 40000)

    def test_t1_empty_pass_filled_normalized(self):
        qnames = set(PLAYER_Q.values()) | set(RANK_FILES.values())
        for comp in ("rank", "player"):
            self.assertEqual(self.t1_exit[("empty", comp)], 0, comp)
            self.assertEqual(self.t1_exit[("filled", comp)], 0, comp)
            e, f = self.check(comp, "t1.json"), self.check(comp, "t1.json", filled=True)
            self.assertEqual((e["verdict"], e["complete"]), ("PASS", True), comp)
            self.assertEqual((f["verdict"], f["complete"]), ("NORMALIZED", True), comp)
            self.assertEqual(f["normalize_transforms"], ["clear_colour_after_num_games_if_sgf_empty"], comp)
            self.assertEqual(set(f["normalize"]), {n for n in f["files"] if n in qnames}, comp)
            for name, d in f["subchecks"]["T1.3"]["files"].items():
                self.assertTrue(d["official_colour_design_record_only"]["matches_official_design"], (comp, name))
                if name in qnames:
                    self.assertGreater(d["counts"]["colour_after_num_games_with_empty_sgf"], 0, name)
                    self.assertNotIn("content_after_num_games", d["counts"], name)
            for sub in ("T1.4", "T1.5"):
                self.assertEqual(f["subchecks"][sub]["verdict"], "PASS", (comp, sub))

    def test_filled_bundle_build(self):
        code, out = self.build
        self.assertEqual(code, 0, out[-2000:])
        rec = read_json(os.path.join(self.fchecks, "bundle_build.json"))
        self.assertEqual({c: rec["t1_built"][c]["verdict"] for c in ("rank", "player")},
                         {"rank": "PASS", "player": "PASS"})
        self.assertEqual(rec["normalized"], {n: ["clear_colour_after_num_games_if_sgf_empty"]
                                             for n in list(PLAYER_Q.values()) + list(RANK_FILES.values())})
        for n in list(RANK_FILES.values()) + list(PLAYER_Q.values()) + list(PLAYER_C.values()):
            self.assertEqual(sha(os.path.join(self.built, n)), sha(os.path.join(self.S, "data", "test", n)), n)

    def test_t2a_sides(self):
        t2 = self.check("player", "t2.json")
        self.assertEqual(t2["verdict"], "FAIL")
        for u in [u for u in t2["units"] if u["unit"] == "a"]:
            self.assertGreater(u["copies"], 0, u["file"])
            self.assertEqual(u["verdict"], "FAIL", u["file"])
            self.assertEqual((u["same_side_copies_record_only"], u["opposite_side_copies_record_only"],
                              u["copies_side_unknown_record_only"]), (u["copies"], 0, 0), u["file"])
        t2r = self.check("rank", "t2.json")
        self.assertEqual(t2r["index"]["part"], "train")
        for u in [u for u in t2r["units"] if u["unit"] == "a"]:
            self.assertEqual(u["copies"], u["games_in_file"], u["file"])            # inner games, own colour
            self.assertEqual(u["same_side_copies_record_only"], u["copies"], u["file"])

    def test_t3b_positive_and_negative_control(self):
        t3 = self.check("player", "t3.json")
        self.assertTrue(all(v["passed"] for v in t3["reference_check"].values()), t3["reference_check"])
        self.assertIn("t3b_raw", t3["reference_check"])
        self.assertEqual(t3["variants"], {"public": "T3b", "private": "T3b"})
        self.assertEqual(t3["verdict"], "OVERLAP")
        for pool in ("public", "private"):
            d = t3["pools"][pool]
            self.assertEqual(d["b_c"], 1.0)
            self.assertTrue(d["overlap"], pool)
            self.assertGreaterEqual(max(d["raw"]["n_above"], d["canon"]["n_above"]), d["needed"], pool)
            self.assertIn("raw", d["registered_T3_record_only"])
        self.assertNotEqual(self.neg_t1_exit, 0)
        n3 = self.check("player", "t3.json", neg=True)
        self.assertEqual((n3["verdict"], n3["variants"]), ("PASS", {"public": "T3b"}))
        d = n3["pools"]["public"]
        self.assertEqual(d["needed"], -(-d["n_candidates"] // 10))
        self.assertFalse(d["overlap"])
        self.assertFalse(d["low_power"])                  # T3b pools are never low_power (rev 8)
        self.assertEqual((d["conclusion"], n3["low_power_pools"]), ("no overlap declared", []))
        self.assertLess(max(d["raw"]["n_above"], d["canon"]["n_above"]), d["needed"])
        self.assertTrue(all(v["passed"] for v in n3["reference_check"].values()), n3["reference_check"])

    def test_t4_t5_official(self):
        t4 = self.check("player", "t4.json")
        self.assertEqual(t4["verdict"], "PASS")
        for pool in ("public", "private"):
            self.assertTrue(t4["pools"][pool]["black_only_pool"], pool)
            self.assertEqual((t4["pools"][pool]["b_c"], t4["pools"][pool]["b_q"]), (1.0, 1.0))
        for name, f in t4["files"].items():
            self.assertFalse(f["null_ranges_applicable"], name)
            for k in ("black_share_null_range", "black_share_inside_null_range", "first_move_point_tvd_null_max",
                      "upper_right_null_range"):
                self.assertEqual(f[k], "not applicable", (name, k))
            self.assertIn("mixed_colour_null_ranges_record_only", f)
        for name, f in self.check("rank", "t4.json")["files"].items():
            self.assertTrue(f["null_ranges_applicable"], name)                   # one colour per question, mixed file
        for comp in ("player", "rank"):
            for name, f in self.check(comp, "t5.json")["files"].items():
                self.assertTrue(f["official_design_match"], (comp, name))
                self.assertEqual(f["histogram"]["5"], 100)


if __name__ == "__main__":
    unittest.main()
