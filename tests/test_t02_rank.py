"""Tests for tools/t02_rank.py (test02, competition 2).

Everything runs on synthetic data in temporary directories outside the repository (the synthetic world of
tests/test_p02_rank.py: players, SGFs and dumps in the notebook's format). No real test data, no label file,
no GPU. The frozen p02_rank_fuse.cmd_apply is the reference: the test-day rewrite of its numeric core must give
byte-identical probability files on the seam set and on re-prefixed question sets.
With C10_SLOW=1 the R20 self-check also runs on the real committed prereg02 inputs (a few seconds).
Run: python -m pytest tests/test_t02_rank.py
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import decide  # noqa: E402
import p02_rank_fuse as fuse  # noqa: E402
import p02_rank_opening as opening  # noqa: E402
import p02_rank_pseudo as pseudo  # noqa: E402
import t02_rank as t02  # noqa: E402
from test_p02_rank import World  # noqa: E402

SETS = ("seam_val", "public", "private")
SPLIT_AT = 7   # synthetic world: 14 questions -> public = 0..6 (question 3 has no parsed game), private = 7..13
# the environment every synthetic dump marker records (the notebook's _DONE_ markers carry these three fields)
DUMP_ENV = {"gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "torch": "2.11.0+cu128", "cudnn": 91900}


def sha(path):
    return pseudo.sha256_file(path)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def rekey(q, prefix):
    assert q.startswith("val_")
    return prefix + q[len("val_"):]


class FakeCheck:
    """Stands in for tools/check_submission.check (the official mode needs 400-question files)."""

    def __init__(self):
        self.calls = []
        self.__file__ = os.path.join(TOOLS, "check_submission.py")

    def check(self, kind, sub, questions, candidates=(), official=False):
        self.calls.append({"kind": kind, "sub": sub, "questions": list(questions), "official": official})
        return []


class Synth:
    """A synthetic prereg02 rank directory (frozen R20 with synthetic parameters, the frozen cmd_apply's output
    on the validation dumps as the committed reference) plus a test02 tree in the spec 2.3 / 5.1 layout."""

    def __init__(self, root, seed=3):
        self.root = root
        w = self.w = World(os.path.join(root, "world"), seed=seed)
        w.write_pseudo()
        w.write_opening()
        assert fuse.main(w.args("repro")) == 0
        assert fuse.main(w.args("tune")) == 0
        self.tuned = json.load(open(os.path.join(w.out, "frozen_config.json"), encoding="utf-8"))
        B = fuse.basis()
        beta = np.random.default_rng(seed).normal(0, 0.3, 9)
        self.params = {"a": 0.4, "gamma": 0.3, "w": 0.7, "beta": [float(x) for x in beta],
                       "b": [float(x) for x in B @ beta]}
        self.build_id = opening.load_opening(os.path.join(w.out, "opening_val.npz"), "val")["meta"]["build_id"]
        fc = dict(self.tuned, combo=dict(t02.R20), params=self.params,
                  dumps={"p02e9_d4": {"weights_sha256": "w_p02e9_d4", "syms": 8}}, opening_build_id=self.build_id)
        pseudo.write_json(os.path.join(w.out, "frozen_config.json"), fc)          # smoke flag for cmd_apply
        assert fuse.main(w.args("apply")) == 0
        decide.run(os.path.join(w.out, "rank_val_probs.csv"), w.out)
        # the "committed prereg02/rank" the tool reads (frozen config with smoke false)
        self.p02 = os.path.join(root, "p02rank")
        os.makedirs(self.p02)
        for n in ("rank_val_probs.csv", "pred_B.csv", "opening_val.npz", "opening_inner.npz", "pseudo_inner.json",
                  "tune_report.json"):
            shutil.copyfile(os.path.join(w.out, n), os.path.join(self.p02, n))
        self.fc = dict(fc, smoke=False)
        pseudo.write_json(os.path.join(self.p02, "frozen_config.json"), self.fc)
        self.D = os.path.join(root, "res", "test02")
        self.R = os.path.join(self.D, "rank")
        self.T = os.path.join(root, "tests")
        self.O = os.path.join(root, "out")
        self.build_tree()

    def patch(self):
        w = self.w
        fits = [(c, opening.game_string(s), li) for c, s, li in w.fit]
        self.fake_check = FakeCheck()
        return {"P02_RANK": self.p02, "P02_DUMPS": w.dumps, "SEAM_QUESTIONS": w.q_csv, "SPLIT_CSV": w.split_csv,
                "PINS": {}, "FROZEN_OPENING_BUILD_ID": self.build_id, "EXPECTED_FIT_GAMES": len(fits),
                "fit_games": lambda: list(fits), "check_submission": self.fake_check}

    # ---- the test02 tree
    def question_rows(self):
        with open(self.w.q_csv, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            header = next(rd)
            return header, list(rd)

    def build_tree(self):
        header, rows = self.question_rows()
        os.makedirs(self.T, exist_ok=True)
        spans = {"public": (0, SPLIT_AT, "pub_"), "private": (SPLIT_AT, len(rows), "priv_")}
        names = t02.TEST_FILES
        for s, (lo, hi, pre) in spans.items():
            with open(os.path.join(self.T, names[s]), "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f, lineterminator="\n")
                wr.writerow(header)
                for r in rows[lo:hi]:
                    wr.writerow([rekey(r[0], pre)] + r[1:])
        write_json(os.path.join(self.T, "MANIFEST.json"), {"format": "c10-test02 manifest v1", "kind": "rehearsal"})
        with np.load(os.path.join(self.w.dumps, "val_keys.npz"), allow_pickle=False) as z:
            keys = {k: z[k] for k in z.files}
        with np.load(os.path.join(self.w.dumps, "val_p02e9_d4.npz"), allow_pickle=False) as z:
            dump = {k: z[k] for k in z.files}
        qid = [str(x) for x in keys["question_id"].tolist()]
        order = list(dict.fromkeys(qid))
        # the prereg02 dump marker (its GPU decides the registered seam tolerance; t02_rank.seam_tol)
        write_json(os.path.join(self.w.dumps, "_DONE_val_p02e9_d4.json"),
                   dict(DUMP_ENV, sha256=sha(os.path.join(self.w.dumps, "val_p02e9_d4.npz"))))
        qfiles = {"seam_val": self.w.q_csv, "public": os.path.join(self.T, names["public"]),
                  "private": os.path.join(self.T, names["private"])}
        sets = {}
        for s in SETS:
            d = os.path.join(self.R, s, "dumps")
            os.makedirs(d, exist_ok=True)
            if s == "seam_val":
                for n in ("val_keys.npz", "val_p02e9_d4.npz"):
                    shutil.copyfile(os.path.join(self.w.dumps, n), os.path.join(d, n))
                n_q = len(order)
            else:
                lo, hi, pre = spans[s]
                want = set(order[lo:hi])
                idx = np.array([i for i, q in enumerate(qid) if q in want])
                np.savez(os.path.join(d, "val_keys.npz"), question_id=np.array([rekey(qid[i], pre) for i in idx]),
                         slot=keys["slot"][idx], color=keys["color"][idx], parsed=keys["parsed"][idx])
                np.savez(os.path.join(d, "val_p02e9_d4.npz"), logits=dump["logits"][idx],
                         weights_sha256=dump["weights_sha256"], syms=dump["syms"])
                n_q = hi - lo
            for stem in ("val_keys", "val_p02e9_d4"):
                write_json(os.path.join(d, "_DONE_%s.json" % stem), dict(DUMP_ENV, sha256=sha(os.path.join(d, stem + ".npz"))))
            write_json(os.path.join(d, "_INPUT.json"), {"set": s, "question_file": os.path.basename(qfiles[s]),
                                                        "sha256": sha(qfiles[s])})
            sets[s] = {"questions_sha256": sha(qfiles[s]), "rows": n_q}
        self.config = {"smoke": False, "rehearsal": True, "weights": {"path": "x", "sha256": "w_p02e9_d4"},
                       "test_bundle": {"manifest_sha256": sha(os.path.join(self.T, "MANIFEST.json")),
                                       "files": {n: sha(os.path.join(self.T, n)) for n in names.values()}},
                       "sets": sets}
        write_json(os.path.join(self.R, "config.json"), self.config)
        write_json(os.path.join(self.R, "run_log.json"), {"status": "completed", "runs": [{"status": "completed"}]})
        write_json(os.path.join(self.R, "seam_check.json"), {"passed": True, "tol": 1e-4, "bitwise_equal": True})
        for n in ("env.json", "timings.json", "readonly_before.json", "readonly_after.json"):
            write_json(os.path.join(self.R, n), {"runs": []})

    def args(self, cmd, *extra, mode="--dry-run"):
        return [cmd, mode, "--results-dir", self.D, "--tests-dir", self.T, "--out-dir", self.O] + list(extra)


class Base(unittest.TestCase):
    synth = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="c10_t02_rank_")
        cls.synth = Synth(os.path.join(cls.tmp, "s"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self._saved = {k: getattr(t02, k) for k in self.synth.patch()}
        for k, v in self.synth.patch().items():
            setattr(t02, k, v)
        shutil.rmtree(self.synth.O, ignore_errors=True)
        self.work = tempfile.mkdtemp(prefix="c10_t02_rank_case_")

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(t02, k, v)
        shutil.rmtree(self.work, ignore_errors=True)

    def run_ok(self, *argv):
        self.assertEqual(t02.main(list(argv)), 0, argv)

    def opening(self):
        self.run_ok(*self.synth.args("opening"))

    def copy_tree(self):
        """A private copy of the test02 tree and test files for a test that tampers with them."""
        s = self.synth
        D, T = os.path.join(self.work, "res", "test02"), os.path.join(self.work, "tests")
        shutil.copytree(s.D, D)
        shutil.copytree(s.T, T)
        return D, T, os.path.join(self.work, "out")

    def args(self, cmd, D, T, O, *extra, mode="--dry-run"):
        return [cmd, mode, "--results-dir", D, "--tests-dir", T, "--out-dir", O] + list(extra)


# ================================================================ the dry run against the frozen cmd_apply
class DryRun(Base):
    def test_opening_seam_is_bitwise_equal_to_the_frozen_build(self):
        s = self.synth
        self.opening()
        rep = json.load(open(os.path.join(s.O, "opening_test_report.json"), encoding="utf-8"))
        self.assertTrue(rep["passed"])
        self.assertTrue(all(rep["seam_check"]["equal"].values()))
        with np.load(os.path.join(s.O, "opening_test.npz"), allow_pickle=False) as z:
            got = {k: z[k] for k in z.files}
        ref = opening.load_opening(os.path.join(s.p02, "opening_val.npz"), "val")
        seam = got["set"] == "seam_val"
        with np.load(os.path.join(s.p02, "opening_val.npz"), allow_pickle=False) as z:
            self.assertEqual(got["ll"][seam].tobytes(), z["ll"].tobytes())
        self.assertEqual(got["o"][seam].tobytes(), np.asarray(ref["o"]).tobytes())
        # public / private slots score the same games as the seam slots they were copied from
        pub = got["set"] == "public"
        n_pub = int(pub.sum())
        self.assertEqual(got["o"][pub].tobytes(), got["o"][seam][:n_pub].tobytes())
        self.assertEqual([str(q) for q in got["question_id"][pub]][:1], ["pub_q_0001"])
        meta = json.loads(str(got["meta"]))
        self.assertEqual(meta["frozen_opening_build_id"], s.build_id)
        self.assertTrue(meta["seam_bitwise_equal"])

    def test_apply_equals_the_frozen_cmd_apply(self):
        s = self.synth
        self.opening()
        self.run_ok(*s.args("apply"))
        O = s.O
        seam = json.load(open(os.path.join(O, "seam_val", "seam_local.json"), encoding="utf-8"))
        self.assertTrue(seam["passed"])
        self.assertTrue(seam["byte_identical_probs"] and seam["byte_identical_pred_B"], seam)
        self.assertEqual(seam["max_abs_diff"], 0.0)
        self.assertEqual(seam["warnings"], [])
        # test rows = the frozen cmd_apply's validation rows under the prefix mapping (public, then private)
        ref = open(os.path.join(s.p02, "rank_val_probs.csv"), encoding="utf-8").read().splitlines()
        got = open(os.path.join(O, "rank_test_probs.csv"), encoding="utf-8").read().splitlines()
        self.assertEqual(got[0], ref[0])
        self.assertEqual(len(got), len(ref))
        for i, (a, b) in enumerate(zip(got[1:], ref[1:])):
            self.assertEqual(a, ("pub_" if i < SPLIT_AT else "priv_") + b[len("val_"):])
        # the unparsed question is blank and gets D; the submission is a byte copy of pred_B.csv
        rows = decide.read_probs(os.path.join(O, "rank_test_probs.csv"))
        self.assertTrue(rows[3]["missing"])
        sub = open(os.path.join(O, "submission_rank.csv"), "rb").read()
        self.assertEqual(sub, open(os.path.join(O, "pred_B.csv"), "rb").read())
        self.assertNotIn(b"\r", sub)
        self.assertTrue(sub.startswith(b"question_id,pred_rank\npub_q_0001,"))
        ref_b = dict((r["question_id"], r["pred_rank"]) for r in csv.DictReader(open(os.path.join(s.p02, "pred_B.csv"))))
        for r in csv.DictReader(open(os.path.join(O, "submission_rank.csv"))):
            self.assertEqual(r["pred_rank"], ref_b["val_" + r["question_id"].split("_", 1)[1]])
        # check_submission --official on [public, private]
        call = s.fake_check.calls[-1]
        self.assertEqual((call["kind"], call["official"]), ("rank", True))
        self.assertEqual([os.path.basename(q) for q in call["questions"]], list(t02.TEST_FILES.values()))
        rep = json.load(open(os.path.join(O, "apply_test_report.json"), encoding="utf-8"))
        self.assertEqual(rep["combo"]["id"], "R20")
        self.assertFalse(rep["fallback_used"])
        self.assertEqual(rep["questions_without_parsed_game"], {"public": 1, "private": 0})
        self.assertEqual(set(rep["sets"]["public"]["rule_b_distribution"]), set(decide.LEVELS))
        self.assertEqual(rep["outputs_sha256"]["submission_rank.csv"], sha(os.path.join(O, "submission_rank.csv")))
        for k in ("tool", "environment", "inputs_sha256", "checks"):
            self.assertIn(k, rep)
        self.assertIn("t02_rank.py_sha256", rep["tool"])

    def test_outputs_are_not_overwritten_without_rerun(self):
        s = self.synth
        self.opening()
        self.assertEqual(t02.main(s.args("opening")), 1)
        self.run_ok(*s.args("apply"))
        self.assertEqual(t02.main(s.args("apply")), 1)
        self.assertEqual(t02.main(s.args("apply", "--rerun")), 1)                       # no reason
        self.run_ok(*s.args("apply", "--rerun", "--reason", "unit test"))
        rep = json.load(open(os.path.join(s.O, "apply_test_report.json"), encoding="utf-8"))
        self.assertEqual(rep["rerun"]["reason"], "unit test")
        self.assertIn("submission_rank.csv", " ".join(rep["rerun"]["replaced_sha256"]))

    def test_seam_failure_writes_no_test_output(self):
        D, T, O = self.copy_tree()
        p = os.path.join(D, "rank", "seam_val", "dumps", "val_p02e9_d4.npz")
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["logits"] = arrays["logits"].copy()
        arrays["logits"][0, :, 0] += 5.0                  # the first question moves by far more than 1e-4
        np.savez(p, **arrays)
        self.run_ok(*self.args("opening", D, T, O))
        self.assertEqual(t02.main(self.args("apply", D, T, O)), 1)
        seam = json.load(open(os.path.join(O, "seam_val", "seam_local.json"), encoding="utf-8"))
        self.assertFalse(seam["passed"])
        self.assertGreater(seam["max_abs_diff"], 1e-4)
        for n in ("rank_test_probs.csv", "pred_B.csv", "submission_rank.csv", "apply_test_report.json"):
            self.assertFalse(os.path.exists(os.path.join(O, n)), n)


# ================================================================ refusals (spec 5.2 fast negatives)
class Refusals(Base):
    def edit_json(self, path, **kw):
        obj = json.load(open(path, encoding="utf-8"))
        obj.update(kw)
        write_json(path, obj)

    def refused(self, argv):
        self.assertEqual(t02.main(argv), 1, argv)
        for n in ("rank_test_probs.csv", "submission_rank.csv"):
            self.assertFalse(os.path.exists(os.path.join(argv[argv.index("--out-dir") + 1], n)), (argv, n))

    def test_tampered_weights_sha256(self):
        D, T, O = self.copy_tree()
        p = os.path.join(D, "rank", "public", "dumps", "val_p02e9_d4.npz")
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["weights_sha256"] = np.array("w_other")
        np.savez(p, **arrays)
        self.run_ok(*self.args("opening", D, T, O))
        self.refused(self.args("apply", D, T, O))

    def test_test_file_differs_from_the_notebook_record(self):
        D, T, O = self.copy_tree()
        with open(os.path.join(T, t02.TEST_FILES["private"]), "a", encoding="utf-8") as f:
            f.write("\n")
        self.refused(self.args("opening", D, T, O))
        self.refused(self.args("apply", D, T, O))

    def test_manifest_differs_from_the_notebook_record(self):
        D, T, O = self.copy_tree()
        write_json(os.path.join(T, "MANIFEST.json"), {"kind": "other"})
        self.refused(self.args("opening", D, T, O))

    def test_input_record_differs(self):
        D, T, O = self.copy_tree()
        self.edit_json(os.path.join(D, "rank", "private", "dumps", "_INPUT.json"), sha256="0" * 64)
        self.refused(self.args("opening", D, T, O))

    def test_out_dir_guards(self):
        D, T, _O = self.copy_tree()
        for out in (os.path.join(REPO, "results", "test02", "rank_dry"),              # dry run inside the repo
                    os.path.join(REPO, "results", "prereg02", "rank_t02_guard_test"),   # decide.py would write here
                    os.path.join(REPO, "results", "prereg01", "x"),
                    os.path.join(REPO, "data", "local", "x"),
                    os.path.join(self.synth.p02, "x")):                                 # the (patched) prereg02 dir
            self.refused(self.args("apply", D, T, out))
            self.assertFalse(os.path.exists(os.path.join(out, "seam_val", "rank_val_probs.csv")))
        with self.assertRaises(t02.T02Error):
            t02.check_out_dir(os.path.join(REPO, "results", "prereg02", "rank"), "real")

    def test_notebook_seam_or_status_not_passed(self):
        for name, kw in (("seam_check.json", {"passed": False}), ("seam_check.json", {"tol": None}),
                         ("run_log.json", {"status": "seam_failed"}),
                         ("run_log.json", {"runs": [{"status": "readonly_changed"}]})):
            with self.subTest(name=name, kw=kw):
                D, T, O = self.copy_tree()
                self.edit_json(os.path.join(D, "rank", name), **kw)
                self.refused(self.args("opening", D, T, O))
                shutil.rmtree(self.work)
                os.makedirs(self.work)

    def test_mode_flags_must_match_the_notebook_record(self):
        D, T, O = self.copy_tree()
        self.refused(self.args("opening", D, T, O, mode="--smoke"))            # the record says smoke false
        self.edit_json(os.path.join(D, "rank", "config.json"), rehearsal=False)
        self.refused(self.args("opening", D, T, O))                             # dry run needs rehearsal true
        self.assertEqual(t02.main(["opening", "--dry-run", "--smoke", "--results-dir", D, "--tests-dir", T,
                                   "--out-dir", O]), 1)
        self.assertEqual(t02.main(["opening", "--dry-run", "--results-dir", D, "--tests-dir", T]), 1)  # no out dir

    def test_real_mode_paths_are_fixed(self):
        s = self.synth
        self.assertEqual(t02.main(["opening", "--results-dir", s.D]), 1)
        self.assertEqual(t02.main(["apply", "--out-dir", s.O]), 1)

    def test_frozen_tool_sha_mismatch(self):
        orig = fuse.tool_state

        def fake_state():
            st = orig()
            st["files_sha256"] = dict(st["files_sha256"], **{"decide.py": "0" * 64})
            return st
        fuse.tool_state = fake_state
        try:
            self.refused(self.synth.args("opening"))
        finally:
            fuse.tool_state = orig

    def test_frozen_config_checks(self):
        s = self.synth
        path = os.path.join(s.p02, "frozen_config.json")
        good = open(path, "rb").read()
        bad_b = dict(s.fc, params=dict(s.params, b=[x + 1e-9 for x in s.params["b"]]))
        for fc in (bad_b, dict(s.fc, combo=dict(t02.R10)), dict(s.fc, smoke=True),
                   dict(s.fc, dumps={"p02e9_d4": {"weights_sha256": "w_p02e9_d4", "syms": 1}}),
                   dict(s.fc, opening_build_id="x")):
            with self.subTest(fc={k: fc[k] for k in ("combo", "smoke", "dumps", "opening_build_id")}):
                pseudo.write_json(path, fc)
                try:
                    self.refused(s.args("opening"))
                finally:
                    with open(path, "wb") as f:
                        f.write(good)
        t02.PINS = {"frozen_config.json": "0" * 64}
        self.refused(s.args("opening"))

    def test_opening_seam_must_be_bitwise(self):
        s = self.synth
        work_p02 = os.path.join(self.work, "p02rank")
        shutil.copytree(s.p02, work_p02)
        p = os.path.join(work_p02, "opening_val.npz")
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["ll"] = arrays["ll"].copy()
        i, j = np.argwhere(arrays["ll"] != 0)[0]
        arrays["ll"][i, j] = np.nextafter(arrays["ll"][i, j], 0)            # one ulp
        np.savez(p, **arrays)
        t02.P02_RANK = work_p02
        self.assertEqual(t02.main(s.args("opening")), 1)
        rep = json.load(open(os.path.join(s.O, "opening_test_report.json"), encoding="utf-8"))
        self.assertFalse(rep["passed"])
        self.assertFalse(rep["seam_check"]["equal"]["ll"])
        self.assertFalse(os.path.exists(os.path.join(s.O, "opening_test.npz")))

    def test_t1_verdicts_and_the_fallback_rule(self):
        D, T, O = self.copy_tree()
        C = os.path.join(self.work, "checks")
        self.run_ok(*self.args("opening", D, T, O, "--checks-dir", C))              # dry run: t1.json optional
        self.refused(self.args("apply", D, T, O, "--checks-dir", C, "--fallback", "R10", "--reason", "x"))  # no t1
        t1 = os.path.join(C, "rank", "t1.json")
        for verdict, argv_extra, ok in (("PASS", ["--fallback", "R10", "--reason", "x"], False),
                                        ("NORMALIZED", ["--fallback", "R10", "--reason", "x"], False),
                                        ("FAIL_STRUCT", [], False), ("FAIL_OPENING", [], False)):
            with self.subTest(verdict=verdict, extra=argv_extra):
                write_json(t1, {"verdict": verdict, "normalize": []})
                self.assertEqual(t02.main(self.args("apply", D, T, O, "--checks-dir", C, *argv_extra)),
                                 0 if ok else 1)
        write_json(t1, {"verdict": "FAIL_STRUCT"})
        self.assertEqual(t02.main(self.args("opening", D, T, O, "--checks-dir", C, "--rerun", "--reason", "x")), 1)
        self.assertEqual(t02.main(self.args("apply", D, T, O, "--checks-dir", C, "--fallback", "R10")), 1)  # reason

    def test_needle_paths_refused(self):
        word = "ans" + "wers"
        for p in ("x/rank_val_%s.csv" % word, "data/local/%s" % word.upper()):
            with self.assertRaises(t02.T02Error):
                t02.refuse_needle(p)
        t02.refuse_needle("data/test/rank_prediction_test_public.csv")

    # ---- review findings 2026-09-26
    def edit_marker(self, D, s, **kw):
        self.edit_json(os.path.join(D, "rank", s, "dumps", "_DONE_val_p02e9_d4.json"), **kw)

    def test_dumps_from_another_environment_are_refused(self):
        """A resumed notebook run may reuse seam_val from one runtime and compute public / private in another; the
        seam does not cover that, so apply writes nothing (not even the seam files)."""
        for s, kw in (("public", {"gpu_name": "NVIDIA A100-SXM4-40GB"}), ("private", {"torch": "9.9.9+cu999"}),
                      ("public", {"cudnn": 1}), ("seam_val", {"gpu_name": None})):
            with self.subTest(set=s, kw=kw):
                D, T, O = self.copy_tree()
                self.run_ok(*self.args("opening", D, T, O))
                self.edit_marker(D, s, **kw)
                self.refused(self.args("apply", D, T, O))
                self.assertFalse(os.path.exists(os.path.join(O, "seam_val", "seam_local.json")))
                shutil.rmtree(self.work)
                os.makedirs(self.work)

    def test_seam_tolerance_follows_the_registered_rule(self):
        # same GPU as the prereg02 dump -> 1e-4; the notebook's 1e-3 is refused
        D, T, O = self.copy_tree()
        self.run_ok(*self.args("opening", D, T, O))
        self.edit_json(os.path.join(D, "rank", "seam_check.json"), tol=1e-3)
        self.refused(self.args("apply", D, T, O))
        # another GPU for all three sets -> 1e-3 is the rule and is accepted; 1e-4 is refused
        for s in SETS:
            self.edit_marker(D, s, gpu_name="NVIDIA A100-SXM4-40GB")
        self.run_ok(*self.args("apply", D, T, O, "--rerun", "--reason", "tol 1e-3"))
        seam = json.load(open(os.path.join(O, "seam_val", "seam_local.json"), encoding="utf-8"))
        self.assertEqual((seam["tol"], seam["tol_check"]["same_gpu"]), (1e-3, False))
        before = sha(os.path.join(O, "submission_rank.csv"))
        self.edit_json(os.path.join(D, "rank", "seam_check.json"), tol=1e-4)
        self.assertEqual(t02.main(self.args("apply", D, T, O, "--rerun", "--reason", "tol 1e-4 with another GPU")), 1)
        self.assertEqual(sha(os.path.join(O, "submission_rank.csv")), before)     # refused before anything changed

    def test_notebook_identity_real_mode_refuses(self):
        nb = os.path.join(self.work, "test02_rank.ipynb")
        cells = [{"cell_type": "code", "source": ["NOTEBOOK_SHA256 = '%s'\n" % ("0" * 64), "x = 1\n"]}]
        with open(nb, "w", encoding="utf-8") as f:
            json.dump({"cells": cells}, f)
        _rec, good = t02.notebook_source_sha256(nb)
        cells[0]["source"][0] = "NOTEBOOK_SHA256 = '%s'\n" % good
        with open(nb, "w", encoding="utf-8") as f:
            json.dump({"cells": cells}, f)
        saved = t02.NOTEBOOK
        t02.NOTEBOOK = nb
        try:
            self.assertTrue(t02.check_notebook_identity({"notebook_sha256": good}, "real"))
            with self.assertRaises(t02.T02Error):
                t02.check_notebook_identity({"notebook_sha256": "f" * 64}, "real")
            self.assertFalse(t02.check_notebook_identity({"notebook_sha256": "f" * 64}, "rehearsal"))   # warning
            self.assertFalse(t02.check_notebook_identity({"notebook_sha256": None}, "dry-run"))
            t02.NOTEBOOK = os.path.join(self.work, "missing.ipynb")
            with self.assertRaises(t02.T02Error):
                t02.check_notebook_identity({"notebook_sha256": good}, "real")
            self.assertIsNone(t02.check_notebook_identity({}, "dry-run"))
        finally:
            t02.NOTEBOOK = saved

    # ---- review findings 2026-09-26 (fixer round)
    def test_real_mode_requires_the_committed_notebook(self):
        """Real mode checks that colab/test02_rank.ipynb itself is committed and clean (as competition 1 does), not
        only that the run's notebook_sha256 equals the working-tree file (finding: an uncommitted rebuild passed)."""
        seen = []

        def fake_commit_check(paths, what):
            seen.append(list(paths))
            raise t02.T02Error("stop after the commit check")
        saved = (t02.load_frozen, t02.require_committed_clean)
        t02.load_frozen = lambda require_git: ({}, {})
        t02.require_committed_clean = fake_commit_check
        try:
            with self.assertRaises(t02.T02Error):
                t02.common_checks({"mode": "real", "rank_dir": self.work}, "apply", False)
        finally:
            t02.load_frozen, t02.require_committed_clean = saved
        self.assertIn(t02.NOTEBOOK, seen[0])
        self.assertIn(t02.SELF, seen[0])

    def test_real_mode_t1_must_be_the_full_raw_t1(self):
        C = os.path.join(self.work, "checks")
        t1 = os.path.join(C, "rank", "t1.json")
        ctx = {"mode": "real", "checks_dir": C}
        full = {"verdict": "PASS", "complete": True, "stage": "raw", "provenance": {"mode": "real"}}
        write_json(t1, full)
        self.assertEqual(t02.t1_status(ctx, "apply", False)["verdict"], "PASS")
        for extra in ({"complete": False, "subchecks_run": ["T1.1", "T1.2", "T1.3"]}, {"stage": "built"},
                      {"provenance": {"mode": "rehearsal"}}):
            with self.subTest(extra=extra):
                write_json(t1, dict(full, **extra))
                with self.assertRaises(t02.T02Error) as cm:
                    t02.t1_status(ctx, "apply", False)
                self.assertIn("full T1", str(cm.exception))
                self.assertEqual(t02.t1_status(dict(ctx, mode="rehearsal"), "apply", False)["verdict"], "PASS")
        write_json(t1, dict(full, verdict="FAIL_OPENING"))
        self.assertEqual(t02.t1_status(ctx, "apply", True)["verdict"], "FAIL_OPENING")

    def test_real_mode_refuses_an_earlier_failed_run(self):
        """A completed run after a failed one in the same run_log is a rerun in the same Drive folder, not the
        registered rerun (PREREG_02 9.2, 10.10)."""
        D, T, O = self.copy_tree()
        R = os.path.join(D, "rank")
        self.edit_json(os.path.join(R, "config.json"), rehearsal=False)
        ctx = {"mode": "real", "rank_dir": R}
        t02.load_notebook(ctx)
        for bad in t02.FAILED_RUN_STATUSES:
            with self.subTest(bad=bad):
                write_json(os.path.join(R, "run_log.json"), {"status": "completed", "runs": [
                    {"run": 1, "status": bad}, {"run": 2, "status": "completed"}]})
                with self.assertRaises(t02.T02Error) as cm:
                    t02.load_notebook(ctx)
                self.assertIn("earlier failed runs", str(cm.exception))
        write_json(os.path.join(R, "run_log.json"), {"status": "completed", "runs": [
            {"run": 1, "status": "running"}, {"run": 2, "status": "completed"}]})      # an interrupted run
        t02.load_notebook(ctx)

    def test_pins_fold_crlf_for_text_files(self):
        lf = os.path.join(self.work, "x.csv")
        with open(lf, "wb") as f:
            f.write(b"a,b\n1,2\n")
        crlf = os.path.join(self.work, "y.csv")
        with open(crlf, "wb") as f:
            f.write(b"a,b\r\n1,2\r\n")
        self.assertEqual(t02.pin_sha256(lf), t02.pin_sha256(crlf))
        self.assertEqual(t02.pin_sha256(lf), sha(lf))
        t02.PINS = {"x.csv": sha(lf)}
        self.assertEqual(t02.pinned(crlf, "x.csv"), sha(lf))
        npz = os.path.join(self.work, "z.npz")
        with open(npz, "wb") as f:
            f.write(b"PK\r\n")
        self.assertNotEqual(t02.pin_sha256(npz), t02.pin_sha256(crlf))
        self.assertEqual(t02.pin_sha256(npz), sha(npz))                           # binary: raw bytes


# ================================================================ the R10 fallback
class Fallback(Base):
    def fallback_file(self, params, **kw):
        obj = {"format": t02.FALLBACK_FORMAT, "combo": dict(t02.R10), "params": params,
               "selfcheck_R20_passed": True, "dumps": self.synth.fc["dumps"]}
        obj.update(kw)
        path = os.path.join(self.work, "fallback_R10.json")
        write_json(path, obj)
        return path

    def test_fallback_apply_uses_r10_on_the_test_sets_and_r20_on_the_seam(self):
        s = self.synth
        D, T, O = self.copy_tree()
        C = os.path.join(self.work, "checks")
        write_json(os.path.join(C, "rank", "t1.json"), {"verdict": "FAIL_OPENING"})
        B = fuse.basis()
        beta = np.linspace(-0.2, 0.3, 9)
        par = {"a": 0.2, "gamma": 0.1, "w": 0.0, "beta": [float(x) for x in beta], "b": [float(x) for x in B @ beta]}
        fb = self.fallback_file(par)
        self.run_ok(*self.args("opening", D, T, O, "--checks-dir", C))
        self.run_ok(*self.args("apply", D, T, O, "--checks-dir", C, "--fallback", "R10", "--reason", "T1.5 failed",
                               "--fallback-file", fb))
        seam = json.load(open(os.path.join(O, "seam_val", "seam_local.json"), encoding="utf-8"))
        self.assertTrue(seam["passed"] and seam["byte_identical_probs"])
        rep = json.load(open(os.path.join(O, "apply_test_report.json"), encoding="utf-8"))
        self.assertTrue(rep["fallback_used"])
        self.assertEqual(rep["combo"]["id"], "R10")
        self.assertEqual(rep["fallback"]["reason"], "T1.5 failed")
        # reference: the frozen cmd_apply with R10 and these parameters on the validation dumps
        w = s.w
        ref_dir = os.path.join(self.work, "ref")
        shutil.copytree(w.out, ref_dir)
        fc = json.load(open(os.path.join(ref_dir, "frozen_config.json"), encoding="utf-8"))
        pseudo.write_json(os.path.join(ref_dir, "frozen_config.json"), dict(fc, combo=dict(t02.R10), params=par))
        args = w.args("apply")
        args[args.index("--out-dir") + 1] = ref_dir
        self.assertEqual(fuse.main(args), 0)
        ref = open(os.path.join(ref_dir, "rank_val_probs.csv"), encoding="utf-8").read().splitlines()
        got = open(os.path.join(O, "rank_test_probs.csv"), encoding="utf-8").read().splitlines()
        for i, (a, b) in enumerate(zip(got[1:], ref[1:])):
            self.assertEqual(a, ("pub_" if i < SPLIT_AT else "priv_") + b[len("val_"):])

    def test_fallback_file_checks(self):
        D, T, O = self.copy_tree()
        C = os.path.join(self.work, "checks")
        write_json(os.path.join(C, "rank", "t1.json"), {"verdict": "FAIL_OPENING"})
        self.run_ok(*self.args("opening", D, T, O, "--checks-dir", C))
        B = fuse.basis()
        par = {"a": 0.0, "gamma": 0.0, "w": 0.0, "beta": [0.1] * 9, "b": [float(x) for x in B @ np.full(9, 0.1)]}
        for kw in ({"selfcheck_R20_passed": False}, {"combo": dict(t02.R20)}, {"format": "x"},
                   {"params": dict(par, b=[0.0] * 10)}, {"dumps": {"p02e9_d4": {"weights_sha256": "y", "syms": 8}}}):
            with self.subTest(kw=list(kw)):
                kw = dict(kw)
                fb = self.fallback_file(kw.pop("params", par), **kw)
                self.assertEqual(t02.main(self.args("apply", D, T, O, "--checks-dir", C, "--fallback", "R10",
                                                    "--reason", "x", "--fallback-file", fb, "--rerun")), 1)

    def test_real_mode_uses_only_the_committed_file_from_before_the_test_data(self):
        fb = self.fallback_file({"a": 0.0})
        # real mode (no mode flag): --fallback-file is refused before anything is read or written
        self.assertEqual(t02.main(["apply", "--fallback", "R10", "--reason", "x", "--fallback-file", fb]), 1)
        saved = t02.git
        try:
            t02.git = lambda *a: (0, "abc123 2026-11-04T11:00:00+08:00")      # not before 11:00
            with self.assertRaises(t02.T02Error):
                t02.fallback_commit(t02.FALLBACK_FILE)
            t02.git = lambda *a: (0, "abc123 2026-11-04T02:59:59+00:00")      # 10:59:59 in Taiwan
            self.assertEqual(t02.fallback_commit(t02.FALLBACK_FILE)["commit"], "abc123")
            t02.git = lambda *a: (128, "")
            with self.assertRaises(t02.T02Error):
                t02.fallback_commit(t02.FALLBACK_FILE)
        finally:
            t02.git = saved


# ================================================================ fallback-fit
class FallbackFit(Base):
    def test_prepare_and_refit_reproduce_the_tune_refit(self):
        """The t02 replication of cmd_tune's fit-round preparation gives the tune refit of its selected combo."""
        s = self.synth
        t02.P02_RANK = s.w.out
        combo = {k: s.tuned["combo"][k] for k in ("id", "source", "pool", "calibrated", "opening")}
        if not combo["calibrated"]:
            self.skipTest("the synthetic tune selected R00 (no parameters)")
        prep = t02.prepare_fit([combo["source"]])
        _th, refit, par = t02.fit_combo(prep, combo, fuse.basis())
        self.assertLessEqual(t02.params_diff(par, s.tuned["params"]), 1e-12)
        self.assertEqual(refit["fit_questions"], len(prep["X"]))

    def test_cli_selfcheck_and_r10(self):
        s = self.synth
        p02 = os.path.join(self.work, "p02rank")
        shutil.copytree(s.p02, p02)
        t02.P02_RANK = p02
        tune = json.load(open(os.path.join(p02, "tune_report.json"), encoding="utf-8"))
        for c in tune["combinations"]:
            if c["id"] == "R10" and c.get("status") != "ok":
                c.update(status="ok", oof={"rank_balanced": 0.5, "question_mean": 0.5})
        pseudo.write_json(os.path.join(p02, "tune_report.json"), tune)
        prep = t02.prepare_fit(["C5"])
        _th, _r, par20 = t02.fit_combo(prep, t02.R20, fuse.basis())
        fc = dict(s.fc, params=par20, inputs_sha256=dict(s.fc["inputs_sha256"], **prep["inputs"]))
        pseudo.write_json(os.path.join(p02, "frozen_config.json"), fc)
        O = os.path.join(self.work, "ff")
        self.run_ok("fallback-fit", "--combo", "R20", "--out-dir", O)
        chk = json.load(open(os.path.join(O, "fallback_selfcheck_R20.json"), encoding="utf-8"))
        self.assertTrue(chk["passed"])
        self.assertEqual(chk["max_abs_diff"], 0.0)
        self.run_ok("fallback-fit", "--combo", "R10", "--out-dir", O)
        fb = json.load(open(os.path.join(O, "fallback_R10.json"), encoding="utf-8"))
        self.assertEqual(fb["combo"], t02.R10)
        self.assertTrue(fb["selfcheck_R20_passed"])
        self.assertEqual(fb["params"]["w"], 0.0)
        np.testing.assert_allclose(fuse.basis() @ np.array(fb["params"]["beta"]), fb["params"]["b"], atol=1e-12)
        r10 = next(c for c in tune["combinations"] if c["id"] == "R10")
        self.assertEqual(fb["inner_oof_rank_balanced"], r10["oof"]["rank_balanced"])
        # never overwritten without --rerun --reason; a failed self-check writes no R10
        self.assertEqual(t02.main(["fallback-fit", "--combo", "R10", "--out-dir", O]), 1)
        self.run_ok("fallback-fit", "--combo", "R10", "--out-dir", O, "--rerun", "--reason", "unit test")
        bad = dict(fc, params=dict(par20, a=par20["a"] + 1e-9))
        pseudo.write_json(os.path.join(p02, "frozen_config.json"), bad)
        O2 = os.path.join(self.work, "ff2")
        self.assertEqual(t02.main(["fallback-fit", "--combo", "R20", "--out-dir", O2]), 1)
        self.assertFalse(json.load(open(os.path.join(O2, "fallback_selfcheck_R20.json"), encoding="utf-8"))["passed"])
        self.assertEqual(t02.main(["fallback-fit", "--combo", "R10", "--out-dir", O2]), 1)
        self.assertFalse(os.path.exists(os.path.join(O2, "fallback_R10.json")))
        # an input that differs from the frozen record is refused
        pseudo.write_json(os.path.join(p02, "frozen_config.json"),
                          dict(fc, inputs_sha256=dict(fc["inputs_sha256"], **{"opening_inner.npz": "0" * 64})))
        self.assertEqual(t02.main(["fallback-fit", "--combo", "R20", "--out-dir", os.path.join(self.work, "ff3")]), 1)

    def test_cli_flag_rules(self):
        for argv in (["fallback-fit"], ["fallback-fit", "--combo", "R20", "--dry-run"],
                     ["fallback-fit", "--combo", "R10", "--fallback", "R10"],
                     ["opening", "--combo", "R20"], ["apply", "--fallback-file", "x.json"],
                     ["fallback-fit", "--combo", "R20", "--out-dir", os.path.join(REPO, "tools")]):
            with self.subTest(argv=argv):
                self.assertEqual(t02.main(argv), 1)


# ================================================================ the committed fallback and the real self-check
class Committed(unittest.TestCase):
    P02 = os.path.join(REPO, "results", "prereg02", "rank")
    FB = os.path.join(REPO, "results", "test02", "rank", "fallback_R10.json")

    def test_committed_fallback_r10(self):
        if not os.path.exists(self.FB):
            self.skipTest("results/test02/rank/fallback_R10.json not written yet")
        fb = json.load(open(self.FB, encoding="utf-8"))
        fc = json.load(open(os.path.join(self.P02, "frozen_config.json"), encoding="utf-8"))
        tune = json.load(open(os.path.join(self.P02, "tune_report.json"), encoding="utf-8"))
        r10 = next(c for c in tune["combinations"] if c["id"] == "R10")
        self.assertEqual(fb["format"], t02.FALLBACK_FORMAT)
        self.assertEqual(fb["combo"], t02.R10)
        self.assertTrue(fb["selfcheck_R20_passed"])
        self.assertEqual(fb["selfcheck_R20"]["max_abs_diff"], 0.0)
        self.assertEqual(fb["dumps"], fc["dumps"])
        self.assertEqual(fb["inner_oof_rank_balanced"], r10["oof"]["rank_balanced"])
        self.assertAlmostEqual(fb["inner_oof_rank_balanced"], 0.5687, places=4)
        np.testing.assert_allclose(fuse.basis() @ np.array(fb["params"]["beta"]), fb["params"]["b"], atol=1e-12)
        self.assertEqual(fb["refit"]["fit_questions"], 25400)
        for k in ("inner_keys.npz", "pseudo_inner.json", "opening_inner.npz", "inner_p02e9_d4.npz"):
            self.assertEqual(fb["inputs_sha256"][k], fc["inputs_sha256"][k], k)

    @unittest.skipUnless(os.environ.get("C10_SLOW"), "set C10_SLOW=1 to refit R20 and R10 on the real inputs")
    def test_real_selfcheck_and_r10_reproduce(self):
        out = tempfile.mkdtemp(prefix="c10_t02_ff_")
        try:
            self.assertEqual(t02.main(["fallback-fit", "--combo", "R20", "--out-dir", out]), 0)
            chk = json.load(open(os.path.join(out, "fallback_selfcheck_R20.json"), encoding="utf-8"))
            self.assertTrue(chk["passed"])
            self.assertEqual(chk["max_abs_diff"], 0.0)
            if os.path.exists(self.FB):
                self.assertEqual(t02.main(["fallback-fit", "--combo", "R10", "--out-dir", out]), 0)
                new = json.load(open(os.path.join(out, "fallback_R10.json"), encoding="utf-8"))
                self.assertEqual(new["params"], json.load(open(self.FB, encoding="utf-8"))["params"])
        finally:
            shutil.rmtree(out, ignore_errors=True)


# ================================================================ small pieces
class Pieces(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t02_rank_p_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_probs(self, name, rows):
        path = os.path.join(self.tmp, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(fuse.PROB_COLUMNS)
            for r in rows:
                w.writerow(r)
        return path

    def probs(self, n, bump=0.0, flip=()):
        rng = np.random.default_rng(1)
        rows = []
        for i in range(n):
            p = rng.dirichlet(np.ones(10))
            if i in flip:
                p = p[::-1]
            if i == 0 and bump:
                k = int(np.argmin(p))
                p[k] += bump
                p[(k + 5) % 10] -= bump
            rows.append(["val_q_%04d" % (i + 1), 12, 12] + [repr(float(x)) for x in p])
        return rows

    def seam(self, mine_rows, ref_rows, tol=1e-4, smoke=False):
        a, b = self.write_probs("mine.csv", mine_rows), self.write_probs("ref.csv", ref_rows)
        da, db = os.path.join(self.tmp, "da"), os.path.join(self.tmp, "db")
        decide.run(a, da)
        decide.run(b, db)
        return t02.seam_compare(a, os.path.join(da, "pred_B.csv"), b, os.path.join(db, "pred_B.csv"), tol, smoke)

    def test_seam_compare(self):
        ref = self.probs(30)
        r = self.seam(ref, ref)
        self.assertTrue(r["passed"] and r["byte_identical_probs"] and r["byte_identical_pred_B"])
        r = self.seam(self.probs(30, bump=5e-5), ref)
        self.assertTrue(r["passed"])
        self.assertFalse(r["byte_identical_probs"])
        r = self.seam(self.probs(30, bump=2e-4), ref)
        self.assertFalse(r["passed"])
        self.assertTrue(self.seam(self.probs(30, bump=2e-4), ref, tol=1e-3)["passed"])
        flips = [i for i in range(30) if decide.rule_b(list(np.array([float(x) for x in ref[i][3:]])[::-1]))
                 != decide.rule_b([float(x) for x in ref[i][3:]])]
        r4 = self.seam(self.probs(30, flip=flips[:4]), ref, tol=1.0)
        self.assertEqual((r4["rule_b_differences"], r4["passed"]), (4, True))
        r5 = self.seam(self.probs(30, flip=flips[:5]), ref, tol=1.0)
        self.assertEqual((r5["rule_b_differences"], r5["passed"]), (5, False))
        # a count mismatch (a question parsed on one side only) fails
        mine = [list(r) for r in ref]
        mine[1] = mine[1][:2] + [0] + [""] * 10
        self.assertFalse(self.seam(mine, ref, tol=1.0)["passed"])

    def test_theta_and_b_check(self):
        par = {"a": 1.0, "gamma": 2.0, "w": 3.0, "beta": [0.5] * 9}
        self.assertEqual(t02.theta_of(par, True).tolist(), [1.0, 2.0, 3.0] + [0.5] * 9)
        self.assertEqual(t02.theta_of(par, False).tolist(), [1.0, 2.0] + [0.5] * 9)
        B = fuse.basis()
        t02.check_b(dict(par, b=list(B @ np.full(9, 0.5))), B, "x")
        with self.assertRaises(t02.T02Error):
            t02.check_b(dict(par, b=list(B @ np.full(9, 0.5) + 1e-11)), B, "x")


class SourceHygiene(unittest.TestCase):
    FILES = ("tools/t02_rank.py", "colab/build_test02_rank_nb.py", "tests/test_t02_rank.py",
             "tests/test_t02_nb_rank.py")

    def test_no_label_file_token(self):
        word = "ans" + "wers"
        for rel in self.FILES:
            path = os.path.join(REPO, *rel.split("/"))
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    self.assertNotIn(word, f.read().lower(), rel)

    def test_rank_tool_imports(self):
        import re
        with open(os.path.join(TOOLS, "t02_rank.py"), encoding="utf-8") as f:
            src = f.read()
        mods = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", src, re.M))
        self.assertFalse({m for m in mods if m.startswith(("p02_player", "t02_player", "t02_checks"))}, mods)
        self.assertLessEqual({m for m in mods if m.startswith(("p02_", "t02_"))},
                             {"p02_rank_fuse", "p02_rank_opening", "p02_rank_pseudo"})
        self.assertNotIn("results/prereg02/player", src.replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
