"""Tests for tools/t04_rank.py and tools/t04_fake_tests.py (spec results/test04/design/spec.md 10.1).

Fast part (default): the Q02 recipe on the committed validation inputs (byte-identical to
results/prereg04/rank/rank_val_probs.csv, the three seam_val dkappa bit for bit); the replay, dkappa and exit-code
rules; the registration-revision check with git mocked and (review round 3, class RealGit) on real read-only git
calls against the Chinese docs/PREREG_04_OPPONENT.md; the notebook-seam recompute (the notebook's own c0_of, torch
CPU); and a dry-run tree built once in a temporary folder outside the repository: t02_fake_tests from-val (the test02
notebook tree), a synthetic test02 apply output (opening_test.npz sliced from the committed opening_val.npz, a
report), t04_fake_tests from-val, then t04_rank.py unpack / opp / apply --dry-run in process. opp's replay is replaced
by the committed val_o4.npz rows (module functions load_tables / o4_scores), so the seam file must still be
byte-identical; the real replay runs in the slow part. Negative drills copy the tree and change one thing. Real-mode
rules run with the module's path constants pointed at temporary trees and git mocked (setattr).
Slow part (C10_SLOW=1 or 2, about 1.5 min): opp's core on seam_val with 6 workers -> o4_seam_val.npz file SHA-256
30d312c0...; a 2,000-slot segment scored alone equals the same rows scored in the concatenation.
No label file is opened; tools/score.py never runs; no GPU.
Run: python -m pytest tests/test_t04_rank.py
"""

import contextlib
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
import t02_fake_tests as tf  # noqa: E402
import t02_rank as t2  # noqa: E402
import t02_test_bundle as tb  # noqa: E402
import t04_fake_tests as f4  # noqa: E402
import t04_rank as t04  # noqa: E402

SLOW = os.environ.get("C10_SLOW", "")
P02 = os.path.join(REPO, "results", "prereg02", "rank")
P03 = os.path.join(REPO, "results", "prereg03")
P04 = os.path.join(REPO, "results", "prereg04", "rank")
SEAM_CSV_SHA = "3c7ab658dc558ce8c6aefdd7b0747de1a60de575d151e9eee96ca6775f442e5c"
DK_SEAM = {"O4": 0.1050265798301427, "SWA": 0.03631175839656169, "SWB": 0.034356617152967346}
PREFIX = {"public": "pub_", "private": "priv_"}
TW = dt.timezone(dt.timedelta(hours=8))
B = None   # the module fixture (setUpModule)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rj(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wj(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)


@contextlib.contextmanager
def patched(mod, **kw):
    old = {k: getattr(mod, k) for k in kw}
    for k, v in kw.items():
        setattr(mod, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(mod, k, v)


def run(args, **patch):
    """t04_rank.main in process -> (exit code, stdout + stderr)."""
    buf = io.StringIO()
    with patched(t04, **patch), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = t04.main(args)
    return code, buf.getvalue()


def val_to_real(q):
    return q if q.startswith("val_") else "val_" + q.split("_", 1)[1]


class FakeO4:
    """Stands in for po.val_scores: the committed validation O4 rows of the (re-keyed) slots; optional injected
    replay failures (global row numbers) and a perturbation of the seam_val o4 (a local drift)."""

    def __init__(self, fail=(), perturb=False):
        with np.load(t04.VAL_O4, allow_pickle=False) as z:
            self.z = {k: z[k] for k in z.files}
        self.index = {(q, int(s)): i for i, (q, s) in enumerate(zip(self.z["question_id"].tolist(),
                                                                     self.z["slot"].tolist()))}
        self.fail, self.perturb, self.calls = set(fail), perturb, []

    def __call__(self, slots, tabs, workers, tmp_root):
        self.calls.append({"n": len(slots), "workers": workers, "tmp_root": tmp_root})
        idx = np.array([self.index[(val_to_real(q), int(s))] for q, s, _c, _g in slots])
        out = {k: self.z[k][idx].copy() for k in ("o4", "t4", "n_moves", "anomalies", "moves_parsed",
                                                  "replay_failed")}
        for g in self.fail:
            out["replay_failed"][g] = True
            out["anomalies"][g] = 1
        if self.perturb:
            out["o4"][0, 0] += 1e-9
        return out


def fake_tables():
    return None, {"t": {"r0": None}, "o": {"r0": None}}, {"sha256": t04.COUNTS_SHA, "count_report_sha256": "fake",
                                                          "file": "fake (tests)"}, t04.TABLES_SHA


FAST_OPP = {"load_tables": fake_tables}


# ================================================================ the dry-run tree (built once)
class Tree:
    def __init__(self, base):
        self.base = base
        S = self.S = os.path.join(base, "s")
        tf.from_val(S, log=lambda *a: None)
        self.D2 = os.path.join(S, "results", "test02")
        self.T = os.path.join(S, "data", "test")
        self.C = os.path.join(self.D2, "checks")
        self.O2 = os.path.join(S, "out", "rank")
        wj(os.path.join(self.C, "rank", "t1.json"),
           {"verdict": "PASS", "comp": "rank", "complete": False, "stage": "normalized",
            "subchecks_run": ["T1.1", "T1.2", "T1.3"], "provenance": {"mode": "dry-run"},
            "note": "tests/test_t04_rank.py: structure-only stand-in (dry run)"})
        self.write_o2(self.O2, "dry-run")
        f4.from_val(S, log=lambda *a: None)
        self.zip = os.path.join(S, "drive", f4.ZIP_NAME)
        self.R4 = os.path.join(base, "r4")
        code, out = run(["unpack", "--dry-run", "--zip", self.zip] + self.dirs(self.R4))
        assert code == 0, out
        self.fake = FakeO4()
        code, out = run(["opp", "--dry-run", "--workers", "3"] + self.dirs(self.R4), o4_scores=self.fake, **FAST_OPP)
        assert code == 0, out
        self.opp_out = out
        code, out = run(["apply", "--dry-run"] + self.dirs(self.R4))
        assert code == 0, out
        self.apply_out = out

    def dirs(self, R4, D2=None, O2=None, T=None, C=None):
        out = ["--results-dir", R4, "--t02-results-dir", D2 or self.D2, "--t02-out-dir", O2 or self.O2,
               "--tests-dir", T or self.T]
        return out + (["--checks-dir", C] if C else [])

    def write_o2(self, O2, mode):
        """test02's apply output as t04 reads it: opening_test.npz sliced from the committed opening_val.npz
        (seam_val whole, public / private = validation questions 1-400 / 401-800 re-keyed), a report, and R20's
        submission (prereg02 pred_B.csv rows 1-800 re-keyed)."""
        os.makedirs(O2, exist_ok=True)
        vk = rf.load_val_keys(os.path.join(P02, "dumps"))
        first = [rows[0] for _, rows in vk["groups"]] + [vk["n"]]
        with np.load(os.path.join(P02, "opening_val.npz"), allow_pickle=False) as z:
            ov = {k: z[k] for k in z.files}
        parts = {"seam_val": (0, vk["n"]), "public": (first[0], first[400]), "private": (first[400], first[800])}
        arr = {k: [] for k in ("set", "question_id", "slot", "color", "ll", "o", "has", "n_terms", "n_moves")}
        for s, (lo, hi) in parts.items():
            q = ov["question_id"][lo:hi].tolist()
            arr["set"] += [s] * (hi - lo)
            arr["question_id"] += q if s == "seam_val" else [PREFIX[s] + x[len("val_"):] for x in q]
            for k in ("slot", "color", "ll", "o", "has", "n_terms", "n_moves"):
                arr[k].append(ov[k][lo:hi])
        qfiles = {"seam_val": t04.SEAM_QUESTIONS, "public": os.path.join(self.T, t2.TEST_FILES["public"]),
                  "private": os.path.join(self.T, t2.TEST_FILES["private"])}
        meta = {"format": t2.OPENING_FORMAT, "seam_bitwise_equal": True,
                "frozen_opening_build_id": t2.FROZEN_OPENING_BUILD_ID,
                "inputs": {s: {"sha256": sha(qfiles[s])} for s in qfiles}, "note": "synthetic (tests)"}
        np.savez(os.path.join(O2, "opening_test.npz"), set=np.array(arr["set"]),
                 question_id=np.array(arr["question_id"]), slot=np.concatenate(arr["slot"]),
                 color=np.concatenate(arr["color"]), ll=np.concatenate(arr["ll"]), o=np.concatenate(arr["o"]),
                 has=np.concatenate(arr["has"]), n_terms=np.concatenate(arr["n_terms"]),
                 n_moves=np.concatenate(arr["n_moves"]), meta=np.array(json.dumps(meta)))
        rows = list(csv.reader(open(os.path.join(P02, "pred_B.csv"), newline="", encoding="utf-8")))
        with open(os.path.join(O2, "submission_rank.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(rows[0])
            for i in range(800):
                q, lab = rows[1 + i]
                w.writerow([(PREFIX["public"] if i < 400 else PREFIX["private"]) + q[len("val_"):], lab])
        gpu = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        wj(os.path.join(O2, "apply_test_report.json"),
           {"mode": mode, "fallback_used": False, "combo": {"id": "R20"},
            "seam_local": {"passed": True, "tol": 1e-4, "tol_check": {"seam_dump_gpu": gpu, "prereg02_dump_gpu": gpu}},
            "check_submission": {"mode": "official", "errors": []},
            "outputs_sha256": {"submission_rank.csv": sha(os.path.join(O2, "submission_rank.csv"))},
            "note": "synthetic (tests)"})

    def copy_r4(self, name, parts=("sw", "opp", "unpack_sw.json")):
        dst = os.path.join(self.base, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        for p in parts:
            src = os.path.join(self.R4, p)
            (shutil.copytree if os.path.isdir(src) else shutil.copyfile)(src, os.path.join(dst, p))
        return dst

    def copy_sw(self, name):
        """A copy of the fake notebook folder (S/drive/sw) to edit before zipping."""
        dst = os.path.join(self.base, name, "sw")
        if os.path.exists(os.path.dirname(dst)):
            shutil.rmtree(os.path.dirname(dst))
        shutil.copytree(os.path.join(self.S, "drive", "sw"), dst)
        return dst


def setUpModule():
    global B
    base = tempfile.mkdtemp(prefix="c10_t04rank_")
    assert not t2.is_inside(base, REPO)
    B = Tree(base)


def tearDownModule():
    if B is not None and not os.environ.get("C10_KEEP"):
        shutil.rmtree(B.base, ignore_errors=True)


def zip_sw(sw, zpath):
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    return f4.write_zip(sw, zpath)


def rewrite_npz(path, **arrays):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d.update(arrays)
    with open(path, "wb") as f:
        np.savez(f, **d)


def update_marker(dumps, stem):
    p = os.path.join(dumps, f"{stem}.npz")
    mp = os.path.join(dumps, f"_DONE_{stem}.json")
    m = rj(mp)
    m.update(sha256=sha(p), bytes=os.path.getsize(p))
    wj(mp, m)


def edit_json(path, fn):
    obj = rj(path)
    fn(obj)
    wj(path, obj)


# ================================================================ recipe on the committed validation inputs
class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vk = rf.load_val_keys(os.path.join(P02, "dumps"))
        cls.c5 = t04.load_c5("seam_val", os.path.join(P02, "dumps"), cls.vk,
                             "423581b4fddd5c2c160ebf691f4e86b9e00c4fe743d5cfb23df7891f4a2ca939")
        with np.load(os.path.join(P02, "opening_val.npz"), allow_pickle=False) as z:
            cls.op = {k: z[k] for k in ("question_id", "slot", "color", "o", "has")}
        cls.op["set"] = np.array(["seam_val"] * len(cls.op["question_id"]))
        with np.load(t04.VAL_O4, allow_pickle=False) as z:
            cls.o4z = {k: z[k] for k in z.files}
        cls.sw = {x: {"d4": rf.load_dump(os.path.join(P03, f"rank_{x}", "dumps", f"val_p03{x}_d4.npz"), cls.vk["n"], 8),
                      "swap": p3.load_swap(os.path.join(P03, f"rank_{x}", "dumps", f"val_p03{x}_s0_swap.npz"),
                                           cls.vk["n"])} for x in ("a", "b")}
        cls.fc = rj(t04.FROZEN_CONFIG)
        cls.fz = cls.fuse(cls.o4z)
        cls.tmp = tempfile.mkdtemp(prefix="c10_t04recipe_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def fuse(cls, o4z):
        return t04.fuse_set("seam_val", cls.vk, t04.SEAM_QUESTIONS, cls.c5, cls.op, o4z, cls.sw, cls.fc, False)

    def test_probs_byte_identical_to_prereg04(self):
        p = os.path.join(self.tmp, "rank_val_probs.csv")
        t2.write_probs(p, self.fz["rows"])
        self.assertEqual(sha(p), SEAM_CSV_SHA)
        self.assertEqual(t2.pin_sha256(os.path.join(P04, "rank_val_probs.csv")), SEAM_CSV_SHA)

    def test_three_dkappa_bit_for_bit(self):
        col = t04.colour_check(self.vk, self.fz)
        self.assertEqual({k: col[k]["dkappa"] for k in DK_SEAM}, DK_SEAM)
        self.assertEqual(col["O4"]["sub_questions"], 2986)
        self.assertEqual(sum(v["sub_questions"] for v in col["O4"]["per_k_record_only"].values()), 2986)

    def test_mask_equal_to_parsed_is_the_unmasked_computation(self):
        parsed = self.vk["parsed"]
        lp = rf.log_softmax(np.where(parsed[:, None], self.o4z["o4"], 0.0))
        S = rf.pool_matrix([[i for i in rows if parsed[i]] for _, rows in self.vk["groups"]], self.vk["n"])
        self.assertEqual(rf.pooled(S, lp, "log").tobytes(), self.fz["E"]["O4"].tobytes())

    def test_failed_slots_leave_E_O4_and_kappa(self):
        o4z = dict(self.o4z, replay_failed=self.o4z["replay_failed"].copy())
        rows0 = self.vk["groups"][0][1]
        o4z["replay_failed"][rows0[0]] = True
        fz = self.fuse(o4z)
        lp = rf.log_softmax(np.where(fz["mask"][:, None], self.o4z["o4"], 0.0))
        want0 = np.mean(lp[rows0[1:]], axis=0)
        self.assertTrue(np.allclose(fz["E"]["O4"][0], want0, rtol=0, atol=1e-12))   # sum of x / k vs mean: rounding
        self.assertFalse(np.allclose(fz["E"]["O4"][0], self.fz["E"]["O4"][0], rtol=0, atol=1e-6))
        self.assertEqual(fz["E"]["O4"][1:].tobytes(), self.fz["E"]["O4"][1:].tobytes())   # other questions untouched
        self.assertEqual(fz["E"]["SWA"].tobytes(), self.fz["E"]["SWA"].tobytes())          # SW still on parsed
        self.assertEqual(fz["n"].tobytes(), self.fz["n"].tobytes())                        # temperature n = C5's
        self.assertFalse(np.array_equal(fz["P"][0], self.fz["P"][0]))
        self.assertEqual(fz["P"][1:].tobytes(), self.fz["P"][1:].tobytes())
        # kappa by hand: the failed slot is not in the O4 / T4 sub-question
        col = t04.colour_check(self.vk, fz)
        sub = p3.val_colour_subquestions(dict(self.vk, parsed=fz["mask"]))
        self.assertNotIn(rows0[0], [i for _, _, g in sub for i in g])
        lpt = rf.log_softmax(np.where(fz["mask"][:, None], self.o4z["t4"], 0.0))
        groups = [g for _, _, g in sub]
        hand = float(np.mean(p3.kappa_questions(lpt, groups) - p3.kappa_questions(lp, groups)))
        self.assertEqual(col["O4"]["dkappa"], hand)
        self.assertNotEqual(col["O4"]["dkappa"], DK_SEAM["O4"])

    def test_empty_o4_mask_gives_a_zero_E(self):
        o4z = dict(self.o4z, replay_failed=self.o4z["replay_failed"].copy())
        rows0 = self.vk["groups"][0][1]
        o4z["replay_failed"][rows0] = True
        fz = self.fuse(o4z)
        self.assertEqual(fz["E"]["O4"][0].tolist(), [0.0] * 10)
        self.assertEqual(fz["info"]["questions_with_empty_o4_mask"], 1)


# ================================================================ pure rules
class Rules(unittest.TestCase):
    def test_load_tables_counts_refusal(self):
        """rev 3 check (items 9, 17): opp's counts file missing or not the one count_report.json records -> 2 with
        the restore hint (p04_opp's P04Error used to surface without it; a missing file was an internal error)."""
        with tempfile.TemporaryDirectory() as tmp:
            with patched(t04.po, COUNTS_REAL=os.path.join(tmp, "absent.npz")):
                with self.assertRaises(t04.T04Error) as cm:
                    t04.load_tables()
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("missing", str(cm.exception))
            self.assertIn("design_scratch", str(cm.exception))
            p = os.path.join(tmp, "counts_fit.npz")
            with open(p, "wb") as f:
                f.write(b"x")

            def differs(c, a):
                raise t04.po.P04Error("sha256 2d711642b726 differs from count_report.json f6c6debe9063")
            with patched(t04.po, COUNTS_REAL=p, registered_counts=differs):
                with self.assertRaises(t04.T04Error) as cm:
                    t04.load_tables()
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("differs from count_report.json", str(cm.exception))
            self.assertIn("design_scratch", str(cm.exception))

    def test_replay_rate_literal(self):
        self.assertTrue(t04.rate_ok(25, 5000))
        self.assertFalse(t04.rate_ok(26, 5000))
        self.assertTrue(t04.rate_ok(0, 19156))
        self.assertFalse(t04.rate_ok(0, 0))

    def stats(self, seam=(0, 5000), pub=(0, 5000), priv=(0, 5000)):
        return {s: {"failures": f, "slots": n} for s, (f, n) in zip(t04.SETS, (seam, pub, priv))}

    def test_opp_decision(self):
        good = {"public": {"dkappa": 0.09, "sub_questions": 400}, "private": {"dkappa": 0.1, "sub_questions": 400}}
        self.assertEqual(t04.opp_decision(self.stats(), True, True, good)[:3], (0, True, True))
        # seam_val 26/5000 -> the registered 'no 04' (3), even though it cannot be byte-identical then
        code, passed, usable, reg, dr = t04.opp_decision(self.stats(seam=(26, 5000)), False, False, good)
        self.assertEqual((code, passed, usable), (3, False, False))
        self.assertTrue(reg[0].startswith("replay:seam_val"))
        # seam_val 1/5000 (<= 0.5 %) is not byte-identical: local drift (4)
        code, passed, usable, reg, dr = t04.opp_decision(self.stats(seam=(1, 5000)), False, False, good)
        self.assertEqual((code, passed, usable, reg), (4, False, False, []))
        self.assertEqual(t04.opp_decision(self.stats(pub=(26, 5000)), True, True, good)[:3], (3, False, False))
        self.assertEqual(t04.opp_decision(self.stats(), True, False, good)[:3], (4, False, False))
        for bad in ({"dkappa": 0.0, "sub_questions": 400}, {"dkappa": -0.01, "sub_questions": 400},
                    {"dkappa": float("nan"), "sub_questions": 400}, {"dkappa": None, "sub_questions": 0}):
            with self.subTest(bad=bad):
                code, passed, usable, reg, _ = t04.opp_decision(self.stats(), True, True, dict(good, private=bad))
                self.assertEqual((code, passed, usable), (3, True, False))
        # review round 1 (rev 3 item 7): with a seam_val drift the O4 dkappa is not judged (recorded only) -> 4
        bad = {"dkappa": -0.01, "sub_questions": 400}
        for fe, de in ((False, True), (True, False), (False, False)):
            with self.subTest(seam_file_equal=fe, seam_dk_equal=de, public_dkappa="negative"):
                code, passed, usable, reg, dr = t04.opp_decision(self.stats(), fe, de, dict(good, public=bad))
                self.assertEqual((code, passed, usable, reg), (4, False, False, []))
                self.assertTrue(dr)
        # a replay rate above 0.5 % stays the registered 3, drift or not
        code, _p, _u, reg, _dr = t04.opp_decision(self.stats(pub=(26, 5000)), False, True, dict(good, public=bad))
        self.assertEqual(code, 3)
        self.assertEqual([r for r in reg if r.startswith("dkappa:")], [])

    def test_dkappa_table(self):
        opp = {"mode": "dry-run", "seam_val": {"file_sha256_equal": True},
               "sets": {s: {"failures": 0, "slots": 5000} for s in t04.SETS}}
        good = {s: {src: {"dkappa": 0.05, "sub_questions": 400} for src in t04.OPP_SOURCES} for s in t04.TEST_SETS}
        self.assertEqual(t04.apply_decision(opp, True, True, good), (True, []))
        for s in t04.TEST_SETS:
            for src in t04.OPP_SOURCES:
                for label, kv in (("zero", {"dkappa": 0.0, "sub_questions": 400}),
                                  ("negative", {"dkappa": -1e-9, "sub_questions": 400}),
                                  ("nan", {"dkappa": float("nan"), "sub_questions": 400}),
                                  ("no sub-question", {"dkappa": None, "sub_questions": 0})):
                    with self.subTest(set=s, source=src, case=label):
                        col = copy.deepcopy(good)
                        col[s][src] = kv
                        usable, reasons = t04.apply_decision(opp, True, True, col)
                        self.assertFalse(usable)
                        self.assertEqual(reasons, [f"dkappa:{s}.{src}: {kv['dkappa']!r} (needs > 0)"])
        opp2 = copy.deepcopy(opp)
        opp2["sets"]["public"]["failures"] = 26
        self.assertFalse(t04.apply_decision(opp2, True, True, good)[0])

    def test_smoke_records_dkappa_only(self):
        """Integration fix (2026-09-30, L2): in --smoke the dkappa signs are recorded only; replay rates still count;
        every other mode is unchanged."""
        bad = {"dkappa": -1e-4, "sub_questions": 12}
        good = {"public": {"dkappa": 0.09, "sub_questions": 12}, "private": {"dkappa": 0.1, "sub_questions": 12}}
        self.assertEqual(t04.opp_decision(self.stats(), False, False, dict(good, private=bad), smoke=True)[:5],
                         (0, True, True, [], []))
        self.assertEqual(t04.opp_decision(self.stats(), True, True, dict(good, private=bad))[:3], (3, True, False))
        self.assertEqual(t04.opp_decision(self.stats(pub=(26, 5000)), True, True, good, smoke=True)[:3],
                         (3, False, False))
        opp = {"mode": "smoke", "seam_val": {"file_sha256_equal": False},
               "sets": {s: {"failures": 0, "slots": 160} for s in t04.SETS}}
        col = {s: {src: {"dkappa": 0.05, "sub_questions": 12} for src in t04.OPP_SOURCES} for s in t04.TEST_SETS}
        col["private"]["SWA"] = bad
        self.assertEqual(t04.apply_decision(opp, True, True, col, smoke=True), (True, []))
        self.assertFalse(t04.apply_decision(dict(opp, mode="dry-run"), True, True, col)[0])
        opp["sets"]["public"]["failures"] = 1
        self.assertFalse(t04.apply_decision(opp, True, True, col, smoke=True)[0])

    def test_forbidden_write_areas(self):
        for rel in ("results/prereg04/x", "results/prereg01", "results/test02/rank", "results/test02_rehearsal/a",
                    "data/local/x", "results/score_ledger.csv"):
            with self.subTest(rel=rel):
                self.assertIsNotNone(t04.forbidden_area(os.path.join(REPO, *rel.split("/"))))
        for rel in ("results/test04", "results/test04/rank", "results/test04_rehearsal"):
            self.assertIsNone(t04.forbidden_area(os.path.join(REPO, *rel.split("/"))))

    def test_label_word_refused(self):
        with self.assertRaises(t04.T04Error) as cm:
            t04.refuse_label_path(os.path.join(tempfile.gettempdir(), "x_" + "ans" + "wers.csv"))
        self.assertEqual(cm.exception.code, 2)

    def test_drive_archives_record_only(self):
        """Review round 2: the Drive sw_failed_* folders (config.json prior_sw_folders) against the local archives:
        recorded, a warning when they differ, never a refusal; absent in the fake / older records."""
        att = [{"n": 1, "name": "attempt_1"}]
        rec, warn = t04.drive_archives_record({"prior_sw_folders": {"sw_failed_1": [], "sw_interrupted_1": []},
                                               "attempt": 1}, att)
        self.assertEqual((rec["checked"], rec["equal"], warn), (True, True, None))
        self.assertEqual(rec["notebook_attempt"], 1)      # an apply-exit-1 archive: the notebook still says 1
        rec, warn = t04.drive_archives_record({"prior_sw_folders": {}}, att)
        self.assertEqual((rec["equal"], rec["local_archives_as_drive_names"]), (False, ["sw_failed_1"]))
        self.assertIn("differ from the local archives", warn)
        rec, warn = t04.drive_archives_record({"prior_sw_folders": {"sw_failed_2": []}}, [])
        self.assertIn("sw_failed_2", warn)
        self.assertEqual(t04.drive_archives_record({}, att)[0]["checked"], False)
        self.assertIsNone(t04.drive_archives_record({}, att)[1])
        # review round 3: an uncounted archive is sw_exempt_N on Drive, a counted one sw_failed_N
        att2 = [{"n": 1, "name": "attempt_1", "record": {"counts_toward_rerun": False}},
                {"n": 2, "name": "attempt_2", "record": {"counts_toward_rerun": True}}]
        rec, warn = t04.drive_archives_record({"prior_sw_folders": {"sw_exempt_1": ["x"], "sw_failed_2": ["y"],
                                                                    "sw_interrupted_1": []}}, att2)
        self.assertEqual((rec["equal"], rec["local_archives_as_drive_names"], warn),
                         (True, ["sw_exempt_1", "sw_failed_2"], None))
        rec, warn = t04.drive_archives_record({"prior_sw_folders": {"sw_failed_1": [], "sw_failed_2": []}}, att2)
        self.assertFalse(rec["equal"])
        self.assertIn("sw_exempt_1", warn)
        self.assertEqual(t04.drive_archive_names(3, True), ("sw_failed_3", "test04_rank_results_failed_3.zip"))
        self.assertEqual(t04.drive_archive_names(3, False), ("sw_exempt_3", "test04_rank_results_exempt_3.zip"))
        self.assertEqual(t04.drive_source_names(None), ("sw", "test04_rank_results.zip"))
        self.assertEqual(t04.drive_source_names("sw_interrupted_12"),
                         ("sw_interrupted_12", "test04_rank_results_interrupted_12.zip"))
        for bad in ("sw_failed_1", "sw_interrupted_0", "sw_interrupted_", "sw_interrupted_1/..", "SW", ""):
            with self.subTest(drive_folder=bad):
                with self.assertRaises(t04.T04Error) as cm:
                    t04.drive_source_names(bad)
                self.assertEqual(cm.exception.code, 2)
        self.assertEqual([r["run"] for r in t04.producing_runs({"runs": [
            {"run": 1, "status": "preflight_failed", "start": "a"}, {"run": 2, "status": "interrupted", "start": "b"},
            {"run": 3, "status": "seam_failed", "start": "c"}, {"run": 4, "status": "completed", "start": "d"}]})],
            [3, 4])

    def test_verdict_kind(self):
        self.assertEqual(t04.verdict_kind({"exit_code": 4}), "drift")
        self.assertEqual(t04.verdict_kind({"exit_code": 3}), "registered")
        self.assertEqual(t04.verdict_kind({"exit_code": 0}), "registered")
        self.assertEqual(t04.verdict_kind({"usable_so_far": False, "unusable_reasons": ["seam_file: x"]}), "drift")
        self.assertEqual(t04.verdict_kind({"usable_so_far": False, "unusable_reasons": ["replay:public: x"]}),
                         "registered")


# ================================================================ the notebook seam recompute
class NotebookSeam(unittest.TestCase):
    def source_nb(self):
        return t04.C0_SOURCE_NB if os.path.exists(t04.C0_SOURCE_NB) else t04.C0_FALLBACK_NB

    def test_c0_of_is_t02s_verbatim(self):
        _qs, _c0, src = t04.c0_functions(self.source_nb())
        _qs2, _c02, ref = t04.c0_functions(os.path.join(REPO, "colab", "test02_rank.ipynb"))
        self.assertEqual(src["c0_of"], ref["c0_of"])
        self.assertEqual(src["question_spans"], ref["question_spans"])
        self.assertIn("torch.from_numpy", src["c0_of"])

    def test_same_verdict_as_the_notebook(self):
        qs, c0, _src = t04.c0_functions(self.source_nb())
        rng = np.random.default_rng(0)
        qids = np.array([f"q{i}" for i in range(6) for _ in range(5)])
        parsed = np.ones(30, dtype=bool)
        parsed[3] = False
        L = rng.normal(0, 2, (30, 8, 10)).astype(np.float32)
        L[:5] = 0.0                                    # question 0: uniform, so a small change stays visible
        spans = qs(qids)
        self.assertEqual(t04.c0_max_diff(c0, spans, parsed, L, L.copy()), 0.0)
        near = L.copy()
        near[0, 0, 0] += np.float32(1e-5)
        far = L.copy()
        far[0, 0, 0] += np.float32(2.0)
        d_near = t04.c0_max_diff(c0, spans, parsed, near, L)
        d_far = t04.c0_max_diff(c0, spans, parsed, far, L)
        self.assertTrue(0 < d_near <= 1e-4 < d_far, (d_near, d_far))
        unparsed = L.copy()
        unparsed[3] += 100
        self.assertEqual(t04.c0_max_diff(c0, spans, parsed, unparsed, L), 0.0)   # unparsed rows never count
        sw = rng.normal(0, 2, (30, 10)).astype(np.float32)
        self.assertEqual(t04.c0_max_diff(c0, spans, parsed, sw[:, None, :], sw[:, None, :]), 0.0)


# ================================================================ registration revision check (git mocked)
class BugRevision(unittest.TestCase):
    DOC = "| rev 1 | a |\n| rev 2 | b |\n| rev 3 | c |\n| rev 4 | (commit) | fix of a bug in t04_rank.py |\n"

    def check(self, n, overturned, doc=DOC, when="2026-11-10T12:00:00+08:00", committed=True):
        def req(paths, what):
            if not committed:
                raise t04.T04Error(f"{what}: not committed", 2)
            return {"files": len(paths)}
        with patched(t04, require_committed=req, git_show_head=lambda rel: doc,
                     git_first_commit_with=lambda rel, needle: ("abc123", when) if needle in doc else (None, None)):
            return t04.check_bug_revision(n, overturned)

    def test_ok(self):
        rec = self.check(4, [("opp_report.json created", "2026-11-05T10:00:00+08:00")])
        self.assertEqual((rec["n"], rec["commit"]), (4, "abc123"))

    def test_refusals(self):
        cases = {"row missing": dict(n=5, overturned=[]),
                 "N < 4": dict(n=3, overturned=[]),
                 "not committed": dict(n=4, overturned=[], committed=False),
                 "revision before the record": dict(n=4, overturned=[("t04_apply_report.json created",
                                                                        "2026-11-11T00:00:00+08:00")])}
        for label, kw in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(t04.T04Error) as cm:
                    self.check(**kw)
                self.assertEqual(cm.exception.code, 2)


class RealGit(unittest.TestCase):
    """Review round 3: the registration-revision check on real git, nothing mocked but require_committed while
    docs/PREREG_04_OPPONENT.md has local edits. t02_rank.git decodes with the locale (cp950 here) and returned None for
    this Chinese file, so every real `--bug-revision N` (archive, opp / apply --rerun) ended in an internal error;
    t04_rank.git decodes UTF-8 (read-only git calls; nothing is written)."""
    REL = "docs/PREREG_04_OPPONENT.md"

    def patch(self):
        rc, st = t04.git("status", "--porcelain", "--", self.REL)
        return {} if rc == 0 and not st else {"require_committed": lambda p, w: {"files": len(p), "rule": "tests"}}

    def test_show_head_of_the_chinese_registration(self):
        text = t04.git_show_head(self.REL)
        self.assertIsInstance(text, str)
        self.assertIn("| rev 2 |", text)
        self.assertIn("事前登記", text)
        self.assertIsNone(t04.git_show_head("docs/no_such_file_t04.md"))
        rc, out = t04.git("show", "HEAD:docs/no_such_file_t04.md")
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, "")

    def test_first_commit_with_a_revision_row(self):
        h, t = t04.git_first_commit_with(self.REL, "| rev 2 |")
        self.assertRegex(h or "", r"^[0-9a-f]{40}$")
        t04.parse_iso(t, "commit time")
        self.assertEqual(t04.git_first_commit_with(self.REL, "| rev 99 |"), (None, None))

    def test_check_bug_revision(self):
        with patched(t04, **self.patch()):
            with self.assertRaises(t04.T04Error) as cm:
                t04.check_bug_revision(99, [("x", "2026-11-05T10:00:00+08:00")])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("has no line starting '| rev 99 |'", str(cm.exception))

    def test_archive_bug_revision(self):
        """The reviewer's command: archive --dry-run --bug-revision N whose row is not committed -> 2 with the
        registration message, never 'unexpected internal error'; nothing is moved."""
        R4 = B.copy_r4("a_realgit")
        edit_json(os.path.join(R4, "sw", "seam_check.json"), lambda s: s.update(passed=False))
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "notebook bug", "--bug-revision",
                         "99"], **self.patch())
        self.assertEqual(code, 2, out)
        self.assertIn("has no line starting '| rev 99 |'", out)
        self.assertNotIn("unexpected internal error", out)
        self.assertNotIn("Traceback", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "failed")))
        self.assertTrue(os.path.isdir(os.path.join(R4, "sw")))


# ================================================================ the dry-run tree
class DryRunPipeline(unittest.TestCase):
    def test_seam_and_outputs(self):
        R = os.path.join(B.R4, "rank")
        self.assertEqual({n: sha(os.path.join(R, "seam_val", n)) for n in t04.P04_SEAM_PINS}, t04.P04_SEAM_PINS)
        sl = rj(os.path.join(R, "seam_val", "seam_local.json"))
        self.assertTrue(sl["passed"] and sl["byte_identical_probs"] and sl["byte_identical_pred_B"])
        self.assertEqual((sl["warnings"], sl["max_abs_diff"], sl["rule_b_differences"]), ([], 0.0, 0))
        self.assertTrue(sl["notebook_seam_local_recompute"]["passed"])
        self.assertEqual(sl["tol_check"]["tol"], 1e-4)
        ref = open(os.path.join(P04, "rank_val_probs.csv"), "rb").read().replace(b"\r\n", b"\n").split(b"\n")
        got = open(os.path.join(R, "rank_test_probs.csv"), "rb").read().split(b"\n")
        self.assertEqual((got[0], len(got), got[-1]), (ref[0], 802, b""))
        for i in range(800):
            self.assertEqual(got[1 + i], (b"pub_" if i < 400 else b"priv_") + ref[1 + i][len(b"val_"):])
        with open(os.path.join(R, "submission_rank.csv"), "rb") as f1, open(os.path.join(R, "pred_B.csv"), "rb") as f2:
            self.assertEqual(f1.read(), f2.read())

    def test_opp_seam_byte_identical_and_rows(self):
        opp = rj(os.path.join(B.R4, "opp", "opp_report.json"))
        self.assertEqual(sha(os.path.join(B.R4, "opp", "o4_seam_val.npz")), t04.VAL_O4_SHA)
        self.assertTrue(opp["seam_val"]["file_sha256_equal"] and opp["seam_val"]["dkappa_equal"])
        self.assertEqual(opp["seam_val"]["dkappa_O4"], DK_SEAM["O4"])
        self.assertEqual((opp["exit_code"], opp["passed"], opp["usable_so_far"]), (0, True, True))
        self.assertEqual({s: opp["sets"][s]["failures"] for s in t04.SETS}, {s: 0 for s in t04.SETS})
        self.assertEqual(B.fake.calls[0]["n"], 19156 + 4787 + 4815)   # one concatenated call
        self.assertEqual(B.fake.calls[0]["workers"], 3)
        self.assertFalse(t2.is_inside(B.fake.calls[0]["tmp_root"], REPO))

    def test_report_fields(self):
        rep = rj(os.path.join(B.R4, "rank", "t04_apply_report.json"))
        bold = ("format", "mode", "combo", "notebook", "seam_local", "replay", "colour_check", "usable",
                "unusable_reasons", "check_submission", "outputs_sha256", "test02", "inputs_bound", "attempt")
        self.assertEqual([k for k in bold if k not in rep], [])
        self.assertEqual((rep["format"], rep["mode"], rep["usable"], rep["exit_code"]),
                         ("c10-test04 rank apply v1", "dry-run", True, 0))
        self.assertEqual(rep["combo"]["id"], "Q02")
        self.assertEqual(rep["combo"]["sources"], ["C5", "OP", "O4", "SWA", "SWB"])
        self.assertEqual(rep["check_submission"]["mode"], "official")
        self.assertEqual(rep["check_submission"]["errors"], [])
        for k in ("seam_check_passed", "test_bundle_equal_to_test02", "local_recompute_passed", "env_same"):
            self.assertIs(rep["notebook"][k], True, k)
        self.assertEqual(rep["notebook"]["keys_equal_to_test02"], {s: True for s in t04.SETS})
        self.assertEqual(rep["notebook"]["earlier_failed_runs"], [])
        want = {"public": {"O4": 0.0930, "SWA": 0.0321, "SWB": 0.0278},
                "private": {"O4": 0.1000, "SWA": 0.0418, "SWB": 0.0364}}
        for s, d in want.items():
            for src, v in d.items():
                got = rep["colour_check"][s][src]["dkappa"]
                self.assertEqual(round(got, 4), v, (s, src))
                self.assertGreater(got, 0)
        self.assertEqual(rep["colour_check"]["public"]["O4"]["dkappa"],
                         rj(os.path.join(B.R4, "opp", "opp_report.json"))["o4_dkappa_early"]["public"]["dkappa"])
        self.assertEqual({s: rep["colour_check"]["seam_val"][s]["dkappa"] for s in DK_SEAM}, DK_SEAM)
        sub = os.path.join(B.R4, "rank", "submission_rank.csv")
        self.assertEqual(rep["outputs_sha256"]["submission_rank.csv"], sha(sub))
        self.assertEqual(rep["replay"]["seam_val"]["equal_to_committed_val_o4"], True)

    def test_inputs_bound(self):
        rep = rj(os.path.join(B.R4, "rank", "t04_apply_report.json"))
        need = ["results/test04/opp/opp_report.json", "results/test04/unpack_sw.json", "results/test04/sw/config.json",
                "results/test04/sw/seam_check.json", "results/test04/rank/seam_val/seam_local.json",
                "results/test02/rank/apply_test_report.json", "results/test02/rank/opening_test.npz",
                "results/test02/checks/rank/t1.json", "results/prereg04/rank/frozen_config.json"]
        need += [f"results/test02/rank/{s}/dumps/{f}" for s in t04.SETS for f in ("val_keys.npz", "val_p02e9_d4.npz")]
        self.assertEqual([k for k in need if k not in rep["inputs_bound"]], [])
        for k, v in rep["inputs_bound"].items():
            self.assertEqual(t2.pin_sha256(rep["inputs_bound_paths"][k]), v, k)

    def test_rule_b_differences_vs_r20(self):
        rep = rj(os.path.join(B.R4, "rank", "t04_apply_report.json"))
        p2 = list(csv.reader(open(os.path.join(P02, "pred_B.csv"), newline="", encoding="utf-8")))
        p4 = list(csv.reader(open(os.path.join(P04, "pred_B.csv"), newline="", encoding="utf-8")))
        hand = sum(1 for i in range(800) if p2[1 + i][1] != p4[1 + i][1])
        self.assertEqual(rep["rule_b_differences_vs_r20"], hand)
        self.assertEqual(hand, 179)

    def test_t02_inputs_untouched(self):
        for d in (B.D2, B.O2, B.T):
            for root, _dirs, files in os.walk(d):
                for f in files:
                    self.assertFalse(f.endswith(".tmp"), os.path.join(root, f))
        self.assertEqual(sha(os.path.join(B.O2, "submission_rank.csv")),
                         rj(os.path.join(B.O2, "apply_test_report.json"))["outputs_sha256"]["submission_rank.csv"])

    def test_reruns_refused_without_rerun(self):
        code, out = run(["apply", "--dry-run"] + B.dirs(B.R4))
        self.assertEqual(code, 2)
        self.assertIn("--rerun", out)
        code, out = run(["opp", "--dry-run"] + B.dirs(B.R4), o4_scores=FakeO4(), **FAST_OPP)
        self.assertEqual(code, 2)
        self.assertIn("--rerun", out)
        code, out = run(["apply", "--dry-run", "--rerun"] + B.dirs(B.R4))
        self.assertEqual(code, 2)
        self.assertIn("--reason", out)
        code, out = run(["unpack", "--dry-run", "--zip", B.zip] + B.dirs(B.R4))
        self.assertEqual(code, 2)
        self.assertIn("already exist", out)

    def test_notebook_env_is_the_shape_official_log_reads(self):
        """Review round 2: the report's notebook.env is flat ({gpu_name, torch, cudnn}, the one environment of the
        15 markers); official_log's rerun-on-G4 check reads that shape (an A100 marker environment is refused)."""
        import official_log as ol
        rep = rj(os.path.join(B.R4, "rank", "t04_apply_report.json"))
        self.assertEqual(sorted(rep["notebook"]["env"]), ["cudnn", "gpu_name", "torch"])
        self.assertEqual((rep["notebook"]["env"]["gpu_name"], ol.T04_RERUN_GPU), (t04.G4_GPU, t04.G4_GPU))

        def g4_problems(r):
            return [p for p in ol.t04_report_problems(r, "r", None, None)[1] if "not only on G4" in p]
        rerun = copy.deepcopy(dict(rep, mode="real", attempt=dict(rep["attempt"], number=2)))
        self.assertEqual(g4_problems(rerun), [])
        for label, change in (("A100 markers", lambda r: r["notebook"]["env"].update(gpu_name="NVIDIA A100-SXM4-80GB")),
                              ("A100 config", lambda r: r["notebook"].update(gpu="NVIDIA A100-SXM4-80GB")),
                              ("no marker environment", lambda r: r["notebook"].pop("env"))):
            with self.subTest(label):
                r = copy.deepcopy(rerun)
                change(r)
                self.assertEqual(len(g4_problems(r)), 1)
                self.assertEqual(g4_problems(dict(r, attempt=dict(r["attempt"], number=1))), [])

    def test_status_reads_only(self):
        before = {os.path.join(r, f): sha(os.path.join(r, f)) for r, _d, fs in os.walk(B.R4) for f in fs}
        code, out = run(["status", "--dry-run"] + B.dirs(B.R4))
        self.assertEqual(code, 0)
        self.assertIn("apply: usable True", out)
        after = {os.path.join(r, f): sha(os.path.join(r, f)) for r, _d, fs in os.walk(B.R4) for f in fs}
        self.assertEqual(before, after)


class DirsAndModes(unittest.TestCase):
    def test_real_mode_paths_fixed(self):
        code, out = run(["opp", "--results-dir", os.path.join(B.base, "elsewhere")], now=lambda: dt.datetime(
            2026, 11, 6, tzinfo=TW))
        self.assertEqual(code, 2)
        self.assertIn("real mode", out)

    def test_results_dir_refusals(self):
        for rel in ("results/test04", "results/test02", "results/prereg04/x", "tools"):
            with self.subTest(rel=rel):
                code, out = run(["opp", "--dry-run"] + B.dirs(os.path.join(REPO, *rel.split("/"))))
                self.assertEqual(code, 2, out)
        code, out = run(["opp", "--rehearsal"] + B.dirs(os.path.join(REPO, "results", "test02")))
        self.assertEqual(code, 2)
        bad = os.path.join(B.base, "x_" + "ans" + "wers")
        code, out = run(["opp", "--dry-run"] + B.dirs(bad))
        self.assertEqual(code, 2)
        self.assertIn("label", out)
        code, out = run(["apply", "--dry-run", "--results-dir", os.path.join(B.base, "q")])
        self.assertEqual(code, 2)
        self.assertIn("needs", out)

    def test_allow_failed_only_in_dry_run_outside_the_repo(self):
        """Review round 2: --allow-failed in real mode, or with --rehearsal in the repository's rehearsal folder, is
        refused (2) before anything is read or written."""
        real_r4 = os.path.join(B.base, "af_real", "results", "test04")
        code, out = run(["unpack", "--allow-failed", "--zip", B.zip], REAL_R4=real_r4,
                        FAILED_ZIP_ROOT=os.path.join(B.base, "af_real", "failed_zips"),
                        now=lambda: dt.datetime(2026, 11, 6, tzinfo=TW))
        self.assertEqual(code, 2, out)
        self.assertIn("--allow-failed is for inspection only", out)
        self.assertFalse(os.path.exists(real_r4))
        reh = os.path.join(B.base, "af_rehearsal")       # stands for results/test04_rehearsal (inside the repository)
        code, out = run(["unpack", "--rehearsal", "--allow-failed", "--zip", B.zip] + B.dirs(reh), REHEARSAL_R4=reh)
        self.assertEqual(code, 2, out)
        self.assertIn("--allow-failed is for inspection only", out)
        self.assertFalse(os.path.exists(os.path.join(reh, "sw")))

    def test_k6(self):
        late = lambda: dt.datetime(2026, 11, 17, 0, 0, 0, tzinfo=TW)    # noqa: E731
        for cmd in (["opp"], ["apply"], ["unpack", "--zip", B.zip]):
            with self.subTest(cmd=cmd[0]):
                code, out = run(cmd, now=late)
                self.assertEqual(code, 3)
                self.assertIn("K6 已過", out)
        self.assertFalse(os.path.exists(os.path.join(REPO, "results", "test04", "opp")))


# ================================================================ negative drills (copies of the tree)
class ApplyDrills(unittest.TestCase):
    def apply(self, R4, **kw):
        extra = kw.pop("extra", [])
        return run(["apply", "--dry-run"] + B.dirs(R4, **kw) + extra)

    def assert_verdict(self, R4, code, stage, reason):
        v = rj(os.path.join(R4, "rank", "attempt_verdict.json"))
        self.assertEqual((v["exit_code"], v["stage"], v["reason"]), (code, stage, reason))
        self.assertFalse(os.path.exists(os.path.join(R4, "rank", "t04_apply_report.json")))
        self.assertFalse(os.path.exists(os.path.join(R4, "rank", "submission_rank.csv")))
        self.assertFalse(os.path.exists(os.path.join(R4, "rank", "rank_test_probs.csv")))

    def test_dump_stage_1_keys(self):
        R4 = B.copy_r4("d_keys")
        d = os.path.join(R4, "sw", "public", "dumps")
        with np.load(os.path.join(d, "val_keys.npz")) as z:
            p = z["parsed"].copy()
        p[5] = ~p[5]
        rewrite_npz(os.path.join(d, "val_keys.npz"), parsed=p)
        update_marker(d, "val_keys")
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "prerequisites", "dump_keys")

    def test_dump_stage_2_environment(self):
        R4 = B.copy_r4("d_env")
        edit_json(os.path.join(R4, "sw", "public", "dumps", "_DONE_val_p03a_d4.json"),
                  lambda m: m.update(gpu_name="NVIDIA A100-SXM4-40GB"))
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "prerequisites", "dump_environment")

    def test_dump_stage_3_markers(self):
        R4 = B.copy_r4("d_mark")
        d = os.path.join(R4, "sw", "public", "dumps")
        with np.load(os.path.join(d, "val_p03a_d4.npz")) as z:
            lg = z["logits"].copy()
        lg[0, 0, 0] += 1
        rewrite_npz(os.path.join(d, "val_p03a_d4.npz"), logits=lg)       # marker NOT updated
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "prerequisites", "dump_markers")

    def test_dump_stage_4_contents(self):
        cases = {
            "weights": lambda d: (rewrite_npz(os.path.join(d, "val_p03b_s0_swap.npz"), weights_sha256=np.array("0" * 64)),
                                  update_marker(d, "val_p03b_s0_swap")),
            "view": lambda d: (rewrite_npz(os.path.join(d, "val_p03a_s0_swap.npz"), view=np.array("s0")),
                               update_marker(d, "val_p03a_s0_swap")),
            "nan rows": lambda d: (self.nan_row(os.path.join(d, "val_p03a_s0_swap.npz")),
                                   update_marker(d, "val_p03a_s0_swap"))}
        for label, fn in cases.items():
            with self.subTest(case=label):
                R4 = B.copy_r4("d_content")
                fn(os.path.join(R4, "sw", "private", "dumps"))
                code, out = self.apply(R4)
                self.assertEqual(code, 1, out)
                self.assert_verdict(R4, 1, "prerequisites", "dump_content")

    @staticmethod
    def nan_row(path):
        with np.load(path) as z:
            lg = z["logits"].copy()
        lg[2] = np.nan
        rewrite_npz(path, logits=lg)

    def test_run_log_earlier_failure_and_resumed(self):
        R4 = B.copy_r4("d_runlog")
        edit_json(os.path.join(R4, "sw", "run_log.json"),
                  lambda o: o.update(runs=[{"run": 1, "status": "seam_failed"}] + [dict(o["runs"][0], run=2)]))
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "prerequisites", "run_log_earlier")
        R4 = B.copy_r4("d_resumed")
        edit_json(os.path.join(R4, "sw", "private", "dumps", "_DONE_val_p03b_d4.json"), lambda m: m.update(run=0))
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "prerequisites", "resumed")

    def test_flags_bundle_weights(self):
        R4 = B.copy_r4("d_flags")
        edit_json(os.path.join(R4, "sw", "config.json"), lambda c: c.update(rehearsal=False))
        code, out = self.apply(R4)
        self.assertEqual(code, 2, out)
        self.assertIn("rehearsal", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "rank", "attempt_verdict.json")))
        R4 = B.copy_r4("d_bundle")
        edit_json(os.path.join(R4, "sw", "config.json"),
                  lambda c: c["test_bundle"].update(manifest_sha256="0" * 64))
        code, out = self.apply(R4)
        self.assertEqual(code, 2, out)
        self.assertIn("test bundle", out)
        R4 = B.copy_r4("d_weights")
        edit_json(os.path.join(R4, "sw", "config.json"), lambda c: c["weights"]["a"].update(sha256="1" * 64))
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "prerequisites", "weights")

    def test_notebook_seam_recompute_disagrees(self):
        R4 = B.copy_r4("d_c0")
        d = os.path.join(R4, "sw", "seam_val", "dumps")
        with np.load(os.path.join(d, "val_p03b_d4.npz")) as z:
            lg = z["logits"].copy()
        lg[:5, 0, :] += np.linspace(0, 0.5, 10, dtype=np.float32)
        rewrite_npz(os.path.join(d, "val_p03b_d4.npz"), logits=lg)
        update_marker(d, "val_p03b_d4")
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "notebook_seam_recompute", "recompute_mismatch")
        v = rj(os.path.join(R4, "rank", "attempt_verdict.json"))
        self.assertFalse(v["details"]["arms"]["b"]["passed"])
        self.assertTrue(v["details"]["arms"]["a"]["passed"])

    # ---- review round 2
    def test_notebook_seam_recompute_swap_disagrees(self):
        """The local recompute checks the swapped C0 too: seam_val val_p03b_s0_swap.npz changed (marker updated) ->
        1 at notebook_seam_recompute (before the local seam, which would also fail)."""
        R4 = B.copy_r4("d_c0swap")
        d = os.path.join(R4, "sw", "seam_val", "dumps")
        with np.load(os.path.join(d, "val_p03b_s0_swap.npz")) as z:
            lg = z["logits"].copy()
        lg[:5, :] += np.linspace(0, 0.5, 10, dtype=np.float32)
        rewrite_npz(os.path.join(d, "val_p03b_s0_swap.npz"), logits=lg)
        update_marker(d, "val_p03b_s0_swap")
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "notebook_seam_recompute", "recompute_mismatch")
        arms = rj(os.path.join(R4, "rank", "attempt_verdict.json"))["details"]["arms"]
        self.assertGreater(arms["b"]["c0_swap_max_abs_diff"], arms["b"]["tol"])
        self.assertEqual(arms["b"]["c0_s0_max_abs_diff"], 0.0)
        self.assertEqual((arms["a"]["passed"], arms["b"]["passed"]), (True, False))

    def test_local_seam_tol_is_the_larger_of_test02_and_test04(self):
        """Rev 3 item 5: tol = max(test02's seam tol, test04's). test02 at 1e-3 (A100) and test04 at 1e-4: tol 1e-3;
        a C5 change of max |dP| in (1e-4, 1e-3] passes with it and fails (1) with test02 at 1e-4."""
        O2 = os.path.join(B.base, "o2_tol1e3")
        shutil.rmtree(O2, ignore_errors=True)
        shutil.copytree(B.O2, O2)
        edit_json(os.path.join(O2, "apply_test_report.json"), lambda r: r["seam_local"].update(tol=1e-3))
        R4 = B.copy_r4("d_tolmax")
        code, out = self.apply(R4, O2=O2)
        self.assertEqual(code, 0, out)
        sl = rj(os.path.join(R4, "rank", "seam_val", "seam_local.json"))
        self.assertEqual((sl["tol_check"]["tol"], sl["tol"]), (1e-3, 1e-3))
        self.assertEqual((sl["tol_check"]["c5"]["test02_seam_tol"], sl["tol_check"]["sw"]["recomputed_tol"]), (1e-3, 1e-4))
        D2 = os.path.join(B.base, "d2_tolmax", "test02")
        shutil.rmtree(os.path.dirname(D2), ignore_errors=True)
        shutil.copytree(B.D2, D2, ignore=shutil.ignore_patterns("player"))
        p = os.path.join(D2, "rank", "seam_val", "dumps", "val_p02e9_d4.npz")
        rows0 = rf.load_val_keys(os.path.join(D2, "rank", "seam_val", "dumps"))["groups"][0][1]
        with np.load(p) as z:
            lg = z["logits"].copy()
        lg[rows0, :, 0] += np.float32(0.002)                     # question 1 only: max |dP| about 5.6e-4
        rewrite_npz(p, logits=lg)
        R4 = B.copy_r4("d_tolmax_mid")
        code, out = self.apply(R4, D2=D2, O2=O2)
        self.assertEqual(code, 0, out)
        sl = rj(os.path.join(R4, "rank", "seam_val", "seam_local.json"))
        self.assertTrue(1e-4 < sl["max_abs_diff"] <= 1e-3, sl["max_abs_diff"])
        self.assertTrue(sl["passed"])
        R4 = B.copy_r4("d_tolmax_mid_1e4")
        code, out = self.apply(R4, D2=D2)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "local_seam", "local_seam_failed")

    def test_o4_file_changed_after_opp(self):
        """Apply prerequisite 1: o4_<set>.npz must still be the file opp_report.json records."""
        R4 = B.copy_r4("d_o4changed")
        p = os.path.join(R4, "opp", "o4_public.npz")
        with np.load(p) as z:
            o4 = z["o4"].copy()
        o4[0, 0] += 1.0
        rewrite_npz(p, o4=o4)
        code, out = self.apply(R4)
        self.assertEqual(code, 2, out)
        self.assertIn("o4_public.npz differs from the report", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "rank")))

    def test_counts_file_changed_after_opp(self):
        """rev 3 check (items 9, 17): a counts file that is not the registered one is a refusal (2) that says to restore
        the file, never 'rerun opp' (after opp's registered decision a new opp needs a revision)."""
        R4 = B.copy_r4("d_counts")
        for now_sha in ("0" * 64, None):
            with self.subTest(counts=now_sha):
                code, out = run(["apply", "--dry-run"] + B.dirs(R4), counts_sha_now=lambda v=now_sha: v)
                self.assertEqual(code, 2, out)
                self.assertIn("counts_fit.npz is " + ("missing" if now_sha is None else "sha256 000000000000"), out)
                self.assertIn("design_scratch", out)
                self.assertNotIn("rerun opp", out)
                self.assertFalse(os.path.exists(os.path.join(R4, "rank")))

    def test_check_submission_failure_leaves_no_test_output(self):
        """rev 3 check (item 8): check_submission fails on the written file -> 4; every test-set output is removed
        (pred_B.csv is a byte copy of the rejected file), seam_val/ stays, attempt_verdict.json says check_submission;
        apply --rerun --reason is then allowed."""
        import types
        import check_submission as cs
        R4 = B.copy_r4("d_checksub")
        fake = types.SimpleNamespace(check=lambda *a, **k: ["row 3: bad"], __file__=cs.__file__)
        code, out = run(["apply", "--dry-run"] + B.dirs(R4), check_submission=fake)
        self.assertEqual(code, 4, out)
        self.assert_verdict(R4, 4, "check_submission", "check_submission")
        for n in t04.TEST_OUTPUTS:
            self.assertFalse(os.path.exists(os.path.join(R4, "rank", n)), n)
        v = rj(os.path.join(R4, "rank", "attempt_verdict.json"))
        self.assertEqual(sorted(v["details"]["removed"]), sorted(t04.TEST_OUTPUTS))
        self.assertTrue(os.path.exists(os.path.join(R4, "rank", "seam_val", "seam_local.json")))
        code, out = run(["apply", "--dry-run", "--rerun", "--reason", "check_submission fixed"] + B.dirs(R4))
        self.assertEqual(code, 0, out)
        self.assertTrue(os.path.exists(os.path.join(R4, "rank", "submission_rank.csv")))

    def test_unpack_record_must_match_failed(self):
        """Apply prerequisite 2: unpack_sw.json still matches results/test04/failed (the archives it saw, its attempt
        number, the unpacked zip not archived since)."""
        def extra_archive(R4):
            os.makedirs(os.path.join(R4, "failed", "attempt_1"))

        def attempt_number(R4):
            edit_json(os.path.join(R4, "unpack_sw.json"), lambda r: r.update(attempt_number=2))

        def zip_archived(R4):
            zsha = rj(os.path.join(R4, "unpack_sw.json"))["zip"]["sha256"]
            wj(os.path.join(R4, "failed", "attempt_1", "attempt.json"),
               {"counts_toward_rerun": False, "failure_class": "before_r20_u1", "zips": {"x.zip": zsha},
                "archived_at": "2026-09-01T00:00:00+08:00"})
            edit_json(os.path.join(R4, "unpack_sw.json"), lambda r: r.update(failed_attempts_seen=["attempt_1"]))
        for label, fn, text in (("an archive appeared", extra_archive, "archives now"),
                                ("attempt number", attempt_number, "attempt number 2 vs 1"),
                                ("the unpacked zip was archived", zip_archived, "the unpacked zip has been archived")):
            with self.subTest(case=label):
                R4 = B.copy_r4("d_unpackrec")
                fn(R4)
                code, out = self.apply(R4)
                self.assertEqual(code, 2, out)
                self.assertIn("does not match results/test04/failed", out)
                self.assertIn(text, out)
                self.assertFalse(os.path.exists(os.path.join(R4, "rank")))

    def test_tol_mismatch(self):
        R4 = B.copy_r4("d_tol")
        edit_json(os.path.join(R4, "sw", "seam_check.json"), lambda s: s.update(tol=1e-3))
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "notebook_seam_recompute", "recompute_mismatch")

    def test_first_run_on_a100(self):
        """Review round 3 (spec 5.3, rev 3 items 4, 5): a first run on A100 (all 15 markers A100, prereg03's dumps on
        G4) has test04's tol 1e-3 (same_gpu false) and passes; the local tol is max(test02's 1e-4, 1e-3) = 1e-3.
        The same markers with the notebook still recording 1e-4 -> 1 (recompute_mismatch)."""
        a100 = "NVIDIA A100-SXM4-40GB"

        def on_a100(R4, nb_tol):
            for s in t04.SETS:
                for st in t04.marker_stems():
                    edit_json(os.path.join(R4, "sw", s, "dumps", f"_DONE_{st}.json"), lambda m: m.update(gpu_name=a100))

            def tol(sc):
                sc["tol"] = nb_tol
                for x in t04.ARMS:
                    sc["arms"][x]["tol"] = nb_tol
            edit_json(os.path.join(R4, "sw", "seam_check.json"), tol)
        R4 = B.copy_r4("d_a100")
        on_a100(R4, 1e-3)
        code, out = self.apply(R4)
        self.assertEqual(code, 0, out)
        sl = rj(os.path.join(R4, "rank", "seam_val", "seam_local.json"))
        sw = sl["tol_check"]["sw"]
        self.assertEqual((sw["recomputed_tol"], sw["notebook_tol"], sw["same_gpu"], sw["seam_dump_gpu"]),
                         (1e-3, 1e-3, False, a100))
        self.assertEqual((sl["tol_check"]["c5"]["test02_seam_tol"], sl["tol_check"]["tol"], sl["tol"]),
                         (1e-4, 1e-3, 1e-3))
        self.assertTrue(sl["passed"])
        R4 = B.copy_r4("d_a100_nb1e4")
        on_a100(R4, 1e-4)
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "notebook_seam_recompute", "recompute_mismatch")
        self.assertIn("tol: notebook 0.0001, from the markers 0.001", out)

    def test_notebook_seam_failed(self):
        R4 = B.copy_r4("d_nbseam")
        edit_json(os.path.join(R4, "sw", "seam_check.json"), lambda s: s.update(passed=False))
        code, out = self.apply(R4)
        self.assertEqual(code, 1, out)
        self.assert_verdict(R4, 1, "prerequisites", "notebook_seam")

    def test_local_seam_fails_writes_no_test_output(self):
        R4 = B.copy_r4("d_local")
        D2 = os.path.join(B.base, "d2_local", "test02")
        shutil.copytree(B.D2, D2, ignore=shutil.ignore_patterns("player"))
        p = os.path.join(D2, "rank", "seam_val", "dumps", "val_p02e9_d4.npz")
        with np.load(p) as z:
            lg = z["logits"].copy()
        lg[:400] *= 3
        rewrite_npz(p, logits=lg)
        code, out = self.apply(R4, D2=D2)
        self.assertEqual(code, 1, out)
        self.assertIn("本機接縫沒過", out)
        self.assert_verdict(R4, 1, "local_seam", "local_seam_failed")
        self.assertFalse(rj(os.path.join(R4, "rank", "seam_val", "seam_local.json"))["passed"])

    # ---- K5 gate round 2
    def test_local_seam_fails_on_the_registered_rerun(self):
        """K5 gate round 2: the local Q02 seam fails (1) on the registered rerun (one counted archive, the unpack record
        of attempt 2): the output and attempt_verdict.json say there is no third run (MSG_SEAM_LAST; this seam_msg call
        has its own first-attempt text), never 'rerun once' again; a following apply --rerun --reason is refused (2)
        with the same pointer, never 'archive, then rerun the notebook' (archive refuses), the verdict kept. The first
        attempt keeps both first-attempt texts."""
        first_local = "走 docs/TESTDAY_04.md「重跑一次」"
        first_rerun = "接縫類失敗要 archive 後重跑 notebook"
        D2 = os.path.join(B.base, "d2_local_rerun", "test02")
        shutil.rmtree(os.path.dirname(D2), ignore_errors=True)
        shutil.copytree(B.D2, D2, ignore=shutil.ignore_patterns("player"))
        p = os.path.join(D2, "rank", "seam_val", "dumps", "val_p02e9_d4.npz")
        with np.load(p) as z:
            lg = z["logits"].copy()
        lg[:400] *= 3
        rewrite_npz(p, logits=lg)
        counted = {"counts_toward_rerun": True, "failure_class": "notebook_seam",
                   "archived_at": "2026-09-01T00:00:00+08:00", "zips": {}, "done_times": {}}
        for label, registered in (("first attempt", False), ("registered rerun", True)):
            with self.subTest(case=label):
                R4 = B.copy_r4("d_local_rerun" if registered else "d_local_first")
                if registered:
                    wj(os.path.join(R4, "failed", "attempt_1", "attempt.json"), counted)
                    edit_json(os.path.join(R4, "unpack_sw.json"),
                              lambda r: r.update(attempt_number=2, failed_attempts_seen=["attempt_1"]))
                code, out = self.apply(R4, D2=D2)
                self.assertEqual(code, 1, out)
                self.assertIn("本機接縫沒過", out)
                self.assert_verdict(R4, 1, "local_seam", "local_seam_failed")
                vpath = os.path.join(R4, "rank", "attempt_verdict.json")
                v = rj(vpath)
                for text in (out, v["message"]):
                    if registered:
                        self.assertIn(t04.MSG_SEAM_LAST, text)
                        self.assertNotIn(first_local, text)
                    else:
                        self.assertIn(first_local, text)
                        self.assertNotIn(t04.MSG_SEAM_LAST, text)
                code, out = self.apply(R4, D2=D2, extra=["--rerun", "--reason", "rerun apply after its seam failure"])
                self.assertEqual(code, 2, out)
                if registered:
                    self.assertIn(t04.MSG_SEAM_LAST, out)
                    self.assertIn("attempt_1", out)
                    self.assertNotIn(first_rerun, out)
                else:
                    self.assertIn(first_rerun, out)
                    self.assertNotIn(t04.MSG_SEAM_LAST, out)
                self.assertEqual(rj(vpath), v)                      # refused before anything was cleared

    def test_every_exit_1_message_goes_through_seam_msg(self):
        """K5 gate round 2: every seam_fail call's message goes through seam_msg(ctx, ...) (unpack's fail1; the 12 in
        check_notebook / check_dumps / notebook_seam_recompute; apply's local seam), and MSG_SEAM is read only as
        seam_msg's default, so no call falls back to 'archive first' on the registered rerun unnoticed. The behaviour
        is tested for unpack, the notebook seam (UnpackDrills) and the local seam (above)."""
        import ast
        with open(t04.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "seam_fail"]

        def via_seam_msg(call):
            return bool(call.args) and any(
                isinstance(m, ast.Call) and isinstance(m.func, ast.Name) and m.func.id == "seam_msg" and m.args
                and isinstance(m.args[0], ast.Name) and m.args[0].id == "ctx" for m in ast.walk(call.args[0]))
        self.assertGreaterEqual(len(calls), 14)
        self.assertEqual([c.lineno for c in calls if not via_seam_msg(c)], [])
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "seam_msg")
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "MSG_SEAM"
                 and isinstance(n.ctx, ast.Load)]
        self.assertEqual(loads, [d for d in fn.args.defaults if isinstance(d, ast.Name) and d.id == "MSG_SEAM"])

    def test_swa_equal_to_a0_is_no_04(self):
        """8.3: public val_p03a_s0_swap.npz rewritten as d4[:, 0, :] (view kept, marker updated): SWA's public
        dkappa is exactly 0.0 -> 3, only the report; then archive and apply --rerun are refused."""
        R4 = B.copy_r4("d_swa0")
        d = os.path.join(R4, "sw", "public", "dumps")
        with np.load(os.path.join(d, "val_p03a_d4.npz")) as z:
            a0 = np.ascontiguousarray(z["logits"][:, 0, :])
        rewrite_npz(os.path.join(d, "val_p03a_s0_swap.npz"), logits=a0)
        update_marker(d, "val_p03a_s0_swap")
        code, out = self.apply(R4)
        self.assertEqual(code, 3, out)
        rep = rj(os.path.join(R4, "rank", "t04_apply_report.json"))
        self.assertEqual(rep["colour_check"]["public"]["SWA"]["dkappa"], 0.0)
        self.assertEqual((rep["usable"], rep["exit_code"], rep["check_submission"]), (False, 3, None))
        self.assertEqual(rep["unusable_reasons"], ["dkappa:public.SWA: 0.0 (needs > 0)"])
        for n in ("submission_rank.csv", "rank_test_probs.csv", "pred_B.csv", "attempt_verdict.json"):
            self.assertFalse(os.path.exists(os.path.join(R4, "rank", n)), n)
        shutil.copyfile(B.zip, os.path.join(R4, f4.ZIP_NAME))
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "drill"])
        self.assertEqual(code, 2, out)
        self.assertIn("registered decision", out)
        code, out = self.apply(R4, extra=["--rerun", "--reason", "drill"])
        self.assertEqual(code, 2, out)
        self.assertIn("bug-revision", out)

    def test_t1_and_test02(self):
        C = os.path.join(B.base, "checks_fo")
        shutil.copytree(B.C, C)
        edit_json(os.path.join(C, "rank", "t1.json"), lambda t: t.update(verdict="FAIL_OPENING"))
        R4 = B.copy_r4("d_t1")
        code, out = self.apply(R4, C=C)
        self.assertEqual(code, 3, out)
        self.assertIn("04 不上場", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "rank")))
        code, out = run(["opp", "--dry-run", "--rerun", "--reason", "x"] + B.dirs(R4, C=C))
        self.assertEqual(code, 2, out)     # opp_report exists (registered): the rerun gate comes first
        edit_json(os.path.join(C, "rank", "t1.json"), lambda t: t.update(verdict="FAIL_STRUCT"))
        code, out = self.apply(R4, C=C)
        self.assertEqual(code, 2, out)
        self.assertIn("FAIL_STRUCT", out)
        O2 = os.path.join(B.base, "o2_fallback")
        shutil.copytree(B.O2, O2)
        edit_json(os.path.join(O2, "apply_test_report.json"), lambda r: r.update(fallback_used=True))
        code, out = self.apply(R4, O2=O2)
        self.assertEqual(code, 2, out)
        self.assertIn("fallback_used", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "rank")))

    def test_opp_says_no_04(self):
        R4 = B.copy_r4("d_opp3")
        edit_json(os.path.join(R4, "opp", "opp_report.json"),
                  lambda r: r.update(exit_code=3, usable_so_far=False, unusable_reasons=["dkappa:public.O4: -0.1"]))
        code, out = self.apply(R4)
        self.assertEqual(code, 3, out)
        self.assertIn("opp 已判定 04 不用", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "rank")))

    def test_venv_drift(self):
        bad = dict(t04.venv_versions(), numpy="0.0.0")
        R4 = B.copy_r4("d_venv")
        code, out = run(["apply", "--dry-run"] + B.dirs(R4), venv_versions=lambda: bad)
        self.assertEqual(code, 4, out)
        self.assert_verdict(R4, 4, "prerequisites", "venv_versions")
        # a previous exit 4 may be rerun with --rerun --reason
        code, out = run(["apply", "--dry-run", "--rerun", "--reason", "venv restored"] + B.dirs(R4))
        self.assertEqual(code, 0, out)
        rep = rj(os.path.join(R4, "rank", "t04_apply_report.json"))
        self.assertIn(os.path.join("rank", "attempt_verdict.json"),
                      " ".join(rep["rerun"]["replaced_sha256"]).replace("/", os.sep))
        self.assertFalse(os.path.exists(os.path.join(R4, "rank", "attempt_verdict.json")))
        # zlib pin too, for opp (no report written)
        R4 = B.copy_r4("d_venv_opp", parts=("sw", "unpack_sw.json"))
        code, out = run(["opp", "--dry-run"] + B.dirs(R4), o4_scores=FakeO4(), ZLIB_PIN="0.0", **FAST_OPP)
        self.assertEqual(code, 4, out)
        self.assertFalse(os.path.exists(os.path.join(R4, "opp", "opp_report.json")))

    def test_apply_rerun_rules(self):
        R4 = B.copy_r4("d_rerun1")
        edit_json(os.path.join(R4, "sw", "seam_check.json"), lambda s: s.update(passed=False))
        self.assertEqual(self.apply(R4)[0], 1)
        edit_json(os.path.join(R4, "sw", "seam_check.json"), lambda s: s.update(passed=True))
        code, out = self.apply(R4, extra=["--rerun", "--reason", "x"])
        self.assertEqual(code, 2, out)
        self.assertIn("archive", out)
        # the finished tree: a report exists -> refused without a revision, allowed with a committed later one
        R4 = B.copy_r4("d_rerun_rep", parts=("sw", "opp", "unpack_sw.json", "rank"))
        code, out = self.apply(R4, extra=["--rerun", "--reason", "x"])
        self.assertEqual(code, 2, out)
        doc = "| rev 4 | x | bug |\n"
        with patched(t04, git_show_head=lambda rel: doc, require_committed=lambda p, w: {},
                     git_first_commit_with=lambda rel, n: ("abc", "2099-01-01T00:00:00+08:00")):
            code, out = self.apply(R4, extra=["--rerun", "--reason", "x", "--bug-revision", "4"])
        self.assertEqual(code, 0, out)
        rep = rj(os.path.join(R4, "rank", "t04_apply_report.json"))
        self.assertEqual(rep["rerun"]["bug_revision"]["n"], 4)
        self.assertEqual(rep["attempt"]["bug_revisions"][0]["n"], 4)


class OppDrills(unittest.TestCase):
    def opp(self, R4, fake, extra=()):
        return run(["opp", "--dry-run"] + B.dirs(R4) + list(extra), o4_scores=fake, **FAST_OPP)

    def test_public_failures_are_no_04_and_rerun_needs_a_revision(self):
        R4 = B.copy_r4("o_fail", parts=("sw", "unpack_sw.json"))
        # public = rows 19156..; 24 failures of 4787 slots: 200 * 24 > 4787 -> no 04
        code, out = self.opp(R4, FakeO4(fail=range(19156, 19156 + 24)))
        self.assertEqual(code, 3, out)
        rep = rj(os.path.join(R4, "opp", "opp_report.json"))
        self.assertEqual((rep["passed"], rep["usable_so_far"], rep["sets"]["public"]["failures"]), (False, False, 24))
        self.assertEqual(rep["sets"]["public"]["anomalies"], 24)
        code, out = self.opp(R4, FakeO4(), extra=["--rerun", "--reason", "again"])
        self.assertEqual(code, 2, out)
        self.assertIn("revision", out)
        doc = "| rev 4 | x | bug |\n"
        with patched(t04, git_show_head=lambda rel: doc, require_committed=lambda p, w: {},
                     git_first_commit_with=lambda rel, n: ("abc", "2099-01-01T00:00:00+08:00")):
            code, out = self.opp(R4, FakeO4(), extra=["--rerun", "--reason", "bug fixed", "--bug-revision", "4"])
        self.assertEqual(code, 0, out)
        # 23 of 4787 passes (200 * 23 <= 4787)
        R4 = B.copy_r4("o_fail23", parts=("sw", "unpack_sw.json"))
        self.assertEqual(self.opp(R4, FakeO4(fail=range(19156, 19156 + 23)))[0], 0)

    def test_seam_failures(self):
        R4 = B.copy_r4("o_seam1", parts=("sw", "unpack_sw.json"))
        code, out = self.opp(R4, FakeO4(fail=[7]))
        self.assertEqual(code, 4, out)     # 1 of 19,156: not byte-identical, within 0.5 % -> local drift
        rep = rj(os.path.join(R4, "opp", "opp_report.json"))
        self.assertEqual((rep["passed"], rep["usable_so_far"], rep["exit_code"]), (False, False, 4))
        self.assertFalse(rep["seam_val"]["file_sha256_equal"])
        self.assertFalse(all(rep["seam_val"]["arrays_equal"].values()))
        code, out = self.opp(R4, FakeO4(), extra=["--rerun", "--reason", "checked the local code"])
        self.assertEqual(code, 0, out)      # a drift report may be rerun
        R4 = B.copy_r4("o_seam96", parts=("sw", "unpack_sw.json"))
        self.assertEqual(self.opp(R4, FakeO4(fail=range(96)))[0], 3)    # 96 of 19,156 > 0.5 %: the literal 3
        R4 = B.copy_r4("o_drift", parts=("sw", "unpack_sw.json"))
        code, out = self.opp(R4, FakeO4(perturb=True))
        self.assertEqual(code, 4, out)
        rep = rj(os.path.join(R4, "opp", "opp_report.json"))
        self.assertFalse(rep["seam_val"]["arrays_equal"]["o4"])
        self.assertTrue(rep["seam_val"]["arrays_equal"]["t4"])

    def test_workers_range(self):
        R4 = B.copy_r4("o_w", parts=("sw", "unpack_sw.json"))
        self.assertEqual(self.opp(R4, FakeO4(), extra=["--workers", "9"])[0], 2)

    # ---- review round 2
    def test_small_replay_failures_mask_and_apply_consistency(self):
        """Rev 3 items 3, 7: with 3 public and 2 private replay failures (<= 0.5 %, parsed slots) opp's registered O4
        dkappa is computed on parsed & ~replay_failed (by hand here) and apply reproduces it bit for bit; apply stops
        with 4 (o4_consistency, no test output) when opp's record differs by one bit (public, private, seam_val)."""
        pub0 = 19156
        fails = []
        for s, pick in (("public", (5, 6, 300)), ("private", (10, 11))):
            vk = rf.load_val_keys(os.path.join(B.D2, "rank", s, "dumps"))
            fails.append((s, [int(i) for i in np.nonzero(np.asarray(vk["parsed"], dtype=bool))[0][list(pick)]]))
        n_pub = rf.load_val_keys(os.path.join(B.D2, "rank", "public", "dumps"))["n"]
        glob_rows = [pub0 + i for i in fails[0][1]] + [pub0 + n_pub + i for i in fails[1][1]]
        R4 = B.copy_r4("o_small", parts=("sw", "unpack_sw.json"))
        code, out = self.opp(R4, FakeO4(fail=glob_rows))
        self.assertEqual(code, 0, out)
        rep = rj(os.path.join(R4, "opp", "opp_report.json"))
        self.assertEqual({s: rep["sets"][s]["failures"] for s in t04.SETS}, {"seam_val": 0, "public": 3, "private": 2})
        base = rj(os.path.join(B.R4, "opp", "opp_report.json"))
        for s in t04.TEST_SETS:
            vk = rf.load_val_keys(os.path.join(B.D2, "rank", s, "dumps"))
            parsed = np.asarray(vk["parsed"], dtype=bool)
            with np.load(os.path.join(R4, "opp", f"o4_{s}.npz"), allow_pickle=False) as z:
                o4, t4, rfail = z["o4"], z["t4"], np.asarray(z["replay_failed"], dtype=bool)
            self.assertEqual(int((parsed & rfail).sum()), 3 if s == "public" else 2)
            mask = parsed & ~rfail
            hand = t04.pf.val_side_kappa(dict(vk, parsed=mask), rf.log_softmax(np.where(mask[:, None], t4, 0.0)),
                                         rf.log_softmax(np.where(mask[:, None], o4, 0.0)))["dkappa"]
            unmasked = t04.pf.val_side_kappa(vk, rf.log_softmax(np.where(parsed[:, None], t4, 0.0)),
                                             rf.log_softmax(np.where(parsed[:, None], o4, 0.0)))["dkappa"]
            self.assertEqual(rep["o4_dkappa_early"][s]["dkappa"], hand, s)
            self.assertNotEqual(hand, unmasked, s)                                   # the mask matters here
            self.assertNotEqual(hand, base["o4_dkappa_early"][s]["dkappa"], s)
        code, out = run(["apply", "--dry-run"] + B.dirs(R4))
        self.assertEqual(code, 0, out)
        ar = rj(os.path.join(R4, "rank", "t04_apply_report.json"))
        for s in t04.TEST_SETS:
            self.assertEqual(ar["colour_check"][s]["O4"]["dkappa"], rep["o4_dkappa_early"][s]["dkappa"], s)
            self.assertEqual(ar["sets"][s]["o4_replay_failed"], 3 if s == "public" else 2)
        # one bit off in opp's record: local drift (4) at o4_consistency, no test output
        for where, reason in (("public", "public_o4_dkappa"), ("private", "private_o4_dkappa"),
                              ("seam_val", "seam_val_o4_dkappa")):
            with self.subTest(record=where):
                R4n = B.copy_r4("o_bit_" + where)

                def bump(r):
                    rec, key = (r["seam_val"], "dkappa_O4") if where == "seam_val" else \
                        (r["o4_dkappa_early"][where], "dkappa")
                    self.assertNotEqual(rec[key] + 1e-15, rec[key])
                    rec[key] = rec[key] + 1e-15
                edit_json(os.path.join(R4n, "opp", "opp_report.json"), bump)
                code, out = run(["apply", "--dry-run"] + B.dirs(R4n))
                self.assertEqual(code, 4, out)
                v = rj(os.path.join(R4n, "rank", "attempt_verdict.json"))
                self.assertEqual((v["exit_code"], v["stage"], v["reason"]), (4, "o4_consistency", reason))
                for n in ("t04_apply_report.json", "submission_rank.csv", "rank_test_probs.csv", "pred_B.csv"):
                    self.assertFalse(os.path.exists(os.path.join(R4n, "rank", n)), n)


class UnpackDrills(unittest.TestCase):
    def unpack(self, zpath, R4, **patch):
        return run(["unpack", "--dry-run", "--zip", zpath] + B.dirs(R4), **patch)

    def zip_with(self, name, fn):
        sw = B.copy_sw(name)
        fn(sw)
        zp = os.path.join(B.base, name, f4.ZIP_NAME)
        zip_sw(sw, zp)
        return zp

    def test_seam_failed(self):
        def fn(sw):
            edit_json(os.path.join(sw, "seam_check.json"), lambda s: s.update(passed=False))
            edit_json(os.path.join(sw, "run_log.json"),
                      lambda o: (o.update(status="seam_failed"), o["runs"][-1].update(status="seam_failed")))
        zp = self.zip_with("u_seam", fn)
        R4 = os.path.join(B.base, "u_seam", "r4")
        code, out = self.unpack(zp, R4)
        self.assertEqual(code, 1, out)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))
        shutil.copyfile(zp, os.path.join(R4, f4.ZIP_NAME)) if os.path.isdir(R4) else (
            os.makedirs(R4), shutil.copyfile(zp, os.path.join(R4, f4.ZIP_NAME)))
        shutil.copyfile(zp, os.path.join(R4, "test04_rank_results (1).zip"))
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "seam failed"])
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(R4, "failed", "attempt_1", "attempt.json"))
        self.assertEqual((att["failure_class"], att["counts_toward_rerun"]), ("notebook_seam", True))
        self.assertEqual(sorted(os.listdir(os.path.join(R4, "failed_zips", "attempt_1"))),
                         ["test04_rank_results (1).zip", "test04_rank_results.zip"])
        self.assertEqual(sorted(att["zips"]), ["test04_rank_results (1).zip", "test04_rank_results.zip"])
        self.assertFalse([f for f in os.listdir(R4) if f.endswith(".zip")])
        # the same (seam-failed) zip again: not a recomputed result -> 2 (review round 1: freshness is checked before
        # the records; it used to say 1 here and send the operator into a refused second archive)
        code, out = self.unpack(zp, R4)
        self.assertEqual(code, 2, out)
        self.assertIn("不是重算的結果", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))
        good = self.zip_with("u_seam_same", lambda sw: None)
        att["zips"]["x.zip"] = sha(good)
        wj(os.path.join(R4, "failed", "attempt_1", "attempt.json"), att)
        code, out = self.unpack(good, R4)
        self.assertEqual(code, 2, out)
        self.assertIn("不是重算的結果", out)

    def test_earlier_failure_warns_outside_the_repo(self):
        zp = self.zip_with("u_earlier", lambda sw: edit_json(
            os.path.join(sw, "run_log.json"),
            lambda o: o.update(runs=[{"run": 1, "status": "seam_failed"}, dict(o["runs"][0], run=1)])))
        R4 = os.path.join(B.base, "u_earlier", "r4")
        code, out = self.unpack(zp, R4)
        self.assertEqual(code, 0, out)
        self.assertIn("WARNING", out)
        self.assertEqual(rj(os.path.join(R4, "unpack_sw.json"))["earlier_failed_runs"],
                         [{"run": 1, "status": "seam_failed"}])

    def test_resumed_bundle_status_members(self):
        cases = {
            "resumed": (lambda sw: edit_json(os.path.join(sw, "public", "dumps", "_DONE_val_p03b_s0_swap.json"),
                                             lambda m: m.update(run=0)), 1, "resumed"),
            "bundle": (lambda sw: edit_json(os.path.join(sw, "config.json"),
                                            lambda c: c["test_bundle"].update(manifest_sha256="0" * 64)), 2, "MANIFEST"),
            "flags": (lambda sw: edit_json(os.path.join(sw, "config.json"), lambda c: c.update(smoke=True)), 2,
                      "smoke"),
            "interrupted": (lambda sw: edit_json(os.path.join(sw, "run_log.json"),
                                                 lambda o: (o.update(status="interrupted"),
                                                            o["runs"][-1].update(status="interrupted"))), 2,
                            "sw_interrupted_N"),
            "missing member": (lambda sw: os.remove(os.path.join(sw, "private", "dumps", "_INPUT.json")), 2,
                               "missing required members")}
        for label, (fn, code_want, text) in cases.items():
            with self.subTest(case=label):
                zp = self.zip_with("u_case", fn)
                R4 = os.path.join(B.base, "u_case", "r4")
                code, out = self.unpack(zp, R4)
                self.assertEqual(code, code_want, out)
                self.assertIn(text, out)
                self.assertFalse(os.path.exists(os.path.join(R4, "sw")))

    def test_unsafe_members(self):
        zp = os.path.join(B.base, "u_bad.zip")
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("rank/config.json", "{}")
        code, out = self.unpack(zp, os.path.join(B.base, "u_bad"))
        self.assertEqual(code, 2)
        self.assertIn("outside sw/", out)
        # the t02 unpacker refuses a test04 zip (members under sw/, not rank/)
        with self.assertRaises(SystemExit):
            tb.cmd_unpack(tb.build_parser().parse_args(
                ["unpack", "--zip", B.zip, "--comp", "rank", "--results-dir", os.path.join(B.base, "u_t02"),
                 "--tests-dir", B.T]))

    def test_second_counted_archive_refused(self):
        R4 = os.path.join(B.base, "u_two", "r4")
        os.makedirs(os.path.join(R4, "failed", "attempt_1"))
        wj(os.path.join(R4, "failed", "attempt_1", "attempt.json"),
           {"counts_toward_rerun": True, "archived_at": "2026-09-01T00:00:00+08:00", "zips": {}, "done_times": {}})
        zp = self.zip_with("u_two_zip", lambda sw: edit_json(os.path.join(sw, "seam_check.json"),
                                                            lambda s: s.update(passed=False)))
        shutil.copyfile(zp, os.path.join(R4, f4.ZIP_NAME))
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "second"])
        self.assertEqual(code, 2, out)
        self.assertIn("登記只准重跑一次", out)
        # the rerun unpack prints the notice, the fresh zip is accepted
        good = self.zip_with("u_two_good", lambda sw: None)
        R4b = os.path.join(B.base, "u_two_b", "r4")
        shutil.copytree(os.path.join(R4, "failed"), os.path.join(R4b, "failed"))
        code, out = self.unpack(good, R4b)
        self.assertEqual(code, 0, out)
        self.assertIn("這是登記的重跑", out)
        self.assertEqual(rj(os.path.join(R4b, "unpack_sw.json"))["attempt_number"], 2)

    def test_archive_refusals(self):
        R4 = os.path.join(B.base, "a_none", "r4")
        os.makedirs(R4)
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "x"])
        self.assertEqual(code, 2)
        self.assertIn("nothing to archive", out)
        R4 = B.copy_r4("a_noevidence")
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "x"])
        self.assertEqual(code, 2)
        self.assertIn("no seam-class evidence", out)
        R4 = B.copy_r4("a_opp")
        edit_json(os.path.join(R4, "sw", "seam_check.json"), lambda s: s.update(passed=False))
        edit_json(os.path.join(R4, "opp", "opp_report.json"), lambda r: r.update(usable_so_far=False))
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "x"])
        self.assertEqual(code, 2)
        self.assertIn("usable_so_far false", out)
        self.assertIn("registered decision", out)
        # rev 3 check (item 9): an opp drift (exit 4) is not a registered decision; it blocks archive until opp is
        # rerun, with or without a PREREG_04 revision
        R4 = B.copy_r4("a_oppdrift")
        edit_json(os.path.join(R4, "sw", "seam_check.json"), lambda s: s.update(passed=False))
        edit_json(os.path.join(R4, "opp", "opp_report.json"), lambda r: r.update(
            exit_code=4, passed=False, usable_so_far=False, unusable_reasons=["seam_file: x"]))
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "x"])
        self.assertEqual(code, 2, out)
        self.assertIn("本機漂移", out)
        self.assertNotIn("registered decision", out)
        with patched(t04, git_show_head=lambda rel: "| rev 4 | x |\n", require_committed=lambda p, w: {},
                     git_first_commit_with=lambda rel, n: ("abc", "2099-01-01T00:00:00+08:00")):
            code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "x", "--bug-revision", "4"])
        self.assertEqual(code, 2, out)
        self.assertIn("本機漂移", out)
        self.assertTrue(os.path.isdir(os.path.join(R4, "sw")))
        self.assertFalse(os.path.exists(os.path.join(R4, "failed")))

    def test_archive_classes(self):
        cases = {"notebook_run_log": lambda R4: edit_json(os.path.join(R4, "sw", "run_log.json"), lambda o: o.update(
                     runs=[{"run": 1, "status": "readonly_changed"}, dict(o["runs"][0], run=2)])),
                 "resumed": lambda R4: edit_json(os.path.join(R4, "sw", "seam_val", "dumps", "_DONE_val_keys.json"),
                                                 lambda m: m.update(run=7)),
                 "apply_exit_1": lambda R4: wj(os.path.join(R4, "rank", "attempt_verdict.json"),
                                               {"exit_code": 1, "stage": "local_seam"})}
        for cls_, fn in cases.items():
            with self.subTest(failure_class=cls_):
                R4 = B.copy_r4("a_cls")
                fn(R4)
                code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "x"])
                self.assertEqual(code, 0, out)
                att = rj(os.path.join(R4, "failed", "attempt_1", "attempt.json"))
                self.assertEqual((att["failure_class"], att["counts_toward_rerun"]), (cls_, True))
                self.assertTrue(os.path.isdir(os.path.join(R4, "opp")))           # opp stays
                self.assertFalse(os.path.exists(os.path.join(R4, "sw")))
                self.assertTrue(os.path.isdir(os.path.join(R4, "failed", "attempt_1", "sw")))
                self.assertTrue(os.path.exists(os.path.join(R4, "failed", "attempt_1", "unpack_sw.json")))
                self.assertIn("unpack_sw.json", os.listdir(os.path.join(R4, "failed", "attempt_1")))
                self.assertEqual(sorted(att["done_times"]), sorted(f"_DONE_{st}.json" for st in t04.marker_stems()))
                self.assertIsNotNone(att["archived_at"])
        # a bug revision: not counted
        R4 = B.copy_r4("a_bug")
        with patched(t04, git_show_head=lambda rel: "| rev 4 | x |\n", require_committed=lambda p, w: {},
                     git_first_commit_with=lambda rel, n: ("abc", "2099-01-01T00:00:00+08:00")):
            code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "bug", "--bug-revision", "4"])
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(R4, "failed", "attempt_1", "attempt.json"))
        self.assertEqual((att["failure_class"], att["counts_toward_rerun"], att["bug_revision"]["n"]),
                         ("bug_revision", False, 4))
        # review round 3: an uncounted archive is renamed sw_exempt_N on Drive (the notebook does not count it)
        self.assertEqual({k: att["drive_rename"][k] for k in ("folder_from", "folder_to", "zip_from", "zip_to")},
                         {"folder_from": "sw", "folder_to": "sw_exempt_1", "zip_from": "test04_rank_results.zip",
                          "zip_to": "test04_rank_results_exempt_1.zip"})
        self.assertIn("Drive: sw -> sw_exempt_1", out)
        # numbering continues
        edit_json(os.path.join(R4, "failed", "attempt_1", "attempt.json"), lambda a: None)
        shutil.copytree(os.path.join(R4, "failed", "attempt_1", "sw"), os.path.join(R4, "sw"))
        edit_json(os.path.join(R4, "sw", "seam_check.json"), lambda s: s.update(passed=False))
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "seam"])
        self.assertEqual(code, 0, out)
        att2 = rj(os.path.join(R4, "failed", "attempt_2", "attempt.json"))
        self.assertEqual(att2["failure_class"], "notebook_seam")
        self.assertEqual((att2["drive_rename"]["folder_to"], att2["drive_rename"]["zip_to"]),
                         ("sw_failed_2", "test04_rank_results_failed_2.zip"))

    # ---- review round 1
    def test_archive_without_a_record_counts(self):
        """An archive without attempt.json, or whose attempt.json has no counts_toward_rerun, counts (the safe side):
        a second seam-class archive is refused and the next unpack is attempt 2."""
        for label, rec in (("no attempt.json", None),
                           ("no counts_toward_rerun", {"archived_at": "2026-09-01T00:00:00+08:00", "zips": {},
                                                       "done_times": {}})):
            with self.subTest(case=label):
                R4 = os.path.join(B.base, "u_norec", "r4")
                shutil.rmtree(os.path.dirname(R4), ignore_errors=True)
                os.makedirs(os.path.join(R4, "failed", "attempt_1"))
                if rec is not None:
                    wj(os.path.join(R4, "failed", "attempt_1", "attempt.json"), rec)
                zp = self.zip_with("u_norec_zip", lambda sw: edit_json(os.path.join(sw, "seam_check.json"),
                                                                       lambda s: s.update(passed=False)))
                shutil.copyfile(zp, os.path.join(R4, f4.ZIP_NAME))
                code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "second"])
                self.assertEqual(code, 2, out)
                self.assertIn("登記只准重跑一次", out)
                self.assertFalse(os.path.exists(os.path.join(R4, "failed", "attempt_2")))
                os.remove(os.path.join(R4, f4.ZIP_NAME))
                good = self.zip_with("u_norec_good", lambda sw: None)
                code, out = self.unpack(good, R4)
                self.assertEqual(code, 0, out)
                self.assertIn("這是登記的重跑", out)
                self.assertEqual(rj(os.path.join(R4, "unpack_sw.json"))["attempt_number"], 2)

    def test_archive_records_the_unpacked_zip(self):
        """unpack --zip may read a zip outside R4 (e.g. Downloads): archive keeps its SHA-256, so the same zip is
        refused as not recomputed wherever it lies."""
        zp = self.zip_with("u_outside", lambda sw: None)
        R4 = os.path.join(B.base, "u_outside", "r4")
        code, out = self.unpack(zp, R4)
        self.assertEqual(code, 0, out)
        os.makedirs(os.path.join(R4, "rank"))
        wj(os.path.join(R4, "rank", "attempt_verdict.json"), {"exit_code": 1, "stage": "local_seam"})
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "apply exit 1"])
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(R4, "failed", "attempt_1", "attempt.json"))
        self.assertEqual(att["zips"], {"unpacked:" + f4.ZIP_NAME: sha(zp)})
        self.assertEqual(att["unpacked_zip"]["sha256"], sha(zp))
        self.assertFalse(os.path.exists(os.path.join(R4, "failed_zips")))
        code, out = self.unpack(zp, R4)
        self.assertEqual(code, 2, out)
        self.assertIn("不是重算的結果", out)
        self.assertIn("unpacked:", out)

    def test_rerun_must_be_on_g4(self):
        """Rev 3 items 4 and 9: with one counted archive, a zip whose markers are not G4's is refused (2) before any
        record check; the first attempt may run on A100."""
        R4 = os.path.join(B.base, "u_g4", "r4")
        shutil.rmtree(os.path.dirname(R4), ignore_errors=True)
        os.makedirs(os.path.join(R4, "failed", "attempt_1"))
        wj(os.path.join(R4, "failed", "attempt_1", "attempt.json"),
           {"counts_toward_rerun": True, "archived_at": "2026-09-01T00:00:00+08:00", "zips": {}, "done_times": {}})

        def a100(sw):
            for s in t04.SETS:
                for st in t04.marker_stems():
                    edit_json(os.path.join(sw, s, "dumps", f"_DONE_{st}.json"),
                              lambda m: m.update(gpu_name="NVIDIA A100-SXM4-40GB"))
        zp = self.zip_with("u_g4_zip", a100)
        code, out = self.unpack(zp, R4)
        self.assertEqual(code, 2, out)
        self.assertIn("登記的重跑只在 G4", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))
        R4b = os.path.join(B.base, "u_g4", "r4b")
        code, out = self.unpack(zp, R4b)
        self.assertEqual(code, 0, out)
        self.assertFalse(rj(os.path.join(R4b, "unpack_sw.json"))["rerun_gpu_check"]["applies"])
        self.assertEqual(t04.rerun_gpu_problems({"x": {"gpu_name": t04.G4_GPU}, "y": {}}), ["y"])

    # ---- review round 2
    def test_two_counted_archives_refuse_unpack(self):
        """Two counted archives already: unpack refuses (2) before any record check and extracts nothing."""
        R4 = os.path.join(B.base, "u_two_counted", "r4")
        shutil.rmtree(os.path.dirname(R4), ignore_errors=True)
        for n in (1, 2):
            wj(os.path.join(R4, "failed", f"attempt_{n}", "attempt.json"),
               {"counts_toward_rerun": True, "archived_at": "2026-09-01T00:00:00+08:00", "zips": {}, "done_times": {}})
        good = self.zip_with("u_two_counted_zip", lambda sw: None)
        code, out = self.unpack(good, R4)
        self.assertEqual(code, 2, out)
        self.assertIn("登記只准重跑一次", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))
        self.assertFalse(os.path.exists(os.path.join(R4, "unpack_sw.json")))

    def test_bug_revision_after_this_failure_and_once(self):
        """Rev 3 item 20: archive --bug-revision N needs rev N committed after this attempt (its run, markers,
        unpack record, apply verdict), and an earlier archive citing the same N refuses it (one revision, one
        uncounted rerun); a new revision may be used once."""
        def git(when, rows=(4,)):
            doc = "".join(f"| rev {n} | x | bug |\n" for n in rows)
            return dict(git_show_head=lambda rel: doc, require_committed=lambda p, w: {},
                        git_first_commit_with=lambda rel, needle: ("abc", when) if needle in doc else (None, None))

        def failed_apply(R4):
            os.makedirs(os.path.join(R4, "rank"), exist_ok=True)
            wj(os.path.join(R4, "rank", "attempt_verdict.json"),
               {"exit_code": 1, "stage": "local_seam", "created": "2026-11-06T12:00:00+08:00"})
        R4 = B.copy_r4("a_bugonce")
        failed_apply(R4)
        archive = ["archive", "--dry-run", "--results-dir", R4, "--reason", "bug in t04_rank apply"]
        code, out = run(archive + ["--bug-revision", "4"], **git("2026-11-06T11:00:00+08:00"))
        self.assertEqual(code, 2, out)                   # committed before this apply verdict
        self.assertIn("not after attempt_verdict.json created", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "failed")))
        code, out = run(archive + ["--bug-revision", "4"], **git("2099-01-01T00:00:00+08:00"))
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(R4, "failed", "attempt_1", "attempt.json"))
        self.assertEqual((att["failure_class"], att["counts_toward_rerun"], att["bug_revision"]["n"]),
                         ("bug_revision", False, 4))
        whats = [o["what"] for o in att["bug_revision"]["overturns"]]
        for need in ("run_log run 1 start", "unpack_sw.json created", "attempt_verdict.json created",
                     "seam_val/dumps/_DONE_val_keys.json time"):
            self.assertIn(need, whats)
        # the next notebook run failed too: the same revision is refused, the failure is not exempted twice
        shutil.copytree(os.path.join(R4, "failed", "attempt_1", "sw"), os.path.join(R4, "sw"))
        shutil.copyfile(os.path.join(R4, "failed", "attempt_1", "unpack_sw.json"), os.path.join(R4, "unpack_sw.json"))
        failed_apply(R4)
        code, out = run(archive + ["--bug-revision", "4"], **git("2099-01-01T00:00:00+08:00"))
        self.assertEqual(code, 2, out)
        self.assertIn("already used by the archive ['attempt_1']", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "failed", "attempt_2")))
        code, out = run(archive + ["--bug-revision", "5"], **git("2099-01-01T00:00:00+08:00", rows=(4, 5)))
        self.assertEqual(code, 0, out)
        self.assertEqual(rj(os.path.join(R4, "failed", "attempt_2", "attempt.json"))["bug_revision"]["n"], 5)

    # ---- review round 3
    def test_archive_a_folder_renamed_as_a_disconnection(self):
        """docs/TESTDAY_04.md 'sw_interrupted_K was a seam failure': the evidence is the zip renamed with the folder
        (test04_rank_results_interrupted_K.zip) or, without a zip, the folder's three JSON records put in R4/sw;
        --drive-folder sw_interrupted_K makes archive name that folder in the Drive rename; a bad name is 2."""
        def failed(sw):
            edit_json(os.path.join(sw, "seam_check.json"), lambda s: s.update(passed=False))
            edit_json(os.path.join(sw, "run_log.json"),
                      lambda o: (o.update(status="seam_failed"), o["runs"][-1].update(status="seam_failed")))
        zp = self.zip_with("u_intfail_zip", failed)
        R4 = os.path.join(B.base, "u_intfail", "r4")
        shutil.rmtree(os.path.dirname(R4), ignore_errors=True)
        os.makedirs(R4)
        shutil.copyfile(zp, os.path.join(R4, "test04_rank_results_interrupted_2.zip"))
        base = ["archive", "--dry-run", "--results-dir", R4, "--reason", "sw_interrupted_2 was a seam failure"]
        for bad in ("sw_failed_1", "sw_interrupted_0", "sw_x"):
            with self.subTest(drive_folder=bad):
                code, out = run(base + ["--drive-folder", bad])
                self.assertEqual(code, 2, out)
                self.assertIn("--drive-folder", out)
                self.assertFalse(os.path.exists(os.path.join(R4, "failed")))
        code, out = run(base + ["--drive-folder", "sw_interrupted_2"])
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(R4, "failed", "attempt_1", "attempt.json"))
        self.assertEqual((att["failure_class"], att["counts_toward_rerun"]), ("notebook_seam", True))
        self.assertIn("test04_rank_results_interrupted_2.zip", att["evidence_source"])
        self.assertEqual({k: att["drive_rename"][k] for k in ("folder_from", "folder_to", "zip_from", "zip_to")},
                         {"folder_from": "sw_interrupted_2", "folder_to": "sw_failed_1",
                          "zip_from": "test04_rank_results_interrupted_2.zip",
                          "zip_to": "test04_rank_results_failed_1.zip"})
        self.assertIn("Drive: sw_interrupted_2 -> sw_failed_1", out)
        # no zip (cut off before packing): the three JSON records downloaded from the Drive folder, put in R4/sw
        R4b = os.path.join(B.base, "u_intfail", "r4b")
        sw = B.copy_sw("u_intfail_json")
        failed(sw)
        os.makedirs(os.path.join(R4b, "sw"))
        for n in ("run_log.json", "seam_check.json", "config.json"):
            shutil.copyfile(os.path.join(sw, n), os.path.join(R4b, "sw", n))
        code, out = run(["archive", "--dry-run", "--results-dir", R4b, "--reason", "sw_interrupted_1 was a seam "
                         "failure (no zip)", "--drive-folder", "sw_interrupted_1"])
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(R4b, "failed", "attempt_1", "attempt.json"))
        self.assertEqual((att["failure_class"], att["drive_rename"]["folder_from"], att["drive_rename"]["folder_to"]),
                         ("notebook_seam", "sw_interrupted_1", "sw_failed_1"))
        self.assertTrue(os.path.isdir(os.path.join(R4b, "failed", "attempt_1", "sw")))
        # --drive-folder belongs to archive
        code, out = run(["status", "--dry-run", "--results-dir", R4b, "--drive-folder", "sw"])
        self.assertEqual(code, 2, out)
        self.assertIn("--drive-folder belongs to archive", out)

    # ---- K5 gate round 1
    def test_seam_failure_of_the_registered_rerun_has_no_third_run(self):
        """K5 gate round 1: with one counted archive, a seam-class failure (exit 1) of the registered rerun says there
        is no third run (docs/TESTDAY_04.md 'rerun once' item 6), never 'archive first' (archive refuses it); the
        first attempt still says 'archive first'. unpack and apply alike (apply's attempt_verdict.json keeps it)."""
        counted = {"counts_toward_rerun": True, "failure_class": "notebook_seam",
                   "archived_at": "2026-09-01T00:00:00+08:00", "zips": {}, "done_times": {}}

        def failed(sw):
            edit_json(os.path.join(sw, "seam_check.json"), lambda s: s.update(passed=False))
            edit_json(os.path.join(sw, "run_log.json"),
                      lambda o: (o.update(status="seam_failed"), o["runs"][-1].update(status="seam_failed")))
        zp = self.zip_with("u_last_zip", failed)
        R4 = os.path.join(B.base, "u_last", "r4")
        shutil.rmtree(os.path.dirname(R4), ignore_errors=True)
        code, out = self.unpack(zp, R4)
        self.assertEqual(code, 1, out)
        self.assertIn(t04.MSG_SEAM, out)
        self.assertNotIn(t04.MSG_SEAM_LAST, out)
        wj(os.path.join(R4, "failed", "attempt_1", "attempt.json"), counted)
        code, out = self.unpack(zp, R4)
        self.assertEqual(code, 1, out)
        self.assertIn("這是登記的重跑（第 2 次", out)
        self.assertIn(t04.MSG_SEAM_LAST, out)
        self.assertNotIn("先 archive", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))
        shutil.copyfile(zp, os.path.join(R4, f4.ZIP_NAME))          # and archive does refuse it
        code, out = run(["archive", "--dry-run", "--results-dir", R4, "--reason", "second seam failure"])
        self.assertEqual(code, 2, out)
        self.assertIn("登記只准重跑一次", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "failed", "attempt_2")))
        # apply: the notebook seam failed on the registered rerun (unpack record of attempt 2)
        R4 = B.copy_r4("u_last_apply")
        edit_json(os.path.join(R4, "sw", "seam_check.json"), lambda s: s.update(passed=False))
        wj(os.path.join(R4, "failed", "attempt_1", "attempt.json"), counted)
        edit_json(os.path.join(R4, "unpack_sw.json"),
                  lambda r: r.update(attempt_number=2, failed_attempts_seen=["attempt_1"]))
        code, out = run(["apply", "--dry-run"] + B.dirs(R4))
        self.assertEqual(code, 1, out)
        self.assertIn(t04.MSG_SEAM_LAST, out)
        self.assertNotIn("先 archive", out)
        v = rj(os.path.join(R4, "rank", "attempt_verdict.json"))
        self.assertEqual((v["exit_code"], v["reason"]), (1, "notebook_seam"))
        self.assertIn(t04.MSG_SEAM_LAST, v["message"])

    def test_allow_failed_inspects_a_real_zip(self):
        """TESTDAY_04 step 4: a real test-day zip (rehearsal false, kind official) that failed its seam can be looked
        at with --dry-run --allow-failed outside the repository; the mode flags become a warning there only."""
        def fn(sw):
            edit_json(os.path.join(sw, "config.json"), lambda c: c.update(smoke=False, rehearsal=False, kind="official"))
            edit_json(os.path.join(sw, "seam_check.json"), lambda s: s.update(passed=False))
            edit_json(os.path.join(sw, "run_log.json"),
                      lambda o: (o.update(status="seam_failed"), o["runs"][-1].update(status="seam_failed")))
        zp = self.zip_with("u_look", fn)
        R4 = os.path.join(B.base, "u_look", "look04")
        code, out = run(["unpack", "--dry-run", "--allow-failed", "--zip", zp] + B.dirs(R4))
        self.assertEqual(code, 0, out)
        rec = rj(os.path.join(R4, "unpack_sw.json"))
        self.assertTrue(rec["allow_failed"])
        self.assertTrue(any("rehearsal=False" in w for w in rec["warnings"]), rec["warnings"])
        self.assertTrue(any("seam" in w for w in rec["warnings"]), rec["warnings"])
        code, out = self.unpack(zp, os.path.join(B.base, "u_look", "r4b"))
        self.assertEqual(code, 1, out)
        zp2 = self.zip_with("u_look_flags", lambda sw: edit_json(os.path.join(sw, "config.json"),
                                                                 lambda c: c.update(rehearsal=False)))
        code, out = self.unpack(zp2, os.path.join(B.base, "u_look_flags", "r4"))
        self.assertEqual(code, 2, out)
        self.assertIn("rehearsal=False", out)

    def test_damaged_member_is_a_refusal(self):
        """A CRC error is exit 2 with a message, never Python's default exit 1 (which means a seam failure)."""
        zp = os.path.join(B.base, "u_crc.zip")
        with zipfile.ZipFile(B.zip) as src, zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as dst:
            for i in src.infolist():
                dst.writestr(i.filename, src.read(i.filename))
        with zipfile.ZipFile(zp) as z:
            info = z.getinfo("sw/run_log.json")
        with open(zp, "r+b") as f:
            f.seek(info.header_offset + 26)
            n, m = struct.unpack("<HH", f.read(4))
            f.seek(info.header_offset + 30 + n + m + 3)
            b = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([b[0] ^ 0x01]))
        R4 = os.path.join(B.base, "u_crc")
        code, out = self.unpack(zp, R4)
        self.assertEqual(code, 2, out)
        self.assertIn("damaged member", out)
        self.assertNotIn("Traceback", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))

    def test_unexpected_error_is_exit_2(self):
        def boom(*a, **k):
            raise KeyError("boom")
        code, out = self.unpack(B.zip, os.path.join(B.base, "u_boom"), freshness=boom)
        self.assertEqual(code, 2, out)
        self.assertIn("unexpected internal error KeyError", out)
        self.assertIn("Traceback", out)


# ================================================================ real mode (constants pointed at temp trees)
class RealMode(unittest.TestCase):
    NOW = dt.datetime(2026, 11, 6, 12, 0, 0, tzinfo=TW)

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="c10_t04real_", dir=B.base)
        self.R4 = os.path.join(self.root, "results", "test04")
        self.O2 = os.path.join(self.root, "o2")
        shutil.copytree(B.O2, self.O2)
        edit_json(os.path.join(self.O2, "apply_test_report.json"), lambda r: r.update(mode="real"))
        self.nb = os.path.join(self.root, "test04_rank.ipynb")
        cells = ["NOTEBOOK_SHA256 = '%s'\n" % ("0" * 64), "x = 1\n"]
        nb = {"cells": [{"cell_type": "code", "source": [c], "metadata": {}, "outputs": [], "execution_count": None}
                        for c in cells], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        wj(self.nb, nb)
        _r, h = t2.notebook_source_sha256(self.nb)
        nb["cells"][0]["source"] = ["NOTEBOOK_SHA256 = '%s'\n" % h]
        wj(self.nb, nb)
        self.nb_sha = h
        self.log = os.path.join(self.root, "official_scores.json")
        self.write_log("2026-09-01T10:00:00+08:00")
        self.patch = dict(REAL_R4=self.R4, REAL_D2=B.D2, REAL_O2=self.O2, REAL_T=B.T, REAL_C=B.C,
                          FAILED_ZIP_ROOT=os.path.join(self.root, "test04_failed"), OFFICIAL_LOG=self.log,
                          NOTEBOOK04=self.nb, now=lambda: self.NOW,
                          require_committed=lambda paths, what: {"files": len(paths), "rule": "mocked (tests)"})

    def write_log(self, uploaded_at, sha_=None, status="uploaded"):
        s = sha_ or sha(os.path.join(self.O2, "submission_rank.csv"))
        wj(self.log, {"format": "c10 official_scores v1", "uploads": [
            {"id": 1, "comp": "rank", "status": status, "file": t04.R20_SUBMISSION_REL, "sha256": s,
             "uploaded_at": uploaded_at if status == "uploaded" else None}]})

    def real_zip(self, name, fn=None, start="2026-11-06T09:00:00+08:00"):
        sw = B.copy_sw(name)

        def edit(c):
            c.update(smoke=False, rehearsal=False, kind="official", notebook_sha256=self.nb_sha)
        edit_json(os.path.join(sw, "config.json"), edit)
        edit_json(os.path.join(sw, "run_log.json"), lambda o: o["runs"][0].update(start=start))
        for s in t04.SETS:
            for st in t04.marker_stems():
                edit_json(os.path.join(sw, s, "dumps", f"_DONE_{st}.json"),
                          lambda m: m.update(time="2026-11-06T09:30:00+08:00"))
        if fn:
            fn(sw)
        zp = os.path.join(self.root, name, f4.ZIP_NAME)
        zip_sw(sw, zp)
        return zp

    def unpack(self, zp):
        return run(["unpack", "--zip", zp], **self.patch)

    def test_r20_first_upload(self):
        rep = rj(os.path.join(self.O2, "apply_test_report.json"))
        with patched(t04, OFFICIAL_LOG=self.log, require_committed=self.patch["require_committed"]):
            r = t04.r20_first_upload({"real": True}, rep)
            self.assertEqual(r["id"], 1)
            for label, kw in (("prepared only", dict(uploaded_at=None, status="prepared")),
                              ("another file", dict(uploaded_at="2026-11-04T12:00:00+08:00", sha_="0" * 64))):
                with self.subTest(case=label):
                    self.write_log(**kw)
                    with self.assertRaises(t04.T04Error) as cm:
                        t04.r20_first_upload({"real": True}, rep)
                    self.assertEqual(cm.exception.code, 2)
                    self.assertIn("R20 的第一次上傳", str(cm.exception))

    def test_unpack_real_ok_then_freshness(self):
        zp = self.real_zip("r_ok")
        code, out = self.unpack(zp)
        self.assertEqual(code, 0, out)
        rec = rj(os.path.join(self.R4, "unpack_sw.json"))
        self.assertEqual((rec["mode"], rec["attempt_number"], rec["r20_first_upload"]["id"]), ("real", 1, 1))
        self.assertIn("下一步：commit results/test04/sw", out)
        # a seam failure found by apply later: archive it (apply_exit_1), then the same zip is refused
        os.makedirs(os.path.join(self.R4, "rank"))
        wj(os.path.join(self.R4, "rank", "attempt_verdict.json"), {"exit_code": 1, "stage": "local_seam"})
        shutil.copyfile(zp, os.path.join(self.R4, f4.ZIP_NAME))
        code, out = run(["archive", "--reason", "local seam failed"], **self.patch)
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"))
        self.assertEqual((att["failure_class"], att["counts_toward_rerun"]), ("apply_exit_1", True))
        self.assertTrue(os.path.exists(os.path.join(self.patch["FAILED_ZIP_ROOT"], "attempt_1", f4.ZIP_NAME)))
        self.assertIn("sw_failed_1", out)
        self.assertIn("登記的重跑，只用 G4", out)
        code, out = self.unpack(zp)
        self.assertEqual(code, 2, out)
        self.assertIn("不是重算的結果", out)
        # a new zip whose seam_val marker times equal the archived ones
        zp2 = self.real_zip("r_old_times", start="2026-12-01T00:00:00+08:00")
        edit_json(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"),
                  lambda a: a.update(archived_at="2026-11-06T10:00:00+08:00"))
        with zipfile.ZipFile(zp2) as z:
            m = json.loads(z.read("sw/seam_val/dumps/_DONE_val_keys.json"))
        edit_json(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"),
                  lambda a: a["done_times"].update({"_DONE_val_keys.json": m["time"]}))
        code, out = self.unpack(zp2)
        self.assertEqual(code, 2, out)
        self.assertIn("is not later than the archived", out)
        # a run that started before the archive
        zp3 = self.real_zip("r_early", fn=lambda sw: None, start="2026-11-06T09:59:00+08:00")
        edit_json(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"), lambda a: a.update(done_times={}))
        code, out = self.unpack(zp3)
        self.assertEqual(code, 2, out)
        self.assertIn("is not after the archive", out)

    def test_before_r20_first_upload(self):
        self.write_log("2026-11-06T10:00:00+08:00")        # after the run start (09:00) and the markers (09:30)
        zp = self.real_zip("r_before")
        code, out = self.unpack(zp)
        self.assertEqual(code, 2, out)
        self.assertIn("R20 上傳前跑的 test04 不用", out)
        os.makedirs(self.R4, exist_ok=True)
        shutil.copyfile(zp, os.path.join(self.R4, f4.ZIP_NAME))
        code, out = run(["archive", "--reason", "ran before R20 U1"], **self.patch)
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"))
        self.assertEqual((att["failure_class"], att["counts_toward_rerun"]), ("before_r20_u1", False))
        # review round 3: the user renames it sw_exempt_1 (the notebook does not count it, no G4-only rule after it)
        self.assertIn("C10/results/test04/sw 改名成 sw_exempt_1", out)
        self.assertIn("test04_rank_results_exempt_1.zip", out)
        self.assertIn("G4 優先、A100 備用", out)

    def test_notebook_identity_and_earlier_failures(self):
        zp = self.real_zip("r_nb", fn=lambda sw: edit_json(os.path.join(sw, "config.json"),
                                                           lambda c: c.update(notebook_sha256="f" * 64)))
        code, out = self.unpack(zp)
        self.assertEqual(code, 1, out)
        self.assertIn("committed colab/test04_rank.ipynb", out)
        os.makedirs(self.R4, exist_ok=True)
        shutil.copyfile(zp, os.path.join(self.R4, f4.ZIP_NAME))
        code, out = run(["archive", "--reason", "old notebook"], **self.patch)
        self.assertEqual(code, 0, out)
        self.assertEqual(rj(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"))["failure_class"],
                         "not_committed_notebook")
        zp = self.real_zip("r_earlier", fn=lambda sw: edit_json(os.path.join(sw, "run_log.json"), lambda o: o.update(
            runs=[{"run": 1, "status": "final_check_failed", "start": "2026-11-06T08:00:00+08:00"},
                  dict(o["runs"][0], run=1)])))
        shutil.rmtree(os.path.join(self.R4, "failed"))
        code, out = self.unpack(zp)
        self.assertEqual(code, 1, out)
        self.assertIn("earlier failed runs", out)

    def test_real_mode_refusals_before_anything(self):
        zp = self.real_zip("r_nolog")
        os.remove(self.log)
        code, out = self.unpack(zp)
        self.assertEqual(code, 2, out)
        self.assertFalse(os.path.exists(self.R4))

    # ---- review round 1
    def log_not_committed(self, paths, what):
        if any(t2.same_path(p, self.log) for p in paths):
            raise t04.T04Error("results/official_scores.json is not committed and clean (mocked)", 2)
        return {"files": len(paths), "rule": "mocked (tests)"}

    def test_archive_refuses_while_r20_u1_cannot_be_checked(self):
        """An uncommitted results/official_scores.json used to turn a counted seam failure into an uncounted
        before_r20_u1 archive (three notebook runs possible). Now archive stops (2) and moves nothing; a second
        seam-class failure after the counted archive is refused whatever the log, also for a run before R20's U1."""
        zp = self.real_zip("r_dirty")
        code, out = self.unpack(zp)
        self.assertEqual(code, 0, out)
        os.makedirs(os.path.join(self.R4, "rank"))
        wj(os.path.join(self.R4, "rank", "attempt_verdict.json"), {"exit_code": 1, "stage": "local_seam"})
        shutil.copyfile(zp, os.path.join(self.R4, f4.ZIP_NAME))
        dirty = dict(self.patch, require_committed=self.log_not_committed)
        code, out = run(["archive", "--reason", "local seam failed"], **dirty)
        self.assertEqual(code, 2, out)
        self.assertIn("先 commit", out)
        self.assertTrue(os.path.isdir(os.path.join(self.R4, "sw")))
        self.assertTrue(os.path.exists(os.path.join(self.R4, f4.ZIP_NAME)))
        self.assertFalse(os.path.exists(os.path.join(self.R4, "failed")))
        code, out = run(["archive", "--reason", "local seam failed"], **self.patch)
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"))
        self.assertEqual((att["failure_class"], att["counts_toward_rerun"], att["before_r20_u1_problems"]),
                         ("apply_exit_1", True, []))

        def failed_later(sw):
            edit_json(os.path.join(sw, "seam_check.json"), lambda s: s.update(passed=False))
            edit_json(os.path.join(sw, "run_log.json"),
                      lambda o: (o.update(status="seam_failed"), o["runs"][-1].update(status="seam_failed")))
            for s in t04.SETS:
                for st in t04.marker_stems():
                    edit_json(os.path.join(sw, s, "dumps", f"_DONE_{st}.json"),
                              lambda m: m.update(time="2026-12-01T09:30:00+08:00"))
        zp2 = self.real_zip("r_dirty2", fn=failed_later, start="2026-12-01T09:00:00+08:00")
        code, out = self.unpack(zp2)
        self.assertEqual(code, 1, out)
        shutil.copyfile(zp2, os.path.join(self.R4, f4.ZIP_NAME))
        for label, patch in (("log committed", self.patch), ("log not committed", dirty)):
            with self.subTest(case=label):
                code, out = run(["archive", "--reason", "second seam failure"], **patch)
                self.assertEqual(code, 2, out)
                self.assertIn("登記只准重跑一次" if patch is self.patch else "先 commit", out)
                self.assertFalse(os.path.exists(os.path.join(self.R4, "failed", "attempt_2")))
        # the same failure logged as a run before R20's U1 (U1 after the run): still refused, not before_r20_u1
        self.write_log("2026-12-05T10:00:00+08:00")
        code, out = run(["archive", "--reason", "second seam failure"], **self.patch)
        self.assertEqual(code, 2, out)
        self.assertIn("登記只准重跑一次", out)
        self.assertFalse(os.path.exists(os.path.join(self.R4, "failed", "attempt_2")))

    # ---- review round 2
    def zip_two_runs(self, name, first_status, run2_status, seam_passed, run1_start, run2_start, marker_time):
        """A Drive folder whose run 1 stopped (preflight / GPU gate) and whose run 2 computed everything."""
        def fn(sw):
            edit_json(os.path.join(sw, "seam_check.json"), lambda s: s.update(passed=seam_passed))

            def log(o):
                r2 = dict(o["runs"][0], run=2, status=run2_status, start=run2_start)
                r1 = {"run": 1, "status": first_status, "start": run1_start, "end": run1_start}
                o.update(status=run2_status, runs=[r1, r2])
            edit_json(os.path.join(sw, "run_log.json"), log)
            for s in t04.SETS:
                for st in t04.marker_stems():
                    edit_json(os.path.join(sw, s, "dumps", f"_DONE_{st}.json"),
                              lambda m: m.update(time=marker_time, run=2))
        return self.real_zip(name, fn=fn, start=run1_start)

    def test_a_stop_before_r20_u1_is_not_the_run(self):
        """Rev 3 item 1: a preflight / GPU-gate stop of the same Drive folder before R20's U1 is not a run. The run
        that computed the outputs after U1: a seam failure is a counted notebook_seam archive (not before_r20_u1),
        a completed run is unpacked; a run that computed before U1 is still refused."""
        self.write_log("2026-11-06T10:00:00+08:00")
        for first in ("preflight_failed", "interrupted"):
            with self.subTest(first_run=first):
                shutil.rmtree(self.R4, ignore_errors=True)
                shutil.rmtree(self.patch["FAILED_ZIP_ROOT"], ignore_errors=True)
                zp = self.zip_two_runs("r_stop_" + first, first, "seam_failed", False, "2026-11-06T09:00:00+08:00",
                                       "2026-11-06T11:00:00+08:00", "2026-11-06T11:30:00+08:00")
                code, out = self.unpack(zp)
                self.assertEqual(code, 1, out)
                shutil.copyfile(zp, os.path.join(self.R4, f4.ZIP_NAME)) if os.path.isdir(self.R4) else (
                    os.makedirs(self.R4), shutil.copyfile(zp, os.path.join(self.R4, f4.ZIP_NAME)))
                code, out = run(["archive", "--reason", "seam failed after U1"], **self.patch)
                self.assertEqual(code, 0, out)
                att = rj(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"))
                self.assertEqual((att["failure_class"], att["counts_toward_rerun"], att["before_r20_u1_problems"]),
                                 ("notebook_seam", True, []))
                shutil.rmtree(self.R4)
                zp = self.zip_two_runs("r_ok_" + first, first, "completed", True, "2026-11-06T09:00:00+08:00",
                                       "2026-11-06T11:00:00+08:00", "2026-11-06T11:30:00+08:00")
                code, out = self.unpack(zp)
                self.assertEqual(code, 0, out)
                self.assertEqual(rj(os.path.join(self.R4, "unpack_sw.json"))["attempt_number"], 1)
        shutil.rmtree(self.R4)
        zp = self.zip_two_runs("r_early_done", "preflight_failed", "completed", True, "2026-11-06T08:00:00+08:00",
                               "2026-11-06T09:00:00+08:00", "2026-11-06T09:30:00+08:00")
        code, out = self.unpack(zp)
        self.assertEqual(code, 2, out)
        self.assertIn("R20 上傳前跑的 test04 不用", out)
        self.assertIn("run_log run 2 (completed) start", out)
        self.assertNotIn("run_log run 1", out)

    def test_repo_notebook_committed_and_self_consistent(self):
        """unpack and archive judge 'not the committed notebook' (a counted class) against the committed, clean,
        self-consistent colab/test04_rank.ipynb only; otherwise they refuse (2, a local problem)."""
        seen = []

        def rc(paths, what):
            seen.extend(os.path.abspath(p) for p in paths)
            return {"files": len(paths), "rule": "mocked (tests)"}

        def rc_dirty(paths, what):
            if any(t2.same_path(p, self.nb) for p in paths):
                raise t04.T04Error(f"{what}: not committed and clean (mocked)", 2)
            return {"files": len(paths), "rule": "mocked (tests)"}
        zp = self.real_zip("r_nbid")
        code, out = run(["unpack", "--zip", zp], **dict(self.patch, require_committed=rc))
        self.assertEqual(code, 0, out)
        self.assertTrue(any(t2.same_path(p, self.nb) for p in seen), seen)
        shutil.rmtree(self.R4)
        code, out = run(["unpack", "--zip", zp], **dict(self.patch, require_committed=rc_dirty))
        self.assertEqual(code, 2, out)
        self.assertIn("not committed and clean (mocked)", out)
        self.assertFalse(os.path.exists(os.path.join(self.R4, "sw")))
        good = rj(self.nb)
        stale = copy.deepcopy(good)
        stale["cells"].append({"cell_type": "code", "source": ["y = 2\n"], "metadata": {}, "outputs": [],
                               "execution_count": None})
        wj(self.nb, stale)                              # a local edit: its NOTEBOOK_SHA256 line is stale now
        code, out = self.unpack(zp)
        self.assertEqual(code, 2, out)
        self.assertIn("is not self-consistent", out)
        self.assertFalse(os.path.exists(os.path.join(self.R4, "sw")))
        os.makedirs(self.R4, exist_ok=True)
        shutil.copyfile(zp, os.path.join(self.R4, f4.ZIP_NAME))
        code, out = run(["archive", "--reason", "x"], **self.patch)
        self.assertEqual(code, 2, out)
        self.assertIn("is not self-consistent", out)
        wj(self.nb, good)
        code, out = run(["archive", "--reason", "x"], **dict(self.patch, require_committed=rc_dirty))
        self.assertEqual(code, 2, out)
        self.assertIn("not committed and clean (mocked)", out)
        self.assertFalse(os.path.exists(os.path.join(self.R4, "failed")))

    def dry_check_test02(self):
        """test02's own checks in dry-run mode (its real mode needs the repository's results/test02); the rest of
        common_prereqs, and opp / apply, run their real-mode code."""
        orig = t04.check_test02

        def f(ctx):
            return orig(dict(ctx, mode="dry-run", real=False, O2=B.O2))
        return f

    def test_opp_and_apply_need_r20_first_upload(self):
        """Spec 3.0 item 6 / 10.1: opp and apply in real mode stop (2) until R20's first upload is logged as uploaded;
        with it, opp records its id and apply gets past the gate."""
        patch = dict(self.patch, check_test02=self.dry_check_test02(), o4_scores=FakeO4(), **FAST_OPP)
        for label, kw in (("no upload logged", None), ("prepared only", dict(uploaded_at=None, status="prepared"))):
            with self.subTest(case=label):
                if kw is None:
                    wj(self.log, {"format": "c10 official_scores v1", "uploads": []})
                else:
                    self.write_log(**kw)
                for cmd in (["opp", "--workers", "2"], ["apply"]):
                    code, out = run(cmd, **patch)
                    self.assertEqual(code, 2, out)
                    self.assertIn(t04.MSG_NO_R20_U1, out)
                self.assertFalse(os.path.exists(os.path.join(self.R4, "opp")))
                self.assertFalse(os.path.exists(os.path.join(self.R4, "rank")))
        self.write_log("2026-11-04T15:05:00+08:00")
        code, out = run(["opp", "--workers", "2"], **patch)
        self.assertEqual(code, 0, out)
        rep = rj(os.path.join(self.R4, "opp", "opp_report.json"))
        self.assertEqual((rep["mode"], rep["test02"]["r20_first_upload_id"], rep["checks"]["r20_first_upload"]["id"]),
                         ("real", 1, 1))
        code, out = run(["apply"], **patch)
        self.assertEqual(code, 2, out)
        self.assertNotIn(t04.MSG_NO_R20_U1, out)
        self.assertIn("run t04_rank.py unpack first", out)

    # ---- K5 gate round 1
    @staticmethod
    def seam_failed(sw):
        edit_json(os.path.join(sw, "seam_check.json"), lambda s: s.update(passed=False))
        edit_json(os.path.join(sw, "run_log.json"),
                  lambda o: (o.update(status="seam_failed"), o["runs"][-1].update(status="seam_failed")))

    def test_rerun_after_an_interrupted_folder_archive(self):
        """K5 gate round 1 (docs/TESTDAY_04.md 'sw_interrupted_K was a seam failure'): the new Drive sw folder already
        holds the notebook's preflight stop ('... 不是斷線') made BEFORE archive --drive-folder sw_interrupted_K, and
        the registered rerun runs in that same sw folder. The freshness check compares the runs that may have produced
        the outputs (producing_runs, rev 3 items 1 and 9), so that earlier preflight stop no longer makes the rerun
        'not recomputed' (runs[0] did: exit 2, and TESTDAY_04's 'wait for the new zip' led nowhere); a producing run
        that started before the archive is still refused."""
        zp = self.real_zip("r_int", fn=self.seam_failed)           # run 1 at 09:00, markers 09:30
        os.makedirs(self.R4)
        shutil.copyfile(zp, os.path.join(self.R4, "test04_rank_results_interrupted_1.zip"))
        code, out = run(["archive", "--reason", "sw_interrupted_1 was a seam failure", "--drive-folder",
                         "sw_interrupted_1"], **self.patch)
        self.assertEqual(code, 0, out)
        self.assertIn("Drive: sw_interrupted_1 -> sw_failed_1", out)
        self.assertIn("下一次執行：登記的重跑，只用 G4", out)
        edit_json(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"),
                  lambda a: a.update(archived_at="2026-11-06T10:00:00+08:00"))
        # still refused: a run that may have produced the outputs started before the archive (10:00)
        for label, first, run1, run2 in (
                ("the computing run started before the archive", "preflight_failed", "2026-11-06T09:40:00+08:00",
                 "2026-11-06T09:50:00+08:00"),
                ("an earlier completed run before the archive", "completed", "2026-11-06T09:40:00+08:00",
                 "2026-11-06T11:00:00+08:00")):
            with self.subTest(case=label):
                zp3 = self.zip_two_runs("r_int_old", first, "completed", True, run1, run2,
                                        "2026-11-06T11:30:00+08:00")
                code, out = self.unpack(zp3)
                self.assertEqual(code, 2, out)
                self.assertIn("不是重算的結果", out)
                self.assertIn("is not after the archive attempt_1", out)
                self.assertFalse(os.path.exists(os.path.join(self.R4, "sw")))
        # the registered rerun: run 1 = the preflight stop at 09:40 (before the archive), run 2 computed everything
        # from 11:00 (markers 11:30), all on G4
        zp2 = self.zip_two_runs("r_int_rerun", "preflight_failed", "completed", True, "2026-11-06T09:40:00+08:00",
                                "2026-11-06T11:00:00+08:00", "2026-11-06T11:30:00+08:00")
        code, out = self.unpack(zp2)
        self.assertEqual(code, 0, out)
        self.assertIn("這是登記的重跑（第 2 次", out)
        rec = rj(os.path.join(self.R4, "unpack_sw.json"))
        self.assertEqual((rec["attempt_number"], rec["counted_archives"]), (2, ["attempt_1"]))
        self.assertEqual([(r["run"], r["status"]) for r in rec["freshness"]["producing_runs"]], [(2, "completed")])
        self.assertEqual(rec["freshness"]["runs0_start"], "2026-11-06T09:40:00+08:00")
        self.assertTrue(rec["rerun_gpu_check"]["applies"])
        self.assertEqual(rec["rerun_gpu_check"]["markers_not_on_g4"], [])

    def test_next_run_after_an_uncounted_archive_that_follows_a_counted_one(self):
        """K5 gate round 1: an uncounted archive (bug_revision) after a counted one prints 'G4 only' for the next run,
        as the notebook (Drive sw_failed_*), unpack (one counted archive) and official_log (attempt >= 2) ask; not the
        first run's 'G4 優先、A100 備用' (kept when no counted archive exists: test_before_r20_first_upload)."""
        wj(os.path.join(self.R4, "failed", "attempt_1", "attempt.json"),
           {"counts_toward_rerun": True, "failure_class": "notebook_seam", "archived_at": "2026-11-06T10:00:00+08:00",
            "zips": {}, "done_times": {}})

        def later(fn):
            def f(sw):
                fn(sw)
                for s in t04.SETS:
                    for st in t04.marker_stems():
                        edit_json(os.path.join(sw, s, "dumps", f"_DONE_{st}.json"),
                                  lambda m: m.update(time="2026-11-06T11:30:00+08:00"))
            return f
        zp = self.real_zip("r_bug", fn=later(self.seam_failed), start="2026-11-06T11:00:00+08:00")
        shutil.copyfile(zp, os.path.join(self.R4, f4.ZIP_NAME))
        git = dict(git_show_head=lambda rel: "| rev 4 | x | bug |\n",
                   git_first_commit_with=lambda rel, n: ("abc", "2099-01-01T00:00:00+08:00"))
        code, out = run(["archive", "--reason", "bug in t04_rank", "--bug-revision", "4"], **dict(self.patch, **git))
        self.assertEqual(code, 0, out)
        att = rj(os.path.join(self.R4, "failed", "attempt_2", "attempt.json"))
        self.assertEqual((att["failure_class"], att["counts_toward_rerun"], att["drive_rename"]["folder_to"]),
                         ("bug_revision", False, "sw_exempt_2"))
        self.assertTrue(att["next_run"]["registered_rerun_g4_only"])
        self.assertIn("下一次仍是登記的重跑，只用 G4", out)
        self.assertNotIn("A100 備用", out)
        # the three places agree: unpack refuses an A100 rerun
        def a100(sw):
            for s in t04.SETS:
                for st in t04.marker_stems():
                    edit_json(os.path.join(sw, s, "dumps", f"_DONE_{st}.json"),
                              lambda m: m.update(gpu_name="NVIDIA A100-SXM4-40GB", time="2026-12-01T09:30:00+08:00"))
        zp2 = self.real_zip("r_bug_a100", fn=a100, start="2026-12-01T09:00:00+08:00")
        code, out = self.unpack(zp2)
        self.assertEqual(code, 2, out)
        self.assertIn("登記的重跑只在 G4", out)


# ================================================================ t04_fake_tests
class FakeTests(unittest.TestCase):
    def test_seam_val_byte_copies_and_slices(self):
        sw = os.path.join(B.S, "drive", "sw")
        for x in ("a", "b"):
            for st in ("val_keys", f"val_p03{x}_d4", f"val_p03{x}_s0_swap"):
                self.assertEqual(sha(os.path.join(sw, "seam_val", "dumps", f"{st}.npz")),
                                 sha(os.path.join(P03, f"rank_{x}", "dumps", f"{st}.npz")))
        with np.load(os.path.join(P03, "rank_a", "dumps", "val_p03a_d4.npz")) as z:
            full = z["logits"]
        rows = {"public": 4787, "private": 4815}
        lo = 0
        for s in ("public", "private"):
            with np.load(os.path.join(sw, s, "dumps", "val_keys.npz")) as k, \
                    np.load(os.path.join(sw, s, "dumps", "val_p03a_d4.npz")) as d:
                self.assertEqual(len(k["question_id"]), rows[s])
                self.assertTrue(all(q.startswith(PREFIX[s]) for q in k["question_id"].tolist()))
                self.assertEqual(d["logits"].tobytes(), full[lo:lo + rows[s]].tobytes())
                self.assertEqual(str(d["weights_sha256"]), t04.pf.SW_WEIGHTS["a"])
            m = rj(os.path.join(sw, s, "dumps", "_DONE_val_p03a_d4.json"))
            self.assertEqual((m["rows"], m["sha256"]), (rows[s], sha(os.path.join(sw, s, "dumps", "val_p03a_d4.npz"))))
            lo += rows[s]
        tree = rj(os.path.join(B.S, "FAKE_TREE_T04.json"))
        self.assertIn("never run content statistics", tree["rule"])
        cfg = rj(os.path.join(sw, "config.json"))
        self.assertEqual((cfg["smoke"], cfg["rehearsal"], cfg["kind"]), (False, True, "rehearsal"))
        self.assertEqual(cfg["test_bundle"], rj(os.path.join(B.D2, "rank", "config.json"))["test_bundle"])
        with zipfile.ZipFile(B.zip) as z:
            names = set(z.namelist())
        self.assertEqual([m for m in t04.required_members() if m not in names], [])
        self.assertTrue(all(n.startswith("sw/") for n in names))

    def test_root_inside_repo_refused(self):
        with self.assertRaises(SystemExit):
            f4.from_val(os.path.join(REPO, "results", "test04", "x"))


# ================================================================ slow: the real replay on seam_val
@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 for the O4 replay of seam_val (~1.5 min, 6 workers)")
class SlowO4SeamVal(unittest.TestCase):
    def test_seam_val_file_byte_identical_and_segment(self):
        import p04_opp as po
        tabs, _info, crec, tsha = t04.load_tables()
        self.assertEqual((crec["sha256"], tsha), (t04.COUNTS_SHA, t04.TABLES_SHA))
        vk = rf.load_val_keys(os.path.join(P02, "dumps"))
        slots = po.read_question_slots(t04.SEAM_QUESTIONS)
        po.slots_match_keys(slots, vk)
        tmp = tempfile.mkdtemp(prefix="c10_t04slow_")
        try:
            res = t04.o4_scores(slots, tabs, 6, tmp)
            path = os.path.join(tmp, "o4_seam_val.npz")
            po.save_npz(path, t04.o4_arrays(vk, res, 0, len(slots)))
            self.assertEqual(sha(path), t04.VAL_O4_SHA)
            seg = t04.o4_scores(slots[5000:7000], tabs, 3, tmp)
            for k in ("o4", "t4", "n_moves", "anomalies", "moves_parsed", "replay_failed"):
                self.assertEqual(seg[k].tobytes(), np.ascontiguousarray(res[k][5000:7000]).tobytes(), k)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
