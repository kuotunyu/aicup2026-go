"""Tests for tools/t10_b.py (PREREG_10 part B on the test day, rev 4) and the B rules added to tools/official_log.py
(fast part; CPU only, no KataGo, no data/local, no tools/score.py).

* BState: B is adopted only with the RESULTS.md line, a score_H10P.json that meets 2.6 (point >= 0, lower bound >
  -0.010, reference re-read) and the freeze; without the line nothing else is looked at (rev 3: exit 3); the line with
  inconsistent files is refused (2).
* TolP / SeamCompare: tol_P = max(0.05, 2.5 m); the GPU seam statistic (failed / counts equal, the per-game max of the
  frozen-LDA projection difference) and the rehearsal record rules (missing -> 'tol_P 【待演練】', exit 2).
* Synthetic: the dry-run P1 stand-in is deterministic and side-correct.
* Cli: p1 / p1-rehearse are exit 3 and write nothing while B is not adopted (real mode and a dry run pointing at a
  not-adopted tree); real mode takes the B directories' defaults only; --synthetic-p1 / --fake-scorer are dry-run only.
* OfficialLogB: tools/official_log.py's rev 4 wrapper gives the rev 3 result unchanged for a report without B, and for
  a 10B report adds the B rules (K5-10 for B, the GPU seam, the pool gates, K6B-10, the bound B files).
* BState13 / OfficialLogB13 (PREREG_10 rev 5, section 16): B adopted through PREREG_13 (results/prereg13: the H13P mark,
  score_H13P.json with the n_view-scaled bound, view_report.json bound to the audit as it was before apply --finalize,
  the ledger's H13P rows, K3-13, K5-13) and official_log.py's rev 5 wrapper; the repository's B is adopted this way.
The end-to-end B drills are tests/test_t10_dryrun.py BTree (C10_SLOW=1).
Run: python -m pytest tests/test_t10_b.py
"""

import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import numpy as np  # noqa: E402

import t10_player as t10  # noqa: E402
import t10_b as tb  # noqa: E402
import t10_b_fake as tbf  # noqa: E402
import official_log as ol  # noqa: E402
import p10_fuse as F  # noqa: E402

H6P = json.load(open(os.path.join(REPO, "results", "prereg06", "player", "score_H6P.json"), encoding="utf-8"))
REAL = tb.p10_paths(tb.REAL_P10)


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f)


def fake_p10(root, mean=0.001, low=-0.004, a_mean=None, mark=True, allowed=True, mark_line=None):
    """A minimal adopted PREREG_10 tree (the real freeze copied; the rest stand-ins)."""
    P = tb.p10_paths(root)
    os.makedirs(os.path.dirname(P["frozen"]), exist_ok=True)
    shutil.copyfile(REAL["frozen"], P["frozen"])
    shutil.copyfile(REAL["lda"], P["lda"])
    write(P["h10p"], {"A_mean": H6P["B_mean"] if a_mean is None else a_mean,
                      "B_minus_A": {"mean_diff": mean, "ci95": [low, 0.01]}})
    write(P["val_top5"], "question_id,top_1,top_2,top_3,top_4,top_5\nq1,a,b,c,d,e\n")
    fc = json.load(open(P["frozen"], encoding="utf-8"))
    write(P["apply"], {"score_allowed": allowed, "w_star": fc["w_star"],
                       "outputs_sha256": {"player_val_top5.csv": tb.t6.sha_lf(P["val_top5"])}})
    rng = np.random.default_rng(0)
    os.makedirs(os.path.dirname(P["val_seam"]), exist_ok=True)
    np.savez(P["val_seam"], side0=rng.normal(size=(400, 770)).astype(np.float32),
             counts0=np.full(400, 50, np.int64), failed=np.zeros(400, bool))
    write(P["val_report"], {"format": F.FORMATS["val_report"], "complete": True, "mode": "synthetic",
                            "keys_sha256": "k", "games": 400,
                            "seam": {"sha256": tb.t6.sha_raw(P["val_seam"])}})
    line = mark_line if mark_line is not None else (tb.B_ADOPTED_MARK if mark else "H10P：B **不採用**")
    write(P["results_md"], "# x\n\n" + line + "\n")
    return P


def ctx_of(root, out):
    return {"real": False, "mode": "dry-run", "P10": root, "O10P1": os.path.join(out, "p1"),
            "A10P1": os.path.join(out, "p1data"), "write_roots": [out], "now": tb.TEST_OPEN.replace(day=1)}


class BState(unittest.TestCase):
    """rev 4 section 14: b_state never refuses; adopted only when everything holds, otherwise a reason."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t10b_")
        self.root = os.path.join(self.tmp, "p10")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def state(self):
        return tb.b_state(t10, ctx_of(self.root, self.tmp))

    def test_nothing_is_not_marked(self):
        os.makedirs(self.root)
        b = self.state()
        self.assertEqual((b["adopted"], b["reason"]), (False, "not_marked"))

    def test_registered_not_adopted(self):
        fake_p10(self.root, mark=False, mean=-1.0)
        b = self.state()
        self.assertEqual((b["adopted"], b["reason"]), (False, "h10p_not_adopted"))
        self.assertIn(b["reason"], tb.SILENT)

    def test_adopted(self):
        fake_p10(self.root)
        b = self.state()
        self.assertTrue(b["adopted"], b)
        self.assertEqual(b["w_star"], 16.0)
        self.assertTrue(b["h10p"]["reference_reread_equal"])
        self.assertEqual(len(b["files_sha256"]), 8)

    def test_mark_variants(self):
        for line in ("- H10P：B **採用**", "* H10P: B **採用**（2026-10-24，score_H10P.json）", "  H10P : B **採用** x"):
            with self.subTest(line):
                shutil.rmtree(self.root, ignore_errors=True)
                fake_p10(self.root, mark_line=line)
                self.assertTrue(self.state()["adopted"])
        self.assertEqual(tb.results_md_marks(os.path.join(self.root, "nope.md")), [])

    def test_inconsistent_is_b_not_used(self):
        cases = ({"mean": -0.0001}, {"low": -0.010}, {"a_mean": H6P["B_mean"] + 1e-9}, {"allowed": "pending_audit"},
                 {"mark_line": tb.B_ADOPTED_MARK + "\nH10P：B **不採用**"}, {"mark_line": "(no mark)"},
                 {"mark": False})
        for kw in cases:
            with self.subTest(kw):
                shutil.rmtree(self.root, ignore_errors=True)
                fake_p10(self.root, **kw)
                b = self.state()
                self.assertEqual((b["adopted"], b["reason"]), (False, "inconsistent"), b)
                self.assertTrue(b["problems"])

    def test_missing_file_and_marks_without_h10p(self):
        P = fake_p10(self.root)
        os.remove(P["val_seam"])
        self.assertEqual(self.state()["reason"], "inconsistent")
        os.remove(P["h10p"])
        self.assertEqual(self.state()["reason"], "inconsistent")          # a mark without score_H10P.json

    def test_k3_10(self):
        fake_p10(self.root)
        ctx = dict(ctx_of(self.root, self.tmp), real=True)
        late, early = "2026-10-26T00:00:00+08:00", "2026-10-25T23:59:59+08:00"
        for h, m, want in ((late, early, "k3_10"), (early, late, "k3_10"), (early, None, "k3_10")):
            with self.subTest(h=h, m=m), mock.patch.object(tb.t6, "committed_clean", return_value=True), \
                    mock.patch.object(tb.t6, "git_commit_time", return_value=h), \
                    mock.patch.object(tb, "mark_first_commit", return_value=m):
                b = tb.b_state(t10, ctx)
                self.assertEqual((b["adopted"], b["reason"]), (False, want))
        self.assertEqual(tb.K3_10.isoformat(), "2026-10-25T23:59:59+08:00")
        self.assertEqual(tb.H10P_LEDGER_ROWS, (2, 4))

    def test_require_adopted(self):
        ctx = ctx_of(self.root, self.tmp)
        with self.assertRaises(t10.T10Error) as cm:
            tb.require_adopted(t10, ctx, {"adopted": False, "reason": "not_marked"}, "p1")
        self.assertEqual(cm.exception.code, 3)
        self.assertFalse(os.path.exists(tb.p1_paths(ctx)["not_used"]))
        bad = {"adopted": False, "reason": "inconsistent", "problems": ["x"]}
        with self.assertRaises(t10.T10Error) as cm:
            tb.require_adopted(t10, ctx, bad, "p1-rehearse", before_test_day=True)
        self.assertEqual(cm.exception.code, 2)
        for b in (bad, {"adopted": False, "reason": "k3_10", "problems": ["late"]}):
            with self.assertRaises(t10.T10Error) as cm:
                tb.require_adopted(t10, ctx, b, "p1")
            self.assertEqual(cm.exception.code, 3)
        ev = json.load(open(tb.p1_paths(ctx)["not_used"], encoding="utf-8"))["events"]
        self.assertEqual([e["reason"] for e in ev], ["b_inconsistent", "b_k3_10"])

    def test_h10p_rule(self):
        ref = H6P["B_mean"]
        for point, low, want in ((0.0, -0.0099, True), (0.0, -0.010, False), (-1e-6, 0.0, False), (0.02, 0.001, True)):
            with self.subTest(point=point, low=low):
                ok, _r = tb.h10p_decision({"A_mean": ref, "B_minus_A": {"mean_diff": point, "ci95": [low, 1]}}, ref)
                self.assertEqual(ok, want)
        ok, r = tb.h10p_decision({"A_mean": ref + 2e-12, "B_minus_A": {"mean_diff": 1, "ci95": [1, 1]}}, ref)
        self.assertFalse(ok)
        self.assertFalse(r["reference_reread_equal"])

    def test_real_repository_b_via_prereg13(self):
        """Rewritten by PREREG_10 rev 5 (was the rev 4 tripwire test_real_repository_is_not_adopted_now): H10P never ran,
        H13P adopted B' (results/prereg13), so the repository's B is adopted through PREREG_13."""
        self.assertFalse(tb.results_md_adopted(REAL["results_md"]))
        self.assertFalse(os.path.exists(REAL["h10p"]))
        self.assertTrue(tb.results_md_adopted(tb.p13_paths(tb.REAL_P13)["results_md13"], tb.MARK13_RE))
        b = tb.b_state(t10, real_ctx())
        self.assertEqual((b["adopted"], b["reason"], b["source"]), (True, None, "PREREG_13"), b.get("problems"))
        self.assertEqual(b["h13p"]["n_view"], 263)
        self.assertEqual(b["w_star"], 16.0)


class SecondReview(unittest.TestCase):
    """Second review: B files bound only with a 10B pool; broken B records never raise in submit; the K3-10 mark
    commit matches the mark LINE."""

    def test_b_inputs_only_with_10b(self):
        with tempfile.TemporaryDirectory(prefix="c10_t10b_") as tmp:
            root = os.path.join(tmp, "p10")
            fake_p10(root)
            ctx = ctx_of(root, tmp)
            b = tb.b_state(t10, ctx)
            write(tb.p1_paths(ctx)["gpu_seam"], {"passed": False})
            self.assertEqual(tb.b_inputs(ctx, b, {"public": "06-OFF", "private": "10A"}), [])
            self.assertEqual(tb.b_inputs(ctx, b, None), [])
            got = tb.b_inputs(ctx, b, {"public": "06-OFF", "private": "10B"})
            self.assertIn(tb.p1_paths(ctx)["gpu_seam"], got)
            self.assertIn(b["paths"]["results_md"], got)
            self.assertEqual(tb.b_inputs(ctx, {"adopted": False}, {"public": "10B"}), [])

    def test_corrupt_rehearsal_record(self):
        with tempfile.TemporaryDirectory(prefix="c10_t10b_") as tmp:
            root = os.path.join(tmp, "p10")
            fake_p10(root)
            ctx = ctx_of(root, tmp)
            b = tb.b_state(t10, ctx)
            write(tb.p1_paths(ctx)["rehearsal"], "{not json")
            self.assertIsNone(tb.tol_state(t10, ctx, b, required=False))
            with self.assertRaises(t10.T10Error) as cm:
                tb.tol_state(t10, dict(ctx, now=tb.TEST_OPEN), b)
            self.assertEqual(cm.exception.code, 3)

    def test_mark_first_commit_matches_the_line(self):
        texts = {"c1": "notes: we hope for **採用** later\n", "c2": "# R\n\n- H10P: B **採用**（x）\n"}
        log = "c1 2026-10-20T00:00:00+08:00\nc2 2026-10-24T00:00:00+08:00\n"

        def run(args, **kw):
            h = args[-1].split(":")[0]
            return mock.Mock(returncode=0, stdout=texts[h].encode("utf-8"))
        with mock.patch.object(tb.t6, "git", return_value=(0, log)), mock.patch("subprocess.run", side_effect=run):
            self.assertEqual(tb.mark_first_commit(REAL["results_md"], "採用"), "2026-10-24T00:00:00+08:00")
            self.assertIsNone(tb.mark_first_commit(REAL["results_md"], "不採用"))


class TolP(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(tb.tol_p(0.0), 0.05)
        self.assertEqual(tb.tol_p(0.019), 0.05)
        self.assertAlmostEqual(tb.tol_p(0.04), 0.1)
        self.assertEqual((tb.TOL_P_FLOOR, tb.TOL_P_FACTOR), (0.05, 2.5))

    def test_rehearsal_record(self):
        with tempfile.TemporaryDirectory(prefix="c10_t10b_") as tmp:
            root = os.path.join(tmp, "p10")
            fake_p10(root)
            ctx = ctx_of(root, tmp)
            b = tb.b_state(t10, ctx)
            with self.assertRaises(t10.T10Error) as cm:
                tb.tol_state(t10, ctx, b)
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("待演練", str(cm.exception))
            late = dict(ctx, now=tb.TEST_OPEN)
            with self.assertRaises(t10.T10Error) as cm:
                tb.tol_state(t10, late, b, cmd="p1 --pool seam_val")
            self.assertEqual(cm.exception.code, 3)                 # B-only after 11/04 11:00
            ev = json.load(open(tb.p1_paths(ctx)["not_used"], encoding="utf-8"))["events"]
            self.assertEqual(ev[-1]["reason"], "no_rehearsal")
            self.assertIsNone(tb.tol_state(t10, ctx, b, required=False))
            rec = {"format": tb.FORMATS["rehearsal"], "mode": "dry-run", "passed": True, "m": 0.03,
                   "tol_P": tb.tol_p(0.03), "scorer": "synthetic",
                   "bound": {"frozen_sha256_lf": b["frozen_sha256_lf"], "val_seam_sha256": b["val_seam_sha256"]}}
            p = tb.p1_paths(ctx)["rehearsal"]
            write(p, rec)
            self.assertAlmostEqual(tb.tol_state(t10, ctx, b)["tol_P"], 0.075)
            for bad in ({"tol_P": 0.05}, {"passed": False}, {"bound": {}}, {"mode": "real"}):
                with self.subTest(bad):
                    write(p, dict(rec, **bad))
                    with self.assertRaises(t10.T10Error):
                        tb.tol_state(t10, ctx, b)


class SeamCompare(unittest.TestCase):
    def setUp(self):
        self.model = F.load_lda(REAL["lda"])
        rng = np.random.default_rng(1)
        self.ref = {"side0": rng.normal(size=(400, 770)).astype(np.float32),
                    "counts0": rng.integers(0, 80, 400), "failed": rng.random(400) < 0.01}

    def test_identical(self):
        r = tb.seam_compare(self.ref["side0"].copy(), self.ref["counts0"].copy(), self.ref["failed"].copy(), self.ref,
                            self.model, 0.05)
        self.assertTrue(r["passed"])
        self.assertEqual(r["max_abs_projected_diff"], 0.0)
        self.assertTrue(r["bitwise_equal_side0"])

    def test_difference_and_tolerance(self):
        x = self.ref["side0"].copy()
        g = int(np.flatnonzero(~self.ref["failed"] & (self.ref["counts0"] > 0))[0])
        x[g] += 1e-3
        r = tb.seam_compare(x, self.ref["counts0"], self.ref["failed"], self.ref, self.model, None)
        m = r["max_abs_projected_diff"]
        self.assertGreater(m, 0)
        za = F.project_p1(x, self.model, [g])
        zb = F.project_p1(self.ref["side0"], self.model, [g])
        self.assertAlmostEqual(m, float(np.max(np.abs(za - zb))), places=12)
        self.assertTrue(tb.seam_compare(x, self.ref["counts0"], self.ref["failed"], self.ref, self.model, m)["passed"])
        r2 = tb.seam_compare(x, self.ref["counts0"], self.ref["failed"], self.ref, self.model, m * 0.999)
        self.assertFalse(r2["passed"])
        self.assertEqual(r2["games_above_tol"], 1)

    def test_counts_or_failed_differ(self):
        c = self.ref["counts0"].copy()
        c[3] += 1
        self.assertFalse(tb.seam_compare(self.ref["side0"], c, self.ref["failed"], self.ref, self.model, 1e9)["passed"])
        f = self.ref["failed"].copy()
        f[5] = not f[5]
        r = tb.seam_compare(self.ref["side0"], self.ref["counts0"], f, self.ref, self.model, 1e9)
        self.assertFalse(r["passed"])
        self.assertIsNone(r["max_abs_projected_diff"])
        r = tb.seam_compare(self.ref["side0"][:399], self.ref["counts0"][:399], self.ref["failed"][:399], self.ref,
                            self.model, 1e9)
        self.assertFalse(r["passed"])


class Synthetic(unittest.TestCase):
    def test_deterministic_and_sides(self):
        sgf = "(;GM[1];B[pd];W[dp];B[pp];W[dd];B[qf])"
        m1, c1, n1 = tb.synthetic_means(sgf, "B")
        m2, c2, n2 = tb.synthetic_means(sgf, "B")
        self.assertTrue(np.array_equal(m1, m2))
        self.assertEqual((list(c1), n1), ([3, 2], 5))
        mw, cw, _ = tb.synthetic_means(sgf, "W")
        self.assertEqual(list(cw), [2, 3])
        self.assertTrue(np.array_equal(mw[0], m1[1]))
        res, _t = tb.synthetic_chunk([{"sgf_content": sgf, "color": "B", "ref": 0},
                                      {"sgf_content": "(;GM[1])", "color": "B", "ref": 1}])
        self.assertEqual(res["counts"].tolist(), [[3, 2], [0, 0]])
        self.assertFalse(res["failed"].any())


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t10b_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = t10.main(argv)
        return code, out.getvalue()

    def test_not_adopted_is_3_and_writes_nothing(self):
        # PREREG_10 rev 5: the repository's B is adopted through PREREG_13, so real mode is no longer 'B not adopted'
        # (the real p1 / p1-rehearse are not run here: they would go on to the test-day checks and the 4090); real
        # status reads it and writes nothing
        code, out = self.run_main(["status"])
        self.assertEqual(code, 0, out)
        self.assertIn("B: adopted via PREREG_13", out)
        self.assertNotIn("B 沒有採用", out)
        self.assertFalse(os.path.exists(tb.REAL_P1_OUT))
        root = os.path.join(self.tmp, "p10")
        fake_p10(root, mark=False, mean=-1.0)
        o10 = os.path.join(self.tmp, "o10")
        dry = ["--dry-run", "--out-dir", o10, "--t06-out-dir", os.path.join(self.tmp, "o6"), "--t06-data-dir",
               os.path.join(self.tmp, "a6"), "--t02-results-dir", os.path.join(self.tmp, "d2"), "--t02-out-dir",
               os.path.join(self.tmp, "o2"), "--tests-dir", os.path.join(self.tmp, "t"), "--prereg10-dir", root]
        for argv in (["p1", "--pool", "public"] + dry, ["p1-rehearse"] + dry):
            code, out = self.run_main(argv)
            self.assertEqual(code, 3, out)
        self.assertEqual(sorted(os.listdir(self.tmp)), ["p10"])

    def test_real_mode_b_dirs(self):
        for k in ("--prereg10-dir", "--p1-out-dir", "--p1-data-dir"):
            with self.subTest(k):
                code, out = self.run_main(["status", k, self.tmp])
                self.assertEqual(code, 2)
                self.assertIn("real mode", out)

    def test_dry_run_dirs_outside_repository(self):
        o10 = os.path.join(self.tmp, "o10")
        code, out = self.run_main(["status", "--dry-run", "--out-dir", o10, "--t06-out-dir", os.path.join(self.tmp, "o6"),
                                   "--t06-data-dir", os.path.join(self.tmp, "a6"), "--p1-out-dir",
                                   os.path.join(REPO, "results", "test10", "p1x")])
        self.assertEqual(code, 2)
        self.assertIn("inside the repository", out)

    def test_scorer_flags_dry_run_only(self):
        ctx = {"real": True}
        for flag in ("fake_scorer", "synthetic_p1"):
            a = mock.Mock(fake_scorer=flag == "fake_scorer", synthetic_p1=flag == "synthetic_p1")
            with self.assertRaises(t10.T10Error):
                tb.scorer_kind(t10, ctx, a)
        self.assertEqual(tb.scorer_kind(t10, {"real": False}, mock.Mock(fake_scorer=False, synthetic_p1=True)),
                         "synthetic")

    def test_constants(self):
        self.assertEqual(tb.K6B_10, t10.K6B_10)
        self.assertEqual(tb.K5_10, t10.K5_10)
        self.assertEqual(tb.K6B_10, ol.K6B_10)
        self.assertEqual(tb.B_REV_ROW, ol.P10_B_ROW)
        self.assertEqual(tb.FAIL_MAX, ol.T10_B_FAIL_MAX)
        self.assertEqual(tb.TEST_OPEN.isoformat(), "2026-11-04T11:00:00+08:00")
        self.assertEqual(set(ol.T10_SOURCES_B), set(t10.SOURCES))
        self.assertEqual(tb.REAL_P1_OUT, os.path.join(REPO, "results", "test10", "p1"))
        self.assertEqual(ol.T10_P1_DIR, "results/test10/p1")
        # PREREG_10 rev 5
        self.assertEqual(tb.K5_13, ol.K5_13)
        self.assertEqual(tb.B13_REV_ROW, ol.P10_B13_ROW)
        self.assertEqual(tb.REAL_P13, os.path.join(REPO, "results", "prereg13"))
        Q = tb.p13_paths(tb.REAL_P13)
        self.assertEqual(set(ol.T10_B13_INPUTS_REQUIRED),
                         {os.path.relpath(Q[k], REPO).replace(os.sep, "/") for k in ("results_md13", "h13p", "view")})
        self.assertEqual(ol.T10_B_H10P_FILE, os.path.relpath(REAL["h10p"], REPO).replace(os.sep, "/"))
        self.assertIn(ol.T10_B_H10P_FILE, ol.T10_B_INPUTS_REQUIRED)


def report_b(sources=("06-OFF", "10B")):
    src = dict(zip(("public", "private"), sources))
    pools = {p: {"b": ({"applied": True, "pool_gate": {"passed": True}, "failed_fraction": 0.001} if s == "10B"
                       else {"applied": False})} for p, s in src.items()}
    bound = {r: "x" for r in ol.T10_B_INPUTS_REQUIRED}
    bound.update({f"results/test10/p1/{p}_p1_report.json": "x" for p, s in src.items() if s == "10B"})
    return {"sources": src, "pools": pools, "inputs_bound": bound,
            "b": {"adopted": True, "w_star": 16.0, "gpu_seam": {"passed": True}, "tol": {"tol_P": 0.05}}}


class OfficialLogB(unittest.TestCase):
    def patches(self, commit_t="2026-11-10T00:00:00+08:00", row_t="2026-10-20T00:00:00+08:00"):
        return [mock.patch.object(ol, "git_commit_time", return_value=("c" * 40, commit_t)),
                mock.patch.object(ol, "git_row_added_on_main", return_value=("r" * 40, row_t)),
                mock.patch.object(ol, "git_head_text", return_value="| rev 3 | x |\n| rev 4 | y |\n")]

    def run_b(self, rep, **kw):
        with contextlib.ExitStack() as st:
            for p in self.patches(**kw):
                st.enter_context(p)
            return ol.t10_b_problems(rep, "R")

    def test_complete_b_report(self):
        self.assertEqual(self.run_b(report_b()), [])
        self.assertEqual(self.run_b(report_b(("10B", "10B"))), [])
        self.assertEqual(self.run_b(report_b(("06-OFF", "10A"))), [])

    def test_b_rules(self):
        rep = report_b()
        rep["pools"]["private"]["b"]["failed_fraction"] = 0.0051
        self.assertTrue(any("0.5 %" in x for x in self.run_b(rep)))
        rep = report_b()
        rep["pools"]["private"]["b"]["pool_gate"]["passed"] = False
        self.assertTrue(any("pool gate" in x for x in self.run_b(rep)))
        rep = report_b()
        rep["b"]["gpu_seam"]["passed"] = False
        self.assertTrue(any("GPU seam" in x for x in self.run_b(rep)))
        rep = report_b()
        del rep["inputs_bound"]["results/test10/p1/private_p1_report.json"]
        self.assertTrue(any("lacks the B file" in x for x in self.run_b(rep)))
        self.assertTrue(any("K6B-10" in x for x in self.run_b(report_b(), commit_t="2026-11-15T00:00:00+08:00")))
        self.assertTrue(any("K5-10 (B)" in x for x in self.run_b(report_b(), row_t="2026-11-01T00:00:00+08:00")))
        rep = report_b()
        rep["sources"]["public"] = "10C"
        self.assertTrue(any("pool sources" in x for x in self.run_b(rep)))

    def test_adopted_without_10b_is_an_a_report(self):
        """Review B1: B adopted but not used in either pool -> only the rev 3 rules (no B files, no K6B-10 needed)."""
        rep = {"sources": {"public": "06-OFF", "private": "10A"}, "b": {"adopted": True}, "inputs_bound": {}}
        with mock.patch.object(ol, "git_commit_time", return_value=(None, None)), \
                mock.patch.object(ol, "git_row_added_on_main", return_value=(None, None)):
            got = ol.t10_report_problems(rep, "R", None, "s")
        want = ol._t10_report_problems_rev3(dict(rep, b={"adopted": False}), "R", None, "s")
        self.assertEqual(got, want)
        self.assertFalse(any("K6B" in x or "lacks the B file" in x or "K5-10 (B)" in x for x in got))

    def test_k5_b_uses_main_first_parent(self):
        with mock.patch.object(ol, "git", return_value=(0, "abc 2026-10-30T10:00:00+08:00\n")) as g, \
                mock.patch.object(ol, "git_head_text", return_value="| rev 4 | x |\n"):
            st = ol.t10_k5_b_state()
        self.assertTrue(st["ok"])
        args = g.call_args[0]
        self.assertIn("main", args)
        self.assertIn("--first-parent", args)

    def test_wrapper(self):
        rep = {"sources": {"public": "10A", "private": "test02"}, "b": {"adopted": False}}
        self.assertEqual(ol.t10_report_problems(rep, "R", None, "s"),
                         ol._t10_report_problems_rev3(rep, "R", None, "s"))
        rb = report_b(("06-OFF", "10B"))
        with contextlib.ExitStack() as st:
            for p in self.patches():
                st.enter_context(p)
            got = ol.t10_report_problems(copy.deepcopy(rb), "R", None, "s")
        self.assertFalse(any("no pool was lifted" in x for x in got))       # 10B counts as a lifted pool
        self.assertFalse(any("this kit has part A only" in x for x in got))
        self.assertFalse(any("lacks the B file" in x for x in got))
        rb["b"]["adopted"] = False
        with contextlib.ExitStack() as st:
            for p in self.patches():
                st.enter_context(p)
            got = ol.t10_report_problems(rb, "R", None, "s")
        self.assertTrue(any("10B needs PREREG_10 B adopted" in x for x in got))   # 10B without B: refused




# ================================================================ PREREG_10 rev 5 (section 16): B adopted through PREREG_13
def real_ctx():
    return {"real": True, "mode": "real", "P10": tb.REAL_P10, "O10P1": tb.REAL_P1_OUT, "A10P1": tb.REAL_P1_DATA,
            "write_roots": [], "now": tb.TEST_OPEN.replace(day=1)}


def fake_p13(tmp, kind="adopted"):
    """A minimal PREREG_10 tree (fake_p10) turned by tools/t10_b_fake.py fake-p13 into 'H10P never ran, B adopted through
    PREREG_13': -> (q10, its p13 folder)."""
    src = os.path.join(tmp, "src10")
    if not os.path.exists(src):
        fake_p10(src)
    q10 = os.path.join(tmp, "q10")
    with contextlib.redirect_stdout(io.StringIO()):
        tbf.cmd_fake_p13(mock.Mock(p10=src, out=q10, kind=kind))
    return q10, q10 + "_p13"


class BState13(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t10b13_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def state(self, q10):
        return tb.b_state(t10, ctx_of(q10, self.tmp))

    def edit(self, path, fn):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        fn(obj)
        os.remove(path)                                         # never through a hard link to the source tree
        write(path, obj)

    def test_adopted(self):
        q10, d13 = fake_p13(self.tmp)
        self.assertEqual(tb.p13_dir(ctx_of(q10, self.tmp)), d13)
        b = self.state(q10)
        self.assertEqual((b["adopted"], b["source"]), (True, "PREREG_13"), b.get("problems"))
        self.assertEqual(b["h13p"]["n_view"], 263)
        self.assertAlmostEqual(b["h13p"]["ci95_low_must_exceed"], -0.006575, places=15)
        self.assertEqual(len(b["files_sha256"]), 13)
        self.assertNotIn(tb.p10_paths(q10)["h10p"], b["files_sha256"])
        rec = tb.b_record(b)
        self.assertEqual(rec["source"], "PREREG_13")
        self.assertIn("h13p", rec)
        self.assertNotIn("h10p", rec)
        got = tb.b_inputs(ctx_of(q10, self.tmp), b, {"public": "06-OFF", "private": "10B"})
        for k in ("results_md13", "h13p", "view", "audit"):
            self.assertIn(tb.p13_paths(d13)[k], got)
        self.assertNotIn(tb.p10_paths(q10)["h10p"], got)
        self.assertEqual(tb.b_inputs(ctx_of(q10, self.tmp), b, {"public": "06-OFF", "private": "10A"}), [])
        self.assertIn("adopted via PREREG_13", tb.status_lines(t10, ctx_of(q10, self.tmp), b)[0])
        src = tb.p10_paths(os.path.join(self.tmp, "src10"))         # the fake copy left the source tree alone
        self.assertTrue(os.path.exists(src["h10p"]))
        self.assertTrue(json.load(open(src["apply"], encoding="utf-8"))["score_allowed"])
        self.assertTrue(tb.b_state(t10, ctx_of(os.path.dirname(src["results_md"]), self.tmp))["adopted"])

    def test_kinds(self):
        for kind, reason in (("not_adopted", "h13p_not_adopted"), ("inconsistent", "inconsistent"),
                             ("audit_unbound", "inconsistent")):
            with self.subTest(kind):
                q10, _d = fake_p13(self.tmp, kind)
                b = self.state(q10)
                self.assertEqual((b["adopted"], b["reason"]), (False, reason), b)
        self.assertIn("h13p_not_adopted", tb.SILENT)
        q10, _d = fake_p13(self.tmp, "audit_unbound")
        self.assertTrue(any("view_report_sha256_at_audit" in x for x in self.state(q10)["problems"]))

    def test_not_adopted_writes_nothing(self):
        q10, _d = fake_p13(self.tmp, "not_adopted")
        ctx = ctx_of(q10, self.tmp)
        with self.assertRaises(t10.T10Error) as cm:
            tb.require_adopted(t10, ctx, tb.b_state(t10, ctx), "p1")
        self.assertEqual(cm.exception.code, 3)
        self.assertFalse(os.path.exists(tb.p1_paths(ctx)["not_used"]))

    def test_inconsistent_files(self):
        def spoil(name, Q, P):
            if name == "score_allowed":
                self.edit(Q["view"], lambda o: o.update(score_allowed="pending_audit"))
            elif name == "gate":
                self.edit(Q["view"], lambda o: o["gates"]["V3"].update(passed=False))
            elif name == "w_star":
                self.edit(Q["view"], lambda o: o.update(w_star=8.0))
            elif name == "top5":
                write(Q["view_b"], "question_id,top_1\nq,x\n")
            elif name == "lda_input":
                self.edit(Q["view"], lambda o: o["inputs_sha256"].update({"p1_lda.npz": "0" * 64}))
            elif name == "audit_sha":
                self.edit(Q["audit"], lambda o: o.update(extra=1))
            elif name == "audit_errors":
                self.edit(Q["audit"], lambda o: o.update(program_errors_affecting_view=["x"]))
            elif name == "p10_top5":
                os.remove(P["val_top5"])
                write(P["val_top5"], "question_id,top_1\nq,y\n")
            elif name == "two_marks":
                write(Q["results_md13"], tb.B13_ADOPTED_MARK + "\n- H13P: B′ **不採用**\n")
            elif name == "no_mark":
                write(Q["results_md13"], "# nothing\n")
            elif name == "missing_audit":
                os.remove(Q["audit"])
            elif name == "n_view":
                self.edit(Q["view"], lambda o: o["view"].update(n_view=300))   # bound -0.0075: the low -0.004 still passes
                self.edit(Q["h13p"], lambda o: o["B_minus_A"].update(ci95=[-0.0070, 0.006]))
                self.edit(Q["view"], lambda o: o["view"].update(n_view=263))   # -0.0070 < -0.006575: not adopted
        names = ("score_allowed", "gate", "w_star", "top5", "lda_input", "audit_sha", "audit_errors", "p10_top5",
                 "two_marks", "no_mark", "missing_audit", "n_view")
        for name in names:
            with self.subTest(name):
                q10, d13 = fake_p13(self.tmp)
                spoil(name, tb.p13_paths(d13), tb.p10_paths(q10))
                b = self.state(q10)
                self.assertEqual((b["adopted"], b["reason"]), (False, "inconsistent"), b)
                self.assertTrue(b["problems"])

    def test_mark_without_score_and_nothing(self):
        q10, d13 = fake_p13(self.tmp)
        os.remove(tb.p13_paths(d13)["h13p"])
        self.assertEqual(self.state(q10)["reason"], "inconsistent")
        shutil.rmtree(d13)
        self.assertEqual(self.state(q10)["reason"], "not_marked")

    def test_prereg10_path_first(self):
        """16.1: PREREG_13 is asked only when the PREREG_10 path is not_marked."""
        q10, d13 = fake_p13(self.tmp)
        shutil.rmtree(q10)
        fake_p10(q10)                                           # H10P adopted: source PREREG_10
        b = self.state(q10)
        self.assertEqual((b["adopted"], b["source"]), (True, "PREREG_10"))
        self.assertEqual(len(b["files_sha256"]), 8)
        self.assertNotIn("p13_paths", b)
        shutil.rmtree(q10)
        fake_p10(q10, mark=False, mean=-1.0)                    # H10P not adopted: no fall-through to PREREG_13
        self.assertEqual(self.state(q10)["reason"], "h10p_not_adopted")
        shutil.rmtree(q10)
        fake_p10(q10, mean=-1.0)                                # H10P inconsistent: no fall-through either
        self.assertEqual(self.state(q10)["reason"], "inconsistent")
        self.assertTrue(os.path.isdir(d13))

    def test_h13p_rule(self):
        for point, low, n, want in ((0.0, -0.0065, 263, True), (0.0, -0.006575, 263, False), (-1e-9, 0.0, 263, False),
                                    (0.0, -0.0099, 400, True), (0.0, -0.0001, None, False), (0.0, -0.0001, 0, False),
                                    (0.0, -0.0001, 401, False), (0.0, -0.0001, True, False)):
            with self.subTest(point=point, low=low, n=n):
                ok, _r = tb.h13p_decision({"B_minus_A": {"mean_diff": point, "ci95": [low, 1]}}, n)
                self.assertEqual(ok, want)

    def test_mark13_variants(self):
        path = os.path.join(self.tmp, "R.md")
        for line, want in (("- H13P: B′ **採用**（x）", ["採用"]), ("H13P：B′ **不採用**", ["不採用"]),
                           ("H13P：B **採用**", []), ("H10P：B **採用**", [])):
            with self.subTest(line):
                write(path, line + "\n")
                self.assertEqual(tb.results_md_marks(path, tb.MARK13_RE), want)
        write(path, tb.B13_ADOPTED_MARK + "\n")
        self.assertEqual(tb.results_md_marks(path, tb.MARK13_RE), ["採用"])
        self.assertEqual(tb.results_md_marks(path), [])                    # the H10P regex ignores it

    def test_k3_13(self):
        q10, _d = fake_p13(self.tmp)
        ctx = dict(ctx_of(q10, self.tmp), real=True)
        late, early = "2026-10-26T00:00:00+08:00", "2026-10-25T23:59:59+08:00"
        for h, m, want in ((late, early, "k3_13"), (early, late, "k3_13"), (early, None, "k3_13")):
            with self.subTest(h=h, m=m), mock.patch.object(tb.t6, "committed_clean", return_value=True), \
                    mock.patch.object(tb.t6, "git_commit_time", return_value=h), \
                    mock.patch.object(tb, "mark_first_commit", return_value=m):
                b = tb.b_state(t10, ctx)
                self.assertEqual((b["adopted"], b["reason"]), (False, want))
        self.assertEqual(tb.K3_13.isoformat(), "2026-10-25T23:59:59+08:00")
        self.assertEqual(tb.K5_13.isoformat(), "2026-10-31T23:59:59+08:00")

    def test_audit_binding_at_the_audit_commit(self):
        """The real-mode binding reads view_report.json at the audit's commit (before apply --finalize), not the
        current file (results/prereg13/RESULTS.md 已知事項); review M1: the current report may differ from that one
        only in the keys apply --finalize writes; review m6: real mode refuses 'fake' records."""
        import hashlib
        q10, d13 = fake_p13(self.tmp)
        Q, P = tb.p13_paths(d13), tb.p10_paths(q10)
        self.edit(Q["audit"], lambda o: o.pop("fake"))
        with open(Q["view"], encoding="utf-8") as f:
            view = json.load(f)
        view.pop("fake")
        pre_obj = {k: v for k, v in view.items() if k not in tb.VIEW_FINALIZE_KEYS}
        pre_obj.update(score_allowed="pending_audit", score_blocked_reasons=["pending_audit"])
        pre = (json.dumps(pre_obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
        sha = hashlib.sha256(pre).hexdigest()
        self.edit(Q["audit"], lambda o: o["outputs_sha256"].update({"view_report.json": sha}))
        view["audit"].update(view_report_sha256_at_audit=sha, sha256=tb.t6.sha_lf(Q["audit"]))
        os.remove(Q["view"])
        write(Q["view"], view)
        with open(P["frozen"], encoding="utf-8") as f:
            fc = json.load(f)
        with open(Q["view"], "rb") as f:
            current = f.read()
        ctx = dict(ctx_of(q10, self.tmp), real=True)
        self.assertEqual(tb.view_problems(dict(ctx, real=False), P, Q, view, fc), [])
        with mock.patch.object(tb.t6, "git", return_value=(0, "a" * 40)), \
                mock.patch.object(tb.t6, "rel_key", side_effect=os.path.basename):
            with mock.patch.object(tb, "git_show_bytes", return_value=pre.replace(b"\n", b"\r\n")):
                self.assertEqual(tb.view_problems(ctx, P, Q, view, fc, {"B_minus_A": {}}), [])
                for key, val in (("view", {"n_view": 300}), ("w_star", 8.0), ("selected", "V3"),
                                 ("gates", {"V1": {"passed": True}, "V2": {"passed": True}, "V3": {"passed": True},
                                            "x": 1})):
                    with self.subTest(changed=key):
                        got = tb.view_problems(ctx, P, Q, dict(view, **{key: val}), fc)
                        self.assertTrue(any("differs from the audited one" in x and key in x for x in got), got)
                got = tb.view_problems(ctx, P, Q, view, fc, {"fake": True})
                self.assertTrue(any("score_H13P.json carries a 'fake' key" in x for x in got))
                got = tb.view_problems(ctx, P, Q, dict(view, fake=True), fc)
                self.assertTrue(any("view_report.json carries a 'fake' key" in x for x in got))
            for blob in (current, None):
                with mock.patch.object(tb, "git_show_bytes", return_value=blob):
                    self.assertTrue(any("at the audit commit" in x for x in tb.view_problems(ctx, P, Q, view, fc)))

    def test_dry_run_never_sees_the_real_prereg13(self):
        """Review m2: the PREREG_13 folder follows the mode, not the path."""
        ctx = dict(real_ctx(), real=False, mode="dry-run")
        self.assertEqual(tb.p13_dir(ctx), os.path.abspath(tb.REAL_P10) + "_p13")
        self.assertEqual(tb.p13_dir(real_ctx()), tb.REAL_P13)
        self.assertEqual(tb.b_state(t10, ctx)["reason"], "not_marked")

    def test_k5_record_never_raises_and_wording(self):
        """Review m3 / m4: k5_b_state(b=None) never raises; K5-13 has its own reason."""
        def boom(T, ctx):
            raise RuntimeError("x")
        with mock.patch.object(tb, "b_state", boom), \
                mock.patch.object(tb, "row_on_main", lambda row, d: {"ok": True, "row": row}):
            st = tb.k5_b_state(t10, real_ctx())
        self.assertFalse(st["ok"])
        self.assertIn("RuntimeError", st["error"])
        self.assertEqual(tb.k5_reason(st)[0], "k5_10_b")
        self.assertEqual(tb.k5_reason({"ok": False, "rev4_ok": True, "prereg13": {"ok": False}})[0], "k5_13")
        self.assertEqual(tb.k5_reason({"ok": False, "rev4_ok": False, "prereg13": {"ok": True}})[0], "k5_10_b")
        self.assertEqual(tb.k5_reason({"ok": False, "rev4_ok": False})[0], "k5_10_b")

    def test_fake_p13_refuses_its_source_and_the_repository(self):
        """Review m7."""
        src = os.path.join(self.tmp, "src10")
        fake_p10(src)
        for p10, out in ((src, src), (tb.REAL_P10, os.path.join(self.tmp, "q"))):
            with self.subTest(p10=p10), self.assertRaises(SystemExit):
                tbf.cmd_fake_p13(mock.Mock(p10=p10, out=out, kind="adopted"))
        self.assertTrue(os.path.exists(tb.p10_paths(src)["h10p"]))

    def test_real_audit_binding_uses_the_audit_commit(self):
        """In the repository: the current (finalized) view_report.json is NOT the audited one, the one at the audit's
        commit is."""
        Q = tb.p13_paths(tb.REAL_P13)
        with open(Q["view"], encoding="utf-8") as f:
            view = json.load(f)
        with open(Q["audit"], encoding="utf-8") as f:
            audit = json.load(f)
        at = view["audit"]["view_report_sha256_at_audit"]
        self.assertEqual(at, audit["outputs_sha256"]["view_report.json"])
        self.assertNotEqual(tb.t6.sha_lf(Q["view"]), at)
        P = tb.p10_paths(tb.REAL_P10)
        with open(P["frozen"], encoding="utf-8") as f:
            fc = json.load(f)
        self.assertEqual(tb.view_problems(real_ctx(), P, Q, view, fc), [])

    def test_ledger13(self):
        view = {"outputs_sha256": {tb.P13_OUT_A: "a" * 64, tb.P13_OUT_B: "b" * 64}}
        score = {"A_mean": 0.9, "B_mean": 0.95}
        head = "utc_time,git_head,note,command,pred_sha256,label_sha256,n,mean,rank_balanced_mean,details\n"

        def row(sha, mean, note="prereg13 H13P (x)"):
            return f"t,h,{note},compare-player,{sha},{'c' * 64},400,{mean},,{{}}\n"
        led = os.path.join(self.tmp, "ledger.csv")
        cases = (([row("a" * 64, 0.9), row("b" * 64, 0.95)], True),
                 ([row("x", 0.8), row("y", 0.85), row("a" * 64, 0.9), row("b" * 64, 0.95)], True),
                 ([row("a" * 64, 0.9)], False), ([row("b" * 64, 0.95), row("a" * 64, 0.9)], False),
                 ([row("a" * 64, 0.9), row("b" * 64, 0.951)], False), ([], False),
                 ([row("a" * 64, 0.9), row("b" * 64, 0.95), row("z", 1, "prereg10 H10P")], True))
        for rows, ok in cases:
            with self.subTest(rows=len(rows), ok=ok):
                write(led, head + "".join(rows))
                with mock.patch.object(tb, "LEDGER", led):
                    self.assertEqual(tb.ledger13_problems(view, score) == [], ok)

    def test_k5_needs_rev5_on_the_prereg13_path(self):
        def rows(r4, r5):
            return lambda row, deadline: {"ok": r4 if row == tb.B_REV_ROW else r5, "row": row}
        ctx = real_ctx()
        for src, r4, r5, want in (("PREREG_13", True, True, True), ("PREREG_13", True, False, False),
                                  ("PREREG_13", False, True, False), ("PREREG_10", True, False, True)):
            with self.subTest(src=src, r4=r4, r5=r5), mock.patch.object(tb, "row_on_main", rows(r4, r5)):
                st = tb.k5_b_state(t10, ctx, {"adopted": True, "source": src})
                self.assertEqual(st["ok"], want)
                self.assertEqual("prereg13" in st, src == "PREREG_13")
        self.assertEqual(tb.k5_b_state(t10, dict(ctx, real=False))["checked"], False)


def report_b13(sources=("06-OFF", "10B")):
    rep = report_b(sources)
    rep["b"]["source"] = "PREREG_13"
    del rep["inputs_bound"][ol.T10_B_H10P_FILE]
    rep["inputs_bound"].update({r: "x" for r in ol.T10_B13_INPUTS_REQUIRED})
    return rep


HEAD45 = "| rev 3 | x |\n| rev 4 | y |\n| rev 5 | z |\n"


class OfficialLogB13(unittest.TestCase):
    """PREREG_10 rev 5 (16.4): the rev 5 wrapper of t10_report_problems."""

    def patches(self, row5_t="2026-10-20T00:00:00+08:00", head=HEAD45):
        def row_added(rel, row):
            return ("r" * 40, row5_t if row == ol.P10_B13_ROW else "2026-10-12T00:00:00+08:00")
        return [mock.patch.object(ol, "git_commit_time", return_value=("c" * 40, "2026-11-10T00:00:00+08:00")),
                mock.patch.object(ol, "git_row_added_on_main", side_effect=row_added),
                mock.patch.object(ol, "git_head_text", return_value=head)]

    def run_wrapper(self, rep, fn=None, **kw):
        with contextlib.ExitStack() as st:
            for p in self.patches(**kw):
                st.enter_context(p)
            return (fn or ol.t10_report_problems)(copy.deepcopy(rep), "R", None, "s")

    def b_only(self, probs):
        """The B problems (the rev 3 part complains about the stand-in report's other fields)."""
        return [x for x in probs if "B file" in x or "K5-1" in x or "K6B" in x or "GPU seam" in x or "pool gate" in x
                or "0.5 %" in x or "10B" in x]

    def test_prereg13_report(self):
        self.assertEqual(self.b_only(self.run_wrapper(report_b13())), [])
        self.assertEqual(self.b_only(self.run_wrapper(report_b13(("10B", "10B")))), [])
        rep = report_b13()
        rep["inputs_bound"][ol.T10_B_H10P_FILE] = "x"                  # an extra bound file is no problem
        self.assertEqual(self.b_only(self.run_wrapper(rep)), [])

    def test_prereg13_rules(self):
        rep = report_b13()
        del rep["inputs_bound"]["results/prereg13/player/view_report.json"]
        self.assertTrue(any("view_report.json" in x and "PREREG_13" in x for x in self.run_wrapper(rep)))
        self.assertTrue(any("K5-13" in x for x in self.run_wrapper(report_b13(), row5_t="2026-11-01T00:00:00+08:00")))
        self.assertTrue(any("K5-13" in x for x in self.run_wrapper(report_b13(), head="| rev 3 | x |\n| rev 4 | y |\n")))
        rep = report_b13()
        for r in ol.T10_B13_INPUTS_REQUIRED:
            del rep["inputs_bound"][r]
        self.assertEqual(len([x for x in self.run_wrapper(rep) if "B adopted through PREREG_13" in x]), 3)
        rep = report_b13()
        rep["pools"]["private"]["b"]["pool_gate"]["passed"] = False      # the rev 4 B rules still apply
        self.assertTrue(any("pool gate" in x for x in self.run_wrapper(rep)))

    def test_other_reports_unchanged(self):
        """A report without source PREREG_13, or without a 10B pool, gets the rev 4 result exactly."""
        reps = (report_b(), report_b(("06-OFF", "10A")), report_b13(("06-OFF", "10A")),
                {"sources": {"public": "10A", "private": "test02"}, "b": {"adopted": False}})
        for rep in reps:
            with self.subTest(rep.get("sources")):
                self.assertEqual(self.run_wrapper(rep), self.run_wrapper(rep, fn=ol._t10_report_problems_rev4))
        rb = report_b()                                                # a PREREG_10 report lacking score_H10P: flagged
        del rb["inputs_bound"][ol.T10_B_H10P_FILE]
        self.assertTrue(any(ol.T10_B_H10P_FILE in x for x in self.run_wrapper(rb)))
