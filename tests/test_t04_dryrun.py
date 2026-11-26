"""End-to-end dry run of the 04 test-day kit on the from-val fake tree (results/test04/design/spec.md 8.1 point 2,
8.2, 8.3, 10.5). Every tool is called through its command line, as on test day.

    C10_SLOW=1 or 2   one tree S (a temporary directory outside the repository, or C10_DRYRUN_DIR/t04dry):
        R20 part (spec 8.1 point 2): t02_fake_tests.py from-val -> t02_checks.py t1 --only T1.1,T1.2,T1.3 (rank)
            -> t02_rank.py opening / apply --dry-run into S/out/rank. Every deterministic output equals
            results/test04/design/r20_fake_golden.json (recorded before any t04 code existed); decide_summary.json
            is compared without its 'input' key (it holds the absolute path of the tree).
        04 part (spec 8.2): t04_fake_tests.py from-val -> t04_rank.py unpack / opp --workers 6 / apply / status, all
            --dry-run. seam_val reproduces prereg04 byte for byte (o4_seam_val.npz, rank_val_probs.csv, pred_B.csv,
            pred_A.csv); the 800 test rows equal prereg04's first 800 validation rows under the fake id prefixes;
            the submission passes check_submission --official; the dkappa values; t04 never changes a file of
            test02 (S/results/test02, S/out/rank, S/data/test are snapshotted around every t04 step).
        Negative drills (spec 8.3): a copy of the notebook zip or of the unpacked tree with one change each, and the
            archive / rerun sequence (t04_rank.py archive also runs outside the real mode on a --results-dir; its
            zips go to R4/failed_zips/). Only the R20-first-upload time check is real-mode only (spec 8.3 says so):
            it belongs to tests/test_t04_rank.py and is skipped here with that reason.
    The 04 part is skipped (with a message) while tools/t04_rank.py or tools/t04_fake_tests.py does not exist yet;
    the R20 part runs regardless.

The fake tree holds validation games: only the replay and the apply pipeline run on it, never a content statistic
(T1.4-T5, dupcheck). Nothing here opens a label file or runs tools/score.py; nothing is written inside the
repository. The replay uses 6 worker processes with one BLAS thread each.
Run: C10_SLOW=1 python -m pytest tests/test_t04_dryrun.py   (C10_KEEP=1 keeps the tree; C10_DRYRUN_DIR=<outside>)
"""

import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOOLS = os.path.join(REPO, "tools")
sys.path.insert(0, TOOLS)
import check_submission  # noqa: E402

SLOW = os.environ.get("C10_SLOW", "")
GOLDEN_PATH = os.path.join(REPO, "results", "test04", "design", "r20_fake_golden.json")
P04 = os.path.join(REPO, "results", "prereg04")
T04_TOOLS = ("t04_rank.py", "t04_fake_tests.py")

# committed prereg04 references (spec 3.3, 5.1, 8.2)
O4_SEAM_SHA = "30d312c02c357f6761567099003885be5e08635b6e6c0b49d71baac137cd9cd4"
Q02_VAL_SHA = {"rank_val_probs.csv": "3c7ab658dc558ce8c6aefdd7b0747de1a60de575d151e9eee96ca6775f442e5c",
               "pred_B.csv": "27b6d225d0afc8db9183ae970ee19946ff819ec5ccb2be6a1816ef4338efff79",
               "pred_A.csv": "6a6bb3c2e63761bc7196661c99999cec499fae1b38bbb3a7bb9d512e26487739"}
DKAPPA_SEAM_VAL = {"O4": 0.1050265798301427, "SWA": 0.03631175839656169, "SWB": 0.034356617152967346}
# the U1 probe (understand_t02.md 3.3), rounded to 4 places; spec 8.2 pins the full values of the first green run
DKAPPA_FAKE_ROUNDED = {"public": {"O4": 0.0930, "SWA": 0.0321, "SWB": 0.0278},
                       "private": {"O4": 0.1000, "SWA": 0.0418, "SWB": 0.0364}}
DKAPPA_FAKE_FULL = {         # the first green run, 2026-09-30 04:42 (branch p04-testday); same bits again at 04:50
    "public": {"O4": 0.09297820220912692, "SWA": 0.03212728346184879, "SWB": 0.02784129798577018},
    "private": {"O4": 0.09999333037362917, "SWA": 0.04181334094598075, "SWB": 0.0363944712752194}}
O4_ROWS = {"public": (0, 4787), "private": (4787, 9602)}          # slots of validation questions 1-400, 401-800
RANK_ROWS = {"public": (0, 400), "private": (400, 800)}
PREFIX = {"public": "pub_", "private": "priv_"}
RANK_FILES = {"public": "rank_prediction_test_public.csv", "private": "rank_prediction_test_private.csv"}
SETS = ("seam_val", "public", "private")
APPLY_REPORT_FIELDS = ("format", "mode", "combo", "notebook", "seam_local", "replay", "colour_check", "usable",
                       "unusable_reasons", "check_submission", "outputs_sha256", "test02", "inputs_bound", "attempt")
TEST_OUTPUTS = ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv")
LATER = "2099-01-01T00:00:00+08:00"          # a notebook run after any archive of this dry run
GUARDED_REPO_PATHS = ("results/test02", "results/test04/sw", "results/test04/opp", "results/test04/rank",
                      "results/test04/failed", "results/test04/unpack_sw.json", "results/official_scores.json",
                      "results/test04_rehearsal", "results/prereg01", "results/prereg02", "results/prereg03",
                      "results/prereg04")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def lf_lines(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n").split(b"\n")


def csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def inside(path, root):
    p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def run_tool(args, expect=0, log=None, env_extra=None):
    """Run PY tools/<args> from the repository root; -> (exit code, stdout+stderr)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("PYTHONUTF8", None)
    env.update(env_extra or {})
    t0 = time.time()
    r = subprocess.run([sys.executable] + list(args), cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    out = r.stdout + r.stderr
    if log is not None:
        log.append({"argv": [os.path.basename(args[0])] + [os.path.basename(a) if os.path.isabs(a) else a
                                                           for a in args[1:]],
                    "exit": r.returncode, "seconds": round(time.time() - t0, 1), "tail": out[-800:]})
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


def snapshot(*roots):
    """{relative path: sha256} of every file under the roots (the file list is part of the snapshot)."""
    out = {}
    for root in roots:
        for d, _dirs, files in os.walk(root):
            for f in files:
                p = os.path.join(d, f)
                out[os.path.relpath(p, os.path.dirname(root))] = sha(p)
    return out


def guarded_repo_state():
    r = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=all", "--ignored", "--"]
                       + list(GUARDED_REPO_PATHS), capture_output=True, timeout=120)
    return r.stdout.decode("utf-8", "replace")


def rezip(src, dst, edits):
    """Copy the zip src to dst; edits {member: fn(bytes) -> bytes} change single members, everything else is kept."""
    seen = set()
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w") as zo:
        for info in zi.infolist():
            data = zi.read(info)
            if info.filename in edits:
                data = edits[info.filename](data)
                seen.add(info.filename)
            zo.writestr(info, data)
    missing = set(edits) - seen
    assert not missing, f"members not in the zip: {missing}"
    return dst


def edit_json_bytes(fn):
    def f(data):
        obj = json.loads(data.decode("utf-8"))
        fn(obj)
        return json.dumps(obj, indent=1, ensure_ascii=False).encode("utf-8")
    return f


def npz_edited(data, fn):
    """The bytes of an npz member after fn(arrays) (same member names, np.savez uncompressed)."""
    import io
    import numpy as np
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    fn(arrays)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def flip_first_parsed(arrays):
    import numpy as np
    p = arrays["parsed"].copy()
    p[int(np.flatnonzero(p)[0])] = False
    arrays["parsed"] = p


def rewrite_npz(path, fn):
    """Rewrite an npz with the same member names (np.savez, uncompressed, like 03's write_dump); fn edits arrays."""
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    fn(arrays)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def refresh_marker(dumps, stem):
    """_DONE_<stem>.json follows the edited npz (sha256, bytes), so only the intended check can fail."""
    mp = os.path.join(dumps, f"_DONE_{stem}.json")
    m = read_json(mp)
    p = os.path.join(dumps, stem + ".npz")
    m.update(sha256=sha(p), bytes=os.path.getsize(p))
    write_json(mp, m)


def dk(v):
    """A dkappa value from a number or a {'dkappa': ...} record."""
    return v.get("dkappa") if isinstance(v, dict) else v


# ================================================================ the tree (built once per session)
class Tree:
    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            cls.built = cls()
        return cls.built

    def __init__(self):
        self.S, self.cleanup = workdir("t04dry")
        S = self.S
        self.log = []
        self.repo_state_before = guarded_repo_state()
        self.tests = os.path.join(S, "data", "test")
        self.D2 = os.path.join(S, "results", "test02")
        self.O2 = os.path.join(S, "out", "rank")
        self.C = os.path.join(self.D2, "checks")
        self.R4 = os.path.join(S, "results", "test04")
        self.zip = os.path.join(S, "drive", "test04_rank_results.zip")
        L = self.log
        t0 = time.time()
        # ---- R20 exactly as the golden run (spec 8.1 point 1)
        run_tool(["tools/t02_fake_tests.py", "from-val", "--root", S], log=L)
        self.t1_exit = run_tool(["tools/t02_checks.py", "t1", "--comp", "rank", "--dry-run", "--root", S,
                                 "--only", "T1.1,T1.2,T1.3"], expect=None, log=L)[0]
        mode = ["--dry-run", "--results-dir", self.D2, "--tests-dir", self.tests, "--out-dir", self.O2]
        run_tool(["tools/t02_rank.py", "opening"] + mode, log=L)
        run_tool(["tools/t02_rank.py", "apply"] + mode, log=L)
        self.r20_seconds = round(time.time() - t0, 1)
        self.t04_missing = [n for n in T04_TOOLS if not os.path.exists(os.path.join(TOOLS, n))]
        self.snap_steps = []
        if self.t04_missing:
            return
        # ---- 04 (spec 8.2)
        self.snap0 = self.t02_snapshot()
        t0 = time.time()
        self.step(["tools/t04_fake_tests.py", "from-val", "--root", S])
        self.step(["tools/t04_rank.py", "unpack", "--dry-run", "--zip", self.zip] + self.dirs())
        self.step(["tools/t04_rank.py", "opp", "--dry-run", "--workers", "6"] + self.dirs(),
                  env_extra={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        self.step(["tools/t04_rank.py", "apply", "--dry-run"] + self.dirs())
        self.status = self.step(["tools/t04_rank.py", "status", "--dry-run"] + self.dirs())
        self.t04_seconds = round(time.time() - t0, 1)

    def dirs(self, R4=None):
        return ["--results-dir", R4 or self.R4, "--t02-results-dir", self.D2, "--t02-out-dir", self.O2,
                "--tests-dir", self.tests, "--checks-dir", self.C]

    def t02_snapshot(self):
        return snapshot(self.D2, self.O2, self.tests)

    def step(self, args, expect=0, env_extra=None):
        """One t04 command with the test02 snapshot checked before and after (spec 8.2)."""
        before = self.t02_snapshot()
        res = run_tool(args, expect=expect, log=self.log, env_extra=env_extra)
        after = self.t02_snapshot()
        self.snap_steps.append((os.path.basename(args[0]) + " " + args[1], before == self.snap0, after == self.snap0))
        if after != before:
            changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
            raise AssertionError(f"{args[:2]} changed test02 files: {changed[:20]}")
        return res

    def finish(self):
        if os.environ.get("C10_KEEP") or not self.cleanup:
            with open(os.path.join(self.S, "t04_dryrun_log.json"), "w", encoding="utf-8") as f:
                json.dump({"r20_seconds": self.r20_seconds, "t04_seconds": getattr(self, "t04_seconds", None),
                           "steps": self.log}, f, indent=1, ensure_ascii=False)
        if self.cleanup and not os.environ.get("C10_KEEP"):
            shutil.rmtree(self.S, ignore_errors=True)


def tearDownModule():
    if Tree.built is not None:
        Tree.built.finish()
        Tree.built = None


# ================================================================ spec 8.1 point 2: R20 equals the golden record
@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (dry run of R20 and 04, about 7-8 min) or 2")
class R20Golden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = read_json(GOLDEN_PATH)
        cls.t = Tree.get()

    def test_structure_check_only(self):
        self.assertEqual(self.t.t1_exit, 0)
        t1 = read_json(os.path.join(self.t.C, "rank", "t1.json"))
        self.assertEqual(t1["verdict"], "PASS")
        self.assertFalse(t1.get("complete"), "content checks must not run on the from-val tree")

    def test_files_equal_golden(self):
        for rel, want in self.g["files_sha256"].items():
            with self.subTest(rel):
                self.assertEqual(sha(os.path.join(self.t.S, *rel.split("/"))), want)
        for rel, kv in self.g["known_values"].items():
            self.assertEqual(kv["expected"], self.g["files_sha256"][rel])

    def test_decide_summaries_normalised(self):
        ds = self.g["decide_summary"]
        self.assertIn("delete the key 'input'", ds["method"])
        for rel, rec in ds.items():
            if rel == "method":
                continue
            with self.subTest(rel):
                obj = read_json(os.path.join(self.t.S, *rel.split("/")))
                self.assertTrue(os.path.isabs(obj.pop("input")))
                norm = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
                self.assertEqual(hashlib.sha256(norm).hexdigest(), rec["normalised_sha256"])
                self.assertEqual(obj, rec["fields"])

    def test_opening_arrays_equal_golden(self):
        import numpy as np
        rec = self.g["opening_test_npz"]
        with np.load(os.path.join(self.t.S, *rec["path"].split("/")), allow_pickle=False) as z:
            self.assertEqual(list(z.files), rec["members"])
            for name, want in rec["arrays"].items():
                with self.subTest(name):
                    a = z[name]
                    got = {"dtype": a.dtype.str, "shape": list(a.shape),
                           "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
                    self.assertEqual(got, want)
        self.assertEqual(set(rec["members"]) - set(rec["arrays"]), set(rec["excluded"]))


# ================================================================ spec 8.2: the 04 pipeline
@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (dry run of R20 and 04, about 7-8 min) or 2")
class T04DryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree.get()
        if cls.t.t04_missing:
            raise unittest.SkipTest(f"tools/{', tools/'.join(cls.t.t04_missing)} not written yet (implementer I1); "
                                    f"the R20 golden part ran")
        cls.rank = os.path.join(cls.t.R4, "rank")
        cls.rep = read_json(os.path.join(cls.rank, "t04_apply_report.json"))

    # ---- the fake tree's rule
    def test_fake_tree_rule_and_no_content_statistics(self):
        tree = read_json(os.path.join(self.t.S, "FAKE_TREE_T04.json"))
        self.assertIn("never run content statistics", tree["rule"])
        self.assertIn("never run content statistics", read_json(os.path.join(self.t.S, "FAKE_TREE.json"))["rule"])
        for c in ("t2", "t3", "t4", "t5"):
            self.assertFalse(os.path.exists(os.path.join(self.t.C, "rank", c + ".json")), c)
        self.assertEqual(glob.glob(os.path.join(self.t.S, "**", "*dupcheck*"), recursive=True), [])

    def test_test02_untouched_by_every_t04_step(self):
        self.assertEqual(self.t.t02_snapshot(), self.t.snap0)
        self.assertTrue(self.t.snap_steps)
        for name, ok_before, ok_after in self.t.snap_steps:
            self.assertTrue(ok_before and ok_after, name)

    def test_unpack_layout(self):
        sw = os.path.join(self.t.R4, "sw")
        for name in ("config.json", "run_log.json", "env.json", "timings.json", "seam_check.json",
                     "readonly_before.json", "readonly_after.json"):
            self.assertTrue(os.path.exists(os.path.join(sw, name)), name)
        for s in SETS:
            d = os.path.join(sw, s, "dumps")
            for x in ("a", "b"):
                for v in ("d4", "s0_swap"):
                    self.assertTrue(os.path.exists(os.path.join(d, f"val_p03{x}_{v}.npz")), (s, x, v))
                    self.assertTrue(os.path.exists(os.path.join(d, f"_DONE_val_p03{x}_{v}.json")), (s, x, v))
            for name in ("val_keys.npz", "_DONE_val_keys.json", "_INPUT.json"):
                self.assertTrue(os.path.exists(os.path.join(d, name)), (s, name))
        u = read_json(os.path.join(self.t.R4, "unpack_sw.json"))
        self.assertEqual(u["format"], "c10-test04 unpack v1")
        self.assertEqual((u.get("zip") or {}).get("sha256"), sha(self.t.zip))
        self.assertEqual(u["attempt_number"], 1)
        seam = os.path.join(sw, "seam_val", "dumps")
        for x in ("a", "b"):                                   # seam_val is a byte copy of 03's validation dump
            ref = os.path.join(P04, "..", "prereg03", f"rank_{x}", "dumps")
            for v in ("d4", "s0_swap"):
                self.assertEqual(sha(os.path.join(seam, f"val_p03{x}_{v}.npz")),
                                 sha(os.path.join(ref, f"val_p03{x}_{v}.npz")), (x, v))

    # ---- opp
    def test_o4_seam_val_byte_identical(self):
        self.assertEqual(sha(os.path.join(self.t.R4, "opp", "o4_seam_val.npz")), O4_SEAM_SHA)
        rep = read_json(os.path.join(self.t.R4, "opp", "opp_report.json"))
        self.assertEqual((rep["format"], rep["mode"], rep["passed"], rep["usable_so_far"]),
                         ("c10-test04 opp v1", "dry-run", True, True))
        self.assertTrue(rep["seam_val"]["file_sha256_equal"])
        self.assertEqual(rep["seam_val"]["dkappa_O4"], DKAPPA_SEAM_VAL["O4"])
        self.assertTrue(rep["seam_val"]["dkappa_equal"])
        for s in SETS:
            self.assertEqual(rep["sets"][s]["failures"], 0, s)
        self.assertEqual(rep["sets"]["seam_val"]["slots"], 19156)
        self.assertEqual(rep["workers"], 6)

    def test_o4_test_sets_equal_validation_rows(self):
        import numpy as np
        with np.load(os.path.join(P04, "opp", "val_o4.npz"), allow_pickle=False) as v:
            ref = {k: v[k] for k in v.files}
        for s, (lo, hi) in O4_ROWS.items():
            with self.subTest(s), np.load(os.path.join(self.t.R4, "opp", f"o4_{s}.npz"), allow_pickle=False) as z:
                self.assertEqual(list(z.files), list(ref))
                q = z["question_id"]
                self.assertEqual(len(q), hi - lo)
                self.assertEqual(["val_" + x[len(PREFIX[s]):] for x in q.tolist()], ref["question_id"][lo:hi].tolist())
                for k in ("slot", "color", "o4", "t4", "n_moves", "anomalies", "moves_parsed", "replay_failed"):
                    self.assertEqual(z[k].dtype, ref[k].dtype, k)
                    self.assertEqual(z[k].tobytes(), ref[k][lo:hi].tobytes(), k)
                self.assertEqual(int(z["replay_failed"].sum()), 0)

    # ---- apply: seam_val
    def test_local_seam_byte_identical_to_prereg04(self):
        seam = os.path.join(self.rank, "seam_val")
        for name, want in Q02_VAL_SHA.items():
            self.assertEqual(sha(os.path.join(seam, name)), want, name)
        sl = read_json(os.path.join(seam, "seam_local.json"))
        self.assertTrue(sl["passed"])
        self.assertTrue(sl["byte_identical_probs"] and sl["byte_identical_pred_B"])
        self.assertEqual((sl["max_abs_diff"], sl["rule_b_differences"], sl["warnings"]), (0.0, 0, []))
        self.assertEqual(sl["mode"], "dry-run")
        self.assertEqual(sl["tol"], 1e-4)

    def test_seam_val_dkappa(self):
        col = self.rep["colour_check"]["seam_val"]
        self.assertEqual(dk(col["O4"]), DKAPPA_SEAM_VAL["O4"])               # bit for bit
        for src in ("SWA", "SWB"):                                           # byte copies: equal as well
            self.assertEqual(dk(col[src]), DKAPPA_SEAM_VAL[src], src)

    # ---- apply: the test sets
    def test_test_rows_equal_prereg04_validation_rows(self):
        ref = lf_lines(os.path.join(P04, "rank", "rank_val_probs.csv"))
        got = lf_lines(os.path.join(self.rank, "rank_test_probs.csv"))
        self.assertEqual(got[0], ref[0])
        self.assertEqual(len(got), 1 + 800 + 1)
        self.assertEqual(got[-1], b"")
        for s, (lo, hi) in RANK_ROWS.items():
            for i in range(lo, hi):
                r = ref[1 + i]
                self.assertTrue(r.startswith(b"val_q_"))
                self.assertEqual(got[1 + i], PREFIX[s].encode() + r[len(b"val_"):], (s, i))

    def test_submission(self):
        sub = os.path.join(self.rank, "submission_rank.csv")
        self.assertEqual(sha(sub), sha(os.path.join(self.rank, "pred_B.csv")))
        ref = csv_rows(os.path.join(P04, "rank", "pred_B.csv"))
        got = csv_rows(sub)
        self.assertEqual(got[0], ["question_id", "pred_rank"])
        self.assertEqual(len(got), 801)
        for s, (lo, hi) in RANK_ROWS.items():
            for i in range(lo, hi):
                q, lab = ref[1 + i]
                self.assertEqual(got[1 + i], [PREFIX[s] + q[len("val_"):], lab])
        qfiles = [os.path.join(self.t.tests, RANK_FILES[p]) for p in ("public", "private")]
        self.assertEqual(check_submission.check("rank", sub, qfiles, official=True), [])
        with open(sub, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r", raw)
        self.assertEqual(self.rep["outputs_sha256"]["submission_rank.csv"], sha(sub))
        r20 = csv_rows(os.path.join(self.t.O2, "submission_rank.csv"))
        self.assertEqual(self.rep["rule_b_differences_vs_r20"], sum(a != b for a, b in zip(got[1:], r20[1:])))
        # independently: the committed prereg02 and prereg04 rule-B predictions differ on as many of questions 1-800
        p02 = csv_rows(os.path.join(REPO, "results", "prereg02", "rank", "pred_B.csv"))[1:801]
        self.assertEqual(self.rep["rule_b_differences_vs_r20"], sum(a != b for a, b in zip(p02, ref[1:801])))
        self.assertGreater(self.rep["rule_b_differences_vs_r20"], 0)

    def test_fake_test_dkappa(self):
        col = self.rep["colour_check"]
        for s, d in DKAPPA_FAKE_ROUNDED.items():
            for src, want in d.items():
                with self.subTest(s=s, src=src):
                    v = dk(col[s][src])
                    self.assertGreater(v, 0)
                    self.assertEqual(round(v, 4), want)
                    if DKAPPA_FAKE_FULL is not None:
                        self.assertEqual(v, DKAPPA_FAKE_FULL[s][src])

    def test_apply_report(self):
        rep = self.rep
        for k in APPLY_REPORT_FIELDS:
            self.assertIn(k, rep)
        self.assertEqual((rep["format"], rep["mode"], rep["usable"], rep["unusable_reasons"]),
                         ("c10-test04 rank apply v1", "dry-run", True, []))
        self.assertEqual(rep["combo"]["id"], "Q02")
        self.assertEqual(rep["combo"]["sources"], ["C5", "OP", "O4", "SWA", "SWB"])
        self.assertEqual(rep["check_submission"]["mode"], "official")
        self.assertEqual(rep["check_submission"]["errors"], [])
        for name in TEST_OUTPUTS:
            self.assertEqual(rep["outputs_sha256"][name], sha(os.path.join(self.rank, name)), name)
        for s in SETS:
            self.assertEqual(rep["replay"][s]["failures"], 0, s)
        self.assertTrue(rep["replay"]["seam_val"]["equal_to_committed_val_o4"])
        self.assertTrue(rep["seam_local"]["passed"])
        nb = rep["notebook"]
        self.assertTrue(nb["seam_check_passed"])
        self.assertEqual(nb["earlier_failed_runs"], [])
        self.assertEqual(nb["keys_equal_to_test02"], {s: True for s in SETS})
        for key in ("results/test04/opp/opp_report.json", "results/test04/unpack_sw.json",
                    "results/test04/sw/config.json", "results/test04/sw/seam_check.json",
                    "results/test04/rank/seam_val/seam_local.json", "results/test02/rank/apply_test_report.json",
                    "results/test02/rank/opening_test.npz", "results/test02/checks/rank/t1.json",
                    "results/prereg04/rank/frozen_config.json"):
            self.assertIn(key, rep["inputs_bound"])
        self.assertFalse(os.path.exists(os.path.join(self.rank, "attempt_verdict.json")))

    def test_official_log_contract(self):
        """The fields official_log reads (spec 3.6, bold) are the ones t04_rank.py apply writes: the report, read as a
        real-mode one, has no problem for official_log, lists every required input, and each bound SHA-256 follows
        official_log's hashing rule (.csv / .json LF-folded, .npz raw)."""
        import official_log as ol
        sub = os.path.join(self.rank, "submission_rank.csv")
        rep = dict(self.rep, mode="real")
        cond, probs = ol.t04_report_problems(rep, "t04_apply_report.json", sub, sha(sub))
        self.assertEqual(probs, [])
        self.assertEqual(cond["replay_rate"], {"seam_val": 0.0, "public": 0.0, "private": 0.0})
        self.assertEqual(cond["dkappa"], {f"{s}.{c}": dk(self.rep["colour_check"][s][c])
                                          for s in ("public", "private") for c in ("O4", "SWA", "SWB")})
        bound, paths = self.rep["inputs_bound"], self.rep["inputs_bound_paths"]
        self.assertEqual(sorted(set(ol.T04_INPUTS_REQUIRED) - set(bound)), [])
        for key, want in bound.items():
            with self.subTest(key):
                p = paths[key] if os.path.isabs(paths[key]) else os.path.join(REPO, *paths[key].split("/"))
                self.assertEqual(ol.sha256_bound(p), want)
        for rel, need in ((os.path.join("opp", "opp_report.json"), {"passed": True, "usable_so_far": True}),
                          (os.path.join("sw", "seam_check.json"), {"passed": True}),
                          (os.path.join("rank", "seam_val", "seam_local.json"), {"passed": True})):
            obj = read_json(os.path.join(self.t.R4, rel))
            self.assertEqual({k: obj.get(k) for k in need}, need, rel)
            self.assertEqual(obj.get("mode", "dry-run"), "dry-run", rel)

    def test_status_is_read_only(self):
        code, out = self.t.status
        self.assertEqual(code, 0)
        before = snapshot(self.t.R4)
        run_tool(["tools/t04_rank.py", "status", "--dry-run"] + self.t.dirs())
        self.assertEqual(snapshot(self.t.R4), before)

    def test_repo_untouched(self):
        self.assertEqual(guarded_repo_state(), self.t.repo_state_before)


# ================================================================ spec 8.3: negative drills on copies
@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (dry run of R20 and 04, about 7-8 min) or 2")
class T04Negative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = Tree.get()
        if cls.t.t04_missing:
            raise unittest.SkipTest(f"tools/{', tools/'.join(cls.t.t04_missing)} not written yet (implementer I1)")
        cls.neg = os.path.join(cls.t.S, "neg")
        os.makedirs(cls.neg, exist_ok=True)
        cls.run_no = read_json(os.path.join(cls.t.R4, "sw", "run_log.json"))["runs"][-1]["run"]

    def r4(self, name, copy_unpacked=False):
        """A fresh R4 for one drill; copy_unpacked copies the unpacked sw/, unpack_sw.json and opp/ of the main run."""
        R4 = os.path.join(self.neg, name, "results", "test04")
        shutil.rmtree(os.path.dirname(os.path.dirname(R4)), ignore_errors=True)
        os.makedirs(R4)
        if copy_unpacked:
            shutil.copytree(os.path.join(self.t.R4, "sw"), os.path.join(R4, "sw"))
            shutil.copytree(os.path.join(self.t.R4, "opp"), os.path.join(R4, "opp"))
            shutil.copyfile(os.path.join(self.t.R4, "unpack_sw.json"), os.path.join(R4, "unpack_sw.json"))
        return R4

    def zip_with(self, name, edits):
        return rezip(self.t.zip, os.path.join(self.neg, name + ".zip"), edits)

    def apply(self, R4, expect, extra=()):
        return self.t.step(["tools/t04_rank.py", "apply", "--dry-run"] + list(extra) + self.t.dirs(R4), expect=expect)

    def unpack(self, R4, z, expect):
        return self.t.step(["tools/t04_rank.py", "unpack", "--dry-run", "--zip", z] + self.t.dirs(R4), expect=expect)

    def archive(self, R4, reason, expect):
        return self.t.step(["tools/t04_rank.py", "archive", "--dry-run", "--results-dir", R4, "--reason", reason],
                           expect=expect)

    def assert_no_test_outputs(self, R4, report=False):
        rank = os.path.join(R4, "rank")
        for name in TEST_OUTPUTS:
            self.assertFalse(os.path.exists(os.path.join(rank, name)), name)
        self.assertEqual(os.path.exists(os.path.join(rank, "t04_apply_report.json")), report)

    def verdict(self, R4):
        return read_json(os.path.join(R4, "rank", "attempt_verdict.json"))

    # ---- apply stage (the unpacked tree is edited)
    def test_mixed_environment(self):
        R4 = self.r4("env", copy_unpacked=True)
        mk = os.path.join(R4, "sw", "public", "dumps", "_DONE_val_p03a_d4.json")
        m = read_json(mk)
        m["gpu_name"] = "NVIDIA A100-SXM4-40GB"
        write_json(mk, m)
        self.apply(R4, expect=1)
        self.assert_no_test_outputs(R4)
        v = self.verdict(R4)
        self.assertEqual((v["format"], v["exit_code"], v["stage"]), ("c10-test04 attempt verdict v1", 1,
                                                                     "prerequisites"))

    def test_npz_changed_marker_not(self):
        R4 = self.r4("marker", copy_unpacked=True)

        def bump(a):
            x = a["logits"].copy()
            x[0, 0, 0] += 1.0
            a["logits"] = x
        rewrite_npz(os.path.join(R4, "sw", "public", "dumps", "val_p03a_d4.npz"), bump)
        self.apply(R4, expect=1)
        self.assert_no_test_outputs(R4)
        self.assertEqual((self.verdict(R4)["exit_code"], self.verdict(R4)["stage"]), (1, "prerequisites"))

    def test_keys_differ_from_test02(self):
        R4 = self.r4("keys", copy_unpacked=True)
        d = os.path.join(R4, "sw", "public", "dumps")
        rewrite_npz(os.path.join(d, "val_keys.npz"), flip_first_parsed)
        refresh_marker(d, "val_keys")
        self.apply(R4, expect=1)
        self.assert_no_test_outputs(R4)
        self.assertEqual((self.verdict(R4)["exit_code"], self.verdict(R4)["stage"]), (1, "prerequisites"))

    def test_swap_equal_to_a0_gives_dkappa_zero(self):
        import numpy as np
        R4 = self.r4("swap", copy_unpacked=True)
        d = os.path.join(R4, "sw", "public", "dumps")
        with np.load(os.path.join(d, "val_p03a_d4.npz"), allow_pickle=False) as z:
            a0 = np.ascontiguousarray(z["logits"][:, 0, :])

        def to_a0(a):
            self.assertEqual(str(a["view"]), "s0_colour_swapped")
            self.assertEqual(a["logits"].shape, a0.shape)
            a["logits"] = a0.astype(a["logits"].dtype)
        rewrite_npz(os.path.join(d, "val_p03a_s0_swap.npz"), to_a0)
        refresh_marker(d, "val_p03a_s0_swap")
        self.apply(R4, expect=3)
        self.assert_no_test_outputs(R4, report=True)
        self.assertFalse(os.path.exists(os.path.join(R4, "rank", "attempt_verdict.json")))
        rep = read_json(os.path.join(R4, "rank", "t04_apply_report.json"))
        self.assertFalse(rep["usable"])
        self.assertTrue(rep["unusable_reasons"])
        self.assertIsNone(rep["check_submission"])
        self.assertEqual(dk(rep["colour_check"]["public"]["SWA"]), 0.0)
        for s, src in (("public", "O4"), ("public", "SWB"), ("private", "SWA")):
            self.assertGreater(dk(rep["colour_check"][s][src]), 0, (s, src))
        code, out = self.apply(R4, expect=2, extra=("--rerun", "--reason", "dry run: must be refused"))
        self.assertIn("--bug-revision", out)
        code, out = self.archive(R4, "dry run: a registered decision is not archived", expect=2)
        self.assertIn("registered decision", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "failed")))

    # ---- unpack stage (the zip is edited)
    def test_notebook_seam_failed(self):
        z = self.zip_with("seam_failed", {
            "sw/seam_check.json": edit_json_bytes(lambda o: o.update(passed=False)),
            "sw/run_log.json": edit_json_bytes(lambda o: (o.update(status="seam_failed"),
                                                          o["runs"][-1].update(status="seam_failed")))})
        R4 = self.r4("seam_failed")
        self.unpack(R4, z, expect=1)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")), "a seam failure is not unpacked")

    def test_earlier_seam_failed_run(self):
        z = self.zip_with("earlier", {"sw/run_log.json": edit_json_bytes(
            lambda o: o["runs"].insert(0, {"run": self.run_no - 1, "status": "seam_failed",
                                           "start": "2026-11-05T09:00:00+08:00"}))})
        R4 = self.r4("earlier")
        code, out = self.unpack(R4, z, expect=0)            # outside the repository: a warning (spec 3.1)
        self.assertIn("seam_failed", out)
        shutil.copytree(os.path.join(self.t.R4, "opp"), os.path.join(R4, "opp"))
        self.apply(R4, expect=1)                            # spec 8.3: apply refuses
        self.assert_no_test_outputs(R4)

    def test_resumed_marker(self):
        z = self.zip_with("resumed", {"sw/public/dumps/_DONE_val_p03b_s0_swap.json": edit_json_bytes(
            lambda o: o.update(run=self.run_no - 1))})
        R4 = self.r4("resumed")
        self.unpack(R4, z, expect=1)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))

    def test_other_test_bundle(self):
        z = self.zip_with("bundle", {"sw/config.json": edit_json_bytes(
            lambda o: o["test_bundle"].update(manifest_sha256="0" * 64))})
        R4 = self.r4("bundle")
        self.unpack(R4, z, expect=2)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))

    def test_unpack_twice_refused(self):
        self.unpack(self.t.R4, self.t.zip, expect=2)        # the targets exist (no --rerun for unpack)

    # ---- reruns
    def test_opp_rerun_rules(self):
        code, out = self.t.step(["tools/t04_rank.py", "opp", "--dry-run", "--workers", "6"] + self.t.dirs(),
                                expect=2)
        self.assertIn("--rerun", out)
        code, out = self.t.step(["tools/t04_rank.py", "opp", "--dry-run", "--workers", "6", "--rerun", "--reason",
                                 "dry run: a registered decision is not re-run"] + self.t.dirs(), expect=2)
        self.assertIn("--bug-revision", out)

    def test_apply_rerun_without_reason(self):
        code, out = self.apply(self.t.R4, expect=2)
        self.assertIn("--rerun", out)

    # ---- output location guard (status is read-only, so a missing guard cannot write into the repo)
    def test_results_dir_inside_the_repo_refused(self):
        before = guarded_repo_state()
        for rel in ("results/test04", "results/test02", "results/prereg04", "results/test04_rehearsal_probe"):
            with self.subTest(rel):
                code, _out = run_tool(["tools/t04_rank.py", "status", "--dry-run"]
                                      + self.t.dirs(os.path.join(REPO, *rel.split("/"))), expect=None)
                self.assertEqual(code, 2)
        self.assertEqual(guarded_repo_state(), before)

    # ---- archive and the one registered rerun (spec 3.1 freshness, 3.7, 5.4)
    def test_seam_failure_is_archived_as_a_counted_attempt(self):
        z = self.zip_with("seam_failed_archive", {
            "sw/seam_check.json": edit_json_bytes(lambda o: o.update(passed=False)),
            "sw/run_log.json": edit_json_bytes(lambda o: (o.update(status="seam_failed"),
                                                          o["runs"][-1].update(status="seam_failed")))})
        R4 = self.r4("seam_failed_archive")
        zr = shutil.copyfile(z, os.path.join(R4, "test04_rank_results.zip"))
        self.unpack(R4, zr, expect=1)
        self.archive(R4, "dry run: notebook seam failed", expect=0)
        att = read_json(os.path.join(R4, "failed", "attempt_1", "attempt.json"))
        self.assertTrue(att["counts_toward_rerun"])
        self.assertIn("notebook_seam", att["classes_found"])
        self.assertEqual(att["failure_class"], "notebook_seam")     # spec 8.3 (integration 2026-09-30: tightened)
        self.assertEqual(att["zips"], {"test04_rank_results.zip": sha(z)})
        self.assertEqual(os.listdir(os.path.join(R4, "failed_zips", "attempt_1")), ["test04_rank_results.zip"])
        self.assertEqual(glob.glob(os.path.join(R4, "*.zip")), [])
        # review round 1: the archived seam-failed zip downloaded again (Drive still had it) is "not a recomputed
        # result" (2, spec 8.3), not a seam failure (1) that would lead into a refused second archive
        code, out = self.unpack(R4, z, expect=2)
        self.assertIn("這不是重算的結果", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))
        self.assertFalse(os.path.exists(os.path.join(R4, "unpack_sw.json")))

    def test_archive_and_rerun_sequence(self):
        """apply exit 1 -> archive (counted, both zips moved) -> the archived zip again: 2 -> a zip with the archived
        seam_val marker times: 2 -> a second counted archive: refused -> a recomputed zip: attempt 2, apply 0 ->
        archive after a registered decision: refused."""
        with zipfile.ZipFile(self.t.zip) as z:
            keys = npz_edited(z.read("sw/public/dumps/val_keys.npz"), flip_first_parsed)
            seam_markers = [n for n in z.namelist() if n.startswith("sw/seam_val/dumps/_DONE_")]
        self.assertTrue(seam_markers)
        z_keys = self.zip_with("rerun_keys", {
            "sw/public/dumps/val_keys.npz": lambda d: keys,
            "sw/public/dumps/_DONE_val_keys.json": edit_json_bytes(
                lambda o: o.update(sha256=hashlib.sha256(keys).hexdigest(), bytes=len(keys)))})
        R4 = self.r4("rerun")
        shutil.copytree(os.path.join(self.t.R4, "opp"), os.path.join(R4, "opp"))
        zr = shutil.copyfile(z_keys, os.path.join(R4, "test04_rank_results.zip"))
        self.unpack(R4, zr, expect=0)
        self.apply(R4, expect=1)
        self.assertEqual(self.verdict(R4)["exit_code"], 1)
        old = shutil.copyfile(self.t.zip, os.path.join(R4, "test04_rank_results (1).zip"))    # the old download
        self.archive(R4, "dry run: apply exit 1 (keys differ from test02)", expect=0)
        att = read_json(os.path.join(R4, "failed", "attempt_1", "attempt.json"))
        self.assertEqual((att["failure_class"], att["counts_toward_rerun"]), ("apply_exit_1", True))
        self.assertEqual(att["zips"], {"test04_rank_results.zip": sha(z_keys),
                                       "test04_rank_results (1).zip": sha(self.t.zip)})
        self.assertTrue(att["done_times"])
        self.assertEqual(sorted(os.listdir(os.path.join(R4, "failed_zips", "attempt_1"))),
                         ["test04_rank_results (1).zip", "test04_rank_results.zip"])
        self.assertFalse(os.path.exists(old))
        for gone in ("sw", "unpack_sw.json", "rank"):
            self.assertFalse(os.path.exists(os.path.join(R4, gone)), gone)
            self.assertTrue(os.path.exists(os.path.join(R4, "failed", "attempt_1", gone)), gone)
        self.assertTrue(os.path.exists(os.path.join(R4, "opp", "opp_report.json")), "opp/ stays")
        # the same zip again (Drive still had the old one): not a recomputed result
        code, out = self.unpack(R4, os.path.join(R4, "failed_zips", "attempt_1", "test04_rank_results.zip"), expect=2)
        self.assertIn("這不是重算的結果", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "sw")))
        # a new zip whose seam_val markers carry the archived times
        z_old_times = self.zip_with("rerun_old_times", {"sw/run_log.json": edit_json_bytes(
            lambda o: o["runs"][0].update(start=LATER))})
        code, out = self.unpack(R4, z_old_times, expect=2)
        self.assertIn("not later than the archived", out)
        # a seam failure of the rerun: the second counted archive is refused
        fresh = {n: edit_json_bytes(lambda o: o.update(time=LATER)) for n in seam_markers}
        z_fail2 = self.zip_with("rerun_fail2", dict(fresh, **{
            "sw/seam_check.json": edit_json_bytes(lambda o: o.update(passed=False)),
            "sw/run_log.json": edit_json_bytes(lambda o: (o.update(status="seam_failed"),
                                                          o["runs"][-1].update(status="seam_failed"),
                                                          o["runs"][0].update(start=LATER)))}))
        zr = shutil.copyfile(z_fail2, os.path.join(R4, "test04_rank_results.zip"))
        self.unpack(R4, zr, expect=1)
        code, out = self.archive(R4, "dry run: second seam failure", expect=2)
        self.assertIn("登記只准重跑一次", out)
        self.assertFalse(os.path.exists(os.path.join(R4, "failed", "attempt_2")))
        os.remove(zr)
        # the recomputed zip: the registered rerun
        z_fresh = self.zip_with("rerun_fresh", dict(fresh, **{"sw/run_log.json": edit_json_bytes(
            lambda o: o["runs"][0].update(start=LATER))}))
        code, out = self.unpack(R4, z_fresh, expect=0)
        self.assertIn("這是登記的重跑（第 2 次、也是最後一次）", out)
        self.assertEqual(read_json(os.path.join(R4, "unpack_sw.json"))["attempt_number"], 2)
        self.apply(R4, expect=0)
        self.assertEqual(sha(os.path.join(R4, "rank", "submission_rank.csv")),
                         sha(os.path.join(self.t.R4, "rank", "submission_rank.csv")))
        rep = read_json(os.path.join(R4, "rank", "t04_apply_report.json"))
        self.assertEqual(rep["attempt"]["number"], 2)
        code, out = self.archive(R4, "dry run: after a registered decision", expect=2)
        self.assertIn("registered decision", out)

    # ---- real mode only
    def test_before_r20_first_upload_real_mode_only(self):
        self.skipTest("the run that computed the outputs (or a marker) before R20's U1 uploaded_at -> unpack 2, archive "
                      "before_r20_u1; a preflight / GPU-gate stop before U1 is not that run (review round 2): real "
                      "mode with results/official_scores.json only (spec 8.3 says so): tests/test_t04_rank.py RealMode")


if __name__ == "__main__":
    unittest.main()
