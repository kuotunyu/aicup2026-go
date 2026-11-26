"""End-to-end dry runs of the 06 test-day kit (results/prereg06/design/spec.md 9.3; PREREG_06 5.1-5.4).

    C10_SLOW=1  Q04Golden (spec 9.3 item 1, about 1.5 min): t02_fake_tests.py from-val -> t02_checks.py t1 (structure
                only) -> t02_player.py run seam_val / public / private -> submit, i.e. the competition-1 part of
                tests/test_t02_dryrun.py FromValPipeline. Every output must equal results/test06/design/
                q04_fake_golden.json, which was recorded on branch p06-testday before tools/t06_player.py existed:
                submission_player.csv and the three top5.csv byte for byte, the reports after normalise() below.
    C10_SLOW=1  FromValDrills (spec 9.3 items 3 and 5, about 10 min): the same from-val tree, then t06_player.py
                feats / seam / apply / submit: the seam byte-identical to H6P, both pools select H6P's combination,
                top5_off / top5_on with the ids mapped back equal player_val_top5.csv / player_val_top5_switch.csv;
                then the drills on copies (exit 3 for FAIL_OPENING, K6', H6P not adopted, both pools on test02's rows;
                exit 2 for Q04 not uploaded; guard level 3 / 0 and test02's failed gate keep test02's rows; level 2,
                an unusable combination and H6P's secondary < 0 give no 06-ON; the 06 gate in process); review 10/3:
                K6' passed after the run is 2 (no not_used.json), the seam refuses seam_val dumps test02 did not
                record, a stale other-pool report is ignored, submit --rerun is refused once its run is in the log.
    C10_SLOW=1  OfficialShapeDrills (spec 9.3 item 4, about 12 min): tools/t06_fake_tests.py official (black only,
                5/10/15/20, the real CNN embedding) -> T1 structure -> test02 -> fake-t2 -> 06: the detector opens in
                both pools and 06-ON is produced; --dup 80 (two questions for 80 players) gives no 06-ON; a T2(b)
                failure gives no 06-ON; both submit reports pass official_log.py's report and inputs_bound checks
                (06-OFF binds no T2).
    The official_log.py drills of item 5 are tests/test_t06_official_log.py; the veto drills tests/test_t06_player.py.

Run as a script to (re)write the golden from an existing from-val tree (only before any t06 code exists):
    PY tests/test_t06_dryrun.py record-golden --tree S --out results/test06/design/q04_fake_golden.json

The trees live outside the repository (C10_DRYRUN_DIR, or a temporary directory). Content checks (T1.4-T5) never run
on the from-val tree (its games are validation games). Nothing here opens a label file or runs tools/score.py.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SLOW = os.environ.get("C10_SLOW", "")
GOLDEN_REL = "results/test06/design/q04_fake_golden.json"
GOLDEN = os.path.join(REPO, *GOLDEN_REL.split("/"))

# ---------------------------------------------------------------- the Q04 golden (spec 9.3 item 1)
PLAYER_FILES = ("out/player/submission_player.csv", "out/player/seam_val/top5.csv", "out/player/public/top5.csv",
                "out/player/private/top5.csv")
PLAYER_REPORTS = ("out/player/seam_val/pool_select.json", "out/player/seam_val/apply_report.json",
                  "out/player/seam_val/seam_local.json", "out/player/public/pool_select.json",
                  "out/player/public/apply_report.json", "out/player/private/pool_select.json",
                  "out/player/private/apply_report.json", "out/player/submit_report.json")
DROP_KEYS = ("provenance", "runtime_s")
# SHA-256 of files that carry a creation time (t1.json, MANIFEST.json, the fake notebook config, the other reports):
# two dry runs on the same code differ only there (checked 2026-10-03 with two runs at the same path)
VOLATILE = (("notebook", "config_sha256"), ("notebook_config_sha256",), ("t1", "sha256"),
            ("test_bundle", "MANIFEST.json"), ("seam", "sha256"), ("seam_local", "sha256"),
            ("pools", "*", "apply_report_sha256"), ("pools", "*", "pool_select_sha256"))
NORMALISE_RULE = ("json.load; drop the keys %s at any depth; replace the values at the key paths %s with "
                  "'<volatile>' (SHA-256 of files with a creation time); a string that is a path with a data/local "
                  "segment (either slash, absolute or relative) becomes 'data/local/<rest>' with forward slashes; "
                  "replace the tree's absolute path (backslash and slash forms) inside every other string with '<S>'; "
                  "sha256 of json.dumps(obj, sort_keys=True, ensure_ascii=False) as UTF-8"
                  % (list(DROP_KEYS), ["/".join(p) for p in VOLATILE]))
# the repository's own data/local inputs: relative on main, absolute in a worktree whose data is a junction (10/6)
DATA_LOCAL = re.compile(r"^(?:.*[\\/])?data[\\/]+local[\\/]+(.+)$", re.S)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _matches(path, pattern):
    return len(path) == len(pattern) and all(p == "*" or p == k for k, p in zip(path, pattern))


def normalise(obj, tree):
    """The report object with the run-dependent parts removed (NORMALISE_RULE)."""
    roots = sorted({os.path.abspath(tree), os.path.abspath(tree).replace("\\", "/")}, key=len, reverse=True)

    def walk(x, path):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k in DROP_KEYS:
                    continue
                p = path + (k,)
                out[k] = "<volatile>" if any(_matches(p, pat) for pat in VOLATILE) else walk(v, p)
            return out
        if isinstance(x, list):
            return [walk(v, path + ("[]",)) for v in x]
        if isinstance(x, str):
            m = DATA_LOCAL.match(x)
            if m:
                return "data/local/" + re.sub(r"[\\/]+", "/", m.group(1))
            for r in roots:
                x = x.replace(r, "<S>")
            return x
        return x
    return walk(copy.deepcopy(obj), ())


def normalised_sha(path, tree):
    obj = normalise(read_json(path), tree)
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def decisions(tree):
    """Human-readable decision fields of the competition-1 dry run (recorded with the golden)."""
    out = {}
    for pool in ("seam_val", "public", "private"):
        ps = read_json(os.path.join(tree, "out", "player", pool, "pool_select.json"))
        ar = read_json(os.path.join(tree, "out", "player", pool, "apply_report.json"))
        out[pool] = {"selected": ps["selected"]["id"], "gate": ps["gate"], "effective": ps["effective"]["id"],
                     "applied": ar["applied"]["id"], "guard_level": ar["guard_level"], "rho": ar["rho"],
                     "self_retrieval_mean_selected": ps["self_retrieval_mean"][ps["selected"]["id"]]}
    sl = read_json(os.path.join(tree, "out", "player", "seam_val", "seam_local.json"))
    out["seam_val"]["seam_passed"] = sl["passed"]
    out["seam_val"]["byte_identical_top5"] = sl["byte_identical_top5"]
    return out


def golden_record(tree):
    return {"files_sha256": {r: sha(os.path.join(tree, *r.split("/"))) for r in PLAYER_FILES},
            "reports_normalised_sha256": {r: normalised_sha(os.path.join(tree, *r.split("/")), tree)
                                          for r in PLAYER_REPORTS},
            "decisions": decisions(tree)}


def record_golden(tree, out):
    """Write the golden from an existing tree (spec 9.3 item 1: before any t06 code exists)."""
    if os.path.exists(os.path.join(REPO, "tools", "t06_player.py")):
        raise SystemExit("tools/t06_player.py exists: the golden is recorded before any t06 code (spec 9.3 item 1)")
    import numpy
    import scipy
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = {"what": "Q04 fake-tree golden (results/prereg06/design/spec.md 9.3 item 1; PREREG_06 5.1): the deterministic "
                   "outputs of test02's competition-1 dry run (t02_fake_tests.py from-val -> t02_checks.py t1 "
                   "--only T1.1,T1.2,T1.3 -> t02_player.py run seam_val / public / private --dry-run -> submit "
                   "--dry-run), recorded on branch p06-testday before tools/t06_player.py existed. "
                   "tests/test_t06_dryrun.py Q04Golden and the pre-merge check must reproduce every value under "
                   "files_sha256 and reports_normalised_sha256.",
           "tree": "S = the from-val tree (tests/test_t02_dryrun.py FromValPipeline workdir('fromval') when recorded; "
                   "tests/test_t06_dryrun.py uses its own workdir); paths below are relative to S",
           "normalise_rule": NORMALISE_RULE,
           **golden_record(tree),
           "provenance": {"git_head": head, "branch": "p06-testday", "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                          "tree": os.path.abspath(tree),
                          "commands": "C10_SLOW=1 C10_KEEP=1 C10_DRYRUN_DIR=<scratch>/golden PY -m pytest -q "
                                      "tests/test_t02_dryrun.py -k FromValPipeline (15 passed), run twice at the same "
                                      "path: the files and the normalised reports were equal; only the VOLATILE "
                                      "values and the dropped keys differed",
                          "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
                          "tools_sha256_lf": {f: hashlib.sha256(open(os.path.join(REPO, "tools", f), "rb").read()
                                                                .replace(b"\r\n", b"\n")).hexdigest()
                                              for f in ("t02_player.py", "t02_fake_tests.py", "t02_checks.py",
                                                        "p02_player_fuse.py", "p02_player_open.py",
                                                        "check_submission.py")}}}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return rec


# ---------------------------------------------------------------- running the tools
def inside(path, root):
    p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def run_tool(args, expect=0, log=None, env_extra=None):
    """Run PY tools/<args> from the repository root; returns (exit code, stdout+stderr)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONUTF8", None)
    if env_extra:
        env.update(env_extra)
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


def build_q04_tree(S, log):
    """test02's competition-1 dry run on a from-val tree at S (the commands of FromValPipeline, player part)."""
    run_tool(["tools/t02_fake_tests.py", "from-val", "--root", S], log=log)
    run_tool(["tools/t02_checks.py", "t1", "--comp", "player", "--dry-run", "--root", S, "--only", "T1.1,T1.2,T1.3"],
             log=log)
    mode = ["--dry-run", "--results-dir", os.path.join(S, "results", "test02"), "--tests-dir",
            os.path.join(S, "data", "test")]
    for pool in ("seam_val", "public", "private"):
        run_tool(["tools/t02_player.py", "run", "--pool", pool] + mode + ["--out-dir", os.path.join(S, "out", "player")],
                 log=log)
    run_tool(["tools/t02_player.py", "submit"] + mode + ["--out-dir", os.path.join(S, "out", "player")], log=log)


@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (06 dry runs)")
class Q04Golden(unittest.TestCase):
    """Spec 9.3 item 1: test02's competition-1 path is byte-for-byte what it was before any t06 code."""

    @classmethod
    def setUpClass(cls):
        cls.S, cls.cleanup = workdir("q04golden")
        cls.log = []
        build_q04_tree(cls.S, cls.log)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "cleanup", False) and not os.environ.get("C10_KEEP"):
            shutil.rmtree(cls.S, ignore_errors=True)

    def test_equal_to_golden(self):
        want = read_json(GOLDEN)
        got = golden_record(self.S)
        self.assertEqual(got["files_sha256"], want["files_sha256"])
        self.assertEqual(got["reports_normalised_sha256"], want["reports_normalised_sha256"])
        self.assertEqual(got["decisions"], want["decisions"])

    def test_golden_rule_unchanged(self):
        self.assertEqual(read_json(GOLDEN)["normalise_rule"], NORMALISE_RULE)


# ---------------------------------------------------------------- the 06 drills (spec 9.3 items 3-5)
P6 = os.path.join(REPO, "results", "prereg06", "player")
AIDEA = "drill: the AIdea data page says 400 candidates and 400 questions per pool, nothing against one per candidate"


def rows_of(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def write_obj(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


def t06_args(S, o6=None, o2=None, checks=None, p6=None, log=None, now=None):
    a = ["--dry-run", "--out-dir", o6 or os.path.join(S, "o6"), "--data-dir", os.path.join(S, "a6"),
         "--t02-results-dir", os.path.join(S, "results", "test02"),
         "--t02-out-dir", o2 or os.path.join(S, "out", "player"), "--tests-dir", os.path.join(S, "data", "test")]
    for k, v in (("--checks-dir", checks), ("--prereg06-dir", p6), ("--log", log), ("--now", now)):
        if v:
            a += [k, v]
    return a


def t06(S, cmd, *extra, expect=0, runlog=None, **kw):
    return run_tool(["tools/t06_player.py"] + cmd.split() + t06_args(S, **kw) + list(extra), expect=expect,
                    log=runlog)


def run_t06_all(S, runlog, submit_extra=()):
    for pool in ("seam_val", "public", "private"):
        t06(S, f"feats --pool {pool}", runlog=runlog)
    t06(S, "seam", runlog=runlog)
    for pool in ("public", "private"):
        t06(S, f"apply --pool {pool}", runlog=runlog)
    t06(S, "submit", *submit_extra, runlog=runlog)


def drill_o6(S, name):
    """A fresh 06 output folder holding the base run's feats reports and seam (apply and submit run again there)."""
    d = os.path.join(S, "drills", name, "o6")
    shutil.rmtree(d, ignore_errors=True)
    for pool in ("seam_val", "public", "private"):
        os.makedirs(os.path.join(d, pool))
        shutil.copyfile(os.path.join(S, "o6", pool, "feats_report.json"), os.path.join(d, pool, "feats_report.json"))
    for n in ("t06_seam.json", "top5_off.csv", "top5_on.csv"):
        shutil.copyfile(os.path.join(S, "o6", "seam_val", n), os.path.join(d, "seam_val", n))
    return d


def drill_o2(S, name, changes):
    """A copy of test02's competition-1 outputs with fields of apply_report.json changed ({pool: {field: value}}): a
    drill fixture; the top-5 files and the other reports stay byte-identical."""
    d = os.path.join(S, "drills", name, "o2")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(S, "out", "player"), d)
    for pool, fields in changes.items():
        p = os.path.join(d, pool, "apply_report.json")
        rep = read_json(p)
        rep.update(fields)
        write_obj(p, rep)
    return d


def drill_p6(S, name, file, fn):
    """A scratch copy of results/prereg06/player with one file changed (--prereg06-dir: its pins are then recorded
    as not checked)."""
    d = os.path.join(S, "drills", name, "p6")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(P6, d)
    obj = read_json(os.path.join(d, file))
    fn(obj)
    write_obj(os.path.join(d, file), obj)
    return d


def q04_log(S, name, uploaded=True):
    p = os.path.join(S, "drills", name, "official_scores.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    e = {"id": 1, "comp": "player", "status": "uploaded" if uploaded else "prepared",
         "file": "results/test02/player/submission_player.csv",
         "sha256": sha(os.path.join(S, "out", "player", "submission_player.csv")),
         "uploaded_at": "2026-11-04T16:05:00+08:00" if uploaded else None, "public_score": None}
    write_obj(p, {"format": "c10 official_scores v1", "uploads": [e], "final_choice": {"rank": None, "player": None}})
    return p


def in_process(argv, patches):
    """tools/t06_player.py main() in this process with module attributes replaced (drills that need a condition
    no fixture file can produce: the 06 gate, an unusable combination)."""
    import contextlib
    import io
    from unittest import mock
    tools = os.path.join(REPO, "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import t06_player as t6
    out = io.StringIO()
    with contextlib.ExitStack() as st:
        for name, value in patches(t6):
            st.enter_context(mock.patch.object(t6, name, value))
        st.enter_context(contextlib.redirect_stdout(out))
        st.enter_context(contextlib.redirect_stderr(out))
        code = t6.main(argv)
    return code, out.getvalue()


@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (06 dry runs)")
class FromValDrills(unittest.TestCase):
    """Spec 9.3 item 3 (06 equals H6P on the from-val tree, whose public and private pools are the validation pool
    with other ids) and the item-5 drills that need no T2 (exit codes, guard levels 3 / 2 / 0, test02's gate and the
    06 gate, an unusable combination, H6P's secondary < 0, H6P not adopted, K6', Q04 not uploaded)."""

    @classmethod
    def setUpClass(cls):
        cls.S, cls.cleanup = workdir("t06_fromval")
        cls.log = []
        build_q04_tree(cls.S, cls.log)
        run_t06_all(cls.S, cls.log, ("--aidea-page-checked", AIDEA))

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "cleanup", False) and not os.environ.get("C10_KEEP"):
            shutil.rmtree(cls.S, ignore_errors=True)

    def o6(self, *parts):
        return os.path.join(self.S, "o6", *parts)

    def test_q04_path_unchanged_on_this_tree(self):
        got = golden_record(self.S)
        want = read_json(GOLDEN)
        self.assertEqual(got["files_sha256"], want["files_sha256"])
        self.assertEqual(got["reports_normalised_sha256"], want["reports_normalised_sha256"])

    def test_seam_byte_identical(self):
        seam = read_json(self.o6("seam_val", "t06_seam.json"))
        self.assertTrue(seam["passed"])
        self.assertTrue(seam["dumps_bitwise_equal"])
        self.assertTrue(seam["byte_identical_top5_off"] and seam["byte_identical_top5_on"])
        self.assertEqual(seam["selected"], read_json(os.path.join(P6, "apply_report.json"))["selected"])
        self.assertTrue(all(seam["bitwise_additions"].values()), seam["bitwise_additions"])
        self.assertTrue(all(seam["dump_independent"].values()), seam["dump_independent"])

    def test_pools_select_as_h6p_and_rows_equal_h6p(self):
        h = read_json(os.path.join(P6, "apply_report.json"))
        ref = {"off": rows_of(os.path.join(P6, "player_val_top5.csv")),
               "on": rows_of(os.path.join(P6, "player_val_top5_switch.csv"))}
        for pool, pre in (("public", "pub_"), ("private", "priv_")):
            rep = read_json(self.o6(pool, "t06_pool.json"))
            self.assertEqual(rep["selected"], h["selected"], pool)
            self.assertTrue(rep["used06"] and rep["gate06"]["passed"] and rep["top5_on_written"], pool)
            for v in ("off", "on"):
                back = [[c.replace(pre, "val_", 1) for c in r] for r in rows_of(self.o6(pool, f"top5_{v}.csv"))]
                self.assertEqual(back, ref[v], f"{pool} top5_{v}")

    def test_submit(self):
        rep = read_json(self.o6("off", "t06_submit_report.json"))
        self.assertTrue(rep["check_submission"]["passed"])
        self.assertEqual(rep["output"]["rows"], 800)
        self.assertEqual(sha(self.o6("off", "submission_player.csv")), rep["output"]["sha256"])
        # no T2 on a from-val tree (its games are validation games): 06-ON is not produced
        self.assertFalse(rep["on_produced"])
        self.assertFalse(rep["on_conditions"]["t2_c_zero_and_abd_not_failed"]["ok"])
        self.assertFalse(os.path.exists(self.o6("on", "submission_player.csv")))

    def test_fail_opening_is_3(self):
        checks = os.path.join(self.S, "drills", "fail_opening", "checks")
        shutil.rmtree(checks, ignore_errors=True)
        shutil.copytree(os.path.join(self.S, "results", "test02", "checks"), checks)
        p = os.path.join(checks, "player", "t1.json")
        t1 = read_json(p)
        t1["verdict"] = "FAIL_OPENING"
        write_obj(p, t1)
        o6 = os.path.join(self.S, "drills", "fail_opening", "o6")
        _code, out = t06(self.S, "apply --pool public", expect=3, o6=o6, checks=checks, runlog=self.log)
        self.assertIn("FAIL_OPENING", out)
        self.assertEqual([e["reason"] for e in read_json(os.path.join(o6, "not_used.json"))["events"]],
                         ["t1_fail_opening"])

    def test_k6p_passed_is_3(self):
        o6 = os.path.join(self.S, "drills", "k6p", "o6")
        _code, out = t06(self.S, "apply --pool public", expect=3, o6=o6, now="2026-11-22T00:00:00+08:00",
                         runlog=self.log)
        self.assertIn("K6' 已過", out)

    def test_h6p_not_adopted_is_3(self):
        p6 = drill_p6(self.S, "h6p_not_adopted", "score_H6P.json",
                      lambda o: o["B_minus_A"].update(ci95=[-0.001, o["B_minus_A"]["ci95"][1]]))
        o6 = os.path.join(self.S, "drills", "h6p_not_adopted", "o6")
        _code, out = t06(self.S, "apply --pool public", expect=3, o6=o6, p6=p6, runlog=self.log)
        self.assertIn("H6P 沒有採用", out)

    def test_q04_not_uploaded_is_2(self):
        o6 = os.path.join(self.S, "drills", "q04_log", "o6")
        _code, out = t06(self.S, "apply --pool public", expect=2, o6=o6, log=q04_log(self.S, "q04_log", False),
                         runlog=self.log)
        self.assertIn("Q04's first upload", out)
        o6 = drill_o6(self.S, "q04_log_ok")
        t06(self.S, "apply --pool public", o6=o6, log=q04_log(self.S, "q04_log_ok", True), runlog=self.log)

    def test_level3_keeps_test02_rows(self):
        o2 = drill_o2(self.S, "level3", {"private": {"guard_level": 3}})
        o6 = drill_o6(self.S, "level3")
        for pool in ("public", "private"):
            t06(self.S, f"apply --pool {pool}", o6=o6, o2=o2, runlog=self.log)
        rep = read_json(os.path.join(o6, "private", "t06_pool.json"))
        self.assertFalse(rep["used06"])
        self.assertIn("level 3", rep["rule"])
        self.assertEqual(sha(os.path.join(o6, "private", "top5_off.csv")), sha(os.path.join(o2, "private", "top5.csv")))
        self.assertFalse(os.path.exists(os.path.join(o6, "private", "top5_on.csv")))
        t06(self.S, "submit", "--aidea-page-checked", AIDEA, o6=o6, o2=o2, runlog=self.log)
        sub = rows_of(os.path.join(o6, "off", "submission_player.csv"))
        q04 = rows_of(os.path.join(o2, "submission_player.csv"))
        self.assertEqual(sub[401:], q04[401:])                     # the private rows are test02's
        self.assertNotEqual(sub[1:401], q04[1:401])
        self.assertFalse(read_json(os.path.join(o6, "off", "t06_submit_report.json"))["on_produced"])

    def test_level3_both_pools_is_3(self):
        o2 = drill_o2(self.S, "level3_both", {"public": {"guard_level": 3}, "private": {"guard_level": 3}})
        o6 = drill_o6(self.S, "level3_both")
        t06(self.S, "apply --pool public", o6=o6, o2=o2, runlog=self.log)
        _code, out = t06(self.S, "apply --pool private", expect=3, o6=o6, o2=o2, runlog=self.log)
        self.assertIn("兩池都退回 test02", out)

    def test_level2_and_level0(self):
        o2 = drill_o2(self.S, "level2", {"public": {"guard_level": 2}})
        o6 = drill_o6(self.S, "level2")
        for pool in ("public", "private"):
            t06(self.S, f"apply --pool {pool}", o6=o6, o2=o2, runlog=self.log)
        rep = read_json(os.path.join(o6, "public", "t06_pool.json"))
        self.assertTrue(rep["used06"])
        self.assertFalse(rep["switch_allowed_at_this_level"])
        t06(self.S, "submit", "--aidea-page-checked", AIDEA, o6=o6, o2=o2, runlog=self.log)
        cond = read_json(os.path.join(o6, "off", "t06_submit_report.json"))["on_conditions"]
        self.assertFalse(cond["both_pools_06_at_guard_level_1"]["ok"])
        o2 = drill_o2(self.S, "level0", {"public": {"guard_level": 0}})
        o6 = drill_o6(self.S, "level0")
        t06(self.S, "apply --pool public", o6=o6, o2=o2, runlog=self.log)
        rep = read_json(os.path.join(o6, "public", "t06_pool.json"))
        self.assertFalse(rep["used06"])
        self.assertIn("level 0", rep["rule"])

    def test_test02_gate_failed_keeps_test02_rows(self):
        o2 = drill_o2(self.S, "q00", {"public": {"gate_passed": False, "guard_level": 0}})
        o6 = drill_o6(self.S, "q00")
        t06(self.S, "apply --pool public", o6=o6, o2=o2, runlog=self.log)
        rep = read_json(os.path.join(o6, "public", "t06_pool.json"))
        self.assertFalse(rep["used06"])
        self.assertIn("gate failed", rep["rule"])
        self.assertEqual(sha(os.path.join(o6, "public", "top5_off.csv")), sha(os.path.join(o2, "public", "top5.csv")))

    def test_06_gate_failed_keeps_test02_rows(self):
        o6 = drill_o6(self.S, "gate06")
        code, out = in_process(["apply", "--pool", "public"] + t06_args(self.S, o6=o6), lambda t6: [("GATE06_MIN", 1.0)])
        self.assertEqual(code, 0, out)
        rep = read_json(os.path.join(o6, "public", "t06_pool.json"))
        self.assertFalse(rep["used06"])
        self.assertFalse(rep["gate06"]["passed"])
        self.assertEqual(sha(os.path.join(o6, "public", "top5_off.csv")),
                         sha(os.path.join(self.S, "out", "player", "public", "top5.csv")))
        self.assertFalse(os.path.exists(os.path.join(o6, "public", "top5_on.csv")))

    def test_unusable_combination_no_switch(self):
        o6 = drill_o6(self.S, "unusable")

        def patches(t6):
            real = t6.load_frozen

            def load(ctx):
                fz, models, rec = real(ctx)
                fz = copy.deepcopy(fz)
                fz["switch"]["V2"]["usable"] = False
                return fz, models, rec
            return [("load_frozen", load)]
        for pool in ("public", "private"):
            code, out = in_process(["apply", "--pool", pool] + t06_args(self.S, o6=o6), patches)
            self.assertEqual(code, 0, out)
        priv = read_json(os.path.join(o6, "private", "t06_pool.json"))
        self.assertEqual(priv["selected"], "V2")
        self.assertFalse(priv["detector"]["usable"])
        self.assertFalse(priv["top5_on_written"])
        self.assertTrue(read_json(os.path.join(o6, "public", "t06_pool.json"))["top5_on_written"])  # forced probe
        code, out = in_process(["submit", "--aidea-page-checked", AIDEA] + t06_args(self.S, o6=o6), patches)
        self.assertEqual(code, 0, out)
        rep = read_json(os.path.join(o6, "off", "t06_submit_report.json"))
        self.assertFalse(rep["on_conditions"]["private_switch_usable_and_detector_open"]["ok"])
        self.assertFalse(rep["on_produced"])

    def test_h6p_secondary_negative_no_on(self):
        p6 = drill_p6(self.S, "h6p_switch_neg", "score_H6P_switch.json",
                      lambda o: o["B_minus_A"].update(mean_diff=-0.0001))
        o6 = drill_o6(self.S, "h6p_switch_neg")
        for pool in ("public", "private"):
            t06(self.S, f"apply --pool {pool}", o6=o6, p6=p6, runlog=self.log)
        t06(self.S, "submit", "--aidea-page-checked", AIDEA, o6=o6, p6=p6, runlog=self.log)
        rep = read_json(os.path.join(o6, "off", "t06_submit_report.json"))
        self.assertFalse(rep["on_conditions"]["h6p_secondary_not_negative"]["ok"])
        self.assertFalse(rep["on_produced"])

    def test_rerun_rules(self):
        _code, out = t06(self.S, "seam", expect=2, runlog=self.log)
        self.assertIn("--rerun --reason", out)

    # ------------------------------------------------------------ review 10/3 (correctness review, minors 1-4)
    def test_k6p_passed_after_the_run_is_2(self):
        """K6' passed with this tree's 06 run already there: refused (2), no not_used.json (the run stays valid)."""
        _code, out = t06(self.S, "apply --pool public", expect=2, now="2026-11-22T00:00:00+08:00", runlog=self.log)
        self.assertIn("K6' 前已 commit 的 06 照常有效", out)
        self.assertFalse(os.path.exists(self.o6("not_used.json")))

    def test_seam_dumps_must_be_test02s(self):
        """The seam reads only the seam_val dumps test02's own seam_val run recorded (dumpsdiff_shuffle drill)."""
        import numpy as np
        d2 = os.path.join(self.S, "drills", "seam_dumps", "results", "test02")
        shutil.rmtree(os.path.dirname(d2), ignore_errors=True)
        shutil.copytree(os.path.join(self.S, "results", "test02", "checks"), os.path.join(d2, "checks"))
        src = os.path.join(self.S, "results", "test02", "player", "seam_val", "dumps")
        dst = os.path.join(d2, "player", "seam_val", "dumps")
        shutil.copytree(src, dst)
        with np.load(os.path.join(src, "cand_games.npz"), allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
        arrs["emb_sum"] = arrs["emb_sum"][np.random.default_rng(7).permutation(len(arrs["emb_sum"]))]
        with open(os.path.join(dst, "cand_games.npz"), "wb") as f:
            np.savez(f, **arrs)
        o6 = drill_o6(self.S, "seam_dumps")
        _code, out = t06(self.S, "seam", "--rerun", "--reason", "drill: seam_val dumps test02 never judged",
                         "--t02-results-dir", d2, expect=2, o6=o6, runlog=self.log)
        self.assertIn("differ from test02's seam_val apply_report.json inputs", out)
        self.assertFalse(os.path.exists(os.path.join(o6, "not_used.json")))

    def test_stale_other_pool_report_ignored(self):
        """'Both pools on test02's rows' counts the other pool only when its report is from the present test02 report
        (stale_other drill: private fell back under an older test02; the present test02 has private at level 1)."""
        o2_old = drill_o2(self.S, "stale_old", {"private": {"guard_level": 3}})
        o6 = drill_o6(self.S, "stale")
        t06(self.S, "apply --pool private", o6=o6, o2=o2_old, runlog=self.log)
        self.assertFalse(read_json(os.path.join(o6, "private", "t06_pool.json"))["used06"])
        o2_new = drill_o2(self.S, "stale_new", {"public": {"guard_level": 3}})
        t06(self.S, "apply --pool public", o6=o6, o2=o2_new, runlog=self.log)        # was exit 3 before the fix
        self.assertFalse(read_json(os.path.join(o6, "public", "t06_pool.json"))["used06"])
        self.assertFalse(os.path.exists(os.path.join(o6, "not_used.json")))
        t06(self.S, "apply --pool private", "--rerun", "--reason", "drill: test02 re-run", o6=o6, o2=o2_new,
            runlog=self.log)
        self.assertTrue(read_json(os.path.join(o6, "private", "t06_pool.json"))["used06"])

    def test_submit_rerun_refused_once_logged(self):
        """submit --rerun after this run's 06-OFF is in the upload log: refused (2), nothing rewritten."""
        rep_p = self.o6("off", "t06_submit_report.json")
        before = sha(rep_p)
        log = q04_log(self.S, "submit_logged", True)
        obj = read_json(log)
        obj["uploads"].append({"id": 2, "comp": "player", "status": "prepared",
                               "file": "results/test06/player/off/submission_player.csv",
                               "sha256": read_json(rep_p)["output"]["sha256"], "uploaded_at": None,
                               "public_score": None, "test06": {"variant": "off", "run_id": read_json(rep_p)["run_id"]}})
        write_obj(log, obj)
        _code, out = t06(self.S, "submit", "--rerun", "--reason", "drill: add the AIdea text afterwards",
                         "--aidea-page-checked", AIDEA, expect=2, log=log, runlog=self.log)
        self.assertIn("already holds 06 entries [2]", out)
        self.assertEqual(sha(rep_p), before)


@unittest.skipUnless(SLOW in ("1", "2"), "set C10_SLOW=1 (06 dry runs)")
class OfficialShapeDrills(unittest.TestCase):
    """Spec 9.3 item 4: an official-shape tree (t06_fake_tests.py official: fit players, black only, 5/10/15/20, the
    real CNN embedding of each game) is one question per candidate: the detector opens in both pools and 06-ON is
    produced; the "same player, two questions" variant (--dup 80) gives no 06-ON; T2(b) failing gives no 06-ON."""

    @classmethod
    def build(cls, label, *extra):
        S, cleanup = workdir(label)
        log = []
        run_tool(["tools/t06_fake_tests.py", "official", "--root", S] + list(extra), log=log)
        run_tool(["tools/t02_checks.py", "t1", "--comp", "player", "--dry-run", "--root", S, "--only",
                  "T1.1,T1.2,T1.3"], log=log)
        mode = ["--dry-run", "--results-dir", os.path.join(S, "results", "test02"), "--tests-dir",
                os.path.join(S, "data", "test"), "--out-dir", os.path.join(S, "out", "player")]
        for pool in ("seam_val", "public", "private"):
            run_tool(["tools/t02_player.py", "run", "--pool", pool] + mode, log=log)
        run_tool(["tools/t02_player.py", "submit"] + mode, log=log)
        run_tool(["tools/t06_fake_tests.py", "fake-t2", "--root", S], log=log)
        run_t06_all(S, log, ("--aidea-page-checked", AIDEA))
        return S, cleanup, log

    @classmethod
    def setUpClass(cls):
        cls.S, cls.cleanup, cls.log = cls.build("t06_official")
        cls.S2, cls.cleanup2, cls.log2 = cls.build("t06_official_dup", "--dup", "80")

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("C10_KEEP"):
            for S, c in ((cls.S, cls.cleanup), (cls.S2, cls.cleanup2)):
                if c:
                    shutil.rmtree(S, ignore_errors=True)

    def test_official_shape_gives_06_on(self):
        S = self.S
        tree = read_json(os.path.join(S, "FAKE_TREE.json"))
        self.assertTrue(tree["t06"]["add_dumps"]["embeddings"]["used"])
        self.assertEqual(tree["t06"]["add_dumps"]["embeddings"]["games_fixed_seed"], 0)
        for pool in ("public", "private"):
            rep = read_json(os.path.join(S, "o6", pool, "t06_pool.json"))
            self.assertEqual(rep["test02"]["guard_level"], 1, pool)
            self.assertEqual(rep["games"]["question"]["W"], 0, pool)
            self.assertEqual(rep["games"]["candidate"]["W"], 0, pool)
            self.assertTrue(rep["detector"]["opens"], pool)
            self.assertTrue(rep["used06"], pool)
        q = rows_of(os.path.join(S, "data", "test", "player_identification_test_public.csv"))
        ks = sorted(int(r[1]) for r in q[1:])
        self.assertEqual({k: ks.count(k) for k in set(ks)}, {5: 100, 10: 100, 15: 100, 20: 100})
        on = read_json(os.path.join(S, "o6", "on", "t06_submit_report.json"))
        self.assertTrue(on["all_on_conditions"], on["on_conditions"])
        self.assertTrue(on["check_submission"]["passed"])
        self.assertGreater(on["top1_differences_off_vs_on"]["private"], 0)

    def test_two_questions_per_player_no_06_on(self):
        S = self.S2
        dup = read_json(os.path.join(S, "FAKE_TREE.json"))["t06"]["dup"]
        self.assertEqual(dup["public"]["candidates_with_two_questions"], 80)
        self.assertEqual(dup["private"]["candidates_without_question"], 80)
        t2 = read_json(os.path.join(S, "results", "test02", "checks", "player", "t2.json"))
        c = [u["copies"] for u in t2["units"] if u["unit"] == "c"]
        priv = read_json(os.path.join(S, "o6", "private", "t06_pool.json"))
        self.assertTrue(any(c) or not priv["detector"]["opens"], (c, priv["detector"]))
        off = read_json(os.path.join(S, "o6", "off", "t06_submit_report.json"))
        self.assertFalse(off["on_produced"])
        self.assertFalse(os.path.exists(os.path.join(S, "o6", "on", "submission_player.csv")))

    def test_reports_pass_official_log_checks(self):
        """The two t06_submit_report.json files are the ones tools/official_log.py reads (field names, inputs_bound):
        copied with their inputs into a temporary repository under the real-mode paths, mode set to real, git
        mocked, ol.t06_report_problems and ol.t06_inputs_problems return nothing (review 10/3, procedure review).
        06-OFF binds no T2, 06-ON does."""
        from unittest import mock
        tools = os.path.join(REPO, "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import official_log as ol
        S = self.S
        root = tempfile.mkdtemp(prefix="c10_t06_ol_")
        maps = [(os.path.join(S, "out", "player"), "results/test02/player"),
                (os.path.join(S, "results", "test02"), "results/test02"),
                (os.path.join(S, "o6"), "results/test06/player")]

        def to_rel(k):
            if not os.path.isabs(k):
                return k                                   # results/prereg06/player/... (already repo-relative)
            for src, dst in maps:
                if inside(k, src):
                    return dst + "/" + os.path.relpath(k, src).replace(os.sep, "/")
            raise AssertionError(f"inputs_bound key outside the tree: {k}")
        try:
            for v in ("off", "on"):
                rep = read_json(os.path.join(S, "o6", v, "t06_submit_report.json"))
                bound = {}
                for k, want in rep["inputs_bound"].items():
                    rel = to_rel(k)
                    src = k if os.path.isabs(k) else os.path.join(REPO, *k.split("/"))
                    dst = os.path.join(root, *rel.split("/"))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(src, dst)
                    bound[rel] = want
                rel_sub = f"results/test06/player/{v}/submission_player.csv"
                sub = os.path.join(root, *rel_sub.split("/"))
                os.makedirs(os.path.dirname(sub), exist_ok=True)
                shutil.copyfile(os.path.join(S, "o6", v, "submission_player.csv"), sub)
                state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
                with mock.patch.object(ol, "REPO", root), mock.patch.object(ol, "git_file_state", return_value=state):
                    probs = ol.t06_report_problems(dict(rep, mode="real", inputs_bound=bound),
                                                   f"results/test06/player/{v}/{ol.T06_REPORT}", sub, sha(sub), v)
                    probs += ol.t06_inputs_problems(bound)
                self.assertEqual(probs, [], v)
                has_t2 = "results/test02/checks/player/t2.json" in bound
                self.assertEqual(has_t2, v == "on", v)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_t2b_failed_no_06_on(self):
        S = self.S
        o6 = drill_o6(S, "t2b")
        checks = os.path.join(S, "drills", "t2b", "checks")
        shutil.rmtree(checks, ignore_errors=True)
        shutil.copytree(os.path.join(S, "results", "test02", "checks"), checks)
        tree = os.path.join(S, "drills", "t2b", "tree")
        shutil.rmtree(tree, ignore_errors=True)
        shutil.copytree(os.path.join(S, "data", "test"), os.path.join(tree, "data", "test"))
        run_tool(["tools/t06_fake_tests.py", "fake-t2", "--root", tree, "--fail", "b"], log=self.log)
        shutil.copyfile(os.path.join(tree, "results", "test02", "checks", "player", "t2.json"),
                        os.path.join(checks, "player", "t2.json"))
        for pool in ("public", "private"):
            t06(S, f"apply --pool {pool}", o6=o6, checks=checks, runlog=self.log)
        t06(S, "submit", "--aidea-page-checked", AIDEA, o6=o6, checks=checks, runlog=self.log)
        off = read_json(os.path.join(o6, "off", "t06_submit_report.json"))
        cond = off["on_conditions"]["t2_c_zero_and_abd_not_failed"]
        self.assertFalse(cond["ok"])
        self.assertIn("b:public", cond["detail"]["abd_failed"])
        self.assertFalse(off["on_produced"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="record the Q04 fake-tree golden (spec 9.3 item 1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record-golden")
    p.add_argument("--tree", required=True)
    p.add_argument("--out", default=GOLDEN)
    a = ap.parse_args(argv)
    rec = record_golden(a.tree, a.out)
    print(json.dumps({k: rec[k] for k in ("files_sha256", "reports_normalised_sha256")}, indent=1))


if __name__ == "__main__":
    main()
