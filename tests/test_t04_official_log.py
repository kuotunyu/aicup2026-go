"""Tests for the PREREG_04 rev 3 additions to tools/official_log.py (results/test04/design/spec.md 8.1 point 4, 9.2,
9.3, 10.3; understand_protocol.md 3.5).

* R20UnchangedWithT04Files: the eleven test_t02_infra.OfficialLog tests run again, unchanged, while a complete
  results/test04 tree (a usable 04 run) sits next to results/test02 but the log has no 04 entry.
* TagVersusNew (spec 8.1 point 4): the tagged r20-testday-pre04 version (git show, executed as a separate module,
  nothing written into the repository) and the working-tree version run the same R20 / player flow on the same
  fixture with the same mocks: the log bytes, the returned records, the stdout and the refusal messages are identical;
  the only difference is the K6 line that status appends.
* T04*: the 04 path (prepare, the refusals, --method, final and --not-04-reason under PREREG_04 5.3, status, public,
  the bound inputs and the archive history).
Synthetic files in a temporary directory only; git is mocked (ol.git_file_state, ol.git_commit_time,
ol.git_head_text, ol.git_row_added). Nothing here opens a label file or runs tools/score.py.
Run: python -m pytest tests/test_t04_official_log.py
"""

import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import official_log as ol  # noqa: E402
import t02_test_bundle as tb  # noqa: E402
import test_t02_infra as ti  # noqa: E402

TAG = "r20-testday-pre04"
K6_PREFIX = "距 11/16 23:59（K6"
NOW = "2026-11-05T06:00:00+00:00"                 # prepared_at of every entry here (14:00 Taiwan time)
REPORT_COMMIT = ("a" * 40, "2026-11-05T12:00:00+08:00")
DKAPPA = {"public": {"O4": 0.0930, "SWA": 0.0321, "SWB": 0.0278},
          "private": {"O4": 0.1000, "SWA": 0.0418, "SWB": 0.0364}}
PREREG04_HEAD = "# PREREG_04\n\n| rev 1 | `cdec913` | first |\n| rev 2 | `82027fa` | second |\n| rev 3 | (commit) | test day |\n"
NPZ_INPUTS = ["results/test02/rank/opening_test.npz"] + [
    f"results/test02/rank/{s}/dumps/{n}.npz" for s in ("seam_val", "public", "private")
    for n in ("val_keys", "val_p02e9_d4")]

# The registration-revision check (spec 3.0; tools/t04_rank.py --bug-revision implements the same definition and may
# reuse these cases). Each overridden record was prepared at RECORD_AT.
RECORD_AT = "2026-11-05T14:00:00+08:00"
REVISION_CASES = [
    dict(name="free text", reason="found a bug in the 04 path", rows=(4,), added="2026-11-06T10:00:00+08:00",
         clean=True, ok=False, msg="without a revision"),
    dict(name="rev below 4", reason="PREREG_04 rev 3: bug", rows=(3,), added="2026-11-06T10:00:00+08:00",
         clean=True, ok=False, msg="rev 4 or later"),
    dict(name="row missing", reason="PREREG_04 rev 4: bug in t04_rank apply", rows=(), added=None,
         clean=True, ok=False, msg="has no row"),
    dict(name="doc not committed", reason="PREREG_04 rev 4: bug", rows=(4,), added="2026-11-06T10:00:00+08:00",
         clean=False, ok=False, msg="committed and unchanged"),
    dict(name="no adding commit", reason="PREREG_04 rev 4: bug", rows=(4,), added=None,
         clean=True, ok=False, msg="no commit found"),
    dict(name="revision before the record", reason="PREREG_04 rev 4: bug", rows=(4,),
         added="2026-11-05T13:59:59+08:00", clean=True, ok=False, msg="not after the records"),
    dict(name="revision at the record time", reason="PREREG_04 rev 4: bug", rows=(4,), added=RECORD_AT,
         clean=True, ok=False, msg="not after the records"),
    dict(name="valid", reason="PREREG_04 rev 4: t04_rank apply swapped SWA and SWB (5.3 item 4)", rows=(4,),
         added="2026-11-06T10:00:00+08:00", clean=True, ok=True, msg=None),
]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


def head_with_rows(*revs):
    return PREREG04_HEAD + "".join(f"| rev {n} | `{n:07d}` | revision {n} |\n" for n in revs)


# ---------------------------------------------------------------- the results/test04 tree of a usable 04 run
class T04Tree:
    """Mixin for a test case that has test_t02_infra.OfficialLog's fixture (repo, tests, rank_rows, state, jw,
    write_reports, prepare, run_ol): adds results/test04 (a usable real-mode 04 run) and the git mocks of the 04 path."""
    mock_now = True

    def t04_setup(self):
        self.commit = REPORT_COMMIT
        self.p04_head = PREREG04_HEAD
        self.rev_added = {}
        self.dirty, self.untracked = set(), set()
        self.patches04 = [
            mock.patch.object(ol, "git_commit_time", side_effect=lambda rel: self.commit),
            mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p04_head),
            mock.patch.object(ol, "git_row_added", side_effect=lambda rel, row: self.rev_added.get(row, (None, None))),
            mock.patch.object(ol, "git_file_state", side_effect=lambda rel: dict(
                self.state, clean=self.state["clean"] and rel not in self.dirty,
                tracked=self.state["tracked"] and rel not in self.untracked))]
        if self.mock_now:
            self.patches04.append(mock.patch.object(ol, "now_iso", return_value=NOW))
        for pt in self.patches04:
            pt.start()
        self.write_t04_inputs()
        self.write04()

    def t04_teardown(self):
        for pt in reversed(self.patches04):
            pt.stop()

    def r(self, rel):
        return os.path.join(self.repo, *rel.split("/"))

    def write_t04_inputs(self):
        self.t04 = self.r("results/test04/rank")
        rows = [[q, "C" if i % 7 == 0 else lab] for i, (q, lab) in enumerate(self.rank_rows)]
        self.f04 = ti.write_csv(os.path.join(self.t04, "submission_rank.csv"), ["question_id", "pred_rank"], rows)
        write_json(self.r(ol.T04_OPP_REPORT), {"format": "c10-test04 opp v1", "mode": "real", "passed": True,
                                                "usable_so_far": True, "unusable_reasons": []})
        write_json(self.r("results/test04/unpack_sw.json"), {"format": "c10-test04 unpack v1", "attempt_number": 1})
        write_json(self.r("results/test04/sw/config.json"), {"format": "c10-test04 rank notebook config v1",
                                                              "smoke": False, "rehearsal": False, "kind": "official"})
        write_json(self.r(ol.T04_SEAM_CHECK), {"passed": True, "tol": 1e-4, "bitwise_equal": True})
        write_json(self.r(ol.T04_SEAM_LOCAL), {"passed": True, "mode": "real", "tol": 1e-4})
        write_json(self.r("results/prereg04/rank/frozen_config.json"), {"combo": {"id": "Q02"}})
        for rel in NPZ_INPUTS:
            os.makedirs(os.path.dirname(self.r(rel)), exist_ok=True)
            with open(self.r(rel), "wb") as f:
                f.write(b"PK-fake-npz " + rel.encode() + b"\r\n")     # raw bytes (a CR stays: npz is not folded)

    def report04(self, **over):
        rep = {
            "format": ol.T04_FORMAT, "mode": "real", "created": "2026-11-05T12:00:00+08:00",
            "combo": {"id": "Q02", "sources": list(ol.T04_COMBO_SOURCES), "terms": ["OP", "O4", "SWA", "SWB"],
                      "path": "gen"},
            "notebook": {"seam_check_passed": True, "seam_check_tol": 1e-4, "seam_check_bitwise_equal": True,
                         "run_log_status": "completed", "earlier_failed_runs": [],
                         "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "same_as_repo_notebook": True,
                         "weights_sha256": {"a": "790b767c", "b": "8e88f6ec"}, "test_bundle_equal_to_test02": True,
                         "local_recompute_passed": True,
                         "keys_equal_to_test02": {"seam_val": True, "public": True, "private": True},
                         "env_same": True,       # review round 2: flat, as tools/t04_rank.py writes it (check_dumps)
                         "env": {"gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "torch": "2.11.0+cu128",
                                 "cudnn": 91900}},
            "seam_local": {"passed": True, "tol": 1e-4, "max_abs_diff": 0.0, "rule_b_differences": 0,
                           "byte_identical_probs": True, "byte_identical_pred_B": True, "warnings": [],
                           "file": "seam_val/seam_local.json"},
            "replay": {"seam_val": {"slots": 19156, "failures": 0, "equal_to_committed_val_o4": True},
                       "public": {"slots": 5000, "failures": 0}, "private": {"slots": 5000, "failures": 0},
                       "max_rate": 0.005},
            "colour_check": {s: {src: {"dkappa": v, "sub_questions": 400} for src, v in d.items()}
                             for s, d in DKAPPA.items()},
            "usable": True, "unusable_reasons": [],
            "check_submission": {"mode": "official", "questions": [], "errors": []},
            "outputs_sha256": {"submission_rank.csv": sha(self.f04)},
            "test02": {"fallback_used": False, "t1_verdict": "PASS", "r20_first_upload_id": 1},
            "inputs_bound": {rel: ol.sha256_bound(self.r(rel)) for rel in ol.T04_INPUTS_REQUIRED},
            "attempt": {"number": 1, "counted_archives": [], "bug_revisions": []},
            "rule_b_differences_vs_r20": 115,
        }
        for k, v in over.items():
            rep[k] = v
        return rep

    def write04(self, **over):
        return write_json(os.path.join(self.t04, ol.T04_REPORT), self.report04(**over))

    def edit04(self, fn):
        """Change the report through fn(report dict), keeping every other field."""
        p = os.path.join(self.t04, ol.T04_REPORT)
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def log_obj(self):
        with open(self.log, encoding="utf-8") as f:
            return json.load(f)

    def r20_uploaded(self, day="2026-11-05"):
        e, _ = self.prepare()
        self.run_ol("uploaded", "--id", str(e["id"]), "--ok-at", f"{day}T14:10", "--uploaded-at", f"{day}T14:15")
        return e

    def prepare04(self, *extra):
        return self.prepare("rank", self.f04, "prereg04 Q02 first upload (PREREG_04 5.3)", *extra)

    def uploaded(self, eid, at="2026-11-05T15:10"):
        return self.run_ol("uploaded", "--id", str(eid), "--ok-at", at, "--uploaded-at", at)

    ATTEMPT_RUN = "2026-11-05T09:00:00+08:00"          # the archived run's start and marker times (archive helper)
    ATTEMPT_DONE = "2026-11-05T09:30:00+08:00"

    def archive(self, n, counts=True, with_report=False, bug_revision=None, failure_class=None):
        """An attempt.json as tools/t04_rank.py archive writes it (runs, done_times); counts=None leaves out
        counts_toward_rerun (an incomplete record, counted on the safe side)."""
        d = self.r(f"results/test04/failed/attempt_{n}")
        cls = failure_class or ("bug_revision" if bug_revision else
                                ("notebook_seam" if counts is not False else "before_r20_u1"))
        att = {"reason": "seam", "failure_class": cls, "counts_toward_rerun": counts, "bug_revision": bug_revision,
               "runs": [{"run": 1, "status": "seam_failed", "start": self.ATTEMPT_RUN}],
               "done_times": {"_DONE_val_keys.json": self.ATTEMPT_DONE}}
        if counts is None:
            del att["counts_toward_rerun"]
        write_json(os.path.join(d, "attempt.json"), att)
        if with_report:
            write_json(os.path.join(d, "rank", ol.T04_REPORT), {"format": ol.T04_FORMAT})


# ---------------------------------------------------------------- R20 unchanged next to a 04 tree
class R20UnchangedWithT04Files(T04Tree, ti.OfficialLog):
    """understand_protocol.md 3.5 point 4: the eleven R20 / player tests of test_t02_infra.OfficialLog, unchanged, with
    a usable results/test04 run next to results/test02 (no 04 entry in the log)."""
    mock_now = False

    def setUp(self):
        super().setUp()
        self.t04_setup()

    def tearDown(self):
        self.t04_teardown()
        super().tearDown()

    def test_r20_entry_has_no_04_fields(self):
        e, out = self.prepare()
        self.assertNotIn("test04", e)
        self.assertNotIn("prereg04_5_3", e)
        self.assertEqual(e["method"], ol.DEFAULT_METHOD["rank"])
        self.assertNotIn("這一份是 04", out)
        self.run_ol("uploaded", "--id", "1", "--ok-at", "2099-11-04T15:00", "--uploaded-at", "2099-11-04T15:05")
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", "1", "--basis", "prereg02")
        self.assertNotIn("prereg04_5_3", fc)
        self.assertEqual(set(self.log_obj()["final_choice"]["rank"]),
                         {"id", "sha256", "file", "method", "basis", "set_at"})
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertEqual(st["rank"]["prereg04"]["entries"], [])
        self.assertIn(K6_PREFIX, out)
        self.assertNotIn("PREREG_04 5.3", out)


# ---------------------------------------------------------------- the tagged version against the new one
def load_tagged_official_log():
    """tools/official_log.py of tag r20-testday-pre04 as a separate module (never written into the repository)."""
    r = subprocess.run(["git", "-C", REPO, "show", f"{TAG}:tools/official_log.py"], capture_output=True, timeout=60)
    if r.returncode != 0:
        raise AssertionError(f"local tag {TAG} missing or without tools/official_log.py (PREREG_04 9 step 1)")
    spec = importlib.util.spec_from_loader("official_log_r20_tag", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.join(REPO, "tools", "official_log.py")
    exec(compile(r.stdout.decode("utf-8"), f"<{TAG}:tools/official_log.py>", "exec"), mod.__dict__)
    return mod


class TagVersusNew(unittest.TestCase):
    """Spec 8.1 point 4 (rev 3 item 13): R20's prepare (U1) -> uploaded -> public -> final -> status, plus the
    player flow and the refusals, give byte-identical logs and stdout in the tagged and the new version; status
    only appends the K6 line."""

    @classmethod
    def setUpClass(cls):
        cls.tag = load_tagged_official_log()
        assert hasattr(cls.tag, "cmd_prepare") and not hasattr(cls.tag, "check_reports_t04"), "not the tagged file"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t04_ol_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, with_t04, r10=False):
        """The same fixture as test_t02_infra.OfficialLog (and, optionally, the 04 tree), at the same path each time.
        r10 (review round 1): T1 FAIL_OPENING on competition 2 and a rank file made with the R10 fallback."""
        repo = os.path.join(self.tmp, "repo")
        shutil.rmtree(repo, ignore_errors=True)
        tests = ti.write_official(os.path.join(repo, "data", "test"))
        files = {n: tb.file_record(os.path.join(tests, n)) for n in tb.ALL_NAMES}
        tb.write_manifest(tests, "official", files, {"rank": "PASS", "player": "PASS"})
        ids = [f"pub_q_{i:04d}" for i in range(1, 401)] + [f"priv_q_{i:04d}" for i in range(1, 401)]
        rank_rows = [[q, "D"] for q in ids]
        rank_file = ti.write_csv(os.path.join(repo, "results", "test02", "rank", "submission_rank.csv"),
                                 ["question_id", "pred_rank"], rank_rows)
        player_file = ti.write_csv(os.path.join(repo, "results", "test02", "player", "submission_player.csv"),
                                   ["question_id"] + [f"top_{i}" for i in range(1, 6)],
                                   [[q] + [f"{q.split('_')[0]}_p_{j:04d}" for j in range(5)] for q in ids])
        R = os.path.join(repo, "results", "test02")
        write_json(os.path.join(R, "rank", "apply_test_report.json"), {
            "mode": "real", "seam_local": {"passed": True}, "fallback_used": bool(r10),
            "check_submission": {"mode": "official", "errors": []},
            "outputs_sha256": {"submission_rank.csv": sha(rank_file)}})
        write_json(os.path.join(R, "player", "submit_report.json"), {
            "mode": "real", "check_submission": {"official": True, "passed": True, "errors": []},
            "output": {"sha256": sha(player_file)}})
        for comp in ("rank", "player"):
            write_json(os.path.join(R, comp, "seam_val", "seam_local.json"), {"passed": True, "mode": "real"})
            write_json(os.path.join(R, "checks", comp, "t1.json"), {
                "verdict": "FAIL_OPENING" if (r10 and comp == "rank") else "PASS", "comp": comp, "complete": True,
                "stage": "raw",
                "subchecks_run": ["T1.1", "T1.2", "T1.3", "T1.4", "T1.5"], "provenance": {"mode": "real"}})
            checks = {}
            for c, verdict in (("T2", "PASS"), ("T3", "RECORD_ONLY"), ("T4", "PASS"), ("T5", "RECORD_ONLY")):
                dp = write_json(os.path.join(R, "checks", comp, c.lower() + ".json"), {"verdict": verdict})
                checks[c] = {"verdict": verdict, "details_file": c.lower() + ".json", "sha256": sha(dp)}
            write_json(os.path.join(R, "checks", comp, "summary.json"), {"checks": checks})
        if with_t04:
            d = os.path.join(repo, "results", "test04", "rank")
            ti.write_csv(os.path.join(d, "submission_rank.csv"), ["question_id", "pred_rank"],
                         [[q, "C"] for q, _ in rank_rows])
            write_json(os.path.join(d, ol.T04_REPORT), {"format": ol.T04_FORMAT, "mode": "real"})
            write_json(os.path.join(d, "seam_val", "seam_local.json"), {"passed": True, "mode": "real"})
        return repo, tests, rank_file, player_file

    def flow(self, mod, with_t04, r10=False):
        """Run the R20 (or R10) / player flow with module mod; -> [(step, stdout, returned record or refusal, log
        bytes)]."""
        repo, tests, rank_file, player_file = self.build(with_t04, r10)
        log = os.path.join(repo, "results", "official_scores.json")
        state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
        patches = [mock.patch.object(mod, "REPO", repo),
                   mock.patch.object(mod, "git_file_state", side_effect=lambda rel: dict(state)),
                   mock.patch.object(mod, "now_iso", return_value="2026-11-04T06:00:00+00:00")]
        for name in ("git_commit_time", "git_head_text", "git_row_added"):     # the R20 path never needs them
            if hasattr(mod, name):
                patches.append(mock.patch.object(mod, name, side_effect=AssertionError(f"{name} on the R20 path")))
        steps = [
            ("prepare rank U1", ["prepare", "--comp", "rank", "--file", rank_file, "--reason", "first upload (U1)",
                                 "--tests-dir", tests]),
            ("prepare rank R10 name", ["prepare", "--comp", "rank", "--file", rank_file, "--reason", "x",
                                       "--method", "prereg02 R10 fallback", "--tests-dir", tests]),
            ("uploaded before ok", ["uploaded", "--id", "1", "--ok-at", "2026-11-04T15:00",
                                    "--uploaded-at", "2026-11-04T14:00"]),
            ("uploaded 1", ["uploaded", "--id", "1", "--ok-at", "2026-11-04T15:00", "--uploaded-at",
                            "2026-11-04T15:05"]),
            ("uploaded 1 again", ["uploaded", "--id", "1", "--ok-at", "2026-11-04T15:00", "--uploaded-at",
                                  "2026-11-04T15:05"]),
            ("public 1", ["public", "--id", "1", "--score", "0.61", "--lb-rank", "12", "--note", "U1"]),
            ("prepare player", ["prepare", "--comp", "player", "--file", player_file, "--reason", "first upload (U1)",
                                "--tests-dir", tests]),
            ("uploaded 2", ["uploaded", "--id", "2", "--ok-at", "2026-11-04T16:00", "--uploaded-at",
                            "2026-11-04T16:02"]),
            ("public 2", ["public", "--id", "2", "--score", "0.88"]),
            ("prepare rank again", ["prepare", "--comp", "rank", "--file", rank_file, "--reason", "re-upload",
                                    "--tests-dir", tests]),
            ("uploaded 3", ["uploaded", "--id", "3", "--ok-at", "2026-11-05T10:00", "--uploaded-at",
                            "2026-11-05T10:01"]),
            ("public 3", ["public", "--id", "3", "--score", "0.55"]),
            ("public 3 again", ["public", "--id", "3", "--score", "0.5"]),
            ("final wrong comp", ["final", "--comp", "player", "--id", "1", "--basis", "x"]),
            ("final rank", ["final", "--comp", "rank", "--id", "1", "--basis", "prereg02 (甲)"]),
            ("final rank again", ["final", "--comp", "rank", "--id", "3", "--basis", "prereg02 (甲), same file"]),
            ("final player", ["final", "--comp", "player", "--id", "2", "--basis", "prereg02 (甲)"]),
            ("status", ["status", "--now", "2026-11-10T00:00"]),
            ("status late", ["status", "--now", "2026-11-30T00:00"]),
        ]
        if r10:     # review round 1: the R10 path (T1 FAIL_OPENING) of the R20 test day, same file names
            steps = [
                ("prepare rank R10 U1", ["prepare", "--comp", "rank", "--file", rank_file, "--reason",
                                         "first upload (U1, T1 FAIL_OPENING: R10)", "--tests-dir", tests]),
                ("prepare rank R20 name", ["prepare", "--comp", "rank", "--file", rank_file, "--reason", "x",
                                           "--method", "prereg02 R20 rule B", "--tests-dir", tests]),
                ("uploaded 1", ["uploaded", "--id", "1", "--ok-at", "2026-11-04T15:00", "--uploaded-at",
                                "2026-11-04T15:05"]),
                ("public 1", ["public", "--id", "1", "--score", "0.58", "--note", "U1 R10"]),
                ("prepare player", ["prepare", "--comp", "player", "--file", player_file, "--reason",
                                    "first upload (U1)", "--tests-dir", tests]),
                ("uploaded 2", ["uploaded", "--id", "2", "--ok-at", "2026-11-04T16:00", "--uploaded-at",
                                "2026-11-04T16:02"]),
                ("public 2", ["public", "--id", "2", "--score", "0.88"]),
                ("prepare rank R10 again", ["prepare", "--comp", "rank", "--file", rank_file, "--reason", "re-upload",
                                            "--tests-dir", tests]),
                ("uploaded 3", ["uploaded", "--id", "3", "--ok-at", "2026-11-05T10:00", "--uploaded-at",
                                "2026-11-05T10:01"]),
                ("public 3", ["public", "--id", "3", "--score", "0.52"]),
                ("final rank", ["final", "--comp", "rank", "--id", "3", "--basis", "prereg02 R10 (T1 FAIL_OPENING)"]),
                ("final player", ["final", "--comp", "player", "--id", "2", "--basis", "prereg02 (甲)"]),
                ("status", ["status", "--now", "2026-11-10T00:00"]),
                ("status late", ["status", "--now", "2026-11-30T00:00"]),
            ]
        out = []
        for pt in patches:
            pt.start()
        try:
            for name, argv in steps:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    try:
                        res = mod.main(argv + ["--log", log])
                    except SystemExit as e:
                        res = ("refused", str(e))
                with open(log, "rb") as f:
                    out.append((name, buf.getvalue(), json.dumps(res, sort_keys=True, ensure_ascii=False), f.read()))
        finally:
            for pt in reversed(patches):
                pt.stop()
        return out

    def compare(self, with_t04, r10=False):
        old, new = self.flow(self.tag, with_t04, r10), self.flow(ol, with_t04, r10)
        self.assertEqual([s[0] for s in old], [s[0] for s in new])
        refused = [s[0] for s in old if s[2].startswith('["refused"')]
        self.assertEqual(refused, ["prepare rank R20 name"] if r10 else
                         ["prepare rank R10 name", "uploaded before ok", "uploaded 1 again", "public 3 again",
                          "final wrong comp"])
        for (name, o_out, o_res, o_log), (_n, n_out, n_res, n_log) in zip(old, new):
            with self.subTest(step=name, with_t04=with_t04, r10=r10):
                self.assertEqual(n_log, o_log, "log bytes differ")
                if name.startswith("status"):
                    o_lines, n_lines = o_out.rstrip("\n").split("\n"), n_out.rstrip("\n").split("\n")
                    self.assertEqual(n_lines[:-1], o_lines)
                    self.assertTrue(n_lines[-1].startswith(K6_PREFIX), n_lines[-1])
                    self.assertEqual(len(n_lines), len(o_lines) + 1)
                    o_st, n_st = json.loads(o_res), json.loads(n_res)
                    self.assertEqual(n_st["rank"].pop("prereg04")["entries"], [])
                    self.assertEqual(n_st, o_st)
                else:
                    self.assertEqual(n_out, o_out, "stdout differs")
                    self.assertEqual(n_res, o_res, "returned record / refusal differs")
        return old, new

    def test_r20_flow_identical(self):
        old, _new = self.compare(with_t04=False)
        log = json.loads(old[-1][3])
        self.assertEqual([e["status"] for e in log["uploads"]], ["uploaded"] * 3)
        self.assertEqual(log["final_choice"]["rank"]["id"], 3)

    def test_r20_flow_identical_next_to_a_04_tree(self):
        self.compare(with_t04=True)

    def test_r10_flow_identical(self):
        """Review round 1: T1 FAIL_OPENING, the R10 fallback file: prepare (U1) -> uploaded -> public -> final ->
        status give the same log bytes and stdout in the tagged and the new version (an appended line that changed
        R10's method or output used to pass every test)."""
        for with_t04 in (False, True):
            old, _new = self.compare(with_t04=with_t04, r10=True)
            log = json.loads(old[-1][3])
            self.assertEqual([(e["method"], e["fallback_used"]) for e in log["uploads"] if e["comp"] == "rank"],
                             [(ol.FALLBACK_METHOD, True)] * 2)
            self.assertEqual(log["final_choice"]["rank"]["id"], 3)

    def test_status_k6_line(self):
        st, out = quiet(ol.main, ["status", "--now", "2026-11-16T23:00:00+08:00", "--log",
                                  os.path.join(self.tmp, "none.json")])
        self.assertIn(f"{K6_PREFIX}：04 的正式執行要在這之前 commit）：0 天 0 小時", out)
        _st, out = quiet(ol.main, ["status", "--now", "2026-11-17T00:00:00+08:00", "--log",
                                   os.path.join(self.tmp, "none.json")])
        self.assertIn(f"{K6_PREFIX}：04 的正式執行要在這之前 commit）：已過", out)
        self.assertEqual(st["rank"]["prereg04"]["entries"], [])


# ---------------------------------------------------------------- the 04 path
class T04Base(T04Tree, ti.Tmp):
    """test_t02_infra.OfficialLog's fixture without its tests (its helpers are plain functions)."""
    jw = ti.OfficialLog.jw
    write_reports = ti.OfficialLog.write_reports
    run_ol = ti.OfficialLog.run_ol
    prepare = ti.OfficialLog.prepare

    def setUp(self):
        ti.Tmp.setUp(self)
        self.repo = self.p("repo")
        self.tests = ti.write_official(os.path.join(self.repo, "data", "test"))
        files = {n: tb.file_record(os.path.join(self.tests, n)) for n in tb.ALL_NAMES}
        tb.write_manifest(self.tests, "official", files, {"rank": "PASS", "player": "PASS"})
        ids = [f"pub_q_{i:04d}" for i in range(1, 401)] + [f"priv_q_{i:04d}" for i in range(1, 401)]
        self.rank_rows = [[q, "D"] for q in ids]
        self.rank_file = ti.write_csv(os.path.join(self.repo, "results", "test02", "rank", "submission_rank.csv"),
                                      ["question_id", "pred_rank"], self.rank_rows)
        self.player_file = ti.write_csv(
            os.path.join(self.repo, "results", "test02", "player", "submission_player.csv"),
            ["question_id"] + [f"top_{i}" for i in range(1, 6)],
            [[q] + [f"{q.split('_')[0]}_p_{j:04d}" for j in range(5)] for q in ids])
        self.log = self.p("repo", "results", "official_scores.json")
        self.state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
        self.write_reports()
        self.patches = [mock.patch.object(ol, "REPO", self.repo)]
        for pt in self.patches:
            pt.start()
        self.t04_setup()

    def tearDown(self):
        self.t04_teardown()
        for pt in reversed(self.patches):
            pt.stop()
        ti.Tmp.tearDown(self)

    def refused(self, fn, *a):
        with self.assertRaises(SystemExit) as cm:
            fn(*a)
        return str(cm.exception)


class T04Prepare(T04Base):
    def test_prepare_04(self):
        self.r20_uploaded()
        e, out = self.prepare04()
        self.assertEqual((e["id"], e["status"], e["file"]), (2, "prepared", "results/test04/rank/submission_rank.csv"))
        self.assertEqual(e["method"], ol.T04_METHOD)
        self.assertFalse(e["fallback_used"])
        t4 = e["test04"]
        self.assertEqual(t4["dkappa"], {f"{s}.{c}": v for s, d in DKAPPA.items() for c, v in d.items()})
        self.assertEqual(t4["replay_rate"], {"seam_val": 0.0, "public": 0.0, "private": 0.0})
        self.assertEqual(t4["r20_first_upload_id"], 1)
        self.assertEqual(t4["combo"], "Q02")
        self.assertEqual(t4["k6"], {"commit": REPORT_COMMIT[0], "committed_at": REPORT_COMMIT[1],
                                    "deadline": "2026-11-16T23:59:59+08:00", "ok": True})
        self.assertEqual(t4["report"], "results/test04/rank/t04_apply_report.json")
        self.assertEqual(t4["report_sha256"], sha(os.path.join(self.t04, ol.T04_REPORT)))
        self.assertEqual((t4["rule_b_differences_vs_r20"], t4["questions_compared_with_r20"]), (115, 800))
        self.assertEqual(t4["inputs_bound_checked"], len(ol.T04_INPUTS_REQUIRED))
        self.assertEqual(len(ol.T04_INPUTS_REQUIRED), 15)
        self.assertEqual(t4["seam_local_tol"], 1e-4)
        for rel in ("results/test04/rank/t04_apply_report.json", ol.T04_SEAM_LOCAL, ol.T04_OPP_REPORT,
                    ol.T04_SEAM_CHECK, "results/test02/rank/apply_test_report.json",
                    "results/test02/checks/rank/t1.json", "results/test02/checks/rank/summary.json"):
            self.assertIn(rel, e["reports_sha256"])
        for s in ("這一份是 04，不是 11/04 那份 R20", "兩份在 800 題裡有 115 題的規則 B 等級不同", "TEAM_11248",
                  sha(self.f04)[:12], "#2", "R20 的第一次上傳是 #1", "public.O4 +0.0930", "private 0.00%",
                  "2026-11-05T12:00:00+08:00"):
            self.assertIn(s, out)
        self.assertNotIn("prereg04_5_3", e)

    def test_needs_r20_first_upload_logged_as_uploaded(self):
        msg = self.refused(self.prepare04)
        self.assertIn("R20's first upload", msg)
        self.prepare()                                              # R20 prepared only
        msg = self.refused(self.prepare04)
        self.assertIn("R20's first upload", msg)
        self.assertEqual(len(self.log_obj()["uploads"]), 1)
        self.uploaded(1, "2026-11-05T14:30")
        e, _ = self.prepare04()
        self.assertEqual(e["test04"]["r20_first_upload_id"], 1)

    def test_replay_boundary(self):
        """PREREG_04 5.2 / spec 7: 25 of 5000 failed slots pass, 26 do not (per set); seam_val needs 0."""
        self.r20_uploaded()
        rp = self.report04()["replay"]
        rp["public"] = {"slots": 5000, "failures": 25}
        rp["private"] = {"slots": 5000, "failures": 25}
        self.write04(replay=rp)
        e, out = self.prepare04()
        self.assertEqual(e["test04"]["replay_rate"]["public"], 0.005)
        self.assertIn("public 0.50%", out)
        for s, bad in (("public", {"slots": 5000, "failures": 26}), ("private", {"slots": 5000, "failures": 26}),
                       ("seam_val", {"slots": 19156, "failures": 1, "equal_to_committed_val_o4": True}),
                       ("seam_val", {"slots": 19156, "failures": 96, "equal_to_committed_val_o4": False})):
            with self.subTest(set=s, bad=bad):
                rp2 = copy.deepcopy(rp)
                rp2[s] = bad
                self.write04(replay=rp2)
                msg = self.refused(self.prepare04)
                self.assertIn(f"O4 replay of {s}", msg)

    def refusal_cases(self):
        """(label, change, expected text): each makes exactly one condition fail."""
        def rep(**over):
            return lambda: self.write04(**over)

        def nb(**over):
            return lambda: self.edit04(lambda r: r["notebook"].update(over))

        def dk(s, src, v):
            return lambda: self.edit04(lambda r: r["colour_check"][s][src].update(dkappa=v))

        def rec(rel, **over):
            def f():
                with open(self.r(rel), encoding="utf-8") as fh:
                    obj = json.load(fh)
                obj.update(over)
                write_json(self.r(rel), obj)
                self.write04()                                      # rebind: only the record's content is wrong
            return f

        def t02(**kw):
            def f():
                self.write_reports(**kw)
                self.write04()
            return f

        def dirty(rel):
            return lambda: self.dirty.add(rel)

        cases = [
            ("T1 FAIL_OPENING", t02(rank={"fallback_used": False}, t1=("FAIL_OPENING", "PASS")), "FAIL_OPENING"),
            ("T1 FAIL_STRUCT", t02(t1=("FAIL_STRUCT", "PASS")), "FAIL_STRUCT"),
            ("T1 structure only", t02(t1_extra={"complete": False, "subchecks_run": ["T1.1"]}), "T1 ran only"),
            ("test02 R10 fallback", t02(rank={"fallback_used": True}), "fallback_used True"),
            ("test02 local seam", t02(rank={"seam_local": {"passed": False}}), "04 needs test02's passed"),
            ("test02 mode", t02(rank={"mode": "rehearsal"}), "04 needs test02's passed"),
            ("format", rep(format="c10-test02 rank apply v1"), "format"),
            ("mode", rep(mode="dry-run"), "mode 'dry-run'"),
            ("combo", rep(combo={"id": "Q01", "sources": ["C5", "OP", "O4"]}), "not the adopted Q02"),
            ("local seam", rep(seam_local={"passed": False}), "local Q02 seam did not pass"),
            ("check_submission errors", rep(check_submission={"mode": "official", "errors": ["row 3"]}),
             "check_submission"),
            ("check_submission missing", rep(check_submission=None), "check_submission"),
            ("outputs sha", rep(outputs_sha256={"submission_rank.csv": "0" * 64}), "records another"),
            ("seam_val not byte-identical",
             lambda: self.edit04(lambda r: r["replay"]["seam_val"].update(equal_to_committed_val_o4=False)),
             "not byte-identical"),
            ("no colour check for private", lambda: self.edit04(lambda r: r["colour_check"].pop("private")),
             "dkappa of O4 on private is None"),
            ("report not committed", dirty("results/test04/rank/t04_apply_report.json"), "committed and unchanged"),
            ("report untracked", lambda: self.untracked.add("results/test04/rank/t04_apply_report.json"),
             "committed and unchanged"),
            ("K6", lambda: setattr(self, "commit", ("b" * 40, "2026-11-17T00:00:00+08:00")), "not by K6"),
            ("never committed", lambda: setattr(self, "commit", (None, None)), "not by K6"),
            ("seam_local.json failed", rec(ol.T04_SEAM_LOCAL, passed=False), "seam_local.json"),
            ("seam_local.json mode", rec(ol.T04_SEAM_LOCAL, mode="dry-run"), "seam_local.json"),
            ("seam_local.json not committed", dirty(ol.T04_SEAM_LOCAL), "committed and unchanged"),
            ("opp not usable", rec(ol.T04_OPP_REPORT, usable_so_far=False), "opp_report.json"),
            ("opp failed", rec(ol.T04_OPP_REPORT, passed=False), "opp_report.json"),
            ("opp mode", rec(ol.T04_OPP_REPORT, mode="rehearsal"), "opp_report.json"),
            ("opp not committed", dirty(ol.T04_OPP_REPORT), "committed and unchanged"),
            ("sw seam_check failed", rec(ol.T04_SEAM_CHECK, passed=False), "seam_check.json"),
            ("sw seam_check not committed", dirty(ol.T04_SEAM_CHECK), "committed and unchanged"),
            ("inputs_bound missing", rep(inputs_bound=None), "輸入已變更"),
            ("inputs_bound lacks a key", lambda: self.edit04(
                lambda r: r["inputs_bound"].pop("results/test02/rank/opening_test.npz")), "inputs_bound lacks"),
            ("inputs_bound outside", lambda: self.edit04(lambda r: r["inputs_bound"].update({"../x.json": "0"})),
             "not a repo-relative path"),
            ("opening changed", lambda: open(self.r("results/test02/rank/opening_test.npz"), "ab").write(b"x"),
             "results/test02/rank/opening_test.npz is"),
            ("input not committed", dirty("results/test02/rank/public/dumps/val_p02e9_d4.npz"),
             "is not committed and clean"),
            ("two counted archives", lambda: (self.archive(1), self.archive(2)), "counted reruns"),
            ("archived decision without revision", lambda: self.archive(1, counts=False, with_report=True),
             "without a committed PREREG_04 revision"),
            ("unreadable archive", lambda: os.makedirs(self.r("results/test04/failed/attempt_1")),
             "attempt.json is missing"),
        ]
        for s in ("public", "private"):
            for src in ("O4", "SWA", "SWB"):
                for v in (0.0, -0.01, None, float("nan")):
                    cases.append((f"dkappa {s}.{src}={v}", dk(s, src, v), f"dkappa of {src} on {s}"))
        for k in ol.T04_NOTEBOOK_TRUE:
            cases.append((f"notebook {k}", nb(**{k: False}), k))
        cases += [("notebook keys", nb(keys_equal_to_test02={"seam_val": True, "public": False, "private": True}),
                   "keys_equal_to_test02"),
                  ("notebook earlier runs", nb(earlier_failed_runs=[{"run": 1, "status": "seam_failed"}]),
                   "earlier_failed_runs"),
                  ("notebook record missing", rep(notebook=None), "seam_check_passed")]
        return cases

    def test_prepare_04_refusals(self):
        self.r20_uploaded()
        for label, change, text in self.refusal_cases():
            with self.subTest(label):
                self.reset_fixture()
                change()
                msg = self.refused(self.prepare04)
                self.assertIn(text, msg)
                self.assertEqual(len(self.log_obj()["uploads"]), 1, "nothing appended")
        self.reset_fixture()
        e, _ = self.prepare04()                                     # the fixture itself is usable
        self.assertEqual(e["id"], 2)

    def reset_fixture(self):
        """Back to the usable fixture between subtests."""
        self.dirty.clear()
        self.untracked.clear()
        self.commit = REPORT_COMMIT
        shutil.rmtree(self.r("results/test04/failed"), ignore_errors=True)
        self.write_reports()
        self.write_t04_inputs()
        self.write04()

    def test_archive_with_a_committed_bug_revision(self):
        self.r20_uploaded()
        self.archive(1, counts=False, with_report=True, bug_revision={"rev": 4, "commit": "e" * 40})
        msg = self.refused(self.prepare04)
        self.assertIn("without a committed PREREG_04 revision", msg)       # HEAD has no rev 4 row yet
        self.p04_head = head_with_rows(4)
        self.rev_added["| rev 4 |"] = ("e" * 40, "2026-11-05T10:00:00+08:00")   # after the archived run (09:30)
        self.archive(2)                                                   # one counted rerun is allowed
        e, _ = self.prepare04()
        self.assertEqual(e["id"], 2)

    def test_bug_revision_archives(self):
        """Review round 2 (rev 3 item 20): an uncounted archive is before_r20_u1 or bug_revision; a bug revision's row
        was committed after the archived run; one revision exempts one rerun only."""
        self.r20_uploaded()
        self.p04_head = head_with_rows(4, 5)
        self.rev_added["| rev 4 |"] = ("e" * 40, "2026-11-05T10:00:00+08:00")
        self.rev_added["| rev 5 |"] = ("f" * 40, "2026-11-05T11:00:00+08:00")
        cases = [
            ("revision before the archived run", lambda: (
                self.rev_added.update({"| rev 4 |": ("e" * 40, "2026-11-05T09:15:00+08:00")}),
                self.archive(1, counts=False, bug_revision={"n": 4, "commit": "e" * 40})), "not after the archived run"),
            ("revision at a marker time", lambda: (
                self.rev_added.update({"| rev 4 |": ("e" * 40, self.ATTEMPT_DONE)}),
                self.archive(1, counts=False, bug_revision={"n": 4, "commit": "e" * 40})), "not after the archived run"),
            ("same revision twice", lambda: (
                self.archive(1, counts=False, bug_revision={"n": 4}), self.archive(2, counts=False, bug_revision={"n": 4})),
             "is cited by 2 archives"),
            ("no row at HEAD", lambda: self.archive(1, counts=False, bug_revision={"n": 6}),
             "is not a committed PREREG_04 revision"),
            ("bug_revision class without a revision", lambda: self.archive(1, counts=False, failure_class="bug_revision"),
             "is not a committed PREREG_04 revision"),
            ("uncounted seam failure", lambda: self.archive(1, counts=False, failure_class="notebook_seam"),
             "counts_toward_rerun false with failure_class 'notebook_seam'"),
            ("no time of the archived run", lambda: (
                self.archive(1, counts=False, bug_revision={"n": 4}),
                self.edit_attempt(1, lambda a: a.update(runs=[], done_times={}))), "records no time"),
        ]
        for label, change, text in cases:
            with self.subTest(label):
                shutil.rmtree(self.r("results/test04/failed"), ignore_errors=True)
                self.rev_added["| rev 4 |"] = ("e" * 40, "2026-11-05T10:00:00+08:00")
                change()
                msg = self.refused(self.prepare04)
                self.assertIn(text, msg)
                self.assertEqual(len(self.log_obj()["uploads"]), 1)
        shutil.rmtree(self.r("results/test04/failed"), ignore_errors=True)
        self.rev_added["| rev 4 |"] = ("e" * 40, "2026-11-05T10:00:00+08:00")
        overturned = "2026-11-05T09:45:00+08:00"          # a time t04_rank archive recorded as overturned
        self.archive(1, counts=False, bug_revision={"n": 4, "overturns": [{"what": "x", "time": overturned}]})
        self.archive(2, counts=False, bug_revision={"n": 5})
        self.archive(3)                                                   # the one counted rerun
        e, _ = self.prepare04()
        self.assertEqual(e["id"], 2)
        self.rev_added["| rev 4 |"] = ("e" * 40, "2026-11-05T09:40:00+08:00")      # before the overturned time
        self.assertIn("not after the archived run", "; ".join(ol.t04_archive_problems()))

    def edit_attempt(self, n, fn):
        p = self.r(f"results/test04/failed/attempt_{n}/attempt.json")
        with open(p, encoding="utf-8") as f:
            att = json.load(f)
        fn(att)
        write_json(p, att)

    def test_archive_without_counts_toward_rerun_counts(self):
        """Review round 2: an attempt.json without counts_toward_rerun counts (the safe side, as t04_rank is_counted):
        with a second counted archive the 04 file is refused."""
        self.r20_uploaded()
        self.archive(1, counts=None)
        self.assertNotIn("counts_toward_rerun", json.load(open(self.r("results/test04/failed/attempt_1/attempt.json"),
                                                               encoding="utf-8")))
        e, _ = self.prepare04()                                          # one counted archive: allowed
        self.assertEqual(e["id"], 2)
        self.archive(2)
        msg = self.refused(self.prepare04)
        self.assertIn("2 counted reruns archived", msg)

    def test_rerun_attempt_must_be_on_g4(self):
        """Review round 1 (rev 3 items 4, 9): a report of attempt 2 (the registered rerun) backs an upload only when
        its notebook ran on G4 (config and the markers' environment); attempt 1 may have run on A100. Review round 2:
        notebook.env is the flat record tools/t04_rank.py writes; a nested per-marker record is read too."""
        self.r20_uploaded()
        g4, a100 = ol.T04_RERUN_GPU, "NVIDIA A100-SXM4-80GB"
        att2 = {"number": 2, "counted_archives": ["attempt_1"], "bug_revisions": []}
        flat = self.report04()["notebook"]["env"]
        for label, gpu, env in (("A100 config", a100, flat),
                                ("A100 markers (flat, as t04_rank writes)", g4, dict(flat, gpu_name=a100)),
                                ("no marker environment", g4, None),
                                ("one A100 marker (nested)", g4, {"seam_val/val_keys": {"gpu_name": g4},
                                                                  "public/val_p03a_d4": {"gpu_name": a100}})):
            with self.subTest(label):
                nb = dict(self.report04()["notebook"], gpu=gpu)
                if env is None:
                    nb.pop("env")
                else:
                    nb["env"] = env
                self.write04(attempt=att2, notebook=nb)
                msg = self.refused(self.prepare04)
                self.assertIn("not only on G4", msg)
        first_on_a100 = self.report04(attempt=dict(att2, number=1),
                                      notebook=dict(self.report04()["notebook"], gpu=a100))
        self.assertEqual(ol.t04_report_problems(first_on_a100, "r", self.f04, sha(self.f04))[1], [])
        self.write04(attempt=att2)
        e, _ = self.prepare04()
        self.assertEqual(e["test04"]["notebook_gpu"], g4)

    def test_method(self):
        self.r20_uploaded()
        msg = self.refused(self.prepare04, "--method", "prereg04 04")
        self.assertIn("does not name Q02", msg)
        msg = self.refused(self.prepare04, "--method", "prereg04 R10 variant")
        self.assertIn("names R10", msg)
        e, _ = self.prepare04("--method", "prereg04 Q02 (04)")
        self.assertEqual(e["method"], "prereg04 Q02 (04)")
        self.assertEqual(ol.method_for_t04(None), ol.T04_METHOD)
        self.assertEqual(ol.method_for_t04("prereg04 Q02"), "prereg04 Q02")
        with self.assertRaises(SystemExit):
            ol.method_for_t04("prereg04 Q020 variant")

    def test_not_04_reason_only_for_r20_files(self):
        self.r20_uploaded()
        msg = self.refused(self.prepare04, "--not-04-reason", "PREREG_04 rev 4: x")
        self.assertIn("R20 / R10 file only", msg)
        msg = self.refused(self.prepare, "player", self.player_file, "U1", "--not-04-reason", "PREREG_04 rev 4: x")
        self.assertIn("R20 / R10 file only", msg)

    def test_a_04_file_is_recognised_by_its_folder_only(self):
        self.assertTrue(ol.is_t04_file("rank", self.f04))
        self.assertFalse(ol.is_t04_file("player", self.f04))
        self.assertFalse(ol.is_t04_file("rank", self.rank_file))
        other = ti.write_csv(self.r("results/test04/failed/attempt_1/rank/submission_rank.csv"),
                             ["question_id", "pred_rank"], self.rank_rows)
        self.assertFalse(ol.is_t04_file("rank", other))
        self.r20_uploaded()
        msg = self.refused(self.prepare, "rank", other, "x")           # R20's path: no apply_test_report.json
        self.assertIn("apply_test_report.json", msg)


class T04Final(T04Base):
    def setUp(self):
        super().setUp()
        self.r20_uploaded()                                          # #1 R20 U1
        self.e04, _ = self.prepare04()                               # #2 04
        self.uploaded(2)

    def test_final_04(self):
        fc, out = self.run_ol("final", "--comp", "rank", "--id", "2", "--basis", "PREREG_04 5.3")
        r = fc["prereg04_5_3"]
        self.assertEqual((r["chosen"], r["eligible_04_entries"], r["not_eligible_04_entries"]), ("04", [2], {}))
        self.assertEqual(r["h4r"], ol.P04_H4R)
        self.assertIn("最終選擇是 04", out)
        self.assertNotIn("還沒上傳", out)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        p = st["rank"]["prereg04"]
        self.assertEqual((p["entries"], p["eligible"], p["final_is_04"], p["violation"], p["redo_r20"]),
                         ([2], [2], True, False, False))
        self.assertTrue(st["rank"]["last_is_final"])
        self.assertIn("最終選擇是 04", out)
        self.assertNotIn("違反", out)
        self.assertIn(K6_PREFIX, out)

    def test_final_r20_needs_a_committed_later_revision(self):
        for case in REVISION_CASES:
            with self.subTest(case["name"]):
                self.p04_head = head_with_rows(*case["rows"])
                row = f"| rev {case['rows'][0] if case['rows'] else 4} |"
                self.rev_added = {row: ("f" * 40, case["added"])} if case["added"] else {}
                self.dirty = set() if case["clean"] else {ol.P04_DOC}
                argv = ["final", "--comp", "rank", "--id", "1", "--basis", "R20 after a 04 bug",
                        "--not-04-reason", case["reason"]]
                if case["ok"]:
                    fc, out = self.run_ol(*argv)
                    r = fc["prereg04_5_3"]
                    self.assertEqual((r["chosen"], r["eligible_04_entries"]), ("R20", [2]))
                    self.assertEqual(r["not_04_reason"], case["reason"])
                    self.assertEqual({k: r["revision"][k] for k in ("rev", "row", "commit", "committed_at")},
                                     {"rev": 4, "row": "| rev 4 |", "commit": "f" * 40,
                                      "committed_at": case["added"]})
                    self.assertEqual(r["revision"]["overridden_at"], [NOW])
                    self.assertIn("依 PREREG_04 rev 4", out)
                else:
                    msg = self.refused(self.run_ol, *argv)
                    self.assertIn(case["msg"], msg)
                    self.assertIsNone(self.log_obj()["final_choice"]["rank"])
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", "1", "--basis", "x")
        self.assertIn("--not-04-reason", msg)

    def test_revision_cases_direct(self):
        """The shared fixture REVISION_CASES against p04_revision itself."""
        for case in REVISION_CASES:
            with self.subTest(case["name"]):
                self.p04_head = head_with_rows(*case["rows"])
                row = f"| rev {case['rows'][0] if case['rows'] else 4} |"
                self.rev_added = {row: ("f" * 40, case["added"])} if case["added"] else {}
                self.dirty = set() if case["clean"] else {ol.P04_DOC}
                if case["ok"]:
                    self.assertEqual(ol.p04_revision(case["reason"], [RECORD_AT], "t")["rev"], 4)
                else:
                    msg = self.refused(ol.p04_revision, case["reason"], [RECORD_AT], "t")
                    self.assertIn(case["msg"], msg)

    def test_final_04_refused_when_it_no_longer_holds(self):
        cases = [
            ("report changed", lambda: self.edit04(lambda r: r.update(created="later")), "changed since prepare"),
            ("report re-committed", lambda: setattr(self, "commit", ("b" * 40, REPORT_COMMIT[1])), "K6"),
            ("file changed", lambda: open(self.f04, "ab").write(b"\n"), "changed since prepare"),
            ("test02 report replaced", lambda: self.write_reports(rank={"note": "re-run after a fix"}), "輸入已變更"),
            ("opening replaced", lambda: open(self.r("results/test02/rank/opening_test.npz"), "wb").write(b"new"),
             "輸入已變更"),
            ("two counted archives", lambda: (self.archive(1), self.archive(2)), "counted reruns"),
        ]
        for label, change, text in cases:
            with self.subTest(label):
                snap = self.snapshot()
                change()
                ok, why = ol.t04_entry_eligible(self.log_obj()["uploads"][1])
                self.assertFalse(ok)
                self.assertIn(text, "; ".join(why))
                msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", "2", "--basis", "5.3")
                self.assertIn("does not meet PREREG_04 5.3", msg)
                self.assertIn(text, msg)
                fc, _ = self.run_ol("final", "--comp", "rank", "--id", "1", "--basis", "PREREG_04 5.3: 04 fails")
                self.assertEqual(fc["prereg04_5_3"]["chosen"], "R20")
                self.assertEqual(fc["prereg04_5_3"]["eligible_04_entries"], [])
                self.assertNotIn("revision", fc["prereg04_5_3"])
                st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
                self.assertTrue(st["rank"]["prereg04"]["redo_r20"])          # the last upload is the 04 one
                self.assertIn("final 改 R20，再 prepare 並重傳 R20", out)
                self.restore(snap)

    def test_notebook_record_rechecked(self):
        """An entry whose (recorded) report has a notebook field false or earlier failed runs never holds."""
        x = self.log_obj()["uploads"][1]
        for field, value in (("env_same", False), ("earlier_failed_runs", [{"run": 1}])):
            with self.subTest(field):
                snap = self.snapshot()
                self.edit04(lambda r: r["notebook"].update({field: value}))
                y = copy.deepcopy(x)
                y["test04"]["report_sha256"] = sha(os.path.join(self.t04, ol.T04_REPORT))
                ok, why = ol.t04_entry_eligible(y)
                self.assertFalse(ok)
                self.assertIn(field, "; ".join(why))
                self.restore(snap)
        self.assertEqual(ol.t04_entry_eligible(x), (True, []))

    def test_final_prepared_entry_says_not_uploaded(self):
        e, _ = self.prepare04()                                      # #3, same file, not uploaded
        _fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e["id"]), "--basis", "5.3")
        self.assertIn("這一筆還沒上傳", out)
        st, _ = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertEqual(st["rank"]["prepared_not_uploaded"], [3])

    def test_prepare_r20_while_04_holds(self):
        msg = self.refused(self.prepare, "rank", None, "final re-upload of R20")
        self.assertIn("meet PREREG_04 5.3", msg)
        self.assertIn("--not-04-reason", msg)
        msg = self.refused(self.prepare, "rank", None, "x", "--not-04-reason", "the 04 path had a bug")
        self.assertIn("without a revision", msg)
        self.p04_head = head_with_rows(4)
        self.rev_added = {"| rev 4 |": ("f" * 40, "2026-11-06T10:00:00+08:00")}
        e, out = self.prepare("rank", None, "R20 per PREREG_04 rev 4", "--not-04-reason", "PREREG_04 rev 4: bug")
        self.assertEqual(e["prereg04_5_3"]["revision"]["rev"], 4)
        self.assertEqual(e["prereg04_5_3"]["eligible_04_entries"], [2])
        self.assertNotIn("test04", e)
        self.assertEqual(e["method"], ol.DEFAULT_METHOD["rank"])
        self.assertIn("依 PREREG_04 rev 4", out)

    def test_prepare_r20_when_no_04_holds(self):
        self.edit04(lambda r: r.update(created="later"))              # the 04 entry no longer holds
        e, out = self.prepare("rank", None, "final: R20 per PREREG_04 5.3 (04 report changed)")
        r = e["prereg04_5_3"]
        self.assertEqual((r["eligible_04_entries"], list(r["not_eligible_04_entries"])), ([], ["2"]))
        self.assertNotIn("revision", r)
        self.assertNotIn("依 PREREG_04", out)

    def snapshot(self):
        keep = {}
        for root, _d, files in os.walk(self.repo):
            for f in files:
                p = os.path.join(root, f)
                with open(p, "rb") as fh:
                    keep[p] = fh.read()
        return keep, self.commit, set(self.dirty)

    def restore(self, snap):
        keep, self.commit, self.dirty = snap[0], snap[1], set(snap[2])
        for root, _d, files in os.walk(self.repo):
            for f in files:
                p = os.path.join(root, f)
                if p not in keep:
                    os.remove(p)
        for p, data in keep.items():
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(data)
        shutil.rmtree(self.r("results/test04/failed"), ignore_errors=True)


class T04Status(T04Base):
    def test_violation_when_final_is_r20_without_revision(self):
        self.r20_uploaded()
        self.run_ol("final", "--comp", "rank", "--id", "1", "--basis", "prereg02 (甲), U5 of docs/TESTDAY.md")
        self.prepare04()
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertIn("違反 PREREG_04 5.3", out)
        self.assertTrue(st["rank"]["prereg04"]["violation"])
        self.uploaded(2)
        self.run_ol("final", "--comp", "rank", "--id", "2", "--basis", "PREREG_04 5.3")
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertNotIn("違反", out)
        self.assertFalse(st["rank"]["prereg04"]["violation"])

    def test_revision_does_not_cover_a_later_04(self):
        """Review round 1: an R20 final set with PREREG_04 rev 4 overrides only the 04 entries that revision covered;
        a new eligible 04 entry prepared after it is a violation again (final refuses the same revision too)."""
        self.r20_uploaded()                                          # #1
        self.prepare04()                                             # #2
        self.uploaded(2)
        self.p04_head = head_with_rows(4)
        self.rev_added = {"| rev 4 |": ("f" * 40, "2026-11-06T10:00:00+08:00")}
        self.run_ol("final", "--comp", "rank", "--id", "1", "--basis", "R20 after a 04 bug",
                    "--not-04-reason", "PREREG_04 rev 4: bug in the 04 path")
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertEqual((st["rank"]["prereg04"]["eligible"], st["rank"]["prereg04"]["violation"]), ([2], False))
        self.assertNotIn("違反", out)
        with mock.patch.object(ol, "now_iso", return_value="2026-11-07T02:00:00+00:00"):     # after rev 4
            self.prepare04()                                         # #3: the 04 run after the fix
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        p = st["rank"]["prereg04"]
        self.assertEqual((p["eligible"], p["final_is_04"], p["violation"]), ([2, 3], False, True))
        self.assertIn("違反 PREREG_04 5.3", out)
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", "1", "--basis", "R20 again",
                           "--not-04-reason", "PREREG_04 rev 4: bug in the 04 path")
        self.assertIn("not after the records", msg)

    def test_final_not_set_yet(self):
        self.r20_uploaded()
        self.prepare04()
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertFalse(st["rank"]["prereg04"]["violation"])
        self.assertIn("最終選擇要設成 04", out)
        self.assertIn("04（PREREG_04 5.3）：紀錄 [2]；現在符合條件的 [2]", out)

    def test_final_04_that_stopped_holding(self):
        self.r20_uploaded()
        self.prepare04()
        self.uploaded(2)
        self.run_ol("final", "--comp", "rank", "--id", "2", "--basis", "PREREG_04 5.3")
        with open(self.r("results/test02/rank/opening_test.npz"), "ab") as f:    # R20 fixed after U1
            f.write(b"again")
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        p = st["rank"]["prereg04"]
        self.assertEqual((p["eligible"], p["final_is_04"], p["redo_r20"], p["violation"]), ([], True, True, False))
        self.assertIn("#2 不符合", out)
        self.assertIn("輸入已變更", out)
        self.assertIn("final 改 R20，再 prepare 並重傳 R20", out)

    def test_status_without_04_entries(self):
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertEqual(st["rank"]["prereg04"], {"entries": [], "eligible": [], "not_eligible": {},
                                                  "final_set": False, "final_is_04": False, "violation": False,
                                                  "redo_r20": False})
        self.assertNotIn("PREREG_04 5.3", out)
        self.assertEqual(out.count(K6_PREFIX), 1)


class T04Public(T04Base):
    def test_same_score_warning_both_ways(self):
        self.r20_uploaded()
        self.run_ol("public", "--id", "1", "--score", "0.6012")
        self.prepare04()
        self.uploaded(2)
        _e, out = self.run_ol("public", "--id", "2", "--score", "0.6012")
        self.assertIn("可能傳成 R20 的檔（兩份同名）：核對 AIdea 的上傳紀錄", out)

    def test_reverse_and_no_warning(self):
        self.r20_uploaded()
        self.prepare04()
        self.uploaded(2)
        _e, out = self.run_ol("public", "--id", "2", "--score", "0.6310")
        self.assertNotIn("WARNING", out)
        _e, out = self.run_ol("public", "--id", "1", "--score", "0.6310")
        self.assertIn("可能傳成 04 的檔（兩份同名）", out)


if __name__ == "__main__":
    unittest.main()
