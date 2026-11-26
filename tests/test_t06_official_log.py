"""Tests for the PREREG_06 rev 3 additions to tools/official_log.py (results/prereg06/design/spec.md 8 and 9.3 item 5).

* TagVersusNew (spec 8, "不變的保證"): the version of tag p04-testday-pre06 (git show, executed as a separate module,
  nothing written into the repository) and the working-tree version run the same flows on the same fixture with the
  same mocks: competition 2's R20, R10 and 04 flows, each with competition 1's Q04 flow (prepare, uploaded, public,
  final, status, and the refusals). The log bytes, the returned records, the stdout and the refusal messages are
  identical, status included (the tag already prints the K6 line; a log without 06 entries adds nothing).
* Q04UnchangedWithT06Files: the same Q04 flow while a complete results/test06 tree sits next to results/test02 but the
  log has no 06 entry: no 06 field appears anywhere.
* T06*: the 06 path (prepare 06-OFF / 06-ON and the order rules, the refusals, --method, public with the same-file
  warning and d, final under PREREG_06 5.4 with the veto, K7' and --not-06-reason, status).
* T06Rev3 (review 10/3): a paused veto designates nothing before K7' (no re-upload prompt); 06-ON kept by d >= 0 but no
  longer the last upload gives 06-OFF; a wrong file found in the 06-OFF / 06-ON pair (or a check that cannot settle
  it) gives 06-OFF; a 06-OFF uploaded again before 06-ON is the one 06-ON pairs with; 06-ON's prepare needs the public
  scores of Q04 and 06-OFF without a same-file warning; a 06 upload whose bytes scored differently before is warned.
Synthetic files in a temporary directory only; git and the clock are mocked (ol.git_file_state, ol.git_commit_time,
ol.git_head_text, ol.git_row_added, ol.now_iso, ol.t06_now). Nothing here opens a label file or runs tools/score.py.
Run: python -m pytest tests/test_t06_official_log.py
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
from datetime import datetime, timedelta, timezone
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import official_log as ol  # noqa: E402
import t02_test_bundle as tb  # noqa: E402
import test_t02_infra as ti  # noqa: E402
import test_t04_official_log as t4  # noqa: E402

TAG = "p04-testday-pre06"
TW = timezone(timedelta(hours=8))
NOW = "2026-11-04T06:00:00+00:00"                 # prepared_at / checked_at of every entry here (14:00 Taiwan time)
CLOCK = datetime(2026, 11, 6, 12, 0, tzinfo=TW)   # ol.t06_now (K7' 11/22 23:59:59)
REPORT06_COMMIT = ("b" * 40, "2026-11-06T10:00:00+08:00")
VETO_COMMIT = ("e" * 40, "2026-11-07T10:00:00+08:00")
PREREG06_HEAD = ("# PREREG_06\n\n| rev 1 | `802eaea` | first |\n| rev 2 | `9d8245d` | second |\n"
                 "| rev 3 | (commit) | test day |\n")
DIFF_Q04 = {"off": {"public": 37, "private": 41}, "on": {"public": 52, "private": 55}}
DIFF_OFF_ON = {"public": 17, "private": 12}


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


def player_rows(ids, shift):
    return [[q] + [f"{q.split('_')[0]}_p_{(j + shift) % 400:04d}" for j in range(5)] for q in ids]


IDS = [f"pub_q_{i:04d}" for i in range(1, 401)] + [f"priv_q_{i:04d}" for i in range(1, 401)]


# ---------------------------------------------------------------- the results/test06 tree of a usable 06 run
def write_t06_tree(repo, run_id="run-1", with_on=True, report_over=None):
    """results/test06/player as tools/t06_player.py submit leaves it (real mode), plus the committed inputs it binds.
    Returns {variant: file path}."""
    r = lambda rel: os.path.join(repo, *rel.split("/"))   # noqa: E731
    write_json(r(ol.T06_SEAM), {"format": ol.T06_SEAM_FORMAT, "mode": "real", "passed": True, "verdict": "passed"})
    for p in ol.T06_POOL_REPORTS:
        write_json(r(p), {"format": ol.T06_POOL_FORMAT, "mode": "real", "used06": True})
    write_json(r("results/prereg06/player/frozen_config.json"), {"k_star": 10})
    write_json(r("results/prereg06/player/score_H6P_switch.json"), {"B_minus_A": {"mean_diff": 0.0125}})
    for p in ("seam_val", "public", "private"):
        for n in ("cand_games", "query_games"):
            rel = f"results/test02/player/{p}/dumps/{n}.npz"
            os.makedirs(os.path.dirname(r(rel)), exist_ok=True)
            with open(r(rel), "wb") as f:
                f.write(b"PK-fake-npz " + rel.encode() + b"\r\n")
    files = {}
    for v, shift in (("off", 5), ("on", 9)):
        if v == "on" and not with_on:
            continue
        files[v] = ti.write_csv(r(f"results/test06/player/{v}/submission_player.csv"),
                                ["question_id"] + [f"top_{i}" for i in range(1, 6)], player_rows(IDS, shift))
    for v, f in files.items():
        rep = report06(repo, v, f, run_id)
        rep.update((report_over or {}).get(v, {}))
        write_json(r(f"results/test06/player/{v}/{ol.T06_REPORT}"), rep)
    return files


def report06(repo, v, f, run_id):
    return {
        "format": ol.T06_FORMAT, "mode": "real", "variant": v, "method": ol.T06_METHOD[v], "run_id": run_id,
        "on_conditions": {k: {"ok": True, "detail": None} for k in ol.T06_ON_CONDITIONS}, "all_on_conditions": True,
        "on_produced": True, "check_submission": {"official": True, "passed": True, "errors": []},
        "output": {"path": f"results/test06/player/{v}/submission_player.csv", "sha256": sha(f)},
        "top1_differences_vs_q04": DIFF_Q04[v], "top1_differences_off_vs_on": DIFF_OFF_ON,
        "seam": {"passed": True}, "h6p": {"adopted": True, "secondary_not_negative": True},
        "inputs_bound": {rel: ol.sha256_bound(os.path.join(repo, *rel.split("/"))) for rel in ol.T06_INPUTS_REQUIRED}}


# ---------------------------------------------------------------- the tagged version against the new one
def load_tagged_official_log():
    """tools/official_log.py of tag p04-testday-pre06 as a separate module (never written into the repository)."""
    r = subprocess.run(["git", "-C", REPO, "show", f"{TAG}:tools/official_log.py"], capture_output=True, timeout=60)
    if r.returncode != 0:
        raise AssertionError(f"local tag {TAG} missing or without tools/official_log.py")
    spec = importlib.util.spec_from_loader("official_log_p04_tag", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.join(REPO, "tools", "official_log.py")
    exec(compile(r.stdout.decode("utf-8"), f"<{TAG}:tools/official_log.py>", "exec"), mod.__dict__)
    return mod


class _Fx04(t4.T04Tree):
    """T04Tree's file writers without its patches (the flows below patch the module under test themselves)."""

    def __init__(self, repo, rank_rows):
        self.repo, self.rank_rows = repo, rank_rows


class TagVersusNew(unittest.TestCase):
    """Spec 8: with no 06 entry in the log, prepare / uploaded / public / final / status give the same log bytes,
    returned values, stdout and refusals in the tag p04-testday-pre06 version and the new one: competition 2's R20,
    R10 and 04 flows, each with competition 1's Q04 flow, with and without a results/test06 tree on disk."""

    @classmethod
    def setUpClass(cls):
        cls.tag = load_tagged_official_log()
        assert hasattr(cls.tag, "check_reports_t04") and not hasattr(cls.tag, "check_reports_t06"), "not the tag"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c10_t06_ol_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, kind, with_t06):
        repo = os.path.join(self.tmp, "repo")
        shutil.rmtree(repo, ignore_errors=True)
        tests = ti.write_official(os.path.join(repo, "data", "test"))
        files = {n: tb.file_record(os.path.join(tests, n)) for n in tb.ALL_NAMES}
        tb.write_manifest(tests, "official", files, {"rank": "PASS", "player": "PASS"})
        rank_rows = [[q, "D"] for q in IDS]
        rank_file = ti.write_csv(os.path.join(repo, "results", "test02", "rank", "submission_rank.csv"),
                                 ["question_id", "pred_rank"], rank_rows)
        player_file = ti.write_csv(os.path.join(repo, "results", "test02", "player", "submission_player.csv"),
                                   ["question_id"] + [f"top_{i}" for i in range(1, 6)], player_rows(IDS, 0))
        R = os.path.join(repo, "results", "test02")
        r10 = kind == "r10"
        write_json(os.path.join(R, "rank", "apply_test_report.json"), {
            "mode": "real", "seam_local": {"passed": True}, "fallback_used": r10,
            "check_submission": {"mode": "official", "errors": []},
            "outputs_sha256": {"submission_rank.csv": sha(rank_file)}})
        write_json(os.path.join(R, "player", "submit_report.json"), {
            "mode": "real", "check_submission": {"official": True, "passed": True, "errors": []},
            "output": {"sha256": sha(player_file)}})
        for comp in ("rank", "player"):
            write_json(os.path.join(R, comp, "seam_val", "seam_local.json"), {"passed": True, "mode": "real"})
            write_json(os.path.join(R, "checks", comp, "t1.json"), {
                "verdict": "FAIL_OPENING" if (r10 and comp == "rank") else "PASS", "comp": comp, "complete": True,
                "stage": "raw", "subchecks_run": ["T1.1", "T1.2", "T1.3", "T1.4", "T1.5"],
                "provenance": {"mode": "real"}})
            checks = {}
            for c, verdict in (("T2", "PASS"), ("T3", "RECORD_ONLY"), ("T4", "PASS"), ("T5", "RECORD_ONLY")):
                dp = write_json(os.path.join(R, "checks", comp, c.lower() + ".json"), {"verdict": verdict})
                checks[c] = {"verdict": verdict, "details_file": c.lower() + ".json", "sha256": sha(dp)}
            write_json(os.path.join(R, "checks", comp, "summary.json"), {"checks": checks})
        f04 = None
        if kind == "04":
            fx = _Fx04(repo, rank_rows)
            fx.write_t04_inputs()
            fx.write04()
            f04 = fx.f04
        if with_t06:
            write_t06_tree(repo)
        return repo, tests, rank_file, player_file, f04

    def flow(self, mod, kind, with_t06):
        repo, tests, rank_file, player_file, f04 = self.build(kind, with_t06)
        log = os.path.join(repo, "results", "official_scores.json")
        state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
        patches = [mock.patch.object(mod, "REPO", repo),
                   mock.patch.object(mod, "git_file_state", side_effect=lambda rel: dict(state)),
                   mock.patch.object(mod, "now_iso", return_value="2026-11-04T06:00:00+00:00"),
                   mock.patch.object(mod, "git_commit_time", side_effect=lambda rel: t4.REPORT_COMMIT),
                   mock.patch.object(mod, "git_head_text", side_effect=lambda rel: t4.PREREG04_HEAD),
                   mock.patch.object(mod, "git_row_added", side_effect=lambda rel, row: (None, None))]
        if hasattr(mod, "t06_now"):
            patches.append(mock.patch.object(mod, "t06_now", side_effect=AssertionError("t06_now on a log without "
                                                                                         "06 entries")))
        rank = [("prepare rank U1", ["prepare", "--comp", "rank", "--file", rank_file, "--reason", "first upload (U1)",
                                     "--tests-dir", tests]),
                ("prepare rank other name", ["prepare", "--comp", "rank", "--file", rank_file, "--reason", "x",
                                             "--method", "prereg02 R20 rule B" if kind == "r10" else
                                             "prereg02 R10 fallback", "--tests-dir", tests]),
                ("uploaded before ok", ["uploaded", "--id", "1", "--ok-at", "2026-11-04T15:00",
                                        "--uploaded-at", "2026-11-04T14:00"]),
                ("uploaded 1", ["uploaded", "--id", "1", "--ok-at", "2026-11-04T15:00", "--uploaded-at",
                                "2026-11-04T15:05"]),
                ("public 1", ["public", "--id", "1", "--score", "0.61", "--lb-rank", "12", "--note", "U1"])]
        player = [("prepare player wrong name", ["prepare", "--comp", "player", "--file", rank_file, "--reason", "x",
                                                 "--tests-dir", tests]),
                  ("prepare player", ["prepare", "--comp", "player", "--file", player_file, "--reason",
                                      "first upload (U1)", "--tests-dir", tests]),
                  ("uploaded 2", ["uploaded", "--id", "2", "--ok-at", "2026-11-04T16:00", "--uploaded-at",
                                  "2026-11-04T16:02"]),
                  ("uploaded 2 again", ["uploaded", "--id", "2", "--ok-at", "2026-11-04T16:00", "--uploaded-at",
                                        "2026-11-04T16:02"]),
                  ("public 2", ["public", "--id", "2", "--score", "0.88"]),
                  ("prepare player again", ["prepare", "--comp", "player", "--file", player_file, "--reason",
                                            "re-upload (U5)", "--tests-dir", tests]),
                  ("uploaded 3", ["uploaded", "--id", "3", "--ok-at", "2026-11-05T09:00", "--uploaded-at",
                                  "2026-11-05T09:01"]),
                  ("public 3", ["public", "--id", "3", "--score", "0.82"]),
                  ("public 3 again", ["public", "--id", "3", "--score", "0.5"])]
        tail = []
        if kind == "04":
            tail = [("prepare 04", ["prepare", "--comp", "rank", "--file", f04, "--reason",
                                    "prereg04 Q02 (PREREG_04 5.3)", "--tests-dir", tests]),
                    ("uploaded 4", ["uploaded", "--id", "4", "--ok-at", "2026-11-05T15:10", "--uploaded-at",
                                    "2026-11-05T15:10"]),
                    ("public 4", ["public", "--id", "4", "--score", "0.61"]),
                    ("final rank R20 refused", ["final", "--comp", "rank", "--id", "1", "--basis", "prereg02"]),
                    ("final rank 04", ["final", "--comp", "rank", "--id", "4", "--basis", "PREREG_04 5.3"])]
        else:
            tail = [("final rank", ["final", "--comp", "rank", "--id", "1", "--basis", "prereg02 (甲)"])]
        tail += [("final wrong comp", ["final", "--comp", "rank", "--id", "2", "--basis", "x"]),
                 ("final player", ["final", "--comp", "player", "--id", "2", "--basis", "prereg02 (甲) U5"]),
                 ("final player again", ["final", "--comp", "player", "--id", "3", "--basis", "U5, same file"]),
                 ("status", ["status", "--now", "2026-11-10T00:00"]),
                 ("status late", ["status", "--now", "2026-11-30T00:00"])]
        out = []
        for pt in patches:
            pt.start()
        try:
            for name, argv in rank + player + tail:
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

    def compare(self, kind, with_t06):
        old, new = self.flow(self.tag, kind, with_t06), self.flow(ol, kind, with_t06)
        self.assertEqual([s[0] for s in old], [s[0] for s in new])
        for (name, o_out, o_res, o_log), (_n, n_out, n_res, n_log) in zip(old, new):
            with self.subTest(step=name, kind=kind, with_t06=with_t06):
                self.assertEqual(n_log, o_log, "log bytes differ")
                self.assertEqual(n_out, o_out, "stdout differs")
                self.assertEqual(n_res, o_res, "returned record / refusal differs")
        return old

    def test_r20_and_q04_flows_identical(self):
        for with_t06 in (False, True):
            old = self.compare("r20", with_t06)
            refused = [s[0] for s in old if s[2].startswith('["refused"')]
            self.assertEqual(refused, ["prepare rank other name", "uploaded before ok", "prepare player wrong name",
                                       "uploaded 2 again", "public 3 again", "final wrong comp"])
            log = json.loads(old[-1][3])
            self.assertEqual(log["final_choice"]["player"]["id"], 3)
            self.assertFalse(any("test06" in e or "prereg06_5_4" in e for e in log["uploads"]))

    def test_r10_and_q04_flows_identical(self):
        for with_t06 in (False, True):
            old = self.compare("r10", with_t06)
            log = json.loads(old[-1][3])
            self.assertEqual([e["method"] for e in log["uploads"] if e["comp"] == "rank"], [ol.FALLBACK_METHOD])

    def test_04_and_q04_flows_identical(self):
        for with_t06 in (False, True):
            old = self.compare("04", with_t06)
            log = json.loads(old[-1][3])
            self.assertEqual(log["final_choice"]["rank"]["id"], 4)
            self.assertEqual(log["final_choice"]["rank"]["prereg04_5_3"]["chosen"], "04")
            refused = [s[0] for s in old if s[2].startswith('["refused"')]
            self.assertIn("final rank R20 refused", refused)


# ---------------------------------------------------------------- the 06 path
class T06Base(ti.Tmp):
    """ti.OfficialLog's fixture (data/test, test02's reports, Q04's file) plus a usable results/test06 run."""
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
        self.rank_rows = [[q, "D"] for q in IDS]
        self.rank_file = ti.write_csv(os.path.join(self.repo, "results", "test02", "rank", "submission_rank.csv"),
                                      ["question_id", "pred_rank"], self.rank_rows)
        self.player_file = ti.write_csv(
            os.path.join(self.repo, "results", "test02", "player", "submission_player.csv"),
            ["question_id"] + [f"top_{i}" for i in range(1, 6)], player_rows(IDS, 0))
        self.log = self.p("repo", "results", "official_scores.json")
        self.state = {"tracked": True, "clean": True, "head": "c" * 40, "file_commit": "d" * 40}
        self.write_reports()
        self.f06 = write_t06_tree(self.repo)
        self.dirty, self.untracked = set(), set()
        self.commits = {}
        self.p06_head = PREREG06_HEAD
        self.rev_added = {}
        self.clock = CLOCK
        self.patches = [
            mock.patch.object(ol, "REPO", self.repo),
            mock.patch.object(ol, "git_file_state", side_effect=lambda rel: dict(
                self.state, clean=rel not in self.dirty, tracked=rel not in self.untracked)),
            mock.patch.object(ol, "git_commit_time", side_effect=lambda rel: self.commits.get(
                rel, VETO_COMMIT if rel == ol.T06_VETO else REPORT06_COMMIT)),
            mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p06_head if rel == ol.P06_DOC
                              else t4.PREREG04_HEAD),
            mock.patch.object(ol, "git_row_added", side_effect=lambda rel, row: self.rev_added.get(row, (None, None))),
            mock.patch.object(ol, "now_iso", return_value=NOW),
            mock.patch.object(ol, "t06_now", side_effect=lambda: self.clock)]
        for pt in self.patches:
            pt.start()

    def tearDown(self):
        for pt in reversed(self.patches):
            pt.stop()
        ti.Tmp.tearDown(self)

    def r(self, rel):
        return os.path.join(self.repo, *rel.split("/"))

    def log_obj(self):
        with open(self.log, encoding="utf-8") as f:
            return json.load(f)

    def refused(self, fn, *a):
        with self.assertRaises(SystemExit) as cm:
            quiet(fn, *a)
        return str(cm.exception)

    def q04_uploaded(self):
        e, _ = self.prepare("player", self.player_file)
        self.run_ol("uploaded", "--id", str(e["id"]), "--ok-at", "2026-11-04T16:00", "--uploaded-at",
                    "2026-11-04T16:05")
        return e

    def prep06(self, v, *extra):
        return self.prepare("player", self.f06[v], f"prereg06 {ol.T06_LABEL[v]} (PREREG_06 5.3)", *extra)

    def up(self, eid, at):
        return self.run_ol("uploaded", "--id", str(eid), "--ok-at", at, "--uploaded-at", at)

    def edit_report(self, v, fn):
        p = self.r(f"results/test06/player/{v}/{ol.T06_REPORT}")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def full_flow(self, off_score=0.90, on_score=0.91, q04_score=0.88):
        """Q04 #1, 06-OFF #2, 06-ON #3 uploaded with public scores. -> (q04, off, on) entries."""
        q = self.q04_uploaded()
        self.run_ol("public", "--id", str(q["id"]), "--score", str(q04_score))
        off, _ = self.prep06("off")
        self.up(off["id"], "2026-11-05T10:00")
        self.run_ol("public", "--id", str(off["id"]), "--score", str(off_score))
        on, _ = self.prep06("on")
        self.up(on["id"], "2026-11-05T11:00")
        self.run_ol("public", "--id", str(on["id"]), "--score", str(on_score))
        return q, off, on

    def write_veto(self, off_id, on_id, d, **over):
        v = {"format": ol.T06_VETO_FORMAT, "mode": "real", "entries": {"q04": 1, "off": off_id, "on": on_id},
             "decided": True, "paused": False, "k7_passed": False, "d": d, "vetoed": d is not None and d < 0,
             "final_variant": "off" if d is not None and d < 0 else "on", "reupload_off_needed": d is not None and d < 0}
        v.update(over)
        return write_json(self.r(ol.T06_VETO), v)


class Q04UnchangedWithT06Files(T06Base):
    def test_q04_entry_has_no_06_fields(self):
        e, out = self.prepare("player", self.player_file)
        self.assertNotIn("test06", e)
        self.assertNotIn("prereg06_5_4", e)
        self.assertEqual(e["method"], ol.DEFAULT_METHOD["player"])
        self.assertNotIn("06-OFF", out)
        self.up(1, "2026-11-04T16:05")
        _e, out = self.run_ol("public", "--id", "1", "--score", "0.88")
        self.assertNotIn("PREREG_06", out)
        fc, _ = self.run_ol("final", "--comp", "player", "--id", "1", "--basis", "prereg02 (甲)")
        self.assertNotIn("prereg06_5_4", fc)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertNotIn("prereg06", st["player"])
        self.assertNotIn("K7'", out)


class T06Prepare(T06Base):
    def test_prepare_off_and_on(self):
        q = self.q04_uploaded()
        e, out = self.prep06("off")
        self.assertEqual(e["test06"]["variant"], "off")
        self.assertEqual(e["method"], ol.T06_METHOD["off"])
        self.assertEqual(e["test06"]["q04_first_upload_id"], q["id"])
        self.assertEqual(e["test06"]["run_id"], "run-1")
        self.assertEqual(e["test06"]["k6p"]["ok"], True)
        self.assertEqual(e["t1_verdict"], "PASS")
        for s in ("這一份是 06-OFF", "public 37", "private 41", "K6'"):
            self.assertIn(s, out)
        self.assertNotIn("強制打開的探測", out)
        self.up(e["id"], "2026-11-05T10:00")
        self.run_ol("public", "--id", str(q["id"]), "--score", "0.88")
        self.run_ol("public", "--id", str(e["id"]), "--score", "0.90")
        on, out = self.prep06("on")
        self.assertEqual(on["test06"]["off_entry_id"], e["id"])
        self.assertIn("public 池是強制打開的探測", out)
        self.assertIn("public 17", out)

    def test_order_rules(self):
        msg = self.refused(self.prep06, "off")
        self.assertIn("log Q04's first upload", msg)
        q = self.q04_uploaded()
        msg = self.refused(self.prep06, "on")
        self.assertIn("upload this 06 run's 06-OFF first", msg)
        off, _ = self.prep06("off")
        msg = self.refused(self.prep06, "on")                          # 06-OFF prepared but not uploaded
        self.assertIn("06-OFF first", msg)
        self.up(off["id"], "2026-11-05T10:00")
        msg = self.refused(self.prep06, "on")                          # rev 3: public scores first
        self.assertIn("log the public scores", msg)
        self.run_ol("public", "--id", str(off["id"]), "--score", "0.90")
        self.assertIn("log the public scores", self.refused(self.prep06, "on"))   # Q04's score is missing too
        self.run_ol("public", "--id", str(q["id"]), "--score", "0.88")
        on, _ = self.prep06("on")
        msg = self.refused(self.prep06, "on")                          # the same file prepared twice
        self.assertIn("already prepared as entry", msg)
        self.up(on["id"], "2026-11-05T11:00")
        msg = self.refused(self.prep06, "on")                          # PREREG_06 5.5 E1
        self.assertIn("one 06-ON per season", msg)
        self.prep06("off")                                             # 06-OFF again is allowed (re-upload)

    def test_on_needs_the_same_run(self):
        self.q04_uploaded()
        off, _ = self.prep06("off")
        self.up(off["id"], "2026-11-05T10:00")
        self.edit_report("on", lambda rep: rep.update(run_id="run-2"))
        msg = self.refused(self.prep06, "on")
        self.assertIn("06-OFF first", msg)

    def test_on_after_k7_refused(self):
        self.q04_uploaded()
        off, _ = self.prep06("off")
        self.up(off["id"], "2026-11-05T10:00")
        self.clock = datetime(2026, 11, 23, 0, 0, tzinfo=TW)
        self.assertIn("K7'", self.refused(self.prep06, "on"))

    def test_refusals(self):
        self.q04_uploaded()
        cases = [
            ("mode", lambda: self.edit_report("off", lambda r: r.update(mode="dry-run")), "mode 'dry-run'"),
            ("variant", lambda: self.edit_report("off", lambda r: r.update(variant="on")), "variant 'on'"),
            ("method", lambda: self.edit_report("off", lambda r: r.update(method="x")), "method 'x'"),
            ("check_submission", lambda: self.edit_report("off", lambda r: r.update(
                check_submission={"official": True, "passed": False})), "check_submission"),
            ("output", lambda: self.edit_report("off", lambda r: r.update(output={"sha256": "0" * 64})),
             "records another"),
            ("seam in report", lambda: self.edit_report("off", lambda r: r.update(seam={"passed": False})),
             "06 seam it records"),
            ("h6p", lambda: self.edit_report("off", lambda r: r.update(h6p={"adopted": False})), "H6P"),
            ("seam file", lambda: write_json(self.r(ol.T06_SEAM), {"format": ol.T06_SEAM_FORMAT, "mode": "real",
                                                                   "passed": False}), "did not pass"),
            ("pool report", lambda: os.remove(self.r(ol.T06_POOL_REPORTS[1])), "is missing"),
            ("t1", lambda: self.write_reports(t1=("PASS", "FAIL_OPENING")), "FAIL_OPENING"),
            ("t02 submit", lambda: self.write_reports(player={"mode": "dry-run"}), "06 needs test02"),
            ("input changed", lambda: write_json(self.r("results/prereg06/player/frozen_config.json"), {"x": 1}),
             "bound"),
            ("input dirty", lambda: self.dirty.add("results/test02/player/public/dumps/cand_games.npz"),
             "not committed and clean"),
            ("report dirty", lambda: self.dirty.add(f"results/test06/player/off/{ol.T06_REPORT}"),
             "must be committed"),
            ("k6p", lambda: self.commits.update({f"results/test06/player/off/{ol.T06_REPORT}":
                                                 ("f" * 40, "2026-11-22T00:00:00+08:00")}), "not by K6'"),
        ]
        for name, act, msg in cases:
            with self.subTest(name):
                snap = self.snapshot()
                act()
                self.assertIn(msg, self.refused(self.prep06, "off"))
                self.restore(snap)
        self.prep06("off")                                             # the fixture is usable again

    def test_on_conditions(self):
        self.q04_uploaded()
        off, _ = self.prep06("off")
        self.up(off["id"], "2026-11-05T10:00")
        self.edit_report("on", lambda r: r["on_conditions"]["t2_c_zero_and_abd_not_failed"].update(ok=False))
        self.assertIn("06-ON conditions not all true", self.refused(self.prep06, "on"))

    def test_method(self):
        self.q04_uploaded()
        self.assertIn("does not name 06-OFF", self.refused(self.prep06, "off", "--method", "prereg06 06-ON"))
        e, _ = self.prep06("off", "--method", "prereg06 06-OFF (甲) first 06 upload")
        self.assertEqual(e["method"], "prereg06 06-OFF (甲) first 06 upload")

    def test_a_06_file_is_recognised_by_its_folder_only(self):
        other = ti.write_csv(self.r("results/test06/player/submission_player.csv"),
                             ["question_id"] + [f"top_{i}" for i in range(1, 6)], player_rows(IDS, 5))
        self.assertFalse(ol.is_t06_file("player", other))
        self.assertFalse(ol.is_t06_file("rank", self.f06["off"]))
        self.assertEqual(ol.t06_variant("player", self.f06["on"]), "on")

    def test_not_06_reason_only_for_q04(self):
        self.q04_uploaded()
        self.assertIn("Q04 (test02) file only", self.refused(self.prep06, "off", "--not-06-reason", "PREREG_06 rev 4"))

    def test_q04_while_06_holds_needs_a_revision(self):
        self.q04_uploaded()
        off, _ = self.prep06("off")
        self.up(off["id"], "2026-11-05T10:00")
        self.assertIn("needs --not-06-reason", self.refused(self.prepare, "player", self.player_file, "re-upload"))
        self.p06_head = PREREG06_HEAD + "| rev 4 | `0000004` | bug |\n"
        self.rev_added["| rev 4 |"] = ("4" * 40, "2026-11-07T10:00:00+08:00")
        e, out = self.prepare("player", self.player_file, "re-upload", "--not-06-reason",
                              "PREREG_06 rev 4: t06 submit swapped the pools")
        self.assertEqual(e["prereg06_5_4"]["revision"]["rev"], 4)
        self.assertIn("依 PREREG_06 rev 4", out)

    def snapshot(self):
        snap = {}
        for root, _d, files in os.walk(os.path.join(self.repo, "results")):
            for n in files:
                p = os.path.join(root, n)
                with open(p, "rb") as f:
                    snap[p] = f.read()
        return snap, set(self.dirty), dict(self.commits)

    def restore(self, snap):
        files, dirty, commits = snap
        for p, b in files.items():
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(b)
        self.dirty, self.commits = dirty, commits
        if os.path.exists(self.log):
            log = self.log_obj()
            if any(e.get("test06") for e in log["uploads"]):
                raise AssertionError("a refused prepare wrote an entry")


class T06Public(T06Base):
    def test_d_line_and_no_warning(self):
        q = self.q04_uploaded()
        self.run_ol("public", "--id", str(q["id"]), "--score", "0.88")
        off, _ = self.prep06("off")
        self.up(off["id"], "2026-11-05T10:00")
        _e, out = self.run_ol("public", "--id", str(off["id"]), "--score", "0.90")
        self.assertNotIn("WARNING", out)
        self.assertNotIn("d = ", out)
        on, _ = self.prep06("on")
        self.up(on["id"], "2026-11-05T11:00")
        _e, out = self.run_ol("public", "--id", str(on["id"]), "--score", "0.885")
        self.assertIn("d = public(06-ON #3) − public(06-OFF #2) = -0.015000 → 否決", out)

    def test_same_file_warning(self):
        for pair, scores in ((("06-OFF", "06-ON"), (0.88, 0.90, 0.90)), (("Q04", "06-ON"), (0.90, 0.91, 0.90)),
                             (("Q04", "06-OFF"), (0.90, 0.90, 0.92))):
            with self.subTest(pair):
                if os.path.exists(self.log):
                    os.remove(self.log)
                q = self.q04_uploaded()
                self.run_ol("public", "--id", str(q["id"]), "--score", str(scores[0]))
                off, _ = self.prep06("off")
                self.up(off["id"], "2026-11-05T10:00")
                _e, out_off = self.run_ol("public", "--id", str(off["id"]), "--score", str(scores[1]))
                if pair == ("Q04", "06-OFF"):                          # rev 3: no 06-ON on an unchecked 06-OFF
                    self.assertIn(f"WARNING (PREREG_06 5.3): {pair[0]}", out_off)
                    self.assertIn("same public score", self.refused(self.prep06, "on"))
                    continue
                on, _ = self.prep06("on")
                self.up(on["id"], "2026-11-05T11:00")
                _e, out_on = self.run_ol("public", "--id", str(on["id"]), "--score", str(scores[2]))
                text = out_off + out_on
                self.assertIn(f"WARNING (PREREG_06 5.3): {pair[0]}", text)
                self.assertIn("可能傳錯檔", text)
                self.assertNotIn("d = public", out_on)

    def test_equal_scores_with_identical_top1_no_warning(self):
        self.edit_report("on", lambda r: r.update(top1_differences_off_vs_on={"public": 0, "private": 3}))
        q, off, on = self.full_flow(off_score=0.90, on_score=0.90)
        log = self.log_obj()
        self.assertEqual([e["public_score"] for e in log["uploads"]], [0.88, 0.90, 0.90])
        lines = ol.t06_public_lines(log, ol.entry(log, on["id"]))
        self.assertEqual(len(lines), 1)
        self.assertIn("d = public(06-ON #3) − public(06-OFF #2) = +0.000000 → 不否決", lines[0])


class T06Final(T06Base):
    def test_d_negative_vetoes(self):
        q, off, on = self.full_flow(off_score=0.90, on_score=0.89)
        self.write_veto(off["id"], on["id"], 0.89 - 0.90)
        msg = self.refused(self.run_ol, "final", "--comp", "player", "--id", str(on["id"]), "--basis", "x")
        self.assertIn("must be 06-OFF", msg)
        msg = self.refused(self.run_ol, "final", "--comp", "player", "--id", str(q["id"]), "--basis", "x")
        self.assertIn("must be 06-OFF", msg)
        fc, out = self.run_ol("final", "--comp", "player", "--id", str(off["id"]), "--basis", "PREREG_06 5.3 veto")
        self.assertEqual(fc["prereg06_5_4"]["designated"], "off")
        self.assertIn("規則指定 off", out)
        st, out = self.run_ol("status", "--now", "2026-11-08T00:00")
        self.assertTrue(st["player"]["prereg06"]["reupload_off_needed"])
        self.assertIn("再上傳一次 06-OFF", out)
        self.assertFalse(st["player"]["prereg06"]["violation"])
        again, _ = self.prep06("off")                                  # the re-upload of 06-OFF
        self.up(again["id"], "2026-11-08T10:00")
        st, out = self.run_ol("status", "--now", "2026-11-08T12:00")
        self.assertFalse(st["player"]["prereg06"]["reupload_off_needed"])
        self.assertTrue(st["player"]["last_is_final"])

    def test_d_zero_and_positive_keep_on(self):
        for on_score in (0.90, 0.92):
            with self.subTest(on_score=on_score):
                if os.path.exists(self.log):
                    os.remove(self.log)
                q, off, on = self.full_flow(off_score=0.90, on_score=on_score)
                self.write_veto(off["id"], on["id"], on_score - 0.90)
                msg = self.refused(self.run_ol, "final", "--comp", "player", "--id", str(off["id"]), "--basis", "x")
                self.assertIn("must be 06-ON", msg)
                fc, _ = self.run_ol("final", "--comp", "player", "--id", str(on["id"]), "--basis", "PREREG_06 5.3")
                self.assertEqual(fc["prereg06_5_4"]["chosen"], "on")
                st, out = self.run_ol("status", "--now", "2026-11-08T00:00")
                self.assertFalse(st["player"]["prereg06"]["reupload_off_needed"])
                self.assertNotIn("再上傳一次 06-OFF", out)

    def test_veto_must_match_the_log(self):
        q, off, on = self.full_flow(off_score=0.90, on_score=0.89)
        self.write_veto(off["id"], on["id"], 0.01, vetoed=False)        # d and vetoed differ from the log
        msg = self.refused(self.run_ol, "final", "--comp", "player", "--id", str(on["id"]), "--basis", "x")
        self.assertIn("veto is not decided", msg)
        self.write_veto(off["id"], on["id"], -0.01)
        self.dirty.add(ol.T06_VETO)
        self.assertIn("veto is not decided", self.refused(self.run_ol, "final", "--comp", "player", "--id",
                                                          str(off["id"]), "--basis", "x"))

    def test_undecided_before_k7_and_off_after(self):
        q, off, on = self.full_flow()
        for eid in (off["id"], on["id"], q["id"]):
            self.assertIn("veto is not decided", self.refused(self.run_ol, "final", "--comp", "player", "--id",
                                                              str(eid), "--basis", "x"))
        st, out = self.run_ol("status", "--now", "2026-11-08T00:00")
        self.assertEqual(st["player"]["prereg06"]["designated"], "undecided")
        self.clock = datetime(2026, 11, 23, 0, 1, tzinfo=TW)            # K7' passed, no veto
        self.assertIn("must be 06-OFF", self.refused(self.run_ol, "final", "--comp", "player", "--id", str(on["id"]),
                                                     "--basis", "x"))
        self.run_ol("final", "--comp", "player", "--id", str(off["id"]), "--basis", "PREREG_06 4: K7'")
        st, out = self.run_ol("status", "--now", "2026-11-23T00:01")
        self.assertTrue(st["player"]["prereg06"]["k7_reupload_off"])
        self.assertIn("再上傳一次 06-OFF", out)
        self.assertIn("距 K7' 11/22 23:59（06-ON 上傳、兩份分數、否決都要在這之前）：已過", out)

    def test_veto_committed_after_k7_is_too_late(self):
        q, off, on = self.full_flow(off_score=0.90, on_score=0.91)
        self.write_veto(off["id"], on["id"], 0.91 - 0.90)
        self.commits[ol.T06_VETO] = ("e" * 40, "2026-11-23T00:30:00+08:00")
        self.clock = datetime(2026, 11, 23, 1, 0, tzinfo=TW)
        self.assertIn("must be 06-OFF", self.refused(self.run_ol, "final", "--comp", "player", "--id", str(on["id"]),
                                                     "--basis", "x"))

    def test_paused_veto_is_undecided_before_k7(self):
        q, off, on = self.full_flow(off_score=0.90, on_score=0.90)
        self.write_veto(off["id"], on["id"], None, decided=False, paused=True, vetoed=None, final_variant=None,
                        outcome="paused", reupload_off_needed=None)
        for eid in (on["id"], off["id"], q["id"]):                     # rev 3: nothing designated before the check
            msg = self.refused(self.run_ol, "final", "--comp", "player", "--id", str(eid), "--basis", "x")
            self.assertIn("veto is not decided", msg)
            self.assertIn("veto paused", msg)
        st, out = self.run_ol("status", "--now", "2026-11-08T00:00")
        p6 = st["player"]["prereg06"]
        self.assertEqual(p6["designated"], "undecided")
        self.assertFalse(p6["reupload_off_needed"])
        self.assertIn("暫停（同名檔警告）", out)
        self.assertIn("核對前不重傳", out)
        self.assertNotIn("再上傳一次 06-OFF", out)
        self.assertNotIn("最終選擇要設成", out)
        self.clock = datetime(2026, 11, 23, 0, 1, tzinfo=TW)            # K7' passed with the veto still paused
        self.assertIn("must be 06-OFF", self.refused(self.run_ol, "final", "--comp", "player", "--id", str(on["id"]),
                                                     "--basis", "x"))
        self.run_ol("final", "--comp", "player", "--id", str(off["id"]), "--basis", "PREREG_06 4: K7'")
        st, out = self.run_ol("status", "--now", "2026-11-23T00:01")
        self.assertEqual(st["player"]["prereg06"]["designated"], "off")
        self.assertIn("再上傳一次 06-OFF", out)

    def test_revision_override(self):
        q, off, on = self.full_flow(off_score=0.90, on_score=0.89)
        self.write_veto(off["id"], on["id"], 0.89 - 0.90)
        for reason, rows, added, msg in (
                ("the veto looked wrong", (), None, "without a revision"),
                ("PREREG_06 rev 3: x", (3,), "2026-11-09T10:00:00+08:00", "rev 4 or later"),
                ("PREREG_06 rev 4: x", (), None, "has no row"),
                ("PREREG_06 rev 4: x", (4,), "2026-11-04T14:00:00+08:00", "not after the records")):
            with self.subTest(reason=reason, rows=rows):
                self.p06_head = PREREG06_HEAD + "".join(f"| rev {n} | `{n:07d}` | r |\n" for n in rows)
                self.rev_added = {f"| rev {n} |": ("9" * 40, added) for n in rows} if added else {}
                self.assertIn(msg, self.refused(self.run_ol, "final", "--comp", "player", "--id", str(on["id"]),
                                                "--basis", "x", "--not-06-reason", reason))
        self.p06_head = PREREG06_HEAD + "| rev 4 | `0000004` | bug |\n"
        self.rev_added = {"| rev 4 |": ("4" * 40, "2026-11-09T10:00:00+08:00")}
        fc, out = self.run_ol("final", "--comp", "player", "--id", str(on["id"]), "--basis", "PREREG_06 rev 4",
                              "--not-06-reason", "PREREG_06 rev 4: veto read the wrong log entry (5.3 step 4)")
        self.assertEqual(fc["prereg06_5_4"]["revision"]["rev"], 4)
        self.assertIn("依 PREREG_06 rev 4", out)
        st, _ = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertFalse(st["player"]["prereg06"]["violation"])

    def test_no_eligible_06_gives_q04(self):
        q, off, on = self.full_flow()
        write_json(self.r("results/test02/player/seam_val/seam_local.json"), {"passed": True, "mode": "real",
                                                                                "rerun": True})
        st, out = self.run_ol("status", "--now", "2026-11-08T00:00")
        self.assertEqual(st["player"]["prereg06"]["designated"], "q04")
        self.assertIn("不符合", out)
        fc, _ = self.run_ol("final", "--comp", "player", "--id", str(q["id"]), "--basis", "U5")
        self.assertEqual(fc["prereg06_5_4"]["chosen"], "q04")

    def test_k6p_rechecked(self):
        q, off, on = self.full_flow()
        self.commits[f"results/test06/player/off/{ol.T06_REPORT}"] = ("f" * 40, "2026-11-22T00:00:00+08:00")
        st, _ = self.run_ol("status", "--now", "2026-11-08T00:00")
        self.assertNotIn(off["id"], st["player"]["prereg06"]["eligible"])
        self.assertEqual(st["player"]["prereg06"]["designated"], "q04")

    def test_final_on_rank_with_06_reason_refused(self):
        self.assertIn("competition-1 Q04 (test02) file only", self.refused(
            self.run_ol, "prepare", "--comp", "rank", "--file", self.rank_file, "--reason", "x", "--tests-dir",
            self.tests, "--not-06-reason", "PREREG_06 rev 4: x"))


class T06Rev3(T06Base):
    """Review 10/3 (correctness and procedure reviews, scenarios S1-S5 / veto_drill2): the rev 3 rules."""

    def final_ok(self, eid):
        fc, _ = self.run_ol("final", "--comp", "player", "--id", str(eid), "--basis", "PREREG_06 5.4")
        return fc

    def test_on_kept_but_not_last_gives_off(self):
        q, off, on = self.full_flow(off_score=0.90, on_score=0.92)
        again, _ = self.prep06("off")                                  # someone uploaded 06-OFF once more anyway
        self.up(again["id"], "2026-11-08T10:00")
        self.write_veto(off["id"], on["id"], 0.92 - 0.90)
        st, _ = self.run_ol("status", "--now", "2026-11-09T00:00")
        p6 = st["player"]["prereg06"]
        self.assertEqual(p6["designated"], "off")
        self.assertIn("no longer the last upload", p6["rule"])
        self.assertIn("must be 06-OFF", self.refused(self.run_ol, "final", "--comp", "player", "--id", str(on["id"]),
                                                     "--basis", "x"))
        self.final_ok(again["id"])
        st, _ = self.run_ol("status", "--now", "2026-11-09T00:00")
        self.assertTrue(st["player"]["last_is_final"])
        self.assertFalse(st["player"]["prereg06"]["violation"])
        self.assertFalse(st["player"]["prereg06"]["reupload_off_needed"])

    def test_wrong_upload_or_inconclusive_gives_off(self):
        for outcome in ol.T06_UPLOAD_CHECK_OFF:
            with self.subTest(outcome):
                if os.path.exists(self.log):
                    os.remove(self.log)
                q, off, on = self.full_flow(off_score=0.90, on_score=0.90)
                self.write_veto(off["id"], on["id"], None, outcome=outcome, final_variant="off", vetoed=None,
                                reupload_off_needed=True, upload_check={"kind": outcome, "text": "AIdea record"})
                st, out = self.run_ol("status", "--now", "2026-11-08T00:00")
                p6 = st["player"]["prereg06"]
                self.assertEqual(p6["designated"], "off")
                self.assertTrue(p6["reupload_off_needed"])
                self.assertIn(f"上傳紀錄核對（{outcome}）", out)
                self.assertIn("再上傳一次 06-OFF", out)
                self.assertIn("must be 06-OFF", self.refused(self.run_ol, "final", "--comp", "player", "--id",
                                                             str(on["id"]), "--basis", "x"))
                again, _ = self.prep06("off")
                self.up(again["id"], "2026-11-08T10:00")
                self.final_ok(again["id"])
                st, _ = self.run_ol("status", "--now", "2026-11-08T12:00")
                self.assertTrue(st["player"]["last_is_final"])
                self.assertFalse(st["player"]["prereg06"]["reupload_off_needed"])

    def test_upload_check_veto_cannot_say_on(self):
        q, off, on = self.full_flow(off_score=0.90, on_score=0.90)
        self.write_veto(off["id"], on["id"], None, outcome="wrong_upload", final_variant="on", vetoed=None)
        st, _ = self.run_ol("status", "--now", "2026-11-08T00:00")
        self.assertEqual(st["player"]["prereg06"]["designated"], "undecided")
        self.assertIn("must give final 06-OFF", st["player"]["prereg06"]["veto_problem"])

    def test_off_uploaded_again_before_on_is_paired(self):
        q = self.q04_uploaded()
        self.run_ol("public", "--id", str(q["id"]), "--score", "0.88")
        off1, _ = self.prep06("off")
        self.up(off1["id"], "2026-11-05T10:00")
        _e, out = self.run_ol("public", "--id", str(off1["id"]), "--score", "0.88")   # it was Q04's file
        self.assertIn("WARNING (PREREG_06 5.3): Q04", out)
        self.assertIn("same public score", self.refused(self.prep06, "on"))
        off2, _ = self.prep06("off")                                   # the right file, once more
        self.up(off2["id"], "2026-11-05T12:00")
        _e, out = self.run_ol("public", "--id", str(off2["id"]), "--score", "0.90")
        self.assertIn(f"WARNING (PREREG_06 rev 3): #{off2['id']} 和 #{off1['id']} 是同一份檔", out)
        on, _ = self.prep06("on")
        self.assertEqual(on["test06"]["off_entry_id"], off2["id"])
        self.up(on["id"], "2026-11-05T13:00")
        _e, out = self.run_ol("public", "--id", str(on["id"]), "--score", "0.91")
        self.assertIn(f"d = public(06-ON #{on['id']}) − public(06-OFF #{off2['id']}) = +0.010000", out)
        self.write_veto(off1["id"], on["id"], 0.91 - 0.88)              # paired with the wrong 06-OFF: not this veto
        st, _ = self.run_ol("status", "--now", "2026-11-06T00:00")
        self.assertEqual(st["player"]["prereg06"]["designated"], "undecided")
        self.write_veto(off2["id"], on["id"], 0.91 - 0.90)
        st, _ = self.run_ol("status", "--now", "2026-11-06T00:00")
        self.assertEqual(st["player"]["prereg06"]["designated"], "on")
        self.final_ok(on["id"])

    def test_reupload_scored_like_another_file_is_warned(self):
        q, off, on = self.full_flow(off_score=0.90, on_score=0.89)
        self.write_veto(off["id"], on["id"], 0.89 - 0.90)
        again, _ = self.prep06("off")
        self.up(again["id"], "2026-11-08T10:00")
        _e, out = self.run_ol("public", "--id", str(again["id"]), "--score", "0.89")   # = 06-ON: dragged 06-ON's file
        self.assertIn(f"WARNING (PREREG_06 rev 3): #{again['id']} 和 #{off['id']} 是同一份檔", out)
        self.assertNotIn("d = public", out)
        log = self.log_obj()
        log["uploads"][-1]["public_score"] = 0.90                       # the right file scores as the first 06-OFF
        lines = ol.t06_public_lines(log, ol.entry(log, again["id"]))
        self.assertEqual(lines, [])                                    # no warning, and no d line for a re-upload


class T06Status(T06Base):
    def test_status_lines(self):
        q, off, on = self.full_flow(off_score=0.90, on_score=0.91)
        st, out = self.run_ol("status", "--now", "2026-11-08T00:00")
        p6 = st["player"]["prereg06"]
        self.assertEqual(p6["entries"], [off["id"], on["id"]])
        self.assertEqual(p6["designated"], "undecided")
        self.assertIn("06（PREREG_06 5.4）", out)
        self.assertIn("距 K6' 11/21 23:59", out)
        self.write_veto(off["id"], on["id"], 0.91 - 0.90)
        st, out = self.run_ol("status", "--now", "2026-11-08T00:00")
        self.assertEqual(st["player"]["prereg06"]["designated"], "on")
        self.assertIn("最終選擇要設成 06-ON", out)
        log = self.log_obj()                                           # a Q04 final set by hand (no revision)
        log["final_choice"]["player"] = {"id": q["id"], "sha256": q["sha256"], "file": q["file"], "method": "x",
                                         "basis": "x", "set_at": NOW}
        with open(self.log, "w", encoding="utf-8") as f:
            json.dump(log, f)
        st, out = self.run_ol("status", "--now", "2026-11-08T00:00")
        self.assertTrue(st["player"]["prereg06"]["violation"])
        self.assertIn("違反 PREREG_06 5.4", out)


if __name__ == "__main__":
    unittest.main()
