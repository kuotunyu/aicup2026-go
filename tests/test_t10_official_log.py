"""Tests for the PREREG_10 rev 3 additions to tools/official_log.py (PREREG_10 4.2, 4.4; competition 1, part A).

* TagVersusNew: the version of tag p09-testday-pre10 (R20, 04, 06, 07, 08 and 09 additions, no 10) and the
  working-tree version run the same flows with the same mocks (test_t09_official_log's R20, R10, 04 and 06 flows,
  the 07 flow and the 08 flow), now with a results/test10 tree on disk: log bytes, returned records, stdout and
  refusals are identical (a log without 10 entries).
* Q04And06UnchangedWithT10Files: Q04 and 06 flows with a complete results/test10 tree next to them and no 10 entry:
  no 10 field appears anywhere.
* T10*: the 10 path: prepare (the file recognised by its folder, the report's own conditions, the bound inputs, K6-10,
  K5-10, --method), the order (Q04 first; this season's 06-OFF uploaded first when there is one; 06-ON kept -> no 10,
  vetoed / K7' passed -> 10), final under PREREG_10 4.4 (10 designated; 06-OFF / Q04 only with --not-10-reason
  citing a later revision; 06-ON kept stays 06-ON; --not-06-reason refused while 10 entries exist), status (the 06
  block defers to the 10 block, no false violation), public (the same-file warning, the 0.05 reminder).
Synthetic files in a temporary directory only; git and the clock are mocked (ol.git_file_state, ol.git_commit_time,
ol.git_head_text, ol.git_row_added, ol.now_iso, ol.t06_now). Nothing here opens a label file or runs tools/score.py.
Run: python -m pytest tests/test_t10_official_log.py
"""

import copy
import json
import os
import sys
import tempfile
import subprocess
import unittest
from datetime import datetime
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import official_log as ol  # noqa: E402
import test_t02_infra as ti  # noqa: E402
import test_t04_official_log as t4o  # noqa: E402
import test_t06_official_log as t6o  # noqa: E402
import test_t07_official_log as t7o  # noqa: E402
import test_t09_official_log as t9o  # noqa: E402
import t10_player as t10p  # noqa: E402

TAG = "p09-testday-pre10"                          # the 09 test-day path before the 10 kit
TW = t6o.TW
sha, write_json, quiet = t4o.sha, t4o.write_json, t4o.quiet
PREREG10_HEAD = "# PREREG_10\n\n| rev 1 | `b9b954d` | first |\n| rev 2 | `4c2c15a` | B |\n| rev 3 | (commit) | test day |\n"
REV2_ADDED = ("a" * 40, "2026-10-20T10:00:00+08:00")       # before K5-10 (10/31 23:59:59)
REPORT10_COMMIT = ("1" * 40, "2026-11-06T18:00:00+08:00")
DIFF_OFF = {"public": 0, "private": 61}
DIFF_Q04 = {"public": 37, "private": 70}


# ---------------------------------------------------------------- the results/test10 tree of a usable 10 run
def write_t10_tree(repo, off_file=None, run_id_06="run-1", on_produced=False, report_over=None):
    """results/test10/player as tools/t10_player.py submit leaves it (real mode; public from 06-OFF, private lifted by
    A), plus the committed inputs it binds. off_file: this season's 06-OFF submission (None: no 06-OFF). -> 10 file."""
    r = lambda rel: os.path.join(repo, *rel.split("/"))   # noqa: E731
    write_json(r(ol.T10_SEAM), {"format": ol.T10_SEAM_FORMAT, "mode": "real", "passed": True, "verdict": "passed"})
    for p in ol.T10_POOL_REPORTS:
        write_json(r(p), {"format": ol.T10_POOL_FORMAT, "mode": "real"})
    for rel in ol.T10_INPUTS_REQUIRED + ("results/test02/checks/player/t4.json",):
        if not os.path.exists(r(rel)):
            if rel.endswith(".csv"):
                ti.write_csv(r(rel), ["question_id"] + [f"top_{i}" for i in range(1, 6)], t6o.player_rows(t6o.IDS, 0))
            else:
                write_json(r(rel), {"fake_input": rel, "passed": True, "format": ol.T06_SEAM_FORMAT, "mode": "real"})
    f10 = ti.write_csv(r("results/test10/player/submission_player.csv"),
                       ["question_id"] + [f"top_{i}" for i in range(1, 6)], t6o.player_rows(t6o.IDS, 7))
    rep = report10(repo, f10, off_file, run_id_06, on_produced)
    rep.update(report_over or {})
    write_json(r(f"results/test10/player/{ol.T10_REPORT}"), rep)
    return f10


def report10(repo, f10, off_file, run_id_06, on_produced):
    bound = {rel: ol.sha256_bound(os.path.join(repo, *rel.split("/")))
             for rel in ol.T10_INPUTS_REQUIRED + ("results/test02/checks/player/t4.json",)}
    return {"format": ol.T10_FORMAT, "mode": "real", "method": ol.T10_METHOD, "run_id": "run10-1",
            "check_submission": {"official": True, "passed": True, "errors": []},
            "output": {"path": "results/test10/player/submission_player.csv", "sha256": sha(f10)},
            "sources": {"public": "06-OFF" if off_file else "test02", "private": "10A"},
            "a_holds": {"public": False, "private": True}, "reference": "06-OFF" if off_file else "Q04",
            "top1_differences_vs_06_off": copy.deepcopy(DIFF_OFF) if off_file else None,
            "top1_differences_vs_q04": copy.deepcopy(DIFF_Q04),
            "t06_off": ({"exists": True, "run_id": run_id_06, "sha256": sha(off_file), "on_produced": on_produced}
                        if off_file else {"exists": False}),
            "seam": {"passed": True}, "t06_seam": {"passed": True}, "h6p": {"adopted": True},
            "t1": {"verdict": "PASS"}, "b": {"adopted": False}, "inputs_bound": bound}


# ---------------------------------------------------------------- the tag version against the new one
class TagVersusNew(t9o.TagVersusNew):
    """A log without 10 entries gives the tag p09-testday-pre10 version's log bytes, records, stdout and refusals,
    with results/test09 and results/test10 trees on disk (the R20 / R10 / 04 flows with and without a 06 tree, the
    "with a 07 tree" flows and the 06 flow of test_t07_official_log)."""

    @classmethod
    def setUpClass(cls):
        cls.tag = t7o.load_version(TAG)
        assert hasattr(cls.tag, "check_reports_t09") and not hasattr(cls.tag, "check_reports_t10"), "not the tag"

    def build(self, kind, with_t06):
        out = super().build(kind, with_t06)
        off = os.path.join(out[0], "results", "test06", "player", "off", "submission_player.csv")
        write_t10_tree(out[0], off if os.path.exists(off) else None)
        return out


class _Flow10:
    """test_t09_official_log's FlowBase with tag p09-testday-pre10 and a results/test10 tree on disk."""

    def setUp(self):
        self.tag = t7o.load_version(TAG)

    def run_with(self, mod):
        real = t9o.write_t09_tree

        def both(repo, rows, *a, **k):
            out = real(repo, rows, *a, **k)
            write_t10_tree(repo)
            return out
        with mock.patch.object(t9o, "write_t09_tree", side_effect=both):
            return super().run_with(mod)

    def check_identical(self):
        old = super().check_identical()
        log = json.loads(old[-1][3])
        self.assertFalse(any("test10" in e or "prereg10_4_4" in e for e in log["uploads"]))
        self.assertNotIn("prereg10_4_4", json.dumps(log["final_choice"]))
        return old


class TagVersusNew07Flow(_Flow10, t9o.TagVersusNew07Flow):
    pass


class TagVersusNew08Flow(_Flow10, t9o.TagVersusNew08Flow):
    pass


# ---------------------------------------------------------------- the 10 path
class T10Base(t6o.T06Base):
    snapshot = t6o.T06Prepare.snapshot

    def restore(self, snap):
        files, dirty, commits = snap
        for p, b in files.items():
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(b)
        self.dirty, self.commits = dirty, commits
        if any(e.get("test10") for e in self.log_obj()["uploads"]):
            raise AssertionError("a refused prepare wrote a 10 entry")

    """test_t06_official_log.T06Base (data/test, test02's reports, Q04's file, a usable 06 run with 06-OFF and 06-ON)
    plus a usable 10 run; PREREG_10 at HEAD has the test-day row rev 3 (added 10/20, before K5-10)."""
    off_in_season = True
    on_produced = False

    def setUp(self):
        super().setUp()
        self.p10_head = PREREG10_HEAD
        self.rev_added["| rev 3 |"] = REV2_ADDED
        self.commits[f"results/test10/player/{ol.T10_REPORT}"] = REPORT10_COMMIT
        self.patches10 = [mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p10_head
                                            if rel == ol.P10_DOC else self.p06_head if rel == ol.P06_DOC
                                            else t4o.PREREG04_HEAD),
                          mock.patch.object(ol, "git_row_added_any",
                                            side_effect=lambda rel, row: self.rev_added.get(row, (None, None)))]
        for pt in self.patches10:
            pt.start()
        self.f10 = write_t10_tree(self.repo, self.f06["off"] if self.off_in_season else None,
                                  on_produced=self.on_produced)

    def tearDown(self):
        for pt in reversed(self.patches10):
            pt.stop()
        super().tearDown()

    def edit10(self, fn):
        p = self.r(f"results/test10/player/{ol.T10_REPORT}")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def prep10(self, *extra):
        return self.prepare("player", self.f10, "prereg10 10 (PREREG_10 4.4)", *extra)

    def off_uploaded(self, at="2026-11-05T10:00", score=0.90):
        q = self.q04_uploaded()
        self.run_ol("public", "--id", str(q["id"]), "--score", "0.88")
        off, _ = self.prep06("off")
        self.up(off["id"], at)
        self.run_ol("public", "--id", str(off["id"]), "--score", str(score))
        return q, off

    def up10(self, at="2026-11-07T10:00"):
        e, out = self.prep10()
        self.up(e["id"], at)
        return e, out


class Q04And06UnchangedWithT10Files(T10Base):
    def test_no_10_fields(self):
        q, off = self.off_uploaded()
        fc, out = self.run_ol("final", "--comp", "player", "--id", str(off["id"]), "--basis", "PREREG_06 5.4")
        self.assertNotIn("prereg10_4_4", fc)
        self.assertIn("prereg06_5_4", fc)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertNotIn("prereg10", st["player"])
        self.assertNotIn("PREREG_10", out)
        self.assertNotIn("K6-10", out)
        self.assertFalse(st["player"]["prereg06"]["violation"])
        log = self.log_obj()
        self.assertFalse(any("test10" in e or "prereg10_4_4" in e for e in log["uploads"]))

    def test_recognised_by_folder_only(self):
        self.assertTrue(ol.is_t10_file("player", self.f10))
        self.assertFalse(ol.is_t10_file("rank", self.f10))
        self.assertFalse(ol.is_t10_file("player", self.f06["off"]))
        self.assertFalse(ol.is_t06_file("player", self.f10))
        other = ti.write_csv(self.r("results/test10/submission_player.csv"),
                             ["question_id"] + [f"top_{i}" for i in range(1, 6)], t6o.player_rows(t6o.IDS, 7))
        self.assertFalse(ol.is_t10_file("player", other))


class T10Prepare(T10Base):
    def test_prepare_after_06_off(self):
        self.assertIn("log Q04's first upload", self.refused(self.prep10))
        q = self.q04_uploaded()
        msg = self.refused(self.prep10)
        self.assertIn("upload this season's 06-OFF", msg)
        self.run_ol("public", "--id", str(q["id"]), "--score", "0.88")
        off, _ = self.prep06("off")
        self.assertIn("upload this season's 06-OFF", self.refused(self.prep10))     # prepared, not uploaded
        self.up(off["id"], "2026-11-05T10:00")
        e, out = self.prep10()
        self.assertEqual(e["method"], ol.T10_METHOD)
        t = e["test10"]
        self.assertEqual((t["q04_first_upload_id"], t["off_entry_id"]), (q["id"], off["id"]))
        self.assertEqual(t["k6_10"]["ok"], True)
        self.assertTrue(t["k5_10"]["ok"])
        self.assertEqual(t["sources"], {"public": "06-OFF", "private": "10A"})
        self.assertNotIn("test06", e)
        self.assertNotIn("prereg06_5_4", e)
        for s in ("這一份是 10", "private 70", "和 06-OFF", "private 61", "K5-10 成立"):
            self.assertIn(s, out)

    def test_06_off_of_another_run_does_not_count(self):
        self.q04_uploaded()
        self.edit_report("off", lambda rep: rep.update(run_id="run-0"))
        off, _ = self.prep06("off")
        self.up(off["id"], "2026-11-05T10:00")
        self.assertIn("upload this season's 06-OFF", self.refused(self.prep10))

    def test_refusals(self):
        self.off_uploaded()
        cases = [
            ("mode", lambda: self.edit10(lambda r: r.update(mode="dry-run")), "mode 'dry-run'"),
            ("method", lambda: self.edit10(lambda r: r.update(method="prereg06 06-OFF")), "method"),
            ("check_submission", lambda: self.edit10(lambda r: r.update(
                check_submission={"official": True, "passed": False})), "check_submission"),
            ("output", lambda: self.edit10(lambda r: r.update(output={"sha256": "0" * 64})), "records another"),
            ("seam in report", lambda: self.edit10(lambda r: r.update(seam={"passed": False})), "10 seam it records"),
            ("06 seam in report", lambda: self.edit10(lambda r: r.update(t06_seam={"passed": False})),
             "06 seam it records"),
            ("h6p", lambda: self.edit10(lambda r: r.update(h6p={"adopted": False})), "H6P"),
            ("t1", lambda: self.edit10(lambda r: r.update(t1={"verdict": "FAIL_OPENING"})), "T1 'FAIL_OPENING'"),
            ("no 10A", lambda: self.edit10(lambda r: r.update(sources={"public": "06-OFF", "private": "06-OFF"})),
             "no pool was lifted"),
            ("10B", lambda: self.edit10(lambda r: r.update(sources={"public": "06-OFF", "private": "10B"})),
             "B adopted"),
            # PREREG_10 rev 4: a 10B pool without the B evidence is refused by the B rules (B adopted without a 10B pool
            # is an A report: tests/test_t10_b.py OfficialLogB)
            ("b adopted", lambda: self.edit10(lambda r: r.update(b={"adopted": True}, sources={
                "public": "06-OFF", "private": "10B"})), "B GPU seam it records did not"),
            ("seam file", lambda: write_json(self.r(ol.T10_SEAM), {"format": ol.T10_SEAM_FORMAT, "mode": "real",
                                                                   "passed": False}), "did not pass"),
            ("pool report", lambda: os.remove(self.r(ol.T10_POOL_REPORTS[1])), "is missing"),
            ("t02 t1", lambda: self.write_reports(t1=("PASS", "FAIL_OPENING")), "FAIL_OPENING"),
            ("input changed", lambda: write_json(self.r("results/prereg02/player/frozen_config.json"), {"x": 1}),
             "bound"),
            ("t4 not bound", lambda: self.edit10(lambda r: r["inputs_bound"].pop(
                "results/test02/checks/player/t4.json")), "lacks"),
            ("input dirty", lambda: self.dirty.add(ol.T10_SEAM), "must be committed"),
            ("k6", lambda: self.commits.update({f"results/test10/player/{ol.T10_REPORT}":
                                                ("f" * 40, "2026-11-22T00:00:00+08:00")}), "not by K6-10"),
            ("k5 late", lambda: self.rev_added.update({"| rev 3 |": ("a" * 40, "2026-11-01T00:00:00+08:00")}),
             "K5-10"),
            ("k5 missing", lambda: setattr(self, "p10_head", "# PREREG_10\n\n| rev 1 | x | first |\n"), "K5-10"),
        ]
        for name, act, msg in cases:
            with self.subTest(name):
                snap = self.snapshot()
                head, commits, added, dirty = self.p10_head, dict(self.commits), dict(self.rev_added), set(self.dirty)
                act()
                self.assertIn(msg, self.refused(self.prep10))
                self.restore(snap)
                self.p10_head, self.commits, self.rev_added, self.dirty = head, commits, added, dirty
        self.prep10()                                                  # the fixture is usable again

    def test_method(self):
        self.off_uploaded()
        self.assertIn("does not name prereg10", self.refused(self.prep10, "--method", "prereg06 06-OFF"))
        e, _ = self.prep10("--method", "prereg10 10 (甲) first 10 upload")
        self.assertEqual(e["method"], "prereg10 10 (甲) first 10 upload")

    def test_other_reasons_on_a_10_file(self):
        self.off_uploaded()
        self.assertIn("Q04 (test02) file only", self.refused(self.prep10, "--not-06-reason", "PREREG_06 rev 4: x"))
        self.assertIn("not 10", self.refused(self.prep10, "--not-10-reason", "PREREG_10 rev 4: x"))

    def test_06_off_and_q04_after_10(self):
        _q, off = self.off_uploaded()
        e10, _ = self.up10()
        self.assertIn("needs --not-10-reason", self.refused(self.prep06, "off"))
        msg = self.refused(self.prepare, "player", self.player_file, "re-upload")
        self.assertIn("needs --not-", msg)
        self.p10_head = PREREG10_HEAD + "| rev 4 | `0000004` | bug |\n"
        self.rev_added["| rev 4 |"] = ("4" * 40, "2026-11-08T10:00:00+08:00")
        e, out = self.prep06("off", "--not-10-reason", "PREREG_10 rev 4: t10 swapped the pools")
        self.assertEqual(e["prereg10_4_4"]["revision"]["rev"], 4)
        self.assertIn("依 PREREG_10 rev 4", out)
        self.assertEqual(e["prereg10_4_4"]["designated"], "10")


class T10NoOffSeason(T10Base):
    """Both pools fell back to test02 in 06 (no 06-OFF this season): 10 is compared with Q04 and follows Q04."""
    off_in_season = False

    def test_prepare_after_q04_only(self):
        self.assertIn("log Q04's first upload", self.refused(self.prep10))
        q = self.q04_uploaded()
        e, out = self.prep10()
        self.assertEqual(e["test10"]["q04_first_upload_id"], q["id"])
        self.assertNotIn("off_entry_id", e["test10"])
        self.assertNotIn("和 06-OFF", out)
        self.up(e["id"], "2026-11-06T10:00")
        fc, out = self.run_ol("final", "--comp", "player", "--id", str(e["id"]), "--basis", "PREREG_10 4.4")
        self.assertEqual(fc["prereg10_4_4"]["chosen"], "10")
        self.assertNotIn("prereg06_5_4", fc)
        msg = self.refused(self.run_ol, "final", "--comp", "player", "--id", str(q["id"]), "--basis", "x")
        self.assertIn("must be 10", msg)


class T10With06On(T10Base):
    on_produced = True

    def test_on_kept_means_no_10(self):
        _q, off, on = self.full_flow(off_score=0.90, on_score=0.91)
        self.assertIn("veto", self.refused(self.prep10))                          # veto not decided
        self.write_veto(off["id"], on["id"], 0.01)
        self.assertIn("06-ON is kept", self.refused(self.prep10))

    def test_on_vetoed_then_10(self):
        _q, off, on = self.full_flow(off_score=0.90, on_score=0.89)
        self.write_veto(off["id"], on["id"], -0.01)
        e, _ = self.prep10()
        self.assertEqual(e["test10"]["off_entry_id"], off["id"])
        self.up(e["id"], "2026-11-07T10:00")
        fc, _ = self.run_ol("final", "--comp", "player", "--id", str(e["id"]), "--basis", "PREREG_10 4.4")
        self.assertEqual((fc["prereg10_4_4"]["designated"], fc["prereg10_4_4"]["chosen"]), ("10", "10"))
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertFalse(st["player"]["prereg06"]["reupload_off_needed"])
        self.assertNotIn("再上傳一次 06-OFF", out)

    def test_on_not_uploaded_waits_for_k7(self):
        self.off_uploaded()
        self.assertIn("06-ON was produced but not uploaded", self.refused(self.prep10))
        self.clock = datetime(2026, 11, 23, 0, 0, tzinfo=TW)
        e, _ = self.prep10()
        self.assertTrue(e["test10"]["t06_on_produced"])


class T10Final(T10Base):
    def test_final_10_and_others(self):
        _q, off = self.off_uploaded()
        e10, _ = self.up10()
        fc, out = self.run_ol("final", "--comp", "player", "--id", str(e10["id"]), "--basis", "PREREG_10 4.4")
        self.assertEqual(fc["prereg10_4_4"]["chosen"], "10")
        self.assertEqual(fc["prereg10_4_4"]["designated_entries"], [e10["id"]])
        self.assertNotIn("prereg06_5_4", fc)
        self.assertIn("PREREG_10 4.4：最終選擇是 10", out)
        msg = self.refused(self.run_ol, "final", "--comp", "player", "--id", str(off["id"]), "--basis", "x")
        self.assertIn("must be 10", msg)
        msg = self.refused(self.run_ol, "final", "--comp", "player", "--id", str(off["id"]), "--basis", "x",
                           "--not-06-reason", "PREREG_06 rev 4: x")
        self.assertIn("--not-10-reason", msg)
        msg = self.refused(self.run_ol, "final", "--comp", "player", "--id", str(off["id"]), "--basis", "x",
                           "--not-10-reason", "PREREG_10 rev 3: x")
        self.assertIn("not after the records", msg)               # rev 3 (the test-day revision) is older
        self.p10_head = PREREG10_HEAD + "| rev 4 | `0000004` | bug |\n"
        self.rev_added["| rev 4 |"] = ("4" * 40, "2026-11-08T10:00:00+08:00")
        fc, _ = self.run_ol("final", "--comp", "player", "--id", str(off["id"]), "--basis", "x", "--not-10-reason",
                            "PREREG_10 rev 4: t10 swapped the pools")
        self.assertEqual(fc["prereg10_4_4"]["revision"]["rev"], 4)

    def test_10_no_longer_eligible(self):
        _q, off = self.off_uploaded()
        e10, _ = self.up10()
        write_json(self.r(ol.T10_SEAM), {"format": ol.T10_SEAM_FORMAT, "mode": "real", "passed": True, "x": 1})
        fc, _ = self.run_ol("final", "--comp", "player", "--id", str(off["id"]), "--basis", "PREREG_10 4.4")
        self.assertEqual(fc["prereg10_4_4"]["designated"], "off")
        self.assertIn(str(e10["id"]), fc["prereg10_4_4"]["not_eligible_10_entries"])

    def test_status(self):
        _q, off = self.off_uploaded()
        e10, _ = self.up10()
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        s10 = st["player"]["prereg10"]
        self.assertEqual((s10["designated"], s10["designated_entries"]), ("10", [e10["id"]]))
        self.assertFalse(st["player"]["prereg06"]["violation"])
        self.assertIn("記錄有 10", out)
        self.assertIn("最終選擇要設成 10", out)
        self.assertIn("K6-10", out)
        self.assertNotIn("最終選擇要設成 06-OFF", out)
        fin = self.log_obj()
        fin["final_choice"]["player"] = {"id": off["id"], "sha256": off["sha256"], "file": off["file"]}
        with open(self.log, "w", encoding="utf-8", newline="\n") as f:
            json.dump(fin, f)
        st, out = self.run_ol("status", "--now", "2026-11-10T00:00")
        self.assertTrue(st["player"]["prereg10"]["violation"])
        self.assertIn("違反 PREREG_10 4.4", out)
        self.assertNotIn("違反 PREREG_06 5.4", out)


class T10Public(T10Base):
    def test_same_score_warning_and_drop(self):
        _q, off = self.off_uploaded(score=0.90)
        e10, _ = self.up10()
        _e, out = self.run_ol("public", "--id", str(e10["id"]), "--score", "0.90")
        self.assertNotIn("WARNING (PREREG_10 4.4)", out)            # 06-OFF and 10 share the public rows here
        self.edit10(lambda r: None)
        e10b, _ = self.up10("2026-11-08T10:00")
        _e, out = self.run_ol("public", "--id", str(e10b["id"]), "--score", "0.88")   # = Q04's public score
        self.assertIn("WARNING (PREREG_10 4.4)", out)
        self.assertIn("（Q04）", out)
        self.assertIn("37 題", out)

    def test_drop_reminder(self):
        _q, off = self.off_uploaded(score=0.90)
        e10, _ = self.up10()
        _e, out = self.run_ol("public", "--id", str(e10["id"]), "--score", "0.80")
        self.assertIn("INVESTIGATE", out)
        self.assertIn("10 比上一份低 0.05 以上：只查錯", out)
        self.assertNotIn("WARNING (PREREG_10", out)


class TestRevisionRowInRealMergeHistory(unittest.TestCase):
    def test_merge_added_revision_row_is_found_by_k5_lookup(self):
        """K5-10's rev 3 row first exists in a merge commit; exercise real git log, not mocked history."""
        row = "| rev 3 | 【待填】 |"
        with tempfile.TemporaryDirectory(prefix="c10-t10-merge-") as td:
            def git(*args):
                env = dict(os.environ, GIT_AUTHOR_DATE="2026-10-20T10:00:00+08:00",
                           GIT_COMMITTER_DATE="2026-10-20T10:00:00+08:00")
                return subprocess.run(["git", "-C", td, *args], check=True,
                                      capture_output=True, text=True, env=env).stdout.strip()

            git("init", "-b", "main")
            git("config", "user.name", "C10 test")
            git("config", "user.email", "c10-test@example.invalid")
            path = os.path.join(td, *ol.P10_DOC.split("/"))
            os.makedirs(os.path.dirname(path))
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("# PREREG_10\n\n| rev 1 | first |\n")
            git("add", ol.P10_DOC)
            git("commit", "-m", "root")
            root = git("rev-parse", "HEAD")

            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write("| rev 2 | B on main |\n")
            git("add", ol.P10_DOC)
            git("commit", "-m", "B revision on main")
            main = git("rev-parse", "HEAD")

            git("checkout", "-b", "p10-testday", root)
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write("| rev 2 | test-day draft |\n")
            git("add", ol.P10_DOC)
            git("commit", "-m", "test-day revision draft")
            conflict = subprocess.run(["git", "-C", td, "merge", "--no-commit", main],
                                      capture_output=True, text=True)
            self.assertNotEqual(conflict.returncode, 0, "fixture should reproduce the real merge conflict")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("# PREREG_10\n\n| rev 1 | first |\n| rev 2 | B on main |\n" + row + "\n")
            git("add", ol.P10_DOC)
            git("commit", "-m", "merge main; renumber test-day revision")
            merge = git("rev-parse", "HEAD")

            with mock.patch.object(ol, "REPO", td):
                self.assertEqual(ol.git_row_added(ol.P10_DOC, row), (None, None))
                found, committed_at = ol.git_row_added_any(ol.P10_DOC, row)
                self.assertEqual(found, merge)
                self.assertTrue(committed_at)
                state = ol.t10_k5_state()
                self.assertEqual(state["commit"], merge)
                self.assertTrue(state["in_head"])
            with mock.patch.object(t10p, "REPO", td), mock.patch.object(t10p.t6, "REPO", td):
                player_state = t10p.k5_10_state({"real": True})
                self.assertEqual(player_state["commit"], merge)
                self.assertTrue(player_state["in_head"])
                self.assertTrue(player_state["ok"])


if __name__ == "__main__":
    unittest.main()
