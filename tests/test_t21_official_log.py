"""Tests for the PREREG_21 additions to tools/official_log.py (docs/PREREG_21_PLAYER_EXACT_COPY.md sections 5, 7, 8).

* TagVersusNew / TagVersusNew07Flow / 08Flow / 09Flow / 14Flow: test_t19_official_log's comparisons of the version of
  main c398d23 (before PREREG_21) with the working-tree version, now with a results/test21 tree on disk.
* PlayerFlowVetoed / PlayerFlowKept: competition 1's flows (Q04 -> 06-OFF -> 06-ON -> veto -> 10 or 06-ON kept, public
  scores, finals, refused prepares, status) run with both versions: log bytes, returned records, stdout and refusals
  are identical (a log without 21 entries).
* T21*: the 21 path: prepare (the file recognised by its folder, the report's own conditions re-read, the base equal
  to final_choice.player and still designated by PREREG_06 5.4 / PREREG_10 4.4 on the log without 21 entries, K5-21,
  K6-21, --method), final (the 06 / 10 rules judged on the base, then the 21 entry; another choice only with
  --not-21-reason), status (no false 06 / 10 violation while 21 is final, also on a 06-ON base), public (the same-file
  warnings; no 06 / 10 warning about a 21 entry), and the report schema of tools/t21_copy.py accepted by official_log.
Synthetic files in a temporary directory only; git and the clock are mocked as in test_t10_official_log. Nothing here
opens a label file or runs tools/score.py.
Run: python -m pytest tests/test_t21_official_log.py
"""

import json
import os
import sys
import tempfile
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
import test_t10_official_log as t10o  # noqa: E402
import test_t14_official_log as t14o  # noqa: E402
import test_t19_official_log as t19o  # noqa: E402

BASE_REF = "c398d23"                               # main before PREREG_21 (the branch point of p21-copy)
sha, write_json = t4o.sha, t4o.write_json
PREREG21_HEAD = "# PREREG_21\n\n| rev 1 | `be180d9` | first |\n| rev 2 | (commit) | implementation |\n"
REV2_ON_MAIN = ("2" * 40, "2026-10-21T10:00:00+08:00")      # before K5-21 (11/03 23:59:59)
T21_REPORT_REL = "results/test21/player/" + ol.T21_REPORT


def changed_default(base_rows):
    """Two changed questions (one per pool) plus one already-first override: {qid: candidate}."""
    pub = next(q for q in base_rows if q.startswith("pub_"))
    priv = next(q for q in base_rows if q.startswith("priv_"))
    pre_pub, pre_priv = pub.split("_")[0], priv.split("_")[0]
    return {pub: f"{pre_pub}_p_0123", priv: f"{pre_priv}_p_0321"}


# ---------------------------------------------------------------- the results/test21 tree of a usable 21 run
def write_t21_tree(repo, base_rel, base_entry_id, changed=None, same_first=(), report_over=None):
    """results/test21 as tools/t21_copy.py apply leaves it (real mode) on the base file base_rel; changed: {qid:
    candidate} for overridden rows that change; same_first: overridden questions whose candidate already is first."""
    r = lambda rel: os.path.join(repo, *rel.split("/"))   # noqa: E731
    with open(r(base_rel), encoding="utf-8", newline="") as f:
        rows = [line.rstrip("\r\n").split(",") for line in f.read().splitlines()][1:]
    base = {x[0]: x[1:] for x in rows}
    changed = dict(changed_default(base) if changed is None else changed)
    new = {q: [c] + [x for x in base[q] if x != c][:4] for q, c in changed.items()}
    f21 = ti.write_csv(r("results/test21/player/submission_player.csv"),
                       ["question_id"] + [f"top_{i}" for i in range(1, 6)],
                       [[x[0]] + new.get(x[0], x[1:]) for x in rows])
    write_json(r(ol.T21_SIM), {"format": "c10-test21 sim v1", "evaluation": {"passed": True}})
    files = {n: {"sha256": sha(r(f"data/test/{n}"))} for n in ol.QUESTION_FILES["player"]}
    pools = {p: {"file": n, "sha256": sha(r(f"data/test/{n}"))} for p, n in zip(("public", "private"),
                                                                                  ol.CANDIDATE_FILES)}
    overrides = [{"pool": "public" if q.startswith("pub_") else "private", "question_id": q, "candidate": c,
                  "copy_slots": [1], "copy_games": [["g"]], "base_top5": base[q], "new_top5": new[q], "changed": True,
                  "base_rank_of_candidate": None} for q, c in changed.items()]
    overrides += [{"pool": "public", "question_id": q, "candidate": base[q][0], "copy_slots": [2],
                   "copy_games": [["h"]], "base_top5": base[q], "new_top5": base[q], "changed": False,
                   "base_rank_of_candidate": 1} for q in same_first]
    by_pool = {}
    for o in overrides:
        if o["changed"]:
            by_pool[o["pool"]] = by_pool.get(o["pool"], 0) + 1
    rep = {"format": ol.T21_FORMAT, "mode": "real", "method": ol.T21_METHOD, "verdict": "used",
           "preflight": {"manifest_sha256": "m", "t1": {"verdict": "PASS"},
                         "k5_and_sim": {"k5_ok": True, "k5_21": "2026-11-03T23:59:59+08:00"}},
           "base": {"entry_id": base_entry_id, "file": base_rel, "sha256": sha(r(base_rel)), "method": "x"},
           "pools": pools, "files": files,
           "questions_overridden": len(overrides), "rows_changed": len(changed), "changed_by_pool": by_pool,
           "overrides": overrides, "conflicts": [],
           "check_submission": {"mode": "official", "errors": []},
           "outputs_sha256": {"submission_player.csv": sha(f21)}}
    rep.update(report_over or {})
    write_json(r(T21_REPORT_REL), rep)
    return f21


def t21_tree_on_q04(repo):
    p = os.path.join(repo, "results", "test02", "player", "submission_player.csv")
    if os.path.exists(p):
        write_t21_tree(repo, "results/test02/player/submission_player.csv", 1)


# ---------------------------------------------------------------- main before PREREG_21 against the new one
class TagVersusNew(t19o.TagVersusNew):
    """A log without 21 entries: the c398d23 version's log bytes, records, stdout and refusals, with results/test19
    and results/test21 trees on disk (the R20 / R10 / 04 / 06 flows)."""

    @classmethod
    def setUpClass(cls):
        cls.tag = t7o.load_version(BASE_REF)
        assert hasattr(cls.tag, "check_reports_t19") and not hasattr(cls.tag, "check_reports_t21"), "not the base"

    def build(self, kind, with_t06):
        out = super().build(kind, with_t06)
        t21_tree_on_q04(out[0])
        return out


class _Flow21:
    def setUp(self):
        self.tag = t7o.load_version(BASE_REF)

    def run_with(self, mod):
        real = t9o.write_t09_tree

        def both(repo, rows, *a, **k):
            out = real(repo, rows, *a, **k)
            t21_tree_on_q04(repo)
            return out
        with mock.patch.object(t9o, "write_t09_tree", side_effect=both):
            return super().run_with(mod)

    def check_identical(self):
        old = super().check_identical()
        log = json.loads(old[-1][3])
        self.assertFalse(any("test21" in e or "prereg21_7" in e for e in log["uploads"]))
        self.assertNotIn("prereg21_7", json.dumps(log["final_choice"]))
        return old


class TagVersusNew07Flow(_Flow21, t19o.TagVersusNew07Flow):
    pass


class TagVersusNew08Flow(_Flow21, t19o.TagVersusNew08Flow):
    pass


class TagVersusNew09Flow(_Flow21, t19o.TagVersusNew09Flow):
    pass


class TagVersusNew14Flow(_Flow21, t19o.TagVersusNew14Flow):
    pass


class _PlayerFlow(t10o.T10Base):
    """Competition 1's flows with the c398d23 version and with the new one (the test helpers' module references
    patched), a results/test21 tree on the 10 file on disk that is never prepared."""
    on_produced = True

    def setUp(self):
        self.tag = t7o.load_version(BASE_REF)

    def tearDown(self):
        pass

    def steps(self, step):
        raise NotImplementedError

    def run_with(self, mod):
        patches = [mock.patch.object(m, "ol", mod) for m in (ti, t4o, t6o, t10o)]
        for pt in patches:
            pt.start()
        try:
            t10o.T10Base.setUp(self)
            write_t21_tree(self.repo, "results/test10/player/submission_player.csv", 99)
            out = []

            def step(name, fn, *a):
                try:
                    res, text = fn(*a)
                except SystemExit as e:
                    res, text = ("refused", str(e)), ""
                js = json.dumps(res, sort_keys=True, ensure_ascii=False)
                for root in (self.repo, json.dumps(self.repo)[1:-1]):
                    text, js = text.replace(root, "<REPO>"), js.replace(root, "<REPO>")
                with open(self.log, "rb") as f:
                    out.append((name, text, js, f.read()))
            self.steps(step)
            return out
        finally:
            t10o.T10Base.tearDown(self)
            for pt in reversed(patches):
                pt.stop()

    def check_identical(self):
        old, new = self.run_with(self.tag), self.run_with(ol)
        self.assertEqual([s[0] for s in old], [s[0] for s in new])
        for (name, o_out, o_res, o_log), (_n, n_out, n_res, n_log) in zip(old, new):
            with self.subTest(step=name):
                self.assertEqual(n_log, o_log)
                self.assertEqual(n_out, o_out)
                self.assertEqual(n_res, o_res)
        log = json.loads(old[-1][3])
        self.assertFalse(any("test21" in e or "prereg21_7" in e for e in log["uploads"]))
        return old

    def first_steps(self, step):
        step("prepare Q04", self.prepare, "player", self.player_file)
        step("uploaded Q04", self.up, 1, "2026-11-04T16:05")
        step("public Q04", self.run_ol, "public", "--id", "1", "--score", "0.88")
        step("prepare 06-OFF", self.prep06, "off")
        step("uploaded 06-OFF", self.up, 2, "2026-11-05T10:00")
        step("public 06-OFF", self.run_ol, "public", "--id", "2", "--score", "0.90")
        step("prepare 10 refused (06-ON pending)", self.prep10)
        step("prepare 06-ON", self.prep06, "on")
        step("uploaded 06-ON", self.up, 3, "2026-11-05T11:00")


class PlayerFlowVetoed(_PlayerFlow):
    def steps(self, step):
        self.first_steps(step)
        step("public 06-ON", self.run_ol, "public", "--id", "3", "--score", "0.89")
        step("final undecided refused", self.run_ol, "final", "--comp", "player", "--id", "2", "--basis", "x")
        step("veto", lambda: (self.write_veto(2, 3, -0.01) and None, ""))
        step("prepare 10", self.prep10)
        step("uploaded 10", self.up, 4, "2026-11-07T10:00")
        step("public 10", self.run_ol, "public", "--id", "4", "--score", "0.90")
        step("final 06-OFF refused", self.run_ol, "final", "--comp", "player", "--id", "2", "--basis", "x")
        step("final 10", self.run_ol, "final", "--comp", "player", "--id", "4", "--basis", "PREREG_10 4.4")
        step("prepare Q04 refused", self.prepare, "player", self.player_file)
        step("status", self.run_ol, "status", "--now", "2026-11-10T00:00")

    def test_identical(self):
        old = self.check_identical()
        self.assertEqual(json.loads(old[16][2])["prereg10_4_4"]["chosen"], "10")
        self.assertIn("refused", old[15][2])
        self.assertTrue(any("test10" in e for e in json.loads(old[-1][3])["uploads"]), "the flow must hold a 10 entry")


class PlayerFlowKept(_PlayerFlow):
    def steps(self, step):
        self.first_steps(step)
        step("public 06-ON", self.run_ol, "public", "--id", "3", "--score", "0.91")
        step("veto", lambda: (self.write_veto(2, 3, 0.01) and None, ""))
        step("prepare 10 refused (kept)", self.prep10)
        step("final 06-OFF refused", self.run_ol, "final", "--comp", "player", "--id", "2", "--basis", "x")
        step("final 06-ON", self.run_ol, "final", "--comp", "player", "--id", "3", "--basis", "PREREG_06 5.4")
        step("status", self.run_ol, "status", "--now", "2026-11-10T00:00")

    def test_identical(self):
        old = self.check_identical()
        self.assertEqual(json.loads(old[13][2])["prereg06_5_4"]["chosen"], "on")
        self.assertIn("06-ON is kept", old[11][2])


# ---------------------------------------------------------------- the 21 path
class T21Base(t10o.T10Base):
    on_produced = False

    def setUp(self):
        super().setUp()
        self.p21_head = PREREG21_HEAD
        self.rev2_on_main = REV2_ON_MAIN
        self.rev21 = {}                               # rows of docs/PREREG_21 (PREREG_10 has its own '| rev 3 |')
        prev = ol.git_head_text                       # the 06 / 10 mock of T10Base
        prev_row = ol.git_row_added
        self.patches21 = [mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p21_head
                                            if rel == ol.P21_DOC else prev(rel)),
                          mock.patch.object(ol, "git_row_added_on_main", side_effect=lambda rel, row: self.rev2_on_main
                                            if rel == ol.P21_DOC and row == "| rev 2 |" else (None, None)),
                          mock.patch.object(ol, "git_row_added", side_effect=lambda rel, row: self.rev21.get(
                              row, (None, None)) if rel == ol.P21_DOC else prev_row(rel, row))]
        for pt in self.patches21:
            pt.start()

    def tearDown(self):
        for pt in reversed(self.patches21):
            pt.stop()
        super().tearDown()

    def edit21(self, fn):
        p = self.r(T21_REPORT_REL)
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def prepare21(self, *extra):
        return self.prepare("player", self.f21, "prereg21 exact-copy layer (PREREG_21 7)", *extra)

    def up21(self, at="2026-11-21T10:00"):
        e21, out = self.prepare21()
        self.up(e21["id"], at)
        return e21, out

    def final(self, eid, *extra):
        return self.run_ol("final", "--comp", "player", "--id", str(eid), "--basis", "x", *extra)

    def ready_10(self, final=True, **kw):
        """Q04 #1, 06-OFF #2, 10 #3 uploaded, final 10, the 21 tree on the 10 file."""
        _q, _off = self.off_uploaded()
        e10, _ = self.up10()
        if final:
            self.run_ol("final", "--comp", "player", "--id", str(e10["id"]), "--basis", "PREREG_10 4.4")
        self.f21 = write_t21_tree(self.repo, "results/test10/player/submission_player.csv", e10["id"], **kw)
        return e10

    def rev3_21(self, at="2026-11-22T10:00:00+08:00"):
        self.p21_head = PREREG21_HEAD + "| rev 3 | `abc` | bug |\n"
        self.rev21["| rev 3 |"] = ("e" * 40, at)


class T21Prepare(T21Base):
    def test_prepare_21(self):
        e10 = self.ready_10(same_first=["pub_q_0400"])
        e21, out = self.prepare21()
        self.assertEqual((e21["id"], e21["file"]), (4, "results/test21/player/submission_player.csv"))
        self.assertEqual(e21["method"], ol.T21_METHOD)
        t21 = e21["test21"]
        self.assertEqual((t21["base_entry_id"], t21["base_file"]), (e10["id"], "results/test10/player/submission_player.csv"))
        self.assertEqual((t21["rows_changed"], t21["questions_overridden"]), (2, 3))
        self.assertEqual(t21["changed_by_pool"], {"public": 1, "private": 1})
        self.assertEqual(t21["base_rules"], {"prereg10_4_4": "10"})
        self.assertEqual(t21["k6_21"]["deadline"], "2026-11-23T18:00:00+08:00")
        for k in ("prereg06_5_4", "prereg10_4_4", "test06", "test10", "prereg21_7"):
            self.assertNotIn(k, e21)
        self.assertIn("這一份是 21", out)
        self.assertIn("public 1、private 1", out)

    def test_before_final_refused(self):
        self.ready_10(final=False)
        self.assertIn("is not final_choice.player", self.refused(self.prepare21))

    def test_base_no_longer_designated(self):
        self.ready_10()
        write_json(self.r(ol.T10_SEAM), {"format": ol.T10_SEAM_FORMAT, "mode": "real", "passed": True, "x": 1})
        self.assertIn("no longer the file the competition-1 rules designate", self.refused(self.prepare21))

    def test_report_conditions(self):
        self.ready_10()
        cases = [("mode", "dry-run", "mode 'dry-run'"), ("verdict", "not_used", "verdict 'not_used'"),
                 ("rows_changed", 3, "rows_changed 3"),
                 ("check_submission", {"mode": "official", "errors": ["x"]}, "check_submission"),
                 ("preflight", {"t1": {"verdict": "FAIL_OPENING"}}, "T1 'FAIL_OPENING'"),
                 ("format", "x", "format 'x'")]
        for key, val, want in cases:
            with self.subTest(key=key, val=val):
                snap = self.r(T21_REPORT_REL)
                with open(snap, "rb") as f:
                    keep = f.read()
                self.edit21(lambda rep: rep.update({key: val}))
                self.assertIn(want, self.refused(self.prepare21))
                with open(snap, "wb") as f:
                    f.write(keep)

    def test_override_rule_and_rows(self):
        self.ready_10()
        self.edit21(lambda rep: rep["overrides"][0].update(new_top5=list(reversed(rep["overrides"][0]["new_top5"]))))
        self.assertIn("are not 'the candidate first, then the base row'", self.refused(self.prepare21))
        self.f21 = write_t21_tree(self.repo, "results/test10/player/submission_player.csv", 3)
        self.edit21(lambda rep: rep["overrides"][0].update(base_top5=["pub_x"] * 5, new_top5=["pub_p_0123"]
                                                           + ["pub_x"] * 4))
        self.assertIn("is not the base file's row", self.refused(self.prepare21))
        self.f21 = write_t21_tree(self.repo, "results/test10/player/submission_player.csv", 3,
                                  changed={"pub_q_0001": "pub_p_0123", "priv_q_0001": "priv_p_0321",
                                           "pub_q_0002": "pub_p_0200"})
        self.edit21(lambda rep: rep.update(rows_changed=2, overrides=[o for o in rep["overrides"]
                                                                       if o["question_id"] != "pub_q_0002"]))
        self.assertIn("differs from its base in other rows", self.refused(self.prepare21))

    def test_test_files_and_base_bytes(self):
        self.ready_10()
        self.edit21(lambda rep: rep["pools"]["public"].update(sha256="0" * 64))
        self.assertIn("is not the file the 21 report read", self.refused(self.prepare21))
        self.f21 = write_t21_tree(self.repo, "results/test10/player/submission_player.csv", 3)
        self.edit21(lambda rep: rep["base"].update(sha256="0" * 64))
        self.assertIn("no longer the base bytes", self.refused(self.prepare21))
        self.f21 = write_t21_tree(self.repo, "results/test10/player/submission_player.csv", 3,
                                  report_over={"base": {"entry_id": 3, "file": "results/test09/rank/x.csv"}})
        self.assertIn("is not in", self.refused(self.prepare21))

    def test_k5_k6_sim(self):
        self.ready_10()
        self.commits[T21_REPORT_REL] = ("f" * 40, "2026-11-23T18:00:01+08:00")
        self.assertIn("K6-21", self.refused(self.prepare21))
        self.commits.pop(T21_REPORT_REL)
        self.rev2_on_main = ("2" * 40, "2026-11-04T00:00:00+08:00")
        self.assertIn("K5-21", self.refused(self.prepare21))
        self.rev2_on_main = (None, None)
        self.assertIn("K5-21", self.refused(self.prepare21))
        self.rev2_on_main = REV2_ON_MAIN
        self.edit21(lambda rep: rep["preflight"].update(k5_and_sim={"k5_ok": False}))
        self.assertIn("K5-21 was not confirmed", self.refused(self.prepare21))
        self.f21 = write_t21_tree(self.repo, "results/test10/player/submission_player.csv", 3)
        write_json(self.r(ol.T21_SIM), {"evaluation": {"passed": False}})
        self.assertIn("simulation did not pass", self.refused(self.prepare21))

    def test_method_and_other_reasons(self):
        self.ready_10()
        self.assertIn("does not name prereg21", self.refused(self.prepare21, "--method", "prereg10 10"))
        for flag in ("--not-06-reason", "--not-10-reason", "--not-21-reason"):
            with self.subTest(flag=flag):
                self.assertIn(flag, self.refused(self.prepare21, flag, "PREREG_10 rev 5: x"))

    def test_other_player_file_after_21_needs_reason(self):
        self.ready_10()
        self.up21()
        self.assertIn("--not-21-reason", self.refused(self.prep10))
        self.rev3_21()
        e, out = self.prep10("--not-21-reason", "PREREG_21 rev 3: bug")
        self.assertEqual(e["prereg21_7"]["revision"]["rev"], 3)
        self.assertIn("這一份不是 21", out)

    def test_rank_file_untouched_by_21(self):
        self.ready_10()
        self.up21()
        msg = self.refused(self.prepare, "rank", self.rank_file, "x", "--not-21-reason", "PREREG_21 rev 3: x")
        self.assertIn("--not-21-reason applies to a competition-1 file", msg)


class T21Final(T21Base):
    def test_final_21_and_status(self):
        e10 = self.ready_10()
        e21, _ = self.up21()
        fc, out = self.final(e21["id"])
        self.assertEqual(fc["id"], e21["id"])
        self.assertEqual(fc["prereg21_7"]["chosen"], "21")
        self.assertEqual(fc["prereg21_7"]["base_entry_id"], e10["id"])
        self.assertEqual(fc["prereg10_4_4"]["chosen"], "10")           # the 10 rule judged the base
        self.assertIn("PREREG_21 7：最終選擇是 21", out)
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        for k in ("prereg06", "prereg10", "prereg21"):
            self.assertFalse(st["player"][k]["violation"], k)
        self.assertTrue(st["player"]["prereg21"]["final_is_21"])
        self.assertTrue(st["player"]["prereg10"]["last_is_designated"])
        self.assertTrue(st["player"]["last_is_final"])
        self.assertNotIn("違反", out)
        self.assertNotIn("最終選擇要設成", out)
        self.assertNotIn("再上傳一次", out)
        self.assertIn("K6-21", out)
        fc, _ = self.final(e21["id"])                                  # 11/22-23: rerun final on 21
        self.assertEqual(fc["prereg21_7"]["chosen"], "21")

    def test_final_21_before_upload_warns(self):
        self.ready_10()
        e21, _ = self.prepare21()
        _fc, out = self.final(e21["id"])
        self.assertIn("還沒上傳", out)
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertFalse(st["player"]["last_is_final"])
        self.assertIn("最後一次上傳不是它", out)

    def test_final_base_after_21_needs_reason(self):
        e10 = self.ready_10()
        e21, _ = self.up21()
        self.assertIn("needs --not-21-reason", self.refused(self.final, e10["id"]))
        self.assertIn("rev 3 or later", self.refused(self.final, e10["id"], "--not-21-reason", "PREREG_21 rev 2: x"))
        self.rev3_21(at="2026-11-04T10:00:00+08:00")                 # before the entries were prepared (14:00)
        self.assertIn("not after the records", self.refused(self.final, e10["id"], "--not-21-reason",
                                                            "PREREG_21 rev 3: bug in t21_copy"))
        self.rev3_21()
        fc, _ = self.final(e10["id"], "--not-21-reason", "PREREG_21 rev 3: bug in t21_copy")
        self.assertEqual(fc["prereg21_7"]["chosen"], "not 21")
        st, _out = self.run_ol("status", "--now", "2026-11-23T00:00")
        self.assertFalse(st["player"]["prereg21"]["violation"])
        self.assertTrue(e21)

    def test_status_violation_without_final_21(self):
        self.ready_10()
        self.up21()
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertTrue(st["player"]["prereg21"]["violation"])
        self.assertIn("違反 PREREG_21 7", out)

    def test_21_no_longer_eligible(self):
        self.ready_10()
        e21, _ = self.up21()
        self.final(e21["id"])
        write_json(self.r(ol.T10_SEAM), {"format": ol.T10_SEAM_FORMAT, "mode": "real", "passed": True, "x": 1})
        self.assertIn("PREREG_10 4.4: the final choice must be 06-OFF", self.refused(self.final, e21["id"]))
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertTrue(st["player"]["prereg21"]["redo_base"])
        self.assertIn("21 不再符合", out)
        with open(self.r("results/test21/player/submission_player.csv"), "ab") as f:
            f.write(b"\n")
        ok, why = ol.t21_entry_eligible(ol.entry(self.log_obj(), e21["id"]), self.log_obj())
        self.assertFalse(ok)
        self.assertTrue(any("changed since prepare" in w for w in why))


class T21OnQ04(T21Base):
    def test_21_on_q04_without_06(self):
        q = self.q04_uploaded()
        self.run_ol("final", "--comp", "player", "--id", str(q["id"]), "--basis", "U5")
        self.f21 = write_t21_tree(self.repo, "results/test02/player/submission_player.csv", q["id"])
        e21, _ = self.up21()
        self.assertEqual(e21["test21"]["base_rules"], {})
        fc, _ = self.final(e21["id"])
        self.assertEqual(fc["prereg21_7"]["chosen"], "21")
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertFalse(st["player"]["prereg21"]["violation"])
        self.assertNotIn("違反", out)


class T21On06On(T21Base):
    """PREREG_21 5: 21 on a kept 06-ON. The 06 rules judge the 21 on the log without 21 entries (06-ON stays the last
    upload there); set aside by a revision, the final is 06-OFF (06-ON cannot be uploaded again, PREREG_06 5.5 E1)."""
    on_produced = True

    def kept(self):
        _q, off, on = self.full_flow(off_score=0.90, on_score=0.91)
        self.write_veto(off["id"], on["id"], 0.01)
        self.run_ol("final", "--comp", "player", "--id", str(on["id"]), "--basis", "PREREG_06 5.4")
        self.f21 = write_t21_tree(self.repo, "results/test06/player/on/submission_player.csv", on["id"])
        return off, on

    def test_21_on_kept_06_on(self):
        off, on = self.kept()
        self.assertIn("06-ON is kept", self.refused(self.prep10))
        e21, _ = self.up21()
        self.assertEqual(e21["test21"]["base_rules"], {"prereg06_5_4": "on"})
        fc, _ = self.final(e21["id"])
        self.assertEqual((fc["prereg06_5_4"]["chosen"], fc["prereg21_7"]["chosen"]), ("on", "21"))
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertFalse(st["player"]["prereg06"]["violation"])
        self.assertFalse(st["player"]["prereg06"]["reupload_off_needed"])
        self.assertFalse(st["player"]["prereg21"]["violation"])
        self.assertNotIn("違反", out)
        self.assertNotIn("再上傳一次 06-OFF", out)
        # 21 set aside: 06-ON is no longer the last upload -> 06-OFF (re-uploaded), with the revision
        self.assertIn("must be 06-OFF", self.refused(self.final, on["id"], "--not-21-reason", "PREREG_21 rev 3: x"))
        self.rev3_21()
        fc, _ = self.final(off["id"], "--not-21-reason", "PREREG_21 rev 3: bug")
        self.assertEqual((fc["prereg06_5_4"]["chosen"], fc["prereg21_7"]["chosen"]), ("off", "not 21"))
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertTrue(st["player"]["prereg06"]["reupload_off_needed"])
        self.assertIn("再上傳一次 06-OFF", out)
        self.assertFalse(st["player"]["prereg21"]["violation"])

    def test_21_on_off_when_vetoed(self):
        _q, off, on = self.full_flow(off_score=0.90, on_score=0.89)
        self.write_veto(off["id"], on["id"], -0.01)
        self.up(self.prep06("off")[0]["id"], "2026-11-06T10:00")      # 06-OFF once more (step 8)
        last_off = self.log_obj()["uploads"][-1]
        self.run_ol("final", "--comp", "player", "--id", str(last_off["id"]), "--basis", "PREREG_06 5.4")
        self.f21 = write_t21_tree(self.repo, "results/test06/player/off/submission_player.csv", last_off["id"])
        e21, _ = self.up21()
        fc, _ = self.final(e21["id"])
        self.assertEqual((fc["prereg06_5_4"]["chosen"], fc["prereg21_7"]["chosen"]), ("off", "21"))
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertFalse(st["player"]["prereg06"]["violation"])
        self.assertFalse(st["player"]["prereg06"]["reupload_off_needed"])
        self.assertNotIn("違反", out)


class T21BeforeFinalAndBaseChange(T21Base):
    """Review 10/20 I1 / I2: between 'uploaded 21' and 'final --id <21>' the 06 / 10 blocks judge the base (no
    'upload again'); after the base changed and a new 21 on the new base became final, status agrees with final."""

    def test_status_before_final_21(self):
        self.ready_10()
        self.up21()
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertFalse(st["player"]["prereg10"]["violation"])
        self.assertTrue(st["player"]["prereg10"]["last_is_designated"])
        self.assertTrue(st["player"]["prereg21"]["violation"])
        self.assertIn("違反 PREREG_21 7", out)
        self.assertNotIn("再上傳一次", out)
        self.assertNotIn("違反 PREREG_10", out)

    def test_base_changed_then_new_21(self):
        self.ready_10()
        e21, _ = self.up21()
        self.final(e21["id"])
        write_json(self.r(ol.T10_SEAM), {"format": ol.T10_SEAM_FORMAT, "mode": "real", "passed": True, "x": 1})
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertTrue(st["player"]["prereg21"]["redo_base"])
        fc, _ = self.final(2)                                         # 06-OFF, no --not-21-reason: the 21 is moot
        self.assertEqual(fc["prereg10_4_4"]["chosen"], "off")
        self.assertNotIn("revision", fc.get("prereg21_7") or {})
        self.f21 = write_t21_tree(self.repo, "results/test06/player/off/submission_player.csv", 2)
        e21b, _ = self.up21("2026-11-22T12:00")
        self.assertEqual(e21b["test21"]["base_rules"], {"prereg10_4_4": "off"})
        fc, _ = self.final(e21b["id"])
        self.assertEqual((fc["prereg10_4_4"]["chosen"], fc["prereg21_7"]["chosen"]), ("off", "21"))
        st, out = self.run_ol("status", "--now", "2026-11-23T00:00")
        for k in ("prereg06", "prereg10", "prereg21"):
            self.assertFalse(st["player"][k]["violation"], k)
        self.assertTrue(st["player"]["last_is_final"])
        self.assertNotIn("再上傳一次", out)
        self.assertNotIn("違反", out)

    def test_moot_06_reason_with_10_entries(self):
        self.ready_10()
        log = self.log_obj()
        got = ol.t21_moot_reasons(log, {"not_06_reason": "PREREG_06 rev 4: x", "not_10_reason": None})
        self.assertIsNone(got["not_06_reason"])
        no10 = dict(log, uploads=[x for x in log["uploads"] if not x.get("test10")])
        self.assertEqual(ol.t21_moot_reasons(no10, {"not_06_reason": "r"})["not_06_reason"], "r")


class T21On06OnBeforeFinal(T21Base):
    on_produced = True

    def test_no_reupload_off_before_final_21(self):
        T21On06On.kept(self)
        self.up21()
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertFalse(st["player"]["prereg06"]["violation"])
        self.assertFalse(st["player"]["prereg06"]["reupload_off_needed"])
        self.assertNotIn("再上傳一次 06-OFF", out)
        self.assertIn("違反 PREREG_21 7", out)


class T21Public(T21Base):
    def test_same_file_warnings(self):
        e10 = self.ready_10()
        self.run_ol("public", "--id", str(e10["id"]), "--score", "0.90")
        e21, _ = self.up21()
        _e, out = self.run_ol("public", "--id", str(e21["id"]), "--score", "0.90")
        self.assertIn("WARNING (PREREG_21)", out)
        for w in ("WARNING (PREREG_10 4.4)", "WARNING (PREREG_06"):
            self.assertNotIn(w, out)

    def test_21_equal_to_q04_score_no_10_warning(self):
        e10 = self.ready_10()
        self.run_ol("public", "--id", str(e10["id"]), "--score", "0.90")
        e21, _ = self.up21()
        _e, out = self.run_ol("public", "--id", str(e21["id"]), "--score", "0.88")      # = Q04's public score
        self.assertNotIn("WARNING (PREREG_10 4.4)", out)
        self.assertNotIn("WARNING (PREREG_21)", out)

    def test_no_public_change_but_other_score(self):
        e10 = self.ready_10(changed={"priv_q_0002": "priv_p_0321"})
        self.run_ol("public", "--id", str(e10["id"]), "--score", "0.90")
        e21, _ = self.up21()
        _e, out = self.run_ol("public", "--id", str(e21["id"]), "--score", "0.91")
        self.assertIn("沒有改任何 public 題", out)


class ToolReportAccepted(unittest.TestCase):
    """The report tools/t21_copy.py apply writes (dry-run, then marked real as a real run would be) passes
    official_log's t21_report_problems: the two files agree on the schema."""

    def test_schema(self):
        import test_t21_copy as tt
        import t21_copy as tc
        with tempfile.TemporaryDirectory(prefix="c10_t21_schema_") as root:
            _b, lib = tt.build_tree(root, {})
            tt.build_tree(root, {("public", 3): [lib[("public", 7)]], ("private", 5): [lib[("private", 6)]]})
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                self.assertEqual(tc.main(["apply", "--dry-run", "--root", root]), 0)
            rp = os.path.join(root, "results", "test21", "player", "t21_copy_report.json")
            rep = tc.read_json(rp)
            rep["mode"] = "real"
            rep["preflight"] = {"t1": {"verdict": "PASS"}}
            f21 = os.path.join(root, "results", "test21", "player", "submission_player.csv")
            with mock.patch.object(ol, "REPO", root):
                self.assertEqual(ol.t21_report_problems(rep, "r", f21, ol.sha256_file(f21)), [])
                rep["overrides"][0]["changed"] = not rep["overrides"][0]["changed"]
                self.assertTrue(ol.t21_report_problems(rep, "r", f21, ol.sha256_file(f21)))


class Constants(unittest.TestCase):
    def test_agree_with_the_tool(self):
        import t21_copy as tc
        self.assertEqual((ol.K5_21, ol.K6_21), (tc.K5_21, tc.K6_21))
        self.assertEqual((ol.T21_FORMAT, ol.T21_METHOD, ol.T21_REPORT), (tc.FORMAT_REPORT, tc.METHOD,
                                                                          "t21_copy_report.json"))
        self.assertEqual(ol.T21_BASE_DIRS, tc.BASE_DIRS)
        self.assertEqual(ol.P21_DOC, tc.DOC)
        self.assertEqual(datetime(2026, 11, 3, 23, 59, 59, tzinfo=t6o.TW), ol.K5_21)


if __name__ == "__main__":
    unittest.main()
