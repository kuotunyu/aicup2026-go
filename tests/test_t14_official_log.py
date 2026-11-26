"""Tests for the PREREG_14 additions to tools/official_log.py (docs/PREREG_14_RANK_EXACT_COPY.md section 7).

* TagVersusNew / TagVersusNew07Flow / TagVersusNew08Flow / TagVersusNew09Flow: the version of main be7dc4d (before
  PREREG_14) and the working-tree version run the same flows with the same mocks (the R20 / R10 / 04 / 06 flows, the
  07 and 08 flows of test_t10_official_log, and a full 09 flow: R20 -> 04 -> 07 -> 08 -> 09 uploaded, public scores,
  final 09, status) with a results/test14 tree on disk: log bytes, returned records, stdout and refusals are identical
  (a log without 14 entries).
* T14*: the 14 path: prepare (the file recognised by its folder, the report's own conditions re-read, the base equal
  to final_choice.rank and still designated by the 04-09 rules, K6-14, --method), final (the 04-09 rules judged on the
  base, then the 14 entry; another choice only with --not-14-reason), status (no false 04-09 violation while 14 is
  final), public (the same-file warnings), and the report schema of tools/t14_copy.py accepted by official_log.
Synthetic files in a temporary directory only; git and the clock are mocked as in test_t09_official_log. Nothing here
opens a label file or runs tools/score.py.
Run: python -m pytest tests/test_t14_official_log.py
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)
import official_log as ol  # noqa: E402
import test_t02_infra as ti  # noqa: E402
import test_t04_official_log as t4o  # noqa: E402
import test_t07_official_log as t7o  # noqa: E402
import test_t09_official_log as t9o  # noqa: E402
import test_t10_official_log as t10o  # noqa: E402

BASE_REF = "be7dc4d"                               # main before PREREG_14 (the branch point of p14-copy)
sha, write_json, quiet = t4o.sha, t4o.write_json, t4o.quiet
PREREG14_HEAD = "# PREREG_14\n\n| rev 1 | `86f882b` | first |\n| rev 2 | (commit) | implementation |\n"
CHANGED = {"pub_q_0002": "6D", "priv_q_0002": "6D"}


# ---------------------------------------------------------------- the results/test14 tree of a usable 14 run
def write_t14_tree(repo, base_rel, base_entry_id, changed=None, report_over=None, same_as_base=()):
    """results/test14 as tools/t14_copy.py apply leaves it (real mode) on the base file base_rel; changed: {qid: new
    level} for the rows that differ from the base; same_as_base: overridden questions whose level is the base's."""
    r = lambda rel: os.path.join(repo, *rel.split("/"))   # noqa: E731
    changed = dict(CHANGED if changed is None else changed)
    with open(r(base_rel), encoding="utf-8", newline="") as f:
        rows = [line.rstrip("\r\n").split(",") for line in f.read().splitlines()][1:]
    base = {q: lv for q, lv in rows}
    new_rows = [[q, changed.get(q, lv)] for q, lv in rows]
    f14 = ti.write_csv(r("results/test14/rank/submission_rank.csv"), ["question_id", "pred_rank"], new_rows)
    pins = {"rows": 1000000, "players": 8007, "distinct_full_keys": 1000000, "keys_with_two_or_more_rows": 0,
            "sha256_sorted_key_moves_colour_level_player": "b" * 64}
    write_json(r(ol.T14_PINS), {"format": "c10-test14 index pins v1", "parts": {"all": pins}})
    write_json(r(ol.T14_SIM), {"format": "c10-test14 sim v1", "evaluation": {"passed": True}})
    files = {n: {"sha256": sha(r(f"data/test/{n}"))} for n in ol.QUESTION_FILES["rank"]}
    overrides = [{"pool": "public" if q.startswith("pub_") else "private", "question_id": q, "base_level": base[q],
                  "new_level": lv, "changed": True, "copy_slots": [1], "source_player": "abc"}
                 for q, lv in changed.items()]
    overrides += [{"pool": "public", "question_id": q, "base_level": base[q], "new_level": base[q], "changed": False,
                   "copy_slots": [2], "source_player": "def"} for q in same_as_base]
    by_pool = {}
    for o in overrides:
        if o["changed"]:
            by_pool[o["pool"]] = by_pool.get(o["pool"], 0) + 1
    rep = {"format": ol.T14_FORMAT, "mode": "real", "method": ol.T14_METHOD, "verdict": "used",
           "preflight": {"manifest_sha256": "m", "t1": {"verdict": "PASS"},
                         "k5_and_sim": {"k5_ok": True, "k5_14": "2026-11-04T11:00:00+08:00"}},
           "base": {"entry_id": base_entry_id, "file": base_rel, "sha256": sha(r(base_rel)), "method": "x"},
           "index_part": "all", "index_pins_check": {"equal": True, "part": "all", "pins": dict(pins)},
           "files": files,
           "questions_overridden": len(overrides), "rows_changed": len(changed), "changed_by_pool": by_pool,
           "overrides": overrides, "conflicts": [],
           "check_submission": {"mode": "official", "errors": []},
           "outputs_sha256": {"submission_rank.csv": sha(f14)}}
    rep.update(report_over or {})
    write_json(r(f"results/test14/rank/{ol.T14_REPORT}"), rep)
    return f14


# ---------------------------------------------------------------- main before PREREG_14 against the new one
class TagVersusNew(t10o.TagVersusNew):
    """A log without 14 entries: the be7dc4d version's log bytes, records, stdout and refusals, with results/test09,
    test10 and test14 trees on disk."""

    @classmethod
    def setUpClass(cls):
        cls.tag = t7o.load_version(BASE_REF)
        assert hasattr(cls.tag, "check_reports_t10") and not hasattr(cls.tag, "check_reports_t14"), "not the base"

    def build(self, kind, with_t06):
        out = super().build(kind, with_t06)
        write_t14_tree(out[0], "results/test02/rank/submission_rank.csv", 1)
        return out


class _Flow14:
    def setUp(self):
        self.tag = t7o.load_version(BASE_REF)

    def run_with(self, mod):
        real = t9o.write_t09_tree

        def both(repo, rows, *a, **k):
            out = real(repo, rows, *a, **k)
            write_t14_tree(repo, "results/test09/rank/submission_rank.csv", 5)
            return out
        with mock.patch.object(t9o, "write_t09_tree", side_effect=both):
            return super().run_with(mod)

    def check_identical(self):
        old = super().check_identical()
        log = json.loads(old[-1][3])
        self.assertFalse(any("test14" in e or "prereg14_7" in e for e in log["uploads"]))
        self.assertNotIn("prereg14_7", json.dumps(log["final_choice"]))
        return old


class TagVersusNew07Flow(_Flow14, t10o.TagVersusNew07Flow):
    pass


class TagVersusNew08Flow(_Flow14, t10o.TagVersusNew08Flow):
    pass


class TagVersusNew09Flow(_Flow14, t9o.FlowBase):
    """R20 -> 04 -> 07 -> 08 -> 09 uploaded, public scores, final 08 refused, final 09, a refused 08 prepare, status."""

    def steps(self, step):
        step("prepare R20", self.prepare)
        step("uploaded R20", self.run_ol, "uploaded", "--id", "1", "--ok-at", "2026-11-05T14:10", "--uploaded-at",
             "2026-11-05T14:15")
        step("prepare 04", self.prepare04)
        step("uploaded 04", self.uploaded, 2, "2026-11-05T15:10")
        step("prepare 07", self.prepare07)
        step("uploaded 07", self.uploaded, 3, "2026-11-06T13:00")
        step("prepare 08", self.prepare08)
        step("uploaded 08", self.uploaded, 4, "2026-11-07T13:00")
        step("public 08", self.run_ol, "public", "--id", "4", "--score", "0.70")
        step("prepare 09", lambda: self.prepare("rank", self.r("results/test09/rank/submission_rank.csv"),
                                                "prereg09 Q09 (PREREG_09 5.3)"))
        step("uploaded 09", self.uploaded, 5, "2026-11-08T13:00")
        step("public 09", self.run_ol, "public", "--id", "5", "--score", "0.70")
        step("final 08 refused", self.run_ol, "final", "--comp", "rank", "--id", "4", "--basis", "x")
        step("final 09", self.run_ol, "final", "--comp", "rank", "--id", "5", "--basis", "PREREG_09 5.3")
        step("prepare 08 again refused", self.prepare08)
        step("status", self.run_ol, "status", "--now", "2026-11-10T00:00")

    def check_identical(self):
        """t09's comparison without its 'no 09 entry' assertion (this flow holds one), plus the 'no 14' one."""
        old, new = self.run_with(self.tag), self.run_with(ol)
        self.assertEqual([s[0] for s in old], [s[0] for s in new])
        for (name, o_out, o_res, o_log), (_n, n_out, n_res, n_log) in zip(old, new):
            with self.subTest(step=name):
                self.assertEqual(n_log, o_log)
                self.assertEqual(n_out, o_out)
                self.assertEqual(n_res, o_res)
        log = json.loads(old[-1][3])
        self.assertFalse(any("test14" in e or "prereg14_7" in e for e in log["uploads"]))
        self.assertNotIn("prereg14_7", json.dumps(log["final_choice"]))
        return old

    def test_09_flow_identical(self):
        old = self.check_identical()
        log = json.loads(old[-1][3])
        self.assertTrue(any("test09" in e for e in log["uploads"]), "the flow must hold a 09 entry")
        self.assertEqual(json.loads(old[13][2])["prereg09_5_3"]["chosen"], "09")
        self.assertIn("refused", old[12][2])


# ---------------------------------------------------------------- the 14 path
class T14Base(t9o.T09Base):
    def setUp(self):
        super().setUp()
        self.p14_head = PREREG14_HEAD
        prev = ol.git_head_text                       # the 04-09 mock of T09Base
        self.patches14 = [mock.patch.object(ol, "git_head_text", side_effect=lambda rel: self.p14_head
                                            if rel == ol.P14_DOC else prev(rel))]
        for pt in self.patches14:
            pt.start()
        self.rev_added["| rev 2 |"] = ("2" * 40, "2026-10-20T10:00:00+08:00")     # PREREG_14 rev 2, before K5-14

    def tearDown(self):
        for pt in reversed(self.patches14):
            pt.stop()
        super().tearDown()

    def ready14(self, final=True, **kw):
        """R20 #1, 04 #2, 07 #3, 08 #4, 09 #5 uploaded, final 09, the 14 tree on the 09 file."""
        self.ready9()
        e9, _ = self.up_09()
        if final:
            self.run_ol("final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "PREREG_09 5.3")
        self.f14 = write_t14_tree(self.repo, "results/test09/rank/submission_rank.csv", e9["id"], **kw)
        return e9

    def prepare14(self, *extra):
        return self.prepare("rank", self.f14, "prereg14 exact-copy layer (PREREG_14 7)", *extra)

    def up_14(self, at="2026-11-21T10:00"):
        e14, out = self.prepare14()
        self.uploaded(e14["id"], at)
        return e14, out

    def edit14(self, fn):
        p = self.r(f"results/test14/rank/{ol.T14_REPORT}")
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        fn(rep)
        write_json(p, rep)

    def rev3_14(self, at="2026-11-22T10:00:00+08:00"):
        self.p14_head = PREREG14_HEAD + "| rev 3 | `abc` | bug |\n"
        self.rev_added["| rev 3 |"] = ("e" * 40, at)


class T14Prepare(T14Base):
    def test_prepare_14(self):
        e9 = self.ready14()
        e14, out = self.prepare14()
        self.assertEqual((e14["id"], e14["file"]), (6, "results/test14/rank/submission_rank.csv"))
        self.assertEqual(e14["method"], ol.T14_METHOD)
        t14 = e14["test14"]
        self.assertEqual((t14["base_entry_id"], t14["base_file"]), (e9["id"], "results/test09/rank/submission_rank.csv"))
        self.assertEqual(t14["rows_changed"], 2)
        self.assertEqual(t14["changed_by_pool"], {"public": 1, "private": 1})
        self.assertEqual(t14["base_rules"], {"prereg09_5_3": "09"})
        self.assertEqual(t14["k6_14"]["deadline"], "2026-11-23T18:00:00+08:00")
        for k in ("prereg04_5_3", "prereg07_5_4", "prereg08_5_4", "prereg09_5_3", "test09"):
            self.assertNotIn(k, e14)
        self.assertIn("這一份是 14", out)
        self.assertIn("public 1、private 1", out)

    def test_before_final_refused(self):
        self.ready14(final=False)
        msg = self.refused(self.prepare14)
        self.assertIn("is not final_choice.rank", msg)

    def test_final_on_another_entry_refused(self):
        self.ready14()
        self.run_ol("final", "--comp", "rank", "--id", "4", "--basis", "x", "--not-09-reason", "PREREG_09 rev 5: x") \
            if False else None
        log = self.log_obj()
        log["final_choice"]["rank"]["id"] = 4
        write_json(self.log, log)
        msg = self.refused(self.prepare14)
        self.assertIn("is not final_choice.rank", msg)

    def test_base_no_longer_designated(self):
        self.ready14()
        with open(self.r("results/test09/gpu_seam.json"), "ab") as f:
            f.write(b" ")
        msg = self.refused(self.prepare14)
        self.assertIn("no longer the file the competition-2 rules designate", msg)

    def test_report_conditions(self):
        self.ready14()
        cases = [("mode", "dry-run", "mode 'dry-run'"), ("verdict", "not_used", "verdict 'not_used'"),
                 ("index_pins_check", {"equal": False, "part": "all"}, "index pins"),
                 ("index_pins_check", {"equal": True, "part": "fit"}, "index pins"),
                 ("rows_changed", 3, "rows_changed 3"),
                 ("check_submission", {"mode": "official", "errors": ["x"]}, "check_submission"),
                 ("preflight", {"t1": {"verdict": "FAIL_OPENING"}}, "T1 'FAIL_OPENING'")]
        for key, val, want in cases:
            with self.subTest(key=key, val=val):
                snap = self.r(f"results/test14/rank/{ol.T14_REPORT}")
                with open(snap, "rb") as f:
                    keep = f.read()
                self.edit14(lambda rep: rep.update({key: val}))
                msg = self.refused(self.prepare14)
                self.assertIn(want, msg)
                with open(snap, "wb") as f:
                    f.write(keep)

    def test_file_differs_in_an_unlisted_row(self):
        self.ready14()
        self.f14 = write_t14_tree(self.repo, "results/test09/rank/submission_rank.csv", 5,
                                  changed=dict(CHANGED, pub_q_0003="5D"))
        self.edit14(lambda rep: rep.update(rows_changed=3, overrides=[o for o in rep["overrides"]
                                                                       if o["question_id"] != "pub_q_0003"]))
        msg = self.refused(self.prepare14)
        self.assertIn("rows_changed 3", msg)
        self.edit14(lambda rep: rep.update(rows_changed=2))
        msg = self.refused(self.prepare14)
        self.assertIn("differs from its base in other rows", msg)

    def test_base_bytes_changed(self):
        self.ready14()
        self.edit14(lambda rep: rep["base"].update(sha256="0" * 64))
        msg = self.refused(self.prepare14)
        self.assertIn("no longer the base bytes", msg)

    def test_k6_14(self):
        self.ready14()
        self.commit = ("f" * 40, "2026-11-23T18:00:01+08:00")
        msg = self.refused(self.prepare14)
        self.assertIn("K6-14", msg)

    def test_method(self):
        self.ready14()
        msg = self.refused(self.prepare14, "--method", "prereg09 Q09")
        self.assertIn("does not name prereg14", msg)

    def test_other_reasons_on_a_14_file_refused(self):
        self.ready14()
        for flag in ("--not-04-reason", "--not-07-reason", "--not-08-reason", "--not-09-reason", "--not-14-reason"):
            with self.subTest(flag=flag):
                msg = self.refused(self.prepare14, flag, "PREREG_09 rev 5: x")
                self.assertIn(flag, msg)

    def test_prepare_base_again_after_14_needs_reason(self):
        self.ready14()
        self.up_14()
        msg = self.refused(self.prepare09)                      # the base itself, while its 14 holds
        self.assertIn("--not-14-reason", msg)
        self.rev3_14()
        e, out = self.prepare09("--not-14-reason", "PREREG_14 rev 3: bug")
        self.assertEqual(e["prereg14_7"]["revision"]["rev"], 3)
        self.assertIn("這一份不是 14", out)

    def test_prepare_08_after_14_needs_no_14_reason(self):
        self.ready14()
        self.up_14()
        self.f08 = t9o.t8o.write_t08_tree(self.repo, self.rank_rows)
        self.rev5_09()
        e, _ = self.prepare08("--not-09-reason", "PREREG_09 rev 5: bug")    # another file: the 09 rule decides
        self.assertNotIn("revision", e.get("prereg14_7") or {})

    def test_k5_rev2_row_late(self):
        self.ready14()
        self.rev_added["| rev 2 |"] = ("2" * 40, "2026-11-04T11:00:01+08:00")
        msg = self.refused(self.prepare14)
        self.assertIn("K5-14", msg)
        self.rev_added["| rev 2 |"] = ("2" * 40, "2026-10-20T10:00:00+08:00")
        self.edit14(lambda rep: rep["preflight"].update(k5_and_sim={"k5_ok": False}))
        msg = self.refused(self.prepare14)
        self.assertIn("K5-14 was not confirmed", msg)

    def test_sim_not_passed(self):
        self.ready14()
        write_json(self.r(ol.T14_SIM), {"evaluation": {"passed": False}})
        msg = self.refused(self.prepare14)
        self.assertIn("simulation did not pass", msg)

    def test_report_bound_to_pins_tests_and_base_levels(self):
        self.ready14()
        self.edit14(lambda rep: rep["index_pins_check"]["pins"].update(rows=999999))
        self.assertIn("index pins ['rows']", self.refused(self.prepare14))
        self.edit14(lambda rep: rep["index_pins_check"]["pins"].update(rows=1000000))
        self.edit14(lambda rep: rep["files"][ol.QUESTION_FILES["rank"][0]].update(sha256="0" * 64))
        self.assertIn("is not the file the 14 report read", self.refused(self.prepare14))
        self.f14 = write_t14_tree(self.repo, "results/test09/rank/submission_rank.csv", 5)
        self.edit14(lambda rep: rep["overrides"][0].update(base_level="6D"))
        self.assertIn("is not the base file's level", self.refused(self.prepare14))


class T14Final(T14Base):
    def test_final_14_and_status(self):
        e9 = self.ready14()
        e14, _ = self.up_14()
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e14["id"]), "--basis", "PREREG_14 7")
        self.assertEqual(fc["id"], e14["id"])
        self.assertEqual(fc["prereg14_7"]["chosen"], "14")
        self.assertEqual(fc["prereg14_7"]["base_entry_id"], e9["id"])
        self.assertEqual(fc["prereg09_5_3"]["chosen"], "09")          # the 09 rule judged the base
        self.assertIn("PREREG_14 7：最終選擇是 14", out)
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        for k in ("prereg04", "prereg07", "prereg08", "prereg09", "prereg14"):
            self.assertFalse(st["rank"][k]["violation"], k)
        self.assertTrue(st["rank"]["prereg14"]["final_is_14"])
        self.assertTrue(st["rank"]["last_is_final"])
        self.assertNotIn("違反", out)
        self.assertNotIn("最終選擇要設成", out)
        self.assertIn("K6-14", out)
        self.assertIn("最後一次上傳就是它", out)

    def test_final_14_before_upload_warns(self):
        self.ready14()
        e14, _ = self.prepare14()
        fc, out = self.run_ol("final", "--comp", "rank", "--id", str(e14["id"]), "--basis", "PREREG_14 7")
        self.assertIn("還沒上傳", out)
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertFalse(st["rank"]["last_is_final"])
        self.assertIn("最後一次上傳不是它", out)
        self.assertNotIn("違反", out)

    def test_final_09_after_14_needs_reason(self):
        e9 = self.ready14()
        e14, _ = self.up_14()
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "x")
        self.assertIn("needs --not-14-reason", msg)
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "x",
                           "--not-14-reason", "PREREG_14 rev 2: x")
        self.assertIn("rev 3 or later", msg)
        self.rev3_14(at="2026-11-04T10:00:00+08:00")       # before the 14 entry was prepared (mocked now 11-05)
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "x",
                           "--not-14-reason", "PREREG_14 rev 3: bug in t14_copy")
        self.assertIn("not after the records", msg)
        self.rev3_14()
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", str(e9["id"]), "--basis", "x",
                            "--not-14-reason", "PREREG_14 rev 3: bug in t14_copy")
        self.assertEqual(fc["prereg14_7"]["chosen"], "not 14")
        st, out = self.run_ol("status", "--now", "2026-11-23T00:00")
        self.assertFalse(st["rank"]["prereg14"]["violation"])

    def test_final_moved_to_08_sets_14_aside(self):
        """PREREG_14 7: when the 04-09 rules move the final to another file, the 14 on the old base is moot (no
        --not-14-reason needed, no 14 violation)."""
        self.ready14()
        e14, _ = self.up_14()
        self.run_ol("final", "--comp", "rank", "--id", str(e14["id"]), "--basis", "PREREG_14 7")
        self.rev5_09()
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", "4", "--basis", "x",
                            "--not-09-reason", "PREREG_09 rev 5: bug in t09_rank apply")
        self.assertEqual(fc["prereg09_5_3"]["chosen"], "08")
        self.assertNotIn("revision", fc.get("prereg14_7") or {})
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertFalse(st["rank"]["prereg14"]["violation"])
        self.assertEqual(st["rank"]["prereg14"]["eligible"], [])
        self.assertIn("final choice moved to #4", out)
        for k in ("prereg08", "prereg09"):
            self.assertFalse(st["rank"][k]["violation"], k)

    def test_14_on_08_base(self):
        """A 14 whose base is 08 (no 09 uploaded)."""
        self.ready9()                                            # R20, 04, 07, 08 uploaded
        self.run_ol("final", "--comp", "rank", "--id", "4", "--basis", "PREREG_08 5.4")
        self.f14 = write_t14_tree(self.repo, "results/test08/rank/submission_rank.csv", 4)
        e14, _ = self.up_14()
        self.assertEqual(e14["test14"]["base_rules"].get("prereg08_5_4"), "08")
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", str(e14["id"]), "--basis", "PREREG_14 7")
        self.assertEqual(fc["prereg08_5_4"]["chosen"], "08")
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        for k in ("prereg04", "prereg07", "prereg08", "prereg14"):
            self.assertFalse(st["rank"][k]["violation"], k)
        self.assertNotIn("違反", out)

    def test_14_on_a_base_chosen_by_revision(self):
        """09 uploaded but set aside by PREREG_09 rev 5: final 08; the 14 on 08 re-checks the base with the recorded
        --not-09-reason, also at final without the flag."""
        self.ready9()
        self.up_09()
        self.rev5_09()
        self.run_ol("final", "--comp", "rank", "--id", "4", "--basis", "x", "--not-09-reason", "PREREG_09 rev 5: bug")
        self.f14 = write_t14_tree(self.repo, "results/test08/rank/submission_rank.csv", 4)
        e14, _ = self.up_14()
        self.assertEqual(e14["test14"]["base_rule_reasons"]["not_09_reason"], "PREREG_09 rev 5: bug")
        fc, _ = self.run_ol("final", "--comp", "rank", "--id", str(e14["id"]), "--basis", "PREREG_14 7")
        self.assertEqual(fc["prereg09_5_3"]["chosen"], "08")
        self.assertEqual(fc["prereg14_7"]["chosen"], "14")
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        for k in ("prereg08", "prereg09", "prereg14"):
            self.assertFalse(st["rank"][k]["violation"], k)

    def test_status_violation_without_final_14(self):
        self.ready14()
        e14, _ = self.up_14()
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertTrue(st["rank"]["prereg14"]["violation"])           # final is still 09
        self.assertIn("違反 PREREG_14 7", out)

    def test_14_no_longer_eligible(self):
        e9 = self.ready14()
        e14, _ = self.up_14()
        self.run_ol("final", "--comp", "rank", "--id", str(e14["id"]), "--basis", "PREREG_14 7")
        with open(self.r("results/test09/gpu_seam.json"), "ab") as f:
            f.write(b" ")
        msg = self.refused(self.run_ol, "final", "--comp", "rank", "--id", str(e14["id"]), "--basis", "x")
        self.assertIn("does not meet PREREG_09 5.3 now", msg)
        st, out = self.run_ol("status", "--now", "2026-11-22T00:00")
        self.assertTrue(st["rank"]["prereg14"]["redo_base"])
        self.assertIn("14 不再符合", out)
        with open(self.r("results/test14/rank/submission_rank.csv"), "ab") as f:
            f.write(b"\n")
        ok, why = ol.t14_entry_eligible(ol.entry(self.log_obj(), e14["id"]), self.log_obj())
        self.assertFalse(ok)
        self.assertTrue(any("changed since prepare" in w for w in why))
        self.assertTrue(e9)


class T14Public(T14Base):
    def test_same_file_warnings(self):
        e9 = self.ready14()
        self.run_ol("public", "--id", str(e9["id"]), "--score", "0.70")
        e14, _ = self.up_14()
        _e, out = self.run_ol("public", "--id", str(e14["id"]), "--score", "0.70")
        self.assertIn("WARNING (PREREG_14)", out)
        for w in ("WARNING (PREREG_09 5.3)", "WARNING (PREREG_08 5.4)", "WARNING (PREREG_07 5.4)",
                  "WARNING (PREREG_04 rev 3)"):
            self.assertNotIn(w, out)

    def test_no_public_change_but_other_score(self):
        e9 = self.ready14(changed={"priv_q_0002": "6D"})
        self.run_ol("public", "--id", str(e9["id"]), "--score", "0.70")
        e14, _ = self.up_14()
        _e, out = self.run_ol("public", "--id", str(e14["id"]), "--score", "0.71")
        self.assertIn("沒有改任何 public 題", out)


class ToolReportAccepted(unittest.TestCase):
    """The report tools/t14_copy.py apply writes (dry-run, then marked real as a real run would be) passes
    official_log's t14_report_problems: the two files agree on the schema."""

    def test_schema(self):
        import test_t14_copy as tt
        import t14_copy as tc
        case = tt.EndToEnd()
        with tempfile.TemporaryDirectory(prefix="c10_t14_schema_") as root:
            train, bpath = case.build(root, {("public", 4): tt.G[1], ("private", 9): tt.G[5]})
            self.assertEqual(case.run_tool("apply", root, train), 0)
            rp = os.path.join(root, "results", "test14", "rank", "t14_copy_report.json")
            rep = tc.read_json(rp)
            rep["mode"] = "real"
            rep["index_pins_check"] = {"equal": True, "part": "all"}
            rep["preflight"] = {"t1": {"verdict": "PASS"}}
            f14 = os.path.join(root, "results", "test14", "rank", "submission_rank.csv")
            with mock.patch.object(ol, "REPO", root):
                probs = ol.t14_report_problems(rep, "r", f14, ol.sha256_file(f14))
            self.assertEqual(probs, [])
            rep["overrides"][0]["changed"] = not rep["overrides"][0]["changed"]
            with mock.patch.object(ol, "REPO", root):
                probs = ol.t14_report_problems(rep, "r", f14, ol.sha256_file(f14))
            self.assertTrue(probs)


if __name__ == "__main__":
    unittest.main()
