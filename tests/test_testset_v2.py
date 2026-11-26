"""PREREG_27 (docs/PREREG_27_TESTSET_V2.md): tools/testset_v2.py and the one official_log.py line.

Synthetic files and throw-away git repositories only; nothing under data/ is read."""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import testset_v2 as tv  # noqa: E402

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.autocrlf=false"]


def run_git(repo, *args):
    r = subprocess.run(GIT + ["-C", repo] + list(args), capture_output=True, text=True)
    if r.returncode:
        raise AssertionError(r.stderr)
    return r.stdout.strip()


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())


def entry(i, comp, final=False):
    return {"id": i, "comp": comp, "status": "uploaded", "file": f"results/x{i}.csv", "sha256": f"{i:064d}"}


def make_log(final_player=False):
    log = {"format": "c10 official_scores v1", "teams": {}, "baselines_public": {},
           "final_choice": {"rank": None, "player": None},
           "uploads": [entry(1, "rank"), entry(2, "player"), entry(3, "rank"), entry(4, "player"),
                       entry(5, "player"), entry(10, "rank")]}
    if final_player:
        log["final_choice"]["player"] = {"id": 5}
    return log


class VoidLog(unittest.TestCase):
    def test_k_moves_player_only(self):
        log = make_log(final_player=True)
        t = tv.void_log(log, {"player": "move", "rank": None}, when="T")
        self.assertEqual(t, {"moved": [2, 4, 5], "marked": []})
        self.assertEqual([e["id"] for e in log["uploads"]], [1, 3, 10])
        self.assertEqual([e["id"] for e in log["voided_uploads"]], [2, 4, 5])
        self.assertTrue(all(e["void"]["by"] == "organizer" and e["void"]["voided_at"] == "T"
                            for e in log["voided_uploads"]))
        self.assertTrue(all("void" not in e for e in log["uploads"]))
        self.assertIsNone(log["final_choice"]["player"])
        self.assertEqual(log["voided_final_choice"]["player"], {"id": 5})

    def test_r0_marks_rank(self):
        log = make_log()
        t = tv.void_log(log, {"player": "move", "rank": "mark"}, when="T")
        self.assertEqual(t["marked"], [1, 3, 10])
        self.assertEqual([e["id"] for e in log["uploads"]], [1, 3, 10])
        self.assertTrue(all(e["void"]["registration"] == tv.REGISTRATION for e in log["uploads"]))

    def test_rr_moves_both_keeps_every_field(self):
        log = make_log()
        before = {e["id"]: dict(e) for e in log["uploads"]}
        tv.void_log(log, {"player": "move", "rank": "move"}, when="T")
        self.assertEqual(log["uploads"], [])
        for e in log["voided_uploads"]:
            self.assertEqual({k: v for k, v in e.items() if k != "void"}, before[e["id"]])


class OfficialLogNumbering(unittest.TestCase):
    """The one added line of tools/official_log.py: new ids continue after the voided v1 entries."""

    def lines(self):
        src = open(os.path.join(REPO, "tools", "official_log.py"), encoding="utf-8").read().splitlines()
        i = next(k for k, s in enumerate(src) if s.strip().startswith('eid = max([e["id"] for e in log["uploads"]]'))
        self.assertIn("PREREG_27", src[i + 1])
        return "\n".join(s.strip() for s in src[i:i + 2])

    def eid(self, log):
        ns = {"log": log}
        exec(self.lines(), {}, ns)
        return ns["eid"]

    def test_continues_after_voided(self):
        log = make_log()
        tv.void_log(log, {"player": "move", "rank": "move"}, when="T")
        self.assertEqual(self.eid(log), 11)

    def test_k_branch(self):
        log = make_log()
        tv.void_log(log, {"player": "move", "rank": None}, when="T")
        self.assertEqual(self.eid(log), 11)

    def test_without_voided_unchanged(self):
        self.assertEqual(self.eid(make_log()), 11)
        self.assertEqual(self.eid({"uploads": []}), 1)


class Archive(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        r = self.r = self.td.name
        run_git(r, "init", "-q")
        write(os.path.join(r, ".gitattributes"), "results/** -text\n")
        write(os.path.join(r, ".gitignore"), "results/test02/*.zip\ndata/\n")
        write(os.path.join(r, "results/test02/null/t3_null.json"), "{}\n")
        write(os.path.join(r, "results/test02/rank/fallback_R10.json"), "{}\n")
        run_git(r, "add", "-A")
        run_git(r, "commit", "-q", "-m", "base")
        self.base = run_git(r, "rev-parse", "HEAD")
        files = ["results/test02/player/submission_player.csv", "results/test02/unpack_player.json",
                 "results/test02/checks/player/t1.json", "results/test02/checks/rank/t1.json",
                 "results/test02/checks/built/rank/t1.json", "results/test02/checks/bundle_build.json",
                 "results/test02/rank/submission_rank.csv", "results/test06/player/off/submission_player.csv",
                 "results/test21/not_used.json", "results/test25/rank/submission_rank.csv"]
        for f in files:
            write(os.path.join(r, f), f"{f}\r\n")
        self.log_bytes = (json.dumps(make_log(), indent=1) + "\n").encode()
        write(os.path.join(r, "results/official_scores.json"), self.log_bytes)
        run_git(r, "add", "-A")
        run_git(r, "commit", "-q", "-m", "v1 outputs")
        run_git(r, "tag", "tv1")

    def tearDown(self):
        self.td.cleanup()

    def arch(self, branch, **kw):
        return tv.archive(branch, repo=self.r, skip_data=True, tag="tv1", base=self.base, log=lambda *a: None,
                          when="T", **kw)

    def test_plan_groups(self):
        p = tv.build_plan("K", repo=self.r, tag="tv1", base=self.base)
        self.assertEqual(p["counts"], {"P": 5, "S": 2, "R": 3})
        self.assertNotIn("results/test02/rank/fallback_R10.json", p["files"])
        self.assertNotIn("results/test02/checks/rank/t1.json", p["files"])
        self.assertEqual(p["void"], {"player": "move", "rank": None})
        self.assertEqual(len(tv.build_plan("RR", repo=self.r, tag="tv1", base=self.base)["files"]), 10)

    def test_archive_k(self):
        rec = self.arch("K")
        self.assertEqual(len(rec["moved"]), 7)
        for m in rec["moved"]:
            self.assertFalse(os.path.exists(os.path.join(self.r, m["from"])))
            got = open(os.path.join(self.r, m["to"]), "rb").read()
            self.assertEqual(got, f"{m['from']}\r\n".encode())       # byte for byte, CRLF kept
        self.assertTrue(os.path.exists(os.path.join(self.r, "results/test02/checks/rank/t1.json")))
        self.assertTrue(os.path.exists(os.path.join(self.r, "results/test25/rank/submission_rank.csv")))
        arch = os.path.join(self.r, tv.ARCH)
        self.assertEqual(open(os.path.join(arch, "official_scores.json"), "rb").read(), self.log_bytes)
        lg = json.load(open(os.path.join(self.r, "results/official_scores.json"), encoding="utf-8"))
        self.assertEqual([e["id"] for e in lg["voided_uploads"]], [2, 4, 5])
        self.assertEqual([e["id"] for e in lg["uploads"]], [1, 3, 10])
        a = json.load(open(os.path.join(arch, "ARCHIVE.json"), encoding="utf-8"))
        self.assertEqual([x["branch"] for x in a["stages"]], ["K"])
        self.assertEqual(rec["staged_unexpected"], [])
        self.assertEqual(a["tag_commit"], run_git(self.r, "rev-parse", "tv1"))
        st = run_git(self.r, "status", "--porcelain")
        self.assertIn("R  results/test02/player/submission_player.csv -> " + tv.ARCH +
                      "/results/test02/player/submission_player.csv", st)
        run_git(self.r, "commit", "-q", "-m", "archive K")
        for m in rec["moved"]:                  # committed blob == the tag's blob (CRLF kept, -text)
            self.assertEqual(run_git(self.r, "rev-parse", "HEAD:" + m["to"]), m["blob"])
            self.assertEqual(run_git(self.r, "rev-parse", "tv1:" + m["from"]), m["blob"])
        with self.assertRaises(SystemExit):     # the same branch again adds nothing
            self.arch("K")
        rec2 = self.arch("RR")                  # competition 2 cleared later: K -> RR moves only the rest
        self.assertEqual(len(rec2["moved"]), 3)
        self.assertEqual(rec2["log"]["moved"], [1, 3, 10])
        self.assertEqual(rec2["staged_unexpected"], [])
        run_git(self.r, "commit", "-q", "-m", "archive RR")
        a = json.load(open(os.path.join(arch, "ARCHIVE.json"), encoding="utf-8"))
        self.assertEqual([x["branch"] for x in a["stages"]], ["K", "RR"])
        self.assertEqual(open(os.path.join(arch, "official_scores.json"), "rb").read(), self.log_bytes)

    def test_r0_then_rr_keeps_first_void(self):
        self.arch("R0")
        run_git(self.r, "commit", "-q", "-m", "R0")
        lg = json.load(open(os.path.join(self.r, "results/official_scores.json"), encoding="utf-8"))
        self.assertEqual([e["void"]["voided_at"] for e in lg["uploads"]], ["T"] * 3)
        tv.archive("RR", repo=self.r, skip_data=True, tag="tv1", base=self.base, log=lambda *a: None, when="T2")
        lg = json.load(open(os.path.join(self.r, "results/official_scores.json"), encoding="utf-8"))
        self.assertEqual(lg["uploads"], [])
        self.assertEqual({e["id"]: e["void"]["voided_at"] for e in lg["voided_uploads"]},
                         {2: "T", 4: "T", 5: "T", 1: "T", 3: "T", 10: "T"})

    def test_data_moves_and_manifest(self):
        write(os.path.join(self.r, "data/test/raw/a.csv"), "x\r\n")
        write(os.path.join(self.r, "data/test06/player/f.npz"), b"\x00\x01")
        write(os.path.join(self.r, "results/test02/test02_player_results.zip"), b"PK")
        write(os.path.join(self.r, "data/test07/hs/c.npz"), b"7")
        rec = tv.archive("K", repo=self.r, skip_data=False, tag="tv1", base=self.base, log=lambda *a: None, when="T")
        ok = {d["from"]: d for d in rec["data_rows"] if d["present"]}
        self.assertEqual(sorted(ok), ["data/test", "data/test06", "results/test02/test02_player_results.zip"])
        self.assertTrue(os.path.isfile(os.path.join(self.r, "data/archive_testset_v1/test/raw/a.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.r, "data/archive_testset_v1/results/test02/"
                                                            "test02_player_results.zip")))
        self.assertFalse(os.path.exists(os.path.join(self.r, "data/test")))
        self.assertTrue(os.path.isdir(os.path.join(self.r, "data/test07")))      # K keeps competition 2's data
        self.assertEqual(ok["data/test"]["files"][0]["sha256"], tv.sha256_bytes(b"x\r\n"))

    def test_failure_is_undone(self):
        write(os.path.join(self.r, "data/test/raw/a.csv"), "x")
        with self.assertRaises(SystemExit) as cm:
            tv.archive("K", repo=self.r, skip_data=False, tag="tv1", base=self.base, log=lambda *a: None,
                       when="T", _fail_after=2)
        self.assertIn("undone", str(cm.exception))
        self.assertTrue(os.path.isfile(os.path.join(self.r, "data/test/raw/a.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.r, "results/test02/player/submission_player.csv")))
        self.assertEqual(run_git(self.r, "diff", "--cached", "--name-only"), "")
        lg = json.load(open(os.path.join(self.r, "results/official_scores.json"), encoding="utf-8"))
        self.assertEqual(lg["uploads"][1]["id"], 2)
        a = json.load(open(os.path.join(self.r, tv.ARCH, "ARCHIVE.json"), encoding="utf-8"))
        self.assertEqual(a["stages"][-1]["status"], "failed")
        with self.assertRaises(SystemExit):     # a failed stage must be cleared by hand first
            self.arch("K")

    def test_archive_rr(self):
        rec = self.arch("RR")
        self.assertEqual(len(rec["moved"]), 10)
        lg = json.load(open(os.path.join(self.r, "results/official_scores.json"), encoding="utf-8"))
        self.assertEqual(lg["uploads"], [])
        self.assertEqual(rec["log"]["moved"], [1, 2, 3, 4, 5, 10])
        for d in ("results/test02/player", "results/test06", "results/test25", "results/test21"):
            self.assertFalse(os.path.exists(os.path.join(self.r, d)), d)      # emptied folders removed
        self.assertTrue(os.path.isdir(os.path.join(self.r, "results/test02/null")))

    def test_refuses_changed_output(self):
        write(os.path.join(self.r, "results/test06/player/off/submission_player.csv"), "changed\n")
        run_git(self.r, "commit", "-q", "-am", "edit after tag")
        with self.assertRaises(SystemExit) as cm:
            self.arch("K")
        self.assertIn("differs from tv1", str(cm.exception))

    def test_unknown_group_stops(self):
        run_git(self.r, "tag", "-d", "tv1")
        write(os.path.join(self.r, "results/test99/x.json"), "{}")
        run_git(self.r, "add", "-A")
        run_git(self.r, "commit", "-q", "-m", "unknown v1 output")
        run_git(self.r, "tag", "tv1")
        with self.assertRaises(SystemExit):
            tv.build_plan("K", repo=self.r, tag="tv1", base=self.base)

    def test_refuses_dirty_results(self):
        write(os.path.join(self.r, "results/test02/player/new.txt"), "x")
        with self.assertRaises(SystemExit):
            self.arch("K")

    def test_refuses_tag_not_ancestor(self):
        run_git(self.r, "checkout", "-q", "--orphan", "other")
        run_git(self.r, "commit", "-q", "-m", "orphan", "--allow-empty")
        with self.assertRaises(SystemExit):
            self.arch("K")

    def test_modified_pre_test_day_file_stops(self):
        run_git(self.r, "tag", "-d", "tv1")
        write(os.path.join(self.r, "results/test02/null/t3_null.json"), "{\"x\": 1}\n")
        run_git(self.r, "commit", "-q", "-am", "modify a pre-11/04 file")
        run_git(self.r, "tag", "tv1")
        with self.assertRaises(SystemExit):
            tv.build_plan("K", repo=self.r, tag="tv1", base=self.base)


def q_rows(prefix, n, k=5, colour="B", tag=""):
    out = []
    for i in range(1, n + 1):
        g = [f"(;SZ[19]HD[0];B[{tag}{i}{j}])" for j in range(k)]
        out.append(",".join([f"{prefix}_q_{i:04d}", str(k)] + g + [""] * (20 - k) + [colour] * k + [""] * (20 - k)))
    return out


HEAD_Q = ",".join(["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, 21)] +
                  [f"color_{i}" for i in range(1, 21)])


def tiny_tree(root, n_q=2, n_p=2, games=100, rank_tag=""):
    files = {}
    for pool, pre in (("public", "pub"), ("private", "priv")):
        files[f"rank_prediction_test_{pool}.csv"] = [HEAD_Q] + q_rows(pre, n_q, tag="r" + rank_tag)
        files[f"player_identification_test_{pool}.csv"] = [HEAD_Q] + q_rows(pre, n_q, tag="p")
        c = ["game_id,player_id,sgf_content,color"]
        g = 0
        for p in range(1, n_p + 1):
            for _ in range(games):
                g += 1
                c.append(f"{pre}_g_{g:07d},{pre}_p_{p:04d},(;SZ[19]HD[0];B[g{g}]),B")
        files[f"player_identification_candidates_{pool}.csv"] = c
    os.makedirs(root, exist_ok=True)
    for n, lines in files.items():
        write(os.path.join(root, n), "\r\n".join(lines) + "\r\n")
    return files


class Probe(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.v1 = os.path.join(self.td.name, "v1", "original")
        tiny_tree(self.v1)
        norm_dir = os.path.join(self.td.name, "v1", "normalised")
        os.makedirs(norm_dir)
        for n in tv.V1_NAMES:
            b = open(os.path.join(self.v1, n), "rb").read().replace(b"HD[0]", b"")
            write(os.path.join(norm_dir, n), b)

    def tearDown(self):
        self.td.cleanup()

    def test_same_bytes(self):
        rep = tv.probe([self.v1], self.v1)
        self.assertTrue(rep["rank_bytes_identical_to_v1"])
        self.assertTrue(rep["player_bytes_identical_to_v1"])
        # the tiny tree is not 400 x 400: every count is a needed change
        whats = {c["what"] for c in rep["needed_changes"]}
        self.assertEqual(whats, {"questions per file", "candidates per pool"})
        self.assertEqual(rep["files"]["rank_prediction_test_public.csv"]["root_props"], {"SZ,HD": 10})

    def test_grown_and_zip(self):
        v2 = os.path.join(self.td.name, "v2")
        tiny_tree(v2, n_q=3, n_p=3, rank_tag="x")
        import zipfile
        z = os.path.join(self.td.name, "dl.zip")
        with zipfile.ZipFile(z, "w") as zf:
            for n in tv.V1_NAMES:
                zf.write(os.path.join(v2, n), "sub/" + n)
        rep = tv.probe([z], self.v1)
        self.assertFalse(rep["rank_bytes_identical_to_v1"])
        self.assertEqual(rep["competition_2_branch_by_bytes"], "RR")
        c = rep["files"]["player_identification_test_public.csv"]["v1_compare"]
        self.assertEqual((c["v1_ids_kept"], c["rows_identical"], c["new_ids"], c["v1_ids_first"]), (2, 2, 1, True))
        r = rep["files"]["rank_prediction_test_public.csv"]["v1_compare"]
        self.assertEqual((r["rows_identical"], r["rows_changed"]), (0, 2))

    def test_new_file_name(self):
        v2 = os.path.join(self.td.name, "v2")
        tiny_tree(v2)
        os.rename(os.path.join(v2, "player_identification_test_public.csv"),
                  os.path.join(v2, "player_identification_test_public_v2.csv"))
        rep = tv.probe([v2], self.v1)
        fn = [c for c in rep["needed_changes"] if c["what"] == "file names"][0]
        self.assertEqual(fn["new"], ["player_identification_test_public_v2.csv"])

    def test_rank_src(self):
        dest = os.path.join(self.td.name, "src")
        rec = tv.rank_src([self.v1], dest, self.v1)
        self.assertEqual(sorted(os.listdir(dest)), sorted(tv.V1_NAMES))
        for n in tv.RANK_NAMES:
            self.assertNotIn(b"HD[0]", open(os.path.join(dest, n), "rb").read())
            self.assertIn("v1_normalised_sha256", rec[n])
        self.assertIn(b"HD[0]", open(os.path.join(dest, "player_identification_test_public.csv"), "rb").read())
        with self.assertRaises(SystemExit):               # dest not empty
            tv.rank_src([self.v1], dest, self.v1)

    def test_rank_src_refuses_changed_rank(self):
        v2 = os.path.join(self.td.name, "v2")
        tiny_tree(v2, rank_tag="x")
        with self.assertRaises(SystemExit) as cm:
            tv.rank_src([v2], os.path.join(self.td.name, "src"), self.v1)
        self.assertIn("branch RR", str(cm.exception))

    def test_rank_missing_is_not_rr(self):
        v2 = os.path.join(self.td.name, "v2")
        tiny_tree(v2)
        for n in tv.RANK_NAMES:
            os.remove(os.path.join(v2, n))
        write(os.path.join(v2, "__MACOSX", "._player_identification_test_public.csv"), b"junk")
        write(os.path.join(v2, "._x.csv"), b"junk")
        rep = tv.probe([v2], self.v1)
        self.assertIsNone(rep["rank_bytes_identical_to_v1"])
        self.assertTrue(rep["competition_2_branch_by_bytes"].startswith("not decidable"))
        self.assertEqual(rep["missing_files"], sorted(tv.RANK_NAMES))
        self.assertFalse([c for c in rep["needed_changes"] if c["what"] == "file names"])

    def test_rank_src_refuses_new_name(self):
        v2 = os.path.join(self.td.name, "v2")
        tiny_tree(v2)
        write(os.path.join(v2, "player_identification_test_new.csv"), "question_id,num_games\r\n")
        with self.assertRaises(SystemExit) as cm:
            tv.rank_src([v2], os.path.join(self.td.name, "src"), self.v1)
        self.assertIn("not the 6 v1 names", str(cm.exception))

    def test_refuses_local(self):
        with self.assertRaises(SystemExit):
            tv.refuse_local(os.path.join(REPO, "data", "local", "x.csv"))
        with self.assertRaises(SystemExit):
            tv.probe([os.path.join(REPO, "data", "local")], self.v1)


class Fake(unittest.TestCase):
    def test_grow_and_same(self):
        with tempfile.TemporaryDirectory() as td:
            v1 = os.path.join(td, "v1")
            tiny_tree(v1, n_q=4, n_p=4)
            rows = []
            for p in range(12):
                for j in range(130):
                    rows.append({"player_id": f"fp{p}", "color": "B", "sgf_content": f"(;SZ[19];B[f{p}x{j}])"})
                for j in range(25):
                    rows.append({"player_id": f"fp{p}", "color": "W", "sgf_content": f"(;SZ[19];B[w{p}x{j}])"})
            root = os.path.join(td, "S")
            rec = tv.fake(root, "grow", extra=2, v1_dir=v1, rows_fit=rows, log=lambda *a: None)
            self.assertEqual(rec["files"]["player_identification_test_public.csv"]["rows"], 6)
            self.assertEqual(rec["files"]["player_identification_candidates_public.csv"]["rows"], 600)
            self.assertEqual(rec["files"]["rank_prediction_test_private.csv"]["rows"], 6)
            rep = tv.probe([os.path.join(root, "download")], v1)
            c = rep["files"]["player_identification_test_public.csv"]["v1_compare"]
            self.assertEqual((c["rows_identical"], c["new_ids"]), (4, 2))
            self.assertEqual(rep["files"]["player_identification_candidates_public.csv"]["games_per_player"],
                             {100: 6})
            last = open(os.path.join(root, "download", "player_identification_test_public.csv"), "rb").read()
            self.assertTrue(last.endswith(b"\r\n") and b"pub_q_0006,10," in last and b"HD[0]" in last)
            root2 = os.path.join(td, "S2")
            tv.fake(root2, "same", swap=2, v1_dir=v1, rows_fit=rows, log=lambda *a: None)
            rep2 = tv.probe([os.path.join(root2, "download")], v1)
            self.assertTrue(rep2["rank_bytes_identical_to_v1"])
            c2 = rep2["files"]["player_identification_candidates_public.csv"]
            self.assertEqual((c2["players"], c2["v1_compare"]["rows_identical"]), (4, 200))
            with self.assertRaises(SystemExit):
                tv.fake(REPO, "grow", v1_dir=v1, rows_fit=rows)


CRLF, LF = chr(13) + chr(10), chr(10)


class Compare(unittest.TestCase):
    def test_compare(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "a.csv"), os.path.join(td, "b.csv")
            write(a, CRLF.join(["question_id,pred_rank", "pub_q_0001,3D", "priv_q_0001,C"]) + CRLF)
            write(b, LF.join(["question_id,pred_rank", "pub_q_0001,3D", "priv_q_0001,B", "pub_q_0401,A"]) + LF)
            r = tv.compare(a, b)
            self.assertEqual((r["common_ids"], r["rows_identical"], r["first_column_differs"], r["v2_rows"]),
                             (2, 1, 1, 3))
            self.assertEqual(r["by_pool"], {"pub_": 0, "priv_": 1})


class Registration(unittest.TestCase):
    def test_prereg_before_tool(self):
        def first(rel):
            r = subprocess.run(["git", "-C", REPO, "log", "--diff-filter=A", "--format=%H %ct", "--", rel],
                               capture_output=True, text=True)
            lines = r.stdout.split()
            return (lines[-2], int(lines[-1])) if lines else (None, None)
        reg, tool = first(tv.REGISTRATION), first("tools/testset_v2.py")
        if tool[0] is None:
            self.skipTest("tools/testset_v2.py not committed yet")
        self.assertIsNotNone(reg[0])
        r = subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", reg[0], tool[0]])
        self.assertEqual(r.returncode, 0)
        self.assertNotEqual(reg[0], tool[0])

    def test_constants(self):
        self.assertEqual(tv.BASE[:7], "f21ff1e")
        self.assertEqual(tv.TAG, "testset-v1-final")
        self.assertFalse(any(p.startswith("data/local") for p in tv.DATA_P + tv.DATA_R))
        self.assertTrue(re.match(r"results/archive_testset_v1$", tv.ARCH))


if __name__ == "__main__":
    unittest.main()
