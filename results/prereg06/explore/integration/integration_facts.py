"""prereg06 explore / integration: the small facts behind REPORT.md. Read-only, CPU, stdlib only (plus the repo's own
tools/official_log.py, imported to show how it routes a file it does not know).

Never opens data/local (validation questions / labels); never runs tools/score.py; writes only
integration_facts.json next to this file.

1. Scoring budget used so far, counted from results/score_ledger.csv (only the note / command columns are read).
2. Weekdays and gaps of the dates used in the proposed timeline.
3. Elapsed hours of earlier cycles from `git log` commit times (read-only git).
4. How tools/official_log.py treats a competition-2 / competition-1 file outside results/test04/rank
   (is_t04_file, method_for): the label it would write today.
5. Name checks: candidate 06 file paths against the hygiene rules of tests/test_p04_hygiene.py,
   tests/test_t02_hygiene.py and tests/test_t04_hygiene.py (the predicates are re-implemented here from those files;
   nothing is created in the repo).
6. The 04 test-day path files that no fingerprint protects today (git ls-files).
Run from the repo root:  .venv/Scripts/python.exe results/prereg06/explore/integration/integration_facts.py
"""
import csv
import datetime as dt
import fnmatch
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
OUT = os.path.join(HERE, "integration_facts.json")
FORBIDDEN = os.path.normcase(os.path.join(REPO, "data", "local"))


def guard(path):
    if os.path.normcase(os.path.abspath(path)).startswith(FORBIDDEN):
        raise SystemExit("refusing to touch data/local")
    return path


def git(*args):
    r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, timeout=120)
    return r.returncode, r.stdout.decode("utf-8", "replace")


# ---------------------------------------------------------------- 1. scoring budget from the ledger
ledger_path = guard(os.path.join(REPO, "results", "score_ledger.csv"))
with open(ledger_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
by_note = {}
for r in rows:
    by_note.setdefault(r["note"], {"rows": 0, "command": r["command"], "utc_time": r["utc_time"],
                                   "git_head": r["git_head"]})
    by_note[r["note"]]["rows"] += 1
method_scorings = {
    k: v for k, v in by_note.items()
    if k.startswith("prereg0") and v["command"] in ("compare-rank", "compare-player", "player")
}
# PREREG_02 section 3: 'new methods' per competition; prereg01 is the baseline and does not count
counted = {"rank": sorted(k for k, v in method_scorings.items()
                          if v["command"] == "compare-rank" and not k.startswith("prereg01")),
           "player": sorted(k for k, v in method_scorings.items()
                            if v["command"] in ("compare-player",) and not k.startswith("prereg01"))}
budget = {"season_max_new_methods_per_competition": 4,
          "source": "docs/PREREG_02_OPENING_FUSION.md section 3 (line 160)",
          "used": {c: len(v) for c, v in counted.items()}, "used_notes": counted,
          "left": {c: 4 - len(v) for c, v in counted.items()},
          "ledger_data_rows": len(rows), "rows_per_compare_command": 2,
          "ledger_rows_pinned_by_gate": 25,
          "pinned_by": "tests/test_t04_hygiene.py LEDGER_ROWS_AT_BASE (runs with C10_T04_GATE=1 or C10_K5=1)"}

# ---------------------------------------------------------------- 2. dates
WD = "一二三四五六日"
dates = ["2026-10-02", "2026-10-09", "2026-10-12", "2026-10-15", "2026-10-16", "2026-10-18", "2026-10-19",
         "2026-10-21", "2026-10-23", "2026-10-27", "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02",
         "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-07", "2026-11-08", "2026-11-10", "2026-11-15",
         "2026-11-16", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21", "2026-11-22", "2026-11-23",
         "2026-11-24", "2026-11-25"]
weekdays = {d: WD[dt.date.fromisoformat(d).weekday()] for d in dates}
D = dt.date.fromisoformat
gaps_days = {
    "10/02 -> 11/03 (days to the K5-type merge deadline)": (D("2026-11-03") - D("2026-10-02")).days,
    "10/02 -> 11/04 (days to the test set)": (D("2026-11-04") - D("2026-10-02")).days,
    "10/27 -> 11/03 (latest recommended H6R -> merge)": (D("2026-11-03") - D("2026-10-27")).days,
    "11/16 -> 11/23 (K6 -> last-upload target)": (D("2026-11-23") - D("2026-11-16")).days,
    "11/19 -> 11/23 (proposed K6' -> last-upload target)": (D("2026-11-23") - D("2026-11-19")).days,
}

# ---------------------------------------------------------------- 3. elapsed hours from git
commits = ["9d64e54", "74486a2", "9e4f8da", "39ebccb", "e55b019", "1218ba8", "cdec913", "82027fa", "e8b24f2",
           "608d702", "6ec5e68", "9f63acd", "62ca557", "62794a3", "5bc8774", "f4853e5", "73138bb"]
ctime = {}
for c in commits:
    rc, out = git("show", "-s", "--format=%cI %s", c)
    if rc == 0:
        t, subj = out.strip().split(" ", 1)
        ctime[c] = {"time": t, "subject": subj[:100]}
T = lambda c: dt.datetime.fromisoformat(ctime[c]["time"])  # noqa: E731
spans = {
    "test02 kit 9d64e54 -> real-mode fixes 74486a2": ("9d64e54", "74486a2"),
    "prereg03 rev1 9e4f8da -> implementation 39ebccb": ("9e4f8da", "39ebccb"),
    "prereg03 implementation 39ebccb -> Colab results unpacked e55b019 (user's Colab run incl. waiting)": (
        "39ebccb", "e55b019"),
    "prereg03 unpacked e55b019 -> tune/frozen_config 1218ba8": ("e55b019", "1218ba8"),
    "prereg04 rev1 cdec913 -> implementation rev2 82027fa": ("cdec913", "82027fa"),
    "prereg04 rev1 cdec913 -> H4R scored e8b24f2": ("cdec913", "e8b24f2"),
    "test04 branch point 6ec5e68 -> design 9f63acd": ("6ec5e68", "9f63acd"),
    "test04 design 9f63acd -> implementation 62ca557": ("9f63acd", "62ca557"),
    "test04 implementation 62ca557 -> merge f4853e5 (K5 gate rounds 1-2)": ("62ca557", "f4853e5"),
    "test04 branch point 6ec5e68 -> merge f4853e5 (whole test-day extension)": ("6ec5e68", "f4853e5"),
}
elapsed = {k: round((T(b) - T(a)).total_seconds() / 3600, 2) for k, (a, b) in spans.items()
           if a in ctime and b in ctime}

# ---------------------------------------------------------------- 4. official_log routing today
sys.path.insert(0, os.path.join(REPO, "tools"))
import official_log as ol  # noqa: E402

probe = {}
for comp, rel in (("rank", "results/test06/rank/submission_rank.csv"),
                  ("rank", "results/test04/rank/submission_rank.csv"),
                  ("rank", "results/test02/rank/submission_rank.csv"),
                  ("player", "results/test06/player/submission_player.csv")):
    p = os.path.join(REPO, *rel.split("/"))
    probe[f"{comp}:{rel}"] = {
        "is_t04_file": ol.is_t04_file(comp, p),
        "report_it_would_require_next_to_the_file": (ol.T04_REPORT if ol.is_t04_file(comp, p)
                                                     else ol.REPORT_OF[comp]),
        "method_label_it_would_write_without_--method": (ol.T04_METHOD if ol.is_t04_file(comp, p)
                                                         else ol.method_for(comp, False, None)),
    }
ol_constants = {"RANK_COMBOS": list(ol.RANK_COMBOS), "T04_DIR": "/".join(ol.T04_DIR), "K6": ol.K6.isoformat(),
                "FINAL_TARGET": ol.FINAL_TARGET.isoformat(), "DEADLINE": ol.DEADLINE.isoformat(),
                "MAX_PER_DAY": ol.MAX_PER_DAY, "DROP_ALERT": ol.DROP_ALERT, "P04_MIN_REV": ol.P04_MIN_REV,
                "BASELINES": ol.BASELINES}

# ---------------------------------------------------------------- 5. name checks against the hygiene tests
with open(os.path.join(REPO, "results", "prereg04", "design", "r20_path_fingerprint.json"), encoding="utf-8") as f:
    FP = json.load(f)
ALLOWED_T02_PREFIX = "results/test02/"
T02_BAD_PREFIXES = ("results/test02/design/", "results/test02/null/", "results/test02/notebook_diff_",
                    "results/test02/rank/fallback_")


def r20_path_new_file_fails(f):
    """tests/test_t04_hygiene.py R20PathNoNewFiles.test_only_test_day_outputs, for one new file f."""
    hit = any(fnmatch.fnmatch(f, p) or (p.endswith("/*") and f.startswith(p[:-1])) for p in FP["patterns"])
    if not hit or f in FP["files"]:
        return False
    return (not f.startswith(ALLOWED_T02_PREFIX)) or f.startswith(T02_BAD_PREFIXES)


def checks(f):
    out = []
    if r20_path_new_file_fails(f):
        out.append("FAIL test_t04_hygiene R20PathNoNewFiles (new file on the R20 test-day path)")
    if f.startswith(("results/prereg01/", "results/prereg02/", "results/prereg03/", "results/prereg04/design/")):
        out.append("FAIL test_p04_hygiene FrozenResults (new file in a frozen result folder)")
    if f.startswith("results/prereg04/"):
        out.append("FAIL test_t04_hygiene FrozenPrereg04 (results/prereg01-04 must equal 6ec5e68, no new file)")
    if f.startswith("results/prereg05/"):
        out.append("not tested, but frozen by CLAUDE.md (results/prereg01-05/**)")
    if fnmatch.fnmatch(f, "tools/p02_*.py"):
        out.append("FAIL test_t04_hygiene FrozenTools (every tools/p02_*.py must be listed in prereg04 frozen_config)")
    if fnmatch.fnmatch(f, "tools/t04_*.py") or fnmatch.fnmatch(f, "tests/test_t04_*.py") \
            or fnmatch.fnmatch(f, "colab/build_test04_*.py") or fnmatch.fnmatch(f, "colab/make_test04_*.py"):
        out.append("caught by the test04 source rules (t04 import rules / label scan): use a t06 / p06 name")
    if f.startswith("results/test02/") and not r20_path_new_file_fails(f):
        out.append("passes the tests, but only test02's own tools write results/test02 (PREREG_04 rev 3: t04 never "
                   "writes there; the same rule should hold for 06)")
    if f.startswith("results/test04/"):
        out.append("not caught by a test, but results/test04 belongs to the 04 test day (official_log treats "
                   "results/test04/rank/ files as 04 files): keep 06 out")
    return out or ["OK (no hygiene test is affected)"]


candidates = [
    "docs/PREREG_06_EXAMPLE.md", "docs/TESTDAY_06.md", "tools/p06_rank_fuse.py", "tools/t06_rank.py",
    "tools/t06_player.py", "tools/t06_fake_tests.py", "colab/build_prereg06_rank_nb.py", "colab/prereg06_rank.ipynb",
    "colab/build_test06_rank_nb.py", "colab/test06_rank.ipynb", "tests/test_p06_hygiene.py", "tests/test_t06_rank.py",
    "results/prereg06/design/p04_path_fingerprint.json", "results/test06/rank/submission_rank.csv",
    "results/test06/player/submission_player.csv",
    # bad ones
    "tools/t02_rank06.py", "colab/test06_from_test02.ipynb", "results/test02/design/p06_notes.md",
    "results/test02/rank06/apply.json", "tools/p02_rank_fuse06.py", "results/prereg04/rank/q06.json",
    "results/prereg04/design/p04_path_fingerprint.json", "results/prereg05/explore/x.md",
    "results/test04/rank06/x.json", "tools/t04_rank06.py",
]
name_checks = {f: checks(f) for f in candidates}

# COLAB_RUN heading rule (test_colab_run_has_the_test04_section): exactly one line starting with this prefix
prefix = "# 測試集 04（test04）"
headings_ok = {h: (not h.startswith(prefix)) for h in ("# 測試集 06（test06）（使用者操作步驟）",
                                                        "# 測試集 04（test04）之後：06")}

# ---------------------------------------------------------------- 6. the 04 test-day path (not fingerprinted today)
rc, out = git("ls-files", "tools/t04_*", "colab/*test04*", "colab/make_test04*", "tests/test_t04_*",
              "docs/TESTDAY_04.md", "results/test04")
p04_path = [x for x in out.splitlines() if x]
rc_t, tags = git("tag", "-l")

# ---------------------------------------------------------------- official test design (docs/DATA.md, read)
design = {"rank_games_per_question_file": 100 * (5 + 10 + 15 + 20),
          "rank_question_files": 2,
          "rank_test_games_total": 2 * 100 * (5 + 10 + 15 + 20),
          "player_candidate_games_per_pool": 400 * 100, "player_pools": 2,
          "player_query_games_per_pool": 100 * (5 + 10 + 15 + 20),
          "source": "docs/DATA.md 官方資料說明 (5/10/15/20 games x 100 questions; 400 candidates x 100 games)"}

out = {"budget": budget, "weekdays": weekdays, "gaps_days": gaps_days, "commit_times": ctime,
       "elapsed_hours": elapsed, "official_log_probe": probe, "official_log_constants": ol_constants,
       "fingerprint": {"n_files": FP["n_files"], "patterns": FP["patterns"]},
       "name_checks": name_checks, "colab_run_heading_ok": headings_ok,
       "p04_test_day_path_files": p04_path, "p04_test_day_path_count": len(p04_path),
       "local_tags": [t for t in tags.splitlines() if t], "official_test_design": design}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
