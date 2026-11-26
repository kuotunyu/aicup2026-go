"""C10 test06, competition 1 (player identification, TEAM_11249): the 06 test-day local tool (06-OFF / 06-ON).

Registration: docs/PREREG_06_PLAYER_STYLE.md sections 2.7, 5.1-5.5 and rev 3 (the test-day procedure). Spec:
results/prereg06/design/spec.md sections 7 and 9.3 (the registration wins over the spec). Claude's steps:
docs/TESTDAY_06.md.

    PY tools/t06_player.py feats  --pool seam_val|public|private [MODE] [DIRS] [--rerun --reason TEXT]
    PY tools/t06_player.py seam                                   [MODE] [DIRS] [--rerun --reason TEXT]
    PY tools/t06_player.py apply  --pool public|private           [MODE] [DIRS] [--rerun --reason TEXT]
    PY tools/t06_player.py submit [--aidea-page-checked TEXT] [--aidea-not-one-to-one]
                                                                  [MODE] [DIRS] [--rerun --reason TEXT]
    PY tools/t06_player.py veto   [--confirmed-upload-check TEXT | --wrong-upload TEXT | --upload-check-inconclusive TEXT]
                                                                  [MODE] [DIRS] [--rerun --reason TEXT]
    PY tools/t06_player.py status                                 [MODE] [DIRS]

MODE  none = real; --dry-run = a fake tree (tools/t02_fake_tests.py, tools/t06_fake_tests.py) with every output
      outside the repository. --now ISO (dry-run, and status) replaces the clock for K6' / K7'.
DIRS  --out-dir O6          06 reports (real: results/test06/player)
      --data-dir A6         06 feature caches (real: data/test06/player; gitignored, SHA-256 in the reports)
      --t02-results-dir D2  test02 notebook root (reads D2/player/<pool>/dumps; real: results/test02)
      --t02-out-dir O2      test02's t02_player.py outputs (real: results/test02/player)
      --tests-dir T         the checked test files + MANIFEST.json (real: data/test; never data/test/raw)
      --checks-dir C        T1 / T2 (reads C/player/{t1,t2,summary}.json; default D2/checks)
      --log L               the upload log (real: results/official_scores.json; dry-run: optional for the gates,
                            required by veto)
      --prereg06-dir P6     the frozen 06 outputs (real: results/prereg06/player; dry-run drills may give a scratch
                            copy, whose pins are then recorded as not checked)
Real mode takes the defaults exactly (other values are refused).

feats   v2 features of one pool (tools/p06_player.MemStore with the frozen LDA_B / LDA_W, exactly as the H6P apply):
        seam_val = data/local/player_val_{candidates,questions}.csv (the only two files read there, under the p06
        guard), public / private = data/test (the files test02 used: SHA-256 equal to test02's apply_report).
        -> A6/<pool>/feats.npz, O6/<pool>/feats_report.json.
seam    the whole H6P apply again on seam_val with test02's seam_val dumps (PREREG_06 5.2; their SHA-256 must equal
        the inputs of test02's seam_val apply_report.json): the selected combination,
        both gates and the detector equal H6P; when the dumps are bitwise equal to prereg02's, top5_off / top5_on are
        byte-identical to player_val_top5.csv / player_val_top5_switch.csv (plus: sr06, the gate values and
        r_forced_cost exactly equal, Q04 reproduced byte for byte, sr06 equal to p06_player.sr06_apply); otherwise
        >= 398/400 identical top-5 rows each. Dump-independent additions (game counts, non-alternating games,
        out-of-vocabulary tokens) always equal H6P. -> O6/seam_val/{t06_seam.json, top5_off.csv, top5_on.csv}.
apply   one test pool, PREREG_06 2.7 (and rev 3): test02's guard level 3, its failed gate (Q00) or an exempt CNN-only
        selection (level 0) -> this pool keeps test02's rows; else sr06 over V2 / V1 / V3 / V0 and R02 (= test02's
        applied combination, p02 functions), PREREG_06 2.4 selection, the 06 gate sr(selected) - sr(R02) >= 0 (else
        test02's rows), CSLSk -> top5_off.csv; the detector with the selected combination's r*_X. top5_on.csv: public
        always when 06 is used there (the forced probe, PREREG_06 5.3), private when level 1, the combination usable
        and the detector open. -> O6/<pool>/{t06_pool.json, top5_off.csv[, top5_on.csv]}. "Both pools on test02's
        rows" (exit 3) counts the other pool's report only when it was made from the present test02 report and seam.
submit  O6/off/submission_player.csv (public rows, then private rows), and O6/on/submission_player.csv when every
        06-ON condition of PREREG_06 5.3 holds (ON_CONDITIONS below); a t06_submit_report.json next to each
        (check_submission --official, the conditions, inputs_bound, top-1 differences against Q04 and off / on).
        06-OFF's inputs_bound has no T2 and no top5_on (inputs of 06-ON only). Refused (2) once the upload log holds
        an entry of this 06 run (any status): a rewritten report would make it not meet PREREG_06 5.4. feats, seam
        and apply --rerun are refused (2) the same way while test02 is unchanged since this run (round-2 review
        10/3; after a test02 re-run 06 is redone from feats, rev 3 item 26).
veto   PREREG_06 5.3 from the upload log, 06-ON paired with the latest 06-OFF of the same run_id uploaded before it:
        the same-file warning pauses (exit 2) until AIdea's upload record is checked: --confirmed-upload-check (no
        wrong file: d decides), --wrong-upload or --upload-check-inconclusive (rev 3: no switch, final 06-OFF, upload
        06-OFF once more); d = public(06-ON) - public(06-OFF); d < 0 vetoes (final 06-OFF, upload 06-OFF once more).
        -> O6/veto.json (outcome: kept / vetoed / wrong_upload / upload_check_inconclusive / k7_passed / paused).
status  read-only: what is done, what is missing, K6' / K7'.

Exit codes: 0 done; 2 refused (fix the precondition, run the same command again; an unexpected internal error also
gives 2 with a traceback: stop and check this tool); 3 the registered "06 is not used" (T1 FAIL_OPENING, K6' passed
without a 06 run committed by K6', H6P not adopted, both pools back on test02's rows, the seam failed with
non-bitwise dumps; final = Q04, PREREG_06 5.4); 4 local drift (the seam failed although its inputs are bitwise those
of H6P, or a dump-independent count differs: stop and check; relaxing anything needs a PREREG_06 revision). A .venv
whose versions differ from frozen_config.json is a refused precondition (2: restore .venv, no upgrade before 11/24);
so are changed frozen files (the 8 pinned H6P outputs, a ledger without exactly the 4 H6P rows: 2, stop and check).
feats / seam / apply / submit check every prerequisite (and the registered exit 3) before their own outputs ("outputs
exist" never hides a 3); after K6' with a 06 run committed by K6' they refuse (2): that run stays valid. No notebook,
no GPU, no tools/score.py; no file whose name holds the validation label word is ever opened; nothing of competition 2
is imported.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p06_style as st  # noqa: E402
import p06_player as p6  # noqa: E402  (installs the data/local guard of the p06 tools)
import p06_sim as sim  # noqa: E402
import p02_player_fuse as pf  # noqa: E402
import p02_player_open as po  # noqa: E402
import check_submission  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse as sp  # noqa: E402

# ------------------------------------------------------------------ paths (module constants; the unit tests point
# them at temporary trees with setattr, the real run never changes them)
RESULTS = os.path.join(REPO, "results")
REAL_O6 = os.path.join(RESULTS, "test06", "player")
REAL_A6 = os.path.join(REPO, "data", "test06", "player")
REAL_D2 = os.path.join(RESULTS, "test02")
REAL_O2 = os.path.join(REAL_D2, "player")
REAL_T = os.path.join(REPO, "data", "test")
REAL_C = os.path.join(REAL_D2, "checks")
REAL_LOG = os.path.join(RESULTS, "official_scores.json")
REAL_P6 = os.path.join(RESULTS, "prereg06", "player")
P06_RESULTS_MD = os.path.join(RESULTS, "prereg06", "RESULTS.md")
P02_PLAYER = os.path.join(RESULTS, "prereg02", "player")
LEDGER = os.path.join(RESULTS, "score_ledger.csv")
LOCAL = os.path.join(REPO, "data", "local")
VAL_C = os.path.join(LOCAL, "player_val_candidates.csv")
VAL_Q = os.path.join(LOCAL, "player_val_questions.csv")
SELF_REL = "tools/t06_player.py"

# ------------------------------------------------------------------ registered values
FROZEN_CONFIG_SHA_LF = "5e5c3b827782678bb3e985cd0caeada61e62951312ee1d6938bf4e331348bfc9"   # = apply_report.json
P06_PINS_LF = {   # results/prereg06/player (csv / json with CRLF folded to LF, npz raw); H6P scored 2026-10-03
    "apply_report.json": "5207662cade5dd795bef97f1bf5b8a07d8cfbebc2d689ce60efe3a4c56078351",
    "frozen_config.json": FROZEN_CONFIG_SHA_LF,
    "lda_B.npz": "ecfee95d0c7d8fdeb499ca9f89a8a6bb840e9a3cf0e999b1baa14ff78d69faa5",
    "lda_W.npz": "e8bc63331a749bce83cc7cab8b7852bec70c6644f38fe89982f0dbbad581292d",
    "player_val_top5.csv": "094223f69bf445a4c450212cc9b4a3eceda9c69e992d1ed7f7c6e02105b818be",
    "player_val_top5_switch.csv": "9dfc8e1b5500d632f4a7f035dd50832723a72556caabb3ba1e57f779a3c7dcb6",
    "score_H6P.json": "e73144af5c3039e5923ed0048252382dab4dfa7f2e54f63b8312cd9717f1569f",
    "score_H6P_switch.json": "073beb887e89e6e30e58ff00bcf1ebbdfb96c591f84005cc13450df16573b04c"}
P02_TOP5_SHA_LF = "69ca74a61ce34f8ffa8949df731df433d474347dce9aabb535156ab8197ead6c"
P02_DUMP_SHA = {"cand_games.npz": "56ed03d12e9a79871a21b5f69dc4bd4bb8c8aa69b1544cb8ec799e80227cfdea",
                "query_games.npz": "665d0888a93f6632eb19edac76abf996d5499cd27c683d9844b97cb274b6f689"}
VENV = {"python": "3.12.13", "numpy": "2.5.3", "scipy": "1.18.1"}      # frozen_config.json env
H6P_LEDGER_PREFIX = "prereg06 H6P"
H6P_LEDGER_ROWS = 4
H6P_RESULTS_MARK = "**採用**"
TW = dt.timezone(dt.timedelta(hours=8))
K6P = dt.datetime(2026, 11, 21, 23, 59, 59, tzinfo=TW)    # PREREG_06 4: a successful official 06 run committed
K7P = dt.datetime(2026, 11, 22, 23, 59, 59, tzinfo=TW)    # PREREG_06 4: 06-ON uploaded, both scores, veto decided
SEAM_MIN_IDENTICAL = 398                                  # PREREG_06 5.2 (= PREREG_02 9.2): >= 398 / 400
GATE06_MIN = 0.0                                          # PREREG_06 2.7: sr(selected) - sr(R02) < 0 -> test02's rows
SCORE_EQUAL_TOL = 1e-9                                    # same-file warning: two public scores "完全相同"
N_OFFICIAL = 400
POOLS = ("seam_val", "public", "private")
TEST_POOLS = ("public", "private")
TEST_FILES = {"public": ("player_identification_test_public.csv", "player_identification_candidates_public.csv"),
              "private": ("player_identification_test_private.csv", "player_identification_candidates_private.csv")}
Q04_FILE_REL = "results/test02/player/submission_player.csv"
SUBMISSION = "submission_player.csv"
REPORT = "t06_submit_report.json"
VARIANTS = ("off", "on")
METHOD = {"off": "prereg06 06-OFF: Q04 + v2 style + CSLS (甲)",
          "on": "prereg06 06-ON: 06-OFF + one-to-one switch (甲)"}
FORMATS = {"feats": "c10-test06 feats v1", "seam": "c10-test06 seam v1", "pool": "c10-test06 pool v1",
           "submit": "c10-test06 submit v1", "veto": "c10-test06 veto v1", "not_used": "c10-test06 not used v1"}
T1_OK = ("PASS", "NORMALIZED")
T2_FORMAT = "c10-test02 t2 v1"
ON_CONDITIONS = ("private_switch_usable_and_detector_open", "h6p_secondary_not_negative", "pools_400_by_400",
                 "t2_c_zero_and_abd_not_failed", "both_pools_06_at_guard_level_1", "aidea_page_not_against_one_to_one",
                 "before_k7")
TOOLS_COMMITTED = ("tools/t06_player.py", "tools/p06_style.py", "tools/p06_sim.py", "tools/p06_player.py",
                   "tools/p02_player_fuse.py", "tools/p02_player_open.py", "tools/p02_moves.py", "tools/c10data.py",
                   "tools/check_submission.py")
P02_DEPS = ("tools/p02_player_fuse.py", "tools/p02_player_open.py", "tools/p02_moves.py", "tools/c10data.py")
_LABEL = "ans" + "wer"

MSG_U5 = "最終照 U5 交 Q04（PREREG_06 5.4；docs/TESTDAY_06.md「06 不用」）"
MSG_REDO = ("test02 的輸入在這一步之後變了（PREREG_06 rev 3：test02 重跑之後 06 從 feats 重做）：照 docs/TESTDAY_06.md 從 "
            "feats 重跑，三步都加 --rerun --reason")


class T06Error(RuntimeError):
    """A stop with its exit code: 2 refused, 3 registered '06 not used', 4 local drift."""

    def __init__(self, message, code=2, reason=None):
        super().__init__(message)
        self.code, self.reason = code, reason


def refuse(msg):
    raise T06Error(msg, 2)


# ================================================================ small helpers
def now_real():
    """The current time (a module function: the unit tests replace it to cross K6' / K7')."""
    return dt.datetime.now(TW)


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_iso(text, what):
    try:
        t = dt.datetime.fromisoformat(str(text).strip())
    except ValueError:
        refuse(f"{what} {text!r} is not an ISO time")
    return t.replace(tzinfo=TW) if t.tzinfo is None else t


def refuse_label(path):
    if path is not None and _LABEL in os.path.abspath(os.fspath(path)).lower():
        refuse(f"refusing {path}: validation label files are never opened by the 06 tools")
    return path


def real_inside(path, root):
    p = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    r = os.path.normcase(os.path.realpath(os.path.abspath(root)))
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def abs_inside(path, root):
    p = os.path.normcase(os.path.abspath(path))
    r = os.path.normcase(os.path.abspath(root))
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def inside(path, root):
    return abs_inside(path, root) or real_inside(path, root)


def check_input(path, allow_local=False):
    """No label file; nothing under data/local (also through a junction or symlink: the realpath is compared too,
    which the p06 guard does not do for the main repository's path) except the two seam_val game files."""
    refuse_label(path)
    if inside(path, LOCAL):
        ok = allow_local and any(os.path.normcase(os.path.realpath(os.path.abspath(path))) ==
                                 os.path.normcase(os.path.realpath(os.path.abspath(p))) for p in (VAL_C, VAL_Q))
        if not ok:
            refuse(f"refusing {path}: data/local is read only for the two seam_val game files")
    return path


def sha_raw(path):
    check_input(path, allow_local=True)
    return st.sha256_file(path)


def sha_lf(path):
    check_input(path, allow_local=True)
    return st.sha256_lf(path)


def sha_bound(path):
    """The inputs_bound rule of tools/official_log.py: .csv / .json with CRLF folded to LF, anything else raw."""
    check_input(path, allow_local=True)
    with open(path, "rb") as f:
        data = f.read()
    if path.lower().endswith((".csv", ".json")):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def bytes_lf(path):
    check_input(path, allow_local=True)
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def read_json(path, what=None):
    check_input(path)
    if not os.path.exists(path):
        refuse(f"missing {shown(path)}" + (f" ({what})" if what else ""))
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except ValueError as e:
        refuse(f"{shown(path)} is not valid JSON ({e})")


def maybe_json(path):
    return read_json(path) if os.path.exists(path) else None


def shown(path):
    return st.rel(path) if st.inside_repo(path) else os.path.abspath(path)


def rel_key(path):
    """Repository-relative key with '/' (real mode), the absolute path outside the repository (dry-run trees)."""
    return st.rel(path)


def git(*args):
    try:
        r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    return r.returncode, (r.stdout or b"").decode("utf-8", "replace").strip()


def committed_clean(path):
    """Tracked by git and unchanged against HEAD (a module function: the unit tests replace it)."""
    rel = st.rel(path)
    if os.path.isabs(rel):
        return False
    rc, _ = git("ls-files", "--error-unmatch", "--", rel)
    if rc != 0:
        return False
    rc, out = git("status", "--porcelain", "--", rel)
    return rc == 0 and not out


def require_committed(ctx, paths, what):
    if not ctx["real"]:
        return
    bad = [shown(p) for p in paths if not committed_clean(p)]
    if bad:
        refuse(f"{what}: not committed and clean: {bad[:6]} (docs/TESTDAY_06.md: commit after every step)")


def git_head():
    rc, out = git("rev-parse", "HEAD")
    return out if rc == 0 else None


def git_commit_time(path):
    """The committer time of the last commit touching path (ISO), None when untracked or outside the repository (a
    module function: the unit tests replace it)."""
    rel = st.rel(path)
    if os.path.isabs(rel):
        return None
    rc, out = git("log", "-1", "--format=%cI", "--", rel)
    return out if rc == 0 and out else None


def env_record():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "executable": sys.executable}


def tool_record():
    return {"git_head": git_head(), "files_sha256_lf": {r: (st.sha256_lf(os.path.join(REPO, *r.split("/")))
                                                            if os.path.exists(os.path.join(REPO, *r.split("/")))
                                                            else None) for r in TOOLS_COMMITTED}}


def jsonable(o):
    return st.json_safe(o)


def guard_write(ctx, path):
    p = os.path.abspath(path)
    refuse_label(p)
    for ro in (os.path.join(RESULTS, "test02"), LOCAL, LEDGER, REAL_LOG) + tuple(
            os.path.join(RESULTS, "prereg0%d" % i) for i in range(1, 10)):
        if inside(p, ro):
            refuse(f"refusing to write {p}: it is inside the read-only {shown(ro)}")
    if not any(abs_inside(p, r) for r in ctx["write_roots"]):
        refuse(f"refusing to write {p}: outside {ctx['write_roots']}")
    return p


def write_json(ctx, path, obj):
    guard_write(ctx, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(obj), f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def write_rows(ctx, path, rows):
    """question_id,top_1..top_5 rows, UTF-8 without BOM, LF (as tools/t02_player.py and p06_player.write_top5)."""
    guard_write(ctx, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id"] + pf.TOP_COLS)
        w.writerows(rows)
    os.replace(tmp, path)
    return path


def read_top5(path):
    check_input(path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != ["question_id"] + pf.TOP_COLS:
        refuse(f"{shown(path)}: header is not question_id,{','.join(pf.TOP_COLS)}")
    return rows[1:]


def guard_existing(ctx, a, paths):
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not a.rerun:
        refuse("outputs exist (%s); use --rerun --reason TEXT to overwrite" % ", ".join(shown(p) for p in existing))
    if a.rerun and not (a.reason or "").strip():
        refuse("--rerun needs --reason TEXT (it is stored in the report)")
    if not a.rerun:
        return None
    return {"reason": a.reason, "previous_sha256": {shown(p): sha_raw(p) for p in existing}}


def refuse_rerun_once_uploaded(ctx, a, pre, cmd):
    """feats / seam / apply --rerun after this 06 run is in the upload log: refused (2), as submit --rerun (PREREG_06
    rev 3 item 18; round-2 review 10/3, minor 7). Rewriting the seam or a pool report gives a new run_id, so the
    uploaded 06-OFF would no longer meet PREREG_06 5.4 (the final choice would fall back to Q04) and a new run would
    get around submit's guard. Allowed when test02 changed since this run (rev 3 item 26: test02 was re-run, 06 is
    redone from feats; the uploaded run already no longer meets 5.4) and when the run is not complete on disk."""
    if not a.rerun:
        return
    seam_p = out_paths(ctx, "seam_val")["seam"]
    pool_ps = {p: out_paths(ctx, p)["pool"] for p in TEST_POOLS}
    if not all(os.path.exists(p) for p in [seam_p] + list(pool_ps.values())):
        return
    run_id = hashlib.sha256("|".join([sha_lf(seam_p)] + [sha_lf(pool_ps[p]) for p in TEST_POOLS])
                            .encode()).hexdigest()
    logged = run_in_log(ctx, run_id)
    if not logged:
        return
    seam = read_json(seam_p)
    stale = [shown(seam_p)] if (seam.get("inputs") or {}).get("test02_seam_local_sha256_lf") != \
        pre["t02"]["seam_local_sha256_lf"] else []
    for p in TEST_POOLS:
        t = read_json(pool_ps[p]).get("test02") or {}
        tp = pre["t02"]["pools"][p]
        if t.get("apply_report_sha256_lf") != sha_lf(tp["apply_report"]) or t.get("top5_sha256") != sha_raw(tp["top5"]):
            stale.append(shown(pool_ps[p]))
    if stale:
        return
    refuse(f"{cmd} --rerun: the upload log already holds 06 entries {logged} of this 06 run (run_id {run_id[:12]}) and "
           f"test02 has not changed since: rewriting its outputs would make them not meet PREREG_06 5.4 (the final "
           f"choice would fall back to Q04). Only a test02 re-run lets 06 be redone (docs/TESTDAY_06.md 'test02 重跑了"
           f"怎麼辦'); otherwise a change needs a PREREG_06 revision (rev 4 on)")


def left(now, until):
    s = (until - now).total_seconds()
    if s <= 0:
        return "已過"
    d, rest = divmod(int(s), 86400)
    return f"{d} 天 {rest // 3600} 小時"


# ================================================================ modes and directories
def resolve(a):
    mode = "dry-run" if a.dry_run else "real"
    given = {"--out-dir": a.out_dir, "--data-dir": a.data_dir, "--t02-results-dir": a.t02_results_dir,
             "--t02-out-dir": a.t02_out_dir, "--tests-dir": a.tests_dir, "--checks-dir": a.checks_dir,
             "--log": a.log, "--prereg06-dir": a.prereg06_dir}
    real = {"--out-dir": REAL_O6, "--data-dir": REAL_A6, "--t02-results-dir": REAL_D2, "--t02-out-dir": REAL_O2,
            "--tests-dir": REAL_T, "--checks-dir": REAL_C, "--log": REAL_LOG, "--prereg06-dir": REAL_P6}
    for v in given.values():
        if v:
            refuse_label(v)
    if mode == "real":
        for k, v in given.items():
            if v and os.path.normcase(os.path.abspath(v)) != os.path.normcase(real[k]):
                refuse(f"real mode: {k} must be {real[k]} (got {v}); other trees need --dry-run")
        if a.now and a.cmd != "status":
            refuse("--now is for --dry-run and status only (real mode uses the clock for K6' / K7')")
        d = {k[2:].replace("-", "_"): v for k, v in real.items()}
    else:
        need = ["--out-dir", "--data-dir"]
        if a.cmd in ("feats", "seam", "apply", "submit"):
            need += ["--t02-results-dir", "--t02-out-dir", "--tests-dir"]
        if a.cmd == "veto":
            need += ["--log"]
        missing = [k for k in need if not given[k]]
        if missing:
            refuse(f"--dry-run needs {', '.join(missing)} (a dry run never falls back to the repository paths)")
        d = {k[2:].replace("-", "_"): (os.path.abspath(v) if v else None) for k, v in given.items()}
        for k in ("out_dir", "data_dir"):
            if st.inside_repo(d[k]):
                refuse(f"--dry-run: --{k.replace('_', '-')} {d[k]} is inside the repository; use a scratch folder")
        if d["checks_dir"] is None and d["t02_results_dir"]:
            d["checks_dir"] = os.path.join(d["t02_results_dir"], "checks")
        if d["prereg06_dir"] is None:
            d["prereg06_dir"] = REAL_P6
    if d.get("tests_dir") and (inside(d["tests_dir"], LOCAL) or
                               os.path.basename(os.path.normpath(d["tests_dir"])).lower() == "raw"):
        refuse(f"--tests-dir {d['tests_dir']}: the checked copies (data/test), never data/local or data/test/raw")
    now = parse_iso(a.now, "--now") if a.now else now_real()
    p6_default = os.path.normcase(os.path.abspath(d["prereg06_dir"])) == os.path.normcase(REAL_P6)
    return {"mode": mode, "real": mode == "real", "O6": d["out_dir"], "A6": d["data_dir"],
            "D2": d["t02_results_dir"], "O2": d["t02_out_dir"], "T": d["tests_dir"], "C": d["checks_dir"],
            "LOG": d["log"], "P6": d["prereg06_dir"], "p6_default": p6_default, "now": now,
            "write_roots": [d["out_dir"], d["data_dir"]]}


def pool_files(ctx, pool):
    """(questions csv, candidates csv, test02 dumps parent) of a pool."""
    if pool == "seam_val":
        q, c = VAL_Q, VAL_C
    else:
        qn, cn = TEST_FILES[pool]
        q, c = os.path.join(ctx["T"], qn), os.path.join(ctx["T"], cn)
    with st.allow_local():
        for p in (q, c):
            check_input(p, allow_local=pool == "seam_val")
            if not os.path.exists(p):
                refuse(f"{pool}: input file missing: {shown(p)}")
    return q, c, os.path.join(ctx["D2"], "player", pool)


def out_paths(ctx, pool):
    return {"dir": os.path.join(ctx["O6"], pool), "feats_report": os.path.join(ctx["O6"], pool, "feats_report.json"),
            "feats": os.path.join(ctx["A6"], pool, "feats.npz"), "pool": os.path.join(ctx["O6"], pool, "t06_pool.json"),
            "seam": os.path.join(ctx["O6"], "seam_val", "t06_seam.json"),
            "off": os.path.join(ctx["O6"], pool, "top5_off.csv"), "on": os.path.join(ctx["O6"], pool, "top5_on.csv")}


# ================================================================ the registered "06 not used"
def not_used(ctx, cmd, reason, message, pool=None, write=True):
    """Exit 3: record the event in O6/not_used.json (commit it; PREREG_06 5.4: final Q04) and stop."""
    if write:
        path = os.path.join(ctx["O6"], "not_used.json")
        rec = maybe_json(path) or {"format": FORMATS["not_used"], "events": []}
        rec["events"].append({"command": cmd, "pool": pool, "reason": reason, "message": message, "mode": ctx["mode"],
                              "created": now_iso(), "now": ctx["now"].isoformat(), "git_head": git_head()})
        write_json(ctx, path, rec)
    raise T06Error(message + "；" + MSG_U5, 3, reason)


# ================================================================ common prerequisites (spec 7.2)
def check_venv():
    got = {k: v for k, v in env_record().items() if k in VENV}
    if got != VENV:
        raise T06Error(f".venv versions {got} differ from the registered {VENV} (frozen_config.json): restore .venv "
                       f"(no upgrade before 11/24) and run the same command again", 2, "venv")
    return got


def git_clean_tracked():
    """Real mode: no tracked file differs from HEAD (spec 7.2 'git 乾淨'; a module function: the tests replace it)."""
    rc, out = git("status", "--porcelain", "--untracked-files=no")
    return rc == 0 and not out, out


def check_tools(ctx, fz):
    """The p06 tools equal frozen_config.json, the p02 dependencies equal its dependency_sha256; real mode: every
    tool this run imports is committed and clean."""
    now = {r: st.sha256_lf(os.path.join(REPO, *r.split("/"))) for r in TOOLS_COMMITTED}
    bad = [r for r, v in (fz.get("tools") or {}).items() if now.get(r) != (v or {}).get("sha256_lf")]
    bad += [r for r in P02_DEPS if (fz.get("dependency_sha256") or {}).get(r) != now.get(r)]
    if bad:
        refuse(f"tools changed since the 06 freeze: {bad} (PREREG_06 5.1: the frozen tools are never edited)")
    require_committed(ctx, [os.path.join(REPO, *r.split("/")) for r in TOOLS_COMMITTED], "the 06 tools")
    return now


def load_frozen(ctx):
    path = os.path.join(ctx["P6"], "frozen_config.json")
    if not os.path.exists(path):
        refuse(f"missing {shown(path)}")
    got = sha_lf(path)
    if ctx["p6_default"] and got != FROZEN_CONFIG_SHA_LF:
        refuse(f"frozen_config.json SHA-256 (LF) {got} != the registered {FROZEN_CONFIG_SHA_LF}")
    require_committed(ctx, [path], "results/prereg06/player/frozen_config.json")
    fz = read_json(path)
    if fz.get("mode") != "real" or fz.get("precommit_preview"):
        refuse("frozen_config.json is not the real freeze")
    models, lda = {}, {}
    for c in "BW":
        f = os.path.join(ctx["P6"], "lda_%s.npz" % c)
        want = ((fz.get("lda") or {}).get(c) or {}).get("sha256")
        if not os.path.exists(f) or sha_raw(f) != want:
            refuse(f"lda_{c}.npz differs from frozen_config.json ({want})")
        models[c] = p6.load_lda(f)
        lda[c] = want
    require_committed(ctx, [os.path.join(ctx["P6"], "lda_%s.npz" % c) for c in "BW"], "the frozen LDA files")
    return fz, models, {"path": shown(path), "sha256_lf": got, "pinned": ctx["p6_default"], "lda_sha256": lda}


def ledger_h6p_rows():
    with open(LEDGER, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return [r for r in rows[1:] if len(r) > 2 and r[2].startswith(H6P_LEDGER_PREFIX)]


def h6p_state(ctx):
    """H6P adopted (PREREG_06 3): the 95 % lower bound of score_H6P.json > 0; real mode also RESULTS.md says adopted
    and the ledger holds the 4 H6P rows. Plus the secondary (one-way) point estimate for 06-ON."""
    pins = {}
    for n, want in P06_PINS_LF.items():
        p = os.path.join(ctx["P6"], n)
        if not os.path.exists(p):
            refuse(f"missing {shown(p)}")
        pins[n] = sha_bound(p) if n.endswith((".csv", ".json")) else sha_raw(p)
        if ctx["p6_default"] and pins[n] != want:
            refuse(f"results/prereg06/player/{n} changed ({pins[n][:12]} != registered {want[:12]}): the H6P outputs "
                   f"are frozen")
    require_committed(ctx, [os.path.join(ctx["P6"], n) for n in P06_PINS_LF], "results/prereg06/player")
    s = read_json(os.path.join(ctx["P6"], "score_H6P.json"))
    sw = read_json(os.path.join(ctx["P6"], "score_H6P_switch.json"))
    lo = ((s.get("B_minus_A") or {}).get("ci95") or [None])[0]
    mean_sw = (sw.get("B_minus_A") or {}).get("mean_diff")
    rec = {"score_H6P": {"ci95_low": lo, "mean_diff": (s.get("B_minus_A") or {}).get("mean_diff")},
           "score_H6P_switch": {"mean_diff": mean_sw}, "pins_checked": ctx["p6_default"], "sha256": pins,
           "adopted": isinstance(lo, (int, float)) and lo > 0,
           "secondary_not_negative": isinstance(mean_sw, (int, float)) and mean_sw >= 0}
    if ctx["real"]:
        require_committed(ctx, [P06_RESULTS_MD, LEDGER], "results/prereg06/RESULTS.md and the ledger")
        with open(P06_RESULTS_MD, encoding="utf-8") as f:
            text = f.read()
        n_rows = len(ledger_h6p_rows())
        rec.update(results_md_says_adopted=H6P_RESULTS_MARK in text, ledger_h6p_rows=n_rows)
        if n_rows != H6P_LEDGER_ROWS:
            refuse(f"results/score_ledger.csv holds {n_rows} '{H6P_LEDGER_PREFIX}' rows, the registered H6P wrote "
                   f"{H6P_LEDGER_ROWS}")
        if not rec["results_md_says_adopted"]:
            rec["adopted"] = False
    return rec


def check_t1(ctx, cmd, pool):
    path = os.path.join(ctx["C"], "player", "t1.json")
    if not os.path.exists(path):
        refuse(f"{shown(path)} missing: test02's T1 comes first (docs/TESTDAY.md)")
    t1 = read_json(path)
    v = t1.get("verdict")
    if ctx["real"]:
        require_committed(ctx, [path], "test02's T1")
        mode = (t1.get("provenance") or {}).get("mode")
        if t1.get("complete") is not True or t1.get("stage") != "raw" or mode != "real":
            refuse(f"{shown(path)}: complete={t1.get('complete')!r}, stage {t1.get('stage')!r}, mode {mode!r}: the "
                   f"full T1 on the raw files comes first (docs/TESTDAY.md)")
    if v == "FAIL_OPENING":
        not_used(ctx, cmd, "t1_fail_opening", "T1 是 FAIL_OPENING：06 整個不用（兩池都不用，PREREG_06 2.7）", pool)
    if v not in T1_OK:
        refuse(f"T1 verdict {v!r}: FAIL_STRUCT stops as test02 does (PREREG_06 2.7)")
    return {"path": shown(path), "sha256_lf": sha_lf(path), "verdict": v}


def check_test02(ctx):
    """test02 competition 1: submit done in this mode with check_submission official passed, the local seam passed,
    both pools' reports and top-5 files consistent; real mode: committed and clean (with the dumps)."""
    O2 = ctx["O2"]
    want_mode = "real" if ctx["real"] else "dry-run"
    sub, rep_p, seam_p = (os.path.join(O2, SUBMISSION), os.path.join(O2, "submit_report.json"),
                          os.path.join(O2, "seam_val", "seam_local.json"))
    for p in (sub, rep_p, seam_p):
        if not os.path.exists(p):
            refuse(f"test02 competition 1 is not done ({shown(p)} missing): docs/TESTDAY.md comes first")
    rep, seam = read_json(rep_p), read_json(seam_p)
    probs = []
    cs = rep.get("check_submission") or {}
    if rep.get("mode") != want_mode or cs.get("official") is not True or cs.get("passed") is not True:
        probs.append(f"submit_report.json mode {rep.get('mode')!r}, check_submission official {cs.get('official')!r} "
                     f"passed {cs.get('passed')!r}")
    if (rep.get("output") or {}).get("sha256") != sha_raw(sub):
        probs.append("submission_player.csv is not the file submit_report.json records")
    if seam.get("passed") is not True or seam.get("mode") != want_mode or seam.get("cnn") is not True:
        probs.append(f"seam_local.json passed {seam.get('passed')!r} mode {seam.get('mode')!r} cnn {seam.get('cnn')!r}")
    pools, files = {}, [sub, rep_p, seam_p]
    for pool in TEST_POOLS:
        d = os.path.join(O2, pool)
        ar_p, ps_p, top = (os.path.join(d, "apply_report.json"), os.path.join(d, "pool_select.json"),
                           os.path.join(d, "top5.csv"))
        for p in (ar_p, ps_p, top):
            if not os.path.exists(p):
                refuse(f"test02 competition 1: {shown(p)} missing")
        ar = read_json(ar_p)
        files += [ar_p, ps_p, top]
        if ar.get("mode") != want_mode or ar.get("cnn") is not True:
            probs.append(f"{pool}: apply_report.json mode {ar.get('mode')!r} cnn {ar.get('cnn')!r}")
        if sha_raw(top) != (ar.get("output") or {}).get("sha256"):
            probs.append(f"{pool}: top5.csv is not the file apply_report.json records")
        if ((rep.get("pools") or {}).get(pool) or {}).get("top5", {}).get("sha256") != sha_raw(top):
            probs.append(f"{pool}: top5.csv is not the file submit_report.json records")
        pools[pool] = {"apply_report": ar_p, "pool_select": ps_p, "top5": top, "report": ar}
    if probs:
        refuse("test02 competition 1 is not usable for 06: " + "; ".join(probs))
    if ctx["real"]:
        dumps = [os.path.join(ctx["D2"], "player", p, "dumps", n) for p in POOLS
                 for n in ("cand_games.npz", "query_games.npz", "_INPUT.json")]
        require_committed(ctx, files + dumps, "test02 competition 1 (reports and dumps)")
    return {"submission": sub, "submit_report": rep_p, "seam_local": seam_p, "pools": pools,
            "submission_sha256": sha_raw(sub), "submit_report_sha256_lf": sha_lf(rep_p),
            "seam_local_sha256_lf": sha_lf(seam_p)}


def q04_first_upload(log):
    ups = [e for e in log.get("uploads", []) if e.get("comp") == "player" and e.get("status") == "uploaded"
           and not e.get("test06") and e.get("file") == Q04_FILE_REL]
    ups.sort(key=lambda e: (parse_iso(e.get("uploaded_at"), "uploaded_at"), e.get("id")))
    return ups[0] if ups else None


def check_q04_upload(ctx):
    path = ctx["LOG"]
    if path is None:
        return {"checked": False, "rule": "dry-run without --log: Q04's first upload is not checked"}
    if not os.path.exists(path):
        refuse(f"{shown(path)} missing: log Q04's first upload first (docs/TESTDAY.md step 8)")
    require_committed(ctx, [path], "results/official_scores.json")
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    e = q04_first_upload(log)
    if e is None:
        refuse(f"Q04's first upload ({Q04_FILE_REL}, status uploaded) is not in {shown(path)}: every 06 step waits "
               f"for it (PREREG_06 5.3 step 1; docs/TESTDAY.md step 8)")
    return {"checked": True, "entry_id": e["id"], "sha256": e.get("sha256"), "uploaded_at": e.get("uploaded_at")}


def run_committed_by_k6p(ctx):
    """K6' passed: a 06-OFF submit report already there (real mode: committed by K6') means the 06 run of the season
    exists and stays valid (tools/official_log.py decides the final upload); no 06 step runs any more."""
    rp = os.path.join(ctx["O6"], "off", REPORT)
    if not os.path.exists(rp):
        return False
    if not ctx["real"]:
        return True
    t = git_commit_time(rp)
    return t is not None and parse_iso(t, "commit time") <= K6P


def common_prereqs(ctx, cmd, pool=None):
    """Spec 7.2, in this order: .venv (2), git clean (2, real mode), tools and the frozen 06 files (2), K6' (3; 2 when
    a 06 run was already committed by K6'), H6P (3), T1 (3 / 2), test02 (2), Q04's first upload (2). Every command
    calls this before it looks at its own outputs, so a registered '06 not used' is never hidden behind 'outputs
    exist'."""
    venv = check_venv()
    if ctx["real"]:
        clean, dirty = git_clean_tracked()
        if not clean:
            refuse(f"git is not clean (tracked changes: {str(dirty)[:300]}): commit after every step "
                   f"(docs/TESTDAY_06.md), then run the same command again")
    fz, models, frozen = load_frozen(ctx)
    tools = check_tools(ctx, fz)
    if ctx["now"] > K6P:
        if run_committed_by_k6p(ctx):
            refuse(f"K6' 已過（{K6P.isoformat()}）：06 的步驟都不再執行；K6' 前已 commit 的 06 照常有效，最終選擇照 "
                   f"PY tools/official_log.py status（docs/TESTDAY_06.md 第 9 步）")
        not_used(ctx, cmd, "k6p", f"K6' 已過（{K6P.isoformat()}；現在 {ctx['now'].isoformat()}）：這一季不用 06", pool)
    h6p = h6p_state(ctx)
    if not h6p["adopted"]:
        not_used(ctx, cmd, "h6p_not_adopted", "H6P 沒有採用（results/prereg06）：這一季不用 06", pool)
    t1 = check_t1(ctx, cmd, pool)
    t02 = check_test02(ctx)
    q04 = check_q04_upload(ctx)
    return {"venv": venv, "fz": fz, "models": models, "frozen": frozen, "tools": tools, "h6p": h6p, "t1": t1,
            "t02": t02, "q04_upload": q04}


# ================================================================ features (feats)
def build_games(d, cand_rows, q_rows):
    """The (id, game_id, colour, sgf) list of p06_player.cmd_apply: candidate games, then question games."""
    by_gid = {r[0]: r for r in cand_rows}
    games = [(g[1], g[0], g[2], by_gid[g[0]][3]) for g in d.cgames]
    qsgf = {(q, s): (c, x) for q, g in q_rows for s, c, x in g}
    games += [(g[0], "%s#%d" % (g[0], g[1]), g[2], qsgf[(g[0], g[1])][1]) for g in d.qgames]
    return games


def load_pool_data(ctx, pool, use_cnn):
    q, c, pdir = pool_files(ctx, pool)
    with st.allow_local():
        d = pf.Data(c, pdir, use_cnn, False, q_path=q)
        cand_rows = pf.read_candidates(c)
        q_rows = pf.read_questions(q)
        shas = {"questions_csv": sha_raw(q), "candidates_csv": sha_raw(c)}
    return d, cand_rows, q_rows, shas, (q, c, pdir)


class StoredStore:
    """A p06_player.MemStore read back from feats.npz (same attributes; the arrays are the saved ones bit for bit)."""

    def __init__(self, z, oraw):
        self.n = int(z["n"])
        self.pid, self.gid, self.color = z["pid"], z["gid"], z["color"]
        self.alt, self.nmov = z["alt"], z["nmov"]
        jo = sp.csr_matrix((z["jo_data"], z["jo_indices"], z["jo_indptr"]), shape=tuple(int(x) for x in z["jo_shape"]))
        self.mats = {"s2": z["s2"], "oraw": oraw, "jo": jo}
        self.oov = {c: int(z["oov_" + c]) for c in "BW"}
        self.Z = {c: dict(zip(z["zrows_" + c].tolist(), z["z_" + c])) for c in "BW"}

    def csr(self, name):
        return self.mats[name]

    def dense(self, name):
        return self.mats[name]


def store_arrays(ms, n_cand):
    jo = ms.mats["jo"].tocsr()
    out = {"n": np.int64(ms.n), "n_cand": np.int64(n_cand), "pid": ms.pid, "gid": ms.gid, "color": ms.color,
           "alt": ms.alt, "nmov": ms.nmov, "s2": ms.mats["s2"], "jo_data": jo.data, "jo_indices": jo.indices,
           "jo_indptr": jo.indptr, "jo_shape": np.array(jo.shape, np.int64)}
    for c in "BW":
        rows = sorted(ms.Z.get(c, {}))
        out["zrows_" + c] = np.array(rows, np.int64)
        out["z_" + c] = (np.vstack([ms.Z[c][r] for r in rows]) if rows else np.zeros((0, st.LDA_DIM)))
        out["oov_" + c] = np.int64(ms.oov.get(c, 0))
    return out


def game_counts(ms, nc):
    col, alt = np.asarray(ms.color), np.asarray(ms.alt)
    nm = np.asarray(ms.nmov)

    def part(sl):
        c, a, m = col[sl], alt[sl], nm[sl]
        return {"B": int((c == "B").sum()), "W": int((c == "W").sum()),
                "not_alternating_B": int(((c == "B") & ~a).sum()), "not_alternating_W": int(((c == "W") & ~a).sum()),
                "without_moves": int((m == 0).sum())}
    return {"candidate": part(slice(0, nc)), "question": part(slice(nc, None))}


def t02_binding(ctx, pool, t02, d_shas):
    """public / private: the files 06 reads are the ones test02 used (its apply_report.json inputs)."""
    if pool == "seam_val":
        return None
    ar = t02["pools"][pool]["report"]
    rec = {k: (ar.get("inputs") or {}).get(k, {}).get("sha256") for k in
           ("candidates_csv", "questions_csv", "cand_games_npz", "query_games_npz")}
    bad = [k for k in ("candidates_csv", "questions_csv") if d_shas.get(k) != rec.get(k)]
    if bad:
        refuse(f"{pool}: {bad} differ from test02's apply_report.json inputs: 06 reads exactly the files test02 used "
               f"(T1 NORMALIZED: the normalized copies in data/test)")
    return rec


def cmd_feats(ctx, a):
    t0 = time.time()
    pool = a.pool
    op = out_paths(ctx, pool)
    pre = common_prereqs(ctx, "feats", pool)
    rerun = guard_existing(ctx, a, [op["feats_report"], op["feats"]])
    refuse_rerun_once_uploaded(ctx, a, pre, "feats")
    d, cand_rows, q_rows, shas, (q, c, _pdir) = load_pool_data(ctx, pool, use_cnn=False)
    binding = t02_binding(ctx, pool, pre["t02"], shas)
    games = build_games(d, cand_rows, q_rows)
    gc, gq = d.G("raw")
    ms = p6.MemStore(games, sp.vstack([gc, gq]).tocsr(), pre["models"])
    arrs = store_arrays(ms, len(d.cgames))
    guard_write(ctx, op["feats"])
    os.makedirs(os.path.dirname(op["feats"]), exist_ok=True)
    tmp = op["feats"] + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(tmp, op["feats"])
    counts = game_counts(ms, len(d.cgames))
    rep = {"format": FORMATS["feats"], "mode": ctx["mode"], "pool": pool,
           "what": "PREREG_06 5.2 feats: per-game v2 features (S2, JO, LDA64 with the frozen LDA_B / LDA_W) of the "
                   "pool's candidate and question games, p06_player.MemStore as in the H6P apply",
           "inputs": {"questions_csv": {"path": shown(q), "sha256": shas["questions_csv"]},
                      "candidates_csv": {"path": shown(c), "sha256": shas["candidates_csv"]}},
           "test02_inputs": binding, "frozen": pre["frozen"],
           "n_candidates": len(d.cand_ids), "n_candidate_games": len(d.cgames), "n_questions": len(d.qids),
           "n_question_games": len(d.qgames), "games": counts, "out_of_vocabulary_tokens": ms.oov,
           "output": {"path": shown(op["feats"]), "sha256": sha_raw(op["feats"]), "rows": int(ms.n)},
           "t1": pre["t1"], "q04_upload": pre["q04_upload"], "rerun": rerun, "tools": tool_record(),
           "env": env_record(), "created": now_iso(), "seconds": round(time.time() - t0, 1)}
    write_json(ctx, op["feats_report"], rep)
    na = {k: v for k, v in counts["candidate"].items() if k.startswith("not_")}
    print(f"feats {pool}: {ms.n} games ({len(d.cgames)} candidate, {len(d.qgames)} question), not alternating "
          f"{na} / question {[counts['question'][k] for k in ('not_alternating_B', 'not_alternating_W')]}, "
          f"oov {ms.oov} -> {shown(op['feats'])} ({time.time() - t0:.1f} s)")
    return 0


def load_feats(ctx, pool, shas_now):
    """The pool's feats.npz as written by feats (report SHA, inputs unchanged; real mode: report committed)."""
    op = out_paths(ctx, pool)
    rep = read_json(op["feats_report"], f"run feats --pool {pool} first")
    if rep.get("format") != FORMATS["feats"] or rep.get("mode") != ctx["mode"] or rep.get("pool") != pool:
        refuse(f"{shown(op['feats_report'])}: format {rep.get('format')!r} mode {rep.get('mode')!r}")
    require_committed(ctx, [op["feats_report"]], f"feats_report.json of {pool}")
    if not os.path.exists(op["feats"]) or sha_raw(op["feats"]) != (rep.get("output") or {}).get("sha256"):
        refuse(f"{shown(op['feats'])} is missing or not the file feats_report.json records: rerun feats --pool {pool}")
    for k in ("questions_csv", "candidates_csv"):
        if (rep.get("inputs") or {}).get(k, {}).get("sha256") != shas_now.get(k):
            refuse(f"{pool}: {k} changed since feats. " + MSG_REDO)
    with np.load(op["feats"], allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    return rep, arrs


# ================================================================ the per-pool computation (seam and apply)
def r02_terms(T, combo, earr, ocanon, cgr, qg):
    """The p02 terms R02 needs, on the same rows as the 06 terms (E0 and raw are the 06 path's own matrices)."""
    terms = {}
    if combo["e"] == "E0":
        terms["E0"] = T["E0"]
    elif combo["e"] == "E1":
        terms["E1"] = pf.e_scores("E1", earr, cgr, earr, qg)[:2]
    if combo["o"] == "raw":
        terms["raw"] = T["O"]
    elif combo["o"] == "canon":
        terms["canon"] = po.open_cos(ocanon, cgr, ocanon, qg)
    return terms


def sr06_pool(ms, d, earr, models, combos, k_star, r02, ocanon=None):
    """PREREG_06 2.3 on a pool: p06_player.sr06_apply's loop with R02 = the given p02 combination (its E0 / E1, raw /
    canon by the p02 functions). Returns ({name: mean} incl. 'R02', {name: [F of rounds 0, 1]})."""
    cand_ids = d.cand_ids
    c_by_pid = {p: d.c_by_player[p] for p in cand_ids}
    ks = [len(d.q_by_id[q]) for q in d.qids]
    cg = {p: [str(ms.gid[r]) for r in c_by_pid[p]] for p in cand_ids}
    row_of = {(p, str(ms.gid[r])): r for p in cand_ids for r in c_by_pid[p]}
    rr = st.sr06_rounds(cg, ks)
    pts = {n: [] for n, _w in combos}
    pts["R02"] = []
    F01 = {n: [] for n, _w in combos}
    for r, items in enumerate(rr):
        cgr = [[row_of[(p, g)] for g in refs] for p, _q, refs in items]
        qsel = [(i, [row_of[(p, g)] for g in q]) for i, (p, q, _refs) in enumerate(items) if q]
        qi = np.array([i for i, _r in qsel], np.int64)
        qg = [rws for _i, rws in qsel]
        T = p6.apply_terms(ms, models, cgr, qg, (earr, cgr, earr, qg))
        for n, w in combos:
            F = sim.combo_F(T, "E0", w)
            pts[n] += st.points(st.ranks(st.csls(F, k_star), qi)).tolist()
            if r in (0, 1):
                F01[n].append(F)
        S = pf.fuse(r02, r02_terms(T, r02, earr, ocanon, cgr, qg))
        pts["R02"] += st.points(st.ranks(S, qi)).tolist()
    return {n: float(np.mean(v)) if v else float("nan") for n, v in pts.items()}, F01


def pool_store(d, arrs):
    """The 06 store of a pool from its feats arrays, with the row identity checked against the pool data."""
    gc, gq = d.G("raw")
    ms = StoredStore(arrs, sp.vstack([gc, gq]).tocsr())
    nc = len(d.cgames)
    want_gid = [g[0] for g in d.cgames] + ["%s#%d" % (g[0], g[1]) for g in d.qgames]
    want_pid = [g[1] for g in d.cgames] + [g[0] for g in d.qgames]
    if int(arrs["n_cand"]) != nc or ms.n != len(want_gid) or ms.gid.tolist() != want_gid \
            or ms.pid.tolist() != want_pid:
        refuse("feats.npz rows do not match the pool's candidate / question games: rerun feats")
    return ms


def compute_pool(d, ms, models, fz, r02):
    """sr06, the PREREG_06 2.4 selection, F of the selected combination, CSLSk top-5 (off), the switch top-5 (on),
    the detector with the selected combination's r*_X. Returns a dict (F kept for the structure checks)."""
    nc = len(d.cgames)
    emb = np.vstack([d.cemb, d.qemb])
    nw = np.concatenate([d.cnw, d.qnw])
    ok = np.concatenate([d.cok, d.qok])
    earr = (emb, nw, ok)
    cgroups = [list(d.c_by_player[p]) for p in d.cand_ids]
    qgroups = [[nc + i for i in d.q_by_id[q]] for q in d.qids]
    ocanon = None
    if r02["o"] == "canon":
        gcc, gqc = d.G("canon")
        ocanon = sp.vstack([gcc, gqc]).tocsr()
    T = p6.apply_terms(ms, models, cgroups, qgroups, (earr, cgroups, earr, qgroups))
    combos = sim.combo_list(fz["combinations"]["V1"])
    order = [n for n, _w in combos]
    kst = fz["k_star"]
    t0 = time.time()
    srm, F01 = sr06_pool(ms, d, earr, models, combos, kst, r02, ocanon)
    chosen = p6.choose_combo({n: srm[n] for n in order}, order)
    w = dict(combos)[chosen]
    F = sim.combo_F(T, "E0", w)
    off = st.top5(st.csls(F, kst), d.cand_ids)
    swc = (fz.get("switch") or {}).get(chosen) or {}
    r_star, usable = swc.get("r_star"), bool(swc.get("usable")) and bool(fz.get("switch_used_at_all"))
    if len(d.qids) <= len(d.cand_ids):
        a = st.hungarian_top1(st.zrows(F))
        on = st.switch_top5(F, d.cand_ids, a)
        det = st.detector(F, F01[chosen])
        opens = bool(usable and r_star is not None and det["r_forced_cost"] <= r_star)
        det_rec = {"r_forced_cost": det["r_forced_cost"], "forced_cost": det["forced_cost"],
                   "pseudo_forced_cost": det["pseudo_forced_cost"], "r_star": r_star, "usable": usable,
                   "opens": opens, "changed_top1": int(sum(1 for x, y in zip(off, on) if x[0] != y[0])),
                   "assignment_distinct": len({t[0] for t in on}) == len(on)}
    else:
        a, on, opens = None, None, False
        det_rec = {"applicable": False, "rule": "more questions than candidates: no one-to-one assignment",
                   "r_star": r_star, "usable": usable, "opens": False}
    q04_06 = st.top5(sim.combo_F(T, "E0", None), d.cand_ids)
    return {"sr06": srm, "order": order, "selected": chosen, "weights": w, "k_star": kst, "F": F, "off": off,
            "on": on, "assignment": a, "detector": det_rec, "opens": opens, "q04_06_path": q04_06,
            "terms": T, "earr": earr, "cgroups": cgroups, "qgroups": qgroups, "combos": combos,
            "sr06_seconds": round(time.time() - t0, 1)}


def q04_p02(d):
    """Q04 with the p02 functions (p06_player.cmd_apply step 3)."""
    c_groups = [d.c_by_player[p] for p in d.cand_ids]
    q_groups = [d.q_by_id[q] for q in d.qids]
    gc, gq = d.G("raw")
    cosr, hasr = po.open_cos(gc, c_groups, gq, q_groups)
    ME, hasE, _ = pf.e_scores("E0", (d.cemb, d.cnw, d.cok), c_groups, (d.qemb, d.qnw, d.qok), q_groups)
    return pf.top_ids(pf.fuse(pf.COMBO_BY_ID["Q04"], {"E0": (ME, hasE), "raw": (cosr, hasr)}), d.cand_ids)


def rows_bytes(qids, tops):
    buf = [",".join(["question_id"] + pf.TOP_COLS)]
    for q, t in zip(qids, tops):
        out = []
        csv.writer(_Collect(out), lineterminator="").writerow([q] + t)
        buf.append(out[0])
    return ("\n".join(buf) + "\n").encode("utf-8")


class _Collect:
    def __init__(self, out):
        self.out = out

    def write(self, s):
        self.out.append(s)


def dumps_bitwise(pool_dumps, ref_dumps):
    """Array-by-array comparison of test02's seam_val dumps with prereg02's (npz bytes carry zip timestamps)."""
    out, ok = {}, True
    for n in ("cand_games.npz", "query_games.npz"):
        pa, pb = os.path.join(pool_dumps, n), os.path.join(ref_dumps, n)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            out[n] = {"compared": False}
            ok = False
            continue
        with np.load(pa, allow_pickle=False) as za, np.load(pb, allow_pickle=False) as zb:
            keys = sorted(set(za.files) | set(zb.files))
            eq = {k: bool(k in za.files and k in zb.files and za[k].dtype == zb[k].dtype and za[k].shape == zb[k].shape
                          and np.array_equal(za[k], zb[k])) for k in keys}
        out[n] = {"compared": True, "arrays_equal": eq, "bitwise_equal": all(eq.values()), "sha256": sha_raw(pa),
                  "reference_sha256": sha_raw(pb)}
        ok = ok and all(eq.values())
    return ok, out


def identical_rows(tops, qids, ref_path):
    ref = {r[0]: r[1:] for r in read_top5(ref_path)}
    return sum(1 for q, t in zip(qids, tops) if ref.get(q) == t), list(ref) == list(qids)


# ================================================================ seam
def cmd_seam(ctx, a):
    t0 = time.time()
    op = out_paths(ctx, "seam_val")
    outs = [op["seam"], op["off"], op["on"]]
    pre = common_prereqs(ctx, "seam", "seam_val")
    rerun = guard_existing(ctx, a, outs)
    refuse_rerun_once_uploaded(ctx, a, pre, "seam")
    fz, models = pre["fz"], pre["models"]
    d, _cr, _qr, shas, (q, c, pdir) = load_pool_data(ctx, "seam_val", use_cnn=True)
    frep, arrs = load_feats(ctx, "seam_val", shas)
    # the seam_val dumps are the ones test02's own seam_val run read (as apply binds public / private): a verdict is
    # never drawn from dumps test02 did not judge (review 10/3)
    t02_seam_ar = os.path.join(ctx["O2"], "seam_val", "apply_report.json")
    require_committed(ctx, [t02_seam_ar], "test02's seam_val apply_report.json")
    t02_seam_inputs = read_json(t02_seam_ar, "test02 competition 1, seam_val").get("inputs") or {}
    seam_dump_bad = [k for k in ("cand_games_npz", "query_games_npz")
                     if d.shas.get(k) != (t02_seam_inputs.get(k) or {}).get("sha256")]
    if seam_dump_bad:
        refuse(f"seam_val: dumps {seam_dump_bad} differ from test02's seam_val apply_report.json inputs. " + MSG_REDO)
    ref_dumps = os.path.join(P02_PLAYER, "dumps")
    for n, want in P02_DUMP_SHA.items():
        if sha_raw(os.path.join(ref_dumps, n)) != want:
            refuse(f"results/prereg02/player/dumps/{n} differs from the pinned SHA-256")
    bitwise, dump_cmp = dumps_bitwise(os.path.join(pdir, "dumps"), ref_dumps)
    p02_top = os.path.join(P02_PLAYER, "player_val_top5.csv")
    if sha_lf(p02_top) != P02_TOP5_SHA_LF:
        refuse("results/prereg02/player/player_val_top5.csv differs from the pinned SHA-256")
    q04_same = rows_bytes(d.qids, q04_p02(d)) == bytes_lf(p02_top)
    ms = pool_store(d, arrs)
    r02 = pf.COMBO_BY_ID["Q04"]                       # PREREG_06: R02 of the validation pool is Q04
    res = compute_pool(d, ms, models, fz, r02)
    q04_06_same = rows_bytes(d.qids, res["q04_06_path"]) == bytes_lf(p02_top)
    srm, chosen = res["sr06"], res["selected"]
    c_by_pid = {p: d.c_by_player[p] for p in d.cand_ids}
    srm_p6, _ = p6.sr06_apply(ms, d.cand_ids, c_by_pid, [len(d.q_by_id[x]) for x in d.qids], res["earr"], models,
                              res["combos"], res["k_star"])
    write_rows(ctx, op["off"], [[x] + t for x, t in zip(d.qids, res["off"])])
    write_rows(ctx, op["on"], [[x] + t for x, t in zip(d.qids, res["on"])])
    P6 = ctx["P6"]
    h6p = read_json(os.path.join(P6, "apply_report.json"))
    ref_off, ref_on = os.path.join(P6, "player_val_top5.csv"), os.path.join(P6, "player_val_top5_switch.csv")
    n_off, same_q_off = identical_rows(res["off"], d.qids, ref_off)
    n_on, same_q_on = identical_rows(res["on"], d.qids, ref_on)
    byte_off = bytes_lf(op["off"]) == bytes_lf(ref_off)
    byte_on = bytes_lf(op["on"]) == bytes_lf(ref_on)
    gate_v0 = srm[chosen] - srm["V0"]
    gate_r02 = srm[chosen] - srm["R02"]
    det = res["detector"]
    h_det = h6p.get("detector") or {}
    n = len(d.qids)
    crit = {
        "selected_equal_h6p": chosen == h6p.get("selected"),
        "score_gate_equal_h6p": (gate_v0 >= p6.GATE_MIN - 1e-12) ==
                                ((h6p.get("gate") or {}).get("sr_selected_minus_V0", -1) >= p6.GATE_MIN - 1e-12),
        "test_day_gate_equal_h6p": (gate_r02 >= GATE06_MIN) == (h6p.get("sr_selected_minus_R02", -1) >= GATE06_MIN),
        "detector_equal_h6p": det["opens"] == h_det.get("would_open"),
        "same_questions": same_q_off and same_q_on,
    }
    if bitwise:
        crit.update({"top5_off_byte_identical": byte_off, "top5_on_byte_identical": byte_on})
    else:
        crit.update({"top5_off_identical_at_least_%d" % SEAM_MIN_IDENTICAL: n_off >= SEAM_MIN_IDENTICAL,
                     "top5_on_identical_at_least_%d" % SEAM_MIN_IDENTICAL: n_on >= SEAM_MIN_IDENTICAL})
    counts = frep.get("games")
    dump_independent = {
        "games_equal_h6p": {k: {kk: vv for kk, vv in v.items() if kk != "without_moves"} for k, v in counts.items()}
        == h6p.get("games"),
        "out_of_vocabulary_equal_h6p": frep.get("out_of_vocabulary_tokens") == h6p.get("out_of_vocabulary_tokens"),
        "lda_equal_h6p": pre["frozen"]["lda_sha256"] == h6p.get("lda_sha256"),
    }
    bitwise_additions = None
    if bitwise:
        bitwise_additions = {
            "q04_p02_byte_identical": q04_same, "q04_06_path_byte_identical": q04_06_same,
            "sr06_equal_h6p_rounded": {k: round(v, 6) for k, v in srm.items()} == h6p.get("sr06"),
            "sr06_equal_p06_sr06_apply": srm == srm_p6,
            "gate_values_equal_h6p": gate_v0 == (h6p.get("gate") or {}).get("sr_selected_minus_V0")
            and gate_r02 == h6p.get("sr_selected_minus_R02"),
            "r_forced_cost_equal_h6p": det.get("r_forced_cost") == h_det.get("r_forced_cost"),
        }
    passed = all(crit.values()) and all(dump_independent.values()) and (
        bitwise_additions is None or all(bitwise_additions.values()))
    if passed:
        verdict, code = "passed", 0
    elif bitwise or not all(dump_independent.values()):
        verdict, code = "local_drift", 4
    else:
        verdict, code = "not_used_dumps_differ", 3
    rep = {"format": FORMATS["seam"], "mode": ctx["mode"], "pool": "seam_val",
           "what": "PREREG_06 5.2 seam: the H6P apply again on the validation pool with test02's seam_val dumps",
           "registered_rule": "selected combination, gate decisions and detector equal H6P; dumps bitwise equal to "
                              "prereg02's -> both top-5 files byte-identical to H6P's; else >= 398/400 identical rows",
           "added_rule": "dump-independent counts always equal H6P; with bitwise dumps also sr06, the gate values, "
                         "r_forced_cost, Q04 (p02 functions and the 06 path) and sr06 = p06_player.sr06_apply exactly",
           "dumps_bitwise_equal": bitwise, "dump_comparison": dump_cmp,
           "test02_seam_local_dumps_bitwise_equal": read_json(pre["t02"]["seam_local"]).get("dumps_bitwise_equal"),
           "criteria": crit, "dump_independent": dump_independent, "bitwise_additions": bitwise_additions,
           "passed": passed, "verdict": verdict,
           "sr06": srm, "sr06_p06_sr06_apply": srm_p6, "selected": chosen, "order": res["order"],
           "k_star": res["k_star"], "gate": {"sr_selected_minus_V0": gate_v0, "score_gate_min": p6.GATE_MIN,
                                             "sr_selected_minus_R02": gate_r02, "test_day_gate_min": GATE06_MIN},
           "detector": det, "h6p": {"selected": h6p.get("selected"), "sr06": h6p.get("sr06"),
                                    "gate": h6p.get("gate"), "sr_selected_minus_R02": h6p.get("sr_selected_minus_R02"),
                                    "detector": h_det},
           "n_questions": n, "n_top5_off_identical": n_off, "n_top5_on_identical": n_on,
           "byte_identical_top5_off": byte_off, "byte_identical_top5_on": byte_on,
           "q04_p02_byte_identical": q04_same, "q04_06_path_byte_identical": q04_06_same,
           "outputs": {"top5_off.csv": sha_raw(op["off"]), "top5_on.csv": sha_raw(op["on"])},
           "inputs": {"questions_csv": shas["questions_csv"], "candidates_csv": shas["candidates_csv"],
                      "dumps": {k: d.shas.get(k) for k in ("cand_games_npz", "query_games_npz")},
                      "feats_report_sha256_lf": sha_lf(op["feats_report"]), "feats_npz": sha_raw(op["feats"]),
                      "test02_seam_local_sha256_lf": pre["t02"]["seam_local_sha256_lf"],
                      "test02_seam_val_apply_report_sha256_lf": sha_lf(t02_seam_ar)},
           "frozen": pre["frozen"], "h6p_state": pre["h6p"], "t1": pre["t1"], "q04_upload": pre["q04_upload"],
           "rerun": rerun, "tools": tool_record(), "env": env_record(), "sr06_seconds": res["sr06_seconds"],
           "created": now_iso(), "seconds": round(time.time() - t0, 1)}
    write_json(ctx, op["seam"], rep)
    print("seam %s: selected %s (sr06 %s), gates %.4f / %.4f, detector %s (r %.4f), top5 off %d/%d on %d/%d, "
          "dumps bitwise %s" % (verdict.upper(), chosen, {k: round(v, 6) for k, v in srm.items()}, gate_v0, gate_r02,
                                det["opens"], det.get("r_forced_cost", float("nan")), n_off, n, n_on, n, bitwise))
    if code == 4:
        raise T06Error("the 06 seam failed although its inputs are bitwise those of H6P (or a dump-independent count "
                       "differs): local drift; commit t06_seam.json, stop and check (PREREG_06 rev 3; relaxing needs "
                       "a revision): %s" % [k for k, v in {**crit, **dump_independent, **(bitwise_additions or {})}
                                             .items() if not v], 4, "seam_drift")
    if code == 3:
        not_used(ctx, "seam", "seam_failed_dumps_differ", "06 的 seam 沒過，而且 test02 的 seam_val dumps 和 prereg02 "
                 "不是逐位元組相同（PREREG_06 rev 3）：這一季不用 06", "seam_val")
    return 0


def load_seam(ctx, pre):
    op = out_paths(ctx, "seam_val")
    rep = read_json(op["seam"], "run seam first")
    require_committed(ctx, [op["seam"]], "t06_seam.json")
    if rep.get("format") != FORMATS["seam"] or rep.get("mode") != ctx["mode"]:
        refuse(f"{shown(op['seam'])}: format {rep.get('format')!r} mode {rep.get('mode')!r}")
    if rep.get("verdict") == "not_used_dumps_differ":
        not_used(ctx, "seam", "seam_failed_dumps_differ", "06 的 seam 沒過（dumps 不同）：這一季不用 06", "seam_val",
                 write=False)
    if rep.get("passed") is not True:
        refuse(f"the 06 seam did not pass ({shown(op['seam'])}, verdict {rep.get('verdict')!r}): no test pool runs")
    if (rep.get("frozen") or {}).get("sha256_lf") != pre["frozen"]["sha256_lf"]:
        refuse("t06_seam.json was made under another frozen_config.json")
    if (rep.get("inputs") or {}).get("test02_seam_local_sha256_lf") != pre["t02"]["seam_local_sha256_lf"]:
        refuse("test02's seam_local.json changed since the 06 seam. " + MSG_REDO)
    return rep, sha_lf(op["seam"])


# ================================================================ apply (one test pool)
def load_t2(ctx):
    """test02's T2 (C/player/t2.json); None when missing. Real mode: committed, real, not a fake fixture."""
    path = os.path.join(ctx["C"], "player", "t2.json")
    if not os.path.exists(path):
        return None, None
    t2 = read_json(path)
    if ctx["real"]:
        require_committed(ctx, [path], "test02's T2")
        if t2.get("format") != T2_FORMAT or (t2.get("provenance") or {}).get("mode") != "real" or t2.get("fake"):
            refuse(f"{shown(path)}: format {t2.get('format')!r}, mode {(t2.get('provenance') or {}).get('mode')!r}, "
                   f"fake {t2.get('fake')!r}")
    return t2, path


def t2_summary(t2):
    """Unit (c) copies per question file and the FAIL units among (a), (b), (d); None when T2 is missing."""
    if t2 is None:
        return None
    units = t2.get("units") or []
    c = {u.get("file"): u.get("copies") for u in units if u.get("unit") == "c"}
    failed = [f"{u.get('unit')}:{u.get('file') or u.get('pool') or ''}" for u in units
              if u.get("unit") in ("a", "b", "d") and u.get("verdict") == "FAIL"]
    have = sorted({u.get("unit") for u in units})
    return {"c_copies": c, "abd_failed": failed, "units_present": have, "verdict": t2.get("verdict"),
            "fake": t2.get("fake")}


def checks_summary(ctx):
    p = os.path.join(ctx["C"], "player", "summary.json")
    s = maybe_json(p)
    return {c: (v or {}).get("verdict") for c, v in ((s or {}).get("checks") or {}).items()} if s else None


def pool_precheck(ar):
    """PREREG_06 2.7 (and rev 3) before any 06 computation: (use 06?, reason, switch allowed at this level)."""
    level, gate = ar.get("guard_level"), ar.get("gate_passed")
    if gate is not True:
        return False, "test02's gate failed in this pool (Q00): test02's rows (PREREG_06 2.7)", False
    if level == 3:
        return False, "test02's guard is at level 3 (opening unreliable, CNN only): test02's rows (PREREG_06 2.7)", False
    if level == 0:
        return False, ("test02 selected a CNN-only combination (guard exempt, level 0): the opening was not judged, "
                       "test02's rows (PREREG_06 rev 3)"), False
    if level == 2:
        return True, "guard level 2: 06 as usual (opening weights already <= 1), no one-to-one (PREREG_06 2.7)", False
    if level == 1:
        return True, "guard level 1", True
    refuse(f"unexpected guard level {level!r} in test02's apply_report.json")


def cmd_apply(ctx, a):
    t0 = time.time()
    pool = a.pool
    op = out_paths(ctx, pool)
    pre = common_prereqs(ctx, "apply", pool)
    rerun = guard_existing(ctx, a, [op["pool"], op["off"], op["on"]])
    refuse_rerun_once_uploaded(ctx, a, pre, "apply")
    fz, models = pre["fz"], pre["models"]
    seam, seam_sha = load_seam(ctx, pre)
    tp = pre["t02"]["pools"][pool]
    ar = tp["report"]
    ps = read_json(tp["pool_select"])
    t2, t2_path = load_t2(ctx)
    t2s = t2_summary(t2)
    use06, why, level_switch = pool_precheck(ar)
    d, _cr, _qr, shas, (q, c, pdir) = load_pool_data(ctx, pool, use_cnn=True)
    binding = t02_binding(ctx, pool, pre["t02"], dict(shas, **{k: d.shas.get(k) for k in
                                                               ("cand_games_npz", "query_games_npz")}))
    dump_bad = [k for k in ("cand_games_npz", "query_games_npz") if d.shas.get(k) != (binding or {}).get(k)]
    if dump_bad:
        refuse(f"{pool}: dumps {dump_bad} differ from test02's apply_report.json inputs. " + MSG_REDO)
    if d.qids != [r[0] for r in read_top5(tp["top5"])]:
        refuse(f"{pool}: question order differs from test02's top5.csv")
    frep, arrs = load_feats(ctx, pool, shas)
    if os.path.exists(op["on"]):
        os.remove(guard_write(ctx, op["on"]))
    applied = ar.get("applied") or {}
    r02 = pf.COMBO_BY_ID.get(applied.get("id"))
    if r02 is None or any(r02[k] != applied.get(k) for k in ("e", "o", "w")):
        refuse(f"{pool}: test02's applied combination {applied!r} is not a p02 combination")
    res, table = None, [{"row": "T1", "value": pre["t1"]["verdict"], "outcome": "PASS / NORMALIZED: continue"},
                        {"row": "test02 gate / guard", "value": {"gate_passed": ar.get("gate_passed"),
                                                                 "guard_level": ar.get("guard_level")},
                         "outcome": why}]
    on_written, on_role, gate06 = False, None, None
    if use06:
        ms = pool_store(d, arrs)
        res = compute_pool(d, ms, models, fz, r02)
        diff = res["sr06"][res["selected"]] - res["sr06"]["R02"]
        gate06 = {"sr_selected_minus_R02": diff, "min": GATE06_MIN, "passed": bool(diff >= GATE06_MIN)}
        table.append({"row": "06 gate", "value": diff, "outcome": "passed" if gate06["passed"] else
                      "sr(selected) - sr(R02) < 0: test02's rows (PREREG_06 2.7)"})
        if not gate06["passed"]:
            use06, why = False, "the 06 gate failed: sr(selected) - sr(R02) < 0, test02's rows (PREREG_06 2.7)"
    if use06:
        write_rows(ctx, op["off"], [[x] + t for x, t in zip(d.qids, res["off"])])
        if res["on"] is None:
            on_role = "no one-to-one assignment: more questions than candidates (no 06-ON)"
        elif pool == "public":
            on_written, on_role = True, "the forced probe of 06-ON (PREREG_06 5.3: public always opens)"
        elif level_switch and res["opens"]:
            on_written, on_role = True, "private: level 1, the combination usable, the detector open"
        if on_written:
            write_rows(ctx, op["on"], [[x] + t for x, t in zip(d.qids, res["on"])])
    else:
        guard_write(ctx, op["off"])
        os.makedirs(os.path.dirname(op["off"]), exist_ok=True)
        shutil.copyfile(tp["top5"], op["off"])
    t02_rows = {r[0]: r[1:] for r in read_top5(tp["top5"])}
    off_rows = read_top5(op["off"])
    rep = {"format": FORMATS["pool"], "mode": ctx["mode"], "pool": pool, "used06": use06,
           "what": "PREREG_06 2.7 / 5.2 apply on one test pool",
           "fallback_to_test02": not use06, "rule": why, "decision_table": table,
           "test02": {"selected": (ar.get("selected") or {}).get("id"), "gate_passed": ar.get("gate_passed"),
                      "effective": (ar.get("effective") or {}).get("id"), "applied": applied,
                      "guard_level": ar.get("guard_level"), "guard_reason": ar.get("guard_reason"),
                      "rho": ar.get("rho"), "n_questions": ar.get("n_questions"),
                      "n_candidates": ar.get("n_candidates"), "n_candidate_games": ar.get("n_candidate_games"),
                      "self_retrieval_mean_selected": (ps.get("self_retrieval_mean") or {}).get(
                          (ps.get("selected") or {}).get("id")),
                      "apply_report_sha256_lf": sha_lf(tp["apply_report"]),
                      "pool_select_sha256_lf": sha_lf(tp["pool_select"]), "top5_sha256": sha_raw(tp["top5"])},
           "r02": r02, "t2": t2s, "t2_path": shown(t2_path) if t2_path else None,
           "checks_t2_t5_record_only": checks_summary(ctx),
           "switch_allowed_at_this_level": level_switch,
           "sr06": res["sr06"] if res else None, "order": res["order"] if res else None,
           "selected": res["selected"] if res else None, "weights": res["weights"] if res else None,
           "k_star": fz["k_star"], "gate06": gate06,
           "score_gate_vs_V0_record_only": (res["sr06"][res["selected"]] - res["sr06"]["V0"]) if res else None,
           "detector": res["detector"] if res else None,
           "top5_on_written": on_written, "top5_on_role": on_role,
           "top1_differences_vs_test02": sum(1 for r in off_rows if t02_rows.get(r[0], [None])[0] != r[1]),
           "outputs": {"top5_off.csv": sha_raw(op["off"]), "top5_on.csv": sha_raw(op["on"]) if on_written else None},
           "n_questions": len(d.qids), "n_candidates": len(d.cand_ids), "n_candidate_games": len(d.cgames),
           "games": frep.get("games"), "out_of_vocabulary_tokens": frep.get("out_of_vocabulary_tokens"),
           "inputs": {"questions_csv": shas["questions_csv"], "candidates_csv": shas["candidates_csv"],
                      "dumps": {k: d.shas.get(k) for k in ("cand_games_npz", "query_games_npz")},
                      "feats_report_sha256_lf": sha_lf(op["feats_report"]), "feats_npz": sha_raw(op["feats"]),
                      "seam_sha256_lf": seam_sha, "t2_sha256_lf": sha_lf(t2_path) if t2_path else None},
           "frozen": pre["frozen"], "t1": pre["t1"], "q04_upload": pre["q04_upload"], "rerun": rerun,
           "tools": tool_record(), "env": env_record(), "sr06_seconds": res["sr06_seconds"] if res else None,
           "created": now_iso(), "seconds": round(time.time() - t0, 1)}
    write_json(ctx, op["pool"], rep)
    msg = (f"apply {pool}: {'06 ' + res['selected'] if use06 else 'test02 rows'} ({why}); "
           + (f"sr06 { {k: round(v, 6) for k, v in res['sr06'].items()} }, detector opens {res['opens']} "
              f"(r {res['detector'].get('r_forced_cost')}), top5_on {'written' if on_written else 'not written'}"
              if res else "no 06 computation"))
    print(msg)
    other = "private" if pool == "public" else "public"
    orep = maybe_json(out_paths(ctx, other)["pool"])
    # the other pool's report counts only when it was made from the present test02 report of that pool and the
    # present seam (a report left from before a test02 re-run is ignored; submit decides then)
    o_ar = pre["t02"]["pools"][other]["apply_report"]
    o_current = bool(orep) and (orep.get("test02") or {}).get("apply_report_sha256_lf") == sha_lf(o_ar) and \
        (orep.get("inputs") or {}).get("seam_sha256_lf") == seam_sha
    if not use06 and orep and o_current and orep.get("mode") == ctx["mode"] and orep.get("used06") is False:
        not_used(ctx, "apply", "both_pools_test02", "兩池都退回 test02 的列：06 不用（PREREG_06 rev 3）", pool)
    return 0


# ================================================================ submit
def on_conditions(pub, priv, h6p, t2s, aidea_checked, aidea_not_o2o, now):
    """PREREG_06 5.3: every 06-ON condition with its value (the pool reports, H6P's secondary, T2, the AIdea page,
    K7'). -> {name: {'ok': bool, 'detail': ...}}."""
    out = {}
    pd = (priv or {}).get("detector") or {}
    out["private_switch_usable_and_detector_open"] = {
        "ok": bool(priv.get("used06") and pd.get("usable") and pd.get("opens")
                   and priv.get("switch_allowed_at_this_level")),
        "detail": {"selected": priv.get("selected"), "usable": pd.get("usable"), "r_forced_cost": pd.get("r_forced_cost"),
                   "r_star": pd.get("r_star"), "opens": pd.get("opens")}}
    out["h6p_secondary_not_negative"] = {"ok": bool(h6p.get("secondary_not_negative")),
                                         "detail": h6p.get("score_H6P_switch")}
    sizes = {p: {"questions": r.get("n_questions"), "candidates": r.get("n_candidates"),
                 "test02_questions": (r.get("test02") or {}).get("n_questions"),
                 "test02_candidates": (r.get("test02") or {}).get("n_candidates")} for p, r in
             (("public", pub), ("private", priv))}
    out["pools_400_by_400"] = {"ok": all(v["questions"] == N_OFFICIAL and v["candidates"] == N_OFFICIAL and
                                         v["test02_questions"] == N_OFFICIAL and v["test02_candidates"] == N_OFFICIAL
                                         for v in sizes.values()), "detail": sizes}
    if t2s is None:
        out["t2_c_zero_and_abd_not_failed"] = {"ok": False, "detail": "t2.json missing"}
    else:
        qn = [TEST_FILES[p][0] for p in TEST_POOLS]
        c_ok = all(t2s["c_copies"].get(n) == 0 for n in qn)
        out["t2_c_zero_and_abd_not_failed"] = {"ok": bool(c_ok and not t2s["abd_failed"] and
                                                          {"a", "b", "c", "d"} <= set(t2s["units_present"])),
                                               "detail": t2s}
    out["both_pools_06_at_guard_level_1"] = {
        "ok": all(r.get("used06") and (r.get("test02") or {}).get("guard_level") == 1 for r in (pub, priv)),
        "detail": {p: {"used06": r.get("used06"), "guard_level": (r.get("test02") or {}).get("guard_level")}
                   for p, r in (("public", pub), ("private", priv))}}
    out["aidea_page_not_against_one_to_one"] = {
        "ok": bool((aidea_checked or "").strip()) and not aidea_not_o2o,
        "detail": {"checked": (aidea_checked or "").strip() or None, "says_not_one_to_one": bool(aidea_not_o2o)}}
    out["before_k7"] = {"ok": now <= K7P, "detail": {"now": now.isoformat(), "k7p": K7P.isoformat()}}
    return out


def top1_of(rows):
    return {r[0]: r[1] for r in rows}


def top1_diff(rows_a, rows_b):
    a, b = top1_of(rows_a), top1_of(rows_b)
    return sum(1 for q, v in a.items() if b.get(q) != v)


def cmd_submit(ctx, a):
    t0 = time.time()
    paths = {v: (os.path.join(ctx["O6"], v, SUBMISSION), os.path.join(ctx["O6"], v, REPORT)) for v in VARIANTS}
    pre = common_prereqs(ctx, "submit")
    rerun = guard_existing(ctx, a, [p for v in VARIANTS for p in paths[v]])
    seam, seam_sha = load_seam(ctx, pre)
    pools, rows_off, rows_on, qfiles, cfiles = {}, [], [], [], []
    for pool in TEST_POOLS:
        op = out_paths(ctx, pool)
        rep = read_json(op["pool"], f"run apply --pool {pool} first")
        require_committed(ctx, [op["pool"], op["off"]] + ([op["on"]] if rep.get("top5_on_written") else []),
                          f"the 06 outputs of {pool}")
        if rep.get("format") != FORMATS["pool"] or rep.get("mode") != ctx["mode"]:
            refuse(f"{shown(op['pool'])}: format {rep.get('format')!r} mode {rep.get('mode')!r}")
        if sha_raw(op["off"]) != rep["outputs"]["top5_off.csv"]:
            refuse(f"{pool}: top5_off.csv changed after apply")
        if rep.get("top5_on_written") and sha_raw(op["on"]) != rep["outputs"]["top5_on.csv"]:
            refuse(f"{pool}: top5_on.csv changed after apply")
        if rep["inputs"].get("seam_sha256_lf") != seam_sha:
            refuse(f"{pool}: was applied under another t06_seam.json; rerun apply --pool {pool}")
        tp = pre["t02"]["pools"][pool]
        if (rep.get("test02") or {}).get("apply_report_sha256_lf") != sha_lf(tp["apply_report"]) or \
                (rep.get("test02") or {}).get("top5_sha256") != sha_raw(tp["top5"]):
            refuse(f"{pool}: test02's reports changed since apply. " + MSG_REDO)
        q, c, _pd = pool_files(ctx, pool)
        if rep["inputs"].get("questions_csv") != sha_raw(q) or rep["inputs"].get("candidates_csv") != sha_raw(c):
            refuse(f"{pool}: the test files changed since apply. " + MSG_REDO)
        body = read_top5(op["off"])
        qids = [r[0] for r in body]
        file_qids = [r[0] for r in read_top5(tp["top5"])]
        if qids != file_qids:
            refuse(f"{pool}: top5_off.csv rows are not in test02's (question-file) order")
        rows_off += body
        if rep.get("top5_on_written"):
            on_body = read_top5(op["on"])
            if [r[0] for r in on_body] != qids:
                refuse(f"{pool}: top5_on.csv rows are not in question-file order")
            rows_on.append((pool, on_body))
        qfiles.append(q)
        cfiles.append(c)
        pools[pool] = {"report": rep, "paths": op, "rows_off": body, "t02_top5": tp["top5"]}
    if not pools["public"]["report"]["used06"] and not pools["private"]["report"]["used06"]:
        not_used(ctx, "submit", "both_pools_test02", "兩池都退回 test02 的列：06-OFF 就是 Q04，不產生交件檔", None)
    run_id = hashlib.sha256("|".join([seam_sha] + [sha_lf(pools[p]["paths"]["pool"]) for p in TEST_POOLS])
                            .encode()).hexdigest()
    logged = run_in_log(ctx, run_id)
    if logged:
        refuse(f"the upload log already holds 06 entries {logged} of this 06 run (run_id {run_id[:12]}): rewriting "
               f"t06_submit_report.json would make them not meet PREREG_06 5.4 (the report changed since prepare), "
               f"and no 06-ON can be added to this run any more (docs/TESTDAY_06.md step 5)")
    t2, t2_path = load_t2(ctx)
    t2s = t2_summary(t2)
    cond = on_conditions(pools["public"]["report"], pools["private"]["report"], pre["h6p"], t2s,
                         a.aidea_page_checked, a.aidea_not_one_to_one, ctx["now"])
    all_on = all(v["ok"] for v in cond.values()) and len(rows_on) == 2
    os.makedirs(os.path.join(ctx["O6"], "off"), exist_ok=True)
    sub_off = write_rows(ctx, paths["off"][0], rows_off)
    sub_on = None
    if all_on:
        sub_on = write_rows(ctx, paths["on"][0], [r for _p, body in rows_on for r in body])
    elif os.path.exists(paths["on"][0]):
        for p in paths["on"]:
            if os.path.exists(p):
                os.remove(guard_write(ctx, p))
    checks = {}
    for v, sub in (("off", sub_off), ("on", sub_on)):
        if sub is None:
            continue
        errs = check_submission.check("player", sub, qfiles, cfiles, official=True)
        checks[v] = {"official": True, "passed": not errs, "errors": errs[:50], "n_errors": len(errs),
                     "question_files": [{"path": shown(x), "sha256": sha_raw(x)} for x in qfiles]}
    t02_rows = {p: read_top5(pools[p]["t02_top5"]) for p in TEST_POOLS}
    diff_q04 = {"off": {p: top1_diff(pools[p]["rows_off"], t02_rows[p]) for p in TEST_POOLS}}
    diff_off_on = None
    if sub_on:
        on_by = dict(rows_on)
        diff_q04["on"] = {p: top1_diff(on_by[p], t02_rows[p]) for p in TEST_POOLS}
        diff_off_on = {p: top1_diff(pools[p]["rows_off"], on_by[p]) for p in TEST_POOLS}
    # 06-OFF binds what 06-OFF is made of; T2 and the top5_on files are inputs of 06-ON only (a later rewrite of
    # t2.json never invalidates an uploaded 06-OFF; review 10/3)
    bound = {"off": inputs_bound(ctx, pre, pools, None, with_on=False),
             "on": inputs_bound(ctx, pre, pools, t2_path, with_on=True) if sub_on else None}
    reps = {}
    for v, sub in (("off", sub_off), ("on", sub_on)):
        if sub is None:
            continue
        with open(sub, "rb") as f:
            raw = f.read()
        reps[v] = {"format": FORMATS["submit"], "mode": ctx["mode"], "variant": v, "method": METHOD[v],
                   "what": "PREREG_06 5.2 / 5.3 submit: submission_player.csv = public rows, then private rows "
                           "(question-file order), question_id,top_1..top_5, UTF-8 without BOM, LF",
                   "run_id": run_id, "on_conditions": cond, "all_on_conditions": all_on, "on_produced": bool(sub_on),
                   "check_submission": checks[v],
                   "output": {"path": shown(sub), "sha256": hashlib.sha256(raw).hexdigest(), "rows": raw.count(b"\n") - 1,
                              "bytes": len(raw)},
                   "top1_differences_vs_q04": diff_q04[v], "top1_differences_off_vs_on": diff_off_on,
                   "q04_submission": {"path": shown(pre["t02"]["submission"]), "sha256": pre["t02"]["submission_sha256"]},
                   "pools": {p: {k: pools[p]["report"].get(k) for k in
                                 ("used06", "fallback_to_test02", "rule", "selected", "sr06", "gate06", "detector",
                                  "top5_on_written", "top5_on_role", "n_questions", "n_candidates")}
                             | {"guard_level": (pools[p]["report"].get("test02") or {}).get("guard_level"),
                                "t06_pool_sha256_lf": sha_lf(pools[p]["paths"]["pool"])} for p in TEST_POOLS},
                   "t6_record_only": {p: {"r_forced_cost": (pools[p]["report"].get("detector") or {}).get("r_forced_cost"),
                                          "changed_top1": (pools[p]["report"].get("detector") or {}).get("changed_top1")}
                                      for p in TEST_POOLS},
                   "data_md_item5_record": {p: {"questions": pools[p]["report"].get("n_questions"),
                                                "candidates": pools[p]["report"].get("n_candidates")}
                                            for p in TEST_POOLS} | {"t2b_failed": [x for x in (t2s or {}).get(
                                                "abd_failed", []) if x.startswith("b:")]},
                   "h6p": pre["h6p"], "seam": {"path": shown(out_paths(ctx, "seam_val")["seam"]),
                                               "sha256_lf": seam_sha, "passed": seam.get("passed")},
                   "t1": pre["t1"], "q04_upload": pre["q04_upload"], "inputs_bound": bound[v],
                   "inputs_bound_rule": "sha256 with CRLF folded to LF for .csv / .json, raw bytes otherwise",
                   "frozen": pre["frozen"], "rerun": rerun, "tools": tool_record(), "env": env_record(),
                   "created": now_iso(), "seconds": round(time.time() - t0, 1)}
    for v, rep in reps.items():
        write_json(ctx, paths[v][1], rep)
    bad = [v for v, ch in checks.items() if not ch["passed"]]
    if bad:
        for v in bad:
            os.remove(guard_write(ctx, paths[v][0]))
        refuse(f"check_submission --official failed for {bad}: {[checks[v]['errors'][:3] for v in bad]}; the "
               f"files were removed (a local bug: stop and check)")
    print("submit: 06-OFF %s (%d rows, sha256 %s); 06-ON %s%s; top-1 vs Q04 %s" % (
        shown(sub_off), len(rows_off), reps["off"]["output"]["sha256"][:12],
        ("%s (sha256 %s)" % (shown(sub_on), reps["on"]["output"]["sha256"][:12])) if sub_on else "not produced",
        "" if sub_on else " (failed: %s)" % [k for k, v in cond.items() if not v["ok"]], diff_q04))
    return 0


def run_in_log(ctx, run_id):
    """Ids of upload-log entries (any status) of this 06 run; [] without a log (dry-run without --log)."""
    path = ctx["LOG"]
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    return [e.get("id") for e in t06_entries(log) if (e.get("test06") or {}).get("run_id") == run_id]


def inputs_bound(ctx, pre, pools, t2_path, with_on):
    files = [pre["t02"]["submission"], pre["t02"]["submit_report"], pre["t02"]["seam_local"],
             os.path.join(ctx["C"], "player", "t1.json")]
    for p in TEST_POOLS:
        tp = pre["t02"]["pools"][p]
        files += [tp["apply_report"], tp["pool_select"], tp["top5"]]
    files += [os.path.join(ctx["D2"], "player", p, "dumps", n) for p in POOLS for n in ("cand_games.npz",
                                                                                       "query_games.npz")]
    if t2_path:
        files.append(t2_path)
    files += [os.path.join(ctx["P6"], n) for n in ("frozen_config.json", "lda_B.npz", "lda_W.npz", "score_H6P.json",
                                                   "score_H6P_switch.json")]
    files += [out_paths(ctx, "seam_val")["seam"]] + [out_paths(ctx, p)["feats_report"] for p in POOLS]
    for p in TEST_POOLS:
        op = out_paths(ctx, p)
        files += [op["pool"], op["off"]] + ([op["on"]] if with_on and pools[p]["report"].get("top5_on_written") else [])
    return {rel_key(f): sha_bound(f) for f in files}


# ================================================================ veto
def t06_entries(log):
    return [e for e in log.get("uploads", []) if e.get("comp") == "player" and e.get("test06")]


def upload_key(e):
    return (parse_iso(e.get("uploaded_at"), "uploaded_at"), e.get("id"))


def paired_off(offs, on):
    """PREREG_06 rev 3: 06-ON is paired with the latest 06-OFF of the same 06 run (run_id) uploaded before it (a 06-OFF
    uploaded again before 06-ON, after a wrong file, is the one that counts; tools/official_log.py pairs the same way);
    None when there is none."""
    t_on = upload_key(on)
    same = [e for e in offs if (e.get("test06") or {}).get("run_id") == on["test06"].get("run_id")
            and upload_key(e) < t_on]
    return same[-1] if same else None


def cmd_veto(ctx, a):
    t0 = time.time()
    given = [k for k, v in (("--confirmed-upload-check", a.confirmed_upload_check), ("--wrong-upload", a.wrong_upload),
                            ("--upload-check-inconclusive", a.upload_check_inconclusive)) if (v or "").strip()]
    if len(given) > 1:
        refuse(f"give at most one of --confirmed-upload-check / --wrong-upload / --upload-check-inconclusive "
               f"(got {given})")
    out = os.path.join(ctx["O6"], "veto.json")
    rerun = guard_existing(ctx, a, [out])
    log_p = ctx["LOG"]
    if not os.path.exists(log_p):
        refuse(f"{shown(log_p)} missing")
    require_committed(ctx, [log_p], "results/official_scores.json")
    with open(log_p, encoding="utf-8") as f:
        log = json.load(f)
    ups = sorted([e for e in t06_entries(log) if e.get("status") == "uploaded"], key=upload_key)
    offs = [e for e in ups if e["test06"].get("variant") == "off"]
    ons = [e for e in ups if e["test06"].get("variant") == "on"]
    if not ons:
        refuse("no uploaded 06-ON entry in the log: nothing to veto (06-OFF is the 06 upload)")
    if len(ons) > 1:
        refuse(f"{len(ons)} uploaded 06-ON entries: PREREG_06 5.5 E1 allows one")
    on = ons[0]
    if not [e for e in offs if upload_key(e) < upload_key(on)]:
        refuse("06-ON was uploaded before any 06-OFF: PREREG_06 5.3 puts 06-OFF first")
    off = paired_off(offs, on)
    if off is None:
        refuse(f"06-OFF and 06-ON come from different 06 runs: no 06-OFF of run_id "
               f"{str(on['test06'].get('run_id'))[:12]} was uploaded before 06-ON #{on['id']}")
    q04 = q04_first_upload(log)
    rec = {"format": FORMATS["veto"], "mode": ctx["mode"], "what": "PREREG_06 5.3 public veto of the one-to-one switch",
           "entries": {"q04": q04 and q04["id"], "off": off["id"], "on": on["id"]},
           "pairing": "06-ON with the latest 06-OFF of the same run_id uploaded before it (PREREG_06 rev 3)",
           "public": {"q04": q04 and q04.get("public_score"), "off": off.get("public_score"),
                      "on": on.get("public_score")},
           "now": ctx["now"].isoformat(), "k7p": K7P.isoformat(), "log_sha256": sha_raw(log_p), "rerun": rerun,
           "confirmed_upload_check": (a.confirmed_upload_check or "").strip() or None, "upload_check": None,
           # PREREG_06 rev 3 item 2: every command but status records the tool SHAs, git HEAD and versions (round-2
           # review 10/3, minor 6)
           "tools": tool_record(), "env": env_record()}
    if ctx["now"] > K7P:
        rec.update(decided=True, paused=False, k7_passed=True, d=None, vetoed=None, final_variant="off",
                   reupload_off_needed=True, outcome="k7_passed",
                   rule="K7' passed before the veto was decided: no switch, final 06-OFF, upload 06-OFF once more "
                        "(PREREG_06 4)")
        write_json(ctx, out, dict(rec, created=now_iso(), seconds=round(time.time() - t0, 1)))
        print("veto: K7' passed: final 06-OFF; upload 06-OFF once more so that it is the last upload")
        return 0
    off_check = [(k, (v or "").strip()) for k, v in (("wrong_upload", a.wrong_upload),
                                                     ("upload_check_inconclusive", a.upload_check_inconclusive))
                 if (v or "").strip()]
    if off_check:
        kind, text = off_check[0]
        rec.update(decided=True, paused=False, k7_passed=False, d=None, vetoed=None, final_variant="off",
                   reupload_off_needed=True, outcome=kind, upload_check={"kind": kind, "text": text},
                   rule="PREREG_06 rev 3 (narrower than 5.3, it only switches the one-to-one off): AIdea's upload "
                        "record shows a wrong file in the 06-OFF / 06-ON pair, or cannot settle it: no switch, final "
                        "06-OFF, upload 06-OFF once more (06-ON cannot be uploaded again, PREREG_06 5.5 E1)")
        write_json(ctx, out, dict(rec, created=now_iso(), seconds=round(time.time() - t0, 1)))
        print(f"veto: {kind} ({text}): no switch, final 06-OFF; upload 06-OFF once more so that it is the last upload")
        return 0
    if off.get("public_score") is None or on.get("public_score") is None:
        refuse("both 06-OFF and 06-ON need their public scores (official_log.py public) before the veto")
    reports = {}
    for v, e in (("off", off), ("on", on)):
        rp = os.path.join(ctx["O6"], v, REPORT)
        rep = read_json(rp)
        if e["test06"].get("report_sha256") not in (None, sha_raw(rp)):
            refuse(f"{shown(rp)} is not the report the log entry #{e['id']} recorded at prepare")
        reports[v] = rep
    diffs = {("q04", "off"): (reports["off"].get("top1_differences_vs_q04") or {}).get("public"),
             ("q04", "on"): (reports["on"].get("top1_differences_vs_q04") or {}).get("public"),
             ("off", "on"): (reports["on"].get("top1_differences_off_vs_on") or {}).get("public")}
    sc = rec["public"]
    warn = [{"pair": list(k), "scores": [sc[k[0]], sc[k[1]]], "public_top1_differences": n}
            for k, n in diffs.items() if sc[k[0]] is not None and sc[k[1]] is not None
            and abs(sc[k[0]] - sc[k[1]]) < SCORE_EQUAL_TOL and (n or 0) > 0]
    rec["same_file_warning"] = warn
    if warn and not rec["confirmed_upload_check"]:
        ids = (q04 and q04["id"], off["id"], on["id"])
        rec.update(decided=False, paused=True, d=None, vetoed=None, final_variant=None, reupload_off_needed=None,
                   outcome="paused",
                   check_items=["請使用者截 AIdea 競賽一的上傳紀錄（檔名、時間），核對 #%s、#%s、#%s 各是哪一份；核對前不重傳、"
                                "不設最終選擇" % ids,
                                "沒有傳錯（分數真的相同）：veto --rerun --reason TEXT --confirmed-upload-check \"<核對了什麼>\"",
                                "06-OFF／06-ON 這一對有傳錯：veto --rerun --reason TEXT --wrong-upload \"<哪一筆其實是哪一份>\""
                                "：不用開關，最終 06-OFF，再上傳一次 06-OFF（絕不用 --confirmed-upload-check）",
                                "核對不出、或 K7' 前來不及：veto --rerun --reason TEXT --upload-check-inconclusive \"<原因>\""
                                "：同上，最終 06-OFF"])
        write_json(ctx, out, dict(rec, created=now_iso(), seconds=round(time.time() - t0, 1)))
        raise T06Error("possible wrong file (three files share the name submission_player.csv): %s; the veto is "
                       "paused; check AIdea's upload record, then veto --rerun with --confirmed-upload-check (no wrong "
                       "file), --wrong-upload or --upload-check-inconclusive (both: final 06-OFF) (PREREG_06 5.3, "
                       "rev 3)" % warn, 2, "same_file_warning")
    d = on["public_score"] - off["public_score"]
    vetoed = d < 0
    rec.update(decided=True, paused=False, k7_passed=False, d=d, vetoed=vetoed,
               final_variant="off" if vetoed else "on", reupload_off_needed=vetoed,
               outcome="vetoed" if vetoed else "kept",
               rule="d = public(06-ON) - public(06-OFF) from results/official_scores.json; d < 0 vetoes (final 06-OFF, "
                    "upload it once more); d >= 0 keeps 06-ON (it must still be the last upload)")
    write_json(ctx, out, dict(rec, created=now_iso(), seconds=round(time.time() - t0, 1)))
    print("veto: d = %.6f - %.6f = %+.6f -> %s" % (on["public_score"], off["public_score"], d,
                                                     "VETOED: final 06-OFF, upload 06-OFF once more" if vetoed else
                                                     "not vetoed: final 06-ON (the last upload)"))
    return 0


# ================================================================ status
def cmd_status(ctx, a):
    now = ctx["now"]
    lines = [f"06 test day ({ctx['mode']})"]
    for pool in POOLS:
        op = out_paths(ctx, pool)
        fr = maybe_json(op["feats_report"])
        lines.append(f"  feats {pool}: {'done' if fr else 'missing'}")
    seam = maybe_json(out_paths(ctx, "seam_val")["seam"])
    lines.append(f"  seam: {seam.get('verdict') if seam else 'missing'}")
    for pool in TEST_POOLS:
        r = maybe_json(out_paths(ctx, pool)["pool"])
        lines.append(f"  apply {pool}: " + ("missing" if not r else
                                            (f"06 {r.get('selected')}" if r.get("used06") else "test02 rows")
                                            + f", top5_on {'yes' if r.get('top5_on_written') else 'no'}"))
    for v in VARIANTS:
        r = maybe_json(os.path.join(ctx["O6"], v, REPORT))
        lines.append(f"  submit {v}: " + (f"{str((r.get('output') or {}).get('sha256'))[:12]}" if r else "not produced"))
    nu = maybe_json(os.path.join(ctx["O6"], "not_used.json"))
    if nu:
        lines.append(f"  06 not used: {[e['reason'] for e in nu.get('events', [])]} -> {MSG_U5}")
    veto = maybe_json(os.path.join(ctx["O6"], "veto.json"))
    on_up = False
    if ctx["LOG"] and os.path.exists(ctx["LOG"]):
        with open(ctx["LOG"], encoding="utf-8") as f:
            log = json.load(f)
        for e in t06_entries(log):
            lines.append(f"  log #{e['id']} 06-{e['test06'].get('variant', '?').upper()} {e['status']} "
                         f"public {e.get('public_score')}")
            on_up |= e["test06"].get("variant") == "on" and e["status"] == "uploaded"
    if veto:
        lines.append("  veto: " + ("paused (same-file warning): 先照 docs/TESTDAY_06.md 第 7 步請使用者截 AIdea 上傳紀錄核對；"
                                   "核對前不重傳、不設最終選擇" if veto.get("paused") else
                                   f"{veto.get('outcome') or ''} d {veto.get('d')}, final 06-"
                                   f"{str(veto.get('final_variant')).upper()}"
                                   + ("; upload 06-OFF once more" if veto.get("reupload_off_needed") else "")))
    if now > K7P and on_up and not (veto and veto.get("decided")):
        lines.append("  K7' 已過、06-ON 已上傳、否決沒判定：最終 06-OFF，再上傳一次 06-OFF（docs/TESTDAY_06.md 第 8 步）")
    lines.append(f"  距 K6' 11/21 23:59（06 的正式執行要在這之前 commit）：{left(now, K6P)}；"
                 f"距 K7' 11/22 23:59（06-ON 上傳、兩份分數、否決都要在這之前）：{left(now, K7P)}")
    print("\n".join(lines))
    return 0


# ================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="t06_player.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dry-run", action="store_true")
        for k in ("--out-dir", "--data-dir", "--t02-results-dir", "--t02-out-dir", "--tests-dir", "--checks-dir",
                  "--log", "--prereg06-dir", "--now"):
            p.add_argument(k, default=None)
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
    p = sub.add_parser("feats")
    common(p)
    p.add_argument("--pool", required=True, choices=POOLS)
    p = sub.add_parser("seam")
    common(p)
    p = sub.add_parser("apply")
    common(p)
    p.add_argument("--pool", required=True, choices=TEST_POOLS)
    p = sub.add_parser("submit")
    common(p)
    p.add_argument("--aidea-page-checked", default=None,
                   help="what the AIdea pages say about the question structure (06-ON needs it; PREREG_06 5.3)")
    p.add_argument("--aidea-not-one-to-one", action="store_true",
                   help="an AIdea page says the pools are not one question per candidate: no 06-ON")
    p = sub.add_parser("veto")
    common(p)
    p.add_argument("--confirmed-upload-check", default=None,
                   help="same-file warning: AIdea's upload record shows no wrong file (what was checked)")
    p.add_argument("--wrong-upload", default=None,
                   help="same-file warning: AIdea's upload record shows a wrong file in the 06-OFF / 06-ON pair -> no "
                        "switch, final 06-OFF (PREREG_06 rev 3)")
    p.add_argument("--upload-check-inconclusive", default=None,
                   help="same-file warning: the upload record cannot settle it (or no time before K7') -> no switch, "
                        "final 06-OFF (PREREG_06 rev 3)")
    p = sub.add_parser("status")
    common(p)
    return ap


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    try:
        a = build_parser().parse_args(argv)
        ctx = resolve(a)
        fn = {"feats": cmd_feats, "seam": cmd_seam, "apply": cmd_apply, "submit": cmd_submit, "veto": cmd_veto,
              "status": cmd_status}[a.cmd]
        return fn(ctx, a)
    except T06Error as e:
        print(f"t06_player: {e} (exit {e.code})", file=sys.stderr)
        return e.code
    except st.Refused as e:
        print(f"t06_player: {e} (exit 2)", file=sys.stderr)
        return 2
    except SystemExit as e:
        if isinstance(e.code, int):
            return e.code
        print(f"t06_player: {e} (exit 2)", file=sys.stderr)
        return 2
    except Exception:     # noqa: BLE001  (an internal error is a 2 with a traceback, never a registered verdict)
        traceback.print_exc()
        print("t06_player: unexpected internal error (exit 2): stop and check the local tool", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
