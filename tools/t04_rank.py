"""C10 test04, competition 2 (rank, TEAM_11248): the 04 test-day local tool (Q02 = R20 + O4 + SWA + SWB).

Registration: docs/PREREG_04_OPPONENT.md section 5 (rev 1) and rev 3 section 11. Spec: results/test04/design/spec.md
sections 2.1, 2.3, 3, 5, 6, 7 (the registration wins over the spec; the spec over the design notes).

    PY tools/t04_rank.py unpack  --zip Z  [MODE] [DIRS] [--allow-failed]
    PY tools/t04_rank.py opp             [MODE] [DIRS] [--workers N] [--rerun --reason TEXT [--bug-revision N]]
    PY tools/t04_rank.py apply           [MODE] [DIRS] [--rerun --reason TEXT [--bug-revision N]]
    PY tools/t04_rank.py status          [MODE] [DIRS]
    PY tools/t04_rank.py archive --reason TEXT [--bug-revision N] [--drive-folder sw|sw_interrupted_K]
                                                                     (real mode; MODE + --results-dir for drills)

MODE  none = real; at most one of --rehearsal, --dry-run, --smoke
DIRS  --results-dir R4      test04 root (reads R4/sw, writes R4/opp, R4/rank, R4/unpack_sw.json)
      --t02-results-dir D2  test02 notebook root (reads D2/rank/<set>/dumps, D2/rank/{config,run_log,seam_check}.json)
      --t02-out-dir O2      test02 apply outputs (reads O2/opening_test.npz, O2/apply_test_report.json)
      --tests-dir T         the test files (normalised copies) + MANIFEST.json; never T/raw
      --checks-dir C        T1 (reads C/rank/t1.json); default D2/checks
Real mode: R4 = results/test04, D2 = results/test02, O2 = results/test02/rank, T = data/test,
C = results/test02/checks exactly. Other modes: R4, D2, O2 and T are required and R4 is outside the repository
(--rehearsal may also use results/test04_rehearsal).

unpack  Extracts the test04 notebook zip (members under sw/) into R4/sw and writes R4/unpack_sw.json, after the
        member checks (no absolute path, '..', backslash, duplicate; everything under sw/; CRC; the required members),
        then, before any record check (review round 1), after an archive: a recomputed zip (other SHA-256 than every
        archived or archived-as-unpacked zip, every run that may have produced output started later (K5 gate round 1:
        a preflight stop of the same folder before the archive does not count), later seam_val markers; else 2) and,
        on the registered rerun, every marker computed on G4 (else 2); then the record checks: a seam_failed /
        readonly_changed / final_check_failed run (last, or before the last in the repository) or a seam_check.json
        that did not pass -> exit 1 (on the registered rerun the message says there is no third run); a run that
        produced no seam result (preflight_failed, interrupted, running) -> 2; every _DONE_ marker from the last run
        (no resumed run) else 1; smoke / rehearsal flags of the mode; the test bundle equal
        to the local MANIFEST and to test02's; the committed colab/test04_rank.ipynb (real: else 1; review round 2:
        the repo notebook must be committed, clean and self-consistent first, else 2); real: every run that may have
        produced output (not a preflight / GPU-gate stop) and every marker after R20's first upload (else 2); the
        Drive sw_failed_* / sw_exempt_* folders the notebook saw against the local archives (recorded, a warning only).
        --allow-failed (inspection outside the repo) turns the seam-class records and the mode flags into warnings.
opp     CPU: replays the three question sets (seam_val = data/local/rank_val_questions.csv, public, private) with
        the registered counts (p04_opp.py functions, imported, never edited) -> R4/opp/o4_<set>.npz and
        opp_report.json. seam_val must be byte-identical to results/prereg04/opp/val_o4.npz (file SHA-256) and its
        O4 dkappa equal to 0.1050265798301427; replay failures > 0.5 % of a set's slots -> exit 3 (no 04: final
        R20); a public / private O4 dkappa <= 0 / NaN / without a sub-question -> exit 3 only while the seam_val
        replay is byte-identical with the registered dkappa (review round 1: else it is recorded only); other seam
        differences, or .venv versions -> exit 4 (local drift).
apply   Q02 per set (seam_val first) from test02's C5 dumps and opening_test.npz, opp's O4 and the notebook's
        swapped logits; the local seam against results/prereg04/rank/rank_val_probs.csv / pred_B.csv
        (t02_rank.seam_compare; tol = max(test02's, test04's)); six dkappa (public / private x O4, SWA, SWB) > 0;
        R4/rank/{rank_test_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json, submission_rank.csv (= pred_B.csv
        byte copy), t04_apply_report.json}; check_submission --official. Exit 1 / 4 write rank/attempt_verdict.json
        instead of the report; exit 3 writes only the report (usable false), never a submission.
status  Read-only summary.
archive Moves a seam-class failed attempt (R4/sw, unpack_sw.json, rank/) to R4/failed/attempt_N/ and every R4/*.zip
        to data/test04_failed/attempt_N/ (other modes: R4/failed_zips/attempt_N/); writes attempt.json (the zips'
        SHA-256, plus the one unpack read when it lay elsewhere). One counted archive at most; never after a
        registered decision (unless --bug-revision N, a committed PREREG_04 rev N). Review round 2: rev N must be
        committed after this attempt's run, markers, unpack_sw.json and attempt_verdict.json, and no earlier archive
        may cite the same N (one revision, one uncounted rerun). Real mode: refuses (2) while R20's first upload
        cannot be checked (committed log) or while colab/test04_rank.ipynb is not committed and self-consistent; only
        a run that really computed its outputs before R20's first upload is before_r20_u1 (a preflight / GPU-gate
        stop of the same folder does not count). Review round 3: it names the Drive rename (attempt.json
        drive_rename): a counted archive -> sw_failed_N, an uncounted one -> sw_exempt_N (the notebook neither counts
        it nor asks G4 after it); --drive-folder sw_interrupted_K when a folder renamed as a disconnection holds a seam
        failure (its zip test04_rank_results_interrupted_K.zip, or its three JSON records put in R4/sw, is the
        evidence).

Exit codes: 0 done; 1 seam-class failure (this notebook run is not used: archive, rerun the notebook once);
2 refused / error (fix the precondition and run the local tool again; an unexpected internal error also gives 2 with
a traceback, never 1: stop and check the local tool); 3 the registered "04 is not used" (final R20, R10 on
FAIL_OPENING; nothing is rerun); 4 local drift (stop and check the local code or .venv; not a Colab rerun).

Team separation: imports rank tools, decide and check_submission only, never a player module. No validation label
file is ever opened (any path whose lower-case text contains that word is refused). No GPU, no tools/score.py.
"""

import argparse
import ast
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SELF = os.path.join(HERE, "t04_rank.py")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import check_submission  # noqa: E402
import decide  # noqa: E402
import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
import p04_opp as po  # noqa: E402
import p04_rank_fuse as pf  # noqa: E402
import t02_rank as t2  # noqa: E402
import t02_test_bundle as tb  # noqa: E402

# ------------------------------------------------------------------ paths (module constants: the unit tests point
# them at temporary trees with setattr; the real run never changes them)
RESULTS = os.path.join(REPO, "results")
REAL_R4 = os.path.join(RESULTS, "test04")
REHEARSAL_R4 = os.path.join(RESULTS, "test04_rehearsal")
REAL_D2 = os.path.join(RESULTS, "test02")
REAL_O2 = os.path.join(REAL_D2, "rank")
REAL_T = os.path.join(REPO, "data", "test")
REAL_C = os.path.join(REAL_D2, "checks")
FAILED_ZIP_ROOT = os.path.join(REPO, "data", "test04_failed")
LOCAL_DIR = os.path.join(REPO, "data", "local")
SEAM_QUESTIONS = os.path.join(LOCAL_DIR, "rank_val_questions.csv")
P03_ROOT = os.path.join(RESULTS, "prereg03")
P04_RANK = os.path.join(RESULTS, "prereg04", "rank")
P04_OPP = os.path.join(RESULTS, "prereg04", "opp")
FROZEN_CONFIG = os.path.join(P04_RANK, "frozen_config.json")
VAL_O4 = os.path.join(P04_OPP, "val_o4.npz")
VAL_REPORT = os.path.join(P04_OPP, "val_report.json")
P04_CHECK_REPORT = os.path.join(P04_RANK, "check_report.json")
OFFICIAL_LOG = os.path.join(RESULTS, "official_scores.json")
NOTEBOOK04 = os.path.join(REPO, "colab", "test04_rank.ipynb")
C0_SOURCE_NB = NOTEBOOK04                     # the notebook whose c0_of / question_spans are recomputed locally
C0_FALLBACK_NB = os.path.join(REPO, "colab", "test02_rank.ipynb")   # outside real mode while test04's is absent
PREREG04_DOC = os.path.join(REPO, "docs", "PREREG_04_OPPONENT.md")
OFFICIAL_LOG_TOOL = os.path.join(HERE, "official_log.py")
R20_SUBMISSION_REL = "results/test02/rank/submission_rank.csv"

# ------------------------------------------------------------------ registered values (spec 3.0-3.3, 5, 6)
FROZEN_CONFIG_SHA = "ae728e9c936a55dfeced255e9b05b5ac6a4955a7101f108e78c53c679e162661"
COUNTS_SHA = "f6c6debe90639585763b9812c235743a6b7f89d8db1c6422ece7bccd550a08a5"
TABLES_SHA = "f380e0d720edd4429e4858ab87a3e1a5efbbd9de84982c002ab214f19f120de5"
P04_OPP_SHA = "b594e0eff135b539616d60f26403ca1a1a37ac3f44855785c36e2cc6be9768fa"
SEAM_QUESTIONS_SHA = "73e51bbfd7b051e73cbb62af4b180c403d2422f122240c5e67e9c92303776db5"
VAL_O4_SHA = "30d312c02c357f6761567099003885be5e08635b6e6c0b49d71baac137cd9cd4"
DKAPPA_O4_SEAM_REF = 0.1050265798301427                  # results/prereg04/rank/apply_report.json val_colour_check
DKAPPA_SW_SEAM_REF = {"SWA": 0.03631175839656169, "SWB": 0.034356617152967346}   # check_report.json (record only)
DKAPPA_TOL = 1e-12
P04_SEAM_PINS = {"rank_val_probs.csv": "3c7ab658dc558ce8c6aefdd7b0747de1a60de575d151e9eee96ca6775f442e5c",
                 "pred_B.csv": "27b6d225d0afc8db9183ae970ee19946ff819ec5ccb2be6a1816ef4338efff79",
                 "pred_A.csv": "6a6bb3c2e63761bc7196661c99999cec499fae1b38bbb3a7bb9d512e26487739"}
P03_SEAM_REF_PINS = {   # results/prereg03/rank_{a,b}/dumps (the notebook's seam references)
    "a": {"val_keys.npz": "055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c",
          "val_p03a_d4.npz": "1e4e5957442596358b087e5f4e69c1fcf9f22f2c7933cc2049066bfab3075ecd",
          "val_p03a_s0_swap.npz": "b355c26bf68069feb0e48ee95c46bd211c2bdbc6b5a48416c8408a210aec3848"},
    "b": {"val_keys.npz": "055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c",
          "val_p03b_d4.npz": "bad2277dd20bf832c02213be781768ba9d8a304922693bee383a28aa3531a627",
          "val_p03b_s0_swap.npz": "0cadff4bae36b2a41f02fddf4366a2081d9912ede4a3063c8b50fd28a2647527"}}
ZLIB_PIN = "1.3.1"   # measured with zlib.ZLIB_RUNTIME_VERSION at design time (spec header; val_report has no zlib)
MAX_RATE = 0.005     # PREREG_04 5.2: more than 0.5 % of a set's slots failed -> no 04 (exactly 0.5 % passes)
COMBO_ID = "Q02"
COMBO_SOURCES = ["C5", "OP", "O4", "SWA", "SWB"]
COMBO_TERMS = ["OP", "O4", "SWA", "SWB"]
C5_DUMP = "p02e9_d4"
SETS = ("seam_val", "public", "private")
TEST_SETS = ("public", "private")
OPP_SOURCES = ("O4", "SWA", "SWB")
ARMS = ("a", "b")
ARM_SOURCE = {"a": "SWA", "b": "SWB"}
SW_ROOT = "sw"
SWAP_VIEW = "s0_colour_swapped"
NB_RECORDS = ("config.json", "run_log.json", "env.json", "timings.json", "seam_check.json", "readonly_before.json",
              "readonly_after.json")
NB_CONFIG_FORMAT = "c10-test04 rank notebook config v1"
NOTEBOOK_REL = "colab/test04_rank.ipynb"
DUMP_ENV_KEYS = ("gpu_name", "torch", "cudnn")
TOL_SAME_GPU, TOL_OTHER_GPU = 1e-4, 1e-3
G4_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"   # rev 3 item 4: the registered rerun runs on G4 only
SEAM_FAILED_STATUSES = t2.FAILED_RUN_STATUSES          # seam_failed, repro_failed, readonly_changed, final_check_failed
NO_RESULT_STATUSES = ("preflight_failed", "interrupted", "running")
COUNTED_CLASSES = ("notebook_run_log", "notebook_seam", "not_committed_notebook", "resumed", "apply_exit_1")
UNCOUNTED_CLASSES = ("before_r20_u1", "bug_revision")
# review round 3: the Drive names archive asks for. A counted archive -> sw_failed_<N> (the notebook counts it toward the
# one rerun and asks G4); an uncounted one -> sw_exempt_<N> (the notebook neither counts it nor asks G4, as unpack and
# official_log: G4 only after a counted archive). --drive-folder: the folder to rename (sw, or sw_interrupted_<K> when a
# folder renamed as a disconnection turns out to hold a seam failure)
DRIVE_COUNTED_PREFIX, DRIVE_EXEMPT_PREFIX = "sw_failed_", "sw_exempt_"
DRIVE_FOLDER_RE = re.compile(r"sw|sw_interrupted_[1-9][0-9]*")
ZIP_EXT = (".json", ".npz", ".csv", ".txt", ".log")
TW = dt.timezone(dt.timedelta(hours=8))
K6 = dt.datetime(2026, 11, 16, 23, 59, 59, tzinfo=TW)
WORKERS_DEFAULT, WORKERS_MAX = 6, 8
MIN_BUG_REVISION = 4
FORMATS = {"unpack": "c10-test04 unpack v1", "opp": "c10-test04 opp v1", "apply": "c10-test04 rank apply v1",
           "seam_local": "c10-test04 rank seam_local v1", "verdict": "c10-test04 attempt verdict v1",
           "attempt": "c10-test04 archive attempt v1"}
VERDICT_STAGES = ("prerequisites", "notebook_seam_recompute", "local_seam", "o4_consistency", "check_submission")

MSG_SEAM = "這次的 04 輸出不用：照 docs/TESTDAY_04.md「重跑一次」先 archive"
# K5 gate round 1: a seam-class failure of the registered rerun (one counted archive already) has no third run; archive
# would refuse ("登記只准重跑一次"), so the message points at 'rerun once' item 6 instead of 'archive first'
MSG_SEAM_LAST = ("這是登記的重跑（已有 1 個占次數的封存），沒有第三次：這次的 04 輸出不用、不 archive，照 docs/TESTDAY_04.md"
                 "「重跑一次」第 6 點停下查清楚，修訂前不上傳 04；只有先 commit 指出程式錯誤的 PREREG_04 修訂（rev 4 起）"
                 "時，才 archive --bug-revision N")
MSG_K6 = "K6 已過：這一季不用 04，最終 R20"
MSG_FAIL_OPENING = ("T1 是 FAIL_OPENING：04 不上場（PREREG_04 5.3），最終是 R10，照 docs/TESTDAY_04.md 第 1 步寫 "
                    "RESULTS")
MSG_FAIL_STRUCT = "T1 是 FAIL_STRUCT：照 test02 停下"
MSG_NO_R20_U1 = "先照 docs/TESTDAY.md 做完 R20 的第一次上傳"
MSG_BEFORE_R20 = ("5.3：R20 上傳前跑的 test04 不用；照「重跑一次」的做法 archive 後重跑，這次不算登記的那一次")
MSG_NOT_FRESH = ("這不是重算的結果（Drive 上可能還是舊的 zip）：刪掉這個 zip，請使用者等新的 zip 出現再下載")
MSG_ONE_RERUN = "登記只准重跑一次：停下查清楚；要再跑只能先寫 PREREG_04 修訂"
MSG_RERUN_G4 = ("登記的重跑只在 G4（PREREG_04 rev 3 第 4、9 條）：這次的完成標記不是 G4 算的，輸出不用、不 archive；停下查清楚"
                "（已 commit 的 notebook 會在 preflight 擋下不在 G4 的重跑）")
MSG_R20_UNCHECKED = ("archive 要先核對 R20 的第一次上傳，才分得出這次是不是在 R20 上傳前跑的（PREREG_04 rev 3 第 1、9 條）：先 commit "
                     "results/official_scores.json（R20 的第一次上傳記成 uploaded）和 results/test02/rank/"
                     "apply_test_report.json，再 archive")
MSG_VENV = "還原 .venv 後重跑（11/24 前不升級 .venv；PREREG_04 rev 3 第 18 條）"
# rev 3 check: the counts file is restored, not recomputed by a new opp (after opp's registered decision a new opp needs a
# PREREG_04 revision, so 'rerun opp' led nowhere)
MSG_COUNTS_RESTORE = ("還原 data/prereg04/counts_fit.npz（SHA-256 %s；同 SHA 的 data/prereg04/design_scratch/counts_fit.npz，"
                      "PREREG_04 rev 3 第 17 條、docs/TESTDAY_04.md 第 1 步），再跑同一個指令" % COUNTS_SHA[:12])
MSG_OPP_DRIFT_ARCHIVE = ("opp_report.json 是本機漂移（結束碼 4），不是登記的判定：先查好本機、opp --rerun --reason TEXT 重跑 opp，"
                         "再 archive（PREREG_04 rev 3 第 9 條；修訂不解除這一條）")


class T04Error(RuntimeError):
    """A stop with its exit code (spec 3.0): 1 seam-class, 2 refused, 3 registered 'no 04', 4 local drift."""

    def __init__(self, message, code=2, stage=None, reason=None, details=None):
        super().__init__(message)
        self.code, self.stage, self.reason, self.details = code, stage, reason, details


def _errors():
    out = []
    for e in (t2.T02Error, getattr(rf, "P02Error", None), p3.P03Error, po.P04Error, pf.P04Error, decide.DecideError):
        if e is not None and e not in out:
            out.append(e)
    return tuple(out)


FOREIGN_ERRORS = _errors()


def refuse(msg):
    raise T04Error(msg, 2)


def seam_fail(msg, stage="prerequisites", reason=None, details=None):
    raise T04Error(msg, 1, stage, reason, details)


def seam_msg(ctx, first=MSG_SEAM):
    """The next step of a seam-class failure (exit 1). K5 gate round 1: with a counted archive already
    (ctx['counted_archives'], set by unpack and apply before their record checks) this is the registered rerun and
    archive would refuse, so MSG_SEAM_LAST ('rerun once' item 6); else `first` (archive, then the one rerun)."""
    return MSG_SEAM_LAST if ctx.get("counted_archives") else first


def drift(msg, stage="prerequisites", reason=None, details=None):
    raise T04Error(msg, 4, stage, reason, details)


# ================================================================ small helpers
def now():
    """The current time (a module function: the unit tests replace it to cross K6)."""
    return dt.datetime.now(TW)


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_iso(text, what):
    try:
        t = dt.datetime.fromisoformat(str(text).strip())
    except ValueError:
        refuse(f"{what} {text!r} is not an ISO time")
    if t.tzinfo is None:
        refuse(f"{what} {text!r} has no time zone (the notebook and the upload log write aware ISO times)")
    return t


_LABEL = "ans" + "wer"


def refuse_label_path(path):
    """Refuse any path whose lower-case text contains the validation label file word (spec 3.0)."""
    if path is not None and _LABEL in os.path.abspath(os.fspath(path)).lower():
        refuse(f"refusing {path}: validation label files are never opened by the test04 tools")
    return path


def sha_raw(path):
    return t2.sha256_file(refuse_label_path(path))


def sha_lf(path):
    return t2.sha256_text(refuse_label_path(path))


def pin_sha(path):
    """.csv / .json with CRLF folded to LF, anything else (.npz) raw bytes (t02_rank.pin_sha256)."""
    return t2.pin_sha256(refuse_label_path(path))


def shown(path):
    return t2.shown(path)


def read_json(path):
    refuse_label_path(path)
    if not os.path.exists(path):
        refuse(f"missing {shown(path)}")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except ValueError as e:
        refuse(f"{shown(path)} is not valid JSON ({e})")


def read_json_or_none(path):
    return read_json(path) if os.path.exists(path) else None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


def forbidden_area(path):
    """Where t04 never writes (spec 3.0): results/prereg0*, results/test02*, data/local, results/score_ledger.csv."""
    p = os.path.abspath(path)
    if t2.is_inside(p, LOCAL_DIR):
        return "data/local"
    if t2.same_path(p, os.path.join(RESULTS, "score_ledger.csv")):
        return "results/score_ledger.csv"
    if t2.is_inside(p, RESULTS) and not t2.same_path(p, RESULTS):
        first = os.path.relpath(os.path.realpath(p), os.path.realpath(RESULTS)).split(os.sep)[0].lower()
        if first.startswith(("prereg0", "test02")):
            return "results/" + first
    return None


def guard_write(ctx, path):
    p = os.path.abspath(path)
    refuse_label_path(p)
    bad = forbidden_area(p)
    if bad:
        refuse(f"refusing to write {p}: it is inside the read-only {bad}")
    if not any(t2.is_inside(p, r) for r in ctx["write_roots"]):
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


def make_tmp():
    d = tempfile.mkdtemp(prefix="c10_t04_")
    if t2.is_inside(d, REPO):
        shutil.rmtree(d, ignore_errors=True)
        refuse(f"the temporary folder {d} is inside the repository; set TEMP outside it")
    return d


def git(*args):
    """(returncode, stdout stripped) of git in the repository, stdout decoded as UTF-8; (None, message) when git cannot
    run. Review round 3: t02_rank.git (text=True) decodes with the locale (cp950 on this machine), so a Chinese file
    such as docs/PREREG_04_OPPONENT.md came back as None and `--bug-revision N` crashed (exit 2, internal error) on
    every real run; official_log.git_head_text reads bytes the same way."""
    try:
        r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    return r.returncode, (r.stdout or b"").decode("utf-8", "replace").strip()


def git_head():
    return t2.git_head()


def require_committed(paths, what):
    """Every path is tracked by git and unchanged against HEAD (real mode; a module function the tests replace)."""
    try:
        return t2.require_committed_clean([os.path.abspath(p) for p in paths], what)
    except t2.T02Error as e:
        refuse(str(e))


def git_show_head(rel):
    """Text of rel at HEAD (UTF-8, see git), or None."""
    rc, out = git("show", f"HEAD:{rel}")
    return out if rc == 0 and isinstance(out, str) else None


def git_first_commit_with(rel, needle):
    """(commit, committer time ISO) of the first commit whose diff adds or removes `needle` in rel (git log -S)."""
    rc, out = git("log", "--reverse", "-S", needle, "--format=%H %cI", "--", rel)
    if rc != 0 or not out:
        return None, None
    first = out.splitlines()[0].split()
    return (first[0], first[1]) if len(first) == 2 else (None, None)


def tool_files():
    names = ["t04_rank.py", "check_submission.py", "official_log.py", "t02_rank.py", "t02_test_bundle.py",
             "p04_opp.py", "p04_rank_fuse.py", "p03_rank_fuse.py", "p02_rank_fuse.py", "p02_rank_opening.py",
             "p02_rank_pseudo.py", "p02_moves.py", "c10data.py", "decide.py"]
    return {n: (sha_lf(os.path.join(HERE, n)) if os.path.exists(os.path.join(HERE, n)) else None) for n in names}


def tool_record():
    return {"git_head": git_head(), "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": tool_files()}


def venv_versions():
    """The versions the byte-identical O4 depends on (a module function: the unit tests replace it)."""
    import scipy
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "zlib": zlib.ZLIB_RUNTIME_VERSION}


def environment():
    return dict(po.environment(), zlib=zlib.ZLIB_RUNTIME_VERSION)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def rate_ok(failures, slots):
    """failures / slots <= 0.005, in integers (200 * failures <= slots): 25 of 5,000 passes, 26 does not."""
    return isinstance(slots, int) and slots > 0 and isinstance(failures, int) and 0 <= failures and \
        200 * failures <= slots


def dkappa_ok(kv):
    v = (kv or {}).get("dkappa")
    return is_num(v) and v > 0 and ((kv or {}).get("sub_questions") or 0) > 0


# ================================================================ modes and directories
def resolve(a, cmd):
    flags = [m for m, on in (("dry-run", a.dry_run), ("smoke", a.smoke), ("rehearsal", a.rehearsal)) if on]
    if len(flags) > 1:
        refuse(f"give at most one of --dry-run, --smoke, --rehearsal (got {flags})")
    mode = flags[0] if flags else "real"
    given = {"--results-dir": a.results_dir, "--t02-results-dir": a.t02_results_dir,
             "--t02-out-dir": a.t02_out_dir, "--tests-dir": a.tests_dir}
    real = {"--results-dir": REAL_R4, "--t02-results-dir": REAL_D2, "--t02-out-dir": REAL_O2,
            "--tests-dir": REAL_T}
    if mode == "real":
        for k, v in given.items():
            if v and not t2.same_path(v, real[k]):
                refuse(f"real mode: {k} must be {real[k]} (got {v}); other trees need --dry-run, --smoke or "
                       f"--rehearsal")
        if a.checks_dir and not t2.same_path(a.checks_dir, REAL_C):
            refuse(f"real mode: --checks-dir must be {REAL_C} (got {a.checks_dir})")
        R4, D2, O2, T, C = REAL_R4, REAL_D2, REAL_O2, REAL_T, REAL_C
    else:
        need = ("--results-dir",) if cmd in ("archive", "status") else tuple(given)
        missing = [k for k in need if not given[k]]
        if missing:
            refuse(f"--{mode} needs {', '.join(missing)} (other modes never fall back to the repository paths)")
        R4 = os.path.abspath(a.results_dir)
        D2 = os.path.abspath(a.t02_results_dir) if a.t02_results_dir else None
        O2 = os.path.abspath(a.t02_out_dir) if a.t02_out_dir else None
        T = os.path.abspath(a.tests_dir) if a.tests_dir else None
        C = os.path.abspath(a.checks_dir) if a.checks_dir else (os.path.join(D2, "checks") if D2 else None)
        if t2.is_inside(R4, REPO) and not (mode == "rehearsal" and t2.same_path(R4, REHEARSAL_R4)):
            refuse(f"--{mode}: --results-dir must be outside the repository (only --rehearsal may use "
                   f"{REHEARSAL_R4}); got {R4}")
    for p in (R4, D2, O2, T, C):
        if p is not None:
            refuse_label_path(p)
    bad = forbidden_area(R4)
    if bad:
        refuse(f"--results-dir {R4} is inside the read-only {bad}")
    if T is not None and (t2.is_inside(T, LOCAL_DIR) or os.path.basename(os.path.normpath(T)).lower() == "raw"):
        refuse(f"--tests-dir {T}: the test files are the checked copies (data/test), never data/local or data/test/raw")
    rehearsal_in_repo = mode == "rehearsal" and t2.same_path(R4, REHEARSAL_R4)
    zips = FAILED_ZIP_ROOT if mode == "real" else os.path.join(R4, "failed_zips")
    return {"mode": mode, "real": mode == "real", "smoke": mode == "smoke", "in_repo": mode == "real" or
            rehearsal_in_repo, "R4": R4, "D2": D2, "O2": O2, "T": T, "C": C,
            "sw": os.path.join(R4, SW_ROOT), "opp": os.path.join(R4, "opp"), "rank": os.path.join(R4, "rank"),
            "unpack": os.path.join(R4, "unpack_sw.json"), "failed": os.path.join(R4, "failed"),
            "failed_zips": zips, "write_roots": [R4] + ([zips] if mode == "real" else [])}


def check_k6(ctx):
    if ctx["real"] and now() > K6:
        raise T04Error(MSG_K6 + f"（K6 {K6.isoformat()}；現在 {now().isoformat()}）", 3, reason="k6")


def paths_record(ctx):
    return {k: ctx[k] for k in ("R4", "D2", "O2", "T", "C")}


def question_files(ctx):
    return {"seam_val": SEAM_QUESTIONS, "public": os.path.join(ctx["T"], t2.TEST_FILES["public"]),
            "private": os.path.join(ctx["T"], t2.TEST_FILES["private"])}


# ================================================================ common prerequisites (spec 3.0)
def load_frozen_config(real):
    """Item 2: results/prereg04/rank/frozen_config.json pinned (LF SHA-256), real mode committed and passing
    po.require_frozen_config(); the pf.cmd_apply checks (:1712-1734) of the frozen parameters."""
    h = pin_sha(FROZEN_CONFIG)
    if h != FROZEN_CONFIG_SHA:
        refuse(f"{shown(FROZEN_CONFIG)}: sha256 {h} is not the registered {FROZEN_CONFIG_SHA}")
    rec = {"path": shown(FROZEN_CONFIG), "sha256": h}
    if real:
        try:
            rec["commit_check"] = po.require_frozen_config()
        except po.P04Error as e:
            refuse(str(e))
    else:
        rec["commit_check"] = "not checked (non-real mode)"
    fc = read_json(FROZEN_CONFIG)
    if fc.get("format") != pf.FROZEN_FORMAT:
        refuse(f"frozen_config.json format {fc.get('format')!r}")
    flags = {k: fc.get(k) for k in ("smoke", "preview", "design_input")}
    if any(v is not False for v in flags.values()):
        refuse(f"frozen_config.json is not a real frozen config {flags}")
    combo = fc.get("combo") or {}
    if combo.get("id") != COMBO_ID or combo.get("terms") != COMBO_TERMS or combo.get("sources") != COMBO_SOURCES \
            or combo.get("terms") != list(pf.TERMS) or combo.get("path") != "gen":
        refuse(f"frozen combination {combo} is not the adopted Q02 (terms {list(pf.TERMS)})")
    Bf = np.asarray(fc["basis"], dtype=np.float64)
    if Bf.shape != (rf.N_LEVELS, rf.N_LEVELS - 1) or not np.array_equal(Bf, pf.B):
        refuse("frozen basis differs from p04_rank_fuse.B")
    par = fc["params"]
    theta = np.asarray(par["theta"], dtype=np.float64)
    if len(theta) != 2 + len(COMBO_TERMS) + rf.N_LEVELS - 1:
        refuse(f"frozen theta has {len(theta)} numbers, expected 15 [a, gamma, w_OP, w_O4, w_SWA, w_SWB, beta1..9]")
    diff = float(np.max(np.abs(pf.B @ np.asarray(par["beta"], dtype=np.float64) - np.asarray(par["b"]))))
    if not diff <= t2.PARAM_TOL:
        refuse(f"frozen parameters: b is not B @ beta (largest difference {diff:.3g})")
    if list(theta[2 + len(COMBO_TERMS):]) != par["beta"]:
        refuse("frozen parameters: theta[6:] differs from beta")
    return fc, rec


def check_frozen_tools(fc):
    """Item 3: the frozen tools equal HEAD (po.tool_state, pf.frozen_tools_check), hash to frozen_config's
    tool.files_sha256, are clean; p04_opp.py also equals val_report.json's record."""
    try:
        ts = po.tool_state()
        ft = pf.frozen_tools_check()
        pts = pf.tool_state()
    except (po.P04Error, pf.P04Error, p3.P03Error) as e:
        refuse(str(e))
    want = (fc.get("tool") or {}).get("files_sha256") or {}
    got = dict(pts["files_sha256"])
    changed = sorted(f for f in set(want) | set(got) if want.get(f) != got.get(f))
    if changed:
        refuse(f"frozen tool files differ from frozen_config.json tool.files_sha256: {changed}")
    if pts["dirty_or_untracked"] is None or pts["dirty_or_untracked"] or ts["dirty_or_untracked"]:
        refuse(f"frozen tools are not clean in git: {pts['dirty_or_untracked']} {ts['dirty_or_untracked']}")
    vr = read_json(VAL_REPORT)
    vr_opp = ((vr.get("tool") or {}).get("files_sha256") or {}).get("p04_opp.py")
    if got.get("p04_opp.py") != vr_opp or vr_opp != P04_OPP_SHA:
        refuse(f"tools/p04_opp.py {str(got.get('p04_opp.py'))[:12]} is not the one of val_report.json "
               f"({str(vr_opp)[:12]}; registered {P04_OPP_SHA[:12]})")
    return {"files_sha256": got, "equal_to_frozen_config": True, "p04_opp_equal_to_val_report": True,
            "frozen_equal_to_head": ft.get("rule")}


def t1_first(ctx):
    """Item 4: C/rank/t1.json read first. FAIL_OPENING -> 3, FAIL_STRUCT -> 2."""
    path = os.path.join(ctx["C"], "rank", "t1.json")
    if not os.path.exists(path):
        if ctx["real"]:
            refuse(f"{path} is missing: T1 comes first (docs/TESTDAY.md)")
        return {"path": shown(path), "verdict": None, "note": f"absent ({ctx['mode']})"}
    v = read_json(path).get("verdict")
    if v == "FAIL_OPENING":
        raise T04Error(MSG_FAIL_OPENING, 3, reason="t1_fail_opening")
    if v == "FAIL_STRUCT":
        refuse(MSG_FAIL_STRUCT)
    return {"path": shown(path), "sha256": pin_sha(path), "verdict": v}


def t2_namespace(ctx):
    return argparse.Namespace(dry_run=ctx["mode"] == "dry-run", smoke=ctx["mode"] == "smoke",
                              rehearsal=ctx["mode"] == "rehearsal", results_dir=ctx["D2"], tests_dir=ctx["T"],
                              out_dir=ctx["O2"], checks_dir=ctx["C"], rerun=False, reason=None, fallback=None,
                              fallback_file=None)


def check_test02(ctx):
    """Item 5: test02's competition 2 is done (t2.common_checks, t2.check_dump_env, the committed real apply report
    without the fallback and with a passed local seam, the committed opening_test.npz)."""
    ctx2 = t2.resolve(t2_namespace(ctx))
    fc2, cfg2, seam2, qfiles2, wsha2, rec2 = t2.common_checks(ctx2, "apply", False)
    env2 = t2.check_dump_env(ctx2)
    rep_path = os.path.join(ctx["O2"], "apply_test_report.json")
    op_path = os.path.join(ctx["O2"], "opening_test.npz")
    if ctx["real"]:
        rec2["t02_outputs_commit_check"] = require_committed([rep_path, op_path],
                                                             "test02's apply_test_report.json and opening_test.npz")
    rep = read_json(rep_path)
    sl, cs = rep.get("seam_local") or {}, rep.get("check_submission") or {}
    problems = []
    if rep.get("mode") != ctx["mode"]:
        problems.append(f"mode {rep.get('mode')!r} (this run is {ctx['mode']})")
    if rep.get("fallback_used") is not False:
        problems.append(f"fallback_used {rep.get('fallback_used')!r} (04 needs the R20 run, not the R10 fallback)")
    if not ctx["smoke"]:
        if sl.get("passed") is not True:
            problems.append("test02's local seam did not pass")
        if cs.get("mode") != "official" or cs.get("errors"):
            problems.append(f"check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
        if not is_num(sl.get("tol")):
            problems.append(f"seam_local.tol {sl.get('tol')!r}")
    if problems:
        refuse(f"{shown(rep_path)}: " + "; ".join(problems))
    qfiles = question_files(ctx)
    op = t2.load_opening_test(op_path, qfiles)
    cpath = os.path.join(ctx["D2"], "rank", "config.json")
    return {"ctx2": ctx2, "cfg": cfg2, "seam": seam2, "rec": rec2, "env": env2, "report": rep,
            "report_path": rep_path, "report_sha256": pin_sha(rep_path), "opening": op, "opening_path": op_path,
            "qfiles": qfiles, "c5_weights": wsha2, "config_path": cpath, "config_sha256": pin_sha(cpath),
            "t1": rec2.get("t1")}


def r20_first_upload(ctx, t02_report):
    """Item 6 (real): results/official_scores.json (committed) has R20's first upload: comp rank, status uploaded,
    file results/test02/rank/submission_rank.csv, sha256 = test02's report outputs_sha256; the earliest one."""
    require_committed([OFFICIAL_LOG], "results/official_scores.json (R20's first upload is logged first)")
    log = read_json(OFFICIAL_LOG)
    want = (t02_report.get("outputs_sha256") or {}).get("submission_rank.csv")
    ups = [e for e in log.get("uploads") or [] if e.get("comp") == "rank" and e.get("status") == "uploaded"
           and e.get("file") == R20_SUBMISSION_REL and want and e.get("sha256") == want and e.get("uploaded_at")]
    if not ups:
        refuse(MSG_NO_R20_U1 + f"（results/official_scores.json 沒有 {R20_SUBMISSION_REL} sha256 {str(want)[:12]} 的 "
                               f"uploaded 紀錄）")
    first = min(ups, key=lambda e: (parse_iso(e["uploaded_at"], "uploaded_at"), e.get("id", 0)))
    return {"id": first.get("id"), "uploaded_at": first["uploaded_at"], "sha256": first.get("sha256"),
            "log_sha256": pin_sha(OFFICIAL_LOG)}


def check_venv():
    """Item 7: Python, numpy, scipy equal val_report.json's environment, zlib equals ZLIB_PIN; else exit 4."""
    env = read_json(VAL_REPORT).get("environment") or {}
    want = {"python": env.get("python"), "numpy": env.get("numpy"), "scipy": env.get("scipy"), "zlib": ZLIB_PIN}
    got = venv_versions()
    bad = {k: {"now": got.get(k), "registered": want[k]} for k in want if got.get(k) != want[k]}
    if bad:
        drift(f".venv 版本和 results/prereg04/opp/val_report.json 記的不同 {bad}：{MSG_VENV}", "prerequisites",
              "venv_versions", bad)
    return {"equal": True, "versions": got, "rule": "python / numpy / scipy = val_report.json environment; zlib = "
                                                    f"{ZLIB_PIN} (design-time measurement)"}


def common_prereqs(ctx, cmd):
    """Spec 3.0 items 1-7 (unpack: item 1 only). -> (fc, t02, record)."""
    rec = {}
    if ctx["real"]:
        rec["own_files"] = require_committed([SELF, os.path.abspath(check_submission.__file__), OFFICIAL_LOG_TOOL],
                                             "tools/t04_rank.py, tools/check_submission.py and tools/official_log.py")
    else:
        rec["own_files"] = f"not checked ({ctx['mode']})"
    if cmd == "unpack":
        if ctx["real"]:   # review round 2: 'not the committed notebook' (1) is judged against the committed file only
            rec["notebook_file"] = require_committed([NOTEBOOK04], NOTEBOOK_REL)
        return None, None, rec
    fc, rec["frozen_config"] = load_frozen_config(ctx["real"])
    rec["frozen_tools"] = check_frozen_tools(fc)
    rec["t1_first"] = t1_first(ctx)
    t02 = check_test02(ctx)
    rec["test02"] = t02["rec"]
    rec["test02_dump_env"] = t02["env"]
    if ctx["real"]:
        t02["r20"] = r20_first_upload(ctx, t02["report"])
    else:
        t02["r20"] = {"id": None, "note": f"not checked ({ctx['mode']})"}
    rec["r20_first_upload"] = t02["r20"]
    rec["environment_check"] = check_venv()
    return fc, t02, rec


# ================================================================ notebook records, attempts
def set_members():
    out = ["val_keys.npz", "_DONE_val_keys.json", "_INPUT.json"]
    for x in ARMS:
        for stem in (f"val_p03{x}_d4", f"val_p03{x}_s0_swap"):
            out += [f"{stem}.npz", f"_DONE_{stem}.json"]
    return out


def required_members():
    return [f"{SW_ROOT}/{n}" for n in NB_RECORDS] + [f"{SW_ROOT}/{s}/dumps/{f}" for s in SETS for f in set_members()]


def marker_stems():
    return ["val_keys"] + [f"val_p03{x}_{v}" for x in ARMS for v in ("d4", "s0_swap")]


def flags_problem(cfg, mode):
    smoke, reh = cfg.get("smoke"), cfg.get("rehearsal")
    want = {"real": (False, False), "dry-run": (False, True), "rehearsal": (False, True), "smoke": (True, None)}[mode]
    if smoke is not want[0] or (want[1] is not None and reh is not want[1]):
        return (f"config.json says smoke={smoke!r}, rehearsal={reh!r}; {mode} mode needs smoke={want[0]}"
                + ("" if want[1] is None else f", rehearsal={want[1]}"))
    return None


def notebook_identity(cfg, mode):
    """(same, record): is the run's notebook_sha256 the one of colab/test04_rank.ipynb (t2.notebook_source_sha256)?
    Review round 2: the callers of real mode require the file committed and clean first (require_committed), and a
    repo notebook whose own NOTEBOOK_SHA256 line does not match its recomputed value is a local problem: real mode
    refuses (2), never a Colab-side 'not the committed notebook' (1, a counted rerun); other modes record it."""
    if not os.path.exists(NOTEBOOK04):
        if mode == "real":
            refuse(f"{NOTEBOOK04} is missing: real mode checks the Colab run against the committed notebook")
        return None, {"repo_notebook": None, "recorded": cfg.get("notebook_sha256"), "note": "repo notebook absent"}
    recorded, recomputed = t2.notebook_source_sha256(NOTEBOOK04)
    if recomputed is None or recorded != recomputed:
        msg = (f"{NOTEBOOK_REL} is not self-consistent (its NOTEBOOK_SHA256 line {str(recorded)[:12]}, recomputed "
               f"{str(recomputed)[:12]}): a local problem, not a Colab one; restore the committed notebook (git "
               f"status / git diff) and run the local tool again")
        if mode == "real":
            refuse(msg)
        return None, {"repo_notebook": NOTEBOOK_REL, "repo_recorded": recorded, "repo_recomputed": recomputed,
                      "run_notebook_sha256": cfg.get("notebook_sha256"), "same": None, "note": msg}
    same = cfg.get("notebook_sha256") == recomputed
    return same, {"repo_notebook": NOTEBOOK_REL, "repo_recorded": recorded, "repo_recomputed": recomputed,
                  "run_notebook_sha256": cfg.get("notebook_sha256"), "same": same}


def done_markers_from_dir(sw):
    out = {}
    for s in SETS:
        for stem in marker_stems():
            p = os.path.join(sw, s, "dumps", f"_DONE_{stem}.json")
            if os.path.exists(p):
                out[f"{s}/dumps/_DONE_{stem}.json"] = read_json(p)
    return out


def run_history(run_log):
    return tb.run_history(run_log)


def last_run_number(run_log):
    h = run_history(run_log)
    return h[-1][0] if h else None


def run_starts(run_log):
    runs = run_log.get("runs") if isinstance(run_log, dict) else None
    return [{"run": r.get("run", i), "status": r.get("status"), "start": r.get("start")}
            for i, r in enumerate(runs if isinstance(runs, list) else [], start=1) if isinstance(r, dict)]


def resumed_markers(markers, run_log):
    last = last_run_number(run_log)
    return sorted(n for n, m in markers.items() if (m or {}).get("run") != last)


def list_attempts(ctx):
    out = []
    for p in glob.glob(os.path.join(ctx["failed"], "attempt_*")):
        m = re.fullmatch(r"attempt_(\d+)", os.path.basename(p))
        if not m or not os.path.isdir(p):
            continue
        ap = os.path.join(p, "attempt.json")
        out.append({"name": os.path.basename(p), "n": int(m.group(1)), "path": p,
                    "record": read_json(ap) if os.path.exists(ap) else None})
    return sorted(out, key=lambda x: x["n"])


def is_counted(att):
    """An archive counts toward the one registered rerun unless its attempt.json says otherwise (a missing record
    counts: the safe side)."""
    rec = att.get("record")
    return rec is None or rec.get("counts_toward_rerun") is not False


def freshness(ctx, zsha, run_log, markers, attempts):
    """After an archive: the zip is not an archived one, every run that may have produced its outputs started after
    the latest archive, and every seam_val marker is later than the one the latest archive recorded.
    -> (problems, record).
    K5 gate round 1: the runs compared are producing_runs (rev 3 items 1 and 9, the same definition as R20-first),
    no longer run_log.runs[0]: after docs/TESTDAY_04.md 'sw_interrupted_K was a seam failure' the registered rerun
    runs in the same Drive sw folder, whose run 1 is the preflight stop made BEFORE the archive; runs[0] refused
    every such rerun as 'not recomputed'. A preflight / GPU-gate / version stop produced nothing (the zip SHA-256, the
    last run's start and the seam_val marker times still catch an old zip)."""
    rec = {"archives": [a["name"] for a in attempts], "zip_sha256": zsha}
    if not attempts:
        rec["rule"] = "no archive yet: nothing to compare"
        return [], rec
    problems = []
    archived = {}
    for a in attempts:
        for name, h in ((a["record"] or {}).get("zips") or {}).items():
            archived[h] = f"{a['name']}/{name}"
    rec["zip_sha256_archived_as"] = archived.get(zsha)
    if zsha in archived:
        problems.append(f"the zip is the archived {archived[zsha]}")
    dated = [a for a in attempts if (a["record"] or {}).get("archived_at")]
    latest = max(dated, key=lambda a: parse_iso(a["record"]["archived_at"], "archived_at")) if dated else None
    starts = run_starts(run_log)
    runs = producing_runs(run_log)
    rec["runs0_start"] = starts[0]["start"] if starts else None      # record only (the rule before K5 gate round 1)
    rec["producing_runs"] = runs
    rec["rule"] = ("every producing run (the last run and every earlier run not preflight_failed / interrupted / "
                   "running) starts after the latest archive; zip SHA-256 not archived; seam_val markers later")
    if latest:
        at = latest["record"]["archived_at"]
        rec["latest_archive"] = {"name": latest["name"], "archived_at": at}
        if not runs:
            problems.append(f"run_log.json has no run to compare with the archive {latest['name']} ({at})")
        for r in runs:
            start = r["start"]
            if not start or parse_iso(start, f"run_log run {r['run']} start") <= parse_iso(at, "archived_at"):
                problems.append(f"run_log run {r['run']} ({r['status']}) start {start} is not after the archive "
                                f"{latest['name']} ({at})")
        done = (latest["record"] or {}).get("done_times") or {}
        cmp = {}
        for name, m in markers.items():
            if not name.startswith("seam_val/"):
                continue
            base = os.path.basename(name)
            old = done.get(base) or done.get(name)
            if old is None:
                continue
            new = (m or {}).get("time")
            later = bool(new) and parse_iso(new, f"{name} time") > parse_iso(old, "archived done time")
            cmp[base] = {"now": new, "archived": old, "later": later}
            if not later:
                problems.append(f"seam_val marker {base} time {new} is not later than the archived {old}")
        rec["seam_val_done_times"] = cmp
    return problems, rec


def rerun_gpu_problems(markers):
    """Rev 3 items 4 and 9 (review round 1): the registered rerun is computed on G4 only; every _DONE_ marker of the
    zip names G4_GPU. -> [marker names that do not]."""
    return sorted(n for n, m in markers.items() if (m or {}).get("gpu_name") != G4_GPU)


def producing_runs(run_log):
    """The runs of a Drive folder that may have produced output (review round 2, rev 3 item 1): the last run, and every
    earlier run whose status is not a no-result one (preflight_failed, interrupted, running: a GPU-gate or preflight
    stop, or a run cut off before it recorded anything, is not a run; TESTDAY_04 / COLAB_RUN reuse the same folder
    after such a stop)."""
    starts = run_starts(run_log)
    if not starts:
        return []
    return [r for r in starts[:-1] if r["status"] not in NO_RESULT_STATUSES] + [starts[-1]]


def r20_before_problems(r20, run_log, markers):
    """PREREG_04 5.3 / rev 3 item 1: R20's first upload precedes the run that computed the outputs (producing_runs:
    review round 2, it used to be run_log.runs[0], possibly a preflight stop) and every marker."""
    up = parse_iso(r20["uploaded_at"], "R20 first upload uploaded_at")
    probs = []
    runs = producing_runs(run_log)
    if not runs:
        probs.append(f"run_log.json has no run to compare with R20's first upload {r20['uploaded_at']}")
    for r in runs:
        start = r["start"]
        if not start or not up < parse_iso(start, f"run_log run {r['run']} start"):
            probs.append(f"run_log run {r['run']} ({r['status']}) start {start} is not after R20's first upload "
                         f"{r20['uploaded_at']}")
    for name, m in sorted(markers.items()):
        t = (m or {}).get("time")
        if not t or not up < parse_iso(t, f"{name} time"):
            probs.append(f"{name} time {t} is not after R20's first upload {r20['uploaded_at']}")
    return probs


def drive_archive_names(n, counts):
    """Review round 3: (Drive folder, zip) names archive asks for attempt n: counted -> sw_failed_<n> /
    test04_rank_results_failed_<n>.zip; uncounted (before_r20_u1, bug_revision) -> sw_exempt_<n> /
    test04_rank_results_exempt_<n>.zip."""
    tag = "failed" if counts else "exempt"
    return f"sw_{tag}_{n}", f"test04_rank_results_{tag}_{n}.zip"


def drive_source_names(folder):
    """Review round 3: (Drive folder, the zip next to it) that archive --drive-folder names: sw ->
    test04_rank_results.zip; sw_interrupted_<K> (renamed as a disconnection, its records show a seam failure) ->
    test04_rank_results_interrupted_<K>.zip."""
    folder = "sw" if folder is None else folder
    if not DRIVE_FOLDER_RE.fullmatch(folder):
        refuse(f"--drive-folder {folder!r}: sw or sw_interrupted_<K> (the Drive folder under C10/results/test04 to rename)")
    if folder == "sw":
        return folder, "test04_rank_results.zip"
    return folder, f"test04_rank_results_interrupted_{folder[len('sw_interrupted_'):]}.zip"


def drive_archives_record(config, attempts):
    """Review round 2 (record only): the archived folders the notebook saw on Drive (config.json prior_sw_folders:
    sw_failed_* and, review round 3, sw_exempt_*) against the local archives (archive asks for sw_failed_<N> when the
    archive counts, sw_exempt_<N> when it does not). A difference is a warning, never a refusal: a naming slip on Drive
    must not cost a valid run. Review round 3: the notebook counts every sw_failed_* folder (also an apply-exit-1
    archive, whose Drive folder passed its own seam) and no sw_exempt_* folder, so with the names right its `attempt`
    equals the local attempt number. -> (record, warning or None)."""
    prior = config.get("prior_sw_folders")
    rec = {"rule": "record only: Drive sw_failed_* / sw_exempt_* folders (config.json prior_sw_folders) vs the local "
                   "archives (counted -> sw_failed_<N>, uncounted -> sw_exempt_<N>)",
           "notebook_attempt": config.get("attempt")}
    if not isinstance(prior, dict):
        rec.update(checked=False, note="config.json has no prior_sw_folders")
        return rec, None
    drive = sorted(str(n) for n in prior if str(n).startswith((DRIVE_COUNTED_PREFIX, DRIVE_EXEMPT_PREFIX)))
    local = sorted(drive_archive_names(x["n"], is_counted(x))[0] for x in attempts)
    rec.update(checked=True, drive_sw_failed=drive, local_archives_as_drive_names=local, equal=drive == local)
    if drive == local:
        return rec, None
    return rec, (f"Drive's sw_failed_* / sw_exempt_* folders {drive} (config.json prior_sw_folders) differ from the "
                 f"local archives {local}: check the Drive folder names with the user before the next step (recorded "
                 f"only)")


def t02_report_for_r20(ctx):
    path = os.path.join(ctx["O2"], "apply_test_report.json")
    if ctx["real"]:
        require_committed([path], "test02's apply_test_report.json")
    return read_json(path)


# ================================================================ unpack
def cmd_unpack(a):
    t0 = time.time()
    ctx = resolve(a, "unpack")
    check_k6(ctx)
    if a.allow_failed and (ctx["in_repo"] or ctx["mode"] != "dry-run"):
        refuse("--allow-failed is for inspection only: --dry-run with a --results-dir outside the repository")
    _fc, _t02, pre = common_prereqs(ctx, "unpack")
    zpath = refuse_label_path(os.path.abspath(a.zip))
    if not os.path.exists(zpath) or not zipfile.is_zipfile(zpath):
        refuse(f"{zpath} is not a zip file")
    zsha = sha_raw(zpath)
    warnings = []

    def fail1(msg, reason):
        if a.allow_failed:
            warnings.append(msg)
            print("WARNING: " + msg)
        else:
            seam_fail(msg + "\n" + seam_msg(ctx), "unpack", reason)

    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        bad = []
        for n in names:
            if n.endswith("/"):
                if n.rstrip("/") != SW_ROOT and tb._member_ok(n.rstrip("/"), SW_ROOT):
                    bad.append(tb._member_ok(n.rstrip("/"), SW_ROOT))
                continue
            p = tb._member_ok(n, SW_ROOT)
            if p is None and not n.lower().endswith(ZIP_EXT):
                p = f"member {n!r}: file type not allowed ({', '.join(ZIP_EXT)})"
            if p:
                bad.append(p)
        if bad:
            refuse(f"{zpath}: {bad[:5]}")
        dup = sorted(n for n in set(names) if names.count(n) > 1)
        if dup:
            refuse(f"{zpath}: repeated members {dup[:5]}")
        files = [n for n in names if not n.endswith("/")]
        for need in (f"{SW_ROOT}/run_log.json", f"{SW_ROOT}/config.json", f"{SW_ROOT}/seam_check.json"):
            if need not in files:
                refuse(f"{zpath}: missing {need}")
        try:   # review round 1: a damaged member (CRC) is a refusal (2), never Python's default exit 1
            damaged = z.testzip()
        except (zipfile.BadZipFile, zlib.error, EOFError, OSError, ValueError) as e:
            damaged = f"? ({e})"
        if damaged:
            refuse(f"{zpath}: damaged member {damaged!r} (CRC or compression error): download the zip again")

        def zjson(name):
            try:
                return json.loads(z.read(name).decode("utf-8"))
            except (ValueError, zipfile.BadZipFile, zlib.error, EOFError) as e:
                refuse(f"{zpath}: {name} is not readable JSON ({e})")

        run_log, config, seam = (zjson(f"{SW_ROOT}/{n}") for n in ("run_log.json", "config.json", "seam_check.json"))
        markers = {n[len(SW_ROOT) + 1:]: zjson(n) for n in files if os.path.basename(n).startswith("_DONE_")}
        status = tb.run_status(run_log)
        history = run_history(run_log)
        print(f"run_log.json runs: {', '.join('%s %s' % (n, s) for n, s in history) or 'none'} (status {status!r})")
        # review round 1: after an archive, the zip must be a recomputed one BEFORE any record check, so that the
        # archived seam-failed zip downloaded again gives 2 ("not a recomputed result"), not 1 (spec 8.3; critique O2)
        attempts = list_attempts(ctx)
        fprobs, frec = freshness(ctx, zsha, run_log, markers, attempts)
        if fprobs:
            refuse(MSG_NOT_FRESH + "\n - " + "\n - ".join(fprobs[:6]))
        counted = [x["name"] for x in attempts if is_counted(x)]
        ctx["counted_archives"] = counted      # K5 gate round 1: the exit-1 message of the registered rerun (seam_msg)
        if len(counted) >= 2:
            refuse(f"{len(counted)} counted archives in {ctx['failed']} ({counted}): " + MSG_ONE_RERUN)
        g4_rec = {"rule": "the registered rerun (one counted archive) is computed on G4 only (rev 3 items 4, 9)",
                  "applies": len(counted) == 1 and not ctx["smoke"], "gpu": G4_GPU}
        if g4_rec["applies"]:
            off = rerun_gpu_problems(markers)
            g4_rec["markers_not_on_g4"] = off
            if off:
                refuse(MSG_RERUN_G4 + "\n - " + "\n - ".join(
                    f"{n}: {(markers[n] or {}).get('gpu_name')!r}" for n in off[:6]))
        if len(counted) == 1:
            print("這是登記的重跑（第 2 次、也是最後一次）")
        drive_rec, drive_warn = drive_archives_record(config, attempts)
        if drive_warn:
            warnings.append(drive_warn)
            print("WARNING: " + drive_warn)
        earlier = tb.earlier_failures(run_log)
        # record checks, in the spec's order (3.1)
        if status in SEAM_FAILED_STATUSES:
            fail1(f"run_log status {status!r}: the notebook's seam / read-only / final check did not pass", "run_log")
        if earlier:
            msg = (f"run_log.json has earlier failed runs {earlier} before the last run: the notebook was run again in "
                   f"the same Drive folder, which is not the registered rerun")
            if ctx["in_repo"]:
                fail1(msg, "run_log_earlier")
            else:
                warnings.append(msg)
                print("WARNING: " + msg)
        if status not in SEAM_FAILED_STATUSES and status != "completed":
            msg = (f"run_log status {status!r}: that run produced no seam result; rename the Drive folder sw to "
                   f"sw_interrupted_N (docs/COLAB_RUN.md) and run the whole notebook again (no resume)")
            if a.allow_failed:
                warnings.append(msg)
            else:
                refuse(msg)
        if status == "completed" and seam.get("passed") is not True:
            fail1(f"seam_check.json passed={seam.get('passed')!r}: the notebook seam did not pass", "notebook_seam")
        resumed = resumed_markers(markers, run_log)
        if resumed:
            fail1(f"markers not from the last run {last_run_number(run_log)!r} (a resumed run): {resumed[:5]}",
                  "resumed")
        if config.get("format") != NB_CONFIG_FORMAT:
            refuse(f"config.json format {config.get('format')!r}; not a {NB_CONFIG_FORMAT} run")
        fp = flags_problem(config, ctx["mode"])
        if fp and a.allow_failed:   # review round 1: inspecting a real test-day zip with --dry-run --allow-failed
            warnings.append(fp)
            print("WARNING: " + fp)
        elif fp:
            refuse(fp)
        try:
            man, man_sha = tb.verify_manifest(ctx["T"], names=tb.COMP_FILES["rank"])
        except tb.BundleError as e:
            refuse(str(e))
        tbc = config.get("test_bundle") or {}
        if tbc.get("manifest_sha256") != man_sha:
            refuse(f"config.json test_bundle.manifest_sha256 {str(tbc.get('manifest_sha256'))[:12]} differs from the "
                   f"local {os.path.join(ctx['T'], 'MANIFEST.json')} ({man_sha[:12]})")
        for n in tb.COMP_FILES["rank"]:
            if (tbc.get("files") or {}).get(n) != man["files"][n]["sha256"]:
                refuse(f"config.json test_bundle.files[{n}] differs from the local MANIFEST")
        c2p = os.path.join(ctx["D2"], "rank", "config.json")
        if ctx["real"]:
            require_committed([c2p], "test02's rank config.json")
        tb2 = read_json(c2p).get("test_bundle") or {}
        same_bundle = tbc.get("manifest_sha256") == tb2.get("manifest_sha256") and all(
            (tbc.get("files") or {}).get(n) == (tb2.get("files") or {}).get(n) for n in tb.COMP_FILES["rank"])
        if not same_bundle:
            refuse("config.json test_bundle differs from test02's rank config.json: not the test bundle of 11/04")
        same_nb, nb_rec = notebook_identity(config, ctx["mode"])
        if same_nb is False:
            msg = (f"the run's notebook_sha256 {str(config.get('notebook_sha256'))[:12]} is not the one of the "
                   f"committed {NOTEBOOK_REL} ({str(nb_rec.get('repo_recomputed'))[:12]})")
            if ctx["real"]:
                fail1(msg, "not_committed_notebook")
            elif ctx["mode"] == "rehearsal":
                warnings.append(msg)
                print("WARNING: " + msg)
        r20 = None
        if ctx["real"]:
            r20 = r20_first_upload(ctx, t02_report_for_r20(ctx))
            probs = r20_before_problems(r20, run_log, markers)
            if probs:
                refuse(MSG_BEFORE_R20 + "\n - " + "\n - ".join(probs[:6]))
        missing = [m for m in required_members() if m not in files]
        if missing:
            refuse(f"{zpath}: missing required members {missing[:6]}" + (f" (+{len(missing) - 6})"
                                                                          if len(missing) > 6 else ""))
        targets = [os.path.join(ctx["R4"], *n.split("/")) for n in files]
        existing = [t for t in targets + [ctx["unpack"]] if os.path.exists(t)]
        if existing:
            refuse(f"target files already exist, nothing extracted: {[shown(t) for t in existing[:4]]}"
                   f"{' (+%d more)' % (len(existing) - 4) if len(existing) > 4 else ''}; to start over, archive first")
        for target in targets:
            guard_write(ctx, target)
        done = []
        try:
            for n, target in zip(files, targets):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(n) as src, open(target + ".tmp", "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                os.replace(target + ".tmp", target)
                done.append(target)
        except BaseException:   # review round 1: never leave a half-extracted sw/ behind
            leftovers = list(done)
            if len(done) < len(targets):
                leftovers.append(targets[len(done)] + ".tmp")
            for t in leftovers:
                if os.path.exists(t):
                    os.remove(t)
            raise
    rec = {"format": FORMATS["unpack"], "mode": ctx["mode"], "created": now_iso(), "seconds": round(time.time() - t0, 1),
           "zip": {"path": zpath, "name": os.path.basename(zpath), "sha256": zsha, "bytes": os.path.getsize(zpath)},
           "members": len(files), "run_status": status, "runs": run_starts(run_log),
           "earlier_failed_runs": [{"run": n, "status": s} for n, s in earlier],
           "smoke": config.get("smoke"), "rehearsal": config.get("rehearsal"), "manifest_sha256": man_sha,
           "notebook_sha256": config.get("notebook_sha256"), "notebook_identity": nb_rec,
           "r20_first_upload": ({"id": r20["id"], "uploaded_at": r20["uploaded_at"]} if r20 else None),
           "attempt_number": len(counted) + 1, "counted_archives": counted,
           "failed_attempts_seen": [x["name"] for x in attempts], "freshness": frec, "rerun_gpu_check": g4_rec,
           "drive_archives": drive_rec,
           "allow_failed": bool(a.allow_failed), "warnings": warnings, "checks": pre, "tool": tool_record(),
           "environment": environment(), "git_head": git_head()}
    write_json(ctx, ctx["unpack"], rec)
    for r in rec["runs"]:
        print(f"  run {r['run']}: {r['status']} (start {r['start']})")
    print(f"unpack: {len(files)} files from {zpath} into {ctx['sw']} (status {status}, attempt "
          f"{rec['attempt_number']}); record {ctx['unpack']}")
    print("下一步：commit results/test04/sw 和 unpack_sw.json，再跑 apply" if ctx["real"] else
          f"next: t04_rank.py opp / apply --{ctx['mode']} with the same directories")
    return 0


# ================================================================ opp
def opp_outputs(ctx):
    return [os.path.join(ctx["opp"], f"o4_{s}.npz") for s in SETS] + [os.path.join(ctx["opp"], "opp_report.json")]


def load_tables():
    """The registered counts (po.registered_counts: committed count_report.json, counts SHA-256) and the two rate
    tables. -> (tabs, info, counts record, tables SHA-256). A module function: the unit tests replace it.
    rev 3 check: a missing or different counts file is a refusal (2) that says how to restore it (rev 3 item 17)."""
    if not os.path.exists(po.COUNTS_REAL):
        refuse("data/prereg04/counts_fit.npz is missing: " + MSG_COUNTS_RESTORE)
    try:
        cpath, crec = po.registered_counts(None, None)
    except po.P04Error as e:
        refuse(f"{e}\n" + MSG_COUNTS_RESTORE)
    if crec.get("sha256") != COUNTS_SHA:
        drift(f"counts {str(crec.get('sha256'))[:12]} are not the registered {COUNTS_SHA[:12]}", "prerequisites",
              "counts")
    counts = po.load_counts(cpath)
    tabs, info = po.build_tables(counts)
    del counts
    return tabs, info, crec, po.tables_record(tabs)["tables_sha256"]


def counts_sha_now():
    """SHA-256 of data/prereg04/counts_fit.npz as it is now (apply re-checks opp's record; a module function)."""
    return sha_raw(po.COUNTS_REAL) if os.path.exists(po.COUNTS_REAL) else None


def o4_scores(slots, tabs, workers, tmp_root):
    """po.val_scores on a list of slots (a module function: the unit tests replace it)."""
    return po.val_scores(slots, tabs, workers=workers, tmp_root=tmp_root)


def o4_arrays(vk, res, lo, hi):
    """cmd_val's arrays (order and dtypes) for rows lo..hi of a val_scores result."""
    return {"question_id": np.array(vk["question_id"]), "slot": np.array(vk["slot"], dtype=np.int64),
            "color": np.array(vk["color"]),
            **{k: np.ascontiguousarray(res[k][lo:hi]) for k in ("o4", "t4", "n_moves", "anomalies", "moves_parsed",
                                                                 "replay_failed")}}


def o4_kappa(vk, z):
    """PREREG_04 rev 3 items 3 and 7: O4 against T4 on the slots C5 parsed and the replay did not fail."""
    mask = np.asarray(vk["parsed"], dtype=bool) & ~np.asarray(z["replay_failed"], dtype=bool)
    lp_o4 = rf.log_softmax(np.where(mask[:, None], np.asarray(z["o4"]), 0.0))
    lp_t4 = rf.log_softmax(np.where(mask[:, None], np.asarray(z["t4"]), 0.0))
    return kappa_record(dict(vk, parsed=mask), lp_t4, lp_o4), mask, lp_o4, lp_t4


def kappa_record(vk, lp_ref, lp_src):
    """pf.val_side_kappa (the registered dkappa) plus record-only fields: questions without a sub-question and the
    mean per question size k."""
    kv = dict(pf.val_side_kappa(vk, lp_ref, lp_src))
    sub = p3.val_colour_subquestions(vk)
    kv["questions"] = len(vk["groups"])
    kv["questions_without_sub_question"] = len(vk["groups"]) - (kv.get("questions_with_a_sub_question") or 0)
    per_k = {}
    if sub:
        groups = [g for _, _, g in sub]
        dk = p3.kappa_questions(lp_ref, groups) - p3.kappa_questions(lp_src, groups)
        ks = np.array([len(vk["groups"][q][1]) for q, _, _ in sub])
        for k in sorted(set(ks.tolist())):
            mk = ks == k
            per_k[str(k)] = {"dkappa": float(np.mean(dk[mk])), "sub_questions": int(mk.sum())}
    kv["per_k_record_only"] = per_k
    return kv


def opp_decision(sets_stats, seam_file_equal, seam_dk_equal, dk_early, smoke=False):
    """Spec 3.2 table. sets_stats {set: {slots, failures}}; dk_early {public/private: kappa record}.
    smoke: the seam checks and the O4 dkappa signs are recorded only (smoke weights, 12 questions per set; the
    replay rates still count). Review round 1 (rev 3 item 7): a public / private O4 dkappa <= 0 is the registered
    'no 04' (3) only while the seam_val replay is byte-identical and its dkappa equals the registered value; with a
    seam_val drift the O4 behind that dkappa is not trusted, so it is recorded only and the drift (4) decides; replay
    rates > 0.5 % stay 3. -> (exit code, passed, usable_so_far, registered reasons, drift reasons)."""
    reg, dr = [], []
    for s in SETS:
        st = sets_stats[s]
        if not rate_ok(st["failures"], st["slots"]):
            reg.append(f"replay:{s}: {st['failures']} of {st['slots']} slots failed (> {MAX_RATE:.1%})")
    seam_trusted = bool(seam_file_equal and seam_dk_equal)
    for s in TEST_SETS:
        if not dkappa_ok(dk_early.get(s)) and not smoke and seam_trusted:
            reg.append(f"dkappa:{s}.O4: {(dk_early.get(s) or {}).get('dkappa')!r} with "
                       f"{(dk_early.get(s) or {}).get('sub_questions')!r} sub-questions (needs > 0)")
    if not seam_file_equal and not smoke:
        dr.append("seam_file: o4_seam_val.npz is not byte-identical to results/prereg04/opp/val_o4.npz")
    if not seam_dk_equal and not smoke:
        dr.append(f"seam_dkappa: seam_val O4 dkappa differs from {DKAPPA_O4_SEAM_REF!r}")
    rates_ok = not any(r.startswith("replay:") for r in reg)
    passed = (smoke or (seam_file_equal and seam_dk_equal)) and rates_ok
    usable = passed and (smoke or all(dkappa_ok(dk_early.get(s)) for s in TEST_SETS))
    code = 3 if reg else (4 if dr else 0)
    return code, bool(passed), bool(usable), reg, dr


def verdict_kind(rep):
    """'registered' (exit 0 or 3: a decision of PREREG_04 5.2) or 'drift' (exit 4)."""
    code = (rep or {}).get("exit_code")
    if code == 4:
        return "drift"
    if code in (0, 3):
        return "registered"
    if (rep or {}).get("usable_so_far") is True:
        return "registered"
    reasons = (rep or {}).get("unusable_reasons") or []
    return "drift" if reasons and all(str(r).startswith(("seam_file", "seam_dkappa")) for r in reasons) \
        else "registered"


def check_bug_revision(n, overturned):
    """Spec 3.0 'registration revision check': N >= 4; docs/PREREG_04_OPPONENT.md committed and clean; HEAD's file has
    a line starting '| rev N |'; the first commit adding it is later than every overturned record time.
    overturned: [(what, ISO time)]. -> record."""
    if n is None:
        return None
    if not isinstance(n, int) or n < MIN_BUG_REVISION:
        refuse(f"--bug-revision {n!r}: a correction is PREREG_04 rev {MIN_BUG_REVISION} or later")
    rel = "docs/PREREG_04_OPPONENT.md"
    require_committed([PREREG04_DOC], rel)
    text = git_show_head(rel)
    needle = f"| rev {n} |"
    line = next((ln for ln in (text or "").splitlines() if ln.startswith(needle)), None)
    if line is None:
        refuse(f"{rel} at HEAD has no line starting {needle!r}: commit the revision (rev 3 item 20) first")
    h, t = git_first_commit_with(rel, needle)
    if t is None:
        refuse(f"no commit adds {needle!r} to {rel}")
    when = parse_iso(t, "revision commit time")
    for what, old in overturned:
        if old is None or not when > parse_iso(old, what):
            refuse(f"PREREG_04 rev {n} was committed at {t}, not after {what} ({old}): a revision overturns only "
                   f"records made before it")
    return {"n": n, "line": line, "commit": h, "committed_at": t,
            "overturns": [{"what": w, "time": o} for w, o in overturned]}


def bug_revision_number(v):
    """N of a recorded bug revision (check_bug_revision's record {'n': N, ...}); None when absent."""
    if isinstance(v, dict):
        v = v.get("n")
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def attempt_times(run_log, markers, unpack_rec, verdict):
    """[(what, ISO time)] of the attempt being archived (review round 2, rev 3 item 20): the start of every run that may
    have produced output (producing_runs), every _DONE_ marker's time, unpack_sw.json 'created' and
    attempt_verdict.json 'created'. A --bug-revision must be committed after all of them."""
    out = [(f"run_log run {r['run']} start", r["start"]) for r in producing_runs(run_log or {}) if r.get("start")]
    out += [(f"{name} time", (m or {}).get("time")) for name, m in sorted((markers or {}).items())
            if (m or {}).get("time")]
    for what, rec in (("unpack_sw.json created", unpack_rec), ("attempt_verdict.json created", verdict)):
        if (rec or {}).get("created"):
            out.append((what, rec["created"]))
    return out


def rerun_record(a, replaced, bug):
    return {"rerun": bool(a.rerun), "reason": a.reason if a.rerun else None, "replaced_sha256": replaced,
            "bug_revision": bug}


def remove_old(ctx, paths):
    replaced = {}
    for p in paths:
        if os.path.exists(p):
            guard_write(ctx, p)
            replaced[shown(p)] = sha_raw(p)
            os.remove(p)
    return replaced


def opp_rerun_gate(ctx, a):
    outs = opp_outputs(ctx)
    existing = [p for p in outs if os.path.exists(p)]
    if not existing:
        return None
    if not a.rerun:
        refuse(f"outputs already exist ({[os.path.basename(p) for p in existing]}); they are never overwritten without "
               f"--rerun --reason TEXT")
    rep = read_json_or_none(os.path.join(ctx["opp"], "opp_report.json"))
    if rep is not None and verdict_kind(rep) == "registered":
        if a.bug_revision is None:
            refuse("opp_report.json is a registered decision (exit 0 or 3): the CPU replay is deterministic, so a "
                   "rerun needs a PREREG_04 revision naming a bug first (--bug-revision N; rev 3 item 20)")
        return check_bug_revision(a.bug_revision, [("opp_report.json created", rep.get("created"))])
    return check_bug_revision(a.bug_revision, [("opp_report.json created", (rep or {}).get("created"))]) \
        if a.bug_revision is not None else None


def cmd_opp(a):
    t0 = time.time()
    ctx = resolve(a, "opp")
    check_k6(ctx)
    w = a.workers if a.workers is not None else WORKERS_DEFAULT
    if not 1 <= w <= WORKERS_MAX:
        refuse(f"--workers {w}: 1-{WORKERS_MAX} (this machine is shared; 6 by default)")
    bug = opp_rerun_gate(ctx, a)
    fc, t02, pre = common_prereqs(ctx, "opp")
    qfiles = t02["qfiles"]
    qsha = {s: sha_raw(qfiles[s]) for s in SETS}
    vr = read_json(VAL_REPORT)
    if qsha["seam_val"] != SEAM_QUESTIONS_SHA or qsha["seam_val"] != (vr.get("questions_file") or {}).get("sha256"):
        refuse(f"{shown(qfiles['seam_val'])}: sha256 {qsha['seam_val'][:12]} is not the registered "
               f"{SEAM_QUESTIONS_SHA[:12]}")
    tabs, tinfo, crec, tsha = load_tables()
    want_tab = ((fc.get("source_sha256") or {}).get("O4") or {}).get("tables_sha256")
    if tsha != want_tab or tsha != TABLES_SHA:
        drift(f"rate tables sha256 {tsha[:12]} differ from frozen_config.json {str(want_tab)[:12]}", "prerequisites",
              "tables")
    vk, slots = {}, {}
    for s in SETS:
        vk[s] = rf.load_val_keys(os.path.join(ctx["D2"], "rank", s, "dumps"))
        sl = po.read_question_slots(qfiles[s])
        if ctx["smoke"]:   # the smoke notebooks keep the first questions of every set (apply checks the same)
            sl = sl[:vk[s]["n"]]
        po.slots_match_keys(sl, vk[s])
        slots[s] = sl
    os.makedirs(ctx["opp"], exist_ok=True)
    replaced = remove_old(ctx, opp_outputs(ctx))
    allslots, bounds = [], {}
    for s in SETS:
        bounds[s] = (len(allslots), len(allslots) + len(slots[s]))
        allslots += slots[s]
    print(f"opp: {', '.join(f'{s} {len(slots[s])}' for s in SETS)} slots, {w} workers ({time.time() - t0:.0f}s)",
          flush=True)
    tmp = make_tmp()
    try:
        res = o4_scores(allslots, tabs, w, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    del tabs
    out, stats, dk_early = {}, {}, {}
    for s in SETS:
        lo, hi = bounds[s]
        arrays = o4_arrays(vk[s], res, lo, hi)
        path = guard_write(ctx, os.path.join(ctx["opp"], f"o4_{s}.npz"))
        po.save_npz(path, arrays)
        out[s] = arrays
        rfl, nm = arrays["replay_failed"], arrays["n_moves"]
        qwf = sum(1 for _, rows in vk[s]["groups"] if rfl[rows].any())
        stats[s] = {"questions": shown(qfiles[s]), "questions_sha256": qsha[s], "questions_count": len(vk[s]["groups"]),
                    "slots": int(len(rfl)), "failures": int(rfl.sum()),
                    "rate": float(rfl.sum()) / len(rfl) if len(rfl) else None, "max_rate": MAX_RATE,
                    "rate_ok": rate_ok(int(rfl.sum()), int(len(rfl))),
                    "anomalies": int((arrays["anomalies"] > 0).sum()),
                    "slots_without_moves": int((arrays["moves_parsed"] == 0).sum()),
                    "slots_without_opponent_moves": int((nm[:, 1].sum(1) == 0).sum()),
                    "questions_with_failed_slots": int(qwf),
                    "questions_with_failed_slots_share_record_only": qwf / len(vk[s]["groups"]) if vk[s]["groups"]
                    else None,
                    "moves": {"target": int(nm[:, 0].sum()), "opponent": int(nm[:, 1].sum())},
                    "output": f"o4_{s}.npz", "output_sha256": sha_raw(path)}
    # seam_val: file SHA-256 (the registered check), array by array (diagnosis only)
    seam_sha = stats["seam_val"]["output_sha256"]
    ref_sha = sha_raw(VAL_O4)
    arrays_equal = {}
    with np.load(VAL_O4, allow_pickle=False) as ref:
        for k, v in out["seam_val"].items():
            r = ref[k] if k in ref.files else None
            arrays_equal[k] = r is not None and r.dtype == v.dtype and r.shape == v.shape and r.tobytes() == v.tobytes()
    file_equal = seam_sha == VAL_O4_SHA and ref_sha == VAL_O4_SHA
    kv_seam, *_ = o4_kappa(vk["seam_val"], out["seam_val"])
    dk_seam = kv_seam.get("dkappa")
    dk_equal = is_num(dk_seam) and abs(dk_seam - DKAPPA_O4_SEAM_REF) <= DKAPPA_TOL
    for s in TEST_SETS:
        dk_early[s], *_ = o4_kappa(vk[s], out[s])
    code, passed, usable, reg, dr = opp_decision(stats, file_equal, dk_equal, dk_early, ctx["smoke"])
    not_judged = [f"{s}.O4: {(dk_early.get(s) or {}).get('dkappa')!r}" for s in TEST_SETS
                  if not dkappa_ok(dk_early.get(s))] if (code == 4 and not ctx["smoke"]) else []
    rep = {"format": FORMATS["opp"], "mode": ctx["mode"], "created": now_iso(), "seconds": None, "workers": w,
           "exit_code": code, "passed": passed, "usable_so_far": usable, "unusable_reasons": reg + dr,
           "problems": reg + dr, "verdict_kind": "drift" if code == 4 else "registered",
           "o4_dkappa_not_judged_while_drift": not_judged,
           "frozen_config": {"path": shown(FROZEN_CONFIG), "sha256": pin_sha(FROZEN_CONFIG), "combo_id": COMBO_ID},
           "counts": {"sha256": crec.get("sha256"), "count_report_sha256": crec.get("count_report_sha256"),
                      "file": crec.get("file")},
           "tables_sha256": tsha, "tables": tinfo,
           "p04_opp_sha256": sha_lf(os.path.join(HERE, "p04_opp.py")),
           "equal_to_val_report_tool": sha_lf(os.path.join(HERE, "p04_opp.py")) == P04_OPP_SHA,
           "test02": {"apply_test_report_sha256": t02["report_sha256"], "t1_verdict": (t02["t1"] or {}).get("verdict"),
                      "r20_first_upload_id": t02["r20"].get("id"),
                      "keys_sha256": {s: vk[s]["sha256"] for s in SETS}},
           "sets": stats,
           "seam_val": {"file_sha256_equal": file_equal, "file_sha256": seam_sha,
                        "reference": "results/prereg04/opp/val_o4.npz", "reference_sha256": VAL_O4_SHA,
                        "reference_file_sha256_now": ref_sha, "arrays_equal": arrays_equal,
                        "dkappa_O4": dk_seam, "dkappa_O4_reference": DKAPPA_O4_SEAM_REF, "dkappa_equal": dk_equal,
                        "dkappa_O4_record": kv_seam, "smoke_record_only": ctx["smoke"]},
           "o4_dkappa_early": dict(dk_early, note="the registered decision is apply's (PREREG_04 rev 3 11.7); apply "
                                                  "must reproduce these bits"),
           "definition": {"replay_failed": "p04_opp.val_scores: (moves_parsed == 0) | (anomalies > 0)",
                          "rate": "failures / slots of the set's question file; > 0.005 -> no 04 (PREREG_04 5.2)",
                          "o4_mask": "parsed (test02 val_keys) & ~replay_failed"},
           "rerun": rerun_record(a, replaced, bug), "checks": pre, "paths": paths_record(ctx),
           "tool": tool_record(), "environment": environment(), "git_head": git_head()}
    rep["seconds"] = round(time.time() - t0, 1)
    write_json(ctx, os.path.join(ctx["opp"], "opp_report.json"), rep)
    for s in SETS:
        st = stats[s]
        print(f"  {s}: {st['failures']} of {st['slots']} slots failed ({st['rate']:.4%}), output sha256 "
              f"{st['output_sha256'][:12]}")
    print(f"  seam_val O4 file byte-identical to val_o4.npz: {file_equal}; seam_val O4 dkappa {dk_seam!r} "
          f"(registered {DKAPPA_O4_SEAM_REF!r}) equal: {dk_equal}")
    print("  O4 dkappa early: " + ", ".join(f"{s} {(dk_early[s] or {}).get('dkappa')!r}" for s in TEST_SETS))
    rel = "results/test04/opp/" if ctx["real"] else ctx["opp"]
    if code == 0:
        print(f"opp: passed, 04 may be usable -> commit {rel}，請使用者跑 test04 notebook ({rep['seconds']}s)")
    elif code == 3:
        print(f"opp: 04 不用（PREREG_04 5.2）：{'; '.join(reg)}\ncommit {rel}；不開 Colab，最終 R20，寫 RESULTS",
              file=sys.stderr)
    else:
        print(f"opp: 本機漂移：{'; '.join(dr)}\n停下查本機程式或 .venv，不請使用者做任何事；查好後 opp --rerun --reason "
              f"TEXT" + (f"\n（seam_val 漂移時 O4 Δκ 不判定，只記錄：{'; '.join(not_judged)}）" if not_judged else ""),
              file=sys.stderr)
    return code


# ================================================================ apply: notebook records and dumps
TEST_OUTPUTS = ("submission_rank.csv", "rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json")


def apply_outputs(ctx):
    rank, seam = ctx["rank"], os.path.join(ctx["rank"], "seam_val")
    out = [os.path.join(seam, n) for n in ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json",
                                           "seam_local.json")]
    out += [os.path.join(rank, n) for n in ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json",
                                            "submission_rank.csv", "t04_apply_report.json", "attempt_verdict.json")]
    out += [os.path.join(rank, "smoke_questions", n) for n in t2.TEST_FILES.values()]
    return out


def load_npz(path):
    refuse_label_path(path)
    if not os.path.exists(path):
        refuse(f"missing {shown(path)}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_opp_report(ctx, fc, t02):
    """Apply prerequisite 1: opp's report of this mode passed with usable_so_far; its outputs and inputs unchanged."""
    rpath = os.path.join(ctx["opp"], "opp_report.json")
    if not os.path.exists(rpath):
        refuse(f"{shown(rpath)} is missing: run t04_rank.py opp first")
    if ctx["real"]:
        require_committed([rpath] + [os.path.join(ctx["opp"], f"o4_{s}.npz") for s in SETS],
                          "results/test04/opp (commit it after opp)")
    rep = read_json(rpath)
    if rep.get("format") != FORMATS["opp"] or rep.get("mode") != ctx["mode"]:
        refuse(f"{shown(rpath)}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (this run: {ctx['mode']})")
    if not (rep.get("passed") is True and rep.get("usable_so_far") is True):
        if verdict_kind(rep) == "drift":
            refuse(f"opp_report.json is a local drift (exit 4: {rep.get('unusable_reasons')}): check the local code or "
                   f".venv, then opp --rerun --reason TEXT")
        raise T04Error(f"opp 已判定 04 不用（{rep.get('unusable_reasons')}）：最終 R20，寫 RESULTS", 3,
                       reason="opp_no_04")
    probs = []
    for s in SETS:
        p = os.path.join(ctx["opp"], f"o4_{s}.npz")
        if not os.path.exists(p) or sha_raw(p) != ((rep.get("sets") or {}).get(s) or {}).get("output_sha256"):
            probs.append(f"o4_{s}.npz differs from the report")
        if ((rep.get("sets") or {}).get(s) or {}).get("questions_sha256") != sha_raw(t02["qfiles"][s]):
            probs.append(f"{s} question file changed since opp")
        keys_now = sha_raw(os.path.join(ctx["D2"], "rank", s, "dumps", "val_keys.npz"))
        if ((rep.get("test02") or {}).get("keys_sha256") or {}).get(s) != keys_now:
            probs.append(f"test02 {s} val_keys.npz changed since opp")
    if (rep.get("frozen_config") or {}).get("sha256") != FROZEN_CONFIG_SHA:
        probs.append("frozen_config sha256")
    if (rep.get("counts") or {}).get("sha256") != COUNTS_SHA or rep.get("tables_sha256") != TABLES_SHA:
        probs.append("counts / tables sha256 recorded in the report")
    if rep.get("p04_opp_sha256") != sha_lf(os.path.join(HERE, "p04_opp.py")) or rep.get("p04_opp_sha256") != \
            P04_OPP_SHA:
        probs.append("p04_opp.py sha256")
    now_counts = counts_sha_now()
    if now_counts != COUNTS_SHA:     # rev 3 check: restore the file (item 17); a new opp is not the remedy
        refuse(f"data/prereg04/counts_fit.npz is {'missing' if now_counts is None else 'sha256 ' + now_counts[:12]}, "
               f"not the registered {COUNTS_SHA[:12]}" + (f" (also: {probs})" if probs else "") + ": "
               + MSG_COUNTS_RESTORE)
    if probs:
        # rev 3 check: after opp's registered decision a new opp needs a PREREG_04 revision (opp_rerun_gate)
        refuse(f"{shown(rpath)} no longer matches its inputs / outputs: {probs}. Restore the changed file(s) from git "
               f"and run apply again; a new opp after its registered decision needs a PREREG_04 revision (opp --rerun "
               f"--reason TEXT --bug-revision N, rev 3 item 20); a test02 rerun under PREREG_02 stops 04 (rev 3 items "
               f"19, 20)")
    return rep, rpath


def check_unpack_record(ctx):
    """Apply prerequisite 2: unpack_sw.json and sw/** (real: committed); the record still matches failed/."""
    if not os.path.exists(ctx["unpack"]):
        refuse(f"{shown(ctx['unpack'])} is missing: run t04_rank.py unpack first")
    rec = read_json(ctx["unpack"])
    if rec.get("format") != FORMATS["unpack"] or rec.get("mode") != ctx["mode"]:
        refuse(f"{shown(ctx['unpack'])}: format {rec.get('format')!r}, mode {rec.get('mode')!r}")
    if ctx["real"]:
        files = sorted({os.path.join(r, f) for r, _d, fs in os.walk(ctx["sw"]) for f in fs} |
                       {os.path.join(ctx["sw"], *m.split("/")[1:]) for m in required_members()})
        require_committed([ctx["unpack"]] + files, "results/test04/unpack_sw.json and results/test04/sw/**")
    attempts = list_attempts(ctx)
    counted = [x["name"] for x in attempts if is_counted(x)]
    archived = {h for x in attempts for h in ((x["record"] or {}).get("zips") or {}).values()}
    probs = []
    if rec.get("failed_attempts_seen") != [x["name"] for x in attempts]:
        probs.append(f"archives now {[x['name'] for x in attempts]}, unpack saw {rec.get('failed_attempts_seen')}")
    if rec.get("attempt_number") != len(counted) + 1:
        probs.append(f"attempt number {rec.get('attempt_number')} vs {len(counted) + 1} now")
    if (rec.get("zip") or {}).get("sha256") in archived:
        probs.append("the unpacked zip has been archived")
    if probs:
        refuse(f"unpack_sw.json does not match results/test04/failed any more ({probs}): archive and unpack again")
    return rec, counted


def check_notebook(ctx, t02):
    """Apply prerequisite 3: the notebook records of sw/."""
    sw = ctx["sw"]
    cfg, log, seam = (read_json(os.path.join(sw, n)) for n in ("config.json", "run_log.json", "seam_check.json"))
    status, hist = tb.run_status(log), run_history(log)
    if status != "completed" or not hist or hist[-1][1] != "completed":
        seam_fail(f"sw/run_log.json status {status!r} (last run {hist[-1] if hist else None}): only a completed run "
                  f"is used. " + seam_msg(ctx), reason="run_log")
    earlier = tb.earlier_failures(log)
    if earlier:
        seam_fail(f"sw/run_log.json has earlier failed runs {earlier}: not the registered rerun. " + seam_msg(ctx),
                  reason="run_log_earlier")
    markers = done_markers_from_dir(sw)
    resumed = resumed_markers(markers, log)
    if resumed:
        seam_fail(f"markers not from the last run (resumed): {resumed[:5]}. " + seam_msg(ctx), reason="resumed")
    if seam.get("passed") is not True:
        seam_fail("sw/seam_check.json did not pass. " + seam_msg(ctx), reason="notebook_seam")
    if not is_num(seam.get("tol")):
        seam_fail(f"sw/seam_check.json tol {seam.get('tol')!r} is not a number. " + seam_msg(ctx), reason="notebook_seam")
    if cfg.get("format") != NB_CONFIG_FORMAT:
        refuse(f"sw/config.json format {cfg.get('format')!r}")
    fp = flags_problem(cfg, ctx["mode"])
    if fp:
        refuse(fp)
    if ctx["real"]:
        require_committed([NOTEBOOK04], NOTEBOOK_REL)
    same, nb_rec = notebook_identity(cfg, ctx["mode"])
    if same is False and ctx["real"]:
        seam_fail(f"the run's notebook_sha256 is not the committed {NOTEBOOK_REL}'s. " + seam_msg(ctx),
                  reason="not_committed_notebook", details=nb_rec)
    if same is False and ctx["mode"] == "rehearsal":
        print(f"WARNING: the rehearsal ran another {NOTEBOOK_REL} ({nb_rec}); recorded", file=sys.stderr)
    tb4, tb2 = cfg.get("test_bundle") or {}, t02["cfg"].get("test_bundle") or {}
    if tb4.get("manifest_sha256") != tb2.get("manifest_sha256") or any(
            (tb4.get("files") or {}).get(n) != (tb2.get("files") or {}).get(n) for n in tb.COMP_FILES["rank"]):
        refuse("sw/config.json test_bundle differs from test02's: not the test bundle of 11/04")
    tfiles = t2.check_test_files({"tests_dir": ctx["T"]}, cfg)
    sets = t2.check_set_inputs({"rank_dir": sw, "tests_dir": ctx["T"]}, cfg, t02["qfiles"])
    w = cfg.get("weights") or {}
    wsha = {x: (w.get(x) or {}).get("sha256") for x in ARMS}
    if not ctx["smoke"] and wsha != {x: pf.SW_WEIGHTS[x] for x in ARMS}:
        seam_fail(f"sw/config.json weights {wsha} are not prereg03's final.pth {pf.SW_WEIGHTS}. " + seam_msg(ctx),
                  reason="weights")
    return {"cfg": cfg, "log": log, "seam": seam, "markers": markers, "identity": nb_rec, "same": same,
            "test_files": tfiles, "sets": sets, "weights_sha256": wsha, "config_sha256": pin_sha(os.path.join(sw,
                                                                                                         "config.json"))}


def check_dumps(ctx, t02, nb):
    """Apply prerequisite 4 (spec 3.3 item 4), in the registered order over every set and arm: keys equal to
    test02's -> one environment (12 logit + 3 key markers) -> markers match their files -> contents. Any -> exit 1."""
    sw = ctx["sw"]
    vk2 = {s: rf.load_val_keys(os.path.join(ctx["D2"], "rank", s, "dumps")) for s in SETS}
    keys_equal = {}
    for s in SETS:
        a_ = load_npz(os.path.join(sw, s, "dumps", "val_keys.npz"))
        b_ = load_npz(os.path.join(ctx["D2"], "rank", s, "dumps", "val_keys.npz"))
        names = ("question_id", "slot", "color", "parsed")
        keys_equal[s] = sorted(a_) == sorted(b_) and all(
            k in a_ and k in b_ and a_[k].dtype == b_[k].dtype and a_[k].shape == b_[k].shape
            and np.array_equal(a_[k], b_[k]) for k in names)
    bad = [s for s in SETS if not keys_equal[s]]
    if bad:
        seam_fail(f"the test04 val_keys.npz of {bad} differ from test02's (question_id / slot / color / parsed): "
                  + seam_msg(ctx), reason="dump_keys", details={"keys_equal": keys_equal})
    markers = {}
    for s in SETS:
        for stem in marker_stems():
            p = os.path.join(sw, s, "dumps", f"_DONE_{stem}.json")
            markers[(s, stem)] = read_json(p)
    env = {f"{s}/{stem}": {k: m.get(k) for k in DUMP_ENV_KEYS} for (s, stem), m in markers.items()}
    ref_env = env[f"seam_val/val_keys"]
    mixed = sorted(k for k, v in env.items() if v != ref_env)
    if mixed:
        seam_fail(f"the test04 dumps were not all computed in one environment (gpu_name / torch / cudnn): {mixed[:6]}. "
                  + seam_msg(ctx), reason="dump_environment", details={"env": env})
    bad = []
    for (s, stem), m in markers.items():
        p = os.path.join(sw, s, "dumps", f"{stem}.npz")
        if not os.path.exists(p):
            bad.append(f"{s}/{stem}.npz missing")
            continue
        if m.get("sha256") != sha_raw(p) or m.get("bytes") != os.path.getsize(p) or m.get("rows") != vk2[s]["n"]:
            bad.append(f"{s}/{stem}: marker sha256 / bytes / rows ({m.get('rows')} vs {vk2[s]['n']}) do not match")
    if bad:
        seam_fail("completion markers do not match their files: " + "; ".join(bad[:6]) + ". " + seam_msg(ctx),
                  reason="dump_markers", details={"problems": bad})
    dumps, bad = {}, []
    for s in SETS:
        n, parsed = vk2[s]["n"], vk2[s]["parsed"]
        dumps[s] = {}
        for x in ARMS:
            pd, ps = (os.path.join(sw, s, "dumps", f"val_p03{x}_{v}.npz") for v in ("d4", "s0_swap"))
            try:
                d4 = rf.load_dump(pd, n, p3.N_SYMS)
                rf.check_val_logits(d4, parsed)
                swp = p3.load_swap(ps, n)
            except FOREIGN_ERRORS as e:
                bad.append(f"{s}/{x}: {e}")
                continue
            zsw = load_npz(ps)
            view = str(zsw.get("view")) if "view" in zsw else None
            fin, nan = np.isfinite(swp["logits"]).all(1), np.isnan(swp["logits"]).all(1)
            want_w = pf.SW_WEIGHTS[x] if not ctx["smoke"] else nb["weights_sha256"].get(x)
            for what, got in (("d4", d4["weights_sha256"]), ("s0_swap", swp["weights_sha256"]),
                              ("d4 marker", markers[(s, f"val_p03{x}_d4")].get("weights_sha256")),
                              ("s0_swap marker", markers[(s, f"val_p03{x}_s0_swap")].get("weights_sha256"))):
                if got != want_w:
                    bad.append(f"{s}/{x} {what}: weights_sha256 {str(got)[:12]} is not {str(want_w)[:12]}")
            if view != SWAP_VIEW:
                bad.append(f"{s}/{x} s0_swap: view {view!r}")
            if not fin[parsed].all() or not nan[~parsed].all():
                bad.append(f"{s}/{x} s0_swap: the NaN rows are not the unparsed rows")
            dumps[s][x] = {"d4": d4, "swap": swp}
    if bad:
        seam_fail("dump contents: " + "; ".join(bad[:6]) + ". " + seam_msg(ctx), reason="dump_content",
                  details={"problems": bad})
    return vk2, dumps, {"keys_equal": keys_equal, "env": ref_env, "env_same": True}


# ================================================================ apply: notebook seam recomputed locally (spec 3.3.5)
def c0_functions(nb_path):
    """question_spans and c0_of from a notebook's code cells (ast source segments), exec'd in a namespace holding
    only np and torch. -> (question_spans, c0_of, sources)."""
    import torch
    with open(refuse_label_path(nb_path), encoding="utf-8") as f:
        nb = json.load(f)
    found = {}
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ("c0_of", "question_spans"):
                found.setdefault(node.name, set()).add(ast.get_source_segment(src, node))
    for name in ("question_spans", "c0_of"):
        if len(found.get(name, ())) != 1:
            refuse(f"{shown(nb_path)}: {len(found.get(name, ()))} distinct definitions of {name} (need exactly one)")
    sources = {k: next(iter(v)) for k, v in found.items()}
    ns = {"np": np, "torch": torch}
    exec(sources["question_spans"] + "\n\n\n" + sources["c0_of"] + "\n", ns)   # noqa: S102 (committed notebook code)
    return ns["question_spans"], ns["c0_of"], sources


def c0_max_diff(c0_of, spans, parsed, L1, L0):
    c1, c0 = c0_of(L1, spans, parsed), c0_of(L0, spans, parsed)
    diffs = [max(abs(a_ - b_) for a_, b_ in zip(x, y)) for x, y in zip(c1, c0) if x is not None]
    return max(diffs) if diffs else 0.0


def seam_tol_sw(ctx, x):
    """1e-4 when the test04 seam_val d4 dump's GPU (its marker) is prereg03's dump GPU (not None), else 1e-3."""
    mine = read_json(os.path.join(ctx["sw"], "seam_val", "dumps", f"_DONE_val_p03{x}_d4.json")).get("gpu_name")
    ref = read_json(os.path.join(P03_ROOT, f"rank_{x}", "dumps", f"_DONE_val_p03{x}_d4.json")).get("gpu_name")
    same = mine is not None and mine == ref
    return (TOL_SAME_GPU if same else TOL_OTHER_GPU), {"seam_dump_gpu": mine, "prereg03_dump_gpu": ref, "same_gpu": same}


def notebook_seam_recompute(ctx, nb):
    """The notebook seam recomputed with the committed prereg03 dumps and the notebook's own c0_of (torch CPU,
    float32): pass / fail and tol must equal seam_check.json's (else exit 1); the values' differences are recorded."""
    seam = nb["seam"]
    if ctx["smoke"]:
        return {"skipped": "smoke: the notebook compared with prereg03_smoke (record only)", "passed": None,
                "tol": float(seam["tol"])}
    nb_path = C0_SOURCE_NB
    if not os.path.exists(nb_path):
        if ctx["real"]:
            refuse(f"{shown(nb_path)} is missing: the local recompute uses the committed notebook's c0_of")
        nb_path = C0_FALLBACK_NB
    question_spans, c0_of, sources = c0_functions(nb_path)
    for x in ARMS:
        for name, want in P03_SEAM_REF_PINS[x].items():
            got = sha_raw(os.path.join(P03_ROOT, f"rank_{x}", "dumps", name))
            if got != want:
                refuse(f"results/prereg03/rank_{x}/dumps/{name}: sha256 {got[:12]} is not the pinned {want[:12]}")
    new_keys = load_npz(os.path.join(ctx["sw"], "seam_val", "dumps", "val_keys.npz"))
    arms, tols = {}, {}
    for x in ARMS:
        ref_dir = os.path.join(P03_ROOT, f"rank_{x}", "dumps")
        ref_keys = load_npz(os.path.join(ref_dir, "val_keys.npz"))
        keq = all(k in new_keys and k in ref_keys and new_keys[k].shape == ref_keys[k].shape
                  and np.array_equal(new_keys[k], ref_keys[k]) for k in ("question_id", "slot", "color", "parsed"))
        tol, trec = seam_tol_sw(ctx, x)
        tols[x] = tol
        rec = {"keys_equal": keq, "tol": tol, **trec}
        if keq:
            parsed = new_keys["parsed"].astype(bool)
            spans = question_spans(new_keys["question_id"])
            d1 = load_npz(os.path.join(ctx["sw"], "seam_val", "dumps", f"val_p03{x}_d4.npz"))["logits"]
            d0 = load_npz(os.path.join(ref_dir, f"val_p03{x}_d4.npz"))["logits"]
            s1 = load_npz(os.path.join(ctx["sw"], "seam_val", "dumps", f"val_p03{x}_s0_swap.npz"))["logits"]
            s0 = load_npz(os.path.join(ref_dir, f"val_p03{x}_s0_swap.npz"))["logits"]
            rec["c0_s0_max_abs_diff"] = c0_max_diff(c0_of, spans, parsed, d1, d0)
            rec["c0_swap_max_abs_diff"] = c0_max_diff(c0_of, spans, parsed, s1[:, None, :], s0[:, None, :])
            rec["bitwise_equal_d4"] = d1.dtype == d0.dtype and d1.shape == d0.shape and d1.tobytes() == d0.tobytes()
            rec["bitwise_equal_swap"] = s1.dtype == s0.dtype and s1.shape == s0.shape and s1.tobytes() == s0.tobytes()
            rec["passed"] = bool(rec["c0_s0_max_abs_diff"] <= tol and rec["c0_swap_max_abs_diff"] <= tol)
        else:
            rec["passed"] = False
        nb_arm = (seam.get("arms") or {}).get(x) or {}
        rec["notebook"] = {k: nb_arm.get(k) for k in ("c0_s0_max_abs_diff", "c0_swap_max_abs_diff", "passed")}
        rec["difference_to_notebook_record_only"] = {
            k: (abs(rec[k] - nb_arm[k]) if is_num(rec.get(k)) and is_num(nb_arm.get(k)) else None)
            for k in ("c0_s0_max_abs_diff", "c0_swap_max_abs_diff")}
        arms[x] = rec
    local_tol = tols["a"] if tols["a"] == tols["b"] else None
    passed = all(arms[x]["passed"] for x in ARMS) and local_tol is not None
    out = {"passed": passed, "tol": local_tol, "notebook_passed": seam.get("passed"), "notebook_tol": seam.get("tol"),
           "arms": arms, "source_notebook": shown(nb_path), "source_sha256": {k: hashlib.sha256(v.encode("utf-8"))
                                                                            .hexdigest() for k, v in sources.items()},
           "rule": "c0_of / question_spans exec'd from the notebook (torch CPU float32); per arm C0 (s0, swap) max "
                   "difference <= tol; tol 1e-4 when the seam_val dump GPU equals prereg03's, else 1e-3"}
    mism = []
    if passed != (seam.get("passed") is True):
        mism.append(f"local pass {passed} vs notebook {seam.get('passed')!r}")
    for x in ARMS:
        nbp = arms[x]["notebook"]["passed"]
        if nbp is not None and nbp != arms[x]["passed"]:
            mism.append(f"arm {x}: local pass {arms[x]['passed']} vs notebook {nbp!r}")
    if local_tol is None or float(seam["tol"]) != local_tol:
        mism.append(f"tol: notebook {seam.get('tol')!r}, from the markers {local_tol!r}")
    out["equal_to_notebook"] = not mism
    if mism:
        seam_fail("the notebook seam recomputed locally does not agree with seam_check.json: " + "; ".join(mism) + ". "
                  + seam_msg(ctx), stage="notebook_seam_recompute", reason="recompute_mismatch", details=out)
    return out


# ================================================================ apply: Q02 per set (spec 3.3 'every set')
def load_c5(name, ddir, vk, want_wsha):
    d = rf.load_dump(os.path.join(ddir, "val_%s.npz" % C5_DUMP), vk["n"], rf.DUMP_S[C5_DUMP])
    if d["weights_sha256"] != want_wsha:
        refuse(f"{name}: C5 dump weights {d['weights_sha256'][:12]} are not {str(want_wsha)[:12]}")
    rf.check_val_logits(d, vk["parsed"])
    return d


def fuse_set(name, vk, qfile, c5, op, o4z, sw, fc, smoke):
    """Q02 for one question set. vk = rf.load_val_keys (test02's keys), c5 = rf.load_dump of val_p02e9_d4,
    op = opening arrays (set, question_id, slot, color, o, has), o4z = o4_<set>.npz arrays, sw = {arm: {'d4', 'swap'}}
    loaded dumps. -> dict(rows, P, ok, n, qids, lp, mask, info)."""
    # C5 (verbatim t2.apply_set :702-704)
    logits = {C5_DUMP: np.where(vk["parsed"][:, None, None], c5["logits"], 0.0).astype(np.float32)}
    lp5 = p3.combo_logprobs(("C5",), logits)
    # question-file alignment (verbatim t2.apply_set :706-719)
    questions = rf.read_question_meta(refuse_label_path(qfile))
    by_q = {q: (n, cols) for q, n, cols in questions}
    groups = vk["groups"]
    qids = [q for q, _ in groups]
    file_q = [q for q, _, _ in questions]
    if smoke:
        if not qids or qids != file_q[:len(qids)]:
            refuse(f"{name}: smoke dump questions are not the first questions of the question file")
    elif qids != file_q:
        refuse(f"{name}: val_keys questions differ from the question file (set or order)")
    for q, rows in groups:
        n, cols = by_q[q]
        if len(rows) != n or [vk["color"][i] for i in rows] != cols:
            refuse(f"{name}: question {q}: val_keys slots/colours differ from the question file")
    # question level
    parsed = vk["parsed"]
    cnn = [[i for i in rows if parsed[i]] for _, rows in groups]
    S = rf.pool_matrix(cnn, vk["n"])
    n_q = np.array([len(c) for c in cnn], dtype=np.float64)
    m = rf.pooled(S, lp5, "log")
    ok = n_q > 0
    # OP (verbatim t2.apply_set :726-738)
    sel = np.nonzero(np.asarray(op["set"]) == name)[0]
    pos = {(q, int(s)): int(j) for q, s, j in zip(rf.str_array(np.asarray(op["question_id"])[sel]),
                                                 np.asarray(op["slot"])[sel].tolist(), sel.tolist())}
    oi = []
    for i, (q, s) in enumerate(zip(vk["question_id"], vk["slot"])):
        j = pos.get((q, s))
        if j is None or str(op["color"][j]) != vk["color"][i]:
            refuse(f"opening_test.npz: no matching entry for {name} question {q} slot {s}")
        oi.append(j)
    oi = np.asarray(oi)
    o = np.asarray(op["o"], dtype=np.float64)[oi]
    has = np.asarray(op["has"], dtype=bool)[oi]
    mop = np.asarray(rf.pool_matrix([[i for i in rows if has[i]] for _, rows in groups], vk["n"]) @ o)
    # O4 (rev 3 item 3: the failed slots leave E_O4 and the O4 / T4 kappa)
    if rf.str_array(o4z["question_id"]) != vk["question_id"] or \
            [int(x) for x in np.asarray(o4z["slot"]).tolist()] != vk["slot"] or \
            rf.str_array(o4z["color"]) != vk["color"]:
        refuse(f"{name}: o4_{name}.npz question_id / slot / color differ from test02's val_keys.npz")
    o4, t4 = np.asarray(o4z["o4"]), np.asarray(o4z["t4"])
    if o4.dtype != np.float64 or o4.shape != (vk["n"], rf.N_LEVELS) or t4.dtype != np.float64 or t4.shape != o4.shape:
        refuse(f"{name}: o4 / t4 must be float64 [{vk['n']}, 10]")
    rfail = np.asarray(o4z["replay_failed"], dtype=bool)
    mask = parsed & ~rfail
    lp_o4 = rf.log_softmax(np.where(mask[:, None], o4, 0.0))
    lp_t4 = rf.log_softmax(np.where(mask[:, None], t4, 0.0))
    S_o4 = rf.pool_matrix([[i for i in rows if mask[i]] for _, rows in groups], vk["n"])
    E = {"OP": mop, "O4": rf.pooled(S_o4, lp_o4, "log")}
    lp = {"O4": lp_o4, "T4": lp_t4}
    # SWA / SWB (masked in float32, then float64: pf.val_arm :965-966)
    for x in ARMS:
        src = ARM_SOURCE[x]
        lp[src] = rf.log_softmax(np.where(parsed[:, None], sw[x]["swap"]["logits"], 0.0).astype(np.float64))
        lp["A0" if x == "a" else "B0"] = rf.log_softmax(
            np.where(parsed[:, None], sw[x]["d4"]["logits"][:, 0, :], 0.0).astype(np.float64))
        E[src] = rf.pooled(S, lp[src], "log")
    theta = np.asarray(fc["params"]["theta"], dtype=np.float64)
    P = np.full((len(groups), rf.N_LEVELS), np.nan)
    if ok.any():
        z, _ = pf.gen_logits(theta, m[ok], [E[t][ok] for t in fc["combo"]["terms"]], n_q[ok])
        P[ok] = rf.softmax(z)
    rows_out = []
    for k, (q, rows) in enumerate(groups):
        probs = [repr(float(x)) for x in P[k]] if ok[k] else [""] * rf.N_LEVELS
        rows_out.append([q, len(rows), int(n_q[k])] + probs)
    info = {"questions": len(groups), "questions_without_parsed_game": int((~ok).sum()),
            "questions_without_parsed_game_ids": [q for q, k in zip(qids, ok) if not k],
            "games": vk["n"], "games_parsed": int(parsed.sum()), "o4_mask_games": int(mask.sum()),
            "o4_replay_failed": int(rfl_sum(rfail)),
            "questions_with_empty_o4_mask": int(sum(1 for c in cnn if c and not any(mask[i] for i in c)))}
    return {"rows": rows_out, "P": P, "ok": ok, "n": n_q, "qids": qids, "lp": lp, "E": E, "mask": mask, "info": info}


def rfl_sum(a):
    return int(np.asarray(a, dtype=bool).sum())


def colour_check(vk, fz):
    """Rev 3 item 7 on one set: O4 vs T4 (mask), SWA vs A0, SWB vs B0 (parsed)."""
    lp = fz["lp"]
    return {"O4": kappa_record(dict(vk, parsed=fz["mask"]), lp["T4"], lp["O4"]),
            "SWA": kappa_record(vk, lp["A0"], lp["SWA"]), "SWB": kappa_record(vk, lp["B0"], lp["SWB"])}


def write_probs(ctx, path, rows):
    guard_write(ctx, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t2.write_probs(path, rows)


def apply_decision(opp_rep, nb_passed, local_passed, colour, smoke=False):
    """Spec 3.3 'usable' without check_submission. smoke: the six dkappa signs are recorded only (smoke-trained
    weights on 12 questions per set; spec 5.3 'smoke: record only'), so the smoke run reaches the output files and
    check_submission. -> (usable, reasons)."""
    reasons = []
    if nb_passed is not True:
        reasons.append("notebook seam")
    if local_passed is not True:
        reasons.append("local seam")
    for s in SETS:
        st = (opp_rep.get("sets") or {}).get(s) or {}
        if not rate_ok(st.get("failures"), st.get("slots")):
            reasons.append(f"replay:{s}: {st.get('failures')} of {st.get('slots')} slots failed (> {MAX_RATE:.1%})")
    if (opp_rep.get("seam_val") or {}).get("file_sha256_equal") is not True and opp_rep.get("mode") != "smoke":
        reasons.append("seam_file: seam_val O4 not byte-identical")
    for s in TEST_SETS:
        for src in OPP_SOURCES:
            kv = (colour.get(s) or {}).get(src)
            if not dkappa_ok(kv) and not smoke:
                reasons.append(f"dkappa:{s}.{src}: {(kv or {}).get('dkappa')!r} (needs > 0)")
    return not reasons, reasons


class Apply:
    def __init__(self, ctx, a, t0):
        self.ctx, self.a, self.t0 = ctx, a, t0
        self.replaced, self.cleared, self.bug = {}, False, None
        self.pre, self.stage = {}, "prerequisites"

    # ---- rerun rules (spec 3.3 'rerunning apply')
    def rerun_gate(self):
        ctx, a = self.ctx, self.a
        existing = [p for p in apply_outputs(ctx) if os.path.exists(p)]
        if not existing:
            if a.bug_revision is not None:
                self.bug = check_bug_revision(a.bug_revision, [])
            return
        if not a.rerun:
            refuse(f"outputs already exist ({[os.path.basename(p) for p in existing][:6]}); they are never overwritten "
                   f"without --rerun --reason TEXT")
        rep = read_json_or_none(os.path.join(ctx["rank"], "t04_apply_report.json"))
        ver = read_json_or_none(os.path.join(ctx["rank"], "attempt_verdict.json"))
        if rep is not None:
            if a.bug_revision is None:
                refuse("t04_apply_report.json exists: this attempt already has the registered decision; a rerun needs a "
                       "PREREG_04 revision naming a bug (--bug-revision N; rev 3 item 20)")
            self.bug = check_bug_revision(a.bug_revision, [("t04_apply_report.json created", rep.get("created"))])
        elif ver is not None and ver.get("exit_code") != 4:
            if a.bug_revision is None:
                # K5 gate round 2: on the registered rerun (a counted archive already) archive refuses, so the refusal
                # points at MSG_SEAM_LAST like the exit-1 message of that apply (seam_msg), not at 'archive, rerun'
                counted = [x["name"] for x in list_attempts(ctx) if is_counted(x)]
                refuse(f"the last apply ended with exit {ver.get('exit_code')!r} ({ver.get('stage')}): "
                       + (f"不是重跑 apply（counted archives {counted}）。" + MSG_SEAM_LAST if counted else
                          "接縫類失敗要 archive 後重跑 notebook，不是重跑 apply"))
            self.bug = check_bug_revision(a.bug_revision, [("attempt_verdict.json created", ver.get("created"))])
        elif a.bug_revision is not None:
            self.bug = check_bug_revision(a.bug_revision, [])

    def clear(self):
        if not self.cleared:
            self.replaced = remove_old(self.ctx, apply_outputs(self.ctx))
            self.cleared = True

    def write_verdict(self, e):
        ctx = self.ctx
        self.clear()
        unp = read_json_or_none(ctx["unpack"])
        rec = {"format": FORMATS["verdict"], "mode": ctx["mode"], "created": now_iso(), "exit_code": e.code,
               "stage": e.stage or self.stage, "reason": e.reason or str(e).splitlines()[0][:200], "message": str(e),
               "details": e.details, "sw_sha256": ((unp or {}).get("zip") or {}).get("sha256"),
               "rerun": rerun_record(self.a, self.replaced, self.bug), "checks": self.pre,
               "paths": paths_record(ctx), "tool": tool_record(), "environment": environment(),
               "seconds": round(time.time() - self.t0, 1)}
        os.makedirs(ctx["rank"], exist_ok=True)
        write_json(ctx, os.path.join(ctx["rank"], "attempt_verdict.json"), rec)

    # ---- the run
    def go(self):
        ctx, a = self.ctx, self.a
        smoke = ctx["smoke"]
        self.rerun_gate()
        fc, t02, self.pre = common_prereqs(ctx, "apply")
        opp, opp_path = check_opp_report(ctx, fc, t02)
        unp, counted = check_unpack_record(ctx)
        ctx["counted_archives"] = counted      # K5 gate round 1: the exit-1 message of the registered rerun (seam_msg)
        nb = check_notebook(ctx, t02)
        vk, dumps, dump_rec = check_dumps(ctx, t02, nb)
        self.stage = "notebook_seam_recompute"
        recompute = notebook_seam_recompute(ctx, nb)
        self.stage = "prerequisites"
        ref = {n: os.path.join(P04_RANK, n) for n in P04_SEAM_PINS}
        ref_sha = {n: pin_sha(p) for n, p in ref.items()}
        if ref_sha != P04_SEAM_PINS:
            refuse(f"results/prereg04/rank seam references changed: {ref_sha}")
        tol_c5 = float(t02["report"]["seam_local"]["tol"]) if is_num((t02["report"].get("seam_local") or {})
                                                                      .get("tol")) else None
        tol_sw = recompute["tol"]
        tol = max(tol_c5, tol_sw) if (tol_c5 is not None and tol_sw is not None) else (tol_sw or tol_c5)
        c5_w = t02["c5_weights"] if smoke else fc["source_sha256"]["C5"]["weights_sha256"]
        o4z = {s: load_npz(os.path.join(ctx["opp"], f"o4_{s}.npz")) for s in SETS}
        self.clear()
        rank, seam_dir = ctx["rank"], os.path.join(ctx["rank"], "seam_val")
        os.makedirs(seam_dir, exist_ok=True)
        guard_write(ctx, seam_dir)

        def run_set(s):
            ddir = os.path.join(ctx["D2"], "rank", s, "dumps")
            c5 = load_c5(s, ddir, vk[s], c5_w)
            fz = fuse_set(s, vk[s], t02["qfiles"][s], c5, t02["opening"], o4z[s], dumps[s], fc, smoke)
            fz["c5_sha256"] = c5["sha256"]
            return fz

        # ---- seam first
        self.stage = "local_seam"
        fzs = run_set("seam_val")
        seam_csv = os.path.join(seam_dir, "rank_val_probs.csv")
        write_probs(ctx, seam_csv, fzs["rows"])
        decide.run(seam_csv, seam_dir)
        sc = t2.seam_compare(seam_csv, os.path.join(seam_dir, "pred_B.csv"), ref["rank_val_probs.csv"],
                             ref["pred_B.csv"], tol, smoke)
        same_pred_a = t2._lf_bytes(os.path.join(seam_dir, "pred_A.csv")) == t2._lf_bytes(ref["pred_A.csv"])
        warnings = []
        nb_bitwise = (nb["seam"].get("bitwise_equal") is True, t02["seam"].get("bitwise_equal") is True)
        if not smoke and all(nb_bitwise) and not (sc["byte_identical_probs"] and sc["byte_identical_pred_B"]):
            warnings.append("both notebooks' seam_val dumps were bitwise equal to their references but the local seam "
                            "files are not byte identical: look for a drift between this tool and "
                            "p04_rank_fuse.cmd_apply")
        col_seam = colour_check(vk["seam_val"], fzs)
        o4_seam = col_seam["O4"].get("dkappa")
        o4_opp = (opp.get("seam_val") or {}).get("dkappa_O4")
        dk_seam = {"O4": {"dkappa": o4_seam, "opp_value": o4_opp, "equal_to_opp": o4_seam == o4_opp,
                          "reference": DKAPPA_O4_SEAM_REF, "record": col_seam["O4"]}}
        cr = read_json_or_none(P04_CHECK_REPORT) or {}
        for src in ("SWA", "SWB"):
            v = col_seam[src].get("dkappa")
            ref_v = ((((cr.get("sources") or {}).get(src) or {}).get("colour") or {}).get("val_sub_questions")
                     or {}).get("dkappa")
            dk_seam[src] = {"dkappa": v, "prereg04_record": ref_v, "prereg04_record_pinned": DKAPPA_SW_SEAM_REF[src],
                            "prereg04_record_file": "results/prereg04/rank/check_report.json sources.%s.colour."
                                                    "val_sub_questions.dkappa" % src,
                            "difference_record_only": (v - ref_v) if is_num(v) and is_num(ref_v) else None,
                            "record": col_seam[src]}
        seam_local = {"format": FORMATS["seam_local"], "what": "test04 local Q02 seam (PREREG_04 5.2, rev 3 item 5): "
                                                               "Q02 on the seam_val inputs vs results/prereg04/rank",
                      "mode": ctx["mode"], "created": now_iso(), **sc, "required": not smoke, "warnings": warnings,
                      "byte_identical_pred_A": same_pred_a,
                      "reference_sha256": ref_sha, "set": fzs["info"],
                      "tol_check": {"tol": tol, "c5": {"test02_seam_tol": tol_c5,
                                                       "seam_dump_gpu": ((t02["report"].get("seam_local") or {})
                                                                         .get("tol_check") or {}).get("seam_dump_gpu"),
                                                       "prereg02_dump_gpu": ((t02["report"].get("seam_local") or {})
                                                                             .get("tol_check") or {})
                                                       .get("prereg02_dump_gpu")},
                                    "sw": {"notebook_tol": nb["seam"].get("tol"), "recomputed_tol": tol_sw,
                                           **{k: ((recompute.get("arms") or {}).get("a") or {}).get(k)
                                              for k in ("seam_dump_gpu", "prereg03_dump_gpu", "same_gpu")},
                                           "arms": {x: {k: ((recompute.get("arms") or {}).get(x) or {}).get(k)
                                                        for k in ("seam_dump_gpu", "prereg03_dump_gpu", "same_gpu",
                                                                  "tol")} for x in ARMS}},
                                    "rule": "max of the two"},
                      "dkappa_seam_val": dk_seam, "notebook_seam_local_recompute": recompute,
                      "notebook_seam_bitwise_equal": {"test04": nb["seam"].get("bitwise_equal"),
                                                      "test02": t02["seam"].get("bitwise_equal")},
                      "outputs_sha256": {n: sha_raw(os.path.join(seam_dir, n)) for n in
                                         ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json")},
                      "tool": tool_record(), "environment": environment()}
        seam_local_path = write_json(ctx, os.path.join(seam_dir, "seam_local.json"), seam_local)
        for w_ in warnings:
            print("WARNING: " + w_, file=sys.stderr)
        print(f"seam: {sc['questions']} questions, max |dP| {sc['max_abs_diff']:.3g} (tol {sc['tol']:g}), rule-B "
              f"differences {sc['rule_b_differences']} (max {sc['rule_b_max']}), byte identical "
              f"{sc['byte_identical_probs']}/{sc['byte_identical_pred_B']}/{same_pred_a} -> "
              f"{'PASSED' if sc['passed'] else 'FAILED'}" + (" (record only: smoke)" if smoke else ""), flush=True)
        print(f"seam_val dkappa: O4 {o4_seam!r} (opp {o4_opp!r}), SWA {dk_seam['SWA']['dkappa']!r}, SWB "
              f"{dk_seam['SWB']['dkappa']!r}")
        if o4_seam != o4_opp and not smoke:
            drift(f"seam_val O4 dkappa {o4_seam!r} differs from opp's {o4_opp!r}: local drift, stop and check the local "
                  f"code", "o4_consistency", "seam_val_o4_dkappa", {"apply": o4_seam, "opp": o4_opp})
        if not sc["passed"] and not smoke:
            seam_fail("本機接縫沒過：" + seam_msg(ctx, "這次的 04 不用、不上傳，走 docs/TESTDAY_04.md「重跑一次」"),
                      "local_seam", "local_seam_failed", {"seam_compare": sc})
        # ---- public, private
        self.stage = "o4_consistency"
        fzt, colour = {}, {"seam_val": dk_seam}
        for s in TEST_SETS:
            fzt[s] = run_set(s)
            colour[s] = colour_check(vk[s], fzt[s])
            v, want = colour[s]["O4"].get("dkappa"), ((opp.get("o4_dkappa_early") or {}).get(s) or {}).get("dkappa")
            if v != want:
                drift(f"{s} O4 dkappa {v!r} differs from opp's {want!r}: local drift, stop and check the local code",
                      "o4_consistency", f"{s}_o4_dkappa", {"apply": v, "opp": want})
        colour["definition"] = ("PREREG_04 rev 3 section 11 item 7: dkappa = mean over sub-questions of "
                                "kappa(target-side reference) - kappa(opponent-side source) "
                                "(p04_rank_fuse.val_side_kappa); O4 vs T4 on parsed & ~replay_failed, SWA vs A0 and "
                                "SWB vs B0 on parsed; > 0 on public and private for each of O4, SWA, SWB")
        usable, reasons = apply_decision(opp, nb["seam"].get("passed"), sc["passed"] or smoke, colour, smoke)
        rep = self.report_base(fc, t02, opp, opp_path, unp, counted, nb, dump_rec, recompute, sc, warnings,
                               seam_local_path, colour, fzs, fzt, vk, o4z)
        if smoke:
            rep["smoke_record_only"] = {
                "rule": "smoke: the local seam and the six dkappa signs are recorded only (smoke weights)",
                "dkappa_not_positive": [f"{s}.{src}: {((colour.get(s) or {}).get(src) or {}).get('dkappa')!r}"
                                        for s in TEST_SETS for src in OPP_SOURCES
                                        if not dkappa_ok((colour.get(s) or {}).get(src))]}
        if not usable:
            rep.update(usable=False, unusable_reasons=reasons, check_submission=None, outputs_sha256={}, exit_code=3,
                       rule_b_differences_vs_r20=None)
            rep["seconds"] = round(time.time() - self.t0, 1)
            write_json(ctx, os.path.join(rank, "t04_apply_report.json"), rep)
            print("apply: 04 不用（PREREG_04 5.2）：" + "; ".join(reasons) + "\ncommit results/test04/rank/；最終 R20，"
                  "寫 RESULTS（不重跑）", file=sys.stderr)
            return 3
        # ---- the test outputs
        rows = fzt["public"]["rows"] + fzt["private"]["rows"]
        probs_csv = os.path.join(rank, "rank_test_probs.csv")
        write_probs(ctx, probs_csv, rows)
        for n in ("pred_A.csv", "pred_B.csv", "decide_summary.json"):
            guard_write(ctx, os.path.join(rank, n))
        dsum = decide.run(probs_csv, rank)
        sub = guard_write(ctx, os.path.join(rank, "submission_rank.csv"))
        t2.atomic_copy(os.path.join(rank, "pred_B.csv"), sub)
        with open(sub, "rb") as f1, open(os.path.join(rank, "pred_B.csv"), "rb") as f2:
            if f1.read() != f2.read():
                refuse("submission_rank.csv is not a byte copy of pred_B.csv")
        self.stage = "check_submission"
        if smoke:
            qdir = os.path.join(rank, "smoke_questions")
            os.makedirs(qdir, exist_ok=True)
            check_q = []
            for s in TEST_SETS:
                dst = guard_write(ctx, os.path.join(qdir, t2.TEST_FILES[s]))
                with open(t02["qfiles"][s], newline="", encoding="utf-8-sig") as f:
                    lines = f.read().splitlines(keepends=True)
                with open(dst, "w", newline="", encoding="utf-8") as f:
                    f.writelines(lines[:1 + len(fzt[s]["qids"])])
                check_q.append(dst)
            errors = check_submission.check("rank", sub, check_q)
            check_mode = "default mode on the smoke subsets of the question files"
        else:
            check_q = [t02["qfiles"]["public"], t02["qfiles"]["private"]]
            errors = check_submission.check("rank", sub, check_q, official=True)
            check_mode = "official"
        if errors:
            # rev 3 check (item 8): a local drift leaves no test-set output; pred_B.csv is a byte copy of the rejected
            # file, so all five go (seam_val/ stays as the record, as for the other exit-1 / exit-4 stops)
            removed = []
            for n in TEST_OUTPUTS:
                p = os.path.join(rank, n)
                if os.path.exists(p):
                    os.remove(p)
                    removed.append(n)
            drift(f"check_submission ({check_mode}) failed on the Q02 file: {errors[:5]}; the test-set outputs "
                  f"{removed} removed. Stop and check the local code", "check_submission", "check_submission",
                  {"errors": errors[:50], "removed": removed})
        preds = dict(t2.read_pred(sub))
        r20_sub = os.path.join(ctx["O2"], "submission_rank.csv")
        r20_pred = dict(t2.read_pred(r20_sub)) if os.path.exists(r20_sub) else None
        diff_r20 = (sum(1 for q, v in preds.items() if r20_pred.get(q) != v) if r20_pred is not None else None)
        level = {s: t2.level_summary(fzt[s]["P"], fzt[s]["ok"], [preds[q] for q in fzt[s]["qids"]]) for s in TEST_SETS}
        rep.update(usable=True, unusable_reasons=[], exit_code=0,
                   check_submission={"mode": check_mode, "questions": [shown(p) for p in check_q], "errors": errors},
                   outputs_sha256={n: sha_raw(os.path.join(rank, n)) for n in
                                   ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json",
                                    "submission_rank.csv")},
                   decide_summary={k: dsum[k] for k in ("n_questions", "n_missing", "missing_question_ids", "n_A_ne_B",
                                                        "distribution_B")},
                   level_summary=level, rule_b_differences_vs_r20=diff_r20,
                   rule_b_differences_vs_r20_file={"path": shown(r20_sub),
                                                   "sha256": sha_raw(r20_sub) if os.path.exists(r20_sub) else None})
        rep["seconds"] = round(time.time() - self.t0, 1)
        write_json(ctx, os.path.join(rank, "t04_apply_report.json"), rep)
        print(f"apply Q02: public {len(fzt['public']['qids'])}, private {len(fzt['private']['qids'])} questions "
              f"({dsum['n_missing']} without a parsed game, filled with {decide.FALLBACK}) -> {sub} (sha256 "
              f"{rep['outputs_sha256']['submission_rank.csv'][:12]}); check_submission {check_mode}: OK; rule-B "
              f"differences vs R20: {diff_r20}")
        print("下一步：commit results/test04/rank/，再照 docs/TESTDAY_04.md 第 6 步 official_log.py prepare"
              if ctx["real"] else f"({ctx['mode']}: nothing is uploaded)")
        return 0

    def report_base(self, fc, t02, opp, opp_path, unp, counted, nb, dump_rec, recompute, sc, warnings,
                    seam_local_path, colour, fzs, fzt, vk, o4z):
        ctx = self.ctx
        seam_s = nb["seam"]
        cfg = nb["cfg"]
        sets_info = {"seam_val": fzs["info"], **{s: fzt[s]["info"] for s in TEST_SETS}}
        bound = self.inputs_bound(t02, opp_path, seam_local_path)
        attempts = list_attempts(ctx)
        return {
            "format": FORMATS["apply"], "what": "test04 competition 2 apply (PREREG_04 5.2, rev 3 section 11): frozen "
                                                "Q02 per set, rule B, submission_rank.csv = pred_B.csv "
                                                "(public rows then private rows)",
            "mode": ctx["mode"], "created": now_iso(), "seconds": None,
            "combo": fc["combo"], "params": fc["params"], "theta": fc["params"]["theta"],
            "notebook": {"seam_check_passed": seam_s.get("passed"), "seam_check_tol": seam_s.get("tol"),
                         "seam_check_bitwise_equal": seam_s.get("bitwise_equal"),
                         "run_log_status": tb.run_status(nb["log"]),
                         "earlier_failed_runs": [{"run": n, "status": s} for n, s in tb.earlier_failures(nb["log"])],
                         "gpu": cfg.get("gpu"), "torch": cfg.get("torch"), "cudnn": cfg.get("cudnn"),
                         "config_sha256": nb["config_sha256"], "notebook_sha256": cfg.get("notebook_sha256"),
                         "same_as_repo_notebook": nb["same"], "weights_sha256": nb["weights_sha256"],
                         "test_bundle_equal_to_test02": True, "local_recompute_passed": recompute.get("passed"),
                         "keys_equal_to_test02": dump_rec["keys_equal"], "env_same": dump_rec["env_same"],
                         "env": dump_rec["env"], "identity": nb["identity"]},
            "seam_local": {"passed": sc["passed"], "tol": sc["tol"], "max_abs_diff": sc["max_abs_diff"],
                           "rule_b_differences": sc["rule_b_differences"],
                           "byte_identical_probs": sc["byte_identical_probs"],
                           "byte_identical_pred_B": sc["byte_identical_pred_B"], "warnings": warnings,
                           "file": "seam_val/seam_local.json", "sha256": pin_sha(seam_local_path)},
            "replay": {"seam_val": {"slots": opp["sets"]["seam_val"]["slots"],
                                    "failures": opp["sets"]["seam_val"]["failures"],
                                    "equal_to_committed_val_o4": (opp.get("seam_val") or {}).get("file_sha256_equal")},
                       **{s: {"slots": opp["sets"][s]["slots"], "failures": opp["sets"][s]["failures"]}
                          for s in TEST_SETS},
                       "max_rate": MAX_RATE, "opp_report_sha256": pin_sha(opp_path)},
            "colour_check": colour,
            "usable": None, "unusable_reasons": None, "check_submission": None, "outputs_sha256": None,
            "sets": sets_info,
            "questions_without_parsed_game": {s: sets_info[s]["questions_without_parsed_game_ids"] for s in TEST_SETS},
            "missing_fill": decide.FALLBACK,
            "games_parsed": {s: sets_info[s]["games_parsed"] for s in TEST_SETS},
            "decide_summary": None, "level_summary": None,
            "inputs_sha256": self.inputs_sha(t02, opp_path, nb, vk, fzs, fzt),
            "test02": {"fallback_used": t02["report"].get("fallback_used"), "t1_verdict": (t02["t1"] or {}).get("verdict"),
                       "r20_first_upload_id": t02["r20"].get("id"), "apply_test_report_sha256": t02["report_sha256"]},
            "inputs_bound": bound["sha256"], "inputs_bound_paths": bound["paths"],
            "inputs_bound_rule": ".csv / .json: sha256 with CRLF folded to LF; other files (.npz): raw bytes",
            "attempt": {"number": unp.get("attempt_number"), "counted_archives": counted,
                        "archives": [x["name"] for x in attempts],
                        "bug_revisions": [b for b in [self.bug] + [(x["record"] or {}).get("bug_revision")
                                                                   for x in attempts] if b]},
            "rule_b_differences_vs_r20": None,
            "checks": self.pre, "rerun": rerun_record(self.a, self.replaced, self.bug),
            "paths": paths_record(ctx), "tool": tool_record(), "environment": environment(), "git_head": git_head(),
            "record_only": "level_summary, per_k, bootstrap intervals, by_colour and the seam_val SWA / SWB dkappa are "
                           "recorded; nothing is chosen from them"}

    def inputs_bound(self, t02, opp_path, seam_local_path):
        """The files official_log re-hashes (keys: the real-mode repository paths)."""
        ctx = self.ctx
        items = [("results/test04/opp/opp_report.json", opp_path),
                 ("results/test04/unpack_sw.json", ctx["unpack"]),
                 ("results/test04/sw/config.json", os.path.join(ctx["sw"], "config.json")),
                 ("results/test04/sw/seam_check.json", os.path.join(ctx["sw"], "seam_check.json")),
                 ("results/test04/rank/seam_val/seam_local.json", seam_local_path),
                 ("results/test02/rank/apply_test_report.json", t02["report_path"]),
                 ("results/test02/rank/opening_test.npz", t02["opening_path"]),
                 ("results/test02/checks/rank/t1.json", os.path.join(ctx["C"], "rank", "t1.json")),
                 ("results/prereg04/rank/frozen_config.json", FROZEN_CONFIG)]
        for s in SETS:
            for f in ("val_keys.npz", "val_%s.npz" % C5_DUMP):
                items.append((f"results/test02/rank/{s}/dumps/{f}", os.path.join(ctx["D2"], "rank", s, "dumps", f)))
        for s in SETS:
            items.append((f"results/test04/opp/o4_{s}.npz", os.path.join(ctx["opp"], f"o4_{s}.npz")))
            for f in ["val_keys.npz"] + [f"val_p03{x}_{v}.npz" for x in ARMS for v in ("d4", "s0_swap")]:
                items.append((f"results/test04/sw/{s}/dumps/{f}", os.path.join(ctx["sw"], s, "dumps", f)))
        out, paths = {}, {}
        for key, p in items:
            if os.path.exists(p):
                out[key] = pin_sha(p)
                paths[key] = shown(p)
        return {"sha256": out, "paths": paths}

    def inputs_sha(self, t02, opp_path, nb, vk, fzs, fzt):
        ctx = self.ctx
        out = {"frozen_config.json": pin_sha(FROZEN_CONFIG), "opp_report.json": pin_sha(opp_path),
               "unpack_sw.json": pin_sha(ctx["unpack"]), "sw/config.json": nb["config_sha256"],
               "sw/seam_check.json": pin_sha(os.path.join(ctx["sw"], "seam_check.json")),
               "test02 apply_test_report.json": t02["report_sha256"], "opening_test.npz": t02["opening"]["sha256"],
               "test02 rank config.json": t02["config_sha256"]}
        t1p = os.path.join(ctx["C"], "rank", "t1.json")
        if os.path.exists(t1p):
            out["t1.json"] = pin_sha(t1p)
        for s in SETS:
            out[f"o4_{s}.npz"] = sha_raw(os.path.join(ctx["opp"], f"o4_{s}.npz"))
            out[f"{s} val_keys (test02)"] = vk[s]["sha256"]
            out[f"{s} val_{C5_DUMP} (test02)"] = (fzs if s == "seam_val" else fzt[s])["c5_sha256"]
            out[f"{s} questions"] = sha_raw(t02["qfiles"][s])
            for x in ARMS:
                for v in ("d4", "s0_swap"):
                    out[f"{s}/val_p03{x}_{v}.npz"] = sha_raw(os.path.join(ctx["sw"], s, "dumps", f"val_p03{x}_{v}.npz"))
        for n in P04_SEAM_PINS:
            out[f"prereg04 {n}"] = pin_sha(os.path.join(P04_RANK, n))
        return out


def cmd_apply(a):
    t0 = time.time()
    ctx = resolve(a, "apply")
    check_k6(ctx)
    run = Apply(ctx, a, t0)
    try:
        return run.go()
    except T04Error as e:
        if e.code in (1, 4):
            run.write_verdict(e)
        raise


# ================================================================ status (read only)
def committed_now(path):
    if not os.path.exists(path) or not t2.is_inside(path, REPO):
        return False
    rc, _ = git("ls-files", "--error-unmatch", "--", t2.repo_rel(path))
    rc2, st = git("status", "--porcelain", "--", t2.repo_rel(path))
    return rc == 0 and rc2 == 0 and not st


def cmd_status(a):
    ctx = resolve(a, "status")
    lines = [f"test04 status ({ctx['mode']}; {ctx['R4']})"]

    def safe(f):
        try:
            return f()
        except (T04Error,) + FOREIGN_ERRORS as e:
            return f"({e})"

    if ctx["D2"] and ctx["O2"] and ctx["C"]:
        t1p = os.path.join(ctx["C"], "rank", "t1.json")
        t1 = safe(lambda: read_json(t1p).get("verdict")) if os.path.exists(t1p) else "absent"
        rp = os.path.join(ctx["O2"], "apply_test_report.json")
        lines.append(f"R20: T1 {t1}; test02 apply_test_report.json "
                     f"{'present' if os.path.exists(rp) else 'absent'}"
                     + (f", committed {committed_now(rp)}" if ctx["real"] else ""))
        if ctx["real"] and os.path.exists(rp):
            r20 = safe(lambda: r20_first_upload(ctx, read_json(rp)))
            lines.append(f"  R20 first upload: {r20 if isinstance(r20, str) else '#%s at %s' % (r20['id'], r20['uploaded_at'])}")
    op = os.path.join(ctx["opp"], "opp_report.json")
    if os.path.exists(op):
        r = safe(lambda: read_json(op))
        if isinstance(r, dict):
            lines.append(f"opp: exit {r.get('exit_code')}, passed {r.get('passed')}, usable_so_far {r.get('usable_so_far')}"
                         + (f", committed {committed_now(op)}" if ctx["real"] else ""))
            lines.append("  replay failures: " + ", ".join(
                f"{s} {((r.get('sets') or {}).get(s) or {}).get('failures')}/{((r.get('sets') or {}).get(s) or {}).get('slots')}"
                for s in SETS))
            lines.append("  O4 dkappa early: " + ", ".join(
                f"{s} {((r.get('o4_dkappa_early') or {}).get(s) or {}).get('dkappa')!r}" for s in TEST_SETS))
    else:
        lines.append("opp: not run")
    if os.path.isdir(ctx["sw"]):
        log = safe(lambda: read_json(os.path.join(ctx["sw"], "run_log.json")))
        seam = safe(lambda: read_json(os.path.join(ctx["sw"], "seam_check.json")))
        cfg = safe(lambda: read_json(os.path.join(ctx["sw"], "config.json")))
        lines.append(f"sw: run_log {tb.run_status(log) if isinstance(log, dict) else log}; notebook seam "
                     f"{seam.get('passed') if isinstance(seam, dict) else seam}; GPU "
                     f"{cfg.get('gpu') if isinstance(cfg, dict) else cfg}"
                     + (f"; committed {committed_now(os.path.join(ctx['sw'], 'config.json'))}" if ctx["real"] else ""))
    else:
        lines.append("sw: not unpacked")
    rp = os.path.join(ctx["rank"], "t04_apply_report.json")
    vp = os.path.join(ctx["rank"], "attempt_verdict.json")
    if os.path.exists(rp):
        r = safe(lambda: read_json(rp))
        if isinstance(r, dict):
            lines.append(f"apply: usable {r.get('usable')} {r.get('unusable_reasons') or ''}; submission sha256 "
                         f"{str((r.get('outputs_sha256') or {}).get('submission_rank.csv'))[:12]}")
            for s in TEST_SETS:
                lines.append(f"  {s}: " + ", ".join(
                    f"{src} {(((r.get('colour_check') or {}).get(s) or {}).get(src) or {}).get('dkappa')!r}"
                    for src in OPP_SOURCES))
    elif os.path.exists(vp):
        v = safe(lambda: read_json(vp))
        if isinstance(v, dict):
            lines.append(f"apply: attempt_verdict exit {v.get('exit_code')} at {v.get('stage')} ({v.get('reason')})")
    else:
        lines.append("apply: not run")
    att = safe(lambda: list_attempts(ctx))
    if isinstance(att, list):
        lines.append(f"failed attempts: {len(att)} ({sum(1 for x in att if is_counted(x))} counted toward the one "
                     f"rerun): {[x['name'] for x in att]}")
    left = K6 - now()
    lines.append(f"K6 {K6.isoformat()}: " + (f"{left.days} days {left.seconds // 3600} h left" if left.total_seconds() > 0
                                            else "passed (no 04 this season)"))
    print("\n".join(lines))
    return 0


# ================================================================ archive
def archive_evidence(ctx):
    """Records of the attempt to archive: from R4/sw when unpacked, else from sw/*.json inside the R4 zips."""
    sw, zips = ctx["sw"], sorted(glob.glob(os.path.join(ctx["R4"], "*.zip")))
    src, log, seam, cfg, markers = None, None, None, None, {}
    if os.path.isdir(sw) and os.path.exists(os.path.join(sw, "run_log.json")):
        src = shown(sw)
        log = read_json(os.path.join(sw, "run_log.json"))
        seam = read_json_or_none(os.path.join(sw, "seam_check.json"))
        cfg = read_json_or_none(os.path.join(sw, "config.json"))
        markers = done_markers_from_dir(sw)
    else:
        preferred = sorted(zips, key=lambda p: (os.path.basename(p) != "test04_rank_results.zip", os.path.basename(p)))
        for zp in preferred:
            try:
                with zipfile.ZipFile(zp) as z:
                    names = set(z.namelist())
                    if f"{SW_ROOT}/run_log.json" not in names:
                        continue

                    def zj(n):
                        return json.loads(z.read(n).decode("utf-8")) if n in names else None

                    log, seam, cfg = zj(f"{SW_ROOT}/run_log.json"), zj(f"{SW_ROOT}/seam_check.json"), \
                        zj(f"{SW_ROOT}/config.json")
                    markers = {n[len(SW_ROOT) + 1:]: zj(n) for n in names if os.path.basename(n).startswith("_DONE_")}
                    src = shown(zp)
                    break
            except (zipfile.BadZipFile, ValueError, KeyError):
                continue
    return {"source": src, "run_log": log, "seam": seam, "config": cfg, "markers": markers, "zips": zips}


def cmd_archive(a):
    t0 = time.time()
    ctx = resolve(a, "archive")
    if not (a.reason or "").strip():
        refuse("archive needs --reason TEXT (recorded in attempt.json)")
    drive_from, drive_zip_from = drive_source_names(a.drive_folder)
    R4 = ctx["R4"]
    ev = archive_evidence(ctx)
    moved = [p for p in (ctx["sw"], ctx["unpack"], ctx["rank"]) if os.path.exists(p)]
    if not moved and not ev["zips"]:
        refuse(f"nothing to archive in {R4} (no sw/, unpack_sw.json, rank/ or *.zip)")
    attempts = list_attempts(ctx)
    counted = [x for x in attempts if is_counted(x)]
    log, seam, cfg, markers = ev["run_log"] or {}, ev["seam"], ev["config"] or {}, ev["markers"]
    classes = []
    status = tb.run_status(log) if log else None
    if seam is not None and seam.get("passed") is not True:
        classes.append("notebook_seam")          # the cause first: a failed seam also leaves run_log seam_failed
    if status in SEAM_FAILED_STATUSES or (log and tb.earlier_failures(log)):
        classes.append("notebook_run_log")
    if ctx["real"] and cfg:
        # review round 2: judged against the committed notebook only (a local edit or a stale checkout is a refusal,
        # never a counted not_committed_notebook archive)
        require_committed([NOTEBOOK04], NOTEBOOK_REL)
        same, _ = notebook_identity(cfg, "real")
        if same is False:
            classes.append("not_committed_notebook")
    if log and resumed_markers(markers, log):
        classes.append("resumed")
    ver = read_json_or_none(os.path.join(ctx["rank"], "attempt_verdict.json"))
    if ver is not None and ver.get("exit_code") == 1:
        classes.append("apply_exit_1")
    before = None
    if ctx["real"] and log:
        # review round 1: a failed check of R20's first upload is a refusal, never evidence of "run before R20"
        # (an uncommitted log used to turn a counted seam failure into an uncounted before_r20_u1 archive)
        try:
            r20 = r20_first_upload(ctx, t02_report_for_r20(ctx))
        except T04Error as e:
            refuse(f"{MSG_R20_UNCHECKED}\n - {e}")
        before = r20_before_problems(r20, log, markers)
        if before:
            classes.insert(0, "before_r20_u1")
    rep = read_json_or_none(os.path.join(ctx["rank"], "t04_apply_report.json"))
    opp = read_json_or_none(os.path.join(ctx["opp"], "opp_report.json"))
    refusals, overturned = [], []
    if rep is not None:
        refusals.append("results/test04/rank/t04_apply_report.json exists: this attempt already has the registered "
                        "decision (not a seam failure)")
        overturned.append(("t04_apply_report.json created", rep.get("created")))
    if opp is not None and opp.get("usable_so_far") is False and verdict_kind(opp) == "drift":
        # rev 3 check: a local drift of opp (exit 4) is not the registered 'no 04'; it blocks archive until opp is
        # rerun, and a PREREG_04 revision does not lift it (like the R20 and notebook checks above)
        refuse(MSG_OPP_DRIFT_ARCHIVE + f"\n - unusable_reasons {opp.get('unusable_reasons')}")
    if opp is not None and opp.get("usable_so_far") is False:
        refusals.append("opp_report.json says usable_so_far false: opp's registered decision is 'no 04', nothing to "
                        "rerun")
        overturned.append(("opp_report.json created", opp.get("created")))
    seam_class = [c for c in classes if c in COUNTED_CLASSES]
    if not seam_class and "before_r20_u1" not in classes:
        refusals.append("no seam-class evidence (run_log, seam_check, notebook identity, resumed markers, apply exit 1, "
                        "or a run before R20's first upload)")
    if seam_class and counted:     # review round 1: before_r20_u1 no longer skips this refusal
        refusals.append(f"{len(counted)} counted archive(s) already ({[x['name'] for x in counted]}): " + MSG_ONE_RERUN)
        overturned.append((f"{counted[-1]['name']} archived_at", (counted[-1]["record"] or {}).get("archived_at")))
    unp = read_json_or_none(ctx["unpack"]) or {}
    bug = None
    if a.bug_revision is not None:
        # review round 2 (rev 3 item 20): the revision must come after THIS failure (its run, markers, unpack record and
        # apply verdict), not only after the records it overrides, and one revision exchanges one uncounted rerun only
        reused = [x["name"] for x in attempts
                  if bug_revision_number((x["record"] or {}).get("bug_revision")) == a.bug_revision]
        if reused:
            refuse(f"PREREG_04 rev {a.bug_revision} was already used by the archive {reused}: one revision exchanges one "
                   f"uncounted rerun; a new error needs a new revision (rev {a.bug_revision + 1} or later)")
        own = attempt_times(log, markers, unp, ver)
        if not own:
            refuse(f"no time of this attempt was found (run_log, _DONE_ markers, unpack_sw.json, attempt_verdict.json): "
                   f"PREREG_04 rev {a.bug_revision} cannot be shown to come after this failure")
        bug = check_bug_revision(a.bug_revision, overturned + own)
    elif refusals:
        refuse("archive refused:\n - " + "\n - ".join(refusals))
    failure_class = "bug_revision" if bug else ("before_r20_u1" if "before_r20_u1" in classes else seam_class[0])
    counts = failure_class in COUNTED_CLASSES
    n = max([x["n"] for x in attempts], default=0) + 1
    drive_to, drive_zip_to = drive_archive_names(n, counts)
    dest = os.path.join(ctx["failed"], f"attempt_{n}")
    zdest = os.path.join(ctx["failed_zips"], f"attempt_{n}")
    if os.path.exists(dest) or os.path.exists(zdest):
        refuse(f"{dest} or {zdest} already exists")
    zip_sha = {os.path.basename(p): sha_raw(p) for p in ev["zips"]}
    # review round 1: the zip unpack read may lie outside R4 (unpack --zip <any path>); its SHA-256 is archived too,
    # so that the freshness check of the next unpack refuses it wherever it is downloaded again
    uz = unp.get("zip") or {}
    if uz.get("sha256") and uz["sha256"] not in zip_sha.values():
        zip_sha["unpacked:" + str(uz.get("name") or os.path.basename(str(uz.get("path") or "zip")))] = uz["sha256"]
    done_times = {os.path.basename(k): (m or {}).get("time") for k, m in markers.items() if k.startswith("seam_val/")}
    seam_local = read_json_or_none(os.path.join(ctx["rank"], "seam_val", "seam_local.json"))
    guard_write(ctx, dest)
    os.makedirs(dest)
    for p in moved:
        guard_write(ctx, p)
        shutil.move(p, os.path.join(dest, os.path.basename(p)))
    if ev["zips"]:
        guard_write(ctx, zdest)
        os.makedirs(zdest)
        for p in ev["zips"]:
            shutil.move(p, os.path.join(zdest, os.path.basename(p)))
    # K5 gate round 1: the next run is the registered rerun (G4 only) whenever a counted archive exists after this one,
    # also after an uncounted one (bug_revision / before_r20_u1) that follows a counted archive: the notebook (Drive
    # sw_failed_*), unpack (len(counted) == 1) and official_log (attempt >= 2) all ask G4 then
    if counts:
        nxt = "登記的重跑，只用 G4（PREREG_04 rev 3 第 4、9 條）"
    elif counted:
        nxt = ("不占重跑次數（" + failure_class + "），但已有 1 個占次數的封存 " + str([x["name"] for x in counted])
               + "（Drive 上有 sw_failed_*）：下一次仍是登記的重跑，只用 G4（PREREG_04 rev 3 第 4、9 條）")
    else:
        nxt = "不占重跑次數（" + failure_class + "），GPU 照第一次的規則：G4 優先、A100 備用"
    rec = {"format": FORMATS["attempt"], "mode": ctx["mode"], "attempt": n, "reason": a.reason.strip(),
           "failure_class": failure_class, "classes_found": classes, "counts_toward_rerun": counts,
           "bug_revision": bug, "archived_at": now_iso(), "evidence_source": ev["source"],
           "zips": zip_sha, "zips_moved_to": shown(zdest) if ev["zips"] else None,
           "unpacked_zip": ({"path": uz.get("path"), "sha256": uz.get("sha256")} if uz else None),
           "moved": [shown(os.path.join(dest, os.path.basename(p))) for p in moved],
           "runs": run_starts(log), "run_log_status": status, "seam_check_passed": (seam or {}).get("passed"),
           "seam_local_passed": (seam_local or {}).get("passed"), "done_times": done_times,
           "gpu": cfg.get("gpu"), "before_r20_u1_problems": before, "refusals_overridden": refusals if bug else [],
           "apply_verdict": ver, "tool": tool_record(), "environment": environment(),
           "drive_rename": {"folder_from": drive_from, "folder_to": drive_to, "zip_from": drive_zip_from,
                            "zip_to": drive_zip_to,
                            "rule": "review round 3: counted -> sw_failed_<N>, uncounted -> sw_exempt_<N>"},
           "next_run": {"registered_rerun_g4_only": bool(counts or counted), "text": nxt,
                        "rule": "K5 gate round 1: G4 only while a counted archive exists (rev 3 items 4, 9)"},
           "seconds": round(time.time() - t0, 1)}
    write_json(ctx, os.path.join(dest, "attempt.json"), rec)
    print(f"archive: attempt {n} ({failure_class}, counts toward the rerun: {counts}) -> {dest}"
          + (f"; zips -> {zdest}" if ev["zips"] else ""))
    print(f"Drive: {drive_from} -> {drive_to}, {drive_zip_from} -> {drive_zip_to}")
    if ctx["real"]:
        print(f"commit: results/test04/failed/attempt_{n}/ (and the removal of results/test04/sw, unpack_sw.json, "
              f"rank/)")
        print(f"給使用者：Drive 的 C10/results/test04/{drive_from} 改名成 {drive_to}，同一層的 {drive_zip_from} 改名成 "
              f"{drive_zip_to}（沒有這個 zip 就只改資料夾）；電腦「下載」裡的舊 zip 刪掉")
        print("下一次執行：" + nxt)
    return 0


# ================================================================ command line
def build_parser():
    ap = argparse.ArgumentParser(prog="t04_rank.py", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["unpack", "opp", "apply", "status", "archive"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--results-dir")
    ap.add_argument("--t02-results-dir")
    ap.add_argument("--t02-out-dir")
    ap.add_argument("--tests-dir")
    ap.add_argument("--checks-dir")
    ap.add_argument("--zip")
    ap.add_argument("--allow-failed", action="store_true")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--reason")
    ap.add_argument("--bug-revision", type=int)
    ap.add_argument("--drive-folder", help="archive only: the Drive folder to rename (sw, default; or sw_interrupted_K)")
    return ap


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    try:
        if a.reason is not None and not a.reason.strip():
            refuse("--reason must not be empty")
        if a.rerun and not a.reason:
            refuse("--rerun needs --reason TEXT (stored in the report)")
        if a.command != "unpack" and (a.zip or a.allow_failed):
            refuse("--zip and --allow-failed belong to unpack")
        if a.command == "unpack" and not a.zip:
            refuse("unpack needs --zip Z")
        if a.command != "opp" and a.workers is not None:
            refuse("--workers belongs to opp")
        if a.command in ("unpack", "status") and (a.rerun or a.bug_revision is not None):
            refuse(f"{a.command} takes no --rerun / --bug-revision (to start over, archive first)")
        if a.command in ("opp", "apply") and a.bug_revision is not None and not a.rerun:
            refuse("--bug-revision N goes with --rerun --reason TEXT")
        if a.command == "archive" and a.rerun:
            refuse("archive takes --reason TEXT [--bug-revision N], no --rerun")
        if a.command != "archive" and a.drive_folder is not None:
            refuse("--drive-folder belongs to archive")
        return {"unpack": cmd_unpack, "opp": cmd_opp, "apply": cmd_apply, "status": cmd_status,
                "archive": cmd_archive}[a.command](a)
    except T04Error as e:
        print(f"ERROR (exit {e.code}): {e}", file=sys.stderr)
        return e.code
    except FOREIGN_ERRORS as e:
        print(f"ERROR (exit 2): {e}", file=sys.stderr)
        return 2
    except tb.BundleError as e:
        print(f"ERROR (exit 2): {e}", file=sys.stderr)
        return 2
    except Exception as e:     # review round 1: never Python's default exit 1 (1 means a seam-class failure)
        traceback.print_exc()
        print(f"ERROR (exit 2): unexpected internal error {type(e).__name__}: {e}. This is not a seam failure: do not "
              f"archive; stop and check the local tool (docs/TESTDAY_04.md)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
