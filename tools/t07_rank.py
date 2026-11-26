"""C10 test07, competition 2 (rank, TEAM_11248): the 07 test-day local tool (Q02H = Q02 + KataGo HumanSL).

Registration: docs/PREREG_07_HUMANSL.md section 5 (rev 1), rev 3 (the adoption line: "H7R scored, Q02H >= 0.57,
reference re-read equal") and the test-day revision (rev 4). Spec: results/prereg07/design/spec.md sections 0, 3.3,
7, 9 (the registration wins over the spec).

    GPY tools/t07_rank.py hs --set seam|public|private [MODE] [DIRS] [--workers W] [--stop-after-min M]
                                                         [--rerun --reason TEXT]       (RTX 4090, .venv-gpu)
    PY  tools/t07_rank.py apply  [MODE] [DIRS] [--rerun --reason TEXT [--bug-revision N]]   (CPU, .venv)
    PY  tools/t07_rank.py status [MODE] [DIRS]                                              (read only)
    (PY = .venv/Scripts/python.exe, GPY = .venv-gpu/Scripts/python.exe; Git Bash; PYTHONDONTWRITEBYTECODE=1)

MODE  none = real; at most one of --rehearsal, --dry-run
DIRS  --results-dir R7      test07 root (writes R7/hs, R7/rank; real results/test07)
      --hs-data-dir H       the HS chunks (real data/test07/hs; --rehearsal data/test07_rehearsal/hs; --dry-run R7/hs_data)
      --t04-results-dir R4  test04 root (reads R4/opp, R4/sw, R4/rank; real results/test04)
      --t02-results-dir D2, --t02-out-dir O2, --tests-dir T, --checks-dir C   as tools/t04_rank.py
Real mode: every directory is the repository path above. --rehearsal: R7 is results/test07_rehearsal (or outside
the repository); --dry-run: R7 outside the repository. The other trees are required outside the real mode.
Drills only (never real): --r20-uploaded-at ISO (R20's first upload time the stale-chunk rules compare with),
--max-games N, --fake-scorer (a deterministic CPU stand-in for the network), --ignore-other-gpu-jobs, --now ISO.

hs     The KataGo HumanSL per-game sums of one set on the local RTX 4090 (PREREG_07 5.2; spec 7.3). seam = the first
       400 games of the validation questions in val_keys order (data/local/rank_val_questions.csv: question id, slot,
       colour and SGF only; registered SHA-256); public / private = test02's packed test files (T, never data/local or
       T/raw), slot by slot equal to test02's val_keys.npz of the set. The question file is read before hs_lib is
       imported (its guard closes data/local for the rest of the process). Chunks of 200 games in H/<set>/ are written
       atomically and resumed (STOP file in H/<set>/ or --stop-after-min end the run after the chunk in progress);
       a chunk with another environment is refused (4), a chunk older than R20's first upload is refused (2: rename
       the folder to H/<set>_stale_<time>/, never deleted by the tool). When every chunk is there: R7/hs/<set>_hs.npz
       (seam: also the per-move log p) and R7/hs/hs_<set>_report.json. Commit them.
apply  Q02H on public and private (PREREG_07 5.2; spec 7.4), in this order (the first failing item decides): the three
       HS sets (committed, chunk SHA-256s, one environment, every chunk later than R20's first upload); the GPU seam
       against results/prereg07/hs/val_hs.npz rows 0-399 and val_seam_moves.npz (failed / counts / n_moves equal,
       per move <= 0.05, per game and profile mean <= 0.002, per game lp_HS <= 0.02); Q02 recomputed with
       tools/t04_rank.py fuse_set equal byte for byte to test04's rank_test_probs.csv, and every input test04 bound
       unchanged; the move cross-check against test04's o4_<set>.npz; the local Q02H seam on the 1,589 validation
       questions against results/prereg07/rank/rank_val_probs.csv; HS failures (200 x failed <= games per set); the
       HS direction check (rho >= max(0.5 x rho_mono, 0.155) per set). Then rank_test_probs.csv, pred_A.csv,
       pred_B.csv, decide_summary.json, submission_rank.csv (= pred_B.csv), check_submission --official and
       R7/rank/t07_apply_report.json.
status Read-only summary.

Exit codes (spec 7.1): 0 done; 2 refused or error (fix the precondition, rerun the same line; an unexpected internal
error is also 2 with a traceback); 3 the registered "07 is not used" (R7/not_used.json; commit; final per PREREG_04
5.3; never rerun); 4 local drift or mismatch (R7/rank/attempt_verdict.json; commit, stop and find out why, then
--rerun --reason; a move cross-check mismatch needs a committed PREREG_07 revision naming the program error first).
There is no exit 1: 07 has no notebook. After exit 3 or a written t07_apply_report.json nothing is rerun except with
--bug-revision N (a committed PREREG_07 rev N >= 5 added after that record).

Discipline: never imports u1_lib, never calls the inner mapping fit, imports hs_lib / pilot_run / pilot_fuse only
after every data/local read of the process; no validation label file is ever opened (any path whose lower-case text
holds that word is refused); tools/score.py is never run; temporary files only under data/test07/tmp (real),
data/test07_rehearsal/tmp or R7/tmp. Competition 1 is never read (PREREG_07 6).
"""

import argparse
import datetime as dt
import glob
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SELF = os.path.join(HERE, "t07_rank.py")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p07_rank_fuse as pf7  # noqa: E402
import t04_rank as t4  # noqa: E402

rf, p3, po, pf4, t2, tb = t4.rf, t4.p3, t4.po, t4.pf, t4.t2, t4.tb
decide, check_submission = t4.decide, t4.check_submission

# ================================================================ paths (module constants; the unit tests patch them)
RESULTS = os.path.join(REPO, "results")
REAL_R7 = os.path.join(RESULTS, "test07")
REHEARSAL_R7 = os.path.join(RESULTS, "test07_rehearsal")
REAL_H = os.path.join(REPO, "data", "test07", "hs")
REHEARSAL_H = os.path.join(REPO, "data", "test07_rehearsal", "hs")
REAL_TMP = os.path.join(REPO, "data", "test07", "tmp")
REHEARSAL_TMP = os.path.join(REPO, "data", "test07_rehearsal", "tmp")
REAL_R4 = t4.REAL_R4
REAL_D2, REAL_O2, REAL_T, REAL_C = t4.REAL_D2, t4.REAL_O2, t4.REAL_T, t4.REAL_C
LOCAL_DIR = os.path.join(REPO, "data", "local")
SEAM_QUESTIONS = os.path.join(LOCAL_DIR, "rank_val_questions.csv")
P07_RANK = os.path.join(RESULTS, "prereg07", "rank")
P07_HS = os.path.join(RESULTS, "prereg07", "hs")
P07_DOC = os.path.join(REPO, "docs", "PREREG_07_HUMANSL.md")
P07_RESULTS = os.path.join(RESULTS, "prereg07", "RESULTS.md")
OFFICIAL_LOG = os.path.join(RESULTS, "official_scores.json")
VAL_KEYS = pf7.VAL_KEYS

# ================================================================ registered values
# the committed prereg07 files the test day reads (SHA-256: .json / .csv with CRLF folded to LF, .npz raw bytes)
P07_PINS = {
    "results/prereg07/rank/frozen_config.json": "93a0ac29615260865d554ee61aec258c633820a87aeb7dfcf8047a5dfd200c0b",
    "results/prereg07/rank/hs_map.npz": "a4334e79c490e4d8102dd5136ac08c0de1060315981f4ccc527ebffa1b84a3e4",
    "results/prereg07/hs/val_hs.npz": "ef2067e8cb633f521c205a09aa4638c79fef3ba0b4e90e6a54bdc046e5a5550f",
    "results/prereg07/hs/val_seam_moves.npz": "3e55fbfa70b43c36c2ebe30bdbca63ea6c68968a1521ec6d63fd2b698f6598ab",
    "results/prereg07/hs/val_report.json": "11745c5c6e8fedd7c01611331b4f4c69ec1c00f423d3c39e267823bc3c56c7ca",
    "results/prereg07/rank/rank_val_probs.csv": "6ae2db6025a903eb5c58af9418dc203c0b6e84dd664049f85dfa850e57dc28c1",
    "results/prereg07/rank/pred_B.csv": "51d7dbf82d50c3848acadb6934069580bd3ade7e6ca71ad5d1023633a18a26fa",
    "results/prereg07/rank/pred_A.csv": "b229be2c414f7af22310bc304d6da9de5b6406698ec8ce9c0d40e7c146b40a41",
    "results/prereg07/rank/apply_report.json": "3ed7f7b0580f02d9c4c9c9255634a689213017e65190dce492465aad332fad6b",
    "results/prereg07/rank/score_H7R.json": "32686f6c91fb0a14301c85ecc1c299844d9346fb2efb8f0fd0b5891f5609cbf5"}
# PREREG_07 rev 3: "H7R scored, Q02H >= 0.57, reference re-read equal" replaces "H7R adopted"
H7R_A_RANK_BALANCED = 0.6236017652948923          # prereg04 H4R, the B row (PREREG_07 3)
H7R_A_MEAN = 0.6245054776753989
H7R_QUESTIONS = 1589
H7R_FLOOR = 0.57
H7R_TOL = 1e-12
RHO_MONO = 0.9502568925299116                     # frozen_config.json hs_direction_ref.rho_mono
RHO_FLOOR = pf7.RHO_FLOOR_TEST                    # 0.155: one-sided p = 0.001 for 400 questions
SEAM_GAMES = 400
TOL_MOVE = 0.05                                   # PREREG_07 5.2 GPU seam
TOL_PROFILE_MEAN = 0.002
TOL_LP = 0.02
LP_REVIEW = 0.005                                 # spec 9: a measured lp_HS difference above this is reviewed
GPU_NAME = "NVIDIA GeForce RTX 4090"
TW = dt.timezone(dt.timedelta(hours=8))
K6_07 = dt.datetime(2026, 11, 19, 23, 59, 59, tzinfo=TW)
P07_MIN_REV = 5                                   # rev 4 registers the test day; an override is rev 5 or later
SETS = ("seam", "public", "private")
TEST_SETS = ("public", "private")
Q_SETS = ("seam_val", "public", "private")        # tools/t04_rank.py set names
COMBO = {"id": "Q02H", "sources": ["C5", "OP", "O4", "SWA", "SWB", "HS"], "terms": ["OP", "O4", "SWA", "SWB", "HS"]}
METHOD = "prereg07 Q02H = Q02 + HumanSL, rule B (甲)"
CHUNK_FORMAT = pf7.PILOT_FORMAT
FORMATS = {"hs": "c10-test07 hs v1", "hs_report": "c10-test07 hs report v1", "gpu_seam": "c10-test07 gpu seam v1",
           "seam_local": "c10-test07 seam_local v1", "apply": "c10-test07 rank apply v1",
           "verdict": "c10-test07 attempt verdict v1", "not_used": "c10-test07 not used v1"}
TEST_OUTPUTS = ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv")
_LABEL = "ans" + "wer"


class T07Error(RuntimeError):
    """A stop with its exit code (spec 7.1): 2 refused, 3 registered 'no 07', 4 local drift / mismatch."""

    def __init__(self, message, code=2, stage=None, reason=None, details=None):
        super().__init__(message)
        self.code, self.stage, self.reason, self.details = code, stage, reason, details


def refuse(msg, reason=None):
    raise T07Error(msg, 2, reason=reason)


def not_used(msg, stage, reason, details=None):
    raise T07Error(msg, 3, stage, reason, details)


def drift(msg, stage, reason, details=None):
    raise T07Error(msg, 4, stage, reason, details)


FOREIGN_ERRORS = tuple(dict.fromkeys(t4.FOREIGN_ERRORS + pf7.ERRORS))


# ================================================================ small helpers
def now():
    """The current time (a module function: the tests replace it to cross K6-07)."""
    return dt.datetime.now(TW)


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_iso(text, what):
    try:
        t = dt.datetime.fromisoformat(str(text).strip())
    except ValueError:
        refuse(f"{what} {text!r} is not an ISO time")
    if t.tzinfo is None:
        t = t.replace(tzinfo=TW)
    return t


def refuse_label_path(path):
    if path is not None and _LABEL in os.path.abspath(os.fspath(path)).lower():
        refuse(f"refusing {path}: validation label files are never opened by the test07 tools")
    return path


def sha_raw(path):
    return t2.sha256_file(refuse_label_path(path))


def sha_lf(path):
    return t2.sha256_text(refuse_label_path(path))


def pin_sha(path):
    """.csv / .json / .md / .txt with CRLF folded to LF, anything else (.npz) raw bytes."""
    if os.path.splitext(path)[1].lower() in (".csv", ".json", ".md", ".txt"):
        return sha_lf(path)
    return sha_raw(path)


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


def load_npz(path):
    refuse_label_path(path)
    if not os.path.exists(path):
        refuse(f"missing {shown(path)}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def jsonable(o):
    return t4.jsonable(o)


def git(*args):
    return t4.git(*args)


def git_head():
    rc, out = git("rev-parse", "HEAD")
    return out if rc == 0 else None


def path_of(rel):
    return os.path.join(REPO, *rel.split("/"))


def committed_clean(path):
    """True when path is tracked and unchanged against HEAD."""
    rel = t2.repo_rel(path) if t2.is_inside(path, REPO) else None
    if rel is None:
        return False
    rc, _ = git("ls-files", "--error-unmatch", "--", rel)
    rc2, st = git("status", "--porcelain", "--", rel)
    return rc == 0 and rc2 == 0 and not st


def require_committed(paths, what):
    bad = [shown(p) for p in paths if not committed_clean(p)]
    if bad:
        refuse(f"{what}: commit first ({bad[:6]} not committed or changed)")
    return {"committed": [shown(p) for p in paths]}


def commit_time(path):
    rc, out = git("log", "-1", "--format=%H %cI", "--", t2.repo_rel(path))
    if rc != 0 or not out:
        return None, None
    h, t = out.split()
    return h, t


def tool_files():
    names = ["t07_rank.py", "t04_rank.py", "p07_rank_fuse.py", "p07_hs.py", "check_submission.py", "official_log.py",
             "t02_rank.py", "t02_test_bundle.py", "p04_opp.py", "p04_rank_fuse.py", "p03_rank_fuse.py",
             "p02_rank_fuse.py", "c10data.py", "decide.py"]
    return {n: (sha_lf(os.path.join(HERE, n)) if os.path.exists(os.path.join(HERE, n)) else None) for n in names}


def tool_record():
    return {"git_head": git_head(), "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": tool_files()}


def rate_ok(failures, games):
    """failures / games <= 0.005, in integers (200 * failures <= games): 25 of 5,000 passes, 26 does not."""
    return isinstance(games, int) and games > 0 and isinstance(failures, int) and 0 <= failures and \
        200 * failures <= games


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ================================================================ modes and directories
def resolve(a, cmd):
    flags = [m for m, on in (("dry-run", a.dry_run), ("rehearsal", a.rehearsal)) if on]
    if len(flags) > 1:
        refuse(f"give at most one of --dry-run, --rehearsal (got {flags})")
    mode = flags[0] if flags else "real"
    given = {"--results-dir": a.results_dir, "--t04-results-dir": a.t04_results_dir,
             "--t02-results-dir": a.t02_results_dir, "--t02-out-dir": a.t02_out_dir, "--tests-dir": a.tests_dir}
    real = {"--results-dir": REAL_R7, "--t04-results-dir": REAL_R4, "--t02-results-dir": REAL_D2,
            "--t02-out-dir": REAL_O2, "--tests-dir": REAL_T}
    drills = [k for k, v in (("--r20-uploaded-at", getattr(a, "r20_uploaded_at", None)),
                             ("--max-games", getattr(a, "max_games", 0)),
                             ("--fake-scorer", getattr(a, "fake_scorer", False)),
                             ("--ignore-other-gpu-jobs", getattr(a, "ignore_other_gpu_jobs", False)),
                             ("--now", getattr(a, "now", None))) if v]
    if mode == "real":
        if drills:
            refuse(f"real mode: {drills} are for drills only (--dry-run / --rehearsal)")
        for k, v in given.items():
            if v and not t2.same_path(v, real[k]):
                refuse(f"real mode: {k} must be {real[k]} (got {v})")
        for k, v, want in (("--checks-dir", a.checks_dir, REAL_C), ("--hs-data-dir", a.hs_data_dir, REAL_H)):
            if v and not t2.same_path(v, want):
                refuse(f"real mode: {k} must be {want} (got {v})")
        R7, H, R4, D2, O2, T, C, TMP = REAL_R7, REAL_H, REAL_R4, REAL_D2, REAL_O2, REAL_T, REAL_C, REAL_TMP
    else:
        need = ["--results-dir"] if cmd == "status" else list(given)
        if mode == "rehearsal" and not a.results_dir:
            given["--results-dir"] = REHEARSAL_R7
        missing = [k for k in need if not given[k]]
        if missing:
            refuse(f"--{mode} needs {', '.join(missing)} (other modes never fall back to the repository paths)")
        R7 = os.path.abspath(given["--results-dir"])
        R4 = os.path.abspath(a.t04_results_dir) if a.t04_results_dir else None
        D2 = os.path.abspath(a.t02_results_dir) if a.t02_results_dir else None
        O2 = os.path.abspath(a.t02_out_dir) if a.t02_out_dir else None
        T = os.path.abspath(a.tests_dir) if a.tests_dir else None
        C = os.path.abspath(a.checks_dir) if a.checks_dir else (os.path.join(D2, "checks") if D2 else None)
        in_repo_ok = mode == "rehearsal" and t2.same_path(R7, REHEARSAL_R7)
        if t2.is_inside(R7, REPO) and not in_repo_ok:
            refuse(f"--{mode}: --results-dir must be outside the repository (only --rehearsal may use "
                   f"{REHEARSAL_R7}); got {R7}")
        if a.hs_data_dir:
            H = os.path.abspath(a.hs_data_dir)
        else:
            H = REHEARSAL_H if in_repo_ok else os.path.join(R7, "hs_data")
        if t2.is_inside(H, REPO) and not t2.is_inside(H, os.path.dirname(REHEARSAL_H)):
            refuse(f"--{mode}: --hs-data-dir inside the repository must be under data/test07_rehearsal (got {H})")
        TMP = REHEARSAL_TMP if in_repo_ok else os.path.join(R7, "tmp")
    for p in (R7, H, R4, D2, O2, T, C):
        if p is not None:
            refuse_label_path(p)
            if pf7.under_data_local(p):
                refuse(f"{p}: never under data/local")
    if T is not None and os.path.basename(os.path.normpath(T)).lower() == "raw":
        refuse(f"--tests-dir {T}: the test files are the checked copies (data/test), never data/test/raw")
    bad = forbidden_area(R7) or forbidden_area(H)
    if bad:
        refuse(f"output folders inside the read-only {bad}")
    return {"mode": mode, "real": mode == "real", "smoke": False, "cmd": cmd, "R7": R7, "H": H, "R4": R4, "D2": D2,
            "O2": O2, "T": T, "C": C, "TMP": TMP, "hs": os.path.join(R7, "hs"), "rank": os.path.join(R7, "rank"),
            "write_roots": [R7, H, TMP]}


def forbidden_area(path):
    """Where t07 never writes: results/prereg0*, results/test02*, results/test04*, results/test06*, data/local, the
    ledger."""
    p = os.path.abspath(path)
    if pf7.under_data_local(p):
        return "data/local"
    if t2.same_path(p, os.path.join(RESULTS, "score_ledger.csv")):
        return "results/score_ledger.csv"
    if t2.is_inside(p, RESULTS) and not t2.same_path(p, RESULTS):
        first = os.path.relpath(os.path.realpath(p), os.path.realpath(RESULTS)).split(os.sep)[0].lower()
        if first.startswith(("prereg0", "test02", "test04", "test06")):
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


def ctx4(ctx):
    """The context tools/t04_rank.py's prerequisite functions read (same mode and trees)."""
    return {"mode": ctx["mode"], "real": ctx["real"], "smoke": False, "D2": ctx["D2"], "O2": ctx["O2"],
            "T": ctx["T"], "C": ctx["C"], "R4": ctx["R4"]}


def t4call(fn, *args):
    """A tools/t04_rank.py prerequisite: its exit code is ours (2 refused, 3 registered 'no 04 / no 07', 4 drift)."""
    try:
        return fn(*args)
    except t4.T04Error as e:
        raise T07Error(f"(t04_rank) {e}", 3 if e.code == 3 else 4 if e.code == 4 else 2,
                       reason=e.reason or "t04_prerequisite")
    except FOREIGN_ERRORS as e:
        refuse(f"(t04_rank) {e}")


# ================================================================ common prerequisites (spec 7.2)
def check_git_clean(ctx, rec):
    """Item 1 (real): no uncommitted change to a tracked file; this tool and its dependencies committed."""
    if not ctx["real"]:
        rec["git"] = f"not checked ({ctx['mode']})"
        return
    rc, st = git("status", "--porcelain", "--untracked-files=no")
    if rc != 0:
        refuse("git is not available")
    if st:
        refuse(f"commit or restore the tracked changes first (git status: {st.splitlines()[:6]})")
    rec["git"] = {"clean": True, "head": git_head(),
                  "own_files": require_committed([SELF, os.path.join(HERE, "t04_rank.py"),
                                                  os.path.join(HERE, "p07_rank_fuse.py"),
                                                  os.path.join(HERE, "check_submission.py"),
                                                  os.path.join(HERE, "official_log.py")], "the test07 tools")}


def check_p07_pins():
    """Item 3: the committed prereg07 files equal their pinned SHA-256s; the four pinned pilot files and
    third_party/katago unchanged."""
    got = {r: (pin_sha(path_of(r)) if os.path.exists(path_of(r)) else None) for r in P07_PINS}
    bad = {r: got[r] for r in P07_PINS if got[r] != P07_PINS[r]}
    if bad:
        refuse(f"prereg07 files differ from the pinned SHA-256s: {sorted(bad)} (never edited after H7R)")
    try:
        pins = pf7.check_pins()
    except pf7.P07Error as e:
        refuse(str(e))
    probs, kg = pf7.katago_sources()
    if probs:
        refuse(f"third_party/katago: {probs}")
    fc = read_json(path_of("results/prereg07/rank/frozen_config.json"))
    if fc.get("combo", {}).get("id") != COMBO["id"] or fc["combo"].get("terms") != COMBO["terms"] \
            or fc.get("score_pending_val_hs") is not True:
        refuse("results/prereg07/rank/frozen_config.json is not the frozen Q02H configuration")
    if float(fc["hs_direction_ref"]["rho_mono"]) != RHO_MONO:
        refuse("frozen_config.json hs_direction_ref.rho_mono differs from the registered value")
    return fc, {"prereg07_sha256": got, "pinned_files": pins, "katago": kg}


def h7r_condition():
    """Item 4 (PREREG_07 rev 3): H7R scored, Q02H rank-balanced >= 0.57, the reference re-read equal to H4R's B row.
    -> record; exit 3 otherwise (07 is not used)."""
    p = path_of("results/prereg07/rank/score_H7R.json")
    if not os.path.exists(p):
        not_used("PREREG_07 rev 3: H7R has not been scored: 07 is not used", "prerequisites", "h7r_missing")
    s = read_json(p)
    a_rb, a_m, b_rb = s.get("A_rank_balanced"), s.get("A_mean"), s.get("B_rank_balanced")
    n = (s.get("B_minus_A_rank_balanced") or {}).get("n")
    rec = {"A_rank_balanced": a_rb, "A_mean": a_m, "B_rank_balanced": b_rb, "questions": n,
           "rule": "PREREG_07 rev 3: H7R scored, Q02H >= 0.57, reference re-read equal (|A - H4R| <= 1e-12)",
           "B_minus_A_rank_balanced": s.get("B_minus_A_rank_balanced")}
    ok = (is_num(a_rb) and is_num(a_m) and is_num(b_rb) and n == H7R_QUESTIONS
          and abs(a_rb - H7R_A_RANK_BALANCED) <= H7R_TOL and abs(a_m - H7R_A_MEAN) <= H7R_TOL and b_rb >= H7R_FLOOR)
    rec["holds"] = bool(ok)
    if not ok:
        not_used(f"PREREG_07 rev 3 condition does not hold ({rec}): 07 is not used", "prerequisites", "h7r", rec)
    return rec


def earlier_success(ctx):
    """A committed t07_apply_report.json with exit_code 0, committed by K6-07 (real mode) -> (bool, record)."""
    p = os.path.join(ctx["rank"], "t07_apply_report.json")
    if not os.path.exists(p):
        return False, None
    rep = read_json(p)
    if rep.get("exit_code") != 0:
        return False, None
    if not ctx["real"]:
        return True, {"note": f"{ctx['mode']}: commit time not checked"}
    h, t = commit_time(p)
    ok = t is not None and parse_iso(t, "commit time") <= K6_07
    return ok, {"commit": h, "committed_at": t}


def check_k6(ctx, a):
    """Item 5: K6-07 not passed; after it, exit 3 without an earlier success, 2 with one (that one stays valid)."""
    t_now = parse_iso(a.now, "--now") if getattr(a, "now", None) else now()
    if t_now <= K6_07:
        return {"now": t_now.isoformat(), "k6_07": K6_07.isoformat(), "passed": False}
    ok, rec = earlier_success(ctx)
    if ok:
        refuse(f"K6-07 {K6_07.isoformat()} has passed; the successful run before it ({rec}) stays valid: nothing is "
               f"rerun", reason="k6_passed_success")
    not_used(f"K6-07 {K6_07.isoformat()} has passed without a successful official 07 run: 07 is not used this "
             f"season (PREREG_07 4); the final choice follows PREREG_04 5.3", "prerequisites", "k6_07")


def r20_upload(ctx, t02rep, a):
    """Item 8: R20's first upload logged as uploaded (real); drills take --r20-uploaded-at. -> record (or None)."""
    if ctx["real"]:
        return t4call(t4.r20_first_upload, ctx4(ctx), t02rep)
    if getattr(a, "r20_uploaded_at", None):
        return {"id": None, "uploaded_at": parse_iso(a.r20_uploaded_at, "--r20-uploaded-at").isoformat(),
                "note": f"{ctx['mode']}: --r20-uploaded-at"}
    return {"id": None, "uploaded_at": None, "note": f"not checked ({ctx['mode']})"}


def t04_state(ctx):
    """Item 9 (apply only): test04's apply report of this mode, exit 0 and usable (committed in real mode). 04 judged
    'not used' (opp or apply) -> 3; not done yet -> 2. -> (report, path)."""
    rank4 = os.path.join(ctx["R4"], "rank")
    rp, vp = os.path.join(rank4, "t04_apply_report.json"), os.path.join(rank4, "attempt_verdict.json")
    op = os.path.join(ctx["R4"], "opp", "opp_report.json")
    opp = read_json_or_none(op)
    if opp is not None and opp.get("exit_code") == 3 and opp.get("mode") == ctx["mode"]:
        not_used("test04 opp judged '04 is not used' (PREREG_04 5.2): 07 is not used either (PREREG_07 5.2)",
                 "prerequisites", "t04_opp_not_used")
    if not os.path.exists(rp):
        refuse("test04 apply has not finished (no t04_apply_report.json"
               + (f"; attempt_verdict.json says exit {read_json(vp).get('exit_code')}" if os.path.exists(vp) else "")
               + "): run docs/TESTDAY_04.md step 5 first", reason="t04_not_done")
    rep = read_json(rp)
    if rep.get("format") != t4.FORMATS["apply"] or rep.get("mode") != ctx["mode"]:
        refuse(f"{shown(rp)}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (this run: {ctx['mode']})")
    if rep.get("exit_code") == 3 or rep.get("usable") is False:
        not_used(f"test04 apply judged '04 is not used' ({rep.get('unusable_reasons')}): 07 is not used either",
                 "prerequisites", "t04_not_used")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True:
        refuse(f"{shown(rp)}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    if ctx["real"]:
        require_committed([rp, os.path.join(rank4, "rank_test_probs.csv"),
                           os.path.join(rank4, "seam_val", "seam_local.json")], "results/test04/rank")
    return rep, rp


def prereqs(ctx, cmd, a):
    """Spec 7.2 items 1-9 in their order. -> (fc07, fc04, t02, t04 report or None, record)."""
    rec = {}
    check_git_clean(ctx, rec)                                            # 1
    if cmd == "apply":                                                   # 2 (.venv; the GPU check is in hs)
        rec["venv"] = t4call(t4.check_venv)
    fc07, rec["prereg07"] = check_p07_pins()                             # 3
    fc04, rec["prereg04_frozen"] = t4call(t4.load_frozen_config, ctx["real"])
    rec["h7r"] = h7r_condition()                                         # 4
    rec["k6_07"] = check_k6(ctx, a)                                      # 5
    rec["t1"] = t4call(t4.t1_first, ctx4(ctx))                           # 6
    t02 = t4call(t4.check_test02, ctx4(ctx))                             # 7
    rec["test02"] = {"report_sha256": t02["report_sha256"], "fallback_used": t02["report"].get("fallback_used")}
    t02["r20"] = r20_upload(ctx, t02["report"], a)                       # 8
    rec["r20_first_upload"] = t02["r20"]
    rep4 = None
    if cmd == "apply":                                                   # 9
        rep4, rec["t04_apply_report"] = t04_state(ctx)
    return fc07, fc04, t02, rep4, rec


# ================================================================ hs (spec 7.3)
def gpu_env(fake):
    import p07_hs as h7
    return h7.gpu_env(fake=fake)


def check_gpu_env(fc07, env, fake):
    """.venv-gpu: torch and cuDNN equal frozen_config's reference (else 4); the GPU model (else 2)."""
    ref = fc07["environment"]["venv_gpu_reference"]
    if fake:
        return {"fake_scorer": True, "reference": ref}
    bad = {k: {"now": env.get(k), "frozen": ref.get(k)} for k in ("python", "torch", "cudnn")
           if str(env.get(k)) != str(ref.get(k))}
    if bad:
        drift(f".venv-gpu differs from frozen_config.json {bad}: restore .venv-gpu (never upgraded before 11/24)",
              "prerequisites", "venv_gpu", bad)
    if env.get("gpu") != GPU_NAME:
        refuse(f"GPU {env.get('gpu')!r} is not {GPU_NAME} (PREREG_07 5.3: the HS runs on the local RTX 4090 only)")
    return {"equal": True, "reference": ref, "driver_record_only": env.get("driver")}


def set_slots(ctx, s):
    """-> (games [{sgf_content, color, ref}], keys record). Reads one question file (before hs_lib is imported)."""
    if s == "seam":
        qfile = SEAM_QUESTIONS
        q_sha = sha_raw(qfile)
        if q_sha != pf7.QUESTIONS_SHA:
            refuse(f"{shown(qfile)}: sha256 {q_sha[:12]} is not the registered {pf7.QUESTIONS_SHA[:12]}")
        vk = rf.load_val_keys(os.path.dirname(VAL_KEYS))
        if vk["sha256"] != pf7.VAL_KEYS_SHA:
            refuse("results/prereg02/rank/dumps/val_keys.npz is not the registered file")
    else:
        qfile = os.path.join(ctx["T"], t2.TEST_FILES[s])
        q_sha = sha_raw(qfile)
        vk = rf.load_val_keys(os.path.join(ctx["D2"], "rank", s, "dumps"))
    slots = po.read_question_slots(qfile)
    try:
        po.slots_match_keys(slots, vk)
    except po.P04Error as e:
        refuse(f"{s}: the question file and test02's val_keys.npz differ: {e}", reason="keys")
    n = SEAM_GAMES if s == "seam" else len(slots)
    games = [{"sgf_content": sgf, "color": col, "ref": i} for i, (_q, _sl, col, sgf) in enumerate(slots[:n])]
    keys = {"question_id": vk["question_id"][:n], "slot": vk["slot"][:n], "color": vk["color"][:n]}
    return games, keys, {"questions_file": shown(qfile), "questions_sha256": q_sha, "keys_file": shown(vk["path"]),
                         "keys_sha256": vk["sha256"], "games": n, "questions": len(dict.fromkeys(keys["question_id"]))}


def chunk_paths(d):
    return sorted(f for f in glob.glob(os.path.join(d, "chunk_*.npz")) if ".tmp" not in os.path.basename(f))


def stale_chunks(d, r20_at):
    """Chunks in d created before R20's first upload (spec 7.3: never reused, never overwritten)."""
    if not r20_at:
        return []
    t20 = parse_iso(r20_at, "R20 uploaded_at")
    out = []
    for p in chunk_paths(d):
        try:
            with np.load(p, allow_pickle=False) as z:
                c = str(z["created_utc"])
        except Exception:  # noqa: BLE001
            out.append((p, "unreadable"))
            continue
        if parse_iso(c, "created_utc") <= t20:
            out.append((p, c))
    return out


def chunk_check(path, refs, keys_sha, env, real=False):
    """An existing chunk is reused only when it is this run's (format, profiles, sides, refs, key file; else 2) and of
    this environment (torch / cuDNN / GPU; else 4)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            got = {"format": str(z["format"]), "profiles": [str(x) for x in z["profiles"]],
                   "sides": [str(x) for x in z["sides"]], "keys_sha256": str(z["keys_sha256"]),
                   "torch": str(z["torch"]), "cudnn": str(z["cudnn"]), "gpu_name": str(z["gpu_name"]),
                   "ckpt_sha256": str(z["ckpt_sha256"])}
            same_refs = np.array_equal(np.asarray(z["ref"]), refs)
    except Exception as e:  # noqa: BLE001
        refuse(f"{path}: unreadable chunk ({e}); move it away and rerun")
    want = {"format": CHUNK_FORMAT, "profiles": pf7.PROFILES, "sides": pf7.RUN["sides"], "keys_sha256": keys_sha}
    diff = [k for k in want if got[k] != want[k]] + ([] if same_refs else ["ref"])
    if diff:
        refuse(f"{path}: an existing chunk differs from this run ({diff}); never overwritten")
    envd = {k: (got[k], str(env[e])) for k, e in (("torch", "torch"), ("cudnn", "cudnn"), ("gpu_name", "gpu"))
            if got[k] != str(env[e])}
    if envd:
        drift(f"{path}: written with another environment {envd}; stop and find out why (spec 3.3)", "hs",
              "chunk_environment", envd)
    if real and (got["ckpt_sha256"] != pf7.CKPT_SHA or got["torch"] == "fake"):
        refuse(f"{path}: a drill chunk (fake scorer) is never reused in the real mode")
    if got["ckpt_sha256"] not in (pf7.CKPT_SHA, "fake"):
        refuse(f"{path}: chunk without the full checkpoint SHA-256")


def hs_outputs(ctx, s):
    return [os.path.join(ctx["hs"], f"{s}_hs.npz"), os.path.join(ctx["hs"], f"hs_{s}_report.json")]


def cmd_hs(a):
    ctx = resolve(a, "hs")
    try:
        return hs_run(a, ctx)
    except T07Error as e:
        if e.code == 3:              # spec 7.1: the registered 'no 07' is written down and committed
            write_json(ctx, os.path.join(ctx["R7"], "not_used.json"),
                       {"format": FORMATS["not_used"], "mode": ctx["mode"], "created": now_iso(), "exit_code": 3,
                        "stage": f"hs --set {a.set}: {e.stage or 'prerequisites'}", "reason": e.reason,
                        "message": str(e), "details": e.details, "tool": tool_record()})
        raise


def hs_run(a, ctx):
    t0 = time.time()
    s = a.set
    if a.max_games and ctx["real"]:
        refuse("--max-games is for drills only")
    fake = bool(a.fake_scorer)
    fc07, _fc04, t02, _r4, pre = prereqs(ctx, "hs", a)
    outs = hs_outputs(ctx, s)
    have = [p for p in outs if os.path.exists(p)]
    rerun = None
    if have:
        if not (a.rerun and a.reason):
            refuse(f"{[shown(p) for p in have]} exist; rerun only with --rerun --reason TEXT (resuming needs no flag)")
        if ctx["real"] and any(committed_clean(p) for p in have):
            refuse("the committed HS outputs are never overwritten")
        rerun = {"reason": a.reason, "old_sha256": {os.path.basename(p): pin_sha(p) for p in have}}
    if not fake:      # review r1: nvidia-smi before torch touches CUDA (our own context would count as a job)
        import p07_hs as h7
        procs, _raw = h7.gpu_compute_processes()
        procs = None if procs is None else [p_ for p_ in procs if p_.get("pid") != os.getpid()]
        if procs is None or (procs and not a.ignore_other_gpu_jobs):
            refuse(f"nvidia-smi lists another compute process (or failed): {procs}; wait, or yield (spec 0)",
                   reason="other_gpu_job")
        pre["nvidia_compute_processes_before"] = procs
    env = gpu_env(fake)
    pre["gpu_env"] = check_gpu_env(fc07, env, fake)
    if s != "seam":
        seam_rep = read_json_or_none(os.path.join(ctx["hs"], "hs_seam_report.json"))
        if seam_rep is None:
            refuse("run hs --set seam first (spec 7.3: seam completes before public and private)")
        se = seam_rep.get("environment") or {}
        if not seam_rep.get("fake_from_val"):
            envd = {k: (se.get(k), env.get(k)) for k in ("torch", "cudnn", "gpu") if str(se.get(k)) != str(env.get(k))}
            if envd:
                drift(f"this environment differs from the seam's {envd}", "hs", "environment_vs_seam", envd)
    chunk_dir = os.path.join(ctx["H"], s)
    stale = stale_chunks(chunk_dir, t02["r20"].get("uploaded_at"))
    if stale:
        refuse(f"{len(stale)} chunk(s) in {shown(chunk_dir)} were written before R20's first upload "
               f"({t02['r20'].get('uploaded_at')}), e.g. {[os.path.basename(p) for p, _ in stale[:3]]}: rename the "
               f"folder to {os.path.basename(chunk_dir)}_stale_<time> (never reused, never deleted by the tool)",
               reason="stale_chunks")
    # ---- the question file and the keys (before hs_lib / pilot_run)
    games, keys, krec = set_slots(ctx, s)
    if a.max_games:
        games = games[:a.max_games]
        keys = {k: v[:a.max_games] for k, v in keys.items()}
    keys_sha = krec["keys_sha256"]
    # ---- only now hs_lib / pilot_run (their guard closes data/local for this process)
    pf7.check_pins()
    if pf7.PILOT_CODE not in sys.path:
        sys.path.insert(0, pf7.PILOT_CODE)
    import pilot_run as PR
    H = PR.H
    import multiprocessing as mp
    os.makedirs(guard_write(ctx, chunk_dir), exist_ok=True)
    refs_all = np.array([g["ref"] for g in games], dtype=np.int64)
    C = pf7.RUN["chunk"]
    chunks = [(ci, b, min(len(games), b + C)) for ci, b in enumerate(range(0, len(games), C))]
    todo = []
    for ci, b, e in chunks:
        p = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
        if os.path.exists(p):
            chunk_check(p, refs_all[b:e], keys_sha, env, ctx["real"])
        else:
            todo.append((ci, b, e))
    extra = sorted(set(chunk_paths(chunk_dir)) - {os.path.join(chunk_dir, f"chunk_{ci:05d}.npz") for ci, _, _ in chunks})
    if extra:
        refuse(f"{shown(chunk_dir)} holds chunks that are not this run's: {[os.path.basename(x) for x in extra[:5]]}")
    print(json.dumps({"time_utc": now_iso(), "msg": "hs start", "set": s, "mode": ctx["mode"], "games": len(games),
                      "chunks": len(chunks), "todo": len(todo)}), flush=True)
    tool_sha = sha_lf(SELF)
    hs_sha = pf7.PINNED["results/prereg07/explore/pilot_humansl/hs_lib.py"]
    stopped = None
    if todo:
        if fake:
            import p07_hs as h7
            sc, ck = h7.FakeScorer(len(pf7.PROFILES)), {"bytes": pf7.CKPT_BYTES, "sha256": "fake"}
        else:
            import torch
            if not torch.cuda.is_available():
                refuse("CUDA is not available")
            ck = H.check_checkpoint(H.CKPT_PATH, full_hash=True)
            if ck.get("sha256") != pf7.CKPT_SHA:
                refuse(f"checkpoint SHA-256 {ck.get('sha256')} is not {pf7.CKPT_SHA} (PREREG_07 1.2: use the backup "
                       f"data/katago_backup/, else stop)", reason="checkpoint")
            torch.backends.cudnn.benchmark = pf7.RUN["cudnn_benchmark"]
            model, _st = H.load_model(H.CKPT_PATH, device="cuda", check=False)
            sc = H.Scorer(model, H.profiles_from_names(pf7.PROFILES), device="cuda", precision=pf7.RUN["precision"],
                          batch=pf7.RUN["batch"], channels_last=pf7.RUN["channels_last"])
        group_positions = max(1, pf7.RUN["batch"] // len(pf7.PROFILES)) * pf7.RUN["group_batches"]
        t_run = time.time()
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for k, (ci, b, e) in enumerate(todo):
                if os.path.exists(os.path.join(chunk_dir, "STOP")):
                    stopped = "STOP file"
                elif a.stop_after_min and time.time() - t_run > a.stop_after_min * 60:
                    stopped = f"--stop-after-min {a.stop_after_min}"
                if stopped:
                    print(json.dumps({"time_utc": now_iso(), "msg": "stopping before the next chunk (rerun the "
                                      "same line to continue)", "reason": stopped, "chunks_left": len(todo) - k}),
                          flush=True)
                    break
                t1 = time.time()
                res = PR.run_chunk(pool, sc, games[b:e], tuple(pf7.RUN["sides"]), group_positions)
                secs = time.time() - t1
                evals = int(len(res["pos_side"]) * len(pf7.PROFILES))
                path = guard_write(ctx, os.path.join(chunk_dir, f"chunk_{ci:05d}.npz"))
                tmp = path + ".tmp.npz"
                np.savez(tmp, format=np.array(CHUNK_FORMAT), profiles=np.array(pf7.PROFILES),
                         sides=np.array(pf7.RUN["sides"]), ref=refs_all[b:e], set=np.array(f"test07_{s}"),
                         precision=np.array(pf7.RUN["precision"]), batch=np.array(pf7.RUN["batch"]),
                         channels_last=np.array(pf7.RUN["channels_last"]), torch=np.array(str(env["torch"])),
                         ckpt_bytes=np.array(ck["bytes"]), ckpt_sha256=np.array(ck["sha256"]),
                         katago_commit=np.array(pf7.KATAGO_COMMIT), hs_lib_sha256=np.array(hs_sha),
                         created_utc=np.array(now_iso()), seconds=np.array(secs), cudnn=np.array(str(env["cudnn"])),
                         gpu_name=np.array(str(env["gpu"])), driver=np.array(str(env.get("driver"))),
                         keys_sha256=np.array(keys_sha), questions_sha256=np.array(krec["questions_sha256"]),
                         tool_sha256=np.array(tool_sha), mode=np.array(ctx["mode"]), **res)
                os.replace(tmp, path)
                el = time.time() - t_run
                print(json.dumps({"time_utc": now_iso(), "msg": "chunk done", "set": s, "chunk": ci, "games": e - b,
                                  "failed": int(res["failed"].sum()), "evals": evals, "seconds": round(secs, 1),
                                  "evals_per_s": round(evals / max(secs, 1e-9), 1), "progress": f"{k + 1}/{len(todo)}",
                                  "eta_min": round(el / (k + 1) * (len(todo) - k - 1) / 60, 1)}), flush=True)
    if stopped:
        print(f"stopped ({stopped}); {shown(chunk_dir)} holds the finished chunks. To continue: delete "
              f"{shown(os.path.join(chunk_dir, 'STOP'))} if present, then run the same line again (resume, not rerun)")
        return 0
    return assemble(ctx, s, chunk_dir, chunks, keys, krec, env, pre, rerun, fake, t0)


def assemble(ctx, s, chunk_dir, chunks, keys, krec, env, pre, rerun, fake, t0):
    rows, parts = [], []
    for ci, _b, _e in chunks:
        p = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
        z = load_npz(p)
        parts.append(z)
        failed = np.asarray(z["failed"], dtype=bool)
        cnt = np.asarray(z["counts"])[:, 0]
        rows.append({"file": os.path.basename(p), "sha256": sha_raw(p), "games": int(len(z["ref"])),
                     "failed": int(failed.sum()), "no_target_moves": int(((cnt == 0) & ~failed).sum()),
                     "created_utc": str(z["created_utc"]), "seconds": float(z["seconds"]),
                     "evals": int(len(z["pos_side"]) * len(pf7.PROFILES)), "errors": json.loads(str(z["errors"])),
                     "torch": str(z["torch"]), "cudnn": str(z["cudnn"]), "gpu_name": str(z["gpu_name"]),
                     "driver": str(z["driver"])})
    ref = np.concatenate([np.asarray(z["ref"]) for z in parts])
    n = len(ref)
    if not np.array_equal(ref, np.arange(n)):
        refuse("chunk refs are not the set's slots in order")
    sums = np.concatenate([np.asarray(z["sums"])[:, 0, :] for z in parts]).astype(np.float64)
    counts = np.concatenate([np.asarray(z["counts"])[:, 0] for z in parts]).astype(np.int64)
    n_moves = np.concatenate([np.asarray(z["n_moves"]) for z in parts]).astype(np.int64)
    failed = np.concatenate([np.asarray(z["failed"]) for z in parts]).astype(bool)
    os.makedirs(guard_write(ctx, ctx["hs"]), exist_ok=True)
    out = guard_write(ctx, os.path.join(ctx["hs"], f"{s}_hs.npz"))
    arrays = {"format": np.array(FORMATS["hs"]), "set": np.array(s), "question_id": np.array(keys["question_id"][:n]),
              "slot": np.array(keys["slot"][:n], dtype=np.int64), "color": np.array(keys["color"][:n]),
              "sums": sums, "counts": counts, "n_moves": n_moves, "failed": failed,
              "profiles": np.array(pf7.PROFILES)}
    if s == "seam":
        offs = np.cumsum([0] + [len(z["ref"]) for z in parts])[:-1]
        keep = [np.asarray(z["pos_side"]) == 0 for z in parts]
        arrays.update(pos_game=np.concatenate([np.asarray(z["pos_game"])[k] + o for z, k, o in zip(parts, keep, offs)]
                                              ).astype(np.int32),
                      pos_turn=np.concatenate([np.asarray(z["pos_turn"])[k] for z, k in zip(parts, keep)]
                                              ).astype(np.int16),
                      pos_logp=np.concatenate([np.asarray(z["pos_logp"])[k] for z, k in zip(parts, keep)]
                                              ).astype(np.float32))
    tmp = out + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    evals = sum(r["evals"] for r in rows)
    secs = sum(r["seconds"] for r in rows)
    fails = int(pf7.hs_failures(failed, n_moves).sum())
    envs = sorted({(r["torch"], r["cudnn"], r["gpu_name"]) for r in rows})
    rep = {"format": FORMATS["hs_report"], "what": "PREREG_07 5.2 test-day HS of one set (no score)", "set": s,
           "mode": ctx["mode"], "created": now_iso(), "fake_scorer": fake, "fake_from_val": False,
           "games": n, "questions": len(dict.fromkeys(keys["question_id"][:n])), "max_games_drill": n < krec["games"],
           "failed": int(failed.sum()), "failed_or_no_move": fails,
           "no_target_moves": int(((counts == 0) & ~failed).sum()), "chunks": rows, "chunk_dir": shown(chunk_dir),
           "earliest_chunk_created_utc": min(r["created_utc"] for r in rows),
           "latest_chunk_created_utc": max(r["created_utc"] for r in rows),
           "chunk_environments": [list(x) for x in envs], "evals": evals, "gpu_seconds": round(secs, 1),
           "evals_per_s": round(evals / max(secs, 1e-9), 1), "environment": env, "keys": krec,
           "output": {"file": f"{s}_hs.npz", "sha256": sha_raw(out)}, "prerequisites": pre, "rerun": rerun,
           "run": pf7.RUN, "profiles": pf7.PROFILES, "tool": tool_record(),
           "seconds_this_call": round(time.time() - t0, 1),
           "next": f"commit {shown(ctx['hs'])}/{s}_hs.npz and hs_{s}_report.json"}
    write_json(ctx, os.path.join(ctx["hs"], f"hs_{s}_report.json"), rep)
    print(f"hs {s}: {n} games, failed {fails}, {evals} evals in {secs:.0f} s ({rep['evals_per_s']} evals/s) -> "
          f"{shown(out)}")
    return 0


# ================================================================ apply (spec 7.4)
def apply_outputs(ctx):
    rank, seam = ctx["rank"], os.path.join(ctx["rank"], "seam_val")
    out = [os.path.join(seam, n) for n in ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json",
                                           "seam_local.json")]
    out += [os.path.join(rank, n) for n in TEST_OUTPUTS + ("t07_apply_report.json", "attempt_verdict.json")]
    out += [os.path.join(ctx["hs"], "gpu_seam.json"), os.path.join(ctx["R7"], "not_used.json")]
    return out


def check_bug_revision(n, overturned):
    """--bug-revision N: PREREG_07 rev N >= 5 committed and clean, its row at HEAD, added after every overturned
    record (ISO times). -> record."""
    if n is None:
        return None
    if n < P07_MIN_REV:
        refuse(f"--bug-revision {n}: an override needs PREREG_07 rev {P07_MIN_REV} or later")
    if not committed_clean(P07_DOC):
        refuse("docs/PREREG_07_HUMANSL.md must be committed and clean")
    row = f"| rev {n} |"
    rc, head = git("show", "HEAD:docs/PREREG_07_HUMANSL.md")
    if rc != 0 or not any(line.startswith(row) for line in head.splitlines()):
        refuse(f"docs/PREREG_07_HUMANSL.md at HEAD has no row starting {row!r}")
    rc, out = git("log", "--reverse", "-S" + row, "--format=%H %cI", "--", "docs/PREREG_07_HUMANSL.md")
    if rc != 0 or not out:
        refuse(f"no commit added {row!r}")
    h, t = out.splitlines()[0].split()
    at = parse_iso(t, "commit time")
    early = [f"{w} {x}" for w, x in overturned if x and at <= parse_iso(x, w)]
    if early:
        refuse(f"PREREG_07 rev {n} ({h[:12]}, {t}) is not later than {early}")
    return {"rev": n, "commit": h, "committed_at": t, "overturns": [{"what": w, "time": x} for w, x in overturned]}


class Apply:
    def __init__(self, ctx, a, t0):
        self.ctx, self.a, self.t0 = ctx, a, t0
        self.stage, self.pre, self.replaced, self.bug, self.rec = "prerequisites", {}, {}, None, {}

    # ---- rerun rules
    def rerun_gate(self):
        ctx, a = self.ctx, self.a
        existing = [p for p in apply_outputs(ctx) if os.path.exists(p)]
        if not existing:
            if a.bug_revision is not None:
                self.bug = check_bug_revision(a.bug_revision, [])
            return
        if not (a.rerun and a.reason):
            refuse(f"outputs already exist ({[os.path.basename(p) for p in existing][:6]}); never overwritten without "
                   f"--rerun --reason TEXT")
        rep = read_json_or_none(os.path.join(ctx["rank"], "t07_apply_report.json"))
        nu = read_json_or_none(os.path.join(ctx["R7"], "not_used.json"))
        decided = [(n, r.get("created")) for n, r in (("t07_apply_report.json", rep), ("not_used.json", nu)) if r]
        ver = read_json_or_none(os.path.join(ctx["rank"], "attempt_verdict.json"))
        if ver and ver.get("reason") == "move_cross_check":    # PREREG_07 5.2: only after a revision naming the bug
            decided.append(("attempt_verdict.json (move cross-check)", ver.get("created")))
        if decided:
            if a.bug_revision is None:
                refuse(f"{[n for n, _ in decided]} hold the registered decision; a rerun needs a PREREG_07 revision "
                       f"naming a program error (--bug-revision N, N >= {P07_MIN_REV})")
            self.bug = check_bug_revision(a.bug_revision, decided)
        elif a.bug_revision is not None:
            self.bug = check_bug_revision(a.bug_revision, [])

    def clear(self):
        removed = {}
        for p in apply_outputs(self.ctx):
            if os.path.exists(p):
                removed[shown(p)] = pin_sha(p)
                guard_write(self.ctx, p)
                os.remove(p)
        self.replaced = removed

    def write_verdict(self, e):
        ctx = self.ctx
        rec = {"format": FORMATS["verdict"], "mode": ctx["mode"], "created": now_iso(), "exit_code": e.code,
               "stage": e.stage or self.stage, "reason": e.reason or str(e).splitlines()[0][:200], "message": str(e),
               "details": e.details, "checks": self.pre, "records": self.rec,
               "rerun": {"reason": self.a.reason, "replaced": self.replaced, "bug_revision": self.bug},
               "tool": tool_record(), "seconds": round(time.time() - self.t0, 1)}
        if e.code == 3:
            write_json(ctx, os.path.join(ctx["R7"], "not_used.json"), dict(rec, format=FORMATS["not_used"]))
        else:
            write_json(ctx, os.path.join(ctx["rank"], "attempt_verdict.json"), rec)

    # ---- step 1: the three HS sets
    def hs_sets(self, t02):
        ctx, out = self.ctx, {}
        for s in SETS:
            npz, rp = hs_outputs(ctx, s)
            for p in (npz, rp):
                if not os.path.exists(p):
                    refuse(f"missing {shown(p)}: run hs --set {s} first", reason="hs_missing")
            if ctx["real"]:
                require_committed([npz, rp], f"the {s} HS (commit it after hs)")
            rep = read_json(rp)
            if rep.get("format") != FORMATS["hs_report"] or rep.get("mode") != ctx["mode"] or rep.get("set") != s:
                refuse(f"{shown(rp)}: format {rep.get('format')!r}, mode {rep.get('mode')!r}, set {rep.get('set')!r}")
            if rep.get("fake_from_val") or rep.get("fake_scorer") or rep.get("max_games_drill"):
                if ctx["real"]:
                    refuse(f"{shown(rp)} is a drill record (fake or --max-games): never in the real mode")
            if (rep.get("output") or {}).get("sha256") != sha_raw(npz):
                refuse(f"{shown(npz)} differs from its report")
            if not rep.get("fake_from_val"):       # review r1: a real record lists every chunk of its games
                want = -(-int(rep.get("games") or 0) // t7_chunk())
                if len(rep.get("chunks") or []) != want or not rep.get("games"):
                    refuse(f"{shown(rp)} lists {len(rep.get('chunks') or [])} chunks for {rep.get('games')} games")
            bad = []
            for ch in rep.get("chunks") or []:
                p = os.path.join(ctx["H"], s, ch["file"])
                if not os.path.exists(p) or sha_raw(p) != ch["sha256"]:
                    bad.append(ch["file"])
            if bad:
                refuse(f"{s}: HS chunks missing or changed since hs: {bad[:5]}")
            out[s] = {"report": rep, "report_path": rp, "npz_path": npz, "z": load_npz(npz)}
        envs = {s: tuple(str((out[s]["report"].get("environment") or {}).get(k)) for k in ("torch", "cudnn", "gpu"))
                for s in SETS}
        if len(set(envs.values())) != 1:
            drift(f"the three HS sets were computed in different environments {envs}", "hs_sets", "hs_environment",
                  {"environments": envs})
        r20_at = (t02["r20"] or {}).get("uploaded_at")
        early = {}
        if r20_at:
            t20 = parse_iso(r20_at, "R20 uploaded_at")
            for s in SETS:
                e_ = out[s]["report"].get("earliest_chunk_created_utc")
                if not e_ or parse_iso(e_, "created_utc") <= t20:
                    early[s] = e_
        if early:
            refuse(f"HS computed before R20's first upload ({r20_at}): {early}; rename the chunk folders and rerun "
                   f"hs (spec 7.3)", reason="hs_before_r20")
        self.rec["hs_sets"] = {s: {"report_sha256": pin_sha(out[s]["report_path"]),
                                   "npz_sha256": sha_raw(out[s]["npz_path"]), "games": out[s]["report"].get("games"),
                                   "failed_or_no_move": out[s]["report"].get("failed_or_no_move"),
                                   "earliest_chunk_created_utc": out[s]["report"].get("earliest_chunk_created_utc"),
                                   "fake_from_val": bool(out[s]["report"].get("fake_from_val")),
                                   "evals_per_s": out[s]["report"].get("evals_per_s")} for s in SETS}
        self.rec["hs_environment"] = dict(zip(("torch", "cudnn", "gpu"), envs["seam"]))
        self.rec["r20_uploaded_at"] = r20_at
        return out

    # ---- step 2: the GPU seam
    def gpu_seam(self, hs, model):
        ctx = self.ctx
        z = hs["seam"]["z"]
        vh = load_npz(path_of("results/prereg07/hs/val_hs.npz"))
        vm = load_npz(path_of("results/prereg07/hs/val_seam_moves.npz"))
        rec = gpu_seam_record(z, vh, vm, model)
        rec.update(format=FORMATS["gpu_seam"], mode=ctx["mode"], created=now_iso(),
                   seam_hs_sha256=sha_raw(hs["seam"]["npz_path"]),
                   reference={"val_hs.npz": P07_PINS["results/prereg07/hs/val_hs.npz"],
                              "val_seam_moves.npz": P07_PINS["results/prereg07/hs/val_seam_moves.npz"]})
        path = write_json(ctx, os.path.join(ctx["hs"], "gpu_seam.json"), rec)
        self.rec["gpu_seam"] = {k: rec[k] for k in ("passed", "counts_equal", "max_abs_move", "max_abs_profile_mean",
                                                    "max_abs_lp_hs", "bitwise_equal_sums", "bitwise_equal_moves")}
        print(f"GPU seam: counts equal {rec['counts_equal']}, max |dlogp| per move {rec['max_abs_move']:.3g} "
              f"(<= {TOL_MOVE}), per game/profile mean {rec['max_abs_profile_mean']:.3g} (<= {TOL_PROFILE_MEAN}), "
              f"lp_HS {rec['max_abs_lp_hs']:.3g} (<= {TOL_LP}) -> {'PASSED' if rec['passed'] else 'FAILED'}",
              flush=True)
        if not rec["passed"]:
            drift("GPU seam failed (local GPU / driver / environment drift; PREREG_07 5.2): stop and find out why",
                  "gpu_seam", "gpu_seam", {k: rec[k] for k in ("counts_equal", "max_abs_move", "max_abs_profile_mean",
                                                                "max_abs_lp_hs", "problems")})
        return path

    # ---- the run
    def go(self):
        ctx, a = self.ctx, self.a
        fc07, fc04, t02, rep4, self.pre = prereqs(ctx, "apply", a)   # spec 7.2: a registered 3 before "outputs exist"
        self.rerun_gate()
        self.stage = "hs_sets"
        hs = self.hs_sets(t02)
        model = pf7.load_hs_map(path_of("results/prereg07/rank/hs_map.npz"))
        self.clear()
        self.stage = "gpu_seam"
        gpu_seam_path = self.gpu_seam(hs, model)
        # ---- step 3: Q02 recomputed, equal to test04's rank_test_probs.csv
        self.stage = "q02_vs_04"
        bound4 = rep4.get("inputs_bound") or {}
        b4probs = t04_bound_problems(ctx, bound4)
        if b4probs:
            refuse(f"test04's bound inputs changed since its apply: {b4probs[:5]} (follow docs/TESTDAY_04.md first)",
                   reason="t04_inputs_changed")
        vk = {s: rf.load_val_keys(os.path.join(ctx["D2"], "rank", s, "dumps")) for s in Q_SETS}
        sw = {s: load_sw(ctx, s, vk[s]) for s in Q_SETS}
        o4z = {s: load_npz(os.path.join(ctx["R4"], "opp", f"o4_{s}.npz")) for s in Q_SETS}
        c5w = fc04["source_sha256"]["C5"]["weights_sha256"]
        fz = {}
        for s in Q_SETS:     # every question-file read of this process happens here (seam_val: data/local)
            ddir = os.path.join(ctx["D2"], "rank", s, "dumps")
            c5 = t4call(t4.load_c5, s, ddir, vk[s], c5w)
            f = t4call(t4.fuse_set, s, vk[s], t02["qfiles"][s], c5, t02["opening"], o4z[s], sw[s], fc04, False)
            f["c5_sha256"] = c5["sha256"]
            f.update(port_c5(vk[s], c5, fc04, f))
            fz[s] = f
        q02_rows = fz["public"]["rows"] + fz["private"]["rows"]
        q02_bytes = probs_bytes(q02_rows)
        ref4 = os.path.join(ctx["R4"], "rank", "rank_test_probs.csv")
        with open(ref4, "rb") as fh:
            ref_bytes = fh.read()
        q02_equal = q02_bytes == ref_bytes.replace(b"\r\n", b"\n")
        self.rec["q02_vs_04"] = {"equal": q02_equal, "computed_sha256": hashlib.sha256(q02_bytes).hexdigest(),
                                 "t04_rank_test_probs_sha256": pin_sha(ref4),
                                 "port_c5_equal": {s: fz[s]["port_equal"] for s in Q_SETS}}
        if not all(fz[s]["port_equal"] for s in Q_SETS):
            drift("the ported C5 lines do not give fuse_set's Q02 bit for bit", "q02_vs_04", "port_c5",
                  self.rec["q02_vs_04"])
        if not q02_equal:
            drift("Q02 recomputed with test04's inputs differs from results/test04/rank/rank_test_probs.csv (local "
                  "problem; PREREG_07 5.2)", "q02_vs_04", "q02_vs_04", self.rec["q02_vs_04"])
        # ---- from here no data/local file is opened: pilot_fuse (hs_lib's guard) for the lp_HS cross-check
        PF = pf7.import_pilot_fuse()
        # ---- step 4: the move cross-check against test04's O4 replay
        self.stage = "move_cross_check"
        lp, cross = {}, {}
        for s in TEST_SETS:
            z = hs[s]["z"]
            keys_ok = (list(map(str, z["question_id"])) == vk[s]["question_id"]
                       and [int(x) for x in z["slot"]] == vk[s]["slot"] and list(map(str, z["color"])) == vk[s]["color"])
            if not keys_ok:
                refuse(f"{s}_hs.npz keys differ from test02's val_keys.npz of {s}", reason="hs_keys")
            o = o4z[s]
            cross[s] = pf7.move_cross_check(z["n_moves"], z["counts"], z["failed"], o["moves_parsed"],
                                            np.asarray(o["n_moves"])[:, 0, :].sum(1), o["replay_failed"],
                                            [f"{q}#{sl}" for q, sl in zip(vk[s]["question_id"], vk[s]["slot"])])
        self.rec["move_cross_check"] = cross
        if any(cross[s]["mismatches"] != 0 for s in TEST_SETS):
            drift(f"move cross-check against test04's o4_<set>.npz: {[cross[s]['mismatches'] for s in TEST_SETS]} "
                  f"mismatches (registered 0): a program or input-binding error; only a committed PREREG_07 revision "
                  f"naming it allows a rerun", "move_cross_check", "move_cross_check", cross)
        # ---- step 5: the local Q02H seam on the validation questions
        self.stage = "local_seam"
        vh = load_npz(path_of("results/prereg07/hs/val_hs.npz"))
        seam = self.local_seam(fz["seam_val"], vk["seam_val"], vh, model, fc07, rep4, PF)
        # ---- steps 6, 7: HS failures and direction
        self.stage = "hs_failures"
        res = {}
        for s in TEST_SETS:
            z = hs[s]["z"]
            res[s] = q02h_set(fz[s], vk[s], z, model, fc07, PF)
        fails = {s: {"failed_or_no_move": res[s]["fail_count"], "games": vk[s]["n"],
                     "ok": rate_ok(res[s]["fail_count"], vk[s]["n"])} for s in TEST_SETS}
        self.rec["hs_failures"] = fails
        bad = [s for s in TEST_SETS if not fails[s]["ok"]]
        if bad:
            not_used(f"HS failures above 0.5 % on {bad} ({fails}): 07 is not used (PREREG_07 5.2)", "hs_failures",
                     "hs_failures", fails)
        self.stage = "hs_direction"
        direc = {s: pf7.direction_record(res[s]["rho"], res[s]["rho_n"], RHO_MONO, RHO_FLOOR) for s in TEST_SETS}
        direc["seam_val_record_only"] = seam["direction"]
        self.rec["hs_direction"] = direc
        bad = [s for s in TEST_SETS if not direc[s]["passed"]]
        if bad:
            not_used(f"HS direction check failed on {bad} ({ {s: direc[s]['rho'] for s in TEST_SETS} } < "
                     f"{direc['public']['threshold']}): 07 is not used (PREREG_07 5.2)", "hs_direction",
                     "hs_direction", direc)
        # ---- step 8: the outputs
        self.stage = "outputs"
        rank = ctx["rank"]
        rows = res["public"]["rows"] + res["private"]["rows"]
        probs_csv = guard_write(ctx, os.path.join(rank, "rank_test_probs.csv"))
        os.makedirs(rank, exist_ok=True)
        t2.write_probs(probs_csv, rows)
        for n in ("pred_A.csv", "pred_B.csv", "decide_summary.json"):
            guard_write(ctx, os.path.join(rank, n))
        dsum = decide.run(probs_csv, rank)
        sub = guard_write(ctx, os.path.join(rank, "submission_rank.csv"))
        t2.atomic_copy(os.path.join(rank, "pred_B.csv"), sub)
        with open(sub, "rb") as f1, open(os.path.join(rank, "pred_B.csv"), "rb") as f2:
            if f1.read() != f2.read():
                refuse("submission_rank.csv is not a byte copy of pred_B.csv")
        self.stage = "check_submission"
        check_q = [t02["qfiles"]["public"], t02["qfiles"]["private"]]
        errors = check_submission.check("rank", sub, check_q, official=True)
        if errors:
            removed = []
            for n in TEST_OUTPUTS:
                p = os.path.join(rank, n)
                if os.path.exists(p):
                    os.remove(p)
                    removed.append(n)
            drift(f"check_submission --official failed on the Q02H file: {errors[:5]}; {removed} removed",
                  "check_submission", "check_submission", {"errors": errors[:50], "removed": removed})
        # ---- step 9: the report
        preds = dict(t2.read_pred(sub))
        pub_ids = set(fz["public"]["qids"])
        diffs = {}
        for name, other in (("04", os.path.join(ctx["R4"], "rank", "submission_rank.csv")),
                            ("R20", os.path.join(ctx["O2"], "submission_rank.csv"))):
            if os.path.exists(other):
                op_ = dict(t2.read_pred(other))
                diffs[name] = {"public": sum(1 for q in pub_ids if op_.get(q) != preds.get(q)),
                               "all": sum(1 for q, v in preds.items() if op_.get(q) != v),
                               "file_sha256": sha_raw(other)}
            else:
                diffs[name] = None
        bound = self.inputs_bound(t02, gpu_seam_path, seam["path"])
        rep = {"format": FORMATS["apply"], "what": "test07 competition 2 apply (PREREG_07 5.2): frozen Q02H per set, "
                                                   "rule B, submission_rank.csv = pred_B.csv (public then private)",
               "mode": ctx["mode"], "created": now_iso(), "exit_code": 0, "usable": True, "unusable_reasons": [],
               "combo": COMBO, "method": METHOD, "theta": fc07["params"]["theta"],
               "h7r": self.pre["h7r"], "t1_verdict": (self.pre.get("t1") or {}).get("verdict"),
               "test02": {"fallback_used": t02["report"].get("fallback_used"),
                          "apply_test_report_sha256": t02["report_sha256"],
                          "r20_first_upload_id": (t02["r20"] or {}).get("id"),
                          "r20_first_uploaded_at": (t02["r20"] or {}).get("uploaded_at")},
               "test04": {"t04_apply_report_sha256": pin_sha(os.path.join(ctx["R4"], "rank", "t04_apply_report.json")),
                          "exit_code": rep4.get("exit_code"), "usable": rep4.get("usable"),
                          "seam_local_tol": (rep4.get("seam_local") or {}).get("tol"),
                          "inputs_bound_checked": len(bound4)},
               "hs_sets": self.rec["hs_sets"], "hs_environment": self.rec["hs_environment"],
               "earliest_hs_chunk_created_utc": min(self.rec["hs_sets"][s]["earliest_chunk_created_utc"] or ""
                                                    for s in SETS),
               "gpu_seam": self.rec["gpu_seam"], "gpu_seam_tolerances": {"move": TOL_MOVE,
                                                                        "profile_mean": TOL_PROFILE_MEAN,
                                                                        "lp_hs": TOL_LP},
               "q02_vs_04": self.rec["q02_vs_04"], "move_cross_check": cross, "seam_local": seam["record"],
               "hs_failures": fails, "hs_direction": direc, "rho_mono": RHO_MONO, "rho_floor": RHO_FLOOR,
               "sets": {s: res[s]["info"] for s in TEST_SETS},
               "rule_b_differences": diffs,
               "questions_without_parsed_game": {s: res[s]["info"]["questions_without_parsed_game_ids"]
                                                 for s in TEST_SETS},
               "check_submission": {"mode": "official", "questions": [shown(p) for p in check_q], "errors": errors},
               "outputs_sha256": {n: sha_raw(os.path.join(rank, n)) for n in TEST_OUTPUTS},
               "decide_summary": {k: dsum[k] for k in ("n_questions", "n_missing", "missing_question_ids",
                                                        "n_A_ne_B", "distribution_B")},
               "level_summary": {s: t2.level_summary(res[s]["P"], res[s]["ok"],
                                                     [preds[q] for q in fz[s]["qids"]]) for s in TEST_SETS},
               "inputs_bound": bound["sha256"], "inputs_bound_paths": bound["paths"],
               "inputs_bound_rule": ".csv / .json: sha256 with CRLF folded to LF; other files (.npz): raw bytes",
               "checks": self.pre, "rerun": {"reason": a.reason, "replaced": self.replaced, "bug_revision": self.bug},
               "paths": {k: ctx[k] for k in ("R7", "H", "R4", "D2", "O2", "T", "C")}, "tool": tool_record(),
               "environment": t4.environment(), "git_head": git_head(),
               "record_only": "level_summary, rule_b_differences (all 800) and the seam_val direction are recorded; "
                              "nothing is chosen from them"}
        rep["seconds"] = round(time.time() - self.t0, 1)
        write_json(ctx, os.path.join(rank, "t07_apply_report.json"), rep)
        print(f"apply Q02H: public {len(fz['public']['qids'])}, private {len(fz['private']['qids'])} questions -> "
              f"{shown(sub)} (sha256 {rep['outputs_sha256']['submission_rank.csv'][:12]}); check_submission official: "
              f"OK; rule-B differences on public vs 04 / R20: {(diffs.get('04') or {}).get('public')} / "
              f"{(diffs.get('R20') or {}).get('public')}")
        print("下一步：commit results/test07/，04 上傳後照 docs/TESTDAY_07.md 第 5 步 official_log.py prepare"
              if ctx["real"] else f"({ctx['mode']}: nothing is uploaded)")
        return 0

    def local_seam(self, fzs, vks, vh, model, fc07, rep4, PF):
        """Spec 7.4 item 5: Q02H on seam_val (test02 / test04 seam inputs + the committed validation HS) against
        results/prereg07/rank/rank_val_probs.csv."""
        ctx = self.ctx
        keys_ok = (list(map(str, vh["question_id"])) == vks["question_id"] and [int(x) for x in vh["slot"]] ==
                   vks["slot"] and list(map(str, vh["color"])) == vks["color"])
        if not keys_ok:
            drift("val_hs.npz keys differ from test02's seam_val val_keys.npz", "local_seam", "seam_structure")
        r = q02h_set(fzs, vks, vh, model, fc07, PF)
        seam_dir = os.path.join(ctx["rank"], "seam_val")
        os.makedirs(guard_write(ctx, seam_dir), exist_ok=True)
        csv_p = guard_write(ctx, os.path.join(seam_dir, "rank_val_probs.csv"))
        t2.write_probs(csv_p, r["rows"])
        for n in ("pred_A.csv", "pred_B.csv", "decide_summary.json"):
            guard_write(ctx, os.path.join(seam_dir, n))
        decide.run(csv_p, seam_dir)
        ref_p, ref_b = os.path.join(P07_RANK, "rank_val_probs.csv"), os.path.join(P07_RANK, "pred_B.csv")
        tol = float((rep4.get("seam_local") or {}).get("tol"))
        sc = t2.seam_compare(csv_p, os.path.join(seam_dir, "pred_B.csv"), ref_p, ref_b, tol, False)
        same_a = t2._lf_bytes(os.path.join(seam_dir, "pred_A.csv")) == t2._lf_bytes(os.path.join(P07_RANK,
                                                                                                 "pred_A.csv"))
        sl4 = rep4.get("seam_local") or {}
        inputs_identical = sl4.get("byte_identical_probs") is True and sl4.get("byte_identical_pred_B") is True
        verdict = seam_verdict(sc, inputs_identical)
        dref = fc07["hs_direction_ref"]
        direction = pf7.direction_record(r["rho"], r["rho_n"], dref["rho_eval"], pf7.RHO_FLOOR_VAL)
        a7 = read_json(os.path.join(P07_RANK, "apply_report.json"))
        direction["equal_to_prereg07_apply"] = r["rho"] == (a7.get("hs_direction") or {}).get("rho")
        if inputs_identical and not direction["equal_to_prereg07_apply"]:
            drift("seam_val HS direction rho differs from results/prereg07/rank/apply_report.json although the inputs "
                  "are byte-identical (spec 7.4 item 7)", "local_seam", "seam_rho", direction)
        rec = {"format": FORMATS["seam_local"], "what": "test07 local Q02H seam (PREREG_07 5.2): Q02H on the seam_val "
                                                       "inputs + val_hs.npz vs results/prereg07/rank",
               "mode": ctx["mode"], "created": now_iso(), **sc, "byte_identical_pred_A": same_a,
               "t04_seam_inputs_byte_identical": inputs_identical, "required_byte_identical": inputs_identical,
               "verdict": verdict, "passed": verdict == "passed", "tol_source": "test04 seam_local.json tol",
               "set": r["info"], "direction": direction,
               "reference_sha256": {n: pin_sha(os.path.join(P07_RANK, n))
                                    for n in ("rank_val_probs.csv", "pred_B.csv", "pred_A.csv")},
               "outputs_sha256": {n: sha_raw(os.path.join(seam_dir, n)) for n in
                                  ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json")}}
        path = write_json(ctx, os.path.join(seam_dir, "seam_local.json"), rec)
        print(f"local seam: {sc['questions']} questions, max |dP| {sc['max_abs_diff']:.3g} (tol {tol:g}), rule-B "
              f"differences {sc['rule_b_differences']}, byte identical {sc['byte_identical_probs']}/"
              f"{sc['byte_identical_pred_B']}/{same_a} (required: {inputs_identical}) -> {verdict}", flush=True)
        self.rec["seam_local"] = {k: rec[k] for k in ("verdict", "passed", "max_abs_diff", "tol", "rule_b_differences",
                                                      "byte_identical_probs", "byte_identical_pred_B",
                                                      "required_byte_identical")}
        if verdict == "structure":
            drift("local seam: questions / order / game counts differ from results/prereg07/rank/rank_val_probs.csv",
                  "local_seam", "seam_structure", {"problems": sc["problems"],
                                                   "count_mismatch_questions": sc["count_mismatch_questions"]})
        if verdict == "drift":
            drift("local seam: test04's seam_val inputs equal prereg04 byte for byte but Q02H does not equal prereg07 "
                  "byte for byte (program drift)", "local_seam", "seam_bytes", self.rec["seam_local"])
        if verdict == "not_used":
            not_used("local seam outside the tolerance while test04's seam and Q02 hold: 07's weights amplify the "
                     "Colab differences of 04's inputs, which the local machine cannot repair: 07 is not used "
                     "(PREREG_07 5.2)", "local_seam", "seam_tol", self.rec["seam_local"])
        return {"path": path, "record": self.rec["seam_local"], "direction": direction}

    def inputs_bound(self, t02, gpu_seam_path, seam_local_path):
        """The files official_log re-hashes (keys: the real-mode repository paths)."""
        ctx = self.ctx
        R4, D2, R7 = ctx["R4"], ctx["D2"], ctx["R7"]
        items = [("results/test02/rank/apply_test_report.json", t02["report_path"]),
                 ("results/test02/rank/opening_test.npz", t02["opening_path"]),
                 ("results/test02/checks/rank/t1.json", os.path.join(ctx["C"], "rank", "t1.json")),
                 ("results/test04/opp/opp_report.json", os.path.join(R4, "opp", "opp_report.json")),
                 ("results/test04/unpack_sw.json", os.path.join(R4, "unpack_sw.json")),
                 ("results/test04/sw/config.json", os.path.join(R4, "sw", "config.json")),
                 ("results/test04/sw/seam_check.json", os.path.join(R4, "sw", "seam_check.json")),
                 ("results/test04/rank/seam_val/seam_local.json", os.path.join(R4, "rank", "seam_val",
                                                                               "seam_local.json")),
                 ("results/test04/rank/t04_apply_report.json", os.path.join(R4, "rank", "t04_apply_report.json")),
                 ("results/test04/rank/rank_test_probs.csv", os.path.join(R4, "rank", "rank_test_probs.csv")),
                 ("results/test04/rank/submission_rank.csv", os.path.join(R4, "rank", "submission_rank.csv")),
                 ("results/prereg04/rank/frozen_config.json", t4.FROZEN_CONFIG),
                 ("results/test07/hs/gpu_seam.json", gpu_seam_path),
                 ("results/test07/rank/seam_val/seam_local.json", seam_local_path)]
        items += [(r, path_of(r)) for r in P07_PINS]
        for s in SETS:
            items += [(f"results/test07/hs/{s}_hs.npz", os.path.join(R7, "hs", f"{s}_hs.npz")),
                      (f"results/test07/hs/hs_{s}_report.json", os.path.join(R7, "hs", f"hs_{s}_report.json"))]
        for s in Q_SETS:
            for f in ("val_keys.npz", "val_%s.npz" % t4.C5_DUMP):
                items.append((f"results/test02/rank/{s}/dumps/{f}", os.path.join(D2, "rank", s, "dumps", f)))
            items.append((f"results/test04/opp/o4_{s}.npz", os.path.join(R4, "opp", f"o4_{s}.npz")))
            for f in ["val_keys.npz"] + [f"val_p03{x}_{v}.npz" for x in t4.ARMS for v in ("d4", "s0_swap")]:
                items.append((f"results/test04/sw/{s}/dumps/{f}", os.path.join(R4, "sw", s, "dumps", f)))
        out, paths = {}, {}
        for key, p in items:
            if os.path.exists(p):
                out[key] = pin_sha(p)
                paths[key] = shown(p)
        return {"sha256": out, "paths": paths}


def t7_chunk():
    return pf7.RUN["chunk"]


def seam_verdict(sc, inputs_identical):
    """Spec 7.4 item 5 from t02_rank.seam_compare's record: 'structure' (questions, order, game counts: exit 4),
    'drift' (04's seam_val inputs equal prereg04 byte for byte but Q02H is not byte-identical: exit 4), 'not_used'
    (outside tol or more than 4 rule-B differences while 04's seam and Q02 hold: exit 3) or 'passed'."""
    if sc["problems"] or sc["count_mismatch_questions"]:
        return "structure"
    if inputs_identical:
        return "passed" if (sc["byte_identical_probs"] and sc["byte_identical_pred_B"]) else "drift"
    return "passed" if sc["passed"] else "not_used"


def t04_bound_problems(ctx, bound):
    """Every input test04's apply bound (keys: real repository paths) still has its SHA-256 (tree paths outside the
    real mode)."""
    probs = []
    if not bound:
        return ["t04_apply_report.json lists no inputs_bound"]
    roots = {"results/test04/": ctx["R4"], "results/test02/checks/": ctx["C"], "results/test02/": ctx["D2"]}
    for key, want in sorted(bound.items()):
        p = None
        if ctx["real"] or key.startswith("results/prereg"):
            p = path_of(key)
        else:
            for pre, root in roots.items():
                if key.startswith(pre):
                    rest = key[len(pre):]
                    if pre == "results/test02/" and rest.startswith("rank/") and rest.split("/")[1] in \
                            ("apply_test_report.json", "opening_test.npz"):
                        p = os.path.join(ctx["O2"], *rest.split("/")[1:])
                    else:
                        p = os.path.join(root, *rest.split("/"))
                    break
        if p is None or not os.path.exists(p):
            probs.append(f"{key} missing")
        elif pin_sha(p) != want:
            probs.append(f"{key} changed")
    return probs


def load_sw(ctx, s, vk):
    """The test04 notebook's swapped and d4 dumps of one set (tools/t04_rank.py check_dumps's readers; their markers
    and environment were judged by test04's apply, whose inputs_bound pins these files)."""
    out = {}
    for x in t4.ARMS:
        pd, ps = (os.path.join(ctx["R4"], "sw", s, "dumps", f"val_p03{x}_{v}.npz") for v in ("d4", "s0_swap"))
        try:
            d4 = rf.load_dump(pd, vk["n"], p3.N_SYMS)
            rf.check_val_logits(d4, vk["parsed"])
            swp = p3.load_swap(ps, vk["n"])
        except FOREIGN_ERRORS as e:
            refuse(f"test04 sw dumps of {s}/{x}: {e}")
        out[x] = {"d4": d4, "swap": swp}
    return out


def port_c5(vk, c5, fc04, fz):
    """fuse_set does not return m (C5 pooled) or S (the parsed pooling matrix): the C5 lines ported verbatim
    (t04_rank.fuse_set; spec 7.4 item 3), checked by recomputing Q02 with them against fuse_set's P bit for bit."""
    logits = {t4.C5_DUMP: np.where(vk["parsed"][:, None, None], c5["logits"], 0.0).astype(np.float32)}
    lp5 = p3.combo_logprobs(("C5",), logits)
    parsed = vk["parsed"]
    cnn = [[i for i in rows if parsed[i]] for _, rows in vk["groups"]]
    S = rf.pool_matrix(cnn, vk["n"])
    n_q = np.array([len(c) for c in cnn], dtype=np.float64)
    m = rf.pooled(S, lp5, "log")
    ok = n_q > 0
    P = pf7.fuse_probs(fc04["params"]["theta"], m, [fz["E"][t] for t in fc04["combo"]["terms"]], n_q, ok)
    same = bool(np.array_equal(ok, fz["ok"]) and np.array_equal(n_q, fz["n"])
                and np.array_equal(np.nan_to_num(P, nan=-1.0), np.nan_to_num(fz["P"], nan=-1.0)))
    return {"S": S, "m": m, "port_equal": same}


def probs_bytes(rows):
    buf = io.StringIO(newline="")
    import csv
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(rf.PROB_COLUMNS)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def q02h_set(fz, vk, z, model, fc07, PF):
    """Q02H of one set from fuse_set's E terms, the ported m / S and the HS arrays z (sums, counts, failed)."""
    lp_hs, hs_ok = pf7.hs_lp(model, z["sums"], z["counts"], z["failed"])
    x = np.zeros_like(np.asarray(z["sums"], dtype=np.float64))
    x[hs_ok] = np.asarray(z["sums"], dtype=np.float64)[hs_ok] / np.asarray(z["counts"])[hs_ok, None].astype(np.float64)
    lp_pf = np.zeros_like(lp_hs)
    if hs_ok.any():
        lp_pf[hs_ok] = PF.mlr_logp({"W": model["W"], "b": model["b"]}, (x[hs_ok] - model["mu"]) / model["sd"])
    if not np.array_equal(lp_pf, lp_hs):
        drift("lp_HS differs from pilot_fuse.mlr_logp on the same map", "hs_failures", "lp_hs_port")
    usable = pf7.usable_games(vk["parsed"], z["failed"], z["counts"])
    E_hs, e_info = pf7.masked_E(fz["S"], lp_hs, usable)
    ok, n = fz["ok"], fz["n"]
    Es = [fz["E"][t] for t in pf7.Q02_TERMS] + [E_hs]
    P = pf7.fuse_probs(fc07["params"]["theta"], fz["m"], Es, n, ok)
    has = pf7.questions_with_usable(fz["S"], usable) & ok
    rho, rho_n = pf7.hs_rho(E_hs, fz["P"], has)
    rows = []
    for k, (q, rws) in enumerate(vk["groups"]):
        probs = [repr(float(v)) for v in P[k]] if ok[k] else [""] * rf.N_LEVELS
        rows.append([q, len(rws), int(n[k])] + probs)
    fail_count = int(pf7.hs_failures(z["failed"], z["n_moves"]).sum())
    info = dict(fz["info"], masked_E_HS=e_info, hs_failed=int(np.asarray(z["failed"]).sum()),
                hs_failed_or_no_move=fail_count, unusable_games=int((~usable).sum()),
                questions_without_usable_hs_game=int((ok & ~has).sum()),
                rule_B_differs_from_Q02=int(((rf.rule_b_np(np.where(ok[:, None], P, 0.0)) !=
                                              rf.rule_b_np(np.where(ok[:, None], fz["P"], 0.0))) & ok).sum()))
    return {"P": P, "ok": ok, "rows": rows, "rho": rho, "rho_n": rho_n, "fail_count": fail_count, "info": info,
            "E_hs": E_hs}


def gpu_seam_record(z, vh, vm, model):
    """PREREG_07 5.2 GPU seam: the 400 seam games against val_hs.npz rows 0-399 and val_seam_moves.npz."""
    n = SEAM_GAMES
    probs = []
    if len(z["sums"]) != n:
        probs.append(f"seam has {len(z['sums'])} games, registered {n}")
    k = min(n, len(z["sums"]))
    f_eq = np.array_equal(np.asarray(z["failed"][:k], dtype=bool), np.asarray(vh["failed"][:k], dtype=bool))
    c_eq = np.array_equal(np.asarray(z["counts"][:k]), np.asarray(vh["counts"][:k]))
    m_eq = np.array_equal(np.asarray(z["n_moves"][:k]), np.asarray(vh["n_moves"][:k]))
    keys_eq = (list(map(str, z["question_id"][:k])) == list(map(str, vh["question_id"][:k]))
               and np.array_equal(np.asarray(z["slot"][:k]), np.asarray(vh["slot"][:k]))
               and list(map(str, z["color"][:k])) == list(map(str, vh["color"][:k])))
    counts_equal = bool(f_eq and c_eq and m_eq and keys_eq and k == n)
    if not counts_equal:
        probs.append(f"failed equal {f_eq}, counts equal {c_eq}, n_moves equal {m_eq}, keys equal {keys_eq}")
    # per move, aligned by (game, turn)
    a_key = np.asarray(z["pos_game"]).astype(np.int64) * 100000 + np.asarray(z["pos_turn"]).astype(np.int64)
    b_key = np.asarray(vm["pos_game"]).astype(np.int64) * 100000 + np.asarray(vm["pos_turn"]).astype(np.int64)
    same_moves = len(a_key) == len(b_key) and np.array_equal(np.sort(a_key), np.sort(b_key))
    if not same_moves:
        probs.append(f"per-move positions differ ({len(a_key)} vs {len(b_key)})")
        max_move = float("inf")
        bit_moves = False
    else:
        ia, ib = np.argsort(a_key, kind="stable"), np.argsort(b_key, kind="stable")
        la = np.asarray(z["pos_logp"], dtype=np.float64)[ia]
        lb = np.asarray(vm["pos_logp"], dtype=np.float64)[ib]
        max_move = float(np.max(np.abs(la - lb))) if len(la) else 0.0
        bit_moves = bool(np.array_equal(np.asarray(z["pos_logp"])[ia], np.asarray(vm["pos_logp"])[ib]))
    ok_g = ~np.asarray(vh["failed"][:k], dtype=bool) & (np.asarray(vh["counts"][:k]) > 0)
    if counts_equal and ok_g.any():
        ma = np.asarray(z["sums"][:k], dtype=np.float64)[ok_g] / np.asarray(z["counts"][:k])[ok_g, None]
        mb = np.asarray(vh["sums"][:k], dtype=np.float64)[ok_g] / np.asarray(vh["counts"][:k])[ok_g, None]
        max_mean = float(np.max(np.abs(ma - mb)))
        la_, _ = pf7.hs_lp(model, z["sums"][:k], z["counts"][:k], z["failed"][:k])
        lb_, _ = pf7.hs_lp(model, vh["sums"][:k], vh["counts"][:k], vh["failed"][:k])
        max_lp = float(np.max(np.abs(la_ - lb_)))
    else:
        max_mean = max_lp = float("inf")
    bit_sums = bool(counts_equal and np.array_equal(np.asarray(z["sums"][:k]), np.asarray(vh["sums"][:k])))
    passed = bool(counts_equal and same_moves and max_move <= TOL_MOVE and max_mean <= TOL_PROFILE_MEAN
                  and max_lp <= TOL_LP)
    fin = lambda v: v if math.isfinite(v) else None  # noqa: E731
    return {"games": int(k), "counts_equal": counts_equal, "moves_compared": int(len(a_key)),
            "max_abs_move": fin(max_move), "max_abs_profile_mean": fin(max_mean), "max_abs_lp_hs": fin(max_lp),
            "tolerances": {"move": TOL_MOVE, "profile_mean": TOL_PROFILE_MEAN, "lp_hs": TOL_LP},
            "lp_hs_review_line": LP_REVIEW, "lp_hs_above_review_line": bool(math.isfinite(max_lp) and
                                                                           max_lp > LP_REVIEW),
            "bitwise_equal_sums": bit_sums, "bitwise_equal_moves": bit_moves, "problems": probs, "passed": passed}


def cmd_apply(a):
    t0 = time.time()
    ctx = resolve(a, "apply")
    run = Apply(ctx, a, t0)
    try:
        return run.go()
    except T07Error as e:
        if e.code in (3, 4):
            run.write_verdict(e)
            if e.code == 3:
                print(f"07 不用（PREREG_07 5.2）：{e}\ncommit {shown(ctx['R7'])}；最終照 PREREG_04 5.3（不重跑）",
                      file=sys.stderr)
        raise


# ================================================================ status (read only)
def cmd_status(a):
    ctx = resolve(a, "status")
    lines = [f"test07 status ({ctx['mode']}; {ctx['R7']})"]
    for s in SETS:
        d = os.path.join(ctx["H"], s)
        n = len(chunk_paths(d)) if os.path.isdir(d) else 0
        stop = os.path.exists(os.path.join(d, "STOP"))
        stale = sorted(os.path.basename(x) for x in glob.glob(os.path.join(ctx["H"], f"{s}_stale_*")))
        rp = os.path.join(ctx["hs"], f"hs_{s}_report.json")
        r = read_json_or_none(rp)
        lines.append(f"hs {s}: {n} chunk(s) in {shown(d)}" + (" STOP file present" if stop else "")
                     + (f"; stale folders {stale}" if stale else "")
                     + (f"; report: {r.get('games')} games, failed {r.get('failed_or_no_move')}, "
                        f"{r.get('evals_per_s')} evals/s" if r else "; no report yet")
                     + (f", committed {committed_clean(rp)}" if r and ctx["real"] else ""))
    rp = os.path.join(ctx["rank"], "t07_apply_report.json")
    vp = os.path.join(ctx["rank"], "attempt_verdict.json")
    np_ = os.path.join(ctx["R7"], "not_used.json")
    if os.path.exists(rp):
        r = read_json(rp)
        lines.append(f"apply: exit {r.get('exit_code')}; submission sha256 "
                     f"{str((r.get('outputs_sha256') or {}).get('submission_rank.csv'))[:12]}; GPU seam "
                     f"{(r.get('gpu_seam') or {}).get('passed')}; local seam {(r.get('seam_local') or {}).get('verdict')}")
    elif os.path.exists(np_):
        r = read_json(np_)
        lines.append(f"07 not used: {r.get('stage')} ({r.get('reason')})")
    elif os.path.exists(vp):
        r = read_json(vp)
        lines.append(f"apply: attempt_verdict exit {r.get('exit_code')} at {r.get('stage')} ({r.get('reason')})")
    else:
        lines.append("apply: not run")
    if ctx["R4"]:
        r4 = read_json_or_none(os.path.join(ctx["R4"], "rank", "t04_apply_report.json"))
        lines.append("test04 apply: " + (f"exit {r4.get('exit_code')}, usable {r4.get('usable')}" if r4 else "not run"))
    left = K6_07 - now()
    lines.append(f"K6-07 {K6_07.isoformat()}: " + (f"{left.days} days {left.seconds // 3600} h left"
                                                   if left.total_seconds() > 0 else "passed"))
    print("\n".join(lines))
    return 0


# ================================================================ command line
def build_parser():
    ap = argparse.ArgumentParser(prog="t07_rank.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("hs", "apply", "status"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--rehearsal", action="store_true")
        for d in ("--results-dir", "--hs-data-dir", "--t04-results-dir", "--t02-results-dir", "--t02-out-dir",
                  "--tests-dir", "--checks-dir"):
            p.add_argument(d)
        p.add_argument("--r20-uploaded-at", help="drills only: R20's first upload time")
        p.add_argument("--now", help="drills only: the current time (K6-07)")
        if name in ("hs", "apply"):
            p.add_argument("--rerun", action="store_true")
            p.add_argument("--reason")
        if name == "hs":
            p.add_argument("--set", required=True, choices=SETS)
            p.add_argument("--workers", type=int, default=pf7.RUN["workers"])
            p.add_argument("--stop-after-min", type=float, default=0)
            p.add_argument("--max-games", type=int, default=0)
            p.add_argument("--fake-scorer", action="store_true")
            p.add_argument("--ignore-other-gpu-jobs", action="store_true")
        if name == "apply":
            p.add_argument("--bug-revision", type=int)
    return ap


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    try:
        return {"hs": cmd_hs, "apply": cmd_apply, "status": cmd_status}[a.cmd](a)
    except T07Error as e:
        print(f"{'REFUSED' if e.code == 2 else 'NOT USED' if e.code == 3 else 'DRIFT'} (exit {e.code}): {e}",
              file=sys.stderr)
        return e.code
    except Exception:  # noqa: BLE001 - an internal error is a refusal with its traceback, never a decision
        traceback.print_exc()
        print("ERROR (exit 2): unexpected internal error of tools/t07_rank.py; stop and check the local tool",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
