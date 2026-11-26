"""C10 test09, competition 2 (rank, TEAM_11248): the 09 test-day local tool (Q09 = Q02H + 4 x P1TO_E, ens3fam).

Registration: docs/PREREG_09_MAPPING_ENSEMBLE.md section 5 (rev 1), the adoption record results/prereg09/RESULTS.md
and the test-day revisions (rev 3, section 13; rev 4, section 14). CPU only, after 08's apply: 09 reads 08's P1 chunks and never runs the
GPU. The 08 test-day tool tools/t08_rank.py (and through it t07 / t04) is imported, never edited.

    PY tools/t09_rank.py apply  [MODE] [DIRS] [--rerun --reason TEXT [--bug-revision N]]   (CPU, .venv)
    PY tools/t09_rank.py status [MODE] [DIRS]                                              (read only)
    (PY = .venv/Scripts/python.exe; Git Bash; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL thread variables)

MODE  none = real; at most one of --rehearsal, --dry-run
DIRS  --results-dir R9      test09 root (writes R9/gpu_seam.json, R9/rank; real results/test09)
      --t08-results-dir R8  test08 root (reads R8/p1, R8/rank; real results/test08)
      --p1-data-dir H       08's P1 chunks (real data/test08/p1; otherwise R8/p1_data, or data/test08_rehearsal/p1
                            for R8 = results/test08_rehearsal)
      --t07-results-dir R7, --t04-results-dir R4, --t02-results-dir D2, --t02-out-dir O2, --tests-dir T,
      --checks-dir C        as tools/t08_rank.py
Real mode: every directory is the repository path above. --rehearsal: R9 is results/test09_rehearsal (or outside the
repository); --dry-run: R9 outside the repository. The other trees are required outside the real mode.
Drills only (never real): --r20-uploaded-at ISO, --now ISO.

apply  Q09 on public and private (PREREG_09 5.2), in this order (the first failing item decides):
       prerequisites: git clean (real); .venv python / numpy / scipy = results/prereg09/rank/frozen_config.json; no
       BLAS thread variable; the committed prereg09 files equal their pinned SHA-256s and record the adopted Q09
       (frozen_config.json gates passed, apply_report.json usable true and finalized); K6-09 not passed; every
       prerequisite of tools/t08_rank.py apply (test02, R20's first upload, T1, H7R, H8R, test04, test07); 08's
       t08_apply_report.json of this mode exit 0, usable and committed (08 judged 'not used' -> 09 not used, 3), every
       input 08 bound unchanged.
       1 the three P1 sets of 08 (R8/p1/p1_<set>_report.json equal to the SHA-256 08's report recorded, every chunk
         in H/<set>/ equal to the SHA-256 its report lists, counts / n_moves / failed equal <set>_p1.npz); X (1,540)
         and EX (44) from the chunks (tools/p09_forward.py, as the validation apply).
       2 the 09 GPU seam: the 400 seam games against the validation chunks data/prereg08/val/chunk_00000-00001.npz
         (SHA-256 against results/prereg08/feat/val_report.json): failed / counts / n_moves equal, the standardised
         EX <= 0.05 over the 42 features other than the two low-probability fractions (t_frac_lowp, o_frac_lowp:
         recorded only, PREREG_09 rev 4) and lp_P1TO_E per game (before the x 4) <= TOL_E (measured on the 4090
         rehearsal before any test data; PREREG_09 rev 3) -> R9/gpu_seam.json; else 4.
       3 Q08 recomputed with 08's frozen map (tools/t08_rank.py q08_set on test02 / 04 / 07's inputs) equal to 08's
         committed rank_test_probs.csv (probabilities within 1e-12, rule A and rule B levels equal; byte equality
         recorded); 09's usable games equal 08's test_lp_p1.npz p1_ok / usable; else 4.
       4 the local Q09 seam on the 1,589 validation questions (07's seam_val inputs + the committed val_lp_p1e.npz)
         against results/prereg09/rank/rank_val_probs.csv / pred_B.csv (08's seam_local.json byte-identical ->
         byte-identical required and the seam_val rho equal to prereg09's apply_report.json; else the tolerance of
         test04, at most 4 rule-B differences; outside it -> 3).
       5 P1 failures (200 x failed <= games per set; 08 already judged) -> 3; 6 the direction check per set
         (rho >= max(0.5 x rho_mono_09, 0.155)) -> 3.
       Then rank_test_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json, submission_rank.csv (= pred_B.csv),
       test_lp_p1e.npz, check_submission --official and R9/rank/t09_apply_report.json.
status Read-only summary.

Exit codes (as tools/t08_rank.py): 0 done; 2 refused or error; 3 the registered "09 is not used" (R9/not_used.json;
commit; the final choice follows PREREG_08 5.4; never rerun); 4 local drift or mismatch (R9/rank/attempt_verdict.json;
commit, stop and find out why, then --rerun --reason). After exit 3 or a written t09_apply_report.json nothing is
rerun except with --bug-revision N (a committed PREREG_09 rev N >= 5 added after that record).

Discipline: never imports torch, u1_lib, p09_map or a player module; imports pilot_fuse / p1_fuse only after every
data/local read of the process; never fits anything (the frozen p1e_map.npz, numpy float64 forward); no validation
label file is ever opened; tools/score.py is never run; temporary files only under data/test09/tmp (real),
data/test09_rehearsal/tmp or R9/tmp. Competition 1 is never read (PREREG_09 6).
"""

import argparse
import datetime as dt
import hashlib
import io
import csv
import json
import math
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SELF = os.path.join(HERE, "t09_rank.py")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t08_rank as t8  # noqa: E402
import p09_rank_fuse as r9  # noqa: E402
import p09_forward as FW  # noqa: E402

t7, t4, pf7, r8 = t8.t7, t8.t4, t8.pf7, t8.r8
rf, t2 = t8.rf, t8.t2
decide, check_submission = t8.decide, t8.check_submission

# ================================================================ paths (module constants; the unit tests patch them)
RESULTS = os.path.join(REPO, "results")
REAL_R9 = os.path.join(RESULTS, "test09")
REHEARSAL_R9 = os.path.join(RESULTS, "test09_rehearsal")
REAL_TMP = os.path.join(REPO, "data", "test09", "tmp")
REHEARSAL_TMP = os.path.join(REPO, "data", "test09_rehearsal", "tmp")
REAL_R8, REAL_H8 = t8.REAL_R8, t8.REAL_H
REHEARSAL_R8, REHEARSAL_H8 = t8.REHEARSAL_R8, t8.REHEARSAL_H
REAL_R7, REAL_R4 = t8.REAL_R7, t8.REAL_R4
REAL_D2, REAL_O2, REAL_T, REAL_C = t8.REAL_D2, t8.REAL_O2, t8.REAL_T, t8.REAL_C
P09_RANK = os.path.join(RESULTS, "prereg09", "rank")
P09_RESULTS = os.path.join(RESULTS, "prereg09", "RESULTS.md")
P09_DOC = os.path.join(REPO, "docs", "PREREG_09_MAPPING_ENSEMBLE.md")
VAL_REPORT = os.path.join(RESULTS, "prereg08", "feat", "val_report.json")

# ================================================================ registered values
# the committed prereg09 files the test day reads (SHA-256: .json / .csv with CRLF folded to LF, .npz raw bytes)
P09_PINS = {
    "results/prereg09/rank/frozen_config.json": "50d38d2787c94569d88ed87a482e872ed64be2d1d20e4f57ffbb07a216663ff5",
    "results/prereg09/rank/p1e_map.npz": "ecc9c9b1747b7b7d40cf53869b6c67c8e6f776ba7498498737fe3eede8e2edba",
    "results/prereg09/rank/rank_val_probs.csv": "3f8d32f4b5b0431c9f4407199ae2c074595428ef9053ab807a5cf3b129e3b4fb",
    "results/prereg09/rank/pred_A.csv": "5c412993c6f72a7169327c8a2575cc7b3be44195ad3e198d30712d73d62432e0",
    "results/prereg09/rank/pred_B.csv": "d1a2f1d673a8b95425dfdc6b392a64a3e7484509d8bbaa3eb3c72f5ee283333b",
    "results/prereg09/rank/val_lp_p1e.npz": "dc70be8a0f57dddb51836f8fb47ab598ac02ffaac6803949fdf4770004463cdf",
    "results/prereg09/rank/apply_report.json": "691cd3a7c06e86947d96af1d2faf152c52006e336ebe15e81124ebeb3fd773b3",
    "results/prereg09/rank/apply_audit.json": "12cb0145f31780d7dcddfa4dd16c726e5e1cf4eb9c0a5906335d2df6d994609d"}
MAP_SHA = P09_PINS["results/prereg09/rank/p1e_map.npz"]
RHO_MONO = 0.9685160520465519                     # frozen_config.json direction_ref.rho_mono (PREREG_09 5.2)
RHO_FLOOR = r9.RHO_FLOOR_TEST                     # 0.155: one-sided p = 0.001 for 400 questions
RHO_EVAL = 0.9633910264688432                     # frozen_config.json direction_ref.rho_eval (seam_val record)
SEAM_GAMES = t8.SEAM_GAMES                        # 400 = the validation chunks 0 and 1
TOL_EX = 0.05                                     # PREREG_09 5.2 GPU seam: standardised EX (the judged 42 features)
# PREREG_09 rev 4 (section 14): the two hard-threshold count features (fraction of moves with log p < log 0.05, target
# and opponent side) jump a step under fp16 rounding; the seam records them (largest difference, games that jumped)
# but does not judge them. lp_P1TO_E <= TOL_E is what guards the predictions.
EX_RECORD_ONLY = ("t_frac_lowp", "o_frac_lowp")
EX_JUDGED = [i for i, n in enumerate(FW.ex_names()) if n not in EX_RECORD_ONLY]
EX_RECORDED = [i for i, n in enumerate(FW.ex_names()) if n in EX_RECORD_ONLY]
# lp_P1TO_E per game (before the x 4): max(0.04, 2.5 x m), m = the 4090 rehearsal's largest difference on games
# off the validation chunk grid (PREREG_09 5.2; rev 3 section 13; results/test09/design/tol_e_measurement.json)
TOL_E = 0.04
LP_REVIEW = 0.005                                 # as 07 / 08: a measured lp difference above this is reviewed
TOL_Q08 = 1e-12                                   # PREREG_08 rev 3 item 9: Q08 recomputed vs 08's file
TW = t8.TW
K6_09 = dt.datetime(2026, 11, 19, 23, 59, 59, tzinfo=TW)
P09_MIN_REV = 5                                   # rev 3 / rev 4 register the test day; an override is rev 5 or later
SETS = t8.SETS
TEST_SETS = t8.TEST_SETS
Q_SETS = t8.Q_SETS
COMBO = {"id": "Q09", "sources": ["C5", "OP", "O4", "SWA", "SWB", "HS", "P1TO_E"],
         "terms": ["OP", "O4", "SWA", "SWB", "HS", "P1TO"], "p1_scale": 4.0, "p1_term": "P1TO_E (ens3fam)"}
METHOD = "prereg09 Q09 = Q02H + 4 x P1TO_E (ens3fam), rule B (甲)"
FORMATS = {"gpu_seam": "c10-test09 gpu seam v1", "seam_local": "c10-test09 seam_local v1",
           "apply": "c10-test09 rank apply v1", "verdict": "c10-test09 attempt verdict v1",
           "not_used": "c10-test09 not used v1", "lp": "c10-test09 test_lp_p1e v1"}
TEST_OUTPUTS = ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv",
                "test_lp_p1e.npz")
POS_KEYS = ("pos_game", "pos_side", "pos_turn", "pos_logp", "pos_entropy")
_LABEL = "ans" + "wer"


class T09Error(RuntimeError):
    """A stop with its exit code: 2 refused, 3 registered 'no 09', 4 local drift / mismatch."""

    def __init__(self, message, code=2, stage=None, reason=None, details=None):
        super().__init__(message)
        self.code, self.stage, self.reason, self.details = code, stage, reason, details


def refuse(msg, reason=None):
    raise T09Error(msg, 2, reason=reason)


def not_used(msg, stage, reason, details=None):
    raise T09Error(msg, 3, stage, reason, details)


def drift(msg, stage, reason, details=None):
    raise T09Error(msg, 4, stage, reason, details)


FOREIGN_ERRORS = tuple(dict.fromkeys(t8.FOREIGN_ERRORS + r9.ERRORS))


def t8call(fn, *args):
    """A tools/t08_rank.py function (and, through it, t07 / t04): its exit code is ours (2 refused, 3 registered
    'no 08 / 07 / 04' = no 09, 4 drift)."""
    try:
        return fn(*args)
    except t8.T08Error as e:
        raise T09Error(f"(t08_rank) {e}", e.code, e.stage, e.reason or "t08_prerequisite", e.details)
    except FOREIGN_ERRORS as e:
        refuse(f"(t08_rank) {e}")


# ================================================================ small helpers
def now():
    """The current time (a module function: the tests replace it to cross K6-09)."""
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
        refuse(f"refusing {path}: validation label files are never opened by the test09 tools")
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


def lf_bytes(path):
    with open(refuse_label_path(path), "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


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
    names = ["t09_rank.py", "t08_rank.py", "t07_rank.py", "t04_rank.py", "p09_rank_fuse.py", "p09_forward.py",
             "p08_rank_fuse.py", "p08_p1.py", "p07_rank_fuse.py", "p07_hs.py", "check_submission.py",
             "official_log.py", "t02_rank.py", "t02_test_bundle.py", "p04_opp.py", "p04_rank_fuse.py",
             "p03_rank_fuse.py", "p02_rank_fuse.py", "c10data.py", "decide.py"]
    return {n: (sha_lf(os.path.join(HERE, n)) if os.path.exists(os.path.join(HERE, n)) else None) for n in names}


def tool_record():
    return {"git_head": git_head(), "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": tool_files()}


def rate_ok(failures, games):
    """failures / games <= 0.005, in integers (200 * failures <= games): 25 of 5,000 passes, 26 does not."""
    return t8.rate_ok(failures, games)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def blas_env_check():
    """The byte seams (Q08 = 08, seam_val = prereg09) need OpenBLAS's default thread count (PREREG_09 2.6)."""
    set_ = {k: os.environ[k] for k in r8.BLAS_ENV if os.environ.get(k)}
    if set_:
        refuse(f"BLAS thread variables are set ({set_}): unset them (the registered byte seams use the default "
               f"OpenBLAS threads; PREREG_09 1.3 item 7 / 2.6)")
    return {k: None for k in r8.BLAS_ENV}


# ================================================================ modes and directories
def resolve(a, cmd):
    flags = [m for m, on in (("dry-run", a.dry_run), ("rehearsal", a.rehearsal)) if on]
    if len(flags) > 1:
        refuse(f"give at most one of --dry-run, --rehearsal (got {flags})")
    mode = flags[0] if flags else "real"
    given = {"--results-dir": a.results_dir, "--t08-results-dir": a.t08_results_dir,
             "--t07-results-dir": a.t07_results_dir, "--t04-results-dir": a.t04_results_dir,
             "--t02-results-dir": a.t02_results_dir, "--t02-out-dir": a.t02_out_dir, "--tests-dir": a.tests_dir}
    real = {"--results-dir": REAL_R9, "--t08-results-dir": REAL_R8, "--t07-results-dir": REAL_R7,
            "--t04-results-dir": REAL_R4, "--t02-results-dir": REAL_D2, "--t02-out-dir": REAL_O2,
            "--tests-dir": REAL_T}
    drills = [k for k, v in (("--r20-uploaded-at", getattr(a, "r20_uploaded_at", None)),
                             ("--now", getattr(a, "now", None))) if v]
    if mode == "real":
        if drills:
            refuse(f"real mode: {drills} are for drills only (--dry-run / --rehearsal)")
        for k, v in given.items():
            if v and not t2.same_path(v, real[k]):
                refuse(f"real mode: {k} must be {real[k]} (got {v})")
        for k, v, want in (("--checks-dir", a.checks_dir, REAL_C), ("--p1-data-dir", a.p1_data_dir, REAL_H8)):
            if v and not t2.same_path(v, want):
                refuse(f"real mode: {k} must be {want} (got {v})")
        R9, R8, H, R7, R4, D2, O2, T, C, TMP = (REAL_R9, REAL_R8, REAL_H8, REAL_R7, REAL_R4, REAL_D2, REAL_O2, REAL_T,
                                                REAL_C, REAL_TMP)
    else:
        need = ["--results-dir"] if cmd == "status" else list(given)
        if mode == "rehearsal" and not a.results_dir:
            given["--results-dir"] = REHEARSAL_R9
        missing = [k for k in need if not given[k]]
        if missing:
            refuse(f"--{mode} needs {', '.join(missing)} (other modes never fall back to the repository paths)")
        R9 = os.path.abspath(given["--results-dir"])
        ab = lambda v: os.path.abspath(v) if v else None   # noqa: E731
        R8, R7, R4, D2, O2, T = (ab(a.t08_results_dir), ab(a.t07_results_dir), ab(a.t04_results_dir),
                                 ab(a.t02_results_dir), ab(a.t02_out_dir), ab(a.tests_dir))
        C = os.path.abspath(a.checks_dir) if a.checks_dir else (os.path.join(D2, "checks") if D2 else None)
        in_repo_ok = mode == "rehearsal" and t2.same_path(R9, REHEARSAL_R9)
        if t2.is_inside(R9, REPO) and not in_repo_ok:
            refuse(f"--{mode}: --results-dir must be outside the repository (only --rehearsal may use "
                   f"{REHEARSAL_R9}); got {R9}")
        if a.p1_data_dir:
            H = os.path.abspath(a.p1_data_dir)
        elif R8 and t2.same_path(R8, REHEARSAL_R8):
            H = REHEARSAL_H8
        else:
            H = os.path.join(R8, "p1_data") if R8 else None
        if R8 and t2.is_inside(R8, REPO) and not t2.same_path(R8, REHEARSAL_R8):
            refuse(f"--{mode}: --t08-results-dir inside the repository must be {REHEARSAL_R8} (got {R8})")
        if H and t2.is_inside(H, REPO) and not t2.is_inside(H, os.path.dirname(REHEARSAL_H8)):
            refuse(f"--{mode}: --p1-data-dir inside the repository must be under data/test08_rehearsal (got {H})")
        TMP = REHEARSAL_TMP if in_repo_ok else os.path.join(R9, "tmp")
    for p in (R9, R8, H, R7, R4, D2, O2, T, C):
        if p is not None:
            refuse_label_path(p)
            if pf7.under_data_local(p):
                refuse(f"{p}: never under data/local")
    if T is not None and os.path.basename(os.path.normpath(T)).lower() == "raw":
        refuse(f"--tests-dir {T}: the test files are the checked copies (data/test), never data/test/raw")
    bad = forbidden_area(R9)
    if bad:
        refuse(f"output folder inside the read-only {bad}")
    return {"mode": mode, "real": mode == "real", "smoke": False, "cmd": cmd, "R9": R9, "R8": R8, "H": H, "R7": R7,
            "R4": R4, "D2": D2, "O2": O2, "T": T, "C": C, "TMP": TMP, "rank9": os.path.join(R9, "rank"),
            "rank": os.path.join(R8, "rank") if R8 else None, "p1": os.path.join(R8, "p1") if R8 else None,
            "write_roots": [R9, TMP]}


def forbidden_area(path):
    """Where t09 never writes: results/prereg0*, results/test02* ... test08*, data/local, the ledger."""
    p = os.path.abspath(path)
    if pf7.under_data_local(p):
        return "data/local"
    if t2.same_path(p, os.path.join(RESULTS, "score_ledger.csv")):
        return "results/score_ledger.csv"
    if t2.is_inside(p, RESULTS) and not t2.same_path(p, RESULTS):
        first = os.path.relpath(os.path.realpath(p), os.path.realpath(RESULTS)).split(os.sep)[0].lower()
        if first.startswith(("prereg0", "test02", "test04", "test06", "test07", "test08")):
            return "results/" + first
    if t2.is_inside(p, os.path.join(REPO, "data")):
        first = os.path.relpath(os.path.realpath(p), os.path.realpath(os.path.join(REPO, "data"))).split(os.sep)[0]
        if first.lower().startswith(("test08", "prereg0")):
            return "data/" + first
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


def save_npz(ctx, path, **arrays):
    guard_write(ctx, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


# ================================================================ prerequisites
def check_git_clean(ctx, rec):
    """Real mode: no uncommitted change to a tracked file; this tool and its dependencies committed."""
    if not ctx["real"]:
        rec["git"] = f"not checked ({ctx['mode']})"
        return
    rc, st = git("status", "--porcelain", "--untracked-files=no")
    if rc != 0:
        refuse("git is not available")
    if st:
        refuse(f"commit or restore the tracked changes first (git status: {st.splitlines()[:6]})")
    own = [SELF] + [os.path.join(HERE, n) for n in ("t08_rank.py", "t07_rank.py", "t04_rank.py", "p09_rank_fuse.py",
                                                    "p09_forward.py", "p08_rank_fuse.py", "check_submission.py",
                                                    "official_log.py")]
    rec["git"] = {"clean": True, "head": git_head(), "own_files": require_committed(own, "the test09 tools")}


def check_venv(fc9):
    """.venv python / numpy / scipy equal frozen_config.json's versions (PREREG_09 5.2); else 4. Never imports
    torch."""
    want = {k: ((fc9.get("environment") or {}).get("versions") or {}).get(k) for k in ("python", "numpy", "scipy")}
    got = r9.venv_versions(False)
    bad = {k: {"now": got.get(k), "frozen": want[k]} for k in want if got.get(k) != want[k]}
    if bad:
        drift(f".venv differs from results/prereg09/rank/frozen_config.json {bad}: restore .venv (never upgraded "
              f"before 11/24)", "prerequisites", "venv_versions", bad)
    return {"equal": True, "versions": got}


def check_p09_pins():
    """The committed prereg09 files equal their pinned SHA-256s; frozen_config.json is the adopted Q09 (gates
    passed, map = p1e_map.npz, scale 4, direction reference); the apply report is usable and finalized; RESULTS.md
    exists. -> (frozen config, loaded map, record)."""
    got = {r: (pin_sha(path_of(r)) if os.path.exists(path_of(r)) else None) for r in P09_PINS}
    bad = sorted(r for r in P09_PINS if got[r] != P09_PINS[r])
    if bad:
        refuse(f"prereg09 files differ from the pinned SHA-256s: {bad} (never edited after the adoption)")
    fc = read_json(path_of("results/prereg09/rank/frozen_config.json"))
    try:
        r9.frozen_usable(fc)
    except FOREIGN_ERRORS as e:
        refuse(f"frozen_config.json: {e}")
    combo = fc.get("combo") or {}
    if combo.get("id") != COMBO["id"] or combo.get("terms") != COMBO["terms"] or float(combo.get("p1_scale", 0)) != \
            FW.SCALE or (fc.get("map") or {}).get("sha256") != MAP_SHA or (fc.get("map") or {}).get("file") != \
            "p1e_map.npz":
        refuse("results/prereg09/rank/frozen_config.json is not the frozen Q09 configuration")
    dref = fc.get("direction_ref") or {}
    if float(dref.get("rho_mono")) != RHO_MONO or float(dref.get("rho_eval")) != RHO_EVAL:
        refuse("frozen_config.json direction_ref differs from the registered values")
    rep = read_json(path_of("results/prereg09/rank/apply_report.json"))
    if rep.get("usable") is not True or not rep.get("finalized") or rep.get("combo") != "Q09":
        not_used("PREREG_09 2.7: the validation apply of Q09 is not usable / finalized: 09 is not used",
                 "prerequisites", "p09_not_adopted")
    if not os.path.exists(P09_RESULTS):
        refuse("results/prereg09/RESULTS.md is missing (the adoption record)")
    try:
        mp = FW.load_map(path_of("results/prereg09/rank/p1e_map.npz"))
    except FOREIGN_ERRORS as e:
        refuse(str(e))
    if mp["scale"] != FW.SCALE:
        refuse("p1e_map.npz scale is not 4")
    return fc, mp, {"prereg09_sha256": got, "adopted": {"apply_usable": True, "finalized": rep.get("finalized"),
                                                        "results_md_sha256": sha_lf(P09_RESULTS)}}


def earlier_success(ctx):
    """A committed t09_apply_report.json with exit_code 0, committed by K6-09 (real mode) -> (bool, record)."""
    p = os.path.join(ctx["rank9"], "t09_apply_report.json")
    if not os.path.exists(p):
        return False, None
    rep = read_json(p)
    if rep.get("exit_code") != 0:
        return False, None
    if not ctx["real"]:
        return True, {"note": f"{ctx['mode']}: commit time not checked"}
    h, t = commit_time(p)
    ok = t is not None and parse_iso(t, "commit time") <= K6_09
    return ok, {"commit": h, "committed_at": t}


def check_k6(ctx, a):
    """K6-09 not passed; after it, exit 3 without an earlier success, 2 with one (that one stays valid)."""
    t_now = parse_iso(a.now, "--now") if getattr(a, "now", None) else now()
    if t_now <= K6_09:
        return {"now": t_now.isoformat(), "k6_09": K6_09.isoformat(), "passed": False}
    ok, rec = earlier_success(ctx)
    if ok:
        refuse(f"K6-09 {K6_09.isoformat()} has passed; the successful run before it ({rec}) stays valid: nothing is "
               f"rerun", reason="k6_passed_success")
    not_used(f"K6-09 {K6_09.isoformat()} has passed without a successful official 09 run: 09 is not used this "
             f"season (PREREG_09 4); the final choice follows PREREG_08 5.4", "prerequisites", "k6_09")


def bound_problems(ctx, bound, roots_extra=None):
    """Every input a test report bound (keys: real repository paths) still has its SHA-256 (tree paths outside the
    real mode)."""
    probs = []
    if not bound:
        return ["the report lists no inputs_bound"]
    roots = {"results/test08/": ctx["R8"], "results/test07/": ctx["R7"], "results/test04/": ctx["R4"],
             "results/test02/checks/": ctx["C"], "results/test02/": ctx["D2"]}
    roots.update(roots_extra or {})
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


def t08_state(ctx):
    """test08's apply of this mode: exit 0 and usable (committed in the real mode), every input it bound unchanged.
    08 judged 'not used' -> 3 (09 is not used either, PREREG_09 5.2); not done yet -> 2. -> (report, record)."""
    rp, vp = os.path.join(ctx["rank"], "t08_apply_report.json"), os.path.join(ctx["rank"], "attempt_verdict.json")
    nu = read_json_or_none(os.path.join(ctx["R8"], "not_used.json"))
    if nu is not None and nu.get("exit_code") == 3 and nu.get("mode") == ctx["mode"]:
        not_used(f"test08 judged '08 is not used' ({nu.get('stage')}: {nu.get('reason')}): 09 is not used either "
                 f"(PREREG_09 5.2)", "prerequisites", "t08_not_used")
    if not os.path.exists(rp):
        refuse("test08 apply has not finished (no t08_apply_report.json"
               + (f"; attempt_verdict.json says exit {read_json(vp).get('exit_code')}" if os.path.exists(vp) else "")
               + "): run docs/TESTDAY_08.md step 4 first", reason="t08_not_done")
    rep = read_json(rp)
    if rep.get("format") != t8.FORMATS["apply"] or rep.get("mode") != ctx["mode"]:
        refuse(f"{shown(rp)}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (this run: {ctx['mode']})")
    if rep.get("exit_code") == 3 or rep.get("usable") is False:
        not_used("test08 apply judged '08 is not used': 09 is not used either", "prerequisites", "t08_not_used")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True or (rep.get("combo") or {}).get("id") != "Q08":
        refuse(f"{shown(rp)}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    files = [rp] + [os.path.join(ctx["rank"], n) for n in t8.TEST_OUTPUTS] + \
        [os.path.join(ctx["rank"], "seam_val", "seam_local.json"), os.path.join(ctx["p1"], "gpu_seam.json")] + \
        [p for s in SETS for p in t8.p1_outputs({"p1": ctx["p1"]}, s)]
    if ctx["real"]:
        require_committed(files, "results/test08")
    missing = [shown(p) for p in files if not os.path.exists(p)]
    if missing:
        refuse(f"test08 outputs missing: {missing[:5]}")
    b8 = bound_problems(ctx, rep.get("inputs_bound") or {})
    if b8:
        refuse(f"test08's bound inputs changed since its apply: {b8[:5]} (follow docs/TESTDAY_08.md first)",
               reason="t08_inputs_changed")
    want = (rep.get("outputs_sha256") or {})
    changed = [n for n in t8.TEST_OUTPUTS if sha_raw(os.path.join(ctx["rank"], n)) != want.get(n)]
    if changed:
        refuse(f"test08 outputs differ from its report: {changed}", reason="t08_outputs_changed")
    return rep, {"path": shown(rp), "sha256": pin_sha(rp), "exit_code": 0, "usable": True,
                 "inputs_bound_checked": len(rep.get("inputs_bound") or {}),
                 "submission_sha256": want.get("submission_rank.csv")}


def prereqs(ctx, a):
    """PREREG_09 5.2 in this order. -> (fc09, map09, fc07, fc08, model08, fc04, t02, rep7, rep4, rep8, record)."""
    rec = {}
    check_git_clean(ctx, rec)                                            # 1
    rec["blas_env"] = blas_env_check()                                   # 2
    fc9, mp, rec["prereg09"] = check_p09_pins()                          # 3 (09 adopted)
    rec["venv"] = check_venv(fc9)
    rec["k6_09"] = check_k6(ctx, a)                                      # 4
    fc07, fc08, model08, fc04, t02, rep7, rep4, rec["t08_prerequisites"] = t8call(t8.prereqs, ctx, "apply", a)  # 5
    rep8, rec["t08_apply_report"] = t08_state(ctx)                       # 6
    return fc9, mp, fc07, fc08, model08, fc04, t02, rep7, rep4, rep8, rec


# ================================================================ features of 08's chunks (pure numpy parts)
def pos_problems(z, G):
    """PREREG_09 rev 4 addendum: the per-position arrays of a chunk are complete when they have one length, pos_game
    is in [0, G), pos_side in {0, 1} and the positions per game and side equal counts exactly
    (bincount(pos_game * 2 + pos_side, minlength = 2 G) reshaped [G, 2] == counts; holds on every 08 validation, GPU
    and drill chunk). -> list of problems (empty when complete)."""
    n = {k: len(np.asarray(z[k])) for k in POS_KEYS}
    if len(set(n.values())) != 1:
        return [f"lengths {n}"]
    pg, ps = np.asarray(z["pos_game"]).astype(np.int64), np.asarray(z["pos_side"]).astype(np.int64)
    if len(pg) and (pg.min() < 0 or pg.max() >= G or ps.min() < 0 or ps.max() > 1):
        return ["pos_game / pos_side out of range"]
    got = np.bincount(pg * 2 + ps, minlength=2 * G).reshape(G, 2)
    want = np.asarray(z["counts"]).astype(np.int64)
    if got.shape != want.shape or not np.array_equal(got, want):
        return [f"positions per game and side differ from counts in {int((got != want).any(1).sum())} games"]
    return []


def read_set_chunks(chunk_dir, files, shas=None):
    """08's P1 chunks of one set in order (SHA-256 checked when given) -> (rows, arrays: ref, means float32, counts,
    n_moves, failed, ex32 [G, 44] float32 raw extra features)."""
    rows, parts = [], []
    for i, f in enumerate(files):
        p = os.path.join(chunk_dir, f)
        if not os.path.exists(p):
            refuse(f"missing chunk {shown(p)}", reason="p1_chunks")
        s = sha_raw(p)
        if shas is not None and s != shas[i]:
            refuse(f"{shown(p)} differs from the SHA-256 its report lists (08's chunks are never edited)",
                   reason="p1_chunks")
        z = load_npz(p)
        miss = [k for k in POS_KEYS if k not in z]
        if miss:
            refuse(f"{shown(p)} holds no per-position arrays {miss}: EX cannot be computed (PREREG_09 1.3 item 1)",
                   reason="p1_positions")
        G = len(z["ref"])
        pos_bad = pos_problems(z, G)
        if pos_bad:
            refuse(f"{shown(p)}: incomplete per-position arrays ({pos_bad}): EX cannot be computed (PREREG_09 1.3 "
                   f"item 1; rev 4 addendum)", reason="p1_positions")
        ex32 = FW.ex_raw32({k: np.asarray(z[k]) for k in POS_KEYS}, G)
        parts.append({"ref": np.asarray(z["ref"]).astype(np.int64), "means": np.asarray(z["means"]),
                      "counts": np.asarray(z["counts"]).astype(np.int64),
                      "n_moves": np.asarray(z["n_moves"]).astype(np.int64),
                      "failed": np.asarray(z["failed"]).astype(bool), "ex32": ex32})
        rows.append({"file": f, "sha256": s, "games": int(G), "positions": int(len(z["pos_side"]))})
    out = {k: np.concatenate([x[k] for x in parts]) for k in ("ref", "means", "counts", "n_moves", "failed", "ex32")}
    return rows, out


def set_features(d, mp):
    """lp_P1TO_E of one set's usable games (not failed, >= 1 target move) with the frozen map (numpy float64)
    -> (lp [G, 10] zeros elsewhere, p1_ok [G])."""
    us = FW.usable_games(d["failed"], d["counts"])
    lp = np.zeros((len(us), FW.L))
    if us.any():
        lp[us] = FW.map_logp(mp, FW.x_raw(d["means"][us], d["counts"][us]), d["ex32"][us])
    return lp, us


def std_ex(ex32, mp):
    return FW.standardise_ex(ex32, mp["mu_ex"], mp["sd_ex"])


def std_x(means, counts, mp):
    return FW.standardise_x(FW.x_raw(means, counts), mp["mu_x"], mp["sd_x"])


def val_seam_chunks():
    """The validation chunks 0 and 1 (= the 400 seam games) of 08's validation P1 run, SHA-256 against
    results/prereg08/feat/val_report.json -> (rows, arrays)."""
    vr = read_json(VAL_REPORT)
    chs = (vr.get("chunks") or [])[:2]
    if len(chs) != 2 or sum(int(c.get("games") or 0) for c in chs) != SEAM_GAMES:
        refuse("results/prereg08/feat/val_report.json does not list two 200-game chunks first")
    files = [c["file"] for c in chs]
    rows, d = read_set_chunks(REPO, [os.path.join(*f.split("/")) for f in files], [c["sha256"] for c in chs])
    if not np.array_equal(d["ref"], np.arange(SEAM_GAMES)):
        refuse("validation chunks 0-1 are not the validation rows 0-399")
    return rows, d


def compare_sets(a, b, mp):
    """Two P1 sets of the same games (a: test day / GPU drill, b: validation): failed / counts / n_moves equal; the
    largest |difference| of the standardised EX, the standardised X (record only) and lp_P1TO_E per game (before the
    x 4) over the games usable in both."""
    n = min(len(a["failed"]), len(b["failed"]))
    eq = {k: bool(len(a[k]) == len(b[k]) and np.array_equal(np.asarray(a[k]).astype(np.int64),
                                                            np.asarray(b[k]).astype(np.int64)))
          for k in ("failed", "counts", "n_moves")}
    counts_equal = all(eq.values())
    ok = FW.usable_games(a["failed"][:n], a["counts"][:n]) & FW.usable_games(b["failed"][:n], b["counts"][:n])
    if counts_equal and ok.any():
        ea, eb = std_ex(a["ex32"][:n][ok], mp), std_ex(b["ex32"][:n][ok], mp)
        xa, xb = std_x(a["means"][:n][ok], a["counts"][:n][ok], mp), std_x(b["means"][:n][ok], b["counts"][:n][ok], mp)
        la = FW.map_logp(mp, FW.x_raw(a["means"][:n][ok], a["counts"][:n][ok]), a["ex32"][:n][ok])
        lb = FW.map_logp(mp, FW.x_raw(b["means"][:n][ok], b["counts"][:n][ok]), b["ex32"][:n][ok])
        per_game = np.max(np.abs(la - lb), axis=1)
        dex = np.abs(ea - eb)
        rec = {"max_abs_std_ex": float(np.max(dex[:, EX_JUDGED])), "max_abs_std_ex_all_44": float(np.max(dex)),
               "frac_lowp_record_only": {"features": list(EX_RECORD_ONLY),
                                         "max_abs_std": float(np.max(dex[:, EX_RECORDED])),
                                         "games_jumped": int((dex[:, EX_RECORDED] > 0).any(1).sum())},
               "max_abs_std_x": float(np.max(np.abs(xa - xb))),
               "max_abs_lp_p1e": float(per_game.max()),
               "lp_p1e_quantiles": {q: float(np.quantile(per_game, float(q))) for q in ("0.5", "0.9", "0.99")},
               "games_lp_above_review_line": int((per_game > LP_REVIEW).sum()),
               "argmax_equal": int((la.argmax(1) == lb.argmax(1)).sum()),
               "bitwise_equal_means": bool(np.array_equal(a["means"][:n], b["means"][:n])),
               "bitwise_equal_ex": bool(np.array_equal(a["ex32"][:n], b["ex32"][:n], equal_nan=True))}
    else:
        rec = {"max_abs_std_ex": None, "max_abs_std_ex_all_44": None, "frac_lowp_record_only": None,
               "max_abs_std_x": None, "max_abs_lp_p1e": None, "bitwise_equal_means": False,
               "bitwise_equal_ex": False}
    return dict(rec, games=int(n), usable_games=int(ok.sum()), equal=eq, counts_equal=counts_equal)


def gpu_seam_record(seam, val, mp):
    """PREREG_09 5.2 GPU seam: the 400 seam games of 08 against the validation chunks 0-1 (the same games, the same
    200-game chunking)."""
    rec = compare_sets(seam, val, mp)
    probs = []
    if len(seam["failed"]) != SEAM_GAMES:
        probs.append(f"seam has {len(seam['failed'])} games, registered {SEAM_GAMES}")
    if not np.array_equal(seam["ref"], np.arange(len(seam["ref"]))):
        probs.append("seam chunk refs are not the games in order")
    if not rec["counts_equal"]:
        probs.append(f"failed / counts / n_moves differ: {rec['equal']}")
    fe, fl = rec["max_abs_std_ex"], rec["max_abs_lp_p1e"]
    passed = bool(not probs and is_num(fe) and is_num(fl) and fe <= TOL_EX and fl <= TOL_E)
    return dict(rec, problems=probs, passed=passed, tolerances={"std_ex": TOL_EX, "lp_p1e": TOL_E},
                std_ex_definition="largest |difference| of the standardised EX over the 42 features other than "
                                  "t_frac_lowp / o_frac_lowp (recorded only; PREREG_09 rev 4)",
                lp_review_line=LP_REVIEW, lp_above_review_line=bool(is_num(fl) and fl > LP_REVIEW),
                lp_definition="lp_P1TO_E = log(mean of the 6 members' probabilities) per game with the frozen "
                              "p1e_map.npz (numpy float64), before the x 4")


def probs_rows_compare(rows, ref_path):
    """Q08 recomputed (rows as t2.write_probs writes them) against 08's committed rank_test_probs.csv: byte equality
    (record), same questions / order / counts, probabilities within TOL_Q08, rule A and rule B levels equal."""
    mine = t7.probs_bytes(rows)
    ref = lf_bytes(ref_path)
    rec = {"byte_equal": mine == ref, "computed_sha256": hashlib.sha256(mine).hexdigest(),
           "t08_sha256": hashlib.sha256(ref).hexdigest(), "tol": TOL_Q08}
    rr = list(csv.reader(io.StringIO(ref.decode("utf-8"))))
    mr = list(csv.reader(io.StringIO(mine.decode("utf-8"))))
    probs, max_d, n_a, n_b = [], 0.0, 0, 0
    if rr[:1] != mr[:1] or len(rr) != len(mr):
        probs.append("header or row count differs")
    else:
        PA, PB, okq = [], [], []
        for x, y in zip(mr[1:], rr[1:]):
            if x[:3] != y[:3]:
                probs.append(f"question / counts differ at {x[0]} / {y[0]}")
                break
            if (x[3] == "") != (y[3] == ""):
                probs.append(f"empty probabilities differ at {x[0]}")
                break
            if x[3] == "":
                continue
            a_, b_ = np.array(x[3:], dtype=np.float64), np.array(y[3:], dtype=np.float64)
            max_d = max(max_d, float(np.max(np.abs(a_ - b_))))
            PA.append(a_)
            PB.append(b_)
        if not probs and PA:
            PA, PB = np.array(PA), np.array(PB)
            n_a = int((PA.argmax(1) != PB.argmax(1)).sum())
            n_b = int((rf.rule_b_np(PA) != rf.rule_b_np(PB)).sum())
    rec.update(problems=probs, max_abs_diff=max_d, rule_a_differences=n_a, rule_b_differences=n_b,
               equal=bool(not probs and max_d <= TOL_Q08 and n_a == 0 and n_b == 0))
    return rec


# ================================================================ apply
def apply_outputs(ctx):
    rank, seam = ctx["rank9"], os.path.join(ctx["rank9"], "seam_val")
    out = [os.path.join(seam, n) for n in ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json",
                                           "seam_local.json")]
    out += [os.path.join(rank, n) for n in TEST_OUTPUTS + ("t09_apply_report.json", "attempt_verdict.json")]
    out += [os.path.join(ctx["R9"], "gpu_seam.json"), os.path.join(ctx["R9"], "not_used.json")]
    return out


def check_bug_revision(n, overturned):
    """--bug-revision N: PREREG_09 rev N >= 5 committed and clean, its row at HEAD, added after every overturned
    record (ISO times). -> record."""
    if n is None:
        return None
    if n < P09_MIN_REV:
        refuse(f"--bug-revision {n}: an override needs PREREG_09 rev {P09_MIN_REV} or later")
    if not committed_clean(P09_DOC):
        refuse("docs/PREREG_09_MAPPING_ENSEMBLE.md must be committed and clean")
    row = f"| rev {n} |"
    rc, head = git("show", "HEAD:docs/PREREG_09_MAPPING_ENSEMBLE.md")
    if rc != 0 or not any(line.startswith(row) for line in head.splitlines()):
        refuse(f"docs/PREREG_09_MAPPING_ENSEMBLE.md at HEAD has no row starting {row!r}")
    rc, out = git("log", "--reverse", "-S" + row, "--format=%H %cI", "--", "docs/PREREG_09_MAPPING_ENSEMBLE.md")
    if rc != 0 or not out:
        refuse(f"no commit added {row!r}")
    h, t = out.splitlines()[0].split()
    at = parse_iso(t, "commit time")
    early = [f"{w} {x}" for w, x in overturned if x and at <= parse_iso(x, w)]
    if early:
        refuse(f"PREREG_09 rev {n} ({h[:12]}, {t}) is not later than {early}")
    return {"rev": n, "commit": h, "committed_at": t, "overturns": [{"what": w, "time": x} for w, x in overturned]}


def seam_verdict(sc, inputs_identical):
    return t7.seam_verdict(sc, inputs_identical)


class Apply:
    def __init__(self, ctx, a, t0):
        self.ctx, self.a, self.t0 = ctx, a, t0
        self.stage, self.pre, self.replaced, self.bug, self.rec = "prerequisites", {}, {}, None, {}

    # ---- rerun rules (as t08)
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
        rep = read_json_or_none(os.path.join(ctx["rank9"], "t09_apply_report.json"))
        nu = read_json_or_none(os.path.join(ctx["R9"], "not_used.json"))
        decided = [(n, r.get("created")) for n, r in (("t09_apply_report.json", rep), ("not_used.json", nu)) if r]
        if decided:
            if a.bug_revision is None:
                refuse(f"{[n for n, _ in decided]} hold the registered decision; a rerun needs a PREREG_09 revision "
                       f"naming a program error (--bug-revision N, N >= {P09_MIN_REV})")
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
            write_json(ctx, os.path.join(ctx["R9"], "not_used.json"), dict(rec, format=FORMATS["not_used"]))
        else:
            write_json(ctx, os.path.join(ctx["rank9"], "attempt_verdict.json"), rec)

    # ---- step 1: 08's three P1 sets
    def p1_sets(self, rep8):
        ctx, out = self.ctx, {}
        for s in SETS:
            npz, rp = t8.p1_outputs({"p1": ctx["p1"]}, s)
            want = ((rep8.get("p1_sets") or {}).get(s) or {})
            if pin_sha(rp) != want.get("report_sha256") or sha_raw(npz) != want.get("npz_sha256"):
                refuse(f"{s}: 08's P1 report / set differ from the SHA-256s t08_apply_report.json recorded",
                       reason="p1_sets")
            rep = read_json(rp)
            if rep.get("format") != t8.FORMATS["p1_report"] or rep.get("mode") != ctx["mode"] or rep.get("set") != s:
                refuse(f"{shown(rp)}: format {rep.get('format')!r}, mode {rep.get('mode')!r}, set {rep.get('set')!r}")
            if (rep.get("fake_from_val") or rep.get("fake_scorer") or rep.get("max_games_drill")) and ctx["real"]:
                refuse(f"{shown(rp)} is a drill record (fake or --max-games): never in the real mode")
            chunks = rep.get("chunks") or []
            if len(chunks) != -(-int(rep.get("games") or 0) // t8.CHUNK) or not rep.get("games"):
                refuse(f"{shown(rp)} lists {len(chunks)} chunks for {rep.get('games')} games")
            rows, d = read_set_chunks(os.path.join(ctx["H"], s), [c["file"] for c in chunks],
                                      [c["sha256"] for c in chunks])
            z = load_npz(npz)
            if not np.array_equal(d["ref"], np.arange(len(z["failed"]))):
                refuse(f"{s}: chunk refs are not the set's slots in order")
            for k in ("counts", "n_moves", "failed"):
                if not np.array_equal(np.asarray(z[k]).astype(d[k].dtype), d[k]):
                    refuse(f"{s}: {s}_p1.npz {k} differs from its chunks")
            d["z"] = z
            out[s] = d
            self.rec.setdefault("p1_sets", {})[s] = {
                "report_sha256": want.get("report_sha256"), "npz_sha256": want.get("npz_sha256"),
                "games": int(len(z["failed"])), "chunks": len(rows), "positions": int(sum(r["positions"] for r in rows)),
                "fake_from_val": bool(rep.get("fake_from_val")),
                "earliest_chunk_created_utc": rep.get("earliest_chunk_created_utc")}
        return out

    # ---- step 2: the 09 GPU seam
    def gpu_seam(self, p1, mp):
        ctx = self.ctx
        vrows, val = val_seam_chunks()
        rec = gpu_seam_record(p1["seam"], val, mp)
        rec.update(format=FORMATS["gpu_seam"], mode=ctx["mode"], created=now_iso(),
                   seam_p1_report_sha256=self.rec["p1_sets"]["seam"]["report_sha256"],
                   reference={"val_chunks": vrows, "val_report.json": sha_lf(VAL_REPORT), "p1e_map.npz": MAP_SHA})
        path = write_json(ctx, os.path.join(ctx["R9"], "gpu_seam.json"), rec)
        self.rec["gpu_seam"] = {k: rec[k] for k in ("passed", "counts_equal", "max_abs_std_ex", "max_abs_std_ex_all_44",
                                                    "frac_lowp_record_only", "max_abs_lp_p1e",
                                                    "max_abs_std_x", "bitwise_equal_means", "bitwise_equal_ex",
                                                    "usable_games")}
        print(f"09 GPU seam: counts equal {rec['counts_equal']}, max |d std EX| (42) {rec['max_abs_std_ex']} (<= {TOL_EX}; "
              f"frac_lowp record only {rec['frac_lowp_record_only']}), "
              f"max |d lp_P1TO_E| {rec['max_abs_lp_p1e']} (<= {TOL_E}) -> {'PASSED' if rec['passed'] else 'FAILED'}",
              flush=True)
        if not rec["passed"]:
            drift("09 GPU seam failed (PREREG_09 5.2): stop and find out why", "gpu_seam", "gpu_seam",
                  {k: rec[k] for k in ("counts_equal", "max_abs_std_ex", "max_abs_lp_p1e", "problems")})
        return path

    # ---- the run
    def go(self):
        ctx, a = self.ctx, self.a
        fc9, mp, fc07, fc08, model08, fc04, t02, rep7, rep4, rep8, self.pre = prereqs(ctx, a)
        self.rerun_gate()
        self.stage = "p1_sets"
        p1 = self.p1_sets(rep8)
        model07 = pf7.load_hs_map(path_of("results/prereg07/rank/hs_map.npz"))
        self.clear()
        self.stage = "gpu_seam"
        gpu_seam_path = self.gpu_seam(p1, mp)
        # ---- step 3: Q08 recomputed (test02 / 04 / 07 inputs, 08's frozen map) equal to 08's rank_test_probs.csv
        self.stage = "q08_vs_08"
        vk = {s: rf.load_val_keys(os.path.join(ctx["D2"], "rank", s, "dumps")) for s in Q_SETS}
        sw = {s: t8call(t7.load_sw, ctx, s, vk[s]) for s in Q_SETS}
        o4z = {s: load_npz(os.path.join(ctx["R4"], "opp", f"o4_{s}.npz")) for s in Q_SETS}
        c5w = fc04["source_sha256"]["C5"]["weights_sha256"]
        fz = {}
        for s in Q_SETS:     # every question-file read of this process happens here (seam_val: data/local)
            ddir = os.path.join(ctx["D2"], "rank", s, "dumps")
            c5 = t8call(t7.t4call, t4.load_c5, s, ddir, vk[s], c5w)
            f = t8call(t7.t4call, t4.fuse_set, s, vk[s], t02["qfiles"][s], c5, t02["opening"], o4z[s], sw[s], fc04,
                       False)
            f["c5_sha256"] = c5["sha256"]
            f.update(t7.port_c5(vk[s], c5, fc04, f))
            fz[s] = f
        if not all(fz[s]["port_equal"] for s in Q_SETS):
            drift("the ported C5 lines do not give fuse_set's Q02 bit for bit", "q08_vs_08", "port_c5",
                  {s: fz[s]["port_equal"] for s in Q_SETS})
        # ---- from here no data/local file is opened: pilot_fuse (hs_lib's guard), then p1_fuse
        PF = pf7.import_pilot_fuse()
        try:
            _L1, F = r8.import_p1()
        except FOREIGN_ERRORS as e:
            refuse(f"P1 code: {e}")
        hs = {s: load_npz(os.path.join(ctx["R7"], "hs", f"{s}_hs.npz")) for s in TEST_SETS}
        h, q8, lpE, okE = {}, {}, {}, {}
        for s in TEST_SETS:
            z = p1[s]["z"]
            keys_ok = (list(map(str, z["question_id"])) == vk[s]["question_id"]
                       and [int(x) for x in z["slot"]] == vk[s]["slot"] and list(map(str, z["color"])) == vk[s]["color"]
                       and list(map(str, hs[s]["question_id"])) == vk[s]["question_id"]
                       and [int(x) for x in hs[s]["slot"]] == vk[s]["slot"])
            if not keys_ok:
                refuse(f"{s}: 08's {s}_p1.npz / 07's {s}_hs.npz keys differ from test02's val_keys.npz", reason="keys")
            h[s] = t8call(t7.q02h_set, fz[s], vk[s], hs[s], model07, fc07, PF)
            vp = {"means": p1[s]["means"].astype(np.float64), "counts": p1[s]["counts"], "n_moves": p1[s]["n_moves"],
                  "failed": p1[s]["failed"]}
            q8[s] = t8.q08_set(fz[s], vk[s], h[s], vp, model08, fc08, F)
            lpE[s], okE[s] = set_features(p1[s], mp)
        cmp8 = probs_rows_compare(q8["public"]["rows"] + q8["private"]["rows"],
                                  os.path.join(ctx["rank"], "rank_test_probs.csv"))
        lp8 = load_npz(os.path.join(ctx["rank"], "test_lp_p1.npz"))
        p1_ok_all = np.concatenate([okE[s] for s in TEST_SETS])
        usable_all = np.concatenate([np.asarray(vk[s]["parsed"], dtype=bool) & okE[s] for s in TEST_SETS])
        lp8_all = np.concatenate([q8[s]["lp"] for s in TEST_SETS])
        cmp8.update(p1_ok_equal_t08=bool(np.array_equal(np.asarray(lp8["p1_ok"], dtype=bool), p1_ok_all)),
                    usable_equal_t08=bool(np.array_equal(np.asarray(lp8["usable"], dtype=bool), usable_all)),
                    lp_p1_max_abs_diff_vs_t08_record_only=float(np.max(np.abs(np.asarray(lp8["lp"]) - lp8_all)))
                    if len(lp8_all) else 0.0)
        self.rec["q08_vs_08"] = cmp8
        print(f"Q08 recomputed vs test08: byte equal {cmp8['byte_equal']}, max |dP| {cmp8['max_abs_diff']:.3g} "
              f"(<= {TOL_Q08}), rule A / B differences {cmp8['rule_a_differences']} / {cmp8['rule_b_differences']}, "
              f"usable equal {cmp8['p1_ok_equal_t08']} / {cmp8['usable_equal_t08']}", flush=True)
        if not (cmp8["equal"] and cmp8["p1_ok_equal_t08"] and cmp8["usable_equal_t08"]):
            drift("Q08 recomputed with test08's inputs differs from results/test08/rank/rank_test_probs.csv (or the "
                  "usable games differ; a local problem, PREREG_09 5.2)", "q08_vs_08", "q08_vs_08", cmp8)
        # ---- step 4: the local Q09 seam on the validation questions
        self.stage = "local_seam"
        vh = load_npz(path_of("results/prereg07/hs/val_hs.npz"))
        seam = self.local_seam(fz["seam_val"], vk["seam_val"], vh, model07, fc07, fc9, PF)
        # ---- steps 5, 6: P1 failures and direction
        self.stage = "p1_failures"
        res = {}
        for s in TEST_SETS:
            usable = np.asarray(vk[s]["parsed"], dtype=bool) & okE[s]
            res[s] = t8.q08_core(fz[s], vk[s], h[s], lpE[s], okE[s], usable, fc9)
            res[s]["fail_count"] = int(pf7.hs_failures(p1[s]["failed"], p1[s]["n_moves"]).sum())
            res[s]["info"]["p1_failed"] = int(np.asarray(p1[s]["failed"]).sum())
            res[s]["info"]["p1_failed_or_no_move"] = res[s]["fail_count"]
            pred_8 = rf.rule_b_np(np.where(res[s]["ok"][:, None], q8[s]["P"], 0.0))
            pred_9 = rf.rule_b_np(np.where(res[s]["ok"][:, None], res[s]["P"], 0.0))
            res[s]["info"]["rule_B_differs_from_Q08"] = int(((pred_9 != pred_8) & res[s]["ok"]).sum())
        fails = {s: {"failed_or_no_move": res[s]["fail_count"], "games": vk[s]["n"],
                     "ok": rate_ok(res[s]["fail_count"], vk[s]["n"])} for s in TEST_SETS}
        self.rec["p1_failures"] = fails
        bad = [s for s in TEST_SETS if not fails[s]["ok"]]
        if bad:
            not_used(f"P1 failures above 0.5 % on {bad} ({fails}): 09 is not used (PREREG_09 5.2)", "p1_failures",
                     "p1_failures", fails)
        self.stage = "p1_direction"
        direc = {s: pf7.direction_record(res[s]["rho"], res[s]["rho_n"], RHO_MONO, RHO_FLOOR) for s in TEST_SETS}
        direc["seam_val_record_only"] = seam["direction"]
        self.rec["p1_direction"] = direc
        bad = [s for s in TEST_SETS if not direc[s]["passed"]]
        if bad:
            not_used(f"P1TO_E direction check failed on {bad} ({ {s: direc[s]['rho'] for s in TEST_SETS} } < "
                     f"{direc['public']['threshold']}): 09 is not used (PREREG_09 5.2)", "p1_direction",
                     "p1_direction", direc)
        # ---- step 7: the outputs
        self.stage = "outputs"
        rank = ctx["rank9"]
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
        save_npz(ctx, os.path.join(rank, "test_lp_p1e.npz"), format=np.array(FORMATS["lp"]),
                 question_id=np.array(vk["public"]["question_id"] + vk["private"]["question_id"]),
                 slot=np.array(vk["public"]["slot"] + vk["private"]["slot"], dtype=np.int64),
                 color=np.array(vk["public"]["color"] + vk["private"]["color"]),
                 lp=np.concatenate([res[s]["lp"] for s in TEST_SETS]),
                 p1_ok=np.concatenate([res[s]["p1_ok"] for s in TEST_SETS]),
                 usable=np.concatenate([res[s]["usable"] for s in TEST_SETS]), scale=np.array(FW.SCALE),
                 map_sha256=np.array(MAP_SHA))
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
            drift(f"check_submission --official failed on the Q09 file: {errors[:5]}; {removed} removed",
                  "check_submission", "check_submission", {"errors": errors[:50], "removed": removed})
        # ---- step 8: the report
        preds = dict(t2.read_pred(sub))
        pub_ids = set(fz["public"]["qids"])
        diffs = {}
        for name, other in (("08", os.path.join(ctx["rank"], "submission_rank.csv")),
                            ("07", os.path.join(ctx["R7"], "rank", "submission_rank.csv")),
                            ("04", os.path.join(ctx["R4"], "rank", "submission_rank.csv")),
                            ("R20", os.path.join(ctx["O2"], "submission_rank.csv"))):
            if os.path.exists(other):
                op_ = dict(t2.read_pred(other))
                diffs[name] = {"public": sum(1 for q in pub_ids if op_.get(q) != preds.get(q)),
                               "all": sum(1 for q, v in preds.items() if op_.get(q) != v),
                               "file_sha256": sha_raw(other)}
            else:
                diffs[name] = None
        bound = self.inputs_bound(t02, gpu_seam_path, seam["path"])
        rep = {"format": FORMATS["apply"], "what": "test09 competition 2 apply (PREREG_09 5.2): frozen Q09 per set, "
                                                   "rule B, submission_rank.csv = pred_B.csv (public then private)",
               "mode": ctx["mode"], "created": now_iso(), "exit_code": 0, "usable": True, "unusable_reasons": [],
               "combo": COMBO, "method": METHOD, "theta": fc9["params"]["theta"],
               "map": {"file": "results/prereg09/rank/p1e_map.npz", "sha256": MAP_SHA,
                       "probe_used": fc9.get("probe_used")},
               "t1_verdict": (self.pre["t08_prerequisites"].get("t1") or {}).get("verdict"),
               "h8r": self.pre["t08_prerequisites"].get("h8r"),
               "test02": {"fallback_used": t02["report"].get("fallback_used"),
                          "apply_test_report_sha256": t02["report_sha256"],
                          "r20_first_upload_id": (t02["r20"] or {}).get("id"),
                          "r20_first_uploaded_at": (t02["r20"] or {}).get("uploaded_at")},
               "test08": {"t08_apply_report_sha256": pin_sha(os.path.join(ctx["rank"], "t08_apply_report.json")),
                          "exit_code": rep8.get("exit_code"), "usable": rep8.get("usable"),
                          "seam_local": rep8.get("seam_local"), "gpu_seam": rep8.get("gpu_seam"),
                          "inputs_bound_checked": len(rep8.get("inputs_bound") or {})},
               "test07": {"t07_apply_report_sha256": pin_sha(os.path.join(ctx["R7"], "rank", "t07_apply_report.json")),
                          "exit_code": rep7.get("exit_code"), "usable": rep7.get("usable")},
               "test04": {"t04_apply_report_sha256": pin_sha(os.path.join(ctx["R4"], "rank", "t04_apply_report.json")),
                          "exit_code": rep4.get("exit_code"), "usable": rep4.get("usable"),
                          "seam_local_tol": (rep4.get("seam_local") or {}).get("tol")},
               "p1_sets": self.rec["p1_sets"],
               "earliest_p1_chunk_created_utc": min(self.rec["p1_sets"][s]["earliest_chunk_created_utc"] or ""
                                                    for s in SETS),
               "gpu_seam": self.rec["gpu_seam"], "gpu_seam_tolerances": {"std_ex": TOL_EX, "lp_p1e": TOL_E},
               "q08_vs_08": self.rec["q08_vs_08"], "seam_local": seam["record"],
               "p1_failures": fails, "p1_direction": direc, "rho_mono": RHO_MONO, "rho_floor": RHO_FLOOR,
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
               "paths": {k: ctx[k] for k in ("R9", "R8", "H", "R7", "R4", "D2", "O2", "T", "C")},
               "tool": tool_record(), "environment": t4.environment(), "git_head": git_head(),
               "record_only": "level_summary, rule_b_differences (all 800), the seam_val direction and the byte "
                              "equality of Q08 are recorded; nothing is chosen from them"}
        rep["seconds"] = round(time.time() - self.t0, 1)
        write_json(ctx, os.path.join(rank, "t09_apply_report.json"), rep)
        print(f"apply Q09: public {len(fz['public']['qids'])}, private {len(fz['private']['qids'])} questions -> "
              f"{shown(sub)} (sha256 {rep['outputs_sha256']['submission_rank.csv'][:12]}); check_submission official: "
              f"OK; rule-B differences on public vs 08 / 07 / 04 / R20: {(diffs.get('08') or {}).get('public')} / "
              f"{(diffs.get('07') or {}).get('public')} / {(diffs.get('04') or {}).get('public')} / "
              f"{(diffs.get('R20') or {}).get('public')}")
        print("下一步：commit results/test09/，08 上傳後照 docs/TESTDAY_09.md 第 4 步 official_log.py prepare"
              if ctx["real"] else f"({ctx['mode']}: nothing is uploaded)")
        return 0

    def local_seam(self, fzs, vks, vh, model07, fc07, fc9, PF):
        """PREREG_09 5.2: Q09 on seam_val (test02 / test04 / 07 seam inputs + the committed val_lp_p1e.npz) against
        results/prereg09/rank/rank_val_probs.csv."""
        ctx = self.ctx
        keys_ok = (list(map(str, vh["question_id"])) == vks["question_id"] and [int(x) for x in vh["slot"]] ==
                   vks["slot"] and list(map(str, vh["color"])) == vks["color"])
        if not keys_ok:
            drift("val_hs.npz keys differ from test02's seam_val val_keys.npz", "local_seam", "seam_structure")
        hq = t8call(t7.q02h_set, fzs, vks, vh, model07, fc07, PF)
        lpz = load_npz(path_of("results/prereg09/rank/val_lp_p1e.npz"))
        if (list(map(str, lpz["question_id"])) != vks["question_id"] or [int(x) for x in lpz["slot"]] != vks["slot"]
                or list(map(str, lpz["color"])) != vks["color"]):
            drift("val_lp_p1e.npz keys differ from test02's seam_val val_keys.npz", "local_seam", "seam_structure")
        if float(lpz["scale"]) != FW.SCALE or str(lpz["map_sha256"]) != MAP_SHA or str(lpz["combo"]) != "Q09":
            refuse("val_lp_p1e.npz is not the frozen map's lp (Q09, scale 4, p1e_map.npz)")
        usable = np.asarray(vks["parsed"], dtype=bool) & np.asarray(lpz["p1_ok"], dtype=bool)
        r = t8.q08_core(fzs, vks, hq, np.asarray(lpz["lp"], dtype=np.float64), np.asarray(lpz["p1_ok"], dtype=bool),
                        usable, fc9)
        seam_dir = os.path.join(ctx["rank9"], "seam_val")
        os.makedirs(guard_write(ctx, seam_dir), exist_ok=True)
        csv_p = guard_write(ctx, os.path.join(seam_dir, "rank_val_probs.csv"))
        t2.write_probs(csv_p, r["rows"])
        for n in ("pred_A.csv", "pred_B.csv", "decide_summary.json"):
            guard_write(ctx, os.path.join(seam_dir, n))
        decide.run(csv_p, seam_dir)
        ref_p, ref_b = os.path.join(P09_RANK, "rank_val_probs.csv"), os.path.join(P09_RANK, "pred_B.csv")
        rep4 = read_json(os.path.join(ctx["R4"], "rank", "t04_apply_report.json"))
        tol = float((rep4.get("seam_local") or {}).get("tol"))
        sc = t2.seam_compare(csv_p, os.path.join(seam_dir, "pred_B.csv"), ref_p, ref_b, tol, False)
        same_a = t2._lf_bytes(os.path.join(seam_dir, "pred_A.csv")) == t2._lf_bytes(os.path.join(P09_RANK,
                                                                                                 "pred_A.csv"))
        sl8 = read_json(os.path.join(ctx["rank"], "seam_val", "seam_local.json"))
        inputs_identical = sl8.get("byte_identical_probs") is True and sl8.get("byte_identical_pred_B") is True
        verdict = seam_verdict(sc, inputs_identical)
        direction = pf7.direction_record(r["rho"], r["rho_n"], RHO_EVAL, r9.RHO_FLOOR_VAL)
        a9 = read_json(os.path.join(P09_RANK, "apply_report.json"))
        direction["equal_to_prereg09_apply"] = r["rho"] == (a9.get("direction") or {}).get("rho")
        if inputs_identical and not direction["equal_to_prereg09_apply"]:
            drift("seam_val P1TO_E direction rho differs from results/prereg09/rank/apply_report.json although 08's "
                  "seam_val is byte-identical to prereg08", "local_seam", "seam_rho", direction)
        rec = {"format": FORMATS["seam_local"], "what": "test09 local Q09 seam (PREREG_09 5.2): Q09 on the seam_val "
                                                       "inputs + val_hs.npz + val_lp_p1e.npz vs results/prereg09/rank",
               "mode": ctx["mode"], "created": now_iso(), **sc, "byte_identical_pred_A": same_a,
               "t08_seam_byte_identical": inputs_identical, "required_byte_identical": inputs_identical,
               "usable_equal_to_prereg09": bool(np.array_equal(usable, np.asarray(lpz["usable"], dtype=bool))),
               "verdict": verdict, "passed": verdict == "passed",
               "tol_source": "test04 t04_apply_report.json seam_local.tol (as 07 / 08)",
               "set": r["info"], "direction": direction,
               "reference_sha256": {n: pin_sha(os.path.join(P09_RANK, n))
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
            drift("local seam: questions / order / game counts differ from results/prereg09/rank/rank_val_probs.csv",
                  "local_seam", "seam_structure", {"problems": sc["problems"],
                                                   "count_mismatch_questions": sc["count_mismatch_questions"]})
        if verdict == "drift":
            drift("local seam: 08's seam_val equals prereg08 byte for byte but Q09 does not equal prereg09 byte for "
                  "byte (program drift)", "local_seam", "seam_bytes", self.rec["seam_local"])
        if verdict == "not_used":
            not_used("local seam outside the tolerance while 08's seam holds: 09 is not used (PREREG_09 5.2)",
                     "local_seam", "seam_tol", self.rec["seam_local"])
        return {"path": path, "record": self.rec["seam_local"], "direction": direction}

    def inputs_bound(self, t02, gpu_seam_path, seam_local_path):
        """The files official_log re-hashes (keys: the real-mode repository paths)."""
        ctx = self.ctx
        R4, R7, R8 = ctx["R4"], ctx["R7"], ctx["R8"]
        items = [("results/test02/rank/apply_test_report.json", t02["report_path"]),
                 ("results/test02/checks/rank/t1.json", os.path.join(ctx["C"], "rank", "t1.json")),
                 ("results/test04/rank/t04_apply_report.json", os.path.join(R4, "rank", "t04_apply_report.json")),
                 ("results/test07/rank/t07_apply_report.json", os.path.join(R7, "rank", "t07_apply_report.json")),
                 ("results/test08/rank/t08_apply_report.json", os.path.join(R8, "rank", "t08_apply_report.json")),
                 ("results/test08/rank/rank_test_probs.csv", os.path.join(R8, "rank", "rank_test_probs.csv")),
                 ("results/test08/rank/submission_rank.csv", os.path.join(R8, "rank", "submission_rank.csv")),
                 ("results/test08/rank/test_lp_p1.npz", os.path.join(R8, "rank", "test_lp_p1.npz")),
                 ("results/test08/rank/seam_val/seam_local.json", os.path.join(R8, "rank", "seam_val",
                                                                               "seam_local.json")),
                 ("results/test08/p1/gpu_seam.json", os.path.join(R8, "p1", "gpu_seam.json")),
                 ("results/test09/gpu_seam.json", gpu_seam_path),
                 ("results/test09/rank/seam_val/seam_local.json", seam_local_path)]
        items += [(r, path_of(r)) for r in P09_PINS]
        items += [(r, path_of(r)) for r in t8.P08_PINS]
        for s in SETS:
            items += [(f"results/test08/p1/{s}_p1.npz", os.path.join(R8, "p1", f"{s}_p1.npz")),
                      (f"results/test08/p1/p1_{s}_report.json", os.path.join(R8, "p1", f"p1_{s}_report.json"))]
        out, paths = {}, {}
        for key, p in items:
            if os.path.exists(p):
                out[key] = pin_sha(p)
                paths[key] = shown(p)
        return {"sha256": out, "paths": paths}


def cmd_apply(a):
    t0 = time.time()
    ctx = resolve(a, "apply")
    run = Apply(ctx, a, t0)
    try:
        return run.go()
    except T09Error as e:
        if e.code in (3, 4):
            run.write_verdict(e)
            if e.code == 3:
                print(f"09 不用（PREREG_09 5.2）：{e}\ncommit {shown(ctx['R9'])}；最終照 PREREG_08 5.4（不重跑）",
                      file=sys.stderr)
        raise


# ================================================================ status (read only)
def cmd_status(a):
    ctx = resolve(a, "status")
    lines = [f"test09 status ({ctx['mode']}; {ctx['R9']})"]
    gp = os.path.join(ctx["R9"], "gpu_seam.json")
    if os.path.exists(gp):
        g = read_json(gp)
        lines.append(f"09 GPU seam: passed {g.get('passed')}, std EX {g.get('max_abs_std_ex')}, lp_P1TO_E "
                     f"{g.get('max_abs_lp_p1e')} (tol {TOL_EX} / {TOL_E})")
    rp = os.path.join(ctx["rank9"], "t09_apply_report.json")
    vp = os.path.join(ctx["rank9"], "attempt_verdict.json")
    np_ = os.path.join(ctx["R9"], "not_used.json")
    if os.path.exists(rp):
        r = read_json(rp)
        lines.append(f"apply: exit {r.get('exit_code')}; submission sha256 "
                     f"{str((r.get('outputs_sha256') or {}).get('submission_rank.csv'))[:12]}; Q08 = 08 "
                     f"{(r.get('q08_vs_08') or {}).get('equal')}; local seam {(r.get('seam_local') or {}).get('verdict')}"
                     + (f", committed {committed_clean(rp)}" if ctx["real"] else ""))
    elif os.path.exists(np_):
        r = read_json(np_)
        lines.append(f"09 not used: {r.get('stage')} ({r.get('reason')})")
    elif os.path.exists(vp):
        r = read_json(vp)
        lines.append(f"apply: attempt_verdict exit {r.get('exit_code')} at {r.get('stage')} ({r.get('reason')})")
    else:
        lines.append("apply: not run")
    if ctx["R8"]:
        r8_ = read_json_or_none(os.path.join(ctx["R8"], "rank", "t08_apply_report.json"))
        lines.append("test08 apply: " + (f"exit {r8_.get('exit_code')}, usable {r8_.get('usable')}" if r8_ else
                                         "not run"))
    left = K6_09 - now()
    lines.append(f"K6-09 {K6_09.isoformat()}: " + (f"{left.days} days {left.seconds // 3600} h left"
                                                   if left.total_seconds() > 0 else "passed"))
    print("\n".join(lines))
    return 0


# ================================================================ command line
def build_parser():
    ap = argparse.ArgumentParser(prog="t09_rank.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("apply", "status"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--rehearsal", action="store_true")
        for d in ("--results-dir", "--t08-results-dir", "--p1-data-dir", "--t07-results-dir", "--t04-results-dir",
                  "--t02-results-dir", "--t02-out-dir", "--tests-dir", "--checks-dir"):
            p.add_argument(d)
        p.add_argument("--r20-uploaded-at", help="drills only: R20's first upload time")
        p.add_argument("--now", help="drills only: the current time (K6-09)")
        if name == "apply":
            p.add_argument("--rerun", action="store_true")
            p.add_argument("--reason")
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
        return {"apply": cmd_apply, "status": cmd_status}[a.cmd](a)
    except T09Error as e:
        print(f"{'REFUSED' if e.code == 2 else 'NOT USED' if e.code == 3 else 'DRIFT'} (exit {e.code}): {e}",
              file=sys.stderr)
        return e.code
    except Exception:  # noqa: BLE001 - an internal error is a refusal with its traceback, never a decision
        traceback.print_exc()
        print("ERROR (exit 2): unexpected internal error of tools/t09_rank.py; stop and check the local tool",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
