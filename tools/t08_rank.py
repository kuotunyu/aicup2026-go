"""C10 test08, competition 2 (rank, TEAM_11248): the 08 test-day local tool (Q08 = Q02H + 4 x P1TO).

Registration: docs/PREREG_08_STRENGTH_PROBE.md section 5 (rev 1), section 3 (the H8R adoption line) and the test-day
revision (rev 3, section 13). The 07 test-day tool tools/t07_rank.py is imported, never edited (its functions give the
Q02 / Q02H inputs, the seam_val path and the prerequisites of test02 / test04).

    GPY tools/t08_rank.py p1 --set seam|public|private [MODE] [DIRS] [--workers W] [--stop-after-min M]
                                                       [--rerun --reason TEXT]          (RTX 4090, .venv-gpu)
    PY  tools/t08_rank.py apply  [MODE] [DIRS] [--rerun --reason TEXT [--bug-revision N]]   (CPU, .venv)
    PY  tools/t08_rank.py status [MODE] [DIRS]                                              (read only)
    (PY = .venv/Scripts/python.exe, GPY = .venv-gpu/Scripts/python.exe; Git Bash; PYTHONDONTWRITEBYTECODE=1; no
    OMP / OPENBLAS / MKL thread variables)

MODE  none = real; at most one of --rehearsal, --dry-run
DIRS  --results-dir R8      test08 root (writes R8/p1, R8/rank; real results/test08)
      --p1-data-dir H       the P1 chunks (real data/test08/p1; --rehearsal data/test08_rehearsal/p1; --dry-run R8/p1_data)
      --t07-results-dir R7  test07 root (reads R7/hs, R7/rank; real results/test07)
      --t04-results-dir R4, --t02-results-dir D2, --t02-out-dir O2, --tests-dir T, --checks-dir C   as t07_rank.py
Real mode: every directory is the repository path above. --rehearsal: R8 is results/test08_rehearsal (or outside
the repository); --dry-run: R8 outside the repository. The other trees are required outside the real mode.
Drills only (never real): --r20-uploaded-at ISO, --max-games N, --fake-scorer (a deterministic CPU stand-in for the
network), --ignore-other-gpu-jobs, --now ISO.

p1     The neutral-metadata P1 features of one set on the local RTX 4090 (PREREG_08 1.3, 5.2): per game and side the
       mean of the 770 numbers (trunk at the move, trunk mean, log p, entropy). seam = the first 400 games of the
       validation questions in val_keys order (data/local/rank_val_questions.csv: question id, slot, colour and SGF
       only); public / private = test02's packed test files, slot by slot equal to test02's val_keys.npz of the set
       and to test07's <set>_hs.npz keys. Prerequisites: test07's apply report of this mode exit 0 (committed in the
       real mode), H8R adopted, K6-08 not passed, no other compute process, .venv-gpu = frozen_config.json, the full
       checkpoint SHA-256. The question file is read before p1_lib is imported (its guard closes data/local for the
       rest of the process). Chunks of 200 games in H/<set>/ are written atomically and resumed (STOP file in
       H/<set>/ or --stop-after-min end the run after the chunk in progress, exit 5); a chunk is reused only when
       its refs, key file, question file, P1 code, checkpoint and run settings (precision / batch / channels_last /
       device) are this run's (else 2); a chunk of another environment is
       refused (4), a chunk older than R20's first upload is refused (2: rename the folder H/<set>_stale_<time>/).
       When every chunk is there: R8/p1/<set>_p1.npz (keys, counts, n_moves, failed, the log p / entropy means;
       seam: also the full means) and R8/p1/p1_<set>_report.json (every chunk's SHA-256). Commit them.
apply  Q08 on public and private (PREREG_08 5.2), in this order (the first failing item decides): the three P1 sets
       (committed, chunk SHA-256s, one environment, every chunk later than R20's first upload); the GPU seam against
       results/prereg08/feat/val_seam_feats.npz (failed / counts / n_moves equal, standardised features <= 0.05,
       lp_P1TO per game <= 0.04, PREREG_08 rev 4); Q02H recomputed with tools/t07_rank.py equal byte for byte to test07's
       rank_test_probs.csv and every input test07 bound unchanged; the move cross-check against test07's
       <set>_hs.npz and test04's o4_<set>.npz; the local Q08 seam on the 1,589 validation questions (07's seam_val
       inputs + the committed val_lp_p1.npz) against results/prereg08/rank/rank_val_probs.csv; P1 failures
       (200 x failed <= games per set); the direction check (rho >= max(0.5 x rho_mono, 0.155) per set). Then
       rank_test_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json, submission_rank.csv (= pred_B.csv),
       test_lp_p1.npz, check_submission --official and R8/rank/t08_apply_report.json.
status Read-only summary.

Exit codes (as tools/t07_rank.py): 0 done; 2 refused or error; 3 the registered "08 is not used" (R8/not_used.json;
commit; the final choice follows PREREG_07 5.4; never rerun); 4 local drift or mismatch (R8/rank/attempt_verdict.json;
commit, stop and find out why, then --rerun --reason; a move cross-check mismatch needs a committed PREREG_08 revision
naming the program error first). After exit 3 or a written t08_apply_report.json nothing is rerun except with
--bug-revision N (a committed PREREG_08 rev N >= 4 added after that record). p1 only: 5 stopped on request (STOP
file or --stop-after-min) before the set was complete; nothing is decided, the finished chunks stay; delete STOP and
run the same line again to resume (rev 3 section 13 addendum: a non-zero code so that a chained launcher stops and
the GPU is freed).

Discipline: never imports u1_lib, never fits a mapping; imports p1_lib / p1_fuse / pilot_fuse / hs_lib only after
every data/local read of the process; no validation label file is ever opened (any path whose lower-case text holds
that word is refused); tools/score.py is never run; temporary files only under data/test08/tmp (real),
data/test08_rehearsal/tmp or R8/tmp. Competition 1 is never read (PREREG_08 6).
"""

import argparse
import datetime as dt
import glob
import hashlib
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
SELF = os.path.join(HERE, "t08_rank.py")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t07_rank as t7  # noqa: E402
import p08_rank_fuse as r8  # noqa: E402

t4, pf7 = t7.t4, t7.pf7
rf, p3, po, t2 = t7.rf, t7.p3, t7.po, t7.t2
decide, check_submission = t7.decide, t7.check_submission

# ================================================================ paths (module constants; the unit tests patch them)
RESULTS = os.path.join(REPO, "results")
REAL_R8 = os.path.join(RESULTS, "test08")
REHEARSAL_R8 = os.path.join(RESULTS, "test08_rehearsal")
REAL_H = os.path.join(REPO, "data", "test08", "p1")
REHEARSAL_H = os.path.join(REPO, "data", "test08_rehearsal", "p1")
REAL_TMP = os.path.join(REPO, "data", "test08", "tmp")
REHEARSAL_TMP = os.path.join(REPO, "data", "test08_rehearsal", "tmp")
REAL_R7 = t7.REAL_R7
REAL_R4 = t7.REAL_R4
REAL_D2, REAL_O2, REAL_T, REAL_C = t7.REAL_D2, t7.REAL_O2, t7.REAL_T, t7.REAL_C
SEAM_QUESTIONS = t7.SEAM_QUESTIONS
P08_RANK = os.path.join(RESULTS, "prereg08", "rank")
P08_FEAT = os.path.join(RESULTS, "prereg08", "feat")
P08_DOC = os.path.join(REPO, "docs", "PREREG_08_STRENGTH_PROBE.md")

# ================================================================ registered values
# the committed prereg08 files the test day reads (SHA-256: .json / .csv with CRLF folded to LF, .npz raw bytes)
P08_PINS = {
    "results/prereg08/rank/frozen_config.json": "421bbeb4524bed8ccf73babd58cac6b16c82f45b1aa57acef2e55bd52be0ba70",
    "results/prereg08/rank/p1_map.npz": "116923629b66198a79da06282aac9ca863acf77bf6a1e1f5d584b526fb54afb2",
    "results/prereg08/rank/rank_val_probs.csv": "7a513367147227e3631bfb7bad36064f3e90fffb6ee358716e389698ce1ccd97",
    "results/prereg08/rank/pred_B.csv": "b02d4f947910726fa25c292e9798e7407b1e802e2fed080d843f56b9beb48872",
    "results/prereg08/rank/pred_A.csv": "e24699e989543bfbfc73b1b710677c93f018747193a9ba18d89fc004cdd17753",
    "results/prereg08/rank/apply_report.json": "63980d79866d2653a4e600cef78cadd9d0d27c828a69e1b34d6ae2309eaf040f",
    "results/prereg08/rank/score_H8R.json": "8ecd2425fe3eef784e28d7587184aec67c0b41b49503e95b42f8532757209047",
    "results/prereg08/rank/val_lp_p1.npz": "8815db14654eefdb8c2a11cfbb709de4254272de734cec415b3dfc2dbb0e59a9",
    "results/prereg08/feat/val_seam_feats.npz": "0a7b9b0010192a4c7f3878e914f9e514a18cfa338d94dc8c49936b6b625b9318",
    "results/prereg08/feat/val_p1.npz": "11b1d38cc116401d44825883b864bf635b2d44e109baa17115fb4093055da4e0",
    "results/prereg08/feat/val_report.json": "2ce9e86f8e639217c38814541f7d1bfb083b609ac194b5e19c8c38c60e653535"}
# PREREG_08 3: H8R adopted = the 95 % lower bound of Q08 - Q02H > 0 on 1,589 questions, the reference (A = Q02H)
# re-read equal to H7R's B row within 1e-12; Q08 >= 0.57 (the sanity line of section 3)
H8R_A_RANK_BALANCED = 0.632702198843505
H8R_A_MEAN = 0.6345453305055525
H8R_QUESTIONS = 1589
H8R_FLOOR = 0.57
H8R_TOL = 1e-12
RHO_MONO = 0.9673017067776464                     # frozen_config.json p1_direction_ref.rho_mono
RHO_FLOOR = r8.RHO_FLOOR_TEST                     # 0.155: one-sided p = 0.001 for 400 questions
SEAM_GAMES = 400
TOL_FEAT = 0.05                                   # PREREG_08 5.2 GPU seam: standardised features
TOL_LP = 0.04                                     # lp_P1TO per game (before the x 4)
LP_REVIEW = 0.005                                 # as 07: a measured lp difference above this is reviewed
GPU_NAME = t7.GPU_NAME
TW = t7.TW
K6_08 = dt.datetime(2026, 11, 19, 23, 59, 59, tzinfo=TW)
P08_MIN_REV = 4                                   # rev 3 registers the test day; an override is rev 4 or later
EXIT_STOPPED = 5                                  # p1 stopped on request (rev 3 section 13 addendum)
SETS = ("seam", "public", "private")
TEST_SETS = ("public", "private")
Q_SETS = t7.Q_SETS
COMBO = {"id": "Q08", "sources": ["C5", "OP", "O4", "SWA", "SWB", "HS", "P1TO"],
         "terms": ["OP", "O4", "SWA", "SWB", "HS", "P1TO"], "p1_scale": 4.0}
METHOD = "prereg08 Q08 = Q02H + 4 x P1TO, rule B (甲)"
CHUNK = r8.RUN["chunk"]
F_DIM = r8.F_DIM
FORMATS = {"chunk": "c10-test08 p1 chunk v1", "p1": "c10-test08 p1 set v1", "p1_report": "c10-test08 p1 report v1",
           "gpu_seam": "c10-test08 gpu seam v1", "seam_local": "c10-test08 seam_local v1",
           "apply": "c10-test08 rank apply v1", "verdict": "c10-test08 attempt verdict v1",
           "not_used": "c10-test08 not used v1", "lp": "c10-test08 test_lp_p1 v1"}
TEST_OUTPUTS = ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv",
                "test_lp_p1.npz")
_LABEL = "ans" + "wer"


class T08Error(RuntimeError):
    """A stop with its exit code: 2 refused, 3 registered 'no 08', 4 local drift / mismatch."""

    def __init__(self, message, code=2, stage=None, reason=None, details=None):
        super().__init__(message)
        self.code, self.stage, self.reason, self.details = code, stage, reason, details


def refuse(msg, reason=None):
    raise T08Error(msg, 2, reason=reason)


def not_used(msg, stage, reason, details=None):
    raise T08Error(msg, 3, stage, reason, details)


def drift(msg, stage, reason, details=None):
    raise T08Error(msg, 4, stage, reason, details)


FOREIGN_ERRORS = tuple(dict.fromkeys(t7.FOREIGN_ERRORS + r8.ERRORS))


def t7call(fn, *args):
    """A tools/t07_rank.py (or, through t7.t4call, tools/t04_rank.py) function: its exit code is ours (2 refused, 3
    registered 'no 07 / no 04' = no 08, 4 drift)."""
    try:
        return fn(*args)
    except t7.T07Error as e:
        raise T08Error(f"(t07_rank) {e}", e.code, e.stage, e.reason or "t07_prerequisite", e.details)
    except FOREIGN_ERRORS as e:
        refuse(f"(t07_rank) {e}")


# ================================================================ small helpers
def now():
    """The current time (a module function: the tests replace it to cross K6-08)."""
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
        refuse(f"refusing {path}: validation label files are never opened by the test08 tools")
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
    names = ["t08_rank.py", "t07_rank.py", "t04_rank.py", "p08_rank_fuse.py", "p08_p1.py", "p07_rank_fuse.py",
             "p07_hs.py", "check_submission.py", "official_log.py", "t02_rank.py", "t02_test_bundle.py", "p04_opp.py",
             "p04_rank_fuse.py", "p03_rank_fuse.py", "p02_rank_fuse.py", "c10data.py", "decide.py"]
    return {n: (sha_lf(os.path.join(HERE, n)) if os.path.exists(os.path.join(HERE, n)) else None) for n in names}


def tool_record():
    return {"git_head": git_head(), "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": tool_files()}


def rate_ok(failures, games):
    """failures / games <= 0.005, in integers (200 * failures <= games): 25 of 5,000 passes, 26 does not."""
    return t7.rate_ok(failures, games)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def blas_env_check():
    """The byte seams (Q02H = 07, seam_val = prereg08) need OpenBLAS's default thread count (PREREG_08 rev 2 12.2)."""
    set_ = {k: os.environ[k] for k in r8.BLAS_ENV if os.environ.get(k)}
    if set_:
        refuse(f"BLAS thread variables are set ({set_}): unset them (the registered byte seams use the default "
               f"OpenBLAS threads; PREREG_08 rev 2 section 12 item 2)")
    return {k: None for k in r8.BLAS_ENV}


# ================================================================ modes and directories
def resolve(a, cmd):
    flags = [m for m, on in (("dry-run", a.dry_run), ("rehearsal", a.rehearsal)) if on]
    if len(flags) > 1:
        refuse(f"give at most one of --dry-run, --rehearsal (got {flags})")
    mode = flags[0] if flags else "real"
    given = {"--results-dir": a.results_dir, "--t07-results-dir": a.t07_results_dir,
             "--t04-results-dir": a.t04_results_dir, "--t02-results-dir": a.t02_results_dir,
             "--t02-out-dir": a.t02_out_dir, "--tests-dir": a.tests_dir}
    real = {"--results-dir": REAL_R8, "--t07-results-dir": REAL_R7, "--t04-results-dir": REAL_R4,
            "--t02-results-dir": REAL_D2, "--t02-out-dir": REAL_O2, "--tests-dir": REAL_T}
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
        for k, v, want in (("--checks-dir", a.checks_dir, REAL_C), ("--p1-data-dir", a.p1_data_dir, REAL_H)):
            if v and not t2.same_path(v, want):
                refuse(f"real mode: {k} must be {want} (got {v})")
        R8, H, R7, R4, D2, O2, T, C, TMP = (REAL_R8, REAL_H, REAL_R7, REAL_R4, REAL_D2, REAL_O2, REAL_T, REAL_C,
                                            REAL_TMP)
    else:
        need = ["--results-dir"] if cmd == "status" else list(given)
        if mode == "rehearsal" and not a.results_dir:
            given["--results-dir"] = REHEARSAL_R8
        missing = [k for k in need if not given[k]]
        if missing:
            refuse(f"--{mode} needs {', '.join(missing)} (other modes never fall back to the repository paths)")
        R8 = os.path.abspath(given["--results-dir"])
        ab = lambda v: os.path.abspath(v) if v else None   # noqa: E731
        R7, R4, D2, O2, T = (ab(a.t07_results_dir), ab(a.t04_results_dir), ab(a.t02_results_dir), ab(a.t02_out_dir),
                             ab(a.tests_dir))
        C = os.path.abspath(a.checks_dir) if a.checks_dir else (os.path.join(D2, "checks") if D2 else None)
        in_repo_ok = mode == "rehearsal" and t2.same_path(R8, REHEARSAL_R8)
        if t2.is_inside(R8, REPO) and not in_repo_ok:
            refuse(f"--{mode}: --results-dir must be outside the repository (only --rehearsal may use "
                   f"{REHEARSAL_R8}); got {R8}")
        if a.p1_data_dir:
            H = os.path.abspath(a.p1_data_dir)
        else:
            H = REHEARSAL_H if in_repo_ok else os.path.join(R8, "p1_data")
        if t2.is_inside(H, REPO) and not t2.is_inside(H, os.path.dirname(REHEARSAL_H)):
            refuse(f"--{mode}: --p1-data-dir inside the repository must be under data/test08_rehearsal (got {H})")
        TMP = REHEARSAL_TMP if in_repo_ok else os.path.join(R8, "tmp")
    for p in (R8, H, R7, R4, D2, O2, T, C):
        if p is not None:
            refuse_label_path(p)
            if pf7.under_data_local(p):
                refuse(f"{p}: never under data/local")
    if T is not None and os.path.basename(os.path.normpath(T)).lower() == "raw":
        refuse(f"--tests-dir {T}: the test files are the checked copies (data/test), never data/test/raw")
    bad = forbidden_area(R8) or forbidden_area(H)
    if bad:
        refuse(f"output folders inside the read-only {bad}")
    return {"mode": mode, "real": mode == "real", "smoke": False, "cmd": cmd, "R8": R8, "H": H, "R7": R7, "R4": R4,
            "D2": D2, "O2": O2, "T": T, "C": C, "TMP": TMP, "p1": os.path.join(R8, "p1"),
            "rank": os.path.join(R8, "rank"), "write_roots": [R8, H, TMP]}


def forbidden_area(path):
    """Where t08 never writes: results/prereg0*, results/test02*, test04*, test06*, test07*, data/local, the ledger."""
    p = os.path.abspath(path)
    if pf7.under_data_local(p):
        return "data/local"
    if t2.same_path(p, os.path.join(RESULTS, "score_ledger.csv")):
        return "results/score_ledger.csv"
    if t2.is_inside(p, RESULTS) and not t2.same_path(p, RESULTS):
        first = os.path.relpath(os.path.realpath(p), os.path.realpath(RESULTS)).split(os.sep)[0].lower()
        if first.startswith(("prereg0", "test02", "test04", "test06", "test07")):
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


def save_npz(ctx, path, **arrays):
    guard_write(ctx, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


# ================================================================ common prerequisites
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
    own = [SELF] + [os.path.join(HERE, n) for n in ("t07_rank.py", "t04_rank.py", "p07_rank_fuse.py",
                                                    "p08_rank_fuse.py", "check_submission.py", "official_log.py")]
    rec["git"] = {"clean": True, "head": git_head(), "own_files": require_committed(own, "the test08 tools")}


def check_p08_pins():
    """The committed prereg08 files equal their pinned SHA-256s; the pinned P1 / pilot / u1 code unchanged; the
    frozen configuration is the adopted Q08. -> (frozen config, map, record)."""
    got = {r: (pin_sha(path_of(r)) if os.path.exists(path_of(r)) else None) for r in P08_PINS}
    bad = sorted(r for r in P08_PINS if got[r] != P08_PINS[r])
    if bad:
        refuse(f"prereg08 files differ from the pinned SHA-256s: {bad} (never edited after H8R)")
    try:
        pins = r8.check_pins()
        p07t = r8.check_p07_pins()
    except FOREIGN_ERRORS as e:
        refuse(f"pinned code: {e}")
    fc = read_json(path_of("results/prereg08/rank/frozen_config.json"))
    combo = fc.get("combo") or {}
    if combo.get("id") != COMBO["id"] or combo.get("terms") != COMBO["terms"] \
            or float(combo.get("p1_scale", 0)) != r8.SCALE or fc.get("score_pending_val") is not True \
            or fc.get("preview") is not False or fc.get("precommit") is not False:
        refuse("results/prereg08/rank/frozen_config.json is not the frozen Q08 configuration")
    if float((fc.get("p1_direction_ref") or {}).get("rho_mono")) != RHO_MONO:
        refuse("frozen_config.json p1_direction_ref.rho_mono differs from the registered value")
    if fc["p1"]["p1_map_npz_sha256"] != P08_PINS["results/prereg08/rank/p1_map.npz"]:
        refuse("frozen_config.json names another p1_map.npz")
    try:
        model = r8.load_p1_map(path_of("results/prereg08/rank/p1_map.npz"))
    except FOREIGN_ERRORS as e:
        refuse(str(e))
    if model["scale"] != r8.SCALE:
        refuse("p1_map.npz scale is not 4")
    return fc, model, {"prereg08_sha256": got, "pinned_code": pins, "p07_pins": p07t}


def h8r_condition():
    """PREREG_08 3: H8R adopted (lower bound > 0 on 1,589 questions, A re-read equal to H7R's B row within 1e-12,
    Q08 >= 0.57). -> record; exit 3 otherwise (08 is not used)."""
    p = path_of("results/prereg08/rank/score_H8R.json")
    if not os.path.exists(p):
        not_used("PREREG_08 3: H8R has not been scored: 08 is not used", "prerequisites", "h8r_missing")
    s = read_json(p)
    a_rb, a_m, b_rb = s.get("A_rank_balanced"), s.get("A_mean"), s.get("B_rank_balanced")
    d = s.get("B_minus_A_rank_balanced") or {}
    n, ci = d.get("n"), d.get("ci95") or [None, None]
    rec = {"A_rank_balanced": a_rb, "A_mean": a_m, "B_rank_balanced": b_rb, "questions": n, "ci95": ci,
           "rule": "PREREG_08 3: lower bound of Q08 - Q02H > 0, 1,589 questions, reference re-read equal to H7R's "
                   "B row (|A - H7R| <= 1e-12), Q08 >= 0.57"}
    ok = (is_num(a_rb) and is_num(a_m) and is_num(b_rb) and n == H8R_QUESTIONS and is_num(ci[0]) and ci[0] > 0
          and abs(a_rb - H8R_A_RANK_BALANCED) <= H8R_TOL and abs(a_m - H8R_A_MEAN) <= H8R_TOL and b_rb >= H8R_FLOOR)
    rec["holds"] = bool(ok)
    if not ok:
        not_used(f"PREREG_08 3: H8R does not hold ({rec}): 08 is not used", "prerequisites", "h8r", rec)
    return rec


def earlier_success(ctx):
    """A committed t08_apply_report.json with exit_code 0, committed by K6-08 (real mode) -> (bool, record)."""
    p = os.path.join(ctx["rank"], "t08_apply_report.json")
    if not os.path.exists(p):
        return False, None
    rep = read_json(p)
    if rep.get("exit_code") != 0:
        return False, None
    if not ctx["real"]:
        return True, {"note": f"{ctx['mode']}: commit time not checked"}
    h, t = commit_time(p)
    ok = t is not None and parse_iso(t, "commit time") <= K6_08
    return ok, {"commit": h, "committed_at": t}


def check_k6(ctx, a):
    """K6-08 not passed; after it, exit 3 without an earlier success, 2 with one (that one stays valid)."""
    t_now = parse_iso(a.now, "--now") if getattr(a, "now", None) else now()
    if t_now <= K6_08:
        return {"now": t_now.isoformat(), "k6_08": K6_08.isoformat(), "passed": False}
    ok, rec = earlier_success(ctx)
    if ok:
        refuse(f"K6-08 {K6_08.isoformat()} has passed; the successful run before it ({rec}) stays valid: nothing is "
               f"rerun", reason="k6_passed_success")
    not_used(f"K6-08 {K6_08.isoformat()} has passed without a successful official 08 run: 08 is not used this "
             f"season (PREREG_08 4); the final choice follows PREREG_07 5.4", "prerequisites", "k6_08")


def t07_state(ctx):
    """test07's apply of this mode: exit 0 and usable (committed in the real mode). 07 judged 'not used' -> 3 (08 is
    not used either, PREREG_08 5.2); not done yet -> 2. -> (report, record)."""
    rank7 = os.path.join(ctx["R7"], "rank")
    rp, vp = os.path.join(rank7, "t07_apply_report.json"), os.path.join(rank7, "attempt_verdict.json")
    nu = read_json_or_none(os.path.join(ctx["R7"], "not_used.json"))
    if nu is not None and nu.get("exit_code") == 3 and nu.get("mode") == ctx["mode"]:
        not_used(f"test07 judged '07 is not used' ({nu.get('stage')}: {nu.get('reason')}): 08 is not used either "
                 f"(PREREG_08 5.2)", "prerequisites", "t07_not_used")
    if not os.path.exists(rp):
        refuse("test07 apply has not finished (no t07_apply_report.json"
               + (f"; attempt_verdict.json says exit {read_json(vp).get('exit_code')}" if os.path.exists(vp) else "")
               + "): run docs/TESTDAY_07.md step 4 first", reason="t07_not_done")
    rep = read_json(rp)
    if rep.get("format") != t7.FORMATS["apply"] or rep.get("mode") != ctx["mode"]:
        refuse(f"{shown(rp)}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (this run: {ctx['mode']})")
    if rep.get("exit_code") == 3 or rep.get("usable") is False:
        not_used("test07 apply judged '07 is not used': 08 is not used either", "prerequisites", "t07_not_used")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True or (rep.get("combo") or {}).get("id") != "Q02H":
        refuse(f"{shown(rp)}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    if ctx["real"]:
        require_committed([rp, os.path.join(rank7, "rank_test_probs.csv"), os.path.join(rank7, "submission_rank.csv"),
                           os.path.join(rank7, "seam_val", "seam_local.json"),
                           os.path.join(ctx["R7"], "hs", "gpu_seam.json")]
                          + [os.path.join(ctx["R7"], "hs", f"{s}_hs.npz") for s in SETS], "results/test07")
    return rep, {"path": shown(rp), "sha256": pin_sha(rp), "exit_code": 0, "usable": True,
                 "submission_sha256": (rep.get("outputs_sha256") or {}).get("submission_rank.csv")}


def r20_upload(ctx, t02rep, a):
    """R20's first upload logged as uploaded (real); drills take --r20-uploaded-at. -> record."""
    if ctx["real"]:
        return t7call(t7.t4call, t4.r20_first_upload, t7.ctx4(ctx), t02rep)
    if getattr(a, "r20_uploaded_at", None):
        return {"id": None, "uploaded_at": parse_iso(a.r20_uploaded_at, "--r20-uploaded-at").isoformat(),
                "note": f"{ctx['mode']}: --r20-uploaded-at"}
    return {"id": None, "uploaded_at": None, "note": f"not checked ({ctx['mode']})"}


def prereqs(ctx, cmd, a):
    """PREREG_08 5.2 in this order. -> (fc07, fc08, model08, fc04, t02, t07 report, t04 report or None, record)."""
    rec = {}
    check_git_clean(ctx, rec)                                            # 1
    if cmd == "apply":                                                   # 2 (.venv; the GPU check is in p1)
        rec["venv"] = t7call(t7.t4call, t4.check_venv)
        rec["blas_env"] = blas_env_check()
    fc07, rec["prereg07"] = t7call(t7.check_p07_pins)                    # 3
    fc08, model08, rec["prereg08"] = check_p08_pins()
    fc04, rec["prereg04_frozen"] = t7call(t7.t4call, t4.load_frozen_config, ctx["real"])
    rec["h7r"] = t7call(t7.h7r_condition)                                # 4
    rec["h8r"] = h8r_condition()
    rec["k6_08"] = check_k6(ctx, a)                                      # 5
    rec["t1"] = t7call(t7.t4call, t4.t1_first, t7.ctx4(ctx))             # 6
    t02 = t7call(t7.t4call, t4.check_test02, t7.ctx4(ctx))               # 7
    rec["test02"] = {"report_sha256": t02["report_sha256"], "fallback_used": t02["report"].get("fallback_used")}
    t02["r20"] = r20_upload(ctx, t02["report"], a)                       # 8
    rec["r20_first_upload"] = t02["r20"]
    rep7, rec["t07_apply_report"] = t07_state(ctx)                       # 9 (p1 and apply)
    rep4 = None
    if cmd == "apply":                                                   # 10
        rep4, rec["t04_apply_report"] = t7call(t7.t04_state, ctx)
    return fc07, fc08, model08, fc04, t02, rep7, rep4, rec


# ================================================================ p1 (GPU)
def gpu_env(fake):
    import p08_p1 as e8
    return e8.gpu_env(fake=fake)


def check_gpu_env(fc08, env, fake):
    """.venv-gpu: python, torch and cuDNN equal frozen_config's reference (else 4); the GPU model (else 2)."""
    ref = fc08["environment"]["venv_gpu_reference"]
    if fake:
        return {"fake_scorer": True, "reference": ref}
    bad = {k: {"now": env.get(k), "frozen": ref.get(k)} for k in ("python", "torch", "cudnn")
           if str(env.get(k)) != str(ref.get(k))}
    if bad:
        drift(f".venv-gpu differs from frozen_config.json {bad}: restore .venv-gpu (never upgraded before 11/24)",
              "prerequisites", "venv_gpu", bad)
    if env.get("gpu") != GPU_NAME:
        refuse(f"GPU {env.get('gpu')!r} is not {GPU_NAME} (PREREG_08 5.3: P1 runs on the local RTX 4090 only)")
    return {"equal": True, "reference": ref, "driver_record_only": env.get("driver")}


def hs_keys_match(ctx, s, keys):
    """The P1 slots of a whole set (before any drill --max-games cut) equal test07's <set>_hs.npz keys row for row,
    with the same number of rows (seam: 400 = 07's seam)."""
    p = os.path.join(ctx["R7"], "hs", f"{s}_hs.npz")
    z = load_npz(p)
    ok = (len(z["question_id"]) == len(keys["question_id"])
          and list(map(str, z["question_id"])) == list(map(str, keys["question_id"]))
          and [int(x) for x in z["slot"]] == [int(x) for x in keys["slot"]]
          and list(map(str, z["color"])) == list(map(str, keys["color"])))
    if not ok:
        refuse(f"{s}: the slots differ from test07's {shown(p)} (PREREG_08 5.2: the same keys as test02 / 07)",
               reason="keys")
    return {"t07_hs_file": shown(p), "t07_hs_sha256": sha_raw(p), "equal": True}


def chunk_paths(d):
    return t7.chunk_paths(d)


def stale_chunks(d, r20_at):
    return t7.stale_chunks(d, r20_at) if r20_at else []


def run_settings(fake):
    """The scorer settings a chunk records (fake scorer: its CPU stand-in values)."""
    if fake:
        return {"precision": "fake-fp32", "batch": int(r8.RUN["batch"]), "channels_last": bool(r8.RUN["channels_last"]),
                "device": "cpu"}
    return {k: (int(r8.RUN[k]) if k == "batch" else bool(r8.RUN[k]) if k == "channels_last" else str(r8.RUN[k]))
            for k in ("precision", "batch", "channels_last", "device")}


def chunk_run_meta(z):
    return {"precision": str(z["precision"]), "batch": int(z["batch"]), "channels_last": bool(z["channels_last"]),
            "device": str(z["device"])}


def chunk_check(path, refs, keys_sha, env, mode, real=False, *, questions_sha, code_sha, fake):
    """An existing chunk is reused only when it is this run's (format, sides, refs, key file, question file, P1 code,
    neutral metadata, mode, scorer settings; else 2) and of this environment (torch / cuDNN / GPU; else 4)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            got = {"format": str(z["format"]), "sides": [str(x) for x in z["sides"]],
                   "keys_sha256": str(z["keys_sha256"]), "questions_sha256": str(z["questions_sha256"]),
                   "code_sha256": str(z["code_sha256"]), "meta_sha256": str(z["meta_sha256"]), "mode": str(z["mode"]),
                   "torch": str(z["torch"]), "cudnn": str(z["cudnn"]), "gpu_name": str(z["gpu_name"]),
                   "ckpt_sha256": str(z["ckpt_sha256"]), "means_shape": list(z["means"].shape), **chunk_run_meta(z)}
            same_refs = np.array_equal(np.asarray(z["ref"]), refs)
    except Exception as e:  # noqa: BLE001
        refuse(f"{path}: unreadable chunk ({e}); move it away and rerun")
    want = {"format": FORMATS["chunk"], "sides": r8.RUN["sides"], "keys_sha256": keys_sha,
            "questions_sha256": questions_sha, "code_sha256": code_sha,
            "meta_sha256": r8.NEUTRAL_META_SHA, "mode": mode, "means_shape": [len(refs), 2, F_DIM], **run_settings(fake)}
    diff = [k for k in want if got[k] != want[k]] + ([] if same_refs else ["ref"])
    if diff:
        refuse(f"{path}: an existing chunk differs from this run ({diff}); never overwritten")
    envd = {k: (got[k], str(env[e])) for k, e in (("torch", "torch"), ("cudnn", "cudnn"), ("gpu_name", "gpu"))
            if got[k] != str(env[e])}
    if envd:
        drift(f"{path}: written with another environment {envd}; stop and find out why", "p1", "chunk_environment",
              envd)
    if real and (got["ckpt_sha256"] != r8.CKPT_SHA or got["torch"] == "fake"):
        refuse(f"{path}: a drill chunk (fake scorer) is never reused in the real mode")
    if got["ckpt_sha256"] != ("fake" if fake else r8.CKPT_SHA):
        refuse(f"{path}: chunk checkpoint {got['ckpt_sha256'][:12]} is not this run's "
               f"({'fake scorer' if fake else 'the full checkpoint SHA-256'})")


def p1_outputs(ctx, s):
    return [os.path.join(ctx["p1"], f"{s}_p1.npz"), os.path.join(ctx["p1"], f"p1_{s}_report.json")]


def cmd_p1(a):
    ctx = resolve(a, "p1")
    try:
        return p1_run(a, ctx)
    except T08Error as e:
        if e.code == 3:              # the registered 'no 08' is written down and committed
            write_json(ctx, os.path.join(ctx["R8"], "not_used.json"),
                       {"format": FORMATS["not_used"], "mode": ctx["mode"], "created": now_iso(), "exit_code": 3,
                        "stage": f"p1 --set {a.set}: {e.stage or 'prerequisites'}", "reason": e.reason,
                        "message": str(e), "details": e.details, "tool": tool_record()})
        raise


def p1_run(a, ctx):
    t0 = time.time()
    s = a.set
    if a.max_games and ctx["real"]:
        refuse("--max-games is for drills only")
    fake = bool(a.fake_scorer)
    _fc07, fc08, _m8, _fc04, t02, _rep7, _r4, pre = prereqs(ctx, "p1", a)
    outs = p1_outputs(ctx, s)
    have = [p for p in outs if os.path.exists(p)]
    rerun = None
    if have:
        if not (a.rerun and a.reason):
            refuse(f"{[shown(p) for p in have]} exist; rerun only with --rerun --reason TEXT (resuming needs no flag)")
        if ctx["real"] and any(committed_clean(p) for p in have):
            refuse("the committed P1 outputs are never overwritten")
        rerun = {"reason": a.reason, "old_sha256": {os.path.basename(p): pin_sha(p) for p in have}}
    if not fake:      # nvidia-smi before torch touches CUDA (our own context would count as a job)
        import p07_hs as h7
        procs, _raw = h7.gpu_compute_processes()
        procs = None if procs is None else [p_ for p_ in procs if p_.get("pid") != os.getpid()]
        if procs is None or (procs and not a.ignore_other_gpu_jobs):
            refuse(f"nvidia-smi lists another compute process (or failed): {procs}; wait, or yield",
                   reason="other_gpu_job")
        pre["nvidia_compute_processes_before"] = procs
    env = gpu_env(fake)
    pre["gpu_env"] = check_gpu_env(fc08, env, fake)
    if s != "seam":
        seam_rep = read_json_or_none(os.path.join(ctx["p1"], "p1_seam_report.json"))
        if seam_rep is None:
            refuse("run p1 --set seam first (the seam completes before public and private)")
        se = seam_rep.get("environment") or {}
        if not seam_rep.get("fake_from_val"):
            envd = {k: (se.get(k), env.get(k)) for k in ("torch", "cudnn", "gpu") if str(se.get(k)) != str(env.get(k))}
            if envd:
                drift(f"this environment differs from the seam's {envd}", "p1", "environment_vs_seam", envd)
    chunk_dir = os.path.join(ctx["H"], s)
    stale = stale_chunks(chunk_dir, t02["r20"].get("uploaded_at"))
    if stale:
        refuse(f"{len(stale)} chunk(s) in {shown(chunk_dir)} were written before R20's first upload "
               f"({t02['r20'].get('uploaded_at')}), e.g. {[os.path.basename(p) for p, _ in stale[:3]]}: rename the "
               f"folder to {os.path.basename(chunk_dir)}_stale_<time> (never reused, never deleted by the tool)",
               reason="stale_chunks")
    # ---- the question file and the keys (before p1_lib / p1_extract)
    games, keys, krec = t7call(t7.set_slots, ctx, s)
    krec["t07_hs_keys"] = hs_keys_match(ctx, s, keys)          # the whole set, before a drill cut
    if a.max_games:
        games = games[:a.max_games]
        keys = {k: v[:a.max_games] for k, v in keys.items()}
    keys_sha = krec["keys_sha256"]
    # ---- only now p1_lib / p1_extract (their guard closes data/local for this process)
    try:
        L1, _F, E = r8.import_p1_extract()
    except FOREIGN_ERRORS as e:
        refuse(f"P1 code: {e}")
    import multiprocessing as mp
    import p08_p1 as e8
    os.makedirs(guard_write(ctx, chunk_dir), exist_ok=True)
    refs_all = np.array([g["ref"] for g in games], dtype=np.int64)
    chunks = e8.chunk_plan(len(games), CHUNK)
    shas = json.dumps(L1.code_shas(), sort_keys=True)
    todo = []
    for ci, b, e in chunks:
        p = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
        if os.path.exists(p):
            chunk_check(p, refs_all[b:e], keys_sha, env, ctx["mode"], ctx["real"],
                        questions_sha=krec["questions_sha256"], code_sha=shas, fake=fake)
        else:
            todo.append((ci, b, e))
    extra = sorted(set(chunk_paths(chunk_dir)) - {os.path.join(chunk_dir, f"chunk_{ci:05d}.npz") for ci, _, _ in chunks})
    if extra:
        refuse(f"{shown(chunk_dir)} holds chunks that are not this run's: {[os.path.basename(x) for x in extra[:5]]}")
    print(json.dumps({"time_utc": now_iso(), "msg": "p1 start", "set": s, "mode": ctx["mode"], "games": len(games),
                      "chunks": len(chunks), "todo": len(todo)}), flush=True)
    tool_sha = sha_lf(SELF)
    stopped = None
    if todo:
        if fake:
            sc, ck = e8.FakeProbeScorer(), {"bytes": r8.CKPT_BYTES, "sha256": "fake"}
            torch_v, precision, device = "fake", "fake-fp32", "cpu"
        else:
            import torch
            if not torch.cuda.is_available():
                refuse("CUDA is not available")
            ck = L1.H.check_checkpoint(L1.H.CKPT_PATH, full_hash=True)
            if ck.get("sha256") != r8.CKPT_SHA:
                refuse(f"checkpoint SHA-256 {ck.get('sha256')} is not {r8.CKPT_SHA} (PREREG_08 1.2: use the backup "
                       f"data/katago_backup/, else stop)", reason="checkpoint")
            torch.backends.cudnn.benchmark = r8.RUN["cudnn_benchmark"]
            model = L1.load_model(device=r8.RUN["device"])
            sc = L1.ProbeScorer(model, device=r8.RUN["device"], precision=r8.RUN["precision"], batch=r8.RUN["batch"],
                                channels_last=r8.RUN["channels_last"])
            torch_v, precision, device = torch.__version__, r8.RUN["precision"], r8.RUN["device"]
        group_positions = r8.RUN["batch"] * r8.RUN["group_batches"]
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
                res, t_gpu = E.run_chunk(pool, sc, games[b:e], group_positions)
                secs = time.time() - t1
                npos = int(len(res["pos_side"]))
                save_npz(ctx, os.path.join(chunk_dir, f"chunk_{ci:05d}.npz"),
                         format=np.array(FORMATS["chunk"]), set=np.array(f"test08_{s}"), sides=np.array(r8.RUN["sides"]),
                         ref=refs_all[b:e], precision=np.array(precision), batch=np.array(r8.RUN["batch"]),
                         channels_last=np.array(r8.RUN["channels_last"]), device=np.array(device),
                         torch=np.array(str(torch_v)), cudnn=np.array(str(env["cudnn"])),
                         gpu_name=np.array(str(env["gpu"])), driver=np.array(str(env.get("driver"))),
                         ckpt_bytes=np.array(ck["bytes"]), ckpt_sha256=np.array(ck["sha256"]),
                         katago_commit=np.array(r8.KATAGO_COMMIT), meta_sha256=np.array(L1.neutral_table_sha()),
                         code_sha256=np.array(shas), keys_sha256=np.array(keys_sha),
                         questions_sha256=np.array(krec["questions_sha256"]), tool_sha256=np.array(tool_sha),
                         mode=np.array(ctx["mode"]), created_utc=np.array(now_iso()), seconds=np.array(secs),
                         gpu_seconds=np.array(t_gpu), **res)
                el = time.time() - t_run
                print(json.dumps({"time_utc": now_iso(), "msg": "chunk done", "set": s, "chunk": ci, "games": e - b,
                                  "failed": int(res["failed"].sum()), "positions": npos, "seconds": round(secs, 1),
                                  "positions_per_s": round(npos / max(secs, 1e-9), 1),
                                  "progress": f"{k + 1}/{len(todo)}",
                                  "eta_min": round(el / (k + 1) * (len(todo) - k - 1) / 60, 1)}), flush=True)
        sc.close()
    if stopped:
        print(f"stopped ({stopped}); {shown(chunk_dir)} holds the finished chunks. To continue: delete "
              f"{shown(os.path.join(chunk_dir, 'STOP'))} if present, then run the same line again (resume, not rerun)")
        print(f"STOPPED (exit {EXIT_STOPPED}): p1 --set {s} is not complete; nothing decided", file=sys.stderr)
        return EXIT_STOPPED
    return assemble(ctx, s, chunk_dir, chunks, keys, krec, env, pre, rerun, fake, t0)


def read_chunks(chunk_dir, chunks, with_means=True):
    """The chunks of a set in order -> (rows, arrays: ref, counts, n_moves, failed, scal [, means])."""
    rows, parts = [], []
    for ci, _b, _e in chunks:
        p = os.path.join(chunk_dir, f"chunk_{ci:05d}.npz")
        z = load_npz(p)
        failed = np.asarray(z["failed"], dtype=bool)
        cnt = np.asarray(z["counts"])[:, 0]
        rows.append({"file": os.path.basename(p), "sha256": sha_raw(p), "games": int(len(z["ref"])),
                     "failed": int(failed.sum()), "no_target_moves": int(((cnt == 0) & ~failed).sum()),
                     "no_move": int((np.asarray(z["n_moves"]) == 0).sum()), "created_utc": str(z["created_utc"]),
                     "seconds": float(z["seconds"]), "positions": int(len(z["pos_side"])),
                     "errors": json.loads(str(z["errors"])), "torch": str(z["torch"]), "cudnn": str(z["cudnn"]),
                     "gpu_name": str(z["gpu_name"]), "driver": str(z["driver"]), "mode": str(z["mode"]),
                     "format": str(z["format"]), "meta_sha256": str(z["meta_sha256"])})
        parts.append(z)
    ref = np.concatenate([np.asarray(z["ref"]) for z in parts])
    out = {"ref": ref, "counts": np.concatenate([np.asarray(z["counts"]) for z in parts]).astype(np.int64),
           "n_moves": np.concatenate([np.asarray(z["n_moves"]) for z in parts]).astype(np.int64),
           "failed": np.concatenate([np.asarray(z["failed"]) for z in parts]).astype(bool)}
    means = np.concatenate([np.asarray(z["means"]) for z in parts]).astype(np.float32)
    out["scal"] = means[:, :, 768:770].astype(np.float64)
    if with_means:
        out["means"] = means
    return rows, out


def assemble(ctx, s, chunk_dir, chunks, keys, krec, env, pre, rerun, fake, t0):
    rows, d = read_chunks(chunk_dir, chunks, with_means=True)
    n = len(d["ref"])
    if not np.array_equal(d["ref"], np.arange(n)):
        refuse("chunk refs are not the set's slots in order")
    arrays = {"format": np.array(FORMATS["p1"]), "set": np.array(s), "question_id": np.array(keys["question_id"][:n]),
              "slot": np.array(keys["slot"][:n], dtype=np.int64), "color": np.array(keys["color"][:n]),
              "counts": d["counts"], "n_moves": d["n_moves"], "failed": d["failed"], "scal": d["scal"],
              "scal_fields": np.array(["logp", "entropy"]), "sides": np.array(r8.RUN["sides"])}
    if s == "seam":
        arrays["means"] = d["means"]
    out = save_npz(ctx, os.path.join(ctx["p1"], f"{s}_p1.npz"), **arrays)
    pos = sum(r["positions"] for r in rows)
    secs = sum(r["seconds"] for r in rows)
    fails = int(pf7.hs_failures(d["failed"], d["n_moves"]).sum())
    envs = sorted({(r["torch"], r["cudnn"], r["gpu_name"]) for r in rows})
    rep = {"format": FORMATS["p1_report"], "what": "PREREG_08 5.2 test-day P1 features of one set (no score)",
           "set": s, "mode": ctx["mode"], "created": now_iso(), "fake_scorer": fake, "fake_from_val": False,
           "games": n, "questions": len(dict.fromkeys(keys["question_id"][:n])), "max_games_drill": n < krec["games"],
           "failed": int(d["failed"].sum()), "failed_or_no_move": fails,
           "no_target_moves": int(((d["counts"][:, 0] == 0) & ~d["failed"]).sum()), "chunks": rows,
           "chunk_dir": shown(chunk_dir), "earliest_chunk_created_utc": min(r["created_utc"] for r in rows),
           "latest_chunk_created_utc": max(r["created_utc"] for r in rows),
           "chunk_environments": [list(x) for x in envs], "positions": pos, "gpu_seconds": round(secs, 1),
           "positions_per_s": round(pos / max(secs, 1e-9), 1), "environment": env, "keys": krec,
           "output": {"file": f"{s}_p1.npz", "sha256": sha_raw(out)}, "prerequisites": pre, "rerun": rerun,
           "run": r8.RUN, "tool": tool_record(), "seconds_this_call": round(time.time() - t0, 1),
           "next": f"commit {shown(ctx['p1'])}/{s}_p1.npz and p1_{s}_report.json"}
    write_json(ctx, os.path.join(ctx["p1"], f"p1_{s}_report.json"), rep)
    print(f"p1 {s}: {n} games, failed {fails}, {pos} positions in {secs:.0f} s ({rep['positions_per_s']} "
          f"positions/s) -> {shown(out)}")
    return 0


# ================================================================ the GPU seam and the P1 term (pure numpy parts)
def std_x(means, counts, mu, sd):
    """p1_fuse.variant_X(.., 'TO') + standardise with the frozen mu / sd (numpy port for the GPU seam record only;
    the predictions use p1_fuse itself): [G, 1540] standardised, missing opponent -> 0; usable [G]."""
    means = np.asarray(means, dtype=np.float64)
    counts = np.asarray(counts)
    T = means[:, 0, :]
    O = means[:, 1, :].copy()
    O[counts[:, 1] == 0] = np.nan
    X = (np.concatenate([T, O], axis=1) - mu) / sd
    X[~np.isfinite(X)] = 0.0
    return X


def lp_np(Xs, model):
    Z = Xs @ model["W"] + model["b"]
    Z = Z - Z.max(1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(1, keepdims=True))


def gpu_seam_record(z, ref, model):
    """PREREG_08 5.2 GPU seam: the 400 seam games (z: seam_p1.npz) against val_seam_feats.npz (ref). failed / counts /
    n_moves / keys equal; the largest |difference| of the standardised 1,540 features and of lp_P1TO (per game, before
    the x 4) over the usable games."""
    n = SEAM_GAMES
    probs = []
    if len(z["failed"]) != n:
        probs.append(f"seam has {len(z['failed'])} games, registered {n}")
    k = min(n, len(z["failed"]), len(ref["failed"]))
    f_eq = np.array_equal(np.asarray(z["failed"][:k], dtype=bool), np.asarray(ref["failed"][:k], dtype=bool))
    c_eq = np.array_equal(np.asarray(z["counts"][:k]).astype(np.int64), np.asarray(ref["counts"][:k]).astype(np.int64))
    m_eq = np.array_equal(np.asarray(z["n_moves"][:k]).astype(np.int64),
                          np.asarray(ref["n_moves"][:k]).astype(np.int64))
    keys_eq = (list(map(str, z["question_id"][:k])) == list(map(str, ref["question_id"][:k]))
               and np.array_equal(np.asarray(z["slot"][:k]).astype(np.int64), np.asarray(ref["slot"][:k]).astype(np.int64))
               and list(map(str, z["color"][:k])) == list(map(str, ref["color"][:k])))
    counts_equal = bool(f_eq and c_eq and m_eq and keys_eq and k == n)
    if not counts_equal:
        probs.append(f"failed equal {f_eq}, counts equal {c_eq}, n_moves equal {m_eq}, keys equal {keys_eq}")
    ok_g = ~np.asarray(ref["failed"][:k], dtype=bool) & (np.asarray(ref["counts"][:k])[:, 0] > 0)
    if counts_equal and ok_g.any():
        xa = std_x(z["means"][:k], z["counts"][:k], model["mu"], model["sd"])[ok_g]
        xb = std_x(ref["means"][:k], ref["counts"][:k], model["mu"], model["sd"])[ok_g]
        max_feat = float(np.max(np.abs(xa - xb)))
        max_lp = float(np.max(np.abs(lp_np(xa, model) - lp_np(xb, model))))
    else:
        max_feat = max_lp = float("inf")
    bit = bool(counts_equal and np.array_equal(np.asarray(z["means"][:k]), np.asarray(ref["means"][:k])))
    passed = bool(counts_equal and max_feat <= TOL_FEAT and max_lp <= TOL_LP)
    fin = lambda v: v if math.isfinite(v) else None  # noqa: E731
    return {"games": int(k), "usable_games": int(ok_g.sum()), "counts_equal": counts_equal,
            "max_abs_std_feature": fin(max_feat), "max_abs_lp_p1": fin(max_lp),
            "tolerances": {"std_feature": TOL_FEAT, "lp_p1": TOL_LP},
            "lp_review_line": LP_REVIEW, "lp_above_review_line": bool(math.isfinite(max_lp) and max_lp > LP_REVIEW),
            "bitwise_equal_means": bit, "problems": probs, "passed": passed,
            "lp_definition": "lp_P1TO = log_softmax(W z + b) per game with the frozen p1_map.npz, before the x 4"}


def p1_cross_check(p, hs, o4, ids):
    """PREREG_08 5.2: P1 n_moves / target moves against test07's HS (n_moves, counts) and test04's O4 replay
    (moves_parsed, n_moves[:, 0, :] total), on the games where neither side failed."""
    a = pf7.move_cross_check(p["n_moves"], np.asarray(p["counts"])[:, 0], p["failed"], hs["n_moves"], hs["counts"],
                             hs["failed"], ids)
    b = pf7.move_cross_check(p["n_moves"], np.asarray(p["counts"])[:, 0], p["failed"], o4["moves_parsed"],
                             np.asarray(o4["n_moves"])[:, 0, :].sum(1), o4["replay_failed"], ids)
    return {"vs_t07_hs": a, "vs_t04_o4": b, "mismatches": int(a["mismatches"] + b["mismatches"])}


# ================================================================ apply
def apply_outputs(ctx):
    rank, seam = ctx["rank"], os.path.join(ctx["rank"], "seam_val")
    out = [os.path.join(seam, n) for n in ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json",
                                           "seam_local.json")]
    out += [os.path.join(rank, n) for n in TEST_OUTPUTS + ("t08_apply_report.json", "attempt_verdict.json")]
    out += [os.path.join(ctx["p1"], "gpu_seam.json"), os.path.join(ctx["R8"], "not_used.json")]
    return out


def check_bug_revision(n, overturned):
    """--bug-revision N: PREREG_08 rev N >= 4 committed and clean, its row at HEAD, added after every overturned
    record (ISO times). -> record."""
    if n is None:
        return None
    if n < P08_MIN_REV:
        refuse(f"--bug-revision {n}: an override needs PREREG_08 rev {P08_MIN_REV} or later")
    if not committed_clean(P08_DOC):
        refuse("docs/PREREG_08_STRENGTH_PROBE.md must be committed and clean")
    row = f"| rev {n} |"
    rc, head = git("show", "HEAD:docs/PREREG_08_STRENGTH_PROBE.md")
    if rc != 0 or not any(line.startswith(row) for line in head.splitlines()):
        refuse(f"docs/PREREG_08_STRENGTH_PROBE.md at HEAD has no row starting {row!r}")
    rc, out = git("log", "--reverse", "-S" + row, "--format=%H %cI", "--", "docs/PREREG_08_STRENGTH_PROBE.md")
    if rc != 0 or not out:
        refuse(f"no commit added {row!r}")
    h, t = out.splitlines()[0].split()
    at = parse_iso(t, "commit time")
    early = [f"{w} {x}" for w, x in overturned if x and at <= parse_iso(x, w)]
    if early:
        refuse(f"PREREG_08 rev {n} ({h[:12]}, {t}) is not later than {early}")
    return {"rev": n, "commit": h, "committed_at": t, "overturns": [{"what": w, "time": x} for w, x in overturned]}


def bound_problems(ctx, bound):
    """Every input test07's apply bound (keys: real repository paths) still has its SHA-256 (tree paths outside the
    real mode)."""
    probs = []
    if not bound:
        return ["t07_apply_report.json lists no inputs_bound"]
    roots = {"results/test07/": ctx["R7"], "results/test04/": ctx["R4"], "results/test02/checks/": ctx["C"],
             "results/test02/": ctx["D2"]}
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


def seam_verdict(sc, inputs_identical):
    return t7.seam_verdict(sc, inputs_identical)


class Apply:
    def __init__(self, ctx, a, t0):
        self.ctx, self.a, self.t0 = ctx, a, t0
        self.stage, self.pre, self.replaced, self.bug, self.rec = "prerequisites", {}, {}, None, {}
        self._code = {}

    # ---- rerun rules (as t07)
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
        rep = read_json_or_none(os.path.join(ctx["rank"], "t08_apply_report.json"))
        nu = read_json_or_none(os.path.join(ctx["R8"], "not_used.json"))
        decided = [(n, r.get("created")) for n, r in (("t08_apply_report.json", rep), ("not_used.json", nu)) if r]
        ver = read_json_or_none(os.path.join(ctx["rank"], "attempt_verdict.json"))
        if ver and ver.get("reason") == "move_cross_check":    # only after a revision naming the bug
            decided.append(("attempt_verdict.json (move cross-check)", ver.get("created")))
        if decided:
            if a.bug_revision is None:
                refuse(f"{[n for n, _ in decided]} hold the registered decision; a rerun needs a PREREG_08 revision "
                       f"naming a program error (--bug-revision N, N >= {P08_MIN_REV})")
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
            write_json(ctx, os.path.join(ctx["R8"], "not_used.json"), dict(rec, format=FORMATS["not_used"]))
        else:
            write_json(ctx, os.path.join(ctx["rank"], "attempt_verdict.json"), rec)

    # ---- step 1: the three P1 sets
    def p1_sets(self, t02):
        ctx, out = self.ctx, {}
        for s in SETS:
            npz, rp = p1_outputs(ctx, s)
            for p in (npz, rp):
                if not os.path.exists(p):
                    refuse(f"missing {shown(p)}: run p1 --set {s} first", reason="p1_missing")
            if ctx["real"]:
                require_committed([npz, rp], f"the {s} P1 (commit it after p1)")
            rep = read_json(rp)
            if rep.get("format") != FORMATS["p1_report"] or rep.get("mode") != ctx["mode"] or rep.get("set") != s:
                refuse(f"{shown(rp)}: format {rep.get('format')!r}, mode {rep.get('mode')!r}, set {rep.get('set')!r}")
            if (rep.get("fake_from_val") or rep.get("fake_scorer") or rep.get("max_games_drill")) and ctx["real"]:
                refuse(f"{shown(rp)} is a drill record (fake or --max-games): never in the real mode")
            if (rep.get("output") or {}).get("sha256") != sha_raw(npz):
                refuse(f"{shown(npz)} differs from its report")
            want = -(-int(rep.get("games") or 0) // CHUNK)
            if len(rep.get("chunks") or []) != want or not rep.get("games"):
                refuse(f"{shown(rp)} lists {len(rep.get('chunks') or [])} chunks for {rep.get('games')} games")
            bad = []
            for ch in rep.get("chunks") or []:
                p = os.path.join(ctx["H"], s, ch["file"])
                if not os.path.exists(p) or sha_raw(p) != ch["sha256"]:
                    bad.append(ch["file"])
                elif ch.get("format") != FORMATS["chunk"] or ch.get("meta_sha256") != r8.NEUTRAL_META_SHA or \
                        ch.get("mode") != ctx["mode"]:
                    bad.append(f"{ch['file']} (format / metadata / mode)")
            if bad:
                refuse(f"{s}: P1 chunks missing or changed since p1: {bad[:5]}")
            meta_bad = self.chunk_meta_problems(s, rep)
            if meta_bad:
                refuse(f"{s}: P1 chunks of another run (key / question file, P1 code, checkpoint or scorer settings): "
                       f"{meta_bad[:5]}", reason="p1_chunk_meta")
            z = load_npz(npz)
            if str(z["format"]) != FORMATS["p1"] or str(z["set"]) != s or len(z["failed"]) != rep["games"]:
                refuse(f"{shown(npz)} is not the {s} P1 set of its report")
            out[s] = {"report": rep, "report_path": rp, "npz_path": npz, "z": z, "code_sha256": self._code[s]}
        envs = {s: tuple(str((out[s]["report"].get("environment") or {}).get(k)) for k in ("torch", "cudnn", "gpu"))
                for s in SETS}
        if len(set(envs.values())) != 1:
            drift(f"the three P1 sets were computed in different environments {envs}", "p1_sets", "p1_environment",
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
            refuse(f"P1 computed before R20's first upload ({r20_at}): {early}; rename the chunk folders and rerun "
                   f"p1", reason="p1_before_r20")
        self.rec["p1_sets"] = {s: {"report_sha256": pin_sha(out[s]["report_path"]),
                                   "npz_sha256": sha_raw(out[s]["npz_path"]), "games": out[s]["report"].get("games"),
                                   "failed_or_no_move": out[s]["report"].get("failed_or_no_move"),
                                   "earliest_chunk_created_utc": out[s]["report"].get("earliest_chunk_created_utc"),
                                   "fake_from_val": bool(out[s]["report"].get("fake_from_val")),
                                   "positions_per_s": out[s]["report"].get("positions_per_s")} for s in SETS}
        codes = {s: out[s]["code_sha256"] for s in SETS}
        if len(set(codes.values())) != 1:
            refuse("the three P1 sets were computed with different P1 code: "
                   + str({s: hashlib.sha256(c.encode()).hexdigest()[:12] for s, c in codes.items()}), reason="p1_code")
        code_now = self.p1_code_problems(codes["seam"])
        if code_now:
            refuse(f"the P1 code the chunks record differs from the files on disk: {code_now}", reason="p1_code")
        self.rec["p1_environment"] = dict(zip(("torch", "cudnn", "gpu"), envs["seam"]))
        self.rec["p1_code_sha256"] = json.loads(codes["seam"])
        self.rec["r20_uploaded_at"] = r20_at
        return out

    def chunk_meta_problems(self, s, rep):
        """Every chunk of a set (rev 3 section 13 addendum): the report's key file and question file (fake_from_val:
        the from-val marks), one P1 code, the frozen checkpoint (fake scorer: 'fake') and the frozen scorer settings;
        the question file of public / private is also the one in the tests folder now (seam: the registered
        validation question file)."""
        ctx = self.ctx
        fv, fk = bool(rep.get("fake_from_val")), bool(rep.get("fake_scorer"))
        keys = rep.get("keys") or {}
        want = {"keys_sha256": "fake_from_val" if fv else keys.get("keys_sha256"),
                "questions_sha256": "fake_from_val" if fv else keys.get("questions_sha256"),
                "ckpt_sha256": "fake" if fk else r8.CKPT_SHA, **run_settings(fk)}
        bad, codes = [], set()
        if not fv:
            if s == "seam":
                q_now = pf7.QUESTIONS_SHA
            else:
                qf = os.path.join(ctx["T"], t2.TEST_FILES[s])
                q_now = sha_raw(qf) if os.path.exists(qf) else None
            if keys.get("questions_sha256") != q_now:
                bad.append(f"report question file {str(keys.get('questions_sha256'))[:12]} is not "
                           f"{str(q_now)[:12]} now")
        for ch in rep.get("chunks") or []:
            p = os.path.join(ctx["H"], s, ch["file"])
            with np.load(p, allow_pickle=False) as z:
                got = {"keys_sha256": str(z["keys_sha256"]), "questions_sha256": str(z["questions_sha256"]),
                       "ckpt_sha256": str(z["ckpt_sha256"]), **chunk_run_meta(z)}
                codes.add(str(z["code_sha256"]))
            d = [k for k in want if got[k] != want[k]]
            if d:
                bad.append(f"{ch['file']} {d}")
        if len(codes) != 1:
            bad.append(f"{len(codes)} different P1 code records")
        self._code[s] = sorted(codes)[0] if codes else ""
        return bad

    @staticmethod
    def p1_code_problems(code_json):
        """The P1 code record of the chunks (LF SHA-256 of p1_lib / p1_extract / p1_fuse and the files p1_lib pins)
        equals the files on disk now."""
        try:
            code = json.loads(code_json)
        except ValueError:
            return ["unreadable code record"]
        need = ("results/prereg08/p1/p1_lib.py", "results/prereg08/p1/p1_extract.py")
        bad = [k for k in need if k not in code]
        for k, v in sorted(code.items()):
            p = path_of(k)
            if not os.path.exists(p) or sha_lf(p) != v:
                bad.append(k)
        return bad

    def test_means(self, p1, s):
        """The full means of public / private from the chunks the report lists (SHA-256 checked again), equal in
        counts / n_moves / failed to <set>_p1.npz."""
        rep, z = p1[s]["report"], p1[s]["z"]
        chunk_dir = os.path.join(self.ctx["H"], s)
        plan = [(i, 0, 0) for i in range(len(rep["chunks"]))]
        rows, d = read_chunks(chunk_dir, plan, with_means=True)
        if [r["sha256"] for r in rows] != [c["sha256"] for c in rep["chunks"]]:
            refuse(f"{s}: P1 chunks changed since p1")
        if not np.array_equal(d["ref"], np.arange(len(z["failed"]))):
            refuse(f"{s}: chunk refs are not the set's slots in order")
        for k in ("counts", "n_moves", "failed"):
            if not np.array_equal(np.asarray(z[k]).astype(d[k].dtype), d[k]):
                refuse(f"{s}: {s}_p1.npz {k} differs from its chunks")
        return {"means": d["means"].astype(np.float64), "counts": d["counts"], "n_moves": d["n_moves"],
                "failed": d["failed"]}

    # ---- step 2: the GPU seam
    def gpu_seam(self, p1, model):
        ctx = self.ctx
        z = p1["seam"]["z"]
        if "means" not in z:
            refuse("seam_p1.npz holds no means")
        ref = load_npz(path_of("results/prereg08/feat/val_seam_feats.npz"))
        rec = gpu_seam_record(z, ref, model)
        rec.update(format=FORMATS["gpu_seam"], mode=ctx["mode"], created=now_iso(),
                   seam_p1_sha256=sha_raw(p1["seam"]["npz_path"]),
                   reference={"val_seam_feats.npz": P08_PINS["results/prereg08/feat/val_seam_feats.npz"],
                              "p1_map.npz": P08_PINS["results/prereg08/rank/p1_map.npz"]})
        path = write_json(ctx, os.path.join(ctx["p1"], "gpu_seam.json"), rec)
        self.rec["gpu_seam"] = {k: rec[k] for k in ("passed", "counts_equal", "max_abs_std_feature", "max_abs_lp_p1",
                                                    "bitwise_equal_means", "usable_games")}
        print(f"GPU seam: counts equal {rec['counts_equal']}, max |d std feature| {rec['max_abs_std_feature']} "
              f"(<= {TOL_FEAT}), max |d lp_P1TO| {rec['max_abs_lp_p1']} (<= {TOL_LP}) -> "
              f"{'PASSED' if rec['passed'] else 'FAILED'}", flush=True)
        if not rec["passed"]:
            drift("GPU seam failed (local GPU / driver / environment drift; PREREG_08 5.2): stop and find out why "
                  "(PREREG_08 5.3)", "gpu_seam", "gpu_seam",
                  {k: rec[k] for k in ("counts_equal", "max_abs_std_feature", "max_abs_lp_p1", "problems")})
        return path

    # ---- the run
    def go(self):
        ctx, a = self.ctx, self.a
        fc07, fc08, model08, fc04, t02, rep7, rep4, self.pre = prereqs(ctx, "apply", a)
        self.rerun_gate()
        self.stage = "p1_sets"
        p1 = self.p1_sets(t02)
        model07 = pf7.load_hs_map(path_of("results/prereg07/rank/hs_map.npz"))
        self.clear()
        self.stage = "gpu_seam"
        gpu_seam_path = self.gpu_seam(p1, model08)
        # ---- step 3: Q02H recomputed, equal to test07's rank_test_probs.csv; 07's bound inputs unchanged
        self.stage = "q02h_vs_07"
        bound7 = rep7.get("inputs_bound") or {}
        b7 = bound_problems(ctx, bound7)
        if b7:
            refuse(f"test07's bound inputs changed since its apply: {b7[:5]} (follow docs/TESTDAY_07.md first)",
                   reason="t07_inputs_changed")
        vk = {s: rf.load_val_keys(os.path.join(ctx["D2"], "rank", s, "dumps")) for s in Q_SETS}
        sw = {s: t7call(t7.load_sw, ctx, s, vk[s]) for s in Q_SETS}
        o4z = {s: load_npz(os.path.join(ctx["R4"], "opp", f"o4_{s}.npz")) for s in Q_SETS}
        c5w = fc04["source_sha256"]["C5"]["weights_sha256"]
        fz = {}
        for s in Q_SETS:     # every question-file read of this process happens here (seam_val: data/local)
            ddir = os.path.join(ctx["D2"], "rank", s, "dumps")
            c5 = t7call(t7.t4call, t4.load_c5, s, ddir, vk[s], c5w)
            f = t7call(t7.t4call, t4.fuse_set, s, vk[s], t02["qfiles"][s], c5, t02["opening"], o4z[s], sw[s], fc04,
                       False)
            f["c5_sha256"] = c5["sha256"]
            f.update(t7.port_c5(vk[s], c5, fc04, f))
            fz[s] = f
        if not all(fz[s]["port_equal"] for s in Q_SETS):
            drift("the ported C5 lines do not give fuse_set's Q02 bit for bit", "q02h_vs_07", "port_c5",
                  {s: fz[s]["port_equal"] for s in Q_SETS})
        # ---- from here no data/local file is opened: pilot_fuse (hs_lib's guard), then p1_fuse
        PF = pf7.import_pilot_fuse()
        try:
            _L1, F = r8.import_p1()
        except FOREIGN_ERRORS as e:
            refuse(f"P1 code: {e}")
        hs = {s: load_npz(os.path.join(ctx["R7"], "hs", f"{s}_hs.npz")) for s in TEST_SETS}
        h = {}
        for s in TEST_SETS:
            keys_ok = (list(map(str, hs[s]["question_id"])) == vk[s]["question_id"]
                       and [int(x) for x in hs[s]["slot"]] == vk[s]["slot"]
                       and list(map(str, hs[s]["color"])) == vk[s]["color"])
            if not keys_ok:
                refuse(f"test07's {s}_hs.npz keys differ from test02's val_keys.npz of {s}", reason="hs_keys")
            h[s] = t7call(t7.q02h_set, fz[s], vk[s], hs[s], model07, fc07, PF)
        q02h_bytes = t7.probs_bytes(h["public"]["rows"] + h["private"]["rows"])
        ref7 = os.path.join(ctx["R7"], "rank", "rank_test_probs.csv")
        with open(ref7, "rb") as fh:
            ref_bytes = fh.read().replace(b"\r\n", b"\n")
        self.rec["q02h_vs_07"] = {"equal": q02h_bytes == ref_bytes,
                                  "computed_sha256": hashlib.sha256(q02h_bytes).hexdigest(),
                                  "t07_rank_test_probs_sha256": pin_sha(ref7), "t07_inputs_bound_checked": len(bound7)}
        if not self.rec["q02h_vs_07"]["equal"]:
            drift("Q02H recomputed with test07's inputs differs from results/test07/rank/rank_test_probs.csv (local "
                  "problem; PREREG_08 5.2)", "q02h_vs_07", "q02h_vs_07", self.rec["q02h_vs_07"])
        # ---- step 4: the move cross-check against test07's HS and test04's O4 replay
        self.stage = "move_cross_check"
        means, cross = {}, {}
        for s in TEST_SETS:
            z = p1[s]["z"]
            keys_ok = (list(map(str, z["question_id"])) == vk[s]["question_id"]
                       and [int(x) for x in z["slot"]] == vk[s]["slot"] and list(map(str, z["color"])) == vk[s]["color"])
            if not keys_ok:
                refuse(f"{s}_p1.npz keys differ from test02's val_keys.npz of {s}", reason="p1_keys")
            means[s] = self.test_means(p1, s)
            cross[s] = p1_cross_check(means[s], hs[s], o4z[s],
                                      [f"{q}#{sl}" for q, sl in zip(vk[s]["question_id"], vk[s]["slot"])])
        self.rec["move_cross_check"] = cross
        if any(cross[s]["mismatches"] != 0 for s in TEST_SETS):
            drift(f"move cross-check against test07's <set>_hs.npz / test04's o4_<set>.npz: "
                  f"{[cross[s]['mismatches'] for s in TEST_SETS]} mismatches (registered 0): a program or "
                  f"input-binding error; only a committed PREREG_08 revision naming it allows a rerun",
                  "move_cross_check", "move_cross_check", cross)
        # ---- step 5: the local Q08 seam on the validation questions
        self.stage = "local_seam"
        vh = load_npz(path_of("results/prereg07/hs/val_hs.npz"))
        seam = self.local_seam(fz["seam_val"], vk["seam_val"], vh, model07, fc07, fc08, PF, F)
        # ---- steps 6, 7: P1 failures and direction
        self.stage = "p1_failures"
        res = {s: q08_set(fz[s], vk[s], h[s], means[s], model08, fc08, F) for s in TEST_SETS}
        fails = {s: {"failed_or_no_move": res[s]["fail_count"], "games": vk[s]["n"],
                     "ok": rate_ok(res[s]["fail_count"], vk[s]["n"])} for s in TEST_SETS}
        self.rec["p1_failures"] = fails
        bad = [s for s in TEST_SETS if not fails[s]["ok"]]
        if bad:
            not_used(f"P1 failures above 0.5 % on {bad} ({fails}): 08 is not used (PREREG_08 5.2)", "p1_failures",
                     "p1_failures", fails)
        self.stage = "p1_direction"
        direc = {s: pf7.direction_record(res[s]["rho"], res[s]["rho_n"], RHO_MONO, RHO_FLOOR) for s in TEST_SETS}
        direc["seam_val_record_only"] = seam["direction"]
        self.rec["p1_direction"] = direc
        bad = [s for s in TEST_SETS if not direc[s]["passed"]]
        if bad:
            not_used(f"P1 direction check failed on {bad} ({ {s: direc[s]['rho'] for s in TEST_SETS} } < "
                     f"{direc['public']['threshold']}): 08 is not used (PREREG_08 5.2)", "p1_direction",
                     "p1_direction", direc)
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
        save_npz(ctx, os.path.join(rank, "test_lp_p1.npz"), format=np.array(FORMATS["lp"]),
                 question_id=np.array(vk["public"]["question_id"] + vk["private"]["question_id"]),
                 slot=np.array(vk["public"]["slot"] + vk["private"]["slot"], dtype=np.int64),
                 color=np.array(vk["public"]["color"] + vk["private"]["color"]),
                 lp=np.concatenate([res[s]["lp"] for s in TEST_SETS]),
                 p1_ok=np.concatenate([res[s]["p1_ok"] for s in TEST_SETS]),
                 usable=np.concatenate([res[s]["usable"] for s in TEST_SETS]), scale=np.array(r8.SCALE),
                 p1_map_sha256=np.array(P08_PINS["results/prereg08/rank/p1_map.npz"]))
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
            drift(f"check_submission --official failed on the Q08 file: {errors[:5]}; {removed} removed",
                  "check_submission", "check_submission", {"errors": errors[:50], "removed": removed})
        # ---- step 9: the report
        preds = dict(t2.read_pred(sub))
        pub_ids = set(fz["public"]["qids"])
        diffs = {}
        for name, other in (("07", os.path.join(ctx["R7"], "rank", "submission_rank.csv")),
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
        rep = {"format": FORMATS["apply"], "what": "test08 competition 2 apply (PREREG_08 5.2): frozen Q08 per set, "
                                                   "rule B, submission_rank.csv = pred_B.csv (public then private)",
               "mode": ctx["mode"], "created": now_iso(), "exit_code": 0, "usable": True, "unusable_reasons": [],
               "combo": COMBO, "method": METHOD, "theta": fc08["params"]["theta"],
               "h8r": self.pre["h8r"], "t1_verdict": (self.pre.get("t1") or {}).get("verdict"),
               "test02": {"fallback_used": t02["report"].get("fallback_used"),
                          "apply_test_report_sha256": t02["report_sha256"],
                          "r20_first_upload_id": (t02["r20"] or {}).get("id"),
                          "r20_first_uploaded_at": (t02["r20"] or {}).get("uploaded_at")},
               "test07": {"t07_apply_report_sha256": pin_sha(os.path.join(ctx["R7"], "rank", "t07_apply_report.json")),
                          "exit_code": rep7.get("exit_code"), "usable": rep7.get("usable"),
                          "seam_local": rep7.get("seam_local"), "inputs_bound_checked": len(bound7)},
               "test04": {"t04_apply_report_sha256": pin_sha(os.path.join(ctx["R4"], "rank", "t04_apply_report.json")),
                          "exit_code": rep4.get("exit_code"), "usable": rep4.get("usable"),
                          "seam_local_tol": (rep4.get("seam_local") or {}).get("tol")},
               "p1_sets": self.rec["p1_sets"], "p1_environment": self.rec["p1_environment"],
               "earliest_p1_chunk_created_utc": min(self.rec["p1_sets"][s]["earliest_chunk_created_utc"] or ""
                                                    for s in SETS),
               "gpu_seam": self.rec["gpu_seam"], "gpu_seam_tolerances": {"std_feature": TOL_FEAT, "lp_p1": TOL_LP},
               "q02h_vs_07": self.rec["q02h_vs_07"], "move_cross_check": cross, "seam_local": seam["record"],
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
               "paths": {k: ctx[k] for k in ("R8", "H", "R7", "R4", "D2", "O2", "T", "C")}, "tool": tool_record(),
               "environment": t4.environment(), "git_head": git_head(),
               "record_only": "level_summary, rule_b_differences (all 800) and the seam_val direction are recorded; "
                              "nothing is chosen from them"}
        rep["seconds"] = round(time.time() - self.t0, 1)
        write_json(ctx, os.path.join(rank, "t08_apply_report.json"), rep)
        print(f"apply Q08: public {len(fz['public']['qids'])}, private {len(fz['private']['qids'])} questions -> "
              f"{shown(sub)} (sha256 {rep['outputs_sha256']['submission_rank.csv'][:12]}); check_submission official: "
              f"OK; rule-B differences on public vs 07 / 04 / R20: {(diffs.get('07') or {}).get('public')} / "
              f"{(diffs.get('04') or {}).get('public')} / {(diffs.get('R20') or {}).get('public')}")
        print("下一步：commit results/test08/，07 上傳後照 docs/TESTDAY_08.md 第 5 步 official_log.py prepare"
              if ctx["real"] else f"({ctx['mode']}: nothing is uploaded)")
        return 0

    def local_seam(self, fzs, vks, vh, model07, fc07, fc08, PF, F):
        """PREREG_08 5.2: Q08 on seam_val (test02 / test04 / 07 seam inputs + the committed val_lp_p1.npz) against
        results/prereg08/rank/rank_val_probs.csv."""
        ctx = self.ctx
        keys_ok = (list(map(str, vh["question_id"])) == vks["question_id"] and [int(x) for x in vh["slot"]] ==
                   vks["slot"] and list(map(str, vh["color"])) == vks["color"])
        if not keys_ok:
            drift("val_hs.npz keys differ from test02's seam_val val_keys.npz", "local_seam", "seam_structure")
        hq = t7call(t7.q02h_set, fzs, vks, vh, model07, fc07, PF)
        lpz = load_npz(path_of("results/prereg08/rank/val_lp_p1.npz"))
        if (list(map(str, lpz["question_id"])) != vks["question_id"] or [int(x) for x in lpz["slot"]] != vks["slot"]
                or list(map(str, lpz["color"])) != vks["color"]):
            drift("val_lp_p1.npz keys differ from test02's seam_val val_keys.npz", "local_seam", "seam_structure")
        if float(lpz["scale"]) != r8.SCALE or str(lpz["p1_map_sha256"]) != P08_PINS["results/prereg08/rank/p1_map.npz"]:
            refuse("val_lp_p1.npz is not the frozen map's lp (scale 4, p1_map.npz)")
        usable = np.asarray(vks["parsed"], dtype=bool) & np.asarray(lpz["p1_ok"], dtype=bool)
        r = q08_core(fzs, vks, hq, np.asarray(lpz["lp"], dtype=np.float64), np.asarray(lpz["p1_ok"], dtype=bool),
                     usable, fc08)
        seam_dir = os.path.join(ctx["rank"], "seam_val")
        os.makedirs(guard_write(ctx, seam_dir), exist_ok=True)
        csv_p = guard_write(ctx, os.path.join(seam_dir, "rank_val_probs.csv"))
        t2.write_probs(csv_p, r["rows"])
        for n in ("pred_A.csv", "pred_B.csv", "decide_summary.json"):
            guard_write(ctx, os.path.join(seam_dir, n))
        decide.run(csv_p, seam_dir)
        ref_p, ref_b = os.path.join(P08_RANK, "rank_val_probs.csv"), os.path.join(P08_RANK, "pred_B.csv")
        rep4_tol = (read_json(os.path.join(ctx["R4"], "rank", "t04_apply_report.json")).get("seam_local") or {}).get("tol")
        tol = float(rep4_tol)
        sc = t2.seam_compare(csv_p, os.path.join(seam_dir, "pred_B.csv"), ref_p, ref_b, tol, False)
        same_a = t2._lf_bytes(os.path.join(seam_dir, "pred_A.csv")) == t2._lf_bytes(os.path.join(P08_RANK,
                                                                                                 "pred_A.csv"))
        sl7 = read_json(os.path.join(ctx["R7"], "rank", "seam_val", "seam_local.json"))
        inputs_identical = sl7.get("byte_identical_probs") is True and sl7.get("byte_identical_pred_B") is True
        verdict = seam_verdict(sc, inputs_identical)
        dref = fc08["p1_direction_ref"]
        direction = pf7.direction_record(r["rho"], r["rho_n"], dref["rho_eval"], r8.RHO_FLOOR_VAL)
        a8 = read_json(os.path.join(P08_RANK, "apply_report.json"))
        direction["equal_to_prereg08_apply"] = r["rho"] == (a8.get("p1_direction") or {}).get("rho")
        q02h7 = t7.probs_bytes(hq["rows"]) == t2._lf_bytes(os.path.join(ctx["R7"], "rank", "seam_val",
                                                                       "rank_val_probs.csv"))
        if inputs_identical and not direction["equal_to_prereg08_apply"]:
            drift("seam_val P1 direction rho differs from results/prereg08/rank/apply_report.json although 07's "
                  "seam_val is byte-identical to prereg07", "local_seam", "seam_rho", direction)
        rec = {"format": FORMATS["seam_local"], "what": "test08 local Q08 seam (PREREG_08 5.2): Q08 on the seam_val "
                                                       "inputs + val_hs.npz + val_lp_p1.npz vs results/prereg08/rank",
               "mode": ctx["mode"], "created": now_iso(), **sc, "byte_identical_pred_A": same_a,
               "t07_seam_byte_identical": inputs_identical, "required_byte_identical": inputs_identical,
               "q02h_equal_to_t07_seam_val_record_only": q02h7, "usable_equal_to_prereg08":
                   bool(np.array_equal(usable, np.asarray(lpz["usable"], dtype=bool))),
               "verdict": verdict, "passed": verdict == "passed",
               "tol_source": "test04 t04_apply_report.json seam_local.tol (as 07)",
               "set": r["info"], "direction": direction,
               "reference_sha256": {n: pin_sha(os.path.join(P08_RANK, n))
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
            drift("local seam: questions / order / game counts differ from results/prereg08/rank/rank_val_probs.csv",
                  "local_seam", "seam_structure", {"problems": sc["problems"],
                                                   "count_mismatch_questions": sc["count_mismatch_questions"]})
        if verdict == "drift":
            drift("local seam: 07's seam_val equals prereg07 byte for byte but Q08 does not equal prereg08 byte for "
                  "byte (program drift)", "local_seam", "seam_bytes", self.rec["seam_local"])
        if verdict == "not_used":
            not_used("local seam outside the tolerance while 07's seam holds: 08's weights amplify the Colab "
                     "differences of 04's inputs, which the local machine cannot repair: 08 is not used (PREREG_08 "
                     "5.2)", "local_seam", "seam_tol", self.rec["seam_local"])
        return {"path": path, "record": self.rec["seam_local"], "direction": direction}

    def inputs_bound(self, t02, gpu_seam_path, seam_local_path):
        """The files official_log re-hashes (keys: the real-mode repository paths)."""
        ctx = self.ctx
        R4, D2, R7, R8 = ctx["R4"], ctx["D2"], ctx["R7"], ctx["R8"]
        items = [("results/test02/rank/apply_test_report.json", t02["report_path"]),
                 ("results/test02/rank/opening_test.npz", t02["opening_path"]),
                 ("results/test02/checks/rank/t1.json", os.path.join(ctx["C"], "rank", "t1.json")),
                 ("results/test04/rank/t04_apply_report.json", os.path.join(R4, "rank", "t04_apply_report.json")),
                 ("results/test04/rank/rank_test_probs.csv", os.path.join(R4, "rank", "rank_test_probs.csv")),
                 ("results/test04/rank/seam_val/seam_local.json", os.path.join(R4, "rank", "seam_val",
                                                                               "seam_local.json")),
                 ("results/test07/rank/t07_apply_report.json", os.path.join(R7, "rank", "t07_apply_report.json")),
                 ("results/test07/rank/rank_test_probs.csv", os.path.join(R7, "rank", "rank_test_probs.csv")),
                 ("results/test07/rank/submission_rank.csv", os.path.join(R7, "rank", "submission_rank.csv")),
                 ("results/test07/rank/seam_val/seam_local.json", os.path.join(R7, "rank", "seam_val",
                                                                               "seam_local.json")),
                 ("results/test07/hs/gpu_seam.json", os.path.join(R7, "hs", "gpu_seam.json")),
                 ("results/prereg04/rank/frozen_config.json", t4.FROZEN_CONFIG),
                 ("results/test08/p1/gpu_seam.json", gpu_seam_path),
                 ("results/test08/rank/seam_val/seam_local.json", seam_local_path)]
        items += [(r, path_of(r)) for r in t7.P07_PINS]
        items += [(r, path_of(r)) for r in P08_PINS]
        for s in SETS:
            items += [(f"results/test07/hs/{s}_hs.npz", os.path.join(R7, "hs", f"{s}_hs.npz")),
                      (f"results/test08/p1/{s}_p1.npz", os.path.join(R8, "p1", f"{s}_p1.npz")),
                      (f"results/test08/p1/p1_{s}_report.json", os.path.join(R8, "p1", f"p1_{s}_report.json"))]
        for s in Q_SETS:
            for f in ("val_keys.npz", "val_%s.npz" % t4.C5_DUMP):
                items.append((f"results/test02/rank/{s}/dumps/{f}", os.path.join(D2, "rank", s, "dumps", f)))
            items.append((f"results/test04/opp/o4_{s}.npz", os.path.join(R4, "opp", f"o4_{s}.npz")))
        out, paths = {}, {}
        for key, p in items:
            if os.path.exists(p):
                out[key] = pin_sha(p)
                paths[key] = shown(p)
        return {"sha256": out, "paths": paths}


def q08_core(fz, vk, hq, lp, p1_ok, usable, fc08):
    """Q08 of one set from fuse_set's E terms and ported m / S (fz), q02h_set's E_HS (hq), lp_P1TO (unscaled) and the
    usable games: masked E_P1 of 4 lp_P1TO (once), the frozen theta; the direction rho against Q02 (fz['P'])."""
    S, ok, n = fz["S"], fz["ok"], fz["n"]
    E_p1, e_info = r8.p1_term(S, lp, usable)
    Es = [fz["E"][t] for t in r8.Q02_TERMS] + [hq["E_hs"], E_p1]
    P = r8.fuse_probs(fc08["params"]["theta"], fz["m"], Es, n, ok)
    has = pf7.questions_with_usable(S, usable) & ok
    rho, rho_n = pf7.hs_rho(E_p1, fz["P"], has)
    rows = []
    for k, (q, rws) in enumerate(vk["groups"]):
        probs = [repr(float(v)) for v in P[k]] if ok[k] else [""] * rf.N_LEVELS
        rows.append([q, len(rws), int(n[k])] + probs)
    pred_h = rf.rule_b_np(np.where(ok[:, None], hq["P"], 0.0))
    pred_8 = rf.rule_b_np(np.where(ok[:, None], P, 0.0))
    info = dict(fz["info"], masked_E_P1=e_info, unusable_p1_games=int((~usable).sum()),
                p1_not_ok=int((~np.asarray(p1_ok, dtype=bool)).sum()),
                questions_without_usable_p1_game=int((ok & ~has).sum()),
                rule_B_differs_from_Q02H=int(((pred_8 != pred_h) & ok).sum()))
    return {"P": P, "ok": ok, "rows": rows, "rho": rho, "rho_n": rho_n, "info": info, "E_p1": E_p1,
            "lp": lp, "p1_ok": np.asarray(p1_ok, dtype=bool), "usable": usable}


def q08_set(fz, vk, hq, vp, model08, fc08, F):
    """Q08 of a test set from its P1 means (vp: means float64, counts, n_moves, failed) with the frozen map
    (p08_rank_fuse.p1_lp_from_map, as the validation apply)."""
    X, us = F.variant_X(vp, "TO")
    lp = np.zeros((vk["n"], rf.N_LEVELS))
    if us.any():
        lp[us] = r8.p1_lp_from_map(F, model08, X[us])
    del X
    usable = np.asarray(vk["parsed"], dtype=bool) & us
    r = q08_core(fz, vk, hq, lp, us, usable, fc08)
    r["fail_count"] = int(pf7.hs_failures(vp["failed"], vp["n_moves"]).sum())
    r["info"]["p1_failed"] = int(np.asarray(vp["failed"]).sum())
    r["info"]["p1_failed_or_no_move"] = r["fail_count"]
    return r


def cmd_apply(a):
    t0 = time.time()
    ctx = resolve(a, "apply")
    run = Apply(ctx, a, t0)
    try:
        return run.go()
    except T08Error as e:
        if e.code in (3, 4):
            run.write_verdict(e)
            if e.code == 3:
                print(f"08 不用（PREREG_08 5.2）：{e}\ncommit {shown(ctx['R8'])}；最終照 PREREG_07 5.4（不重跑）",
                      file=sys.stderr)
        raise


# ================================================================ status (read only)
def cmd_status(a):
    ctx = resolve(a, "status")
    lines = [f"test08 status ({ctx['mode']}; {ctx['R8']})"]
    for s in SETS:
        d = os.path.join(ctx["H"], s)
        n = len(chunk_paths(d)) if os.path.isdir(d) else 0
        stop = os.path.exists(os.path.join(d, "STOP"))
        stale = sorted(os.path.basename(x) for x in glob.glob(os.path.join(ctx["H"], f"{s}_stale_*")))
        rp = os.path.join(ctx["p1"], f"p1_{s}_report.json")
        r = read_json_or_none(rp)
        lines.append(f"p1 {s}: {n} chunk(s) in {shown(d)}" + (" STOP file present" if stop else "")
                     + (f"; stale folders {stale}" if stale else "")
                     + (f"; report: {r.get('games')} games, failed {r.get('failed_or_no_move')}, "
                        f"{r.get('positions_per_s')} positions/s" if r else "; no report yet")
                     + (f", committed {committed_clean(rp)}" if r and ctx["real"] else ""))
    rp = os.path.join(ctx["rank"], "t08_apply_report.json")
    vp = os.path.join(ctx["rank"], "attempt_verdict.json")
    np_ = os.path.join(ctx["R8"], "not_used.json")
    if os.path.exists(rp):
        r = read_json(rp)
        lines.append(f"apply: exit {r.get('exit_code')}; submission sha256 "
                     f"{str((r.get('outputs_sha256') or {}).get('submission_rank.csv'))[:12]}; GPU seam "
                     f"{(r.get('gpu_seam') or {}).get('passed')}; local seam {(r.get('seam_local') or {}).get('verdict')}")
    elif os.path.exists(np_):
        r = read_json(np_)
        lines.append(f"08 not used: {r.get('stage')} ({r.get('reason')})")
    elif os.path.exists(vp):
        r = read_json(vp)
        lines.append(f"apply: attempt_verdict exit {r.get('exit_code')} at {r.get('stage')} ({r.get('reason')})")
    else:
        lines.append("apply: not run")
    if ctx["R7"]:
        r7 = read_json_or_none(os.path.join(ctx["R7"], "rank", "t07_apply_report.json"))
        lines.append("test07 apply: " + (f"exit {r7.get('exit_code')}, usable {r7.get('usable')}" if r7 else "not run"))
    left = K6_08 - now()
    lines.append(f"K6-08 {K6_08.isoformat()}: " + (f"{left.days} days {left.seconds // 3600} h left"
                                                   if left.total_seconds() > 0 else "passed"))
    print("\n".join(lines))
    return 0


# ================================================================ command line
def build_parser():
    ap = argparse.ArgumentParser(prog="t08_rank.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("p1", "apply", "status"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--rehearsal", action="store_true")
        for d in ("--results-dir", "--p1-data-dir", "--t07-results-dir", "--t04-results-dir", "--t02-results-dir",
                  "--t02-out-dir", "--tests-dir", "--checks-dir"):
            p.add_argument(d)
        p.add_argument("--r20-uploaded-at", help="drills only: R20's first upload time")
        p.add_argument("--now", help="drills only: the current time (K6-08)")
        if name in ("p1", "apply"):
            p.add_argument("--rerun", action="store_true")
            p.add_argument("--reason")
        if name == "p1":
            p.add_argument("--set", required=True, choices=SETS)
            p.add_argument("--workers", type=int, default=r8.RUN["workers"])
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
        return {"p1": cmd_p1, "apply": cmd_apply, "status": cmd_status}[a.cmd](a)
    except T08Error as e:
        print(f"{'REFUSED' if e.code == 2 else 'NOT USED' if e.code == 3 else 'DRIFT'} (exit {e.code}): {e}",
              file=sys.stderr)
        return e.code
    except Exception:  # noqa: BLE001 - an internal error is a refusal with its traceback, never a decision
        traceback.print_exc()
        print("ERROR (exit 2): unexpected internal error of tools/t08_rank.py; stop and check the local tool",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
