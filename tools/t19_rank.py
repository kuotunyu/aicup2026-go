"""C10 test19, competition 2 (rank, TEAM_11248): the 19 test-day local tool (Q19 = Q09 + 4 x SET, the question-level
set model S19 as a 7th fusion term).

Registration: docs/PREREG_19_SET_MODEL.md section 4 (rev 1) and the test-day revision (rev 3). CPU only, after 09's
apply: 19 reads 08's P1 chunks (the ones 09 read) and never runs the GPU. tools/t09_rank.py (and through it t08 / t07 /
t04) is imported, never edited.

    PY tools/t19_rank.py apply  [MODE] [DIRS] [--rerun --reason TEXT [--bug-revision N]]   (CPU, .venv)
    PY tools/t19_rank.py status [MODE] [DIRS]                                              (read only)
    (PY = .venv/Scripts/python.exe; Git Bash; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL thread variables)

MODE  none = real; at most one of --rehearsal, --dry-run
DIRS  --results-dir R19     test19 root (writes R19/set_seam.json, R19/rank; real results/test19)
      --t09-results-dir R9  test09 root (reads R9/gpu_seam.json, R9/rank; real results/test09)
      --t08-results-dir R8, --p1-data-dir H, --t07-results-dir R7, --t04-results-dir R4, --t02-results-dir D2,
      --t02-out-dir O2, --tests-dir T, --checks-dir C   as tools/t09_rank.py
Real mode: every directory is the repository path. --rehearsal: R19 is results/test19_rehearsal (or outside the
repository); --dry-run: R19 outside the repository. The other trees are required outside the real mode.
Drills only (never real): --r20-uploaded-at ISO, --now ISO.

apply  Q19 on public and private (PREREG_19 4.2), in this order (the first failing item decides):
       prerequisites: git clean (real); no BLAS thread variable; the committed prereg19 files equal their pinned
       SHA-256s and record the adopted Q19 (frozen_config.json Q19, apply_report.json usable true and finalized, the
       audit committed; else 3); .venv python / numpy / scipy = frozen_config.json (else 4); K6-19; every prerequisite
       of tools/t09_rank.py apply (and through it t08 / t07 / t04 / test02, R20's first upload, T1, H7R, H8R, the 09
       adoption, 08's apply); 09's t09_apply_report.json of this mode exit 0, usable and committed (09 judged 'not
       used' -> 19 not used, 3), every input 09 bound and 09's outputs unchanged.
       1 the three P1 sets of 08 (tools/t09_rank.py Apply.p1_sets: reports, chunk SHA-256s, per-position arrays).
       2 Q09 recomputed with 09's frozen map on test02 / 04 / 07's inputs equal to 09's committed rank_test_probs.csv
         (probabilities within 1e-12, rule A and rule B levels equal; byte equality recorded); lp_P1TO_E within 1e-12
         and p1_ok / usable equal to 09's test_lp_p1e.npz; else 4.
       3 the SET GPU seam: the 400 seam games against the validation chunks 0-1, each game as a one-game question,
         largest lp_SET difference -> R19/set_seam.json (recorded only, PREREG_19 4.2 step 2).
       4 the local Q19 seam on the 1,589 validation questions (09's seam_val inputs + the committed val_lp_p1e.npz and
         val_set.npz) against results/prereg19/rank/rank_val_probs.csv / pred_B.csv (09's seam_local.json
         byte-identical -> byte-identical required, else 4; otherwise the tolerance of test04, at most 4 rule-B
         differences, outside it -> 3).
       5 the SET direction check per set (rho >= max(0.5 x rho_mono_19, 0.155)) -> 3.
       Then rank_test_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json, submission_rank.csv (= pred_B.csv),
       test_lp_set.npz, check_submission --official (else 4) and R19/rank/t19_apply_report.json.
status Read-only summary.

Exit codes (as tools/t09_rank.py): 0 done; 2 refused or error; 3 the registered "19 is not used" (R19/not_used.json;
commit; the final choice follows PREREG_09 5.3; never rerun); 4 local drift or mismatch (R19/rank/attempt_verdict.json;
commit, stop and find out why, then --rerun --reason). After exit 3 or a written t19_apply_report.json nothing is
rerun except with --bug-revision N (a committed PREREG_19 rev N >= 4 added after that record).

Discipline: never imports torch, u1_lib, p09_map, p19_set or a player module; never fits anything (the frozen theta,
the frozen S19 model files, numpy float64 forward); no validation label file is ever opened; tools/score.py is never
run; temporary files only under data/test19/tmp (real), data/test19_rehearsal/tmp or R19/tmp. Competition 1 is never
read (PREREG_19 6).
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SELF = os.path.join(HERE, "t19_rank.py")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t09_rank as t9  # noqa: E402
import p19_forward as SF  # noqa: E402

t8, t7, t4, pf7, r8, r9, FW = t9.t8, t9.t7, t9.t4, t9.pf7, t9.r8, t9.r9, t9.FW
rf, t2 = t9.rf, t9.t2
decide, check_submission = t9.decide, t9.check_submission

# ================================================================ paths (module constants; the unit tests patch them)
RESULTS = os.path.join(REPO, "results")
REAL_R19 = os.path.join(RESULTS, "test19")
REHEARSAL_R19 = os.path.join(RESULTS, "test19_rehearsal")
REAL_TMP = os.path.join(REPO, "data", "test19", "tmp")
REHEARSAL_TMP = os.path.join(REPO, "data", "test19_rehearsal", "tmp")
REAL_R9 = t9.REAL_R9
REHEARSAL_R9 = t9.REHEARSAL_R9
P19_RANK = os.path.join(RESULTS, "prereg19", "rank")
P19_DOC = os.path.join(REPO, "docs", "PREREG_19_SET_MODEL.md")
P19_DOC_REL = "docs/PREREG_19_SET_MODEL.md"

# ================================================================ registered values
# the committed prereg19 files the test day reads (SHA-256: .json / .csv with CRLF folded to LF, .npz raw bytes)
P19_PINS = {
    "results/prereg19/rank/frozen_config.json": "1ba68c5af4571b20dd98ab2d6207462ea9b97d3557b4985ee4684263aeca6644",
    "results/prereg19/rank/s19_seed0.npz": SF.MODEL_SHA256[0],
    "results/prereg19/rank/s19_seed1.npz": SF.MODEL_SHA256[1],
    "results/prereg19/rank/s19_seed2.npz": SF.MODEL_SHA256[2],
    "results/prereg19/rank/rank_val_probs.csv": "9833ad333deb5bfcf9097e32e98b5d661f4f61b2f9a16ee0a0e68aba81c30790",
    "results/prereg19/rank/pred_A.csv": "70580eab800124bbe5d153767956f9c68178e2d6d67a64477f6af811af751ecc",
    "results/prereg19/rank/pred_B.csv": "e48b341bfa1de3068b8a283750f32d309ce9def885f519b3934befc44e147942",
    "results/prereg19/rank/val_set.npz": "28c82ef51cb58d6f4f4d835bba312ce54988849799ba687609571163cbe55a1e",
    "results/prereg19/rank/apply_report.json": "5f4eafa990dd88ae51aefd6355dc204c2bd4f17336e92b92a189c0ca97b25510",
    "results/prereg19/rank/apply_audit.json": "a58a7fb8751ac1b8a54726aacd050ce76d3c544e028e89f10b51219d0535b013"}
TERMS = ["OP", "O4", "SWA", "SWB", "HS", "P1TO", "SET"]
COMBO = {"id": "Q19", "sources": ["C5", "OP", "O4", "SWA", "SWB", "HS", "P1TO_E", "SET"], "terms": TERMS,
         "p1_scale": 4.0, "set_scale": 4.0, "p1_term": "P1TO_E (ens3fam, prereg09)",
         "set_term": "SET (S19, 3 seeds, prereg19)"}
METHOD = "prereg19 Q19 = Q09 + 4 x SET (S19 set model), rule B (甲)"
RHO_MONO = 0.9585505165633248                                   # frozen_config.json direction_ref.rho_mono (PREREG_19 4.2 step 5)
RHO_EVAL = 0.9543009493356425                                   # frozen_config.json direction_ref.rho_eval (seam_val record)
RHO_FLOOR = r9.RHO_FLOOR_TEST                     # 0.155: one-sided p = 0.001 for 400 questions
TOL_Q09 = 1e-12                                   # PREREG_19 4.2 step 1: Q09 recomputed vs 09's file
LP_REVIEW = t9.LP_REVIEW
TW = t9.TW
K6_19 = dt.datetime(2026, 11, 19, 23, 59, 59, tzinfo=TW)
P19_MIN_REV = 4                                   # rev 3 registers the test day; an override is rev 4 or later
SETS, TEST_SETS, Q_SETS = t9.SETS, t9.TEST_SETS, t9.Q_SETS
FORMATS = {"set_seam": "c10-test19 set seam v1", "seam_local": "c10-test19 seam_local v1",
           "apply": "c10-test19 rank apply v1", "verdict": "c10-test19 attempt verdict v1",
           "not_used": "c10-test19 not used v1", "lp": "c10-test19 test_lp_set v1"}
TEST_OUTPUTS = ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv",
                "test_lp_set.npz")
T09_OUTPUTS = t9.TEST_OUTPUTS
_LABEL = "ans" + "wer"


class T19Error(RuntimeError):
    """A stop with its exit code: 2 refused, 3 registered 'no 19', 4 local drift / mismatch."""

    def __init__(self, message, code=2, stage=None, reason=None, details=None):
        super().__init__(message)
        self.code, self.stage, self.reason, self.details = code, stage, reason, details


def refuse(msg, reason=None):
    raise T19Error(msg, 2, reason=reason)


def not_used(msg, stage, reason, details=None):
    raise T19Error(msg, 3, stage, reason, details)


def drift(msg, stage, reason, details=None):
    raise T19Error(msg, 4, stage, reason, details)


FOREIGN_ERRORS = tuple(dict.fromkeys(t9.FOREIGN_ERRORS + (SF.SetForwardError,)))


def t9call(fn, *args):
    """A tools/t09_rank.py function (and, through it, t08 / t07 / t04): its exit code is ours (2 refused, 3 registered
    'no 09 / 08 / 07 / 04' = no 19, 4 drift)."""
    try:
        return fn(*args)
    except t9.T09Error as e:
        raise T19Error(f"(t09_rank) {e}", e.code, e.stage, e.reason or "t09_prerequisite", e.details)
    except FOREIGN_ERRORS as e:
        refuse(f"(t09_rank) {e}")


# ================================================================ small helpers
def now():
    """The current time (a module function: the tests replace it to cross K6-19)."""
    return dt.datetime.now(TW)


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_iso(text, what):
    try:
        t = dt.datetime.fromisoformat(str(text).strip())
    except ValueError:
        refuse(f"{what} {text!r} is not an ISO time")
    return t.replace(tzinfo=TW) if t.tzinfo is None else t


def refuse_label_path(path):
    if path is not None and _LABEL in os.path.abspath(os.fspath(path)).lower():
        refuse(f"refusing {path}: validation label files are never opened by the test19 tools")
    return path


def sha_raw(path):
    return t2.sha256_file(refuse_label_path(path))


def sha_lf(path):
    return t2.sha256_text(refuse_label_path(path))


def pin_sha(path):
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


def git(*args):
    return t4.git(*args)


def git_head():
    rc, out = git("rev-parse", "HEAD")
    return out if rc == 0 else None


def path_of(rel):
    return os.path.join(REPO, *rel.split("/"))


def committed_clean(path):
    return t9.committed_clean(path)


def require_committed(paths, what):
    bad = [shown(p) for p in paths if not committed_clean(p)]
    if bad:
        refuse(f"{what}: commit first ({bad[:6]} not committed or changed)")
    return {"committed": [shown(p) for p in paths]}


def commit_time(path):
    return t9.commit_time(path)


def tool_files():
    names = ["t19_rank.py", "p19_forward.py", "t09_rank.py", "t08_rank.py", "t07_rank.py", "t04_rank.py",
             "p09_rank_fuse.py", "p09_forward.py", "p08_rank_fuse.py", "p08_p1.py", "p07_rank_fuse.py", "p07_hs.py",
             "check_submission.py", "official_log.py", "t02_rank.py", "t02_test_bundle.py", "p04_opp.py",
             "p04_rank_fuse.py", "p03_rank_fuse.py", "p02_rank_fuse.py", "c10data.py", "decide.py"]
    return {n: (sha_lf(os.path.join(HERE, n)) if os.path.exists(os.path.join(HERE, n)) else None) for n in names}


def tool_record():
    return {"git_head": git_head(), "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": tool_files()}


def is_num(v):
    return t9.is_num(v)


def blas_env_check():
    set_ = {k: os.environ[k] for k in r8.BLAS_ENV if os.environ.get(k)}
    if set_:
        refuse(f"BLAS thread variables are set ({set_}): unset them (the registered byte seams use the default "
               f"OpenBLAS threads; PREREG_19 2.1)")
    return {k: None for k in r8.BLAS_ENV}


# ================================================================ modes and directories
def resolve(a, cmd):
    """t09_rank.resolve for the 09 / 08 / 07 / 04 / test02 trees (as t09 sees them) plus the test19 root."""
    flags = [m for m, on in (("dry-run", a.dry_run), ("rehearsal", a.rehearsal)) if on]
    if len(flags) > 1:
        refuse(f"give at most one of --dry-run, --rehearsal (got {flags})")
    mode = flags[0] if flags else "real"
    if mode == "real":
        for k, v, want in (("--results-dir", a.results_dir, REAL_R19), ("--t09-results-dir", a.t09_results_dir,
                                                                        REAL_R9)):
            if v and not t2.same_path(v, want):
                refuse(f"real mode: {k} must be {want} (got {v})")
        R19, R9, TMP = REAL_R19, REAL_R9, REAL_TMP
    else:
        if mode == "rehearsal" and not a.results_dir:
            a.results_dir = REHEARSAL_R19
        if not a.results_dir:
            refuse(f"--{mode} needs --results-dir (other modes never fall back to the repository paths)")
        R19 = os.path.abspath(a.results_dir)
        in_repo_ok = mode == "rehearsal" and t2.same_path(R19, REHEARSAL_R19)
        if t2.is_inside(R19, REPO) and not in_repo_ok:
            refuse(f"--{mode}: --results-dir must be outside the repository (only --rehearsal may use "
                   f"{REHEARSAL_R19}); got {R19}")
        if not a.t09_results_dir and cmd != "status":
            refuse(f"--{mode} needs --t09-results-dir")
        R9 = os.path.abspath(a.t09_results_dir) if a.t09_results_dir else None
        if R9 and t2.is_inside(R9, REPO) and not t2.same_path(R9, REHEARSAL_R9):
            refuse(f"--{mode}: --t09-results-dir inside the repository must be {REHEARSAL_R9} (got {R9})")
        TMP = REHEARSAL_TMP if in_repo_ok else os.path.join(R19, "tmp")
    a9 = argparse.Namespace(**vars(a))
    a9.results_dir = R9
    if R9 is None:                                  # status outside the real mode without a test09 tree
        ctx = {"R9": None}
    else:
        try:
            ctx = t9.resolve(a9, cmd)
        except t9.T09Error as e:
            raise T19Error(f"(t09_rank) {e}", e.code)
    for p in (R19,):
        refuse_label_path(p)
        if pf7.under_data_local(p):
            refuse(f"{p}: never under data/local")
    bad = forbidden_area(R19)
    if bad:
        refuse(f"output folder inside the read-only {bad}")
    ctx.update(mode=mode, real=mode == "real", cmd=cmd, R19=R19, rank19=os.path.join(R19, "rank"), TMP19=TMP,
               write_roots19=[R19, TMP])
    return ctx


def forbidden_area(path):
    """Where t19 never writes: results/prereg*, results/test02 ... test14, data/local, data/test0*, the ledger."""
    p = os.path.abspath(path)
    if pf7.under_data_local(p):
        return "data/local"
    if t2.same_path(p, os.path.join(RESULTS, "score_ledger.csv")):
        return "results/score_ledger.csv"
    if t2.is_inside(p, RESULTS) and not t2.same_path(p, RESULTS):
        first = os.path.relpath(os.path.realpath(p), os.path.realpath(RESULTS)).split(os.sep)[0].lower()
        if first.startswith(("prereg", "test02", "test04", "test06", "test07", "test08", "test09", "test10",
                             "test11", "test14")):
            return "results/" + first
    if t2.is_inside(p, os.path.join(REPO, "data")):
        first = os.path.relpath(os.path.realpath(p), os.path.realpath(os.path.join(REPO, "data"))).split(os.sep)[0]
        if first.lower().startswith(("test0", "prereg")):
            return "data/" + first
    return None


def guard_write(ctx, path):
    p = os.path.abspath(path)
    refuse_label_path(p)
    bad = forbidden_area(p)
    if bad:
        refuse(f"refusing to write {p}: it is inside the read-only {bad}")
    if not any(t2.is_inside(p, r) for r in ctx["write_roots19"]):
        refuse(f"refusing to write {p}: outside {ctx['write_roots19']}")
    return p


def write_json(ctx, path, obj):
    guard_write(ctx, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(t4.jsonable(obj), f, indent=1, ensure_ascii=False, allow_nan=False)
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
    if not ctx["real"]:
        rec["git"] = f"not checked ({ctx['mode']})"
        return
    rc, st = git("status", "--porcelain", "--untracked-files=no")
    if rc != 0:
        refuse("git is not available")
    if st:
        refuse(f"commit or restore the tracked changes first (git status: {st.splitlines()[:6]})")
    own = [SELF, os.path.join(HERE, "p19_forward.py"), os.path.join(HERE, "t09_rank.py"),
           os.path.join(HERE, "official_log.py"), os.path.join(HERE, "check_submission.py")]
    rec["git"] = {"clean": True, "head": git_head(), "own_files": require_committed(own, "the test19 tools")}


def check_p19_pins():
    """The committed prereg19 files equal their pinned SHA-256s; frozen_config.json is the frozen Q19 (terms, scales,
    models, direction reference); the validation apply is usable and finalized (else 3). -> (frozen config, models,
    record)."""
    got = {r_: (pin_sha(path_of(r_)) if os.path.exists(path_of(r_)) else None) for r_ in P19_PINS}
    bad = sorted(r_ for r_ in P19_PINS if got[r_] != P19_PINS[r_])
    if bad:                              # checked before the adoption flag is read (PREREG_19 rev 3 review)
        refuse(f"prereg19 files differ from the pinned SHA-256s: {bad} (never edited after the adoption)")
    rp = path_of("results/prereg19/rank/apply_report.json")
    rep = read_json_or_none(rp)
    if rep is None or rep.get("usable") is not True or not rep.get("finalized") or rep.get("combo") != "Q19":
        not_used("PREREG_19 2.2: the validation apply of Q19 is not usable / finalized: 19 is not used",
                 "prerequisites", "p19_not_adopted")
    fc = read_json(path_of("results/prereg19/rank/frozen_config.json"))
    combo = fc.get("combo") or {}
    dref = fc.get("direction_ref") or {}
    if fc.get("format") != "c10-prereg19 rank frozen_config v1" or combo.get("id") != "Q19" or \
            combo.get("terms") != TERMS or float(combo.get("p1_scale", 0)) != 4.0 or \
            float(combo.get("set_scale", 0)) != 4.0 or len((fc.get("params") or {}).get("theta") or []) != 18:
        refuse("results/prereg19/rank/frozen_config.json is not the frozen Q19 configuration")
    if float(dref.get("rho_mono")) != RHO_MONO or float(dref.get("rho_eval")) != RHO_EVAL:
        refuse("frozen_config.json direction_ref differs from the registered values")
    if (fc.get("p1e_map") or {}).get("sha256") != t9.MAP_SHA:
        refuse("frozen_config.json does not name 09's p1e_map.npz")
    try:
        models = SF.load_models(P19_RANK)
    except FOREIGN_ERRORS as e:
        refuse(str(e))
    return fc, models, {"prereg19_sha256": got, "adopted": {"apply_usable": True, "finalized": rep.get("finalized")}}


def check_venv(fc19):
    want = {k: ((fc19.get("environment") or {}).get("versions") or {}).get(k) for k in ("python", "numpy", "scipy")}
    got = r9.venv_versions(False)
    bad = {k: {"now": got.get(k), "frozen": want[k]} for k in want if got.get(k) != want[k]}
    if bad:
        drift(f".venv differs from results/prereg19/rank/frozen_config.json {bad}: restore .venv (never upgraded "
              f"before 11/24)", "prerequisites", "venv_versions", bad)
    return {"equal": True, "versions": got}


def earlier_success(ctx):
    p = os.path.join(ctx["rank19"], "t19_apply_report.json")
    if not os.path.exists(p):
        return False, None
    rep = read_json(p)
    if rep.get("exit_code") != 0:
        return False, None
    if not ctx["real"]:
        return True, {"note": f"{ctx['mode']}: commit time not checked"}
    h, t = commit_time(p)
    ok = t is not None and parse_iso(t, "commit time") <= K6_19
    return ok, {"commit": h, "committed_at": t}


def check_k6(ctx, a):
    t_now = parse_iso(a.now, "--now") if getattr(a, "now", None) else now()
    if t_now <= K6_19:
        return {"now": t_now.isoformat(), "k6_19": K6_19.isoformat(), "passed": False}
    ok, rec = earlier_success(ctx)
    if ok:
        refuse(f"K6-19 {K6_19.isoformat()} has passed; the successful run before it ({rec}) stays valid: nothing is "
               f"rerun", reason="k6_passed_success")
    not_used(f"K6-19 {K6_19.isoformat()} has passed without a successful official 19 run: 19 is not used this "
             f"season (PREREG_19 5); the final choice follows PREREG_09 5.3", "prerequisites", "k6_19")


def t09_state(ctx):
    """test09's apply of this mode: exit 0 and usable (committed in the real mode), every input it bound and its
    outputs unchanged. 09 judged 'not used' -> 3 (19 is not used either); not done yet -> 2. -> (report, record)."""
    R9 = ctx["R9"]
    rp, vp = os.path.join(ctx["rank9"], "t09_apply_report.json"), os.path.join(ctx["rank9"], "attempt_verdict.json")
    nu = read_json_or_none(os.path.join(R9, "not_used.json"))
    if nu is not None and nu.get("exit_code") == 3 and nu.get("mode") == ctx["mode"]:
        not_used(f"test09 judged '09 is not used' ({nu.get('stage')}: {nu.get('reason')}): 19 is not used either "
                 f"(PREREG_19 4.2)", "prerequisites", "t09_not_used")
    if not os.path.exists(rp):
        refuse("test09 apply has not finished (no t09_apply_report.json"
               + (f"; attempt_verdict.json says exit {read_json(vp).get('exit_code')}" if os.path.exists(vp) else "")
               + "): run docs/TESTDAY_09.md step 3 first", reason="t09_not_done")
    rep = read_json(rp)
    if rep.get("format") != t9.FORMATS["apply"] or rep.get("mode") != ctx["mode"]:
        refuse(f"{shown(rp)}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (this run: {ctx['mode']})")
    if rep.get("exit_code") == 3 or rep.get("usable") is False:
        not_used("test09 apply judged '09 is not used': 19 is not used either", "prerequisites", "t09_not_used")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True or (rep.get("combo") or {}).get("id") != "Q09":
        refuse(f"{shown(rp)}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    files = [rp] + [os.path.join(ctx["rank9"], n) for n in T09_OUTPUTS] + \
        [os.path.join(ctx["rank9"], "seam_val", "seam_local.json"), os.path.join(R9, "gpu_seam.json")]
    if ctx["real"]:
        require_committed(files, "results/test09")
    missing = [shown(p) for p in files if not os.path.exists(p)]
    if missing:
        refuse(f"test09 outputs missing: {missing[:5]}")
    b9 = t9.bound_problems(ctx, rep.get("inputs_bound") or {}, {"results/test09/": R9})
    if b9:
        refuse(f"test09's bound inputs changed since its apply: {b9[:5]} (follow docs/TESTDAY_09.md first)",
               reason="t09_inputs_changed")
    want = rep.get("outputs_sha256") or {}
    changed = [n for n in T09_OUTPUTS if sha_raw(os.path.join(ctx["rank9"], n)) != want.get(n)]
    if changed:
        refuse(f"test09 outputs differ from its report: {changed}", reason="t09_outputs_changed")
    return rep, {"path": shown(rp), "sha256": pin_sha(rp), "exit_code": 0, "usable": True,
                 "inputs_bound_checked": len(rep.get("inputs_bound") or {}),
                 "submission_sha256": want.get("submission_rank.csv")}


def prereqs(ctx, a):
    """PREREG_19 4.2 in this order. -> (fc19, models, t09 prerequisites tuple, rep9, record)."""
    rec = {}
    check_git_clean(ctx, rec)                                            # 1
    rec["blas_env"] = blas_env_check()                                   # 2
    fc19, models, rec["prereg19"] = check_p19_pins()                     # 3 (19 adopted)
    rec["venv"] = check_venv(fc19)
    rec["k6_19"] = check_k6(ctx, a)                                      # 4
    p9 = t9call(t9.prereqs, ctx, a)                                      # 5 (09 / 08 / 07 / 04 / test02)
    rec["t09_prerequisites"] = p9[-1]
    rep9, rec["t09_apply_report"] = t09_state(ctx)                       # 6
    return fc19, models, p9[:-1], rep9, rec


# ================================================================ the SET term of a set and the seams (pure parts)
def set_term_of(fz, vk, p1s, usable, models, mp):
    """E_SET of one set's questions (frozen S19, 09's mu_x / sd_x; usable = C5 parsed & P1 usable)."""
    return SF.question_set_term(models, p1s["means"], p1s["counts"], usable, fz["S"], mp["mu_x"], mp["sd_x"])


def q19_core(fz, vk, hq, q9, E_set, has_set, fc19):
    """Q19 of one set: the Q02 terms and m / S of fz, E_HS of q02h_set (hq), E_P1E of Q09's q08_core (q9), E_SET, the
    frozen theta; the SET direction rho against Q02 (fz['P'])."""
    S, ok, n = fz["S"], fz["ok"], fz["n"]
    Es = [fz["E"][t] for t in r8.Q02_TERMS] + [hq["E_hs"], q9["E_p1"], E_set]
    P = r8.fuse_probs(fc19["params"]["theta"], fz["m"], Es, n, ok)
    has = np.asarray(has_set, dtype=bool) & ok
    rho, rho_n = pf7.hs_rho(E_set, fz["P"], has)
    rows = []
    for k, (q, rws) in enumerate(vk["groups"]):
        probs = [repr(float(v)) for v in P[k]] if ok[k] else [""] * rf.N_LEVELS
        rows.append([q, len(rws), int(n[k])] + probs)
    pred_9 = rf.rule_b_np(np.where(ok[:, None], q9["P"], 0.0))
    pred_19 = rf.rule_b_np(np.where(ok[:, None], P, 0.0))
    info = {"questions": int(len(ok)), "questions_without_parsed_game": int((~ok).sum()),
            "questions_without_set_game": int((ok & ~has).sum()),
            "rule_B_differs_from_Q09": int(((pred_19 != pred_9) & ok).sum())}
    return {"P": P, "ok": ok, "rows": rows, "rho": rho, "rho_n": rho_n, "info": info, "E_set": E_set, "has": has}


def set_seam_record(seam, val, models, mp):
    """PREREG_19 4.2 step 2 (record only): the 400 seam games against the validation chunks 0-1, each game as a
    one-game question: the largest lp_SET difference over the games usable in both."""
    n = min(len(seam["failed"]), len(val["failed"]))
    ok = FW.usable_games(seam["failed"][:n], seam["counts"][:n]) & FW.usable_games(val["failed"][:n],
                                                                                    val["counts"][:n])
    rec = {"games": int(n), "usable_games": int(ok.sum()),
           "bitwise_equal_means": bool(np.array_equal(seam["means"][:n], val["means"][:n]))}
    if ok.any():
        la = SF.single_game_logp(models, FW.x_raw(seam["means"][:n][ok], seam["counts"][:n][ok]), mp["mu_x"],
                                 mp["sd_x"])
        lb = SF.single_game_logp(models, FW.x_raw(val["means"][:n][ok], val["counts"][:n][ok]), mp["mu_x"],
                                 mp["sd_x"])
        per = np.max(np.abs(la - lb), axis=1)
        rec.update(max_abs_lp_set=float(per.max()),
                   lp_set_quantiles={q: float(np.quantile(per, float(q))) for q in ("0.5", "0.9", "0.99")},
                   games_lp_above_review_line=int((per > LP_REVIEW).sum()),
                   argmax_equal=int((la.argmax(1) == lb.argmax(1)).sum()))
    else:
        rec.update(max_abs_lp_set=None)
    return dict(rec, judged=False, review_line=LP_REVIEW,
                note="recorded only (PREREG_19 4.2 step 2): 09's GPU seam on the same chunks is the judged one")


# ================================================================ apply
def apply_outputs(ctx):
    rank, seam = ctx["rank19"], os.path.join(ctx["rank19"], "seam_val")
    out = [os.path.join(seam, n) for n in ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json",
                                           "seam_local.json")]
    out += [os.path.join(rank, n) for n in TEST_OUTPUTS + ("t19_apply_report.json", "attempt_verdict.json")]
    out += [os.path.join(ctx["R19"], "set_seam.json"), os.path.join(ctx["R19"], "not_used.json")]
    return out


def check_bug_revision(n, overturned):
    """--bug-revision N: PREREG_19 rev N >= 4 committed and clean, its row at HEAD, added after every overturned
    record (ISO times). -> record."""
    if n is None:
        return None
    if n < P19_MIN_REV:
        refuse(f"--bug-revision {n}: an override needs PREREG_19 rev {P19_MIN_REV} or later")
    if not committed_clean(P19_DOC):
        refuse(f"{P19_DOC_REL} must be committed and clean")
    row = f"| rev {n} |"
    import subprocess
    r = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{P19_DOC_REL}"], capture_output=True, timeout=60)
    head = r.stdout if r.returncode == 0 else b""
    if not any(line.startswith(row) for line in head.decode("utf-8", "replace").splitlines()):
        refuse(f"{P19_DOC_REL} at HEAD has no row starting {row!r}")
    rc, out = git("log", "--reverse", "-S" + row, "--format=%H %cI", "--", P19_DOC_REL)
    if rc != 0 or not out:
        refuse(f"no commit added {row!r}")
    h, t = out.splitlines()[0].split()
    at = parse_iso(t, "commit time")
    early = [f"{w} {x}" for w, x in overturned if x and at <= parse_iso(x, w)]
    if early:
        refuse(f"PREREG_19 rev {n} ({h[:12]}, {t}) is not later than {early}")
    return {"rev": n, "commit": h, "committed_at": t, "overturns": [{"what": w, "time": x} for w, x in overturned]}


class Apply:
    def __init__(self, ctx, a, t0):
        self.ctx, self.a, self.t0 = ctx, a, t0
        self.stage, self.pre, self.replaced, self.bug, self.rec = "prerequisites", {}, {}, None, {}
        self.cleared = False

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
        rep = read_json_or_none(os.path.join(ctx["rank19"], "t19_apply_report.json"))
        nu = read_json_or_none(os.path.join(ctx["R19"], "not_used.json"))
        decided = [(n, r.get("created")) for n, r in (("t19_apply_report.json", rep), ("not_used.json", nu)) if r]
        if decided:
            if a.bug_revision is None:
                refuse(f"{[n for n, _ in decided]} hold the registered decision; a rerun needs a PREREG_19 revision "
                       f"naming a program error (--bug-revision N, N >= {P19_MIN_REV})")
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
        self.cleared = True

    def write_verdict(self, e):
        """Writes not_used.json (3) or attempt_verdict.json (4); never over or next to a recorded decision
        (t19_apply_report.json / not_used.json) unless this run passed the rerun gate (PREREG_19 rev 3 review)."""
        ctx = self.ctx
        decided = [p for p in (os.path.join(ctx["rank19"], "t19_apply_report.json"),
                               os.path.join(ctx["R19"], "not_used.json")) if os.path.exists(p)]
        if decided and not self.cleared:
            raise T19Error(f"{e} -- not recorded: {[shown(p) for p in decided]} already hold the registered decision "
                           f"and this run did not pass the rerun gate (nothing was written)", 2, e.stage, e.reason,
                           e.details)
        rec = {"format": FORMATS["verdict"], "mode": ctx["mode"], "created": now_iso(), "exit_code": e.code,
               "stage": e.stage or self.stage, "reason": e.reason or str(e).splitlines()[0][:200], "message": str(e),
               "details": e.details, "checks": self.pre, "records": self.rec,
               "rerun": {"reason": self.a.reason, "replaced": self.replaced, "bug_revision": self.bug},
               "tool": tool_record(), "seconds": round(time.time() - self.t0, 1)}
        if e.code == 3:
            write_json(ctx, os.path.join(ctx["R19"], "not_used.json"), dict(rec, format=FORMATS["not_used"]))
        else:
            write_json(ctx, os.path.join(ctx["rank19"], "attempt_verdict.json"), rec)

    def go(self):
        ctx, a = self.ctx, self.a
        existing = [p for p in apply_outputs(ctx) if os.path.exists(p)]
        if existing and not (a.rerun and a.reason):       # before the prerequisites (PREREG_19 rev 3 review)
            refuse(f"outputs already exist ({[os.path.basename(p) for p in existing][:6]}); never overwritten without "
                   f"--rerun --reason TEXT")
        fc19, models, p9, rep9, self.pre = prereqs(ctx, a)
        fc9, mp, fc07, fc08, model08, fc04, t02, rep7, rep4, rep8 = p9
        self.rerun_gate()
        # ---- step 1: 08's three P1 sets (09's reader and checks)
        self.stage = "p1_sets"
        a9 = t9.Apply(ctx, a, self.t0)
        try:
            p1 = a9.p1_sets(rep8)
        except t9.T09Error as e:
            raise T19Error(f"(t09_rank) {e}", e.code, "p1_sets", e.reason, e.details)
        self.rec["p1_sets"] = a9.rec["p1_sets"]
        for s in SETS:
            want = (rep9.get("p1_sets") or {}).get(s) or {}
            got = self.rec["p1_sets"][s]
            if (want.get("report_sha256"), want.get("npz_sha256")) != (got["report_sha256"], got["npz_sha256"]):
                refuse(f"{s}: 08's P1 set differs from the one 09's report records", reason="p1_sets")
        model07 = pf7.load_hs_map(path_of("results/prereg07/rank/hs_map.npz"))
        self.clear()
        # ---- step 2: Q09 recomputed (test02 / 04 / 07 inputs, 09's frozen map) equal to 09's rank_test_probs.csv
        self.stage = "q09_vs_09"
        vk = {s: rf.load_val_keys(os.path.join(ctx["D2"], "rank", s, "dumps")) for s in Q_SETS}
        sw = {s: t9.t8call(t7.load_sw, ctx, s, vk[s]) for s in Q_SETS}
        o4z = {s: load_npz(os.path.join(ctx["R4"], "opp", f"o4_{s}.npz")) for s in Q_SETS}
        c5w = fc04["source_sha256"]["C5"]["weights_sha256"]
        fz = {}
        for s in Q_SETS:     # every question-file read of this process happens here (seam_val: data/local)
            ddir = os.path.join(ctx["D2"], "rank", s, "dumps")
            c5 = t9.t8call(t7.t4call, t4.load_c5, s, ddir, vk[s], c5w)
            f = t9.t8call(t7.t4call, t4.fuse_set, s, vk[s], t02["qfiles"][s], c5, t02["opening"], o4z[s], sw[s],
                          fc04, False)
            f["c5_sha256"] = c5["sha256"]
            f.update(t7.port_c5(vk[s], c5, fc04, f))
            fz[s] = f
        if not all(fz[s]["port_equal"] for s in Q_SETS):
            drift("the ported C5 lines do not give fuse_set's Q02 bit for bit", "q09_vs_09", "port_c5",
                  {s: fz[s]["port_equal"] for s in Q_SETS})
        # ---- from here no data/local file is opened: pilot_fuse (hs_lib's guard)
        PF = pf7.import_pilot_fuse()
        hs = {s: load_npz(os.path.join(ctx["R7"], "hs", f"{s}_hs.npz")) for s in TEST_SETS}
        h, q9, us, E_set, lp_set, has_set = {}, {}, {}, {}, {}, {}
        for s in TEST_SETS:
            z = p1[s]["z"]
            keys_ok = (list(map(str, z["question_id"])) == vk[s]["question_id"]
                       and [int(x) for x in z["slot"]] == vk[s]["slot"] and list(map(str, z["color"])) == vk[s]["color"]
                       and list(map(str, hs[s]["question_id"])) == vk[s]["question_id"]
                       and [int(x) for x in hs[s]["slot"]] == vk[s]["slot"])
            if not keys_ok:
                refuse(f"{s}: 08's {s}_p1.npz / 07's {s}_hs.npz keys differ from test02's val_keys.npz", reason="keys")
            h[s] = t9.t8call(t7.q02h_set, fz[s], vk[s], hs[s], model07, fc07, PF)
            lpE, okE = t9.set_features(p1[s], mp)
            us[s] = np.asarray(vk[s]["parsed"], dtype=bool) & okE
            q9[s] = t8.q08_core(fz[s], vk[s], h[s], lpE, okE, us[s], fc9)
            E_set[s], lp_set[s], has_set[s] = set_term_of(fz[s], vk[s], p1[s], us[s], models, mp)
        cmp9 = t9.probs_rows_compare(q9["public"]["rows"] + q9["private"]["rows"],
                                     os.path.join(ctx["rank9"], "rank_test_probs.csv"))
        cmp9["t09_sha256"] = cmp9.pop("t08_sha256")
        lp9 = load_npz(os.path.join(ctx["rank9"], "test_lp_p1e.npz"))
        lp_all = np.concatenate([q9[s]["lp"] for s in TEST_SETS])
        cmp9.update(p1_ok_equal_t09=bool(np.array_equal(np.asarray(lp9["p1_ok"], dtype=bool),
                                                        np.concatenate([q9[s]["p1_ok"] for s in TEST_SETS]))),
                    usable_equal_t09=bool(np.array_equal(np.asarray(lp9["usable"], dtype=bool),
                                                         np.concatenate([us[s] for s in TEST_SETS]))),
                    lp_p1e_max_abs_diff_vs_t09=float(np.max(np.abs(np.asarray(lp9["lp"]) - lp_all)))
                    if len(lp_all) else 0.0)
        cmp9["lp_p1e_equal"] = bool(cmp9["lp_p1e_max_abs_diff_vs_t09"] <= TOL_Q09)
        self.rec["q09_vs_09"] = cmp9
        print(f"Q09 recomputed vs test09: byte equal {cmp9['byte_equal']}, max |dP| {cmp9['max_abs_diff']:.3g} "
              f"(<= {TOL_Q09}), rule A / B differences {cmp9['rule_a_differences']} / {cmp9['rule_b_differences']}, "
              f"usable equal {cmp9['p1_ok_equal_t09']} / {cmp9['usable_equal_t09']}, lp_P1TO_E max |d| "
              f"{cmp9['lp_p1e_max_abs_diff_vs_t09']:.3g}", flush=True)
        if not (cmp9["equal"] and cmp9["p1_ok_equal_t09"] and cmp9["usable_equal_t09"] and cmp9["lp_p1e_equal"]):
            drift("Q09 recomputed with test09's inputs differs from results/test09/rank/rank_test_probs.csv (or the "
                  "usable games / lp_P1TO_E differ; a local problem, PREREG_19 4.2 step 1)", "q09_vs_09",
                  "q09_vs_09", cmp9)
        # ---- step 3: the SET GPU seam (record only)
        self.stage = "set_seam"
        vrows, val = t9.val_seam_chunks()
        sseam = set_seam_record(p1["seam"], val, models, mp)
        sseam.update(format=FORMATS["set_seam"], mode=ctx["mode"], created=now_iso(),
                     seam_p1_report_sha256=self.rec["p1_sets"]["seam"]["report_sha256"],
                     reference={"val_chunks": vrows, "val_report.json": sha_lf(t9.VAL_REPORT)})
        set_seam_path = write_json(ctx, os.path.join(ctx["R19"], "set_seam.json"), sseam)
        self.rec["set_seam_record_only"] = {k: sseam.get(k) for k in ("max_abs_lp_set", "usable_games",
                                                                      "bitwise_equal_means",
                                                                      "games_lp_above_review_line")}
        print(f"SET GPU seam (record only): max |d lp_SET| {sseam.get('max_abs_lp_set')}, bitwise equal means "
              f"{sseam['bitwise_equal_means']}", flush=True)
        # ---- step 4: the local Q19 seam on the validation questions
        self.stage = "local_seam"
        vh = load_npz(path_of("results/prereg07/hs/val_hs.npz"))
        seam = self.local_seam(fz["seam_val"], vk["seam_val"], vh, model07, fc07, fc9, fc19, PF)
        # ---- step 5: the SET direction
        self.stage = "set_direction"
        res = {}
        for s in TEST_SETS:
            res[s] = q19_core(fz[s], vk[s], h[s], q9[s], E_set[s], has_set[s], fc19)
        direc = {s: pf7.direction_record(res[s]["rho"], res[s]["rho_n"], RHO_MONO, RHO_FLOOR) for s in TEST_SETS}
        direc["seam_val_record_only"] = seam["direction"]
        self.rec["set_direction"] = direc
        bad = [s for s in TEST_SETS if not direc[s]["passed"]]
        if bad:
            not_used(f"SET direction check failed on {bad} ({ {s: direc[s]['rho'] for s in TEST_SETS} } < "
                     f"{direc['public']['threshold']}): 19 is not used (PREREG_19 4.2 step 4)", "set_direction",
                     "set_direction", direc)
        # ---- the outputs
        self.stage = "outputs"
        rank = ctx["rank19"]
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
        save_npz(ctx, os.path.join(rank, "test_lp_set.npz"), format=np.array(FORMATS["lp"]),
                 question_id=np.array(fz["public"]["qids"] + fz["private"]["qids"]),
                 lp=np.concatenate([lp_set[s] for s in TEST_SETS]),
                 has=np.concatenate([res[s]["has"] for s in TEST_SETS]),
                 usable=np.concatenate([us[s] for s in TEST_SETS]), scale=np.array(SF.SCALE),
                 models_sha256=np.array([SF.MODEL_SHA256[x] for x in SF.SEEDS]), map_sha256=np.array(t9.MAP_SHA))
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
            drift(f"check_submission --official failed on the Q19 file: {errors[:5]}; {removed} removed",
                  "check_submission", "check_submission", {"errors": errors[:50], "removed": removed})
        # ---- the report
        preds = dict(t2.read_pred(sub))
        pub_ids = set(fz["public"]["qids"])
        diffs = {}
        for name, other in (("09", os.path.join(ctx["rank9"], "submission_rank.csv")),
                            ("08", os.path.join(ctx["rank"], "submission_rank.csv")),
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
        bound = self.inputs_bound(t02, set_seam_path, seam["path"])
        rep = {"format": FORMATS["apply"], "what": "test19 competition 2 apply (PREREG_19 4.2): frozen Q19 per set, "
                                                   "rule B, submission_rank.csv = pred_B.csv (public then private)",
               "mode": ctx["mode"], "created": now_iso(), "exit_code": 0, "usable": True, "unusable_reasons": [],
               "combo": COMBO, "method": METHOD, "theta": fc19["params"]["theta"],
               "models": {str(x): SF.MODEL_SHA256[x] for x in SF.SEEDS},
               "t1_verdict": (((self.pre.get("t09_prerequisites") or {}).get("t08_prerequisites") or {}).get("t1")
                              or {}).get("verdict"),
               "test09": {"t09_apply_report_sha256": pin_sha(os.path.join(ctx["rank9"], "t09_apply_report.json")),
                          "exit_code": rep9.get("exit_code"), "usable": rep9.get("usable"),
                          "seam_local": rep9.get("seam_local"), "gpu_seam": rep9.get("gpu_seam"),
                          "inputs_bound_checked": len(rep9.get("inputs_bound") or {})},
               "test02": {"fallback_used": t02["report"].get("fallback_used"),
                          "r20_first_upload_id": (t02["r20"] or {}).get("id"),
                          "r20_first_uploaded_at": (t02["r20"] or {}).get("uploaded_at")},
               "p1_sets": self.rec["p1_sets"],
               "earliest_p1_chunk_created_utc": min(self.rec["p1_sets"][s]["earliest_chunk_created_utc"] or ""
                                                    for s in SETS),
               "q09_vs_09": self.rec["q09_vs_09"], "set_seam_record_only": self.rec["set_seam_record_only"],
               "seam_local": seam["record"], "set_direction": direc, "rho_mono": RHO_MONO, "rho_floor": RHO_FLOOR,
               "p1_failures": rep9.get("p1_failures"),
               "sets": {s: res[s]["info"] for s in TEST_SETS},
               "rule_b_differences": diffs,
               "check_submission": {"mode": "official", "questions": [shown(p) for p in check_q], "errors": errors},
               "outputs_sha256": {n: sha_raw(os.path.join(rank, n)) for n in TEST_OUTPUTS},
               "decide_summary": {k: dsum[k] for k in ("n_questions", "n_missing", "missing_question_ids",
                                                        "n_A_ne_B", "distribution_B")},
               "level_summary": {s: t2.level_summary(res[s]["P"], res[s]["ok"],
                                                     [preds[q] for q in fz[s]["qids"]]) for s in TEST_SETS},
               "inputs_bound": bound["sha256"], "inputs_bound_paths": bound["paths"],
               "inputs_bound_rule": ".csv / .json: sha256 with CRLF folded to LF; other files (.npz): raw bytes",
               "checks": self.pre, "rerun": {"reason": a.reason, "replaced": self.replaced, "bug_revision": self.bug},
               "paths": {k: ctx[k] for k in ("R19", "R9", "R8", "H", "R7", "R4", "D2", "O2", "T", "C")},
               "tool": tool_record(), "environment": t4.environment(), "git_head": git_head(),
               "record_only": "level_summary, rule_b_differences, the SET GPU seam, the seam_val direction and the byte "
                              "equality of Q09 are recorded; nothing is chosen from them"}
        rep["seconds"] = round(time.time() - self.t0, 1)
        write_json(ctx, os.path.join(rank, "t19_apply_report.json"), rep)
        print(f"apply Q19: public {len(fz['public']['qids'])}, private {len(fz['private']['qids'])} questions -> "
              f"{shown(sub)} (sha256 {rep['outputs_sha256']['submission_rank.csv'][:12]}); check_submission official: "
              f"OK; rule-B differences on public vs 09 / 08: {(diffs.get('09') or {}).get('public')} / "
              f"{(diffs.get('08') or {}).get('public')}")
        print("下一步：commit results/test19/，09 上傳後照 docs/TESTDAY_19.md 第 4 步 official_log.py prepare"
              if ctx["real"] else f"({ctx['mode']}: nothing is uploaded)")
        return 0

    def local_seam(self, fzs, vks, vh, model07, fc07, fc9, fc19, PF):
        """PREREG_19 4.2 step 3: Q19 on seam_val (test02 / test04 / 07 seam inputs + the committed val_lp_p1e.npz and
        val_set.npz) against results/prereg19/rank/rank_val_probs.csv."""
        ctx = self.ctx
        keys_ok = (list(map(str, vh["question_id"])) == vks["question_id"] and [int(x) for x in vh["slot"]] ==
                   vks["slot"] and list(map(str, vh["color"])) == vks["color"])
        if not keys_ok:
            drift("val_hs.npz keys differ from test02's seam_val val_keys.npz", "local_seam", "seam_structure")
        hq = t9.t8call(t7.q02h_set, fzs, vks, vh, model07, fc07, PF)
        lpz = load_npz(path_of("results/prereg09/rank/val_lp_p1e.npz"))
        if (list(map(str, lpz["question_id"])) != vks["question_id"] or [int(x) for x in lpz["slot"]] != vks["slot"]
                or list(map(str, lpz["color"])) != vks["color"]):
            drift("val_lp_p1e.npz keys differ from test02's seam_val val_keys.npz", "local_seam", "seam_structure")
        usable = np.asarray(vks["parsed"], dtype=bool) & np.asarray(lpz["p1_ok"], dtype=bool)
        q9 = t8.q08_core(fzs, vks, hq, np.asarray(lpz["lp"], dtype=np.float64), np.asarray(lpz["p1_ok"], dtype=bool),
                         usable, fc9)
        vs = load_npz(path_of("results/prereg19/rank/val_set.npz"))
        if [str(x) for x in vs["question_id"]] != [q for q, _ in vks["groups"]]:
            drift("val_set.npz questions differ from test02's seam_val val_keys.npz", "local_seam", "seam_structure")
        if float(vs["scale"]) != SF.SCALE or [str(x) for x in vs["models_sha256"]] != \
                [SF.MODEL_SHA256[x] for x in SF.SEEDS]:
            refuse("val_set.npz is not the frozen S19's lp_SET (scale 4, the registered models)")
        has = np.asarray(vs["has"], dtype=bool)
        E_set = np.zeros((len(has), rf.N_LEVELS))
        E_set[has] = SF.SCALE * np.asarray(vs["lp"], dtype=np.float64)[has]
        r = q19_core(fzs, vks, hq, q9, E_set, has, fc19)
        seam_dir = os.path.join(ctx["rank19"], "seam_val")
        os.makedirs(guard_write(ctx, seam_dir), exist_ok=True)
        csv_p = guard_write(ctx, os.path.join(seam_dir, "rank_val_probs.csv"))
        t2.write_probs(csv_p, r["rows"])
        for n in ("pred_A.csv", "pred_B.csv", "decide_summary.json"):
            guard_write(ctx, os.path.join(seam_dir, n))
        decide.run(csv_p, seam_dir)
        ref_p, ref_b = os.path.join(P19_RANK, "rank_val_probs.csv"), os.path.join(P19_RANK, "pred_B.csv")
        rep4 = read_json(os.path.join(ctx["R4"], "rank", "t04_apply_report.json"))
        tol = float((rep4.get("seam_local") or {}).get("tol"))
        sc = t2.seam_compare(csv_p, os.path.join(seam_dir, "pred_B.csv"), ref_p, ref_b, tol, False)
        same_a = t2._lf_bytes(os.path.join(seam_dir, "pred_A.csv")) == t2._lf_bytes(os.path.join(P19_RANK,
                                                                                                 "pred_A.csv"))
        sl9 = read_json(os.path.join(ctx["rank9"], "seam_val", "seam_local.json"))
        inputs_identical = sl9.get("byte_identical_probs") is True and sl9.get("byte_identical_pred_B") is True
        verdict = t9.seam_verdict(sc, inputs_identical)
        direction = pf7.direction_record(r["rho"], r["rho_n"], RHO_EVAL, r9.RHO_FLOOR_VAL)
        a19 = read_json(os.path.join(P19_RANK, "apply_report.json"))
        direction["equal_to_prereg19_apply"] = r["rho"] == (a19.get("direction") or {}).get("rho")
        if inputs_identical and not direction["equal_to_prereg19_apply"]:
            drift("seam_val SET direction rho differs from results/prereg19/rank/apply_report.json although 09's "
                  "seam_val is byte-identical to prereg09", "local_seam", "seam_rho", direction)
        rec = {"format": FORMATS["seam_local"], "what": "test19 local Q19 seam (PREREG_19 4.2): Q19 on the seam_val "
                                                       "inputs + val_hs.npz + val_lp_p1e.npz + val_set.npz vs "
                                                       "results/prereg19/rank",
               "mode": ctx["mode"], "created": now_iso(), **sc, "byte_identical_pred_A": same_a,
               "t09_seam_byte_identical": inputs_identical, "required_byte_identical": inputs_identical,
               "usable_equal_to_prereg19": bool(np.array_equal(usable, np.asarray(vs["usable"], dtype=bool))),
               "verdict": verdict, "passed": verdict == "passed",
               "tol_source": "test04 t04_apply_report.json seam_local.tol (as 07 / 08 / 09)",
               "set": r["info"], "direction": direction,
               "reference_sha256": {n: pin_sha(os.path.join(P19_RANK, n))
                                    for n in ("rank_val_probs.csv", "pred_B.csv", "pred_A.csv", "val_set.npz")},
               "outputs_sha256": {n: sha_raw(os.path.join(seam_dir, n)) for n in
                                  ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json")}}
        path = write_json(ctx, os.path.join(seam_dir, "seam_local.json"), rec)
        print(f"local seam: {sc['questions']} questions, max |dP| {sc['max_abs_diff']:.3g} (tol {tol:g}), rule-B "
              f"differences {sc['rule_b_differences']}, byte identical {sc['byte_identical_probs']}/"
              f"{sc['byte_identical_pred_B']}/{same_a} (required: {inputs_identical}) -> {verdict}", flush=True)
        self.rec["seam_local"] = {k: rec[k] for k in ("verdict", "passed", "max_abs_diff", "tol", "rule_b_differences",
                                                      "byte_identical_probs", "byte_identical_pred_B",
                                                      "required_byte_identical", "usable_equal_to_prereg19")}
        if verdict == "structure" or not rec["usable_equal_to_prereg19"]:
            drift("local seam: questions / order / game counts / usable games differ from results/prereg19/rank",
                  "local_seam", "seam_structure", {"problems": sc["problems"],
                                                   "count_mismatch_questions": sc["count_mismatch_questions"],
                                                   "usable_equal": rec["usable_equal_to_prereg19"]})
        if verdict == "drift":
            drift("local seam: 09's seam_val equals prereg09 byte for byte but Q19 does not equal prereg19 byte for "
                  "byte (program drift)", "local_seam", "seam_bytes", self.rec["seam_local"])
        if verdict == "not_used":
            not_used("local seam outside the tolerance while 09's seam holds: 19 is not used (PREREG_19 4.2)",
                     "local_seam", "seam_tol", self.rec["seam_local"])
        return {"path": path, "record": self.rec["seam_local"], "direction": direction}

    def inputs_bound(self, t02, set_seam_path, seam_local_path):
        """The files official_log re-hashes (keys: the real-mode repository paths)."""
        ctx = self.ctx
        R4, R7, R8, R9 = ctx["R4"], ctx["R7"], ctx["R8"], ctx["R9"]
        items = [("results/test02/rank/apply_test_report.json", t02["report_path"]),
                 ("results/test02/checks/rank/t1.json", os.path.join(ctx["C"], "rank", "t1.json")),
                 ("results/test04/rank/t04_apply_report.json", os.path.join(R4, "rank", "t04_apply_report.json")),
                 ("results/test07/rank/t07_apply_report.json", os.path.join(R7, "rank", "t07_apply_report.json")),
                 ("results/test08/rank/t08_apply_report.json", os.path.join(R8, "rank", "t08_apply_report.json")),
                 ("results/test09/rank/t09_apply_report.json", os.path.join(R9, "rank", "t09_apply_report.json")),
                 ("results/test09/rank/rank_test_probs.csv", os.path.join(R9, "rank", "rank_test_probs.csv")),
                 ("results/test09/rank/submission_rank.csv", os.path.join(R9, "rank", "submission_rank.csv")),
                 ("results/test09/rank/test_lp_p1e.npz", os.path.join(R9, "rank", "test_lp_p1e.npz")),
                 ("results/test09/rank/seam_val/seam_local.json", os.path.join(R9, "rank", "seam_val",
                                                                               "seam_local.json")),
                 ("results/test09/gpu_seam.json", os.path.join(R9, "gpu_seam.json")),
                 ("results/test19/set_seam.json", set_seam_path),
                 ("results/test19/rank/seam_val/seam_local.json", seam_local_path)]
        items += [(r_, path_of(r_)) for r_ in P19_PINS]
        items += [(r_, path_of(r_)) for r_ in t9.P09_PINS]
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
    except T19Error as e:
        if e.code in (3, 4):
            run.write_verdict(e)
            if e.code == 3:
                print(f"19 不用（PREREG_19 4.2）：{e}\ncommit {shown(ctx['R19'])}；最終照 PREREG_09 5.3（不重跑）",
                      file=sys.stderr)
        raise


# ================================================================ status (read only)
def cmd_status(a):
    ctx = resolve(a, "status")
    lines = [f"test19 status ({ctx['mode']}; {ctx['R19']})"]
    rp = os.path.join(ctx["rank19"], "t19_apply_report.json")
    vp = os.path.join(ctx["rank19"], "attempt_verdict.json")
    np_ = os.path.join(ctx["R19"], "not_used.json")
    if os.path.exists(rp):
        r = read_json(rp)
        lines.append(f"apply: exit {r.get('exit_code')}; submission sha256 "
                     f"{str((r.get('outputs_sha256') or {}).get('submission_rank.csv'))[:12]}; Q09 = 09 "
                     f"{(r.get('q09_vs_09') or {}).get('equal')}; local seam {(r.get('seam_local') or {}).get('verdict')}"
                     + (f", committed {committed_clean(rp)}" if ctx["real"] else ""))
    elif os.path.exists(np_):
        r = read_json(np_)
        lines.append(f"19 not used: {r.get('stage')} ({r.get('reason')})")
    elif os.path.exists(vp):
        r = read_json(vp)
        lines.append(f"apply: attempt_verdict exit {r.get('exit_code')} at {r.get('stage')} ({r.get('reason')})")
    else:
        lines.append("apply: not run")
    if ctx.get("R9"):
        r9_ = read_json_or_none(os.path.join(ctx["R9"], "rank", "t09_apply_report.json"))
        lines.append("test09 apply: " + (f"exit {r9_.get('exit_code')}, usable {r9_.get('usable')}" if r9_ else
                                         "not run"))
    left = K6_19 - now()
    lines.append(f"K6-19 {K6_19.isoformat()}: " + (f"{left.days} days {left.seconds // 3600} h left"
                                                   if left.total_seconds() > 0 else "passed"))
    print("\n".join(lines))
    return 0


# ================================================================ command line
def build_parser():
    ap = argparse.ArgumentParser(prog="t19_rank.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("apply", "status"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--rehearsal", action="store_true")
        for d in ("--results-dir", "--t09-results-dir", "--t08-results-dir", "--p1-data-dir", "--t07-results-dir",
                  "--t04-results-dir", "--t02-results-dir", "--t02-out-dir", "--tests-dir", "--checks-dir"):
            p.add_argument(d)
        p.add_argument("--r20-uploaded-at", help="drills only: R20's first upload time")
        p.add_argument("--now", help="drills only: the current time (K6-19)")
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
    except T19Error as e:
        print(f"{'REFUSED' if e.code == 2 else 'NOT USED' if e.code == 3 else 'DRIFT'} (exit {e.code}): {e}",
              file=sys.stderr)
        return e.code
    except Exception:  # noqa: BLE001 - an internal error is a refusal with its traceback, never a decision
        traceback.print_exc()
        print("ERROR (exit 2): unexpected internal error of tools/t19_rank.py; stop and check the local tool",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
