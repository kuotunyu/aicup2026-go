"""C10 test25, competition 2 (rank, TEAM_11248): the 25 test-day local tool (Q25 = Q19 with the SET term's models S19
replaced by S25, the set model retrained with the validation players).

Registration: docs/PREREG_25_SET_MORE_PLAYERS.md section 4 (rev 1) and the test-day revision (rev 3). CPU only, after
19's committed official run: 25 reads 08's P1 chunks (the ones 09 and 19 read) and never runs the GPU.
tools/t19_rank.py, t09_rank.py (and through them t08 / t07 / t04 / t02) are imported, never edited.

    PY tools/t25_rank.py apply  [MODE] [DIRS] [--rerun --reason TEXT [--bug-revision N]]   (CPU, .venv)
    PY tools/t25_rank.py status [MODE] [DIRS]                                              (read only)
    (PY = .venv/Scripts/python.exe; Git Bash; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL thread variables;
     from a path without 'AppData\\Local' (hs_lib's data/local guard), e.g. the repository or a subst drive)

MODE  none = real; --dry-run (every directory given, the test25 root outside the repository)
DIRS  --results-dir R25     test25 root (writes R25/set_seam.json, R25/rank; real results/test25)
      --t19-results-dir R19, --t09-results-dir R9, --t08-results-dir R8, --p1-data-dir H, --t07-results-dir R7,
      --t04-results-dir R4, --t02-results-dir D2, --t02-out-dir O2, --tests-dir T, --checks-dir C
Drills only (never real): --now ISO.

No data/local (PREREG_25 4.2): t25 never runs the 09 / 08 / 07 / 04 / test02 prerequisite chain (it hashes the seam_val
question file, which lives under data/local) and never touches seam_val. It binds 19's committed official run instead
and recomputes Q09 and Q19 on public and private to prove that the inputs and the program did not change.

apply  Q25 on public and private (PREREG_25 4.2), in this order (the first failing item decides):
       prerequisites: git clean (real); no BLAS thread variable; the committed prereg25 files equal their pinned
       SHA-256s (else 2) and record the adoption (adoption.json usable and finalized; else 3); .venv python / numpy /
       scipy = frozen_config.json (else 4); K6-25; 19's official run (t19_apply_report.json of this mode, exit 0,
       usable, committed in the real mode, every input it bound and its outputs unchanged, T1 PASS / NORMALIZED, no
       R10 fallback; else 2); the frozen 09 / 07 / 04 configurations (the readers of t09 / t07 / t04).
       1 the three P1 sets of 08 (tools/t09_rank.py Apply.p1_sets), equal to the ones 19's report records.
       2 Q09 recomputed with 09's frozen map equal to results/test09/rank/rank_test_probs.csv (probabilities within
         1e-12, rule A and rule B levels equal); lp_P1TO_E within 1e-12, p1_ok / usable equal to test_lp_p1e.npz; else 4.
       3 Q19 recomputed with the frozen S19 and 19's theta equal to results/test19/rank/rank_test_probs.csv (same
         conditions); lp_SET19 within 1e-12, has / usable equal to 19's test_lp_set.npz; else 4.
       4 the SET25 GPU seam (S25 on the 400 seam games against the validation chunks 0-1) -> R25/set_seam.json
         (recorded only).
       5 the SET25 direction check per set (rho >= max(0.5 x rho_mono_25, 0.155)) -> 3.
       Then rank_test_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json, submission_rank.csv (= pred_B.csv),
       test_lp_set.npz, check_submission --official (else 4) and R25/rank/t25_apply_report.json.
status Read-only summary.

Exit codes (as tools/t19_rank.py): 0 done; 2 refused or error; 3 the registered "25 is not used" (R25/not_used.json;
commit; the final choice follows PREREG_19 4.3; never rerun); 4 local drift or mismatch (R25/rank/attempt_verdict.json;
commit, stop and find out why, then --rerun --reason). After exit 3 or a written t25_apply_report.json nothing is
rerun except with --bug-revision N (a committed PREREG_25 rev N >= 4 added after that record).

Discipline: never imports torch, u1_lib, p09_map, p19_set, p25_set or a player module; never fits anything (the
frozen theta, the frozen S25 / S19 model files, numpy float64 forward); no validation question or label file is ever
opened; tools/score.py is never run; writes only under R25. Competition 1 is never read (PREREG_25 6).
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SELF = os.path.join(HERE, "t25_rank.py")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t19_rank as t19  # noqa: E402
import p19_forward as SF  # noqa: E402
import p25_forward as SF25  # noqa: E402

t9 = t19.t9
t8, t7, t4, pf7, r8, r9, FW, rf, t2 = t9.t8, t9.t7, t9.t4, t9.pf7, t9.r8, t9.r9, t9.FW, t9.rf, t9.t2
decide, check_submission = t9.decide, t9.check_submission

# ================================================================ paths (module constants; the unit tests patch them)
RESULTS = os.path.join(REPO, "results")
REAL_R25 = os.path.join(RESULTS, "test25")
REAL = {"R19": t19.REAL_R19, "R9": t9.REAL_R9, "R8": t9.REAL_R8, "H": t9.REAL_H8, "R7": t9.REAL_R7,
        "R4": t9.REAL_R4, "D2": t9.REAL_D2, "O2": t9.REAL_O2, "T": t9.REAL_T, "C": t9.REAL_C}
DIR_FLAGS = {"R19": "--t19-results-dir", "R9": "--t09-results-dir", "R8": "--t08-results-dir", "H": "--p1-data-dir",
             "R7": "--t07-results-dir", "R4": "--t04-results-dir", "D2": "--t02-results-dir", "O2": "--t02-out-dir",
             "T": "--tests-dir", "C": "--checks-dir"}
P25_RANK = os.path.join(RESULTS, "prereg25", "rank")
P25_DOC = os.path.join(REPO, "docs", "PREREG_25_SET_MORE_PLAYERS.md")
P25_DOC_REL = "docs/PREREG_25_SET_MORE_PLAYERS.md"

# ================================================================ registered values
# the committed prereg25 files the test day reads (SHA-256: .json with CRLF folded to LF, .npz raw bytes)
P25_PINS = {
    "results/prereg25/rank/frozen_config.json": "30c000acbea5bbd17b4758dc646687d05b3494d015a12a08b88d8b4024d58fe3",
    "results/prereg25/rank/freeze_report.json": "9684e221dc727035155cdf2b565901469c4ef3f69fbcd83d61f5c6d37aa2f474",
    "results/prereg25/rank/freeze_audit.json": "9e64abb7576658f49124dabc4f8159ad1ab4b7fcd9db5a34403a461f3dc8764b",
    "results/prereg25/rank/adoption.json": "40b100357cc3ebb679b512a05c39fc0d8a067bb427e76128706cd72c8db31605",
    "results/prereg25/rank/s25_seed0.npz": SF25.MODEL_SHA256[0],
    "results/prereg25/rank/s25_seed1.npz": SF25.MODEL_SHA256[1],
    "results/prereg25/rank/s25_seed2.npz": SF25.MODEL_SHA256[2]}
TERMS = ["OP", "O4", "SWA", "SWB", "HS", "P1TO", "SET"]
COMBO = {"id": "Q25", "sources": ["C5", "OP", "O4", "SWA", "SWB", "HS", "P1TO_E", "SET"], "terms": TERMS,
         "p1_scale": 4.0, "set_scale": 4.0, "p1_term": "P1TO_E (ens3fam, prereg09)",
         "set_term": "SET (S25 = S19 retrained with the validation players, 3 seeds, prereg25)"}
METHOD = "prereg25 Q25 = Q19 with S25 (set model + validation players), rule B (甲)"
RHO_MONO = 0.9576161336208181                            # frozen_config.json direction_ref.rho_mono (PREREG_25 4.2 step 5)
RHO_EVAL = 0.9538300566495522                            # frozen_config.json direction_ref.rho_eval (record)
RHO_FLOOR = r9.RHO_FLOOR_TEST                    # 0.155: one-sided p = 0.001 for 400 questions
TOL = 1e-12                                      # PREREG_25 4.2 steps 2 and 3
TW = t9.TW
K6_25 = dt.datetime(2026, 11, 18, 23, 59, 59, tzinfo=TW)
P25_MIN_REV = 4                                  # rev 3 registers the test day; an override is rev 4 or later
SETS, TEST_SETS = t9.SETS, t9.TEST_SETS
T1_OK = ("PASS", "NORMALIZED")
FORMATS = {"set_seam": "c10-test25 set seam v1", "apply": "c10-test25 rank apply v1",
           "verdict": "c10-test25 attempt verdict v1", "not_used": "c10-test25 not used v1",
           "lp": "c10-test25 test_lp_set v1"}
TEST_OUTPUTS = ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv",
                "test_lp_set.npz")
T19_OUTPUTS = t19.TEST_OUTPUTS
_LABEL = "ans" + "wer"


class T25Error(RuntimeError):
    """A stop with its exit code: 2 refused, 3 registered 'no 25', 4 local drift / mismatch."""

    def __init__(self, message, code=2, stage=None, reason=None, details=None):
        super().__init__(message)
        self.code, self.stage, self.reason, self.details = code, stage, reason, details


def refuse(msg, reason=None):
    raise T25Error(msg, 2, reason=reason)


def not_used(msg, stage, reason, details=None):
    raise T25Error(msg, 3, stage, reason, details)


def drift(msg, stage, reason, details=None):
    raise T25Error(msg, 4, stage, reason, details)


KIT_ERRORS = (t19.T19Error, t9.T09Error, t8.T08Error, t7.T07Error, t4.T04Error, t2.T02Error)
FOREIGN_ERRORS = tuple(dict.fromkeys(t19.FOREIGN_ERRORS + KIT_ERRORS))


def call(fn, *args):
    """A function of t19 / t09 / t08 / t07 / t04 / t02: its refusal is ours (2); its drift (4) stays 4; its 'not used'
    (3) is a refusal here (25 never decides another kit's registration)."""
    try:
        return fn(*args)
    except KIT_ERRORS as e:
        raise T25Error(f"({fn.__module__}) {e}", 4 if getattr(e, "code", 2) == 4 else 2, getattr(e, "stage", None),
                       getattr(e, "reason", None), getattr(e, "details", None))
    except FOREIGN_ERRORS as e:
        refuse(f"({getattr(fn, '__module__', '?')}) {e}")


# ================================================================ small helpers
def now():
    """The current time (a module function: the tests replace it to cross K6-25)."""
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
    if path is not None:
        p = os.path.abspath(os.fspath(path))
        if _LABEL in p.lower():
            refuse(f"refusing {path}: validation label files are never opened by the test25 tools")
        if pf7.under_data_local(p):
            refuse(f"refusing {path}: data/local is never opened by the test25 tools")
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
    names = ["t25_rank.py", "p25_forward.py", "t19_rank.py", "p19_forward.py", "t09_rank.py", "t08_rank.py",
             "t07_rank.py", "t04_rank.py", "p09_rank_fuse.py", "p09_forward.py", "p08_rank_fuse.py", "p07_rank_fuse.py",
             "check_submission.py", "official_log.py", "t02_rank.py", "p04_rank_fuse.py", "p03_rank_fuse.py",
             "p02_rank_fuse.py", "decide.py"]
    return {n: (sha_lf(os.path.join(HERE, n)) if os.path.exists(os.path.join(HERE, n)) else None) for n in names}


def tool_record():
    return {"git_head": git_head(), "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": tool_files()}


def blas_env_check():
    set_ = {k: os.environ[k] for k in r8.BLAS_ENV if os.environ.get(k)}
    if set_:
        refuse(f"BLAS thread variables are set ({set_}): unset them (the registered byte seams use the default "
               f"OpenBLAS threads)")
    return {k: None for k in r8.BLAS_ENV}


# ================================================================ modes and directories
def forbidden_area(path):
    """Where t25 never writes: results/prereg*, every other results/test*, data/local, data/test*, the ledger."""
    p = os.path.abspath(path)
    if pf7.under_data_local(p):
        return "data/local"
    if t2.same_path(p, os.path.join(RESULTS, "score_ledger.csv")):
        return "results/score_ledger.csv"
    if t2.is_inside(p, RESULTS) and not t2.same_path(p, RESULTS):
        first = os.path.relpath(os.path.realpath(p), os.path.realpath(RESULTS)).split(os.sep)[0].lower()
        if first.startswith("prereg") or (first.startswith("test") and first != "test25"):
            return "results/" + first
    if t2.is_inside(p, os.path.join(REPO, "data")):
        first = os.path.relpath(os.path.realpath(p), os.path.realpath(os.path.join(REPO, "data"))).split(os.sep)[0]
        if first.lower().startswith(("test", "prereg")):          # data/local: under_data_local above
            return "data/" + first
    return None


def resolve(a, cmd):
    """Real mode: every directory is the repository path. --dry-run: every directory given, R25 outside the
    repository. -> ctx (the keys t09's Apply.p1_sets and t07.load_sw read, plus the test25 / test19 roots)."""
    mode = "dry-run" if a.dry_run else "real"
    given = {k: getattr(a, DIR_FLAGS[k].lstrip("-").replace("-", "_")) for k in DIR_FLAGS}
    if mode == "real":
        if getattr(a, "now", None):
            refuse("real mode: --now is for drills only (--dry-run)")
        if a.results_dir and not t2.same_path(a.results_dir, REAL_R25):
            refuse(f"real mode: --results-dir must be {REAL_R25} (got {a.results_dir})")
        for k, v in given.items():
            if v and not t2.same_path(v, REAL[k]):
                refuse(f"real mode: {DIR_FLAGS[k]} must be {REAL[k]} (got {v})")
        R25, dirs = REAL_R25, dict(REAL)
    else:
        missing = (["--results-dir"] if not a.results_dir else []) + \
            ([DIR_FLAGS[k] for k, v in given.items() if not v] if cmd != "status" else [])
        if missing:
            refuse(f"--dry-run needs {', '.join(missing)} (it never falls back to the repository paths)")
        R25 = os.path.abspath(a.results_dir)
        if t2.is_inside(R25, REPO):
            refuse(f"--dry-run: --results-dir must be outside the repository (got {R25})")
        dirs = {k: (os.path.abspath(v) if v else None) for k, v in given.items()}
    for p in [R25] + [v for v in dirs.values() if v]:
        refuse_label_path(p)
    if os.path.basename(os.path.normpath(dirs["T"] or "x")).lower() == "raw":
        refuse("--tests-dir: the test files are the checked copies (data/test), never data/test/raw")
    bad = forbidden_area(R25)
    if bad:
        refuse(f"output folder inside the read-only {bad}")
    ctx = dict(dirs, mode=mode, real=mode == "real", smoke=False, cmd=cmd, R25=R25, rank25=os.path.join(R25, "rank"),
               rank19=os.path.join(dirs["R19"], "rank") if dirs["R19"] else None,
               rank9=os.path.join(dirs["R9"], "rank") if dirs["R9"] else None,
               rank=os.path.join(dirs["R8"], "rank") if dirs["R8"] else None,
               p1=os.path.join(dirs["R8"], "p1") if dirs["R8"] else None, write_roots25=[R25])
    ctx["TMP"] = os.path.join(R25, "tmp")
    return ctx


def guard_write(ctx, path):
    p = os.path.abspath(path)
    refuse_label_path(p)
    bad = forbidden_area(p)
    if bad:
        refuse(f"refusing to write {p}: it is inside the read-only {bad}")
    if not any(t2.is_inside(p, r) for r in ctx["write_roots25"]):
        refuse(f"refusing to write {p}: outside {ctx['write_roots25']}")
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
    own = [SELF] + [os.path.join(HERE, n) for n in ("p25_forward.py", "p19_forward.py", "t19_rank.py", "t09_rank.py",
                                                    "official_log.py", "check_submission.py")]
    rec["git"] = {"clean": True, "head": git_head(), "own_files": require_committed(own, "the test25 tools")}


def check_p25_pins():
    """The committed prereg25 files equal their pinned SHA-256s (else 2); adoption.json records the adopted Q25 (usable
    and finalized; else 3); frozen_config.json is the frozen Q25 (terms, scales, theta, direction reference, 09's map).
    -> (frozen config, S25 models, record)."""
    got = {r_: (pin_sha(path_of(r_)) if os.path.exists(path_of(r_)) else None) for r_ in P25_PINS}
    bad = sorted(r_ for r_ in P25_PINS if got[r_] != P25_PINS[r_])
    if bad:                              # checked before the adoption flag is read (as t19)
        refuse(f"prereg25 files differ from the pinned SHA-256s: {bad} (never edited after the adoption)")
    ad = read_json_or_none(path_of("results/prereg25/rank/adoption.json"))
    if ad is None or ad.get("usable") is not True or not ad.get("finalized") or ad.get("combo") != "Q25":
        not_used("PREREG_25 2.4: Q25 is not adopted (adoption.json not usable / finalized): 25 is not used",
                 "prerequisites", "p25_not_adopted")
    fc = read_json(path_of("results/prereg25/rank/frozen_config.json"))
    combo = fc.get("combo") or {}
    dref = fc.get("direction_ref") or {}
    if fc.get("format") != "c10-prereg25 rank frozen_config v1" or combo.get("id") != "Q25" or \
            combo.get("terms") != TERMS or float(combo.get("p1_scale", 0)) != 4.0 or \
            float(combo.get("set_scale", 0)) != 4.0 or len((fc.get("params") or {}).get("theta") or []) != 18:
        refuse("results/prereg25/rank/frozen_config.json is not the frozen Q25 configuration")
    if float(dref.get("rho_mono")) != RHO_MONO or float(dref.get("rho_eval")) != RHO_EVAL:
        refuse("frozen_config.json direction_ref differs from the registered values")
    if (fc.get("p1e_map") or {}).get("sha256") != t9.MAP_SHA:
        refuse("frozen_config.json does not name 09's p1e_map.npz")
    try:
        models = SF25.load_models(P25_RANK)
    except FOREIGN_ERRORS as e:
        refuse(str(e))
    return fc, models, {"prereg25_sha256": got, "adopted": {"usable": True, "finalized": ad.get("finalized")}}


def check_p19_pins():
    """19's committed files (t19_rank.P19_PINS) and the frozen S19 models: Q19 is recomputed with them (step 3)."""
    got = {r_: (pin_sha(path_of(r_)) if os.path.exists(path_of(r_)) else None) for r_ in t19.P19_PINS}
    bad = sorted(r_ for r_ in t19.P19_PINS if got[r_] != t19.P19_PINS[r_])
    if bad:
        refuse(f"prereg19 files differ from the pinned SHA-256s: {bad}")
    fc = read_json(path_of("results/prereg19/rank/frozen_config.json"))
    if (fc.get("combo") or {}).get("id") != "Q19" or (fc.get("combo") or {}).get("terms") != TERMS:
        refuse("results/prereg19/rank/frozen_config.json is not the frozen Q19 configuration")
    try:
        models = SF.load_models(t19.P19_RANK)
    except FOREIGN_ERRORS as e:
        refuse(str(e))
    return fc, models, {"prereg19_sha256": got}


def check_venv(fc25):
    want = {k: ((fc25.get("environment") or {}).get("versions") or {}).get(k) for k in ("python", "numpy", "scipy")}
    got = r9.venv_versions(False)
    bad = {k: {"now": got.get(k), "frozen": want[k]} for k in want if got.get(k) != want[k]}
    if bad:
        drift(f".venv differs from results/prereg25/rank/frozen_config.json {bad}: restore .venv (never upgraded "
              f"before 11/24)", "prerequisites", "venv_versions", bad)
    return {"equal": True, "versions": got}


def earlier_success(ctx):
    p = os.path.join(ctx["rank25"], "t25_apply_report.json")
    if not os.path.exists(p):
        return False, None
    rep = read_json(p)
    if rep.get("exit_code") != 0:
        return False, None
    if not ctx["real"]:
        return True, {"note": f"{ctx['mode']}: commit time not checked"}
    h, t = commit_time(p)
    ok = t is not None and parse_iso(t, "commit time") <= K6_25
    return ok, {"commit": h, "committed_at": t}


def check_k6(ctx, a):
    t_now = parse_iso(a.now, "--now") if getattr(a, "now", None) else now()
    if t_now <= K6_25:
        return {"now": t_now.isoformat(), "k6_25": K6_25.isoformat(), "passed": False}
    ok, rec = earlier_success(ctx)
    if ok:
        refuse(f"K6-25 {K6_25.isoformat()} has passed; the successful run before it ({rec}) stays valid: nothing is "
               f"rerun", reason="k6_passed_success")
    not_used(f"K6-25 {K6_25.isoformat()} has passed without a successful official 25 run: 25 is not used this "
             f"season (PREREG_25 5); the final choice follows PREREG_19 4.3", "prerequisites", "k6_25")


def bound_path(ctx, key):
    """The tree path of a key of 19's inputs_bound (keys are real repository paths)."""
    if ctx["real"] or key.startswith("results/prereg"):
        return path_of(key)
    roots = {"results/test19/": ctx["R19"], "results/test09/": ctx["R9"], "results/test08/": ctx["R8"],
             "results/test07/": ctx["R7"], "results/test04/": ctx["R4"], "results/test02/checks/": ctx["C"],
             "results/test02/": ctx["D2"]}
    for pre, root in roots.items():
        if key.startswith(pre):
            rest = key[len(pre):].split("/")
            if pre == "results/test02/" and rest[0] == "rank" and len(rest) == 2 and \
                    rest[1] in ("apply_test_report.json", "opening_test.npz"):
                return os.path.join(ctx["O2"], rest[1])
            return os.path.join(root, *rest)
    return None


def t19_state(ctx):
    """19's official run of this mode: exit 0, usable, Q19 (committed in the real mode); every input it bound and its
    outputs unchanged; T1 PASS / NORMALIZED and no R10 fallback (re-read from the files it bound). 19 not done or not
    used -> 2 (25 builds on 19's run). -> (report, record)."""
    R19 = ctx["R19"]
    rp = os.path.join(ctx["rank19"], "t19_apply_report.json")
    if os.path.exists(os.path.join(R19, "not_used.json")):
        refuse("test19 judged '19 is not used': 25 needs 19's successful official run (PREREG_25 4.2)",
               reason="t19_not_used")
    if not os.path.exists(rp):
        refuse("test19 apply has not finished (no t19_apply_report.json): 25 builds on it (PREREG_25 4.2)",
               reason="t19_not_done")
    rep = read_json(rp)
    modes = ("real",) if ctx["real"] else ("real", "dry-run")   # a dry run may rehearse on 19's official run
    if rep.get("format") != t19.FORMATS["apply"] or rep.get("mode") not in modes:
        refuse(f"{shown(rp)}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (this run: {ctx['mode']})")
    if rep.get("exit_code") != 0 or rep.get("usable") is not True or (rep.get("combo") or {}).get("id") != "Q19":
        refuse(f"{shown(rp)}: exit_code {rep.get('exit_code')!r}, usable {rep.get('usable')!r}")
    files = [rp] + [os.path.join(ctx["rank19"], n) for n in T19_OUTPUTS] + \
        [os.path.join(R19, "set_seam.json"), os.path.join(ctx["rank19"], "seam_val", "seam_local.json")]
    missing = [shown(p) for p in files if not os.path.exists(p)]
    if missing:
        refuse(f"test19 outputs missing: {missing[:5]}")
    if ctx["real"]:
        require_committed(files, "results/test19")
    bound = rep.get("inputs_bound") or {}
    probs = [] if bound else ["the 19 report lists no inputs_bound"]
    for key, want in sorted(bound.items()):
        p = bound_path(ctx, key)
        if p is None or not os.path.exists(p):
            probs.append(f"{key} missing")
        elif pin_sha(p) != want:
            probs.append(f"{key} changed")
    if probs:
        refuse(f"test19's bound inputs changed since its apply: {probs[:5]} (PREREG_25 4.2)",
               reason="t19_inputs_changed")
    want = rep.get("outputs_sha256") or {}
    changed = [n for n in T19_OUTPUTS if sha_raw(os.path.join(ctx["rank19"], n)) != want.get(n)]
    if changed:
        refuse(f"test19 outputs differ from its report: {changed}", reason="t19_outputs_changed")
    t1p = bound_path(ctx, "results/test02/checks/rank/t1.json")
    v = read_json(t1p).get("verdict") if t1p and os.path.exists(t1p) else None
    if ctx["real"] and v not in T1_OK:
        refuse(f"T1 verdict {v!r}: 25 plays only with T1 PASS or NORMALIZED")
    t2p = bound_path(ctx, "results/test02/rank/apply_test_report.json")
    t2rep = read_json(t2p)
    if t2rep.get("fallback_used") is not False:
        refuse(f"test02 fallback_used {t2rep.get('fallback_used')!r}: 25 needs the R20 run (FAIL_OPENING: never)")
    return rep, {"path": shown(rp), "sha256": pin_sha(rp), "exit_code": 0, "usable": True,
                 "inputs_bound_checked": len(bound), "submission_sha256": want.get("submission_rank.csv"),
                 "t1_verdict": v, "t02_fallback_used": False}


def frozen_readers(ctx):
    """The frozen 09 / 07 / 04 configurations through their own readers (pins and formats; no data/local)."""
    fc9, mp, rec9 = call(t9.check_p09_pins)
    fc07, rec7 = call(t7.check_p07_pins)
    fc04, rec4 = call(t7.t4call, t4.load_frozen_config, ctx["real"])
    model07 = pf7.load_hs_map(path_of("results/prereg07/rank/hs_map.npz"))
    return fc9, mp, fc07, fc04, model07, {"prereg09": rec9, "prereg07": rec7, "prereg04_frozen": rec4}


def prereqs(ctx, a):
    """PREREG_25 4.2 in this order. -> (fc25, models25, fc19, models19, rep19, readers, record)."""
    rec = {}
    check_git_clean(ctx, rec)                                            # 1
    rec["blas_env"] = blas_env_check()
    fc25, models25, rec["prereg25"] = check_p25_pins()                   # 2 (25 adopted)
    rec["venv"] = check_venv(fc25)                                       # 3
    rec["k6_25"] = check_k6(ctx, a)                                      # 4
    rep19, rec["t19_apply_report"] = t19_state(ctx)                      # 5 (19's official run)
    fc19, models19, rec["prereg19"] = check_p19_pins()
    readers = frozen_readers(ctx)
    rec["frozen_readers"] = readers[-1]
    return fc25, models25, fc19, models19, rep19, readers[:-1], rec


# ================================================================ test02's opening without seam_val
def load_opening(ctx, qfiles):
    """test02's opening_test.npz (the checks of t02_rank's opening loader for public and private only: it also hashes
    the seam_val question file, which is under data/local and never opened here)."""
    path = os.path.join(ctx["O2"], "opening_test.npz")
    try:
        z = t2.rp.load_npz(refuse_label_path(path), ["set", "question_id", "slot", "color", "o", "has", "meta"])
    except FOREIGN_ERRORS as e:
        refuse(f"{shown(path)}: {e}")
    meta = json.loads(str(z["meta"]))
    if meta.get("format") != t2.OPENING_FORMAT or meta.get("seam_bitwise_equal") is not True:
        refuse(f"{shown(path)}: not a passed {t2.OPENING_FORMAT} file")
    if meta.get("frozen_opening_build_id") != t2.FROZEN_OPENING_BUILD_ID:
        refuse(f"{shown(path)}: made for another frozen opening build")
    for s in TEST_SETS:
        if (meta["inputs"].get(s) or {}).get("sha256") != sha_raw(qfiles[s]):
            refuse(f"{shown(path)}: {s} was scored on another question file")
    z["meta"] = meta
    z["sha256"] = sha_raw(path)
    return z


# ================================================================ pure parts
def compare_lp(mine, ref, keys=("lp",)):
    """lp arrays within TOL and has / usable equal (19's test_lp_set.npz)."""
    rec = {"question_id_equal": [str(x) for x in mine["question_id"]] == [str(x) for x in ref["question_id"]]}
    for k in ("has", "usable"):
        rec[f"{k}_equal"] = bool(np.array_equal(np.asarray(mine[k], dtype=bool), np.asarray(ref[k], dtype=bool)))
    a, b = np.asarray(mine["lp"], dtype=np.float64), np.asarray(ref["lp"], dtype=np.float64)
    has = np.asarray(ref["has"], dtype=bool)
    rec["lp_max_abs_diff"] = float(np.max(np.abs(a[has] - b[has]))) if a.shape == b.shape and has.any() else \
        (0.0 if a.shape == b.shape else None)
    rec["equal"] = bool(rec["question_id_equal"] and rec["has_equal"] and rec["usable_equal"]
                        and rec["lp_max_abs_diff"] is not None and rec["lp_max_abs_diff"] <= TOL)
    return rec


def rows_compare(rows, ref_path):
    """t09_rank.probs_rows_compare (byte equality recorded; same questions / order / counts, probabilities within
    1e-12, rule A and rule B levels equal)."""
    rec = t9.probs_rows_compare(rows, ref_path)
    rec["reference_sha256_lf"] = rec.pop("t08_sha256")
    rec["tol"] = TOL
    rec["equal"] = bool(rec["equal"] and rec["max_abs_diff"] <= TOL)
    return rec


# ================================================================ apply
def apply_outputs(ctx):
    rank = ctx["rank25"]
    out = [os.path.join(rank, n) for n in TEST_OUTPUTS + ("t25_apply_report.json", "attempt_verdict.json")]
    out += [os.path.join(ctx["R25"], "set_seam.json"), os.path.join(ctx["R25"], "not_used.json")]
    return out


def check_bug_revision(n, overturned):
    """--bug-revision N: PREREG_25 rev N >= 4 committed and clean, its row at HEAD, added after every overturned
    record (ISO times). -> record."""
    if n is None:
        return None
    if n < P25_MIN_REV:
        refuse(f"--bug-revision {n}: an override needs PREREG_25 rev {P25_MIN_REV} or later")
    if not committed_clean(P25_DOC):
        refuse(f"{P25_DOC_REL} must be committed and clean")
    row = f"| rev {n} |"
    r = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{P25_DOC_REL}"], capture_output=True, timeout=60)
    head = r.stdout if r.returncode == 0 else b""
    if not any(line.startswith(row) for line in head.decode("utf-8", "replace").splitlines()):
        refuse(f"{P25_DOC_REL} at HEAD has no row starting {row!r}")
    rc, out = git("log", "--reverse", "-S" + row, "--format=%H %cI", "--", P25_DOC_REL)
    if rc != 0 or not out:
        refuse(f"no commit added {row!r}")
    h, t = out.splitlines()[0].split()
    at = parse_iso(t, "commit time")
    early = [f"{w} {x}" for w, x in overturned if x and at <= parse_iso(x, w)]
    if early:
        refuse(f"PREREG_25 rev {n} ({h[:12]}, {t}) is not later than {early}")
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
        rep = read_json_or_none(os.path.join(ctx["rank25"], "t25_apply_report.json"))
        nu = read_json_or_none(os.path.join(ctx["R25"], "not_used.json"))
        decided = [(n, r.get("created")) for n, r in (("t25_apply_report.json", rep), ("not_used.json", nu)) if r]
        if decided:
            if a.bug_revision is None:
                refuse(f"{[n for n, _ in decided]} hold the registered decision; a rerun needs a PREREG_25 revision "
                       f"naming a program error (--bug-revision N, N >= {P25_MIN_REV})")
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
        (t25_apply_report.json / not_used.json) unless this run passed the rerun gate."""
        ctx = self.ctx
        decided = [p for p in (os.path.join(ctx["rank25"], "t25_apply_report.json"),
                               os.path.join(ctx["R25"], "not_used.json")) if os.path.exists(p)]
        if decided and not self.cleared:
            raise T25Error(f"{e} -- not recorded: {[shown(p) for p in decided]} already hold the registered decision "
                           f"and this run did not pass the rerun gate (nothing was written)", 2, e.stage, e.reason,
                           e.details)
        rec = {"format": FORMATS["verdict"], "mode": ctx["mode"], "created": now_iso(), "exit_code": e.code,
               "stage": e.stage or self.stage, "reason": e.reason or str(e).splitlines()[0][:200], "message": str(e),
               "details": e.details, "checks": self.pre, "records": self.rec,
               "rerun": {"reason": self.a.reason, "replaced": self.replaced, "bug_revision": self.bug},
               "tool": tool_record(), "seconds": round(time.time() - self.t0, 1)}
        if e.code == 3:
            write_json(ctx, os.path.join(ctx["R25"], "not_used.json"), dict(rec, format=FORMATS["not_used"]))
        else:
            write_json(ctx, os.path.join(ctx["rank25"], "attempt_verdict.json"), rec)

    def go(self):
        ctx, a = self.ctx, self.a
        existing = [p for p in apply_outputs(ctx) if os.path.exists(p)]
        if existing and not (a.rerun and a.reason):       # before the prerequisites (as t19)
            refuse(f"outputs already exist ({[os.path.basename(p) for p in existing][:6]}); never overwritten without "
                   f"--rerun --reason TEXT")
        fc25, models25, fc19, models19, rep19, readers, self.pre = prereqs(ctx, a)
        fc9, mp, fc07, fc04, model07 = readers
        self.rerun_gate()
        # ---- step 1: 08's three P1 sets (09's reader and checks), equal to the ones 19 recorded
        self.stage = "p1_sets"
        rep8 = read_json(os.path.join(ctx["rank"], "t08_apply_report.json"))
        ctx9 = dict(ctx, mode=rep19["mode"], real=rep19["mode"] == "real")   # 08's sets of the run 19 bound
        a9 = t9.Apply(ctx9, a, self.t0)
        p1 = call(a9.p1_sets, rep8)
        self.rec["p1_sets"] = a9.rec["p1_sets"]
        for s in SETS:
            want = (rep19.get("p1_sets") or {}).get(s) or {}
            got = self.rec["p1_sets"][s]
            if (want.get("report_sha256"), want.get("npz_sha256")) != (got["report_sha256"], got["npz_sha256"]):
                refuse(f"{s}: 08's P1 set differs from the one 19's report records", reason="p1_sets")
        self.clear()
        # ---- step 2: Q09 recomputed on public / private (test02 / 04 / 07 inputs, 09's frozen map)
        self.stage = "q09_vs_09"
        qfiles = {s: os.path.join(ctx["T"], t2.TEST_FILES[s]) for s in TEST_SETS}
        opening = load_opening(ctx, qfiles)
        vk = {s: rf.load_val_keys(os.path.join(ctx["D2"], "rank", s, "dumps")) for s in TEST_SETS}
        sw = {s: call(t7.load_sw, ctx, s, vk[s]) for s in TEST_SETS}
        o4z = {s: load_npz(os.path.join(ctx["R4"], "opp", f"o4_{s}.npz")) for s in TEST_SETS}
        c5w = fc04["source_sha256"]["C5"]["weights_sha256"]
        fz = {}
        for s in TEST_SETS:
            ddir = os.path.join(ctx["D2"], "rank", s, "dumps")
            c5 = call(t7.t4call, t4.load_c5, s, ddir, vk[s], c5w)
            f = call(t7.t4call, t4.fuse_set, s, vk[s], qfiles[s], c5, opening, o4z[s], sw[s], fc04, False)
            f["c5_sha256"] = c5["sha256"]
            f.update(t7.port_c5(vk[s], c5, fc04, f))
            fz[s] = f
        if not all(fz[s]["port_equal"] for s in TEST_SETS):
            drift("the ported C5 lines do not give fuse_set's Q02 bit for bit", "q09_vs_09", "port_c5",
                  {s: fz[s]["port_equal"] for s in TEST_SETS})
        PF = pf7.import_pilot_fuse()
        hs = {s: load_npz(os.path.join(ctx["R7"], "hs", f"{s}_hs.npz")) for s in TEST_SETS}
        h, q9, us = {}, {}, {}
        E19, lp19, has19, E25, lp25, has25 = {}, {}, {}, {}, {}, {}
        for s in TEST_SETS:
            z = p1[s]["z"]
            keys_ok = (list(map(str, z["question_id"])) == vk[s]["question_id"]
                       and [int(x) for x in z["slot"]] == vk[s]["slot"] and list(map(str, z["color"])) == vk[s]["color"]
                       and list(map(str, hs[s]["question_id"])) == vk[s]["question_id"]
                       and [int(x) for x in hs[s]["slot"]] == vk[s]["slot"])
            if not keys_ok:
                refuse(f"{s}: 08's {s}_p1.npz / 07's {s}_hs.npz keys differ from test02's val_keys.npz", reason="keys")
            h[s] = call(t7.q02h_set, fz[s], vk[s], hs[s], model07, fc07, PF)
            lpE, okE = t9.set_features(p1[s], mp)
            us[s] = np.asarray(vk[s]["parsed"], dtype=bool) & okE
            q9[s] = call(t8.q08_core, fz[s], vk[s], h[s], lpE, okE, us[s], fc9)
            E19[s], lp19[s], has19[s] = call(t19.set_term_of, fz[s], vk[s], p1[s], us[s], models19, mp)
            E25[s], lp25[s], has25[s] = call(t19.set_term_of, fz[s], vk[s], p1[s], us[s], models25, mp)
        cmp9 = rows_compare(q9["public"]["rows"] + q9["private"]["rows"], os.path.join(ctx["rank9"],
                                                                                        "rank_test_probs.csv"))
        lp9 = load_npz(os.path.join(ctx["rank9"], "test_lp_p1e.npz"))
        lp_all = np.concatenate([q9[s]["lp"] for s in TEST_SETS])
        cmp9.update(p1_ok_equal_t09=bool(np.array_equal(np.asarray(lp9["p1_ok"], dtype=bool),
                                                        np.concatenate([q9[s]["p1_ok"] for s in TEST_SETS]))),
                    usable_equal_t09=bool(np.array_equal(np.asarray(lp9["usable"], dtype=bool),
                                                         np.concatenate([us[s] for s in TEST_SETS]))),
                    lp_p1e_max_abs_diff_vs_t09=float(np.max(np.abs(np.asarray(lp9["lp"]) - lp_all)))
                    if np.asarray(lp9["lp"]).shape == lp_all.shape and len(lp_all) else None)
        cmp9["lp_p1e_equal"] = bool(cmp9["lp_p1e_max_abs_diff_vs_t09"] is not None
                                    and cmp9["lp_p1e_max_abs_diff_vs_t09"] <= TOL)
        self.rec["q09_vs_09"] = cmp9
        print(f"Q09 recomputed vs test09: byte equal {cmp9['byte_equal']}, max |dP| {cmp9['max_abs_diff']:.3g}, "
              f"rule A / B differences {cmp9['rule_a_differences']} / {cmp9['rule_b_differences']}, usable equal "
              f"{cmp9['p1_ok_equal_t09']} / {cmp9['usable_equal_t09']}, lp_P1TO_E max |d| "
              f"{cmp9['lp_p1e_max_abs_diff_vs_t09']}", flush=True)
        if not (cmp9["equal"] and cmp9["p1_ok_equal_t09"] and cmp9["usable_equal_t09"] and cmp9["lp_p1e_equal"]):
            drift("Q09 recomputed differs from results/test09/rank/rank_test_probs.csv (or the usable games / "
                  "lp_P1TO_E differ; a local problem, PREREG_25 4.2 step 2)", "q09_vs_09", "q09_vs_09", cmp9)
        # ---- step 3: Q19 recomputed (frozen S19, 19's theta) equal to 19's official run
        self.stage = "q19_vs_19"
        res19 = {s: call(t19.q19_core, fz[s], vk[s], h[s], q9[s], E19[s], has19[s], fc19) for s in TEST_SETS}
        cmp19 = rows_compare(res19["public"]["rows"] + res19["private"]["rows"],
                             os.path.join(ctx["rank19"], "rank_test_probs.csv"))
        mine19 = {"question_id": fz["public"]["qids"] + fz["private"]["qids"],
                  "lp": np.concatenate([lp19[s] for s in TEST_SETS]),
                  "has": np.concatenate([res19[s]["has"] for s in TEST_SETS]),
                  "usable": np.concatenate([us[s] for s in TEST_SETS])}
        cmp19["lp_set19"] = compare_lp(mine19, load_npz(os.path.join(ctx["rank19"], "test_lp_set.npz")))
        self.rec["q19_vs_19"] = cmp19
        print(f"Q19 recomputed vs test19: byte equal {cmp19['byte_equal']}, max |dP| {cmp19['max_abs_diff']:.3g}, "
              f"rule A / B differences {cmp19['rule_a_differences']} / {cmp19['rule_b_differences']}, lp_SET19 "
              f"{cmp19['lp_set19']}", flush=True)
        if not (cmp19["equal"] and cmp19["lp_set19"]["equal"]):
            drift("Q19 recomputed differs from results/test19/rank (rank_test_probs.csv or test_lp_set.npz; a local "
                  "problem, PREREG_25 4.2 step 3)", "q19_vs_19", "q19_vs_19", cmp19)
        # ---- step 4: the SET25 GPU seam (record only)
        self.stage = "set_seam"
        vrows, val = call(t9.val_seam_chunks)
        sseam = call(t19.set_seam_record, p1["seam"], val, models25, mp)
        sseam.update(format=FORMATS["set_seam"], mode=ctx["mode"], created=now_iso(), models="S25",
                     seam_p1_report_sha256=self.rec["p1_sets"]["seam"]["report_sha256"],
                     reference={"val_chunks": vrows, "val_report.json": sha_lf(t9.VAL_REPORT)})
        sseam["note"] = sseam["note"].replace("PREREG_19 4.2 step 2", "PREREG_25 4.2 step 4, as PREREG_19 4.2 step 2")
        set_seam_path = write_json(ctx, os.path.join(ctx["R25"], "set_seam.json"), sseam)
        self.rec["set_seam_record_only"] = {k: sseam.get(k) for k in ("max_abs_lp_set", "usable_games",
                                                                      "bitwise_equal_means",
                                                                      "games_lp_above_review_line")}
        print(f"SET25 GPU seam (record only): max |d lp_SET| {sseam.get('max_abs_lp_set')}", flush=True)
        # ---- step 5: the SET25 direction
        self.stage = "set_direction"
        res = {s: call(t19.q19_core, fz[s], vk[s], h[s], q9[s], E25[s], has25[s], fc25) for s in TEST_SETS}
        direc = {s: pf7.direction_record(res[s]["rho"], res[s]["rho_n"], RHO_MONO, RHO_FLOOR) for s in TEST_SETS}
        self.rec["set_direction"] = direc
        bad = [s for s in TEST_SETS if not direc[s]["passed"]]
        if bad:
            not_used(f"SET25 direction check failed on {bad} ({ {s: direc[s]['rho'] for s in TEST_SETS} } < "
                     f"{direc['public']['threshold']}): 25 is not used (PREREG_25 4.2 step 5)", "set_direction",
                     "set_direction", direc)
        # ---- the outputs
        self.stage = "outputs"
        rank = ctx["rank25"]
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
                 lp=np.concatenate([lp25[s] for s in TEST_SETS]),
                 has=np.concatenate([res[s]["has"] for s in TEST_SETS]),
                 usable=np.concatenate([us[s] for s in TEST_SETS]), scale=np.array(SF25.SCALE),
                 models_sha256=np.array([SF25.MODEL_SHA256[x] for x in SF25.SEEDS]),
                 map_sha256=np.array(t9.MAP_SHA))
        self.stage = "check_submission"
        check_q = [qfiles["public"], qfiles["private"]]
        errors = check_submission.check("rank", sub, check_q, official=True)
        if errors:
            removed = []
            for n in TEST_OUTPUTS:
                p = os.path.join(rank, n)
                if os.path.exists(p):
                    guard_write(ctx, p)
                    os.remove(p)
                    removed.append(n)
            drift(f"check_submission --official failed on the Q25 file: {errors[:5]}; {removed} removed",
                  "check_submission", "check_submission", {"errors": errors[:50], "removed": removed})
        # ---- the report
        preds = dict(t2.read_pred(sub))
        pub_ids, prv_ids = set(fz["public"]["qids"]), set(fz["private"]["qids"])
        diffs = {}
        for name, other in (("19", os.path.join(ctx["rank19"], "submission_rank.csv")),
                            ("09", os.path.join(ctx["rank9"], "submission_rank.csv")),
                            ("08", os.path.join(ctx["rank"], "submission_rank.csv")),
                            ("07", os.path.join(ctx["R7"], "rank", "submission_rank.csv")),
                            ("04", os.path.join(ctx["R4"], "rank", "submission_rank.csv")),
                            ("R20", os.path.join(ctx["O2"], "submission_rank.csv"))):
            if os.path.exists(other):
                op_ = dict(t2.read_pred(other))
                diffs[name] = {"public": sum(1 for q in pub_ids if op_.get(q) != preds.get(q)),
                               "private": sum(1 for q in prv_ids if op_.get(q) != preds.get(q)),
                               "all": sum(1 for q, v in preds.items() if op_.get(q) != v),
                               "file_sha256": sha_raw(other)}
            else:
                diffs[name] = None
        bound = self.inputs_bound(set_seam_path, qfiles)
        rep = {"format": FORMATS["apply"], "what": "test25 competition 2 apply (PREREG_25 4.2): frozen Q25 per set, "
                                                   "rule B, submission_rank.csv = pred_B.csv (public then private)",
               "mode": ctx["mode"], "created": now_iso(), "exit_code": 0, "usable": True, "unusable_reasons": [],
               "combo": COMBO, "method": METHOD, "theta": fc25["params"]["theta"],
               "models": {str(x): SF25.MODEL_SHA256[x] for x in SF25.SEEDS},
               "t1_verdict": self.pre["t19_apply_report"]["t1_verdict"],
               "test19": {"t19_apply_report_sha256": pin_sha(os.path.join(ctx["rank19"], "t19_apply_report.json")),
                          "exit_code": rep19.get("exit_code"), "usable": rep19.get("usable"),
                          "submission_sha256": (rep19.get("outputs_sha256") or {}).get("submission_rank.csv"),
                          "inputs_bound_checked": len(rep19.get("inputs_bound") or {})},
               "p1_sets": self.rec["p1_sets"],
               "earliest_p1_chunk_created_utc": min(self.rec["p1_sets"][s]["earliest_chunk_created_utc"] or ""
                                                    for s in SETS),
               "q09_vs_09": self.rec["q09_vs_09"], "q19_vs_19": self.rec["q19_vs_19"],
               "set_seam_record_only": self.rec["set_seam_record_only"],
               "set_direction": direc, "rho_mono": RHO_MONO, "rho_floor": RHO_FLOOR,
               "p1_failures": rep19.get("p1_failures"),
               "sets": {s: dict(res[s]["info"], rule_B_differs_from_Q19=int(
                   ((rf.rule_b_np(np.where(res[s]["ok"][:, None], res[s]["P"], 0.0)) !=
                     rf.rule_b_np(np.where(res19[s]["ok"][:, None], res19[s]["P"], 0.0))) & res[s]["ok"]).sum()))
                        for s in TEST_SETS},
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
               "paths": {k: ctx[k] for k in ("R25", "R19", "R9", "R8", "H", "R7", "R4", "D2", "O2", "T", "C")},
               "tool": tool_record(), "environment": t4.environment(), "git_head": git_head(),
               "data_local": "never opened, listed or hashed (PREREG_25 4.2: no seam_val, no 09-04 prerequisite chain)",
               "record_only": "level_summary, rule_b_differences, the SET25 GPU seam and the byte equality of Q09 / Q19 "
                              "are recorded; nothing is chosen from them"}
        rep["seconds"] = round(time.time() - self.t0, 1)
        write_json(ctx, os.path.join(rank, "t25_apply_report.json"), rep)
        print(f"apply Q25: public {len(fz['public']['qids'])}, private {len(fz['private']['qids'])} questions -> "
              f"{shown(sub)} (sha256 {rep['outputs_sha256']['submission_rank.csv'][:12]}); check_submission official: "
              f"OK; rule-B differences vs 19 public / private: {(diffs.get('19') or {}).get('public')} / "
              f"{(diffs.get('19') or {}).get('private')}")
        print("下一步：commit results/test25/，照 docs/TESTDAY_25.md 第 4 步 official_log.py prepare"
              if ctx["real"] else f"({ctx['mode']}: nothing is uploaded)")
        return 0

    def inputs_bound(self, set_seam_path, qfiles):
        """The files official_log re-hashes (keys: the real-mode repository paths)."""
        ctx = self.ctx
        R19, R9, R8 = ctx["R19"], ctx["R9"], ctx["R8"]
        items = [("results/test19/rank/t19_apply_report.json", os.path.join(R19, "rank", "t19_apply_report.json")),
                 ("results/test19/rank/rank_test_probs.csv", os.path.join(R19, "rank", "rank_test_probs.csv")),
                 ("results/test19/rank/submission_rank.csv", os.path.join(R19, "rank", "submission_rank.csv")),
                 ("results/test19/rank/test_lp_set.npz", os.path.join(R19, "rank", "test_lp_set.npz")),
                 ("results/test19/set_seam.json", os.path.join(R19, "set_seam.json")),
                 ("results/test19/rank/seam_val/seam_local.json", os.path.join(R19, "rank", "seam_val",
                                                                               "seam_local.json")),
                 ("results/test09/rank/t09_apply_report.json", os.path.join(R9, "rank", "t09_apply_report.json")),
                 ("results/test09/rank/rank_test_probs.csv", os.path.join(R9, "rank", "rank_test_probs.csv")),
                 ("results/test09/rank/test_lp_p1e.npz", os.path.join(R9, "rank", "test_lp_p1e.npz")),
                 ("results/test08/rank/t08_apply_report.json", os.path.join(R8, "rank", "t08_apply_report.json")),
                 ("results/test02/rank/apply_test_report.json", os.path.join(ctx["O2"], "apply_test_report.json")),
                 ("results/test02/rank/opening_test.npz", os.path.join(ctx["O2"], "opening_test.npz")),
                 ("results/test02/checks/rank/t1.json", os.path.join(ctx["C"], "rank", "t1.json")),
                 ("results/test25/set_seam.json", set_seam_path)]
        items += [(r_, path_of(r_)) for r_ in P25_PINS]
        items += [(r_, path_of(r_)) for r_ in t19.P19_PINS]
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
    except T25Error as e:
        if e.code in (3, 4):
            run.write_verdict(e)
            if e.code == 3:
                print(f"25 不用（PREREG_25 4.2）：{e}\ncommit {shown(ctx['R25'])}；最終照 PREREG_19 4.3（不重跑）",
                      file=sys.stderr)
        raise


# ================================================================ status (read only)
def cmd_status(a):
    ctx = resolve(a, "status")
    lines = [f"test25 status ({ctx['mode']}; {ctx['R25']})"]
    rp = os.path.join(ctx["rank25"], "t25_apply_report.json")
    vp = os.path.join(ctx["rank25"], "attempt_verdict.json")
    np_ = os.path.join(ctx["R25"], "not_used.json")
    if os.path.exists(rp):
        r = read_json(rp)
        lines.append(f"apply: exit {r.get('exit_code')}; submission sha256 "
                     f"{str((r.get('outputs_sha256') or {}).get('submission_rank.csv'))[:12]}; Q09 = 09 "
                     f"{(r.get('q09_vs_09') or {}).get('equal')}; Q19 = 19 {(r.get('q19_vs_19') or {}).get('equal')}"
                     + (f", committed {committed_clean(rp)}" if ctx["real"] else ""))
    elif os.path.exists(np_):
        r = read_json(np_)
        lines.append(f"25 not used: {r.get('stage')} ({r.get('reason')})")
    elif os.path.exists(vp):
        r = read_json(vp)
        lines.append(f"apply: attempt_verdict exit {r.get('exit_code')} at {r.get('stage')} ({r.get('reason')})")
    else:
        lines.append("apply: not run")
    if ctx.get("rank19"):
        r19 = read_json_or_none(os.path.join(ctx["rank19"], "t19_apply_report.json"))
        lines.append("test19 apply: " + (f"exit {r19.get('exit_code')}, usable {r19.get('usable')}" if r19 else
                                         "not run"))
    left = K6_25 - now()
    lines.append(f"K6-25 {K6_25.isoformat()}: " + (f"{left.days} days {left.seconds // 3600} h left"
                                                   if left.total_seconds() > 0 else "passed"))
    print("\n".join(lines))
    return 0


# ================================================================ command line
def build_parser():
    ap = argparse.ArgumentParser(prog="t25_rank.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("apply", "status"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--results-dir")
        for flag in DIR_FLAGS.values():
            p.add_argument(flag)
        p.add_argument("--now", help="drills only: the current time (K6-25)")
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
    except T25Error as e:
        print(f"{'REFUSED' if e.code == 2 else 'NOT USED' if e.code == 3 else 'DRIFT'} (exit {e.code}): {e}",
              file=sys.stderr)
        return e.code
    except Exception:  # noqa: BLE001 - an internal error is a refusal with its traceback, never a decision
        traceback.print_exc()
        print("ERROR (exit 2): unexpected internal error of tools/t25_rank.py; stop and check the local tool",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
