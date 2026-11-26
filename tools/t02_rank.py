"""C10 test02, competition 2 (rank, TEAM_11248): the test-day local tool.

Registration: docs/PREREG_02_OPENING_FUSION.md section 9 (rev 5). Spec: results/test02/design/spec.md
sections 3.0 and 3.2 (the registration wins over the spec; the spec wins over the design notes).

    PY tools/t02_rank.py opening      [--dry-run|--smoke|--rehearsal] [--results-dir D] [--tests-dir T]
                                      [--out-dir O] [--checks-dir C] [--rerun --reason TEXT]
    PY tools/t02_rank.py apply        [same flags] [--fallback R10 --reason TEXT] [--fallback-file F]
    PY tools/t02_rank.py fallback-fit --combo R20|R10 [--out-dir O] [--rerun --reason TEXT]

D  the test02 notebook's result root; the rank notebook's files are in D/rank/ (spec 2.3)
T  the official test files as checked by T1 (normalized copies when T1 said NORMALIZED) + MANIFEST.json
O  where this tool writes; C  T1 results (default D/checks; read: C/rank/t1.json)
Real mode (no mode flag): D = results/test02, T = data/test, O = results/test02/rank exactly.
--dry-run (the fake tree of t02_fake_tests.py from-val), --rehearsal (real Colab output of a rehearsal
zip), --smoke (the notebook's CPU SMOKE output): D, T and O must be given; O outside the repository.

opening   The frozen opening n-gram (M = 12, alpha = 20; p02_rank_opening.OpeningModel, counts from
          c10data.load_rows(part="fit") only) scores every slot of the seam set (the local validation
          question file), public and private. The seam part's ll, o, has, n_terms must be bitwise equal
          to the committed results/prereg02/rank/opening_val.npz (this replaces p02_rank_fuse.py's
          build_id check, which can never hold for a new build). Writes O/opening_test.npz (set,
          question_id, slot, color, ll, o, has, n_terms, n_moves, meta) and O/opening_test_report.json.
apply     The frozen R20 (results/prereg02/rank/frozen_config.json) on each set's dumps, in the order
          seam_val, public, private, by calling the frozen functions of p02_rank_fuse (the numeric core of
          its cmd_apply). The three sets' val_p02e9_d4 dumps must carry the same gpu_name / torch / cudnn in
          their _DONE markers (a resumed notebook run reuses finished sets). Seam first:
          O/seam_val/rank_val_probs.csv + decide.py; seam_local.json passes iff max |dP| against
          results/prereg02/rank/rank_val_probs.csv <= tol and at most 4 rule-B differences against
          results/prereg02/rank/pred_B.csv (registered, rev 5 item 2); tol is derived here from the markers
          (1e-4 when the seam_val dump's GPU equals the prereg02 dump's GPU, else 1e-3) and must equal the
          notebook's seam_check.json tol (smoke: the notebook's value, recorded only). A failed seam writes no
          test output. Then O/rank_test_probs.csv (public rows in question-file order, then
          private; 800 rows), decide.py -> pred_A.csv, pred_B.csv, decide_summary.json in O,
          submission_rank.csv = byte copy of pred_B.csv, check_submission --official, apply_test_report.json.
          --fallback R10 (only when T1 says FAIL_OPENING): the test sets use results/test02/rank/
          fallback_R10.json (no opening term); the seam still uses the frozen R20, because the registered
          local seam is R20 on the validation questions, whose opening features are not in question. Real
          mode uses only that committed file (no --fallback-file), committed before 2026-11-04 11:00 +08:00.
fallback-fit  Before 11/04, on the committed prereg02 inputs only (inner pseudo-questions, inner dumps,
          opening_inner.npz; no OOF, no evaluation, no validation input): the fit-round preparation of
          p02_rank_fuse.cmd_tune, then its refit on all fit rounds. --combo R20 is a self-check against the
          frozen parameters (<= 1e-12) -> O/fallback_selfcheck_R20.json. --combo R10 repeats that self-check
          and writes O/fallback_R10.json (C5, log pooling, calibrated, no opening).

Real mode refuses unless: frozen_config.json is committed and equals HEAD; the frozen tool files hash to
frozen_config's tool.files_sha256 and are clean; this file, tools/check_submission.py and
colab/test02_rank.ipynb are committed and clean; the rank notebook outputs of spec 2.3 are tracked and unchanged;
the run's notebook_sha256 is the one of that committed notebook (rehearsal: warning only); run_log status is
"completed" and no earlier run in run_log failed (seam_failed / repro_failed / readonly_changed /
final_check_failed: a run again in the same Drive folder is not the registered rerun); seam_check.json passed;
config.json says smoke false and rehearsal false; data/test files and MANIFEST.json hash to config.json's
test_bundle record; C/rank/t1.json is the full T1 on the raw files (complete, stage raw, mode real) with verdict
PASS or NORMALIZED (FAIL_OPENING only with --fallback R10; opening accepts it too). Outputs are never
overwritten without --rerun --reason (the
reason and the replaced files' SHA-256 go into the report). Reports record git HEAD, every input and output
SHA-256, this tool's SHA-256 and the Python/numpy/scipy versions.

Team separation (PREREG_02 section 3): imports only rank tools, decide, c10data and check_submission; never
a player module. No validation label file is ever opened (paths whose name contains that word are refused).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SELF = os.path.join(HERE, "t02_rank.py")
sys.path.insert(0, HERE)
import c10data  # noqa: E402
import check_submission  # noqa: E402
import decide  # noqa: E402
import p02_rank_fuse as rf  # noqa: E402
import p02_rank_opening as ro  # noqa: E402
import p02_rank_pseudo as rp  # noqa: E402

# ------------------------------------------------------------------ fixed inputs (module constants so that
# the unit tests can point them at synthetic data; the real run never changes them)
P02_RANK = os.path.join(REPO, "results", "prereg02", "rank")
P02_DUMPS = os.path.join(P02_RANK, "dumps")
SEAM_QUESTIONS = os.path.join(REPO, "data", "local", "rank_val_questions.csv")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
LOCAL_DIR = os.path.join(REPO, "data", "local")
REAL_RESULTS = os.path.join(REPO, "results", "test02")
REAL_TESTS = os.path.join(REPO, "data", "test")
REAL_OUT = os.path.join(REAL_RESULTS, "rank")
FALLBACK_FILE = os.path.join(REAL_OUT, "fallback_R10.json")

# SHA-256 of the committed prereg02 files this tool reads (spec 3.2); None disables a pin (tests only)
PINS = {
    "frozen_config.json": "44a16e09679cc6815d41d9a1255882f4a38599cd97cbf66996d447709b4f2efa",
    "opening_val.npz": "75ffec64f9e949ac0be52ce9e72a4a19bb978a25de2b6ed9d355aac7a1a5ae85",
    "rank_val_probs.csv": "a83700432fce5d7d26653fdaee345b0c894729fdff0029a244ed94e06c803b63",
    "pred_B.csv": "863a62582424adddc9c2e1821fcc056008bd651e103c19ec8ada98a14f87a2fd",
}
FROZEN_OPENING_BUILD_ID = "c0cb86b47cd3f95c75940cc71b3b7ca24d43c3390c1a510bf96c6ae68e0f93a6"
EXPECTED_FIT_GAMES = c10data.EXPECTED["fit"]["games"]

TEST_FILES = {"public": "rank_prediction_test_public.csv", "private": "rank_prediction_test_private.csv"}
SETS = ("seam_val", "public", "private")
DUMP_NAME = "p02e9_d4"
NOTEBOOK_FILES = ("config.json", "run_log.json", "env.json", "timings.json", "seam_check.json",
                  "readonly_before.json", "readonly_after.json")
SET_FILES = ("val_keys.npz", "val_%s.npz" % DUMP_NAME, "_DONE_val_keys.json", "_DONE_val_%s.json" % DUMP_NAME,
             "_INPUT.json")
R20 = {"id": "R20", "source": "C5", "pool": "log", "calibrated": True, "opening": True}
R10 = {"id": "R10", "source": "C5", "pool": "log", "calibrated": True, "opening": False}
T1_OK = ("PASS", "NORMALIZED")
FAILED_RUN_STATUSES = ("seam_failed", "repro_failed", "readonly_changed", "final_check_failed")
SEAM_MAX_B_DIFF, SEAM_QUESTIONS_FULL = 4, 1589
PARAM_TOL = 1e-12            # b = B beta, and the R20 self-check against the frozen parameters
OPENING_FORMAT = "c10-test02 rank opening_test v1"
FALLBACK_FORMAT = "c10-test02 rank fallback_R10 v1"
SELFCHECK_FORMAT = "c10-test02 rank fallback_selfcheck_R20 v1"


class T02Error(RuntimeError):
    pass


ERRORS = (T02Error, rp.P02Error, decide.DecideError)


# ================================================================ small helpers
def now_iso():
    return rp.now_iso()


def sha256_file(path):
    return rp.sha256_file(path)


def sha256_text(path):
    """SHA-256 of a source file with CRLF folded to LF (core.autocrlf=true; same rule as the frozen tools)."""
    return rf.sha256_text(path)


def _real(p):
    return os.path.normcase(os.path.realpath(p))


def same_path(a, b):
    return _real(a) == _real(b)


def is_inside(path, root):
    return rp.is_inside(path, root)


_NEEDLE = "ans" + "wer"


def refuse_needle(path):
    """Refuse any path whose name looks like a validation label file (the rank tool's own copy of the rule)."""
    if _NEEDLE in os.path.basename(os.path.normpath(os.fspath(path))).lower():
        raise T02Error(f"refusing {path}: validation label files are never read by the test-day tools")
    return path


def read_json(path):
    refuse_needle(path)
    if not os.path.exists(path):
        raise T02Error(f"missing {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    rp.write_json(path, obj)


def git(*args):
    try:
        r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    return r.returncode, r.stdout.strip()


def git_head():
    rc, h = git("rev-parse", "HEAD")
    return h if rc == 0 else None


def repo_rel(path):
    return os.path.relpath(os.path.realpath(path), os.path.realpath(REPO)).replace(os.sep, "/")


def shown(path):
    """Repo-relative path inside the repository, else absolute (for reports)."""
    return repo_rel(path) if is_inside(path, REPO) else os.path.abspath(path)


def tracked(path):
    if not is_inside(path, REPO):
        return False
    rc, _ = git("ls-files", "--error-unmatch", "--", repo_rel(path))
    return rc == 0


def require_committed_clean(paths, what):
    """Every path exists, is tracked by git and is unchanged against HEAD (index and working tree)."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise T02Error(f"{what}: missing {[shown(p) for p in missing][:5]}")
    outside = [p for p in paths if not is_inside(p, REPO)]
    if outside:
        raise T02Error(f"{what}: outside the repository: {outside[:3]}")
    rels = [repo_rel(p) for p in paths]
    rc, _ = git("ls-files", "--error-unmatch", "--", *rels)
    if rc != 0:
        raise T02Error(f"{what}: not all tracked by git; commit them first ({rels[:3]}...)")
    rc, st = git("status", "--porcelain", "--", *rels)
    if rc != 0 or st:
        raise T02Error(f"{what}: changed against HEAD: {st[:300]}")
    return {"files": len(rels), "rule": "tracked by git and unchanged against HEAD", "git_head": git_head()}


def environment():
    import scipy
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__}


def tool_record():
    return {"git_head": git_head(), "t02_rank.py_sha256": sha256_text(SELF),
            "t02_rank.py_tracked": tracked(SELF),
            "check_submission.py_sha256": sha256_text(os.path.abspath(check_submission.__file__)),
            "sha256_rule": "sha256 of the file with CRLF folded to LF",
            "frozen_tools": rf.tool_state()}


NOTEBOOK = os.path.join(REPO, "colab", "test02_rank.ipynb")
_NB_HASH_RE = re.compile(r"NOTEBOOK_SHA256 = '([0-9a-f]{64})'")


def notebook_source_sha256(path):
    """(recorded value, recomputed value) of a test02 notebook's notebook_sha256 (rule: sha256 of
    json.dumps([cell sources], ensure_ascii=False) with the recorded value replaced by 64 zeros)."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    sources = ["".join(c["source"]) for c in nb["cells"]]
    found = sorted({m for s in sources for m in _NB_HASH_RE.findall(s)})
    if len(found) != 1:
        return None, None
    zeroed = [s.replace("NOTEBOOK_SHA256 = '%s'" % found[0], "NOTEBOOK_SHA256 = '%s'" % ("0" * 64)) for s in sources]
    return found[0], hashlib.sha256(json.dumps(zeroed, ensure_ascii=False).encode("utf-8")).hexdigest()


def check_notebook_identity(cfg, mode):
    """Is the Colab run's notebook_sha256 the one of the committed colab/test02_rank.ipynb? Real mode refuses
    otherwise (as competition 1 does; a notebook left on Drive from before a fix would pass unnoticed), a
    rehearsal warns, dry runs and smoke only record. Returns True / False, or None when the repo notebook is absent
    outside real mode."""
    if not os.path.exists(NOTEBOOK):
        if mode == "real":
            raise T02Error(f"{NOTEBOOK} is missing: real mode checks the Colab run against the committed notebook")
        return None
    recorded, recomputed = notebook_source_sha256(NOTEBOOK)
    same = recomputed is not None and recorded == recomputed and cfg.get("notebook_sha256") == recomputed
    if not same and mode == "real":
        raise T02Error(f"the notebook run's notebook_sha256 {str(cfg.get('notebook_sha256'))[:12]} is not the one of "
                       f"the committed colab/test02_rank.ipynb ({str(recomputed)[:12]}): the Colab run used another "
                       f"version of the notebook; its outputs are not used")
    if not same and mode == "rehearsal":
        print(f"WARNING: the notebook run's notebook_sha256 {str(cfg.get('notebook_sha256'))[:12]} is not the one "
              f"of the repository's colab/test02_rank.ipynb ({str(recomputed)[:12]}); recorded in the report",
              file=sys.stderr)
    return same


def readonly_dirs():
    return (os.path.join(REPO, "results", "prereg01"), os.path.join(REPO, "results", "prereg02"), LOCAL_DIR,
            P02_RANK)


def check_out_dir(out_dir, mode):
    for ro_dir in readonly_dirs():
        if is_inside(out_dir, ro_dir):
            raise T02Error(f"refusing to write into {out_dir}: it is inside the read-only {ro_dir}")
    if mode != "real" and is_inside(out_dir, REPO):
        raise T02Error(f"--{mode} output must be outside the repository (got {out_dir}); use a scratch directory")


TEXT_PIN_EXT = (".json", ".csv")


def pin_sha256(path):
    """The SHA-256 a pin is compared with: .json / .csv with CRLF folded to LF (a checkout with core.autocrlf=true
    may turn them into CRLF; the pins are the LF bytes), anything else (.npz) raw bytes."""
    return sha256_text(path) if path.lower().endswith(TEXT_PIN_EXT) else sha256_file(path)


def pinned(path, name):
    """SHA-256 of a committed prereg02 input, checked against the pin (text files CRLF-folded, see pin_sha256)."""
    refuse_needle(path)
    if not os.path.exists(path):
        raise T02Error(f"missing input {path}")
    h = pin_sha256(path)
    want = PINS.get(name)
    if want is not None and h != want:
        raise T02Error(f"{shown(path)}: sha256 {h} is not the pinned {want}; the prereg02 files must not change")
    return h


def remove_outputs(paths, a):
    """Refuse existing outputs unless --rerun --reason; with it, delete them first and record their SHA-256."""
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not a.rerun:
        raise T02Error(f"outputs already exist ({[os.path.basename(p) for p in existing][:6]}); they are never "
                       f"overwritten without --rerun --reason TEXT")
    replaced = {shown(p): sha256_file(p) for p in existing}
    for p in existing:
        os.remove(p)
    return {"rerun": bool(a.rerun), "reason": a.reason if a.rerun else None, "replaced_sha256": replaced}


def atomic_copy(src, dst):
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


# ================================================================ mode, notebook records, test files, T1
def resolve(a):
    flags = [m for m, on in (("dry-run", a.dry_run), ("smoke", a.smoke), ("rehearsal", a.rehearsal)) if on]
    if len(flags) > 1:
        raise T02Error(f"give at most one of --dry-run, --smoke, --rehearsal (got {flags})")
    mode = flags[0] if flags else "real"
    if mode == "real":
        rd, td, od = (a.results_dir or REAL_RESULTS), (a.tests_dir or REAL_TESTS), (a.out_dir or REAL_OUT)
        for got, want, name in ((rd, REAL_RESULTS, "--results-dir"), (td, REAL_TESTS, "--tests-dir"),
                                (od, REAL_OUT, "--out-dir")):
            if not same_path(got, want):
                raise T02Error(f"real mode: {name} must be {want} (got {got}); other trees need --dry-run, "
                               f"--smoke or --rehearsal")
    else:
        missing = [n for n, v in (("--results-dir", a.results_dir), ("--tests-dir", a.tests_dir),
                                  ("--out-dir", a.out_dir)) if not v]
        if missing:
            raise T02Error(f"--{mode} needs {', '.join(missing)}")
        rd, td, od = a.results_dir, a.tests_dir, a.out_dir
    check_out_dir(od, mode)
    cd = a.checks_dir or os.path.join(rd, "checks")
    if mode == "real" and not same_path(cd, os.path.join(REAL_RESULTS, "checks")):
        raise T02Error(f"real mode: --checks-dir must be {os.path.join(REAL_RESULTS, 'checks')} (got {cd})")
    for p in (rd, td, od, cd):
        refuse_needle(p)
    if is_inside(td, LOCAL_DIR) or is_inside(rd, LOCAL_DIR):
        raise T02Error("--tests-dir / --results-dir must not point into data/local")
    return {"mode": mode, "results_dir": os.path.abspath(rd), "rank_dir": os.path.join(os.path.abspath(rd), "rank"),
            "tests_dir": os.path.abspath(td), "out_dir": os.path.abspath(od), "checks_dir": os.path.abspath(cd)}


def notebook_paths(rank_dir):
    out = [os.path.join(rank_dir, f) for f in NOTEBOOK_FILES]
    for s in SETS:
        out += [os.path.join(rank_dir, s, "dumps", f) for f in SET_FILES]
    return out


def load_notebook(ctx):
    rd = ctx["rank_dir"]
    cfg = read_json(os.path.join(rd, "config.json"))
    log = read_json(os.path.join(rd, "run_log.json"))
    seam = read_json(os.path.join(rd, "seam_check.json"))
    runs = log.get("runs") if isinstance(log, dict) else None
    last = runs[-1].get("status") if runs and isinstance(runs[-1], dict) else None
    top = log.get("status") if isinstance(log, dict) else None
    if top != "completed" or last != "completed":
        raise T02Error(f"the notebook run_log.json status is {top!r} (last run {last!r}); only a completed run "
                       f"is used (seam_failed, repro_failed and readonly_changed runs never are)")
    failed = [(r.get("run", i), r.get("status")) for i, r in enumerate(runs[:-1], start=1)
              if isinstance(r, dict) and r.get("status") in FAILED_RUN_STATUSES]
    if failed and ctx["mode"] == "real":
        raise T02Error(f"run_log.json has earlier failed runs {failed}: the notebook was run again in the same Drive "
                       f"folder, which is not the registered rerun (PREREG_02 rev 5 9.2, rev 6 10.10); this run's "
                       f"outputs are not used (docs/TESTDAY.md step 4)")
    if seam.get("passed") is not True:
        raise T02Error("the notebook's seam_check.json did not pass: nothing from that run is used or uploaded "
                       "(PREREG_02 rev 5, section 9 item 2)")
    if not isinstance(seam.get("tol"), (int, float)) or isinstance(seam.get("tol"), bool):
        raise T02Error("seam_check.json has no numeric 'tol'")
    smoke, reh = cfg.get("smoke"), cfg.get("rehearsal")
    want = {"real": (False, False), "dry-run": (False, True), "rehearsal": (False, True), "smoke": (True, None)}
    ws, wr = want[ctx["mode"]]
    if smoke is not ws or (wr is not None and reh is not wr):
        raise T02Error(f"config.json says smoke={smoke!r}, rehearsal={reh!r}; {ctx['mode']} mode needs "
                       f"smoke={ws}" + ("" if wr is None else f", rehearsal={wr}"))
    return cfg, log, seam


def check_test_files(ctx, cfg):
    """data/test (or T) must hold the bytes the notebook used: config.json test_bundle SHA-256s."""
    td = ctx["tests_dir"]
    tb = cfg.get("test_bundle") or {}
    files = tb.get("files") or {}
    man = os.path.join(td, "MANIFEST.json")
    if not os.path.exists(man):
        raise T02Error(f"missing {man}")
    out = {"MANIFEST.json": sha256_file(man)}
    if out["MANIFEST.json"] != tb.get("manifest_sha256"):
        raise T02Error(f"{man}: sha256 {out['MANIFEST.json'][:12]} differs from the notebook's config.json "
                       f"test_bundle.manifest_sha256 {str(tb.get('manifest_sha256'))[:12]}")
    for name in TEST_FILES.values():
        p = refuse_needle(os.path.join(td, name))
        if not os.path.exists(p):
            raise T02Error(f"missing test file {p}")
        out[name] = sha256_file(p)
        if out[name] != files.get(name):
            raise T02Error(f"{name}: sha256 {out[name][:12]} differs from the notebook's config.json record "
                           f"{str(files.get(name))[:12]}: the notebook ran on other bytes")
    return out


def question_files(ctx):
    return {"seam_val": SEAM_QUESTIONS, "public": os.path.join(ctx["tests_dir"], TEST_FILES["public"]),
            "private": os.path.join(ctx["tests_dir"], TEST_FILES["private"])}


def check_set_inputs(ctx, cfg, qfiles):
    """Each set's _INPUT.json (written by the notebook before its dumps) and config.json sets record."""
    out = {}
    sets = cfg.get("sets") or {}
    for s in SETS:
        h = sha256_file(refuse_needle(qfiles[s]))
        rec = read_json(os.path.join(ctx["rank_dir"], s, "dumps", "_INPUT.json"))
        if rec.get("set") != s or rec.get("sha256") != h:
            raise T02Error(f"{s}/dumps/_INPUT.json names set {rec.get('set')!r} with sha256 "
                           f"{str(rec.get('sha256'))[:12]}, but the local question file hashes to {h[:12]}")
        if (sets.get(s) or {}).get("questions_sha256") != h:
            raise T02Error(f"config.json sets.{s}.questions_sha256 differs from the local question file")
        out[s] = {"questions": shown(qfiles[s]), "questions_sha256": h}
    return out


def t1_status(ctx, command, fallback):
    path = os.path.join(ctx["checks_dir"], "rank", "t1.json")
    if not os.path.exists(path):
        if ctx["mode"] == "real" or fallback:
            raise T02Error(f"{path} is missing: run tools/t02_checks.py t1 --comp rank first")
        return {"path": shown(path), "verdict": None, "note": f"absent (allowed in {ctx['mode']} mode)"}
    obj = read_json(path)
    v = obj.get("verdict")
    if ctx["mode"] == "real":
        # the full T1 on the raw files (review finding 2026-09-26: a structure-only t1.json used to pass)
        mode = (obj.get("provenance") or {}).get("mode")
        if obj.get("complete") is not True or obj.get("stage") != "raw" or mode != "real":
            raise T02Error(f"{shown(path)}: complete={obj.get('complete')!r} (ran {obj.get('subchecks_run')}), stage "
                           f"{obj.get('stage')!r}, mode {mode!r}; real mode needs the full T1 (T1.1-T1.5) on "
                           f"data/test/raw: PY tools/t02_checks.py t1 --comp rank --rerun --reason TEXT (without --only)")
    if fallback:
        if v != "FAIL_OPENING":
            raise T02Error(f"--fallback R10 is allowed only when T1 says FAIL_OPENING (t1.json verdict {v!r})")
    elif command == "opening":
        if v not in T1_OK + ("FAIL_OPENING",):
            raise T02Error(f"T1 verdict {v!r}: stop and write a revision before any prediction")
    elif v not in T1_OK:
        hint = (" (FAIL_OPENING: rank uses --fallback R10 --reason TEXT)" if v == "FAIL_OPENING"
                else " (stop and write a revision before any prediction)")
        raise T02Error(f"T1 verdict {v!r}{hint}")
    return {"path": shown(path), "sha256": sha256_file(path), "verdict": v}


def load_frozen(require_git):
    path = os.path.join(P02_RANK, "frozen_config.json")
    h = pinned(path, "frozen_config.json")
    commit = rf.require_committed(path) if require_git else None
    fc = read_json(path)
    if fc.get("format") != rf.FROZEN_FORMAT or fc.get("smoke") is not False:
        raise T02Error(f"{path}: not the real {rf.FROZEN_FORMAT} file")
    B = np.asarray(fc["basis"], dtype=np.float64)
    if B.shape != (rf.N_LEVELS, rf.N_LEVELS - 1) or not np.array_equal(B, rf.basis()):
        raise T02Error("frozen basis differs from p02_rank_fuse.basis()")
    if {k: fc["combo"].get(k) for k in R20} != R20:
        raise T02Error(f"frozen combination is {fc['combo']}, the registered one is R20")
    check_b(fc["params"], B, "frozen_config.json")
    if set(fc["dumps"]) != {DUMP_NAME} or fc["dumps"][DUMP_NAME].get("syms") != rf.DUMP_S[DUMP_NAME]:
        raise T02Error(f"frozen dumps {fc['dumps']} are not {{{DUMP_NAME}: 8 symmetries}}")
    if fc.get("opening_build_id") != FROZEN_OPENING_BUILD_ID:
        raise T02Error("frozen opening_build_id differs from the pinned build")
    state = rf.tool_state()
    changed = [f for f, v in state["files_sha256"].items() if fc["tool"]["files_sha256"].get(f) != v]
    if changed:
        raise T02Error(f"frozen tool files changed since tune: {changed}")
    if require_git and state["dirty_or_untracked"] != []:
        raise T02Error(f"frozen tool files are not clean in git: {state['dirty_or_untracked']}")
    return fc, {"path": shown(path), "sha256": h, "commit_check": commit}


def check_b(par, B, what):
    diff = float(np.max(np.abs(B @ np.asarray(par["beta"], dtype=np.float64) - np.asarray(par["b"]))))
    if not diff <= PARAM_TOL:
        raise T02Error(f"{what}: b is not B @ beta (largest difference {diff:.3g})")


def theta_of(par, opening):
    return np.array([par["a"], par["gamma"]] + ([par["w"]] if opening else []) + list(par["beta"]),
                    dtype=np.float64)


def common_checks(ctx, command, fallback):
    real = ctx["mode"] == "real"
    fc, frozen = load_frozen(require_git=real)
    rec = {"frozen_config": frozen}
    if real:
        # the notebook too (as competition 1 does): check_notebook_identity compares the run with this file, so it
        # must be the committed one (review finding 2026-09-26: an uncommitted rebuild used to be accepted)
        rec["own_files_commit_check"] = require_committed_clean(
            [SELF, os.path.abspath(check_submission.__file__), NOTEBOOK],
            "tools/t02_rank.py, tools/check_submission.py and colab/test02_rank.ipynb")
        rec["notebook_outputs_commit_check"] = require_committed_clean(notebook_paths(ctx["rank_dir"]),
                                                                       "the rank notebook outputs (spec 2.3)")
    cfg, log, seam = load_notebook(ctx)
    rec["notebook"] = {"config_sha256": sha256_file(os.path.join(ctx["rank_dir"], "config.json")),
                       "run_log_status": log.get("status"), "seam_check_passed": seam.get("passed"),
                       "seam_check_tol": seam.get("tol"), "seam_check_bitwise_equal": seam.get("bitwise_equal"),
                       "smoke": cfg.get("smoke"), "rehearsal": cfg.get("rehearsal"), "gpu": cfg.get("gpu"),
                       "torch": cfg.get("torch"), "cudnn": cfg.get("cudnn"), "weights": cfg.get("weights"),
                       "notebook": cfg.get("notebook"), "notebook_sha256": cfg.get("notebook_sha256")}
    same = check_notebook_identity(cfg, ctx["mode"])
    if same is not None:
        rec["notebook"]["same_as_repo_notebook"] = same
    rec["test_files_sha256"] = check_test_files(ctx, cfg)
    qfiles = question_files(ctx)
    rec["sets"] = check_set_inputs(ctx, cfg, qfiles)
    rec["t1"] = t1_status(ctx, command, fallback)
    wsha = (cfg.get("weights") or {}).get("sha256")
    if ctx["mode"] == "smoke":
        if not wsha:
            raise T02Error("config.json has no weights.sha256")
    elif wsha != fc["dumps"][DUMP_NAME]["weights_sha256"]:
        raise T02Error(f"the notebook used weights {str(wsha)[:12]}, not the frozen epoch_9.pth "
                       f"{fc['dumps'][DUMP_NAME]['weights_sha256'][:12]}")
    return fc, cfg, seam, qfiles, wsha, rec


# ================================================================ opening
def fit_games():
    """(colour, canonical string, level index) of every fit game, as p02_rank_opening.build counts them."""
    rows = c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=LOCAL_DIR)
    out = [(ro.norm_colour(r["color"]), ro.game_string(r["sgf_content"]), ro.LI[r["rank"]]) for r in rows]
    del rows
    return out


def opening_params():
    return {"M": ro.M, "alpha": ro.ALPHA, "n_points": ro.N_POINTS, "levels": c10data.LEVELS,
            "canonical": "p02_moves.canon(mv, 12): D4 transform making the first 12 points smallest, "
                         "ties -> smallest symmetry index",
            "counted_on": "load_rows(part='fit') only"}


def cmd_opening(a):
    t0 = time.time()
    ctx = resolve(a)
    fc, cfg, seam, qfiles, wsha, rec = common_checks(ctx, "opening", fallback=False)
    O = ctx["out_dir"]
    os.makedirs(O, exist_ok=True)
    out_npz, out_rep = os.path.join(O, "opening_test.npz"), os.path.join(O, "opening_test_report.json")
    rerun = remove_outputs([out_npz, out_rep], a)
    ref_path = os.path.join(P02_RANK, "opening_val.npz")
    ref_sha = pinned(ref_path, "opening_val.npz")
    ref = ro.load_opening(ref_path, "val")
    with np.load(ref_path, allow_pickle=False) as z:
        ref_ll, ref_nt = z["ll"], z["n_terms"]
    if ref["meta"]["build_id"] != FROZEN_OPENING_BUILD_ID:
        raise T02Error("opening_val.npz build_id is not the frozen opening build")

    slots = {}
    for s in SETS:
        slots[s] = [(q, i, c, ro.game_string(sgf)) for q, i, c, sgf in ro.read_questions(qfiles[s])]
    print("slots: " + ", ".join(f"{s} {len(v)}" for s, v in slots.items()) + f" ({time.time() - t0:.0f}s)",
          flush=True)
    model = ro.OpeningModel([(v[2], v[3]) for s in SETS for v in slots[s]])
    fit = fit_games()
    model.count(fit)
    del fit
    if model.n_fit != EXPECTED_FIT_GAMES:
        raise T02Error(f"counted {model.n_fit} fit games, expected {EXPECTED_FIT_GAMES}")
    print(f"counted {model.n_fit} fit games: {len(model.ctx_index)} contexts ({time.time() - t0:.0f}s)", flush=True)
    # scored and normalised per set, exactly as the frozen build scored its validation block
    scored = {}
    for s in SETS:
        ll, nt = model.score([(v[2], v[3]) for v in slots[s]])
        has = nt > 0
        o = np.where(has[:, None], ro.log_softmax(ll), 0.0) if len(ll) else np.zeros((0, 10))
        scored[s] = (ll, nt, has, o)

    ll, nt, has, o = scored["seam_val"]
    sv = slots["seam_val"]
    checks = {
        "question_id": [v[0] for v in sv] == rf.str_array(ref["question_id"]),
        "slot": [v[1] for v in sv] == [int(x) for x in np.asarray(ref["slot"]).tolist()],
        "color": [v[2] for v in sv] == rf.str_array(ref["color"]),
    }
    same_shape = ll.shape == ref_ll.shape
    checks.update({
        "ll": same_shape and ll.dtype == ref_ll.dtype and ll.tobytes() == ref_ll.tobytes(),
        "o": same_shape and o.dtype == np.asarray(ref["o"]).dtype and o.tobytes() == np.asarray(ref["o"]).tobytes(),
        "has": same_shape and np.array_equal(has, np.asarray(ref["has"], dtype=bool)),
        "n_terms": same_shape and np.array_equal(nt.astype(np.int16), np.asarray(ref_nt)),
    })
    passed = all(checks.values())
    max_ll = float(np.max(np.abs(ll - ref_ll))) if same_shape and len(ll) else None

    inputs = {s: {"path": shown(qfiles[s]), "sha256": sha256_file(qfiles[s]), "slots": len(slots[s]),
                  "questions": len({v[0] for v in slots[s]})} for s in SETS}
    params = opening_params()
    h = hashlib.sha256()
    h.update(json.dumps(params, sort_keys=True).encode())
    for s in SETS:
        h.update(scored[s][0].tobytes())
    meta = {"format": OPENING_FORMAT, "what": "test02 opening n-gram scores (tools/t02_rank.py opening)",
            "params": params, "build_id": h.hexdigest(), "frozen_opening_build_id": FROZEN_OPENING_BUILD_ID,
            "seam_bitwise_equal": passed, "created": now_iso(), "fit_games": model.n_fit,
            "fit_games_by_level": dict(zip(c10data.LEVELS, model.fit_by_level)), "inputs": inputs,
            "reference": {"path": shown(ref_path), "sha256": ref_sha},
            "tool_sha256": {"t02_rank.py": sha256_text(SELF),
                            **{n: rf.sha256_text(os.path.join(HERE, n))
                               for n in ("p02_rank_opening.py", "p02_moves.py", "c10data.py")}}}
    report = {"what": "test02 rank opening (spec 3.2): seam part bitwise equal to results/prereg02/rank/"
                      "opening_val.npz, then public and private scored with the same model",
              "created": meta["created"], "mode": ctx["mode"], "passed": passed,
              "seam_check": {"reference": meta["reference"], "equal": checks, "max_abs_ll_diff": max_ll,
                             "slots": len(sv)},
              "sets": {s: dict(inputs[s], with_score=int(scored[s][2].sum()),
                               mean_terms=float(scored[s][1].mean()) if len(scored[s][1]) else None,
                               terms_histogram={str(k): int(v) for k, v in
                                                zip(*np.unique(scored[s][1], return_counts=True))})
                       for s in SETS},
              "fit_games": model.n_fit, "keys_counted": {"contexts": len(model.ctx_index),
                                                         "entries": len(model.cnt_index)},
              "checks": rec, "rerun": rerun, "paths": {k: v for k, v in ctx.items() if k != "mode"},
              "tool": tool_record(), "environment": environment()}
    if not passed:
        report["seconds"] = round(time.time() - t0, 1)
        write_json(out_rep, report)
        print(f"ERROR: the seam part of the opening scores is not bitwise equal to opening_val.npz: {checks}; "
              f"no opening_test.npz written", file=sys.stderr)
        return 1
    arrays = {"set": [], "question_id": [], "slot": [], "color": [], "n_moves": []}
    for s in SETS:
        for q, i, c, st in slots[s]:
            arrays["set"].append(s)
            arrays["question_id"].append(q)
            arrays["slot"].append(i)
            arrays["color"].append(c)
            arrays["n_moves"].append(len(st) // 2)
    tmp = out_npz + ".tmp.npz"
    np.savez(tmp, set=np.array(arrays["set"]), question_id=np.array(arrays["question_id"]),
             slot=np.array(arrays["slot"], dtype=np.int16), color=np.array(arrays["color"]),
             ll=np.concatenate([scored[s][0] for s in SETS]), o=np.concatenate([scored[s][3] for s in SETS]),
             has=np.concatenate([scored[s][2] for s in SETS]),
             n_terms=np.concatenate([scored[s][1] for s in SETS]).astype(np.int16),
             n_moves=np.array(arrays["n_moves"], dtype=np.int16),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))
    os.replace(tmp, out_npz)
    report["output"] = {"path": shown(out_npz), "sha256": sha256_file(out_npz), "build_id": meta["build_id"]}
    report["seconds"] = round(time.time() - t0, 1)
    write_json(out_rep, report)
    print(f"opening: seam part bitwise equal to opening_val.npz ({len(sv)} slots); public "
          f"{len(slots['public'])}, private {len(slots['private'])} slots -> {out_npz} ({report['seconds']}s)")
    return 0


def load_opening_test(path, qfiles):
    z = rp.load_npz(refuse_needle(path), ["set", "question_id", "slot", "color", "o", "has", "meta"])
    meta = json.loads(str(z["meta"]))
    if meta.get("format") != OPENING_FORMAT or meta.get("seam_bitwise_equal") is not True:
        raise T02Error(f"{path}: not a passed {OPENING_FORMAT} file")
    if meta.get("frozen_opening_build_id") != FROZEN_OPENING_BUILD_ID:
        raise T02Error(f"{path}: made for another frozen opening build")
    for s in SETS:
        if (meta["inputs"].get(s) or {}).get("sha256") != sha256_file(qfiles[s]):
            raise T02Error(f"{path}: {s} was scored on another question file; rerun opening")
    z["meta"] = meta
    z["sha256"] = sha256_file(path)
    return z


# ================================================================ apply
def apply_set(name, ddir, qfile, combo, theta, B, op, smoke, expect_wsha):
    """The numeric core of p02_rank_fuse.cmd_apply for one question set; returns (rows, info, P, ok, qids)."""
    vk = rf.load_val_keys(ddir)
    d = rf.load_dump(os.path.join(ddir, "val_%s.npz" % DUMP_NAME), vk["n"], rf.DUMP_S[DUMP_NAME])
    if d["weights_sha256"] != expect_wsha:
        raise T02Error(f"{name}: dump weights {d['weights_sha256'][:12]} are not {expect_wsha[:12]}")
    rf.check_val_logits(d, vk["parsed"])
    logits = {DUMP_NAME: np.where(vk["parsed"][:, None, None], d["logits"], 0.0).astype(np.float32)}
    lp = rf.source_logprobs(combo["source"], logits)

    questions = rf.read_question_meta(refuse_needle(qfile))
    by_q = {q: (n, cols) for q, n, cols in questions}
    groups = vk["groups"]
    qids = [q for q, _ in groups]
    file_q = [q for q, _, _ in questions]
    if smoke:
        if not qids or qids != file_q[:len(qids)]:
            raise T02Error(f"{name}: smoke dump questions are not the first questions of the question file")
    elif qids != file_q:
        raise T02Error(f"{name}: val_keys questions differ from the question file (set or order)")
    for q, rows in groups:
        n, cols = by_q[q]
        if len(rows) != n or [vk["color"][i] for i in rows] != cols:
            raise T02Error(f"{name}: question {q}: val_keys slots/colours differ from the question file")

    cnn_groups = [[i for i in rows if vk["parsed"][i]] for _, rows in groups]
    S = rf.pool_matrix(cnn_groups, vk["n"])
    m = rf.pooled(S, lp, combo["pool"])
    n = np.array([len(g) for g in cnn_groups], dtype=np.float64)
    if combo["opening"]:
        sel = np.nonzero(np.asarray(op["set"]) == name)[0]
        pos = {(q, int(s)): int(j) for q, s, j in zip(rf.str_array(np.asarray(op["question_id"])[sel]),
                                                     np.asarray(op["slot"])[sel].tolist(), sel.tolist())}
        oi = []
        for i, (q, s) in enumerate(zip(vk["question_id"], vk["slot"])):
            j = pos.get((q, s))
            if j is None or str(op["color"][j]) != vk["color"][i]:
                raise T02Error(f"opening_test.npz: no matching entry for {name} question {q} slot {s}")
            oi.append(j)
        oi = np.asarray(oi)
        o = np.asarray(op["o"], dtype=np.float64)[oi]
        has = np.asarray(op["has"], dtype=bool)[oi]
        mop = np.asarray(rf.pool_matrix([[i for i in rows if has[i]] for _, rows in groups], vk["n"]) @ o)
    else:
        mop = np.zeros_like(m)
    ok = n > 0
    P = np.full((len(groups), rf.N_LEVELS), np.nan)
    if ok.any():
        z, _ = rf.question_logits(theta, m[ok], mop[ok], n[ok], combo["opening"], B)
        P[ok] = rf.softmax(z)
    rows_out = []
    for k, (q, rows) in enumerate(groups):
        probs = [repr(float(x)) for x in P[k]] if ok[k] else [""] * rf.N_LEVELS
        rows_out.append([q, len(rows), int(n[k])] + probs)
    info = {"combo": combo["id"], "questions": len(groups), "questions_without_parsed_game": int((~ok).sum()),
            "questions_without_parsed_game_ids": [q for q, k in zip(qids, ok) if not k],
            "games": vk["n"], "games_parsed": int(vk["parsed"].sum()),
            "inputs_sha256": {"val_keys.npz": vk["sha256"], "val_%s.npz" % DUMP_NAME: d["sha256"],
                              "questions": sha256_file(qfile)},
            "dump_weights_sha256": d["weights_sha256"], "dump_marker": d["marker"]}
    return rows_out, info, P, ok, qids


def write_probs(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(rf.PROB_COLUMNS)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)
    decide.read_probs(path)   # must pass decide.py's strict reader


def read_pred(path):
    with open(refuse_needle(path), newline="", encoding="utf-8") as f:
        return [(r["question_id"], r["pred_rank"]) for r in csv.DictReader(f)]


def _lf_bytes(path):
    with open(refuse_needle(path), "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def seam_compare(mine_csv, mine_b, ref_csv, ref_b, tol, smoke):
    """Registered local seam (rev 5 item 2): max |dP| <= tol and at most 4 rule-B differences."""
    mine, ref = decide.read_probs(mine_csv), decide.read_probs(ref_csv)
    ref_by = {r["question_id"]: r for r in ref}
    ids = [r["question_id"] for r in mine]
    problems = []
    if smoke:
        if ids != [r["question_id"] for r in ref][:len(ids)]:
            problems.append("smoke seam questions are not the first reference questions")
    elif ids != [r["question_id"] for r in ref]:
        problems.append("seam questions differ from the reference (set or order)")
    max_diff, count_bad = 0.0, []
    for r in mine:
        rr = ref_by.get(r["question_id"])
        if rr is None or (r["n_games"], r["n_parsed"]) != (rr["n_games"], rr["n_parsed"]) \
                or (r["probs"] is None) != (rr["probs"] is None):
            count_bad.append(r["question_id"])
            continue
        if r["probs"] is not None:
            max_diff = max(max_diff, max(abs(x - y) for x, y in zip(r["probs"], rr["probs"])))
    pb_mine, pb_ref = dict(read_pred(mine_b)), dict(read_pred(ref_b))
    b_diff = [q for q in ids if pb_mine.get(q) != pb_ref.get(q)]
    allowed = SEAM_MAX_B_DIFF if not smoke else max(1, math.ceil(SEAM_MAX_B_DIFF * len(ids) / SEAM_QUESTIONS_FULL))
    passed = not problems and not count_bad and max_diff <= tol and len(b_diff) <= allowed
    same_probs = _lf_bytes(mine_csv) == _lf_bytes(ref_csv)      # CRLF folded: the reference may be a CRLF checkout
    same_b = _lf_bytes(mine_b) == _lf_bytes(ref_b)
    return {"passed": passed, "questions": len(ids), "max_abs_diff": max_diff, "tol": tol,
            "rule_b_differences": len(b_diff), "rule_b_max": allowed, "rule_b_differing_questions": b_diff[:50],
            "count_mismatch_questions": count_bad[:50], "problems": problems,
            "byte_identical_probs": same_probs, "byte_identical_pred_B": same_b}


DUMP_ENV_KEYS = ("gpu_name", "torch", "cudnn")
SEAM_TOL_SAME_GPU, SEAM_TOL_OTHER_GPU = 1e-4, 1e-3


def dump_marker(ddir):
    return read_json(os.path.join(ddir, "_DONE_val_%s.json" % DUMP_NAME))


def check_dump_env(ctx):
    """The val_p02e9_d4 dumps of seam_val, public and private must come from one environment (gpu_name, torch,
    cudnn of their _DONE markers): a resumed notebook run reuses finished sets, and the seam only covers the
    environment that computed seam_val (PREREG_02 rev 5 item 2; review finding 2026-09-26)."""
    env = {}
    for s in SETS:
        m = dump_marker(os.path.join(ctx["rank_dir"], s, "dumps"))
        env[s] = {k: m.get(k) for k in DUMP_ENV_KEYS}
    mixed = [s for s in SETS if env[s] != env["seam_val"]]
    if mixed:
        raise T02Error(f"the dumps of {mixed} were not computed in the environment of the seam_val dump "
                       f"(gpu_name / torch / cudnn: {env}); the seam does not cover them. Nothing is written; the "
                       f"notebook must recompute all sets (docs/TESTDAY.md step 4)")
    return {"same": True, "keys": list(DUMP_ENV_KEYS), "env": env["seam_val"]}


def seam_tol(ctx, seam_nb, smoke):
    """The registered tolerance (rev 5 item 2): 1e-4 when the GPU that wrote the seam_val dump (its _DONE marker)
    is the prereg02 dump GPU (results/prereg02/rank/dumps/_DONE_val_p02e9_d4.json), else 1e-3; the notebook's
    seam_check.json tol must equal it. --smoke: the notebook compared with prereg02_smoke, whose marker is not
    here, so its value is used and only recorded (the local seam is record-only in smoke)."""
    nb_tol = float(seam_nb["tol"])
    if smoke:
        return nb_tol, {"tol": nb_tol, "notebook_tol": nb_tol, "rule": "smoke: the notebook's value (record only)"}
    mine = dump_marker(os.path.join(ctx["rank_dir"], "seam_val", "dumps")).get("gpu_name")
    ref = dump_marker(P02_DUMPS).get("gpu_name")
    same = mine is not None and mine == ref
    want = SEAM_TOL_SAME_GPU if same else SEAM_TOL_OTHER_GPU
    rec = {"tol": want, "notebook_tol": nb_tol, "seam_dump_gpu": mine, "prereg02_dump_gpu": ref, "same_gpu": same,
           "rule": "1e-4 when the seam_val dump's GPU (its _DONE marker) equals the prereg02 dump's GPU, else 1e-3"}
    if nb_tol != want:
        raise T02Error(f"seam_check.json tol {nb_tol:g} is not the registered {want:g} for the seam_val dump GPU "
                       f"{mine!r} (prereg02: {ref!r}); the notebook's seam check does not follow the rule")
    return want, rec


def apply_outputs(O, smoke):
    seam = os.path.join(O, "seam_val")
    out = [os.path.join(seam, n) for n in ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv",
                                           "decide_summary.json", "seam_local.json")]
    out += [os.path.join(O, n) for n in ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json",
                                         "submission_rank.csv", "apply_test_report.json")]
    if smoke:
        out += [os.path.join(O, "smoke_questions", n) for n in TEST_FILES.values()]
    return out


FALLBACK_COMMIT_BEFORE = "2026-11-04T11:00:00+08:00"    # registered: fitted and committed before 11/04


def fallback_commit(path):
    """(commit hash, committer time) of the last commit that touched the fallback file; it must be before the test
    data existed (PREREG_02 rev 5 9.6: the R10 parameters are fitted and committed before 11/04)."""
    import datetime as _dt
    rc, out = git("log", "-1", "--format=%H %cI", "--", repo_rel(path))
    if rc != 0 or not out:
        raise T02Error(f"{shown(path)}: no commit found")
    h, t = out.split()
    when, limit = _dt.datetime.fromisoformat(t), _dt.datetime.fromisoformat(FALLBACK_COMMIT_BEFORE)
    if not when < limit:
        raise T02Error(f"{shown(path)} was last committed at {t} ({h[:12]}), not before {FALLBACK_COMMIT_BEFORE}: the "
                       f"registered R10 parameters are the ones committed before the test data existed")
    return {"commit": h, "committed_at": t, "must_be_before": FALLBACK_COMMIT_BEFORE}


def load_fallback(path, real, fc):
    refuse_needle(path)
    commit = require_committed_clean([path], "fallback_R10.json") if real else None
    if real:
        commit["last_commit"] = fallback_commit(path)
    obj = read_json(path)
    if obj.get("format") != FALLBACK_FORMAT or obj.get("selfcheck_R20_passed") is not True:
        raise T02Error(f"{path}: not a {FALLBACK_FORMAT} file with a passed R20 self-check")
    if {k: obj["combo"].get(k) for k in R10} != R10:
        raise T02Error(f"{path}: combination {obj['combo']} is not R10")
    check_b(obj["params"], rf.basis(), "fallback_R10.json")
    if obj.get("dumps") != fc["dumps"]:
        raise T02Error(f"{path}: made for other dumps than the frozen configuration")
    return obj, {"path": shown(path), "sha256": sha256_file(path), "commit_check": commit}


def level_summary(P, ok, preds):
    dist = {lv: sum(1 for p in preds if p == lv) for lv in c10data.LEVELS}
    mean = ({lv: float(x) for lv, x in zip(c10data.LEVELS, np.mean(P[ok], axis=0))} if ok.any() else None)
    return {"rule_b_distribution": dist, "mean_probability_per_level": mean}


def cmd_apply(a):
    t0 = time.time()
    ctx = resolve(a)
    smoke, real = ctx["mode"] == "smoke", ctx["mode"] == "real"
    fallback = a.fallback is not None
    if fallback and not a.reason:
        raise T02Error("--fallback R10 needs --reason TEXT")
    if real and a.fallback_file:
        raise T02Error("real mode uses only the committed results/test02/rank/fallback_R10.json (no --fallback-file)")
    fc, cfg, seam_nb, qfiles, wsha, rec = common_checks(ctx, "apply", fallback)
    rec["dump_env"] = check_dump_env(ctx)
    tol, tol_rec = seam_tol(ctx, seam_nb, smoke)
    O = ctx["out_dir"]
    for d in (O, os.path.join(O, "seam_val")):
        if is_inside(d, P02_RANK) or is_inside(d, os.path.join(REPO, "results", "prereg02")):
            raise T02Error(f"decide.py output dir {d} is inside prereg02; refused")
    B = rf.basis()
    r20_theta = theta_of(fc["params"], True)
    if fallback:
        fb, fb_rec = load_fallback(a.fallback_file or FALLBACK_FILE, real, fc)
        test_combo, test_par = R10, fb["params"]
    else:
        fb_rec, test_combo, test_par = None, R20, fc["params"]
    test_theta = theta_of(test_par, test_combo["opening"])
    op = load_opening_test(os.path.join(O, "opening_test.npz"), qfiles)
    expect_wsha = wsha if smoke else fc["dumps"][DUMP_NAME]["weights_sha256"]

    os.makedirs(os.path.join(O, "seam_val"), exist_ok=True)
    rerun = remove_outputs(apply_outputs(O, smoke), a)

    # ---- seam first (always the frozen R20: the registered local seam)
    seam_dir = os.path.join(O, "seam_val")
    rows, seam_info, _P, _ok, _q = apply_set("seam_val", os.path.join(ctx["rank_dir"], "seam_val", "dumps"),
                                             qfiles["seam_val"], R20, r20_theta, B, op, smoke, expect_wsha)
    seam_csv = os.path.join(seam_dir, "rank_val_probs.csv")
    write_probs(seam_csv, rows)
    decide.run(seam_csv, seam_dir)
    ref_csv, ref_b = os.path.join(P02_RANK, "rank_val_probs.csv"), os.path.join(P02_RANK, "pred_B.csv")
    ref_sha = {"rank_val_probs.csv": pinned(ref_csv, "rank_val_probs.csv"), "pred_B.csv": pinned(ref_b, "pred_B.csv")}
    sc = seam_compare(seam_csv, os.path.join(seam_dir, "pred_B.csv"), ref_csv, ref_b, tol, smoke)
    warnings = []   # (smoke: the notebook compared with prereg02_smoke, this tool with the real prereg02 files)
    if not smoke and seam_nb.get("bitwise_equal") is True and \
            not (sc["byte_identical_probs"] and sc["byte_identical_pred_B"]):
        warnings.append("the notebook dumps were bitwise equal to prereg02 but the local seam files are not byte "
                        "identical: look for a drift between this tool and p02_rank_fuse.cmd_apply")
    seam_local = {"what": "test02 local seam (PREREG_02 rev 5 section 9 item 2): R20 on the seam dumps vs "
                          "results/prereg02/rank/rank_val_probs.csv and pred_B.csv",
                  "created": now_iso(), "mode": ctx["mode"], **sc,
                  "required": not smoke, "notebook_seam_bitwise_equal": seam_nb.get("bitwise_equal"),
                  "warnings": warnings, "reference_sha256": ref_sha, "set": seam_info, "tol_check": tol_rec,
                  "dump_env": rec["dump_env"],
                  "outputs_sha256": {n: sha256_file(os.path.join(seam_dir, n))
                                     for n in ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv",
                                               "decide_summary.json")}}
    write_json(os.path.join(seam_dir, "seam_local.json"), seam_local)
    for w_ in warnings:
        print("WARNING: " + w_, file=sys.stderr)
    print(f"seam: {sc['questions']} questions, max |dP| {sc['max_abs_diff']:.3g} (tol {sc['tol']:g}), rule-B "
          f"differences {sc['rule_b_differences']} (max {sc['rule_b_max']}), byte identical "
          f"{sc['byte_identical_probs']}/{sc['byte_identical_pred_B']} -> {'PASSED' if sc['passed'] else 'FAILED'}"
          + ("" if not smoke else " (recorded only in smoke mode)"), flush=True)
    if not sc["passed"] and not smoke:
        print("ERROR: the local seam failed: no test output is written and nothing from this run is uploaded "
              "(PREREG_02 rev 5, section 9 item 2). A rerun must recompute everything: follow docs/TESTDAY.md step 4 "
              "(archive results/test02/rank, the user renames the Drive folder to rank_failed_1 before running "
              "the notebook again)", file=sys.stderr)
        return 1

    # ---- test sets
    rows_all, sets_info, per_set = [], {}, {}
    for s in ("public", "private"):
        rows, info, P, ok, qids = apply_set(s, os.path.join(ctx["rank_dir"], s, "dumps"), qfiles[s], test_combo,
                                            test_theta, B, op, smoke, expect_wsha)
        rows_all += rows
        sets_info[s] = info
        per_set[s] = (P, ok, qids)
    probs_csv = os.path.join(O, "rank_test_probs.csv")
    write_probs(probs_csv, rows_all)
    dsum = decide.run(probs_csv, O)
    sub = os.path.join(O, "submission_rank.csv")
    atomic_copy(os.path.join(O, "pred_B.csv"), sub)
    with open(sub, "rb") as f1, open(os.path.join(O, "pred_B.csv"), "rb") as f2:
        if f1.read() != f2.read():
            raise T02Error("submission_rank.csv is not a byte copy of pred_B.csv")
    preds = dict(read_pred(sub))
    for s in ("public", "private"):
        P, ok, qids = per_set[s]
        sets_info[s].update(level_summary(P, ok, [preds[q] for q in qids]))
    if smoke:
        qdir = os.path.join(O, "smoke_questions")
        os.makedirs(qdir, exist_ok=True)
        check_q = []
        for s in ("public", "private"):
            n_q = len(per_set[s][2])
            dst = os.path.join(qdir, TEST_FILES[s])
            with open(qfiles[s], newline="", encoding="utf-8-sig") as f:
                lines = f.read().splitlines(keepends=True)
            with open(dst, "w", newline="", encoding="utf-8") as f:
                f.writelines(lines[:1 + n_q])
            check_q.append(dst)
        errors = check_submission.check("rank", sub, check_q)
        check_mode = "default mode on the smoke subsets of the question files"
    else:
        check_q = [qfiles["public"], qfiles["private"]]
        try:
            errors = check_submission.check("rank", sub, check_q, official=True)
        except TypeError as e:
            raise T02Error(f"tools/check_submission.py has no official mode ({e})") from None
        check_mode = "official"
    if errors:
        raise T02Error(f"check_submission ({check_mode}) failed: {errors[:5]}")

    report = {
        "what": "test02 competition 2 apply (PREREG_02 rev 5 section 9 items 3 and 5): frozen R20 per set, rule B, "
                "submission_rank.csv = pred_B.csv (public rows then private rows)",
        "created": now_iso(), "seconds": round(time.time() - t0, 1), "mode": ctx["mode"],
        "combo": test_combo, "params": test_par, "fallback_used": fallback,
        "fallback": {"file": fb_rec, "reason": a.reason} if fallback else None,
        "seam_combo": R20, "gate": fc.get("gate"),
        "seam_local": {"passed": sc["passed"], "max_abs_diff": sc["max_abs_diff"], "tol": sc["tol"],
                       "tol_check": tol_rec,
                       "rule_b_differences": sc["rule_b_differences"],
                       "byte_identical_probs": sc["byte_identical_probs"],
                       "byte_identical_pred_B": sc["byte_identical_pred_B"], "warnings": warnings},
        "sets": dict({"seam_val": seam_info}, **sets_info),
        "questions_without_parsed_game": {s: sets_info[s]["questions_without_parsed_game"] for s in sets_info},
        "missing_fill": decide.FALLBACK,
        "games_parsed": {s: sets_info[s]["games_parsed"] for s in sets_info},
        "decide_summary": {k: dsum[k] for k in ("n_questions", "n_missing", "n_A_ne_B", "distribution_B")},
        "check_submission": {"mode": check_mode, "questions": [shown(p) for p in check_q], "errors": errors},
        "checks": rec, "rerun": rerun, "paths": {k: v for k, v in ctx.items() if k != "mode"},
        "inputs_sha256": {"opening_test.npz": op["sha256"], **{f"prereg02 {k}": v for k, v in ref_sha.items()}},
        "outputs_sha256": {n: sha256_file(os.path.join(O, n)) for n in
                           ("rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json",
                            "submission_rank.csv")},
        "tool": tool_record(), "environment": environment(),
        "record_only": "rule_b_distribution and mean_probability_per_level per set are recorded for VALIDATION "
                       "rev 2's check of the official per-level counts; nothing is chosen from them",
    }
    write_json(os.path.join(O, "apply_test_report.json"), report)
    print(f"apply {test_combo['id']}: public {sets_info['public']['questions']}, private "
          f"{sets_info['private']['questions']} questions ({dsum['n_missing']} without a parsed game, filled with "
          f"{decide.FALLBACK}) -> {sub} (sha256 {report['outputs_sha256']['submission_rank.csv'][:12]}); "
          f"check_submission {check_mode}: OK")
    return 0


# ================================================================ fallback-fit
def prepare_fit(sources):
    """cmd_tune's fit-round preparation (p02_rank_fuse.py, tune: inner keys, pseudo_inner.json, banks, opening
    alignment, dumps, build_qset of the fit rounds) with the frozen functions; no eval rounds, no OOF."""
    keys = rp.load_inner_keys(P02_DUMPS)
    pseudo = rp.load_pseudo(os.path.join(P02_RANK, "pseudo_inner.json"), verify=True)
    if pseudo["obj"]["inputs"]["inner_keys_sha256"] != keys["sha256"]:
        raise T02Error("pseudo_inner.json was built from a different inner_keys.npz")
    split = c10data.read_split(SPLIT_CSV)
    inner_set = c10data.players_for(split, "inner")
    players = pseudo["players"]
    outside = sorted({p for p in keys["player_id"] if p not in inner_set}
                     | {p["player_id"] for p in players if p["player_id"] not in inner_set})
    if outside:
        raise T02Error(f"{len(outside)} player_ids are not split-v1 inner players, e.g. {outside[:3]}")
    dump_row = {}
    for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"])):
        dump_row[(r, g)] = i
    player_bank = []
    for p in players:
        rows = []
        for gid in p["bank"]:
            i = dump_row.get((p["rank"], gid))
            if i is None or keys["player_id"][i] != p["player_id"]:
                raise T02Error(f"bank game {gid} of {p['player_id']} is not in inner_keys for that player")
            rows.append((p["rank"], gid))
        player_bank.append(rows)
    fold_of = [p["fold"] for p in players]
    op = ro.load_opening(os.path.join(P02_RANK, "opening_inner.npz"), "inner")
    op_row = {(r, g): i for i, (r, g) in enumerate(zip(rf.str_array(op["rank"]), rf.str_array(op["game_id"])))}
    op_pid = rf.str_array(op["player_id"])
    idx = []
    for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"])):
        j = op_row.get((r, g))
        if j is None or op_pid[j] != keys["player_id"][i]:
            raise T02Error(f"inner game ({r}, {g}) missing from opening_inner.npz or player differs")
        idx.append(j)
    idx = np.asarray(idx)
    op_o = np.asarray(op["o"], dtype=np.float64)[idx]
    op_has = np.asarray(op["has"], dtype=bool)[idx]
    loaded = {}
    for s in sources:
        for name, _which in rf.SOURCES[s]:
            if name not in loaded:
                d = rf.load_dump(os.path.join(P02_DUMPS, f"inner_{name}.npz"), keys["n"], rf.DUMP_S[name])
                if not np.isfinite(d["logits"]).all():
                    raise T02Error(f"{d['path']}: non-finite logits (every inner game is parsed)")
                loaded[name] = d
    X_fit, S_fit = rf.build_qset(pseudo["fit"], players, dump_row, fold_of, keys["n"], op_o, op_has, player_bank)
    inputs = {"inner_keys.npz": keys["sha256"], "pseudo_inner.json": pseudo["sha256"],
              "opening_inner.npz": op["sha256"], "players_split.csv": sha256_file(SPLIT_CSV)}
    inputs.update({f"inner_{n}.npz": d["sha256"] for n, d in loaded.items()})
    return {"X": X_fit, "S": S_fit, "loaded": loaded, "inputs": inputs, "players": len(players)}


def fit_combo(prep, combo, B):
    """cmd_tune's refit of one combination on all fit questions (rf.source_logprobs, rf.pooled, rf.fit)."""
    lp = rf.source_logprobs(combo["source"], {n: d["logits"] for n, d in prep["loaded"].items()})
    X = prep["X"]
    X.m[(combo["source"], combo["pool"])] = rf.pooled(prep["S"], lp, combo["pool"])
    theta, refit = rf.fit(X, combo["source"], combo["pool"], combo["opening"], B)
    refit["fit_questions"] = len(X)
    return theta, refit, rf.params_dict(theta, combo["opening"], B)


def params_diff(p, q):
    vals = [abs(p[k] - q[k]) for k in ("a", "gamma", "w")]
    vals += [abs(x - y) for k in ("beta", "b") for x, y in zip(p[k], q[k])]
    if len(p["beta"]) != len(q["beta"]) or len(p["b"]) != len(q["b"]):
        return float("inf")
    return float(max(vals))


def cmd_fallback_fit(a):
    t0 = time.time()
    O = a.out_dir or REAL_OUT
    check_out_dir(O, "real")
    if is_inside(O, REPO) and not is_inside(O, REAL_RESULTS):
        raise T02Error(f"--out-dir inside the repository must be under results/test02 (got {O})")
    target = os.path.join(O, "fallback_selfcheck_R20.json" if a.combo == "R20" else "fallback_R10.json")
    if os.path.exists(target) and tracked(target):
        raise T02Error(f"{shown(target)} is committed (registered before 11/04); it is never refitted in place. "
                       f"A change needs a PREREG revision first")
    fc, frozen = load_frozen(require_git=is_inside(os.path.join(P02_RANK, "frozen_config.json"), REPO))
    os.makedirs(O, exist_ok=True)
    rerun = remove_outputs([target], a)
    prep = prepare_fit(["C5"])
    for k, v in prep["inputs"].items():
        want = fc["inputs_sha256"].get(k)
        if want is not None and want != v:
            raise T02Error(f"{k}: sha256 {v[:12]} differs from the frozen record {want[:12]}")
    B = rf.basis()
    print(f"fit questions: {len(prep['X'])} ({time.time() - t0:.0f}s)", flush=True)
    _th, refit20, par20 = fit_combo(prep, R20, B)
    diff = params_diff(par20, fc["params"])
    tune_path = os.path.join(P02_RANK, "tune_report.json")
    tune = read_json(tune_path)
    self_rec = {"combo": R20, "params": par20, "frozen_params": fc["params"], "max_abs_diff": diff,
                "tolerance": PARAM_TOL, "passed": bool(diff <= PARAM_TOL), "refit": refit20,
                "frozen_refit_loss": (tune.get("refit") or {}).get("loss"),
                "frozen_refit_fit_questions": (tune.get("refit") or {}).get("fit_questions")}
    common = {"created": now_iso(), "inputs_sha256": prep["inputs"],
              "frozen_config": frozen, "tune_report": {"path": shown(tune_path), "sha256": sha256_file(tune_path)},
              "procedure": "p02_rank_fuse.cmd_tune fit-round preparation (load_inner_keys, load_pseudo(verify=True), "
                           "load_opening(opening_inner.npz, 'inner'), build_qset of the fit rounds 0-49) and its "
                           "refit on all fit questions (source_logprobs, pooled, fit: two starts, L-BFGS-B); no "
                           "OOF, no eval rounds, no validation input",
              "rerun": rerun, "tool": tool_record(), "environment": environment()}
    print(f"R20 self-check: largest parameter difference {diff:.3g} (tolerance {PARAM_TOL:g}) -> "
          f"{'PASSED' if self_rec['passed'] else 'FAILED'}", flush=True)
    if a.combo == "R20":
        obj = {"format": SELFCHECK_FORMAT,
               "what": "R20 refit with the frozen functions on the committed inner inputs vs frozen_config.json",
               **self_rec, **common, "seconds": round(time.time() - t0, 1)}
        write_json(target, obj)
        print(f"wrote {target}")
        return 0 if self_rec["passed"] else 1
    if not self_rec["passed"]:
        raise T02Error("the R20 self-check failed; fallback_R10.json is not written")
    _th, refit10, par10 = fit_combo(prep, R10, B)
    combos = {c["id"]: c for c in tune["combinations"]}
    r10 = combos.get("R10") or {}
    if {k: r10.get(k) for k in R10} != R10 or r10.get("status") != "ok":
        raise T02Error("tune_report.json has no usable R10 entry")
    r00 = combos.get("R00") or {}
    obj = {"format": FALLBACK_FORMAT,
           "what": "PREREG_02 rev 5, section 9 item 6: the registered R10 fallback for a T1.4/T1.5 failure "
                   "(opening features unreliable): the same tune procedure refitted on all fit questions, "
                   "fitted before any test data exists; never scored on the local validation set",
           "combo": R10, "source_definition": rf.SOURCE_TEXT["C5"],
           "params": par10, "formula": fc["formula"],
           "decision_rule": fc["decision_rule"], "basis": "p02_rank_fuse.basis() (= frozen_config.json basis)",
           "dumps": fc["dumps"], "refit": refit10,
           "inner_oof_rank_balanced": r10["oof"]["rank_balanced"],
           "inner_oof_question_mean": r10["oof"]["question_mean"],
           "inner_oof_source": "results/prereg02/rank/tune_report.json combinations R10 oof (5-fold, eval rounds)",
           "R00_inner_oof_rank_balanced": (r00.get("oof") or {}).get("rank_balanced"),
           "selfcheck_R20_passed": True, "selfcheck_R20": self_rec,
           **common, "seconds": round(time.time() - t0, 1)}
    write_json(target, obj)
    print(f"R10: a {par10['a']:.6f}, gamma {par10['gamma']:.6f}, inner OOF rank-balanced "
          f"{obj['inner_oof_rank_balanced']:.4f} -> {target}")
    return 0


# ================================================================ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["opening", "apply", "fallback-fit"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--results-dir")
    ap.add_argument("--tests-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--checks-dir")
    ap.add_argument("--fallback", choices=["R10"])
    ap.add_argument("--fallback-file")
    ap.add_argument("--combo", choices=["R20", "R10"])
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--reason")
    a = ap.parse_args(argv)
    try:
        if a.reason is not None and not a.reason.strip():
            raise T02Error("--reason must not be empty")
        if a.rerun and not a.reason:
            raise T02Error("--rerun needs --reason TEXT (stored in the report)")
        if a.command == "fallback-fit":
            bad = [f for f, v in (("--dry-run", a.dry_run), ("--smoke", a.smoke), ("--rehearsal", a.rehearsal),
                                  ("--results-dir", a.results_dir), ("--tests-dir", a.tests_dir),
                                  ("--checks-dir", a.checks_dir), ("--fallback", a.fallback),
                                  ("--fallback-file", a.fallback_file)) if v]
            if bad or not a.combo:
                raise T02Error(f"fallback-fit takes --combo R20|R10 [--out-dir O] [--rerun --reason TEXT] only "
                               f"(got {bad or 'no --combo'})")
            return cmd_fallback_fit(a)
        if a.combo:
            raise T02Error("--combo belongs to fallback-fit")
        if a.command == "opening":
            if a.fallback or a.fallback_file:
                raise T02Error("--fallback belongs to apply")
            return cmd_opening(a)
        if a.fallback_file and not a.fallback:
            raise T02Error("--fallback-file needs --fallback R10")
        return cmd_apply(a)
    except ERRORS as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
