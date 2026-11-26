"""PREREG_09 competition 2: Q09 = Q02H + 4 x P1TO_E (ens3fam), fallback Q08P = Q02H + 4 x P1TO (LR refit)
(docs/PREREG_09_MAPPING_ENSEMBLE.md rev 1, sections 1-2).

    PY = .venv/Scripts/python.exe   (CPU; Git Bash; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL thread variable)
    PY tools/p09_rank_fuse.py check  [--precommit | --out-dir data/prereg09/tmp/<x>]   -> results/prereg09/rank/check_report.json
    PY tools/p09_rank_fuse.py tune   [--fallback] [--precommit | --out-dir ...] [--rerun --reason T --revision N]
                                     -> tune_report.json (Q09) / tune_fallback_report.json (Q08P);
                                        gates passed: + frozen_config.json and p1e_map.npz (Q09) / p1p_map.npz (Q08P)
    PY tools/p09_rank_fuse.py apply  [--rerun --reason T]
                                     -> rank_val_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json,
                                        check_submission.txt, val_lp_p1e.npz, apply_report.json (no score)
    PY tools/p09_rank_fuse.py apply --finalize    (after apply_audit.json is committed: binds it, sets usable)
The other steps (manifest, prep, selfcheck, train, status) are in tools/p09_map.py.

Modes (as tools/p08_rank_fuse.py)
    real        every input and every earlier step committed and clean (tools included); writes results/prereg09/rank.
    --precommit the same, except that "committed and clean" is not required of the p09 tools and the earlier p09
                records (recorded by SHA-256 instead; "precommit": true).
    --out-dir   preview under data/prereg09/tmp/ (not tmp/mapping): reports there, prepared arrays and weights in
                <out-dir>/map; flagged "preview": true and never accepted by a later real step.
apply has neither: it needs the committed frozen_config.json and map.

Discipline
    * data/local is read only by apply (the question file: question_id / num_games / color_i / SGF), before
      pilot_fuse / hs_lib / p1_lib are imported; check and tune never read it (u1_lib's and hs_lib's guards).
    * u1_lib (through p08_rank_fuse._import_u1) only in check-free inner code: Inner, cmd_tune (INNER_ONLY_FUNCS).
      apply never imports u1_lib or torch and never fits anything; its forward pass is tools/p09_forward.py (numpy).
    * pinned files are checked before use: the p08 tools before they are imported (top of this file), the 08 frozen
      files, the exploration scripts (PREREG_09 1.1), the P3 sets.json.
    * temporary files only under data/prereg09/tmp/; tools/score.py is never run; the validation label file is never
      named or opened (refuse_label_path on every path).
Exit codes: 0 done; 1 the step ran and says stop (gates not passed / check stop / a seam failed); 2 refused or error.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


class P09Error(RuntimeError):
    pass


class Stop(RuntimeError):
    """The step ran; its registered outcome is a stop (exit 1)."""


def _sha_lf_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


# 09 imports the p08 tools (which pin and import the p07 tools): checked before the import (PREREG_09 1.1)
P08_TOOL_PINS = {"tools/p08_rank_fuse.py": "64f42bd956f7808947e0c85abe1f9fa60d9958cb8e769fc889c6283c188d481e",
                 "tools/p08_p1.py": "a10c77c4bd97472b464f39080816090f0f64986195384b28e1d95dfc76e6677f"}
_bad = [r for r, s in P08_TOOL_PINS.items() if _sha_lf_file(os.path.join(REPO, *r.split("/"))) != s]
if _bad:
    raise P09Error(f"pinned p08 tools changed: {_bad} (PREREG_09 1.1: never edited)")

import p08_rank_fuse as R  # noqa: E402
import p09_forward as FW  # noqa: E402

rf, p3, p4, P7 = R.rf, R.p3, R.p4, R.P7

# ================================================================ constants (PREREG_09 rev 1)
_LABEL = "ans" + "wer"
LEVELS = rf.LEVELS
L = rf.N_LEVELS
P08_FROZEN_PINS = {  # pin_sha: .json / .csv LF, .npz raw (PREREG_09 1.1)
    "results/prereg08/rank/frozen_config.json": "421bbeb4524bed8ccf73babd58cac6b16c82f45b1aa57acef2e55bd52be0ba70",
    "results/prereg08/rank/p1_map.npz": "116923629b66198a79da06282aac9ca863acf77bf6a1e1f5d584b526fb54afb2",
    "results/prereg08/rank/rank_val_probs.csv": "7a513367147227e3631bfb7bad36064f3e90fffb6ee358716e389698ce1ccd97",
    "results/prereg08/rank/pred_B.csv": "b02d4f947910726fa25c292e9798e7407b1e802e2fed080d843f56b9beb48872",
    "results/prereg08/rank/val_lp_p1.npz": "8815db14654eefdb8c2a11cfbb709de4254272de734cec415b3dfc2dbb0e59a9",
    "results/prereg08/rank/apply_report.json": "63980d79866d2653a4e600cef78cadd9d0d27c828a69e1b34d6ae2309eaf040f",
    "results/prereg08/rank/score_H8R.json": "8ecd2425fe3eef784e28d7587184aec67c0b41b49503e95b42f8532757209047",
    "results/prereg08/rank/tune_report.json": "995c7cd9cf8f4a01c4e888872c253f075e0246e9f603cbfa09f84bb015aa1b06",
    "results/prereg08/feat/inner_manifest.json": "0ce2114d637f5205b666e46cfc2b05b9201982e06e231a85212f479f49a415d3",
    "results/prereg08/feat/val_p1.npz": "11b1d38cc116401d44825883b864bf635b2d44e109baa17115fb4093055da4e0",
    "results/prereg08/feat/val_report.json": "2ce9e86f8e639217c38814541f7d1bfb083b609ac194b5e19c8c38c60e653535",
    "results/prereg08/feat/val_seam_feats.npz": "0a7b9b0010192a4c7f3878e914f9e514a18cfa338d94dc8c49936b6b625b9318"}
EXPLORE_DIR = os.path.join(REPO, "results", "prereg09", "explore", "mapping")
EXPLORE_PINS = {  # LF SHA-256 (PREREG_09 1.1: specification only, never imported by the tools)
    "results/prereg09/explore/mapping/prep.py": "48fec73cb10014a4e775b6d425486d449fdd89dbfd4a4143c7f1e2a5827aa219",
    "results/prereg09/explore/mapping/train.py": "5f53180992ada3a4f1fef433a0024ce4d2aa2c10613729cbca1484311e80a67b",
    "results/prereg09/explore/mapping/combos.py": "de7690186b647f18f1be82db167a71080be8b0e0c3f936aa636acb2d3c608073",
    "results/prereg09/explore/mapping/ev_lib.py": "0f2ccb9c2690f1256b53df370200bf0a32de3068c8d378934e1a7d3096c0bf1b",
    "results/prereg09/explore/mapping/run_eval.py": "aff31dbf4e13b3bd7ca64e791fc68dccb579e92b4a4a9ed78e0ed401b37bbb10",
    "results/prereg09/explore/p3/p3_sets.py": "7f77071747d5a3b2d85e4814cbc4834c7aee8f6b95a866da3f2d69e320dd5d6d"}
P3_SETS = os.path.join(REPO, "data", "prereg09", "p3", "sets.json")
DATA_PINS = {"data/prereg09/p3/sets.json": "164d8bd7b0f212110e222f0a92aeb20079d91ae538c2f0414d563b8478d92eba"}
P3_DATA = os.path.join(REPO, "data", "prereg09", "p3")
P3_DIR = os.path.join(P3_DATA, "probe")
P3_CHUNKS = 600
P3_GAMES = 120000
P3_FAIL_MAX = 600
P3_CUTOFF = datetime.datetime(2026, 10, 10, 23, 59, 59, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
INNER_USABLE = 73169            # bank / mono usable games (results/prereg09/explore/mapping prep_record.json)
PROBE_SETS = ("p1_probe", "p2_probe", "p3_probe")
SOURCE_OF = {"p1_probe": "P1", "p2_probe": "P2", "p3_probe": "P3"}
SCALE = FW.SCALE
TOL = 1e-12
GATE = 0.010
ENS_TOL = 0.002
TORCH_TOL = 1e-4
RHO_FLOOR_VAL = 0.078
RHO_FLOOR_TEST = 0.155
MONO_QUESTIONS = R.MONO_QUESTIONS
Q02_TERMS, Q02H_TERMS, Q08_TERMS = R.Q02_TERMS, R.Q02H_TERMS, R.Q08_TERMS
# seams (PREREG_09 2.3 item 1)
REF = {"Q02H": 0.6242752597994481, "Q08": 0.6986195853659671, "Q08_mono": 0.6946737091777063,
       "ENS": 0.7113584160645174, "ENS_mono": 0.7154116187609454}
# the exploration's member records (data/prereg09/tmp/mapping/v_<name>.json; record only)
EXPLORE_MEMBERS = {"mlpclm256": {"cv_logloss": 1.5537489223576944, "e_star": 24},
                   "mlpclm256_s1": {"cv_logloss": 1.5533713856299027, "e_star": 24},
                   "mlpclm256_s2": {"cv_logloss": 1.5535387602861421, "e_star": 24},
                   "mlp512": {"cv_logloss": 1.57392979647472, "e_star": 24},
                   "mlp512_s1": {"cv_logloss": 1.5733580482896476, "e_star": 15},
                   "lowrank3_ex": {"cv_logloss": 1.5780535777773015, "e_star": 30}}
EXPLORE_TMP = os.path.join(REPO, "data", "prereg09", "tmp", "mapping")
VENV = {"python": "3.12.13", "torch": "2.14.0+cpu", "numpy": "2.5.3", "scipy": "1.18.1", "threads": 8}
OUT = os.path.join(REPO, "results", "prereg09", "rank")
FEAT = os.path.join(REPO, "results", "prereg09", "feat")
MANIFEST = os.path.join(FEAT, "p3_manifest.json")
MAP_DIR = os.path.join(REPO, "data", "prereg09", "map")
TMP = os.path.join(REPO, "data", "prereg09", "tmp")
P08_OUT = R.OUT
P08_FEAT = R.FEAT
P08_MANIFEST = R.MANIFEST
P08_MAP = os.path.join(P08_OUT, "p1_map.npz")
P08_FROZEN = os.path.join(P08_OUT, "frozen_config.json")
P08_PROBS = os.path.join(P08_OUT, "rank_val_probs.csv")
P08_VAL_LP = os.path.join(P08_OUT, "val_lp_p1.npz")
P08_VAL_REPORT = os.path.join(P08_FEAT, "val_report.json")
P08_VAL_P1 = os.path.join(P08_FEAT, "val_p1.npz")
PREREG = os.path.join(REPO, "docs", "PREREG_09_MAPPING_ENSEMBLE.md")
REV1_COMMIT = "73b4deb"
TOOL_FILES = ("tools/p09_map.py", "tools/p09_rank_fuse.py", "tools/p09_forward.py")
NO_PYCACHE_DIRS = ("results/prereg09", "results/prereg08", "results/prereg07", "results/prereg05/explore/u1",
                   "third_party/katago")
FORMATS = {"manifest": "c10-prereg09 p3_manifest v1", "prep": "c10-prereg09 prep_record v1",
           "check": "c10-prereg09 rank check_report v1", "selfcheck": "c10-prereg09 rank selfcheck v1",
           "train": "c10-prereg09 rank train_report v1", "train_fallback": "c10-prereg09 rank train_fallback_report v1",
           "tune": "c10-prereg09 rank tune_report v1", "tune_fallback": "c10-prereg09 rank tune_fallback_report v1",
           "frozen": "c10-prereg09 rank frozen_config v1", "apply": "c10-prereg09 rank apply_report v1",
           "audit": "c10-prereg09 apply_audit v1", "member": "c10-prereg09 member v1",
           "moments": "c10-prereg09 moments v1", "fallback_fit": "c10-prereg09 fallback fit v1",
           "val_lp": "c10-prereg09 val_lp_p1e v1", "p1p_map": R.FORMATS["p1_map"]}
# functions of this file that may import u1_lib / hs_baseline (tests/test_p09_hygiene.py checks)
INNER_ONLY_FUNCS = ("Inner", "cmd_tune")
# functions of this file that may import torch or tools/p09_map.py (training code); apply never does
TORCH_FUNCS = ("torch_check", "venv_versions")   # venv_versions imports torch only with with_torch=True

ERRORS = (P09Error, FW.ForwardError) + R.ERRORS


# ================================================================ small helpers
def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def refuse_label_path(path):
    if path is not None and _LABEL in os.path.abspath(os.fspath(path)).lower():
        raise P09Error(f"refusing {path}: the validation label file is never opened by the prereg09 tools")
    return path


under_data_local = P7.under_data_local
inside = P7.inside
rel = P7.rel
path_of = P7.path_of
git = P7.git
git_head = P7.git_head
commit_time = P7.commit_time
tracked_clean = P7.tracked_clean
jsonable = P7.jsonable


def sha_raw(path):
    h = hashlib.sha256()
    with open(refuse_label_path(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(path):
    with open(refuse_label_path(path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def pin_sha(path):
    return sha_lf(path) if os.path.splitext(path)[1].lower() in (".json", ".csv", ".py", ".txt", ".md") \
        else sha_raw(path)


def read_json(path):
    with open(refuse_label_path(path), encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    refuse_label_path(path)
    if under_data_local(path):
        raise P09Error("refusing to write under data/local")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(obj), f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def require_committed(paths, what, precommit=False):
    missing = [rel(p) for p in paths if not os.path.exists(p)]
    if missing:
        raise P09Error(f"{what}: missing {missing}")
    rec = {"files": {rel(p): pin_sha(p) for p in paths}}
    probs = tracked_clean(paths)
    rec["commit_problems"] = probs
    if probs and not precommit:
        raise P09Error(f"{what}: {probs} (commit each step before the next)")
    rec["precommit_skipped"] = bool(probs and precommit)
    return rec


def code_files():
    return list(TOOL_FILES) + list(P08_TOOL_PINS) + list(R.PINNED) + list(EXPLORE_PINS)


def tool_state():
    files = code_files()
    rc, st = git(["status", "--porcelain", "--untracked-files=all", "--"] + files)
    dirty = [ln[3:] for ln in st.splitlines() if ln.strip()] if rc == 0 else None
    return {"git_head": git_head(), "dirty_or_untracked": dirty, "files_sha256_rule": "sha256, CRLF folded to LF",
            "files_sha256": {f: (sha_lf(path_of(f)) if os.path.exists(path_of(f)) else None) for f in files}}


def require_tools(ts, precommit):
    if ts["git_head"] is None or ts["dirty_or_untracked"] is None:
        raise P09Error("git is not available")
    if ts["dirty_or_untracked"] and not precommit:
        raise P09Error(f"commit the tools first (uncommitted: {ts['dirty_or_untracked']})")


def tools_equal_record(ts, recorded):
    rec = (recorded or {}).get("files_sha256") or {}
    return [f for f in code_files() if rec.get(f) != ts["files_sha256"].get(f)]


def code_differs(ts, recorded, files):
    """Files (of `files`) whose current LF SHA-256 differs from a record's tool SHA-256s -> list."""
    rec = (recorded or {}).get("files_sha256") or {}
    return [f for f in files if rec.get(f) != ts["files_sha256"].get(f)]


TRAINING_CODE = ("tools/p09_map.py", "tools/p09_forward.py")


def selfcheck_accept(c, sr, revision):
    """PREREG_09 2.3 item 2: a selfcheck that did not pass stops train / tune, except when its only problem is the
    0.002 ensemble tolerance and a committed PREREG_09 revision N >= 2, committed after selfcheck.json, records why
    (a numerical-path difference, not a logic error) -> None (passed) or the override record; raises otherwise."""
    if sr.get("passed") is True:
        if revision:
            raise P09Error("--revision is only for a selfcheck that did not pass")
        return None
    if not sr.get("only_ensemble_tolerance_failed"):
        raise P09Error(f"selfcheck.json did not pass ({sr.get('problems')}); train / tune do not run")
    if not revision or revision < 2:
        raise P09Error("selfcheck.json failed only the 0.002 ensemble tolerance: continue only with --revision N "
                       "(N >= 2, a committed PREREG_09 revision written after selfcheck.json; PREREG_09 2.3 item 2)")
    if c.preview:
        return {"revision": revision, "preview": True, "note": "preview: the revision commit is not checked"}
    h, t = revision_committed(revision)
    st = commit_time(os.path.join(c.out_dir, "selfcheck.json"))
    if not st or datetime.datetime.fromisoformat(t) <= datetime.datetime.fromisoformat(st):
        raise P09Error(f"PREREG_09 rev {revision} ({t}) is not later than the committed selfcheck.json ({st})")
    return {"revision": revision, "commit": h, "commit_time": t, "selfcheck_commit_time": st}


def check_pins():
    """-> {rel: sha}; raises unless the 08 pins (p08_rank_fuse.check_pins), the 08 frozen files, the exploration
    scripts and the P3 sets.json keep their SHA-256."""
    got = R.check_pins()
    bad = []
    for r, want in P08_FROZEN_PINS.items():
        got[r] = pin_sha(path_of(r)) if os.path.exists(path_of(r)) else None
        if got[r] != want:
            bad.append(r)
    for r, want in EXPLORE_PINS.items():
        got[r] = sha_lf(path_of(r)) if os.path.exists(path_of(r)) else None
        if got[r] != want:
            bad.append(r)
    for r, want in DATA_PINS.items():
        got[r] = sha_raw(path_of(r)) if os.path.exists(path_of(r)) else None
        if got[r] != want:
            bad.append(r)
    for r, want in P08_TOOL_PINS.items():
        got[r] = sha_lf(path_of(r))
        if got[r] != want:
            bad.append(r)
    if bad:
        raise P09Error(f"pinned files changed: {bad} (never edit them; PREREG_09 1.1 / 2.2 item 3)")
    return got


def no_pycache():
    found = []
    for d in NO_PYCACHE_DIRS:
        for root, dirs, _files in os.walk(path_of(d)):
            found.extend(rel(os.path.join(root, x)) for x in dirs if x == "__pycache__")
    return found


BLAS_ENV = R.BLAS_ENV
PREVIEW_THREADS_OK = "C10_P09_PREVIEW_THREADS"


def blas_env_check(preview):
    """PREREG_09 2.6: tune / apply (and every other step) refuse a BLAS thread variable; a preview may opt in with
    C10_P09_PREVIEW_THREADS=1 (its numbers are then not comparable)."""
    set_ = {k: os.environ[k] for k in BLAS_ENV if os.environ.get(k)}
    if set_ and not (preview and os.environ.get(PREVIEW_THREADS_OK) == "1"):
        raise P09Error(f"BLAS thread variables are set ({set_}): unset them (PREREG_09 1.3 item 7 / 2.6)")
    return set_


def environment_cpu():
    return {**R.environment_cpu(), "registered_venv": VENV}


def venv_versions(with_torch):
    import scipy
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}
    if with_torch:
        import torch
        out["torch"] = torch.__version__
    return out


def venv_check(with_torch, preview):
    """Real runs: .venv equals PREREG_09 1.3 item 7 (python, numpy, scipy, and torch when training)."""
    got = venv_versions(with_torch)
    bad = {k: (got[k], VENV[k]) for k in got if got[k] != VENV[k]}
    if bad and not preview:
        raise P09Error(f".venv differs from PREREG_09 1.3 item 7: {bad}")
    return {**got, "differences": bad}


# ================================================================ modes and outputs
class Ctx:
    pass


def resolve(a, cmd):
    c = Ctx()
    c.cmd = cmd
    c.precommit = bool(getattr(a, "precommit", False))
    od = getattr(a, "out_dir", None)
    if c.precommit and od:
        raise P09Error("--precommit writes the real folder; it cannot be combined with --out-dir")
    if od:
        c.out_dir = os.path.abspath(od)
        if (not inside(c.out_dir, TMP) or inside(c.out_dir, EXPLORE_TMP) or inside(c.out_dir, OUT)
                or os.path.normcase(os.path.realpath(c.out_dir)) == os.path.normcase(os.path.realpath(TMP))):
            raise P09Error(f"--out-dir (preview) must be a folder under {rel(TMP)} other than tmp/mapping "
                           f"(got {c.out_dir})")
        c.preview = True
        c.map_dir = os.path.join(c.out_dir, "map")
    else:
        c.out_dir = OUT
        c.preview = False
        c.map_dir = MAP_DIR
    if under_data_local(c.out_dir) or under_data_local(c.map_dir):
        raise P09Error("refusing to write under data/local")
    return c


def mode_flags(c):
    return {"preview": c.preview, "precommit": c.precommit}


def guard_outputs(c, a, names, folder=None):
    folder = folder or c.out_dir
    paths = [os.path.join(folder, n) for n in names]
    have = [p for p in paths if os.path.exists(p)]
    if not have:
        return None
    if not getattr(a, "rerun", False) or not getattr(a, "reason", None):
        raise P09Error(f"outputs exist ({[os.path.basename(p) for p in have]}); rerun only with --rerun --reason TEXT")
    if not c.preview:
        tracked = [p for p in have if git(["ls-files", "--error-unmatch", "--", rel(p)])[0] == 0]
        if tracked:
            raise P09Error(f"committed outputs are never overwritten: {[rel(p) for p in tracked]}")
    return {"reason": a.reason, "old_sha256": {os.path.basename(p): pin_sha(p) for p in have}}


def prior_report(c, name, fmt, precommit_ok, folder=None):
    """An earlier step's report in this mode -> (report, sha, commit record)."""
    p = os.path.join(folder or c.out_dir, name)
    if not os.path.exists(p):
        raise P09Error(f"run the earlier step first: {name} missing in {folder or c.out_dir}")
    r = read_json(p)
    if r.get("format") != fmt:
        raise P09Error(f"{name}: not a {fmt} file")
    if bool(r.get("preview")) != c.preview:
        raise P09Error(f"{name} is a {'preview' if r.get('preview') else 'real'} record; this run is "
                       f"{'a preview' if c.preview else 'real'} (a preview is never accepted by a later step)")
    commit = None if c.preview else require_committed([p], name, precommit=precommit_ok)
    return r, pin_sha(p), commit


def tune_rerun_check(rerun, preview, revision, old_created):
    """Real tune --rerun needs a committed PREREG_09 revision N >= 2 later than the tune it replaces (2.5)."""
    if rerun is None or preview:
        return rerun
    if not revision or revision < 2:
        raise P09Error("tune is judged once: --rerun needs --revision N >= 2 (a committed PREREG_09 revision that "
                       "names a program error)")
    h, t = revision_committed(revision)
    if old_created and datetime.datetime.fromisoformat(t) <= datetime.datetime.fromisoformat(old_created):
        raise P09Error(f"PREREG_09 rev {revision} ({t}) is not later than the tune it replaces ({old_created})")
    rerun["revision"] = {"rev": revision, "commit": h, "commit_time": t, "replaced_tune_created": old_created}
    return rerun


def revision_committed(n):
    rc, out = git(["log", "--format=%H %cI", "-S", f"| rev {n} |", "--", rel(PREREG)])
    lines = [x for x in out.splitlines() if x.strip()] if rc == 0 else []
    if not lines:
        raise P09Error(f"PREREG_09 rev {n} is not in any commit")
    h, t = lines[-1].split()
    return h, t


# ================================================================ P3 rule (PREREG_09 1.2) and the manifest binding
def p3_decision(present, expected, valid, failed, now, cutoff=None):
    """Fixed rule, no choice: manifest only after 600 / 600 chunks or after 10/10 23:59:59 (+08:00); P3 is used iff
    at that moment all 600 chunks are there, pass the chunk check, and failed games <= 600."""
    cutoff = P3_CUTOFF if cutoff is None else cutoff
    complete = bool(present == expected and valid)
    if not complete and now <= cutoff:
        raise P09Error(f"P3 has {present}/{expected} chunks{'' if valid else ' (some fail the chunk check)'} and "
                       f"it is before {cutoff.isoformat()}: manifest waits (PREREG_09 1.2)")
    used = bool(complete and failed <= P3_FAIL_MAX)
    if used:
        reason = f"600/600 chunks, failed games {failed} <= {P3_FAIL_MAX}"
    elif not complete:
        reason = f"P3 incomplete at the manifest ({present}/{expected} chunks, valid {valid}) after the cutoff"
    else:
        reason = f"P3 failed games {failed} > {P3_FAIL_MAX}"
    return {"complete": complete, "chunks": int(present), "chunks_expected": int(expected), "valid": bool(valid),
            "failed": int(failed), "failed_limit": P3_FAIL_MAX, "used": used, "reason": reason,
            "cutoff": cutoff.isoformat(), "decided_at": now.isoformat(timespec="seconds"),
            "rule": "PREREG_09 1.2: used iff 600/600 chunks present and valid at the manifest and failed <= 600"}


def probe_used_label(sets):
    return "+".join(SOURCE_OF[s] for s in sets)


def manifest_binding(c, precommit, required=True):
    """results/prereg09/feat/p3_manifest.json (committed in real mode) -> (manifest or None, sha, commit)."""
    if not os.path.exists(MANIFEST):
        if required and not c.preview:
            raise P09Error(f"{rel(MANIFEST)} missing: run p09_map.py manifest first")
        return None, None, None
    commit = None if c.preview else require_committed([MANIFEST], "p3_manifest.json", precommit=precommit)
    man = read_json(MANIFEST)
    if man.get("format") != FORMATS["manifest"]:
        raise P09Error("p3_manifest.json: wrong format")
    return man, sha_lf(MANIFEST), commit


def probe_sets_from_manifest(man):
    used = bool(man and (man.get("p3") or {}).get("used") is True)
    return ["p1_probe", "p2_probe"] + (["p3_probe"] if used else [])


def verify_p3_chunks(man):
    """Every P3 chunk the manifest lists still has its SHA-256, no other chunk file is in the folder -> problems."""
    probs = []
    rows = ((man.get("p3_chunks") or {}).get("chunks")) or []
    listed = {os.path.normcase(os.path.abspath(path_of(r["file"]))) for r in rows}
    for r in rows:
        p = path_of(r["file"])
        if not os.path.exists(p) or sha_raw(p) != r["sha256"]:
            probs.append(f"p3_probe: {r['file']} missing or changed since the manifest")
    on_disk = {os.path.normcase(os.path.abspath(f)) for f in R.chunk_files(P3_DIR)}
    extra = sorted(on_disk - listed)
    if extra:
        probs.append(f"p3_probe: chunk files not in the manifest: {[rel(x) for x in extra][:5]}")
    if len(rows) != P3_CHUNKS:
        probs.append(f"p3_probe: {len(rows)} chunks in the manifest, registered {P3_CHUNKS}")
    return probs


# ================================================================ prepared arrays (tools/p09_map.py prep)
PREP_FILES = ("probe_X.npy", "probe_ex.npy", "probe_meta.npz", "inner_X.npy", "inner_ex.npy", "inner_meta.npz")


class Prep:
    """The prepared raw arrays of tools/p09_map.py prep (float32; X with NaN opponent halves, EX raw) and their
    metadata. Standardisation happens on use (PREREG_09 1.3 item 1; p09_forward)."""

    def __init__(self, map_dir):
        self.dir = map_dir
        rec_path = os.path.join(map_dir, "prep_record.json")
        if not os.path.exists(rec_path):
            raise P09Error(f"{rel(rec_path) if inside(rec_path, REPO) else rec_path} missing: run p09_map.py prep")
        self.record = read_json(rec_path)
        if self.record.get("format") != FORMATS["prep"]:
            raise P09Error("prep_record.json: wrong format")
        self.record_sha = sha_lf(rec_path)
        self.probe_X = np.load(os.path.join(map_dir, "probe_X.npy"), mmap_mode="r")
        self.probe_ex = np.load(os.path.join(map_dir, "probe_ex.npy"), mmap_mode="r")
        self.inner_X = np.load(os.path.join(map_dir, "inner_X.npy"), mmap_mode="r")
        self.inner_ex = np.load(os.path.join(map_dir, "inner_ex.npy"), mmap_mode="r")
        with np.load(os.path.join(map_dir, "probe_meta.npz"), allow_pickle=False) as z:
            self.y = np.asarray(z["y"]).astype(np.int64)
            self.players = np.asarray(z["players"]).astype(str)
            self.src = np.asarray(z["src"]).astype(str)
            self.ref = np.asarray(z["ref"]).astype(np.int64)
        with np.load(os.path.join(map_dir, "inner_meta.npz"), allow_pickle=False) as z:
            self.inner_ref = np.asarray(z["ref"]).astype(np.int64)
            self.inner_src = np.asarray(z["src"]).astype(str)
        n, ni = len(self.y), len(self.inner_ref)
        if self.probe_X.shape != (n, FW.X_DIM) or self.probe_ex.shape != (n, FW.NEX):
            raise P09Error("probe arrays and probe_meta.npz disagree")
        if self.inner_X.shape != (ni, FW.X_DIM) or self.inner_ex.shape != (ni, FW.NEX):
            raise P09Error("inner arrays and inner_meta.npz disagree")

    def files_sha256(self):
        return {f: sha_raw(os.path.join(self.dir, f)) for f in PREP_FILES}

    def subset(self, sets):
        """Probe rows of the given sets (in storage order: P1, P2, P3) -> index array."""
        return np.nonzero(np.isin(self.src, np.array(list(sets))))[0]

    def moments(self, idx, block=8192):
        """mu / sd of X (p1_fuse.standardise) and EX (train.py load) over the probe rows idx, bit for bit as the
        in-memory numpy calls (p09_forward.nan_moments)."""
        idx = np.asarray(idx, dtype=np.int64)

        def xb():
            for s in range(0, len(idx), block):
                yield np.asarray(self.probe_X[idx[s:s + block]], dtype=np.float64)

        def eb():
            for s in range(0, len(idx), block):
                yield np.asarray(self.probe_ex[idx[s:s + block]], dtype=np.float32).astype(np.float64)

        mu, sd = FW.nan_moments(xb)
        mu_x, sd_x = FW.fix_x_moments(mu, sd)
        mu, sd = FW.nan_moments(eb)
        mu_ex, sd_ex = FW.fix_ex_moments(mu, sd)
        return {"mu_x": mu_x, "sd_x": sd_x, "mu_ex": mu_ex, "sd_ex": sd_ex, "rows": int(len(idx))}

    def inner_rows(self, sets=("bank", "mono")):
        return np.nonzero(np.isin(self.inner_src, np.array(list(sets))))[0]


def prep_binding(c, prep, check_rec):
    """The prepared arrays equal what check_report.json recorded -> record (raises on a difference)."""
    want = (check_rec.get("prep") or {}).get("files_sha256") or {}
    got = prep.files_sha256()
    bad = [f for f in PREP_FILES if want.get(f) != got[f]]
    if bad:
        raise P09Error(f"prepared arrays differ from check_report.json: {bad} (rerun prep and check)")
    if (check_rec.get("prep") or {}).get("record_sha256") != prep.record_sha:
        raise P09Error("prep_record.json differs from the one check recorded")
    return {"files_sha256": got, "record_sha256": prep.record_sha}


def save_moments(path, mom, label):
    np.savez(path, format=np.array(FORMATS["moments"]), mu_x=mom["mu_x"], sd_x=mom["sd_x"], mu_ex=mom["mu_ex"],
             sd_ex=mom["sd_ex"], rows=np.array(mom["rows"]), probe_used=np.array(label))


def load_moments(path):
    with np.load(path, allow_pickle=False) as z:
        if str(z["format"]) != FORMATS["moments"]:
            raise P09Error(f"{path}: not a moments file")
        return {k: np.asarray(z[k]) for k in ("mu_x", "sd_x", "mu_ex", "sd_ex")} | \
            {"rows": int(z["rows"]), "probe_used": str(z["probe_used"])}


def load_member(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if str(d.get("format")) != FORMATS["member"]:
        raise P09Error(f"{path}: not a member file")
    kind = str(d["kind"])
    return {"name": str(d["name"]), "kind": kind, "seed": int(d["seed"]), "e_star": int(d["e_star"]),
            "params": {p: d[f"p_{p}"] for p in FW.PARAMS[kind]}, "cv": json.loads(str(d["cv"])),
            "probe_used": str(d["probe_used"]), "rows": int(d["rows"])}


def assemble_map(path, members_dir, record):
    """p1e_map.npz from a training folder (member_<name>.npz + moments.npz) -> map record."""
    mom = load_moments(os.path.join(members_dir, "moments.npz"))
    members = [load_member(os.path.join(members_dir, f"member_{n}.npz")) for n in FW.member_names()]
    for m, spec in zip(members, FW.MEMBERS):
        if (m["name"], m["kind"], m["seed"]) != (spec["name"], spec["kind"], spec["seed"]):
            raise P09Error(f"member file {m['name']} differs from the registered ens3fam member {spec}")
        if m["probe_used"] != mom["probe_used"] or m["rows"] != mom["rows"]:
            raise P09Error(f"member {m['name']} was trained on another probe than moments.npz")
    rec = dict(record, probe_used=mom["probe_used"], rows=mom["rows"],
               cv={m["name"]: m["cv"] for m in members}, e_star={m["name"]: m["e_star"] for m in members})
    FW.save_map(path, members, mom["mu_x"], mom["sd_x"], mom["mu_ex"], mom["sd_ex"], rec)
    return rec


def inner_map_lp(prep, mp, with_members=False):
    """lp_P1TO_E rows of every prepared inner game (numpy float64 forward)."""
    return FW.map_logp(mp, prep.inner_X, prep.inner_ex, with_members=with_members)


def inner_lr_lp(prep, W, b, mu, sd):
    """lp of a multinomial LR map on the inner rows with p08 tune's call pattern (one call for all bank rows, one for
    all mono rows; float64 standardise as p1_fuse.standardise) -> rows aligned with prep.inner_ref."""
    out = np.zeros((len(prep.inner_ref), L))
    for s in ("bank", "mono"):
        rows = prep.inner_rows([s])
        if len(rows):
            X = np.asarray(prep.inner_X[rows[0]:rows[-1] + 1], dtype=np.float64)
            if len(X) != len(rows):
                raise P09Error(f"inner {s} rows are not contiguous")
            out[rows] = FW.lr_logp(W, b, FW.standardise_x(X, mu, sd))
            del X
    return out


# ================================================================ gates, decisions (pure)
def gate_record(rb_new, rb_base, sc_new, sc_base, boot, mono_diff, compared):
    """G1, G2, G4 of PREREG_09 2.5 (candidate vs Q08); the per-question mean difference is recorded only."""
    delta = rb_new - rb_base
    d = boot.stats(sc_new) - boot.stats(sc_base)
    lo, hi = p4.ci95(d)
    g1 = {"threshold": GATE, "delta": delta, "passed": bool(delta >= GATE - TOL)}
    g2 = {"ci95": [lo, hi], "se": float(np.std(d, ddof=1)), "passed": bool(lo > 0),
          "bootstrap": f"players within each level, {rf.N_BOOT} draws, default_rng({rf.BOOT_SEED}), paired"}
    bw = {col: ((mono_diff.get("by_colour") or {}).get(col) or {}).get("rank_balanced") for col in p3.MONO_COLOURS}
    ci = mono_diff.get("ci95") or [None, None]
    g4 = {"rank_balanced_difference": mono_diff.get("rank_balanced"), "ci95": ci,
          "by_colour_point_estimates": bw, "questions": mono_diff.get("n_questions"),
          "registered_questions": MONO_QUESTIONS,
          "passed": bool(mono_diff.get("n_questions") == MONO_QUESTIONS and ci[0] is not None and ci[0] > 0
                         and all(v is not None and v > 0 for v in bw.values())),
          "rule": "all 120,544 single-colour inner pseudo-questions: paired PlayerBootstrap 95 % lower bound > 0, "
                  "black and white differences > 0"}
    qmd = {"question_mean_difference": float(np.mean(np.asarray(sc_new) - np.asarray(sc_base))),
           "note": "recorded only (PREREG_09 2.5)"}
    passed = g1["passed"] and g2["passed"] and g4["passed"]
    return {"compared": compared, "G1": g1, "G2": g2, "G4": g4, "question_mean": qmd, "passed": bool(passed),
            "reason": "G1, G2, G4 passed" if passed else "a gate failed (PREREG_09 2.5)"}


def epoch_rule(oof_p, y):
    """PREREG_09 1.3 item 3: per epoch the mean OOF log-loss (float64 of the float32 probabilities, floor 1e-300);
    e* = the first epoch with the smallest one (1-based) -> (e*, per-epoch log-losses)."""
    oof_p = np.asarray(oof_p)
    n = oof_p.shape[1]
    ll = [float(-np.log(np.clip(oof_p[e][np.arange(n), y].astype(np.float64), 1e-300, None)).mean())
          for e in range(oof_p.shape[0])]
    return int(np.argmin(ll)) + 1, ll


def apply_decision(same_as_08, fail_count, q02_ok, q02h_ok, q08_ok, direction, formats_ok, audit):
    """usable of PREREG_09 2.7 -> (False / 'pending_audit', reasons). apply never writes True (finalize does)."""
    reasons = []
    if not same_as_08.get("passed"):
        reasons.append(f"usable mask / failures differ from 08's val_p1.npz / val_lp_p1.npz: {same_as_08}")
    if fail_count != 0:
        reasons.append(f"validation P1: {fail_count} failed games (registered 0)")
    if q02_ok is not True:
        reasons.append("Q02 does not reproduce results/prereg04/rank/rank_val_probs.csv byte for byte")
    if q02h_ok is not True:
        reasons.append("Q02H does not reproduce results/prereg07/rank/rank_val_probs.csv byte for byte")
    if q08_ok is not True:
        reasons.append("Q08 does not reproduce results/prereg08/rank/rank_val_probs.csv byte for byte")
    if not direction.get("passed"):
        reasons.append(f"direction check: rho {direction.get('rho')} < {direction.get('threshold')}")
    if formats_ok is not True:
        reasons.append("check_submission did not pass for pred_A.csv and pred_B.csv")
    if audit == "bug":
        reasons.append("apply_audit.json found a program error affecting the predictions")
    if reasons:
        return False, reasons
    return "pending_audit", ["apply_audit.json of these outputs not committed yet (PREREG_09 2.7); then "
                             "apply --finalize"]


AUDIT_BOUND_OUTPUTS = ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "val_lp_p1e.npz")
AUDIT_BOUND_INPUTS = ("frozen_config.json", "map", "val_report.json (08)")


def audit_binding(audit, rep, audit_commit, apply_commit):
    probs = []
    if audit.get("format") != FORMATS["audit"]:
        probs.append(f"audit format {audit.get('format')!r} != {FORMATS['audit']!r}")
    if not isinstance(audit.get("program_errors_affecting_predictions"), list):
        probs.append("audit field program_errors_affecting_predictions is not a list")
    for n_ in AUDIT_BOUND_OUTPUTS:
        if (audit.get("outputs_sha256") or {}).get(n_) != (rep.get("outputs_sha256") or {}).get(n_):
            probs.append(f"audit outputs_sha256[{n_}] differs from apply_report.json")
    for n_ in AUDIT_BOUND_INPUTS:
        if (audit.get("inputs_sha256") or {}).get(n_) != (rep.get("inputs_sha256") or {}).get(n_):
            probs.append(f"audit inputs_sha256[{n_}] differs from apply_report.json")
    if (not apply_commit or not audit_commit
            or datetime.datetime.fromisoformat(audit_commit) <= datetime.datetime.fromisoformat(apply_commit)):
        probs.append(f"audit commit {audit_commit} is not later than apply_report.json's {apply_commit}")
    return probs


def audit_state(rep=None):
    p = os.path.join(OUT, "apply_audit.json")
    if not os.path.exists(p):
        return "missing", None
    if tracked_clean([p]):
        return "missing", {"note": "apply_audit.json exists but is not committed"}
    a = read_json(p)
    if a.get("format") != FORMATS["audit"]:
        return "missing", {"note": "apply_audit.json has the wrong format"}
    rec = {"sha256": sha_lf(p), "commit": commit_time(p)}
    if rep is not None:
        probs = audit_binding(a, rep, rec["commit"], commit_time(os.path.join(OUT, "apply_report.json")))
        if probs:
            rec["binding_problems"] = probs
            return "unbound", rec
    return ("bug" if a.get("program_errors_affecting_predictions") else "ok"), rec


def finalize_decision(rep, audit, audit_rec):
    if rep.get("usable") is not False and rep.get("usable") != "pending_audit":
        raise P09Error(f"usable is already {rep.get('usable')!r}")
    if audit == "missing":
        raise P09Error("apply_audit.json is not committed (finalize only after the audit)")
    if audit == "unbound":
        raise P09Error(f"apply_audit.json is not an audit of these outputs: {(audit_rec or {}).get('binding_problems')}")
    blocked = [r for r in rep.get("usable_blocked_reasons") or [] if not r.startswith("apply_audit.json")]
    if rep.get("usable") is False and blocked:
        raise P09Error(f"usable is false for other reasons: {blocked}")
    if audit == "bug":
        return False, ["apply_audit.json found a program error affecting the predictions"]
    if audit != "ok":
        raise P09Error(f"audit state {audit!r}")
    return True, []


def frozen_usable(fc):
    if fc.get("format") != FORMATS["frozen"]:
        raise P09Error("frozen_config.json is not a c10-prereg09 frozen configuration")
    if fc.get("preview"):
        raise P09Error("frozen_config.json is a preview record (never accepted by a later step)")
    if fc.get("precommit"):
        raise P09Error("frozen_config.json was written in --precommit mode (rerun tune in real mode)")
    if fc.get("gates_passed") is not True or (fc.get("combo") or {}).get("id") not in ("Q09", "Q08P"):
        raise P09Error("frozen_config.json does not record passed gates for Q09 / Q08P")
    return True


# ================================================================ the inner evaluation (port of ev_lib.Inner)
class Inner:
    """u1_lib's inner inputs (training players only; hs_lib / u1_lib install the data/local guard), the HS baseline,
    the masked E_HS, Q02, Q02H and Q08 (frozen p1_map.npz through the prepared inner rows with p08 tune's call
    pattern). evaluate(lp_rows) puts a candidate's lp in the P1TO slot (x 4) through p08_rank_fuse.run_terms."""

    def __init__(self, prep, log=print):
        t0 = time.time()
        self.PF = R.import_pilot_fuse()
        _L1, self.F = R.import_p1()
        self.U = R._import_u1()
        d = self.U.load_inputs(verbose=False)
        self.d = d
        inner = d["inner"]
        self.n_inner = inner["keys"]["n"]
        lp_hs, _cov, self.hs_rec = self.F.hs_baseline(self.PF, self.U, d)
        d["lp"]["HS"] = lp_hs
        hs_usable, _f, _c = R.pilot_hs_usable(self.PF, self.n_inner)
        self.E_hs, self.hs_info = R.masked_inner(d, lp_hs, hs_usable)
        self.ref = prep.inner_ref
        if len(np.unique(self.ref)) != len(self.ref) or (len(self.ref) and self.ref.max() >= self.n_inner):
            raise P09Error("prepared inner refs are not distinct inner rows")
        self.covered = np.zeros(self.n_inner, dtype=bool)
        self.covered[self.ref] = True
        self.Xf, self.Xe, self.Xm, self.ms = inner["X_fit"], inner["X_eval"], inner["X_mono"], inner["ms"]
        self.boot, self.mboot = inner["boot"], inner["mono_boot"]
        self.q02 = self.run(Q02_TERMS, {})
        self.h = self.run(Q02H_TERMS, {"HS": self.E_hs})
        with np.load(P08_MAP, allow_pickle=False) as z:
            m08 = {k: np.asarray(z[k]) for k in ("W", "b", "mu", "sd")}
        self.lp_q08_rows = inner_lr_lp(prep, m08["W"], m08["b"], m08["mu"], m08["sd"])
        self.q08 = self.evaluate(self.lp_q08_rows)
        log(f"Inner ready in {time.time() - t0:.0f}s: Q02H {self.h['rb']!r}, Q08 {self.q08['rb']!r}, "
            f"Q08 single-colour {self.q08['mono_rb']!r}")

    def run(self, terms, Emap):
        P, Pm, folds, aux = R.run_terms(self.d, terms, Emap)
        if np.isnan(Pm).any():
            raise P09Error("a single-colour question without an OOF fold")
        m, sc = rf.metrics(rf.rule_b_np(P), self.Xe.y, self.Xe.n, self.Xe.player)
        mrec, msc = p3.mono_score_record(rf.rule_b_np(Pm), self.Xm, self.ms, self.mboot)
        for fr in folds:
            fr["params"] = p4.gen_params(fr["theta"], terms)
        return {"P": P, "Pm": Pm, "folds": folds, "terms": terms, "m": m, "sc": sc, "mrec": mrec, "msc": msc,
                "aux": aux, "rb": m["rank_balanced"], "mono_rb": mrec["rank_balanced"]}

    def lp_full(self, lp_rows):
        lp = np.zeros((self.n_inner, L))
        lp[self.ref] = lp_rows
        return lp

    def E_p1(self, lp_rows, scale=SCALE):
        return R.masked_inner(self.d, self.lp_full(lp_rows), self.covered, scale=scale)

    def evaluate(self, lp_rows, scale=SCALE):
        E, _info = self.E_p1(lp_rows, scale)
        return self.run(Q08_TERMS, {"HS": self.E_hs, "P1TO": E})

    def compare(self, r, base):
        dd = self.boot.stats(r["sc"]) - self.boot.stats(base["sc"])
        md = p3.mono_diff_record(self.ms, self.mboot, r["msc"], base["msc"])
        return {"inner_diff": r["rb"] - base["rb"], "inner_ci95": [float(np.percentile(dd, 2.5)),
                                                                   float(np.percentile(dd, 97.5))],
                "mono_diff": md["rank_balanced"], "mono_ci95": md["ci95"],
                "mono_by_colour": {c: (md["by_colour"][c] or {}).get("rank_balanced") for c in md["by_colour"]}}

    def short(self, r, base=None):
        out = {"inner_rank_balanced": r["rb"], "inner_question_mean": r["m"]["question_mean"],
               "single_colour_rank_balanced": r["mono_rb"],
               "single_colour_by_colour": {c: r["mrec"]["by_colour"][c]["rank_balanced"] for c in r["mrec"]["by_colour"]},
               "fusion_weights_by_fold": [fr["params"]["w"] for fr in r["folds"]]}
        if base is not None:
            out["vs"] = self.compare(r, base)
        return out


def leaves_close(a, b, tol=TOL):
    return P7.leaves_close(a, b, tol=tol)


def direction_ref(inner, E_cand, q02):
    """rho_eval, rho_mono: Spearman between the expected level of softmax(E_P1E) (masked mean of 4 lp) and Q02's OOF
    expected level, questions with >= 1 usable game (as PREREG_08 / tools/p08_rank_fuse.py tune)."""
    d = inner.d["inner"]
    has_e = P7.questions_with_usable(d["S_eval"], inner.covered)
    has_m = P7.questions_with_usable(d["S_mono"], inner.covered)
    rho_eval, n_e = P7.hs_rho(E_cand["eval"], q02["P"], has_e)
    rho_mono, n_m = P7.hs_rho(E_cand["mono"], q02["Pm"], has_m)
    return {"rho_eval": rho_eval, "eval_questions": n_e, "rho_mono": rho_mono, "mono_questions": n_m,
            "floor_validation": RHO_FLOOR_VAL, "floor_test_per_set": RHO_FLOOR_TEST,
            "validation_threshold": max(0.5 * rho_eval, RHO_FLOOR_VAL),
            "test_threshold": max(0.5 * rho_mono, RHO_FLOOR_TEST),
            "definition": "Spearman (average ranks) between sum_k k softmax(E_P1)_k (E_P1 = masked mean of 4 lp of "
                          "the candidate map) and sum_k k P_Q02,k (Q02 OOF), questions with >= 1 usable game"}


# ================================================================ check (PREREG_09 2.2)
def cmd_check(a):
    t0 = time.time()
    c = resolve(a, "check")
    blas_env_check(c.preview)
    rerun = guard_outputs(c, a, ["check_report.json"])
    ts = tool_state()
    if not c.preview:
        require_tools(ts, c.precommit)
    probs, notes = [], []
    try:
        pins = check_pins()
    except P09Error as e:
        pins = None
        probs.append(str(e))
    man, man_sha, man_commit = manifest_binding(c, c.precommit)
    prep = Prep(c.map_dir)
    pr = prep.record
    # 1 the P3 manifest
    if man is None:
        notes.append("preview without p3_manifest.json: P3 not judged")
        sets = ["p1_probe", "p2_probe"]
    else:
        sets = probe_sets_from_manifest(man)
        ex = man.get("exploration_scripts_sha256_lf") or {}
        if pins is not None and any(ex.get(k) != pins.get(k) for k in EXPLORE_PINS):
            probs.append("p3_manifest.json recorded other exploration-script SHA-256s than the pinned ones")
        mc = man.get("manifest_code_sha256") or {}
        for k in ("results/prereg08/p1/p1_lib.py", "results/prereg08/p1/p1_extract.py"):
            if mc.get(k) != R.PINNED[k]:
                probs.append(f"p3_manifest.json was not made with the pinned {k}")
        p3c = man.get("p3_chunks") or {}
        if p3c.get("refs_match_sets") is not True and (man.get("p3") or {}).get("used"):
            probs.append("p3_manifest.json: P3 chunk refs differ from sets.json")
        p3r = man.get("p3") or {}
        if p3r.get("used") is True:
            if not (p3r.get("complete") is True and int(p3r.get("failed", 10 ** 9)) <= P3_FAIL_MAX):
                probs.append(f"p3_manifest.json p3 record inconsistent with PREREG_09 1.2: {p3r}")
            probs.extend(verify_p3_chunks(man))
            if p3c.get("games_planned") != P3_GAMES:
                probs.append(f"P3 sets.json has {p3c.get('games_planned')} games, registered {P3_GAMES}")
        if (man.get("sets_sha256") or {}).get("data/prereg09/p3/sets.json") != DATA_PINS["data/prereg09/p3/sets.json"]:
            probs.append("p3_manifest.json does not record the pinned P3 sets.json")
    label = probe_used_label(sets)
    if pr.get("probe_used") != label:
        if c.preview:
            notes.append(f"preview prep probe {pr.get('probe_used')} (manifest rule says {label})")
            label = pr.get("probe_used")
        else:
            probs.append(f"prep used probe {pr.get('probe_used')}, the manifest rule says {label}")
    if not c.preview and pr.get("limited"):
        probs.append("prep_record.json is a limited preview prep")
    if not c.preview and pr.get("preview"):
        probs.append("prep_record.json is a preview prep")
    if man is not None and pr.get("p3_manifest_sha256") != man_sha:
        probs.append("prep was made from another p3_manifest.json")
    # 2 P1 / P2 / bank / mono chunks equal 08's inner_manifest.json
    man08 = R.read_json(P08_MANIFEST)
    names08 = list(R.P1_SET_NAMES) + ["p2_probe"]
    t = time.time()
    c08 = R.verify_manifest_chunks(man08, names08)
    probs.extend(c08)
    chunks_rec = {"p08_sets_verified": names08, "problems": c08, "seconds": round(time.time() - t, 1)}
    if pr.get("p08_inner_manifest_sha256") != P08_FROZEN_PINS["results/prereg08/feat/inner_manifest.json"]:
        probs.append("prep did not bind 08's inner_manifest.json")
    # 4 disjointness (P3 players vs inner / validation; P3 games vs P1 / P2)
    dj = (man or {}).get("disjoint") or {}
    if man is not None and dj.get("passed") is not True:
        probs.append(f"P3 players / games not disjoint: {dj}")
    dj08 = man08.get("disjoint") or {}
    if dj08.get("passed") is not True:
        probs.append("08 inner_manifest.json disjoint record did not pass")
    # 5 bank / mono usable games
    n_inner = len(prep.inner_ref)
    inner_counts = {s: int((prep.inner_src == s).sum()) for s in ("bank", "mono")}
    if n_inner != INNER_USABLE:
        (notes if c.preview else probs).append(f"bank / mono usable games {n_inner}, registered {INNER_USABLE}")
    # the probe rows in the prepared arrays
    pc = {s: int((prep.src == s).sum()) for s in PROBE_SETS}
    rec_sets = pr.get("sets") or {}
    for s in PROBE_SETS:
        if pc[s] != ((rec_sets.get(s) or {}).get("used") or 0):
            probs.append(f"{s}: {pc[s]} prepared rows, prep_record.json says {(rec_sets.get(s) or {}).get('used')}")
    pyc = no_pycache()
    if pyc:
        probs.append(f"__pycache__ present: {pyc}")
    t = time.time()
    prep_rec = {"record_sha256": prep.record_sha, "files_sha256": prep.files_sha256(),
                "probe_used": pr.get("probe_used"), "limited": pr.get("limited"),
                "probe_rows": pc, "inner_rows": inner_counts, "seconds_hashing": None}
    prep_rec["seconds_hashing"] = round(time.time() - t, 1)
    rep = {"format": FORMATS["check"], "what": "PREREG_09 2.2 check: are the inputs usable (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c), "rerun": rerun,
           "status": "ok" if not probs else "stop", "problems": probs, "notes": notes,
           "p3_manifest": {"sha256": man_sha, "commit": man_commit, "p3": (man or {}).get("p3")},
           "probe_used": label, "probe_sets": sets, "p08_chunks": chunks_rec, "disjoint_p3": dj,
           "disjoint_p08": dj08, "inner_usable_games": n_inner, "prep": prep_rec, "pinned_sha256": pins,
           "tool": ts, "environment": environment_cpu(), "no_pycache_after": not no_pycache()}
    write_json(os.path.join(c.out_dir, "check_report.json"), rep)
    print(f"check: {rep['status']}" + ("" if not probs else f" ({'; '.join(probs)})"))
    if probs:
        raise Stop("check says stop")
    return 0


def check_inputs_unchanged(c, cr):
    """The inputs a committed check_report.json judged are unchanged: the pins, the p3_manifest.json SHA-256 and,
    when P3 is used, its chunks (the prepared arrays are checked by prep_binding) -> manifest record."""
    check_pins()
    man, man_sha, man_commit = manifest_binding(c, c.precommit, required=not c.preview)
    if man_sha != (cr.get("p3_manifest") or {}).get("sha256"):
        raise P09Error("p3_manifest.json differs from the one check used")
    if man is not None and (man.get("p3") or {}).get("used"):
        cp = verify_p3_chunks(man)
        if cp:
            raise P09Error(f"P3 chunks differ from p3_manifest.json: {cp[:5]}")
    return {"sha256": man_sha, "commit": man_commit}


def check_prior(c, ts, precommit):
    """check_report.json passed in this mode (committed in real mode) -> (report, sha, commit). A committed check
    stays valid while its inputs are unchanged: the callers re-verify them (the manifest SHA, the prepared arrays'
    SHA-256s, the pins, the P3 chunks), not the identity of the tool files."""
    cr, cr_sha, cr_commit = prior_report(c, "check_report.json", FORMATS["check"], precommit)
    if cr.get("status") != "ok":
        raise P09Error("check_report.json says stop")
    return cr, cr_sha, cr_commit


# ================================================================ tune (PREREG_09 2.4-2.6)
def torch_check(prep, map_path):
    """PREREG_09 2.6 on every prepared inner game -> p09_map.torch_numpy_max_diff's record. Judged
    (max_abs_lp_diff <= 1e-4): torch eval forward with the stored float32 weights cast to float64 and float64
    standardised input, against the numpy float64 forward (ensemble lp). Recorded only: torch float32 as trained
    (float32 input) against numpy (lp and probability differences)."""
    import p09_map
    return p09_map.torch_numpy_max_diff(map_path, prep.inner_X, prep.inner_ex)


def train_binding(c, ts, fallback, precommit):
    """train_report.json (or train_fallback_report.json) in this mode, made by the current tools and the prepared
    arrays check recorded; its weight files unchanged -> (report, sha, folder)."""
    name = "train_fallback_report.json" if fallback else "train_report.json"
    fmt = FORMATS["train_fallback" if fallback else "train"]
    p = os.path.join(c.out_dir, name)
    if not os.path.exists(p):
        raise P09Error(f"{name} missing: run p09_map.py train{' --fallback' if fallback else ''} first")
    tr = read_json(p)
    if tr.get("format") != fmt or bool(tr.get("preview")) != c.preview:
        raise P09Error(f"{name}: wrong format or mode")
    if tr.get("complete") is not True:
        raise P09Error(f"{name}: training not complete")
    diff = code_differs(ts, tr.get("tool"), TRAINING_CODE)
    if diff:
        raise P09Error(f"{name} was made by other training code: {diff}")
    folder = path_of(tr["folder"]) if not os.path.isabs(tr["folder"]) else tr["folder"]
    bad = [f for f, sha in (tr.get("files_sha256") or {}).items()
           if not os.path.exists(os.path.join(folder, f)) or sha_raw(os.path.join(folder, f)) != sha]
    if bad or not tr.get("files_sha256"):
        raise P09Error(f"training outputs missing or changed since {name}: {bad}")
    return tr, pin_sha(p), folder


def cmd_tune(a):
    t0 = time.time()
    c = resolve(a, "tune")
    blas_env_check(c.preview)
    fb = bool(getattr(a, "fallback", False))
    venv = venv_check(with_torch=not fb, preview=c.preview)
    rep_name = "tune_fallback_report.json" if fb else "tune_report.json"
    map_name = "p1p_map.npz" if fb else "p1e_map.npz"
    frozen_path = os.path.join(c.out_dir, "frozen_config.json")
    map_path = os.path.join(c.out_dir, map_name)
    if os.path.exists(frozen_path) and (c.preview is False and git(["ls-files", "--error-unmatch", "--",
                                                                       rel(frozen_path)])[0] == 0):
        raise P09Error("frozen_config.json is committed (frozen); tune never overwrites it")
    ts = tool_state()
    if not c.preview:
        require_tools(ts, c.precommit)
    first = None
    if fb:
        # Q08P only after Q09 was judged and did not pass (PREREG_09 2.4)
        first, first_sha, first_commit = prior_report(c, "tune_report.json", FORMATS["tune"], False)
        fallback_allowed(first)
        if os.path.exists(frozen_path):
            raise P09Error("frozen_config.json exists: Q09 passed, Q08P is never fitted")
    rerun = guard_outputs(c, a, [rep_name, "frozen_config.json", map_name])
    old = os.path.join(c.out_dir, rep_name)
    rerun = tune_rerun_check(rerun, c.preview, getattr(a, "revision", None),
                             read_json(old).get("created") if os.path.exists(old) else None)
    cr, cr_sha, cr_commit = check_prior(c, ts, c.precommit)
    sr, sr_sha, sr_commit = prior_report(c, "selfcheck.json", FORMATS["selfcheck"], c.precommit)
    diff = code_differs(ts, sr.get("tool"), TRAINING_CODE)
    if diff:
        raise P09Error(f"training code changed since selfcheck: {diff}")
    if (sr.get("check_report") or {}).get("sha256") != cr_sha:
        raise P09Error("selfcheck.json was made from another check_report.json")
    man, man_sha, man_commit = manifest_binding(c, c.precommit)
    if man_sha != (cr.get("p3_manifest") or {}).get("sha256"):
        raise P09Error("p3_manifest.json differs from the one check used")
    if man is not None and (man.get("p3") or {}).get("used"):
        cp = verify_p3_chunks(man)
        if cp:
            raise P09Error(f"P3 chunks differ from p3_manifest.json: {cp[:5]}")
    prep = Prep(c.map_dir)
    prep_rec = prep_binding(c, prep, cr)
    tr, tr_sha, tr_folder = train_binding(c, ts, fb, c.precommit)
    tr_path = os.path.join(c.out_dir, "train_fallback_report.json" if fb else "train_report.json")
    sc_override = selfcheck_accept(c, sr, (tr.get("selfcheck_override") or {}).get("revision"))
    if tr.get("prep_record_sha256") != prep.record_sha or tr.get("check_report_sha256") != cr_sha:
        raise P09Error("training was made from other prepared arrays / another check_report.json")
    pins = check_pins()
    inner = Inner(prep)
    probs = []
    seam = {"Q02H": inner.h["rb"], "Q08": inner.q08["rb"], "Q08_mono": inner.q08["mono_rb"]}
    for k in seam:
        if abs(seam[k] - REF[k]) > TOL:
            probs.append(f"{k} {seam[k]!r} != {REF[k]!r} (seam, PREREG_09 2.3 item 1)")
    if probs and not c.preview:
        raise Stop(f"fusion seam failed: {probs}")
    os.makedirs(TMP, exist_ok=True)
    tmp_map = os.path.join(TMP, f"{map_name[:-4]}_{os.getpid()}.npz")
    record_only = {}
    if not fb:
        cid = "Q09"
        map_rec = assemble_map(tmp_map, tr_folder, {"probe_used": tr.get("probe_used"), "games": tr.get("rows"),
                                                    "players": tr.get("players"), "sets_sha256": tr.get("sets_sha256"),
                                                    "train_report_sha256": tr_sha})
        mp = FW.load_map(tmp_map)
        lp_rows, mem_rows = inner_map_lp(prep, mp, with_members=True)
        cand = inner.evaluate(lp_rows)
        # record only: each member alone (x 4), the P1+P2-only Q09 of selfcheck (the model-swap ablation)
        record_only["members_alone_x4"] = {}
        for spec, lr_ in zip(FW.MEMBERS, mem_rows):
            record_only["members_alone_x4"][spec["name"]] = inner.short(inner.evaluate(lr_), inner.q08)
        del mem_rows
        record_only["selfcheck_P1P2_Q09"] = selfcheck_ensemble_record(c, prep, inner, sr)
    else:
        cid = "Q08P"
        fit = load_fallback_fit(os.path.join(tr_folder, "p1p_fit.npz"))
        lp_rows = inner_lr_lp(prep, fit["W"], fit["b"], fit["mu"], fit["sd"])
        cand = inner.evaluate(lp_rows)
        map_rec = {"probe_used": fit["probe_used"], "games": fit["games"], "players": fit["players"],
                   "lambda": fit["lambda"], "train_fallback_report_sha256": tr_sha}
        R.save_p1_map(tmp_map, {"mu": fit["mu"], "sd": fit["sd"], "lambda": fit["lambda"], "cv": fit["cv"],
                                "model": {"W": fit["W"], "b": fit["b"], "loss": fit["refit_loss"]}},
                      {"probe_used": fit["probe_used"], "games": fit["games"], "players": fit["players"],
                       "sets_sha256": fit["sets_sha256"]})
        m2 = R.load_p1_map(tmp_map)
        if not all(np.array_equal(m2[k], fit[k]) for k in ("mu", "sd", "W", "b")):
            os.remove(tmp_map)
            raise P09Error("p1p_map.npz round trip differs")
    # round trip of the map file (the lp of the gates come from the saved file)
    q, h, e8 = inner.q02, inner.h, inner.q08
    E_cand, cand_info = inner.E_p1(lp_rows)
    gates = gate_record(cand["rb"], e8["rb"], cand["sc"], e8["sc"], inner.boot,
                        p3.mono_diff_record(inner.ms, inner.mboot, cand["msc"], e8["msc"]), f"{cid} vs Q08")
    mono_diff = p3.mono_diff_record(inner.ms, inner.mboot, cand["msc"], e8["msc"])
    rows = {}
    for nm, r_ in (("Q02", q), ("Q02H", h), ("Q08", e8), (cid, cand)):
        m = dict(r_["m"])
        m.update(inner.boot.summary(r_["sc"], None if nm == "Q08" else e8["sc"]))
        for k in ("diff_vs_R00_se", "diff_vs_R00_ci95"):
            if k in m:
                m[k.replace("R00", "Q08")] = m.pop(k)
        rows[nm] = {"terms": r_["terms"], "oof": m, "single_colour": p4.short_mono(r_["mrec"]),
                    "folds": [{"fold": fr["fold"], "loss": fr["loss"], "params": fr["params"],
                               "bounds_hit": P7.bound_hits(fr["params"]["w"])} for fr in r_["folds"]]}
    rows[cid]["oof"]["per_level_diff_vs_Q08"] = {lv: cand["m"]["per_level"][lv] - e8["m"]["per_level"][lv]
                                                 for lv in cand["m"]["per_level"]}
    se = p3.se_val_record(cand["rb"] - e8["rb"], cand["sc"] - e8["sc"], inner.Xe.y)
    se["adoption_probability_at_fraction_of_delta"] = p4.adoption(se["delta"], se["se_val"])
    dref = direction_ref(inner, E_cand, q)
    mono_unusable = int(sum(1 for g in inner.ms["groups"] if not inner.covered[np.asarray(g)].all()))
    # refit on all inner fit pseudo-questions (p04 gen_fit_record, as 08)
    theta, refit = p4.gen_fit_record(cand["aux"]["mf"], cand["aux"]["Ef"], inner.Xf.n, inner.Xf.y)
    params = p4.gen_params(theta, Q08_TERMS)
    Pfit = R.fuse_probs(theta, cand["aux"]["mf"], cand["aux"]["Ef"], inner.Xf.n, np.ones(len(inner.Xf), dtype=bool))
    mfit, _ = rf.metrics(rf.rule_b_np(Pfit), inner.Xf.y, inner.Xf.n, inner.Xf.player)
    refit.update(fit_questions=len(inner.Xf), params=params, bounds_hit=P7.bound_hits(params["w"]),
                 in_sample_fit_rank_balanced=mfit["rank_balanced"], in_sample_fit_question_mean=mfit["question_mean"],
                 oof_eval_rank_balanced=cand["rb"])
    # torch vs numpy forward of the same weights (Q09 only; last, after every fusion number is computed)
    tcheck = None
    if not fb:
        tcheck = torch_check(prep, tmp_map)
        tcheck["passed"] = bool(tcheck["max_abs_lp_diff"] <= TORCH_TOL)
        if not tcheck["passed"]:
            probs.append(f"torch vs numpy forward max |lp| difference {tcheck['max_abs_lp_diff']!r} > {TORCH_TOL} "
                         f"(a program error: not frozen; the report records the gates once)")
    passed = bool(gates["passed"]) and not probs
    sets_sha = {"p1": R.DATA_PINS["data/prereg08/p1/sets.json"], "p2": R.DATA_PINS["data/prereg08/p2/sets.json"],
                "p3": DATA_PINS["data/prereg09/p3/sets.json"]}
    environment = {"venv": environment_cpu(), "versions": venv, "training": tr.get("environment"),
                   "training_venv": tr.get("venv")}
    report = {"format": FORMATS["tune_fallback" if fb else "tune"],
              "what": f"PREREG_09 2.4-2.6 tune: {cid} vs Q08 on the inner pseudo-questions, 5-fold OOF, gates G1 G2 G4 "
                      f"(judged once)",
              "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c), "rerun": rerun,
              "combo": cid, "seam": seam, "problems": probs, "selfcheck_override": sc_override,
              "combinations": rows, "gates": gates,
              "gates_passed": passed, "single_colour_minus_Q08": mono_diff, "se_val": se,
              "direction_ref": dref, "refit": refit, "record_only": record_only,
              "masked_E": {"HS": inner.hs_info, "P1": cand_info}, "hs_mapping": inner.hs_rec,
              "mono_questions_with_unusable_p1_game": mono_unusable, "torch_vs_numpy": tcheck, "map": map_rec,
              "questions": {"fit": len(inner.Xf), "eval": len(inner.Xe), "single_colour": len(inner.Xm)},
              "check": {"sha256": cr_sha, "commit": cr_commit}, "selfcheck": {"sha256": sr_sha, "commit": sr_commit},
              "train_report": {"sha256": tr_sha, "folder": tr.get("folder")}, "prep": prep_rec,
              "p3_manifest": {"sha256": man_sha, "commit": man_commit},
              "first_tune_report": None if first is None else {"sha256": first_sha, "commit": first_commit},
              "pinned_sha256": pins, "tool": ts, "environment": environment}
    os.makedirs(c.out_dir, exist_ok=True)
    if not passed:
        os.remove(tmp_map)
        write_json(os.path.join(c.out_dir, rep_name), report)
        print(f"{cid}: NOT frozen (gates passed {gates['passed']}: {gates['reason']}; problems {probs}); no "
              f"frozen_config.json, no {map_name} (PREREG_09 2.5)")
        raise Stop("gates not passed" if not probs else f"tune problems: {probs}")
    os.replace(tmp_map, map_path)
    map_sha = sha_raw(map_path)
    if not fb:
        mp2 = FW.load_map(map_path)
        if not np.array_equal(inner_map_lp(prep, mp2), lp_rows):
            raise P09Error("p1e_map.npz changed while moving it")
    p08f = {r_: pin_sha(path_of(r_)) for r_ in P08_FROZEN_PINS}
    frozen = {
        "format": FORMATS["frozen"],
        "what": f"PREREG_09 competition 2 frozen configuration {cid} (inner only; commit with {map_name} before "
                f"apply)",
        "created": report["created"], **mode_flags(c), "gates_passed": True,
        "combo": {"id": cid, "terms": Q08_TERMS, "path": "gen", "pool": "log", "calibrated": True, "opening": True,
                  "p1_scale": SCALE, "p1_term": "P1TO_E (ens3fam)" if not fb else "P1TO (LR refit)",
                  "map_file": map_name},
        "params": params,
        "folds": [{"fold": fr["fold"], "loss": fr["loss"], "starts": fr.get("starts"), "params": fr["params"],
                   "bounds_hit": P7.bound_hits(fr["params"]["w"])} for fr in cand["folds"]],
        "basis": [[float(x) for x in row] for row in p4.B],
        "formula": "p = softmax(e^a (n/12)^gamma (m_C5 + w_OP m_OP + w_O4 E_O4 + w_SWA E_SWA + w_SWB E_SWB + w_HS E_HS "
                   "+ w_P1 E_P1 + B beta)); E_P1 = mean over the question's usable games (C5 parsed, P1 not failed, "
                   ">= 1 target move) of 4 lp, 0 without one; lp = " +
                   ("log(mean of the 6 ens3fam members' probabilities), numpy float64 forward (tools/p09_forward.py)"
                    if not fb else "log_softmax(W z(x) + b) of p1p_map.npz"),
        "decision_rule": "B, tools/decide.py (argmax p_k + e^-1 (p_{k-1} + p_{k+1}))",
        "map": {"file": map_name, "sha256": map_sha, **map_rec}, "probe_used": map_rec.get("probe_used"),
        "members": ([{k: m[k] for k in ("name", "kind", "seed", "e_star")} for m in FW.load_map(map_path)["members"]]
                    if not fb else None),
        "probe_reason": ((man or {}).get("p3") or {}).get("reason"), "sets_sha256": sets_sha,
        "prereg08_frozen_sha256": p08f, "pinned_sha256": pins, "p3_manifest_sha256": man_sha,
        "gates": gates, "direction_ref": dref, "se_val": se,
        "oof": {k: cand["m"][k] for k in ("rank_balanced", "question_mean", "per_level", "per_n_bin",
                                          "pred_distribution")},
        "Q08_oof_rank_balanced": e8["rb"], "Q02H_oof_rank_balanced": h["rb"], "Q02_oof_rank_balanced": q["rb"],
        "single_colour": {"questions_sha256": inner.ms["sha256"], cid: p4.short_mono(cand["mrec"]),
                          "Q08": p4.short_mono(e8["mrec"]), f"{cid}_minus_Q08": mono_diff},
        "refit": {k: refit[k] for k in ("loss", "starts", "fit_questions", "bounds_hit", "in_sample_fit_rank_balanced",
                                        "oof_eval_rank_balanced")},
        "torch_vs_numpy": tcheck, "selfcheck_override": sc_override,
        "training_seconds": training_seconds(map_rec, tr, fb),
        "check": {"check_report_sha256": cr_sha, "selfcheck_sha256": sr_sha, "p3_manifest_sha256": man_sha,
                  "train_report_sha256": tr_sha, "train_report": os.path.basename(tr_path), "tune_report": rep_name},
        "tool": ts, "environment": environment, "tool_files_at_tune": ts["files_sha256"]}
    write_json(os.path.join(c.out_dir, rep_name), report)
    write_json(frozen_path, frozen)
    g = gates
    print(f"{cid} vs Q08: G1 {g['G1']['delta']!r} G2 {g['G2']['ci95']} G4 {g['G4']['rank_balanced_difference']!r} "
          f"{g['G4']['ci95']} B/W {g['G4']['by_colour_point_estimates']} -> PASSED")
    print(f"wrote {rep_name}, frozen_config.json, {map_name} in {c.out_dir}"
          f"{' (PREVIEW)' if c.preview else '; commit them before apply'}")
    return 0


def selfcheck_ensemble_record(c, prep, inner, sr):
    """Record only: the P1+P2-only Q09 of selfcheck (its weights, x 4) on the inner pseudo-questions."""
    folder = (sr.get("weights") or {}).get("folder")
    shas = (sr.get("weights") or {}).get("files_sha256") or {}
    if not folder or not shas:
        return {"available": False}
    folder = path_of(folder) if not os.path.isabs(folder) else folder
    if any(not os.path.exists(os.path.join(folder, f)) or sha_raw(os.path.join(folder, f)) != s
           for f, s in shas.items()):
        return {"available": False, "note": "selfcheck weights changed or missing"}
    tmp = os.path.join(TMP, f"p1e_map_selfcheck_{os.getpid()}.npz")
    try:
        assemble_map(tmp, folder, {"what": "selfcheck P1+P2 (record only)"})
        lp = inner_map_lp(prep, FW.load_map(tmp))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"available": True, **inner.short(inner.evaluate(lp), inner.q08)}


def fallback_allowed(first):
    """PREREG_09 2.4: Q08P only after Q09 was judged once and its gates did not pass (a program problem such as a
    failed torch check needs a revision and a Q09 rerun, not the fallback)."""
    if (first.get("gates") or {}).get("passed") is not False:
        raise P09Error("the fallback runs only after Q09's committed tune_report.json says the gates did not pass")
    if first.get("problems"):
        raise P09Error(f"Q09's tune_report.json lists program problems {first['problems']}: fix them (revision, "
                       f"tune --rerun); the fallback is not for program errors")
    return True


def training_seconds(map_rec, tr, fb):
    """Per member CV / refit seconds (frozen_config.json, PREREG_09 1.3 item 7)."""
    if fb:
        return {"lr_fit_seconds": tr.get("seconds")}
    out = {}
    for name, cv in (map_rec.get("cv") or {}).items():
        mem = (tr.get("members") or {}).get(name) or {}
        out[name] = {"cv": cv.get("seconds_cv"), "refit": cv.get("seconds_refit"), "member_total": mem.get("seconds"),
                     "reused_file": mem.get("reused")}
    return out


def load_fallback_fit(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if str(d.get("format")) != FORMATS["fallback_fit"]:
        raise P09Error(f"{path}: not a fallback fit file")
    return {"W": d["W"], "b": d["b"], "mu": d["mu"], "sd": d["sd"], "lambda": float(d["lambda"]),
            "cv": json.loads(str(d["cv"])), "refit_loss": float(d["refit_loss"]), "probe_used": str(d["probe_used"]),
            "games": int(d["games"]), "players": int(d["players"]),
            "sets_sha256": json.loads(str(d["sets_sha256"]))}


# ================================================================ apply (PREREG_09 2.7)
def apply_prerequisites():
    """-> (frozen config, 08 val report, records). Real mode only; never reads data/local."""
    fpath = os.path.join(OUT, "frozen_config.json")
    if not os.path.exists(fpath):
        raise P09Error("frozen_config.json missing (tune did not pass or is not run)")
    fc = read_json(fpath)
    frozen_usable(fc)
    mpath = os.path.join(OUT, fc["map"]["file"])
    rec = {"frozen_config": require_committed([fpath, mpath], "frozen_config.json / map (commit them first)")}
    if sha_raw(mpath) != fc["map"]["sha256"]:
        raise P09Error("the map differs from the SHA-256 frozen_config.json records")
    design = [MANIFEST] + [os.path.join(OUT, n) for n in ("check_report.json", "selfcheck.json",
                                                          fc["check"]["train_report"], fc["check"]["tune_report"])]
    rec["design_records"] = require_committed(design, "p3_manifest / check / selfcheck / train / tune reports")
    want = {MANIFEST: fc["check"]["p3_manifest_sha256"],
            os.path.join(OUT, "check_report.json"): fc["check"]["check_report_sha256"],
            os.path.join(OUT, "selfcheck.json"): fc["check"]["selfcheck_sha256"],
            os.path.join(OUT, fc["check"]["train_report"]): fc["check"]["train_report_sha256"]}
    bad = [rel(p) for p, s in want.items() if sha_lf(p) != s]
    if bad:
        raise P09Error(f"design records differ from frozen_config.json: {bad}")
    ts = tool_state()
    require_tools(ts, False)
    diff = tools_equal_record(ts, fc.get("tool"))
    if diff:
        raise P09Error(f"tool files differ from the ones tune recorded: {diff}")
    rec["tool"] = ts
    rec["frozen_commit_time"] = commit_time(fpath)
    rec["pins"] = check_pins()
    vr = R.read_json(P08_VAL_REPORT)
    if vr.get("format") != R.FORMATS["val_report"] or vr.get("mode") != "real" or vr.get("status") != "ok":
        raise P09Error("results/prereg08/feat/val_report.json is not a real, ok 08 validation report")
    if len(vr.get("chunks") or []) != (R.VAL_GAMES + R.RUN["chunk"] - 1) // R.RUN["chunk"]:
        raise P09Error("08 val_report.json does not list the 96 validation chunks")
    return fc, vr, rec


def read_val_ex(vr, vk):
    """The 08 validation chunks (SHA-256 against val_report.json, real mode, the pinned val_keys) -> float32 raw extra
    features [19,156, 44] in val_keys order and the per-chunk SHA-256."""
    parts, refs, shas = [], [], []
    for ch in vr["chunks"]:
        p = path_of(ch["file"])
        s = sha_raw(p)
        if s != ch["sha256"]:
            raise P09Error(f"{ch['file']} changed since 08's val_report.json")
        with np.load(p, allow_pickle=False) as z:
            if str(z["format"]) != R.FORMATS["val_chunk"] or str(z["meta_sha256"]) != R.NEUTRAL_META_SHA:
                raise P09Error(f"{ch['file']}: not an 08 validation chunk with the neutral metadata")
            if str(z["mode"]) != "real" or str(z["keys_sha256"]) != P7.VAL_KEYS_SHA:
                raise P09Error(f"{ch['file']}: not a real-mode chunk of the pinned val_keys")
            G = len(z["ref"])
            refs.append(np.asarray(z["ref"]).astype(np.int64))
            parts.append(FW.ex_raw32({k: np.asarray(z[k]) for k in ("pos_game", "pos_side", "pos_turn", "pos_logp",
                                                                     "pos_entropy")}, G))
        shas.append(s)
    ref = np.concatenate(refs)
    if not np.array_equal(ref, np.arange(vk["n"])):
        raise P09Error("validation chunk refs are not the val_keys rows in order")
    return np.concatenate(parts), shas


def val_probe_duplicates(fc, val_hashes):
    """Record only (PREREG_09 2.7): validation games whose move sequence equals a probe game's (P1, P2 and, when
    used, P3). Reads the training SGFs (c10data, part fit, no data/local)."""
    import c10data
    import p08_p1
    t = time.time()
    paths = [R.P1_SETS, R.P2_SETS] + ([P3_SETS] if "P3" in str(fc.get("probe_used")) else [])
    want = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            want |= {(g["rank"], g["game_id"]) for g in json.load(f)["probe"]["games"]}
    rows = c10data.load_rows(os.path.join(REPO, "data", "training"), R.SPLIT_CSV, part="fit", local_dir=None,
                             check_counts=True)
    probe_h = {p08_p1.move_hash(x["sgf_content"]) for x in rows if (x["rank"], x["game_id"]) in want}
    del rows
    probe_h.discard(None)
    hits = [i for i, h in enumerate(val_hashes) if h is not None and h in probe_h]
    return {"probe_used": fc.get("probe_used"), "probe_games_hashed": len(want), "validation_games": len(val_hashes),
            "duplicates": len(hits), "first_rows": hits[:20], "seconds": round(time.time() - t, 1),
            "note": "record only (expected 0)"}


def q09_core(q, core, vp, ex32, fc, mp_or_lr):
    """Q09 / Q08P on the validation questions: lp of the usable games (P1 not failed, >= 1 target move, C5 parsed)
    with the frozen map, masked E_P1 of 4 lp, the frozen theta; the direction check. Pure (no file, no data/local).
    mp_or_lr: a loaded p1e map (Q09) or an LR dict {mu, sd, W, b} (Q08P). -> dict."""
    vk, S, n, ok, m, E04, P_q02 = q["vk"], q["S"], q["n"], q["ok"], q["m"], q["E04"], q["P"]
    us = FW.usable_games(vp["failed"], vp["counts"])
    X = FW.x_raw(vp["means"], vp["counts"])
    lp = np.zeros((vk["n"], L))
    if us.any():
        if fc["combo"]["id"] == "Q09":
            lp[us] = FW.map_logp(mp_or_lr, X[us], ex32[us])
        else:
            lp[us] = FW.lr_logp(mp_or_lr["W"], mp_or_lr["b"], FW.standardise_x(X[us], mp_or_lr["mu"],
                                                                               mp_or_lr["sd"]))
    del X
    usable = np.asarray(vk["parsed"], dtype=bool) & us
    E, e_info = R.p1_term(S, lp, usable)
    Es = [E04[t] for t in Q02_TERMS] + [core["E_hs"], E]
    P = R.fuse_probs(fc["params"]["theta"], m, Es, n, ok)
    has = P7.questions_with_usable(S, usable) & ok
    rho, nq = P7.hs_rho(E, P_q02, has)
    direction = P7.direction_record(rho, nq, fc["direction_ref"]["rho_eval"], RHO_FLOOR_VAL)
    return {"P": P, "lp": lp, "p1_ok": us, "usable": usable, "E": E, "e_info": e_info, "has": has,
            "direction": direction, "csv_bytes": P7.probs_csv_bytes(vk["groups"], P, ok, n)}


def cmd_apply(a):
    t0 = time.time()
    out_files = ["rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "check_submission.txt",
                 "val_lp_p1e.npz", "apply_report.json"]
    if a.finalize:
        return apply_finalize(a, out_files)
    blas_env_check(False)
    venv = venv_check(with_torch=False, preview=False)
    fc, vr, pre = apply_prerequisites()
    c = Ctx()
    c.out_dir, c.preview, c.precommit = OUT, False, False
    rerun = guard_outputs(c, a, out_files)
    cid = fc["combo"]["id"]
    # ---- 1 Q02 (prereg04's readers), 2 the question file and the validation move sequences (before any guard)
    q = P7.q02_validation()
    if not q["seam"]["passed"]:
        raise Stop(f"Q02 does not reproduce results/prereg04/rank/rank_val_probs.csv ({q['seam']}); nothing written")
    q_sha = P7.check_question_file(q["vk"])
    vk, ok = q["vk"], q["ok"]
    import p04_opp
    import p08_p1
    slots = p04_opp.read_question_slots(P7.QUESTIONS)
    p04_opp.slots_match_keys(slots, vk)
    val_hashes = [p08_p1.move_hash(sgf) for (_q, _s, _c, sgf) in slots]
    del slots
    overlap = val_probe_duplicates(fc, val_hashes)
    # ---- 3 from here on no data/local file is opened (pilot_fuse imports hs_lib and its guard)
    PF = R.import_pilot_fuse()
    with np.load(R.P07_VAL_HS, allow_pickle=False) as z:
        vh = {k: z[k] for k in z.files}
    core = P7.apply_core(PF, q, vh, P7.load_hs_map(R.P07_HS_MAP), R.read_json(R.P07_FROZEN))
    with open(R.P07_PROBS, "rb") as f:
        ref07 = f.read().replace(b"\r\n", b"\n")
    q02h_ok = bool(core["csv_bytes"] == ref07 and hashlib.sha256(core["csv_bytes"]).hexdigest() == R.P07_PROBS_SHA)
    if not q02h_ok:
        raise Stop("Q02H does not reproduce results/prereg07/rank/rank_val_probs.csv byte for byte; nothing written")
    # ---- 4 the 08 validation P1 chunks (SHA-256 against 08's val_report.json) and Q08 with 08's frozen map
    _L1, F = R.import_p1()
    vp = R.read_val_chunks(vr, vk)
    ex32, chunk_shas = read_val_ex(vr, vk)
    with np.load(P08_VAL_P1, allow_pickle=False) as z:
        v1 = {k: z[k] for k in z.files}
    with np.load(P08_VAL_LP, allow_pickle=False) as z:
        v8 = {k: z[k] for k in z.files}
    with np.load(R.P7.VAL_O4, allow_pickle=False) as z:
        o4z = {k: z[k] for k in z.files}
    fc08 = R.read_json(P08_FROZEN)
    q8 = R.q08_core(F, q, core, vp, vh, o4z, R.load_p1_map(P08_MAP), fc08)
    with open(P08_PROBS, "rb") as f:
        ref08 = f.read().replace(b"\r\n", b"\n")
    q08_ok = bool(q8["csv_bytes"] == ref08 and hashlib.sha256(ref08).hexdigest()
                  == P08_FROZEN_PINS["results/prereg08/rank/rank_val_probs.csv"])
    if not q08_ok:
        raise Stop("Q08 does not reproduce results/prereg08/rank/rank_val_probs.csv byte for byte; nothing written")
    # ---- 5 Q09 (or Q08P) with the frozen map: numpy forward, x 4 once
    if cid == "Q09":
        model = FW.load_map(os.path.join(OUT, fc["map"]["file"]))
    else:
        model = R.load_p1_map(os.path.join(OUT, fc["map"]["file"]))
    if fc["combo"]["p1_scale"] != SCALE:
        raise P09Error("frozen_config.json scale is not 4")
    qc = q09_core(q, core, vp, ex32, fc, model)
    same = {"counts_equal_val_p1": bool(np.array_equal(np.asarray(v1["counts"]).astype(np.int64), vp["counts"])),
            "failed_equal_val_p1": bool(np.array_equal(np.asarray(v1["failed"]).astype(bool), vp["failed"])),
            "p1_ok_equal_val_lp_p1": bool(np.array_equal(np.asarray(v8["p1_ok"]).astype(bool), qc["p1_ok"])),
            "usable_equal_val_lp_p1": bool(np.array_equal(np.asarray(v8["usable"]).astype(bool), qc["usable"]))}
    same["passed"] = all(same.values())
    fail_count = int(P7.hs_failures(vp["failed"], vp["n_moves"]).sum())
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, "rank_val_probs.csv")
    with open(out_csv + ".tmp", "wb") as f:
        f.write(qc["csv_bytes"])
    os.replace(out_csv + ".tmp", out_csv)
    lp_path = os.path.join(OUT, "val_lp_p1e.npz")
    np.savez(lp_path + ".tmp.npz", format=np.array(FORMATS["val_lp"]), combo=np.array(cid),
             question_id=np.array(vk["question_id"]), slot=np.array(vk["slot"], dtype=np.int64),
             color=np.array(vk["color"]), lp=qc["lp"], p1_ok=qc["p1_ok"], usable=qc["usable"], scale=np.array(SCALE),
             map_sha256=np.array(fc["map"]["sha256"]), chunk_sha256=np.array(chunk_shas))
    os.replace(lp_path + ".tmp.npz", lp_path)
    import decide
    decide.read_probs(out_csv)
    py = sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r1 = subprocess.run([py, os.path.join(HERE, "decide.py"), out_csv, OUT], capture_output=True, text=True,
                        cwd=REPO, env=env)
    if r1.returncode != 0:
        raise P09Error(f"decide.py failed: {r1.stderr[-2000:]}")
    txt, formats_ok = [], True
    for pred in ("pred_A.csv", "pred_B.csv"):
        r2 = subprocess.run([py, os.path.join(HERE, "check_submission.py"), "rank", os.path.join(OUT, pred),
                             "--questions", P7.QUESTIONS_REL], capture_output=True, text=True, cwd=REPO, env=env)
        txt.append(f"$ check_submission.py rank results/prereg09/rank/{pred} --questions data/local/"
                   f"rank_val_questions.csv\n(exit {r2.returncode})\n{r2.stdout}{r2.stderr}")
        formats_ok = formats_ok and r2.returncode == 0
    with open(os.path.join(OUT, "check_submission.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(txt))
    audit, audit_rec = audit_state()
    usable, reasons = apply_decision(same, fail_count, q["seam"]["passed"], q02h_ok, q08_ok, qc["direction"],
                                     formats_ok, audit)
    pred_8 = rf.rule_b_np(np.where(ok[:, None], q8["P"], 0.0))
    pred_9 = rf.rule_b_np(np.where(ok[:, None], qc["P"], 0.0))
    dl = (pred_9 - pred_8)[ok]
    has = qc["has"] & q8["has"]
    rho_e, n_e = P7.hs_rho(qc["E"], rf.softmax(np.asarray(q8["E_p1"], dtype=np.float64)), has)
    rep = {"format": FORMATS["apply"], "what": f"PREREG_09 2.7 apply: {cid} on the validation questions (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "mode": "real", "rerun": rerun,
           "combo": cid, "prerequisites": pre, "venv": venv,
           "q02_reproduction": q["seam"], "q02h_reproduction": {"passed": q02h_ok},
           "q08_reproduction": {"passed": q08_ok, "computed_sha256": hashlib.sha256(q8["csv_bytes"]).hexdigest()},
           "same_as_08": same, "p1_failures": {"failed_or_no_move": fail_count, "failed": int(vp["failed"].sum()),
                                               "unusable_games": int((~qc["usable"]).sum())},
           "direction": qc["direction"], "masked_E": qc["e_info"], "usable": usable,
           "usable_blocked_reasons": reasons, "audit": audit_rec,
           "record_only": {
               "rule_B_differs_from_Q08": int(((pred_9 != pred_8) & ok).sum()),
               "level_difference_Q09_minus_Q08": {str(int(k)): int(v) for k, v in
                                                  zip(*np.unique(dl, return_counts=True))},
               "rho_E_cand_vs_E_P1_expected_level": {"rho": rho_e, "questions": n_e},
               "validation_probe_duplicates": overlap,
               "Q08_direction": q8["direction"], "Q08_cross_check_mismatches": q8["cross"]["mismatches"]},
           "questions": len(vk["groups"]), "questions_without_parsed_game": int((~ok).sum()),
           "questions_without_usable_p1_game": int((ok & ~qc["has"]).sum()),
           "pred_distribution": dict(zip(LEVELS, np.bincount(pred_9[ok], minlength=L).tolist())),
           "se_val_record_from_tune": fc.get("se_val"),
           "inputs_sha256": {"frozen_config.json": sha_lf(os.path.join(OUT, "frozen_config.json")),
                             "map": sha_raw(os.path.join(OUT, fc["map"]["file"])),
                             "val_report.json (08)": sha_lf(P08_VAL_REPORT), "val_p1.npz (08)": sha_raw(P08_VAL_P1),
                             "p1_map.npz (08)": sha_raw(P08_MAP), "frozen_config.json (08)": sha_lf(P08_FROZEN),
                             "val_hs.npz": sha_raw(R.P07_VAL_HS), "questions": q_sha, "val_keys.npz": vk["sha256"],
                             "val_o4.npz": sha_raw(P7.VAL_O4), **q["vsha"]},
           "val_chunk_sha256": chunk_shas,
           "outputs_sha256": {n_: pin_sha(os.path.join(OUT, n_)) for n_ in out_files[:6]},
           "decide_stdout": r1.stdout[-2000:], "tool": tool_state(), "environment": environment_cpu(),
           "next": "commit; apply_audit.json by another agent (no labels, no scores) -> commit; apply --finalize"}
    write_json(os.path.join(OUT, "apply_report.json"), rep)
    print(f"apply {cid}: Q02 / Q02H / Q08 reproduction {q['seam']['passed']} / {q02h_ok} / {q08_ok}; same as 08 "
          f"{same['passed']}; rho {qc['direction']['rho']:.4f} (threshold {qc['direction']['threshold']:.4f}); "
          f"failed {fail_count}; formats {formats_ok}; rule B differs from Q08 on "
          f"{rep['record_only']['rule_B_differs_from_Q08']} questions")
    print(f"usable: {usable} {reasons}")
    return 0


def apply_finalize(a, out_files):
    rp = os.path.join(OUT, "apply_report.json")
    if require_committed([rp] + [os.path.join(OUT, f) for f in out_files[:6]], "apply outputs")["commit_problems"]:
        raise P09Error("commit apply's outputs first")
    rep = read_json(rp)
    for n_, sha in rep["outputs_sha256"].items():
        if pin_sha(os.path.join(OUT, n_)) != sha:
            raise P09Error(f"{n_} changed since apply")
    audit, audit_rec = audit_state(rep)
    usable, reasons = finalize_decision(rep, audit, audit_rec)
    rep["usable"], rep["usable_blocked_reasons"] = usable, reasons
    rep["audit"] = audit_rec
    rep["finalized"] = now_iso()
    rep["finalized_by_tool"] = tool_state()["files_sha256"]
    write_json(rp, rep)
    print(f"usable: {rep['usable']}")
    return 0


# ================================================================ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--precommit", action="store_true")
        p.add_argument("--out-dir", default=None)
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
        if name == "tune":
            p.add_argument("--fallback", action="store_true")
            p.add_argument("--revision", type=int, default=None)
    p = sub.add_parser("apply")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    fn = {"check": cmd_check, "tune": cmd_tune, "apply": cmd_apply}[a.cmd]
    try:
        return fn(a)
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 1
    except ERRORS as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - p1_lib / hs_lib / u1_lib refusals are refusals too
        if type(e).__name__ in ("P1Error", "HSError", "LocalDataRefused"):
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
