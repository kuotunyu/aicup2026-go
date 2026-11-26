"""PREREG_07 competition 2: Q02H = Q02 + KataGo HumanSL term (docs/PREREG_07_HUMANSL.md rev 1, sections 2.1-2.7;
results/prereg07/design/spec.md sections 0, 2, 4).

    PY = .venv/Scripts/python.exe   (CPU; Git Bash; PYTHONDONTWRITEBYTECODE=1)
    PY tools/p07_rank_fuse.py check      [--precommit | --out-dir data/prereg07/tmp/<x>]  -> results/prereg07/rank/check_report.json
    PY tools/p07_rank_fuse.py selfcheck  [--precommit | --out-dir ...]                    -> .../selfcheck.json
    PY tools/p07_rank_fuse.py tune       [--precommit | --out-dir ...]                    -> .../tune_report.json
                                                                (gates passed: + frozen_config.json, hs_map.npz)
    PY tools/p07_rank_fuse.py apply      [--accept-data-failures TEXT]                  -> rank_val_probs.csv,
                                                                pred_A/B.csv, decide_summary.json,
                                                                check_submission.txt, apply_report.json
    PY tools/p07_rank_fuse.py apply --finalize   (after prescore_audit.json is committed: score_allowed only)

Modes
    real        every input and every earlier step committed and clean (tools included); writes results/prereg07/rank.
    --precommit the same checks and outputs as real, except that "committed and clean" is not required of the p07
                tools, inner_manifest.json, check_report.json and selfcheck.json (they are recorded by SHA-256 instead;
                the report says "precommit": true). For the first build only, when the parent session commits the
                tools and the design reports after reviewing them. Later steps (p07_hs.py val, apply) require the
                committed files to have exactly the SHA-256s recorded here.
    --out-dir   preview under data/prereg07/tmp/ (flagged "preview": true; never accepted by a later step).
apply has neither: it needs the committed frozen_config.json, val_hs.npz, val_report.json and val_seam_moves.npz.

Discipline (spec 0)
    * data/local is read only by apply (the question file's question_id / num_games / color_i columns through the
      prereg04 readers), and only before pilot_fuse / hs_lib are imported. check / selfcheck / tune never read it
      (u1_lib's and hs_lib's import-time guards make it impossible after their import).
    * u1_lib is imported only by check, selfcheck and tune (_import_u1); apply never imports it nor calls
      pilot_fuse.fit_mapping (which imports it).
    * the four pinned files (hs_lib.py, pilot_run.py, pilot_fuse.py, u1_lib.py) are imported only after their SHA-256
      (CRLF folded to LF) equals the pinned value; never copied or edited.
    * temporary files only under data/prereg07/tmp/; never %TEMP%. tools/score.py is never run.
    * the validation label file is never named or opened (every path goes through refuse_label_path).
Exit codes: 0 done; 1 the step ran and says stop (status stop / gates not passed); 2 refused or error.
"""

import argparse
import ast
import csv
import datetime
import glob
import hashlib
import io
import json
import math
import os
import platform
import re
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
import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
import p04_rank_fuse as p4  # noqa: E402

# ================================================================ constants (spec 2; PREREG_07 1.2-1.7)
_LABEL = "ans" + "wer"
LEVELS = rf.LEVELS
L = rf.N_LEVELS
PILOT_CODE = os.path.join(REPO, "results", "prereg07", "explore", "pilot_humansl")
U1_DIR = os.path.join(REPO, "results", "prereg05", "explore", "u1")
PINNED = {  # rel path -> SHA-256 (CRLF folded to LF), PREREG_07 1.2 / spec 0
    "results/prereg07/explore/pilot_humansl/hs_lib.py":
        "d19b6a8615ef757891ff2d451fe5b96b19f31cc706a87862e49a7688d0bde97c",
    "results/prereg07/explore/pilot_humansl/pilot_run.py":
        "b0d127fd1d5a83af5b17ec3df1dccc85d94b11c3479d33cac07448a3796a3936",
    "results/prereg07/explore/pilot_humansl/pilot_fuse.py":
        "37fb2dff0f37032bf08acc7f27213cafd30533ab024973f92b5b28f41c6ac0c6",
    "results/prereg05/explore/u1/u1_lib.py":
        "225dd51cff961ae3608e003432388e0b6d6a053fb4dac26add5425c002dc3ff7"}
TOOL_FILES = ("tools/p07_hs.py", "tools/p07_rank_fuse.py")
NO_PYCACHE_DIRS = ("results/prereg07", "results/prereg05/explore/u1", "third_party/katago")
KATAGO_COMMIT = "d91ea855110dae533f0aada947b2b7d78cc8a4e1"
SOURCE_TXT = os.path.join(REPO, "third_party", "katago", "SOURCE.txt")
SOURCE_TXT_SHA = "039d5a1cbc815b4d69b68dc8c01163354820ac00b5070ef136c14a040b5ba20b"
SOURCE_TXT_ROWS = 15
CKPT = os.path.join(REPO, "data", "katago", "b18c384nbt-humanv0.ckpt")
CKPT_BACKUP = os.path.join(REPO, "data", "katago_backup", "b18c384nbt-humanv0.ckpt")
CKPT_BYTES = 323752318
CKPT_SHA = "84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27"
PROFILES = ["rank_20k", "rank_15k", "rank_12k", "rank_10k", "rank_8k", "rank_6k", "rank_4k", "rank_2k",
            "rank_1d", "rank_2d", "rank_3d", "rank_5d", "rank_7d", "rank_9d"]
RUN = {"precision": "fp16", "batch": 1024, "channels_last": True, "group_batches": 16, "workers": 8,
       "cudnn_benchmark": True, "sides": ["target"], "chunk": 200}
METADATA_TEXT = ("C++ makeBasicRankProfile 'rank_<r>' preset: both players human, the same rank for both sides, "
                 "ratedness unknown, byo-yomi 1200 s + 5 x 30 s, source KGS, date 2020-03-01, board area 361")
RULES_TEXT = "Japanese (KO_SIMPLE, SCORING_TERRITORY, TAX_SEKI, no suicide), komi 6.5, no button, encore 0"
PILOT_FORMAT = "c10-prereg07 pilot chunk v1"
PLAN = os.path.join(REPO, "data", "prereg07", "pilot", "plan.json")
PLAN_SHA = "8494e9afd79c06249bd81545263b86caef7b9e77f3cd631a1e354368cfe33946"
PILOT_DATA = os.path.join(REPO, "data", "prereg07", "pilot")
SET_CHUNKS = {"fitmap": 15, "bank": 60, "mono": 307}
SET_GAMES = {"fitmap": 3000, "bank": 11813, "mono": 61356}
PILOT_REPORT = os.path.join(PILOT_CODE, "pilot_fuse_report.json")
GPU_BENCH = os.path.join(PILOT_CODE, "hs_gpu_bench.json")
INNER_O4 = os.path.join(REPO, "results", "prereg04", "opp", "inner_o4.npz")
VAL_O4 = os.path.join(REPO, "results", "prereg04", "opp", "val_o4.npz")
INNER_KEYS = os.path.join(REPO, "results", "prereg02", "rank", "dumps", "inner_keys.npz")
VAL_KEYS = os.path.join(REPO, "results", "prereg02", "rank", "dumps", "val_keys.npz")
VAL_KEYS_SHA = "055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c"
QUESTIONS = p3.QUESTIONS                                   # data/local/rank_val_questions.csv (apply / val only)
QUESTIONS_REL = "data/local/rank_val_questions.csv"        # relative to REPO (no realpath on it after the guard)
QUESTIONS_SHA = "73e51bbfd7b051e73cbb62af4b180c403d2422f122240c5e67e9c92303776db5"
VAL_QUESTIONS = 1589
VAL_GAMES = 19156
P04_FROZEN = os.path.join(REPO, "results", "prereg04", "rank", "frozen_config.json")
P04_FROZEN_SHA = "ae728e9c936a55dfeced255e9b05b5ac6a4955a7101f108e78c53c679e162661"
P04_PROBS = os.path.join(REPO, "results", "prereg04", "rank", "rank_val_probs.csv")
P04_PROBS_SHA = "3c7ab658dc558ce8c6aefdd7b0747de1a60de575d151e9eee96ca6775f442e5c"
VAL_O4_SHA = "30d312c02c357f6761567099003885be5e08635b6e6c0b49d71baac137cd9cd4"
# pilot numbers selfcheck reproduces (spec 2; PREREG_07 2.3)
PILOT = {"Q02": 0.6101211982700248, "Q02H": 0.6242752597994481, "diff": 0.014154061529423334,
         "ci95": [0.004729748170068845, 0.023937551131530375], "question_mean_diff": 0.012494963827849034,
         "lambda": 1e-4, "cv_logloss": 1.7571755879185869, "refit_loss": 1.7425066323027925, "eval_questions": 127200}
TOL = 1e-12
GATE = 0.010
MONO_QUESTIONS = 120544
MONO_SHA = "45d0f14acd98786adfd7cae040edc551420f095aaf6118d52edb47ddf91a5622"
MONO_FAIL_MAX = 306                       # 0.5 % of 61,356 mono games
NEW_SUBSET_QUESTIONS = 108719
VAL_FAIL_MAX = 95                         # 0.5 % of 19,156 validation games (PREREG_07 2.7)
RHO_FLOOR_VAL = 0.078
RHO_FLOOR_TEST = 0.155
Q02_TERMS = ["OP", "O4", "SWA", "SWB"]
Q02H_TERMS = Q02_TERMS + ["HS"]
R20HS_TERMS = ["OP", "HS"]
OUT = os.path.join(REPO, "results", "prereg07", "rank")
HS_OUT = os.path.join(REPO, "results", "prereg07", "hs")
MANIFEST = os.path.join(HS_OUT, "inner_manifest.json")
VAL_DATA = os.path.join(REPO, "data", "prereg07", "val")
TMP = os.path.join(REPO, "data", "prereg07", "tmp")
PREREG = os.path.join(REPO, "docs", "PREREG_07_HUMANSL.md")
REV1_COMMIT = "5957a01"
FORMATS = {"check": "c10-prereg07 rank check_report v1", "selfcheck": "c10-prereg07 rank selfcheck v1",
           "tune": "c10-prereg07 rank tune_report v1", "frozen": "c10-prereg07 rank frozen_config v1",
           "apply": "c10-prereg07 rank apply_report v1", "hs_map": "c10-prereg07 hs_map v1",
           "manifest": "c10-prereg07 inner_manifest v1", "val_report": "c10-prereg07 val_report v1",
           "audit": "c10-prereg07 prescore_audit v1"}
# functions of this file that may import u1_lib or call pilot_fuse.fit_mapping (tests/test_p07_hygiene.py checks)
INNER_ONLY_FUNCS = ("_import_u1", "cmd_check", "cmd_selfcheck", "cmd_tune", "_inner_setup", "_fit_map")


class P07Error(RuntimeError):
    pass


class Stop(RuntimeError):
    """The step ran; its registered outcome is a stop (exit 1)."""


ERRORS = (P07Error, p4.P04Error, p3.P03Error, rf.P02Error)


# ================================================================ small helpers
def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def refuse_label_path(path):
    if path is not None and _LABEL in os.path.abspath(os.fspath(path)).lower():
        raise P07Error(f"refusing {path}: the validation label file is never opened by the prereg07 tools")
    return path


def under_data_local(path):
    """True for a path with the consecutive components data / local (realpath, component-wise)."""
    s = os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))
    parts = s.split(os.sep)
    return any(a == "data" and b == "local" for a, b in zip(parts, parts[1:]))


def inside(path, root):
    p = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    r = os.path.normcase(os.path.realpath(os.path.abspath(root)))
    return p == r or p.startswith(r + os.sep)


def rel(path):
    return os.path.relpath(os.path.realpath(path), os.path.realpath(REPO)).replace(os.sep, "/")


def path_of(r):
    return os.path.join(REPO, *r.split("/"))


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
    """Text files (.json / .csv / .py / .txt / .md) with CRLF folded to LF, anything else (.npz) raw bytes."""
    return sha_lf(path) if os.path.splitext(path)[1].lower() in (".json", ".csv", ".py", ".txt", ".md") \
        else sha_raw(path)


def read_json(path):
    with open(refuse_label_path(path), encoding="utf-8") as f:
        return json.load(f)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) else f
    return o


def write_json(path, obj):
    refuse_label_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(obj), f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def git(args):
    try:
        r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    return r.returncode, r.stdout.decode("utf-8", "replace").strip()


def git_head():
    rc, out = git(["rev-parse", "HEAD"])
    return out if rc == 0 else None


def commit_time(path):
    rc, out = git(["log", "-1", "--format=%cI", "--", rel(path)])
    return out if rc == 0 and out else None


def tracked_clean(paths):
    """-> list of problems (empty when every path is tracked by git and unchanged in the working tree)."""
    probs = []
    for p in paths:
        r = rel(p)
        rc, _ = git(["ls-files", "--error-unmatch", "--", r])
        if rc != 0:
            probs.append(f"{r} not committed")
            continue
        rc, st = git(["status", "--porcelain", "--", r])
        if rc != 0 or st:
            probs.append(f"{r} changed in the working tree")
    return probs


def require_committed(paths, what, precommit=False):
    """Real mode: every path committed and clean. --precommit: recorded by SHA-256 instead."""
    rec = {"files": {rel(p): pin_sha(p) for p in paths if os.path.exists(p)}}
    missing = [rel(p) for p in paths if not os.path.exists(p)]
    if missing:
        raise P07Error(f"{what}: missing {missing}")
    probs = tracked_clean(paths)
    rec["commit_problems"] = probs
    if probs and not precommit:
        raise P07Error(f"{what}: {probs} (commit each step before the next)")
    rec["precommit_skipped"] = bool(probs and precommit)
    return rec


def tool_state():
    files = list(TOOL_FILES) + list(PINNED)
    rc, st = git(["status", "--porcelain", "--untracked-files=all", "--"] + files)
    dirty = [ln[3:] for ln in st.splitlines() if ln.strip()] if rc == 0 else None
    return {"git_head": git_head(), "dirty_or_untracked": dirty, "files_sha256_rule": "sha256, CRLF folded to LF",
            "files_sha256": {f: (sha_lf(path_of(f)) if os.path.exists(path_of(f)) else None) for f in files}}


def require_tools(ts, precommit):
    if ts["git_head"] is None or ts["dirty_or_untracked"] is None:
        raise P07Error("git is not available")
    if ts["dirty_or_untracked"] and not precommit:
        raise P07Error(f"commit the tools first (uncommitted: {ts['dirty_or_untracked']})")


def tools_equal_record(ts, recorded):
    """The current tool files equal the SHA-256s a frozen record holds -> list of differing files."""
    rec = (recorded or {}).get("files_sha256") or {}
    return [f for f in list(TOOL_FILES) + list(PINNED) if rec.get(f) != ts["files_sha256"].get(f)]


def environment_cpu():
    import scipy
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__}


def nvidia_query():
    """Driver version and GPU names from nvidia-smi (record only; None when unavailable)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        rows = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        out = [dict(zip(("name", "driver"), [y.strip() for y in x.split(",")])) for x in rows]
        return {"gpus": out, "driver": out[0]["driver"] if out else None}
    except (OSError, subprocess.SubprocessError):
        return {"gpus": [], "driver": None}


def no_pycache():
    found = []
    for d in NO_PYCACHE_DIRS:
        for root, dirs, _files in os.walk(path_of(d)):
            found.extend(rel(os.path.join(root, x)) for x in dirs if x == "__pycache__")
    return found


def check_pins():
    """-> {rel: sha}; raises unless every pinned file has its pinned SHA-256 (CRLF folded to LF)."""
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    got = {r: sha_lf(path_of(r)) for r in PINNED}
    bad = [r for r in PINNED if got[r] != PINNED[r]]
    if bad:
        raise P07Error(f"pinned files changed: {bad} (never edit them; PREREG_07 1.2)")
    return got


def import_pilot_fuse():
    """pilot_fuse (and through it hs_lib, whose import installs the data/local guard). Never imports u1_lib."""
    check_pins()
    if PILOT_CODE not in sys.path:
        sys.path.insert(0, PILOT_CODE)
    import pilot_fuse as PF
    for mod, r in ((PF, "pilot_fuse.py"), (PF.H, "hs_lib.py")):
        if not inside(mod.__file__, PILOT_CODE) or os.path.basename(mod.__file__) != r:
            raise P07Error(f"{r} imported from {mod.__file__}")
    return PF


def _import_u1():
    """u1_lib (check / selfcheck / tune only)."""
    check_pins()
    if U1_DIR not in sys.path:
        sys.path.insert(0, U1_DIR)
    import u1_lib as U
    if not inside(U.__file__, U1_DIR):
        raise P07Error(f"u1_lib imported from {U.__file__}")
    return U


def katago_sources():
    """third_party/katago: SOURCE.txt's own SHA-256 and every listed file (LF) equal its row -> (problems, record)."""
    probs = []
    if sha_lf(SOURCE_TXT) != SOURCE_TXT_SHA:
        probs.append("SOURCE.txt SHA-256 differs")
    rows = []
    with open(SOURCE_TXT, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s+(\S+) \| ([0-9a-f]{40}) \| ([0-9a-f]{64}) \| (\d+)\s*$", line)
            if m:
                rows.append(m.groups())
    if len(rows) != SOURCE_TXT_ROWS:
        probs.append(f"SOURCE.txt lists {len(rows)} files, expected {SOURCE_TXT_ROWS}")
    bad = []
    for path, _blob, sha, _n in rows:
        p = os.path.join(REPO, "third_party", "katago", *path.split("/"))
        if not os.path.exists(p) or sha_lf(p) != sha:
            bad.append(path)
    if bad:
        probs.append(f"third_party/katago files differ from SOURCE.txt: {bad}")
    rc, txt = git(["status", "--porcelain", "--untracked-files=all", "--", "third_party/katago"])
    if rc != 0 or txt:
        probs.append(f"third_party/katago not clean in git: {txt[:200]}")
    return probs, {"source_txt_sha256": SOURCE_TXT_SHA, "files": len(rows), "differing": bad}


def checkpoint_record():
    """Full SHA-256 of the HumanSL checkpoint (PREREG_07 1.2), backup recorded."""
    rec = {"path": rel(CKPT), "bytes": os.path.getsize(CKPT) if os.path.exists(CKPT) else None}
    rec["sha256"] = sha_raw(CKPT) if rec["bytes"] == CKPT_BYTES else None
    rec["ok"] = rec["sha256"] == CKPT_SHA
    if os.path.exists(CKPT_BACKUP):
        rec["backup"] = {"path": rel(CKPT_BACKUP), "bytes": os.path.getsize(CKPT_BACKUP)}
    return rec


# ================================================================ masked E_HS, direction check, move cross-check
def usable_games(parsed, failed, counts):
    """C5 parsed, HS not failed, at least one target move (spec 4.5)."""
    return np.asarray(parsed, dtype=bool) & ~np.asarray(failed, dtype=bool) & (np.asarray(counts) > 0)


def masked_E(S, lp, usable, force_mask=False):
    """E_HS(q) = mean of lp_HS over the question's usable games (0 vector without one). S = the C5-parsed pooling
    matrix (rf.pool_matrix). When every game S uses is usable, this is exactly rf.pooled(S, lp, 'log') (bit for bit
    with the pilot); otherwise the masked, renormalised matrix (as t04_rank.fuse_set does for O4).
    -> (E [Q,10], info)."""
    S = S.tocsr()
    usable = np.asarray(usable, dtype=bool)
    used = S.indices
    n_unusable = int((~usable[used]).sum()) if len(used) else 0
    info = {"games_in_questions": int(len(np.unique(used))), "unusable_slots_in_questions": n_unusable,
            "path": "pooled"}
    if n_unusable == 0 and not force_mask:
        return rf.pooled(S, lp, "log"), info
    groups = [[int(c) for c in S.indices[S.indptr[q]:S.indptr[q + 1]] if usable[c]] for q in range(S.shape[0])]
    Sm = rf.pool_matrix(groups, S.shape[1])
    lpm = np.where(usable[:, None], lp, 0.0)
    info.update(path="masked", questions_without_usable_game=int(sum(1 for g in groups if not g)))
    return rf.pooled(Sm, lpm, "log"), info


def questions_with_usable(S, usable):
    S = S.tocsr()
    usable = np.asarray(usable, dtype=bool)
    return np.array([bool(usable[S.indices[S.indptr[q]:S.indptr[q + 1]]].any()) for q in range(S.shape[0])])


def expected_level(P):
    return np.asarray(P, dtype=np.float64) @ np.arange(L, dtype=np.float64)


def hs_rho(E_hs, P_q02, has):
    """Spearman rho between the HS-only expected level softmax(E_HS) and Q02's expected level (spec 4.6)."""
    from scipy import stats
    has = np.asarray(has, dtype=bool)
    eh = expected_level(rf.softmax(np.asarray(E_hs, dtype=np.float64)))
    eq = expected_level(P_q02)
    r = stats.spearmanr(eh[has], eq[has])
    return float(r.statistic if hasattr(r, "statistic") else r[0]), int(has.sum())


def direction_record(rho, n, ref, floor):
    thr = max(0.5 * float(ref), float(floor))
    return {"rho": rho, "questions": n, "reference": ref, "floor": floor, "threshold": thr,
            "passed": bool(rho is not None and not math.isnan(rho) and rho >= thr),
            "rule": "rho >= max(0.5 x reference, floor) (spec 4.6)"}


def move_cross_check(hs_total, hs_target, hs_failed, o_total, o_target, o_failed, ids=None):
    """HS total / target move counts against prereg04's O4 replay (another replay program, same keys), on the games
    where neither side failed (spec 4.7 / 3.1). -> record (mismatches must be 0)."""
    hs_total, hs_target = np.asarray(hs_total, dtype=np.int64), np.asarray(hs_target, dtype=np.int64)
    o_total, o_target = np.asarray(o_total, dtype=np.int64), np.asarray(o_target, dtype=np.int64)
    hf, of = np.asarray(hs_failed, dtype=bool), np.asarray(o_failed, dtype=bool)
    both = ~hf & ~of
    bad = both & ((hs_total != o_total) | (hs_target != o_target))
    idx = np.nonzero(bad)[0][:20]
    first = [{"row": int(i), "id": (None if ids is None else ids[int(i)]), "hs_total": int(hs_total[i]),
              "o4_total": int(o_total[i]), "hs_target": int(hs_target[i]), "o4_target": int(o_target[i])}
             for i in idx]
    return {"compared": int(both.sum()), "mismatches": int(bad.sum()), "first_mismatches": first,
            "only_hs_failed": int((hf & ~of).sum()), "only_o4_failed": int((of & ~hf).sum()),
            "both_failed": int((hf & of).sum())}


def hs_failures(failed, n_moves):
    """'Failed' for the 0.5 % rules = failed or no move at all (spec 4.5)."""
    return np.asarray(failed, dtype=bool) | (np.asarray(n_moves) == 0)


def hs_lp(model, sums, counts, failed):
    """Per-game lp_HS from target-side sums [S,14] and counts [S] with the frozen map (mu, sd, W, b) ->
    (lp [S,10] (0 rows where not computable), ok [S])."""
    sums = np.asarray(sums, dtype=np.float64)
    c = np.asarray(counts).astype(np.float64)
    ok = ~np.asarray(failed, dtype=bool) & (c > 0)
    x = np.zeros_like(sums)
    x[ok] = sums[ok] / c[ok, None]
    lp = np.zeros((len(sums), L), dtype=np.float64)
    if ok.any():
        Z = ((x[ok] - model["mu"]) / model["sd"]) @ model["W"] + model["b"]
        m = np.max(Z, axis=-1, keepdims=True)
        lp[ok] = Z - m - np.log(np.sum(np.exp(Z - m), axis=-1, keepdims=True))   # = hs_lib.log_softmax_np
    return lp, ok


def load_hs_map(path):
    with np.load(path, allow_pickle=False) as z:
        m = {k: z[k] for k in z.files}
    if str(m.get("format")) != FORMATS["hs_map"] or list(m["profiles"]) != PROFILES:
        raise P07Error(f"{path}: not a {FORMATS['hs_map']} file with the registered profiles")
    return {"mu": m["mu"], "sd": m["sd"], "W": m["W"], "b": m["b"], "lambda": float(m["lambda"])}


def fuse_probs(theta, m, Es, n, ok):
    """Q02H probabilities with the generalised calibration (p04 gen_logits) -> [Q,10] (NaN rows when not ok)."""
    P = np.full((len(n), L), np.nan)
    if np.asarray(ok).any():
        z, _ = p4.gen_logits(np.asarray(theta, dtype=np.float64), m[ok], [E[ok] for E in Es], n[ok])
        P[ok] = rf.softmax(z)
    return P


def probs_csv_bytes(groups, P, ok, n):
    """rank_val_probs.csv in the prereg01 layout (p04 cmd_apply's writer), in memory."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(rf.PROB_COLUMNS)
    for k, (q, rows) in enumerate(groups):
        probs_k = [repr(float(x)) for x in P[k]] if ok[k] else [""] * L
        w.writerow([q, len(rows), int(n[k])] + probs_k)
    return buf.getvalue().encode("utf-8")


def gate_record(rb_h, rb_q, sc_h, sc_q, boot, mono_diff, mono_failed):
    """G1-G4 of PREREG_07 2.5, Q02H vs Q02."""
    delta = rb_h - rb_q
    d = boot.stats(sc_h) - boot.stats(sc_q)
    lo, hi = p4.ci95(d)
    g3 = float(np.mean(np.asarray(sc_h) - np.asarray(sc_q)))
    g1 = {"threshold": GATE, "delta": delta, "passed": bool(delta >= GATE - TOL)}
    g2 = {"ci95": [lo, hi], "se": float(np.std(d, ddof=1)), "passed": bool(lo > 0),
          "bootstrap": f"players within each level, {rf.N_BOOT} draws, default_rng({rf.BOOT_SEED}), paired"}
    g3r = {"question_mean_difference": g3, "passed": bool(g3 >= -TOL),
           "note": "mean per-question difference on the inner eval pseudo-questions, >= 0 with 1e-12 slack"}
    bw = {col: (mono_diff["by_colour"].get(col) or {}).get("rank_balanced") for col in p3.MONO_COLOURS}
    g4_core = bool(mono_diff["ci95"][0] > 0 and all(v is not None and v > 0 for v in bw.values()))
    g4 = {"rank_balanced_difference": mono_diff["rank_balanced"], "ci95": mono_diff["ci95"],
          "by_colour_point_estimates": bw, "questions": mono_diff["n_questions"], "mono_failed_games": mono_failed,
          "mono_failed_limit": MONO_FAIL_MAX,
          "passed": bool(g4_core and mono_failed <= MONO_FAIL_MAX and mono_diff["n_questions"] == MONO_QUESTIONS),
          "rule": "all 120,544 single-colour inner pseudo-questions (masked E_HS): paired PlayerBootstrap 95 % lower "
                  "bound > 0, black and white point estimates > 0; mono failed games <= 306"}
    passed = g1["passed"] and g2["passed"] and g3r["passed"] and g4["passed"]
    return {"compared": "Q02H vs Q02", "G1": g1, "G2": g2, "G3": g3r, "G4": g4, "passed": bool(passed),
            "reason": "all four gates passed" if passed else
            "a gate failed: not submitted, competition 2 unchanged (04 -> R20 -> R10), the scoring slot is kept"}


def bound_hits(w, lo=0.0, hi=4.0, eps=1e-9):
    return {t: ("upper" if abs(v - hi) <= eps else "lower") for t, v in w.items() if abs(v - hi) <= eps or v <= lo + eps}


def apply_decision(fail_count, accept_data_failures, q02_repro_ok, cross, direction, audit, formats_ok):
    """score_allowed of PREREG_07 2.7 -> (value True / False / 'pending_audit', reasons)."""
    reasons = []
    if fail_count > 0:
        if not accept_data_failures:
            reasons.append(f"validation HS: {fail_count} failed games (registered 0): stop and find out why "
                           f"(PREREG_07 2.7; a data property is accepted only with --accept-data-failures)")
        elif fail_count > VAL_FAIL_MAX:
            reasons.append(f"validation HS: {fail_count} failed games > {VAL_FAIL_MAX} (0.5 %): not submitted")
    if q02_repro_ok is not True:
        reasons.append("Q02 does not reproduce results/prereg04/rank/rank_val_probs.csv byte for byte")
    if cross.get("mismatches") != 0:
        reasons.append(f"move cross-check against val_o4.npz: {cross.get('mismatches')} mismatches (registered 0)")
    if not direction.get("passed"):
        reasons.append(f"HS direction check: rho {direction.get('rho')} < {direction.get('threshold')}")
    if formats_ok is not True:
        reasons.append("check_submission did not pass for pred_A.csv and pred_B.csv")
    if audit == "bug":
        reasons.append("prescore_audit.json found a program error affecting Q02H's predictions")
    if reasons:
        return False, reasons
    # apply never writes True: the audit must be of these very outputs, committed after them (apply --finalize)
    return "pending_audit", ["prescore_audit.json of these outputs not committed yet (PREREG_07 2.7); then "
                             "apply --finalize"]


# ================================================================ modes and outputs
class Ctx:
    pass


def resolve(a, cmd):
    c = Ctx()
    c.cmd = cmd
    c.precommit = bool(getattr(a, "precommit", False))
    od = getattr(a, "out_dir", None)
    if c.precommit and od:
        raise P07Error("--precommit writes the real folder; it cannot be combined with --out-dir")
    if od:
        c.out_dir = os.path.abspath(od)
        if not inside(c.out_dir, TMP) or inside(c.out_dir, OUT):
            raise P07Error(f"--out-dir (preview) must be under {rel(TMP)} (got {c.out_dir})")
        c.preview = True
    else:
        c.out_dir = OUT
        c.preview = False
    if under_data_local(c.out_dir):
        raise P07Error("refusing to write under data/local")
    return c


def mode_flags(c):
    return {"preview": c.preview, "precommit": c.precommit}


def guard_outputs(c, a, names):
    """Outputs that exist are refused unless --rerun --reason; the old SHA-256s are recorded."""
    paths = [os.path.join(c.out_dir, n) for n in names]
    have = [p for p in paths if os.path.exists(p)]
    if not have:
        return None
    if not getattr(a, "rerun", False) or not getattr(a, "reason", None):
        raise P07Error(f"outputs exist ({[os.path.basename(p) for p in have]}); rerun only with --rerun --reason TEXT")
    if not c.preview:
        tracked = [p for p in have if git(["ls-files", "--error-unmatch", "--", rel(p)])[0] == 0]
        if tracked:
            raise P07Error(f"committed outputs are never overwritten: {[rel(p) for p in tracked]}")
    return {"reason": a.reason, "old_sha256": {os.path.basename(p): pin_sha(p) for p in have}}


def prior_report(c, name, fmt, precommit_ok):
    """An earlier step's report in this mode (real / precommit / preview) -> (report, sha, commit record)."""
    p = os.path.join(c.out_dir, name)
    if not os.path.exists(p):
        raise P07Error(f"run the earlier step first: {name} missing in {c.out_dir}")
    r = read_json(p)
    if r.get("format") != fmt:
        raise P07Error(f"{name}: not a {fmt} file")
    if bool(r.get("preview")) != c.preview:
        raise P07Error(f"{name} is a {'preview' if r.get('preview') else 'real'} record; this run is "
                       f"{'a preview' if c.preview else 'real'}")
    commit = None if c.preview else require_committed([p], name, precommit=precommit_ok)
    return r, pin_sha(p), commit


def verify_manifest_chunks(man):
    """Every chunk listed in the manifest has its SHA-256 now, and no other chunk file sits in the set folders
    (pilot_fuse.load_set globs every chunk_*.npz) -> list of problems."""
    probs = []
    for s, n_exp in SET_CHUNKS.items():
        rows = ((man.get("sets") or {}).get(s) or {}).get("chunks") or []
        listed = {os.path.normcase(os.path.abspath(path_of(r["file"]))) for r in rows}
        for r in rows:
            p = path_of(r["file"])
            if not os.path.exists(p) or sha_raw(p) != r["sha256"]:
                probs.append(f"{s}: {r['file']} missing or changed since the manifest")
        on_disk = {os.path.normcase(os.path.abspath(f)) for f in
                   glob.glob(os.path.join(PILOT_DATA, s, "chunk_*.npz")) if ".tmp" not in f}
        extra = sorted(on_disk - listed)
        if extra or ((man.get("sets") or {}).get(s) or {}).get("extra_files"):
            probs.append(f"{s}: chunk files not in the manifest: {[rel(x) for x in extra][:5]}")
        if len(rows) != n_exp:
            probs.append(f"{s}: {len(rows)} chunks in the manifest, registered {n_exp}")
    return probs


def manifest_binding(c, precommit):
    if not os.path.exists(MANIFEST):
        raise P07Error(f"{rel(MANIFEST)} missing: run p07_hs.py manifest first")
    commit = None if c.preview else require_committed([MANIFEST], "inner_manifest.json", precommit=precommit)
    man = read_json(MANIFEST)
    if man.get("format") != FORMATS["manifest"]:
        raise P07Error("inner_manifest.json: wrong format")
    return man, sha_lf(MANIFEST), commit


# ================================================================ check (PREREG_07 2.2; spec 4.1)
def cmd_check(a):
    t0 = time.time()
    c = resolve(a, "check")
    rerun = guard_outputs(c, a, ["check_report.json"])
    ts = tool_state()
    if not c.preview:
        require_tools(ts, c.precommit)
    probs = []
    man, man_sha, man_commit = manifest_binding(c, c.precommit)
    if man.get("complete") is not True:
        probs.append("inner_manifest.json: complete is not true (all three sets must be run)")
    man_tool_diff = tools_equal_record(ts, man.get("tool"))
    if man_tool_diff:
        probs.append(f"inner_manifest.json was made by other tool files: {man_tool_diff}")
    # 1 chunks: count, SHA-256, games, ref ranges (the manifest checked refs against the plan)
    chunk_rec = {}
    for s in ("fitmap", "bank", "mono"):
        rows = (man.get("sets") or {}).get(s, {}).get("chunks") or []
        if len(rows) != SET_CHUNKS[s]:
            probs.append(f"{s}: {len(rows)} chunks in the manifest, registered {SET_CHUNKS[s]}")
        bad_sha, games, meta_bad = [], 0, []
        for r in rows:
            p = path_of(r["file"])
            if not os.path.exists(p) or sha_raw(p) != r["sha256"]:
                bad_sha.append(r["file"])
                continue
            games += r["games"]
            with np.load(p, allow_pickle=False) as z:
                meta = {"format": str(z["format"]), "profiles": [str(x) for x in z["profiles"]],
                        "sides": [str(x) for x in z["sides"]], "precision": str(z["precision"]),
                        "batch": int(z["batch"]), "channels_last": bool(z["channels_last"]),
                        "katago_commit": str(z["katago_commit"]), "hs_lib_sha256": str(z["hs_lib_sha256"])}
            want = {"format": PILOT_FORMAT, "profiles": PROFILES, "sides": RUN["sides"], "precision": RUN["precision"],
                    "batch": RUN["batch"], "channels_last": RUN["channels_last"], "katago_commit": KATAGO_COMMIT,
                    "hs_lib_sha256": PINNED["results/prereg07/explore/pilot_humansl/hs_lib.py"]}
            diff = sorted(k for k in want if meta[k] != want[k])
            if diff:
                meta_bad.append((r["file"], diff))
        if bad_sha:
            probs.append(f"{s}: chunks differ from the manifest SHA-256 or are missing: {bad_sha[:5]}")
        if meta_bad:
            probs.append(f"{s}: chunk metadata differ from PREREG_07 1.2-1.4: {meta_bad[:3]}")
        if games != SET_GAMES[s]:
            probs.append(f"{s}: {games} games, registered {SET_GAMES[s]}")
        tot = (man.get("sets") or {}).get(s, {}).get("totals") or {}
        chunk_rec[s] = {"chunks": len(rows), "games": games, "sha_mismatch": bad_sha, "meta_mismatch": meta_bad,
                        "failed": tot.get("failed"), "no_target_moves": tot.get("no_target_moves")}
        if s in ("fitmap", "bank") and (tot.get("failed") != 0 or tot.get("no_target_moves") != 0):
            probs.append(f"{s}: failed {tot.get('failed')} / no target move {tot.get('no_target_moves')} (must be 0)")
    if (man.get("refs_match_plan") or {}).get("all") is not True:
        probs.append("inner_manifest.json: chunk refs do not all match the plan")
    probs.extend(p_ for p_ in verify_manifest_chunks(man) if "not in the manifest" in p_)
    xc = man.get("move_cross_check") or {}
    if xc.get("mismatches") != 0:
        probs.append(f"move cross-check (HS vs inner_o4.npz): {xc.get('mismatches')} mismatches (registered 0)")
    # 3 checkpoint and KataGo sources
    ck = checkpoint_record()
    if not ck["ok"]:
        probs.append(f"checkpoint SHA-256 / size differ from PREREG_07 1.2: {ck}")
    if man.get("checkpoint", {}).get("sha256") != CKPT_SHA:
        probs.append("inner_manifest.json does not record the full checkpoint SHA-256")
    kp, krec = katago_sources()
    probs.extend(kp)
    # plan and its inputs
    plan_sha = sha_raw(PLAN)
    if plan_sha != PLAN_SHA or man.get("plan_sha256") != PLAN_SHA:
        probs.append(f"plan.json SHA-256 {plan_sha} (manifest {man.get('plan_sha256')}) != {PLAN_SHA}")
    plan = read_json(PLAN)
    if plan.get("profiles") != PROFILES:
        probs.append("plan.json profiles differ from PREREG_07 1.4")
    inputs_now = {}
    for name, rp in (("inner_keys.npz", "results/prereg02/rank/dumps/inner_keys.npz"),
                     ("pseudo_inner.json", "results/prereg02/rank/pseudo_inner.json"),
                     ("u1_oof.npz", "results/prereg05/explore/u1/u1_oof.npz"),
                     ("players_split.csv", "results/split_v1/players_split.csv")):
        inputs_now[name] = sha_raw(path_of(rp))
        if inputs_now[name] != plan["inputs_sha256"][name]:
            probs.append(f"plan input {name} changed: {inputs_now[name]} != {plan['inputs_sha256'][name]}")
    # 6 u1_lib.load_inputs (frozen p02 / p03 / p04 tools and records)
    PF = import_pilot_fuse()
    sets = {s: PF.load_set(s, len(PROFILES), PILOT_DATA) for s in ("fitmap", "bank", "mono")}
    for s, st in sets.items():
        if st is None or len(st["meta"]) != 1:
            probs.append(f"{s}: pilot_fuse.load_set meta must have exactly one value (got {None if st is None else st['meta']})")
        if st is not None and (st["counts"][:, 1] != 0).any():
            probs.append(f"{s}: opponent-side counts present (sides must be target only)")
    U = _import_u1()
    try:
        d = U.load_inputs(verbose=False)
        u1_ok = True
        u1_rec = {"inner_games": d["inner"]["keys"]["n"], "fit": len(d["inner"]["X_fit"]),
                  "eval": len(d["inner"]["X_eval"]), "single_colour": len(d["inner"]["X_mono"]),
                  "mono_sha256": d["inner"]["ms"]["sha256"]}
        if d["inner"]["ms"]["sha256"] != MONO_SHA:
            probs.append("single-colour pseudo-questions SHA-256 differs")
    except Exception as e:  # noqa: BLE001 - recorded as a stop
        u1_ok, u1_rec = False, {"error": f"{type(e).__name__}: {e}"}
        probs.append(f"u1_lib.load_inputs failed: {e}")
    pyc = no_pycache()
    if pyc:
        probs.append(f"__pycache__ present: {pyc}")
    rep = {"format": FORMATS["check"], "what": "PREREG_07 2.2 check: are the inputs usable (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c), "rerun": rerun,
           "status": "ok" if not probs else "stop", "problems": probs,
           "inner_manifest": {"sha256": man_sha, "commit": man_commit}, "chunks": chunk_rec,
           "move_cross_check": xc, "checkpoint": ck, "katago": krec,
           "plan": {"sha256": plan_sha, "inputs_sha256_now": inputs_now},
           "u1_load_inputs": {"ok": u1_ok, **u1_rec}, "pinned_sha256": check_pins(), "tool": ts,
           "environment": environment_cpu(), "no_pycache_after": not no_pycache()}
    write_json(os.path.join(c.out_dir, "check_report.json"), rep)
    print(f"check: {rep['status']}" + ("" if not probs else f" ({'; '.join(probs)})"))
    if probs:
        raise Stop("check says stop")
    return 0


# ================================================================ inner setup shared by selfcheck and tune
def _inner_setup(PF, U):
    """Pilot sets, fitmap labels and the u1 inputs (inner files only)."""
    plan = read_json(PLAN)
    sets = {s: PF.load_set(s, len(PROFILES), PILOT_DATA) for s in ("fitmap", "bank", "mono")}
    fm = sets["fitmap"]
    lv_of = {lv: i for i, lv in enumerate(U.LEVELS)}
    yfit = np.array([lv_of[plan["sets"]["fitmap"]["games"][j]["rank"]] for j in fm["ref"]], dtype=np.int64)
    d = U.load_inputs(verbose=False)
    return plan, sets, yfit, d


def _fit_map(PF, sets, yfit):
    x, ok = PF.curves(sets["fitmap"], 0)
    return PF.fit_mapping(x[ok], yfit[ok], seed=0), int(ok.sum())


def inner_lp(PF, sets, logp, n_inner):
    """Pilot logic: lp_HS of every ok bank / mono game (0 elsewhere) and the covered mask."""
    covered = np.zeros(n_inner, dtype=bool)
    lp = np.zeros((n_inner, L))
    for name in ("bank", "mono"):
        st = sets[name]
        x, ok = PF.curves(st, 0)
        lp[st["ref"][ok]] = logp(x[ok])
        covered[st["ref"][ok]] = True
    return lp, covered


def inner_failed(sets, n_inner):
    """failed / no-target-move flags of the inner games from the chunks (games not in any chunk: failed=False,
    counts=0 -> not usable)."""
    failed = np.zeros(n_inner, dtype=bool)
    counts = np.zeros(n_inner, dtype=np.int64)
    for name in ("bank", "mono"):
        st = sets[name]
        failed[st["ref"]] = st["failed"]
        counts[st["ref"]] = st["counts"][:, 0]
    return failed, counts


def run_terms(d, terms, E_hs):
    """u1_lib.run_gen with the HS term taken from E_hs = {'fit', 'eval', 'mono'} (masked E_HS)."""
    inner, lp = d["inner"], d["lp"]
    Xf, Xe, Sf, Se, Xm, Sm = (inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"], inner["X_mono"],
                              inner["S_mono"])
    lp5 = lp["C5"]
    mf, me, mm = rf.pooled(Sf, lp5, "log"), rf.pooled(Se, lp5, "log"), rf.pooled(Sm, lp5, "log")

    def qt(X, S, which):
        return [E_hs[which] if t == "HS" else (X.mop if t == "OP" else rf.pooled(S, lp[t], "log")) for t in terms]

    Ef, Ee, Em = qt(Xf, Sf, "fit"), qt(Xe, Se, "eval"), qt(Xm, Sm, "mono")
    P, folds = p4.gen_oof_record(Xf, Xe, mf, me, Ef, Ee)
    if not np.array_equal(p4.gen_fold_predict(folds, me, Ee, Xe.n, Xe.fold), P):
        raise P07Error("fold records do not reproduce the OOF probabilities")
    Pm = p4.gen_fold_predict(folds, mm, Em, Xm.n, Xm.fold)
    return P, Pm, folds, {"mf": mf, "Ef": Ef}


def masked_inner_E(d, lp_hs, usable, force=False):
    inner = d["inner"]
    out, info = {}, {}
    for which, S in (("fit", inner["S_fit"]), ("eval", inner["S_eval"]), ("mono", inner["S_mono"])):
        out[which], info[which] = masked_E(S, lp_hs, usable, force_mask=force)
    return out, info


def leaves_close(a, b, path="", tol=TOL):
    """Numeric leaves of two JSON-like objects within tol -> list of differences (structure must match)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                out.extend(leaves_close(a[k], b[k], f"{path}/{k}", tol))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: lengths {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(leaves_close(x, y, f"{path}[{i}]", tol))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(float(a) - float(b)) > tol:
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


# ================================================================ selfcheck (PREREG_07 2.3; spec 4.2)
def cmd_selfcheck(a):
    t0 = time.time()
    c = resolve(a, "selfcheck")
    rerun = guard_outputs(c, a, ["selfcheck.json"])
    ts = tool_state()
    if not c.preview:
        require_tools(ts, c.precommit)
    cr, cr_sha, cr_commit = prior_report(c, "check_report.json", FORMATS["check"], c.precommit)
    if cr.get("status") != "ok":
        raise P07Error("check_report.json says stop; selfcheck does not run")
    if tools_equal_record(ts, cr.get("tool")):
        raise P07Error(f"tools changed since check: {tools_equal_record(ts, cr.get('tool'))}")
    man, man_sha, _mc = manifest_binding(c, c.precommit)
    if man_sha != (cr.get("inner_manifest") or {}).get("sha256"):
        raise P07Error("inner_manifest.json differs from the one check used")
    chunk_probs = verify_manifest_chunks(man)
    if chunk_probs:
        raise P07Error(f"pilot chunks differ from inner_manifest.json: {chunk_probs[:5]}")
    PF = import_pilot_fuse()
    U = _import_u1()
    plan, sets, yfit, d = _inner_setup(PF, U)
    probs = []
    # mapping
    mp, n_fit = _fit_map(PF, sets, yfit)
    pr = read_json(PILOT_REPORT)
    cv_ref = pr["mapping_side0"]["cv"]
    map_rec = {"lambda": mp["lambda"], "cv": mp["cv"], "refit_loss": mp["model"]["loss"], "games": n_fit,
               "success": mp["model"]["success"]}
    if mp["lambda"] != PILOT["lambda"]:
        probs.append(f"lambda {mp['lambda']} != {PILOT['lambda']}")
    dcv = leaves_close({k: v["logloss"] for k, v in mp["cv"].items()}, {k: v["logloss"] for k, v in cv_ref.items()})
    if dcv:
        probs.append(f"CV log-loss differs from the pilot: {dcv}")
    if abs(mp["cv"][str(mp["lambda"])]["logloss"] - PILOT["cv_logloss"]) > TOL:
        probs.append("CV log-loss of the chosen lambda differs from 1.7571755879185869")
    if abs(mp["model"]["loss"] - PILOT["refit_loss"]) > TOL:
        probs.append(f"refit loss {mp['model']['loss']!r} != {PILOT['refit_loss']!r}")
    # inner lp (pilot logic) and Q02 / Q02H out of fold through u1_lib.run_gen
    inner = d["inner"]
    n_inner = inner["keys"]["n"]
    lp_hs, covered = inner_lp(PF, sets, mp["logp"], n_inner)
    d["lp"]["HS"] = lp_hs
    Xe, Xm, ms = inner["X_eval"], inner["X_mono"], inner["ms"]
    if len(Xe) != PILOT["eval_questions"]:
        probs.append(f"{len(Xe)} inner eval questions, pilot {PILOT['eval_questions']}")
    eval_cov = np.array([covered[np.asarray(g)].all() for g in inner["eval_groups"]])
    if not eval_cov.all():
        probs.append(f"{int((~eval_cov).sum())} eval questions with an unscored game")
    mono_cov = np.array([covered[np.asarray(g)].all() for g in ms["groups"]])
    saved = U.load_saved()
    P_q, Pm_q, _th_q = U.run_gen(d, Q02_TERMS)
    P_h, Pm_h, th_h = U.run_gen(d, Q02H_TERMS)
    seam = {"eval_P_equal_u1": bool(np.array_equal(P_q, saved["eval_P_Q02"])),
            "mono_P_equal_u1": bool(np.array_equal(Pm_q, saved["mono_P_Q02"]))}
    if not all(seam.values()):
        probs.append(f"Q02 does not reproduce u1_oof.npz: {seam}")
    sc_q, sc_h = U.score(U.rule_b_np(P_q), Xe.y), U.score(U.rule_b_np(P_h), Xe.y)
    rb_q, rb_h = U.rank_balanced(sc_q, Xe.y), U.rank_balanced(sc_h, Xe.y)
    dd = inner["boot"].stats(sc_h) - inner["boot"].stats(sc_q)
    ci = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
    qmd = float(np.mean(sc_h - sc_q))
    got = {"Q02": rb_q, "Q02H": rb_h, "diff": rb_h - rb_q, "ci95": ci, "question_mean_diff": qmd}
    dq = leaves_close(got, {k: PILOT[k] for k in got})
    if dq:
        probs.append(f"Q02 / Q02H inner numbers differ from the pilot: {dq}")
    # mono, pilot logic on the covered questions, vs the committed full-coverage pilot_fuse_report.json
    sub = {"y": ms["y"][mono_cov], "player": ms["player"][mono_cov], "colour": ms["colour"][mono_cov],
           "k": ms["k"][mono_cov], "n_players": ms["n_players"]}
    sc_mq, sc_mh = U.score(U.rule_b_np(Pm_q), Xm.y), U.score(U.rule_b_np(Pm_h), Xm.y)
    sboot = U.PlayerBootstrap(sub["player"], sub["y"], sub["n_players"])
    mono_got = U.p3.mono_diff_record(sub, sboot, sc_mh[mono_cov], sc_mq[mono_cov])
    mono_ref = pr["combinations"]["Q02+HS"]["mono_diff_vs_Q02"]
    coverage = f"{int(mono_cov.sum())}/{len(mono_cov)} questions"
    mono_failed = int(sets["mono"]["failed"].sum())
    if pr["gate"]["mono"]["coverage"] != f"{MONO_QUESTIONS}/{MONO_QUESTIONS} questions" and mono_failed == 0:
        probs.append(f"pilot_fuse_report.json coverage {pr['gate']['mono']['coverage']}: not the full-coverage run")
    if coverage != pr["gate"]["mono"]["coverage"]:
        probs.append(f"mono coverage {coverage} != the report's {pr['gate']['mono']['coverage']}")
    dm = leaves_close({k: mono_got[k] for k in ("rank_balanced", "ci95", "by_colour", "by_k")},
                      {k: mono_ref[k] for k in ("rank_balanced", "ci95", "by_colour", "by_k")})
    if dm:
        probs.append(f"mono numbers differ from pilot_fuse_report.json: {dm}")
    # masked E_HS: on inner (no unusable game in any question) it must be pooled bit for bit
    failed, counts = inner_failed(sets, n_inner)
    usable = usable_games(np.ones(n_inner, dtype=bool), failed, counts)
    E_m, E_info = masked_inner_E(d, lp_hs, usable)
    E_f, E_finfo = masked_inner_E(d, lp_hs, usable, force=True)
    pooled_ref = {"fit": rf.pooled(inner["S_fit"], lp_hs, "log"), "eval": rf.pooled(inner["S_eval"], lp_hs, "log"),
                  "mono": rf.pooled(inner["S_mono"], lp_hs, "log")}
    masked_equal = {k: bool(np.array_equal(E_m[k], pooled_ref[k])) for k in E_m}
    forced_maxdiff = {k: float(np.max(np.abs(E_f[k] - pooled_ref[k]))) for k in E_f}
    P_hm, Pm_hm, _ = run_terms(d, Q02H_TERMS, E_m)[:3]
    masked_P_equal = {"eval": bool(np.array_equal(P_hm, P_h)), "mono": bool(np.array_equal(Pm_hm, Pm_h))}
    if not all(masked_equal.values()) or not all(masked_P_equal.values()):
        probs.append(f"masked E_HS differs from the pooled version on inner: {masked_equal} {masked_P_equal}")
    rep = {"format": FORMATS["selfcheck"],
           "what": "PREREG_07 2.3 selfcheck: the main-branch tool reproduces the pilot (a check of the port)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c), "rerun": rerun,
           "passed": not probs, "problems": probs,
           "check_report": {"sha256": cr_sha, "commit": cr_commit},
           "mapping": map_rec, "pilot_reference": PILOT,
           "q02_seam_vs_u1_oof": seam, "inner_eval": got,
           "folds_Q02H": [list(map(float, t)) for t in th_h],
           "mono_pilot_logic": {"coverage": coverage, "difference": mono_got,
                                "reference": "results/prereg07/explore/pilot_humansl/pilot_fuse_report.json "
                                             "combinations.Q02+HS.mono_diff_vs_Q02",
                                "reference_sha256": sha_lf(PILOT_REPORT), "mono_failed_games": mono_failed},
           "masked_E_HS": {"equal_to_pooled": masked_equal, "Q02H_P_equal": masked_P_equal, "info": E_info,
                           "forced_mask_max_abs_diff_record_only": forced_maxdiff},
           "inner_coverage": {"inner_games_scored": int(covered.sum()), "unusable_in_questions":
                              {k: v["unusable_slots_in_questions"] for k, v in E_info.items()}},
           "pinned_sha256": check_pins(), "tool": ts, "environment": environment_cpu()}
    write_json(os.path.join(c.out_dir, "selfcheck.json"), rep)
    print(f"selfcheck: {'passed' if not probs else 'FAILED'}; Q02 {rb_q!r}, Q02H {rb_h!r}, diff {rb_h - rb_q!r}, "
          f"CI {ci}; mono {mono_got['rank_balanced']!r} ({coverage})")
    if probs:
        for p in probs:
            print("  -", p)
        raise Stop("selfcheck failed")
    return 0


# ================================================================ tune (PREREG_07 2.4-2.6; spec 4.3)
def revision_committed(n):
    """PREREG_07 rev n row present in the committed registration -> (commit, time) or raises."""
    rc, out = git(["log", "--format=%H %cI", "-S", f"| rev {n} |", "--", rel(PREREG)])
    lines = [x for x in out.splitlines() if x.strip()] if rc == 0 else []
    if not lines:
        raise P07Error(f"PREREG_07 rev {n} is not in any commit")
    h, t = lines[-1].split()
    return h, t


def cmd_tune(a):
    t0 = time.time()
    c = resolve(a, "tune")
    frozen_path = os.path.join(c.out_dir, "frozen_config.json")
    map_path = os.path.join(c.out_dir, "hs_map.npz")
    if not c.preview and git(["ls-files", "--error-unmatch", "--", rel(frozen_path)])[0] == 0:
        raise P07Error("frozen_config.json is committed (frozen); tune never overwrites it")
    rerun = guard_outputs(c, a, ["tune_report.json", "frozen_config.json", "hs_map.npz"])
    if rerun is not None and not c.preview:
        if not getattr(a, "revision", None) or a.revision < 2:
            raise P07Error("tune is judged once: --rerun needs --revision N >= 2 (a committed PREREG_07 revision "
                           "that names a program error)")
        h, t = revision_committed(a.revision)
        old = os.path.join(c.out_dir, "tune_report.json")
        old_created = read_json(old).get("created") if os.path.exists(old) else None
        if old_created and datetime.datetime.fromisoformat(t) <= datetime.datetime.fromisoformat(old_created):
            raise P07Error(f"PREREG_07 rev {a.revision} ({t}) is not later than the tune it replaces ({old_created})")
        rerun["revision"] = {"rev": a.revision, "commit": h, "commit_time": t, "replaced_tune_created": old_created}
    ts = tool_state()
    if not c.preview:
        require_tools(ts, c.precommit)
    cr, cr_sha, cr_commit = prior_report(c, "check_report.json", FORMATS["check"], c.precommit)
    sr, sr_sha, sr_commit = prior_report(c, "selfcheck.json", FORMATS["selfcheck"], c.precommit)
    if cr.get("status") != "ok" or sr.get("passed") is not True:
        raise P07Error("check_report.json / selfcheck.json did not pass; tune does not run")
    for what, r in (("check_report.json", cr), ("selfcheck.json", sr)):
        diff = tools_equal_record(ts, r.get("tool"))
        if diff:
            raise P07Error(f"{what} was made by other tool files ({diff}); rerun it with the current tools")
    man, man_sha, man_commit = manifest_binding(c, c.precommit)
    if man_sha != (cr.get("inner_manifest") or {}).get("sha256"):
        raise P07Error("inner_manifest.json differs from the one check used")
    if (sr.get("check_report") or {}).get("sha256") != cr_sha:
        raise P07Error("selfcheck.json was made from another check_report.json")
    chunk_probs = verify_manifest_chunks(man)
    if chunk_probs:
        raise P07Error(f"pilot chunks differ from inner_manifest.json: {chunk_probs[:5]}")
    PF = import_pilot_fuse()
    U = _import_u1()
    plan, sets, yfit, d = _inner_setup(PF, U)
    inner = d["inner"]
    n_inner = inner["keys"]["n"]
    Xf, Xe, Xm, ms = inner["X_fit"], inner["X_eval"], inner["X_mono"], inner["ms"]
    boot, mboot = inner["boot"], inner["mono_boot"]
    # 2 the frozen per-game map (fitmap only), round trip through hs_map.npz
    mp, n_fit = _fit_map(PF, sets, yfit)
    lp_mem, covered = inner_lp(PF, sets, mp["logp"], n_inner)
    os.makedirs(TMP, exist_ok=True)
    tmp_map = os.path.join(TMP, f"hs_map_{os.getpid()}.npz")
    np.savez(tmp_map, format=np.array(FORMATS["hs_map"]), mu=mp["mu"], sd=mp["sd"], W=mp["model"]["W"],
             b=mp["model"]["b"], **{"lambda": np.array(mp["lambda"])}, profiles=np.array(PROFILES),
             cv=np.array(json.dumps(mp["cv"], sort_keys=True)), fitmap_games=np.array(n_fit),
             refit_loss=np.array(mp["model"]["loss"]))
    model = load_hs_map(tmp_map)
    failed, counts = inner_failed(sets, n_inner)
    # reread map, per set (the pilot's call pattern: one mlr_logp call per set; BLAS results depend on the batch)
    lp_pf = np.zeros_like(lp_mem)
    lp_p07 = np.zeros_like(lp_mem)
    ok_file = np.zeros(n_inner, dtype=bool)
    for name in ("bank", "mono"):
        st = sets[name]
        x, okx = PF.curves(st, 0)
        lp_pf[st["ref"][okx]] = PF.mlr_logp({"W": model["W"], "b": model["b"]}, (x[okx] - model["mu"]) / model["sd"])
        lpx, okh = hs_lp(model, st["sums"][:, 0, :], st["counts"][:, 0], st["failed"])
        lp_p07[st["ref"][okh]] = lpx[okh]
        ok_file[st["ref"][okh]] = True
    sums = np.zeros((n_inner, len(PROFILES)))
    for name in ("bank", "mono"):
        sums[sets[name]["ref"]] = sets[name]["sums"][:, 0, :]
    lp_one, _ok1 = hs_lp(model, sums, counts, failed)
    roundtrip = {"pilot_fuse_mlr_logp_equal": bool(np.array_equal(lp_pf, lp_mem)),
                 "p07_hs_lp_per_set_equal": bool(np.array_equal(lp_p07, lp_mem)),
                 "one_batch_max_abs_diff_record_only": float(np.max(np.abs(lp_one - lp_mem))),
                 "ok_equal_covered": bool(np.array_equal(ok_file, covered)),
                 "note": "per set = the pilot's call pattern; one batch over all inner rows differs by BLAS blocking "
                         "only (recorded)"}
    if not (roundtrip["pilot_fuse_mlr_logp_equal"] and roundtrip["p07_hs_lp_per_set_equal"]
            and roundtrip["ok_equal_covered"]):
        os.remove(tmp_map)
        raise P07Error(f"hs_map.npz round trip differs: {roundtrip}")
    lp_hs = lp_mem
    d["lp"]["HS"] = lp_hs
    # 3 masked E_HS
    usable = usable_games(np.ones(n_inner, dtype=bool), failed, counts)
    E_hs, E_info = masked_inner_E(d, lp_hs, usable)
    # PREREG_07 2.5 / spec 4.5: failed = failed or no move at all (the manifest counts both from the chunks)
    mono_failed = int(((man.get("sets") or {}).get("mono", {}).get("totals") or {}).get("failed_or_no_move", -1))
    if mono_failed < int(sets["mono"]["failed"].sum()):
        raise P07Error("inner_manifest.json mono failed_or_no_move is missing or smaller than the chunks' failed")
    mono_nomove = int(((sets["mono"]["counts"][:, 0] == 0) & ~sets["mono"]["failed"]).sum())
    # 4 OOF
    res = {}
    for cid, terms in (("Q02", Q02_TERMS), ("Q02H", Q02H_TERMS), ("R20+HS", R20HS_TERMS)):
        t = time.time()
        P, Pm, folds, aux = run_terms(d, terms, E_hs)
        if np.isnan(Pm).any():
            raise P07Error(f"{cid}: a single-colour question without an OOF fold")
        m, sc = rf.metrics(rf.rule_b_np(P), Xe.y, Xe.n, Xe.player)
        mrec, msc = p3.mono_score_record(rf.rule_b_np(Pm), Xm, ms, mboot)
        for fr in folds:
            fr["params"] = p4.gen_params(fr["theta"], terms)
        res[cid] = {"P": P, "Pm": Pm, "folds": folds, "terms": terms, "m": m, "sc": sc, "mrec": mrec, "msc": msc,
                    "aux": aux}
        print(f"{cid}: inner eval rank-balanced {m['rank_balanced']!r}; single-colour {mrec['rank_balanced']:.4f} "
              f"({time.time() - t:.0f}s)", flush=True)
    q, h = res["Q02"], res["Q02H"]
    if abs(q["m"]["rank_balanced"] - PILOT["Q02"]) > TOL or abs(h["m"]["rank_balanced"] - PILOT["Q02H"]) > TOL:
        raise P07Error("Q02 / Q02H differ from the selfcheck numbers")
    mono_diff = p3.mono_diff_record(ms, mboot, h["msc"], q["msc"])
    gates = gate_record(h["m"]["rank_balanced"], q["m"]["rank_balanced"], h["sc"], q["sc"], boot, mono_diff,
                        mono_failed)
    # 5 record only
    rows_seen = set(int(x) for x in sets["bank"]["ref"])
    for i in (0, 1):
        with np.load(os.path.join(PILOT_DATA, "mono", f"chunk_{i:05d}.npz"), allow_pickle=False) as z:
            rows_seen |= set(int(x) for x in z["ref"])
    new_mask = np.array([any(int(r) not in rows_seen for r in g) for g in ms["groups"]])
    new_rec = None
    new_ok = int(new_mask.sum()) == NEW_SUBSET_QUESTIONS
    if new_mask.any():
        sub = {"y": ms["y"][new_mask], "player": ms["player"][new_mask], "colour": ms["colour"][new_mask],
               "k": ms["k"][new_mask], "n_players": ms["n_players"]}
        new_rec = p3.mono_diff_record(sub, rf.PlayerBootstrap(sub["player"], sub["y"], sub["n_players"]),
                                      h["msc"][new_mask], q["msc"][new_mask])
    new_rec = {"questions": int(new_mask.sum()), "registered_questions": NEW_SUBSET_QUESTIONS,
               "count_matches": new_ok, "difference": new_rec,
               "definition": "single-colour pseudo-questions with at least one game outside the bank and mono "
                             "chunks 0 and 1 (not covered by the second interim look); recorded only"}
    rows = {}
    for cid, r in res.items():
        m = dict(r["m"])
        m.update(boot.summary(r["sc"], None if cid == "Q02" else q["sc"]))
        for k in ("diff_vs_R00_se", "diff_vs_R00_ci95"):
            if k in m:
                m[k.replace("R00", "Q02")] = m.pop(k)
        if cid != "Q02":
            m["diff_vs_Q02"] = m["rank_balanced"] - q["m"]["rank_balanced"]
            m["question_mean_diff_vs_Q02"] = float(np.mean(r["sc"] - q["sc"]))
            m["per_level_diff_vs_Q02"] = {lv: m["per_level"][lv] - q["m"]["per_level"][lv] for lv in m["per_level"]}
            m["single_colour_diff_vs_Q02"] = p3.mono_diff_record(ms, mboot, r["msc"], q["msc"])
        rows[cid] = {"terms": r["terms"], "oof": m, "single_colour": p4.short_mono(r["mrec"]),
                     "folds": [{"fold": fr["fold"], "loss": fr["loss"], "params": fr["params"],
                                "bounds_hit": bound_hits(fr["params"]["w"])} for fr in r["folds"]]}
    se = p3.se_val_record(h["m"]["rank_balanced"] - q["m"]["rank_balanced"], h["sc"] - q["sc"], Xe.y)
    se["adoption_probability_at_fraction_of_delta"] = p4.adoption(se["delta"], se["se_val"])
    # HS direction references (spec 4.6): HS-only expected level vs Q02's OOF expected level
    has_e = questions_with_usable(inner["S_eval"], usable)
    has_m = questions_with_usable(inner["S_mono"], usable)
    rho_eval, n_e = hs_rho(E_hs["eval"], q["P"], has_e)
    rho_mono, n_m = hs_rho(E_hs["mono"], q["Pm"], has_m)
    direction_ref = {"rho_eval": rho_eval, "eval_questions": n_e, "rho_mono": rho_mono, "mono_questions": n_m,
                     "floor_validation": RHO_FLOOR_VAL, "floor_test_per_set": RHO_FLOOR_TEST,
                     "validation_threshold": max(0.5 * rho_eval, RHO_FLOOR_VAL),
                     "test_threshold": max(0.5 * rho_mono, RHO_FLOOR_TEST),
                     "definition": "Spearman (average ranks) between sum_k k softmax(E_HS)_k and sum_k k P_Q02,k "
                                   "(Q02 OOF), questions with >= 1 usable game"}
    # 6 refit on all inner fit questions (gates passed)
    theta, refit = p4.gen_fit_record(h["aux"]["mf"], h["aux"]["Ef"], Xf.n, Xf.y)
    params = p4.gen_params(theta, Q02H_TERMS)
    Pfit = fuse_probs(theta, h["aux"]["mf"], h["aux"]["Ef"], Xf.n, np.ones(len(Xf), dtype=bool))
    mfit, _ = rf.metrics(rf.rule_b_np(Pfit), Xf.y, Xf.n, Xf.player)
    refit.update(fit_questions=len(Xf), params=params, bounds_hit=bound_hits(params["w"]),
                 in_sample_fit_rank_balanced=mfit["rank_balanced"], in_sample_fit_question_mean=mfit["question_mean"],
                 oof_eval_rank_balanced=h["m"]["rank_balanced"],
                 note="in-sample = the refit applied to the inner fit pseudo-questions it was fitted on; OOF = the "
                      "5-fold out-of-fold score on the inner eval pseudo-questions (G1's number)")
    blocked = [] if gates["passed"] else [gates["reason"]]
    tune_tool_sha = ts["files_sha256"]
    inputs = {"inner_manifest.json": man_sha, "check_report.json": cr_sha, "selfcheck.json": sr_sha,
              "plan.json": PLAN_SHA, "pilot_fuse_report.json": sha_lf(PILOT_REPORT),
              "inner_o4.npz": sha_raw(INNER_O4), **{k: v for k, v in d["inner"]["inputs"].items()},
              **d["sha"]}
    chunk_torch = sorted({str(v) for v in (man.get("torch_values") or [])})
    bench_env = read_json(GPU_BENCH)["environment"]
    nv = nvidia_query()
    environment = {"venv": environment_cpu(),
                   "venv_gpu_reference": {"python": bench_env["python"],
                                          "torch": chunk_torch[0] if len(chunk_torch) == 1 else chunk_torch,
                                          "cudnn": bench_env["cudnn"], "gpu": bench_env["gpu"],
                                          "numpy": bench_env["numpy"],
                                          "sources": "torch: the pilot chunks' torch field; python / cuDNN / GPU / "
                                                     "numpy: hs_gpu_bench.json environment (committed)"},
                   "nvidia_driver_record_only": nv}
    report = {"format": FORMATS["tune"],
              "what": "PREREG_07 2.4-2.6 tune: Q02 vs Q02H on the inner pseudo-questions, 5-fold OOF, gates G1-G4",
              "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c), "rerun": rerun,
              "combinations": rows, "gates": gates, "score_allowed_at_tune": not blocked,
              "score_blocked_reasons": blocked, "score_pending_val_hs": bool(gates["passed"]),
              "record_only": {"R20+HS": rows["R20+HS"]["oof"]["rank_balanced"], "new_subset_G4": new_rec,
                              "single_colour_Q02H_minus_Q02": mono_diff,
                              "mono_no_target_move_games": mono_nomove},
              "se_val": se, "hs_direction_ref": direction_ref, "refit": refit,
              "hs_map": {"lambda": mp["lambda"], "cv": mp["cv"], "refit_loss": mp["model"]["loss"],
                         "fitmap_games": n_fit, "roundtrip": roundtrip},
              "masked_E_HS": E_info, "mono_failed_games": mono_failed,
              "questions": {"fit": len(Xf), "eval": len(Xe), "single_colour": len(Xm)},
              "inputs_sha256": inputs, "check": {"sha256": cr_sha, "commit": cr_commit},
              "selfcheck": {"sha256": sr_sha, "commit": sr_commit}, "inner_manifest_commit": man_commit,
              "pinned_sha256": check_pins(), "tool": ts, "environment": environment}
    os.makedirs(c.out_dir, exist_ok=True)
    if new_ok is not True:
        report["score_pending_val_hs"] = False
        report["score_blocked_reasons"] = blocked + [f"new-subset question count {int(new_mask.sum())} != "
                                                     f"{NEW_SUBSET_QUESTIONS}: stop and find out why (spec 4.3 item 5)"]
        os.remove(tmp_map)
        write_json(os.path.join(c.out_dir, "tune_report.json"), report)
        raise Stop("new-subset count differs")
    if not gates["passed"]:
        os.remove(tmp_map)
        write_json(os.path.join(c.out_dir, "tune_report.json"), report)
        print(f"gates NOT passed: {gates['reason']}; no frozen_config.json, no hs_map.npz (PREREG_07 2.6)")
        raise Stop("gates not passed")
    os.replace(tmp_map, map_path)
    model2 = load_hs_map(map_path)
    if not all(np.array_equal(model2[k], model[k]) for k in ("mu", "sd", "W", "b")):
        raise P07Error("hs_map.npz changed while moving it")
    map_sha = sha_raw(map_path)
    p04fc = read_json(P04_FROZEN)
    frozen = {
        "format": FORMATS["frozen"],
        "what": "PREREG_07 competition 2 frozen configuration Q02H = Q02 + HS (inner only; commit with hs_map.npz "
                "before p07_hs.py val and apply)",
        "created": report["created"], **mode_flags(c),
        "combo": {"id": "Q02H", "sources": ["C5", "OP", "O4", "SWA", "SWB", "HS"], "terms": Q02H_TERMS,
                  "path": "gen", "pool": "log", "calibrated": True, "opening": True},
        "params": params,
        "folds": [{"fold": fr["fold"], "loss": fr["loss"], "starts": fr["starts"], "params": fr["params"],
                   "bounds_hit": bound_hits(fr["params"]["w"])} for fr in h["folds"]],
        "basis": [[float(x) for x in row] for row in p4.B],
        "formula": "p = softmax(e^a (n/12)^gamma (m_C5 + w_OP m_OP + w_O4 E_O4 + w_SWA E_SWA + w_SWB E_SWB + w_HS E_HS "
                   "+ B beta)); E_HS = mean over the question's usable games (C5 parsed, HS not failed, >= 1 target "
                   "move) of lp_HS, 0 without one; n = C5-parsed games",
        "decision_rule": "B, tools/decide.py (argmax p_k + e^-1 (p_{k-1} + p_{k+1}))",
        "hs": {"profiles": PROFILES, "metadata": METADATA_TEXT, "rules": RULES_TEXT,
               "side": "target only (the question's colour)", **{k: RUN[k] for k in RUN},
               "checkpoint": {"path": rel(CKPT), "bytes": CKPT_BYTES, "sha256": CKPT_SHA,
                              "url": "https://github.com/lightvector/KataGo/releases/download/v1.15.0/"
                                     "b18c384nbt-humanv0.ckpt", "weights": "model (not swa_model)"},
               "katago_commit": KATAGO_COMMIT, "source_txt_sha256": SOURCE_TXT_SHA, "pinned_sha256": dict(PINNED),
               "hs_map_npz_sha256": map_sha, "inner_manifest_sha256": man_sha, "plan_sha256": PLAN_SHA,
               "map": {"lambda": mp["lambda"], "fitmap_games": n_fit, "refit_loss": mp["model"]["loss"],
                       "standardise": "fitmap mean / sd (ddof 0; sd 0 -> 1)"}},
        "prereg04": {"frozen_config_sha256": sha_raw(P04_FROZEN), "source_sha256": p04fc["source_sha256"],
                     "opening_build_id": p04fc.get("opening_build_id"),
                     "rank_val_probs_sha256": sha_raw(P04_PROBS), "val_o4_sha256": sha_raw(VAL_O4)},
        "gates": gates, "hs_direction_ref": direction_ref, "se_val": se, "score_pending_val_hs": True,
        "score_allowed_note": "gates passed; score_allowed is decided by apply (PREREG_07 2.7) after the validation HS",
        "oof": {k: h["m"][k] for k in ("rank_balanced", "question_mean", "per_level", "per_n_bin",
                                       "pred_distribution")},
        "Q02_oof_rank_balanced": q["m"]["rank_balanced"],
        "single_colour": {"questions_sha256": ms["sha256"], "Q02H": p4.short_mono(h["mrec"]),
                          "Q02": p4.short_mono(q["mrec"]), "Q02H_minus_Q02": mono_diff},
        "refit": {k: refit[k] for k in ("loss", "starts", "fit_questions", "bounds_hit", "in_sample_fit_rank_balanced",
                                        "oof_eval_rank_balanced")},
        "check": {"check_report_sha256": cr_sha, "selfcheck_sha256": sr_sha},
        "inputs_sha256": inputs, "tool": ts, "environment": environment,
        "tool_files_at_tune": tune_tool_sha}
    write_json(os.path.join(c.out_dir, "tune_report.json"), report)
    write_json(frozen_path, frozen)
    g = gates
    print(f"Q02H vs Q02: G1 {g['G1']['delta']!r} G2 {g['G2']['ci95']} G3 {g['G3']['question_mean_difference']!r} "
          f"G4 {g['G4']['rank_balanced_difference']!r} {g['G4']['ci95']} B/W {g['G4']['by_colour_point_estimates']} "
          f"-> {'PASSED' if g['passed'] else 'not passed'}")
    print(f"rho_eval {rho_eval:.4f}, rho_mono {rho_mono:.4f}; new subset {new_rec['questions']} questions")
    print(f"wrote tune_report.json, frozen_config.json, hs_map.npz in {c.out_dir}"
          f"{' (PREVIEW)' if c.preview else '; commit frozen_config.json + hs_map.npz before p07_hs.py val'}")
    return 0


# ================================================================ apply (PREREG_07 2.7; spec 4.4)
def design_records_bound(fc):
    """val / apply: inner_manifest.json, check_report.json, selfcheck.json and tune_report.json committed and clean,
    and the manifest / check / selfcheck equal to the SHA-256s frozen_config.json records (closes the --precommit
    path: PREREG_07 2.2 item 1 and the G4 boundary need the committed manifest). -> record."""
    paths = [MANIFEST] + [os.path.join(OUT, n) for n in ("check_report.json", "selfcheck.json", "tune_report.json")]
    rec = require_committed(paths, "inner_manifest.json / check_report.json / selfcheck.json / tune_report.json")
    want = {MANIFEST: fc["hs"]["inner_manifest_sha256"],
            os.path.join(OUT, "check_report.json"): fc["check"]["check_report_sha256"],
            os.path.join(OUT, "selfcheck.json"): fc["check"]["selfcheck_sha256"]}
    bad = [rel(p) for p, sha in want.items() if sha_lf(p) != sha]
    if bad:
        raise P07Error(f"design records differ from the SHA-256s frozen_config.json records: {bad}")
    return rec


def apply_prerequisites(finalize=False):
    """-> (frozen config, records). Real mode only; everything committed and clean; never reads data/local."""
    fpath = os.path.join(OUT, "frozen_config.json")
    mpath = os.path.join(OUT, "hs_map.npz")
    hs_files = [os.path.join(HS_OUT, n) for n in ("val_hs.npz", "val_report.json", "val_seam_moves.npz")]
    rec = {"frozen_config": require_committed([fpath, mpath], "frozen_config.json / hs_map.npz"),
           "val_hs": require_committed(hs_files, "val_hs.npz / val_report.json / val_seam_moves.npz")}
    fc = read_json(fpath)
    if fc.get("format") != FORMATS["frozen"] or fc.get("preview") or fc.get("score_pending_val_hs") is not True:
        raise P07Error("frozen_config.json is not a passed (gates) real frozen configuration")
    if sha_raw(mpath) != fc["hs"]["hs_map_npz_sha256"]:
        raise P07Error("hs_map.npz differs from the SHA-256 frozen_config.json records")
    rec["design_records"] = design_records_bound(fc)
    ts = tool_state()
    require_tools(ts, False)
    diff = tools_equal_record(ts, fc.get("tool"))
    if diff:
        raise P07Error(f"tool files differ from the ones tune recorded: {diff}")
    vr = read_json(hs_files[1])
    if vr.get("format") != FORMATS["val_report"] or vr.get("mode") != "real":
        raise P07Error("val_report.json is not a real p07_hs.py val report")
    ft = commit_time(fpath)
    first = vr.get("first_chunk_started_utc")
    if not ft or not first or datetime.datetime.fromisoformat(first) <= datetime.datetime.fromisoformat(ft):
        raise P07Error(f"val_report.json start {first} is not later than frozen_config.json's commit {ft}")
    if vr.get("frozen_config_sha256") != sha_lf(fpath):
        raise P07Error("val_report.json names another frozen_config.json")
    bad = []
    for ch in vr.get("chunks") or []:
        p = path_of(ch["file"])
        if not os.path.exists(p) or sha_raw(p) != ch["sha256"]:
            bad.append(ch["file"])
    if bad:
        raise P07Error(f"validation HS chunks missing or changed: {bad[:5]}")
    if sha_raw(hs_files[0]) != vr.get("val_hs_sha256") or sha_raw(hs_files[2]) != vr.get("val_seam_moves_sha256"):
        raise P07Error("val_hs.npz / val_seam_moves.npz differ from val_report.json")
    rec.update(frozen_commit_time=ft, val_first_chunk_started=first, tool=ts)
    return fc, vr, rec


AUDIT_BOUND_OUTPUTS = ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json")


def audit_state(rep=None):
    """prescore_audit.json (written by another agent, no scores): committed, format c10-prereg07 prescore_audit v1,
    its outputs_sha256 (rank_val_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json; LF-folded) equal to
    apply_report.json's, its inputs_sha256 val_hs.npz / hs_map.npz equal to apply's, and committed after
    apply_report.json. Field program_errors_affecting_Q02H: a list (empty = none found). -> (state, record)."""
    p = os.path.join(OUT, "prescore_audit.json")
    if not os.path.exists(p):
        return "missing", None
    if tracked_clean([p]):
        return "missing", {"note": "prescore_audit.json exists but is not committed"}
    a = read_json(p)
    if a.get("format") != FORMATS["audit"]:
        return "missing", {"note": "prescore_audit.json has the wrong format"}
    rec = {"sha256": sha_lf(p), "commit": commit_time(p)}
    if rep is not None:
        probs = []
        for n_ in AUDIT_BOUND_OUTPUTS:
            if (a.get("outputs_sha256") or {}).get(n_) != rep["outputs_sha256"].get(n_):
                probs.append(f"audit outputs_sha256[{n_}] differs from apply_report.json")
        for n_ in ("val_hs.npz", "hs_map.npz"):
            if (a.get("inputs_sha256") or {}).get(n_) != rep["inputs_sha256"].get(n_):
                probs.append(f"audit inputs_sha256[{n_}] differs from apply_report.json")
        at = commit_time(os.path.join(OUT, "apply_report.json"))
        if (not at or not rec["commit"]
                or datetime.datetime.fromisoformat(rec["commit"]) <= datetime.datetime.fromisoformat(at)):
            probs.append(f"audit commit {rec['commit']} is not later than apply_report.json's {at}")
        if probs:
            rec["binding_problems"] = probs
            return "unbound", rec
    return ("bug" if a.get("program_errors_affecting_Q02H") else "ok"), rec


def q02_validation():
    """Q02 on the validation questions with prereg04's frozen parameters and readers (no data/local file: the C5 /
    SW / O4 / opening validation inputs are committed results), and its byte comparison with
    results/prereg04/rank/rank_val_probs.csv (PREREG_07 2.7). -> dict."""
    ns = argparse.Namespace(smoke=False, design_o4=False, p02_dir=None, p02_local_dir=None, p03_dir=None,
                            opp_dir=None, split=None, questions=None, out_dir=None)
    c4 = p4.resolve(ns, "tune")             # real-mode paths; out_dir = results/prereg04/rank, never written here
    frozen_tools = p4.start(c4)
    fc4 = read_json(P04_FROZEN)
    if sha_raw(P04_FROZEN) != P04_FROZEN_SHA or fc4["combo"]["id"] != "Q02" or fc4["combo"]["terms"] != Q02_TERMS:
        raise P07Error("results/prereg04/rank/frozen_config.json is not the pinned Q02 configuration")
    vk, lp5, _vlog, vsha, _wsha = p4.val_c5(c4)
    if vk["sha256"] != VAL_KEYS_SHA or vk["n"] != VAL_GAMES or len(vk["groups"]) != VAL_QUESTIONS:
        raise P07Error("val_keys.npz is not the registered one (1,589 questions, 19,156 games)")
    lpv = {"C5": lp5}
    for s, x in (("SWA", "a"), ("SWB", "b")):
        va = p4.val_arm(c4, x, vk)
        if va["sw"]["sha256"] != p4.SW_SHA256[x]["val"]:
            raise P07Error(f"{s}: validation swapped dump differs from PREREG_04 1.3")
        lpv[s] = va["SW"]
    o4probs, o4rec, ov = p4.load_o4_val(c4, vk)
    if o4probs or o4rec.get("sha256") != VAL_O4_SHA:
        raise P07Error(f"validation O4: {o4probs} / SHA-256 {o4rec.get('sha256')}")
    lpv["O4"] = rf.log_softmax(ov["O4"])
    S, n, mop, _op_o, _op_has, op = p4.val_question_inputs(c4, vk, fc4["opening_build_id"])
    ok = n > 0
    m = rf.pooled(S, lp5, "log")
    E04 = {"OP": mop, "O4": rf.pooled(S, lpv["O4"], "log"), "SWA": rf.pooled(S, lpv["SWA"], "log"),
           "SWB": rf.pooled(S, lpv["SWB"], "log")}
    P_q02 = fuse_probs(fc4["params"]["theta"], m, [E04[t] for t in Q02_TERMS], n, ok)
    q02_bytes = probs_csv_bytes(vk["groups"], P_q02, ok, n)
    with open(P04_PROBS, "rb") as f:
        ref_bytes = f.read()
    ref_lf = ref_bytes.replace(b"\r\n", b"\n")
    seam = {"raw_equal": q02_bytes == ref_bytes,
            "lf_equal": q02_bytes == ref_lf,
            "reference_sha256_lf": hashlib.sha256(ref_lf).hexdigest(),
            "computed_sha256": hashlib.sha256(q02_bytes).hexdigest()}
    seam["passed"] = bool(seam["lf_equal"] and seam["reference_sha256_lf"] == P04_PROBS_SHA)
    return {"vk": vk, "S": S, "n": n, "ok": ok, "m": m, "E04": E04, "P": P_q02, "seam": seam,
            "frozen_tools": frozen_tools, "vsha": vsha, "op": op}


def check_question_file(vk):
    """data/local/rank_val_questions.csv (question_id / num_games / color_i only) equals val_keys row by row -> SHA."""
    qfile = refuse_label_path(QUESTIONS)
    q_sha = sha_raw(qfile)
    if q_sha != QUESTIONS_SHA:
        raise P07Error(f"question file SHA-256 {q_sha} != {QUESTIONS_SHA}")
    questions = rf.read_question_meta(qfile)
    by_q = {qq: (n_, cols) for qq, n_, cols in questions}
    groups = vk["groups"]
    if [qq for qq, _ in groups] != [qq for qq, _, _ in questions]:
        raise P07Error("val_keys questions differ from the question file (set or order)")
    for qq, rows in groups:
        n_, cols = by_q[qq]
        if len(rows) != n_ or [vk["color"][i] for i in rows] != cols:
            raise P07Error(f"question {qq}: val_keys slots / colours differ from the question file")
    return q_sha


def apply_core(PF, q, vh, model, fc):
    """Q02H on the validation questions from q02_validation()'s inputs, the validation HS arrays and the frozen map;
    lp_HS checked against pilot_fuse.mlr_logp, masked E_HS, move cross-check against val_o4.npz, direction check,
    failures. No file is written; nothing under data/local is read. -> dict."""
    vk, S, n, ok, m, E04, P_q02 = q["vk"], q["S"], q["n"], q["ok"], q["m"], q["E04"], q["P"]
    if ([str(x) for x in vh["question_id"]] != vk["question_id"] or [int(x) for x in vh["slot"]] != vk["slot"]
            or [str(x) for x in vh["color"]] != vk["color"]):
        raise P07Error("val_hs.npz keys differ from val_keys.npz")
    lp_hs, hs_ok = hs_lp(model, vh["sums"], vh["counts"], vh["failed"])
    x_pf = np.zeros_like(np.asarray(vh["sums"], dtype=np.float64))
    x_pf[hs_ok] = vh["sums"][hs_ok] / vh["counts"][hs_ok, None].astype(np.float64)
    lp_pf = np.zeros_like(lp_hs)
    lp_pf[hs_ok] = PF.mlr_logp({"W": model["W"], "b": model["b"]}, (x_pf[hs_ok] - model["mu"]) / model["sd"])
    if not np.array_equal(lp_pf, lp_hs):
        raise P07Error("lp_HS differs from pilot_fuse.mlr_logp on the same map")
    usable = usable_games(vk["parsed"], vh["failed"], vh["counts"])
    E_hs, e_info = masked_E(S, lp_hs, usable)
    Es = [E04[t] for t in Q02_TERMS] + [E_hs]
    P = fuse_probs(fc["params"]["theta"], m, Es, n, ok)
    with np.load(VAL_O4, allow_pickle=False) as z:
        o4z = {k: z[k] for k in z.files}
    if ([str(x) for x in o4z["question_id"]] != vk["question_id"] or [int(x) for x in o4z["slot"]] != vk["slot"]
            or [str(x) for x in o4z["color"]] != vk["color"]):
        raise P07Error("val_o4.npz keys differ from val_keys.npz")
    ids = [f"{qq}#{s}" for qq, s in zip(vk["question_id"], vk["slot"])]
    cross = move_cross_check(vh["n_moves"], vh["counts"], vh["failed"], o4z["moves_parsed"],
                             np.asarray(o4z["n_moves"])[:, 0, :].sum(1), o4z["replay_failed"], ids)
    has = questions_with_usable(S, usable) & ok
    rho, nq = hs_rho(E_hs, P_q02, has)
    direction = direction_record(rho, nq, fc["hs_direction_ref"]["rho_eval"], RHO_FLOOR_VAL)
    fail_count = int(hs_failures(vh["failed"], vh["n_moves"]).sum())
    return {"P": P, "E_hs": E_hs, "e_info": e_info, "usable": usable, "has": has, "cross": cross,
            "direction": direction, "fail_count": fail_count, "lp_hs": lp_hs,
            "csv_bytes": probs_csv_bytes(vk["groups"], P, ok, n)}


def cmd_apply(a):
    t0 = time.time()
    out_files = ["rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "check_submission.txt",
                 "apply_report.json"]
    if a.finalize:
        return apply_finalize(a, out_files)
    fc, vr, pre = apply_prerequisites()
    c = Ctx()
    c.out_dir, c.preview, c.precommit = OUT, False, False
    rerun = guard_outputs(c, a, out_files)
    # ---- Q02 through prereg04's readers, then the question file (both before pilot_fuse / hs_lib are imported)
    q = q02_validation()
    if not q["seam"]["passed"]:
        raise Stop(f"Q02 does not reproduce results/prereg04/rank/rank_val_probs.csv byte for byte ({q['seam']}); "
                   f"stop and find out why (PREREG_07 2.7); nothing was written")
    q_sha = check_question_file(q["vk"])
    vk, S, n, ok, m, E04, P_q02, q02 = q["vk"], q["S"], q["n"], q["ok"], q["m"], q["E04"], q["P"], q["seam"]
    groups, frozen_tools, vsha, op = vk["groups"], q["frozen_tools"], q["vsha"], q["op"]
    # ---- from here on no data/local file is opened (pilot_fuse imports hs_lib and its guard)
    PF = import_pilot_fuse()
    with np.load(os.path.join(HS_OUT, "val_hs.npz"), allow_pickle=False) as z:
        vh = {k: z[k] for k in z.files}
    model = load_hs_map(os.path.join(OUT, "hs_map.npz"))
    core = apply_core(PF, q, vh, model, fc)
    P, E_hs, e_info, usable, has = core["P"], core["E_hs"], core["e_info"], core["usable"], core["has"]
    cross, direction, fail_count, lp_hs = core["cross"], core["direction"], core["fail_count"], core["lp_hs"]
    rho = direction["rho"]
    out_csv = os.path.join(OUT, "rank_val_probs.csv")
    with open(out_csv + ".tmp", "wb") as f:
        f.write(core["csv_bytes"])
    os.replace(out_csv + ".tmp", out_csv)
    import decide
    decide.read_probs(out_csv)
    # ---- decide.py and check_submission.py (subprocesses; this process never reads data/local again)
    py = sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r1 = subprocess.run([py, os.path.join(HERE, "decide.py"), out_csv, OUT], capture_output=True, text=True,
                        cwd=REPO, env=env)
    if r1.returncode != 0:
        raise P07Error(f"decide.py failed: {r1.stderr[-2000:]}")
    txt, formats_ok = [], True
    for pred in ("pred_A.csv", "pred_B.csv"):
        r2 = subprocess.run([py, os.path.join(HERE, "check_submission.py"), "rank", os.path.join(OUT, pred),
                             "--questions", QUESTIONS_REL], capture_output=True, text=True, cwd=REPO, env=env)
        txt.append(f"$ check_submission.py rank results/prereg07/rank/{pred} --questions data/local/"
                   f"rank_val_questions.csv\n(exit {r2.returncode})\n{r2.stdout}{r2.stderr}")
        formats_ok = formats_ok and r2.returncode == 0
    with open(os.path.join(OUT, "check_submission.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(txt))
    audit, audit_rec = audit_state()
    allowed, reasons = apply_decision(fail_count, getattr(a, "accept_data_failures", None), q02["passed"], cross,
                                      direction, audit, formats_ok)
    pred_q02 = rf.rule_b_np(np.where(ok[:, None], P_q02, 0.0))
    pred_h = rf.rule_b_np(np.where(ok[:, None], P, 0.0))
    rep = {"format": FORMATS["apply"], "what": "PREREG_07 2.7 apply: Q02H on the validation questions (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "mode": "real", "rerun": rerun,
           "prerequisites": pre, "q02_reproduction": q02, "frozen_tools_p04": frozen_tools,
           "score_allowed": allowed, "score_blocked_reasons": reasons,
           "accept_data_failures": getattr(a, "accept_data_failures", None),
           "hs_failures": {"failed_or_no_move": fail_count, "failed": int(np.asarray(vh["failed"]).sum()),
                           "no_target_move_not_failed": int(((np.asarray(vh["counts"]) == 0) &
                                                             ~np.asarray(vh["failed"])).sum()),
                           "unusable_games": int((~usable).sum()), "limit": VAL_FAIL_MAX},
           "move_cross_check": cross, "hs_direction": direction, "masked_E_HS": e_info, "audit": audit_rec,
           "questions": len(groups), "questions_without_parsed_game": int((~ok).sum()),
           "questions_without_usable_hs_game": int((ok & ~has).sum()),
           "rule_B_differs_from_Q02": int(((pred_h != pred_q02) & ok).sum()),
           "pred_distribution": dict(zip(LEVELS, np.bincount(pred_h[ok], minlength=L).tolist())),
           "lp_hs_summary": {"mean": np.mean(lp_hs[usable], axis=0) if usable.any() else None},
           "E_hs_summary": {"mean": np.mean(E_hs[has], axis=0) if has.any() else None},
           "se_val_record_from_tune": fc.get("se_val"),
           "inputs_sha256": {"frozen_config.json": sha_lf(os.path.join(OUT, "frozen_config.json")),
                             "hs_map.npz": sha_raw(os.path.join(OUT, "hs_map.npz")),
                             "val_hs.npz": sha_raw(os.path.join(HS_OUT, "val_hs.npz")),
                             "val_report.json": sha_lf(os.path.join(HS_OUT, "val_report.json")),
                             "val_report_status": vr.get("status"), "questions": q_sha,
                             "val_keys.npz": vk["sha256"], "val_o4.npz": sha_raw(VAL_O4),
                             "opening_val.npz": op["sha256"], **vsha},
           "outputs_sha256": {n_: pin_sha(os.path.join(OUT, n_)) for n_ in out_files[:5]},
           "decide_stdout": r1.stdout[-2000:], "tool": tool_state(), "environment": environment_cpu(),
           "next": "commit; prescore_audit.json by another agent (no scores) -> commit; apply --finalize; H7R "
                   "(PREREG_07 3) only when score_allowed is true"}
    write_json(os.path.join(OUT, "apply_report.json"), rep)
    print(f"apply Q02H: Q02 reproduction {q02['passed']}; cross-check mismatches {cross['mismatches']}; rho "
          f"{rho:.4f} (threshold {direction['threshold']:.4f}); failed {fail_count}; formats {formats_ok}")
    print(f"score_allowed: {allowed} {reasons}")
    return 0


def apply_finalize(a, out_files):
    rp = os.path.join(OUT, "apply_report.json")
    if require_committed([rp] + [os.path.join(OUT, f) for f in out_files[:5]], "apply outputs")["commit_problems"]:
        raise P07Error("commit apply's outputs first")
    rep = read_json(rp)
    for n_, sha in rep["outputs_sha256"].items():
        if pin_sha(os.path.join(OUT, n_)) != sha:
            raise P07Error(f"{n_} changed since apply")
    if rep.get("score_allowed") is not False and rep.get("score_allowed") != "pending_audit":
        raise P07Error(f"score_allowed is already {rep.get('score_allowed')!r}")
    audit, audit_rec = audit_state(rep)
    if audit == "missing":
        raise P07Error("prescore_audit.json is not committed")
    if audit == "unbound":
        raise P07Error(f"prescore_audit.json is not an audit of these outputs: {audit_rec['binding_problems']}")
    blocked = [r for r in rep["score_blocked_reasons"] if not r.startswith("prescore_audit.json not committed")]
    if rep.get("score_allowed") is False and blocked:
        raise P07Error(f"score_allowed is false for other reasons: {blocked}")
    if audit == "bug":
        rep["score_allowed"], rep["score_blocked_reasons"] = False, [
            "prescore_audit.json found a program error affecting Q02H's predictions"]
    else:
        rep["score_allowed"], rep["score_blocked_reasons"] = True, []
    rep["audit"] = audit_rec
    rep["finalized"] = now_iso()
    rep["finalized_by_tool"] = tool_state()["files_sha256"]
    write_json(rp, rep)
    print(f"score_allowed: {rep['score_allowed']}")
    return 0


# ================================================================ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "selfcheck", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--precommit", action="store_true")
        p.add_argument("--out-dir", default=None)
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
        if name == "tune":
            p.add_argument("--revision", type=int, default=None)
    p = sub.add_parser("apply")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--accept-data-failures", default=None, metavar="TEXT")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    fn = {"check": cmd_check, "selfcheck": cmd_selfcheck, "tune": cmd_tune, "apply": cmd_apply}[a.cmd]
    try:
        return fn(a)
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 1
    except ERRORS as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
