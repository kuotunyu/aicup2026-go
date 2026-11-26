"""PREREG_08 competition 2: Q08 = Q02H + 4 x P1TO (docs/PREREG_08_STRENGTH_PROBE.md rev 1, sections 1.4-1.6 and 2;
results/prereg08/design/spec.md sections A, B, D).

    PY = .venv/Scripts/python.exe   (CPU; Git Bash; PYTHONDONTWRITEBYTECODE=1)
    PY tools/p08_rank_fuse.py check      [--precommit | --out-dir data/prereg08/tmp/<x>]  -> results/prereg08/rank/check_report.json
    PY tools/p08_rank_fuse.py selfcheck  [--precommit | --out-dir ...]                    -> .../selfcheck.json
    PY tools/p08_rank_fuse.py tune       [--precommit | --out-dir ...] [--rerun --reason T --revision N]
                                                                -> .../tune_report.json
                                                                (gates passed: + frozen_config.json, p1_map.npz)
    PY tools/p08_rank_fuse.py apply      [--accept-data-failures TEXT] [--rerun --reason T]
                                                                -> rank_val_probs.csv, pred_A/B.csv, decide_summary.json,
                                                                   check_submission.txt, val_lp_p1.npz, apply_report.json
    PY tools/p08_rank_fuse.py apply --finalize   (after prescore_audit.json is committed: score_allowed only)

Modes (as tools/p07_rank_fuse.py)
    real        every input and every earlier step committed and clean (tools included); writes results/prereg08/rank.
    --precommit the same checks and outputs as real, except that "committed and clean" is not required of the p08
                tools, inner_manifest.json, check_report.json and selfcheck.json (recorded by SHA-256 instead; the
                report says "precommit": true). For the first build only. Later steps (p08_p1.py val, apply) require
                the committed files to have exactly the SHA-256s recorded.
    --out-dir   preview under data/prereg08/tmp/ (flagged "preview": true; never accepted by a later step).
apply has neither: it needs the committed frozen_config.json, p1_map.npz, val_p1.npz, val_seam_feats.npz and
val_report.json.

Discipline (spec A)
    * data/local is read only by p08_p1.py val and by apply (through tools/p07_rank_fuse.py's q02_validation /
      check_question_file: question_id / num_games / color_i), and only before p1_lib / pilot_fuse / hs_lib are
      imported. check / selfcheck / tune never read it (u1_lib's and hs_lib's import-time guards).
    * u1_lib is imported only by check, selfcheck and tune (_import_u1); apply never imports it and never calls
      pilot_fuse.fit_mapping or p1_fuse.fit_mapping_grouped.
    * the pinned files (spec B) are imported only after their SHA-256 (CRLF folded to LF) equals the pinned value;
      never copied or edited. tools/p07_rank_fuse.py is checked before it is imported (top of this file).
    * temporary files only under data/prereg08/tmp/; never %TEMP%. tools/score.py is never run.
    * the validation label file is never named or opened (every path goes through refuse_label_path).
Exit codes: 0 done; 1 the step ran and says stop (gates not passed / check stop); 2 refused or error.
"""

import argparse
import datetime
import glob
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


class P08Error(RuntimeError):
    pass


class Stop(RuntimeError):
    """The step ran; its registered outcome is a stop (exit 1)."""


def _sha_lf_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


# 08 imports the p07 tools: they are pinned too (spec B) and checked before the import
P07_TOOL_PINS = {"tools/p07_rank_fuse.py": "8e144e2fe14001af6592130611c1bfce3f49c7bae8208137d2bb3f38ed5b7142",
                 "tools/p07_hs.py": "b27ea371bdc1477558effb9ca86c540805dcced971d983226e15fb5e19b7b8dc"}
_bad = [r for r, s in P07_TOOL_PINS.items() if _sha_lf_file(os.path.join(REPO, *r.split("/"))) != s]
if _bad:
    raise P08Error(f"pinned p07 tools changed: {_bad} (spec B: never edited)")

import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
import p04_rank_fuse as p4  # noqa: E402
import p07_rank_fuse as P7  # noqa: E402

# ================================================================ constants (spec B; PREREG_08 1.2-2.7)
_LABEL = "ans" + "wer"
LEVELS = rf.LEVELS
L = rf.N_LEVELS
P1_CODE = os.path.join(REPO, "results", "prereg08", "p1")
P2_CODE = os.path.join(REPO, "results", "prereg08", "p2")
U1_DIR = P7.U1_DIR
PILOT_CODE = P7.PILOT_CODE
PINNED = {  # rel path -> SHA-256 (CRLF folded to LF), PREREG_08 1.2 / spec B
    "results/prereg07/explore/pilot_humansl/hs_lib.py":
        "d19b6a8615ef757891ff2d451fe5b96b19f31cc706a87862e49a7688d0bde97c",
    "results/prereg07/explore/pilot_humansl/pilot_run.py":
        "b0d127fd1d5a83af5b17ec3df1dccc85d94b11c3479d33cac07448a3796a3936",
    "results/prereg07/explore/pilot_humansl/pilot_fuse.py":
        "37fb2dff0f37032bf08acc7f27213cafd30533ab024973f92b5b28f41c6ac0c6",
    "results/prereg05/explore/u1/u1_lib.py":
        "225dd51cff961ae3608e003432388e0b6d6a053fb4dac26add5425c002dc3ff7",
    "results/prereg08/p1/p1_lib.py": "062600168326f9eecd4ec1073e54881c475f8f276e376164eab20d373bbff511",
    "results/prereg08/p1/p1_extract.py": "2ee8eb127150e672b62bcb30b70c4fac46b8cd6e32c7e4eb19d54f746a6e3f9c",
    "results/prereg08/p1/p1_fuse.py": "ad356c8ca9f26e1e761677afa07b5aa369ca31e5318f61eb6a6cbda33b970953",
    "results/prereg08/p1/p1_plan.json": "0cd46bf90c764cae3824f5906e72e13336ab7276ef5fa7edb2ec6831911b228d",
    "results/prereg08/p2/p2_sets.py": "98bad3d85a32e80c065226b0cc8cd1065b423cdb39dce89829a54b0cbf381c8f",
    **P07_TOOL_PINS}
P1_SETS = os.path.join(REPO, "data", "prereg08", "p1", "sets.json")
P2_SETS = os.path.join(REPO, "data", "prereg08", "p2", "sets.json")
DATA_PINS = {  # raw SHA-256
    "data/prereg08/p1/sets.json": "f0c494962788615d76080b5481de5b762d71f3ad37f8add55a09efef82acb7c7",
    "data/prereg08/p2/sets.json": "2b68f7ae27b2b6bbe00a3920065495d2a5d6b73fd7df33854f4bf99ae5390049",
    "data/prereg07/pilot/plan.json": "8494e9afd79c06249bd81545263b86caef7b9e77f3cd631a1e354368cfe33946"}
P07_FROZEN = os.path.join(REPO, "results", "prereg07", "rank", "frozen_config.json")
P07_HS_MAP = os.path.join(REPO, "results", "prereg07", "rank", "hs_map.npz")
P07_PROBS = os.path.join(REPO, "results", "prereg07", "rank", "rank_val_probs.csv")
P07_PRED_B = os.path.join(REPO, "results", "prereg07", "rank", "pred_B.csv")
P07_VAL_HS = os.path.join(REPO, "results", "prereg07", "hs", "val_hs.npz")
P07_FROZEN_SHA = "93a0ac29615260865d554ee61aec258c633820a87aeb7dfcf8047a5dfd200c0b"
P07_HS_MAP_SHA = "a4334e79c490e4d8102dd5136ac08c0de1060315981f4ccc527ebffa1b84a3e4"
P07_PROBS_SHA = "6ae2db6025a903eb5c58af9418dc203c0b6e84dd664049f85dfa850e57dc28c1"
P07_PRED_B_SHA = "51d7dbf82d50c3848acadb6934069580bd3ade7e6ca71ad5d1023633a18a26fa"
P07_VAL_HS_SHA = "ef2067e8cb633f521c205a09aa4638c79fef3ba0b4e90e6a54bdc046e5a5550f"
P07_PINS = {"results/prereg07/rank/frozen_config.json": P07_FROZEN_SHA,      # pin_sha: .json / .csv LF, .npz raw
            "results/prereg07/rank/hs_map.npz": P07_HS_MAP_SHA,
            "results/prereg07/rank/rank_val_probs.csv": P07_PROBS_SHA,
            "results/prereg07/rank/pred_B.csv": P07_PRED_B_SHA,
            "results/prereg07/hs/val_hs.npz": P07_VAL_HS_SHA,
            "results/prereg04/opp/val_o4.npz": P7.VAL_O4_SHA,
            "results/prereg04/rank/rank_val_probs.csv": P7.P04_PROBS_SHA}
TOOL_FILES = ("tools/p08_p1.py", "tools/p08_rank_fuse.py")
NO_PYCACHE_DIRS = ("results/prereg08", "results/prereg07", "results/prereg05/explore/u1", "third_party/katago")
KATAGO_COMMIT = P7.KATAGO_COMMIT
CKPT_SHA = P7.CKPT_SHA
CKPT_BYTES = P7.CKPT_BYTES
CHUNK_FORMAT = "c10-prereg08 p1 chunk v1"
NEUTRAL_META_SHA = "404bb7096c63136d7642f5abc98ec84d3c826290451ed649fcbe37c903307282"   # p1_lib.neutral_table_sha()
RUN = {"precision": "fp16", "device": "cuda", "batch": 1024, "channels_last": True, "group_batches": 4,
       "workers": 20, "chunk": 200, "sides": ["target", "opponent"], "cudnn_benchmark": True, "pad": True}
F_DIM = 770
P1_DATA = os.path.join(REPO, "data", "prereg08", "p1")
P2_DATA = os.path.join(REPO, "data", "prereg08", "p2")
PILOT_DATA = P7.PILOT_DATA
SETS = {  # PREREG_08 2.2 item 1
    "p1_probe": {"dir": os.path.join(P1_DATA, "probe"), "sets": P1_SETS, "key": "probe", "chunks": 150,
                 "games": 30000},
    "bank": {"dir": os.path.join(P1_DATA, "bank"), "sets": P1_SETS, "key": "bank", "chunks": 60, "games": 11813},
    "mono": {"dir": os.path.join(P1_DATA, "mono"), "sets": P1_SETS, "key": "mono", "chunks": 307, "games": 61356},
    "p2_probe": {"dir": os.path.join(P2_DATA, "probe"), "sets": P2_SETS, "key": "probe", "chunks": 600,
                 "games": 120000}}
P1_SET_NAMES = ("p1_probe", "bank", "mono")
P2_FAIL_MAX = 600
P2_CUTOFF = datetime.datetime(2026, 10, 8, 23, 59, 59, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
LAMBDAS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
MAXITER = 2000
SCALE = 4.0
TOL = 1e-12
GATE = 0.010
MONO_QUESTIONS = P7.MONO_QUESTIONS
MONO_SHA = P7.MONO_SHA
VAL_FAIL_MAX = 95
RHO_FLOOR_VAL = 0.078
RHO_FLOOR_TEST = 0.155
VAL_QUESTIONS = P7.VAL_QUESTIONS
VAL_GAMES = P7.VAL_GAMES
Q02_TERMS = ["OP", "O4", "SWA", "SWB"]
Q02H_TERMS = Q02_TERMS + ["HS"]
Q08_TERMS = Q02H_TERMS + ["P1TO"]
# P1 numbers selfcheck reproduces (spec D; PREREG_08 2.3; results/prereg08/p1/p1_fuse_report.json, p1_explore_report)
P1REF = {"Q02H": 0.6242752597994481, "lambda": 0.01, "cv_logloss": 1.6790734173594255,
         "refit_loss": 1.585088334325938,
         "x1": {"rank_balanced": 0.675034627957959, "diff": 0.0507593681585109,
                "ci95": [0.03701489044315354, 0.06537945687999477], "mono_diff": 0.04964601127852308,
                "mono_B": 0.055909493750024186, "mono_W": 0.04309679963025359},
         "x4": {"rank_balanced": 0.6767709161878595, "diff": 0.05249565638841136,
                "ci95": [0.0386137095020332, 0.06756280199569697], "mono_diff": 0.05075085423026038}}
OUT = os.path.join(REPO, "results", "prereg08", "rank")
FEAT = os.path.join(REPO, "results", "prereg08", "feat")
MANIFEST = os.path.join(FEAT, "inner_manifest.json")
VAL_DATA = os.path.join(REPO, "data", "prereg08", "val")
TMP = os.path.join(REPO, "data", "prereg08", "tmp")
PREREG = os.path.join(REPO, "docs", "PREREG_08_STRENGTH_PROBE.md")
REV1_COMMIT = "543315a"
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
FORMATS = {"check": "c10-prereg08 rank check_report v1", "selfcheck": "c10-prereg08 rank selfcheck v1",
           "tune": "c10-prereg08 rank tune_report v1", "frozen": "c10-prereg08 rank frozen_config v1",
           "apply": "c10-prereg08 rank apply_report v1", "p1_map": "c10-prereg08 p1_map v1",
           "manifest": "c10-prereg08 inner_manifest v1", "val_report": "c10-prereg08 val_report v1",
           "audit": "c10-prereg08 prescore_audit v1", "val_chunk": "c10-prereg08 val chunk v1",
           "val_lp": "c10-prereg08 val_lp_p1 v1"}
# functions of this file that may import u1_lib or call a mapping fit (tests/test_p08_hygiene.py checks)
INNER_ONLY_FUNCS = ("_import_u1", "cmd_check", "cmd_selfcheck", "cmd_tune", "_fit_p1_map")


ERRORS = (P08Error, P7.P07Error, p4.P04Error, p3.P03Error, rf.P02Error)


# ================================================================ small helpers
def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def refuse_label_path(path):
    if path is not None and _LABEL in os.path.abspath(os.fspath(path)).lower():
        raise P08Error(f"refusing {path}: the validation label file is never opened by the prereg08 tools")
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
        raise P08Error("refusing to write under data/local")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(obj), f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def require_committed(paths, what, precommit=False):
    """Real mode: every path committed and clean. --precommit: recorded by SHA-256 instead."""
    missing = [rel(p) for p in paths if not os.path.exists(p)]
    if missing:
        raise P08Error(f"{what}: missing {missing}")
    rec = {"files": {rel(p): pin_sha(p) for p in paths}}
    probs = tracked_clean(paths)
    rec["commit_problems"] = probs
    if probs and not precommit:
        raise P08Error(f"{what}: {probs} (commit each step before the next)")
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
        raise P08Error("git is not available")
    if ts["dirty_or_untracked"] and not precommit:
        raise P08Error(f"commit the tools first (uncommitted: {ts['dirty_or_untracked']})")


def tools_equal_record(ts, recorded):
    """The current tool files equal the SHA-256s a record holds -> list of differing files."""
    rec = (recorded or {}).get("files_sha256") or {}
    return [f for f in list(TOOL_FILES) + list(PINNED) if rec.get(f) != ts["files_sha256"].get(f)]


def check_pins():
    """-> {rel: sha}; raises unless every pinned code file (LF) and data file (raw) has its pinned SHA-256."""
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    got = {r: sha_lf(path_of(r)) for r in PINNED}
    bad = [r for r in PINNED if got[r] != PINNED[r]]
    for r, want in DATA_PINS.items():
        got[r] = sha_raw(path_of(r)) if os.path.exists(path_of(r)) else None
        if got[r] != want:
            bad.append(r)
    if bad:
        raise P08Error(f"pinned files changed: {bad} (never edit them; PREREG_08 1.2 / spec B)")
    return got


def check_p07_pins():
    """07 / 04 committed files 08 builds on (spec B) -> {rel: sha}; raises on a difference."""
    got = {r: (pin_sha(path_of(r)) if os.path.exists(path_of(r)) else None) for r in P07_PINS}
    bad = [r for r in P07_PINS if got[r] != P07_PINS[r]]
    if bad:
        raise P08Error(f"07 / 04 frozen files differ from spec B: {bad}")
    return got


def no_pycache():
    found = []
    for d in NO_PYCACHE_DIRS:
        for root, dirs, _files in os.walk(path_of(d)):
            found.extend(rel(os.path.join(root, x)) for x in dirs if x == "__pycache__")
    return found


BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
PREVIEW_THREADS_OK = "C10_P08_PREVIEW_THREADS"


def environment_cpu():
    return {**P7.environment_cpu(), "blas_env": {k: os.environ.get(k) for k in BLAS_ENV}, "cpu_count": os.cpu_count()}


def blas_env_check(preview):
    """The byte seams (Q02 = 04, Q02H = 07, u1_oof.npz) and the 1e-12 reproductions were computed with OpenBLAS's
    default thread count: a BLAS thread variable changes the rounding (10/6: OMP_NUM_THREADS=4 turns 07's
    rank_val_probs.csv 6ae2db60... into 987136a6...). Real runs refuse it; a preview may opt in with
    C10_P08_PREVIEW_THREADS=1 (its numbers are then not comparable)."""
    set_ = {k: os.environ[k] for k in BLAS_ENV if os.environ.get(k)}
    if set_ and not (preview and os.environ.get(PREVIEW_THREADS_OK) == "1"):
        raise P08Error(f"BLAS thread variables are set ({set_}): unset them (the registered seams use the default "
                       f"OpenBLAS threads)")
    return set_


def import_p1():
    """p1_lib and p1_fuse (their import installs hs_lib's data/local guard). Never imports u1_lib."""
    check_pins()
    if P1_CODE not in sys.path:
        sys.path.insert(0, P1_CODE)
    import p1_lib as L1
    import p1_fuse as F
    for mod, r in ((L1, "p1_lib.py"), (F, "p1_fuse.py")):
        if not inside(mod.__file__, P1_CODE) or os.path.basename(mod.__file__) != r:
            raise P08Error(f"{r} imported from {mod.__file__}")
    if L1.neutral_table_sha() != NEUTRAL_META_SHA:
        raise P08Error("p1_lib.neutral_table_sha() differs from the P1 chunks' metadata SHA")
    return L1, F


def import_p1_extract():
    """p1_extract (sets, set_games, run_chunk); same guard as import_p1."""
    L1, F = import_p1()
    import p1_extract as E
    if not inside(E.__file__, P1_CODE):
        raise P08Error(f"p1_extract imported from {E.__file__}")
    return L1, F, E


def import_pilot_fuse():
    check_pins()
    return P7.import_pilot_fuse()


def _import_u1():
    """u1_lib (check / selfcheck / tune only)."""
    check_pins()
    if U1_DIR not in sys.path:
        sys.path.insert(0, U1_DIR)
    import u1_lib as U
    if not inside(U.__file__, U1_DIR):
        raise P08Error(f"u1_lib imported from {U.__file__}")
    return U


# ================================================================ chunks (PREREG_08 2.2 items 1, 2, 4)
def chunk_meta(path):
    """Metadata, counts and SHA-256 of one P1-format chunk (never the feature arrays) -> (record, ref, failed,
    target counts, n_moves)."""
    with np.load(path, allow_pickle=False) as z:
        ref = np.asarray(z["ref"]).astype(np.int64)
        failed = np.asarray(z["failed"], dtype=bool)
        cnt = np.asarray(z["counts"]).astype(np.int64)
        nm = np.asarray(z["n_moves"]).astype(np.int64)
        try:
            code = json.loads(str(z["code_sha256"]))
        except ValueError:
            code = {}
        meta = {"format": str(z["format"]), "set": str(z["set"]), "sides": [str(x) for x in z["sides"]],
                "precision": str(z["precision"]), "device": str(z["device"]), "batch": int(z["batch"]),
                "channels_last": bool(z["channels_last"]), "katago_commit": str(z["katago_commit"]),
                "meta_sha256": str(z["meta_sha256"]),
                "p1_lib_sha256": code.get("results/prereg08/p1/p1_lib.py"),
                "p1_extract_sha256": code.get("results/prereg08/p1/p1_extract.py"),
                "torch": str(z["torch"]), "created_utc": str(z["created_utc"]),
                "ckpt_sha256_in_chunk": str(z["ckpt_sha256"])}
        means_shape = list(z["means"].shape)
    rec = {"file": rel(path), "sha256": sha_raw(path), "games": int(len(ref)), "failed": int(failed.sum()),
           "no_target_moves": int(((cnt[:, 0] == 0) & ~failed).sum()), "no_move": int((nm == 0).sum()),
           "failed_or_no_move": int((failed | (nm == 0)).sum()), "means_shape": means_shape, **meta}
    return rec, ref, failed, cnt[:, 0], nm


def chunk_meta_want(set_key):
    return {"format": CHUNK_FORMAT, "set": set_key, "sides": RUN["sides"], "precision": RUN["precision"],
            "device": RUN["device"], "batch": RUN["batch"], "channels_last": RUN["channels_last"],
            "katago_commit": KATAGO_COMMIT, "meta_sha256": NEUTRAL_META_SHA,
            "p1_lib_sha256": PINNED["results/prereg08/p1/p1_lib.py"],
            "p1_extract_sha256": PINNED["results/prereg08/p1/p1_extract.py"]}


def chunk_meta_problems(rec, set_key):
    want = chunk_meta_want(set_key)
    diff = sorted(k for k in want if rec.get(k) != want[k])
    if rec.get("means_shape", [None, 2, F_DIM])[1:] != [2, F_DIM]:
        diff.append("means_shape")
    return diff


def chunk_files(d):
    return sorted(f for f in glob.glob(os.path.join(d, "chunk_*.npz")) if ".tmp" not in f)


def p2_decision(present, expected, valid, failed, now, cutoff=P2_CUTOFF):
    """PREREG_08 1.4 (fixed rule, no choice): manifest only after 600 / 600 chunks or after the cutoff; P2 is used
    iff at that moment all 600 chunks are there, pass the chunk check, and failed games <= 600. -> record (raises
    when the manifest may not run yet)."""
    complete = bool(present == expected and valid)
    if not complete and now <= cutoff:
        raise P08Error(f"P2 has {present}/{expected} chunks{'' if valid else ' (some fail the chunk check)'} and "
                       f"it is before {cutoff.isoformat()}: manifest waits (PREREG_08 1.4)")
    used = bool(complete and failed <= P2_FAIL_MAX)
    if used:
        reason = f"600/600 chunks, failed games {failed} <= {P2_FAIL_MAX}"
    elif not complete:
        reason = f"P2 incomplete at the manifest ({present}/{expected} chunks, valid {valid}) after the cutoff"
    else:
        reason = f"P2 failed games {failed} > {P2_FAIL_MAX}"
    return {"complete": complete, "chunks": int(present), "chunks_expected": int(expected), "valid": bool(valid),
            "failed": int(failed), "failed_limit": P2_FAIL_MAX, "used": used, "reason": reason,
            "cutoff": cutoff.isoformat(), "decided_at": now.isoformat(timespec="seconds"),
            "rule": "PREREG_08 1.4: used iff 600/600 chunks present and valid at the manifest and failed <= 600"}


def verify_manifest_chunks(man, names):
    """Every chunk listed in the manifest for the given sets has its SHA-256 now, the count is the registered one,
    and no other chunk file sits in the set folder (p1_fuse.load_p1_set globs every chunk_*.npz) -> problems."""
    probs = []
    for s in names:
        rows = ((man.get("sets") or {}).get(s) or {}).get("chunks") or []
        listed = {os.path.normcase(os.path.abspath(path_of(r["file"]))) for r in rows}
        for r in rows:
            p = path_of(r["file"])
            if not os.path.exists(p) or sha_raw(p) != r["sha256"]:
                probs.append(f"{s}: {r['file']} missing or changed since the manifest")
        on_disk = {os.path.normcase(os.path.abspath(f)) for f in chunk_files(SETS[s]["dir"])}
        extra = sorted(on_disk - listed)
        if extra:
            probs.append(f"{s}: chunk files not in the manifest: {[rel(x) for x in extra][:5]}")
        if len(rows) != SETS[s]["chunks"]:
            probs.append(f"{s}: {len(rows)} chunks in the manifest, registered {SETS[s]['chunks']}")
    return probs


def sets_used(man):
    return list(P1_SET_NAMES) + (["p2_probe"] if (man.get("p2") or {}).get("used") is True else [])


# ================================================================ terms, gates, decisions (pure)
def p1_term(S, lp, usable, scale=SCALE, force_mask=False):
    """E_P1(q) = mean of scale x lp_P1TO over the question's usable games (0 without one); scale applied once,
    before pooling (as p1_explore.py: lp * s, then the fusion). -> (E, info)."""
    E, info = P7.masked_E(S, float(scale) * np.asarray(lp, dtype=np.float64), usable, force_mask=force_mask)
    info["scale"] = float(scale)
    return E, info


def fuse_probs(theta, m, Es, n, ok):
    """Q08 probabilities with the generalised calibration (p04 gen_logits; OP, O4, SWA, SWB, HS, P1TO)."""
    return P7.fuse_probs(theta, m, Es, n, ok)


def probe_rows(parts):
    """Probe games of the mapping. parts: [(name, X [G,d], usable [G], ref [G], games (sets.json probe list),
    duplicate refs)] in order (P1 first, then P2). Unusable games (failed or no target move) and the manifest's
    duplicate refs are dropped. y = level index; player key = '<rank>:<player_id>' (shared by both sets.json, so a
    player in both lands in one CV fold). -> dict X, y, players, source, ref, record."""
    lv_of = {lv: i for i, lv in enumerate(LEVELS)}
    Xs, ys, pls, src, refs, rec = [], [], [], [], [], {}
    for name, X, usable, ref, games, dups in parts:
        ref = np.asarray(ref, dtype=np.int64)
        usable = np.asarray(usable, dtype=bool)
        dup = np.isin(ref, np.asarray(sorted(set(int(x) for x in dups)), dtype=np.int64))
        keep = usable & ~dup
        Xs.append(np.asarray(X)[keep])
        ys.append(np.array([lv_of[games[j]["rank"]] for j in ref[keep]], dtype=np.int64))
        pls.append(np.array([games[j]["rank"] + ":" + games[j]["player_id"] for j in ref[keep]]))
        src.append(np.full(int(keep.sum()), name))
        refs.append(ref[keep])
        rec[name] = {"games_in_chunks": int(len(ref)), "unusable": int((~usable).sum()),
                     "duplicates_excluded": int((dup & usable).sum()), "used": int(keep.sum())}
    players = np.concatenate(pls) if pls else np.zeros(0, dtype=str)
    out = {"X": np.concatenate(Xs) if Xs else None, "y": np.concatenate(ys), "players": players,
           "source": np.concatenate(src), "ref": np.concatenate(refs), "record": rec}
    out["record"]["total"] = {"games": int(len(out["y"])), "players": int(len(set(players.tolist())))}
    return out


def gate_record(rb_new, rb_base, sc_new, sc_base, boot, mono_diff):
    """G1, G2, G4 of PREREG_08 2.5, Q08 vs Q02H (the per-question mean difference is recorded only)."""
    delta = rb_new - rb_base
    d = boot.stats(sc_new) - boot.stats(sc_base)
    lo, hi = p4.ci95(d)
    g1 = {"threshold": GATE, "delta": delta, "passed": bool(delta >= GATE - TOL)}
    g2 = {"ci95": [lo, hi], "se": float(np.std(d, ddof=1)), "passed": bool(lo > 0),
          "bootstrap": f"players within each level, {rf.N_BOOT} draws, default_rng({rf.BOOT_SEED}), paired"}
    bw = {col: ((mono_diff.get("by_colour") or {}).get(col) or {}).get("rank_balanced") for col in p3.MONO_COLOURS}
    g4 = {"rank_balanced_difference": mono_diff.get("rank_balanced"), "ci95": mono_diff.get("ci95"),
          "by_colour_point_estimates": bw, "questions": mono_diff.get("n_questions"),
          "registered_questions": MONO_QUESTIONS,
          "passed": bool(mono_diff.get("n_questions") == MONO_QUESTIONS and mono_diff["ci95"][0] > 0
                         and all(v is not None and v > 0 for v in bw.values())),
          "rule": "all 120,544 single-colour inner pseudo-questions: paired PlayerBootstrap 95 % lower bound > 0, "
                  "black and white point estimates > 0"}
    qmd = {"question_mean_difference": float(np.mean(np.asarray(sc_new) - np.asarray(sc_base))),
           "note": "recorded only (PREREG_08 2.5)"}
    passed = g1["passed"] and g2["passed"] and g4["passed"]
    return {"compared": "Q08 vs Q02H", "G1": g1, "G2": g2, "G4": g4, "question_mean": qmd, "passed": bool(passed),
            "reason": "G1, G2, G4 passed" if passed else
            "a gate failed: not submitted, competition 2 unchanged (07 -> 04 -> R20 -> R10); RESULTS.md, then end"}


def apply_decision(fail_count, accept_data_failures, q02_ok, q02h_ok, cross, direction, audit, formats_ok):
    """score_allowed of PREREG_08 2.7 -> (value True / False / 'pending_audit', reasons). apply never writes True."""
    reasons = []
    if fail_count > 0:
        if not accept_data_failures:
            reasons.append(f"validation P1: {fail_count} failed games (registered 0): stop and find out why "
                           f"(PREREG_08 2.7; a data property is accepted only with --accept-data-failures)")
        elif fail_count > VAL_FAIL_MAX:
            reasons.append(f"validation P1: {fail_count} failed games > {VAL_FAIL_MAX} (0.5 %): not submitted")
    if q02_ok is not True:
        reasons.append("Q02 does not reproduce results/prereg04/rank/rank_val_probs.csv byte for byte")
    if q02h_ok is not True:
        reasons.append("Q02H does not reproduce results/prereg07/rank/rank_val_probs.csv byte for byte")
    if cross.get("mismatches") != 0:
        reasons.append(f"move cross-check (P1 vs val_hs.npz / val_o4.npz): {cross.get('mismatches')} mismatches "
                       f"(registered 0)")
    if not direction.get("passed"):
        reasons.append(f"P1 direction check: rho {direction.get('rho')} < {direction.get('threshold')}")
    if formats_ok is not True:
        reasons.append("check_submission did not pass for pred_A.csv and pred_B.csv")
    if audit == "bug":
        reasons.append("prescore_audit.json found a program error affecting Q08's predictions")
    if reasons:
        return False, reasons
    return "pending_audit", ["prescore_audit.json of these outputs not committed yet (PREREG_08 2.7); then "
                             "apply --finalize"]


AUDIT_BOUND_OUTPUTS = ("rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json")
AUDIT_BOUND_INPUTS = ("val_p1.npz", "p1_map.npz", "val_hs.npz")


def audit_binding(audit, rep, audit_commit, apply_commit):
    """prescore_audit.json bound to apply_report.json: format, the four outputs' and three inputs' SHA-256 equal,
    committed after apply_report.json -> list of problems (empty = bound)."""
    probs = []
    if audit.get("format") != FORMATS["audit"]:
        probs.append(f"audit format {audit.get('format')!r} != {FORMATS['audit']!r}")
    if not isinstance(audit.get("program_errors_affecting_Q08"), list):
        probs.append("audit field program_errors_affecting_Q08 is not a list")
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
    """-> (state 'missing' / 'unbound' / 'bug' / 'ok', record)."""
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
        probs = audit_binding(a, rep, rec["commit"], commit_time(os.path.join(OUT, "apply_report.json")))
        if probs:
            rec["binding_problems"] = probs
            return "unbound", rec
    return ("bug" if a.get("program_errors_affecting_Q08") else "ok"), rec


def finalize_decision(rep, audit, audit_rec):
    """apply --finalize: -> (score_allowed, reasons); raises when it may not finalize."""
    if rep.get("score_allowed") is not False and rep.get("score_allowed") != "pending_audit":
        raise P08Error(f"score_allowed is already {rep.get('score_allowed')!r}")
    if audit == "missing":
        raise P08Error("prescore_audit.json is not committed (finalize only after the audit)")
    if audit == "unbound":
        raise P08Error(f"prescore_audit.json is not an audit of these outputs: {(audit_rec or {}).get('binding_problems')}")
    blocked = [r for r in rep.get("score_blocked_reasons") or [] if not r.startswith("prescore_audit.json")]
    if rep.get("score_allowed") is False and blocked:
        raise P08Error(f"score_allowed is false for other reasons: {blocked}")
    if audit == "bug":
        return False, ["prescore_audit.json found a program error affecting Q08's predictions"]
    if audit != "ok":
        raise P08Error(f"audit state {audit!r}")
    return True, []


def frozen_usable(fc):
    """frozen_config.json accepted by val / apply: the 08 format, real (not preview), gates passed."""
    if fc.get("format") != FORMATS["frozen"]:
        raise P08Error("frozen_config.json is not a c10-prereg08 frozen configuration")
    if fc.get("preview"):
        raise P08Error("frozen_config.json is a preview record (never accepted by a later step)")
    if fc.get("score_pending_val") is not True:
        raise P08Error("frozen_config.json does not say score_pending_val (gates passed)")
    return True


def tune_rerun_check(rerun, preview, revision, old_created, revision_lookup):
    """Real-mode tune --rerun needs a committed PREREG_08 revision N >= 2 later than the tune it replaces."""
    if rerun is None or preview:
        return rerun
    if not revision or revision < 2:
        raise P08Error("tune is judged once: --rerun needs --revision N >= 2 (a committed PREREG_08 revision that "
                       "names a program error)")
    h, t = revision_lookup(revision)
    if old_created and datetime.datetime.fromisoformat(t) <= datetime.datetime.fromisoformat(old_created):
        raise P08Error(f"PREREG_08 rev {revision} ({t}) is not later than the tune it replaces ({old_created})")
    rerun["revision"] = {"rev": revision, "commit": h, "commit_time": t, "replaced_tune_created": old_created}
    return rerun


def revision_committed(n):
    rc, out = git(["log", "--format=%H %cI", "-S", f"| rev {n} |", "--", rel(PREREG)])
    lines = [x for x in out.splitlines() if x.strip()] if rc == 0 else []
    if not lines:
        raise P08Error(f"PREREG_08 rev {n} is not in any commit")
    h, t = lines[-1].split()
    return h, t


# ================================================================ modes and outputs
class Ctx:
    pass


def resolve(a, cmd):
    c = Ctx()
    c.cmd = cmd
    c.precommit = bool(getattr(a, "precommit", False))
    od = getattr(a, "out_dir", None)
    if c.precommit and od:
        raise P08Error("--precommit writes the real folder; it cannot be combined with --out-dir")
    if od:
        c.out_dir = os.path.abspath(od)
        if not inside(c.out_dir, TMP) or inside(c.out_dir, OUT):
            raise P08Error(f"--out-dir (preview) must be under {rel(TMP)} (got {c.out_dir})")
        c.preview = True
    else:
        c.out_dir = OUT
        c.preview = False
    if under_data_local(c.out_dir):
        raise P08Error("refusing to write under data/local")
    return c


def mode_flags(c):
    return {"preview": c.preview, "precommit": c.precommit}


def guard_outputs(c, a, names):
    paths = [os.path.join(c.out_dir, n) for n in names]
    have = [p for p in paths if os.path.exists(p)]
    if not have:
        return None
    if not getattr(a, "rerun", False) or not getattr(a, "reason", None):
        raise P08Error(f"outputs exist ({[os.path.basename(p) for p in have]}); rerun only with --rerun --reason TEXT")
    if not c.preview:
        tracked = [p for p in have if git(["ls-files", "--error-unmatch", "--", rel(p)])[0] == 0]
        if tracked:
            raise P08Error(f"committed outputs are never overwritten: {[rel(p) for p in tracked]}")
    return {"reason": a.reason, "old_sha256": {os.path.basename(p): pin_sha(p) for p in have}}


def prior_report(c, name, fmt, precommit_ok):
    """An earlier step's report in this mode (real / precommit / preview) -> (report, sha, commit record)."""
    p = os.path.join(c.out_dir, name)
    if not os.path.exists(p):
        raise P08Error(f"run the earlier step first: {name} missing in {c.out_dir}")
    r = read_json(p)
    if r.get("format") != fmt:
        raise P08Error(f"{name}: not a {fmt} file")
    if bool(r.get("preview")) != c.preview:
        raise P08Error(f"{name} is a {'preview' if r.get('preview') else 'real'} record; this run is "
                       f"{'a preview' if c.preview else 'real'} (a preview is never accepted by a later step)")
    commit = None if c.preview else require_committed([p], name, precommit=precommit_ok)
    return r, pin_sha(p), commit


def manifest_binding(c, precommit):
    if not os.path.exists(MANIFEST):
        raise P08Error(f"{rel(MANIFEST)} missing: run p08_p1.py manifest first")
    commit = None if c.preview else require_committed([MANIFEST], "inner_manifest.json", precommit=precommit)
    man = read_json(MANIFEST)
    if man.get("format") != FORMATS["manifest"]:
        raise P08Error("inner_manifest.json: wrong format")
    return man, sha_lf(MANIFEST), commit


# ================================================================ check (PREREG_08 2.2; spec D)
def cmd_check(a):
    t0 = time.time()
    c = resolve(a, "check")
    blas_env_check(c.preview)
    rerun = guard_outputs(c, a, ["check_report.json"])
    ts = tool_state()
    if not c.preview:
        require_tools(ts, c.precommit)
    pins = check_pins()
    probs = []
    man, man_sha, man_commit = manifest_binding(c, c.precommit)
    man_tool_diff = tools_equal_record(ts, man.get("tool"))
    if man_tool_diff:
        probs.append(f"inner_manifest.json was made by other tool files: {man_tool_diff}")
    if man.get("complete") is not True:
        probs.append("inner_manifest.json: complete is not true (P1 probe, bank and mono must all be there)")
    p2 = man.get("p2") or {}
    names = sets_used(man)
    L1, _F = import_p1()
    if L1.neutral_table_sha() != NEUTRAL_META_SHA:
        probs.append("p1_lib.neutral_table_sha() differs")
    # 1, 2, 4: every used set: chunk count, SHA-256 now, games, refs, metadata
    chunk_rec = {}
    for s in names:
        srec = (man.get("sets") or {}).get(s) or {}
        rows = srec.get("chunks") or []
        if len(rows) != SETS[s]["chunks"]:
            probs.append(f"{s}: {len(rows)} chunks in the manifest, registered {SETS[s]['chunks']}")
        bad_sha, meta_bad, refs_bad, games = [], [], [], 0
        tot = {"failed": 0, "no_target_moves": 0, "failed_or_no_move": 0}
        for r in rows:
            p = path_of(r["file"])
            if not os.path.exists(p):
                bad_sha.append(r["file"])
                continue
            rec, _ref, _f, _c, _n = chunk_meta(p)
            if rec["sha256"] != r["sha256"]:
                bad_sha.append(r["file"])
                continue
            games += rec["games"]
            for k in tot:
                tot[k] += rec[k]
            diff = chunk_meta_problems(rec, SETS[s]["key"])
            if diff:
                meta_bad.append((r["file"], diff))
            if r.get("ref_matches_sets") is not True:
                refs_bad.append(r["file"])
        if bad_sha:
            probs.append(f"{s}: chunks differ from the manifest SHA-256 or are missing: {bad_sha[:5]}")
        if meta_bad:
            probs.append(f"{s}: chunk metadata differ from PREREG_08 1.2-1.3: {meta_bad[:3]}")
        if refs_bad:
            probs.append(f"{s}: chunk refs differ from sets.json: {refs_bad[:5]}")
        if games != SETS[s]["games"]:
            probs.append(f"{s}: {games} games, registered {SETS[s]['games']}")
        if s in ("bank", "mono") and (tot["failed"] != 0 or tot["no_target_moves"] != 0):
            probs.append(f"{s}: failed {tot['failed']} / no target move {tot['no_target_moves']} (must be 0)")
        chunk_rec[s] = {"chunks": len(rows), "games": games, "sha_mismatch": bad_sha, "meta_mismatch": meta_bad,
                        "refs_mismatch": refs_bad, **tot}
    probs.extend(verify_manifest_chunks(man, names))
    if p2.get("used") is True:
        if not (p2.get("complete") is True and int(p2.get("failed", 10 ** 9)) <= P2_FAIL_MAX):
            probs.append(f"inner_manifest.json p2 record inconsistent with PREREG_08 1.4: {p2}")
    # 3 checkpoint and KataGo sources
    ck = P7.checkpoint_record()
    if not ck["ok"]:
        probs.append(f"checkpoint SHA-256 / size differ from PREREG_08 1.2: {ck}")
    if (man.get("checkpoint") or {}).get("sha256") != CKPT_SHA:
        probs.append("inner_manifest.json does not record the full checkpoint SHA-256")
    kp, krec = P7.katago_sources()
    probs.extend(kp)
    # 5 move cross-check
    xc = man.get("move_cross_check") or {}
    for k in ("vs_pilot_hs", "vs_inner_o4"):
        if (xc.get(k) or {}).get("mismatches") != 0:
            probs.append(f"move cross-check {k}: {(xc.get(k) or {}).get('mismatches')} mismatches (registered 0)")
    # 6 disjoint, duplicates
    dj = man.get("disjoint") or {}
    if dj.get("passed") is not True:
        probs.append(f"probe players / games not disjoint: {dj}")
    dup = man.get("duplicates") or {}
    if not isinstance(dup.get("p1_probe"), list) or not isinstance(dup.get("p2_probe"), list):
        probs.append("inner_manifest.json: duplicates record missing")
    # 7 u1_lib.load_inputs; pilot plan
    plan_sha = sha_raw(P7.PLAN)
    if plan_sha != P7.PLAN_SHA:
        probs.append(f"data/prereg07/pilot/plan.json SHA-256 {plan_sha} != {P7.PLAN_SHA}")
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
    p07p = check_p07_pins()
    pyc = no_pycache()
    if pyc:
        probs.append(f"__pycache__ present: {pyc}")
    rep = {"format": FORMATS["check"], "what": "PREREG_08 2.2 check: are the inputs usable (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c), "rerun": rerun,
           "status": "ok" if not probs else "stop", "problems": probs,
           "inner_manifest": {"sha256": man_sha, "commit": man_commit}, "sets_used": names, "p2": p2,
           "chunks": chunk_rec, "move_cross_check": xc, "disjoint": dj,
           "duplicates": {"p1_probe": len(dup.get("p1_probe") or []), "p2_probe": len(dup.get("p2_probe") or [])},
           "checkpoint": ck, "katago": krec, "pilot_plan_sha256": plan_sha,
           "u1_load_inputs": {"ok": u1_ok, **u1_rec}, "pinned_sha256": pins, "p07_frozen_sha256": p07p,
           "tool": ts, "environment": environment_cpu(), "no_pycache_after": not no_pycache()}
    write_json(os.path.join(c.out_dir, "check_report.json"), rep)
    print(f"check: {rep['status']}" + ("" if not probs else f" ({'; '.join(probs)})"))
    if probs:
        raise Stop("check says stop")
    return 0


# ================================================================ inner pieces shared by selfcheck and tune
def _plan():
    plan = read_json(os.path.join(P1_CODE, "p1_plan.json"))
    if plan["mapping"]["lambdas"] != LAMBDAS:
        raise P08Error("p1_plan.json lambda grid differs from PREREG_08 1.5")
    if plan["fixed_before_results"]["primary_combination"] != "Q02H+P1TO" or \
            plan["fixed_before_results"]["primary_variant"] != "TO":
        raise P08Error("p1_plan.json primary differs from PREREG_08")
    return plan


def _load_sets_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fit_p1_map(PF, U, F, pr):
    """p1_fuse.fit_mapping_grouped on the probe rows (plan lambda grid, maxiter 2000)."""
    return F.fit_mapping_grouped(PF, U, pr["X"], pr["y"], pr["players"], LAMBDAS, maxiter=MAXITER,
                                 log=lambda *x, **k: print(*x, flush=True))


def inner_p1_lp(F, st, mp_logp, n_inner):
    """p1_fuse's call pattern: lp_P1TO of every usable bank / mono game, one mp['logp'] call per set; covered mask."""
    lp = np.zeros((n_inner, L))
    covered = np.zeros(n_inner, dtype=bool)
    for name in ("bank", "mono"):
        s = st[name]
        X, us = F.variant_X(s, "TO")
        lp[s["ref"][us]] = mp_logp(X[us])
        covered[s["ref"][us]] = True
    return lp, covered


def pilot_hs_usable(PF, n_inner):
    """HS usable inner games from the pilot chunks (C5 parsed is all-true on inner; as p07 tune)."""
    with open(P7.PLAN, encoding="utf-8") as f:
        plan = json.load(f)
    sets = {s: PF.load_set(s, len(plan["profiles"]), PILOT_DATA) for s in ("bank", "mono")}
    failed, counts = P7.inner_failed(sets, n_inner)
    return P7.usable_games(np.ones(n_inner, dtype=bool), failed, counts), failed, counts


def run_terms(d, terms, Emap):
    """u1_lib.run_gen's generalised OOF with the HS / P1TO terms taken from Emap[t] = {'fit', 'eval', 'mono'}
    (masked E)."""
    inner, lp = d["inner"], d["lp"]
    Xf, Xe, Sf, Se, Xm, Sm = (inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"], inner["X_mono"],
                              inner["S_mono"])
    lp5 = lp["C5"]
    mf, me, mm = rf.pooled(Sf, lp5, "log"), rf.pooled(Se, lp5, "log"), rf.pooled(Sm, lp5, "log")

    def qt(X, S, which):
        return [Emap[t][which] if t in Emap else (X.mop if t == "OP" else rf.pooled(S, lp[t], "log")) for t in terms]

    Ef, Ee, Em = qt(Xf, Sf, "fit"), qt(Xe, Se, "eval"), qt(Xm, Sm, "mono")
    P, folds = p4.gen_oof_record(Xf, Xe, mf, me, Ef, Ee)
    if not np.array_equal(p4.gen_fold_predict(folds, me, Ee, Xe.n, Xe.fold), P):
        raise P08Error("fold records do not reproduce the OOF probabilities")
    Pm = p4.gen_fold_predict(folds, mm, Em, Xm.n, Xm.fold)
    return P, Pm, folds, {"mf": mf, "Ef": Ef}


def masked_inner(d, lp, usable, scale=None, force=False):
    inner = d["inner"]
    out, info = {}, {}
    for which, S in (("fit", inner["S_fit"]), ("eval", inner["S_eval"]), ("mono", inner["S_mono"])):
        if scale is None:
            out[which], info[which] = P7.masked_E(S, lp, usable, force_mask=force)
        else:
            out[which], info[which] = p1_term(S, lp, usable, scale=scale, force_mask=force)
    return out, info


def load_p1_sets(F, man, names):
    """P1-format chunks of the used sets through p1_fuse.load_p1_set (its format checks), after the manifest
    binding."""
    st = {}
    for s in names:
        st[s] = F.load_p1_set(SETS[s]["key"], os.path.dirname(SETS[s]["dir"]))
        if st[s] is None or len(st[s]["ref"]) != SETS[s]["games"]:
            raise P08Error(f"{s}: {None if st[s] is None else len(st[s]['ref'])} games, registered {SETS[s]['games']}")
    return st


# ================================================================ selfcheck (PREREG_08 2.3; spec D)
def cmd_selfcheck(a):
    t0 = time.time()
    c = resolve(a, "selfcheck")
    blas_env_check(c.preview)
    rerun = guard_outputs(c, a, ["selfcheck.json"])
    ts = tool_state()
    if not c.preview:
        require_tools(ts, c.precommit)
    cr, cr_sha, cr_commit = prior_report(c, "check_report.json", FORMATS["check"], c.precommit)
    if cr.get("status") != "ok":
        raise P08Error("check_report.json says stop; selfcheck does not run")
    if tools_equal_record(ts, cr.get("tool")):
        raise P08Error(f"tools changed since check: {tools_equal_record(ts, cr.get('tool'))}")
    man, man_sha, _mc = manifest_binding(c, c.precommit)
    if man_sha != (cr.get("inner_manifest") or {}).get("sha256"):
        raise P08Error("inner_manifest.json differs from the one check used")
    chunk_probs = verify_manifest_chunks(man, P1_SET_NAMES)
    if chunk_probs:
        raise P08Error(f"P1 chunks differ from inner_manifest.json: {chunk_probs[:5]}")
    _plan()
    PF = import_pilot_fuse()
    _L1, F = import_p1()
    U = _import_u1()
    probs = []
    sets1 = _load_sets_json(P1_SETS)
    st = load_p1_sets(F, man, P1_SET_NAMES)
    d = U.load_inputs(verbose=False)
    inner = d["inner"]
    n_inner = inner["keys"]["n"]
    Xe, Xm, ms = inner["X_eval"], inner["X_mono"], inner["ms"]
    # baseline (= p1_fuse.hs_baseline = tools/p07_rank_fuse.py selfcheck)
    lp_hs, hs_cov, hs_rec = F.hs_baseline(PF, U, d)
    d["lp"]["HS"] = lp_hs
    saved = U.load_saved()
    P_q, Pm_q, _ = U.run_gen(d, Q02_TERMS)
    P_h, Pm_h, _ = U.run_gen(d, Q02H_TERMS)
    seam = {"Q02_eval_P_equal_u1": bool(np.array_equal(P_q, saved["eval_P_Q02"])),
            "Q02_mono_P_equal_u1": bool(np.array_equal(Pm_q, saved["mono_P_Q02"]))}
    sc_h, msc_h = U.score(U.rule_b_np(P_h), Xe.y), U.score(U.rule_b_np(Pm_h), Xm.y)
    rb_h = U.rank_balanced(sc_h, Xe.y)
    seam["Q02H_rank_balanced"] = rb_h
    if not (seam["Q02_eval_P_equal_u1"] and seam["Q02_mono_P_equal_u1"]):
        probs.append(f"Q02 does not reproduce u1_oof.npz: {seam}")
    if abs(rb_h - P1REF["Q02H"]) > TOL:
        probs.append(f"Q02H inner eval {rb_h!r} != {P1REF['Q02H']!r}")
    # the P1-only mapping exactly as p1_fuse (all usable P1 probe games, chunk order)
    pr = st["p1_probe"]
    X, usable = F.variant_X(pr, "TO")
    rows = probe_rows([("p1_probe", X, usable, pr["ref"], sets1["probe"]["games"], [])])
    del X
    mp = _fit_p1_map(PF, U, F, rows)
    map_rec = {"lambda": mp["lambda"], "cv": mp["cv"], "refit_loss": mp["model"]["loss"],
               "games": rows["record"]["total"], "success": mp["model"]["success"], "nit": mp["model"]["nit"]}
    if mp["lambda"] != P1REF["lambda"]:
        probs.append(f"lambda {mp['lambda']} != {P1REF['lambda']}")
    if abs(mp["cv"][str(mp["lambda"])]["logloss"] - P1REF["cv_logloss"]) > TOL:
        probs.append(f"CV log-loss {mp['cv'][str(mp['lambda'])]['logloss']!r} != {P1REF['cv_logloss']!r}")
    if abs(mp["model"]["loss"] - P1REF["refit_loss"]) > TOL:
        probs.append(f"refit loss {mp['model']['loss']!r} != {P1REF['refit_loss']!r}")
    lp, covered = inner_p1_lp(F, st, mp["logp"], n_inner)
    boot = inner["boot"]
    # x1 (p1_fuse.py): u1_lib.run_gen with lp_P1TO
    d["lp"]["P1TO"] = lp
    P1x, Pm1x, _ = U.run_gen(d, Q08_TERMS)
    sc1, msc1 = U.score(U.rule_b_np(P1x), Xe.y), U.score(U.rule_b_np(Pm1x), Xm.y)
    dd = boot.stats(sc1) - boot.stats(sc_h)
    mono_cov = np.array([covered[np.asarray(g)].all() and hs_cov[np.asarray(g)].all() for g in ms["groups"]])
    sub = {"y": ms["y"][mono_cov], "player": ms["player"][mono_cov], "colour": ms["colour"][mono_cov],
           "k": ms["k"][mono_cov], "n_players": ms["n_players"]}
    sboot = U.PlayerBootstrap(sub["player"], sub["y"], sub["n_players"])
    md1 = U.p3.mono_diff_record(sub, sboot, msc1[mono_cov], msc_h[mono_cov])
    x1 = {"rank_balanced": U.rank_balanced(sc1, Xe.y), "diff": U.rank_balanced(sc1, Xe.y) - rb_h,
          "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
          "mono_diff": md1["rank_balanced"], "mono_B": md1["by_colour"]["B"]["rank_balanced"],
          "mono_W": md1["by_colour"]["W"]["rank_balanced"]}
    dx1 = P7.leaves_close(x1, P1REF["x1"])
    if dx1:
        probs.append(f"Q02H + P1TO x1 differs from p1_fuse_report.json: {dx1}")
    # x4 (p1_explore.py E1): lp x 4 before the fusion
    d["lp"]["P1S"] = lp * SCALE
    P4x, Pm4x, _ = U.run_gen(d, Q02H_TERMS + ["P1S"])
    sc4, msc4 = U.score(U.rule_b_np(P4x), Xe.y), U.score(U.rule_b_np(Pm4x), Xm.y)
    dd4 = boot.stats(sc4) - boot.stats(sc_h)
    x4 = {"rank_balanced": U.rank_balanced(sc4, Xe.y), "diff": U.rank_balanced(sc4, Xe.y) - rb_h,
          "ci95": [float(np.percentile(dd4, 2.5)), float(np.percentile(dd4, 97.5))],
          "mono_diff": U.rank_balanced(msc4, Xm.y) - U.rank_balanced(msc_h, Xm.y)}
    dx4 = P7.leaves_close(x4, P1REF["x4"])
    if dx4:
        probs.append(f"Q02H + P1TO x4 differs from p1_explore_report.json: {dx4}")
    # masked E_HS / E_P1 (no unusable inner game) = pooled bit for bit, and the masked Q08 = the x4 run
    hs_usable, _hf, _hc = pilot_hs_usable(PF, n_inner)
    E_hs, hs_info = masked_inner(d, lp_hs, hs_usable)
    E_p1, p1_info = masked_inner(d, lp, covered, scale=SCALE)
    pooled_hs = {k: rf.pooled(inner[f"S_{k}"], lp_hs, "log") for k in ("fit", "eval", "mono")}
    pooled_p1 = {k: rf.pooled(inner[f"S_{k}"], lp * SCALE, "log") for k in ("fit", "eval", "mono")}
    eq = {"E_HS": {k: bool(np.array_equal(E_hs[k], pooled_hs[k])) for k in E_hs},
          "E_P1": {k: bool(np.array_equal(E_p1[k], pooled_p1[k])) for k in E_p1}}
    Pmk, Pmmk, _f, _aux = run_terms(d, Q08_TERMS, {"HS": E_hs, "P1TO": E_p1})
    eq["Q08_P_equal_x4"] = {"eval": bool(np.array_equal(Pmk, P4x)), "mono": bool(np.array_equal(Pmmk, Pm4x))}
    if not (all(eq["E_HS"].values()) and all(eq["E_P1"].values()) and all(eq["Q08_P_equal_x4"].values())):
        probs.append(f"masked E differs from the pooled version on inner: {eq}")
    rep = {"format": FORMATS["selfcheck"],
           "what": "PREREG_08 2.3 selfcheck: the main-branch tool reproduces P1 (a check of the port, P1 probe only)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c), "rerun": rerun,
           "passed": not probs, "problems": probs, "check_report": {"sha256": cr_sha, "commit": cr_commit},
           "inner_manifest_sha256": man_sha, "reference": P1REF, "seam": seam, "hs_mapping": hs_rec,
           "p1_only_mapping": map_rec, "x1": x1, "x4": x4,
           "mono_coverage": f"{int(mono_cov.sum())}/{len(mono_cov)} questions", "masked_equal": eq,
           "masked_info": {"HS": hs_info, "P1": p1_info},
           "pinned_sha256": check_pins(), "tool": ts, "environment": environment_cpu()}
    write_json(os.path.join(c.out_dir, "selfcheck.json"), rep)
    print(f"selfcheck: {'passed' if not probs else 'FAILED'}; Q02H {rb_h!r}; x1 {x1['diff']!r} {x1['ci95']}; "
          f"x4 {x4['diff']!r}; lambda {mp['lambda']}")
    if probs:
        for p in probs:
            print("  -", p)
        raise Stop("selfcheck failed")
    return 0


# ================================================================ tune (PREREG_08 2.4-2.6; spec D)
def save_p1_map(path, mp, rec):
    np.savez(path, format=np.array(FORMATS["p1_map"]), variant=np.array("TO"), mu=mp["mu"], sd=mp["sd"],
             W=mp["model"]["W"], b=mp["model"]["b"], **{"lambda": np.array(mp["lambda"])},
             cv=np.array(json.dumps(mp["cv"], sort_keys=True)), probe_used=np.array(rec["probe_used"]),
             games=np.array(rec["games"]), players=np.array(rec["players"]),
             sets_sha256=np.array(json.dumps(rec["sets_sha256"], sort_keys=True)),
             refit_loss=np.array(mp["model"]["loss"]), scale=np.array(SCALE), features=np.array(2 * F_DIM))


def load_p1_map(path):
    with np.load(path, allow_pickle=False) as z:
        m = {k: z[k] for k in z.files}
    if str(m.get("format")) != FORMATS["p1_map"] or str(m.get("variant")) != "TO" or m["W"].shape != (2 * F_DIM, L):
        raise P08Error(f"{path}: not a {FORMATS['p1_map']} file (variant TO, 1,540 x 10)")
    return {"mu": m["mu"], "sd": m["sd"], "W": m["W"], "b": m["b"], "lambda": float(m["lambda"]),
            "probe_used": str(m["probe_used"]), "scale": float(m["scale"])}


def p1_lp_from_map(F, model, X):
    """lp_P1TO = log_softmax(W z(x) + b) with the frozen map (p1_fuse.standardise / logp_of; NaN opponent -> 0)."""
    return F.logp_of({"W": model["W"], "b": model["b"]}, F.standardise(X, model["mu"], model["sd"])[0])


def cmd_tune(a):
    t0 = time.time()
    c = resolve(a, "tune")
    blas_env_check(c.preview)
    frozen_path = os.path.join(c.out_dir, "frozen_config.json")
    map_path = os.path.join(c.out_dir, "p1_map.npz")
    if not c.preview and git(["ls-files", "--error-unmatch", "--", rel(frozen_path)])[0] == 0:
        raise P08Error("frozen_config.json is committed (frozen); tune never overwrites it")
    rerun = guard_outputs(c, a, ["tune_report.json", "frozen_config.json", "p1_map.npz"])
    old = os.path.join(c.out_dir, "tune_report.json")
    rerun = tune_rerun_check(rerun, c.preview, getattr(a, "revision", None),
                             read_json(old).get("created") if os.path.exists(old) else None, revision_committed)
    ts = tool_state()
    if not c.preview:
        require_tools(ts, c.precommit)
    cr, cr_sha, cr_commit = prior_report(c, "check_report.json", FORMATS["check"], c.precommit)
    sr, sr_sha, sr_commit = prior_report(c, "selfcheck.json", FORMATS["selfcheck"], c.precommit)
    if cr.get("status") != "ok" or sr.get("passed") is not True:
        raise P08Error("check_report.json / selfcheck.json did not pass; tune does not run")
    for what, r in (("check_report.json", cr), ("selfcheck.json", sr)):
        diff = tools_equal_record(ts, r.get("tool"))
        if diff:
            raise P08Error(f"{what} was made by other tool files ({diff}); rerun it with the current tools")
    man, man_sha, man_commit = manifest_binding(c, c.precommit)
    if man_sha != (cr.get("inner_manifest") or {}).get("sha256"):
        raise P08Error("inner_manifest.json differs from the one check used")
    if (sr.get("check_report") or {}).get("sha256") != cr_sha:
        raise P08Error("selfcheck.json was made from another check_report.json")
    names = sets_used(man)
    chunk_probs = verify_manifest_chunks(man, names)
    if chunk_probs:
        raise P08Error(f"chunks differ from inner_manifest.json: {chunk_probs[:5]}")
    use_p2 = "p2_probe" in names
    _plan()
    PF = import_pilot_fuse()
    _L1, F = import_p1()
    U = _import_u1()
    sets1 = _load_sets_json(P1_SETS)
    sets2 = _load_sets_json(P2_SETS) if use_p2 else None
    st = load_p1_sets(F, man, names)
    dup = man.get("duplicates") or {}
    if len({str(v) for v in (man.get("torch_values") or [])}) != 1:     # review minor 7: one torch reference
        raise P08Error(f"inner_manifest.json torch_values {man.get('torch_values')}: need exactly one torch version")
    d = U.load_inputs(verbose=False)
    inner = d["inner"]
    n_inner = inner["keys"]["n"]
    Xf, Xe, Xm, ms = inner["X_fit"], inner["X_eval"], inner["X_mono"], inner["ms"]
    boot, mboot = inner["boot"], inner["mono_boot"]
    lp_hs, _hs_cov, hs_rec = F.hs_baseline(PF, U, d)
    d["lp"]["HS"] = lp_hs
    hs_usable, _hf, _hc = pilot_hs_usable(PF, n_inner)
    # 1 probe set, 2 the final mapping, round trip through p1_map.npz
    parts = []
    for s, sj in (("p1_probe", sets1), ("p2_probe", sets2)):
        if s not in names:
            continue
        X, us = F.variant_X(st[s], "TO")
        parts.append((s, X, us, st[s]["ref"], sj["probe"]["games"], dup.get(s) or []))
    X = us = None                        # review minor 2: drop the last probe copy before the fit
    pr = probe_rows(parts)
    del parts
    print(f"mapping: {pr['record']}", flush=True)
    mp = _fit_p1_map(PF, U, F, pr)
    probe_used = "P1+P2" if use_p2 else "P1"
    map_rec = {"probe_used": probe_used, "games": pr["record"]["total"]["games"],
               "players": pr["record"]["total"]["players"],
               "sets_sha256": {"p1": DATA_PINS["data/prereg08/p1/sets.json"],
                               "p2": DATA_PINS["data/prereg08/p2/sets.json"] if use_p2 else None},
               "probe_record": pr["record"], "p2": man.get("p2")}
    del pr
    lp_mem, covered = inner_p1_lp(F, st, mp["logp"], n_inner)
    os.makedirs(TMP, exist_ok=True)
    tmp_map = os.path.join(TMP, f"p1_map_{os.getpid()}.npz")
    save_p1_map(tmp_map, mp, map_rec)
    model = load_p1_map(tmp_map)
    lp_file = np.zeros_like(lp_mem)
    for name in ("bank", "mono"):
        X, us = F.variant_X(st[name], "TO")
        lp_file[st[name]["ref"][us]] = p1_lp_from_map(F, model, X[us])
    roundtrip = {"per_set_equal": bool(np.array_equal(lp_file, lp_mem)),
                 "note": "per set = p1_fuse's call pattern (one call each for bank and mono)"}
    if not roundtrip["per_set_equal"]:
        os.remove(tmp_map)
        raise P08Error(f"p1_map.npz round trip differs: {roundtrip}")
    lp = lp_mem
    # 3 masked E (HS and P1TO x 4)
    E_hs, hs_info = masked_inner(d, lp_hs, hs_usable)
    E_p1, p1_info = masked_inner(d, lp, covered, scale=SCALE)
    E_p1x1, _ = masked_inner(d, lp, covered, scale=1.0)
    combos = {"Q02": (Q02_TERMS, {}), "Q02H": (Q02H_TERMS, {"HS": E_hs}),
              "Q08": (Q08_TERMS, {"HS": E_hs, "P1TO": E_p1}),
              "Q08_x1": (Q08_TERMS, {"HS": E_hs, "P1TO": E_p1x1}),
              "Q02+P1TO_x4": (Q02_TERMS + ["P1TO"], {"P1TO": E_p1}),
              "R20+P1TO_x4": (["OP", "P1TO"], {"P1TO": E_p1})}
    p1only = None
    if use_p2:
        # record only: Q08 with the mapping of the P1 probe alone (the effect of P2)
        X, us = F.variant_X(st["p1_probe"], "TO")
        pr1 = probe_rows([("p1_probe", X, us, st["p1_probe"]["ref"], sets1["probe"]["games"],
                           dup.get("p1_probe") or [])])
        del X
        mp1 = _fit_p1_map(PF, U, F, pr1)
        lp1, _cov1 = inner_p1_lp(F, st, mp1["logp"], n_inner)
        E_p1only, _ = masked_inner(d, lp1, covered, scale=SCALE)
        combos["Q08_P1probe_map"] = (Q08_TERMS, {"HS": E_hs, "P1TO": E_p1only})
        p1only = {"lambda": mp1["lambda"], "cv": mp1["cv"][str(mp1["lambda"])], "refit_loss": mp1["model"]["loss"],
                  "games": pr1["record"]["total"]}
    # 4 OOF
    res = {}
    for cid, (terms, Emap) in combos.items():
        t = time.time()
        P, Pm, folds, aux = run_terms(d, terms, Emap)
        if np.isnan(Pm).any():
            raise P08Error(f"{cid}: a single-colour question without an OOF fold")
        m, sc = rf.metrics(rf.rule_b_np(P), Xe.y, Xe.n, Xe.player)
        mrec, msc = p3.mono_score_record(rf.rule_b_np(Pm), Xm, ms, mboot)
        for fr in folds:
            fr["params"] = p4.gen_params(fr["theta"], terms)
        res[cid] = {"P": P, "Pm": Pm, "folds": folds, "terms": terms, "m": m, "sc": sc, "mrec": mrec, "msc": msc,
                    "aux": aux}
        print(f"{cid}: inner eval rank-balanced {m['rank_balanced']!r}; single-colour {mrec['rank_balanced']:.4f} "
              f"({time.time() - t:.0f}s)", flush=True)
    q, h, e = res["Q02"], res["Q02H"], res["Q08"]
    if abs(h["m"]["rank_balanced"] - P1REF["Q02H"]) > TOL:
        raise P08Error("Q02H differs from the selfcheck number")
    # 5 gates (Q08 vs Q02H)
    mono_diff = p3.mono_diff_record(ms, mboot, e["msc"], h["msc"])
    gates = gate_record(e["m"]["rank_balanced"], h["m"]["rank_balanced"], e["sc"], h["sc"], boot, mono_diff)
    rows = {}
    for cid, r in res.items():
        m = dict(r["m"])
        m.update(boot.summary(r["sc"], None if cid == "Q02H" else h["sc"]))
        for k in ("diff_vs_R00_se", "diff_vs_R00_ci95"):
            if k in m:
                m[k.replace("R00", "Q02H")] = m.pop(k)
        if cid != "Q02H":
            m["diff_vs_Q02H"] = m["rank_balanced"] - h["m"]["rank_balanced"]
            m["question_mean_diff_vs_Q02H"] = float(np.mean(r["sc"] - h["sc"]))
            m["per_level_diff_vs_Q02H"] = {lv: m["per_level"][lv] - h["m"]["per_level"][lv] for lv in m["per_level"]}
            m["single_colour_diff_vs_Q02H"] = p3.mono_diff_record(ms, mboot, r["msc"], h["msc"])
        rows[cid] = {"terms": r["terms"], "oof": m, "single_colour": p4.short_mono(r["mrec"]),
                     "folds": [{"fold": fr["fold"], "loss": fr["loss"], "params": fr["params"],
                                "bounds_hit": P7.bound_hits(fr["params"]["w"])} for fr in r["folds"]]}
    se = p3.se_val_record(e["m"]["rank_balanced"] - h["m"]["rank_balanced"], e["sc"] - h["sc"], Xe.y)
    se["adoption_probability_at_fraction_of_delta"] = p4.adoption(se["delta"], se["se_val"])
    # 6 direction references: softmax(E_P1) expected level vs Q02's OOF expected level
    has_e = P7.questions_with_usable(inner["S_eval"], covered)
    has_m = P7.questions_with_usable(inner["S_mono"], covered)
    rho_eval, n_e = P7.hs_rho(E_p1["eval"], q["P"], has_e)
    rho_mono, n_m = P7.hs_rho(E_p1["mono"], q["Pm"], has_m)
    direction_ref = {"rho_eval": rho_eval, "eval_questions": n_e, "rho_mono": rho_mono, "mono_questions": n_m,
                     "floor_validation": RHO_FLOOR_VAL, "floor_test_per_set": RHO_FLOOR_TEST,
                     "validation_threshold": max(0.5 * rho_eval, RHO_FLOOR_VAL),
                     "test_threshold": max(0.5 * rho_mono, RHO_FLOOR_TEST),
                     "definition": "Spearman (average ranks) between sum_k k softmax(E_P1)_k (E_P1 = masked mean of "
                                   "4 lp_P1TO) and sum_k k P_Q02,k (Q02 OOF), questions with >= 1 usable game"}
    mono_unusable = int(sum(1 for g in ms["groups"] if not covered[np.asarray(g)].all()))
    # 7 refit (gates passed)
    theta, refit = p4.gen_fit_record(e["aux"]["mf"], e["aux"]["Ef"], Xf.n, Xf.y)
    params = p4.gen_params(theta, Q08_TERMS)
    Pfit = fuse_probs(theta, e["aux"]["mf"], e["aux"]["Ef"], Xf.n, np.ones(len(Xf), dtype=bool))
    mfit, _ = rf.metrics(rf.rule_b_np(Pfit), Xf.y, Xf.n, Xf.player)
    refit.update(fit_questions=len(Xf), params=params, bounds_hit=P7.bound_hits(params["w"]),
                 in_sample_fit_rank_balanced=mfit["rank_balanced"], in_sample_fit_question_mean=mfit["question_mean"],
                 oof_eval_rank_balanced=e["m"]["rank_balanced"])
    blocked = [] if gates["passed"] else [gates["reason"]]
    inputs = {"inner_manifest.json": man_sha, "check_report.json": cr_sha, "selfcheck.json": sr_sha,
              "pilot_plan.json": P7.PLAN_SHA, "inner_o4.npz": sha_raw(P7.INNER_O4),
              **{k: v for k, v in d["inner"]["inputs"].items()}, **d["sha"]}
    torch_vals = sorted({str(v) for v in (man.get("torch_values") or [])})
    fc07 = read_json(P07_FROZEN)
    ref07 = (fc07.get("environment") or {}).get("venv_gpu_reference") or {}
    environment = {"venv": environment_cpu(),
                   "venv_gpu_reference": {"python": ref07.get("python"),
                                          "torch": torch_vals[0] if len(torch_vals) == 1 else torch_vals,
                                          "cudnn": ref07.get("cudnn"), "gpu": ref07.get("gpu"),
                                          "numpy": ref07.get("numpy"),
                                          "sources": "torch: the P1 / P2 chunks' torch field (inner_manifest.json); "
                                                     "python / cuDNN / GPU / numpy: results/prereg07/rank/"
                                                     "frozen_config.json venv_gpu_reference (the same .venv-gpu)"},
                   "nvidia_driver_record_only": P7.nvidia_query()}
    report = {"format": FORMATS["tune"],
              "what": "PREREG_08 2.4-2.6 tune: Q02H vs Q08 on the inner pseudo-questions, 5-fold OOF, gates G1 G2 G4",
              "created": now_iso(), "seconds": round(time.time() - t0, 1), **mode_flags(c), "rerun": rerun,
              "combinations": rows, "gates": gates, "score_allowed_at_tune": not blocked,
              "score_blocked_reasons": blocked, "score_pending_val": bool(gates["passed"]),
              "record_only": {"Q08_x1": rows["Q08_x1"]["oof"]["rank_balanced"],
                              "Q02+P1TO_x4": rows["Q02+P1TO_x4"]["oof"]["rank_balanced"],
                              "R20+P1TO_x4": rows["R20+P1TO_x4"]["oof"]["rank_balanced"],
                              "Q08_P1probe_map": (rows["Q08_P1probe_map"]["oof"]["rank_balanced"]
                                                  if "Q08_P1probe_map" in rows else None),
                              "p1_only_mapping": p1only, "single_colour_Q08_minus_Q02H": mono_diff,
                              "mono_questions_with_unusable_p1_game": mono_unusable},
              "se_val": se, "p1_direction_ref": direction_ref, "refit": refit,
              "p1_map": {"lambda": mp["lambda"], "cv": mp["cv"], "refit_loss": mp["model"]["loss"],
                         "refit_success": mp["model"]["success"], "refit_nit": mp["model"]["nit"], **map_rec,
                         "roundtrip": roundtrip, "lambda_at_grid_end_record_only":
                             mp["lambda"] in (LAMBDAS[0], LAMBDAS[-1])},
              "masked_E": {"HS": hs_info, "P1TO": p1_info}, "hs_mapping": hs_rec,
              "questions": {"fit": len(Xf), "eval": len(Xe), "single_colour": len(Xm)},
              "inputs_sha256": inputs, "check": {"sha256": cr_sha, "commit": cr_commit},
              "selfcheck": {"sha256": sr_sha, "commit": sr_commit}, "inner_manifest_commit": man_commit,
              "pinned_sha256": check_pins(), "p07_frozen_sha256": check_p07_pins(), "tool": ts,
              "environment": environment}
    os.makedirs(c.out_dir, exist_ok=True)
    if not gates["passed"]:
        os.remove(tmp_map)
        write_json(os.path.join(c.out_dir, "tune_report.json"), report)
        print(f"gates NOT passed: {gates['reason']}; no frozen_config.json, no p1_map.npz (PREREG_08 2.5)")
        raise Stop("gates not passed")
    os.replace(tmp_map, map_path)
    model2 = load_p1_map(map_path)
    if not all(np.array_equal(model2[k], model[k]) for k in ("mu", "sd", "W", "b")):
        raise P08Error("p1_map.npz changed while moving it")
    map_sha = sha_raw(map_path)
    frozen = {
        "format": FORMATS["frozen"],
        "what": "PREREG_08 competition 2 frozen configuration Q08 = Q02H + 4 x P1TO (inner only; commit with "
                "p1_map.npz before p08_p1.py val and apply)",
        "created": report["created"], **mode_flags(c),
        "combo": {"id": "Q08", "sources": ["C5", "OP", "O4", "SWA", "SWB", "HS", "P1TO"], "terms": Q08_TERMS,
                  "path": "gen", "pool": "log", "calibrated": True, "opening": True, "p1_scale": SCALE},
        "params": params,
        "folds": [{"fold": fr["fold"], "loss": fr["loss"], "starts": fr["starts"], "params": fr["params"],
                   "bounds_hit": P7.bound_hits(fr["params"]["w"])} for fr in e["folds"]],
        "basis": [[float(x) for x in row] for row in p4.B],
        "formula": "p = softmax(e^a (n/12)^gamma (m_C5 + w_OP m_OP + w_O4 E_O4 + w_SWA E_SWA + w_SWB E_SWB + w_HS E_HS "
                   "+ w_P1 E_P1 + B beta)); E_P1 = mean over the question's usable games (C5 parsed, P1 not failed, "
                   ">= 1 target move) of 4 lp_P1TO, 0 without one; E_HS as PREREG_07; n = C5-parsed games",
        "decision_rule": "B, tools/decide.py (argmax p_k + e^-1 (p_{k-1} + p_{k+1}))",
        "p1": {"variant": "TO", "features": 2 * F_DIM, "scale": SCALE, "p1_map_npz_sha256": map_sha,
               "probe_used": probe_used, "probe_reason": (man.get("p2") or {}).get("reason"),
               "sets_sha256": map_rec["sets_sha256"], "games": map_rec["games"], "players": map_rec["players"],
               "lambda": mp["lambda"], "refit_loss": mp["model"]["loss"], "run": RUN,
               "checkpoint": {"path": rel(P7.CKPT), "bytes": CKPT_BYTES, "sha256": CKPT_SHA,
                              "weights": "model (not swa_model)"},
               "katago_commit": KATAGO_COMMIT, "source_txt_sha256": P7.SOURCE_TXT_SHA,
               "neutral_metadata_sha256": NEUTRAL_META_SHA, "inner_manifest_sha256": man_sha},
        "pinned_sha256": dict(PINNED), "data_pins_sha256": dict(DATA_PINS),
        "prereg07": {"frozen_config_sha256": P07_FROZEN_SHA, "hs_map_npz_sha256": P07_HS_MAP_SHA,
                     "rank_val_probs_sha256": P07_PROBS_SHA, "val_hs_sha256": P07_VAL_HS_SHA},
        "gates": gates, "p1_direction_ref": direction_ref, "se_val": se, "score_pending_val": True,
        "score_allowed_note": "gates passed; score_allowed is decided by apply (PREREG_08 2.7) after the validation P1",
        "oof": {k: e["m"][k] for k in ("rank_balanced", "question_mean", "per_level", "per_n_bin",
                                       "pred_distribution")},
        "Q02H_oof_rank_balanced": h["m"]["rank_balanced"], "Q02_oof_rank_balanced": q["m"]["rank_balanced"],
        "single_colour": {"questions_sha256": ms["sha256"], "Q08": p4.short_mono(e["mrec"]),
                          "Q02H": p4.short_mono(h["mrec"]), "Q08_minus_Q02H": mono_diff},
        "refit": {k: refit[k] for k in ("loss", "starts", "fit_questions", "bounds_hit", "in_sample_fit_rank_balanced",
                                        "oof_eval_rank_balanced")},
        "check": {"check_report_sha256": cr_sha, "selfcheck_sha256": sr_sha, "inner_manifest_sha256": man_sha},
        "inputs_sha256": inputs, "tool": ts, "environment": environment, "tool_files_at_tune": ts["files_sha256"]}
    write_json(os.path.join(c.out_dir, "tune_report.json"), report)
    write_json(frozen_path, frozen)
    g = gates
    print(f"Q08 vs Q02H: G1 {g['G1']['delta']!r} G2 {g['G2']['ci95']} G4 {g['G4']['rank_balanced_difference']!r} "
          f"{g['G4']['ci95']} B/W {g['G4']['by_colour_point_estimates']} -> {'PASSED' if g['passed'] else 'not passed'}")
    print(f"rho_eval {rho_eval:.4f}, rho_mono {rho_mono:.4f}; probe {probe_used}, lambda {mp['lambda']}")
    print(f"wrote tune_report.json, frozen_config.json, p1_map.npz in {c.out_dir}"
          f"{' (PREVIEW)' if c.preview else '; commit frozen_config.json + p1_map.npz before p08_p1.py val'}")
    return 0


# ================================================================ apply (PREREG_08 2.7; spec D)
def design_records_bound(fc):
    """val / apply: inner_manifest.json, check_report.json, selfcheck.json, tune_report.json committed and clean and
    equal to the SHA-256s frozen_config.json records."""
    paths = [MANIFEST] + [os.path.join(OUT, n) for n in ("check_report.json", "selfcheck.json", "tune_report.json")]
    rec = require_committed(paths, "inner_manifest.json / check_report.json / selfcheck.json / tune_report.json")
    want = {MANIFEST: fc["check"]["inner_manifest_sha256"],
            os.path.join(OUT, "check_report.json"): fc["check"]["check_report_sha256"],
            os.path.join(OUT, "selfcheck.json"): fc["check"]["selfcheck_sha256"]}
    bad = [rel(p) for p, sha in want.items() if sha_lf(p) != sha]
    if bad:
        raise P08Error(f"design records differ from the SHA-256s frozen_config.json records: {bad}")
    return rec


def frozen_prerequisites():
    """frozen_config.json and p1_map.npz committed, usable, bound; tools equal tune's. -> (fc, record)."""
    fpath = os.path.join(OUT, "frozen_config.json")
    mpath = os.path.join(OUT, "p1_map.npz")
    rec = {"frozen_config": require_committed([fpath, mpath], "frozen_config.json / p1_map.npz (commit them first)")}
    fc = read_json(fpath)
    frozen_usable(fc)
    if sha_raw(mpath) != fc["p1"]["p1_map_npz_sha256"]:
        raise P08Error("p1_map.npz differs from the SHA-256 frozen_config.json records")
    rec["design_records"] = design_records_bound(fc)
    ts = tool_state()
    require_tools(ts, False)
    diff = tools_equal_record(ts, fc.get("tool"))
    if diff:
        raise P08Error(f"tool files differ from the ones tune recorded: {diff}")
    rec["tool"] = ts
    rec["frozen_commit_time"] = commit_time(fpath)
    return fc, rec


def apply_prerequisites():
    """-> (frozen config, val report, records). Real mode only; never reads data/local."""
    fc, rec = frozen_prerequisites()
    fpath = os.path.join(OUT, "frozen_config.json")
    feat = [os.path.join(FEAT, n) for n in ("val_p1.npz", "val_seam_feats.npz", "val_report.json")]
    rec["val_p1"] = require_committed(feat, "val_p1.npz / val_seam_feats.npz / val_report.json")
    vr = read_json(feat[2])
    if vr.get("format") != FORMATS["val_report"] or vr.get("mode") != "real" or vr.get("status") not in ("ok", "stop"):
        raise P08Error("val_report.json is not a real p08_p1.py val report")
    ft = rec["frozen_commit_time"]
    first = vr.get("first_chunk_started_utc")
    if not ft or not first or datetime.datetime.fromisoformat(first) <= datetime.datetime.fromisoformat(ft):
        raise P08Error(f"val_report.json start {first} is not later than frozen_config.json's commit {ft}")
    if vr.get("frozen_config_sha256") != sha_lf(fpath):
        raise P08Error("val_report.json names another frozen_config.json")
    bad = []
    for ch in vr.get("chunks") or []:
        p = path_of(ch["file"])
        if not os.path.exists(p) or sha_raw(p) != ch["sha256"]:
            bad.append(ch["file"])
    if bad or len(vr.get("chunks") or []) != (VAL_GAMES + RUN["chunk"] - 1) // RUN["chunk"]:
        raise P08Error(f"validation P1 chunks missing or changed: {bad[:5]}")
    if sha_raw(feat[0]) != vr.get("val_p1_sha256") or sha_raw(feat[1]) != vr.get("val_seam_feats_sha256"):
        raise P08Error("val_p1.npz / val_seam_feats.npz differ from val_report.json")
    rec["p07_pins"] = check_p07_pins()
    rec.update(val_first_chunk_started=first)
    return fc, vr, rec


def read_val_chunks(vr, vk):
    """The validation P1 chunks val_report.json lists (SHA-256 checked) -> per-game arrays in val_keys order."""
    parts = []
    for ch in vr["chunks"]:
        p = path_of(ch["file"])
        if sha_raw(p) != ch["sha256"]:
            raise P08Error(f"{ch['file']} changed since val_report.json")
        with np.load(p, allow_pickle=False) as z:
            if str(z["format"]) != FORMATS["val_chunk"] or str(z["meta_sha256"]) != NEUTRAL_META_SHA:
                raise P08Error(f"{ch['file']}: not a {FORMATS['val_chunk']} chunk with the neutral metadata")
            if str(z["mode"]) != "real" or str(z["keys_sha256"]) != P7.VAL_KEYS_SHA:
                raise P08Error(f"{ch['file']}: not a real-mode chunk of the pinned val_keys")
            parts.append({k: np.asarray(z[k]) for k in ("ref", "means", "counts", "n_moves", "failed")})
    ref = np.concatenate([x["ref"] for x in parts])
    if not np.array_equal(ref, np.arange(vk["n"])):
        raise P08Error("validation chunk refs are not the val_keys rows in order")
    return {"ref": ref, "means": np.concatenate([x["means"] for x in parts]).astype(np.float64),
            "counts": np.concatenate([x["counts"] for x in parts]).astype(np.int64),
            "n_moves": np.concatenate([x["n_moves"] for x in parts]).astype(np.int64),
            "failed": np.concatenate([x["failed"] for x in parts]).astype(bool)}


def val_cross_check(vp, vh, o4z, ids):
    """PREREG_08 2.7: P1 n_moves = val_hs n_moves = val_o4 moves_parsed; P1 target moves = val_hs counts =
    val_o4 n_moves[:, 0, :] total; on the games where neither side failed."""
    a = P7.move_cross_check(vp["n_moves"], vp["counts"][:, 0], vp["failed"], vh["n_moves"], vh["counts"],
                            vh["failed"], ids)
    b = P7.move_cross_check(vp["n_moves"], vp["counts"][:, 0], vp["failed"], o4z["moves_parsed"],
                            np.asarray(o4z["n_moves"])[:, 0, :].sum(1), o4z["replay_failed"], ids)
    return {"vs_val_hs": a, "vs_val_o4": b, "mismatches": int(a["mismatches"] + b["mismatches"])}


def q08_core(F, q, core, vp, vh, o4z, model, fc):
    """Q08 on the validation questions from q02_validation()'s inputs, apply_core's Q02H (core), the validation P1
    arrays (vp: means, counts, n_moves, failed in val_keys order), val_hs.npz, val_o4.npz and the frozen map; masked
    E_P1 of 4 lp_P1TO, the move cross-check, the direction check and the failures. No file is written; nothing under
    data/local is read. -> dict."""
    vk, S, n, ok, m, E04, P_q02 = q["vk"], q["S"], q["n"], q["ok"], q["m"], q["E04"], q["P"]
    if len(vp["failed"]) != vk["n"]:
        raise P08Error("validation P1 arrays do not cover val_keys")
    if ([str(x) for x in o4z["question_id"]] != vk["question_id"] or [int(x) for x in o4z["slot"]] != vk["slot"]
            or [str(x) for x in o4z["color"]] != vk["color"]):
        raise P08Error("val_o4.npz keys differ from val_keys.npz")
    X, us = F.variant_X(vp, "TO")
    lp = np.zeros((vk["n"], L))
    if us.any():
        lp[us] = p1_lp_from_map(F, model, X[us])
    del X
    usable = np.asarray(vk["parsed"], dtype=bool) & us
    E_p1, e_info = p1_term(S, lp, usable)
    Es = [E04[t] for t in Q02_TERMS] + [core["E_hs"], E_p1]
    P = fuse_probs(fc["params"]["theta"], m, Es, n, ok)
    ids = [f"{qq}#{s}" for qq, s in zip(vk["question_id"], vk["slot"])]
    cross = val_cross_check(vp, vh, o4z, ids)
    has = P7.questions_with_usable(S, usable) & ok
    rho, nq = P7.hs_rho(E_p1, P_q02, has)
    direction = P7.direction_record(rho, nq, fc["p1_direction_ref"]["rho_eval"], RHO_FLOOR_VAL)
    fail_count = int(P7.hs_failures(vp["failed"], vp["n_moves"]).sum())
    return {"P": P, "lp": lp, "p1_ok": us, "usable": usable, "E_p1": E_p1, "e_info": e_info, "has": has,
            "cross": cross, "direction": direction, "fail_count": fail_count,
            "csv_bytes": P7.probs_csv_bytes(vk["groups"], P, ok, n)}


def cmd_apply(a):
    t0 = time.time()
    out_files = ["rank_val_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "check_submission.txt",
                 "val_lp_p1.npz", "apply_report.json"]
    if a.finalize:
        return apply_finalize(a, out_files)
    blas_env_check(False)
    fc, vr, pre = apply_prerequisites()
    c = Ctx()
    c.out_dir, c.preview, c.precommit = OUT, False, False
    rerun = guard_outputs(c, a, out_files)
    # ---- 1 Q02 through prereg04's readers, 2 the question file (both before p1_lib / pilot_fuse / hs_lib)
    q = P7.q02_validation()
    if not q["seam"]["passed"]:
        raise Stop(f"Q02 does not reproduce results/prereg04/rank/rank_val_probs.csv byte for byte ({q['seam']}); "
                   f"stop and find out why (PREREG_08 2.7); nothing was written")
    q_sha = P7.check_question_file(q["vk"])
    vk, ok = q["vk"], q["ok"]
    # ---- 3 from here on no data/local file is opened (pilot_fuse imports hs_lib and its guard)
    PF = import_pilot_fuse()
    with np.load(P07_VAL_HS, allow_pickle=False) as z:
        vh = {k: z[k] for k in z.files}
    model07 = P7.load_hs_map(P07_HS_MAP)
    fc07 = read_json(P07_FROZEN)
    core = P7.apply_core(PF, q, vh, model07, fc07)                       # 4 Q02H (07 untouched)
    with open(P07_PROBS, "rb") as f:
        ref07 = f.read().replace(b"\r\n", b"\n")
    q02h = {"computed_sha256": hashlib.sha256(core["csv_bytes"]).hexdigest(), "registered_sha256": P07_PROBS_SHA,
            "bytes_equal_committed": core["csv_bytes"] == ref07}
    q02h["passed"] = bool(q02h["bytes_equal_committed"] and q02h["computed_sha256"] == P07_PROBS_SHA)
    if not q02h["passed"]:
        raise Stop(f"Q02H does not reproduce results/prereg07/rank/rank_val_probs.csv byte for byte ({q02h}); stop "
                   f"and find out why (PREREG_08 2.7); nothing was written")
    # ---- 5 validation P1 features -> lp_P1TO with the frozen map
    _L1, F = import_p1()
    vp = read_val_chunks(vr, vk)
    with np.load(os.path.join(FEAT, "val_p1.npz"), allow_pickle=False) as z:
        v1 = {k: z[k] for k in z.files}
    if ([str(x) for x in v1["question_id"]] != vk["question_id"] or [int(x) for x in v1["slot"]] != vk["slot"]
            or [str(x) for x in v1["color"]] != vk["color"]):
        raise P08Error("val_p1.npz keys differ from val_keys.npz")
    for k in ("counts", "n_moves", "failed"):
        if not np.array_equal(np.asarray(v1[k]).astype(vp[k].dtype), vp[k]):
            raise P08Error(f"val_p1.npz {k} differs from the chunks")
    model = load_p1_map(os.path.join(OUT, "p1_map.npz"))
    if model["scale"] != SCALE or fc["p1"]["scale"] != SCALE:
        raise P08Error("p1_map.npz / frozen_config.json scale is not 4")
    with np.load(P7.VAL_O4, allow_pickle=False) as z:
        o4z = {k: z[k] for k in z.files}
    # ---- 6 masked E_P1 (x 4, once), Q08 with the frozen theta; cross-check, direction, failures
    qc = q08_core(F, q, core, vp, vh, o4z, model, fc)
    lp, us, usable, e_info, P = qc["lp"], qc["p1_ok"], qc["usable"], qc["e_info"], qc["P"]
    cross, direction, fail_count, has = qc["cross"], qc["direction"], qc["fail_count"], qc["has"]
    rho = direction["rho"]
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, "rank_val_probs.csv")
    with open(out_csv + ".tmp", "wb") as f:
        f.write(qc["csv_bytes"])
    os.replace(out_csv + ".tmp", out_csv)
    lp_path = os.path.join(OUT, "val_lp_p1.npz")
    np.savez(lp_path + ".tmp.npz", format=np.array(FORMATS["val_lp"]), question_id=np.array(vk["question_id"]),
             slot=np.array(vk["slot"], dtype=np.int64), color=np.array(vk["color"]), lp=lp, p1_ok=us,
             usable=usable, scale=np.array(SCALE), p1_map_sha256=np.array(sha_raw(os.path.join(OUT, "p1_map.npz"))))
    os.replace(lp_path + ".tmp.npz", lp_path)
    import decide
    decide.read_probs(out_csv)
    # ---- 7 decide.py and check_submission.py (subprocesses; this process never reads data/local again)
    py = sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r1 = subprocess.run([py, os.path.join(HERE, "decide.py"), out_csv, OUT], capture_output=True, text=True,
                        cwd=REPO, env=env)
    if r1.returncode != 0:
        raise P08Error(f"decide.py failed: {r1.stderr[-2000:]}")
    txt, formats_ok = [], True
    for pred in ("pred_A.csv", "pred_B.csv"):
        r2 = subprocess.run([py, os.path.join(HERE, "check_submission.py"), "rank", os.path.join(OUT, pred),
                             "--questions", P7.QUESTIONS_REL], capture_output=True, text=True, cwd=REPO, env=env)
        txt.append(f"$ check_submission.py rank results/prereg08/rank/{pred} --questions data/local/"
                   f"rank_val_questions.csv\n(exit {r2.returncode})\n{r2.stdout}{r2.stderr}")
        formats_ok = formats_ok and r2.returncode == 0
    with open(os.path.join(OUT, "check_submission.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(txt))
    audit, audit_rec = audit_state()
    allowed, reasons = apply_decision(fail_count, getattr(a, "accept_data_failures", None), q["seam"]["passed"],
                                      q02h["passed"], cross, direction, audit, formats_ok)
    pred_h = rf.rule_b_np(np.where(ok[:, None], core["P"], 0.0))
    pred_8 = rf.rule_b_np(np.where(ok[:, None], P, 0.0))
    rep = {"format": FORMATS["apply"], "what": "PREREG_08 2.7 apply: Q08 on the validation questions (no score)",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "mode": "real", "rerun": rerun,
           "prerequisites": pre, "q02_reproduction": q["seam"], "q02h_reproduction": q02h,
           "frozen_tools_p04": q["frozen_tools"], "score_allowed": allowed, "score_blocked_reasons": reasons,
           "accept_data_failures": getattr(a, "accept_data_failures", None),
           "p1_failures": {"failed_or_no_move": fail_count, "failed": int(vp["failed"].sum()),
                           "no_target_move_not_failed": int(((vp["counts"][:, 0] == 0) & ~vp["failed"]).sum()),
                           "unusable_games": int((~usable).sum()), "limit": VAL_FAIL_MAX},
           "move_cross_check": cross, "p1_direction": direction, "masked_E_P1": e_info, "audit": audit_rec,
           "questions": len(vk["groups"]), "questions_without_parsed_game": int((~ok).sum()),
           "questions_without_usable_p1_game": int((ok & ~has).sum()),
           "rule_B_differs_from_Q02H": int(((pred_8 != pred_h) & ok).sum()),
           "pred_distribution": dict(zip(LEVELS, np.bincount(pred_8[ok], minlength=L).tolist())),
           "lp_p1_summary": {"mean": np.mean(lp[usable], axis=0) if usable.any() else None},
           "se_val_record_from_tune": fc.get("se_val"),
           "inputs_sha256": {"frozen_config.json": sha_lf(os.path.join(OUT, "frozen_config.json")),
                             "p1_map.npz": sha_raw(os.path.join(OUT, "p1_map.npz")),
                             "val_p1.npz": sha_raw(os.path.join(FEAT, "val_p1.npz")),
                             "val_report.json": sha_lf(os.path.join(FEAT, "val_report.json")),
                             "val_hs.npz": sha_raw(P07_VAL_HS), "hs_map.npz (07)": sha_raw(P07_HS_MAP),
                             "frozen_config.json (07)": sha_lf(P07_FROZEN), "questions": q_sha,
                             "val_keys.npz": vk["sha256"], "val_o4.npz": sha_raw(P7.VAL_O4), **q["vsha"]},
           "outputs_sha256": {n_: pin_sha(os.path.join(OUT, n_)) for n_ in out_files[:6]},
           "decide_stdout": r1.stdout[-2000:], "tool": tool_state(), "environment": environment_cpu(),
           "next": "commit; prescore_audit.json by another agent (no scores) -> commit; apply --finalize; H8R "
                   "(PREREG_08 3) only when score_allowed is true"}
    write_json(os.path.join(OUT, "apply_report.json"), rep)
    print(f"apply Q08: Q02 / Q02H reproduction {q['seam']['passed']} / {q02h['passed']}; cross-check mismatches "
          f"{cross['mismatches']}; rho {rho:.4f} (threshold {direction['threshold']:.4f}); failed {fail_count}; "
          f"formats {formats_ok}")
    print(f"score_allowed: {allowed} {reasons}")
    return 0


def apply_finalize(a, out_files):
    rp = os.path.join(OUT, "apply_report.json")
    if require_committed([rp] + [os.path.join(OUT, f) for f in out_files[:6]], "apply outputs")["commit_problems"]:
        raise P08Error("commit apply's outputs first")
    rep = read_json(rp)
    for n_, sha in rep["outputs_sha256"].items():
        if pin_sha(os.path.join(OUT, n_)) != sha:
            raise P08Error(f"{n_} changed since apply")
    audit, audit_rec = audit_state(rep)
    allowed, reasons = finalize_decision(rep, audit, audit_rec)
    rep["score_allowed"], rep["score_blocked_reasons"] = allowed, reasons
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
    except Exception as e:  # noqa: BLE001 - p1_lib / hs_lib / u1_lib refusals are refusals too
        if type(e).__name__ in ("P1Error", "HSError", "LocalDataRefused"):
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
