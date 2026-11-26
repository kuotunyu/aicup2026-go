"""C10 competition 2: the local side of the registered Colab GPU fallback for 07 (HS) and 08 (P1)
(docs/PREREG_11_GPU_FALLBACK.md; Claude's steps docs/TESTDAY_GPU_FALLBACK.md; the user's Colab steps
docs/COLAB_GPU_FALLBACK.md). The Colab side is colab_gpu/gpu_fallback.ipynb, which runs colab_gpu/gpu_fallback_core.py.

    PY tools/g11_gpu_fallback.py kit                                   -> data/test11/kit/C10_gpu_kit_v1.zip (+ tree/),
                                                                          results/test11/kit_manifest.json
    PY tools/g11_gpu_fallback.py input --mode rehearsal                -> data/test11/upload/C10_gpu_input_rehearsal.zip,
                                                                          results/test11/inputs/rehearsal.json
    PY tools/g11_gpu_fallback.py input --mode real --stage hs|p1       (test day, after the trigger is committed)
    PY tools/g11_gpu_fallback.py input --mode dry-run --stage hs|p1 [TREE DIRS] --out-dir D --record-dir R   (drills)
    PY tools/g11_gpu_fallback.py input --mode smoke --out-dir D --record-dir R   (fit-player fixture games; CPU smoke)
    PY tools/g11_gpu_fallback.py trigger --stage hs|p1 --reason CODE --evidence TEXT   -> results/test11/trigger_<stage>.json
    PY tools/g11_gpu_fallback.py import --stage hs|p1 --set seam|public|private --from COLAB_RESULTS [MODE] [DIRS]
    PY tools/g11_gpu_fallback.py rehearsal-compare --from COLAB_RESULTS --work-dir OUTSIDE [--json OUT]
    PY tools/g11_gpu_fallback.py fake-from-val --stage hs|p1 --input I --out O    (drills: the validation chunks of the
                                                                                    same games in the Colab format)
    PY tools/g11_gpu_fallback.py status
    (PY = .venv/Scripts/python.exe; Git Bash; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL thread variables)

kit      The pinned KataGo code (third_party/katago), hs_lib / pilot_run / pilot_fuse, p1_lib / p1_extract / p1_fuse,
         u1_lib (its SHA-256 is part of the P1 code record), the checkpoint (full SHA-256) and
         colab_gpu/gpu_fallback_core.py, with KIT_MANIFEST.json: every file's raw and LF SHA-256 and the registered
         constants copied from tools/p07_rank_fuse.py, tools/p08_rank_fuse.py, tools/t07_rank.py, tools/t08_rank.py.
         The zip is deterministic (sorted members, fixed time stamps).
input    The games and keys of the sets as JSON (the same bytes the importer rebuilds later). rehearsal: validation
         rows 0-399 (seam, the validation chunking) and 4,787-5,786 (offgrid: the first 1,000 games of the fake private
         set, off the 200-game grid), read from the registered validation question file (SHA-256 checked; question
         content and colour only) and checked row by row against val_keys.npz; runs hs and p1 on both. real / dry-run:
         seam / public / private through tools/t07_rank.py set_slots after tools/t07_rank.py prereqs (hs) or
         tools/t08_rank.py prereqs + hs_keys_match (p1); real needs the stage's committed trigger.
trigger  The objective condition (PREREG_11 2): hardware_failure, gpu_exit2, venv_gpu_unrestorable or deadline_11_14
         (refused before 2026-11-14 23:59:59 +08:00). Commit the file before any Colab run of the test day.
import   One set of one stage from the downloaded Colab results: the run summary and every chunk's SHA-256, the
         input (rebuilt now through set_slots: same bytes), every chunk through the unchanged t07_rank.chunk_check /
         t08_rank.chunk_check plus the fields they do not read (set label, mode, settings, KataGo commit, hs_lib /
         P1 code, checkpoint size, created_utc), one environment for the whole stage (the seam's), real mode: the
         environment passed a committed rehearsal (results/test11/rehearsal_*.json), the trigger was committed before
         the first chunk, every chunk is later than R20's first upload, one Colab attempt (run_id) per stage. Then the
         chunks are copied to data/test07/hs/<set>/ (data/test08/p1/<set>/) and the unchanged t07_rank.assemble /
         t08_rank.assemble write <set>_hs.npz / hs_<set>_report.json (<set>_p1.npz / p1_<set>_report.json) exactly as
         `hs` / `p1` would, followed by results/test11/import_<stage>_<set>.json. Then apply runs unchanged.
rehearsal-compare  The Colab rehearsal chunks against the 4090 validation chunks of the same games, with the comparison
         functions of the test-day tools imported unchanged: t07_rank.gpu_seam_record and t08_rank.gpu_seam_record /
         t09_rank.gpu_seam_record (the registered seams, same games and chunking), t07_fake_tests.compare,
         t08_fake_tests.compare and t09_fake_tests.compare (offgrid). Verdict per stage: allowed when every seam passes
         its tolerance and 2.5 x every off-grid difference is within the current tolerance (PREREG_11 3).
Exit codes: 0 done; 2 refused; 3 the registered 'not used' (from the test-day prerequisites; written down as hs / p1
do); 4 environment drift or a seam / format mismatch (stop and find out why).

Discipline: never imports hs_lib, u1_lib, pilot_*, p1_* or torch (the Colab core does, on Colab); the validation
question file is read only by `input --mode rehearsal` and, through tools/t07_rank.py set_slots, by input / import
of the seam (the registered GPU seam); no label file is ever opened; tools/score.py is never run; no file under
results/prereg0*, results/test0[2-9]/design or colab/ is written.
"""

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SELF = os.path.join(HERE, "g11_gpu_fallback.py")
CORE_DIR = os.path.join(REPO, "colab_gpu")
for _p in (HERE, CORE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gpu_fallback_core as core  # noqa: E402
import t07_rank as t7  # noqa: E402
import t08_rank as t8  # noqa: E402

pf7, r8, t2, po, rf = t7.pf7, t8.r8, t7.t2, t7.po, t7.rf

# ================================================================ paths
RESULTS = os.path.join(REPO, "results")
REAL_R10 = os.path.join(RESULTS, "test11")
DATA10 = os.path.join(REPO, "data", "test11")
KIT_DIR = os.path.join(DATA10, "kit")
UPLOAD_DIR = os.path.join(DATA10, "upload")
UPLOAD_OLD = os.path.join(DATA10, "upload_old")
INPUT_TREES = os.path.join(DATA10, "inputs")
P10_DOC = os.path.join(REPO, "docs", "PREREG_11_GPU_FALLBACK.md")
CORE = os.path.join(CORE_DIR, "gpu_fallback_core.py")
VAL_HS = os.path.join(RESULTS, "prereg07", "hs", "val_hs.npz")
VAL_REPORT7 = os.path.join(RESULTS, "prereg07", "hs", "val_report.json")
VAL_P1 = os.path.join(RESULTS, "prereg08", "feat", "val_p1.npz")
VAL_REPORT8 = os.path.join(RESULTS, "prereg08", "feat", "val_report.json")
FIXTURE = os.path.join(REPO, "data", "prereg07", "fixtures", "fit_sample.json")

# ================================================================ registered values (PREREG_11)
KIT_NAME = "C10_gpu_kit_v1.zip"
KIT_ROOT = "c10kit"
INPUT_ROOT = "c10input"
STAGES = ("hs", "p1")
SETS = ("seam", "public", "private")
REHEARSAL_ROWS = {"seam": (0, 400), "offgrid": (4787, 5787)}    # validation rows (val_keys order), PREREG_11 3
OFFGRID_FIRST_QUESTION = 400                                      # row 4,787 = the first game of question 401
TRIGGERS = ("hardware_failure", "gpu_exit2", "venv_gpu_unrestorable", "deadline_11_14")
DEADLINE = dt.datetime(2026, 11, 14, 23, 59, 59, tzinfo=t7.TW)
REHEARSAL_CUTOFF = dt.datetime(2026, 11, 4, 11, 0, 0, tzinfo=t7.TW)   # PREREG_11 3.6: test data opens 11/04 11:00
FACTOR = 2.5                                                      # PREREG_11 3 rule 1: 2.5 x off-grid (record only since rev 2)
RULE_VERSION = 2                    # PREREG_11 rev 2: off-grid x 1.0 <= tolerance; rule-1 records are never allowed
FORMATS = {"kit_record": "c10-g11 kit record v1", "input_record": "c10-g11 input record v1",
           "trigger": "c10-g11 trigger v1", "import": "c10-g11 import v1", "rehearsal": "c10-g11 rehearsal compare v1",
           "set": "c10-g11 set v1"}
_LABEL = "ans" + "wer"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
STAGE_TOOLS = {"hs": "t07", "p1": "t08"}


class G11Error(RuntimeError):
    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


def refuse(msg):
    raise G11Error(msg, 2)


def drift(msg):
    raise G11Error(msg, 4)


# ================================================================ helpers
def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_iso(text, what):
    try:
        t = dt.datetime.fromisoformat(str(text).strip())
    except ValueError:
        refuse(f"{what} {text!r} is not an ISO time")
    return t.replace(tzinfo=t7.TW) if t.tzinfo is None else t


def no_label(path):
    if path is not None and _LABEL in os.path.abspath(os.fspath(path)).lower():
        refuse(f"refusing {path}: validation label files are never opened")
    return path


def sha_raw(path):
    return core.sha_raw(no_label(path))


def sha_lf(path):
    return core.sha_lf(no_label(path))


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def rel(path):
    return os.path.relpath(path, REPO).replace("\\", "/") if t2.is_inside(path, REPO) else os.path.abspath(path)


def read_json(path):
    no_label(path)
    if not os.path.exists(path):
        refuse(f"missing {rel(path)}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    no_label(path)
    p = os.path.abspath(path)
    for bad in ("prereg0", "test02", "test04", "test06", "test07", "test08", "test09"):
        if t2.is_inside(p, RESULTS) and os.path.relpath(p, RESULTS).replace("\\", "/").split("/")[0].startswith(bad):
            refuse(f"refusing to write {p}: g11 never writes results/{bad}* (the test-day tools do)")
    core.write_json_atomic(p, json.loads(json.dumps(obj, default=t7.jsonable)))
    return p


def load_npz(path):
    no_label(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def git(*args):
    return t7.git(*args)


def committed_clean(path):
    return t7.committed_clean(path)


def commit_time(path):
    return t7.commit_time(path)


def in_repo(path):
    """Inside the repository by its written path or by its real path (data/ may be a junction in a worktree)."""
    a = os.path.normcase(os.path.abspath(path))
    r = os.path.normcase(os.path.abspath(REPO))
    return a == r or a.startswith(r + os.sep) or t2.is_inside(path, REPO)


def outside_repo(path, what):
    if in_repo(path):
        refuse(f"{what} {path} must be outside the repository")
    return os.path.abspath(path)


def tool_record():
    return {"git_head": t7.git_head(), "g11_gpu_fallback.py": sha_lf(SELF), "gpu_fallback_core.py": sha_lf(CORE)}


# ================================================================ kit
def kit_files():
    """[(kit relative path, source path)] of the kit, sorted."""
    files = []
    for d, _dirs, fs in os.walk(os.path.join(REPO, "third_party", "katago")):
        for f in fs:
            if f.endswith((".pyc",)) or "__pycache__" in d:
                continue
            p = os.path.join(d, f)
            files.append((os.path.relpath(p, REPO).replace("\\", "/"), p))
    for r in ("results/prereg07/explore/pilot_humansl/hs_lib.py", "results/prereg07/explore/pilot_humansl/pilot_run.py",
              "results/prereg07/explore/pilot_humansl/pilot_fuse.py", "results/prereg08/p1/p1_lib.py",
              "results/prereg08/p1/p1_extract.py", "results/prereg08/p1/p1_fuse.py",
              "results/prereg05/explore/u1/u1_lib.py", core.SELF_REL, core.CKPT_REL):
        files.append((r, os.path.join(REPO, *r.split("/"))))
    return sorted(files)


def p1_code_record():
    """The P1 code record of the 4090 validation chunks (results/prereg08/feat/val_report.json chunk 0, SHA-256
    checked), each entry equal to the file on disk now (LF SHA-256). -> dict."""
    vr = read_json(VAL_REPORT8)
    ch = vr["chunks"][0]
    p = os.path.join(REPO, *ch["file"].split("/"))
    if sha_raw(p) != ch["sha256"]:
        refuse(f"{ch['file']} differs from val_report.json")
    with np.load(p, allow_pickle=False) as z:
        rec = json.loads(str(z["code_sha256"]))
    now = {k: sha_lf(os.path.join(REPO, *k.split("/"))) for k in rec}
    if now != rec:
        refuse(f"the P1 code on disk differs from the validation run's record: {[k for k in rec if now[k] != rec[k]]}")
    return rec


def kit_constants():
    return {"profiles": list(pf7.PROFILES), "hs_run": dict(pf7.RUN), "p1_run": dict(r8.RUN),
            "hs_chunk_format": t7.CHUNK_FORMAT, "p1_chunk_format": t8.FORMATS["chunk"],
            "ckpt_sha256": pf7.CKPT_SHA, "ckpt_bytes": pf7.CKPT_BYTES, "katago_commit": pf7.KATAGO_COMMIT,
            "hs_lib_sha256": pf7.PINNED["results/prereg07/explore/pilot_humansl/hs_lib.py"],
            "neutral_meta_sha256": r8.NEUTRAL_META_SHA, "p1_code_record": p1_code_record(),
            "sources": "tools/p07_rank_fuse.py (PROFILES, RUN, CKPT, PINNED), tools/p08_rank_fuse.py (RUN, "
                       "NEUTRAL_META_SHA), tools/t07_rank.py CHUNK_FORMAT, tools/t08_rank.py FORMATS['chunk']; the P1 "
                       "code record of the 4090 validation chunks"}


def build_kit_manifest():
    if pf7.CKPT_SHA != r8.CKPT_SHA or pf7.KATAGO_COMMIT != r8.KATAGO_COMMIT:
        refuse("07 and 08 pin different checkpoints / KataGo commits")
    try:
        pins = pf7.check_pins()
    except pf7.P07Error as e:
        refuse(str(e))
    probs, kg = pf7.katago_sources()
    if probs:
        refuse(f"third_party/katago: {probs}")
    c = kit_constants()
    pins_lf = dict(sorted({**pins, **c["p1_code_record"]}.items()))
    files = {}
    for r, src in kit_files():
        if not os.path.exists(src):
            refuse(f"kit file {r} missing")
        if r == core.CKPT_REL:
            files[r] = {"sha256": pf7.CKPT_SHA, "bytes": pf7.CKPT_BYTES, "hashed_at_build": True}
            continue
        files[r] = {"sha256": sha_raw(src), "sha256_lf": sha_lf(src), "bytes": os.path.getsize(src)}
    stable = {r: v.get("sha256_lf", v["sha256"]) for r, v in files.items()}     # line endings never change the id
    ident = sha_bytes(json.dumps({"files": stable, "constants": c}, sort_keys=True).encode("utf-8"))
    return {"format": core.KIT_FORMAT, "kit_id": f"g11-kit-v1-{ident[:12]}", "files": files, "pins_lf": pins_lf,
            "constants": c, "katago_sources": kg,
            "rule": "files: raw SHA-256 of the kit copy (the working-tree bytes) and LF SHA-256; pins_lf: the registered "
                    "LF SHA-256 (PREREG_07 1.2, PREREG_08 1.2) every Colab run re-checks"}


def zip_write(zf, arcname, src=None, data=None, stored=False):
    info = zipfile.ZipInfo(arcname, date_time=ZIP_TIME)
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    if data is not None:
        zf.writestr(info, data)
        return
    with open(src, "rb") as fi, zf.open(info, "w", force_zip64=True) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def cmd_kit(a):
    out = os.path.abspath(a.out_dir or KIT_DIR)
    rec_dir = os.path.abspath(a.record_dir or REAL_R10)
    M = build_kit_manifest()
    t0 = time.time()
    ck = sha_raw(os.path.join(REPO, *core.CKPT_REL.split("/")))
    if ck != pf7.CKPT_SHA:
        refuse(f"data/katago checkpoint SHA-256 {ck} is not {pf7.CKPT_SHA} (PREREG_07 1.2: use data/katago_backup)")
    mbytes = (json.dumps(M, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    tree = os.path.join(out, "tree")
    shutil.rmtree(tree, ignore_errors=True)
    for r, src in kit_files():
        dst = os.path.join(tree, *r.split("/"))
        if r == core.CKPT_REL:
            link_or_copy(src, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
    with open(os.path.join(tree, "KIT_MANIFEST.json"), "wb") as f:
        f.write(mbytes)
    zpath = os.path.join(os.path.abspath(a.zip_dir) if a.zip_dir else UPLOAD_DIR, KIT_NAME)
    if not a.no_zip:
        os.makedirs(os.path.dirname(zpath), exist_ok=True)
        tmp = zpath + ".tmp"
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
            zip_write(zf, f"{KIT_ROOT}/KIT_MANIFEST.json", data=mbytes)
            for r, src in kit_files():
                zip_write(zf, f"{KIT_ROOT}/{r}", src=src, stored=(r == core.CKPT_REL))
        os.replace(tmp, zpath)
    core.verify_kit(tree, json.loads(mbytes), ckpt_hash=False)
    rec = {"format": FORMATS["kit_record"], "kit_id": M["kit_id"], "created": now_iso(),
           "zip": None if a.no_zip else {"file": KIT_NAME, "sha256": sha_raw(zpath), "bytes": os.path.getsize(zpath)},
           "manifest_sha256": sha_bytes(mbytes), "manifest": M, "tool": tool_record(),
           "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(rec_dir, "kit_manifest.json"), rec)
    print(f"kit {M['kit_id']}: {len(M['files'])} files -> {zpath if not a.no_zip else tree}"
          + (f" ({rec['zip']['bytes'] / 1e6:.1f} MB, sha256 {rec['zip']['sha256'][:12]})" if rec["zip"] else ""))
    return 0


def current_kit(rec_dir):
    p = os.path.join(rec_dir, "kit_manifest.json")
    rec = read_json(p)
    M = build_kit_manifest()
    if rec["manifest"]["kit_id"] != M["kit_id"]:
        refuse(f"{rel(p)} records kit {rec['manifest']['kit_id']}, the files now give {M['kit_id']}: rebuild the kit")
    return rec


# ================================================================ inputs
def set_file_bytes(name, games, keys, keys_sha, questions_sha, source):
    """The JSON bytes of one input set (deterministic; the importer rebuilds them)."""
    d = {"format": FORMATS["set"], "set": name, "games": [{"sgf_content": g["sgf_content"], "color": g["color"],
                                                          "ref": int(g["ref"])} for g in games],
         "keys": {"question_id": [str(x) for x in keys["question_id"]], "slot": [int(x) for x in keys["slot"]],
                  "color": [str(x) for x in keys["color"]]},
         "keys_sha256": keys_sha, "questions_sha256": questions_sha, "source": source}
    return (json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def rehearsal_sets():
    """Validation rows of REHEARSAL_ROWS from the registered question file (SHA-256) checked against val_keys.npz."""
    qfile = t7.SEAM_QUESTIONS
    q_sha = sha_raw(qfile)
    if q_sha != pf7.QUESTIONS_SHA:
        refuse(f"the validation question file is not the registered one ({q_sha[:12]})")
    vk = rf.load_val_keys(os.path.dirname(pf7.VAL_KEYS))
    if vk["sha256"] != pf7.VAL_KEYS_SHA:
        refuse("val_keys.npz is not the registered file")
    slots = po.read_question_slots(qfile)
    po.slots_match_keys(slots, vk)
    first = {}
    for i, q in enumerate(vk["question_id"]):
        first.setdefault(q, i)
    starts = sorted(first.values())
    if starts[OFFGRID_FIRST_QUESTION] != REHEARSAL_ROWS["offgrid"][0]:
        refuse(f"question 401 starts at row {starts[OFFGRID_FIRST_QUESTION]}, registered {REHEARSAL_ROWS['offgrid'][0]}")
    out = {}
    for name, (r0, r1) in REHEARSAL_ROWS.items():
        games = [{"sgf_content": slots[r][3], "color": slots[r][2], "ref": i} for i, r in enumerate(range(r0, r1))]
        keys = {"question_id": vk["question_id"][r0:r1], "slot": vk["slot"][r0:r1], "color": vk["color"][r0:r1]}
        out[name] = (games, keys, vk["sha256"], q_sha, {"validation_rows": [r0, r1], "file": "the registered "
                                                          "validation question file (pf7.QUESTIONS_SHA)"})
    return out


def smoke_sets(n=6):
    d = read_json(FIXTURE)
    games = [{"sgf_content": g["sgf_content"], "color": g["color"], "ref": i} for i, g in enumerate(d["games"][:n])]
    keys = {"question_id": [f"fit_{g['game_id']}" for g in d["games"][:n]], "slot": [1] * n,
            "color": [g["color"] for g in d["games"][:n]]}
    s = sha_raw(FIXTURE)
    return {"smoke": (games, keys, s, s, {"fixture": "data/prereg07/fixtures/fit_sample.json (fit players only)"})}


def ns_for(stage, a, fake=False):
    """An argparse namespace with the fields tools/t07_rank.py / t08_rank.py resolve and prereqs read."""
    g = lambda k: getattr(a, k, None)  # noqa: E731
    ns = argparse.Namespace(dry_run=g("mode") == "dry-run" or bool(g("dry_run")), rehearsal=False,
                            results_dir=g("results_dir"), t04_results_dir=g("t04_results_dir"),
                            t02_results_dir=g("t02_results_dir"), t02_out_dir=g("t02_out_dir"),
                            tests_dir=g("tests_dir"), checks_dir=g("checks_dir"), r20_uploaded_at=g("r20_uploaded_at"),
                            max_games=g("max_games") or 0, fake_scorer=fake, ignore_other_gpu_jobs=False, now=g("now"),
                            set=g("set"), rerun=False, reason=None, workers=1, stop_after_min=0)
    if stage == "hs":
        ns.hs_data_dir = g("data_dir")
    else:
        ns.p1_data_dir = g("data_dir")
        ns.t07_results_dir = g("t07_results_dir")
    return ns


def stage_ctx(stage, ns):
    """-> (ctx, prereqs record, extra) through the unchanged test-day prerequisites of the stage."""
    if stage == "hs":
        ctx = t7.resolve(ns, "hs")
        fc07, _fc04, t02, _r4, pre = t7.prereqs(ctx, "hs", ns)
        return ctx, pre, {"fc": fc07, "t02": t02}
    ctx = t8.resolve(ns, "p1")
    _fc07, fc08, _m8, _fc04, t02, _rep7, _r4, pre = t8.prereqs(ctx, "p1", ns)
    return ctx, pre, {"fc": fc08, "t02": t02}


def test_sets(stage, ctx):
    """seam / public / private through t07_rank.set_slots (p1: also equal to test07's HS keys)."""
    out = {}
    for s in SETS:
        games, keys, krec = (t7.set_slots(ctx, s) if stage == "hs" else t8.t7call(t7.set_slots, ctx, s))
        if stage == "p1":
            krec["t07_hs_keys"] = t8.hs_keys_match(ctx, s, keys)
        out[s] = (games, keys, krec["keys_sha256"], krec["questions_sha256"],
                  {"keys_file": krec["keys_file"], "questions_file": krec["questions_file"], "games": krec["games"]})
    return out


def write_input(sets, runs, mode, input_id, out_dir, rec_dir, kit_rec, extra=None, tree_dir=None, man_extra=None):
    root = os.path.join(tree_dir or out_dir, input_id)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "sets"))
    man = {"format": core.INPUT_FORMAT, "input_id": input_id, "mode": mode, "kit_id": kit_rec["kit_id"],
           "created": now_iso(), "git_head": t7.git_head(), "runs": runs, "sets": {}, **(man_extra or {})}
    blobs = {}
    for name, (games, keys, ksha, qsha, source) in sets.items():
        b = set_file_bytes(name, games, keys, ksha, qsha, source)
        blobs[f"sets/{name}.json"] = b
        with open(os.path.join(root, "sets", f"{name}.json"), "wb") as f:
            f.write(b)
        man["sets"][name] = {"file": f"sets/{name}.json", "sha256": sha_bytes(b), "games": len(games),
                             "keys_sha256": ksha, "questions_sha256": qsha, "source": source}
    mb = (json.dumps(man, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with open(os.path.join(root, "INPUT_MANIFEST.json"), "wb") as f:
        f.write(mb)
    zpath = os.path.join(out_dir, f"C10_gpu_input_{input_id}.zip")
    os.makedirs(out_dir, exist_ok=True)
    if t2.same_path(out_dir, UPLOAD_DIR):        # the upload folder holds the kit, the notebook and one input only
        for old in glob.glob(os.path.join(UPLOAD_DIR, "C10_gpu_input_*.zip")):
            os.makedirs(UPLOAD_OLD, exist_ok=True)
            dst = os.path.join(UPLOAD_OLD, f"{os.path.splitext(os.path.basename(old))[0]}_{int(time.time())}.zip")
            os.replace(old, dst)
            print(f"moved the earlier input {os.path.basename(old)} -> {rel(dst)}")
    with zipfile.ZipFile(zpath + ".tmp", "w") as zf:
        zip_write(zf, f"{INPUT_ROOT}/INPUT_MANIFEST.json", data=mb)
        for k in sorted(blobs):
            zip_write(zf, f"{INPUT_ROOT}/{k}", data=blobs[k])
    os.replace(zpath + ".tmp", zpath)
    core.load_input(root)
    rec = {"format": FORMATS["input_record"], "input_id": input_id, "mode": mode, "created": man["created"],
           "manifest_sha256": sha_bytes(mb), "manifest": man, "zip": {"file": os.path.basename(zpath),
                                                                         "sha256": sha_raw(zpath),
                                                                         "bytes": os.path.getsize(zpath)},
           "tool": tool_record(), **(extra or {})}
    write_json(os.path.join(rec_dir, "inputs", f"{input_id}.json"), rec)
    print(f"input {input_id} ({mode}): " + ", ".join(f"{k} {v['games']} games" for k, v in man["sets"].items())
          + f" -> {zpath} (sha256 {rec['zip']['sha256'][:12]})")
    return rec


def record_dir_for(mode, a):
    if mode == "real" or mode == "rehearsal":
        if a.record_dir and not t2.same_path(a.record_dir, REAL_R10):
            refuse(f"--mode {mode}: the records go to results/test11")
        return REAL_R10
    if not a.record_dir:
        refuse(f"--mode {mode} needs --record-dir (outside the repository)")
    return outside_repo(a.record_dir, "--record-dir")


def cmd_input(a):
    mode = a.mode
    rec_dir = record_dir_for(mode, a)
    if mode in ("dry-run", "smoke"):
        if not a.out_dir:
            refuse(f"--mode {mode} needs --out-dir (outside the repository)")
        out_dir = outside_repo(a.out_dir, "--out-dir")
    else:
        if a.out_dir and not t2.same_path(a.out_dir, UPLOAD_DIR):
            refuse(f"--mode {mode}: the input goes to {rel(UPLOAD_DIR)}")
        out_dir = UPLOAD_DIR
    kit_rec = current_kit(REAL_R10 if mode in ("real", "rehearsal") else rec_dir)["manifest"]
    if mode == "rehearsal":
        sets = rehearsal_sets()
        runs = [{"stage": st, "set": s} for st in STAGES for s in ("seam", "offgrid")]
        return 0 if write_input(sets, runs, mode, "rehearsal", out_dir, rec_dir, kit_rec, tree_dir=INPUT_TREES) else 2
    if mode == "smoke":
        sets = smoke_sets()
        runs = [{"stage": st, "set": "smoke"} for st in STAGES]
        return 0 if write_input(sets, runs, mode, "smoke", out_dir, rec_dir, kit_rec) else 2
    if not a.stage:
        refuse(f"--mode {mode} needs --stage hs|p1")
    trig, man_extra = None, None
    if mode == "real":
        trig = trigger_state(a.stage, required=True)
        allowed = allowed_environments(a.stage)
        if not allowed:
            refuse(f"no committed rehearsal before {REHEARSAL_CUTOFF.isoformat()} allows an environment for {a.stage} "
                   f"(PREREG_11 3): the fallback cannot be used for {a.stage}")
        man_extra = {"allowed_environments": {a.stage: [k for k, _f in allowed]},
                     "allowed_by": {a.stage: [f for _k, f in allowed]}}
    ns = ns_for(a.stage, a)
    try:
        ctx, pre, _x = stage_ctx(a.stage, ns)
        sets = test_sets(a.stage, ctx)
    except (t7.T07Error, t8.T08Error) as e:
        raise G11Error(f"({STAGE_TOOLS[a.stage]}_rank prerequisites) {e}", e.code)
    runs = [{"stage": a.stage, "set": s} for s in SETS]
    stamp = "test" if mode == "real" else mode.replace("-", "")
    return 0 if write_input(sets, runs, mode, f"{a.stage}_{stamp}", out_dir, rec_dir, kit_rec,
                            {"trigger": trig, "prerequisites": pre},
                            tree_dir=INPUT_TREES if mode == "real" else None, man_extra=man_extra) else 2


# ================================================================ trigger
def trigger_path(stage, rec_dir=REAL_R10):
    return os.path.join(rec_dir, f"trigger_{stage}.json")


def trigger_state(stage, required, rec_dir=REAL_R10):
    p = trigger_path(stage, rec_dir)
    if not os.path.exists(p):
        if required:
            refuse(f"no trigger for {stage}: PREREG_11 2 (record the objective condition with `trigger`, commit it, "
                   f"then build the input)")
        return None
    rec = read_json(p)
    if rec.get("format") != FORMATS["trigger"] or rec.get("stage") != stage or rec.get("reason") not in TRIGGERS:
        refuse(f"{rel(p)} is not a trigger record of {stage}")
    if not committed_clean(p):
        refuse(f"{rel(p)}: commit it first (the trigger is committed before any Colab run)")
    h, t = commit_time(p)
    return {"path": rel(p), "sha256": sha_lf(p), "reason": rec["reason"], "commit": h, "committed_at": t}


def cmd_trigger(a):
    p = trigger_path(a.stage)
    if os.path.exists(p):
        refuse(f"{rel(p)} exists (one trigger per stage)")
    if a.reason not in TRIGGERS:
        refuse(f"--reason must be one of {TRIGGERS}")
    if not a.evidence or len(a.evidence.strip()) < 20:
        refuse("--evidence: write what was observed (the failing command, its exit code and message, or the 4090 state)")
    t_now = dt.datetime.now(t7.TW)
    why = trigger_problem(a.reason, t_now)
    if why:
        refuse(why)
    done = {a.stage: finished_on_4090(a.stage)}
    if len(done[a.stage]) == len(SETS):
        refuse(f"the 4090 already finished every {a.stage} set ({done}): no fallback (PREREG_11 2: never both paths)")
    rec = {"format": FORMATS["trigger"], "stage": a.stage, "reason": a.reason, "evidence": a.evidence.strip(),
           "created": now_iso(), "created_tw": t_now.isoformat(timespec="seconds"),
           "sets_finished_on_4090": done[a.stage], "tool": tool_record(),
           "rule": "PREREG_11 2: recorded and committed before any test-day Colab run of this stage; the 4090 path of "
                   "this stage is not used afterwards (its partial chunks are moved aside, never mixed in)"}
    write_json(p, rec)
    print(f"trigger {a.stage}: {a.reason} -> {rel(p)} (commit it now, before the Colab run)")
    return 0


def finished_on_4090(stage, is_committed=None):
    """The sets of a stage whose 4090 result (<set>_<stage>.npz and its report) is committed and clean (review minor f:
    a file that is only present does not count)."""
    is_committed = is_committed or committed_clean
    root = os.path.join(RESULTS, "test07" if stage == "hs" else "test08", stage)
    return [s for s in SETS if is_committed(os.path.join(root, f"{s}_{stage}.npz"))
            and is_committed(os.path.join(root, f"{stage}_{s}_report.json"))]


def trigger_problem(reason, t_now):
    """PREREG_11 2: None when the reason may be recorded at t_now, else the refusal text."""
    if reason not in TRIGGERS:
        return f"--reason must be one of {TRIGGERS}"
    if reason == "deadline_11_14" and t_now <= DEADLINE:
        return f"deadline_11_14 holds only after {DEADLINE.isoformat()} (now {t_now.isoformat()})"
    return None


# ================================================================ import
def allowed_environments(stage, rec_dir=REAL_R10, kit_id=None):
    """Committed rehearsal verdicts that allow an environment for the stage, for the current kit -> [(key, file)]."""
    kit_id = kit_id or build_kit_manifest()["kit_id"]
    out = []
    for p in sorted(glob.glob(os.path.join(rec_dir, "rehearsal_*.json"))):
        try:
            rec = read_json(p)
        except G11Error:
            continue
        clean = committed_clean(p)
        if record_allows(rec, stage, kit_id, clean, commit_time(p)[1] if clean else None):
            out.append((rec["environment"][stage], rel(p)))
    return out


def record_allows(rec, stage, kit_id, committed, committed_at):
    """One rehearsal record allows its environment for the stage when it is a rehearsal record of the current kit
    (review round 2: a record of an earlier kit does not count after a rebuild), its verdict for the stage is allowed,
    and it is committed and clean with its last commit before the test data opened (PREREG_11 3.6). PREREG_11 rev 2:
    only records made under the revised rule (rule_version 2) count; a rule-1 record must be regenerated."""
    v = (rec.get("verdict") or {}).get(stage) or {}
    return bool(rec.get("format") == FORMATS["rehearsal"] and rec.get("rule_version") == RULE_VERSION
                and v.get("rule_version") == RULE_VERSION and rec.get("kit_id") == kit_id
                and v.get("allowed") is True and committed and rehearsal_in_time(committed_at))


def rehearsal_in_time(committed_at):
    """PREREG_11 3.6 (review major 2): a rehearsal record counts only when its last commit is before the test data
    opened, 2026-11-04 11:00 +08:00."""
    if not committed_at:
        return False
    return parse_iso(committed_at, "rehearsal commit time") < REHEARSAL_CUTOFF


def colab_results(src, stage, s):
    """The downloaded Colab results root: run summary, input manifest, the stage's ENV.json and the chunks of the set
    (SHA-256 against the run summary). -> record."""
    src = os.path.abspath(src)
    for f in ("RUN_SUMMARY.json", "INPUT_MANIFEST.json", "KIT_MANIFEST.json"):
        if not os.path.exists(os.path.join(src, f)):
            refuse(f"{src}: {f} missing (unpack the whole results zip)")
    summ = read_json(os.path.join(src, "RUN_SUMMARY.json"))
    env_rec = read_json(os.path.join(src, stage, "ENV.json"))
    if env_rec.get("format") != core.ENV_FORMAT or env_rec.get("stage") != stage:
        refuse(f"{stage}/ENV.json is not the stage's environment record")
    runs = [r for r in summ.get("runs") or [] if r["stage"] == stage and r["set"] == s]
    if summ.get("format") != core.SUMMARY_FORMAT or len(runs) != 1 or not runs[0]["done"]:
        refuse(f"RUN_SUMMARY.json: {stage} {s} is not complete ({[{k: r[k] for k in ('stage', 'set', 'done')} for r in summ.get('runs') or []]})")
    run = runs[0]
    if run["run_id"] != env_rec["run_id"] or run["environment"] != env_rec["key"]:
        drift(f"{stage} {s}: the run summary and ENV.json name different runs / environments")
    cdir = os.path.join(src, stage, s)
    files = sorted(f for f in os.listdir(cdir) if f.startswith("chunk_") and f.endswith(".npz") and ".tmp" not in f)
    if files != sorted(run["chunk_sha256"]):
        refuse(f"{stage} {s}: chunk files {files[:3]}... differ from the run summary's list")
    bad = [f for f in files if sha_raw(os.path.join(cdir, f)) != run["chunk_sha256"][f]]
    if bad:
        refuse(f"{stage} {s}: chunks changed since the Colab run summary: {bad[:5]}")
    return {"src": src, "summary": summ, "env_rec": env_rec, "run": run, "chunk_dir": cdir, "files": files,
            "summary_sha256": sha_raw(os.path.join(src, "RUN_SUMMARY.json")),
            "input_manifest_sha256": sha_raw(os.path.join(src, "INPUT_MANIFEST.json"))}


def chunk_extra_problems(stage, path, mode, set_label, M):
    """The chunk fields t07 / t08 chunk_check do not read, against the registered values."""
    c = M["constants"]
    z = load_npz(path)
    want = {"set": set_label, "mode": mode, "katago_commit": c["katago_commit"], "ckpt_bytes": c["ckpt_bytes"]}
    if stage == "hs":
        run = c["hs_run"]
        want.update(format=c["hs_chunk_format"], precision=run["precision"], batch=run["batch"],
                    channels_last=run["channels_last"], hs_lib_sha256=c["hs_lib_sha256"])
    else:
        want.update(format=c["p1_chunk_format"])
    got = {k: (z[k].item() if z[k].shape == () else None) for k in want if k in z}
    bad = [k for k in want if k not in z or got.get(k) != want[k]]
    try:
        parse_iso(str(z["created_utc"]), "created_utc")
    except G11Error:
        bad.append("created_utc")
    return bad, str(z["created_utc"])


def import_set(a):
    stage, s = a.stage, a.set
    if stage not in STAGES or s not in SETS:
        refuse(f"--stage {stage} --set {s}")
    t0 = time.time()
    mode = a.mode
    real = mode == "real"
    rec_dir = REAL_R10 if real else (outside_repo(a.record_dir, "--record-dir") if a.record_dir else None)
    if rec_dir is None:
        refuse(f"--mode {mode} needs --record-dir (outside the repository)")
    trig = trigger_state(stage, required=True) if real else None
    if real:
        for p in (SELF, CORE):
            if not committed_clean(p):
                refuse(f"{rel(p)} must be committed and clean (real mode)")
    attempt = attempt_from_results(rec_dir, stage, a.from_dir, mode)   # first sight of ENV.json, before any check
    col = colab_results(a.from_dir, stage, s)
    if real and col["summary"].get("fake_from_val"):
        refuse("the results are a drill (fake-from-val): never in the real mode")
    if col["env_rec"]["run_id"] != attempt["run_id"]:
        refuse("the stage's ENV.json changed while importing")
    first = os.path.join(col["chunk_dir"], col["files"][0]) if col["files"] else None
    fake = bool(first) and str(load_npz(first)["ckpt_sha256"]) == "fake"
    ns = ns_for(stage, a, fake=fake)
    # ---- the test-day prerequisites of the stage (unchanged tools); a registered 3 is written down as hs / p1 do
    try:
        ctx, pre, x = stage_ctx(stage, ns)
    except (t7.T07Error, t8.T08Error) as e:
        if e.code == 3:
            _write_not_used(stage, s, ns, e)
        raise G11Error(f"({STAGE_TOOLS[stage]}_rank prerequisites) {e}", e.code)
    T = t7 if stage == "hs" else t8
    outs = t7.hs_outputs(ctx, s) if stage == "hs" else t8.p1_outputs(ctx, s)
    have = [p for p in outs if os.path.exists(p)]
    if have:
        refuse(f"{[rel(p) for p in have]} exist: never overwritten (a 4090 result of this set exists, or the set was "
               f"imported already; PREREG_11 2: the two paths are never mixed)")
    dest = os.path.join(ctx["H"], s)
    if real:
        late = late_4090_chunks(ctx["H"], s, parse_iso(trig["committed_at"], "trigger commit time"))
        if late:
            refuse(f"4090 chunks written after the trigger commit exist ({late[:3]}): PREREG_11 2 (never both paths)")
    if os.path.isdir(dest) and any(f.startswith("chunk_") for f in os.listdir(dest)):
        refuse(f"{rel(dest)} holds chunks (a 4090 attempt?): move the folder to {os.path.basename(dest)}_4090_<time> "
               f"first (docs/TESTDAY_GPU_FALLBACK.md; never deleted, never mixed)")
    # ---- the input of the Colab run against what the question files give now
    inrec_p = os.path.join(rec_dir, "inputs", f"{col['summary']['input_id']}.json")
    inrec = read_json(inrec_p)
    if real and not committed_clean(inrec_p):
        refuse(f"{rel(inrec_p)} must be committed")
    if inrec["manifest_sha256"] != col["input_manifest_sha256"]:
        refuse("the Colab run's INPUT_MANIFEST.json is not the recorded input")
    man = inrec["manifest"]
    if man["mode"] != mode:
        refuse(f"the Colab input is mode {man['mode']!r}, this import is {mode!r}")
    if {"stage": stage, "set": s} not in man["runs"]:
        refuse(f"the Colab input has no run {stage} {s}")
    try:
        games, keys, krec = t7.set_slots(ctx, s) if stage == "hs" else t8.t7call(t7.set_slots, ctx, s)
        if stage == "p1":
            krec["t07_hs_keys"] = t8.hs_keys_match(ctx, s, keys)
    except (t7.T07Error, t8.T08Error) as e:
        raise G11Error(f"({STAGE_TOOLS[stage]}_rank set_slots) {e}", e.code)
    src_set = man["sets"][s]
    rebuilt = set_file_bytes(s, games, keys, krec["keys_sha256"], krec["questions_sha256"], src_set["source"])
    if sha_bytes(rebuilt) != src_set["sha256"]:
        refuse(f"{s}: the Colab input set differs from the question / key files now (SHA-256)")
    n_all = len(games)
    if a.max_games:
        if real:
            refuse("--max-games is for drills only")
        games = games[:a.max_games]
        keys = {k: v[:a.max_games] for k, v in keys.items()}
    # ---- the environment of the stage
    env_rec = col["env_rec"]
    env = dict(env_rec["environment"])
    key = core.env_key(env)
    if key != env_rec["key"]:
        drift("ENV.json key and environment differ")
    if s != "seam":
        seam_rep = t7.read_json_or_none(os.path.join(ctx["hs" if stage == "hs" else "p1"],
                                                     f"{'hs' if stage == 'hs' else 'p1'}_seam_report.json"))
        if seam_rep is None:
            refuse(f"import {stage} --set seam first (the seam completes before public and private)")
        if core.env_key(seam_rep.get("environment") or {}) != key:
            drift(f"{s}: environment {key} differs from the seam's {core.env_key(seam_rep.get('environment') or {})} "
                  f"(PREREG_11: one environment per stage)")
    prior = sorted(glob.glob(os.path.join(rec_dir, f"import_{stage}_*.json")))
    for p in prior:
        r = read_json(p)
        if r.get("mode") == mode and (r.get("run_id") != env_rec["run_id"] or r.get("input_id") != man["input_id"]):
            refuse(f"{rel(p)} imported another Colab attempt ({r.get('run_id')}): one Colab attempt per stage "
                   f"(PREREG_11 4)")
    allowed = None
    if real:
        if fake:
            refuse("fake-scorer chunks in the real mode")
        allowed = [f for k, f in allowed_environments(stage) if k == key]
        if not allowed:
            refuse(f"environment {key} did not pass a committed rehearsal for {stage} (results/test11/rehearsal_*.json; "
                   f"PREREG_11 3)")
    # ---- every chunk
    C = t7.t7_chunk() if stage == "hs" else t8.CHUNK
    plan = core.chunk_plan(len(games), C)
    if [core.chunk_name(ci) for ci, _, _ in plan] != col["files"]:
        refuse(f"{s}: {len(col['files'])} chunks for {len(games)} games (want {len(plan)})")
    refs = np.arange(len(games), dtype=np.int64)
    M_colab = read_json(os.path.join(col["src"], "KIT_MANIFEST.json"))
    M = current_kit(REAL_R10 if mode in ("real", "rehearsal") else rec_dir)["manifest"]   # local constants (minor b)
    if M_colab.get("kit_id") != M["kit_id"] or M["kit_id"] != man.get("kit_id"):
        refuse(f"the Colab run used kit {M_colab.get('kit_id')}, the recorded kit is {M['kit_id']}")
    core_sha = M["files"][core.SELF_REL]["sha256_lf"]
    code_json = None
    label = core.CHUNK_SET_PREFIX[stage] + s
    created = []
    t_trig = parse_iso(trig["committed_at"], "trigger commit time") if trig else None
    try:
        for ci, b, e in plan:
            p = os.path.join(col["chunk_dir"], core.chunk_name(ci))
            if stage == "hs":
                t7.chunk_check(p, refs[b:e], krec["keys_sha256"], env, real)
                with np.load(p, allow_pickle=False) as z:
                    if str(z["questions_sha256"]) != krec["questions_sha256"]:
                        refuse(f"{os.path.basename(p)}: question file SHA-256 differs")
            else:
                with np.load(p, allow_pickle=False) as z:
                    cj = str(z["code_sha256"])
                code_json = code_json or cj
                probs = t8.Apply.p1_code_problems(cj)
                if probs or json.loads(cj) != M["constants"]["p1_code_record"]:
                    refuse(f"{os.path.basename(p)}: the P1 code record differs from the files ({probs})")
                t8.chunk_check(p, refs[b:e], krec["keys_sha256"], env, ctx["mode"], real,
                               questions_sha=krec["questions_sha256"], code_sha=code_json, fake=fake)
            bad, c_utc = chunk_extra_problems(stage, p, mode, label, M)
            bad += chunk_origin_problems(p, core_sha, real)
            if bad:
                refuse(f"{os.path.basename(p)}: fields differ from the registered values: {bad}")
            created.append(c_utc)
            if t_trig and parse_iso(c_utc, "created_utc") <= t_trig:
                refuse(f"{os.path.basename(p)} ({c_utc}) is not later than the trigger commit ({trig['committed_at']})")
    except (t7.T07Error, t8.T08Error) as e:
        raise G11Error(f"({STAGE_TOOLS[stage]}_rank chunk_check) {e}", e.code)
    stale = t7.stale_chunks(col["chunk_dir"], x["t02"]["r20"].get("uploaded_at"))
    if stale:
        refuse(f"{len(stale)} chunk(s) were written before R20's first upload ({x['t02']['r20'].get('uploaded_at')})")
    # ---- copy and assemble (the unchanged tools)
    os.makedirs(T.guard_write(ctx, dest), exist_ok=True)
    for f in col["files"]:
        dst = T.guard_write(ctx, os.path.join(dest, f))
        shutil.copyfile(os.path.join(col["chunk_dir"], f), dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        if sha_raw(dst) != col["run"]["chunk_sha256"][f]:
            refuse(f"copy of {f} differs")
    pre = dict(pre)
    pre["gpu_env"] = {"colab_fallback": True, "environment": key, "driver_record_only": env.get("driver"),
                      "allowed_by": allowed if real else f"not checked ({mode})",
                      "frozen_reference_not_used": x["fc"]["environment"]["venv_gpu_reference"],
                      "rule": "PREREG_11 3: an environment that passed the committed rehearsal replaces the "
                              ".venv-gpu / RTX 4090 check of hs / p1"}
    pre["colab_fallback"] = {"registration": "docs/PREREG_11_GPU_FALLBACK.md", "trigger": trig,
                             "input_id": man["input_id"], "run_id": env_rec["run_id"], "kit_id": M.get("kit_id"),
                             "results_dir": rel(col["src"]), "run_summary_sha256": col["summary_sha256"],
                             "tool": tool_record()}
    try:
        T.assemble(ctx, s, dest, plan, keys, krec, env, pre, None, fake, t0)
    except (t7.T07Error, t8.T08Error) as e:
        raise G11Error(f"({STAGE_TOOLS[stage]}_rank assemble) {e}", e.code)
    npz, rp = outs
    rec = {"format": FORMATS["import"], "stage": stage, "set": s, "mode": mode, "created": now_iso(),
           "input_id": man["input_id"], "run_id": env_rec["run_id"], "environment": key,
           "environment_full": env, "fake_scorer": fake, "max_games_drill": bool(a.max_games and len(games) < n_all),
           "games": len(games), "chunks": {f: col["run"]["chunk_sha256"][f] for f in col["files"]},
           "earliest_chunk_created_utc": min(created), "latest_chunk_created_utc": max(created),
           "copied_to": rel(dest), "outputs": {os.path.basename(npz): sha_raw(npz), os.path.basename(rp): sha_lf(rp)},
           "trigger": trig, "allowed_by": allowed, "kit_id": M.get("kit_id"),
           "results_dir": rel(col["src"]), "run_summary_sha256": col["summary_sha256"], "tool": tool_record(),
           "next": f"commit {rel(npz)}, {rel(rp)} and this record; then the next set, then apply (unchanged)"}
    write_json(os.path.join(rec_dir, f"import_{stage}_{s}.json"), rec)
    print(f"import {stage} {s}: {len(games)} games, {len(col['files'])} chunks, environment {key} -> {rel(npz)}, "
          f"{rel(rp)}")
    return 0


FAKE_FROM_VAL_TOOL = "g11 fake-from-val (validation chunks, same games)"


def chunk_origin_problems(path, core_sha, strict):
    """Review blocker / minor c: a chunk of a real run (strict) was written by the kit's core (tool_sha256 = the core's
    LF SHA-256), took time (seconds > 0) and used the real network (checkpoint SHA, torch not 'fake'). Drills also
    accept the fake-from-val marker and fake-scorer chunks."""
    z = load_npz(path)
    tool, secs = str(z["tool_sha256"]), float(z["seconds"])
    if not strict:
        return [] if tool in (core_sha, FAKE_FROM_VAL_TOOL) else ["tool_sha256"]
    bad = []
    if tool != core_sha:
        bad.append("tool_sha256 (not the kit's core)")
    if not secs > 0:
        bad.append("seconds")
    if str(z["ckpt_sha256"]) != g11_ckpt() or str(z["torch"]) == "fake":
        bad.append("fake scorer")
    return bad


def g11_ckpt():
    return pf7.CKPT_SHA


def attempt_from_results(rec_dir, stage, src, mode):
    """Review round 2 minor 3: the attempt is the stage's ENV.json (written before the first chunk), so it is recorded
    as soon as the results folder holds one, before the run summary or any chunk is checked (an unfinished Colab run
    counts as the attempt, PREREG_11 2.3)."""
    src = os.path.abspath(src)
    p = os.path.join(src, stage, "ENV.json")
    if not os.path.exists(p):
        refuse(f"{src}: no {stage}/ENV.json (the Colab run never started this stage; unpack the whole results zip)")
    env_rec = read_json(p)
    if env_rec.get("format") != core.ENV_FORMAT or env_rec.get("stage") != stage:
        refuse(f"{stage}/ENV.json is not the stage's environment record")
    return attempt_record(rec_dir, stage, {"env_rec": env_rec, "summary": {"input_id": env_rec.get("input_id")},
                                           "src": src}, mode)


def attempt_record(rec_dir, stage, col, mode):
    """Review minor d: rec_dir/attempt_<stage>.json is written the first time an import sees a Colab run of the stage
    (even when the import is refused later; commit it); another run_id afterwards is a second attempt (refused)."""
    p = os.path.join(rec_dir, f"attempt_{stage}.json")
    run_id, input_id = col["env_rec"]["run_id"], col["summary"].get("input_id")
    if os.path.exists(p):
        r = read_json(p)
        if r.get("mode") == mode and (r.get("run_id") != run_id or r.get("input_id") != input_id):
            refuse(f"{rel(p)} records Colab run {r.get('run_id')} of {r.get('input_id')}: one Colab attempt per stage "
                   f"(PREREG_11 2.3); this is run {run_id}")
        return r
    r = {"format": "c10-g11 attempt v1", "stage": stage, "mode": mode, "run_id": run_id, "input_id": input_id,
         "environment": col["env_rec"]["key"], "first_seen": now_iso(), "results_dir": rel(col["src"]),
         "rule": "PREREG_11 2.3: one Colab attempt per stage; commit this file as soon as it is written"}
    write_json(p, r)
    print(f"attempt record {rel(p)} written (commit it)")
    return r


def late_4090_chunks(H, s, t_trig):
    """Review minor e: chunks in H/<set>_4090_*/ written after the trigger commit (the 4090 path kept running)."""
    out = []
    for p in sorted(glob.glob(os.path.join(H, f"{s}_4090_*", "chunk_*.npz"))):
        try:
            with np.load(p, allow_pickle=False) as z:
                c = str(z["created_utc"])
        except Exception:  # noqa: BLE001
            out.append(f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} (unreadable)")
            continue
        if parse_iso(c, "created_utc") > t_trig:
            out.append(f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} {c}")
    return out


def _write_not_used(stage, s, ns, e):
    """As t07_rank.cmd_hs / t08_rank.cmd_p1: the registered 'not used' is written down (R/not_used.json)."""
    try:
        ctx = t7.resolve(ns, "hs") if stage == "hs" else t8.resolve(ns, "p1")
        T = t7 if stage == "hs" else t8
        root = ctx["R7"] if stage == "hs" else ctx["R8"]
        T.write_json(ctx, os.path.join(root, "not_used.json"),
                     {"format": T.FORMATS["not_used"], "mode": ctx["mode"], "created": now_iso(), "exit_code": 3,
                      "stage": f"g11 import {stage} --set {s}: {e.stage or 'prerequisites'}", "reason": e.reason,
                      "message": str(e), "details": e.details, "tool": T.tool_record()})
    except Exception:  # noqa: BLE001
        traceback.print_exc()


# ================================================================ rehearsal compare
def rehearsal_ctx(root, name):
    d = os.path.join(root, name)
    return {"mode": "rehearsal", "real": False, "R7": d, "R8": d, "hs": os.path.join(d, "hs"),
            "p1": os.path.join(d, "p1"), "H": os.path.join(d, "data"), "TMP": os.path.join(d, "tmp"),
            "write_roots": [d]}


def val_keys_rows(npz_path, r0, r1, prefix=None):
    z = load_npz(npz_path)
    q = [str(x) for x in z["question_id"][r0:r1]]
    if prefix:
        q = [prefix + x[len("val_"):] for x in q]
    return {"question_id": q, "slot": [int(x) for x in z["slot"][r0:r1]], "color": [str(x) for x in z["color"][r0:r1]]}


def stage_verdict(checks, measured, tol):
    """PREREG_11 rev 2 (rule_version 2): allowed iff every seam passes (the pinned tolerances), failed / counts /
    n_moves are equal, and every off-grid difference (x 1.0) is within the current tolerance (a missing measurement is
    not within). 2.5 x the off-grid differences (rule 1) is recorded only."""
    m = {k: (None if measured.get(k) is None else float(measured[k])) for k in tol}
    within = {k: bool(m[k] is not None and m[k] <= tol[k]) for k in tol}
    scaled = {k: (None if m[k] is None else FACTOR * m[k]) for k in tol}
    allowed = bool(checks.get("seam_passed") is True and checks.get("counts_equal") is True and all(within.values()))
    return {"allowed": allowed, "rule_version": RULE_VERSION, "checks": checks, "tolerances_now": tol,
            "offgrid_measured": measured, "within_current_tolerance": within,
            "record_only_offgrid_times_2_5": scaled,
            "record_only_times_2_5_within": {k: bool(scaled[k] is not None and scaled[k] <= tol[k]) for k in tol},
            "rule": "PREREG_11 rev 2 (rule_version 2): allowed iff every seam passes with the pinned tolerances (same "
                    "games and chunking as the test day) and every off-grid difference (x 1.0) <= the current "
                    "tolerance; 2.5 x off-grid is recorded only; the tolerances are never changed (they live in the "
                    "pinned test-day tools)"}


def zero_tripwire(out):
    """Review round 2 item 9: a GPU other than the RTX 4090 whose off-grid differences are all exactly 0.0 (both
    stages) has almost surely not computed anything itself (fp16 under another batch composition is never bit-equal):
    refused. -> None or the refusal text."""
    gpus = {st: str((out["environment"].get(st) or {}).get("gpu")) for st in STAGES}
    if any("4090" in g for g in gpus.values()):
        return None
    vals = [v for st in STAGES for v in (out["verdict"][st].get("offgrid_measured") or {}).values()]
    if vals and all(v is not None and float(v) == 0.0 for v in vals):
        return (f"every off-grid difference is exactly 0.0 on {gpus}: these are not that GPU's own numbers "
                f"(tripwire); no record written")
    return None


def cmd_rehearsal_compare(a):
    src = os.path.abspath(a.from_dir)
    base = outside_repo(a.work_dir, "--work-dir")
    os.makedirs(base, exist_ok=True)       # never deletes anything: a fresh subfolder per call (review round 2)
    work = os.path.join(base, f"g11_compare_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f')}")
    os.makedirs(work)
    inrec = read_json(os.path.join(REAL_R10, "inputs", "rehearsal.json"))
    if sha_raw(os.path.join(src, "INPUT_MANIFEST.json")) != inrec["manifest_sha256"]:
        refuse("the Colab results are not of the recorded rehearsal input")
    kit = current_kit(REAL_R10)["manifest"]
    M = read_json(os.path.join(src, "KIT_MANIFEST.json"))
    if M.get("kit_id") != kit["kit_id"]:
        refuse(f"the rehearsal ran kit {M.get('kit_id')}, the recorded kit is {kit['kit_id']}")
    summ = read_json(os.path.join(src, "RUN_SUMMARY.json"))
    if summ.get("fake_from_val") or summ.get("fake_scorer") or summ.get("max_games") or summ.get("mode") != "rehearsal":
        refuse("the results are not a complete GPU rehearsal run (fake-from-val, fake scorer, --max-games or another "
               "mode): review blocker")
    core_sha = kit["files"][core.SELF_REL]["sha256_lf"]
    M = kit                                   # the local constants (the Colab copy only has to carry the same id)
    out = {"format": FORMATS["rehearsal"], "rule_version": RULE_VERSION, "created": now_iso(), "results_dir": rel(src),
           "kit_id": kit["kit_id"],
           "input_id": "rehearsal", "rows": REHEARSAL_ROWS, "tool": tool_record(), "environment": {}, "stages": {},
           "verdict": {}}
    model7 = pf7.load_hs_map(t7.path_of("results/prereg07/rank/hs_map.npz"))
    model8 = {k: v for k, v in load_npz(t7.path_of("results/prereg08/rank/p1_map.npz")).items()
              if k in ("mu", "sd", "W", "b")}
    import t07_fake_tests as t7f
    import t08_fake_tests as t8f
    import t09_fake_tests as t9f
    import t09_rank as t9
    mp9 = t9f.FW.load_map(t9f.P1E_MAP)
    for stage in STAGES:
        sets = {}
        env_rec = read_json(os.path.join(src, stage, "ENV.json"))
        env = env_rec["environment"]
        out["environment"][stage] = core.env_key(env)
        out.setdefault("environment_full", {})[stage] = env
        T = t7 if stage == "hs" else t8
        ctx = rehearsal_ctx(work, "test07" if stage == "hs" else "test08")
        for name, (r0, r1) in REHEARSAL_ROWS.items():
            col = colab_results(src, stage, name)
            if col["env_rec"]["run_id"] != env_rec["run_id"]:
                drift(f"{stage} {name}: another run than the stage's")
            sname = "seam" if name == "seam" else "private"
            dest = os.path.join(ctx["H"], sname)
            os.makedirs(dest, exist_ok=True)
            for f in col["files"]:
                shutil.copyfile(os.path.join(col["chunk_dir"], f), os.path.join(dest, f))
            for f in col["files"]:
                bad, _c = chunk_extra_problems(stage, os.path.join(dest, f), "rehearsal", core.CHUNK_SET_PREFIX[stage] + name,
                                               M)
                bad += chunk_origin_problems(os.path.join(dest, f), core_sha, True)
                if bad:
                    refuse(f"{stage} {name} {f}: {bad}")
            keys = val_keys_rows(VAL_HS if stage == "hs" else VAL_P1, r0, r1, None if name == "seam" else "priv_")
            krec = {"games": r1 - r0, "keys_sha256": pf7.VAL_KEYS_SHA, "questions_sha256": pf7.QUESTIONS_SHA,
                    "note": "rehearsal (validation rows)"}
            plan = core.chunk_plan(r1 - r0, t7.t7_chunk() if stage == "hs" else t8.CHUNK)
            T.assemble(ctx, sname, dest, plan, keys, krec, env, {"rehearsal": True}, None, False, time.time())
            sets[name] = {"chunks": len(plan), "games": r1 - r0}
        if stage == "hs":
            z = load_npz(os.path.join(ctx["hs"], "seam_hs.npz"))
            seam = t7.gpu_seam_record(z, load_npz(VAL_HS), load_npz(t7.path_of("results/prereg07/hs/val_seam_moves.npz")),
                                      model7)
            off = t7f.compare(os.path.join(ctx["hs"], "private_hs.npz"), "private")
            tol = {"profile_mean": t7.TOL_PROFILE_MEAN, "lp_hs": t7.TOL_LP}
            measured = {"profile_mean": off["max_abs_profile_mean"], "lp_hs": off["max_abs_lp_hs"]}
            counts_ok = bool(seam["counts_equal"] and all(off["equal"].values()))
            out["stages"]["hs"] = {"sets": sets, "seam_t07_gpu_seam_record": seam, "offgrid_t07_fake_tests_compare": off}
            checks = {"seam_passed": bool(seam["passed"]), "counts_equal": counts_ok}
        else:
            z = load_npz(os.path.join(ctx["p1"], "seam_p1.npz"))
            seam8 = t8.gpu_seam_record(z, load_npz(t7.path_of("results/prereg08/feat/val_seam_feats.npz")), model8)
            off8 = t8f.compare(ctx["R8"], ctx["H"], "private")
            _r, sd = t9.read_set_chunks(os.path.join(ctx["H"], "seam"), [c for c in sorted(os.listdir(
                os.path.join(ctx["H"], "seam"))) if c.startswith("chunk_")])
            _vr, vd = t9.val_seam_chunks()
            seam9 = t9.gpu_seam_record(sd, vd, mp9)
            off9 = t9f.compare(ctx["R8"], ctx["H"], "private")
            tol = {"std_feature": t8.TOL_FEAT, "lp_p1": t8.TOL_LP, "std_ex": t9.TOL_EX, "lp_p1e": t9.TOL_E}
            measured = {"std_feature": off8["max_abs_std_feature"], "lp_p1": off8["max_abs_lp_p1"],
                        "std_ex": off9["max_abs_std_ex"], "lp_p1e": off9["max_abs_lp_p1e"]}
            counts_ok = bool(seam8["counts_equal"] and all(off8["equal"].values()) and seam9["counts_equal"]
                             and off9["counts_equal"])
            out["stages"]["p1"] = {"sets": sets, "seam_t08_gpu_seam_record": seam8, "seam_t09_gpu_seam_record": seam9,
                                   "offgrid_t08_fake_tests_compare": off8, "offgrid_t09_fake_tests_compare": off9}
            checks = {"seam_passed": bool(seam8["passed"] and seam9["passed"]), "counts_equal": counts_ok}
        out["verdict"][stage] = stage_verdict(checks, measured, tol)
    trip = zero_tripwire(out)
    if trip:
        refuse(trip)
    path = os.path.abspath(a.json) if a.json else os.path.join(
        REAL_R10, "rehearsal_" + re.sub(r"[^a-z0-9]+", "-", out["environment"]["hs"]["gpu"].lower()).strip("-") + ".json")
    write_json(path, out)
    for stage in STAGES:
        v = out["verdict"][stage]
        print(f"rehearsal {stage} on {out['environment'][stage]}: {'ALLOWED' if v['allowed'] else 'NOT ALLOWED'}; "
              f"checks {v['checks']}; off-grid {v['offgrid_measured']} vs {v['tolerances_now']} (rule_version "
              f"{v['rule_version']}; x 2.5 record only {v['record_only_offgrid_times_2_5']})")
    print(f"-> {rel(path)} (commit it; the importer reads only committed rehearsal records)")
    return 0


# ================================================================ drills: the validation chunks in the Colab format
def val_chunk_index(report):
    vr = read_json(report)
    starts = np.cumsum([0] + [int(c["games"]) for c in vr["chunks"]])
    return vr, starts


def val_game_rows(stage, rows):
    """Per validation row: the chunk arrays of that game (positions in order). -> (list of dicts, env, ck fields)."""
    vr, starts = val_chunk_index(VAL_REPORT7 if stage == "hs" else VAL_REPORT8)
    need = sorted(set(int(np.searchsorted(starts, r, side="right") - 1) for r in rows))
    per, meta = {}, None
    for ci in need:
        ch = vr["chunks"][ci]
        p = os.path.join(REPO, *ch["file"].split("/"))
        if sha_raw(p) != ch["sha256"]:
            refuse(f"{ch['file']} differs from its validation report")
        z = load_npz(p)
        if meta is None:
            meta = {k: z[k] for k in z if z[k].shape == () and k not in ("errors", "created_utc", "seconds",
                                                                         "gpu_seconds")}
        pg = np.asarray(z["pos_game"])
        errs = {e["game"]: e["error"] for e in json.loads(str(z["errors"]))}
        G = len(z["ref"])
        for g in range(G):
            r = int(starts[ci]) + g
            sel = np.flatnonzero(pg == g)
            d = {k: np.asarray(z[k])[sel] for k in z if k.startswith("pos_")}
            for k in ("counts", "n_moves", "failed", "sums" if stage == "hs" else "means"):
                d[k] = np.asarray(z[k])[g]
            d["error"] = errs.get(g)
            per[r] = d
    return [per[int(r)] for r in rows], meta, vr["environment"]


def res_of(stage, games):
    """pilot_run.run_chunk / p1_extract.run_chunk result layout from per-game parts (positions in game order)."""
    out = {}
    main = "sums" if stage == "hs" else "means"
    out[main] = np.stack([g[main] for g in games])
    out["counts"] = np.stack([g["counts"] for g in games])
    out["n_moves"] = np.array([g["n_moves"] for g in games])
    out["failed"] = np.array([g["failed"] for g in games])
    out["errors"] = json.dumps([{"game": i, "error": g["error"]} for i, g in enumerate(games) if g["error"]])
    keys = ("pos_game", "pos_turn", "pos_side", "pos_logp") if stage == "hs" else \
        ("pos_game", "pos_side", "pos_turn", "pos_logp", "pos_entropy")
    for k in keys:
        parts = [np.full(len(g["pos_side"]), i, dtype=g[k].dtype) if k == "pos_game" else g[k]
                 for i, g in enumerate(games)]
        out[k] = np.concatenate(parts)
    return out


def cmd_fake_from_val(a):
    """Drills only: the Colab-format chunks a run on the validation chunks' GPU would have written for the games of a
    dry-run input (seam: validation rows 0-399; public / private: the fake prefixes re-mapped to the validation rows).
    Refused for rehearsal and real inputs (review blocker)."""
    inp = os.path.abspath(a.input)
    out = outside_repo(a.out, "--out")
    I = core.load_input(inp)
    if I["mode"] != "dry-run":
        refuse("fake-from-val is for dry-run inputs only (never rehearsal or real: review blocker)")
    kit_m = read_json(os.path.join(a.kit, "KIT_MANIFEST.json"))
    runs = write_val_chunks(I, inp, a.stage, out, kit_m, a.kit, relabel_gpu=a.relabel_gpu)
    print(f"fake-from-val {a.stage}: " + ", ".join(f"{r['set']} {r['games']} games / {r['chunks']} chunks"
                                                   for r in runs) + f" -> {out}")
    return 0


def write_val_chunks(I, inp, stage, out, kit_m, kit_dir, relabel_gpu=None, tool_sha=FAKE_FROM_VAL_TOOL, seconds=0.0,
                     summary_extra=None):
    """The validation chunks of the input's games in the Colab layout (library function for the drills and for the
    tests that show rehearsal-compare rejects such data; the CLI allows dry-run inputs only)."""
    vz = load_npz(VAL_HS if stage == "hs" else VAL_P1)
    idx = {(str(q), int(sl)): i for i, (q, sl) in enumerate(zip(vz["question_id"], vz["slot"]))}
    env_rec = None
    os.makedirs(out, exist_ok=True)
    runs = []
    for r in I["runs"]:
        if r["stage"] != stage:
            continue
        s = r["set"]
        d = I["sets_data"][s]
        rows = []
        for q, sl in zip(d["keys"]["question_id"], d["keys"]["slot"]):
            qq = q if q.startswith("val_") else "val_" + q.split("_", 1)[1]
            rows.append(idx[(qq, int(sl))])
        games, meta, venv = val_game_rows(stage, rows)
        env = {k: venv.get(k) for k in ("python", "torch", "numpy", "cuda", "cudnn", "gpu", "driver")}
        if relabel_gpu:              # drills: the 4090 numbers under another GPU name (apply must not care)
            env = dict(env, gpu=relabel_gpu)
        if env_rec is None:
            env_rec = core.stage_environment(out, stage, env, I["input_id"], I["mode"])
        cdir = os.path.join(out, stage, s)
        os.makedirs(cdir, exist_ok=True)
        size = kit_m["constants"]["hs_run"]["chunk"] if stage == "hs" else kit_m["constants"]["p1_run"]["chunk"]
        shas = {}
        for ci, b, e in core.chunk_plan(len(games), size):
            res = res_of(stage, games[b:e])
            ck = {"bytes": int(meta["ckpt_bytes"]), "sha256": str(meta["ckpt_sha256"])}
            common = dict(M=kit_m, refs=np.arange(b, e, dtype=np.int64), set_label=core.CHUNK_SET_PREFIX[stage] + s,
                          env=env, ck=ck, keys_sha=d["keys_sha256"], questions_sha=d["questions_sha256"],
                          tool_sha=tool_sha, mode=I["mode"], created_utc=core.now_iso(), seconds=seconds, res=res)
            if stage == "hs":
                fields = core.hs_fields(**common)
            else:
                fields = core.p1_fields(**common, torch_v=str(meta["torch"]), settings=core.p1_settings(kit_m, False),
                                        meta_sha=str(meta["meta_sha256"]), code_json=str(meta["code_sha256"]),
                                        gpu_seconds=seconds)
            p = core.save_chunk(os.path.join(cdir, core.chunk_name(ci)), fields)
            shas[core.chunk_name(ci)] = core.sha_raw(p)
        runs.append({"stage": stage, "set": s, "done": True, "stopped": None, "games": len(games),
                     "chunks": len(shas), "computed_now": len(shas), "run_id": env_rec["run_id"],
                     "environment": core.env_key(env), "chunk_sha256": shas})
    shutil.copyfile(os.path.join(inp, "INPUT_MANIFEST.json"), os.path.join(out, "INPUT_MANIFEST.json"))
    shutil.copyfile(os.path.join(kit_dir, "KIT_MANIFEST.json"), os.path.join(out, "KIT_MANIFEST.json"))
    prev = os.path.join(out, "RUN_SUMMARY.json")
    old = read_json(prev)["runs"] if os.path.exists(prev) else []
    summary = {"format": core.SUMMARY_FORMAT, "core": core.CORE_FORMAT, "input_id": I["input_id"], "mode": I["mode"],
               "kit_id": kit_m.get("kit_id"), "finished_utc": now_iso(), "exit_code": 0,
               "runs": [r for r in old if r["stage"] != stage] + runs,
               **({"fake_from_val": True} if summary_extra is None else summary_extra)}
    core.write_json_atomic(prev, summary)
    return runs


# ================================================================ status
def cmd_status(a):
    rec_dir = os.path.abspath(a.record_dir) if a.record_dir else REAL_R10
    print(f"PREREG_11 GPU fallback records in {rel(rec_dir)}:")
    kp = os.path.join(rec_dir, "kit_manifest.json")
    if os.path.exists(kp):
        k = read_json(kp)
        print(f"- kit {k['kit_id']}: zip {(k.get('zip') or {}).get('sha256', '')[:12]} "
              f"({'committed' if committed_clean(kp) else 'NOT committed'})")
    else:
        print("- kit: not built")
    for p in sorted(glob.glob(os.path.join(rec_dir, "inputs", "*.json"))):
        r = read_json(p)
        sets = ", ".join(f"{k} {v['games']}" for k, v in r["manifest"]["sets"].items())
        print(f"- input {r['input_id']} ({r['mode']}): {sets} ({'committed' if committed_clean(p) else 'NOT committed'})")
    for p in sorted(glob.glob(os.path.join(rec_dir, "rehearsal_*.json"))):
        r = read_json(p)
        print(f"- rehearsal {os.path.basename(p)}: " + "; ".join(
            f"{st} {r['environment'][st]['gpu']} -> {'allowed' if r['verdict'][st]['allowed'] else 'NOT allowed'}"
            for st in STAGES) + f" ({'committed' if committed_clean(p) else 'NOT committed'})")
    for st in STAGES:
        t = trigger_path(st, rec_dir)
        print(f"- trigger {st}: " + (f"{read_json(t)['reason']} ({'committed' if committed_clean(t) else 'NOT committed'})"
                                     if os.path.exists(t) else "none (the 4090 path applies)"))
    for p in sorted(glob.glob(os.path.join(rec_dir, "import_*.json"))):
        r = read_json(p)
        print(f"- {os.path.basename(p)}: {r['games']} games, {r['environment']['gpu']} "
              f"({'committed' if committed_clean(p) else 'NOT committed'})")
    return 0


# ================================================================ main
def add_tree_args(p):
    p.add_argument("--results-dir", help="R7 (hs) or R8 (p1); real: results/test07 / results/test08")
    p.add_argument("--data-dir", help="the chunk folder root (hs: data/test07/hs; p1: data/test08/p1)")
    p.add_argument("--t07-results-dir")
    p.add_argument("--t04-results-dir")
    p.add_argument("--t02-results-dir")
    p.add_argument("--t02-out-dir")
    p.add_argument("--tests-dir")
    p.add_argument("--checks-dir")
    p.add_argument("--r20-uploaded-at")
    p.add_argument("--now")
    p.add_argument("--record-dir", help="drills: the folder of the g11 records (outside the repository)")


def build_parser():
    ap = argparse.ArgumentParser(prog="g11_gpu_fallback.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kit")
    k.add_argument("--out-dir")
    k.add_argument("--record-dir")
    k.add_argument("--zip-dir", help="where the kit zip goes (default data/test11/upload, the folder the user uploads)")
    k.add_argument("--no-zip", action="store_true")
    i = sub.add_parser("input")
    i.add_argument("--mode", required=True, choices=("rehearsal", "real", "dry-run", "smoke"))
    i.add_argument("--stage", choices=STAGES)
    i.add_argument("--out-dir")
    add_tree_args(i)
    t = sub.add_parser("trigger")
    t.add_argument("--stage", required=True, choices=STAGES)
    t.add_argument("--reason", required=True)
    t.add_argument("--evidence", required=True)
    m = sub.add_parser("import")
    m.add_argument("--stage", required=True, choices=STAGES)
    m.add_argument("--set", required=True, choices=SETS)
    m.add_argument("--from", dest="from_dir", required=True)
    m.add_argument("--mode", default="real", choices=("real", "dry-run"))
    m.add_argument("--max-games", type=int, default=0)
    add_tree_args(m)
    r = sub.add_parser("rehearsal-compare")
    r.add_argument("--from", dest="from_dir", required=True)
    r.add_argument("--work-dir", required=True)
    r.add_argument("--json")
    f = sub.add_parser("fake-from-val")
    f.add_argument("--stage", required=True, choices=STAGES)
    f.add_argument("--input", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--kit", required=True)
    f.add_argument("--relabel-gpu", help="drills only: record this GPU name instead of the validation run's")
    s = sub.add_parser("status")
    s.add_argument("--record-dir")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    fn = {"kit": cmd_kit, "input": cmd_input, "trigger": cmd_trigger, "import": import_set,
          "rehearsal-compare": cmd_rehearsal_compare, "fake-from-val": cmd_fake_from_val, "status": cmd_status}[a.cmd]
    try:
        return fn(a)
    except G11Error as e:
        print(f"{'REFUSED' if e.code == 2 else 'NOT USED' if e.code == 3 else 'STOP: DRIFT / MISMATCH'} "
              f"(exit {e.code}): {e}", file=sys.stderr)
        return e.code
    except (core.Refused, core.Drift) as e:
        print(f"REFUSED (exit {e.code}): {e}", file=sys.stderr)
        return e.code
    except (t7.T07Error, t8.T08Error) as e:
        print(f"(test-day tool) exit {e.code}: {e}", file=sys.stderr)
        return e.code
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
