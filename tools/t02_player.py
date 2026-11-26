"""Competition 1 (player identification, TEAM_11249) on the official test pools: PREREG_02 rev 5
section 9 items 2, 4 and 5; implementation spec results/test02/design/spec.md 3.0 and 3.3.

    PY tools/t02_player.py run --pool seam_val|public|private [--dry-run|--smoke|--rehearsal]
                               [--results-dir D] [--tests-dir T] [--out-dir O]
                               [--no-cnn --reason TEXT] [--rerun --reason TEXT]
    PY tools/t02_player.py submit [same mode flags] [--rerun --reason TEXT]

    extra options (defaults in brackets): --checks-dir [D/checks]; --seam-ref
    [results/prereg02/player; only --smoke may change it]; --seam-candidates / --seam-questions
    [data/local/player_val_candidates.csv / player_val_questions.csv]

Per pool, the frozen prereg02 functions of tools/p02_player_fuse.py and tools/p02_player_open.py
are called in the order of cmd_select / cmd_apply (no frozen file is changed or re-written):
    select  pf.Data(candidates, D/player/<pool>, cnn, smoke) -> pf.self_retrieval(strict) ->
            means -> pf.choose -> gate (selected - Q00 >= 0.02) -> sr_med = median max_c cosO
    D1      (rev 5, test time) gate failed -> effective = Q00 (selected and effective recorded)
    guard   thresholds from results/prereg02/player/frozen_config.json (guard_calibration.json
            SHA pinned), rho denominator = this pool's own sr_med, b_c / b_q of this pool
    apply   pf.Data(candidates, ..., q_path=questions) -> cosO / E terms -> pf.guard_decision ->
            pf.fuse -> pf.top_ids; rows in question-file order (asserted)
Outputs O/<pool>/{pool_select.json, top5.csv, apply_report.json}; seam_val also
O/seam_val/seam_local.json. public / private refuse to run until seam_local.json passed.
submit writes O/submission_player.csv (header, public rows, private rows; LF) and
O/submit_report.json after check_submission.check(..., official=True) returns no problem.

Seam (registered, PREREG_02 rev 5 9.2): on the validation pool the self-retrieval selects Q04, the
gate passes, the guard is at level 1 and the top-5 equals results/prereg02/player/player_val_top5.csv
on >= 398 of 400 questions. Implementation additions (recorded in seam_local.json):
    * opening-only quantities do not depend on the Colab dumps, so the Q22 / Q23 means and the
      self-retrieval medians must equal prereg02's exactly (a difference is a local bug);
    * when the seam_val dumps are bitwise equal to prereg02's (compared here, array by array), all
      24 means and the top-5 file must be identical too ("expected identical" in 9.2).
    * --no-cnn (rev 4 item 4 fallback {Q22, Q23}): the seam is the opening-only part above.
    * --smoke: the reference is --seam-ref (a prereg02 smoke tools dir) and every question must match.

Modes: real (default; D = results/test02, T = data/test exactly; outputs inside results/test02);
--dry-run (fake tree from tools/t02_fake_tests.py), --smoke (notebook SMOKE outputs) and
--rehearsal (a Colab rehearsal) write outside the repository only. Real mode refuses unless the
frozen player config is committed and equal to HEAD, the frozen player tools are unchanged, this
file and tools/check_submission.py are committed and clean, the notebook outputs of spec 2.3 are
tracked and unchanged, the notebook run completed with its reproduction check passed and no earlier run
in its run_log failed (a run again in the same Drive folder is not the registered rerun), config.json
says smoke = rehearsal = false, the test files and MANIFEST.json have the SHA-256 the notebook
recorded, and results/test02/checks/player/t1.json is the full T1 on the raw files (complete, stage raw,
mode real) and says PASS, NORMALIZED or FAIL_OPENING (PREREG_02 rev 5
9.6: with FAIL_OPENING competition 1 keeps its per-pool self-retrieval and guard; the verdict is recorded in
pool_select.json, apply_report.json and submit_report.json; FAIL_STRUCT or a missing t1.json stops).
Existing outputs are never overwritten without --rerun --reason. Nothing here opens a label file
(any input whose name contains that word is refused) or imports a competition-2 module.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import check_submission  # noqa: E402
import p02_player_fuse as pf  # noqa: E402
import p02_player_open as po  # noqa: E402

REPO = po.REPO
TOOL_REL = "tools/t02_player.py"
CHECK_REL = "tools/check_submission.py"
POOLS = ("seam_val", "public", "private")
TEST_POOLS = ("public", "private")
TEST_FILES = {"public": ("player_identification_test_public.csv", "player_identification_candidates_public.csv"),
              "private": ("player_identification_test_private.csv", "player_identification_candidates_private.csv")}
PREFIX = {"public": ("pub_", "pub_p_", "pub_"), "private": ("priv_", "priv_p_", "priv_")}   # question, player, game
VAL_QUESTIONS = ("data", "local", "player_val_questions.csv")
VAL_CANDIDATES = ("data", "local", "player_val_candidates.csv")
SUBMISSION = "submission_player.csv"
NOTEBOOK_REL = "colab/test02_player.ipynb"
NB_SHA_RE = re.compile(r"NB_SOURCE_SHA256 = '([0-9a-f]{64})'")
NB_SHA_PLACEHOLDER = "@@NB_SOURCE_SHA256@@"

# prereg02 player pins (spec 2.2 / 3.3; all in results/prereg02/player, SHA-256 of the LF text / bytes)
FROZEN_CONFIG_SHA256 = "d70726ab198643c749b8dadc78f37b52c69bd831513e36ec564a9d04e2c7f3de"
GUARD_CALIBRATION_SHA256 = "793c16a79080e57d4c60941a1fc311f3c0d950e19c42bafaff16a54f7ad43a28"
SR_REPORT_SHA256 = "9c06f6ac852092adfe4d0c2824da011baf481006a32fcf0a2051971df28cfc2e"
VAL_TOP5_SHA256 = "69ca74a61ce34f8ffa8949df731df433d474347dce9aabb535156ab8197ead6c"
WEIGHTS_SHA256 = "356eb4f65488925e2a552ea382979780c903ef515b85d329a3805db837b7773b"
P02_DUMP_SHA256 = {"cand_games.npz": "56ed03d12e9a79871a21b5f69dc4bd4bb8c8aa69b1544cb8ec799e80227cfdea",
                   "query_games.npz": "665d0888a93f6632eb19edac76abf996d5499cd27c683d9844b97cb274b6f689"}

# registered seam (PREREG_02 rev 5 9.2)
SEAM_COMBO = "Q04"
SEAM_LEVEL = 1
SEAM_MAX_DIFFERENT = 2          # >= 398 of 400
OPEN_ONLY = ("Q22", "Q23")      # opening-only combinations: independent of the Colab dumps

# notebook outputs of spec 2.3 (tracked by git in real mode)
NB_FILES = ("config.json", "run_log.json", "env.json", "timings.json", "repro_check.json", "seam_p02.json",
            "readonly_before.json", "readonly_after.json")
DUMP_FILES = ("cand_games.npz", "query_games.npz", "_INPUT.json")
T1_OK = ("PASS", "NORMALIZED")
T1_ALLOWED = T1_OK + ("FAIL_OPENING",)   # rev 5 9.6: FAIL_OPENING -> competition 1 keeps the per-pool procedure
NB_FAILED = ("seam_failed", "repro_failed", "readonly_changed", "final_check_failed", "preflight_failed")
# an earlier run with one of these statuses in the same run_log: the notebook was run again in the same Drive folder
# (a preflight stop, e.g. the GPU gate, and an interrupted run may be run again; these may not: PREREG_02 9.2, 10.10)
EARLIER_FAILED = ("seam_failed", "repro_failed", "readonly_changed", "final_check_failed")


# ---------------------------------------------------------------- small helpers
def die(msg):
    raise SystemExit("t02_player: " + msg)


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def norm(p):
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def same_path(a, b):
    return norm(a) == norm(b)


def sha(path):
    return po.sha256_file(path)


def read_json(path):
    with open(po.refuse_forbidden(path), encoding="utf-8") as f:
        return json.load(f)


def maybe_json(path):
    return read_json(path) if os.path.exists(path) else None


def disp(path):
    return pf._disp(path)


def git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True)


def git_head():
    return pf.git_head()


def git_tracked_clean(paths, what):
    """paths (absolute) must be tracked by git and unchanged against HEAD."""
    rels = [pf._rel(p) for p in paths]
    r = git("ls-files", "--error-unmatch", "--", *rels)
    if r.returncode != 0:
        die(f"{what}: not all tracked by git; commit them first ({r.stderr.strip()[:300]})")
    st = git("status", "--porcelain", "--", *rels)
    if st.returncode != 0 or st.stdout.strip():
        die(f"{what}: changed against HEAD ({st.stdout.strip()[:300]})")
    return rels


def question_ids(path):
    csv.field_size_limit(10 ** 9)
    with open(po.refuse_forbidden(path), newline="", encoding="utf-8-sig") as f:
        return [r["question_id"] for r in csv.DictReader(f)]


def write_csv_lf(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def read_top5(path):
    with open(po.refuse_forbidden(path), newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != ["question_id"] + pf.TOP_COLS:
        die(f"{path}: header is not question_id,{','.join(pf.TOP_COLS)}")
    return rows[1:]


def lib_versions():
    """numpy / scipy of this .venv: the exact-equality additions of the local seam (opening-only means, the
    self-retrieval medians) depend on them, not on Colab (review finding 2026-09-26)."""
    import scipy
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}


def provenance():
    return dict({"git_head": git_head(), "tool": TOOL_REL,
                 "tool_sha256": po.sha256_text_file(os.path.abspath(__file__))}, **lib_versions(),
                **{"frozen_tools_sha256": po.source_shas(), "argv": sys.argv[1:], "created": now_iso()})


# ---------------------------------------------------------------- modes and paths
def mode_of(a):
    return "dry-run" if a.dry_run else "smoke" if a.smoke else "rehearsal" if a.rehearsal else "real"


def resolve(a):
    a.mode = mode_of(a)
    a.repo_results = os.path.join(REPO, "results", "test02")
    a.results_dir = os.path.abspath(a.results_dir or a.repo_results)
    a.tests_dir = os.path.abspath(a.tests_dir or os.path.join(REPO, "data", "test"))
    if a.out_dir is None:
        if a.mode != "real":
            die("--%s needs --out-dir outside the repository" % a.mode)
        a.out_dir = os.path.join(a.repo_results, "player")
    a.out_dir = os.path.abspath(a.out_dir)
    a.checks_dir = os.path.abspath(a.checks_dir or os.path.join(a.results_dir, "checks"))
    a.player_dir = os.path.join(a.results_dir, "player")
    default_ref = os.path.join(REPO, "results", "prereg02", "player")
    a.seam_ref = os.path.abspath(a.seam_ref or default_ref)
    if a.mode != "smoke" and not same_path(a.seam_ref, default_ref):
        die("--seam-ref may be changed only with --smoke (real, dry-run and rehearsal compare with prereg02)")
    a.seam_candidates = os.path.abspath(a.seam_candidates or os.path.join(REPO, *VAL_CANDIDATES))
    a.seam_questions = os.path.abspath(a.seam_questions or os.path.join(REPO, *VAL_QUESTIONS))
    a.frozen_path = os.path.join(a.seam_ref, "frozen_config.json")
    if (a.no_cnn or a.rerun) and not (a.reason or "").strip():
        die("--no-cnn and --rerun need --reason TEXT (it is stored in the report)")
    check_paths(a)
    return a


def check_paths(a):
    """Out-dir guard (spec 3.0) and the mode rules for where inputs and outputs may be."""
    for ro in (os.path.join(REPO, "results", "prereg01"), os.path.join(REPO, "results", "prereg02"),
               os.path.join(REPO, "data", "local")):
        if po.inside(a.out_dir, ro):
            die(f"refusing to write into {a.out_dir!r}: inside the read-only {ro!r}")
    for p in (a.results_dir, a.tests_dir, a.checks_dir, a.seam_ref, a.seam_candidates, a.seam_questions):
        po.refuse_forbidden(p)
    if a.mode == "real":
        if not same_path(a.results_dir, a.repo_results):
            die(f"real mode: --results-dir must be {a.repo_results} (got {a.results_dir})")
        if not same_path(a.tests_dir, os.path.join(REPO, "data", "test")):
            die(f"real mode: --tests-dir must be {os.path.join(REPO, 'data', 'test')} (got {a.tests_dir})")
        if not po.inside(a.out_dir, a.repo_results):
            die(f"real mode: --out-dir must be inside {a.repo_results}")
        if not same_path(a.checks_dir, os.path.join(a.repo_results, "checks")):
            die("real mode: --checks-dir must be results/test02/checks")
    elif po.inside_repo(a.out_dir):
        die(f"--{a.mode}: --out-dir {a.out_dir!r} is inside the repository; use a scratch directory")


def pool_files(a, pool):
    """(questions csv, candidates csv, dumps parent dir) of a pool."""
    if pool == "seam_val":
        q, c = a.seam_questions, a.seam_candidates
    else:
        qn, cn = TEST_FILES[pool]
        q, c = os.path.join(a.tests_dir, qn), os.path.join(a.tests_dir, cn)
    for p in (q, c):
        po.refuse_forbidden(p)
        if not os.path.exists(p):
            die(f"{pool}: input file missing: {p}")
    return q, c, os.path.join(a.player_dir, pool)


# ---------------------------------------------------------------- frozen prereg02 references
def load_frozen(a):
    """The frozen player config (the guard source) and its pins."""
    path = a.frozen_path
    if not os.path.exists(path):
        die(f"{path} missing")
    head = None
    if a.mode == "real":
        head = pf.check_committed(path)
    fz = read_json(path)
    info = {"path": disp(path), "sha256": po.sha256_text_file(path), "git_head": head}
    if a.mode != "smoke":
        if info["sha256"] != FROZEN_CONFIG_SHA256:
            die(f"frozen_config.json SHA-256 {info['sha256']} != pinned {FROZEN_CONFIG_SHA256}")
        gpath = os.path.join(a.seam_ref, "guard_calibration.json")
        gsha = po.sha256_text_file(gpath)
        if gsha != GUARD_CALIBRATION_SHA256 or (fz.get("guard") or {}).get("sha256") != GUARD_CALIBRATION_SHA256:
            die(f"guard_calibration.json SHA-256 {gsha} / frozen guard.sha256 "
                f"{(fz.get('guard') or {}).get('sha256')} != pinned {GUARD_CALIBRATION_SHA256}")
        if fz.get("smoke") is not False or (fz.get("guard") or {}).get("smoke") is not False:
            die("frozen_config.json or its guard is a smoke record")
        wsha = fz["inputs"]["cand_games_npz"].get("weights_sha256")
        if wsha != WEIGHTS_SHA256:
            die(f"frozen weights_sha256 {wsha} != pinned {WEIGHTS_SHA256}")
        info["guard_calibration_sha256"] = gsha
    if fz.get("guard") is not None and set(fz["guard"].get("thresholds", {})) != set(po.ORIENTS):
        die(f"{path}: guard has no thresholds for {list(po.ORIENTS)}")
    return fz, info


def check_frozen_tools(a, fz):
    now = po.source_shas()
    changed = [k for k, v in (fz.get("tools_sha256") or {}).items() if now.get(k) != v]
    if a.mode == "real":
        if changed:
            die(f"frozen player tools changed since select: {changed}")
        dirty = pf.tools_dirty()
        if dirty:
            die(f"uncommitted changes in the frozen player tools: {dirty}")
    elif changed and a.mode != "smoke":
        die(f"frozen player tools changed since select: {changed}")
    return {"now": now, "changed": changed}


# ---------------------------------------------------------------- the notebook's records
def notebook_state(player_dir):
    cfg = maybe_json(os.path.join(player_dir, "config.json"))
    rl = maybe_json(os.path.join(player_dir, "run_log.json"))
    runs = (rl or {}).get("runs") or []
    last = runs[-1].get("status") if runs else None
    top = (rl or {}).get("status")
    repro = maybe_json(os.path.join(player_dir, "repro_check.json"))
    seam = maybe_json(os.path.join(player_dir, "seam_p02.json"))
    earlier = [(r.get("run", i), r.get("status")) for i, r in enumerate(runs[:-1], start=1)
               if isinstance(r, dict) and r.get("status") in EARLIER_FAILED]
    return {"config": cfg, "config_sha256": (po.sha256_text_file(os.path.join(player_dir, "config.json"))
                                             if cfg is not None else None),
            "last_run_status": last, "run_log_status": top, "earlier_failed_runs": earlier,
            "completed": last == "completed" and top in (None, "completed"),
            "repro_passed": (repro or {}).get("passed"), "seam_p02": seam}


def check_notebook(a, nb, use_cnn):
    """config.json flags for the mode, run completed, reproduction passed (CNN runs only)."""
    cfg = nb["config"]
    problems = []
    if use_cnn:
        if cfg is None:
            problems.append("config.json missing")
        if not nb["completed"]:
            problems.append("notebook run not completed (last run %r, run_log status %r)"
                            % (nb["last_run_status"], nb["run_log_status"]))
        if nb["repro_passed"] is not True:
            problems.append("repro_check.json passed = %r" % nb["repro_passed"])
        if a.mode == "real" and nb.get("earlier_failed_runs"):
            problems.append("run_log.json has earlier failed runs %s: the notebook was run again in the same Drive "
                            "folder, which is not the registered rerun (PREREG_02 rev 5 9.2, rev 6 10.10; "
                            "docs/TESTDAY.md step 4)" % nb["earlier_failed_runs"])
    if cfg is not None:
        smoke, reh = cfg.get("smoke"), cfg.get("rehearsal")
        want = {"real": (False, False), "rehearsal": (False, True), "smoke": (True, None),
                "dry-run": (False, None)}[a.mode]
        if smoke is not want[0]:
            problems.append("config.json smoke = %r (this mode needs %r)" % (smoke, want[0]))
        if want[1] is not None and reh is not want[1]:
            problems.append("config.json rehearsal = %r (this mode needs %r)" % (reh, want[1]))
        w = (cfg.get("weights") or {}).get("sha256")
        if a.mode != "smoke" and w != WEIGHTS_SHA256:
            problems.append("config.json weights.sha256 %r != pinned %s" % (w, WEIGHTS_SHA256))
    ident = notebook_identity(cfg)
    if a.mode == "real" and cfg is not None and not ident["matches_committed_notebook"]:
        problems.append("config.json notebook / notebook_sha256 (%r, %r) is not the committed %s (%s): the Colab "
                        "run used another version of the notebook" % (cfg.get("notebook"), cfg.get("notebook_sha256"),
                                                                      NOTEBOOK_REL, ident["committed_source_sha256"]))
    if problems:
        die("notebook results in %s are not usable: %s" % (a.player_dir, "; ".join(problems)))
    return ident


def notebook_source_sha256(nb_path):
    """(source SHA-256, embedded value) of a test02 notebook: sha256 of json.dumps([[cell_type, source], ...],
    ensure_ascii=False, separators=(',', ':')) with the value of NB_SOURCE_SHA256 replaced by the placeholder
    (the rule colab/build_test02_player_nb.py uses). (None, None) when the value is not there exactly once."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]
    found = [m for _t, src in cells for m in NB_SHA_RE.findall(src)]
    if len(found) != 1:
        return None, None
    blob = json.dumps([[t, NB_SHA_RE.sub("NB_SOURCE_SHA256 = '%s'" % NB_SHA_PLACEHOLDER, src)] for t, src in cells],
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), found[0]


def notebook_identity(cfg):
    """Does the notebook record name the committed colab/test02_player.ipynb? (enforced in real mode)"""
    path = os.path.join(REPO, *NOTEBOOK_REL.split("/"))
    committed, embedded = notebook_source_sha256(path) if os.path.exists(path) else (None, None)
    got = (cfg or {}).get("notebook_sha256")
    return {"notebook": (cfg or {}).get("notebook"), "notebook_sha256": got,
            "committed_notebook": NOTEBOOK_REL, "committed_source_sha256": committed,
            "committed_embedded_sha256": embedded,
            "matches_committed_notebook": bool(committed and committed == embedded and got == committed
                                               and (cfg or {}).get("notebook") == NOTEBOOK_REL)}


def check_test_bundle(a, cfg, names, need_record):
    """The pool's test files: SHA-256 equal to data/test/MANIFEST.json (t02_test_bundle build) and to the notebook's
    config.json test_bundle.files[name] (spec 3.0); the MANIFEST equal to test_bundle.manifest_sha256. The notebook
    record is required for a CNN run (need_record) and checked whenever it exists (a --no-cnn run may have none)."""
    mpath = os.path.join(a.tests_dir, "MANIFEST.json")
    if not os.path.exists(mpath):
        die(f"{mpath} missing")
    man = read_json(mpath)
    ms = sha(mpath)
    tb = (cfg or {}).get("test_bundle")
    if need_record and not tb:
        die("the notebook's config.json has no test_bundle record")
    out = {"MANIFEST.json": ms, "notebook_record_checked": bool(tb)}
    for n in names:
        s = sha(os.path.join(a.tests_dir, n))
        m = ((man.get("files") or {}).get(n) or {}).get("sha256")
        if m != s:
            die(f"{n}: SHA-256 {s} differs from MANIFEST.json ({m})")
        if tb and (tb.get("files") or {}).get(n) != s:
            die(f"{n}: SHA-256 {s} differs from the notebook's config.json record {(tb.get('files') or {}).get(n)}")
        out[n] = s
    if tb and tb.get("manifest_sha256") != ms:
        die(f"MANIFEST.json SHA-256 {ms} differs from the notebook's record {tb.get('manifest_sha256')}")
    return out


def check_t1(a, required):
    path = os.path.join(a.checks_dir, "player", "t1.json")
    if not os.path.exists(path):
        if required:
            die(f"{path} missing: run t02_checks.py t1 --comp player first")
        return {"path": disp(path), "verdict": None, "checked": False}
    t1 = read_json(path)
    v = t1.get("verdict")
    if a.mode == "real":
        # the full T1 on the raw files (review finding 2026-09-26: a structure-only t1.json used to pass)
        mode = (t1.get("provenance") or {}).get("mode")
        if t1.get("complete") is not True or t1.get("stage") != "raw" or mode != "real":
            die(f"{path}: complete={t1.get('complete')!r} (ran {t1.get('subchecks_run')}), stage {t1.get('stage')!r}, "
                f"mode {mode!r}; real mode needs the full T1 (T1.1-T1.5) on data/test/raw: PY tools/t02_checks.py t1 "
                f"--comp player --rerun --reason TEXT (without --only)")
    if v not in T1_ALLOWED:
        die(f"T1 verdict {v!r} (need {', '.join(T1_ALLOWED)}); FAIL_STRUCT stops, see PREREG_02 rev 5 9.6")
    return {"path": disp(path), "sha256": po.sha256_text_file(path), "verdict": v,
            "normalize": t1.get("normalize"), "checked": True, "fail_opening": v == "FAIL_OPENING",
            "rule": ("PREREG_02 rev 5 9.6: T1.4 / T1.5 failed; competition 1 keeps the per-pool self-retrieval and "
                     "guard (they react on their own); T3 is 'not evaluable'") if v == "FAIL_OPENING" else None}


def check_inputs_record(a, pool, pdir, q, c, cfg):
    """_INPUT.json of the pool's dumps and config.json sets[pool] must name exactly these files."""
    qs, cs = sha(q), sha(c)
    rec = maybe_json(os.path.join(pdir, "dumps", "_INPUT.json"))
    if rec is None:
        die(f"{pool}: dumps/_INPUT.json missing in {pdir}")
    if rec.get("sha256") != qs or rec.get("candidates_sha256") != cs:
        die(f"{pool}: dumps/_INPUT.json (questions {rec.get('sha256')}, candidates "
            f"{rec.get('candidates_sha256')}) does not match {q} ({qs}) / {c} ({cs})")
    st = ((cfg or {}).get("sets") or {}).get(pool)
    if st is None or st.get("questions_sha256") != qs or st.get("candidates_sha256") != cs:
        die(f"{pool}: config.json sets[{pool!r}] does not record these question / candidate files")
    return {"questions_sha256": qs, "candidates_sha256": cs, "input_json": rec}


def real_mode_checks(a, use_cnn):
    """Spec 3.0: this tool and check_submission committed and clean; the notebook outputs of spec 2.3
    tracked by git and unchanged (all of them for a CNN run; with --no-cnn whatever the run left)."""
    git_tracked_clean([os.path.join(REPO, *r.split("/")) for r in (TOOL_REL, CHECK_REL, NOTEBOOK_REL)],
                      "tools/t02_player.py, tools/check_submission.py and colab/test02_player.ipynb")
    files = [os.path.join(a.player_dir, n) for n in NB_FILES]
    files += [os.path.join(a.player_dir, p, "dumps", n) for p in POOLS for n in DUMP_FILES]
    if use_cnn:
        missing = [disp(f) for f in files if not os.path.exists(f)]
        if missing:
            die("notebook outputs missing: %s" % missing[:6])
    files = [f for f in files if os.path.exists(f)]
    if files:
        git_tracked_clean(files, "notebook outputs in results/test02/player")
    return {"files_tracked_and_clean": len(files)}


def no_cnn_basis(a, pool, nb):
    """PREREG_02 rev 4 item 4: the fallback list {Q22, Q23} only when the pool's CNN dumps are missing
    or the notebook's Q00 reproduction failed (the 7-day window is recorded by Claude in RESULTS)."""
    dd = os.path.join(a.player_dir, pool, "dumps")
    missing = not all(os.path.exists(os.path.join(dd, k)) for k in ("cand_games.npz", "query_games.npz"))
    basis = {"dumps_missing": missing, "repro_passed": nb["repro_passed"],
             "last_run_status": nb["last_run_status"],
             "rule": "rev 4 item 4: CNN dumps missing, or the notebook's Q00 reproduction failed"}
    basis["accepted"] = bool(missing or nb["repro_passed"] is False)
    if a.mode in ("real", "rehearsal") and not basis["accepted"]:
        die("--no-cnn refused: the dumps are there and the reproduction did not fail (%s)" % json.dumps(basis))
    return basis


# ---------------------------------------------------------------- dumps vs prereg02 (seam)
def dumps_bitwise(pool_dumps, ref_dumps):
    """Array-by-array comparison of the seam_val dumps with prereg02's (npz bytes carry zip timestamps)."""
    out = {"reference": disp(ref_dumps), "files": {}}
    ok = True
    for n in ("cand_games.npz", "query_games.npz"):
        pa, pb = os.path.join(pool_dumps, n), os.path.join(ref_dumps, n)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            out["files"][n] = {"compared": False}
            ok = False
            continue
        with np.load(pa, allow_pickle=False) as za, np.load(pb, allow_pickle=False) as zb:
            keys = sorted(set(za.files) | set(zb.files))
            eq = {k: bool(k in za.files and k in zb.files and za[k].dtype == zb[k].dtype
                          and za[k].shape == zb[k].shape and np.array_equal(za[k], zb[k])) for k in keys}
            diff = None
            if "emb_sum" in za.files and "emb_sum" in zb.files and za["emb_sum"].shape == zb["emb_sum"].shape:
                diff = float(np.max(np.abs(za["emb_sum"].astype(np.float64) - zb["emb_sum"].astype(np.float64)))
                             ) if za["emb_sum"].size else 0.0
        out["files"][n] = {"compared": True, "arrays_equal": eq, "bitwise_equal": all(eq.values()),
                           "max_abs_emb_diff": diff, "sha256": sha(pa), "reference_sha256": sha(pb)}
        ok = ok and all(eq.values())
    out["bitwise_equal"] = ok
    return out


# ---------------------------------------------------------------- the per-pool procedure
def select(d, combos, strict):
    """cmd_select's numeric part (p02_player_fuse.py:762-797) on one pool."""
    try:
        rounds, per_round, max_cos, n_queries, n_missing = pf.self_retrieval(d, combos, strict=strict,
                                                                             log=lambda s: None)
    except ValueError as e:
        die("self-retrieval refused (%s). PREREG_02 rev 5 9.4: a candidate without exactly 100 games stops "
            "competition 1 until a revision is written" % e)
    means, rows = {}, []
    for c in combos:
        allp = [x for rr in per_round[c["id"]] for x in rr]
        means[c["id"]] = pf._stats(allp)
        rmeans = [pf._stats(rr) for rr in per_round[c["id"]]]
        pts = np.asarray(allp)
        rows.append({"id": c["id"], "e": c["e"], "o": c["o"], "w": c["w"], "mean": means[c["id"]],
                     "round_means": rmeans,
                     "round_sd": float(np.std([m for m in rmeans if np.isfinite(m)], ddof=1))
                     if sum(np.isfinite(rmeans)) > 1 else None,
                     "top1_share": float((pts == 1.0).mean()) if len(pts) else None,
                     "top5_share": float((pts > 0).mean()) if len(pts) else None})
    order = [c["id"] for c in combos]
    best_id, chosen = pf.choose(means, order)
    q00 = means.get("Q00")
    if q00 is not None:
        gate = {"q00_mean": q00, "selected_mean": means[chosen], "selected_minus_q00": means[chosen] - q00,
                "threshold": pf.GATE_MIN, "passed": bool(means[chosen] - q00 >= pf.GATE_MIN)}
    else:
        gate = {"q00_mean": None, "selected_mean": means[chosen], "selected_minus_q00": None,
                "threshold": pf.FALLBACK_GATE_MIN, "passed": bool(means[chosen] >= pf.FALLBACK_GATE_MIN),
                "rule": "PREREG_02 rev 3 7.7: no Q00 (CNN dumps not used), gate = selected mean >= %.2f"
                        % pf.FALLBACK_GATE_MIN}
    sr_med = {o: (float(np.median(max_cos[o])) if max_cos[o] else None) for o in po.ORIENTS}
    return {"rounds": rounds, "per_round": per_round, "means": means, "rows": rows, "best_id": best_id,
            "selected": chosen, "gate": gate, "sr_med": sr_med, "n_queries": n_queries, "n_missing": n_missing,
            "n_with_tokens": {o: len(max_cos[o]) for o in po.ORIENTS}}


def effective_combo(sel_id, gate, use_cnn):
    """PREREG_02 rev 5 9.4 (D1): at test time a failed gate means Q00 for that pool."""
    if gate["passed"]:
        return pf.COMBO_BY_ID[sel_id], "gate passed: the selected combination"
    if use_cnn:
        return pf.COMBO_BY_ID["Q00"], "gate failed: Q00 (PREREG_02 rev 5 9.4, a submission is mandatory)"
    return pf.COMBO_BY_ID[sel_id], ("gate failed with the opening-only fallback list: no Q00 exists, the "
                                    "selection is kept (a submission is mandatory; not covered by rev 5)")


def sr_top1_distinct(d, rounds, combo):
    """Distinct top-1 candidates per self-retrieval round under `combo` (record-only, for T6)."""
    row = {(g[1], g[0]): i for i, g in enumerate(d.cgames)}
    c_arr = (d.cemb, d.cnw, d.cok) if d.use_cnn else None
    out = []
    for items in rounds:
        qi = [i for i, (_p, q, _r) in enumerate(items) if q]
        if not qi:
            out.append(0)
            continue
        c_groups = [[row[(p, g)] for g in refs] for p, _q, refs in items]
        q_groups = [[row[(items[i][0], g)] for g in items[i][1]] for i in qi]
        terms = {}
        if combo["o"] is not None:
            G = d.G(combo["o"])[0]
            terms[combo["o"]] = po.open_cos(G, c_groups, G, q_groups)
        if combo["e"] is not None:
            terms[combo["e"]] = pf.e_scores(combo["e"], c_arr, c_groups, c_arr, q_groups)[:2]
        S = pf.fuse(combo, terms)
        out.append(len({t[0] for t in pf.top_ids(S, list(range(S.shape[1])), n=1)}))
    return out


def apply(d, effective, guard, sr_med, use_cnn):
    """cmd_apply's numeric part (p02_player_fuse.py:942-969) with this pool's selection and denominator."""
    c_groups = [d.c_by_player[p] for p in d.cand_ids]
    q_groups = [d.q_by_id[q] for q in d.qids]
    terms, rho, num = {}, {}, {}
    for o in po.ORIENTS:
        gc, gq = d.G(o)
        cos, has = po.open_cos(gc, c_groups, gq, q_groups)
        terms[o] = (cos, has)
        mx = cos.max(1)[has] if len(cos) else np.zeros(0)
        num[o] = float(np.median(mx)) if len(mx) else None
        den = sr_med.get(o)
        rho[o] = num[o] / den if (num[o] is not None and den) else None
    n_c_missing = {}
    if use_cnn:
        c_arr, q_arr = (d.cemb, d.cnw, d.cok), (d.qemb, d.qnw, d.qok)
        for k in pf.E_KINDS:
            M, has, n_c_missing[k] = pf.e_scores(k, c_arr, c_groups, q_arr, q_groups)
            terms[k] = (M, has)
    b_c = po.colour_share(g[2] for g in d.cgames)
    b_q = po.colour_share(g[2] for g in d.qgames)
    applied, level, reason = pf.guard_decision(effective, rho, guard, b_c, b_q, use_cnn)
    S = pf.fuse(applied, terms)
    tops = pf.top_ids(S, d.cand_ids)
    fb = pf.all_missing(applied, terms)
    for k in np.flatnonzero(fb):
        if tops[k] != d.cand_ids[:pf.TOP_N]:
            raise AssertionError("fallback question not filled with the first sorted candidates")
    return {"tops": tops, "applied": applied, "level": level, "reason": reason, "rho": rho, "num": num,
            "b_c": b_c, "b_q": b_q, "terms": terms, "n_c_missing": n_c_missing, "fallback": fb}


def check_question_order(d, q_path, smoke):
    """Rows are written in question-file order (spec 3.3 step 4)."""
    file_ids = question_ids(q_path)
    if smoke:
        keep = set(d.qids)
        want = [q for q in file_ids if q in keep]
    else:
        want = file_ids
    if d.qids != want:
        die("question order differs from the question file %s" % q_path)
    return len(file_ids)


def check_pool_prefixes(pool, d):
    """Only the pool's own ids: pub_/priv_ questions, pub_p_/priv_p_ players, pub_/priv_ games."""
    if pool not in PREFIX:
        return
    qp, pp, gp = PREFIX[pool]
    bad = ([q for q in d.qids if not q.startswith(qp)][:3] + [p for p in d.cand_ids if not p.startswith(pp)][:3]
           + [g[0] for g in d.cgames if not g[0].startswith(gp)][:3])
    if bad:
        die(f"{pool}: ids without the pool's prefix, e.g. {bad}")


# ---------------------------------------------------------------- seam rule
def seam_rule(a, sel, app, tops, qids, top5_path, use_cnn, bitwise):
    ref = a.seam_ref
    sr_path, top_path = os.path.join(ref, "sr_report.json"), os.path.join(ref, "player_val_top5.csv")
    ref_sr = read_json(sr_path)
    ref_fz = read_json(a.frozen_path)
    ref_app = read_json(os.path.join(ref, "apply_report.json"))
    pins = {"sr_report_sha256": po.sha256_text_file(sr_path), "top5_sha256": po.sha256_text_file(top_path)}
    if a.mode != "smoke":
        if pins["sr_report_sha256"] != SR_REPORT_SHA256 or pins["top5_sha256"] != VAL_TOP5_SHA256:
            die("prereg02 sr_report.json / player_val_top5.csv differ from the pinned SHA-256 (%s)" % pins)
    ref_means = {r["id"]: r["mean"] for r in ref_sr["combinations"]}
    diffs = {c: (abs(sel["means"][c] - ref_means[c]) if c in ref_means else None) for c in sel["means"]}
    finite = [v for v in diffs.values() if v is not None and math.isfinite(v)]
    means_exact = all(c in ref_means and sel["means"][c] == ref_means[c] for c in sel["means"])
    open_exact = all(sel["means"][c] == ref_means.get(c) for c in OPEN_ONLY if c in sel["means"])
    ref_med = ref_fz.get("sr_max_cos_median") or {}
    med_exact = all(sel["sr_med"][o] == ref_med.get(o) for o in po.ORIENTS)
    ref_rows = {r[0]: r[1:] for r in read_top5(top_path)}
    n_ident = sum(1 for q, t in zip(qids, tops) if ref_rows.get(q) == t)
    same_q = [q for q in qids] == list(ref_rows)
    with open(top5_path, "rb") as f1, open(top_path, "rb") as f2:
        byte_ident = f1.read().replace(b"\r\n", b"\n") == f2.read().replace(b"\r\n", b"\n")
    n = len(qids)
    if a.mode == "smoke":       # the reference is a prereg02 smoke run: its own selection, gate and level
        want_sel, want_gate = ref_fz["selected"]["id"], (ref_fz.get("gate") or {}).get("passed")
        want_level, need = ref_app["guard_level"], n
    else:
        want_sel, want_gate, want_level, need = SEAM_COMBO, True, SEAM_LEVEL, n - SEAM_MAX_DIFFERENT
    crit = {"opening_only_means_exact": open_exact, "sr_max_cos_median_exact": med_exact}
    added = ["opening_only_means_exact", "sr_max_cos_median_exact"]
    if use_cnn:
        crit.update({"selected_is_%s" % want_sel: sel["selected"] == want_sel,
                     ("gate_passed" if want_gate else "gate_as_reference"): sel["gate"]["passed"] is want_gate,
                     "guard_level_%d" % want_level: app["level"] == want_level,
                     "same_questions": same_q,
                     "top5_identical_at_least_%d" % need: n_ident >= need,
                     "expected_identical_when_dumps_bitwise": (not bitwise) or (means_exact and byte_ident)})
        added.append("expected_identical_when_dumps_bitwise")
    failed_added = [k for k in added if not crit[k]]
    return {
        "what": "PREREG_02 rev 5 9.2 local seam (competition 1) on the validation pool",
        "reference": {"dir": disp(ref), **pins},
        "registered_rule": ("self-retrieval selects Q04, gate passed, guard level 1, top-5 identical to "
                            "player_val_top5.csv on >= 398/400 questions" if use_cnn else
                            "--no-cnn: opening-only; the registered CNN seam does not apply"),
        "added_rule": ("opening-only means (Q22, Q23) and the self-retrieval medians equal prereg02 exactly "
                       "(they do not depend on the dumps); when the seam_val dumps are bitwise equal to "
                       "prereg02's, all means and the top-5 file must be identical"),
        "dumps_bitwise_equal": bitwise,
        "means_exact": means_exact,
        "means_max_abs_diff": max(finite) if finite else None,
        "means_abs_diff": diffs,
        "sr_max_cos_median": sel["sr_med"],
        "sr_max_cos_median_reference": ref_med,
        "selected": sel["selected"],
        "gate": sel["gate"],
        "guard_level": app["level"],
        "n_questions": n,
        "n_top5_identical": n_ident,
        "byte_identical_top5": byte_ident,
        "criteria": crit,
        "passed": all(crit.values()),
        "failed_implementation_additions": failed_added,
        "environment": lib_versions(),
        "environment_note": ("the implementation additions (exact equality of the opening-only means, the "
                             "self-retrieval medians and, with bitwise dumps, all means and the top-5 file) do not "
                             "depend on Colab; when only they fail, look for a local bug or a numpy / scipy change "
                             "in .venv (versions above; prereg02 ran with the same .venv) before anything else"),
    }


# ---------------------------------------------------------------- outputs and rerun
def guard_existing(a, paths):
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not a.rerun:
        die("outputs exist (%s); use --rerun --reason TEXT to overwrite" % ", ".join(disp(p) for p in existing))
    if not a.rerun:
        return None
    return {"reason": a.reason, "previous_sha256": {disp(p): sha(p) for p in existing}}


def cmd_run(a):
    t0 = time.time()
    pool = a.pool
    use_cnn = not a.no_cnn
    smoke = a.mode == "smoke"
    q_path, c_path, pdir = pool_files(a, pool)
    odir = os.path.join(a.out_dir, pool)
    outs = [os.path.join(odir, n) for n in ("pool_select.json", "top5.csv", "apply_report.json")]
    if pool == "seam_val":
        outs.append(os.path.join(odir, "seam_local.json"))
    rerun = guard_existing(a, outs)

    fz, frozen_info = load_frozen(a)
    tools = check_frozen_tools(a, fz)
    nb = notebook_state(a.player_dir)
    ident = check_notebook(a, nb, use_cnn)
    basis = no_cnn_basis(a, pool, nb) if a.no_cnn else None
    t1 = check_t1(a, required=a.mode != "smoke")
    real = real_mode_checks(a, use_cnn) if a.mode == "real" else None
    names = TEST_FILES[pool] if pool in TEST_FILES else ()
    bundle = check_test_bundle(a, nb["config"], names, need_record=use_cnn)
    inputs_rec = check_inputs_record(a, pool, pdir, q_path, c_path, nb["config"]) if use_cnn else None

    seam = None
    if pool != "seam_val":
        sp = os.path.join(a.out_dir, "seam_val", "seam_local.json")
        if not os.path.exists(sp):
            die(f"run --pool seam_val first ({sp} missing)")
        seam = read_json(sp)
        if seam.get("passed") is not True:
            die("the seam did not pass (%s): no test pool is run (PREREG_02 rev 5 9.2)" % sp)
        if seam.get("mode") != a.mode or seam.get("cnn") != use_cnn:
            die("seam_local.json was made in mode %r / cnn %r, this run is %r / %r"
                % (seam.get("mode"), seam.get("cnn"), a.mode, use_cnn))
        if seam.get("notebook_config_sha256") != nb["config_sha256"]:
            die("the notebook's config.json changed since the seam run")
        seam = {"path": disp(sp), "sha256": po.sha256_text_file(sp), "passed": True}

    # select (candidate file + cand_games.npz only)
    combos = pf.COMBOS if use_cnn else [pf.COMBO_BY_ID[c] for c in pf.FALLBACK_IDS]
    ds = pf.Data(c_path, pdir, use_cnn, smoke)
    if use_cnn:
        check_dump_weights(a, ds.weights, nb["config"])
    check_pool_prefixes(pool, ds)
    t_sel = time.time()
    sel = select(ds, combos, strict=not smoke)
    t_sel = time.time() - t_sel
    effective, eff_why = effective_combo(sel["selected"], sel["gate"], use_cnn)
    guard = fz.get("guard")

    # apply (full candidate games, the question file)
    t_app = time.time()
    da = pf.Data(c_path, pdir, use_cnn, smoke, q_path=q_path)
    if use_cnn:
        check_dump_weights(a, da.weights, nb["config"])
    if len(da.cand_ids) < pf.TOP_N:
        die(f"only {len(da.cand_ids)} candidates; need at least {pf.TOP_N}")
    check_pool_prefixes(pool, da)
    n_file_questions = check_question_order(da, q_path, smoke)
    app = apply(da, effective, guard, sel["sr_med"], use_cnn)
    t_app = time.time() - t_app
    sr_top1 = sr_top1_distinct(ds, sel["rounds"], app["applied"])

    os.makedirs(odir, exist_ok=True)
    top5_path = os.path.join(odir, "top5.csv")
    write_csv_lf(top5_path, ["question_id"] + pf.TOP_COLS, [[q] + t for q, t in zip(da.qids, app["tops"])])
    prov = provenance()
    ins = {k: {"path": disp(da.paths[k]), "sha256": v} for k, v in da.shas.items()}
    common = {"pool": pool, "mode": a.mode, "cnn": use_cnn, "no_cnn_reason": a.reason if a.no_cnn else None,
              "no_cnn_basis": basis, "rerun": rerun, "frozen_config": frozen_info,
              "frozen_tools": tools, "t1": t1, "real_mode_checks": real, "test_bundle": bundle,
              "notebook": {"config_sha256": nb["config_sha256"], "identity": ident,
                           "last_run_status": nb["last_run_status"],
                           "repro_passed": nb["repro_passed"],
                           "seam_p02_bitwise_equal": (nb["seam_p02"] or {}).get("bitwise_equal")},
              "input_record": inputs_rec, "seam": seam, "inputs": ins, "provenance": prov}
    pool_select = dict(common, **{
        "what": "PREREG_02 rev 5 9.4: self-retrieval selection on this pool (cmd_select's steps)",
        "split": "as prereg02 sr_report.json (sha256 order, 5 parts, 10 rounds, k = 5..20)",
        "n_candidates": len(ds.cand_ids), "n_candidate_games": len(ds.cgames), "rounds": po.SR_ROUNDS,
        "pseudo_queries_per_round": sel["n_queries"], "pseudo_queries_missing_term": sel["n_missing"],
        "combinations": sel["rows"], "self_retrieval_mean": sel["means"],
        "best": {"id": sel["best_id"], "mean": sel["means"][sel["best_id"]]},
        "selected": pf.COMBO_BY_ID[sel["selected"]], "gate": sel["gate"],
        "effective": effective, "effective_rule": eff_why,
        "sr_max_cos_median": sel["sr_med"], "n_pseudo_queries_with_tokens": sel["n_with_tokens"],
        "guard": guard, "runtime_s": round(t_sel, 1)})
    fb = app["fallback"]
    apply_report = dict(common, **{
        "what": "PREREG_02 rev 5 9.4: this pool's combination on the full candidate pool and its question "
                "file, with the guard (cmd_apply's steps; rho denominator = this pool's sr_max_cos_median)",
        "selected": pf.COMBO_BY_ID[sel["selected"]], "gate_passed": sel["gate"]["passed"],
        "effective": effective, "effective_rule": eff_why, "applied": app["applied"],
        "guard_level": app["level"], "guard_reason": app["reason"],
        "guard_changed_combination": app["applied"]["id"] != effective["id"],
        "b_c_candidate_black_share": app["b_c"], "b_q_question_black_share": app["b_q"],
        "rho": app["rho"], "rho_numerator_max_cos_median": app["num"],
        "rho_denominator_sr_max_cos_median": sel["sr_med"], "guard": guard,
        "n_questions": len(da.qids), "n_questions_in_file": n_file_questions,
        "n_candidates": len(da.cand_ids), "n_candidate_games": len(da.cgames),
        "n_question_games": len(da.qgames),
        "n_questions_without_term": {k: int((~np.asarray(v[1], bool)).sum()) for k, v in app["terms"].items()},
        "n_candidates_without_windows": app["n_c_missing"], "n_fallback_questions": int(fb.sum()),
        "n_distinct_top1": len({t[0] for t in app["tops"]}),
        "sr_top1_distinct_per_round": sr_top1,
        "sr_top1_distinct_rule": "record-only (T6): distinct top-1 candidates per self-retrieval round under "
                                 "the applied combination",
        "dump_weights_sha256": da.weights,
        "output": {"path": disp(top5_path), "sha256": sha(top5_path), "rows": len(da.qids)},
        "submit_allowed": True,
        "submit_rule": "PREREG_02 rev 5 9.4: every pool is submitted (gate failed -> Q00; fallback list at "
                       "level 3 keeps its selection)",
        "runtime_s": round(t_app, 1)})
    po.write_json(os.path.join(odir, "pool_select.json"), pool_select)
    po.write_json(os.path.join(odir, "apply_report.json"), apply_report)
    msg = ("%s: selected %s (best %s %.6f; Q00 %s) gate %s -> effective %s -> applied %s, guard level %d "
           "(rho %s), %d questions, %d fallback"
           % (pool, sel["selected"], sel["best_id"], sel["means"][sel["best_id"]],
              "%.6f" % sel["means"]["Q00"] if "Q00" in sel["means"] else "n/a", sel["gate"]["passed"],
              effective["id"], app["applied"]["id"], app["level"],
              {o: (round(v, 4) if v is not None else None) for o, v in app["rho"].items()},
              len(da.qids), int(fb.sum())))
    if t1.get("fail_opening"):
        msg += "\nT1 is FAIL_OPENING: the per-pool procedure is kept (PREREG_02 rev 5 9.6); recorded in the reports"
    if pool == "seam_val":
        bit = None
        if use_cnn:
            ref_dumps = os.path.join(a.seam_ref, "dumps")
            if os.path.isdir(ref_dumps):
                cmp_ = dumps_bitwise(os.path.join(pdir, "dumps"), ref_dumps)
                if a.mode != "smoke":
                    for n, s in P02_DUMP_SHA256.items():
                        if cmp_["files"].get(n, {}).get("reference_sha256") not in (None, s):
                            die(f"prereg02 dumps/{n} differs from the pinned SHA-256 {s}")
                bit = cmp_["bitwise_equal"]
            else:
                cmp_ = {"reference": None, "note": "no reference dumps; the notebook's seam_p02.json is used"}
                bit = (nb["seam_p02"] or {}).get("bitwise_equal") is True
        else:
            cmp_ = {"note": "--no-cnn: no dump is read"}
        rule = seam_rule(a, sel, app, app["tops"], da.qids, top5_path, use_cnn, bool(bit))
        seam_out = dict(rule, **{"mode": a.mode, "cnn": use_cnn, "dump_comparison": cmp_,
                                 "notebook_config_sha256": nb["config_sha256"],
                                 "notebook_seam_p02": nb["seam_p02"],
                                 "outputs": {"top5_sha256": sha(top5_path)}, "rerun": rerun, "provenance": prov})
        po.write_json(os.path.join(odir, "seam_local.json"), seam_out)
        msg += "\nseam %s: %s" % ("PASSED" if seam_out["passed"] else "FAILED",
                                  json.dumps(seam_out["criteria"]))
        if seam_out["failed_implementation_additions"]:
            msg += ("\nseam: the failed exact-equality checks %s do not depend on the Colab dumps: look for a local "
                    "bug or a change of numpy / scipy in .venv (%s)"
                    % (seam_out["failed_implementation_additions"],
                       {k: v.split()[0] for k, v in seam_out["environment"].items()}))
        print(msg)
        print("wrote", odir, "(%.1f s)" % (time.time() - t0))
        if not seam_out["passed"]:
            print("ERROR: the local seam failed: no test pool is run and nothing from this run is uploaded (PREREG_02 "
                  "rev 5 9.2). A rerun follows docs/TESTDAY.md step 4 (archive results/test02/player; the user renames "
                  "the Drive folder to player_failed_1 before running the notebook again)", file=sys.stderr)
            sys.exit(1)
        return
    print(msg)
    print("wrote", odir, "(%.1f s)" % (time.time() - t0))


def check_dump_weights(a, weights, cfg):
    w = [v for v in weights.values()]
    if a.mode == "smoke":
        want = ((cfg or {}).get("weights") or {}).get("sha256")
    else:
        want = WEIGHTS_SHA256
    if any(v != want for v in w):
        die(f"dump weights_sha256 {weights} != {want}")


# ---------------------------------------------------------------- submit
def cmd_submit(a):
    t0 = time.time()
    smoke = a.mode == "smoke"
    sub = os.path.join(a.out_dir, SUBMISSION)
    rep_path = os.path.join(a.out_dir, "submit_report.json")
    rerun = guard_existing(a, [sub, rep_path])
    fz, frozen_info = load_frozen(a)
    tools = check_frozen_tools(a, fz)
    t1 = check_t1(a, required=not smoke)
    nb = notebook_state(a.player_dir)
    seam_path = os.path.join(a.out_dir, "seam_val", "seam_local.json")
    if not os.path.exists(seam_path) or read_json(seam_path).get("passed") is not True:
        die(f"{seam_path} missing or not passed")
    seam_sha = po.sha256_text_file(seam_path)
    pools, rows, qfiles, cfiles = {}, [], [], []
    for pool in TEST_POOLS:
        q_path, c_path, _pdir = pool_files(a, pool)
        odir = os.path.join(a.out_dir, pool)
        need = [os.path.join(odir, n) for n in ("pool_select.json", "apply_report.json", "top5.csv")]
        missing = [p for p in need if not os.path.exists(p)]
        if missing:
            die(f"run --pool {pool} first ({missing})")
        ps, ar = read_json(need[0]), read_json(need[1])
        if ar.get("mode") != a.mode or ps.get("mode") != a.mode:
            die(f"{pool}: reports were made in mode {ar.get('mode')!r}, this is {a.mode!r}")
        if sha(need[2]) != ar["output"]["sha256"]:
            die(f"{pool}: top5.csv changed after run (SHA-256 differs from apply_report.json)")
        if (ar.get("seam") or {}).get("sha256") != seam_sha:
            die(f"{pool}: was run under another seam_local.json than the current one; rerun the pool")
        now_in = {"candidates_csv": sha(c_path), "questions_csv": sha(q_path)}
        for k, v in now_in.items():
            if ar["inputs"].get(k, {}).get("sha256") != v:
                die(f"{pool}: {k} changed since run --pool {pool}")
        check_test_bundle(a, nb["config"], TEST_FILES[pool], need_record=bool(ar.get("cnn")))
        body = read_top5(need[2])
        file_ids = question_ids(q_path)
        got = [r[0] for r in body]
        if not smoke and got != file_ids:
            die(f"{pool}: top5.csv rows are not in question-file order")
        if smoke and got != [q for q in file_ids if q in set(got)]:
            die(f"{pool}: top5.csv rows are not in question-file order")
        rows += body
        qfiles.append(q_path)
        cfiles.append(c_path)
        pools[pool] = {"selected": ar["selected"]["id"], "gate_passed": ar["gate_passed"],
                       "effective": ar["effective"]["id"], "applied": ar["applied"]["id"],
                       "guard_level": ar["guard_level"], "guard_reason": ar["guard_reason"], "rho": ar["rho"],
                       "b_c": ar["b_c_candidate_black_share"], "b_q": ar["b_q_question_black_share"],
                       "n_fallback": ar["n_fallback_questions"], "cnn": ar["cnn"], "rows": len(body),
                       "top5": {"path": disp(need[2]), "sha256": ar["output"]["sha256"]},
                       "pool_select_sha256": po.sha256_text_file(need[0]),
                       "apply_report_sha256": po.sha256_text_file(need[1]),
                       "inputs": {"questions": {"path": disp(q_path), "sha256": now_in["questions_csv"]},
                                  "candidates": {"path": disp(c_path), "sha256": now_in["candidates_csv"]}}}
    real = real_mode_checks(a, any(x["cnn"] for x in pools.values())) if a.mode == "real" else None
    os.makedirs(a.out_dir, exist_ok=True)
    write_csv_lf(sub, ["question_id"] + pf.TOP_COLS, rows)
    check_q = qfiles
    if smoke:       # SMOKE dumps cover a few questions: check against subset question files (prereg02 smoke practice)
        check_q = smoke_subset_questions(a.out_dir, qfiles, {r[0] for r in rows})
    errors = check_submission.check("player", sub, check_q, cfiles, official=not smoke)
    report = {"what": "PREREG_02 rev 5 9.5: submission_player.csv = public rows then private rows (question-file "
                      "order), question_id,top_1..top_5, UTF-8 without BOM, LF",
              "mode": a.mode, "pools": pools, "t1": t1, "frozen_config": frozen_info, "frozen_tools": tools,
              "real_mode_checks": real, "rerun": rerun, "seam_local": {"path": disp(seam_path), "sha256": seam_sha},
              "check_submission": {"official": not smoke, "errors": errors[:50], "n_errors": len(errors),
                                   "passed": not errors,
                                   "question_files": [{"path": disp(q), "sha256": sha(q)} for q in check_q]},
              "provenance": provenance()}
    if errors:
        os.remove(sub)
        report["output"] = None
        po.write_json(rep_path, report)
        die("check_submission found %d problems (first: %s); %s removed" % (len(errors), errors[0], SUBMISSION))
    with open(sub, "rb") as f:
        raw = f.read()
    report["output"] = {"path": disp(sub), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows),
                        "bytes": len(raw)}
    report["runtime_s"] = round(time.time() - t0, 1)
    po.write_json(rep_path, report)
    print("%s: %d rows (public %d, private %d), sha256 %s -> %s" % (
        SUBMISSION, len(rows), pools["public"]["rows"], pools["private"]["rows"],
        report["output"]["sha256"][:12], sub))


def smoke_subset_questions(out_dir, qfiles, keep):
    """--smoke only: copies of the question files holding just the questions the SMOKE dumps covered."""
    d = os.path.join(out_dir, "smoke_subset")
    os.makedirs(d, exist_ok=True)
    out = []
    csv.field_size_limit(10 ** 9)
    for q in qfiles:
        with open(po.refuse_forbidden(q), newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            fields, rows = rd.fieldnames, [r for r in rd if r["question_id"] in keep]
        p = os.path.join(d, os.path.basename(q))
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        out.append(p)
    return out


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        m = p.add_mutually_exclusive_group()
        m.add_argument("--dry-run", action="store_true", help="fake tree; output outside the repository")
        m.add_argument("--smoke", action="store_true", help="notebook SMOKE outputs; output outside the repository")
        m.add_argument("--rehearsal", action="store_true", help="Colab rehearsal outputs; output outside the repository")
        p.add_argument("--results-dir", default=None, help="notebook results root (default results/test02)")
        p.add_argument("--tests-dir", default=None, help="test files (default data/test)")
        p.add_argument("--out-dir", default=None, help="default results/test02/player (real mode)")
        p.add_argument("--checks-dir", default=None, help="t02_checks outputs (default RESULTS_DIR/checks)")
        p.add_argument("--seam-ref", default=None, help="prereg02 player dir (only --smoke may change it)")
        p.add_argument("--seam-candidates", default=None, help="default data/local/player_val_candidates.csv")
        p.add_argument("--seam-questions", default=None, help="default data/local/player_val_questions.csv")
        p.add_argument("--rerun", action="store_true", help="overwrite existing outputs (needs --reason)")
        p.add_argument("--reason", default=None)

    r = sub.add_parser("run", help="one pool: self-retrieval, gate, guard, apply")
    common(r)
    r.add_argument("--pool", required=True, choices=POOLS)
    r.add_argument("--no-cnn", action="store_true", help="registered fallback list {Q22, Q23} (needs --reason)")
    s = sub.add_parser("submit", help="public + private -> submission_player.csv")
    common(s)
    s.set_defaults(no_cnn=False, pool=None)
    a = resolve(ap.parse_args(argv))
    {"run": cmd_run, "submit": cmd_submit}[a.cmd](a)


if __name__ == "__main__":
    main()
