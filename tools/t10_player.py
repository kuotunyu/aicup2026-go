"""C10 test10, competition 1 (player identification, TEAM_11249): the 10 test-day tool (PREREG_10 parts A and B).

Registration: docs/PREREG_10_PLAYER_GUARD_P1.md sections 1, 2, 4 and 8 (rev 3, section 12: the test-day revision of A;
rev 4, section 13: the test-day revision of B). Claude's steps: docs/TESTDAY_10.md. Part B (the KataGo P1 LDA term,
tools/t10_b.py) runs only once B is ADOPTED (results/prereg10/RESULTS.md marks H10P adopted, score_H10P.json and the
freeze committed); until then every B hook refuses (exit 3, "B not adopted", nothing written) and every 10 is 10A or
06-OFF / test02 rows, exactly as rev 3.

    PY tools/t10_player.py seam                         [MODE] [DIRS] [--rerun --reason TEXT]
    PY tools/t10_player.py apply --pool public|private  [MODE] [DIRS] [--rerun --reason TEXT]
    PY tools/t10_player.py submit                       [MODE] [DIRS] [--rerun --reason TEXT]
    PY tools/t10_player.py status                       [MODE] [DIRS]
    .venv-gpu/Scripts/python.exe tools/t10_player.py p1 --pool seam_val|public|private  [MODE] [DIRS] [B]
                                                        (part B, 4090; exit 3 while B is not adopted)
    .venv-gpu/Scripts/python.exe tools/t10_player.py p1-rehearse   [MODE] [B]  (part B: tol_P, before test data)

MODE  none = real; --dry-run = a fake tree (tools/t10_fake_tests.py, tools/t06_fake_tests.py) with every output
      outside the repository. --now ISO (dry-run, and status) replaces the clock for K6-10.
DIRS  --out-dir O10         10 reports and top-5 files (real: results/test10/player)
      --t06-out-dir O6      06 reports (real: results/test06/player; read only)
      --t06-data-dir A6     06 feature caches (real: data/test06/player; read only)
      --t02-results-dir D2  test02 notebook root (real: results/test02)
      --t02-out-dir O2      test02's t02_player.py outputs (real: results/test02/player)
      --tests-dir T         the checked test files (real: data/test; never data/test/raw)
      --checks-dir C        T1 / T2 / T4 (reads C/player/{t1,t2,t4}.json; default D2/checks)
      --log L               the upload log (real: results/official_scores.json)
      --prereg06-dir P6     the frozen 06 outputs (real: results/prereg06/player)
      --prereg10-dir P10    the PREREG_10 B files (real: results/prereg10; validation chunks data/prereg10/val)
      --p1-out-dir R        B's P1 reports, GPU seam, rehearsal (real: results/test10/p1; dry-run default O10_p1)
      --p1-data-dir H       B's P1 chunks (real: data/test10/p1; dry-run default O10_p1data)
Real mode takes the defaults exactly (other values are refused).
B     --workers N (featurization processes, default 4); dry-run only: --fake-scorer (p1_extract with a CPU stand-in
      probe), --synthetic-p1 (no KataGo featurization), --max-games N (extraction drill, no report).

seam    PREREG_10 1.4 / 4.2: on seam_val with the 06 seam's inputs (test02's seam_val dumps, the 06 feats), the A rule
        is evaluated on test02's seam_val apply_report.json (it must not hold: level 1, mixed colours) and the 06
        level-1 computation that 10A uses (tools/t06_player.py compute_pool, R02 = Q04) is run again: its top-5 must
        be byte-identical to the 06 seam's top5_off.csv (and to results/prereg06/player/player_val_top5.csv when the
        06 seam saw bitwise dumps), the selected combination equal. Else exit 4 (local drift).
        -> O10/seam_val/{t10_seam.json, top5.csv}.
apply   one test pool, PREREG_10 1.1-1.3: A holds when (1) T1 is PASS / NORMALIZED, (2) test02's apply_report.json has
        b_c and b_q exactly 1 and T4 records the pool as black only, (3) test02's guard level is 3, its gate passed,
        the effective combination has an opening term, that orientation's rho is computable and below the frozen
        theta2 of results/prereg02/player/frozen_config.json, and p02_player_fuse.guard_decision on the report's rho
        and the frozen guard gives exactly the report's applied combination, level and guard_reason (else exit 2),
        (4) 06 is usable (H6P adopted, the 06 feats and seam done, this pool's t06_pool.json made from the present
        test02 report and 06 seam). A holds -> 06's level-1 computation (sr06 over V2 / V3 / V0 [V1], PREREG_06 2.4,
        CSLSk*) with the 06 gate against R02' = test02's EFFECTIVE combination (not its applied CNN-only one):
        passed -> 10A rows, failed -> test02's rows; no one-to-one in this pool. A does not hold -> the 06 pool's
        top5_off.csv (06-OFF rows, or test02's rows where 06 fell back). -> O10/<pool>/{t10_pool.json, top5.csv}.
submit  O10/submission_player.csv = public rows, then private rows (test02's order), check_submission --official,
        O10/t10_submit_report.json (per pool: 06-OFF / 10A / test02, top-1 differences against 06-OFF and Q04,
        inputs_bound). Every row equal to this season's 06-OFF (Q04 when there is no 06-OFF): no file, exit 3
        ("10 not needed"). Refused (2) once the upload log holds a 10 entry of this run.
status  read-only: what is done, what is missing, K6B-10 and K6-10.
p1      part B (tools/t10_b.py): seam_val = the GPU seam (the first 400 validation games again on the 4090 against
        results/prereg10/feat/val_seam_feats.npz, projected max difference <= tol_P; failed -> B not used, exit 3 for
        B only); public / private = the pool's P1 chunks (resumable; STOP -> exit 5). Needs the committed rehearsal
        record (tol_P); K6B-10 passed -> exit 3. With B adopted, seam also recomputes 10B on seam_val against
        results/prereg10/player/player_val_top5.csv and apply adds w*·z(T_P1) to the pools that used 06 when the pool
        gate sr(10B) - sr(selected) >= 0 and P1 failed <= 0.5 % (source 10B).
p1-rehearse  part B, before any test data: m and tol_P = max(0.05, 2.5 m) -> results/test10/p1/p1_rehearsal.json.

Exit codes as tools/t06_player.py: 0 done; 2 refused (fix the precondition and run the same command again; an
unexpected internal error is also 2 with a traceback: stop and check); 3 the registered "not used" (T1
FAIL_OPENING, H6P not adopted, the 06 seam failed with other dumps, K6-10 passed without a committed 10 run, K5-10 not
met, 10 not needed, B not adopted; final choice by PREREG_06 5.4 / PREREG_10 4.4; for p1: B not used this season -
GPU seam failed, K6B-10 passed, K5-10 for B not met - 10 keeps part A); 4 local drift (the 10 seam does not reproduce
06 or the validation 10B); 5 p1 stopped on request (STOP file; resume with the same line). The 06 tools are imported, never changed (PREREG_10 1.2): tools/t06_player.py load_pool_data,
pool_store, load_feats, compute_pool and their checks. No notebook, no tools/score.py; no label file is ever opened;
nothing of competition 2 is imported (B shares only the KataGo P1 extraction code, PREREG_10 5); the GPU is used only
by p1 / p1-rehearse; data/local is read only for the two seam_val game files (the p06 guard).
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t06_player as t6  # noqa: E402  (imports p06_style / p06_player: the data/local guard of the p06 tools)
import p02_player_fuse as pf  # noqa: E402
import p02_player_open as po  # noqa: E402
import check_submission  # noqa: E402
import t10_b as tb  # noqa: E402  (part B; numpy only at import, p1_lib / torch only inside p1)

import numpy as np  # noqa: E402

# ------------------------------------------------------------------ paths (module constants; the tests may setattr)
RESULTS = os.path.join(REPO, "results")
REAL_O10 = os.path.join(RESULTS, "test10", "player")
P02_FROZEN = os.path.join(RESULTS, "prereg02", "player", "frozen_config.json")
P10_DOC = "docs/PREREG_10_PLAYER_GUARD_P1.md"
SELF_REL = "tools/t10_player.py"

# ------------------------------------------------------------------ registered values (PREREG_10)
TW = t6.TW
K5_10 = dt.datetime(2026, 10, 31, 23, 59, 59, tzinfo=TW)     # 3: the test-day extension (A) merged
K6B_10 = dt.datetime(2026, 11, 14, 23, 59, 59, tzinfo=TW)    # 3: B's P1 blocks and GPU seam (B adopted only)
K6_10 = dt.datetime(2026, 11, 21, 23, 59, 59, tzinfo=TW)     # 3: one successful official 10 run committed
TESTDAY_REV_ROW = "| rev 3 |"                                # the test-day revision of PREREG_10 (K5-10)
# PREREG_10 rev 4: whether B is adopted is read from the committed PREREG_10 files (t10_b.b_state), not a constant
METHOD = "prereg10 10: 06-OFF + all-black level-3 lift (A) [+ P1 LDA (B)] (甲)"
FORMATS = {"seam": "c10-test10 seam v1", "pool": "c10-test10 pool v1", "submit": "c10-test10 submit v1",
           "not_used": "c10-test10 not used v1"}
TEST_POOLS = t6.TEST_POOLS
SUBMISSION = "submission_player.csv"
REPORT = "t10_submit_report.json"
T1_OK = t6.T1_OK
A_CONDITIONS = ("t1_pass_or_normalized", "all_black", "level3_by_rho", "06_usable")
SOURCES = ("06-OFF", "10A", "10B", "test02")
TOOLS = ("tools/t10_player.py", "tools/t10_b.py", "tools/p10_fuse.py", "tools/p10_apply.py") + t6.TOOLS_COMMITTED
_LABEL = "ans" + "wer"

MSG_FINAL = "最終照 PREREG_06 5.4（06-OFF → Q04；docs/TESTDAY_10.md「10 不用」）"
MSG_REDO = ("06 或 test02 在這一步之後變了（PREREG_10 4.2：06 從 feats 重做時 10 也從頭重做）：照 docs/TESTDAY_10.md 從 seam "
            "重跑，三步都加 --rerun --reason")


def me():
    """This module (t10_b's functions take it for refuse / write_json / guard_write, never importing it back)."""
    return sys.modules[__name__]


class T10Error(RuntimeError):
    """A stop with its exit code: 2 refused, 3 registered 'not used', 4 local drift."""

    def __init__(self, message, code=2, reason=None):
        super().__init__(message)
        self.code, self.reason = code, reason


def refuse(msg):
    raise T10Error(msg, 2)


# ================================================================ helpers
def now_real():
    return dt.datetime.now(TW)


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def shown(path):
    return t6.shown(path)


def read_json(path, what=None):
    if not os.path.exists(path):
        refuse(f"missing {shown(path)}" + (f" ({what})" if what else ""))
    return t6.read_json(path, what)


def maybe_json(path):
    return t6.read_json(path) if os.path.exists(path) else None


def sha_raw(path):
    return t6.sha_raw(path)


def sha_lf(path):
    return t6.sha_lf(path)


def git(*args):
    return t6.git(*args)


def committed_clean(path):
    return t6.committed_clean(path)


def require_committed(ctx, paths, what):
    if not ctx["real"]:
        return
    bad = [shown(p) for p in paths if not committed_clean(p)]
    if bad:
        refuse(f"{what}: not committed and clean: {bad[:6]} (docs/TESTDAY_10.md: commit after every step)")


def git_commit_time(path):
    return t6.git_commit_time(path)


def guard_write(ctx, path):
    p = os.path.abspath(path)
    if _LABEL in p.lower():
        refuse(f"refusing {p}: validation label files are never written or opened by the 10 tools")
    ro = [os.path.join(RESULTS, "test02"), os.path.join(RESULTS, "test06"), os.path.join(REPO, "data", "test06"),
          t6.LOCAL, t6.LEDGER, t6.REAL_LOG] + [os.path.join(RESULTS, "prereg%02d" % i) for i in range(1, 11)]
    for r in ro:
        if t6.inside(p, r):
            refuse(f"refusing to write {p}: it is inside the read-only {shown(r)}")
    if not any(t6.abs_inside(p, r) for r in ctx["write_roots"]):
        refuse(f"refusing to write {p}: outside {ctx['write_roots']}")
    return p


def write_json(ctx, path, obj):
    guard_write(ctx, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(t6.jsonable(obj), f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def write_rows(ctx, path, rows):
    """question_id,top_1..top_5, UTF-8 without BOM, LF (as tools/t06_player.py write_rows)."""
    guard_write(ctx, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id"] + pf.TOP_COLS)
        w.writerows(rows)
    os.replace(tmp, path)
    return path


def guard_existing(ctx, a, paths):
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not a.rerun:
        refuse("outputs exist (%s); use --rerun --reason TEXT to overwrite" % ", ".join(shown(p) for p in existing))
    if a.rerun and not (a.reason or "").strip():
        refuse("--rerun needs --reason TEXT (it is stored in the report)")
    if not a.rerun:
        return None
    return {"reason": a.reason, "previous_sha256": {shown(p): sha_raw(p) for p in existing}}


def left(now, until):
    return t6.left(now, until)


def tool_record():
    return {"git_head": t6.git_head(),
            "files_sha256_lf": {r: (t6.st.sha256_lf(os.path.join(REPO, *r.split("/")))
                                    if os.path.exists(os.path.join(REPO, *r.split("/"))) else None) for r in TOOLS}}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ================================================================ modes and directories
def resolve(a):
    """The 06 context of tools/t06_player.py resolve (O6 / A6 / D2 / O2 / T / C / LOG / P6, real defaults exactly)
    plus O10, the only folder this tool writes."""
    if a.out_dir:
        t6.refuse_label(a.out_dir)
    cmd6 = "status" if a.cmd in ("status", "p1-rehearse") else "apply"
    ns = argparse.Namespace(cmd=cmd6, dry_run=a.dry_run, out_dir=a.t06_out_dir, data_dir=a.t06_data_dir,
                            t02_results_dir=a.t02_results_dir, t02_out_dir=a.t02_out_dir, tests_dir=a.tests_dir,
                            checks_dir=a.checks_dir, log=a.log, prereg06_dir=a.prereg06_dir, now=a.now)
    try:
        ctx = t6.resolve(ns)
    except t6.T06Error as e:
        raise T10Error(str(e).replace("--out-dir", "--t06-out-dir").replace("--data-dir", "--t06-data-dir"),
                       e.code, e.reason)
    if ctx["real"]:
        if a.out_dir and os.path.normcase(os.path.abspath(a.out_dir)) != os.path.normcase(REAL_O10):
            refuse(f"real mode: --out-dir must be {REAL_O10} (got {a.out_dir}); other trees need --dry-run")
        o10 = REAL_O10
    else:
        if not a.out_dir:
            refuse("--dry-run needs --out-dir (a dry run never falls back to the repository paths)")
        o10 = os.path.abspath(a.out_dir)
        if t6.st.inside_repo(o10):
            refuse(f"--dry-run: --out-dir {o10} is inside the repository; use a scratch folder")
    if ctx["real"] and a.now and a.cmd != "status":
        refuse("--now is for --dry-run and status only (real mode uses the clock)")
    # rev 4 section 14 (M1): a dry run sees B only with an explicit --prereg10-dir (the default is an absent folder),
    # so the A dry runs stay A-only after B is adopted in the repository
    b_dirs = {"--prereg10-dir": (getattr(a, "prereg10_dir", None), tb.REAL_P10, o10 + "_no_prereg10"),
              "--p1-out-dir": (getattr(a, "p1_out_dir", None), tb.REAL_P1_OUT, o10 + "_p1"),
              "--p1-data-dir": (getattr(a, "p1_data_dir", None), tb.REAL_P1_DATA, o10 + "_p1data")}
    got = {}
    for k, (v, real, dry) in b_dirs.items():
        if v:
            t6.refuse_label(v)
        if ctx["real"]:
            if v and os.path.normcase(os.path.abspath(v)) != os.path.normcase(real):
                refuse(f"real mode: {k} must be {real} (got {v}); other trees need --dry-run")
            got[k] = real
        else:
            got[k] = os.path.abspath(v) if v else dry
            if k != "--prereg10-dir" and t6.st.inside_repo(got[k]):
                refuse(f"--dry-run: {k} {got[k]} is inside the repository; use a scratch folder")
    ctx = dict(ctx, O10=o10, P10=got["--prereg10-dir"], O10P1=got["--p1-out-dir"], A10P1=got["--p1-data-dir"],
               write_roots=[o10, got["--p1-out-dir"], got["--p1-data-dir"]])
    return ctx


def out_paths(ctx, pool):
    d = os.path.join(ctx["O10"], pool)
    return {"dir": d, "pool": os.path.join(d, "t10_pool.json"), "top5": os.path.join(d, "top5.csv"),
            "seam": os.path.join(ctx["O10"], "seam_val", "t10_seam.json"),
            "seam_top5": os.path.join(ctx["O10"], "seam_val", "top5.csv")}


def t06_paths(ctx, pool):
    return t6.out_paths(ctx, pool)


# ================================================================ the registered "not used"
def not_used(ctx, cmd, reason, message, pool=None, write=True):
    """Exit 3: record the event in O10/not_used.json (commit it) and stop."""
    if write:
        path = os.path.join(ctx["O10"], "not_used.json")
        rec = maybe_json(path) or {"format": FORMATS["not_used"], "events": []}
        rec["events"].append({"command": cmd, "pool": pool, "reason": reason, "message": message, "mode": ctx["mode"],
                              "created": now_iso(), "now": ctx["now"].isoformat(), "git_head": t6.git_head()})
        write_json(ctx, path, rec)
    raise T10Error(message + "；" + MSG_FINAL, 3, reason)


def from_t06(ctx, cmd, pool, fn, *args):
    """Call a tools/t06_player.py function; its registered 'not used' (3) becomes 10's (recorded in O10, never in
    the 06 folder), its other stops keep their code."""
    try:
        return fn(*args)
    except t6.T06Error as e:
        if e.code == 3:
            not_used(ctx, cmd, "t06_" + str(e.reason or "not_used"), "06 不能用（tools/t06_player.py: %s）：A 不用" % e,
                     pool)
        raise T10Error(str(e), e.code, e.reason)


# ================================================================ common prerequisites (PREREG_10 4.2, t06 rev 3 item 4)
def check_t1(ctx, cmd, pool):
    """test02's T1 (as tools/t06_player.py check_t1, without writing into the 06 folder)."""
    path = os.path.join(ctx["C"], "player", "t1.json")
    if not os.path.exists(path):
        refuse(f"{shown(path)} missing: test02's T1 comes first (docs/TESTDAY.md)")
    t1 = read_json(path)
    v = t1.get("verdict")
    if ctx["real"]:
        require_committed(ctx, [path], "test02's T1")
        mode = (t1.get("provenance") or {}).get("mode")
        if t1.get("complete") is not True or t1.get("stage") != "raw" or mode != "real":
            refuse(f"{shown(path)}: complete={t1.get('complete')!r}, stage {t1.get('stage')!r}, mode {mode!r}: the "
                   f"full T1 on the raw files comes first (docs/TESTDAY.md)")
    if v == "FAIL_OPENING":
        not_used(ctx, cmd, "t1_fail_opening", "T1 是 FAIL_OPENING：06 整個不用，A 也不用（PREREG_10 1.3、PREREG_06 2.7）",
                 pool)
    if v not in T1_OK:
        refuse(f"T1 verdict {v!r}: FAIL_STRUCT stops as test02 does (PREREG_06 2.7)")
    return {"path": shown(path), "sha256_lf": sha_lf(path), "verdict": v}


def run_committed_by_k6(ctx):
    rp = os.path.join(ctx["O10"], REPORT)
    if not os.path.exists(rp):
        return False
    if not ctx["real"]:
        return True
    t = git_commit_time(rp)
    return t is not None and t6.parse_iso(t, "commit time") <= K6_10


def load_p02_guard(ctx):
    """The frozen PREREG_02 player guard (theta1 / theta2 per orientation); real mode: committed."""
    require_committed(ctx, [P02_FROZEN], "results/prereg02/player/frozen_config.json")
    fz = t6.read_json(P02_FROZEN)
    g = fz.get("guard") or {}
    if set((g.get("thresholds") or {})) != set(po.ORIENTS) or g.get("smoke") is not False:
        refuse("results/prereg02/player/frozen_config.json has no real guard for both orientations")
    return g, sha_lf(P02_FROZEN)


def load_t06_feats_reports(ctx):
    """The 06 feats of the three pools ran with exit 0 (their reports exist in this mode; real mode: committed)."""
    out = {}
    for pool in t6.POOLS:
        p = t06_paths(ctx, pool)["feats_report"]
        rep = read_json(p, f"tools/t06_player.py feats --pool {pool} first (PREREG_10 1.1 item 4)")
        if rep.get("format") != t6.FORMATS["feats"] or rep.get("mode") != ctx["mode"] or rep.get("pool") != pool:
            refuse(f"{shown(p)}: format {rep.get('format')!r} mode {rep.get('mode')!r}")
        require_committed(ctx, [p], f"the 06 feats report of {pool}")
        out[pool] = {"path": p, "sha256_lf": sha_lf(p)}
    return out


def common_prereqs(ctx, cmd, pool=None):
    """In this order: .venv (2), git clean (2, real), the frozen 06 files and tools (2; t10 committed), K6-10 (3; 2 when
    a 10 run was committed by K6-10), H6P (3), T1 (3 / 2), test02 (2), Q04's first upload (2), the 06 seam (3 when it
    failed with other dumps, else 2), the 06 feats of the three pools (2)."""
    venv = from_t06(ctx, cmd, pool, t6.check_venv)
    if ctx["real"]:
        clean, dirty = t6.git_clean_tracked()
        if not clean:
            refuse(f"git is not clean (tracked changes: {str(dirty)[:300]}): commit after every step "
                   f"(docs/TESTDAY_10.md), then run the same command again")
    fz, models, frozen = from_t06(ctx, cmd, pool, t6.load_frozen, ctx)
    tools = from_t06(ctx, cmd, pool, t6.check_tools, ctx, fz)
    require_committed(ctx, [os.path.join(REPO, *SELF_REL.split("/")), os.path.join(HERE, "t10_b.py")],
                      "tools/t10_player.py and tools/t10_b.py")
    if ctx["now"] > K6_10:
        if run_committed_by_k6(ctx):
            refuse(f"K6-10 已過（{K6_10.isoformat()}）：10 的步驟都不再執行；K6-10 前已 commit 的 10 照常有效，最終選擇照 "
                   f"PY tools/official_log.py status（docs/TESTDAY_10.md 第 6 步）")
        not_used(ctx, cmd, "k6_10", f"K6-10 已過（{K6_10.isoformat()}；現在 {ctx['now'].isoformat()}）：這一季不用 10",
                 pool)
    h6p = from_t06(ctx, cmd, pool, t6.h6p_state, ctx)
    if not h6p["adopted"]:
        not_used(ctx, cmd, "h6p_not_adopted", "H6P 沒有採用：06 不能用，A 不用（PREREG_10 1.3）", pool)
    t1 = check_t1(ctx, cmd, pool)
    t02 = from_t06(ctx, cmd, pool, t6.check_test02, ctx)
    q04 = from_t06(ctx, cmd, pool, t6.check_q04_upload, ctx)
    pre = {"venv": venv, "fz": fz, "models": models, "frozen": frozen, "tools": tools, "h6p": h6p, "t1": t1,
           "t02": t02, "q04_upload": q04}
    seam6, seam6_sha = from_t06(ctx, cmd, pool, t6.load_seam, ctx, pre)
    pre.update(seam6=seam6, seam6_sha=seam6_sha, feats6=load_t06_feats_reports(ctx))
    guard, guard_sha = load_p02_guard(ctx)
    pre.update(guard=guard, guard_sha=guard_sha, b=tb.b_state(me(), ctx))
    return pre


# ================================================================ the A rule (PREREG_10 1.1; a pure function)
def a_conditions(ar, t4_pool, guard, t1_verdict, usable06):
    """PREREG_10 1.1 for one pool. ar: test02's apply_report.json; t4_pool: T4's record of this pool (t4.json
    pools[pool]) or None; guard: the frozen PREREG_02 guard; t1_verdict: T1; usable06: (bool, detail) of item 4.
    -> {"conditions": {name: {"ok", ...}}, "holds": bool, "errors": [inconsistencies: exit 2]}."""
    cond, errors = {}, []
    cond["t1_pass_or_normalized"] = {"ok": t1_verdict in T1_OK, "verdict": t1_verdict}
    b_c, b_q = ar.get("b_c_candidate_black_share"), ar.get("b_q_question_black_share")
    black = b_c == 1 and b_q == 1 and finite(b_c) and finite(b_q)
    t4 = None
    if t4_pool is not None:
        t4 = {k: t4_pool.get(k) for k in ("b_c", "b_q", "black_only_pool")}
        t4_black = bool(t4_pool.get("black_only_pool")) and t4_pool.get("b_c") == 1 and t4_pool.get("b_q") == 1
        if t4_black != black:
            errors.append(f"T4 records black_only_pool {t4_pool.get('black_only_pool')!r} (b_c {t4_pool.get('b_c')!r}, "
                          f"b_q {t4_pool.get('b_q')!r}) but test02's apply_report.json has b_c {b_c!r}, b_q {b_q!r}")
    elif black:
        errors.append("test02's apply_report.json has b_c = b_q = 1 but T4 (checks/player/t4.json) has no record of "
                      "this pool: run T4 (docs/TESTDAY.md step 3) and commit it")
    cond["all_black"] = {"ok": bool(black and (t4_pool is not None) and not errors),
                         "b_c": b_c, "b_q": b_q, "t4": t4,
                         "rule": "b_c and b_q of test02's apply_report.json exactly 1 (p02_player_open.colour_share), "
                                 "T4 black_only_pool true"}
    level, gate = ar.get("guard_level"), ar.get("gate_passed")
    eff = ar.get("effective") or {}
    combo = pf.COMBO_BY_ID.get(eff.get("id"))
    if combo is None or any(combo[k] != eff.get(k) for k in ("e", "o", "w")):
        errors.append(f"test02's effective combination {eff!r} is not a p02 combination")
        combo = None
    o = (combo or {}).get("o")
    rho = (ar.get("rho") or {}).get(o) if o else None
    th = (guard.get("thresholds") or {}).get(o) if o else None
    got_th = ((ar.get("guard") or {}).get("thresholds") or {})
    if got_th != guard.get("thresholds"):
        errors.append("the guard thresholds in test02's apply_report.json differ from results/prereg02/player/"
                      "frozen_config.json")
    recomputed = None
    if combo is not None and ar.get("rho") is not None:
        rho_all = {k: v for k, v in (ar.get("rho") or {}).items()}
        applied, lvl, why = pf.guard_decision(combo, rho_all, guard, b_c, b_q, True)
        recomputed = {"applied": applied["id"], "level": lvl, "reason": why}
        want = {"applied": (ar.get("applied") or {}).get("id"), "level": level, "reason": ar.get("guard_reason")}
        if gate is True and recomputed != want:
            errors.append(f"the guard recomputed from the report's rho and the frozen thresholds {recomputed} differs "
                          f"from test02's apply_report.json {want} (PREREG_10 1.1 item 3: exit 2)")
    ok3 = bool(level == 3 and gate is True and o is not None and finite(rho) and th is not None
               and th.get("theta2_enabled") is True and rho < th["theta2"]
               and recomputed is not None and recomputed["level"] == 3 and not errors)
    cond["level3_by_rho"] = {"ok": ok3, "guard_level": level, "gate_passed": gate, "effective": eff.get("id"),
                             "orientation": o, "rho": rho, "theta2": (th or {}).get("theta2"),
                             "theta2_enabled": (th or {}).get("theta2_enabled"),
                             "guard_reason": ar.get("guard_reason"), "recomputed": recomputed}
    cond["06_usable"] = {"ok": bool(usable06[0]), "detail": usable06[1]}
    holds = all(c["ok"] for c in cond.values()) and not errors
    return {"conditions": cond, "holds": holds, "errors": errors}


def t06_pool_state(ctx, pool, pre):
    """This pool's 06 report: made from the present test02 report and 06 seam (PREREG_06 rev 3 item 8). -> (usable,
    detail, report, paths); a missing or stale report is refused (2): run tools/t06_player.py apply first."""
    op6 = t06_paths(ctx, pool)
    rep = read_json(op6["pool"], f"tools/t06_player.py apply --pool {pool} first")
    require_committed(ctx, [op6["pool"], op6["off"]], f"the 06 outputs of {pool}")
    if rep.get("format") != t6.FORMATS["pool"] or rep.get("mode") != ctx["mode"]:
        refuse(f"{shown(op6['pool'])}: format {rep.get('format')!r} mode {rep.get('mode')!r}")
    tp = pre["t02"]["pools"][pool]
    stale = []
    if (rep.get("test02") or {}).get("apply_report_sha256_lf") != sha_lf(tp["apply_report"]) or \
            (rep.get("test02") or {}).get("top5_sha256") != sha_raw(tp["top5"]):
        stale.append("test02's reports changed since the 06 apply")
    if (rep.get("inputs") or {}).get("seam_sha256_lf") != pre["seam6_sha"]:
        stale.append("the 06 seam changed since the 06 apply")
    if not os.path.exists(op6["off"]) or sha_raw(op6["off"]) != (rep.get("outputs") or {}).get("top5_off.csv"):
        stale.append("top5_off.csv is not the file t06_pool.json records")
    if stale:
        refuse(f"{pool}: {stale}: run tools/t06_player.py apply --pool {pool} --rerun first. " + MSG_REDO)
    detail = {"h6p_adopted": pre["h6p"]["adopted"], "t06_seam_passed": pre["seam6"].get("passed") is True,
              "t06_feats_three_pools": sorted(pre["feats6"]), "t06_pool_current": True,
              "t06_pool_used06": rep.get("used06")}
    usable = detail["h6p_adopted"] and detail["t06_seam_passed"] and len(pre["feats6"]) == 3
    return usable, detail, rep, op6


def load_t4(ctx):
    path = os.path.join(ctx["C"], "player", "t4.json")
    if not os.path.exists(path):
        return None, None
    t4 = read_json(path)
    if ctx["real"]:
        require_committed(ctx, [path], "test02's T4")
        if (t4.get("provenance") or {}).get("mode") != "real":
            refuse(f"{shown(path)}: mode {(t4.get('provenance') or {}).get('mode')!r}: the real T4 comes first")
    return t4, path


def load_pool(ctx, pool, pre):
    """test02's pool data with the checks of tools/t06_player.py cmd_apply (the files and dumps test02 used, the
    question order of test02's top5.csv) and the 06 feats store."""
    d, _cr, _qr, shas, _f = t6.load_pool_data(ctx, pool, use_cnn=True)
    if pool != "seam_val":
        tp = pre["t02"]["pools"][pool]
        binding = t6.t02_binding(ctx, pool, pre["t02"], dict(shas, **{k: d.shas.get(k) for k in
                                                                   ("cand_games_npz", "query_games_npz")}))
        dump_bad = [k for k in ("cand_games_npz", "query_games_npz") if d.shas.get(k) != (binding or {}).get(k)]
        if dump_bad:
            refuse(f"{pool}: dumps {dump_bad} differ from test02's apply_report.json inputs. " + MSG_REDO)
        if d.qids != [r[0] for r in t6.read_top5(tp["top5"])]:
            refuse(f"{pool}: question order differs from test02's top5.csv")
    frep, arrs = t6.load_feats(ctx, pool, shas)
    ms = t6.pool_store(d, arrs)
    return d, ms, shas, frep


def level1_06(d, ms, pre, r02):
    """PREREG_10 1.2: 06's level-1 computation (tools/t06_player.py compute_pool) with R02' = r02, and the 06 gate."""
    res = t6.compute_pool(d, ms, pre["models"], pre["fz"], r02)
    diff = res["sr06"][res["selected"]] - res["sr06"]["R02"]
    gate = {"sr_selected_minus_R02prime": diff, "min": t6.GATE06_MIN, "passed": bool(diff >= t6.GATE06_MIN),
            "r02prime": r02["id"]}
    return res, gate


# ================================================================ seam
def cmd_seam(ctx, a):
    t0 = time.time()
    op = out_paths(ctx, "seam_val")
    pre = common_prereqs(ctx, "seam", "seam_val")
    rerun = guard_existing(ctx, a, [op["seam"], op["seam_top5"]])
    refuse_rerun_once_uploaded(ctx, a, "seam")
    ar_p = os.path.join(ctx["O2"], "seam_val", "apply_report.json")
    require_committed(ctx, [ar_p], "test02's seam_val apply_report.json")
    ar = read_json(ar_p)
    seam6 = pre["seam6"]
    op6 = t06_paths(ctx, "seam_val")
    t06_top5 = op6["off"]
    if not os.path.exists(t06_top5) or sha_raw(t06_top5) != (seam6.get("outputs") or {}).get("top5_off.csv"):
        refuse("the 06 seam's top5_off.csv is missing or not the file t06_seam.json records")
    a_rec = a_conditions(ar, None, pre["guard"], pre["t1"]["verdict"],
                         (True, {"rule": "seam_val: the 06 seam passed"}))
    if a_rec["errors"]:
        refuse("seam_val: " + "; ".join(a_rec["errors"]))
    d, ms, shas, frep = load_pool(ctx, "seam_val", pre)
    seam_inputs = (seam6.get("inputs") or {})
    bad = [k for k in ("cand_games_npz", "query_games_npz")
           if d.shas.get(k) != (seam_inputs.get("dumps") or {}).get(k)]
    if bad or seam_inputs.get("feats_npz") != sha_raw(t06_paths(ctx, "seam_val")["feats"]):
        refuse(f"seam_val: the dumps {bad} or the feats differ from the 06 seam's inputs. " + MSG_REDO)
    r02 = pf.COMBO_BY_ID["Q04"]                         # PREREG_06: R02 of the validation pool is Q04
    res, gate = level1_06(d, ms, pre, r02)
    rows = [[x] + t for x, t in zip(d.qids, res["off"])]
    write_rows(ctx, op["seam_top5"], rows)
    mine = t6.bytes_lf(op["seam_top5"])
    ref = os.path.join(ctx["P6"], "player_val_top5.csv")
    crit = {"a_not_triggered": not a_rec["holds"],
            "selected_equal_06_seam": res["selected"] == seam6.get("selected"),
            "gate_decision_equal_06_seam": gate["passed"] == (
                ((seam6.get("gate") or {}).get("sr_selected_minus_R02", -1)) >= t6.GATE06_MIN),
            "top5_byte_identical_to_06_seam_top5_off": mine == t6.bytes_lf(t06_top5)}
    if seam6.get("dumps_bitwise_equal") is True:
        crit["top5_byte_identical_to_h6p"] = mine == t6.bytes_lf(ref)
    b_seam = None
    if pre["b"]["adopted"]:     # PREREG_10 4.2 / rev 4: the validation 10B is reproduced; a failure is B-only (sec. 14)
        try:
            b_seam = tb.seam_10b(me(), ctx, pre["b"], d, ms, res, seam6.get("dumps_bitwise_equal") is True)
        except Exception as e:  # noqa: BLE001
            b_seam = {"passed": False, "error": f"{type(e).__name__}: {e}"}
        if not b_seam["passed"]:
            tb.record_b_not_used(ctx, "seam", "val_10b_seam_failed",
                                 f"seam_val 的驗證集 10B 對不上 results/prereg10/player/player_val_top5.csv：{b_seam}")
    passed = all(crit.values())
    rep = {"format": FORMATS["seam"], "mode": ctx["mode"], "pool": "seam_val",
           "what": "PREREG_10 1.4 / 4.2 seam: the A rule on test02's seam_val report (must not hold) and 06's level-1 "
                   "computation (the 10A path) again on seam_val with the 06 seam's inputs",
           "criteria": crit, "passed": passed, "verdict": "passed" if passed else "local_drift",
           "a_rule": a_rec, "selected": res["selected"], "sr06": res["sr06"], "gate": gate,
           "outputs": {"top5.csv": sha_raw(op["seam_top5"])},
           "inputs": {"t06_seam_sha256_lf": pre["seam6_sha"], "t06_seam_top5_off_sha256": sha_raw(t06_top5),
                      "test02_seam_val_apply_report_sha256_lf": sha_lf(ar_p),
                      "test02_seam_local_sha256_lf": pre["t02"]["seam_local_sha256_lf"],
                      "feats_report_sha256_lf": sha_lf(op6["feats_report"]), "feats_npz": sha_raw(op6["feats"]),
                      "p02_frozen_sha256_lf": pre["guard_sha"]},
           "b": b_record(pre["b"]), "b_seam": b_seam, "frozen": pre["frozen"], "t1": pre["t1"],
           "q04_upload": pre["q04_upload"],
           "rerun": rerun, "tools": tool_record(), "env": t6.env_record(), "sr06_seconds": res["sr06_seconds"],
           "created": now_iso(), "seconds": round(time.time() - t0, 1)}
    write_json(ctx, op["seam"], rep)
    print("seam %s: A %s on seam_val; selected %s; top5 equal to the 06 seam %s%s%s" % (
        rep["verdict"].upper(), "holds (drift)" if a_rec["holds"] else "not triggered", res["selected"],
        crit["top5_byte_identical_to_06_seam_top5_off"],
        "" if "top5_byte_identical_to_h6p" not in crit else ", to H6P %s" % crit["top5_byte_identical_to_h6p"],
        "" if b_seam is None else "; 10B equal to the validation 10B %s (byte %s, rows %d/%d)" % (
            b_seam["passed"], b_seam["byte_identical"], b_seam["rows_identical"], b_seam["rows"])))
    if not passed:
        raise T10Error("the 10 seam does not reproduce 06 on seam_val: local drift; commit t10_seam.json, stop and "
                       "check (PREREG_10 4.2): %s" % [k for k, v in crit.items() if not v], 4, "seam_drift")
    return 0


def load_seam(ctx, pre):
    op = out_paths(ctx, "seam_val")
    rep = read_json(op["seam"], "run seam first")
    require_committed(ctx, [op["seam"], op["seam_top5"]], "t10_seam.json")
    if rep.get("format") != FORMATS["seam"] or rep.get("mode") != ctx["mode"]:
        refuse(f"{shown(op['seam'])}: format {rep.get('format')!r} mode {rep.get('mode')!r}")
    if rep.get("passed") is not True:
        refuse(f"the 10 seam did not pass ({shown(op['seam'])}): no test pool runs")
    if (rep.get("inputs") or {}).get("t06_seam_sha256_lf") != pre["seam6_sha"] or \
            (rep.get("inputs") or {}).get("test02_seam_local_sha256_lf") != pre["t02"]["seam_local_sha256_lf"]:
        refuse("the 06 seam or test02's seam changed since the 10 seam. " + MSG_REDO)
    return rep, sha_lf(op["seam"])


def b_seam_problem(seam10, b):
    """B-only (rev 4 section 14): None when the 10 seam reproduced the validation 10B under the present B files, else the
    reason B is not used in apply."""
    if not b.get("adopted"):
        return None
    rb = seam10.get("b") or {}
    if rb.get("adopted") is not True or rb.get("frozen_sha256_lf") != b["frozen_sha256_lf"]:
        return "seam_without_b"
    if (seam10.get("b_seam") or {}).get("passed") is not True:
        return "val_10b_seam_failed"
    return None


def b_record(b=None, extra=None):
    return tb.b_record(b, extra)


# ================================================================ apply (one test pool)
def cmd_apply(ctx, a):
    t0 = time.time()
    pool = a.pool
    op = out_paths(ctx, pool)
    pre = common_prereqs(ctx, "apply", pool)
    rerun = guard_existing(ctx, a, [op["pool"], op["top5"]])
    refuse_rerun_once_uploaded(ctx, a, "apply")
    seam10, seam10_sha = load_seam(ctx, pre)
    tp = pre["t02"]["pools"][pool]
    ar = tp["report"]
    usable, detail, rep6, op6 = t06_pool_state(ctx, pool, pre)
    t4, t4_path = load_t4(ctx)
    t4_pool = ((t4 or {}).get("pools") or {}).get(pool)
    a_rec = a_conditions(ar, t4_pool, pre["guard"], pre["t1"]["verdict"], (usable, detail))
    if a_rec["errors"]:
        refuse(f"{pool}: " + "; ".join(a_rec["errors"]))
    res, gate, source, rule, d, ms = None, None, None, None, None, None
    t02_rows = t6.read_top5(tp["top5"])
    if a_rec["holds"]:
        d, ms, shas, frep = load_pool(ctx, pool, pre)
        eff = ar["effective"]
        r02 = pf.COMBO_BY_ID[eff["id"]]
        res, gate = level1_06(d, ms, pre, r02)
        if gate["passed"]:
            rows = [[x] + t for x, t in zip(d.qids, res["off"])]
            source, rule = "10A", ("A holds: 06's level-1 computation, the 06 gate against R02' = test02's effective "
                                   "%s passed (PREREG_10 1.2)" % eff["id"])
        else:
            rows = t02_rows
            source, rule = "test02", ("A holds, but the 06 gate against R02' = %s failed: this pool keeps test02's "
                                      "rows (PREREG_10 1.2)" % eff["id"])
    else:
        rows = t6.read_top5(op6["off"])
        failed = [k for k, v in a_rec["conditions"].items() if not v["ok"]]
        source = "06-OFF" if rep6.get("used06") else "test02"
        rule = (f"A does not hold ({failed}): the 06 pool's top5_off.csv "
                f"({'06-OFF rows' if rep6.get('used06') else 'test02 rows, 06 fell back'}; PREREG_10 1.3)")
    b_pool = None
    if pre["b"]["adopted"] and source in ("06-OFF", "10A"):   # PREREG_10 4.2 / rev 4: B only on pools that used 06
        why = b_seam_problem(seam10, pre["b"])
        if why:
            b_pool = {"applied": False, "rows": None, "reason": why}
        else:
            try:                                              # rev 4 section 14: any B problem -> this pool keeps A
                if res is None:                               # 06-OFF: the 06 computation again (R02 = applied)
                    d, ms, shas, frep = load_pool(ctx, pool, pre)
                    applied = ar.get("applied") or {}
                    res = t6.compute_pool(d, ms, pre["models"], pre["fz"], pf.COMBO_BY_ID[applied["id"]])
                    if [[x] + t for x, t in zip(d.qids, res["off"])] != rows:
                        raise T10Error("06 recomputed for B differs from the 06 pool's top5_off.csv", 4, "drift")
                b_pool = tb.pool_term(me(), ctx, pool, pre, pre["b"], d, ms, res)
            except Exception as e:  # noqa: BLE001
                b_pool = {"applied": False, "rows": None, "reason": f"b_error: {type(e).__name__}: {e}"}
        if not b_pool["applied"] and b_pool.get("reason") != "p1_not_done":
            tb.record_b_not_used(ctx, "apply", str(b_pool.get("reason")).split(":")[0],
                                 f"{pool}: B 不加（{b_pool.get('reason')}）", pool)
        if b_pool["applied"]:
            rows, source = b_pool["rows"], "10B"
            rule += "; B: 10B = CSLS_k*(F_sel + w*·z(T_P1)), pool gate %+.6f, P1 failed %.4f%% (PREREG_10 rev 4)" % (
                b_pool["pool_gate"]["sr_10B_minus_selected"], 100 * b_pool["failed_fraction"])
        else:
            rule += f"; B not added in this pool ({b_pool.get('reason')})"
    if [r[0] for r in rows] != [r[0] for r in t02_rows]:
        refuse(f"{pool}: rows are not in test02's (question-file) order")
    write_rows(ctx, op["top5"], rows)
    off6 = t6.read_top5(op6["off"])
    rec = {"format": FORMATS["pool"], "mode": ctx["mode"], "pool": pool, "source": source, "rule": rule,
           "what": "PREREG_10 1.1-1.3 (part A) and rev 4 (part B when adopted) apply on one test pool",
           "a_rule": a_rec, "a_holds": a_rec["holds"], "gate06_vs_r02prime": gate,
           "sr06": res["sr06"] if res else None, "selected": res["selected"] if res else None,
           "weights": res["weights"] if res else None, "k_star": pre["fz"]["k_star"],
           "one_to_one": "never in a pool of A (PREREG_10 1.2; PREREG_06 5.3 (e))",
           "test02": {"guard_level": ar.get("guard_level"), "guard_reason": ar.get("guard_reason"),
                      "effective": (ar.get("effective") or {}).get("id"), "applied": (ar.get("applied") or {}).get("id"),
                      "gate_passed": ar.get("gate_passed"), "rho": ar.get("rho"),
                      "b_c": ar.get("b_c_candidate_black_share"), "b_q": ar.get("b_q_question_black_share"),
                      "apply_report_sha256_lf": sha_lf(tp["apply_report"]), "top5_sha256": sha_raw(tp["top5"])},
           "t06": {"used06": rep6.get("used06"), "rule": rep6.get("rule"), "t06_pool_sha256_lf": sha_lf(op6["pool"]),
                   "top5_off_sha256": sha_raw(op6["off"])},
           "top1_differences_vs_t06_top5_off": t6.top1_diff(rows, off6),
           "top1_differences_vs_test02": t6.top1_diff(rows, t02_rows),
           "b": b_record(pre["b"], {"pool": None if b_pool is None else {k: v for k, v in b_pool.items()
                                                                        if k != "rows"}}),
           "outputs": {"top5.csv": sha_raw(op["top5"])},
           "inputs": {"t10_seam_sha256_lf": seam10_sha, "t06_seam_sha256_lf": pre["seam6_sha"],
                      "t06_feats_report_sha256_lf": {p: v["sha256_lf"] for p, v in pre["feats6"].items()},
                      "t4_sha256_lf": sha_lf(t4_path) if t4_path else None, "p02_frozen_sha256_lf": pre["guard_sha"]},
           "frozen": pre["frozen"], "t1": pre["t1"], "q04_upload": pre["q04_upload"], "rerun": rerun,
           "tools": tool_record(), "env": t6.env_record(), "sr06_seconds": res["sr06_seconds"] if res else None,
           "created": now_iso(), "seconds": round(time.time() - t0, 1)}
    write_json(ctx, op["pool"], rec)
    print(f"apply {pool}: {source} ({rule})" + (f"; sr06 { {k: round(v, 6) for k, v in res['sr06'].items()} }, "
                                                f"selected {res['selected']}" if res else ""))
    return 0


# ================================================================ submit
def k5_10_state(ctx):
    """K5-10: the test-day revision row of PREREG_10 was committed by 2026-10-31 23:59:59 and is in HEAD (real mode);
    dry-run: recorded, not checked."""
    if not ctx["real"]:
        return {"checked": False, "ok": None, "rule": "dry-run: K5-10 is checked in real mode only"}
    # -m: the row first appears in a merge commit (main merged into p10-testday); plain -S skips merges
    rc, out = git("log", "--reverse", "-m", "-S" + TESTDAY_REV_ROW, "--format=%H %cI", "--", P10_DOC)
    first = out.splitlines()[0].split() if rc == 0 and out else None
    try:
        r = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{P10_DOC}"], capture_output=True, timeout=60)
        head = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        head = ""
    in_head = any(line.startswith(TESTDAY_REV_ROW) for line in head.splitlines())
    ok = bool(first and in_head and t6.parse_iso(first[1], "commit time") <= K5_10)
    return {"checked": True, "ok": ok, "row": TESTDAY_REV_ROW, "commit": first[0] if first else None,
            "committed_at": first[1] if first else None, "deadline": K5_10.isoformat(), "in_head": in_head}


def t06_off_state(ctx, pre):
    """This season's 06-OFF (results/test06/player/off): (rows, record) when it exists and was made from the present
    06 pool reports; None when there is none; refused (2) when it is stale."""
    sub = os.path.join(ctx["O6"], "off", t6.SUBMISSION)
    rp = os.path.join(ctx["O6"], "off", t6.REPORT)
    if not os.path.exists(rp):
        used = [p for p in TEST_POOLS if (maybe_json(t06_paths(ctx, p)["pool"]) or {}).get("used06")]
        if used:
            refuse(f"06 was used in {used} (t06_pool.json used06) but there is no 06-OFF submit report "
                   f"({shown(rp)}): run tools/t06_player.py submit first; 10 is never compared with Q04 then")
        return None, {"exists": False}
    rep = read_json(rp)
    require_committed(ctx, [rp, sub], "the 06-OFF submission")
    probs = []
    if not os.path.exists(sub) or sha_raw(sub) != (rep.get("output") or {}).get("sha256"):
        probs.append("submission_player.csv is not the file t06_submit_report.json records")
    for pool in TEST_POOLS:
        want = ((rep.get("pools") or {}).get(pool) or {}).get("t06_pool_sha256_lf")
        if want != sha_lf(t06_paths(ctx, pool)["pool"]):
            probs.append(f"{pool}: 06-OFF was made from another t06_pool.json")
    if probs:
        refuse(f"the 06-OFF in {shown(rp)} is not the present 06 run ({probs}): redo 06 submit first (docs/TESTDAY_06.md)")
    on_rp = os.path.join(ctx["O6"], "on", t6.REPORT)
    return t6.read_top5(sub), {"exists": True, "run_id": rep.get("run_id"), "sha256": sha_raw(sub),
                               "report_sha256": sha_raw(rp), "path": shown(sub),
                               "on_produced": bool(rep.get("on_produced")) and os.path.exists(on_rp)}


def run_in_log(ctx, run_id):
    path = ctx["LOG"]
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    return [e.get("id") for e in log.get("uploads", []) if e.get("comp") == "player"
            and (e.get("test10") or {}).get("run_id") == run_id]


def current_run_id(ctx):
    paths = [out_paths(ctx, "seam_val")["seam"]] + [out_paths(ctx, p)["pool"] for p in TEST_POOLS]
    if not all(os.path.exists(p) for p in paths):
        return None
    return hashlib.sha256("|".join(sha_lf(p) for p in paths).encode()).hexdigest()


def refuse_rerun_once_uploaded(ctx, a, cmd):
    """seam / apply / submit --rerun after this 10 run is in the upload log: refused (2), as tools/t06_player.py does
    for 06 (a rewritten report would make the uploaded 10 not meet PREREG_10 4.4)."""
    if not a.rerun:
        return
    rid = current_run_id(ctx)
    logged = run_in_log(ctx, rid) if rid else []
    if logged:
        refuse(f"{cmd} --rerun: the upload log already holds 10 entries {logged} of this 10 run (run_id {rid[:12]}); "
               f"rewriting its outputs would make them not meet PREREG_10 4.4. A change needs a PREREG_10 revision")


def inputs_bound(ctx, pre, has_off, sources=None):
    files = [pre["t02"]["submission"], pre["t02"]["submit_report"], pre["t02"]["seam_local"],
             os.path.join(ctx["C"], "player", "t1.json"), os.path.join(ctx["C"], "player", "t4.json"),
             os.path.join(ctx["O2"], "seam_val", "apply_report.json"), P02_FROZEN]
    for p in TEST_POOLS:
        tp = pre["t02"]["pools"][p]
        files += [tp["apply_report"], tp["pool_select"], tp["top5"]]
    files += [os.path.join(ctx["D2"], "player", p, "dumps", n) for p in t6.POOLS for n in ("cand_games.npz",
                                                                                         "query_games.npz")]
    files += [os.path.join(ctx["P6"], n) for n in ("frozen_config.json", "lda_B.npz", "lda_W.npz", "score_H6P.json")]
    files += [t06_paths(ctx, "seam_val")["seam"]] + [t06_paths(ctx, p)["feats_report"] for p in t6.POOLS]
    for p in TEST_POOLS:
        files += [t06_paths(ctx, p)["pool"], t06_paths(ctx, p)["off"]]
    if has_off:
        files += [os.path.join(ctx["O6"], "off", t6.SUBMISSION), os.path.join(ctx["O6"], "off", t6.REPORT)]
    op = out_paths(ctx, "seam_val")
    files += [op["seam"], op["seam_top5"]]
    for p in TEST_POOLS:
        files += [out_paths(ctx, p)["pool"], out_paths(ctx, p)["top5"]]
    files += tb.b_inputs(ctx, pre.get("b"), sources)
    return {t6.rel_key(f): t6.sha_bound(f) for f in files if os.path.exists(f)}


def b_submit_extra(ctx, pre, pools):
    """The B part of the submit report (B adopted): the GPU seam, tol_P, K5-10 for B, which pools are 10B."""
    if not pre["b"]["adopted"]:
        return {"not_used_reason": pre["b"].get("reason"), "problems": pre["b"].get("problems")} \
            if pre["b"].get("reason") not in (None,) + tb.SILENT else None
    try:                                       # rev 4 section 14: a broken B record is 'B not used', never a stop
        seam = tb.gpu_seam_state(me(), ctx, required=False)
    except Exception as e:  # noqa: BLE001
        seam = {"passed": False, "error": f"{type(e).__name__}: {e}"}
    try:
        tol = tb.tol_state(me(), ctx, pre["b"], required=False)
    except Exception as e:  # noqa: BLE001
        tol = {"error": f"{type(e).__name__}: {e}"}
    gp = tb.p1_paths(ctx)["gpu_seam"]
    return {"gpu_seam": None if seam is None else {"passed": seam.get("passed"),
                                                   "max_abs_projected_diff": seam.get("max_abs_projected_diff"),
                                                   "tol_P": seam.get("tol_P"),
                                                   "sha256_lf": sha_lf(gp) if os.path.exists(gp) else None},
            "tol": tol, "k5_10_b": tb.k5_b_state(me(), ctx),
            "pools_10B": [p for p in TEST_POOLS if pools[p]["report"].get("source") == "10B"],
            "k6b_10": tb.K6B_10.isoformat()}


def cmd_submit(ctx, a):
    t0 = time.time()
    sub_p, rep_p = os.path.join(ctx["O10"], SUBMISSION), os.path.join(ctx["O10"], REPORT)
    pre = common_prereqs(ctx, "submit")
    rerun = guard_existing(ctx, a, [sub_p, rep_p])
    seam10, seam10_sha = load_seam(ctx, pre)
    t4_path = os.path.join(ctx["C"], "player", "t4.json")
    pools, rows, qfiles, cfiles = {}, [], [], []
    for pool in TEST_POOLS:
        op = out_paths(ctx, pool)
        rep = read_json(op["pool"], f"run apply --pool {pool} first")
        require_committed(ctx, [op["pool"], op["top5"]], f"the 10 outputs of {pool}")
        if rep.get("format") != FORMATS["pool"] or rep.get("mode") != ctx["mode"]:
            refuse(f"{shown(op['pool'])}: format {rep.get('format')!r} mode {rep.get('mode')!r}")
        if sha_raw(op["top5"]) != (rep.get("outputs") or {}).get("top5.csv"):
            refuse(f"{pool}: top5.csv changed after apply")
        tp = pre["t02"]["pools"][pool]
        op6 = t06_paths(ctx, pool)
        ins = rep.get("inputs") or {}
        stale = []
        if ins.get("t10_seam_sha256_lf") != seam10_sha or ins.get("t06_seam_sha256_lf") != pre["seam6_sha"]:
            stale.append("a seam changed")
        if (rep.get("test02") or {}).get("apply_report_sha256_lf") != sha_lf(tp["apply_report"]) or \
                (rep.get("test02") or {}).get("top5_sha256") != sha_raw(tp["top5"]):
            stale.append("test02's reports changed")
        if (rep.get("t06") or {}).get("t06_pool_sha256_lf") != sha_lf(op6["pool"]) or \
                (rep.get("t06") or {}).get("top5_off_sha256") != sha_raw(op6["off"]):
            stale.append("the 06 pool report changed")
        if ins.get("t06_feats_report_sha256_lf") != {p: v["sha256_lf"] for p, v in pre["feats6"].items()}:
            stale.append("the 06 feats changed")
        if ins.get("t4_sha256_lf") != (sha_lf(t4_path) if os.path.exists(t4_path) else None):
            stale.append("T4 changed")
        rb = rep.get("b") or {}
        bp = rb.get("pool") or {}
        if bp.get("applied") and (not pre["b"]["adopted"] or rb.get("frozen_sha256_lf") != pre["b"].get(
                "frozen_sha256_lf")):
            stale.append("the B state changed since this pool got 10B (apply again: B problems now degrade to A)")
        if bp.get("reason") == "p1_not_done":
            print(f"note: {pool} was applied before its P1 was done (A only). To add B, run p1 and then apply "
                  f"--pool {pool} --rerun before K6B-10 (docs/TESTDAY_10.md step 2B)")
        p1r = tb.p1_paths(ctx, pool)["report"]
        if bp.get("applied") and bp.get("p1_report_sha256_lf") != (sha_lf(p1r) if os.path.exists(p1r) else None):
            stale.append("the pool's P1 report changed")
        if (rep.get("source") == "10B") != bool(bp.get("applied")):
            stale.append("source 10B without an applied B record (or the reverse)")
        if stale:
            refuse(f"{pool}: {stale} since the 10 apply. " + MSG_REDO)
        if rep.get("source") not in SOURCES:
            refuse(f"{pool}: source {rep.get('source')!r}")
        body = t6.read_top5(op["top5"])
        if [r[0] for r in body] != [r[0] for r in t6.read_top5(tp["top5"])]:
            refuse(f"{pool}: top5.csv rows are not in test02's (question-file) order")
        rows += body
        q, c, _pd = t6.pool_files(ctx, pool)
        qfiles.append(q)
        cfiles.append(c)
        pools[pool] = {"report": rep, "rows": body, "path": op["pool"]}
    off_rows, off_rec = t06_off_state(ctx, pre)
    q04_rows = t6.read_top5(pre["t02"]["submission"])
    ref_name, ref_rows = ("06-OFF", off_rows) if off_rows is not None else ("Q04", q04_rows)
    run_id = current_run_id(ctx)
    logged = run_in_log(ctx, run_id)
    if logged:
        refuse(f"the upload log already holds 10 entries {logged} of this 10 run (run_id {run_id[:12]}): rewriting "
               f"t10_submit_report.json would make them not meet PREREG_10 4.4")
    k5 = k5_10_state(ctx)
    if k5["checked"] and not k5["ok"]:
        not_used(ctx, "submit", "k5_10", f"K5-10 不成立（{k5}）：A 沒趕上，這一季不用 10（PREREG_10 3）")
    if rows == ref_rows:
        for p in (sub_p, rep_p):
            if os.path.exists(p):
                os.remove(guard_write(ctx, p))
        not_used(ctx, "submit", "10_not_needed", f"兩池的列都和 {ref_name} 逐列相同：10 不需要，不產生交件檔（PREREG_10 4.2）")
    write_rows(ctx, sub_p, rows)
    errs = check_submission.check("player", sub_p, qfiles, cfiles, official=True)
    checks = {"official": True, "passed": not errs, "errors": errs[:50], "n_errors": len(errs),
              "question_files": [{"path": shown(x), "sha256": sha_raw(x)} for x in qfiles]}
    by = {}
    i = 0
    for pool in TEST_POOLS:
        n = len(pools[pool]["rows"])
        by[pool] = (ref_rows[i:i + n], q04_rows[i:i + n], off_rows[i:i + n] if off_rows is not None else None)
        i += n
    with open(sub_p, "rb") as f:
        raw = f.read()
    rep = {"format": FORMATS["submit"], "mode": ctx["mode"], "method": METHOD,
           "what": "PREREG_10 4.2 submit: submission_player.csv = public rows, then private rows (test02's order), "
                   "question_id,top_1..top_5, UTF-8 without BOM, LF",
           "run_id": run_id, "check_submission": checks,
           "output": {"path": t6.shown(sub_p), "sha256": hashlib.sha256(raw).hexdigest(), "rows": raw.count(b"\n") - 1,
                      "bytes": len(raw)},
           "sources": {p: pools[p]["report"]["source"] for p in TEST_POOLS},
           "a_holds": {p: pools[p]["report"]["a_holds"] for p in TEST_POOLS},
           "reference": ref_name,
           "top1_differences_vs_06_off": ({p: t6.top1_diff(pools[p]["rows"], by[p][2]) for p in TEST_POOLS}
                                          if off_rows is not None else None),
           "top1_differences_vs_q04": {p: t6.top1_diff(pools[p]["rows"], by[p][1]) for p in TEST_POOLS},
           "pools": {p: {k: pools[p]["report"].get(k) for k in ("source", "rule", "a_holds", "selected", "sr06",
                                                                 "gate06_vs_r02prime", "test02", "t06")}
                     | {"b": (pools[p]["report"].get("b") or {}).get("pool")}
                     | {"t10_pool_sha256_lf": sha_lf(pools[p]["path"]),
                        "a_conditions": {k: v["ok"] for k, v in pools[p]["report"]["a_rule"]["conditions"].items()}}
                     for p in TEST_POOLS},
           "t06_off": off_rec, "q04_submission": {"path": shown(pre["t02"]["submission"]),
                                                  "sha256": pre["t02"]["submission_sha256"]},
           "seam": {"path": shown(out_paths(ctx, "seam_val")["seam"]), "sha256_lf": seam10_sha,
                    "passed": seam10.get("passed")},
           "t06_seam": {"sha256_lf": pre["seam6_sha"], "passed": pre["seam6"].get("passed")},
           "h6p": pre["h6p"], "t1": pre["t1"], "q04_upload": pre["q04_upload"], "k5_10": k5,
           "b": b_record(pre["b"], b_submit_extra(ctx, pre, pools)),
           "k6_10": K6_10.isoformat(), "inputs_bound": inputs_bound(ctx, pre, off_rows is not None,
                                                                      {p: pools[p]["report"].get("source")
                                                                       for p in TEST_POOLS}),
           "inputs_bound_rule": "sha256 with CRLF folded to LF for .csv / .json, raw bytes otherwise",
           "frozen": pre["frozen"], "rerun": rerun, "tools": tool_record(), "env": t6.env_record(),
           "created": now_iso(), "seconds": round(time.time() - t0, 1)}
    write_json(ctx, rep_p, rep)
    if errs:
        os.remove(guard_write(ctx, sub_p))
        refuse(f"check_submission --official failed: {errs[:3]}; the file was removed (a local bug: stop and check)")
    print("submit: 10 %s (%d rows, sha256 %s); sources %s; top-1 vs %s %s, vs Q04 %s" % (
        shown(sub_p), len(rows), rep["output"]["sha256"][:12], rep["sources"], ref_name,
        rep["top1_differences_vs_06_off"] if off_rows is not None else rep["top1_differences_vs_q04"],
        rep["top1_differences_vs_q04"]))
    return 0


# ================================================================ status and B
def cmd_status(ctx, a):
    now = ctx["now"]
    try:
        b = tb.b_state(me(), ctx)
    except T10Error as e:
        b = {"adopted": False, "error": str(e)}
    lines = [f"10 test day ({ctx['mode']}; B {'adopted' if b['adopted'] else 'not adopted'})"]
    s = maybe_json(out_paths(ctx, "seam_val")["seam"])
    lines.append(f"  seam: {s.get('verdict') if s else 'missing'}")
    for pool in TEST_POOLS:
        r = maybe_json(out_paths(ctx, pool)["pool"])
        lines.append(f"  apply {pool}: " + ("missing" if not r else f"{r.get('source')} (A {'holds' if r.get('a_holds') else 'no'})"))
    r = maybe_json(os.path.join(ctx["O10"], REPORT))
    lines.append("  submit: " + (f"{str((r.get('output') or {}).get('sha256'))[:12]} sources {r.get('sources')}"
                                 if r else "not produced"))
    nu = maybe_json(os.path.join(ctx["O10"], "not_used.json"))
    if nu:
        lines.append(f"  10 not used: {[e['reason'] for e in nu.get('events', [])]} -> {MSG_FINAL}")
    if b.get("error"):
        lines.append(f"  B: the PREREG_10 B files are not consistent: {b['error']}")
    else:
        lines += tb.status_lines(me(), ctx, b)
    lines.append(f"  距 K6B-10 11/14 23:59（B 採用時兩池的 P1 和 GPU 接縫）：{left(now, K6B_10)}"
                 f"{'（B 沒有採用，不適用）' if not b['adopted'] else ''}；距 K6-10 11/21 23:59（10 的正式執行要在這之前 commit）："
                 f"{left(now, K6_10)}")
    print("\n".join(lines))
    return 0


def cmd_p1(ctx, a):
    """PREREG_10 4.2 p1 (part B, 4090; tools/t10_b.py). B not adopted: the registered 'B not used' (exit 3), nothing
    written, before any other check (as rev 3)."""
    tb.require_adopted(me(), ctx, tb.b_state(me(), ctx), f"p1 --pool {a.pool}")
    pre = common_prereqs(ctx, f"p1 --pool {a.pool}", a.pool)
    return tb.cmd_p1(me(), ctx, a, pre, pre["b"])


def cmd_rehearse(ctx, a):
    """PREREG_10 rev 4: the rehearsal of the GPU seam before test data (m, tol_P). B not adopted: exit 3."""
    b = tb.b_state(me(), ctx)
    tb.require_adopted(me(), ctx, b, "p1-rehearse", before_test_day=ctx["now"] < tb.TEST_OPEN)
    from_t06(ctx, "p1-rehearse", None, t6.check_venv)
    if ctx["real"]:
        clean, dirty = t6.git_clean_tracked()
        if not clean:
            refuse(f"git is not clean (tracked changes: {str(dirty)[:300]}): commit first")
        require_committed(ctx, [os.path.join(REPO, *SELF_REL.split("/")), os.path.join(HERE, "t10_b.py")],
                          "tools/t10_player.py and tools/t10_b.py")
    return tb.cmd_rehearse(me(), ctx, a, b)


# ================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="t10_player.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dry-run", action="store_true")
        for k in ("--out-dir", "--t06-out-dir", "--t06-data-dir", "--t02-results-dir", "--t02-out-dir", "--tests-dir",
                  "--checks-dir", "--log", "--prereg06-dir", "--now"):
            p.add_argument(k, default=None)
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
    def bopts(p):
        for k in ("--prereg10-dir", "--p1-out-dir", "--p1-data-dir"):
            p.add_argument(k, default=None)
    for name in ("seam", "submit", "status"):
        p = sub.add_parser(name)
        common(p)
        bopts(p)
    p = sub.add_parser("apply")
    common(p)
    bopts(p)
    p.add_argument("--pool", required=True, choices=TEST_POOLS)
    for name in ("p1", "p1-rehearse"):
        p = sub.add_parser(name)
        common(p)
        bopts(p)
        if name == "p1":
            p.add_argument("--pool", required=True, choices=("seam_val",) + tuple(TEST_POOLS))
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--max-games", type=int, default=0)
        p.add_argument("--fake-scorer", action="store_true")
        p.add_argument("--synthetic-p1", action="store_true")
    return ap


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    try:
        a = build_parser().parse_args(argv)
        ctx = resolve(a)
        fn = {"seam": cmd_seam, "apply": cmd_apply, "submit": cmd_submit, "status": cmd_status, "p1": cmd_p1,
              "p1-rehearse": cmd_rehearse}[a.cmd]
        return fn(ctx, a)
    except T10Error as e:
        print(f"t10_player: {e} (exit {e.code})", file=sys.stderr)
        return e.code
    except t6.T06Error as e:
        print(f"t10_player: {e} (exit {e.code})", file=sys.stderr)
        return e.code
    except t6.st.Refused as e:
        print(f"t10_player: {e} (exit 2)", file=sys.stderr)
        return 2
    except SystemExit as e:
        if isinstance(e.code, int):
            return e.code
        print(f"t10_player: {e} (exit 2)", file=sys.stderr)
        return 2
    except Exception:     # noqa: BLE001
        traceback.print_exc()
        print("t10_player: unexpected internal error (exit 2): stop and check the local tool", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
