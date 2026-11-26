"""C10 test10, competition 1: part B of the 10 test-day kit (PREREG_10 sections 2, 4.2-4.4; rev 4, section 13).

B adds w*·z(T_P1), the KataGo HumanSL neutral-probe (P1) LDA term frozen in results/prereg10/player, to the pools of 10
that used 06 (06-OFF or 10A). Nothing of B runs until B is ADOPTED (b_state): results/prereg10/RESULTS.md carries the
line B_ADOPTED_MARK, results/prereg10/player/score_H10P.json is committed and meets the registered non-inferiority rule
(B - A point estimate >= 0 and 95 % lower bound > -0.010, reference re-read), frozen_config.json says B passed. Until
then every B path is the registered "B not adopted" (exit 3, nothing written) and 10 is part A only, exactly as rev 3.
PREREG_10 rev 5 (section 16): when the PREREG_10 path is 'not_marked', B can also be ADOPTED through PREREG_13
(results/prereg13/RESULTS.md carries B13_ADOPTED_MARK, score_H13P.json meets PREREG_13 6.3 with the n_view scaling,
view_report.json score_allowed true and bound to the audit as it was before apply --finalize, the ledger's H13P rows,
K3-13; K5 for B also needs the rev 5 row merged into main by K5-13). B itself is the same frozen B either way.

Called by tools/t10_player.py (never run directly):
    p1 --pool seam_val           GPU seam: the first 400 validation games extracted again on the 4090 and compared with
                                 the committed results/prereg10/feat/val_seam_feats.npz: failed flags and target-move
                                 counts equal, the per-game max |difference| of the frozen-LDA projection (T_P1's
                                 64 dims) over the usable games <= tol_P. Failed: B is not used this season (exit 3,
                                 for B only; 10 keeps part A).
    p1 --pool public|private     the pool's candidate games (file order) then question games (question / slot order),
                                 target side = the row's colour (official: black). Chunks of 200 games in
                                 data/test10/p1/<pool>/, resumable; STOP file there -> exit 5 after the chunk in
                                 progress. Report results/test10/p1/<pool>_p1_report.json.
    p1-rehearse                  BEFORE any test data (now < 11/04 11:00, no official test file in data/test): the
                                 same 400-game extraction and comparison; m = the max projected difference;
                                 tol_P = max(0.05, 2.5 m) -> results/test10/p1/p1_rehearsal.json. Real-mode p1 refuses
                                 (exit 2) until this record is committed (PREREG_10 rev 4: tol_P 【待演練】).
    apply (per pool)             10B = C_SLS_k*(F_sel + w*·z(T_P1)); used only when the pool gate sr(10B) - sr(selected)
                                 >= 0 on the pool's own sr06 pseudo pools and P1 failed games <= 0.5 %.
    seam (seam_val)              10B recomputed from the committed validation P1 chunks must equal
                                 results/prereg10/player/player_val_top5.csv (byte for byte when the 06 seam saw
                                 bitwise dumps; else >= 398 / 400 rows), else exit 4.
Scorers: real (.venv-gpu, RTX 4090, the frozen p1_lib / p1_extract, imported only inside the run, after the game
files are read); dry-run only: --fake-scorer (p1_extract.run_chunk with a deterministic CPU stand-in for the probe) and
--synthetic-p1 (no KataGo featurization: a deterministic 770-dim summary of the SGF moves, for the large fake trees).
No tools/score.py, no label file; data/local only for the two validation game files (the p06 guard).
"""

import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
P1_DIR = os.path.join(REPO, "results", "prereg08", "p1")
for _p in (HERE, P1_DIR):                         # P1_DIR: the spawn workers unpickle p1_extract._featurize
    if _p not in sys.path:
        sys.path.append(_p)
import numpy as np  # noqa: E402

import t06_player as t6  # noqa: E402  (p06 guard; the 06 functions)
import p10_fuse as F  # noqa: E402  (numpy only: load_side0, project_p1, z_p1, fuse_10b, load_lda)
import p10_apply as PA  # noqa: E402  (val_keys, keys_sha, sr06_with_p1, frozen_usable)
import p02_player_fuse as pf  # noqa: E402
import p02_moves as pm  # noqa: E402

st = t6.st

# ------------------------------------------------------------------ paths (module constants; tests may setattr)
RESULTS = os.path.join(REPO, "results")
REAL_P10 = os.path.join(RESULTS, "prereg10")
REAL_P1_OUT = os.path.join(RESULTS, "test10", "p1")
REAL_P1_DATA = os.path.join(REPO, "data", "test10", "p1")
LEDGER = os.path.join(RESULTS, "score_ledger.csv")
P06_PLAYER = os.path.join(RESULTS, "prereg06", "player")
P10_DOC = "docs/PREREG_10_PLAYER_GUARD_P1.md"

# ------------------------------------------------------------------ registered values (PREREG_10 2, 3, 4.2; rev 4)
TW = t6.TW
K5_10 = dt.datetime(2026, 10, 31, 23, 59, 59, tzinfo=TW)     # 3: B's test-day revision (rev 4) committed and merged
K6B_10 = dt.datetime(2026, 11, 14, 23, 59, 59, tzinfo=TW)    # 3: both pools' P1 and the GPU seam committed
K3_10 = dt.datetime(2026, 10, 25, 23, 59, 59, tzinfo=TW)     # 3: H10P, RESULTS.md committed (else no B this season)
TEST_OPEN = dt.datetime(2026, 11, 4, 11, 0, 0, tzinfo=TW)    # the test set opens: the rehearsal comes before
B_REV_ROW = "| rev 4 |"                                      # B's test-day revision of PREREG_10 (K5-10 for B)
B_ADOPTED_MARK = "H10P：B **採用**"                          # the line results/prereg10/RESULTS.md carries when adopted
H10P_POINT_MIN = 0.0                                         # 2.6: B - A point estimate >= 0
H10P_LOW_ABOVE = -0.010                                      # 2.6: 95 % lower bound > -0.010
H10P_REF_TOL = 1e-12                                         # 2.6: A side re-read = score_H6P.json's B side
H10P_LEDGER_PREFIX = "prereg10 H10P"
H10P_LEDGER_ROWS = (2, 4)                                    # 2.6: 2 rows, or 4 with one logged bug re-run
TOL_P_FLOOR = 0.05                                           # 4.2: tol_P = max(0.05, 2.5 m)
TOL_P_FACTOR = 2.5
SEAM_GAMES = 400
SEAM_MIN_IDENTICAL = t6.SEAM_MIN_IDENTICAL                   # 398 / 400 when the 06 seam's dumps are not bitwise
FAIL_MAX = F.FAIL_MAX                                        # 0.5 % P1 failed games per pool
GATE_B_MIN = 0.0                                             # pool gate sr(10B) - sr(selected) >= 0
CHUNK = 200
GROUP_POSITIONS = 4096
RUN = {"device": "cuda", "precision": "fp16", "batch": 1024, "channels_last": True, "cudnn_benchmark": True}
GPU_NAME = "NVIDIA GeForce RTX 4090"
VAL_SET = PA.VAL_SET
EXIT_STOPPED = 5
SCORERS = ("real", "fake", "synthetic")
FORMATS = {"p1_report": "c10-test10 p1 report v1", "gpu_seam": "c10-test10 gpu seam v1",
           "rehearsal": "c10-test10 p1 rehearsal v1", "b_not_used": "c10-test10 b not used v1",
           "chunk": F.CHUNK_FORMAT}
SYN = {"meta_sha256": "synthetic", "code_sha256": "synthetic", "ckpt_sha256": "synthetic", "ckpt_bytes": 0}
_LABEL = "ans" + "wer"

# ------------------------------------------------------------------ PREREG_10 rev 5 (section 16): B adopted via PREREG_13
REAL_P13 = os.path.join(RESULTS, "prereg13")                 # dry runs: <--prereg10-dir> + "_p13" (p13_dir)
K3_13 = dt.datetime(2026, 10, 25, 23, 59, 59, tzinfo=TW)     # PREREG_13 7: H13P, score_H13P.json, RESULTS.md committed
K5_13 = dt.datetime(2026, 10, 31, 23, 59, 59, tzinfo=TW)     # PREREG_13 7: PREREG_10 rev 5 + t10 merged (--no-ff)
B13_REV_ROW = "| rev 5 |"                                    # PREREG_10 rev 5 (K5-13 for the PREREG_13 path)
B13_ADOPTED_MARK = "H13P：B′ **採用**"                       # the line results/prereg13/RESULTS.md carries (PREREG_13 6.4)
H13P_LEDGER_PREFIX = "prereg13 H13P"
H13P_LEDGER_ROWS = (2, 4)                                    # PREREG_13 6.5: 2 rows, or 4 with one logged bug re-run
H13P_POINT_MIN = 0.0                                         # PREREG_13 6.3: B - A point estimate >= 0
H13P_LOW_PER_400 = -0.010                                    # 6.3: 95 % lower bound > -0.010 * n_view / 400
H13P_LEDGER_MEAN_TOL = 1e-12
P13_FORMATS = {"view": "c10-prereg13 player view_report v1", "audit": "c10-prereg13 player prescore_audit v1"}
P13_OUT_A, P13_OUT_B = "view_06_top5.csv", "view_10b_top5.csv"
P13_AUDIT_INPUTS = ("p1_lda.npz", "frozen_config.json", "val_report.json")
VIEW_FINALIZE_KEYS = ("score_allowed", "score_blocked_reasons", "audit", "finalized", "finalized_by_tool")  # p13_view


# ================================================================ small helpers
def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def tol_p(m):
    """PREREG_10 4.2: tol_P = max(0.05, 2.5 m)."""
    return max(TOL_P_FLOOR, TOL_P_FACTOR * float(m))


def p10_paths(p10):
    pl, fe = os.path.join(p10, "player"), os.path.join(p10, "feat")
    return {"results_md": os.path.join(p10, "RESULTS.md"), "frozen": os.path.join(pl, "frozen_config.json"),
            "lda": os.path.join(pl, "p1_lda.npz"), "h10p": os.path.join(pl, "score_H10P.json"),
            "apply": os.path.join(pl, "apply_report.json"), "val_top5": os.path.join(pl, "player_val_top5.csv"),
            "val_report": os.path.join(fe, "val_report.json"), "val_seam": os.path.join(fe, "val_seam_feats.npz"),
            "val_dir": os.path.join(p10, "val") if os.path.normcase(os.path.abspath(p10)) != os.path.normcase(REAL_P10)
            else os.path.join(REPO, "data", "prereg10", "val")}


def p13_dir(ctx):
    """PREREG_10 rev 5: the PREREG_13 folder. Real: results/prereg13. Dry runs: the --prereg10-dir path + '_p13', so the
    A trees (absent prereg10 folder) and the H10P fake trees and their drill copies never see a PREREG_13 tree."""
    if ctx["real"]:
        return REAL_P13
    return os.path.abspath(ctx["P10"]) + "_p13"     # a dry run never sees results/prereg13, even with the real P10


def p13_paths(p13):
    pl = os.path.join(p13, "player")
    return {"results_md13": os.path.join(p13, "RESULTS.md"), "h13p": os.path.join(pl, "score_H13P.json"),
            "view": os.path.join(pl, "view_report.json"), "audit": os.path.join(pl, "prescore_audit.json"),
            "view_a": os.path.join(pl, P13_OUT_A), "view_b": os.path.join(pl, P13_OUT_B)}


def p1_paths(ctx, pool=None):
    o, a = ctx["O10P1"], ctx["A10P1"]
    out = {"gpu_seam": os.path.join(o, "gpu_seam.json"), "rehearsal": os.path.join(o, "p1_rehearsal.json"),
           "not_used": os.path.join(o, "b_not_used.json"), "rehearsal_dir": os.path.join(a, "rehearsal")}
    if pool:
        out.update(report=os.path.join(o, f"{pool}_p1_report.json"), chunks=os.path.join(a, pool))
    return out


def bstop(T, msg, code, reason):
    raise T.T10Error(msg, code, reason)


def record_b_not_used(ctx, cmd, reason, message, pool=None):
    """Append an event to results/test10/p1/b_not_used.json (B is not used in this step / pool; 10 keeps part A).
    Never O10/not_used.json (that file is 10's own 'not used'). Only inside the B output folder."""
    path = p1_paths(ctx)["not_used"]
    if not t6.abs_inside(os.path.abspath(path), ctx["O10P1"]):
        raise RuntimeError(f"refusing to write {path}")
    rec = t6.maybe_json(path) or {"format": FORMATS["b_not_used"], "events": []}
    rec["events"].append({"command": cmd, "pool": pool, "reason": reason, "message": message, "mode": ctx["mode"],
                          "created": now_iso(), "now": ctx["now"].isoformat(), "git_head": t6.git_head()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(t6.jsonable(rec), f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(path + ".tmp", path)


def b_not_used(T, ctx, cmd, reason, message, pool=None):
    """B is not used this season (exit 3, for B only: 10 keeps part A), recorded in b_not_used.json (commit it)."""
    record_b_not_used(ctx, cmd, reason, message, pool)
    bstop(T, message + "；B 這一季不用，10 只有 A（PREREG_10 4.2、rev 4）", 3, reason)


# ================================================================ is B adopted? (PREREG_10 2.6, rev 4)
MARK_RE = re.compile(r"^\s*(?:[-*+]\s*)?H10P\s*[：:]\s*B\s*\*\*(採用|不採用)\*\*")
MARK13_RE = re.compile(r"^\s*(?:[-*+]\s*)?H13P\s*[：:]\s*B′\s*\*\*(採用|不採用)\*\*")    # PREREG_10 rev 5 (16.1)
SILENT = ("not_marked", "h10p_not_adopted", "h13p_not_adopted")   # the registered 'B not adopted': exit 3, nothing
#                                                                    written (rev 3; rev 5 adds PREREG_13's)


def results_md_marks(path, regex=MARK_RE):
    """The adoption marks of results/prereg10/RESULTS.md: lines 'H10P：B **採用**' / 'H10P：B **不採用**' (a bullet
    before, an ASCII colon, text after are accepted). -> list of '採用' / '不採用'. regex=MARK13_RE: the 'H13P：B′'
    lines of results/prereg13/RESULTS.md (PREREG_10 rev 5)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [m.group(1) for m in (regex.match(x) for x in f.read().splitlines()) if m]


def results_md_adopted(path, regex=MARK_RE):
    return results_md_marks(path, regex) == ["採用"]


def ledger_rows(prefix, path=None):
    import csv
    path = path or LEDGER
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return [r for r in rows[1:] if len(r) > 2 and r[2].startswith(prefix)]


def ledger_h10p_rows(path=None):
    return len(ledger_rows(H10P_LEDGER_PREFIX, path))


def h10p_decision(score, h6p_b_mean):
    """2.6 on score_H10P.json (score.py compare-player): -> (adopted, record). The A side must re-read H6P's B side."""
    bma = score.get("B_minus_A") or {}
    point, ci = bma.get("mean_diff"), bma.get("ci95") or [None, None]
    low = ci[0] if ci else None
    ref = score.get("A_mean")
    ref_ok = finite(ref) and finite(h6p_b_mean) and abs(ref - h6p_b_mean) <= H10P_REF_TOL
    adopted = bool(finite(point) and finite(low) and point >= H10P_POINT_MIN and low > H10P_LOW_ABOVE and ref_ok)
    return adopted, {"mean_diff": point, "ci95_low": low, "A_mean": ref, "h6p_B_mean": h6p_b_mean,
                     "reference_reread_equal": ref_ok,
                     "rule": "B - A >= 0 and 95 % lower bound > -0.010 (PREREG_10 2.6), A side = score_H6P B side"}


def h13p_decision(score, n_view):
    """PREREG_13 6.3 on score_H13P.json: B - A point estimate >= 0 and 95 % lower bound > -0.010 * n_view / 400
    (n_view from view_report.json). -> (adopted, record)."""
    bma = score.get("B_minus_A") or {}
    point, ci = bma.get("mean_diff"), bma.get("ci95") or [None, None]
    low = ci[0] if ci else None
    nv_ok = isinstance(n_view, int) and not isinstance(n_view, bool) and 0 < n_view <= 400
    bound = H13P_LOW_PER_400 * n_view / 400 if nv_ok else None
    adopted = bool(nv_ok and finite(point) and finite(low) and point >= H13P_POINT_MIN and low > bound)
    return adopted, {"mean_diff": point, "ci95_low": low, "n_view": n_view, "ci95_low_must_exceed": bound,
                     "A_mean": score.get("A_mean"), "B_mean": score.get("B_mean"),
                     "rule": "B - A >= 0 and 95 % lower bound > -0.010 * n_view / 400 (PREREG_13 6.3)"}


def mark_first_commit(path, word, regex=MARK_RE):
    """Committer time of the first commit whose RESULTS.md has a mark LINE (MARK_RE) with this word (real mode,
    K3-10): commits that touched the file, oldest first, read at each commit (other '**採用**' text does not count).
    regex=MARK13_RE: results/prereg13/RESULTS.md (K3-13, PREREG_10 rev 5)."""
    import subprocess
    rel = os.path.relpath(path, REPO).replace(os.sep, "/")
    rc, out = t6.git("log", "--reverse", "--format=%H %cI", "--", rel)
    for line in (out.splitlines() if rc == 0 and out else []):
        h, t = line.split()
        r = subprocess.run(["git", "-C", REPO, "show", f"{h}:{rel}"], capture_output=True, timeout=60)
        text = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""
        if any((m := regex.match(x)) and m.group(1) == word for x in text.splitlines()):
            return t
    return None


def git_show_bytes(rel, commit):
    """The blob of rel at commit (None when it is not there)."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", REPO, "show", f"{commit}:{rel}"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def p10_side_problems(ctx, P, require_allowed=True, h10p_ledger=True):
    """The PREREG_10 files B rests on (rev 4 section 13 item 1): the freeze, the LDA, the apply report (score_allowed
    true only when require_allowed: the PREREG_13 path does not ask for it, PREREG_10 rev 5 16.1), the validation P1
    report; real mode also the H10P ledger rows (h10p_ledger), the mock-path tools and p1_lib / p1_extract.
    -> (problems, frozen_config, val_report)."""
    probs, fc, vr = [], {}, {}
    try:
        fc = t6.read_json(P["frozen"])
        try:
            PA.frozen_usable(fc)
        except F.Refused as e:
            probs.append(f"frozen_config.json: {e}")
        if t6.sha_raw(P["lda"]) != (fc.get("lda") or {}).get("sha256"):
            probs.append("p1_lda.npz differs from frozen_config.json")
        ap = t6.read_json(P["apply"])
        if require_allowed and ap.get("score_allowed") is not True:
            probs.append(f"apply_report.json score_allowed {ap.get('score_allowed')!r} (not finalized)")
        if (ap.get("outputs_sha256") or {}).get("player_val_top5.csv") != t6.sha_lf(P["val_top5"]):
            probs.append("player_val_top5.csv is not the file apply_report.json records")
        if float(ap.get("w_star", float("nan"))) != float(fc.get("w_star", float("nan"))):
            probs.append("apply_report.json w* differs from frozen_config.json")
        vr = t6.read_json(P["val_report"])
        if vr.get("format") != F.FORMATS["val_report"] or not vr.get("complete"):
            probs.append("val_report.json is not a complete validation extraction")
        if (vr.get("seam") or {}).get("sha256") != t6.sha_raw(P["val_seam"]):
            probs.append("val_seam_feats.npz is not the file val_report.json records")
        if ctx["real"]:
            if vr.get("mode") != "real":
                probs.append(f"val_report.json mode {vr.get('mode')!r}")
            if vr.get("frozen_config_sha256") != t6.sha_lf(P["frozen"]):
                probs.append("val_report.json was made under another frozen_config.json")
            if h10p_ledger:
                n = ledger_h10p_rows()
                if n not in H10P_LEDGER_ROWS:
                    probs.append(f"results/score_ledger.csv holds {n} '{H10P_LEDGER_PREFIX}' rows (registered "
                                 f"{H10P_LEDGER_ROWS}: H10P, or H10P and one logged bug re-run)")
            diff = F.tools_differ(F.tool_state(), fc.get("tools"))
            if diff:
                probs.append(f"mock-path tools changed since the freeze: {diff} (T_P1 / fusion must be the frozen "
                             f"code)")
            for r, want_sha in (fc.get("p1_code_sha256") or {}).items():
                if t6.sha_lf(os.path.join(REPO, *r.split("/"))) != want_sha:
                    probs.append(f"{r} differs from frozen_config.json")
    except Exception as e:  # noqa: BLE001
        probs.append(f"B files unreadable: {type(e).__name__}: {e}")
    return probs, fc, vr


def adopted_record(base, P, fc, vr, files, source, **extra):
    return dict(base, adopted=True, reason=None, source=source, w_star=float(fc["w_star"]),
                csls_k=int(fc.get("csls_k", F.CSLS_K)),
                frozen_sha256_lf=t6.sha_lf(P["frozen"]), lda_sha256=t6.sha_raw(P["lda"]),
                val_seam_sha256=t6.sha_raw(P["val_seam"]), val_keys_sha256=vr.get("keys_sha256"),
                val_games=vr.get("games"), checkpoint=fc.get("checkpoint"),
                neutral_metadata_sha256=fc.get("neutral_metadata_sha256"),
                chunk_code_sha256=fc.get("chunk_code_sha256"), paths=P,
                files_sha256={t6.rel_key(p): t6.sha_bound(p) for p in files.values()}, **extra)


def b10_state(T, ctx):
    """The PREREG_10 path (rev 4 section 14 item 2, unchanged by rev 5). -> B record (see b_state)."""
    P = p10_paths(ctx["P10"])
    base = {"adopted": False, "prereg10_dir": t6.shown(ctx["P10"]), "problems": [],
            "rule": "PREREG_10 2.6 / rev 4: B is adopted when score_H10P.json (committed by K3-10) meets the "
                    f"non-inferiority rule, results/prereg10/RESULTS.md carries exactly one mark {B_ADOPTED_MARK!r} "
                    "and the freeze / apply / validation P1 files agree; rev 5: or through PREREG_13 "
                    f"(results/prereg13/RESULTS.md {B13_ADOPTED_MARK!r}, score_H13P.json, view_report.json)"}
    marks = results_md_marks(P["results_md"])
    has_h10p = os.path.exists(P["h10p"]) and (not ctx["real"] or t6.committed_clean(P["h10p"]))
    if not has_h10p:
        if not marks:
            return dict(base, reason="not_marked")
        return dict(base, reason="inconsistent", problems=[f"RESULTS.md has marks {marks} but score_H10P.json is not "
                                                           f"committed"])
    h6p = t6.read_json(os.path.join(P06_PLAYER, "score_H6P.json"))
    try:
        dec, h10p = h10p_decision(t6.read_json(P["h10p"]), h6p.get("B_mean"))
    except Exception as e:  # noqa: BLE001
        return dict(base, reason="inconsistent", problems=[f"score_H10P.json unreadable: {e}"])
    base["h10p"] = h10p
    want = "採用" if dec else "不採用"
    if marks != [want]:
        return dict(base, reason="inconsistent",
                    problems=[f"score_H10P.json says {want} (PREREG_10 2.6) but RESULTS.md has the marks {marks} "
                              f"(exactly one, consistent, is required)"])
    if not dec:
        return dict(base, reason="h10p_not_adopted")
    probs = []
    if ctx["real"]:
        if not t6.committed_clean(P["results_md"]):
            probs.append("results/prereg10/RESULTS.md is not committed and clean")
        t_h = t6.git_commit_time(P["h10p"])
        t_m = mark_first_commit(P["results_md"], "採用")
        late = {n: t for n, t in (("score_H10P.json", t_h), ("RESULTS.md mark", t_m))
                if t is None or t6.parse_iso(t, "commit time") > K3_10}
        base["k3_10"] = {"deadline": K3_10.isoformat(), "score_H10P_commit": t_h, "mark_commit": t_m}
        if late and not probs:
            return dict(base, reason="k3_10", problems=[f"committed after K3-10 {K3_10.isoformat()}: {late}"])
    missing = [t6.shown(p) for k, p in P.items() if k != "val_dir" and not os.path.exists(p)]
    if missing:
        return dict(base, reason="inconsistent", problems=probs + [f"missing {missing}"])
    if ctx["real"]:
        dirty = [t6.shown(p) for k, p in P.items() if k != "val_dir" and not t6.committed_clean(p)]
        dirty += [] if t6.committed_clean(LEDGER) else ["results/score_ledger.csv"]
        if dirty:
            probs.append(f"not committed and clean: {dirty}")
    more, fc, vr = p10_side_problems(ctx, P)
    probs += more
    if probs:
        return dict(base, reason="inconsistent", problems=probs)
    files = {k: p for k, p in P.items() if k != "val_dir"}
    return adopted_record(base, P, fc, vr, files, "PREREG_10", results_md_mark=B_ADOPTED_MARK)


def view_problems(ctx, P, Q, view, fc, score=None):
    """PREREG_10 rev 5 16.1 on results/prereg13/player/view_report.json and prescore_audit.json. The audit is bound to
    the view report BEFORE apply --finalize rewrote it: view_report.json's audit.view_report_sha256_at_audit must equal
    the audit's outputs_sha256['view_report.json'] (and, real mode, the SHA of view_report.json at the audit's commit),
    never the SHA of the current (finalized) file."""
    probs = []
    if view.get("format") != P13_FORMATS["view"]:
        probs.append(f"view_report.json format {view.get('format')!r}")
    if view.get("score_allowed") is not True:
        probs.append(f"view_report.json score_allowed {view.get('score_allowed')!r} (PREREG_13 5.3)")
    if view.get("preview"):
        probs.append("view_report.json is a preview")
    gates = view.get("gates") or {}
    bad = [g for g in ("V1", "V2", "V3") if (gates.get(g) or {}).get("passed") is not True]
    if bad:
        probs.append(f"view_report.json gates not passed: {bad} (PREREG_13 4)")
    if float(view.get("w_star", float("nan"))) != float(fc.get("w_star", float("nan"))):
        probs.append("view_report.json w* differs from frozen_config.json")
    ins = view.get("inputs_sha256") or {}
    for name, got in (("p1_lda.npz", t6.sha_raw(P["lda"])), ("frozen_config.json", t6.sha_lf(P["frozen"])),
                      ("val_report.json", t6.sha_lf(P["val_report"])),
                      ("p10_apply_report.json", t6.sha_lf(P["apply"]))):
        if ins.get(name) != got:
            probs.append(f"view_report.json was made under another {name}")
    outs = view.get("outputs_sha256") or {}
    for name, key in ((P13_OUT_A, "view_a"), (P13_OUT_B, "view_b")):
        if not outs.get(name) or outs.get(name) != t6.sha_lf(Q[key]):
            probs.append(f"{name} is not the file view_report.json records")
    au = t6.read_json(Q["audit"])
    if au.get("format") != P13_FORMATS["audit"]:
        probs.append(f"prescore_audit.json format {au.get('format')!r}")
    if au.get("independent_recompute_identical") is not True:
        probs.append("prescore_audit.json: the independent recompute is not identical")
    if au.get("program_errors_affecting_view") != []:
        probs.append(f"prescore_audit.json program errors {au.get('program_errors_affecting_view')!r}")
    a_outs, a_ins = au.get("outputs_sha256") or {}, au.get("inputs_sha256") or {}
    for name in (P13_OUT_A, P13_OUT_B):
        if a_outs.get(name) != outs.get(name):
            probs.append(f"prescore_audit.json bound another {name}")
    for name in P13_AUDIT_INPUTS:
        if a_ins.get(name) != ins.get(name):
            probs.append(f"prescore_audit.json bound another {name}")
    rec = view.get("audit") or {}
    at_audit = rec.get("view_report_sha256_at_audit")
    if not at_audit or at_audit != a_outs.get("view_report.json"):
        probs.append("view_report.json audit.view_report_sha256_at_audit differs from prescore_audit.json "
                     "outputs_sha256['view_report.json']")
    if rec.get("sha256") != t6.sha_lf(Q["audit"]):
        probs.append("view_report.json audit.sha256 is not the committed prescore_audit.json")
    if ctx["real"]:
        rel_a, rel_v = t6.rel_key(Q["audit"]), t6.rel_key(Q["view"])
        rc, h = t6.git("log", "-1", "--format=%H", "--", rel_a)
        blob = git_show_bytes(rel_v, h) if rc == 0 and h else None
        sha = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() if blob is not None else None
        if sha is None or sha != at_audit:
            probs.append(f"view_report.json at the audit commit {h} has SHA {sha}, not the audited {at_audit}")
        else:
            # the current (finalized) report must be the audited one plus only what apply --finalize writes, so n_view
            # (the adoption bound), the gates, w*, the selection and the bound inputs are the audited values
            try:
                old = json.loads(blob.decode("utf-8"))
            except ValueError as e:
                old = None
                probs.append(f"view_report.json at the audit commit is not JSON ({e})")
            if old is not None:
                strip = lambda d: {k: v for k, v in d.items() if k not in VIEW_FINALIZE_KEYS}  # noqa: E731
                diff = sorted(k for k in set(strip(old)) | set(strip(view)) if old.get(k) != view.get(k))
                if diff:
                    probs.append(f"view_report.json differs from the audited one in {diff} (apply --finalize may only "
                                 f"change {list(VIEW_FINALIZE_KEYS)})")
        for name, obj in (("score_H13P.json", score), ("view_report.json", view), ("prescore_audit.json", au)):
            if isinstance(obj, dict) and "fake" in obj:
                probs.append(f"{name} carries a 'fake' key (tools/t10_b_fake.py): not a real PREREG_13 record")
    return probs


def ledger13_problems(view, score):
    """Real mode: the ledger holds 2 or 4 'prereg13 H13P' rows; the last pair (A, B) scored the two view files and
    carries score_H13P.json's means (PREREG_13 6.1, 6.2, 6.5)."""
    rows = ledger_rows(H13P_LEDGER_PREFIX)
    if len(rows) not in H13P_LEDGER_ROWS:
        return [f"results/score_ledger.csv holds {len(rows)} '{H13P_LEDGER_PREFIX}' rows (registered "
                f"{H13P_LEDGER_ROWS})"]
    probs = []
    outs = view.get("outputs_sha256") or {}
    for r, name, mk in ((rows[-2], P13_OUT_A, "A_mean"), (rows[-1], P13_OUT_B, "B_mean")):
        if len(r) < 8 or r[4] != outs.get(name):
            probs.append(f"the ledger's H13P row for {name} scored another file")
            continue
        try:
            ok = finite(score.get(mk)) and abs(float(r[7]) - score[mk]) <= H13P_LEDGER_MEAN_TOL
        except ValueError:
            ok = False
        if not ok:
            probs.append(f"the ledger's H13P mean for {name} differs from score_H13P.json {mk}")
    return probs


def b13_state(T, ctx, base10):
    """PREREG_10 rev 5 (section 16): B adopted through PREREG_13. Asked only when the PREREG_10 path is 'not_marked'.
    -> B record (see b_state); adopted records carry source 'PREREG_13'."""
    P = p10_paths(ctx["P10"])
    d13 = p13_dir(ctx)
    Q = p13_paths(d13)
    base = dict(base10, prereg13_dir=t6.shown(d13))
    marks = results_md_marks(Q["results_md13"], MARK13_RE)
    has = os.path.exists(Q["h13p"]) and (not ctx["real"] or t6.committed_clean(Q["h13p"]))
    if not has:
        if not marks:
            return dict(base, reason="not_marked")
        return dict(base, reason="inconsistent", problems=[f"results/prereg13/RESULTS.md has marks {marks} but "
                                                           f"score_H13P.json is not committed"])
    try:
        score = t6.read_json(Q["h13p"])
        view = t6.read_json(Q["view"]) if os.path.exists(Q["view"]) else {}
        dec, h13p = h13p_decision(score, (view.get("view") or {}).get("n_view"))
    except Exception as e:  # noqa: BLE001
        return dict(base, reason="inconsistent", problems=[f"score_H13P.json / view_report.json unreadable: {e}"])
    base["h13p"] = h13p
    want = "採用" if dec else "不採用"
    if marks != [want]:
        return dict(base, reason="inconsistent",
                    problems=[f"score_H13P.json says {want} (PREREG_13 6.3, n_view {h13p['n_view']!r}) but "
                              f"results/prereg13/RESULTS.md has the marks {marks} (exactly one, consistent, is "
                              f"required)"])
    if not dec:
        return dict(base, reason="h13p_not_adopted")
    probs = []
    if ctx["real"]:
        if not t6.committed_clean(Q["results_md13"]):
            probs.append("results/prereg13/RESULTS.md is not committed and clean")
        t_h = t6.git_commit_time(Q["h13p"])
        t_m = mark_first_commit(Q["results_md13"], "採用", MARK13_RE)
        late = {n: t for n, t in (("score_H13P.json", t_h), ("RESULTS.md mark", t_m))
                if t is None or t6.parse_iso(t, "commit time") > K3_13}
        base["k3_13"] = {"deadline": K3_13.isoformat(), "score_H13P_commit": t_h, "mark_commit": t_m}
        if late and not probs:
            return dict(base, reason="k3_13", problems=[f"committed after K3-13 {K3_13.isoformat()}: {late}"])
    p10_files = {k: p for k, p in P.items() if k not in ("val_dir", "h10p")}
    files = dict(p10_files, **Q)
    missing = [t6.shown(p) for p in files.values() if not os.path.exists(p)]
    if missing:
        return dict(base, reason="inconsistent", problems=probs + [f"missing {missing}"])
    if ctx["real"]:
        dirty = [t6.shown(p) for p in files.values() if not t6.committed_clean(p)]
        dirty += [] if t6.committed_clean(LEDGER) else ["results/score_ledger.csv"]
        if dirty:
            probs.append(f"not committed and clean: {dirty}")
    more, fc, vr = p10_side_problems(ctx, P, require_allowed=False, h10p_ledger=False)
    probs += more
    if not more:
        try:
            probs += view_problems(ctx, P, Q, view, fc, score)
            if ctx["real"]:
                probs += ledger13_problems(view, score)
        except Exception as e:  # noqa: BLE001
            probs.append(f"PREREG_13 files unreadable: {type(e).__name__}: {e}")
    if probs:
        return dict(base, reason="inconsistent", problems=probs)
    return adopted_record(base, P, fc, vr, files, "PREREG_13", results_md_mark=B13_ADOPTED_MARK, p13_paths=Q,
                          n_view=h13p["n_view"])


def b_state(T, ctx):
    """-> B record; never refuses (PREREG_10 rev 4 section 14: any B problem means 'B not used', 10 continues with A).
    adopted True only when everything holds; 'source' says which registration adopted B: 'PREREG_10' (H10P) or,
    PREREG_10 rev 5 (section 16), 'PREREG_13' (H13P), asked only when the PREREG_10 path is 'not_marked'.
    Otherwise 'reason':
      not_marked / h10p_not_adopted   the registered 'B not adopted' (silent: exit 3 for p1, nothing written)
      h13p_not_adopted                (rev 5) the same for PREREG_13
      inconsistent                    the B files disagree (before the test day p1-rehearse refuses with 2; on the test
                                      day: B not used, recorded)
      k3_10 / k3_13                   score_H10P.json / score_H13P.json or the RESULTS.md mark committed after K3-10 /
                                      K3-13 (B not used)"""
    b = b10_state(T, ctx)
    if b.get("reason") != "not_marked":
        return b
    return b13_state(T, ctx, {k: v for k, v in b.items() if k != "reason"})


def require_adopted(T, ctx, b, cmd, before_test_day=False):
    """p1 / p1-rehearse: B not adopted -> exit 3. The registered 'not adopted' writes nothing (rev 3); other reasons are
    recorded in b_not_used.json; an inconsistency found by the rehearsal (before the test day) is refused (2)."""
    if b["adopted"]:
        return
    if b.get("reason") in SILENT or b.get("reason") is None:
        bstop(T, "B 沒有採用（PREREG_10 2、rev 3／rev 4／rev 5）：p1 不跑，10 只有 A", 3, "b_not_adopted")
    if before_test_day and b.get("reason") == "inconsistent":
        src = "PREREG_13 (results/prereg13, PREREG_10 rev 5)" if b.get("prereg13_dir") and b.get("h13p") is not None \
            else "PREREG_10"
        T.refuse(f"the B files disagree ({src}: {b.get('problems')}): fix them before the test day (PREREG_10 rev 4 / "
                 f"rev 5)")
    b_not_used(T, ctx, cmd, "b_" + str(b.get("reason")), f"B 不能用（{b.get('reason')}：{b.get('problems')}）")


def row_on_main(row, deadline):
    """The row of PREREG_10 is in HEAD and in main's first-parent history, added there (the merge into main, or a direct
    commit on main) at or before deadline."""
    rc, out = t6.git("log", "main", "--first-parent", "--diff-merges=first-parent", "--reverse", "-S" + row,
                     "--format=%H %cI", "--", P10_DOC)
    first = out.splitlines()[0].split() if rc == 0 and out else None
    import subprocess
    try:
        r = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{P10_DOC}"], capture_output=True, timeout=60)
        head = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        head = ""
    in_head = any(line.startswith(row) for line in head.splitlines())
    ok = bool(first and in_head and t6.parse_iso(first[1], "commit time") <= deadline)
    return {"ok": ok, "row": row, "main_commit": first[0] if first else None,
            "committed_at": first[1] if first else None, "deadline": deadline.isoformat(), "in_head": in_head}


def k5_reason(k5):
    """-> (reason, message) for a failed K5 record: K5-10 (B) when the rev 4 row is the problem, K5-13 when only the
    rev 5 row (B adopted through PREREG_13) is."""
    if k5.get("ok") is False and k5.get("error"):
        return "k5_10_b", f"K5（B）查不了：{k5['error']}"
    p13 = k5.get("prereg13")
    if p13 is not None and not p13.get("ok") and k5.get("rev4_ok"):
        return "k5_13", "K5-13 不成立：PREREG_10 rev 5（B 由 PREREG_13 採用）沒有在 10/31 23:59 前合併進 main"
    return "k5_10_b", "K5-10（B）不成立：PREREG_10 rev 4 沒有在 10/31 23:59 前合併進 main"


def k5_b_state(T, ctx, b=None):
    """K5-10 for B (rev 4 section 14): the rev 4 row of PREREG_10 is in HEAD and in main's first-parent history, added
    there (the merge into main, or a direct commit on main) at or before K5-10 (real mode). PREREG_10 rev 5 (16.2):
    when B is adopted through PREREG_13 (b, or b_state when b is None) the rev 5 row must also be there by K5-13."""
    if not ctx["real"]:
        return {"checked": False, "ok": None, "rule": "dry-run: K5-10 (B) is checked in real mode only"}
    r4 = row_on_main(B_REV_ROW, K5_10)
    out = dict(r4, checked=True, rev4_ok=r4["ok"],
               rule="the row is in main's first-parent history by K5-10 (merged into main)")
    if b is None:
        try:
            b = b_state(T, ctx)
        except Exception as e:  # noqa: BLE001  (a record only: never stop the caller, rev 4 section 14)
            return dict(out, ok=False, error=f"b_state: {type(e).__name__}: {e}")
    if b.get("source") == "PREREG_13":
        r5 = row_on_main(B13_REV_ROW, K5_13)
        out.update(ok=bool(r4["ok"] and r5["ok"]), prereg13=dict(r5, rule="PREREG_10 rev 5 (16.2): K5-13, the rev 5 "
                                                                         "row merged into main by 10/31"))
    return out


def load_model(b):
    return F.load_lda(b["paths"]["lda"])


# ================================================================ tol_P (the rehearsal record)
def tol_state(T, ctx, b, required=True, cmd="p1", pool=None):
    """The committed rehearsal record (real: committed before the test set opened, real scorer) and its tol_P.
    Missing or unusable: before 11/04 11:00 refused (2: 'tol_P 【待演練】', rehearse first); from then on B is not used
    this season (exit 3 for B, recorded). required=False: None instead (records only)."""
    p = p1_paths(ctx)["rehearsal"]
    probs = []
    rec = None
    if not os.path.exists(p):
        probs.append(f"tol_P 【待演練】: {t6.shown(p)} is missing")
    else:
        try:
            rec = t6.read_json(p)
        except Exception as e:  # noqa: BLE001
            rec = {"unreadable": f"{type(e).__name__}: {e}"}
            probs.append(f"unreadable: {rec['unreadable']}")
        if rec.get("format") != FORMATS["rehearsal"] or rec.get("mode") != ctx["mode"]:
            probs.append(f"format {rec.get('format')!r} mode {rec.get('mode')!r}")
        if rec.get("passed") is not True:
            probs.append("the rehearsal did not pass (failed flags / move counts differ): tol_P is not set")
        m = rec.get("m")
        if not finite(m) or not finite(rec.get("tol_P")) or abs(rec["tol_P"] - tol_p(m)) > 1e-15:
            probs.append(f"m {m!r} / tol_P {rec.get('tol_P')!r} do not follow tol_P = max(0.05, 2.5 m)")
        bound = rec.get("bound") or {}
        if bound.get("frozen_sha256_lf") != b.get("frozen_sha256_lf") or \
                bound.get("val_seam_sha256") != b.get("val_seam_sha256"):
            probs.append("made under another frozen_config.json / val_seam_feats.npz")
    commit = None
    if rec is not None and ctx["real"]:
        if rec.get("scorer") != "real":
            probs.append(f"scorer {rec.get('scorer')!r} (the real-mode tol_P comes from the 4090)")
        if not t6.committed_clean(p):
            probs.append("not committed and clean")
        else:
            commit = t6.git_commit_time(p)
            if commit is None or t6.parse_iso(commit, "commit time") >= TEST_OPEN:
                probs.append(f"committed at {commit}, not before the test set opened ({TEST_OPEN.isoformat()})")
    if probs:
        if not required:
            return None
        if ctx["now"] >= TEST_OPEN:
            b_not_used(T, ctx, cmd, "no_rehearsal", f"沒有可用的演練紀錄（{probs}），測試集已開放，tol_P 不能再量", pool)
        T.refuse(f"{t6.shown(p)}: {probs} (PREREG_10 rev 4: tol_P comes from a rehearsal record committed before "
                 f"{TEST_OPEN.isoformat()}; run t10_player.py p1-rehearse first, until then p1 refuses)")
    return {"path": t6.shown(p), "sha256_lf": t6.sha_lf(p), "m": rec["m"], "tol_P": rec["tol_P"],
            "committed_at": commit}


# ================================================================ the games of a set
def val_games(T, ctx, b, n=SEAM_GAMES):
    """The first n validation games in the order of extract --set val (candidate games in file order, ...), read inside
    the p06 guard's allow_local (the two game files, SHA-256 = 06's inputs); whole key list SHA = val_report's."""
    rep06 = t6.read_json(os.path.join(P06_PLAYER, "apply_report.json"))
    with st.allow_local():
        in_sha = {"candidates_csv": t6.sha_raw(t6.VAL_C), "questions_csv": t6.sha_raw(t6.VAL_Q)}
        cand_rows = pf.read_candidates(t6.VAL_C)
        q_rows = pf.read_questions(t6.VAL_Q)
    if any(in_sha[k] != (rep06.get("inputs") or {}).get(k) for k in in_sha):
        T.refuse("the validation game files differ from the ones 06 applied to")
    keys, cols = PA.val_keys(cand_rows, q_rows)
    if PA.keys_sha(keys) != b["val_keys_sha256"] or len(keys) != b["val_games"]:
        T.refuse("the validation game order differs from results/prereg10/feat/val_report.json")
    sgfs = [r[3] for r in cand_rows] + [x for _q, g in q_rows for _s, _c, x in g]
    games = [{"sgf_content": sgfs[j], "color": cols[j], "ref": j} for j in range(min(n, len(keys)))]
    return games, {"keys_sha256": PA.keys_sha(keys[:len(games)]), "games": len(games), "inputs_sha256": in_sha,
                   "val_keys_sha256": b["val_keys_sha256"]}


def pool_games(T, ctx, pool, pre):
    """A test pool's games: candidate games (file order), then question games (question / slot order); the files are
    the ones test02 used (apply_report.json inputs)."""
    q, c, _pd = t6.pool_files(ctx, pool)
    shas = {"questions_csv": t6.sha_raw(q), "candidates_csv": t6.sha_raw(c)}
    t6.t02_binding(ctx, pool, pre["t02"], shas)
    cand_rows = pf.read_candidates(c)
    q_rows = pf.read_questions(q)
    keys, cols = PA.val_keys(cand_rows, q_rows)
    if any(x not in ("B", "W") for x in cols):
        T.refuse(f"{pool}: a game colour is not B or W")
    sgfs = [r[3] for r in cand_rows] + [x for _q, g in q_rows for _s, _c, x in g]
    games = [{"sgf_content": x, "color": cc, "ref": j} for j, (x, cc) in enumerate(zip(sgfs, cols))]
    return games, {"keys_sha256": PA.keys_sha(keys), "games": len(games), "inputs_sha256": shas,
                   "black_share": round(sum(1 for x in cols if x == "B") / max(1, len(cols)), 6)}


def pool_keys(ctx, pool):
    q, c, _pd = t6.pool_files(ctx, pool)
    with st.allow_local():
        keys, _cols = PA.val_keys(pf.read_candidates(c), pf.read_questions(q))
    return keys


# ================================================================ scorers
class FakeProbeScorer:
    """Deterministic CPU stand-in for p1_lib.ProbeScorer (dry runs only; the formula of tools/p10_p1.py's)."""

    def position_features(self, feats):
        T = len(feats["move"])
        out = np.zeros((T, 770), dtype=np.float32)
        if T:
            nleg = feats["legal"].sum(1).astype(np.float64)
            mv = feats["move"].astype(np.float64)
            j = np.arange(384, dtype=np.float64)
            out[:, :384] = np.sin((mv[:, None] + 1.0) * (j + 1.0) * 1e-3)
            out[:, 384:768] = np.cos(nleg[:, None] * (j + 1.0) * 1e-4)
            out[:, 768] = -np.log(nleg)
            out[:, 769] = 0.9 * np.log(nleg)
        return out

    def close(self):
        pass


def synthetic_means(sgf, target):
    """--synthetic-p1 (dry runs only): a deterministic 770-dim summary of the SGF's moves per side (point frequencies
    of all moves 0-360, of the first 20 moves 361-721, log(1 + moves) 768). -> (means [2, 770], counts [2], n)."""
    mv = pm.parse_moves(sgf, 361)
    means = np.zeros((2, 770), np.float32)
    counts = np.zeros(2, np.int32)
    for side, who in ((0, target), (1, "W" if target == "B" else "B")):
        pts = [p for c, p in mv if c == who]
        counts[side] = len(pts)
        if not pts:
            continue
        v = np.zeros(770, np.float64)
        for k, p in enumerate(pts):
            x, y = pm.to_xy(p)
            v[x * 19 + y] += 1.0
            if k < 20:
                v[361 + x * 19 + y] += 1.0
        v[:361] /= len(pts)
        v[361:722] /= min(20, len(pts))
        v[768] = math.log1p(len(pts))
        v[769] = len(pts) / max(1, len(mv))
        means[side] = v
    return means, counts, len(mv)


def synthetic_chunk(games):
    G = len(games)
    means = np.zeros((G, 2, 770), np.float32)
    counts = np.zeros((G, 2), np.int32)
    n_moves = np.zeros(G, np.int16)
    failed = np.zeros(G, bool)
    for i, g in enumerate(games):
        means[i], counts[i], n_moves[i] = synthetic_means(g["sgf_content"], g["color"])
    z = lambda dt_: np.zeros(0, dt_)  # noqa: E731
    return {"means": means, "counts": counts, "n_moves": n_moves, "failed": failed, "errors": "[]",
            "pos_game": z(np.int32), "pos_side": z(np.int8), "pos_turn": z(np.int16), "pos_logp": z(np.float32),
            "pos_entropy": z(np.float32)}, 0.0


def scorer_kind(T, ctx, a):
    kinds = [k for k, on in (("fake", getattr(a, "fake_scorer", False)), ("synthetic", getattr(a, "synthetic_p1",
                                                                                                False))) if on]
    if len(kinds) > 1:
        T.refuse("--fake-scorer and --synthetic-p1 exclude each other")
    if kinds and ctx["real"]:
        T.refuse(f"--{'fake-scorer' if kinds[0] == 'fake' else 'synthetic-p1'} is for --dry-run only")
    return kinds[0] if kinds else "real"


def gpu_check(T, ctx, a, kind):
    """nvidia-smi before torch touches CUDA: no other compute process (PREREG_10 4.2)."""
    if kind != "real":
        return {"checked": False, "scorer": kind}
    from p1_lib import H  # noqa: F401  (hs_lib's data/local guard: the game files are read before this)
    procs, _raw = H.other_compute_processes()
    procs = [p for p in procs if p.get("pid") != os.getpid()]
    if procs:
        T.refuse(f"nvidia-smi lists another compute process: {procs}; wait, or let it yield (PREREG_10 4.3)")
    return {"checked": True, "other_compute_processes": procs}


def open_scorer(T, b, kind):
    """-> (scorer or None, environment record, chunk constants)."""
    if kind == "synthetic":
        return None, {"scorer": "synthetic", "torch": "none", "cudnn": "none", "gpu": "none"}, dict(SYN)
    import p1_lib as P
    consts = {"meta_sha256": P.neutral_table_sha(), "code_sha256": json.dumps(P.code_shas(), sort_keys=True)}
    if kind == "fake":
        consts.update(ckpt_sha256="fake", ckpt_bytes=0)
        return FakeProbeScorer(), {"scorer": "fake", "torch": "fake", "cudnn": "fake", "gpu": "cpu"}, consts
    if consts["meta_sha256"] not in (b.get("neutral_metadata_sha256") or []):
        T.refuse("the neutral metadata table differs from frozen_config.json")
    if b.get("chunk_code_sha256") and consts["code_sha256"] not in b["chunk_code_sha256"]:
        T.refuse("the P1 code differs from the code of the frozen mock chunks (frozen_config.json)")
    import torch
    if not torch.cuda.is_available():
        T.refuse("CUDA is not available (.venv-gpu on the RTX 4090)")
    gpu = torch.cuda.get_device_name(0)
    if gpu != GPU_NAME:
        T.refuse(f"GPU {gpu!r} is not {GPU_NAME} (PREREG_10 4.2: P1 runs on the local 4090)")
    ck = P.H.check_checkpoint(P.H.CKPT_PATH, full_hash=True)
    if ck.get("sha256") != (b.get("checkpoint") or {}).get("sha256"):
        T.refuse("the checkpoint differs from frozen_config.json (use data/katago_backup/, else stop)")
    consts.update(ckpt_sha256=ck["sha256"], ckpt_bytes=ck["bytes"])
    torch.backends.cudnn.benchmark = RUN["cudnn_benchmark"]
    model = P.load_model(device=RUN["device"])
    sc = P.ProbeScorer(model, device=RUN["device"], precision=RUN["precision"], batch=RUN["batch"],
                       channels_last=RUN["channels_last"])
    env = {"scorer": "real", "torch": torch.__version__, "cudnn": str(torch.backends.cudnn.version()), "gpu": gpu,
           "cuda": torch.version.cuda}
    return sc, env, consts


# ================================================================ extraction (resumable chunks)
def chunk_ok(T, path, refs, set_name, keys_sha, mode, env, consts):
    """An existing chunk is reused only when it is this run's (else refused: never overwritten)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            got = {k: str(z[k]) for k in ("format", "set", "keys_sha256", "mode", "scorer", "torch", "cudnn",
                                          "gpu_name", "meta_sha256", "code_sha256", "ckpt_sha256")}
            same = np.array_equal(np.asarray(z["ref"], np.int64), refs)
    except Exception as e:  # noqa: BLE001
        T.refuse(f"{t6.shown(path)}: unreadable chunk ({e}); move it away and run again")
    want = {"format": FORMATS["chunk"], "set": set_name, "keys_sha256": keys_sha, "mode": mode,
            "scorer": env["scorer"], "torch": str(env["torch"]), "cudnn": str(env["cudnn"]),
            "gpu_name": str(env["gpu"]), "meta_sha256": consts["meta_sha256"], "code_sha256": consts["code_sha256"],
            "ckpt_sha256": consts["ckpt_sha256"]}
    diff = [k for k in want if got[k] != want[k]] + ([] if same else ["ref"])
    if diff:
        T.refuse(f"{t6.shown(path)}: an existing chunk differs from this run ({diff}); never overwritten")


def extract(T, ctx, a, games, set_name, keys_sha, chunk_dir, kind, b):
    """-> (chunk paths, env, stopped reason or None). Chunks of CHUNK games; STOP in chunk_dir ends the run before the
    next chunk (resume: delete STOP and run the same line)."""
    import multiprocessing as mp
    os.makedirs(T.guard_write(ctx, chunk_dir), exist_ok=True)
    refs = np.array([g["ref"] for g in games], np.int64)
    plan = [(ci, s, min(len(games), s + CHUNK)) for ci, s in enumerate(range(0, len(games), CHUNK))]
    path = lambda ci: os.path.join(chunk_dir, "chunk_%05d.npz" % ci)  # noqa: E731
    present = [path(ci) for ci, _s, _e in plan if os.path.exists(path(ci))]
    extra = sorted(set(F.chunk_files(chunk_dir)) - {path(ci) for ci, _s, _e in plan})
    if extra:
        T.refuse(f"{t6.shown(chunk_dir)} holds chunks that are not this run's: {[os.path.basename(x) for x in extra[:5]]}")
    todo = [(ci, s, e) for ci, s, e in plan if not os.path.exists(path(ci))]
    if todo and os.path.exists(os.path.join(chunk_dir, "STOP")):
        print(json.dumps({"time": now_iso(), "msg": "STOP file present: no chunk started", "chunks_left": len(todo)}),
              flush=True)
        return [path(ci) for ci, _s, _e in plan], None, "STOP file"
    sc, env, consts = open_scorer(T, b, kind) if (todo or present) else (None, None, None)
    for ci, s, e in plan:
        if os.path.exists(path(ci)):
            chunk_ok(T, path(ci), refs[s:e], set_name, keys_sha, ctx["mode"], env, consts)
    print(json.dumps({"time": now_iso(), "msg": "p1 start", "set": set_name, "mode": ctx["mode"], "scorer": kind,
                      "games": len(games), "chunks": len(plan), "todo": len(todo)}), flush=True)
    stopped = None
    if todo:
        PX = None
        if kind != "synthetic":
            import p1_extract as PX
        t_run = time.time()
        pool = mp.get_context("spawn").Pool(a.workers) if PX is not None else None
        try:
            for k, (ci, s, e) in enumerate(todo):
                if os.path.exists(os.path.join(chunk_dir, "STOP")):
                    stopped = "STOP file"
                    print(json.dumps({"time": now_iso(), "msg": "STOP file: stopping before the next chunk",
                                      "chunks_left": len(todo) - k}), flush=True)
                    break
                t0 = time.time()
                if PX is None:
                    res, t_gpu = synthetic_chunk(games[s:e])
                else:
                    res, t_gpu = PX.run_chunk(pool, sc, games[s:e], GROUP_POSITIONS)
                secs = time.time() - t0
                T.guard_write(ctx, path(ci) + ".tmp.npz")
                save_chunk(path(ci), set_name, refs[s:e], keys_sha, ctx["mode"], env, consts, res, secs, t_gpu)
                el = time.time() - t_run
                print(json.dumps({"time": now_iso(), "msg": "chunk done", "set": set_name, "chunk": ci,
                                  "games": e - s, "failed": int(res["failed"].sum()), "seconds": round(secs, 1),
                                  "progress": "%d/%d" % (k + 1, len(todo)),
                                  "eta_min": round(el / (k + 1) * (len(todo) - k - 1) / 60, 1)}), flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            if sc is not None:
                sc.close()
    return [path(ci) for ci, _s, _e in plan], env, stopped


def save_chunk(path, set_name, refs, keys_sha, mode, env, consts, res, secs, t_gpu):
    """One chunk, atomically (tmp + os.replace): the p1_extract chunk fields plus this run's identity."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, format=np.array(FORMATS["chunk"]), set=np.array(set_name), sides=np.array(["target", "opponent"]),
             ref=np.asarray(refs, np.int64), precision=np.array(RUN["precision"] if env["scorer"] == "real"
                                                                 else env["scorer"]),
             meta_sha256=np.array(consts["meta_sha256"]), ckpt_bytes=np.array(consts["ckpt_bytes"]),
             ckpt_sha256=np.array(consts["ckpt_sha256"]), code_sha256=np.array(consts["code_sha256"]),
             keys_sha256=np.array(keys_sha), mode=np.array(mode), scorer=np.array(env["scorer"]),
             torch=np.array(str(env["torch"])), cudnn=np.array(str(env["cudnn"])), gpu_name=np.array(str(env["gpu"])),
             created_utc=np.array(dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")),
             seconds=np.array(secs), gpu_seconds=np.array(t_gpu), **res)
    os.replace(tmp, path)


def chunk_records(paths):
    out = []
    for p in paths:
        r = F.chunk_record(p)
        with np.load(p, allow_pickle=False) as z:
            r.update(mode=str(z["mode"]), scorer=str(z["scorer"]), torch=str(z["torch"]), cudnn=str(z["cudnn"]),
                     gpu_name=str(z["gpu_name"]), keys_sha256=str(z["keys_sha256"]), seconds=float(z["seconds"]))
        out.append(r)
    return out


def stopped_exit(T, chunk_dir, why):
    print(f"stopped ({why}); {t6.shown(chunk_dir)} holds the finished chunks. To continue: delete "
          f"{t6.shown(os.path.join(chunk_dir, 'STOP'))}, then run the same line again (resume, not rerun)")
    print(f"STOPPED (exit {EXIT_STOPPED}): p1 is not complete; nothing decided", file=sys.stderr)
    return EXIT_STOPPED


# ================================================================ the GPU seam (and the rehearsal's m)
def seam_compare(side0, counts0, failed, ref, model, tol):
    """The 400 re-extracted validation games against val_seam_feats.npz: failed flags and target-move counts equal;
    the per-game max |difference| of the frozen-LDA projection over the usable games. -> record."""
    n = SEAM_GAMES
    k = min(n, len(failed), len(ref["failed"]))
    f_eq = np.array_equal(np.asarray(failed[:k], bool), np.asarray(ref["failed"][:k], bool))
    c_eq = np.array_equal(np.asarray(counts0[:k], np.int64), np.asarray(ref["counts0"][:k], np.int64))
    counts_equal = bool(f_eq and c_eq and k == n)
    ok = ~np.asarray(ref["failed"][:k], bool) & (np.asarray(ref["counts0"][:k]) > 0)
    per_game = None
    if counts_equal and ok.any():
        rows = np.flatnonzero(ok)
        za = F.project_p1(np.asarray(side0[:k], np.float32), model, rows)
        zb = F.project_p1(np.asarray(ref["side0"][:k], np.float32), model, rows)
        per_game = np.max(np.abs(za - zb), axis=1)
        m = float(per_game.max())
    else:
        m = float("inf")
    passed = bool(counts_equal and math.isfinite(m) and (tol is None or m <= tol))
    return {"games": int(k), "usable_games": int(ok.sum()), "failed_equal": bool(f_eq), "counts_equal": counts_equal,
            "max_abs_projected_diff": m if math.isfinite(m) else None,
            "games_above_tol": (int((per_game > tol).sum()) if per_game is not None and tol is not None else None),
            "bitwise_equal_side0": bool(counts_equal and np.array_equal(np.asarray(side0[:k], np.float32),
                                                                        np.asarray(ref["side0"][:k], np.float32))),
            "tol_P": tol, "passed": passed,
            "definition": "per game: max over the 64 dims of |LDA projection (frozen p1_lda.npz) of the re-extracted "
                          "side-0 mean - that of val_seam_feats.npz|; max over the usable games (PREREG_10 4.2)"}


def load_ref(b):
    with np.load(b["paths"]["val_seam"], allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ("side0", "counts0", "failed")}


def side0_of(paths, n, set_name):
    X, _usable, failed, have, cnt = F.load_side0(paths, n, set_name)
    return X, cnt, failed, have


# ================================================================ commands
def cmd_rehearse(T, ctx, a, b):
    """PREREG_10 rev 4: m before any test data -> results/test10/p1/p1_rehearsal.json (commit it)."""
    t0 = time.time()
    kind = scorer_kind(T, ctx, a)
    if ctx["now"] >= TEST_OPEN:
        T.refuse(f"the rehearsal comes before the test set opens ({TEST_OPEN.isoformat()}); now {ctx['now'].isoformat()}"
                 f": tol_P cannot be measured any more (B is not used this season without it)")
    if ctx["real"]:
        present = [n for pool in t6.TEST_POOLS for n in t6.TEST_FILES[pool] if os.path.exists(os.path.join(t6.REAL_T,
                                                                                                          n))]
        if present:
            T.refuse(f"official test files already in data/test ({present}): the rehearsal comes before test data")
    pp = p1_paths(ctx)
    if os.path.exists(pp["rehearsal"]) and not (a.rerun and (a.reason or "").strip()):
        T.refuse(f"{t6.shown(pp['rehearsal'])} exists; --rerun --reason TEXT (a committed record is never overwritten)")
    if os.path.exists(pp["rehearsal"]) and ctx["real"] and t6.committed_clean(pp["rehearsal"]):
        T.refuse("the committed rehearsal record is never overwritten")
    games, krec = val_games(T, ctx, b)
    gpu = gpu_check(T, ctx, a, kind)
    paths, env, stopped = extract(T, ctx, a, games, "t10_rehearsal", krec["keys_sha256"], pp["rehearsal_dir"], kind, b)
    if stopped:
        return stopped_exit(T, pp["rehearsal_dir"], stopped)
    X, cnt, failed, have = side0_of(paths, len(games), "t10_rehearsal")
    rec = seam_compare(X, cnt, failed, load_ref(b), load_model(b), None)
    m = rec["max_abs_projected_diff"]
    out = {"format": FORMATS["rehearsal"], "mode": ctx["mode"], "scorer": kind,
           "what": "PREREG_10 rev 4: the 4090 rehearsal of the GPU seam before test data; m = max projected "
                   "difference of the 400 re-extracted validation games, tol_P = max(0.05, 2.5 m)",
           "passed": rec["counts_equal"] and m is not None, "m": m,
           "tol_P": tol_p(m) if m is not None else None, "compare": rec, "keys": krec, "environment": env,
           "gpu_check": gpu, "chunks": chunk_records(paths),
           "bound": {"frozen_sha256_lf": b["frozen_sha256_lf"], "val_seam_sha256": b["val_seam_sha256"],
                     "lda_sha256": b["lda_sha256"]},
           "test_open": TEST_OPEN.isoformat(), "now": ctx["now"].isoformat(), "rerun": ({"reason": a.reason}
                                                                                    if a.rerun else None),
           "tools": T.tool_record(), "created": now_iso(), "seconds": round(time.time() - t0, 1)}
    T.write_json(ctx, pp["rehearsal"], out)
    print(f"p1-rehearse: counts equal {rec['counts_equal']}, m {m}, tol_P {out['tol_P']} -> {t6.shown(pp['rehearsal'])}"
          f" (commit it before 11/04 11:00)")
    if not out["passed"]:
        T.refuse("the rehearsal's failed flags / move counts differ from val_seam_feats.npz: tol_P is not set; stop "
                 "and find out why (PREREG_10 rev 4)")
    return 0


def gpu_seam_state(T, ctx, required=True):
    p = p1_paths(ctx)["gpu_seam"]
    if not os.path.exists(p):
        if required:
            T.refuse("run p1 --pool seam_val first (the GPU seam comes before the test pools)")
        return None
    rec = t6.read_json(p)
    if rec.get("format") != FORMATS["gpu_seam"] or rec.get("mode") != ctx["mode"]:
        T.refuse(f"{t6.shown(p)}: format {rec.get('format')!r} mode {rec.get('mode')!r}")
    T.require_committed(ctx, [p], "gpu_seam.json")
    return rec


def cmd_p1(T, ctx, a, pre, b):
    """p1 --pool seam_val|public|private (B adopted; the caller has checked the 10 prerequisites)."""
    t0 = time.time()
    pool = a.pool
    kind = scorer_kind(T, ctx, a)
    if a.max_games and ctx["real"]:
        T.refuse("--max-games is for drills only")
    k5 = k5_b_state(T, ctx, b)
    if k5["checked"] and not k5["ok"]:
        reason, msg = k5_reason(k5)
        b_not_used(T, ctx, f"p1 --pool {pool}", reason, f"{msg}（{k5}）", pool)
    if ctx["now"] > K6B_10:
        b_not_used(T, ctx, f"p1 --pool {pool}", "k6b_10",
                   f"K6B-10 已過（{K6B_10.isoformat()}；現在 {ctx['now'].isoformat()}）", pool)
    tol = tol_state(T, ctx, b, cmd=f"p1 --pool {pool}", pool=pool)
    pp = p1_paths(ctx, pool)
    seam = gpu_seam_state(T, ctx, required=pool != "seam_val")
    if pool != "seam_val" and seam.get("passed") is not True:
        b_not_used(T, ctx, f"p1 --pool {pool}", "gpu_seam_failed", "B 的 GPU 接縫沒過（gpu_seam.json）", pool)
    outs = [pp["report"]] + ([pp["gpu_seam"]] if pool == "seam_val" else [])
    have = [p for p in outs if os.path.exists(p)]
    rerun = None
    if have:
        if not (a.rerun and (a.reason or "").strip()):
            T.refuse(f"{[t6.shown(p) for p in have]} exist; --rerun --reason TEXT (resuming needs no flag)")
        if ctx["real"] and any(t6.committed_clean(p) for p in have):
            T.refuse("the committed P1 reports are never overwritten")
        rerun = {"reason": a.reason, "previous_sha256": {t6.shown(p): t6.sha_raw(p) for p in have}}
    if pool == "seam_val":
        games, krec = val_games(T, ctx, b)
    else:
        games, krec = pool_games(T, ctx, pool, pre)
    if a.max_games:
        games = games[:a.max_games]
    gpu = gpu_check(T, ctx, a, kind)
    set_name = f"t10_{pool}"
    paths, env, stopped = extract(T, ctx, a, games, set_name, krec["keys_sha256"] if not a.max_games else
                                  PA.keys_sha([str(g["ref"]) for g in games]), pp["chunks"], kind, b)
    if stopped:
        return stopped_exit(T, pp["chunks"], stopped)
    if a.max_games:
        print(f"p1 {pool}: drill of {len(games)} games done (--max-games: no report, nothing decided)")
        return 0
    if pool != "seam_val":
        senv = seam.get("environment") or {}
        envd = {k: (senv.get(k), env.get(k)) for k in ("scorer", "torch", "cudnn", "gpu") if str(senv.get(k)) !=
                str(env.get(k))}
        if envd:
            T.refuse(f"this environment differs from the GPU seam's {envd}: run the seam again in this environment")
    recs = chunk_records(paths)
    X, cnt, failed, have_ = side0_of(paths, len(games), set_name)
    if not have_.all():
        T.refuse("games without a chunk")
    rep = {"format": FORMATS["p1_report"], "mode": ctx["mode"], "pool": pool, "scorer": kind,
           "what": "PREREG_10 4.2 p1: KataGo HumanSL neutral-probe P1 (side 0 = the row's colour) of the pool's games",
           "games": len(games), "failed": int(failed.sum()), "failed_fraction": float(failed.sum()) / len(games),
           "no_target_moves": int(((cnt == 0) & ~failed).sum()), "keys": krec, "chunk_dir": t6.shown(pp["chunks"]),
           "chunks": recs, "environment": env, "gpu_check": gpu, "tol": tol, "k5_10_b": k5,
           "b": {k: b[k] for k in ("frozen_sha256_lf", "lda_sha256", "val_seam_sha256", "w_star")},
           "gpu_seam_sha256_lf": t6.sha_lf(pp["gpu_seam"]) if pool != "seam_val" else None,
           "t1": pre["t1"], "rerun": rerun, "tools": T.tool_record(), "created": now_iso(),
           "seconds": round(time.time() - t0, 1),
           "positions_per_s_record_only": None}
    if pool == "seam_val":
        rec = seam_compare(X, cnt, failed, load_ref(b), load_model(b), tol["tol_P"])
        gs = dict(rec, format=FORMATS["gpu_seam"], mode=ctx["mode"], scorer=kind, environment=env,
                  tol=tol, report_sha256_lf=None, created=now_iso(),
                  bound={"val_seam_sha256": b["val_seam_sha256"], "lda_sha256": b["lda_sha256"],
                         "frozen_sha256_lf": b["frozen_sha256_lf"]})
        T.write_json(ctx, pp["report"], rep)
        gs["report_sha256_lf"] = t6.sha_lf(pp["report"])
        T.write_json(ctx, pp["gpu_seam"], gs)
        print(f"GPU seam: counts equal {rec['counts_equal']}, max projected diff {rec['max_abs_projected_diff']} "
              f"(tol_P {tol['tol_P']}) -> {'PASSED' if rec['passed'] else 'FAILED'}")
        if not rec["passed"]:
            b_not_used(T, ctx, "p1 --pool seam_val", "gpu_seam_failed",
                       f"B 的 GPU 接縫沒過（max {rec['max_abs_projected_diff']} > tol_P {tol['tol_P']} 或盤的失敗／手數不同）",
                       pool)
        return 0
    T.write_json(ctx, pp["report"], rep)
    print(f"p1 {pool}: {len(games)} games, failed {rep['failed']} ({rep['failed_fraction']:.4%}), no target moves "
          f"{rep['no_target_moves']} -> {t6.shown(pp['report'])} (commit it before K6B-10 {K6B_10.isoformat()})")
    return 0


# ================================================================ apply: the B term on one pool
def p1_pool_state(T, ctx, pool, pre, b):
    """-> (usable: bool, record, chunk paths). Not usable for a registered reason (GPU seam failed, K6B-10 passed
    without the P1 committed, K5-10 for B not met) -> (False, record, None). P1 missing before K6B-10: refused (2)."""
    rec = {"pool": pool}
    k5 = k5_b_state(T, ctx, b)
    rec["k5_10_b"] = k5
    if k5["checked"] and not k5["ok"]:
        reason, msg = k5_reason(k5)
        return False, dict(rec, reason=reason, rule=msg), None
    seam = gpu_seam_state(T, ctx, required=False)
    pp = p1_paths(ctx, pool)
    if seam is not None and seam.get("passed") is not True:
        return False, dict(rec, reason="gpu_seam_failed", rule="the B GPU seam failed: B not used (PREREG_10 4.2)"), None
    if seam is None or not os.path.exists(pp["report"]):
        if ctx["now"] > K6B_10:
            return False, dict(rec, reason="k6b_10", rule="K6B-10 passed without this pool's P1 / the GPU seam"), None
        # rev 4 section 14: never block A. Before K6B-10 the pool keeps A now; apply --rerun after p1 adds B
        print(f"note: {pool}: B is adopted but {'the GPU seam' if seam is None else 'this pool’s P1'} is not done: "
              f"this pool keeps A for now. Run t10_player.py p1 --pool {'seam_val' if seam is None else pool} (4090), "
              f"then apply --pool {pool} --rerun --reason \"B P1 done\" before K6B-10 ({K6B_10.isoformat()})")
        return False, dict(rec, reason="p1_not_done", rule="the pool's P1 / the GPU seam not done yet (apply again "
                                                             "after p1, before K6B-10)"), None
    rep = t6.read_json(pp["report"])
    T.require_committed(ctx, [pp["report"]], f"{pool}_p1_report.json")
    if rep.get("format") != FORMATS["p1_report"] or rep.get("mode") != ctx["mode"] or rep.get("pool") != pool:
        T.refuse(f"{t6.shown(pp['report'])}: format {rep.get('format')!r} mode {rep.get('mode')!r}")
    if rep.get("gpu_seam_sha256_lf") != t6.sha_lf(pp["gpu_seam"]):
        T.refuse(f"{pool}: the P1 report was made under another gpu_seam.json")
    if (rep.get("b") or {}).get("lda_sha256") != b["lda_sha256"]:
        T.refuse(f"{pool}: the P1 report was made under another frozen LDA")
    if ctx["real"]:
        times = {n: t6.git_commit_time(p) for n, p in (("p1_report", pp["report"]), ("gpu_seam", pp["gpu_seam"]))}
        late = {n: t for n, t in times.items() if t is None or t6.parse_iso(t, "commit time") > K6B_10}
        rec["committed_at"] = times
        if late:
            return False, dict(rec, reason="k6b_10", rule=f"committed after K6B-10: {late}"), None
    files = [os.path.join(pp["chunks"], c["file"]) for c in rep.get("chunks") or []]
    bad = [os.path.basename(f) for f, c in zip(files, rep["chunks"]) if not os.path.exists(f)
           or t6.sha_raw(f) != c["sha256"]]
    if bad:
        T.refuse(f"{pool}: P1 chunks changed or missing since the p1 report: {bad[:5]}")
    keys = pool_keys(ctx, pool)
    if PA.keys_sha(keys) != (rep.get("keys") or {}).get("keys_sha256"):
        T.refuse(f"{pool}: the pool's games differ from the P1 report (test files changed?)")
    rec.update(p1_report_sha256_lf=t6.sha_lf(pp["report"]), gpu_seam_sha256_lf=t6.sha_lf(pp["gpu_seam"]),
               failed=rep.get("failed"), games=rep.get("games"), failed_fraction=rep.get("failed_fraction"))
    return True, rec, (files, keys)


def pool_term(T, ctx, pool, pre, b, d, ms, res):
    """PREREG_10 2.1 / 4.2 on one pool that used 06: 10B rows, the pool gate on the pool's sr06 pseudo pools, the
    failed share. -> record with 'applied' and 'rows'."""
    ok, rec, extra = p1_pool_state(T, ctx, pool, pre, b)
    if not ok:
        return dict(rec, applied=False, rows=None)
    files, keys = extra
    if [str(x) for x in ms.gid.tolist()] != keys:
        T.refuse(f"{pool}: the 06 store's game order differs from the P1 extraction order")
    X, usable, failed, have, cnt = F.load_side0(files, len(keys), f"t10_{pool}")
    if not have.all():
        T.refuse(f"{pool}: games without a P1 chunk")
    model = load_model(b)
    Z = np.zeros((len(keys), st.LDA_DIM))
    rows = np.flatnonzero(usable)
    Z[rows] = F.project_p1(X, model, rows)
    kst, w = pre["fz"]["k_star"], b["w_star"]
    zT = F.z_p1(Z, res["cgroups"], res["qgroups"], usable)
    top = st.top5(F.fuse_10b(res["F"], zT, w, kst), d.cand_ids)
    c_by_pid = {p: d.c_by_player[p] for p in d.cand_ids}
    ks = [len(d.q_by_id[q]) for q in d.qids]
    t0 = time.time()
    srm, sr10 = PA.sr06_with_p1(t6.p6, ms, d.cand_ids, c_by_pid, ks, res["earr"], pre["models"], res["combos"], kst,
                                Z, usable, w)
    sel = res["selected"]
    if abs(srm[sel] - res["sr06"][sel]) > 1e-12:
        T.refuse(f"{pool}: sr06 of {sel} recomputed with P1 ({srm[sel]}) differs from the 06 computation "
                 f"({res['sr06'][sel]}): local drift, stop and check")
    gate = sr10[sel] - srm[sel]
    frac = float(failed.sum()) / len(keys)
    applied = bool(gate >= GATE_B_MIN and frac <= FAIL_MAX)
    why = [] if applied else (["pool gate sr(10B) - sr(selected) < 0"] if gate < GATE_B_MIN else []) + \
        ([f"P1 failed games {frac:.4%} > 0.5 %"] if frac > FAIL_MAX else [])
    rows_out = [[q] + t for q, t in zip(d.qids, top)]
    rec.update(applied=applied, rows=rows_out if applied else None, w_star=w, k_star=kst, selected=sel,
               pool_gate={"sr_selected": srm[sel], "sr_10B": sr10[sel], "sr_10B_minus_selected": gate,
                          "min": GATE_B_MIN, "passed": bool(gate >= GATE_B_MIN)},
               failed_fraction=frac, fail_max=FAIL_MAX, usable_games=int(usable.sum()),
               no_target_moves=int(((cnt == 0) & ~failed).sum()),
               questions_without_usable_p1=int(sum(1 for g in res["qgroups"] if not any(usable[j] for j in g))),
               top1_differences_vs_selected=int(sum(1 for r, o in zip(rows_out, res["off"]) if r[1] != o[0])),
               reason=None if applied else "; ".join(why), sr06_10B_seconds=round(time.time() - t0, 1),
               rule="10B = CSLS_k*(F_sel + w* z(T_P1)) when sr(10B) - sr(selected) >= 0 and P1 failed <= 0.5 % "
                    "(PREREG_10 2.1, 4.2)")
    return rec


# ================================================================ seam: 10B on seam_val against the committed validation 10B
def seam_10b(T, ctx, b, d, ms, res, dumps_bitwise):
    """PREREG_10 4.2: 10B recomputed on seam_val from the committed validation P1 chunks and the frozen LDA (rows of
    the 06 seam computation, R02 = Q04) against results/prereg10/player/player_val_top5.csv."""
    P = b["paths"]
    vr = t6.read_json(P["val_report"])
    keys = [str(x) for x in ms.gid.tolist()]
    if PA.keys_sha(keys) != vr.get("keys_sha256") or len(keys) != vr.get("games"):
        T.refuse("seam_val: the 06 store's game order differs from the validation P1 extraction (val_report.json)")
    files = F.chunk_files(P["val_dir"])
    want = {r["file"]: r["sha256"] for r in vr.get("chunks") or []}
    if sorted(os.path.basename(f) for f in files) != sorted(want):
        T.refuse("seam_val: the validation P1 chunk files differ from val_report.json")
    bad = [os.path.basename(f) for f in files if t6.sha_raw(f) != want[os.path.basename(f)]]
    if bad:
        T.refuse(f"seam_val: validation P1 chunks changed since val_report.json: {bad[:5]}")
    X, usable, failed, have, _cnt = F.load_side0(files, len(keys), vr.get("set") or VAL_SET)
    if not have.all():
        T.refuse("seam_val: validation games without a P1 chunk")
    Z = np.zeros((len(keys), st.LDA_DIM))
    rows = np.flatnonzero(usable)
    Z[rows] = F.project_p1(X, load_model(b), rows)
    zT = F.z_p1(Z, res["cgroups"], res["qgroups"], usable)
    top = st.top5(F.fuse_10b(res["F"], zT, b["w_star"], res["k_star"]), d.cand_ids)
    mine = t6.rows_bytes(d.qids, top)
    ref_b = t6.bytes_lf(P["val_top5"])
    ref_rows = t6.read_top5(P["val_top5"])
    same_rows = sum(1 for (q, t), r in zip(zip(d.qids, top), ref_rows) if [q] + t == r)
    rec = {"bitwise_required": bool(dumps_bitwise), "byte_identical": mine == ref_b,
           "rows_identical": same_rows, "rows": len(d.qids), "min_identical": SEAM_MIN_IDENTICAL,
           "reference": t6.shown(P["val_top5"]), "reference_sha256_lf": t6.sha_lf(P["val_top5"]),
           "sha256": hashlib.sha256(mine).hexdigest(), "w_star": b["w_star"], "k_star": res["k_star"],
           "changed_top1_vs_06_off": int(sum(1 for t, o in zip(top, res["off"]) if t[0] != o[0]))}
    rec["passed"] = bool(rec["byte_identical"] if dumps_bitwise else same_rows >= SEAM_MIN_IDENTICAL
                         and len(ref_rows) == len(d.qids))
    return rec


# ================================================================ records for t10_player
def b_record(b, extra=None):
    if not b or not b.get("adopted"):
        return {"adopted": False, "reason": (b or {}).get("reason"), "problems": (b or {}).get("problems") or [],
                "rule": "PREREG_10 2: B (10B = 10A + w*·z(T_P1)) is not adopted / not usable; no P1 term is added "
                        "(t10_player.py p1 refuses, exit 3)"}
    out = {"adopted": True, "w_star": b["w_star"], "frozen_sha256_lf": b["frozen_sha256_lf"],
           "lda_sha256": b["lda_sha256"], "val_seam_sha256": b["val_seam_sha256"],
           "source": b.get("source", "PREREG_10"), "results_md_mark": b["results_md_mark"],
           "rule": "PREREG_10 2 / rev 4: 10B = CSLS_k*(F_sel + w* z(T_P1)) on the pools that used 06 (06-OFF / 10A), "
                   "when the GPU seam passed, the pool's P1 was committed by K6B-10, sr(10B) - sr(selected) >= 0 and "
                   "P1 failed <= 0.5 %"}
    if out["source"] == "PREREG_13":                        # PREREG_10 rev 5 (16.5): H13P instead of H10P
        out.update(h13p=b.get("h13p"), prereg13_dir=b.get("prereg13_dir"))
    else:
        out["h10p"] = b.get("h10p")
    if extra:
        out.update(extra)
    return out


def b_inputs(ctx, b, sources=None):
    """The B files a 10 with B binds (inputs_bound): only when some pool's source is 10B (rev 4 section 14: a B file
    that is not used, e.g. an uncommitted failed gpu_seam.json, never blocks an A upload)."""
    if not b or not b.get("adopted") or "10B" not in (sources or {}).values():
        return []
    pp = p1_paths(ctx)
    files = [p for k, p in b["paths"].items() if k != "val_dir"] + [pp["rehearsal"], pp["gpu_seam"]]
    files += list((b.get("p13_paths") or {}).values())    # PREREG_10 rev 5 (16.5): results/prereg13 when B came from it
    files += [p1_paths(ctx, p)["report"] for p in ("seam_val",) + tuple(t6.TEST_POOLS)]
    return [f for f in files if os.path.exists(f)]


def status_lines(T, ctx, b):
    if not b.get("adopted"):
        extra = "" if b.get("reason") in SILENT else f" -- B not used: {b.get('reason')} {b.get('problems')}"
        return [f"  B: not adopted ({b.get('reason')}; p1 exit 3; 10 is part A only){extra}"]
    pp = p1_paths(ctx)
    if b.get("source") == "PREREG_13":
        h = b.get("h13p") or {}
        lines = [f"  B: adopted via PREREG_13 (PREREG_10 rev 5; w* {b['w_star']}, H13P {h.get('mean_diff')}, low "
                 f"{h.get('ci95_low')} > {h.get('ci95_low_must_exceed')}, n_view {h.get('n_view')})"]
    else:
        lines = [f"  B: adopted (w* {b['w_star']}, H10P {b['h10p'].get('mean_diff')}, low {b['h10p'].get('ci95_low')})"]
    k5 = k5_b_state(T, ctx, b)
    if k5["checked"]:
        p13 = k5.get("prereg13")
        lines.append(f"  B K5: {'ok' if k5['ok'] else 'NOT MET'} (rev 4 {k5.get('rev4_ok')} {k5.get('main_commit')}"
                     + (f"; rev 5 {p13.get('ok')} {p13.get('main_commit')} by {p13.get('deadline')}" if p13 else "")
                     + (f"; {k5['error']}" if k5.get("error") else "") + ")")
    r = t6.maybe_json(pp["rehearsal"])
    lines.append("  B rehearsal: " + (f"m {r.get('m')}, tol_P {r.get('tol_P')}" if r else "missing (tol_P 【待演練】)"))
    s = t6.maybe_json(pp["gpu_seam"])
    lines.append("  B GPU seam: " + ("missing" if not s else ("passed" if s.get("passed") else "FAILED (B not used)")))
    for pool in t6.TEST_POOLS:
        q = p1_paths(ctx, pool)
        r = t6.maybe_json(q["report"])
        n = len(F.chunk_files(q["chunks"])) if os.path.isdir(q["chunks"]) else 0
        stop = os.path.exists(os.path.join(q["chunks"], "STOP"))
        lines.append(f"  B p1 {pool}: " + (f"done ({r.get('games')} games, failed {r.get('failed')})" if r else
                                           f"{n} chunk(s){' STOP present' if stop else ''}"))
    nu = t6.maybe_json(pp["not_used"])
    if nu:
        lines.append(f"  B not used: {[e['reason'] for e in nu.get('events', [])]}")
    return lines
