"""PREREG_10 (competition 1) part B, the validation side (docs/PREREG_10_PLAYER_GUARD_P1.md section 2.5): apply,
apply --finalize and the helpers tools/p10_p1.py extract --set val uses. Split from tools/p10_fuse.py (PREREG_10 rev 2)
so that the mock / freeze tool binding covers only the mock path (p10_fuse.MOCK_TOOL_FILES); apply binds every tool
(APPLY_TOOL_FILES) and requires the mock-path files to equal the ones frozen_config.json records.

    PY tools/p10_fuse.py apply [--rerun --reason T]      (forwards here; or PY tools/p10_apply.py apply ...)
    PY tools/p10_fuse.py apply --finalize                 (after prescore_audit.json is committed)

Discipline as tools/p10_fuse.py: the p06 data/local guard is on (imported through p10_fuse); only the two validation
game files are opened, inside p06_style.allow_local(); tools/score.py is never run or imported; no torch / p1_lib.
Exit codes: 0 done (score_allowed 'pending_audit' or finalized); 1 registered stop / blocked; 2 refused or error.
"""
import datetime
import hashlib
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p10_fuse as F  # noqa: E402  (installs the p06 guard)
from p10_fuse import (FAIL_MAX, FEAT, FORMATS, OUT, P06_PLAYER, Ctx, Refused, Stop, blas_env_check,  # noqa: E402
                      chunk_files, commit_time, fuse_10b, git, load_lda, load_side0, meta, now_iso, path_of,
                      pin_sha, project_p1, read_json, rel, require_committed, require_tools, sha_lf, sha_raw, sim,
                      st, tool_state, tracked_clean, write_json, z_p1)

VAL_DIR = path_of("data/prereg10/val")
VAL_REPORT = os.path.join(FEAT, "val_report.json")
P02_PLAYER = path_of("results/prereg02/player")
LOCAL = path_of("data/local")
VAL_CANDIDATES = os.path.join(LOCAL, "player_val_candidates.csv")
VAL_QUESTIONS = os.path.join(LOCAL, "player_val_questions.csv")
VAL_SET = "p10val"
APPLY_TOOL_FILES = F.MOCK_TOOL_FILES + ("tools/p10_apply.py", "tools/p10_p1.py", "tools/p06_player.py",
                                        "tools/check_submission.py")


def frozen_usable(fc):
    if fc.get("format") != FORMATS["frozen"]:
        raise Refused("frozen_config.json is not a c10-prereg10 frozen configuration")
    if fc.get("preview"):
        raise Refused("frozen_config.json is a preview record (never accepted by a later step)")
    if fc.get("b_passed") is not True or fc.get("score_pending_val") is not True:
        raise Refused("frozen_config.json does not say B passed")
    return True


def val_keys(cand_rows, q_rows):
    """Validation game order of extract --set val and apply: candidate games (file order), then question games
    (question order, slot order). -> (keys, colours)."""
    keys = [str(r[0]) for r in cand_rows]
    cols = [r[2] for r in cand_rows]
    for q, g in q_rows:
        for s, col, _sgf in g:
            keys.append("%s#%d" % (q, s))
            cols.append(col)
    return keys, cols


def keys_sha(keys):
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def sr06_with_p1(P6, ms, cand_ids, c_by_pid, ks, earr, models, combos, kst, Z, usable, w_star):
    """sr06 pseudo pools (PREREG_06 2.3, as p06_player.sr06_apply, every round) with 10B added for every combination.
    -> ({combo: mean}, {combo: 10B mean})."""
    cg = {p: [str(ms.gid[r]) for r in c_by_pid[p]] for p in cand_ids}
    row_of = {(p, str(ms.gid[r])): r for p in cand_ids for r in c_by_pid[p]}
    rr = st.sr06_rounds(cg, ks)
    pts = {n: [] for n, _w in combos}
    pts["R02"] = []
    p10 = {n: [] for n, _w in combos}
    for r, items in enumerate(rr):
        cgr = [[row_of[(p, g)] for g in refs] for p, _q, refs in items]
        qsel = [(i, [row_of[(p, g)] for g in q]) for i, (p, q, _refs) in enumerate(items) if q]
        qi = np.array([i for i, _r in qsel], np.int64)
        qg = [rws for _i, rws in qsel]
        T = P6.apply_terms(ms, models, cgr, qg, (earr, cgr, earr, qg))
        zT = z_p1(Z, cgr, qg, usable)
        for n, w in combos:
            F = sim.combo_F(T, "E0", w)
            pts[n] += st.points(st.ranks(st.csls(F, kst), qi)).tolist()
            p10[n] += st.points(st.ranks(fuse_10b(F, zT, w_star, kst), qi)).tolist()
        pts["R02"] += st.points(st.ranks(sim.combo_F(T, "E0", None), qi)).tolist()
    return ({n: float(np.mean(v)) for n, v in pts.items()}, {n: float(np.mean(v)) for n, v in p10.items()})


AUDIT_BOUND_OUTPUTS = ("player_val_top5.csv",)
AUDIT_BOUND_INPUTS = ("p1_lda.npz", "frozen_config.json", "val_report.json")


def apply_decision(same06, srm_ok, fail_frac, gate_ok, fmt_ok, audit):
    """score_allowed of 2.5 -> (False / 'pending_audit', reasons). apply never writes True."""
    reasons = []
    if not same06:
        reasons.append("06 is not reproduced byte for byte")
    if not srm_ok:
        reasons.append("the sr06 means differ from results/prereg06/player/apply_report.json")
    if fail_frac > FAIL_MAX:
        reasons.append(f"P1 failed games {fail_frac:.4%} > 0.5 %")
    if not gate_ok:
        reasons.append("pool gate sr(10B) - sr(06 selected) < 0: B not used this season (not submitted)")
    if not fmt_ok:
        reasons.append("check_submission did not pass")
    if audit == "bug":
        reasons.append("prescore_audit.json found a program error")
    if reasons:
        return False, reasons
    return "pending_audit", ["prescore_audit.json of these outputs not committed yet (PREREG_10 2.5); then "
                             "apply --finalize"]


def audit_binding(audit, rep, audit_commit, apply_commit):
    probs = []
    if audit.get("format") != FORMATS["audit"]:
        probs.append(f"audit format {audit.get('format')!r} != {FORMATS['audit']!r}")
    if not isinstance(audit.get("program_errors_affecting_10B"), list):
        probs.append("audit field program_errors_affecting_10B is not a list")
    if audit.get("independent_recompute_identical") is not True:
        probs.append("audit does not say the independent T_P1 / Top-5 recompute is identical")
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


def audit_state(out_dir, rep=None):
    p = os.path.join(out_dir, "prescore_audit.json")
    if not os.path.exists(p):
        return "missing", None
    if tracked_clean([p]):
        return "missing", {"note": "prescore_audit.json exists but is not committed"}
    au = read_json(p)
    rec = {"sha256": sha_lf(p), "commit": commit_time(p)}
    if rep is not None:
        probs = audit_binding(au, rep, rec["commit"], commit_time(os.path.join(out_dir, "apply_report.json")))
        if probs:
            rec["binding_problems"] = probs
            return "unbound", rec
    return ("bug" if au.get("program_errors_affecting_10B") else "ok"), rec


def finalize_decision(rep, audit, audit_rec):
    if rep.get("score_allowed") is not False and rep.get("score_allowed") != "pending_audit":
        raise Refused(f"score_allowed is already {rep.get('score_allowed')!r}")
    if audit == "missing":
        raise Refused("prescore_audit.json is not committed (finalize only after the audit)")
    if audit == "unbound":
        raise Refused(f"prescore_audit.json is not an audit of these outputs: "
                      f"{(audit_rec or {}).get('binding_problems')}")
    blocked = [r for r in rep.get("score_blocked_reasons") or [] if not r.startswith("prescore_audit.json")]
    if rep.get("score_allowed") is False and blocked:
        raise Refused(f"score_allowed is false for other reasons: {blocked}")
    if audit == "bug":
        return False, ["prescore_audit.json found a program error"]
    if audit != "ok":
        raise Refused(f"audit state {audit!r}")
    return True, []


def apply_prerequisites():
    """-> (frozen config, its commit, lda model, val report, val chunk files)."""
    fp = os.path.join(OUT, "frozen_config.json")
    lp = os.path.join(OUT, "p1_lda.npz")
    fcommit = require_committed([fp, lp], "frozen_config.json + p1_lda.npz")
    fc = read_json(fp)
    frozen_usable(fc)
    if sha_raw(lp) != fc["lda"]["sha256"]:
        raise Refused("p1_lda.npz differs from frozen_config.json")
    for f, sha in fc["p06_frozen_sha256"].items():
        if pin_sha(os.path.join(P06_PLAYER, f)) != sha:
            raise Refused(f"results/prereg06/player/{f} differs from frozen_config.json")
    vcommit = require_committed([VAL_REPORT], "val_report.json")
    vr = read_json(VAL_REPORT)
    if vr.get("format") != FORMATS["val_report"] or vr.get("mode") != "real" or not vr.get("complete"):
        raise Refused("val_report.json is not a complete real validation extraction")
    if vr.get("frozen_config_sha256") != sha_lf(fp):
        raise Refused("val_report.json was made under another frozen_config.json")
    files = chunk_files(VAL_DIR)
    want = {r["file"]: r["sha256"] for r in vr["chunks"]}
    if sorted(os.path.basename(f) for f in files) != sorted(want):
        raise Refused("the validation chunk files differ from val_report.json")
    bad = [os.path.basename(f) for f in files if sha_raw(f) != want[os.path.basename(f)]]
    if bad:
        raise Refused(f"validation chunks changed since val_report.json: {bad[:5]}")
    return fc, {"frozen": fcommit, "val_report": vcommit}, load_lda(lp), vr, files


def run_check_submission(path, q, cnd):
    r = subprocess.run([sys.executable, os.path.join(HERE, "check_submission.py"), "player", path,
                        "--questions", q, "--candidates", cnd], capture_output=True, text=True)
    return {"returncode": r.returncode, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}


def apply_core(cand_path, q_path, p02_dir, p06_dir, vfiles, val_set, vr, model, w_star, out_dir, smoke=False):
    """2.5 on one pool: 06 recomputed with the 06 functions (byte-identical to p06_dir/player_val_top5.csv, else
    Stop), the sr06 pseudo pools with 10B (pool gate), 10B's Top-5 written to out_dir/player_val_top5.csv,
    check_submission. -> report dict (the caller adds its bindings and writes it)."""
    import scipy.sparse as sp
    import p02_player_fuse as pf
    import p06_player as P6
    fz = read_json(os.path.join(p06_dir, "frozen_config.json"))
    rep06 = read_json(os.path.join(p06_dir, "apply_report.json"))
    ref_path = os.path.join(p06_dir, "player_val_top5.csv")
    with st.allow_local():
        d = pf.Data(cand_path, p02_dir, True, smoke, q_path=q_path)
        cand_rows = pf.read_candidates(cand_path)
        q_rows = pf.read_questions(q_path)
    if any(d.shas.get(k) != rep06["inputs"].get(k) for k in d.shas):
        raise Refused("the validation game files or prereg02 dumps differ from the ones 06 applied to")
    keys, _cols = val_keys(cand_rows, q_rows)
    if keys_sha(keys) != vr["keys_sha256"] or len(keys) != vr["games"]:
        raise Refused("validation game order differs from val_report.json")
    # ---- 06 recomputed with the 06 functions (as p06_player.cmd_apply step 4-6)
    c_groups = [d.c_by_player[p] for p in d.cand_ids]
    q_groups = [d.q_by_id[q] for q in d.qids]
    gc, gq = d.G("raw")
    models = {cc: P6.load_lda(os.path.join(p06_dir, "lda_%s.npz" % cc)) for cc in "BW"}
    by_gid = {r[0]: r for r in cand_rows}
    games = [(g[1], g[0], g[2], by_gid[g[0]][3]) for g in d.cgames]
    qsgf = {(q, s): (cc, x) for q, g in q_rows for s, cc, x in g}
    games += [(g[0], "%s#%d" % (g[0], g[1]), g[2], qsgf[(g[0], g[1])][1]) for g in d.qgames]
    if [str(g[1]) for g in games] != keys:
        raise Refused("06's game order differs from the P1 extraction order")
    nc = len(d.cgames)
    ms = P6.MemStore(games, sp.vstack([gc, gq]).tocsr(), models)
    earr = (np.vstack([d.cemb, d.qemb]), np.concatenate([d.cnw, d.qnw]), np.concatenate([d.cok, d.qok]))
    cgroups = [list(g) for g in c_groups]
    qgroups = [[nc + i for i in g] for g in q_groups]
    T = P6.apply_terms(ms, models, cgroups, qgroups, (earr, cgroups, earr, qgroups))
    combos = sim.combo_list(fz["combinations"]["V1"])
    order = [n for n, _w in combos]
    kst = fz["k_star"]
    ks = [len(g) for g in q_groups]
    # ---- P1 of the validation games, frozen LDA
    X, usable, failed, have, cnt = load_side0(vfiles, len(keys), val_set)
    if not have.all():
        raise Refused("validation games without a chunk")
    Z = np.zeros((len(keys), st.LDA_DIM))
    rows = np.flatnonzero(usable)
    Z[rows] = project_p1(X, model, rows)
    c_by_pid = {p: d.c_by_player[p] for p in d.cand_ids}
    srm, sr10 = sr06_with_p1(P6, ms, d.cand_ids, c_by_pid, ks, earr, models, combos, kst, Z, usable, w_star)
    srm_ok = all(round(srm[n], 6) == rep06["sr06"][n] for n in rep06["sr06"])
    chosen = P6.choose_combo({n: srm[n] for n in order}, order)
    if chosen != rep06["selected"]:
        raise Stop(f"06 selection {chosen} differs from 06's {rep06['selected']}")
    F = sim.combo_F(T, "E0", dict(combos)[chosen])
    main06 = st.top5(st.csls(F, kst), d.cand_ids)
    os.makedirs(out_dir, exist_ok=True)
    out_main = os.path.join(out_dir, "player_val_top5.csv")
    tmp06 = os.path.join(out_dir, "p06_repro.tmp.csv")
    P6.write_top5(tmp06, d.qids, main06)
    with open(tmp06, "rb") as f:
        b06 = f.read()
    os.remove(tmp06)
    with open(ref_path, "rb") as f:
        same06 = b06 == f.read()
    if not same06:
        raise Stop("06 recomputed with the 06 functions differs from results/prereg06/player/player_val_top5.csv")
    zT = z_p1(Z, cgroups, qgroups, usable)
    main10 = st.top5(fuse_10b(F, zT, w_star, kst), d.cand_ids)
    P6.write_top5(out_main, d.qids, main10)
    gate = sr10[chosen] - srm[chosen]
    fmt = run_check_submission(out_main, q_path, cand_path)
    fail_frac = float(failed.sum()) / len(keys)
    allowed, reasons = apply_decision(same06, srm_ok, fail_frac, gate >= 0, fmt["returncode"] == 0, None)
    rep = {"format": FORMATS["apply"], "what": "PREREG_10 2.5 apply of 10B on the local validation pool",
           "p06_reproduction": {"identical": same06, "sha256": hashlib.sha256(b06).hexdigest(),
                                "reference": rel(ref_path), "reference_sha256": sha_raw(ref_path)},
           "sr06": {n: round(v, 6) for n, v in srm.items()}, "sr06_equals_06_report": srm_ok,
           "sr06_10B": {n: round(v, 6) for n, v in sr10.items()}, "selected": chosen,
           "pool_gate": {"sr_10B_minus_selected": gate, "passed": bool(gate >= 0)}, "w_star": w_star, "k_star": kst,
           "p1": {"games": len(keys), "failed": int(failed.sum()), "failed_fraction": fail_frac,
                  "no_target_moves": int(((cnt == 0) & ~failed).sum()), "usable": int(usable.sum()),
                  "questions_without_usable_p1": int(sum(1 for g in qgroups if not any(usable[j] for j in g)))},
           "changed_top1_vs_06": int(sum(1 for x, y in zip(main06, main10) if x[0] != y[0])),
           "format_check": fmt,
           "outputs_sha256": {"player_val_top5.csv": sha_lf(out_main)},
           "inputs_sha256": dict(d.shas), "score_allowed": allowed, "score_blocked_reasons": reasons}
    return rep


def cmd_apply(a):
    if a.finalize:
        return apply_finalize(a)
    t0 = time.time()
    blas_env_check(False)
    ts = tool_state(APPLY_TOOL_FILES)
    require_tools(ts)
    out_main = os.path.join(OUT, "player_val_top5.csv")
    out_rep = os.path.join(OUT, "apply_report.json")
    rerun = None
    if os.path.exists(out_rep) or os.path.exists(out_main):
        if not (a.rerun and a.reason):
            raise Refused("apply outputs exist; rerun only with --rerun --reason TEXT")
        if git(["ls-files", "--error-unmatch", "--", rel(out_rep)])[0] == 0:
            raise Refused("committed apply outputs are never overwritten")
        rerun = {"reason": a.reason}
    fc, commits, model, vr, vfiles = apply_prerequisites()
    diff = F.tools_differ(ts, fc.get("tools"))
    if diff:
        raise Refused(f"mock-path tools changed since the freeze: {diff} (T_P1 / fusion must be the frozen code)")
    rep = apply_core(VAL_CANDIDATES, VAL_QUESTIONS, P02_PLAYER, P06_PLAYER, vfiles, VAL_SET, vr, model,
                     float(fc["w_star"]), OUT)
    rep["inputs_sha256"].update({"p1_lda.npz": sha_raw(os.path.join(OUT, "p1_lda.npz")),
                                 "frozen_config.json": sha_lf(os.path.join(OUT, "frozen_config.json")),
                                 "val_report.json": sha_lf(VAL_REPORT)})
    rep.update({"commits": commits, "rerun": rerun})
    rep.update(meta(_real_ctx(), t0, ts))
    write_json(out_rep, rep)
    print(f"apply: 06 reproduced {rep['p06_reproduction']['identical']}, pool gate "
          f"{rep['pool_gate']['sr_10B_minus_selected']:+.6f}, score_allowed {rep['score_allowed']} "
          f"{rep['score_blocked_reasons']}")
    return 0 if rep["score_allowed"] == "pending_audit" else 1


def _real_ctx():
    c = Ctx()
    c.preview = False
    return c


def apply_finalize(_a):
    rp = os.path.join(OUT, "apply_report.json")
    mp = os.path.join(OUT, "player_val_top5.csv")
    require_committed([rp, mp], "apply outputs")
    rep = read_json(rp)
    for n_, sha in rep["outputs_sha256"].items():
        if sha_lf(os.path.join(OUT, n_)) != sha:
            raise Refused(f"{n_} changed since apply")
    audit, audit_rec = audit_state(OUT, rep)
    allowed, reasons = finalize_decision(rep, audit, audit_rec)
    rep["score_allowed"], rep["score_blocked_reasons"] = allowed, reasons
    rep["audit"] = audit_rec
    rep["finalized"] = now_iso()
    rep["finalized_by_tool"] = tool_state(APPLY_TOOL_FILES)["files_sha256"]
    write_json(rp, rep)
    print(f"score_allowed: {allowed}")
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return run(cmd_apply, a)


def run(fn, a):
    try:
        return fn(a)
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 1
    except (Refused, st.Refused) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
