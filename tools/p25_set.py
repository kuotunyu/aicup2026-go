"""PREREG_25 competition 2: freeze Q25 = Q19 with the SET term's three models S19 replaced by S25 (S19's set model
retrained with the validation players), and record its adoption after the independent audit
(docs/PREREG_25_SET_MORE_PLAYERS.md rev 1, section 2).

    PY = .venv/Scripts/python.exe   (CPU; Git Bash; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL thread variable;
                                     from a path without 'AppData\\Local' (u1_lib's data/local guard), e.g. a subst drive)
    PY tools/p25_set.py freeze      -> results/prereg25/rank/s25_seed{0,1,2}.npz (copies), freeze_report.json,
                                       frozen_config.json (run once; commit them, then the independent audit)
    PY tools/p25_set.py finalize    (after results/prereg25/rank/freeze_audit.json is committed) -> adoption.json

freeze (PREREG_25 2.1): the inner evaluation (tools/p09_rank_fuse.Inner on data/prereg12/q09_inner, as 19's freeze),
    seams 1 (Q02H / Q08 / Q09), M (model files, exploration records at 1965276, 19's models), P (inner files), E (own
    pooling = Inner.E_p1), F19 (Q19 OOF with the frozen S19 = 19's numbers) and F (Q25 OOF with S25 = the exploration's
    eval_report.json, fold weights included), then ONE fit of the 7-term theta on all inner fit pseudo-questions
    (p04_rank_fuse.gen_fit_record) and the direction references.
finalize (PREREG_25 2.3 / 2.4): binds the committed freeze_audit.json to the frozen files and writes adoption.json
    (usable true unless the audit lists a program error affecting the predictions).

There is no validation apply (PREREG_25 2.2): no validation question, feature or label is read, and no path under
data/local is ever opened, listed or hashed by this tool. Never imports torch; tools/score.py is never run; the ledger
gets no row. Exit codes: 0 done; 1 the step ran and says stop (a seam failed); 2 refused or error.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import p19_set as P19  # noqa: E402  (19's freeze helpers, imported unchanged)
import p19_forward as SF  # noqa: E402
import p25_forward as SF25  # noqa: E402

r9, R, rf, p4, P7, FW = P19.r9, P19.R, P19.rf, P19.p4, P19.P7, P19.FW


class P25Error(RuntimeError):
    pass


Stop = P19.Stop
ERRORS = (P25Error,) + P19.ERRORS

# ================================================================ registered values (PREREG_25 rev 1)
L = rf.N_LEVELS
TOL = 1e-12
SCALE = SF25.SCALE
TERMS = list(P19.TERMS)                                # OP, O4, SWA, SWB, HS, P1TO (= P1TO_E), SET (S25)
EXPLORE_COMMIT = "1965276"                             # the independent audit of the exploration
EXPLORE_FILES = ("results/prereg25/explore/PLAN.md", "results/prereg25/explore/p25_lib.py",
                 "results/prereg25/explore/p25_train.py", "results/prereg25/explore/p25_eval.py",
                 "results/prereg25/explore/train_record.json", "results/prereg25/explore/eval_report.json",
                 "results/prereg25/explore/audit.json", "results/prereg25/explore/independent_audit.json")
# the exploration's judged candidate (eval_report.json, 4aa42f5): seam F; Q19 (19's freeze and the baseline): seam F19
REF_Q25 = {"inner": 0.7456519871546352, "mono": 0.7482450052195216}
REF_Q19 = dict(P19.REF_Q19)
REF_Q09 = dict(P19.REF_Q09)
P19_FROZEN = "results/prereg19/rank/frozen_config.json"
P19_FROZEN_SHA = "1ba68c5af4571b20dd98ab2d6207462ea9b97d3557b4985ee4684263aeca6644"   # LF (t19_rank.P19_PINS)
INNER_PREP = P19.INNER_PREP
MODELS_SRC = os.path.join(REPO, "data", "prereg25", "models")
OUT = os.path.join(REPO, "results", "prereg25", "rank")
PREREG = os.path.join(REPO, "docs", "PREREG_25_SET_MORE_PLAYERS.md")
PREREG_REL = "docs/PREREG_25_SET_MORE_PLAYERS.md"
TOOL_FILES = ("tools/p25_set.py", "tools/p25_forward.py", "tools/p19_set.py") + tuple(P19.TOOL_FILES[1:])
FORMATS = {"freeze": "c10-prereg25 rank freeze_report v1", "frozen": "c10-prereg25 rank frozen_config v1",
           "audit": "c10-prereg25 freeze_audit v1", "adoption": "c10-prereg25 adoption v1"}
AUDIT_BOUND = ("frozen_config.json", "freeze_report.json", "s25_seed0.npz", "s25_seed1.npz", "s25_seed2.npz")
MIN_REV_RERUN = 2

path_of, rel, git, commit_time, tracked_clean = P7.path_of, P7.rel, P7.git, P7.commit_time, P7.tracked_clean
sha_raw, sha_lf, pin_sha, read_json, write_json = r9.sha_raw, r9.sha_lf, r9.pin_sha, r9.read_json, r9.write_json
git_blob, show = P19.git_blob, P19.show


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(*x):
    print(*x, flush=True)


def tool_state():
    rc, st = git(["status", "--porcelain", "--untracked-files=all", "--"] + list(TOOL_FILES))
    dirty = [ln[3:] for ln in st.splitlines() if ln.strip()] if rc == 0 else None
    return {"git_head": P7.git_head(), "dirty_or_untracked": dirty, "files_sha256_rule": "sha256, CRLF folded to LF",
            "files_sha256": {f: (sha_lf(path_of(f)) if os.path.exists(path_of(f)) else None) for f in TOOL_FILES}}


def require_clean_tools():
    ts = tool_state()
    if ts["git_head"] is None or ts["dirty_or_untracked"] is None:
        raise P25Error("git is not available")
    if ts["dirty_or_untracked"]:
        raise P25Error(f"commit the tools first (uncommitted: {ts['dirty_or_untracked']})")
    if tracked_clean([PREREG]):
        raise P25Error(f"{PREREG_REL} must be committed and clean")
    return ts


def venv_record():
    return r9.venv_versions(False)


def check_pins():
    """19's pins (p19_set.check_pins: its exploration and the prereg09 files), 19's frozen configuration and models,
    and the prereg25 exploration's files as committed at 1965276 (LF SHA-256 of the commit's bytes)."""
    got = dict(P19.check_pins())
    bad = []
    for r_ in EXPLORE_FILES:
        blob = git_blob(EXPLORE_COMMIT, r_)
        want = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() if blob is not None else None
        got[r_] = sha_lf(path_of(r_)) if os.path.exists(path_of(r_)) else None
        if want is None or got[r_] != want:
            bad.append(r_)
    got[P19_FROZEN] = sha_lf(path_of(P19_FROZEN)) if os.path.exists(path_of(P19_FROZEN)) else None
    if got[P19_FROZEN] != P19_FROZEN_SHA:
        bad.append(P19_FROZEN)
    for s, p in zip(SF.SEEDS, P19.model_paths()):
        got[rel(p)] = SF.sha256_file(p) if os.path.exists(p) else None
        if got[rel(p)] != SF.MODEL_SHA256[s]:
            bad.append(rel(p))
    if bad:
        raise P25Error(f"pinned files changed: {bad} (PREREG_25 1.1: never edited)")
    return got


def revision_after(n, when):
    """PREREG_25 rev N (N >= MIN_REV_RERUN) at HEAD of the committed, clean registration, added after `when` (ISO)."""
    if n is None or n < MIN_REV_RERUN:
        raise P25Error(f"a rerun needs --revision N (PREREG_25 rev {MIN_REV_RERUN} or later naming the program error)")
    if tracked_clean([PREREG]):
        raise P25Error(f"{PREREG_REL} must be committed and clean")
    row = f"| rev {n} |"
    head = git_blob("HEAD", PREREG_REL)
    if head is None or not any(x.startswith(row) for x in head.decode("utf-8", "replace").splitlines()):
        raise P25Error(f"{PREREG_REL} at HEAD has no row starting {row!r}")
    rc, out = git(["log", "--reverse", "-S" + row, "--format=%H %cI", "--", PREREG_REL])
    if rc != 0 or not out:
        raise P25Error(f"no commit added {row!r}")
    h, t = out.splitlines()[0].split()
    if when and datetime.datetime.fromisoformat(t) <= datetime.datetime.fromisoformat(when):
        raise P25Error(f"PREREG_25 rev {n} ({t}) is not later than the record it replaces ({when})")
    return {"rev": n, "commit": h, "committed_at": t}


# ================================================================ the models
def model_paths(folder=OUT):
    return [os.path.join(folder, SF25.model_name(s)) for s in SF25.SEEDS]


def install_models():
    """PREREG_25 1.2: the three exploration model files copied byte for byte into results/prereg25/rank (once)."""
    rec = {}
    for s, dst in zip(SF25.SEEDS, model_paths()):
        src = os.path.join(MODELS_SRC, SF25.model_name(s))
        if not os.path.exists(dst):
            if not os.path.exists(src) or SF25.sha256_file(src) != SF25.MODEL_SHA256[s]:
                raise P25Error(f"{show(src)} is missing or not the registered model of seed {s}")
            os.makedirs(OUT, exist_ok=True)
            shutil.copyfile(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        got = SF25.sha256_file(dst)
        src_sha = SF25.sha256_file(src) if os.path.exists(src) else None
        if got != SF25.MODEL_SHA256[s] or src_sha != SF25.MODEL_SHA256[s]:
            raise P25Error(f"{rel(dst)} / {show(src)} are not the registered model of seed {s}")
        rec[str(s)] = {"file": show(dst), "sha256": got, "source": show(src), "source_sha256": src_sha,
                       "best_step": SF25.BEST_STEP[s]}
    tr = read_json(path_of("results/prereg25/explore/train_record.json"))
    for s in SF25.SEEDS:
        if (tr.get("models_sha256") or {}).get(str(s)) != SF25.MODEL_SHA256[s] or \
                ((tr.get("seeds") or {}).get(str(s)) or {}).get("best_step") != SF25.BEST_STEP[s]:
            raise P25Error(f"seed {s}: the registered SHA-256 / step differ from the exploration's train_record.json")
    return rec


# ================================================================ freeze (PREREG_25 2.1)
def fold_weight_diff(cand, ref_w):
    my_w = [fr["params"]["w"] for fr in cand["folds"]]
    if len(my_w) != len(ref_w) or not all(set(x) == set(TERMS) for x in ref_w):
        return None
    return max((abs(x[k] - y[k]) for x, y in zip(my_w, ref_w) for k in TERMS), default=None)


def cmd_freeze(a):
    t0 = time.time()
    r9.blas_env_check(False)
    venv = venv_record()
    fc19 = read_json(path_of(P19_FROZEN))
    want_v = {k: ((fc19.get("environment") or {}).get("versions") or {}).get(k) for k in ("python", "numpy", "scipy")}
    if {k: venv[k] for k in want_v} != want_v:
        raise P25Error(f".venv {venv} differs from prereg19's frozen versions {want_v}")
    frozen_path, rep_path = os.path.join(OUT, "frozen_config.json"), os.path.join(OUT, "freeze_report.json")
    rerun = None
    if os.path.exists(frozen_path) or os.path.exists(rep_path):
        if not (a.rerun and a.reason):
            raise P25Error("freeze ran already (frozen_config.json / freeze_report.json exist): run once; a rerun "
                           "needs --rerun --reason TEXT --revision N")
        prior = [p for p in (frozen_path, rep_path) if os.path.exists(p)]
        when = max((commit_time(p) or "") for p in prior) or None
        rerun = dict(revision_after(a.revision, when), reason=a.reason,
                     replaced={rel(p): pin_sha(p) for p in prior})
    ts = require_clean_tools()
    pins = check_pins()
    models = install_models()
    par25 = SF25.load_models(OUT)
    par19 = SF.load_models(P19.OUT)
    seam_p = P19.inner_prep_seam()
    if not seam_p["ok"]:
        raise P25Error(f"seam P: {seam_p}")
    # ---- the inner evaluation (u1_lib through p09_rank_fuse.Inner: training players only)
    prep = P19.InnerPrep(INNER_PREP)
    inner = r9.Inner(prep, log=log)
    mp = FW.load_map(path_of("results/prereg09/rank/p1e_map.npz"))
    lp_q09 = r9.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    q09 = inner.run(list(r9.Q08_TERMS), {"HS": inner.E_hs, "P1TO": E_q09})
    seam1 = {"Q02H": [inner.h["rb"], r9.REF["Q02H"]], "Q08": [inner.q08["rb"], r9.REF["Q08"]],
             "Q08_mono": [inner.q08["mono_rb"], r9.REF["Q08_mono"]], "Q09": [q09["rb"], REF_Q09["inner"]],
             "Q09_mono": [q09["mono_rb"], REF_Q09["mono"]]}
    seam1_ok = all(abs(x - y) <= TOL for x, y in seam1.values())
    se = P19.seam_e(inner, lp_q09)
    log(f"seam 1 {seam1_ok} {seam1}; seam E {se}")
    if not (seam1_ok and se["ok"]):
        raise Stop(f"seam 1 / E failed: {seam1} {se}")
    # ---- seam F19: 19's frozen S19 through this tool = 19's freeze and the exploration's baseline
    tc = time.time()
    E19, info19 = P19.inner_set_terms(prep, inner, par19, mp["mu_x"], mp["sd_x"])
    q19 = inner.run(TERMS, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E19})
    fr19 = read_json(path_of("results/prereg19/rank/freeze_report.json"))
    w19 = [f["params"]["w"] for f in (((fr19.get("oof") or {}).get("Q19") or {}).get("folds") or [])]
    seamf19 = {"inner": [q19["rb"], REF_Q19["inner"]], "mono": [q19["mono_rb"], REF_Q19["mono"]],
               "fold_weights_max_abs_diff_vs_prereg19_freeze": fold_weight_diff(q19, w19)}
    seamf19["ok"] = bool(abs(q19["rb"] - REF_Q19["inner"]) <= TOL and abs(q19["mono_rb"] - REF_Q19["mono"]) <= TOL
                         and seamf19["fold_weights_max_abs_diff_vs_prereg19_freeze"] is not None
                         and seamf19["fold_weights_max_abs_diff_vs_prereg19_freeze"] <= TOL)
    log(f"seam F19 {seamf19} ({time.time() - tc:.0f}s)")
    if not seamf19["ok"]:
        raise Stop(f"seam F19 failed (the tool does not reproduce 19's frozen Q19 OOF): {seamf19}")
    del E19
    # ---- seam F: the exploration's judged candidate Q25, reproduced
    tc = time.time()
    E_set, set_info = P19.inner_set_terms(prep, inner, par25, mp["mu_x"], mp["sd_x"])
    cand = inner.run(TERMS, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_set})
    ev = read_json(path_of("results/prereg25/explore/eval_report.json"))
    ev_c = ev.get("candidate") or {}
    w_diff = fold_weight_diff(cand, ev_c.get("fusion_weights_by_fold") or [])
    seamf = {"inner": [cand["rb"], REF_Q25["inner"]], "mono": [cand["mono_rb"], REF_Q25["mono"]],
             "eval_report_inner": ev_c.get("inner_rank_balanced"),
             "eval_report_mono": ev_c.get("single_colour_rank_balanced"), "fold_weights_max_abs_diff": w_diff,
             "eval_report_gates_passed": (ev.get("gates") or {}).get("passed")}
    seamf["ok"] = bool(abs(cand["rb"] - REF_Q25["inner"]) <= TOL and abs(cand["mono_rb"] - REF_Q25["mono"]) <= TOL
                       and seamf["eval_report_inner"] == REF_Q25["inner"]
                       and seamf["eval_report_mono"] == REF_Q25["mono"] and w_diff is not None and w_diff <= TOL
                       and seamf["eval_report_gates_passed"] is True)
    log(f"SET term {set_info} ({time.time() - tc:.0f}s); seam F {seamf}")
    if not seamf["ok"]:
        raise Stop(f"seam F failed (the tool does not reproduce the exploration's candidate): {seamf}")
    cmp_ = inner.compare(cand, q19)
    # ---- ONE fit of the 7-term theta on all inner fit pseudo-questions (PREREG_25 1.3)
    theta, refit = p4.gen_fit_record(cand["aux"]["mf"], cand["aux"]["Ef"], inner.Xf.n, inner.Xf.y)
    params = p4.gen_params(theta, TERMS)
    Pfit = R.fuse_probs(theta, cand["aux"]["mf"], cand["aux"]["Ef"], inner.Xf.n, np.ones(len(inner.Xf), dtype=bool))
    mfit, _ = rf.metrics(rf.rule_b_np(Pfit), inner.Xf.y, inner.Xf.n, inner.Xf.player)
    refit.update(fit_questions=len(inner.Xf), bounds_hit=P7.bound_hits(params["w"]),
                 in_sample_fit_rank_balanced_record_only=mfit["rank_balanced"],
                 in_sample_fit_question_mean_record_only=mfit["question_mean"])
    dref = r9.direction_ref(inner, E_set, inner.q02)
    dref["definition"] = dref["definition"].replace("E_P1 = masked mean of 4 lp of the candidate map",
                                                    "E_SET = 4 lp_SET25 of the question's usable games")
    folds = [{"fold": fr["fold"], "loss": fr["loss"], "params": fr["params"],
              "bounds_hit": P7.bound_hits(fr["params"]["w"])} for fr in cand["folds"]]
    common = {"created": now_iso(), "tool": ts, "venv": venv, "pinned_sha256": pins, "models": models,
              "inner_prep": {"dir": show(INNER_PREP), **seam_p}, "rerun": rerun}
    report = {"format": FORMATS["freeze"], "what": "PREREG_25 2.1 freeze: seams 1 / M / P / E / F19 / F, then one fit "
                                                   "of the 7-term theta on all inner fit pseudo-questions",
              **common, "seconds": round(time.time() - t0, 1), "seam_1": seam1, "seam_e": se, "seam_f19": seamf19,
              "seam_f": seamf, "set_term": set_info, "oof_vs_q19_record": cmp_,
              "oof": {"Q25": {"inner_rank_balanced": cand["rb"], "single_colour_rank_balanced": cand["mono_rb"],
                              "folds": folds},
                      "Q19": {"inner_rank_balanced": q19["rb"], "single_colour_rank_balanced": q19["mono_rb"]},
                      "Q09": {"inner_rank_balanced": q09["rb"], "single_colour_rank_balanced": q09["mono_rb"]}},
              "refit": refit, "params": params, "direction_ref": dref}
    frozen = {"format": FORMATS["frozen"],
              "what": "PREREG_25 competition 2 frozen configuration Q25 = Q19 with the SET term's models S19 replaced "
                      "by S25 (inner only; adopted on the exploration's single judgement, "
                      "results/prereg25/explore/eval_report.json, and the independent audit)",
              **common, "adopted_on": {"eval_report": "results/prereg25/explore/eval_report.json",
                                       "eval_commit": "4aa42f5",
                                       "audit": "results/prereg25/explore/independent_audit.json",
                                       "audit_commit": EXPLORE_COMMIT, "gates": ev.get("gates")},
              "combo": {"id": "Q25", "terms": TERMS, "path": "gen", "pool": "log", "calibrated": True, "opening": True,
                        "p1_scale": SCALE, "p1_term": "P1TO_E (ens3fam, results/prereg09/rank/p1e_map.npz)",
                        "set_scale": SCALE, "set_term": "SET (S25, 3 seeds, results/prereg25/rank/s25_seed*.npz)"},
              "params": params,
              "formula": "p = softmax(e^a (n/12)^gamma (m_C5 + w_OP m_OP + w_O4 E_O4 + w_SWA E_SWA + w_SWB E_SWB + "
                         "w_HS E_HS + w_P1 E_P1E + w_SET E_SET + B beta)); E_SET = 4 lp_SET25, lp_SET25 = log(mean "
                         "over the 3 S25 seeds of head(mean over the usable games of h(x))), 0 without a usable game "
                         "(tools/p25_forward.py -> tools/p19_forward.py)",
              "decision_rule": "B, tools/decide.py (argmax p_k + e^-1 (p_{k-1} + p_{k+1}))",
              "p1e_map": {"file": "results/prereg09/rank/p1e_map.npz",
                          "sha256": P19.P09_PINS["results/prereg09/rank/p1e_map.npz"]},
              "q09_frozen_sha256": P19.P09_PINS["results/prereg09/rank/frozen_config.json"],
              "q19_frozen_sha256": P19_FROZEN_SHA,
              "direction_ref": dref,
              "oof": report["oof"], "refit": {k: refit[k] for k in ("loss", "starts", "fit_questions", "bounds_hit")},
              "environment": {"versions": venv, "cpu": r9.environment_cpu()}}
    write_json(rep_path, report)
    write_json(frozen_path, frozen)
    w = params["w"]
    log(f"frozen Q25: w {json.dumps({k: round(v, 4) for k, v in w.items()})}, a {params['a']:.4f}, gamma "
        f"{params['gamma']:.4f}; bounds hit {refit['bounds_hit']}; rho_eval {dref['rho_eval']:.4f}, rho_mono "
        f"{dref['rho_mono']:.4f}; {time.time() - t0:.0f}s")
    log("commit results/prereg25/rank/ (frozen_config.json, freeze_report.json, s25_seed*.npz), then the independent "
        "audit (freeze_audit.json), then finalize")
    return 0


# ================================================================ finalize (PREREG_25 2.3 / 2.4)
def audit_binding(audit, frozen_commit, audit_commit):
    probs = []
    if audit.get("format") != FORMATS["audit"]:
        probs.append(f"audit format {audit.get('format')!r} != {FORMATS['audit']!r}")
    if not isinstance(audit.get("program_errors_affecting_predictions"), list):
        probs.append("audit field program_errors_affecting_predictions is not a list")
    for n_ in AUDIT_BOUND:
        p = os.path.join(OUT, n_)
        if not os.path.exists(p) or (audit.get("inputs_sha256") or {}).get(n_) != pin_sha(p):
            probs.append(f"audit inputs_sha256[{n_}] differs from the committed file")
    if (not frozen_commit or not audit_commit
            or datetime.datetime.fromisoformat(audit_commit) <= datetime.datetime.fromisoformat(frozen_commit)):
        probs.append(f"audit commit {audit_commit} is not later than frozen_config.json's {frozen_commit}")
    return probs


def cmd_finalize(a):
    ap = os.path.join(OUT, "adoption.json")
    if os.path.exists(ap):
        raise P25Error("adoption.json exists: the adoption is recorded once (a change needs a revision)")
    frozen = [os.path.join(OUT, n_) for n_ in AUDIT_BOUND]
    r9.require_committed(frozen, "frozen files (commit them first)")
    fc = read_json(os.path.join(OUT, "frozen_config.json"))
    if fc.get("format") != FORMATS["frozen"] or (fc.get("combo") or {}).get("id") != "Q25" or \
            (fc.get("combo") or {}).get("terms") != TERMS or \
            len((fc.get("params") or {}).get("theta") or []) != 2 + len(TERMS) + L - 1:
        raise P25Error("frozen_config.json is not the frozen Q25 configuration")
    for s, p in zip(SF25.SEEDS, model_paths()):
        if SF25.sha256_file(p) != SF25.MODEL_SHA256[s] or (fc["models"][str(s)] or {}).get("sha256") != \
                SF25.MODEL_SHA256[s]:
            raise P25Error(f"{rel(p)} is not the registered model")
    pins = check_pins()
    ts = require_clean_tools()
    audp = os.path.join(OUT, "freeze_audit.json")
    if not os.path.exists(audp) or tracked_clean([audp]):
        raise P25Error("freeze_audit.json is not committed (finalize only after the independent audit, PREREG_25 2.3)")
    audit = read_json(audp)
    probs = audit_binding(audit, commit_time(os.path.join(OUT, "frozen_config.json")), commit_time(audp))
    if probs:
        raise P25Error(f"freeze_audit.json is not an audit of these frozen files: {probs}")
    errors = audit["program_errors_affecting_predictions"]
    rec = {"format": FORMATS["adoption"], "what": "PREREG_25 2.4: adoption record of Q25 (inner only; no validation "
                                                  "apply, PREREG_25 2.2)",
           "combo": "Q25", "usable": not errors,
           "usable_blocked_reasons": (["freeze_audit.json found a program error affecting the predictions"]
                                      if errors else []),
           "finalized": now_iso(), "audit": {"file": rel(audp), "sha256": sha_lf(audp), "commit": commit_time(audp),
                                             "auditor": audit.get("auditor"), "verdict": audit.get("verdict")},
           "frozen_sha256": {n_: pin_sha(os.path.join(OUT, n_)) for n_ in AUDIT_BOUND},
           "frozen_commit": commit_time(os.path.join(OUT, "frozen_config.json")),
           "params": fc["params"], "direction_ref": {k: fc["direction_ref"][k] for k in ("rho_eval", "rho_mono")},
           "pinned_sha256": pins, "tool": ts}
    write_json(ap, rec)
    log(f"adoption: usable {rec['usable']} {rec['usable_blocked_reasons']}; commit {rel(ap)}")
    return 0


# ================================================================ command line
def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason")
    p.add_argument("--revision", type=int)
    sub.add_parser("finalize")
    a = ap.parse_args(argv)
    try:
        return {"freeze": cmd_freeze, "finalize": cmd_finalize}[a.cmd](a)
    except Stop as e:
        print(f"STOP (exit 1): {e}", file=sys.stderr)
        return 1
    except ERRORS as e:
        print(f"REFUSED (exit 2): {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
