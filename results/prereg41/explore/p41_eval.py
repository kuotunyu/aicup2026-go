"""prereg41 exploration (competition 2): inner-only judgement of PLAN.md (Q41 = Q25's seven terms with the per-term
(n/12)^gamma_t fusion of p41_lib), judged once. CPU only (.venv; numpy / scipy).

    Git Bash, repository root on a subst drive; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP /
    OPENBLAS / MKL variable
    S=results/prereg41/explore
    $S/p41_eval.py eval            -> $S/eval_report.json (judged once; code committed and clean)
    $S/p41_eval.py eval --smoke    -> data/prereg41/smoke/eval_report.json (SET term replaced by default_rng(12345)
                                      noise for both Q25-noise and Q41-noise: pipeline and timing only, never a result)

Inputs and terms exactly as prereg25 / prereg40's rebuild of Q25 (p23_eval.s19_term with the frozen S25 models,
Q09's p1e_map P1TO, Inner.E_hs); the Q25 baseline is tools/p09_rank_fuse.Inner.run unchanged. Never the validation
answers, never validation features, never tools/score.py.
"""

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P23 = os.path.join(REPO, "results", "prereg23", "explore")
for p in (os.path.join(REPO, "tools"), P23, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p23_eval as E23  # noqa: E402
import p23_lib as K23  # noqa: E402
import p19_lib as K19  # noqa: E402
import p41_lib as F  # noqa: E402

r, ft_eval, FW = E23.r, E23.ft_eval, E23.FW
rf, p3 = r.rf, r.p3
guard_path = E23.guard_path
TOL = 1e-12
GRAD_TOL = 1e-5
DATA = os.path.join(REPO, "data", "prereg41")
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "eval_started.json")
REPORT = os.path.join(HERE, "eval_report.json")
P25 = os.path.join(REPO, "results", "prereg25", "explore")
S25_TRAIN_RECORD = os.path.join(P25, "train_record.json")
S25_EVAL_REPORT = os.path.join(P25, "eval_report.json")
S25_MODEL_FORMAT = "c10-prereg25 s25 model v1"
REF_Q25 = 0.7456519871546352
REF_Q25_MONO = 0.7482450052195216
OWN_FILES = ("PLAN.md", "p41_lib.py", "p41_eval.py", "test_p41.py")
IMPORTED = tuple(E23.IMPORTED) + ("results/prereg23/explore/p23_eval.py", "results/prereg23/explore/p23_lib.py",
                                  "results/prereg25/explore/train_record.json",
                                  "results/prereg25/explore/eval_report.json", "tools/p04_rank_fuse.py",
                                  "tools/p02_rank_fuse.py")
FORMAT = "c10-prereg41 explore eval v1"
NOISE_SEED = 12345


def log(*x):
    print(*x, flush=True)


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES] + list(IMPORTED)
    rel = sorted(set(rel) | E23.loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = E23.git("status", "--porcelain", "--", *rel)
    return {"head": E23.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: E23.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def once_guard(a, cs, out_path):
    prior = [p for p in (out_path, STARTED) if os.path.exists(p)]
    if not prior:
        return None
    if not (a.rerun and a.reason):
        raise SystemExit(f"{prior} exist: judged once (--rerun --reason TEXT only after a commit that names a "
                         f"concrete program error)")
    heads = {(r.read_json(p).get("code") or {}).get("head") for p in prior}
    if cs["head"] in heads:
        raise SystemExit("--rerun needs a newer commit than the earlier run's (the fix of the named error)")
    return sorted(h for h in heads if h)


def s25_models():
    tr = r.read_json(S25_TRAIN_RECORD)
    pars, seam = [], {"models": {}, "ok": True}
    for s in (0, 1, 2):
        sr = tr["seeds"][str(s)]
        path = os.path.join(REPO, *sr["model_file"].split("/"))
        sha = E23.sha_file(path)
        par, seed, best_step, msmoke = E23.load_npz_model(path, S25_MODEL_FORMAT)
        ok = bool(sha == tr["models_sha256"][str(s)] and seed == s and not msmoke and best_step == sr["best_step"])
        seam["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": ok}
        seam["ok"] = bool(seam["ok"] and ok)
        pars.append(K19.p64(par))
    return pars, seam


# ---------------------------------------------------------------- the inputs of tools/p08_rank_fuse.run_terms
def gather(inner, terms, Emap):
    """m_C5 and the per-question terms for fit / eval / mono, exactly as p08_rank_fuse.run_terms builds them."""
    d = inner.d
    di, lp = d["inner"], d["lp"]
    out = {}
    for which, X, S in (("fit", di["X_fit"], di["S_fit"]), ("eval", di["X_eval"], di["S_eval"]),
                        ("mono", di["X_mono"], di["S_mono"])):
        m = rf.pooled(S, lp["C5"], "log")
        Es = [Emap[t][which] if t in Emap else (X.mop if t == "OP" else rf.pooled(S, lp[t], "log")) for t in terms]
        out[which] = (X, m, Es)
    return out


def scored(inner, P, Pm):
    m, sc = rf.metrics(rf.rule_b_np(P), inner.Xe.y, inner.Xe.n, inner.Xe.player)
    mrec, msc = p3.mono_score_record(rf.rule_b_np(Pm), inner.Xm, inner.ms, inner.mboot)
    return {"P": P, "Pm": Pm, "sc": sc, "msc": msc, "m": m, "mrec": mrec, "rb": m["rank_balanced"],
            "mono_rb": mrec["rank_balanced"]}


def run_pt(inner, g, base, terms, free):
    (Xf, mf, Ef), (Xe, me, Ee), (Xm, mm, Em) = g["fit"], g["eval"], g["mono"]
    P, folds = F.oof_record(Xf, Xe, mf, me, Ef, Ee, base["folds"], free)
    if not np.array_equal(F.fold_predict(folds, me, Ee, Xe.n, Xe.fold, free), P):
        raise SystemExit("fold records do not reproduce the OOF probabilities")
    Pm = F.fold_predict(folds, mm, Em, Xm.n, Xm.fold, free)
    if np.isnan(Pm).any():
        raise SystemExit("a single-colour question without an OOF fold")
    res = scored(inner, P, Pm)
    res["folds"] = folds
    res["params"] = [F.params(fr["theta"], terms, free) for fr in folds]
    return res


def seam_x(inner, g, base, terms):
    """Each fold's Q25 parameters through this fusion (all gammas tied to Q25's) reproduce Q25's OOF probabilities."""
    J = len(terms)
    out = {}
    for free_name, free in (("all_free", [True] * J), ("none_free", [False] * J)):
        folds = [{"fold": b["fold"], "theta": F.from_base(b["theta"], J, free)} for b in base["folds"]]
        (_Xe, me, Ee), (_Xm, mm, Em) = g["eval"], g["mono"]
        P = F.fold_predict(folds, me, Ee, inner.Xe.n, inner.Xe.fold, free)
        Pm = F.fold_predict(folds, mm, Em, inner.Xm.n, inner.Xm.fold, free)
        out[free_name] = {"eval_max_abs": float(np.max(np.abs(P - base["P"]))),
                          "mono_max_abs": float(np.max(np.abs(Pm - base["Pm"])))}
    out["ok"] = bool(all(v["eval_max_abs"] <= TOL and v["mono_max_abs"] <= TOL for v in out.values()))
    return out


def seam_g(g, terms, n_sub=3000):
    """Analytic vs central-difference gradient of the objective on fold 0's first n_sub fit questions."""
    Xf, mf, Ef = g["fit"]
    idx = np.flatnonzero(Xf.fold != 0)[:n_sub]
    m, Es, n, y = mf[idx], [E[idx] for E in Ef], Xf.n[idx], Xf.y[idx]
    wq = rf.level_weights(y)
    J = len(terms)
    rng = np.random.default_rng(41)
    worst = 0.0
    for _ in range(2):
        free = [True] * J
        th = np.concatenate([[rng.uniform(-1, 0), rng.uniform(-0.5, 1)], rng.uniform(0.1, 1.0, J),
                             rng.uniform(-0.5, 1.0, J), 0.3 * rng.standard_normal(F.L - 1)])
        _l, gr = F.objective(th, m, Es, n, y, wq, free)
        for i in range(len(th)):
            e = np.zeros_like(th)
            e[i] = 1e-6
            num = (F.objective(th + e, m, Es, n, y, wq, free)[0] - F.objective(th - e, m, Es, n, y, wq, free)[0]) / 2e-6
            worst = max(worst, abs(num - gr[i]) / (abs(num) + 1e-3))
    return {"questions": int(len(idx)), "max_rel_diff": worst, "tol": GRAD_TOL, "ok": bool(worst <= GRAD_TOL)}


def short(inner, res, base):
    return {"inner_rank_balanced": res["rb"], "inner_question_mean": res["m"]["question_mean"],
            "single_colour_rank_balanced": res["mono_rb"],
            "single_colour_by_colour": {c: res["mrec"]["by_colour"][c]["rank_balanced"] for c in res["mrec"]["by_colour"]},
            "vs": inner.compare(res, base)}


def fold_record(res):
    return [{"fold": fr["fold"], "loss": fr["loss"], "q25_loss_same_fold": fr["q25_loss_same_fold"],
             "winner": fr["winner"], "starts": fr["starts"], "params": p}
            for fr, p in zip(res["folds"], res["params"])]


def cmd_eval(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(SMOKE_DIR, "eval_report.json") if smoke else REPORT
    cs = code_state()
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg41 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path)
    base_rec = {"format": FORMAT, "smoke": smoke, "code": cs}
    s25, sm = s25_models()
    prep = ft_eval.InnerPrep(guard_path(E23.INNER_PREP))
    map_sha = r.sha_raw(E23.Q09_MAP)
    if map_sha != r.read_json(E23.Q09_FROZEN)["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    want = (r.read_json(E23.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    inner_sha = E23.sha_file(os.path.join(E23.INNER_PREP, "inner_X.npy"))
    sm["inner_X_sha_equal_check"] = bool(inner_sha == want.get("inner_X.npy"))
    sm["ok"] = bool(sm["ok"] and sm["inner_X_sha_equal_check"])
    inner = r.Inner(prep, log=log)
    ref = E23.refs()
    mp = FW.load_map(E23.Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    terms9 = list(r.Q08_TERMS)
    q09 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_q09})
    s19_pars, _ = E23.s19_models()
    E_s19, _lp19, _h19 = E23.s19_term(prep, inner, mp, s19_pars)
    terms = terms9 + ["SET"]
    q19 = inner.run(terms, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s19})
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]], "Q19": [q19["rb"], ref["Q19"]],
            "Q19_mono": [q19["mono_rb"], ref["Q19_mono"]]}
    del q09, q19, E_s19
    E_s25, _lp25, _h25 = E23.s19_term(prep, inner, mp, s25)
    Emap = {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25}
    q25 = inner.run(terms, Emap)
    seam["Q25"] = [q25["rb"], REF_Q25]
    seam["Q25_mono"] = [q25["mono_rb"], REF_Q25_MONO]
    rep25 = r.read_json(S25_EVAL_REPORT)
    seam_ref_bound = bool(rep25["candidate"]["inner_rank_balanced"] == REF_Q25
                          and rep25["candidate"]["single_colour_rank_balanced"] == REF_Q25_MONO)
    seam1 = bool(seam_ref_bound and all(abs(x - y) <= TOL for x, y in seam.values()))
    seam1_bitwise = all(x == y for x, y in seam.values())
    g = gather(inner, terms, Emap)
    sx = seam_x(inner, g, q25, terms)
    sg = seam_g(g, terms)
    log(f"seam 1 {seam1} (bitwise {seam1_bitwise}) {seam}; seam X {sx}; seam G {sg}; seam M {sm['ok']}")
    if not (seam1 and sx["ok"] and sg["ok"] and sm["ok"]):
        E23.write_json(out_path, {**base_rec, "status": "stop: seam failed", "seam": seam,
                                  "seam_ref_bound": seam_ref_bound, "seam_x": sx, "seam_g": sg, "seam_m": sm})
        return 1
    # ---- the judged part starts here (after every seam, before any candidate number)
    if smoke:
        rng = np.random.default_rng(NOISE_SEED)
        Enoise = {w: 4.0 * rng.standard_normal(v.shape) * (np.abs(v).sum(1, keepdims=True) > 0)
                  for w, v in E_s25.items()}
        Emap = {"HS": inner.E_hs, "P1TO": E_q09, "SET": Enoise}
        q25 = inner.run(terms, Emap)
        g = gather(inner, terms, Emap)
    else:
        E23.write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                 "rerun_reason": a.reason if a.rerun else None})
    tc = time.time()
    J = len(terms)
    cand = run_pt(inner, g, q25, terms, [True] * J)
    cmp_ = inner.compare(cand, q25)
    gates = K23.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                      cmp_["mono_by_colour"])
    log(f"Q41 {cand['rb']!r} vs Q25 {q25['rb']!r}: {json.dumps(gates)} ({time.time() - tc:.0f}s)")
    # ---- record only
    counts = E23.by_count(inner, cand, q25)
    levels = E23.by_level(inner, cand, q25)
    cand_short = short(inner, cand, q25)
    cand_folds = fold_record(cand)
    del cand
    q41s = run_pt(inner, g, q25, terms, [t == "SET" for t in terms])
    q41s_rec = {**short(inner, q41s, q25), "folds": fold_record(q41s)}
    del q41s
    status = "gates passed (worth a PREREG_41; not adoption)" if gates["passed"] else "gates not passed: stop"
    rep = {**base_rec, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("SET term replaced by noise for both Q25 and Q41: numbers are meaningless" if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: inner-only judgement of Q41 = Q25's seven terms with per-term (n/12)^gamma_t fusion "
                   "(5-fold player OOF on the fit pseudo-questions); no validation scoring",
           "status": status, "seam": seam, "seam_ref_bound": seam_ref_bound, "seam1_bitwise": seam1_bitwise,
           "seam_x": sx, "seam_g": sg, "seam_m": sm,
           "candidate": {"name": "Q41", "terms": terms, **cand_short, "folds": cand_folds},
           "baseline": {"name": "Q25", "terms": terms, "inner_rank_balanced": q25["rb"],
                        "single_colour_rank_balanced": q25["mono_rb"],
                        "fusion_weights_by_fold": [fr["params"]["w"] for fr in q25["folds"]],
                        "gamma_by_fold": [fr["params"]["gamma"] for fr in q25["folds"]]},
           "gates": gates,
           "record_only": {"by_game_count": counts, "by_level": levels, "Q41S_only_SET_own_gamma": q41s_rec},
           "q09_map_sha256": map_sha, "peak_memory_record": E23.peak_memory()}
    E23.write_json(out_path, rep)
    log(f"-> {out_path}: {status} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
