"""prereg53 exploration (competition 2): stage 0b (PLAN.md 2) and the inner-only judgement (PLAN.md 3) of C53 = Q25's
OOF probabilities through the decision-level calibration of p53_lib (10 log-prior offsets + 1 temperature) before
rule B. CPU only (.venv; numpy / scipy).

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg53/explore
    $S/p53_eval.py stage0b          -> $S/stage0b.json (once; code committed and clean; no inner data)
    $S/p53_eval.py eval             -> $S/eval_report.json (once; stage0b.json committed and passed)
    $S/p53_eval.py stage0b|eval --smoke -> data/prereg53/smoke/ (noise in place of the calibrated probabilities:
                                           pipeline, time and memory only, never a result)

Q25 is rebuilt exactly as prereg41 did (tools/p09_rank_fuse.Inner, p23_eval.s19_term with the frozen S25 models,
Q09's p1e_map P1TO). Never the validation answers, never validation features, never tools/score.py, never a
competition-1 file.
"""

import argparse
import ctypes
import glob
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P23 = os.path.join(REPO, "results", "prereg23", "explore")
P40 = os.path.join(REPO, "results", "prereg40", "explore")
for p in (os.path.join(REPO, "tools"), P23, P40, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

THREADS = 8


def limit_threads(n=THREADS):
    """PLAN.md 6: OpenBLAS threads through its own API (no environment variable) and the CPU affinity."""
    rec = {"blas": [], "affinity": None}
    base = os.path.dirname(os.path.dirname(np.__file__))
    for d in ("numpy.libs", "scipy.libs"):
        for path in glob.glob(os.path.join(base, d, "*openblas*.dll")):
            lib = ctypes.CDLL(path)
            for name in ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads"):
                fn = getattr(lib, name, None)
                if fn is not None:
                    fn(ctypes.c_int(n))
                    rec["blas"].append(f"{d}/{name}({n})")
                    break
    try:
        import psutil
        pr = psutil.Process()
        cpus = sorted(pr.cpu_affinity())[:n]
        pr.cpu_affinity(cpus)
        pr.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        rec["affinity"] = cpus
    except Exception as e:  # record only
        rec["affinity_error"] = repr(e)
    return rec


THREAD_REC = limit_threads()

import p23_eval as E23  # noqa: E402
import p23_lib as K23  # noqa: E402
import p23_train as T23  # noqa: E402
import p19_lib as K19  # noqa: E402
import p40_lib as L40  # noqa: E402
import p53_lib as C  # noqa: E402

r, ft_eval, FW = E23.r, E23.ft_eval, E23.FW
rf, p3, p4 = r.rf, r.p3, r.p4
TOL = 1e-12
DATA = os.path.join(REPO, "data", "prereg53")
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED_0B = os.path.join(DATA, "stage0b_started.json")
STARTED = os.path.join(DATA, "eval_started.json")
STAGE0B = os.path.join(HERE, "stage0b.json")
REPORT = os.path.join(HERE, "eval_report.json")
P25 = os.path.join(REPO, "results", "prereg25", "explore")
S25_TRAIN_RECORD = os.path.join(P25, "train_record.json")
S25_EVAL_REPORT = os.path.join(P25, "eval_report.json")
P40_STAGE0B = os.path.join(P40, "stage0b.json")
S25_MODEL_FORMAT = "c10-prereg25 s25 model v1"
REF_Q25 = 0.7456519871546352
REF_Q25_MONO = 0.7482450052195216
OWN_FILES = ("PLAN.md", "p53_lib.py", "p53_eval.py", "test_p53.py")
IMPORTED = tuple(E23.IMPORTED) + ("results/prereg23/explore/p23_eval.py", "results/prereg23/explore/p23_lib.py",
                                  "results/prereg40/explore/p40_lib.py", "results/prereg40/explore/stage0b.json",
                                  "results/prereg25/explore/train_record.json",
                                  "results/prereg25/explore/eval_report.json", "tools/p04_rank_fuse.py",
                                  "tools/p02_rank_fuse.py")
FORMAT = "c10-prereg53 explore eval v1"
FORMAT_0B = "c10-prereg53 explore stage0b v1"
NOISE_SEED = 12345
VARIANTS = ("full", "delta", "temp")


def log(*x):
    print(*x, flush=True)


def code_state(extra=()):
    """Own files: git status must be empty. Imported files: content (CR at end of line ignored) equal to HEAD; the
    main checkout's bytes differ from this worktree's checkout only by line endings (recorded)."""
    own = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES + tuple(extra)]
    rel = sorted(set(own) | set(IMPORTED) | E23.loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty_own = E23.git("status", "--porcelain", "--", *own)
    dirty_imp = E23.git("diff", "--ignore-cr-at-eol", "--name-only", "HEAD", "--", *rel)
    untracked = E23.git("ls-files", "--others", "--exclude-standard", "--", *rel)
    dirty = "\n".join(x for x in (dirty_own, dirty_imp, untracked) if x)
    return {"head": E23.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: E23.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def once_guard(a, cs, out_path, started):
    prior = [p for p in (out_path, started) if os.path.exists(p)]
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
    return pars, seam, tr


def per_level(sc, y):
    return [float(sc[y == k].mean()) for k in range(C.L)]


def boot_diff(sc_a, sc_b, y, player_idx, n_players):
    bt = rf.PlayerBootstrap(player_idx, y, n_players)
    d = bt.stats(sc_a) - bt.stats(sc_b)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ================================================================ stage 0b (PLAN.md 2)
def cmd_stage0b(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(SMOKE_DIR, "stage0b.json") if smoke else STAGE0B
    cs = code_state()
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg53 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path, STARTED_0B)
        E23.write_json(STARTED_0B, {"format": FORMAT_0B + " started", "created": r.now_iso(), "code": cs})
    s25, sm, tr = s25_models()
    mp = FW.load_map(E23.Q09_MAP)
    prep23 = T23.ProbeOnly(L40.PROBE_PREP)
    rows = L40.Rows(T23.probe_colours(prep23))
    del prep23
    _ra, _rh, _pa, pool_h = L40.split_pools(rows)
    hold, hold_stats = K19.holdout_set(pool_h, L40.HOLD_PER, L40.HOLD_SEED)
    seam_h = {"holdout_questions_equal_s25": bool({str(n): int(len(v[1])) for n, v in hold.items()}
                                                  == tr["players"]["holdout_questions"]),
              "holdout_sampling_equal_s25": bool(hold_stats == tr["players"]["holdout_sampling"])}
    seam_h["ok"] = bool(all(seam_h.values()))
    if not (sm["ok"] and seam_h["ok"]):
        E23.write_json(out_path, {"format": FORMAT_0B, "smoke": smoke, "code": cs, "status": "stop: seam failed",
                                  "seam_m": sm, "seam_h": seam_h})
        return 1

    def feats(idx):
        Xr, EXr = rows.raw(idx)
        return L40.features32(Xr, EXr, mp)[:, :FW.X_DIM]

    lp_by_n = L40.holdout_lp(K19.set_logp_np, s25, hold, feats)
    n25, _per = L40.mean_nll(lp_by_n, hold)
    ref40 = r.read_json(P40_STAGE0B)["nll"]["S25"]
    seam_n = {"s25_holdout_nll": n25, "prereg40_stage0b_S25": ref40, "abs_diff": abs(n25 - ref40),
              "ok": bool(abs(n25 - ref40) <= 1e-9)}
    lp, y, nq, keys = [], [], [], []
    for n in sorted(lp_by_n):
        idx, yy = hold[n]
        lp.append(FW.lse_mean(lp_by_n[n]))
        y.append(np.asarray(yy))
        nq.append(np.full(len(yy), n))
        kq = rows.keys[idx]
        if not (kq == kq[:, :1]).all():
            raise SystemExit("a held-out question mixes players")
        keys.extend(kq[:, 0].tolist())
    lp, y, nq = np.concatenate(lp), np.concatenate(y).astype(np.int64), np.concatenate(nq)
    log(f"0b inputs: {len(y)} questions, {len(set(keys))} players; seam N {seam_n}")
    if not seam_n["ok"]:
        E23.write_json(out_path, {"format": FORMAT_0B, "smoke": smoke, "code": cs, "status": "stop: seam failed",
                                  "seam_m": sm, "seam_h": seam_h, "seam_n": seam_n})
        return 1
    if smoke:
        lp = rf.log_softmax(2.0 * np.random.default_rng(NOISE_SEED).standard_normal(lp.shape))
    res = C.stage0b(lp, y, keys, VARIANTS)
    pred0 = rf.rule_b_np(res["P0"])
    sc0 = rf.official_score(pred0, y)
    rb0 = C.rank_balanced_score(pred0, y)
    ukeys = sorted(set(keys))
    pidx = np.array([ukeys.index(k) for k in keys])
    out = {}
    for v in VARIANTS:
        Pc = res["cand"][v]
        if np.isnan(Pc).any():
            raise SystemExit("a held-out question without a calibrated prediction")
        pred = rf.rule_b_np(Pc)
        sc = rf.official_score(pred, y)
        rb = C.rank_balanced_score(pred, y)
        out[v] = {"rank_balanced": rb, "delta_vs_B0": rb - rb0, "nll": C.nll(Pc, y),
                  "ci95_vs_B0": boot_diff(sc, sc0, y, pidx, len(ukeys)),
                  "decisions_changed": int((pred != pred0).sum()),
                  "per_n_delta": {str(n): C.rank_balanced_score(pred[nq == n], y[nq == n])
                                  - C.rank_balanced_score(pred0[nq == n], y[nq == n]) for n in sorted(set(nq))},
                  "per_level": [{"level": r.LEVELS[k], "candidate": c, "B0": b, "diff": c - b}
                                for k, (c, b) in enumerate(zip(per_level(sc, y), per_level(sc0, y)))],
                  "pred_distribution": np.bincount(pred, minlength=C.L).tolist()}
    v = C.verdict_0b(out["full"]["delta_vs_B0"])
    folds = res["folds"]
    rec = {"format": FORMAT_0B, "smoke": smoke, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "code": cs, "earlier_run_heads": earlier, "threads": THREAD_REC,
           "smoke_note": "S25 log-probabilities replaced by noise: numbers are meaningless" if smoke else None,
           "what": "PLAN.md 2: S25's 4,000 held-out questions; B0 = 2-fold player cross-fit stack z = a lp_S25 + b; "
                   "C = B0 + decision calibration (10 offsets + temperature) fitted on the other fold's nested base "
                   "probabilities; rule B official score, rank-balanced",
           "seam_m": sm, "seam_h": seam_h, "seam_n": seam_n,
           "questions": int(len(y)), "players": len(ukeys), "fold_sizes": np.bincount(res["fold"]).tolist(),
           "B0": {"rank_balanced": rb0, "nll": C.nll(res["P0"], y), "stack_folds": res["base_stack"],
                  "pred_distribution": np.bincount(pred0, minlength=C.L).tolist()},
           "candidate_full": out["full"], "verdict": v,
           "status": ("passed: run the inner judgement (PLAN.md 3)" if v["passed"] else "not passed: stop (no inner use)"),
           "record_only": {"delta_only": out["delta"], "temperature_only": out["temp"], "folds": folds},
           "peak_memory_record": E23.peak_memory()}
    E23.write_json(out_path, rec)
    log(f"stage 0b: B0 {rb0:.5f}, C {out['full']['rank_balanced']:.5f} -> {v} -> {out_path}")
    return 0


# ================================================================ inner judgement (PLAN.md 3)
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


def nested_fit_probs(g):
    """PLAN.md 3.1 seam N: for each outer fold f, the fit questions of the other folds predicted by fusions fitted on
    the fit questions of folds not in {f, g} -> {f: (rows, P)}."""
    Xf, mf, Ef = g["fit"]
    out, recs = {}, []
    for f in range(5):
        rows = np.flatnonzero(Xf.fold != f)
        P = np.full((len(rows), C.L), np.nan)
        pos = {int(q): i for i, q in enumerate(rows)}
        for h in range(5):
            if h == f:
                continue
            tr = (Xf.fold != f) & (Xf.fold != h)
            ev = np.flatnonzero(Xf.fold == h)
            th, rec = p4.gen_fit_record(mf[tr], [E[tr] for E in Ef], Xf.n[tr], Xf.y[tr])
            z, _ = p4.gen_logits(th, mf[ev], [E[ev] for E in Ef], Xf.n[ev])
            P[[pos[int(q)] for q in ev]] = rf.softmax(z)
            recs.append({"outer": f, "inner_fold": h, "fit_questions": int(tr.sum()), "predicted": int(len(ev)),
                         "loss": rec["loss"], "theta": [float(x) for x in th]})
        out[f] = (rows, P)
    ok = bool(all(not np.isnan(P).any() and len(rows) == int((Xf.fold != f).sum()) for f, (rows, P) in out.items()))
    return out, recs, ok


def calibrated_oof(nested, Xf, base_P, base_Pm, fold_e, fold_m, variant):
    P = np.full(base_P.shape, np.nan)
    Pm = np.full(base_Pm.shape, np.nan)
    recs = []
    for f in range(5):
        rows, Pn = nested[f]
        th, rec = C.fit(Pn, Xf.y[rows], variant)
        P[fold_e == f] = C.apply(th, base_P[fold_e == f])
        Pm[fold_m == f] = C.apply(th, base_Pm[fold_m == f])
        rec["fold"] = f
        recs.append(rec)
    if np.isnan(P).any() or np.isnan(Pm).any():
        raise SystemExit("a question without a calibrated OOF prediction")
    return P, Pm, recs


def scored(inner, P, Pm):
    m, sc = rf.metrics(rf.rule_b_np(P), inner.Xe.y, inner.Xe.n, inner.Xe.player)
    mrec, msc = p3.mono_score_record(rf.rule_b_np(Pm), inner.Xm, inner.ms, inner.mboot)
    return {"P": P, "Pm": Pm, "sc": sc, "msc": msc, "m": m, "mrec": mrec, "rb": m["rank_balanced"],
            "mono_rb": mrec["rank_balanced"]}


def short(inner, res, base):
    return {"inner_rank_balanced": res["rb"], "inner_question_mean": res["m"]["question_mean"],
            "single_colour_rank_balanced": res["mono_rb"],
            "single_colour_by_colour": {c: res["mrec"]["by_colour"][c]["rank_balanced"] for c in res["mrec"]["by_colour"]},
            "pred_distribution": res["m"]["pred_distribution"],
            "decisions_changed_eval": int((rf.rule_b_np(res["P"]) != rf.rule_b_np(base["P"])).sum()),
            "vs": inner.compare(res, base)}


def cmd_eval(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(SMOKE_DIR, "eval_report.json") if smoke else REPORT
    cs = code_state(() if smoke else ("stage0b.json",))
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg53 files or the imported tools are not committed and clean: {cs['dirty']}")
        s0 = r.read_json(STAGE0B)
        if s0.get("smoke") or not (s0.get("verdict") or {}).get("passed"):
            raise SystemExit("stage0b.json is not a passed stage 0b: PLAN.md 2 says stop")
        earlier = once_guard(a, cs, out_path, STARTED)
    base_rec = {"format": FORMAT, "smoke": smoke, "code": cs, "threads": THREAD_REC}
    s25, sm, _tr = s25_models()
    prep = ft_eval.InnerPrep(E23.guard_path(E23.INNER_PREP))
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
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]]}
    del q09
    terms = terms9 + ["SET"]
    E_s25, _lp25, _h25 = E23.s19_term(prep, inner, mp, s25)
    del _lp25, _h25
    Emap = {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25}
    q25 = inner.run(terms, Emap)
    seam["Q25"] = [q25["rb"], REF_Q25]
    seam["Q25_mono"] = [q25["mono_rb"], REF_Q25_MONO]
    rep25 = r.read_json(S25_EVAL_REPORT)
    seam_ref_bound = bool(rep25["candidate"]["inner_rank_balanced"] == REF_Q25
                          and rep25["candidate"]["single_colour_rank_balanced"] == REF_Q25_MONO)
    seam1 = bool(seam_ref_bound and all(abs(x - y) <= TOL for x, y in seam.values()))
    seam1_bitwise = all(x == y for x, y in seam.values())
    log(f"seam 1 {seam1} (bitwise {seam1_bitwise}) {seam}; seam M {sm['ok']} ({time.time() - t0:.0f}s, "
        f"{E23.peak_memory()})")
    g = gather(inner, terms, Emap)
    Xf = g["fit"][0]
    tn = time.time()
    nested, nested_recs, seam_n_ok = nested_fit_probs(g)
    del g
    seam_n = {"ok": seam_n_ok, "fits": len(nested_recs), "seconds": round(time.time() - tn, 1),
              "fit_questions_total": int(len(Xf.y)),
              "per_outer": {str(f): int(len(rows)) for f, (rows, _P) in nested.items()}}
    # seam I: theta = 0 is the identity on Q25's OOF probabilities
    th0 = np.zeros(C.N_PARAMS)
    Pi, Pmi = C.apply(th0, q25["P"]), C.apply(th0, q25["Pm"])
    seam_i = {"eval_max_abs": float(np.max(np.abs(Pi - q25["P"]))), "mono_max_abs": float(np.max(np.abs(Pmi - q25["Pm"]))),
              "eval_decisions_equal": bool(np.array_equal(rf.rule_b_np(Pi), rf.rule_b_np(q25["P"]))),
              "mono_decisions_equal": bool(np.array_equal(rf.rule_b_np(Pmi), rf.rule_b_np(q25["Pm"])))}
    seam_i["ok"] = bool(seam_i["eval_max_abs"] <= TOL and seam_i["mono_max_abs"] <= TOL
                        and seam_i["eval_decisions_equal"] and seam_i["mono_decisions_equal"])
    del Pi, Pmi
    log(f"seam N {seam_n}; seam I {seam_i} ({time.time() - t0:.0f}s, {E23.peak_memory()})")
    if not (seam1 and sm["ok"] and seam_n["ok"] and seam_i["ok"]):
        E23.write_json(out_path, {**base_rec, "status": "stop: seam failed", "seam": seam,
                                  "seam_ref_bound": seam_ref_bound, "seam_m": sm, "seam_n": seam_n,
                                  "seam_i": seam_i})
        return 1
    # ---- the judged part starts here (after every seam, before any candidate number)
    base_P, base_Pm = q25["P"], q25["Pm"]
    if smoke:
        rng = np.random.default_rng(NOISE_SEED)
        nested = {f: (rows, rf.softmax(2.0 * rng.standard_normal(P.shape))) for f, (rows, P) in nested.items()}
        base_P = rf.softmax(2.0 * rng.standard_normal(base_P.shape))
        base_Pm = rf.softmax(2.0 * rng.standard_normal(base_Pm.shape))
        q25 = scored(inner, base_P, base_Pm)
    else:
        E23.write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                 "rerun_reason": a.reason if a.rerun else None})
    tc = time.time()
    res, fold_recs = {}, {}
    for v in VARIANTS:
        P, Pm, recs = calibrated_oof(nested, Xf, base_P, base_Pm, inner.Xe.fold, inner.Xm.fold, v)
        res[v] = scored(inner, P, Pm)
        fold_recs[v] = recs
        log(f"{v}: {res[v]['rb']!r} vs Q25 {q25['rb']!r} ({time.time() - tc:.0f}s)")
    cand = res["full"]
    cmp_ = inner.compare(cand, q25)
    gates = K23.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                      cmp_["mono_by_colour"])
    log(f"C53 {cand['rb']!r} vs Q25 {q25['rb']!r}: {json.dumps(gates)}")
    counts = E23.by_count(inner, cand, q25)
    levels = E23.by_level(inner, cand, q25)
    status = "gates passed (worth a PREREG_53; not adoption)" if gates["passed"] else "gates not passed: stop"
    rep = {**base_rec, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("calibrated probabilities replaced by noise: numbers are meaningless" if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md 3: inner-only judgement of C53 = Q25's OOF probabilities through a decision calibration "
                   "(10 log-prior offsets + 1 temperature, fitted on nested OOF fit-question probabilities of the "
                   "other folds) before rule B; no validation scoring",
           "status": status, "seam": seam, "seam_ref_bound": seam_ref_bound, "seam1_bitwise": seam1_bitwise,
           "seam_m": sm, "seam_n": seam_n, "seam_i": seam_i,
           "candidate": {"name": "C53", **short(inner, cand, q25), "folds": fold_recs["full"]},
           "baseline": {"name": "Q25", "terms": terms, "inner_rank_balanced": q25["rb"],
                        "single_colour_rank_balanced": q25["mono_rb"], "pred_distribution": q25["m"]["pred_distribution"]},
           "gates": gates,
           "record_only": {"by_game_count": counts, "by_level": levels,
                           "delta_only": {**short(inner, res["delta"], q25), "folds": fold_recs["delta"]},
                           "temperature_only": {**short(inner, res["temp"], q25), "folds": fold_recs["temp"]},
                           "nested_fusions": nested_recs},
           "stage0b_sha256_lf": None if smoke else E23.sha_lf(STAGE0B),
           "q09_map_sha256": map_sha, "peak_memory_record": E23.peak_memory()}
    E23.write_json(out_path, rep)
    log(f"-> {out_path}: {status} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage0b", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"stage0b": cmd_stage0b, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
