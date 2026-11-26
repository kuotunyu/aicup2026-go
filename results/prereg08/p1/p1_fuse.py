"""prereg08 P1, step 2 (CPU): map the probe features to the 10 levels, fuse on top of Q02H out of fold, gate.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg08/p1/p1_fuse.py
        -> results/prereg08/p1/p1_fuse_report.json
    options: --p1-dir (chunk folder, default data/prereg08/p1; a synthetic folder for the mechanics test),
             --sets (default <p1-dir>/sets.json), --out, --allow-partial-mono (default on: report G4 as incomplete)

Everything follows results/prereg08/p1/p1_plan.json (read here; its primary variant decides the gate):
1 Baseline Q02H exactly as tools/p07_rank_fuse.py selfcheck (= the PREREG_07 pilot): pilot_fuse.fit_mapping on the
  pilot fitmap curves -> lp_HS for the inner bank / mono games, u1_lib.run_gen(d, OP O4 SWA SWB HS). Seam: Q02 equals
  u1_oof.npz bit for bit and Q02H's inner eval rank-balanced score equals 0.6242752597994481 within 1e-12.
2 P1 mappings (variants T, O, TO) on the probe games only (fit players), multinomial LR, lambda by player-grouped
  5-fold CV log-loss; lp_P1x for every scored inner game (0 = neutral for unusable / unscored games).
3 Fusion: u1_lib.run_gen with the extra term(s); paired PlayerBootstrap on the inner eval questions; single-colour
  questions as the pilot (covered questions; G4 needs all 120,544).
Pinned files (hs_lib, pilot_fuse, u1_lib) are imported only after their SHA-256 check; data/local is guarded (hs_lib,
u1_lib). No validation file, no score.py, nothing written outside --out.
"""

import argparse
import glob
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402

import p1_lib as P  # noqa: E402  (pin check of hs_lib, data/local guard)

H = P.H
REPO = P.REPO
PLAN = os.path.join(HERE, "p1_plan.json")
P1_DIR = os.path.join(REPO, "data", "prereg08", "p1")
OUT = os.path.join(HERE, "p1_fuse_report.json")
PILOT_DATA = os.path.join(REPO, "data", "prereg07", "pilot")
PILOT_PLAN_SHA = "8494e9afd79c06249bd81545263b86caef7b9e77f3cd631a1e354368cfe33946"
PILOT_Q02H = 0.6242752597994481
TOL = 1e-12
GATE = 0.010
L = 10
Q02_TERMS = ["OP", "O4", "SWA", "SWB"]
Q02H_TERMS = Q02_TERMS + ["HS"]
MONO_QUESTIONS = 120544
COMBOS = {"Q02H": Q02H_TERMS,
          "Q02H+P1TO": Q02H_TERMS + ["P1TO"],
          "Q02H+P1T": Q02H_TERMS + ["P1T"],
          "Q02H+P1T+P1O": Q02H_TERMS + ["P1T", "P1O"],
          "Q02+P1TO": Q02_TERMS + ["P1TO"],
          "R20+P1TO": ["OP", "P1TO"]}


def import_pinned():
    P.check_pin("results/prereg07/explore/pilot_humansl/pilot_fuse.py")
    P.check_pin("results/prereg05/explore/u1/u1_lib.py")
    if P.U1_DIR not in sys.path:
        sys.path.insert(0, P.U1_DIR)
    import pilot_fuse as PF
    import u1_lib as U
    return PF, U


# ================================================================ chunks
def load_p1_set(name, p1_dir):
    files = sorted(f for f in glob.glob(os.path.join(p1_dir, name, "chunk_*.npz")) if ".tmp" not in f)
    if not files:
        return None
    ref, means, counts, failed, meta = [], [], [], [], set()
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            if str(z["format"]) != P.CHUNK_FORMAT or str(z["set"]) != name:
                raise P.P1Error(f"{f}: format / set differ")
            if str(z["meta_sha256"]) != P.neutral_table_sha():
                raise P.P1Error(f"{f}: neutral metadata differs from p1_lib's")
            ref.append(z["ref"])
            means.append(z["means"].astype(np.float64))
            counts.append(z["counts"])
            failed.append(z["failed"])
            meta.add((str(z["precision"]), str(z["device"]), str(z["ckpt_sha256"])))
    out = {"files": len(files), "ref": np.concatenate(ref), "means": np.concatenate(means),
           "counts": np.concatenate(counts), "failed": np.concatenate(failed), "meta": sorted(meta)}
    if len(np.unique(out["ref"])) != len(out["ref"]):
        raise P.P1Error(f"{name}: duplicate refs across chunks")
    return out


def variant_X(st, variant):
    """-> X [G, d] (raw, opponent-missing rows NaN for later mean fill), usable [G] bool."""
    usable = (~st["failed"]) & (st["counts"][:, 0] > 0)
    T = st["means"][:, 0, :].copy()
    O = st["means"][:, 1, :].copy()
    O[st["counts"][:, 1] == 0] = np.nan
    if variant == "T":
        X = T
    elif variant == "O":
        X = O
    elif variant == "TO":
        X = np.concatenate([T, O], axis=1)
    else:
        raise P.P1Error(f"variant {variant!r}")
    return X, usable


# ================================================================ mapping: multinomial LR, player-grouped CV
def mlr_fit(PF, X, y, lam, x0=None, maxiter=2000):
    d = X.shape[1]
    Y1 = np.eye(L)[y]
    x0 = np.zeros(d * L + L) if x0 is None else x0
    r = optimize.minimize(PF.mlr_obj, x0, args=(X, Y1, lam), jac=True, method="L-BFGS-B",
                          options={"maxiter": maxiter})
    return {"W": r.x[:d * L].reshape(d, L), "b": r.x[d * L:], "loss": float(r.fun), "success": bool(r.success),
            "nit": int(r.nit), "theta": r.x}


def logp_of(m, Xs):
    return H.log_softmax_np(Xs @ m["W"] + m["b"])


def standardise(X, mu=None, sd=None):
    if mu is None:
        mu = np.nanmean(X, 0)
        sd = np.nanstd(X, 0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
    Xs = (X - mu) / sd
    Xs[~np.isfinite(Xs)] = 0.0          # missing opponent side -> probe mean
    return Xs, mu, sd


def player_folds(players, seed=0, k=5):
    uniq = np.array(sorted(set(players)))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {p: int(i % k) for i, p in zip(range(len(uniq)), uniq[perm])}
    return np.array([fold_of[p] for p in players])


def fit_mapping_grouped(PF, U, X, y, players, lambdas, maxiter=2000, log=print):
    Xs, mu, sd = standardise(X)
    fold = player_folds(players)
    lams = sorted(lambdas, reverse=True)
    cv = {str(l_): {"ll": [], "acc": [], "sc": [], "nit": []} for l_ in lams}
    for f in range(5):
        tr, te = fold != f, fold == f
        x0 = None
        for lam in lams:
            t = time.time()
            m = mlr_fit(PF, Xs[tr], y[tr], lam, x0=x0, maxiter=maxiter)
            x0 = m["theta"]
            lp = logp_of(m, Xs[te])
            c = cv[str(lam)]
            c["ll"].append(-lp[np.arange(len(lp)), y[te]])
            c["acc"].append(np.argmax(lp, 1) == y[te])
            c["sc"].append(U.score(U.rule_b_np(np.exp(lp)), y[te]))
            c["nit"].append(m["nit"])
            log(f"  fold {f} lambda {lam}: CV log-loss {c['ll'][-1].mean():.4f}, nit {m['nit']}, "
                f"{time.time() - t:.0f}s")
    rec = {}
    for lam in lams:
        c = cv[str(lam)]
        rec[str(lam)] = {"logloss": float(np.concatenate(c["ll"]).mean()), "top1": float(np.concatenate(c["acc"]).mean()),
                         "rule_b_score_per_game": float(np.concatenate(c["sc"]).mean()), "nit": c["nit"]}
    best = min(lams, key=lambda l_: rec[str(l_)]["logloss"])
    m = mlr_fit(PF, Xs, y, best, maxiter=maxiter)
    return {"mu": mu, "sd": sd, "model": m, "lambda": best, "cv": rec,
            "logp": lambda Xn: logp_of(m, standardise(Xn, mu, sd)[0])}


# ================================================================ baseline HS (= tools/p07_rank_fuse.py selfcheck)
def hs_baseline(PF, U, d):
    if P.sha_raw(os.path.join(PILOT_DATA, "plan.json")) != PILOT_PLAN_SHA:
        raise P.P1Error("data/prereg07/pilot/plan.json differs from the pinned pilot plan")
    with open(os.path.join(PILOT_DATA, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    N = len(plan["profiles"])
    sets = {s: PF.load_set(s, N, PILOT_DATA) for s in ("fitmap", "bank", "mono")}
    fm = sets["fitmap"]
    lv_of = {lv: i for i, lv in enumerate(U.LEVELS)}
    yfit = np.array([lv_of[plan["sets"]["fitmap"]["games"][j]["rank"]] for j in fm["ref"]], dtype=np.int64)
    x, ok = PF.curves(fm, 0)
    mp = PF.fit_mapping(x[ok], yfit[ok], seed=0)
    n_inner = d["inner"]["keys"]["n"]
    lp = np.zeros((n_inner, L))
    covered = np.zeros(n_inner, dtype=bool)
    for name in ("bank", "mono"):
        st = sets[name]
        xx, okk = PF.curves(st, 0)
        lp[st["ref"][okk]] = mp["logp"](xx[okk])
        covered[st["ref"][okk]] = True
    return lp, covered, {"lambda": mp["lambda"], "cv_logloss": mp["cv"][str(mp["lambda"])]["logloss"],
                         "refit_loss": mp["model"]["loss"]}


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1-dir", default=P1_DIR)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--maxiter", type=int, default=None, help="tests only (default: the plan's 2000)")
    ap.add_argument("--synthetic", action="store_true", help="the chunk folder is synthetic (mechanics test)")
    a = ap.parse_args(argv)
    t0 = time.time()
    with open(a.plan, encoding="utf-8") as f:
        plan = json.load(f)
    primary = plan["fixed_before_results"]["primary_combination"]
    if primary not in COMBOS:
        raise P.P1Error(f"plan primary {primary!r} not in {list(COMBOS)}")
    sets_path = a.sets or os.path.join(a.p1_dir, "sets.json")
    sets_sha = P.sha_raw(sets_path)
    if not a.synthetic and sets_sha != plan["sets"]["sets_json_sha256"]:
        raise P.P1Error(f"{sets_path}: SHA-256 {sets_sha} != plan {plan['sets']['sets_json_sha256']}")
    with open(sets_path, encoding="utf-8") as f:
        sets = json.load(f)
    maxiter = a.maxiter or 2000
    lambdas = plan["mapping"]["lambdas"]
    PF, U = import_pinned()
    rep = {"format": "c10-prereg08 p1 fuse report v1", "what": plan["what"], "created_utc": H.utc_now(),
           "plan_sha256": P.sha_lf(a.plan), "primary": primary, "p1_dir": os.path.relpath(a.p1_dir, REPO),
           "synthetic": bool(a.synthetic), "sets_sha256": sets_sha, "code_sha256": P.code_shas()}
    st = {k: load_p1_set(k, a.p1_dir) for k in ("probe", "bank", "mono")}
    if st["probe"] is None or st["bank"] is None:
        raise P.P1Error("probe and bank must be extracted first")
    for k in ("probe", "bank"):
        if len(st[k]["ref"]) != len(sets[k]["games"] if k == "probe" else sets[k]["rows"]):
            raise P.P1Error(f"{k}: {len(st[k]['ref'])} games extracted, sets.json has more (run it to the end)")
    rep["sets"] = {k: (None if v is None else {"games": int(len(v["ref"])), "chunks": v["files"],
                                               "failed": int(v["failed"].sum()), "meta": v["meta"]})
                   for k, v in st.items()}
    # ---- inputs and baseline
    d = U.load_inputs(verbose=False)
    inner = d["inner"]
    n_inner = inner["keys"]["n"]
    lp_hs, hs_cov, hs_rec = hs_baseline(PF, U, d)
    d["lp"]["HS"] = lp_hs
    rep["hs_mapping"] = hs_rec
    # ---- P1 mappings on the probe games
    pr = st["probe"]
    lv_of = {lv: i for i, lv in enumerate(U.LEVELS)}
    games = sets["probe"]["games"]
    y_all = np.array([lv_of[games[j]["rank"]] for j in pr["ref"]], dtype=np.int64)
    pl_all = np.array([games[j]["rank"] + ":" + games[j]["player_id"] for j in pr["ref"]])
    maps, rep["mappings"] = {}, {}
    for v in ("T", "O", "TO"):
        X, usable = variant_X(pr, v)
        if v == "O":
            usable = usable & (pr["counts"][:, 1] > 0)
        print(f"mapping {v}: {int(usable.sum())} probe games, {X.shape[1]} features", flush=True)
        mp = fit_mapping_grouped(PF, U, X[usable], y_all[usable], pl_all[usable], lambdas, maxiter=maxiter)
        maps[v] = mp
        rep["mappings"][v] = {"games": int(usable.sum()), "players": int(len(set(pl_all[usable]))),
                              "features": int(X.shape[1]), "lambda": mp["lambda"], "cv": mp["cv"],
                              "refit_loss": mp["model"]["loss"], "refit_success": mp["model"]["success"],
                              "refit_nit": mp["model"]["nit"]}
        print(f"mapping {v}: lambda {mp['lambda']}, CV {mp['cv'][str(mp['lambda'])]}", flush=True)
    # ---- inner lp
    covered = np.zeros(n_inner, dtype=bool)
    lps = {f"P1{v}": np.zeros((n_inner, L)) for v in maps}
    for name in ("bank", "mono"):
        s = st[name]
        if s is None:
            continue
        for v, mp in maps.items():
            X, usable = variant_X(s, v)
            lps[f"P1{v}"][s["ref"][usable]] = mp["logp"](X[usable])
        covered[s["ref"][(~s["failed"]) & (s["counts"][:, 0] > 0)]] = True
    d["lp"].update(lps)
    Xe, Xm, ms = inner["X_eval"], inner["X_mono"], inner["ms"]
    eval_cov = np.array([covered[np.asarray(g)].all() for g in inner["eval_groups"]])
    if not eval_cov.all():
        raise P.P1Error(f"{int((~eval_cov).sum())} inner eval questions have an unscored P1 game")
    mono_cov = np.array([covered[np.asarray(g)].all() and hs_cov[np.asarray(g)].all() for g in ms["groups"]])
    rep["coverage"] = {"inner_games_scored": int(covered.sum()), "eval_questions": int(len(eval_cov)),
                       "mono_questions_covered": int(mono_cov.sum()), "mono_questions": int(len(mono_cov))}
    # ---- fusion
    saved = U.load_saved()
    res = {}
    for cid, terms in COMBOS.items():
        t = time.time()
        Pe, Pm, th = U.run_gen(d, terms)
        res[cid] = {"P": Pe, "Pm": Pm, "th": th, "terms": terms, "sc": U.score(U.rule_b_np(Pe), Xe.y),
                    "msc": U.score(U.rule_b_np(Pm), Xm.y)}
        print(f"{cid}: inner eval rank-balanced {U.rank_balanced(res[cid]['sc'], Xe.y):.4f} ({time.time() - t:.0f}s)",
              flush=True)
    P_q, Pm_q, _ = U.run_gen(d, Q02_TERMS)
    rb_h = U.rank_balanced(res["Q02H"]["sc"], Xe.y)
    seam = {"Q02_eval_P_equal_u1": bool(np.array_equal(P_q, saved["eval_P_Q02"])),
            "Q02_mono_P_equal_u1": bool(np.array_equal(Pm_q, saved["mono_P_Q02"])),
            "Q02H_rank_balanced": rb_h, "Q02H_pilot": PILOT_Q02H,
            "Q02H_equal_pilot": bool(abs(rb_h - PILOT_Q02H) <= TOL)}
    rep["seam"] = seam
    if not (seam["Q02_eval_P_equal_u1"] and seam["Q02_mono_P_equal_u1"] and seam["Q02H_equal_pilot"]):
        raise P.P1Error(f"baseline does not reproduce: {seam}")
    boot = inner["boot"]
    base = res["Q02H"]
    sub = {"y": ms["y"][mono_cov], "player": ms["player"][mono_cov], "colour": ms["colour"][mono_cov],
           "k": ms["k"][mono_cov], "n_players": ms["n_players"]}
    sboot = U.PlayerBootstrap(sub["player"], sub["y"], sub["n_players"]) if mono_cov.any() else None
    rows = {}
    for cid, r in res.items():
        m = {"terms": r["terms"], "rank_balanced": U.rank_balanced(r["sc"], Xe.y), "question_mean": float(r["sc"].mean()),
             "per_level": {lv: float(r["sc"][Xe.y == i].mean()) for i, lv in enumerate(U.LEVELS)},
             "weights_by_fold": [list(map(float, th[2:2 + len(r["terms"])])) for th in r["th"]],
             "mono_rank_balanced_covered": U.rank_balanced(r["msc"], Xm.y, mono_cov) if mono_cov.any() else None}
        if cid != "Q02H":
            dd = boot.stats(r["sc"]) - boot.stats(base["sc"])
            m["diff_vs_Q02H"] = m["rank_balanced"] - rb_h
            m["diff_vs_Q02H_ci95"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            m["question_mean_diff_vs_Q02H"] = float(np.mean(r["sc"] - base["sc"]))
            if sboot is not None:
                m["mono_diff_vs_Q02H"] = U.p3.mono_diff_record(sub, sboot, r["msc"][mono_cov], base["msc"][mono_cov])
        rows[cid] = m
    rep["combinations"] = rows
    g = rows[primary]
    gate = {"primary": primary, "G1_delta": g["diff_vs_Q02H"], "G1_passed": bool(g["diff_vs_Q02H"] >= GATE - TOL),
            "G2_ci95": g["diff_vs_Q02H_ci95"], "G2_passed": bool(g["diff_vs_Q02H_ci95"][0] > 0),
            "question_mean_diff": g["question_mean_diff_vs_Q02H"]}
    mono = g.get("mono_diff_vs_Q02H")
    complete = bool(mono_cov.all() and len(mono_cov) == MONO_QUESTIONS)
    if mono is not None:
        bw = {c: (mono["by_colour"].get(c) or {}).get("rank_balanced") for c in ("B", "W")}
        gate["G4"] = {"coverage": f"{int(mono_cov.sum())}/{len(mono_cov)} questions", "complete": complete,
                      "rank_balanced_difference": mono["rank_balanced"], "ci95": mono["ci95"], "by_colour": bw,
                      "passed": bool(complete and mono["ci95"][0] > 0
                                     and all(v is not None and v > 0 for v in bw.values()))}
    else:
        gate["G4"] = {"coverage": "0 questions", "complete": False, "passed": False}
    gate["passed"] = bool(gate["G1_passed"] and gate["G2_passed"] and gate["G4"]["passed"])
    gate["status"] = ("passed" if gate["passed"] else
                      "G4 pending (mono incomplete)" if gate["G1_passed"] and gate["G2_passed"] and not complete
                      else "not passed")
    rep["gate"] = gate
    rep["seconds"] = round(time.time() - t0, 1)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=float)
    os.replace(tmp, a.out)
    print(json.dumps(gate, indent=1, default=float))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
