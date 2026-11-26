"""Step 3 = analysis (b): one-to-one structure.

Same candidates as the official-shaped pools; the owners of the 400 questions vary:
  o2o  every candidate owns exactly one question (as our local validation; official unknown)
  u390 / u375 / u350 / u300  only m distinct candidates own questions, the rest drawn again from them
  wr   owners drawn with replacement from the 400 candidates (about 253 distinct)
For every pool the score matrix S (default: the opening-only Q22-equivalent) is re-ranked by:
  base        S as is
  hung_raw    Hungarian on S (one candidate per question, all distinct), ranks 2-5 by S
  hung_z      Hungarian on row-z-scored S, ranks 2-5 by S
  sink_t      balanced Sinkhorn on exp(z(S)/t), full ranking by the plan
  hsink_t     Hungarian top-1, ranks 2-5 by the Sinkhorn plan
  usink_t_l   unbalanced Sinkhorn (column marginal penalty lam), full ranking by the plan
  csls_K      CSLS hubness correction
Detector statistics are label-free functions of S (c1h_eval.detector_stats + forced cost + bootstrap
stability). Tuning seeds 100-109 (o2o, wr), evaluation seeds 0-19 (all scenarios).

    python s3_assign.py [--term base|<fusion json name>] [--tag TAG]
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TUNE_SEEDS = list(range(100, 110))
EVAL_SEEDS = list(range(20))
SCEN_EVAL = ["o2o", "u390", "u375", "u350", "u300", "wr"]
SCEN_TUNE = ["o2o", "wr"]
TAUS = [0.1, 0.25, 0.5, 1.0, 2.0]
LAMS = [0.03, 0.1, 0.3, 1.0, 3.0]
CSLS_K = [1, 2, 3, 5, 10, 20]
ISOFT_B = [0.5, 1.0, 2.0, 4.0]
N_BOOT = 5


def score_matrix(term, rec, cache, G_cache):
    import c1h_eval as E
    import c1h_lib as L
    if term == "base":
        if "base" not in G_cache:
            G_cache["base"] = L.load_family("base")
        return E.cos_term(G_cache["base"], rec)
    import c1h_fused as FU
    return FU.fused_matrix(term, rec, cache, G_cache)


def one(args):
    seed, scen, term = args
    sys.path.insert(0, HERE)
    import c1h_eval as E
    import c1h_lib as L
    t = time.time()
    c = L.load_cache()
    G_cache = {}
    rec = L.make_pool(c, seed, scen)
    S = score_matrix(term, rec, c, G_cache)
    own = rec["owner"]
    Z = L.zrows(S)
    out = {"seed": seed, "scen": scen, "n_distinct": int(len(set(own.tolist()))), "n_repairs": rec["n_repairs"],
           "k": rec["k"].tolist()}
    res = {}

    def ev(name, F):
        r = L.ranks(F, own)
        res[name] = {"score": float(L.pts(r).mean()), "top1": float((r == 1).mean()),
                     "top5": float((r <= 5).mean()),
                     "per_k": {str(k): float(L.pts(r[rec["k"] == k]).mean()) for k in L.K_VALUES}}

    ev("base", S)
    a_raw = E.hungarian_top1(S)
    ev("hung_raw", E.with_top1(S, a_raw))
    a_z = E.hungarian_top1(Z)
    ev("hung_z", E.with_top1(S, a_z))
    for t_ in TAUS:
        P = E.sinkhorn(Z, t_, iters=100)
        ev("sink_%g" % t_, P)
        ev("hsink_%g" % t_, E.with_top1(P, a_z))
        for lam in LAMS:
            ev("usink_%g_%g" % (t_, lam), E.sinkhorn(Z, t_, iters=100, lam=lam))
    for K in CSLS_K:
        C = E.csls(S, K)
        ev("csls_%d" % K, C)
        ev("hcsls_%d" % K, E.with_top1(C, a_z))           # Hungarian top-1, ranks 2-5 by CSLS
    for b in ISOFT_B:                                      # inverted softmax (column-normalised)
        ev("isoft_%g" % b, Z * b - np.logaddexp.reduce(Z * b, axis=0, keepdims=True))
    # detector statistics (label-free)
    det = E.detector_stats(S)
    top = Z.argmax(1)
    det["forced_cost_z"] = float(np.mean(Z[np.arange(len(top)), top] - Z[np.arange(len(top)), a_z]))
    det["changed_by_assignment"] = float(np.mean(a_z != top))
    # bootstrap stability of the assignment: resample each question's games with replacement
    if term == "base":
        rng = random.Random("boot|%s|%d" % (scen, seed))
        same = []
        for b in range(N_BOOT):
            rb = dict(rec)
            rb["q_rows"] = [[rng.choice(q) for _ in q] for q in rec["q_rows"]]
            Sb = E.cos_term(G_cache["base"], rb)
            same.append(float(np.mean(E.hungarian_top1(L.zrows(Sb)) == a_z)))
        det["boot_assign_same"] = float(np.mean(same))
    # column view: per candidate, best z among questions; share of candidates whose best z < 3
    colmax = Z.max(0)
    det["colmax_z_median"] = float(np.median(colmax))
    det["share_cand_colmax_z_lt3"] = float(np.mean(colmax < 3.0))
    out["det"] = det
    out["res"] = res
    out["seconds"] = round(time.time() - t, 1)
    return out


def agg(rows, method, metric="score"):
    import c1h_lib as L
    return [round(v, 4) for v in L.mean_ci([r["res"][method][metric] for r in rows])]


def agg_diff(rows, method, base="base"):
    import c1h_lib as L
    return [round(v, 4) for v in L.mean_ci([r["res"][method]["score"] - r["res"][base]["score"] for r in rows])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default="base")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    import c1h_lib as L
    t0 = time.time()
    jobs = [(s, sc, a.term) for s in TUNE_SEEDS for sc in SCEN_TUNE] + \
           [(s, sc, a.term) for s in EVAL_SEEDS for sc in SCEN_EVAL]
    if a.quick:
        jobs = [(0, "o2o", a.term), (0, "wr", a.term)]
    with Pool(int(os.environ.get('C1H_WORKERS', '8'))) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    with open(os.path.join(L.SCRATCH, "s3_rows_%s.json" % a.tag), "w", encoding="utf-8") as f:
        json.dump(rows, f)
    tune = {sc: [r for r in rows if r["seed"] in TUNE_SEEDS and r["scen"] == sc] for sc in SCEN_TUNE}
    ev = {sc: [r for r in rows if r["seed"] in EVAL_SEEDS and r["scen"] == sc] for sc in SCEN_EVAL}
    methods = list(rows[0]["res"])
    tune_tab = {m: {sc: round(float(np.mean([r["res"][m]["score"] - r["res"]["base"]["score"] for r in tune[sc]])), 4)
                    for sc in SCEN_TUNE} for m in methods} if tune["o2o"] else {}
    # selection on the tuning pools only
    sel = {}
    if tune_tab:
        sinks = [m for m in methods if m.startswith("sink_")]
        hs = [m for m in methods if m.startswith("hsink_")]
        us = [m for m in methods if m.startswith("usink_")]
        cs = [m for m in methods if m.startswith("csls_")]
        sel["sink"] = max(sinks, key=lambda m: tune_tab[m]["o2o"])
        sel["hsink"] = max(hs, key=lambda m: tune_tab[m]["o2o"])
        safe = [m for m in us if tune_tab[m]["wr"] >= 0.0]
        sel["usink_safe"] = max(safe, key=lambda m: tune_tab[m]["o2o"]) if safe else None
        sel["usink_best_o2o"] = max(us, key=lambda m: tune_tab[m]["o2o"])
        sel["csls"] = max(cs, key=lambda m: tune_tab[m]["o2o"] + tune_tab[m]["wr"])
        iso = [m for m in methods if m.startswith("isoft_")]
        sel["isoft"] = max(iso, key=lambda m: tune_tab[m]["o2o"] + tune_tab[m]["wr"])
        hc = [m for m in methods if m.startswith("hcsls_")]
        sel["hcsls"] = max(hc, key=lambda m: tune_tab[m]["o2o"])
    eval_tab = {}
    for m in methods:
        eval_tab[m] = {sc: {"score": agg(ev[sc], m), "diff_vs_base": agg_diff(ev[sc], m),
                            "top1": agg(ev[sc], m, "top1")} for sc in SCEN_EVAL if ev[sc]}
    det_keys = list(rows[0]["det"])
    det_tab = {sc: {k: {"min": round(float(np.min([r["det"][k] for r in ev[sc]])), 4),
                        "median": round(float(np.median([r["det"][k] for r in ev[sc]])), 4),
                        "max": round(float(np.max([r["det"][k] for r in ev[sc]])), 4)} for k in det_keys}
               for sc in SCEN_EVAL if ev[sc]}
    out = {"term": a.term, "tune_seeds": TUNE_SEEDS, "eval_seeds": EVAL_SEEDS,
           "n_distinct_owners": {sc: [r["n_distinct"] for r in ev[sc]] for sc in SCEN_EVAL if ev[sc]},
           "n_repairs": {sc: [r["n_repairs"] for r in ev[sc]] for sc in SCEN_EVAL if ev[sc]},
           "tune_mean_gain": tune_tab, "selected_on_tune": sel, "eval": eval_tab, "detector": det_tab,
           "total_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, "s3_assign_%s.json" % a.tag), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    keym = ["base", "hung_raw", "hung_z"] + [v for v in sel.values() if v]
    for m in keym:
        print(m, {sc: eval_tab[m][sc]["diff_vs_base"] for sc in eval_tab[m]})
    print("base", {sc: eval_tab["base"][sc]["score"] for sc in eval_tab["base"]})
    print(json.dumps(sel))
    print(json.dumps(det_tab, indent=0)[:4000])
    print("total %.0fs" % (time.time() - t0))
