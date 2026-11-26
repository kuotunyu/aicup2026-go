"""Step 10 (continuation run): reconcile with results/prereg06/explore/prior_art/sim_c1_assignment.py.

The sibling's dual softmax is softmax_row(S / tau) * softmax_col(S / tau) on the RAW opening cosine
S (tau = 0.02, 0.05 picked without held-out seeds); s8_switch.py uses the same family on row
z-scores (rank by b Z - logsumexp_col(b Z)). Here the sibling's exact formula, on the base term,
with tau chosen on the selection seeds 100-109 + 200-219 and reported on the evaluation seeds 0-19,
scenarios o2o, d80 (= the sibling's dup80 structure) and wr. CSLS10 and Hungarian on raw S are
repeated for the same pools (they match the sibling per seed on o2o seeds 0-4, see report).
Base term only (cheap)."""
import sys
sys.dont_write_bytecode = True
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SEL = list(range(100, 110)) + list(range(200, 220))
EVAL = list(range(20))
SCEN = ["o2o", "d80", "wr"]
TAUS = [0.01, 0.02, 0.03, 0.05, 0.1]


def one(seed):
    sys.path.insert(0, HERE)
    import c1h_eval as E
    import c1h_lib as L
    c = L.load_cache()
    G = L.load_family("base")
    out = []
    for sc in SCEN:
        rec = L.make_pool(c, seed, sc)
        S = E.cos_term(G, rec)
        res = {}

        def ev(name, M):
            r = L.ranks(M, rec["owner"])
            res[name] = float(L.pts(r).mean())
        ev("base", S)
        ev("hung_raw", E.with_top1(S, E.hungarian_top1(S)))
        ev("csls10", E.csls(S, 10))
        for t in TAUS:
            Z = S / t
            ev("dsm_raw_%g" % t, (Z - logsumexp(Z, axis=1, keepdims=True)) + (Z - logsumexp(Z, axis=0, keepdims=True)))
        out.append({"seed": seed, "scen": sc, "res": res})
    return out


if __name__ == "__main__":
    import c1h_lib as L
    t0 = time.time()
    with Pool(int(os.environ.get("C1H_WORKERS", "8"))) as pool:
        rows = [x for r in pool.map(one, SEL + EVAL, chunksize=1) for x in r]
    names = list(rows[0]["res"])

    def diffs(seeds, sc, m):
        rr = [r for r in rows if r["seed"] in seeds and r["scen"] == sc]
        return np.array([r["res"][m] - r["res"]["base"] for r in rr])
    sel_tab = {m: {sc: round(float(diffs(SEL, sc, m).mean()), 4) for sc in SCEN} for m in names}
    dsm = ["dsm_raw_%g" % t for t in TAUS]
    safe = [m for m in dsm if min(sel_tab[m].values()) >= 0]
    chosen = max(safe or dsm, key=lambda m: np.mean(list(sel_tab[m].values())))
    ev = {m: {sc: [round(v, 4) for v in L.mean_ci(diffs(EVAL, sc, m))] for sc in SCEN} for m in names}
    sib5 = {m: {sc: round(float(np.mean([r["res"][m] - r["res"]["base"] for r in rows
                                         if r["seed"] in range(5) and r["scen"] == sc])), 4) for sc in ("o2o", "d80")}
            for m in names}
    out = {"what": __doc__, "selection_mean_diff_vs_base": sel_tab, "dsm_tau_chosen_on_selection": chosen,
           "eval_diff_vs_base": ev, "eval_seeds_0_4_mean_diff (sibling's seeds)": sib5,
           "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, "s10_reconcile_dsm.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"chosen": chosen, "sel": sel_tab, "eval": ev, "sib5": sib5}, indent=1))
