"""Step 8 (continuation run) = analysis (b) completed: one-to-one gain / loss, label-free detectors,
hard gate vs soft corrections, at three feature levels, thresholds chosen on selection seeds and
evaluated on held-out seeds. Replaces the unfinished s3b_switch.py of the first run (never run; its
safety-margin formula moved the threshold the wrong way when the limit fell below the o2o median).

Feature levels (score matrix F, questions x candidates):
  base     opening-only Q22-equivalent (the opening term of Q04)
  proxyA   z(base) + z(style): the Q04 stand-in of s5b (CNN replaced by the style NB term)
  v2       fused_v2_nopost.json (s5b): base + lda64 + joseki_own + style2
Scenarios (who owns the 400 questions; same 400 candidates and candidate games):
  o2o, u390, u375, u350, d80 (= prior_art 'dup80': 80 own none, 80 own two), u300, wr
Methods on F (Z = row z-score of F):
  base; hung (Hungarian on Z fixes Top-1, ranks 2-5 by F); csls<k>; hcsls<k> (Hungarian Top-1, ranks
  2-5 by CSLS k); dsm<b> (dual softmax = rank by bZ - logsumexp over questions of bZ); usink<t>_<lam>
  (unbalanced Sinkhorn on exp(Z / t), column-marginal KL weight lam).
Label-free detector statistics of F: forced_cost (mean z(top-1) - z(Hungarian choice)), changed (share
of questions whose Hungarian choice is not their top-1), n0 (candidates that are nobody's top-1),
mutual_nn; and the same statistics on a PSEUDO pool built from the candidate games only (each
candidate: 20 of its 100 games as a pseudo question with the pool's k list capped at 20, the other 80
as references; one pseudo question per candidate = one-to-one by construction; 2 folds averaged),
giving self-calibrated ratios forced_cost / forced_cost_pseudo etc. that do not depend on the scale
of the feature set.

Seeds: selection 100-109 and 200-219 (every choice: CSLS k, dual-softmax b, Sinkhorn t / lam, gate
thresholds); evaluation 0-19 (reported). One job = one seed (the LDA fit is shared by the scenarios).

    python s8_switch.py            (writes s8_rows.json to the scratchpad, s8_switch.json here)
    python s8_switch.py --analyse  (re-analyses the saved rows only)
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEL_SEEDS = list(range(100, 110)) + list(range(200, 220))
EVAL_SEEDS = list(range(20))
SCEN = ["o2o", "u390", "u375", "u350", "d80", "u300", "wr"]
LEVELS_F = ["base", "proxyA", "v2"]
CSLS_K = [1, 2, 3, 5, 10]
HCSLS_K = [3, 5, 10]
DSM_B = [0.5, 1.0, 2.0, 4.0]
USINK = [(0.25, 1.0), (0.25, 3.0), (0.5, 1.0), (0.5, 3.0), (1.0, 1.0), (1.0, 3.0)]
DET = ["forced_cost", "changed", "n0", "mutual_nn"]


def methods_and_stats(F, owner, ks, with_methods=True):
    import c1h_eval as E
    import c1h_lib as L
    Z = L.zrows(F)
    a = E.hungarian_top1(Z)
    top = Z.argmax(1)
    n, m = Z.shape
    st = {"forced_cost": float(np.mean(Z[np.arange(n), top] - Z[np.arange(n), a])),
          "changed": float(np.mean(a != top)),
          "n0": int((np.bincount(top, minlength=m) == 0).sum()),
          "mutual_nn": float(np.mean(Z.argmax(0)[top] == np.arange(n)))}
    if not with_methods:
        return None, st
    res = {}

    def ev(name, M):
        r = L.ranks(M, owner)
        p = L.pts(r)
        res[name] = [round(float(p.mean()), 6), round(float((r == 1).mean()), 6)] + \
            [round(float(p[ks == k].mean()), 6) for k in L.K_VALUES]
    ev("base", F)
    ev("hung", E.with_top1(F, a))
    C = {}
    for k in CSLS_K:
        C[k] = E.csls(F, k)
        ev("csls%d" % k, C[k])
    for k in HCSLS_K:
        ev("hcsls%d" % k, E.with_top1(C[k], a))
    for b in DSM_B:
        ev("dsm%g" % b, Z * b - np.logaddexp.reduce(Z * b, axis=0, keepdims=True))
    for t, lam in USINK:
        ev("usink%g_%g" % (t, lam), E.sinkhorn(Z, t, iters=100, lam=lam))
    return res, st


def pseudo_rec(rec, fold):
    """One-to-one pseudo pool from the candidate games only (no question game is used)."""
    ks = np.minimum(rec["k"], 20)
    cand, q = [], []
    for i, rows in enumerate(rec["cand_rows"]):
        blk = rows[fold * 20:(fold + 1) * 20]
        q.append(blk[:int(ks[i])])
        cand.append(rows[:fold * 20] + rows[(fold + 1) * 20:])
    return {"cand_rows": cand, "q_rows": q, "owner": np.arange(len(cand)), "k": ks,
            "seed": rec["seed"], "scenario": "pseudo%d" % fold}


def level_mats(rec, c, Gc, v2):
    import c1h_eval as E
    import c1h_fused as FU
    T = {}
    for t in sorted(set(["base", "style"] + v2["terms"])):
        T[t] = FU.term(t, rec, c, Gc)
    return {"base": T["base"],
            "proxyA": E.fuse([T["base"], T["style"]], [1.0, 1.0]),
            "v2": E.fuse([T[t] for t in v2["terms"]], v2["weights"])}


def one_seed(seed):
    sys.path.insert(0, HERE)
    import c1h_lib as L
    t0 = time.time()
    c = L.load_cache()
    with open(os.path.join(HERE, "fused_v2_nopost.json"), encoding="utf-8") as f:
        v2 = json.load(f)
    Gc = {}
    rows = []
    # pseudo pools (candidate games only; same for every scenario of this seed)
    rec0 = L.make_pool(c, seed, "o2o")
    pst = {lv: [] for lv in LEVELS_F}
    for fold in (0, 1):
        pr = pseudo_rec(rec0, fold)
        M = level_mats(pr, c, Gc, v2)
        for lv in LEVELS_F:
            pst[lv].append(methods_and_stats(M[lv], pr["owner"], pr["k"], with_methods=False)[1])
    pseudo = {lv: {s: float(np.mean([d[s] for d in pst[lv]])) for s in DET} for lv in LEVELS_F}
    for scen in SCEN:
        rec = rec0 if scen == "o2o" else L.make_pool(c, seed, scen)
        M = level_mats(rec, c, Gc, v2)
        row = {"seed": seed, "scen": scen, "n_distinct": int(len(set(rec["owner"].tolist()))),
               "n_repairs": rec["n_repairs"], "lv": {}}
        for lv in LEVELS_F:
            res, st = methods_and_stats(M[lv], rec["owner"], rec["k"])
            row["lv"][lv] = {"res": res, "det": st, "pseudo": pseudo[lv]}
        rows.append(row)
    return {"seed": seed, "rows": rows, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------- analysis
def ci(x):
    import c1h_lib as L
    return [round(v, 4) for v in L.mean_ci(x)]


def stat_value(d, name):
    """Detector statistic of one (row, level) dict; 'r_' = ratio to the pseudo pool, 'd_' = difference."""
    if name.startswith("r_"):
        s = name[2:]
        return d["det"][s] / max(d["pseudo"][s], 1e-9)
    if name.startswith("d_"):
        s = name[2:]
        return d["det"][s] - d["pseudo"][s]
    return d["det"][name]


DIRECTION = {"forced_cost": "le", "changed": "le", "n0": "le", "mutual_nn": "ge"}


def analyse(rows):
    import c1h_lib as L
    sel = [r for r in rows if r["seed"] in SEL_SEEDS]
    evl = [r for r in rows if r["seed"] in EVAL_SEEDS]
    methods = list(rows[0]["lv"]["base"]["res"])
    out = {"seeds": {"selection": SEL_SEEDS, "evaluation": EVAL_SEEDS}, "scenarios": SCEN,
           "n_distinct_owners": {sc: sorted(set(r["n_distinct"] for r in rows if r["scen"] == sc)) for sc in SCEN},
           "levels": {}}

    def sc_of(rr, lv, m, idx=0):
        return np.array([r["lv"][lv]["res"][m][idx] for r in rr])

    for lv in LEVELS_F:
        o = {}
        # 1. every method, per scenario, evaluation seeds: score and paired diff vs base
        o["eval_by_method"] = {}
        for m in methods:
            o["eval_by_method"][m] = {}
            for sc in SCEN:
                rr = [r for r in evl if r["scen"] == sc]
                o["eval_by_method"][m][sc] = {"score": ci(sc_of(rr, lv, m)),
                                              "diff_vs_base": ci(sc_of(rr, lv, m) - sc_of(rr, lv, "base")),
                                              "top1": round(float(sc_of(rr, lv, m, 1).mean()), 4),
                                              "per_k": {str(k): round(float(sc_of(rr, lv, m, 2 + i).mean()), 4)
                                                        for i, k in enumerate(L.K_VALUES)}}
        # 2. selection of the always-on soft methods on the selection seeds
        def sel_mean(m, scs=SCEN):
            return {sc: float((sc_of([r for r in sel if r["scen"] == sc], lv, m)
                               - sc_of([r for r in sel if r["scen"] == sc], lv, "base")).mean()) for sc in scs}
        chosen = {}
        fam = {"csls": ["csls%d" % k for k in CSLS_K], "dsm": ["dsm%g" % b for b in DSM_B],
               "usink": ["usink%g_%g" % tl for tl in USINK]}
        sel_tab = {m: {k: round(v, 4) for k, v in sel_mean(m).items()} for m in methods}
        o["selection_mean_gain_vs_base"] = sel_tab
        for f, ms in fam.items():
            # rule: largest mean gain over the 7 scenarios among those never negative on any scenario
            ok = [m for m in ms if min(sel_tab[m].values()) >= 0]
            chosen[f + "_safe"] = max(ok, key=lambda m: np.mean(list(sel_tab[m].values()))) if ok else None
            chosen[f + "_o2o"] = max(ms, key=lambda m: sel_tab[m]["o2o"])
        chosen["hcsls"] = max(["hcsls%d" % k for k in HCSLS_K], key=lambda m: sel_tab[m]["o2o"])
        o["chosen_on_selection"] = chosen
        # 3. gates: ON = Hungarian variant, OFF = the safe CSLS (or base); threshold on selection seeds
        gates = {}
        off_m = chosen["csls_safe"] or "base"
        stats = list(DET) + ["r_" + s for s in ("forced_cost", "changed", "n0")] + ["d_mutual_nn"]
        for on_m in ("hung", chosen["hcsls"]):
            for s in stats:
                how = DIRECTION[s.split("_", 1)[1] if s[:2] in ("r_", "d_") else s]
                vals = np.array([stat_value(r["lv"][lv], s) for r in sel])
                gain = sc_of(sel, lv, on_m) - sc_of(sel, lv, off_m)
                scen_of = np.array([r["scen"] for r in sel])
                order = np.sort(np.unique(vals))
                best, best_obj = None, None
                # thresholds: every observed value on the selection pools (ON if stat <= thr, or >= for
                # 'ge'). Feasible: on the selection pools the switched rule loses nothing on average in
                # any scenario (mean diff vs OFF >= -0.0005) and no single non-o2o pool loses more than
                # 0.005. Among feasible thresholds: the largest mean (over the 7 scenarios) of the
                # mean diff vs OFF. 'no threshold' (always OFF) is the fallback.
                for thr in order:
                    on = vals <= thr if how == "le" else vals >= thr
                    d = np.where(on, gain, 0.0)
                    means = [d[scen_of == sc].mean() for sc in SCEN]
                    worst = d[(scen_of != "o2o")].min() if np.any(scen_of != "o2o") else 0.0
                    if min(means) < -0.0005 or worst < -0.005:
                        continue
                    obj = float(np.mean(means))
                    if best_obj is None or obj > best_obj:
                        best, best_obj = float(thr), obj
                if best is None or best_obj <= 0:
                    gates["%s|%s|%s" % (on_m, off_m, s)] = {"how": how, "thr": None}
                    continue

                def apply(rr):
                    v = np.array([stat_value(r["lv"][lv], s) for r in rr])
                    on = v <= best if how == "le" else v >= best
                    return on, np.where(on, sc_of(rr, lv, on_m), sc_of(rr, lv, off_m))
                g = {"how": how, "thr": round(float(best), 5), "by_scenario": {}}
                for nm, rows_ in (("selection", sel), ("evaluation", evl)):
                    g["by_scenario"][nm] = {}
                    for sc in SCEN:
                        rr = [r for r in rows_ if r["scen"] == sc]
                        on, sw = apply(rr)
                        g["by_scenario"][nm][sc] = {
                            "on_share": round(float(on.mean()), 3),
                            "score": round(float(sw.mean()), 4),
                            "diff_vs_off": ci(sw - sc_of(rr, lv, off_m)),
                            "diff_vs_base": ci(sw - sc_of(rr, lv, "base")),
                            "pools_lost_vs_off": int(np.sum(sw < sc_of(rr, lv, off_m) - 1e-12))}
                gates["%s|%s|%s" % (on_m, off_m, s)] = g
        o["gates"] = gates
        # detector statistics per scenario (evaluation seeds): min / median / max
        o["detector_eval"] = {s: {sc: [round(float(f(np.array([stat_value(r["lv"][lv], s) for r in evl if r["scen"] == sc]))), 4)
                                       for f in (np.min, np.median, np.max)] for sc in SCEN} for s in stats}
        out["levels"][lv] = o
    # transfer: thresholds chosen at one level, applied at another (absolute vs self-calibrated)
    tr = {}
    for src, dst in (("base", "v2"), ("v2", "base"), ("base", "proxyA"), ("proxyA", "v2")):
        for key, g in out["levels"][src]["gates"].items():
            if g.get("thr") is None or not key.startswith("hung|"):
                continue
            s = key.split("|")[2]
            how = g["how"]
            off_dst = out["levels"][dst]["chosen_on_selection"]["csls_safe"] or "base"
            per = {}
            for sc in SCEN:
                rr = [r for r in evl if r["scen"] == sc]
                v = np.array([stat_value(r["lv"][dst], s) for r in rr])
                on = v <= g["thr"] if how == "le" else v >= g["thr"]
                sw = np.where(on, sc_of(rr, dst, "hung"), sc_of(rr, dst, off_dst))
                per[sc] = {"on_share": round(float(on.mean()), 3),
                           "diff_vs_off": round(float((sw - sc_of(rr, dst, off_dst)).mean()), 4)}
            tr["%s->%s|%s" % (src, dst, s)] = per
    out["threshold_transfer_eval"] = tr
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    import c1h_lib as L
    t0 = time.time()
    path = os.path.join(L.SCRATCH, "s8_rows.json")
    if not a.analyse:
        seeds = EVAL_SEEDS + SEL_SEEDS
        with Pool(int(os.environ.get("C1H_WORKERS", "8"))) as pool:
            res = []
            for r in pool.imap_unordered(one_seed, seeds, chunksize=1):
                res.append(r)
                print("seed %d done %.0fs (%d/%d)" % (r["seed"], r["seconds"], len(res), len(seeds)), flush=True)
        rows = [row for r in res for row in r["rows"]]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "seconds": {r["seed"]: r["seconds"] for r in res}}, f)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    out = analyse(rows)
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "s8_switch.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for lv in LEVELS_F:
        o = out["levels"][lv]
        print("==", lv, o["chosen_on_selection"])
        for m in ["base", "hung", o["chosen_on_selection"]["csls_safe"], o["chosen_on_selection"]["dsm_safe"],
                  o["chosen_on_selection"]["usink_safe"], o["chosen_on_selection"]["hcsls"]]:
            if m:
                print("  ", m, {sc: o["eval_by_method"][m][sc]["diff_vs_base"][0] for sc in SCEN})
    print("total %.0fs" % (time.time() - t0))
