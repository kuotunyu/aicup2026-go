"""Step 9 (continuation run): robustness of the analysis (c) gains to the split, and analysis (a) for
the fused score (where the remaining misses are).

Evaluation seeds 0-19, one-to-one pools (P120), two splits of each player's black games:
  random  shuffled (as everywhere else; seed s = official_c1_blackonly_sim 'black_random' pool s)
  ord     file order (official_c1_blackonly_sim 'black_fileorder': candidates = first 100 black
          games in file order, questions = the next k); same players, same k.
Score matrices (all specs fixed beforehand on the tuning seeds 100-109; nothing is chosen here):
  base         opening-only Q22-equivalent
  v1, v1_csls10   fused_nopost.json of the first run, without / with CSLS k=10 (its tune choice)
  v2, v2_csls3    fused_v2_nopost.json (s5b), without / with CSLS k=3 (its tune choice)
  proxyA       z(base) + z(style)  (Q04 stand-in of s5b)
  proxyA_v2, proxyA_v2_csls3   proxyA + the non-base v2 terms with the v2 weights (as is)
  v2_hung      v2 with Hungarian Top-1 (valid only if one-to-one)
Miss analysis (random split): rank distribution by k for base and v2_csls3; misses by level and by
the candidate's self-consistency (selfcos of s2: mean cosO of a 20-game fold of the candidate's own
games to its other 80, base tokens; label-free)."""
import sys
sys.dont_write_bytecode = True
import json
import os
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SEEDS = list(range(20))


def one(seed):
    sys.path.insert(0, HERE)
    import c1h_eval as E
    import c1h_fused as FU
    import c1h_lib as L
    t0 = time.time()
    c = L.load_cache()
    with open(os.path.join(HERE, "fused_nopost.json"), encoding="utf-8") as f:
        v1 = json.load(f)
    with open(os.path.join(HERE, "fused_v2_nopost.json"), encoding="utf-8") as f:
        v2 = json.load(f)
    rest = [(t, w) for t, w in zip(v2["terms"], v2["weights"]) if t != "base"]
    Gc = {}
    out = {"seed": seed}
    for split in ("random", "ord"):
        rec = L.make_pool(c, seed, "o2o", split=split)
        rec["scenario"] = "o2o|" + split
        T = {t: FU.term(t, rec, c, Gc) for t in sorted(set(["base", "style"] + v1["terms"] + v2["terms"]))}
        F = {"base": T["base"],
             "v1": E.fuse([T[t] for t in v1["terms"]], v1["weights"]),
             "v2": E.fuse([T[t] for t in v2["terms"]], v2["weights"]),
             "proxyA": E.fuse([T["base"], T["style"]], [1.0, 1.0]),
             "proxyA_v2": E.fuse([T["base"], T["style"]] + [T[t] for t, _ in rest], [1.0, 1.0] + [w for _, w in rest])}
        F["v1_csls10"] = E.csls(F["v1"], 10)
        F["v2_csls3"] = E.csls(F["v2"], 3)
        F["proxyA_v2_csls3"] = E.csls(F["proxyA_v2"], 3)
        F["v2_hung"] = E.with_top1(F["v2"], E.hungarian_top1(L.zrows(F["v2"])))
        res, ranks = {}, {}
        for name, M in F.items():
            r = L.ranks(M, rec["owner"])
            ranks[name] = r.tolist()
            res[name] = L.summary(r, rec["k"])
        out[split] = {"res": res, "k": rec["k"].tolist()}
        if split == "random":
            out[split]["ranks"] = {n: ranks[n] for n in ("base", "v2", "v2_csls3", "proxyA", "proxyA_v2_csls3")}
            out[split]["level"] = [c["level"][p] for p in rec["pool"]]
            G = Gc.get("base") if "base" in Gc else L.load_family("base")
            selfcos = np.zeros(len(rec["pool"]))
            for j in range(5):
                ref = [rows[:j * 20] + rows[(j + 1) * 20:] for rows in rec["cand_rows"]]
                qq = [rows[j * 20:(j + 1) * 20] for rows in rec["cand_rows"]]
                selfcos += np.diag(L.po.open_cos(G, ref, G, qq)[0]) / 5
            out[split]["selfcos"] = selfcos.tolist()
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    import c1h_lib as L
    t0 = time.time()
    with Pool(int(os.environ.get("C1H_WORKERS", "8"))) as pool:
        res = pool.map(one, SEEDS, chunksize=1)
    res.sort(key=lambda d: d["seed"])
    names = list(res[0]["random"]["res"])

    def ci(x):
        return [round(v, 4) for v in L.mean_ci(x)]
    out = {"what": __doc__, "seeds": SEEDS, "by_split": {}}
    for split in ("random", "ord"):
        tab = {}
        for n in names:
            sc = np.array([d[split]["res"][n]["score"] for d in res])
            b = np.array([d[split]["res"]["base"]["score"] for d in res])
            tab[n] = {"score": ci(sc), "diff_vs_base": ci(sc - b),
                      "top1": ci([d[split]["res"][n]["top1"] for d in res]),
                      "top5": ci([d[split]["res"][n]["top5"] for d in res]),
                      "per_k": {str(k): round(float(np.mean([d[split]["res"][n]["per_k"][str(k)]["score"] for d in res])), 4)
                                for k in L.K_VALUES}}
            for refn in ("proxyA", "v1_csls10"):
                r0 = np.array([d[split]["res"][refn]["score"] for d in res])
                tab[n]["diff_vs_" + refn] = ci(sc - r0)
        out["by_split"][split] = tab
    # split effect on the gains: paired over seeds (same players and k, different games)
    out["gain_ord_minus_random"] = {}
    for n in names:
        g = [(d["ord"]["res"][n]["score"] - d["ord"]["res"]["base"]["score"])
             - (d["random"]["res"][n]["score"] - d["random"]["res"]["base"]["score"]) for d in res]
        out["gain_ord_minus_random"][n] = ci(g)
    # miss analysis on the random split
    K = np.concatenate([d["random"]["k"] for d in res])
    LV = np.concatenate([d["random"]["level"] for d in res])
    SC = np.concatenate([d["random"]["selfcos"] for d in res])
    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 400)]
    qs = np.quantile(SC, np.linspace(0, 1, 6))
    miss = {}
    for n in ("base", "v2", "v2_csls3", "proxyA", "proxyA_v2_csls3"):
        R = np.concatenate([d["random"]["ranks"][n] for d in res])
        lost = 1 - L.pts(R)
        mm = {"rank_distribution_by_k": {str(k): {"%d-%d" % b: round(float(((R[K == k] >= b[0]) & (R[K == k] <= b[1])).mean()), 4)
                                                 for b in bins} for k in L.K_VALUES},
              "points_lost_per_question": round(float(lost.mean()), 4),
              "share_of_lost_points_by_k": {str(k): round(float(lost[K == k].sum() / max(lost.sum(), 1e-12)), 3)
                                            for k in L.K_VALUES},
              "lost_in_rank_2_5": round(float(lost[(R >= 2) & (R <= 5)].sum() / len(R)), 4),
              "lost_in_rank_gt5": round(float(lost[R > 5].sum() / len(R)), 4),
              "by_level": {lv: {"score": round(float(L.pts(R[LV == lv]).mean()), 4),
                                "miss_top5": round(float((R[LV == lv] > 5).mean()), 4)} for lv in L.LEVELS},
              "by_selfcos_quintile": []}
        for i in range(5):
            m = (SC >= qs[i]) & ((SC <= qs[i + 1]) if i == 4 else (SC < qs[i + 1]))
            mm["by_selfcos_quintile"].append({"range": [round(float(qs[i]), 4), round(float(qs[i + 1]), 4)],
                                              "score": round(float(L.pts(R[m]).mean()), 4),
                                              "top1": round(float((R[m] == 1).mean()), 4),
                                              "share_of_lost_points": round(float(lost[m].sum() / max(lost.sum(), 1e-12)), 3)})
        miss[n] = mm
    out["miss_analysis_random"] = miss
    out["seconds_by_seed"] = [d["seconds"] for d in res]
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "s9_split_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for split in ("random", "ord"):
        print(split, {n: (out["by_split"][split][n]["score"][0], out["by_split"][split][n]["diff_vs_base"][0]) for n in names})
    print("ord-random gain", {n: out["gain_ord_minus_random"][n][0] for n in names})
    print("total %.0fs" % (time.time() - t0))
