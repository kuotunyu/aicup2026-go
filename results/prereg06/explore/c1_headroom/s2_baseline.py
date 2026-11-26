"""Step 2 = analysis (a): Q22-equivalent (opening-only, raw, both token sets, N = 10; the opening term
of Q04) on official-shaped simulated pools of fit players (P120 black_random, as
results/test02/design/official_structure/official_c1_blackonly_sim.py).

Reports score / top1 / top5 by k, the rank distribution of the true player, where the points are
lost, misses by level and by label-free 'distinctiveness' of the candidate (computed from the
candidate's own 100 games only), and checks seeds 0-4 against the committed sim json."""
import sys
sys.dont_write_bytecode = True
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEEDS = list(range(20))


def per_pool(seed):
    sys.path.insert(0, HERE)
    import c1h_lib as L
    import c1h_eval as E
    t = time.time()
    c = L.load_cache()
    G = L.load_family("base")
    rec = L.make_pool(c, seed, "o2o")
    S = E.cos_term(G, rec)
    r = L.ranks(S, rec["owner"])
    pool = rec["pool"]
    # distinctiveness of each candidate, from its candidate games only
    first = []
    ent3 = []
    for rows in rec["cand_rows"]:
        f = Counter(c["gpts"][g][:2] for g in rows)
        first.append(max(f.values()) / len(rows))
        t3 = Counter(c["gpts"][g][0:2] + c["gpts"][g][4:6] + c["gpts"][g][8:10] for g in rows)
        p = np.array(list(t3.values()), float) / len(rows)
        ent3.append(float(-(p * np.log(p)).sum()))
    # confusability: cosine between candidate profiles (same weighting, IDF over candidates)
    CC = L.po.open_cos(G, rec["cand_rows"], G, rec["cand_rows"])[0]
    np.fill_diagonal(CC, -1)
    nn_cand = CC.max(1)
    # self-consistency: profile of 80 games vs the other 20 (5 folds), mean cosine to itself
    selfcos = np.zeros(len(pool))
    for j in range(5):
        ref = [rows[:j * 20] + rows[(j + 1) * 20:] for rows in rec["cand_rows"]]
        qq = [rows[j * 20:(j + 1) * 20] for rows in rec["cand_rows"]]
        cs = L.po.open_cos(G, ref, G, qq)[0]
        selfcos += np.diag(cs) / 5
    lv = [c["level"][p] for p in pool]
    return {"seed": seed, "ranks": r.tolist(), "k": rec["k"].tolist(), "level": lv,
            "first_share": first, "ent3": ent3, "nn_cand_cos": nn_cand.tolist(), "selfcos": selfcos.tolist(),
            "summary": L.summary(r, rec["k"]), "seconds": round(time.time() - t, 1)}


def binned(x, r, nb=5):
    import c1h_lib as L
    x, r = np.asarray(x), np.asarray(r)
    qs = np.quantile(x, np.linspace(0, 1, nb + 1))
    out = []
    for i in range(nb):
        m = (x >= qs[i]) & ((x <= qs[i + 1]) if i == nb - 1 else (x < qs[i + 1]))
        out.append({"range": [round(float(qs[i]), 4), round(float(qs[i + 1]), 4)], "n": int(m.sum()),
                    "score": round(float(L.pts(r[m]).mean()), 4), "top1": round(float((r[m] == 1).mean()), 4),
                    "miss_top5": round(float((r[m] > 5).mean()), 4)})
    return out


if __name__ == "__main__":
    import c1h_lib as L
    t0 = time.time()
    with Pool(int(os.environ.get('C1H_WORKERS', '8'))) as pool:
        res = pool.map(per_pool, SEEDS, chunksize=1)
    res.sort(key=lambda d: d["seed"])
    # reproduction of the committed sim (raw|N10|both, P120|black_random, seeds 0-4)
    with open(os.path.join(L.REPO, "results", "test02", "design", "official_structure",
                           "official_c1_blackonly_sim.json"), encoding="utf-8") as f:
        ref = json.load(f)["tables"]["P120|black_random"]["raw|N10|both"]["score_by_seed"]
    repro = [{"seed": d["seed"], "here": round(d["summary"]["score"], 4), "sim_json": ref[d["seed"]]}
             for d in res if d["seed"] < len(ref)]
    R = np.concatenate([d["ranks"] for d in res])
    K = np.concatenate([d["k"] for d in res])
    LV = np.concatenate([d["level"] for d in res])
    scores = [d["summary"]["score"] for d in res]
    by_k = {}
    for k in L.K_VALUES:
        xs = [d["summary"]["per_k"][str(k)] for d in res]
        by_k[str(k)] = {m: [round(v, 4) for v in L.mean_ci([x[m] for x in xs])] for m in ("score", "top1", "top5")}
    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, 50), (51, 400)]
    rank_dist = {"%d-%d" % b: round(float(((R >= b[0]) & (R <= b[1])).mean()), 4) for b in bins}
    rank_dist_by_k = {str(k): {"%d-%d" % b: round(float(((R[K == k] >= b[0]) & (R[K == k] <= b[1])).mean()), 4)
                               for b in bins} for k in L.K_VALUES}
    lost = 1 - L.pts(R)
    loss = {"total_lost_per_question": round(float(lost.mean()), 4),
            "lost_in_rank_2_5": round(float(lost[(R >= 2) & (R <= 5)].sum() / len(R)), 4),
            "lost_in_rank_gt5": round(float(lost[R > 5].sum() / len(R)), 4)}
    by_level = {lv: {"score": round(float(L.pts(R[LV == lv]).mean()), 4),
                     "top1": round(float((R[LV == lv] == 1).mean()), 4),
                     "miss_top5": round(float((R[LV == lv] > 5).mean()), 4)} for lv in L.LEVELS}
    dist = {name: binned(np.concatenate([d[name] for d in res]), R)
            for name in ("first_share", "ent3", "nn_cand_cos", "selfcos")}
    # misses at k=5 vs k=20 by selfcos
    out = {
        "what": "analysis (a): opening-only Q22-equivalent (raw, both sets, N=10, p02 code) on P120 black_random "
                "official-shaped pools of fit players (400 candidates x 100 black games, 400 questions, k 5/10/15/20 "
                "x 100, one question per candidate)",
        "seeds": SEEDS,
        "reproduction_vs_official_c1_blackonly_sim_json": repro,
        "score_mean_ci_over_pools": [round(v, 4) for v in L.mean_ci(scores)],
        "top1_mean_ci": [round(v, 4) for v in L.mean_ci([d["summary"]["top1"] for d in res])],
        "top5_mean_ci": [round(v, 4) for v in L.mean_ci([d["summary"]["top5"] for d in res])],
        "score_by_seed": [round(s, 4) for s in scores],
        "by_k": by_k,
        "rank_distribution": rank_dist,
        "rank_distribution_by_k": rank_dist_by_k,
        "points_lost": loss,
        "by_level": by_level,
        "by_distinctiveness_quintile": dist,
        "distinctiveness_defs": {
            "first_share": "share of the candidate's 100 games whose first move is its most common first move",
            "ent3": "entropy of the (own moves 1-3) tuple over the candidate's 100 games",
            "nn_cand_cos": "cosO between the candidate's profile and the most similar other candidate profile",
            "selfcos": "mean cosO of the candidate's 20-game fold to its own 80-game profile (5 folds)"},
        "seconds_by_pool": [d["seconds"] for d in res],
        "total_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "s2_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(L.SCRATCH, "s2_ranks.npz"), R=R, K=K)
    print(json.dumps({k: out[k] for k in ("reproduction_vs_official_c1_blackonly_sim_json", "score_mean_ci_over_pools",
                                         "top1_mean_ci", "top5_mean_ci", "by_k", "rank_distribution", "points_lost",
                                         "by_level")}, indent=1))
    print(json.dumps(dist, indent=1))
