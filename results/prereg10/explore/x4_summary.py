"""prereg10 explore x4: summarise x1_scenarios rows (data/prereg10/tmp/x1_rows.json) -> x1_scenarios.json.

Per scenario and method: mean score / Top-1 / Top-5 over the pools (seeds), 95% t interval, per-k scores.
Paired comparisons with 95% bootstrap intervals over pools (10,000 resamples of the 20 pool-level differences;
the same seed = the same 400 players in every scenario except big*):
  * each scenario minus base (same method);
  * within a scenario: V2 - Q04, V2 - E0, V2noOpen - E0, 06eff - V2 (the cost of the PREREG_02 guard).
Guard: rho_raw per pool, the level counts.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IN = os.path.join(REPO, "data", "prereg10", "tmp", "x1_rows.json")
OUT = os.path.join(HERE, "x1_scenarios.json")
METHODS = ("E0", "O", "L", "JO", "S2", "Q04", "V0c", "V2nc", "V2", "V2noOpen", "V2noE", "06eff")
PAIRS = (("V2", "Q04"), ("V2", "E0"), ("V2noOpen", "E0"), ("06eff", "V2"), ("V2", "V2nc"), ("V2noOpen", "V2"))
RNG = np.random.default_rng(20261010)


def tci(x):
    from scipy import stats
    x = np.asarray(x, float)
    m = float(x.mean())
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return [round(m, 4), round(m - h, 4), round(m + h, 4)]


def boot(d, B=10000):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return [round(float(d.mean()), 4), round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def main():
    with open(IN, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    scens = []
    for r in rows:
        if r["scen"] not in scens:
            scens.append(r["scen"])
    by = {(r["scen"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    out = {"what": "prereg10 x1: 06 and its parts in harder mock pools of fit players (summary of x1_scenarios.py)",
           "seeds": seeds, "scenarios": {}}
    for sc in scens:
        R = [by[(sc, s)] for s in seeds if (sc, s) in by]
        d = {"n_pools": len(R), "n_cand": R[0]["n_cand"], "n_q": R[0]["n_q"],
             "mean_distinct_owners": float(np.mean([r["n_distinct_owner"] for r in R])),
             "mean_k": round(float(np.mean([np.mean(r["k"]) for r in R])), 3),
             "lda_train_players_mean": float(np.mean([r["lda_train"][0] for r in R])), "methods": {}}
        for m in METHODS:
            if m not in R[0]["res"]:
                continue
            d["methods"][m] = {
                "score": tci([r["res"][m]["score"] for r in R]),
                "top1": round(float(np.mean([r["res"][m]["top1"] for r in R])), 4),
                "top5": round(float(np.mean([r["res"][m]["top5"] for r in R])), 4),
                "by_k": {k: round(float(np.mean([r["res"][m]["by_k"][k] for r in R if k in r["res"][m]["by_k"]])), 4)
                         for k in R[0]["res"][m]["by_k"]}}
        d["within_pairs_boot"] = {"%s-%s" % (a, b): boot([r["res"][a]["score"] - r["res"][b]["score"] for r in R])
                                  for a, b in PAIRS if a in R[0]["res"] and b in R[0]["res"]}
        if sc != "base":
            P = [(by[(sc, s)], by[("base", s)]) for s in seeds if (sc, s) in by and ("base", s) in by]
            d["minus_base_boot"] = {m: boot([a["res"][m]["score"] - b["res"][m]["score"] for a, b in P])
                                    for m in METHODS if m in R[0]["res"] and m in P[0][1]["res"]}
        if R[0]["guard"]:
            rho = [r["guard"]["rho"] for r in R]
            lv = [r["guard"]["level"] for r in R]
            d["guard"] = {"rho_mean": round(float(np.mean(rho)), 4), "rho_min": round(float(np.min(rho)), 4),
                          "rho_max": round(float(np.max(rho)), 4),
                          "levels": {str(k): int(lv.count(k)) for k in (1, 2, 3)}}
        if R[0].get("info"):
            inf = [r["info"] for r in R]
            d["novel_info"] = {"mean_k_eff": round(float(np.mean([i["mean_k_eff"] for i in inf])), 3),
                               "mean_replaced_owner": float(np.mean([i["n_replaced_owner"] for i in inf])),
                               "mean_distinct_owner": float(np.mean([i["n_distinct_owner"] for i in inf])),
                               "median_novel_games_per_player": float(np.mean(
                                   [i["n_novel_games_per_player"]["median"] for i in inf]))}
        out["scenarios"][sc] = d
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    # compact table
    print("%-8s %6s %6s %6s %6s %6s %6s %6s %6s %6s  rho  lv" % ("scen", "E0", "O", "L", "Q04", "V0c", "V2",
                                                                  "noOpen", "noE", "06eff"))
    for sc, d in out["scenarios"].items():
        mm = d["methods"]
        g = d.get("guard")
        print("%-8s " % sc + " ".join("%6.4f" % mm[m]["score"][0] if m in mm else "   -  "
                                      for m in ("E0", "O", "L", "Q04", "V0c", "V2", "V2noOpen", "V2noE", "06eff"))
              + ("  %.3f %s" % (g["rho_mean"], g["levels"]) if g else ""))


if __name__ == "__main__":
    sys.exit(main())
