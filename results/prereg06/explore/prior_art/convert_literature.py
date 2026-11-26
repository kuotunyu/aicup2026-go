"""Put published strength-estimation numbers on our competition-2 scale and next to our own
inner (training-player) numbers for the same number of games k.

Our scale (competition 2): exact = 1, adjacent level = 1/e, else 0.
For a paper that reports accuracy (exact) and accuracy+-1 (exact or adjacent), the converted
score is   exact + e^-1 * (acc_pm1 - exact).
This is only a rough bridge: different server (Fox vs Yugen no Ma), different level groups,
argmax/rounded predictions instead of our expected-score rule, and the paper's test players may
overlap its training players (not stated).

Our side: results/prereg05/explore/u1/u1_oof.npz, arrays mono_* (single-colour inner pseudo
questions, k = 5/10/15/20; inner = training players only; OOF probabilities of Q02 = the method
"04"). Decision = rule B (expected score). Numbers are level-balanced (mean over the 10 levels).
Reproduction check: the per-k rule-B score must equal u1_analysis.json mono.by_n.Q02.

Usage (Git Bash, repo root):
  .venv/Scripts/python.exe results/prereg06/explore/prior_art/convert_literature.py
Writes convert_literature.json next to this script. CPU, a few seconds.
"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "results", "prereg05", "explore", "u1"))
import u1_lib  # noqa: E402,F401  (installs the data/local guard on import)
import numpy as np  # noqa: E402

E1 = math.exp(-1)

# ---------------------------------------------------------------- literature (exact, acc+-1) in %
LIT = {
    "Kuboki2025_player_specific_proposed": {
        "src": "arXiv 2505.00279 Table 4 (Go, Fox, 11 groups, player-specific sampling)",
        "k": {5: (58.1, 95.3), 10: (61.0, 96.2), 15: (63.2, 96.6)}},
    "Kuboki2025_player_specific_ChenSE": {
        "src": "arXiv 2505.00279 Table 4 (Chen et al. SE re-evaluated, player-specific)",
        "k": {5: (51.6, 93.1), 10: (59.4, 96.0), 15: (61.1, 96.9)}},
    "Kuboki2025_player_specific_no_strength": {
        "src": "arXiv 2505.00279 Table 7 (w/o strength score = HumanSL priors + losses only)",
        "k": {5: (53.1, 93.0), 10: (56.5, 94.5), 15: (57.8, 95.1)}},
    "Kuboki2025_random_sampling_proposed": {
        "src": "arXiv 2505.00279 Table 4 (random sampling of games from many players of a group)",
        "k": {5: (70.4, 99.5), 10: (82.8, 99.9), 15: (89.4, 100.0)}},
}


def conv(exact, pm1):
    return (exact + E1 * (pm1 - exact)) / 100.0


def main():
    out = {"formula": "score = exact + e^-1 * (acc_pm1 - exact)", "literature": {}, "ours": {}}
    for name, d in LIT.items():
        out["literature"][name] = {"src": d["src"], "by_k": {
            str(k): {"exact": e / 100, "acc_pm1": p / 100, "score": round(conv(e, p), 4)}
            for k, (e, p) in d["k"].items()}}

    z = np.load(os.path.join(REPO, "results", "prereg05", "explore", "u1", "u1_oof.npz"),
                allow_pickle=True)
    ref = json.load(open(os.path.join(REPO, "results", "prereg05", "explore", "u1", "u1_analysis.json"),
                         encoding="utf-8"))["mono"]["by_n"]
    L = 10
    S = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            S[i, j] = 1.0 if i == j else (E1 if abs(i - j) == 1 else 0.0)
    y, n = z["mono_y"], z["mono_n"]
    for comb in ("Q00", "Q02"):
        P = z["mono_P_%s" % comb]
        pred = (P @ S).argmax(1)
        d = np.abs(pred - y)
        res = {}
        for k in (5, 10, 15, 20):
            m = n == k
            ex = np.mean([np.mean(d[m & (y == l)] == 0) for l in range(L)])
            pm1 = np.mean([np.mean(d[m & (y == l)] <= 1) for l in range(L)])
            sc = np.mean([np.mean(S[pred[m & (y == l)], l]) for l in range(L)])
            res[str(k)] = {"questions": int(m.sum()), "exact_rb": round(float(ex), 4),
                           "acc_pm1_rb": round(float(pm1), 4), "score_rb": round(float(sc), 4),
                           "u1_reference_score": round(float(ref[str(k)][comb]), 4),
                           "reproduced": bool(abs(sc - ref[str(k)][comb]) < 1e-12)}
        res["mean_over_k_5_10_15_20"] = round(float(np.mean([res[str(k)]["score_rb"] for k in (5, 10, 15, 20)])), 4)
        out["ours"][comb] = res

    path = os.path.join(HERE, "convert_literature.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for name, d in out["literature"].items():
        print(name, {k: v["score"] for k, v in d["by_k"].items()})
    for comb, d in out["ours"].items():
        print(comb, {k: (v["exact_rb"], v["acc_pm1_rb"], v["score_rb"], v["reproduced"])
                     for k, v in d.items() if isinstance(v, dict)}, "mean", d["mean_over_k_5_10_15_20"])
    print("wrote", path)


if __name__ == "__main__":
    main()
