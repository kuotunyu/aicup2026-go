"""Competition 2: how much can transductive label-proportion tricks add on top of rule B?

Inputs: results/prereg05/explore/u1/u1_oof.npz, arrays mono_* = single-colour inner pseudo
questions (training players only), with OOF probabilities of Q02 (= method "04").
Nothing under data/local is read (u1_lib guard); no validation answers; score.py not used.

A simulated "test file" has 400 questions, 100 for each k in {5,10,15,20}. True level counts
come from one of three scenarios:
  bal   : exactly 40 per level (what a balanced test design would give)
  multi : counts ~ Multinomial(400, uniform)  (players drawn at random, not stratified)
  skew  : counts proportional to the number of inner players per level (A has 3.6x 1D)
Within a level, questions are drawn from distinct inner players where possible (1D has only 34
inner players, so up to 6 repeats there in 'bal'), each with a random k slot and a random round.

Decisions compared (all from the same Q02 probabilities):
  B      : rule B, per-question argmax of expected score (what 04 does)
  HUNG40 : maximise the file's total expected score subject to exactly 40 answers per level
           (linear_sum_assignment over 400 slots); assumes the 'bal' design
  EM     : Saerens-Latinne-Decaestecker EM re-estimates the file's level prior, re-weights the
           probabilities (training prior = uniform), then rule B; makes no balance assumption
  EMHUNG : EM prior, then assignment with slot counts = round(400 * EM prior)  (record only)

Output: sim_c2_label_matching.json next to this script (mean file score per decision, and the
mean / 2.5% / 97.5% of the per-file difference vs B over R files). CPU, about 1-3 minutes.

Usage (Git Bash, repo root):
  .venv/Scripts/python.exe results/prereg06/explore/prior_art/sim_c2_label_matching.py
"""
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "results", "prereg05", "explore", "u1"))
import u1_lib  # noqa: E402,F401  (data/local guard)
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

R = 1000
L = 10
KS = (5, 10, 15, 20)
E1 = math.exp(-1)
S = np.array([[1.0 if i == j else (E1 if abs(i - j) == 1 else 0.0) for j in range(L)] for i in range(L)])


def em_prior(P, iters=200, tol=1e-10):
    """Saerens et al. 2002: training prior uniform; returns the adjusted prior and posteriors."""
    pt = np.full(L, 1.0 / L)
    w = np.ones(L)
    for _ in range(iters):
        Q = P * w
        Q /= Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.abs(new - pt).max() < tol:
            pt = new
            break
        pt = new
        w = pt / (1.0 / L)
    Q = P * (pt / (1.0 / L))
    Q /= Q.sum(1, keepdims=True)
    return pt, Q


def assign(E, counts):
    """E: [n, L] expected score of answering level j; counts: slots per level (sum n)."""
    cols = np.repeat(np.arange(L), counts)
    r, c = linear_sum_assignment(-E[:, cols])
    out = np.empty(E.shape[0], int)
    out[r] = cols[c]
    return out


def counts_round(p, n):
    raw = p * n
    c = np.floor(raw).astype(int)
    rest = n - c.sum()
    c[np.argsort(-(raw - c))[:rest]] += 1
    return c


def main():
    t0 = time.time()
    z = np.load(os.path.join(REPO, "results", "prereg05", "explore", "u1", "u1_oof.npz"), allow_pickle=True)
    P_all, y_all, n_all, pl_all = z["mono_P_Q02"], z["mono_y"], z["mono_n"], z["mono_player"]
    player_level = z["player_level"]
    # index: (player, k) -> question rows
    idx = {}
    for i, (p, k) in enumerate(zip(pl_all.tolist(), n_all.tolist())):
        idx.setdefault((p, k), []).append(i)
    players_by_level = {l: np.flatnonzero(player_level == l) for l in range(L)}
    n_players = np.array([len(players_by_level[l]) for l in range(L)])

    rng = np.random.default_rng(20261002)
    results = {}
    for scen in ("bal", "multi", "skew"):
        rows = {d: [] for d in ("B", "HUNG40", "EM", "EMHUNG")}
        em_l1 = []
        for r in range(R):
            if scen == "bal":
                cnt = np.full(L, 40)
            elif scen == "multi":
                cnt = rng.multinomial(400, np.full(L, 1.0 / L))
            else:
                cnt = counts_round(n_players / n_players.sum(), 400)
            # k slots: 100 of each k, shuffled over the 400 questions (levels in blocks)
            kslots = rng.permutation(np.repeat(KS, 100))
            q, pos = [], 0
            for l in range(L):
                used = set()
                for j in range(cnt[l]):
                    k = int(kslots[pos])
                    pos += 1
                    elig = [int(p) for p in players_by_level[l] if (int(p), k) in idx]
                    fresh = [p for p in elig if p not in used]
                    p = fresh[rng.integers(len(fresh))] if fresh else elig[rng.integers(len(elig))]
                    used.add(p)
                    cand = idx[(p, k)]
                    q.append(cand[rng.integers(len(cand))])
            q = np.array(q)
            P, y = P_all[q], y_all[q]
            E = P @ S
            dec = {"B": E.argmax(1), "HUNG40": assign(E, np.full(L, 40))}
            pt, Q = em_prior(P)
            dec["EM"] = (Q @ S).argmax(1)
            dec["EMHUNG"] = assign(Q @ S, counts_round(pt, len(q)))
            em_l1.append(float(np.abs(pt - cnt / cnt.sum()).sum()))
            for d, pred in dec.items():
                rows[d].append(float(S[pred, y].mean()))
        base = np.array(rows["B"])
        res = {"files": R, "mean_score": {d: round(float(np.mean(v)), 4) for d, v in rows.items()},
               "diff_vs_B": {}, "em_prior_L1_error_mean": round(float(np.mean(em_l1)), 4)}
        for d in ("HUNG40", "EM", "EMHUNG"):
            diff = np.array(rows[d]) - base
            res["diff_vs_B"][d] = {"mean": round(float(diff.mean()), 4),
                                   "p2.5": round(float(np.percentile(diff, 2.5)), 4),
                                   "p97.5": round(float(np.percentile(diff, 97.5)), 4),
                                   "share_files_better": round(float((diff > 0).mean()), 3)}
        results[scen] = res
        print(scen, json.dumps(res), flush=True)
    out = {"what": __doc__.strip().splitlines()[0], "inputs": "u1_oof.npz mono_P_Q02 (inner, OOF)",
           "R": R, "seed": 20261002, "inner_players_per_level": n_players.tolist(),
           "results": results, "seconds": round(time.time() - t0, 1),
           "caveats": ["questions are re-drawn from a finite inner pool (636 players, 1D only 34), so files "
                       "share players; the spread is a lower bound on real file-to-file noise",
                       "Q02 OOF probabilities were calibrated on balanced inner pseudo questions",
                       "official level counts per file are unknown"]}
    with open(os.path.join(HERE, "sim_c2_label_matching.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
