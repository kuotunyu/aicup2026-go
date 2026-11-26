"""prereg06 critic: two small CPU spot checks (inner training players only; no validation data, no scoring tool).

Check A (comparability of Kuboki 2025 with our numbers):
    Kuboki et al. report the same predictor under "random sampling" (k games drawn from many different players of one
    group) and "player-specific" sampling (k games of one player). We build the same two settings for Q02 on inner
    single-colour games with the c2_headroom virtual-question machinery (b_decompose.Model, Q02 fold parameters), so
    the ratio "player-specific / random" can be compared with the paper's. Random-sampling questions mix players from
    several folds, so each question is scored under every fold's parameter set and the 5 results are reported (the
    fold parameters are the low-dimensional Q02 fusion/calibration; the R20 CNN never saw inner players).

Check B (leaderboard noise for comparing two methods on the same file):
    Official-shaped competition-2 files (400 single-colour questions, 100 each of k = 5/10/15/20, 10 per level per k,
    players distinct inside a level) drawn from u1_oof.npz mono_* (Q02 and Q00 = R20 OOF probabilities, rule B).
    Reports the file-to-file SD of each method's score and of the paired difference Q02 - Q00.

Writes cr_checks.json next to this file. Nothing else is written.
"""

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C2 = os.path.join(os.path.dirname(HERE), "c2_headroom")
sys.path.insert(0, C2)
import hc_lib as H  # noqa: E402  (imports u1_lib, which installs the data/local guard)
import b_decompose as BD  # noqa: E402

U = H.U
L = 10


def check_b(z, n_files=2000, seed=7):
    rng = np.random.default_rng(seed)
    y, n, pl = z["mono_y"], z["mono_n"], z["mono_player"]
    s02 = H.score(H.rule_b(z["mono_P_Q02"]), y)
    s00 = H.score(H.rule_b(z["mono_P_Q00"]), y)
    # index: (level, k) -> player -> question rows
    idx = {}
    for r in range(len(y)):
        idx.setdefault((int(y[r]), int(n[r])), {}).setdefault(int(pl[r]), []).append(r)
    f02, f00 = np.zeros(n_files), np.zeros(n_files)
    reuse = 0
    for f in range(n_files):
        rows = []
        for lev in range(L):
            used = set()
            for k in H.KS:
                cand = [p for p in idx[(lev, k)] if p not in used]
                if len(cand) < 10:                       # small level: allow a player twice in the file
                    cand = list(idx[(lev, k)])
                    reuse += 1
                pick = rng.choice(len(cand), 10, replace=False)
                for j in pick:
                    p = cand[j]
                    used.add(p)
                    qs = idx[(lev, k)][p]
                    rows.append(qs[rng.integers(len(qs))])
        rows = np.asarray(rows)
        f02[f], f00[f] = s02[rows].mean(), s00[rows].mean()
    d = f02 - f00
    return {
        "files": n_files, "questions_per_file": 400,
        "Q02_file_score": H.summary(f02), "Q00_file_score": H.summary(f00),
        "paired_diff_Q02_minus_Q00": H.summary(d),
        "share_files_diff_le_0": float((d <= 0).mean()),
        "cells_with_player_reuse": reuse,
        "players_per_level_k": {f"{H.LEVELS[l]}_{k}": len(idx[(l, k)]) for l in range(L) for k in H.KS},
        "per_question_sd_Q02": float(s02.std(ddof=1)),
        "per_question_share_changed_Q02_vs_Q00": float((H.rule_b(z["mono_P_Q02"]) != H.rule_b(z["mono_P_Q00"])).mean()),
        "note": "official metric = plain mean over the 400 questions; cells balanced 10 per level per k",
    }


def check_a(d, z, model, n_per=1500, seed=11):
    rng = np.random.default_rng(seed)
    keys = d["inner"]["keys"]
    pid, plevel, pfold = z["player_id"], z["player_level"], z["player_fold"]
    index = {p: i for i, p in enumerate(pid)}
    pools = {}
    for r, (p, c) in enumerate(zip(keys["player_id"], keys["color"])):
        i = index.get(p)
        if i is not None:
            pools.setdefault((i, c), []).append(r)
    by_lc = {}
    for (i, c), rows in pools.items():
        by_lc.setdefault((int(plevel[i]), c), []).append((i, rows))
    out = {"n_questions_per_level_colour_k": n_per, "players_per_level_colour": {
        f"{H.LEVELS[l]}{c}": len(by_lc.get((l, c), [])) for l in range(L) for c in ("B", "W")}}
    res = {}
    for k in (1, 5, 10, 15, 20):
        per_fold = []
        for f in range(5):
            sc_all, y_all = [], []
            rs = np.random.default_rng(seed + k)          # same questions for every fold parameter set
            for l in range(L):
                for c in ("B", "W"):
                    units = by_lc.get((l, c), [])
                    if len(units) < k:
                        continue
                    G = np.empty((n_per, k), dtype=np.int64)
                    for q in range(n_per):
                        sel = rs.choice(len(units), k, replace=False)
                        G[q] = [units[s][1][rs.integers(len(units[s][1]))] for s in sel]
                    P = model.probs(G, f)
                    sc_all.append(H.score(H.rule_b(P), np.full(n_per, l)))
                    y_all.append(np.full(n_per, l))
            sc_all, y_all = np.concatenate(sc_all), np.concatenate(y_all)
            exact = float(np.mean([(sc_all[y_all == l] == 1.0).mean() for l in range(L)]))
            pm1 = float(np.mean([(sc_all[y_all == l] > 0).mean() for l in range(L)]))
            per_fold.append({"rb": H.rank_balanced(sc_all, y_all), "exact_rb": exact, "pm1_rb": pm1})
        res[str(k)] = {"by_fold_params": per_fold,
                       "rb_mean_over_fold_params": float(np.mean([x["rb"] for x in per_fold])),
                       "rb_range": [float(min(x["rb"] for x in per_fold)), float(max(x["rb"] for x in per_fold))]}
    out["random_sampling_Q02"] = res
    return out


def main():
    t0 = time.time()
    z = U.load_saved()
    out = {"what": "prereg06 critic spot checks (inner training players only; design stage; NOT validation)"}
    out["check_b_lb_noise"] = check_b(z)
    print(f"check B done ({time.time() - t0:.0f}s)", flush=True)
    d = U.load_inputs()
    with open(os.path.join(H.U1, "u1_repro.json"), encoding="utf-8") as fh:
        thetas = json.load(fh)["fold_thetas"]["Q02"]
    model = BD.Model(d["lp"], d["inner"]["op_o"], d["inner"]["op_has"], thetas)
    # reconstruction sanity (same as b_decompose)
    mg, mf, mP = z["mono_games"], z["mono_fold"], z["mono_P_Q02"]
    sel = np.nonzero(mf == 0)[0][:2000]
    out["recon_max_abs_diff_2000q"] = float(np.abs(model.probs(mg[sel], 0) - mP[sel]).max())
    out["check_a_kuboki_comparability"] = check_a(d, z, model)
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "cr_checks.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
