"""p04 proxy: per-move strength of the proxy on all inner games (fit counts only; inner is out of sample).

  top-1 (pooled, level-free choice model): share of target moves whose point has the highest pooled rate
    (ties split evenly)
  log-likelihood per move (nats): pooled table vs the table of the player's true level (alpha 20)
  level information per move = true-level minus pooled (nats and bits)
  per-level: mean per-move log-likelihood ratio of each level's table vs pooled, by true level (a 10x10 matrix,
    rows = true level); the diagonal minus the row mean says how much a move favours its own level
"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def main():
    z = np.load(os.path.join(HERE, "proxy_inner.npz"))
    s = z["scored"]
    ll = z["ll"][s]
    nm = z["n_moves"][s]
    y = z["label"].astype(int)[s]
    pooled_ll = z["pooled_ll"][s]
    hits = z["pooled_top1_hits"][s]
    variants = [str(v) for v in z["variants"]]
    out = {}
    nt = nm[:, 0].sum(1)
    no = nm[:, 1].sum(1)
    out["games"] = int(s.sum())
    out["target_moves"] = int(nt.sum())
    out["opponent_moves"] = int(no.sum())
    out["pooled_top1"] = float(hits.sum() / nt.sum())
    out["pooled_ll_per_move"] = float(pooled_ll.sum() / nt.sum())
    for v in ("t_choice_a5", "t_choice_a20", "t_choice_a100", "o_choice_a20"):
        j = variants.index(v)
        side_n = nt if v.startswith("t") else no
        tot = ll[:, j].sum(1)                                  # [G,10]
        true_ll = tot[np.arange(len(y)), y].sum() / side_n.sum()
        rec = {"true_level_ll_per_move": float(true_ll)}
        if v.startswith("t"):
            rec["level_info_per_move_nats"] = float(true_ll - out["pooled_ll_per_move"])
            rec["level_info_per_move_bits"] = rec["level_info_per_move_nats"] / math.log(2)
            # matrix: rows true level, cols table level, mean per-move ll minus pooled
            M = np.zeros((10, 10))
            for i in range(10):
                mk = y == i
                M[i] = (tot[mk].sum(0) - pooled_ll[mk].sum()) / side_n[mk].sum()
            rec["per_move_llr_vs_pooled_matrix"] = {LEVELS[i]: [round(float(x), 5) for x in M[i]] for i in range(10)}
            rec["diag_minus_rowmean"] = {LEVELS[i]: float(M[i, i] - M[i].mean()) for i in range(10)}
        # posterior per move if the moves were independent: per game, mean over moves of the level contrast
        out[v] = rec
    with open(os.path.join(HERE, "movestats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))
    for v in ("t_choice_a20",):
        print(v, json.dumps({k: x for k, x in out[v].items() if k != "per_move_llr_vs_pooled_matrix"}, indent=1))
        for lv, row in out[v]["per_move_llr_vs_pooled_matrix"].items():
            print(f"{lv:>3s} " + " ".join(f"{x:+.4f}" for x in row))


if __name__ == "__main__":
    main()
