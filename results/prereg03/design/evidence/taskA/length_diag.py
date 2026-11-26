"""Inner-only diagnostic: epoch-9 per-game accuracy / NLL by game length (moves), to judge whether
the 250-state history window (games >= 250 moves lose their opening) is a limiting factor.
Games via c10data.load_rows(part='inner') only; dumps read-only."""
import json
import os
import re
import sys

import numpy as np

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import load_rows  # noqa: E402

D = os.path.join(REPO, "results", "prereg02", "rank", "dumps")
MOVE = re.compile(r";[BW]\[")


def lsm(x):
    m = x.max(-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


rows = load_rows(os.path.join(REPO, "data", "training"),
                 os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                 part="inner", local_dir=os.path.join(REPO, "data", "local"))
nmoves = {(r["rank"], r["game_id"]): len(MOVE.findall(r["sgf_content"])) for r in rows}
k = np.load(os.path.join(D, "inner_keys.npz"), allow_pickle=False)
y = k["label"].astype(np.int64)
L = np.array([nmoves[(str(r), str(g))] for r, g in zip(k["rank"], k["game_id"])])
z = np.load(os.path.join(D, "inner_p02e9_d4.npz"), allow_pickle=False)["logits"].astype(np.float64)
l8 = lsm(np.mean([lsm(z[:, s, :]) for s in range(8)], axis=0))
out = {"n": int(len(y)), "moves_quantiles": {q: int(np.quantile(L, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
       "frac_ge_250": float((L >= 250).mean()), "frac_ge_360": float((L >= 360).mean())}
bins = [(100, 149), (150, 199), (200, 249), (250, 299), (300, 999)]
out["by_len"] = {}
for lo, hi in bins:
    m = (L >= lo) & (L <= hi)
    out["by_len"][f"{lo}-{hi}"] = {"n": int(m.sum()), "acc_d4": float((l8[m].argmax(1) == y[m]).mean()),
                                   "nll_d4": float(-l8[m][np.arange(m.sum()), y[m]].mean()),
                                   "level_mix": np.bincount(y[m], minlength=10).tolist()}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "length_diag.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
