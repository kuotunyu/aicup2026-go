"""Judge check (fit only): per-level concentration of games over players vs R20 inner OOF per level."""
import sys, os, json
sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import load_rows, LEVELS
from collections import Counter, defaultdict
import numpy as np
rows = load_rows(os.path.join(REPO, "data", "training"), os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                 part="fit", local_dir=os.path.join(REPO, "data", "local"))
cnt = defaultdict(Counter)
for r in rows:
    cnt[r["rank"]][r["player_id"]] += 1
oof = dict(zip(LEVELS, [.742, .618, .577, .536, .467, .488, .472, .576, .631, .747]))
out = {}
tot_eff = 0
for L in LEVELS:
    n = np.array(sorted(cnt[L].values(), reverse=True), dtype=float)
    p = n / n.sum()
    eff = 1 / np.sum(p ** 2)
    k = max(1, int(round(0.1 * len(n))))
    out[L] = dict(players=len(n), games=int(n.sum()), eff_players=round(float(eff), 1),
                  top10pct_share=round(float(n[:k].sum() / n.sum()), 3), max_games=int(n[0]),
                  median=float(np.median(n)), r20_oof=oof[L])
    tot_eff += eff
for L in LEVELS:
    print(L, out[L])
e = np.array([out[L]["eff_players"] / out[L]["players"] for L in LEVELS]); s = np.array([oof[L] for L in LEVELS])
print("sum eff", tot_eff, "corr(eff/players, oof)", np.corrcoef(e, s)[0, 1])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conc.json"), "w"), indent=1)
