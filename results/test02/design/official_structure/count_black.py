"""Count fit players by number of black games (training data only, via c10data.load_rows)."""
import json
import os
import sys
from collections import Counter, defaultdict

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import LEVELS, load_rows  # noqa: E402

rows = load_rows(os.path.join(REPO, "data", "training"),
                 os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                 part="fit", local_dir=os.path.join(REPO, "data", "local"))
nb = Counter()
nall = Counter()
lvl = {}
first_white_before_black = 0
seen_w = defaultdict(bool)
order_ok = defaultdict(lambda: True)
for r in rows:
    p = r["player_id"]
    lvl[p] = r["rank"]
    nall[p] += 1
    if r["color"] == "B":
        nb[p] += 1
        if seen_w[p]:
            order_ok[p] = False
    else:
        seen_w[p] = True
out = {}
for t in (100, 105, 110, 115, 120, 140, 160, 200):
    out[">=%d black" % t] = {lv: sum(1 for p in lvl if lvl[p] == lv and nb[p] >= t) for lv in LEVELS}
    out[">=%d black" % t]["total"] = sum(1 for p in lvl if nb[p] >= t)
out[">=105 all (mixed sim rule)"] = {lv: sum(1 for p in lvl if lvl[p] == lv and nall[p] >= 105) for lv in LEVELS}
elig = [p for p in lvl if nb[p] >= 105]
out["black_then_white_in_file_order_among_>=105_black"] = sum(order_ok[p] for p in elig)
out["n_elig_105"] = len(elig)
print(json.dumps(out, indent=1))
