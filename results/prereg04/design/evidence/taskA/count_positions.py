"""Exact per-move sample counts (target moves in the 250-state window whose 8-state history is inside
the window or reaches the empty root), fit and inner, per level. c10data.load_rows only; CPU."""
import json, os, re, sys
from collections import defaultdict
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import load_rows, LEVELS
MOVE = re.compile(r";([BW])\[")
H = 8
out = {}
for part in ("fit", "inner"):
    rows = load_rows(os.path.join(REPO, "data", "training"), os.path.join(REPO, "results", "split_v1", "players_split.csv"), part=part, local_dir=None)
    per = defaultdict(int); games = defaultdict(int); mn = 10**9; mx = 0
    for r in rows:
        cols = MOVE.findall(r["sgf_content"])
        n = len(cols)
        lo = max(0, n - 249)
        trunc = n + 1 > 250
        c = 0
        for k in range(lo, n):
            if cols[k] != r["color"]:
                continue
            if trunc and (k - lo) < H - 1:
                continue
            c += 1
        per[r["rank"]] += c; games[r["rank"]] += 1
        mn = min(mn, c); mx = max(mx, c)
    out[part] = {"positions": sum(per.values()), "games": sum(games.values()), "min_per_game": mn, "max_per_game": mx,
                 "per_level": {lv: per[lv] for lv in LEVELS}}
    print(part, out[part], flush=True)
    del rows
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "count_positions.json"), "w"), indent=1)
