"""Per-move positions of the pseudo_inner bank games (inner players only; H = 8 rule). CPU."""
import json, os, re, sys
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import load_rows
MOVE = re.compile(r";([BW])\[")
H = 8
p = json.load(open(os.path.join(REPO, "results", "prereg02", "rank", "pseudo_inner.json"), encoding="utf-8"))
bank = {(pl["rank"], g) for pl in p["players"] for g in pl["bank"]}
rows = load_rows(os.path.join(REPO, "data", "training"), os.path.join(REPO, "results", "split_v1", "players_split.csv"), part="inner", local_dir=None)
pos = opp = games = 0
for r in rows:
    if (r["rank"], r["game_id"]) not in bank:
        continue
    cols = MOVE.findall(r["sgf_content"]); n = len(cols); lo = max(0, n - 249); trunc = n + 1 > 250
    for k in range(lo, n):
        if trunc and (k - lo) < H - 1:
            continue
        if cols[k] == r["color"]: pos += 1
        else: opp += 1
    games += 1
out = {"bank_games": games, "bank_target_positions": pos, "bank_opponent_positions": opp}
print(out)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "count_bank.json"), "w"), indent=1)
