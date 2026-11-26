"""Count target-player moves in fit / inner (training players only, via c10data.load_rows).
No data/local file is read (local_dir=None). CPU only."""
import json, os, re, sys, time
from collections import defaultdict
import numpy as np
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import load_rows, LEVELS
MOVE = re.compile(r";([BW])\[([a-s]{2}|tt|)\]")
WIN = 249  # transitions visible in the official 250-state window
out = {}
for part in ("fit", "inner"):
    t0 = time.time()
    rows = load_rows(os.path.join(REPO, "data", "training"), os.path.join(REPO, "results", "split_v1", "players_split.csv"), part=part, local_dir=None)
    t_load = time.time() - t0
    agg = defaultdict(lambda: defaultdict(float))
    lens = []
    anomalies = defaultdict(int)
    per_player = defaultdict(int)
    for r in rows:
        s = r["sgf_content"]
        mv = MOVE.findall(s)
        n = len(mv)
        cols = [c for c, _ in mv]
        passes = sum(1 for _, p in mv if p in ("", "tt"))
        if passes: anomalies["games_with_pass"] += 1
        if any(tag in s for tag in ("AB[", "AW[", "HA[")): anomalies["setup_or_handicap"] += 1
        if cols and cols[0] != "B": anomalies["first_move_not_B"] += 1
        if any(cols[i] == cols[i+1] for i in range(n-1)): anomalies["non_alternating"] += 1
        # sgf-level count consistency: number of ';' minus 1 (root)
        if s.count(";") - 1 != n: anomalies["semicolon_mismatch"] += 1
        tc = r["color"]
        tgt = sum(1 for c in cols if c == tc)
        lo = max(0, n - WIN)
        tgt_win = sum(1 for c in cols[lo:] if c == tc)
        lv = r["rank"]
        a = agg[lv]
        a["games"] += 1; a["moves"] += n; a["target_moves"] += tgt; a["target_moves_in_window"] += tgt_win
        a["games_ge_250_moves"] += (n >= 250)
        per_player[r["player_id"]] += tgt
        lens.append(n)
    tot = defaultdict(float)
    for lv in agg:
        for k, v in agg[lv].items(): tot[k] += v
    lens = np.array(lens)
    pp = np.array(sorted(per_player.values()))
    out[part] = {
        "load_seconds": round(t_load, 1), "games": int(tot["games"]),
        "moves_total": int(tot["moves"]), "target_moves": int(tot["target_moves"]),
        "target_moves_in_250_state_window": int(tot["target_moves_in_window"]),
        "window_coverage": tot["target_moves_in_window"] / tot["target_moves"],
        "mean_moves_per_game": float(lens.mean()), "median_moves": float(np.median(lens)),
        "p90_moves": float(np.percentile(lens, 90)), "max_moves": int(lens.max()), "min_moves": int(lens.min()),
        "frac_games_ge_250": float((lens >= 250).mean()),
        "mean_target_moves_per_game": tot["target_moves"] / tot["games"],
        "mean_target_moves_in_window_per_game": tot["target_moves_in_window"] / tot["games"],
        "anomalies": dict(anomalies),
        "players": len(per_player),
        "target_moves_per_player_quantiles": {q: int(np.percentile(pp, q)) for q in (10, 50, 90, 99)},
        "top_10pct_players_share_of_target_moves": float(pp[-max(1, len(pp)//10):].sum() / pp.sum()),
        "per_level": {lv: {"games": int(agg[lv]["games"]), "target_moves": int(agg[lv]["target_moves"]),
                           "target_moves_in_window": int(agg[lv]["target_moves_in_window"]),
                           "mean_moves": agg[lv]["moves"] / agg[lv]["games"],
                           "frac_ge_250": agg[lv]["games_ge_250_moves"] / agg[lv]["games"]} for lv in LEVELS},
    }
    print(part, json.dumps(out[part])[:300], flush=True)
    del rows
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "count_moves.json"), "w"), indent=1)
print("done")
