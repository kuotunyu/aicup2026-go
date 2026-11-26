"""Step 0: read the fit training games once and cache the black games of players with >= 105 black
games (scratchpad, not the repository). Also records the counts used in the report."""
import sys
sys.dont_write_bytecode = True
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c1h_lib as L  # noqa: E402

t0 = time.time()
c = L.build_cache()
nb = c["n_black"]
out = {
    "fit_rows": c["n_rows_fit"], "fit_players": len(c["n_all"]),
    "players_black_ge_105": sum(1 for p in nb if nb[p] >= 105),
    "players_black_ge_120": sum(1 for p in nb if nb[p] >= 120),
    "players_black_ge_200": sum(1 for p in nb if nb[p] >= 200),
    "cached_black_games": len(c["gid"]),
    "games_not_alternating_BW": c["bad_alternation"],
    "black_ge_120_by_level": {lv: sum(1 for p in nb if nb[p] >= 120 and c["level"][p] == lv) for lv in L.LEVELS},
    "load_seconds": c["load_seconds"], "total_seconds": round(time.time() - t0, 1),
}
print(json.dumps(out, indent=1))
with open(os.path.join(HERE, "s0_cache_counts.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
