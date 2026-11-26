"""Build the small, seeded fit-game sample used by test_hs_lib.py and the benchmarks (fit players only).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg07/explore/pilot_humansl/hs_fixtures.py

Reads the training CSVs through tools/c10data.load_rows(part='fit') (split v1 fit players; no local_dir, so nothing
under data/local is touched; hs_lib's guard is installed first). Writes data/prereg07/fixtures/fit_sample.json
(git-ignored): per level, PER_LEVEL games from PER_LEVEL distinct fit players, chosen with random.Random(SEED).
"""

import json
import os
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hs_lib as H  # noqa: E402  (installs the data/local guard)
from c10data import LEVELS, load_rows  # noqa: E402

SEED = 20261002
PER_LEVEL = 12
OUT = os.path.join(H.REPO, "data", "prereg07", "fixtures", "fit_sample.json")


def main():
    t0 = time.time()
    rows = load_rows(os.path.join(H.REPO, "data", "training"),
                     os.path.join(H.REPO, "results", "split_v1", "players_split.csv"),
                     part="fit", local_dir=None, check_counts=True)
    t_load = time.time() - t0
    rng = random.Random(SEED)
    by_player = {}
    for r in rows:
        by_player.setdefault((r["rank"], r["player_id"]), []).append(r)
    games = []
    for lv in LEVELS:
        players = sorted(p for (rk, p) in by_player if rk == lv)
        for p in rng.sample(players, PER_LEVEL):
            cand = sorted(by_player[(lv, p)], key=lambda r: r["game_id"])
            r = rng.choice(cand)
            games.append({k: r[k] for k in ("player_id", "game_id", "rank", "color", "sgf_content")})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"what": "seeded fit-player sample for the PREREG_07 HumanSL pilot tests (not validation data)",
                   "seed": SEED, "per_level": PER_LEVEL, "levels": LEVELS, "seconds_load_rows": t_load,
                   "fit_rows": len(rows), "created_utc": H.utc_now(), "games": games}, f, ensure_ascii=False)
    print(json.dumps({"out": OUT, "games": len(games), "seconds_load_rows": round(t_load, 1)}))


if __name__ == "__main__":
    main()
