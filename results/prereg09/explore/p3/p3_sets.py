"""prereg09 P3 (CPU, exploration): a third probe batch for the P1 mapping -> data/prereg09/p3/sets.json.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg09/explore/p3/p3_sets.py

Why (results/prereg08/p1/p1_explore_report.json E2, after the P1 gate): the TO mapping's player-grouped CV log-loss
was still falling at 10k / 20k / 30k probe games (1.763 / 1.709 / 1.679). P2 adds PER_LEVEL more fit games per level,
none of them in the P1 probe, same round-robin rule (p1_extract.build_probe) with its own seed.
The file has the P1 sets.json layout (format 'c10-prereg08 p1 sets v1') so p1_extract.py runs it unchanged:
    .venv-gpu/Scripts/python.exe results/prereg08/p1/p1_extract.py run --set probe \
        --sets data/prereg08/p2/sets.json --out-dir data/prereg08/p2
bank / mono are copied from the P1 sets.json (not extracted again). Fit players only; never inner, never validation.
"""

import json
import os
import sys

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(REPO, "results", "prereg08", "p1"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import p1_extract as E  # noqa: E402

PER_LEVEL = 12000
SEED = 20261009
OUT = os.path.join(REPO, "data", "prereg09", "p3", "sets.json")


def main():
    from c10data import load_rows
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; not overwritten")
    s1 = E.load_sets(E.SETS)
    taken = {(g["rank"], g["game_id"]) for g in s1["probe"]["games"]}
    P2 = os.path.join(REPO, "data", "prereg08", "p2", "sets.json")
    s2 = E.load_sets(P2)
    taken |= {(g["rank"], g["game_id"]) for g in s2["probe"]["games"]}
    rows = load_rows(E.TRAIN_DIR, E.SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    rows = [r for r in rows if (r["rank"], r["game_id"]) not in taken]
    probe = E.build_probe(rows, per_level=PER_LEVEL, seed=SEED)
    inner = {r["player_id"] for r in load_rows(E.TRAIN_DIR, E.SPLIT_CSV, part="inner", local_dir=None,
                                               check_counts=True)}
    assert not ({g["player_id"] for g in probe} & inner), "inner player in P3 probe"
    assert not ({(g["rank"], g["game_id"]) for g in probe} & taken), "P1/P2 probe game in P3 probe"
    assert len({(g["rank"], g["game_id"]) for g in probe}) == len(probe), "duplicate game"
    per_player = {}
    for g in probe:
        per_player[(g["rank"], g["player_id"])] = per_player.get((g["rank"], g["player_id"]), 0) + 1
    out = {"format": E.SETS_FORMAT, "created_utc": E.H.utc_now(),
           "probe": {"source": "fit", "seed": SEED, "per_level": PER_LEVEL, "games": probe,
                     "excludes": "every game of data/prereg08/p1/sets.json and data/prereg08/p2/sets.json probes"},
           "bank": s1["bank"], "mono": s1["mono"],
           "inputs_sha256": {**s1["inputs_sha256"], "p1_sets.json": E.P.sha_raw(E.SETS), "p2_sets.json": E.P.sha_raw(P2)},
           "counts": {"probe": len(probe), "probe_players": len(per_player),
                      "probe_games_per_player_max": max(per_player.values()),
                      "bank": s1["counts"]["bank"], "mono": s1["counts"]["mono"]}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    E.write_json_atomic(OUT, out)
    print(json.dumps({"written": OUT, "sha256": E.P.sha_raw(OUT), "counts": out["counts"]}, indent=1))


if __name__ == "__main__":
    main()
