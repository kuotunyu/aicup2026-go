"""Shared, leak-checked access to the training data (split v1). Stdlib only.

Every training script (local or Colab) must read games through load_rows();
it keeps only split-v1 training players and refuses to return anything that
also appears in the local validation files.

    from c10data import load_rows
    rows = load_rows("data/training", "results/split_v1/players_split.csv",
                     part="fit", local_dir="data/local")
    # pandas users: df = pandas.DataFrame(rows)

part:
    "train" - all split-v1 training players (6,407 players, 793,863 games)
    "fit"   - training players minus the inner validation players
    "inner" - inner validation players (10% of training players, for choosing
              checkpoints / hyper-parameters; never the local validation set)
"""

import csv
import hashlib
import os
from collections import defaultdict

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
SALT = "c10-split-v1"
INNER_FRAC = 0.1
COLUMNS = ["player_id", "game_id", "rank", "color", "sgf_content"]
LOCAL_FILES_WITH_GAMES = ["rank_val_questions.csv", "player_val_questions.csv",
                          "player_val_candidates.csv"]

# Filled from the frozen split; load_rows asserts against these numbers.
EXPECTED = {
    "train": {"players": 6407, "games": 793863},
    "fit": {"players": 5767, "games": 720307},
    "inner": {"players": 640, "games": 73556},
}


class LeakError(RuntimeError):
    pass


def _h(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def read_split(split_csv):
    """player_id -> dict(rank, n_games, bin, split)."""
    with open(split_csv, newline="", encoding="utf-8") as f:
        return {r["player_id"]: {"rank": r["rank"], "n_games": int(r["n_games"]),
                                 "bin": r["bin"], "split": r["split"]}
                for r in csv.DictReader(f)}


def inner_players(split):
    """Inner validation players: 10% of each (rank, bin) stratum of the training players."""
    strata = defaultdict(list)
    for pid, info in split.items():
        if info["split"] == "train":
            strata[(info["rank"], info["bin"])].append(pid)
    inner = set()
    for pids in strata.values():
        pids.sort(key=lambda x: _h(SALT, "inner", x))
        inner.update(pids[:int(INNER_FRAC * len(pids) + 0.5)])
    return inner


def players_for(split, part):
    train = {pid for pid, info in split.items() if info["split"] == "train"}
    val = set(split) - train
    inner = inner_players(split)
    if inner & val:
        raise LeakError("inner validation players overlap the local validation set")
    chosen = {"train": train, "fit": train - inner, "inner": inner}[part]
    if chosen & val:
        raise LeakError(f"part {part!r} contains local validation players")
    return chosen


def _sgf_hashes_in_local(local_dir):
    csv.field_size_limit(10**8)
    hashes = set()
    for name in LOCAL_FILES_WITH_GAMES:
        path = os.path.join(local_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "sgf_content" in r:
                    hashes.add(hashlib.sha1(r["sgf_content"].encode("utf-8")).digest())
                else:
                    for i in range(1, int(r["num_games"]) + 1):
                        hashes.add(hashlib.sha1(r[f"sgf_{i}"].encode("utf-8")).digest())
    return hashes


def load_rows(train_dir, split_csv, part="fit", local_dir=None, check_counts=True):
    """Return the chosen training rows as a list of dicts (keys = COLUMNS).

    Files are named explicitly (train_<LEVEL>.csv); macOS '._train_*.csv'
    debris in the folder is never read.
    """
    csv.field_size_limit(10**8)
    split = read_split(split_csv)
    keep = players_for(split, part)
    rows = []
    for lv in LEVELS:
        with open(os.path.join(train_dir, f"train_{lv}.csv"), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if next(reader) != COLUMNS:
                raise ValueError(f"unexpected header in train_{lv}.csv")
            for rec in reader:
                if rec[0] in keep:
                    rows.append(dict(zip(COLUMNS, rec)))

    got = {"players": len({r["player_id"] for r in rows}), "games": len(rows)}
    if check_counts and got != EXPECTED[part]:
        raise LeakError(f"part {part!r}: expected {EXPECTED[part]}, got {got}")

    if local_dir is not None:
        local = _sgf_hashes_in_local(local_dir)
        if not local:
            raise LeakError(f"no validation files found in {local_dir!r}; cannot run the leak check")
        overlap = sum(1 for r in rows
                      if hashlib.sha1(r["sgf_content"].encode("utf-8")).digest() in local)
        if overlap:
            raise LeakError(f"{overlap} training games also appear in the local validation files")
    return rows


if __name__ == "__main__":
    import json
    import sys
    train_dir, split_csv, local_dir = sys.argv[1:4]
    report = {}
    for part in ("train", "fit", "inner"):
        rows = load_rows(train_dir, split_csv, part, local_dir, check_counts=False)
        per_rank = defaultdict(int)
        for r in rows:
            per_rank[r["rank"]] += 1
        report[part] = {"players": len({r["player_id"] for r in rows}), "games": len(rows),
                        "games_by_rank": {lv: per_rank[lv] for lv in LEVELS},
                        "matches_expected": {"players": len({r["player_id"] for r in rows}),
                                             "games": len(rows)} == EXPECTED[part]}
    print(json.dumps(report, indent=1))
