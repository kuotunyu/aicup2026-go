"""Inventory of the official training set (stdlib only).

Usage:
    python tools/inventory.py data/training results/data_inventory.json

Reads every train_*.csv, checks the columns and labels, and summarises
players, colours, SGF properties, move counts and duplicate games.
Only aggregate numbers are written; no game content leaves this machine.
"""

import csv
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
EXPECTED_COLUMNS = ["player_id", "game_id", "rank", "color", "sgf_content"]

MOVE_RE = re.compile(r";([BW])\[([a-t]{2})?\]")  # "tt" or empty = pass
PROP_RE = re.compile(r"(?<![A-Za-z])([A-Z]{1,2})\[")
SZ_RE = re.compile(r"SZ\[(\d+)\]")


def quantiles(values, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {f"p{int(q * 100)}": s[min(n - 1, int(q * (n - 1) + 0.5))] for q in qs}


def main(train_dir, out_path):
    csv.field_size_limit(10**8)
    files = sorted(glob.glob(os.path.join(train_dir, "train_*.csv")))
    report = {"files": {}, "overall": {}}

    player_ranks = defaultdict(set)          # player_id -> ranks seen
    player_games = Counter()                 # player_id -> games (all files)
    game_id_files = Counter()                # game_id -> number of files using it
    sgf_owners = defaultdict(list)           # sha1(sgf) -> [(rank, player, colour)]
    total_rows = 0

    for path in files:
        level = os.path.basename(path)[len("train_"):-len(".csv")]
        rows = 0
        header = None
        rank_mismatch = 0
        bad_colour = 0
        game_ids = Counter()
        players = Counter()
        colours = Counter()
        sizes = Counter()
        props = Counter()
        moves = []
        passes = 0
        non_alternating = 0
        first_colour = Counter()
        empty_sgf = 0
        player_colour = defaultdict(Counter)
        colour_runs = 0
        prev_colour = None

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for rec in reader:
                rows += 1
                pid, gid, rank, colour, sgf = rec
                if rank != level:
                    rank_mismatch += 1
                if colour not in ("B", "W"):
                    bad_colour += 1
                game_ids[gid] += 1
                players[pid] += 1
                colours[colour] += 1
                if colour != prev_colour:
                    colour_runs += 1
                    prev_colour = colour
                player_colour[pid][colour] += 1
                player_ranks[pid].add(rank)
                player_games[pid] += 1

                if not sgf.strip():
                    empty_sgf += 1
                    continue
                m = SZ_RE.search(sgf)
                sizes[m.group(1) if m else "none"] += 1
                for p in set(PROP_RE.findall(sgf)):
                    props[p] += 1
                mv = MOVE_RE.findall(sgf)
                moves.append(len(mv))
                if mv:
                    first_colour[mv[0][0]] += 1
                passes += sum(1 for c, xy in mv if not xy or xy == "tt")
                if any(mv[i][0] == mv[i + 1][0] for i in range(len(mv) - 1)):
                    non_alternating += 1
                digest = hashlib.sha1(sgf.encode("utf-8")).digest()
                sgf_owners[digest].append((rank, pid, colour))

        total_rows += rows
        for gid in game_ids:
            game_id_files[gid] += 1
        gpp = list(players.values())
        mixed_colour_players = sum(1 for c in player_colour.values() if len(c) == 2)
        report["files"][level] = {
            "header": header,
            "header_ok": header == EXPECTED_COLUMNS,
            "rows": rows,
            "rank_label_mismatch": rank_mismatch,
            "bad_colour_values": bad_colour,
            "duplicate_game_ids": sum(1 for v in game_ids.values() if v > 1),
            "unique_players": len(players),
            "games_per_player": quantiles(gpp),
            "players_with_at_least": {
                str(k): sum(1 for v in gpp if v >= k) for k in (5, 20, 100, 200)
            },
            "colour_counts": dict(colours),
            "colour_runs_in_row_order": colour_runs,
            "players_with_both_colours": mixed_colour_players,
            "board_sizes": dict(sizes),
            "sgf_properties_share": {k: round(v / rows, 4) for k, v in props.most_common()},
            "moves_per_game": quantiles(moves),
            "games_under_20_moves": sum(1 for v in moves if v < 20),
            "pass_moves_total": passes,
            "non_alternating_games": non_alternating,
            "first_move_colour": dict(first_colour),
            "empty_sgf": empty_sgf,
        }
        print(f"{level}: {rows} rows, {len(players)} players", flush=True)

    # Duplicate games across the whole set
    dup_groups = [v for v in sgf_owners.values() if len(v) > 1]
    dup_kinds = Counter()
    for owners in dup_groups:
        pids = {o[1] for o in owners}
        cols = {o[2] for o in owners}
        ranks = {o[0] for o in owners}
        if len(pids) == 1:
            dup_kinds["same_player"] += 1
        elif cols == {"B", "W"}:
            dup_kinds["both_sides_of_one_game"] += 1
        else:
            dup_kinds["different_players_same_colour"] += 1
        if len(ranks) > 1:
            dup_kinds["spans_two_ranks"] += 1

    multi_rank_players = sum(1 for r in player_ranks.values() if len(r) > 1)
    all_gpp = list(player_games.values())
    report["overall"] = {
        "files_found": [os.path.basename(p) for p in files],
        "levels_missing": [lv for lv in LEVELS if lv not in report["files"]],
        "total_rows": total_rows,
        "unique_players": len(player_games),
        "players_in_more_than_one_rank": multi_rank_players,
        "games_per_player": quantiles(all_gpp),
        "players_with_at_least": {
            str(k): sum(1 for v in all_gpp if v >= k) for k in (5, 20, 100, 200)
        },
        "game_ids_used_in_more_than_one_file": sum(1 for v in game_id_files.values() if v > 1),
        "unique_sgf": len(sgf_owners),
        "duplicate_sgf_groups": len(dup_groups),
        "duplicate_sgf_kinds": dict(dup_kinds),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
