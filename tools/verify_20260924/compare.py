# -*- coding: utf-8 -*-
"""Compare my independent recomputation with results/data_inventory.json (read-only)."""
import csv
import json
import os
from collections import Counter
from multiprocessing import Pool

DATA_DIR = r"D:\AI-Portfolio\competitions\aicup2026-go\data\training"
INV = r"D:\AI-Portfolio\competitions\aicup2026-go\results\data_inventory.json"
HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def player_counts(level):
    csv.field_size_limit(10 ** 8)
    pc = Counter()
    with open(os.path.join(DATA_DIR, "train_%s.csv" % level), encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            pc[row[0]] += 1
    v = list(pc.values())
    return level, {str(k): sum(1 for x in v if x >= k) for k in (5, 20, 100, 200)}


if __name__ == "__main__":
    inv = json.load(open(INV, encoding="utf-8"))
    mine = json.load(open(os.path.join(HERE, "recompute_out.json"), encoding="utf-8"))
    with Pool(10) as p:
        thr = dict(p.map(player_counts, LEVELS))
    rows = []

    def chk(item, iv, mv):
        rows.append((item, iv, mv, iv == mv))

    qmap = {"p0": "min", "p10": "p10_inv", "p25": "p25_inv", "p50": "median_inv_def",
            "p75": "p75_inv", "p90": "p90_inv", "p100": "max"}
    for lv in LEVELS:
        a = inv["files"][lv]
        b = mine["per_file"][lv]
        chk(lv + " header", a["header"], b["header"])
        chk(lv + " rows", a["rows"], b["n_rows"])
        chk(lv + " rank_mismatch", a["rank_label_mismatch"], sum(v for k, v in b["rank_values"].items() if k != lv))
        chk(lv + " bad_colour", a["bad_colour_values"], sum(v for k, v in b["color_values"].items() if k not in ("B", "W")))
        chk(lv + " colour_counts", a["colour_counts"], b["color_values"])
        chk(lv + " unique_players", a["unique_players"], b["n_players"])
        chk(lv + " players_both_colours", a["players_with_both_colours"], b["players_mixed_colors"])
        chk(lv + " games_per_player quantiles", a["games_per_player"], {k: b["games_per_player"][v] for k, v in qmap.items()})
        chk(lv + " players_with_at_least", a["players_with_at_least"], thr[lv])
        chk(lv + " moves quantiles", a["moves_per_game"], {k: b["moves"][v] for k, v in qmap.items()})
        chk(lv + " pass_moves_total", a["pass_moves_total"], b["pass_moves_total"])
        chk(lv + " non_alternating", a["non_alternating_games"], b["n_rows"] - b["strict_alternating"])
        chk(lv + " first_move_colour", a["first_move_colour"], {"B": b["start_black"]} if not b["start_white"] else {"B": b["start_black"], "W": b["start_white"]})
        chk(lv + " board_sizes", a["board_sizes"], b["sz_values"])
        chk(lv + " props", sorted(a["sgf_properties_share"].items()), sorted((k, round(v / b["n_rows"], 4)) for k, v in b["prop_games"].items()))
        chk(lv + " duplicate_game_ids(in-file)", a["duplicate_game_ids"], 0 if b["game_id_num_contiguous"] and b["game_id_num_max"] - b["game_id_num_min"] + 1 == b["n_rows"] else "?")
        chk(lv + " games_under_20_moves", a["games_under_20_moves"], 0 if b["moves"]["min"] >= 20 else "?")
    o = inv["overall"]
    m = mine["overall"]
    chk("overall total_rows", o["total_rows"], m["n_rows"])
    chk("overall unique_players", o["unique_players"], m["n_players"])
    chk("overall players_in_more_than_one_rank", o["players_in_more_than_one_rank"], m["players_in_2plus_levels"])
    chk("overall games_per_player", o["games_per_player"], {k: m["games_per_player"][v] for k, v in qmap.items()})
    chk("overall players_with_at_least", o["players_with_at_least"], {str(k): m["players_ge_%d" % k] for k in (5, 20, 100, 200)})
    chk("overall unique_sgf", o["unique_sgf"], m["n_rows"] - m["dup_full_sgf"]["redundant_rows"])
    chk("overall duplicate_sgf_groups", o["duplicate_sgf_groups"], m["dup_full_sgf"]["n_groups"])
    for r in rows:
        print("OK " if r[3] else "XX ", r[0], "| inv:", json.dumps(r[1]), "| mine:", json.dumps(r[2]))
    print("mismatches:", sum(1 for r in rows if not r[3]), "of", len(rows))
    json.dump(rows, open(os.path.join(HERE, "compare_out.json"), "w", encoding="utf-8"), indent=1)
