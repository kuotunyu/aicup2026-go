# -*- coding: utf-8 -*-
"""Supplementary stats: move-count floor, last mover vs row color, per-level concentration."""
import csv
import json
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

DATA_DIR = r"D:\AI-Portfolio\competitions\aicup2026-go\data\training"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
MOVE = re.compile(r";([BW])\[")


def work(level):
    csv.field_size_limit(10 ** 8)
    path = os.path.join(DATA_DIR, "train_%s.csv" % level)
    nm_counter = Counter()
    last_is_row_color = 0
    parity_even = 0
    n = 0
    pc = Counter()
    first_moves = Counter()
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for pid, gid, rank, color, sgf in rd:
            n += 1
            moves = MOVE.findall(sgf)
            nm = len(moves)
            nm_counter[nm] += 1
            if moves and moves[-1] == color:
                last_is_row_color += 1
            if nm % 2 == 0:
                parity_even += 1
            pc[pid] += 1
            i = sgf.find(";B[")
            first_moves[sgf[i + 3:i + 5]] += 1
    tot = sorted(pc.values(), reverse=True)
    k10 = max(1, len(tot) // 10)
    return {
        "level": level,
        "n": n,
        "eq100": nm_counter[100],
        "le110": sum(v for k, v in nm_counter.items() if k <= 110),
        "ge300": sum(v for k, v in nm_counter.items() if k >= 300),
        "ge350": sum(v for k, v in nm_counter.items() if k >= 350),
        "lt100": sum(v for k, v in nm_counter.items() if k < 100),
        "last_mover_is_row_color": last_is_row_color,
        "even_move_count": parity_even,
        "top10pct_players_share": round(sum(tot[:k10]) / n, 4),
        "top1_player_share": round(tot[0] / n, 4),
        "players_lt5": sum(1 for v in tot if v < 5),
        "players_lt10": sum(1 for v in tot if v < 10),
        "first_move_top3": first_moves.most_common(3),
        "distinct_first_moves": len(first_moves),
    }


if __name__ == "__main__":
    with Pool(10) as p:
        res = p.map(work, LEVELS)
    with open(os.path.join(OUT_DIR, "recompute_extra_out.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for r in res:
        print(json.dumps(r))
