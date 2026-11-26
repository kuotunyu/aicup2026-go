"""prereg06 explore / katago: CPU check of KataGo's *python* input featurisation on our SGFs (no network weights).

What it measures (fit players only, through tools/c10data.load_rows; nothing under data/local is ever opened):
  1. Move-count statistics over ALL fit games: total moves per game and the number of moves made by the
     row's own player ("target moves"), overall and per level. These feed the compute estimate in REPORT.md.
  2. On a small seeded sample (default 6 games per level), replays every game with KataGo's python Board and
     builds the model input (Features.fill_row_features, 22 spatial + 19 global channels) at every position
     where the target player is to move, with Japanese rules / komi 6.5 (GameState.RULES_JAPANESE).
     Reports: illegal-move count, positions per second on ONE core, and the split between board play and
     feature filling.

Usage (repo root, Git Bash):
  .venv/Scripts/python.exe results/prereg06/explore/katago/kg_feat_bench.py --katago-python <clone>/python
Writes results/prereg06/explore/katago/kg_feat_bench.json (only file written).
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, HERE)

from kg_guard import install_guard  # noqa: E402  (makes any access under data/local raise)

install_guard()
from c10data import LEVELS, load_rows  # noqa: E402

MOVE_RE = re.compile(r";([BW])\[([a-s]{2})\]")


def parse_all_moves(sgf):
    return [(c, ord(p[0]) - 97, ord(p[1]) - 97) for c, p in MOVE_RE.findall(sgf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago-python", required=True, help="path to KataGo/python (clone, read-only)")
    ap.add_argument("--per-level", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--out", default=os.path.join(HERE, "kg_feat_bench.json"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.katago_python))
    from katago.game.board import Board
    from katago.game.features import Features
    from katago.game.gamestate import GameState

    t0 = time.time()
    rows = load_rows(os.path.join(REPO, "data", "training"),
                     os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                     part="fit", local_dir=None, check_counts=True)
    t_load = time.time() - t0

    # ---- 1. move-count statistics over all fit games
    tot = {lv: [] for lv in LEVELS}
    tgt = {lv: [] for lv in LEVELS}
    for r in rows:
        n = len(MOVE_RE.findall(r["sgf_content"]))
        k = (n + 1) // 2 if r["color"] == "B" else n // 2
        tot[r["rank"]].append(n)
        tgt[r["rank"]].append(k)
    all_tot = [x for lv in LEVELS for x in tot[lv]]
    all_tgt = [x for lv in LEVELS for x in tgt[lv]]
    stats = {
        "games": len(all_tot),
        "total_moves_mean": statistics.fmean(all_tot),
        "total_moves_median": statistics.median(all_tot),
        "target_moves_mean": statistics.fmean(all_tgt),
        "target_moves_median": statistics.median(all_tgt),
        "target_moves_p90": float(np.percentile(all_tgt, 90)),
        "target_moves_max": max(all_tgt),
        "per_level_target_moves_mean": {lv: statistics.fmean(tgt[lv]) for lv in LEVELS},
        "per_level_games": {lv: len(tgt[lv]) for lv in LEVELS},
    }

    # ---- 2. featurisation timing on a seeded sample
    rng = random.Random(args.seed)
    by_level = {lv: [r for r in rows if r["rank"] == lv] for lv in LEVELS}
    sample = []
    for lv in LEVELS:
        sample += rng.sample(by_level[lv], args.per_level)

    config = {"version": 15}  # Features only reads the version (22 spatial / 19 global channels for v10-v17)
    feats = Features(config, 19)
    rules = GameState.RULES_JAPANESE
    illegal = 0
    n_pos = 0
    t_play = 0.0
    t_feat = 0.0
    nonzero_ladder = 0
    per_game = []
    for r in sample:
        mv = parse_all_moves(r["sgf_content"])
        target = Board.BLACK if r["color"] == "B" else Board.WHITE
        board = Board(size=19)
        boards = [board.copy()]
        moves = []
        g_pos = 0
        g_t0 = time.perf_counter()
        ok = True
        for c, x, y in mv:
            pla = Board.BLACK if c == "B" else Board.WHITE
            if pla != board.pla:
                ok = False
                break
            if pla == target:
                opp = Board.get_opp(pla)
                bin_input = np.zeros((1, 19 * 19, 22), dtype=np.float32)
                glob_input = np.zeros((1, 19), dtype=np.float32)
                a = time.perf_counter()
                feats.fill_row_features(board, pla, opp, boards, moves, len(moves), rules, bin_input, glob_input, 0)
                t_feat += time.perf_counter() - a
                if bin_input[0, :, 14:18].any():
                    nonzero_ladder += 1
                g_pos += 1
            loc = board.loc(x, y)
            a = time.perf_counter()
            try:
                board.play(pla, loc)
            except Exception:
                illegal += 1
                ok = False
                break
            moves.append((pla, loc))
            boards.append(board.copy())
            t_play += time.perf_counter() - a
        n_pos += g_pos
        per_game.append({"rank": r["rank"], "color": r["color"], "moves": len(mv), "target_positions": g_pos,
                         "replayed_ok": ok, "seconds": time.perf_counter() - g_t0})

    out = {
        "what": "KataGo python featurisation on fit-player SGFs (no network weights); Japanese rules komi 6.5",
        "katago_python": os.path.abspath(args.katago_python),
        "seconds_load_rows": t_load,
        "move_stats_fit": stats,
        "sample_games": len(sample),
        "sample_seed": args.seed,
        "illegal_or_out_of_turn_games": sum(1 for g in per_game if not g["replayed_ok"]),
        "illegal_moves": illegal,
        "target_positions": n_pos,
        "positions_with_any_ladder_feature": nonzero_ladder,
        "seconds_feature_fill": t_feat,
        "seconds_board_play_and_copy": t_play,
        "positions_per_second_one_core_features_only": n_pos / t_feat if t_feat else None,
        "positions_per_second_one_core_including_play": n_pos / (t_feat + t_play) if t_feat else None,
        "per_game": per_game,
        "note": "single process on a shared desktop CPU; other jobs were running, so treat speed as a rough lower bound",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "per_game"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
