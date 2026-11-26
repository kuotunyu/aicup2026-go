# -*- coding: utf-8 -*-
"""Independent recomputation of AI CUP 2026 Go training-set statistics.

Standard library only. CPU only. Reads the 10 train_<LEVEL>.csv files and writes
recompute_out.json next to this script. Does not modify anything in the repo.
"""
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

DATA_DIR = r"D:\AI-Portfolio\competitions\aicup2026-go\data\training"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]

# Fast path: root node with simple props, then only single-prop B/W move nodes.
FAST = re.compile(r"\(;((?:[A-Z]+\[[^\]\\]*\])+)((?:;[BW]\[[^\]\\]*\])*)\)")
ROOTPROP = re.compile(r"([A-Z]+)\[([^\]\\]*)\]")
MOVE = re.compile(r";([BW])\[([^\]\\]*)\]")
# Generic tokenizer (slow path)
TOKEN = re.compile(r"\s*(?:(;)|(\()|(\))|([A-Za-z]+)((?:\s*\[(?:[^\]\\]|\\.)*\])+))", re.S)
VALS = re.compile(r"\[((?:[^\]\\]|\\.)*)\]", re.S)
GAMEID = re.compile(r"g_(\d+)")
PID = re.compile(r"[0-9a-f]{16}")


def parse_sgf(s):
    """Return (nodes, n_open_paren, garbage) where nodes = list of list[(ident, [values])]."""
    m = FAST.fullmatch(s)
    if m:
        root = [(k, [v]) for k, v in ROOTPROP.findall(m.group(1))]
        nodes = [root] + [[(c, [v])] for c, v in MOVE.findall(m.group(2))]
        return nodes, 1, False, True
    nodes = []
    pos = 0
    n = len(s)
    nparen = 0
    garbage = False
    while pos < n:
        t = TOKEN.match(s, pos)
        if not t:
            if s[pos:].strip():
                garbage = True
            break
        pos = t.end()
        if t.group(1):
            nodes.append([])
        elif t.group(2):
            nparen += 1
        elif t.group(3):
            pass
        else:
            ident = t.group(4)
            vals = VALS.findall(t.group(5))
            if not nodes:
                nodes.append([])
                garbage = True
            nodes[-1].append((ident, vals))
    return nodes, nparen, garbage, False


def is_pass(v, sz):
    return v == "" or (v == "tt" and sz <= 19)


def valid_coord(v, sz):
    if len(v) != 2:
        return False
    a, b = ord(v[0]) - 97, ord(v[1]) - 97
    return 0 <= a < sz and 0 <= b < sz


def work(level):
    path = os.path.join(DATA_DIR, "train_%s.csv" % level)
    csv.field_size_limit(10 ** 8)
    t0 = time.time()
    res = {
        "level": level,
        "file_bytes": os.path.getsize(path),
        "header": None,
        "n_rows": 0,
        "bad_len_rows": 0,
        "rank_counter": Counter(),
        "color_counter": Counter(),
        "player_counts": Counter(),
        "player_color": defaultdict(Counter),
        "moves": [],
        "moves_by_color_row": {"B": [], "W": []},
        "games_with_pass": 0,
        "pass_moves_total": 0,
        "games_pass_only_at_end": 0,
        "games_ending_two_passes": 0,
        "games_ending_one_pass": 0,
        "start_black": 0,
        "start_white": 0,
        "zero_moves": 0,
        "strict_alternating": 0,
        "alt_violation_examples": [],
        "prop_games": Counter(),       # ident -> games containing it (any node)
        "prop_root_games": Counter(),  # ident -> games whose root node contains it
        "prop_nonroot_nonmove": Counter(),
        "sz_values": Counter(),
        "invalid_coord_games": 0,
        "multi_move_nodes_games": 0,
        "root_has_move_games": 0,
        "nonmove_nodes_after_root_games": 0,
        "variation_games": 0,
        "garbage_games": 0,
        "fast_path": 0,
        "sgf_has_whitespace": 0,
        "sgf_leading_trailing_ws": 0,
        "sgf_not_paren": 0,
        "game_ids": [],
        "bad_game_id_format": 0,
        "bad_player_id_format": 0,
        "player_noncontiguous": 0,
        "records": [],  # (sha1_full, sha1_moves, player_id, game_id, color, n_moves)
        "move_seq_len_minus_nodes": 0,
    }
    last_player = None
    seen_players_closed = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        res["header"] = next(rd)
        for row in rd:
            res["n_rows"] += 1
            if len(row) != 5:
                res["bad_len_rows"] += 1
                continue
            pid, gid, rank, color, sgf = row
            res["rank_counter"][rank] += 1
            res["color_counter"][color] += 1
            res["player_counts"][pid] += 1
            res["player_color"][pid][color] += 1
            if not PID.fullmatch(pid):
                res["bad_player_id_format"] += 1
            gm = GAMEID.fullmatch(gid)
            if not gm:
                res["bad_game_id_format"] += 1
            res["game_ids"].append(gid)
            # contiguity of player rows
            if pid != last_player:
                if pid in seen_players_closed:
                    res["player_noncontiguous"] += 1
                if last_player is not None:
                    seen_players_closed.add(last_player)
                last_player = pid
            if sgf != sgf.strip():
                res["sgf_leading_trailing_ws"] += 1
            if any(ch in sgf for ch in "\n\r\t "):
                res["sgf_has_whitespace"] += 1
            if not (sgf.startswith("(") and sgf.rstrip().endswith(")")):
                res["sgf_not_paren"] += 1

            nodes, nparen, garbage, fast = parse_sgf(sgf)
            if fast:
                res["fast_path"] += 1
            if garbage:
                res["garbage_games"] += 1
            if nparen > 1:
                res["variation_games"] += 1
            idents_all = set()
            idents_root = set()
            sz = 19
            if nodes:
                for ident, vals in nodes[0]:
                    idents_root.add(ident)
                    if ident == "SZ":
                        res["sz_values"][vals[0] if vals else ""] += 1
                        try:
                            sz = int(vals[0])
                        except Exception:
                            pass
            if "SZ" not in idents_root:
                res["sz_values"]["<missing>"] += 1
            seq = []
            multi = False
            nonmove_after_root = False
            invalid = False
            npass = 0
            pass_positions = []
            for ni, node in enumerate(nodes):
                mv = [(k, v) for k, v in node if k in ("B", "W")]
                for k, _ in node:
                    idents_all.add(k)
                    if ni > 0 and k not in ("B", "W"):
                        res["prop_nonroot_nonmove"][k] += 1
                if ni == 0 and mv:
                    res["root_has_move_games"] += 1
                if len(mv) > 1:
                    multi = True
                if ni > 0 and not mv:
                    nonmove_after_root = True
                if mv:
                    k, vals = mv[0]
                    v = vals[0] if vals else ""
                    seq.append(k)
                    if is_pass(v, sz):
                        npass += 1
                        pass_positions.append(len(seq) - 1)
                    elif not valid_coord(v, sz):
                        invalid = True
            for k in idents_all:
                res["prop_games"][k] += 1
            for k in idents_root:
                res["prop_root_games"][k] += 1
            if multi:
                res["multi_move_nodes_games"] += 1
            if nonmove_after_root:
                res["nonmove_nodes_after_root_games"] += 1
            if invalid:
                res["invalid_coord_games"] += 1
            nm = len(seq)
            res["moves"].append(nm)
            if color in ("B", "W"):
                res["moves_by_color_row"][color].append(nm)
            if nm == 0:
                res["zero_moves"] += 1
            else:
                if seq[0] == "B":
                    res["start_black"] += 1
                else:
                    res["start_white"] += 1
            alt = all(seq[i] == ("B" if i % 2 == 0 else "W") for i in range(nm))
            if alt:
                res["strict_alternating"] += 1
            elif len(res["alt_violation_examples"]) < 5:
                # first index of violation
                bad = next(i for i in range(nm) if seq[i] != ("B" if i % 2 == 0 else "W"))
                res["alt_violation_examples"].append({"game_id": gid, "n_moves": nm, "first_bad_index": bad})
            if npass:
                res["games_with_pass"] += 1
                res["pass_moves_total"] += npass
                if all(p >= nm - 2 for p in pass_positions):
                    res["games_pass_only_at_end"] += 1
                if nm >= 1 and pass_positions and pass_positions[-1] == nm - 1:
                    if nm >= 2 and (nm - 2) in pass_positions:
                        res["games_ending_two_passes"] += 1
                    else:
                        res["games_ending_one_pass"] += 1
            h_full = hashlib.sha1(sgf.encode("utf-8")).hexdigest()[:20]
            h_moves = hashlib.sha1("".join("%s%s" % (k, v[0] if v else "") for node in nodes[1:] for k, v in node if k in ("B", "W")).encode()).hexdigest()[:20]
            res["records"].append((h_full, h_moves, pid, gid, color, nm))
    res["seconds"] = round(time.time() - t0, 1)
    # convert for pickling
    res["player_color"] = {k: dict(v) for k, v in res["player_color"].items()}
    return res


def inv_q(sorted_vals, q):
    n = len(sorted_vals)
    return sorted_vals[int(q * (n - 1) + 0.5)]


def dist(vals):
    s = sorted(vals)
    return {
        "n": len(s),
        "min": s[0],
        "p10_inv": inv_q(s, 0.10),
        "p25_inv": inv_q(s, 0.25),
        "median_inv_def": inv_q(s, 0.5),
        "median_statistics": statistics.median(s),
        "p75_inv": inv_q(s, 0.75),
        "p90_inv": inv_q(s, 0.90),
        "max": s[-1],
        "mean": round(sum(s) / len(s), 3),
    }


def main():
    t0 = time.time()
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        results = pool.map(work, LEVELS)
    out = {"per_file": {}, "overall": {}}
    player_levels = defaultdict(set)
    player_total = Counter()
    all_gids = Counter()
    gid_level = {}
    full_groups = defaultdict(list)
    move_groups = defaultdict(list)
    prop_games = Counter()
    for r in results:
        lv = r["level"]
        pc = r["player_counts"]
        for p, c in pc.items():
            player_levels[p].add(lv)
            player_total[p] += c
        for g in r["game_ids"]:
            all_gids[g] += 1
            gid_level[g] = lv
        for (hf, hm, pid, gid, color, nm) in r["records"]:
            full_groups[hf].append((lv, pid, gid, color, nm))
            move_groups[hm].append((lv, pid, gid, color, nm))
        prop_games.update(r["prop_games"])
        gnums = sorted(int(g[2:]) for g in r["game_ids"] if GAMEID.fullmatch(g))
        per_player_counts = list(pc.values())
        # per-player color share
        pcol = r["player_color"]
        mixed = sum(1 for d in pcol.values() if len(d) > 1)
        only_b = sum(1 for d in pcol.values() if set(d) == {"B"})
        only_w = sum(1 for d in pcol.values() if set(d) == {"W"})
        out["per_file"][lv] = {
            "file_bytes": r["file_bytes"],
            "header": r["header"],
            "n_rows": r["n_rows"],
            "bad_len_rows": r["bad_len_rows"],
            "rank_values": dict(r["rank_counter"]),
            "rank_all_equal_level": set(r["rank_counter"]) == {lv},
            "color_values": dict(r["color_counter"]),
            "n_players": len(pc),
            "games_per_player": dist(per_player_counts),
            "players_mixed_colors": mixed,
            "players_only_B": only_b,
            "players_only_W": only_w,
            "moves": dist(r["moves"]),
            "moves_when_row_color_B": dist(r["moves_by_color_row"]["B"]) if r["moves_by_color_row"]["B"] else None,
            "moves_when_row_color_W": dist(r["moves_by_color_row"]["W"]) if r["moves_by_color_row"]["W"] else None,
            "zero_move_games": r["zero_moves"],
            "games_with_pass": r["games_with_pass"],
            "pass_moves_total": r["pass_moves_total"],
            "games_pass_only_in_last_two": r["games_pass_only_at_end"],
            "games_ending_two_passes": r["games_ending_two_passes"],
            "games_ending_one_pass": r["games_ending_one_pass"],
            "start_black": r["start_black"],
            "start_white": r["start_white"],
            "strict_alternating": r["strict_alternating"],
            "alt_violation_examples": r["alt_violation_examples"],
            "prop_games": dict(r["prop_games"]),
            "prop_root_games": dict(r["prop_root_games"]),
            "prop_nonroot_nonmove": dict(r["prop_nonroot_nonmove"]),
            "sz_values": dict(r["sz_values"]),
            "invalid_coord_games": r["invalid_coord_games"],
            "multi_move_nodes_games": r["multi_move_nodes_games"],
            "root_has_move_games": r["root_has_move_games"],
            "nonmove_nodes_after_root_games": r["nonmove_nodes_after_root_games"],
            "variation_games": r["variation_games"],
            "garbage_games": r["garbage_games"],
            "fast_path_games": r["fast_path"],
            "sgf_has_whitespace": r["sgf_has_whitespace"],
            "sgf_leading_trailing_ws": r["sgf_leading_trailing_ws"],
            "sgf_not_paren": r["sgf_not_paren"],
            "bad_game_id_format": r["bad_game_id_format"],
            "bad_player_id_format": r["bad_player_id_format"],
            "player_rows_noncontiguous_events": r["player_noncontiguous"],
            "game_id_num_min": gnums[0] if gnums else None,
            "game_id_num_max": gnums[-1] if gnums else None,
            "game_id_num_contiguous": (gnums[-1] - gnums[0] + 1 == len(gnums)) if gnums else None,
            "seconds": r["seconds"],
        }
    # overall
    tot = sorted(player_total.values())
    multi_level = {p: sorted(l, key=LEVELS.index) for p, l in player_levels.items() if len(l) > 1}
    ml_counter = Counter(len(l) for l in multi_level.values())
    ml_pairs = Counter(tuple(l) for l in multi_level.values())
    out["overall"]["n_rows"] = sum(r["n_rows"] for r in results)
    out["overall"]["n_players"] = len(player_total)
    out["overall"]["players_in_2plus_levels"] = len(multi_level)
    out["overall"]["players_by_n_levels"] = dict(ml_counter)
    out["overall"]["multi_level_combos_top"] = [["|".join(k), v] for k, v in ml_pairs.most_common(15)]
    out["overall"]["games_per_player"] = dist(tot)
    for th in (5, 20, 100, 200):
        out["overall"]["players_ge_%d" % th] = sum(1 for v in tot if v >= th)
    for th in (1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000):
        out["overall"]["players_ge_hist_%d" % th] = sum(1 for v in tot if v >= th)
    out["overall"]["top10_player_game_counts"] = sorted(tot, reverse=True)[:10]
    out["overall"]["share_rows_from_top1pct_players"] = round(sum(sorted(tot, reverse=True)[: max(1, len(tot) // 100)]) / sum(tot), 4)
    # game ids
    dup_gid = {g: c for g, c in all_gids.items() if c > 1}
    out["overall"]["n_unique_game_ids"] = len(all_gids)
    out["overall"]["game_ids_repeated"] = len(dup_gid)
    # properties overall
    out["overall"]["prop_games"] = dict(prop_games)
    # duplicates: full sgf
    def dup_summary(groups):
        g2 = [v for v in groups.values() if len(v) > 1]
        size_c = Counter(len(v) for v in g2)
        same_level = sum(1 for v in g2 if len({x[0] for x in v}) == 1)
        cross_level = len(g2) - same_level
        same_player = sum(1 for v in g2 if len({x[1] for x in v}) == 1)
        diff_player = len(g2) - same_player
        mixed_color = sum(1 for v in g2 if len({x[3] for x in v}) > 1)
        same_player_same_color = sum(1 for v in g2 if len({x[1] for x in v}) == 1 and len({x[3] for x in v}) == 1)
        bw_pair_diff_player = sum(1 for v in g2 if len(v) == 2 and {x[3] for x in v} == {"B", "W"} and v[0][1] != v[1][1])
        nm = sorted(v[0][4] for v in g2)
        rows_in = sum(len(v) for v in g2)
        extra_rows = sum(len(v) - 1 for v in g2)
        # level-distance for cross-level
        lvl_gap = Counter()
        for v in g2:
            ls = sorted({LEVELS.index(x[0]) for x in v})
            if len(ls) > 1:
                lvl_gap[ls[-1] - ls[0]] += 1
        per_level_rows = Counter(x[0] for v in g2 for x in v)
        return {
            "n_groups": len(g2),
            "rows_in_groups": rows_in,
            "redundant_rows": extra_rows,
            "group_size_counts": dict(size_c),
            "groups_same_level": same_level,
            "groups_cross_level": cross_level,
            "cross_level_max_gap_counts": dict(lvl_gap),
            "groups_same_player": same_player,
            "groups_diff_players": diff_player,
            "groups_mixed_color": mixed_color,
            "groups_same_player_same_color": same_player_same_color,
            "size2_groups_BW_diff_player": bw_pair_diff_player,
            "moves_in_dup_games": dist(nm) if nm else None,
            "rows_in_groups_per_level": dict(per_level_rows),
        }
    out["overall"]["dup_full_sgf"] = dup_summary(full_groups)
    out["overall"]["dup_move_sequence_only"] = dup_summary(move_groups)
    # number of players touched by cross-player duplicates
    tp = set()
    for v in full_groups.values():
        if len(v) > 1 and len({x[1] for x in v}) > 1:
            for x in v:
                tp.add(x[1])
    out["overall"]["players_touched_by_cross_player_dups"] = len(tp)
    out["seconds_total"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "recompute_out.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=list)
    print(json.dumps(out["overall"], ensure_ascii=False, indent=1, default=list))
    print("total seconds", out["seconds_total"])


if __name__ == "__main__":
    main()
