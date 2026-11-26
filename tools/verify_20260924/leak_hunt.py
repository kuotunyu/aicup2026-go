"""Leak-hunt verification of the AI CUP 2026 Go training set (stdlib only, CPU only).

Usage:
    python leak_hunt.py pass1      # parse all rows -> pass1.pkl (hashes + per-row metadata)
    python leak_hunt.py analyze    # groups, prefix pairs, row order, ids, moves, pool sizes -> leak_hunt.json

Only aggregate numbers are written. No game content is printed or saved
(pass 2 keeps move strings of candidate games in memory only).
"""

import csv
import hashlib
import json
import os
import pickle
import random
import re
import statistics
import sys
import time
from array import array
from collections import Counter, defaultdict

DATA = r"D:\AI-Portfolio\competitions\aicup2026-go\data\training"
OUT = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
PREFIXES = [10, 20, 30, 40, 50, 60, 80, 100]

WS = re.compile(r"\s+")
MOVE = re.compile(r";([BWbw])\[([A-Za-z]*)\]")
GID = re.compile(r"^g_(\d+)$")
FLIP = str.maketrans("abcdefghijklmnopqrs", "srqponmlkjihgfedcba")


def h8(s):
    return int.from_bytes(hashlib.blake2b(s.encode("ascii"), digest_size=8).digest(), "little")


def cands(xs, ys, fx, fy, n):
    X, Y, FX, FY = xs[:n], ys[:n], fx[:n], fy[:n]
    # the 8 symmetries of the board (dihedral group D4) on (x-string, y-string)
    return (X + Y, FX + Y, X + FY, FX + FY, Y + X, FY + X, Y + FX, FY + FX)


def transforms(xs, ys):
    fx, fy = xs.translate(FLIP), ys.translate(FLIP)
    return [(xs, ys), (fx, ys), (xs, fy), (fx, fy), (ys, xs), (fy, xs), (ys, fx), (fy, fx)]


def split_xy(seq):
    return seq[0::2], seq[1::2]


def quant(v, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    s = sorted(v)
    n = len(s)
    if not n:
        return {}
    return {f"p{int(q * 100)}": s[min(n - 1, int(q * (n - 1) + 0.5))] for q in qs}


# --------------------------------------------------------------------------- pass 1
def pass1():
    csv.field_size_limit(10**8)
    t0 = time.time()
    lvl = array("b")
    pidx = array("i")
    colr = array("b")
    nmov = array("H")
    gidn = array("i")
    rowi = array("i")
    lastmine = array("b")
    raw_full = array("Q")
    can_full = array("Q")
    raw_p = {n: array("Q") for n in PREFIXES}
    can_p = {n: array("Q") for n in PREFIXES}
    players = {}
    plist = []
    fmt = Counter()
    for li, level in enumerate(LEVELS):
        path = os.path.join(DATA, f"train_{level}.csv")
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            fmt[f"header_{level}"] = int(header == ["player_id", "game_id", "rank", "color", "sgf_content"])
            for ri, rec in enumerate(r):
                if len(rec) != 5:
                    fmt["bad_field_count"] += 1
                    continue
                pid, gid, rank, color, sgf = rec
                if rank != level:
                    fmt["rank_mismatch"] += 1
                t = WS.sub("", sgf)
                if t != sgf:
                    fmt["sgf_had_whitespace"] += 1
                mv = MOVE.findall(t)
                n = len(mv)
                if n:
                    cs, xy = zip(*mv)
                    cols = "".join(cs)
                    seq0 = "".join(xy)
                else:
                    cols, seq0 = "", ""
                seq = seq0.lower()
                if seq != seq0:
                    fmt["uppercase_coords"] += 1
                if cols != cols.upper():
                    fmt["lowercase_colour_letters"] += 1
                    cols = cols.upper()
                if len(seq) != 2 * n:
                    fmt["coord_not_2_chars"] += 1
                if any(c not in "abcdefghijklmnopqrs" for c in set(seq)):
                    fmt["coord_outside_a_s"] += 1
                if cols != ("BW" * ((n + 1) // 2))[:n]:
                    fmt["not_alternating_from_B"] += 1
                # anything other than SZ + moves?
                if t.count("[") != n + 1 or "SZ[19]" not in t:
                    fmt["extra_properties_or_no_SZ19"] += 1
                if not (t.startswith("(;SZ[19]") and t.endswith(")")):
                    fmt["unexpected_wrapper"] += 1
                if pid not in players:
                    players[pid] = len(plist)
                    plist.append(pid)
                m = GID.match(gid)
                lvl.append(li)
                pidx.append(players[pid])
                colr.append(0 if color == "B" else (1 if color == "W" else 9))
                nmov.append(n)
                gidn.append(int(m.group(1)) if m else -1)
                rowi.append(ri)
                last_col = cols[-1] if n else "?"
                lastmine.append(1 if last_col == color else 0)
                xs, ys = split_xy(seq)
                fx, fy = xs.translate(FLIP), ys.translate(FLIP)
                raw_full.append(h8(xs + "|" + ys))
                can_full.append(h8(min(cands(xs, ys, fx, fy, n))))
                for p in PREFIXES:
                    if n >= p:
                        raw_p[p].append(h8(xs[:p] + ys[:p]))
                        can_p[p].append(h8(min(cands(xs, ys, fx, fy, p))))
                    else:
                        raw_p[p].append(0)
                        can_p[p].append(0)
        print(f"{level}: done, total rows {len(lvl)}, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "pass1.pkl"), "wb") as f:
        pickle.dump(dict(lvl=lvl, pidx=pidx, colr=colr, nmov=nmov, gidn=gidn, rowi=rowi,
                         lastmine=lastmine, raw_full=raw_full, can_full=can_full,
                         raw_p=raw_p, can_p=can_p, plist=plist, fmt=dict(fmt)), f, protocol=4)
    print("format flags:", dict(fmt), flush=True)


# --------------------------------------------------------------------------- helpers
def groups_of(arr, valid=None):
    cnt = Counter(arr)
    cnt.pop(0, None)
    dups = {h for h, c in cnt.items() if c > 1}
    members = defaultdict(list)
    if dups:
        for i, h in enumerate(arr):
            if h in dups:
                members[h].append(i)
    return members


def pair_stats(members, D):
    lvl, pidx, colr = D["lvl"], D["pidx"], D["colr"]
    tot = same_p = same_l = diff_col = diff_col_same_p = 0
    games = 0
    sizes = Counter()
    for idx in members.values():
        k = len(idx)
        games += k
        sizes[k] += 1
        tot += k * (k - 1) // 2
        pc = Counter(pidx[i] for i in idx)
        same_p += sum(c * (c - 1) // 2 for c in pc.values())
        lc = Counter(lvl[i] for i in idx)
        same_l += sum(c * (c - 1) // 2 for c in lc.values())
        cc = Counter(colr[i] for i in idx)
        diff_col += cc.get(0, 0) * cc.get(1, 0)
        pcc = defaultdict(lambda: [0, 0])
        for i in idx:
            pcc[pidx[i]][colr[i]] += 1
        diff_col_same_p += sum(b * w for b, w in pcc.values())
    return {
        "groups": len(members),
        "games_in_groups": games,
        "pairs_total": tot,
        "pairs_same_player": same_p,
        "pairs_diff_player_same_level": same_l - same_p,
        "pairs_cross_level": tot - same_l,
        "pairs_diff_colour_diff_player": diff_col - diff_col_same_p,
        "pairs_diff_colour_same_player": diff_col_same_p,
        "largest_group": max(sizes) if sizes else 0,
        "group_size_hist_top": dict(sorted(sizes.items())[:10]),
        "groups_size_gt_10": sum(v for k, v in sizes.items() if k > 10),
    }


def lcp_str(a, b):
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # a[:lo]==b[:lo], a[:hi]!=b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def lcp_moves(xa, ya, xb, yb):
    return min(lcp_str(xa, xb), lcp_str(ya, yb))


# --------------------------------------------------------------------------- analysis
def analyze():
    csv.field_size_limit(10**8)
    t0 = time.time()
    with open(os.path.join(OUT, "pass1.pkl"), "rb") as f:
        D = pickle.load(f)
    lvl, pidx, colr, nmov, gidn, rowi = D["lvl"], D["pidx"], D["colr"], D["nmov"], D["gidn"], D["rowi"]
    plist = D["plist"]
    N = len(lvl)
    R = {"format_flags": D["fmt"], "rows": N}

    # ---- 1. full-sequence duplicates (raw moves / symmetry-canonical moves)
    R["full_moves_dup_raw"] = pair_stats(groups_of(D["raw_full"]), D)
    R["full_moves_dup_symmetry"] = pair_stats(groups_of(D["can_full"]), D)
    print("full dup done", time.time() - t0, flush=True)

    # ---- 2. prefix groups
    R["prefix_pairs_raw"] = {}
    R["prefix_pairs_symmetry"] = {}
    for p in PREFIXES:
        R["prefix_pairs_raw"][p] = pair_stats(groups_of(D["raw_p"][p]), D)
        R["prefix_pairs_symmetry"][p] = pair_stats(groups_of(D["can_p"][p]), D)
        print("prefix", p, R["prefix_pairs_raw"][p]["pairs_total"], R["prefix_pairs_symmetry"][p]["pairs_total"],
              time.time() - t0, flush=True)

    # pass 2: load move strings for the 60-move groups (raw + symmetry) and measure how long they stay identical
    g60r = groups_of(D["raw_p"][60])
    g60c = groups_of(D["can_p"][60])
    need = set()
    for idx in list(g60r.values()) + list(g60c.values()):
        need.update(idx)
    R["pass2_games_loaded"] = len(need)
    seqs = {}
    if need:
        base = 0
        for li, level in enumerate(LEVELS):
            path = os.path.join(DATA, f"train_{level}.csv")
            with open(path, newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                next(r)
                for ri, rec in enumerate(r):
                    gi = base + ri
                    if gi in need:
                        t = WS.sub("", rec[4])
                        mv = MOVE.findall(t)
                        seqs[gi] = split_xy("".join(xy for _, xy in mv).lower())
                base += ri + 1
    print("pass2 loaded", len(seqs), time.time() - t0, flush=True)

    def lcp_hist(members, use_sym, cap_pairs=2_000_000):
        buckets = Counter()
        detail = Counter()
        n_pairs = 0
        sample_note = None
        rng = random.Random(0)
        for idx in members.values():
            pairs = [(idx[a], idx[b]) for a in range(len(idx)) for b in range(a + 1, len(idx))]
            for i, j in pairs:
                n_pairs += 1
                if n_pairs > cap_pairs:
                    sample_note = f"capped at {cap_pairs} pairs"
                    break
                xa, ya = seqs[i]
                xb, yb = seqs[j]
                if use_sym:
                    L = max(lcp_moves(tx, ty, xb, yb) for tx, ty in transforms(xa, ya))
                else:
                    L = lcp_moves(xa, ya, xb, yb)
                na, nb = nmov[i], nmov[j]
                if L == na == nb:
                    kind = "identical_full_game"
                elif L == min(na, nb):
                    kind = "one_is_prefix_of_other"
                else:
                    kind = "diverge"
                sp = "same_player" if pidx[i] == pidx[j] else ("diff_player_same_level" if lvl[i] == lvl[j] else "cross_level")
                sc = "same_colour" if colr[i] == colr[j] else "diff_colour"
                detail[(kind, sp, sc)] += 1
                if kind == "diverge":
                    if L < 70:
                        b = "60-69"
                    elif L < 80:
                        b = "70-79"
                    elif L < 100:
                        b = "80-99"
                    elif L < 150:
                        b = "100-149"
                    elif L < 200:
                        b = "150-199"
                    else:
                        b = "200+"
                    buckets[b] += 1
                else:
                    buckets[kind] += 1
        return {
            "pairs": n_pairs,
            "common_prefix_length_buckets": dict(buckets),
            "by_kind_player_colour": {"|".join(k): v for k, v in sorted(detail.items(), key=lambda kv: -kv[1])},
            "note": sample_note,
        }

    R["prefix60_pairs_detail_raw"] = lcp_hist(g60r, False)
    R["prefix60_pairs_detail_symmetry"] = lcp_hist(g60c, True)
    print("lcp done", time.time() - t0, flush=True)

    # how many players / levels are touched by prefix-60 groups; biggest group composition
    def touched(members):
        ps, ls = set(), Counter()
        big = sorted(members.values(), key=len, reverse=True)[:5]
        comp = []
        for idx in big:
            comp.append({"size": len(idx), "players": len({pidx[i] for i in idx}),
                         "levels": dict(Counter(LEVELS[lvl[i]] for i in idx)),
                         "colours": dict(Counter("BW"[colr[i]] for i in idx)),
                         "moves_min": min(nmov[i] for i in idx), "moves_max": max(nmov[i] for i in idx)})
        for idx in members.values():
            for i in idx:
                ps.add(pidx[i])
                ls[LEVELS[lvl[i]]] += 1
        return {"players_touched": len(ps), "games_by_level": dict(ls), "top5_groups": comp}

    R["prefix60_touch_raw"] = touched(g60r)
    R["prefix60_touch_symmetry"] = touched(g60c)

    # ---- 3. game_id / row order
    order = {}
    level_rows = defaultdict(list)
    for i in range(N):
        level_rows[lvl[i]].append(i)
    gid_global = Counter()
    for li, level in enumerate(LEVELS):
        idx = level_rows[li]
        n = len(idx)
        g = [gidn[i] for i in idx]
        for x in g:
            gid_global[x] += 1
        seq_ok = all(g[k] == k + 1 for k in range(n))
        pl = [pidx[i] for i in idx]
        cl = [colr[i] for i in idx]
        mvs = [nmov[i] for i in idx]
        # player contiguity
        runs = 1 + sum(1 for k in range(1, n) if pl[k] != pl[k - 1])
        pc = Counter(pl)
        adj_same = sum(1 for k in range(1, n) if pl[k] == pl[k - 1])
        exp_adj_same = (n - 1) * sum((c / n) * ((c - 1) / (n - 1)) for c in pc.values())
        first, last = {}, {}
        for k, p in enumerate(pl):
            first.setdefault(p, k)
            last[p] = k
        span_big = [(last[p] - first[p]) / n for p, c in pc.items() if c >= 100]
        # colour pattern
        c_runs = 1 + sum(1 for k in range(1, n) if cl[k] != cl[k - 1])
        dec = []
        for d in range(10):
            a, b = d * n // 10, (d + 1) * n // 10
            seg_c = cl[a:b]
            seg_m = mvs[a:b]
            seg_p = [pc[p] for p in pl[a:b]]
            dec.append({"B_share": round(seg_c.count(0) / len(seg_c), 4),
                        "mean_moves": round(sum(seg_m) / len(seg_m), 1),
                        "mean_games_of_player": round(sum(seg_p) / len(seg_p), 1),
                        "distinct_players": len(set(pl[a:b]))})
        # within-player colour order: are a player's games colour-blocked?
        wp_runs = 0
        wp_exp = 0.0
        wp_players = 0
        per_player_rows = defaultdict(list)
        for k, p in enumerate(pl):
            per_player_rows[p].append(k)
        for p, ks in per_player_rows.items():
            if len(ks) < 20:
                continue
            cc = [cl[k] for k in ks]
            m = len(cc)
            nb = cc.count(0)
            nw = m - nb
            wp_runs += 1 + sum(1 for k in range(1, m) if cc[k] != cc[k - 1])
            wp_exp += 1 + 2 * nb * nw / m
            wp_players += 1
        # within-player: do file-consecutive games share openings more than random pairs of that player?
        cons_same = cons_tot = 0
        rnd_same = rnd_tot = 0
        rng = random.Random(1)
        h10 = D["raw_p"][20]
        for p, ks in per_player_rows.items():
            if len(ks) < 20:
                continue
            gl = [idx[k] for k in ks]
            for a in range(1, len(gl)):
                cons_tot += 1
                cons_same += h10[gl[a]] == h10[gl[a - 1]]
            for _ in range(len(gl) - 1):
                a, b = rng.sample(range(len(gl)), 2)
                rnd_tot += 1
                rnd_same += h10[gl[a]] == h10[gl[b]]
        # B share of each player (is every player exactly balanced?)
        pb = defaultdict(lambda: [0, 0])
        for p, c in zip(pl, cl):
            pb[p][c] += 1
        balanced = sum(1 for b, w in pb.values() if abs(b - w) <= 1)
        big_share = [b / (b + w) for b, w in pb.values() if b + w >= 50]
        order[level] = {
            "game_id_is_g_rowindex_plus1_all_rows": seq_ok,
            "game_id_min": min(g), "game_id_max": max(g), "game_id_unparsable": sum(1 for x in g if x < 0),
            "player_runs": runs, "players": len(pc),
            "adjacent_rows_same_player": adj_same,
            "adjacent_rows_same_player_expected_if_shuffled": round(exp_adj_same, 1),
            "players_ge100_span_fraction_of_file": quant(span_big) if span_big else {},
            "colour_runs": c_runs, "colour_runs_expected_if_shuffled": round(1 + 2 * cl.count(0) * cl.count(1) / n, 1),
            "deciles": dec,
            "within_player_colour_runs(players>=20)": wp_runs,
            "within_player_colour_runs_expected": round(wp_exp, 1),
            "within_player_players": wp_players,
            "consecutive_same_player_same_first20": [cons_same, cons_tot],
            "random_same_player_same_first20": [rnd_same, rnd_tot],
            "players_colour_balanced_within_1": balanced,
            "B_share_of_players_ge50_games": quant(big_share) if big_share else {},
        }
    R["row_order"] = order
    R["game_id_shared_across_files"] = sum(1 for v in gid_global.values() if v > 1)
    R["game_id_distinct_numbers_all_files"] = len(gid_global)
    print("row order done", time.time() - t0, flush=True)

    # ---- 4. player_id format
    lens = Counter(len(p) for p in plist)
    charset = Counter()
    for p in plist:
        if re.fullmatch(r"[0-9a-f]+", p):
            charset["lower_hex"] += 1
        elif re.fullmatch(r"[0-9A-Fa-f]+", p):
            charset["hex_mixed_case"] += 1
        else:
            charset["other"] += 1
    lower_collide = len(plist) - len({p.lower() for p in plist})
    ham1 = 0
    L = max(lens)
    for pos in range(L):
        buck = Counter(p[:pos] + "*" + p[pos + 1:] for p in plist if len(p) == L)
        ham1 += sum(c * (c - 1) // 2 for c in buck.values())
    pref = {}
    for k in (4, 5, 6, 8):
        c = Counter(p[:k] for p in plist)
        pref[k] = sum(v * (v - 1) // 2 for v in c.values())
    npl = len(plist)
    exp_pref = {k: round(npl * (npl - 1) / 2 / 16**k, 2) for k in (4, 5, 6, 8)}
    char_freq = Counter("".join(plist))
    pl_levels = defaultdict(set)
    for i in range(N):
        pl_levels[pidx[i]].add(lvl[i])
    R["player_id"] = {
        "n_players": npl, "length_counts": dict(lens), "charset": dict(charset),
        "case_insensitive_collisions": lower_collide,
        "pairs_hamming_distance_1": ham1,
        "pairs_sharing_prefix_k": pref, "expected_prefix_pairs_if_random_hex": exp_pref,
        "char_freq_min_max": [min(char_freq.values()), max(char_freq.values())],
        "players_in_more_than_one_level": sum(1 for s in pl_levels.values() if len(s) > 1),
    }

    # ---- 5. moves per game by level
    mv_stats = {}
    for li, level in enumerate(LEVELS):
        v = [nmov[i] for i in level_rows[li]]
        lm = [D["lastmine"][i] for i in level_rows[li]]
        c = Counter(v)
        mv_stats[level] = {
            "mean": round(sum(v) / len(v), 2), "median": statistics.median(v),
            "stdev": round(statistics.pstdev(v), 2),
            "min": min(v), "max": max(v),
            "exactly_100": c[100], "moves_100_to_104": [c[k] for k in range(100, 105)],
            "moves_95_to_99": [c[k] for k in range(95, 100)],
            "le_120": sum(1 for x in v if x <= 120), "ge_300": sum(1 for x in v if x >= 300),
            "odd_move_count_share": round(sum(1 for x in v if x % 2) / len(v), 4),
            "last_move_by_the_labelled_player_share": round(sum(lm) / len(lm), 4),
        }
    allv = list(nmov)
    R["moves"] = {"by_level": mv_stats, "all_exactly_100": sum(1 for x in allv if x == 100),
                  "all_min": min(allv), "all_lt_100": sum(1 for x in allv if x < 100)}

    # ---- 6. per-level players by game count
    pool = {}
    pg = Counter(pidx)
    for li, level in enumerate(LEVELS):
        c = Counter(pidx[i] for i in level_rows[li])
        vals = list(c.values())
        pool[level] = {k: sum(1 for x in vals if x >= int(k[2:])) for k in ("ge5", "ge20", "ge25", "ge100", "ge105", "ge120", "ge200")}
        pool[level]["games_in_players_ge120"] = sum(x for x in vals if x >= 120)
        pool[level]["players"] = len(vals)
    tot = {k: sum(pool[l][k] for l in LEVELS) for k in pool["D"]}
    R["pool_sizes"] = {"by_level": pool, "total": tot}

    with open(os.path.join(OUT, "leak_hunt.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, default=str)
    print("done", time.time() - t0, flush=True)


# --------------------------------------------------------------------------- detail of the long-prefix pairs
def detail():
    """Who is behind the 60/100-move identical openings (players anonymised as P1, P2, ...)."""
    csv.field_size_limit(10**8)
    with open(os.path.join(OUT, "pass1.pkl"), "rb") as f:
        D = pickle.load(f)
    lvl, pidx, colr, nmov, rowi = D["lvl"], D["pidx"], D["colr"], D["nmov"], D["rowi"]
    pg = Counter(pidx)
    g60r = groups_of(D["raw_p"][60])
    g60c = groups_of(D["can_p"][60])
    g100r = groups_of(D["raw_p"][100])
    g100c = groups_of(D["can_p"][100])
    need = set()
    for g in (g60r, g60c, g100r, g100c):
        for idx in g.values():
            need.update(idx)
    seqs = {}
    base = 0
    for li, level in enumerate(LEVELS):
        with open(os.path.join(DATA, f"train_{level}.csv"), newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for ri, rec in enumerate(r):
                if base + ri in need:
                    mv = MOVE.findall(WS.sub("", rec[4]))
                    seqs[base + ri] = split_xy("".join(xy for _, xy in mv).lower())
            base += ri + 1
    anon = {}

    def A(p):
        if p not in anon:
            anon[p] = f"P{len(anon) + 1}"
        return anon[p]

    def best(i, j):
        xa, ya = seqs[i]
        xb, yb = seqs[j]
        res = [(lcp_moves(tx, ty, xb, yb), t) for t, (tx, ty) in enumerate(transforms(xa, ya))]
        return max(res)

    out = {}
    # players involved in 60-move groups
    for name, g in (("raw60", g60r), ("sym60", g60c)):
        per = defaultdict(lambda: {"games_in_groups": 0, "colours": Counter()})
        for idx in g.values():
            for i in idx:
                d = per[pidx[i]]
                d["games_in_groups"] += 1
                d["colours"]["BW"[colr[i]]] += 1
        out[f"{name}_players"] = {
            A(p): {"level": None,
                   "games_in_groups": d["games_in_groups"], "total_games": pg[p], "colours": dict(d["colours"])}
            for p, d in sorted(per.items(), key=lambda kv: -kv[1]["games_in_groups"])
        }
        # fill the level cheaply
        for idx in g.values():
            for i in idx:
                out[f"{name}_players"][A(pidx[i])]["level"] = LEVELS[lvl[i]]
        # diff-player pairs
        pp = Counter()
        for idx in g.values():
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    if pidx[i] != pidx[j]:
                        L, t = best(i, j) if name.startswith("sym") else (lcp_moves(*seqs[i], *seqs[j]), 0)
                        key = (A(pidx[i]), A(pidx[j])) if A(pidx[i]) < A(pidx[j]) else (A(pidx[j]), A(pidx[i]))
                        pp[(key[0], key[1], LEVELS[lvl[i]], LEVELS[lvl[j]], "BW"[colr[i]] + "BW"[colr[j]], L)] += 1
        out[f"{name}_diff_player_pairs(pA,pB,lvlA,lvlB,colours,common_moves)"] = [list(k) + [v] for k, v in sorted(pp.items())]
    # every 100-move pair in full
    for name, g in (("raw100", g100r), ("sym100", g100c)):
        rows = []
        for idx in g.values():
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    L, t = best(i, j)
                    rows.append({"pA": A(pidx[i]), "pB": A(pidx[j]), "lvl": [LEVELS[lvl[i]], LEVELS[lvl[j]]],
                                 "colours": "BW"[colr[i]] + "BW"[colr[j]], "moves": [nmov[i], nmov[j]],
                                 "common_moves": L, "symmetry_id": t,
                                 "row_gap_in_file": abs(rowi[i] - rowi[j]) if lvl[i] == lvl[j] else None})
        out[f"{name}_pairs"] = rows
    with open(os.path.join(OUT, "leak_hunt_detail.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    {"pass1": pass1, "analyze": analyze, "detail": detail}[sys.argv[1]]()
