"""prereg39 explore (competition 1, v2 new pools): the game lists of the KataGo P1 extraction (PLAN.md 2).

    PY results/prereg39/explore/p39_lists.py build     # -> data/prereg39/lists/{L,D,pub2,priv2,E,E2}.json
                                                       #    + results/prereg39/explore/lists_record.json
    PY results/prereg39/explore/p39_lists.py build --e-seeds 0-19 --dry    # statistics only, nothing written

CPU only; reads the fit feature-store meta (data/prereg06/feats/meta.npz), the prereg36 move-cache meta, the row
lists of the prereg36 pools (data/prereg36/terms_D|E: cand_rows / q_flat only, no score term is read), the
PREREG_10 mock set list (data/prereg10/mock/sets.json, to count reusable games) and the official new-pool files.
Never the validation answers folder, never data/prereg10/val, no competition-2 feature.

A fit game is named by (rank, game_id, player_id, colour); `row` is its feature-store row (D, E) or -1 (L games of
players outside the store). An official game is named by (pool, item) in p36_lib.pool_items order.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P36 = os.path.join(REPO, "results", "prereg36", "explore")
for p in (P36, os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import p36_lib as L36  # noqa: E402
from c10data import LEVELS  # noqa: E402

DATA = os.path.join(REPO, "data", "prereg39")
LISTS = os.path.join(DATA, "lists")
RECORD = os.path.join(HERE, "lists_record.json")
TERMS = {"D": os.path.join(REPO, "data", "prereg36", "terms_D"), "E": os.path.join(REPO, "data", "prereg36", "terms_E")}
MOCK_SETS = os.path.join(REPO, "data", "prereg10", "mock", "sets.json")
MOCK_DIR = os.path.join(REPO, "data", "prereg10", "mock")
TESTS = os.path.join(REPO, "data", "test_v2_download", "x", "tests")
L_PER_LEVEL = 120
L_GAMES = 30
D_SEEDS = list(range(100, 110))
E_SEEDS = list(range(0, 10))          # PLAN.md 2: list E = seeds 0-9, list E2 = the extra games of seeds 10-19
E2_SEEDS = list(range(10, 20))
POS_PER_GAME = 222.0                  # results/prereg10/explore/REPORT.md (measured pilot)
POS_PER_S = 1012.0
CHUNK = 200
FORMAT = "c10-prereg39 list v1"
csv.field_size_limit(10 ** 9)


def below_normal():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def _h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_seeds(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def l_players(T, level_of):
    """Per level the first L_PER_LEVEL T players by sha256('p39|L|pid') (all when a level has fewer)."""
    out = []
    for lv in LEVELS:
        pl = sorted((p for p in T if level_of[p] == lv), key=lambda p: _h("p39|L|%s" % p))
        out += pl[:L_PER_LEVEL]
    return out


def l_games(pid, games):
    """games: [(game_id, colour)] of one player (not bad). Ordered by sha256('p39|L|pid|game_id'); the first 15 of
    each colour, then topped up in the same order to L_GAMES from what is left. -> indices into games."""
    order = sorted(range(len(games)), key=lambda i: _h("p39|L|%s|%s" % (pid, games[i][0])))
    half = L_GAMES // 2
    take, n = [], {"B": 0, "W": 0}
    for i in order:
        c = games[i][1]
        if n[c] < half:
            take.append(i)
            n[c] += 1
    chosen = set(take)
    for i in order:
        if len(take) >= L_GAMES:
            break
        if i not in chosen:
            take.append(i)
            chosen.add(i)
    return sorted(take, key=order.index)


def pool_rows(which, seeds):
    rows = set()
    for s in seeds:
        with np.load(os.path.join(TERMS[which], "pool_%03d.npz" % s)) as z:
            rows.update(np.asarray(z["cand_rows"]).ravel().tolist())
            rows.update(np.asarray(z["q_flat"]).ravel().tolist())
    return sorted(int(r) for r in rows)


def official_items(pool):
    cands = list(csv.DictReader(open(os.path.join(TESTS, "player_identification_candidates2_%s.csv" % pool),
                                     encoding="utf-8", newline="")))
    qs = list(csv.DictReader(open(os.path.join(TESTS, "player_identification_test2_%s.csv" % pool),
                                  encoding="utf-8", newline="")))
    items, _ids, _cr, _qr, _q = L36.pool_items(cands, qs)
    return items


def build(e_seeds):
    with np.load(os.path.join(L36.FEATS, "meta.npz"), allow_pickle=False) as z:
        s_rank, s_gid, s_pid, s_col = z["rank"], z["game_id"], z["player_id"], z["color"]
    cache = L36.Cache()
    E, D, T, _per = L36.splits(cache)
    Es, Ds, Ts = set(E), set(D), set(T)
    # ---- L
    by = {}
    for i, (p, g, r, c, bad) in enumerate(zip(cache.pid.tolist(), cache.gid.tolist(), cache.rank.tolist(),
                                              cache.color.tolist(), cache.bad.tolist())):
        if p in Ts and not bad:
            by.setdefault(p, []).append((g, c, r))
    level_of = {p: v[0][2] for p, v in by.items()}
    if any(len({x[2] for x in v}) != 1 for v in by.values()):
        raise SystemExit("a T player has more than one rank")
    lp = l_players(sorted(by), level_of)
    store_key = {(str(a), str(b), str(c)): i for i, (a, b, c) in
                 enumerate(zip(s_rank.tolist(), s_gid.tolist(), s_pid.tolist()))}
    Lg = []
    for p in lp:
        g = by[p]
        for i in l_games(p, [(x[0], x[1]) for x in g]):
            Lg.append({"rank": g[i][2], "game_id": g[i][0], "player_id": p, "color": g[i][1],
                       "row": store_key.get((g[i][2], g[i][0], p), -1)})
    # ---- D / E
    def store_games(rows, players, name):
        out = []
        for r in rows:
            p = str(s_pid[r])
            if p not in players:
                raise SystemExit("%s pool row %d is not a %s player" % (name, r, name))
            out.append({"rank": str(s_rank[r]), "game_id": str(s_gid[r]), "player_id": p, "color": str(s_col[r]),
                        "row": int(r)})
        return out
    Dg = store_games(pool_rows("D", D_SEEDS), Ds, "D")
    Eg = store_games(pool_rows("E", e_seeds), Es, "E")
    first = {g["row"] for g in Eg}
    E2g = [g for g in store_games(pool_rows("E", E2_SEEDS), Es, "E") if g["row"] not in first]         if list(e_seeds) == E_SEEDS else []
    # ---- mock reuse (PREREG_10 mock: black games of 926 store players, side 0 = the player)
    with open(MOCK_SETS, encoding="utf-8") as f:
        mock = json.load(f)["games"]
    have_chunk = {int(x[6:11]) for x in os.listdir(MOCK_DIR) if x.startswith("chunk_") and x.endswith(".npz")
                  and ".tmp" not in x}
    mock_ref = {g["store_row"]: j for j, g in enumerate(mock) if g["color"] == "B" and (j // CHUNK) in have_chunk}
    for name, gl in (("L", Lg), ("D", Dg), ("E", Eg), ("E2", E2g)):
        for g in gl:
            j = mock_ref.get(g["row"], -1) if g["row"] >= 0 else -1
            if j >= 0 and (mock[j]["player_id"] != g["player_id"] or mock[j]["game_id"] != g["game_id"]
                           or g["color"] != "B"):
                raise SystemExit("mock game %d does not match store row %d" % (j, g["row"]))
            g["mock_ref"] = j
    off = {pool: [{"pool": pool, "item": i, "color": c} for i, (_s, c) in enumerate(official_items(pool))]
           for pool in ("public", "private")}
    lists = {"L": Lg, "D": Dg, "pub2": off["public"], "priv2": off["private"], "E": Eg, "E2": E2g}
    # ---- checks
    pl = {n: {g["player_id"] for g in lists[n]} for n in ("L", "D", "E", "E2")}
    pl["E"] |= pl.pop("E2")
    if (pl["L"] & (Es | Ds)) or (pl["D"] & pl["E"]) or not pl["L"] <= Ts:
        raise SystemExit("the L / D / E players overlap")
    keys = [(g["rank"], g["game_id"], g["player_id"]) for n in ("L", "D", "E", "E2") for g in lists[n]]
    if len(keys) != len(set(keys)):
        raise SystemExit("a game is in two lists")

    def stat(gl, fit=True):
        n = len(gl)
        fresh = sum(1 for g in gl if g.get("mock_ref", -1) < 0)
        d = {"games": n, "fresh_games": fresh, "reused_from_prereg10_mock": n - fresh,
             "black": sum(1 for g in gl if g["color"] == "B"), "white": sum(1 for g in gl if g["color"] == "W"),
             "est_positions_fresh": int(round(fresh * POS_PER_GAME)),
             "est_hours_4090_fresh": round(fresh * POS_PER_GAME / POS_PER_S / 3600, 2),
             "chunks_fresh": (fresh + CHUNK - 1) // CHUNK}
        if fit:
            d["players"] = len({g["player_id"] for g in gl})
        return d
    rec = {"format": "c10-prereg39 lists record v1", "l_per_level": L_PER_LEVEL, "l_games": L_GAMES,
           "d_seeds": D_SEEDS, "e_seeds": list(e_seeds), "e2_seeds": E2_SEEDS if E2g else [], "pos_per_game": POS_PER_GAME, "pos_per_s": POS_PER_S,
           "split": {"E": len(E), "D": len(D), "T": len(T)},
           "l_players_by_level": {lv: sum(1 for p in lp if level_of[p] == lv) for lv in LEVELS},
           "l_players_in_store": sum(1 for p in lp if any(g["row"] >= 0 for g in Lg if g["player_id"] == p)),
           "sets": {n: stat(lists[n], n in ("L", "D", "E", "E2")) for n in lists},
           "mock_chunks_present": len(have_chunk)}
    rec["total"] = {"games": sum(v["games"] for v in rec["sets"].values()),
                    "fresh_games": sum(v["fresh_games"] for v in rec["sets"].values()),
                    "est_hours_4090_fresh": round(sum(v["fresh_games"] for v in rec["sets"].values())
                                                  * POS_PER_GAME / POS_PER_S / 3600, 2)}
    return lists, rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--e-seeds", default=None, help="statistics for another E seed range (needs --dry)")
    b.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    below_normal()
    if a.e_seeds and not a.dry:
        raise SystemExit("--e-seeds is for --dry statistics; the written lists use the PLAN's E seeds")
    lists, rec = build(parse_seeds(a.e_seeds) if a.e_seeds else E_SEEDS)
    if not a.dry:
        os.makedirs(LISTS, exist_ok=True)
        rec["lists_sha256"] = {}
        for n, gl in lists.items():
            p = os.path.join(LISTS, n + ".json")
            if os.path.exists(p):
                raise SystemExit("%s exists; not overwritten" % p)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"format": FORMAT, "set": n, "games": gl}, f)
            rec["lists_sha256"][n] = sha_file(p)
        rec["tool_sha256"] = sha_file(os.path.abspath(__file__))
        with open(RECORD, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
