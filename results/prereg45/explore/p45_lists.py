"""prereg45 (competition 1, v2 new pools): the game lists of the extra KataGo P1 extraction (PLAN.md 1).

    PY results/prereg45/explore/p45_lists.py build [--dry]   # -> data/prereg45/lists/{M1,M2}.json
                                                             #    + results/prereg45/explore/lists_record.json
M1 = the T players not in prereg39's L, each by p39_lists.l_games (30 games); M2 = every T player, the next 30 games
in the same order (games already in L / M1 removed first), by the same rule. CPU only; reads the prereg36 move-cache
meta and prereg39's L list. Never the validation answers folder, never data/prereg10/val.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P39 = os.path.join(REPO, "results", "prereg39", "explore")
for p in (P39, os.path.join(REPO, "results", "prereg36", "explore"), os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import p39_lists as PL  # noqa: E402
import p36_lib as L36  # noqa: E402

DATA = os.path.join(REPO, "data", "prereg45")
LISTS = os.path.join(DATA, "lists")
RECORD = os.path.join(HERE, "lists_record.json")
L39 = os.path.join(REPO, "data", "prereg39", "lists", "L.json")
FORMAT = "c10-prereg45 list v1"


def build():
    with open(os.path.join(P39, "lists_record.json"), encoding="utf-8") as f:
        want = json.load(f)["lists_sha256"]["L"]
    if PL.sha_file(L39) != want:
        raise SystemExit("prereg39 L list differs from its lists_record.json")
    with open(L39, encoding="utf-8") as f:
        Lg = json.load(f)["games"]
    cache = L36.Cache()
    E, D, T, _per = L36.splits(cache)
    Ts, Es, Ds = set(T), set(E), set(D)
    by = {}
    for p, g, r, c, bad in zip(cache.pid.tolist(), cache.gid.tolist(), cache.rank.tolist(), cache.color.tolist(),
                               cache.bad.tolist()):
        if p in Ts and not bad:
            by.setdefault(p, []).append((g, c, r))
    lp = {g["player_id"] for g in Lg}
    if not lp <= Ts:
        raise SystemExit("an L player is not in T")
    used = {(g["player_id"], g["game_id"]) for g in Lg}

    def take(p, exclude):
        g = [x for x in by[p] if (p, x[0]) not in exclude]
        return [{"rank": g[i][2], "game_id": g[i][0], "player_id": p, "color": g[i][1], "row": -1, "mock_ref": -1}
                for i in PL.l_games(p, [(x[0], x[1]) for x in g])]
    M1 = []
    for p in sorted(Ts - lp, key=lambda q: PL._h("p45|M1|%s" % q)):
        M1 += take(p, used)
    used |= {(g["player_id"], g["game_id"]) for g in M1}
    M2 = []
    for p in sorted(Ts, key=lambda q: PL._h("p45|M2|%s" % q)):
        M2 += take(p, used)
    lists = {"M1": M1, "M2": M2}
    keys = [(g["rank"], g["game_id"], g["player_id"]) for gl in (Lg, M1, M2) for g in gl]
    if len(keys) != len(set(keys)):
        raise SystemExit("a game is in two lists")
    if any(g["player_id"] in (Es | Ds) for gl in lists.values() for g in gl):
        raise SystemExit("an E / D player in the lists")

    def stat(gl):
        n = len(gl)
        return {"games": n, "players": len({g["player_id"] for g in gl}),
                "black": sum(1 for g in gl if g["color"] == "B"), "white": sum(1 for g in gl if g["color"] == "W"),
                "est_hours_4090": round(n * PL.POS_PER_GAME / PL.POS_PER_S / 3600, 2),
                "chunks": (n + PL.CHUNK - 1) // PL.CHUNK}
    rec = {"format": "c10-prereg45 lists record v1", "split": {"E": len(E), "D": len(D), "T": len(T)},
           "L_prereg39": stat(Lg), "sets": {n: stat(v) for n, v in lists.items()},
           "pos_per_game": PL.POS_PER_GAME, "pos_per_s": PL.POS_PER_S,
           "players_with_games": {"L+M1+M2": len({g["player_id"] for gl in (Lg, M1, M2) for g in gl})}}
    rec["total_new"] = {"games": len(M1) + len(M2),
                        "est_hours_4090": round((len(M1) + len(M2)) * PL.POS_PER_GAME / PL.POS_PER_S / 3600, 2)}
    return lists, rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    PL.below_normal()
    lists, rec = build()
    if not a.dry:
        os.makedirs(LISTS, exist_ok=True)
        rec["lists_sha256"] = {}
        for n, gl in lists.items():
            p = os.path.join(LISTS, n + ".json")
            if os.path.exists(p):
                raise SystemExit("%s exists; not overwritten" % p)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"format": FORMAT, "set": n, "games": gl}, f)
            rec["lists_sha256"][n] = PL.sha_file(p)
        rec["tool_sha256"] = PL.sha_file(os.path.abspath(__file__))
        with open(RECORD, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
