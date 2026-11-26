"""prereg30 (CPU): the pilot game list (PLAN.md 2).

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg30/explore/p30_data.py sets
        -> data/prereg30/pilot/sets.json (p1_extract sets format; probe source 'fit'), data/prereg30/pilot/pilot_map.npz
           and results/prereg30/explore/pilot_inventory.json (commit it before the GPU run)

The pilot = every probe row (data/prereg17/prep/probe_meta.npz, 270,000 fit games, row order kept) whose player is held
out by S25 (results/prereg25/explore/p25_lib.holdout_bit, salt 'c10-prereg25|holdout|'). Games come from the original
probe sets.json files by (src, ref); the player key is checked against probe_meta. Only fit players: no inner game, no
validation player, nothing under data/local (every path goes through p30_lib.guard).
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "results", "prereg25", "explore"))

import numpy as np  # noqa: E402

import p25_lib as L25  # noqa: E402

PROBE_META = os.path.join(REPO, "data", "prereg17", "prep", "probe_meta.npz")
SETS_JSON = {"p1_probe": os.path.join(REPO, "data", "prereg08", "p1", "sets.json"),
             "p2_probe": os.path.join(REPO, "data", "prereg08", "p2", "sets.json"),
             "p3_probe": os.path.join(REPO, "data", "prereg09", "p3", "sets.json")}
OLD_CHUNKS = {"p1_probe": os.path.join(REPO, "data", "prereg08", "p1", "probe"),
              "p2_probe": os.path.join(REPO, "data", "prereg08", "p2", "probe"),
              "p3_probe": os.path.join(REPO, "data", "prereg09", "p3", "probe")}
OUT = os.path.join(REPO, "data", "prereg30", "pilot")
SETS_OUT = os.path.join(OUT, "sets.json")
MAP_OUT = os.path.join(OUT, "pilot_map.npz")
INVENTORY = os.path.join(HERE, "pilot_inventory.json")
SETS_FORMAT = "c10-prereg08 p1 sets v1"     # p1_extract.load_sets accepts only this format
P1_POS_PER_S = 1234.2                        # prereg25 V extraction, mean positions/s over 375 chunks (PLAN.md 2)


def _guard(p):
    q = os.path.abspath(p).replace("\\", "/").lower()
    if any(f in q for f in ("/data/local", "rank_val_answers", "player_val_answers")):
        raise SystemExit(f"refused path {p}")
    return p


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(_guard(tmp), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def pilot_rows():
    """-> (rows [i] of probe_meta held out by S25, src, ref, players, y) in probe_meta order."""
    with np.load(_guard(PROBE_META), allow_pickle=False) as z:
        players = np.asarray(z["players"]).astype(str)
        src = np.asarray(z["src"]).astype(str)
        ref = np.asarray(z["ref"]).astype(np.int64)
        y = np.asarray(z["y"]).astype(np.int64)
    hb = np.array([L25.holdout_bit(k) for k in players], dtype=bool)
    rows = np.flatnonzero(hb)
    return rows, src, ref, players, y


def old_chunk_counts(want):
    """want: {src: set(ref)} -> {(src, ref): (counts [2], failed, n_moves)} from the original probe chunks (read-only;
    only the ref / counts / failed / n_moves arrays are loaded)."""
    out = {}
    for s, d in OLD_CHUNKS.items():
        need = want.get(s, set())
        if not need:
            continue
        for f in sorted(glob.glob(os.path.join(_guard(d), "chunk_*.npz"))):
            with np.load(f, allow_pickle=False) as c:
                rf = np.asarray(c["ref"]).astype(np.int64)
                m = np.flatnonzero(np.isin(rf, list(need)))
                if len(m) == 0:
                    continue
                cnt, fail, nm = np.asarray(c["counts"]), np.asarray(c["failed"]), np.asarray(c["n_moves"])
                for i in m:
                    out[(s, int(rf[i]))] = (cnt[i].astype(np.int64), bool(fail[i]), int(nm[i]))
    return out


def cmd_sets(a):
    t0 = time.time()
    rows, src, ref, players, y = pilot_rows()
    games_src = {}
    for s, p in SETS_JSON.items():
        with open(_guard(p), encoding="utf-8") as f:
            games_src[s] = json.load(f)["probe"]["games"]
    games = []
    for i in rows:
        g = games_src[str(src[i])][int(ref[i])]
        if f"{g['rank']}:{g['player_id']}" != str(players[i]):
            raise SystemExit(f"probe row {i}: sets.json player differs from probe_meta")
        games.append({"rank": g["rank"], "game_id": g["game_id"], "player_id": g["player_id"], "color": g["color"]})
    if len({(g["rank"], g["game_id"]) for g in games}) != len(games):
        raise SystemExit("duplicate game in the pilot list")
    want = collections.defaultdict(set)
    for i in rows:
        want[str(src[i])].add(int(ref[i]))
    old = old_chunk_counts(want)
    if len(old) != len(rows):
        raise SystemExit(f"{len(old)} of {len(rows)} pilot games found in the original chunks")
    positions = int(sum(int(old[(str(src[i]), int(ref[i]))][0].sum()) for i in rows))
    per_level = collections.OrderedDict()
    for lv in L25.LEVELS:
        m = np.array([g["rank"] == lv for g in games])
        per_level[lv] = {"players": len({g["player_id"] for g, k in zip(games, m) if k}), "games": int(m.sum())}
    colours = collections.Counter(g["color"] for g in games)
    key_list = "\n".join(f"{g['rank']}:{g['game_id']}:{g['player_id']}:{g['color']}" for g in games)
    sets = {"format": SETS_FORMAT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "what": "prereg30 pilot: probe games of the fit players held out by S25 (PLAN.md 2)",
            "probe": {"source": "fit", "games": games},
            "bank": {"source": "inner", "rows": []}, "mono": {"source": "inner", "rows": []},
            "counts": {"probe": len(games), "bank": 0, "mono": 0}}
    os.makedirs(_guard(OUT), exist_ok=True)
    if os.path.exists(SETS_OUT) and not a.force:
        raise SystemExit(f"{SETS_OUT} exists; not overwritten (use --force only before any GPU chunk exists)")
    write_json(SETS_OUT, sets)
    tmp = MAP_OUT + ".tmp.npz"
    np.savez(_guard(tmp), probe_row=rows.astype(np.int64), src=src[rows], ref=ref[rows], players=players[rows],
             y=y[rows])
    os.replace(tmp, MAP_OUT)
    inv = {"format": "c10-prereg30 pilot inventory v1",
           "what": "PLAN.md 2: pilot games = probe rows whose player is held out by S25 (fit players only)",
           "games": len(games), "players": len({g["rank"] + ":" + g["player_id"] for g in games}),
           "per_level": per_level, "colours": dict(colours),
           "games_by_src": {s: int((src[rows] == s).sum()) for s in SETS_JSON},
           "positions_from_original_chunks": positions,
           "positions_per_game": round(positions / len(games), 2),
           "gpu_hours_at_p1_speed": round(positions / P1_POS_PER_S / 3600, 3),
           "p1_speed_reference_positions_per_s": P1_POS_PER_S,
           "game_list_sha256": hashlib.sha256(key_list.encode("utf-8")).hexdigest(),
           "sets_json_sha256": L_sha(SETS_OUT), "pilot_map_sha256": L_sha(MAP_OUT),
           "probe_meta_sha256": L_sha(PROBE_META),
           "seconds": round(time.time() - t0, 1)}
    write_json(INVENTORY, inv)
    print(json.dumps({k: v for k, v in inv.items() if k != "per_level"}, indent=1))
    return 0


def L_sha(path, block=1 << 20):
    h = hashlib.sha256()
    with open(_guard(path), "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return {"sets": cmd_sets}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
