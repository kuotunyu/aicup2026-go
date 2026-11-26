"""prereg64: rows for the GPU job (PLAN step 2, prototypes for every labelled non-inner player that is not a c49
head class). CPU only, no vectors, no score.

    PY results/prereg64/explore/p64_rows.py rows
        -> data/prereg64/proto_rows.npy  (rows of the prereg49 move cache, ascending)
           results/prereg64/explore/rows.json  (counts, overlaps, sha256)

Prototype owners = cache49 players that are NOT c49 head classes (p49_lib.t49 = every player outside E with >= 20
games; checked equal to the class ids in data/advice/gemini_1121/c49_head.npz), NOT E players (they already have
c49 player-side vectors: data/prereg49/emb/.../store_E.npz) and NOT inner players (split v1 'inner' and the players
of results/prereg02/rank/dumps/inner_keys.npz). All their games, every row (unparsed games refused).
The inner rows for job 1 are prereg58's data/prereg58/inner_rows.npy (rows of data/prereg44/cache/inner), reused.
"""
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg36", "explore"),
          os.path.join(REPO, "results", "prereg49", "explore"), os.path.join(REPO, "results", "prereg58", "explore")):
    if p not in sys.path:
        sys.path.insert(0, p)
import p58_lib as K58  # noqa: E402
import p49_lib as L49  # noqa: E402

DATA = os.path.join(REPO, "data", "prereg64")
PROTO_ROWS = os.path.join(DATA, "proto_rows.npy")
ROWS_JSON = os.path.join(HERE, "rows.json")
HEAD_GEMINI = os.path.join(REPO, "data", "advice", "gemini_1121", "c49_head.npz")


def cmd_rows():
    if os.path.exists(PROTO_ROWS) or os.path.exists(ROWS_JSON):
        raise SystemExit("proto_rows.npy / rows.json exists")
    cache = L49.Cache49()
    pid = np.asarray(cache.pid).astype(str)
    E = set(L49.e_players())
    inner_split, val = L49.extra_players()
    with np.load(K58.guard(K58.INNER_KEYS), allow_pickle=False) as z:
        inner_keys = set(np.asarray(z["player_id"]).astype(str).tolist())
    inner = set(inner_split) | inner_keys
    T49 = L49.t49(cache, sorted(E))
    with np.load(K58.guard(HEAD_GEMINI), allow_pickle=False) as z:
        gem_ids = np.asarray(z["class_id"]).astype(str).tolist()
    if gem_ids != list(T49):
        raise SystemExit("c49_head.npz class ids differ from p49_lib.t49")
    head = set(T49)
    owners = sorted(set(pid.tolist()) - head - E - inner)
    rows = np.flatnonzero(np.isin(pid, np.array(owners))).astype(np.int64)
    if np.asarray(cache.bad)[rows].any():
        raise SystemExit("unparsed games among the prototype rows")
    rk = np.asarray(cache.rank).astype(str)
    u, first, n = np.unique(pid[rows], return_index=True, return_counts=True)
    one_rank = len(set(zip(pid[rows].tolist(), rk[rows].tolist()))) == len(u)
    if not one_rank:
        raise SystemExit("a prototype owner with two ranks")
    os.makedirs(DATA, exist_ok=True)
    np.save(PROTO_ROWS, rows)
    rep = {"format": "c10-prereg64 rows v1", "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "what": "every game of the cache49 players outside the c49 head classes, outside E and outside inner",
           "cache_players": int(len(set(pid.tolist()))), "cache_games": int(cache.n),
           "head_classes": len(head), "E_players": len(E), "inner_players_split": len(inner_split),
           "inner_players_keys": len(inner_keys), "inner_players_union": len(inner),
           "inner_in_head": len(inner & head), "val_players": len(val), "val_in_head": len(val & head),
           "owners": int(len(u)), "owner_games": int(len(rows)),
           "owners_val": len(set(u.tolist()) & val),
           "games_per_owner": {"min": int(n.min()), "median": float(np.median(n)), "max": int(n.max())},
           "owners_per_rank": np.bincount(K58.rank_index(rk[rows][first]), minlength=K58.L).tolist(),
           "first_row": int(rows[0]), "last_row": int(rows[-1]),
           "cache_codes_sha256": cache.meta.get("codes_sha256"),
           "proto_rows_sha256": K58.sha256_file(PROTO_ROWS),
           "inner_rows_job1": "data/prereg58/inner_rows.npy",
           "inner_rows_sha256": K58.sha256_file(K58.INNER_ROWS),
           "inner_rows_count": int(len(np.load(K58.INNER_ROWS)))}
    K58.write_json(ROWS_JSON, rep)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["rows"]:
        raise SystemExit("usage: p64_rows.py rows")
    sys.exit(cmd_rows())
