"""prereg49 (PLAN.md 1, 2): the move cache of every training player outside E, and the T49 class list.

    PY results/prereg49/explore/p49_lib.py build      # -> data/prereg49/cache + results/prereg49/explore/data_record.json

Cache = prereg36's cache unchanged (first 720,307 games, same order) + the split v1 inner players' games + the split v1
validation players' games (official training CSVs, per level, file order; streamed, one row at a time). Only player
ids and moves are used. The validation-answer folder is never opened (paths are guarded like prereg25's).
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg36", "explore")):
    if p not in sys.path:
        sys.path.insert(0, p)
import c10data  # noqa: E402
import p36_lib as L  # noqa: E402

CACHE49 = os.path.join(REPO, "data", "prereg49", "cache")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
RECORD = os.path.join(HERE, "data_record.json")
FORBIDDEN = ("/data/local", "rank_val_answers", "player_val_answers")
N36 = 720307


def guard(p):
    q = os.path.abspath(p).replace("\\", "/").lower()
    if any(f in q for f in FORBIDDEN):
        raise SystemExit("refused path %s" % p)
    return p


def extra_players():
    """(inner players, validation players) of split v1, from the committed split file only."""
    split = c10data.read_split(guard(SPLIT_CSV))
    inner = c10data.players_for(split, "inner")
    val = {pid for pid, info in split.items() if info["split"] == "val"}
    return inner, val


def stream_rows(players):
    """Rows (player_id, game_id, rank, color, sgf) of the given players: train_<LEVEL>.csv in c10data.LEVELS order,
    file order inside a file."""
    csv.field_size_limit(10 ** 8)
    for lv in c10data.LEVELS:
        with open(guard(os.path.join(TRAIN_DIR, "train_%s.csv" % lv)), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if next(reader) != c10data.COLUMNS:
                raise SystemExit("unexpected header in train_%s.csv" % lv)
            for rec in reader:
                if rec[0] in players:
                    if rec[2] != lv:
                        raise SystemExit("rank %s in train_%s.csv" % (rec[2], lv))
                    yield rec


def build(d=CACHE49):
    t0 = time.time()
    old = L.Cache()                                     # data/prereg36/cache
    if old.n != N36:
        raise SystemExit("prereg36 cache has %d games" % old.n)
    E, D, T36, _per = L.splits(old)
    inner, val = extra_players()
    fit = set(np.unique(old.pid).tolist())
    if inner & fit or val & fit or inner & val:
        raise SystemExit("inner / validation players overlap the fit players or each other")
    Es = set(E)
    if Es & (inner | val):
        raise SystemExit("E players among the new players")
    # E games (move sequences) for the duplicate check
    e_keys = set()
    for i in np.flatnonzero(np.isin(old.pid, list(Es))).tolist():
        e_keys.add(old.game(i).tobytes())
    parts, off, meta, bad = [], [], {k: [] for k in ("player_id", "game_id", "rank", "color")}, []
    end = int(old.off[-1])
    dup_e = 0
    counts = {}
    for name, players in (("inner", inner), ("val", val)):
        n0 = len(bad)
        for rec in stream_rows(players):
            try:
                c, b = L.moves_of(rec[4]), False
            except Exception:
                c, b = np.zeros(0, np.uint16), True
            if len(c) and c.tobytes() in e_keys:
                dup_e += 1
            parts.append(c)
            end += len(c)
            off.append(end)
            bad.append(b)
            for k, v in zip(("player_id", "game_id", "rank", "color"), rec[:4]):
                meta[k].append(v)
        counts[name] = {"games": len(bad) - n0,
                        "players": len(set(meta["player_id"][n0:]))}
    if dup_e:
        raise SystemExit("%d new games equal an E game" % dup_e)
    os.makedirs(d, exist_ok=True)
    codes = np.concatenate([np.asarray(old.codes)] + parts)
    del parts
    np.save(os.path.join(d, "codes.npy"), codes)
    n_moves = int(len(codes))
    del codes
    np.save(os.path.join(d, "off.npy"), np.concatenate([old.off, np.asarray(off, np.int64)]))
    np.savez(os.path.join(d, "meta.npz"),
             player_id=np.concatenate([old.pid, np.array(meta["player_id"], old.pid.dtype)]),
             game_id=np.concatenate([old.gid, np.array(meta["game_id"], old.gid.dtype)]),
             rank=np.concatenate([old.rank, np.array(meta["rank"], old.rank.dtype)]),
             color=np.concatenate([old.color, np.array(meta["color"], old.color.dtype)]),
             bad=np.concatenate([old.bad, np.asarray(bad, bool)]))
    n = N36 + len(bad)
    info = {"format": "c10-prereg49 cache v1",
            "source": "prereg36 cache (first %d games, unchanged) + split v1 inner + validation players' games from "
                      "data/training/train_<LEVEL>.csv (LEVELS order, file order)" % N36,
            "games": n, "players": len(fit) + counts["inner"]["players"] + counts["val"]["players"],
            "moves": n_moves, "unparsed_games": int(old.bad.sum()) + int(sum(bad)),
            "codes_sha256": L.sha256_file(os.path.join(d, "codes.npy")), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(d, "cache.json"), info)
    new = Cache49(d)
    rec = check(old, new, E, D, T36)
    rec.update({"added": counts, "new_games_equal_to_an_E_game": dup_e, "cache": info,
                "inputs_sha256": {"players_split.csv": L.sha256_file(SPLIT_CSV),
                                  "prereg36_codes": old.meta["codes_sha256"]}})
    L.write_json(RECORD, rec)
    return rec


class Cache49(L.Cache):
    def __init__(self, d=CACHE49, mmap=True):
        super().__init__(d, mmap)


def e_players():
    E, _D, _T, _per = L.splits(L.Cache())
    return E


def t49(cache, E):
    """Every cache player outside E with >= T_MIN_GAMES games, sorted (= class indices)."""
    u, n = np.unique(cache.pid, return_counts=True)
    Es = set(E)
    return sorted(p for p, k in zip(u.tolist(), n.tolist()) if p not in Es and k >= L.T_MIN_GAMES)


def check(old, new, E, D, T36):
    a = np.asarray(new.codes[:int(old.off[-1])])
    same = bool(np.array_equal(a, np.asarray(old.codes)) and np.array_equal(new.off[:N36 + 1], old.off)
                and all(np.array_equal(getattr(new, k)[:N36], getattr(old, k))
                        for k in ("pid", "gid", "rank", "color", "bad")))
    if not same:
        raise SystemExit("the first %d games of the new cache differ from prereg36's" % N36)
    T = t49(new, E)
    if set(T) & set(E):
        raise SystemExit("T49 contains E players")
    if not set(T36) <= set(T):
        raise SystemExit("T49 does not contain prereg36's T")
    u, n = np.unique(new.pid, return_counts=True)
    cnt = dict(zip(u.tolist(), n.tolist()))
    return {"format": "c10-prereg49 data record v1",
            "first_games_identical_to_prereg36": same,
            "E": {"players": len(E), "sha256_sorted_ids": L.sha256_ids(E)},
            "T36": {"players": len(T36), "sha256_sorted_ids": L.sha256_ids(T36)},
            "T49": {"players": len(T), "games": int(sum(cnt[p] for p in T)), "sha256_sorted_ids": L.sha256_ids(T),
                    "from_D": len(set(T) & set(D)), "added_vs_T36": len(T) - len(T36)},
            "T49_and_E_disjoint": True}


if __name__ == "__main__":
    if sys.argv[1:] != ["build"]:
        raise SystemExit("usage: p49_lib.py build")
    print(json.dumps(build(), indent=1))
