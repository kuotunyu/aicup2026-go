"""prereg44 explore: player roles, move caches and S25's held-out questions (PLAN.md 2, 3.1, 5). CPU only.

    S=results/prereg44/explore; PY=.venv/Scripts/python.exe (repository root on the subst drive)
    $PY $S/p44_data.py roles               # seam D only (prints, writes nothing)
    $PY $S/p44_data.py cache --part train  # data/prereg44/cache/train  (A players, every official game)
    $PY $S/p44_data.py cache --part inner  # data/prereg44/cache/inner  (inner_keys.npz order)
    $PY $S/p44_data.py cache --part hold   # data/prereg44/hold/hold.npz + data/prereg44/cache/hold
    $PY $S/p44_data.py record              # -> results/prereg44/explore/data_record.json (all three caches, seams)

Reads only the official training CSVs, the split v1 file, inner_keys.npz (prereg02, not an answer file), and for the
held-out questions the files prereg40's stage 0b reads (probe_meta / sets.json / v_meta; prereg40's p40_lib.Rows
memory-maps the probe and V feature arrays but no feature is read here). Never the validation answers.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p44_lib as K  # noqa: E402

HOLD_DIR = os.path.join(K.DATA, "hold")
HOLD_NPZ = os.path.join(HOLD_DIR, "hold.npz")
RECORD = os.path.join(HERE, "data_record.json")
FORMAT = "c10-prereg44 data record v1"


def log(*x):
    print(*x, flush=True)


def ids_of(keys):
    return {k.split(":", 1)[1] for k in keys}


def cmd_roles(_a):
    rl = K.roles()
    sd = K.seam_d(rl)
    log(sd)
    return 0 if sd["ok"] else 1


def build_train():
    rl = K.roles()
    sd = K.seam_d(rl)
    if not sd["ok"]:
        raise SystemExit(f"seam D failed: {sd}")
    want = ids_of(rl["A"])
    keys = rl["A"]

    def rows():
        for r in K.stream_rows(want):
            if K.key(r["rank"], r["player_id"]) not in keys:
                raise SystemExit(f"player {r['player_id']} of {r['rank']} is not an A key")
            yield r

    meta = K.build_cache(rows(), os.path.join(K.CACHE, "train"), "train",
                         "official training CSVs, every game of the A players (prereg25 salt, holdout_bit == 0)")
    log(meta)
    return meta


def build_inner():
    with np.load(K.guard(K.INNER_KEYS), allow_pickle=False) as z:
        ik = {k: np.asarray(z[k]).astype(str) for k in ("rank", "game_id", "player_id", "color")}
    rl = K.roles()
    if {K.key(r, p) for r, p in zip(ik["rank"], ik["player_id"])} != rl["inner"]:
        raise SystemExit("inner_keys.npz players differ from c10data's inner players")
    want = set(ik["player_id"].tolist())
    by = {}
    for r in K.stream_rows(want):
        by[(r["rank"], r["game_id"])] = r
    rows = []
    for rk, g, p, c in zip(ik["rank"], ik["game_id"], ik["player_id"], ik["color"]):
        r = by[(rk, g)]
        if r["player_id"] != p or r["color"] != c:
            raise SystemExit(f"inner row ({rk}, {g}): player / colour differ from inner_keys.npz")
        rows.append(r)
    del by
    meta = K.build_cache(iter(rows), os.path.join(K.CACHE, "inner"), "inner",
                         "official training CSVs, the inner games in results/prereg02/rank/dumps/inner_keys.npz order")
    meta["inner_keys_sha256"] = K.sha256_file(K.INNER_KEYS)
    K.write_json(os.path.join(K.CACHE, "inner", "cache.json"), meta)
    log(meta)
    return meta


def hold_questions():
    """S25's 4,000 held-out questions (p40_lib.Rows + split_pools + p19_lib.holdout_set) as game keys."""
    p40 = os.path.join(K.REPO, "results", "prereg40", "explore")
    if p40 not in sys.path:
        sys.path.insert(0, p40)
    import json
    import p40_lib as L40
    import p23_train as T23
    import p19_lib as K19
    prep = T23.ProbeOnly(L40.PROBE_PREP)
    rows = L40.Rows(T23.probe_colours(prep))
    _ra, _rh, _pa, pool_h = L40.split_pools(rows)
    hold, stats = K19.holdout_set(pool_h, L40.HOLD_PER, L40.HOLD_SEED)
    tr = K.read_json(L40.S25_TRAIN_RECORD)["players"]
    seam_h = {"holdout_questions": [{str(n): int(len(v[1])) for n, v in hold.items()}, tr["holdout_questions"]],
              "holdout_sampling": [stats, tr["holdout_sampling"]]}
    seam_h["ok"] = bool(all(a == b for a, b in seam_h.values()))
    if not seam_h["ok"]:
        raise SystemExit(f"seam H failed: {seam_h}")
    games = {}
    for s, p in T23.SETS_JSON.items():
        with open(K.guard(p), encoding="utf-8") as f:
            games[s] = json.load(f)["probe"]["games"]
    with open(K.guard(os.path.join(K.REPO, "data", "prereg25", "v", "sets.json")), encoding="utf-8") as f:
        vgames = json.load(f)["probe"]["games"]
    with np.load(K.guard(os.path.join(L40.V_PREP, "v_meta.npz")), allow_pickle=False) as z:
        vref = np.asarray(z["ref"]).astype(np.int64)

    def gkey(i):
        i = int(i)
        if i < rows.n_probe:
            g = games[str(prep.src[i])][int(prep.ref[i])]
        else:
            g = vgames[int(vref[i - rows.n_probe])]
        k = K.key(g["rank"], g["player_id"])
        if k != str(rows.keys[i]) or g["color"] != str(rows.col[i]):
            raise SystemExit(f"row {i}: sets.json game differs from the row's player / colour")
        return (g["rank"], g["game_id"], g["player_id"], g["color"])

    out = {}
    for n, (idx, y) in hold.items():
        out[n] = ([[gkey(i) for i in q] for q in idx], y)
    return out, seam_h


def build_hold():
    q, seam_h = hold_questions()
    uniq = sorted({g for n in q for qq in q[n][0] for g in qq}, key=lambda t: (K.LEVELS.index(t[0]), t[1]))
    pos = {g[:2]: i for i, g in enumerate(uniq)}
    rl = K.roles()
    hkeys = {K.key(g[0], g[2]) for g in uniq}
    if not hkeys <= rl["H"] or hkeys & rl["A"]:
        raise SystemExit("held-out question games outside H")
    want = {g[2] for g in uniq}
    rows = [None] * len(uniq)
    for r in K.stream_rows(want):
        i = pos.get((r["rank"], r["game_id"]))
        if i is not None:
            if (r["rank"], r["game_id"], r["player_id"], r["color"]) != uniq[i]:
                raise SystemExit(f"held-out game {uniq[i]}: CSV row differs")
            rows[i] = r
    if any(r is None for r in rows):
        raise SystemExit("held-out game missing from the CSVs")
    meta = K.build_cache(iter(rows), os.path.join(K.CACHE, "hold"), "hold",
                         "official training CSVs, the distinct games of S25's 4,000 held-out questions")
    del rows
    os.makedirs(K.guard(HOLD_DIR), exist_ok=True)
    arrs = {}
    for n, (qs, y) in q.items():
        arrs[f"idx_{n}"] = np.array([[pos[g[:2]] for g in qq] for qq in qs], np.int64)
        arrs[f"y_{n}"] = np.asarray(y, np.int64)
        arrs[f"player_{n}"] = np.array([K.key(qq[0][0], qq[0][2]) for qq in qs])
        arrs[f"colour_{n}"] = np.array([qq[0][3] for qq in qs])
        for qq in qs:
            if len({(g[0], g[2], g[3]) for g in qq}) != 1:
                raise SystemExit("a held-out question mixes players or colours")
    np.savez(HOLD_NPZ, ks=np.array(sorted(q)), **arrs)
    rec = {"seam_h": seam_h, "hold_npz_sha256": K.sha256_file(HOLD_NPZ),
           "questions": {str(n): int(len(q[n][1])) for n in q}, "distinct_games": len(uniq),
           "players": len(hkeys)}
    K.write_json(os.path.join(HOLD_DIR, "hold_record.json"), rec)
    log(meta, rec)
    return meta


def cmd_cache(a):
    t0 = time.time()
    {"train": build_train, "inner": build_inner, "hold": build_hold}[a.part]()
    log(f"done in {time.time() - t0:.0f}s")
    return 0


def cmd_record(_a):
    rl = K.roles()
    sd = K.seam_d(rl)
    caches = {p: K.read_json(os.path.join(K.CACHE, p, "cache.json")) for p in ("train", "inner", "hold")}
    for p, m in caches.items():
        d = os.path.join(K.CACHE, p)
        if K.sha256_file(os.path.join(d, "codes.npy")) != m["codes_sha256"]:
            raise SystemExit(f"{p}: codes.npy SHA differs from cache.json")
    tr = K.open_cache(os.path.join(K.CACHE, "train"))
    tkeys = {K.key(r, p) for r, p in zip(tr.rank.tolist(), tr.pid.tolist())}
    idx = K.level_index(tr)
    hold_rec = K.read_json(os.path.join(HOLD_DIR, "hold_record.json"))
    rec = {"format": FORMAT, "plan": "PLAN.md 2, 3.1, 5 (data statistics, no model number)",
           "seam_d": sd, "train_players_equal_A": bool(tkeys == rl["A"]),
           "train_inner_overlap_player_id": len(ids_of(tkeys) & ids_of(rl["inner"])),
           "train_H_overlap_player_id": len(ids_of(tkeys) & ids_of(rl["H"])),
           "caches": caches, "hold": hold_rec,
           "sampler": {"players_per_level_ge_min_games": [int(len(b)) for b in idx["by_level"]],
                       "min_player_games": K.MIN_PLAYER_GAMES, "dropped_players": idx["dropped_players"],
                       "sampled_games": int(len(idx["games"]))},
           "code_sha256_lf": {f: K.sha_lf(os.path.join(HERE, f)) for f in ("p44_lib.py", "p44_data.py")}}
    ok = (sd["ok"] and rec["train_players_equal_A"] and rec["train_inner_overlap_player_id"] == 0
          and rec["train_H_overlap_player_id"] == 0 and hold_rec["seam_h"]["ok"])
    rec["ok"] = bool(ok)
    K.write_json(RECORD, rec)
    log({k: rec[k] for k in ("ok", "train_players_equal_A", "train_inner_overlap_player_id", "sampler")})
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("roles")
    c = sub.add_parser("cache")
    c.add_argument("--part", required=True, choices=["train", "inner", "hold"])
    sub.add_parser("record")
    a = ap.parse_args(argv)
    return {"roles": cmd_roles, "cache": cmd_cache, "record": cmd_record}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
