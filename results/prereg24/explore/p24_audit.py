"""prereg24 phase 0 (competition 2): read-only data audit for a larger S19 training set (CPU, .venv).

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe
    results/prereg24/explore/p24_audit.py   -> results/prereg24/explore/audit.json

What it counts (no model, no score, no feature array):
  - the fit rows of split v1 (tools/c10data.load_rows part 'fit', local_dir None: training players minus inner
    players; validation players are never returned) and the inner players (part 'inner', player ids only);
  - the probe games of S19 (P1 + P2 + P3 sets.json) and the rows of probe_meta.npz;
  - the fit games outside the probe ("extra"), per level and colour, their players, and how they fall on S19's A / H
    player split (salt c10-prereg19);
  - eligible players per (n, level, colour) for the question sampler with the probe alone and with probe + extra;
  - move-sequence duplicates of the extra games against the probe, the inner games and within themselves;
  - moves of the extra games (P1 evaluates one position per move) and the 4090 hours at the throughput the P3 probe
    run measured with the same p1_extract.py (data/prereg09/p3/run_log.jsonl 'run end' line).
Never reads data/local, never reads validation games, never runs tools/score.py.
"""

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg19", "explore")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import c10data  # noqa: E402
import p19_lib as K19  # noqa: E402

TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
SETS = {"p1_probe": os.path.join(REPO, "data", "prereg08", "p1", "sets.json"),
        "p2_probe": os.path.join(REPO, "data", "prereg08", "p2", "sets.json"),
        "p3_probe": os.path.join(REPO, "data", "prereg09", "p3", "sets.json")}
PROBE_META = os.path.join(REPO, "data", "prereg17", "prep", "probe_meta.npz")
P3_RUNLOG = os.path.join(REPO, "data", "prereg09", "p3", "run_log.jsonl")
P2_RUNLOG = os.path.join(REPO, "data", "prereg08", "p2", "run_log.jsonl")
OUT = os.path.join(HERE, "audit.json")
FORBIDDEN = ("/data/local",)
LEVELS = c10data.LEVELS


def guard(p):
    q = os.path.abspath(p).replace("\\", "/").lower()
    if any(f in q for f in FORBIDDEN):
        raise SystemExit(f"refused path {p}")
    return p


def move_hash(sgf):
    """sha256 of the move sequence (original orientation): every ';B[..]' / ';W[..]' token in order."""
    toks = []
    i = 0
    while True:
        j = sgf.find(";", i)
        if j < 0:
            break
        k = sgf.find("]", j)
        if k < 0:
            break
        t = sgf[j + 1:k + 1]
        if t[:2] in ("B[", "W["):
            toks.append(t)
        i = j + 1
    return hashlib.sha256("".join(toks).encode("ascii")).hexdigest(), len(toks)


def run_end(path):
    with open(guard(path), encoding="utf-8") as f:
        for line in f:
            if '"run end"' in line:
                d = json.loads(line)
    return {"positions": d["positions"], "seconds": d["seconds"], "positions_per_s": d["positions"] / d["seconds"]}


def eligible(games_by_pc, level_of, n):
    """players per level and colour with >= n games of that colour."""
    out = [[0, 0] for _ in LEVELS]
    for (pk, c), cnt in games_by_pc.items():
        if cnt >= n:
            out[level_of[pk]][0 if c == "B" else 1] += 1
    return out


def main():
    t0 = time.time()
    fit = c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part="fit", local_dir=None, check_counts=True)
    inner = c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part="inner", local_dir=None, check_counts=True)
    split = c10data.read_split(SPLIT_CSV)
    split_counts = Counter(v["split"] for v in split.values())
    inner_players = {r["player_id"] for r in inner}
    fit_players = {r["player_id"] for r in fit}
    probe = {}
    for s, p in SETS.items():
        with open(guard(p), encoding="utf-8") as f:
            for g in json.load(f)["probe"]["games"]:
                key = (g["rank"], g["game_id"])
                if key in probe:
                    raise SystemExit(f"probe game twice: {key}")
                probe[key] = s
    with np.load(guard(PROBE_META), allow_pickle=False) as z:
        meta_rows = int(len(z["y"]))
        meta_players = len(set(np.asarray(z["players"]).astype(str).tolist()))
    fit_keys = {(r["rank"], r["game_id"]) for r in fit}
    probe_not_fit = len(set(probe) - fit_keys)
    level_ix = {lv: i for i, lv in enumerate(LEVELS)}
    # ---- per row facts
    h_probe, h_inner = set(), set()
    probe_pc, all_pc = Counter(), Counter()
    level_of = {}
    extra = []
    for r in fit:
        pk = f"{r['rank']}:{r['player_id']}"
        level_of[pk] = level_ix[r["rank"]]
        all_pc[(pk, r["color"])] += 1
        if (r["rank"], r["game_id"]) in probe:
            probe_pc[(pk, r["color"])] += 1
            h_probe.add(move_hash(r["sgf_content"])[0])
        else:
            extra.append(r)
    for r in inner:
        h_inner.add(move_hash(r["sgf_content"])[0])
    per_level = {}
    ex_hashes = Counter()
    moves_total = 0
    ab = {"A": {"games": 0, "players": set(), "moves": 0}, "H": {"games": 0, "players": set(), "moves": 0}}
    extra_players = set()
    per_player_extra = Counter()
    dup_probe = dup_inner = 0
    for lv in LEVELS:
        per_level[lv] = {"fit_games": 0, "fit_players": set(), "probe_games": 0, "extra_games": 0,
                         "extra_games_B": 0, "extra_games_W": 0, "extra_players": set(), "extra_moves": 0,
                         "extra_A_games": 0, "extra_H_games": 0}
    for r in fit:
        d = per_level[r["rank"]]
        d["fit_games"] += 1
        d["fit_players"].add(r["player_id"])
        if (r["rank"], r["game_id"]) in probe:
            d["probe_games"] += 1
    for r in extra:
        pk = f"{r['rank']}:{r['player_id']}"
        h, nm = move_hash(r["sgf_content"])
        ex_hashes[h] += 1
        dup_probe += h in h_probe
        dup_inner += h in h_inner
        moves_total += nm
        d = per_level[r["rank"]]
        d["extra_games"] += 1
        d["extra_games_" + r["color"]] += 1
        d["extra_players"].add(r["player_id"])
        d["extra_moves"] += nm
        role = "H" if K19.holdout_bit(pk) else "A"
        d[f"extra_{role}_games"] += 1
        ab[role]["games"] += 1
        ab[role]["players"].add(pk)
        ab[role]["moves"] += nm
        extra_players.add(pk)
        per_player_extra[pk] += 1
    dup_within = sum(c - 1 for c in ex_hashes.values() if c > 1)
    probe_pk = {pk for (pk, _c) in probe_pc}
    new_players = extra_players - probe_pk
    elig = {}
    for n in K19.KS:
        elig[str(n)] = {"probe": eligible(probe_pc, level_of, n), "probe_plus_extra": eligible(all_pc, level_of, n)}
    pp = np.array(sorted(per_player_extra.values()))
    thr = {"p3": run_end(P3_RUNLOG), "p2": run_end(P2_RUNLOG)}
    hours = {k: moves_total / v["positions_per_s"] / 3600.0 for k, v in thr.items()}
    hours_a = {k: ab["A"]["moves"] / v["positions_per_s"] / 3600.0 for k, v in thr.items()}
    rec = {
        "format": "c10-prereg24 explore audit v1",
        "what": "read-only counts: fit rows of split v1 minus the S19 probe; no model, no score, no data/local",
        "split_v1_players_by_split": dict(split_counts),
        "fit": {"players": len(fit_players), "games": len(fit)},
        "inner": {"players": len(inner_players), "games": len(inner), "players_in_fit": len(inner_players & fit_players)},
        "probe": {"games": len(probe), "games_per_set": dict(Counter(probe.values())), "players": len(probe_pk),
                  "probe_meta_rows": meta_rows, "probe_meta_players": meta_players,
                  "probe_games_not_in_fit": probe_not_fit},
        "extra": {"games": len(extra), "players": len(extra_players), "players_not_in_probe": len(new_players),
                  "moves": moves_total, "mean_moves": moves_total / max(1, len(extra)),
                  "games_per_player": {"min": int(pp.min()), "median": float(np.median(pp)), "max": int(pp.max()),
                                       "p90": float(np.percentile(pp, 90)),
                                       "players_ge_20": int((pp >= 20).sum()), "players_ge_100": int((pp >= 100).sum())},
                  "top10pct_players_share": float(np.sort(pp)[::-1][:max(1, len(pp) // 10)].sum() / pp.sum()),
                  "on_S19_split": {k: {"games": v["games"], "players": len(v["players"]), "moves": v["moves"]}
                                   for k, v in ab.items()},
                  "move_duplicates": {"vs_probe": dup_probe, "vs_inner": dup_inner, "within_extra": dup_within,
                                      "method": "sha256 of the ';B[..]' / ';W[..]' token sequence, original orientation"}},
        "per_level": {lv: {k: (len(v) if isinstance(v, set) else v) for k, v in d.items()}
                      for lv, d in per_level.items()},
        "eligible_players_n_level_colour": elig,
        "gpu_estimate": {"throughput_runs": thr,
                         "hours_4090_all_extra": hours,
                         "hours_4090_extra_of_A_players": hours_a,
                         "note": "same code (results/prereg08/p1/p1_extract.py run, fp16, batch 1024) on fit games; "
                                 "positions = moves; P3 used --workers 12, P2 --workers 20"},
        "seconds": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: rec[k] for k in ("fit", "inner", "probe", "extra", "gpu_estimate")}, indent=1))


if __name__ == "__main__":
    main()
