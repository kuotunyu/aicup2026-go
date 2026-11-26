"""Build the local validation set (split v1). Stdlib only, deterministic.

Usage:
    python tools/make_val.py data/training data/local results/split_v1

Players, not games, are split: every player belongs to exactly one rank, so a
random game-level split would let a model score by recognising the player.

Split: within each (rank, games-per-player bin) the players are ordered by a
salted SHA-256 of their id and the first 20% go to validation.

Outputs (game content -> data/local, not in git; ids and hashes -> results):
  data/local/rank_val_questions.csv     official test layout, one question per val player with >=5 games
  data/local/rank_val_answers.csv       question_id, rank, player_id, num_games
  data/local/player_val_candidates.csv  pool: 400 val players x 100 games (aliased ids)
  data/local/player_val_questions.csv   one query of 5-20 other games per pool player
  data/local/player_val_answers.csv     question_id, player_id (alias), rank
  results/split_v1/players_split.csv    player_id, rank, n_games, bin, split
  results/split_v1/manifest.json        parameters, counts, SHA-256 of every file
"""

import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
SALT = "c10-split-v1"
VAL_FRAC = 0.2
BINS = [(1, 4), (5, 24), (25, 104), (105, 10**9)]
K_MIN, K_MAX = 5, 20
POOL_GAMES = 100
POOL_PLAYERS = 400
MAX_SLOTS = 20


def h(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def rng_for(*parts):
    return random.Random(int(h(*parts)[:16], 16))


def bin_of(n):
    for lo, hi in BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    raise ValueError(n)


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def question_header():
    return (["question_id", "num_games"]
            + [f"sgf_{i}" for i in range(1, MAX_SLOTS + 1)]
            + [f"color_{i}" for i in range(1, MAX_SLOTS + 1)])


def question_row(qid, games):
    sgfs = [g[2] for g in games] + [""] * (MAX_SLOTS - len(games))
    cols = [g[1] for g in games] + [""] * (MAX_SLOTS - len(games))
    return [qid, len(games)] + sgfs + cols


def main(train_dir, local_dir, results_dir):
    csv.field_size_limit(10**8)
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # player_id -> {"rank": str, "games": [(game_id, colour, sgf)]}
    players = {}
    source_hashes = {}
    for lv in LEVELS:
        path = os.path.join(train_dir, f"train_{lv}.csv")
        source_hashes[os.path.basename(path)] = sha256_file(path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for pid, gid, rank, colour, sgf in reader:
                p = players.setdefault(pid, {"rank": rank, "games": []})
                if p["rank"] != rank:
                    raise SystemExit(f"player {pid} appears in two ranks")
                p["games"].append((gid, colour, sgf))
        print(f"read {lv}", flush=True)

    # ---- player split -------------------------------------------------
    strata = defaultdict(list)
    for pid, p in players.items():
        strata[(p["rank"], bin_of(len(p["games"])))].append(pid)
    split = {}
    for key, pids in strata.items():
        pids.sort(key=lambda x: h(SALT, "split", x))
        n_val = int(VAL_FRAC * len(pids) + 0.5)
        for i, pid in enumerate(pids):
            split[pid] = "val" if i < n_val else "train"

    with open(os.path.join(results_dir, "players_split.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "rank", "n_games", "bin", "split"])
        for pid in sorted(players, key=lambda x: (LEVELS.index(players[x]["rank"]), x)):
            p = players[pid]
            w.writerow([pid, p["rank"], len(p["games"]), bin_of(len(p["games"])), split[pid]])

    val_players = [pid for pid in players if split[pid] == "val"]

    # ---- competition 2: rank questions ----------------------------------
    rank_q = [pid for pid in val_players if len(players[pid]["games"]) >= K_MIN]
    rank_q.sort(key=lambda x: h(SALT, "rank-order", x))
    rq_path = os.path.join(local_dir, "rank_val_questions.csv")
    ra_path = os.path.join(local_dir, "rank_val_answers.csv")
    with open(rq_path, "w", newline="", encoding="utf-8") as fq, \
         open(ra_path, "w", newline="", encoding="utf-8") as fa:
        wq, wa = csv.writer(fq), csv.writer(fa)
        wq.writerow(question_header())
        wa.writerow(["question_id", "rank", "player_id", "num_games"])
        for i, pid in enumerate(rank_q, 1):
            games = players[pid]["games"]
            rng = rng_for(SALT, "rank-q", pid)
            k = min(rng.randint(K_MIN, K_MAX), len(games))
            chosen = rng.sample(games, k)
            qid = f"val_q_{i:04d}"
            wq.writerow(question_row(qid, chosen))
            wa.writerow([qid, players[pid]["rank"], pid, k])

    # ---- competition 1: candidate pool + queries -------------------------
    eligible = [pid for pid in val_players if len(players[pid]["games"]) >= POOL_GAMES + K_MIN]
    eligible.sort(key=lambda x: h(SALT, "pool-pick", x))
    pool = eligible[:POOL_PLAYERS]
    pc_path = os.path.join(local_dir, "player_val_candidates.csv")
    pq_path = os.path.join(local_dir, "player_val_questions.csv")
    pa_path = os.path.join(local_dir, "player_val_answers.csv")
    with open(pc_path, "w", newline="", encoding="utf-8") as fc, \
         open(pq_path, "w", newline="", encoding="utf-8") as fq, \
         open(pa_path, "w", newline="", encoding="utf-8") as fa:
        wc, wq, wa = csv.writer(fc), csv.writer(fq), csv.writer(fa)
        wc.writerow(["game_id", "player_id", "sgf_content", "color"])
        wq.writerow(question_header())
        wa.writerow(["question_id", "player_id", "rank"])
        # Candidates and questions are written in two independent orders so
        # that row position never reveals the answer (rev 2, audit 9/24).
        cand_order = sorted(pool, key=lambda x: h(SALT, "player-order", x))
        query_order = sorted(pool, key=lambda x: h(SALT, "pq-order", x))
        draws = {}
        game_no = 0
        for pid in cand_order:
            alias = "val_p_" + h(SALT, "alias", pid)[:16]
            rng = rng_for(SALT, "pool", pid)
            games = list(players[pid]["games"])
            rng.shuffle(games)
            cand, rest = games[:POOL_GAMES], games[POOL_GAMES:]
            k = min(rng.randint(K_MIN, K_MAX), len(rest))
            draws[pid] = (alias, rest[:k])
            for gid, colour, sgf in cand:
                game_no += 1
                wc.writerow([f"val_g_{game_no:06d}", alias, sgf, colour])
        for i, pid in enumerate(query_order, 1):
            alias, query = draws[pid]
            qid = f"val_pq_{i:04d}"
            wq.writerow(question_row(qid, query))
            wa.writerow([qid, alias, players[pid]["rank"]])

    # ---- manifest --------------------------------------------------------
    counts = Counter((players[p]["rank"], split[p]) for p in players)
    rank_q_by_rank = Counter(players[p]["rank"] for p in rank_q)
    pool_by_rank = Counter(players[p]["rank"] for p in pool)
    outputs = [rq_path, ra_path, pc_path, pq_path, pa_path,
               os.path.join(results_dir, "players_split.csv")]
    manifest = {
        "version": "split_v1",
        "revision": 2,
        "revision_note": "rev 2 (2026-09-24, before any model was scored): competition-1 questions "
                         "are written in an order independent of the candidate file; contents unchanged",
        "params": {"salt": SALT, "val_frac": VAL_FRAC, "bins": BINS, "k_min": K_MIN,
                   "k_max": K_MAX, "pool_games": POOL_GAMES, "pool_players": POOL_PLAYERS},
        "source_sha256": source_hashes,
        "players": {"total": len(players), "val": len(val_players),
                    "train": len(players) - len(val_players)},
        "players_by_rank": {lv: {"train": counts[(lv, "train")], "val": counts[(lv, "val")]} for lv in LEVELS},
        "train_games": sum(len(p["games"]) for pid, p in players.items() if split[pid] == "train"),
        "val_games": sum(len(p["games"]) for pid, p in players.items() if split[pid] == "val"),
        "rank_questions": {"total": len(rank_q), "by_rank": {lv: rank_q_by_rank[lv] for lv in LEVELS}},
        "player_pool": {"players": len(pool), "eligible_val_players": len(eligible),
                        "by_rank": {lv: pool_by_rank[lv] for lv in LEVELS},
                        "candidate_games": len(pool) * POOL_GAMES},
        "output_sha256": {os.path.basename(p): sha256_file(p) for p in outputs},
    }
    with open(os.path.join(results_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: manifest[k] for k in ("players", "players_by_rank", "train_games", "val_games",
                                               "rank_questions", "player_pool")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
