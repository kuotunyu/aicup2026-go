"""Competition 1 (player identification): opening-only retrieval on simulated pools of FIT players.

Exploration used to freeze the opening-token design of PREREG_02 (N=10 own moves, raw
orientation, both token sets). Training data only: games are read through
tools/c10data.load_rows(part="fit"), which drops the local validation players and runs the
leak check against the data/local question/candidate files. No validation games, no answer
files, no GPU.

Run from the repo root:
    python results/prereg02/explore/opening_player_sim.py
Output: opening_player_sim.json next to this script.

Consolidates the scratchpad scripts of 2026-09-26 (judge/opening_sim_rerun.py,
judge/opening_both.py, judge/opening_colour_robust.py, judge_meth/opening_sim_verify.py).
Parsing, tokenisation, pool construction, TF-IDF and scoring are unchanged; the new parts are
the colour-separated split, the canon orientation for every token set, and more seeds.
"""
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import LEVELS, load_rows  # noqa: E402

OUT = os.path.join(HERE, "opening_player_sim.json")

N_OWN = 10             # own moves per game used for tokens
SGF_CHARS = 2000       # moves are parsed from the first 2000 SGF characters
MAX_MOVES = 60         # at most 60 moves kept per game
CANON_K = 6            # canon: D4 transform chosen on the first 6 moves
MIN_GAMES = 105        # eligible players: >= 105 fit games (100 candidates + >= 5 query)
POOL_PER_LEVEL = 40    # pool = 40 players per level = 400 players
N_CAND = 100           # candidate games per player
K_MIN, K_MAX = 5, 20   # query size k ~ randint(5, 20) per player
SEEDS = [0, 1, 2, 3, 4]
VERIFY_SEEDS = [11, 12, 13]  # the seeds of the scratchpad re-check judge_meth/opening_sim_verify.py

SPLITS = {
    "random": "shuffle the player's games (rng seeded 'pool|<seed>'); candidates = first 100, "
              "query = next k (fewer if the player has < 100 + k games)",
    "fileorder": "games in file order, no shuffle; candidates = first 100, query = next k. In the "
                 "training files each player's black games come first, then the white games",
    "colour_separated": "games ordered black first then white, each colour in file order (for the "
                        "eligible fit players this is exactly the file order, see "
                        "colour_order_check); candidates = first 100, query = the LAST "
                        "q = min(k, n - 100) games, so candidates and query never overlap and "
                        "the query is mostly the opposite colour of the candidates",
}
TOKEN_SETS = {
    "col": "colour-prefixed: own positional move 'B3:dd' + own unordered move 'B*:dd' for the "
           "first 10 own moves, plus bigrams 'Bq:Bdd>Wpp' of the first 10 moves of both colours "
           "(scratchpad name 'raw+seq')",
    "nocol": "colour-free versions: 'o3:dd', 'o*:dd', bigrams 'q:dd>pp' (scratchpad name 'nocol')",
    "both": "col + nocol",
    "own": "supplementary: col without the bigrams (scratchpad name 'raw'); only used to "
           "reproduce the scratchpad numbers (raw-vs-canon comparison, and the scratchpad "
           "'raw+seq fileorder' run, which silently had no bigrams, see reproduction_of_scratchpad)",
}
ORIENTATIONS = {
    "raw": "coordinates as in the SGF",
    "canon": "each game is transformed by the D4 symmetry that makes the coordinate string of "
             "its first 6 moves lexicographically minimal (transform applied to all moves)",
}
# (split, orientation, token sets) evaluated for every seed in SEEDS
CELLS = [
    ("random", "raw", ["col", "nocol", "both", "own"]),
    ("random", "canon", ["col", "nocol", "both", "own"]),
    ("fileorder", "raw", ["col", "nocol", "both", "own"]),
    ("colour_separated", "raw", ["col", "nocol", "both", "own"]),
]
# Values in the scratchpad JSONs (score by seed, 4 decimals), to check the reproduction.
# judge/opening_colour_robust.py ran mode "raw+seq fileorder"; its tokeniser added the bigrams
# only when mode.endswith("+seq"), which is False for that string, so that run used the 'own'
# token set (no bigrams), not 'col'. It is therefore compared with fileorder|raw|own here.
SCRATCHPAD_NOTES = {
    ("fileorder", "raw", "own"): "scratchpad label 'raw+seq fileorder', but the bigrams were not "
                                 "added (mode.endswith('+seq') is False for 'raw+seq fileorder'), so "
                                 "it measured the 'own' set; the real col/fileorder value is in "
                                 "results.fileorder.raw.col",
}
# key -> (source, reported mean score, {seed: score})
SCRATCHPAD = {
    ("random", "raw", "col"): ("judge/opening_sim_rerun.json 'raw+seq N=10'", 0.901,
                               {0: 0.8888, 1: 0.9044, 2: 0.937, 3: 0.8746, 4: 0.9004}),
    ("random", "raw", "own"): ("judge/opening_sim_rerun.json 'raw N=10'", 0.8278, {0: 0.8104, 1: 0.8451}),
    ("random", "canon", "own"): ("judge/opening_sim_rerun.json 'canon N=10'", 0.7756, {0: 0.7531, 1: 0.798}),
    ("random", "raw", "both"): ("judge/opening_both.json 'both N=10'", 0.889, {0: 0.8804, 1: 0.8975}),
    ("fileorder", "raw", "both"): ("judge/opening_both.json 'both fileorder N=10'", 0.7661,
                                   {0: 0.767, 1: 0.7651}),
    ("fileorder", "raw", "own"): ("judge/opening_colour_robust.json 'raw+seq fileorder N=10'", 0.6609,
                                  {0: 0.6513, 1: 0.6705}),
    ("random", "raw", "nocol"): ("judge/opening_colour_robust.json 'nocol N=10'", 0.8365,
                                 {0: 0.8298, 1: 0.8432}),
    ("fileorder", "raw", "nocol"): ("judge/opening_colour_robust.json 'nocol fileorder N=10'", 0.7918,
                                    {0: 0.7863, 1: 0.7973}),
    ("verify", "raw", "col"): ("judge_meth/opening_sim_verify.json 'raw+seq N=10'", 0.9018,
                               {11: 0.9014, 12: 0.9003, 13: 0.9036}),
    ("verify", "raw", "own"): ("judge_meth/opening_sim_verify.json 'raw N=10'", 0.8435, {11: 0.8435}),
}


# ---------------------------------------------------------------- data (fit players only)
t0 = time.time()
rows = load_rows(os.path.join(REPO, "data", "training"),
                 os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                 part="fit", local_dir=os.path.join(REPO, "data", "local"))
n_rows = len(rows)
MV = re.compile(r";([BW])\[([a-s]{2})\]")
by_player = defaultdict(list)   # player -> [(target colour, [(colour, 'ab'), ...])] in file order
level = {}
for r in rows:
    by_player[r["player_id"]].append((r["color"], MV.findall(r["sgf_content"][:SGF_CHARS])[:MAX_MOVES]))
    level[r["player_id"]] = r["rank"]
del rows
assert all(c in ("B", "W") for g in by_player.values() for c, _ in g), "unexpected colour value"
t_load = time.time() - t0
print("loaded %d fit games of %d players in %.0fs" % (n_rows, len(by_player), t_load), flush=True)

elig = defaultdict(list)
for p, g in by_player.items():
    if len(g) >= MIN_GAMES:
        elig[level[p]].append(p)
n_elig = {lv: len(elig[lv]) for lv in LEVELS}
print("eligible players by level:", n_elig, flush=True)


def colour_sorted(g):
    return [x for x in g if x[0] == "B"] + [x for x in g if x[0] == "W"]


eligible_all = [p for lv in LEVELS for p in elig[lv]]
colour_order_check = {
    "eligible_players": len(eligible_all),
    "black_then_white_equals_file_order": sum(colour_sorted(by_player[p]) == by_player[p]
                                              for p in eligible_all),
}


# ---------------------------------------------------------------- tokens
def tf(i, x, y):
    """D4 transform i (0..7) on 0..18 coordinates."""
    if i & 4:
        x, y = y, x
    if i & 1:
        x = 18 - x
    if i & 2:
        y = 18 - y
    return x, y


def enc(x, y):
    return chr(97 + x) + chr(97 + y)


def canon(mv, K=CANON_K):
    """Choose the D4 transform making the first K moves' coordinate string minimal."""
    best = None
    for i in range(8):
        s = "".join(enc(*tf(i, ord(m[0]) - 97, ord(m[1]) - 97)) for _, m in mv[:K])
        if best is None or s < best[0]:
            best = (s, i)
    i = best[1]
    return [(c, enc(*tf(i, ord(m[0]) - 97, ord(m[1]) - 97))) for c, m in mv]


def own_tokens(color, mv, N=N_OWN):
    own = [m for c, m in mv if c == color][:N]
    toks = []
    for i, m in enumerate(own):
        toks.append("%s%d:%s" % (color, i, m))   # positional own move
        toks.append("%s*:%s" % (color, m))       # unordered own move
    return toks


def col_tokens(color, mv, N=N_OWN):
    toks = own_tokens(color, mv, N)
    seq = [c + m for c, m in mv[:N]]
    for i in range(1, len(seq)):
        toks.append("%sq:%s>%s" % (color, seq[i - 1], seq[i]))  # both-colour bigrams
    return toks


def nocol_tokens(color, mv, N=N_OWN):
    own = [m for c, m in mv if c == color][:N]
    toks = ["o%d:%s" % (i, m) for i, m in enumerate(own)] + ["o*:%s" % m for m in own]
    seq = [m for c, m in mv[:N]]
    toks += ["q:%s>%s" % (seq[i - 1], seq[i]) for i in range(1, len(seq))]
    return toks


def tokens(game, token_set, orient):
    color, mv = game
    if orient == "canon":
        mv = canon(mv)
    if token_set == "col":
        return col_tokens(color, mv)
    if token_set == "nocol":
        return nocol_tokens(color, mv)
    if token_set == "both":
        return col_tokens(color, mv) + nocol_tokens(color, mv)
    if token_set == "own":
        return own_tokens(color, mv)
    raise ValueError(token_set)


# ---------------------------------------------------------------- pools and retrieval
def make_pool(seed, split):
    """400 players (40 per level), their candidate games and query games."""
    rng = random.Random("pool|%d" % seed)
    pool = []
    for lv in LEVELS:
        pl = sorted(elig[lv], key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        pool += pl[:POOL_PER_LEVEL]
    cand, query = [], []
    for p in pool:
        g = list(by_player[p])
        if split == "random":
            rng.shuffle(g)
        elif split == "colour_separated":
            g = colour_sorted(g)
        elif split != "fileorder":
            raise ValueError(split)
        k = rng.randint(K_MIN, K_MAX)
        cand.append(g[:N_CAND])
        if split == "colour_separated":
            q = min(k, len(g) - N_CAND)
            query.append(g[len(g) - q:])
        else:
            query.append(g[N_CAND:N_CAND + k])
    return pool, cand, query


def score_of(rank):
    return math.exp(-(rank - 1)) if rank <= 5 else 0.0


def evaluate(cand, query, token_set, orient):
    """TF (1 + ln count) x IDF over the pool's candidates, L2, cosine; official metric."""
    def vec(games):
        c = Counter()
        for gm in games:
            c.update(tokens(gm, token_set, orient))
        return c
    C = [vec(x) for x in cand]
    Q = [vec(x) for x in query]
    vocab = {}
    for c in C + Q:
        for t in c:
            vocab.setdefault(t, len(vocab))

    def mat(cs):
        r, cidx, v = [], [], []
        for i, c in enumerate(cs):
            for t, n in c.items():
                r.append(i)
                cidx.append(vocab[t])
                v.append(1.0 + math.log(n))
        return sp.csr_matrix((v, (r, cidx)), shape=(len(cs), len(vocab)))

    Cm, Qm = mat(C), mat(Q)
    n = len(cand)
    df = np.asarray((Cm > 0).sum(0)).ravel()
    idf = np.log((1 + n) / (1 + df)) + 1
    Cm = Cm.multiply(idf).tocsr()
    Qm = Qm.multiply(idf).tocsr()

    def l2(m):
        norm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel()) + 1e-12
        return sp.diags(1 / norm) @ m

    S = (l2(Qm) @ l2(Cm).T).toarray()
    scores, top1, top5 = [], 0, 0
    for i in range(n):
        order = np.argsort(-S[i], kind="stable")
        r = int(np.where(order == i)[0][0]) + 1
        scores.append(score_of(r))
        top1 += r == 1
        top5 += r <= 5
    return float(np.mean(scores)), top1 / n, top5 / n


def colour_stats(cand, query):
    cb, qb, share, qn = [], [], [], []
    for c, q in zip(cand, query):
        nb = sum(col == "B" for col, _ in c)
        cb.append(nb / len(c))
        qb.append(sum(col == "B" for col, _ in q) / len(q))
        share.append(np.mean([(nb if col == "B" else len(c) - nb) / len(c) for col, _ in q]))
        qn.append(len(q))
    return cb, qb, share, qn


def summarise(runs, seeds):
    return {"score": round(float(np.mean([x[0] for x in runs])), 4),
            "top1": round(float(np.mean([x[1] for x in runs])), 4),
            "top5": round(float(np.mean([x[2] for x in runs])), 4),
            "seeds": list(seeds),
            "score_by_seed": [round(x[0], 4) for x in runs],
            "top1_by_seed": [round(x[1], 4) for x in runs],
            "top5_by_seed": [round(x[2], 4) for x in runs]}


# ---------------------------------------------------------------- run
raw_runs = defaultdict(list)          # (split, orient, token_set) -> [(score, top1, top5)] per seed
colour_diag = defaultdict(lambda: [[], [], [], []])
splits_needed = []
for split, _, _ in CELLS:
    if split not in splits_needed:
        splits_needed.append(split)
for split in splits_needed:
    ts = time.time()
    for seed in SEEDS:
        pool, cand, query = make_pool(seed, split)
        for part, vals in zip(colour_diag[split], colour_stats(cand, query)):
            part.extend(vals)
        for sp_, orient, token_sets in CELLS:
            if sp_ != split:
                continue
            for t in token_sets:
                res = evaluate(cand, query, t, orient)
                raw_runs[(split, orient, t)].append(res)
                print("%-16s %-5s %-5s seed %d  score %.4f top1 %.4f top5 %.4f"
                      % (split, orient, t, seed, *res), flush=True)
    print("split %s done in %.0fs" % (split, time.time() - ts), flush=True)

verify_runs = defaultdict(list)
for seed in VERIFY_SEEDS:
    pool, cand, query = make_pool(seed, "random")
    for t in ("col", "own"):
        res = evaluate(cand, query, t, "raw")
        verify_runs[t].append(res)
        print("verify random raw %-5s seed %d  score %.4f" % (t, seed, res[0]), flush=True)

results = {}
for (split, orient, t), runs in raw_runs.items():
    results.setdefault(split, {}).setdefault(orient, {})[t] = summarise(runs, SEEDS)

table_main = {split: {t: results[split]["raw"][t]["score"] for t in ("col", "nocol", "both")}
              for split in ("random", "fileorder", "colour_separated")}
table_orientation = {t: {o: results["random"][o][t]["score"] for o in ("raw", "canon")}
                     for t in ("col", "nocol", "both", "own")}

reproduction = {}
for key, (src, old_mean, old) in SCRATCHPAD.items():
    split, orient, t = key
    if split == "verify":
        new_raw = dict(zip(VERIFY_SEEDS, (r[0] for r in verify_runs[t])))
        name = "random|%s|%s|seeds %s" % (orient, t, ",".join(map(str, sorted(old))))
    else:
        new_raw = dict(zip(SEEDS, (r[0] for r in raw_runs[key])))
        name = "%s|%s|%s|seeds %s" % (split, orient, t, ",".join(map(str, sorted(old))))
    new_mean = round(float(np.mean([new_raw[s] for s in old])), 4)
    reproduction[name] = {"source": src,
                          "old_score": old_mean,
                          "new_score_same_seeds": new_mean,
                          "old_score_by_seed": {str(s): v for s, v in old.items()},
                          "new_score_by_seed": {str(s): round(new_raw[s], 4) for s in old},
                          "identical": new_mean == old_mean
                          and all(round(new_raw[s], 4) == v for s, v in old.items())}
    if key in SCRATCHPAD_NOTES:
        reproduction[name]["note"] = SCRATCHPAD_NOTES[key]

out = {
    "what": "Competition 1 opening-only player retrieval on simulated pools of FIT players "
            "(training data only, via tools/c10data.load_rows(part='fit'); no validation games, "
            "no answer files). Exploration used to freeze the PREREG_02 opening-token design.",
    "data": {"part": "fit", "games": n_rows, "players": len(by_player),
             "eligible_players_by_level": n_elig, "eligible_rule": "players with >= %d fit games" % MIN_GAMES},
    "setup": {"own_moves_N": N_OWN, "moves_parsed": "first %d moves found in the first %d SGF characters"
              % (MAX_MOVES, SGF_CHARS),
              "pool": "%d players per level x %d levels, chosen per seed by sha256('<seed>|<player_id>')"
              % (POOL_PER_LEVEL, len(LEVELS)),
              "candidates_per_player": N_CAND, "query_size": "k = randint(%d, %d) per player" % (K_MIN, K_MAX),
              "seeds": SEEDS,
              "vector": "per player: counts over games, weight 1 + ln(count), times "
                        "idf = ln((1 + 400) / (1 + df)) + 1 with df over the pool's candidate "
                        "vectors, L2-normalised; cosine similarity query vs candidate vectors",
              "metric": "official: rank r of the true player among 400, score e^-(r-1) for r <= 5, else 0; "
                        "mean over the 400 queries; top1 / top5 = share with r = 1 / r <= 5; "
                        "cells report the mean over seeds and the value for each seed"},
    "splits": SPLITS,
    "token_sets": TOKEN_SETS,
    "orientations": ORIENTATIONS,
    "colour_order_check": colour_order_check,
    "colour_diagnostics_by_split": {
        split: {"candidate_black_share": round(float(np.mean(d[0])), 4),
                "query_black_share": round(float(np.mean(d[1])), 4),
                "query_colour_share_in_candidates": round(float(np.mean(d[2])), 4),
                "mean_query_games": round(float(np.mean(d[3])), 4),
                "note": "means over players and seeds; query_colour_share_in_candidates = for each "
                        "query game, the share of that player's candidates with the same colour"}
        for split, d in colour_diag.items()},
    "table_main_score_raw_orientation": table_main,
    "table_raw_vs_canon_random_split": table_orientation,
    "table_supplementary_own_raw_orientation": {split: results[split]["raw"]["own"]["score"]
                                                for split in ("random", "fileorder", "colour_separated")},
    "results": results,
    "verify_seeds_random_raw": {t: summarise(r, VERIFY_SEEDS) for t, r in verify_runs.items()},
    "reproduction_of_scratchpad": reproduction,
    "runtime_s": {"load": round(t_load), "total": round(time.time() - t0)},
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(table_main, indent=1))
print(json.dumps(table_orientation, indent=1))
print("reproduction identical:", {k: v["identical"] for k, v in reproduction.items()})
print("done in %.0fs -> %s" % (time.time() - t0, OUT))
