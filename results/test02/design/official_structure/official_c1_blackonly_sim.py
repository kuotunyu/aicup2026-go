"""Competition 1 (player identification): opening-only retrieval on simulated pools of FIT players,
rebuilt to the OFFICIAL structure described on AIdea (read by the user on 2026-09-28):
candidates = 400 players x 100 BLACK games, questions = 400 (k = 5 / 10 / 15 / 20, 100 each),
query games BLACK only, answer always one of the 400 candidates, Top-5 output.

Adapted copy of results/prereg02/explore/opening_player_sim.py (repo file untouched). Tokens,
TF-IDF and the official metric are those of the explore script; the pool / split / k logic is new.
The split-guard statistic rho and the guard level use tools/p02_player_open.py itself (imported
read-only: pool_open_stats, guard_level) with the registered thresholds of
results/prereg02/player/guard_calibration.json.

Data discipline: training rows only via tools/c10data.load_rows(part='fit'); no validation games,
no answer files, tools/score.py never run, CPU only. Writes only next to this script.
"""
import sys
sys.dont_write_bytecode = True   # never write .pyc into the repo's tools/__pycache__

import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import LEVELS, load_rows  # noqa: E402
import p02_moves  # noqa: E402
import p02_player_open as P  # noqa: E402  (read-only use: pool_open_stats, guard_level, open_cos)

SMOKE = "--smoke" in sys.argv
OUT = os.path.join(HERE, "official_c1_blackonly_sim%s.json" % ("_smoke" if SMOKE else ""))

SGF_CHARS = 2000       # explore parse: moves from the first 2000 SGF characters
MAX_MOVES = 60
CANON_K = 6            # explore canon
N_CAND = 100
POOL_PER_LEVEL = 4 if SMOKE else 40
K_VALUES = (5, 10, 15, 20)
SEEDS_MAIN = [0] if SMOKE else [0, 1, 2, 3, 4]
SEEDS_P105 = [0] if SMOKE else [0, 1, 2]
N_LIST = (10, 20)
# (orientation, N) combinations evaluated on every pool; canon10 = registered p02 orientation
ORIENT_N = [("raw", 10), ("canon6", 10), ("canon10", 10), ("raw", 20), ("canon6", 20)]
TOKEN_SETS = ("col", "nocol", "both")

EXPLORE_JSON = os.path.join(REPO, "results", "prereg02", "explore", "opening_player_sim.json")
GUARD_JSON = os.path.join(REPO, "results", "prereg02", "player", "guard_calibration.json")

# ---------------------------------------------------------------- data (fit players only)
t0 = time.time()
rows = load_rows(os.path.join(REPO, "data", "training"),
                 os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                 part="fit", local_dir=os.path.join(REPO, "data", "local"))
n_rows = len(rows)
MV = re.compile(r";([BW])\[([a-s]{2})\]")
n_all, n_black, level = Counter(), Counter(), {}
for r in rows:
    n_all[r["player_id"]] += 1
    n_black[r["player_id"]] += r["color"] == "B"
    level[r["player_id"]] = r["rank"]
keep = {p for p in n_all if n_all[p] >= 105}          # superset of every design below
games = defaultdict(list)   # pid -> [(game_id, colour, explore_moves, p02_moves)] in file order
for r in rows:
    p = r["player_id"]
    if p in keep:
        sgf = r["sgf_content"]
        games[p].append((r["game_id"], r["color"], MV.findall(sgf[:SGF_CHARS])[:MAX_MOVES],
                         p02_moves.parse_moves(sgf)))
del rows
for p, g in games.items():
    if len({x[0] for x in g}) != len(g):
        raise ValueError("duplicate game id for player %s" % p)
t_load = time.time() - t0
print("loaded %d fit games of %d players in %.0fs" % (n_rows, len(n_all), t_load), flush=True)

black_counts = {}
for t in (100, 105, 110, 115, 120):
    c = {lv: sum(1 for p in n_all if level[p] == lv and n_black[p] >= t) for lv in LEVELS}
    c["total"] = sum(c[lv] for lv in LEVELS)
    black_counts[">=%d black games" % t] = c
file_order_black_first = sum(
    1 for p in n_all if n_black[p] >= 105 and
    [x[1] for x in games[p]] == sorted([x[1] for x in games[p]]))   # 'B' < 'W'
print(json.dumps(black_counts), flush=True)


# ---------------------------------------------------------------- tokens (explore spelling)
def tf(i, x, y):
    if i & 4:
        x, y = y, x
    if i & 1:
        x = 18 - x
    if i & 2:
        y = 18 - y
    return x, y


def enc(x, y):
    return chr(97 + x) + chr(97 + y)


def canon6(mv, K=CANON_K):
    best = None
    for i in range(8):
        s = "".join(enc(*tf(i, ord(m[0]) - 97, ord(m[1]) - 97)) for _, m in mv[:K])
        if best is None or s < best[0]:
            best = (s, i)
    i = best[1]
    return [(c, enc(*tf(i, ord(m[0]) - 97, ord(m[1]) - 97))) for c, m in mv]


def orient_mv(mv, orient):
    if orient == "raw":
        return mv
    if orient == "canon6":
        return canon6(mv)
    if orient == "canon10":
        return p02_moves.canon(mv, 10)
    raise ValueError(orient)


def col_tokens(color, mv, N):
    own = [m for c, m in mv if c == color][:N]
    toks = []
    for i, m in enumerate(own):
        toks.append("%s%d:%s" % (color, i, m))
        toks.append("%s*:%s" % (color, m))
    seq = [c + m for c, m in mv[:N]]
    for i in range(1, len(seq)):
        toks.append("%sq:%s>%s" % (color, seq[i - 1], seq[i]))
    return toks


def nocol_tokens(color, mv, N):
    own = [m for c, m in mv if c == color][:N]
    toks = ["o%d:%s" % (i, m) for i, m in enumerate(own)] + ["o*:%s" % m for m in own]
    seq = [m for c, m in mv[:N]]
    toks += ["q:%s>%s" % (seq[i - 1], seq[i]) for i in range(1, len(seq))]
    return toks


# ---------------------------------------------------------------- retrieval
def count_matrix(tok_lists):
    vocab, r, c, v = {}, [], [], []
    for i, toks in enumerate(tok_lists):
        for t, n in Counter(toks).items():
            r.append(i)
            c.append(vocab.setdefault(t, len(vocab)))
            v.append(n)
    return sp.csr_matrix((np.asarray(v, float), (np.asarray(r, np.int64), np.asarray(c, np.int64))),
                         shape=(len(tok_lists), len(vocab)))


def indicator(groups, n):
    sizes = [len(g) for g in groups]
    rr = np.repeat(np.arange(len(groups)), sizes)
    cc = np.fromiter((i for g in groups for i in g), dtype=np.int64, count=int(sum(sizes)))
    return sp.csr_matrix((np.ones(len(cc)), (rr, cc)), shape=(len(groups), n))


def cos_matrix(G, cand_groups, query_groups):
    """explore evaluate(): per player 1 + ln(count), idf = ln((1+P)/(1+df)) + 1 (df over
    candidates), L2 (+1e-12), cosine."""
    C = (indicator(cand_groups, G.shape[0]) @ G).tocsr()
    Q = (indicator(query_groups, G.shape[0]) @ G).tocsr()
    C.eliminate_zeros()
    Q.eliminate_zeros()
    Pn = C.shape[0]
    df = np.diff(C.tocsc().indptr)
    idf = np.log((1.0 + Pn) / (1.0 + df)) + 1.0

    def weigh(m):
        m = m.copy()
        m.data = 1.0 + np.log(m.data)
        m = m.multiply(idf).tocsr()
        norm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel()) + 1e-12
        return sp.diags(1.0 / norm) @ m

    return np.asarray((weigh(Q) @ weigh(C).T).toarray())


def ranks_of(S):
    """True candidate of query i is candidate i; ties -> smaller pool index (= explore's stable
    argsort)."""
    n = S.shape[0]
    st = S[np.arange(n), np.arange(n)][:, None]
    idx = np.arange(S.shape[1])[None, :]
    return 1 + (S > st).sum(1) + ((S == st) & (idx < np.arange(n)[:, None])).sum(1)


def pts(r):
    r = np.asarray(r, float)
    return np.where(r <= 5, np.exp(-(r - 1.0)), 0.0)


def summary(r, ks=None):
    r = np.asarray(r)
    out = {"score": float(pts(r).mean()), "top1": float((r == 1).mean()), "top5": float((r <= 5).mean())}
    if ks is not None:
        ks = np.asarray(ks)
        out["per_k"] = {str(k): {"score": float(pts(r[ks == k]).mean()), "top1": float((r[ks == k] == 1).mean()),
                                 "top5": float((r[ks == k] <= 5).mean()), "n": int((ks == k).sum())}
                        for k in sorted(set(ks.tolist()))}
    return out


def evaluate_pool(cand, query, ks):
    """cand / query: per pool player a list of (colour, explore_moves). All (orient, N, set)."""
    flat = [g for lst in cand for g in lst] + [g for lst in query for g in lst]
    cg, qg, pos = [], [], 0
    for lst in cand:
        cg.append(list(range(pos, pos + len(lst))))
        pos += len(lst)
    for lst in query:
        qg.append(list(range(pos, pos + len(lst))))
        pos += len(lst)
    res = {}
    by_orient = defaultdict(list)
    for o, N in ORIENT_N:
        by_orient[o].append(N)
    for o, Ns in by_orient.items():
        mvs = [(c, orient_mv(mv, o)) for c, mv in flat]
        for N in Ns:
            Gc = count_matrix([col_tokens(c, mv, N) for c, mv in mvs])
            Gn = count_matrix([nocol_tokens(c, mv, N) for c, mv in mvs])
            for ts, G in (("col", Gc), ("nocol", Gn), ("both", sp.hstack([Gc, Gn]).tocsr())):
                res["%s|N%d|%s" % (o, N, ts)] = summary(ranks_of(cos_matrix(G, cg, qg)), ks)
    return res


# ---------------------------------------------------------------- pools
def choose_pool(elig, seed):
    pool = []
    for lv in LEVELS:
        pl = sorted(elig[lv], key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        if len(pl) < POOL_PER_LEVEL:
            raise ValueError("level %s has only %d eligible players" % (lv, len(pl)))
        pool += pl[:POOL_PER_LEVEL]
    return pool


def k_list(n):
    per = n // len(K_VALUES)
    return [k for k in K_VALUES for _ in range(per)]


def assign_k_free(pool, seed):
    ks = k_list(len(pool))
    random.Random("k|%d" % seed).shuffle(ks)
    return dict(zip(pool, ks))


def assign_k_capped(pool, seed):
    """P105: k <= n_black - 100; players with the smallest cap choose first, each draw weighted by
    the remaining counts of the allowed k values (100 of each k in total)."""
    rng = random.Random("kcap|%d" % seed)
    cap = {p: min(20, n_black[p] - N_CAND) for p in pool}
    remaining = Counter(k_list(len(pool)))
    order = sorted(pool, key=lambda p: (cap[p], rng.random()))
    out = {}
    for p in order:
        allowed = [k for k in K_VALUES if k <= cap[p] and remaining[k] > 0]
        if not allowed:
            raise ValueError("cannot assign k to %s (cap %d)" % (p, cap[p]))
        k = rng.choices(allowed, weights=[remaining[k] for k in allowed])[0]
        out[p] = k
        remaining[k] -= 1
    return out


def make_pool(design, split, seed):
    """Returns pool, kmap, cand_ids {p: [gid]}, query_ids {p: [gid]}."""
    thr = 120 if design == "P120" else 105
    elig = defaultdict(list)
    for p in sorted(n_all):
        if n_black[p] >= thr:
            elig[level[p]].append(p)
    pool = choose_pool(elig, seed)
    kmap = assign_k_free(pool, seed) if design == "P120" else assign_k_capped(pool, seed)
    rng = random.Random("%s|%d" % ({"black_random": "pool", "mixed_random": "mixed"}.get(split, "none"), seed))
    cand, query = {}, {}
    for p in pool:
        if split == "mixed_random":
            g = list(games[p])
            rng.shuffle(g)
        else:
            g = [x for x in games[p] if x[1] == "B"]
            if split == "black_random":
                rng.shuffle(g)
            elif split != "black_fileorder":
                raise ValueError(split)
        k = kmap[p]
        assert len(g) >= N_CAND + k
        cand[p] = [x[0] for x in g[:N_CAND]]
        query[p] = [x[0] for x in g[N_CAND:N_CAND + k]]
    return pool, kmap, cand, query


with open(GUARD_JSON, encoding="utf-8") as f:
    GUARD = json.load(f)["guard"]


def run_pool(design, split, seed):
    ts = time.time()
    pool, kmap, cand_ids, query_ids = make_pool(design, split, seed)
    gmap = {(p, x[0]): x for p in pool for x in games[p]}
    cand = [[(gmap[(p, g)][1], gmap[(p, g)][2]) for g in cand_ids[p]] for p in pool]
    query = [[(gmap[(p, g)][1], gmap[(p, g)][2]) for g in query_ids[p]] for p in pool]
    ks = [kmap[p] for p in pool]
    cells = evaluate_pool(cand, query, ks)
    # guard statistic with the registered p02 code (both sets, raw / canon(mv, 10), p02 parse)
    g02 = {(p, g): (gmap[(p, g)][1], gmap[(p, g)][3]) for p in pool for g in cand_ids[p] + query_ids[p]}
    st = P.pool_open_stats(pool, cand_ids, query_ids, g02, strict=True)
    b_c, b_q = st.pop("b_c"), st.pop("b_q")
    guard = {}
    for o in ("raw", "canon"):
        lvl_, why = P.guard_level(st[o]["rho"], GUARD[o], b_c, b_q)
        guard[o] = {"level": lvl_, "why": why}
    # per-k median of max cosO over the real queries (p02 code path, sorted pool order)
    sp_ = sorted(pool)
    keys = sorted(g02)
    row = {k: i for i, k in enumerate(keys)}
    per_k_real_med = {}
    for o in ("raw", "canon"):
        (G,) = P.token_matrices([[g02[k] for k in keys]], o)
        cos, has = P.open_cos(G, [[row[(p, g)] for g in cand_ids[p]] for p in sp_],
                              G, [[row[(p, g)] for g in query_ids[p]] for p in sp_])
        mx = cos.max(1)
        kk = np.array([kmap[p] for p in sp_])
        per_k_real_med[o] = {str(k): float(np.median(mx[(kk == k) & has])) for k in K_VALUES}
    cb = float(np.mean([c == "B" for lst in cand for c, _ in lst]))
    qb = float(np.mean([c == "B" for lst in query for c, _ in lst]))
    rec = {"design": design, "split": split, "seed": seed, "n_players": len(pool),
           "k_counts": dict(Counter(str(k) for k in ks)), "mean_query_games": float(np.mean(ks)),
           "candidate_black_share": cb, "query_black_share": qb,
           "cells": cells, "guard_stats": st, "b_c": b_c, "b_q": b_q, "guard": guard,
           "real_max_cos_median_by_k": per_k_real_med,
           "seconds": round(time.time() - ts, 1)}
    c = cells["raw|N10|both"]
    print("%-5s %-16s seed %d  raw N10 both %.4f col %.4f nocol %.4f | canon6 both %.4f | rho raw %.4f "
          "canon %.4f -> level raw %d canon %d | p02 real_open raw %.4f (%.0fs)"
          % (design, split, seed, c["score"], cells["raw|N10|col"]["score"], cells["raw|N10|nocol"]["score"],
             cells["canon6|N10|both"]["score"], st["raw"]["rho"], st["canon"]["rho"],
             guard["raw"]["level"], guard["canon"]["level"], st["raw"]["real_open_score"],
             time.time() - ts), flush=True)
    return rec


# ---------------------------------------------------------------- reproduction of the mixed sim
def explore_make_pool(seed):
    """make_pool(seed, 'random') of opening_player_sim.py (>= 105 fit games, randint k)."""
    rng = random.Random("pool|%d" % seed)
    elig = defaultdict(list)
    for p in n_all:   # dict insertion order = first appearance in the files, as in explore
        if n_all[p] >= 105:
            elig[level[p]].append(p)
    pool = []
    for lv in LEVELS:
        pl = sorted(elig[lv], key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        pool += pl[:40]
    cand, query = [], []
    for p in pool:
        g = [(x[1], x[2]) for x in games[p]]
        rng.shuffle(g)
        k = rng.randint(5, 20)
        cand.append(g[:N_CAND])
        query.append(g[N_CAND:N_CAND + k])
    return cand, query


repro = {}
if not SMOKE:
    with open(EXPLORE_JSON, encoding="utf-8") as f:
        EX = json.load(f)
    for seed in (0, 1):
        cand, query = explore_make_pool(seed)
        flat = [g for lst in cand for g in lst] + [g for lst in query for g in lst]
        cg, qg, pos = [], [], 0
        for lst in cand:
            cg.append(list(range(pos, pos + len(lst))))
            pos += len(lst)
        for lst in query:
            qg.append(list(range(pos, pos + len(lst))))
            pos += len(lst)
        for o, key in (("raw", "raw"), ("canon6", "canon")):
            mvs = [(c, orient_mv(mv, o)) for c, mv in flat]
            Gc = count_matrix([col_tokens(c, mv, 10) for c, mv in mvs])
            Gn = count_matrix([nocol_tokens(c, mv, 10) for c, mv in mvs])
            for ts, G in (("col", Gc), ("nocol", Gn), ("both", sp.hstack([Gc, Gn]).tocsr())):
                s = summary(ranks_of(cos_matrix(G, cg, qg)))["score"]
                old = EX["results"]["random"][key][ts]["score_by_seed"][seed]
                repro["random|%s|%s|seed%d" % (key, ts, seed)] = {"new": round(s, 4), "explore_json": old,
                                                                   "identical": round(s, 4) == old}
    print("reproduction of explore mixed sim:", json.dumps(repro), flush=True)

# ---------------------------------------------------------------- run
records = []
for split in ("black_random", "black_fileorder", "mixed_random"):
    for seed in SEEDS_MAIN:
        records.append(run_pool("P120", split, seed))
for split in ("black_random", "black_fileorder"):
    for seed in SEEDS_P105:
        records.append(run_pool("P105", split, seed))


def agg(design, split, cell):
    rs = [r for r in records if r["design"] == design and r["split"] == split]
    xs = [r["cells"][cell] for r in rs]
    out = {m: round(float(np.mean([x[m] for x in xs])), 4) for m in ("score", "top1", "top5")}
    out["score_by_seed"] = [round(x["score"], 4) for x in xs]
    out["per_k"] = {k: {m: round(float(np.mean([x["per_k"][k][m] for x in xs])), 4) for m in ("score", "top1", "top5")}
                    for k in xs[0]["per_k"]}
    return out


tables = {}
for design, splits in (("P120", ("black_random", "black_fileorder", "mixed_random")),
                       ("P105", ("black_random", "black_fileorder"))):
    for split in splits:
        tables["%s|%s" % (design, split)] = {cell: agg(design, split, cell) for cell in records[0]["cells"]}

guard_summary = {}
for design, splits in (("P120", ("black_random", "black_fileorder", "mixed_random")),
                       ("P105", ("black_random", "black_fileorder"))):
    for split in splits:
        rs = [r for r in records if r["design"] == design and r["split"] == split]
        guard_summary["%s|%s" % (design, split)] = {
            o: {"rho_by_seed": [round(r["guard_stats"][o]["rho"], 4) for r in rs],
                "rho_mean": round(float(np.mean([r["guard_stats"][o]["rho"] for r in rs])), 4),
                "real_max_cos_median_by_seed": [round(r["guard_stats"][o]["real_max_cos_median"], 4) for r in rs],
                "sr_max_cos_median_by_seed": [round(r["guard_stats"][o]["sr_max_cos_median"], 4) for r in rs],
                "level_by_seed": [r["guard"][o]["level"] for r in rs],
                "why_by_seed": [r["guard"][o]["why"] for r in rs],
                "p02_real_open_score_mean": round(float(np.mean([r["guard_stats"][o]["real_open_score"] for r in rs])), 4),
                "p02_sr_open_score_mean": round(float(np.mean([r["guard_stats"][o]["sr_open_score"] for r in rs])), 4),
                "real_max_cos_median_by_k_mean": {k: round(float(np.mean([r["real_max_cos_median_by_k"][o][k] for r in rs])), 4)
                                                  for k in map(str, K_VALUES)}}
            for o in ("raw", "canon")}
        guard_summary["%s|%s" % (design, split)]["b_c_b_q"] = [(round(r["b_c"], 4), round(r["b_q"], 4)) for r in rs]

out = {
    "what": "C1 opening-only retrieval, official structure (black-only candidates and queries, "
            "k = 5/10/15/20 x 100), simulated on FIT players only",
    "smoke": SMOKE,
    "data": {"part": "fit", "games": n_rows, "players": len(n_all),
             "fit_players_by_black_games": black_counts,
             "players_>=105_black_whose_file_order_is_black_then_white": file_order_black_first},
    "designs": {
        "P120": "eligible = fit players with >= 120 black games (any k fits); %d per level chosen by "
                "sha256('<seed>|<player_id>'); k: 100 x each of 5/10/15/20 shuffled with "
                "Random('k|<seed>')" % POOL_PER_LEVEL,
        "P105": "eligible = fit players with >= 105 black games; same player choice; k capped at "
                "n_black - 100 (smallest caps choose first, draws weighted by remaining counts, "
                "100 of each k)"},
    "splits": {
        "black_random": "player's black games shuffled with Random('pool|<seed>'); candidates = first "
                        "100, query = next k",
        "black_fileorder": "player's black games in file order; candidates = first 100, query = next k",
        "mixed_random": "control (P120 players, same k): all games of both colours shuffled with "
                        "Random('mixed|<seed>'); candidates = first 100, query = next k"},
    "seeds": {"P120": SEEDS_MAIN, "P105": SEEDS_P105},
    "orientations": {"raw": "SGF coordinates", "canon6": "explore canon (D4 minimising first 6 moves)",
                     "canon10": "p02_moves.canon(mv, 10) (registered p02 orientation)"},
    "tokens": "explore spelling; N = own moves and whole-game moves used; col / nocol / both",
    "metric": "rank r of the true player among 400; e^-(r-1) if r <= 5 else 0; top1 / top5 shares",
    "guard": "rho = P.pool_open_stats (tools/p02_player_open.py, both sets, p02 parse, raw / "
             "canon(mv,10)): median over real queries of max cosO / median over self-retrieval "
             "pseudo-queries (sha256 5 parts of 20, k = randint(5,20), 10 rounds); level = "
             "P.guard_level with the thresholds in results/prereg02/player/guard_calibration.json",
    "registered_thresholds": {o: {k: GUARD[o][k] for k in ("rho_R", "rho_F", "rho_X", "theta1", "theta2")}
                              for o in ("raw", "canon")},
    "reproduction_of_explore_mixed_sim": repro,
    "tables": tables,
    "guard_summary": guard_summary,
    "records": records,
    "runtime_s": {"load": round(t_load), "total": round(time.time() - t0)},
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("done in %.0fs -> %s" % (time.time() - t0, OUT), flush=True)
