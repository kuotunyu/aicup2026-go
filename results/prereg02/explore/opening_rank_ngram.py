"""Competition 2 (rank): level-conditional opening n-gram, counted on FIT games and evaluated on
INNER pseudo-questions.

Exploration used to freeze the PREREG_02 opening n-gram (M=12 moves, alpha=20, aggregation
'mean', decision rule B). Training data only: games are read through
tools/c10data.load_rows(part="fit") and load_rows(part="inner"), which drop the local
validation players and run the leak check against the data/local question/candidate files.
No validation games, no answer files, no GPU.

Run from the repo root:
    python results/prereg02/explore/opening_rank_ngram.py
Output: opening_rank_ngram.json next to this script.

Consolidates the scratchpad scripts p02/opening_rank_nb.py (M=12) and
p02/opening_rank_nb_m20.py (M=20) of 2026-09-26. Canonical form, smoothing formula, stopping
rule, pseudo-questions, aggregation and decision rules are unchanged. Two changes that do not
alter any number:
  * only the (colour, prefix) contexts and (colour, prefix, move) entries that occur in the
    evaluated inner games are kept while counting the fit games (the counts of those keys are
    the same integers as in the full table, and contexts never seen in fit are dropped, so
    'stop at the first unseen context' behaves exactly as before); this keeps M=20 in memory;
  * the pseudo-questions are drawn once and reused for every cell (the source redrew the same
    questions with the same seeds for every cell).
"""
import itertools
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
from c10data import LEVELS, load_rows  # noqa: E402

OUT = os.path.join(HERE, "opening_rank_ngram.json")

M_GRID = (12, 20)
ALPHAS = (5.0, 20.0, 60.0)
AGGREGATIONS = {"mean": "beta = 1/k", "sqrt": "beta = 1/sqrt(k)", "sum": "beta = 1"}
RULES = ("A", "B")
ROUNDS = 10              # pseudo-question rounds, rng seeded 1000 + round
K_MIN, K_MAX = 5, 20     # games per pseudo-question: k = min(randint(5, 20), n)
MIN_INNER_GAMES = 5      # inner players with fewer games get no pseudo-question
FROZEN = {"M": 12, "alpha": 20.0, "aggregation": "mean", "rule": "B"}
# Cells computed by the scratchpad exploration before M and alpha were frozen.
LOOKED_AT_BEFORE_FREEZE = {(12, 5.0, a) for a in AGGREGATIONS} | {(12, 20.0, a) for a in AGGREGATIONS} \
    | {(20, 20.0, "mean"), (20, 60.0, "mean")}
# Rank-balanced means in the scratchpad JSONs (4 decimals), to check the reproduction.
SCRATCHPAD = {
    "p02/opening_rank_nb.json": (12, {"alpha5.0_mean_A": 0.4071, "alpha5.0_mean_B": 0.412,
                                      "alpha5.0_sqrt_A": 0.4071, "alpha5.0_sqrt_B": 0.4067,
                                      "alpha5.0_sum_A": 0.4071, "alpha5.0_sum_B": 0.4072,
                                      "alpha20.0_mean_A": 0.4143, "alpha20.0_mean_B": 0.4228,
                                      "alpha20.0_sqrt_A": 0.4143, "alpha20.0_sqrt_B": 0.4162,
                                      "alpha20.0_sum_A": 0.4143, "alpha20.0_sum_B": 0.4151}),
    "p02/opening_rank_nb_m20.json": (20, {"alpha20.0_mean_A": 0.4113, "alpha20.0_mean_B": 0.4185,
                                          "alpha60.0_mean_A": 0.4126, "alpha60.0_mean_B": 0.424}),
}

MOVE = re.compile(r";([BW])\[([a-s])([a-s])\]")
M_MAX = max(M_GRID)
LI = {lv: i for i, lv in enumerate(LEVELS)}
E1 = math.exp(-1)
SYMS = [(r, f) for r in range(4) for f in (False, True)]


def tf(p, s):
    x, y = p
    rot, flip = s
    if flip:
        x = 18 - x
    for _ in range(rot):
        x, y = y, 18 - x
    return (x, y)


TF = [{(x, y): tf((x, y), s) for x in range(19) for y in range(19)} for s in SYMS]


def canon(raw, M):
    """First M moves, D4 transform giving the lexicographically smallest tuple of points."""
    seq = [(ord(raw[i]) - 97, ord(raw[i + 1]) - 97) for i in range(0, min(len(raw), 2 * M), 2)]
    return min(tuple(T[p] for p in seq) for T in TF)


def load(part):
    """[(player_id, level, target colour 'B'/'W', first M_MAX move coordinates as 'ddpp...')]"""
    rows = load_rows(os.path.join(REPO, "data", "training"),
                     os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                     part=part, local_dir=os.path.join(REPO, "data", "local"))
    out = [(r["player_id"], r["rank"], "B" if r["color"].upper().startswith("B") else "W",
            "".join(m.group(2) + m.group(3) for m in itertools.islice(MOVE.finditer(r["sgf_content"]), M_MAX)))
           for r in rows]
    del rows
    return out


t0 = time.time()
fit = load("fit")
print("fit", len(fit), "games, %.0fs" % (time.time() - t0), flush=True)
inner = load("inner")
print("inner", len(inner), "games, %.0fs" % (time.time() - t0), flush=True)
t_load = time.time() - t0

by_p = defaultdict(list)
lvl = {}
for pid, lv, c, raw in inner:
    by_p[pid].append((c, raw))
    lvl[pid] = lv

# pseudo-questions (same draws as the source: one rng per round, players in sorted order)
questions = []
for rnd in range(ROUNDS):
    rng = random.Random(1000 + rnd)
    for pid in sorted(by_p):
        n = len(by_p[pid])
        if n < MIN_INNER_GAMES:
            continue
        k = min(rng.randint(K_MIN, K_MAX), n)
        questions.append((pid, rng.sample(range(n), k)))
n_q_players = sum(len(g) >= MIN_INNER_GAMES for g in by_p.values())
q_per_level = defaultdict(int)
for pid, _ in questions:
    q_per_level[lvl[pid]] += 1


def score(pred, truth):
    d = abs(pred - truth)
    return 1.0 if d == 0 else (E1 if d == 1 else 0.0)


def rule_b(p):
    best, bi = -1, 0
    for k in range(10):
        v = p[k] + E1 * ((p[k - 1] if k > 0 else 0) + (p[k + 1] if k < 9 else 0))
        if v > best + 1e-15:
            best, bi = v, k
    return bi


cells = []
count_stats = {}
for M in M_GRID:
    tm = time.time()
    inner_seq = {pid: [(c, canon(raw, M)) for c, raw in g] for pid, g in by_p.items()}
    # keys that the inner games can ask for
    ctx = {}
    cnt = {}
    for g in inner_seq.values():
        for c, seq in g:
            for L in range(len(seq)):
                ctx.setdefault((c, seq[:L]), [0] * 10)
                cnt.setdefault((c, seq[:L], seq[L]), [0] * 10)
    # count the fit games on those keys (ctx keys are prefix-closed, so stop at the first miss)
    for _, lv, c, raw in fit:
        i = LI[lv]
        seq = canon(raw, M)
        for L in range(len(seq)):
            pre = seq[:L]
            cx = ctx.get((c, pre))
            if cx is None:
                break
            cx[i] += 1
            cm = cnt.get((c, pre, seq[L]))
            if cm is not None:
                cm[i] += 1
    ctx = {k: v for k, v in ctx.items() if any(v)}   # never seen in fit -> unseen context
    cnt = {k: v for k, v in cnt.items() if any(v)}
    count_stats["M=%d" % M] = {"inner_contexts_seen_in_fit": len(ctx), "inner_entries_seen_in_fit": len(cnt),
                               "count_seconds": round(time.time() - tm)}
    print("M=%d counted in %.0fs" % (M, time.time() - tm), flush=True)

    def game_ll(c, seq, alpha):
        """per-level log-likelihood of the opening, Dirichlet-smoothed toward the pooled model"""
        ll = [0.0] * 10
        for L in range(len(seq)):
            pre = seq[:L]
            cx = ctx.get((c, pre))
            cm = cnt.get((c, pre, seq[L]))
            if cx is None:
                break  # unseen context: no information from here on
            tot_cx = sum(cx)
            tot_cm = sum(cm) if cm else 0
            pg = (tot_cm + 1.0) / (tot_cx + 361.0)
            for i in range(10):
                ll[i] += math.log(((cm[i] if cm else 0) + alpha * pg) / (cx[i] + alpha))
        return ll

    for alpha in ALPHAS:
        gll = {pid: [game_ll(c, s, alpha) for c, s in g] for pid, g in inner_seq.items()}
        for agg in AGGREGATIONS:
            for rule in RULES:
                per_level = defaultdict(list)
                for pid, idx in questions:
                    k = len(idx)
                    s = [sum(gll[pid][j][i] for j in idx) for i in range(10)]
                    beta = {"mean": 1.0 / k, "sqrt": 1.0 / math.sqrt(k), "sum": 1.0}[agg]
                    m = max(s)
                    w = [math.exp(beta * (x - m)) for x in s]
                    z = sum(w)
                    p = [x / z for x in w]
                    pred = max(range(10), key=lambda i: p[i]) if rule == "A" else rule_b(p)
                    per_level[LI[lvl[pid]]].append(score(pred, LI[lvl[pid]]))
                rb = sum(sum(v) / len(v) for v in per_level.values()) / 10
                allv = [x for v in per_level.values() for x in v]
                cell = {"M": M, "alpha": alpha, "aggregation": agg, "rule": rule,
                        "rank_balanced_mean": round(rb, 4),
                        "question_mean": round(sum(allv) / len(allv), 4),
                        "looked_at_before_freeze": (M, alpha, agg) in LOOKED_AT_BEFORE_FREEZE,
                        "frozen_choice": (M == FROZEN["M"] and alpha == FROZEN["alpha"]
                                          and agg == FROZEN["aggregation"] and rule == FROZEN["rule"])}
                if agg == "mean":
                    cell["per_level_mean"] = {LEVELS[i]: round(sum(per_level[i]) / len(per_level[i]), 4)
                                              for i in range(10)}
                cells.append(cell)
                print("M=%d alpha=%g %-4s %s  rank-balanced %.4f" % (M, alpha, agg, rule, rb), flush=True)
    del ctx, cnt, inner_seq

grid = {}
for c in cells:
    if c["aggregation"] == "mean":
        grid.setdefault("M=%d" % c["M"], {}).setdefault("alpha=%g" % c["alpha"], {})[c["rule"]] = \
            c["rank_balanced_mean"]

lookup = {(c["M"], c["alpha"], c["aggregation"], c["rule"]): c["rank_balanced_mean"] for c in cells}
reproduction = {}
for src, (M, old) in SCRATCHPAD.items():
    for key, v in old.items():
        a, agg, rule = key.split("_")
        new = lookup[(M, float(a[len("alpha"):]), agg, rule)]
        reproduction["M=%d %s" % (M, key)] = {"source": src, "old": v, "new": new, "identical": new == v}

out = {
    "what": "Competition 2 opening n-gram: per-level model of the first M moves (D4-canonical), "
            "counted on FIT games, evaluated on INNER pseudo-questions (training data only, via "
            "tools/c10data.load_rows(part='fit'/'inner'); no validation games, no answer files). "
            "Exploration used to freeze M=12, alpha=20, aggregation 'mean', rule B for PREREG_02.",
    "data": {"fit_games": len(fit), "inner_games": len(inner), "inner_players": len(by_p),
             "inner_players_with_pseudo_questions": n_q_players, "rounds": ROUNDS,
             "pseudo_questions_total": len(questions),
             "pseudo_questions_by_level": {lv: q_per_level[lv] for lv in LEVELS}},
    "setup": {
        "moves": "first M moves matched by ;[BW][xy] in the SGF (passes skipped), both colours",
        "canonical_form": "the D4 symmetry (4 rotations x flip) giving the lexicographically smallest "
                          "tuple of the first M points",
        "context": "(target player's colour, canonical prefix); every move of both colours is predicted",
        "smoothing": "ll_k += log((N_k(c,pre,mv) + alpha*pg) / (N_k(c,pre) + alpha)), "
                     "pg = (N(c,pre,mv) + 1) / (N(c,pre) + 361), N_k = fit count at level k, N = sum over levels",
        "stop": "at the first context (c, pre) never seen in fit",
        "question": "sum of per-game ll over the k games, p = softmax(beta * sum)",
        "aggregations": AGGREGATIONS,
        "rules": {"A": "argmax p", "B": "argmax p_k + e^-1 (p_{k-1} + p_{k+1}) (as tools/decide.py)"},
        "pseudo_questions": "for each round r in 0..9: rng = Random(1000 + r); for every inner player "
                            "(sorted id) with >= 5 inner games: k = min(randint(5, 20), n), k games "
                            "sampled without replacement",
        "metric": "official: 1 exact, e^-1 one level off, else 0; rank_balanced_mean = mean over the "
                  "10 levels of the per-level mean (main local metric); question_mean = plain mean",
    },
    "frozen_design": FROZEN,
    "grid_rank_balanced_mean_aggregation": grid,
    "cells": cells,
    "count_stats": count_stats,
    "reproduction_of_scratchpad": reproduction,
    "runtime_s": {"load": round(t_load), "total": round(time.time() - t0)},
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(grid, indent=1))
print("reproduction identical:", all(v["identical"] for v in reproduction.values()))
print("done in %.0fs -> %s" % (time.time() - t0, OUT))
