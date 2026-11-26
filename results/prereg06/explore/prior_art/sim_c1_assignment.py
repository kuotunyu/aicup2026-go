"""Competition 1: what do pool-level (transductive) tricks add to opening retrieval, and what do
they cost when the one-question-per-candidate assumption is wrong?

Pools follow the official structure (AIdea data page, read 2026-09-28): 400 candidates x 100 BLACK
games, 400 questions with k = 5/10/15/20 (100 each), black only. Built exactly like design P120 /
split black_random of results/test02/design/official_structure/official_c1_blackonly_sim.py
(fit players only, 40 per level, >= 120 black games), seeds 0-4. Scores use only the
opening-retrieval part of Q04 (raw orientation, N = 10, token set 'both'); the CNN part of Q04 is
not included, so absolute numbers are lower than Q04's.

Scenarios
  one2one : every candidate is the answer of exactly one question (the sim's default)
  dup80   : 80 candidates get no question, 80 other candidates get two questions (disjoint
            query games), 240 get one; 400 questions in total, k still 100 each
Decisions
  base    : rank candidates by cosine (what Q04's opening part does)
  hung    : Hungarian (linear_sum_assignment) on the 400x400 cosine matrix fixes Top-1;
            Top-2..5 = the next best by cosine
  csls10  : CSLS hubness correction, S' = 2S - r_cand (r = mean of the 10 best query cosines)
  dsm_t   : dual softmax  softmax_row(S/t) * softmax_col(S/t), t in {0.02, 0.05}
Metric: official, e^-(r-1) for r <= 5.

Data discipline: u1_lib guard first (nothing under data/local can be opened), training rows via
tools/c10data.load_rows(part='fit') WITHOUT local_dir; no validation files, no score.py; CPU only.
Runtime: about 5 minutes to load, then ~10-20 s per pool.

Usage (Git Bash, repo root):
  .venv/Scripts/python.exe results/prereg06/explore/prior_art/sim_c1_assignment.py
Writes sim_c1_assignment.json next to this script.
"""
import sys

sys.dont_write_bytecode = True
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "results", "prereg05", "explore", "u1"))
import u1_lib  # noqa: E402,F401  (data/local guard, installed on import)
sys.path.insert(0, os.path.join(REPO, "tools"))
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from c10data import LEVELS, load_rows  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
N_CAND, PER_LEVEL, K_VALUES, N_TOK = 100, 40, (5, 10, 15, 20), 10
SGF_CHARS, MAX_MOVES = 2000, 60
MV = re.compile(r";([BW])\[([a-s]{2})\]")
# reference: official_c1_blackonly_sim.json, P120 black_random raw N10 both, seeds 0-4
REF = [0.9184, 0.9213, 0.9263, 0.9232, 0.9319]

t0 = time.time()
rows = load_rows(os.path.join(REPO, "data", "training"),
                 os.path.join(REPO, "results", "split_v1", "players_split.csv"), part="fit")
n_black, level = Counter(), {}
for r in rows:
    n_black[r["player_id"]] += r["color"] == "B"
    level[r["player_id"]] = r["rank"]
games = defaultdict(list)   # pid -> [(colour, moves)] black games, file order
for r in rows:
    p = r["player_id"]
    if r["color"] == "B" and n_black[p] >= 120:
        games[p].append(("B", MV.findall(r["sgf_content"][:SGF_CHARS])[:MAX_MOVES]))
del rows
t_load = time.time() - t0
print("loaded in %.0fs" % t_load, flush=True)


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


def ranks_from_order(order, true):
    """order: [nq, >=5] candidate indices best first; rank of the true candidate (6 if absent)."""
    out = np.full(len(true), 6)
    for i, t in enumerate(true):
        hit = np.flatnonzero(order[i, :5] == t)
        if len(hit):
            out[i] = hit[0] + 1
    return out


def stable_order(S):
    return np.argsort(-S, axis=1, kind="stable")


def pts(r):
    r = np.asarray(r, float)
    return np.where(r <= 5, np.exp(-(r - 1.0)), 0.0)


def summ(r, ks):
    r, ks = np.asarray(r), np.asarray(ks)
    return {"score": round(float(pts(r).mean()), 4), "top1": round(float((r == 1).mean()), 4),
            "top5": round(float((r <= 5).mean()), 4),
            "per_k": {str(k): round(float(pts(r[ks == k]).mean()), 4) for k in K_VALUES}}


def decisions(S):
    out = {"base": stable_order(S)}
    a_r, a_c = linear_sum_assignment(-S)
    top1 = np.empty(S.shape[0], int)
    top1[a_r] = a_c
    o = stable_order(S)
    hung = np.empty((S.shape[0], 5), int)
    for i in range(S.shape[0]):
        rest = [c for c in o[i, :6] if c != top1[i]][:4]
        hung[i] = [top1[i]] + rest
    out["hung"] = hung
    rc = np.sort(S, axis=0)[-10:, :].mean(0)          # per candidate: mean of its 10 best queries
    out["csls10"] = stable_order(2 * S - rc[None, :])
    for t in (0.02, 0.05):
        Z = S / t
        A = np.exp(Z - Z.max(1, keepdims=True))
        A /= A.sum(1, keepdims=True)
        B = np.exp(Z - Z.max(0, keepdims=True))
        B /= B.sum(0, keepdims=True)
        out["dsm_%g" % t] = stable_order(A * B)
    return out


def build_pool(seed):
    elig = defaultdict(list)
    for p in sorted(n_black):
        if n_black[p] >= 120:
            elig[level[p]].append(p)
    pool = []
    for lv in LEVELS:
        pl = sorted(elig[lv], key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        pool += pl[:PER_LEVEL]
    ks = [k for k in K_VALUES for _ in range(len(pool) // len(K_VALUES))]
    random.Random("k|%d" % seed).shuffle(ks)
    kmap = dict(zip(pool, ks))
    rng = random.Random("pool|%d" % seed)
    shuffled = {}
    for p in pool:
        g = list(games[p])
        rng.shuffle(g)
        shuffled[p] = g
    return pool, kmap, shuffled


def run(seed):
    ts = time.time()
    pool, kmap, g = build_pool(seed)
    cand = [g[p][:N_CAND] for p in pool]
    rec = {"seed": seed}
    # ---- one2one: query i = next k games of candidate i
    q1 = [g[p][N_CAND:N_CAND + kmap[p]] for p in pool]
    true1 = np.arange(len(pool))
    ks1 = [kmap[p] for p in pool]
    # ---- dup80: 80 candidates without a question, 80 with two (second query = following games)
    rng = random.Random("dup|%d" % seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    absent = set(order[:80])
    big = [i for i in order[80:] if len(g[pool[i]]) >= N_CAND + 40]
    doubles = set(big[:80])
    if len(doubles) < 80:
        raise ValueError("not enough players with >= 140 black games for dup80")
    ks2 = [k for k in K_VALUES for _ in range(100)]
    random.Random("k2|%d" % seed).shuffle(ks2)
    q2, true2, j = [], [], 0
    for i in range(len(pool)):
        if i in absent:
            continue
        reps = 2 if i in doubles else 1
        start = N_CAND
        for _ in range(reps):
            k = ks2[j]
            j += 1
            q2.append(g[pool[i]][start:start + k])
            true2.append(i)
            start += k
    assert j == 400 and len(q2) == 400
    for name, query, true, ks in (("one2one", q1, true1, ks1), ("dup80", q2, np.array(true2), ks2)):
        flat = [x for lst in cand for x in lst] + [x for lst in query for x in lst]
        cg, qg, pos = [], [], 0
        for lst in cand:
            cg.append(list(range(pos, pos + len(lst))))
            pos += len(lst)
        for lst in query:
            qg.append(list(range(pos, pos + len(lst))))
            pos += len(lst)
        Gc = count_matrix([col_tokens(c, mv, N_TOK) for c, mv in flat])
        Gn = count_matrix([nocol_tokens(c, mv, N_TOK) for c, mv in flat])
        S = cos_matrix(sp.hstack([Gc, Gn]).tocsr(), cg, qg)
        rec[name] = {d: summ(ranks_from_order(o, true), ks) for d, o in decisions(S).items()}
    rec["reference_base_one2one"] = REF[seed]
    rec["reproduced"] = bool(abs(rec["one2one"]["base"]["score"] - REF[seed]) < 5e-5)
    rec["seconds"] = round(time.time() - ts, 1)
    print(seed, json.dumps({s: {d: v["score"] for d, v in rec[s].items()} for s in ("one2one", "dup80")}),
          "repro", rec["reproduced"], "%.0fs" % rec["seconds"], flush=True)
    return rec


recs = [run(s) for s in SEEDS]
agg = {}
for scen in ("one2one", "dup80"):
    agg[scen] = {}
    for d in recs[0][scen]:
        sc = np.array([r[scen][d]["score"] for r in recs])
        base = np.array([r[scen]["base"]["score"] for r in recs])
        agg[scen][d] = {"score_mean": round(float(sc.mean()), 4),
                        "top1_mean": round(float(np.mean([r[scen][d]["top1"] for r in recs])), 4),
                        "diff_vs_base_mean": round(float((sc - base).mean()), 4),
                        "diff_vs_base_min": round(float((sc - base).min()), 4),
                        "diff_vs_base_max": round(float((sc - base).max()), 4)}
out = {"what": __doc__.strip().splitlines()[0], "seeds": SEEDS, "load_seconds": round(t_load),
       "aggregate": agg, "records": recs, "total_seconds": round(time.time() - t0)}
with open(os.path.join(HERE, "sim_c1_assignment.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(agg, indent=1))
print("done in %.0fs" % (time.time() - t0))
