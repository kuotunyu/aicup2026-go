"""prereg06 explore c1_headroom: shared helpers (competition 1, player identification).

Training data only: fit players through tools/c10data.load_rows(part='fit', local_dir=None).
install_guard() makes every open / listdir / scandir / stat on a path under data/local raise, so
nothing under data/local (validation questions and answers) can be read, even by accident; the
leak check of load_rows (which would open data/local) is therefore switched off (local_dir=None).
fit players are disjoint from the validation players by construction (split v1).
CPU only. tools/score.py is never imported or run. Nothing is written inside the repository except
next to these scripts; the big cache goes to the session scratchpad.
"""
import sys
sys.dont_write_bytecode = True

import builtins
import hashlib
import io
import os
import pickle
import random
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
SCRATCH = os.environ.get(
    "C10_C1H_SCRATCH",
    r"C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go"
    r"\a8d08794-7564-42a1-933e-30b4ef8b21ce\scratchpad\c1_headroom")

# ---------------------------------------------------------------- guard: data/local is never touched
_FORBIDDEN = os.path.normcase(os.path.abspath(os.path.join(REPO, "data", "local")))


def _forbidden(p):
    try:
        if isinstance(p, int):
            return False
        s = os.path.normcase(os.path.abspath(os.fspath(p)))
    except TypeError:
        return False
    # prefix test on the absolute path and on the resolved path (u1_lib's extra substring test
    # 'data\\local' also matched '...\\AppData\\Local\\Temp\\...', i.e. the scratchpad, so it is not used)
    if s == _FORBIDDEN or s.startswith(_FORBIDDEN + os.sep):
        return True
    try:
        r = os.path.normcase(os.path.realpath(s))
    except (OSError, ValueError):
        return False
    return r == _FORBIDDEN or r.startswith(_FORBIDDEN + os.sep)


class LocalDataRefused(RuntimeError):
    pass


def install_guard():
    if getattr(builtins, "_c10_c1h_guard", False):
        return
    real_open, real_listdir, real_scandir, real_stat = builtins.open, os.listdir, os.scandir, os.stat

    def g_open(file, *a, **k):
        if _forbidden(file):
            raise LocalDataRefused(f"c1h guard: refusing to open {file}")
        return real_open(file, *a, **k)

    def g_listdir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"c1h guard: refusing to list {path}")
        return real_listdir(path)

    def g_scandir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"c1h guard: refusing to scan {path}")
        return real_scandir(path)

    def g_stat(path, *a, **k):
        if _forbidden(path):
            raise LocalDataRefused(f"c1h guard: refusing to stat {path}")
        return real_stat(path, *a, **k)

    builtins.open = g_open
    io.open = g_open
    os.listdir = g_listdir
    os.scandir = g_scandir
    os.stat = g_stat
    builtins._c10_c1h_guard = True


install_guard()
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from c10data import LEVELS, load_rows  # noqa: E402
import p02_moves  # noqa: E402
import p02_player_open as po  # noqa: E402  (read-only: token functions, open_cos, true_ranks, points)

CACHE = os.path.join(SCRATCH, "cache_black.pkl")
N_CAND = 100
K_VALUES = (5, 10, 15, 20)
PER_LEVEL = 40
MIN_BLACK = 120          # P120 design of results/test02/design/official_structure


# ---------------------------------------------------------------- cache of fit players' black games
def build_cache():
    """All fit games of players with >= 105 black games, black games only, in file order."""
    t0 = time.time()
    os.makedirs(SCRATCH, exist_ok=True)
    with open(CACHE + ".probe", "wb") as f:      # fail fast if the cache path cannot be written
        f.write(b"ok")
    os.remove(CACHE + ".probe")
    rows = load_rows(os.path.join(REPO, "data", "training"),
                     os.path.join(REPO, "results", "split_v1", "players_split.csv"),
                     part="fit", local_dir=None)
    n_all, n_black, level = Counter(), Counter(), {}
    for r in rows:
        n_all[r["player_id"]] += 1
        n_black[r["player_id"]] += r["color"] == "B"
        level[r["player_id"]] = r["rank"]
    keep = {p for p in n_all if n_black[p] >= 105}
    gid, gpid, gpts, glen = [], [], [], []
    by_player = defaultdict(list)
    bad_alt = 0
    for r in rows:
        p = r["player_id"]
        if p not in keep or r["color"] != "B":
            continue
        mv = p02_moves.parse_moves(r["sgf_content"], limit=10**6)
        cols = "".join(c for c, _ in mv)
        if cols != ("BW" * (len(mv) // 2 + 1))[:len(mv)]:
            bad_alt += 1
        by_player[p].append(len(gid))
        gid.append(r["game_id"])
        gpid.append(p)
        gpts.append("".join(m for _, m in mv))
        glen.append(len(mv))
    n_rows = len(rows)
    del rows
    cache = {"n_rows_fit": n_rows, "n_all": dict(n_all), "n_black": dict(n_black), "level": level,
             "gid": gid, "gpid": gpid, "gpts": gpts, "glen": np.asarray(glen, np.int32),
             "by_player": dict(by_player), "bad_alternation": bad_alt,
             "load_seconds": round(time.time() - t0, 1)}
    os.makedirs(SCRATCH, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


_CACHE = None


def load_cache():
    global _CACHE
    if _CACHE is None:
        with open(CACHE, "rb") as f:
            _CACHE = pickle.load(f)
    return _CACHE


def moves_of(pts, limit=None):
    """points string -> [(colour, point)] (games alternate B/W from black, checked in build_cache)."""
    n = len(pts) // 2 if limit is None else min(limit, len(pts) // 2)
    return [("B" if i % 2 == 0 else "W", pts[2 * i:2 * i + 2]) for i in range(n)]


# ---------------------------------------------------------------- pools (official shape, P120)
def k_list(n):
    per = n // len(K_VALUES)
    return [k for k in K_VALUES for _ in range(per)]


def choose_pool(cache, seed, min_black=MIN_BLACK, per_level=PER_LEVEL):
    """Exactly choose_pool of official_c1_blackonly_sim.py (P120)."""
    elig = defaultdict(list)
    for p in sorted(cache["n_all"]):
        if cache["n_black"][p] >= min_black:
            elig[cache["level"][p]].append(p)
    pool = []
    for lv in LEVELS:
        pl = sorted(elig[lv], key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        if len(pl) < per_level:
            raise ValueError("level %s has only %d eligible players" % (lv, len(pl)))
        pool += pl[:per_level]
    return pool


def make_pool(cache, seed, scenario="o2o", n_cand=N_CAND, split="random"):
    """Candidates and questions of one simulated pool.

    o2o: the official_c1_blackonly_sim P120 'black_random' pool (seed s gives exactly its pool):
         candidate i = first 100 of player i's shuffled black games, question i = next k_i games.
    Other scenarios keep the same 400 candidates and candidate games, and change who owns the
    400 questions (k: 100 of each of 5/10/15/20 in all scenarios):
         'wr'     owners drawn with replacement from the 400 candidates;
         'u<m>'   m distinct candidates own questions (each >= 1), the other 400 - m questions are
                  drawn with replacement from those m.
         'd<m>'   (added in the continuation run) the sibling prior_art 'dup80' structure for m = 80:
                  m candidates own no question, m other candidates own two, the rest own one.
    split='random' (default, as before): the player's black games shuffled; split='ord': the
    player's black games in file order (official_c1_blackonly_sim 'black_fileorder'); the candidate
    set, k list and owners do not depend on split.
    A player's question games are consecutive chunks of their shuffled black games after the
    candidate games. Infeasible k (not enough games) is repaired by swapping k between questions.
    Returns dict(pool, cand_rows [400 lists of cache row ids], q_rows [400 lists], owner [400],
    k [400], n_repairs)."""
    pool = choose_pool(cache, seed)
    n = len(pool)
    ks = k_list(n)
    random.Random("k|%d" % seed).shuffle(ks)
    rng = random.Random("pool|%d" % seed)
    queue = {}
    cand_rows = []
    for p in pool:
        g = list(cache["by_player"][p])
        rng.shuffle(g)
        if split == "ord":
            g = list(cache["by_player"][p])
        elif split != "random":
            raise ValueError(split)
        cand_rows.append(g[:n_cand])
        queue[p] = g[n_cand:]
    if scenario == "o2o":
        owner = list(range(n))
    else:
        r2 = random.Random("scen|%s|%d" % (scenario, seed))
        if scenario == "wr":
            owner = [r2.randrange(n) for _ in range(n)]
        elif scenario.startswith("d"):
            m = int(scenario[1:])
            perm = list(range(n))
            r2.shuffle(perm)
            owner = perm[:n - m] + perm[n - 2 * m:n - m]     # perm[n-m:] own none, perm[n-2m:n-m] own two
            r2.shuffle(owner)
        elif scenario.startswith("u"):
            m = int(scenario[1:])
            used = r2.sample(range(n), m)
            owner = used + [r2.choice(used) for _ in range(n - m)]
            r2.shuffle(owner)
        else:
            raise ValueError(scenario)
    ks = list(ks)
    # feasibility repair: each owner needs sum of its k <= len(queue)
    cap = [len(queue[pool[i]]) for i in range(n)]
    n_rep = 0
    r3 = random.Random("repair|%s|%d" % (scenario, seed))
    for _ in range(10000):
        need = Counter()
        for q in range(n):
            need[owner[q]] += ks[q]
        bad = [i for i in need if need[i] > cap[i]]
        if not bad:
            break
        i = bad[0]
        qs = sorted([q for q in range(n) if owner[q] == i], key=lambda q: -ks[q])
        q = qs[0]
        cands = [q2 for q2 in range(n) if owner[q2] != i and ks[q2] < ks[q]
                 and need[owner[q2]] - ks[q2] + ks[q] <= cap[owner[q2]]]
        if cands:
            q2 = r3.choice(cands)
            ks[q], ks[q2] = ks[q2], ks[q]
        else:
            others = [j for j in set(owner) if j != i and need[j] + ks[q] <= cap[j]]
            owner[q] = r3.choice(others)
        n_rep += 1
    else:
        raise RuntimeError("could not repair k assignment")
    q_rows = []
    pos = Counter()
    for q in range(n):
        p = pool[owner[q]]
        q_rows.append(queue[p][pos[owner[q]]:pos[owner[q]] + ks[q]])
        pos[owner[q]] += ks[q]
    return {"pool": pool, "cand_rows": cand_rows, "q_rows": q_rows, "owner": np.asarray(owner),
            "k": np.asarray(ks), "n_repairs": n_rep, "scenario": scenario, "seed": seed}


# ---------------------------------------------------------------- per-game token matrices
def token_matrix(tok_fn, rows=None, cache=None):
    """CSR count matrix (len(rows) x V) of tok_fn(points string) over cache games (all if rows None).
    Returns (matrix, vocab dict)."""
    cache = cache or load_cache()
    rows = range(len(cache["gpts"])) if rows is None else rows
    vocab, r, c, v = {}, [], [], []
    for i, gi in enumerate(rows):
        for t, n in Counter(tok_fn(cache["gpts"][gi])).items():
            r.append(i)
            c.append(vocab.setdefault(t, len(vocab)))
            v.append(n)
    M = sp.csr_matrix((np.asarray(v, np.float64), (np.asarray(r, np.int64), np.asarray(c, np.int64))),
                      shape=(len(rows), len(vocab)))
    return M, vocab


def family_path(name):
    return os.path.join(SCRATCH, "fam_%s.npz" % name)


def load_family(name):
    return sp.load_npz(family_path(name)).tocsr()


# ---------------------------------------------------------------- scoring
def cos_pool(G, pool_rec):
    """cosO (questions x candidates) with the exact p02 weighting (po.open_cos)."""
    cos, has = po.open_cos(G, pool_rec["cand_rows"], G, pool_rec["q_rows"])
    return cos, has


def ranks(S, owner):
    return po.true_ranks(S, owner)


def pts(r):
    return po.points(r)


def summary(r, ks):
    r, ks = np.asarray(r), np.asarray(ks)
    out = {"score": float(pts(r).mean()), "top1": float((r == 1).mean()), "top5": float((r <= 5).mean())}
    out["per_k"] = {str(k): {"score": float(pts(r[ks == k]).mean()), "top1": float((r[ks == k] == 1).mean()),
                             "top5": float((r[ks == k] <= 5).mean())} for k in K_VALUES}
    return out


def zrows(M):
    M = np.asarray(M, np.float64)
    sd = M.std(1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (M - M.mean(1, keepdims=True)) / sd


def mean_ci(x, alpha=0.05):
    """mean and t-interval over pools."""
    from scipy import stats
    x = np.asarray(x, np.float64)
    m = float(x.mean())
    if len(x) < 2:
        return m, float("nan"), float("nan")
    h = float(stats.t.ppf(1 - alpha / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - h, m + h
