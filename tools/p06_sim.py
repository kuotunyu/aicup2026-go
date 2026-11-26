"""PREREG_06 (competition 1): fit-player simulation pools, pseudo (sr06) pools, per-pool LDA refits and the
per-pool score terms used by the design steps of tools/p06_player.py (tune, switch, dmix, selfcheck).

Only fit players (the feature store written by 'p06_player.py build' from c10data.load_rows(part="fit",
local_dir=None)). make_pool / choose_pool are verbatim ports of results/prereg06/explore/c1_headroom/c1h_lib.py
(official shape: 40 players per level among the fit players with >= 120 black games, 100 candidate black games,
k = 100 x each of 5 / 10 / 15 / 20; structures o2o, u<m>, wr, d<m>); positions in each player's list of black
games are what is drawn, so the pools are identical whatever the row numbering. dmix_pool builds the
black-and-white pools of PREREG_06 2.8 (players with >= 105 games, k = min(randint(5, 20), remaining games) as
tools/make_val.py).
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p02_player_fuse as pf  # noqa: E402
import p02_player_open as po  # noqa: E402
import p06_style as st  # noqa: E402
from c10data import LEVELS  # noqa: E402

REPO = st.REPO
N_CAND = 100
K_VALUES = (5, 10, 15, 20)
PER_LEVEL = 40
MIN_BLACK = 120
MIN_LDA = 105
DMIX_MIN = 105
STRUCTURES = ("o2o", "u390", "u375", "u350", "d80", "u300", "wr", "d1", "d2", "d3", "d5")
NON_O2O = ("u390", "u375", "u350", "d80", "u300", "wr")
NEAR_O2O = ("d1", "d2", "d3", "d5")
V2_W = (2.66, 0.5, 0.5)
V3_W = (1.33, 0.25, 0.25)
CSLS_KS = (10, 20, 40)


# ================================================================ the fit feature store
class Store:
    """Per-game features of the fit players with >= 105 games (both colours), file order (build output)."""

    def __init__(self, feat_dir, load=("oraw", "jo", "s2", "s34")):
        self.dir = feat_dir
        with np.load(os.path.join(feat_dir, "meta.npz"), allow_pickle=False) as z:
            self.rank = z["rank"]
            self.gid = z["game_id"]
            self.pid = z["player_id"]
            self.color = z["color"]
            self.alt = z["alternating"]
            self.nmov = z["n_moves"]
        self.n = len(self.gid)
        self.mats = {}
        for name in load:
            if name in ("s2", "s34"):
                self.mats[name] = np.load(os.path.join(feat_dir, name + ".npy"), mmap_mode="r")
            else:
                self.mats[name] = load_csr(os.path.join(feat_dir, name))
        self.vocab = {}
        self._by_player = None
        self._cache = None

    def csr(self, name):
        if name not in self.mats:
            self.mats[name] = load_csr(os.path.join(self.dir, name))
        return self.mats[name]

    def dense(self, name):
        if name not in self.mats:
            self.mats[name] = np.load(os.path.join(self.dir, name + ".npy"), mmap_mode="r")
        return self.mats[name]

    def by_player(self):
        """{pid: [rows]} all games, file order."""
        if self._by_player is None:
            d = defaultdict(list)
            for i, p in enumerate(self.pid.tolist()):
                d[p].append(i)
            self._by_player = dict(d)
        return self._by_player

    def black_cache(self):
        """The c1h_lib cache view: n_all / n_black / level for every store player, by_player = black-game rows of
        players with >= 105 black games (file order)."""
        if self._cache is None:
            bp = self.by_player()
            n_all, n_black, level, byb = {}, {}, {}, {}
            col = self.color
            rk = self.rank
            for p, rows in bp.items():
                b = [r for r in rows if col[r] == "B"]
                n_all[p] = len(rows)
                n_black[p] = len(b)
                level[p] = str(rk[rows[0]])
                if len(b) >= MIN_LDA:
                    byb[p] = b
            self._cache = {"n_all": n_all, "n_black": n_black, "level": level, "by_player": byb}
        return self._cache


def save_csr(prefix, M, vocab):
    M = M.tocsr()
    os.makedirs(prefix, exist_ok=True)
    np.save(os.path.join(prefix, "data.npy"), M.data.astype(np.float64))
    np.save(os.path.join(prefix, "indices.npy"), M.indices.astype(np.int32))
    np.save(os.path.join(prefix, "indptr.npy"), M.indptr.astype(np.int64))
    with open(os.path.join(prefix, "vocab.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"shape": list(M.shape), "vocab": list(vocab)}, f, ensure_ascii=False)


def load_csr(prefix):
    with open(os.path.join(prefix, "vocab.json"), encoding="utf-8") as f:
        meta = json.load(f)
    data = np.load(os.path.join(prefix, "data.npy"), mmap_mode="r")
    ind = np.load(os.path.join(prefix, "indices.npy"), mmap_mode="r")
    ptr = np.load(os.path.join(prefix, "indptr.npy"), mmap_mode="r")
    M = sp.csr_matrix((data, ind, ptr), shape=tuple(meta["shape"]), copy=False)
    M.vocab = meta["vocab"]
    return M


# ================================================================ pools (verbatim ports of c1h_lib)
def k_list(n):
    per = n // len(K_VALUES)
    return [k for k in K_VALUES for _ in range(per)]


def choose_pool(cache, seed, min_black=MIN_BLACK, per_level=PER_LEVEL):
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
    """c1h_lib.make_pool (see its docstring); rows are whatever cache['by_player'] holds."""
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
            owner = perm[:n - m] + perm[n - 2 * m:n - m]
            r2.shuffle(owner)
        elif scenario.startswith("u"):
            m = int(scenario[1:])
            used = r2.sample(range(n), m)
            owner = used + [r2.choice(used) for _ in range(n - m)]
            r2.shuffle(owner)
        else:
            raise ValueError(scenario)
    ks = list(ks)
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


def dmix_pool(store, seed, per_level=PER_LEVEL, min_games=DMIX_MIN):
    """PREREG_06 2.8 D-mix pool: fit players with >= 105 games (both colours), 40 per level chosen by
    sha256('{seed}|{pid}'); rng = random.Random('pool|{seed}'); per player: games (file order) shuffled, first
    100 = candidate games, next k = min(rng.randint(5, 20), remaining) = the question (one question per player)."""
    bp = store.by_player()
    cache = store.black_cache()
    elig = defaultdict(list)
    for p in sorted(bp):
        if len(bp[p]) >= min_games:
            elig[cache["level"][p]].append(p)
    rng = random.Random("pool|%d" % seed)
    pool = []
    for lv in LEVELS:
        pl = sorted(elig[lv], key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        if len(pl) < per_level:
            raise ValueError("level %s has only %d eligible players" % (lv, len(pl)))
        pool += pl[:per_level]
    cand_rows, q_rows, ks = [], [], []
    for p in pool:
        g = list(bp[p])
        rng.shuffle(g)
        k = min(rng.randint(5, 20), len(g) - N_CAND)
        cand_rows.append(g[:N_CAND])
        q_rows.append(g[N_CAND:N_CAND + k])
        ks.append(k)
    return {"pool": pool, "cand_rows": cand_rows, "q_rows": q_rows, "owner": np.arange(len(pool)),
            "k": np.asarray(ks), "scenario": "dmix", "seed": seed}


def sr_pool(store, rec, rounds=(0, 1)):
    """sr06 pseudo pools (PREREG_06 2.3) of a simulated pool: candidates' games only; one pseudo question per
    candidate (one-to-one). Candidates are the pool's 400 in sorted player-id order; returns a list of
    {cand_rows, q_rows, owner, cand_ids} (cand_rows = the 80 reference games), in pool order of the candidates
    mapped to sorted ids (owner = position in sorted order)."""
    pids = [store.pid[r[0]] for r in rec["cand_rows"]]
    order = sorted(range(len(pids)), key=lambda i: pids[i])
    cg = {pids[i]: [str(store.gid[r]) for r in rec["cand_rows"][i]] for i in range(len(pids))}
    row_of = {(pids[i], str(store.gid[r])): r for i in range(len(pids)) for r in rec["cand_rows"][i]}
    rr = st.sr06_rounds(cg, rec["k"], n_rounds=max(rounds) + 1)
    out = []
    for r in rounds:
        items = rr[r]
        out.append({"cand_rows": [[row_of[(p, g)] for g in refs] for p, _q, refs in items],
                    "q_rows": [[row_of[(p, g)] for g in q] for p, q, _refs in items],
                    "owner": np.arange(len(items)), "cand_ids": [p for p, _q, _r in items], "round": r})
    return out, [pids[i] for i in order]


# ================================================================ LDA in the simulations
class LdaData:
    """ldain rows of one colour's LDA training set (written by 'p06_player.py lda'); ldain of any other store row
    (for example a D-mix player with < 105 games of that colour) is computed with the same function."""

    def __init__(self, feat_dir, colour, store=None):
        self.X = np.load(os.path.join(feat_dir, "ldain_%s.npy" % colour), mmap_mode="r")
        self.rows = np.load(os.path.join(feat_dir, "ldain_%s_rows.npy" % colour))
        self.cols = np.load(os.path.join(feat_dir, "ldain_%s_cols.npy" % colour))
        self.pos = {int(r): i for i, r in enumerate(self.rows.tolist())}
        self.store = store

    def fit_excluding(self, store, players):
        """LDA on the colour's training games of every training player outside `players` (PREREG_06 1.6: one refit
        per simulated pool)."""
        pid = store.pid[self.rows]
        keep = ~np.isin(pid, np.asarray(sorted(players)))
        m = st.lda_fit(np.asarray(self.X[np.flatnonzero(keep)], np.float64), pid[keep])
        m["n_players"] = int(len(set(pid[keep].tolist())))
        m["n_games"] = int(keep.sum())
        return m

    def ldain(self, rows):
        rows = [int(r) for r in rows]
        out = np.zeros((len(rows), self.X.shape[1]), np.float32)
        have = [i for i, r in enumerate(rows) if r in self.pos]
        if have:
            out[have] = self.X[np.asarray([self.pos[rows[i]] for i in have], np.int64)]
        miss = [i for i, r in enumerate(rows) if r not in self.pos]
        if miss:
            mr = np.asarray([rows[i] for i in miss], np.int64)
            out[miss] = st.ldain_matrix(np.asarray(self.store.dense("s2")[mr]), self.store.csr("resp")[mr], self.cols)
        return out

    def project(self, rows, model):
        return st.lda_project(np.asarray(self.ldain(rows), np.float64), model)


# ================================================================ score terms of one pool
def _split_usable(store, groups, colour):
    """Per group: rows of the colour that are usable (alternating, >= 1 move)."""
    col, alt = store.color, store.alt
    return [[r for r in g if col[r] == colour and alt[r]] for g in groups]


def term_O(store, rec):
    cos, has = po.open_cos(store.csr("oraw"), rec["cand_rows"], store.csr("oraw"), rec["q_rows"])
    return cos, has


def term_JO(store, rec, colours=("B", "W")):
    cg = [[r for r in g if store.alt[r] and store.color[r] in colours] for g in rec["cand_rows"]]
    qg = [[r for r in g if store.alt[r] and store.color[r] in colours] for g in rec["q_rows"]]
    G = store.csr("jo")
    cos, has = po.open_cos(G, cg, G, qg)
    has = has & np.array([len(g) > 0 for g in qg])
    return cos, has


def term_nb(store, rec, name, colours=("B", "W")):
    X = store.dense(name)
    per, qc = {}, {}
    for c in colours:
        cg = _split_usable(store, rec["cand_rows"], c)
        qg = _split_usable(store, rec["q_rows"], c)
        if not any(len(g) for g in cg) and not any(len(g) for g in qg):
            continue
        rows = sorted({r for g in cg + qg for r in g})
        loc = {r: i for i, r in enumerate(rows)}
        Xc = np.asarray(X[np.asarray(rows, np.int64)], np.float64) if rows else np.zeros((0, X.shape[1]))
        T, ch = st.nb_term(Xc, [[loc[r] for r in g] for g in cg], [[loc[r] for r in g] for g in qg])
        per[c] = (T, ch)
        qc[c] = [len(g) for g in qg]
    if not per:
        n = len(rec["q_rows"])
        return np.zeros((n, len(rec["cand_rows"]))), np.zeros(n, bool)
    return st.colour_terms(per, qc)


def term_nb_raw(store, rec, name, colour):
    """T_S2 (or the proxy-A NB) of one colour before the colour merge (selfcheck compares it value by value)."""
    X = store.dense(name)
    cg = _split_usable(store, rec["cand_rows"], colour)
    qg = _split_usable(store, rec["q_rows"], colour)
    rows = sorted({r for g in cg + qg for r in g})
    loc = {r: i for i, r in enumerate(rows)}
    Xc = np.asarray(X[np.asarray(rows, np.int64)], np.float64)
    return st.nb_term(Xc, [[loc[r] for r in g] for g in cg], [[loc[r] for r in g] for g in qg])[0]


def term_L(store, rec, lda, models, colours=("B", "W")):
    """lda: {colour: LdaData or callable rows -> ldain}, models: {colour: model}."""
    per, qc = {}, {}
    for c in colours:
        if c not in models:
            continue
        cg = _split_usable(store, rec["cand_rows"], c)
        qg = _split_usable(store, rec["q_rows"], c)
        if not any(len(g) for g in cg) and not any(len(g) for g in qg):
            continue
        rows = sorted({r for g in cg + qg for r in g})
        loc = {r: i for i, r in enumerate(rows)}
        Z = lda[c].project(rows, models[c]) if rows else np.zeros((0, st.LDA_DIM))
        Zc, nc = st.group_means(Z, [[loc[r] for r in g] for g in cg])
        Zq, _nq = st.group_means(Z, [[loc[r] for r in g] for g in qg])
        per[c] = (st.lda_term(Zc, Zq), nc > 0)
        qc[c] = [len(g) for g in qg]
    if not per:
        n = len(rec["q_rows"])
        return np.zeros((n, len(rec["cand_rows"]))), np.zeros(n, bool)
    return st.colour_terms(per, qc)


class Cnn:
    """Fit-player CNN sums (cnn_fit black_ge105.npz) addressed by store row."""

    def __init__(self, path, store):
        with np.load(path, allow_pickle=False) as z:
            emb = z["emb_sum"]
            nw = z["n_windows"]
            gid = z["game_id"]
            pid = z["player_id"]
            rk = z["rank"]
        key = {(str(a), str(b), str(c)): i for i, (a, b, c) in enumerate(zip(rk.tolist(), gid.tolist(), pid.tolist()))}
        pos = np.full(store.n, -1, np.int64)
        for r in range(store.n):
            i = key.get((str(store.rank[r]), str(store.gid[r]), str(store.pid[r])))
            if i is not None and store.color[r] == "B":
                pos[r] = i
        self.pos = pos
        self.emb = emb
        self.nw = nw

    def arrays(self, rows):
        idx = self.pos[np.asarray(rows, np.int64)]
        if (idx < 0).any():
            raise KeyError("CNN embedding missing for %d rows" % int((idx < 0).sum()))
        return pf._cnn_arrays(self.emb[idx], self.nw[idx], np.ones(len(idx), bool))


def term_E0(cnn, rec):
    rows = sorted({r for g in rec["cand_rows"] + rec["q_rows"] for r in g})
    loc = {r: i for i, r in enumerate(rows)}
    arr = cnn.arrays(rows)
    cg = [[loc[r] for r in g] for g in rec["cand_rows"]]
    qg = [[loc[r] for r in g] for g in rec["q_rows"]]
    M, has, _ = pf.e_scores("E0", arr, cg, arr, qg)
    return M, has


def pool_terms(store, rec, lda, models, cnn=None, want=("O", "JO", "S2", "L", "PA", "E0"), colours=("B", "W")):
    T = {}
    if "O" in want:
        T["O"] = term_O(store, rec)
    if "JO" in want:
        T["JO"] = term_JO(store, rec, colours)
    if "S2" in want:
        T["S2"] = term_nb(store, rec, "s2", colours)
    if "PA" in want:
        T["PA"] = term_nb(store, rec, "s34", colours)
    if "L" in want:
        T["L"] = term_L(store, rec, lda, models, colours)
    if "E0" in want and cnn is not None:
        T["E0"] = term_E0(cnn, rec)
    return T


# ================================================================ combinations
def combo_F(T, e_src, w, open_only=False):
    """F = z(E) + 1 z(O) + wL z(L) + wJ z(JO) + wS z(S2); e_src 'E0' (real CNN) or 'PA' (proxy A); w = None for
    V0 (no v2). open_only: no E term (the 4th usability condition of PREREG_06 2.6)."""
    terms, ws = [], []
    if not open_only:
        terms.append(T[e_src])
        ws.append(1.0)
    terms.append(T["O"])
    ws.append(1.0)
    if w is not None:
        for name, x in zip(("L", "JO", "S2"), w):
            terms.append(T[name])
            ws.append(x)
    return st.fuse(terms, ws)


def combo_list(v1):
    """[(name, weights)] in the registered order V2, V1 (if any), V3, V0."""
    out = [("V2", V2_W)]
    if v1 is not None:
        out.append(("V1", tuple(v1)))
    out += [("V3", V3_W), ("V0", None)]
    return out


def pool_scores(F, owner, ks_off=(None,) + CSLS_KS, with_switch=True):
    """Scores of the off rules (CSLS k, None = no correction) and the switch (Hungarian + CSLS3), detector stats."""
    out = {}
    for k in ks_off:
        out["off_%s" % ("none" if k is None else k)] = st.score(st.csls(F, k), owner)
    if with_switch:
        Z = st.zrows(F)
        a = st.hungarian_top1(Z)
        top = Z.argmax(1)
        n = Z.shape[0]
        out["on"] = st.score(st.with_top1(st.csls(F, st.CSLS_SWITCH_K), a), owner)
        out["forced_cost"] = float(np.mean(Z[np.arange(n), top] - Z[np.arange(n), a]))
        out["changed"] = float(np.mean(a != top))
    return out


def forced_cost(F):
    Z = st.zrows(F)
    a = st.hungarian_top1(Z)
    top = Z.argmax(1)
    n = Z.shape[0]
    return float(np.mean(Z[np.arange(n), top] - Z[np.arange(n), a]))
