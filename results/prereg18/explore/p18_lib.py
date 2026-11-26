"""prereg18 exploration (competition 2): same-player linking + pooling layer, pure numpy / scipy helpers of PLAN.md.

Nothing here reads files; p18_eval.py feeds the arrays. Every random draw takes an explicit numpy Generator.
"""

import hashlib

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import csgraph

L = 10
E1 = float(np.exp(-1.0))
TOL = 1e-12

# ---------------------------------------------------------------- registered constants (PLAN.md)
SPLIT_SALT = "c10-prereg18|split|"
T_MIN_GAMES = 10
LDA_DIM = 128
EPS_REL = 1e-3
B_FLOOR = 1e-6
POOL_Q = 400
KS = (5, 10, 15, 20)
PER_K = 100
COLOURS = ("B", "W")
COLOUR_RETRIES = 10
MAX_POOL_ATTEMPTS = 50
PRECISION_TARGET = 0.99
NULL_ON_TARGET = 0.005
MAX_COMPONENT = 4
SEED_CAL, SEED_EVAL, SEED_SMOKE = 1018, 18, 9018
# id, name, p_multi, multi sizes, calibration pools, evaluation pools, role
SCENARIOS = (
    (0, "S0", 0.0, (), 300, 500, "G1"),
    (1, "S30", 0.30, (2, 3), 200, 200, "G2"),
    (2, "S10", 0.10, (2, 3), 0, 200, "record"),
    (3, "S30x", 0.30, (2, 3, 4), 0, 200, "record"),
    (4, "S100", 1.00, (2, 3, 4), 0, 100, "record"),
)
SMOKE_POOLS = {"cal": 5, "eval": 3}
G1_MIN_DIFF = -0.001
G1_MAX_FALSE_ON = 0.01
G2_MIN_DIFF = 0.010


def scenario(sid):
    for s in SCENARIOS:
        if s[0] == sid:
            return {"id": s[0], "name": s[1], "p_multi": s[2], "sizes": s[3], "cal": s[4], "eval": s[5],
                    "role": s[6]}
    raise KeyError(sid)


# ---------------------------------------------------------------- player split (PLAN.md 1)
def split_bit(player_key):
    return int(hashlib.sha256((SPLIT_SALT + player_key).encode("utf-8")).hexdigest()[:8], 16) % 2


# ---------------------------------------------------------------- LDA + PLDA (PLAN.md 2)
class Accum:
    """Streaming sums for S_w / S_b over games of players (float64)."""

    def __init__(self, n_players, d):
        self.sum = np.zeros((n_players, d))
        self.cnt = np.zeros(n_players)
        self.zz = np.zeros((d, d))

    def add(self, Z, pidx):
        Z = np.asarray(Z, dtype=np.float64)
        ind = sparse.csr_matrix((np.ones(len(pidx)), (pidx, np.arange(len(pidx)))), shape=(len(self.cnt), len(pidx)))
        self.sum += np.asarray(ind @ Z)
        self.cnt += np.bincount(pidx, minlength=len(self.cnt))
        self.zz += Z.T @ Z

    def scatter(self):
        ok = self.cnt > 0
        mean = self.sum[ok] / self.cnt[ok, None]
        N = self.cnt[ok].sum()
        Sw = (self.zz - (self.sum[ok].T / self.cnt[ok]) @ self.sum[ok]) / N
        Sw = 0.5 * (Sw + Sw.T)
        c = mean.mean(0)
        D = mean - c
        Sb = D.T @ D / len(mean)
        return Sw, Sb, c, mean, self.cnt[ok]


def lda_fit(Sw, Sb, dim=LDA_DIM, eps_rel=EPS_REL):
    """Top-dim generalised eigenvectors of (Sb, Sw + eps I), v' (Sw + eps I) v = 1, sign: largest |component| > 0."""
    d = Sw.shape[0]
    eps = eps_rel * float(np.mean(np.diag(Sw)))
    w, V = linalg.eigh(Sb, Sw + eps * np.eye(d))
    order = np.argsort(w)[::-1][:dim]
    V = V[:, order]
    idx = np.argmax(np.abs(V), axis=0)
    sgn = np.sign(V[idx, np.arange(V.shape[1])])
    sgn[sgn == 0] = 1.0
    return V * sgn, w[order], eps


def plda_prior(U_means, counts, floor=B_FLOOR):
    """b_j = var_p(u_bar_pj) - mean_p(1/n_p), floored (PLAN.md 2 item 2)."""
    v = np.var(U_means, axis=0)
    return np.maximum(v - float(np.mean(1.0 / np.asarray(counts, dtype=np.float64))), floor)


def plda_llr(M, n, b):
    """Pairwise same-player vs different-player log-likelihood ratio, diagonal two-covariance model with unit
    per-game within variance. M [Q, d] question means, n [Q] games, b [d] -> [Q, Q] (diagonal = 0)."""
    M = np.asarray(M, dtype=np.float64)
    w = 1.0 / np.asarray(n, dtype=np.float64)
    Q = len(w)
    out = np.zeros((Q, Q))
    W1, W2 = w[:, None], w[None, :]
    for j in range(M.shape[1]):
        bj = float(b[j])
        x = M[:, j][:, None]
        y = M[:, j][None, :]
        v1, v2 = bj + W1, bj + W2
        det = bj * (W1 + W2) + W1 * W2
        quad = (v2 * x * x - 2.0 * bj * x * y + v1 * y * y) / det
        out += -0.5 * np.log(det) + 0.5 * np.log(v1 * v2) - 0.5 * quad + 0.5 * (x * x / v1 + y * y / v2)
    np.fill_diagonal(out, 0.0)
    return out


def plda_llr_direct(a, na, bvec, nb, b):
    """Reference: the same LLR from explicit multivariate normal densities (tests only)."""
    from scipy.stats import multivariate_normal as mvn
    B = np.diag(b)
    Wa, Wb = np.eye(len(b)) / na, np.eye(len(b)) / nb
    S = np.block([[B + Wa, B], [B, B + Wb]])
    same = mvn(mean=np.zeros(2 * len(b)), cov=S).logpdf(np.concatenate([a, bvec]))
    diff = mvn(mean=np.zeros(len(b)), cov=B + Wa).logpdf(a) + mvn(mean=np.zeros(len(b)), cov=B + Wb).logpdf(bvec)
    return same - diff


# ---------------------------------------------------------------- pools (PLAN.md 3)
class PlayerGames:
    """Usable games of the players of one role: games[p][c] = array of game ids (rows) of colour c."""

    def __init__(self, keys, games_by_colour):
        self.keys = list(keys)
        self.games = games_by_colour
        self.count = np.array([[len(g[c]) for c in COLOURS] for g in games_by_colour], dtype=np.int64)


def make_groups(rng, p_multi, sizes):
    """Steps 1-2: shuffled game counts cut into player groups -> [(ks, colours)] in creation order."""
    ks = np.repeat(np.array(KS, dtype=np.int64), PER_K)
    ks = ks[rng.permutation(len(ks))]
    groups, i = [], 0
    while i < len(ks):
        g = 1
        if p_multi > 0 and rng.random() < p_multi:
            g = int(sizes[rng.integers(0, len(sizes))])
        g = min(g, len(ks) - i)
        kk = [int(x) for x in ks[i:i + g]]
        cols = [COLOURS[int(x)] for x in rng.integers(0, 2, size=g)]
        groups.append([kk, cols])
        i += g
    return groups


def need_of(kk, cols):
    return np.array([sum(k for k, c in zip(kk, cols) if c == col) for col in COLOURS], dtype=np.int64)


def try_pool(rng, pg, p_multi, sizes):
    """One attempt at a pool -> (questions, failed). questions: list of dicts (player, colour, k, games, group)."""
    groups = make_groups(rng, p_multi, sizes)
    order = sorted(range(len(groups)), key=lambda i: (-sum(groups[i][0]), i))
    used = np.zeros(len(pg.keys), dtype=bool)
    assign = {}
    for gi in order:
        kk, cols = groups[gi]
        chosen = None
        for _t in range(COLOUR_RETRIES + 1):
            need = need_of(kk, cols)
            ok = np.nonzero(~used & np.all(pg.count >= need[None, :], axis=1))[0]
            if len(ok):
                chosen = int(ok[rng.integers(0, len(ok))])
                break
            cols = [COLOURS[int(x)] for x in rng.integers(0, 2, size=len(kk))]
            groups[gi][1] = cols
        if chosen is None:
            return None, True
        used[chosen] = True
        assign[gi] = chosen
    questions = []
    for gi, (kk, cols) in enumerate(groups):
        p = assign[gi]
        perm = {c: rng.permutation(pg.games[p][c]) for c in COLOURS}
        pos = {c: 0 for c in COLOURS}
        for k, c in zip(kk, cols):
            g = perm[c][pos[c]:pos[c] + k]
            pos[c] += k
            questions.append({"player": p, "colour": c, "k": k, "games": np.sort(g), "group": gi})
    return questions, False


def make_pool(seed_base, sid, idx, pg, p_multi, sizes):
    """PLAN.md 3: default_rng([seed_base, scenario, pool, attempt]) until an attempt succeeds."""
    for att in range(MAX_POOL_ATTEMPTS):
        rng = np.random.default_rng([seed_base, sid, idx, att])
        qs, failed = try_pool(rng, pg, p_multi, sizes)
        if not failed:
            return qs, att
    raise RuntimeError(f"pool {seed_base}/{sid}/{idx}: no feasible attempt in {MAX_POOL_ATTEMPTS}")


def check_pool(qs, p_multi, sizes, pg=None):
    """Seam S -> list of problems (with pg: every game of a question is a game of that player in that colour)."""
    probs = []
    if pg is not None:
        for q in qs:
            if not np.all(np.isin(q["games"], pg.games[q["player"]][q["colour"]])):
                probs.append("a question not single-colour / not its player's games")
    last = max(q["group"] for q in qs) if qs else -1
    if len(qs) != POOL_Q:
        probs.append(f"{len(qs)} questions")
    ks = np.array([q["k"] for q in qs])
    for k in KS:
        if int((ks == k).sum()) != PER_K:
            probs.append(f"k={k}: {(ks == k).sum()}")
    for q in qs:
        if len(q["games"]) != q["k"] or len(np.unique(q["games"])) != q["k"]:
            probs.append("question games wrong")
    by_group = {}
    for q in qs:
        by_group.setdefault(q["group"], []).append(q)
    players = [g[0]["player"] for g in by_group.values()]
    if len(set(players)) != len(players):
        probs.append("a player in two groups")
    for g in by_group.values():
        if len({q["player"] for q in g}) != 1:
            probs.append("group with two players")
        allg = np.concatenate([q["games"] for q in g])
        if len(np.unique(allg)) != len(allg):
            probs.append("overlapping games within a player")
        allowed = {1} | set(sizes if p_multi > 0 else ())
        if len(g) not in allowed and not (g[0]["group"] == last and len(g) < max(allowed)):
            probs.append(f"group size {len(g)} not allowed")
    return probs


def same_matrix(qs):
    p = np.array([q["player"] for q in qs])
    return p[:, None] == p[None, :]


def question_means(qs, U, row_of):
    """Mean of the projected games of each question -> ([Q, d], n [Q]). row_of maps a game id to a row of U."""
    M = np.stack([U[row_of[q["games"]]].mean(0) for q in qs])
    n = np.array([q["k"] for q in qs], dtype=np.float64)
    return M, n


# ---------------------------------------------------------------- thresholds (PLAN.md 4 items 1-2)
def upper_pairs(A):
    iu = np.triu_indices(A.shape[0], 1)
    return A[iu]


def choose_tau(llr, same, target=PRECISION_TARGET):
    """Largest m with precision of the top m pairs >= target; tau = the m-th LLR (links: LLR >= tau)."""
    llr = np.asarray(llr, dtype=np.float64)
    same = np.asarray(same, dtype=bool)
    o = np.argsort(-llr, kind="stable")
    tp = np.cumsum(same[o])
    prec = tp / np.arange(1, len(o) + 1)
    ok = np.nonzero(prec >= target)[0]
    if not len(ok):
        return float("inf"), {"m": 0, "tp": 0, "precision": None, "recall": 0.0}
    m = int(ok[-1]) + 1
    tau = float(llr[o[m - 1]])
    sel = llr >= tau
    tot = int(same.sum())
    return tau, {"m": m, "links_at_tau": int(sel.sum()), "tp": int((sel & same).sum()),
                 "precision": float((sel & same).sum() / sel.sum()),
                 "recall": float((sel & same).sum() / tot) if tot else None, "true_pairs": tot}


def choose_c(counts, target=NULL_ON_TARGET):
    counts = np.asarray(counts, dtype=np.int64)
    c = 1
    while np.mean(counts >= c) > target:
        c += 1
    return c


# ---------------------------------------------------------------- linking (PLAN.md 4 item 3)
def components(link):
    """Connected components of a boolean [Q, Q] link matrix -> labels [Q]."""
    A = sparse.csr_matrix(np.triu(link, 1))
    _n, lab = csgraph.connected_components(A, directed=False)
    return lab


def pooled_groups(link, max_size=MAX_COMPONENT):
    """-> list of index lists (components of size 2..max_size); larger components are left alone."""
    lab = components(link)
    out, too_big = [], []
    for c in np.unique(lab):
        idx = np.nonzero(lab == c)[0]
        if 2 <= len(idx) <= max_size:
            out.append([int(i) for i in idx])
        elif len(idx) > max_size:
            too_big.append(len(idx))
    return out, too_big


# ---------------------------------------------------------------- scores
def official_score(pred, y):
    d = np.abs(np.asarray(pred) - np.asarray(y))
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def rank_balanced(sc, y):
    sc, y = np.asarray(sc, dtype=np.float64), np.asarray(y)
    per = [sc[y == k].mean() for k in range(L) if (y == k).any()]
    return float(np.mean(per))


def gates(d0, false_on, d30, lo30):
    g1 = bool(d0 >= G1_MIN_DIFF and false_on <= G1_MAX_FALSE_ON)
    g2 = bool(d30 >= G2_MIN_DIFF and lo30 > 0)
    return {"G1": {"diff": d0, "min": G1_MIN_DIFF, "false_on_rate": false_on, "max_false_on": G1_MAX_FALSE_ON,
                   "passed": g1},
            "G2": {"diff": d30, "min": G2_MIN_DIFF, "ci95_lower": lo30, "passed": g2},
            "passed": bool(g1 and g2)}
