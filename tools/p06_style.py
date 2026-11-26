"""PREREG_06 (competition 1): per-game v2 style features, the score terms, fusion, CSLS, the one-to-one switch
and the 06 self-retrieval split. Pure functions (stdlib, numpy, scipy), plus the shared discipline helpers of the
p06 tools (data/local guard, report metadata) at the end of the file.

Definitions: docs/PREREG_06_PLAYER_STYLE.md section 1-2 (frozen); interface: results/prereg06/design/spec.md
section 2. The exploration functions of results/prereg06/explore/c1_headroom (c1h_feats, c1h_lda, c1h_eval,
s8_switch) are ported here, never imported (tests/test_p06_style.py checks the port against them).

Moves: parse_all(sgf) = every p02_moves.MOVE_RE match in order (no 60-move limit; passes never match).
A game 'alternates' when the colour letters are B, W, B, W, ... from move 0. Non-alternating games, and games
without any move, never enter a v2 term (JO, S2, LDA); callers count them. On alternating games the target's
own moves are i = 0, 2, 4, ... (black) or i = 1, 3, 5, ... (white); the opponent's previous move is i - 1 and
the own previous move i - 2.
"""
import builtins
import datetime
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import threading
from collections import Counter

import numpy as np
import scipy.linalg as sla
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p02_moves  # noqa: E402
import p02_player_open as po  # noqa: E402

TOP_N = 5
Z_EPS = 1e-12
SZ = 19
LDA_DIM = 64
LDA_REG = 1e-3
CSLS_SWITCH_K = 3
SR_SALT = "c10-prereg06|sr"
SR_K_SALT = "c10-prereg06|sr-k"
SR_ROUNDS = 10
SR_PARTS = 5
SR_PART_GAMES = 20
JO_MOVES = 80
PH2 = (0, 20, 50, 100, 150, 10 ** 6)


# ---------------------------------------------------------------- moves
def parse_all(sgf):
    """[(colour, 'xy'), ...]: every board move of the game, in order."""
    return [(m.group(1), m.group(2)) for m in p02_moves.MOVE_RE.finditer(sgf)]


def alternates(mv):
    """True when the colours are B, W, B, W, ... from the first move (and there is at least one move)."""
    if not mv:
        return False
    return all(c == ("B" if i % 2 == 0 else "W") for i, (c, _p) in enumerate(mv))


def usable(mv):
    return alternates(mv)


def _xy(p):
    return ord(p[0]) - 97, ord(p[1]) - 97


def _start(target):
    if target == "B":
        return 0
    if target == "W":
        return 1
    raise ValueError("target colour must be B or W, got %r" % (target,))


# ---------------------------------------------------------------- JO: own corner joseki tokens (1.3)
def _corner(x, y):
    if x == 9 or y == 9:
        return None
    return (x > 9) * 2 + (y > 9)


def _to_tl(x, y, cnr):
    if cnr & 2:
        x = 18 - x
    if cnr & 1:
        y = 18 - y
    return x, y


def joseki_seqs(mv, target, max_moves=JO_MOVES):
    """Per corner (moves among the first max_moves of the game, both colours): moves mapped to the top-left
    corner, transposed so that the first off-diagonal move has x < y; each move = side ('o' target, 'p'
    opponent) + point (= c1h_feats.joseki_seqs with 'o' for the target colour)."""
    seqs = {0: [], 1: [], 2: [], 3: []}
    for c, p in mv[:max_moves]:
        x, y = _xy(p)
        cn = _corner(x, y)
        if cn is None:
            continue
        seqs[cn].append((("o" if c == target else "p"), _to_tl(x, y, cn)))
    out = []
    for cn in range(4):
        s = seqs[cn]
        if not s:
            continue
        tr = False
        for _side, (x, y) in s:
            if x != y:
                tr = x > y
                break
        out.append([side + (chr(97 + y) + chr(97 + x) if tr else chr(97 + x) + chr(97 + y)) for side, (x, y) in s])
    return out


def jo_tokens(mv, target):
    """'{C}K{n}:{first n own corner moves}' for n = 1..min(4, own moves) per corner (C = target colour)."""
    _start(target)
    toks = []
    for s in joseki_seqs(mv, target):
        own = [m for m in s if m[0] == "o"]
        for n in range(1, min(4, len(own)) + 1):
            toks.append("%sK%d:%s" % (target, n, "".join(own[:n])))
    return toks


# ---------------------------------------------------------------- S2: 66 style shares (1.4)
def style2(mv, target):
    """1 + 5 phases x 13 shares of the target's moves (c1h_feats.style2_vec for any target colour)."""
    st = _start(target)
    n = len(mv)
    P = [_xy(p) for _c, p in mv]
    out = [n / 100.0]
    for a, b in zip(PH2[:-1], PH2[1:]):
        idx = [i for i in range(st, n, 2) if a <= i < b]
        m = max(len(idx), 1)
        lines = [0] * 5
        rel = [0] * 4
        selfnear = 0
        zone = [0] * 3
        for i in idx:
            x, y = P[i]
            ln = min(x, y, 18 - x, 18 - y)
            lines[min(ln, 4)] += 1
            zone[0 if (min(x, 18 - x) < 6 and min(y, 18 - y) < 6) else (2 if ln >= 6 else 1)] += 1
            if i >= 1:
                d = max(abs(x - P[i - 1][0]), abs(y - P[i - 1][1]))
                rel[0 if d <= 1 else (1 if d == 2 else (2 if d <= 4 else 3))] += 1
            if i >= 2:
                selfnear += max(abs(x - P[i - 2][0]), abs(y - P[i - 2][1])) <= 2
        out += [v / m for v in lines] + [v / m for v in rel] + [selfnear / m] + [v / m for v in zone]
    return np.asarray(out, np.float64)


def style34(mv, target):
    """The 34 coarse style shares of c1h_feats.style_vec (proxy A of PREREG_06 2.5 / 2.6, simulations only)."""
    st = _start(target)
    n = len(mv)
    P = [_xy(p) for _c, p in mv]
    out = [n / 100.0]
    for ph in range(3):
        idx = [i for i in range(st, n, 2) if _phase(i) == ph]
        m = max(len(idx), 1)
        lines = [0] * 5
        rel = [0] * 4
        selfnear = 0
        for i in idx:
            x, y = P[i]
            lines[min(min(x, y, 18 - x, 18 - y), 4)] += 1
            if i >= 1:
                d = max(abs(x - P[i - 1][0]), abs(y - P[i - 1][1]))
                rel[0 if d <= 1 else (1 if d == 2 else (2 if d <= 4 else 3))] += 1
            if i >= 2:
                selfnear += max(abs(x - P[i - 2][0]), abs(y - P[i - 2][1])) <= 2
        out += [v / m for v in lines] + [v / m for v in rel] + [selfnear / m, len(idx) / 100.0]
    return np.asarray(out, np.float64)


# ---------------------------------------------------------------- resp tokens and the LDA input (1.5)
def _phase(i):
    return 0 if i < 60 else (1 if i < 150 else 2)


def _rel(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    d = max(dx, dy)
    if d <= 2:
        return "%d%d" % (min(dx, dy), max(dx, dy))
    return "m" if d <= 4 else "f"


def resp_tokens(mv, target):
    """Per own move i: L{ph}, R{ph} (i >= 1), S{ph} (i >= 2), RS{ph} (i >= 1; '-' when i < 2)."""
    st = _start(target)
    n = len(mv)
    P = [_xy(p) for _c, p in mv]
    toks = []
    for i in range(st, n, 2):
        ph = _phase(i)
        x, y = P[i]
        line = min(x, y, 18 - x, 18 - y) + 1
        toks.append("L%d:%d" % (ph, min(line, 5)))
        if i >= 1:
            toks.append("R%d:%s" % (ph, _rel(P[i], P[i - 1])))
        if i >= 2:
            toks.append("S%d:%s" % (ph, _rel(P[i], P[i - 2])))
        if i >= 1:
            toks.append("RS%d:%s|%s" % (ph, _rel(P[i], P[i - 1]), _rel(P[i], P[i - 2]) if i >= 2 else "-"))
    return toks


def ldain_from(s2, resp_counts, vocab_index):
    """66 + |vocab| values (float32, as the exploration's dense_ldain.npy): S2, then for every vocabulary token
    sqrt(count / in-vocabulary total); all 0 when the in-vocabulary total is 0. resp_counts: Counter."""
    v = np.zeros(len(vocab_index), np.float64)
    tot = 0
    for t, n in resp_counts.items():
        j = vocab_index.get(t)
        if j is not None:
            v[j] = n
            tot += n
    if tot > 0:
        v = np.sqrt(v / tot)
    return np.concatenate([np.asarray(s2, np.float64), v]).astype(np.float32)


def ldain(mv, target, vocab_index):
    return ldain_from(style2(mv, target), Counter(resp_tokens(mv, target)), vocab_index)


def ldain_matrix(S2, R, vocab_cols):
    """Batch ldain: S2 (n x 66), R CSR counts (n x V_all), vocab_cols = column indices of R in vocabulary order.
    Same values as ldain() row by row (float32)."""
    Rv = R[:, vocab_cols].astype(np.float64).tocsr()
    tot = np.asarray(Rv.sum(1)).ravel()
    inv = np.zeros_like(tot)
    inv[tot > 0] = 1.0 / tot[tot > 0]
    Rn = np.sqrt(np.asarray(Rv.multiply(inv[:, None]).todense()))
    return np.hstack([np.asarray(S2, np.float64), Rn]).astype(np.float32)


# ---------------------------------------------------------------- LDA64 (1.6)
def lda_fit(X, labels, reg=LDA_REG, d=LDA_DIM):
    """c1h_lda.fit on X (float array, rows = games): standardise (sd < 1e-9 -> 1), Sw within players, Sb between
    player means weighted by games, both / total games; Sw += reg * trace(Sw) / D * I; eigh(Sb, Sw);
    directions by decreasing eigenvalue. Returns dict(W [D x d], ev [d], mu, sd)."""
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    X = (X - mu) / sd
    lab = np.asarray(labels)
    _uniq, inv = np.unique(lab, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    M = np.zeros((len(n), X.shape[1]))
    np.add.at(M, inv, X)
    M /= n[:, None]
    Xw = X - M[inv]
    Sw = Xw.T @ Xw / len(X)
    Sb = (M * n[:, None]).T @ M / len(X)
    D = Sw.shape[0]
    Sw_r = Sw + reg * np.trace(Sw) / D * np.eye(D)
    ev, W = sla.eigh(Sb, Sw_r)
    order = np.argsort(-ev)
    return {"W": W[:, order][:, :d], "ev": ev[order][:d], "mu": mu, "sd": sd}


def lda_project(X, model, d=LDA_DIM):
    return ((np.asarray(X, np.float64) - model["mu"]) / model["sd"]) @ model["W"][:, :d]


# ---------------------------------------------------------------- per-colour terms (1.6-1.8)
def group_means(V, groups):
    """Mean of V rows per group (zeros for an empty group) and the group sizes."""
    n = np.array([len(g) for g in groups], np.float64)
    A = po.indicator(groups, V.shape[0])
    S = np.asarray(A @ V)
    out = np.zeros_like(S)
    out[n > 0] = S[n > 0] / n[n > 0, None]
    return out, n


def lda_term(Zc, Zq):
    """T_L = - squared Euclidean distance between question and candidate profiles."""
    return -(np.sum(Zq * Zq, 1)[:, None] - 2 * Zq @ Zc.T + np.sum(Zc * Zc, 1)[None, :])


def nb_term(S2, cand_groups, q_groups):
    """T_S2 of one colour (c1h_eval.style_ll with per-candidate n_c and s2w over candidates with >= 2 games).
    Rows / columns with no game are filled by colour_terms; here they get 0."""
    nq, nc = len(q_groups), len(cand_groups)
    out = np.zeros((nq, nc))
    cg = [np.asarray(g, np.int64) for g in cand_groups]
    has_c = np.array([len(g) > 0 for g in cg])
    mu = np.zeros((nc, S2.shape[1]))
    for j, g in enumerate(cg):
        if len(g):
            mu[j] = S2[g].mean(0)
    vv = [S2[g].var(0) for g in cg if len(g) >= 2]
    s2w = (np.mean(np.vstack(vv), 0) if vv else np.zeros(S2.shape[1])) + 1e-6
    ncg = np.array([len(g) for g in cg], np.float64)
    ncg_safe = np.where(ncg > 0, ncg, 1.0)
    for i, q in enumerate(q_groups):
        if not len(q):
            continue
        xq = S2[np.asarray(q, np.int64)].mean(0)
        var = s2w[None, :] / len(q) + s2w[None, :] / ncg_safe[:, None]
        out[i] = -0.5 * (((xq[None, :] - mu) ** 2) / var).sum(1)
    return out, has_c


def colour_terms(per_colour, q_counts):
    """1.8: per_colour = {colour: (T [nq x nc], cand_has [nc])}; q_counts = {colour: usable games per question}.
    T(q) = sum over colours with games of (share of the question's usable games) * z(row, candidates without
    that colour filled with the row minimum first). Returns (T, has) with has = question has any usable game."""
    cols = sorted(per_colour)
    nq = len(next(iter(q_counts.values())))
    total = np.zeros(nq)
    for c in cols:
        total += np.asarray(q_counts[c], np.float64)
    nc = per_colour[cols[0]][0].shape[1]
    out = np.zeros((nq, nc))
    for c in cols:
        T, ch = per_colour[c]
        k = np.asarray(q_counts[c], np.float64)
        rows = np.flatnonzero(k > 0)
        if not len(rows):
            continue
        T = np.array(T[rows], np.float64, copy=True)
        if not ch.all():
            if ch.any():
                T[:, ~ch] = T[:, ch].min(1, keepdims=True)
            else:
                T[:] = 0.0
        out[rows] += (k[rows] / total[rows])[:, None] * zscore_rows(T, np.ones(len(rows), bool))
    return out, total > 0


def jo_cos(G, cand_groups, q_groups):
    """cosO of p02 (p02_player_open.open_cos) on the JO token matrix."""
    return po.open_cos(G, cand_groups, G, q_groups)


# ---------------------------------------------------------------- fusion, CSLS, one-to-one
def zscore_rows(M, row_ok):
    """z over the candidates of each question; 0 where the term is missing or std < 1e-12 (= p02)."""
    M = np.asarray(M, np.float64)
    Z = np.zeros(M.shape)
    if M.size == 0:
        return Z
    mu = M.mean(1)
    sd = M.std(1)
    good = np.asarray(row_ok, bool) & np.isfinite(sd) & (sd >= Z_EPS)
    Z[good] = (M[good] - mu[good, None]) / sd[good, None]
    return Z


def zrows(M):
    return zscore_rows(M, np.ones(np.asarray(M).shape[0], bool))


def fuse(terms, weights):
    """terms: [(matrix, row_ok)], weights: [w]; sum of w * z(term); weight 0 terms skipped."""
    s = None
    for (M, ok), w in zip(terms, weights):
        if w == 0:
            continue
        z = zscore_rows(M, ok) * float(w)
        s = z if s is None else s + z
    return s


def csls(F, k):
    """C = 2F - r_c - r_q (c1h_eval.csls); r_c = mean of the k largest F of the candidate over the questions,
    r_q = mean of the k largest F of the question over the candidates. k None -> F."""
    if k is None:
        return np.asarray(F, np.float64)
    F = np.asarray(F, np.float64)
    rc = -np.sort(-F, axis=0)[:k].mean(0)
    rq = -np.sort(-F, axis=1)[:, :k].mean(1)
    return 2 * F - rc[None, :] - rq[:, None]


def hungarian_top1(Z):
    """Rows assigned to distinct columns maximising the sum of Z (needs rows <= columns: every row gets one)."""
    Z = np.asarray(Z, np.float64)
    if Z.ndim != 2 or Z.shape[0] > Z.shape[1]:
        raise ValueError("hungarian_top1 needs rows <= columns, got shape %s" % (Z.shape,))
    r, c = linear_sum_assignment(-Z)
    a = np.empty(Z.shape[0], np.int64)
    a[r] = c
    return a


def with_top1(S, a):
    """Assigned candidate first (+inf), the others by S."""
    F = np.array(S, np.float64, copy=True)
    F[np.arange(len(a)), a] = np.inf
    return F


def detector(F_real, F_pseudo_list):
    """s8_switch statistics: Z = row z of F; forced_cost = mean Z(top-1) - Z(Hungarian choice); r_forced_cost =
    forced_cost / mean forced_cost of the pseudo pools (rounds 0 and 1)."""
    def fc(F):
        Z = zrows(F)
        a = hungarian_top1(Z)
        top = Z.argmax(1)
        n, m = Z.shape
        return (float(np.mean(Z[np.arange(n), top] - Z[np.arange(n), a])), float(np.mean(a != top)),
                int((np.bincount(top, minlength=m) == 0).sum()), a)
    fr, ch, n0, a = fc(F_real)
    fp = [fc(F)[0] for F in F_pseudo_list]
    den = float(np.mean(fp)) if fp else float("nan")
    return {"forced_cost": fr, "changed": ch, "n0": n0, "pseudo_forced_cost": fp,
            "r_forced_cost": fr / max(den, 1e-9), "assignment": a}


def switch_matrix(F, a=None, k_rest=CSLS_SWITCH_K):
    """Ranking matrix of the switch: rank 1 = Hungarian on z(F) (or the given assignment), ranks 2-5 by CSLS3."""
    if a is None:
        a = hungarian_top1(zrows(F))
    return with_top1(csls(F, k_rest), a)


def top5(S, cand_ids, n=TOP_N):
    """Top-n candidate ids per row, ties by candidate id (cand_ids sorted)."""
    idx = np.arange(S.shape[1])
    out = []
    for row in np.asarray(S, np.float64):
        order = np.lexsort((idx, -row + 0.0))
        out.append([cand_ids[i] for i in order[:n]])
    return out


def switch_top5(F, cand_ids, a=None, k_rest=CSLS_SWITCH_K):
    return top5(switch_matrix(F, a, k_rest), cand_ids)


def ranks(S, owner):
    return po.true_ranks(np.asarray(S, np.float64), owner)


def points(r):
    return po.points(r)


def score(S, owner):
    return float(points(ranks(S, owner)).mean())


# ---------------------------------------------------------------- sr06 (2.3)
def _hash_int(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def sr_order(player_id, game_ids):
    return sorted(game_ids, key=lambda g: hashlib.sha256(("%s|%s|%s" % (SR_SALT, player_id, g)).encode("utf-8"))
                  .hexdigest())


def sr06_rounds(cand_games, ks, n_rounds=SR_ROUNDS, strict=True, query_filter=None):
    """cand_games: {player_id: [game_id, ...]}; ks: the pool's question num_games list. Returns rounds[r] =
    [(player_id, query_ids, reference_ids)] in sorted player order. Every candidate's games are ordered by
    sha256('c10-prereg06|sr|pid|gid') and cut by position into 5 parts; round r uses part r mod 5; K = min(k, 20)
    sorted ascending, then shuffled with random.Random(int(sha256('c10-prereg06|sr-k|r')[:16], 16)); candidate i
    (sorted ids) takes K[i mod |K|]; query = the first k games of the part (hash order), references = the other
    4 parts. strict: exactly 100 distinct games per candidate. query_filter(pid, gid) -> bool: the query is the
    first k games of the part that pass it (apply's record-only black-only check); the references are unchanged."""
    K0 = sorted(min(int(k), SR_PART_GAMES) for k in ks)
    if not K0:
        raise ValueError("empty K list")
    pids = sorted(cand_games)
    parts = {}
    for p in pids:
        g = list(cand_games[p])
        if len(set(g)) != len(g):
            raise ValueError("candidate %s: duplicate game ids" % p)
        if strict and len(g) != SR_PARTS * SR_PART_GAMES:
            raise ValueError("candidate %s has %d games; sr06 needs exactly 100" % (p, len(g)))
        o = sr_order(p, g)
        n = len(o)
        parts[p] = [o[j * n // SR_PARTS:(j + 1) * n // SR_PARTS] for j in range(SR_PARTS)]
    rounds = []
    for r in range(n_rounds):
        K = list(K0)
        random.Random(_hash_int("%s|%d" % (SR_K_SALT, r))).shuffle(K)
        j = r % SR_PARTS
        items = []
        for i, p in enumerate(pids):
            k = K[i % len(K)]
            part = parts[p][j]
            q = part if query_filter is None else [g for g in part if query_filter(p, g)]
            items.append((p, q[:k], [g for jj, pp in enumerate(parts[p]) if jj != j for g in pp]))
        rounds.append(items)
    return rounds


# ---------------------------------------------------------------- statistics
def mean_ci(x, alpha=0.05):
    from scipy import stats
    x = np.asarray(x, np.float64)
    m = float(x.mean())
    if len(x) < 2:
        return [m, None, None]
    h = float(stats.t.ppf(1 - alpha / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return [m, m - h, m + h]


# ================================================================ discipline helpers (stdlib)
_FORBIDDEN_DIR = os.path.normcase(os.path.abspath(os.path.join(REPO, "data", "local")))
_ANS = "ans" + "wers"
LOCAL_ALLOWED = ("player_val_candidates.csv", "player_val_questions.csv")
_ALLOW = {"on": False}


class Refused(RuntimeError):
    pass


def under_local(p):
    """Prefix test on the absolute and the resolved path (never a substring test: AppData\\Local)."""
    if isinstance(p, int):
        return False
    try:
        s = os.path.normcase(os.path.abspath(os.fspath(p)))
    except TypeError:
        return False
    if s == _FORBIDDEN_DIR or s.startswith(_FORBIDDEN_DIR + os.sep):
        return True
    try:
        r = os.path.normcase(os.path.realpath(s))
    except (OSError, ValueError):
        return False
    return r == _FORBIDDEN_DIR or r.startswith(_FORBIDDEN_DIR + os.sep)


def refuse_marker(p):
    if p is not None and not isinstance(p, int) and _ANS in os.path.basename(str(os.fspath(p))).lower():
        raise Refused("refusing %s: p06 tools never read answer files" % (p,))
    return p


def _allowed_local(p):
    if not _ALLOW["on"]:
        return False
    s = os.path.normcase(os.path.abspath(os.fspath(p)))
    return any(s == os.path.normcase(os.path.join(_FORBIDDEN_DIR, n)) for n in LOCAL_ALLOWED)


def install_guard():
    """open / os.open / io.FileIO / listdir / scandir / stat / lstat on data/local raise (except the two game files
    while allow_local() is on);
    any path whose name contains the answer-file marker raises."""
    if getattr(builtins, "_c10_p06_guard", False):
        return
    real_open, real_listdir, real_scandir, real_stat = builtins.open, os.listdir, os.scandir, os.stat

    busy = threading.local()

    def chk(p, what, allow_file=False):
        if isinstance(p, int) or getattr(busy, "on", False):
            return                       # re-entrant call from realpath() inside the check itself
        busy.on = True
        try:
            refuse_marker(p)
            if under_local(p) and not (allow_file and _allowed_local(p)):
                raise Refused("p06 guard: refusing to %s %s (data/local)" % (what, p))
        finally:
            busy.on = False

    def g_open(file, *a, **k):
        chk(file, "open", True)
        return real_open(file, *a, **k)

    def g_listdir(path="."):
        chk(path, "list")
        return real_listdir(path)

    def g_scandir(path="."):
        chk(path, "scan")
        return real_scandir(path)

    def g_stat(path, *a, **k):
        chk(path, "stat", True)
        return real_stat(path, *a, **k)

    real_osopen, real_lstat, real_fileio = os.open, os.lstat, io.FileIO

    def g_osopen(path, *a, **k):
        chk(path, "open", True)
        return real_osopen(path, *a, **k)

    def g_lstat(path, *a, **k):
        chk(path, "stat", True)
        return real_lstat(path, *a, **k)

    class GFileIO(real_fileio):
        def __init__(self, file, *a, **k):
            chk(file, "open", True)
            super().__init__(file, *a, **k)

    builtins.open, io.open, os.listdir, os.scandir, os.stat = g_open, g_open, g_listdir, g_scandir, g_stat
    os.open, os.lstat, io.FileIO = g_osopen, g_lstat, GFileIO
    builtins._c10_p06_guard = True


class allow_local:
    """Context manager: the two validation game files may be opened (apply only)."""

    def __enter__(self):
        _ALLOW["on"] = True
        return self

    def __exit__(self, *exc):
        _ALLOW["on"] = False
        return False


def sha256_file(path):
    h = hashlib.sha256()
    with open(refuse_marker(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_lf(path):
    with open(refuse_marker(path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def git(*args):
    r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def inside_repo(path):
    a = os.path.normcase(os.path.abspath(path))
    r = os.path.normcase(REPO)
    return a == r or a.startswith(r + os.sep)


def rel(path):
    """Repository-relative path with '/', or the absolute path when outside the repository."""
    if not inside_repo(path):
        return os.path.abspath(path)
    return os.path.relpath(os.path.abspath(path), REPO).replace(os.sep, "/")


def file_committed_clean(path):
    r = rel(path)
    if not git("ls-files", r):
        return False
    return subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", r]).returncode == 0


P06_TOOLS = ("tools/p06_style.py", "tools/p06_sim.py", "tools/p06_player.py")


def tools_state(names=P06_TOOLS):
    out = {}
    for n in names:
        p = os.path.join(REPO, n)
        out[n] = {"sha256_lf": sha256_lf(p) if os.path.exists(p) else None,
                  "committed_and_clean": file_committed_clean(p) if os.path.exists(p) else False}
    return out


def env_versions():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "executable": os.path.relpath(sys.executable, REPO).replace(os.sep, "/")
            if os.path.abspath(sys.executable).startswith(REPO) else sys.executable}


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(json_safe(obj), f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(refuse_marker(path), encoding="utf-8") as f:
        return json.load(f)
