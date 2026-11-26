"""Competition 1 (player identification), PREREG_02 section 2.2: opening tokens, TF-IDF cosine
(cosO), the self-retrieval split, and the calibration and rule of the three-level split guard.
Stdlib + numpy + scipy only.

Shared preprocessing comes only from tools/c10data.py and tools/p02_moves.py. This module never
imports a rank tool and never reads a rank output. It never opens a validation answer file.

Opening tokens (frozen by PREREG_02 2.2 / plan.md 4.3), per game:
    own = the target player's first 10 moves, seq = the first 10 moves of the game (both colours)
    colour-prefixed set: '{c}{i}:{m}' (i = 0..9), '{c}*:{m}', '{c}q:{c1}{m1}>{c2}{m2}' (seq bigrams)
    colour-free set:     'o{i}:{m}',             'o*:{m}',   'q:{m1}>{m2}'
    both sets are always used together (token spelling as in
    results/prereg02/explore/opening_player_sim.py, sets 'col' and 'nocol').
    Moves: p02_moves.parse_moves (first 60 moves). Orientation 'raw' = coordinates as in the SGF;
    'canon' = p02_moves.canon(mv, 10): the D4 symmetry that minimises the first 10 moves, applied
    to the whole game.
Vectors (one per candidate, one per question): token counts summed over its games, weight
1 + ln(count), times IDF = ln((1 + P) / (1 + df)) + 1 with P = number of candidates in the pool
and df counted over the candidate vectors only (in self-retrieval: over the reference vectors);
a token seen only in a query gets df = 0. Rows are L2-normalised (norm + 1e-12, as in explore);
cosO = cosine between a question vector and each candidate vector.

Self-retrieval split (PREREG_02 2.2 '候選池自我檢索', plan.md 4.3 'select'):
    each candidate's games are sorted by sha256('c10-prereg02|sr|{player_id}|{game_id}') and cut
    by position into 5 parts (100 games -> 5 x 20). Round r = 0..9 uses part r mod 5, with
    rng = random.Random(int(sha256('c10-prereg02|sr-k|{r}').hexdigest()[:16], 16)); candidates in
    sorted id order draw k = rng.randint(5, 20) and query = rng.sample(part, k); the references are
    the games of the other 4 parts (80 games). The remaining 20 - k games of the part are unused.

Split guard (PREREG_02 rev 2, 2.2 '切法防護', three levels):
    rho = median over the real questions of max_c cosO / median over the self-retrieval
    pseudo-queries of max_c cosO (same pool, orientation of the selected combination; questions
    without any token are left out). b_c = share of the pool's candidate games in which the
    target player is black, b_q = the same over the question games.
    Calibration (this file, 'calibrate-guard'): simulated pools of FIT players only (training
    data via c10data.load_rows(part='fit')), built exactly like make_pool() of
    results/prereg02/explore/opening_player_sim.py: 40 players per level among the fit players
    with >= 105 games, chosen per level by sha256('{seed}|{player_id}'); rng =
    random.Random('pool|{seed}'); candidates = the first 100 games, real query = the next k =
    rng.randint(5, 20) games, after ordering the player's games by the split:
        R 'random'           shuffled with rng
        F 'fileorder'        file order (each player's black games come first)
        X 'colour_separated' black games then white games, each in file order; the query is the
                             LAST q = min(k, n - 100) games (almost all white)
    3 pools per split (seeds 0, 1, 2). Per orientation: rho_R, rho_F, rho_X = mean rho of the
    split's pools; theta1 = (rho_R + rho_F) / 2, theta2 = (rho_F + rho_X) / 2; a level is
    disabled when its adjacent gap (rho_R - rho_F for theta1, rho_F - rho_X for theta2) is < 0.05;
    when theta2 is disabled, theta_RX = (rho_R + rho_X) / 2 is the only threshold (below it:
    level 3).
    Rule (guard_level / apply in p02_player_fuse.py), not applied to CNN-only combinations:
        level 3 if rho < theta2 or |b_c - b_q| > 0.5: CNN only (the E part; Q22 / Q23 -> Q00)
        level 2 if theta2 <= rho < theta1: w = min(w, 1); Q22 / Q23 -> E0 + same orientation, w = 1
        level 1 otherwise: the selected combination
        (theta1 disabled: level 2 never fires; theta2 disabled: level 3 iff rho < theta_RX or
        the colour condition, else level 1)

    python tools/p02_player_open.py calibrate-guard [--out-dir results/prereg02/player]
        [--train-dir data/training] [--split-csv results/split_v1/players_split.csv]
        [--local-dir data/local] [--smoke]

    writes OUT_DIR/guard_calibration.json (about 3-4 minutes, most of it reading the fit data).
    OUT_DIR inside results/prereg01 is always refused.
    --smoke: 4 players per level and seed 0 only (code-path check; the numbers mean nothing);
    refused when OUT_DIR is inside the repository.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p02_moves  # noqa: E402  (shared preprocessing)
from c10data import LEVELS, load_rows  # noqa: E402

N_OWN = 10                  # own moves and whole-game moves used for tokens
N_CANON = 10                # canon(mv, 10)
ORIENTS = ("raw", "canon")
K_MIN, K_MAX = 5, 20        # query size
SR_SALT = "c10-prereg02|sr"
SR_K_SALT = "c10-prereg02|sr-k"
SR_ROUNDS = 10
SR_PARTS = 5
SR_GAMES = 100              # games per candidate expected in a real pool
TOP_N = 5

CAL_SEEDS = (0, 1, 2)
CAL_SPLITS = ("random", "fileorder", "colour_separated")
CAL_LABEL = {"random": "R", "fileorder": "F", "colour_separated": "X"}
CAL_MIN_GAMES = 105         # eligible fit players: >= 100 candidate games + >= 5 query games
CAL_PER_LEVEL = 40          # 40 players x 10 levels = 400 candidates
CAL_MIN_GAP = 0.05
COLOUR_MAX_DIFF = 0.5       # |b_c - b_q| > 0.5 -> level 3
SMOKE_PER_LEVEL = 4
SMOKE_SEEDS = (0,)

# Validation answer files must never be opened; the needle is spelled in two halves so that
# no source file of the p02 tools contains the word (checked by tests/test_p02_player.py).
_FORBIDDEN = "ans" + "wers"


# ---------------------------------------------------------------- small helpers
def refuse_forbidden(path):
    """Stop before opening anything that looks like a validation answer file."""
    if path is not None and _FORBIDDEN in os.path.basename(str(path)).lower():
        raise SystemExit(f"refusing to open {path!r}: p02 tools never read answer files")
    return path


def inside(path, root):
    real = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    root = os.path.normcase(os.path.realpath(os.path.abspath(root)))
    return real == root or real.startswith(root.rstrip(os.sep) + os.sep)


def inside_repo(path):
    return inside(path, REPO)


def check_out_dir(out_dir, smoke, extra_readonly=()):
    """Never write into prereg01's results (results/prereg01, or a --p01-dir given as extra_readonly),
    with or without --smoke; --smoke outputs must also stay outside the repository."""
    for ro in (os.path.join(REPO, "results", "prereg01"),) + tuple(p for p in extra_readonly if p):
        if inside(out_dir, ro):
            raise SystemExit(f"refusing to write into {out_dir!r}: it is inside the read-only {ro!r}")
    if smoke and inside_repo(out_dir):
        raise SystemExit(f"--smoke refused: output dir {out_dir!r} is inside the repository; "
                         f"use a scratch directory")


def sha256_file(path):
    h = hashlib.sha256()
    with open(refuse_forbidden(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_file(path):
    """SHA-256 of a text file with CRLF folded to LF (git autocrlf may rewrite line ends)."""
    with open(refuse_forbidden(path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def source_shas():
    names = ["p02_player_open.py", "p02_player_fuse.py", "p02_moves.py", "c10data.py"]
    out = {}
    for n in names:
        p = os.path.join(HERE, n)
        out[n] = sha256_text_file(p) if os.path.exists(p) else None
    return out


def _json_safe(o):
    """numpy scalars -> Python; NaN / inf -> None (strict JSON)."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_safe(obj), f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _hash_int(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


# ---------------------------------------------------------------- moves and tokens
def moves_of(sgf):
    """[(colour, 'ab'), ...]: the first 60 moves, via the shared p02_moves parser."""
    return p02_moves.parse_moves(sgf)


def orient_moves(mv, orient):
    if orient == "raw":
        return mv
    if orient == "canon":
        return p02_moves.canon(mv, N_CANON)
    raise ValueError(f"unknown orientation {orient!r}")


def col_tokens(color, mv, n=N_OWN):
    own = [m for c, m in mv if c == color][:n]
    toks = []
    for i, m in enumerate(own):
        toks.append("%s%d:%s" % (color, i, m))      # positional own move
        toks.append("%s*:%s" % (color, m))          # unordered own move
    seq = [c + m for c, m in mv[:n]]
    for i in range(1, len(seq)):
        toks.append("%sq:%s>%s" % (color, seq[i - 1], seq[i]))   # whole-game bigrams
    return toks


def nocol_tokens(color, mv, n=N_OWN):
    own = [m for c, m in mv if c == color][:n]
    toks = ["o%d:%s" % (i, m) for i, m in enumerate(own)] + ["o*:%s" % m for m in own]
    seq = [m for c, m in mv[:n]]
    toks += ["q:%s>%s" % (seq[i - 1], seq[i]) for i in range(1, len(seq))]
    return toks


def game_tokens(color, mv, orient):
    """Both token sets of one game, for colour 'B' or 'W' of the target player."""
    if color not in ("B", "W"):
        raise ValueError(f"target colour must be 'B' or 'W', got {color!r}")
    mv = orient_moves(mv, orient)
    return col_tokens(color, mv) + nocol_tokens(color, mv)


def token_matrices(game_sets, orient):
    """game_sets: list of lists of (colour, moves). Returns one CSR count matrix (games x V) per
    set, all with the same columns (one shared vocabulary)."""
    vocab = {}
    parts = []
    for games in game_sets:
        rows, cols, vals = [], [], []
        for i, (color, mv) in enumerate(games):
            for t, n in Counter(game_tokens(color, mv, orient)).items():
                rows.append(i)
                cols.append(vocab.setdefault(t, len(vocab)))
                vals.append(n)
        parts.append((len(games), rows, cols, vals))
    V = len(vocab)
    return [sp.csr_matrix((np.asarray(v, dtype=np.float64), (np.asarray(r, dtype=np.int64),
                                                               np.asarray(c, dtype=np.int64))),
                          shape=(n, V)) for n, r, c, v in parts]


def indicator(groups, n_rows):
    """groups: list of lists of row indices -> CSR (len(groups) x n_rows) of ones."""
    sizes = [len(g) for g in groups]
    r = np.repeat(np.arange(len(groups)), sizes)
    c = np.fromiter((i for g in groups for i in g), dtype=np.int64, count=int(sum(sizes)))
    return sp.csr_matrix((np.ones(len(c)), (r, c)), shape=(len(groups), n_rows))


def _l2_rows(m):
    norm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel()) + 1e-12
    return sp.diags(1.0 / norm) @ m


def open_cos(G_cand, cand_groups, G_query, query_groups):
    """cosO (n_queries x n_candidates) and, per query, whether it has any token.

    G_cand / G_query: game x token count matrices with the same columns (may be the same
    matrix). IDF uses the candidate vectors only (P = number of candidate groups)."""
    C = (indicator(cand_groups, G_cand.shape[0]) @ G_cand).tocsr()
    Q = (indicator(query_groups, G_query.shape[0]) @ G_query).tocsr()
    C.eliminate_zeros()
    Q.eliminate_zeros()
    P = C.shape[0]
    df = np.diff(C.tocsc().indptr)
    idf = np.log((1.0 + P) / (1.0 + df)) + 1.0

    def weigh(m):
        m = m.copy()
        m.data = 1.0 + np.log(m.data)
        return _l2_rows(m.multiply(idf).tocsr())

    cos = np.asarray((weigh(Q) @ weigh(C).T).toarray())
    has = np.diff(Q.indptr) > 0
    return cos, has


def colour_share(colours):
    """Share of games in which the target player is black (None for no game)."""
    colours = list(colours)
    return sum(c == "B" for c in colours) / len(colours) if colours else None


# ---------------------------------------------------------------- ranking helpers
def true_ranks(S, true_idx):
    """1-based rank of the true candidate per row; ties go to the smaller candidate index
    (candidates are kept in sorted id order, so ties go to the smaller id)."""
    n_q, n_c = S.shape
    true_idx = np.asarray(true_idx, dtype=np.int64)
    st = S[np.arange(n_q), true_idx][:, None]
    idx = np.arange(n_c)[None, :]
    return 1 + (S > st).sum(1) + ((S == st) & (idx < true_idx[:, None])).sum(1)


def points(ranks):
    """Official metric: e^-(r-1) for r <= 5, else 0."""
    ranks = np.asarray(ranks, dtype=np.float64)
    return np.where(ranks <= TOP_N, np.exp(-(ranks - 1.0)), 0.0)


# ---------------------------------------------------------------- self-retrieval split
def sr_order(player_id, game_ids):
    return sorted(game_ids, key=lambda g: hashlib.sha256(
        f"{SR_SALT}|{player_id}|{g}".encode("utf-8")).hexdigest())


def sr_parts(player_id, game_ids, n_parts=SR_PARTS):
    order = sr_order(player_id, game_ids)
    n = len(order)
    return [order[j * n // n_parts:(j + 1) * n // n_parts] for j in range(n_parts)]


def sr_rounds(cand_games, strict=True, n_rounds=SR_ROUNDS):
    """cand_games: {player_id: [game_id, ...]}. Returns rounds[r] = [(player_id, query_ids,
    reference_ids), ...] in sorted player_id order. strict: every candidate must have exactly
    SR_GAMES distinct games (5 parts of 20). Non-strict (smoke / synthetic tests): any count; a
    part smaller than k gives a query of the whole part and an empty part gives no query (the
    candidate still has its references). With 100 games both modes give the same draws."""
    pids = sorted(cand_games)
    parts = {}
    for p in pids:
        gids = list(cand_games[p])
        if len(set(gids)) != len(gids):
            raise ValueError(f"candidate {p}: duplicate game ids")
        if strict and len(gids) != SR_GAMES:
            raise ValueError(f"candidate {p} has {len(gids)} games; self-retrieval needs exactly "
                             f"{SR_GAMES} (5 parts of 20)")
        parts[p] = sr_parts(p, gids)
    rounds = []
    for r in range(n_rounds):
        rng = random.Random(_hash_int(f"{SR_K_SALT}|{r}"))
        j = r % SR_PARTS
        items = []
        for p in pids:
            k = rng.randint(K_MIN, K_MAX)
            part = parts[p][j]
            query = rng.sample(part, min(k, len(part)))
            refs = [g for jj, pp in enumerate(parts[p]) if jj != j for g in pp]
            items.append((p, query, refs))
        rounds.append(items)
    return rounds


# ---------------------------------------------------------------- guard rule
def guard_thresholds(rho_r, rho_f, rho_x):
    """theta1, theta2, theta_RX and which levels are enabled (PREREG_02 rev 2, 2.2)."""
    gap1, gap2 = rho_r - rho_f, rho_f - rho_x
    return {"rho_R": rho_r, "rho_F": rho_f, "rho_X": rho_x,
            "theta1": (rho_r + rho_f) / 2.0, "theta2": (rho_f + rho_x) / 2.0,
            "theta_RX": (rho_r + rho_x) / 2.0,
            "gap1_R_minus_F": gap1, "gap2_F_minus_X": gap2, "gap_R_minus_X": rho_r - rho_x,
            "theta1_enabled": bool(gap1 >= CAL_MIN_GAP), "theta2_enabled": bool(gap2 >= CAL_MIN_GAP)}


def guard_level(rho, th, b_c, b_q):
    """Guard level 1 / 2 / 3 for one pool (th = guard_thresholds of the orientation) and why."""
    colour_bad = b_c is not None and b_q is not None and abs(b_c - b_q) > COLOUR_MAX_DIFF
    rho_ok = rho is not None and np.isfinite(rho)
    why = []
    if colour_bad:
        why.append("|b_c - b_q| = %.4f > %.1f" % (abs(b_c - b_q), COLOUR_MAX_DIFF))
    if not rho_ok:
        why.append("rho not computable (no question with an opening token)")
        return (3 if colour_bad else 1), "; ".join(why) or "rho not computable"
    if not th["theta2_enabled"]:
        if rho < th["theta_RX"]:
            why.append("theta2 disabled; rho %.4f < theta_RX %.4f" % (rho, th["theta_RX"]))
            return 3, "; ".join(why)
        if colour_bad:
            return 3, "; ".join(why)
        return 1, "theta2 disabled; rho %.4f >= theta_RX %.4f" % (rho, th["theta_RX"])
    if rho < th["theta2"]:
        why.append("rho %.4f < theta2 %.4f" % (rho, th["theta2"]))
        return 3, "; ".join(why)
    if colour_bad:
        return 3, "; ".join(why)
    if th["theta1_enabled"] and rho < th["theta1"]:
        return 2, "theta2 %.4f <= rho %.4f < theta1 %.4f" % (th["theta2"], rho, th["theta1"])
    if th["theta1_enabled"]:
        return 1, "rho %.4f >= theta1 %.4f, colour shares within %.1f" % (rho, th["theta1"], COLOUR_MAX_DIFF)
    return 1, "theta1 disabled; rho %.4f >= theta2 %.4f, colour shares within %.1f" % (
        rho, th["theta2"], COLOUR_MAX_DIFF)


# ---------------------------------------------------------------- guard: rho on one pool
def pool_open_stats(pids, cand_games, query_games, games, strict=True):
    """Opening statistics of one pool, for both orientations.

    pids: candidate ids (sorted here). cand_games / query_games: {pid: [game_id]} (query_games
    may lack a pid = that candidate has no real query). games: {(pid, game_id): (colour, moves)}.
    Returns {orient: {...}} with the median of max_c cosO over the real queries and over the
    self-retrieval pseudo-queries, rho, and the opening-only official score of both
    (diagnostic), plus the colour shares b_c, b_q."""
    pids = sorted(pids)
    keys = sorted({(p, g) for p in pids for g in cand_games[p]} |
                  {(p, g) for p in pids for g in query_games.get(p, [])})
    row = {k: i for i, k in enumerate(keys)}
    seq = [games[k] for k in keys]
    rounds = sr_rounds({p: cand_games[p] for p in pids}, strict=strict)
    out = {"b_c": colour_share(games[(p, g)][0] for p in pids for g in cand_games[p]),
           "b_q": colour_share(games[(p, g)][0] for p in pids for g in query_games.get(p, []))}
    for orient in ORIENTS:
        (G,) = token_matrices([seq], orient)
        c_groups = [[row[(p, g)] for g in cand_games[p]] for p in pids]
        q_p = [i for i, p in enumerate(pids) if query_games.get(p)]
        q_groups = [[row[(pids[i], g)] for g in query_games[pids[i]]] for i in q_p]
        cos, has = open_cos(G, c_groups, G, q_groups)
        real_max = cos.max(1)[has]
        real_pts = points(true_ranks(cos, q_p))
        sr_max, sr_pts = [], []
        for items in rounds:
            c_groups = [[row[(p, g)] for g in refs] for p, _q, refs in items]
            qi = [i for i, (_p, q, _r) in enumerate(items) if q]
            if not qi:
                continue
            q_groups = [[row[(items[i][0], g)] for g in items[i][1]] for i in qi]
            cos, has = open_cos(G, c_groups, G, q_groups)
            sr_max.extend(cos.max(1)[has].tolist())
            sr_pts.extend(points(true_ranks(cos, qi)).tolist())
        real_med = float(np.median(real_max)) if len(real_max) else float("nan")
        sr_med = float(np.median(sr_max)) if sr_max else float("nan")
        out[orient] = {
            "real_max_cos_median": real_med,
            "sr_max_cos_median": sr_med,
            "rho": real_med / sr_med if sr_med > 0 else float("nan"),
            "n_real_queries": len(q_p),
            "n_real_queries_with_tokens": int(len(real_max)),
            "n_sr_queries_with_tokens": len(sr_max),
            "real_open_score": float(np.mean(real_pts)) if len(real_pts) else float("nan"),
            "sr_open_score": float(np.mean(sr_pts)) if sr_pts else float("nan"),
        }
    return out


def colour_sorted(g):
    return [x for x in g if x[1] == "B"] + [x for x in g if x[1] == "W"]


def sim_pool(by_player, elig, seed, split, per_level=CAL_PER_LEVEL):
    """make_pool() of results/prereg02/explore/opening_player_sim.py. by_player[p] = [(game_id,
    colour, sgf), ...] in file order. Returns (pool, {pid: candidate game ids}, {pid: query ids})."""
    rng = random.Random("pool|%d" % seed)
    pool = []
    for lv in LEVELS:
        pl = sorted(elig[lv], key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        pool += pl[:per_level]
    cand, query = {}, {}
    for p in pool:
        g = list(by_player[p])
        if split == "random":
            rng.shuffle(g)
        elif split == "colour_separated":
            g = colour_sorted(g)
        elif split != "fileorder":
            raise ValueError(split)
        k = rng.randint(K_MIN, K_MAX)
        cand[p] = [x[0] for x in g[:SR_GAMES]]
        if split == "colour_separated":
            q = min(k, len(g) - SR_GAMES)
            query[p] = [x[0] for x in g[len(g) - q:]]
        else:
            query[p] = [x[0] for x in g[SR_GAMES:SR_GAMES + k]]
    return pool, cand, query


def calibrate(by_player, level, per_level, seeds, log=print):
    """Guard calibration on the fit pools. by_player: {pid: [(game_id, colour, sgf), ...]} in file
    order; level: {pid: level}."""
    elig = defaultdict(list)
    for p, g in by_player.items():
        if len(g) >= CAL_MIN_GAMES:
            elig[level[p]].append(p)
    moves_cache = {}
    pools = []
    for split in CAL_SPLITS:
        for seed in seeds:
            t = time.time()
            pool, cand, query = sim_pool(by_player, elig, seed, split, per_level)
            games = {}
            for p in pool:
                for gid, color, sgf in by_player[p]:
                    key = (p, gid)
                    if key in games:
                        raise ValueError(f"duplicate game id {gid} of player {p} in the fit data")
                    if key not in moves_cache:
                        moves_cache[key] = moves_of(sgf)
                    games[key] = (color, moves_cache[key])
            stats = pool_open_stats(pool, cand, query, games, strict=True)
            pools.append({"split": split, "seed": seed, "n_candidates": len(pool),
                          "mean_query_games": float(np.mean([len(query[p]) for p in pool])),
                          "b_c": stats.pop("b_c"), "b_q": stats.pop("b_q"),
                          "stats": stats})
            log("pool %-16s seed %d  rho raw %.4f canon %.4f  open score raw %.4f canon %.4f  "
                "b_c %.3f b_q %.3f (%.0fs)"
                % (split, seed, stats["raw"]["rho"], stats["canon"]["rho"],
                   stats["raw"]["real_open_score"], stats["canon"]["real_open_score"],
                   pools[-1]["b_c"], pools[-1]["b_q"], time.time() - t))
    guard = {}
    for orient in ORIENTS:
        m = {s: float(np.mean([x["stats"][orient]["rho"] for x in pools if x["split"] == s]))
             for s in CAL_SPLITS}
        guard[orient] = guard_thresholds(m["random"], m["fileorder"], m["colour_separated"])
    return {"eligible_players_by_level": {lv: len(elig[lv]) for lv in LEVELS},
            "pools": pools, "guard": guard}


def cmd_calibrate_guard(args):
    t0 = time.time()
    check_out_dir(args.out_dir, args.smoke)
    rows = load_rows(args.train_dir, args.split_csv, part="fit", local_dir=args.local_dir)
    n_games = len(rows)
    by_player = defaultdict(list)
    level = {}
    for r in rows:
        if r["color"] not in ("B", "W"):
            raise ValueError(f"unexpected colour {r['color']!r} in game {r['game_id']}")
        by_player[r["player_id"]].append((r["game_id"], r["color"], r["sgf_content"]))
        level[r["player_id"]] = r["rank"]
    del rows
    t_load = time.time() - t0
    print("loaded %d fit games of %d players in %.0fs" % (n_games, len(by_player), t_load), flush=True)
    per_level = SMOKE_PER_LEVEL if args.smoke else CAL_PER_LEVEL
    seeds = SMOKE_SEEDS if args.smoke else CAL_SEEDS
    cal = calibrate(by_player, level, per_level, seeds, log=lambda s: print(s, flush=True))
    out = {
        "what": "PREREG_02 rev 2 2.2 split-guard calibration (切法防護 theta1 / theta2) on simulated "
                "pools of FIT players; training data only via tools/c10data.load_rows(part='fit'); "
                "no validation games, no answer files, no CNN",
        "smoke": bool(args.smoke),
        "data": {"part": "fit", "games": n_games, "players": len(by_player),
                 "eligible_rule": "fit players with >= %d games" % CAL_MIN_GAMES,
                 "eligible_players_by_level": cal["eligible_players_by_level"]},
        "setup": {
            "pools": "%d splits x %d seeds; %d players per level; candidates = first %d games of "
                     "the split's order; real query = next k = rng.randint(%d, %d) games (R, F) or "
                     "the last min(k, n - %d) games (X) (make_pool of "
                     "results/prereg02/explore/opening_player_sim.py)"
                     % (len(CAL_SPLITS), len(seeds), per_level, SR_GAMES, K_MIN, K_MAX, SR_GAMES),
            "splits": {"random (R)": "games shuffled with rng = random.Random('pool|<seed>')",
                       "fileorder (F)": "games in file order (each player's black games first)",
                       "colour_separated (X)": "black games then white games, each in file order; "
                                               "query = the last games (almost all white)"},
            "seeds": list(seeds),
            "players_chosen_by": "per level, sort eligible players by sha256('<seed>|<player_id>'), "
                                 "take the first %d" % per_level,
            "tokens": "own first %d moves + whole-game first %d moves; colour-prefixed and "
                      "colour-free sets together; orientations %s" % (N_OWN, N_OWN, list(ORIENTS)),
            "vector": "1 + ln(count) x IDF (df over candidate vectors, P = candidates), L2, cosine",
            "self_retrieval": "sha256('%s|<player_id>|<game_id>') order, 5 parts of 20, round r uses "
                              "part r mod 5, rng seeded sha256('%s|<r>')[:16], k = randint(%d, %d), "
                              "references = other 80 games, IDF from references; %d rounds"
                              % (SR_SALT, SR_K_SALT, K_MIN, K_MAX, SR_ROUNDS),
            "rho": "median over real queries of max_c cosO / median over self-retrieval "
                   "pseudo-queries of max_c cosO (queries without any token are left out)",
            "colour_shares": "b_c = share of candidate games with the target player black, b_q = "
                             "the same over query games (pool level)",
        },
        "pools": cal["pools"],
        "guard": cal["guard"],
        "min_gap": CAL_MIN_GAP,
        "colour_max_diff": COLOUR_MAX_DIFF,
        "rule": "not applied to CNN-only combinations (Q00, Q01). Orientation = that of the selected "
                "combination. level 3 if rho < theta2 or |b_c - b_q| > 0.5: CNN only (E part of the "
                "combination; Q22 / Q23 -> Q00). level 2 if theta2 <= rho < theta1: w = min(w, 1); "
                "Q22 -> Q04, Q23 -> Q14 (E0 + same orientation, w = 1). level 1 otherwise: the "
                "selected combination. theta1 disabled (gap1 < 0.05): level 2 never fires. theta2 "
                "disabled (gap2 < 0.05): theta_RX is the only threshold, level 3 iff rho < theta_RX "
                "or the colour condition, else level 1",
        "tools_sha256": source_shas(),
        "runtime_s": {"load": round(t_load, 1), "total": round(time.time() - t0, 1)},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "guard_calibration.json")
    write_json(path, out)
    print(json.dumps(cal["guard"], indent=1))
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate-guard", help="theta1 / theta2 of the guard from simulated fit pools")
    c.add_argument("--out-dir", default=os.path.join(REPO, "results", "prereg02", "player"))
    c.add_argument("--train-dir", default=os.path.join(REPO, "data", "training"))
    c.add_argument("--split-csv", default=os.path.join(REPO, "results", "split_v1", "players_split.csv"))
    c.add_argument("--local-dir", default=os.path.join(REPO, "data", "local"))
    c.add_argument("--smoke", action="store_true",
                   help="tiny pools for a code-path check; output dir must be outside the repo")
    args = ap.parse_args(argv)
    if args.cmd == "calibrate-guard":
        cmd_calibrate_guard(args)


if __name__ == "__main__":
    main()
