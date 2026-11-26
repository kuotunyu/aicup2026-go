"""prereg36 explore (competition 1, v2 new pools): shared code of the both-colour per-move metric CNN (PLAN.md).

  * splits(): E (judge), D (dev), T (CNN training) players - PLAN.md 1
  * fast board replay -> the tutorial's 17 planes (third_party/aicup2026_tutorial/utils.py
    SGFParsePlayerIdentification.extract_features, value for value; test_p36.py compares them)
  * 8 board symmetries
  * P36Net (ResNet 10 x 128 -> global average -> Linear 128 -> 256), segment mean pooling, cosine-softmax head
  * the move cache of the fit games (data/prereg36/cache, built from tools/c10data.load_rows(part="fit",
    local_dir=None)): every worker replays games from compact move codes instead of parsing SGF text

    PY results/prereg36/explore/p36_lib.py cache          # build data/prereg36/cache (CPU, a few minutes)
    PY results/prereg36/explore/p36_lib.py split-record   # -> results/prereg36/explore/split_record.json
    PY results/prereg36/explore/p36_lib.py replay-speed   # games/s of the replay, one process

Only fit players. The validation answers are never opened (load_rows is called with local_dir=None).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
P35 = os.path.join(REPO, "results", "prereg35", "explore")
for _p in (TOOLS, P35):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from c10data import LEVELS, load_rows  # noqa: E402

TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
FEATS = os.path.join(REPO, "data", "prereg06", "feats")
DATA = os.path.join(REPO, "data", "prereg36")
CACHE = os.path.join(DATA, "cache")
CKPT = os.path.join(DATA, "ckpt")
EMB = os.path.join(DATA, "emb")

E_PER_LEVEL = 60
D_PER_LEVEL = 30
T_MIN_GAMES = 20
HIST = 8
N_PLANES = 2 * HIST + 1
SEG_MOVES = 24
BATCH_GAMES = 96
EMB_DIM = 256
CHANNELS = 128
BLOCKS = 10
SCALE = 30.0


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def git_head():
    return git("rev-parse", "HEAD").stdout.strip()


def git_committed(path):
    """True if the file is tracked and identical to HEAD."""
    rel = os.path.relpath(os.path.abspath(path), REPO).replace("\\", "/")
    if git("ls-files", "--error-unmatch", "--", rel).returncode != 0:
        return False
    return git("status", "--porcelain", "--", rel).stdout.strip() == ""


# ================================================================ players (PLAN.md 1)
def _h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_players(store_level, fit_games):
    """store_level: {pid: level} of the feature-store players (fit players with >= 105 games);
    fit_games: {pid: number of fit games, both colours}. Returns E, D, T (sorted lists) and by-level counts.
    E = per level the first 120 by sha256('p36|E|pid'); D = per level the first 30 of the rest by
    sha256('p36|D|pid') (all of them when a level has fewer than 30 left); T = every fit player outside E and D
    with >= 20 games."""
    E, D, per = [], [], {}
    for lv in LEVELS:
        pl = [p for p, v in store_level.items() if v == lv]
        e = sorted(pl, key=lambda p: _h("p36|E|%s" % p))[:E_PER_LEVEL]
        es = set(e)
        d = sorted((p for p in pl if p not in es), key=lambda p: _h("p36|D|%s" % p))[:D_PER_LEVEL]
        E += e
        D += d
        per[lv] = {"store": len(pl), "E": len(e), "D": len(d)}
    held = set(E) | set(D)
    T = sorted(p for p, n in fit_games.items() if p not in held and n >= T_MIN_GAMES)
    return sorted(E), sorted(D), T, per


def store_levels(feat_dir=FEATS):
    with np.load(os.path.join(feat_dir, "meta.npz"), allow_pickle=False) as z:
        pid, rk = z["player_id"], z["rank"]
    first = np.unique(pid, return_index=True)[1]
    return {str(pid[i]): str(rk[i]) for i in first.tolist()}   # the level of a player's first store row (= p35)


def splits(cache=None, feat_dir=FEATS):
    """E, D, T, per-level counts; game counts from the move cache (= the fit rows of c10data.load_rows)."""
    cache = cache or Cache()
    u, n = np.unique(cache.pid, return_counts=True)
    return split_players(store_levels(feat_dir), dict(zip(u.tolist(), n.tolist())))


def split_record(path=None, cache=None):
    cache = cache or Cache()
    E, D, T, per = splits(cache)
    u, n = np.unique(cache.pid, return_counts=True)
    cnt = dict(zip(u.tolist(), n.tolist()))
    assert not (set(E) & set(D)) and not (set(T) & (set(E) | set(D)))
    rec = {"format": "c10-prereg36 split v1",
           "rule": "E: store players per level, first %d by sha256('p36|E|pid'); D: of the rest, first %d by "
                   "sha256('p36|D|pid'); T: fit players outside E and D with >= %d games"
                   % (E_PER_LEVEL, D_PER_LEVEL, T_MIN_GAMES),
           "fit": {"players": int(len(u)), "games": int(n.sum())},
           "store_players": int(sum(v["store"] for v in per.values())), "by_level": per,
           "E": {"players": len(E), "games": int(sum(cnt[p] for p in E)), "sha256_sorted_ids": sha256_ids(E)},
           "D": {"players": len(D), "games": int(sum(cnt[p] for p in D)), "sha256_sorted_ids": sha256_ids(D)},
           "T": {"players": len(T), "games": int(sum(cnt[p] for p in T)), "sha256_sorted_ids": sha256_ids(T)},
           "levels_with_fewer_D_than_planned": {lv: v["D"] for lv, v in per.items() if v["D"] < D_PER_LEVEL},
           "cache": {k: v for k, v in cache.meta.items() if k != "seconds"}}
    if path:
        write_json(path, rec)
    return rec


# ================================================================ SGF -> move codes
# code = colour * 361 + row * 19 + col; colour 0 = black, 1 = white; (row, col) as sgfmill (row 0 = bottom line).
_STRICT = re.compile(r"\(;(?:[A-Z]+\[[^\]\\]*\])*((?:;[BW]\[[a-t]{0,2}\])*)\)\s*")
_MOVE = re.compile(r";([BW])\[([a-t]{0,2})\]")
_A = ord("a")


def moves_of(sgf):
    """Non-pass moves of the main line after the root node, as uint16 codes - the moves the tutorial's
    extract_features replays (root-node moves and passes are skipped there). Plain 'one move per node' SGF is
    parsed with a regular expression; anything else goes through sgfmill exactly as the tutorial does."""
    m = _STRICT.fullmatch(sgf)
    if m is not None:
        out = []
        for c, xy in _MOVE.findall(m.group(1)):
            if xy == "" or xy == "tt":
                continue
            if len(xy) != 2 or "t" in xy:
                out = None
                break
            out.append((361 if c == "W" else 0) + (18 - (ord(xy[1]) - _A)) * 19 + (ord(xy[0]) - _A))
        if out is not None:
            return np.asarray(out, np.uint16)
    from sgfmill import sgf as _sgf
    game = _sgf.Sgf_game.from_string(sgf)
    out = []
    for node in game.get_main_sequence()[1:]:
        color, move = node.get_move()
        if move is None:
            continue
        out.append((361 if color == "w" else 0) + move[0] * 19 + move[1])
    return np.asarray(out, np.uint16)


# ================================================================ board replay
_W = 21
_NB = [None] * (_W * _W)
_TO21 = [0] * 361
_TO19 = [0] * (_W * _W)
for _r in range(19):
    for _c in range(19):
        _i = (_r + 1) * _W + (_c + 1)
        _NB[_i] = (_i - 1, _i + 1, _i - _W, _i + _W)
        _TO21[_r * 19 + _c] = _i
        _TO19[_i] = _r * 19 + _c
_EMPTY21 = [3] * (_W * _W)
for _i in _TO21:
    _EMPTY21[_i] = 0


def _dead(board, start):
    """The group of `start` if it has no liberty, else None."""
    c = board[start]
    seen = {start}
    stack = [start]
    while stack:
        p = stack.pop()
        for q in _NB[p]:
            v = board[q]
            if v == 0:
                return None
            if v == c and q not in seen:
                seen.add(q)
                stack.append(q)
    return seen


def replay(codes, n=None):
    """Board before every move: int8 (n + 1, 361), +1 black, -1 white; row t = the position before move t
    (row n = after the last replayed move). Rules of sgfmill.boards.Board.play: opponent groups without liberties
    are removed; if none was, the played stone's own group is removed when it has no liberty; playing on an
    occupied point raises ValueError."""
    codes = np.asarray(codes)
    n = len(codes) if n is None else min(int(n), len(codes))
    board = list(_EMPTY21)
    ct, cp, cv = [], [], []
    lst = codes[:n].tolist()
    for t in range(n):
        code = lst[t]
        if code >= 361:
            c, o, p19 = 2, 1, code - 361
        else:
            c, o, p19 = 1, 2, code
        p = _TO21[p19]
        if board[p]:
            raise ValueError("move %d on an occupied point" % t)
        board[p] = c
        ct.append(t)
        cp.append(p19)
        cv.append(1 if c == 1 else -1)
        captured = False
        for q in _NB[p]:
            if board[q] == o:
                g = _dead(board, q)
                if g:
                    captured = True
                    ov = -1 if o == 1 else 1
                    for s in g:
                        board[s] = 0
                        ct.append(t)
                        cp.append(_TO19[s])
                        cv.append(ov)
        if not captured:
            g = _dead(board, p)
            if g:
                ov = -1 if c == 1 else 1
                for s in g:
                    board[s] = 0
                    ct.append(t)
                    cp.append(_TO19[s])
                    cv.append(ov)
    D = np.zeros((n + 1, 361), np.int8)
    if ct:
        np.add.at(D, (np.asarray(ct, np.int64) + 1, np.asarray(cp, np.int64)), np.asarray(cv, np.int8))
    return np.cumsum(D, axis=0, dtype=np.int8)


def target_moves(codes, target_color):
    """Indices (among the non-pass moves) of the target's moves that give a sample: the tutorial appends the
    position before every non-pass move to an 8-long history and emits a sample at the target's moves once the
    history is full, i.e. from the 8th non-pass move of the game on."""
    col = np.asarray(codes) >= 361
    want = target_color.lower() == "w"
    t = np.flatnonzero(col == want)
    return t[t >= HIST - 1]


def planes_at(S, t, target_color):
    """17 planes at the target-move indices t from the replay S: planes 0-7 the target's stones in the positions
    before the last 8 non-pass moves (oldest first; the last one = the position the target is about to play in),
    planes 8-15 the opponent's, plane 16 all ones for black / zeros for white."""
    t = np.asarray(t, np.int64)
    black = target_color.lower() == "b"
    H = S[t[:, None] + np.arange(-(HIST - 1), 1)[None, :]]          # (m, 8, 361)
    out = np.empty((len(t), N_PLANES, 361), np.uint8)
    v = 1 if black else -1
    out[:, :HIST] = H == v
    out[:, HIST:2 * HIST] = H == -v
    out[:, 2 * HIST] = 1 if black else 0
    return out.reshape(len(t), N_PLANES, 19, 19)


def planes_for_codes(codes, target_color, k=None, rng=None):
    """All samples of the game (k None), or a random subset of at most k target moves (sorted; the game is only
    replayed up to the last chosen move and only the chosen planes are built). Returns (planes, move indices)."""
    t = target_moves(codes, target_color)
    if k is not None and len(t) > k:
        t = np.sort(rng.choice(t, size=k, replace=False))
    if len(t) == 0:
        return np.zeros((0, N_PLANES, 19, 19), np.uint8), t
    S = replay(codes, int(t[-1]))
    return planes_at(S, t, target_color), t


def planes_for_game(sgf, target_color):
    """uint8 (n_target_moves, 17, 19, 19), value for value what the tutorial's
    SGFParsePlayerIdentification().extract_features(sgf, target_color.lower(), mode='inference') returns."""
    return planes_for_codes(moves_of(sgf), target_color)[0]


def planes_for_game_subset(sgf, target_color, k, rng):
    return planes_for_codes(moves_of(sgf), target_color, k, rng)


# ================================================================ symmetries
def sym(x, s):
    """Board symmetry s in 0..7 on the last two axes: (s >= 4: left-right flip first), then s % 4 quarter turns."""
    if s >= 4:
        x = x[..., ::-1]
    if s % 4:
        x = np.rot90(x, s % 4, axes=(-2, -1))
    return x


# ================================================================ move cache of the fit games
class Cache:
    """data/prereg36/cache: codes.npy (uint16, all games concatenated), off.npy (int64, n + 1), meta.npz
    (player_id, game_id, rank, color, bad), cache.json. Rows in load_rows order (fit)."""

    def __init__(self, d=CACHE, mmap=True):
        self.dir = d
        with open(os.path.join(d, "cache.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.codes = np.load(os.path.join(d, "codes.npy"), mmap_mode="r" if mmap else None)
        self.off = np.load(os.path.join(d, "off.npy"))
        with np.load(os.path.join(d, "meta.npz"), allow_pickle=False) as z:
            self.pid, self.gid, self.rank, self.color, self.bad = (z[k] for k in
                                                                   ("player_id", "game_id", "rank", "color", "bad"))
        self.n = len(self.pid)

    def game(self, i):
        return np.asarray(self.codes[self.off[i]:self.off[i + 1]])

    def key_index(self):
        return {(str(a), str(b), str(c)): i for i, (a, b, c) in
                enumerate(zip(self.rank.tolist(), self.gid.tolist(), self.pid.tolist()))}


def build_cache(d=CACHE, train_dir=TRAIN_DIR, split_csv=SPLIT_CSV):
    t0 = time.time()
    rows = load_rows(train_dir, split_csv, part="fit", local_dir=None)
    os.makedirs(d, exist_ok=True)
    parts, off, bad = [], [0], []
    for r in rows:
        try:
            c = moves_of(r["sgf_content"])
            b = False
        except Exception:
            c, b = np.zeros(0, np.uint16), True
        parts.append(c)
        bad.append(b)
        off.append(off[-1] + len(c))
        r["sgf_content"] = None
    codes = np.concatenate(parts) if parts else np.zeros(0, np.uint16)
    del parts
    np.save(os.path.join(d, "codes.npy"), codes)
    np.save(os.path.join(d, "off.npy"), np.asarray(off, np.int64))
    np.savez(os.path.join(d, "meta.npz"), player_id=np.array([r["player_id"] for r in rows]),
             game_id=np.array([r["game_id"] for r in rows]), rank=np.array([r["rank"] for r in rows]),
             color=np.array([r["color"] for r in rows]), bad=np.asarray(bad, bool))
    meta = {"format": "c10-prereg36 cache v1", "source": "c10data.load_rows(part='fit', local_dir=None)",
            "games": len(rows), "players": len({r["player_id"] for r in rows}), "moves": int(len(codes)),
            "unparsed_games": int(sum(bad)), "codes_sha256": sha256_file(os.path.join(d, "codes.npy")),
            "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(d, "cache.json"), meta)
    return meta


# ================================================================ model
def build_model(n_classes=None):
    """Returns (P36Net, CosHead or None). torch is imported here so that the replay workers and the CPU tools do
    not need it."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualBlock(nn.Module):        # = third_party/aicup2026_tutorial/network.py
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + x)

    class P36Net(nn.Module):
        """(P, 17, 19, 19) -> (P, 256) per-move embedding (not normalised)."""

        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv2d(N_PLANES, CHANNELS, 3, padding=1)
            self.bn_in = nn.BatchNorm2d(CHANNELS)
            self.layers = nn.Sequential(*[ResidualBlock(CHANNELS) for _ in range(BLOCKS)])
            self.fc = nn.Linear(CHANNELS, EMB_DIM)

        def forward(self, x):
            out = F.relu(self.bn_in(self.conv_in(x)))
            out = self.layers(out)
            return self.fc(out.mean(dim=(2, 3)))

    class CosHead(nn.Module):
        """Cosine softmax: logits = scale * cos(embedding, class weight); no margin."""

        def __init__(self, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, EMB_DIM) * 0.01)

        def forward(self, emb_normalised):
            return SCALE * F.linear(emb_normalised, F.normalize(self.weight.float(), dim=1))

    return P36Net(), (CosHead(n_classes) if n_classes else None)


def segment_mean(e, seg_index, n_seg):
    """Mean of the rows of e per segment (float32); segments without rows give zeros."""
    import torch
    e = e.float()
    s = torch.zeros(n_seg, e.shape[1], dtype=e.dtype, device=e.device).index_add_(0, seg_index, e)
    c = torch.zeros(n_seg, dtype=e.dtype, device=e.device).index_add_(
        0, seg_index, torch.ones(len(seg_index), dtype=e.dtype, device=e.device))
    return s / c.clamp(min=1)[:, None]


def embed_segments(net, x, seg_index, n_seg):
    """Segment vector = mean of the per-move embeddings of the segment, then L2-normalised."""
    import torch.nn.functional as F
    return F.normalize(segment_mean(net(x), seg_index, n_seg), dim=1)


# ================================================================ official pools
def pool_items(cands, qs):
    """The pool's games in the order of p35_apply.cmd_apply: candidate rows in file order, then every question's
    games. Returns items [(sgf, colour)], cand_ids (sorted), cand_rows (by sorted id), q_rows, question ids."""
    cand_ids = sorted({r["player_id"] for r in cands})
    items, cand_rows = [], {c: [] for c in cand_ids}
    for r in cands:
        cand_rows[r["player_id"]].append(len(items))
        items.append((r["sgf_content"], r["color"]))
    q_rows, qids = [], []
    for q in qs:
        g = []
        for i in range(1, int(q["num_games"]) + 1):
            g.append(len(items))
            items.append((q["sgf_%d" % i], q["color_%d" % i]))
        q_rows.append(g)
        qids.append(q["question_id"])
    return items, cand_ids, [cand_rows[c] for c in cand_ids], q_rows, qids


# ================================================================ command line
def _replay_speed(a):
    cache = Cache()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, cache.n, a.games)
    games = [(cache.game(int(i)), str(cache.color[i])) for i in idx]
    out = {}
    t = time.perf_counter()
    n = sum(len(planes_for_codes(c, col)[0]) for c, col in games)
    out["all_moves"] = {"games_per_s": round(a.games / (time.perf_counter() - t), 1), "positions": int(n)}
    t = time.perf_counter()
    n = sum(len(planes_for_codes(c, col, SEG_MOVES, rng)[0]) for c, col in games)
    out["subset_%d" % SEG_MOVES] = {"games_per_s": round(a.games / (time.perf_counter() - t), 1), "positions": int(n)}
    t = time.perf_counter()
    for c, _col in games:
        replay(c)
    out["replay_only"] = {"games_per_s": round(a.games / (time.perf_counter() - t), 1)}
    out["mean_moves"] = float(np.mean([len(c) for c, _ in games]))
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    s = sub.add_parser("split-record")
    s.add_argument("--out", default=os.path.join(HERE, "split_record.json"))
    r = sub.add_parser("replay-speed")
    r.add_argument("--games", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "cache":
        print(json.dumps(build_cache(), indent=1))
        return 0
    if a.cmd == "split-record":
        rec = split_record(a.out)
        print(json.dumps({k: rec[k] for k in ("fit", "by_level", "E", "D", "T")}, indent=1))
        return 0
    return _replay_speed(a)


if __name__ == "__main__":
    sys.exit(main())
