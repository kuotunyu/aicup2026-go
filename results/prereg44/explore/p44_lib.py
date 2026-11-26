"""prereg44 explore (competition 2): the rank-direct per-move CNN CNN44 (PLAN.md 3). Shared code.

  * guard(): refuses the validation answers and every competition-1 data folder (PLAN.md 1, 9)
  * roles(): inner / A / H players exactly as S25 (prereg25's salt; PLAN.md 2) - local only (imports p25_lib lazily)
  * stream_rows() / build_cache(): move caches of official training games (prereg36's cache format, p36_lib.moves_of)
  * RankBatches: level-uniform -> player-uniform -> game-uniform sampler, <= 24 target moves, one random symmetry
  * build(): GPSENet trunk (copied from prereg42 V2) -> mean of the per-move vectors -> Linear(256 -> 10)
  * Ema (copied from prereg42), InferChunks / infer_logits (all target moves of a game, identity symmetry)

Competition-1 *code* is imported (results/prereg36/explore/p36_lib.py: board replay, the official 17 planes, the 8
symmetries, segment_mean); no competition-1 weight, vector, cache or feature is ever read (guard()).
torch is imported inside the functions, so the replay workers and the CPU tools do not need it at import time.
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
P36 = os.path.join(REPO, "results", "prereg36", "explore")
for _p in (TOOLS, P36):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import c10data  # noqa: E402
import p36_lib as B  # noqa: E402  (competition-1 code only: replay, planes, symmetries, segment_mean)

LEVELS = list(c10data.LEVELS)
L = len(LEVELS)
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
INNER_KEYS = os.path.join(REPO, "results", "prereg02", "rank", "dumps", "inner_keys.npz")
S25_TRAIN_RECORD = os.path.join(REPO, "results", "prereg25", "explore", "train_record.json")
DATA = os.path.join(REPO, "data", "prereg44")
CACHE = os.path.join(DATA, "cache")
RUN = os.path.join(DATA, "run")
LOGITS = os.path.join(DATA, "logits")

# ---------------------------------------------------------------- PLAN.md 3 (frozen settings)
SEED = 44
BATCH_GAMES = 96
SEG_MOVES = 24
MIN_PLAYER_GAMES = 5
TOTAL_STEPS = 200000
CKPT_EVERY = 20000
LR, WD, PCT = 2e-3, 1e-4, 0.05
EMA_DECAY = 0.999
CANDIDATE_STEPS = (140000, 160000, 180000, 200000)
CANDIDATE_WEIGHTS = ("raw", "ema")
# GPSENet (prereg42 V2, p42_lib.py at commit 249d1f5)
CHANNELS = 128
BLOCKS = 10
EMB_DIM = 256
SE_RATIO = 8
GPOOL_BLOCKS = (2, 5, 8)
GPOOL_CH = 32
P42_SOURCE = "results/prereg42/explore/p42_lib.py @ 249d1f5 (branch p42-recipe): SE, GPSEBlock, GPSENet, Ema"

FORBIDDEN = ("/data/local", "rank_val_answers", "player_val_answers", "/data/prereg06",
             "/data/prereg35", "/data/prereg36", "/data/prereg37", "/data/prereg38", "/data/prereg39",
             "/data/prereg40/", "/data/prereg41", "/data/prereg42", "/data/prereg43")


def guard(p):
    """Refuse the validation answers and every competition-1 data folder (and prereg40's competition-2 S40 data, which
    this plan does not use either)."""
    q = os.path.abspath(os.fspath(p)).replace("\\", "/").lower()
    if any(f in q + ("/" if os.path.isdir(p) else "") for f in FORBIDDEN):
        raise SystemExit(f"prereg44 guard: refused path {p}")
    return p


def sha256_file(p):
    h = hashlib.sha256()
    with open(guard(p), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(p):
    with open(guard(p), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(guard(tmp), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(guard(path), encoding="utf-8") as f:
        return json.load(f)


def key(rank, pid):
    return f"{rank}:{pid}"


# ---------------------------------------------------------------- PLAN.md 2: player roles (as S25)
def roles(split_csv=SPLIT_CSV):
    """-> dict: inner / A / H as sets of '<rank>:<player_id>' and the per-level counts. A / H = fit + validation
    players split by prereg25's salt (p25_lib.holdout_bit); inner = c10data.players_for(split, 'inner')."""
    p25 = os.path.join(REPO, "results", "prereg25", "explore")
    if p25 not in sys.path:
        sys.path.insert(0, p25)
    import p25_lib
    split = c10data.read_split(guard(split_csv))
    fit = c10data.players_for(split, "fit")
    inner = c10data.players_for(split, "inner")
    val = p25_lib.val_players(split_csv)
    pool = {key(split[p]["rank"], p) for p in fit} | {key(r, p) for p, r in val.items()}
    A = {k for k in pool if p25_lib.holdout_bit(k) == 0}
    H = pool - A
    I = {key(split[p]["rank"], p) for p in inner}

    def per_level(s):
        c = [0] * L
        for k in s:
            c[LEVELS.index(k.split(":", 1)[0])] += 1
        return c

    ids = lambda s: {k.split(":", 1)[1] for k in s}  # noqa: E731
    return {"A": A, "H": H, "inner": I, "fit_players": len(fit), "val_players": len(val),
            "A_per_level": per_level(A), "H_per_level": per_level(H),
            "overlap": {"A_H": len(A & H), "AH_inner_keys": len((A | H) & I),
                        "AH_inner_player_id": len(ids(A | H) & ids(I))}}


def seam_d(rl, s25_record=S25_TRAIN_RECORD):
    """PLAN.md 2 seam D: A / H counts equal S25's train_record.json, no overlap."""
    p = read_json(s25_record)["players"]
    out = {"A_players": [len(rl["A"]), p["A_players"]], "H_players": [len(rl["H"]), p["H_players"]],
           "A_per_level": [rl["A_per_level"], p["A_players_per_level"]],
           "H_per_level": [rl["H_per_level"], p["H_players_per_level"]], "overlap": rl["overlap"]}
    out["ok"] = bool(all(a == b for a, b in (out["A_players"], out["H_players"], out["A_per_level"],
                                             out["H_per_level"]))
                     and not any(rl["overlap"].values()))
    return out


# ---------------------------------------------------------------- caches (PLAN.md 3.1)
def stream_rows(want_player_ids, train_dir=TRAIN_DIR):
    """Rows (dict of c10data.COLUMNS) of the official training CSVs whose player_id is in want_player_ids, file
    order (D .. 6D); the row's rank must be the file's level. One row at a time (no SGF list in memory)."""
    csv.field_size_limit(10 ** 8)
    for lv in LEVELS:
        with open(guard(os.path.join(train_dir, f"train_{lv}.csv")), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if next(reader) != c10data.COLUMNS:
                raise ValueError(f"unexpected header in train_{lv}.csv")
            for rec in reader:
                if rec[0] in want_player_ids:
                    r = dict(zip(c10data.COLUMNS, rec))
                    if r["rank"] != lv:
                        raise ValueError(f"row {r['game_id']} of train_{lv}.csv has rank {r['rank']}")
                    yield r


def build_cache(rows, d, label, source):
    """prereg36's cache format at d from an iterable of rows: codes.npy (uint16), off.npy, meta.npz (player_id,
    game_id, rank, color, bad), cache.json."""
    t0 = time.time()
    os.makedirs(guard(d), exist_ok=True)
    parts, off, bad, pid, gid, rk, col = [], [0], [], [], [], [], []
    for r in rows:
        try:
            c = B.moves_of(r["sgf_content"])
            b = False
        except Exception:
            c, b = np.zeros(0, np.uint16), True
        parts.append(c)
        bad.append(b)
        off.append(off[-1] + len(c))
        pid.append(r["player_id"])
        gid.append(r["game_id"])
        rk.append(r["rank"])
        col.append(r["color"])
    codes = np.concatenate(parts) if parts else np.zeros(0, np.uint16)
    del parts
    np.save(os.path.join(d, "codes.npy"), codes)
    np.save(os.path.join(d, "off.npy"), np.asarray(off, np.int64))
    np.savez(os.path.join(d, "meta.npz"), player_id=np.array(pid), game_id=np.array(gid), rank=np.array(rk),
             color=np.array(col), bad=np.asarray(bad, bool))
    meta = {"format": "c10-prereg44 cache v1", "label": label, "source": source, "games": len(pid),
            "players": len(set(zip(rk, pid))), "moves": int(len(codes)), "unparsed_games": int(sum(bad)),
            "games_per_level": [int(sum(1 for x in rk if x == lv)) for lv in LEVELS],
            "codes_sha256": sha256_file(os.path.join(d, "codes.npy")),
            "off_sha256": sha256_file(os.path.join(d, "off.npy")),
            "meta_sha256": sha256_file(os.path.join(d, "meta.npz")),
            "bytes": int(sum(os.path.getsize(os.path.join(d, f)) for f in ("codes.npy", "off.npy", "meta.npz"))),
            "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(d, "cache.json"), meta)
    return meta


def open_cache(d):
    return B.Cache(guard(d))


def cache_keys(c):
    return [(str(r), str(g), str(p), str(co)) for r, g, p, co in
            zip(c.rank.tolist(), c.gid.tolist(), c.pid.tolist(), c.color.tolist())]


# ---------------------------------------------------------------- sampler (PLAN.md 3.3)
def level_index(cache):
    """Players of every level with >= MIN_PLAYER_GAMES parsed games: (games grouped by player, player offsets,
    level of every player, players per level). Deterministic order (sorted keys)."""
    lab = np.array([LEVELS.index(r) for r in cache.rank.tolist()], np.int64)
    keys = np.array([key(r, p) for r, p in zip(cache.rank.tolist(), cache.pid.tolist())])
    ok = ~np.asarray(cache.bad, bool)
    uniq, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv[ok], minlength=len(uniq))
    keep = cnt >= MIN_PLAYER_GAMES
    order = np.flatnonzero(ok & keep[inv])
    order = order[np.argsort(inv[order], kind="stable")]
    pl = np.flatnonzero(keep)
    p_off = np.concatenate([[0], np.cumsum(cnt[pl])]).astype(np.int64)
    p_level = np.array([LEVELS.index(uniq[i].split(":", 1)[0]) for i in pl], np.int64)
    by_level = [np.flatnonzero(p_level == k) for k in range(L)]
    lab_ok = lab[order]
    assert (lab_ok == np.repeat(p_level, np.diff(p_off))).all()
    return {"games": order, "p_off": p_off, "p_level": p_level, "by_level": by_level,
            "players": [str(uniq[i]) for i in pl], "dropped_players": int((~keep).sum())}


class RankBatches:
    """Item s: x uint8 (P, 17, 19, 19), seg (P,), y (B,) level. Batch s is drawn from default_rng([SEED, s]) inside
    the worker (independent of the number of workers)."""

    def __init__(self, cache_dir, off, is_white, idx, seed=SEED, batch_games=BATCH_GAMES, moves=SEG_MOVES):
        self.cache_dir, self.off, self.is_white = cache_dir, off, is_white
        self.games, self.p_off, self.by_level = idx["games"], idx["p_off"], idx["by_level"]
        self.seed, self.batch_games, self.moves = seed, batch_games, moves
        self.codes = None
        if any(len(b) == 0 for b in self.by_level):
            raise SystemExit("a level without players")

    def __len__(self):
        return 1 << 62

    def _codes(self):
        if self.codes is None:
            self.codes = np.load(os.path.join(self.cache_dir, "codes.npy"), mmap_mode="r")
        return self.codes

    def batch(self, s):
        codes_all = self._codes()
        rng = np.random.default_rng([self.seed, int(s)])
        xs, segs, ys = [], [], []
        while len(ys) < self.batch_games:
            k = int(rng.integers(L))
            pl = self.by_level[k]
            p = int(pl[rng.integers(len(pl))])
            a, b = int(self.p_off[p]), int(self.p_off[p + 1])
            g = int(self.games[a + int(rng.integers(b - a))])
            codes = np.asarray(codes_all[self.off[g]:self.off[g + 1]])
            try:
                x, _t = B.planes_for_codes(codes, "w" if self.is_white[g] else "b", self.moves, rng)
            except ValueError:
                continue
            if len(x) == 0:
                continue
            xs.append(B.sym(x, int(rng.integers(8))))
            segs.append(np.full(len(x), len(ys), np.int64))
            ys.append(k)
        return {"x": np.ascontiguousarray(np.concatenate(xs)), "seg": np.concatenate(segs),
                "y": np.asarray(ys, np.int64)}

    def __getitem__(self, s):
        import torch
        return {k: torch.from_numpy(v) for k, v in self.batch(s).items()}


# ---------------------------------------------------------------- model (PLAN.md 3.2)
def build():
    """RankNet: GPSENet trunk (prereg42 V2, copied) -> per-move 256 -> segment mean -> Linear(256, 10).
    forward(x, seg, n_seg) -> (n_seg, 10) logits (float32). trunk(x) may run under fp16 autocast."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SE(nn.Module):
        def __init__(self, c, r=SE_RATIO):
            super().__init__()
            self.fc1 = nn.Linear(c, c // r)
            self.fc2 = nn.Linear(c // r, c)

        def forward(self, x):
            s = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=(2, 3))))))
            return x * s[:, :, None, None]

    class GPSEBlock(nn.Module):
        def __init__(self, c, gpool):
            super().__init__()
            self.conv1 = nn.Conv2d(c, c, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(c)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(c)
            self.se = SE(c)
            self.gpool = gpool
            if gpool:
                self.convg = nn.Conv2d(c, GPOOL_CH, 3, padding=1)
                self.bng = nn.BatchNorm2d(GPOOL_CH)
                self.lin = nn.Linear(2 * GPOOL_CH, c)

        def forward(self, x):
            out = self.bn1(self.conv1(x))
            if self.gpool:
                g = F.relu(self.bng(self.convg(x)))
                gp = torch.cat([g.mean(dim=(2, 3)), g.amax(dim=(2, 3))], 1)
                out = out + self.lin(gp)[:, :, None, None]
            out = self.bn2(self.conv2(F.relu(out)))
            return F.relu(self.se(out) + x)

    class GPSENet(nn.Module):
        def __init__(self):
            super().__init__()
            c = CHANNELS
            self.conv_in = nn.Conv2d(B.N_PLANES, c, 3, padding=1)
            self.bn_in = nn.BatchNorm2d(c)
            self.layers = nn.Sequential(*[GPSEBlock(c, i in GPOOL_BLOCKS) for i in range(BLOCKS)])
            self.fc = nn.Linear(2 * c, EMB_DIM)

        def forward(self, x):
            out = self.layers(F.relu(self.bn_in(self.conv_in(x))))
            return self.fc(torch.cat([out.mean(dim=(2, 3)), out.amax(dim=(2, 3))], 1))

    class RankNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = GPSENet()
            self.head = nn.Linear(EMB_DIM, L)

        def pool_head(self, e, seg, n_seg):
            return self.head(B.segment_mean(e, seg, n_seg).float())

        def forward(self, x, seg, n_seg):
            return self.pool_head(self.trunk(x), seg, n_seg)

    return RankNet()


# ---------------------------------------------------------------- EMA (prereg42, copied)
class Ema:
    """EMA of the floating parameters of a module (decay EMA_DECAY); buffers (BN statistics) copied."""

    def __init__(self, module, decay=EMA_DECAY):
        import torch
        self.decay = decay
        self.params = {k: v.detach().clone().float() for k, v in module.named_parameters()}
        self.buffers = {k: v.detach().clone() for k, v in module.named_buffers()}
        self._torch = torch

    def update(self, module):
        with self._torch.no_grad():
            for k, v in module.named_parameters():
                self.params[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            for k, v in module.named_buffers():
                self.buffers[k].copy_(v.detach())

    def state_dict(self):
        out = {k: v.clone() for k, v in self.params.items()}
        out.update({k: v.clone() for k, v in self.buffers.items()})
        return out

    def load_state_dict(self, sd):
        for k in self.params:
            self.params[k].copy_(sd[k])
        for k in self.buffers:
            self.buffers[k].copy_(sd[k])


# ---------------------------------------------------------------- inference (PLAN.md 3.4)
class InferChunks:
    """Item i: games [i * chunk, (i + 1) * chunk) of a cache with every target move (identity symmetry):
    x (P, 17, 19, 19), seg (P,) local game index, first (game index of local 0), n (games), bad (n,) bool."""

    def __init__(self, cache_dir, off, is_white, n_games, chunk=32):
        self.cache_dir, self.off, self.is_white, self.n_games, self.chunk = cache_dir, off, is_white, n_games, chunk
        self.codes = None

    def __len__(self):
        return (self.n_games + self.chunk - 1) // self.chunk

    def item(self, i):
        if self.codes is None:
            self.codes = np.load(os.path.join(self.cache_dir, "codes.npy"), mmap_mode="r")
        a, b = i * self.chunk, min((i + 1) * self.chunk, self.n_games)
        xs, segs, bad = [], [], np.zeros(b - a, bool)
        for j, g in enumerate(range(a, b)):
            codes = np.asarray(self.codes[self.off[g]:self.off[g + 1]])
            try:
                x, _t = B.planes_for_codes(codes, "w" if self.is_white[g] else "b")
            except ValueError:
                x = np.zeros((0, B.N_PLANES, 19, 19), np.uint8)
            if len(x) == 0:
                bad[j] = True
                continue
            xs.append(x)
            segs.append(np.full(len(x), j, np.int64))
        x = np.concatenate(xs) if xs else np.zeros((0, B.N_PLANES, 19, 19), np.uint8)
        seg = np.concatenate(segs) if segs else np.zeros(0, np.int64)
        return {"x": np.ascontiguousarray(x), "seg": seg, "first": np.int64(a), "n": np.int64(b - a), "bad": bad}

    def __getitem__(self, i):
        import torch
        return {k: torch.from_numpy(np.asarray(v)) for k, v in self.item(i).items()}


def infer_logits(net, cache, device, workers=0, chunk=32, max_positions=4096, log=None):
    """(n_games, 10) float32 logits of every game of the cache (0 rows and bad = True for games without a target
    move), positions (n_games,) int64. net in eval mode; fp16 autocast on cuda."""
    import torch
    from torch.utils.data import DataLoader
    cuda = str(device).startswith("cuda")
    ds = InferChunks(cache.dir, cache.off, cache.color == "W", cache.n, chunk)
    dl = DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=cuda,
                    prefetch_factor=4 if workers else None)
    out = np.zeros((cache.n, L), np.float32)
    pos = np.zeros(cache.n, np.int64)
    bad = np.zeros(cache.n, bool)
    t0 = time.time()
    net.eval()
    with torch.no_grad():
        for it, b in enumerate(dl):
            a, n = int(b["first"]), int(b["n"])
            bad[a:a + n] = b["bad"].numpy()
            x, seg = b["x"], b["seg"]
            if len(seg):
                x = x.to(device, non_blocking=True)
                seg = seg.to(device, non_blocking=True)
                es = []
                for s in range(0, len(seg), max_positions):
                    with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                        es.append(net.trunk(x[s:s + max_positions].float()).float())
                lg = net.pool_head(torch.cat(es), seg, n)
                out[a:a + n] = lg.float().cpu().numpy()
                pos[a:a + n] = np.bincount(seg.cpu().numpy(), minlength=n)
            if log and it % 200 == 0:
                log(f"infer {a + n}/{cache.n} games, {time.time() - t0:.0f}s")
    out[bad] = 0.0
    return out, pos, bad


def log_softmax(z):
    z = np.asarray(z, np.float64)
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


GATE_0B = -0.010
REF_Q25 = 0.7456519871546352
REF_Q25_MONO = 0.7482450052195216


# ---------------------------------------------------------------- stage 0b stacking (PLAN.md 5), numpy / scipy
def stack_logits(theta, Fs):
    """z = sum_j theta_j F_j + b (b_0 = 0); theta = [w_1..w_J, b_1..b_9]."""
    J = len(Fs)
    z = sum(theta[j] * Fs[j] for j in range(J))
    b = np.concatenate([[0.0], theta[J:]])
    return z + b[None, :]


def stack_objective(theta, Fs, y, w):
    z = stack_logits(theta, Fs)
    lp = log_softmax(z)
    P = np.exp(lp)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    loss = float(-(w * lp[np.arange(len(y)), y]).sum())
    G = w[:, None] * (P - Y)
    g = np.array([float((G * F).sum()) for F in Fs] + list(G.sum(0)[1:]))
    return loss, g


def stack_weights(y):
    """Every level the same total weight (sum 1)."""
    cnt = np.bincount(y, minlength=L).astype(np.float64)
    present = (cnt > 0).sum()
    return 1.0 / (present * cnt[y])


def fit_stack(Fs, y, start=None):
    from scipy.optimize import minimize
    J = len(Fs)
    th0 = np.zeros(J + L - 1) if start is None else np.asarray(start, np.float64)
    if start is None:
        th0[0] = 1.0
    w = stack_weights(y)
    r = minimize(stack_objective, th0, args=(Fs, y, w), jac=True, method="L-BFGS-B",
                 options={"maxiter": 1000})
    return r.x, float(r.fun), bool(r.success)


def crossfit_stack(Fs, y, fold):
    """2-fold player cross-fit: fold f's parameters fitted on the other fold -> (lp [Q, 10], params per fold)."""
    lp = np.full((len(y), L), np.nan)
    par = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        th, loss, ok = fit_stack([F[tr] for F in Fs], y[tr])
        lp[te] = log_softmax(stack_logits(th, [F[te] for F in Fs]))
        par.append({"fold": f, "theta": th.tolist(), "fit_loss": loss, "converged": ok, "n_fit": int(tr.sum())})
    return lp, par


def player_fold(keys):
    return np.array([int(hashlib.sha256(f"c10-prereg44|0b|{k}".encode("utf-8")).hexdigest()[:8], 16) % 2
                     for k in keys], np.int64)


def mean_nll(lp, y):
    return float(-lp[np.arange(len(y)), y].mean())


def balanced_game_nll(lp, y):
    """PLAN.md 4: mean over levels of the mean per-game -log p(true level)."""
    v = -lp[np.arange(len(y)), y]
    return float(np.mean([v[y == k].mean() for k in range(L) if (y == k).any()]))
