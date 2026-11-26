"""prereg42 explore (competition 1, v2 new pools): the training-recipe variants of the prereg36 metric CNN (PLAN.md 2).

  * VARIANTS: V1 episode (exam-shaped sets + prototypical loss), V2 SE + global-pooling-bias trunk, V3 move-order
    transformer pooling, V4 sub-center cosine-margin head. Everything else is prereg36's (p36_lib, imported, not edited).
    V3b (PLAN.md revision 3, added after V3 failed by attention collapse) = transformer pooling residual over the
    mean (no CLS, LayerNorm'd tokens, zero-initialised readout, pooling lr x0.1).
  * GameBatches: the prereg36 sampler (one game, <= 24 moves) - for V2 / V4 it returns exactly p36_train.TrainSet's
    batches (same random stream; test_p42.py checks it) plus the move numbers that V3 needs.
  * EpisodeBatches: V1's sampler (32 players x two disjoint sets of k in {1, 2, 3, 5} games, ~36 moves per set).
  * build(spec, n_classes): EmbedNet (trunk -> per-move vectors -> pooling -> one vector per game) and the head.
  * Ema: exponential moving average of the EmbedNet parameters (buffers copied).
  * sym_t: the 8 board symmetries of p36_lib.sym on torch tensors (TTA).

torch is imported inside the functions, so the replay workers and the CPU tools do not need it at import time.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P36 = os.path.join(REPO, "results", "prereg36", "explore")
P38 = os.path.join(REPO, "results", "prereg38", "explore")
for _p in (P36, P38):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import p36_lib as L  # noqa: E402

DATA = os.path.join(REPO, "data", "prereg42")
EMB = os.path.join(DATA, "emb")

TOTAL_STEPS = 40000
CKPT_EVERY = 8000
LR, WD, PCT, SEED = 2e-3, 1e-4, 0.05, 0
EMA_DECAY = 0.999
# V1 (episode)
EP_PLAYERS = 32
EP_SET_MOVES = 36
EP_KS = (1, 2, 3, 5)
PROTO_SCALE = 20.0
# V2 (trunk)
SE_RATIO = 8
GPOOL_BLOCKS = (2, 5, 8)
GPOOL_CH = 32
# V3 (pooling)
TX_LAYERS, TX_HEADS, TX_FF, TX_DROPOUT = 2, 4, 512, 0.1
POS_BUCKET, POS_BUCKETS = 6, 64
CHUNK_MOVES = 24
# V3b (PLAN.md revision 3): residual over mean pooling
TXRES_POS_STD = 0.02
POOL_LR_MULT = {"txres": 0.1}
TX_POOLS = ("tx", "txres")
# V4 (head)
SUBCENTERS = 2
MARGIN = 0.1
MARGIN_RAMP = 4000

VARIANTS = {
    "V1": {"name": "episode", "sampler": "episode", "trunk": "p36", "pool": "mean", "head": "cos"},
    "V2": {"name": "se_gpool_trunk", "sampler": "game", "trunk": "gpse", "pool": "mean", "head": "cos"},
    "V3": {"name": "transformer_pooling", "sampler": "game", "trunk": "p36", "pool": "tx", "head": "cos"},
    "V3b": {"name": "transformer_residual_pooling", "sampler": "game", "trunk": "p36", "pool": "txres", "head": "cos"},
    "V4": {"name": "subcenter_margin_head", "sampler": "game", "trunk": "p36", "pool": "mean", "head": "subcenter"},
}
# prereg36 / prereg37 checkpoints (the base system's CNN) are embedded with this spec
P36_SPEC = {"name": "p36", "sampler": "game", "trunk": "p36", "pool": "mean", "head": "cos"}


def is_tx(spec):
    """V3 / V3b: a transformer pooling that takes move numbers and has the full / chunk24 embed modes."""
    return spec["pool"] in TX_POOLS


def spec_of(variant):
    if variant in VARIANTS:
        return dict(VARIANTS[variant])
    if variant == "p36":
        return dict(P36_SPEC)
    raise SystemExit("unknown variant %r" % variant)


# ================================================================ samplers
class GameBatches:
    """Item s = batch s of prereg36's sampler: x uint8 (P, 17, 19, 19), segment (P,), labels (B,), move number of
    every position (P,). The draws are exactly those of p36_train.TrainSet.batch (same rng, same order)."""

    def __init__(self, cache_dir, off, is_white, t_games, t_off, seed=SEED, batch_games=L.BATCH_GAMES,
                 moves=L.SEG_MOVES):
        self.cache_dir, self.off, self.is_white = cache_dir, off, is_white
        self.t_games, self.t_off = t_games, t_off
        self.seed, self.batch_games, self.moves = seed, batch_games, moves
        self.codes = None

    def __len__(self):
        return 1 << 62

    def _codes(self):
        if self.codes is None:
            self.codes = np.load(os.path.join(self.cache_dir, "codes.npy"), mmap_mode="r")
        return self.codes

    def batch(self, s):
        codes_all = self._codes()
        rng = np.random.default_rng([self.seed, int(s)])
        n_t = len(self.t_off) - 1
        xs, segs, ys, ts = [], [], [], []
        while len(ys) < self.batch_games:
            p = int(rng.integers(n_t))
            g = int(self.t_games[self.t_off[p] + int(rng.integers(self.t_off[p + 1] - self.t_off[p]))])
            codes = np.asarray(codes_all[self.off[g]:self.off[g + 1]])
            try:
                x, t = L.planes_for_codes(codes, "w" if self.is_white[g] else "b", self.moves, rng)
            except ValueError:
                continue
            if len(x) == 0:
                continue
            xs.append(L.sym(x, int(rng.integers(8))))
            segs.append(np.full(len(x), len(ys), np.int64))
            ts.append(np.asarray(t, np.int64))
            ys.append(p)
        return {"x": np.ascontiguousarray(np.concatenate(xs)), "seg": np.concatenate(segs),
                "y": np.asarray(ys, np.int64), "t": np.concatenate(ts)}

    def __getitem__(self, s):
        import torch
        return {k: torch.from_numpy(v) for k, v in self.batch(s).items()}


class EpisodeBatches(GameBatches):
    """V1: item s = EP_PLAYERS different T players; each gets two disjoint sets of games A, B with k_A, k_B drawn
    from EP_KS (games uniform among the player's games, both colours), EP_SET_MOVES // k random target moves per
    game, one random symmetry per game. Returns x, seg (game number), gset (set number of every game: 2 i for A,
    2 i + 1 for B), y (player of set pair i), t. A player whose A or B set ends without a sample is drawn again."""

    def __init__(self, cache_dir, off, is_white, t_games, t_off, seed=SEED, players=EP_PLAYERS,
                 set_moves=EP_SET_MOVES):
        super().__init__(cache_dir, off, is_white, t_games, t_off, seed)
        self.players, self.set_moves = players, set_moves

    def batch(self, s):
        codes_all = self._codes()
        rng = np.random.default_rng([self.seed, int(s)])
        n_t = len(self.t_off) - 1
        used = set()
        xs, segs, ts, gset, ys = [], [], [], [], []
        n_games = 0
        while len(ys) < self.players:
            p = int(rng.integers(n_t))
            if p in used:
                continue
            used.add(p)
            ka, kb = int(rng.choice(EP_KS)), int(rng.choice(EP_KS))
            ng = int(self.t_off[p + 1] - self.t_off[p])
            pick = rng.choice(ng, size=ka + kb, replace=False)
            sets = []
            for part, k in ((pick[:ka], ka), (pick[ka:], kb)):
                got = []
                for j in part.tolist():
                    g = int(self.t_games[self.t_off[p] + j])
                    codes = np.asarray(codes_all[self.off[g]:self.off[g + 1]])
                    try:
                        x, t = L.planes_for_codes(codes, "w" if self.is_white[g] else "b",
                                                  max(1, self.set_moves // k), rng)
                    except ValueError:
                        continue
                    if len(x) == 0:
                        continue
                    got.append((L.sym(x, int(rng.integers(8))), np.asarray(t, np.int64)))
                sets.append(got)
            if not sets[0] or not sets[1]:
                continue
            i = len(ys)
            for side, got in enumerate(sets):
                for x, t in got:
                    xs.append(x)
                    ts.append(t)
                    segs.append(np.full(len(x), n_games, np.int64))
                    gset.append(2 * i + side)
                    n_games += 1
            ys.append(p)
        return {"x": np.ascontiguousarray(np.concatenate(xs)), "seg": np.concatenate(segs),
                "gset": np.asarray(gset, np.int64), "y": np.asarray(ys, np.int64), "t": np.concatenate(ts)}


def t_index(cache):
    """The T players (sorted ids = class indices), their games grouped by player, offsets; asserts no E / D game."""
    E, D, T, _per = L.splits(cache)
    cls = {p: i for i, p in enumerate(T)}
    lab = np.array([cls.get(p, -1) for p in cache.pid.tolist()], np.int64)
    games = np.flatnonzero(lab >= 0)
    order = games[np.argsort(lab[games], kind="stable")]
    t_off = np.concatenate([[0], np.cumsum(np.bincount(lab[games], minlength=len(T)))]).astype(np.int64)
    held = set(E) | set(D)
    assert not any(p in held for p in np.unique(cache.pid[order]).tolist())
    assert (np.diff(t_off) >= L.T_MIN_GAMES).all()
    return T, order, t_off, {"E": L.sha256_ids(E), "D": L.sha256_ids(D), "T": L.sha256_ids(T)}


def make_batches(spec, cache, seed=SEED, small=None):
    """small: (games or players per batch, moves per game or per set) for --smoke."""
    T, order, t_off, ids = t_index(cache)
    is_white = cache.color == "W"
    if spec["sampler"] == "episode":
        kw = {} if small is None else {"players": small[0], "set_moves": small[1]}
        ds = EpisodeBatches(cache.dir, cache.off, is_white, order, t_off, seed, **kw)
    else:
        kw = {} if small is None else {"batch_games": small[0], "moves": small[1]}
        ds = GameBatches(cache.dir, cache.off, is_white, order, t_off, seed, **kw)
    return ds, T, ids


# ================================================================ models
def pos_bucket(t):
    return (t // POS_BUCKET).clamp(max=POS_BUCKETS - 1)


def build(spec, n_classes=None):
    """(EmbedNet, head or None). EmbedNet.forward(x, seg, n_seg, t) -> (n_seg, 256) game vectors (not normalised);
    EmbedNet.trunk(x) -> (P, 256) per-move vectors; EmbedNet.pool(e, seg, n_seg, t)."""
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
        """Residual block (prereg36's) + squeeze-excitation; optionally a KataGo-style global-pooling bias: 32
        channels of the input pooled (mean, max) -> Linear -> per-channel bias after the first convolution."""

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
            c = L.CHANNELS
            self.conv_in = nn.Conv2d(L.N_PLANES, c, 3, padding=1)
            self.bn_in = nn.BatchNorm2d(c)
            self.layers = nn.Sequential(*[GPSEBlock(c, i in GPOOL_BLOCKS) for i in range(L.BLOCKS)])
            self.fc = nn.Linear(2 * c, L.EMB_DIM)

        def forward(self, x):
            out = self.layers(F.relu(self.bn_in(self.conv_in(x))))
            return self.fc(torch.cat([out.mean(dim=(2, 3)), out.amax(dim=(2, 3))], 1))

    class MeanPool(nn.Module):
        def forward(self, e, seg, n_seg, t=None):
            return L.segment_mean(e, seg, n_seg)

    class TxPool(nn.Module):
        """Per-move vectors of one game + move-number embedding, a CLS token, a small pre-norm transformer
        encoder; the CLS output -> LayerNorm -> Linear. Positions of a segment keep their batch order."""

        def __init__(self, d=L.EMB_DIM):
            super().__init__()
            self.pos = nn.Embedding(POS_BUCKETS, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, TX_HEADS, TX_FF, dropout=TX_DROPOUT, batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, TX_LAYERS, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, d)

        def forward(self, e, seg, n_seg, t):
            d = e.shape[1]
            cnt = torch.bincount(seg, minlength=n_seg)
            lmax = int(cnt.max()) if len(seg) else 0
            start = torch.cumsum(cnt, 0) - cnt
            order = torch.argsort(seg, stable=True)
            seg_s = seg[order]
            idx = torch.arange(len(seg), device=e.device) - start[seg_s]
            h = (e.float() + self.pos(pos_bucket(t)).float())[order]
            pad = torch.zeros(n_seg, lmax + 1, d, dtype=h.dtype, device=e.device)
            pad[:, 0] = self.cls.float()[0, 0]
            pad[seg_s, idx + 1] = h
            mask = torch.arange(lmax + 1, device=e.device)[None, :] > cnt[:, None]
            y = self.enc(pad, src_key_padding_mask=mask)
            return self.out(self.norm(y[:, 0]))

    class TxResPool(nn.Module):
        """V3b: g = mean of the per-move vectors + Linear(LayerNorm(masked mean of the transformer outputs)).
        Tokens = LayerNorm(e) + move-number embedding (N(0, TXRES_POS_STD^2)); no CLS. The Linear starts at zero, so
        at initialisation g is exactly prereg36's mean pooling (L.segment_mean)."""

        def __init__(self, d=L.EMB_DIM):
            super().__init__()
            self.inorm = nn.LayerNorm(d)
            self.pos = nn.Embedding(POS_BUCKETS, d)
            nn.init.normal_(self.pos.weight, std=TXRES_POS_STD)
            layer = nn.TransformerEncoderLayer(d, TX_HEADS, TX_FF, dropout=TX_DROPOUT, batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, TX_LAYERS, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, d)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, e, seg, n_seg, t):
            d = e.shape[1]
            base = L.segment_mean(e, seg, n_seg)
            if len(seg) == 0:
                return base
            cnt = torch.bincount(seg, minlength=n_seg)
            lmax = int(cnt.max())
            start = torch.cumsum(cnt, 0) - cnt
            order = torch.argsort(seg, stable=True)
            seg_s = seg[order]
            idx = torch.arange(len(seg), device=e.device) - start[seg_s]
            h = (self.inorm(e.float()) + self.pos(pos_bucket(t)).float())[order]
            pad = torch.zeros(n_seg, lmax, d, dtype=h.dtype, device=e.device)
            pad[seg_s, idx] = h
            valid = torch.arange(lmax, device=e.device)[None, :] < cnt[:, None]
            mask = ~valid
            mask[:, 0] = False                 # an empty segment attends to one zero token; its readout is dropped
            y = self.enc(pad, src_key_padding_mask=mask)
            keep = valid.to(y.dtype)[..., None]
            ym = (y * keep).sum(1) / keep.sum(1).clamp(min=1)
            return base + self.out(self.norm(ym)).float() * (cnt > 0).to(base.dtype)[:, None]

    class EmbedNet(nn.Module):
        def __init__(self, trunk, pool):
            super().__init__()
            self.trunk, self.pool = trunk, pool

        def forward(self, x, seg, n_seg, t):
            return self.pool(self.trunk(x), seg, n_seg, t)

    class SubCenterHead(nn.Module):
        """Cosine softmax with SUBCENTERS centres per class (max cosine) and an additive margin self.m."""

        def __init__(self, n):
            super().__init__()
            self.n = n
            self.weight = nn.Parameter(torch.randn(n * SUBCENTERS, L.EMB_DIM) * 0.01)
            self.m = 0.0

        def forward(self, emb_normalised, y=None):
            cos = F.linear(emb_normalised.float(), F.normalize(self.weight.float(), dim=1))
            cos = cos.view(len(cos), self.n, SUBCENTERS).amax(-1)
            if y is not None and self.m > 0:
                cos = cos - self.m * F.one_hot(y, self.n).to(cos.dtype)
            return L.SCALE * cos

    p36net, coshead = L.build_model(n_classes if (n_classes and spec["head"] == "cos") else None)
    trunk = GPSENet() if spec["trunk"] == "gpse" else p36net
    pool = {"tx": TxPool, "txres": TxResPool}.get(spec["pool"], MeanPool)()
    head = None
    if n_classes:
        head = SubCenterHead(n_classes) if spec["head"] == "subcenter" else coshead
    return EmbedNet(trunk, pool), head


def param_groups(net, head, spec, lr=LR):
    """AdamW parameter groups. V3b: the pooling parameters get lr x POOL_LR_MULT[pool] (PLAN.md revision 3); every
    other variant: one group of all parameters, as in the original runs."""
    mult = POOL_LR_MULT.get(spec["pool"])
    if mult is None:
        return [{"params": list(net.parameters()) + list(head.parameters()), "lr": lr}]
    pool = [p for n, p in net.named_parameters() if n.startswith("pool.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("pool.")] + list(head.parameters())
    return [{"params": rest, "lr": lr}, {"params": pool, "lr": lr * mult}]


def margin_at(step):
    return MARGIN * min(1.0, step / float(MARGIN_RAMP))


def head_logits(head, spec, emb_normalised, y=None):
    if spec["head"] == "subcenter":
        return head(emb_normalised, y)
    return head(emb_normalised)


# ================================================================ EMA
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


# ================================================================ symmetries on tensors (TTA)
def sym_t(x, s):
    """p36_lib.sym on a torch tensor (last two axes): s >= 4 flips left-right first, then s % 4 quarter turns."""
    import torch
    if s >= 4:
        x = torch.flip(x, dims=(-1,))
    if s % 4:
        x = torch.rot90(x, s % 4, dims=(-2, -1))
    return x


def chunk_segments(seg, n_seg, chunk=CHUNK_MOVES):
    """V3 'chunk24': within every game (segment) its positions, in their order, dealt round-robin into
    max(1, round(n / chunk)) chunks. Returns (chunk id per position, game of every chunk, number of chunks)."""
    seg = np.asarray(seg)
    cnt = np.bincount(seg, minlength=n_seg)
    nch = np.maximum(1, np.rint(cnt / float(chunk)).astype(np.int64))
    base = np.concatenate([[0], np.cumsum(nch)[:-1]])
    order = np.argsort(seg, kind="stable")
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    rank = np.empty(len(seg), np.int64)
    rank[order] = np.arange(len(seg)) - start[seg[order]]
    cid = base[seg] + rank % nch[seg]
    owner = np.repeat(np.arange(n_seg), nch)
    return cid, owner, int(nch.sum())
