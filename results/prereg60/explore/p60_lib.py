"""prereg60 (PLAN.md Method): the 18-plane input, samples at every move of the target, the widened network.

Nothing of prereg36 / 42 / 49 is edited; their code is imported. What changes lives here:

  * planes (PLAN change 1): 18 planes = prereg36's 17 (p36_lib.planes_at) + plane 17 = 1 at the point of the target's
    move t, 0 elsewhere.
  * samples (PLAN change 2): EVERY non-pass move of the target gives a sample (p36_lib.target_moves: only from the
    8th non-pass move of the game on). A history position that does not exist (before the first move of the game)
    is an empty board: planes 0-7 / 8-15 of such a slot are all zero. Because the position before move 0 IS the
    empty board, slot j of the sample at move t shows the position before move max(t - 7 + j, 0).
    Per-game sampling is p36_lib.planes_for_codes' rule on the larger set: training takes a uniform random subset
    of at most k = 24 of the target's moves (rng.choice without replacement, sorted), the game vector takes all of
    them. `count` of a game = number of non-pass moves of the target (prereg36's count + the target's moves among
    the first 7 non-pass moves of the game, i.e. + 3 or 4 in a normal game); a game whose replay hits an illegal
    move has count 0, as before.
  * GameBatches60 / GameSet60: p42_lib.GameBatches / p42_embed.GameSet with the planes function above. They are
    classes of this module (not monkey-patches) because DataLoader workers are fresh processes on Windows.
    early=False, action=False reproduce the old samples (planes 0-16 identical, plane 17 zero) - seam / tests.
  * widen(net): the first convolution gets an 18th input channel with zero weights, so the widened network equals
    the 17-plane network whatever plane 17 holds.

torch is imported inside the functions (the replay workers do not need it at import time).
"""
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (os.path.join(REPO, "results", r, "explore") for r in ("prereg36", "prereg42", "prereg43", "prereg49",
                                                                 "prereg50")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import p42_lib as V  # noqa: E402
import p42_embed as EM  # noqa: E402

L = V.L
HIST = L.HIST
N_OLD = L.N_PLANES                  # 17
N_PLANES = N_OLD + 1                # 18
ACTION = N_OLD                      # index of the new plane


# ================================================================ planes
def target_moves60(codes, target_color, early=True):
    """Indices (among the non-pass moves) of the target's moves that give a sample: all of them (early=True, PLAN
    change 2) or prereg36's (early=False: from the 8th non-pass move of the game on)."""
    col = np.asarray(codes) >= 361
    t = np.flatnonzero(col == (target_color.lower() == "w"))
    return t if early else t[t >= HIST - 1]


def planes_at60(S, t, target_color, codes, action=True):
    """18 planes at the target-move indices t from the replay S (p36_lib.replay: row i = position before move i).
    Planes 0-15 as p36_lib.planes_at, with history slots before the start of the game = the empty board (= S[0]);
    plane 16 the colour plane; plane 17 one-hot at the point of move t (all zero when action=False)."""
    t = np.asarray(t, np.int64)
    m = len(t)
    black = target_color.lower() == "b"
    idx = np.maximum(t[:, None] + np.arange(-(HIST - 1), 1)[None, :], 0)
    H = S[idx]                                                      # (m, 8, 361)
    out = np.zeros((m, N_PLANES, 361), np.uint8)
    v = 1 if black else -1
    out[:, :HIST] = H == v
    out[:, HIST:2 * HIST] = H == -v
    out[:, 2 * HIST] = 1 if black else 0
    if action and m:
        out[np.arange(m), ACTION, np.asarray(codes)[t].astype(np.int64) % 361] = 1
    return out.reshape(m, N_PLANES, 19, 19)


def planes_for_codes60(codes, target_color, k=None, rng=None, early=True, action=True):
    """p36_lib.planes_for_codes with 18 planes and the sample set of target_moves60: all samples of the game (k
    None), or a random subset of at most k target moves (sorted). Returns (planes, move indices)."""
    t = target_moves60(codes, target_color, early)
    if k is not None and len(t) > k:
        t = np.sort(rng.choice(t, size=k, replace=False))
    if len(t) == 0:
        return np.zeros((0, N_PLANES, 19, 19), np.uint8), t
    S = L.replay(codes, int(t[-1]))
    return planes_at60(S, t, target_color, codes, action), t


# ================================================================ samplers
class GameBatches60(V.GameBatches):
    """p42_lib.GameBatches (prereg36's sampler: 96 games x <= 24 moves, rng default_rng([seed, s])) with the planes
    of planes_for_codes60. The loop is the parent's, line for line; with early=False the draws are identical."""

    def __init__(self, cache_dir, off, is_white, t_games, t_off, seed=V.SEED, batch_games=L.BATCH_GAMES,
                 moves=L.SEG_MOVES, early=True, action=True):
        super().__init__(cache_dir, off, is_white, t_games, t_off, seed, batch_games, moves)
        self.early, self.action = bool(early), bool(action)

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
                x, t = planes_for_codes60(codes, "w" if self.is_white[g] else "b", self.moves, rng, self.early,
                                          self.action)
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


class GameSet60(EM.GameSet):
    """p42_embed.GameSet (item i = 16 games, all samples of every game) with the planes of planes_for_codes60."""

    def __init__(self, codes, start, end, is_white, early=True, action=True):
        super().__init__(codes, start, end, is_white)
        self.early, self.action = bool(early), bool(action)

    def __getitem__(self, i):
        import torch
        if isinstance(self.codes, str):
            self.codes = np.load(self.codes, mmap_mode="r")
        xs, seg, ts = [], [], []
        for j, g in enumerate(range(i * EM.CHUNK, min(self.n, (i + 1) * EM.CHUNK))):
            try:
                x, t = planes_for_codes60(np.asarray(self.codes[self.start[g]:self.end[g]]),
                                          "w" if self.is_white[g] else "b", None, None, self.early, self.action)
            except ValueError:
                continue
            if len(x):
                xs.append(x)
                seg.append(np.full(len(x), j, np.int64))
                ts.append(np.asarray(t, np.int64))
        if not xs:
            z = torch.zeros(0, dtype=torch.int64)
            return i, torch.zeros((0, N_PLANES, 19, 19), dtype=torch.uint8), z, z
        return i, torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(seg)), \
            torch.from_numpy(np.concatenate(ts))


# ================================================================ model
def widen(net):
    """Replace net.trunk.conv_in (17 input planes) by an 18-plane convolution: the old weights and bias, the new
    input channel all zero. In place; returns net."""
    import torch
    import torch.nn as nn
    old = net.trunk.conv_in
    if old.in_channels != N_OLD:
        raise ValueError("conv_in has %d input planes, expected %d" % (old.in_channels, N_OLD))
    new = nn.Conv2d(N_PLANES, old.out_channels, old.kernel_size, padding=old.padding)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :N_OLD].copy_(old.weight)
        new.bias.copy_(old.bias)
    net.trunk.conv_in = new.to(old.weight.device)
    return net


def build_wide(spec, n_classes=None):
    """(18-plane EmbedNet, head or None): p42_lib.build, then widen (the new channel is zero until weights are
    loaded)."""
    net, head = V.build(spec, n_classes)
    return widen(net), head


def widen_state(sd):
    """A 17-plane EmbedNet state dict -> the 18-plane one (new channel zero). Tensors are cloned."""
    import torch
    out = {k: v.clone() for k, v in sd.items()}
    w = out["trunk.conv_in.weight"]
    if w.shape[1] != N_OLD:
        raise ValueError("conv_in weight has %d input planes" % w.shape[1])
    out["trunk.conv_in.weight"] = torch.cat([w, torch.zeros_like(w[:, :1])], 1)
    return out


# ================================================================ schedule and checkpoints
def lr_factor(s, warmup, total):
    """Multiplier of the peak learning rate at optimiser step s (0-based): linear warm-up (s + 1) / warmup for the
    first `warmup` steps, then a cosine from 1 (s = warmup) to 0 (s = total)."""
    if s < warmup:
        return (s + 1) / float(warmup)
    return 0.5 * (1.0 + np.cos(np.pi * (s - warmup) / float(total - warmup)))


def step_of(path):
    return int(re.search(r"step_(\d+)\.pt$", path).group(1))


def last_ckpt(d):
    """The checkpoint with the highest step number (p50_run.last_checkpoint_by_step, not the file-name order)."""
    import p50_run
    return p50_run.last_checkpoint_by_step(d)


def prune(d, keep_steps, latest):
    """Remove every step_*.pt of d whose step is neither kept (PLAN: 10,000 / 20,000 / 30,000) nor `latest`.
    Returns the removed file names."""
    gone = []
    for p in glob.glob(os.path.join(d, "step_*.pt")):
        s = step_of(p)
        if s not in keep_steps and s != latest:
            os.remove(p)
            gone.append(os.path.basename(p))
    return sorted(gone)
