"""prereg08 P1 (competition 2): KataGo HumanSL "strength probe" features, shared library.

One forward pass per position with NEUTRAL metadata (both ranks unknown); per position we read
    * the trunk output (after the final norm + activation, 384 channels) at the location of the move actually played;
    * the trunk output mean-pooled over the 19 x 19 board (384);
    * log p(played move) under the main policy head (channel 0), illegal moves masked before the log-softmax;
    * the entropy of that masked policy.
Each is averaged over the game's moves, separately for the target side (the row's player) and the opponent side.
Per game and side that is a 770-vector: [0:384] trunk at move, [384:768] board mean, 768 log p, 769 entropy.

Reused, never edited: results/prereg07/explore/pilot_humansl/hs_lib.py (pinned SHA-256, imported only after the pin
check; its import installs the data/local guard), the vendored KataGo python code in third_party/katago.

Featurisation is KataGo's own Features.fill_row_features, with one change of speed only: the ladder search
(Features.iterLadders, ~90 % of the CPU time) is memoised per board within a game. fill_row_features calls it on the
current board, the previous one and the one before, so every board was searched up to three times. The memo is keyed by
the board's zobrist hash AND verified against the full board array, so a hit returns exactly what a new search would.
tests (test_p1.py) check that the inputs equal hs_lib.featurize_game's bit for bit.

Neutral metadata (fixed in p1_plan.json): the C++ 'rank_<r>' preset of the PREREG_07 pilot (both human, ratedness
unknown, byo-yomi 1200 s + 5 x 30 s, KGS, 2020-03-01) except that BOTH ranks are unknown (rank-unknown flags on,
no rank bits). Rules: hs_lib.RULES (Japanese, komi 6.5).

No file is written by this module.
"""

import datetime
import hashlib
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PILOT_CODE = os.path.join(REPO, "results", "prereg07", "explore", "pilot_humansl")
U1_DIR = os.path.join(REPO, "results", "prereg05", "explore", "u1")

# pinned files (SHA-256 with CRLF folded to LF), same values as tools/p07_rank_fuse.py PINNED
PINNED = {
    "results/prereg07/explore/pilot_humansl/hs_lib.py":
        "d19b6a8615ef757891ff2d451fe5b96b19f31cc706a87862e49a7688d0bde97c",
    "results/prereg07/explore/pilot_humansl/pilot_fuse.py":
        "37fb2dff0f37032bf08acc7f27213cafd30533ab024973f92b5b28f41c6ac0c6",
    "results/prereg05/explore/u1/u1_lib.py":
        "225dd51cff961ae3608e003432388e0b6d6a053fb4dac26add5425c002dc3ff7"}


class P1Error(RuntimeError):
    pass


def sha_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def sha_raw(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def check_pin(rel):
    got = sha_lf(os.path.join(REPO, *rel.split("/")))
    if got != PINNED[rel]:
        raise P1Error(f"{rel}: SHA-256 {got} != pinned {PINNED[rel]} (pinned files are never edited)")
    return got


check_pin("results/prereg07/explore/pilot_humansl/hs_lib.py")
if PILOT_CODE not in sys.path:
    sys.path.insert(0, PILOT_CODE)
import numpy as np  # noqa: E402

import hs_lib as H  # noqa: E402  (installs the data/local guard: any open / listdir / stat under data/local raises)
from katago.game.board import Board  # noqa: E402
from katago.game.features import Features  # noqa: E402
from katago.game.sgfmetadata import SGFMetadata  # noqa: E402

POS_LEN = H.POS_LEN
N_POLICY = H.N_POLICY
C_TRUNK = 384
F_DIM = 2 * C_TRUNK + 2          # trunk at move, board mean, log p, entropy
F_SLICES = {"trunk_at_move": (0, 384), "board_mean": (384, 768), "logp": (768, 769), "entropy": (769, 770)}
NEUTRAL_DATE = datetime.date(2020, 3, 1)
CHUNK_FORMAT = "c10-prereg08 p1 chunk v1"


# ================================================================ neutral metadata
def neutral_sgfmeta():
    return SGFMetadata(
        inverseBRank=0, inverseWRank=0, bRankIsUnknown=True, wRankIsUnknown=True, bIsHuman=True, wIsHuman=True,
        gameIsUnrated=False, gameRatednessIsUnknown=True,
        tcIsUnknown=False, tcIsByoYomi=True, mainTimeSeconds=1200.0, periodTimeSeconds=30.0, byoYomiPeriods=5,
        gameDate=NEUTRAL_DATE, source=SGFMetadata.SOURCE_KGS)


def neutral_table():
    """[3, 192] float32: row [pla] for pla in (Board.BLACK=1, Board.WHITE=2); row 0 unused (zeros)."""
    t = np.zeros((3, SGFMetadata.METADATA_INPUT_NUM_CHANNELS), dtype=np.float32)
    m = neutral_sgfmeta()
    for pla in (Board.BLACK, Board.WHITE):
        t[pla] = m.get_metadata_row(nextPlayer=pla, boardArea=POS_LEN * POS_LEN)
    return t


def neutral_table_sha():
    return hashlib.sha256(neutral_table().tobytes()).hexdigest()


# ================================================================ featurisation (KataGo features, memoised ladders)
class MemoFeatures(Features):
    """Features with iterLadders memoised per board (zobrist key, verified against the board array)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.memo = {}
        self.hits = 0
        self.misses = 0

    def reset(self):
        self.memo.clear()

    def iterLadders(self, board, f):
        key = int(board.zobrist)
        arr = board.board.tobytes()
        hit = self.memo.get(key)
        if hit is not None and hit[0] == arr:
            self.hits += 1
            for loc, pos, wm in hit[1]:
                f(loc, pos, list(wm))
            return
        self.misses += 1
        calls = []
        Features.iterLadders(self, board, lambda loc, pos, wm: calls.append((loc, pos, tuple(wm))))
        self.memo[key] = (arr, calls)
        for loc, pos, wm in calls:
            f(loc, pos, list(wm))


_FEATURES = None


def _features():
    global _FEATURES
    if _FEATURES is None:
        _FEATURES = MemoFeatures({"version": 15}, POS_LEN)
    return _FEATURES


def featurize_game(sgf, target, sides=("target", "opponent")):
    """Same output as hs_lib.featurize_game (bit for bit; tests check it), with the memoised ladder search."""
    features = _features()
    features.reset()
    try:
        return H.featurize_game(sgf, target, sides=sides, features=features)
    finally:
        features.reset()


# ================================================================ model and per-position features
class ProbeScorer:
    """Batched forward passes with the neutral metadata; per position -> float32 [F_DIM] features.

    precision 'fp32' / 'fp16' / 'bf16' (autocast; fp16 only on CUDA). pad: pad every forward pass to `batch` rows
    (copies of the last real row, dropped) so that cuDNN autotunes one shape only; default on for CUDA. Rows are
    evaluated independently in eval mode, so padding never changes a real row."""

    def __init__(self, model, device="cpu", precision="fp32", batch=1024, channels_last=False, pad=None):
        import torch
        self.torch = torch
        self.model = model
        self.device = torch.device(device)
        self.precision = precision
        self.batch = int(batch)
        self.channels_last = bool(channels_last)
        self.pad = (self.device.type == "cuda") if pad is None else bool(pad)
        if precision not in ("fp32", "fp16", "bf16"):
            raise P1Error(f"precision {precision!r}")
        if precision == "fp16" and self.device.type != "cuda":
            raise P1Error("fp16 autocast is for CUDA only")
        if channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.meta = torch.tensor(neutral_table(), device=self.device)       # [3, 192]
        self._trunk = None
        self._hook = model.act_trunkfinal.register_forward_hook(self._grab)

    def _grab(self, _mod, _inp, out):
        self._trunk = out

    def close(self):
        self._hook.remove()

    def _forward(self, b, g, m):
        torch = self.torch
        if self.channels_last:
            b = b.contiguous(memory_format=torch.channels_last)
        self._trunk = None
        if self.precision == "fp32":
            out = self.model(b, g, input_meta=m)
        else:
            dt = torch.float16 if self.precision == "fp16" else torch.bfloat16
            with torch.autocast(device_type=self.device.type, dtype=dt):
                out = self.model(b, g, input_meta=m)
        if self._trunk is None:
            raise P1Error("trunk hook did not fire")
        return out[0][0][:, 0, :].float(), self._trunk.float()             # [B,362], [B,384,19,19]

    def position_features(self, feats):
        """feats: a featurize_game dict (or a concatenation) -> float32 [T, F_DIM]."""
        torch = self.torch
        T = len(feats["move"])
        out = np.zeros((T, F_DIM), dtype=np.float32)
        if T == 0:
            return out
        per = self.batch
        with torch.inference_mode():
            for s in range(0, T, per):
                e = min(T, s + per)
                P = e - s
                rows = np.arange(s, e) if not self.pad or P == per else \
                    np.concatenate([np.arange(s, e), np.full(per - P, e - 1)])
                R = len(rows)
                b = torch.as_tensor(feats["bin"][rows], device=self.device).float()
                g = torch.as_tensor(feats["glob"][rows], device=self.device).float()
                pla = torch.as_tensor(feats["pla"][rows].astype(np.int64), device=self.device)
                lm = torch.as_tensor(feats["legal"][rows], device=self.device)
                mv = torch.as_tensor(feats["move"][rows].astype(np.int64), device=self.device)
                logits, trunk = self._forward(b, g, self.meta[pla])
                logits = logits.masked_fill(~lm, float("-inf"))
                lp = torch.log_softmax(logits, dim=-1)
                lp_move = lp.gather(1, mv.view(R, 1)).squeeze(1)
                p = lp.exp()
                ent = -torch.where(lm, p * lp, torch.zeros_like(lp)).sum(1)
                flat = trunk.reshape(R, C_TRUNK, POS_LEN * POS_LEN)
                at = flat.gather(2, mv.clamp(max=POS_LEN * POS_LEN - 1).view(R, 1, 1).expand(R, C_TRUNK, 1)).squeeze(2)
                mean = flat.mean(2)
                f = torch.cat([at, mean, lp_move.view(R, 1), ent.view(R, 1)], dim=1)
                out[s:e] = f[:P].cpu().numpy()
        if not np.isfinite(out).all():
            raise P1Error("non-finite probe feature (mask or precision problem)")
        return out


def game_means(pf, side):
    """pf [T, F_DIM] of one game, side [T] (0 target, 1 opponent) -> means [2, F_DIM] float64 (0 where no move),
    counts [2]."""
    means = np.zeros((2, F_DIM), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for s in (0, 1):
        m = side == s
        counts[s] = int(m.sum())
        if counts[s]:
            means[s] = pf[m].astype(np.float64).mean(0)
    return means, counts


def load_model(device="cpu"):
    """KataGo HumanSL model ('model' weights) through hs_lib's checked loader."""
    model, _state = H.load_model(H.CKPT_PATH, device=device, check=True)
    return model


def code_shas():
    out = {}
    for name in ("p1_lib.py", "p1_extract.py", "p1_fuse.py"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            out[f"results/prereg08/p1/{name}"] = sha_lf(p)
    out.update({k: sha_lf(os.path.join(REPO, *k.split("/"))) for k in PINNED})
    return out
