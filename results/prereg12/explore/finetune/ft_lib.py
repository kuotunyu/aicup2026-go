"""prereg12 exploration (competition 2): fine-tuning pilot of the KataGo HumanSL network on our rank labels.
Shared library of ft_pilot.py (sets, cache, bench, train, extract) and ft_eval.py. See PLAN.md in this folder.

Model (PLAN.md section 2): KataGo b18c384nbt-humanv0 ('model' weights, hs_lib's checked loader), neutral metadata
(p1_lib.neutral_table, both ranks unknown), forward stopped at model.act_trunkfinal (a forward hook raises TrunkDone,
so the policy / value heads are never run). The last K trunk blocks and the final trunk norm are trainable, the rest is
frozen and the whole KataGo net stays in eval mode (its norms use the stored statistics). New head (RankHead):
    per position  z = [trunk at the played point (384), trunk board mean (384)] -> LayerNorm -> Linear(768, H) -> GELU
    per game      mean of z over the target's positions, mean over the opponent's positions, has-opponent flag,
                  target colour one-hot -> Linear(2H + 3, 10) = level logits (lp = log_softmax)
Training: game-level cross entropy on the target's level, fit players only (a sample of up to POS_PER_SIDE positions
per side per game); inference on inner games pools every position of the game.

Never touches data/local (importing p1_lib installs hs_lib's guard), never reads validation features.
No file is written by this module except through the explicit save helpers called by ft_pilot.py.
"""

import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
P1_CODE = os.path.join(REPO, "results", "prereg08", "p1")
for _p in (TOOLS, P1_CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import p1_lib as P  # noqa: E402  (pins hs_lib, installs the data/local guard)

H = P.H
Board = P.Board

DATA = os.path.join(REPO, "data", "prereg12")
STOP_FILE = os.path.join(DATA, "STOP")
SETS_PATH = os.path.join(DATA, "sets.json")
CACHE_DIR = os.path.join(DATA, "cache")
TRAIN_DIR = os.path.join(DATA, "train")
FT_DIR = os.path.join(DATA, "ft")
LOG_PATH = os.path.join(DATA, "run_log.jsonl")
P1_SETS = os.path.join(REPO, "data", "prereg08", "p1", "sets.json")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
TRAIN_CSV_DIR = os.path.join(REPO, "data", "training")

LEVELS = list(H.LEVELS)
N_LEVELS = len(LEVELS)
POS_LEN = P.POS_LEN
N_BIN = 22
N_GLOB = 19
N_BITS = N_BIN * POS_LEN * POS_LEN          # 7942
N_BYTES = (N_BITS + 7) // 8                  # 993
C_TRUNK = P.C_TRUNK
CACHE_FORMAT = "c10-prereg12 ft cache v1"
FT_FORMAT = "c10-prereg12 ft inner v1"
SETS_FORMAT = "c10-prereg12 ft sets v1"

# PLAN.md section 2 (fixed before any GPU run)
CONFIG = {
    "k_trainable_blocks": 4,
    "hidden": 256,
    "dropout": 0.1,
    "pos_per_side": 16,
    "batch_games": 8,
    "lr_head": 1e-3,
    "lr_trunk": 2e-5,
    "wd_head": 1e-2,
    "wd_trunk": 0.0,
    "head_only_steps": 1000,
    "trunk_warmup_steps": 1000,
    "cosine_floor": 0.1,
    "grad_clip": 1.0,
    "epochs": 1,
    "max_train_gpu_hours": 3.0,
    "eval_every": 2500,
    "ckpt_every": 500,
    "seed": 20261012,
    "precision_cuda": "fp16",
}
TRAIN_PER_LEVEL = 20000
HOLDOUT_PER_LEVEL = 300
HOLDOUT_PLAYER_FRAC = 0.05
SETS_SEED = 20261012
CACHE_CHUNK = {"train": 1000, "holdout": 1000, "bank": 200, "mono": 200}


class FTError(RuntimeError):
    pass


# ================================================================ small utilities
def utc_now():
    return H.utc_now()


def sha_raw(path, block=1 << 20):
    return P.sha_raw(path, block)


def sha_lf(path):
    return P.sha_lf(path)


def rel(path):
    return os.path.relpath(path, REPO).replace("\\", "/")


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def log(msg, log_path=None, **kv):
    rec = {"time_utc": utc_now(), "msg": msg, **kv}
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    p = log_path or LOG_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def stop_requested(stop_file=None):
    return os.path.exists(stop_file or STOP_FILE)


def other_gpu_compute():
    """nvidia-smi rows of Type exactly 'C' (another compute job). Must be called before any CUDA call."""
    rows, _ = H.other_compute_processes()
    return rows


def code_shas():
    out = {}
    for name in ("ft_lib.py", "ft_pilot.py", "ft_eval.py"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            out[f"results/prereg12/explore/finetune/{name}"] = sha_lf(p)
    out.update(P.code_shas())
    return out


def move_key(sgf):
    """SHA-1 of the move sequence (duplicate detection between training and inner games, original orientation)."""
    mv = H.MOVE_RE.findall(sgf)
    return hashlib.sha1("".join(c + a + b for c, a, b in mv).encode("ascii")).hexdigest()


# ================================================================ featurisation
def select_turns(sgf, target, per_side, seed):
    """Turn indices to featurise for a training game: up to per_side of the target's moves and up to per_side of the
    opponent's moves, uniformly without replacement (numpy default_rng(seed)); sorted."""
    moves = H.parse_moves(sgf)
    tgt = Board.BLACK if target == "B" else Board.WHITE
    t_idx = [i for i, (pla, _x, _y) in enumerate(moves) if pla == tgt]
    o_idx = [i for i, (pla, _x, _y) in enumerate(moves) if pla != tgt]
    rng = np.random.default_rng(seed)
    pick = []
    for idx in (t_idx, o_idx):
        if len(idx) <= per_side:
            pick += idx
        else:
            pick += [idx[j] for j in rng.choice(len(idx), size=per_side, replace=False)]
    return sorted(pick)


def featurize_turns(sgf, target, turns=None):
    """hs_lib.featurize_game (sides target and opponent) restricted to the given turn indices (None = every move),
    with p1_lib's memoised ladder search. Same arrays as p1_lib.featurize_game on those rows (tests check it)."""
    if turns is None:
        return P.featurize_game(sgf, target, sides=("target", "opponent"))
    want = set(int(t) for t in turns)
    features = P._features()
    features.reset()
    try:
        rules = H.RULES
        tgt = Board.BLACK if target == "B" else Board.WHITE
        moves = H.parse_moves(sgf)
        board = Board(size=POS_LEN)
        boards = [board.copy()]
        hist = []
        nb, ng = features.bin_input_shape[0], features.global_input_shape[0]
        bins, globs, legal, mv, plas, side, turn = [], [], [], [], [], [], []
        row_bin = np.zeros((1, POS_LEN * POS_LEN, nb), dtype=np.float32)
        row_glob = np.zeros((1, ng), dtype=np.float32)
        for i, (pla, x, y) in enumerate(moves):
            if board.pla != pla:
                raise H.HSError(f"move {i + 1}: {pla} to play but the board expects {board.pla}")
            loc = board.loc(x, y)
            if i in want:
                row_bin.fill(0.0)
                row_glob.fill(0.0)
                features.fill_row_features(board, pla, Board.get_opp(pla), boards, hist, len(hist), rules,
                                           row_bin, row_glob, 0)
                lm = H.legal_mask(board, pla)
                pos = y * POS_LEN + x
                if not lm[pos]:
                    raise H.HSError(f"move {i + 1} ({x}, {y}) is illegal under the mask")
                bins.append(row_bin[0].reshape(POS_LEN, POS_LEN, nb).transpose(2, 0, 1).astype(np.uint8))
                globs.append(row_glob[0].copy())
                legal.append(lm)
                mv.append(pos)
                plas.append(pla)
                side.append(0 if pla == tgt else 1)
                turn.append(i)
            board.play(pla, loc)
            hist.append((pla, loc))
            boards.append(board.copy())
        T = len(mv)
        return {"bin": np.stack(bins) if T else np.zeros((0, nb, POS_LEN, POS_LEN), np.uint8),
                "glob": np.stack(globs) if T else np.zeros((0, ng), np.float32),
                "legal": np.stack(legal) if T else np.zeros((0, H.N_POLICY), bool),
                "move": np.asarray(mv, dtype=np.int16), "pla": np.asarray(plas, dtype=np.int8),
                "side": np.asarray(side, dtype=np.int8), "turn": np.asarray(turn, dtype=np.int16),
                "n_moves": len(moves)}
    finally:
        features.reset()


def pack_bin(b):
    """uint8 0/1 [T, 22, 19, 19] -> uint8 [T, 993] (np.packbits, big bit order)."""
    return np.packbits(np.asarray(b, dtype=np.uint8).reshape(len(b), -1), axis=1)


def unpack_bin_np(bits):
    return np.unpackbits(bits, axis=1, count=N_BITS).reshape(len(bits), N_BIN, POS_LEN, POS_LEN)


def featurize_job(args):
    """Pool worker: (game index in chunk, sgf, colour, turns or None) -> (gi, packed dict or None, error)."""
    gi, sgf, colour, turns = args
    try:
        f = featurize_turns(sgf, colour, turns)
    except H.HSError as e:
        return gi, None, str(e)
    return gi, {"bits": pack_bin(f["bin"]), "glob": f["glob"].astype(np.float32), "move": f["move"],
                "pla": f["pla"], "side": f["side"], "turn": f["turn"], "n_moves": f["n_moves"]}, None


def build_cache_chunk(pool, games, turns_of, mapper=None):
    """games: list of dicts (sgf_content, color, ...); turns_of(i, game) -> turn list or None.
    -> dict of arrays (positions concatenated in game order; gidx = game index in the chunk)."""
    G = len(games)
    args = [(gi, g["sgf_content"], g["color"], turns_of(gi, g)) for gi, g in enumerate(games)]
    it = pool.imap(featurize_job, args, chunksize=2) if pool is not None else map(featurize_job, args)
    parts = [None] * G
    failed = np.zeros(G, dtype=bool)
    n_moves = np.zeros(G, dtype=np.int16)
    errors = []
    for gi, f, err in it:
        if err is not None:
            failed[gi] = True
            errors.append({"game": int(gi), "error": err})
            continue
        parts[gi] = f
        n_moves[gi] = f["n_moves"]
    counts = np.zeros((G, 2), dtype=np.int32)
    cols = {k: [] for k in ("bits", "glob", "move", "pla", "side", "turn", "gidx")}
    for gi, f in enumerate(parts):
        if f is None:
            continue
        T = len(f["move"])
        for k in ("bits", "glob", "move", "pla", "side", "turn"):
            cols[k].append(f[k])
        cols["gidx"].append(np.full(T, gi, dtype=np.int32))
        counts[gi, 0] = int((f["side"] == 0).sum())
        counts[gi, 1] = int((f["side"] == 1).sum())
    empty = {"bits": np.zeros((0, N_BYTES), np.uint8), "glob": np.zeros((0, N_GLOB), np.float32),
             "move": np.zeros(0, np.int16), "pla": np.zeros(0, np.int8), "side": np.zeros(0, np.int8),
             "turn": np.zeros(0, np.int16), "gidx": np.zeros(0, np.int32)}
    out = {k: (np.concatenate(v) if v else empty[k]) for k, v in cols.items()}
    out.update({"counts": counts, "failed": failed, "n_moves": n_moves, "errors": json.dumps(errors)})
    return out


def load_cache_chunk(path):
    with np.load(path, allow_pickle=False) as z:
        if str(z["format"]) != CACHE_FORMAT:
            raise FTError(f"{path}: not a {CACHE_FORMAT} chunk")
        return {k: z[k] for k in z.files}


# ================================================================ model
_TORCH = {}


def torch_mod():
    if "torch" not in _TORCH:
        import torch
        _TORCH["torch"] = torch
    return _TORCH["torch"]


def make_classes():
    """torch classes (built lazily so that CPU-only tools importing ft_lib never import torch)."""
    if "classes" in _TORCH:
        return _TORCH["classes"]
    torch = torch_mod()
    nn = torch.nn

    class TrunkDone(Exception):
        pass

    class RankHead(nn.Module):
        def __init__(self, c=C_TRUNK, hidden=256, dropout=0.1, n_levels=N_LEVELS):
            super().__init__()
            self.norm = nn.LayerNorm(2 * c)
            self.proj = nn.Linear(2 * c, hidden)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.cls = nn.Linear(2 * hidden + 3, n_levels)
            self.hidden = hidden

        def pos_hidden(self, trunk, move):
            """trunk [P, C, 19, 19] (float32), move [P] long -> hidden [P, H]."""
            P_, C = trunk.shape[0], trunk.shape[1]
            flat = trunk.reshape(P_, C, POS_LEN * POS_LEN)
            at = flat.gather(2, move.clamp(max=POS_LEN * POS_LEN - 1).view(P_, 1, 1).expand(P_, C, 1)).squeeze(2)
            z = torch.cat([at, flat.mean(2)], dim=1)
            return self.act(self.proj(self.norm(z)))

        def game_logits_from_sums(self, sums, counts, colour):
            """sums [G, 2, H] (sum of hidden over the side's positions), counts [G, 2], colour [G] (0 B, 1 W)."""
            c = counts.clamp(min=1).unsqueeze(-1).to(sums.dtype)
            mean = sums / c
            has_opp = (counts[:, 1] > 0).to(sums.dtype).unsqueeze(1)
            col = torch.nn.functional.one_hot(colour.long(), 2).to(sums.dtype)
            x = torch.cat([self.drop(mean[:, 0]), self.drop(mean[:, 1]), has_opp, col], dim=1)
            return self.cls(x)

        def game_logits(self, h, seg, n_games, colour):
            """h [P, H], seg [P] = game * 2 + side -> logits [G, 10]."""
            sums = torch.zeros(n_games * 2, h.shape[1], dtype=h.dtype, device=h.device).index_add_(0, seg, h)
            counts = torch.bincount(seg, minlength=n_games * 2).view(n_games, 2)
            return self.game_logits_from_sums(sums.view(n_games, 2, -1), counts, colour)

    class FTNet(nn.Module):
        """KataGo base (always eval mode; only the last k blocks and the final trunk norm trainable) + RankHead."""

        def __init__(self, base, k, hidden=256, dropout=0.1, meta_table=None):
            super().__init__()
            self.base = base
            self.k = int(k)
            for p in base.parameters():
                p.requires_grad_(False)
            n = len(base.blocks)
            if not 0 <= self.k <= n:
                raise FTError(f"k {k} outside 0..{n}")
            if self.k > 0:
                for blk in base.blocks[n - self.k:]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
                for p in base.norm_trunkfinal.parameters():
                    p.requires_grad_(True)
            self.head = RankHead(hidden=hidden, dropout=dropout)
            mt = P.neutral_table() if meta_table is None else meta_table
            self.register_buffer("meta", torch.tensor(mt, dtype=torch.float32), persistent=False)
            self._trunk = None
            self._active = False
            self._hook = base.act_trunkfinal.register_forward_hook(self._grab)
            self.base.eval()

        def _grab(self, _m, _i, out):
            if self._active:          # only inside trunk(): other users of the base model run it to the end
                self._trunk = out
                raise TrunkDone()

        def train(self, mode=True):
            super().train(mode)
            self.base.eval()          # KataGo norms keep their stored statistics
            return self

        def trunk(self, b, g, pla):
            self._trunk = None
            self._active = True
            try:
                self.base(b, g, input_meta=self.meta[pla.long()])
            except TrunkDone:
                pass
            finally:
                self._active = False
            if self._trunk is None:
                raise FTError("trunk hook did not fire")
            t, self._trunk = self._trunk, None
            return t

        def trainable_named(self):
            return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    class TinyBase(nn.Module):
        """CPU smoke stand-in with KataGo's interface (blocks, norm_trunkfinal, act_trunkfinal, forward(b, g,
        input_meta)). Never used for any reported number."""

        def __init__(self, n_blocks=6, c=C_TRUNK):
            super().__init__()
            self.conv_spatial = nn.Conv2d(N_BIN, c, 3, padding=1, bias=False)
            self.linear_global = nn.Linear(N_GLOB, c, bias=False)
            self.meta_lin = nn.Linear(192, c, bias=False)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
                                         for _ in range(n_blocks)])
            self.norm_trunkfinal = nn.BatchNorm2d(c)
            self.act_trunkfinal = nn.ReLU()
            self.policy = nn.Conv2d(c, 1, 1)

        def forward(self, b, g, input_meta=None):
            out = self.conv_spatial(b) + self.linear_global(g)[:, :, None, None]
            if input_meta is not None:
                out = out + self.meta_lin(input_meta)[:, :, None, None]
            for blk in self.blocks:
                out = out + blk(out)
            out = self.act_trunkfinal(self.norm_trunkfinal(out))
            return self.policy(out)

    _TORCH["classes"] = {"TrunkDone": TrunkDone, "RankHead": RankHead, "FTNet": FTNet, "TinyBase": TinyBase}
    return _TORCH["classes"]


def load_base(device="cpu", tiny=False, seed=0):
    torch = torch_mod()
    if tiny:
        torch.manual_seed(seed)
        base = make_classes()["TinyBase"]().to(device)
        base.eval()
        return base, {"tiny": True}
    ck = H.check_checkpoint(H.CKPT_PATH, full_hash=False)
    base = P.load_model(device=device)
    return base, {"tiny": False, "checkpoint": ck, "katago_commit": H.KATAGO_COMMIT}


def build_net(device="cpu", tiny=False, k=None, hidden=None, dropout=None, seed=None):
    torch = torch_mod()
    cfg = CONFIG
    seed = cfg["seed"] if seed is None else seed
    base, info = load_base(device, tiny, seed)
    torch.manual_seed(seed)
    net = make_classes()["FTNet"](base, cfg["k_trainable_blocks"] if k is None else k,
                                  hidden=cfg["hidden"] if hidden is None else hidden,
                                  dropout=cfg["dropout"] if dropout is None else dropout).to(device)
    info["n_blocks"] = len(base.blocks)
    info["trainable_params"] = int(sum(p.numel() for _n, p in net.trainable_named()))
    info["head_params"] = int(sum(p.numel() for p in net.head.parameters()))
    return net, info


def unpack_bits_torch(bits):
    """uint8 [P, 993] tensor -> float32 [P, 22, 19, 19] on the same device (big bit order as np.packbits)."""
    torch = torch_mod()
    shifts = torch.arange(7, -1, -1, device=bits.device, dtype=torch.uint8)
    b = (bits.unsqueeze(-1) >> shifts) & 1
    return b.reshape(bits.shape[0], -1)[:, :N_BITS].reshape(-1, N_BIN, POS_LEN, POS_LEN).float()


def autocast_ctx(device, precision):
    torch = torch_mod()
    import contextlib
    if precision == "fp32":
        return contextlib.nullcontext()
    dt = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=torch.device(device).type, dtype=dt)


def save_trainable(net):
    """state of the trainable parameters (names -> CPU float32 tensors) + every head parameter."""
    sd = {n: p.detach().float().cpu().clone() for n, p in net.trainable_named()}
    for n, p in net.head.state_dict().items():
        sd[f"head.{n}"] = p.detach().float().cpu().clone()
    return sd


def load_trainable(net, sd):
    torch = torch_mod()
    own = dict(net.named_parameters())
    hs = {}
    for n, t in sd.items():
        if n.startswith("head."):
            hs[n[5:]] = t
        elif n in own:
            with torch.no_grad():
                own[n].copy_(t.to(own[n].device, own[n].dtype))
        else:
            raise FTError(f"checkpoint parameter {n} not in the model")
    net.head.load_state_dict(hs)
