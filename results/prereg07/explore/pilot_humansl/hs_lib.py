"""PREREG_07 pilot (competition 2): KataGo HumanSL scorer on our SGFs.

What it does
    For one game (our sgf_content string) and a list of rank profiles, replay the game with KataGo's python Board,
    build KataGo's input features at every position where the chosen side is to move, run the HumanSL network once per
    (position, profile), mask illegal moves, log-softmax, and return log p(actual move | profile). Per game and per
    profile it also returns the sum and the count of those log-probabilities (target player's moves; optionally the
    opponent's moves as a second side).

Fixed choices (documented, see README.md in this folder)
    * Network: b18c384nbt-humanv0.ckpt (KataGo v1.15.0 release asset), the plain "model" weights (not "swa_model"),
      exactly as KataGo issue #1154 loaded it when python matched the C++ engine. Loaded with torch.load(...,
      weights_only=True) after KataGo's load_model module registered its safe globals (defaultdict, float).
    * KataGo python code: third_party/katago (commit d91ea855, copied unmodified; SOURCE.txt).
    * Rules: Japanese, komi 6.5 (KataGo's GameState.RULES_JAPANESE: simple ko, territory scoring, seki tax, no suicide).
      Our SGFs carry no RU / KM / HA (only SZ[19] and moves); Yugen no Ma is a Nihon Ki-in server, so Japanese rules
      with 6.5 komi is the assumption. It only enters through 3 global inputs (komi/20 with the territory-scoring
      prisoner adjustment, territory scoring flag, seki tax flag).
    * Metadata: the C++ humanSLProfile preset (cpp/neuralnet/sgfmetadata.cpp makeBasicRankProfile): both players human,
      the same rank for both sides, ratedness unknown, byo-yomi 1200 s main + 5 x 30 s, source KGS, date 2020-03-01
      ("rank_<r>") or 2016-09-01 ("preaz_<r>"). Options: source FOX, any date, opponent rank "same" / "unknown" / a rank.
    * Illegal moves are masked to -inf before log-softmax (as the C++ engine does with -1e30): occupied points, the
      simple-ko point and every suicide (single- or multi-stone; Japanese rules forbid both). Pass is always legal.
      KataGo's python gamestate.get_model_outputs does NOT mask (softmax over all 362), so the parity test compares
      unmasked probabilities with it and masked ones with our own renormalisation.
    * Policy = main head, policy channel 0 (what gamestate's "policy0" and the C++ engine with rootPolicyOptimism 0 use).
    * No symmetry averaging (symmetry 0 = original orientation), as issue #1154's parity setting.

Safety: importing this module installs a guard: any open / listdir / scandir / stat of a path under data/local raises.
Nothing under data/local (validation questions and answers) is ever needed. No file is written by this module.
"""

import builtins
import datetime
import hashlib
import io
import math
import os
import re
import sys

sys.dont_write_bytecode = True   # never leave __pycache__ in third_party/ or results/

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
KATAGO_PY = os.path.join(REPO, "third_party", "katago", "python")
for _p in (KATAGO_PY, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------- guard: nothing under data/local is ever opened
_FORBIDDEN = os.path.normcase(os.path.abspath(os.path.join(REPO, "data", "local")))


class LocalDataRefused(RuntimeError):
    pass


def _forbidden(p):
    """True for this repository's data/local and for any path with the consecutive components data/local (another
    checkout or worktree). Component-wise on purpose: the plain substring test of kg_guard.py / u1_lib.py also
    refuses ...\\AppData\\Local\\... (the Windows temp folder), which is not validation data."""
    try:
        if isinstance(p, int):
            return False
        s = os.path.normcase(os.path.abspath(os.fspath(p)))
    except TypeError:
        return False
    if s == _FORBIDDEN or s.startswith(_FORBIDDEN + os.sep):
        return True
    parts = s.split(os.sep)
    return any(a == "data" and b == "local" for a, b in zip(parts, parts[1:]))


def install_guard():
    """Same guard as results/prereg06/explore/katago/kg_guard.py (own copy, stdlib only)."""
    if getattr(builtins, "_c10_hs_guard", False):
        return
    real_open, real_listdir, real_scandir, real_stat = builtins.open, os.listdir, os.scandir, os.stat

    def g_open(file, *a, **k):
        if _forbidden(file):
            raise LocalDataRefused(f"hs guard: refusing to open {file}")
        return real_open(file, *a, **k)

    def g_listdir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"hs guard: refusing to list {path}")
        return real_listdir(path)

    def g_scandir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"hs guard: refusing to scan {path}")
        return real_scandir(path)

    def g_stat(path, *a, **k):
        if _forbidden(path):
            raise LocalDataRefused(f"hs guard: refusing to stat {path}")
        return real_stat(path, *a, **k)

    builtins.open = g_open
    io.open = g_open
    os.listdir = g_listdir
    os.scandir = g_scandir
    os.stat = g_stat
    builtins._c10_hs_guard = True


install_guard()

from katago.game.board import Board  # noqa: E402
from katago.game.features import Features  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from katago.game.sgfmetadata import SGFMetadata  # noqa: E402

# ---------------------------------------------------------------- constants
KATAGO_COMMIT = "d91ea855110dae533f0aada947b2b7d78cc8a4e1"
CKPT_PATH = os.path.join(REPO, "data", "katago", "b18c384nbt-humanv0.ckpt")
CKPT_URL = "https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.ckpt"
CKPT_BYTES = 323752318
# measured locally on 2026-10-02 after the download reached 323,752,318 bytes; the GitHub API publishes no digest for
# this v1.15.0 asset ("digest": null), so this pins our copy, it does not authenticate it
CKPT_SHA256 = "84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27"
POS_LEN = 19
N_POLICY = POS_LEN * POS_LEN + 1      # 361 points + pass
PASS_POS = POS_LEN * POS_LEN
RULES = dict(GameState.RULES_JAPANESE)   # simple ko, territory scoring, seki tax, no suicide, komi 6.5
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]

# KataGo's rank scale (cpp/neuralnet/sgfmetadata.cpp getInverseRank): 9d=1 ... 1d=9, 1k=10 ... 20k=29
RANKS = [f"{k}k" for k in range(20, 0, -1)] + [f"{d}d" for d in range(1, 10)]   # weakest -> strongest
SOURCES = {"KGS": SGFMetadata.SOURCE_KGS, "FOX": SGFMetadata.SOURCE_FOX, "OGS": SGFMetadata.SOURCE_OGS}
PRESET_DATE = {"rank": datetime.date(2020, 3, 1), "preaz": datetime.date(2016, 9, 1)}

# The SGFs are "(;SZ[19];B[pd];W[dp]...)" (tools/t02_checks.FAST_SGF); anything else is reported, not guessed.
FAST_SGF = re.compile(r"\(;SZ\[19\]((?:;[BW]\[[a-s]{2}\])*)\)\s*")
MOVE_RE = re.compile(r";([BW])\[([a-s])([a-s])\]")


class HSError(RuntimeError):
    pass


# ---------------------------------------------------------------- rank profiles and metadata rows
def inverse_rank(rank):
    """'9d' -> 1, '1d' -> 9, '1k' -> 10, '20k' -> 29 (KataGo's scale)."""
    r = rank.lower()
    if r.endswith("d") and r[:-1].isdigit() and 1 <= int(r[:-1]) <= 9:
        return 10 - int(r[:-1])
    if r.endswith("k") and r[:-1].isdigit() and 1 <= int(r[:-1]) <= 20:
        return 9 + int(r[:-1])
    raise HSError(f"unknown rank {rank!r} (KataGo presets: 20k..1k, 1d..9d)")


class Profile:
    """One metadata condition. name 'rank_5k' / 'preaz_5k' equal the C++ humanSLProfile presets; other settings get
    the name '<source><yyyymmdd>_<rank>[_opp<x>]'."""

    def __init__(self, rank, source="KGS", date=None, opp="same", preset="rank"):
        inverse_rank(rank)
        if source not in SOURCES:
            raise HSError(f"source {source!r} not in {sorted(SOURCES)}")
        if opp not in ("same", "unknown"):
            inverse_rank(opp)
        self.rank = rank.lower()
        self.source = source
        self.date = date if date is not None else PRESET_DATE[preset]
        self.opp = opp if opp in ("same", "unknown") else opp.lower()
        is_preset = source == "KGS" and opp == "same" and self.date in PRESET_DATE.values()
        if is_preset:
            kind = "rank" if self.date == PRESET_DATE["rank"] else "preaz"
            self.name = f"{kind}_{self.rank}"
        else:
            self.name = f"{source.lower()}{self.date.strftime('%Y%m%d')}_{self.rank}" + \
                ("" if opp == "same" else f"_opp{self.opp}")

    @classmethod
    def parse(cls, name):
        """'rank_5k', 'preaz_2d', 'fox20220601_3d', 'kgs20200301_3d_oppunknown'."""
        m = re.fullmatch(r"(rank|preaz)_(\d+[kd])", name)
        if m:
            return cls(m.group(2), preset=m.group(1))
        m = re.fullmatch(r"(kgs|fox|ogs)(\d{8})_(\d+[kd])(?:_opp(unknown|\d+[kd]))?", name)
        if m:
            d = datetime.datetime.strptime(m.group(2), "%Y%m%d").date()
            return cls(m.group(3), source=m.group(1).upper(), date=d, opp=m.group(4) or "same")
        raise HSError(f"cannot parse profile name {name!r}")

    def sgfmeta(self, pla):
        """SGFMetadata for a position where pla (Board.BLACK / Board.WHITE) is to move; the target is pla."""
        inv = inverse_rank(self.rank)
        meta = SGFMetadata(
            inverseBRank=inv, inverseWRank=inv, bIsHuman=True, wIsHuman=True,
            gameIsUnrated=False, gameRatednessIsUnknown=True,
            tcIsUnknown=False, tcIsByoYomi=True, mainTimeSeconds=1200.0, periodTimeSeconds=30.0, byoYomiPeriods=5,
            gameDate=self.date, source=SOURCES[self.source])
        if self.opp != "same":
            other_unknown = self.opp == "unknown"
            other_inv = 0 if other_unknown else inverse_rank(self.opp)
            if pla == Board.BLACK:
                meta.inverseWRank, meta.wRankIsUnknown = other_inv, other_unknown
            else:
                meta.inverseBRank, meta.bRankIsUnknown = other_inv, other_unknown
        return meta

    def row(self, pla):
        return self.sgfmeta(pla).get_metadata_row(nextPlayer=pla, boardArea=POS_LEN * POS_LEN)

    def __repr__(self):
        return f"Profile({self.name})"


def profiles_from_names(names):
    return [Profile.parse(n) for n in names]


def metadata_table(profiles):
    """[N, 3, 192] float32: row [p, pla] for pla in (Board.BLACK=1, Board.WHITE=2); index 0 unused (zeros)."""
    t = np.zeros((len(profiles), 3, SGFMetadata.METADATA_INPUT_NUM_CHANNELS), dtype=np.float32)
    for i, p in enumerate(profiles):
        for pla in (Board.BLACK, Board.WHITE):
            t[i, pla] = p.row(pla)
    return t


# ---------------------------------------------------------------- games
def parse_moves(sgf):
    """-> [(pla, x, y)] with pla Board.BLACK / WHITE, x = column, y = row (SGF 'aa' = top-left = KataGo (0, 0)).
    Raises HSError for anything that is not '(;SZ[19];B[..];W[..]...)' alternating from black."""
    m = FAST_SGF.fullmatch(sgf)
    if not m:
        raise HSError("sgf_content is not the plain '(;SZ[19];B[..];W[..]...)' form")
    out = []
    for i, (c, a, b) in enumerate(MOVE_RE.findall(m.group(1))):
        pla = Board.BLACK if c == "B" else Board.WHITE
        if pla != (Board.BLACK if i % 2 == 0 else Board.WHITE):
            raise HSError(f"move {i + 1}: colours do not alternate from black")
        out.append((pla, ord(a) - 97, ord(b) - 97))
    return out


def legal_mask(board, pla):
    """[362] bool, Japanese rules: empty, not the simple-ko point, not suicide (single or multi stone); pass legal."""
    mask = np.zeros(N_POLICY, dtype=bool)
    mask[PASS_POS] = True
    ko = board.simple_ko_point
    bb = board.board
    for y in range(POS_LEN):
        base = board.loc(0, y)
        for x in range(POS_LEN):
            loc = base + x
            if bb[loc] != Board.EMPTY or loc == ko:
                continue
            if board.would_be_suicide(pla, loc):
                continue
            mask[y * POS_LEN + x] = True
    return mask


def featurize_game(sgf, target, sides=("target",), features=None, rules=None):
    """Replay one game; build KataGo inputs at every position where a wanted side is to move.

    target: 'B' or 'W' (the row's player colour). sides: subset of ('target', 'opponent').
    -> dict: bin uint8 [T, 22, 19, 19] (NCHW, all 0/1), glob float32 [T, 19], legal bool [T, 362],
       move int16 [T] (policy index of the move actually played), pla int8 [T] (1 B / 2 W), side int8 [T]
       (0 target, 1 opponent), turn int16 [T] (0-based move number), n_moves (moves in the game)."""
    features = features or Features({"version": 15}, POS_LEN)   # v10-v17: 22 spatial / 19 global channels
    rules = rules or RULES
    tgt = Board.BLACK if target == "B" else Board.WHITE
    want = set()
    if "target" in sides:
        want.add(tgt)
    if "opponent" in sides:
        want.add(Board.get_opp(tgt))
    if not want:
        raise HSError(f"sides {sides!r}: nothing to score")
    moves = parse_moves(sgf)
    board = Board(size=POS_LEN)
    boards = [board.copy()]
    hist = []
    nb, ng = features.bin_input_shape[0], features.global_input_shape[0]
    bins, globs, legal, mv, plas, side, turn = [], [], [], [], [], [], []
    row_bin = np.zeros((1, POS_LEN * POS_LEN, nb), dtype=np.float32)
    row_glob = np.zeros((1, ng), dtype=np.float32)
    for i, (pla, x, y) in enumerate(moves):
        if board.pla != pla:
            raise HSError(f"move {i + 1}: {pla} to play but the board expects {board.pla}")
        loc = board.loc(x, y)
        if pla in want:
            row_bin.fill(0.0)
            row_glob.fill(0.0)
            features.fill_row_features(board, pla, Board.get_opp(pla), boards, hist, len(hist), rules,
                                       row_bin, row_glob, 0)
            lm = legal_mask(board, pla)
            pos = y * POS_LEN + x
            if not lm[pos]:
                raise HSError(f"move {i + 1} ({x}, {y}) is illegal under the mask")
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
    out = {"bin": np.stack(bins) if T else np.zeros((0, nb, POS_LEN, POS_LEN), np.uint8),
           "glob": np.stack(globs) if T else np.zeros((0, ng), np.float32),
           "legal": np.stack(legal) if T else np.zeros((0, N_POLICY), bool),
           "move": np.asarray(mv, dtype=np.int16), "pla": np.asarray(plas, dtype=np.int8),
           "side": np.asarray(side, dtype=np.int8), "turn": np.asarray(turn, dtype=np.int16),
           "n_moves": len(moves)}
    return out


# ---------------------------------------------------------------- checkpoint and model
def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_checkpoint(path=CKPT_PATH, full_hash=True):
    """Size (and, by default, SHA-256) of the checkpoint against the pinned values -> record dict; raises on mismatch."""
    size = os.path.getsize(path)
    if size != CKPT_BYTES:
        raise HSError(f"{path}: {size} bytes, expected {CKPT_BYTES} (download unfinished or wrong file)")
    rec = {"path": os.path.relpath(path, REPO).replace("\\", "/"), "bytes": size}
    if full_hash:
        sha = sha256_file(path)
        if CKPT_SHA256 != "TO_BE_FILLED" and sha != CKPT_SHA256:
            raise HSError(f"{path}: sha256 {sha}, expected {CKPT_SHA256}")
        rec["sha256"] = sha
    return rec


# Every global the checkpoint's pickle references (pickletools scan of archive/data.pkl, 2026-10-02): nothing else is
# ever resolved by the restricted loader below. torch.FloatStorage / LongStorage are resolved by torch itself.
ALLOWED_GLOBALS = {("collections", "OrderedDict"), ("collections", "defaultdict"),
                   ("builtins", "float"), ("__builtin__", "float"), ("builtins", "set"), ("__builtin__", "set"),
                   ("torch._utils", "_rebuild_tensor_v2")}
CKPT_GLOBALS = ["__builtin__ float", "__builtin__ set", "collections OrderedDict", "collections defaultdict",
                "torch FloatStorage", "torch LongStorage", "torch._utils _rebuild_tensor_v2"]


def pickle_globals(path=CKPT_PATH):
    """Globals referenced by the checkpoint's pickle, found with pickletools WITHOUT executing it."""
    import pickletools
    import zipfile
    with zipfile.ZipFile(path) as z:
        pkl = [n for n in z.namelist() if n.endswith("/data.pkl")]
        if len(pkl) != 1:
            raise HSError(f"{path}: expected one data.pkl, found {pkl}")
        data = z.read(pkl[0])
    found, strs = set(), []
    for op, arg, _pos in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strs.append(arg)
        elif op.name == "GLOBAL":
            found.add(arg)
        elif op.name == "STACK_GLOBAL":
            found.add(" ".join(strs[-2:]))
    return sorted(found)


def _safe_pickle_module():
    import pickle
    import types

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in ALLOWED_GLOBALS:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"hs_lib restricted loader: global {module}.{name} is not allowed")

    def load(f, **kw):
        return RestrictedUnpickler(f, **kw).load()

    return types.SimpleNamespace(__name__="hs_lib_restricted_pickle", Unpickler=RestrictedUnpickler, load=load,
                                 UnpicklingError=pickle.UnpicklingError)


def load_state(path=CKPT_PATH):
    """-> (state dict, how). First torch.load(weights_only=True) after katago.train.load_model registered KataGo's safe
    globals (defaultdict, float). torch's weights-only unpickler refuses SETITEMS on a defaultdict (the checkpoint's
    running-metrics entries), which KataGo's own loader reports as 'run migrate_defaultdict_to_dict.py'. We do not
    rewrite the pinned file; instead, after checking that the pickle references only CKPT_GLOBALS (scan without
    execution), we load it with a restricted unpickler that resolves only ALLOWED_GLOBALS (plus torch's storage types)."""
    import pickle
    import torch
    import katago.train.load_model  # noqa: F401  (side effect: torch.serialization.add_safe_globals)
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
    except pickle.UnpicklingError as e:
        if "defaultdict" not in str(e):
            raise
    found = pickle_globals(path)
    if found != CKPT_GLOBALS:
        raise HSError(f"{path}: pickle globals {found} differ from the pinned list {CKPT_GLOBALS}")
    st = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_safe_pickle_module())
    return st, "restricted_unpickler"


def build_model(state, device="cpu", use_swa=False):
    """KataGo's Model from the checkpoint's own config; strict load of 'model' (default) or 'swa_model'."""
    import torch
    from katago.train.load_model import load_model_state_dict, load_swa_model_state_dict
    from katago.train.model_pytorch import Model
    model = Model(state["config"], POS_LEN)
    model.initialize()
    if use_swa:
        from torch.optim.swa_utils import AveragedModel
        sd = load_swa_model_state_dict(state)
        if sd is None:
            raise HSError("checkpoint has no swa_model")
        swa = AveragedModel(model)
        swa.load_state_dict(sd)
        model = swa.module
    else:
        model.load_state_dict(load_model_state_dict(state))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_model(path=CKPT_PATH, device="cpu", use_swa=False, check=True):
    """-> (model, state); state["_hs_load"] records how the checkpoint was unpickled."""
    if check:
        check_checkpoint(path, full_hash=False)
    state, how = load_state(path)
    model = build_model(state, device=device, use_swa=use_swa)
    state["_hs_load"] = how
    return model, state


# ---------------------------------------------------------------- inference
class Scorer:
    """Batched HumanSL policy evaluation for many positions x profiles.

    precision: 'fp32' (no autocast), 'fp16' / 'bf16' (torch.autocast on CUDA; on CPU only bf16 is allowed).
    batch: number of network evaluations per forward pass (positions x profiles).
    pad: pad the last, partial forward pass of every logp_moves call up to the full number of positions (copies of
      the last position, discarded afterwards), so that every forward pass has the same shape. Default: on for CUDA.
      Reason (measured 2026-10-03 on the RTX 4090, cudnn.benchmark on): every new input shape costs ~8.5-10.6 s of
      cuDNN autotuning, and without padding each game / chunk ends with a new shape. Rows are evaluated
      independently (eval mode), so padding never changes a real row's inputs; it only fixes the batch shape."""

    def __init__(self, model, profiles, device="cpu", precision="fp32", batch=256, channels_last=False, pad=None):
        import torch
        self.torch = torch
        self.model = model
        self.profiles = list(profiles)
        self.N = len(self.profiles)
        self.device = torch.device(device)
        self.precision = precision
        self.batch = int(batch)
        self.channels_last = bool(channels_last)
        self.pad = (self.device.type == "cuda") if pad is None else bool(pad)
        if precision not in ("fp32", "fp16", "bf16"):
            raise HSError(f"precision {precision!r}")
        if precision == "fp16" and self.device.type != "cuda":
            raise HSError("fp16 autocast is for CUDA only")
        if channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.meta = torch.tensor(metadata_table(self.profiles), device=self.device)   # [N, 3, 192]

    def _forward(self, binp, glob, meta):
        torch = self.torch
        if self.channels_last:
            binp = binp.contiguous(memory_format=torch.channels_last)
        if self.precision == "fp32":
            out = self.model(binp, glob, input_meta=meta)
        else:
            dt = torch.float16 if self.precision == "fp16" else torch.bfloat16
            with torch.autocast(device_type=self.device.type, dtype=dt):
                out = self.model(binp, glob, input_meta=meta)
        return out[0][0][:, 0, :].float()     # main head, policy channel 0 -> [B, 362] float32

    def policy_logits(self, bin_u8, glob, pla, profile_idx):
        """Raw policy logits for explicit (position, profile) pairs -> float32 numpy [B, 362]."""
        torch = self.torch
        with torch.inference_mode():
            b = torch.as_tensor(bin_u8, device=self.device).float()
            g = torch.as_tensor(glob, device=self.device).float()
            m = self.meta[torch.as_tensor(profile_idx, device=self.device).long(),
                          torch.as_tensor(pla, device=self.device).long()]
            return self._forward(b, g, m).cpu().numpy()

    def logp_moves(self, feats):
        """feats: one featurize_game dict (or a concatenation). -> float32 [T, N]: log p(played move) under each
        profile, illegal moves masked before the log-softmax."""
        torch = self.torch
        T, N = len(feats["move"]), self.N
        out = np.zeros((T, N), dtype=np.float32)
        if T == 0:
            return out
        per = max(1, self.batch // N)                  # positions per forward pass
        with torch.inference_mode():
            for s in range(0, T, per):
                e = min(T, s + per)
                P = e - s
                # rows of this pass; with padding, the last real row is repeated up to `per` rows (dropped below)
                rows = np.arange(s, e) if not self.pad or P == per else \
                    np.concatenate([np.arange(s, e), np.full(per - P, e - 1)])
                R = len(rows)
                b = torch.as_tensor(feats["bin"][rows], device=self.device).float()      # [R,22,19,19]
                g = torch.as_tensor(feats["glob"][rows], device=self.device).float()     # [R,19]
                pla = torch.as_tensor(feats["pla"][rows].astype(np.int64), device=self.device)
                lm = torch.as_tensor(feats["legal"][rows], device=self.device)           # [R,362]
                mv = torch.as_tensor(feats["move"][rows].astype(np.int64), device=self.device)
                # profile-major expansion: row (j, i) = profile j, position i
                bb = b.unsqueeze(0).expand(N, R, *b.shape[1:]).reshape(N * R, *b.shape[1:])
                gg = g.unsqueeze(0).expand(N, R, g.shape[1]).reshape(N * R, g.shape[1])
                mm = self.meta[:, pla, :].reshape(N * R, -1)                                # [N*R,192]
                logits = self._forward(bb, gg, mm).view(N, R, N_POLICY)
                logits = logits.masked_fill(~lm.unsqueeze(0), float("-inf"))
                lp = torch.log_softmax(logits, dim=-1)
                sel = lp.gather(2, mv.view(1, R, 1).expand(N, R, 1)).squeeze(2)            # [N,R]
                out[s:e] = sel[:, :P].t().cpu().numpy()
        if not np.isfinite(out).all():
            raise HSError("non-finite log-probability of a played move (mask or precision problem)")
        return out


def game_sums(logp, side):
    """-> sums [2, N] float64 (side 0 target, 1 opponent) and counts [2] int."""
    N = logp.shape[1]
    sums = np.zeros((2, N), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for s in (0, 1):
        m = side == s
        counts[s] = int(m.sum())
        if counts[s]:
            sums[s] = logp[m].astype(np.float64).sum(0)
    return sums, counts


def score_game(scorer, sgf, target, sides=("target",)):
    """Convenience: featurise + score one game -> dict(logp [T,N], side, turn, sums [2,N], counts [2])."""
    f = featurize_game(sgf, target, sides=sides)
    lp = scorer.logp_moves(f)
    sums, counts = game_sums(lp, f["side"])
    return {"logp": lp, "side": f["side"], "turn": f["turn"], "sums": sums, "counts": counts,
            "n_moves": f["n_moves"]}


# ---------------------------------------------------------------- KataGo's own single-position path (parity reference)
def gamestate_policy(model, moves, sgfmeta, rules=None):
    """KataGo's GameState.get_model_outputs on the position after `moves` ([(pla, x, y)]) -> unmasked policy0 [362]
    (softmax over all 362 outputs, KataGo's python behaviour)."""
    gs = GameState(POS_LEN, rules or RULES)
    for pla, x, y in moves:
        gs.play(pla, gs.board.loc(x, y))
    out = gs.get_model_outputs(model, sgfmeta=sgfmeta)
    return np.asarray(out["policy0"], dtype=np.float64), gs


def masked_renorm(p, legal):
    q = np.where(legal, p, 0.0)
    return q / q.sum()


def rank_index(rank):
    """Position of a KataGo rank in RANKS (0 = 20k ... 28 = 9d)."""
    return RANKS.index(rank.lower())


def other_compute_processes():
    """Rows of nvidia-smi's process table whose Type is exactly 'C' (pure compute): another job on the GPU.
    -> (rows, raw output). Desktop apps show as 'C+G' / 'G' and are ignored."""
    import subprocess
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    rows, in_proc = [], False
    for line in out.splitlines():
        if "Processes:" in line:
            in_proc = True
            continue
        if in_proc:
            m = re.match(r"\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(C|G|C\+G)\s+(.*?)\s+(\S+)\s*\|", line)
            if m and m.group(2) == "C":
                rows.append({"pid": int(m.group(1)), "name": m.group(3).strip()})
    return rows, out


def environment():
    import platform
    import torch
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
           "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
           "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        env.update(cuda_version=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                   gpu=torch.cuda.get_device_name(0))
    return env


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def finite_or_raise(x, what):
    if not np.all(np.isfinite(x)):
        raise HSError(f"{what}: non-finite values")
    return x


def log_softmax_np(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=axis, keepdims=True)
    return z - m - np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))


def nan_to_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
