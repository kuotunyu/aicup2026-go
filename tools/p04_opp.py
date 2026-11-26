"""PREREG_04 1.2 (competition 2): O4, the opponent-move count table (CPU only; plan.md 5 'tools/p04_opp.py').

    PY = .venv/Scripts/python.exe
    PY tools/p04_opp.py selftest  [--out-dir O]                            -> selftest.json
    PY tools/p04_opp.py count     [--out-dir O --workers 8 --stride S --limit N]
                                  -> data/prereg04/counts_fit.npz + results/prereg04/opp/count_report.json
    PY tools/p04_opp.py inner     [--out-dir O --counts C --workers 8 --stride S --limit N]
                                  -> results/prereg04/opp/inner_o4.npz + inner_report.json
    PY tools/p04_opp.py dupcheck  [--out-dir O]                            -> results/prereg04/opp/dupcheck.json
    PY tools/p04_opp.py val       [--out-dir O --workers 8]                -> results/prereg04/opp/val_o4.npz + val_report.json

Definition (frozen; ported verbatim from results/prereg04/design/evidence/B_proxy/feat.py, count.py and
score_inner.py, same arithmetic in the same order; only the file handling, the worker arguments and the two
variants that PREREG_04 registers are kept):
  cell of an empty point, seen from the player to move:
      cell = ((((pat * 8 + tac) * 7 + d_opp) * 3 + d_own) * 3 + phase)       557,928 cells
    pat   3x3 neighbourhood (empty / own / opponent / off-board), canonical under the 8 symmetries (1,107 classes)
    tac   1 * captures + 2 * saves (own group in atari) + 4 * ataris
    d_opp distance d = dx + dy + max(dx, dy) to the opponent's last move: <=2, 3, 4, 5-6, 7-10, >=11, none
    d_own distance to the mover's own previous move: <=3, 4-6, >=7 or none
    phase move index in the game (0-based, both colours) <30, <100, >=100
  counts (fit games only, tools/c10data.load_rows(part="fit"); the level is always the TARGET player's level):
    target moves (mover = the labelled player): C_t[level, played cell] += 1, A_t[level, every empty cell] += 1
    opponent moves (mover = the other player):  C_o, A_o the same way
  rates (alpha = 20; per side s in {t, o}):
    r(c)   = (C(c) + 1) / (A(c) + 1 / r0),  r0 = sum C / sum A          (pooled over the 10 levels)
    r_k(c) = (C_k(c) + 20) / (A_k(c) + 20 / r(c))
  per move:  log P_k = log r_k(cell of the played point) - log sum_{empty q} r_k(cell(q))
  per game:  O4 = sum over the OPPONENT's moves (opponent table), T4 = sum over the target's own moves (target
             table, reference only), each kept per phase and summed over the phases -> 10-vector (float64).
  The whole SGF is replayed (every ';B[xx]' / ';W[xx]' move of p02_moves.MOVE_RE; passes skipped), not the
  notebook's 250-position window. A move onto an occupied point is an anomaly; the replay stops there.

Subcommands
  selftest  Capture logic vs sgfmill on 150 fit games (every position before every move; 0 mismatches) and the
            features vs a slow per-point reference on >= 89 positions (0 mismatches); constants 1,107 / 557,928.
            selftest.json is committed with count_report.json; p04_rank_fuse.py check requires it passed.
  count     Fit games, 8 spawn workers (design: about 72 min). Pinned (PREREG_04 1.2; any miss = stop): 720,307
            games, target 79,985,811 moves, opponent 79,985,352, 0 anomalies, 557,928 cells, and the arrays equal
            to data/prereg04/design_scratch/counts_fit.npz when that file is there with the manifest's SHA-256.
            The encoded fit moves are also compared with the design's moves_fit.npz / moves_fit_moves.npy. The
            workers' int32 accumulators are guarded before the workers start (int32_preflight: 361 x each shard's
            encoded moves of one side < 2^32; on the fit games 7 workers is the fewest that passes, 8 is registered)
            and again in merge_shards (361 x counted shard moves < 2^32, no negative partial).
  inner     The 73,556 inner games in prereg02's inner_keys.npz row order -> o4, t4 [N, 10] float64 and the move
            counts; <= 1e-12 against the design's proxy_inner_all.npz (o_choice_a20 / t_choice_a20) when the counts
            are the design's arrays; the two tables are compared with counts_fit_tables.npy slices 4 / 1.
  dupcheck  The validation question games (data/local/rank_val_questions.csv only; no label file) against the fit
            games as exact move sequences (dupcheck.py): a match with a fit row of the same target colour ('original
            colour') or of the other target colour ('colour swapped': the same game from the other player's side);
            inner vs fit recorded the same way. >= 1 % of the validation slots matched either way -> status 'stop'
            (exit 1; PREREG_04 2.2 item 3: write a revision first).
  val       Only after results/prereg04/rank/frozen_config.json is committed and unchanged at HEAD (PREREG_04 1.2,
            2.4; no bypass, preview or not) and is a real frozen config (its format; smoke / preview / design_input
            false) whose selection contains O4 (2.6), with this tool committed and the registered counts (committed
            count_report.json, same SHA-256). Replays every validation slot with the question file's color_i as the
            target colour, rows in prereg02's val_keys.npz order; replay failures (no move, or an anomaly) must be 0,
            else stop. inner and val record the SHA-256 of the two rate tables (tables_sha256; tune freezes it).

Real mode = the registered outputs (defaults above): tools/p04_opp.py must be committed and unchanged, no
--stride/--limit, inner/val read the registered counts whose SHA-256 must equal the committed count_report.json.
--out-dir outside the repository gives a preview (scratch); inside the repository only the registered folders
are accepted. Every report records this file's SHA-256 (CRLF folded to LF), the frozen tools' hashes (checked
equal to HEAD: p02_*, c10data, decide, p03_rank_fuse), git HEAD, Python / numpy / scipy versions.
No path containing 'answer' is ever opened. No GPU, no score.py, no validation labels.
"""

import argparse
import csv
import glob
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from array import array

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c10data  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
from c10data import LEVELS  # noqa: E402
from p02_moves import MOVE_RE  # noqa: E402

# ================================================================ paths and registered numbers
RESULTS = os.path.join(REPO, "results")
DATA = os.path.join(REPO, "data")
TRAIN_DIR = os.path.join(DATA, "training")
LOCAL_DIR = os.path.join(DATA, "local")
SPLIT_CSV = os.path.join(RESULTS, "split_v1", "players_split.csv")
QUESTIONS = os.path.join(LOCAL_DIR, "rank_val_questions.csv")
P02_DUMPS = os.path.join(RESULTS, "prereg02", "rank", "dumps")
P04_ROOT = os.path.join(RESULTS, "prereg04")
OPP_OUT = os.path.join(P04_ROOT, "opp")
FROZEN_CONFIG = os.path.join(P04_ROOT, "rank", "frozen_config.json")
DESIGN = os.path.join(P04_ROOT, "design")
MANIFEST = os.path.join(DESIGN, "scratch_manifest.json")
DESIGN_SRC = os.path.join(DESIGN, "evidence", "B_proxy")
DESIGN_SCRATCH = os.path.join(DATA, "prereg04", "design_scratch")
COUNTS_DIR = os.path.join(DATA, "prereg04")
COUNTS_NAME = "counts_fit.npz"
COUNTS_REAL = os.path.join(COUNTS_DIR, COUNTS_NAME)
COUNT_REPORT = "count_report.json"
INNER_NAME, INNER_REPORT = "inner_o4.npz", "inner_report.json"
VAL_NAME, VAL_REPORT = "val_o4.npz", "val_report.json"
DUP_NAME = "dupcheck.json"
SELFTEST_NAME = "selftest.json"
OWN_FILE = "p04_opp.py"
FROZEN_FILES = p3.FROZEN_TOOL_FILES + ("p03_rank_fuse.py",)
PORTED_FROM = ("feat.py", "count.py", "score_inner.py", "prep.py", "t_feat.py", "t_feat2.py", "dupcheck.py")

PINNED_COUNT = {"games": 720307, "target_moves": 79985811, "opponent_moves": 79985352, "anomalies": 0,
                "n_cells": 557928}
N_PAT_PINNED = 1107
INNER_GAMES = c10data.EXPECTED["inner"]["games"]     # 73,556
VAL_QUESTIONS, VAL_SLOTS = p3.VAL_QUESTIONS_FULL, p3.VAL_GAMES_FULL   # 1,589 / 19,156
ALPHA = 20.0
TOL = 1e-12
DUP_STOP_SHARE = 0.01
SELFTEST_SEED = 1
SELFTEST_CAPTURE_GAMES = 150
SELFTEST_FEATURE_GAMES = 12
SELFTEST_FEATURE_STEP = 7
SELFTEST_MIN_POSITIONS = 89
SIDES = ("t", "o")                                   # table index 0 = target (T4), 1 = opponent (O4)
DESIGN_VARIANT = {"t": "t_choice_a20", "o": "o_choice_a20"}   # score_inner.py variant names
DESIGN_TABLE = {"t": 1, "o": 4}                      # score_inner.py build_tables slices
FORMAT = {"count": "c10-prereg04 opp count v1", "inner": "c10-prereg04 opp inner v1",
          "val": "c10-prereg04 opp val v1", "dupcheck": "c10-prereg04 opp dupcheck v1",
          "selftest": "c10-prereg04 opp selftest v1"}


class P04Error(RuntimeError):
    pass


ERRORS = (P04Error,) + p3.ERRORS


# ================================================================ feat.py (verbatim port)
SIZE = 19
NPT = 361
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
N_TAC, N_DOPP, N_DOWN, N_PHASE = 8, 7, 3, 3


def _neighbours():
    out = []
    for p in range(NPT):
        y, x = divmod(p, SIZE)
        nb = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < SIZE and 0 <= xx < SIZE:
                nb.append(yy * SIZE + xx)
        out.append(tuple(nb))
    return tuple(out)


NEI = _neighbours()


def _grid(code):
    g = np.zeros((3, 3), dtype=np.int64)
    for d, (dy, dx) in enumerate(OFFS):
        g[1 + dy, 1 + dx] = (code >> (2 * d)) & 3
    return g


def _enc(g):
    return int(sum((4 ** d) * int(g[1 + dy, 1 + dx]) for d, (dy, dx) in enumerate(OFFS)))


def _canon_table():
    # off-board masks that can occur on a 19x19 board
    masks = set()
    for y, x in itertools.product((0, 1, 18), repeat=2):
        masks.add(frozenset(d for d, (dy, dx) in enumerate(OFFS)
                            if not (0 <= y + dy < SIZE and 0 <= x + dx < SIZE)))
    canon = np.full(4 ** 8, -1, dtype=np.int64)
    for code in range(4 ** 8):
        g = _grid(code)
        off = frozenset(d for d, (dy, dx) in enumerate(OFFS) if g[1 + dy, 1 + dx] == 3)
        if off not in masks:
            continue
        best = None
        for k in range(4):
            for fl in (False, True):
                h = np.rot90(g, k)
                if fl:
                    h = np.fliplr(h)
                v = _enc(h)
                best = v if best is None or v < best else best
        canon[code] = best
    uniq = np.unique(canon[canon >= 0])
    lut = np.full(4 ** 8, -1, dtype=np.int64)
    ok = canon >= 0
    lut[ok] = np.searchsorted(uniq, canon[ok])
    return lut, len(uniq)


CANON, N_PAT = _canon_table()
N_CELLS = N_PAT * N_TAC * N_DOPP * N_DOWN * N_PHASE


def _dist_tables():
    ys, xs = np.divmod(np.arange(NPT), SIZE)
    dy = np.abs(ys[:, None] - ys[None, :])
    dx = np.abs(xs[:, None] - xs[None, :])
    d = dx + dy + np.maximum(dx, dy)
    opp = np.select([d <= 2, d == 3, d == 4, d <= 6, d <= 10], [0, 1, 2, 3, 4], 5)
    own = np.select([d <= 3, d <= 6], [0, 1], 2)
    opp = np.vstack([opp, np.full((1, NPT), 6)]).astype(np.int64)   # row NPT = no previous move
    own = np.vstack([own, np.full((1, NPT), 2)]).astype(np.int64)
    return opp, own


DOPP, DOWN = _dist_tables()
STRUCT = np.zeros((3, 3, 3), dtype=bool)
STRUCT[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
# (stone slice, neighbour slice) on [T,19,19] for the four directions
DIRS = [((slice(None), slice(1, None), slice(None)), (slice(None), slice(None, -1), slice(None))),
        ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
        ((slice(None), slice(None), slice(1, None)), (slice(None), slice(None), slice(None, -1))),
        ((slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None)))]


def phase_of(i):
    return 0 if i < 30 else (1 if i < 100 else 2)


def replay(moves, want):
    """moves: uint16 codes of one game. want(i, colour_value) -> keep a snapshot before move i.
    Returns (snapshots bytes list, meta list of (i, colour value 1/2, point, last_opp, last_own), n_anomalies)."""
    board = bytearray(NPT)
    snaps, meta = [], []
    last = {1: NPT, 2: NPT}
    anomalies = 0
    for i, code in enumerate(moves.tolist()):
        c = 2 if code >= 512 else 1
        p = code & 511
        o = 3 - c
        if board[p] != 0:
            anomalies += 1
            break
        if want(i, c):
            snaps.append(bytes(board))
            meta.append((i, c, p, last[o], last[c]))
        board[p] = c
        for q in NEI[p]:
            if board[q] == o:
                # does the group of q still have a liberty?
                stack, seen, free = [q], {q}, False
                while stack and not free:
                    x = stack.pop()
                    for y in NEI[x]:
                        v = board[y]
                        if v == 0:
                            free = True
                            break
                        if v == o and y not in seen:
                            seen.add(y)
                            stack.append(y)
                if not free:
                    for x in seen:
                        board[x] = 0
        last[c] = p
    return snaps, meta, anomalies


CANON32 = CANON.astype(np.int32)
W = SIZE + 2
PAD = W * W
NB4 = np.array([-W, W, -1, 1])
NB8 = np.array([dy * W + dx for dy, dx in OFFS])
INNER = ((np.arange(NPT) // SIZE + 1) * W + np.arange(NPT) % SIZE + 1)  # 19x19 flat -> padded flat


def cells_of(snaps, meta):
    """-> cell_e (cells of every empty point, position-major), pos_e (position of each), played cell [T],
    phase [T]. Features are computed at the empty points only."""
    T = len(snaps)
    b = np.frombuffer(b"".join(snaps), dtype=np.uint8).reshape(T, NPT)
    mc = np.array([m[1] for m in meta], dtype=np.uint8)
    own = b == mc[:, None]
    emp = b == 0
    opp = ~own & ~emp
    rel = own.astype(np.uint8) + 2 * opp.astype(np.uint8)
    prel = np.full((T, PAD), 3, dtype=np.uint8)
    prel[:, INNER] = rel
    lab_own, n_own = ndimage.label(own.reshape(T, SIZE, SIZE), structure=STRUCT)
    lab_opp, n_opp = ndimage.label(opp.reshape(T, SIZE, SIZE), structure=STRUCT)
    plo = np.zeros((T, PAD), dtype=np.int32)
    plp = np.zeros((T, PAD), dtype=np.int32)
    plo[:, INNER] = lab_own.reshape(T, NPT)
    plp[:, INNER] = lab_opp.reshape(T, NPT)
    prel = prel.ravel()
    plo = plo.ravel()
    plp = plp.ravel()
    te, pe = np.nonzero(emp)                       # empties, position-major
    q = te * PAD + INNER[pe]                       # padded flat index
    # liberties: distinct (group, empty point) pairs, deduplicated among the 4 sides of each empty point
    Lo = plo[q[:, None] + NB4]                     # [n_e, 4]
    Lp = plp[q[:, None] + NB4]
    libs_o = _libs4(Lo, n_own)
    libs_p = _libs4(Lp, n_opp)
    no = libs_o[Lo]
    npp = libs_p[Lp]
    tac = (npp == 1).any(1).astype(np.int64) + 2 * (no == 1).any(1) + 4 * (npp == 2).any(1)
    code = np.zeros(len(q), dtype=np.int32)
    for d in range(8):
        code |= prel[q + NB8[d]].astype(np.int32) << (2 * d)
    pat = CANON32[code]
    if (pat < 0).any():
        raise AssertionError("unrealisable 3x3 pattern")
    idx = np.array([m[0] for m in meta])
    phase = np.where(idx < 30, 0, np.where(idx < 100, 1, 2))
    lo_ = np.array([m[3] for m in meta])
    lw_ = np.array([m[4] for m in meta])
    dopp = DOPP[lo_[te], pe]
    down = DOWN[lw_[te], pe]
    cell = (((pat.astype(np.int64) * N_TAC + tac) * N_DOPP + dopp) * N_DOWN + down) * N_PHASE + phase[te]
    # played cell: the empty entry of each position whose point is the played point
    pp = np.array([m[2] for m in meta])
    hit = np.flatnonzero(pe == pp[te])
    if len(hit) != T or not np.array_equal(te[hit], np.arange(T)):
        raise AssertionError("played point not empty")
    return cell, te, cell[hit], phase


def _libs4(L, n):
    valid = L > 0
    valid[:, 1] &= L[:, 1] != L[:, 0]
    valid[:, 2] &= (L[:, 2] != L[:, 0]) & (L[:, 2] != L[:, 1])
    valid[:, 3] &= (L[:, 3] != L[:, 0]) & (L[:, 3] != L[:, 1]) & (L[:, 3] != L[:, 2])
    libs = np.bincount(L[valid], minlength=n + 1)
    libs[0] = 0
    return libs


# ================================================================ prep.py (verbatim port): SGF -> move codes
LET = {c: i for i, c in enumerate("abcdefghijklmnopqrs")}


def encode(sgf, buf):
    """move code (uint16) = colour_bit * 512 + point, colour_bit 0 = B, 1 = W; point = y * 19 + x with x = column
    letter index, y = row letter index of the SGF point (passes do not match the regex and are skipped)."""
    n = 0
    for m in MOVE_RE.finditer(sgf):
        c, p = m.group(1), m.group(2)
        buf.append((512 if c == "W" else 0) + LET[p[1]] * 19 + LET[p[0]])
        n += 1
    return n


def encode_one(sgf):
    buf = array("H")
    encode(sgf, buf)
    return np.frombuffer(buf, dtype=np.uint16).copy()


def target_colour_value(colour):
    """'B' -> 1, 'W' -> 2 (count.py: np.where(color == 'B', 1, 2))."""
    if colour not in ("B", "W"):
        raise P04Error(f"colour must be 'B' or 'W', got {colour!r}")
    return 1 if colour == "B" else 2


def load_part(part, train_dir=TRAIN_DIR, split_csv=SPLIT_CSV, local_dir=LOCAL_DIR, loader=None):
    """prep.py's run(part) in memory: training rows ONLY through c10data.load_rows (leak-checked against the local
    question / candidate files; no label file), moves encoded compactly, row order of load_rows."""
    for pth in (train_dir, split_csv, local_dir):
        refuse_answer_path(pth)
    t0 = time.time()
    loader = loader or c10data.load_rows
    rows = loader(train_dir, split_csv, part=part, local_dir=local_dir)
    buf = array("H")
    lens = array("i")
    rank, gid, pid, col = [], [], [], []
    for r in rows:
        lens.append(encode(r["sgf_content"], buf))
        rank.append(r["rank"])
        gid.append(r["game_id"])
        pid.append(r["player_id"])
        col.append(r["color"])
    n = len(rows)
    del rows
    lens = np.frombuffer(lens, dtype=np.int32).astype(np.int64)
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    out = {"moves": np.frombuffer(buf, dtype=np.uint16).copy(), "off": off, "rank": np.array(rank),
           "game_id": np.array(gid), "player_id": np.array(pid), "color": np.array(col),
           "label": np.array([LEVELS.index(x) for x in rank], dtype=np.int8)}
    print(f"{part}: {n} games, {len(out['moves'])} moves ({time.time() - t0:.0f}s)", flush=True)
    return out


# ================================================================ count.py (verbatim port; args instead of env)
_G = {}


def _count_init(path_npy, off, label, tcol):
    _G["mv"] = np.load(path_npy, mmap_mode="r")
    _G["off"], _G["label"], _G["tcol"] = off, label, tcol


def _count_shard(args):
    w, nw, stride, limit, tmp_dir = args
    off, label, tcol = _G["off"], _G["label"], _G["tcol"]
    G = len(off) - 1 if not limit else min(limit, len(off) - 1)
    games = list(range(w * stride, G, nw * stride))
    # this shard's moves, copied once into private memory (small: ~16 MB)
    mv = _G["mv"]
    acc = {k: np.zeros(10 * N_CELLS, dtype=np.int32) for k in ("C_t", "A_t", "C_o", "A_o")}
    t0 = time.time()
    n_games = n_t = n_o = anomalies = 0
    for j, g in enumerate(games):
        m = np.array(mv[off[g]:off[g + 1]])
        tc = int(tcol[g])
        lv = int(label[g]) * N_CELLS
        snaps, meta, an = replay(m, lambda i, c: True)
        anomalies += an
        for side in ("t", "o"):
            sel = [k for k, x in enumerate(meta) if (x[1] == tc) == (side == "t")]
            if not sel:
                continue
            cell, te, played, _ = cells_of([snaps[k] for k in sel], [meta[k] for k in sel])
            np.add.at(acc["C_" + side], played + lv, 1)
            np.add.at(acc["A_" + side], cell + lv, 1)
            if side == "t":
                n_t += len(sel)
            else:
                n_o += len(sel)
        n_games += 1
        if w == 0 and (j + 1) % 2000 == 0:
            print(f"worker 0: {j + 1}/{len(games)} games ({time.time() - t0:.0f}s)", flush=True)
    path = os.path.join(tmp_dir, f"_shard_{w}.npz")
    np.savez(path, **acc)
    del acc
    return path, {"games": n_games, "target_moves": n_t, "opponent_moves": n_o, "anomalies": anomalies,
                  "seconds": time.time() - t0}


def count_tables(moves, workers=8, stride=1, limit=0, tmp_root=None):
    """count.py main() without the file names: -> (tot {C_t, A_t, C_o, A_o} int64 [10, N_CELLS], stats list)."""
    if stride < 1 or limit < 0 or workers < 1:
        raise P04Error(f"stride >= 1, limit >= 0, workers >= 1 (got {stride}, {limit}, {workers})")
    pre = int32_preflight(moves, workers, stride, limit)          # before any worker starts (merge_shards re-checks)
    if pre["max_shard_bound"] >= INT32_WRAP:
        raise P04Error(f"{workers} workers: a shard holds up to {pre['max_shard_moves']} moves of one side, 361 x that "
                       f"= {pre['max_shard_bound']} >= 2^32, so an int32 accumulator could wrap past zero unseen; use "
                       f"more workers (the registered count uses 8; on the fit games 7 is the fewest that passes)")
    tmp = tempfile.mkdtemp(prefix="_p04count_", dir=tmp_root)
    try:
        npy = os.path.join(tmp, "moves_fit_moves.npy")
        np.save(npy, moves["moves"])
        off = moves["off"]
        label = moves["label"].astype(np.int64)
        tcol = np.where(moves["color"] == "B", 1, 2).astype(np.int64)
        print(f"fit games {len(off) - 1}, N_CELLS {N_CELLS}, workers {workers}, stride {stride}, limit {limit}",
              flush=True)
        with mp.get_context("spawn").Pool(workers, initializer=_count_init, initargs=(npy, off, label, tcol)) as pool:
            outs = pool.map(_count_shard, [(w, workers, stride, limit, tmp) for w in range(workers)])
        tot = merge_shards(outs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tot = {k: v.reshape(10, N_CELLS) for k, v in tot.items()}
    return tot, [o[1] for o in outs]


INT32_WRAP = 2 ** 32
MAX_EMPTY_POINTS = 361


def int32_bound(stats):
    """Upper bound of one cell's partial count in one shard: A_s gains at most 361 per move of side s (one per empty
    point), C_s at most 1, so every partial is <= 361 x that shard's moves of that side."""
    return MAX_EMPTY_POINTS * max(int(stats["target_moves"]), int(stats["opponent_moves"]))


def shard_moves(moves, workers, stride=1, limit=0):
    """Per worker, the (target, opponent) encoded moves of the games _count_shard gives it (range(w * stride, G,
    workers * stride)); a move is the target's when its colour is the game's target colour. The replay stops at an
    anomaly, so these are upper bounds of the moves the worker counts."""
    off, mv = moves["off"], moves["moves"]
    G = len(off) - 1 if not limit else min(limit, len(off) - 1)
    white = np.asarray(moves["color"]) == "W"
    out = []
    for w in range(workers):
        t = o = 0
        for g in range(w * stride, G, workers * stride):
            a, b = int(off[g]), int(off[g + 1])
            nw = int(np.count_nonzero(mv[a:b] >= 512))
            nt = nw if white[g] else (b - a) - nw
            t += nt
            o += (b - a) - nt
        out.append((t, o))
    return out


def int32_preflight(moves, workers, stride=1, limit=0):
    """int32_bound of every shard before the workers start (the encoded moves bound the counted ones)."""
    per = shard_moves(moves, workers, stride, limit)
    big = max((max(t, o) for t, o in per), default=0)
    return {"workers": workers, "max_shard_moves": big, "max_shard_bound": MAX_EMPTY_POINTS * big,
            "limit": INT32_WRAP}


def merge_shards(outs):
    """count.py's merge (int64 sum of the shards' int32 accumulators, shard order) with an int32 guard.
    The workers accumulate in int32 (count.py verbatim). A partial that passed 2^31 - 1 wraps; while the true partial
    is below 2^32 the wrapped value is negative. The guard therefore requires, per shard, 361 x moves < 2^32 (no
    partial can reach 2^32) and no negative entry (none reached 2^31): together they prove no partial overflowed.
    Otherwise stop (more workers make the shards smaller)."""
    tot = {k: np.zeros(10 * N_CELLS, dtype=np.int64) for k in ("C_t", "A_t", "C_o", "A_o")}
    for path, st in outs:
        bound = int32_bound(st)
        if bound >= INT32_WRAP:
            raise P04Error(f"shard {path}: 361 x {max(st['target_moves'], st['opponent_moves'])} moves = {bound} >= "
                           f"2^32; the int32 accumulators could wrap past zero unseen (use more workers)")
        with np.load(path) as s:
            for k in tot:
                part = s[k]
                if part.dtype != np.int32 or (part < 0).any():
                    raise P04Error(f"shard {path}: {k} is {part.dtype} with {int((part < 0).sum())} negative entries; "
                                   f"an int32 accumulator overflowed")
                tot[k] += part.astype(np.int64)
        os.remove(path)
    return tot


def count_summary(tot, stats, stride, limit, seconds):
    """count.py's count_report fields."""
    return {"games": sum(s["games"] for s in stats), "target_moves": sum(s["target_moves"] for s in stats),
            "opponent_moves": sum(s["opponent_moves"] for s in stats),
            "anomalies": sum(s["anomalies"] for s in stats), "n_cells": N_CELLS, "n_patterns": N_PAT,
            "cells_seen_target": int((tot["A_t"].sum(0) > 0).sum()),
            "cells_played_target": int((tot["C_t"].sum(0) > 0).sum()),
            "target_moves_by_level": tot["C_t"].sum(1).tolist(),
            "stride": stride, "limit": limit, "worker_seconds": [round(s["seconds"], 1) for s in stats],
            "wall_seconds": round(seconds, 1),
            "int32_guard": {"rule": "per shard 361 x max(target, opponent moves) < 2^32 and no negative int32 "
                                    "partial (merge_shards): no worker accumulator overflowed",
                            "max_shard_bound": max((int32_bound(s) for s in stats), default=0),
                            "limit": INT32_WRAP, "max_count": int(max(int(v.max()) for v in tot.values()))
                            if tot["A_t"].size else 0}}


def save_counts(path, tot):
    """np.savez of C_t, A_t, C_o, A_o in count.py's order (tmp + rename; the zip members carry no time stamp)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **{k: tot[k] for k in ("C_t", "A_t", "C_o", "A_o")})
    os.replace(tmp, path)


def load_counts(path):
    refuse_answer_path(path)
    if not os.path.exists(path):
        raise P04Error(f"missing counts file {path}")
    with np.load(path, allow_pickle=False) as z:
        tot = {k: z[k] for k in ("C_t", "A_t", "C_o", "A_o")}
    for k, v in tot.items():
        if v.shape != (10, N_CELLS) or v.dtype != np.int64:
            raise P04Error(f"{path}: {k} is {v.dtype} {v.shape}, expected int64 (10, {N_CELLS})")
    return tot


# ================================================================ score_inner.py (verbatim port, alpha-20 variants)
def build_tables(counts):
    """score_inner.build_tables' two registered slices: [2, N_CELLS, 10] float64, index 0 = target side
    (t_choice_a20, design slice 1), index 1 = opponent side (o_choice_a20, design slice 4)."""
    tabs = np.zeros((2, N_CELLS, 10), dtype=np.float64)
    info = {}
    for si, side in enumerate(SIDES):
        C, A = counts["C_" + side].astype(np.float64), counts["A_" + side].astype(np.float64)
        Cp, Ap = C.sum(0), A.sum(0)
        r0 = Cp.sum() / Ap.sum()
        r = (Cp + 1.0) / (Ap + 1.0 / r0)
        info[side] = {"r0": r0, "moves": float(Cp.sum())}
        tabs[si] = ((C + ALPHA) / (A + ALPHA / r[None, :])).T
    return tabs, info


def tables_record(tabs):
    """SHA-256 of the two rate tables as used (C-contiguous float64 bytes) with their shape and dtype: inner and val
    record it, tune freezes it, apply compares it (plan.md 5: the O4 tables' SHA in frozen_config.json)."""
    arr = np.ascontiguousarray(tabs)
    return {"tables_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "tables_array": {"shape": list(arr.shape), "dtype": str(arr.dtype), "order": "C",
                             "index": "[side (0 = target / T4, 1 = opponent / O4), cell, level]"}}


_S = {}


def _score_init(tab_path):
    _S["tab"] = np.load(tab_path, mmap_mode="r")


def score_game(m, tc, tab):
    """score_inner._game for the two registered variants: -> ll [2 sides, 3 phases, 10], n_moves [2, 3], anomalies.
    Side 0 = the target's own moves (T4, target table), side 1 = the opponent's moves (O4, opponent table)."""
    snaps, meta, an = replay(m, lambda i, c: True)
    ll = np.zeros((2, 3, 10))
    nm = np.zeros((2, 3), dtype=np.int64)
    for si, side in enumerate(SIDES):
        sel = [k for k, x in enumerate(meta) if (x[1] == tc) == (side == "t")]
        if not sel:
            continue
        cell, te, played, phase = cells_of([snaps[k] for k in sel], [meta[k] for k in sel])
        starts = np.flatnonzero(np.r_[True, te[1:] != te[:-1]])
        nm[si] = np.bincount(phase, minlength=3)
        T = tab[si]
        num = np.log(T[played])
        den = np.log(np.add.reduceat(T[cell], starts, axis=0))
        per = num - den
        for ph in range(3):
            mk = phase == ph
            if mk.any():
                ll[si, ph] = per[mk].sum(0)
    return ll, nm, an


def _score_chunk(items):
    tab = _S["tab"]
    return [(g,) + score_game(m, tc, tab) for g, m, tc in items]


def score_games(games, tabs, workers=8, tmp_root=None, chunk=200):
    """games: list of (row, uint16 moves, target colour value). -> ll [N, 2, 3, 10], nm [N, 2, 3], anomalies [N]
    for rows 0..N-1 (rows not given stay NaN / 0)."""
    n = 1 + max([g for g, _, _ in games], default=-1)
    ll = np.full((n, 2, 3, 10), np.nan)
    nm = np.zeros((n, 2, 3), dtype=np.int64)
    an = np.zeros(n, dtype=np.int64)
    tmp = tempfile.mkdtemp(prefix="_p04score_", dir=tmp_root)
    t0 = time.time()
    try:
        tab_path = os.path.join(tmp, "tables.npy")
        np.save(tab_path, tabs)
        chunks = [games[i:i + chunk] for i in range(0, len(games), chunk)]
        with mp.get_context("spawn").Pool(workers, initializer=_score_init, initargs=(tab_path,)) as pool:
            for k, res in enumerate(pool.imap_unordered(_score_chunk, chunks)):
                for g, a, b, c in res:
                    ll[g], nm[g], an[g] = a, b, c
                if (k + 1) % 20 == 0:
                    print(f"{k + 1}/{len(chunks)} chunks ({time.time() - t0:.0f}s)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ll, nm, an


def side_vectors(ll):
    """[N, 2, 3, 10] -> (t4, o4) [N, 10]: the phase sum exactly as fuse.py (llv[:, j].sum(1))."""
    return ll[:, 0].sum(1), ll[:, 1].sum(1)


# ================================================================ provenance, paths, git
def refuse_answer_path(path):
    """Never open (or write) a path that looks like a validation label file (anything containing 'answer')."""
    if path is not None and "answer" in os.path.abspath(str(path)).lower():
        raise P04Error(f"refusing the path {path}: no path containing 'answer' is ever used by p04_opp")
    return path


def sha256_file(path):
    refuse_answer_path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    env = p3.environment()
    env["executable"] = sys.executable
    return env


def tool_state(repo=REPO):
    """This file and the frozen tools: SHA-256 with CRLF folded to LF, equality with HEAD, git HEAD, dirty files."""
    ft = p3.frozen_tools_check()                      # p02_rank_*, p02_moves, c10data, decide == HEAD (refuses)
    b = p3.head_blob("tools/p03_rank_fuse.py")
    work = p3.sha256_text(os.path.join(HERE, "p03_rank_fuse.py"))
    if b is None or p3.sha256_bytes_text(b) != work:
        raise P04Error("tools/p03_rank_fuse.py differs from HEAD; it is never edited (PREREG_04 2.1)")
    # the other tools/p02_*.py (p02_player_*) as well, like p04_rank_fuse.frozen_files(): every p02_* == HEAD
    for f in sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "p02_*.py"))):
        if f in ft["files_sha256"]:
            continue
        w, hb = p3.sha256_text(os.path.join(HERE, f)), p3.head_blob("tools/" + f)
        if hb is None or p3.sha256_bytes_text(hb) != w:
            raise P04Error(f"tools/{f} differs from HEAD; the p02 tools are never edited (PREREG_04 2.1)")
        ft["files_sha256"][f] = w
    p3.imported_from_tools()
    if not p3.same_path(c10data.__file__, os.path.join(HERE, "c10data.py")) or \
            not p3.same_path(p3.__file__, os.path.join(HERE, "p03_rank_fuse.py")):
        raise P04Error("c10data / p03_rank_fuse imported from somewhere else than tools/")
    rels = ["tools/" + f for f in sorted(set((OWN_FILE,) + FROZEN_FILES) | set(ft["files_sha256"]))]
    # binary: text mode strips the output, which would cut the leading status blank of the first line
    rc, st = p3.git(["status", "--porcelain", "--untracked-files=all", "--"] + rels, binary=True, repo=repo)
    lines = st.decode("utf-8", "replace").splitlines() if rc == 0 else None
    files = {OWN_FILE: p3.sha256_text(os.path.abspath(__file__))}
    files.update(ft["files_sha256"])
    files["p03_rank_fuse.py"] = work
    design = {}
    for f in PORTED_FROM:
        p = os.path.join(DESIGN_SRC, f)
        design[f] = p3.sha256_text(p) if os.path.exists(p) else None
    return {"git_head": p3.git_head(),
            "dirty_or_untracked": ([ln[3:] for ln in lines if ln.strip()] if lines is not None else None),
            "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": files, "frozen_equal_to_head": True,
            "ported_from_design_sha256": design}


def require_own_committed(tools):
    """Real mode: tools/p04_opp.py (and the frozen tools) committed and unchanged."""
    dirty = tools["dirty_or_untracked"]
    if tools["git_head"] is None or dirty is None or dirty:
        raise P04Error(f"real mode needs the tools committed and unchanged (HEAD {tools['git_head']}, uncommitted "
                       f"{dirty}); use --out-dir outside the repository for a preview")


def git_tracked_clean(path, repo=REPO):
    """(ok, why): the file exists, is tracked by git in `repo` and has no change in the working tree."""
    if not os.path.exists(path):
        return False, "missing"
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(repo)).replace(os.sep, "/")
    if rel.startswith(".."):
        return False, "outside the repository"
    rc, _ = p3.git(["ls-files", "--error-unmatch", "--", rel], repo=repo)
    if rc != 0:
        return False, "not committed"
    rc, st = p3.git(["status", "--porcelain", "--", rel], repo=repo)
    if rc != 0:
        return False, "git status failed"
    if st:
        return False, f"changed in the working tree ({st[:80]})"
    return True, "committed and unchanged"


FROZEN_CONFIG_FORMAT = "c10-prereg04 rank frozen_config v1"      # p04_rank_fuse.FROZEN_FORMAT


def require_frozen_config(path=FROZEN_CONFIG, repo=REPO):
    """PREREG_04 1.2 / 2.4 / 2.6: the validation O4 exists only after frozen_config.json is committed (no bypass), and
    only for a real frozen config (p04_rank_fuse tune: the registered format, not smoke / preview / design input)
    whose selected combination contains O4 (2.6: val runs when the selection contains O4)."""
    ok, why = git_tracked_clean(path, repo)
    if not ok:
        raise P04Error(f"val refused: {path} is {why}; the validation questions' O4 may be computed only after "
                       f"frozen_config.json is committed (PREREG_04 1.2, 2.4)")
    try:
        with open(path, encoding="utf-8") as f:
            fc = json.load(f)
    except (OSError, ValueError) as e:
        raise P04Error(f"val refused: {path} is not readable JSON ({e})")
    if not isinstance(fc, dict) or fc.get("format") != FROZEN_CONFIG_FORMAT:
        raise P04Error(f"val refused: {path} is not a {FROZEN_CONFIG_FORMAT!r} file")
    flags = {k: fc.get(k) for k in ("smoke", "preview", "design_input")}
    if any(v is not False for v in flags.values()):
        raise P04Error(f"val refused: {path} is not a real frozen config (smoke / preview / design_input {flags})")
    sources = (fc.get("combo") or {}).get("sources")
    if not isinstance(sources, list) or "O4" not in sources:
        raise P04Error(f"val refused: the frozen selection {(fc.get('combo') or {}).get('id')!r} ({sources}) does not "
                       f"contain O4; the validation O4 is computed only for a selection with O4 (PREREG_04 2.6)")
    rc, head = p3.git(["rev-parse", "HEAD"], repo=repo)
    return {"path": os.path.relpath(path, repo).replace(os.sep, "/"), "sha256": sha256_file(path),
            "git_head": head if rc == 0 else None, "state": why, "combo": (fc.get("combo") or {}).get("id")}


def manifest_sha(name):
    """SHA-256 of a design scratch file as recorded in results/prereg04/design/scratch_manifest.json."""
    if not os.path.exists(MANIFEST):
        return None
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    for e in man.get("large_files", []):
        if e.get("file") == f"data/prereg04/design_scratch/{name}":
            return e.get("sha256")
    return None


def design_file(name):
    """(path or None, record): a design scratch file usable as a reference only when its SHA-256 equals the
    manifest's."""
    path = os.path.join(DESIGN_SCRATCH, name)
    want = manifest_sha(name)
    rec = {"file": f"data/prereg04/design_scratch/{name}", "manifest_sha256": want}
    if not os.path.exists(path):
        rec["state"] = "absent"
        return None, rec
    got = sha256_file(path)
    rec["sha256"] = got
    if want is None or got != want:
        rec["state"] = "sha256 differs from the manifest; not used"
        return None, rec
    rec["state"] = "present, sha256 equal to the manifest"
    return path, rec


def write_json(path, obj):
    refuse_answer_path(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def save_npz(path, arrays):
    """np.savez_compressed (tmp + rename); the zip members carry no time stamp, so the bytes are reproducible."""
    refuse_answer_path(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


class Ctx:
    pass


def resolve_out(a, default_dir, cmd):
    """Registered location (real mode) or a folder outside the repository (preview); nothing else in the repo."""
    c = Ctx()
    c.cmd = cmd
    c.out_dir = os.path.abspath(a.out_dir) if getattr(a, "out_dir", None) else default_dir
    refuse_answer_path(c.out_dir)
    in_repo = p3.is_inside(c.out_dir, REPO)
    if in_repo and not p3.same_path(c.out_dir, default_dir):
        raise P04Error(f"--out-dir inside the repository must be {default_dir} (got {c.out_dir}); use a folder "
                       f"outside the repository for a preview")
    c.real = in_repo
    c.preview = not in_repo
    subset = bool(getattr(a, "stride", 1) != 1 or getattr(a, "limit", 0))
    if c.real and subset:
        raise P04Error("--stride / --limit only for a preview (--out-dir outside the repository)")
    c.subset = subset
    c.tools = tool_state()
    if c.real:
        require_own_committed(c.tools)
    os.makedirs(c.out_dir, exist_ok=True)
    return c


def report_head(c, a):
    return {"format": FORMAT[c.cmd], "command": c.cmd, "argv": getattr(a, "argv_list", None),
            "preview": c.preview, "registered_output": c.real, "created": p3.now_iso(), "tool": c.tools,
            "environment": environment(), "workers": getattr(a, "workers", None)}


# ================================================================ selftest (t_feat.py / t_feat2.py ports)
def capture_mismatches(m):
    """t_feat.py part 1 for one game: every snapshot before move i equal to sgfmill's board -> 0 / 1, anomalies."""
    from sgfmill import boards
    snaps, meta, an = replay(m, lambda i, c: True)
    B = boards.Board(19)
    ok = True
    for i, code in enumerate(m.tolist()):
        if i >= len(snaps):
            break
        c = "w" if code >= 512 else "b"
        p = code & 511
        y, x = divmod(p, 19)
        s = np.frombuffer(snaps[i], dtype=np.uint8).reshape(19, 19)
        for yy in range(19):
            for xx in range(19):
                v = B.get(18 - yy, xx)  # sgfmill row 0 = bottom
                e = 0 if v is None else (1 if v == "b" else 2)
                if s[yy, xx] != e:
                    ok = False
        if not ok:
            break
        B.play(18 - y, x, c)
    return int(not ok), an


def ref_cell(snap, m):
    """t_feat2.py's slow per-point reference: {empty point: cell} for one position."""
    i, c, p, lo, lw = m
    b = np.frombuffer(snap, dtype=np.uint8).reshape(19, 19)

    def rel(y, x):
        if not (0 <= y < 19 and 0 <= x < 19):
            return 3
        v = b[y, x]
        return 0 if v == 0 else (1 if v == c else 2)

    def group(y, x):
        col = b[y, x]
        st = [(y, x)]
        seen = {(y, x)}
        libs = set()
        while st:
            a, bb = st.pop()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = a + dy, bb + dx
                if 0 <= yy < 19 and 0 <= xx < 19:
                    if b[yy, xx] == 0:
                        libs.add((yy, xx))
                    elif b[yy, xx] == col and (yy, xx) not in seen:
                        seen.add((yy, xx))
                        st.append((yy, xx))
        return len(libs)

    out = {}
    for y in range(19):
        for x in range(19):
            if b[y, x]:
                continue
            g = np.zeros((3, 3), dtype=np.int64)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy or dx:
                        g[1 + dy, 1 + dx] = rel(y + dy, x + dx)
            pat = CANON[_enc(g)]
            cap = sav = ata = 0
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r = rel(y + dy, x + dx)
                if r in (1, 2):
                    L = group(y + dy, x + dx)
                    if r == 2 and L == 1:
                        cap = 1
                    if r == 1 and L == 1:
                        sav = 1
                    if r == 2 and L == 2:
                        ata = 1
            tac = cap + 2 * sav + 4 * ata
            pt = y * 19 + x
            dopp = DOPP[lo, pt]
            down = DOWN[lw, pt]
            ph = phase_of(i)
            out[pt] = (((pat * 8 + tac) * 7 + dopp) * 3 + down) * 3 + ph
    return out


def feature_mismatches(m, tc, step=SELFTEST_FEATURE_STEP):
    """t_feat2.py for one game, both sides (the design checked the target side): -> (positions, mismatches)."""
    ncheck = nbad = 0
    for side in ("t", "o"):
        snaps, meta, an = replay(m, lambda i, c: (c == tc) == (side == "t"))
        if not meta:
            continue
        cell, te, played, ph = cells_of(snaps, meta)
        for t in range(0, len(meta), step):
            ref = ref_cell(snaps[t], meta[t])
            mine = cell[te == t]
            pts = np.flatnonzero(np.frombuffer(snaps[t], dtype=np.uint8) == 0)
            got = dict(zip(pts.tolist(), mine.tolist()))
            ncheck += 1
            if got != ref:
                nbad += 1
            if ref[meta[t][2]] != played[t]:
                nbad += 1
    return ncheck, nbad


def run_selftest(fit):
    off = fit["off"]
    G = len(off) - 1
    rng = np.random.default_rng(SELFTEST_SEED)
    gs = rng.choice(G, 300, replace=False)
    t0 = time.time()
    cap_bad = cap_an = 0
    for g in gs[:SELFTEST_CAPTURE_GAMES]:
        b, an = capture_mismatches(fit["moves"][off[g]:off[g + 1]])
        cap_bad += b
        cap_an += an
    t1 = time.time()
    npos = nbad = 0
    for g in gs[:SELFTEST_FEATURE_GAMES]:
        tc = 1 if fit["color"][g] == "B" else 2
        a, b = feature_mismatches(fit["moves"][off[g]:off[g + 1]], tc)
        npos += a
        nbad += b
    t2 = time.time()
    rec = {"games_drawn": "numpy default_rng(1).choice(fit games, 300, replace=False), fit in load_rows order",
           "capture": {"games": SELFTEST_CAPTURE_GAMES, "games_with_mismatch": cap_bad, "anomalies": cap_an,
                       "seconds": round(t1 - t0, 1)},
           "features": {"games": SELFTEST_FEATURE_GAMES, "both_sides": True, "every_nth_position": SELFTEST_FEATURE_STEP,
                        "positions": npos, "mismatches": nbad, "min_positions": SELFTEST_MIN_POSITIONS,
                        "seconds": round(t2 - t1, 1)},
           "constants": {"n_patterns": N_PAT, "n_cells": N_CELLS}}
    problems = []
    if cap_bad or cap_an:
        problems.append(f"capture: {cap_bad} games differ from sgfmill, {cap_an} anomalies")
    if npos < SELFTEST_MIN_POSITIONS or nbad:
        problems.append(f"features: {nbad} mismatches on {npos} positions")
    if N_PAT != N_PAT_PINNED or N_CELLS != PINNED_COUNT["n_cells"]:
        problems.append(f"constants {N_PAT} / {N_CELLS}")
    rec["problems"] = problems
    rec["passed"] = not problems
    return rec


def cmd_selftest(a):
    c = resolve_out(a, OPP_OUT, "selftest")
    import sgfmill
    fit = load_part("fit")
    rep = report_head(c, a)
    rep["environment"]["sgfmill"] = getattr(sgfmill, "__version__", None)
    rep.update(run_selftest(fit))
    write_json(os.path.join(c.out_dir, SELFTEST_NAME), rep)
    print(json.dumps({k: rep[k] for k in ("capture", "features", "passed")}), flush=True)
    return 0 if rep["passed"] else 1


# ================================================================ count
def compare_arrays(mine, ref, names):
    """{name: equal?} plus the first difference."""
    out = {}
    for k in names:
        a, b = np.asarray(mine[k]), np.asarray(ref[k])
        out[k] = bool(a.shape == b.shape and a.dtype.kind == b.dtype.kind and np.array_equal(a, b))
    return out


def moves_seam(fit):
    """The encoded fit moves vs the design's moves_fit.npz (+ moves_fit_moves.npy), array by array."""
    path, rec = design_file("moves_fit.npz")
    if path is None:
        return rec, True
    with np.load(path, allow_pickle=False) as z:
        ref = {k: z[k] for k in ("off", "rank", "game_id", "player_id", "color", "label")}
    npy, rec_npy = design_file("moves_fit_moves.npy")
    rec["moves_npy"] = rec_npy
    ref["moves"] = np.load(npy, mmap_mode="r") if npy else None
    names = ["off", "rank", "game_id", "player_id", "color", "label"] + (["moves"] if npy else [])
    eq = compare_arrays(fit, ref, names)
    rec["arrays_equal"] = eq
    return rec, all(eq.values())


def cmd_count(a):
    c = resolve_out(a, OPP_OUT, "count")
    npz_dir = COUNTS_DIR if c.real else c.out_dir
    npz_path = os.path.join(npz_dir, COUNTS_NAME)
    rep_path = os.path.join(c.out_dir, COUNT_REPORT)
    if c.real and (os.path.exists(rep_path) or os.path.exists(npz_path)):
        raise P04Error(f"{rep_path} or {npz_path} already exists; the registered count runs once (remove it "
                       f"deliberately to rerun)")
    os.makedirs(npz_dir, exist_ok=True)
    t0 = time.time()
    fit = load_part("fit")
    seam, seam_ok = moves_seam(fit)
    tot, stats = count_tables(fit, workers=a.workers, stride=a.stride, limit=a.limit, tmp_root=npz_dir)
    summ = count_summary(tot, stats, a.stride, a.limit, time.time() - t0)
    save_counts(npz_path, tot)
    rep = report_head(c, a)
    rep.update(summ)
    rep["counts_file"] = (os.path.relpath(npz_path, REPO).replace(os.sep, "/") if c.real else npz_path)
    rep["counts_sha256"] = sha256_file(npz_path)
    rep["counts_bytes"] = os.path.getsize(npz_path)
    rep["fit_rows_load"] = {"loader": "tools/c10data.load_rows(part='fit')", "games": len(fit["off"]) - 1,
                            "moves": int(len(fit["moves"]))}
    rep["design_moves_seam"] = seam
    problems = [] if seam_ok else ["encoded fit moves differ from the design's moves_fit.npz"]
    full = not c.subset
    if full:
        pins = {k: (summ[k], v, summ[k] == v) for k, v in PINNED_COUNT.items()}
        rep["pinned"] = {k: {"got": g, "registered": v, "equal": e} for k, (g, v, e) in pins.items()}
        problems += [f"{k}: {g} != {v}" for k, (g, v, e) in pins.items() if not e]
        dpath, drec = design_file(COUNTS_NAME)
        if dpath is not None:
            with np.load(dpath, allow_pickle=False) as z:
                ref = {k: z[k] for k in z.files}
            drec["arrays_equal"] = compare_arrays(tot, ref, ("C_t", "A_t", "C_o", "A_o"))
            drec["file_sha256_equal"] = drec.get("sha256") == rep["counts_sha256"]
            if not all(drec["arrays_equal"].values()):
                problems.append("count arrays differ from the design's counts_fit.npz")
        rep["design_counts"] = drec
    else:
        rep["pinned"] = "not applied (subset preview: --stride / --limit)"
    rep["problems"] = problems
    rep["status"] = "ok" if not problems else "stop"
    write_json(rep_path, rep)
    print(json.dumps({k: rep[k] for k in ("games", "target_moves", "opponent_moves", "anomalies", "n_cells",
                                         "counts_sha256", "status", "wall_seconds")}), flush=True)
    return 0 if not problems else 1


# ================================================================ inner
def registered_counts(c, counts_arg):
    """Real mode: data/prereg04/counts_fit.npz whose SHA-256 equals the committed count_report.json (status ok)."""
    rep_path = os.path.join(OPP_OUT, COUNT_REPORT)
    if counts_arg and not p3.same_path(counts_arg, COUNTS_REAL):
        raise P04Error(f"--counts: real mode uses {COUNTS_REAL} only (preview with --out-dir outside the repository)")
    ok, why = git_tracked_clean(rep_path)
    if not ok:
        raise P04Error(f"{rep_path} is {why}; run count and commit its report first")
    with open(rep_path, encoding="utf-8") as f:
        rep = json.load(f)
    if rep.get("status") != "ok" or rep.get("preview") is not False:
        raise P04Error(f"{rep_path}: status {rep.get('status')!r}, preview {rep.get('preview')!r}; needs a real, ok count")
    got = sha256_file(COUNTS_REAL)
    if got != rep.get("counts_sha256"):
        raise P04Error(f"{COUNTS_REAL}: sha256 {got[:12]} differs from count_report.json {str(rep.get('counts_sha256'))[:12]}")
    return COUNTS_REAL, {"file": "data/prereg04/counts_fit.npz", "sha256": got, "count_report": rep_path,
                         "count_report_sha256": sha256_file(rep_path), "equal_to_design_arrays":
                         all((rep.get("design_counts") or {}).get("arrays_equal", {"x": False}).values())}


def counts_for(c, a):
    if c.real:
        path, rec = registered_counts(c, a.counts)
    else:
        path = os.path.abspath(a.counts) if a.counts else COUNTS_REAL
        rec = {"file": path, "sha256": sha256_file(path)}
    dsha = manifest_sha(COUNTS_NAME)
    rec["equal_to_design_file_sha256"] = rec["sha256"] == dsha
    arrays_are_design = rec["equal_to_design_file_sha256"] or bool(rec.get("equal_to_design_arrays"))
    return path, rec, arrays_are_design


def tables_seam(tabs):
    """The two tables vs the design's counts_fit_tables.npy slices (1 = t_choice_a20, 4 = o_choice_a20)."""
    path, rec = design_file("counts_fit_tables.npy")
    if path is None:
        return rec, True
    ref = np.load(path, mmap_mode="r")
    eq = {}
    for si, side in enumerate(SIDES):
        eq[DESIGN_VARIANT[side]] = bool(np.array_equal(tabs[si], ref[DESIGN_TABLE[side]]))
    rec["tables_equal"] = eq
    return rec, all(eq.values())


def inner_rows(inner, keys):
    """load_rows(part='inner') rows placed in inner_keys.npz order; player and colour must agree."""
    pos = {(r, g): i for i, (r, g) in enumerate(zip(inner["rank"].tolist(), inner["game_id"].tolist()))}
    if len(pos) != len(inner["rank"]):
        raise P04Error("inner rows: (rank, game_id) not unique")
    idx = []
    for r, g, p, col in zip(keys["rank"], keys["game_id"], keys["player_id"], keys["color"]):
        i = pos.get((r, g))
        if i is None:
            raise P04Error(f"inner_keys row ({r}, {g}) not in load_rows(part='inner')")
        if str(inner["player_id"][i]) != p or str(inner["color"][i]) != col:
            raise P04Error(f"inner_keys row ({r}, {g}): player / colour differ from load_rows")
        idx.append(i)
    if len(set(idx)) != len(idx) or len(idx) != len(inner["rank"]):
        raise P04Error("inner_keys does not cover load_rows(part='inner') one to one")
    return np.array(idx, dtype=np.int64)


def subset_rows(n, stride, limit):
    rows = list(range(0, n, stride))
    return rows[:limit] if limit else rows


def design_inner_compare(keys, rows, ll, t4, o4, nm):
    """o4 / t4 (and the per-phase sums, move counts) vs the design's proxy_inner_all.npz."""
    path, rec = design_file("proxy_inner_all.npz")
    if path is None:
        return rec, None
    with np.load(path, allow_pickle=False) as z:
        V = [str(v) for v in z["variants"]]
        pos = {(r, g): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
        idx = np.array([pos[(keys["rank"][i], keys["game_id"][i])] for i in rows])
        if not np.array_equal(z["player_id"][idx], np.array([keys["player_id"][i] for i in rows])):
            raise P04Error("proxy_inner_all.npz player ids differ from inner_keys")
        zll = z["ll"][idx]
        znm = z["n_moves"][idx]
    r = np.array(rows)
    out = {}
    for si, side in enumerate(SIDES):
        v = V.index(DESIGN_VARIANT[side])
        mine = (t4 if side == "t" else o4)[r]
        ref = zll[:, v].sum(1)
        out[DESIGN_VARIANT[side]] = {"max_abs_diff": float(np.max(np.abs(mine - ref))) if len(r) else 0.0,
                                     "bit_identical": bool(np.array_equal(mine, ref)),
                                     "per_phase_bit_identical": bool(np.array_equal(ll[r, si], zll[:, v]))}
    out["n_moves_equal"] = bool(np.array_equal(nm[r], znm))
    out["rows_compared"] = int(len(r))
    rec["compare"] = out
    ok = out["n_moves_equal"] and all(out[DESIGN_VARIANT[s]]["max_abs_diff"] <= TOL for s in SIDES)
    return rec, ok


def cmd_inner(a):
    c = resolve_out(a, OPP_OUT, "inner")
    out_npz = os.path.join(c.out_dir, INNER_NAME)
    out_rep = os.path.join(c.out_dir, INNER_REPORT)
    if c.real and (os.path.exists(out_npz) or os.path.exists(out_rep)):
        raise P04Error(f"{out_npz} already exists; the registered inner run is made once")
    t0 = time.time()
    cpath, crec, arrays_are_design = counts_for(c, a)
    counts = load_counts(cpath)
    tabs, info = build_tables(counts)
    del counts
    tseam, tseam_ok = tables_seam(tabs) if arrays_are_design else ({"state": "not compared (other counts)"}, True)
    keys = p3.load_inner_keys(P02_DUMPS)
    inner = load_part("inner")
    idx = inner_rows(inner, keys)
    n = keys["n"]
    rows = subset_rows(n, a.stride, a.limit)
    games = []
    for i in rows:
        g = idx[i]
        games.append((i, inner["moves"][inner["off"][g]:inner["off"][g + 1]],
                      target_colour_value(keys["color"][i])))
    ll, nm, an = score_games(games, tabs, workers=a.workers)
    if ll.shape[0] < n:          # rows after the last scored one (subset preview)
        pad = n - ll.shape[0]
        ll = np.concatenate([ll, np.full((pad, 2, 3, 10), np.nan)])
        nm = np.concatenate([nm, np.zeros((pad, 2, 3), dtype=np.int64)])
        an = np.concatenate([an, np.zeros(pad, dtype=np.int64)])
    scored = np.zeros(n, dtype=bool)
    scored[rows] = True
    t4, o4 = side_vectors(ll)
    arrays = {"rank": np.array(keys["rank"]), "game_id": np.array(keys["game_id"]),
              "player_id": np.array(keys["player_id"]), "color": np.array(keys["color"]), "label": keys["label"],
              "o4": o4, "t4": t4, "n_moves": nm, "anomalies": an, "scored": scored}
    save_npz(out_npz, arrays)
    rep = report_head(c, a)
    lab = keys["label"][scored]
    rep.update({"games": int(n), "scored_games": int(scored.sum()), "anomalies": int(an.sum()),
                "games_without_opponent_moves": int((nm[scored, 1].sum(1) == 0).sum()),
                "games_without_target_moves": int((nm[scored, 0].sum(1) == 0).sum()),
                "moves": {"target": int(nm[scored, 0].sum()), "opponent": int(nm[scored, 1].sum())},
                "per_game_argmax_accuracy_record_only": {
                    "o4": float(np.mean(o4[scored].argmax(1) == lab)) if len(lab) else None,
                    "t4": float(np.mean(t4[scored].argmax(1) == lab)) if len(lab) else None},
                "counts": crec, "tables": info, **tables_record(tabs), "design_tables": tseam, "alpha": ALPHA,
                "inner_keys": {"path": "results/prereg02/rank/dumps/inner_keys.npz", "sha256": keys["sha256"]},
                "output": os.path.basename(out_npz), "output_sha256": sha256_file(out_npz),
                "arrays": {"o4": "sum over the opponent's moves (opponent table), [N, 10] float64",
                           "t4": "sum over the target's own moves (target table; reference only)",
                           "n_moves": "[N, 2 (target, opponent), 3 phases]", "rows": "inner_keys.npz order"}})
    problems = []
    if not tseam_ok:
        problems.append("tables differ from the design's counts_fit_tables.npy")
    if an.sum():
        problems.append(f"{int(an.sum())} anomalies (registered 0)")
    if not c.subset and n != INNER_GAMES:
        problems.append(f"inner games {n} != {INNER_GAMES}")
    if arrays_are_design:
        drec, dok = design_inner_compare(keys, rows, ll, t4, o4, nm)
        if dok is False:
            problems.append("o4 / t4 differ from the design's proxy_inner_all.npz by more than 1e-12")
    else:
        drec = {"state": "not compared (counts are not the design's arrays)"}
    rep["design_proxy"] = drec
    rep["problems"] = problems
    rep["status"] = "ok" if not problems else "stop"
    rep["seconds"] = round(time.time() - t0, 1)
    write_json(out_rep, rep)
    print(json.dumps({"scored_games": rep["scored_games"], "anomalies": rep["anomalies"], "design_proxy":
                      drec.get("compare", drec.get("state")), "status": rep["status"], "seconds": rep["seconds"]}),
          flush=True)
    return 0 if not problems else 1


# ================================================================ validation question file (questions only)
def read_question_slots(path):
    """[(question_id, slot, colour, sgf)] in file order, slots 1..num_games; only the question file's own columns."""
    refuse_answer_path(path)
    csv.field_size_limit(10**8)
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n = int(r["num_games"])
            for i in range(1, n + 1):
                col = r[f"color_{i}"].strip()
                if col not in ("B", "W"):
                    raise P04Error(f"{r['question_id']} slot {i}: colour {col!r}")
                out.append((r["question_id"], i, col, r[f"sgf_{i}"]))
    return out


def slots_match_keys(slots, vk):
    """The question file's (question_id, slot, colour) equal prereg02's val_keys.npz row by row."""
    if len(slots) != vk["n"]:
        raise P04Error(f"question file has {len(slots)} slots, val_keys.npz {vk['n']}")
    for i, (q, s, col, _) in enumerate(slots):
        if (q, s, col) != (vk["question_id"][i], vk["slot"][i], vk["color"][i]):
            raise P04Error(f"slot row {i}: question file ({q}, {s}, {col}) != val_keys "
                           f"({vk['question_id'][i]}, {vk['slot'][i]}, {vk['color'][i]})")


def val_scores(slots, tabs, workers=8, tmp_root=None):
    """O4 / T4 of every slot (target colour = the question file's colour). Replay failure = no parsed move, or an
    anomaly (a move onto an occupied point stops the replay)."""
    games, n_moves_parsed = [], np.zeros(len(slots), dtype=np.int64)
    for i, (_q, _s, col, sgf) in enumerate(slots):
        m = encode_one(sgf)
        n_moves_parsed[i] = len(m)
        games.append((i, m, target_colour_value(col)))
    ll, nm, an = score_games(games, tabs, workers=workers, tmp_root=tmp_root)
    t4, o4 = side_vectors(ll)
    fail = (n_moves_parsed == 0) | (an > 0)
    return {"o4": o4, "t4": t4, "n_moves": nm, "anomalies": an, "moves_parsed": n_moves_parsed,
            "replay_failed": fail}


def cmd_val(a):
    c = resolve_out(a, OPP_OUT, "val")
    fz = require_frozen_config()                       # no bypass, preview or not
    require_own_committed(c.tools)                     # the validation O4 only from committed code, preview or not
    out_npz = os.path.join(c.out_dir, VAL_NAME)
    out_rep = os.path.join(c.out_dir, VAL_REPORT)
    if c.real and (os.path.exists(out_npz) or os.path.exists(out_rep)):
        raise P04Error(f"{out_npz} already exists; the registered val run is made once")
    t0 = time.time()
    cpath, crec = registered_counts(c, None)          # always the registered, committed counts
    counts = load_counts(cpath)
    tabs, info = build_tables(counts)
    del counts
    vk = p3.load_val_keys(P02_DUMPS)
    slots = read_question_slots(QUESTIONS)
    slots_match_keys(slots, vk)
    if len(slots) != VAL_SLOTS or len(vk["groups"]) != VAL_QUESTIONS:
        raise P04Error(f"validation: {len(slots)} slots / {len(vk['groups'])} questions, registered "
                       f"{VAL_SLOTS} / {VAL_QUESTIONS}")
    res = val_scores(slots, tabs, workers=a.workers)
    arrays = {"question_id": np.array(vk["question_id"]), "slot": np.array(vk["slot"], dtype=np.int64),
              "color": np.array(vk["color"]), "o4": res["o4"], "t4": res["t4"], "n_moves": res["n_moves"],
              "anomalies": res["anomalies"], "moves_parsed": res["moves_parsed"],
              "replay_failed": res["replay_failed"]}
    save_npz(out_npz, arrays)
    nf = int(res["replay_failed"].sum())
    rep = report_head(c, a)
    rep.update({"frozen_config": fz, "counts": crec, "tables": info, **tables_record(tabs), "alpha": ALPHA,
                "questions_file": {"path": "data/local/rank_val_questions.csv", "sha256": sha256_file(QUESTIONS),
                                   "columns_read": "question_id, num_games, color_i, sgf_i"},
                "val_keys": {"path": "results/prereg02/rank/dumps/val_keys.npz", "sha256": vk["sha256"]},
                "slots": len(slots), "questions": len(vk["groups"]), "replay_failures": nf,
                "anomalies": int(res["anomalies"].sum()),
                "slots_without_moves": int((res["moves_parsed"] == 0).sum()),
                "slots_without_opponent_moves": int((res["n_moves"][:, 1].sum(1) == 0).sum()),
                "moves": {"target": int(res["n_moves"][:, 0].sum()), "opponent": int(res["n_moves"][:, 1].sum())},
                "output": os.path.basename(out_npz), "output_sha256": sha256_file(out_npz)})
    rep["problems"] = [f"{nf} replay failures (registered 0)"] if nf else []
    rep["status"] = "ok" if not nf else "stop"
    rep["seconds"] = round(time.time() - t0, 1)
    write_json(out_rep, rep)
    print(json.dumps({"slots": rep["slots"], "replay_failures": nf, "status": rep["status"]}), flush=True)
    return 0 if not nf else 1


# ================================================================ dupcheck (dupcheck.py extended to the questions)
def seq_hash(m):
    return hashlib.sha1(np.ascontiguousarray(m, dtype=np.uint16).tobytes()).digest()


COLOUR_BIT = {"B": 1, "W": 2}


def colour_bit(col):
    col = str(col)
    if col not in COLOUR_BIT:
        raise P04Error(f"target colour {col!r} (expected 'B' or 'W')")
    return COLOUR_BIT[col]


def fit_hashes(fit):
    """{sha1 of a fit game's move sequence: OR of the fit rows' target-colour bits (B = 1, W = 2)}, as dupcheck.py's
    hash -> fit rows (it compares each hit row's colour with the other game's target colour)."""
    off = fit["off"]
    h = {}
    for g in range(len(off) - 1):
        k = seq_hash(fit["moves"][off[g]:off[g + 1]])
        h[k] = h.get(k, 0) | colour_bit(fit["color"][g])
    return h


def match_games(seqs, colours, h):
    """dupcheck.py's comparison. seqs: uint16 move sequences, colours: each one's target colour ('B' / 'W').
    The same move sequence (colour and point of every move) is in fit
      same  as a fit row whose target colour equals this one's (a copy of the same player's side),
      other as a fit row whose target colour differs (the same game seen from the other player: 'colour swapped').
    -> (same bool [N], other bool [N], empty bool [N]); an empty sequence is never a match."""
    n = len(seqs)
    if len(colours) != n:
        raise P04Error(f"{n} sequences, {len(colours)} colours")
    same, other, empty = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for i, (m, col) in enumerate(zip(seqs, colours)):
        if len(m) == 0:
            empty[i] = True
            continue
        bits = h.get(seq_hash(m), 0)
        mine = colour_bit(col)
        same[i] = bool(bits & mine)
        other[i] = bool(bits & ~mine & 3)
    return same, other, empty


def dup_record(slots, same, other, empty):
    anym = same | other
    n = len(slots)
    share = float(anym.sum() / n) if n else 0.0
    return {"slots": n, "questions": len({s[0] for s in slots}), "slots_without_moves": int(empty.sum()),
            "slots_in_fit_same_target_colour": int(same.sum()),
            "slots_in_fit_other_target_colour": int(other.sum()),
            "slots_in_fit_any": int(anym.sum()), "share_of_slots": share,
            "questions_with_a_match": len({slots[i][0] for i in np.flatnonzero(anym)}),
            "matched_slots": [[slots[i][0], int(slots[i][1]), "+".join(x for x, f in (("same", same[i]),
                                                                                     ("other", other[i])) if f)]
                              for i in np.flatnonzero(anym)],
            "stop_share": DUP_STOP_SHARE, "stop": bool(share >= DUP_STOP_SHARE)}


def cmd_dupcheck(a):
    c = resolve_out(a, OPP_OUT, "dupcheck")
    out = os.path.join(c.out_dir, DUP_NAME)
    if c.real and os.path.exists(out):
        raise P04Error(f"{out} already exists")
    t0 = time.time()
    fit = load_part("fit")
    h = fit_hashes(fit)
    nfit = len(fit["off"]) - 1
    del fit
    inner = load_part("inner")
    iseq = [inner["moves"][inner["off"][g]:inner["off"][g + 1]] for g in range(len(inner["off"]) - 1)]
    io, isw, ie = match_games(iseq, inner["color"].tolist(), h)
    del inner, iseq
    vk = p3.load_val_keys(P02_DUMPS)
    slots = read_question_slots(QUESTIONS)
    slots_match_keys(slots, vk)
    if len(slots) != VAL_SLOTS or len(vk["groups"]) != VAL_QUESTIONS:
        raise P04Error(f"validation: {len(slots)} slots / {len(vk['groups'])} questions, registered "
                       f"{VAL_SLOTS} / {VAL_QUESTIONS}")
    vo, vs, ve = match_games([encode_one(s[3]) for s in slots], [s[2] for s in slots], h)
    val = dup_record(slots, vo, vs, ve)
    rep = report_head(c, a)
    rep.update({"what": "exact move sequences (colour + point of every move, p02_moves.MOVE_RE), sha1 of the uint16 "
                        "codes, as dupcheck.py; a match is 'same target colour' when a fit row with that sequence has "
                        "the slot's target colour (color_i) and 'other target colour' (colour swapped: the same game "
                        "from the other player's side) when a fit row has the other colour; the 1 % stop counts "
                        "either",
                "fit_games": nfit, "fit_unique_sequences": len(h),
                "inner_vs_fit_record_only": {"games": len(io), "also_in_fit": int((io | isw).sum()),
                                             "also_in_fit_same_colour": int(io.sum()),
                                             "also_in_fit_other_colour": int(isw.sum()),
                                             "without_moves": int(ie.sum()),
                                             "design": "dupcheck.py all: games 73,556, also_in_fit 0, "
                                                       "also_in_fit_other_colour 0"},
                "validation_vs_fit": val,
                "val_keys": {"path": "results/prereg02/rank/dumps/val_keys.npz", "sha256": vk["sha256"],
                             "slots_equal_to_question_file": True},
                "questions_file": {"path": "data/local/rank_val_questions.csv", "sha256": sha256_file(QUESTIONS),
                                   "columns_read": "question_id, num_games, color_i, sgf_i"},
                "local_files_read": ["data/local/rank_val_questions.csv (this check)",
                                     "c10data.load_rows' leak check: SHA-1 of the games in rank_val_questions.csv, "
                                     "player_val_questions.csv, player_val_candidates.csv (no label file)"],
                "seconds": round(time.time() - t0, 1)})
    rep["status"] = "stop" if val["stop"] else "ok"
    rep["problems"] = ([f"{val['slots_in_fit_any']} validation slots ({val['share_of_slots']:.2%}) are fit games: "
                        f">= 1 %, write a revision before tune (PREREG_04 2.2 item 3)"] if val["stop"] else [])
    write_json(out, rep)
    print(json.dumps({"validation": {k: val[k] for k in ("slots", "slots_in_fit_any", "share_of_slots", "stop")},
                      "inner": rep["inner_vs_fit_record_only"], "status": rep["status"]}), flush=True)
    return 1 if val["stop"] else 0


# ================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="p04_opp.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "count", "inner", "dupcheck", "val"):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", default=None, help="outside the repository = preview (scratch)")
        if name in ("count", "inner", "val"):
            s.add_argument("--workers", type=int, default=8)
        if name in ("count", "inner"):
            s.add_argument("--stride", type=int, default=1, help="preview only: every S-th game")
            s.add_argument("--limit", type=int, default=0, help="preview only: games below N (count) / first N rows")
        if name == "inner":
            s.add_argument("--counts", default=None, help="preview only: another counts npz")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    a.argv_list = argv
    fn = {"selftest": cmd_selftest, "count": cmd_count, "inner": cmd_inner, "dupcheck": cmd_dupcheck,
          "val": cmd_val}[a.cmd]
    try:
        return fn(a)
    except ERRORS as e:
        print(f"p04_opp: error: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
