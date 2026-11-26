"""Board replay and per-move candidate features for the p04 proxy (CPU, numpy + scipy.ndimage).

A cell (the feature key of an empty point, seen from the player to move) is
    cell = ((((pat * 8 + tac) * 7 + d_opp) * 3 + d_own) * 3 + phase)
  pat   : 3x3 neighbourhood (empty / own / opponent / off-board), canonical under the 8 symmetries
  tac   : 1 * captures (adjacent opponent group with 1 liberty) + 2 * saves (adjacent own group with 1 liberty)
          + 4 * ataris (adjacent opponent group with 2 liberties)
  d_opp : distance d = dx + dy + max(dx, dy) to the opponent's last move, buckets <=2, 3, 4, 5-6, 7-10, >=11, none
  d_own : distance to the mover's own previous move, buckets <=3, 4-6, >=7 or none
  phase : move index in the game (0-based, both colours) <30, <100, >=100
Every empty point is a candidate (ko and suicide are not excluded).
"""
import itertools

import numpy as np
from scipy import ndimage

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
