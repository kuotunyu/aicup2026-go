"""Shared move preprocessing for PREREG_02 (both teams). Stdlib only.

This file and tools/c10data.py are the only code shared by the rank team
(competition 2) and the player team (competition 1); see PREREG_02 section 3.

    from p02_moves import parse_moves, canon, transform
    mv = parse_moves(sgf_content)        # [('B', 'pd'), ('W', 'dp'), ...], at most 60 moves
    cv = canon(mv, 12)                   # whole game in the canonical orientation

Moves
    The regex ;([BW])\\[([a-s]{2})\\] is matched in SGF order and the first
    MAX_MOVES = 60 matches are kept. A pass (B[] or B[tt]) does not match, so it
    is skipped and does not use up one of the 60. A move is a tuple
    (colour, point): colour 'B' or 'W', point the two SGF letters, e.g. 'pd'
    (first letter = column from the left, second = row from the top, a..s).

The 8 symmetries (index s = 0..7), same convention as the notebook's d4()
    The rank notebook turns a game into feature planes with
    third_party/aicup2026_tutorial/utils.py: sgfmill reads the SGF point 'XY'
    as (row, col) = (18 - Y, X), and _prepare_board stores the stone at
    board_np[col, row], so on the planes (dims 2, 3) the point sits at
        plane (i, j) = (X, 18 - Y)          (X, Y = letter index 0..18).
    The notebook's d4(x, s) is: y = flip(x, dims=(3,)) if s >= 4 else x, then
    rot90(y, k=s % 4, dims=(2, 3)); s = 0 is the original orientation.
    On plane coordinates that is: s >= 4 first maps (i, j) -> (i, 18 - j), then
    k = s % 4 times (i, j) -> (18 - j, i) (one torch/numpy rot90 step).
    sym_point() is exactly that map, taken back to SGF letters, so symmetry s
    here is symmetry s of the notebook dumps. In SGF letters it works out to:
    s >= 4 first maps (X, Y) -> (X, 18 - Y), then k times (X, Y) -> (Y, 18 - X).

Canonical form
    canon_sym(mv, M) is the s whose transform makes the tuple of the first M
    points smallest (points compared as their two-letter strings, which is the
    same order as comparing (X, Y) integer pairs); ties go to the smallest s.
    canon(mv, M) applies that one transform to the whole game (all moves in
    mv, not only the first M). The first M points of canon() are the same for
    all 8 orientations of a game; later moves are too unless the first M
    points are themselves symmetric (a tie), in which case the smallest s wins.
"""

import itertools
import re

MOVE_RE = re.compile(r";([BW])\[([a-s]{2})\]")
MAX_MOVES = 60
N_SYMS = 8
SIZE = 19
LETTERS = "abcdefghijklmnopqrs"
assert len(LETTERS) == SIZE


def parse_moves(sgf, limit=MAX_MOVES):
    """The first `limit` moves of an SGF string as [(colour, point), ...]."""
    return [(m.group(1), m.group(2)) for m in itertools.islice(MOVE_RE.finditer(sgf), limit)]


# ---------------------------------------------------------------- coordinates
def to_xy(point):
    """'pd' -> (15, 3): (column from the left, row from the top), 0..18."""
    return LETTERS.index(point[0]), LETTERS.index(point[1])


def from_xy(x, y):
    return LETTERS[x] + LETTERS[y]


def to_plane(x, y):
    """SGF (X, Y) -> index (i, j) on the rank feature planes (dims 2, 3)."""
    return x, SIZE - 1 - y


def from_plane(i, j):
    return i, SIZE - 1 - j


def plane_sym(i, j, s):
    """The notebook's d4 on one plane position: flip dim 3 if s >= 4, then rot90 k = s % 4."""
    if not 0 <= s < N_SYMS:
        raise ValueError(f"symmetry index {s} not in 0..7")
    if s >= 4:
        j = SIZE - 1 - j
    for _ in range(s % 4):
        i, j = SIZE - 1 - j, i
    return i, j


def sym_xy(x, y, s):
    """Symmetry s on SGF integer coordinates (same s as the notebook's d4)."""
    return from_plane(*plane_sym(*to_plane(x, y), s))


# point -> transformed point, one dict per symmetry
SYM_TABLE = tuple({from_xy(x, y): from_xy(*sym_xy(x, y, s)) for x in range(SIZE) for y in range(SIZE)}
                  for s in range(N_SYMS))


def sym_point(point, s):
    return SYM_TABLE[s][point]


def transform(mv, s):
    """Apply symmetry s to every move of a game; colours are unchanged."""
    t = SYM_TABLE[s]
    return [(c, t[p]) for c, p in mv]


def compose(s1, s2):
    """The index s3 with transform(transform(mv, s1), s2) == transform(mv, s3)."""
    t1, t2 = SYM_TABLE[s1], SYM_TABLE[s2]
    want = {p: t2[t1[p]] for p in t1}
    for s3 in range(N_SYMS):
        if SYM_TABLE[s3] == want:
            return s3
    raise AssertionError("the 8 symmetries are not closed under composition")


def inverse(s):
    for s2 in range(N_SYMS):
        if compose(s, s2) == 0:
            return s2
    raise AssertionError(f"symmetry {s} has no inverse")


# ---------------------------------------------------------------- canonical form
def canon_sym(mv, M):
    """Index of the symmetry making the first M points smallest; ties -> smallest index."""
    head = [p for _, p in mv[:M]]
    best_s, best = 0, None
    for s in range(N_SYMS):
        t = SYM_TABLE[s]
        key = tuple(t[p] for p in head)
        if best is None or key < best:
            best_s, best = s, key
    return best_s


def canon(mv, M):
    """The whole game in the orientation chosen on its first M moves (see canon_sym)."""
    return transform(mv, canon_sym(mv, M))


def points_string(mv):
    """'pddp...' : the points of the moves concatenated (two letters per move)."""
    return "".join(p for _, p in mv)
