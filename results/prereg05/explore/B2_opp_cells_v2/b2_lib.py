"""prereg05 exploration B2 (finer opponent-move cells): shared helpers. Design stage, inner/fit training players only.

Guard: u1_lib's guard refuses every path whose text contains 'data\\local', which also matches this machine's
scratchpad (C:\\Users\\...\\AppData\\Local\\Temp\\...), where B2 must keep its large count files. This module therefore
installs an equivalent guard FIRST (the repository's data/local, any path with a '\\data\\local' segment pair, any path
containing 'answer' or 'rank_val' is refused for open / listdir / scandir / stat) and marks it installed, so that
importing u1_lib afterwards does not install its own. guard_selftest() proves the refusals.

Nothing here writes outside results/prereg05/explore/B2_opp_cells_v2 and the session scratchpad.
"""

import builtins
import hashlib
import io
import os
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
U1_DIR = os.path.join(HERE, "..", "u1")
for p in (TOOLS, os.path.abspath(U1_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

SCRATCH = os.environ.get(
    "C10_B2_SCRATCH",
    r"C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go"
    r"\5c8715ae-39b0-4c78-beba-a69cba3713cd\scratchpad\b2")

# ---------------------------------------------------------------- guard (installed before u1_lib is imported)
_FORBIDDEN = os.path.normcase(os.path.abspath(os.path.join(REPO, "data", "local")))
_SEG = os.sep + "data" + os.sep + "local"


def _forbidden(p):
    try:
        if isinstance(p, int):
            return False
        s = os.path.normcase(os.path.abspath(os.fspath(p)))
    except TypeError:
        return False
    if s == _FORBIDDEN or s.startswith(_FORBIDDEN + os.sep):
        return True
    if s.endswith(_SEG) or (_SEG + os.sep) in s:
        return True
    return ("answer" in s) or ("rank_val" in s)


class LocalDataRefused(RuntimeError):
    pass


def install_guard():
    if getattr(builtins, "_c10_b2_guard", False):
        return
    real_open, real_listdir, real_scandir, real_stat = builtins.open, os.listdir, os.scandir, os.stat

    def g_open(file, *a, **k):
        if _forbidden(file):
            raise LocalDataRefused(f"B2 guard: refusing to open {file}")
        return real_open(file, *a, **k)

    def g_listdir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"B2 guard: refusing to list {path}")
        return real_listdir(path)

    def g_scandir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"B2 guard: refusing to scan {path}")
        return real_scandir(path)

    def g_stat(path, *a, **k):
        if _forbidden(path):
            raise LocalDataRefused(f"B2 guard: refusing to stat {path}")
        return real_stat(path, *a, **k)

    builtins.open = g_open
    io.open = g_open
    os.listdir = g_listdir
    os.scandir = g_scandir
    os.stat = g_stat
    builtins._c10_b2_guard = True
    builtins._c10_u1_guard = True          # u1_lib.install_guard() sees a guard and does not add its own


install_guard()


def guard_selftest():
    """Every way in to the repository's data/local is refused; the scratchpad and data/prereg04 are allowed."""
    bad = [os.path.join(REPO, "data", "local"), os.path.join(REPO, "data", "local", "rank_val_questions.csv"),
           os.path.join(REPO, "data", "local", "x.csv")]
    refused = {}
    for p in bad:
        r = []
        for fn in (lambda q: open(q, "rb"), os.listdir, os.stat, os.scandir):
            try:
                fn(p)
                r.append(False)
            except LocalDataRefused:
                r.append(True)
            except OSError:
                r.append(False)
        refused[os.path.relpath(p, REPO).replace(os.sep, "/")] = all(r)
    allowed = {"scratchpad": not _forbidden(SCRATCH),
               "data/prereg04/design_scratch": not _forbidden(os.path.join(REPO, "data", "prereg04", "design_scratch"))}
    return {"refused_open_listdir_stat_scandir": refused, "allowed": allowed,
            "passed": all(refused.values()) and all(allowed.values())}


import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import p04_opp as po  # noqa: E402   (read-only import; frozen)

N_CELLS = po.N_CELLS                       # 557,928
N_LINE, N_LIB = 4, 3
N_BOTH = N_CELLS * N_LINE * N_LIB           # 6,695,136
assert N_CELLS == 557928 and N_BOTH == 6695136
VARIANTS = ("V0", "V1", "V2", "V3")
NCELLS_V = {"V0": N_CELLS, "V1": N_CELLS * N_LINE, "V2": N_CELLS * N_LIB, "V3": N_BOTH}
ALPHA = 20.0


def _line4_table():
    r, c = np.divmod(np.arange(po.NPT), po.SIZE)
    line = np.minimum(np.minimum(r, c), np.minimum(18 - r, 18 - c)) + 1
    return np.where(line <= 2, 0, np.where(line == 3, 1, np.where(line == 4, 2, 3))).astype(np.int64), line


LINE4, LINE = _line4_table()


def lib3_of(approx):
    approx = np.asarray(approx)
    return np.where(approx <= 1, 0, np.where(approx == 2, 1, 2)).astype(np.int64)


def variant_index(nb, v):
    """N_both index -> the variant's cell index (integer marginal of N_both)."""
    if v == "V0":
        return nb // (N_LINE * N_LIB)
    if v == "V1":
        return nb // N_LIB                                  # cell * 4 + line4
    if v == "V2":
        return (nb // (N_LINE * N_LIB)) * N_LIB + nb % N_LIB  # cell * 3 + lib3
    if v == "V3":
        return nb
    raise ValueError(v)


def both_cells(snaps, meta, cell, te):
    """N_both of every empty point (same order as cells_of) and the extra parts. cell / te from po.cells_of."""
    T = len(snaps)
    b = np.frombuffer(b"".join(snaps), dtype=np.uint8).reshape(T, po.NPT)
    mc = np.array([m[1] for m in meta], dtype=np.uint8)
    own = b == mc[:, None]
    emp = b == 0
    te2, pe = np.nonzero(emp)                       # same call as cells_of -> same order
    if not np.array_equal(te2, te):
        raise AssertionError("empty-point order differs from cells_of")
    lab_own, n_own = ndimage.label(own.reshape(T, po.SIZE, po.SIZE), structure=po.STRUCT)
    plo = np.zeros((T, po.PAD), dtype=np.int32)
    plo[:, po.INNER] = lab_own.reshape(T, po.NPT)
    pemp = np.zeros((T, po.PAD), dtype=bool)
    pemp[:, po.INNER] = emp
    plo = plo.ravel()
    pemp = pemp.ravel()
    q = te * po.PAD + po.INNER[pe]
    Lo = plo[q[:, None] + po.NB4]                   # own block labels on the 4 sides (0 = none / off-board)
    libs_o = po._libs4(Lo, n_own)                   # liberties per own block label
    valid = Lo > 0                                  # distinct blocks among the 4 sides (same rule as _libs4)
    valid[:, 1] &= Lo[:, 1] != Lo[:, 0]
    valid[:, 2] &= (Lo[:, 2] != Lo[:, 0]) & (Lo[:, 2] != Lo[:, 1])
    valid[:, 3] &= (Lo[:, 3] != Lo[:, 0]) & (Lo[:, 3] != Lo[:, 1]) & (Lo[:, 3] != Lo[:, 2])
    n_empty_nb = pemp[q[:, None] + po.NB4].sum(1).astype(np.int64)
    add = np.where(valid, libs_o[Lo] - 1, 0).sum(1).astype(np.int64)
    approx = n_empty_nb + add
    l4 = LINE4[pe]
    l3 = lib3_of(approx)
    nb = (cell.astype(np.int64) * N_LINE + l4) * N_LIB + l3
    return nb, pe, approx


def opp_positions(m, tc):
    """Replay one game (po.replay) and keep the positions before the opponent's moves (mover colour != target)."""
    snaps, meta, an = po.replay(m, lambda i, c: c != tc)
    return snaps, meta, an


def game_both(m, tc):
    """-> (nb_all, te, nb_played, phase, n_positions, anomalies) of the opponent side of one game; None if no move."""
    snaps, meta, an = opp_positions(m, tc)
    if not meta:
        return None, an
    cell, te, played, phase = po.cells_of(snaps, meta)
    nb, pe, _ = both_cells(snaps, meta, cell, te)
    pp = np.array([x[2] for x in meta])
    hit = np.flatnonzero(pe == pp[te])
    if len(hit) != len(meta) or not np.array_equal(te[hit], np.arange(len(meta))):
        raise AssertionError("played point not empty")
    if not np.array_equal(nb[hit] // (N_LINE * N_LIB), played):
        raise AssertionError("played N_both does not map back to the played O4 cell")
    return (nb, te, nb[hit], phase, len(meta)), an


# ---------------------------------------------------------------- slow reference (seam b)
def ref_extra(snap, mv):
    """Per empty point: (line4, lib3, approx) by explicit BFS (no vectorisation)."""
    i, c, p, lo, lw = mv
    b = np.frombuffer(snap, dtype=np.uint8).reshape(19, 19)

    def block(y, x):
        col = b[y, x]
        st, seen, libs = [(y, x)], {(y, x)}, set()
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
        return frozenset(seen), len(libs)

    out = {}
    for y in range(19):
        for x in range(19):
            if b[y, x]:
                continue
            n_empty = 0
            blocks = {}
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if not (0 <= yy < 19 and 0 <= xx < 19):
                    continue
                v = b[yy, xx]
                if v == 0:
                    n_empty += 1
                elif v == c:
                    s, L = block(yy, xx)
                    blocks[s] = L
            approx = n_empty + sum(L - 1 for L in blocks.values())
            line = min(y, x, 18 - y, 18 - x) + 1
            l4 = 0 if line <= 2 else (1 if line == 3 else (2 if line == 4 else 3))
            l3 = 0 if approx <= 1 else (1 if approx == 2 else 2)
            out[y * 19 + x] = (l4, l3, approx)
    return out


def ref_both(snap, mv):
    base = po.ref_cell(snap, mv)
    ext = ref_extra(snap, mv)
    return {pt: (base[pt] * N_LINE + ext[pt][0]) * N_LIB + ext[pt][1] for pt in base}


# ---------------------------------------------------------------- tables (registered formula, opponent side)
def rate_table(C, A):
    """[10, n] int counts -> [n, 10] float64 with the registered formula (p04_opp.build_tables, one side)."""
    C, A = C.astype(np.float64), A.astype(np.float64)
    Cp, Ap = C.sum(0), A.sum(0)
    r0 = Cp.sum() / Ap.sum()
    r = (Cp + 1.0) / (Ap + 1.0 / r0)
    return ((C + ALPHA) / (A + ALPHA / r[None, :])).T, r, r0


def pooled_rate(C, A):
    C, A = C.astype(np.float64), A.astype(np.float64)
    Cp, Ap = C.sum(0), A.sum(0)
    r0 = Cp.sum() / Ap.sum()
    return (Cp + 1.0) / (Ap + 1.0 / r0), r0


def marginal(X, v):
    """[10, N_BOTH] -> [10, NCELLS_V[v]] integer marginal."""
    Y = X.reshape(10, N_CELLS, N_LINE, N_LIB)
    if v == "V0":
        return Y.sum(axis=(2, 3))
    if v == "V1":
        return Y.sum(axis=3).reshape(10, N_CELLS * N_LINE)
    if v == "V2":
        return Y.sum(axis=2).reshape(10, N_CELLS * N_LIB)
    if v == "V3":
        return X
    raise ValueError(v)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_sha(name):
    return po.manifest_sha(name)
