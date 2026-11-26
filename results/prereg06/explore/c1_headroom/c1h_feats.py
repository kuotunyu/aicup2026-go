"""Token families and dense style features for the c1_headroom exploration (black = target player).

Every function takes the points string of one game (2 letters per move, B/W alternating from black,
checked in c1h_lib.build_cache) and returns a list of tokens. 'base' is exactly the p02 Q22 / Q04
opening term (tools/p02_player_open.game_tokens, raw orientation, first 60 moves parsed)."""
import sys
sys.dont_write_bytecode = True

import numpy as np

import c1h_lib as L

po = L.po
SZ = 19


def mv60(pts):
    return L.moves_of(pts, 60)


def xy(p):
    return ord(p[0]) - 97, ord(p[1]) - 97


# ---------------------------------------------------------------- opening families
def f_base(pts):
    return po.game_tokens("B", mv60(pts), "raw")


def f_n20(pts):
    mv = mv60(pts)
    return po.col_tokens("B", mv, 20) + po.nocol_tokens("B", mv, 20)


def f_n30(pts):
    mv = mv60(pts)
    return po.col_tokens("B", mv, 30) + po.nocol_tokens("B", mv, 30)


def f_deep(pts):
    """Own moves 11..30 (positional and unordered) and whole-game bigrams of moves 11..40: only what
    the base tokens do not see."""
    mv = mv60(pts)
    own = [m for c, m in mv if c == "B"][10:30]
    toks = ["d%d:%s" % (i + 10, m) for i, m in enumerate(own)] + ["d*:%s" % m for m in own]
    seq = [m for _c, m in mv[10:40]]
    toks += ["dq:%s>%s" % (seq[i - 1], seq[i]) for i in range(1, len(seq))]
    return toks


PREFIX_L = (12, 16, 20, 25, 30, 40, 50, 60)


def f_prefix(pts):
    """Exact raw prefixes of the game (both colours) of length L: repeated long sequences."""
    n = len(pts) // 2
    return ["P%d:%s" % (Lx, pts[:2 * Lx]) for Lx in PREFIX_L if n >= Lx]


def f_opp(pts):
    """The opponent's (white) first 10 moves, positional and unordered (raw)."""
    w = [m for c, m in mv60(pts) if c == "W"][:10]
    return ["w%d:%s" % (i, m) for i, m in enumerate(w)] + ["w*:%s" % m for m in w]


# ---------------------------------------------------------------- joseki (corner sequences)
def _corner(x, y):
    if x == 9 or y == 9:
        return None
    return (x > 9) * 2 + (y > 9)


def _to_tl(x, y, cnr):
    if cnr & 2:
        x = 18 - x
    if cnr & 1:
        y = 18 - y
    return x, y


def joseki_seqs(pts, max_moves=80):
    """Per corner: the moves that land in that 9x9 corner (first max_moves of the game), mapped to the
    top-left corner, transposed so that the first off-diagonal move has x < y; each move = side
    ('o' own black, 'p' opponent white) + point."""
    mv = L.moves_of(pts, max_moves)
    seqs = {0: [], 1: [], 2: [], 3: []}
    for c, p in mv:
        x, y = xy(p)
        cn = _corner(x, y)
        if cn is None:
            continue
        seqs[cn].append((("o" if c == "B" else "p"), _to_tl(x, y, cn)))
    out = []
    for cn in range(4):
        s = seqs[cn]
        if not s:
            continue
        tr = False
        for _side, (x, y) in s:
            if x != y:
                tr = x > y
                break
        out.append([side + (chr(97 + y) + chr(97 + x) if tr else chr(97 + x) + chr(97 + y)) for side, (x, y) in s])
    return out


def f_joseki(pts):
    toks = []
    for s in joseki_seqs(pts):
        for n in range(1, min(6, len(s)) + 1):
            toks.append("J%d:%s" % (n, "".join(s[:n])))
    return toks


def f_joseki_own(pts):
    """Joseki prefixes keyed on the target's own corner moves only (order kept, opponent moves dropped)."""
    toks = []
    for s in joseki_seqs(pts):
        own = [m for m in s if m[0] == "o"]
        for n in range(1, min(4, len(own)) + 1):
            toks.append("K%d:%s" % (n, "".join(own[:n])))
    return toks


# ---------------------------------------------------------------- style tokens over the whole game
def _phase(i):
    return 0 if i < 60 else (1 if i < 150 else 2)


def _rel(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    d = max(dx, dy)
    if d <= 2:
        return "%d%d" % (min(dx, dy), max(dx, dy))
    return "m" if d <= 4 else "f"


def f_resp(pts):
    """Per own move: relation to the previous opponent move and to the own previous move
    (chebyshev <= 2 exact shape up to symmetry, 3-4 'm', >= 5 'f'), and the line, by phase."""
    n = len(pts) // 2
    P = [xy(pts[2 * i:2 * i + 2]) for i in range(n)]
    toks = []
    for i in range(0, n, 2):
        ph = _phase(i)
        x, y = P[i]
        line = min(x, y, 18 - x, 18 - y) + 1
        toks.append("L%d:%d" % (ph, min(line, 5)))
        if i >= 1:
            toks.append("R%d:%s" % (ph, _rel(P[i], P[i - 1])))
        if i >= 2:
            toks.append("S%d:%s" % (ph, _rel(P[i], P[i - 2])))
        if i >= 1:
            toks.append("RS%d:%s|%s" % (ph, _rel(P[i], P[i - 1]), _rel(P[i], P[i - 2]) if i >= 2 else "-"))
    return toks


FAMILIES = {"base": f_base, "n20": f_n20, "n30": f_n30, "deep": f_deep, "prefix": f_prefix,
            "opp": f_opp, "joseki": f_joseki, "joseki_own": f_joseki_own, "resp": f_resp}


# ---------------------------------------------------------------- dense style features
STYLE_NAMES = None


def style_vec(pts):
    """Dense per-game style features (shares), 1 + 3 phases x 11 = 34 values."""
    n = len(pts) // 2
    P = [xy(pts[2 * i:2 * i + 2]) for i in range(n)]
    out = [n / 100.0]
    for ph in range(3):
        idx = [i for i in range(0, n, 2) if _phase(i) == ph]
        m = max(len(idx), 1)
        lines = [0] * 5
        rel = [0] * 4
        selfnear = 0
        for i in idx:
            x, y = P[i]
            lines[min(min(x, y, 18 - x, 18 - y), 4)] += 1
            if i >= 1:
                d = max(abs(x - P[i - 1][0]), abs(y - P[i - 1][1]))
                rel[0 if d <= 1 else (1 if d == 2 else (2 if d <= 4 else 3))] += 1
            if i >= 2:
                selfnear += max(abs(x - P[i - 2][0]), abs(y - P[i - 2][1])) <= 2
        out += [v / m for v in lines] + [v / m for v in rel] + [selfnear / m, len(idx) / 100.0]
    return np.asarray(out, np.float64)


PH2 = (0, 20, 50, 100, 150, 10**6)


def style2_vec(pts):
    """Finer dense style features: 5 move-number phases (own moves 1-10, 11-25, 26-50, 51-75, 76+)
    x (line 1..5+, chebyshev distance to the previous opponent move 1 / 2 / 3-4 / 5+, distance to
    the own previous move <= 2, share of moves in a corner 9x9 / side / centre) + game length."""
    n = len(pts) // 2
    P = [xy(pts[2 * i:2 * i + 2]) for i in range(n)]
    out = [n / 100.0]
    for a, b in zip(PH2[:-1], PH2[1:]):
        idx = [i for i in range(0, n, 2) if a <= i < b]
        m = max(len(idx), 1)
        lines = [0] * 5
        rel = [0] * 4
        selfnear = 0
        zone = [0] * 3
        for i in idx:
            x, y = P[i]
            ln = min(x, y, 18 - x, 18 - y)
            lines[min(ln, 4)] += 1
            zone[0 if (min(x, 18 - x) < 6 and min(y, 18 - y) < 6) else (2 if ln >= 6 else 1)] += 1
            if i >= 1:
                d = max(abs(x - P[i - 1][0]), abs(y - P[i - 1][1]))
                rel[0 if d <= 1 else (1 if d == 2 else (2 if d <= 4 else 3))] += 1
            if i >= 2:
                selfnear += max(abs(x - P[i - 2][0]), abs(y - P[i - 2][1])) <= 2
        out += [v / m for v in lines] + [v / m for v in rel] + [selfnear / m] + [v / m for v in zone]
    return np.asarray(out, np.float64)
