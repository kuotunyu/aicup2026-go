import sys, time
sys.dont_write_bytecode = True
import numpy as np
from sgfmill import boards
import feat
print("N_PAT", feat.N_PAT, "N_CELLS", feat.N_CELLS)
z = np.load("moves_inner.npz")
mv, off, col = z["moves"], z["off"], z["color"]
rng = np.random.default_rng(1)
gs = rng.choice(len(off) - 1, 300, replace=False)
# 1. capture logic vs sgfmill (final position after all moves)
bad = 0
for g in gs[:150]:
    m = mv[off[g]:off[g + 1]]
    snaps, meta, an = feat.replay(m, lambda i, c: True)
    # board before last move + last move: rebuild final via replay of all and compare the snapshot sequence at last move
    B = boards.Board(19)
    ok = True
    for i, code in enumerate(m.tolist()):
        c = "w" if code >= 512 else "b"
        p = code & 511
        y, x = divmod(p, 19)
        # compare snapshot i (before move i)
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
    bad += (not ok) + an
print("games with mismatch vs sgfmill or anomalies:", bad, "of 150")
# 2. benchmark: target moves only
t0 = time.time(); npos = 0
for g in gs:
    m = mv[off[g]:off[g + 1]]
    tc = 1 if col[g] == "B" else 2
    snaps, meta, an = feat.replay(m, lambda i, c: c == tc)
t1 = time.time()
for g in gs:
    m = mv[off[g]:off[g + 1]]
    tc = 1 if col[g] == "B" else 2
    snaps, meta, an = feat.replay(m, lambda i, c: c == tc)
    cell, played, ph = feat.cells_of(snaps, meta)
    npos += len(meta)
t2 = time.time()
print(f"replay {1000*(t1-t0)/len(gs):.2f} ms/game; replay+cells {1000*(t2-t1)/len(gs):.2f} ms/game; {npos/len(gs):.1f} target pos/game")
# pattern stats on the sample
print("tac distribution of played:", np.bincount((played // (feat.N_DOPP*feat.N_DOWN*feat.N_PHASE)) % 8, minlength=8))
