import sys, time
sys.dont_write_bytecode = True
import numpy as np
import feat
z = np.load("moves_inner.npz")
mv, off, col = z["moves"], z["off"], z["color"]
gs = np.random.default_rng(1).choice(len(off) - 1, 300, replace=False)
# reference: slow per-point computation for a few positions
def ref_cell(snap, m):
    i, c, p, lo, lw = m
    b = np.frombuffer(snap, dtype=np.uint8).reshape(19, 19)
    def rel(y, x):
        if not (0 <= y < 19 and 0 <= x < 19): return 3
        v = b[y, x]; return 0 if v == 0 else (1 if v == c else 2)
    def group(y, x):
        col = b[y, x]; st=[(y,x)]; seen={(y,x)}; libs=set()
        while st:
            a, bb = st.pop()
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                yy, xx = a+dy, bb+dx
                if 0<=yy<19 and 0<=xx<19:
                    if b[yy,xx]==0: libs.add((yy,xx))
                    elif b[yy,xx]==col and (yy,xx) not in seen: seen.add((yy,xx)); st.append((yy,xx))
        return len(libs)
    out = {}
    for y in range(19):
        for x in range(19):
            if b[y,x]: continue
            g = np.zeros((3,3),dtype=np.int64)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy or dx: g[1+dy,1+dx] = rel(y+dy,x+dx)
            pat = feat.CANON[feat._enc(g)]
            cap=sav=ata=0
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                r = rel(y+dy,x+dx)
                if r in (1,2):
                    L = group(y+dy,x+dx)
                    if r==2 and L==1: cap=1
                    if r==1 and L==1: sav=1
                    if r==2 and L==2: ata=1
            tac = cap+2*sav+4*ata
            pt = y*19+x
            dopp = feat.DOPP[lo, pt]; down = feat.DOWN[lw, pt]
            ph = feat.phase_of(i)
            out[pt] = (((pat*8+tac)*7+dopp)*3+down)*3+ph
    return out
nbad = 0; ncheck = 0
for g in gs[:6]:
    tc = 1 if col[g] == "B" else 2
    snaps, meta, an = feat.replay(mv[off[g]:off[g+1]], lambda i, c: c == tc)
    cell, te, played, ph = feat.cells_of(snaps, meta)
    for t in range(0, len(meta), 7):
        ref = ref_cell(snaps[t], meta[t])
        mine = cell[te == t]
        pts = np.flatnonzero(np.frombuffer(snaps[t], dtype=np.uint8) == 0)
        got = dict(zip(pts.tolist(), mine.tolist()))
        ncheck += 1
        if got != ref: nbad += 1
        if ref[meta[t][2]] != played[t]: nbad += 1
print("positions checked", ncheck, "mismatches", nbad)
data = []
for g in gs:
    tc = 1 if col[g] == "B" else 2
    data.append(feat.replay(mv[off[g]:off[g+1]], lambda i, c: c == tc))
t0 = time.time()
for s, m, a in data:
    feat.cells_of(s, m)
print(f"cells_of {1000*(time.time()-t0)/len(data):.2f} ms/game")
