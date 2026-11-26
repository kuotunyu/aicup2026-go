"""p04 proxy, step 2: level-conditioned per-move log-likelihood of every inner game (fit counts only).

Choice model (per level k, 'selection rate' of a cell, smoothed toward the pooled rate):
    r(c)    = (C(c) + 1) / (A(c) + 1 / r0),  r0 = sum C / sum A        (pooled over the 10 levels)
    r_k(c)  = (C_k(c) + alpha) / (A_k(c) + alpha / r(c))               (alpha pseudo-choices at the pooled rate)
    log P_k(move = p | position) = log r_k(cell(p)) - log sum_{empty q} r_k(cell(q))
Naive-Bayes model (no normalisation over the candidates, 'which kinds of moves does level k play'):
    P(c) = (C(c) + 0.5) / (sum C + 0.5 N_CELLS);  P_k(c) = (C_k(c) + 20 P(c)) / (sum_c C_k(c) + 20)
Per game: the sum over the mover's moves, kept per phase (<30, <100, >=100 moves) -> ll[variant, phase, level].
Variants: 0 t_choice_a5, 1 t_choice_a20, 2 t_choice_a100, 3 t_nb, 4 o_choice_a20, 5 o_nb
('t' = the target player's moves, 'o' = the opponent's moves, both scored with the target's level tables).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

import feat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NW = int(os.environ.get("P04_WORKERS", "8"))
COUNTS = os.environ.get("P04_COUNTS", "counts_fit.npz")
PROXY_OUT = os.environ.get("P04_PROXY_OUT", "proxy_inner.npz")
LIMIT = int(os.environ.get("P04_LIMIT", "0"))
VARIANTS = ["t_choice_a5", "t_choice_a20", "t_choice_a100", "t_nb", "o_choice_a20", "o_nb", "t_choice_pat",
            "t_choice_ctx"]
KIND = {"t_choice_a5": ("t", "choice", 0), "t_choice_a20": ("t", "choice", 1), "t_choice_a100": ("t", "choice", 2),
        "t_nb": ("t", "nb", 3), "o_choice_a20": ("o", "choice", 4), "o_nb": ("o", "nb", 5),
        "t_choice_pat": ("t", "choice", 7), "t_choice_ctx": ("t", "choice", 8)}
# coarse keys (backoff check for sparsity), alpha 20, expanded back to the full cell index:
#   t_choice_pat: (3x3 pattern, tactical) only;  t_choice_ctx: (tactical, d_opp, d_own, phase) only
DIMS = (feat.N_PAT, feat.N_TAC, feat.N_DOPP, feat.N_DOWN, feat.N_PHASE)


def coarse_rates(C, A, keep, alpha=20.0):
    """Marginalise [10, N_CELLS] counts onto the axes in keep, choice rates as the full model, expanded back."""
    Cr = C.reshape((10,) + DIMS)
    Ar = A.reshape((10,) + DIMS)
    drop = tuple(1 + i for i in range(5) if i not in keep)
    Cm = Cr.sum(axis=drop, keepdims=True)
    Am = Ar.sum(axis=drop, keepdims=True)
    Cp, Ap = Cm.sum(0, keepdims=True), Am.sum(0, keepdims=True)
    r0 = Cp.sum() / Ap.sum()
    r = (Cp + 1.0) / (Ap + 1.0 / r0)
    rk = (Cm + alpha) / (Am + alpha / r)
    return np.broadcast_to(rk, (10,) + DIMS).reshape(10, -1).T
_G = {}


def build_tables(counts):
    tabs = np.zeros((9, feat.N_CELLS, 10), dtype=np.float64)
    info = {}
    for side in ("t", "o"):
        C, A = counts["C_" + side].astype(np.float64), counts["A_" + side].astype(np.float64)
        Cp, Ap = C.sum(0), A.sum(0)
        r0 = Cp.sum() / Ap.sum()
        r = (Cp + 1.0) / (Ap + 1.0 / r0)
        Pn = (Cp + 0.5) / (Cp.sum() + 0.5 * feat.N_CELLS)
        nb = (C + 20.0 * Pn[None, :]) / (C.sum(1, keepdims=True) + 20.0)
        info[side] = {"r0": r0, "moves": float(Cp.sum())}
        if side == "t":
            tabs[6] = r[:, None]                     # pooled (level-free) rate, repeated in the 10 columns
            for j, alpha in enumerate((5.0, 20.0, 100.0)):
                tabs[j] = ((C + alpha) / (A + alpha / r[None, :])).T
            tabs[3] = np.log(nb).T
            tabs[7] = coarse_rates(C, A, keep=(0, 1))
            tabs[8] = coarse_rates(C, A, keep=(1, 2, 3, 4))
        else:
            tabs[4] = ((C + 20.0) / (A + 20.0 / r[None, :])).T
            tabs[5] = np.log(nb).T
    return tabs, info


def _init(mv_path, tab_path, off, tcol):
    _G["mv"] = np.load(mv_path, mmap_mode="r")
    _G["tab"] = np.load(tab_path, mmap_mode="r")
    _G["off"], _G["tcol"] = off, tcol


def _game(g):
    mv, tab, off = _G["mv"], _G["tab"], _G["off"]
    m = np.asarray(mv[off[g]:off[g + 1]])
    tc = int(_G["tcol"][g])
    snaps, meta, an = feat.replay(m, lambda i, c: True)
    ll = np.zeros((len(VARIANTS), 3, 10))
    nm = np.zeros((2, 3), dtype=np.int64)
    extra = np.zeros(2)                              # target side: pooled choice ll sum, top-1 hits (pooled)
    for si, side in enumerate(("t", "o")):
        sel = [k for k, x in enumerate(meta) if (x[1] == tc) == (side == "t")]
        if not sel:
            continue
        cell, te, played, phase = feat.cells_of([snaps[k] for k in sel], [meta[k] for k in sel])
        starts = np.flatnonzero(np.r_[True, te[1:] != te[:-1]])
        nm[si] = np.bincount(phase, minlength=3)
        if side == "t":
            pr = tab[6][:, 0]
            pe_ = pr[cell]
            mx = np.maximum.reduceat(pe_, starts)
            nmax = np.add.reduceat((pe_ == mx[te]).astype(np.float64), starts)
            pp = pr[played]
            extra[0] = float(np.sum(np.log(pp) - np.log(np.add.reduceat(pe_, starts))))
            extra[1] = float(np.sum((pp == mx) / nmax))
        for v in VARIANTS:
            s, kind, j = KIND[v]
            if s != side:
                continue
            vi = VARIANTS.index(v)
            T = tab[j]
            if kind == "choice":
                num = np.log(T[played])
                den = np.log(np.add.reduceat(T[cell], starts, axis=0))
                per = num - den
            else:
                per = T[played]
            for ph in range(3):
                mk = phase == ph
                if mk.any():
                    ll[vi, ph] = per[mk].sum(0)
    return g, ll, nm, an, extra


def _chunk(gs):
    return [_game(g) for g in gs]


def main():
    t0 = time.time()
    counts = np.load(os.path.join(HERE, COUNTS))
    tabs, info = build_tables({k: counts[k] for k in counts.files})
    tab_path = os.path.join(HERE, COUNTS.replace(".npz", "_tables.npy"))
    np.save(tab_path, tabs)
    del tabs
    z = np.load(os.path.join(HERE, "moves_inner.npz"))
    mv_path = os.path.join(HERE, "moves_inner_moves.npy")
    if not os.path.exists(mv_path):
        np.save(mv_path, z["moves"])
    off = z["off"]
    G = len(off) - 1
    GS = G if not LIMIT else LIMIT
    tcol = np.where(z["color"] == "B", 1, 2).astype(np.int64)
    todo = list(range(GS))
    if os.environ.get("P04_ONLY_BANK") == "1":
        # only the games the inner pseudo-questions use (each bank player's first <= 20 games, prereg02's bank)
        import json as _j
        pj = _j.load(open("D:/AI-Portfolio/competitions/aicup2026-go/results/prereg02/rank/pseudo_inner.json", encoding="utf-8"))
        want = {(p["rank"], gid) for p in pj["players"] for gid in p["bank"]}
        zr, zg = z["rank"].tolist(), z["game_id"].tolist()   # read each array once (NpzFile re-reads on access)
        todo = [g for g in range(G) if (zr[g], zg[g]) in want]
        if len(todo) != len(want):
            raise SystemExit(f"bank games found {len(todo)} of {len(want)}")
        if LIMIT:
            todo = todo[:LIMIT]
    scored = np.zeros(G, dtype=bool)
    scored[todo] = True
    chunks = [todo[i:i + 200] for i in range(0, len(todo), 200)]
    ll = np.zeros((G, len(VARIANTS), 3, 10))
    nm = np.zeros((G, 2, 3), dtype=np.int64)
    extra = np.zeros((G, 2))
    anomalies = 0
    with Pool(NW, initializer=_init, initargs=(mv_path, tab_path, off, tcol)) as pool:
        for k, res in enumerate(pool.imap_unordered(_chunk, chunks)):
            for g, a, b, an, ex in res:
                ll[g], nm[g], extra[g] = a, b, ex
                anomalies += an
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(chunks)} chunks ({time.time() - t0:.0f}s)", flush=True)
    np.savez(os.path.join(HERE, PROXY_OUT), rank=z["rank"], game_id=z["game_id"],
             player_id=z["player_id"], color=z["color"], label=z["label"], ll=ll, n_moves=nm, pooled_ll=extra[:, 0], pooled_top1_hits=extra[:, 1], scored=scored,
             variants=np.array(VARIANTS))
    rep = {"games": G, "scored_games": int(scored.sum()), "anomalies": anomalies, "tables": info, "variants": VARIANTS,
           "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, PROXY_OUT.replace(".npz", "_report.json")), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
