"""p04 proxy, step 1: count cells on the fit games (training players minus inner; load_rows part 'fit').

For every move of every fit game, from the mover's side:
  target moves (mover = the labelled player): C_t[level, played cell] += 1, A_t[level, cell of every empty point] += 1
  opponent moves (mover = the other player, labelled with the target's level): C_o, A_o the same way
Writes counts_fit.npz (int64 [10, N_CELLS] x 4) and count_report.json.
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
LIMIT = int(os.environ.get("P04_LIMIT", "0"))
STRIDE = int(os.environ.get("P04_STRIDE", "1"))
OUTNAME = os.environ.get("P04_OUT", "counts_fit.npz")
FLUSH = 150

_G = {}


def _init(path_npy, off, label, tcol):
    _G["mv"] = np.load(path_npy, mmap_mode="r")
    _G["off"], _G["label"], _G["tcol"] = off, label, tcol


def _shard(args):
    w, nw = args
    off, label, tcol = _G["off"], _G["label"], _G["tcol"]
    G = len(off) - 1 if not LIMIT else min(LIMIT, len(off) - 1)
    games = list(range(w * STRIDE, G, nw * STRIDE))
    # this shard's moves, copied once into private memory (small: ~16 MB)
    mv = _G["mv"]
    acc = {k: np.zeros(10 * feat.N_CELLS, dtype=np.int32) for k in ("C_t", "A_t", "C_o", "A_o")}
    t0 = time.time()
    n_games = n_t = n_o = anomalies = 0
    for j, g in enumerate(games):
        m = np.array(mv[off[g]:off[g + 1]])
        tc = int(tcol[g])
        lv = int(label[g]) * feat.N_CELLS
        snaps, meta, an = feat.replay(m, lambda i, c: True)
        anomalies += an
        for side in ("t", "o"):
            sel = [k for k, x in enumerate(meta) if (x[1] == tc) == (side == "t")]
            if not sel:
                continue
            cell, te, played, _ = feat.cells_of([snaps[k] for k in sel], [meta[k] for k in sel])
            np.add.at(acc["C_" + side], played + lv, 1)
            np.add.at(acc["A_" + side], cell + lv, 1)
            if side == "t":
                n_t += len(sel)
            else:
                n_o += len(sel)
        n_games += 1
        if w == 0 and (j + 1) % 2000 == 0:
            print(f"worker 0: {j + 1}/{len(games)} games ({time.time() - t0:.0f}s)", flush=True)
    path = os.path.join(HERE, f"_shard_{w}.npz")
    np.savez(path, **acc)
    del acc
    return path, {"games": n_games, "target_moves": n_t, "opponent_moves": n_o, "anomalies": anomalies,
                  "seconds": time.time() - t0}


def main():
    t0 = time.time()
    z = np.load(os.path.join(HERE, "moves_fit.npz"))
    npy = os.path.join(HERE, "moves_fit_moves.npy")
    if not os.path.exists(npy):
        np.save(npy, z["moves"])
    off = z["off"]
    label = z["label"].astype(np.int64)
    tcol = np.where(z["color"] == "B", 1, 2).astype(np.int64)
    print(f"fit games {len(off) - 1}, N_CELLS {feat.N_CELLS}, workers {NW}", flush=True)
    with Pool(NW, initializer=_init, initargs=(npy, off, label, tcol)) as pool:
        outs = pool.map(_shard, [(w, NW) for w in range(NW)])
    tot = {k: np.zeros(10 * feat.N_CELLS, dtype=np.int64) for k in ("C_t", "A_t", "C_o", "A_o")}
    for path, _ in outs:
        s = np.load(path)
        for k in tot:
            tot[k] += s[k].astype(np.int64)
        s.close()
        os.remove(path)
    tot = {k: v.reshape(10, feat.N_CELLS) for k, v in tot.items()}
    stats = [o[1] for o in outs]
    np.savez(os.path.join(HERE, OUTNAME), **tot)
    rep = {"games": sum(s["games"] for s in stats), "target_moves": sum(s["target_moves"] for s in stats),
           "opponent_moves": sum(s["opponent_moves"] for s in stats),
           "anomalies": sum(s["anomalies"] for s in stats), "n_cells": feat.N_CELLS, "n_patterns": feat.N_PAT,
           "cells_seen_target": int((tot["A_t"].sum(0) > 0).sum()),
           "cells_played_target": int((tot["C_t"].sum(0) > 0).sum()),
           "target_moves_by_level": tot["C_t"].sum(1).tolist(),
           "stride": STRIDE, "worker_seconds": [round(s["seconds"], 1) for s in stats], "wall_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, OUTNAME.replace(".npz", "_report.json")), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
