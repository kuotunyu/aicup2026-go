"""prereg09 mapping exploration, step 1 (CPU, low memory): stream the P1 / P2 probe and the bank / mono chunks into
float32 memory-mapped arrays under data/prereg09/tmp/mapping/ (file-backed, so they add no commit charge while the
P3 extraction runs).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg09/explore/mapping/prep.py

Writes (data/prereg09/tmp/mapping/):
    probe_X.npy   [150000, 1540] float32  TO features standardised with the FROZEN p1_map.npz mu / sd
                  (= p1_fuse.standardise on the same 150k probe games; missing opponent side -> 0)
    probe_ex.npy  [150000, NEX]  float32  extra per-game features from the stored per-position arrays (raw)
    probe_meta.npz  y (level index), players ('<rank>:<player_id>'), fold (p1_fuse.player_folds seed 0), src
    inner_X.npy / inner_ex.npy / inner_meta.npz  the same for every usable bank / mono game (+ inner ref)
Order of the probe rows = tune's probe_rows: P1 chunks (sorted), then P2 chunks. Only fit-probe / bank / mono chunks
are read; never data/local, never validation features, never data/prereg09/p3.
"""

import glob
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TMP = os.path.join(REPO, "data", "prereg09", "tmp", "mapping")
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
CHUNK_FORMAT = "c10-prereg08 p1 chunk v1"
META_SHA = "404bb7096c63136d7642f5abc98ec84d3c826290451ed649fcbe37c903307282"
P1_MAP = os.path.join(REPO, "results", "prereg08", "rank", "p1_map.npz")
SETS = {"p1": os.path.join(REPO, "data", "prereg08", "p1", "sets.json"),
        "p2": os.path.join(REPO, "data", "prereg08", "p2", "sets.json")}
DIRS = {"p1_probe": os.path.join(REPO, "data", "prereg08", "p1", "probe"),
        "p2_probe": os.path.join(REPO, "data", "prereg08", "p2", "probe"),
        "bank": os.path.join(REPO, "data", "prereg08", "p1", "bank"),
        "mono": os.path.join(REPO, "data", "prereg08", "p1", "mono")}
TURN_EDGES = [0, 30, 80, 150, 10 ** 6]
QS = [0.1, 0.25, 0.5, 0.75, 0.9]
LOWP = np.log(0.05)


def ex_names():
    out = []
    for side in ("t", "o"):
        for q in ("logp", "ent"):
            out += [f"{side}_{q}_std"] + [f"{side}_{q}_q{int(100 * p)}" for p in QS]
            out += [f"{side}_{q}_turn{TURN_EDGES[i]}_{TURN_EDGES[i + 1]}" for i in range(len(TURN_EDGES) - 1)]
        out += [f"{side}_frac_lowp", f"{side}_log1p_count"]
    return out


NEX = len(ex_names())


def player_folds(players, seed=0, k=5):
    """= results/prereg08/p1/p1_fuse.py player_folds."""
    uniq = np.array(sorted(set(players)))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {p: int(i % k) for i, p in zip(range(len(uniq)), uniq[perm])}
    return np.array([fold_of[p] for p in players])


def chunk_files(d):
    return sorted(f for f in glob.glob(os.path.join(d, "chunk_*.npz")) if ".tmp" not in f)


def extra_features(z, G):
    """per game and side: std / quantiles of logp and entropy, turn-bucket means, low-prob fraction, log1p count."""
    ex = np.full((G, NEX), np.nan, dtype=np.float64)
    pg, ps, pt = z["pos_game"], z["pos_side"], z["pos_turn"]
    lp, en = z["pos_logp"].astype(np.float64), z["pos_entropy"].astype(np.float64)
    order = np.lexsort((ps, pg))
    pg, ps, pt, lp, en = pg[order], ps[order], pt[order], lp[order], en[order]
    key = pg.astype(np.int64) * 2 + ps
    bounds = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(key)]])
    per_side = NEX // 2
    for s0, e0 in zip(starts, ends):
        if e0 <= s0:
            continue
        g, s = int(pg[s0]), int(ps[s0])
        row = []
        for v in (lp[s0:e0], en[s0:e0]):
            row.append(v.std())
            row += list(np.quantile(v, QS))
            t = pt[s0:e0]
            for i in range(len(TURN_EDGES) - 1):
                m = (t >= TURN_EDGES[i]) & (t < TURN_EDGES[i + 1])
                row.append(v[m].mean() if m.any() else np.nan)
        row.append(float((lp[s0:e0] < LOWP).mean()))
        row.append(np.log1p(e0 - s0))
        ex[g, s * per_side:(s + 1) * per_side] = row
    return ex


def stream(names, mu, sd, games_of, out_prefix, n_expected=None):
    files = [(n, f) for n in names for f in chunk_files(DIRS[n])]
    # first pass: sizes and usability
    G = 0
    for n, f in files:
        with np.load(f, allow_pickle=False) as z:
            if str(z["format"]) != CHUNK_FORMAT or str(z["meta_sha256"]) != META_SHA:
                raise RuntimeError(f"{f}: format / metadata")
            us = (~z["failed"]) & (z["counts"][:, 0] > 0)
            G += int(us.sum())
    if n_expected is not None and G != n_expected:
        raise RuntimeError(f"{names}: {G} usable games, expected {n_expected}")
    X = np.lib.format.open_memmap(os.path.join(TMP, f"{out_prefix}_X.npy"), mode="w+", dtype=np.float32,
                                  shape=(G, 1540))
    EX = np.lib.format.open_memmap(os.path.join(TMP, f"{out_prefix}_ex.npy"), mode="w+", dtype=np.float32,
                                   shape=(G, NEX))
    refs, srcs, ys, pls = [], [], [], []
    i = 0
    t0 = time.time()
    for k, (n, f) in enumerate(files):
        with np.load(f, allow_pickle=False) as z:
            us = (~z["failed"]) & (z["counts"][:, 0] > 0)
            means = z["means"].astype(np.float64)
            T, O = means[:, 0, :], means[:, 1, :].copy()
            O[z["counts"][:, 1] == 0] = np.nan
            Xr = (np.concatenate([T, O], 1) - mu) / sd
            Xr[~np.isfinite(Xr)] = 0.0
            ex = extra_features(z, len(us))
            ref = z["ref"]
        g = int(us.sum())
        X[i:i + g] = Xr[us].astype(np.float32)
        EX[i:i + g] = ex[us].astype(np.float32)
        refs.append(ref[us])
        srcs += [n] * g
        if games_of is not None:
            gm = games_of[n]
            ys += [LEVELS.index(gm[j]["rank"]) for j in ref[us]]
            pls += [gm[j]["rank"] + ":" + gm[j]["player_id"] for j in ref[us]]
        i += g
        if k % 100 == 0:
            print(f"  {out_prefix} {k}/{len(files)} chunks, {i} games, {time.time() - t0:.0f}s", flush=True)
    X.flush()
    EX.flush()
    del X, EX
    meta = {"ref": np.concatenate(refs), "src": np.array(srcs)}
    if games_of is not None:
        meta["y"] = np.array(ys, dtype=np.int64)
        meta["players"] = np.array(pls)
        meta["fold"] = player_folds(meta["players"])
    np.savez(os.path.join(TMP, f"{out_prefix}_meta.npz"), **meta)
    return G


def main():
    os.makedirs(TMP, exist_ok=True)
    with np.load(P1_MAP, allow_pickle=False) as m:
        mu, sd = m["mu"], m["sd"]
        assert str(m["probe_used"]) == "P1+P2" and int(m["games"]) == 150000
    games = {}
    for k in ("p1", "p2"):
        with open(SETS[k], encoding="utf-8") as f:
            games[f"{k}_probe"] = json.load(f)["probe"]["games"]
    gp = stream(["p1_probe", "p2_probe"], mu, sd, games, "probe", n_expected=150000)
    gi = stream(["bank", "mono"], mu, sd, None, "inner")
    with open(os.path.join(TMP, "prep_record.json"), "w", encoding="utf-8") as f:
        json.dump({"probe_games": gp, "inner_games": gi, "ex_names": ex_names(), "turn_edges": TURN_EDGES,
                   "quantiles": QS}, f, indent=1)
    print("probe", gp, "inner usable", gi)


if __name__ == "__main__":
    main()
