"""prereg05 exploration B2, step 1: count the opponent moves of a 25 % fit-player subsample in N_both cells.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg05/explore/B2_opp_cells_v2/b2_count.py [--frac 0.25]
    -> scratchpad/b2/counts_both_<pct>.npz-like pair C_o_<pct>.npy, A_o_<pct>.npy (int64 [10, 6,695,136])
    -> results/prereg05/explore/B2_opp_cells_v2/b2_count_<pct>.json

Fit games only (design moves_fit.npz + moves_fit_moves.npy, SHA-256 checked against the prereg04 scratch manifest);
no load_rows / load_part, nothing under data/local. 4 spawn workers, int32 accumulators guarded like p04_opp.
Seams before the count (any failure stops): (b) slow per-point reference on >= 50 positions, (a) the first 300
subsample games through tools/p04_opp._count_shard == the N_both marginal.
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # numpy + scipy with 24 OpenBLAS threads commit ~1.5 GB per process on this machine

import argparse  # noqa: E402
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import time

import b2_lib as B  # noqa: I100  (installs the guard first)
import numpy as np

po = B.po
DS = os.path.join(B.REPO, "data", "prereg04", "design_scratch")
FIT_NPZ = os.path.join(DS, "moves_fit.npz")
FIT_NPY = os.path.join(DS, "moves_fit_moves.npy")
SALT = "c10-p05-b2|"
WORKERS = 4
TIME_LIMIT = 55 * 60
INT32_WRAP = 2 ** 32
SEAM_A_GAMES = 300
SEAM_B_GAMES, SEAM_B_STEP, SEAM_B_MIN = 10, 5, 50

_G = {}


def _init(npy, off, label, tcol, t_start):
    _G["mv"] = np.load(npy, mmap_mode="r")
    _G["off"], _G["label"], _G["tcol"], _G["t_start"] = off, label, tcol, t_start


def file_zeros(path, n, dtype):
    """A zero-filled array backed by a .npy file in the scratchpad (file pages, not commit charge: the machine is
    shared and its commit limit is nearly used up by other jobs)."""
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n,))


def count_games(games, mv, off, label, tcol, C, A, t_start=None, verbose=False):
    """Add the opponent side of the given fit rows to the int32 N_both accumulators C, A (flat [10 * N_BOTH])."""
    n_g = n_o = n_e = an_tot = retries = 0
    t0 = time.time()
    timed_out = False
    for j, g in enumerate(games):
        if t_start is not None and (j % 200 == 0) and time.time() - t_start > TIME_LIMIT:
            timed_out = True
            break
        m = np.array(mv[off[g]:off[g + 1]])
        tc = int(tcol[g])
        lv = int(label[g]) * B.N_BOTH
        for attempt in range(30):                  # the machine is shared: retry a transient MemoryError
            try:
                res, an = B.game_both(m, tc)
                if res is not None:
                    ic, ia = res[2] + lv, res[0] + lv
                break
            except MemoryError:
                retries += 1
                time.sleep(5)
        else:
            raise MemoryError(f"game row {g}: 30 MemoryErrors in a row")
        an_tot += an
        n_g += 1
        if res is None:
            continue
        np.add.at(C, ic, 1)
        np.add.at(A, ia, 1)
        n_o += res[4]
        n_e += len(ia)
        if verbose and (j + 1) % 2000 == 0:
            print(f"worker 0: {j + 1}/{len(games)} games ({time.time() - t0:.0f}s)", flush=True)
    return C, A, {"games": n_g, "opponent_moves": n_o, "empty_points": n_e, "anomalies": an_tot,
                  "seconds": time.time() - t0, "timed_out": timed_out, "memory_retries": retries}


def _shard(args):
    w, games, tmp = args
    pc, pa = os.path.join(tmp, f"_C_{w}.npy"), os.path.join(tmp, f"_A_{w}.npy")
    C, A = file_zeros(pc, 10 * B.N_BOTH, np.int32), file_zeros(pa, 10 * B.N_BOTH, np.int32)
    C, A, st = count_games(games, _G["mv"], _G["off"], _G["label"], _G["tcol"], C, A, _G["t_start"],
                           verbose=(w == 0))
    C.flush()
    A.flush()
    del C, A
    return pc, pa, st


def marginal_v0_chunked(X):
    """flat [10 * N_BOTH] -> [10, N_CELLS] int64 sum over line4 and lib3, one level at a time (small memory)."""
    out = np.zeros((10, B.N_CELLS), dtype=np.int64)
    for k in range(10):
        out[k] = np.asarray(X[k * B.N_BOTH:(k + 1) * B.N_BOTH]).reshape(B.N_CELLS, 12).astype(np.int64).sum(1)
    return out


def load_fit():
    z = np.load(FIT_NPZ, allow_pickle=False)
    out = {k: z[k] for k in ("off", "player_id", "color", "label")}
    z.close()
    return out


def subsample(fit, frac):
    pid, lab = fit["player_id"], fit["label"].astype(np.int64)
    u, first = np.unique(pid, return_index=True)
    lvl = lab[first]
    # one level per player
    chk = {}
    for p_, l_ in zip(pid.tolist(), lab.tolist()):
        if chk.setdefault(p_, l_) != l_:
            raise SystemExit(f"player {p_} has two levels")
    sel, per = [], {}
    for k in range(10):
        ids = sorted(u[lvl == k].tolist(), key=lambda s: B.hashlib.sha256((SALT + s).encode("utf-8")).hexdigest())
        n_take = math.ceil(frac * len(ids))
        sel.extend(ids[:n_take])
        per[po.LEVELS[k]] = {"players": len(ids), "selected": n_take}
    sset = set(sel)
    rows = np.flatnonzero(np.isin(pid, np.array(sorted(sset))))
    lst = "\n".join(sorted(sset))
    rec = {"fit_players": int(len(u)), "selected_players": len(sset), "per_level": per,
           "games": int(len(rows)), "list_sha256": B.hashlib.sha256(lst.encode("utf-8")).hexdigest(),
           "list_sha256_definition": "sha256 of '\\n'.join(sorted(selected player_id)) (UTF-8)",
           "rule": f"per level, sort by sha256('{SALT}' + player_id) hex ascending, first ceil({frac} x players)"}
    return rows, rec, sorted(sset)


def seam_b(fit, mv, rows, tcol):
    rng = np.random.default_rng(2)
    gs = rng.choice(rows, SEAM_B_GAMES, replace=False)
    npos = nbad = npts = 0
    t0 = time.time()
    for g in gs:
        m = np.array(mv[fit["off"][g]:fit["off"][g + 1]])
        tc = int(tcol[g])
        snaps, meta, an = B.opp_positions(m, tc)
        cell, te, played, phase = po.cells_of(snaps, meta)
        nb, pe, _ = B.both_cells(snaps, meta, cell, te)
        for t in range(0, len(meta), SEAM_B_STEP):
            ref = B.ref_both(snaps[t], meta[t])
            mk = te == t
            got = dict(zip(pe[mk].tolist(), nb[mk].tolist()))
            npos += 1
            npts += len(ref)
            if got != ref:
                nbad += 1
    return {"games": SEAM_B_GAMES, "games_drawn": "numpy default_rng(2).choice(subsample rows, 10)",
            "every_nth_opponent_position": SEAM_B_STEP, "positions": npos, "empty_points_compared": npts,
            "position_mismatches": nbad, "min_positions": SEAM_B_MIN, "seconds": round(time.time() - t0, 1),
            "reference": "p04_opp.ref_cell (slow per-point O4 cell) + per-point BFS blocks / liberties / line",
            "passed": bool(npos >= SEAM_B_MIN and nbad == 0)}


def seam_a(fit, mv, rows, tcol, tmp):
    games = rows[:SEAM_A_GAMES]
    off = fit["off"]
    parts = [np.array(mv[off[g]:off[g + 1]]) for g in games]
    sub_off = np.zeros(len(games) + 1, dtype=np.int64)
    np.cumsum([len(p) for p in parts], out=sub_off[1:])
    npy = os.path.join(tmp, "seam_a_moves.npy")
    np.save(npy, np.concatenate(parts).astype(np.uint16))
    sub_label = fit["label"][games].astype(np.int64)
    sub_tcol = tcol[games]
    t0 = time.time()
    po._count_init(npy, sub_off, sub_label, sub_tcol)
    path, st = po._count_shard((0, 1, 1, 0, tmp))
    with np.load(path) as s:
        ref_C, ref_A = s["C_o"].reshape(10, B.N_CELLS).astype(np.int64), s["A_o"].reshape(10, B.N_CELLS).astype(np.int64)
    os.remove(path)
    po._G.clear()
    t1 = time.time()
    C = file_zeros(os.path.join(tmp, "seam_a_C.npy"), 10 * B.N_BOTH, np.int32)
    A = file_zeros(os.path.join(tmp, "seam_a_A.npy"), 10 * B.N_BOTH, np.int32)
    C, A, st2 = count_games(games, mv, off, fit["label"], tcol, C, A)
    mC = marginal_v0_chunked(C)
    mA = marginal_v0_chunked(A)
    del C, A
    eqC, eqA = bool(np.array_equal(mC, ref_C)), bool(np.array_equal(mA, ref_A))
    return {"games": len(games), "rows": "the first 300 subsample rows in moves_fit order",
            "p04_opp_count_shard": {"opponent_moves": st["opponent_moves"], "anomalies": st["anomalies"],
                                    "seconds": round(t1 - t0, 1)},
            "b2": {"opponent_moves": st2["opponent_moves"], "anomalies": st2["anomalies"],
                   "seconds": round(time.time() - t1, 1)},
            "C_o_equal": eqC, "A_o_equal": eqA, "C_o_total": int(ref_C.sum()), "A_o_total": int(ref_A.sum()),
            "passed": bool(eqC and eqA and st["opponent_moves"] == st2["opponent_moves"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.25)
    a = ap.parse_args()
    pct = f"{a.frac * 100:g}".replace(".", "p")
    T0 = time.time()
    os.makedirs(B.SCRATCH, exist_ok=True)
    rep = {"what": "prereg05 exploration B2 step 1: opponent-move N_both counts on a fit-player subsample (design "
                   "stage; training players only)", "created": po.p3.now_iso(), "frac": a.frac,
           "guard": B.guard_selftest()}
    if not rep["guard"]["passed"]:
        raise SystemExit(f"guard self-test failed: {rep['guard']}")
    sha = {"moves_fit.npz": B.sha256_file(FIT_NPZ), "moves_fit_moves.npy": B.sha256_file(FIT_NPY)}
    man = {k: B.manifest_sha(k) for k in sha}
    rep["inputs"] = {k: {"sha256": sha[k], "manifest_sha256": man[k], "equal": sha[k] == man[k]} for k in sha}
    print("inputs", rep["inputs"], f"({time.time() - T0:.0f}s)", flush=True)
    if not all(v["equal"] for v in rep["inputs"].values()):
        raise SystemExit("design moves files differ from the manifest")
    fit = load_fit()
    mv = np.load(FIT_NPY, mmap_mode="r")
    if len(fit["off"]) - 1 != 720307 or int(fit["off"][-1]) != len(mv):
        raise SystemExit("fit arrays do not match the registered 720,307 games")
    tcol = np.where(fit["color"] == "B", 1, 2).astype(np.int64)
    rows, srec, sel = subsample(fit, a.frac)
    rep["subsample"] = srec
    print("subsample", json.dumps({k: srec[k] for k in ("selected_players", "games", "list_sha256")}),
          f"({time.time() - T0:.0f}s)", flush=True)

    tmp = tempfile.mkdtemp(prefix="_b2count_", dir=B.SCRATCH)
    try:
        rep["seam_b"] = seam_b(fit, mv, rows, tcol)
        print("seam b", rep["seam_b"], flush=True)
        if not rep["seam_b"]["passed"]:
            raise SystemExit("seam b failed")
        rep["seam_a"] = seam_a(fit, mv, rows, tcol, tmp)
        print("seam a", rep["seam_a"], flush=True)
        if not rep["seam_a"]["passed"]:
            raise SystemExit("seam a failed")

        # int32 preflight per shard: 361 x the shard's encoded opponent moves < 2^32
        off = fit["off"]
        shards = [rows[w::WORKERS] for w in range(WORKERS)]
        pre = []
        for gs in shards:
            o = 0
            for g in gs:
                seg = mv[off[g]:off[g + 1]]
                nw = int(np.count_nonzero(np.asarray(seg) >= 512))
                o += nw if tcol[g] == 1 else (len(seg) - nw)       # opponent = the other colour
            pre.append(o)
        rep["int32_preflight"] = {"max_shard_opponent_moves": max(pre), "bound": 361 * max(pre), "limit": INT32_WRAP}
        if 361 * max(pre) >= INT32_WRAP:
            raise SystemExit(f"int32 preflight failed: {rep['int32_preflight']}")
        t_start = time.time()
        print(f"count: {len(rows)} games, {WORKERS} workers (preflight {rep['int32_preflight']})", flush=True)
        with mp.get_context("spawn").Pool(WORKERS, initializer=_init,
                                          initargs=(FIT_NPY, off, fit["label"].astype(np.int64), tcol, t_start)) as pool:
            outs = pool.map(_shard, [(w, shards[w], tmp) for w in range(WORKERS)])
        wall = time.time() - t_start
        stats = [o[2] for o in outs]
        rep["count"] = {"games": sum(s["games"] for s in stats), "opponent_moves": sum(s["opponent_moves"] for s in stats),
                        "empty_points": sum(s["empty_points"] for s in stats),
                        "memory_retries": sum(s["memory_retries"] for s in stats),
                        "anomalies": sum(s["anomalies"] for s in stats), "wall_seconds": round(wall, 1),
                        "worker_seconds": [round(s["seconds"], 1) for s in stats],
                        "timed_out": any(s["timed_out"] for s in stats), "workers": WORKERS}
        print("count", rep["count"], flush=True)
        if rep["count"]["timed_out"]:
            rep["status"] = "timed out after 55 min: rerun with --frac 0.125"
            with open(os.path.join(B.HERE, f"b2_count_{pct}.json"), "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1, ensure_ascii=False)
            raise SystemExit(rep["status"])
        out = {}
        for name, idx in (("C_o", 0), ("A_o", 1)):
            p = os.path.join(B.SCRATCH, f"{name}_{pct}.npy")
            X = np.lib.format.open_memmap(p, mode="w+", dtype=np.int64, shape=(10, B.N_BOTH))
            parts = [np.load(o[idx], mmap_mode="r") for o in outs]
            neg = 0
            for k in range(10):
                row = np.zeros(B.N_BOTH, dtype=np.int64)
                for part in parts:
                    seg = part[k * B.N_BOTH:(k + 1) * B.N_BOTH]
                    neg += int((seg < 0).sum())
                    row += seg
                X[k] = row
            if any(part.dtype != np.int32 for part in parts) or neg:
                raise SystemExit(f"{name}: int32 overflow ({neg} negative partials)")
            del parts, seg, part, row
            X.flush()
            out[name] = {"file": p, "dtype": "int64", "shape": [10, B.N_BOTH], "total": int(X.sum()),
                         "max": int(X.max())}
            if name == "C_o":
                rep["moves_by_level"] = X.sum(1).tolist()
            else:
                seen = np.zeros(B.N_BOTH, dtype=bool)
                for k in range(10):
                    seen |= np.asarray(X[k]) > 0
                rep["cells_seen"] = {"N_both": int(seen.sum()),
                                     "V0": int(seen.reshape(B.N_CELLS, 12).any(1).sum())}
            del X
            out[name]["sha256"] = B.sha256_file(p)
        for o in outs:
            for f in o[:2]:
                os.remove(f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if out["C_o"]["total"] != rep["count"]["opponent_moves"] or out["A_o"]["total"] != rep["count"]["empty_points"]:
        raise SystemExit("C_o / A_o totals differ from the counted opponent moves / empty points")
    rep["counts"] = out
    rep["seconds_total"] = round(time.time() - T0, 1)
    rep["status"] = "ok"
    with open(os.path.join(B.HERE, f"b2_count_{pct}.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print("done", json.dumps({k: rep[k] for k in ("cells_seen", "seconds_total", "status")}), flush=True)


if __name__ == "__main__":
    main()
