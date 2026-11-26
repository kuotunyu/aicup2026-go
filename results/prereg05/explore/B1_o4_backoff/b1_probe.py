"""prereg05 exploration B1: O4 count-table smoothing / level back-off (cells unchanged, the existing fit counts).

    .venv/Scripts/python.exe results/prereg05/explore/B1_o4_backoff/b1_probe.py replay [--limit N]
    .venv/Scripts/python.exe results/prereg05/explore/B1_o4_backoff/b1_probe.py fuse
    -> results/prereg05/explore/B1_o4_backoff/b1_o4v.npz, b1_report.json   (b1_variants.json is the fixed input)

Design-stage numbers on inner pseudo-questions (split-v1 inner players; the count tables come from fit players only).
Not validation results. No scoring, no score ledger, no GPU, nothing under data/local (u1_lib guard), no file outside
this folder is written (the replay's temporary table file goes to the session scratchpad and is removed).

Seams (any failure = stop):
  (a) V0 table SHA (p04_opp.tables_record of build_tables(counts)) = results/prereg04/opp/inner_report.json
  (b) V0 replayed o4 of all 73,556 inner games = results/prereg04/opp/inner_o4.npz o4 (<= 1e-12; expected bitwise)
  (c) u1_lib.run_gen(d, ['OP', 'O4', 'SWA', 'SWB']) = u1_oof.npz eval_P_Q02 / mono_P_Q02 (<= 1e-12), rank-balanced
      0.6101211982700248, single colour 0.6106683012815998
Variants (b1_variants.json, SHA-256 pinned below): V0 registered, V1 adjacent-level smoothing (0.25), V2 coarse-cell
level back-off (pat, tac, phase), V3 alpha 50. Each replaces O4 inside Q02: run_gen(d, ['OP', 'O4v', 'SWA', 'SWB']).
"""

import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_U1 = os.path.join(_HERE, "..", "u1")
if _U1 not in sys.path:
    sys.path.insert(0, os.path.abspath(_U1))
import u1_lib as U  # noqa: E402  (installs the data/local guard before anything else is opened)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import p04_opp as OPP  # noqa: E402

HERE = _HERE
REPO = U.REPO
VARIANTS_FILE = os.path.join(HERE, "b1_variants.json")
VARIANTS_SHA256 = "6699f12d2d14f96e1a3024d0657fba910a5e43bf6e7146ee6061f26a3151f914"
OUT_NPZ = os.path.join(HERE, "b1_o4v.npz")
OUT_REPORT = os.path.join(HERE, "b1_report.json")
COUNTS = os.path.join(REPO, "data", "prereg04", "counts_fit.npz")
COUNT_REPORT = os.path.join(REPO, "results", "prereg04", "opp", "count_report.json")
INNER_REPORT = os.path.join(REPO, "results", "prereg04", "opp", "inner_report.json")
INNER_O4 = os.path.join(REPO, "results", "prereg04", "opp", "inner_o4.npz")
MOVES_NPZ = os.path.join(REPO, "data", "prereg04", "design_scratch", "moves_inner.npz")
MOVES_NPY = os.path.join(REPO, "data", "prereg04", "design_scratch", "moves_inner_moves.npy")
MANIFEST = os.path.join(REPO, "results", "prereg04", "design", "scratch_manifest.json")
P02_DUMPS = os.path.join(REPO, "results", "prereg02", "rank", "dumps")
U1_OOF = os.path.join(HERE, "..", "u1", "u1_oof.npz")
# temporary table file / preview report (removed after the run). The session temp folder cannot be used: its path
# contains 'AppData/Local', which the u1_lib guard (any path containing 'data' + sep + 'local') refuses.
SCRATCH = os.path.join(_HERE, "_tmp_b1")
WORKERS = 4
TOL = 1e-12
VIDS = ("V0", "V1", "V2", "V3")
Q02_RB = 0.6101211982700248
Q02_MONO_RB = 0.6106683012815998
GATE_PASS, GATE_WEAK = 0.008, 0.005
LEVELS = U.LEVELS
L = 10


class B1Error(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def variants():
    sha = sha256_file(VARIANTS_FILE)
    if sha != VARIANTS_SHA256:
        raise B1Error(f"b1_variants.json SHA-256 {sha} != pinned {VARIANTS_SHA256} (variants are fixed before the run)")
    v = read_json(VARIANTS_FILE)
    ids = [x["id"] for x in v["variants"]]
    if tuple(ids) != VIDS:
        raise B1Error(f"variants {ids}")
    return v, sha


# ================================================================ tables (opponent side only)
def per_level_table(C, A, alpha, prior):
    """[N_CELLS, 10]: (C_k + alpha) / (A_k + alpha / prior_k) with C, A [10, N] float64, prior [N] or [10, N]."""
    pr = prior[None, :] if prior.ndim == 1 else prior
    return ((C + alpha) / (A + alpha / pr)).T


def pooled_rate(C, A):
    """The registered pooled prior r(c) and r0 from per-level arrays [10, N] (float64)."""
    Cp, Ap = C.sum(0), A.sum(0)
    r0 = Cp.sum() / Ap.sum()
    return (Cp + 1.0) / (Ap + 1.0 / r0), r0


def smooth_levels(X, lam):
    """V1: X~_k = (1 - 2 lam) X_k + lam X_{k-1} + lam X_{k+1}; ends (1 - lam) X_end + lam X_neighbour."""
    Y = np.empty_like(X)
    Y[1:-1] = (1.0 - 2.0 * lam) * X[1:-1] + lam * X[:-2] + lam * X[2:]
    Y[0] = (1.0 - lam) * X[0] + lam * X[1]
    Y[-1] = lam * X[-2] + (1.0 - lam) * X[-1]
    return Y


def coarse_index():
    cell = np.arange(OPP.N_CELLS, dtype=np.int64)
    phase = cell % 3
    tac = (cell // 63) % 8
    pat = cell // 504
    if pat.max() != OPP.N_PAT - 1:
        raise B1Error("coarse cell decomposition does not match N_PAT")
    # cross-check against the registered encoding
    if not np.array_equal(((((pat * 8 + tac) * 7 + (cell // 9) % 7) * 3 + (cell // 3) % 3) * 3 + phase), cell):
        raise B1Error("cell decomposition does not invert the p04_opp encoding")
    return (pat * 8 + tac) * 3 + phase, OPP.N_PAT * 8 * 3


def build_variant_tables(counts, vdef):
    """-> {vid: [N_CELLS, 10] float64} (opponent side) plus a record."""
    C = counts["C_o"].astype(np.float64)
    A = counts["A_o"].astype(np.float64)
    tabs, rec = {}, {}
    full, info = OPP.build_tables(counts)
    tabs["V0"] = np.ascontiguousarray(full[1])
    rec["V0"] = {"r0": info["o"]["r0"], "moves": info["o"]["moves"]}
    seam = OPP.tables_record(full)
    del full
    r, r0 = pooled_rate(C, A)
    if not np.array_equal(per_level_table(C, A, 20.0, r), tabs["V0"]):
        raise B1Error("own V0 arithmetic differs from p04_opp.build_tables")
    vd = {x["id"]: x for x in vdef["variants"]}
    # V1
    lam = float(vd["V1"]["lambda"])
    Cs, As = smooth_levels(C, lam), smooth_levels(A, lam)
    r1, r01 = pooled_rate(Cs, As)
    rec["V1"] = {"lambda": lam, "alpha": float(vd["V1"]["alpha"]),
                 "pooled_sums_equal": bool(np.array_equal(Cs.sum(0), C.sum(0)) and np.array_equal(As.sum(0), A.sum(0))),
                 "pooled_rate_equal_registered": bool(np.array_equal(r1, r)), "r0": r01}
    tabs["V1"] = np.ascontiguousarray(per_level_table(Cs, As, float(vd["V1"]["alpha"]), r1))
    del Cs, As, r1
    # V2
    cidx, ncoarse = coarse_index()
    Cb = np.stack([np.bincount(cidx, weights=C[k], minlength=ncoarse) for k in range(L)])
    Ab = np.stack([np.bincount(cidx, weights=A[k], minlength=ncoarse) for k in range(L)])
    if not (np.array_equal(Cb.sum(1), C.sum(1)) and np.array_equal(Ab.sum(1), A.sum(1))):
        raise B1Error("coarse sums differ from the fine sums")
    rb_, rb0 = pooled_rate(Cb, Ab)
    alpha2 = float(vd["V2"]["alpha"])
    rbk = per_level_table(Cb, Ab, alpha2, rb_).T                     # [10, ncoarse]
    ratio = rbk / rb_[None, :]                                       # r_bar_k / r_bar
    prior = r[None, :] * ratio[:, cidx]                              # [10, N]
    tabs["V2"] = np.ascontiguousarray(per_level_table(C, A, alpha2, prior))
    seen = (Ab.sum(0) > 0)
    rec["V2"] = {"alpha": alpha2, "coarse_cells": int(ncoarse), "coarse_cells_seen": int(seen.sum()),
                 "r0_coarse": rb0, "r0_equal_fine": bool(rb0 == r0),
                 "level_ratio_range_seen_cells": [float(ratio[:, seen].min()), float(ratio[:, seen].max())]}
    del Cb, Ab, rbk, ratio, prior
    # V3
    a3 = float(vd["V3"]["alpha"])
    tabs["V3"] = np.ascontiguousarray(per_level_table(C, A, a3, r))
    rec["V3"] = {"alpha": a3}
    for v in VIDS:
        t = tabs[v]
        if t.shape != (OPP.N_CELLS, L) or not np.isfinite(t).all() or (t <= 0).any():
            raise B1Error(f"{v}: table not finite / positive")
        rec.setdefault(v, {})["sha256"] = hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()
        rec[v]["max_abs_log_ratio_vs_V0"] = float(np.max(np.abs(np.log(t) - np.log(tabs["V0"]))))
    return tabs, rec, seam


# ================================================================ replay (4 spawn workers, opponent side only)
_W = {}


def _init(tab_path, moves_path, off, tcol):
    _W["T"] = np.load(tab_path, mmap_mode="r")         # [N_CELLS, 4, 10]
    _W["mv"] = np.load(moves_path, mmap_mode="r")
    _W["off"], _W["tc"] = off, tcol


def score_game_multi(m, tc, T):
    """p04_opp.score_game's opponent side with V tables at once -> ll [V, 3, 10], n_moves [3], anomalies.
    Same arithmetic in the same order: per move log T[played] - log sum_{empty} T[cell], per phase sum, (phase sum
    later)."""
    snaps, meta, an = OPP.replay(m, lambda i, c: c != tc)
    V = T.shape[1]
    ll = np.zeros((V, 3, L))
    nm = np.zeros(3, dtype=np.int64)
    if not meta:
        return ll, nm, an
    cell, te, played, phase = OPP.cells_of(snaps, meta)
    starts = np.flatnonzero(np.r_[True, te[1:] != te[:-1]])
    nm[:] = np.bincount(phase, minlength=3)
    num = np.log(T[played])                                 # [Tm, V, 10]
    den = np.log(np.add.reduceat(T[cell], starts, axis=0))
    per = num - den
    for ph in range(3):
        mk = phase == ph
        if mk.any():
            ll[:, ph] = per[mk].sum(0)
    return ll, nm, an


MEM_RETRIES, MEM_SLEEP = 60, 5.0


def _chunk(rows):
    """Deterministic per game; a MemoryError (the machine's commit limit is shared with other jobs) is retried after a
    pause -- the same computation again, so the values do not depend on the retry."""
    T, mv, off, tc = _W["T"], _W["mv"], _W["off"], _W["tc"]
    out, retries = [], 0
    for i, g in rows:
        m = np.asarray(mv[off[g]:off[g + 1]])
        for _attempt in range(MEM_RETRIES):
            try:
                res = score_game_multi(m, int(tc[i]), T)
                break
            except MemoryError:
                retries += 1
                time.sleep(MEM_SLEEP)
        else:
            raise MemoryError(f"game row {i}: {MEM_RETRIES} MemoryErrors in a row")
        out.append((i,) + res + (retries,))
    return out


def inner_order():
    """moves_inner rows placed in inner_keys.npz order (rank, game_id); player and colour must agree."""
    keys = U.p3.load_inner_keys(P02_DUMPS)
    with np.load(MOVES_NPZ, allow_pickle=False) as z:
        mi = {k: z[k] for k in ("off", "rank", "game_id", "player_id", "color", "label")}
    pos = {(r, g): i for i, (r, g) in enumerate(zip(mi["rank"].tolist(), mi["game_id"].tolist()))}
    if len(pos) != len(mi["rank"]):
        raise B1Error("moves_inner: (rank, game_id) not unique")
    idx = []
    for r, g, p, col in zip(keys["rank"], keys["game_id"], keys["player_id"], keys["color"]):
        i = pos.get((r, g))
        if i is None:
            raise B1Error(f"inner_keys row ({r}, {g}) not in moves_inner")
        if str(mi["player_id"][i]) != p or str(mi["color"][i]) != col:
            raise B1Error(f"inner_keys row ({r}, {g}): player / colour differ from moves_inner")
        idx.append(i)
    idx = np.asarray(idx, dtype=np.int64)
    if len(set(idx.tolist())) != len(idx) or len(idx) != len(mi["rank"]):
        raise B1Error("inner_keys does not cover moves_inner one to one")
    if not np.array_equal(mi["label"][idx].astype(np.int64), keys["label"]):
        raise B1Error("labels differ")
    return keys, idx, mi["off"]


def cmd_replay(a):
    t0 = time.time()
    vdef, vsha = variants()
    rep = {"what": "prereg05 exploration B1 (O4 smoothing / back-off). DESIGN-STAGE inner numbers, not validation "
                   "results; nothing scored; data/local never opened (u1_lib guard)",
           "variants_file": {"file": "results/prereg05/explore/B1_o4_backoff/b1_variants.json", "sha256": vsha},
           "created": U.p4.now_iso(), "git_head": U.p3.git_head(), "environment": U.p3.environment(),
           "workers": WORKERS, "limit": a.limit}
    # inputs
    cr = read_json(COUNT_REPORT)
    csha = sha256_file(COUNTS)
    man = {e["file"]: e["sha256"] for e in read_json(MANIFEST)["large_files"]}
    msha = {"moves_inner.npz": sha256_file(MOVES_NPZ), "moves_inner_moves.npy": sha256_file(MOVES_NPY)}
    rep["inputs"] = {"counts_fit.npz": csha, "count_report_counts_sha256": cr["counts_sha256"],
                     "counts_equal": csha == cr["counts_sha256"], **msha,
                     "moves_equal_manifest": all(man.get("data/prereg04/design_scratch/" + k) == v
                                                 for k, v in msha.items())}
    if not rep["inputs"]["counts_equal"] or not rep["inputs"]["moves_equal_manifest"]:
        raise B1Error(f"input SHA-256 mismatch: {rep['inputs']}")
    counts = OPP.load_counts(COUNTS)
    tabs, trec, seam = build_variant_tables(counts, vdef)
    del counts
    ir = read_json(INNER_REPORT)
    rep["seam_a"] = {"tables_sha256": seam["tables_sha256"], "inner_report_tables_sha256": ir["tables_sha256"],
                     "passed": seam["tables_sha256"] == ir["tables_sha256"]}
    rep["tables"] = trec
    print(f"tables built; seam (a) {rep['seam_a']['passed']} ({time.time() - t0:.0f}s)", flush=True)
    if not rep["seam_a"]["passed"]:
        write_json(OUT_REPORT if not a.limit else os.path.join(SCRATCH, "b1_report_preview.json"), rep)
        raise B1Error("seam (a) failed")
    T = np.ascontiguousarray(np.stack([tabs[v] for v in VIDS], axis=1))     # [N_CELLS, 4, 10]
    del tabs
    keys, idx, off = inner_order()
    n = keys["n"]
    tc = np.array([OPP.target_colour_value(c) for c in keys["color"]], dtype=np.int8)
    rows = list(range(n)) if not a.limit else list(range(0, n, max(1, n // a.limit)))[:a.limit]
    LL = np.full((n, len(VIDS), 3, L), np.nan)
    NM = np.zeros((n, 3), dtype=np.int64)
    AN = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    os.makedirs(SCRATCH, exist_ok=True)
    tsha = hashlib.sha256(T.view(np.uint8)).hexdigest()           # T is C-contiguous: same bytes as tobytes()
    partial = os.path.join(SCRATCH, "partial_replay.npz")
    resumed = 0
    if not a.limit and os.path.exists(partial):
        with np.load(partial, allow_pickle=False) as z:
            if str(z["tables_sha256"]) == tsha and str(z["inner_keys_sha256"]) == keys["sha256"]:
                LL, NM, AN, done = z["LL"].copy(), z["NM"].copy(), z["AN"].copy(), z["done"].copy()
                resumed = int(done.sum())
        print(f"resuming: {resumed} games already replayed with the same tables", flush=True)

    def save_partial():
        if not a.limit:
            np.savez(partial + ".tmp.npz", LL=LL, NM=NM, AN=AN, done=done, tables_sha256=np.array(tsha),
                     inner_keys_sha256=np.array(keys["sha256"]))
            os.replace(partial + ".tmp.npz", partial)

    items = [(i, int(idx[i])) for i in rows if not done[i]]
    chunks = [items[i:i + 200] for i in range(0, len(items), 200)]
    tmp = tempfile.mkdtemp(prefix="_b1tab_", dir=SCRATCH)
    t1 = time.time()
    mem_retries = 0
    try:
        tab_path = os.path.join(tmp, "tables4.npy")
        np.save(tab_path, T)
        del T
        with mp.get_context("spawn").Pool(WORKERS, initializer=_init,
                                          initargs=(tab_path, MOVES_NPY, off, tc)) as pool:
            for k, res in enumerate(pool.imap_unordered(_chunk, chunks)):
                for i, ll, nm, an, _r in res:
                    LL[i], NM[i], AN[i] = ll, nm, an
                    done[i] = True
                mem_retries += res[-1][4] if res else 0
                if (k + 1) % 40 == 0:
                    el = time.time() - t1
                    print(f"{k + 1}/{len(chunks)} chunks ({el:.0f}s, eta {el / (k + 1) * (len(chunks) - k - 1):.0f}s, "
                          f"memory retries {mem_retries})", flush=True)
                    save_partial()
    except BaseException:
        save_partial()
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    replay_s = time.time() - t1
    if not done[np.array(rows)].all():
        raise B1Error("not every row replayed")
    rep["replay_resumed_games"] = resumed
    rep["replay_memory_retries"] = mem_retries
    r = np.array(rows)
    O = {v: LL[:, j].sum(1) for j, v in enumerate(VIDS)}                    # phase sum as p04_opp.side_vectors
    with np.load(INNER_O4, allow_pickle=False) as z:
        o4_reg, nm_reg = z["o4"], z["n_moves"]
    d = np.abs(O["V0"][r] - o4_reg[r])
    rep["seam_b"] = {"rows_compared": int(len(r)), "max_abs_diff": float(d.max()),
                     "bit_identical": bool(np.array_equal(O["V0"][r], o4_reg[r])),
                     "n_moves_equal": bool(np.array_equal(NM[r], nm_reg[r, 1])),
                     "anomalies": int(AN[r].sum()), "inner_o4_sha256": sha256_file(INNER_O4)}
    rep["seam_b"]["passed"] = bool(rep["seam_b"]["max_abs_diff"] <= TOL and rep["seam_b"]["n_moves_equal"]
                                   and rep["seam_b"]["anomalies"] == 0)
    lab = keys["label"][r]
    diag = {}
    for v in VIDS:
        lp = U.p4.log_softmax(O[v][r])
        tot_moves = NM[r].sum()
        diag[v] = {"per_game_top1": float(np.mean(lp.argmax(1) == lab)),
                   "mean_logp_true_level": float(np.mean(lp[np.arange(len(r)), lab])),
                   "nats_per_opponent_move_vs_uniform": float((lp[np.arange(len(r)), lab] + np.log(L)).sum() / tot_moves),
                   "mean_abs_diff_vs_V0_raw": float(np.mean(np.abs(O[v][r] - O["V0"][r])))}
    rep["per_game_diagnostics_record_only"] = diag
    rep["replay"] = {"games": int(len(r)), "opponent_moves": int(NM[r].sum()), "seconds": round(replay_s, 1),
                     "workers": WORKERS, "chunks": len(chunks),
                     "games_without_opponent_moves": int((NM[r].sum(1) == 0).sum())}
    rep["seconds_replay_stage"] = round(time.time() - t0, 1)
    print(json.dumps({"seam_b": rep["seam_b"], "diag": diag, "replay_s": round(replay_s, 1)}), flush=True)
    if a.limit:
        write_json(os.path.join(SCRATCH, "b1_report_preview.json"), rep)
        print("preview (subset): nothing written in the repository", flush=True)
        return 0 if rep["seam_b"]["passed"] else 1
    if not rep["seam_b"]["passed"]:
        rep["status"] = "stop: seam (b) failed"
        write_json(OUT_REPORT, rep)
        raise B1Error("seam (b) failed")
    np.savez_compressed(OUT_NPZ, o4v=np.stack([O[v] for v in VIDS], axis=1), variants=np.array(VIDS),
                        rank=np.array(keys["rank"]), game_id=np.array(keys["game_id"]),
                        player_id=np.array(keys["player_id"]), color=np.array(keys["color"]),
                        n_moves_opponent=NM)
    rep["output_npz"] = {"file": "results/prereg05/explore/B1_o4_backoff/b1_o4v.npz", "sha256": sha256_file(OUT_NPZ),
                         "arrays": "o4v [73556, 4 (V0..V3), 10] float64 (opponent-side per-game sums, inner_keys order)"}
    rep["status"] = "replay done; fuse pending"
    write_json(OUT_REPORT, rep)
    print(f"saved {OUT_NPZ} ({time.time() - t0:.0f}s)", flush=True)
    return 0


# ================================================================ fusion and comparison
def pb_diff(boot, sa, sb):
    d = boot.stats(sa) - boot.stats(sb)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float(np.std(d, ddof=1))


def test_shape(sc, y, k):
    return float(np.mean([U.rank_balanced(sc, y, k == kk) for kk in (5, 10, 15, 20)]))


def test_shape_boot(boot, sc, y, k):
    """PlayerBootstrap draws of the official-shape statistic (each level 1/10 within each k, each k 1/4)."""
    st = []
    for kk in (5, 10, 15, 20):
        mk = k == kk
        s = np.bincount(boot.player[mk], weights=sc[mk], minlength=boot.n_players)
        c = np.bincount(boot.player[mk], minlength=boot.n_players).astype(np.float64)
        st.append(np.mean([s[dd].sum(1) / c[dd].sum(1) for dd in boot.draws], axis=0))
    return np.mean(st, axis=0)


def compare(sv, s2, Xe, bank, boot, mv, m2, ms, mboot, qboot_fn):
    y, n = Xe.y, Xe.n
    out = {}
    rb_v, rb_2 = U.rank_balanced(sv, y), U.rank_balanced(s2, y)
    ci, se = pb_diff(boot, sv, s2)
    out["mixed_127200"] = {
        "rank_balanced": rb_v, "Q02_rank_balanced": rb_2, "delta": rb_v - rb_2,
        "pboot_ci95": ci, "pboot_se": se,
        "qboot_ci95_record_only": qboot_fn(sv, s2, y),
        "question_mean_delta": float(np.mean(sv - s2)),
        "per_level_delta": {LEVELS[i]: float(sv[y == i].mean() - s2[y == i].mean()) for i in range(L)},
        "per_n_bin_delta": {f"{lo}-{hi}": U.rank_balanced(sv, y, (n >= lo) & (n <= hi)) -
                            U.rank_balanced(s2, y, (n >= lo) & (n <= hi)) for lo, hi in ((5, 7), (8, 12), (13, 20))},
        "bank20_players_by_n_delta": {str(k): U.rank_balanced(sv, y, (bank == 20) & (n == k)) -
                                      U.rank_balanced(s2, y, (bank == 20) & (n == k)) for k in (5, 10, 15, 20)},
        "se_val_record": None}
    sev = U.p3.se_val_record(rb_v - rb_2, sv - s2, y)
    sev["adoption_probability_at_fraction_of_delta"] = U.p4.adoption(sev["delta"], sev["se_val"])
    out["mixed_127200"]["se_val_record"] = sev
    md = U.p3.mono_diff_record(ms, mboot, mv, m2)
    out["single_colour_120544"] = md
    ts_v, ts_2 = test_shape(mv, ms["y"], ms["k"]), test_shape(m2, ms["y"], ms["k"])
    dts = test_shape_boot(mboot, mv, ms["y"], ms["k"]) - test_shape_boot(mboot, m2, ms["y"], ms["k"])
    out["official_shape_single_colour"] = {
        "definition": "single-colour questions; mean over k in 5/10/15/20 of the rank-balanced mean at that k",
        "variant": ts_v, "Q02": ts_2, "delta": ts_v - ts_2,
        "pboot_ci95_record_only": [float(np.percentile(dts, 2.5)), float(np.percentile(dts, 97.5))]}
    return out


def cmd_fuse(a):
    t0 = time.time()
    vdef, vsha = variants()
    rep = read_json(OUT_REPORT)
    if rep.get("status") != "replay done; fuse pending" or not rep["seam_a"]["passed"] or not rep["seam_b"]["passed"]:
        raise B1Error(f"replay stage not passed: {rep.get('status')}")
    if rep["variants_file"]["sha256"] != vsha:
        raise B1Error("variants file changed since the replay")
    if sha256_file(OUT_NPZ) != rep["output_npz"]["sha256"]:
        raise B1Error("b1_o4v.npz changed since the replay")
    import u1_analyze as UA                                          # qboot (record only)
    d = U.load_inputs()
    inner = d["inner"]
    keys = inner["keys"]
    with np.load(OUT_NPZ, allow_pickle=False) as z:
        o4v = z["o4v"]
        for k in ("rank", "game_id", "player_id", "color"):
            if [str(x) for x in z[k]] != keys[k]:
                raise B1Error(f"b1_o4v.npz {k} differ from inner_keys")
    # the O4 loaded by the registered path equals V0's log_softmax
    lpv0 = U.p4.log_softmax(o4v[:, 0])
    if not np.array_equal(lpv0, d["lp"]["O4"]):
        raise B1Error("log_softmax(V0) differs from inner_sources' O4")
    Xe, Xm, ms, boot, mboot = inner["X_eval"], inner["X_mono"], inner["ms"], inner["boot"], inner["mono_boot"]
    bank_of = np.array([len(p["bank"]) for p in inner["players"]], dtype=np.int64)
    bank = bank_of[Xe.player]
    # seam (c)
    u1 = U.load_saved(U1_OOF)
    ts = time.time()
    P2, Pm2, th2 = U.run_gen(d, ["OP", "O4", "SWA", "SWB"])
    q02_s = round(time.time() - ts, 1)
    m2, s2 = U.metrics(U.rule_b_np(P2), Xe.y, Xe.n, Xe.player)
    mm2, ms2 = U.metrics(U.rule_b_np(Pm2), Xm.y, Xm.n, Xm.player)
    seam_c = {"eval_P_max_abs_diff": float(np.max(np.abs(P2 - u1["eval_P_Q02"]))),
              "mono_P_max_abs_diff": float(np.max(np.abs(Pm2 - u1["mono_P_Q02"]))),
              "rank_balanced": m2["rank_balanced"], "recorded": Q02_RB,
              "single_colour_rank_balanced": mm2["rank_balanced"], "single_colour_recorded": Q02_MONO_RB,
              "seconds": q02_s}
    seam_c["passed"] = bool(seam_c["eval_P_max_abs_diff"] <= TOL and seam_c["mono_P_max_abs_diff"] <= TOL
                            and abs(m2["rank_balanced"] - Q02_RB) <= TOL
                            and abs(mm2["rank_balanced"] - Q02_MONO_RB) <= TOL)
    rep["seam_c"] = seam_c
    print(f"seam (c): {seam_c} ({time.time() - t0:.0f}s)", flush=True)
    if not seam_c["passed"]:
        rep["status"] = "stop: seam (c) failed"
        write_json(OUT_REPORT, rep)
        raise B1Error("seam (c) failed")
    q02_w = [th[2 + 1] for th in th2]
    res = {}
    terms = ["OP", "O4v", "SWA", "SWB"]
    for j, v in enumerate(VIDS):
        ts = time.time()
        d["lp"]["O4v"] = U.p4.log_softmax(o4v[:, j])
        P, Pm, th = U.run_gen(d, terms)
        sec = round(time.time() - ts, 1)
        mv_, sv = U.metrics(U.rule_b_np(P), Xe.y, Xe.n, Xe.player)
        mmv, smv = U.metrics(U.rule_b_np(Pm), Xm.y, Xm.n, Xm.player)
        r = compare(sv, s2, Xe, bank, boot, smv, ms2, ms, mboot, UA.qboot_diff)
        r["fold_w_O4v"] = [float(t[2 + 1]) for t in th]
        r["fold_w_Q02_O4"] = [float(x) for x in q02_w]
        r["fold_w_all"] = [{t: float(x) for t, x in zip(terms, t_[2:2 + len(terms)])} for t_ in th]
        r["single_colour_rank_balanced"] = mmv["rank_balanced"]
        r["identical_to_Q02_probabilities"] = bool(np.array_equal(P, P2))
        r["fusion_seconds"] = sec
        res[v] = r
        mx = r["mixed_127200"]
        print(f"{v}: rb {mx['rank_balanced']:.6f} delta {mx['delta']:+.5f} pboot {mx['pboot_ci95']} "
              f"mono {r['single_colour_120544']['rank_balanced']:+.5f} "
              f"B {r['single_colour_120544']['by_colour']['B']['rank_balanced']:+.5f} "
              f"W {r['single_colour_120544']['by_colour']['W']['rank_balanced']:+.5f} w_O4v {r['fold_w_O4v']} "
              f"({sec}s)", flush=True)
    del d["lp"]["O4v"]
    # success signal (fixed in b1_variants.json)
    verdict = {}
    for v in VIDS:
        if v == "V0":
            continue
        mx, sc_ = res[v]["mixed_127200"], res[v]["single_colour_120544"]
        mono_ok = sc_["rank_balanced"] > 0 and all(sc_["by_colour"][c]["rank_balanced"] > 0 for c in ("B", "W"))
        lo = mx["pboot_ci95"][0]
        if mx["delta"] >= GATE_PASS and lo > 0 and mono_ok:
            verdict[v] = "pass"
        elif GATE_WEAK <= mx["delta"] < GATE_PASS and lo > 0:
            verdict[v] = "weak"
        else:
            verdict[v] = "stop"
    overall = "pass" if "pass" in verdict.values() else ("weak" if "weak" in verdict.values() else "stop")
    rep["fusion"] = {"terms": terms, "rule": "B", "oof": "5 folds, fold f fitted on the other folds' players' fit "
                     "questions (p04_rank_fuse.gen_oof_record)", "eval_questions": int(len(Xe)),
                     "single_colour_questions": int(len(Xm)), "fit_questions": int(len(inner["X_fit"])),
                     "inputs_sha256": dict(inner["inputs"], **d["sha"])}
    rep["results"] = res
    rep["verdict_by_variant"] = verdict
    rep["verdict"] = overall
    rep["success_signal"] = vdef["success_signal"]
    rep["status"] = "done"
    rep["seconds_fuse_stage"] = round(time.time() - t0, 1)
    write_json(OUT_REPORT, rep)
    print(json.dumps({"verdict": overall, "by_variant": verdict, "seconds": rep["seconds_fuse_stage"]}), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("replay")
    r.add_argument("--limit", type=int, default=0, help="subset preview (scratchpad only)")
    sp.add_parser("fuse")
    a = p.parse_args(argv)
    return cmd_replay(a) if a.cmd == "replay" else cmd_fuse(a)


if __name__ == "__main__":
    sys.exit(main())
