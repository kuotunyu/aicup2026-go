"""prereg65 (competition 2): Q65 = w_A * log_softmax(Q25) + w_O * OPP4B (Gemini Neon's Q65B), judged once on inner
against Q64; M8 merge simulation. CPU only, .venv (numpy / scipy), 4 cores, <= 3 GB (p58_lib.limits). prereg58 /
prereg59 / prereg64 / prereg23 / tools are imported unchanged.

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
    S=results/prereg65/explore;  PY=.venv/Scripts/python.exe
    $PY $S/p65_eval.py kernel    -> $S/kernel.json   (all-D c49 table M_c49; no inner game)
    $PY $S/p65_eval.py seam      -> $S/seam.json     (Q25, Q64 bit for bit; no OPP4B number)
    $PY $S/p65_eval.py seamp     -> $S/seam_p.json   (numpy OPP_P1_v2 on Neon's H rows vs his torch record;
                                                      inner players not among probe / V keys)
    $PY $S/p65_eval.py seamn     -> $S/seam_n.json   (Neon's D n H Q64_proxy / OPP4B_EX_l25_p1.5 re-run)
    $PY $S/p65_eval.py eval      -> $S/eval.json     (once; everything above committed and clean)
    $PY $S/p65_eval.py results   -> $S/RESULTS.md
    $PY $S/p65_eval.py mergesim  -> $S/merge_sim.json (Neon 04 section 3, M8; no inner)
    $PY $S/p65_eval.py freeze    -> results/prereg65/rank/frozen_config.json (+ copies), only when eval passed
Never data/local, never tools/score.py, never a test-set file here (the test-day tool is tools/t65_rank.py).
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg19", "explore"),
          os.path.join(REPO, "results", "prereg31", "explore"), os.path.join(REPO, "results", "prereg36", "explore"),
          os.path.join(REPO, "results", "prereg39", "explore"), os.path.join(REPO, "results", "prereg58", "explore"),
          os.path.join(REPO, "results", "prereg59", "explore"), os.path.join(REPO, "results", "prereg64", "explore"),
          HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p58_eval as E58  # noqa: E402
import p58_lib as K58  # noqa: E402
import p59_lib as K59  # noqa: E402
import p64_eval as E64  # noqa: E402
import p64_lib as K64  # noqa: E402
import p65_lib as K  # noqa: E402

TERM = "OPP4B"
HARD_STOP = "2026-11-23T12:00:00"
REF_Q64 = 0.7610187048035557
REF_Q64_MONO = 0.7599313807253795
NEON_P = {"ens": (1.59235, 0.48575), "mlp0": (1.59816, 0.4821), "mlp1": (1.59854, 0.48156),
          "clm0": (1.60705, 0.48415), "clm1": (1.60728, 0.48324)}          # 02a_opp_p1_v2_record.json
NEON_N = {"Q64_proxy": 0.74839, "OPP4B": 0.76584, "weights_Q64": None, "weights_OPP4B": [0.2294, 0.6102],
          "n_questions": None}                                               # 04 json lines 48-49, 576-615
NEON_SIM_W = [0.2162, 0.538]                                                 # 04 json lines 1281-1283
P_TOL = 1e-4
log, git, rel, committed_clean, peak_gb = E58.log, E58.git, E58.rel, E58.committed_clean, E58.peak_gb
FROZEN = os.path.join(HERE, "frozen.json")
NOTES = os.path.join(HERE, "NOTES.md")
KERNEL = os.path.join(HERE, "kernel.json")
SEAM = os.path.join(HERE, "seam.json")
SEAM_P = os.path.join(HERE, "seam_p.json")
SEAM_N = os.path.join(HERE, "seam_n.json")
EVAL = os.path.join(HERE, "eval.json")
MERGE_SIM = os.path.join(HERE, "merge_sim.json")
RESULTS = os.path.join(HERE, "RESULTS.md")
STARTED = os.path.join(K.DATA, "eval_started.json")
RANK_OUT = os.path.join(REPO, "results", "prereg65", "rank")
INNER_PREP = os.path.join(REPO, "data", "prereg12", "q09_inner")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def code_state():
    mods = E58.loaded_repo_modules()
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "prereg65_dirty": git("status", "--porcelain", "--", "results/prereg65"),
            "imported_dirty": git("status", "--porcelain", "--", *mods) if mods else "",
            "sha256_lf": {x: K58.sha_lf(os.path.join(REPO, *x.split("/"))) for x in mods}}


def require_clean(paths):
    for p in paths:
        if not committed_clean(p):
            raise SystemExit(f"{rel(p)} is not committed and clean")
    cs = code_state()
    if cs["prereg65_dirty"] or cs["imported_dirty"]:
        raise SystemExit(f"not committed and clean: {cs['prereg65_dirty']} {cs['imported_dirty']}")
    return cs


def holdout_bit_s25(key):
    h = hashlib.sha256(f"c10-prereg25|holdout|{key}".encode("utf-8")).hexdigest()[:8]
    return int(int(h, 16) % 10 == 0)


def load_protos():
    pr = E64.load_prototypes()
    if K58.sha256_file(K64.PROTO) != "6ffe6682b6584ee08c544247ef2f6ec9371ab08c59fb726b741b6fe5ff5565f7":
        raise SystemExit("prototypes.npz is not prereg64's frozen file")
    return pr


# ================================================================ D (table games)
def d_games():
    """c49 opponent-side vectors of the D players' games + (r, h, c, fold, own head row)."""
    v = E58.load_vec(K64.OPP_D)
    E64.check_vec(v, K64.OPP_D)
    with np.load(K58.guard(K58.FEATS_META), allow_pickle=False) as z:
        meta = {k: np.asarray(z[k]).astype(str)[v["store_row"]] for k in ("rank", "game_id", "player_id", "color")}
    hd, jrec = E58.hd_join(meta["rank"], meta["game_id"], meta["player_id"], meta["color"])
    Wn, hid, wr = K64.load_head()
    hidx = {p: i for i, p in enumerate(hid.tolist())}
    own = np.array([hidx[p] for p in meta["player_id"].tolist()], np.int64)
    return {"emb": v["emb"], "count": v["count"], "store_row": v["store_row"], "meta": meta, "hd": hd, "jrec": jrec,
            "r": K58.rank_index(meta["rank"]), "h": K58.hd_group(hd), "c": K58.colour_index(meta["color"]),
            "f": np.array([K58.player_fold(p) for p in meta["player_id"].tolist()], np.int64), "own": own,
            "Wn": Wn, "wr": wr, "hid": hid}


def cmd_kernel(a):
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(KERNEL):
        raise SystemExit("kernel.json exists (once)")
    cs = require_clean([FROZEN, NOTES])
    pr = load_protos()
    D = d_games()
    ok = D["count"] > 0
    qc = K.q_c49(D["emb"], D["Wn"], D["wr"], pr["proto"], pr["rank"], pr["src"], D["own"])
    M = K.c49_table(qc, D["r"], D["h"], D["c"], ok)
    Mp = K.load_m_p1()
    N = K.class_counts(D["wr"], pr["rank"], pr["src"])
    rep = {"format": "c10-prereg65 kernel v1", "created": now(), "seconds": round(time.time() - t0, 1), "code": cs,
           "what": "frozen.json: q_c49 (head tau1 70 own row left out, E / X prototypes tau2 50 with b -1 / -2, prior "
                   "correction) on all D games with a sample; M_c49 = table (add 0.5), HD 1 rows shrunk 0.25 toward the "
                   "rank-difference kernel, / mean over r (Neon 04 lines 852-855). No inner game is read.",
           "M_c49": M.tolist(), "M_P1_sha256_file": K.SCOND_SHA, "M_P1": Mp.tolist(),
           "class_counts": {k: v.tolist() for k, v in N.items()},
           "games": {"total": int(len(ok)), "with_sample": int(ok.sum())}, "hd_join": D["jrec"],
           "record_mean_max_q": float(qc[ok].max(axis=1).mean()),
           "vector_file": {"file": rel(K64.OPP_D), "sha256": K58.sha256_file(K64.OPP_D)},
           "prototypes_sha256": K58.sha256_file(K64.PROTO), "head_sha256": K58.sha256_file(K64.HEAD),
           "peak_working_set_gb": peak_gb(st)}
    K58.write_json(KERNEL, rep)
    log(json.dumps({k: rep[k] for k in ("games", "record_mean_max_q", "seconds", "peak_working_set_gb")}))
    return 0


def load_kernel():
    k = K58.read_json(KERNEL)
    M = np.asarray(k["M_c49"], np.float64)
    Mp = np.asarray(k["M_P1"], np.float64)
    if M.shape != (3, 2, K.L, K.L) or not np.array_equal(Mp, K.load_m_p1()):
        raise SystemExit("kernel.json is not the frozen table pair")
    return M, Mp, k


# ================================================================ inner pieces
def fuse_inner(inner, sets, q25, O):
    """prereg64 cmd_eval's fold loop with the term O -> (cand, folds)."""
    r = E58.harness()[2]
    rf = r.rf
    E23 = E58.harness()[1]
    Xf, Xe, Xm = sets["fit"][2], sets["eval"][2], sets["mono"][2]
    P = np.zeros((len(Xe), K.L))
    Pm = np.full((len(Xm), K.L), np.nan)
    folds = []
    for fr in q25["folds"]:
        f = fr["fold"]
        tr, ev, mv = Xf.fold != f, Xe.fold == f, Xm.fold == f
        A_tr = K64.log_softmax(E64.anchor(sets, fr["theta"], "fit", tr))
        ws, frec = K64.fit_weights([A_tr, O["fit"][tr]], Xf.y[tr])
        P[ev] = rf.softmax(K64.fused(ws, [K64.log_softmax(E64.anchor(sets, fr["theta"], "eval", ev)), O["eval"][ev]]))
        Pm[mv] = rf.softmax(K64.fused(ws, [K64.log_softmax(E64.anchor(sets, fr["theta"], "mono", mv)), O["mono"][mv]]))
        folds.append({"fold": f, "w_A": float(ws[0]), "w_O": float(ws[1]), "fit_questions": int(tr.sum()), **frec})
    if np.isnan(Pm).any():
        raise SystemExit("a single-colour question without a fold")
    s = E23.scored(inner, P, Pm)
    return {"P": P, "Pm": Pm, **s, "folds": folds}


def pooled(inner, lg):
    di = inner.d["inner"]
    return {w: K.question_term(S, lg) for w, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"]))}


def q64_rebuild(inner, sets, q25):
    M64, pr64, _k = E64.load_kernel()
    lg, _rec = E64.inner_opp3(inner, M64, pr64)
    return fuse_inner(inner, sets, q25, {w: K64.question_term(S, lg) for w, S in
                                         (("fit", inner.d["inner"]["S_fit"]), ("eval", inner.d["inner"]["S_eval"]),
                                          ("mono", inner.d["inner"]["S_mono"]))})


def inner_hd(inner):
    keys = inner.d["inner"]["keys"]
    hd, jrec = E58.hd_join(keys["rank"], keys["game_id"], keys["player_id"], keys["color"])
    return hd, jrec


def inner_p1(inner):
    """q_P1 [n_inner, 10] on the opponent half of inner_X (rows by inner_meta ref); ok = row exists and the opponent
    half is finite somewhere (x_raw sets it NaN without an opponent move)."""
    X = np.load(K58.guard(os.path.join(INNER_PREP, "inner_X.npy")), mmap_mode="r")
    with np.load(K58.guard(os.path.join(INNER_PREP, "inner_meta.npz")), allow_pickle=False) as z:
        ref = np.asarray(z["ref"], np.int64)
    if len(ref) != len(X) or len(np.unique(ref)) != len(ref) or ref.max() >= inner.n_inner:
        raise SystemExit("inner_meta ref is not a distinct row map")
    op1 = K.load_opp_p1()
    q = np.zeros((inner.n_inner, K.L))
    ok = np.zeros(inner.n_inner, bool)
    for s in range(0, len(ref), 8192):
        e = min(len(ref), s + 8192)
        O = np.asarray(X[s:e, 770:], np.float64)
        has = np.isfinite(O).any(axis=1)
        q[ref[s:e]] = K.predict_opp_p1(op1, O)
        ok[ref[s:e]] = has
    return q, ok, {"rows": int(len(ref)), "rows_with_opponent": int(ok.sum()),
                   "inner_X_sha256": K58.sha256_file(os.path.join(INNER_PREP, "inner_X.npy"))}


def inner_opp4b(inner, M, Mp, pr):
    Wn, hid, wr = K64.load_head()
    v = E58.load_vec(K64.OPP_INNER)
    E64.check_vec(v, K64.OPP_INNER, "opp")
    rows = np.load(K58.INNER_ROWS)
    if not np.array_equal(v["store_row"], rows) or v["rows_cfg"]["cache_dir"] != "data/prereg44/cache/inner":
        raise SystemExit("inner vectors are not of prereg58's inner rows")
    keys = inner.d["inner"]["keys"]
    kpid = np.asarray(keys["player_id"]).astype(str)
    if set(pr["player_id"].tolist()) & set(kpid.tolist()):
        raise SystemExit("an inner player has a prototype")
    hd, jrec = inner_hd(inner)
    h, c = K58.hd_group(hd), K58.colour_index(keys["color"])
    hidx = {p: i for i, p in enumerate(hid.tolist())}
    own = np.array([hidx.get(p, -1) for p in kpid[rows].tolist()], np.int64)
    qc = K.q_c49(v["emb"], Wn, wr, pr["proto"], pr["rank"], pr["src"], own)
    qp, okp, prec = inner_p1(inner)
    vals = K.game_values(qc, M, h[rows], c[rows], v["count"], qp[rows], Mp, okp[rows])
    lg = np.zeros((inner.n_inner, K.L))
    lg[rows] = vals
    if not np.isfinite(lg).all():
        raise SystemExit("not finite")
    has = v["count"] > 0
    al = np.clip((qc.max(axis=1) - 0.1) / 0.9, 0, 1) ** K.P_POW
    rec = {"games_in_questions": int(len(rows)), "games_without_c49_sample": int((~has).sum()),
           "games_without_opponent_p1": int((~okp[rows]).sum()),
           "own_head_row_left_out_games": int((own >= 0).sum()), "hd_join": jrec, "p1": prec,
           "mean_max_q_c49_record": float(qc[has].max(axis=1).mean()), "mean_alpha_record": float(al[has].mean()),
           "vector_file": {"file": rel(K64.OPP_INNER), "sha256": K58.sha256_file(K64.OPP_INNER)}}
    return lg, rec, hd


# ================================================================ seam (Q25, Q64)
def cmd_seam(a):
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(SEAM):
        raise SystemExit("seam.json exists")
    cs = require_clean([FROZEN, NOTES, KERNEL])
    _mods, inner, terms7, Emap, q25, seam25 = E58.build_q25()
    sets = E64.inner_sets(inner, terms7, Emap)
    q64 = q64_rebuild(inner, sets, q25)
    rec = {"Q25": [q25["rb"], K58.REF_Q25], "Q25_mono": [q25["mono_rb"], K58.REF_Q25_MONO],
           "Q25_seam_ok": seam25["ok"], "Q64": [q64["rb"], REF_Q64], "Q64_mono": [q64["mono_rb"], REF_Q64_MONO],
           "Q64_bitwise": bool(q64["rb"] == REF_Q64), "Q64_mono_bitwise": bool(q64["mono_rb"] == REF_Q64_MONO)}
    rec["ok"] = bool(rec["Q25_seam_ok"] and rec["Q64_bitwise"] and rec["Q64_mono_bitwise"])
    rep = {"format": "c10-prereg65 seam v1", "created": now(), "code": cs,
           "what": "Q25 (p58_eval.build_q25) and Q64 (prereg64 kernel / OPP3 / fold fit, this module's fold loop) on "
                   "inner bit for bit; no OPP4B number", **rec, "seconds": round(time.time() - t0, 1),
           "peak_working_set_gb": peak_gb(st)}
    K58.write_json(SEAM, rep)
    log(json.dumps({k: v for k, v in rep.items() if k != "code"}))
    return 0 if rec["ok"] else 1


# ================================================================ seam P (numpy OPP_P1_v2 on Neon's H rows)
def cmd_seamp(a):
    import numpy.lib.format as nfmt
    import p09_forward as FW09
    import p19_train as T19
    import p31_lib as L31
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(SEAM_P):
        raise SystemExit("seam_p.json exists")
    cs = require_clean([FROZEN, NOTES])
    op1 = K.load_opp_p1()
    mp = FW09.load_map(K58.guard(os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")))
    mu_eq = bool(np.array_equal(op1["mu_t"], np.asarray(mp["mu_x"][:770], np.float64))
                 and np.array_equal(op1["sd_t"], np.asarray(mp["sd_x"][:770], np.float64)))
    probe_prep = K58.guard(T19.PROBE_PREP)
    with np.load(os.path.join(probe_prep, "probe_meta.npz"), allow_pickle=False) as zp:
        p_y = np.asarray(zp["y"], np.int64)
        p_keys = np.asarray(zp["players"]).astype(str)
        p_src = np.asarray(zp["src"]).astype(str)
        p_ref = np.asarray(zp["ref"], np.int64)
    v_prep = K58.guard(os.path.join(REPO, "data", "prereg25", "prep"))
    with np.load(os.path.join(v_prep, "v_meta.npz"), allow_pickle=False) as zv:
        vy = np.asarray(zv["y"], np.int64)
        vk = np.asarray(zv["players"]).astype(str)
    n_probe = len(p_y)
    keys = np.concatenate([p_keys, vk])
    y_all = np.concatenate([p_y, vy])
    with open(os.path.join(v_prep, "prep_record.json"), encoding="utf-8") as f:
        v_rec = json.load(f)
    pg = L31.probe_games(p_src, p_ref, p_keys, T19.SETS_JSON)
    vg = L31.v_games(os.path.join(v_prep, "v_meta.npz"), L31.V_SETS, v_rec["sets_sha256"])
    all_g = L31.concat_games(pg, vg)
    hd_all, jrec = L31.HdTable().join(all_g["rank"], all_g["game_id"], all_g["player_id"], all_g["color"])
    if not jrec["ok"]:
        raise SystemExit("HD join failed")
    # inner players vs probe / V players (record; NOTES 10)
    inner_ids = E64.inner_player_ids()
    kp = {k.split(":", 1)[1] if ":" in k else k for k in set(keys.tolist())}
    overlap_inner = len(kp & inner_ids)
    del pg, vg, all_g
    gc.collect()
    bit_H = {k: holdout_bit_s25(k) for k in set(keys.tolist())}
    sel = (hd_all <= 1) & np.array([bit_H[k] == 1 for k in keys.tolist()], bool)
    y_sel = y_all[sel]
    Xh = np.empty((int(sel.sum()), 770), np.float64)
    w = 0
    for path, m in ((os.path.join(probe_prep, "probe_X.npy"), sel[:n_probe]),
                    (os.path.join(v_prep, "v_X.npy"), sel[n_probe:])):
        with open(K58.guard(path), "rb") as f:
            ver = nfmt.read_magic(f)
            shape, fortran, dtype = (nfmt.read_array_header_1_0(f) if ver == (1, 0) else nfmt.read_array_header_2_0(f))
            if fortran or shape != (len(m), 1540) or dtype != np.float32:
                raise SystemExit(f"{path}: unexpected layout")
            pos = 0
            while pos < shape[0]:
                take = min(4096, shape[0] - pos)
                buf = np.fromfile(f, dtype=np.float32, count=take * 1540).reshape(take, 1540)
                mm = m[pos:pos + take]
                if mm.any():
                    k_ = int(mm.sum())
                    Xh[w:w + k_] = buf[mm, :770]
                    w += k_
                pos += take
    if w != len(Xh):
        raise SystemExit("row count")
    out = {}
    for name, members in (("ens", ("mlp0", "mlp1", "clm0", "clm1")), ("mlp0", ("mlp0",)), ("mlp1", ("mlp1",)),
                          ("clm0", ("clm0",)), ("clm1", ("clm1",))):
        q = K.predict_opp_p1(op1, Xh, members=members)
        lp = np.log(np.clip(q, 1e-300, None))
        nll = float(-lp[np.arange(len(y_sel)), y_sel].mean())
        pred = np.argmax(q @ K.SM, axis=1)
        sc = K.SM[pred, y_sel]
        rb = float(np.mean([sc[y_sel == k].mean() for k in range(10)]))
        ref = NEON_P[name]
        out[name] = {"nll": nll, "rank_balanced": rb, "neon": list(ref),
                     "abs_diff": [abs(nll - ref[0]), abs(rb - ref[1])],
                     "ok": bool(abs(nll - ref[0]) <= P_TOL and abs(rb - ref[1]) <= P_TOL)}
        log(f"seam P {name}: nll {nll:.6f} (Neon {ref[0]}), rb {rb:.6f} (Neon {ref[1]})")
    rep = {"format": "c10-prereg65 seam P v1", "created": now(), "code": cs,
           "what": "numpy OPP_P1_v2 (opp_p1_v2.npz) on Neon's H rows (probe + V, HD <= 1, S25 holdout bit 1; 02a lines "
                   "103-145) vs his torch record (02a_opp_p1_v2_record.json), tolerance 1e-4 on the 5-decimal record",
           "h_rows": int(len(Xh)), "neon_h_rows": 31772, "mu_sd_equal_p1e_map_first_770": mu_eq, "members": out,
           "inner_players_among_probe_V_keys": overlap_inner, "probe_V_players": len(kp),
           "ok": bool(all(x["ok"] for x in out.values()) and len(Xh) == 31772 and mu_eq and overlap_inner == 0),
           "seconds": round(time.time() - t0, 1), "peak_working_set_gb": peak_gb(st)}
    K58.write_json(SEAM_P, rep)
    log(json.dumps({k: v for k, v in rep.items() if k not in ("code", "members")}))
    return 0 if rep["ok"] else 1


# ================================================================ Neon's sections 1-3 (seam N, M8 simulation)
def fit_scalar_oof(terms, y, fold, mask, l2_w=1e-3):
    """04 lines 201-220 verbatim in effect."""
    from scipy.optimize import minimize
    K_t = len(terms)
    oof_Z = np.zeros((len(y), K.L), np.float64)
    ws_list = []
    for f in range(5):
        tr = (fold != f) & mask
        te = (fold == f) & mask
        if not te.any():
            continue
        y_tr = y[tr]
        T_tr = [t[tr] for t in terms]
        T_te = [t[te] for t in terms]

        def obj(p, T_tr=T_tr, y_tr=y_tr):
            Z = sum(p[i] * T_tr[i] for i in range(K_t))
            lp = K.log_softmax(Z)
            return float(-lp[np.arange(len(y_tr)), y_tr].mean() + l2_w * np.sum(p * p))
        res = minimize(obj, np.full(K_t, 0.4), method="L-BFGS-B", bounds=[(0.0, 4.0)] * K_t)
        ws_list.append(res.x)
        oof_Z[te] = sum(res.x[i] * T_te[i] for i in range(K_t))
    return oof_Z, K.decode(oof_Z[mask]), [round(float(x), 4) for x in np.mean(ws_list, axis=0)], \
        np.mean(ws_list, axis=0).tolist()


def neon_p1_compact():
    """04 lines 222-313: S25 hidden vectors and q_P1 of the D / E / E2 lists of prereg39 (mock reuse)."""
    import p09_forward as FW09
    import p10_fuse as F10
    import p19_forward as FW19
    import p39_eval as P39E
    q09_map = FW09.load_map(K58.guard(os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")))
    s25 = []
    for sd in range(3):
        with np.load(K58.guard(os.path.join(REPO, "results", "prereg25", "rank", f"s25_seed{sd}.npz")),
                     allow_pickle=False) as z:
            s25.append({k: np.asarray(z[f"p_{k}"], np.float64) for k in FW19.PARAM_NAMES})
    op1 = K.load_opp_p1()
    with open(K58.guard(os.path.join(REPO, "data", "prereg10", "mock", "sets.json")), encoding="utf-8") as f:
        mock_n = len(json.load(f)["games"])
    mock_means = np.zeros((mock_n, 2, 770), np.float32)
    mock_counts = np.zeros((mock_n, 2), np.int64)
    mock_usable = np.zeros(mock_n, bool)
    for mp_f in F10.chunk_files(K58.guard(os.path.join(REPO, "data", "prereg10", "mock"))):
        with np.load(K58.guard(mp_f), allow_pickle=False) as z:
            r, f_, c = np.asarray(z["ref"], np.int64), np.asarray(z["failed"], bool), np.asarray(z["counts"], np.int64)
            mock_means[r], mock_counts[r], mock_usable[r] = np.asarray(z["means"], np.float32), c, (~f_) & (c[:, 0] > 0)
    rows_l, H_l, q_l, ok_l = [], [[], [], []], [], []
    for name in ("D", "E", "E2"):
        games = P39E.read_list(name)
        n = len(games)
        means = np.zeros((n, 2, 770), np.float32)
        counts = np.zeros((n, 2), np.int64)
        usable = np.zeros(n, bool)
        for ch in F10.chunk_files(K58.guard(os.path.join(REPO, "data", "prereg39", "p1", name))):
            with np.load(K58.guard(ch), allow_pickle=False) as z:
                r, f_, c = np.asarray(z["ref"], np.int64), np.asarray(z["failed"], bool), np.asarray(z["counts"], np.int64)
                means[r], counts[r], usable[r] = np.asarray(z["means"], np.float32), c, (~f_) & (c[:, 0] > 0)
        reuse = np.array([g.get("mock_ref", -1) for g in games], np.int64)
        idx = np.flatnonzero(reuse >= 0)
        if len(idx):
            means[idx], counts[idx], usable[idx] = mock_means[reuse[idx]], mock_counts[reuse[idx]], mock_usable[reuse[idx]]
        H_out = [np.zeros((n, FW19.H_DIM), np.float32) for _ in s25]
        q_p1 = np.zeros((n, K.L), np.float64)
        for s in range(0, n, 8192):
            e = min(n, s + 8192)
            Xs = FW09.standardise_x(FW09.x_raw(means[s:e], counts[s:e]), q09_map["mu_x"], q09_map["sd_x"])
            for par, Hs in zip(s25, H_out):
                Hs[s:e] = FW19.h_np(par, Xs).astype(np.float32)
            q_p1[s:e] = K.predict_opp_p1(op1, means[s:e, 1, :])
        rows_l.append(np.array([int(g["row"]) for g in games], np.int64))
        for i in range(3):
            H_l[i].append(H_out[i])
        q_l.append(q_p1)
        ok_l.append(usable & (counts[:, 1] > 0))
        del means, counts
        gc.collect()
    del mock_means, mock_counts, mock_usable
    gc.collect()
    rows = np.concatenate(rows_l)
    return {"rows": rows, "H": [np.concatenate(H_l[i], axis=0) for i in range(3)], "q": np.concatenate(q_l, axis=0),
            "ok": np.concatenate(ok_l), "row_to_p1": {int(r): i for i, r in enumerate(rows.tolist())}, "s25": s25}


def oof_R(q, r, h, c, f, ok, bayes, lam):
    R = np.ones((len(q), K.L), np.float64)
    tot = K59.table_counts(q, r, h, c, ok)
    Ktot = K58.kernel_counts(q, r, h, c, ok) if lam > 0 else None
    for fo in range(5):
        cnt_f = tot - K59.table_counts(q, r, h, c, ok & (f == fo))
        if bayes:
            Km_f = K58.kernel_matrix(K58.kernel(Ktot - K58.kernel_counts(q, r, h, c, ok & (f == fo)))) if lam > 0 else None
            Mf = K.table_bayes(cnt_f, K_hd1=Km_f, lam_hd1=lam)
        else:
            Mf = K59.table(cnt_f)
        te = np.flatnonzero(ok & (f == fo))
        R[te] = K.game_R(q[te], Mf, h[te], c[te])
    return R


def neon_section1(P1, Mp):
    """04 lines 315-519 for Q64_V0_orig and OPP4B_EX_l25_p1.5 only."""
    import p19_forward as FW19
    D = d_games()
    pr = load_protos()
    sr_D, r_D, h_D, c_D, f_D, hd_raw = D["store_row"], D["r"], D["h"], D["c"], D["f"], D["hd"]
    pid_D, rk_D = D["meta"]["player_id"], D["meta"]["rank"]
    ok_all = D["count"] > 0
    ok_le1 = ok_all & (hd_raw <= 1)
    p1_i = np.array([P1["row_to_p1"].get(int(x), -1) for x in sr_D.tolist()], np.int64)
    has_p1 = p1_i >= 0
    ok_p1 = np.zeros(len(sr_D), bool)
    ok_p1[has_p1] = P1["ok"][p1_i[has_p1]]
    q_p1 = np.full((len(sr_D), K.L), 0.1, np.float64)
    q_p1[has_p1] = P1["q"][p1_i[has_p1]]
    H_D = [np.zeros((len(sr_D), FW19.H_DIM), np.float32) for _ in range(3)]
    for i in range(3):
        H_D[i][has_p1] = P1["H"][i][p1_i[has_p1]]
    q_V0 = K64.q_opp(D["emb"], D["Wn"], D["wr"], pr["proto"], pr["rank"], D["own"])
    q_EX = K.q_c49(D["emb"], D["Wn"], D["wr"], pr["proto"], pr["rank"], pr["src"], D["own"])
    R_p1 = K.game_R(q_p1, Mp, h_D, c_D)
    R_p1[~ok_p1] = 1.0
    R_V0 = oof_R(q_V0, r_D, h_D, c_D, f_D, ok_all, False, 0.0)
    R_EX = oof_R(q_EX, r_D, h_D, c_D, f_D, ok_all, True, K.LAM_HD1)
    by_p = defaultdict(lambda: {0: [], 1: [], "r": 0, "f": 0, "is_H": False})
    for i in range(len(sr_D)):
        if ok_le1[i] and ok_p1[i]:
            p = pid_D[i]
            by_p[p][int(c_D[i])].append(i)
            by_p[p]["r"] = int(r_D[i])
            by_p[p]["f"] = int(f_D[i])
            by_p[p]["is_H"] = (holdout_bit_s25(f"{rk_D[i]}:{p}") == 1)
    rng = np.random.default_rng(20261122)
    grp, y, fo, kk, isH = [], [], [], [], []
    for p in sorted(by_p):
        d = by_p[p]
        for ci in (0, 1):
            pool = d[ci]
            for kv in (5, 10, 15, 20):
                if len(pool) >= kv:
                    for _ in range(5):
                        grp.append(rng.choice(pool, size=kv, replace=False).tolist())
                        y.append(d["r"])
                        fo.append(d["f"])
                        kk.append(kv)
                        isH.append(d["is_H"])
    y, fo, kk, isH = (np.asarray(y, np.int64), np.asarray(fo, np.int64), np.asarray(kk, np.int64),
                      np.asarray(isH, bool))
    _, _, lp_s25, _ = FW19.set_term(P1["s25"], H_D, grp, len(sr_D), scale=4.0)
    E_s25 = ((kk / 12.0) ** 0.40)[:, None] * (4.0 * lp_s25)

    def mk(lg):
        return np.array([10.0 * lg[g].mean(axis=0) for g in grp])

    E_V0 = mk(np.log(R_V0 + K.EPS))
    E_4B = mk(K.lr_lg(q_EX, R_EX, R_p1, K.P_POW))
    out = {"n_players_DH": int(sum(d["is_H"] for d in by_p.values())), "n_questions_DH": int(isH.sum())}
    for name, terms in (("Q64_proxy", [E_s25, E_V0]), ("OPP4B", [E_s25, E_4B])):
        _, pred, w4, wf = fit_scalar_oof(terms, y, fo, isH)
        rb = K.rb_score(pred, y[isH])
        out[name] = {"rb": rb, "rb_5": round(rb, 5), "weights_4": w4, "weights": wf}
    return out


def cmd_seamn(a):
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(SEAM_N):
        raise SystemExit("seam_n.json exists")
    cs = require_clean([FROZEN, NOTES, KERNEL])
    _M, Mp, _k = load_kernel()
    P1 = neon_p1_compact()
    log(f"P1 compact {len(P1['rows'])} games ({time.time() - t0:.0f}s)")
    s1 = neon_section1(P1, Mp)
    chk = {"Q64_proxy": s1["Q64_proxy"]["rb_5"] == NEON_N["Q64_proxy"],
           "OPP4B": s1["OPP4B"]["rb_5"] == NEON_N["OPP4B"],
           "OPP4B_weights": s1["OPP4B"]["weights_4"] == NEON_N["weights_OPP4B"],
           "abs_diff_OPP4B": abs(s1["OPP4B"]["rb"] - NEON_N["OPP4B"]),
           "abs_diff_Q64_proxy": abs(s1["Q64_proxy"]["rb"] - NEON_N["Q64_proxy"])}
    rep = {"format": "c10-prereg65 seam N v1", "created": now(), "code": cs,
           "what": "Neon's D n H (04 section 1) re-run with p65_lib: Q64_proxy and OPP4B_EX_l25_p1.5 (NOTES 7 rule: "
                   "unrounded value rounded to 5 decimals = record, weights rounded to 4 = record)",
           "section1": s1, "neon": NEON_N, "check": chk,
           "ok": bool(chk["Q64_proxy"] and chk["OPP4B"] and chk["OPP4B_weights"]),
           "seconds": round(time.time() - t0, 1), "peak_working_set_gb": peak_gb(st)}
    K58.write_json(SEAM_N, rep)
    log(json.dumps({k: v for k, v in rep.items() if k != "code"}))
    return 0 if rep["ok"] else 1


def neon_section2_3(P1, Mp, log_):
    """04 lines 521-824 (section 2 for the OPP4B_S0_l25_p1.5 weights and per-game values, section 3 M8 only)."""
    import p19_forward as FW19
    import p31_lib as L31
    with np.load(K58.guard(K58.FEATS_META), allow_pickle=False) as zm:
        meta = {k: np.asarray(zm[k]).astype(str) for k in ("rank", "game_id", "player_id", "color")}
    with np.load(K58.guard(K58.OPP_D), allow_pickle=False) as zd:
        s0_D, c0_D, sr0_D = np.asarray(zd["emb"], np.float32), np.asarray(zd["count"], np.int64), np.asarray(zd["store_row"], np.int64)
    with np.load(K58.guard(K58.OPP_E), allow_pickle=False) as ze:
        s0_E, c0_E, sr0_E = np.asarray(ze["emb"], np.float32), np.asarray(ze["count"], np.int64), np.asarray(ze["store_row"], np.int64)
    sr = np.concatenate([sr0_D, sr0_E])
    s0 = np.concatenate([s0_D, s0_E], axis=0)
    c0 = np.concatenate([c0_D, c0_E], axis=0)
    del s0_D, s0_E, c0_D, c0_E, sr0_D, sr0_E
    keep = np.array([int(x) in P1["row_to_p1"] for x in sr.tolist()], bool)
    sr, s0, c0 = sr[keep], s0[keep], c0[keep]
    idx = np.array([P1["row_to_p1"][int(x)] for x in sr.tolist()], np.int64)
    rk, gid, pid, col = meta["rank"][sr], meta["game_id"][sr], meta["player_id"][sr], meta["color"][sr]
    r_, c_ = K58.rank_index(rk), K58.colour_index(col)
    hd_raw, jr = L31.HdTable().join(rk, gid, pid, col)
    if not jr["ok"]:
        raise SystemExit("HD join failed")
    h_ = K58.hd_group(hd_raw)
    f_ = np.array([K58.player_fold(p) for p in pid.tolist()], np.int64)
    ok = P1["ok"][idx] & (c0 > 0) & (hd_raw <= 1)
    Wn0, wr0, _ = K58.load_head(K58.guard(K58.HEAD_NPZ))
    with np.load(K58.guard(K59.PROTO), allow_pickle=False) as zp:
        p0_ids, p0_proto, p0_rk = np.asarray(zp["player_id"]).astype(str), np.asarray(zp["proto"], np.float64), np.asarray(zp["rank"], np.int64)
    p0_map = {p: i for i, p in enumerate(p0_ids.tolist())}
    own0 = np.array([p0_map.get(p, -1) for p in pid.tolist()], np.int64)
    pts = K59.class_parts(s0, Wn0, wr0, p0_proto, p0_rk, own=own0, tau1=(70,), tau2=(50,))
    del s0
    gc.collect()
    N_h0, N_p0 = np.bincount(wr0, minlength=K.L).astype(np.float64), np.bincount(p0_rk, minlength=K.L).astype(np.float64)
    SA0, mA0 = pts["A"][70]
    SB0, mB0 = pts["B"][50]
    m0 = np.maximum(mA0, mB0 - 1.0)
    q70 = (SA0 / N_h0[None, :] * N_h0.mean()) * np.exp(mA0 - m0)[:, None] + (SB0 / N_p0[None, :] * N_p0.mean()) * np.exp(mB0 - 1.0 - m0)[:, None]
    q70 /= np.maximum(q70.sum(axis=1, keepdims=True), 1e-300)
    del pts
    R_l25 = oof_R(q70, r_, h_, c_, f_, ok, True, 0.25)
    R_p1 = K.game_R(P1["q"][idx], Mp, h_, c_)
    by_p = defaultdict(lambda: {0: [], 1: [], "r": 0, "f": 0, "is_H": False})
    for i in range(len(sr)):
        if ok[i]:
            p = pid[i]
            by_p[p][int(c_[i])].append(i)
            by_p[p]["r"] = int(r_[i])
            by_p[p]["f"] = int(f_[i])
            by_p[p]["is_H"] = (holdout_bit_s25(f"{rk[i]}:{p}") == 1)
    rng = np.random.default_rng(20261123)
    grp, y, fo, kk = [], [], [], []
    for p in sorted(by_p):
        d = by_p[p]
        if not d["is_H"]:
            continue
        for ci in (0, 1):
            pool = d[ci]
            for kv in (5, 10, 15, 20):
                if len(pool) >= kv:
                    for _ in range(5):
                        grp.append(rng.choice(pool, size=kv, replace=False).tolist())
                        y.append(d["r"])
                        fo.append(d["f"])
                        kk.append(kv)
    y, fo, kk = np.asarray(y, np.int64), np.asarray(fo, np.int64), np.asarray(kk, np.int64)
    H_DE = [P1["H"][i][idx] for i in range(3)]
    _, _, lp_s25, _ = FW19.set_term(P1["s25"], H_DE, grp, len(sr), scale=4.0)
    E_s25 = ((kk / 12.0) ** 0.40)[:, None] * (4.0 * lp_s25)
    lg4b = K.lr_lg(q70, R_l25, R_p1, K.P_POW)
    E_4b = np.array([10.0 * lg4b[g].mean(axis=0) for g in grp])
    _, pred, w_eh, w_full = fit_scalar_oof([E_s25, E_4b], y, fo, np.ones(len(y), bool))
    sec2 = {"n_players": int(sum(d["is_H"] for d in by_p.values())), "n_questions": int(len(y)),
            "OPP4B_S0_l25_p1.5": {"rb": K.rb_score(pred, y), "weights": w_eh, "weights_unrounded": w_full}}
    log_(f"section 2: {json.dumps(sec2)}")
    # ---- section 3 (04 lines 669-824), M8 only
    with np.load(K58.guard(K64.OWN_E), allow_pickle=False) as ze:
        own_E, cnt_E, sr_E = np.asarray(ze["emb"], np.float32), np.asarray(ze["count"], np.int64), np.asarray(ze["store_row"], np.int64)
    sr_to = {int(x): i for i, x in enumerate(sr.tolist())}
    eh = defaultdict(lambda: {"B": [], "W": [], "r": 0})
    own_l2 = K64.l2(own_E)
    for i in range(len(sr_E)):
        if cnt_E[i] <= 0:
            continue
        di = sr_to.get(int(sr_E[i]), -1)
        if di < 0 or not ok[di]:
            continue
        if holdout_bit_s25(f"{rk[di]}:{pid[di]}") != 1:
            continue
        eh[pid[di]][col[di]].append((i, di))
        eh[pid[di]]["r"] = int(r_[di])
    elig = sorted(p for p, d in eh.items() if len(d["B"]) >= 20 and len(d["W"]) >= 20)
    w = w_eh
    sec3 = {"n_elig_players": len(elig), "n_pools": 100, "base_weights": w, "modes": {}}
    for mode, _frac in (("zero_repeat_0pct", 0.0), ("moderate_repeat_35pct", 0.35), ("paired_repeat_100pct", 1.0)):
        gains, changed, edges, base_rbs = [], [], [], []
        for sd in range(100):
            rp = np.random.default_rng(9000 + sd)
            if mode == "zero_repeat_0pct":
                sel = [(p, rp.choice(["B", "W"]), int(rp.choice([5, 10, 15, 20]))) for p in elig]
            elif mode == "moderate_repeat_35pct":
                sel = []
                reps = set(rp.choice(elig, size=int(round(0.35 * len(elig))), replace=False).tolist())
                for p in elig:
                    c1 = str(rp.choice(["B", "W"]))
                    if p in reps:
                        sel.append((p, c1, int(rp.choice([5, 10]))))
                        sel.append((p, c1, int(rp.choice([5, 10]))))
                    else:
                        sel.append((p, c1, int(rp.choice([5, 10, 15, 20]))))
            else:
                sel = []
                for p in elig:
                    c1 = str(rp.choice(["B", "W"]))
                    sel.append((p, c1, 10))
                    sel.append((p, c1, 10))
            used = defaultdict(set)
            eg, g_own, g_de, qc, qy, qn = [], [], [], [], [], []
            cur = 0
            for p, cc, kv in sel:
                avail = [pp for pp in eh[p][cc] if pp[0] not in used[(p, cc)]]
                if len(avail) < kv:
                    continue
                ch_i = rp.choice(len(avail), size=kv, replace=False)
                chosen = [avail[j] for j in ch_i]
                for oi, _di in chosen:
                    used[(p, cc)].add(oi)
                eg.append(own_l2[[oi for oi, _ in chosen]])
                g_own.append(list(range(cur, cur + kv)))
                cur += kv
                g_de.append([di for _, di in chosen])
                qc.append(cc)
                qy.append(eh[p]["r"])
                qn.append(kv)
            S = K64.cc_whiten_sim(np.vstack(eg), g_own, qc)
            qy, qn = np.asarray(qy, np.int64), np.asarray(qn, np.int64)
            _, _, lp_p, _ = FW19.set_term(P1["s25"], H_DE, g_de, len(sr), scale=4.0)
            E_s = ((qn / 12.0) ** 0.40)[:, None] * (4.0 * lp_p)
            E_o = np.array([10.0 * lg4b[g].mean(axis=0) for g in g_de])
            lp_base = K.log_softmax(w[0] * E_s + w[1] * E_o)
            pr_b = K.decode(lp_base)
            rb_b = K.rb_score(pr_b, qy)
            base_rbs.append(rb_b)
            pr_m, _lpn, ned, _p0 = K.m8(lp_base, S, qc)
            gains.append(K.rb_score(pr_m, qy) - rb_b)
            changed.append(int((pr_m != pr_b).sum()))
            edges.append(ned)
        sec3["modes"][mode] = {"base_rb_mean": float(np.mean(base_rbs)), "M8": {
            "gain_mean": float(np.mean(gains)), "gain_min": float(np.min(gains)), "gain_max": float(np.max(gains)),
            "changed_mean": float(np.mean(changed)), "changed_max": int(np.max(changed)),
            "pools_with_any_change": int(sum(1 for x in changed if x > 0)), "edges_mean": float(np.mean(edges))}}
        log_(f"M8 {mode}: {json.dumps(sec3['modes'][mode])}")
    return sec2, sec3


def cmd_mergesim(a):
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(MERGE_SIM):
        raise SystemExit("merge_sim.json exists")
    cs = require_clean([FROZEN, NOTES, KERNEL, EVAL])
    _M, Mp, _k = load_kernel()
    P1 = neon_p1_compact()
    sec2, sec3 = neon_section2_3(P1, Mp, log)
    z0, z35 = sec3["modes"]["zero_repeat_0pct"]["M8"], sec3["modes"]["moderate_repeat_35pct"]["M8"]
    passed = bool(z0["changed_max"] == 0 and z0["pools_with_any_change"] == 0 and z35["gain_mean"] > 0)
    neon = {"weights": NEON_SIM_W, "n_elig_players": 46, "base_rb_0pct": 0.75324,
            "M8": {"0pct": [0.0, 0], "35pct": [0.00039, 46], "100pct": [0.00344, 90]}}
    rep = {"format": "c10-prereg65 merge simulation v1", "created": now(), "code": cs,
           "what": "Neon 04 section 3 (M8) on E n H with the S25 + OPP4B_S0 base and the section-2 weights; pass = 0 "
                   "questions changed in all 100 pools at 0% AND mean gain > 0 at 35% (PLAN step 4)",
           "section2": sec2, "section3": sec3, "neon_record": neon,
           "reproduces_neon": {"weights": sec2["OPP4B_S0_l25_p1.5"]["weights"] == NEON_SIM_W,
                               "n_elig": sec3["n_elig_players"] == 46,
                               "gain_35_5dec": round(z35["gain_mean"], 5) == 0.00039,
                               "gain_100_5dec": round(sec3["modes"]["paired_repeat_100pct"]["M8"]["gain_mean"], 5) == 0.00344},
           "passed": passed, "rule": "0% changed = 0 in 100/100 pools and 35% mean gain > 0",
           "seconds": round(time.time() - t0, 1), "peak_working_set_gb": peak_gb(st)}
    K58.write_json(MERGE_SIM, rep)
    log(json.dumps({k: rep[k] for k in ("reproduces_neon", "passed", "seconds", "peak_working_set_gb")}))
    return 0


# ================================================================ the judgement
def hd_groups_record(inner, cand, base, hd):
    E23 = E58.harness()[1]
    di = inner.d["inner"]
    share = np.asarray(di["S_eval"] @ (np.asarray(hd) == 1).astype(np.float64)).ravel()
    Xe, ms = inner.Xe, inner.ms
    ye, pe = np.asarray(Xe.y), np.asarray(Xe.player)
    n_players = int(ms["n_players"])
    return {name: E23.subset_diff(cand["sc"], base["sc"], ye, pe, n_players, m)
            for name, m in (("hd1_eq_0", share == 0), ("hd1_gt_0", share > 0), ("hd1_gt_025", share > 0.25))}


def cmd_eval(a):
    st = K58.limits(log)
    t0 = time.time()
    if now() >= HARD_STOP:
        raise SystemExit("past the hard stop")
    for p in (KERNEL, SEAM, SEAM_P, SEAM_N):
        if not committed_clean(p) or not K58.read_json(p)["ok" if p != KERNEL else "format"]:
            raise SystemExit(f"{rel(p)} is not committed / ok")
    if os.path.exists(EVAL) or os.path.exists(STARTED):
        raise SystemExit("eval.json / eval_started.json exist: judged once")
    cs = require_clean([FROZEN, NOTES, KERNEL, SEAM, SEAM_P, SEAM_N])
    M, Mp, krec = load_kernel()
    pr = load_protos()
    mods, inner, terms7, Emap, q25, seam25 = E58.build_q25()
    E41, E23, r, K23 = mods
    sets = E64.inner_sets(inner, terms7, Emap)
    q64 = q64_rebuild(inner, sets, q25)
    srec = {"Q25": [q25["rb"], q25["mono_rb"]], "Q25_seam_ok": seam25["ok"], "Q64": [q64["rb"], q64["mono_rb"]],
            "Q64_bitwise": bool(q64["rb"] == REF_Q64 and q64["mono_rb"] == REF_Q64_MONO)}
    log(f"seams in the judgement run: {srec} ({time.time() - t0:.0f}s)")
    if not (seam25["ok"] and srec["Q64_bitwise"]):
        raise SystemExit("seam failed inside the judgement run")
    os.makedirs(K.DATA, exist_ok=True)
    K58.write_json(STARTED, {"format": "c10-prereg65 eval started", "created": now(), "code": cs})
    # ---- OPP4B (first inner number of it below)
    lg, trec, hd = inner_opp4b(inner, M, Mp, pr)
    O = pooled(inner, lg)
    cand = fuse_inner(inner, sets, q25, O)
    vs64 = inner.compare(cand, q64)
    vs25 = inner.compare(cand, q25)
    gates = K23.gates(vs64["inner_diff"], vs64["inner_ci95"], vs64["mono_diff"], vs64["mono_ci95"],
                      vs64["mono_by_colour"])
    gates25 = K23.gates(vs25["inner_diff"], vs25["inner_ci95"], vs25["mono_diff"], vs25["mono_ci95"],
                        vs25["mono_by_colour"])
    adopt = bool(gates["passed"])
    log(f"Q65 {cand['rb']!r} vs Q64 {q64['rb']!r}: {json.dumps(gates)}")
    ones = {w: np.ones(len(O[w]), bool) for w in ("eval", "mono")}
    alone = E23.alone(inner, {w: O[w] for w in ("eval", "mono")}, ones)
    rep = {"format": "c10-prereg65 eval v1", "created": now(), "seconds": round(time.time() - t0, 1), "code": cs,
           "what": "PLAN.md steps 1-3, judged once on inner: Q65 = w_A * log_softmax(Q25 with its fold parameters) + "
                   "w_O * OPP4B (Neon's two-channel term, 10 x mean, flat), weights by prereg64's scalar fit per player "
                   "fold; rule B; vs Q64 (the current upload). Inner used for the fourth time (prereg58, 59, 64, 65).",
           "adopt_steps_1_3": adopt,
           "status": ("steps 1-3 passed all three gates vs Q64" if adopt else "gates vs Q64 not passed: not adopted"),
           "seam": srec, "term": trec, "kernel_sha256_lf": K58.sha_lf(KERNEL),
           "candidate": {"name": "Q65", "inner_rank_balanced": cand["rb"], "single_colour_rank_balanced": cand["mono_rb"],
                         "single_colour_by_colour": {c: cand["mrec"]["by_colour"][c]["rank_balanced"]
                                                     for c in cand["mrec"]["by_colour"]},
                         "vs_Q64": vs64, "vs_Q25": vs25},
           "baselines": {"Q64": {"inner_rank_balanced": q64["rb"], "single_colour_rank_balanced": q64["mono_rb"]},
                         "Q25": {"inner_rank_balanced": q25["rb"], "single_colour_rank_balanced": q25["mono_rb"]}},
           "gates": gates, "record_gates_vs_Q25": gates25,
           "weights_by_fold": cand["folds"],
           "record_only": {"OPP4B_alone_rule_b": alone,
                           "by_game_count_vs_Q64": E23.by_count(inner, cand, q64),
                           "by_level_vs_Q64": E23.by_level(inner, cand, q64),
                           "by_hd_group_vs_Q64": hd_groups_record(inner, cand, q64, hd),
                           "by_game_count_vs_Q25": E23.by_count(inner, cand, q25),
                           "by_level_vs_Q25": E23.by_level(inner, cand, q25)},
           "peak_working_set_gb": peak_gb(st)}
    K58.write_json(EVAL, rep)
    log(f"-> {EVAL}: {rep['status']} ({time.time() - t0:.0f}s, peak {peak_gb(st)} GB)")
    return 0


# ================================================================ RESULTS.md
def results_text(e, ms=None):
    c, g = e["candidate"], e["gates"]
    v4, v5 = c["vs_Q64"], c["vs_Q25"]
    ro = e["record_only"]
    A = []
    a = A.append
    a("# prereg65 (competition 2): OPP4B (Neon's two-channel opponent term) on Q25 - steps 1-3 "
      + ("PASS" if e["adopt_steps_1_3"] else "NOT adopted"))
    a("")
    a(f"Written mechanically from `eval.json` ({e['created']}); numbers only. PLAN.md, NOTES.md, frozen.json. "
      "Inner used for the fourth time (prereg58, 59, 64, 65).")
    a("")
    a("## Seams")
    a("")
    s = e["seam"]
    a(f"- In the judgement run: Q25 {s['Q25'][0]!r} / {s['Q25'][1]!r} (seam ok {s['Q25_seam_ok']}); Q64 "
      f"{s['Q64'][0]!r} / {s['Q64'][1]!r} bit for bit {s['Q64_bitwise']}. seam.json, seam_p.json, seam_n.json committed before.")
    a("")
    a("## Judgement (inner, once)")
    a("")
    a("| | Q64 | Q65 | diff | 95% interval | gate |")
    a("|---|---|---|---|---|---|")
    a(f"| inner eval | {e['baselines']['Q64']['inner_rank_balanced']:.7f} | {c['inner_rank_balanced']:.7f} | "
      f"{v4['inner_diff']:+.5f} | [{v4['inner_ci95'][0]:+.5f}, {v4['inner_ci95'][1]:+.5f}] | "
      f"G1 {'pass' if g['G1']['passed'] else 'no'}; G2 {'pass' if g['G2']['passed'] else 'no'} |")
    bc = v4["mono_by_colour"]
    a(f"| single colour | {e['baselines']['Q64']['single_colour_rank_balanced']:.7f} | "
      f"{c['single_colour_rank_balanced']:.7f} | {v4['mono_diff']:+.5f} | [{v4['mono_ci95'][0]:+.5f}, "
      f"{v4['mono_ci95'][1]:+.5f}] | G4 {'pass' if g['G4']['passed'] else 'no'} (B {bc.get('B'):+.4f}, "
      f"W {bc.get('W'):+.4f}) |")
    a("")
    a(f"**steps 1-3: {e['status']}.**")
    a("")
    a("## Record only")
    a("")
    a(f"- vs Q25: inner {v5['inner_diff']:+.5f} [{v5['inner_ci95'][0]:+.5f}, {v5['inner_ci95'][1]:+.5f}], single "
      f"colour {v5['mono_diff']:+.5f} [{v5['mono_ci95'][0]:+.5f}, {v5['mono_ci95'][1]:+.5f}].")
    a("- Weights by fold (w_A, w_O): " + "; ".join(f"{w['w_A']:.4f}, {w['w_O']:.4f}" for w in e["weights_by_fold"]) + ".")
    al = ro["OPP4B_alone_rule_b"]
    a(f"- OPP4B alone (rule B): {al['inner_rank_balanced']:.4f} (single colour {al['single_colour_rank_balanced']:.4f}).")
    a("- By rank class (eval, Q65 - Q64): " + ", ".join(f"{x['level']} {x['diff']:+.4f}"
                                                         for x in ro["by_level_vs_Q64"]["eval"]) + ".")
    b4 = ro["by_game_count_vs_Q64"]
    a("- Eval questions with exactly n games (vs Q64): " + "; ".join(
        f"{n} {b4['eval_by_n_exact'][n]['diff']:+.4f}" for n in sorted(b4["eval_by_n_exact"], key=int)) + ".")
    hg = ro["by_hd_group_vs_Q64"]
    a("- By share of HD = 1 games (eval, vs Q64): " + "; ".join(
        f"{k} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] (n {v['n_questions']})" for k, v in hg.items()) + ".")
    t = e["term"]
    a(f"- Inner question games {t['games_in_questions']}; without c49 sample {t['games_without_c49_sample']}; without "
      f"opponent P1 {t['games_without_opponent_p1']}; mean max q_c49 {t['mean_max_q_c49_record']:.4f}; mean alpha "
      f"{t['mean_alpha_record']:.4f}.")
    a("")
    if ms:
        a("## M8 merge simulation (Neon 04 section 3; E n H, 100 pools)")
        a("")
        a(f"- Base weights (section 2 refit) {ms['section2']['OPP4B_S0_l25_p1.5']['weights']} (Neon {ms['neon_record']['weights']}); "
          f"eligible players {ms['section3']['n_elig_players']}.")
        for mode, x in ms["section3"]["modes"].items():
            m = x["M8"]
            a(f"- {mode}: base {x['base_rb_mean']:.5f}; gain mean {m['gain_mean']:+.5f} (min {m['gain_min']:+.5f}, max "
              f"{m['gain_max']:+.5f}); changed per pool {m['changed_mean']:.2f} (max {m['changed_max']}); pools with a "
              f"change {m['pools_with_any_change']}/100.")
        a("")
        a(f"**M8 simulation: {'PASS' if ms['passed'] else 'NOT passed'}** ({ms['rule']}).")
        a("")
    return "\n".join(A)


def cmd_results(a):
    if not committed_clean(EVAL):
        raise SystemExit("eval.json is not committed and clean")
    ms = None
    if os.path.exists(MERGE_SIM):
        if not committed_clean(MERGE_SIM):
            raise SystemExit("merge_sim.json is not committed and clean")
        ms = K58.read_json(MERGE_SIM)
    with open(K58.guard(RESULTS), "w", encoding="utf-8", newline="\n") as f:
        f.write(results_text(K58.read_json(EVAL), ms))
    log(f"-> {RESULTS}")
    return 0


# ================================================================ freeze (only after a pass)
def cmd_freeze(a):
    import p04_rank_fuse as p4
    st = K58.limits(log)
    t0 = time.time()
    out = os.path.join(RANK_OUT, "frozen_config.json")
    if os.path.exists(out):
        raise SystemExit("frozen_config.json exists (run once)")
    cs = require_clean([KERNEL, SEAM, SEAM_P, SEAM_N, EVAL, MERGE_SIM])
    ev, ms = K58.read_json(EVAL), K58.read_json(MERGE_SIM)
    if ev.get("adopt_steps_1_3") is not True:
        raise SystemExit("eval.json: steps 1-3 not passed")
    if K58.sha_lf(KERNEL) != ev["kernel_sha256_lf"]:
        raise SystemExit("kernel.json is not the judged one")
    M, Mp, krec = load_kernel()
    pr = load_protos()
    mods, inner, terms7, Emap, q25, seam = E58.build_q25()
    if not seam["ok"]:
        raise SystemExit("seam Q25 failed")
    mf, Ef7 = q25["aux"]["mf"], q25["aux"]["Ef"]
    Xf = inner.Xf
    q25f = os.path.join(REPO, "results", "prereg25", "rank", "frozen_config.json")
    fc25 = K58.read_json(q25f)
    th7, rec7 = p4.gen_fit_record(mf, Ef7, Xf.n, Xf.y)
    ref7 = np.asarray(fc25["params"]["theta"], np.float64)
    seam_z = {"theta_bitwise": bool(np.array_equal(th7, ref7)), "loss_bitwise": bool(rec7["loss"] == fc25["refit"]["loss"])}
    seam_z["ok"] = bool(seam_z["theta_bitwise"] and seam_z["loss_bitwise"])
    if not seam_z["ok"]:
        raise SystemExit("seam Z failed")
    lg, trec, _hd = inner_opp4b(inner, M, Mp, pr)
    seam_t = {"mean_max_q": [trec["mean_max_q_c49_record"], ev["term"]["mean_max_q_c49_record"]],
              "mean_alpha": [trec["mean_alpha_record"], ev["term"]["mean_alpha_record"]]}
    seam_t["ok"] = bool(seam_t["mean_max_q"][0] == seam_t["mean_max_q"][1] and seam_t["mean_alpha"][0] == seam_t["mean_alpha"][1])
    if not seam_t["ok"]:
        raise SystemExit(f"seam T failed {seam_t}")
    O_fit = K.question_term(inner.d["inner"]["S_fit"], lg)
    z, _t = p4.gen_logits(ref7, mf, Ef7, Xf.n)
    ws, frec = K64.fit_weights([K64.log_softmax(z), O_fit], Xf.y)
    os.makedirs(RANK_OUT, exist_ok=True)
    shutil.copyfile(KERNEL, os.path.join(RANK_OUT, "kernel.json"))
    rep = {"format": "c10-prereg65 rank frozen_config v1", "created": now(), "seconds": round(time.time() - t0, 1),
           "what": "Q65 = w_A * log_softmax(Q25 frozen gen_logits) + w_O * OPP4B; (w_A, w_O) fitted once on all inner fit "
                   "questions; M8 applied per test pool when merge_sim passed",
           "code": cs, "adopted_on": {"eval_sha256_lf": K58.sha_lf(EVAL), "gates_vs_Q64": ev["gates"],
                                      "merge_sim_sha256_lf": K58.sha_lf(MERGE_SIM), "merge_sim_passed": ms["passed"]},
           "q25_theta": ref7.tolist(), "w_A": float(ws[0]), "w_O": float(ws[1]), "fit": frec,
           "fit_questions": int(len(Xf)), "w_at_bound": [bool(w <= 1e-8 or w >= K64.W_BOUND - 1e-8) for w in ws],
           "term": {"name": TERM, "tau1": K.TAU1, "tau2": K.TAU2, "b_E": K.B_E, "b_X": K.B_X, "lam_hd1": K.LAM_HD1,
                    "p_pow": K.P_POW, "scale": K.SCALE, "kernel_sha256_lf": K58.sha_lf(KERNEL),
                    "opp_p1_sha256": K.OPP_P1_SHA, "m_p1_file_sha256": K.SCOND_SHA,
                    "prototypes_sha256": K58.sha256_file(K64.PROTO), "head_sha256": K58.sha256_file(K64.HEAD),
                    "inner_mean_max_q": trec["mean_max_q_c49_record"], "inner_mean_alpha": trec["mean_alpha_record"]},
           "merge": {"apply": bool(ms["passed"]), "name": "M8", "threshold": K.M8_THR, "guard": K.M8_GUARD,
                     "beta": K.M8_BETA, "lam": K64.MERGE_LAM},
           "seams": {"Q25_oof": seam, "Z": seam_z, "T": seam_t}, "peak_working_set_gb": peak_gb(st)}
    K58.write_json(out, rep)
    log(f"frozen Q65: w_A {ws[0]:.4f}, w_O {ws[1]:.4f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"kernel": cmd_kernel, "seam": cmd_seam, "seamp": cmd_seamp, "seamn": cmd_seamn, "eval": cmd_eval,
            "results": cmd_results, "mergesim": cmd_mergesim, "freeze": cmd_freeze}
    for c in cmds:
        sub.add_parser(c)
    a = ap.parse_args(argv)
    return cmds[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
