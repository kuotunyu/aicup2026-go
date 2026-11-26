"""prereg47 explore: the per-move distribution term T_M (PLAN.md). Seam, D selection, the one E judgement, official
pools. CPU only; no new P1 extraction (the stored per-position arrays of the P1 chunks are read).

    PY results/prereg47/explore/p47_eval.py seam                # -> seam.json + data/prereg47/models/lda.npz
    PY results/prereg47/explore/p47_eval.py select              # -> tune.json (commit before judge)
    PY results/prereg47/explore/p47_eval.py judge --e-use 6     # once -> eval.json
    PY results/prereg47/explore/p47_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg47/explore/p47_eval.py assemble --src <official tests dir>

Base = v2-6 (results/prereg43/explore/tune.json + eval.json), rebuilt by prereg46's base_parts (prereg45's v26,
base_fused, seam_scores): FP = F_base + wP z_P before its CSLS. M47 = CSLS_k(FP + wM z(T_M)).
Per game (PLAN.md 1), target side (pos_side 0) of the chunk's per-position arrays: segments pos_turn < 60, 60-149,
>= 150 and the whole game; per segment pos_logp and pos_entropy each mean, sd (ddof 0), 10 / 50 / 90 percentiles
(numpy "linear"), the share of pos_logp < -4 and the segment's share of the game's target moves (12); a segment
without a move -> its 12 values are 0 and its flag is 1 (4 flags) -> 52 values. Usable = chunk present, not
failed, per-position arrays present, >= 10 target positions.
T_M (PLAN.md 2): tools/p10_fuse.fit_p1_lda on the usable target-side games of S = L (prereg39) + M1 (prereg45),
players with >= 2 usable games; dims 16 / 32 = the first 16 / 32 LDA directions (one fit). Per pool, as v2-6's P1
term (prereg39 _term): "b4" = prereg38's B4 of the pool's projected games (candidate ids only), cosine of the
prototypes; "none" = group means, - squared distance (p06_style.lda_term). z = zscore_rows per question; a question
without a usable game -> row 0; a candidate without one -> the row minimum of the others.
Grid (fixed order): dim (16, 32) x pw (b4, none) x wM (0.25, 0.5, 1, 2, 4, 8) x k (10, 20, 40); first maximum of
the D mean. Gate: D(M47) - 0.7467766213119217 >= 0.01. judge (E pools, use 6): mean >= +0.01, pool bootstrap
(10,000, seed 43) 95% low > 0, higher in >= 18 of 20. Nothing here opens the validation-answer folder or runs
tools/score.py.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P46 = os.path.join(REPO, "results", "prereg46", "explore")
for p in (P46, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p46_eval as E46  # noqa: E402   (v2-6 rebuilt bit for bit: base_parts; imports p45 / p43 / p39)

E43, E45, P39E = E46.E43, E46.E45, E46.P39E
BK, EV, st, L, F10 = E46.BK, E46.EV, E46.st, E46.L, E46.F10

D_V26, E_V26 = E46.D_V26, E46.E_V26
V26_APPLY = E46.V26_APPLY
V1_SUBMISSION = E46.V1_SUBMISSION
DATA = os.path.join(REPO, "data", "prereg47")
MODEL_FILE = os.path.join(DATA, "models", "lda.npz")
SEAM_JSON = os.path.join(HERE, "seam.json")
OUT_APPLY = os.path.join(REPO, "results", "prereg47", "apply")
TARGET_SIDE = 0
SEG_EDGES = (60, 150)                   # pos_turn < 60, 60..149, >= 150
SEGMENTS = ("open", "mid", "end", "all")
PCTS = (10, 50, 90)
BLUNDER = -4.0
MIN_MOVES = 10
N_STATS = 12
N_FEAT = len(SEGMENTS) * N_STATS + len(SEGMENTS)       # 52
S_SETS = ("L", "M1")
DIM_GRID = (16, 32)
PW_GRID = ("b4", "none")
WM_GRID = (0.25, 0.5, 1, 2, 4, 8)
K_GRID = (10, 20, 40)
SEAM_K = 20
SEAM_TERM = (DIM_GRID[0], PW_GRID[0])
GATE = 0.01
COMP1_E_JUDGES = E46.COMP1_E_JUDGES + ("prereg47",)
RAM_NEED_MB = 9000
LDA_KEYS = ("m", "s", "mu", "sd", "W", "ev")
_rel, read_json, first_max, peak_mb = E43._rel, E43.read_json, E43.first_max, E45.peak_mb


def feature_names():
    st_names = ["logp_mean", "logp_sd", "logp_p10", "logp_p50", "logp_p90", "ent_mean", "ent_sd", "ent_p10",
                "ent_p50", "ent_p90", "logp_lt_-4_share", "moves_share"]
    return ["%s.%s" % (s, n) for s in SEGMENTS for n in st_names] + ["%s.missing" % s for s in SEGMENTS]


# ================================================================ per-game features (PLAN.md 1)
def _group_pct(vals, key, nkey, q):
    """numpy 'linear' percentile q of vals per key (0..nkey-1); groups without values -> 0."""
    order = np.lexsort((vals, key))
    v, k = vals[order], key[order]
    cnt = np.bincount(k, minlength=nkey)
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    out = np.zeros(nkey)
    ok = cnt > 0
    pos = (q / 100.0) * (cnt[ok] - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, cnt[ok] - 1)
    t = pos - lo
    a, b = v[start[ok] + lo], v[start[ok] + hi]
    out[ok] = a + (b - a) * t
    return out


def chunk_features(gidx, turn, logp, ent, n_games):
    """Target-side positions of one chunk -> (F [n_games, 52], n_target [n_games]). gidx = in-chunk game index."""
    gidx = np.asarray(gidx, np.int64)
    turn = np.asarray(turn, np.int64)
    logp = np.asarray(logp, np.float64)
    ent = np.asarray(ent, np.float64)
    seg = np.where(turn < SEG_EDGES[0], 0, np.where(turn < SEG_EDGES[1], 1, 2))
    ns = len(SEGMENTS)
    key = np.concatenate([gidx * ns + seg, gidx * ns + 3])          # every move once in its segment, once in "all"
    lp = np.concatenate([logp, logp])
    en = np.concatenate([ent, ent])
    nkey = n_games * ns
    cnt = np.bincount(key, minlength=nkey).astype(np.float64)
    ok = cnt > 0
    safe = np.where(ok, cnt, 1.0)
    cols = []
    for v in (lp, en):
        mean = np.bincount(key, weights=v, minlength=nkey) / safe
        var = np.bincount(key, weights=(v - mean[key]) ** 2, minlength=nkey) / safe
        cols += [mean, np.sqrt(var)] + [_group_pct(v, key, nkey, q) for q in PCTS]
    cols.append(np.bincount(key, weights=(lp < BLUNDER).astype(np.float64), minlength=nkey) / safe)
    n_target = np.bincount(gidx, minlength=n_games).astype(np.float64)
    cols.append(cnt / np.repeat(np.where(n_target > 0, n_target, 1.0), ns))
    S = np.stack(cols, 1)                                            # [nkey, 12]
    S[~ok] = 0.0
    F = np.concatenate([S.reshape(n_games, ns * N_STATS), (~ok).reshape(n_games, ns).astype(np.float64)], 1)
    return F, n_target.astype(np.int64)


def load_moves(paths, n, want_set, need=None):
    """Chunks -> dict(F [n, 52] float64, usable, have, failed, no_pos, n_target). need: games to compute (others
    only marked as present)."""
    F = np.zeros((n, N_FEAT))
    have = np.zeros(n, bool)
    failed = np.zeros(n, bool)
    no_pos = np.zeros(n, bool)
    ntar = np.zeros(n, np.int64)
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            if str(z["format"]) != F10.CHUNK_FORMAT:
                raise SystemExit("%s: chunk format %r" % (_rel(p), str(z["format"])))
            if want_set is not None and str(z["set"]) != want_set:
                raise SystemExit("%s: set %r != %r" % (_rel(p), str(z["set"]), want_set))
            if [str(s) for s in z["sides"]] != ["target", "opponent"]:
                raise SystemExit("%s: sides %r" % (_rel(p), z["sides"].tolist()))
            ref = np.asarray(z["ref"], np.int64)
            if len(ref) and (ref.min() < 0 or ref.max() >= n):
                raise SystemExit("%s: ref outside 0..%d" % (_rel(p), n - 1))
            if have[ref].any():
                raise SystemExit("%s: game refs seen in another chunk" % _rel(p))
            have[ref] = True
            failed[ref] = z["failed"]
            if need is not None and not need[ref].any():
                continue
            if not all(k in z.files for k in ("pos_game", "pos_side", "pos_turn", "pos_logp", "pos_entropy")):
                no_pos[ref] = True
                continue
            g, side = np.asarray(z["pos_game"], np.int64), np.asarray(z["pos_side"])
            sel = side == TARGET_SIDE
            if len(g) and (g.min() < 0 or g.max() >= len(ref)):
                raise SystemExit("%s: pos_game outside the chunk" % _rel(p))
            Fc, nt = chunk_features(g[sel], z["pos_turn"][sel], z["pos_logp"][sel], z["pos_entropy"][sel], len(ref))
            if not np.array_equal(nt, np.asarray(z["counts"])[:, TARGET_SIDE]):
                raise SystemExit("%s: target positions != counts[:, 0]" % _rel(p))
            F[ref] = Fc
            ntar[ref] = nt
    usable = have & ~failed & ~no_pos & (ntar >= MIN_MOVES)
    if not np.isfinite(F[usable]).all():
        raise SystemExit("non-finite per-move features")
    F[~usable] = 0.0
    return {"F": F, "usable": usable, "have": have, "failed": failed, "no_pos": no_pos, "n_target": ntar}


_MOCK_M = {}


def mock_moves(refs):
    """Per-move features of PREREG_10 mock games (the chunks' stored per-position arrays), cached for the needed
    refs."""
    n = len(read_json(P39E.MOCK_SETS)["games"])
    need = np.zeros(n, bool)
    need[refs] = True
    if _MOCK_M and not (need & ~_MOCK_M["need"]).any():
        return _MOCK_M
    if _MOCK_M:
        need |= _MOCK_M["need"]
    r = load_moves(F10.chunk_files(P39E.MOCK_DIR), n, F10.MOCK_SET, need)
    r["need"] = need
    _MOCK_M.clear()
    _MOCK_M.update(r)
    return _MOCK_M


def set_record(r, mock):
    return {"games": int(len(r["usable"])), "usable": int(r["usable"].sum()),
            "failed": int(r["failed"].sum()), "no_position_arrays": int(r["no_pos"].sum()),
            "under_%d_target_moves" % MIN_MOVES: int((r["have"] & ~r["failed"] & ~r["no_pos"]
                                                       & (r["n_target"] < MIN_MOVES)).sum()),
            "mock_reused": mock}


def load_moves_set(name, p1_dir=None, games=None):
    """P39E.load_set's games and order -> dict(games, F, usable, rec). Mock-reused games from the PREREG_10 chunks
    (no per-position arrays there -> not usable, counted)."""
    if games is None:          # L / M1: prereg45's set_source; D, E, E2, pub2, priv2: prereg39's lists and chunks
        games, p1_dir = E45.set_source(name) if name in S_SETS else (P39E.read_list(name), P39E.P1)
    p1_dir = P39E.P1 if p1_dir is None else p1_dir
    n = len(games)
    d = os.path.join(p1_dir, name)
    paths = F10.chunk_files(d) if os.path.isdir(d) else []
    r = load_moves(paths, n, "p39" + name)
    reuse = np.array([g.get("mock_ref", -1) for g in games], np.int64)
    if r["have"][reuse >= 0].any():
        raise SystemExit("%s: a reusable game was also extracted" % name)
    mock = {"games": int((reuse >= 0).sum()), "usable": 0, "no_position_arrays": 0}
    if (reuse >= 0).any():
        idx = np.flatnonzero(reuse >= 0)
        m = mock_moves(reuse[idx])
        if not m["have"][reuse[idx]].all():
            raise SystemExit("%s: a PREREG_10 mock chunk is missing" % name)
        for k in ("F", "usable", "have", "failed", "no_pos", "n_target"):
            r[k][idx] = m[k][reuse[idx]]
        mock["usable"] = int(r["usable"][idx].sum())
        mock["no_position_arrays"] = int(r["no_pos"][idx].sum())
    if not r["have"].all():
        raise SystemExit("%s: %d games have no P1 chunk" % (name, int((~r["have"]).sum())))
    return {"games": games, "F": r["F"], "usable": r["usable"], "rec": set_record(r, mock)}


# ================================================================ LDA (PLAN.md 2)
def fit_lda():
    """F10.fit_p1_lda on the usable target-side games of S = L + M1 (labels = players). -> (model, record)."""
    Fs, us, labs, rec = [], [], [], {}
    for n in S_SETS:
        s = load_moves_set(n)
        Fs.append(s["F"])
        us.append(s["usable"])
        labs += [g["player_id"] for g in s["games"]]
        rec[n] = s["rec"]
    T = set(E43.splits()[2])
    if not set(labs) <= T:
        raise SystemExit("an S player is not a T (training) player")
    model = F10.fit_p1_lda(np.concatenate(Fs), np.asarray(labs), np.concatenate(us))
    if model["W"].shape[1] < max(DIM_GRID):
        raise SystemExit("the LDA has fewer than %d directions" % max(DIM_GRID))
    return model, rec


def model_sha(model):
    return E45.arrays_sha(model, LDA_KEYS)


def save_lda(model, path=MODEL_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, n_games=np.array(model["n_games"]), n_players=np.array(model["n_players"]),
             **{k: model[k] for k in LDA_KEYS})
    os.replace(tmp, path)
    return {"file": _rel(path), "file_sha256": L.sha256_file(path), "arrays_sha256": model_sha(model),
            "n_games": model["n_games"], "n_players": model["n_players"], "directions": int(model["W"].shape[1]),
            "features": feature_names()}


def load_lda(rec):
    """The LDA saved by seam, checked against its record (seam.json / tune.json)."""
    p = E43._abs(rec["file"])
    with np.load(p, allow_pickle=False) as z:
        model = {k: z[k] for k in LDA_KEYS}
        model.update(n_games=int(z["n_games"]), n_players=int(z["n_players"]))
    if model_sha(model) != rec["arrays_sha256"]:
        raise SystemExit("%s differs from its record" % rec["file"])
    return model


def committed_lda():
    if not L.git_committed(SEAM_JSON):
        raise SystemExit("seam.json is not committed (or differs from HEAD): run seam and commit it first")
    s = read_json(SEAM_JSON)
    if not all(s.get(w, {}).get("per_pool_identical") for w in ("D", "E")):
        raise SystemExit("seam.json does not hold a passed D and E seam")
    return load_lda(s["lda"]), s["lda"]


def project(F, model, dim):
    return st.lda_project((np.asarray(F, np.float64) - model["m"]) / model["s"], model, dim)


# ================================================================ the term T_M
def move_ctx(which, model):
    """Per-move projections of the D or E (+E2) store rows in p1_ctx's order."""
    names = ("D",) if which == "D" else ("E", "E2")
    sets = [load_moves_set(n) for n in names]
    F = np.concatenate([s["F"] for s in sets])
    usable = np.concatenate([s["usable"] for s in sets])
    games = [g for s in sets for g in s["games"]]
    pos = {g["row"]: i for i, g in enumerate(games)}
    if len(pos) != len(games):
        raise SystemExit("a store row is in two lists")
    return {"Z": {d: project(F, model, d) for d in DIM_GRID}, "usable": usable, "pos": pos,
            "sets": {n: s["rec"] for n, s in zip(names, sets)}}


def move_z(Z, usable, cand_groups, q_groups, pw):
    """Groups = indices into Z. -> (z [nq x nc], has_q). P39E._term = v2-6's P1 term structure."""
    cg = [[g for g in grp if usable[g]] for grp in cand_groups]
    qg = [[g for g in grp if usable[g]] for grp in q_groups]
    T, has = P39E._term(Z, cg, qg, pw)
    z = st.zscore_rows(T, has)
    if not np.isfinite(z).all():
        raise SystemExit("non-finite T_M")
    return z, np.asarray(has, bool)


def pools_move_z(pools, mctx, dim, pw):
    out, miss = [], []
    for p in pools:
        cg, qg = E46.pool_groups(p, mctx["pos"])
        z, has = move_z(mctx["Z"][dim], mctx["usable"], cg, qg, pw)
        out.append(z)
        miss.append(int((~has).sum()))
    return out, miss


def m47_scores(FPs, zMs, wM, k, pools):
    """M47 per pool: CSLS_k(FP + wM z_M), FP = v2-6 before its CSLS."""
    return [float(st.score(st.csls(FP + wM * zM, k), p["owner"])) for FP, zM, p in zip(FPs, zMs, pools)]


# ================================================================ seam
def cmd_seam(a):
    t0 = time.time()
    base = E45.v26()
    model, srec = fit_lda()
    E46._MOCK_OPP.clear()
    rec = {"format": "c10-prereg47 seam v1", "base": base["record"], "git_head": L.git_head(),
           "rule": "wM = 0, k = %d: M47 per pool == v2-6's recorded per-pool scores, bit for bit" % SEAM_K,
           "lda": save_lda(model, a.model), "S": srec}
    print(json.dumps({"S": srec, "lda": {k: rec["lda"][k] for k in ("n_games", "n_players", "arrays_sha256")}}),
          flush=True)
    for which in (("D",) if a.skip_e else ("D", "E")):
        pools = EV.load_pools(EV.TERMS_DIR[which], EV.D_SEEDS if which == "D" else EV.E_SEEDS)
        FPs, ctx, vec, _per = E46.base_parts(base, which, pools)
        del ctx
        mctx = move_ctx(which, model)
        _MOCK_M.clear()
        zM, miss = pools_move_z(pools, mctx, *SEAM_TERM)
        per = m47_scores(FPs, zM, 0.0, SEAM_K, pools)
        want = base["d_per_pool"] if which == "D" else base["e_per_pool"]
        mean = D_V26 if which == "D" else E_V26
        rec[which] = {"per_pool": per, "mean": float(np.mean(per)), "want_mean": mean, "want_per_pool": want,
                      "per_pool_identical": per == want, "mean_identical": float(np.mean(per)) == mean,
                      "seam_term": list(SEAM_TERM), "vectors": vec, "move_sets": mctx["sets"],
                      "questions_without_usable_game_per_pool": miss}
        print(json.dumps({which: {k: rec[which][k] for k in ("mean", "per_pool_identical", "mean_identical",
                                                              "move_sets")}}), flush=True)
        if not rec[which]["per_pool_identical"] or not rec[which]["mean_identical"]:
            L.write_json(a.out, rec)
            raise SystemExit("seam %s failed" % which)
        del FPs, mctx, zM, pools
    rec["peak_working_set_mb"] = peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("peak_working_set_mb", "seconds")}))
    return 0


# ================================================================ select (D)
def no_e_results_yet(here=HERE, out_apply=OUT_APPLY):
    found = glob.glob(os.path.join(here, "eval*.json"))
    if os.path.isdir(out_apply):
        found.append(out_apply)
    if found:
        raise SystemExit("prereg47 E / official results exist before select: %s" % [_rel(p) for p in found])


def grid_rows(FPs, zM_by, pools, terms):
    """zM_by: {(dim, pw): [z per pool]}; terms = feasible (dim, pw) in grid order. Rows in the fixed grid order."""
    rows = []
    for dim in DIM_GRID:
        for pw in PW_GRID:
            if (dim, pw) not in terms:
                continue
            for wM in WM_GRID:
                for k in K_GRID:
                    per = m47_scores(FPs, zM_by[(dim, pw)], wM, k, pools)
                    rows.append({"dim": dim, "pw": pw, "wM": wM, "k": k, "mean": float(np.mean(per)),
                                 "per_pool": per})
    return rows


def cmd_select(a):
    t0 = time.time()
    no_e_results_yet()
    model, lrec = committed_lda()
    base = E45.v26()
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    FPs, ctx, vec, base_per = E46.base_parts(base, "D", pools)
    del ctx
    mctx = move_ctx("D", model)
    _MOCK_M.clear()
    zM_by, feas, miss = {}, {}, None
    for dim in DIM_GRID:
        for pw in PW_GRID:
            try:
                zM_by[(dim, pw)], m = pools_move_z(pools, mctx, dim, pw)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as ex:   # b4 not possible -> none only
                feas["%d/%s" % (dim, pw)] = {"feasible": False, "reason": "%s: %s" % (type(ex).__name__, ex)}
                continue
            feas["%d/%s" % (dim, pw)] = {"feasible": True}
            miss = m
    terms = [t for t in zM_by]
    if not terms:
        raise SystemExit("no (dim, pw) variant is feasible: %s" % feas)
    rows = grid_rows(FPs, zM_by, pools, terms)
    top = first_max(rows, key=lambda r: r["mean"])
    alone = [float(st.score(st.csls(z, top["k"]), p["owner"])) for z, p in zip(zM_by[(top["dim"], top["pw"])], pools)]
    gate = top["mean"] - D_V26
    rec = {"format": "c10-prereg47 tune v1", "d_seeds": EV.D_SEEDS, "grid_order": ["dim", "pw", "wM", "k"],
           "grid": {"dim": list(DIM_GRID), "pw": list(PW_GRID), "wM": list(WM_GRID), "k": list(K_GRID)},
           "feasibility": feas, "rows": [{k: r[k] for k in ("dim", "pw", "wM", "k", "mean")} for r in rows],
           "best_per_dim_pw": {"%d/%s" % t: first_max([r for r in rows if (r["dim"], r["pw"]) == t],
                                                      key=lambda r: r["mean"])["mean"] for t in terms},
           "c47": {"dim": top["dim"], "pw": top["pw"], "wM": top["wM"], "csls_k": top["k"], "mean": top["mean"],
                   "per_pool": top["per_pool"], "minus_v26_D": E43.diff_rec(top["per_pool"], base_per)},
           "record_only_T_M_alone_D": {"dim": top["dim"], "pw": top["pw"], "csls_k": top["k"],
                                       "mean": float(np.mean(alone)), "per_pool": alone,
                                       "rule": "CSLS_k(z(T_M)) with c47's dim / pw / k; not used for any choice"},
           "lda": lrec, "seam_sha256": st.sha256_lf(SEAM_JSON),
           "base": base["record"], "base_D": {"per_pool": base_per, "mean": float(np.mean(base_per)),
                                              "identical_to_recorded": True},
           "cnn_vectors_D": vec, "move_sets": mctx["sets"], "questions_without_usable_game_per_pool": miss,
           "gate": {"rule": "D(M47) - D(v2-6) >= %.2f, D(v2-6) = %r" % (GATE, D_V26), "value": gate,
                    "pass": bool(gate >= GATE)},
           "peak_working_set_mb": peak_mb(), "seconds": round(time.time() - t0, 1), "git_head": L.git_head()}
    L.write_json(a.out, rec)
    print(json.dumps({"c47": {k: rec["c47"][k] for k in ("dim", "pw", "wM", "csls_k", "mean")},
                      "T_M_alone": rec["record_only_T_M_alone_D"]["mean"], "best_per_dim_pw": rec["best_per_dim_pw"],
                      "gate": rec["gate"], "feasibility": feas,
                      "peak_working_set_mb": rec["peak_working_set_mb"]}, indent=1))
    return 0


# ================================================================ judge (E, once)
def e_judgements_so_far(repo=REPO):
    return [d for d in COMP1_E_JUDGES if glob.glob(os.path.join(repo, "results", d, "explore", "eval*.json"))]


def judge_ready(tune, e_use, prior):
    if not tune["gate"]["pass"]:
        raise SystemExit("the D gate did not pass: the E pools are not used (PLAN.md 3)")
    if "prereg47" in prior:
        raise SystemExit("prereg47 already has an E judgement")
    if e_use != len(prior) + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (e_use, len(prior), prior))


def cmd_judge(a):
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("tune.json is not committed (or differs from HEAD); commit it before the judgement")
    tune = read_json(a.tune)
    prior = e_judgements_so_far()
    judge_ready(tune, a.e_use, prior)
    base = E45.v26()
    E46.check_base_unchanged(base, tune)
    model = load_lda(tune["lda"])
    c = tune["c47"]
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    Es = set(E43.splits()[0])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in E43.store().pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    FPs, ctx, vec, bper = E46.base_parts(base, "E", pools)
    del ctx
    mctx = move_ctx("E", model)
    _MOCK_M.clear()
    zM, miss = pools_move_z(pools, mctx, c["dim"], c["pw"])
    per = m47_scores(FPs, zM, c["wM"], c["csls_k"], pools)
    byk = [EV.score_by_k(st.csls(FP + c["wM"] * z, c["csls_k"]), p) for FP, z, p in zip(FPs, zM, pools)]
    rec = {"format": "c10-prereg47 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": prior, "c47": c, "cnn_vectors_E": vec, "base": tune["base"],
           "seam_base_E_identical": True, "tune_sha256": st.sha256_lf(a.tune), "git_head": L.git_head(),
           "lda": tune["lda"], "move_sets": mctx["sets"], "questions_without_usable_game_per_pool": miss,
           "means": {"c47": float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {"c47": per, "base": bper}, "by_k": {"c47": EV.mean_by_k(byk)}}
    j = E43.judge_rule(per, bper)
    rec["c47_minus_base"] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"] = j["rule"].replace("c*", "c47")
    rec["adopt"] = j["adopt"]
    rec["peak_working_set_mb"] = peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "c47_minus_base", "by_k", "adopt")}, indent=1))
    return 0


# ================================================================ official pools
def cmd_apply(a):
    t0 = time.time()
    tune, ev = read_json(a.tune), read_json(a.eval)
    if not ev["adopt"] or ev["tune_sha256"] != st.sha256_lf(a.tune):
        raise SystemExit("eval.json does not adopt c47 (or belongs to another tune.json): keep v2-6")
    base = E45.v26()
    E46.check_base_unchanged(base, tune)
    model = load_lda(tune["lda"])
    rec, items, cand_ids, qids = E43.official_rec(a.src, a.pool)
    name = {"public": "pub2", "private": "priv2"}[a.pool]
    Pset, Lset = P39E.load_set(name), P39E.load_set("L")
    if len(Pset["games"]) != len(items) or any(g["color"] != c for g, (_s, c) in zip(Pset["games"], items)):
        raise SystemExit("the P1 list of the pool does not match the pool's games")
    if Pset["meta"] != Lset["meta"]:
        raise SystemExit("L and pool chunks have different metadata")
    bad = int((~Pset["usable"]).sum())
    if bad > P39E.FAIL_MAX * len(items):
        raise SystemExit("%d of %d pool games have no usable P1 (> 0.5%%): stop" % (bad, len(items)))
    models = P39E.fit_models(Lset)
    del Lset
    if P39E.models_sha(models) != base["lda_sha256"]:
        raise SystemExit("the L LDA refit differs from prereg43 tune.json")
    colour = np.array([c for _s, c in items])
    zb, _hb = P39E.p1_terms(Pset["X"], Pset["usable"], colour, rec["cand_rows"], rec["q_rows"], models,
                            E43.P1_PW)[E43.P1_VARIANT]
    del models
    path, _choice = E43.cstar_file(base["tune"], a.pool)
    e = E43.pool_emb(path, items, base["tune"]["c_star"]["ckpt_sha256"])
    F = BK.fuse_cfg(rec, BK.e_term(rec, e, base["cfg"]), base["cfg"])
    FP = F + base["wP"] * zb
    # seam: v2-6 rebuilt on this pool == prereg43's uploaded top5
    data_b = P39E.top5_bytes(qids, st.top5(st.csls(FP, base["k"]), cand_ids))
    with open(os.path.join(V26_APPLY, a.pool, "top5.csv"), "rb") as f:
        if f.read() != data_b:
            raise SystemExit("seam: v2-6 rebuilt on %s differs from results/prereg43/apply/%s/top5.csv"
                             % (a.pool, a.pool))
    M = load_moves_set(name, p1_dir=P39E.P1, games=Pset["games"])
    c = tune["c47"]
    zM, has = move_z(project(M["F"], model, c["dim"]), M["usable"], rec["cand_rows"], rec["q_rows"], c["pw"])
    data = P39E.top5_bytes(qids, st.top5(st.csls(FP + c["wM"] * zM, c["csls_k"]), cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg47 apply v1", "pool": a.pool, "c47": c, "vectors": _rel(path),
           "vectors_sha256": L.sha256_file(path), "cfg": base["cfg"], "v26_top5_equals_prereg43_apply": True,
           "questions": len(qids), "candidates": len(cand_ids), "games": len(items),
           "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "games_without_P1": bad, "move_set": M["rec"],
           "questions_without_usable_move_game": int((~has).sum()),
           "tune_sha256": st.sha256_lf(a.tune), "eval_sha256": st.sha256_lf(a.eval),
           "top5_sha256": L.sha256_file(out), "differs_from_v26_rows": int(sum(
               x != y for x, y in zip(data.split(b"\n"), data_b.split(b"\n")))),
           "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    return E43.cmd_assemble(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("seam")
    m.add_argument("--skip-e", action="store_true")
    m.add_argument("--out", default=SEAM_JSON)
    m.add_argument("--model", default=MODEL_FILE)
    s = sub.add_parser("select")
    s.add_argument("--out", default=os.path.join(HERE, "tune.json"))
    j = sub.add_parser("judge")
    j.add_argument("--e-use", type=int, required=True)
    j.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    j.add_argument("--out", default=os.path.join(HERE, "eval.json"))
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--pool", required=True, choices=["public", "private"])
    p.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    p.add_argument("--eval", default=os.path.join(HERE, "eval.json"))
    p.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", default=V1_SUBMISSION)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    if a.cmd != "assemble":
        BK.polite()                 # BELOW_NORMAL, 4 logical processors, 1 BLAS thread (prereg38's manners)
        BK.wait_for_ram(need_mb=RAM_NEED_MB)
    return {"seam": cmd_seam, "select": cmd_select, "judge": cmd_judge, "apply": cmd_apply,
            "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
