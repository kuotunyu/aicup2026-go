"""prereg46 explore: the opponent-fingerprint term T_O (PLAN.md). Seam, D selection, the one E judgement, official
pools. CPU only; no new P1 extraction.

    PY results/prereg46/explore/p46_eval.py seam                # -> seam.json (v2-6 rebuilt with wO = 0, D and E)
    PY results/prereg46/explore/p46_eval.py select              # -> tune.json (commit before judge)
    PY results/prereg46/explore/p46_eval.py judge --e-use 6     # once -> eval.json
    PY results/prereg46/explore/p46_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg46/explore/p46_eval.py assemble --src <official tests dir>

Base = v2-6 (results/prereg43/explore/tune.json + eval.json), rebuilt by prereg45's code path (p45_eval.v26,
base_fused, seam_scores): F_base = fuse_cfg(...) of c*, the P1 term z_P joint / b4 of the L-LDA64, wP 8.
M46 = CSLS_k( (F_base + wP z_P) + wO z(T_O) ).
T_O (PLAN.md 1): per game the OPPONENT side of the stored P1 chunks (means[:, 1]; usable = chunk present, not
failed, counts[:, 1] > 0), projected with v2-6's P1 projection (the joint L-LDA64 of prereg39, fit on the target
side of L as always), then
    pw "b4":   prereg38's B4 transform of the pool's opponent vectors (per-game L2, whitening lam 0.02 + LDA 32 with
               the within-candidate covariance of the CANDIDATES' opponent vectors, classes = candidate ids), then L2;
    pw "none": per-game L2 only.
Similarity of question q and candidate c from the cosine of every (q game, c game) pair:
    "max":  the maximum over all pairs;   "mmax": per q game the maximum over c's games, averaged over q's games.
z = p06_style.zscore_rows per question; a question without a usable opponent game -> its row is 0; a candidate
without one -> its column = the row minimum of the other candidates.
Grid (fixed order): pw (b4, none) x agg (max, mmax) x wO (0.25, 0.5, 1, 2, 4, 8) x k (10, 20, 40); first maximum
of the D mean. Gate: D(M46) - 0.7467766213119217 >= 0.01. judge (E pools, use 6): mean >= +0.01, pool bootstrap
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
P43 = os.path.join(REPO, "results", "prereg43", "explore")
P45 = os.path.join(REPO, "results", "prereg45", "explore")
for p in (P43, P45, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import p43_eval as E43  # noqa: E402   (v2-6's code path)
import p45_eval as E45  # noqa: E402   (v26(), base_fused(), seam_scores(): v2-6 rebuilt and checked bit for bit)

P39E = E43.P39E
BK, EV, st, L = E43.BK, E43.EV, E43.st, E43.L
F10 = P39E.F10

D_V26, E_V26 = E45.D_V26, E45.E_V26
V26_APPLY = E45.V26_APPLY
OUT_APPLY = os.path.join(REPO, "results", "prereg46", "apply")
V1_SUBMISSION = E45.V1_SUBMISSION
OPP_SIDE = 1                                  # chunk "sides" = ['target', 'opponent']
PW_GRID = ("b4", "none")
AGG_GRID = ("max", "mmax")
WO_GRID = (0.25, 0.5, 1, 2, 4, 8)
K_GRID = (10, 20, 40)
SEAM_K = 20
GATE = 0.01
COMP1_E_JUDGES = E43.COMP1_E_JUDGES + ("prereg43", "prereg45", "prereg46")
RAM_NEED_MB = 9000
_abs, _rel, read_json, first_max, peak_mb = E43._abs, E43._rel, E43.read_json, E43.first_max, E45.peak_mb


# ================================================================ opponent-side P1 (PLAN.md 1)
def load_side(paths, n, want_set, side):
    """= p10_fuse.load_side0 for any side: (X float32 [n, 770], usable, failed, have, counts of that side)."""
    X = np.zeros((n, F10.P1_DIM), np.float32)
    failed = np.zeros(n, bool)
    have = np.zeros(n, bool)
    cnt = np.zeros(n, np.int64)
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
            X[ref] = z["means"][:, side, :]
            failed[ref] = z["failed"]
            cnt[ref] = z["counts"][:, side]
            have[ref] = True
    usable = have & ~failed & (cnt > 0)
    return X, usable, failed, have, cnt


_MOCK_OPP = {}


def mock_side(side):
    if side not in _MOCK_OPP:
        n = len(read_json(P39E.MOCK_SETS)["games"])
        X, usable, _f, have, _c = load_side(F10.chunk_files(P39E.MOCK_DIR), n, F10.MOCK_SET, side)
        _MOCK_OPP[side] = {"X": X, "usable": usable, "have": have}
    return _MOCK_OPP[side]


def load_set_side(name, side=OPP_SIDE, p1_dir=None, games=None):
    """P39E.load_set's games and order, the given side. -> dict(games, X, usable, failed (fresh chunks), not_usable
    (all games, incl. the PREREG_10 mock ones))."""
    p1_dir = P39E.P1 if p1_dir is None else p1_dir
    games = P39E.read_list(name) if games is None else games
    n = len(games)
    d = os.path.join(p1_dir, name)
    paths = F10.chunk_files(d) if os.path.isdir(d) else []
    X, usable, failed, have, cnt = load_side(paths, n, "p39" + name, side)
    reuse = np.array([g.get("mock_ref", -1) for g in games], np.int64)
    if have[reuse >= 0].any():
        raise SystemExit("%s: a reusable game was also extracted" % name)
    if (reuse >= 0).any():
        m = mock_side(side)
        idx = np.flatnonzero(reuse >= 0)
        if not m["have"][reuse[idx]].all():
            raise SystemExit("%s: a PREREG_10 mock chunk is missing" % name)
        X[idx] = m["X"][reuse[idx]]
        usable[idx] = m["usable"][reuse[idx]]
        have[idx] = True
    if not have.all():
        raise SystemExit("%s: %d games have no P1 chunk" % (name, int((~have).sum())))
    return {"games": games, "X": X, "usable": usable, "failed": int(failed.sum()),
            "not_usable": int((~usable).sum())}


def opp_ctx(which, models):
    """Opponent vectors of the D or E (+E2) store rows in p1_ctx's order, projected with v2-6's joint L-LDA64."""
    names = ("D",) if which == "D" else ("E", "E2")
    sets = [load_set_side(n) for n in names]
    X = np.concatenate([s["X"] for s in sets])
    usable = np.concatenate([s["usable"] for s in sets])
    games = [g for s in sets for g in s["games"]]
    pos = {g["row"]: i for i, g in enumerate(games)}
    Z = F10.project_p1(X, models[E43.P1_VARIANT])
    return {"Z": Z, "usable": usable, "pos": pos,
            "sets": {n: {"games": len(s["games"]), "usable": int(s["usable"].sum()), "failed": s["failed"],
                         "not_usable": s["not_usable"]} for n, s in zip(names, sets)}}


# ================================================================ the term T_O
def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.where(n > 0, X / np.where(n > 0, n, 1.0), 0.0)


def opp_vectors(Z, cand_groups, q_groups, pw):
    """Groups = usable game indices into Z. -> (Vc, gc, Vq, gq): L2 rows after pw, group index per row."""
    gc = np.array([i for i, g in enumerate(cand_groups) for _ in g], np.int64)
    gq = np.array([i for i, g in enumerate(q_groups) for _ in g], np.int64)
    ic = np.array([j for g in cand_groups for j in g], np.int64)
    iq = np.array([j for g in q_groups for j in g], np.int64)
    Xc, Xq = np.asarray(Z[ic], np.float64), np.asarray(Z[iq], np.float64)
    if pw == "b4":
        G = {"nc": len(cand_groups), "nq": len(q_groups), "Xc": Xc, "Xq": Xq, "gc": gc, "gq": gq,
             "wc": np.zeros(len(ic), bool), "wq": np.zeros(len(iq), bool), "ncg": np.ones(len(ic)),
             "nqg": np.ones(len(iq))}
        Xc, Xq = BK.transform(G, P39E.B4_P1)
    elif pw != "none":
        raise ValueError(pw)
    return _l2(Xc), gc, _l2(Xq), gq


def agg_terms(Vc, gc, Vq, gq, nc, nq, aggs=AGG_GRID):
    """-> {agg: (T [nq x nc], has_q)}. gc / gq are non-decreasing (groups are concatenated in order)."""
    has_c = np.bincount(gc, minlength=nc) > 0 if len(gc) else np.zeros(nc, bool)
    has_q = np.bincount(gq, minlength=nq) > 0 if len(gq) else np.zeros(nq, bool)
    out = {}
    if has_c.sum() < 2 or not len(gq):
        return {a: (np.zeros((nq, nc)), np.zeros(nq, bool)) for a in aggs}
    if np.any(np.diff(gc) < 0) or np.any(np.diff(gq) < 0):
        raise ValueError("groups are not contiguous")
    S = Vq @ Vc.T                                            # [q games x c games] cosines
    cs = np.flatnonzero(np.r_[True, gc[1:] != gc[:-1]])
    M = np.maximum.reduceat(S, cs, axis=1)                   # [q games x candidates with games]
    del S
    qs = np.flatnonzero(np.r_[True, gq[1:] != gq[:-1]])
    nqg = np.bincount(gq, minlength=nq)[has_q].astype(np.float64)
    for a in aggs:
        if a == "max":
            R = np.maximum.reduceat(M, qs, axis=0)
        elif a == "mmax":
            R = np.add.reduceat(M, qs, axis=0) / nqg[:, None]
        else:
            raise ValueError(a)
        T = np.zeros((nq, nc))
        T[np.ix_(np.flatnonzero(has_q), np.flatnonzero(has_c))] = R
        if not has_c.all():
            T[:, ~has_c] = T[:, has_c].min(1, keepdims=True)
        out[a] = (T, has_q.copy())
    return out


def opp_z(Z, usable, cand_groups, q_groups, pw, aggs=AGG_GRID):
    """Groups = indices into Z (all games of the pool). -> {agg: (z [nq x nc], has_q)}."""
    cg = [[g for g in grp if usable[g]] for grp in cand_groups]
    qg = [[g for g in grp if usable[g]] for grp in q_groups]
    Vc, gc, Vq, gq = opp_vectors(Z, cg, qg, pw)
    T = agg_terms(Vc, gc, Vq, gq, len(cg), len(qg), aggs)
    return {a: (st.zscore_rows(t, h), h) for a, (t, h) in T.items()}


def pool_groups(pool, pos):
    return ([[pos[int(r)] for r in g] for g in pool["cand_rows"]], [[pos[int(r)] for r in g] for g in pool["q_rows"]])


def pools_opp_z(pools, octx, pw, aggs=AGG_GRID):
    """-> {agg: [z per pool]}, and per-pool counts of questions without an opponent vector."""
    out, miss = {a: [] for a in aggs}, []
    for p in pools:
        cg, qg = pool_groups(p, octx["pos"])
        r = opp_z(octx["Z"], octx["usable"], cg, qg, pw, aggs)
        for a in aggs:
            if not np.isfinite(r[a][0]).all():
                raise SystemExit("pool %s: non-finite T_O" % p.get("seed"))
            out[a].append(r[a][0])
        miss.append(int((~r[aggs[0]][1]).sum()))
    return out, miss


# ================================================================ the systems
def m46_scores(FPs, zOs, wO, k, pools):
    """M46 per pool: CSLS_k((F_base + wP z_P) + wO z_O). FPs = F_base + wP z_P (v2-6 before its CSLS)."""
    return [float(st.score(st.csls(FP + wO * zO, k), p["owner"])) for FP, zO, p in zip(FPs, zOs, pools)]


def base_parts(base, which, pools, check_side0=False):
    """v2-6 on the D / E pools -> (F_base + wP z_P per pool, {"models", "lda_sha256", "side0_identical"}, CNN vector
    record, v2-6 per-pool scores), v2-6 checked bit for bit. Only the LDA models of the P1 ctx are kept (RAM)."""
    Fs, vec = E45.base_fused(base, which, pools)
    ctx = E43.p1_ctx(which)
    if ctx["lda_sha256"] != base["lda_sha256"]:
        raise SystemExit("the P1 LDA refit on L differs from prereg43 tune.json")
    zP = [E43.p1_z(p, ctx) for p in pools]
    per = E45.seam_scores(base, which, Fs, zP, pools)          # v2-6 itself, bit for bit
    FPs = [F + base["wP"] * z for F, z in zip(Fs, zP)]
    del Fs, zP
    s0 = side0_check(ctx, which) if check_side0 else None
    keep = {"models": ctx["models"], "lda_sha256": ctx["lda_sha256"], "side0_identical": s0}
    del ctx
    P39E._MOCK.clear()
    _MOCK_OPP.pop(0, None)
    return FPs, keep, vec, per


def side0_check(ctx, which):
    """The side loader of this file with side 0 == P39E.load_set's X / usable (the loader is the same as v2-6's)."""
    names = ("D",) if which == "D" else ("E", "E2")
    sets = [load_set_side(n, side=0) for n in names]
    X = np.concatenate([s["X"] for s in sets])
    u = np.concatenate([s["usable"] for s in sets])
    ok = bool(np.array_equal(X, ctx["X"]) and np.array_equal(u, ctx["usable"]))
    if not ok:
        raise SystemExit("side-0 loader of p46 differs from P39E.load_set on %s" % which)
    return ok


# ================================================================ seam
def cmd_seam(a):
    t0 = time.time()
    base = E45.v26()
    rec = {"format": "c10-prereg46 seam v1", "base": base["record"], "git_head": L.git_head(),
           "rule": "wO = 0, k = %d: M46 per pool == v2-6's recorded per-pool scores, bit for bit" % SEAM_K}
    for which in (("D",) if a.skip_e else ("D", "E")):
        pools = EV.load_pools(EV.TERMS_DIR[which], EV.D_SEEDS if which == "D" else EV.E_SEEDS)
        FPs, ctx, vec, _per = base_parts(base, which, pools, check_side0=True)
        s0 = ctx["side0_identical"]
        octx = opp_ctx(which, ctx["models"])
        models_sha = ctx["lda_sha256"]
        del ctx
        zO, miss = pools_opp_z(pools, octx, PW_GRID[0], (AGG_GRID[0],))
        per = m46_scores(FPs, zO[AGG_GRID[0]], 0.0, SEAM_K, pools)
        want = base["d_per_pool"] if which == "D" else base["e_per_pool"]
        mean = D_V26 if which == "D" else E_V26
        same = [x == y for x, y in zip(per, want)]
        rec[which] = {"per_pool": per, "mean": float(np.mean(per)), "want_mean": mean, "want_per_pool": want,
                      "per_pool_identical": all(same) and len(per) == len(want),
                      "mean_identical": float(np.mean(per)) == mean, "seam_term": [PW_GRID[0], AGG_GRID[0]],
                      "side0_loader_identical_to_p39": s0, "lda_sha256": models_sha, "vectors": vec,
                      "opponent_sets": octx["sets"], "questions_without_opponent_vector_per_pool": miss}
        print(json.dumps({which: {k: rec[which][k] for k in ("mean", "per_pool_identical", "mean_identical")}}),
              flush=True)
        if not rec[which]["per_pool_identical"] or not rec[which]["mean_identical"]:
            L.write_json(a.out, rec)
            raise SystemExit("seam %s failed" % which)
        del FPs, octx, zO, pools
        _MOCK_OPP.clear()
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
        raise SystemExit("prereg46 E / official results exist before select: %s" % [_rel(p) for p in found])


def grid_rows(FPs, zO_by, pools, pws):
    """zO_by: {(pw, agg): [z per pool]}. Rows in the fixed grid order."""
    rows = []
    for pw in pws:
        for agg in AGG_GRID:
            for wO in WO_GRID:
                for k in K_GRID:
                    per = m46_scores(FPs, zO_by[(pw, agg)], wO, k, pools)
                    rows.append({"pw": pw, "agg": agg, "wO": wO, "k": k, "mean": float(np.mean(per)),
                                 "per_pool": per})
    return rows


def cmd_select(a):
    t0 = time.time()
    no_e_results_yet()
    base = E45.v26()
    pools = EV.load_pools(EV.TERMS_DIR["D"], EV.D_SEEDS)
    FPs, ctx, vec, base_per = base_parts(base, "D", pools)
    octx = opp_ctx("D", ctx["models"])
    del ctx
    zO_by, miss, feas = {}, None, {}
    for pw in PW_GRID:
        try:
            r, m = pools_opp_z(pools, octx, pw)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as ex:   # b4 not possible -> only "none"
            feas[pw] = {"feasible": False, "reason": "%s: %s" % (type(ex).__name__, ex)}
            continue
        feas[pw] = {"feasible": True}
        for agg in AGG_GRID:
            zO_by[(pw, agg)] = r[agg]
        miss = m
    pws = [pw for pw in PW_GRID if feas[pw]["feasible"]]
    if not pws:
        raise SystemExit("no pw variant is feasible: %s" % feas)
    rows = grid_rows(FPs, zO_by, pools, pws)
    top = first_max(rows, key=lambda r: r["mean"])
    alone = [float(st.score(st.csls(z, top["k"]), p["owner"])) for z, p in zip(zO_by[(top["pw"], top["agg"])], pools)]
    gate = top["mean"] - D_V26
    rec = {"format": "c10-prereg46 tune v1", "d_seeds": EV.D_SEEDS, "grid_order": ["pw", "agg", "wO", "k"],
           "grid": {"pw": list(PW_GRID), "agg": list(AGG_GRID), "wO": list(WO_GRID), "k": list(K_GRID)},
           "pw_feasibility": feas, "rows": [{k: r[k] for k in ("pw", "agg", "wO", "k", "mean")} for r in rows],
           "best_per_pw_agg": {"%s/%s" % (pw, agg): first_max([r for r in rows if r["pw"] == pw and r["agg"] == agg],
                                                              key=lambda r: r["mean"])["mean"]
                               for pw in pws for agg in AGG_GRID},
           "c46": {"pw": top["pw"], "agg": top["agg"], "wO": top["wO"], "csls_k": top["k"], "mean": top["mean"],
                   "per_pool": top["per_pool"], "minus_v26_D": E43.diff_rec(top["per_pool"], base_per)},
           "record_only_T_O_alone_D": {"pw": top["pw"], "agg": top["agg"], "csls_k": top["k"],
                                       "mean": float(np.mean(alone)), "per_pool": alone,
                                       "rule": "CSLS_k(z(T_O)) with c46's pw / agg / k; not used for any choice"},
           "base": base["record"], "base_D": {"per_pool": base_per, "mean": float(np.mean(base_per)),
                                              "identical_to_recorded": True},
           "cnn_vectors_D": vec, "opponent_sets": octx["sets"],
           "questions_without_opponent_vector_per_pool": miss,
           "gate": {"rule": "D(M46) - D(v2-6) >= %.2f, D(v2-6) = %r" % (GATE, D_V26), "value": gate,
                    "pass": bool(gate >= GATE)},
           "peak_working_set_mb": peak_mb(), "seconds": round(time.time() - t0, 1), "git_head": L.git_head()}
    L.write_json(a.out, rec)
    print(json.dumps({"c46": {k: rec["c46"][k] for k in ("pw", "agg", "wO", "csls_k", "mean")},
                      "T_O_alone": rec["record_only_T_O_alone_D"]["mean"], "best_per_pw_agg": rec["best_per_pw_agg"],
                      "gate": rec["gate"], "pw_feasibility": feas,
                      "peak_working_set_mb": rec["peak_working_set_mb"]}, indent=1))
    return 0


# ================================================================ judge (E, once)
def e_judgements_so_far(repo=E43.REPO):
    return [d for d in COMP1_E_JUDGES if glob.glob(os.path.join(repo, "results", d, "explore", "eval*.json"))]


def judge_ready(tune, e_use, prior):
    if not tune["gate"]["pass"]:
        raise SystemExit("the D gate did not pass: the E pools are not used (PLAN.md 2)")
    if "prereg46" in prior:
        raise SystemExit("prereg46 already has an E judgement")
    if e_use != len(prior) + 1:
        raise SystemExit("--e-use %d but the E pools were judged %d times before (%s)" % (e_use, len(prior), prior))


def check_base_unchanged(base, tune):
    if base["record"]["tune_sha256_lf"] != tune["base"]["tune_sha256_lf"] or \
            base["record"]["eval_sha256_lf"] != tune["base"]["eval_sha256_lf"]:
        raise SystemExit("v2-6's records changed since tune.json (write a revision first)")


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
    check_base_unchanged(base, tune)
    c = tune["c46"]
    pools = EV.load_pools(EV.TERMS_DIR["E"], EV.E_SEEDS)
    Es = set(E43.splits()[0])
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in E43.store().pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    FPs, ctx, vec, bper = base_parts(base, "E", pools)
    octx = opp_ctx("E", ctx["models"])
    del ctx
    zO, miss = pools_opp_z(pools, octx, c["pw"], (c["agg"],))
    per = m46_scores(FPs, zO[c["agg"]], c["wO"], c["csls_k"], pools)
    byk = [EV.score_by_k(st.csls(FP + c["wO"] * z, c["csls_k"]), p) for FP, z, p in zip(FPs, zO[c["agg"]], pools)]
    rec = {"format": "c10-prereg46 eval v1", "e_seeds": EV.E_SEEDS, "e_pools_use_number": a.e_use,
           "earlier_e_judgements": prior, "c46": c, "cnn_vectors_E": vec, "base": tune["base"],
           "seam_base_E_identical": True, "tune_sha256": st.sha256_lf(a.tune), "git_head": L.git_head(),
           "opponent_sets": octx["sets"], "questions_without_opponent_vector_per_pool": miss,
           "means": {"c46": float(np.mean(per)), "base": float(np.mean(bper))},
           "per_pool": {"c46": per, "base": bper}, "by_k": {"c46": EV.mean_by_k(byk)}}
    j = E43.judge_rule(per, bper)
    rec["c46_minus_base"] = {x: j[x] for x in ("mean", "ci95", "pools_higher")}
    rec["rule"] = j["rule"].replace("c*", "c46")
    rec["adopt"] = j["adopt"]
    rec["peak_working_set_mb"] = peak_mb()
    rec["seconds"] = round(time.time() - t0, 1)
    L.write_json(a.out, rec)
    print(json.dumps({x: rec[x] for x in ("means", "c46_minus_base", "by_k", "adopt")}, indent=1))
    return 0


# ================================================================ official pools
def cmd_apply(a):
    t0 = time.time()
    tune, ev = read_json(a.tune), read_json(a.eval)
    if not ev["adopt"] or ev["tune_sha256"] != st.sha256_lf(a.tune):
        raise SystemExit("eval.json does not adopt c46 (or belongs to another tune.json): keep v2-6")
    base = E45.v26()
    check_base_unchanged(base, tune)
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
    O = load_set_side(name, games=Pset["games"])
    Zo = F10.project_p1(O["X"], models[E43.P1_VARIANT])
    c = tune["c46"]
    zO, has = opp_z(Zo, O["usable"], rec["cand_rows"], rec["q_rows"], c["pw"], (c["agg"],))[c["agg"]]
    data = P39E.top5_bytes(qids, st.top5(st.csls(FP + c["wO"] * zO, c["csls_k"]), cand_ids))
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "top5.csv")
    with open(out, "wb") as f:
        f.write(data)
    G = BK.pool_games(e, rec)
    rep = {"format": "c10-prereg46 apply v1", "pool": a.pool, "c46": c, "vectors": _rel(path),
           "vectors_sha256": L.sha256_file(path), "cfg": base["cfg"], "v26_top5_equals_prereg43_apply": True,
           "questions": len(qids), "candidates": len(cand_ids), "games": len(items),
           "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "games_without_P1": bad, "games_without_opponent_P1": int((~O["usable"]).sum()),
           "questions_without_opponent_P1": int((~np.asarray(has)).sum()),
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
    m.add_argument("--out", default=os.path.join(HERE, "seam.json"))
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
