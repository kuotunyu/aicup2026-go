"""prereg59 control checks on the kernel players (CONTROLS_PLAN.md; record only). CPU, .venv, 4 cores, <= 3 GB
(p58_lib.limits). Imports p58 / p59 code unchanged. No inner game, no test game, never data/local.

    PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe results/prereg59/explore/p59_controls.py
    -> results/prereg59/explore/controls.json
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import p59_eval as EV  # noqa: E402
import p59_lib as K  # noqa: E402

E58, K58 = EV.E58, EV.K58
OUT = os.path.join(HERE, "controls.json")
NS = (5, 10, 15, 20)
DRAWS = 20
SEED_Q, SEED_SHUF, SEED_TER = 5901, 5903, 5904
CENT_TAUS = (30, 50, 80, 120)
log = EV.log


def main():
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(OUT):
        raise SystemExit("controls.json exists")
    if not EV.committed_clean(os.path.join(HERE, "CONTROLS_PLAN.md")):
        raise SystemExit("CONTROLS_PLAN.md is not committed and clean")
    M, par, pr, krec = EV.load_kernel(K.KERNEL, K.PROTO)
    g, Wn, wr, hrec, _files = E58.kernel_games()
    players = sorted(set(g["player_id"].tolist()))
    pindex = {p: i for i, p in enumerate(pr["player_id"].tolist())}
    own = np.array([pindex[p] for p in g["player_id"].tolist()], np.int64)
    hd, _jrec = E58.hd_join(g["rank"], g["game_id"], g["player_id"], g["color"])
    r_g = K58.rank_index(g["rank"])
    h_g, c_g = K58.hd_group(hd), K58.colour_index(g["color"])
    ok = g["count"] > 0
    if not ok.all():
        raise SystemExit("a kernel game without a sample (kernel.json has none)")
    pidx = {p: i for i, p in enumerate(players)}
    gp = np.array([pidx[p] for p in g["player_id"].tolist()], np.int64)
    P = len(players)
    p_rank = np.zeros(P, np.int64)
    p_rank[gp] = r_g
    p_fold = np.array([K58.player_fold(p) for p in players], np.int64)
    f_g = p_fold[gp]
    G = len(gp)
    SM = K58.score_matrix()
    lprior = np.zeros((K.FOLDS, K.L))
    for f in range(K.FOLDS):
        pri = np.bincount(p_rank[p_fold != f], minlength=K.L) + 1.0
        lprior[f] = np.log(pri / pri.sum())
    idx = np.arange(G)
    by_p = [idx[gp == i] for i in range(P)]

    def cf_ll(q):
        """Per-game l_g(r) with M cross-fitted by player fold from this q."""
        Mf = K.fold_tables(q, r_g, h_g, c_g, f_g, ok)
        lg = np.zeros((G, K.L))
        for f in range(K.FOLDS):
            te = np.flatnonzero(f_g == f)
            lg[te] = K.game_ll(q[te], Mf[f], h_g[te], c_g[te])
        return lg

    # ------------------------------------------------ real q, all grid points (seams b, a; control 5)
    parts = K.class_parts(g["emb"], Wn, wr, pr["proto"], pr["rank"], own)
    log(f"class parts ({time.time() - t0:.0f}s, peak {EV.peak_gb(st)} GB)")
    kgrid = {(x["tau1"], x["tau2"], x["b"]): x for x in krec["per_grid_point"]}
    grid_rec, seam_b = [], True
    for t1, t2, b in K.GRID:
        q = K.combine(parts, t1, t2, b)
        ll, lln, _bf = K.heldout_ll(q, r_g, h_g, c_g, f_g, ok)
        seam_b &= bool(ll == kgrid[(t1, t2, b)]["heldout_mean_loglik"]
                       and lln == kgrid[(t1, t2, b)]["record_only_heldout_mean_log_posterior_true_rank"])
        z = lprior[f_g] + cf_ll(q)
        z -= z.max(axis=1, keepdims=True)
        z -= np.log(np.exp(z).sum(axis=1, keepdims=True))
        lt = z[idx, r_g]
        per_player = np.bincount(gp, weights=lt, minlength=P) / np.bincount(gp, minlength=P)
        grid_rec.append({"tau1": t1, "tau2": t2, "b": b, "heldout_mean_loglik": ll,
                         "per_game_no_prior_log_posterior": lln,
                         "per_player_fold_prior_log_posterior": float(per_player.mean())})
    if not seam_b:
        raise SystemExit("seam b failed: the grid's held-out likelihoods are not kernel.json's")
    log(f"seam b ok ({time.time() - t0:.0f}s)")

    def position(key):
        v = np.array([x[key] for x in grid_rec])
        ci = [i for i, x in enumerate(grid_rec) if (x["tau1"], x["tau2"], x["b"]) == par][0]
        bi = int(np.argmax(v))
        return {"chosen": float(v[ci]), "chosen_position_of_60": int((v > v[ci]).sum()) + 1,
                "best": float(v[bi]), "best_point": [grid_rec[bi][k] for k in ("tau1", "tau2", "b")],
                "chosen_minus_best": float(v[ci] - v[bi]), "worst": float(v.min()),
                "near_top": bool(v[bi] - v[ci] <= 0.005 or (v > v[ci]).sum() < 10),
                "uniform_reference": float(-np.log(K.L))}

    c5 = {"per_player_fold_prior": position("per_player_fold_prior_log_posterior"),
          "per_game_no_prior_kernel_json": position("per_game_no_prior_log_posterior"),
          "registered_objective": position("heldout_mean_loglik"), "grid": grid_rec}
    q_real = K.combine(parts, *par)
    del parts
    lg_real = cf_ll(q_real)

    # seam a: kernel.json's standalone record, its own draw order
    rng = np.random.default_rng(EV.SEED)
    res = {n: [] for n in EV.NS}
    for f in range(K.FOLDS):
        for i in np.flatnonzero(p_fold == f).tolist():
            for n in EV.NS:
                if len(by_p[i]) < n:
                    continue
                draws = np.stack([lg_real[rng.choice(by_p[i], n, replace=False)].sum(0) for _ in range(EV.DRAWS)])
                res[n] += SM[K58.decode(lprior[f][None] + draws), p_rank[i]].tolist()
    sa = krec["standalone_score_record"]
    seam_a = {f"n{n}": [float(np.mean(res[n])), sa[f"n{n}"]["mean"]] for n in EV.NS}
    if not all(a == b for a, b in seam_a.values()):
        raise SystemExit(f"seam a failed: {seam_a}")
    log(f"seam a ok: {seam_a}")

    # ------------------------------------------------ direct soft-max: seam c, top-1 / top-10 removed, top-1 prob
    t1, t2, b = par
    Pn, prk = pr["proto"], pr["rank"]
    cr_all = np.concatenate([np.asarray(wr), prk])
    onehot = np.zeros((len(cr_all), K.L))
    onehot[np.arange(len(cr_all)), cr_all] = 1.0
    q_dir, q_d1, q_d10 = (np.zeros((G, K.L)) for _ in range(3))
    top1p, top1_is_proto, top1_rank = np.zeros(G), np.zeros(G, bool), np.zeros(G, np.int64)
    nh = len(wr)

    def sm_rank(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        Pm = np.exp(Z)
        Pm /= Pm.sum(axis=1, keepdims=True)
        return Pm

    for s in range(0, G, 4000):
        e = K.l2(g["emb"][s:s + 4000])
        Cp = e @ Pn.T
        o = own[s:s + 4000]
        Cp[np.arange(len(o)), o] = -np.inf
        Z = np.concatenate([t1 * (e @ Wn.T), t2 * Cp + b], axis=1)
        Pm = sm_rank(Z)
        q_dir[s:s + 4000] = Pm @ onehot
        a1 = Pm.argmax(axis=1)
        top1p[s:s + 4000] = Pm[np.arange(len(o)), a1]
        top1_is_proto[s:s + 4000] = a1 >= nh
        top1_rank[s:s + 4000] = cr_all[a1]
        a10 = np.argpartition(-Z, 10, axis=1)[:, :10]
        Z1 = Z.copy()
        Z1[np.arange(len(o)), a1] = -np.inf
        q_d1[s:s + 4000] = sm_rank(Z1) @ onehot
        np.put_along_axis(Z, a10, -np.inf, axis=1)
        q_d10[s:s + 4000] = sm_rank(Z) @ onehot
    seam_c = float(np.abs(q_dir - q_real).max())
    if seam_c > 1e-9:
        raise SystemExit(f"seam c failed: {seam_c}")
    log(f"seam c ok {seam_c:.2e} ({time.time() - t0:.0f}s, peak {EV.peak_gb(st)} GB)")

    # ------------------------------------------------ control q's
    q_const = np.zeros((G, K.L))
    for f in range(K.FOLDS):
        q_const[f_g == f] = q_real[f_g != f].mean(axis=0)
    rs = np.random.default_rng(SEED_SHUF)
    donor = idx.copy()
    cellk = K.cell(r_g, h_g, c_g)
    kept = 0
    for ck in np.unique(cellk):
        m = np.flatnonzero(cellk == ck)
        if len(np.unique(gp[m])) < 2:
            kept += len(m)
            continue
        d = rs.choice(m, len(m))
        bad = gp[d] == gp[m]
        while bad.any():
            d[bad] = rs.choice(m, int(bad.sum()))
            bad = gp[d] == gp[m]
        donor[m] = d
    ps = K.class_parts(g["emb"][donor], Wn, wr, Pn, prk, own[donor], tau1=(t1,), tau2=(t2,))
    q_shuf = K.combine(ps, *par)
    del ps
    cent = K.l2(np.stack([Wn[np.asarray(wr) == s].mean(axis=0) for s in range(K.L)]))
    cent_rec, q_cent, best = [], None, None
    for s in range(0, G, 20000):
        pass
    Cc = np.concatenate([K.l2(g["emb"][s:s + 20000]) @ cent.T for s in range(0, G, 20000)])
    for tau in CENT_TAUS:
        qc = sm_rank(tau * Cc)
        ll, lln, _bf = K.heldout_ll(qc, r_g, h_g, c_g, f_g, ok)
        cent_rec.append({"tau": tau, "heldout_mean_loglik": ll})
        if best is None or ll > best:
            best, q_cent, cent_tau = ll, qc, tau
    log(f"control q's ready ({time.time() - t0:.0f}s)")

    # ------------------------------------------------ questions (shared) and scores
    rq = np.random.default_rng(SEED_Q)
    mono_ok = h_g <= 1
    Q = {}
    for shape in ("any", "mono"):
        for n in NS:
            gi, pi = [], []
            for i in range(P):
                if shape == "any":
                    pools = [by_p[i]] if len(by_p[i]) >= n else []
                else:
                    pools = [x for x in (by_p[i][mono_ok[by_p[i]] & (c_g[by_p[i]] == c)] for c in (0, 1))
                             if len(x) >= n]
                if not pools:
                    continue
                for _ in range(DRAWS):
                    pool = pools[int(rq.integers(len(pools)))]
                    gi.append(rq.choice(pool, n, replace=False))
                    pi.append(i)
            Q[(shape, n)] = (np.array(gi), np.array(pi))

    def score(lg, gi, pi):
        z = lprior[p_fold[pi]] + (lg[gi].sum(axis=1) if lg is not None else 0.0)
        sc = SM[K58.decode(z), p_rank[pi]]
        pm = np.bincount(pi, weights=sc, minlength=P)[np.unique(pi)] / DRAWS
        return {"mean": float(sc.mean()), "se_by_player": float(pm.std(ddof=1) / np.sqrt(len(pm))),
                "questions": int(len(sc)), "players": int(len(pm)),
                "by_rank": [float(sc[p_rank[pi] == k].mean()) for k in range(K.L)]}

    variants = {"prior_only": None, "1_real": lg_real, "2_constant_q": cf_ll(q_const),
                "3_shuffled_within_rank": cf_ll(q_shuf), "3b_rank_centroids": cf_ll(q_cent),
                "4a_top1_removed": cf_ll(q_d1), "4a_top10_removed": cf_ll(q_d10)}
    scores = {v: {shape: {f"n{n}": score(lg, *Q[(shape, n)]) for n in NS} for shape in ("any", "mono")}
              for v, lg in variants.items()}
    # 4b: terciles
    rt = np.random.default_rng(SEED_TER)
    ter = {}
    for name, val in (("max_s_q", q_real.max(axis=1)), ("top1_class_prob", top1p)):
        th = np.quantile(val, [1 / 3, 2 / 3])
        tg = np.digitize(val, th)
        ter[name] = {"thresholds": th.tolist(), "tercile": {}}
        for t, tn in enumerate(("lowest", "middle", "highest")):
            cell = {"games": int((tg == t).sum()), "games_per_rank": np.bincount(r_g[tg == t], minlength=K.L).tolist()}
            for n in (5, 10):
                gi, pi = [], []
                for i in range(P):
                    pool = by_p[i][tg[by_p[i]] == t]
                    if len(pool) < n:
                        continue
                    for _ in range(DRAWS):
                        gi.append(rt.choice(pool, n, replace=False))
                        pi.append(i)
                gi, pi = np.array(gi), np.array(pi)
                cell[f"n{n}"] = {"term": score(lg_real, gi, pi), "prior_only": score(None, gi, pi)["mean"]}
            ter[name]["tercile"][tn] = cell
    rep = {"format": "c10-prereg59 controls v1", "created": EV.now(), "seconds": round(time.time() - t0, 1),
           "code": EV.code_state(), "plan": "CONTROLS_PLAN.md (committed before this run)",
           "what": "record only: kernel players (D, E), cross-fitted by player fold; no inner game, no test game",
           "tau1_tau2_b": list(par), "players": P, "games": G,
           "seams": {"a_standalone_record_bitwise": seam_a, "b_grid_loglik_bitwise": seam_b,
                     "c_direct_softmax_max_abs_diff": seam_c},
           "questions": {"draws_per_player": DRAWS, "seed": SEED_Q,
                         "mono_games_share": float(mono_ok.mean())},
           "scores": scores,
           "control3": {"seed": SEED_SHUF, "games_kept_own_vector": int(kept),
                        "mean_max_q_real": float(q_real.max(axis=1).mean()),
                        "mean_max_q_shuffled": float(q_shuf.max(axis=1).mean())},
           "control3b": {"centroids": "L2 mean of S0's unit head rows per rank (no prototype)", "tau": cent_tau,
                         "grid": cent_rec, "mean_max_q": float(q_cent.max(axis=1).mean())},
           "control4a": {"top1_class_prob_mean": float(top1p.mean()), "top1_class_prob_median": float(np.median(top1p)),
                         "top1_is_prototype_share": float(top1_is_proto.mean()),
                         "top1_class_same_rank_as_player": float((top1_rank == r_g).mean()),
                         "top1_class_within_one_rank": float((np.abs(top1_rank - r_g) <= 1).mean()),
                         "mean_max_q_top1_removed": float(q_d1.max(axis=1).mean()),
                         "mean_max_q_top10_removed": float(q_d10.max(axis=1).mean())},
           "control4b": ter, "control5": c5, "peak_working_set_gb": EV.peak_gb(st)}
    K58.write_json(OUT, rep)
    for v in scores:
        log(v.ljust(26) + " | ".join(" ".join(f"{scores[v][sh]['n%d' % n]['mean']:.4f}" for n in NS)
                                     for sh in ("any", "mono")))
    log(json.dumps({k: c5[k] for k in ("per_player_fold_prior", "per_game_no_prior_kernel_json")}))
    log(f"-> {OUT} ({time.time() - t0:.0f}s, peak {EV.peak_gb(st)} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
