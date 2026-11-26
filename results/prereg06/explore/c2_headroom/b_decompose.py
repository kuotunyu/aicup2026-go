"""c2_headroom (b): where Q02's loss comes from, on the official shape (single colour, k games).

Inner players only (prereg02 inner dumps, PREREG_03 single-colour pools). No validation data.

Q02's question model is linear in per-game means (tools/p04_rank_fuse.py gen_logits):
    logits = exp(a) * (n/12)^gamma * ( mean_g v_f[g] + w_OP * mean_g o[g] + B @ beta_f )
    v_f[g] = C5[g] + w_O4 * O4[g] + w_SWA * SWA[g] + w_SWB * SWB[g]      (per-game log-probabilities)
with the 5 fold parameter sets of PREREG_04 tune (u1_repro.json fold_thetas Q02). So a "virtual question" can be built
from ANY subset of a player's same-colour inner games, with the player's own fold parameters (OOF). The script first
checks that this reproduces mono_P_Q02 of u1_oof.npz.

Outputs b_decompose.json:
  recon_check            max |P - mono_P_Q02|
  curve_panelA / B       score vs k on fixed panels of (player, colour) pools (A: pool >= 20 games, B: pool >= 80)
  full_pool              one question per pool using every same-colour inner game of that player
  decomposition          k = 5 / 10 / 15 / 20 on panel A: loss split into "wrong even with the whole pool"
                         (player level) and the rest (question level, i.e. too few games); variance shares
  more_info              test-shape score (mean over k = 5/10/15/20) if every question had f times as many games
                         (f = 2, 4), panel B, + curve fit s(k) = s_inf - c * k^-beta
  shared_bias            per-player within-level residual of the mean per-game expected level for each per-game
                         source (own moves: C5, T4; opponent's moves: O4, SWA, SWB); split-half reliabilities and
                         disattenuated cross-source correlations
"""

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hc_lib as H  # noqa: E402

U = H.U
p4 = U.p4
L = 10
RNG_SEED = 20261002


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class Model:
    def __init__(self, lp, op_o, op_has, thetas):
        self.thetas = [np.asarray(t, dtype=np.float64) for t in thetas]
        self.v, self.ow, self.bias, self.a, self.g = [], [], [], [], []
        for th in self.thetas:
            a, g = th[0], th[1]
            w = th[2:6]                     # OP, O4, SWA, SWB (p4.TERMS order)
            beta = th[6:]
            self.v.append(lp["C5"] + w[1] * lp["O4"] + w[2] * lp["SWA"] + w[3] * lp["SWB"])
            self.ow.append(w[0] * op_o)
            self.bias.append(p4.B @ beta)
            self.a.append(a)
            self.g.append(g)
        self.has = op_has.astype(np.float64)

    def u(self, games, fold):
        """games [Q, n] (int rows, -1 = padding) of questions all in this fold -> u [Q, 10], n [Q]."""
        mask = games >= 0
        gi = np.where(mask, games, 0)
        n = mask.sum(1).astype(np.float64)
        v = self.v[fold][gi] * mask[..., None]
        m = v.sum(1) / n[:, None]
        h = self.has[gi] * mask
        nh = h.sum(1)
        o = (self.ow[fold][gi] * h[..., None]).sum(1) / np.maximum(nh, 1)[:, None]
        return m + o + self.bias[fold], n

    def probs(self, games, fold):
        u, n = self.u(games, fold)
        t = np.exp(self.a[fold]) * np.power(n / 12.0, self.g[fold])
        return softmax(t[:, None] * u)


def score_rule_b(P, y):
    return H.score(H.rule_b(P), y)


def rb_units(vals, unit_level):
    """level-balanced mean of per-unit values (units weighted equally inside a level)."""
    return float(np.mean([vals[unit_level == l].mean() for l in range(L) if (unit_level == l).any()]))


class UnitBoot:
    """players resampled within level (1000 draws, default_rng(0)); a player's units (colour pools) move together."""

    def __init__(self, unit_player, unit_level, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        players = np.unique(unit_player)
        plev = {p: unit_level[unit_player == p][0] for p in players}
        self.units_of = {p: np.nonzero(unit_player == p)[0] for p in players}
        self.draws = []
        for _ in range(n_boot):
            sel = []
            for l in range(L):
                mem = [p for p in players if plev[p] == l]
                if not mem:
                    continue
                pick = rng.integers(0, len(mem), size=len(mem))
                sel.append(np.concatenate([self.units_of[mem[i]] for i in pick]))
            self.draws.append(sel)

    def ci(self, vals):
        st = [np.mean([vals[s].mean() for s in sel]) for sel in self.draws]
        return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def main():
    t0 = time.time()
    d = U.load_inputs()
    inner, lp = d["inner"], d["lp"]
    keys = inner["keys"]
    z = U.load_saved()
    with open(os.path.join(H.U1, "u1_repro.json"), encoding="utf-8") as fh:
        thetas = json.load(fh)["fold_thetas"]["Q02"]
    model = Model(lp, inner["op_o"], inner["op_has"], thetas)
    out = {"what": "c2_headroom (b): Q02 loss decomposition on inner single-colour pools (design stage; NOT "
                   "validation)", "thetas": "results/prereg05/explore/u1/u1_repro.json fold_thetas Q02"}

    # ---------------- reconstruction check against u1_oof.npz mono_P_Q02
    mg, mf, mP = z["mono_games"], z["mono_fold"], z["mono_P_Q02"]
    md = 0.0
    for f in range(5):
        sel = np.nonzero(mf == f)[0]
        for s in np.array_split(sel, 20):
            md = max(md, float(np.abs(model.probs(mg[s], f) - mP[s]).max()))
    out["recon_check"] = {"max_abs_diff_vs_mono_P_Q02": md, "questions": int(len(mf))}
    print(f"recon max diff {md:.3e} ({time.time() - t0:.0f}s)", flush=True)
    assert md < 1e-9

    # ---------------- pools (player, colour) -> inner rows
    pid = z["player_id"]
    plevel, pfold = z["player_level"], z["player_fold"]
    index = {p: i for i, p in enumerate(pid)}
    pools = {}
    for r, (p, c) in enumerate(zip(keys["player_id"], keys["color"])):
        i = index.get(p)
        if i is not None:
            pools.setdefault((i, c), []).append(r)
    units = sorted(pools)
    u_player = np.array([u[0] for u in units])
    u_level = plevel[u_player]
    u_fold = pfold[u_player]
    u_N = np.array([len(pools[u]) for u in units])
    out["pools"] = {"units": len(units), "N_quantiles": H.pct(u_N, (0, 10, 25, 50, 75, 90, 100)),
                    "units_N_ge_20": int((u_N >= 20).sum()), "units_N_ge_40": int((u_N >= 40).sum()),
                    "units_N_ge_80": int((u_N >= 80).sum()),
                    "units_N_ge_80_per_level": {H.LEVELS[l]: int(((u_N >= 80) & (u_level == l)).sum())
                                                for l in range(L)}}
    rng = np.random.default_rng(RNG_SEED)
    R = 32

    def unit_scores(sel_units, k, rounds=R):
        """per-unit mean rule-B score over `rounds` random k-subsets (k = None: whole pool once)."""
        vals = np.zeros(len(sel_units))
        allsc = []
        for f in range(5):
            idx = [j for j, ui in enumerate(sel_units) if u_fold[ui] == f]
            if not idx:
                continue
            if k is None:
                Nmax = max(u_N[sel_units[j]] for j in idx)
                G = np.full((len(idx), Nmax), -1, dtype=np.int64)
                for row, j in enumerate(idx):
                    rows = pools[units[sel_units[j]]]
                    G[row, :len(rows)] = rows
                P = model.probs(G, f)
                y = u_level[np.array(sel_units)[idx]]
                sc = score_rule_b(P, y)
                vals[idx] = sc
                continue
            G = np.empty((len(idx) * rounds, k), dtype=np.int64)
            for row, j in enumerate(idx):
                rows = np.asarray(pools[units[sel_units[j]]])
                for r in range(rounds):
                    G[row * rounds + r] = rows[rng.choice(len(rows), k, replace=False)]
            P = model.probs(G, f)
            y = np.repeat(u_level[np.array(sel_units)[idx]], rounds)
            sc = score_rule_b(P, y).reshape(len(idx), rounds)
            vals[idx] = sc.mean(1)
            allsc.append((np.array(idx), sc))
        return vals, allsc

    # ---------------- curves
    panels = {"A_N_ge_20": (20, (1, 2, 3, 5, 7, 10, 15, 20)), "B_N_ge_80": (80, (5, 10, 15, 20, 30, 40, 60, 80))}
    store = {}
    for name, (nmin, ks) in panels.items():
        sel = np.nonzero(u_N >= nmin)[0]
        boot = UnitBoot(u_player[sel], u_level[sel])
        res = {"units": int(len(sel)), "players": int(len(np.unique(u_player[sel]))),
               "units_per_level": {H.LEVELS[l]: int((u_level[sel] == l).sum()) for l in range(L)}, "by_k": {}}
        for k in ks:
            vals, allsc = unit_scores(list(sel), k)
            store[(name, k)] = (vals, allsc)
            res["by_k"][str(k)] = {"rb": rb_units(vals, u_level[sel]), "ci95": boot.ci(vals)}
        vals, _ = unit_scores(list(sel), None)
        store[(name, "full")] = (vals, None)
        res["full_pool"] = {"rb": rb_units(vals, u_level[sel]), "ci95": boot.ci(vals),
                            "N_mean": float(u_N[sel].mean()), "N_median": float(np.median(u_N[sel]))}
        # test shape at f = 1, 2, 4 where available
        ts = {}
        for f_ in (1, 2, 4):
            kk = [k * f_ for k in (5, 10, 15, 20)]
            if all((name, k) in store for k in kk):
                v = np.mean([store[(name, k)][0] for k in kk], axis=0)
                ts[str(f_)] = {"rb": rb_units(v, u_level[sel]), "ci95": boot.ci(v), "ks": kk}
                if f_ > 1:
                    v1 = np.mean([store[(name, k)][0] for k in (5, 10, 15, 20)], axis=0)
                    ts[str(f_)]["gain_vs_f1"] = rb_units(v - v1, u_level[sel])
                    ts[str(f_)]["gain_ci95"] = boot.ci(v - v1)
        res["test_shape"] = ts
        # curve fit s(k) = s_inf - c * k^-beta on the panel's points
        kx = np.array(ks, dtype=np.float64)
        sy = np.array([res["by_k"][str(k)]["rb"] for k in ks])
        try:
            popt, _ = curve_fit(lambda k, s, c, b: s - c * np.power(k, -b), kx, sy, p0=(0.7, 0.3, 0.5),
                                maxfev=20000)
            res["curve_fit"] = {"s_inf": float(popt[0]), "c": float(popt[1]), "beta": float(popt[2]),
                                "max_abs_resid": float(np.max(np.abs(sy - (popt[0] - popt[1] * kx ** -popt[2]))))}
            sk = lambda k: popt[0] - popt[1] * k ** -popt[2]  # noqa: E731
            res["curve_fit"]["test_shape_f"] = {str(f_): float(np.mean([sk(f_ * k) for k in (5, 10, 15, 20)]))
                                                for f_ in (1, 2, 4, 8)}
        except Exception as e:  # pragma: no cover
            res["curve_fit"] = {"error": str(e)}
        out["curve_" + name] = res
        print(f"panel {name} done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- decomposition on panel A (k = 5/10/15/20) against the whole-pool answer
    name = "A_N_ge_20"
    sel = np.nonzero(u_N >= 20)[0]
    lev = u_level[sel]
    full = store[(name, "full")][0]
    dec = {}
    for k in (5, 10, 15, 20):
        vals, allsc = store[(name, k)]
        loss = 1.0 - vals
        wrong_full = full < 1.0                      # whole pool answer not exact
        far_full = full == 0.0
        w = np.array([1.0 / (lev == l).sum() for l in lev])
        # variance shares of the question scores
        Q = np.concatenate([sc for _, sc in allsc], axis=0)        # [units, R] (fold-ordered)
        order = np.concatenate([idx for idx, _ in allsc])
        lv_o = lev[order]
        within = Q.var(axis=1, ddof=1).mean()
        between_raw = Q.mean(1).var(ddof=1)
        between = between_raw - within / Q.shape[1]
        total = Q.var(ddof=1)
        dec[str(k)] = {
            "rb": rb_units(vals, lev),
            "full_pool_rb": rb_units(full, lev),
            "loss_rb": rb_units(loss, lev),
            "loss_share_from_units_whose_full_pool_answer_is_not_exact": float((w * loss * wrong_full).sum() /
                                                                               (w * loss).sum()),
            "loss_share_from_units_whose_full_pool_answer_is_2plus_off": float((w * loss * far_full).sum() /
                                                                               (w * loss).sum()),
            "units_full_pool_not_exact_share_rb": rb_units(wrong_full.astype(float), lev),
            "units_full_pool_2plus_off_share_rb": rb_units(far_full.astype(float), lev),
            "variance_share_between_units": float(between / total),
            "variance_share_between_units_within_level_only": float(
                np.mean([Q[lv_o == l].mean(1).var(ddof=1) - Q[lv_o == l].var(1, ddof=1).mean() / Q.shape[1]
                         for l in range(L) if (lv_o == l).sum() > 2]) /
                np.mean([Q[lv_o == l].var(ddof=1) for l in range(L) if (lv_o == l).sum() > 2])),
        }
    out["decomposition_panelA"] = dec

    # ---------------- shared player-level bias across sources (own moves vs opponent's moves)
    srcs = {"C5": lp["C5"], "T4": lp["T4"], "O4": lp["O4"], "SWA": lp["SWA"], "SWB": lp["SWB"]}
    ev = {s: (softmax(v) * np.arange(L)).sum(1) for s, v in srcs.items()}   # per-game expected level index
    q02v = None
    rows_of = {}
    for r, p in enumerate(keys["player_id"]):
        i = index.get(p)
        if i is not None:
            rows_of.setdefault(i, []).append(r)
    pl = np.array(sorted(i for i in rows_of if len(rows_of[i]) >= 20))
    lvl = plevel[pl]

    def resid(mean_vals):
        r = mean_vals.copy()
        for l in range(L):
            m = lvl == l
            r[m] -= mean_vals[m].mean()
        return r

    full_res, half1, half2 = {}, {}, {}
    hr = np.random.default_rng(RNG_SEED + 1)
    splits = {}
    for i in pl:
        rows = np.array(rows_of[i])
        perm = hr.permutation(len(rows))
        splits[i] = (rows[perm[: len(rows) // 2]], rows[perm[len(rows) // 2:]])
    for s in srcs:
        full_res[s] = resid(np.array([ev[s][rows_of[i]].mean() for i in pl]))
        half1[s] = resid(np.array([ev[s][splits[i][0]].mean() for i in pl]))
        half2[s] = resid(np.array([ev[s][splits[i][1]].mean() for i in pl]))
    rel = {}
    for s in srcs:
        rho = float(np.corrcoef(half1[s], half2[s])[0, 1])
        rel[s] = {"split_half_r": rho, "reliability_spearman_brown": 2 * rho / (1 + rho)}
    cross = {}
    names = list(srcs)
    for a_i, a in enumerate(names):
        for b in names[a_i + 1:]:
            r_ab = float(np.corrcoef(full_res[a], full_res[b])[0, 1])
            # cross-half correlation (independent game halves) avoids shared-game noise
            r_x = float(np.mean([np.corrcoef(half1[a], half2[b])[0, 1], np.corrcoef(half2[a], half1[b])[0, 1]]))
            r_aa = rel[a]["split_half_r"]
            r_bb = rel[b]["split_half_r"]
            cross[f"{a}~{b}"] = {"r_full": r_ab, "r_cross_half": r_x,
                                 "disattenuated_cross_half": r_x / np.sqrt(r_aa * r_bb)}
    # Q02 player-level signed error (whole-pool question, each colour, mean over colours) vs residuals
    sgn = np.full(len(pl), np.nan)
    for j, i in enumerate(pl):
        e = []
        for c in ("B", "W"):
            if (i, c) in pools and len(pools[(i, c)]) >= 5:
                G = np.array([pools[(i, c)]])
                P = model.probs(G, int(pfold[i]))
                e.append(float((P[0] * np.arange(L)).sum()) - plevel[i])
        if e:
            sgn[j] = np.mean(e)
    ok = ~np.isnan(sgn)
    q02v = {s: float(np.corrcoef(sgn[ok], full_res[s][ok])[0, 1]) for s in srcs}
    out["shared_bias"] = {
        "players": int(len(pl)), "min_inner_games": 20,
        "per_game_statistic": "expected level index under softmax of the per-game source log-probabilities; "
                              "per player mean over all inner games (both colours); residual = minus the "
                              "mean over players of the same level",
        "sources": {"C5": "own moves, prereg02 CNN (R20 source)", "T4": "own moves, O4 count tables",
                    "O4": "opponent's moves, count tables", "SWA": "opponent's moves, prereg03 CNN arm A",
                    "SWB": "opponent's moves, prereg03 CNN arm B"},
        "reliability": rel, "cross": cross,
        "corr_with_Q02_whole_pool_signed_error": q02v, "players_with_q02_error": int(ok.sum())}
    out["seconds"] = round(time.time() - t0, 1)
    print(H.dump(out, "b_decompose.json"))


if __name__ == "__main__":
    main()
