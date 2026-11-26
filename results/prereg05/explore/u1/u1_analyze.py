"""prereg05 exploration U1, step 2: where does Q02 (R20 + O4 + SWA + SWB) lose points on the inner pseudo-questions?

    .venv/Scripts/python.exe results/prereg05/explore/u1/u1_analyze.py   (needs u1_oof.npz from u1_repro.py)
    -> results/prereg05/explore/u1/u1_analysis.json

Design-stage numbers on inner pseudo-questions only (prereg02 eval rounds 1000-1199 of the 636 split-v1 inner bank
players, and the PREREG_03 single-colour set). Not validation results. Rule B everywhere unless stated.
Two bootstraps are reported for every difference:
  qboot   paired, questions resampled within each level (2,000 draws, default_rng(0)) -- what the task asked for; too
          narrow here, because the 200 rounds of one player reuse the same <= 20 games
  pboot   paired PlayerBootstrap of tools/p02_rank_fuse.py (players within each level, 1,000 draws, default_rng(0))
          -- the registered one (PREREG_04 G2); use this one for decisions
"""

import json
import os
import time

import numpy as np

import u1_lib as U

T0 = time.time()
L = 10
LV = U.LEVELS
QBOOT, QSEED = 2000, 0


# ================================================================ helpers
def rb(sc, y, mask=None):
    return U.rank_balanced(sc, y, mask)


def per_level(sc, y, mask=None):
    sc, y = np.asarray(sc, float), np.asarray(y)
    if mask is not None:
        sc, y = sc[mask], y[mask]
    return {LV[i]: (float(sc[y == i].mean()) if (y == i).any() else None) for i in range(L)}


def qboot_diff(sa, sb, y, mask=None, draws=QBOOT, seed=QSEED):
    """Paired, questions resampled within each level; statistic rank-balanced mean difference."""
    d = np.asarray(sa, float) - np.asarray(sb, float)
    y = np.asarray(y)
    if mask is not None:
        d, y = d[mask], y[mask]
    rng = np.random.default_rng(seed)
    per = []
    for i in range(L):
        di = d[y == i]
        if not len(di):
            continue
        acc = np.empty(draws)
        for s in range(0, draws, 200):
            e = min(draws, s + 200)
            idx = rng.integers(0, len(di), size=(e - s, len(di)))
            acc[s:e] = di[idx].mean(1)
        per.append(acc)
    st = np.mean(per, axis=0)
    return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


class PB:
    """PlayerBootstrap wrapper (tools/p02_rank_fuse.py), cached per question subset."""

    def __init__(self, player, y, n_players=636):
        self.player, self.y, self.n_players = player, y, n_players
        self.cache = {}

    def get(self, mask):
        key = None if mask is None else hash(mask.tobytes())
        if key not in self.cache:
            if mask is None:
                self.cache[key] = U.PlayerBootstrap(self.player, self.y, self.n_players)
            else:
                self.cache[key] = U.PlayerBootstrap(self.player[mask], self.y[mask], self.n_players)
        return self.cache[key]

    def diff(self, sa, sb, mask=None):
        b = self.get(mask)
        if mask is not None:
            sa, sb = sa[mask], sb[mask]
        d = b.stats(sa) - b.stats(sb)
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float(np.std(d, ddof=1))

    def se(self, sa, mask=None):
        b = self.get(mask)
        return float(np.std(b.stats(sa if mask is None else sa[mask]), ddof=1))


def diff_record(sa, sb, y, pb, mask=None):
    ci, se = pb.diff(sa, sb, mask)
    return {"delta": rb(sa, y, mask) - rb(sb, y, mask), "pboot_ci95": ci, "pboot_se": se,
            "qboot_ci95": qboot_diff(sa, sb, y, mask),
            "question_mean_delta": float(np.mean((sa - sb) if mask is None else (sa - sb)[mask])),
            "per_level_delta": {k: (a - b if a is not None else None) for (k, a), b in
                                zip(per_level(sa, y, mask).items(), per_level(sb, y, mask).values())}}


def wnll(P, y):
    """Level-weighted NLL (the calibration's objective without the penalty) and plain NLL."""
    ll = -np.log(np.maximum(P[np.arange(len(y)), y], 1e-300))
    cnt = np.bincount(y, minlength=L).astype(float)
    w = 1.0 / (np.count_nonzero(cnt) * cnt[y])
    return float(np.sum(w * ll)), float(ll.mean())


def decide_b(P, c=U.E1):
    left = np.zeros_like(P)
    right = np.zeros_like(P)
    left[:, 1:] = P[:, :-1]
    right[:, :-1] = P[:, 1:]
    return np.argmax(P + c * (left + right), axis=1)


def basic(P, y, mask=None):
    if mask is not None:
        P, y = P[mask], y[mask]
    pred = U.rule_b_np(P)
    d = pred - y
    sc = U.score(pred, y)
    ES = U.expected_scores(P)
    es_chosen = ES[np.arange(len(y)), pred]
    cnt = np.bincount(y, minlength=L).astype(float)
    w = 1.0 / (np.count_nonzero(cnt) * cnt[y])     # rank-balanced weights (sum 1)
    out = {"n_questions": int(len(y)), "rank_balanced": rb(sc, y), "question_mean": float(sc.mean()),
           "exact_rate_rb": float(np.sum(w * (d == 0))), "adjacent_rate_rb": float(np.sum(w * (np.abs(d) == 1))),
           "far_rate_rb": float(np.sum(w * (np.abs(d) >= 2))),
           "loss_rb": 1 - rb(sc, y),
           "loss_from_adjacent_rb": float(np.sum(w * (np.abs(d) == 1)) * (1 - U.E1)),
           "loss_from_far_rb": float(np.sum(w * (np.abs(d) >= 2))),
           "expected_score_of_choice_rb": float(np.sum(w * es_chosen)),
           "realized_minus_expected_rb": float(np.sum(w * (sc - es_chosen))),
           "mean_signed_error_levels_rb": float(np.sum(w * d)),
           "p_true_mean_rb": float(np.sum(w * P[np.arange(len(y)), y])),
           "nll_level_weighted": wnll(P, y)[0], "nll_plain": wnll(P, y)[1]}
    return out


# ================================================================ main
def main():
    z = U.load_saved()
    rep = {"what": "prereg05 exploration U1: breakdown of Q02's inner OOF score (design stage; inner pseudo-questions "
                   "only; NOT validation results)",
           "source": "results/prereg05/explore/u1/u1_oof.npz (u1_repro.py; reproduces PREREG_04 tune exactly)",
           "rule": "decision rule B (tools/decide.py) unless stated; rank-balanced mean = mean over the 10 levels of "
                   "the per-level mean official score",
           "bootstraps": {"qboot": f"paired, questions within each level, {QBOOT} draws, default_rng({QSEED}); too "
                                   f"narrow (200 rounds per player reuse the same <= 20 games)",
                          "pboot": "paired PlayerBootstrap (players within each level, 1000 draws, default_rng(0)), "
                                   "as PREREG_04 G2 -- the honest one"}}
    for setname in ("eval", "mono"):
        y = z[f"{setname}_y"]
        n = z[f"{setname}_n"]
        player = z[f"{setname}_player"]
        fold = z[f"{setname}_fold"]
        P0, P1, P2 = z[f"{setname}_P_Q00"], z[f"{setname}_P_Q01"], z[f"{setname}_P_Q02"]
        pb = PB(player, y)
        pred0, pred2 = U.rule_b_np(P0), U.rule_b_np(P2)
        sc0, sc1, sc2 = U.score(pred0, y), U.score(U.rule_b_np(P1), y), U.score(pred2, y)
        R = {}
        R["overall"] = {"Q00": basic(P0, y), "Q01": basic(P1, y), "Q02": basic(P2, y),
                        "Q02_minus_Q00": diff_record(sc2, sc0, y, pb), "Q02_se_pboot": pb.se(sc2)}
        print(f"[{setname}] Q00 {R['overall']['Q00']['rank_balanced']:.4f}  Q02 {R['overall']['Q02']['rank_balanced']:.4f}"
              f"  delta {R['overall']['Q02_minus_Q00']['delta']:+.4f} pboot {R['overall']['Q02_minus_Q00']['pboot_ci95']}"
              f" ({time.time() - T0:.0f}s)", flush=True)

        # ---- per level
        d2 = pred2 - y
        lvl = {}
        for i in range(L):
            mk = y == i
            dd = d2[mk]
            s = sc2[mk].mean()
            lvl[LV[i]] = {
                "questions": int(mk.sum()), "players": int(len(np.unique(player[mk]))),
                "Q00": float(sc0[mk].mean()), "Q02": float(s), "Q02_minus_Q00": float(s - sc0[mk].mean()),
                "exact": float((dd == 0).mean()), "adjacent_low": float((dd == -1).mean()),
                "adjacent_high": float((dd == 1).mean()), "far_low": float((dd <= -2).mean()),
                "far_high": float((dd >= 2).mean()), "mean_signed_error": float(dd.mean()),
                "lost_points_in_rank_balanced": float((1 - s) / L),
                "lost_from_adjacent": float((np.abs(dd) == 1).mean() * (1 - U.E1) / L),
                "lost_from_far": float((np.abs(dd) >= 2).mean() / L),
                "p_true_mean": float(P2[mk, i].mean()),
                "expected_score_of_choice": float(U.expected_scores(P2[mk])[np.arange(mk.sum()), pred2[mk]].mean()),
                "pboot_se": pb.se(sc2, mk)}
        R["per_level"] = lvl

        # ---- confusion matrix
        cm = np.zeros((L, L), dtype=np.int64)
        np.add.at(cm, (y, pred2), 1)
        cm0 = np.zeros((L, L), dtype=np.int64)
        np.add.at(cm0, (y, pred0), 1)
        R["confusion_Q02"] = {"rows_true_cols_pred": cm.tolist(),
                              "row_percent": np.round(100 * cm / cm.sum(1, keepdims=True), 2).tolist(),
                              "Q00_row_percent": np.round(100 * cm0 / cm0.sum(1, keepdims=True), 2).tolist(),
                              "pred_distribution": dict(zip(LV, np.bincount(pred2, minlength=L).tolist())),
                              "true_distribution": dict(zip(LV, np.bincount(y, minlength=L).tolist()))}

        # ---- by number of games
        byn = {}
        for k in sorted(np.unique(n)):
            mk = n == k
            byn[str(int(k))] = {"questions": int(mk.sum()), "levels": int(len(np.unique(y[mk]))),
                                "Q00": rb(sc0, y, mk), "Q02": rb(sc2, y, mk),
                                "Q02_question_mean": float(sc2[mk].mean()),
                                "exact_rb": rb((pred2 == y).astype(float), y, mk)}
        R["by_n"] = byn
        bins = {}
        for lo, hi in ((5, 7), (8, 12), (13, 20), (5, 5), (10, 10), (15, 15), (20, 20)):
            mk = (n >= lo) & (n <= hi)
            if mk.any():
                bins[f"{lo}-{hi}"] = {"questions": int(mk.sum()), "Q00": rb(sc0, y, mk), "Q02": rb(sc2, y, mk),
                                      "Q02_minus_Q00": diff_record(sc2, sc0, y, pb, mk) if setname == "mono" or
                                      (lo, hi) in ((5, 7), (8, 12), (13, 20)) else None,
                                      "Q02_per_level": per_level(sc2, y, mk),
                                      "Q02_pboot_se": pb.se(sc2, mk)}
        R["by_n_bin"] = bins
        if setname == "eval":
            bank = z["player_bank_size"][player]
            mk20 = bank == 20
            R["by_n_bank20_players_only"] = {
                "note": "players with a 20-game bank (508 of 636; every level present): k is uniform 5..20 for them, "
                        "so the n curve is not confounded with small-bank players",
                **{str(k): {"questions": int((mk20 & (n == k)).sum()), "Q00": rb(sc0, y, mk20 & (n == k)),
                            "Q02": rb(sc2, y, mk20 & (n == k))} for k in range(5, 21)}}
            ts = [rb(sc2, y, mk20 & (n == k)) for k in (5, 10, 15, 20)]
            ts0 = [rb(sc0, y, mk20 & (n == k)) for k in (5, 10, 15, 20)]
            R["test_shape_proxy"] = {
                "definition": "bank-20 players, n in {5,10,15,20}: mean of the four per-n rank-balanced means "
                              "(official questions are 100 each of 5/10/15/20 games, single colour)",
                "Q02": float(np.mean(ts)), "Q00": float(np.mean(ts0)), "Q02_per_n": dict(zip(("5", "10", "15", "20"), ts))}
            small = z["player_bank_size"] < 20
            R["small_bank_players"] = {
                "players": int(small.sum()),
                "by_level": dict(zip(LV, np.bincount(z["player_level"][small], minlength=L).tolist())),
                "Q02_rank_balanced_their_questions": rb(sc2, y, ~mk20),
                "Q02_rank_balanced_bank20_questions": rb(sc2, y, mk20)}

        # ---- by colour
        if setname == "eval":
            nb = z["eval_n_black"]
            frac = nb / n
            col = {}
            for name, mk in (("all_black", nb == n), ("all_white", nb == 0), ("mixed", (nb > 0) & (nb < n)),
                             ("black_share_lt_0.4", frac < 0.4), ("black_share_0.4_0.6", (frac >= 0.4) & (frac <= 0.6)),
                             ("black_share_gt_0.6", frac > 0.6)):
                col[name] = {"questions": int(mk.sum()), "levels": int(len(np.unique(y[mk]))),
                             "Q00": rb(sc0, y, mk), "Q02": rb(sc2, y, mk),
                             "mean_n": float(n[mk].mean()) if mk.any() else None}
            R["by_colour_mix"] = col
            gl = z["game_level"]
            gc = z["game_colour"]
            R["inner_games_black_share_by_level"] = {LV[i]: float((gc[gl == i] == "B").mean()) for i in range(L)}
        else:
            colour = z["mono_colour"]
            col = {}
            for cname in ("B", "W"):
                mk = colour == cname
                col[cname] = {"questions": int(mk.sum()), "Q00": rb(sc0, y, mk), "Q02": rb(sc2, y, mk),
                              "Q02_minus_Q00": diff_record(sc2, sc0, y, pb, mk),
                              "Q02_per_level": per_level(sc2, y, mk),
                              "Q02_by_k": {str(k): rb(sc2, y, mk & (n == k)) for k in (5, 10, 15, 20)},
                              "Q02_mean_signed_error_by_level": {LV[i]: float(d2[mk & (y == i)].mean())
                                                                 for i in range(L) if (mk & (y == i)).any()}}
            R["by_colour"] = col
            # test-shaped headroom cells: each level 1/10, each k 1/4 (official: 100 questions each of k = 5/10/15/20)
            cells = []
            for i in range(L):
                for k in (5, 10, 15, 20):
                    mk = (y == i) & (n == k)
                    if mk.any():
                        cells.append({"level": LV[i], "k": int(k), "questions": int(mk.sum()),
                                      "Q02": float(sc2[mk].mean()), "Q00": float(sc0[mk].mean()),
                                      "lost_test_shape": float((1 - sc2[mk].mean()) / (L * 4)),
                                      "lost_far_test_shape": float((np.abs(d2[mk]) >= 2).mean() / (L * 4))})
            ts = float(np.mean([rb(sc2, y, n == k) for k in (5, 10, 15, 20)]))
            R["test_shape_cells"] = {
                "definition": "single-colour questions; test-shape score = mean over k in 5/10/15/20 of the "
                              "rank-balanced mean at that k; lost_test_shape = (1 - cell mean) / 40",
                "Q02_test_shape": ts,
                "Q00_test_shape": float(np.mean([rb(sc0, y, n == k) for k in (5, 10, 15, 20)])),
                "lost_by_k": {str(k): float(sum(c["lost_test_shape"] for c in cells if c["k"] == k))
                              for k in (5, 10, 15, 20)},
                "lost_by_level": {lv: float(sum(c["lost_test_shape"] for c in cells if c["level"] == lv))
                                  for lv in LV},
                "cells": sorted(cells, key=lambda c: -c["lost_test_shape"])}

        # ---- calibration
        ES = U.expected_scores(P2)
        es = ES[np.arange(len(y)), pred2]
        cnt = np.bincount(y, minlength=L).astype(float)
        w = 1.0 / (L * cnt[y])
        edges = np.quantile(es, np.linspace(0, 1, 11))
        rel = []
        for b in range(10):
            mk = (es >= edges[b]) & ((es < edges[b + 1]) if b < 9 else (es <= edges[b + 1]))
            rel.append({"es_range": [float(edges[b]), float(edges[b + 1])], "questions": int(mk.sum()),
                        "mean_expected": float(es[mk].mean()), "mean_realized": float(sc2[mk].mean()),
                        "w_mean_expected": float(np.sum(w[mk] * es[mk]) / np.sum(w[mk])),
                        "w_mean_realized": float(np.sum(w[mk] * sc2[mk]) / np.sum(w[mk]))})
        bypred = {}
        for k in range(L):
            mk = pred2 == k
            if mk.any():
                bypred[LV[k]] = {"questions": int(mk.sum()), "mean_expected": float(es[mk].mean()),
                                 "mean_realized": float(sc2[mk].mean()),
                                 "w_mean_expected": float(np.sum(w[mk] * es[mk]) / np.sum(w[mk])),
                                 "w_mean_realized": float(np.sum(w[mk] * sc2[mk]) / np.sum(w[mk]))}
        byn_cal = {}
        for lo, hi in ((5, 7), (8, 12), (13, 20)):
            mk = (n >= lo) & (n <= hi)
            byn_cal[f"{lo}-{hi}"] = {"w_mean_expected": float(np.sum(w[mk] * es[mk]) / np.sum(w[mk])),
                                     "w_mean_realized": float(np.sum(w[mk] * sc2[mk]) / np.sum(w[mk]))}
        top1 = P2.max(1)
        edges1 = np.quantile(top1, np.linspace(0, 1, 11))
        rel1 = []
        for b in range(10):
            mk = (top1 >= edges1[b]) & ((top1 < edges1[b + 1]) if b < 9 else (top1 <= edges1[b + 1]))
            rel1.append({"p_top1_range": [float(edges1[b]), float(edges1[b + 1])],
                         "w_mean_p_top1": float(np.sum(w[mk] * top1[mk]) / np.sum(w[mk])),
                         "w_hit_rate": float(np.sum(w[mk] * (P2[mk].argmax(1) == y[mk])) / np.sum(w[mk]))})
        # level prior implied by the OOF probabilities (level-weighted average P) vs the balanced 0.1
        prior = (w[:, None] * P2).sum(0)
        R["calibration_Q02"] = {
            "note": "w_* = rank-balanced weights (each level 1/10), the weighting the calibration fit uses",
            "expected_score_of_choice_vs_realized_by_decile": rel, "by_predicted_level": bypred,
            "by_n_bin": byn_cal, "top1_probability_reliability": rel1,
            "implied_level_prior_rank_balanced": dict(zip(LV, [float(x) for x in prior])),
            "mean_expected_level_minus_true_by_true_level": {
                LV[i]: float(((P2[y == i] * np.arange(L)).sum(1) - i).mean()) for i in range(L)}}

        # ---- decision-rule headroom
        pa = P2.argmax(1)
        order = np.argsort(-ES, axis=1, kind="stable")
        rank_true = np.argmax(order == y[:, None], axis=1)        # 0 = rule-B choice is the truth
        orc = {}
        for k in (2, 3):
            hit = rank_true < k
            pred_o = np.where(hit, y, pred2)
            orc[f"oracle_top{k}_by_expected_score"] = rb(U.score(pred_o, y), y)
        orderp = np.argsort(-P2, axis=1, kind="stable")
        hitp = (orderp[:, :2] == y[:, None]).any(1)
        orc["oracle_top2_by_probability"] = rb(U.score(np.where(hitp, y, pred2), y), y)
        best_adj = np.where(np.abs(pred2 - y) == 1, y, pred2)
        orc["oracle_fix_every_adjacent_error"] = rb(U.score(best_adj, y), y)
        far_to_adj = np.where(np.abs(pred2 - y) >= 2, y + np.sign(pred2 - y), pred2)
        orc["oracle_turn_every_far_error_into_adjacent"] = rb(U.score(far_to_adj, y), y)
        R["decision_headroom"] = {
            "rule_B": rb(sc2, y), "rule_A_argmax": rb(U.score(pa, y), y),
            "rule_A_minus_B": diff_record(U.score(pa, y), sc2, y, pb),
            "questions_A_differs_from_B": int((pa != pred2).sum()),
            "true_level_position_in_rule_B_ranking_rb": {str(k): float(np.sum(w * (rank_true == k)))
                                                         for k in range(4)},
            **orc}

        # ---- cross-fitted post-hoc decision adjustments (fit on 4 player folds, apply to the 5th)
        if setname == "eval":
            R["crossfit_adjustments"] = crossfit(P2, y, fold, sc2, pb)
            R["player_level"] = player_analysis(z, sc2, pred2, y, player, n)
        else:
            R["player_level"] = player_analysis(z, sc2, pred2, y, player, n, mono=True)
        rep[setname] = R
        print(f"[{setname}] done ({time.time() - T0:.0f}s)", flush=True)

    with open(os.path.join(U.HERE, "u1_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(f"wrote u1_analysis.json ({time.time() - T0:.0f}s)")


def adj_P(P, tau=1.0, bias=None):
    lp = np.log(np.maximum(P, 1e-300)) * tau
    if bias is not None:
        lp = lp + bias
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


def crossfit(P, y, fold, sc_ref, pb):
    """Post-hoc changes of the decision only (P from Q02 fixed): (1) temperature tau on log P, (2) per-level bias on
    log P (coordinate ascent, 3 passes, steps of 0.05 in [-1, 1]), (3) neighbour weight c in rule B. Each is chosen
    on the eval questions of 4 player folds (rank-balanced score) and applied to the 5th fold."""
    out = {}
    taus = np.round(np.arange(0.5, 2.01, 0.05), 3)
    cs = np.round(np.arange(0.0, 0.81, 0.02), 3)
    steps = np.round(np.arange(-1.0, 1.0001, 0.05), 3)
    pred_t = np.empty(len(y), dtype=np.int64)
    pred_b = np.empty(len(y), dtype=np.int64)
    pred_c = np.empty(len(y), dtype=np.int64)
    chosen = {"tau": [], "bias": [], "c": []}
    for f in range(5):
        tr, te = fold != f, fold == f
        Pt, yt = P[tr], y[tr]
        # (1) temperature
        best = max(taus, key=lambda t: rb(U.score(U.rule_b_np(adj_P(Pt, t)), yt), yt))
        chosen["tau"].append(float(best))
        pred_t[te] = U.rule_b_np(adj_P(P[te], best))
        # (3) neighbour weight
        bc = max(cs, key=lambda c: rb(U.score(decide_b(Pt, c), yt), yt))
        chosen["c"].append(float(bc))
        pred_c[te] = decide_b(P[te], bc)
        # (2) per-level bias
        lpt = np.log(np.maximum(Pt, 1e-300))
        bias = np.zeros(L)

        def val(b):
            return rb(U.score(U.rule_b_np(adj_P(Pt, 1.0, b)), yt), yt)

        cur = val(bias)
        for _ in range(3):
            for k in range(1, L):     # bias[0] fixed at 0 (rule B is invariant to a common shift)
                base = bias[k]
                vals = []
                for s in steps:
                    bb = bias.copy()
                    bb[k] = s
                    vals.append(val(bb))
                j = int(np.argmax(vals))
                if vals[j] > cur + 1e-12:
                    bias[k], cur = steps[j], vals[j]
                else:
                    bias[k] = base
        del lpt
        chosen["bias"].append([float(x) for x in bias])
        pred_b[te] = U.rule_b_np(adj_P(P[te], 1.0, bias))
        print(f"  crossfit fold {f}: tau {best}, c {bc}, bias {np.round(bias, 2).tolist()}", flush=True)
    for name, pr in (("temperature", pred_t), ("per_level_bias", pred_b), ("neighbour_weight_c", pred_c)):
        sc = U.score(pr, y)
        out[name] = {"rank_balanced": rb(sc, y), "minus_Q02": diff_record(sc, sc_ref, y, pb)}
    out["chosen_per_fold"] = chosen
    out["note"] = ("decision-only changes on Q02's OOF probabilities, cross-fitted over the 5 player folds (choose on 4, "
                   "apply to the 5th); exploration after seeing inner, design-stage only")
    return out


def player_analysis(z, sc, pred, y, player, n, mono=False):
    """Is the loss concentrated in some players (systematic) or spread (sampling noise of few games)?"""
    npl = 636
    cntq = np.bincount(player, minlength=npl).astype(float)
    have = cntq > 0
    msc = np.bincount(player, weights=sc, minlength=npl)[have] / cntq[have]
    mse = np.bincount(player, weights=(pred - y), minlength=npl)[have] / cntq[have]
    lvl = z["player_level"][have]
    # rank-balanced share of loss: each level 1/10, each player within a level by its question share
    wq = 1.0 / (L * np.bincount(y, minlength=L)[y])
    lossp = np.bincount(player, weights=wq * (1 - sc), minlength=npl)[have]
    tot = lossp.sum()
    srt = np.argsort(-lossp)
    cum = np.cumsum(lossp[srt]) / tot
    out = {"players": int(have.sum()),
           "player_mean_score_quantiles": {q: float(np.quantile(msc, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "players_mean_score_below_0.368": int((msc < U.E1).sum()),
           "players_mean_score_below_0.2": int((msc < 0.2).sum()),
           "share_of_loss_from_worst_10pct_players": float(cum[int(round(0.1 * len(cum))) - 1]),
           "share_of_loss_from_worst_25pct_players": float(cum[int(round(0.25 * len(cum))) - 1]),
           "players_systematic_bias_ge_1_level": int((np.abs(mse) >= 1).sum()),
           "players_systematic_bias_ge_0.5_level": int((np.abs(mse) >= 0.5).sum()),
           "by_level": {}}
    # variance decomposition of the per-question score: between-player vs within-player
    gm = sc.mean()
    pm = np.bincount(player, weights=sc, minlength=npl) / np.maximum(cntq, 1)
    between = float(np.sum(cntq * (pm - gm) ** 2) / len(sc))
    total = float(np.var(sc))
    out["score_variance_between_players_share"] = between / total
    for i in range(L):
        mk = lvl == i
        out["by_level"][LV[i]] = {"players": int(mk.sum()),
                                  "player_mean_score_median": float(np.median(msc[mk])),
                                  "players_mean_score_below_0.368": int((msc[mk] < U.E1).sum()),
                                  "player_mean_signed_error_quantiles": [float(np.quantile(mse[mk], q))
                                                                         for q in (0.1, 0.5, 0.9)],
                                  "players_bias_le_-1": int((mse[mk] <= -1).sum()),
                                  "players_bias_ge_+1": int((mse[mk] >= 1).sum())}
    if not mono:
        # same player, more games: bank-20 players, n 5-7 vs 18-20 (mean per player)
        bank = z["player_bank_size"][player]
        a = (bank == 20) & (n <= 7)
        b = (bank == 20) & (n >= 18)
        ca = np.bincount(player[a], minlength=npl)
        cb = np.bincount(player[b], minlength=npl)
        ok = (ca > 0) & (cb > 0)
        sa = np.bincount(player[a], weights=sc[a], minlength=npl)[ok] / ca[ok]
        sb = np.bincount(player[b], weights=sc[b], minlength=npl)[ok] / cb[ok]
        out["bank20_same_player_n5to7_vs_n18to20"] = {
            "players": int(ok.sum()), "mean_score_n5to7": float(sa.mean()), "mean_score_n18to20": float(sb.mean()),
            "players_still_below_0.368_at_n18to20": int((sb < U.E1).sum()),
            "players_below_0.368_at_both": int(((sa < U.E1) & (sb < U.E1)).sum())}
    return out


if __name__ == "__main__":
    main()
