"""Task B part 1: where R20 loses points on inner (OOF), without any validation data.

Run from anywhere with the repo .venv python. Writes r20_diag.json and r20_oof.npz next to this file.
"""
import json
import time

import numpy as np
from scipy import optimize

from common import (B, DAN, E1, KYU, L, LEVELS, OUT, Boot, F, exp_scores, gen_fit, gen_predict, groups_of,
                    load_all, lsm, mix_source, oof_gen, per_level, r4, rank_balanced, rule_b, score, sym_source)

t0 = time.time()
D = load_all()
Xf, Xe, Sf, Se = D["X_fit"], D["X_eval"], D["S_fit"], D["S_eval"]
keys, dumps, players = D["keys"], D["dumps"], D["players"]
y, n, fold, pl = Xe.y, Xe.n, Xe.fold, Xe.player
out = {"what": "R20 error analysis on inner pseudo-questions (5-fold OOF, as tune_report)", "inputs": {}}
print(f"loaded {time.time() - t0:.0f}s: fit {len(Xf)}, eval {len(Xe)}", flush=True)

# ------------------------------------------------------------------ reproduce R20 / R10 exactly
lp5 = F.source_logprobs("C5", dumps)
assert np.allclose(lp5, sym_source(dumps["p02e9_d4"], range(8)))
for s in ("C5",):
    Xf.m[(s, "log")] = F.pooled(Sf, lp5, "log")
    Xe.m[(s, "log")] = F.pooled(Se, lp5, "log")


def oof_tool(source, pool, opening):
    P = np.zeros((len(Xe), L))
    for f in range(5):
        ev, tr = Xe.fold == f, Xf.fold != f
        th, _ = F.fit(Xf.subset(tr), source, pool, opening, B)
        P[ev] = F.predict(th, Xe.subset(ev), source, pool, opening, B)
    return P


P20 = oof_tool("C5", "log", True)
P10 = oof_tool("C5", "log", False)
pred20, pred10 = rule_b(P20), rule_b(P10)
sc20, sc10 = score(pred20, y), score(pred10, y)
tr = json.load(open(F.os.path.join(F.REPO, "results", "prereg02", "rank", "tune_report.json"), encoding="utf-8"))
reg = {c["id"]: c["oof"] for c in tr["combinations"] if c["status"] == "ok"}
out["reproduction"] = {"R20_rank_balanced": rank_balanced(sc20, y), "R20_tune_report": reg["R20"]["rank_balanced"],
                       "R10_rank_balanced": rank_balanced(sc10, y), "R10_tune_report": reg["R10"]["rank_balanced"],
                       "R20_exact_match": abs(rank_balanced(sc20, y) - reg["R20"]["rank_balanced"]) < 1e-12,
                       "R10_exact_match": abs(rank_balanced(sc10, y) - reg["R10"]["rank_balanced"]) < 1e-12}
print("repro", out["reproduction"], flush=True)
np.savez_compressed(F.os.path.join(OUT, "r20_oof.npz"), P20=P20, P10=P10, y=y, n=n, fold=fold, player=pl)
boot = Boot(pl, y, len(players))
fr = json.load(open(F.os.path.join(F.REPO, "results", "prereg02", "rank", "frozen_config.json"), encoding="utf-8"))
th_frozen = np.array([fr["params"]["a"], fr["params"]["gamma"], fr["params"]["w"]] + fr["params"]["beta"])

# ------------------------------------------------------------------ 1. confusion, distance, kyu/dan
d = pred20 - y
C = np.zeros((L, L), dtype=np.int64)
np.add.at(C, (y, pred20), 1)
rown = C / C.sum(1, keepdims=True)
lvl = {}
for i in range(L):
    mk = y == i
    ad = np.abs(d[mk])
    lvl[LEVELS[i]] = {
        "score": sc20[mk].mean(), "exact": np.mean(ad == 0), "adjacent": np.mean(ad == 1), "off2": np.mean(ad == 2),
        "off3plus": np.mean(ad >= 3), "off2plus": np.mean(ad >= 2),
        "lost_points": 1 - sc20[mk].mean(), "lost_from_adjacent": (1 - E1) * np.mean(ad == 1),
        "lost_from_off2plus": np.mean(ad >= 2),
        "share_of_lost_from_off2plus": np.mean(ad >= 2) / max(1e-12, 1 - sc20[mk].mean()),
        "pred_higher": np.mean(d[mk] > 0), "pred_lower": np.mean(d[mk] < 0), "mean_signed_error": d[mk].mean(),
        "cross_kyu_dan": np.mean(np.isin(pred20[mk], DAN) if i in KYU else np.isin(pred20[mk], KYU))}
kyu_m, dan_m = np.isin(y, KYU), np.isin(y, DAN)
lb_off2 = np.mean([np.mean(np.abs(d[y == i]) >= 2) for i in range(L)])
out["confusion"] = {"rows_true_cols_pred_counts": C.tolist(),
                    "row_normalised": np.round(rown, 4).tolist(), "levels": LEVELS,
                    "per_level": r4(lvl),
                    "level_balanced": r4({"exact": np.mean([np.mean(d[y == i] == 0) for i in range(L)]),
                                          "adjacent": np.mean([np.mean(np.abs(d[y == i]) == 1) for i in range(L)]),
                                          "off2plus": lb_off2,
                                          "lost_total": 1 - rank_balanced(sc20, y),
                                          "lost_from_adjacent": np.mean([(1 - E1) * np.mean(np.abs(d[y == i]) == 1) for i in range(L)]),
                                          "lost_from_off2plus": lb_off2}),
                    "kyu_vs_dan": r4({"kyu_rank_balanced": np.mean([sc20[y == i].mean() for i in KYU]),
                                      "dan_rank_balanced": np.mean([sc20[y == i].mean() for i in DAN]),
                                      "kyu_off2plus": np.mean([np.mean(np.abs(d[y == i]) >= 2) for i in KYU]),
                                      "dan_off2plus": np.mean([np.mean(np.abs(d[y == i]) >= 2) for i in DAN]),
                                      "kyu_true_pred_dan": np.mean(np.isin(pred20[kyu_m], DAN)),
                                      "dan_true_pred_kyu": np.mean(np.isin(pred20[dan_m], KYU)),
                                      "A_true_pred_dan": np.mean(np.isin(pred20[y == 3], DAN)),
                                      "1D_true_pred_kyu": np.mean(np.isin(pred20[y == 4], KYU)),
                                      "two_group_accuracy_level_balanced": np.mean(
                                          [np.mean(np.isin(pred20[y == i], KYU if i in KYU else DAN)) for i in range(L)])})}
print(f"1 done {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ 2. calibration
S = exp_scores(P20)
Emax = S.max(1)
Epred = S[np.arange(len(y)), pred20]
assert np.allclose(Emax, Epred)
nll = -np.log(np.maximum(P20[np.arange(len(y)), y], 1e-300))


def lvl_bal(v, mask=None):
    mask = np.ones(len(y), bool) if mask is None else mask
    return float(np.mean([v[mask & (y == i)].mean() for i in range(L) if (mask & (y == i)).any()]))


cal = {"model_expected_score_question_mean": Emax.mean(), "realised_question_mean": sc20.mean(),
       "model_expected_level_balanced_by_true_level": lvl_bal(Emax), "realised_rank_balanced": rank_balanced(sc20, y),
       "nll_level_balanced": lvl_bal(nll), "nll_uniform": float(np.log(10))}
# by predicted level (unbiased reliability grouping) and by true level
cal["by_pred_level"] = {LEVELS[k]: {"n": int((pred20 == k).sum()), "model_expected": Emax[pred20 == k].mean(),
                                    "realised": sc20[pred20 == k].mean(),
                                    "p_pred_level": P20[pred20 == k, k].mean(),
                                    "exact_rate": np.mean(y[pred20 == k] == k)} for k in range(L)}
cal["by_true_level"] = {LEVELS[i]: {"model_expected": Emax[y == i].mean(), "realised": sc20[y == i].mean(),
                                    "mean_p_true": P20[y == i, i].mean(), "mean_p_max": P20[y == i].max(1).mean(),
                                    "nll": nll[y == i].mean(),
                                    "mean_pred_index_minus_true": (pred20[y == i] - i).mean(),
                                    "mean_expected_index_minus_true": (P20[y == i] @ np.arange(L) - i).mean()}
                        for i in range(L)}
# calibration-in-the-large per level, level-balanced weighting (target 0.1 each)
wq = F.level_weights(y)
cal["mean_prob_level_weighted"] = {LEVELS[k]: float(np.sum(wq * P20[:, k])) for k in range(L)}
# reliability of the rule-B expected score (10 equal-count bins)
qs = np.quantile(Emax, np.linspace(0, 1, 11))
bins = np.clip(np.searchsorted(qs, Emax, side="right") - 1, 0, 9)
rel = []
for b in range(10):
    mk = bins == b
    rel.append({"bin": b, "n": int(mk.sum()), "model_expected": Emax[mk].mean(), "realised": sc20[mk].mean()})
cal["reliability_expected_score"] = rel
cal["ece_expected_score"] = float(sum(r["n"] * abs(r["model_expected"] - r["realised"]) for r in rel) / len(y))
# top-1 (argmax) calibration
am = P20.argmax(1)
pmax = P20.max(1)
qb = np.quantile(pmax, np.linspace(0, 1, 11))
bb = np.clip(np.searchsorted(qb, pmax, side="right") - 1, 0, 9)
cal["reliability_top1"] = [{"bin": b, "mean_pmax": pmax[bb == b].mean(), "hit": np.mean(am[bb == b] == y[bb == b])}
                           for b in range(10)]
cal["ece_top1"] = float(sum(np.sum(bb == b) * abs(pmax[bb == b].mean() - np.mean(am[bb == b] == y[bb == b]))
                            for b in range(10)) / len(y))
# class-wise reliability (level-weighted): for each class k, bins of p_k
cw = {}
edges = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0001]
for k in range(L):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (P20[:, k] >= lo) & (P20[:, k] < hi)
        if mk.sum() < 50:
            continue
        rows.append({"p_range": f"{lo}-{min(hi, 1)}", "n": int(mk.sum()),
                     "mean_p": float(np.sum(wq[mk] * P20[mk, k]) / wq[mk].sum()),
                     "freq": float(np.sum(wq[mk] * (y[mk] == k)) / wq[mk].sum())})
    cw[LEVELS[k]] = rows
cal["classwise_reliability_level_weighted"] = cw


# by n (bins and single n) + per-bin optimal temperature multiplier on log P20
def temp_mult(mask):
    lp = np.log(np.maximum(P20[mask], 1e-300))
    yy, ww = y[mask], F.level_weights(y[mask])

    def f(c):
        return -np.sum(ww * lsm(c * lp)[np.arange(len(yy)), yy])
    r = optimize.minimize_scalar(f, bounds=(0.2, 5), method="bounded")
    return float(r.x)


byn = {}
for lo, hi in ((5, 7), (8, 12), (13, 20), (5, 5), (10, 10), (15, 15), (20, 20)):
    mk = (n >= lo) & (n <= hi)
    byn[f"{lo}-{hi}"] = {"questions": int(mk.sum()), "rank_balanced": lvl_bal(sc20, mk), "question_mean": sc20[mk].mean(),
                         "model_expected": Emax[mk].mean(), "mean_pmax": pmax[mk].mean(), "nll_lb": lvl_bal(nll, mk),
                         "off2plus_lb": lvl_bal((np.abs(d) >= 2).astype(float), mk),
                         "optimal_temperature_multiplier": temp_mult(mk)}
cal["by_n"] = byn
cal["per_n_rank_balanced"] = {int(k): lvl_bal(sc20, n == k) for k in range(5, 21)}
out["calibration"] = r4(cal)
print(f"2 done {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ 3. per-game vs per-question
Ge = groups_of(Se)
a_, g_, w_ = th_frozen[0], th_frozen[1], th_frozen[2]
b_ = B @ th_frozen[3:]
opg = np.where(D["op_has"][:, None], D["op_o"], 0.0)
t1 = np.exp(a_) * (1 / 12.0) ** g_
Pg = F.softmax(t1 * (lp5 + w_ * opg + b_))  # single-game question with the frozen R20 parameters
pred_g = rule_b(Pg)
lab = keys["label"]
sc_g = score(pred_g, lab)
eidx_g = np.exp(lp5) @ np.arange(L)
bank_rows = np.unique(np.concatenate(Ge))
pg = {}
for i in range(L):
    mk = lab == i
    mb = np.zeros(len(lab), bool)
    mb[bank_rows] = True
    mb &= mk
    pg[LEVELS[i]] = {"games": int(mk.sum()), "acc_argmax_C5": np.mean(lp5[mk].argmax(1) == i),
                     "within1_argmax_C5": np.mean(np.abs(lp5[mk].argmax(1) - i) <= 1),
                     "single_game_ruleB_score": sc_g[mk].mean(), "single_game_ruleB_score_bank": sc_g[mb].mean(),
                     "acc_by_colour": {c: float(np.mean(lp5[mk & (np.asarray(keys["color"]) == c)].argmax(1) == i))
                                       for c in ("B", "W")}}
per_game = {"per_level": pg, "single_game_rank_balanced": float(np.mean([sc_g[lab == i].mean() for i in range(L)])),
            "acc_argmax_level_balanced": float(np.mean([np.mean(lp5[lab == i].argmax(1) == i) for i in range(L)]))}
# agreement of per-game decisions with the question decision
agree = np.array([np.mean(pred_g[g] == pred20[q]) for q, g in enumerate(Ge)])
within1 = np.array([np.mean(np.abs(pred_g[g] - pred20[q]) <= 1) for q, g in enumerate(Ge)])
spread = np.array([eidx_g[g].std() for g in Ge])
qa = np.quantile(agree, [0, 1 / 3, 2 / 3, 1])
ab = np.clip(np.searchsorted(qa, agree, side="right") - 1, 0, 2)
per_game["agreement"] = {"mean_frac_games_same_decision": agree.mean(), "mean_frac_games_within1": within1.mean(),
                         "by_agreement_tercile": [{"tercile": b, "agree_range": [float(agree[ab == b].min()), float(agree[ab == b].max())],
                                                   "question_mean": sc20[ab == b].mean(),
                                                   "off2plus": np.mean(np.abs(d[ab == b]) >= 2)} for b in range(3)],
                         "spearman_spread_vs_abs_error": float(np.corrcoef(np.argsort(np.argsort(spread)),
                                                                           np.argsort(np.argsort(np.abs(d))))[0, 1])}
# alternative poolers
vote = np.array([np.bincount(pred_g[g], minlength=L).argmax() for g in Ge])
meanidx = np.array([int(np.clip(np.round(eidx_g[g].mean()), 0, 9)) for g in Ge])
prob_pool = F.pooled(Se, lp5, "prob")
per_game["poolers_rank_balanced"] = {"R20_log_pool_calibrated": rank_balanced(sc20, y),
                                     "majority_vote_of_single_game_ruleB": rank_balanced(score(vote, y), y),
                                     "round_mean_expected_index_uncal": rank_balanced(score(meanidx, y), y)}
out["per_game"] = r4(per_game)

# player-level asymptote: all of a player's inner games (not only the 20-game bank)
pid = np.asarray(keys["player_id"])
upl, inv = np.unique(pid, return_inverse=True)
ng = np.bincount(inv)
mlog = np.zeros((len(upl), L))
np.add.at(mlog, inv, lp5)
mlog /= ng[:, None]
mop = np.zeros((len(upl), L))
np.add.at(mop, inv, opg)
nop = np.bincount(inv, weights=D["op_has"].astype(float))
mop /= np.maximum(nop, 1)[:, None]
ylab = np.zeros(len(upl), int)
ylab[inv] = lab
asym = {}
for nmin in (5, 20, 50, 100):
    mk = ng >= nmin
    for nt in (20,):
        z = np.exp(a_) * (nt / 12.0) ** g_ * (mlog[mk] + w_ * mop[mk] + b_)
        pr = rule_b(F.softmax(z))
        s_ = score(pr, ylab[mk])
        asym[f"players_ge{nmin}_games"] = {"players": int(mk.sum()),
                                            "rank_balanced_all_games_R20params_n20": rank_balanced(s_, ylab[mk]),
                                            "exact_level_balanced": float(np.mean([np.mean(pr[ylab[mk] == i] == i) for i in range(L) if (ylab[mk] == i).any()])),
                                            "off2plus_level_balanced": float(np.mean([np.mean(np.abs(pr[ylab[mk] == i] - i) >= 2) for i in range(L) if (ylab[mk] == i).any()])),
                                            "per_level": per_level(s_, ylab[mk]),
                                            "mean_games_used": float(ng[mk].mean())}
asym["R20_oof_questions_with_n20"] = {"rank_balanced": lvl_bal(sc20, n == 20), "questions": int((n == 20).sum())}
# variance decomposition of the per-game expected level index (players with >= 20 inner games)
vd = {}
for i in range(L):
    ps = np.nonzero((ylab == i) & (ng >= 20))[0]
    means = np.array([eidx_g[inv == p].mean() for p in ps])
    within = np.array([eidx_g[inv == p].var() for p in ps])
    vd[LEVELS[i]] = {"players": int(len(ps)), "player_mean_of_E_index_mean": means.mean(), "between_player_sd": means.std(),
                     "within_player_sd": float(np.sqrt(within.mean())),
                     "icc_single_game": float(means.var() / (means.var() + within.mean()))}
asym["expected_index_variance_decomposition"] = vd
out["player_asymptote"] = r4(asym)
print(f"3 done {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ 4. symmetry / epoch ensembles (log pool + opening, tune protocol)
Ef_op, Ee_op = [Xf.mop], [Xe.mop]
srcs = {
    "p02e9_s0 (C4)": sym_source(dumps["p02e9_d4"], [0]),
    "p02e9_s0,s4 (flip pair)": sym_source(dumps["p02e9_d4"], [0, 4]),
    "p02e9_rot4 s0-3": sym_source(dumps["p02e9_d4"], [0, 1, 2, 3]),
    "p02e9_8sym (C5=R20)": lp5,
    "p01e5_8sym (C2)": sym_source(dumps["p01e5_d4"], range(8)),
}
lp_p01e5 = srcs["p01e5_8sym (C2)"]
srcs["mix p02e9_8 + p01e5_8"] = mix_source([lp5, lp_p01e5])
srcs["mix p02e9_8 + p01e3,e4,e5 s0"] = mix_source([lp5, sym_source(dumps["p01e3"], [0]), sym_source(dumps["p01e4"], [0]),
                                                   sym_source(dumps["p01e5_d4"], [0])])
srcs["mix p02e9_8 x2 + p01e5_8"] = mix_source([lp5, lp5, lp_p01e5])
ens = {}
for name, lp in srcs.items():
    mf, me = F.pooled(Sf, lp, "log"), F.pooled(Se, lp, "log")
    P, _ = oof_gen(Xf, Xe, mf, Ef_op, me, Ee_op)
    s_ = score(rule_b(P), y)
    rec = {"rank_balanced": rank_balanced(s_, y), "question_mean": s_.mean(),
           "per_game_acc_level_balanced": float(np.mean([np.mean(lp[lab == i].argmax(1) == i) for i in range(L)])),
           "per_game_nll_level_balanced": float(np.mean([-lp[lab == i, i].mean() for i in range(L)]))}
    if name != "p02e9_8sym (C5=R20)":
        rec["diff_vs_R20"] = rank_balanced(s_, y) - rank_balanced(sc20, y)
        rec.update(boot.diff(s_, sc20))
        rec.pop("per_level_ci95")
    else:
        rec["same_as_tool_R20"] = bool(np.array_equal(rule_b(P), pred20))
    ens[name] = rec
    print(name, round(rec["rank_balanced"], 4), flush=True)
# two separate terms: m_p02 + w1 m_p01 (+ opening)
mf2, me2 = F.pooled(Sf, lp_p01e5, "log"), F.pooled(Se, lp_p01e5, "log")
P, ths = oof_gen(Xf, Xe, Xf.m[("C5", "log")], [Xf.mop, mf2], Xe.m[("C5", "log")], [Xe.mop, me2])
s_ = score(rule_b(P), y)
ens["two-term m_p02e9_8 + w1 m_p01e5_8"] = {"rank_balanced": rank_balanced(s_, y), "question_mean": s_.mean(),
                                            "diff_vs_R20": rank_balanced(s_, y) - rank_balanced(sc20, y),
                                            "w_open_by_fold": [float(t[2]) for t in ths], "w_p01_by_fold": [float(t[3]) for t in ths]}
ens["two-term m_p02e9_8 + w1 m_p01e5_8"].update({k: v for k, v in boot.diff(s_, sc20).items() if k != "per_level_ci95"})
ens["per_symmetry_per_game_acc_p02e9"] = [float(np.mean(dumps["p02e9_d4"][:, s, :].argmax(1) == lab)) for s in range(8)]
ens["per_epoch_per_game_acc_s0"] = {nm: float(np.mean(dumps[nm][:, 0, :].argmax(1) == lab))
                                    for nm in ("p01e1", "p01e2", "p01e3", "p01e4", "p01e5_d4", "p02e9_d4")}
out["ensembles"] = r4(ens)
print(f"4 done {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ 5. opening term by level
opn = {"R10_vs_R20_rank_balanced": [rank_balanced(sc10, y), rank_balanced(sc20, y)],
       "per_level_R10": per_level(sc10, y), "per_level_R20": per_level(sc20, y),
       "per_level_diff_R20_minus_R10": {k: per_level(sc20, y)[k] - per_level(sc10, y)[k] for k in LEVELS}}
opn.update({"bootstrap_R20_minus_R10": boot.diff(sc20, sc10)})
# ablation: R20 frozen-fold params with w forced to 0 at inference (no refit)
P_w0 = np.zeros_like(P20)
for f in range(5):
    ev, trm = Xe.fold == f, Xf.fold != f
    th, _ = F.fit(Xf.subset(trm), "C5", "log", True, B)
    th = th.copy()
    th[2] = 0.0
    P_w0[ev] = F.predict(th, Xe.subset(ev), "C5", "log", True, B)
s_w0 = score(rule_b(P_w0), y)
opn["R20_with_w0_at_inference_rank_balanced"] = rank_balanced(s_w0, y)
# opening-only (calibrated; the opening mean as the only source)
zf, ze = np.zeros_like(Xf.mop), np.zeros_like(Xe.mop)
Pop, _ = oof_gen(Xf, Xe, Xf.mop, [], Xe.mop, [])
s_op = score(rule_b(Pop), y)
opn["opening_only_calibrated"] = {"rank_balanced": rank_balanced(s_op, y), "per_level": per_level(s_op, y)}
Pcnn, _ = oof_gen(Xf, Xe, Xf.m[("C5", "log")], [], Xe.m[("C5", "log")], [])
opn["check_R10_via_gen"] = rank_balanced(score(rule_b(Pcnn), y), y)
opn["coverage_by_level"] = {LEVELS[i]: {"frac_games_with_opening_score": float(D["op_has"][lab == i].mean()),
                                        "mean_terms": float(D["op_nterms"][lab == i].mean())} for i in range(L)}
opn["per_game_opening_acc_level_balanced"] = float(np.mean([np.mean(D["op_o"][lab == i].argmax(1) == i) for i in range(L)]))
out["opening"] = r4(opn)
print(f"5 done {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ 6. upper bounds for rule B on inner
ub = {"R20_oof": rank_balanced(sc20, y),
      "U1_model_self_expected_question_mean": Emax.mean(), "U1_realised_question_mean": sc20.mean()}
# U2: R20 family fitted in-sample on all eval questions
m_e = Xe.m[("C5", "log")]
th_in = gen_fit(m_e, [Xe.mop], n, y)
P_in = gen_predict(th_in, m_e, [Xe.mop], n)
ub["U2_R20_family_in_sample_on_eval"] = rank_balanced(score(rule_b(P_in), y), y)
P_ev, _ = oof_gen(Xf, Xe, None, None, m_e, [Xe.mop], fit_on="eval")
ub["R20_family_oof_fit_on_eval_rounds_other_folds"] = rank_balanced(score(rule_b(P_ev), y), y)


# U3: full 10x10 matrix recalibration z = (n/12)^g (A m + w m_op + b)
def full_obj(th, m, mo, nn, yy, ww, lam):
    g, w = th[0], th[1]
    A = th[2:2 + L * L].reshape(L, L)
    beta = th[2 + L * L:]
    t = np.power(nn / 12.0, g)
    u = m @ A.T + w * mo + B @ beta
    z = t[:, None] * u
    ls = lsm(z)
    Q = len(yy)
    loss = -np.sum(ww * ls[np.arange(Q), yy]) + lam * np.sum((A - np.eye(L) * np.exp(1.16)) ** 2) + F.PENALTY * beta @ beta
    G = np.exp(ls)
    G[np.arange(Q), yy] -= 1.0
    G *= ww[:, None]
    gz = np.sum(G * z, 1)
    tG = t[:, None] * G
    gA = tG.T @ m + 2 * lam * (A - np.eye(L) * np.exp(1.16))
    gw = np.sum(tG * mo)
    gb = B.T @ tG.sum(0) + 2 * F.PENALTY * beta
    return loss, np.concatenate([[np.sum(gz * np.log(nn / 12.0)), gw], gA.ravel(), gb])


def full_fit(m, mo, nn, yy, lam=1e-4):
    x0 = np.concatenate([[0.44, 0.35], (np.eye(L) * np.exp(1.16)).ravel(), np.zeros(L - 1)])
    r = optimize.minimize(full_obj, x0, args=(m, mo, nn, yy, F.level_weights(yy), lam), jac=True, method="L-BFGS-B",
                          options={"maxiter": 2000})
    return r.x


def full_pred(th, m, mo, nn):
    A = th[2:2 + L * L].reshape(L, L)
    return F.softmax(np.power(nn / 12.0, th[0])[:, None] * (m @ A.T + th[1] * mo + B @ th[2 + L * L:]))


thA = full_fit(m_e, Xe.mop, n, y)
ub["U3_full_matrix_in_sample"] = rank_balanced(score(rule_b(full_pred(thA, m_e, Xe.mop, n)), y), y)
PA = np.zeros_like(P20)
for f in range(5):
    ev = fold == f
    th = full_fit(m_e[~ev], Xe.mop[~ev], n[~ev], y[~ev])
    PA[ev] = full_pred(th, m_e[ev], Xe.mop[ev], n[ev])
sA = score(rule_b(PA), y)
ub["U3_full_matrix_oof_fit_on_eval_rounds_other_folds"] = rank_balanced(sA, y)
ub["U3_oof_vs_R20_bootstrap"] = {k: v for k, v in boot.diff(sA, sc20).items() if k != "per_level_ci95"}
PAf = np.zeros_like(P20)
m_f = Xf.m[("C5", "log")]
for f in range(5):
    ev, trm = fold == f, Xf.fold != f
    th = full_fit(m_f[trm], Xf.mop[trm], Xf.n[trm], Xf.y[trm])
    PAf[ev] = full_pred(th, m_e[ev], Xe.mop[ev], n[ev])
ub["U3_full_matrix_oof_fit_on_fit_rounds (tune protocol)"] = rank_balanced(score(rule_b(PAf), y), y)


# U4: decision offsets maximising rank-balanced directly (coordinate search on S + delta)
def rb_offsets(Sx, yy, delta):
    return rank_balanced(score(np.argmax(Sx + delta, 1), yy), yy)


def search_offsets(Sx, yy, passes=3, grid=np.arange(-0.15, 0.1501, 0.005)):
    delta = np.zeros(L)
    best = rb_offsets(Sx, yy, delta)
    for _ in range(passes):
        for k in range(L):
            for v in grid:
                dd = delta.copy()
                dd[k] = v
                s = rb_offsets(Sx, yy, dd)
                if s > best + 1e-12:
                    best, delta = s, dd
    return delta, best


dl, best_in = search_offsets(S, y)
ub["U4_ruleB_offsets_in_sample"] = best_in
ub["U4_offsets"] = dl.tolist()
PD = np.zeros(len(y), int)
for f in range(5):
    ev = fold == f
    dlf, _ = search_offsets(S[~ev], y[~ev], passes=2)
    PD[ev] = np.argmax(S[ev] + dlf, 1)
ub["U4_offsets_oof"] = rank_balanced(score(PD, y), y)
# U5: oracle given the top-2 rule-B candidates (perfect choice between the two best expected scores)
top2 = np.argsort(-S, 1)[:, :2]
s_t2 = np.maximum(score(top2[:, 0], y), score(top2[:, 1], y))
ub["U5_oracle_best_of_top2_ruleB_candidates"] = rank_balanced(s_t2, y)
ub["U6_oracle_true_level_within1_of_pred_scored_exact"] = rank_balanced(np.where(np.abs(d) <= 1, 1.0, 0.0), y)
out["upper_bounds"] = r4(ub)
out["seconds"] = round(time.time() - t0, 1)
with open(F.os.path.join(OUT, "r20_diag.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print(json.dumps(out["upper_bounds"], indent=1))
print(f"done {time.time() - t0:.0f}s")
