"""Task B part 2b: level signal of the cheap SGF features (fit/inner only) and their value on top of R20.

Inputs: sgf_features.npz (from sgf_features.py), r20_oof.npz (from r20_diag.py), the prereg02 inner
artifacts via common.load_all(). Per-game models are trained on fit games only and applied to inner
games; question-level terms are tuned with the tune protocol (fit rounds of the other folds -> eval
rounds of the held-out fold). Writes feat_signal.json next to this file.
"""
import json
import time

import numpy as np
from scipy import optimize, sparse

from common import (L, LEVELS, OUT, Boot, F, lsm, oof_gen, per_level, r4, rank_balanced, rule_b, score, load_all)

t0 = time.time()
Z = np.load(F.os.path.join(OUT, "sgf_features.npz"), allow_pickle=False)
X = Z["X"].astype(np.float64)
names = [str(s) for s in Z["names"]]
lev = Z["level"].astype(np.int64)
inner = Z["is_inner"].astype(bool)
fitm = ~inner
assert fitm.sum() == 720307 and inner.sum() == 73556, (fitm.sum(), inner.sum())
out = {"what": "cheap SGF features: level signal on fit/inner and value on top of R20 (inner OOF)",
       "games": {"fit": int(fitm.sum()), "inner": int(inner.sum())}}
LOG2 = np.log(2)


# ------------------------------------------------------------------ 1. per-feature MI and level trend
def edges_for(col):
    u = np.unique(col)
    if len(u) <= 20:
        return (u[:-1] + u[1:]) / 2
    return np.unique(np.quantile(col, np.linspace(0, 1, 21)[1:-1]))


def mi_bits(col, lv, edges, weights=None):
    b = np.searchsorted(edges, col, side="right")
    nb = len(edges) + 1
    J = np.zeros((nb, L))
    np.add.at(J, (b, lv), 1.0 if weights is None else weights)
    p = J / J.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / (px @ py)[nz])) / LOG2), nb


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


feat = {}
Xf, Xi = X[fitm], X[inner]
lf, li = lev[fitm], lev[inner]
rng = np.random.default_rng(0)
sub = rng.choice(len(Xf), 200000, replace=False)
for j, nm in enumerate(names):
    e = edges_for(Xf[:, j])
    mf, nb = mi_bits(Xf[:, j], lf, e)
    mi_, _ = mi_bits(Xi[:, j], li, e)
    feat[nm] = {"mi_bits_fit": mf, "mi_bits_inner": mi_, "bins": nb,
                "bias_bits_fit": (nb - 1) * (L - 1) / (2 * len(Xf) * LOG2),
                "spearman_with_level_fit": spearman(Xf[sub, j], lf[sub]),
                "mean_by_level_fit": [float(Xf[lf == i, j].mean()) for i in range(L)]}
order = sorted(feat, key=lambda k: -feat[k]["mi_bits_fit"])
out["per_feature"] = r4({k: feat[k] for k in order})
nm_i = names.index("n_moves")
out["n_moves_by_level_fit"] = r4({LEVELS[i]: {"mean": Xf[lf == i, nm_i].mean(), "median": float(np.median(Xf[lf == i, nm_i])),
                                              "frac_lt150": np.mean(Xf[lf == i, nm_i] < 150),
                                              "frac_eq100": np.mean(Xf[lf == i, nm_i] == 100)} for i in range(L)})
print(f"MI done {time.time() - t0:.0f}s; top: {[(k, round(feat[k]['mi_bits_fit'], 4)) for k in order[:8]]}", flush=True)


# ------------------------------------------------------------------ 2. per-game multinomial logistic models (fit -> inner)
def design_binned(cols, ref=None):
    """One-hot quantile bins (edges from fit) per column -> sparse; ref = (edges list)."""
    edges = ref or [edges_for(Xf[:, j]) for j in cols]
    blocks = []
    off = 0
    rows_all, cols_all = [], []
    def build(M):
        r_, c_ = [], []
        o = 0
        for e, j in zip(edges, cols):
            b = np.searchsorted(e, M[:, j], side="right")
            r_.append(np.arange(len(M)))
            c_.append(o + b)
            o += len(e) + 1
        R = np.concatenate(r_)
        C = np.concatenate(c_)
        return sparse.csr_matrix((np.ones(len(R)), (R, C)), shape=(len(M), o))
    return build, edges


def fit_mlr(A, y, lam, maxiter=300):
    n, d = A.shape
    Y = np.zeros((n, L))
    Y[np.arange(n), y] = 1
    # level-balanced weights (fit games are close to balanced; make it exact)
    cnt = np.bincount(y, minlength=L)
    w = (1.0 / (L * cnt))[y]

    def f(th):
        W = th[:d * L].reshape(d, L)
        b = th[d * L:]
        z = A @ W + b
        ls = lsm(z)
        loss = -np.sum(w * ls[np.arange(n), y]) + lam * np.sum(W * W)
        G = (np.exp(ls) - Y) * w[:, None]
        gW = A.T @ G + 2 * lam * W
        return loss, np.concatenate([np.asarray(gW).ravel(), G.sum(0)])
    r = optimize.minimize(f, np.zeros(d * L + L), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return r.x[:d * L].reshape(d, L), r.x[d * L:], r


def eval_lp(lp, y):
    ce = float(np.mean([-lp[y == i, i].mean() for i in range(L)]))
    acc = float(np.mean([np.mean(lp[y == i].argmax(1) == i) for i in range(L)]))
    return {"ce_nats_level_balanced": ce, "info_bits": (np.log(10) - ce) / LOG2, "acc_level_balanced": acc}


GROUPS = {
    "length_ending": ["n_moves", "target_moved_last", "unique_points_frac", "last20_mean_line", "last20_t_line12",
                      "first_replay_frac", "t_endgame_mean_line"],
    "lines": ["t_line1", "t_line2", "t_line3", "t_line4", "t_line5plus", "t_early_line12", "t_early_line34",
              "t_early_center", "o_line1", "o_line2", "t_early_line1_count"],
    "locality_tenuki": ["t_mean_dist_prev", "t_tenuki", "t_tenuki_mid", "t_contact", "t_diag_contact", "t_mean_dist_own",
                        "t_own_adjacent", "o_tenuki", "o_contact"],
    "captures": ["replay_rate", "t_captured_rate", "o_captured_rate", "ko_points"],
    "first_moves": ["t_first_44", "t_first_34", "t_first_33", "t_first_other", "t_first10_corner"],
}
GROUPS["all"] = [n for n in names if n != "is_black"]

D = load_all(dump_names=("p02e9_d4",))
keys = D["keys"]
row_of = {(r, g): i for i, (r, g) in enumerate(zip(Z["rank"].tolist(), Z["game_id"].tolist()))}
map_inner = np.asarray([row_of[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
assert inner[map_inner].all() and (lev[map_inner] == keys["label"]).all()
assert (Z["player_id"][map_inner] == np.asarray(keys["player_id"])).all()
lp5 = F.source_logprobs("C5", D["dumps"])
opg = np.where(D["op_has"][:, None], D["op_o"], 0.0)
pg = {"CNN_C5_per_game": eval_lp(lp5, keys["label"]),
      "opening_per_game (no-score games uniform)": eval_lp(np.where(D["op_has"][:, None], D["op_o"], -np.log(10)), keys["label"])}
models = {}
for gname, cols_n in GROUPS.items():
    cols = [names.index(c) for c in cols_n]
    build, edges = design_binned(cols)
    Af = build(Xf)
    W, b, r = fit_mlr(Af, lf, lam=1e-6)
    lpf = lsm(np.asarray(build(X[map_inner]) @ W) + b)
    pg[f"features[{gname}] binned-MLR"] = eval_lp(lpf, keys["label"])
    pg[f"features[{gname}] binned-MLR"]["fit_ce"] = float(r.fun)
    models[gname] = lpf
    print(gname, r4(pg[f"features[{gname}] binned-MLR"]), f"{time.time() - t0:.0f}s", flush=True)
# CNN + features at the per-game level (additive in log space, weight fitted on inner folds by player)
out["per_game_inner"] = r4(pg)

# ------------------------------------------------------------------ 3. question level: tune protocol on inner pseudo-questions
Xfq, Xeq, Sf, Se = D["X_fit"], D["X_eval"], D["S_fit"], D["S_eval"]
y = Xeq.y
R = np.load(F.os.path.join(OUT, "r20_oof.npz"))
assert (R["y"] == y).all()
sc20 = score(rule_b(R["P20"]), y)
boot = Boot(Xeq.player, y, len(D["players"]))
m5f, m5e = F.pooled(Sf, lp5, "log"), F.pooled(Se, lp5, "log")
q = {"R20_oof_rank_balanced": rank_balanced(sc20, y), "R20_per_level": per_level(sc20, y)}
for gname, lpf in models.items():
    mff, mfe = F.pooled(Sf, lpf, "log"), F.pooled(Se, lpf, "log")
    Pa, _ = oof_gen(Xfq, Xeq, mff, [], mfe, [])
    sa = score(rule_b(Pa), y)
    Pb, ths = oof_gen(Xfq, Xeq, m5f, [Xfq.mop, mff], m5e, [Xeq.mop, mfe])
    sb = score(rule_b(Pb), y)
    rec = {"features_only_calibrated": rank_balanced(sa, y),
           "R20_plus_features": rank_balanced(sb, y), "diff_vs_R20": rank_balanced(sb, y) - rank_balanced(sc20, y),
           "question_mean_diff": float(sb.mean() - sc20.mean()),
           "w_feat_by_fold": [float(t[3]) for t in ths], "w_open_by_fold": [float(t[2]) for t in ths]}
    bd = boot.diff(sb, sc20)
    rec.update({"ci95": bd["ci95"], "frac_le0": bd["frac_le0"]})
    if gname == "all":
        rec["per_level_R20_plus_all"] = per_level(sb, y)
        rec["per_level_diff"] = {k: per_level(sb, y)[k] - per_level(sc20, y)[k] for k in LEVELS}
        rec["per_level_ci95"] = bd["per_level_ci95"]
        rec["features_only_per_level"] = per_level(sa, y)
    q[gname] = rec
    print(gname, r4({k: v for k, v in rec.items() if not isinstance(v, dict)}), f"{time.time() - t0:.0f}s", flush=True)
out["question_level"] = r4(q)
out["seconds"] = round(time.time() - t0, 1)
with open(F.os.path.join(OUT, "feat_signal.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print(f"done {time.time() - t0:.0f}s")
