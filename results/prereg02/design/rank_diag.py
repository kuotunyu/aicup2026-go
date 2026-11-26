"""Label-free diagnostics of prereg01 rank outputs. Never touches *answers*.csv."""
import json, math
import numpy as np, pandas as pd

R = r"D:\AI-Portfolio\competitions\aicup2026-go"
LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
E1 = math.exp(-1)
# public per-level question counts from docs/VALIDATION.md (not answers)
VAL_COUNTS = dict(D=178, C=163, B=207, A=308, **{"1D": 87, "2D": 122, "3D": 124, "4D": 128, "5D": 131, "6D": 141})
cnt = np.array([VAL_COUNTS[l] for l in LV], float)
prior_val = cnt / cnt.sum()

df = pd.read_csv(fr"{R}\results\prereg01\rank\rank_val_probs.csv", dtype={"question_id": str})
P = df[[f"p_{l}" for l in LV]].to_numpy(float)
P = P / P.sum(1, keepdims=True)
n = df["n_games"].to_numpy()
N = len(df)
out = {}

def ent(p):
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(1)

def scoreB(p):
    s = p.copy()
    s[:, 1:] += E1 * p[:, :-1]
    s[:, :-1] += E1 * p[:, 1:]
    return s

def dist(idx):
    return {LV[k]: int((idx == k).sum()) for k in range(10)}

mx = P.max(1); H = ent(P); am = P.argmax(1)
srt = np.sort(P, 1)
out["max_prob"] = {q: float(np.quantile(mx, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
out["max_prob_mean"] = float(mx.mean())
out["top1_minus_top2_median"] = float(np.median(srt[:, -1] - srt[:, -2]))
out["top1_minus_top2_lt_0.02_frac"] = float(((srt[:, -1] - srt[:, -2]) < 0.02).mean())
out["entropy_nats"] = {q: float(np.quantile(H, q)) for q in (0.05, 0.5, 0.95)}
out["entropy_uniform"] = math.log(10)
out["entropy_norm_median"] = float(np.median(H) / math.log(10))

# peakedness vs number of games
bins = [(5, 7), (8, 10), (11, 13), (14, 16), (17, 20)]
rows = []
for lo, hi in bins:
    m = (n >= lo) & (n <= hi)
    rows.append(dict(games=f"{lo}-{hi}", n_q=int(m.sum()), maxp_mean=round(float(mx[m].mean()), 4),
                     entropy_mean=round(float(H[m].mean()), 4)))
out["by_n_games"] = rows
out["spearman_n_vs_maxp"] = float(pd.Series(n).corr(pd.Series(mx), method="spearman"))
out["spearman_n_vs_entropy"] = float(pd.Series(n).corr(pd.Series(H), method="spearman"))

# top-2 adjacency, unimodality
top2 = np.argsort(-P, 1)[:, :2]
gap = np.abs(top2[:, 0] - top2[:, 1])
out["top2_gap_hist"] = {int(g): int((gap == g).sum()) for g in np.unique(gap)}
out["top2_adjacent_frac"] = float((gap == 1).mean())
loc = np.zeros(N, int)
for k in range(10):
    left = P[:, k - 1] if k > 0 else -np.inf
    right = P[:, k + 1] if k < 9 else -np.inf
    loc += ((P[:, k] > left) & (P[:, k] > right)).astype(int)
out["n_local_maxima_hist"] = {int(v): int((loc == v).sum()) for v in np.unique(loc)}
out["both_ends_in_top3_frac"] = float(np.mean([(0 in r) and (9 in r) for r in np.argsort(-P, 1)[:, :3]]))
out["p_D_plus_p_6D_mean"] = float((P[:, 0] + P[:, 9]).mean())

# predicted vs public counts
out["pred_A"] = dist(am)
out["pred_B"] = dist(scoreB(P).argmax(1))
out["val_counts_public"] = VAL_COUNTS
out["mean_prob_x_N (expected count under model)"] = {LV[k]: round(float(P[:, k].sum()), 1) for k in range(10)}
out["mean_prob"] = [round(float(x), 4) for x in P.mean(0)]
out["prior_val"] = [round(float(x), 4) for x in prior_val]

# expected level index and spread
idx = np.arange(10)
mu = P @ idx
sd = np.sqrt(P @ idx**2 - mu**2)
out["E_level_index_quantiles"] = {q: round(float(np.quantile(mu, q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
out["E_level_hist_(rounded)"] = dist(np.clip(np.rint(mu), 0, 9).astype(int))
out["sd_level_median"] = float(np.median(sd))
# model-implied expected score (self-consistency) vs realised aggregate numbers in RESULTS.md
sB = scoreB(P)
out["model_expected_exact_A"] = float(mx.mean())
iB = sB.argmax(1)
out["model_expected_exact_B"] = float(P[np.arange(N), iB].mean())
out["model_expected_score_A_per_q"] = float(sB[np.arange(N), am].mean())
out["model_expected_score_B_per_q"] = float(sB.max(1).mean())
adjA = sB[np.arange(N), am] - mx
out["model_expected_adjacent_A"] = float(adjA.mean() / E1)
out["realised_from_RESULTS_md"] = dict(A_exact=0.3499, A_adj=0.3537, A_mean=0.4800, B_exact=0.3304, B_adj=0.4393, B_mean=0.4920)

# power sharpening sensitivity (does not change A; changes B)
sens = {}
for T in (0.5, 1.5, 2, 3, 5, 10):
    Q = P ** T; Q /= Q.sum(1, keepdims=True)
    iT = scoreB(Q).argmax(1)
    sens[str(T)] = dict(changed_vs_B=int((iT != iB).sum()), same_as_A=int((iT == am).sum()),
                        model_exact=round(float(Q.max(1).mean()), 4), dist=dist(iT))
out["B_under_power_T"] = sens
# temperature implied by realised A exact rate (DIAGNOSTIC ONLY - fitted to the scored val aggregate, not for tuning)
Ts = np.linspace(1, 12, 1101)
vals = []
for T in Ts:
    Q = P ** T; Q /= Q.sum(1, keepdims=True); vals.append(Q.max(1).mean())
vals = np.array(vals)
out["T_where_model_exact_equals_0.3499"] = float(Ts[np.argmin(abs(vals - 0.3499))])

# EM prior re-estimation (Saerens et al. 2002), label-free; assumes train prior ~ uniform
def em(P, train_prior, iters=1000):
    w = np.ones(10) / 10
    for _ in range(iters):
        Q = P * (w / train_prior); Q /= Q.sum(1, keepdims=True)
        w_new = Q.mean(0)
        if np.abs(w_new - w).max() < 1e-9: break
        w = w_new
    return w, Q
# training prior over games for fit players (per game) from players_split: counts only for train split, not val labels
sp = pd.read_csv(fr"{R}\results\split_v1\players_split.csv")
tr = sp[sp.split == "train"].groupby("rank")["n_games"].sum()
train_prior = np.array([tr[l] for l in LV], float); train_prior /= train_prior.sum()
out["train_game_prior (train split, incl. inner)"] = [round(float(x), 4) for x in train_prior]
for T in (1, 2, 4):
    Q = P ** T; Q /= Q.sum(1, keepdims=True)
    w, Qa = em(Q, train_prior)
    out[f"EM_prior_T{T}"] = [round(float(x), 4) for x in w]
    out[f"EM_prior_T{T}_argmax_dist"] = dist(Qa.argmax(1))
out["EM_note"] = "compare EM_prior_* with prior_val; EM assumes calibrated posteriors"

# per-level: how confident when predicted as that level (argmax)
out["mean_maxp_by_pred_A"] = {LV[k]: round(float(mx[am == k].mean()), 4) if (am == k).any() else None for k in range(10)}
# per-level mean of p_k across questions where k is argmax vs not (spread of that column)
out["col_quantiles_p"] = {LV[k]: [round(float(np.quantile(P[:, k], q)), 3) for q in (0.1, 0.5, 0.9, 0.99)] for k in range(10)}

# games length per question from SGF text (question content, not answers)
qs = pd.read_csv(fr"{R}\data\local\rank_val_questions.csv", dtype={"question_id": str})
assert list(qs.question_id) == list(df.question_id)
sgf = [c for c in qs.columns if c.startswith("sgf_")]
mv = qs[sgf].apply(lambda col: col.fillna("").str.count(r";[BW]\["))
avg_moves = (mv.sum(1) / qs["num_games"]).to_numpy()
out["avg_moves_per_game_quantiles"] = {q: round(float(np.quantile(avg_moves, q)), 1) for q in (0.05, 0.5, 0.95)}
out["spearman_avgmoves_vs_maxp"] = float(pd.Series(avg_moves).corr(pd.Series(mx), method="spearman"))
out["spearman_avgmoves_vs_Elevel"] = float(pd.Series(avg_moves).corr(pd.Series(mu), method="spearman"))
# argmax at an end vs E level in the middle -> U-shape signal
mid = (mu > 2.5) & (mu < 6.5)
out["argmax_at_end_but_Elevel_in_3..6"] = int(((am == 0) | (am == 9))[mid].sum())
out["n_Elevel_in_3..6"] = int(mid.sum())

print(json.dumps(out, indent=1, ensure_ascii=False))
