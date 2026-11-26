"""Label-free diagnostics of prereg01 player outputs. Never touches *answers*.csv."""
import json
import numpy as np, pandas as pd

R = r"D:\AI-Portfolio\competitions\aicup2026-go"
z = np.load(fr"{R}\results\prereg01\player\player_val_embeddings.npz")
cid, C, qid, Q = z["candidate_ids"], z["centroids"].astype(np.float64), z["question_ids"], z["queries"].astype(np.float64)
top5 = pd.read_csv(fr"{R}\results\prereg01\player\player_val_top5.csv")
out = {}
out["shapes"] = dict(C=C.shape, Q=Q.shape)
nC, nQ = np.linalg.norm(C, axis=1), np.linalg.norm(Q, axis=1)
qq = lambda x: {p: round(float(np.quantile(x, p)), 4) for p in (0.0, 0.05, 0.5, 0.95, 1.0)}
out["centroid_norm"] = qq(nC)
out["query_norm"] = qq(nQ)
out["l2_normalised?"] = dict(centroids=bool(np.allclose(nC, 1, atol=1e-3)), queries=bool(np.allclose(nQ, 1, atol=1e-3)))

# reproduce official ranking (Euclidean on unnormalised means, names sorted)
D = np.sqrt(((Q[:, None, :] - C[None, :, :]) ** 2).sum(-1))
order = np.argsort(D, 1, kind="stable")
rep = [[cid[j] for j in order[i, :5]] for i in range(len(Q))]
t5 = top5.set_index("question_id").loc[qid][[f"top_{k}" for k in range(1, 6)]].values.tolist()
out["reproduces_official_top5"] = int(sum(a == b for a, b in zip(rep, t5)))

def stats(S, name, larger_better=True):
    """S: similarity (larger better) or distance."""
    o = np.argsort(-S if larger_better else S, 1, kind="stable")
    s_sorted = np.take_along_axis(S, o, 1)
    m12 = np.abs(s_sorted[:, 0] - s_sorted[:, 1])
    m15 = np.abs(s_sorted[:, 0] - s_sorted[:, 4])
    spread = S.std(1)
    top1 = o[:, 0]
    N1 = np.bincount(top1, minlength=len(C))
    N5 = np.bincount(o[:, :5].ravel(), minlength=len(C))
    N10 = np.bincount(o[:, :10].ravel(), minlength=len(C))
    sk = lambda x: float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    return o, dict(
        margin12_over_rowstd=qq(m12 / spread),
        margin15_over_rowstd=qq(m15 / spread),
        frac_margin12_lt_005std=round(float((m12 / spread < 0.05).mean()), 4),
        distinct_top1=int((N1 > 0).sum()),
        max_times_top1=int(N1.max()),
        top1_count_hist={int(k): int(v) for k, v in zip(*np.unique(N1, return_counts=True))},
        cand_never_in_any_top5=int((N5 == 0).sum()),
        cand_never_in_any_top10=int((N10 == 0).sum()),
        N5_max=int(N5.max()), N5_skewness=round(sk(N5), 3),
        top10_hubs_share_of_all_top5_slots=round(float(np.sort(N5)[-10:].sum() / N5.sum()), 4),
    ), N1, N5

res = {}
o_e, res["euclid_official"], N1e, N5e = stats(D, "euclid", larger_better=False)
out["corr_centroid_norm_vs_N5_euclid (spearman)"] = float(pd.Series(nC).corr(pd.Series(N5e), method="spearman"))
out["corr_centroid_norm_vs_mean_dist (spearman)"] = float(pd.Series(nC).corr(pd.Series(D.mean(0)), method="spearman"))

Cn = C / nC[:, None]; Qn = Q / nQ[:, None]
S_cos = Qn @ Cn.T
o_c, res["cosine"], N1c, N5c = stats(S_cos, "cos")
out["corr_centroid_norm_vs_N5_cosine (spearman)"] = float(pd.Series(nC).corr(pd.Series(N5c), method="spearman"))

# centering by the candidate-pool mean (label-free, uses only candidate embeddings) then cosine
mu = C.mean(0)
Cc = C - mu; Qc = Q - mu
Ccn = Cc / np.linalg.norm(Cc, axis=1, keepdims=True); Qcn = Qc / np.linalg.norm(Qc, axis=1, keepdims=True)
S_cc = Qcn @ Ccn.T
o_cc, res["centered_cosine(pool mean)"], _, N5cc = stats(S_cc, "ccos")
out["norm_of_pool_mean"] = float(np.linalg.norm(mu))
out["mean_cos_between_centroids"] = float(((Cn @ Cn.T).sum() - len(C)) / (len(C) * (len(C) - 1)))
out["mean_cos_query_to_centroid"] = float(S_cos.mean())
out["mean_cos_between_centroids_after_centering"] = float(((Ccn @ Ccn.T).sum() - len(C)) / (len(C) * (len(C) - 1)))

# CSLS (k=10): 2cos(q,c) - r_C(q) - r_Q(c); r_Q(c) needs the question set (available at test time too)
k = 10
rq = np.sort(S_cos, 1)[:, -k:].mean(1)
rc = np.sort(S_cos, 0)[-k:, :].mean(0)
S_csls = 2 * S_cos - rq[:, None] - rc[None, :]
o_cs, res["csls_k10_cosine"], _, _ = stats(S_csls, "csls")
# CSLS with candidate-side hubness from candidates only (no question set): r_C'(c) = mean cos to its k nearest other centroids
CC = Cn @ Cn.T; np.fill_diagonal(CC, -np.inf)
rcc = np.sort(CC, 1)[:, -k:].mean(1)
S_cs2 = S_cos - 0.5 * rcc[None, :]
o_cs2, res["cand_only_local_scaling_k10"], _, _ = stats(S_cs2, "cs2")

out["similarity_variants"] = res

def agree(o1, o2):
    t1 = (o1[:, 0] == o2[:, 0]).mean()
    ov = np.mean([len(set(a[:5]) & set(b[:5])) / 5 for a, b in zip(o1, o2)])
    return dict(top1_same=round(float(t1), 4), top5_overlap=round(float(ov), 4))
out["agreement_vs_official"] = {
    "cosine": agree(o_e, o_c), "centered_cosine": agree(o_e, o_cc),
    "csls_k10": agree(o_e, o_cs), "cand_only_scaling": agree(o_e, o_cs2)}

# how many questions' top-1 is a candidate that is top-1 for >=2 questions (conflicts under 1-to-1)
out["questions_whose_top1_is_shared_euclid"] = int(np.isin(o_e[:, 0], np.where(N1e >= 2)[0]).sum())
# mutual nearest neighbours
nn_c_for_q = o_e[:, 0]
nn_q_for_c = np.argsort(D, 0, kind="stable")[0]
out["mutual_nn_pairs_euclid"] = int(sum(nn_q_for_c[nn_c_for_q[i]] == i for i in range(len(Q))))

# query stats vs num_games
qs = pd.read_csv(fr"{R}\data\local\player_val_questions.csv")
ng = qs.set_index("question_id").loc[qid]["num_games"].to_numpy()
out["num_games_quantiles"] = qq(ng)
out["spearman_numgames_vs_querynorm"] = float(pd.Series(ng).corr(pd.Series(nQ), method="spearman"))
m12 = np.sort(D, 1)[:, 1] - np.sort(D, 1)[:, 0]
out["spearman_numgames_vs_margin12_euclid"] = float(pd.Series(ng).corr(pd.Series(m12 / D.std(1)), method="spearman"))
out["spearman_numgames_vs_min_dist"] = float(pd.Series(ng).corr(pd.Series(D.min(1)), method="spearman"))
# nearest centroid distance vs typical
out["min_dist_over_median_dist"] = qq(D.min(1) / np.median(D, 1))
# effective dimensionality of centroids (participation ratio)
ev = np.linalg.eigvalsh(np.cov(C.T))[::-1]
out["centroid_PR_dim"] = float(ev.sum() ** 2 / (ev ** 2).sum())
out["centroid_top5_var_share"] = float(ev[:5].sum() / ev.sum())
evq = np.linalg.eigvalsh(np.cov(Q.T))[::-1]
out["query_PR_dim"] = float(evq.sum() ** 2 / (evq ** 2).sum())
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
