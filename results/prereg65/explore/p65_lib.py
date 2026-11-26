"""prereg65 (competition 2): OPP4B, the two-channel opponent term of Gemini Neon's Q65B (PLAN.md, NOTES.md,
frozen.json). numpy / scipy only; prereg58 / prereg59 / prereg64 and tools/p09_forward are imported, never edited.
Every function is a port of data/advice/gemini_1121n/04_final_candidate_stack.py (line pointers in the docstrings).

  q_c49   head rows tau1 70 (own row left out) + E prototypes tau2 50 - 1 + X prototypes tau2 50 - 2, prior-corrected
  M_c49   all-D table (add 0.5), HD = 1 rows shrunk 0.25 toward the rank-difference kernel, / mean over r
  q_P1    OPP_P1_v2 (4 numpy members) on the opponent-side P1 means; M_P1 = Neon's frozen M_p1_tr
  l_g(r)  log(alpha R_c49 + (1 - alpha) R_P1 + 1e-9), alpha = clip((max q_c49 - 0.1) / 0.9, 0, 1) ** 1.5
  OPP4B   10 x mean over the question's games (flat)
  M8      same-player merge inside a test pool (04 lines 691-748, M8 branch)
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg58", "explore"),
          os.path.join(REPO, "results", "prereg59", "explore"), os.path.join(REPO, "results", "prereg64", "explore")):
    if p not in sys.path:
        sys.path.insert(0, p)
import p58_lib as K58  # noqa: E402
import p59_lib as K59  # noqa: E402
import p64_lib as K64  # noqa: E402

L = K58.L
NEON = os.path.join(REPO, "data", "advice", "gemini_1121n")
OPP_P1_NPZ = os.path.join(NEON, "opp_p1_v2.npz")
SCOND_NPZ = os.path.join(NEON, "c2_2_scond_p1.npz")
OPP_P1_SHA = "e47912f7afe65ff053c7f3c802062f58f23b4b50c96a9e196d701e7a43b49f5c"
SCOND_SHA = "7ef4b1d6d22353f728782e0f5f233398f93baddad6bd1e45cbba08863751d1f1"
DATA = os.path.join(REPO, "data", "prereg65")

TAU1, TAU2, B_E, B_X = 70, 50, -1.0, -2.0      # 04 lines 351-354, 373-374
LAM_HD1 = 0.25                                  # 04 line 410
P_POW = 1.5                                     # 04 line 489, REPORT line 217
SCALE = 10.0                                    # 04 line 456
BLOCK = 4000
EPS = K58.EPS                                   # 1e-9
M8_THR, M8_GUARD, M8_BETA = 0.80, 2.5, 0.85     # 04 lines 699-701, 731
ii = np.arange(L)
SM = np.where(np.abs(ii[:, None] - ii[None, :]) == 0, 1.0,
              np.where(np.abs(ii[:, None] - ii[None, :]) == 1, math.exp(-1.0), 0.0))


# ---------------------------------------------------------------- Neon's small helpers (04 lines 120-135)
def softmax(Z):
    z = Z - np.max(Z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def log_softmax(Z):
    z = Z - np.max(Z, axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def decode(Z):
    return np.argmax(softmax(Z) @ SM, axis=1)


def rb_score(pred, y):
    sc = SM[pred, y]
    present = [k for k in range(L) if np.any(y == k)]
    return float(np.mean([sc[y == k].mean() for k in present]))


# ---------------------------------------------------------------- c49 channel (04 lines 345-382)
def class_counts(wr, pr_rank, src):
    src = np.asarray(src).astype(str)
    return {"head": np.bincount(np.asarray(wr), minlength=L).astype(np.float64),
            "E": np.bincount(np.asarray(pr_rank)[src == "E"], minlength=L).astype(np.float64),
            "X": np.bincount(np.asarray(pr_rank)[src == "X"], minlength=L).astype(np.float64)}


def c49_parts(emb, Wn, wr, Pn, pr_rank, src, own_head=None):
    src = np.asarray(src).astype(str)
    isE, isX = src == "E", src == "X"
    pr_rank = np.asarray(pr_rank)
    return {"A": K59._rank_sums(emb, Wn, np.asarray(wr), (TAU1,), own_head, BLOCK)[TAU1],
            "E": K59._rank_sums(emb, Pn[isE], pr_rank[isE], (TAU2,), None, BLOCK)[TAU2],
            "X": K59._rank_sums(emb, Pn[isX], pr_rank[isX], (TAU2,), None, BLOCK)[TAU2]}


def q_from_parts(parts, N):
    """04 lines 368-374, 382 verbatim in effect (splitEX)."""
    SA, mA = parts["A"]
    SE, mE = parts["E"]
    SX, mX = parts["X"]
    SA_u = SA / N["head"][None, :] * N["head"].mean()
    SE_u = SE / N["E"][None, :] * N["E"].mean()
    SX_u = SX / N["X"][None, :] * N["X"].mean()
    mm = np.maximum(mA, np.maximum(mE + B_E, mX + B_X))
    q = SA_u * np.exp(mA - mm)[:, None] + SE_u * np.exp(mE + B_E - mm)[:, None] + SX_u * np.exp(mX + B_X - mm)[:, None]
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-300)


def q_c49(emb, Wn, wr, Pn, pr_rank, src, own_head=None):
    return q_from_parts(c49_parts(emb, Wn, wr, Pn, pr_rank, src, own_head), class_counts(wr, pr_rank, src))


# ---------------------------------------------------------------- tables (04 lines 179-199)
def table_bayes(counts, K_hd1=None, lam_hd1=0.0):
    M = K59.table(counts)
    if lam_hd1 > 0.0 and K_hd1 is not None:
        for cc in range(2):
            Km = K_hd1[1, cc] / np.maximum(K_hd1[1, cc].sum(axis=1, keepdims=True), 1e-12)
            M[1, cc] = (1.0 - lam_hd1) * M[1, cc] + lam_hd1 * Km
    q_bar = np.maximum(M.mean(axis=2, keepdims=True), 1e-12)
    return M / q_bar


def c49_table(q, r, h, c, ok, lam=LAM_HD1):
    """The all-D table of the term (04 lines 853-855)."""
    K = K58.kernel_matrix(K58.kernel(K58.kernel_counts(q, r, h, c, ok)))
    return table_bayes(K59.table_counts(q, r, h, c, ok), K_hd1=K, lam_hd1=lam)


def game_R(q, M_mat, h, c):
    out = np.zeros((len(q), L), np.float64)
    for hh in range(3):
        for cc in range(2):
            m = (np.asarray(h) == hh) & (np.asarray(c) == cc)
            if m.any():
                out[m] = q[m] @ M_mat[hh, cc].T
    return np.maximum(out, 0.0)


def lr_lg(q_c, R_c49, R_fallback, p_pow=P_POW):
    al = np.clip((np.max(q_c, axis=1, keepdims=True) - 0.1) / 0.9, 0.0, 1.0) ** p_pow
    return np.log(al * R_c49 + (1.0 - al) * R_fallback + EPS)


# ---------------------------------------------------------------- P1 channel (04 lines 229-250)
def load_opp_p1(path=OPP_P1_NPZ):
    if K58.sha256_file(path) != OPP_P1_SHA:
        raise SystemExit("opp_p1_v2.npz is not the frozen file")
    with np.load(K58.guard(path), allow_pickle=False) as z:
        return {k: np.asarray(z[k], np.float64) for k in z.files}


def load_m_p1(path=SCOND_NPZ):
    if K58.sha256_file(path) != SCOND_SHA:
        raise SystemExit("c2_2_scond_p1.npz is not the frozen file")
    with np.load(K58.guard(path), allow_pickle=False) as z:
        M = np.asarray(z["M_p1_tr"], np.float64)
    if M.shape != (3, 2, L, L) or not np.isfinite(M).all():
        raise SystemExit("M_p1_tr is not a [3, 2, 10, 10] table")
    return M


def predict_opp_p1(op1, X_side_raw, block=8192, members=("mlp0", "mlp1", "clm0", "clm1")):
    """-> [N, 10] mean of the members' probabilities (04 lines 234-250); members selectable for the seam."""
    import p09_forward as FW09
    N = len(X_side_raw)
    q_ens = np.zeros((N, L), np.float64)
    for s in range(0, N, block):
        e = min(N, s + block)
        Xs = FW09.standardise_x(np.asarray(X_side_raw[s:e], np.float64), op1["mu_t"], op1["sd_t"])
        ps = []
        for m in members:
            k, i = m[:3], int(m[3:])
            h = FW09.gelu(Xs @ op1[f"{k}{i}_W1"].T + op1[f"{k}{i}_b1"])
            if k == "mlp":
                ps.append(softmax(h @ op1[f"mlp{i}_W2"].T + op1[f"mlp{i}_b2"]))
            else:
                f = h @ op1[f"clm{i}_W2"].T + op1[f"clm{i}_b2"]
                ps.append(np.exp(FW09.clm_logp(f, op1[f"clm{i}_t0"], op1[f"clm{i}_dt"])))
        q_ens[s:e] = np.mean(ps, axis=0)
    return q_ens


# ---------------------------------------------------------------- the term
def game_values(qc, M_c49, h, c, c49_count, q_p1, M_p1, p1_ok):
    """[G, 10] l_g(r); 0 for a game without a c49 sample; R_P1 = 1 for a game without an opponent P1 vector."""
    R_c = game_R(qc, M_c49, h, c)
    R_p = game_R(q_p1, M_p1, h, c)
    R_p[~np.asarray(p1_ok, bool)] = 1.0
    lg = lr_lg(qc, R_c, R_p)
    lg[np.asarray(c49_count) <= 0] = 0.0
    return lg


def question_term(S, lg):
    return SCALE * np.asarray(S @ lg)


# ---------------------------------------------------------------- M8 (04 lines 691-748, M8 branch verbatim)
def m8(lp_base, S, q_cols, thr=M8_THR):
    """-> (pred_out, lp_new, edges, pred0)"""
    Q = len(lp_base)
    lp_norm = log_softmax(lp_base)
    P0 = np.exp(lp_norm)
    E_rk = P0 @ np.arange(L, dtype=np.float64)
    pred0 = decode(lp_norm)
    same_c = (np.asarray(q_cols)[:, None] == np.asarray(q_cols)[None, :])
    rk_ok = (np.abs(E_rk[:, None] - E_rk[None, :]) <= M8_GUARD)
    adj = (S > thr) & same_c & rk_ok
    np.fill_diagonal(adj, False)
    W_adj = np.where(adj, 1.0, 0.0)
    lp_new = lp_norm + M8_BETA * (W_adj @ lp_norm)
    pr_new = decode(lp_new)
    pred_out = pred0.copy()
    for i in range(Q):
        nbrs = np.flatnonzero(adj[i])
        if len(nbrs) == 0:
            continue
        if pr_new[i] != pred0[i]:
            grp = [i] + nbrs.tolist()
            if all(pr_new[u] == pr_new[i] for u in grp) and (pr_new[i] in set(pred0[grp].tolist())):
                pred_out[i] = pr_new[i]
    return pred_out, lp_new, int(np.triu(adj, 1).sum()), pred0
