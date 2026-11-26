"""prereg64 (competition 2): opponent term v3 with c49 + Gemini's anchor fusion + same-player soft merge
(PLAN.md, NOTES.md, frozen.json). numpy / scipy only; prereg58 / prereg59 code is imported, never edited.

  OPP3   q_g = soft-max over c49 head rows (50 * cos, own row left out) and prototypes (30 * cos + 1), summed by rank;
         l_g(r) = log(sum_s q_g[s] M[h_g, c_g][r][s] + 1e-9); question value = 10 x mean over its games
  fusion Z = w_A * log_softmax(Q25 logits) + w_O * OPP3, (w_A, w_O) by Gemini 04's eval_scalar_oof objective
  merge  colour-centred, within-question whitened (lam 0.3) cosine of question means; L += 1.0 * (W * n/12) @ L
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "results", "prereg58", "explore"), os.path.join(REPO, "results", "prereg59", "explore")):
    if p not in sys.path:
        sys.path.insert(0, p)
import p58_lib as K58  # noqa: E402
import p59_lib as K59  # noqa: E402

TAG = K58.TAG
DATA = os.path.join(REPO, "data", "prereg64")
HEAD = os.path.join(DATA, "c49_head.npz")
C49_CKPT_SHA = "c091d3af1797a19ce98dc0a670eea211ceca2de4b58ee4a3c62b73b854748e12"
OPP_D = os.path.join(REPO, "data", "prereg56", "emb_opp_D", TAG, "store_D.npz")
OWN_E = os.path.join(REPO, "data", "prereg49", "emb", TAG, "store_E.npz")
OWN_PROTO = os.path.join(DATA, "emb_proto", TAG, "rows_proto.npz")
PROTO_ROWS = os.path.join(DATA, "proto_rows.npy")
OPP_INNER = os.path.join(DATA, "emb_opp_inner", TAG, "rows_inner.npz")
OPP_RANK = os.path.join(REPO, "data", "prereg56", "emb_opp_rank", TAG, "pool_%s.npz")
OWN_RANK = os.path.join(REPO, "data", "prereg56", "emb_rank", TAG, "pool_%s.npz")
PROTO = os.path.join(DATA, "prototypes.npz")
KERNEL = os.path.join(HERE, "kernel.json")
SEAM = os.path.join(HERE, "seam.json")
EVAL = os.path.join(HERE, "eval.json")
MERGE_SIM = os.path.join(HERE, "merge_sim.json")
RESULTS = os.path.join(HERE, "RESULTS.md")
STARTED = os.path.join(DATA, "eval_started.json")

L = K58.L
TAU1, TAU2, BIAS = 50, 30, 1          # frozen.json (Gemini 03 lines 532-535)
SCALE = 10.0                          # 04 line 412
BLOCK = 4000
W_START, W_BOUND, W_RIDGE = 0.4, 4.0, 1e-3     # 04 lines 424-430
MERGE_LAM, MERGE_THR, MERGE_NDIV, MERGE_BETA = 0.3, 0.65, 12.0, 1.0   # 04 lines 109, 156-161
guard = K58.guard
l2 = K59.l2


def load_head(path=HEAD):
    with np.load(guard(path), allow_pickle=False) as z:
        Wn = l2(np.asarray(z["weight"], np.float64))
        ids = np.asarray(z["class_id"]).astype(str)
        cr = np.asarray(z["class_rank"], np.int64)
        sha = str(z["ckpt_sha256"])
    if sha != C49_CKPT_SHA:
        raise SystemExit("c49_head.npz is not of the c49 checkpoint")
    return Wn, ids, cr


def q_opp(emb, Wn, wr, Pn, pr, own_head=None):
    """q [G, 10]: soft-max over head logits TAU1 * cos (own_head[g] >= 0 left out) and prototype logits
    TAU2 * cos + BIAS, summed by rank (exactly Gemini's _rank_sums / combine calls)."""
    parts = {"A": K59._rank_sums(emb, Wn, np.asarray(wr), (TAU1,), own_head, BLOCK),
             "B": K59._rank_sums(emb, Pn, np.asarray(pr), (TAU2,), None, BLOCK)}
    return K59.combine(parts, TAU1, TAU2, BIAS)


def game_values(q, M, h, c, count):
    lg = K59.game_ll(q, M, h, c)
    lg[np.asarray(count) == 0] = 0.0
    return lg


def question_term(S, lg):
    return SCALE * np.asarray(S @ lg)


# ---------------------------------------------------------------- fusion (Gemini 04 eval_scalar_oof)
def log_softmax(z):
    z = np.asarray(z, np.float64)
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def fit_weights(terms, y):
    """(w_1..w_K) minimising mean NLL + 1e-3 * sum w^2, L-BFGS-B from 0.4, bounds [0, 4] (04 lines 422-430)."""
    K_t = len(terms)
    y = np.asarray(y)

    def nll(ws):
        Z = sum(ws[i] * terms[i] for i in range(K_t))
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(np.maximum(P[np.arange(len(P)), y], 1e-15))) + W_RIDGE * np.sum(ws * ws))

    res = minimize(nll, np.full(K_t, W_START), method="L-BFGS-B", bounds=[(0.0, W_BOUND)] * K_t)
    return np.asarray(res.x, np.float64), {"loss": float(res.fun), "nit": int(res.nit), "success": bool(res.success)}


def fused(ws, terms):
    return sum(ws[i] * terms[i] for i in range(len(terms)))


# ---------------------------------------------------------------- merge (Gemini 04 cc_whiten_sim + 156-161)
def cc_whiten_sim(e_games, q_groups, q_cols, lam=MERGE_LAM):
    """04 lines 109-130, verbatim in effect."""
    e_c = np.array(e_games, np.float64, copy=True)
    for cv in ("B", "W"):
        gc_idx = [g for qi, grp in enumerate(q_groups) if q_cols[qi] == cv for g in grp]
        if gc_idx:
            mc = e_c[gc_idx].mean(axis=0)
            for g in gc_idx:
                e_c[g] -= mc
    e_c = l2(e_c)
    means = np.zeros((len(q_groups), e_c.shape[1]))
    Z_all = []
    for qi, grp in enumerate(q_groups):
        means[qi] = e_c[grp].mean(axis=0)
        Z_all.append(e_c[grp] - means[qi])
    R = np.vstack(Z_all)
    W = (R.T @ R) / max(1, len(R) - len(q_groups))
    d = W.shape[0]
    W_reg = (1.0 - lam) * W + lam * (np.trace(W) / d) * np.eye(d)
    evals, evecs = np.linalg.eigh(W_reg)
    P = (evecs * (1.0 / np.sqrt(np.maximum(evals, 1e-12)))) @ evecs.T
    pw = l2((means - means.mean(axis=0, keepdims=True)) @ P)
    S = pw @ pw.T
    np.fill_diagonal(S, -1.0)
    return S


def merge(lp, S, n, thr=MERGE_THR, ndiv=MERGE_NDIV, beta=MERGE_BETA):
    """L_i += beta * sum_j ((S_ij - thr) / (1 - thr))_+ * (n_j / ndiv) * L_j  (04 lines 157-161)."""
    W_adj = np.where(S > thr, (S - thr) / (1.0 - thr), 0.0)
    np.fill_diagonal(W_adj, 0.0)
    ng_w = np.asarray(n, np.float64) / ndiv
    return np.asarray(lp, np.float64) + beta * (W_adj * ng_w[None, :]) @ np.asarray(lp, np.float64), W_adj


def groups(W_adj):
    """Connected components (size >= 2) of the merge graph -> list of sorted index lists."""
    n = len(W_adj)
    seen = np.zeros(n, bool)
    out = []
    A = W_adj > 0
    for i in range(n):
        if seen[i] or not A[i].any():
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in np.flatnonzero(A[k] | A[:, k]):
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        out.append(sorted(comp))
    return out
