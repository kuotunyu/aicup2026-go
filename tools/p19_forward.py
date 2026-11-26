"""PREREG_19 competition 2: the numpy float64 forward pass of the question-level set model S19 (lp_SET, E_SET),
shared by tools/p19_set.py (freeze, validation apply) and tools/t19_rank.py (test day)
(docs/PREREG_19_SET_MODEL.md rev 1, section 1.2; a port of results/prereg19/explore/p19_lib.py, which stays the
specification).

Pure numpy / scipy: this module never imports torch, u1_lib, hs_lib, pilot_* or p1_* (tests/test_p19_hygiene.py).

    per game   h(x) = GELU(W2 GELU(W1 x + b1) + b2)        x: the 1,540 P1 TO features standardised with Q09's frozen
                                                            p1e_map.npz mu_x / sd_x (p09_forward.standardise_x)
    question   v = mean of h over the question's usable games (sparse averaging matrix; no usable game -> E = 0)
    head       f = R2 GELU(R1 v + c1) + c2;  lp = cumulative link of p09_forward.clm_logp (t0, dt)
    ensemble   lp_SET = log(max(mean over the 3 seeds of exp(lp), 1e-300))   (p09_forward.lse_mean)
    term       E_SET = 4 x lp_SET on questions with >= 1 usable game, 0 elsewhere
Weights are stored float32 and read as float64; h is computed in blocks of 8,192 rows (as the exploration).
"""

import hashlib

import numpy as np
from scipy import sparse

import p09_forward as FW

L = FW.L
SCALE = 4.0
H_DIM = 64
BLOCK = 8192
SEEDS = (0, 1, 2)
MODEL_FORMAT = "c10-prereg19 s19 model v1"
PARAM_NAMES = ("W1", "b1", "W2", "b2", "R1", "c1", "R2", "c2", "t0", "dt")
SHAPES = {"W1": (256, FW.X_DIM), "b1": (256,), "W2": (H_DIM, 256), "b2": (H_DIM,), "R1": (32, H_DIM), "c1": (32,),
          "R2": (1, 32), "c2": (1,), "t0": (), "dt": (L - 2,)}
# PREREG_19 1.2: the three model files, byte for byte the exploration's (results/prereg19/explore/train_record.json)
MODEL_SHA256 = {0: "b0d7ea70e77d4e2ab145e42415b13a5c987c68d676c0c2a07760fb91d6859949",
                1: "8b8729ea5a0fed3948134cc1a8c8bc54302e9e3b0a8bda7d3a3945b5e70b1ffd",
                2: "b3bc704a4ba19329d99dacf0724c7a9d70f325f5dbb8d08ee6c08c122b7c9519"}
BEST_STEP = {0: 10000, 1: 8500, 2: 7000}


class SetForwardError(RuntimeError):
    pass


def model_name(seed):
    return f"s19_seed{int(seed)}.npz"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model(path, seed):
    """One S19 model file (the registered SHA-256, format, seed, step, not smoke) -> float64 parameters."""
    seed = int(seed)
    if sha256_file(path) != MODEL_SHA256.get(seed):
        raise SetForwardError(f"{path}: SHA-256 is not the registered S19 seed {seed} model (PREREG_19 1.2)")
    with np.load(path, allow_pickle=False) as z:
        if str(z["format"]) != MODEL_FORMAT or int(z["seed"]) != seed or int(z["best_step"]) != BEST_STEP[seed] \
                or bool(z["smoke"]):
            raise SetForwardError(f"{path}: not the S19 model of seed {seed}")
        par = {k: np.asarray(z[f"p_{k}"]).astype(np.float64) for k in PARAM_NAMES}
    bad = [k for k in PARAM_NAMES if par[k].shape != SHAPES[k]]
    if bad:
        raise SetForwardError(f"{path}: parameter shapes {bad} differ")
    return par


def load_models(folder):
    """The three seeds in order -> [params]."""
    import os
    return [load_model(os.path.join(folder, model_name(s)), s) for s in SEEDS]


def h_np(par, Xs):
    """Per-game 64-dim representation; Xs float64 [N, 1540] standardised (dropout off)."""
    a = FW.gelu(Xs @ par["W1"].T + par["b1"])
    return FW.gelu(a @ par["W2"].T + par["b2"])


def head_np(par, V):
    """Pooled representation [Q, 64] -> log P(y = k) [Q, 10]."""
    f = FW.gelu(V @ par["R1"].T + par["c1"]) @ par["R2"].T + par["c2"]
    return FW.clm_logp(f.reshape(-1, 1), par["t0"], par["dt"])


def game_h(par_list, Xraw, mu_x, sd_x, block=BLOCK):
    """h of every seed for raw TO features (float64 or float32 [N, 1540], NaN opponent halves) -> [[N, 64]]."""
    n = len(Xraw)
    H = [np.zeros((n, H_DIM)) for _ in par_list]
    for s in range(0, n, block):
        Xs = FW.standardise_x(np.asarray(Xraw[s:s + block], dtype=np.float64), mu_x, sd_x)
        for par, Hs in zip(par_list, H):
            Hs[s:s + block] = h_np(par, Xs)
    return H


def usable_groups(S, usable):
    """Rows of a pooling matrix S (csr [Q, N]) -> per question the usable column indices (order kept)."""
    S = S.tocsr()
    usable = np.asarray(usable, dtype=bool)
    return [[int(c) for c in S.indices[S.indptr[q]:S.indptr[q + 1]] if usable[c]] for q in range(S.shape[0])]


def mean_matrix(groups, n_cols):
    """Sparse averaging matrix (= p02_rank_fuse.pool_matrix on the usable games; empty row for an empty group)."""
    rows, cols, vals = [], [], []
    for q, g in enumerate(groups):
        if len(g):
            rows.extend([q] * len(g))
            cols.extend(g)
            vals.extend([1.0 / len(g)] * len(g))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), n_cols))


def set_term(par_list, H_list, groups, n_cols, scale=SCALE):
    """E_SET = scale x lse_mean over the seeds of head(mean of usable h) per question; 0 without a usable game.
    H_list[s]: [n_cols, 64] per-game h of seed s. -> (E [Q, 10], [E per seed], lp [Q, 10], has [Q])."""
    M = mean_matrix(groups, n_cols)
    has = np.array([len(g) > 0 for g in groups], dtype=bool)
    lps = [head_np(par, np.asarray(M @ H)) for par, H in zip(par_list, H_list)]
    lp = FW.lse_mean(lps)
    E = np.zeros((len(groups), L))
    E[has] = scale * lp[has]
    per_seed = []
    for x in lps:
        e = np.zeros((len(groups), L))
        e[has] = scale * x[has]
        per_seed.append(e)
    return E, per_seed, lp, has


def question_set_term(par_list, means, counts, usable, S, mu_x, sd_x, scale=SCALE):
    """E_SET of the questions of one set from its P1 chunk means (float32 [N, 2, 770]), counts and the usable games
    (C5 parsed, P1 not failed, >= 1 target move); S the [Q, N] pooling matrix of the set. Only usable rows are run
    through the network. -> (E, lp, has)."""
    usable = np.asarray(usable, dtype=bool)
    n = len(usable)
    idx = np.flatnonzero(usable)
    H = [np.zeros((n, H_DIM)) for _ in par_list]
    if len(idx):
        Hu = game_h(par_list, FW.x_raw(np.asarray(means)[idx], np.asarray(counts)[idx]), mu_x, sd_x)
        for Hs, hu in zip(H, Hu):
            Hs[idx] = hu
    E, _per, lp, has = set_term(par_list, H, usable_groups(S, usable), n, scale)
    return E, lp, has


def single_game_logp(par_list, Xraw, mu_x, sd_x):
    """lp_SET of every game as a one-game question (the GPU seam record, PREREG_19 4.2 step 2) -> [N, 10]."""
    H = game_h(par_list, Xraw, mu_x, sd_x)
    return FW.lse_mean([head_np(par, h) for par, h in zip(par_list, H)])
