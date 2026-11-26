"""prereg17 exploration (competition 2): pure numpy helpers of PLAN.md (TabICLv2 as a 7th member of Q09's ensemble).

PCA fitted on the probe only, the stratified probe context, the 7-member mean, the inner chunk plan, the gates.
No torch, no tabicl, no u1_lib here (the TabICL step runs in its own environment, p17_tabicl.py).
"""

import numpy as np

L = 10
PCA_DIM = 128
CONTEXT_PER_LEVEL = 3000
CONTEXT_SEED = 20261017
CHUNK = 2000
SMOKE_SEED = 12345
SMOKE_PER_LEVEL = 100
LSE_FLOOR = 1e-300
GATE = 0.010
TOL = 1e-12
CKPT_NAME = "tabicl-classifier-v2-20260212.ckpt"
CKPT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
WHEEL_SHA256 = "c80b8ec2719be26bc0bf1dd758a2361f58019d8f7877038b785529a425789ce4"
TABICL_VERSION = "2.2.0"
TORCH_VERSION = "2.11.0+cu128"
# PLAN.md 2.2 (everything but model_path / allow_auto_download / device / random_state is the 2.2.0 default)
TABICL_PARAMS = {"n_estimators": 8, "norm_methods": None, "feat_shuffle_method": "latin",
                 "class_shuffle_method": "shift", "outlier_threshold": 4.0, "softmax_temperature": 0.9,
                 "average_logits": True, "support_many_classes": True, "batch_size": 8, "kv_cache": False,
                 "allow_auto_download": False, "checkpoint_version": CKPT_NAME, "device": "cuda", "use_amp": "auto",
                 "use_fa3": "auto", "offload_mode": "auto", "random_state": 0}


# ---------------------------------------------------------------- PCA (PLAN.md 2.1 item 3)
def pca_moments(blocks):
    """Streaming float64 sum and sum of outer products over row blocks -> (n, mean, covariance (population))."""
    n, s, ss = 0, None, None
    for X in blocks:
        X = np.asarray(X, dtype=np.float64)
        if s is None:
            s = np.zeros(X.shape[1])
            ss = np.zeros((X.shape[1], X.shape[1]))
        n += len(X)
        s += X.sum(axis=0)
        ss += X.T @ X
    if not n:
        raise ValueError("no rows")
    mu = s / n
    C = ss / n - np.outer(mu, mu)
    return n, mu, (C + C.T) / 2.0


def pca_from_cov(C, k):
    """Top-k eigenvectors (columns, eigenvalue descending) of a symmetric matrix; each column's largest-|.| entry
    positive -> (V [d, k], eigenvalues [k], total variance)."""
    w, V = np.linalg.eigh(C)
    order = np.argsort(w, kind="stable")[::-1][:k]
    w, V = w[order], V[:, order].copy()
    for j in range(V.shape[1]):
        i = int(np.argmax(np.abs(V[:, j])))
        if V[i, j] < 0:
            V[:, j] = -V[:, j]
    return V, w, float(np.trace(C))


def pca_project(X, mu, V):
    return (np.asarray(X, dtype=np.float64) - mu) @ V


# ---------------------------------------------------------------- context (PLAN.md 2.1 item 4)
def stratified_context(y, per_level=CONTEXT_PER_LEVEL, seed=CONTEXT_SEED, levels=L):
    """Probe rows: per level 0..9 in order, rng.choice(rows of that level in storage order, per_level, no
    replacement) with one default_rng(seed); the union sorted."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    out = []
    for k in range(levels):
        rows = np.nonzero(y == k)[0]
        if len(rows) < per_level:
            raise ValueError(f"level {k}: {len(rows)} rows < {per_level}")
        out.append(rng.choice(rows, per_level, replace=False))
    return np.sort(np.concatenate(out))


def smoke_context(ctx_rows, y, per_level=SMOKE_PER_LEVEL):
    """The smoke context: the first per_level rows of each level of the registered (sorted) context."""
    y = np.asarray(y)
    keep = []
    for k in range(L):
        r = ctx_rows[y[ctx_rows] == k]
        keep.append(r[:per_level])
    return np.sort(np.concatenate(keep))


# ---------------------------------------------------------------- chunks (PLAN.md 2.2)
def chunk_bounds(n, size=CHUNK):
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def normalise_rows(P):
    P = np.asarray(P, dtype=np.float64)
    s = P.sum(axis=1, keepdims=True)
    return P / s, float(np.max(np.abs(s - 1.0))) if len(P) else 0.0


def assemble_chunks(records, n):
    """records: [(start, end, prob [end-start, 10])] in any order -> prob [n, 10]; the ranges must tile 0..n."""
    recs = sorted(records, key=lambda x: x[0])
    pos = 0
    for s, e, p in recs:
        if s != pos or e <= s or np.asarray(p).shape != (e - s, L):
            raise ValueError(f"chunk [{s}, {e}) does not continue at {pos} or has a wrong shape")
        pos = e
    if pos != n:
        raise ValueError(f"chunks cover {pos} of {n} rows")
    return np.concatenate([np.asarray(p) for _s, _e, p in recs])


# ---------------------------------------------------------------- the 7-member mean (PLAN.md 2.3)
def mean7_lp(member_lps, p_t):
    """log(max((sum of exp(member lp) + p_t) / 7, 1e-300)); member order as given (6 lps of Q09's map)."""
    if len(member_lps) != 6:
        raise ValueError("six Q09 members expected")
    ps = [np.exp(x) for x in member_lps] + [np.asarray(p_t, dtype=np.float64)]
    m = np.mean(ps, 0)
    return np.log(np.clip(m, LSE_FLOOR, None))


def lse_mean(lps):
    """p09_forward.lse_mean (for seam M)."""
    m = np.mean([np.exp(x) for x in lps], 0)
    return np.log(np.clip(m, LSE_FLOOR, None))


# ---------------------------------------------------------------- player overlap (PLAN.md 2.4)
def player_overlap(ctx_players, inner_rank, inner_pid):
    """ctx_players 'rank:player_id' strings; inner per-game rank and player_id -> (full-key overlap, id-only overlap)."""
    ctx = set(str(x) for x in ctx_players)
    inner_full = {f"{r}:{p}" for r, p in zip(inner_rank, inner_pid)}
    if not ctx or not inner_full or not {x.split(":", 1)[0] for x in ctx} <= {str(r) for r in inner_rank}:
        raise ValueError("player keys are empty or the context rank prefixes are not inner rank strings")
    ctx_ids = {x.split(":", 1)[1] for x in ctx}
    return len(ctx & inner_full), len(ctx_ids & set(str(p) for p in inner_pid))


# ---------------------------------------------------------------- gates (PLAN.md 4)
def gates(cmp_):
    by = cmp_["mono_by_colour"]
    g1 = cmp_["inner_diff"] >= GATE
    g2 = cmp_["inner_ci95"][0] > 0
    g4 = (cmp_["mono_ci95"][0] is not None and cmp_["mono_ci95"][0] > 0
          and all(by.get(c) is not None and by[c] >= -TOL for c in ("B", "W")))
    return {"G1": {"threshold": GATE, "delta": cmp_["inner_diff"], "passed": bool(g1)},
            "G2": {"ci95": cmp_["inner_ci95"], "passed": bool(g2)},
            "G4": {"mono_diff": cmp_["mono_diff"], "mono_ci95": cmp_["mono_ci95"], "by_colour": by,
                   "passed": bool(g4)},
            "passed": bool(g1 and g2 and g4)}
