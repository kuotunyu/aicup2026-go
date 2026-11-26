"""prereg23 exploration (competition 2): pure numpy / scipy parts of PLAN.md (second question-level set model SET2).

SET2 = S19 of prereg19 with (1) per-game input [X 1,540, EX 44] = 1,584, (2) question pooling concat(mean, population
std) of the 64-dim h -> 128, (3) 5 seeds. The sampler, held-out construction and score helpers are imported from
results/prereg19/explore/p19_lib.py (unchanged). No torch here.
"""

import hashlib
import os
import sys

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P19_DIR = os.path.join(REPO, "results", "prereg19", "explore")
for p in (os.path.join(REPO, "tools"), P19_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import p09_forward as FW  # noqa: E402
import p19_lib as K19  # noqa: E402

L = K19.L
KS = K19.KS
COLOURS = K19.COLOURS
BATCH = K19.BATCH
STEPS = K19.STEPS
EVAL_EVERY = K19.EVAL_EVERY
HOLD_PER = K19.HOLD_PER
HOLD_SEED = 23023
SEEDS = (0, 1, 2, 3, 4)
SAMPLER_BASE = 23
X_DIM = FW.X_DIM
NEX = FW.NEX
D_IN = X_DIM + NEX
HIDDEN = K19.HIDDEN
H_DIM = K19.H_DIM
POOL_DIM = 2 * H_DIM
RHO = K19.RHO
DROP_IN = K19.DROP_IN
DROP = K19.DROP
LR = K19.LR
WD = K19.WD
PCT_START = K19.PCT_START
THREADS = K19.THREADS
SCALE = K19.SCALE
GATE = 0.005
TOL = 1e-12
TORCH_TOL = K19.TORCH_TOL
SMOKE = dict(K19.SMOKE)
PARAM_NAMES = K19.PARAM_NAMES
SALT = "c10-prereg23|holdout|"

# re-exported from prereg19 (unchanged)
PlayerPool = K19.PlayerPool
sample_question = K19.sample_question
sample_batch = K19.sample_batch
holdout_set = K19.holdout_set
official_score = K19.official_score
rank_balanced = K19.rank_balanced
per_level = K19.per_level
usable_groups = K19.usable_groups
mean_matrix = K19.mean_matrix
lse_mean = K19.lse_mean
p64 = K19.p64


# ---------------------------------------------------------------- players (PLAN.md 2)
def holdout_bit(key):
    """1 = H (held out), 0 = A (training); key = '<rank>:<player_id>'."""
    h = hashlib.sha256(f"{SALT}{key}".encode("utf-8")).hexdigest()
    return int(int(h[:8], 16) % 10 == 0)


# ---------------------------------------------------------------- features (PLAN.md 2)
def features(Xraw, EXraw, mp):
    """[standardise_x(X), standardise_ex(EX) with non-finite -> 0] -> float64 [N, 1584]."""
    X = FW.standardise_x(np.asarray(Xraw, dtype=np.float64), mp["mu_x"], mp["sd_x"])
    EX = FW.standardise_ex(EXraw, mp["mu_ex"], mp["sd_ex"])
    EX[~np.isfinite(EX)] = 0.0
    return np.concatenate([X, EX], axis=1)


# ---------------------------------------------------------------- numpy forward (PLAN.md 3.6)
def h_np(par, X):
    """Per-game 64-dim representation; X float64 [N, 1584] standardised."""
    a = FW.gelu(X @ par["W1"].T + par["b1"])
    return FW.gelu(a @ par["W2"].T + par["b2"])


def head_np(par, V):
    """Pooled representation [Q, 128] -> log P(y = k) [Q, 10]."""
    f = FW.gelu(V @ par["R1"].T + par["c1"]) @ par["R2"].T + par["c2"]
    return FW.clm_logp(f.reshape(-1, 1), par["t0"], par["dt"])


def mean_std_dense(H):
    """[Q, n, 64] -> [Q, 128] = [mean, population std] (two-pass)."""
    m = H.mean(axis=1)
    v = ((H - m[:, None, :]) ** 2).mean(axis=1)
    return np.concatenate([m, np.sqrt(v)], axis=1)


def set_logp_np(par, X_games):
    """Questions [Q, n, 1584] float64 -> lp [Q, 10]."""
    Q, n, d = X_games.shape
    H = h_np(par, X_games.reshape(Q * n, d)).reshape(Q, n, -1)
    return head_np(par, mean_std_dense(H))


def pool_mean_std(groups, H):
    """Per question the [mean, population std] of H rows in the group (two-pass); zeros for an empty group.
    groups: list of column-index lists; H [n_cols, 64] -> [Q, 128]."""
    Q, D = len(groups), H.shape[1]
    lens = np.array([len(g) for g in groups], dtype=np.int64)
    out = np.zeros((Q, 2 * D))
    if lens.sum() == 0:
        return out
    cols = np.concatenate([np.asarray(g, dtype=np.int64) for g in groups if len(g)])
    qid = np.repeat(np.arange(Q), lens)
    M = sparse.csr_matrix((np.ones(len(cols)), (qid, cols)), shape=(Q, H.shape[0]))
    has = lens > 0
    s = np.asarray(M @ H)
    mean = np.zeros((Q, D))
    mean[has] = s[has] / lens[has, None]
    dev = (H[cols] - mean[qid]) ** 2
    ss = np.zeros((Q, D))
    np.add.at(ss, qid, dev)
    var = np.zeros((Q, D))
    var[has] = ss[has] / lens[has, None]
    out[:, :D] = mean
    out[:, D:] = np.sqrt(var)
    return out


def set_term(par_list, H_list, groups, scale=SCALE):
    """E_SET2 = scale * lse_mean over seeds of head([mean, std] of usable h) per question; 0 without a usable game."""
    has = np.array([len(g) > 0 for g in groups], dtype=bool)
    lps = [head_np(par, pool_mean_std(groups, H)) for par, H in zip(par_list, H_list)]
    lp = lse_mean(lps)
    E = np.zeros((len(groups), L))
    E[has] = scale * lp[has]
    per_seed = []
    for x in lps:
        e = np.zeros((len(groups), L))
        e[has] = scale * x[has]
        per_seed.append(e)
    return E, per_seed, lp, has


# ---------------------------------------------------------------- gates (PLAN.md 6)
def gates(inner_diff, inner_ci, mono_diff, mono_ci, by_colour):
    g1 = inner_diff >= GATE
    g2 = inner_ci[0] > 0
    g4 = (mono_ci[0] is not None and mono_ci[0] > 0
          and all(by_colour.get(c) is not None and by_colour[c] >= -TOL for c in COLOURS))
    return {"G1": {"threshold": GATE, "delta": inner_diff, "passed": bool(g1)},
            "G2": {"ci95": inner_ci, "passed": bool(g2)},
            "G4": {"mono_diff": mono_diff, "mono_ci95": mono_ci, "by_colour": by_colour, "passed": bool(g4)},
            "passed": bool(g1 and g2 and g4)}
