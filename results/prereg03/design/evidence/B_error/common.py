"""Shared loader for the task-B error analysis (prereg 03 prep). Read-only on the repo.

Reads only inner-side prereg02 artifacts: dumps/inner_*.npz, dumps/inner_keys.npz,
pseudo_inner.json, opening_inner.npz, frozen_config.json, tune_report.json.
Never reads validation dumps, data/local/*answers*, or runs score.py.
"""
import math
import os
import sys

sys.dont_write_bytecode = True  # do not drop .pyc files into the repo's tools/
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402

import p02_rank_fuse as F  # noqa: E402
from c10data import LEVELS  # noqa: E402
from p02_rank_opening import load_opening  # noqa: E402
from p02_rank_pseudo import load_inner_keys, load_pseudo, str_array  # noqa: E402

RANK = os.path.join(REPO, "results", "prereg02", "rank")
DDIR = os.path.join(RANK, "dumps")
OUT = os.path.dirname(os.path.abspath(__file__))
E1 = math.exp(-1)
L = 10
KYU = [0, 1, 2, 3]
DAN = [4, 5, 6, 7, 8, 9]


def load_all(dump_names=("p01e1", "p01e2", "p01e3", "p01e4", "p01e5_d4", "p02e9_d4")):
    keys = load_inner_keys(DDIR)
    pseudo = load_pseudo(os.path.join(RANK, "pseudo_inner.json"), verify=True)
    players = pseudo["players"]
    dump_row = {(r, g): i for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"]))}
    player_bank = [[(p["rank"], gid) for gid in p["bank"]] for p in players]
    fold_of = [p["fold"] for p in players]
    op = load_opening(os.path.join(RANK, "opening_inner.npz"), "inner")
    op_row = {(r, g): i for i, (r, g) in enumerate(zip(str_array(op["rank"]), str_array(op["game_id"])))}
    idx = np.asarray([op_row[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
    op_o = np.asarray(op["o"], dtype=np.float64)[idx]
    op_has = np.asarray(op["has"], dtype=bool)[idx]
    op_nterms = np.asarray(op["n_terms"])[idx]
    dumps = {}
    for name in dump_names:
        dumps[name] = F.load_dump(os.path.join(DDIR, f"inner_{name}.npz"), keys["n"], F.DUMP_S[name])["logits"]
    X_fit, S_fit = F.build_qset(pseudo["fit"], players, dump_row, fold_of, keys["n"], op_o, op_has, player_bank)
    X_eval, S_eval = F.build_qset(pseudo["eval"], players, dump_row, fold_of, keys["n"], op_o, op_has, player_bank)
    return dict(keys=keys, pseudo=pseudo, players=players, dump_row=dump_row, op_o=op_o, op_has=op_has,
                op_nterms=op_nterms, dumps=dumps, X_fit=X_fit, S_fit=S_fit, X_eval=X_eval, S_eval=S_eval)


def groups_of(S):
    return [S.indices[S.indptr[q]:S.indptr[q + 1]] for q in range(S.shape[0])]


def lsm(x):
    return F.log_softmax(x)


def sym_source(dump, syms):
    """log_softmax(mean_s log_softmax(z_s)) over the chosen symmetries (float64)."""
    z = dump.astype(np.float64)
    parts = [lsm(z[:, s, :]) for s in syms]
    return parts[0] if len(parts) == 1 else lsm(np.mean(parts, axis=0))


def mix_source(lps):
    return lsm(np.mean(lps, axis=0))


# ---------------------------------------------------------------- generalised calibration
# z = e^a (n/12)^g (m + sum_j w_j E_j + B beta); level-weighted NLL + 0.01||beta||^2 (same as p02)
B = F.basis()


def _unpack(th, J):
    return th[0], th[1], th[2:2 + J], th[2 + J:]


def gen_logits(th, m, Es, n):
    J = len(Es)
    a, g, w, beta = _unpack(th, J)
    t = math.exp(a) * np.power(n / 12.0, g)
    u = m + B @ beta
    for j in range(J):
        u = u + w[j] * Es[j]
    return t[:, None] * u, t


def gen_obj(th, m, Es, n, y, wq):
    J = len(Es)
    z, t = gen_logits(th, m, Es, n)
    ls = lsm(z)
    Q = len(y)
    a, g, w, beta = _unpack(th, J)
    loss = -np.sum(wq * ls[np.arange(Q), y]) + F.PENALTY * float(beta @ beta)
    G = np.exp(ls)
    G[np.arange(Q), y] -= 1.0
    G *= wq[:, None]
    gz = np.sum(G * z, axis=1)
    grad = [np.sum(gz), np.sum(gz * np.log(n / 12.0))]
    for j in range(J):
        grad.append(np.sum(t * np.sum(G * Es[j], axis=1)))
    gb = (t[:, None] * G).sum(0)
    return loss, np.concatenate([grad, B.T @ gb + 2 * F.PENALTY * beta])


def gen_fit(m, Es, n, y, wbounds=None):
    J = len(Es)
    wq = F.level_weights(y)
    wb = wbounds or [(0.0, 4.0)] * J
    bnds = [F.BOUNDS["a"], F.BOUNDS["gamma"]] + wb + [(None, None)] * (L - 1)
    best = None
    for a0 in F.STARTS:
        x0 = np.zeros(2 + J + L - 1)
        x0[0] = a0
        r = optimize.minimize(gen_obj, x0, args=(m, Es, n, y, wq), jac=True, method="L-BFGS-B",
                              bounds=bnds, options={"maxiter": F.MAXITER})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def gen_predict(th, m, Es, n):
    z, _ = gen_logits(th, m, Es, n)
    return F.softmax(z)


def oof_gen(Xf, Xe, mf, Ef, me, Ee, wbounds=None, fit_on="fit"):
    """5-fold OOF: fit on questions of players not in fold f, predict eval questions of fold f.
    fit_on='fit' uses the fit rounds (as tune); 'eval' uses eval rounds of the other folds."""
    P = np.zeros((len(Xe), L))
    thetas = []
    for f in range(5):
        ev = Xe.fold == f
        if fit_on == "fit":
            tr = Xf.fold != f
            th = gen_fit(mf[tr], [E[tr] for E in Ef], Xf.n[tr], Xf.y[tr], wbounds)
        else:
            tr = Xe.fold != f
            th = gen_fit(me[tr], [E[tr] for E in Ee], Xe.n[tr], Xe.y[tr], wbounds)
        P[ev] = gen_predict(th, me[ev], [E[ev] for E in Ee], Xe.n[ev])
        thetas.append(th)
    return P, thetas


# ---------------------------------------------------------------- scores
def rule_b(P):
    return F.rule_b_np(P)


def exp_scores(P):
    """S_k = p_k + e^-1 (p_{k-1} + p_{k+1})."""
    S = P.copy()
    S[:, 1:] += E1 * P[:, :-1]
    S[:, :-1] += E1 * P[:, 1:]
    return S


def score(pred, y):
    return F.official_score(pred, y)


def rank_balanced(sc, y):
    return float(np.mean([sc[y == i].mean() for i in range(L) if (y == i).any()]))


def per_level(sc, y):
    return {LEVELS[i]: float(sc[y == i].mean()) for i in range(L) if (y == i).any()}


class Boot:
    """Players resampled within level (1000x, seed 0), same as tune's PlayerBootstrap."""

    def __init__(self, player, y, n_players, n_boot=1000, seed=0):
        self.player, self.n_players = player, n_players
        level_of = np.full(n_players, -1)
        level_of[player] = y
        rng = np.random.default_rng(seed)
        self.draws = []
        for i in range(L):
            mem = np.nonzero(level_of == i)[0]
            if len(mem):
                self.draws.append(mem[rng.integers(0, len(mem), size=(n_boot, len(mem)))])
        self.count = np.bincount(player, minlength=n_players).astype(np.float64)

    def per_level_stats(self, sc):
        s = np.bincount(self.player, weights=sc, minlength=self.n_players)
        return np.stack([s[d].sum(1) / self.count[d].sum(1) for d in self.draws], axis=1)  # [n_boot, L]

    def diff(self, sc, sc_ref):
        """Paired bootstrap of rank-balanced(sc) - rank-balanced(sc_ref); per-level CIs too."""
        a, b = self.per_level_stats(sc), self.per_level_stats(sc_ref)
        d = a.mean(1) - b.mean(1)
        dl = a - b
        return {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "frac_le0": float(np.mean(d <= 0)),
                "per_level_ci95": {LEVELS[i]: [float(np.percentile(dl[:, i], 2.5)),
                                               float(np.percentile(dl[:, i], 97.5))]
                                   for i in range(dl.shape[1])}}


def rank_balanced_from_stats(st):
    return float(np.mean(st.mean(0)))


def r4(x):
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x
