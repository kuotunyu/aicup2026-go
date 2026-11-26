"""prereg16 exploration (competition 2): consistency / tail features (PLAN.md sections 2 and 3). Pure numpy / scipy.

Per-move series (target side) -> per-game statistics -> question-level 73 features; multinomial logistic regression
M; nested player-fold cross-fitting (M of outer fold f never trains on fold f's players; the fusion's CT inputs for
fold g != f come from M trained without folds f and g).
"""

import numpy as np
from scipy import optimize, sparse

L = 10
N_PROF = 14
SERIES = ("h_rank", "h_slope", "h_acc", "p_logp", "p_surp")
HIGH_IS_BAD = {"p_surp"}
PHASES = ((0, 60), (60, 150), (150, None))
WORST_GAME = 5
WORST_Q = 10
LOW_P = float(np.log(0.01))
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)
MAXITER = 1000
N_FOLDS = 5
PER_SERIES = ("std", "q10", "q90", "worst5", "ph_open", "ph_mid", "ph_late")
GAME_STAT_NAMES = [f"{s}_{x}" for s in SERIES for x in PER_SERIES] + ["frac_h_acc_lt_log001", "frac_p_logp_lt_log001"]
N_GAME_STATS = len(GAME_STAT_NAMES)   # 37
A_NAMES = [f"A_{x}" for x in GAME_STAT_NAMES]
B_NAMES = [f"B_{s}_worst10" for s in SERIES] + [f"B_{s}_pooled_std" for s in SERIES]
C_NAMES = ([f"C_{s}_gamemean_std" for s in SERIES] + ["C_eHS_std", "C_eP1E_std", "C_eHS_range", "C_eP1E_range",
                                                        "C_eP1E_min", "C_eP1E_max"])
D_NAMES = ["D_eHS_mean", "D_eP1E_mean", "D_log_games", "D_mean_log_moves"]
E_NAMES = [f"E_{c}_x_eP1E" for c in C_NAMES]
BASE_NAMES = A_NAMES + B_NAMES + C_NAMES + D_NAMES          # 62
ALL_NAMES = BASE_NAMES + E_NAMES                            # 73
C_IDX = np.arange(len(A_NAMES) + len(B_NAMES), len(A_NAMES) + len(B_NAMES) + len(C_NAMES))
ANCHOR_IDX = BASE_NAMES.index("D_eP1E_mean")
assert N_GAME_STATS == 37 and len(BASE_NAMES) == 62 and len(ALL_NAMES) == 73


# ---------------------------------------------------------------- per-move series
def hs_series(logp14):
    """[T, 14] HS log p -> (h_rank, h_slope, h_acc), float64."""
    x = np.asarray(logp14, dtype=np.float64)
    z = x - x.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    h_rank = p @ np.arange(N_PROF, dtype=np.float64)
    return h_rank, x[:, N_PROF - 1] - x[:, 0], x[:, N_PROF - 1].copy()


def p1_series(logp, entropy):
    lp = np.asarray(logp, dtype=np.float64)
    return lp, -lp - np.asarray(entropy, dtype=np.float64)


def _bad_view(x, name):
    """Values oriented so that the worst moves are the smallest."""
    return -x if name in HIGH_IS_BAD else x


def game_stats(series, turns):
    """series: dict name -> 1-d array of one game's target moves (h_* share turns['h'], p_* share turns['p']).
    -> (37 stats, means [5], sums [5], sumsq [5], counts [5], worst10 [5, 10] in the bad view, NaN padded)."""
    st = np.full(N_GAME_STATS, np.nan)
    means, sums, sumsq, cnts = np.full(5, np.nan), np.zeros(5), np.zeros(5), np.zeros(5)
    worst = np.full((5, WORST_Q), np.nan)
    j = 0
    for i, s in enumerate(SERIES):
        x = np.asarray(series[s], dtype=np.float64)
        t = np.asarray(turns["h" if s.startswith("h_") else "p"])
        n = len(x)
        if n:
            means[i] = x.mean()
            sums[i], sumsq[i], cnts[i] = x.sum(), float(np.sum(x * x)), n
            st[j] = x.std()
            st[j + 1] = np.percentile(x, 10)
            st[j + 2] = np.percentile(x, 90)
            b = np.sort(_bad_view(x, s))
            w = b[:WORST_GAME].mean()
            st[j + 3] = -w if s in HIGH_IS_BAD else w
            k = min(WORST_Q, n)
            worst[i, :k] = b[:k]
            for ph, (lo, hi) in enumerate(PHASES):
                m = (t >= lo) if hi is None else ((t >= lo) & (t < hi))
                if m.any():
                    st[j + 4 + ph] = x[m].mean()
        j += len(PER_SERIES)
    ha, pl = np.asarray(series["h_acc"]), np.asarray(series["p_logp"])
    st[j] = np.mean(ha < LOW_P) if len(ha) else np.nan
    st[j + 1] = np.mean(pl < LOW_P) if len(pl) else np.nan
    return st, means, sums, sumsq, cnts, worst


def expected_level(lp):
    lp = np.asarray(lp, dtype=np.float64)
    z = lp - lp.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p @ np.arange(L, dtype=np.float64)


# ---------------------------------------------------------------- per-game table
class GameTable:
    """Per-game arrays over all inner rows (NaN / 0 where not filled)."""

    def __init__(self, n):
        self.n = n
        self.stats = np.full((n, N_GAME_STATS), np.nan)
        self.means = np.full((n, 5), np.nan)
        self.sums = np.zeros((n, 5))
        self.sumsq = np.zeros((n, 5))
        self.cnts = np.zeros((n, 5))
        self.worst = np.full((n, 5, WORST_Q), np.nan)
        self.filled_h = np.zeros(n, dtype=bool)
        self.filled_p = np.zeros(n, dtype=bool)
        self.e_hs = np.full(n, np.nan)
        self.e_p1e = np.full(n, np.nan)

    def put(self, row, series, turns):
        st, mu, s1, s2, c, w = game_stats(series, turns)
        self.stats[row], self.means[row], self.sums[row], self.sumsq[row] = st, mu, s1, s2
        self.cnts[row], self.worst[row] = c, w


def per_game_from_moves(h_moves, p_moves, rows, table):
    """h_moves[row] = (logp14 [T, 14], turns [T]); p_moves[row] = (logp [T'], entropy [T'], turns [T']) for target
    moves; fills table for the given rows (both present)."""
    for row in rows:
        lp14, th = h_moves[row]
        lpp, ent, tp = p_moves[row]
        hr, hsl, hac = hs_series(lp14)
        pl, ps = p1_series(lpp, ent)
        table.put(row, {"h_rank": hr, "h_slope": hsl, "h_acc": hac, "p_logp": pl, "p_surp": ps},
                  {"h": th, "p": tp})


# ---------------------------------------------------------------- question level
def groups_of(S):
    """CSR averaging matrix -> list of game-row arrays per question."""
    S = sparse.csr_matrix(S)
    return [S.indices[S.indptr[i]:S.indptr[i + 1]].astype(np.int64) for i in range(S.shape[0])]


def padded(groups, usable):
    """-> [Q, Gmax] game rows of usable games, -1 padded."""
    gs = [g[usable[g]] for g in groups]
    gmax = max(1, max((len(g) for g in gs), default=1))
    out = np.full((len(gs), gmax), -1, dtype=np.int64)
    for i, g in enumerate(gs):
        out[i, :len(g)] = g
    return out


def _gather(arr, idx):
    """arr [n, ...] at idx [Q, G] (-1 -> NaN)."""
    a = np.asarray(arr, dtype=np.float64)
    out = a[np.where(idx < 0, 0, idx)]
    out[idx < 0] = np.nan
    return out


def _nanmean0(x, axis):
    cnt = np.sum(~np.isnan(x), axis=axis)
    s = np.nansum(x, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def question_features(groups, usable, T, block=4096):
    """-> (F [Q, 62] base features with NaN where undefined, n_usable [Q])."""
    idx_all = padded(groups, usable)
    Q = len(groups)
    F = np.full((Q, len(BASE_NAMES)), np.nan)
    nu = np.sum(idx_all >= 0, axis=1)
    logm = np.where(T.cnts[:, 0] > 0, np.log(np.maximum(T.cnts[:, 0], 1)), np.nan)
    for s0 in range(0, Q, block):
        sl = slice(s0, min(Q, s0 + block))
        idx = idx_all[sl]
        has = (idx >= 0).any(axis=1)
        # A: mean over usable games of the 37 per-game stats (NaN ignored)
        A = _nanmean0(_gather(T.stats, idx), axis=1)
        # B: worst 10 over the union of the games' worst-10 lists; pooled std
        W = _gather(T.worst, idx)                                   # [q, G, 5, 10]
        W = np.transpose(W, (0, 2, 1, 3)).reshape(len(idx), 5, -1)  # [q, 5, G*10]
        W = np.sort(W, axis=2)[:, :, :WORST_Q]                      # NaN sort last
        Bw = _nanmean0(W, axis=2)
        for i, s in enumerate(SERIES):
            if s in HIGH_IS_BAD:
                Bw[:, i] = -Bw[:, i]
        s1 = np.nansum(_gather(T.sums, idx), axis=1)
        s2 = np.nansum(_gather(T.sumsq, idx), axis=1)
        c = np.nansum(_gather(T.cnts, idx), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = s1 / c
            Bs = np.sqrt(np.maximum(s2 / c - mu * mu, 0.0))
        Bs[c <= 0] = np.nan
        # C: across games
        gm = _gather(T.means, idx)                                  # [q, G, 5]
        eh = _gather(T.e_hs, idx)
        ep = _gather(T.e_p1e, idx)
        C = _c_safe(gm, eh, ep)
        D = np.column_stack([_nanmean0(eh, 1), _nanmean0(ep, 1), np.log(np.maximum(nu[sl], 1)).astype(np.float64),
                             _nanmean0(_gather(logm, idx), 1)])
        D[~has] = np.nan
        F[sl] = np.column_stack([A, Bw, Bs, C, D])
    return F, nu


def _c_safe(gm, eh, ep):
    """C block when some questions have no usable game (rows of all-NaN stay NaN without warnings)."""
    q = gm.shape[0]
    out = np.full((q, len(C_NAMES)), np.nan)
    ok = ~np.all(np.isnan(ep), axis=1)
    if ok.any():
        out[ok] = np.column_stack([np.nanstd(gm[ok], axis=1), np.nanstd(eh[ok], axis=1), np.nanstd(ep[ok], axis=1),
                                   np.nanmax(eh[ok], axis=1) - np.nanmin(eh[ok], axis=1),
                                   np.nanmax(ep[ok], axis=1) - np.nanmin(ep[ok], axis=1),
                                   np.nanmin(ep[ok], axis=1), np.nanmax(ep[ok], axis=1)])
    return out


# ---------------------------------------------------------------- standardisation and design matrix
def std_fit(F):
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(F, axis=0)
            sd = np.nanstd(F, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return mu, sd


def design(F, mu, sd, interactions=True):
    Z = (F - mu) / sd
    Z = np.where(np.isnan(Z), 0.0, Z)
    if interactions:
        Z = np.column_stack([Z, Z[:, C_IDX] * Z[:, [ANCHOR_IDX]]])
    return Z


# ---------------------------------------------------------------- multinomial logistic regression M
def _lsm(z):
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def level_weights(y):
    present = np.unique(y)
    cnt = np.bincount(y, minlength=L).astype(np.float64)
    return 1.0 / (len(present) * cnt[y])


def lr_obj(th, Z, y, wq, lam):
    d = Z.shape[1]
    W = th[:d * L].reshape(d, L)
    b = th[d * L:]
    ls = _lsm(Z @ W + b)
    sw = wq.sum()
    Q = len(y)
    loss = -np.sum(wq * ls[np.arange(Q), y]) / sw + lam * float(np.sum(W * W))
    G = np.exp(ls)
    G[np.arange(Q), y] -= 1.0
    G *= (wq / sw)[:, None]
    gW = Z.T @ G + 2.0 * lam * W
    return loss, np.concatenate([gW.ravel(), G.sum(axis=0)])


def lr_fit(Z, y, lam):
    wq = level_weights(y)
    th0 = np.zeros(Z.shape[1] * L + L)
    r = optimize.minimize(lr_obj, th0, args=(Z, y, wq, lam), jac=True, method="L-BFGS-B",
                          options={"maxiter": MAXITER})
    return {"theta": r.x, "d": Z.shape[1], "loss": float(r.fun), "nit": int(r.nit), "success": bool(r.success)}


def lr_logp(m, Z):
    d = m["d"]
    return _lsm(Z @ m["theta"][:d * L].reshape(d, L) + m["theta"][d * L:])


def balanced_nll(lp, y):
    wq = level_weights(y)
    return float(-np.sum(wq * lp[np.arange(len(y)), y]) / wq.sum())


def fit_predict(F_tr, y_tr, lam, F_out_list, interactions=True):
    mu, sd = std_fit(F_tr)
    m = lr_fit(design(F_tr, mu, sd, interactions), y_tr, lam)
    return m, [lr_logp(m, design(F, mu, sd, interactions)) for F in F_out_list]


# ---------------------------------------------------------------- nested cross-fitting (PLAN.md 3.2)
def nested_crossfit(Ff, yf, foldf, playerf, Fe, folde, playere, Fm, foldm, playerm, lambdas=LAMBDAS,
                    interactions=True, log=None):
    """-> dict: ct_fit[f] [n_fit, 10] (rows of fold f NaN), ct_eval [n_eval, 10], ct_mono [n_mono, 10], records."""
    nf, ne, nm = len(yf), len(folde), len(foldm)
    ct_fit = {}
    ct_eval = np.full((ne, L), np.nan)
    ct_mono = np.full((nm, L), np.nan)
    recs = []
    for f in range(N_FOLDS):
        cv = {}
        preds = {}
        for lam in lambdas:
            P = np.full((nf, L), np.nan)
            for g in range(N_FOLDS):
                if g == f:
                    continue
                tr = (foldf != f) & (foldf != g)
                ev = foldf == g
                assert not (set(playerf[tr]) & set(playerf[ev])), "nested M: training and predicted players overlap"
                assert not (set(playerf[tr]) & set(playerf[foldf == f])), "nested M: trained on fold f players"
                _m, (lp,) = fit_predict(Ff[tr], yf[tr], lam, [Ff[ev]], interactions)
                P[ev] = lp
            mask = foldf != f
            cv[lam] = balanced_nll(P[mask], yf[mask])
            preds[lam] = P
        best = min(lambdas, key=lambda l_: (cv[l_], -l_))
        ct_fit[f] = preds[best]
        tr = foldf != f
        ev, mv = folde == f, foldm == f
        pl_tr = set(playerf[tr])
        assert not (pl_tr & set(playere[ev])) and not (pl_tr & set(playerm[mv])), "M_f trained on fold f players"
        m, (lpe, lpm) = fit_predict(Ff[tr], yf[tr], best, [Fe[ev], Fm[mv]], interactions)
        ct_eval[ev] = lpe
        ct_mono[mv] = lpm
        recs.append({"fold": f, "cv_balanced_nll": {str(k): v for k, v in cv.items()}, "lambda": best,
                     "train_questions": int(tr.sum()), "eval_questions": int(ev.sum()),
                     "mono_questions": int(mv.sum()), "final_fit": {k: m[k] for k in ("loss", "nit", "success")}})
        if log:
            log(f"M fold {f}: lambda {best}, cv {cv[best]:.5f}, nit {m['nit']}")
    if np.isnan(ct_eval).any() or np.isnan(ct_mono).any():
        raise RuntimeError("a question without a cross-fitted CT")
    return {"ct_fit": ct_fit, "ct_eval": ct_eval, "ct_mono": ct_mono, "folds": recs}


# ---------------------------------------------------------------- fusion fold loop (PLAN.md 3.2 steps 2-4)
def fold_loop(p4, rf, Xf, Xe, Xm, mf, me, mm, Ef, Ee, Em, ct=None):
    """Q09's generalised calibration with each fold fitted on folds != f (p4.gen_fit_record), predicting the eval
    and single-colour questions of fold f. ct=None reproduces p4.gen_oof_record / gen_fold_predict."""
    P = np.zeros((len(Xe.n), L))
    Pm = np.full((len(Xm.n), L), np.nan)
    folds = []
    for f in range(N_FOLDS):
        tr = Xf.fold != f
        ev = Xe.fold == f
        mv = Xm.fold == f
        Ef_f = [E[tr] for E in Ef]
        Ee_f = [E[ev] for E in Ee]
        Em_f = [E[mv] for E in Em]
        if ct is not None:
            if np.isnan(ct["ct_fit"][f][tr]).any():
                raise RuntimeError(f"fold {f}: a fit question without a nested CT")
            Ef_f.append(ct["ct_fit"][f][tr])
            Ee_f.append(ct["ct_eval"][ev])
            Em_f.append(ct["ct_mono"][mv])
        th, rec = p4.gen_fit_record(mf[tr], Ef_f, Xf.n[tr], Xf.y[tr])
        z, _ = p4.gen_logits(th, me[ev], Ee_f, Xe.n[ev])
        P[ev] = rf.softmax(z)
        if mv.any():
            zm, _ = p4.gen_logits(np.asarray([float(x) for x in th], dtype=np.float64), mm[mv], Em_f, Xm.n[mv])
            Pm[mv] = rf.softmax(zm)
        rec.update({"fold": f, "fit_questions": int(tr.sum()), "eval_questions": int(ev.sum()),
                    "theta": [float(x) for x in th]})
        folds.append(rec)
    return P, Pm, folds
