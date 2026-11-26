"""prereg38 explore (competition 1, v2 new pools): scoring-backend modules on top of the prereg36 metric CNN
(PLAN.md). CPU only. This round touches the D pools only (seeds 100-109); nothing here loads the E pools.

    PY results/prereg38/explore/p38_backend.py seam                 # B0 == results/prereg36/explore/tune.json
    PY results/prereg38/explore/p38_backend.py single               # every module alone -> data/prereg38/single.json
    PY results/prereg38/explore/p38_backend.py greedy               # assembly -> results/prereg38/explore/tune_D.json
    PY results/prereg38/explore/p38_backend.py halfcheck            # D-only diagnostic -> halfcheck_D.json
    PY results/prereg38/explore/p38_backend.py variant --kept B4 --out results/prereg38/explore/tune_D_B4.json
    PY results/prereg38/explore/p38_backend.py report               # -> results/prereg38/explore/ABLATION_D.md
  For the main session, NOT run in the D-only round (PLAN.md 6):
    PY ... retune --emb-D <store_D.npz of the adopted checkpoint> --base-tune <its tune.json> --out <tune file>
    PY ... judge --tune <tune file> --emb-E <store_E.npz> --base-tune <tune.json> --base-emb-E <store_E.npz>
           [--base-eval <eval.json>]                                 # once; the tune file must be committed
    PY ... apply --src <official tests dir> --pool public|private --emb <pool_<pool>.npz> [--tune ... --eval ...]

One code path for any pool: backend_scores(pool, emb, cfg) -> questions x candidates ranking matrix, where
pool = {"T": {term: (matrix, row_ok)}, "cand_rows": [[row]], "q_rows": [[row]]} (rows = feature-store rows, or
item numbers of an official pool) and emb = p36_eval.Emb with .colour[row] in ('B', 'W'). That function never
reads the answers (pool["owner"]); the oracle bound (B7) is the separate oracle_linked_scores().
Reuses results/prereg36/explore/p36_eval.py, p36_lib.py, tools/p06_style.py, results/prereg35/explore/p35_sim.py
unchanged. No validation answers, no competition-2 features.
"""
import argparse
import copy
import ctypes
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P36 = os.path.join(REPO, "results", "prereg36", "explore")
if P36 not in sys.path:
    sys.path.insert(0, P36)
import p36_lib as L  # noqa: E402
import p36_eval as pe  # noqa: E402
import p06_sim as sim  # noqa: E402
import p06_style as st  # noqa: E402
import p35_sim as ps  # noqa: E402

DATA = os.path.join(REPO, "data", "prereg38")
EMB_D = os.path.join(L.EMB, "step_40000", "store_D.npz")
P36_TUNE = os.path.join(P36, "tune.json")
SEAM_TOL = 1e-12
D_SEEDS = pe.D_SEEDS
BOOT, BOOT_SEED = 10000, 38
KEEP_GAIN = 0.003
WE = (2, 4, 8, 16)
WE_K = (1, 2, 4, 8, 16)
KS = (10, 20, 40)
GMUL = (0, 0.5, 1, 2)
G_TUNED = ("O", "L", "JO", "PA")
QS_TEMP = 0.05

# The structure / numbers of B0 = M36 (prereg36 tune.json: wE 4, CSLS 20; G weights of prereg35).
BASE = {
    "gnorm": False,          # B3a: L2-normalise every game vector first
    "pool_w": "mean",        # B3a: "mean" | "count" (games weighted by their number of sample moves)
    "centre_colour": False,  # B1a
    "whiten": None,          # B4: {"lam": float, "lda": None | int}
    "sim": "proto",          # "proto" | "colour" (B1b) | "pair" (B3b) | "krecip" (B6c)
    "alpha": None,           # B1b: weight of the cross-colour z score
    "pair": None,            # B3b: {"agg": "mean" | "top" | "lse", "m": int, "tau": float}
    "krecip": None,          # B6c: {"k1": int, "k2": int, "lam": float}
    "aqe": None,             # B6a: {"n": int, "beta": float}   (sim "proto" only)
    "qsmooth": None,         # B6b: {"n": int, "beta": float}
    "wE": 4,                 # scalar, or {"1": w, "2": w, "3": w, "5": w} (B2a; other k -> nearest key)
    "gmul": None,            # B2b: {"1": {"O": m, "L": m, "JO": m, "PA": m}, ...}
    "rowz": False,           # rows of F re-standardised before the normaliser (on with B2)
    "norm": ["csls", 20],    # ["none"] | ["csls", k] | ["asnorm", N] | ["dual", tau]
}


# ================================================================ process manners (PLAN.md 7)
def polite(cores=4):
    """BELOW_NORMAL priority and the last `cores` logical processors only (no thread environment variables)."""
    if os.name != "nt":
        return
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    h = k.GetCurrentProcess()
    k.SetPriorityClass(h, 0x00004000)
    n = os.cpu_count() or cores
    mask = sum(1 << i for i in range(max(0, n - cores), n))
    k.SetProcessAffinityMask(h, mask)
    blas_threads(1)


def blas_threads(n):
    """numpy's bundled OpenBLAS limited to n threads at run time (not through the environment): its default of
    one thread per logical processor thrashes on the 4 processors of polite() (measured: 16 s instead of 0.14 s
    for one B0 evaluation; the B0 mean is bit for bit the same with 1 thread)."""
    import glob
    for lib in glob.glob(os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs", "*openblas*.dll")):
        try:
            dll = ctypes.CDLL(lib)
        except OSError:
            continue
        for name in ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads",
                     "openblas_set_num_threads64_", "openblas_set_num_threads"):
            f = getattr(dll, name, None)
            if f is not None:
                f(ctypes.c_int(int(n)))
                return True
    return False


def free_ram_mb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1048576.0


def wait_for_ram(need_mb=6144, tries=20, pause=60):
    if os.name != "nt":
        return
    for _ in range(tries):
        f = free_ram_mb()
        if f >= need_mb:
            return
        print("free RAM %.0f MB < %d MB: waiting" % (f, need_mb), flush=True)
        time.sleep(pause)
    raise SystemExit("free RAM stayed below %d MB: not starting" % need_mb)


# ================================================================ the pool's games
def _colour_is_white(emb, row):
    return str(emb.colour[int(row)]) == "W"


def pool_games(emb, pool):
    """Games with a sample, per side: X (n x d), g (index of the candidate / question), white (bool), cnt."""
    out = {"nc": len(pool["cand_rows"]), "nq": len(pool["q_rows"])}
    for side, groups in (("c", pool["cand_rows"]), ("q", pool["q_rows"])):
        idx, g, white = [], [], []
        for i, rows in enumerate(groups):
            for r in rows:
                j = emb.pos[int(r)]
                if emb.count[j] > 0:
                    idx.append(j)
                    g.append(i)
                    white.append(_colour_is_white(emb, r) if emb.colour is not None else False)
        idx = np.asarray(idx, np.int64)
        out["X" + side] = np.asarray(emb.emb[idx], np.float64)
        out["g" + side] = np.asarray(g, np.int64)
        out["w" + side] = np.asarray(white, bool)
        out["n" + side + "g"] = np.asarray(emb.count[idx], np.float64)
    return out


def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def whitening(Xc, gc, lam, lda=None):
    """(mu, P): x -> (x - mu) @ P. P = W_lam^(-1/2) [@ top-`lda` between-candidate directions], W = within-
    candidate covariance of the candidate games, W_lam = (1 - lam) W + lam (tr W / d) I. Candidate ids only."""
    mu = Xc.mean(0)
    Z = Xc - mu
    d = Z.shape[1]
    ids, inv = np.unique(gc, return_inverse=True)
    sums = np.zeros((len(ids), d))
    np.add.at(sums, inv, Z)
    cnt = np.bincount(inv, minlength=len(ids)).astype(np.float64)
    means = sums / cnt[:, None]
    R = Z - means[inv]
    W = R.T @ R / max(1.0, len(Z) - len(ids))
    Wl = (1.0 - lam) * W + lam * (np.trace(W) / d) * np.eye(d)
    ev, U = np.linalg.eigh(Wl)
    ev = np.maximum(ev, 1e-12)
    P = (U / np.sqrt(ev)) @ U.T
    if lda:
        Mw = means @ P
        B = (Mw * cnt[:, None]).T @ Mw / cnt.sum()
        bv, V = np.linalg.eigh(B)
        P = P @ V[:, np.argsort(-bv)[:int(lda)]]
    return mu, P


def transform(G, cfg):
    """The game vectors after gnorm -> colour centring -> whitening (PLAN.md 3: B3a, B1a, B4). No answers."""
    Xc, Xq = G["Xc"], G["Xq"]
    if cfg.get("gnorm"):
        Xc, Xq = _l2(Xc), _l2(Xq)
    if cfg.get("centre_colour"):
        Xc, Xq = Xc.copy(), Xq.copy()
        for c in (False, True):
            mc, mq = G["wc"] == c, G["wq"] == c
            n = int(mc.sum() + mq.sum())
            if n:
                mean = (Xc[mc].sum(0) + Xq[mq].sum(0)) / n
                Xc[mc] -= mean
                Xq[mq] -= mean
    wh = cfg.get("whiten")
    if wh:
        mu, P = whitening(Xc, G["gc"], float(wh["lam"]), wh.get("lda"))
        Xc, Xq = (Xc - mu) @ P, (Xq - mu) @ P
    return Xc, Xq


def proto(X, g, n, w=None):
    """L2-normalised (weighted) mean of the rows of every group; has[i] False and zeros when the group is empty."""
    V = np.zeros((n, X.shape[1]))
    if len(X):
        np.add.at(V, g, X if w is None else X * w[:, None])
    cnt = np.bincount(g, minlength=n) if len(X) else np.zeros(n)
    nv = np.linalg.norm(V, axis=1)
    has = (cnt > 0) & (nv > 0)
    V[has] /= nv[has, None]
    V[~has] = 0.0
    return V, has


def _protos(G, Xc, Xq, cfg):
    wc = G["ncg"] if cfg.get("pool_w") == "count" else None
    wq = G["nqg"] if cfg.get("pool_w") == "count" else None
    C, hc = proto(Xc, G["gc"], G["nc"], wc)
    Q, hq = proto(Xq, G["gq"], G["nq"], wq)
    return Q, hq, C, hc


def _z_valid(M, valid):
    """Row z over the valid entries only; 0 elsewhere and for rows with < 2 valid entries or no spread."""
    Z = np.zeros(M.shape)
    for i in range(M.shape[0]):
        v = valid[i]
        if v.sum() >= 2:
            x = M[i, v]
            sd = x.std()
            if sd >= 1e-12:
                Z[i, v] = (x - x.mean()) / sd
    return Z


def colour_sim(G, Xc, Xq, alpha):
    """B1b: per question colour c, z(same-colour prototypes) + alpha z(cross-colour prototypes); a missing same
    entry takes the cross z (and the reverse); the question's colours averaged with its numbers of games."""
    nq, nc = G["nq"], G["nc"]
    E, wsum = np.zeros((nq, nc)), np.zeros(nq)
    for c in (False, True):
        mq = G["wq"] == c
        Q, hq = proto(Xq[mq], G["gq"][mq], nq)
        Cs, hs = proto(Xc[G["wc"] == c], G["gc"][G["wc"] == c], nc)
        Cx, hx = proto(Xc[G["wc"] != c], G["gc"][G["wc"] != c], nc)
        vs = hq[:, None] & hs[None, :]
        vx = hq[:, None] & hx[None, :]
        zs, zx = _z_valid(Q @ Cs.T, vs), _z_valid(Q @ Cx.T, vx)
        zs2 = np.where(vs, zs, zx)
        zx2 = np.where(vx, zx, zs)
        n_qc = np.bincount(G["gq"][mq], minlength=nq).astype(np.float64) * hq
        E += n_qc[:, None] * (zs2 + float(alpha) * zx2)
        wsum += n_qc
    ok = wsum > 0
    E[ok] /= wsum[ok, None]
    return E, ok


def pair_sim(G, Xc, Xq, spec):
    """B3b: cosine of every question game with every candidate game; over the candidate's games: mean / mean of
    the m largest / tau * log mean exp(s / tau); then the mean over the question's games."""
    nq, nc = G["nq"], G["nc"]
    S = _l2(Xq) @ _l2(Xc).T                                   # question games x candidate games
    order = np.argsort(G["gc"], kind="stable")
    cnt = np.bincount(G["gc"], minlength=nc)
    width = int(cnt.max()) if len(cnt) else 0
    pad = np.full((nc, width), -1, np.int64)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    for j in range(nc):
        pad[j, :cnt[j]] = order[start[j]:start[j] + cnt[j]]
    mask = pad >= 0
    T = S[:, np.where(mask, pad, 0)]                          # q games x candidates x width
    agg = spec["agg"]
    if agg == "mean":
        A = np.where(mask[None], T, 0.0).sum(2) / np.maximum(cnt, 1)[None, :]
    elif agg == "top":
        m = int(spec["m"])
        Ts = -np.sort(-np.where(mask[None], T, -np.inf), axis=2)[:, :, :m]
        fin = np.isfinite(Ts)
        A = np.where(fin, Ts, 0.0).sum(2) / np.maximum(fin.sum(2), 1)
    elif agg == "lse":
        tau = float(spec["tau"])
        Z = np.where(mask[None], T / tau, -np.inf)
        mx = Z.max(2)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        A = tau * (mx + np.log(np.where(mask[None], np.exp(Z - mx[:, :, None]), 0.0).sum(2)
                               / np.maximum(cnt, 1)[None, :] + 1e-300))
    else:
        raise ValueError(agg)
    A[:, cnt == 0] = 0.0
    E = np.zeros((nq, nc))
    np.add.at(E, G["gq"], A)
    nqg = np.bincount(G["gq"], minlength=nq)
    ok = nqg > 0
    E[ok] /= nqg[ok, None]
    return E, ok


def k_reciprocal(Q, C, k1, k2, lam):
    """Zhong et al. 2017 re-ranking on the joint [questions; candidates] prototypes; returns the final distance
    questions x candidates (smaller = closer)."""
    X = np.vstack([Q, C])
    nq, n = len(Q), len(Q) + len(C)
    orig = np.maximum(2.0 - 2.0 * (X @ X.T), 0.0)
    orig = (orig / np.maximum(orig.max(0), 1e-12)).T
    rank = np.argsort(orig, axis=1, kind="stable")
    V = np.zeros((n, n))
    half = int(round(k1 / 2.0)) + 1
    for i in range(n):
        fwd = rank[i, :k1 + 1]
        krec = fwd[np.any(rank[fwd, :k1 + 1] == i, axis=1)]
        exp = krec
        for cand in krec:
            cf = rank[cand, :half]
            ckr = cf[np.any(rank[cf, :half] == cand, axis=1)]
            if len(np.intersect1d(ckr, krec)) > 2.0 / 3.0 * len(ckr):
                exp = np.append(exp, ckr)
        exp = np.unique(exp)
        w = np.exp(-orig[i, exp])
        V[i, exp] = w / w.sum()
    if k2 > 1:
        V = np.stack([V[rank[i, :k2]].mean(0) for i in range(n)])
    jac = np.zeros((nq, n))
    for i in range(nq):
        nz = np.nonzero(V[i])[0]
        mn = np.minimum(V[i, nz][None, :], V[:, nz]).sum(1)
        jac[i] = 1.0 - mn / (2.0 - mn)
    final = jac * (1.0 - lam) + orig[:nq] * lam
    return final[:, nq:]


# ================================================================ E term, fusion, normalisers
def e_term(pool, emb, cfg, G=None):
    """(questions x candidates matrix of the CNN term, question ok). B0: exactly p36_eval.e36."""
    plain = not (cfg.get("gnorm") or cfg.get("centre_colour") or cfg.get("whiten")) and cfg.get("pool_w", "mean") == "mean"
    s = cfg.get("sim", "proto")
    if plain and s == "proto":
        return pe.e36(emb, pool)
    G = pool_games(emb, pool) if G is None else G
    Xc, Xq = transform(G, cfg)
    if s == "proto":
        Q, hq, C, _hc = _protos(G, Xc, Xq, cfg)
        return Q @ C.T, hq
    if s == "colour":
        return colour_sim(G, Xc, Xq, cfg["alpha"])
    if s == "pair":
        return pair_sim(G, Xc, Xq, cfg["pair"])
    if s == "krecip":
        Q, hq, C, _hc = _protos(G, Xc, Xq, cfg)
        kr = cfg["krecip"]
        return -k_reciprocal(Q, C, int(kr["k1"]), int(kr["k2"]), float(kr["lam"])), hq
    raise ValueError(s)


def q_sizes(pool):
    return np.array([len(g) for g in pool["q_rows"]], np.int64)


def _per_k(value, ks, keys=("1", "2", "3", "5")):
    """Vector over the questions of a per-k setting {"1": v, ...}; a k outside the keys takes the nearest key."""
    kk = np.array([int(x) for x in keys])
    vals = np.array([float(value[x]) for x in keys])
    return vals[np.abs(ks[:, None] - kk[None, :]).argmin(1)]


def fuse_cfg(pool, E, cfg, zt=None):
    """F = sum_n w_n z(T_n) + wE z(E36) (weights per question size when cfg has per-k settings)."""
    w, _kG = pe.g_combo()
    wE, gmul = cfg.get("wE", 4), cfg.get("gmul")
    if not isinstance(wE, dict) and not gmul:
        F = st.fuse([pool["T"][n] for n in ps.TERMS] + [E], list(w) + [wE])
    else:
        ks = q_sizes(pool)
        F = np.zeros(np.asarray(E[0]).shape)
        for n, wn in zip(ps.TERMS, w):
            if wn == 0:
                continue
            z = st.zscore_rows(*pool["T"][n]) if zt is None else zt[n]
            mul = _per_k({k: gmul[k].get(n, 1.0) for k in gmul}, ks, tuple(gmul)) if gmul else 1.0
            F = F + z * (float(wn) * (mul[:, None] if gmul else 1.0))
        we = _per_k(wE, ks, tuple(wE)) if isinstance(wE, dict) else np.full(len(ks), float(wE))
        F = F + st.zscore_rows(*E) * we[:, None]
    return st.zrows(F) if cfg.get("rowz") else F


def normalise(F, norm):
    kind = norm[0]
    if kind == "none":
        return np.asarray(F, np.float64)
    if kind == "csls":
        return st.csls(F, int(norm[1]))
    if kind == "asnorm":
        n = int(norm[1])
        tq = -np.sort(-F, axis=1)[:, :n]
        tc = -np.sort(-F, axis=0)[:n]
        sq, sc = np.maximum(tq.std(1), 1e-9), np.maximum(tc.std(0), 1e-9)
        return 0.5 * ((F - tq.mean(1)[:, None]) / sq[:, None] + (F - tc.mean(0)[None, :]) / sc[None, :])
    if kind == "dual":
        Z = st.zrows(F) / float(norm[1])

        def lsm(A, ax):
            m = A.max(ax, keepdims=True)
            return A - m - np.log(np.exp(A - m).sum(ax, keepdims=True))
        return lsm(Z, 1) + lsm(Z, 0)
    raise ValueError(kind)


def q_smooth(F, Qp, hq, n, beta, temp=QS_TEMP):
    """B6b: F'[q] = F[q] + beta * sum over q's n nearest questions (prototype cosine) of softmax weight * F[q']."""
    A = Qp @ Qp.T
    A[~hq] = -np.inf
    A[:, ~hq] = -np.inf
    np.fill_diagonal(A, -np.inf)
    n = min(int(n), max(0, len(F) - 1))
    if n == 0:
        return F
    nb = np.argsort(-A, axis=1, kind="stable")[:, :n]
    s = np.take_along_axis(A, nb, 1)
    fin = np.isfinite(s)
    s = np.where(fin, s, -1e9) / temp
    a = np.exp(s - s.max(1, keepdims=True)) * fin
    a = a / np.maximum(a.sum(1, keepdims=True), 1e-300)
    return F + float(beta) * np.einsum("qn,qnc->qc", a, F[nb])


def backend_scores(pool, emb, cfg, cache=None):
    """The ranking matrix (questions x candidates) of a configuration. Reads pool['T'], ['cand_rows'],
    ['q_rows'] only. cache: an optional dict of the caller for the E term of this pool."""
    key = json.dumps({k: cfg.get(k) for k in ("gnorm", "pool_w", "centre_colour", "whiten", "sim", "alpha", "pair",
                                              "krecip")}, sort_keys=True)
    if cache is not None and key in cache:
        E = cache[key]
    else:
        E = e_term(pool, emb, cfg)
        if cache is not None:
            cache[key] = E
    zt = None
    if cache is not None:
        if "__zt" not in cache:
            cache["__zt"] = {n: st.zscore_rows(*pool["T"][n]) for n in ps.TERMS}
        zt = cache["__zt"]
    F = fuse_cfg(pool, E, cfg, zt)
    need_protos = cfg.get("aqe") or cfg.get("qsmooth")
    if need_protos:
        pk = "__protos" + json.dumps({k: cfg.get(k) for k in ("gnorm", "pool_w", "centre_colour", "whiten")},
                                     sort_keys=True)
        if cache is not None and pk in cache:
            Q, hq, C = cache[pk]
        else:
            G = pool_games(emb, pool)
            Xc, Xq = transform(G, cfg)
            Q, hq, C, _hc = _protos(G, Xc, Xq, cfg)
            if cache is not None:
                cache[pk] = (Q, hq, C)
        if cfg.get("aqe"):
            if cfg.get("sim", "proto") != "proto":
                raise ValueError("aqe needs sim 'proto'")
            a = cfg["aqe"]
            top = np.argsort(-F, axis=1, kind="stable")[:, :int(a["n"])]
            Q2 = _l2(Q + float(a["beta"]) * C[top].mean(1))
            Q2[~hq] = 0.0
            F = fuse_cfg(pool, (Q2 @ C.T, hq), cfg, zt)
        if cfg.get("qsmooth"):
            s = cfg["qsmooth"]
            F = q_smooth(F, Q, hq, s["n"], s["beta"])
            if cfg.get("rowz"):
                F = st.zrows(F)
    return normalise(F, cfg.get("norm", ["csls", 20]))


def oracle_linked_scores(pool, emb, cfg, owner):
    """B7, ORACLE (uses the answers; an upper bound only, never a submission): the questions of one owner merged.
    Per term the member rows are z-scored and averaged; the CNN vectors come from all the merged games."""
    owner = np.asarray(owner)
    ids, inv = np.unique(owner, return_inverse=True)
    T = {}
    for n in ps.TERMS:
        M, ok = pool["T"][n]
        Z = st.zscore_rows(M, ok)
        A = np.zeros((len(ids), Z.shape[1]))
        np.add.at(A, inv, Z)
        A /= np.bincount(inv, minlength=len(ids))[:, None]
        okm = np.zeros(len(ids), bool)
        np.logical_or.at(okm, inv, np.asarray(ok, bool))
        T[n] = (A, okm)
    q_rows = [[] for _ in ids]
    for i, g in zip(inv.tolist(), pool["q_rows"]):
        q_rows[i] += list(g)
    merged = {"T": T, "cand_rows": pool["cand_rows"], "q_rows": q_rows}
    return backend_scores(merged, emb, cfg)[inv]


def sha_json(path):
    """sha256 of a JSON record with CRLF read as LF: the same value before and after a git checkout that converts
    line endings (the record files of this folder are text for git)."""
    return st.sha256_lf(path)


# ================================================================ evaluation on pools (answers used here only)
def bootstrap(d, n=BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, np.float64)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


class Bench:
    """The pools, their caches and the B0 per-pool scores."""

    def __init__(self, pools, emb, base_cfg=None):
        self.pools, self.emb = pools, emb
        self.caches = [{} for _ in pools]
        self.n_eval = 0
        self.base_cfg = copy.deepcopy(BASE if base_cfg is None else base_cfg)
        self.b0 = self.run(self.base_cfg)

    def run(self, cfg, oracle=False):
        per, byk = [], []
        for p, c in zip(self.pools, self.caches):
            S = oracle_linked_scores(p, self.emb, cfg, p["owner"]) if oracle else backend_scores(p, self.emb, cfg, c)
            per.append(st.score(S, p["owner"]))
            byk.append(pe.score_by_k(S, p))
        self.n_eval += 1
        return {"mean": float(np.mean(per)), "per_pool": [float(x) for x in per], "by_k": pe.mean_by_k(byk)}

    def row(self, name, cfg, res, note=None):
        d = np.asarray(res["per_pool"]) - np.asarray(self.b0["per_pool"])
        r = {"name": name, "cfg": cfg, "mean": res["mean"], "diff_b0": float(d.mean()), "ci95": bootstrap(d),
             "pools_higher": int((d > 0).sum()), "by_k": res["by_k"], "per_pool": res["per_pool"]}
        if note:
            r["note"] = note
        return r


def with_(cfg, **kw):
    c = copy.deepcopy(cfg)
    c.update(copy.deepcopy(kw))
    return c


def norm_grid(norm, all_kinds=False):
    full = [["none"]] + [["csls", k] for k in (5, 10, 20, 40, 80)] + [["asnorm", n] for n in (20, 50, 100)] + \
           [["dual", t] for t in (0.5, 1, 2)]
    if all_kinds:
        return full
    if norm[0] == "csls":
        return [["csls", k] for k in KS]
    return [x for x in full if x[0] == norm[0]]


def tune_perk_we(bench, cfg, start=None):
    """Coordinate ascent (one pass, k = 1, 2, 3, 5) of the per-k wE over WE_K; rowz on."""
    cur = with_(cfg, rowz=True)
    w0 = cfg["wE"] if isinstance(cfg.get("wE"), dict) else {k: float(cfg.get("wE", 4)) for k in ("1", "2", "3", "5")}
    cur["wE"] = dict(start or w0)
    best = bench.run(cur)
    for k in ("1", "2", "3", "5"):
        for v in WE_K:
            if v == cur["wE"][k]:
                continue
            c = with_(cur)
            c["wE"][k] = v
            r = bench.run(c)
            if r["mean"] > best["mean"]:
                best, cur = r, c
    return cur, best


def tune_gmul(bench, cfg):
    """B2b: coordinate ascent (one pass; k outer, terms O, L, JO, PA inner) of the per-k G multipliers."""
    cur = with_(cfg, rowz=True)
    if not cur.get("gmul"):
        cur["gmul"] = {k: {n: 1.0 for n in G_TUNED} for k in ("1", "2", "3", "5")}
    best = bench.run(cur)
    for k in ("1", "2", "3", "5"):
        for n in G_TUNED:
            for v in GMUL:
                if v == cur["gmul"][k][n]:
                    continue
                c = with_(cur)
                c["gmul"][k][n] = v
                r = bench.run(c)
                if r["mean"] > best["mean"]:
                    best, cur = r, c
    return cur, best


def tune_numeric(bench, cfg, all_norms=False):
    """wE (scalar grid, or per-k coordinate ascent when cfg's wE is per-k) x the normaliser's grid."""
    best = None
    for nm in norm_grid(cfg["norm"], all_norms):
        if isinstance(cfg.get("wE"), dict):
            c, r = tune_perk_we(bench, with_(cfg, norm=nm))
            cands = [(c, r)]
        else:
            cands = []
            for w in WE:
                c = with_(cfg, wE=w, norm=nm)
                cands.append((c, bench.run(c)))
        for c, r in cands:
            if best is None or r["mean"] > best[1]["mean"]:
                best = (c, r)
    return best


# ---------------------------------------------------------------- module settings (PLAN.md 3)
def module_settings(name):
    """[(label, {cfg overrides})] of a module's structural / numeric settings (without wE and the normaliser)."""
    if name == "B1a":
        return [("centre", {"centre_colour": True})]
    if name == "B1b":
        return [("alpha=%g" % a, {"sim": "colour", "alpha": a}) for a in (0, 0.25, 0.5, 1)]
    if name == "B3a":
        return [("gnorm", {"gnorm": True}), ("count", {"pool_w": "count"}),
                ("gnorm+count", {"gnorm": True, "pool_w": "count"})]
    if name == "B3b":
        out = [("mean", {"sim": "pair", "pair": {"agg": "mean"}})]
        out += [("top%d" % m, {"sim": "pair", "pair": {"agg": "top", "m": m}}) for m in (1, 2, 3, 5)]
        out += [("lse%g" % t, {"sim": "pair", "pair": {"agg": "lse", "tau": t}}) for t in (0.05, 0.1, 0.2)]
        return out
    if name == "B4":
        out = []
        for gn in (False, True):
            for lam in (0.02, 0.05, 0.1, 0.3, 0.5, 0.7):      # 0.02, 0.05: PLAN revision 2
                for lda in (None, 32, 64, 128):
                    out.append(("gnorm=%d lam=%g lda=%s" % (gn, lam, lda),
                                {"gnorm": gn, "whiten": {"lam": lam, "lda": lda}}))
        return out
    if name == "B6a":
        return [("n=%d beta=%g" % (n, b), {"aqe": {"n": n, "beta": b}}) for n in (1, 3) for b in (0.5, 1)]
    if name == "B6b":
        return [("n=%d beta=%g" % (n, b), {"qsmooth": {"n": n, "beta": b}}) for n in (1, 2, 3) for b in (0.25, 0.5, 1)]
    if name == "B6c":
        return [("k1=%d k2=%d lam=%g" % (a, b, c), {"sim": "krecip", "krecip": {"k1": a, "k2": b, "lam": c}})
                for a in (10, 20) for b in (3, 6) for c in (0.3, 0.5)]
    raise ValueError(name)


SIM_MODULES = ("B1b", "B3b", "B6c")       # they set cfg["sim"]: mutually exclusive
GRID_MODULES = ("B1a", "B1b", "B3a", "B3b", "B4", "B6a", "B6b", "B6c")


def compatible(cfg, over):
    c = with_(cfg, **over)
    if c.get("aqe") and c.get("sim", "proto") != "proto":
        return False
    return True


def best_setting(bench, cfg, name, all_norms=False):
    """Best (cfg, result, label, all rows) of a module's settings on top of cfg, numbers re-tuned for each."""
    best, rows = None, []
    for label, over in module_settings(name):
        if not compatible(cfg, over):
            continue
        c, r = tune_numeric(bench, with_(cfg, **over), all_norms)
        rows.append({"label": label, "cfg": c, "mean": r["mean"]})
        if best is None or r["mean"] > best[1]["mean"]:
            best = (c, r, label)
    return best, rows


# ================================================================ commands
def load_bench():
    with open(P36_TUNE, encoding="utf-8") as f:
        tune = json.load(f)
    if tune["checkpoint"]["name"] != "step_40000.pt" or tune["M"] != {"wE": 4, "csls_k": 20,
                                                                   "tune_mean": tune["M"]["tune_mean"]}:
        raise SystemExit("prereg36 tune.json is not the M36 this plan was written for")
    store = sim.Store(L.FEATS, load=())
    emb = pe.Emb.load(EMB_D, colour=store.color)
    if emb.sha != tune["checkpoint"]["sha256"]:
        raise SystemExit("store_D.npz was not made from the prereg36 checkpoint")
    pools = pe.load_pools(pe.TERMS_DIR["D"], D_SEEDS)
    bench = Bench(pools, emb)
    seam = {"b0_mean": bench.b0["mean"], "tune_mean": tune["M"]["tune_mean"],
            "abs_diff": abs(bench.b0["mean"] - tune["M"]["tune_mean"])}
    seam["ok"] = seam["abs_diff"] <= SEAM_TOL
    if not seam["ok"]:
        raise SystemExit("seam B0 failed: %r" % seam)
    return bench, seam


def cmd_seam(a):
    bench, seam = load_bench()
    seam["by_k"] = bench.b0["by_k"]
    print(json.dumps(seam, indent=1))
    return 0


def cmd_single(a):
    t0 = time.time()
    bench, seam = load_bench()
    rows, grids = [bench.row("B0", BASE, bench.b0)], {}
    for name in GRID_MODULES:
        (c, r, label), g = best_setting(bench, BASE, name)
        rows.append(bench.row(name, c, r, label))
        grids[name] = g
        print(name, label, round(r["mean"], 4), flush=True)
    # B2a / B2b
    best = None
    for k in KS:
        c, r = tune_perk_we(bench, with_(BASE, norm=["csls", k]))
        if best is None or r["mean"] > best[1]["mean"]:
            best = (c, r)
    rz = [(with_(BASE, rowz=True, norm=["csls", k]),) for k in KS]
    rz = max(((c[0], bench.run(c[0])) for c in rz), key=lambda x: x[1]["mean"])
    rows.append(bench.row("B2ref", rz[0], rz[1], "rowz only (wE 4 for every k)"))
    rows.append(bench.row("B2a", best[0], best[1], "per-k wE"))
    c, r = tune_gmul(bench, best[0])
    rows.append(bench.row("B2b", c, r, "per-k wE + per-k G multipliers"))
    print("B2", round(best[1]["mean"], 4), round(r["mean"], 4), flush=True)
    # B5
    g5, best = [], None
    for nm in norm_grid(BASE["norm"], True):
        c = with_(BASE, norm=nm)
        r = bench.run(c)
        g5.append({"label": "/".join(str(x) for x in nm), "cfg": c, "mean": r["mean"]})
        if best is None or r["mean"] > best[1]["mean"]:
            best = (c, r, g5[-1]["label"])
    rows.append(bench.row("B5", best[0], best[1], best[2]))
    grids["B5"] = g5
    # B7 (oracle)
    r = bench.run(BASE, oracle=True)
    shared = []
    for p in bench.pools:
        cnt = np.bincount(p["owner"])
        shared.append(float((cnt[p["owner"]] >= 2).mean()))
    rows.append(bench.row("B7_ORACLE", BASE, r, "ORACLE: same-owner questions merged; share of questions with a "
                                                "sibling %.3f" % float(np.mean(shared))))
    out = {"format": "c10-prereg38 single v1", "seam": seam, "d_seeds": D_SEEDS, "rows": rows, "grids": grids,
           "evaluations": bench.n_eval, "seconds": round(time.time() - t0, 1), "git_head": L.git_head()}
    os.makedirs(DATA, exist_ok=True)
    L.write_json(os.path.join(DATA, "single.json"), out)
    print(json.dumps([{k: r[k] for k in ("name", "mean", "diff_b0", "ci95", "note") if k in r} for r in rows],
                     indent=1))
    return 0


def cmd_greedy(a):
    t0 = time.time()
    with open(os.path.join(DATA, "single.json"), encoding="utf-8") as f:
        single = json.load(f)
    bench, seam = load_bench()
    gain = {r["name"]: r["diff_b0"] for r in single["rows"]}
    order = sorted((n for n in gain if n not in ("B0", "B7_ORACLE", "B2ref")), key=lambda n: -gain[n])
    cur, cur_res, steps, used_sim = copy.deepcopy(BASE), bench.b0, [], False
    for name in order:
        if name in SIM_MODULES and used_sim:
            steps.append({"module": name, "kept": False, "reason": "another similarity module is already in"})
            continue
        if name == "B2a":
            c, r = max((tune_perk_we(bench, with_(cur, norm=nm)) for nm in norm_grid(cur["norm"])),
                       key=lambda x: x[1]["mean"])
            label = "per-k wE"
        elif name == "B2b":
            base = cur if isinstance(cur.get("wE"), dict) else tune_perk_we(bench, cur)[0]
            c, r = tune_gmul(bench, base)
            label = "per-k G multipliers"
        elif name == "B5":
            c, r = tune_numeric(bench, cur, all_norms=True)
            label = "/".join(str(x) for x in c["norm"])
        else:
            res = best_setting(bench, cur, name)[0]
            if res is None:
                steps.append({"module": name, "kept": False, "reason": "not compatible with the current assembly"})
                continue
            c, r, label = res
        d = r["mean"] - cur_res["mean"]
        keep = d >= KEEP_GAIN
        steps.append({"module": name, "label": label, "mean": r["mean"], "gain_over_current": d, "kept": bool(keep),
                      "cfg": c})
        print(name, label, round(r["mean"], 4), "%+.4f" % d, "KEEP" if keep else "skip", flush=True)
        if keep:
            cur, cur_res = c, r
            used_sim = used_sim or name in SIM_MODULES
    final_cfg, final_res = tune_numeric(bench, cur, all_norms=False)
    if final_res["mean"] < cur_res["mean"]:
        final_cfg, final_res = cur, cur_res
    row = bench.row("ASSEMBLED", final_cfg, final_res)
    halves = {}
    d = np.asarray(final_res["per_pool"]) - np.asarray(bench.b0["per_pool"])
    halves["pools_100_104"], halves["pools_105_109"] = float(d[:5].mean()), float(d[5:].mean())
    out = {"format": "c10-prereg38 tune_D v1", "seam": seam, "d_seeds": D_SEEDS,
           "emb": {"ckpt": bench.emb.name, "ckpt_sha256": bench.emb.sha},
           "kept": [s["module"] for s in steps if s["kept"]],
           "b0": {"cfg": BASE, "mean": bench.b0["mean"], "by_k": bench.b0["by_k"], "per_pool": bench.b0["per_pool"]},
           "single": single["rows"], "single_grids": single["grids"], "order": order, "greedy_steps": steps,
           "recommended": row, "recommended_diff_by_half": halves, "keep_gain": KEEP_GAIN,
           "note": "D pools are the tuning pools (296 players, 10 pools): the gains are optimistic; no adoption "
                   "claim. The E pools were not loaded.",
           "evaluations": bench.n_eval, "seconds": round(time.time() - t0, 1), "git_head": L.git_head()}
    L.write_json(a.out, out)
    print(json.dumps({k: row[k] for k in ("mean", "diff_b0", "ci95", "pools_higher", "by_k", "cfg")}, indent=1))
    return 0


# ================================================================ fixed structure: re-tune, half check
def same_kind(name, over, ref):
    """True if a module setting has the structure (kind of method) of the reference configuration."""
    if name == "B4":
        return bool(over.get("gnorm", False)) == bool(ref.get("gnorm", False))
    if name == "B3a":
        return bool(over.get("gnorm", False)) == bool(ref.get("gnorm", False)) and \
            over.get("pool_w", "mean") == ref.get("pool_w", "mean")
    if name == "B3b":
        return over["pair"]["agg"] == ref["pair"]["agg"]
    return True


def retune_structure(bench, kept, ref_cfg):
    """PLAN.md 6: the structure (kept modules, in their greedy order, each with the kind of method of ref_cfg) is
    fixed; every number is chosen again on the bench's pools over the grids of PLAN.md 3."""
    cur, res = copy.deepcopy(bench.base_cfg), bench.b0
    for name in kept:
        if name == "B2a":
            cur, res = max((tune_perk_we(bench, with_(cur, norm=nm)) for nm in norm_grid(cur["norm"])),
                           key=lambda x: x[1]["mean"])
        elif name == "B2b":
            base = cur if isinstance(cur.get("wE"), dict) else tune_perk_we(bench, cur)[0]
            cur, res = tune_gmul(bench, base)
        elif name == "B5":
            cur, res = tune_numeric(bench, with_(cur, norm=ref_cfg["norm"]))
        else:
            best = None
            for _label, over in module_settings(name):
                if not same_kind(name, over, ref_cfg) or not compatible(cur, over):
                    continue
                c, r = tune_numeric(bench, with_(cur, **over))
                if best is None or r["mean"] > best[1]["mean"]:
                    best = (c, r)
            cur, res = best
    c, r = tune_numeric(bench, cur)
    return (c, r) if r["mean"] >= res["mean"] else (cur, res)


def cmd_halfcheck(a):
    """D-only diagnostic (PLAN.md revision 3): numbers chosen on 5 of the D pools, scored on the other 5."""
    t0 = time.time()
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    bench, _seam = load_bench()
    ref, kept = tune["recommended"]["cfg"], tune["kept"]
    halves = [(bench.pools[:5], bench.pools[5:]), (bench.pools[5:], bench.pools[:5])]
    out = {"format": "c10-prereg38 halfcheck v1", "tune_sha256": sha_json(a.tune), "variants": {}}
    for label, mods in (("first_kept_only", kept[:1]), ("all_kept", kept)):
        d, picked = [], []
        for tr, te in halves:
            bt, be = Bench(tr, bench.emb), Bench(te, bench.emb)
            c, _r = retune_structure(bt, mods, ref)
            r = be.run(c)
            d += (np.asarray(r["per_pool"]) - np.asarray(be.b0["per_pool"])).tolist()
            picked.append(c)
        out["variants"][label] = {"modules": mods, "held_out_diff_b0_per_pool": d,
                                  "held_out_diff_b0_mean": float(np.mean(d)), "ci95": bootstrap(np.asarray(d)),
                                  "pools_higher": int((np.asarray(d) > 0).sum()),
                                  "cfg_tuned_on_first_half": picked[0], "cfg_tuned_on_second_half": picked[1]}
    a_, b_ = (np.asarray(out["variants"][k]["held_out_diff_b0_per_pool"]) for k in ("all_kept", "first_kept_only"))
    out["all_minus_first"] = {"mean": float((a_ - b_).mean()), "ci95": bootstrap(a_ - b_),
                              "pools_higher": int((a_ - b_ > 0).sum())}
    out["seconds"], out["git_head"] = round(time.time() - t0, 1), L.git_head()
    L.write_json(a.out, out)
    print(json.dumps({k: {x: v[x] for x in ("modules", "held_out_diff_b0_mean", "ci95", "pools_higher")}
                      for k, v in out["variants"].items()}, indent=1))
    print(json.dumps(out["all_minus_first"]))
    return 0


def cmd_variant(a):
    """D-only (PLAN.md revision 4): a sub-structure of the greedy result (e.g. --kept B4), numbers chosen on the
    10 D pools; same file layout as tune_D.json's 'recommended' so that retune / judge / apply accept it."""
    t0 = time.time()
    with open(a.structure, encoding="utf-8") as f:
        ref = json.load(f)
    kept = [x for x in a.kept.split(",") if x]
    if not set(kept) <= set(ref["kept"]):
        raise SystemExit("--kept must be a subset of %r" % ref["kept"])
    bench, seam = load_bench()
    c, r = retune_structure(bench, kept, ref["recommended"]["cfg"])
    out = {"format": "c10-prereg38 variant v1", "seam": seam, "d_seeds": D_SEEDS, "kept": kept,
           "structure_sha256": sha_json(a.structure),
           "emb": {"ckpt": bench.emb.name, "ckpt_sha256": bench.emb.sha},
           "b0": {"cfg": bench.base_cfg, "mean": bench.b0["mean"], "by_k": bench.b0["by_k"]},
           "recommended": bench.row("VARIANT", c, r), "seconds": round(time.time() - t0, 1),
           "git_head": L.git_head()}
    L.write_json(a.out, out)
    print(json.dumps({k: out["recommended"][k] for k in ("mean", "diff_b0", "ci95", "pools_higher", "by_k", "cfg")},
                     indent=1))
    return 0


def cmd_oracle(a):
    """D-only, ORACLE, record only (PLAN.md revision 5): B7 (same-owner questions merged with the answers) on top
    of a tuned configuration. An upper bound of question linking; never a submission."""
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    bench, _seam = load_bench()
    cfg = tune["recommended"]["cfg"]
    plain, orc = bench.run(cfg), bench.run(cfg, oracle=True)
    d = np.asarray(orc["per_pool"]) - np.asarray(plain["per_pool"])
    out = {"format": "c10-prereg38 oracle v1", "ORACLE": True, "tune_sha256": sha_json(a.tune), "kept": tune["kept"],
           "plain": {"mean": plain["mean"], "by_k": plain["by_k"]},
           "oracle_linked": {"mean": orc["mean"], "by_k": orc["by_k"]},
           "oracle_minus_plain": {"mean": float(d.mean()), "ci95": bootstrap(d), "pools_higher": int((d > 0).sum())},
           "git_head": L.git_head()}
    L.write_json(a.out, out)
    print(json.dumps(out, indent=1))
    return 0


def base_cfg_of(base_tune):
    """The configuration of a prereg36-format tune.json (M = G terms + wE z(E36), CSLS k)."""
    return with_(BASE, wE=base_tune["M"]["wE"], norm=["csls", int(base_tune["M"]["csls_k"])])


def cmd_retune(a):
    """For the main session when another checkpoint is adopted (PLAN.md 6): same structure, numbers chosen again
    on the D pools with that checkpoint's D vectors. Not run in this round."""
    t0 = time.time()
    with open(a.structure, encoding="utf-8") as f:
        ref = json.load(f)
    with open(a.base_tune, encoding="utf-8") as f:
        base = json.load(f)
    store = sim.Store(L.FEATS, load=())
    emb = pe.Emb.load(a.emb_D, colour=store.color)
    if emb.sha != base["checkpoint"]["sha256"]:
        raise SystemExit("--emb-D was not made from the checkpoint of --base-tune")
    bench = Bench(pe.load_pools(pe.TERMS_DIR["D"], D_SEEDS), emb, base_cfg_of(base))
    seam = {"base_mean": bench.b0["mean"], "tune_mean": base["M"]["tune_mean"],
            "abs_diff": abs(bench.b0["mean"] - base["M"]["tune_mean"])}
    seam["ok"] = seam["abs_diff"] <= SEAM_TOL
    if not seam["ok"]:
        raise SystemExit("seam failed: the base system's D mean is not the one of --base-tune: %r" % seam)
    c, r = retune_structure(bench, ref["kept"], ref["recommended"]["cfg"])
    out = {"format": "c10-prereg38 retune v1", "seam": seam, "d_seeds": D_SEEDS, "kept": ref["kept"],
           "structure_sha256": sha_json(a.structure), "base_tune_sha256": sha_json(a.base_tune),
           "emb": {"ckpt": emb.name, "ckpt_sha256": emb.sha},
           "b0": {"cfg": bench.base_cfg, "mean": bench.b0["mean"], "by_k": bench.b0["by_k"]},
           "recommended": bench.row("ASSEMBLED", c, r), "seconds": round(time.time() - t0, 1),
           "git_head": L.git_head()}
    L.write_json(a.out, out)
    print(json.dumps({k: out["recommended"][k] for k in ("mean", "diff_b0", "ci95", "pools_higher", "cfg")},
                     indent=1))
    return 0


# ================================================================ E judgement and official pools (main session)
ADOPT_MEAN = 0.01


def judge_numbers(pools, emb_new, cfg, emb_base, base_cfg):
    """Per-pool scores of the assembled backend and of the adopted system, and the rule of PLAN.md 6."""
    N, Bs, bN, bB = [], [], [], []
    for p in pools:
        SN, SB = backend_scores(p, emb_new, cfg), backend_scores(p, emb_base, base_cfg)
        N.append(st.score(SN, p["owner"]))
        Bs.append(st.score(SB, p["owner"]))
        bN.append(pe.score_by_k(SN, p))
        bB.append(pe.score_by_k(SB, p))
    N, Bs = np.array(N), np.array(Bs)
    d = N - Bs
    lo, hi = bootstrap(d)
    need = int(np.ceil(0.9 * len(d)))
    return {"means": {"backend": float(N.mean()), "adopted_system": float(Bs.mean())},
            "per_pool": {"backend": N.tolist(), "adopted_system": Bs.tolist()},
            "backend_minus_adopted": {"mean": float(d.mean()), "ci95": [lo, hi], "pools_higher": int((d > 0).sum())},
            "by_k": {"backend": pe.mean_by_k(bN), "adopted_system": pe.mean_by_k(bB)},
            "rule": "adopt the backend if mean difference >= %.2f and ci95 low > 0 and higher in >= %d of %d pools "
                    "(pool bootstrap, %d samples, seed %d)" % (ADOPT_MEAN, need, len(d), BOOT, BOOT_SEED),
            "adopt": bool(d.mean() >= ADOPT_MEAN and lo > 0 and int((d > 0).sum()) >= need)}


def cmd_judge(a):
    """THE ONE JUDGEMENT ON THE E POOLS (PLAN.md 6). For the main session; never run in the D-only round."""
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("the tune file is not committed (or differs from HEAD); commit it before the judgement")
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    with open(a.base_tune, encoding="utf-8") as f:
        base = json.load(f)
    store = sim.Store(L.FEATS, load=())
    e_new = pe.Emb.load(a.emb_E, colour=store.color)
    e_base = pe.Emb.load(a.base_emb_E, colour=store.color)
    if e_new.sha != tune["emb"]["ckpt_sha256"]:
        raise SystemExit("--emb-E was not made from the checkpoint the backend was tuned with")
    if e_base.sha != base["checkpoint"]["sha256"]:
        raise SystemExit("--base-emb-E was not made from the checkpoint of --base-tune")
    pools = pe.load_pools(pe.TERMS_DIR["E"], pe.E_SEEDS)
    E, _D, _T, _per = L.splits()
    Es = set(E)
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in store.pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    rec = {"format": "c10-prereg38 eval v1", "e_seeds": pe.E_SEEDS, "cfg": tune["recommended"]["cfg"],
           "adopted_system": {"cfg": base_cfg_of(base), "checkpoint": base["checkpoint"]},
           "tune_sha256": sha_json(a.tune), "base_tune_sha256": sha_json(a.base_tune),
           "git_head": L.git_head()}
    rec.update(judge_numbers(pools, e_new, tune["recommended"]["cfg"], e_base, base_cfg_of(base)))
    if a.base_eval:
        with open(a.base_eval, encoding="utf-8") as f:
            want = json.load(f)["means"]["M"]
        rec["seam_adopted_system"] = {"recomputed": rec["means"]["adopted_system"], "recorded": want,
                                      "ok": abs(rec["means"]["adopted_system"] - want) <= SEAM_TOL}
        if not rec["seam_adopted_system"]["ok"]:
            raise SystemExit("seam failed: the adopted system's E mean is not the recorded one: %r"
                             % rec["seam_adopted_system"])
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("means", "backend_minus_adopted", "by_k", "adopt")}, indent=1))
    return 0


def cmd_apply(a):
    """An official new pool through backend_scores (after an adopting judgement). For the main session."""
    import csv
    import p35_apply as pa
    t0 = time.time()
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    with open(a.eval, encoding="utf-8") as f:
        ev = json.load(f)
    if not ev["adopt"] or ev["tune_sha256"] != sha_json(a.tune):
        raise SystemExit("eval.json does not adopt this backend (or belongs to another tune file)")
    cfg = tune["recommended"]["cfg"]
    cands, qs = pa.read_pool(a.src, a.pool)
    items, cand_ids, cand_rows, q_rows, qids = L.pool_items(cands, qs)
    if not all(c.startswith(pa.PREFIX[a.pool] + "p_") for c in cand_ids):
        raise SystemExit("candidate id prefix")
    e = pe.Emb.load(a.emb, colour=np.asarray([c for _s, c in items]))
    if e.sha != tune["emb"]["ckpt_sha256"] or len(e.emb) != len(items):
        raise SystemExit("pool embeddings do not match the tuned checkpoint / the pool's games")
    fit = sim.Store(L.FEATS, load=())
    store = pa.PoolStore(items, fit.csr("resp").vocab)
    ldas, models = {}, {}
    for c in ("B", "W"):
        ld = sim.LdaData(L.FEATS, c, fit)
        models[c] = ld.fit_excluding(fit, set())
        ldas[c] = pa.PoolLda(store, ld.cols)
    rec = {"cand_rows": cand_rows, "q_rows": q_rows}
    rec["T"] = sim.pool_terms(store, rec, ldas, models, None, want=("O", "JO", "S2", "L", "PA"))
    S = backend_scores(rec, e, cfg)
    tops = st.top5(S, cand_ids)
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    path = os.path.join(od, "top5.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"])
        for q, t in zip(qids, tops):
            wr.writerow([q] + t)
    G = pool_games(e, rec)
    rep = {"format": "c10-prereg38 apply v1", "pool": a.pool, "cfg": cfg, "checkpoint": tune["emb"],
           "questions": len(qids), "candidates": len(cand_ids), "games": len(items),
           "games_without_vector": int((e.count == 0).sum()),
           "candidates_without_vector": int(G["nc"] - len(np.unique(G["gc"]))),
           "questions_without_vector": int(G["nq"] - len(np.unique(G["gq"]))),
           "terms_rows_without_signal": {n: int((~np.asarray(rec["T"][n][1])).sum()) for n in ps.TERMS},
           "tune_sha256": sha_json(a.tune), "eval_sha256": sha_json(a.eval),
           "top5_sha256": L.sha256_file(path), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def _fmt_k(by_k):
    return " | ".join("%.3f" % by_k[k] for k in ("1", "2", "3", "5"))


def cmd_report(a):
    with open(a.tune, encoding="utf-8") as f:
        t = json.load(f)
    L_ = ["# prereg38：計分後端模組消融（只在 D 池；E 池沒有載入）", "",
          "資料：`data/prereg36/emb/step_40000/store_D.npz`、`data/prereg36/terms_D`（10 池，每池 300 位候選、200 題）。"
          "數字全部由 `p38_backend.py single`／`greedy` 產生（`tune_D.json`）。",
          "", "**D 池是挑選用的（296 位玩家、格點很多），表上的增益偏樂觀；這一輪不宣稱採用。**", "",
          "接縫 B0：重算 %.15f，prereg36 tune.json %.15f，差 %.1e → %s。"
          % (t["seam"]["b0_mean"], t["seam"]["tune_mean"], t["seam"]["abs_diff"], "相同" if t["seam"]["ok"] else "不同"),
          "", "## 1. 每個模組單獨對 B0（各自的最佳設定；wE、CSLS k 一起在小格點重選）", "",
          "| 模組 | 最佳設定 | D 池平均 | 對 B0 | 95% 區間 | 較高池數 | 1 盤 | 2 盤 | 3 盤 | 5 盤 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in t["single"]:
        c = r["cfg"]
        setting = r.get("note", "") or "M36"
        wE = c["wE"] if not isinstance(c["wE"], dict) else "/".join("%g" % c["wE"][k] for k in ("1", "2", "3", "5"))
        setting += "；wE %s；%s" % (wE, "/".join(str(x) for x in c["norm"]))
        L_.append("| %s | %s | %.4f | %+.4f | [%+.4f, %+.4f] | %d/10 | %s |" % (
            r["name"], setting, r["mean"], r["diff_b0"], r["ci95"][0], r["ci95"][1], r["pools_higher"],
            _fmt_k(r["by_k"])))
    L_ += ["", "B7 是 oracle（用正解把同一人的題目合併），只是上限，不能進交件。", "",
           "## 2. 貪心組裝（依單獨增益由大到小；比目前的組合再多 ≥ +%.3f 才留下）" % t["keep_gain"], "",
           "| 順序 | 模組 | 設定 | 加入後 D 池平均 | 比目前多 | 留下 |", "|---|---|---|---|---|---|"]
    for i, s in enumerate(t["greedy_steps"], 1):
        if "mean" in s:
            L_.append("| %d | %s | %s | %.4f | %+.4f | %s |" % (i, s["module"], s["label"], s["mean"],
                                                               s["gain_over_current"], "是" if s["kept"] else "否"))
        else:
            L_.append("| %d | %s | — | — | — | 否（%s） |" % (i, s["module"], s["reason"]))
    r = t["recommended"]
    L_ += ["", "## 3. 建議的組裝設定", "",
           "- D 池平均 **%.4f**，對 B0 %+.4f，區間 [%+.4f, %+.4f]，%d/10 池較高；前 5 池 %+.4f、後 5 池 %+.4f。"
           % (r["mean"], r["diff_b0"], r["ci95"][0], r["ci95"][1], r["pools_higher"],
              t["recommended_diff_by_half"]["pools_100_104"], t["recommended_diff_by_half"]["pools_105_109"]),
           "- 依盤數（1／2／3／5）：%s；B0：%s。" % (_fmt_k(r["by_k"]).replace(" | ", "／"),
                                             _fmt_k(t["b0"]["by_k"]).replace(" | ", "／")),
           "- 設定：", "", "```json", json.dumps(r["cfg"], indent=1, ensure_ascii=False), "```", ""]
    if os.path.exists(a.half):
        with open(a.half, encoding="utf-8") as f:
            h = json.load(f)
        if h["tune_sha256"] != sha_json(a.tune):
            raise SystemExit("halfcheck_D.json belongs to another tune_D.json")
        L_ += ["## 4. 附加診斷（PLAN 修訂 3）：數值在 5 個 D 池上選、在另外 5 個 D 池上看（兩個方向合計 10 池）", "",
               "同一批 296 位玩家、不同的棋和題目，所以只能看出「數值有沒有對池子過度調整」，不能代替 E 池。", "",
               "| 結構 | 沒參與挑選的 5 池上對 B0 | 95% 區間 | 較高池數 |", "|---|---|---|---|"]
        for _k, v in h["variants"].items():
            L_.append("| %s | %+.4f | [%+.4f, %+.4f] | %d/10 |" % ("＋".join(v["modules"]), v["held_out_diff_b0_mean"],
                                                                 v["ci95"][0], v["ci95"][1], v["pools_higher"]))
        m = h["all_minus_first"]
        L_ += ["", "全部留下的模組 − 只有第一個模組：%+.4f，區間 [%+.4f, %+.4f]，%d/10 池較高。"
               % (m["mean"], m["ci95"][0], m["ci95"][1], m["pools_higher"]), ""]
    if os.path.exists(a.variant):
        with open(a.variant, encoding="utf-8") as f:
            v = json.load(f)
        if v["structure_sha256"] != sha_json(a.tune):
            raise SystemExit("the variant file belongs to another tune_D.json")
        r = v["recommended"]
        L_ += ["## 5. 只留 %s 的設定（PLAN 修訂 4；`%s`）" % ("＋".join(v["kept"]), os.path.basename(a.variant)), "",
               "- D 池平均 **%.4f**，對 B0 %+.4f，區間 [%+.4f, %+.4f]，%d/10 池較高；依盤數（1／2／3／5）：%s。"
               % (r["mean"], r["diff_b0"], r["ci95"][0], r["ci95"][1], r["pools_higher"],
                  _fmt_k(r["by_k"]).replace(" | ", "／")),
               "- 設定：", "", "```json", json.dumps(r["cfg"], indent=1, ensure_ascii=False), "```", ""]
    if os.path.exists(a.oracle):
        with open(a.oracle, encoding="utf-8") as f:
            o = json.load(f)
        m = o["oracle_minus_plain"]
        L_ += ["## 6. 題目互連的上限，疊在 %s 上（PLAN 修訂 5；oracle，只記錄）" % "＋".join(o["kept"]), "",
               "用正解把同一人的題目合併：D 池平均 %.4f → **%.4f**（%+.4f，區間 [%+.4f, %+.4f]，%d/10 池較高）；"
               "依盤數（1／2／3／5）：%s → %s。這是上限，不能進交件；不用正解的版本（B6b）在 D 池是負的。"
               % (o["plain"]["mean"], o["oracle_linked"]["mean"], m["mean"], m["ci95"][0], m["ci95"][1],
                  m["pools_higher"], _fmt_k(o["plain"]["by_k"]).replace(" | ", "／"),
                  _fmt_k(o["oracle_linked"]["by_k"]).replace(" | ", "／")), ""]
    L_ += ["## 注意", "",
           "- B4 只用候選庫的棋和候選 id（每池分開）；題目的正解不參與（`test_p38.py` 的 NoLeak 測試：一般路徑讀不到正解）。",
           "- B1（執色）單獨有效，但加在 B4 之後沒有增益：類內白化已經把同一人黑白棋之間的差異當成類內變異壓掉。",
           "- B4 的 λ 在加了 0.02、0.05 之後仍選到最小值 0.02，但 0.02／0.05／0.1 幾乎一樣（見 `tune_D.json` 的 `single_grids`）。",
           "- D 池每池 300 位候選（3,000 盤）估 256 維的類內共變異；E 池和官方池是 600 位（6,000 盤）。CSLS k 等和池大小有關的值"
           "直接沿用 D 上選到的。",
           "- 第一次 `single` 因執行緒互搶中止（PLAN 修訂 1）；`greedy` 用最終程式重跑過一次，結果和第一次相同。",
           "- `retune`／`judge`／`apply` 三個指令這一輪沒有執行（只有合成資料的單元測試）。", ""]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L_))
    print("wrote", a.out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seam")
    sub.add_parser("single")
    g = sub.add_parser("greedy")
    g.add_argument("--out", default=os.path.join(HERE, "tune_D.json"))
    r = sub.add_parser("report")
    r.add_argument("--tune", default=os.path.join(HERE, "tune_D.json"))
    r.add_argument("--out", default=os.path.join(HERE, "ABLATION_D.md"))
    r.add_argument("--half", default=os.path.join(HERE, "halfcheck_D.json"))
    r.add_argument("--variant", default=os.path.join(HERE, "tune_D_B4.json"))
    r.add_argument("--oracle", default=os.path.join(HERE, "oracle_D_B4.json"))
    o = sub.add_parser("oracle")
    o.add_argument("--tune", default=os.path.join(HERE, "tune_D_B4.json"))
    o.add_argument("--out", default=os.path.join(HERE, "oracle_D_B4.json"))
    v = sub.add_parser("variant")
    v.add_argument("--structure", default=os.path.join(HERE, "tune_D.json"))
    v.add_argument("--kept", required=True, help="comma-separated subset of the kept modules, e.g. B4")
    v.add_argument("--out", required=True)
    h = sub.add_parser("halfcheck")
    h.add_argument("--tune", default=os.path.join(HERE, "tune_D.json"))
    h.add_argument("--out", default=os.path.join(HERE, "halfcheck_D.json"))
    t = sub.add_parser("retune", help="main session: another checkpoint, same structure (PLAN.md 6)")
    t.add_argument("--structure", default=os.path.join(HERE, "tune_D.json"))
    t.add_argument("--emb-D", required=True)
    t.add_argument("--base-tune", required=True, help="prereg36-format tune.json of the adopted system")
    t.add_argument("--out", required=True)
    j = sub.add_parser("judge", help="main session: the one judgement on the E pools (PLAN.md 6)")
    j.add_argument("--tune", default=os.path.join(HERE, "tune_D.json"))
    j.add_argument("--emb-E", required=True)
    j.add_argument("--base-tune", default=P36_TUNE)
    j.add_argument("--base-emb-E", required=True)
    j.add_argument("--base-eval", default=None, help="prereg36-format eval.json: seam of the adopted system")
    j.add_argument("--out", default=os.path.join(HERE, "eval.json"))
    q = sub.add_parser("apply", help="main session: an official new pool (after an adopting judgement)")
    q.add_argument("--src", required=True)
    q.add_argument("--pool", required=True, choices=["public", "private"])
    q.add_argument("--emb", required=True)
    q.add_argument("--tune", default=os.path.join(HERE, "tune_D.json"))
    q.add_argument("--eval", default=os.path.join(HERE, "eval.json"))
    q.add_argument("--out", default=os.path.join(REPO, "results", "prereg38", "apply"))
    a = ap.parse_args(argv)
    if a.cmd not in ("report",):
        polite()
        wait_for_ram()
    return {"seam": cmd_seam, "single": cmd_single, "greedy": cmd_greedy, "report": cmd_report,
            "halfcheck": cmd_halfcheck, "variant": cmd_variant, "oracle": cmd_oracle,
            "retune": cmd_retune, "judge": cmd_judge, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
