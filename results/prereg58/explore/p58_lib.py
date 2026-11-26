"""prereg58 (competition 2): shared parts of the opponent-identity term (PLAN.md "Method"). numpy only.

  q_g[s]      = sum over S0 classes j of rank s of softmax_j(tau * cos(o_g, w_j))          class_q()
  K[h, c][d]  = P(opponent's class rank - player's rank = d | HD group h, player's colour c)  kernel_counts() / kernel()
  l_g(r)      = log(sum_s q_g[s] K[h_g, c_g][s - r] + 1e-9)                                game_ll()
  l(r)        = sum over the question's games of l_g(r)

Never the validation answers (guard()), never competition-1 test data, never tools/score.py.
"""
import hashlib
import json
import os
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "data", "prereg58")
TAG = "V2_step_320000_raw_tta1"
S0_CKPT = os.path.join(REPO, "data", "prereg43", "S0", "ckpt", "step_320000.pt")
HEAD_NPZ = os.path.join(DATA, "s0_head.npz")
OPP_D = os.path.join(DATA, "emb_opp_D", TAG, "store_D.npz")
OPP_E = os.path.join(DATA, "emb_opp_E", TAG, "store_E.npz")
OPP_INNER = os.path.join(DATA, "emb_opp_inner", TAG, "rows_inner.npz")
INNER_ROWS = os.path.join(DATA, "inner_rows.npy")
INNER_CACHE = os.path.join(REPO, "data", "prereg44", "cache", "inner")     # prereg36 cache format, inner_keys order
INNER_KEYS = os.path.join(REPO, "results", "prereg02", "rank", "dumps", "inner_keys.npz")
FEATS_META = os.path.join(REPO, "data", "prereg06", "feats", "meta.npz")   # (rank, game_id, player_id, color) of store rows
KERNEL = os.path.join(HERE, "kernel.json")
SEAM = os.path.join(HERE, "seam.json")
EVAL = os.path.join(HERE, "eval.json")
STARTED = os.path.join(DATA, "eval_started.json")

RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")     # weak -> strong, index 0..9
L = len(RANKS)
ND = 2 * L - 1                       # rank differences -9 .. +9
TAUS = (20, 30, 50, 80, 120)         # PLAN.md
SMOOTH = 0.5                         # add-0.5 on every cell of K[h, c][.]
EPS = 1e-9
N_HD, N_COL = 3, 2                   # HD group 0 / 1 / 2+; player's colour B = 0, W = 1
FOLDS = 5
REF_Q25 = 0.7456519871546352
REF_Q25_MONO = 0.7482450052195216
GATE_G1 = 0.005
RAM_LIMIT_GB = 3.0
CORES = (0, 1, 2, 3)

FORBIDDEN = ("/data/local", "rank_val_answers", "player_val_answers", "/data/test", "pool_public", "pool_private",
             "/emb_opp_rank")


def guard(p):
    """Refuse the validation answers and every test-set file (this experiment builds no test prediction)."""
    q = os.path.abspath(os.fspath(p)).replace("\\", "/").lower()
    if any(f in q for f in FORBIDDEN):
        raise SystemExit(f"prereg58 guard: refused path {p}")
    return p


def limits(log=print):
    """The shared-machine rules: this process on 4 cores (affinity; no BLAS variable is set, so the number of
    OpenBLAS threads and therefore the arithmetic stay those of the earlier inner evaluations) and a watchdog that
    ends the process when its working set passes RAM_LIMIT_GB. -> a dict whose 'peak' is updated (bytes)."""
    import psutil
    proc = psutil.Process()
    proc.cpu_affinity(list(CORES))
    state = {"peak": 0}

    def run():
        while True:
            ws = proc.memory_info().rss
            state["peak"] = max(state["peak"], ws)
            if ws > RAM_LIMIT_GB * 2 ** 30:
                log(f"prereg58 watchdog: working set {ws / 2 ** 30:.2f} GB > {RAM_LIMIT_GB} GB - stopping")
                os._exit(97)
            time.sleep(0.1)

    threading.Thread(target=run, daemon=True).start()
    return state


def sha256_file(p):
    h = hashlib.sha256()
    with open(guard(p), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(p):
    with open(guard(p), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(guard(tmp), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(guard(path), encoding="utf-8") as f:
        return json.load(f)


def rank_index(ranks):
    idx = {r: i for i, r in enumerate(RANKS)}
    return np.array([idx[str(x)] for x in ranks], np.int64)


def colour_index(col):
    col = np.asarray(col).astype(str)
    out = np.where(col == "B", 0, np.where(col == "W", 1, -1)).astype(np.int64)
    if (out < 0).any():
        raise ValueError("a colour that is not B / W")
    return out


def hd_group(hd):
    hd = np.asarray(hd, np.int64)
    if (hd < 0).any():
        raise ValueError("a game without HD")
    return np.minimum(hd, 2)


def player_fold(pid):
    return int(hashlib.sha256(f"c10-prereg58|fold|{pid}".encode("utf-8")).hexdigest()[:8], 16) % FOLDS


# ---------------------------------------------------------------- the S0 class head
def load_head(path=HEAD_NPZ):
    """(Wn [C, 256] float64 L2-normalised class weights, class rank [C], record)."""
    with np.load(guard(path), allow_pickle=False) as z:
        W = np.asarray(z["weight"], np.float64)
        cr = np.asarray(z["class_rank"], np.int64)
        rec = {"classes": int(len(W)), "ckpt_sha256": str(z["ckpt_sha256"]), "ckpt_step": int(z["ckpt_step"]),
               "class_ids_sha256": str(z["class_ids_sha256"]), "sha256": None,
               "classes_per_rank": np.bincount(cr, minlength=L).tolist()}
    rec["sha256"] = sha256_file(path)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    return Wn, cr, rec


def class_q(emb, Wn, cls_rank, taus, block=4000):
    """{tau: q [G, 10] float64}: the soft-max over the S0 classes of tau * cos(game vector, class weight), summed by
    the class's rank. A zero vector (a game without a sample) gets the classes-per-rank shares; callers mask it."""
    emb = np.asarray(emb)
    G = len(emb)
    M = np.zeros((len(cls_rank), L))
    M[np.arange(len(cls_rank)), cls_rank] = 1.0
    out = {t: np.zeros((G, L)) for t in taus}
    for s in range(0, G, block):
        e = np.asarray(emb[s:s + block], np.float64)
        nrm = np.linalg.norm(e, axis=1, keepdims=True)
        C = (e / np.where(nrm > 0, nrm, 1.0)) @ Wn.T
        for t in taus:
            Z = t * C
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            out[t][s:s + block] = P @ M
    return out


# ---------------------------------------------------------------- the kernel
def kernel_counts(q, r, h, c, mask=None):
    """Soft counts [3, 2, 19]: cell (h, c, d + 9) += q_g[r_g + d] for every game g (r_g = the player's rank); each
    game adds a total of 1 (the opponent's class rank is not observed, q_g is its posterior)."""
    q, r, h, c = np.asarray(q), np.asarray(r), np.asarray(h), np.asarray(c)
    if mask is None:
        mask = np.ones(len(r), bool)
    C = np.zeros((N_HD, N_COL, ND))
    for s in range(L):
        d = s - r + (L - 1)
        np.add.at(C, (h[mask], c[mask], d[mask]), q[mask, s])
    return C


def kernel(counts):
    K = np.asarray(counts, np.float64) + SMOOTH
    return K / K.sum(axis=2, keepdims=True)


def kernel_matrix(K):
    """M[h, c][r, s] = K[h, c][s - r + 9]."""
    r = np.arange(L)[:, None]
    s = np.arange(L)[None, :]
    return np.asarray(K)[:, :, s - r + (L - 1)]


def game_ll(q, K, h, c):
    """l_g(r) = log(sum_s q_g[s] K[h_g, c_g][s - r] + 1e-9) -> [G, 10]."""
    q, h, c = np.asarray(q, np.float64), np.asarray(h), np.asarray(c)
    M = kernel_matrix(K)
    out = np.zeros((len(q), L))
    for hh in range(N_HD):
        for cc in range(N_COL):
            m = (h == hh) & (c == cc)
            if m.any():
                out[m] = q[m] @ M[hh, cc].T
    return np.log(out + EPS)


# ---------------------------------------------------------------- the competition score
def score_matrix():
    i = np.arange(L)
    d = np.abs(i[:, None] - i[None, :])
    return np.where(d == 0, 1.0, np.where(d == 1, np.exp(-1.0), 0.0))


def decode(logpost):
    """Answer with the largest expected competition score under softmax(logpost) -> [N] rank index."""
    z = np.asarray(logpost, np.float64)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return np.argmax(p @ score_matrix(), axis=1)


def rank_balanced(sc, y):
    sc, y = np.asarray(sc, np.float64), np.asarray(y)
    return float(np.mean([sc[y == k].mean() for k in range(L) if (y == k).any()]))
