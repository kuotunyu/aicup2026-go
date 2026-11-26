"""prereg56 addition 1: are the opponents a few recurring identities?  CPU.
    PY results/prereg56/explore/p56_opp_probe.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p56_probe as P  # noqa: E402

R = P.REPO
TAG = P.TAG


def knn_stat(X, grp, k=10, block=2000):
    """cosine to the k-th nearest neighbour among vectors of other groups; and the count of neighbours above thr."""
    n = len(X)
    out = np.zeros(n)
    top = []
    for s in range(0, n, block):
        S = X[s:s + block] @ X.T
        S[grp[s:s + block, None] == grp[None, :]] = -1.0
        part = np.partition(S, -k, axis=1)[:, -k]
        out[s:s + block] = part
        top.append(S)
        del S
        top.pop()
    return out


def count_above(X, grp, thr, block=2000):
    n = len(X)
    c = np.zeros(n, np.int64)
    for s in range(0, n, block):
        S = X[s:s + block] @ X.T
        S[grp[s:s + block, None] == grp[None, :]] = -1.0
        c[s:s + block] = (S > thr).sum(1)
    return c


def pair_thr(X, grp, q=0.999, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(X), n), rng.integers(0, len(X), n)
    m = grp[i] != grp[j]
    return float(np.quantile(np.einsum("ij,ij->i", X[i[m]], X[j[m]]), q))


def run(name, Pv, Ov, grp, res):
    thr = pair_thr(Pv, grp)
    kp, ko = knn_stat(Pv, grp), knn_stat(Ov, grp)
    cp, co = count_above(Pv, grp, thr), count_above(Ov, grp, thr)
    r = {"games": int(len(Pv)), "groups": int(len(set(grp.tolist()))), "thr_p999_player_pairs": thr,
         "player_side": {"knn10_median": float(np.median(kp)), "knn10_p90": float(np.quantile(kp, 0.9)),
                         "nbrs_above_thr_median": float(np.median(cp)), "nbrs_above_thr_p90": float(np.quantile(cp, 0.9)),
                         "nbrs_above_thr_max": int(cp.max())},
         "opponent_side": {"knn10_median": float(np.median(ko)), "knn10_p90": float(np.quantile(ko, 0.9)),
                           "nbrs_above_thr_median": float(np.median(co)), "nbrs_above_thr_p90": float(np.quantile(co, 0.9)),
                           "nbrs_above_thr_max": int(co.max()),
                           "frac_games_with_ge50_nbrs": float((co >= 50).mean()),
                           "frac_games_with_ge10_nbrs": float((co >= 10).mean())},
         "player_frac_ge10": float((cp >= 10).mean()), "player_frac_ge50": float((cp >= 50).mean())}
    res[name] = r
    print(name, json.dumps(r), flush=True)


def main():
    res = {}
    Pv, Ov, grp, g0 = [], [], [], 0
    for split in ("public", "private"):
        p, _ = P.load_emb(os.path.join(R, "data", "prereg56", "emb_rank", TAG, "pool_%s.npz" % split))
        o, _ = P.load_emb(os.path.join(R, "data", "prereg56", "emb_opp_rank", TAG, "pool_%s.npz" % split))
        _c, q, _col, n = P.pool_layout(None, os.path.join(P.RK, "rank_prediction_test_%s.csv" % split))
        g = np.zeros(n, np.int64)
        for k, rows in enumerate(q.values()):
            g[rows] = g0 + k
        g0 += len(q)
        Pv.append(p), Ov.append(o), grp.append(g)
    run("rank_test", np.concatenate(Pv), np.concatenate(Ov), np.concatenate(grp), res)
    pD, rows = P.load_emb(os.path.join(R, "data", "prereg56", "emb_D", TAG, "store_D.npz"))
    oD, rows2 = P.load_emb(os.path.join(R, "data", "prereg56", "emb_opp_D", TAG, "store_D.npz"))
    assert (rows == rows2).all()
    with np.load(os.path.join(P.FEATS, "meta.npz"), allow_pickle=False) as z:
        pid = z["player_id"][rows]
    _u, g = np.unique(pid, return_inverse=True)
    run("D_store", pD, oD, g, res)
    with open(os.path.join(HERE, "opp_probe.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
