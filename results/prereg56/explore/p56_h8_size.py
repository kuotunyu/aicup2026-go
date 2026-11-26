"""prereg56 addition 3: how much rank information is in 'who is the opponent'?  D players, opponent side, CPU."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p56_probe as P  # noqa: E402

R, TAG = P.REPO, P.TAG
sys.path.insert(0, os.path.join(R, "results", "prereg49", "explore"))
import p49_run as R49  # noqa: E402
import p49_lib as P49  # noqa: E402

RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
NR = 10


def score_matrix():
    S = np.zeros((NR, NR))
    for i in range(NR):
        for j in range(NR):
            S[i, j] = 1.0 if i == j else (np.exp(-1.0) if abs(i - j) == 1 else 0.0)
    return S


def main():
    import torch
    ck = torch.load(os.path.join(R, "data", "prereg49", "S0", "ckpt", "step_320000.pt"), map_location="cpu",
                    weights_only=False)
    W = P.l2(ck["head"]["weight"].numpy()).astype(np.float32)
    del ck
    cache = P49.Cache49()
    T = R49.t_index49(cache)[0]
    with np.load(os.path.join(P.FEATS, "meta.npz"), allow_pickle=False) as z:
        mpid = z["player_id"]
    order = {n: i for i, n in enumerate(RANKS)}
    rank_of = {p: order[str(r)] for p, r in zip(np.asarray(cache.pid).tolist(), np.asarray(cache.rank).tolist())}
    cls_rank = np.array([rank_of[p] for p in T])
    oD, rows = P.load_emb(os.path.join(R, "data", "prereg56", "emb_opp_D", TAG, "store_D.npz"))
    gp = mpid[rows]
    players = sorted(set(gp.tolist()))
    keep = np.array([p not in set(players) for p in T])          # D's own classes removed
    Wk, rk = W[keep], cls_rank[keep]
    M = np.zeros((int(keep.sum()), NR), np.float32)
    M[np.arange(len(rk)), rk] = 1.0
    taus = [10, 20, 30, 50]
    Q = {t: np.zeros((len(oD), NR), np.float32) for t in taus}
    for s in range(0, len(oD), 4000):
        C = oD[s:s + 4000].astype(np.float32) @ Wk.T
        for t in taus:
            Z = t * C
            Z -= Z.max(1, keepdims=True)
            Pz = np.exp(Z)
            Pz /= Pz.sum(1, keepdims=True)
            Q[t][s:s + 4000] = Pz @ M
    prank = np.array([rank_of[p] for p in players])
    by = {p: np.flatnonzero(gp == p) for p in players}
    grank = np.array([rank_of[p] for p in gp.tolist()])
    rng = np.random.default_rng(56)
    fold = rng.permutation(len(players)) % 5
    SM = score_matrix()
    res = {"players": len(players), "games": int(len(oD)), "rank_counts": np.bincount(prank, minlength=NR).tolist()}

    def kernel(q, idx):
        K = np.full(2 * NR - 1, 1e-3)
        for d in range(-(NR - 1), NR):
            s = grank[idx] + d
            ok = (s >= 0) & (s < NR)
            K[d + NR - 1] += q[idx[ok], s[ok]].sum()
        return K / K.sum()

    def loglik(q, K, g):
        # L[r] = sum_g log sum_s q[g, s] K[s - r]
        out = np.zeros(NR)
        for r in range(NR):
            k = np.array([K[s - r + NR - 1] for s in range(NR)])
            out[r] = np.log(q[g] @ k + 1e-9).sum()
        return out

    per_n = {n: [] for n in (5, 10, 20)}
    prior_only = []
    chosen = []
    for f in range(5):
        tr = [p for p, ff in zip(players, fold) if ff != f]
        te = [p for p, ff in zip(players, fold) if ff == f]
        tri = np.concatenate([by[p] for p in tr])
        prior = np.bincount([rank_of[p] for p in tr], minlength=NR) + 1.0
        prior = prior / prior.sum()
        best = None
        for t in taus:
            K = kernel(Q[t], tri)
            k_g = np.array([[K[s - r + NR - 1] for s in range(NR)] for r in range(NR)])   # [r, s]
            ll = float(np.log((Q[t][tri] * k_g[grank[tri]]).sum(1) + 1e-9).mean())
            if best is None or ll > best[0]:
                best = (ll, t, K)
        _ll, t, K = best
        chosen.append(t)
        prior_only += [float(SM[int(np.argmax(prior @ SM)), rank_of[p]]) for p in te]
        for p in te:
            for n in per_n:
                if len(by[p]) < n:
                    continue
                for _ in range(20):
                    g = rng.choice(by[p], n, replace=False)
                    post = np.log(prior) + loglik(Q[t], K, g)
                    post = np.exp(post - post.max())
                    post /= post.sum()
                    per_n[n].append(float(SM[int(np.argmax(post @ SM)), rank_of[p]]))
    res["tau_chosen_per_fold"] = chosen
    res["score_prior_only"] = float(np.mean(prior_only))
    for n in per_n:
        res["score_opponent_id_only_n%d" % n] = float(np.mean(per_n[n]))
    print(json.dumps(res, indent=1))
    with open(os.path.join(HERE, "h8_size.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
