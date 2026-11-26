"""prereg56 addition 2 (H8): are the opponents labelled training players?  CPU."""
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


def main():
    import torch
    ck = torch.load(os.path.join(R, "data", "prereg49", "S0", "ckpt", "step_320000.pt"), map_location="cpu",
                    weights_only=False)
    W = P.l2(ck["head"]["weight"].numpy()).astype(np.float32)
    del ck
    cache = P49.Cache49()
    T = R49.t_index49(cache)[0]
    assert len(T) == len(W)
    with np.load(os.path.join(P.FEATS, "meta.npz"), allow_pickle=False) as z:
        mpid, mrank = z["player_id"], z["rank"]
    rank_of = {}
    for p, r in zip(np.asarray(cache.pid).tolist(), np.asarray(cache.rank).tolist()):
        rank_of[p] = str(r)             # (fix: the feature store's meta only covers part of the players)
    names = sorted(set(rank_of.values()))
    print("rank names", names, flush=True)
    order = {n: i for i, n in enumerate(RANKS)} if set(names) <= set(RANKS) else {n: i for i, n in enumerate(names)}
    cls_rank = np.array([order.get(rank_of.get(p, ""), -1) for p in T])

    def mx(emb):
        out, arg = np.zeros(len(emb), np.float32), np.zeros(len(emb), np.int64)
        for s in range(0, len(emb), 4000):
            S = emb[s:s + 4000].astype(np.float32) @ W.T
            out[s:s + 4000], arg[s:s + 4000] = S.max(1), S.argmax(1)
        return out, arg

    def load(*parts):
        return P.load_emb(os.path.join(R, "data", *parts))

    E, _ = load("prereg49", "emb", TAG, "store_E.npz")
    D, rowsD = load("prereg56", "emb_D", TAG, "store_D.npz")
    oD, _ = load("prereg56", "emb_opp_D", TAG, "store_D.npz")
    rp = np.concatenate([load("prereg56", "emb_rank", TAG, "pool_%s.npz" % s)[0] for s in ("public", "private")])
    ro = np.concatenate([load("prereg56", "emb_opp_rank", TAG, "pool_%s.npz" % s)[0] for s in ("public", "private")])
    rng = np.random.default_rng(0)
    nE, _ = mx(E[rng.choice(len(E), 40000, replace=False)])
    t99, t50 = float(np.quantile(nE, 0.99)), float(np.median(nE))
    res = {"null_E_single_game": {"p50": t50, "p95": float(np.quantile(nE, 0.95)), "p99": t99}}
    for name, emb in (("D_player_side(optimistic positive)", D), ("D_opponent_side", oD),
                      ("rank_test_opponent_side", ro), ("rank_test_player_side", rp)):
        m, a = mx(emb)
        res[name] = {"n": int(len(m)), "p50": float(np.median(m)), "p95": float(np.quantile(m, 0.95)),
                     "frac_above_null_p99": float((m > t99).mean()), "frac_above_null_p95":
                     float((m > np.quantile(nE, 0.95)).mean())}
        if name == "D_opponent_side":
            drank = np.array([order.get(rank_of[p], -1) for p in mpid[rowsD].tolist()])
            diff = cls_rank[a] - drank
            for lab, sel in (("above_null_p99", m > t99), ("below_null_p50", m < t50)):
                d = diff[sel]
                res[name]["rank_diff_%s" % lab] = {"n": int(sel.sum()), "frac_0": float((d == 0).mean()),
                                                   "frac_abs_le1": float((np.abs(d) <= 1).mean()),
                                                   "mean_abs": float(np.abs(d).mean())}
        print(name, json.dumps(res[name]), flush=True)
    print(json.dumps(res["null_E_single_game"]))
    with open(os.path.join(HERE, "h8_fixed.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
