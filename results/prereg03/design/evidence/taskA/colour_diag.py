"""Inner-only diagnostic (training players; no validation data): does the epoch-9 model's
per-game output carry target-colour-dependent error that a colour-specific bias fixes?

Reads results/prereg02/rank/dumps/inner_keys.npz and inner_p02e9_d4.npz (read-only).
Player-level 2-fold split (hash of player_id) to fit a per-colour 10-dim bias + temperature
on one half and evaluate NLL / accuracy on the other half.
"""
import hashlib
import json
import os
import sys

import numpy as np
from scipy import optimize

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
D = os.path.join(REPO, "results", "prereg02", "rank", "dumps")
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def lsm(x):
    m = x.max(-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


k = np.load(os.path.join(D, "inner_keys.npz"), allow_pickle=False)
y = k["label"].astype(np.int64)
col = np.array([str(c) for c in k["color"]])
pid = np.array([str(p) for p in k["player_id"]])
z = np.load(os.path.join(D, "inner_p02e9_d4.npz"), allow_pickle=False)["logits"].astype(np.float64)
l0 = lsm(z[:, 0, :])
l8 = lsm(np.mean([lsm(z[:, s, :]) for s in range(8)], axis=0))
isB = col == "B"
out = {"n": int(len(y)), "frac_black": float(isB.mean())}

for name, l in (("s0", l0), ("d4", l8)):
    r = {}
    for tag, m in (("all", np.ones_like(isB)), ("B", isB), ("W", ~isB)):
        r[tag] = {"n": int(m.sum()), "acc": float((l[m].argmax(1) == y[m]).mean()),
                  "nll": float(-l[m][np.arange(m.sum()), y[m]].mean())}
    # mean predicted distribution vs true distribution per colour
    r["mean_pred_B"] = np.exp(l[isB]).mean(0).round(4).tolist()
    r["mean_pred_W"] = np.exp(l[~isB]).mean(0).round(4).tolist()
    r["true_dist_B"] = (np.bincount(y[isB], minlength=10) / isB.sum()).round(4).tolist()
    r["true_dist_W"] = (np.bincount(y[~isB], minlength=10) / (~isB).sum()).round(4).tolist()
    out[name] = r

# player-level 2-fold: fit (T, bias_B[10], bias_W[10]) vs (T, bias[10]) on half, eval on other half
half = np.array([int(hashlib.sha256(("c10-p03diag|" + p).encode()).hexdigest()[:8], 16) % 2 for p in pid])


def nll_fn(theta, l, y, isB, per_colour):
    T = np.exp(theta[0])
    if per_colour:
        bB, bW = theta[1:11], theta[11:21]
        b = np.where(isB[:, None], bB[None, :], bW[None, :])
    else:
        b = theta[1:11][None, :]
    zz = lsm(T * l + b)
    return -zz[np.arange(len(y)), y].mean()


res = {}
for name, l in (("s0", l0), ("d4", l8)):
    rr = {}
    for per_colour in (False, True):
        tot_nll, tot_acc, n = 0.0, 0.0, 0
        for f in (0, 1):
            tr, ev = half != f, half == f
            k0 = 21 if per_colour else 11
            r = optimize.minimize(nll_fn, np.zeros(k0), args=(l[tr], y[tr], isB[tr], per_colour),
                                  method="L-BFGS-B", options={"maxiter": 300})
            T = np.exp(r.x[0])
            if per_colour:
                b = np.where(isB[ev][:, None], r.x[1:11][None, :], r.x[11:21][None, :])
            else:
                b = r.x[1:11][None, :]
            zz = lsm(T * l[ev] + b)
            tot_nll += -zz[np.arange(ev.sum()), y[ev]].sum()
            tot_acc += (zz.argmax(1) == y[ev]).sum()
            n += ev.sum()
        rr["per_colour" if per_colour else "shared"] = {"nll": tot_nll / n, "acc": tot_acc / n}
    res[name] = rr
out["bias_2fold"] = res
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "colour_diag.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
