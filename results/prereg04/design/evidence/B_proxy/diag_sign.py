"""Diagnosis only (not a method): let the proxy weights take either sign, to see where the target-side proxy's
optimum lies given R20 (the registered calibration keeps w >= 0). Inner 5-fold OOF, rule B, as fuse.py."""
import json, math, os, sys, types
sys.dont_write_bytecode = True
import numpy as np
from scipy import optimize
import fuse as F
rf, p3 = F.rf, F.p3


def fit_b(m, Es, n, y, wb):
    J = len(Es)
    wq = rf.level_weights(y)
    bnds = [rf.BOUNDS["a"], rf.BOUNDS["gamma"]] + wb + [(None, None)] * 9
    best = None
    for a0 in rf.STARTS:
        x0 = np.zeros(2 + J + 9); x0[0] = a0
        r = optimize.minimize(F.gen_obj, x0, args=(m, Es, n, y, wq), jac=True, method="L-BFGS-B", bounds=bnds,
                              options={"maxiter": rf.MAXITER})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def oof_b(Xf, Xe, mf, me, Ef, Ee, wb):
    P = np.zeros((len(Xe), 10)); ws = []
    for f in range(5):
        ev, tr = Xe.fold == f, Xf.fold != f
        th = fit_b(mf[tr], [E[tr] for E in Ef], Xf.n[tr], Xf.y[tr], wb)
        z, _ = F.gen_logits(th, me[ev], [E[ev] for E in Ee], Xe.n[ev])
        P[ev] = rf.softmax(z); ws.append([round(float(x), 4) for x in th[2:2 + len(Ef)]])
    return P, ws


c = types.SimpleNamespace(p02_dumps=os.path.join(p3.P02_RANK, "dumps"), p02_local=p3.P02_RANK, split=p3.SPLIT_CSV,
                          smoke=False)
inner = p3.prepare_inner(c)
keys, Xf, Xe, Sf, Se, boot = inner["keys"], inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"], inner["boot"]
lp = p3.combo_logprobs(("C5",), {p3.C5_DUMP: p3.load_c5_inner(c, keys["n"])["logits"]})
z = np.load("proxy_inner.npz")
row = {(r, g): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
idx = np.array([row[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
V = [str(v) for v in z["variants"]]
ll = z["ll"][idx]
px = {v: rf.log_softmax(ll[:, V.index(v)].sum(1)) for v in ("t_choice_a20", "o_choice_a20")}
mf, me = rf.pooled(Sf, lp, "log"), rf.pooled(Se, lp, "log")
pf = {v: np.asarray(Sf @ o) for v, o in px.items()}
pe = {v: np.asarray(Se @ o) for v, o in px.items()}
opf, ope = Xf.mop, Xe.mop
P0, _ = oof_b(Xf, Xe, mf, me, [opf], [ope], [(0.0, 4.0)])
m0, sc0 = rf.metrics(rf.rule_b_np(P0), Xe.y, Xe.n, Xe.player)
out = {"R20": m0["rank_balanced"]}
for name, Ef, Ee, wb in (
        ("R20 + PX[t] w free sign", [opf, pf["t_choice_a20"]], [ope, pe["t_choice_a20"]], [(0.0, 4.0), (-4.0, 4.0)]),
        ("R20 + PX[t] + PX[o] both free sign", [opf, pf["t_choice_a20"], pf["o_choice_a20"]],
         [ope, pe["t_choice_a20"], pe["o_choice_a20"]], [(0.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)])):
    P, ws = oof_b(Xf, Xe, mf, me, Ef, Ee, wb)
    m, sc = rf.metrics(rf.rule_b_np(P), Xe.y, Xe.n, Xe.player)
    d = boot.stats(sc) - boot.stats(sc0)
    out[name] = {"rank_balanced": m["rank_balanced"], "diff_vs_R20": m["rank_balanced"] - m0["rank_balanced"],
                 "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                 "question_mean_diff": float(np.mean(sc - sc0)), "weights_by_fold": ws}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open("diag_sign.json", "w"), indent=1)
