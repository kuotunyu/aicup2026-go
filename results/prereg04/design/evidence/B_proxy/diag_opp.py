"""Opponent-side proxy (o_choice_a20): does its gain survive on top of the prereg03 CNN ensembles, and in which game
phase is it? Inner 5-fold OOF, rule B, registered calibration form (all w in [0, 4]), as fuse.py."""
import json, os, sys, types
sys.dont_write_bytecode = True
import numpy as np
import fuse as F
rf, p3 = F.rf, F.p3
c = types.SimpleNamespace(p02_dumps=os.path.join(p3.P02_RANK, "dumps"), p02_local=p3.P02_RANK, split=p3.SPLIT_CSV,
                          smoke=False)
inner = p3.prepare_inner(c)
keys, Xf, Xe, Sf, Se, boot = inner["keys"], inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"], inner["boot"]
n = keys["n"]
logits = {p3.C5_DUMP: p3.load_c5_inner(c, n)["logits"]}
for arm, name in (("a", "p03a_d4"), ("b", "p03b_d4")):
    logits[name] = rf.load_dump(os.path.join(p3.P03_ROOT, f"rank_{arm}", "dumps", f"inner_{name}.npz"), n, 8)["logits"]
lp = {cid: p3.combo_logprobs(dict(p3.COMBOS)[cid], logits) for cid in ("P00", "P03", "P04")}
z = np.load("proxy_inner.npz")
row = {(r, g): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
idx = np.array([row[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
V = [str(v) for v in z["variants"]]
ll = z["ll"][idx]
o = V.index("o_choice_a20")
px = {"o": rf.log_softmax(ll[:, o].sum(1))}
for ph, nm in enumerate(("open", "mid", "late")):
    px["o_" + nm] = rf.log_softmax(ll[:, o, ph])
mf = {k: rf.pooled(Sf, v, "log") for k, v in lp.items()}
me = {k: rf.pooled(Se, v, "log") for k, v in lp.items()}
pf = {k: np.asarray(Sf @ v) for k, v in px.items()}
pe = {k: np.asarray(Se @ v) for k, v in px.items()}
opf, ope = Xf.mop, Xe.mop
P0, _ = F.gen_oof(Xf, Xe, mf["P00"], me["P00"], [opf], [ope])
m0, sc0 = rf.metrics(rf.rule_b_np(P0), Xe.y, Xe.n, Xe.player)
out = {"R20": m0["rank_balanced"]}
runs = [("P03 (C5+C6A+C6B, opening)", "P03", [opf], [ope]),
        ("P03 + PX[o]", "P03", [opf, pf["o"]], [ope, pe["o"]]),
        ("R20 + C6A (sep w) + PX[o]", "P00", [opf, mf["P04"], pf["o"]], [ope, me["P04"], pe["o"]]),
        ("R20 + PX[o] by phase (3 w)", "P00", [opf] + [pf["o_" + k] for k in ("open", "mid", "late")],
         [ope] + [pe["o_" + k] for k in ("open", "mid", "late")]),
        ("R10 + PX[o] (no opening)", "P00", [pf["o"]], [pe["o"]])]
for name, cid, Ef, Ee in runs:
    P, ws = F.gen_oof(Xf, Xe, mf[cid], me[cid], Ef, Ee)
    m, sc = rf.metrics(rf.rule_b_np(P), Xe.y, Xe.n, Xe.player)
    d = boot.stats(sc) - boot.stats(sc0)
    out[name] = {"rank_balanced": m["rank_balanced"], "diff_vs_R20": m["rank_balanced"] - m0["rank_balanced"],
                 "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                 "question_mean_diff": float(np.mean(sc - sc0)),
                 "per_n_bin": {k: v.get("rank_balanced") for k, v in m["per_n_bin"].items()},
                 "weights_by_fold": [[round(x, 4) for x in w] for w in ws]}
    print(name, json.dumps(out[name]), flush=True)
out["R20_per_n_bin"] = {k: v.get("rank_balanced") for k, v in m0["per_n_bin"].items()}
print("R20 per n bin", out["R20_per_n_bin"])
json.dump(out, open("diag_opp.json", "w"), indent=1)
