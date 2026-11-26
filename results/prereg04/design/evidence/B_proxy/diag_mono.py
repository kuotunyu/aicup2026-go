"""Single-colour inner pseudo-questions (prereg03 rev 3 (a), p03_rank_fuse.mono_set: 5/10/15/20 games of one colour,
like the official competition-2 questions). Parameters are fitted per fold on the registered mixed-colour fit
questions (as fuse.py / p03 tune) and applied to the single-colour questions of that fold's players. Needs the proxy
of every inner game (proxy_inner_all.npz). Record only."""
import json, os, sys, types
sys.dont_write_bytecode = True
import numpy as np
import fuse as F
rf, p3 = F.rf, F.p3
c = types.SimpleNamespace(p02_dumps=os.path.join(p3.P02_RANK, "dumps"), p02_local=p3.P02_RANK, split=p3.SPLIT_CSV,
                          smoke=False)
inner = p3.prepare_inner(c)
keys, Xf, Sf = inner["keys"], inner["X_fit"], inner["S_fit"]
ms = p3.mono_set(inner)
Xm, Sm = p3.mono_qset(inner, ms)
print("mono questions", len(Xm), ms["sha256"][:12], flush=True)
lp = p3.combo_logprobs(("C5",), {p3.C5_DUMP: p3.load_c5_inner(c, keys["n"])["logits"]})
z = np.load("proxy_inner_all.npz")
assert z["scored"].all()
row = {(r, g): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
idx = np.array([row[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
V = [str(v) for v in z["variants"]]
ll = z["ll"][idx]
px = {v: rf.log_softmax(ll[:, V.index(v)].sum(1)) for v in ("t_choice_a20", "o_choice_a20")}
# the bank-only proxy file must agree with this one on the bank games
zb = np.load("proxy_inner.npz")
sb = zb["scored"]
same = bool(np.allclose(zb["ll"][sb], z["ll"][sb], rtol=0, atol=1e-9))
mf, mm = rf.pooled(Sf, lp, "log"), rf.pooled(Sm, lp, "log")
pf = {v: np.asarray(Sf @ o) for v, o in px.items()}
pm = {v: np.asarray(Sm @ o) for v, o in px.items()}
boot = rf.PlayerBootstrap(ms["player"], ms["y"], ms["n_players"])


def oof_mono(Ef, Em):
    P = np.zeros((len(Xm), 10))
    for f in range(5):
        tr, ev = Xf.fold != f, Xm.fold == f
        th = F.gen_fit(mf[tr], [E[tr] for E in Ef], Xf.n[tr], Xf.y[tr])
        zz, _ = F.gen_logits(th, mm[ev], [E[ev] for E in Em], Xm.n[ev])
        P[ev] = rf.softmax(zz)
    return P


out = {"questions": len(Xm), "mono_sha256": ms["sha256"], "bank_scores_equal_all_file": same}
res = {}
for name, Ef, Em in (("R20", [Xf.mop], [Xm.mop]),
                     ("R20 + PX[o_choice_a20]", [Xf.mop, pf["o_choice_a20"]], [Xm.mop, pm["o_choice_a20"]]),
                     ("R20 + PX[t_choice_a20]", [Xf.mop, pf["t_choice_a20"]], [Xm.mop, pm["t_choice_a20"]])):
    P = oof_mono(Ef, Em)
    m, sc = rf.metrics(rf.rule_b_np(P), Xm.y, Xm.n, Xm.player)
    res[name] = sc
    rec = {"rank_balanced": m["rank_balanced"], "question_mean": m["question_mean"],
           "by_colour": {col: p3.score_summary(sc, Xm.y, ms["colour"] == col)["rank_balanced"] for col in ("B", "W")},
           "by_k": {str(k): p3.score_summary(sc, Xm.y, ms["k"] == k)["rank_balanced"] for k in (5, 10, 15, 20)}}
    if name != "R20":
        d = boot.stats(sc) - boot.stats(res["R20"])
        rec["diff_vs_R20"] = m["rank_balanced"] - out["R20"]["rank_balanced"]
        rec["ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        rec["question_mean_diff"] = float(np.mean(sc - res["R20"]))
        for col in ("B", "W"):
            mk = ms["colour"] == col
            bt = p3.sub_bootstrap(ms, mk)
            dd = bt.stats(sc[mk]) - bt.stats(res["R20"][mk])
            rec[f"diff_{col}"] = rec["by_colour"][col] - out["R20"]["by_colour"][col]
            rec[f"ci95_{col}"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
    out[name] = rec
    print(name, json.dumps(rec), flush=True)
json.dump(out, open("diag_mono.json", "w"), indent=1)
