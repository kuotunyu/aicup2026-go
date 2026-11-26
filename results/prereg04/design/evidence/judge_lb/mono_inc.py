"""Judge probe: single-colour pseudo-questions, increment of SW on top of O (Q02 - Q01), by colour and k. Inner only."""
import json, os, sys, time, types
sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
P04 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(P04, "B_proxy"))
import numpy as np
import p02_rank_fuse as rf, p03_rank_fuse as p3, fuse as F
t0 = time.time()
c = types.SimpleNamespace(p02_dumps=os.path.join(p3.P02_RANK, "dumps"), p02_local=p3.P02_RANK, split=p3.SPLIT_CSV, smoke=False)
inner = p3.prepare_inner(c)
keys, Xf, Sf, Xe, Se = inner["keys"], inner["X_fit"], inner["S_fit"], inner["X_eval"], inner["S_eval"]
n = keys["n"]; ms = p3.mono_set(inner); Xm, Sm = p3.mono_qset(inner, ms)
lp00 = p3.combo_logprobs(("C5",), {p3.C5_DUMP: p3.load_c5_inner(c, n)["logits"]})
z = np.load(os.path.join(P04, "B_proxy", "proxy_inner_all.npz"))
row = {(r, g): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
idx = np.array([row[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
V = [str(v) for v in z["variants"]]; ll = z["ll"][idx]
src = {"O": rf.log_softmax(ll[:, V.index("o_choice_a20")].sum(1))}
for arm in ("a", "b"):
    ddir = os.path.join(p3.P03_ROOT, f"rank_{arm}", "dumps")
    src[f"SW{arm.upper()}"] = rf.log_softmax(p3.load_swap(os.path.join(ddir, f"inner_p03{arm}_s0_swap.npz"), n)["logits"].astype(np.float64))
pool = lambda S, lp: rf.pooled(S, lp, "log")
mf, mm, me = pool(Sf, lp00), pool(Sm, lp00), pool(Se, lp00)
Ef = {k: pool(Sf, v) for k, v in src.items()}; Em = {k: pool(Sm, v) for k, v in src.items()}; Ee = {k: pool(Se, v) for k, v in src.items()}
bm = rf.PlayerBootstrap(ms["player"], ms["y"], ms["n_players"])
res, scm, sce = {}, {}, {}
for name, terms in {"R20": [], "Q01": ["O"], "Q02": ["O", "SWA", "SWB"]}.items():
    Pm = np.zeros((len(Xm), 10)); Pe = np.zeros((len(Xe), 10))
    for f in range(5):
        tr = Xf.fold != f
        th = F.gen_fit(mf[tr], [Xf.mop[tr]] + [Ef[t][tr] for t in terms], Xf.n[tr], Xf.y[tr])
        ev = Xm.fold == f; zz, _ = F.gen_logits(th, mm[ev], [Xm.mop[ev]] + [Em[t][ev] for t in terms], Xm.n[ev]); Pm[ev] = rf.softmax(zz)
        ev = Xe.fold == f; zz, _ = F.gen_logits(th, me[ev], [Xe.mop[ev]] + [Ee[t][ev] for t in terms], Xe.n[ev]); Pe[ev] = rf.softmax(zz)
    m, scm[name] = rf.metrics(rf.rule_b_np(Pm), Xm.y, Xm.n, Xm.player)
    m2, sce[name] = rf.metrics(rf.rule_b_np(Pe), Xe.y, Xe.n, Xe.player)
    res[name] = {"mono_rb": m["rank_balanced"], "mixed_rb": m2["rank_balanced"],
                 "by_k": {str(k): p3.score_summary(scm[name], Xm.y, ms["k"] == k)["rank_balanced"] for k in (5, 10, 15, 20)}}
d = bm.stats(scm["Q02"]) - bm.stats(scm["Q01"])
inc = {"mono_Q02_minus_Q01": res["Q02"]["mono_rb"] - res["Q01"]["mono_rb"], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
       "by_k": {k: res["Q02"]["by_k"][k] - res["Q01"]["by_k"][k] for k in res["Q01"]["by_k"]}}
for col in ("B", "W"):
    mk = ms["colour"] == col; bt = p3.sub_bootstrap(ms, mk)
    dd = bt.stats(scm["Q02"][mk]) - bt.stats(scm["Q01"][mk])
    inc[f"{col}"] = [float(np.mean(dd)), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
inc["se_val_mixed_Q02_vs_Q01"] = p3.se_val_record(res["Q02"]["mixed_rb"] - res["Q01"]["mixed_rb"], sce["Q02"] - sce["Q01"], Xe.y)["se_val"]
for nm in ("Q01", "Q02"):
    r = p3.se_val_record(res[nm]["mixed_rb"] - res["R20"]["mixed_rb"], sce[nm] - sce["R20"], Xe.y)
    inc[f"se_val_{nm}_vs_R20"] = r["se_val"]; inc[f"adopt_{nm}_at_inner"] = r["adoption_probability"]
out = {"combos": res, "increment": inc, "seconds": time.time() - t0}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mono_inc.json"), "w"), indent=1)
