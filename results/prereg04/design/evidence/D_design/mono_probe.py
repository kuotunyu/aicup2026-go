"""p04 design probe 2 (inner only, record only): single-colour pseudo-questions (official structure) for the no-training
candidates, recorded SE_val / adoption probability (p03.se_val_record, inner per-question differences scaled to the
validation level counts; no validation file read), and the side-kappa reference kappa(target-side) - kappa(opponent-side)
for the proxy pair and the prereg03 A arm (correct vs colour-swapped). Read-only on the repo."""
import json
import math
import os
import sys
import time
import types

sys.dont_write_bytecode = True
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
BPROXY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "B_proxy")
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, BPROXY)
import numpy as np  # noqa: E402

import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
import fuse as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
c = types.SimpleNamespace(p02_dumps=os.path.join(p3.P02_RANK, "dumps"), p02_local=p3.P02_RANK, split=p3.SPLIT_CSV,
                          smoke=False)
inner = p3.prepare_inner(c)
keys, Xf, Sf, Xe, Se = inner["keys"], inner["X_fit"], inner["S_fit"], inner["X_eval"], inner["S_eval"]
n = keys["n"]
ms = p3.mono_set(inner)
Xm, Sm = p3.mono_qset(inner, ms)
print("mono questions", len(Xm), ms["sha256"][:12], flush=True)
lp00 = p3.combo_logprobs(("C5",), {p3.C5_DUMP: p3.load_c5_inner(c, n)["logits"]})
z = np.load(os.path.join(BPROXY, "proxy_inner_all.npz"))
assert z["scored"].all()
row = {(r, g): i for i, (r, g) in enumerate(zip(z["rank"].tolist(), z["game_id"].tolist()))}
idx = np.array([row[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
assert all(z["player_id"][idx] == np.array(keys["player_id"]))
V = [str(v) for v in z["variants"]]
ll = z["ll"][idx]
src = {"T_px": rf.log_softmax(ll[:, V.index("t_choice_a20")].sum(1)),
       "O_px": rf.log_softmax(ll[:, V.index("o_choice_a20")].sum(1))}
for arm in ("a", "b"):
    ddir = os.path.join(p3.P03_ROOT, f"rank_{arm}", "dumps")
    d = rf.load_dump(os.path.join(ddir, f"inner_p03{arm}_d4.npz"), n, 8)
    src[f"{arm.upper()}0"] = rf.log_softmax(d["logits"][:, 0, :].astype(np.float64))
    src[f"SW{arm.upper()}"] = rf.log_softmax(p3.load_swap(os.path.join(ddir, f"inner_p03{arm}_s0_swap.npz"), n)["logits"]
                                             .astype(np.float64))

pool = lambda S, lp: rf.pooled(S, lp, "log")  # noqa: E731
mf, mm, me = pool(Sf, lp00), pool(Sm, lp00), pool(Se, lp00)
Ef = {k: pool(Sf, v) for k, v in src.items()}
Em = {k: pool(Sm, v) for k, v in src.items()}
Ee = {k: pool(Se, v) for k, v in src.items()}
bm = rf.PlayerBootstrap(ms["player"], ms["y"], ms["n_players"])

combos = {"R20": [], "Q01 R20+O_px": ["O_px"], "R20+SWA+SWB": ["SWA", "SWB"], "Q02 R20+O_px+SWA+SWB": ["O_px", "SWA", "SWB"]}
out = {"mono_questions": len(Xm), "mono_sha256": ms["sha256"], "mono": {}, "mixed_se_val": {}, "side_kappa": {}}
sc_m, sc_e = {}, {}
for name, terms in combos.items():
    Pm = np.zeros((len(Xm), 10))
    Pe = np.zeros((len(Xe), 10))
    for f in range(5):
        tr = Xf.fold != f
        th = F.gen_fit(mf[tr], [Xf.mop[tr]] + [Ef[t][tr] for t in terms], Xf.n[tr], Xf.y[tr])
        ev = Xm.fold == f
        zz, _ = F.gen_logits(th, mm[ev], [Xm.mop[ev]] + [Em[t][ev] for t in terms], Xm.n[ev])
        Pm[ev] = rf.softmax(zz)
        ev = Xe.fold == f
        zz, _ = F.gen_logits(th, me[ev], [Xe.mop[ev]] + [Ee[t][ev] for t in terms], Xe.n[ev])
        Pe[ev] = rf.softmax(zz)
    m, sc = rf.metrics(rf.rule_b_np(Pm), Xm.y, Xm.n, Xm.player)
    me_, sce = rf.metrics(rf.rule_b_np(Pe), Xe.y, Xe.n, Xe.player)
    sc_m[name], sc_e[name] = sc, sce
    rec = {"rank_balanced": m["rank_balanced"], "question_mean": m["question_mean"],
           "by_colour": {col: p3.score_summary(sc, Xm.y, ms["colour"] == col)["rank_balanced"] for col in ("B", "W")},
           "by_k": {str(k): p3.score_summary(sc, Xm.y, ms["k"] == k)["rank_balanced"] for k in (5, 10, 15, 20)},
           "mixed_rank_balanced": me_["rank_balanced"]}
    if name != "R20":
        d = bm.stats(sc) - bm.stats(sc_m["R20"])
        rec["diff_vs_R20"] = m["rank_balanced"] - out["mono"]["R20"]["rank_balanced"]
        rec["ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        rec["question_mean_diff"] = float(np.mean(sc - sc_m["R20"]))
        for col in ("B", "W"):
            mk = ms["colour"] == col
            bt = p3.sub_bootstrap(ms, mk)
            dd = bt.stats(sc[mk]) - bt.stats(sc_m["R20"][mk])
            rec[f"diff_{col}"] = rec["by_colour"][col] - out["mono"]["R20"]["by_colour"][col]
            rec[f"ci95_{col}"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        delta = me_["rank_balanced"] - out["mono"]["R20"]["mixed_rank_balanced"]
        se = p3.se_val_record(delta, sce - sc_e["R20"], Xe.y)
        out["mixed_se_val"][name] = {"delta_inner": delta, "se_val": se["se_val"],
                                     "adoption_probability_at_delta": se["adoption_probability"],
                                     "adoption_probability_at_half_delta":
                                         0.5 * (1 + math.erf((0.5 * delta / se["se_val"] - 1.96) / np.sqrt(2))),
                                     "adoption_probability_at_0.75_delta":
                                         0.5 * (1 + math.erf((0.75 * delta / se["se_val"] - 1.96) / np.sqrt(2)))}
    out["mono"][name] = rec
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                            if k not in ("by_colour", "by_k")}),
          json.dumps(out["mixed_se_val"].get(name, {})), f"({time.time() - t0:.0f}s)", flush=True)

for a, b in (("T_px", "O_px"), ("A0", "SWA"), ("B0", "SWB")):
    ka, kb = p3.kappa_questions(src[a], ms["groups"]), p3.kappa_questions(src[b], ms["groups"])
    dk = ka - kb
    rec = {"kappa_target_side": float(np.mean(ka)), "kappa_opponent_side": float(np.mean(kb)),
           "dkappa": float(np.mean(dk)), "ci95": p3.ci95(p3.boot_plain_mean(bm, dk))}
    for col in ("B", "W"):
        mk = ms["colour"] == col
        rec[f"dkappa_{col}"] = float(np.mean(dk[mk]))
    for k in (5, 10, 15, 20):
        rec[f"dkappa_k{k}"] = float(np.mean(dk[ms["k"] == k]))
    out["side_kappa"][f"{a} - {b}"] = rec
    print("side kappa", a, "-", b, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}),
          flush=True)
out["seconds"] = time.time() - t0
json.dump(out, open(os.path.join(HERE, "mono_probe.json"), "w", encoding="utf-8"), indent=1)
