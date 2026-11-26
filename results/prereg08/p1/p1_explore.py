"""prereg08 P1 follow-up exploration (CPU, inner only, after the P1 gate passed; record only, not a gate).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg08/p1/p1_explore.py
        -> results/prereg08/p1/p1_explore_report.json, data/prereg08/p1/lp_P1TO.npz

Questions (written after seeing p1_fuse_report.json; the fused weight of P1TO sat at its cap 4.0 in every fold):
  E1 scale: lp_P1TO multiplied by s in {1, 2, 4, 8} before the fusion (same as raising the weight cap to 4 s;
     the frozen p04 fusion code is not touched). Combos Q02H+P1TO and R20+P1TO.
  E2 learning curve: TO mapping CV (player-grouped, the plan's lambda grid) on 10k / 20k / 30k probe games,
     to see whether a bigger probe would still help.
Same inputs, pins and guards as p1_fuse.py; no validation file, no score.py.
"""

import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p1_fuse as F  # noqa: E402

P = F.P
OUT = os.path.join(HERE, "p1_explore_report.json")
LP_OUT = os.path.join(F.REPO, "data", "prereg08", "p1", "lp_P1TO.npz")
SCALES = [1.0, 2.0, 4.0, 8.0]


def main():
    t0 = time.time()
    with open(F.PLAN, encoding="utf-8") as f:
        plan = json.load(f)
    sets_path = os.path.join(F.P1_DIR, "sets.json")
    if P.sha_raw(sets_path) != plan["sets"]["sets_json_sha256"]:
        raise P.P1Error("sets.json SHA differs from the plan")
    with open(sets_path, encoding="utf-8") as f:
        sets = json.load(f)
    lambdas = plan["mapping"]["lambdas"]
    PF, U = F.import_pinned()
    st = {k: F.load_p1_set(k, F.P1_DIR) for k in ("probe", "bank", "mono")}
    d = U.load_inputs(verbose=False)
    inner = d["inner"]
    n_inner = inner["keys"]["n"]
    lp_hs, hs_cov, _ = F.hs_baseline(PF, U, d)
    d["lp"]["HS"] = lp_hs
    pr = st["probe"]
    lv_of = {lv: i for i, lv in enumerate(U.LEVELS)}
    games = sets["probe"]["games"]
    y_all = np.array([lv_of[games[j]["rank"]] for j in pr["ref"]], dtype=np.int64)
    pl_all = np.array([games[j]["rank"] + ":" + games[j]["player_id"] for j in pr["ref"]])
    X, usable = F.variant_X(pr, "TO")
    rep = {"format": "c10-prereg08 p1 explore v1", "created_utc": P.H.utc_now(),
           "note": "written after the P1 gate (record only); inner only, no validation", "code_sha256": P.code_shas()}
    # ---- E2 learning curve (player-grouped subsets: whole players, seeded)
    players = np.unique(pl_all[usable])
    rng = np.random.default_rng(1)
    order = rng.permutation(players)
    lc = {}
    for frac in (1 / 3, 2 / 3):
        keep = set(order[:int(round(frac * len(order)))])
        m = usable & np.array([p in keep for p in pl_all])
        mp = F.fit_mapping_grouped(PF, U, X[m], y_all[m], pl_all[m], lambdas, maxiter=2000, log=lambda *a, **k: None)
        lc[f"{int(m.sum())}"] = {"lambda": mp["lambda"], "cv": mp["cv"][str(mp["lambda"])]}
        print("E2", int(m.sum()), lc[f"{int(m.sum())}"], flush=True)
    mp = F.fit_mapping_grouped(PF, U, X[usable], y_all[usable], pl_all[usable], lambdas, maxiter=2000)
    lc[f"{int(usable.sum())}"] = {"lambda": mp["lambda"], "cv": mp["cv"][str(mp["lambda"])]}
    rep["E2_learning_curve"] = lc
    # ---- inner lp
    lp = np.zeros((n_inner, F.L))
    for name in ("bank", "mono"):
        s = st[name]
        Xs, us = F.variant_X(s, "TO")
        lp[s["ref"][us]] = mp["logp"](Xs[us])
    os.makedirs(os.path.dirname(LP_OUT), exist_ok=True)
    np.savez_compressed(LP_OUT, lp=lp, lp_hs=lp_hs)
    # ---- E1 scale
    Xe, Xm = inner["X_eval"], inner["X_mono"]
    base = {}
    for cid, terms in (("Q02H", F.Q02H_TERMS), ("R20", ["OP"])):
        Pe, Pm, th = U.run_gen(d, terms)
        base[cid] = (U.score(U.rule_b_np(Pe), Xe.y), U.score(U.rule_b_np(Pm), Xm.y))
    rb_h = U.rank_balanced(base["Q02H"][0], Xe.y)
    if abs(rb_h - F.PILOT_Q02H) > F.TOL:
        raise P.P1Error(f"Q02H seam {rb_h}")
    boot = inner["boot"]
    ms = inner["ms"]
    sboot = U.PlayerBootstrap(ms["player"], ms["y"], ms["n_players"])
    rows = {}
    for s in SCALES:
        d["lp"]["P1S"] = lp * s
        for cid, terms in (("Q02H+P1TO", F.Q02H_TERMS + ["P1S"]), ("R20+P1TO", ["OP", "P1S"])):
            Pe, Pm, th = U.run_gen(d, terms)
            sc, msc = U.score(U.rule_b_np(Pe), Xe.y), U.score(U.rule_b_np(Pm), Xm.y)
            dd = boot.stats(sc) - boot.stats(base["Q02H"][0])
            mdd = sboot.stats(msc) - sboot.stats(base["Q02H"][1])
            r = {"rank_balanced": U.rank_balanced(sc, Xe.y), "diff_vs_Q02H": U.rank_balanced(sc, Xe.y) - rb_h,
                 "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                 "mono_rank_balanced": U.rank_balanced(msc, Xm.y),
                 "mono_diff_vs_Q02H": U.rank_balanced(msc, Xm.y) - U.rank_balanced(base["Q02H"][1], Xm.y),
                 "mono_ci95": [float(np.percentile(mdd, 2.5)), float(np.percentile(mdd, 97.5))],
                 "weights_by_fold": [list(map(float, t[2:2 + len(terms)])) for t in th]}
            rows[f"{cid} x{s:g}"] = r
            print(f"E1 {cid} x{s:g}: {r['rank_balanced']:.4f} ({r['diff_vs_Q02H']:+.4f}) mono "
                  f"{r['mono_diff_vs_Q02H']:+.4f} w {[round(w[-1], 2) for w in r['weights_by_fold']]}", flush=True)
    rep["E1_scale"] = rows
    rep["Q02H_rank_balanced"] = rb_h
    rep["seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=float)
    print("->", OUT)


if __name__ == "__main__":
    main()
