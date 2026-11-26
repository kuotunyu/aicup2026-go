"""Step 7 (robustness of analysis (c)): does the gain of the new terms (style, joseki, n30) come from
telling LEVELS apart rather than PLAYERS? The official pool's level mix is unknown.

Pools of 90 candidates (fit players with >= 120 black games), one question each (o2o), k cycling
5/10/15/20: 'single' = all 90 from one level (levels with >= 90 such players), 'mixed' = 9 per level.
Same weights as fused_nopost.json (chosen on the 400-candidate tuning pools). Compare the gain over
the opening-only base in single-level vs mixed pools.
Continuation run: also the v2 fusion of s5b (fused_v2_nopost.json: base + lda64 + joseki_own + style2;
the LDA is fitted on every cached fit player outside the 90-player pool) and CSLS k=3 on it."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LEVELS_SINGLE = ["D", "C", "B", "1D", "2D", "3D", "4D", "5D"]
SEEDS = [0, 1, 2]


def make(c, kind, lv, seed):
    import c1h_lib as L
    elig = defaultdict(list)
    for p in sorted(c["n_all"]):
        if c["n_black"][p] >= 120:
            elig[c["level"][p]].append(p)
    key = lambda p: hashlib.sha256(("%d|%s|lv" % (seed, p)).encode()).hexdigest()
    if kind == "single":
        pool = sorted(elig[lv], key=key)[:90]
    else:
        pool = [p for l in L.LEVELS for p in sorted(elig[l], key=key)[:9]]
    ks = [L.K_VALUES[i % 4] for i in range(len(pool))]
    random.Random("k7|%s|%s|%d" % (kind, lv, seed)).shuffle(ks)
    rng = random.Random("pool7|%s|%s|%d" % (kind, lv, seed))
    cand, q = [], []
    for p, k in zip(pool, ks):
        g = list(c["by_player"][p])
        rng.shuffle(g)
        cand.append(g[:100])
        q.append(g[100:100 + k])
    return {"cand_rows": cand, "q_rows": q, "owner": np.arange(len(pool)), "k": np.asarray(ks)}


def one(args):
    kind, lv, seed = args
    sys.path.insert(0, HERE)
    import c1h_eval as E
    import c1h_fused as FU
    import c1h_lib as L
    c = L.load_cache()
    with open(os.path.join(HERE, "fused_nopost.json"), encoding="utf-8") as f:
        spec = json.load(f)
    with open(os.path.join(HERE, "fused_v2_nopost.json"), encoding="utf-8") as f:
        spec2 = json.load(f)
    rec = make(c, kind, lv, seed)
    rec["seed"], rec["scenario"] = seed, "s7|%s|%s" % (kind, lv)
    Gc = {}
    T = {t: FU.term(t, rec, c, Gc) for t in sorted(set(spec["terms"] + spec2["terms"]))}
    out = {"kind": kind, "level": lv, "seed": seed}

    def sc(S):
        return float(L.pts(L.ranks(S, rec["owner"])).mean())
    out["base"] = sc(T["base"])
    out["fused"] = sc(E.fuse([T[t] for t in spec["terms"]], spec["weights"]))
    nost = [(t, w) for t, w in zip(spec["terms"], spec["weights"]) if t != "style"]
    out["fused_no_style"] = sc(E.fuse([T[t] for t, _ in nost], [w for _, w in nost]))
    out["base_style"] = sc(E.fuse([T["base"], T["style"]], [1.0, 2.0]))
    out["base_josown"] = sc(E.fuse([T["base"], T["joseki_own"]], [1.0, 1.0]))
    out["style_alone"] = sc(T["style"])
    out["fused_csls5"] = sc(E.csls(E.fuse([T[t] for t in spec["terms"]], spec["weights"]), 5))
    out["base_csls5"] = sc(E.csls(T["base"], 5))
    F2 = E.fuse([T[t] for t in spec2["terms"]], spec2["weights"])
    out["fused_v2"] = sc(F2)
    out["fused_v2_csls3"] = sc(E.csls(F2, 3))
    out["lda_alone"] = sc(T["lda64"])
    out["base_lda"] = sc(E.fuse([T["base"], T["lda64"]], [1.0, 2.0]))
    out["proxyA"] = sc(E.fuse([T["base"], T["style"]], [1.0, 1.0]))
    rest = [(t, w) for t, w in zip(spec2["terms"], spec2["weights"]) if t != "base"]
    out["proxyA_plus_v2"] = sc(E.fuse([T["base"], T["style"]] + [T[t] for t, _ in rest], [1.0, 1.0] + [w for _, w in rest]))
    out["proxyA_plus_v2_csls3"] = sc(E.csls(E.fuse([T["base"], T["style"]] + [T[t] for t, _ in rest],
                                                   [1.0, 1.0] + [w for _, w in rest]), 3))
    return out


if __name__ == "__main__":
    import c1h_lib as L
    t0 = time.time()
    jobs = [("single", lv, s) for lv in LEVELS_SINGLE for s in SEEDS] + [("mixed", "all", s) for s in range(24)]
    with Pool(int(os.environ.get("C1H_WORKERS", "8"))) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    keys = ["base", "fused", "fused_no_style", "base_style", "base_josown", "style_alone", "fused_csls5", "base_csls5",
            "fused_v2", "fused_v2_csls3", "lda_alone", "base_lda", "proxyA", "proxyA_plus_v2", "proxyA_plus_v2_csls3"]
    tab = {}
    for kind in ("single", "mixed"):
        rr = [r for r in rows if r["kind"] == kind]
        tab[kind] = {k: [round(v, 4) for v in L.mean_ci([r[k] for r in rr])] for k in keys}
        tab[kind].update({"gain_" + k: [round(v, 4) for v in L.mean_ci([r[k] - r["base"] for r in rr])]
                          for k in keys if k != "base"})
    tab["single_by_level_gain_fused"] = {lv: round(float(np.mean([r["fused"] - r["base"] for r in rows
                                                                   if r["level"] == lv])), 4) for lv in LEVELS_SINGLE}
    tab["single_by_level_style_alone"] = {lv: round(float(np.mean([r["style_alone"] for r in rows
                                                                    if r["level"] == lv])), 4) for lv in LEVELS_SINGLE}
    for k in ("fused_v2", "fused_v2_csls3", "lda_alone", "base"):
        tab["single_by_level_" + k] = {lv: round(float(np.mean([r[k] for r in rows if r["level"] == lv])), 4)
                                       for lv in LEVELS_SINGLE}
    # paired by seed is not possible (different players); the comparison is single vs mixed means
    tab["gain_v2_single_minus_mixed"] = round(tab["single"]["gain_fused_v2"][0] - tab["mixed"]["gain_fused_v2"][0], 4)
    tab["gain_proxyA_plus_v2_over_proxyA"] = {
        kind: [round(v, 4) for v in L.mean_ci([r["proxyA_plus_v2"] - r["proxyA"] for r in rows if r["kind"] == kind])]
        for kind in ("single", "mixed")}
    out = {"what": __doc__, "table": tab, "rows": rows, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, "s7_level_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(tab, indent=1))
