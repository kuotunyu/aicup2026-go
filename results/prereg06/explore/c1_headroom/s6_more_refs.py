"""Step 6 (side analysis): how much do more reference games per candidate help?

Relevant only if the official candidates turn out to be training players whose other games we could
legitimately add (T2 / T3 on 11/04 decide; not assumed here). Pools: fit players with >= 220 black
games, up to 40 per level (fewer where a level has fewer), questions k 5/10/15/20 in equal shares.
Same players and the same question games (shuffled black games 200 .. 200+k); candidate profiles
built from the first 100 vs the first 200 shuffled black games. Opening-only base term."""
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
SEEDS = list(range(10))


def one(seed):
    sys.path.insert(0, HERE)
    import c1h_eval as E
    import c1h_lib as L
    c = L.load_cache()
    G = L.load_family("base")
    elig = defaultdict(list)
    for p in sorted(c["n_all"]):
        if c["n_black"][p] >= 220:
            elig[c["level"][p]].append(p)
    pool = []
    for lv in L.LEVELS:
        pl = sorted(elig[lv], key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        pool += pl[:40]
    n = len(pool)
    ks = [L.K_VALUES[i % 4] for i in range(n)]
    random.Random("k6|%d" % seed).shuffle(ks)
    rng = random.Random("pool6|%d" % seed)
    shuf = {}
    for p in pool:
        g = list(c["by_player"][p])
        rng.shuffle(g)
        shuf[p] = g
    q_rows = [shuf[p][200:200 + k] for p, k in zip(pool, ks)]
    out = {"seed": seed, "n": n}
    for nc in (50, 100, 150, 200):
        rec = {"cand_rows": [shuf[p][:nc] for p in pool], "q_rows": q_rows}
        S = E.cos_term(G, rec)
        r = L.ranks(S, np.arange(n))
        out[str(nc)] = {"score": float(L.pts(r).mean()), "top1": float((r == 1).mean())}
    return out


if __name__ == "__main__":
    import c1h_lib as L
    t0 = time.time()
    with Pool(int(os.environ.get('C1H_WORKERS', '8'))) as pool:
        res = pool.map(one, SEEDS, chunksize=1)
    tab = {nc: {"score": [round(v, 4) for v in L.mean_ci([r[nc]["score"] for r in res])],
                "top1": [round(v, 4) for v in L.mean_ci([r[nc]["top1"] for r in res])],
                "diff_vs_100": [round(v, 4) for v in L.mean_ci([r[nc]["score"] - r["100"]["score"] for r in res])]}
           for nc in ("50", "100", "150", "200")}
    out = {"what": __doc__, "pool_sizes": [r["n"] for r in res], "table": tab, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, "s6_more_refs.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
