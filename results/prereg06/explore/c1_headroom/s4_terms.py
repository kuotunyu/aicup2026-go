"""Step 4: compute every candidate term of analysis (c) on the o2o official-shaped pools (tuning seeds
100-109, evaluation seeds 0-19) and store them (float32) in the scratchpad for s5_combine.py."""
import sys
sys.dont_write_bytecode = True
import json
import os
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEEDS = list(range(100, 110)) + list(range(20))


def one(seed):
    sys.path.insert(0, HERE)
    import c1h_fused as FU
    import c1h_lib as L
    t = time.time()
    c = L.load_cache()
    rec = L.make_pool(c, seed, "o2o")
    G_cache = {}
    out, secs = {}, {}
    names = sys.argv[1:] if len(sys.argv) > 1 else FU.TERMS
    path = os.path.join(L.SCRATCH, "terms_o2o_%d.npz" % seed)
    old = {}
    if os.path.exists(path):
        with np.load(path) as z:
            old = {k: z[k] for k in z.files}
    for n in names:
        t1 = time.time()
        out[n] = FU.term(n, rec, c, G_cache).astype(np.float32)
        secs[n] = round(time.time() - t1, 1)
        for kk in [kk for kk in G_cache if kk != "base"]:
            del G_cache[kk]                      # keep memory per worker small
    old.update(out)
    old["owner"] = rec["owner"]
    old["k"] = rec["k"]
    np.savez(path, **old)
    return seed, secs, round(time.time() - t, 1)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(int(os.environ.get('C1H_WORKERS', '4'))) as pool:
        res = pool.map(one, SEEDS, chunksize=1)
    print(json.dumps(res[:3]), "total %.0fs" % (time.time() - t0))
    with open(os.path.join(HERE, "s4_terms_timing.json"), "w", encoding="utf-8") as f:
        json.dump({"per_seed": res, "total_seconds": round(time.time() - t0, 1), "terms": sys.argv[1:] or "all"}, f, indent=1)
