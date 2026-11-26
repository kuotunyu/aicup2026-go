"""Step 1: per-game token count matrices (one per family, one global vocabulary each) and the dense
style matrix over all cached black games. Saved in the scratchpad. At most 8 worker processes."""
import sys
sys.dont_write_bytecode = True
import json
import os
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def work(name):
    sys.path.insert(0, HERE)
    import numpy as np
    import scipy.sparse as sp
    import c1h_lib as L
    import c1h_feats as F
    t = time.time()
    c = L.load_cache()
    if name in ("style", "style2"):
        fn = F.style_vec if name == "style" else F.style2_vec
        X = np.vstack([fn(p) for p in c["gpts"]])
        np.save(os.path.join(L.SCRATCH, "dense_%s.npy" % name), X)
        return name, X.shape, round(time.time() - t, 1)
    M, vocab = L.token_matrix(F.FAMILIES[name], cache=c)
    sp.save_npz(L.family_path(name), M)
    return name, M.shape, round(time.time() - t, 1)


if __name__ == "__main__":
    names = sys.argv[1:] or ["base", "n20", "n30", "deep", "prefix", "opp", "joseki", "joseki_own", "resp", "style"]
    t0 = time.time()
    with Pool(min(8, len(names))) as pool:
        res = pool.map(work, names, chunksize=1)
    out = {n: {"shape": list(s), "seconds": sec} for n, s, sec in res}
    print(json.dumps(out, indent=1), "total %.0fs" % (time.time() - t0))
    path = os.path.join(HERE, "s1_families.json")
    old = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    old.update(out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=1)
