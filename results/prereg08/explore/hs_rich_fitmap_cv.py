"""prereg08 exploration (CPU only, fitmap only): is there more per-game level information in the PREREG_07 HumanSL
per-move outputs than the 14-d mean curve the pilot uses?

Reads ONLY data/prereg07/pilot/fitmap/chunk_*.npz (fit players, one game each, 300 per level; no inner, no mono, no
validation) and plan.json for the fitmap labels. Same 10-class L2 logistic regression, same lambda grid and the same
5-fold split (default_rng(0).permutation(3000) % 5) as pilot_fuse.fit_mapping. Feature variants:
  V0 mean      : x[j] = mean over target moves of log p(move | profile j)   (the PREREG_07 HS curve; reproduces CV)
  V1 phase     : V0 computed separately on turns <60, 60-149, >=150 (missing phase -> game mean) + 3 presence flags
  V2 moments   : V0 + per-profile SD + 10/50/90 % quantiles of the per-move log p
  V3 centred   : V0 + mean over moves of the per-move log p centred on its across-profile mean (curve shape only)
                 + mean per-move softmax over profiles (per-move posterior, uniform prior)
  V4 all       : V1 + V2 + V3 + log(target moves)
Output: hs_rich_fitmap_cv.json next to this file. No inner labels, no fusion, no score.py.
"""
import glob
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PH = os.path.join(REPO, "results", "prereg07", "explore", "pilot_humansl")
sys.path.insert(0, PH)
sys.path.insert(0, os.path.join(REPO, "results", "prereg05", "explore", "u1"))
import numpy as np  # noqa: E402

import hs_lib as H  # noqa: E402  (data/local guard)
import pilot_fuse as PF  # noqa: E402

PILOT = os.path.join(REPO, "data", "prereg07", "pilot")


def main():
    t0 = time.time()
    import u1_lib as U
    plan = json.load(open(os.path.join(PILOT, "plan.json"), encoding="utf-8"))
    files = sorted(f for f in glob.glob(os.path.join(PILOT, "fitmap", "chunk_*.npz")) if ".tmp" not in f)
    G = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            ref, failed = z["ref"], z["failed"]
            pg, pt, ps, pl = z["pos_game"], z["pos_turn"], z["pos_side"], z["pos_logp"]
            for gi in range(len(ref)):
                m = (pg == gi) & (ps == 0)
                G.append((int(ref[gi]), bool(failed[gi]), pt[m].astype(np.int64), pl[m].astype(np.float64)))
    lv = {l: i for i, l in enumerate(U.LEVELS)}
    feats = {k: [] for k in ("V0", "V1", "V2", "V3", "V4")}
    y = []
    for ref, failed, turn, lp in G:
        if failed or len(turn) == 0:
            continue
        y.append(lv[plan["sets"]["fitmap"]["games"][ref]["rank"]])
        v0 = lp.mean(0)
        ph = []
        flags = []
        for lo, hi in ((0, 60), (60, 150), (150, 10 ** 6)):
            m = (turn >= lo) & (turn < hi)
            ph.append(lp[m].mean(0) if m.any() else v0)
            flags.append(float(m.any()))
        v1 = np.concatenate(ph + [np.array(flags)])
        v2 = np.concatenate([v0, lp.std(0), *np.quantile(lp, [0.1, 0.5, 0.9], axis=0)])
        c = lp - lp.mean(1, keepdims=True)
        post = np.exp(lp - lp.max(1, keepdims=True))
        post /= post.sum(1, keepdims=True)
        v3 = np.concatenate([v0, c.mean(0), post.mean(0)])
        v4 = np.concatenate([v1, v2[14:], v3[14:], [np.log(len(turn))]])
        for k, v in zip(("V0", "V1", "V2", "V3", "V4"), (v0, v1, v2, v3, v4)):
            feats[k].append(v)
    y = np.array(y)
    out = {"what": "prereg08 explore: richer HS per-game features, fitmap-only 5-fold CV (same folds/lambdas as "
                   "pilot_fuse.fit_mapping)", "created_utc": H.utc_now(), "games": int(len(y)),
           "chunks": len(files), "variants": {}}
    for k, X in feats.items():
        X = np.array(X)
        r = PF.fit_mapping(X, y)
        best = str(r["lambda"])
        out["variants"][k] = {"dims": int(X.shape[1]), "lambda": r["lambda"], "cv_best": r["cv"][best],
                              "cv_all": r["cv"]}
        print(k, X.shape[1], r["lambda"], r["cv"][best], flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(HERE, "hs_rich_fitmap_cv.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
