"""Step 5 = analysis (c): which extra signals raise the opening-only score on the o2o pools.

Every term alone; base + w * term (z-fusion as p02's fuse, w chosen on the tuning pools 100-109);
hubness corrections (CSLS, column centring) on base; greedy forward selection of terms with weights
on the tuning pools; everything reported on the evaluation pools 0-19 as the paired difference to the
base (mean and t-interval over the 20 pools). Writes fused_best.json (spec for s3_assign.py)."""
import sys
sys.dont_write_bytecode = True
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c1h_eval as E  # noqa: E402
import c1h_lib as L  # noqa: E402

TUNE = list(range(100, 110))
EVAL = list(range(20))
WGRID = [0.1, 0.25, 0.5, 1.0, 2.0]


def load(seed):
    with np.load(os.path.join(L.SCRATCH, "terms_o2o_%d.npz" % seed)) as z:
        d = {k: z[k].astype(np.float64) if z[k].dtype == np.float32 else z[k] for k in z.files}
    d["Z"] = {k: L.zrows(v) for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim == 2}
    return d


def sc(S, d):
    r = L.ranks(S, d["owner"])
    return float(L.pts(r).mean()), r


def fused(d, terms, ws):
    s = 0
    for t, w in zip(terms, ws):
        s = s + w * d["Z"][t]
    return s


def per_pool(D, terms, ws, post=None):
    out = []
    for d in D:
        S = fused(d, terms, ws)
        if post and post.startswith("csls"):
            S = E.csls(S, int(post[4:]))
        elif post == "colc":
            S = E.col_center(S)
        out.append(sc(S, d)[0])
    return np.asarray(out)


def per_k(D, terms, ws, base_terms=("base",), base_ws=(1.0,)):
    res = {}
    for k in L.K_VALUES:
        diffs = []
        for d in D:
            m = d["k"] == k
            r1 = L.ranks(fused(d, terms, ws), d["owner"])
            r0 = L.ranks(fused(d, list(base_terms), list(base_ws)), d["owner"])
            diffs.append(L.pts(r1[m]).mean() - L.pts(r0[m]).mean())
        res[str(k)] = [round(v, 4) for v in L.mean_ci(diffs)]
    return res


if __name__ == "__main__":
    t0 = time.time()
    DT = [load(s) for s in TUNE]
    DE = [load(s) for s in EVAL]
    names = [k for k in DE[0]["Z"]]
    base_t = per_pool(DT, ["base"], [1.0])
    base_e = per_pool(DE, ["base"], [1.0])
    out = {"base_eval": [round(v, 4) for v in L.mean_ci(base_e)], "base_tune_mean": round(float(base_t.mean()), 4),
           "alone": {}, "with_base": {}, "hubness": {}}
    for n in names:
        e = per_pool(DE, [n], [1.0])
        out["alone"][n] = {"eval": [round(v, 4) for v in L.mean_ci(e)],
                           "diff_vs_base": [round(v, 4) for v in L.mean_ci(e - base_e)]}
    for n in names:
        if n == "base":
            continue
        tg = {w: float((per_pool(DT, ["base", n], [1.0, w]) - base_t).mean()) for w in WGRID}
        w = max(WGRID, key=lambda x: tg[x])
        e = per_pool(DE, ["base", n], [1.0, w])
        out["with_base"][n] = {"tune_gain_by_w": {str(k): round(v, 4) for k, v in tg.items()}, "w": w,
                               "eval_diff_vs_base": [round(v, 4) for v in L.mean_ci(e - base_e)],
                               "eval_score": [round(v, 4) for v in L.mean_ci(e)]}
        print(n, out["with_base"][n]["w"], out["with_base"][n]["eval_diff_vs_base"], flush=True)
    for post in ("csls5", "csls10", "csls20", "colc"):
        tg = float((per_pool(DT, ["base"], [1.0], post) - base_t).mean())
        e = per_pool(DE, ["base"], [1.0], post)
        out["hubness"][post] = {"tune_gain": round(tg, 4), "eval_diff_vs_base": [round(v, 4) for v in L.mean_ci(e - base_e)]}
    # greedy forward selection on the tuning pools
    cur_t, cur_w = ["base"], [1.0]
    cur_score = float(base_t.mean())
    path = []
    pool_names = [n for n in names if n != "base"]
    for step in range(6):
        best = None
        for n in pool_names:
            if n in cur_t:
                continue
            for w in WGRID:
                s = float(per_pool(DT, cur_t + [n], cur_w + [w]).mean())
                if best is None or s > best[0]:
                    best = (s, n, w)
        if best is None or best[0] - cur_score < 0.001:
            break
        cur_t.append(best[1])
        cur_w.append(best[2])
        cur_score = best[0]
        path.append({"add": best[1], "w": best[2], "tune_mean": round(cur_score, 4)})
        print("greedy add", path[-1], flush=True)
    # coordinate refinement of the weights on the tuning pools
    for _ in range(2):
        for i in range(1, len(cur_t)):
            cand = sorted(set([cur_w[i] * f for f in (0.5, 0.75, 1.0, 1.33, 2.0)]))
            best = max(cand, key=lambda w: float(per_pool(DT, cur_t, cur_w[:i] + [w] + cur_w[i + 1:]).mean()))
            cur_w[i] = best
    e = per_pool(DE, cur_t, cur_w)
    t_ = per_pool(DT, cur_t, cur_w)
    posts = {}
    for post in (None, "csls10", "colc"):
        tg = float((per_pool(DT, cur_t, cur_w, post) - base_t).mean())
        ee = per_pool(DE, cur_t, cur_w, post)
        posts[str(post)] = {"tune_gain": round(tg, 4), "eval_diff_vs_base": [round(v, 4) for v in L.mean_ci(ee - base_e)],
                            "eval_score": [round(v, 4) for v in L.mean_ci(ee)]}
    best_post = max(posts, key=lambda p: posts[p]["tune_gain"])
    out["greedy"] = {"path": path, "terms": cur_t, "weights": cur_w, "tune_mean": round(float(t_.mean()), 4),
                     "eval_score": [round(v, 4) for v in L.mean_ci(e)],
                     "eval_diff_vs_base": [round(v, 4) for v in L.mean_ci(e - base_e)],
                     "eval_diff_by_k": per_k(DE, cur_t, cur_w),
                     "post": posts, "post_selected_on_tune": best_post}
    # eval top1/top5 of the final fused (no post)
    rr = [L.ranks(fused(d, cur_t, cur_w), d["owner"]) for d in DE]
    r0 = [L.ranks(d["Z"]["base"], d["owner"]) for d in DE]
    out["greedy"]["eval_top1"] = [round(v, 4) for v in L.mean_ci([(r == 1).mean() for r in rr])]
    out["greedy"]["eval_top5"] = [round(v, 4) for v in L.mean_ci([(r <= 5).mean() for r in rr])]
    out["base_eval_top1"] = [round(v, 4) for v in L.mean_ci([(r == 1).mean() for r in r0])]
    out["base_eval_top5"] = [round(v, 4) for v in L.mean_ci([(r <= 5).mean() for r in r0])]
    spec = {"terms": cur_t, "weights": cur_w, "post": None if best_post == "None" else best_post}
    with open(os.path.join(HERE, "fused_best.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "s5_combine.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("base_eval", "alone", "hubness", "greedy")}, indent=1))
