"""Step 5b (continuation run) = analysis (c) v2 and the proxy part of analysis (d).

Same o2o official-shaped pools and the same method as s5_combine.py (all choices on the tuning pools
100-109, every reported number on the evaluation pools 0-19, paired differences with t-intervals over
the 20 pools), with two new terms computed by s4_terms.py in the continuation run:
  style2    finer dense style features (c1h_feats.style2_vec), Gaussian NB as 'style'
  lda<d>    c1h_lda: player-discriminant projection (d dims) learned on fit players outside the pool
s5_combine.json / fused_best.json / fused_nopost.json of the first run are left untouched; this
writes s5b_combine_v2.json, fused_v2_best.json, fused_v2_nopost.json.

(d) proxy: the CNN part of Q04 cannot be computed for fit players on this machine (the prereg01
player weights are only on Drive and the CNN has no CPU path here). Q04 = z(CNN) + 1.0 z(open). Two
stand-ins for z(CNN), from the same pools:
  proxyA = style  (Gaussian NB on 34 style values; alone ~0.38 here; the CNN alone scored 0.4006 in
                   the prereg02 candidate-pool self-retrieval, results/prereg02/player/sr_report.json)
  proxyB = lda32  (a much stronger style embedding; pessimistic stand-in: if the CNN already carried
                   all of it, the new terms would add only what is left)
For each, Q04proxy = z(base) + 1.0 z(stand-in); the greedy forward selection is re-run on top of it
(stand-in kept, its weight fixed at 1) and the gain over Q04proxy is reported."""
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
from s5_combine import TUNE, EVAL, load, fused, per_pool  # noqa: E402

WGRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
POSTS = (None, "csls3", "csls5", "csls10", "csls20")


def ci(x):
    return [round(v, 4) for v in L.mean_ci(x)]


def by_k(D, terms, ws, post, ref_terms, ref_ws, ref_post):
    out = {}
    for k in L.K_VALUES:
        d1, s1, t1, t5 = [], [], [], []
        for d in D:
            m = d["k"] == k
            r1 = L.ranks(postf(fused(d, terms, ws), post), d["owner"])
            r0 = L.ranks(postf(fused(d, ref_terms, ref_ws), ref_post), d["owner"])
            d1.append(L.pts(r1[m]).mean() - L.pts(r0[m]).mean())
            s1.append(L.pts(r1[m]).mean())
            t1.append((r1[m] == 1).mean())
            t5.append((r1[m] <= 5).mean())
        out[str(k)] = {"score": ci(s1), "top1": ci(t1), "top5": ci(t5), "diff_vs_ref": ci(d1)}
    return out


def postf(S, post):
    if post and post.startswith("csls"):
        return E.csls(S, int(post[4:]))
    return S


def pp(D, terms, ws, post=None):
    return per_pool(D, terms, ws, post)


def greedy(DT, start_t, start_w, pool_names, fixed=1, steps=8):
    cur_t, cur_w = list(start_t), list(start_w)
    cur = float(pp(DT, cur_t, cur_w).mean())
    path = []
    for _ in range(steps):
        best = None
        for n in pool_names:
            if n in cur_t:
                continue
            for w in WGRID:
                s = float(pp(DT, cur_t + [n], cur_w + [w]).mean())
                if best is None or s > best[0]:
                    best = (s, n, w)
        if best is None or best[0] - cur < 0.001:
            break
        cur_t.append(best[1])
        cur_w.append(best[2])
        cur = best[0]
        path.append({"add": best[1], "w": best[2], "tune_mean": round(cur, 4)})
        print("  greedy add", path[-1], flush=True)
    for _ in range(2):                                   # coordinate refinement of the added weights
        for i in range(fixed, len(cur_t)):
            cand = sorted(set([cur_w[i] * f for f in (0.5, 0.75, 1.0, 1.33, 2.0)]))
            cur_w[i] = max(cand, key=lambda w: float(pp(DT, cur_t, cur_w[:i] + [w] + cur_w[i + 1:]).mean()))
    return cur_t, cur_w, path


def evaluate(DT, DE, terms, ws, ref_terms, ref_ws, ref_post=None):
    ref_t = pp(DT, ref_terms, ref_ws, ref_post)
    ref_e = pp(DE, ref_terms, ref_ws, ref_post)
    posts = {}
    for post in POSTS:
        tg = pp(DT, terms, ws, post)
        ee = pp(DE, terms, ws, post)
        posts[str(post)] = {"tune_gain_vs_ref": round(float((tg - ref_t).mean()), 4),
                            "eval_score": ci(ee), "eval_diff_vs_ref": ci(ee - ref_e)}
    sel = max(posts, key=lambda p: posts[p]["tune_gain_vs_ref"])
    selp = None if sel == "None" else sel
    rr = [L.ranks(postf(fused(d, terms, ws), selp), d["owner"]) for d in DE]
    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 400)]
    R = np.concatenate(rr)
    K = np.concatenate([d["k"] for d in DE])
    return {"terms": terms, "weights": [round(w, 4) for w in ws], "ref_terms": ref_terms, "ref_weights": ref_ws,
            "ref_post": ref_post, "ref_eval_score": ci(ref_e), "post": posts, "post_selected_on_tune": sel,
            "selected_eval_top1": ci([(r == 1).mean() for r in rr]),
            "selected_eval_top5": ci([(r <= 5).mean() for r in rr]),
            "selected_rank_distribution": {"%d-%d" % b: round(float(((R >= b[0]) & (R <= b[1])).mean()), 4) for b in bins},
            "selected_rank_distribution_k5": {"%d-%d" % b: round(float(((R[K == 5] >= b[0]) & (R[K == 5] <= b[1])).mean()), 4)
                                              for b in bins},
            "selected_by_k_vs_ref": by_k(DE, terms, ws, selp, ref_terms, ref_ws, ref_post),
            "nopost_by_k_vs_ref": by_k(DE, terms, ws, None, ref_terms, ref_ws, ref_post)}


if __name__ == "__main__":
    t0 = time.time()
    DT = [load(s) for s in TUNE]
    DE = [load(s) for s in EVAL]
    names = list(DE[0]["Z"])
    base_t = pp(DT, ["base"], [1.0])
    base_e = pp(DE, ["base"], [1.0])
    out = {"what": __doc__, "terms_available": names, "base_eval": ci(base_e), "alone": {}, "with_base": {}}
    for n in ("style2", "lda16", "lda32", "lda64", "lda128"):
        e = pp(DE, [n], [1.0])
        out["alone"][n] = {"tune_mean": round(float(pp(DT, [n], [1.0]).mean()), 4), "eval": ci(e)}
        tg = {w: float((pp(DT, ["base", n], [1.0, w]) - base_t).mean()) for w in WGRID}
        w = max(WGRID, key=lambda x: tg[x])
        ee = pp(DE, ["base", n], [1.0, w])
        out["with_base"][n] = {"tune_gain_by_w": {str(k): round(v, 4) for k, v in tg.items()}, "w": w,
                               "eval_diff_vs_base": ci(ee - base_e), "eval_score": ci(ee)}
        print(n, out["alone"][n], out["with_base"][n]["w"], out["with_base"][n]["eval_diff_vs_base"], flush=True)
    ldas = ["lda16", "lda32", "lda64", "lda128"]
    lda_best = max(ldas, key=lambda n: max(out["with_base"][n]["tune_gain_by_w"].values()))
    out["lda_selected_on_tune"] = lda_best
    others = ["n20", "n30", "deep", "joseki", "joseki_own", "style", "style2", "gw10", "knn1", "knn3", "opp",
              "resp", "prefix", "pknn1", lda_best]
    others = [n for n in others if n in names]
    # v1 of the first run (fused_nopost.json) for reference on the same pools
    with open(os.path.join(HERE, "fused_nopost.json"), encoding="utf-8") as f:
        v1 = json.load(f)
    # ---- v2: greedy from the opening-only base
    print("v2 greedy", flush=True)
    t2, w2, path2 = greedy(DT, ["base"], [1.0], others)
    v2 = evaluate(DT, DE, t2, w2, ["base"], [1.0])
    v2["path"] = path2
    v2_vs_v1 = evaluate(DT, DE, t2, w2, v1["terms"], v1["weights"])
    v2["eval_diff_vs_v1_nopost_by_post"] = {p: v2_vs_v1["post"][p]["eval_diff_vs_ref"] for p in v2_vs_v1["post"]}
    v1c = pp(DE, v1["terms"], v1["weights"], "csls10")
    selp = None if v2["post_selected_on_tune"] == "None" else v2["post_selected_on_tune"]
    v2["eval_diff_selected_vs_v1_csls10"] = ci(pp(DE, t2, w2, selp) - v1c)
    out["v2"] = v2
    spec = {"terms": t2, "weights": w2, "post": selp}
    with open(os.path.join(HERE, "fused_v2_best.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    with open(os.path.join(HERE, "fused_v2_nopost.json"), "w", encoding="utf-8") as f:
        json.dump(dict(spec, post=None), f, indent=1)
    # ---- (d) proxies of Q04
    out["proxy"] = {}
    for pname, stand in (("proxyA_style", "style"), ("proxyB_lda", lda_best)):
        print(pname, flush=True)
        st, sw = ["base", stand], [1.0, 1.0]
        tp, wp, pathp = greedy(DT, st, sw, [n for n in others if n != stand], fixed=2)
        ev = evaluate(DT, DE, tp, wp, st, sw)
        ev["path"] = pathp
        ev["q04proxy_eval_score"] = ci(pp(DE, st, sw))
        ev["q04proxy_gain_vs_base"] = ci(pp(DE, st, sw) - base_e)
        ev["q04proxy_csls10_gain"] = ci(pp(DE, st, sw, "csls10") - pp(DE, st, sw))
        # the v2 spec as is (weights chosen from the opening-only base), on top of the proxy
        tt = st + [t for t in t2 if t not in st]
        ww = sw + [w for t, w in zip(t2, w2) if t not in st]
        ev["v2_terms_added_as_is"] = {"terms": tt, "weights": [round(w, 4) for w in ww],
                                      "eval_diff_vs_proxy_nopost": ci(pp(DE, tt, ww) - pp(DE, st, sw)),
                                      "eval_diff_vs_proxy_csls10": ci(pp(DE, tt, ww, "csls10") - pp(DE, st, sw))}
        out["proxy"][pname] = ev
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "s5b_combine_v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"v2_terms": t2, "v2_w": w2, "v2_post": v2["post"], "v2_sel": v2["post_selected_on_tune"],
                      "v2_by_k": v2["selected_by_k_vs_ref"],
                      "proxy": {k: {"terms": v["terms"], "post": v["post"], "q04proxy": v["q04proxy_eval_score"],
                                    "as_is": v["v2_terms_added_as_is"]} for k, v in out["proxy"].items()}}, indent=1))
    print("total %.0fs" % (time.time() - t0))
