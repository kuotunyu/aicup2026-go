"""SUPERSEDED (continuation run): never run to completion; replaced by s8_switch.py. Its safety
margin thr - 0.25 * (thr - median_o2o) moves the threshold to the permissive side whenever the
limit falls below the o2o median, and it only looked at the base term. Do not use.

Step 3b: a label-free switch for the one-to-one assignment.

Calibration pools: seeds 200-219, every scenario (o2o, u390, u375, u350, u300, wr). For each pool:
detector statistics of S and the score of the methods selected in s3 (on the tuning pools 100-109).
The threshold of each statistic is set on the calibration pools only (rule: the loosest threshold
that switches on no calibration pool where the assignment loses score, then a safety margin);
error rates and the switched score are then reported on the evaluation pools 0-19 (rows of s3).

    python s3b_switch.py [--tag base]
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CAL_SEEDS = list(range(200, 220))
SCEN = ["o2o", "u390", "u375", "u350", "u300", "wr"]
STATS = [("forced_cost_z", "le"), ("n0", "le"), ("changed_by_assignment", "le"), ("boot_assign_same", "ge"),
         ("mutual_nn", "ge"), ("share_cand_colmax_z_lt3", "le")]


def one(args):
    seed, scen, term, methods = args
    sys.path.insert(0, HERE)
    import s3_assign as S3
    import c1h_lib as L
    r = S3.one((seed, scen, term))
    r["res"] = {m: r["res"][m] for m in methods}
    return r


def decide(v, thr, how):
    return v <= thr if how == "le" else v >= thr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--term", default="base")
    a = ap.parse_args()
    import c1h_lib as L
    t0 = time.time()
    with open(os.path.join(HERE, "s3_assign_%s.json" % a.tag), encoding="utf-8") as f:
        s3 = json.load(f)
    with open(os.path.join(L.SCRATCH, "s3_rows_%s.json" % a.tag), encoding="utf-8") as f:
        rows_eval = [r for r in json.load(f) if r["seed"] < 100]
    sel = s3["selected_on_tune"]
    on_m = ["hung_raw", "hung_z", sel["hsink"], sel["hcsls"]]
    off_m = ["base", sel["csls"], sel["isoft"]] + ([sel["usink_safe"]] if sel.get("usink_safe") else [])
    methods = list(dict.fromkeys(on_m + off_m))
    cal_path = os.path.join(L.SCRATCH, "s3b_cal_%s.json" % a.tag)
    if os.path.exists(cal_path):
        with open(cal_path, encoding="utf-8") as f:
            rows_cal = json.load(f)
    else:
        with Pool(int(os.environ.get('C1H_WORKERS', '8'))) as pool:
            rows_cal = pool.map(one, [(s, sc, a.term, methods) for s in CAL_SEEDS for sc in SCEN], chunksize=1)
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(rows_cal, f)
    out = {"tag": a.tag, "methods_on": on_m, "methods_off": off_m, "cal_seeds": CAL_SEEDS, "rules": {}}
    # gain of each 'on' method over each 'off' method per scenario (calibration and eval)
    for name, rows in (("cal", rows_cal), ("eval", rows_eval)):
        out["gain_" + name] = {
            "%s_vs_%s" % (mo, mf): {sc: [round(v, 4) for v in L.mean_ci(
                [r["res"][mo]["score"] - r["res"][mf]["score"] for r in rows if r["scen"] == sc])] for sc in SCEN}
            for mo in on_m for mf in off_m[:2]}
        out["stat_" + name] = {s: {sc: [round(float(np.min([r["det"][s] for r in rows if r["scen"] == sc])), 4),
                                        round(float(np.median([r["det"][s] for r in rows if r["scen"] == sc])), 4),
                                        round(float(np.max([r["det"][s] for r in rows if r["scen"] == sc])), 4)]
                                   for sc in SCEN} for s, _ in STATS if s in rows[0]["det"]}
    # thresholds on calibration pools: switch ON method mo (vs OFF method mf) when stat passes thr
    stats_here = [(s, how) for s, how in STATS if s in rows_cal[0]["det"]]
    for mo, mf in ((sel["hcsls"], sel["csls"]), ("hung_raw", "base"), (sel["hsink"], sel["csls"])):
        for s, how in stats_here:
            vals = np.array([r["det"][s] for r in rows_cal])
            gain = np.array([r["res"][mo]["score"] - r["res"][mf]["score"] for r in rows_cal])
            bad = vals[gain < 0]
            if how == "le":
                lim = bad.min() if len(bad) else vals.max()
                good_o2o = np.array([r["det"][s] for r in rows_cal if r["scen"] == "o2o"])
                thr = float(min(lim, good_o2o.max()))
                thr_safe = thr - 0.25 * (thr - float(np.median(good_o2o))) if len(bad) else thr
            else:
                lim = bad.max() if len(bad) else vals.min()
                good_o2o = np.array([r["det"][s] for r in rows_cal if r["scen"] == "o2o"])
                thr = float(max(lim, good_o2o.min()))
                thr_safe = thr + 0.25 * (float(np.median(good_o2o)) - thr) if len(bad) else thr
            res = {}
            for nm, rows in (("cal", rows_cal), ("eval", rows_eval)):
                per = {}
                for sc in SCEN:
                    rr = [r for r in rows if r["scen"] == sc]
                    on = [decide(r["det"][s], thr_safe, how) for r in rr]
                    sw = [r["res"][mo]["score"] if o else r["res"][mf]["score"] for r, o in zip(rr, on)]
                    per[sc] = {"switched_on_share": round(float(np.mean(on)), 3),
                               "score_switch": round(float(np.mean(sw)), 4),
                               "score_off": round(float(np.mean([r["res"][mf]["score"] for r in rr])), 4),
                               "score_on": round(float(np.mean([r["res"][mo]["score"] for r in rr])), 4),
                               "score_base": round(float(np.mean([r["res"]["base"]["score"] for r in rr])), 4),
                               "pools_where_switch_lost_vs_off": int(sum(1 for r, o in zip(rr, on)
                                                                     if o and r["res"][mo]["score"] < r["res"][mf]["score"]))}
                res[nm] = per
            out["rules"]["%s|%s|%s" % (mo, mf, s)] = {"how": how, "thr_limit": round(thr, 4), "thr": round(thr_safe, 4),
                                                     "n_cal_pools_with_loss": int((gain < 0).sum()), "result": res}
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "s3b_switch_%s.json" % a.tag), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for k, v in out["rules"].items():
        print(k, v["how"], v["thr"], {sc: (v["result"]["eval"][sc]["switched_on_share"], v["result"]["eval"][sc]["score_switch"],
                                         v["result"]["eval"][sc]["score_off"]) for sc in SCEN})
    print(json.dumps(out["gain_eval"], indent=0)[:3000])
