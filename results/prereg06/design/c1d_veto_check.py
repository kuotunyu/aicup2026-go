"""PREREG_06 design helper (competition 1): how a public-LB veto of the one-to-one switch would behave in the
c1_headroom simulations. Reads only the per-pool rows that results/prereg06/explore/c1_headroom/s8_switch.py
saved (fit players, official-shaped pools, 400 candidates x 100 black games, 100 questions each of k = 5/10/15/20).
No data/local, no tools/score.py, no new simulation.

For every pool and feature level (base / proxyA / v2 of s8_switch.py) it takes
    d = score(switch ON) - score(switch OFF)
with ON = 'hcsls3' (Hungarian Top-1, ranks 2-5 by CSLS k = 3) and OFF = 'csls10' (PREREG_06: CSLS10 always on),
and also ON = 'hung' (Hungarian Top-1, ranks 2-5 by the raw fused score). One pool plays the role of the public pool:
d is exactly what 'public(06-ON) - public(06-OFF)' would show if the public pool looked like that pool.
Reported per scenario: share of pools with d below several thresholds (o2o: a false veto; other scenarios: the
veto's power if the switch had been opened by mistake), and the same restricted to pools where the s8 gate
(r_forced_cost <= the level's threshold) would have opened the switch.

    .venv/Scripts/python.exe results/prereg06/design/c1d_veto_check.py [--rows PATH] [--out PATH]

Default --rows: the session scratchpad copy of s8_rows.json; default --out: c1d_veto_check.json in the same
scratch folder (pass --out results/prereg06/design/c1d_veto_check.json to keep it with the design files).
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os

import numpy as np

SCRATCH = os.environ.get(
    "C10_C1H_SCRATCH",
    r"C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go"
    r"\a8d08794-7564-42a1-933e-30b4ef8b21ce\scratchpad\c1_headroom")
SCEN = ["o2o", "u390", "u375", "u350", "d80", "u300", "wr"]
LEVELS = ["base", "proxyA", "v2"]
# r_forced_cost gate thresholds chosen on the selection seeds by s8_switch.py (s8_switch.json, hung|<off>|r_forced_cost)
GATE_THR = {"base": 1.30116, "proxyA": 1.59776, "v2": 1.58233}
EVAL = set(range(20))
SEL = set(range(100, 110)) | set(range(200, 220))
CUTS = [0.0, -0.001, -0.002, -0.005, -0.01]


def r_forced(d):
    return d["det"]["forced_cost"] / max(d["pseudo"]["forced_cost"], 1e-9)


def summarise(ds):
    ds = np.asarray(ds, np.float64)
    if not len(ds):
        return {"n": 0}
    out = {"n": int(len(ds)), "mean": round(float(ds.mean()), 5), "min": round(float(ds.min()), 5),
           "p05": round(float(np.percentile(ds, 5)), 5), "median": round(float(np.median(ds)), 5),
           "max": round(float(ds.max()), 5)}
    for c in CUTS:
        out["share_d_lt_%g" % c] = round(float((ds < c - 1e-12).mean()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=os.path.join(SCRATCH, "s8_rows.json"))
    ap.add_argument("--out", default=os.path.join(SCRATCH, "c1d_veto_check.json"))
    a = ap.parse_args()
    with open(a.rows, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    out = {"what": __doc__, "rows_file": a.rows, "n_rows": len(rows), "gate_thresholds": GATE_THR,
           "off_method": "csls10", "levels": {}}
    for lv in LEVELS:
        o = {}
        for on in ("hcsls3", "hung"):
            o[on] = {}
            for part, seeds in (("evaluation", EVAL), ("selection", SEL), ("all", EVAL | SEL)):
                o[on][part] = {}
                for sc in SCEN:
                    rr = [r for r in rows if r["scen"] == sc and r["seed"] in seeds]
                    d_all = [r["lv"][lv]["res"][on][0] - r["lv"][lv]["res"]["csls10"][0] for r in rr]
                    d_open = [r["lv"][lv]["res"][on][0] - r["lv"][lv]["res"]["csls10"][0] for r in rr
                              if r_forced(r["lv"][lv]) <= GATE_THR[lv]]
                    o[on][part][sc] = {"if_switch_on": summarise(d_all),
                                       "pools_where_gate_opens": summarise(d_open)}
        out["levels"][lv] = o
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for lv in LEVELS:
        print("==", lv)
        for sc in SCEN:
            x = out["levels"][lv]["hcsls3"]["all"][sc]["if_switch_on"]
            g = out["levels"][lv]["hcsls3"]["all"][sc]["pools_where_gate_opens"]
            print("  %-5s n=%d mean %+.4f min %+.4f med %+.4f max %+.4f  P(d<0)=%.3f P(d<-0.002)=%.3f | gate-open n=%d"
                  % (sc, x["n"], x["mean"], x["min"], x["median"], x["max"], x["share_d_lt_0"],
                     x["share_d_lt_-0.002"], g["n"]))
    print("written", a.out)


if __name__ == "__main__":
    main()
