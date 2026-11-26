"""prereg09 mapping exploration, step 3: inner evaluation of trained variants (ev_lib.Inner; preview only).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg09/explore/mapping/run_eval.py \
        [--watch] [--scales 4,8] <variant> ...

-> results/prereg09/explore/mapping/eval_baseline.json, eval_<variant>.json
--watch: after the listed variants, keep evaluating every new data/prereg09/tmp/mapping/v_*.npz (until STOP_EVAL).
"""

import argparse
import glob
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import ev_lib  # noqa: E402

TMP = ev_lib.TMP


def one(inner, name, scales):
    out = os.path.join(HERE, f"eval_{name}.json")
    z = np.load(os.path.join(TMP, f"v_{name}.npz"))
    lp = z["lp_inner"]
    with open(os.path.join(TMP, f"v_{name}.json"), encoding="utf-8") as f:
        rec = json.load(f)
    rec["inner"] = {f"x{s:g}": inner.evaluate(lp, s, name) for s in scales}
    ev_lib.dump(out, rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--scales", default="4")
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    scales = [float(s) for s in a.scales.split(",")]
    inner = ev_lib.Inner()
    ev_lib.dump(os.path.join(HERE, "eval_baseline.json"), inner.baseline_record())
    done = set()
    for n in a.names:
        one(inner, n, scales)
        done.add(n)
    while a.watch and not os.path.exists(os.path.join(TMP, "STOP_EVAL")):
        new = [os.path.basename(p)[2:-4] for p in sorted(glob.glob(os.path.join(TMP, "v_*.npz")), key=os.path.getmtime)]
        new = [n for n in new if n not in done and not os.path.exists(os.path.join(HERE, f"eval_{n}.json"))
               and os.path.exists(os.path.join(TMP, f"v_{n}.json"))]
        for n in new:
            one(inner, n, scales)
            done.add(n)
        time.sleep(20)


if __name__ == "__main__":
    main()
