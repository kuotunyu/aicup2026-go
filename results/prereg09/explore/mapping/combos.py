"""prereg09 mapping exploration, step 4: ensembles of trained variants, two-term fusions and scale sweeps
(ev_lib.Inner; preview only).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg09/explore/mapping/combos.py SPEC.json
SPEC: {"ensembles": [[name, ...], ...], "multi": [[name, ...], ...], "scales": {"name or ens:a+b": [s, ...]},
       "out": "combos_<tag>.json"}
Ensemble = mean of the members' probabilities (inner lp and probe OOF; CV log-loss recorded).
Multi = several P1 terms in the fusion, each with its own fitted weight (first term in the P1TO slot).
"""

import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import ev_lib  # noqa: E402

TMP = ev_lib.TMP


def load_v(name):
    f = os.path.join(TMP, f"v_{name}.npz")
    z = np.load(f if os.path.exists(f) else os.path.join(TMP, f"{name}.npz"))   # x_lrfrozen: the frozen p1_map
    return z["lp_inner"], z["oof"].astype(np.float64)


def lse_mean(lps):
    m = np.mean([np.exp(x) for x in lps], 0)
    return np.log(np.clip(m, 1e-300, None))


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        spec = json.load(f)
    y = np.load(os.path.join(TMP, "probe_meta.npz"))["y"]
    inner = ev_lib.Inner()
    out = {"baseline": inner.baseline_record(), "ensembles": {}, "multi": {}, "scales": {}}
    path = os.path.join(HERE, spec["out"])
    cache = {}

    def get(name):
        if name not in cache:
            if name.startswith("ens:"):
                parts = [load_v(n) for n in name[4:].split("+")]
                cache[name] = (lse_mean([p[0] for p in parts]), lse_mean([p[1] for p in parts]))
            else:
                cache[name] = load_v(name)
        return cache[name]

    if spec.get("greedy"):
        # Caruana forward selection with replacement on the PROBE OOF log-loss only (no inner number is used)
        g = spec["greedy"]
        cands = g["candidates"]
        P = {n: np.exp(get(n)[1]) for n in cands}
        chosen, cur, trace = [], None, []
        for step in range(g.get("max_members", 8)):
            best = None
            for n in cands:
                m = P[n] if cur is None else (cur * len(chosen) + P[n]) / (len(chosen) + 1)
                ll = float(-np.log(np.clip(m[np.arange(len(y)), y], 1e-300, None)).mean())
                if best is None or ll < best[1]:
                    best = (n, ll, m)
            if trace and best[1] >= trace[-1]["cv_logloss"] - 1e-5:
                break
            chosen.append(best[0])
            cur = best[2]
            trace.append({"add": best[0], "cv_logloss": best[1]})
            print("greedy", trace[-1], flush=True)
        del P
        lp = lse_mean([get(n)[0] for n in chosen])
        out["greedy"] = {"candidates": cands, "chosen_with_replacement": chosen, "trace": trace,
                         **{f"x{s:g}": inner.evaluate(lp, float(s), "greedy") for s in g.get("scales", [4])}}
        ev_lib.dump(path, out)
    for members in spec.get("ensembles", []):
        name = "ens:" + "+".join(members)
        lp, oof = get(name)
        rec = {"members": members, "cv_logloss": float(-oof[np.arange(len(y)), y].mean()),
               "cv_top1": float((oof.argmax(1) == y).mean())}
        rec["x4"] = inner.evaluate(lp, 4.0, name)
        out["ensembles"][name] = rec
        ev_lib.dump(path, out)
    for members in spec.get("multi", []):
        name = "multi:" + "+".join(members)
        lps = [get(n)[0] for n in members]
        out["multi"][name] = {"members": members, "x4": inner.evaluate(lps, 4.0, name)}
        ev_lib.dump(path, out)
    for name, scales in spec.get("scales", {}).items():
        lp = get(name)[0]
        out["scales"][name] = {f"x{s:g}": inner.evaluate(lp, float(s), name) for s in scales}
        ev_lib.dump(path, out)
    print("->", path)


if __name__ == "__main__":
    main()
