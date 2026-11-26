"""prereg09 mapping exploration: collect eval_*.json and combos_*.json into summary.json and a markdown table.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg09/explore/mapping/summarize.py
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def row(name, cv, r, secs=None):
    v = r["vs_Q08"]
    return {"name": name, "cv_logloss": cv, "scale": r["scale"], "inner": r["inner_rank_balanced"],
            "inner_diff_vs_Q08": v["inner_diff"], "inner_ci95": v["inner_ci95"],
            "mono": r["mono_rank_balanced"], "mono_diff_vs_Q08": v["mono_diff"], "mono_ci95": v["mono_ci95"],
            "mono_B": v["mono_by_colour"].get("B"), "mono_W": v["mono_by_colour"].get("W"),
            "inner_diff_vs_Q02H": r["vs_Q02H"]["inner_diff"],
            "criterion": bool(v["inner_diff"] >= 0.010 and v["inner_ci95"][0] > 0 and v["mono_ci95"][0] > 0
                              and (v["mono_by_colour"].get("B") or 0) > 0 and (v["mono_by_colour"].get("W") or 0) > 0),
            "train_seconds": secs}


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "eval_*.json"))):
        if p.endswith("eval_baseline.json"):
            continue
        r = json.load(open(p, encoding="utf-8"))
        for k, ev in r["inner"].items():
            rows.append(row(r["variant"], r["cv"]["logloss"], ev, r.get("seconds")))
    for p in sorted(glob.glob(os.path.join(HERE, "combos_*.json"))):
        c = json.load(open(p, encoding="utf-8"))
        for name, e in c.get("ensembles", {}).items():
            cv = None if "@" in name else e["cv_logloss"]
            rows.append(row(name + f" [{os.path.basename(p)}]", cv, e["x4"]))
        for name, e in c.get("multi", {}).items():
            rows.append(row(name + f" [{os.path.basename(p)}]", None, e["x4"]))
        for name, e in c.get("scales", {}).items():
            for k, ev in e.items():
                rows.append(row(name + f" [{os.path.basename(p)}]", None, ev))
        if "greedy" in c:
            g = c["greedy"]
            for k in ("x4", "x8"):
                if k in g:
                    rows.append(row("greedy(CV only): " + "+".join(g["chosen_with_replacement"]),
                                    g["trace"][-1]["cv_logloss"], g[k]))
    base = json.load(open(os.path.join(HERE, "eval_baseline.json"), encoding="utf-8"))
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": base, "rows": rows}, f, indent=1)
    print("| variant | scale | CV ll | inner | d vs Q08 [95% CI] | mono d [95% CI] | B | W | crit |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cv = f"{r['cv_logloss']:.4f}" if r["cv_logloss"] is not None else "-"
        print(f"| {r['name']} | x{r['scale']:g} | {cv} | {r['inner']:.4f} | {r['inner_diff_vs_Q08']:+.4f} "
              f"[{r['inner_ci95'][0]:+.4f}, {r['inner_ci95'][1]:+.4f}] | {r['mono_diff_vs_Q08']:+.4f} "
              f"[{r['mono_ci95'][0]:+.4f}, {r['mono_ci95'][1]:+.4f}] | {r['mono_B']:+.4f} | {r['mono_W']:+.4f} | "
              f"{'yes' if r['criterion'] else ''} |")


if __name__ == "__main__":
    main()
