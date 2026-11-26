"""c2_headroom (a): collect the six a_transductive_<scenario>.json files into a_summary.json (and print a table)."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = ("bal_cell", "bal_level", "near_bal", "prop_players", "dir10", "dir3")
METHODS = ("EM", "SK", "H40", "H10k", "ORC")


def r4(x):
    return None if x is None else round(float(x), 4)


def main():
    out = {"what": "summary of a_transductive_<scenario>.json (Q02 and Q00 on simulated 400-question files; "
                   "design stage, NOT validation)", "scenarios": {}}
    thr = None
    for s in SCEN:
        with open(os.path.join(HERE, f"a_transductive_{s}.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        thr = d["gate_thresholds_bal_cell_Q02"]
        sc = d["scenarios"][s]
        rec = {"files": sc["files"], "reuse_slots_mean": r4(sc["reuse_slots_mean"]), "max_use_max": sc["max_use_max"],
               "true_count_range_mean": r4(sc["true_count_range_mean"])}
        for cb in ("Q02", "Q00"):
            c = sc["combos"][cb]
            rc = {"B_qm_mean": r4(c["B_qm"]["mean"]), "B_qm_p2.5_p97.5": [r4(c["B_qm"]["p2.5"]), r4(c["B_qm"]["p97.5"])]}
            for m in METHODS:
                g = c[m]["gain_qm"]
                rc[m] = {"gain_qm_mean": r4(g["mean"]), "files_p5": r4(g["p5"]), "files_p95": r4(g["p95"]),
                         "files_min": r4(g["min"]), "share_files_worse": r4(c[m]["share_files_worse_qm"]),
                         "gain_rb_mean": r4(c[m]["gain_rb"]["mean"])}
            rc["gated"] = {k: {"accept_rate": r4(v["accept_rate"]), "gain_qm_mean": r4(v["gain_qm"]["mean"]),
                               "files_p5": r4(v["gain_qm"]["p5"]), "files_min": r4(v["gain_qm"]["min"])}
                           for k, v in c["gated"].items()}
            rc["gate_stat_means"] = {k: r4(v["mean"]) for k, v in c["gate_stats"].items()}
            rc["hist_B_mean"] = [round(x, 1) for x in c["hist_B_mean"]]
            rec[cb] = rc
        pb = sc.get("player_boot_mean_gain_Q02")
        if pb:
            rec["Q02_player_boot"] = {
                "qm": {m: {"mean": r4(v["mean"]), "ci95": [r4(x) for x in v["ci95"]]} for m, v in pb["qm"].items()},
                "rb": {m: {"mean": r4(v["mean"]), "ci95": [r4(x) for x in v["ci95"]]} for m, v in pb["rb"].items()},
                "gated_qm": {m: {"mean": r4(v["mean"]), "ci95": [r4(x) for x in v["ci95"]]}
                             for m, v in pb["gated_qm"].items()}}
        out["scenarios"][s] = rec
    out["gate_thresholds_bal_cell_Q02"] = {g: {k: r4(v) for k, v in t.items()} for g, t in thr.items()}
    with open(os.path.join(HERE, "a_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    for s, rec in out["scenarios"].items():
        q = rec["Q02"]
        print(f"\n{s}: B qm {q['B_qm_mean']}  reuse {rec['reuse_slots_mean']}  range {rec['true_count_range_mean']}")
        for m in METHODS:
            pb = rec.get("Q02_player_boot", {}).get("qm", {}).get(m, {})
            print(f"  {m:5s} gain {q[m]['gain_qm_mean']:+.4f} boot {pb.get('ci95')}  p5 {q[m]['files_p5']:+.4f} "
                  f"min {q[m]['files_min']:+.4f} worse {q[m]['share_files_worse']:.2f} | Q00 {rec['Q00'][m]['gain_qm_mean']:+.4f}")
        for k, v in q["gated"].items():
            print(f"  gated {k:28s} acc {v['accept_rate']:.2f} gain {v['gain_qm_mean']:+.4f} p5 {v['files_p5']:+.4f} "
                  f"min {v['files_min']:+.4f}")
        print("  gate stat means", q["gate_stat_means"])
        print("  hist_B", q["hist_B_mean"])
    print(json.dumps(out["gate_thresholds_bal_cell_Q02"]))


if __name__ == "__main__":
    main()
