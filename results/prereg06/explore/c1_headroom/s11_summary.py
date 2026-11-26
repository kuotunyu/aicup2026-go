"""Step 11: collect the headline numbers of the c1_headroom exploration from the result JSONs of this
folder into c1h_summary.json (every value carries the file and key it came from). Reads only files
next to this script; computes nothing new."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def J(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


s2, s5, s5b, s8, s9 = J("s2_baseline.json"), J("s5_combine.json"), J("s5b_combine_v2.json"), J("s8_switch.json"), J("s9_split_check.json")
s7, s6, s10 = J("s7_level_check.json"), J("s6_more_refs.json"), J("s10_reconcile_dsm.json")
SC = s8["scenarios"]


def m8(lv, m, key="score"):
    return {sc: s8["levels"][lv]["eval_by_method"][m][sc][key] for sc in SC}


def gate(lv, key):
    g = s8["levels"][lv]["gates"][key]
    return {"thr": g["thr"], "eval": {sc: {k: g["by_scenario"]["evaluation"][sc][k] for k in ("on_share", "score", "diff_vs_base")}
                                      for sc in SC}}


out = {
    "note": "mean and 95% t-interval over 20 evaluation pools (seeds 0-19) unless stated; choices made on tuning / selection seeds",
    "a_baseline_opening_only (s2_baseline.json)": {
        "score": s2["score_mean_ci_over_pools"], "top1": s2["top1_mean_ci"], "top5": s2["top5_mean_ci"],
        "by_k": s2["by_k"], "rank_distribution": s2["rank_distribution"], "points_lost": s2["points_lost"]},
    "c_v1_first_run (s5_combine.json greedy)": {
        "terms": s5["greedy"]["terms"], "weights": s5["greedy"]["weights"],
        "diff_vs_base_nopost": s5["greedy"]["post"]["None"]["eval_diff_vs_base"],
        "diff_vs_base_csls10": s5["greedy"]["post"]["csls10"]["eval_diff_vs_base"]},
    "c_v2 (s5b_combine_v2.json v2)": {
        "terms": s5b["v2"]["terms"], "weights": s5b["v2"]["weights"], "post_selected_on_tune": s5b["v2"]["post_selected_on_tune"],
        "by_post": {p: s5b["v2"]["post"][p] for p in s5b["v2"]["post"]},
        "selected_by_k": s5b["v2"]["selected_by_k_vs_ref"],
        "selected_top1": s5b["v2"]["selected_eval_top1"], "selected_top5": s5b["v2"]["selected_eval_top5"]},
    "c_terms_alone_and_with_base": {"first_run (s5_combine.json)": {n: s5["with_base"][n]["eval_diff_vs_base"] for n in s5["with_base"]},
                                    "continuation (s5b_combine_v2.json)": {n: s5b["with_base"][n]["eval_diff_vs_base"] for n in s5b["with_base"]},
                                    "alone_continuation": s5b["alone"]},
    "d_proxy (s5b_combine_v2.json proxy)": {
        p: {"q04proxy_score": v["q04proxy_eval_score"], "q04proxy_gain_vs_base": v["q04proxy_gain_vs_base"],
            "greedy_terms": v["terms"], "post_selected": v["post_selected_on_tune"],
            "gain_by_post": {pp: v["post"][pp]["eval_diff_vs_ref"] for pp in v["post"]},
            "v2_terms_as_is": v["v2_terms_added_as_is"]} for p, v in s5b["proxy"].items()},
    "b_switch (s8_switch.json)": {
        "chosen_on_selection": {lv: s8["levels"][lv]["chosen_on_selection"] for lv in s8["levels"]},
        "hung_diff_vs_base": {lv: m8(lv, "hung", "diff_vs_base") for lv in s8["levels"]},
        "recommended_gate_v2 hcsls3|csls10|r_forced_cost": gate("v2", "hcsls3|csls10|r_forced_cost"),
        "recommended_gate_proxyA hcsls3|csls5|r_forced_cost": gate("proxyA", "hcsls3|csls5|r_forced_cost"),
        "recommended_gate_base hcsls3|csls3|r_forced_cost": gate("base", "hcsls3|csls3|r_forced_cost"),
        "detector_r_forced_cost_eval": {lv: s8["levels"][lv]["detector_eval"]["r_forced_cost"] for lv in s8["levels"]},
        "detector_forced_cost_eval": {lv: s8["levels"][lv]["detector_eval"]["forced_cost"] for lv in s8["levels"]},
        "transfer": {k: v for k, v in s8["threshold_transfer_eval"].items() if k.endswith("|forced_cost") or k.endswith("|r_forced_cost")}},
    "robustness": {
        "split_ord_vs_random_gain_diff (s9_split_check.json)": s9["gain_ord_minus_random"],
        "ord_scores (s9)": {n: s9["by_split"]["ord"][n]["score"] for n in s9["by_split"]["ord"]},
        "level_single_vs_mixed (s7_level_check.json)": {k: {kk: s7["table"][k][kk] for kk in ("base", "fused_v2", "fused_v2_csls3", "gain_fused_v2", "gain_fused_v2_csls3", "lda_alone")}
                                                        for k in ("single", "mixed")},
        "more_refs_opening_only (s6_more_refs.json)": s6["table"]},
    "reconcile_prior_art (s10_reconcile_dsm.json)": {"tau_chosen": s10["dsm_tau_chosen_on_selection"],
                                                    "eval": s10["eval_diff_vs_base"],
                                                    "seeds_0_4": s10["eval_seeds_0_4_mean_diff (sibling's seeds)"]},
    "a_misses_v2_csls3 (s9 miss_analysis_random)": s9["miss_analysis_random"]["v2_csls3"],
}
with open(os.path.join(HERE, "c1h_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("ok", len(json.dumps(out)))
