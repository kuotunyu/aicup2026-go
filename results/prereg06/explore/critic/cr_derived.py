"""prereg06 critic: numbers derived from files other topics wrote (no data access, no scoring tool).

1. Competition-1 pool-to-pool SD, back-computed from the 95% t intervals over the 20 evaluation pools in
   c1_headroom/s8_switch.json and s5b_combine_v2.json (SD = half-width * sqrt(20) / t_{0.975,19}).
2. Ratio "player-specific / random sampling" for Kuboki 2025 (prior_art/convert_literature.json and
   katago/kg_prior_art_convert.json) and for our Q02 (critic/cr_checks.json random sampling vs
   prior_art/convert_literature.json ours.Q02 single-colour player-specific).
Writes cr_derived.json next to this file.
"""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import os  # noqa: E402

from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)


def load(*p):
    with open(os.path.join(EX, *p), encoding="utf-8") as fh:
        return json.load(fh)


t = float(stats.t.ppf(0.975, 19))
s8 = load("c1_headroom", "s8_switch.json")
s5b = load("c1_headroom", "s5b_combine_v2.json")


def sd_from_ci(ci):
    return (ci[1] - ci[0]) / 2 * 20 ** 0.5 / t


c1 = {
    "t_0975_19": t,
    "base_o2o_score_ci": s8["levels"]["base"]["eval_by_method"]["base"]["o2o"]["score"],
    "base_o2o_pool_sd": sd_from_ci(s8["levels"]["base"]["eval_by_method"]["base"]["o2o"]["score"][1:]),
    "proxyA_o2o_pool_sd": sd_from_ci(s8["levels"]["proxyA"]["eval_by_method"]["base"]["o2o"]["score"][1:]),
    "v2_nopost_o2o_pool_sd": sd_from_ci(s8["levels"]["v2"]["eval_by_method"]["base"]["o2o"]["score"][1:]),
    "v2_csls10_wr_pool_sd": sd_from_ci(s8["levels"]["v2"]["eval_by_method"]["csls10"]["wr"]["score"][1:]),
    "paired_sd_v2csls3_minus_base": sd_from_ci(s5b["v2"]["post"]["csls3"]["eval_diff_vs_ref"][1:]),
    "paired_sd_proxyA_v2terms_csls10": sd_from_ci(
        s5b["proxy"]["proxyA_style"]["v2_terms_added_as_is"]["eval_diff_vs_proxy_csls10"][1:]),
}

conv = load("prior_art", "convert_literature.json")
cr = load("critic", "cr_checks.json")
kub_ps = conv["literature"]["Kuboki2025_player_specific_proposed"]["by_k"]
kub_rs = conv["literature"]["Kuboki2025_random_sampling_proposed"]["by_k"]
ours_ps = conv["ours"]["Q02"]
ours_rs = cr["check_a_kuboki_comparability"]["random_sampling_Q02"]
ratio = {}
for k in ("5", "10", "15"):
    ratio[k] = {
        "kuboki_player_specific": kub_ps[k]["score"], "kuboki_random": kub_rs[k]["score"],
        "kuboki_ratio": kub_ps[k]["score"] / kub_rs[k]["score"],
        "ours_player_specific_mono": ours_ps[k]["score_rb"],
        "ours_random_mono": ours_rs[k]["rb_mean_over_fold_params"],
        "ours_ratio": ours_ps[k]["score_rb"] / ours_rs[k]["rb_mean_over_fold_params"],
    }
ratio["ours_random_k20"] = ours_rs["20"]["rb_mean_over_fold_params"]
ratio["note"] = ("Kuboki random 5 games 0.811 vs ours random 20 games: per-game evidence in the paper's setting "
                 "(Fox, SE + HumanSL, both trained on Fox) is about 4x ours; the player-specific/random ratios agree")

out = {"c1_noise": c1, "kuboki_vs_ours": ratio}
with open(os.path.join(HERE, "cr_derived.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
