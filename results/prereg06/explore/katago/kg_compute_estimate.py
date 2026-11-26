"""prereg06 explore / katago: compute ESTIMATE for HumanSL features on a Colab G4 (arithmetic only, no data access).

Inputs that were measured or read (paths in REPORT.md):
  - target moves per game: kg_feat_bench.json (mean over all 720,307 fit games)
  - python featurisation speed per core: kg_feat_bench.json
  - workload sizes: task statement (800 test questions, 1,589 local-validation questions x 12.06 games),
    results/prereg02/RESULTS.md line 63 (inner pseudo-question bank 11,813 games)
  - G4 compute units per hour 8.71: docs/COLAB_RUN.md line 35 (second-hand measurement)
Inputs that are ASSUMPTIONS (scenarios): GPU evals/s 5k / 10k / 20k; number of rank profiles N; CPU workers.
Writes results/prereg06/explore/katago/kg_compute_estimate.json.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    fb = json.load(open(os.path.join(HERE, "kg_feat_bench.json"), encoding="utf-8"))
    t_moves = fb["move_stats_fit"]["target_moves_mean"]
    feat_rate_core = fb["positions_per_second_one_core_including_play"]
    units_per_hour = 8.71

    workloads = {
        "test_public_private": 800 * 12.5,
        "local_validation": 1589 * 12.06,
        "inner_bank": 11813,
        "fit_calibration_20k": 20000,
        "fit_calibration_60k": 60000,
    }
    scenarios_evals_per_s = {"pessimistic_5k": 5000, "central_10k": 10000, "optimistic_20k": 20000}
    profile_counts = [12, 26]
    colour_factor = {"target_moves_only": 1, "both_colours": 2}

    rows = []
    for nprof in profile_counts:
        for cname, cf in colour_factor.items():
            for fitname in ("fit_calibration_20k", "fit_calibration_60k"):
                games = {k: v for k, v in workloads.items() if not k.startswith("fit_")}
                games[fitname] = workloads[fitname]
                positions = {k: v * t_moves * cf for k, v in games.items()}
                evals = {k: p * nprof for k, p in positions.items()}
                total_evals = sum(evals.values())
                row = {"profiles": nprof, "colours": cname, "fit_sample": fitname,
                       "games_total": sum(games.values()), "positions_total": sum(positions.values()),
                       "evals_total": total_evals,
                       "evals_test_only": evals["test_public_private"]}
                for sname, r in scenarios_evals_per_s.items():
                    h = total_evals / r / 3600
                    row[f"gpu_hours_{sname}"] = h
                    row[f"g4_units_{sname}"] = h * units_per_hour
                    row[f"test_day_minutes_{sname}"] = evals["test_public_private"] / r / 60
                rows.append(row)

    # CPU featurisation (python) needed per second to keep the GPU busy, versus what 40 / 24 workers give
    feed = {}
    for nprof in profile_counts:
        for sname, r in scenarios_evals_per_s.items():
            feed[f"N{nprof}_{sname}_positions_per_s_needed"] = r / nprof
    feed["python_featurisation_positions_per_s_one_core_measured"] = feat_rate_core
    feed["python_featurisation_positions_per_s_40_workers_if_linear"] = feat_rate_core * 40
    feed["python_featurisation_positions_per_s_24_workers_if_linear"] = feat_rate_core * 24

    # Named plans used in REPORT.md section 7 (N = 14-profile pilot grid unless stated).
    # inner games used by the registered pseudo-questions: bank 11,813 (eval/fit questions) and 73,169 in the union
    # with the single-colour (mono) questions, counted from results/prereg05/explore/u1/u1_oof.npz (eval_games, mono_games).
    plans_games = {
        "pilot": [("fit_sample_N14", 10000, 14, 1), ("inner_bank_N14", 11813, 14, 1),
                  ("fit_sample_full_grid_N26", 1000, 26, 1), ("variant_fox_source", 500, 14, 1),
                  ("variant_date_2022", 500, 14, 1), ("variant_both_colours", 500, 14, 2)],
        "full_after_go": [("inner_rest_for_mono_gate", 73169 - 11813, 14, 1), ("local_validation", 1589 * 12.06, 14, 1)],
        "test_day": [("test_public_private", 800 * 12.5, 14, 1)],
    }
    plans = {}
    for pname, parts in plans_games.items():
        ev = sum(g * t_moves * cf * n for _, g, n, cf in parts)
        plans[pname] = {"parts": [{"name": a, "games": g, "profiles": n, "colour_factor": cf} for a, g, n, cf in parts],
                        "evals": ev,
                        **{f"gpu_hours_{s}": ev / r / 3600 for s, r in scenarios_evals_per_s.items()},
                        **{f"g4_units_{s}": ev / r / 3600 * units_per_hour for s, r in scenarios_evals_per_s.items()}}

    out = {"what": "ESTIMATE ONLY; GPU evals/s are scenarios, not measurements",
           "target_moves_per_game_measured": t_moves,
           "g4_units_per_hour": units_per_hour,
           "workload_games": workloads,
           "rows": rows,
           "plans": plans,
           "cpu_feed": feed}
    with open(os.path.join(HERE, "kg_compute_estimate.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print(r["profiles"], r["colours"], r["fit_sample"], f"games={r['games_total']:.0f}",
              f"evals={r['evals_total'] / 1e6:.1f}M",
              " ".join(f"{s}:{r['gpu_hours_' + s]:.2f}h/{r['g4_units_' + s]:.0f}u/test{r['test_day_minutes_' + s]:.0f}m"
                       for s in scenarios_evals_per_s))
    print(json.dumps(feed, indent=1))


if __name__ == "__main__":
    main()
