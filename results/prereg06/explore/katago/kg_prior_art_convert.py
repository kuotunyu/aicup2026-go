"""prereg06 explore / katago: convert published Go rank-estimation accuracies into this competition's metric.

Metric: exact level = 1, adjacent level = 1/e, else 0. A paper's (exact %, within-one %) pair gives
    score = exact + (within_one - exact) / e.
Numbers are copied from Kuboki et al., arXiv 2505.00279 v1 (https://arxiv.org/html/2505.00279v1), Tables 1, 2, 4
(Fox data, 11 rank groups). This is an analogy only: different server, different groups, different split.
Writes results/prereg06/explore/katago/kg_prior_art_convert.json.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = math.exp(-1.0)

# (n games, exact %, within-one %)
TABLES = {
    "table4_player_specific_chen_SE": [(5, 51.6, 93.1), (10, 59.4, 96.0), (15, 61.1, 96.9)],
    "table4_player_specific_kuboki_all": [(5, 58.1, 95.3), (10, 61.0, 96.2), (15, 63.2, 96.6)],
    "table4_random_chen_SE": [(5, 59.8, 97.5), (10, 74.1, 99.7), (15, 81.3, 100.0)],
    "table4_random_kuboki_all": [(5, 70.4, 99.5), (10, 82.8, 99.9), (15, 89.4, 100.0)],
    "table2_random_use_all": [(1, 37.5, 82.8), (5, 64.8, 98.8), (10, 79.6, 99.9), (15, 87.5, 100.0), (20, 91.7, 100.0)],
    "table2_random_without_prior": [(1, 32.3, 77.3), (5, 58.1, 97.7), (10, 73.0, 99.7), (15, 81.1, 99.9), (20, 86.4, 100.0)],
    "table2_random_without_strength": [(1, 34.4, 77.8), (5, 60.0, 96.9), (10, 73.5, 99.5), (15, 82.0, 99.8), (20, 87.7, 100.0)],
}


def score(exact, within):
    return exact / 100 + (within - exact) / 100 * E1


def main():
    conv = {k: {str(n): score(a, b) for n, a, b in v} for k, v in TABLES.items()}
    deltas = {
        "player_specific_kuboki_minus_chen": {n: conv["table4_player_specific_kuboki_all"][n]
                                              - conv["table4_player_specific_chen_SE"][n]
                                              for n in conv["table4_player_specific_chen_SE"]},
        "random_use_all_minus_without_prior": {n: conv["table2_random_use_all"][n]
                                               - conv["table2_random_without_prior"][n]
                                               for n in conv["table2_random_use_all"]},
    }
    out = {"source": "https://arxiv.org/html/2505.00279v1", "formula": "exact + (within_one - exact)/e",
           "converted": conv, "deltas": deltas}
    with open(os.path.join(HERE, "kg_prior_art_convert.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
