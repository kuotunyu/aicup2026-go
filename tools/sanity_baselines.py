"""Floor scores on the local validation set: constant ranks, oracle, random Top-5,
and an order-only Top-5 that must score about 0 (row order must not reveal answers).

Usage:
    python tools/sanity_baselines.py data/local results/split_v1/sanity_baselines.json results/score_ledger.csv
These are reference floors only; they use no game content. Every score is logged.
"""

import csv
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import score  # noqa: E402

NOTE = "sanity floors (not a method): tools/sanity_baselines.py"


def main(local_dir, out_path, ledger):
    ra = os.path.join(local_dir, "rank_val_answers.csv")
    pa = os.path.join(local_dir, "player_val_answers.csv")
    pc = os.path.join(local_dir, "player_val_candidates.csv")
    _, rank_ans = score._read(ra)
    _, player_ans = score._read(pa)
    _, cands = score._read(pc)
    pool = sorted({r["player_id"] for r in cands})
    file_order = []
    for r in cands:
        if not file_order or file_order[-1] != r["player_id"]:
            file_order.append(r["player_id"])
    tmp = tempfile.mkdtemp()
    out = {"rank": {}, "player": {}}

    def rank_file(name, fn):
        path = os.path.join(tmp, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["question_id", "pred_rank"])
            for r in rank_ans:
                w.writerow([r["question_id"], fn(r)])
        res = score._summary(score.score_rank(path, ra))
        score.append_ledger(ledger, f"{NOTE} [{name}]", "rank", [(None, path)], ra, res)
        return {"mean": res["mean"], "rank_balanced_mean": res["rank_balanced_mean"]}

    def player_file(name, rows):
        path = os.path.join(tmp, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["question_id"] + score.TOP)
            w.writerows(rows)
        res = score._summary(score.score_player(path, pa, pc))
        score.append_ledger(ledger, f"{NOTE} [{name}]", "player", [(None, path)], pa, res)
        return {"mean": res["mean"]}

    for lv in score.LEVELS:
        out["rank"][f"constant_{lv}"] = rank_file(f"constant_{lv}.csv", lambda r, lv=lv: lv)
    out["rank"]["oracle"] = rank_file("oracle.csv", lambda r: r["rank"])

    rng = random.Random(0)
    out["player"]["random_top5_seed0"] = player_file(
        "random_top5.csv", [[r["question_id"]] + rng.sample(pool, 5) for r in player_ans])
    out["player"]["order_only"] = player_file(
        "order_only.csv", [[r["question_id"]] + [file_order[(i + j) % len(file_order)] for j in range(5)]
                           for i, r in enumerate(player_ans)])
    out["player"]["oracle"] = player_file(
        "oracle_player.csv", [[r["question_id"], r["player_id"]]
                              + [p for p in pool if p != r["player_id"]][:4] for r in player_ans])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
