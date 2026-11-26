"""Local scorer for both competitions (official formulas). Stdlib only.

Competition 2 (rank):   exact 1, adjacent level e^-1, otherwise 0; mean over questions.
Competition 1 (player): true player at Top-5 position r scores e^-(r-1), otherwise 0; mean.

Usage:
    python tools/score.py rank    PRED.csv ANSWERS.csv [--json OUT]
    python tools/score.py player  PRED.csv ANSWERS.csv CANDIDATES.csv [--json OUT]
    python tools/score.py compare-rank   PRED_A.csv PRED_B.csv ANSWERS.csv
    python tools/score.py compare-player PRED_A.csv PRED_B.csv ANSWERS.csv CANDIDATES.csv

PRED for rank:   question_id,pred_rank
PRED for player: question_id,top_1,top_2,top_3,top_4,top_5
Missing, extra or duplicate questions, unknown labels, repeated or out-of-pool
players raise SubmissionError. This scorer is lenient about surrounding spaces
in values; how the official system treats spaces, case or extra columns is
unknown, so check real uploads with tools/check_submission.py (strict).

Scoring against the local validation answers (*_val_answers.csv) requires
--ledger results/score_ledger.csv --note "<prereg id and reason>"; every such
score is appended to that ledger (append-only, committed with the results).
"""

import csv
import datetime
import hashlib
import json
import math
import os
import random
import subprocess
import sys

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
LEVEL_INDEX = {lv: i for i, lv in enumerate(LEVELS)}
E1 = math.exp(-1)
TOP = [f"top_{i}" for i in range(1, 6)]


class SubmissionError(ValueError):
    pass


def _read(path):
    csv.field_size_limit(10**8)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def _index_predictions(path, needed_cols, expected_qids):
    cols, rows = _read(path)
    missing_cols = [c for c in needed_cols if c not in cols]
    if missing_cols:
        raise SubmissionError(f"missing columns: {missing_cols}")
    preds = {}
    for r in rows:
        qid = r["question_id"]
        if qid in preds:
            raise SubmissionError(f"duplicate question_id: {qid}")
        preds[qid] = r
    absent = [q for q in expected_qids if q not in preds]
    extra = [q for q in preds if q not in expected_qids]
    if absent:
        raise SubmissionError(f"{len(absent)} questions missing, e.g. {absent[:3]}")
    if extra:
        raise SubmissionError(f"{len(extra)} unknown question_ids, e.g. {extra[:3]}")
    return preds


def rank_points(pred, truth):
    d = abs(LEVEL_INDEX[pred] - LEVEL_INDEX[truth])
    return 1.0 if d == 0 else (E1 if d == 1 else 0.0)


def score_rank(pred_path, answers_path):
    _, answers = _read(answers_path)
    truth = {r["question_id"]: r["rank"] for r in answers}
    preds = _index_predictions(pred_path, ["question_id", "pred_rank"], truth)
    per_q = {}
    confusion = {t: {p: 0 for p in LEVELS} for t in LEVELS}
    for qid, t in truth.items():
        p = preds[qid]["pred_rank"].strip()
        if p not in LEVEL_INDEX:
            raise SubmissionError(f"{qid}: unknown rank {p!r}")
        per_q[qid] = rank_points(p, t)
        confusion[t][p] += 1
    per_rank = {}
    for lv in LEVELS:
        qs = [q for q, t in truth.items() if t == lv]
        if qs:
            per_rank[lv] = {
                "n": len(qs),
                "mean": sum(per_q[q] for q in qs) / len(qs),
                "exact": sum(1 for q in qs if per_q[q] == 1.0) / len(qs),
                "adjacent": sum(1 for q in qs if per_q[q] == E1) / len(qs),
            }
    n = len(per_q)
    return {
        "n": n,
        "mean": sum(per_q.values()) / n,
        "rank_balanced_mean": sum(v["mean"] for v in per_rank.values()) / len(per_rank),
        "exact_rate": sum(1 for v in per_q.values() if v == 1.0) / n,
        "adjacent_rate": sum(1 for v in per_q.values() if v == E1) / n,
        "per_rank": per_rank,
        "confusion_truth_by_pred": confusion,
        "per_question": per_q,
    }


def score_player(pred_path, answers_path, candidates_path):
    _, answers = _read(answers_path)
    truth = {r["question_id"]: r["player_id"] for r in answers}
    _, cands = _read(candidates_path)
    pool = {r["player_id"] for r in cands}
    preds = _index_predictions(pred_path, ["question_id"] + TOP, truth)
    per_q = {}
    for qid, t in truth.items():
        picks = [preds[qid][c].strip() for c in TOP]
        if len(set(picks)) != 5:
            raise SubmissionError(f"{qid}: top-5 players are not distinct")
        outside = [p for p in picks if p not in pool]
        if outside:
            raise SubmissionError(f"{qid}: players not in this candidate pool: {outside[:2]}")
        per_q[qid] = math.exp(-picks.index(t)) if t in picks else 0.0
    n = len(per_q)
    return {
        "n": n,
        "mean": sum(per_q.values()) / n,
        "top1_rate": sum(1 for v in per_q.values() if v == 1.0) / n,
        "top5_rate": sum(1 for v in per_q.values() if v > 0) / n,
        "per_question": per_q,
    }


def paired_bootstrap(per_q_a, per_q_b, strata=None, n_boot=10000, seed=0):
    """B minus A, resampling questions (one question = one player).

    With strata (question_id -> rank) the resampling is done within each rank
    and the statistic is the rank-balanced mean, matching score_rank's
    rank_balanced_mean. Without strata it is the plain mean over questions.
    """
    qids = sorted(per_q_a)
    if sorted(per_q_b) != qids:
        raise ValueError("the two predictions cover different questions")
    diff = {q: per_q_b[q] - per_q_a[q] for q in qids}
    groups = {}
    for q in qids:
        groups.setdefault(strata[q] if strata else "all", []).append(diff[q])
    groups = [g for _, g in sorted(groups.items())]

    def stat(gs):
        return sum(sum(g) / len(g) for g in gs) / len(gs)

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(stat([[g[rng.randrange(len(g))] for _ in g] for g in groups]))
    means.sort()
    return {
        "n": len(qids),
        "strata": len(groups),
        "mean_diff": stat(groups),
        "ci95": [means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]],
        "share_resamples_leq_0": sum(1 for m in means if m <= 0) / n_boot,
    }


def _summary(result):
    return {k: v for k, v in result.items() if k != "per_question"}


LEDGER_COLUMNS = ["utc_time", "git_head", "note", "command", "pred_sha256", "answers_sha256",
                  "n", "mean", "rank_balanced_mean", "details"]


def _sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        return ""


def append_ledger(ledger_path, note, command, pred_paths, answers_path, result):
    """Append one row per scored prediction file. The ledger is append-only."""
    new = not os.path.exists(ledger_path)
    with open(ledger_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LEDGER_COLUMNS)
        for label, path in pred_paths:
            res = result[label] if label else result
            w.writerow([
                datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                _git_head(), note, command, _sha256(path), _sha256(answers_path),
                res.get("n", ""), res.get("mean", ""), res.get("rank_balanced_mean", ""),
                json.dumps({k: v for k, v in res.items()
                            if k in ("exact_rate", "adjacent_rate", "top1_rate", "top5_rate")}),
            ])


def main(argv):
    out = ledger = note = None
    for flag in ("--json", "--ledger", "--note"):
        if flag in argv:
            i = argv.index(flag)
            value = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
            if flag == "--json":
                out = value
            elif flag == "--ledger":
                ledger = value
            else:
                note = value
    if not argv:
        raise SystemExit(__doc__)
    cmd, args = argv[0], argv[1:]
    answers_arg = {"rank": 1, "player": 1, "compare-rank": 2, "compare-player": 2}.get(cmd)
    if answers_arg is None:
        raise SystemExit(__doc__)
    if "_val_answers" in os.path.basename(args[answers_arg]) and not (ledger and note):
        raise SystemExit("scoring against the local validation answers needs "
                         "--ledger results/score_ledger.csv --note \"<prereg id and reason>\"")
    if cmd == "rank":
        full = score_rank(*args)
        res = _summary(full)
        scored = [(None, args[0])]
        ledger_result = res
    elif cmd == "player":
        full = score_player(*args)
        res = _summary(full)
        scored = [(None, args[0])]
        ledger_result = res
    elif cmd == "compare-rank":
        a, b = score_rank(args[0], args[2]), score_rank(args[1], args[2])
        _, answers = _read(args[2])
        strata = {r["question_id"]: r["rank"] for r in answers}
        res = {"A_rank_balanced": a["rank_balanced_mean"], "B_rank_balanced": b["rank_balanced_mean"],
               "A_mean": a["mean"], "B_mean": b["mean"],
               "B_minus_A_rank_balanced": paired_bootstrap(a["per_question"], b["per_question"], strata),
               "B_minus_A_mean": paired_bootstrap(a["per_question"], b["per_question"])}
        scored = [("A", args[0]), ("B", args[1])]
        ledger_result = {"A": _summary(a), "B": _summary(b)}
    else:  # compare-player
        a, b = score_player(args[0], args[2], args[3]), score_player(args[1], args[2], args[3])
        res = {"A_mean": a["mean"], "B_mean": b["mean"],
               "B_minus_A": paired_bootstrap(a["per_question"], b["per_question"])}
        scored = [("A", args[0]), ("B", args[1])]
        ledger_result = {"A": _summary(a), "B": _summary(b)}
    if ledger:
        append_ledger(ledger, note, cmd, scored, args[answers_arg], ledger_result)
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
