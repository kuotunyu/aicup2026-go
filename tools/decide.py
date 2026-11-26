"""Turn per-question rank probabilities into predictions (PREREG_01 point 3). Stdlib only.

    python tools/decide.py rank_val_probs.csv OUT_DIR

Input columns: question_id,n_games,n_parsed,p_D,p_C,p_B,p_A,p_1D,p_2D,p_3D,p_4D,p_5D,p_6D
(one row per question; the p_* columns are the averaged softmax, full floats).

Writes, in OUT_DIR:
    pred_A.csv           rule A (official): the level with the largest probability
    pred_B.csv           rule B (expected score): the level k maximising
                         p_k + e^-1 * (p_{k-1} + p_{k+1}); neighbours past either
                         end (below D, above 6D) count as 0, there is no wrap-around
    decide_summary.json  question count, missing questions, how many A != B,
                         level distribution of each rule, input/output SHA-256
Both prediction files have the columns question_id,pred_rank, rows in input order.

Ties go to the smallest level index (the weaker level), the same as numpy.argmax.

Missing questions: only a question with n_parsed == 0 ("no parsable game in the
whole question", PREREG_01 point 3) is missing; it gets D under both rules (the
official notebook's fallback) and is listed in the summary. Its p_* columns
should be blank (that is what the notebook writes); if they are filled in they
are still validated, then ignored, and counted as missing_with_probs_ignored.

Everything else that is not a clean row stops the run with exit code 1 and a
message naming the line, the question_id and the reason:
    - n_parsed > 0 but some or all p_* blank (never silently filled with D)
    - a row with fewer fields than the header (truncated) or more fields
    - p_* not a number, not finite, negative, or summing to more than
      SUM_TOL away from 1
    - a missing or duplicate column, an empty or duplicate question_id,
      n_games or n_parsed not an integer, negative, or n_parsed > n_games
Probabilities are read with float() and never rounded.

Sum tolerance (SUM_TOL = 1e-5): the notebook takes a float32 softmax per game
and averages it in float32, so the ten probabilities of a question do not add
up to exactly 1. Measured on CPU torch 2.14 / numpy 2.5 by emulating that path
(float32 softmax, float32 mean over 1-20 games, repr, float, math.fsum) on
300,000 synthetic questions with logit scales 1 to 100: median |sum - 1| is
about 2e-8, the largest 2.8e-7; over 12 million single float32 softmax rows
(a question with one parsed game) the largest is 3.8e-7. The 48 questions of
the CPU smoke runs' rank_val_probs.csv are at most 9.7e-8 away. The Colab run
uses CUDA softmax, whose rounding was not measured, so the guard was widened
from 1e-6 to 1e-5 before any real run (2026-09-25); that still rejects a real
error (a dropped or duplicated column, a wrong divisor) and never changes which
level wins. A row past the tolerance is not renormalised or
repaired: the whole run stops and nothing is written, so an answer is never
changed quietly.

Outputs: at the start the three output files (and any leftover *.tmp of them)
are deleted from OUT_DIR, so a failed run never leaves an older prediction
behind that looks current. The new files are written as *.tmp and renamed only
after all three are complete; if any step fails, none of the three is left.
"""

import csv
import hashlib
import json
import math
import os
import sys

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
PROB_COLS = ["p_" + lv for lv in LEVELS]
REQUIRED_COLS = ["question_id", "n_games", "n_parsed"] + PROB_COLS
E1 = math.exp(-1)
SUM_TOL = 1e-5  # see "Sum tolerance" in the module docstring
FALLBACK = "D"
OUT_NAMES = ("pred_A.csv", "pred_B.csv", "decide_summary.json")
TMP_SUFFIX = ".tmp"


class DecideError(ValueError):
    pass


def argmax_first(values):
    """Index of the largest value; ties go to the first one (like numpy.argmax)."""
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def b_scores(p):
    """Expected score of guessing each level if p were the true distribution."""
    n = len(p)
    out = []
    for k in range(n):
        left = p[k - 1] if k > 0 else 0.0
        right = p[k + 1] if k < n - 1 else 0.0
        out.append(p[k] + E1 * (left + right))
    return out


def rule_a(p):
    return LEVELS[argmax_first(p)]


def rule_b(p):
    return LEVELS[argmax_first(b_scores(p))]


def _int_field(row, col, where):
    raw = row[col].strip()
    try:
        return int(raw)
    except ValueError:
        raise DecideError(f"{where}: {col} is not an integer: {raw!r}") from None


def read_probs(path):
    """Read and validate the probability file.

    Returns a list of dicts: question_id, n_games, n_parsed, missing (True when
    n_parsed == 0), probs (list of 10 floats, or None when every p_* is blank).
    """
    csv.field_size_limit(10**8)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        cols = next(reader, [])
        rows = []
        for fields in reader:
            if fields:  # a blank line holds no question (csv.DictReader skips it too)
                rows.append((reader.line_num, fields))
    absent = [c for c in REQUIRED_COLS if c not in cols]
    if absent:
        raise DecideError(f"missing columns: {absent}")
    dup_cols = sorted({c for c in cols if cols.count(c) > 1})
    if dup_cols:
        raise DecideError(f"duplicate column names in the header: {dup_cols}")
    if not rows:
        raise DecideError("no questions in the input")

    qid_at = cols.index("question_id")
    seen = set()
    out = []
    for line, fields in rows:
        qid = fields[qid_at].strip() if qid_at < len(fields) else ""
        where = f"line {line} (question_id {qid!r})"
        if len(fields) > len(cols):
            raise DecideError(f"{where}: {len(fields)} fields, more than the {len(cols)} columns "
                              f"of the header")
        if len(fields) < len(cols):
            raise DecideError(f"{where}: truncated row, {len(fields)} fields but the header has "
                              f"{len(cols)} columns; no value for {cols[len(fields):]}")
        r = dict(zip(cols, fields))
        if not qid:
            raise DecideError(f"{where}: empty question_id")
        if qid in seen:
            raise DecideError(f"{where}: duplicate question_id")
        seen.add(qid)

        n_games = _int_field(r, "n_games", where)
        n_parsed = _int_field(r, "n_parsed", where)
        if n_games < 0:
            raise DecideError(f"{where}: n_games is negative: {n_games}")
        if n_parsed < 0:
            raise DecideError(f"{where}: n_parsed is negative: {n_parsed}")
        if n_parsed > n_games:
            raise DecideError(f"{where}: n_parsed > n_games ({n_parsed} > {n_games})")

        raw = [r[c].strip() for c in PROB_COLS]
        blank = [c for c, v in zip(PROB_COLS, raw) if v == ""]
        if blank and n_parsed > 0:
            raise DecideError(f"{where}: n_parsed={n_parsed} > 0 but probability columns are "
                              f"blank: {blank}; only a question with n_parsed == 0 may be "
                              f"filled with {FALLBACK}")
        if blank and len(blank) < len(PROB_COLS):
            raise DecideError(f"{where}: some probability columns blank: {blank}")

        probs = None
        if not blank:
            probs = []
            for c, v in zip(PROB_COLS, raw):
                try:
                    x = float(v)
                except ValueError:
                    raise DecideError(f"{where}: {c} is not a number: {v!r}") from None
                if not math.isfinite(x):
                    raise DecideError(f"{where}: {c} is not finite: {v!r}")
                if x < 0:
                    raise DecideError(f"{where}: {c} is negative: {v!r}")
                probs.append(x)
            total = math.fsum(probs)
            if abs(total - 1.0) > SUM_TOL:
                raise DecideError(f"{where}: probabilities sum to {total!r}, "
                                  f"more than {SUM_TOL} away from 1")

        out.append({"question_id": qid, "n_games": n_games, "n_parsed": n_parsed,
                    "missing": n_parsed == 0, "probs": probs})
    return out


def decide(records):
    """Apply both rules. Returns (pred_a, pred_b, missing_qids), each in input order."""
    pred_a, pred_b, missing = [], [], []
    for rec in records:
        if rec["missing"]:
            pred_a.append(FALLBACK)
            pred_b.append(FALLBACK)
            missing.append(rec["question_id"])
        else:
            pred_a.append(rule_a(rec["probs"]))
            pred_b.append(rule_b(rec["probs"]))
    return pred_a, pred_b, missing


def _distribution(preds):
    return {lv: sum(1 for p in preds if p == lv) for lv in LEVELS}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_pred(path, qids, preds):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id", "pred_rank"])
        for q, p in zip(qids, preds):
            w.writerow([q, p])


def _remove(paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def _same_file_key(p):
    return os.path.normcase(os.path.realpath(p))


def run(in_path, out_dir):
    finals = [os.path.join(out_dir, n) for n in OUT_NAMES]
    tmps = [p + TMP_SUFFIX for p in finals]
    if _same_file_key(in_path) in {_same_file_key(p) for p in finals + tmps}:
        raise DecideError(f"the input {in_path} is one of the output files; "
                          f"write the outputs to another directory")
    # Old outputs go first, so a run that fails below leaves none of them behind.
    _remove(finals + tmps)

    records = read_probs(in_path)  # raises before anything is written
    pred_a, pred_b, missing = decide(records)
    qids = [r["question_id"] for r in records]

    changes = {}
    for a, b in zip(pred_a, pred_b):
        if a != b:
            key = f"{a}->{b}"
            changes[key] = changes.get(key, 0) + 1

    os.makedirs(out_dir, exist_ok=True)
    tmp_a, tmp_b, tmp_s = tmps
    try:
        _write_pred(tmp_a, qids, pred_a)
        _write_pred(tmp_b, qids, pred_b)
        summary = {
            "n_questions": len(records),
            "n_missing": len(missing),
            "missing_question_ids": missing,
            "missing_rule": "n_parsed == 0 (no parsable game in the question); "
                            "n_parsed > 0 with blank probabilities is an error",
            "missing_with_probs_ignored": sum(1 for r in records
                                              if r["missing"] and r["probs"] is not None),
            "missing_fallback": FALLBACK,
            "n_A_ne_B": sum(1 for a, b in zip(pred_a, pred_b) if a != b),
            "A_to_B_changes": dict(sorted(changes.items())),
            "distribution_A": _distribution(pred_a),
            "distribution_B": _distribution(pred_b),
            "rule_A": "argmax p_k",
            "rule_B": "argmax p_k + e^-1*(p_{k-1}+p_{k+1}), neighbours past D or 6D count as 0",
            "e_minus_1": E1,
            "tie_break": "smallest level index (weaker level), same as numpy.argmax",
            "sum_tolerance": SUM_TOL,
            "levels": LEVELS,
            "input": os.path.abspath(in_path),
            "input_sha256": _sha256(in_path),
            "pred_A_sha256": _sha256(tmp_a),
            "pred_B_sha256": _sha256(tmp_b),
        }
        with open(tmp_s, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
            f.write("\n")
        for tmp, final in zip(tmps, finals):  # the summary is renamed last
            os.replace(tmp, final)
    except BaseException:
        _remove(finals + tmps)
        raise
    return summary


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        s = run(argv[0], argv[1])
    except (DecideError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        left = [n for n in OUT_NAMES if os.path.exists(os.path.join(argv[1], n))]
        if left:  # only when an old output could not be deleted (e.g. a locked file)
            print(f"WARNING: could not remove {left} in {argv[1]}; they are NOT from this run",
                  file=sys.stderr)
        else:
            print(f"nothing written; no {', '.join(OUT_NAMES)} in {argv[1]}", file=sys.stderr)
        return 1
    print(f"questions: {s['n_questions']}")
    print(f"missing (n_parsed == 0, no parsable game; filled with {FALLBACK}): {s['n_missing']}"
          f"  of which probabilities given but ignored: {s['missing_with_probs_ignored']}")
    if s["missing_question_ids"]:
        print("missing question_ids: " + ", ".join(s["missing_question_ids"]))
    print(f"A != B: {s['n_A_ne_B']}")
    print("A distribution: " + " ".join(f"{k}={v}" for k, v in s["distribution_A"].items()))
    print("B distribution: " + " ".join(f"{k}={v}" for k, v in s["distribution_B"].items()))
    print(f"wrote pred_A.csv, pred_B.csv, decide_summary.json in {argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
