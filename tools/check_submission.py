"""Strict pre-upload check of a submission file (no answers needed). Stdlib only.

    python tools/check_submission.py rank   SUB.csv --questions PUB_Q.csv PRIV_Q.csv [--official]
    python tools/check_submission.py player SUB.csv --questions PUB_Q.csv PRIV_Q.csv \
                                                    --candidates PUB_C.csv PRIV_C.csv [--official]

Question file i is paired with candidate file i: a question from questions[i]
may only name players from candidates[i] (official rule: pub_ questions use
pub_p_ players, priv_ questions use priv_p_ players). The check is stricter
than tools/score.py on purpose: exact header, no extra columns, no spaces,
upper-case labels, every question exactly once. Exit code 1 on any problem,
and on bad arguments (usage on stderr, no traceback).

--official (test day, PREREG_02 section 9 item 5) adds the official structure:
exactly two question files (public, then private) with 400 rows each, ids
starting with pub_ / priv_; for player, two candidate pools of 40,000 rows =
400 players x exactly 100 games, prefixes pub_p_ / priv_p_ and pub_ / priv_,
no player or game shared between the pools, every pick from its question's
own pool; exactly 800 rows in the order of the question files concatenated
(public rows in file order, then private rows); file name submission_rank.csv
or submission_player.csv.
"""

import csv
import os
import sys
from collections import Counter

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
TOP = [f"top_{i}" for i in range(1, 6)]
HEADERS = {"rank": ["question_id", "pred_rank"], "player": ["question_id"] + TOP}

# official test-day structure (tutorial README at 359e06f, PREREG_02 rev 5 section 9)
OFFICIAL_FILE_NAMES = {"rank": "submission_rank.csv", "player": "submission_player.csv"}
OFFICIAL_POOLS = ("public", "private")
OFFICIAL_Q_PREFIX = ("pub_", "priv_")
OFFICIAL_P_PREFIX = ("pub_p_", "priv_p_")
OFFICIAL_G_PREFIX = ("pub_", "priv_")
OFFICIAL_QUESTIONS = 400
OFFICIAL_PLAYERS = 400
OFFICIAL_GAMES_PER_PLAYER = 100
OFFICIAL_CAND_ROWS = OFFICIAL_PLAYERS * OFFICIAL_GAMES_PER_PLAYER
OFFICIAL_ROWS = 2 * OFFICIAL_QUESTIONS

USAGE = """usage:
  python tools/check_submission.py rank   SUB --questions PUB_Q PRIV_Q [--official]
  python tools/check_submission.py player SUB --questions PUB_Q PRIV_Q --candidates PUB_C PRIV_C [--official]
"""


def _question_ids(path):
    csv.field_size_limit(10**8)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r["question_id"] for r in csv.DictReader(f)]


def _pool(path):
    csv.field_size_limit(10**8)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["player_id"] for r in csv.DictReader(f)}


def _pool_rows(path):
    """[(game_id, player_id)] in file order (official mode needs counts and game ids)."""
    csv.field_size_limit(10**8)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [(r["game_id"], r["player_id"]) for r in csv.DictReader(f)]


def _examples(items, k=3):
    items = list(items)
    return items[:k] + (["..."] if len(items) > k else [])


def _official_questions(i, path, ids):
    """Official structure of question file i (0 = public, 1 = private)."""
    errors = []
    pool, prefix = OFFICIAL_POOLS[i], OFFICIAL_Q_PREFIX[i]
    if len(ids) != OFFICIAL_QUESTIONS:
        errors.append(f"{path}: {len(ids)} questions; official {pool} files have exactly {OFFICIAL_QUESTIONS}")
    bad = [q for q in ids if not q.startswith(prefix)]
    if bad:
        errors.append(f"{path}: {len(bad)} question ids do not start with {prefix!r} (question file {i + 1} "
                      f"of --questions must be the {pool} file), e.g. {_examples(bad)}")
    return errors


def _official_pools(path_rows):
    """Official structure of the two candidate pools; path_rows = [(path, [(game_id, player_id)])]."""
    errors = []
    for i, (path, rows) in enumerate(path_rows):
        pool = OFFICIAL_POOLS[i]
        if len(rows) != OFFICIAL_CAND_ROWS:
            errors.append(f"{path}: {len(rows)} candidate rows; official {pool} pools have exactly "
                          f"{OFFICIAL_CAND_ROWS}")
        per_player = Counter(p for _g, p in rows)
        if len(per_player) != OFFICIAL_PLAYERS:
            errors.append(f"{path}: {len(per_player)} distinct player_id; official pools have exactly "
                          f"{OFFICIAL_PLAYERS}")
        off = sorted((p, n) for p, n in per_player.items() if n != OFFICIAL_GAMES_PER_PLAYER)
        if off:
            errors.append(f"{path}: {len(off)} candidates without exactly {OFFICIAL_GAMES_PER_PLAYER} games, "
                          f"e.g. {['%s has %d' % x for x in off[:3]]}")
        bad_p = sorted(p for p in per_player if not p.startswith(OFFICIAL_P_PREFIX[i]))
        if bad_p:
            errors.append(f"{path}: {len(bad_p)} player_id do not start with {OFFICIAL_P_PREFIX[i]!r} "
                          f"(candidate file {i + 1} must be the {pool} pool), e.g. {_examples(bad_p)}")
        games = Counter(g for g, _p in rows)
        dup_g = sorted(g for g, n in games.items() if n > 1)
        if dup_g:
            errors.append(f"{path}: {len(dup_g)} game_id appear more than once, e.g. {_examples(dup_g)}")
        bad_g = sorted(g for g in games if not g.startswith(OFFICIAL_G_PREFIX[i]))
        if bad_g:
            errors.append(f"{path}: {len(bad_g)} game_id do not start with {OFFICIAL_G_PREFIX[i]!r}, "
                          f"e.g. {_examples(bad_g)}")
    if len(path_rows) == 2:
        (_pa, ra), (_pb, rb) = path_rows
        shared_p = sorted({p for _g, p in ra} & {p for _g, p in rb})
        if shared_p:
            errors.append(f"the public and private pools share {len(shared_p)} player_id, e.g. {_examples(shared_p)}")
        shared_g = sorted({g for g, _p in ra} & {g for g, _p in rb})
        if shared_g:
            errors.append(f"the public and private pools share {len(shared_g)} game_id, e.g. {_examples(shared_g)}")
    return errors


def check(kind, sub_path, question_files, candidate_files=(), official=False):
    """List of problems (empty = OK). official=True adds the test-day structure checks."""
    if kind not in HEADERS:
        return [f"kind must be one of {sorted(HEADERS)}, got {kind!r}"]
    question_files, candidate_files = list(question_files), list(candidate_files)
    if not question_files:
        return ["no question files given"]
    errors = []
    if official:
        want = OFFICIAL_FILE_NAMES[kind]
        if os.path.basename(sub_path) != want:
            errors.append(f"file name must be exactly {want} for an official {kind} upload; "
                          f"got {os.path.basename(sub_path)}")
        if len(question_files) != 2:
            errors.append(f"official mode needs exactly 2 question files (public, then private); "
                          f"got {len(question_files)}")
        if kind == "player" and len(candidate_files) != 2:
            errors.append(f"official mode needs exactly 2 candidate files (public, then private); "
                          f"got {len(candidate_files)}")
    with open(sub_path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("file starts with a UTF-8 BOM; write plain UTF-8")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        return errors + [f"not UTF-8: {e}"]
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return errors + ["empty file"]
    header, body = rows[0], rows[1:]
    if header != HEADERS[kind]:
        errors.append(f"header must be exactly {','.join(HEADERS[kind])}; got {','.join(header)}")
        return errors

    owner = {}   # question_id -> index of its question file
    order = []   # question ids of all question files, concatenated in --questions order
    for i, path in enumerate(question_files):
        try:
            ids = _question_ids(path)
        except KeyError:
            return errors + [f"{path}: no question_id column"]
        here = set()
        for q in ids:
            if q in here:
                errors.append(f"question {q} appears twice in {path}")
            elif q in owner:
                errors.append(f"question {q} appears in two question files")
            here.add(q)
            owner[q] = i
            order.append(q)
        if official and i < 2:
            errors += _official_questions(i, path, ids)
    if kind == "player":
        if official:
            try:
                prows = [(p, _pool_rows(p)) for p in candidate_files]
            except KeyError as e:
                return errors + [f"candidate file without column {e}"]
            errors += _official_pools(prows[:2])
            pools = [{pid for _g, pid in r} for _p, r in prows]
        else:
            pools = [_pool(p) for p in candidate_files]
        if len(pools) != len(question_files):
            errors.append("give one candidate file per question file")
            return errors

    seen = set()
    for n, row in enumerate(body, 2):
        if not row:
            errors.append(f"line {n}: blank line")
            continue
        if len(row) != len(header):
            errors.append(f"line {n}: {len(row)} fields, expected {len(header)}")
            continue
        if any(v != v.strip() or v == "" for v in row):
            errors.append(f"line {n}: empty value or surrounding spaces")
        qid = row[0]
        if qid in seen:
            errors.append(f"line {n}: duplicate question {qid}")
        seen.add(qid)
        if qid not in owner:
            errors.append(f"line {n}: unknown question {qid}")
            continue
        if kind == "rank":
            if row[1] not in LEVELS:
                errors.append(f"line {n}: {qid} rank {row[1]!r} is not one of {LEVELS}")
        else:
            picks = row[1:]
            if len(set(picks)) != 5:
                errors.append(f"line {n}: {qid} top-5 players are not distinct")
            wrong = []
            if official and owner[qid] < 2:
                mine = OFFICIAL_P_PREFIX[owner[qid]]
                wrong = [p for p in picks if not p.startswith(mine)]
                if wrong:
                    other = [OFFICIAL_POOLS[j] for j, pre in enumerate(OFFICIAL_P_PREFIX)
                             if any(p.startswith(pre) for p in wrong)]
                    errors.append(f"line {n}: {qid} is a {OFFICIAL_POOLS[owner[qid]]} question but names "
                                  f"players that are not {mine}* ({'/'.join(other) or 'no'} pool): {wrong[:2]}")
            outside = [p for p in picks if p not in pools[owner[qid]] and p not in wrong]
            if outside:
                errors.append(f"line {n}: {qid} names players outside its own pool: {outside[:2]}")
    missing = [q for q in owner if q not in seen]
    if missing:
        errors.append(f"{len(missing)} questions missing (whole file would be invalid), e.g. {missing[:3]}")
    if official:
        if len(body) != OFFICIAL_ROWS:
            errors.append(f"official submissions have exactly {OFFICIAL_ROWS} rows after the header "
                          f"({OFFICIAL_QUESTIONS} public + {OFFICIAL_QUESTIONS} private); got {len(body)}")
        for n, (row, want) in enumerate(zip(body, order), 2):
            got = row[0] if row else ""
            if got != want:
                errors.append(f"line {n}: row order differs from the question files (public rows in file "
                              f"order, then private rows): expected {want}, got {got or '(blank)'}")
                break
    return errors


def _usage_error(message):
    sys.stderr.write(f"error: {message}\n{USAGE}")
    sys.exit(1)


def parse_args(argv):
    """(kind, sub, questions, candidates, official); bad arguments -> usage on stderr and exit 1."""
    if argv and argv[0] in ("-h", "--help"):
        sys.stdout.write(USAGE)
        sys.exit(0)
    if not argv:
        _usage_error("missing KIND (rank or player) and SUB")
    kind = argv[0]
    if kind not in HEADERS:
        _usage_error(f"KIND must be rank or player, got {kind!r}")
    if len(argv) < 2 or argv[1].startswith("-"):
        _usage_error("missing SUB (the submission file)")
    sub = argv[1]
    qs, cs, cur, official = [], [], None, False
    for a in argv[2:]:
        if a in ("-h", "--help"):
            sys.stdout.write(USAGE)
            sys.exit(0)
        if a == "--questions":
            cur = qs
        elif a == "--candidates":
            if kind == "rank":
                _usage_error("--candidates is only for player submissions")
            cur = cs
        elif a == "--official":
            official, cur = True, None
        elif a.startswith("-"):
            _usage_error(f"unknown flag {a!r}")
        elif cur is None:
            _usage_error(f"unexpected argument {a!r}: file names go after --questions / --candidates")
        else:
            cur.append(a)
    return kind, sub, qs, cs, official


def main(argv):
    kind, sub, qs, cs, official = parse_args(argv)
    try:
        errors = check(kind, sub, qs, cs, official=official)
        n_q = sum(len(_question_ids(p)) for p in qs)
    except (OSError, UnicodeDecodeError, csv.Error, KeyError) as e:
        print(f"FAIL: cannot read an input file: {e}")
        sys.exit(1)
    if errors:
        print(f"FAIL ({len(errors)} problems, {n_q} questions expected)")
        for e in errors[:50]:
            print(" -", e)
        sys.exit(1)
    if official:
        print(f"OK (official): {kind} {os.path.basename(sub)} covers all {n_q} questions, "
              f"{n_q} rows in official order")
    else:
        print(f"OK: {kind} submission covers all {n_q} questions and passes every strict check")


if __name__ == "__main__":
    main(sys.argv[1:])
