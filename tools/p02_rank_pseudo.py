"""PREREG_02 competition 2: fixed inner bank, pseudo-questions and folds (plan.md 3.3).

    python tools/p02_rank_pseudo.py [--p02-dir results/prereg02/rank]
                                    [--out-dir results/prereg02/rank] [--smoke ...]

Reads
    <p02-dir>/dumps/inner_keys.npz   rank, game_id, player_id, color, label of
                                     every inner game the Colab notebook dumped
    load_rows(part="inner")          the leak-checked training rows (tools/c10data.py)
Writes
    <out-dir>/pseudo_inner.json

Only inner players (split v1, tools/c10data.players_for(split, "inner")) are
used; the script stops if any dumped game is not an inner game of load_rows or
any player is not an inner player. No validation-set label file is read.

Procedure (fixed in PREREG_02 1.2 and plan.md 3.3)
  * n = the player's inner games that are both in the dump and in load_rows
    (in the real run the dump holds all 73,556 inner games, so n is the
    player's number of inner games).
  * bank: players with n >= 5 (636 in the real run). Each bank player's games
    are ordered by sha256("c10-prereg02|bank|{rank}|{game_id}") (hex string)
    and the first min(20, n) form the bank (11,813 games).
  * player order: (LEVELS.index(rank), player_id).
  * round r: rng = random.Random(int(sha256("c10-prereg02|pq|{r}")[:16], 16));
    for every bank player in player order: k = min(rng.randint(5, 20), bank
    size), rng.sample(bank, k) (as tools/make_val.py). Every round draws for
    every bank player, so a player's draws do not depend on who is kept.
  * fit questions: rounds 0..49, kept only for players with n >= 20 (508
    players, bank of exactly 20 games, so k does not depend on the level).
    eval questions: rounds 1000..1199, all bank players.
  * folds: within each level, players ordered by
    sha256("c10-prereg02|fold|{player_id}") are dealt to folds 0,1,2,3,4,0,...

Output format: players (in player order) with their bank as a list of
game_ids; each round is one string of space-separated codes, one code per
kept player in the order of that purpose's player list, one character per
drawn game = its position in the bank ("0".."9", "a".."j"), in draw order.
p02_rank_fuse.py tune decodes the file and re-draws every round to check it.

--smoke (local smoke runs on the notebook's SMOKE dumps only): the output
directory must be outside the repository; <p02-dir>/config.json must say
"smoke": true (the notebook writes it) and the dump must be smaller than the
real one; --min-games / --fit-min-games may lower the two thresholds so
that a 400-game smoke dump still yields questions. Without --smoke both
thresholds are fixed at 5 and 20, the dump must hold all inner games and
must not come from a SMOKE notebook run.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from c10data import EXPECTED, LEVELS, load_rows, players_for, read_split  # noqa: E402

SALT = "c10-prereg02"
MIN_GAMES = 5
FIT_MIN_GAMES = 20
BANK_MAX = 20
K_MIN, K_MAX = 5, 20
N_FOLDS = 5
FIT_ROUNDS = list(range(0, 50))
EVAL_ROUNDS = list(range(1000, 1200))
CODE = "0123456789abcdefghij"
assert len(CODE) == BANK_MAX
OUT_NAME = "pseudo_inner.json"
FORMAT = "c10-prereg02 pseudo_inner v1"

DEFAULTS = {
    "p02_dir": os.path.join(REPO, "results", "prereg02", "rank"),
    "out_dir": os.path.join(REPO, "results", "prereg02", "rank"),
    "train_dir": os.path.join(REPO, "data", "training"),
    "split": os.path.join(REPO, "results", "split_v1", "players_split.csv"),
    "local_dir": os.path.join(REPO, "data", "local"),
}


# ================================================================ common helpers
# (also used by p02_rank_opening.py and p02_rank_fuse.py; rank side only)
class P02Error(RuntimeError):
    pass


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _real(p):
    return os.path.normcase(os.path.realpath(p))


def is_inside(path, root):
    p, r = _real(path), _real(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def refuse_label_file(path):
    """Stop if a path looks like one of the validation label files (data/local/*answer*)."""
    if "answer" in os.path.basename(path).lower():
        raise P02Error(f"refusing to open {path}: validation label files are never read by p02 tools")
    return path


def check_out_dir(out_dir, smoke, extra_readonly=()):
    """Refuse output directories that could overwrite prereg01 or pollute the repo in smoke runs."""
    for ro in (os.path.join(REPO, "results", "prereg01"),) + tuple(extra_readonly):
        if ro and is_inside(out_dir, ro):
            raise P02Error(f"refusing to write into {out_dir}: it is inside the read-only {ro}")
    if smoke and is_inside(out_dir, REPO):
        raise P02Error(f"--smoke output must be outside the repository (got {out_dir}); "
                       f"use a scratch directory")


def notebook_smoke_flag(p02_dir):
    """'smoke' in the notebook's own <p02-dir>/config.json (None when the file or a boolean flag is missing)."""
    path = os.path.join(p02_dir, "config.json")
    if not os.path.exists(path):
        return None
    with open(refuse_label_file(path), encoding="utf-8") as f:
        obj = json.load(f)
    flag = obj.get("smoke") if isinstance(obj, dict) else None
    return flag if isinstance(flag, bool) else None


def check_dump_kind(p02_dir, smoke):
    """--smoke only on dumps that the notebook wrote in SMOKE mode, and a real run never on SMOKE dumps.

    --smoke skips the git checks, so without this a scratch-dir run on the real dumps could produce the
    validation predictions before frozen_config.json is committed (PREREG_02 section 3)."""
    flag = notebook_smoke_flag(p02_dir)
    if smoke and flag is not True:
        raise P02Error(f"--smoke needs dumps from a SMOKE notebook run: {os.path.join(p02_dir, 'config.json')} "
                       f"must say \"smoke\": true (got {flag!r})")
    if not smoke and flag is True:
        raise P02Error(f"{p02_dir} holds SMOKE dumps (config.json \"smoke\": true); real runs refuse them")
    return flag


def check_smoke_size(n, full, what, smoke):
    """--smoke refuses dumps of the real size (a smoke dump holds a few hundred games)."""
    if smoke and n >= full:
        raise P02Error(f"--smoke refused: {what} has {n} rows, the size of the real dump ({full})")


def write_json(path, obj):
    """Write JSON atomically (tmp + rename), UTF-8, LF line ends, floats as repr."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def load_npz(path, required):
    """np.load without pickle; every key in `required` must be present."""
    refuse_label_file(path)
    if not os.path.exists(path):
        raise P02Error(f"missing input: {path}")
    try:
        with np.load(path, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
    except ValueError as e:
        raise P02Error(f"{path}: cannot be read without pickle ({e}); string arrays must be numpy "
                       f"unicode ('<U'), not object") from None
    absent = [k for k in required if k not in out]
    if absent:
        raise P02Error(f"{path}: missing arrays {absent} (has {sorted(out)})")
    return out


def str_array(a):
    return [str(x) for x in np.asarray(a).tolist()]


def load_inner_keys(dumps_dir):
    """inner_keys.npz as lists, with the basic contract checks."""
    path = os.path.join(dumps_dir, "inner_keys.npz")
    z = load_npz(path, ["rank", "game_id", "player_id", "color", "label"])
    keys = {k: str_array(z[k]) for k in ("rank", "game_id", "player_id", "color")}
    label = np.asarray(z["label"]).astype(np.int64)
    n = len(label)
    if any(len(v) != n for v in keys.values()):
        raise P02Error(f"{path}: arrays of different lengths")
    bad_rank = sorted({r for r in keys["rank"] if r not in LEVELS})
    if bad_rank:
        raise P02Error(f"{path}: unknown rank values {bad_rank[:5]}")
    if any(LEVELS[l] != r for l, r in zip(label.tolist(), keys["rank"])):
        raise P02Error(f"{path}: label does not match rank (label must be LEVELS.index(rank))")
    bad_col = sorted({c for c in keys["color"] if c not in ("B", "W")})
    if bad_col:
        raise P02Error(f"{path}: colour values must be 'B'/'W', got {bad_col[:5]}")
    pairs = list(zip(keys["rank"], keys["game_id"]))
    if len(set(pairs)) != n:
        raise P02Error(f"{path}: (rank, game_id) keys are not unique")
    keys["label"] = label
    keys["n"] = n
    keys["path"] = path
    keys["sha256"] = sha256_file(path)
    return keys


# ================================================================ the procedure
def bank_sort_key(rank, game_id):
    return sha_hex(f"{SALT}|bank|{rank}|{game_id}")


def fold_sort_key(player_id):
    return sha_hex(f"{SALT}|fold|{player_id}")


def round_rng(r):
    return random.Random(int(sha_hex(f"{SALT}|pq|{r}")[:16], 16))


def draw_round(r, bank_sizes):
    """Positions drawn in round r for every bank player (bank_sizes in player order).

    rng.sample(range(n), k) makes the same random calls and picks the same
    positions as rng.sample(bank, k) on a bank of n games.
    """
    rng = round_rng(r)
    out = []
    for n in bank_sizes:
        k = min(rng.randint(K_MIN, K_MAX), n)
        out.append(rng.sample(range(n), k))
    return out


def assign_folds(players):
    """players: [(player_id, rank)] -> {player_id: fold}; dealt round-robin within each level."""
    by_level = defaultdict(list)
    for pid, rank in players:
        by_level[rank].append(pid)
    fold = {}
    for rank, pids in by_level.items():
        for i, pid in enumerate(sorted(pids, key=fold_sort_key)):
            fold[pid] = i % N_FOLDS
    return fold


def build_bank(games, min_games=MIN_GAMES):
    """games: [(player_id, rank, game_id)] -> ordered player list with banks.

    Returns [{"player_id", "rank", "n_games", "bank": [game_id, ...]}] in player order.
    """
    by_player = defaultdict(list)
    rank_of = {}
    for pid, rank, gid in games:
        if rank_of.setdefault(pid, rank) != rank:
            raise P02Error(f"player {pid} has games at two levels")
        by_player[pid].append(gid)
    players = []
    for pid in sorted(by_player, key=lambda p: (LEVELS.index(rank_of[p]), p)):
        gids = by_player[pid]
        if len(set(gids)) != len(gids):
            raise P02Error(f"player {pid}: duplicate game ids")
        if len(gids) < min_games:
            continue
        rank = rank_of[pid]
        ordered = sorted(gids, key=lambda g: bank_sort_key(rank, g))
        players.append({"player_id": pid, "rank": rank, "n_games": len(gids),
                        "bank": ordered[:BANK_MAX]})
    return players


def encode(positions):
    return "".join(CODE[i] for i in positions)


def decode(code):
    return [CODE.index(ch) for ch in code]


def make_questions(players, fit_min_games=FIT_MIN_GAMES):
    sizes = [len(p["bank"]) for p in players]
    fit_idx = [i for i, p in enumerate(players) if p["n_games"] >= fit_min_games]
    fit_rounds = []
    for r in FIT_ROUNDS:
        draws = draw_round(r, sizes)
        fit_rounds.append({"round": r, "draws": " ".join(encode(draws[i]) for i in fit_idx)})
    eval_rounds = []
    for r in EVAL_ROUNDS:
        draws = draw_round(r, sizes)
        eval_rounds.append({"round": r, "draws": " ".join(encode(d) for d in draws)})
    return {"fit": {"players": fit_idx, "rounds": fit_rounds},
            "eval": {"players": list(range(len(players))), "rounds": eval_rounds}}


def build(keys, inner_rows, inner_set, smoke=False, min_games=MIN_GAMES, fit_min_games=FIT_MIN_GAMES):
    """keys: load_inner_keys(); inner_rows: {(rank, game_id): (player_id, color)} from load_rows."""
    n = keys["n"]
    if not smoke and n != EXPECTED["inner"]["games"]:
        raise P02Error(f"inner_keys has {n} games, the real run must have all "
                       f"{EXPECTED['inner']['games']} inner games (use --smoke only for smoke dumps)")
    outside = sorted({p for p in keys["player_id"] if p not in inner_set})
    if outside:
        raise P02Error(f"{len(outside)} dumped players are not split-v1 inner players, e.g. {outside[:3]}")
    games = []
    for rank, gid, pid, col in zip(keys["rank"], keys["game_id"], keys["player_id"], keys["color"]):
        got = inner_rows.get((rank, gid))
        if got is None:
            raise P02Error(f"dumped game ({rank}, {gid}) is not an inner game of load_rows")
        if got != (pid, col):
            raise P02Error(f"dumped game ({rank}, {gid}): player/colour {pid}/{col} but load_rows "
                           f"says {got[0]}/{got[1]}")
        games.append((pid, rank, gid))
    players = build_bank(games, min_games)
    if not players:
        raise P02Error("no player has enough dumped inner games for a pseudo-question")
    fold = assign_folds([(p["player_id"], p["rank"]) for p in players])
    for p in players:
        p["fold"] = fold[p["player_id"]]
        p["fit"] = p["n_games"] >= fit_min_games
    questions = make_questions(players, fit_min_games)

    def by_level(pl):
        c = defaultdict(int)
        for p in pl:
            c[p["rank"]] += 1
        return {lv: c[lv] for lv in LEVELS}

    fit_players = [p for p in players if p["fit"]]
    per_fold = {f: by_level([p for p in players if p["fold"] == f]) for f in range(N_FOLDS)}
    per_fold_fit = {f: by_level([p for p in fit_players if p["fold"] == f]) for f in range(N_FOLDS)}
    n_eval_q = len(EVAL_ROUNDS) * len(players)
    n_fit_q = len(FIT_ROUNDS) * len(fit_players)
    return {
        "format": FORMAT,
        "what": "PREREG_02 competition 2: fixed inner bank, pseudo-questions and folds (plan.md 3.3). "
                "Inner players only; no validation data.",
        "params": {"salt": SALT, "min_games": min_games, "fit_min_games": fit_min_games,
                   "bank_max": BANK_MAX, "k_min": K_MIN, "k_max": K_MAX, "n_folds": N_FOLDS,
                   "fit_rounds": [FIT_ROUNDS[0], FIT_ROUNDS[-1]],
                   "eval_rounds": [EVAL_ROUNDS[0], EVAL_ROUNDS[-1]],
                   "bank_order": "sha256('c10-prereg02|bank|{rank}|{game_id}') hex, first min(20, n)",
                   "player_order": "(LEVELS.index(rank), player_id)",
                   "round_rng": "random.Random(int(sha256('c10-prereg02|pq|{r}')[:16], 16)); every "
                                "bank player in player order: k = min(randint(5, 20), bank size), "
                                "sample(bank, k)",
                   "fit_questions": "rounds 0..49, players with n >= fit_min_games only",
                   "eval_questions": "rounds 1000..1199, all bank players",
                   "folds": "within each level, players ordered by sha256('c10-prereg02|fold|{player_id}') "
                            "hex, dealt to folds 0..4 in turn",
                   "code": "one character per drawn game = its bank position, alphabet " + CODE},
        "smoke": bool(smoke),
        "inputs": {"inner_keys": os.path.abspath(keys["path"]), "inner_keys_sha256": keys["sha256"],
                   "inner_keys_games": n},
        "counts": {"bank_players": len(players),
                   "bank_players_by_level": by_level(players),
                   "bank_games": sum(len(p["bank"]) for p in players),
                   "fit_players": len(fit_players),
                   "fit_players_by_level": by_level(fit_players),
                   "players_by_fold_and_level": {str(f): v for f, v in per_fold.items()},
                   "fit_players_by_fold_and_level": {str(f): v for f, v in per_fold_fit.items()},
                   "fit_questions": n_fit_q, "eval_questions": n_eval_q},
        "players": players,
        "questions": questions,
    }


# ================================================================ reading it back
def load_pseudo(path, verify=True):
    """Decode pseudo_inner.json; with verify, re-draw every round and compare."""
    refuse_label_file(path)
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if obj.get("format") != FORMAT:
        raise P02Error(f"{path}: not a {FORMAT} file")
    players = obj["players"]
    sizes = [len(p["bank"]) for p in players]
    out = {"obj": obj, "players": players, "sha256": sha256_file(path)}
    for purpose in ("fit", "eval"):
        part = obj["questions"][purpose]
        idx = part["players"]
        rows = []  # (round, player index, [positions])
        for rd in part["rounds"]:
            codes = rd["draws"].split(" ") if rd["draws"] else []
            if len(codes) != len(idx):
                raise P02Error(f"{path}: {purpose} round {rd['round']} has {len(codes)} draws "
                               f"for {len(idx)} players")
            if verify:
                want = draw_round(rd["round"], sizes)
                if [encode(want[i]) for i in idx] != codes:
                    raise P02Error(f"{path}: {purpose} round {rd['round']} differs from a fresh draw")
            for i, c in zip(idx, codes):
                rows.append((rd["round"], i, decode(c)))
        out[purpose] = rows
    if verify:
        fold = assign_folds([(p["player_id"], p["rank"]) for p in players])
        if any(fold[p["player_id"]] != p["fold"] for p in players):
            raise P02Error(f"{path}: folds differ from a fresh assignment")
        rounds_seen = {"fit": [rd["round"] for rd in obj["questions"]["fit"]["rounds"]],
                       "eval": [rd["round"] for rd in obj["questions"]["eval"]["rounds"]]}
        if rounds_seen != {"fit": FIT_ROUNDS, "eval": EVAL_ROUNDS}:
            raise P02Error(f"{path}: rounds are not fit 0..49 and eval 1000..1199")
    return out


# ================================================================ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--p02-dir", default=DEFAULTS["p02_dir"],
                    help="notebook result dir holding dumps/ (default results/prereg02/rank)")
    ap.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    ap.add_argument("--train-dir", default=DEFAULTS["train_dir"])
    ap.add_argument("--split", default=DEFAULTS["split"])
    ap.add_argument("--local-dir", default=DEFAULTS["local_dir"],
                    help="data/local; load_rows checks no training game is in its question files")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--min-games", type=int, default=MIN_GAMES, help="--smoke only")
    ap.add_argument("--fit-min-games", type=int, default=FIT_MIN_GAMES, help="--smoke only")
    a = ap.parse_args(argv)
    try:
        if not a.smoke and (a.min_games != MIN_GAMES or a.fit_min_games != FIT_MIN_GAMES):
            raise P02Error("--min-games / --fit-min-games may only change with --smoke")
        if a.min_games < 1 or a.fit_min_games < a.min_games:
            raise P02Error("need 1 <= --min-games <= --fit-min-games")
        check_out_dir(a.out_dir, a.smoke)
        check_dump_kind(a.p02_dir, a.smoke)
        keys = load_inner_keys(os.path.join(a.p02_dir, "dumps"))
        check_smoke_size(keys["n"], EXPECTED["inner"]["games"], "inner_keys.npz", a.smoke)
        os.makedirs(a.out_dir, exist_ok=True)
        print(f"inner_keys: {keys['n']} games", flush=True)
        split = read_split(a.split)
        inner_set = players_for(split, "inner")
        rows = load_rows(a.train_dir, a.split, part="inner", local_dir=a.local_dir)
        inner_rows = {(r["rank"], r["game_id"]): (r["player_id"], r["color"]) for r in rows}
        if len(inner_rows) != len(rows):
            raise P02Error("load_rows(part='inner') has duplicate (rank, game_id) keys")
        del rows
        obj = build(keys, inner_rows, inner_set, smoke=a.smoke, min_games=a.min_games,
                    fit_min_games=a.fit_min_games)
        obj["inputs"]["players_split_sha256"] = sha256_file(a.split)
        obj["created"] = now_iso()
        out = os.path.join(a.out_dir, OUT_NAME)
        write_json(out, obj)
        load_pseudo(out, verify=True)  # read back and re-draw
    except P02Error as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    c = obj["counts"]
    print(f"bank: {c['bank_players']} players, {c['bank_games']} games; fit players "
          f"{c['fit_players']} {c['fit_players_by_level']}")
    print(f"questions: fit {c['fit_questions']}, eval {c['eval_questions']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
