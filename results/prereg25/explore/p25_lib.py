"""prereg25 exploration (competition 2): shared pure-python parts (no torch, no GPU).

- the validation players of split v1 and their games, read only from the committed split file
  (results/split_v1/players_split.csv, column split == 'val', via tools/c10data.read_split) and the official training
  CSVs (data/training/train_<LEVEL>.csv; the rank label is the file's level). Never data/local.
- move-sequence keys (original orientation and the canonical key over the 8 board symmetries).
- the V sets: p1_extract.build_probe's round robin (same rule, per level with its own quota and seeded generator) run
  three times on the validation players' games, with the P1 / P2 / P3 quotas scaled per level by (validation players of
  the level / fit players of the level).
"""

import csv
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg08", "p1")):
    if p not in sys.path:
        sys.path.insert(0, p)

import c10data  # noqa: E402

LEVELS = list(c10data.LEVELS)
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
FORBIDDEN = ("/data/local", "rank_val_answers", "player_val_answers")

# PLAN.md 2: the probe batches the S19 rows came from (per level), mirrored on the validation players
PROBE_BATCHES = (("v1", 3000), ("v2", 12000), ("v3", 12000))   # = P1, P2, P3 per-level quotas
V_SEEDS = {"v1": 20261125, "v2": 20261126, "v3": 20261127}
SETS_FORMAT = "c10-prereg08 p1 sets v1"                          # p1_extract.load_sets accepts only this format
SOURCE = "val"

TOKEN_RE = re.compile(r";([BW])\[([a-t]{0,2})\]")


def guard(p):
    q = os.path.abspath(p).replace("\\", "/").lower()
    if any(f in q for f in FORBIDDEN):
        raise SystemExit(f"refused path {p}")
    return p


# ---------------------------------------------------------------- validation players (split v1 file only)
def val_players(split_csv=SPLIT_CSV):
    """player_id -> rank for split == 'val' (the committed split v1 file)."""
    split = c10data.read_split(guard(split_csv))
    return {pid: info["rank"] for pid, info in split.items() if info["split"] == "val"}


def load_val_rows(train_dir=TRAIN_DIR, split_csv=SPLIT_CSV):
    """Every game of the validation players from the official training CSVs (keys = c10data.COLUMNS). The rank of a
    row is its file's level; it must equal the split file's rank of the player."""
    csv.field_size_limit(10 ** 8)
    vp = val_players(split_csv)
    rows = []
    for lv in LEVELS:
        with open(guard(os.path.join(train_dir, f"train_{lv}.csv")), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if next(reader) != c10data.COLUMNS:
                raise ValueError(f"unexpected header in train_{lv}.csv")
            for rec in reader:
                if rec[0] in vp:
                    r = dict(zip(c10data.COLUMNS, rec))
                    if r["rank"] != lv or vp[rec[0]] != lv:
                        raise ValueError(f"player {rec[0]}: rank {r['rank']} / file {lv} / split {vp[rec[0]]}")
                    rows.append(r)
    if {r["player_id"] for r in rows} != set(vp):
        raise ValueError("some validation players have no game in the training CSVs")
    return rows


# ---------------------------------------------------------------- move keys
def _syms():
    def t(s, x, y):
        a, b = [(x, y), (18 - x, y), (x, 18 - y), (18 - x, 18 - y),
                (y, x), (18 - y, x), (y, 18 - x), (18 - y, 18 - x)][s]
        return a, b
    return t


_T = _syms()


_MAPS = [{chr(97 + x) + chr(97 + y): (lambda ab: chr(97 + ab[0]) + chr(97 + ab[1]))(_T(s, x, y))
           for x in range(19) for y in range(19)} for s in range(8)]


def tokens(sgf):
    """[(colour, 'xy' or '' for a pass)] of the main line, in order."""
    return [(c, "" if p in ("", "tt") else p) for c, p in TOKEN_RE.findall(sgf)]


def _key(toks, s):
    m = _MAPS[s]
    return hashlib.sha256("".join(f"{c}[{m[p] if p else ''}]" for c, p in toks).encode("ascii")).hexdigest()


def move_key(sgf):
    """sha256 of the move sequence in the original orientation -> (key, number of moves)."""
    toks = tokens(sgf)
    return _key(toks, 0), len(toks)


def sym_keys(sgf):
    """the 8 move-sequence keys of the game under the 8 board symmetries (index 0 = original)."""
    toks = tokens(sgf)
    return [_key(toks, s) for s in range(8)]


# ---------------------------------------------------------------- V sets (PLAN.md 2.2)
def scaled_quotas(val_per_level, fit_per_level):
    """per batch, per level: round(probe quota x validation players / fit players)."""
    return {b: {lv: int(round(q * val_per_level[lv] / fit_per_level[lv])) for lv in LEVELS}
            for b, q in PROBE_BATCHES}


def build_v_sets(rows, quotas):
    """build_probe's round robin per level with that level's quota, batch after batch on the games not yet taken
    (as P2 / P3 excluded P1 / P1+P2). -> {batch: [game dicts]} (deterministic)."""
    taken = set()
    out = {}
    for b, _q in PROBE_BATCHES:
        games = []
        for lv in LEVELS:
            lv_rows = [r for r in rows if r["rank"] == lv and (r["rank"], r["game_id"]) not in taken]
            games += _build_one_level(lv_rows, lv, quotas[b][lv], V_SEEDS[b])
        for g in games:
            taken.add((g["rank"], g["game_id"]))
        out[b] = games
    return out


def _build_one_level(lv_rows, lv, quota, seed):
    """results/prereg08/p1/p1_extract.build_probe for one level, line for line (players sorted then shuffled, round
    robin, each turn one random not-yet-taken game of that player, until the quota), except: the quota is per level and
    the generator is random.Random('<seed>|<level>') (one per level, so a level's draw does not depend on the others)."""
    if quota == 0:
        return []
    import random
    by_player = {}
    for r in lv_rows:
        by_player.setdefault(r["player_id"], []).append(r)
    rng = random.Random(f"{seed}|{lv}")
    players = sorted(by_player)
    rng.shuffle(players)
    pools = {p: sorted(by_player[p], key=lambda x: x["game_id"]) for p in players}
    out, taken = [], 0
    while taken < quota:
        moved = False
        for p in players:
            if taken >= quota:
                break
            pool = pools[p]
            if not pool:
                continue
            r = pool.pop(rng.randrange(len(pool)))
            out.append({"rank": lv, "game_id": r["game_id"], "player_id": p, "color": r["color"]})
            taken += 1
            moved = True
        if not moved:
            raise ValueError(f"level {lv}: only {taken} validation games, {quota} wanted")
    return out


# ---------------------------------------------------------------- S25 player roles (PLAN.md 2.5)
SALT = "c10-prereg25|holdout|"


def holdout_bit(key):
    """1 = H (held out, step choice only), 0 = A (training); key = '<rank>:<player_id>' (fresh salt)."""
    return int(int(hashlib.sha256(f"{SALT}{key}".encode("utf-8")).hexdigest()[:8], 16) % 10 == 0)


def loaded_repo_modules(repo=REPO):
    """Repository files of every imported module. (p19_train / p23_eval build the prefix as REPO + os.sep, which
    misses everything when the repository root is a drive root such as the subst drive Q:\; here the prefix is
    normalised.)"""
    root = os.path.join(os.path.abspath(repo), "").lower()
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).lower().startswith(root):
            out.add(os.path.relpath(os.path.abspath(f), repo).replace("\\", "/"))
    return out
