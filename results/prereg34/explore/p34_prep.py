"""prereg34 exploration (competition 2), stage 1b input: OGS P1 chunks -> raw TO feature rows + metadata (PLAN.md 3).
CPU only (.venv). No model, no label, no score; never the answers folder.

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg34/explore
    $S/p34_prep.py prep  [--p1-dir data/prereg34/p1] [--sample <sample_games.csv.gz>; default sets.json games_file]
                         -> data/prereg34/prep/{ogs_X.npy, ogs_meta.npz, prep_record.json}   (once; refuses to overwrite)
    $S/p34_prep.py synth -> data/prereg34/smoke/prep/ (synthetic OGS players: raw X = mu + sd * noise of the frozen Q09
                         map, so the standardised rows are N(0, 1) noise; for the align / train / eval smokes only)

Input format (the same as prereg25's V extraction, results/prereg25/explore/p25_prep.py):
- <p1-dir>/sets.json: {"probe": {"games": [...]}}; game j of the list is the chunk row with ref = j; every game carries
  at least player_id, game_id and color (p34_extract.py sets; an ogs_rank there must equal the sample file's);
- <chunks-dir (default p1-dir)>/probe/chunk_{i:05d}.npz (p1_extract run --out-dir <chunks-dir>), CHUNK = 200 games
  per chunk, in the P1 chunk format (results/prereg08/p1), read by tools/p09_map.read_chunk_rows unchanged
  (format and neutral-metadata checks, usable games = not failed and >= 1 target move: the unusable rows are dropped
  and counted);
- the stage 1a sample file sample_games.csv.gz (p34_ogs.py sample): columns player_id, game_id, ogs_rank, color,
  start_time, speed, n_moves, sgf, bin, player_ogs_median. Each sets game is joined to its sample row by (player_id,
  game_id, color); the row gives ogs_rank (the game's), start_time and the player's OGS median (the window median of
  PLAN.md 2, the x of the alignment in PLAN.md 3).

The stored X is the raw float32 TO features (as v_X.npy); the standardisation with the frozen Q09 map's mu_x / sd_x
(exactly what S25 used: p09_forward.standardise_x, non-finite -> 0) is done by OgsRows.standardised in the readers.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P25 = os.path.join(REPO, "results", "prereg25", "explore")
for p in (os.path.join(REPO, "tools"), P25, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p25_lib as L25  # noqa: E402  (guard; puts tools/ and results/prereg08/p1 on sys.path)
import p09_forward as FW  # noqa: E402
import p09_map as M  # noqa: E402

guard = L25.guard
DATA = os.path.join(REPO, "data", "prereg34")
P1_DIR = os.path.join(DATA, "p1")
PREP = os.path.join(DATA, "prep")
SMOKE_DIR = os.path.join(DATA, "smoke")
SMOKE_PREP = os.path.join(SMOKE_DIR, "prep")
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
CHUNK = 200
BLOCK = 8192
NOISE_SEED = 12345            # = p19_train.NOISE_SEED (smoke noise)
OGS_NOISE_STREAM = 34         # smoke noise of the OGS rows: default_rng([NOISE_SEED, 34, block start])
PREP_FORMAT = "c10-prereg34 explore ogs prep v1"
SAMPLE_COLUMNS = ("player_id", "game_id", "ogs_rank", "color", "start_time", "speed", "n_moves", "sgf", "bin",
                  "player_ogs_median")
META_KEYS = ("ref", "players", "color", "game_id", "ogs_rank", "start_time", "player_median")
OGS_PREFIX = "ogs_"


def log(*x):
    print(*x, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha_file(path):
    h = hashlib.sha256()
    with open(guard(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(path):
    try:
        return os.path.relpath(path, REPO).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


# ---------------------------------------------------------------- metadata join (pure)
def read_sample_index(path):
    """sample_games.csv.gz -> {(player_id, game_id, color): (ogs_rank, start_time, player_median)}."""
    csv.field_size_limit(1 << 24)
    out = {}
    with gzip.open(guard(path), "rt", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        head = tuple(next(rd))
        if head != SAMPLE_COLUMNS:
            raise SystemExit(f"{path}: header {head} is not {SAMPLE_COLUMNS}")
        for row in rd:
            k = (row[0], row[1], row[3])
            if k in out:
                raise SystemExit(f"{path}: row {k} twice")
            out[k] = (float(row[2]), int(row[4]), float(row[9]))
    return out


def build_meta(games, refs, sample_index):
    """games: the sets.json probe games; refs: the usable chunk refs (indices into games) in row order ->
    dict of META_KEYS arrays. Every game must be an OGS game (ids with the 'ogs_' prefix) found in the sample file,
    and a player's median must be the same on all his rows."""
    refs = np.asarray(refs, dtype=np.int64)
    n = len(refs)
    players = np.empty(n, dtype=object)
    color = np.empty(n, dtype="<U1")
    gid = np.empty(n, dtype=object)
    rank = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.int64)
    med = np.empty(n, dtype=np.float64)
    seen_med = {}
    for i, j in enumerate(refs.tolist()):
        g = games[j]
        p, gi, c = str(g["player_id"]), str(g["game_id"]), str(g["color"])
        if not (p.startswith(OGS_PREFIX) and gi.startswith(OGS_PREFIX)) or c not in ("B", "W"):
            raise SystemExit(f"sets game {j}: not an OGS row ({p}, {gi}, {c})")
        s = sample_index.get((p, gi, c))
        if s is None:
            raise SystemExit(f"sets game {j}: ({p}, {gi}, {c}) is not in the sample file")
        if "ogs_rank" in g and float(g["ogs_rank"]) != s[0]:
            raise SystemExit(f"sets game {j}: ogs_rank differs from the sample file")
        if seen_med.setdefault(p, s[2]) != s[2]:
            raise SystemExit(f"player {p}: two OGS medians")
        players[i], color[i], gid[i], rank[i], t[i], med[i] = p, c, gi, s[0], s[1], s[2]
    if len(set(zip(players.tolist(), gid.tolist()))) != n:
        raise SystemExit("a (player, game) pair twice in the usable rows")
    return {"ref": refs, "players": players.astype(str), "color": color, "game_id": gid.astype(str),
            "ogs_rank": rank, "start_time": t, "player_median": med}


# ---------------------------------------------------------------- reader used by align / train / eval
class OgsRows:
    """data/prereg34/prep (or the smoke prep): raw X (mmap) and the META_KEYS arrays."""

    def __init__(self, d=PREP):
        self.dir = d
        self.X = np.load(guard(os.path.join(d, "ogs_X.npy")), mmap_mode="r")
        with np.load(guard(os.path.join(d, "ogs_meta.npz")), allow_pickle=False) as z:
            self.meta = {k: np.asarray(z[k]) for k in META_KEYS}
        self.meta["players"] = self.meta["players"].astype(str)
        self.meta["color"] = self.meta["color"].astype(str)
        self.meta["game_id"] = self.meta["game_id"].astype(str)
        self.n = len(self.meta["ref"])
        if self.X.shape != (self.n, FW.X_DIM):
            raise SystemExit("ogs_X.npy and ogs_meta.npz disagree")
        self.record = read_json(os.path.join(d, "prep_record.json"))

    def files_sha256(self):
        return {f: sha_file(os.path.join(self.dir, f)) for f in ("ogs_X.npy", "ogs_meta.npz")}

    def standardised_into(self, out, rows, mu, sd, smoke):
        """Standardise OGS rows `rows` (ascending indices) into out[0:len(rows)] (float32), blockwise; in smoke the
        features are default_rng([NOISE_SEED, 34, s]) noise (s = block start in `rows`)."""
        rows = np.asarray(rows, dtype=np.int64)
        for s in range(0, len(rows), BLOCK):
            e = min(s + BLOCK, len(rows))
            if smoke:
                out[s:e] = np.random.default_rng([NOISE_SEED, OGS_NOISE_STREAM, s]).standard_normal(
                    (e - s, FW.X_DIM)).astype(np.float32)
            else:
                out[s:e] = FW.standardise_x(np.asarray(self.X[rows[s:e]], dtype=np.float64), mu, sd).astype(
                    np.float32)
        return out

    def standardised(self, mu, sd, smoke, rows=None):
        rows = np.arange(self.n) if rows is None else rows
        return self.standardised_into(np.empty((len(rows), FW.X_DIM), dtype=np.float32), rows, mu, sd, smoke)


def read_json(path):
    with open(guard(path), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- prep
def cmd_prep(a):
    t0 = time.time()
    out_rec = os.path.join(PREP, "prep_record.json")
    for f in ("ogs_X.npy", "ogs_meta.npz", "prep_record.json"):
        if os.path.exists(os.path.join(PREP, f)):
            raise SystemExit(f"{rel(os.path.join(PREP, f))} exists: prep runs once")
    sets_path = os.path.join(a.p1_dir, "sets.json")
    sets = read_json(sets_path)
    games = sets["probe"]["games"]
    n_ch = (len(games) + CHUNK - 1) // CHUNK
    files = [os.path.join(a.chunks_dir or a.p1_dir, "probe", f"chunk_{i:05d}.npz") for i in range(n_ch)]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} OGS P1 chunks missing (first {missing[0]})")
    sample = a.sample or os.path.join(REPO, *sets["probe"]["games_file"].split("/"))
    if "games_file_sha256" in sets["probe"] and sha_file(sample) != sets["probe"]["games_file_sha256"]:
        raise SystemExit(f"{sample}: SHA-256 differs from sets.json games_file_sha256")
    sample_index = read_sample_index(sample)
    n = sum(M.usable_count(p, ()) for p in files)
    os.makedirs(PREP, exist_ok=True)
    tmp = os.path.join(PREP, "ogs_X.npy.tmp")
    X = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, FW.X_DIM))
    refs, shas, unusable, i = [], [], 0, 0
    for k, p in enumerate(files):
        c = M.read_chunk_rows(p, None, (), None)
        expect = np.arange(k * CHUNK, min(len(games), (k + 1) * CHUNK))
        if not np.all(np.isin(c["ref"], expect)) or c["games"] != len(expect):
            raise SystemExit(f"{p}: refs / games outside the chunk's games")
        g = len(c["ref"])
        X[i:i + g] = c["X"]
        refs.append(c["ref"])
        shas.append(c["sha"])
        unusable += c["unusable"]
        i += g
    if i != n:
        raise SystemExit(f"wrote {i} rows, expected {n}")
    X.flush()
    del X
    meta = build_meta(games, np.concatenate(refs) if refs else np.zeros(0, np.int64), sample_index)
    np.savez(os.path.join(PREP, "ogs_meta.npz"), **meta)
    os.replace(tmp, os.path.join(PREP, "ogs_X.npy"))
    rec = {"format": PREP_FORMAT, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "what": "PLAN.md 3: OGS P1 chunks -> raw TO features (tools/p09_map.read_chunk_rows, unchanged) + metadata "
                   "joined from the stage 1a sample file by (player_id, game_id, color)",
           "sets_file": rel(sets_path), "sets_sha256": sha_file(sets_path), "sample_file": rel(sample),
           "sample_sha256": sha_file(sample), "chunks": n_ch, "games": len(games), "rows": int(n),
           "unusable": int(unusable), "players": int(len(set(meta["players"].tolist()))), "chunk_sha256": shas,
           "files_sha256": {f: sha_file(os.path.join(PREP, f)) for f in ("ogs_X.npy", "ogs_meta.npz")},
           "seconds": round(time.time() - t0, 1)}
    write_json(out_rec, rec)
    log(json.dumps({k: rec[k] for k in ("chunks", "games", "rows", "unusable", "players", "seconds")}))
    return 0


# ---------------------------------------------------------------- synthetic OGS rows (smokes only)
def synth_meta(n_players=300, per_player=25, seed=3434):
    """Synthetic OGS players: median uniform on [15, 36] (15 kyu .. 7 dan), per_player rows each, colours random,
    start times increasing (pure; smoke and unit tests)."""
    rng = np.random.default_rng(seed)
    pl, col, gid, rk, t, med = [], [], [], [], [], []
    g = 0
    for i in range(n_players):
        m = float(np.round(rng.uniform(15.0, 36.0), 3))
        for j in range(per_player):
            pl.append(f"{OGS_PREFIX}{1000 + i}")
            col.append("B" if rng.integers(2) == 0 else "W")
            gid.append(f"{OGS_PREFIX}{50000 + g}")
            rk.append(m + float(rng.uniform(-1.0, 1.0)))
            t.append(1_600_000_000 + 3600 * j + i)
            med.append(m)
            g += 1
    n = len(pl)
    return {"ref": np.arange(n, dtype=np.int64), "players": np.array(pl), "color": np.array(col),
            "game_id": np.array(gid), "ogs_rank": np.array(rk), "start_time": np.array(t, dtype=np.int64),
            "player_median": np.array(med)}


def cmd_synth(a):
    out = SMOKE_PREP
    meta = synth_meta(a.players, a.per_player)
    n = len(meta["ref"])
    mp = FW.load_map(Q09_MAP)
    mu, sd = np.asarray(mp["mu_x"], dtype=np.float64), np.asarray(mp["sd_x"], dtype=np.float64)
    os.makedirs(out, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(out, "ogs_X.npy"), mode="w+", dtype=np.float32, shape=(n, FW.X_DIM))
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        z = np.random.default_rng([NOISE_SEED, 3400, s]).standard_normal((e - s, FW.X_DIM))
        X[s:e] = (mu + np.where(np.isfinite(sd), sd, 1.0) * z).astype(np.float32)
    X.flush()
    del X
    np.savez(os.path.join(out, "ogs_meta.npz"), **meta)
    write_json(os.path.join(out, "prep_record.json"),
               {"format": PREP_FORMAT + " SYNTHETIC SMOKE", "smoke": True, "rows": int(n), "unusable": 0,
                "players": int(a.players), "note": "synthetic players and noise features: never a result",
                "files_sha256": {f: sha_file(os.path.join(out, f)) for f in ("ogs_X.npy", "ogs_meta.npz")}})
    log(f"synthetic OGS prep: {a.players} players, {n} rows -> {rel(out)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--p1-dir", default=P1_DIR, help="folder of sets.json")
    p.add_argument("--chunks-dir", default=None, help="p1_extract --out-dir (chunks in <dir>/probe); default --p1-dir")
    p.add_argument("--sample", default=None, help="default: sets.json probe.games_file (SHA checked)")
    p = sub.add_parser("synth")
    p.add_argument("--players", type=int, default=300)
    p.add_argument("--per-player", type=int, default=25)
    a = ap.parse_args(argv)
    return {"prep": cmd_prep, "synth": cmd_synth}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
