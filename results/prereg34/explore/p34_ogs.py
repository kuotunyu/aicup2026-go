"""prereg34 explore, stage 1a (PLAN.md section 2): parse the OGS game dump and draw the P1 sample.

The OGS dump (za3k.com/ogs, one OGS API game object per line, gzip) is downloaded, untrusted data. Standard library
and numpy only, no import of any repository module; run isolated (`-I`), every path given as an argument, one thread:

    PY -I results/prereg34/explore/p34_ogs.py parse  --src <ogs .json.gz> --out-dir <dir>
    PY -I results/prereg34/explore/p34_ogs.py sample --rows <dir written by parse> --out-dir <dir>
                                                     [--official-csv data/training/train_D.csv]

parse   streams the gzip line by line; a substring prefilter (19 wide, handicap 0) before json.loads; every filter of
        PLAN section 2 (first failing reason counted); accepted games become official-format SGF
        "(;SZ[19];B[xy];W[xy]...)" (the root of the official v1.0 training files holds SZ[19] only: no GM/FF) and two
        rows (black side, white side). Writes rows.csv.gz (player_id, game_id, ogs_rank, color, start_time, speed,
        n_moves, sgf), rows_index.bin (the same rows without the SGF, numeric, for the sample step) and
        parse_record.json.
sample  the player window rule, the 22 one-stone bins (15 kyu .. 7 dan), default_rng(34), <= 150 players per bin and
        each player's newest 25 kept rows. Writes sample_games.csv.gz and sample_record.json (with the PLAN 1a stop
        conditions evaluated).

Both refuse to overwrite their outputs. Exit 0: done (sample: no stop); 3: sample hit a registered 1a stop; 1: refusal.

Definitions fixed here (PLAN section 2 leaves them open):
- OGS rank scale: 0 = 30 kyu, 29 = 1 kyu, 30 = 1 dan, 36 = 7 dan. Bin = floor(player median rank) clipped to
  [15, 36] (15 = "15 kyu" .. 36 = "7 dan"; 22 bins). Nominal 12 kyu .. 6 dan = floor(median) in [18, 35].
- A row's time is the game's start_time (epoch seconds); ties are broken by game_id. Date window
  2019-01-01 00:00:00 .. 2025-05-31 23:59:59 in Taiwan time (UTC+8, no daylight saving since 1979), inclusive.
- Window rule: a player's rows sorted old -> new; the run starts as the newest row and grows backwards one row at a
  time while every row of the grown run is within WIN_TOL = 2.0 (inclusive; PLAN revision 1) of the grown run's median (median of an even count
  = mean of the two middle values); it stops at the first row that breaks this, or at 40 rows (the kept rows are the
  newest 40 of the run, so a longer run changes nothing). Run >= 20 rows -> the player is accepted; kept rows = the
  run (<= 40); player rank = median of the kept rows.
- Sampling: one generator default_rng(34); bins in ascending order; inside a bin the players sorted by numeric OGS
  id, then rng.permutation; the first 150; each player's newest 25 kept rows.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
from bisect import insort
from datetime import datetime, timedelta, timezone

import numpy as np

PARSE_FORMAT = "c10-prereg34 ogs parse v1"
SAMPLE_FORMAT = "c10-prereg34 ogs sample v1"

TPE = timezone(timedelta(hours=8))
T_MIN = int(datetime(2019, 1, 1, 0, 0, 0, tzinfo=TPE).timestamp())
T_MAX = int(datetime(2025, 5, 31, 23, 59, 59, tzinfo=TPE).timestamp())
MIN_MOVES = 100
COLUMNS = ("player_id", "game_id", "ogs_rank", "color", "start_time", "speed", "n_moves", "sgf")
SPEEDS = ("live", "blitz", "rapid", "unknown", "other")      # correspondence never passes the filter
IDX_DTYPE = np.dtype([("pid", "<i8"), ("gid", "<i8"), ("rank", "<f8"), ("t", "<i8"), ("row", "<i8"),
                      ("color", "i1"), ("speed", "i1"), ("nmoves", "<i4")])

# window and sampling
WIN_TOL = 2.0  # PLAN revision 1 (was 1.0)
WIN_MIN = 20
WIN_KEEP = 40
BIN_LO, BIN_HI = 15, 36                 # 15 kyu .. 7 dan on the OGS scale (0 = 30 kyu, 30 = 1 dan)
NOMINAL_LO, NOMINAL_HI = 18, 35         # 12 kyu .. 6 dan
PER_BIN = 150
PER_PLAYER = 25
SEED = 34

# 1a stop conditions (PLAN section 2)
STOP_ERR_RATE = 0.01
STOP_MIN_PLAYERS = 1500
STOP_MIN_NOMINAL = 1000

PREFILTER_W = (b'"width":19,', b'"width": 19,', b'"width":19}', b'"width": 19}')
PREFILTER_H = (b'"handicap":0,', b'"handicap": 0,', b'"handicap":0}', b'"handicap": 0}')

REASONS = ("prefilter", "json_error", "not_object", "size", "handicap", "ranked", "phase", "outcome_cancellation",
           "initial_state", "initial_player", "free_handicap_placement", "speed_correspondence", "start_time",
           "player_ids", "game_id", "professional", "rank_not_numeric", "bot_username", "moves_type", "bad_coord",
           "middle_pass", "too_few_moves")
LETTERS = "abcdefghijklmnopqrs"


class Reject(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _is_int(v):
    return type(v) is int


def _is_num(v):
    return (type(v) is int or type(v) is float) and math.isfinite(v)


def prefilter(line):
    """Cheap substring screen: a game that passes every filter must contain both a 19-wide and a 0-handicap key."""
    return any(p in line for p in PREFILTER_W) and any(p in line for p in PREFILTER_H)


def check_game(g, notes=None):
    """Apply PLAN section 2 to one decoded game. Returns (moves without trailing passes, speed); raises Reject.

    notes (a dict of counters, optional) collects accepted-but-noteworthy values (record only)."""
    if not isinstance(g, dict):
        raise Reject("not_object")
    if not (_is_int(g.get("width")) and _is_int(g.get("height")) and g["width"] == 19 and g["height"] == 19):
        raise Reject("size")
    if not (_is_int(g.get("handicap")) and g["handicap"] == 0):
        raise Reject("handicap")
    r = g.get("ranked")
    if not (r is True or (_is_int(r) and r == 1)):
        raise Reject("ranked")
    if g.get("phase") != "finished":
        raise Reject("phase")
    if g.get("outcome") == "Cancellation":
        raise Reject("outcome_cancellation")
    st = g.get("initial_state")
    if not (isinstance(st, dict) and st.get("black") == "" and st.get("white") == ""):
        raise Reject("initial_state")
    if g.get("initial_player") != "black":
        raise Reject("initial_player")
    f = g.get("free_handicap_placement")
    if not (f is False or (_is_int(f) and f == 0)):
        raise Reject("free_handicap_placement")
    tc = g.get("time_control")
    speed = tc.get("speed") if isinstance(tc, dict) else None
    if speed == "correspondence":
        raise Reject("speed_correspondence")
    t = g.get("start_time")
    if not (_is_int(t) and T_MIN <= t <= T_MAX):
        raise Reject("start_time")
    pl = g.get("players")
    if not isinstance(pl, dict):
        raise Reject("player_ids")
    b, w = pl.get("black"), pl.get("white")
    if not (isinstance(b, dict) and isinstance(w, dict) and _is_int(b.get("id")) and _is_int(w.get("id"))
            and b["id"] > 0 and w["id"] > 0 and b["id"] != w["id"]):
        raise Reject("player_ids")
    if not (_is_int(g.get("game_id")) and g["game_id"] > 0):
        raise Reject("game_id")
    for p in (b, w):
        pro = p.get("professional")
        if not (pro is False or pro is None or (_is_int(pro) and pro == 0)):
            raise Reject("professional")
    if not (_is_num(b.get("rank")) and _is_num(w.get("rank"))):
        raise Reject("rank_not_numeric")
    for p in (b, w):
        u = p.get("username")
        if isinstance(u, str) and "bot" in u.lower():
            raise Reject("bot_username")
    mv = g.get("moves")
    if not isinstance(mv, list):
        raise Reject("moves_type")
    xy = []
    for m in mv:
        if not (isinstance(m, list) and len(m) >= 2 and _is_int(m[0]) and _is_int(m[1])):
            raise Reject("bad_coord")
        x, y = m[0], m[1]
        if x == -1 and y == -1:
            xy.append(None)
        elif 0 <= x <= 18 and 0 <= y <= 18:
            xy.append((x, y))
        else:
            raise Reject("bad_coord")
    while xy and xy[-1] is None:
        xy.pop()
    if any(m is None for m in xy):
        raise Reject("middle_pass")
    if len(xy) < MIN_MOVES:
        raise Reject("too_few_moves")
    if notes is not None:
        for p in (b, w):
            if p.get("professional") is None:
                notes["professional_none_sides"] = notes.get("professional_none_sides", 0) + 1
            u = p.get("username")
            if not isinstance(u, str):
                notes["username_not_str_sides"] = notes.get("username_not_str_sides", 0) + 1
        if r is not True:
            notes["ranked_int_1"] = notes.get("ranked_int_1", 0) + 1
        if speed is None:
            notes["speed_missing"] = notes.get("speed_missing", 0) + 1
        if len(mv) != len(xy):
            notes["trailing_pass_games"] = notes.get("trailing_pass_games", 0) + 1
    return xy, speed


def to_sgf(xy):
    """Official v1.0 format: root SZ[19] only, alternating B/W from black, x = column, y = row, a = 0."""
    parts = ["(;SZ[19]"]
    for i, (x, y) in enumerate(xy):
        parts.append(";%s[%s%s]" % ("BW"[i & 1], LETTERS[x], LETTERS[y]))
    parts.append(")")
    return "".join(parts)


def game_rows(g, xy, speed):
    """Two rows per game: black side, then white side."""
    sgf = to_sgf(xy)
    out = []
    for c, key in (("B", "black"), ("W", "white")):
        p = g["players"][key]
        out.append({"player_id": "ogs_%d" % p["id"], "game_id": "ogs_%d" % g["game_id"],
                    "ogs_rank": float(p["rank"]), "color": c, "start_time": g["start_time"],
                    "speed": speed if isinstance(speed, str) else "", "n_moves": len(xy), "sgf": sgf})
    return out


def speed_code(s):
    if s in ("live", "blitz", "rapid"):
        return SPEEDS.index(s)
    return 3 if s in ("", None) else 4


class HashingReader(io.RawIOBase):
    """Wraps a binary file: counts and SHA-256-hashes the compressed bytes as gzip reads them."""

    def __init__(self, f):
        self.f = f
        self.h = hashlib.sha256()
        self.n = 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.f.read(len(b))
        k = len(data)
        b[:k] = data
        self.h.update(data)
        self.n += k
        return k


def _refuse(paths):
    for p in paths:
        if os.path.exists(p):
            print("refusing to overwrite %s" % p, file=sys.stderr)
            return True
    return False


def cmd_parse(a):
    os.makedirs(a.out_dir, exist_ok=True)
    out_rows = os.path.join(a.out_dir, "rows.csv.gz")
    out_idx = os.path.join(a.out_dir, "rows_index.bin")
    out_rec = os.path.join(a.out_dir, "parse_record.json")
    if _refuse([out_rows, out_idx, out_rec]):
        return 1
    t0 = time.time()
    counts = {r: 0 for r in REASONS}
    notes = {}
    n_lines = n_json = n_acc = n_rows = 0
    first_hash = last_hash = None
    players = set()
    idx_tmp = out_idx + ".part"
    raw = open(a.src, "rb")
    hr = HashingReader(raw)
    gz = gzip.GzipFile(fileobj=io.BufferedReader(hr, buffer_size=1 << 20), mode="rb")
    buf = []
    with open(out_rows + ".part", "wb") as fo_raw, open(idx_tmp, "wb") as fi:
        fo = io.TextIOWrapper(gzip.GzipFile(fileobj=fo_raw, mode="wb", compresslevel=3, mtime=0),
                              encoding="utf-8", newline="")
        wr = csv.writer(fo, lineterminator="\n")
        wr.writerow(COLUMNS)
        for line in gz:
            n_lines += 1
            if n_lines == 1:
                first_hash = hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()
            last_line = line
            if not prefilter(line):
                counts["prefilter"] += 1
                continue
            n_json += 1
            try:
                g = json.loads(line)
            except ValueError:
                counts["json_error"] += 1
                continue
            try:
                xy, speed = check_game(g, notes)
            except Reject as e:
                counts[e.reason] += 1
                continue
            n_acc += 1
            for r in game_rows(g, xy, speed):
                wr.writerow([r[c] if c != "ogs_rank" else repr(r[c]) for c in COLUMNS])
                pid = int(r["player_id"][4:])
                players.add(pid)
                buf.append((pid, g["game_id"], r["ogs_rank"], r["start_time"], n_rows, 0 if r["color"] == "B" else 1,
                            speed_code(speed), r["n_moves"]))
                n_rows += 1
            if len(buf) >= 200000:
                np.array(buf, dtype=IDX_DTYPE).tofile(fi)
                buf = []
            if a.progress and n_lines % a.progress == 0:
                print("lines %d accepted %d %.0fs" % (n_lines, n_acc, time.time() - t0), file=sys.stderr, flush=True)
        if buf:
            np.array(buf, dtype=IDX_DTYPE).tofile(fi)
        fo.close()
    # drain the gzip stream so the hash covers the whole file
    while gz.read(1 << 20):
        pass
    gz.close()
    rest = raw.read()
    hr.h.update(rest)
    hr.n += len(rest)
    raw.close()
    if n_lines:
        last_hash = hashlib.sha256(last_line.rstrip(b"\r\n")).hexdigest()
    os.replace(idx_tmp, out_idx)
    os.replace(out_rows + ".part", out_rows)
    elapsed = time.time() - t0
    errors = counts["json_error"]
    rec = {
        "format": PARSE_FORMAT,
        "src": os.path.basename(a.src),
        "src_bytes": os.path.getsize(a.src),
        "src_bytes_read": hr.n,
        "src_sha256": hr.h.hexdigest(),
        "first_line_sha256": first_hash,
        "last_line_sha256": last_hash,
        "lines": n_lines,
        "json_parsed_attempts": n_json,
        "json_errors": errors,
        "error_rate_of_json_attempts": errors / n_json if n_json else 0.0,
        "error_rate_of_lines": errors / n_lines if n_lines else 0.0,
        "rejected_first_reason": counts,
        "accepted_games": n_acc,
        "rows_written": n_rows,
        "players_seen": len(players),
        "accepted_notes": notes,
        "date_window_epoch": [T_MIN, T_MAX],
        "elapsed_s": round(elapsed, 1),
        "seconds_per_million_lines": round(elapsed / n_lines * 1e6, 1) if n_lines else None,
        "outputs": {"rows": os.path.basename(out_rows), "rows_sha256": _sha256_file(out_rows),
                    "index": os.path.basename(out_idx), "index_sha256": _sha256_file(out_idx),
                    "index_dtype": IDX_DTYPE.descr},
    }
    _write_json(out_rec, rec)
    print(json.dumps({k: rec[k] for k in ("lines", "accepted_games", "rows_written", "players_seen", "json_errors",
                                          "elapsed_s")}))
    return 0


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write_json(p, obj):
    with open(p + ".part", "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=False, ensure_ascii=True)
        f.write("\n")
    os.replace(p + ".part", p)


def median_sorted(s):
    n = len(s)
    if n % 2:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def window(ranks):
    """ranks: one player's ranks sorted old -> new. Returns (run length (capped at WIN_KEEP), accepted, rank)."""
    n = len(ranks)
    if n < WIN_MIN:
        return 0, False, None
    s = []
    k = 0
    for j in range(n - 1, -1, -1):
        r = float(ranks[j])
        cand = list(s)
        insort(cand, r)
        med = median_sorted(cand)
        if cand[-1] - med > WIN_TOL or med - cand[0] > WIN_TOL:
            break
        s = cand
        k += 1
        if k >= WIN_KEEP:
            break
    if k < WIN_MIN:
        return k, False, None
    return k, True, median_sorted(s)


def bin_of(rank):
    return min(max(int(math.floor(rank)), BIN_LO), BIN_HI)


def select_players(idx):
    """idx: structured rows (IDX_DTYPE). Returns (players list of dicts, per-player stats)."""
    order = np.lexsort((idx["gid"], idx["t"], idx["pid"]))
    pid = idx["pid"][order]
    cut = np.flatnonzero(np.diff(pid)) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(pid)]))
    accepted = []
    n_players = len(starts)
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s < WIN_MIN:
            continue
        rows = order[s:e]
        k, ok, rk = window(idx["rank"][rows])
        if ok:
            accepted.append({"pid": int(pid[s]), "rank": rk, "kept": rows[e - s - k:]})
    return accepted, n_players


def draw(accepted):
    """Bins ascending, one default_rng(SEED); players sorted by id, permuted, first PER_BIN; newest PER_PLAYER rows."""
    rng = np.random.default_rng(SEED)
    bins = {b: [] for b in range(BIN_LO, BIN_HI + 1)}
    for p in accepted:
        bins[bin_of(p["rank"])].append(p)
    chosen = []
    for b in range(BIN_LO, BIN_HI + 1):
        ps = sorted(bins[b], key=lambda p: p["pid"])
        perm = rng.permutation(len(ps))
        for i in perm[:PER_BIN].tolist():
            p = ps[i]
            chosen.append({"bin": b, "pid": p["pid"], "rank": p["rank"], "rows": p["kept"][-PER_PLAYER:]})
    return chosen, {b: len(v) for b, v in bins.items()}


def _quantiles(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return None
    q = np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n": int(a.size), "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]),
            "p90": float(q[4])}


def official_moves(path):
    csv.field_size_limit(1 << 24)
    nm = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            nm.append(row["sgf_content"].count(";") - 1)
    return _quantiles(nm)


def cmd_sample(a):
    os.makedirs(a.out_dir, exist_ok=True)
    out_games = os.path.join(a.out_dir, "sample_games.csv.gz")
    out_rec = os.path.join(a.out_dir, "sample_record.json")
    if _refuse([out_games, out_rec]):
        return 1
    t0 = time.time()
    with open(os.path.join(a.rows, "parse_record.json"), encoding="utf-8") as f:
        prec = json.load(f)
    mm = np.memmap(os.path.join(a.rows, "rows_index.bin"), dtype=IDX_DTYPE, mode="r")
    # a game listed twice in the dump: keep its first rows only
    _u, first = np.unique(mm["gid"] * 2 + mm["color"], return_index=True)
    del _u
    first.sort()
    n_dup = int(len(mm) - len(first))
    idx = np.array(mm[first]) if n_dup else np.array(mm)
    del mm, first
    accepted, n_players = select_players(idx)
    chosen, bin_players = draw(accepted)
    want = {}
    for c in chosen:
        for r in c["rows"].tolist():
            want[int(idx["row"][r])] = c
    # second pass: the SGF of the chosen rows
    csv.field_size_limit(1 << 24)
    got = []
    with gzip.open(os.path.join(a.rows, "rows.csv.gz"), "rt", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        assert tuple(head) == COLUMNS, head
        for i, row in enumerate(rd):
            c = want.get(i)
            if c is not None:
                got.append((c["bin"], c["pid"], int(row[4]), row[1], c["rank"], row))
    assert len(got) == len(want), (len(got), len(want))
    got.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    with open(out_games + ".part", "wb") as fo_raw:
        fo = io.TextIOWrapper(gzip.GzipFile(fileobj=fo_raw, mode="wb", compresslevel=6, mtime=0),
                              encoding="utf-8", newline="")
        wr = csv.writer(fo, lineterminator="\n")
        wr.writerow(COLUMNS + ("bin", "player_ogs_median"))
        for b, _pid, _t, _gid, rk, row in got:
            wr.writerow(row + [b, repr(rk)])
        fo.close()
    os.replace(out_games + ".part", out_games)

    def name(b):
        return "%dk" % (30 - b) if b < 30 else "%dd" % (b - 29)

    per_bin = {}
    for b in range(BIN_LO, BIN_HI + 1):
        rows = [g for g in got if g[0] == b]
        per_bin[name(b)] = {"bin": b, "players_available": bin_players[b],
                            "players_sampled": len({g[1] for g in rows}), "rows_sampled": len(rows)}
    rows_all = [g[5] for g in got]
    years = {}
    speeds = {}
    for r in rows_all:
        y = datetime.fromtimestamp(int(r[4]), TPE).year
        years[str(y)] = years.get(str(y), 0) + 1
        speeds[r[5] or "unknown"] = speeds.get(r[5] or "unknown", 0) + 1
    n_b = sum(1 for r in rows_all if r[3] == "B")
    acc_floor = np.array([math.floor(p["rank"]) for p in accepted], dtype=np.int64)
    n_nominal = int(np.sum((acc_floor >= NOMINAL_LO) & (acc_floor <= NOMINAL_HI)))
    err_rate = prec["error_rate_of_json_attempts"]
    reasons = []
    if err_rate > STOP_ERR_RATE:
        reasons.append("parse error rate %.4f > %.2f" % (err_rate, STOP_ERR_RATE))
    if len(accepted) < STOP_MIN_PLAYERS:
        reasons.append("accepted players %d < %d" % (len(accepted), STOP_MIN_PLAYERS))
    if n_nominal < STOP_MIN_NOMINAL:
        reasons.append("players 12k..6d nominal %d < %d" % (n_nominal, STOP_MIN_NOMINAL))
    rec = {
        "format": SAMPLE_FORMAT,
        "parse_record_lines": prec["lines"],
        "parse_rows_sha256": prec["outputs"]["rows_sha256"],
        "rows_in": int(len(idx)) + n_dup,
        "duplicate_rows_dropped": n_dup,
        "players_in": n_players,
        "players_with_ge_20_rows": int(sum(1 for _ in _counts_ge(idx["pid"], WIN_MIN))),
        "players_accepted_window": len(accepted),
        "players_below_15k_clipped": int(np.sum(acc_floor < BIN_LO)),
        "players_above_7d_clipped": int(np.sum(acc_floor > BIN_HI)),
        "players_nominal_12k_6d": n_nominal,
        "kept_rows_per_accepted_player": _quantiles([len(p["kept"]) for p in accepted]),
        "per_bin": per_bin,
        "players_sampled": len(chosen),
        "rows_sampled": len(got),
        "black_share": n_b / len(rows_all) if rows_all else None,
        "moves_sampled": _quantiles([int(r[6]) for r in rows_all]),
        "moves_official": official_moves(a.official_csv) if a.official_csv else None,
        "official_csv": os.path.basename(a.official_csv) if a.official_csv else None,
        "year": dict(sorted(years.items())),
        "speed": dict(sorted(speeds.items())),
        "stop": bool(reasons),
        "stop_reasons": reasons,
        "elapsed_s": round(time.time() - t0, 1),
        "outputs": {"sample_games": os.path.basename(out_games), "sha256": _sha256_file(out_games)},
    }
    _write_json(out_rec, rec)
    print(json.dumps({k: rec[k] for k in ("players_accepted_window", "players_nominal_12k_6d", "players_sampled",
                                          "rows_sampled", "stop", "stop_reasons")}))
    return 3 if reasons else 0


def _counts_ge(pid, k):
    _u, c = np.unique(pid, return_counts=True)
    return (x for x in c.tolist() if x >= k)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("--src", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--progress", type=int, default=0, help="print a progress line every N lines (0: never)")
    s = sub.add_parser("sample")
    s.add_argument("--rows", required=True, help="the directory written by parse")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--official-csv", default=None, help="record only: one official training CSV for move counts")
    a = ap.parse_args(argv)
    return cmd_parse(a) if a.cmd == "parse" else cmd_sample(a)


if __name__ == "__main__":
    sys.exit(main())
