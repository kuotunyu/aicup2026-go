"""prereg31 explore, stage 0a (PLAN.md 1.2, 1.3, section 2, rev 1): the HD table and its statistics.

Reads the official training data v1.1 (a downloaded archive: untrusted data). Standard library and numpy only, no
import of any repository module; run isolated from the repository root, every data path given as an argument:

    PY -I results/prereg31/explore/p31_hd.py build --v10 data/training --v11 data/training_v1.1/extracted/training
        --tarball data/training_v1.1/aicup2026_go_training_v1.1.tar.xz
        --out data/prereg31/hd_table.npz --record data/prereg31/hd_build.json
    PY -I results/prereg31/explore/p31_hd.py stats --table data/prereg31/hd_table.npz
        --build-record data/prereg31/hd_build.json --test-dir data/test_rootprops/original
        --out results/prereg31/explore/hd_stats.json

build  seam A row by row (v1.1 sgf_content with its HD[n] removed == v1.0 byte for byte; player_id, game_id, rank,
       color equal), row counts per file equal, (rank, game_id) unique, the tarball's SHA-256 and its members against
       the extracted files; fingerprint F per game. Writes the table only when everything holds; the record always.
stats  PLAN section 2 items 1-5 and the two non-naive-Bayes numbers of item 6; the Z / C1 decision of 1.3; the sanity
       stop and the distribution guard. Writes hd_stats.json with "stage0a": "PASS" or "STOP".

Both refuse to overwrite their outputs (run once). Exit 0: PASS; 3: a registered stop; 1: refusal.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import sys
import tarfile
import time
from fractions import Fraction

import numpy as np

TABLE_FORMAT = "c10-prereg31 hd table v1"
BUILD_FORMAT = "c10-prereg31 hd build v1"
STATS_FORMAT = "c10-prereg31 hd stats v1"
RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")      # weak -> strong (the repository's LEVELS)
COLOURS = ("B", "W")
TRAIN_HEADER = ["player_id", "game_id", "rank", "color", "sgf_content"]
TARBALL_SHA256 = "e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e"      # PLAN 1.1
TARBALL_BYTES = 220710452
RANK_TEST_FILES = {"public": "rank_prediction_test_public.csv", "private": "rank_prediction_test_private.csv"}
PLAYER_TEST_FILES = {"public": "player_identification_test_public.csv",
                     "private": "player_identification_test_private.csv"}
PLAYER_CAND_FILES = {"public": "player_identification_candidates_public.csv",
                     "private": "player_identification_candidates_private.csv"}
N_SLOTS = 20
SANITY_F2_MIN = Fraction(9, 10)          # PLAN 1.3
SANITY_F01_MAX = Fraction(1, 2)
GUARD_MARGIN = Fraction(3, 100)          # PLAN section 2 item 5
HD_RE = re.compile(r"HD\[([^\]]*)\]")
PLAIN_V10 = re.compile(r"\(;SZ\[19\](?:;[BW]\[[a-t]{0,2}\])*\)")
PLAIN_V11 = re.compile(r"\(;SZ\[19\]HD\[[0-9]\](?:;[BW]\[[a-t]{0,2}\])*\)")
MOVE_RE = re.compile(r";\s*[BW]\s*\[\s*([a-z]{0,2})\s*\]")
STAR = ("pd", "dp")
EXAMPLES = 10
ENC = "latin-1"                          # one byte <-> one character: equal strings are equal bytes


class Refuse(SystemExit):
    def __init__(self, msg):
        super().__init__(f"p31_hd: {msg}")


# ================================================================ one game
class HdError(ValueError):
    pass


def root_end(sgf):
    """Index just after the root node: the second ';' (the first one opens the root), or the end of the text."""
    a = sgf.find(";")
    if a < 0:
        raise HdError("no_root")
    b = sgf.find(";", a + 1)
    return len(sgf) if b < 0 else b


def parse_hd(sgf):
    """(n, sgf without its HD[n]). Exactly one HD property in the root node, value one or more digits, 0-9."""
    end = root_end(sgf)
    found = [m for m in HD_RE.finditer(sgf, 0, end)]
    if not found:
        raise HdError("hd_missing")
    if len(found) > 1:
        raise HdError("hd_multiple")
    m = found[0]
    if m.start() > 0 and sgf[m.start() - 1].isalpha():
        raise HdError("hd_identifier")           # e.g. XHD[1]: another property
    v = m.group(1)
    if not re.fullmatch(r"[0-9]+", v):
        raise HdError("hd_value")
    n = int(v)
    if n > 9:
        raise HdError("hd_range")
    return n, sgf[:m.start()] + sgf[m.end():]


def fingerprint(sgf):
    """F: no move of either colour at pd and none at dp in the whole game (PLAN 1.3)."""
    if (";B[pd]" in sgf or ";W[pd]" in sgf or ";B[dp]" in sgf or ";W[dp]" in sgf):
        return False
    if "pd" not in sgf and "dp" not in sgf:
        return True
    return not any(m.group(1) in STAR for m in MOVE_RE.finditer(sgf))     # unusual spacing only


def hd_group(n):
    return 0 if n == 0 else (1 if n == 1 else 2)


# ================================================================ small helpers
def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def frac(a, b):
    return None if not b else Fraction(int(a), int(b))


def fl(x):
    return None if x is None else float(x)


def csv_rows(path):
    """Streaming rows of one CSV (byte-preserving decoding; a UTF-8 BOM before the header is dropped)."""
    csv.field_size_limit(1 << 26)
    with open(path, encoding=ENC, newline="") as f:
        first = True
        for row in csv.reader(f):
            if first:
                first = False
                if row and row[0].startswith("\xef\xbb\xbf"):
                    row = [row[0][3:]] + row[1:]
            yield row


def tool_record():
    here = os.path.abspath(__file__)
    return {"file": "results/prereg31/explore/p31_hd.py", "sha256_lf": sha_lf(here),
            "python": sys.version.split()[0], "numpy": np.__version__, "isolated": bool(sys.flags.isolated)}


# ================================================================ build: seam A over one pair of files
PROBLEMS = ("width", "player_id", "game_id", "rank", "color", "hd_parse", "sgf")


def compare_row(r10, r11, rank):
    """(problems, hd, F, colour index). problems: the names of PROBLEMS that this row fails (empty: the row is
    seam-A equal). hd / F / colour are None when they cannot be read."""
    if len(r10) != 5 or len(r11) != 5:
        return ["width"], None, None, None
    probs = []
    if r10[0] != r11[0]:
        probs.append("player_id")
    if r10[1] != r11[1]:
        probs.append("game_id")
    if r10[2] != r11[2] or r10[2] != rank:
        probs.append("rank")
    if r10[3] != r11[3] or r10[3] not in COLOURS:
        probs.append("color")
    try:
        hd, stripped = parse_hd(r11[4])
    except HdError:
        return probs + ["hd_parse"], None, None, None
    if stripped != r10[4]:
        probs.append("sgf")
    col = COLOURS.index(r10[3]) if r10[3] in COLOURS else None
    return probs, hd, fingerprint(r10[4]), col


def scan_pair(path10, path11, rank):
    """Seam A over one rank file. Returns (record, arrays or None). arrays only when every row is equal, the row
    counts are equal and game_id is unique in the file."""
    rec = {"rank": rank, "rows_v10": 0, "rows_v11": 0, "header_ok": None, "problems": {k: 0 for k in PROBLEMS},
           "rows_equal": 0, "rows_not_equal": 0, "duplicate_game_ids": 0, "plain_form_v10": 0, "plain_form_v11": 0,
           "examples": []}
    it10, it11 = csv_rows(path10), csv_rows(path11)
    h10, h11 = next(it10, None), next(it11, None)
    rec["header_ok"] = bool(h10 == TRAIN_HEADER and h11 == TRAIN_HEADER)
    seen = set()
    pid, gid, col, hd, fp = [], [], [], [], []
    for i, (r10, r11) in enumerate(itertools.zip_longest(it10, it11), start=1):
        if r10 is not None:
            rec["rows_v10"] += 1
        if r11 is not None:
            rec["rows_v11"] += 1
        if r10 is None or r11 is None:
            continue
        probs, n, f, c = compare_row(r10, r11, rank)
        if len(r10) == 5:
            rec["plain_form_v10"] += PLAIN_V10.fullmatch(r10[4]) is not None
            if r10[1] in seen:
                rec["duplicate_game_ids"] += 1
            seen.add(r10[1])
        if len(r11) == 5:
            rec["plain_form_v11"] += PLAIN_V11.fullmatch(r11[4]) is not None
        if probs:
            rec["rows_not_equal"] += 1
            for k in probs:
                rec["problems"][k] += 1
            if len(rec["examples"]) < EXAMPLES:
                rec["examples"].append({"row": i, "problems": probs})
            continue
        rec["rows_equal"] += 1
        pid.append(r10[0])
        gid.append(r10[1])
        col.append(c)
        hd.append(n)
        fp.append(f)
    rec["rows_ok"] = bool(rec["rows_v10"] == rec["rows_v11"] and rec["rows_v10"] > 0)
    rec["ok"] = bool(rec["header_ok"] and rec["rows_ok"] and rec["rows_not_equal"] == 0
                     and rec["duplicate_game_ids"] == 0)
    if not rec["ok"]:
        return rec, None
    return rec, {"player_id": np.array(pid, dtype="S"), "game_id": np.array(gid, dtype="S"),
                 "color": np.array(col, dtype=np.uint8), "hd": np.array(hd, dtype=np.int8),
                 "F": np.array(fp, dtype=np.bool_)}


def tar_members(tarball):
    """({base name: sha256}, [names seen twice]) of the regular members named train_<rank>.csv, read as a stream
    (nothing is extracted)."""
    want = {f"train_{r}.csv" for r in RANKS}
    out, twice = {}, []
    with tarfile.open(tarball, "r:xz") as tf:
        for m in tf:
            base = m.name.replace("\\", "/").rsplit("/", 1)[-1]
            if not m.isfile() or base not in want:
                continue
            h = hashlib.sha256()
            f = tf.extractfile(m)
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
            if base in out:
                twice.append(base)
            out[base] = h.hexdigest()
    return out, twice


def save_table(path, parts):
    """parts: [(rank index, arrays)] in RANKS order."""
    wid_g = max(a["game_id"].dtype.itemsize for _r, a in parts)
    wid_p = max(a["player_id"].dtype.itemsize for _r, a in parts)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, format=np.array(TABLE_FORMAT), ranks=np.array(RANKS),
             rank=np.concatenate([np.full(len(a["hd"]), r, dtype=np.uint8) for r, a in parts]),
             game_id=np.concatenate([a["game_id"].astype(f"S{wid_g}") for _r, a in parts]),
             player_id=np.concatenate([a["player_id"].astype(f"S{wid_p}") for _r, a in parts]),
             color=np.concatenate([a["color"] for _r, a in parts]),
             hd=np.concatenate([a["hd"] for _r, a in parts]),
             F=np.concatenate([a["F"] for _r, a in parts]))
    os.replace(tmp, path)


def load_table(path):
    with np.load(path, allow_pickle=False) as z:
        if str(z["format"]) != TABLE_FORMAT:
            raise Refuse(f"{path}: not a {TABLE_FORMAT!r} file")
        t = {k: z[k] for k in ("ranks", "rank", "game_id", "player_id", "color", "hd", "F")}
    if tuple(str(x) for x in t["ranks"]) != RANKS:
        raise Refuse(f"{path}: another rank order")
    n = len(t["hd"])
    if any(len(t[k]) != n for k in ("rank", "game_id", "player_id", "color", "F")):
        raise Refuse(f"{path}: columns of different lengths")
    return t


def build(v10, v11, tarball, out, log=print):
    """The build record; writes the table when (and only when) every check holds."""
    rec = {"format": BUILD_FORMAT, "created": now(), "tool": tool_record(),
           "args": {"v10": v10, "v11": v11, "tarball": tarball, "out": out}}
    stop = []
    tb = {"bytes": os.path.getsize(tarball), "sha256": sha_file(tarball), "expected_sha256": TARBALL_SHA256,
          "expected_bytes": TARBALL_BYTES}
    tb["ok"] = bool(tb["sha256"] == TARBALL_SHA256 and tb["bytes"] == TARBALL_BYTES)
    rec["tarball"] = tb
    log(f"tarball sha ok={tb['ok']}")
    members, twice = tar_members(tarball)
    files, parts = {}, []
    for r, rank in enumerate(RANKS):
        name = f"train_{rank}.csv"
        p10, p11 = os.path.join(v10, name), os.path.join(v11, name)
        fr, arrays = scan_pair(p10, p11, rank)
        fr["v10"] = {"bytes": os.path.getsize(p10), "sha256": sha_file(p10)}
        fr["v11"] = {"bytes": os.path.getsize(p11), "sha256": sha_file(p11), "tar_member_sha256": members.get(name)}
        fr["v11"]["matches_tarball"] = bool(members.get(name) == fr["v11"]["sha256"] and name not in twice)
        files[rank] = fr
        if arrays is not None:
            parts.append((r, arrays))
        log(f"{name}: rows {fr['rows_v10']}/{fr['rows_v11']} equal {fr['rows_equal']} not_equal "
            f"{fr['rows_not_equal']} dup {fr['duplicate_game_ids']} tar {fr['v11']['matches_tarball']}")
    rec["files"] = files
    tot = {k: sum(f[k] for f in files.values())
           for k in ("rows_v10", "rows_v11", "rows_equal", "rows_not_equal", "duplicate_game_ids",
                     "plain_form_v10", "plain_form_v11")}
    tot["problems"] = {k: sum(f["problems"][k] for f in files.values()) for k in PROBLEMS}
    rec["totals"] = tot
    checks = {"tarball_sha": tb["ok"],
              "extracted_matches_tarball": all(f["v11"]["matches_tarball"] for f in files.values()),
              "headers": all(f["header_ok"] for f in files.values()),
              "row_counts_equal": all(f["rows_ok"] for f in files.values()),
              "rows_all_equal": tot["rows_not_equal"] == 0,
              "keys_unique": tot["duplicate_game_ids"] == 0}
    rec["seam_a"] = {"checks": checks, "ok": bool(all(checks.values()))}
    stop = [k for k, v in checks.items() if not v]
    rec["table"] = None
    if not stop and len(parts) == len(RANKS):
        save_table(out, parts)
        rec["table"] = {"path": out, "sha256": sha_file(out), "bytes": os.path.getsize(out),
                        "rows": int(sum(len(a["hd"]) for _r, a in parts)), "format": TABLE_FORMAT}
    rec["status"] = "PASS" if not stop else "STOP"
    rec["stop_reasons"] = stop
    return rec


def cmd_build(a):
    for p in (a.out, a.record):
        if os.path.exists(p):
            raise Refuse(f"{p} exists: build runs once")
    rec = build(a.v10, a.v11, a.tarball, a.out)
    write_json(a.record, rec)
    print(f"build: {rec['status']} {rec['stop_reasons']} rows equal {rec['totals']['rows_equal']}")
    return 0 if rec["status"] == "PASS" else 3


# ================================================================ stats: pure parts
def table_counts(t):
    """N[rank][colour][HD group 0 / 1 / >=2] and the same for games with fingerprint F."""
    g = np.where(t["hd"] == 0, 0, np.where(t["hd"] == 1, 1, 2))
    N = np.zeros((len(RANKS), 2, 3), dtype=np.int64)
    NF = np.zeros((len(RANKS), 2, 3), dtype=np.int64)
    np.add.at(N, (t["rank"], t["color"], g), 1)
    np.add.at(NF, (t["rank"], t["color"], g), t["F"].astype(np.int64))
    return N, NF


def flag_count(n3, encoding):
    """Games whose flag is 1 among the HD-group counts n3 = [HD 0, HD 1, HD >= 2] (PLAN 1.3)."""
    if encoding == "Z":
        return int(n3[1])
    if encoding == "C1":
        return int(n3[1]) + int(n3[2])
    raise ValueError(encoding)


def flag_of(hd, encoding):
    """The flag of one game: Z = [HD == 1], C1 = [HD >= 1]."""
    if encoding == "Z":
        return int(hd == 1)
    if encoding == "C1":
        return int(hd >= 1)
    raise ValueError(encoding)


def fingerprint_decision(n_train, nf_train, n_test, nf_test):
    """PLAN 1.3. n_train / nf_train: games and F games of HD 0 / HD 1 / HD >= 2 in training; n_test / nf_test: of
    HD 0 / HD 1 in the competition-2 test games. Exact rational arithmetic; floats only in the record."""
    tot = sum(int(x) for x in n_train)
    f = [frac(nf_train[i], n_train[i]) for i in range(3)]
    pi = [frac(n_train[i], tot) for i in range(3)]
    g = [frac(nf_test[i], n_test[i]) for i in range(2)]
    sanity = bool(None not in f and f[2] >= SANITY_F2_MIN and f[0] <= SANITY_F01_MAX and f[1] <= SANITY_F01_MAX)
    s1 = D = s0 = d0 = d1 = decision = None
    if None not in f and None not in pi and None not in g:
        if pi[1] + pi[2] > 0:
            s1 = pi[2] * (f[2] - f[1]) / (pi[1] + pi[2])
        if pi[0] + pi[2] > 0:
            s0 = pi[2] * (f[2] - f[0]) / (pi[0] + pi[2])
        D = (g[1] - g[0]) - (f[1] - f[0])
        d1, d0 = g[1] - f[1], g[0] - f[0]
        if s1 is not None:
            decision = "C1" if D >= s1 / 2 else "Z"
    return {"train_games": [int(x) for x in n_train], "train_F": [int(x) for x in nf_train],
            "test_games": [int(x) for x in n_test], "test_F": [int(x) for x in nf_test],
            "f0": fl(f[0]), "f1": fl(f[1]), "f2": fl(f[2]), "pi0": fl(pi[0]), "pi1": fl(pi[1]), "pi2": fl(pi[2]),
            "g0": fl(g[0]), "g1": fl(g[1]),
            "sanity": {"rule": "f2 >= 0.90 and f0 <= 0.50 and f1 <= 0.50", "ok": sanity},
            "s1": fl(s1), "D": fl(D), "half_s1": fl(None if s1 is None else s1 / 2),
            "rule": "D >= s1 / 2 -> C1, else Z", "decision": decision,
            "record_only": {"d1": fl(d1), "d0": fl(d0), "s0": fl(s0)}}


def guard(N, test_n, encoding):
    """PLAN section 2 item 5. N[rank][colour][group]: training; test_n[colour][group]: the competition-2 test games
    (both pools). p(r, c) = flag share in training; t(c) = HD[1] share of the test games of question colour c.
    Passes when min_r p - 0.03 <= t <= max_r p + 0.03 for both colours."""
    out = {"encoding": encoding, "margin": fl(GUARD_MARGIN), "p": {}, "t": {}, "bounds": {}}
    ok_all = True
    ps = {}
    for c, col in enumerate(COLOURS):
        p = [frac(flag_count(N[r][c], encoding), sum(int(x) for x in N[r][c])) for r in range(len(RANKS))]
        t = frac(test_n[c][1], sum(int(x) for x in test_n[c]))
        ps[col] = p
        out["t"][col] = fl(t)
        if t is None or None in p:
            out["bounds"][col] = {"ok": False}
            ok_all = False
            continue
        lo, hi = min(p) - GUARD_MARGIN, max(p) + GUARD_MARGIN
        ok = bool(lo <= t <= hi)
        ok_all = ok_all and ok
        out["bounds"][col] = {"min_p": fl(min(p)), "max_p": fl(max(p)), "lo": fl(lo), "hi": fl(hi), "ok": ok}
    for r, rank in enumerate(RANKS):
        out["p"][rank] = {col: fl(ps[col][r]) for col in COLOURS}
    out["ok"] = bool(ok_all)
    rec = {"mean_p": {}, "adjacent_pairs": [f"{RANKS[r + 1]}-{RANKS[r]}" for r in range(len(RANKS) - 1)],
           "adjacent_diff": {}}
    for col in COLOURS:
        p = ps[col]
        if None in p:
            rec["mean_p"][col] = None
            rec["adjacent_diff"][col] = None
            continue
        rec["mean_p"][col] = fl(sum(p) / len(p))
        rec["adjacent_diff"][col] = [fl(p[r + 1] - p[r]) for r in range(len(p) - 1)]
    return out, rec


def shares(n3):
    n3 = [int(x) for x in n3]
    n = sum(n3)
    return {"hd0": n3[0], "hd1": n3[1], "hd_ge2": n3[2], "n": n,
            "share_hd0": fl(frac(n3[0], n)), "share_hd1": fl(frac(n3[1], n)), "share_hd_ge2": fl(frac(n3[2], n)),
            "flag_share_Z": fl(frac(flag_count(n3, "Z"), n)), "flag_share_C1": fl(frac(flag_count(n3, "C1"), n))}


def train_tables(N, NF, hd):
    out = {"ranks": list(RANKS), "by_rank_colour": {}, "by_rank": {}, "by_colour": {}, "fingerprint_by_rank": {}}
    for r, rank in enumerate(RANKS):
        out["by_rank_colour"][rank] = {col: shares(N[r][c]) for c, col in enumerate(COLOURS)}
        out["by_rank"][rank] = shares(N[r].sum(axis=0))
        n, nf = N[r].sum(axis=0), NF[r].sum(axis=0)
        out["fingerprint_by_rank"][rank] = {"f0": fl(frac(nf[0], n[0])), "f1": fl(frac(nf[1], n[1]))}
    for c, col in enumerate(COLOURS):
        out["by_colour"][col] = shares(N[:, c, :].sum(axis=0))
    out["overall"] = shares(N.sum(axis=(0, 1)))
    vals, cnt = np.unique(hd, return_counts=True)
    out["hd_values"] = {str(int(v)): int(k) for v, k in zip(vals, cnt)}
    return out


# ================================================================ stats: the official test files
def question_games(path):
    """(colour, sgf) of every game of a question file (slots 1..num_games), or (None, None) for a game that cannot
    be read (bad num_games, empty SGF, colour not B / W). Raises Refuse when the header is not the official one."""
    it = csv_rows(path)
    header = next(it, None)
    want = ["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, N_SLOTS + 1)] \
        + [f"color_{i}" for i in range(1, N_SLOTS + 1)]
    if header != want:
        raise Refuse(f"{path}: another header")
    for row in it:
        if not row:
            continue
        if len(row) != len(want) or not re.fullmatch(r"[0-9]+", row[1]) or not 1 <= int(row[1]) <= N_SLOTS:
            yield None, None
            continue
        for i in range(int(row[1])):
            sgf, col = row[2 + i], row[2 + N_SLOTS + i]
            yield (col, sgf) if (sgf and col in COLOURS) else (None, None)


def candidate_games(path):
    it = csv_rows(path)
    header = next(it, None)
    if header != ["game_id", "player_id", "sgf_content", "color"]:
        raise Refuse(f"{path}: another header")
    for row in it:
        if not row:
            continue
        yield (row[3], row[2]) if (len(row) == 4 and row[2] and row[3] in COLOURS) else (None, None)


def tally_games(games):
    """n[colour][HD group], nF[colour][HD group], HD value counts, games, unreadable games."""
    n = [[0, 0, 0], [0, 0, 0]]
    nf = [[0, 0, 0], [0, 0, 0]]
    vals, total, bad = {}, 0, 0
    for col, sgf in games:
        total += 1
        if col is None:
            bad += 1
            continue
        try:
            hd, _stripped = parse_hd(sgf)
        except HdError:
            bad += 1
            continue
        c, g = COLOURS.index(col), hd_group(hd)
        n[c][g] += 1
        nf[c][g] += fingerprint(sgf)
        vals[str(hd)] = vals.get(str(hd), 0) + 1
    return {"games": total, "unreadable": bad, "n": n, "nF": nf, "hd_values": dict(sorted(vals.items()))}


def add_tally(a, b):
    return {"games": a["games"] + b["games"], "unreadable": a["unreadable"] + b["unreadable"],
            "n": [[a["n"][c][g] + b["n"][c][g] for g in range(3)] for c in range(2)],
            "nF": [[a["nF"][c][g] + b["nF"][c][g] for g in range(3)] for c in range(2)],
            "hd_values": {k: a["hd_values"].get(k, 0) + b["hd_values"].get(k, 0)
                          for k in sorted(set(a["hd_values"]) | set(b["hd_values"]))}}


def test_shares(t):
    """HD[1] share by question colour and overall (denominator: every readable game of that colour)."""
    out = dict(t)
    for c, col in enumerate(COLOURS):
        out[f"t_{col}"] = fl(frac(t["n"][c][1], sum(t["n"][c])))
    out["t_all"] = fl(frac(t["n"][0][1] + t["n"][1][1], sum(t["n"][0]) + sum(t["n"][1])))
    return out


def stats(table_path, build_rec, test_dir, log=print):
    out = {"format": STATS_FORMAT, "created": now(), "tool": tool_record(),
           "registration": "results/prereg31/explore/PLAN.md (1.2, 1.3, section 2, rev 1)",
           "args": {"table": table_path, "test_dir": test_dir}, "build": build_rec}
    stop = []
    sa = build_rec.get("seam_a", {})
    out["seam_a"] = {"ok": bool(sa.get("ok")), "checks": sa.get("checks"), "totals": build_rec.get("totals"),
                     "rows_by_file": {k: [v["rows_v10"], v["rows_v11"]]
                                      for k, v in build_rec.get("files", {}).items()}}
    if build_rec.get("format") != BUILD_FORMAT or build_rec.get("status") != "PASS" or not sa.get("ok") \
            or not build_rec.get("table"):
        stop.append("seam_a")
        out.update({"encoding": None, "sanity_ok": None, "guard_ok": None, "stage0a": "STOP", "stop_reasons": stop})
        return out
    sha = sha_file(table_path)
    out["table"] = {"sha256": sha, "sha256_build_record": build_rec["table"]["sha256"],
                    "sha_matches_build_record": bool(sha == build_rec["table"]["sha256"])}
    t = load_table(table_path)
    keys = set(zip(t["rank"].tolist(), t["game_id"].tolist()))
    out["table"].update({"rows": int(len(t["hd"])), "unique_keys": len(keys),
                         "rows_match_build_record": bool(len(t["hd"]) == build_rec["table"]["rows"])})
    if not out["table"]["sha_matches_build_record"] or not out["table"]["rows_match_build_record"]:
        stop.append("table_sha")
    if len(keys) != len(t["hd"]):
        stop.append("keys_unique")
    N, NF = table_counts(t)
    out["train"] = train_tables(N, NF, t["hd"])
    # the official test files
    src = read_json(os.path.join(test_dir, "SOURCE.json"))["files"]
    files, tall = {}, {}
    for kind, names, reader in (("rank", RANK_TEST_FILES, question_games), ("player_test", PLAYER_TEST_FILES,
                                question_games), ("player_candidates", PLAYER_CAND_FILES, candidate_games)):
        for pool, name in names.items():
            p = os.path.join(test_dir, name)
            got = sha_file(p)
            files[name] = {"sha256": got, "source_json_sha256": src.get(name, {}).get("sha256"),
                           "ok": bool(got == src.get(name, {}).get("sha256"))}
            tall[(kind, pool)] = tally_games(reader(p))
            log(f"{name}: games {tall[(kind, pool)]['games']} unreadable {tall[(kind, pool)]['unreadable']}")
    out["test_files"] = files
    if not all(f["ok"] for f in files.values()):
        stop.append("test_sha")
    rank_all = add_tally(tall[("rank", "public")], tall[("rank", "private")])
    out["test_rank"] = {"public": test_shares(tall[("rank", "public")]),
                        "private": test_shares(tall[("rank", "private")]), "combined": test_shares(rank_all)}
    out["test_player_record_only"] = {f"{kind}_{pool}": test_shares(tall[(kind, pool)])
                                      for kind in ("player_test", "player_candidates")
                                      for pool in ("public", "private")}
    if rank_all["unreadable"]:
        stop.append("test_unreadable")
    # 1.3: fingerprint and the Z / C1 decision
    n_tr, nf_tr = N.sum(axis=(0, 1)), NF.sum(axis=(0, 1))
    n_te = [rank_all["n"][0][g] + rank_all["n"][1][g] for g in range(2)]
    nf_te = [rank_all["nF"][0][g] + rank_all["nF"][1][g] for g in range(2)]
    fp = fingerprint_decision(n_tr, nf_tr, n_te, nf_te)
    fp["test_hd_ge2_games"] = rank_all["n"][0][2] + rank_all["n"][1][2]
    out["fingerprint"] = fp
    out["encoding"] = fp["decision"]
    out["sanity_ok"] = bool(fp["sanity"]["ok"])
    if not out["sanity_ok"] or fp["decision"] is None:
        stop.append("fingerprint_sanity")
    # section 2 item 5: the distribution guard, under the decided encoding
    if fp["decision"] is None:
        out["guard"], out["guard_ok"], out["record_only"] = None, None, None
    else:
        g, rec = guard(N, rank_all["n"], fp["decision"])
        out["guard"], out["guard_ok"], out["record_only"] = g, bool(g["ok"]), rec
        if not g["ok"]:
            stop.append("distribution_guard")
    out["stage0a"] = "PASS" if not stop else "STOP"
    out["stop_reasons"] = stop
    return out


def cmd_stats(a):
    if os.path.exists(a.out):
        raise Refuse(f"{a.out} exists: stats runs once")
    out = stats(a.table, read_json(a.build_record), a.test_dir)
    write_json(a.out, out)
    print(f"stats: stage0a {out['stage0a']} {out['stop_reasons']} encoding {out['encoding']} "
          f"sanity {out['sanity_ok']} guard {out['guard_ok']}")
    return 0 if out["stage0a"] == "PASS" else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--v10", required=True)
    b.add_argument("--v11", required=True)
    b.add_argument("--tarball", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--record", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--table", required=True)
    s.add_argument("--build-record", required=True)
    s.add_argument("--test-dir", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_build(a) if a.cmd == "build" else cmd_stats(a)


if __name__ == "__main__":
    sys.exit(main())
