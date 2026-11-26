"""C10 PREREG_14: the competition-2 exact-copy layer (docs/PREREG_14_RANK_EXACT_COPY.md). CPU only, no GPU.

A competition-2 test game that is an exact copy (PREREG_14 section 2) of an official training game -- the whole move
sequence equal up to the 8 board symmetries (the T2 full key of tools/t02_checks.py), the same move count, at least
100 moves, black first and strictly alternating, and the training row on the SAME side (its colour equals the
question's colour) -- names the training player who played that side. One player has one level in the official
training data, so a question whose copies all point to one and the same training player P takes P's level
(section 4); every other question keeps the base file's level. The base file is the competition-2 upload that
official_log.py final already designates (results/official_scores.json final_choice.rank, an uploaded entry).

    PY tools/t14_copy.py pins   [--rerun --reason TEXT]        -> results/test14/design/index_pins.json
    PY tools/t14_copy.py sim    [--rerun --reason TEXT]        -> results/test14/design/sim.json (fit players only)
    PY tools/t14_copy.py scan   [--rerun --reason TEXT]        -> results/test14/scan/scan.json (record only)
    PY tools/t14_copy.py apply  [--rerun --reason TEXT]        -> results/test14/rank/{submission_rank.csv,
                                                                  overrides.csv, t14_copy_report.json} or
                                                                  results/test14/not_used.json
    PY tools/t14_copy.py status
    PY tools/t14_copy.py fake   --root S [--seed-file-tag TAG] (non-real: an official-shaped fake tree, section 6)
    scan / apply / status also take  --dry-run --root S [--index-part fit|train] [--train-dir D] [--exclude FILE]
                                     [--split-csv F] [--pins F]

Modes
    real (default)  data/test (MANIFEST kind official, the rank question files equal to it), the raw T1 of
                    results/test02/checks/rank/t1.json (committed, complete, stage raw, mode real, PASS or
                    NORMALIZED), the index of ALL official training rows (section 3: 1,000,000 rows, 8,007 players)
                    whose pins must equal the committed results/test14/design/index_pins.json, the base from the
                    committed results/official_scores.json. This tool, t02_checks.py, check_submission.py and
                    c10data.py must be committed and clean. Outputs only under results/test14/.
    --dry-run       any tree S (default S/data/test, S/results/official_scores.json, S/results/test14). The index
                    never holds split-v1 'val' players (part fit, default, or train), and --exclude FILE (JSON list
                    of [player_id, game_id]) removes held-out games (the simulation). Writes only under S, and S
                    must be outside the repository.

Exit codes: 0 done (apply: 14 written; scan / pins / sim: written), 2 refusal (a precondition; a traceback is 2 too),
3 the registered 'not used' (no override, the 14 file would equal the base, T1 not PASS / NORMALIZED, an earlier
not_used.json), 4 local drift (index pins differ, a player with two levels, check_submission failed, the base file
differs from its log entry).

Never opens data/local, never imports tools/score.py, decide.py, a competition-1 tool or a model tool; reads the
training files only by their explicit names data/training/train_<LEVEL>.csv. Training player ids are the
anonymised ids of the official data; nothing here links them to anything outside it (PREREG_14 5).
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c10data  # noqa: E402  (LEVELS, the split-v1 player sets)
import check_submission  # noqa: E402
import t02_checks as t2  # noqa: E402  (T2 game keys: MOVE_RE, canon_points, game_key; the question-file reader)

# ---------------------------------------------------------------- registered (PREREG_14)
MIN_MOVES = 100                       # section 2.4
LEVELS = list(c10data.LEVELS)
TRAIN_COLUMNS = list(c10data.COLUMNS)
FORMAT_REPORT = "c10-test14 copy apply v1"
FORMAT_SCAN = "c10-test14 copy scan v1"
FORMAT_PINS = "c10-test14 index pins v1"
FORMAT_SIM = "c10-test14 sim v1"
FORMAT_NOT_USED = "c10-test14 not used v1"
METHOD = "prereg14 exact-copy layer (PREREG_14 4) on the designated competition-2 file"
TW = timezone(timedelta(hours=8))
K5_14 = datetime(2026, 11, 4, 11, 0, 0, tzinfo=TW)        # PREREG_14 8: rev 2, pins and sim merged before this
K6_14 = datetime(2026, 11, 23, 18, 0, 0, tzinfo=TW)       # PREREG_14 8: the 14 report's last commit
RANK_FILES = tuple(name for _pool, _kind, name in t2.FILES["rank"])   # public, private
POOLS = ("public", "private")
BASE_DIRS = ("results/test02/rank", "results/test04/rank", "results/test07/rank", "results/test08/rank",
             "results/test09/rank")
BASE_DIRS = BASE_DIRS + ("results/test19/rank",)   # PREREG_19 4.1 / 4.3: a 19 file can be the base
SIM_SALT_GAME = "c10-p14|sim"
SIM_SALT_PLAYER = "c10-p14|player"
SIM_SALT_KIND = "c10-p14|kind"
SIM_SALT_BASE = "c10-p14|base"
SIM_KINDS = (("N", 200), ("C1", 50), ("CH", 50), ("CA", 50), ("XS", 13), ("XO", 13), ("XT", 12), ("XC", 12))
SIM_EXPECT_OVERRIDE = ("C1", "CH", "CA", "XS")
SIM_SIZES = (5, 10, 15, 20)
SIM_MIN_COLOUR_GAMES = 40

# ---------------------------------------------------------------- paths
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
DATA_TEST = os.path.join(REPO, "data", "test")
LOG = os.path.join(REPO, "results", "official_scores.json")
OUT = os.path.join(REPO, "results", "test14")
T1_JSON = os.path.join(REPO, "results", "test02", "checks", "rank", "t1.json")
PINS = os.path.join(OUT, "design", "index_pins.json")
SIM = os.path.join(OUT, "design", "sim.json")
OWN = ("tools/t14_copy.py", "tools/t02_checks.py", "tools/check_submission.py", "tools/c10data.py")
_FORBIDDEN = "ans" + "wers"
_LOCAL = os.path.join(REPO, "data", "local")


class Refuse(Exception):
    code = 2


class NotUsed(Exception):
    code = 3


class Drift(Exception):
    code = 4


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- small helpers
def _real(p):
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def inside(path, root):
    a, b = _real(path), _real(root)
    return a == b or a.startswith(b.rstrip("\\/") + os.sep)


def guard_read(path):
    """Never data/local, never a file whose name holds the label-file word."""
    if inside(path, _LOCAL) or _FORBIDDEN in os.path.basename(str(path)).lower():
        raise Refuse(f"refusing to open {path!r}")
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(guard_read(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lf(path):
    with open(guard_read(path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(guard_read(path), encoding="utf-8") as f:
        return json.load(f)


def git(*args):
    r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout.strip()


def committed_clean(rel):
    rc_t, _ = git("ls-files", "--error-unmatch", "--", rel)
    rc_s, st = git("status", "--porcelain", "--", rel)
    return rc_t == 0 and rc_s == 0 and not st


def git_head():
    rc, h = git("rev-parse", "HEAD")
    return h if rc == 0 else None


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path):
    try:
        return os.path.relpath(os.path.realpath(path), os.path.realpath(REPO)).replace(os.sep, "/")
    except ValueError:
        return path


def provenance(mode, extra=None):
    d = {"tool": "tools/t14_copy.py", "tool_sha256_lf": sha256_lf(os.path.join(HERE, "t14_copy.py")),
         "t02_checks_sha256_lf": sha256_lf(os.path.join(HERE, "t02_checks.py")), "mode": mode,
         "git_head": git_head(), "python": platform.python_version(), "created_utc": now_iso()}
    d.update(extra or {})
    return d


# ---------------------------------------------------------------- moves and keys (PREREG_14 section 2)
def parse_game(sgf):
    """(points, colours) of every MOVE_RE match (the T2 parse, no move limit)."""
    full = t2.MOVE_RE.findall(sgf or "")
    return [p for _, p in full], "".join(c for c, _ in full)


def alternating_from_black(cols):
    return cols == ("BW" * (len(cols) // 2 + 1))[:len(cols)] and cols[:1] == "B"


def full_key(pts):
    """The T2 full key: SHA-1[:12] of the D4-minimal whole-game point string (t02_checks.game_key)."""
    return t2.game_key(pts)


# ---------------------------------------------------------------- the training index (PREREG_14 section 3)
def players_of_part(part, split_csv=SPLIT_CSV):
    """None for 'all' (every row); else the split-v1 player set (never 'val' players). Another player table (tests)
    never brings back a player that results/split_v1/players_split.csv marks 'val'."""
    if part == "all":
        return None
    if part not in ("train", "fit"):
        raise Refuse(f"unknown index part {part!r}")
    keep = c10data.players_for(c10data.read_split(split_csv), part)
    if os.path.isfile(SPLIT_CSV) and _real(split_csv) != _real(SPLIT_CSV):
        keep = keep - {pid for pid, info in c10data.read_split(SPLIT_CSV).items() if info["split"] != "train"}
    return keep


class Index:
    """full key -> training rows (player, level, colour, move count). Rows read from train_<LEVEL>.csv in LEVELS
    order, file order inside a file."""

    def __init__(self, train_dir, part, exclude=None, split_csv=SPLIT_CSV):
        t0 = time.time()
        keep = players_of_part(part, split_csv)
        exclude = set(map(tuple, exclude or ()))
        self.part = part
        self.rows = []                     # (player_id, game_id, level, colour, nmov)
        self.by_key = {}
        self.level_of = {}
        self.stats = Counter()
        csv.field_size_limit(10 ** 8)
        for lv in LEVELS:
            path = os.path.join(train_dir, f"train_{lv}.csv")
            with open(guard_read(path), newline="", encoding="utf-8") as f:
                rd = csv.reader(f)
                if next(rd) != TRAIN_COLUMNS:
                    raise Drift(f"unexpected header in {path}")
                for rec in rd:
                    pid, gid, rank, colour, sgf = rec
                    if keep is not None and pid not in keep:
                        continue
                    if exclude and (pid, gid) in exclude:
                        self.stats["excluded_rows"] += 1
                        continue
                    if rank != lv:
                        raise Drift(f"{path}: row {gid} has rank {rank!r}")
                    if colour not in ("B", "W"):
                        raise Drift(f"{path}: row {gid} has colour {colour!r}")
                    old = self.level_of.setdefault(pid, rank)
                    if old != rank:
                        raise Drift(f"training player {pid} has two levels ({old}, {rank}): PREREG_14 3")
                    pts, cols = parse_game(sgf)
                    n = len(pts)
                    if not alternating_from_black(cols):
                        self.stats["rows_not_alternating_from_black"] += 1
                    if n < MIN_MOVES:
                        self.stats["rows_under_min_moves"] += 1
                    i = len(self.rows)
                    self.rows.append((pid, gid, rank, colour, n))
                    k = full_key(pts) if n else None
                    if k is None:
                        continue
                    hit = self.by_key.get(k)
                    if hit is None:
                        self.by_key[k] = i
                    elif isinstance(hit, list):
                        hit.append(i)
                    else:
                        self.by_key[k] = [hit, i]
        self.seconds = round(time.time() - t0, 1)
        log(f"index part {part}: {len(self.rows)} rows, {len(self.level_of)} players in {self.seconds}s")

    def lookup(self, k):
        hit = self.by_key.get(k)
        if hit is None:
            return []
        return hit if isinstance(hit, list) else [hit]

    def pins(self):
        multi_row = multi_player = multi_level = 0
        lines = []
        for k, hit in self.by_key.items():
            rows = hit if isinstance(hit, list) else [hit]
            if len(rows) > 1:
                multi_row += 1
                if len({self.rows[i][0] for i in rows}) > 1:
                    multi_player += 1
                if len({self.rows[i][2] for i in rows}) > 1:
                    multi_level += 1
            for i in rows:
                pid, _gid, lv, colour, n = self.rows[i]
                lines.append(f"{k.hex()}|{n}|{colour}|{lv}|{pid}")
        lines.sort()
        return {"part": self.part, "rows": len(self.rows), "players": len(self.level_of),
                "rows_per_level": {lv: sum(1 for r in self.rows if r[2] == lv) for lv in LEVELS},
                "distinct_full_keys": len(self.by_key), "keys_with_two_or_more_rows": multi_row,
                "keys_with_two_or_more_players": multi_player, "keys_with_two_or_more_levels": multi_level,
                "rows_not_alternating_from_black": self.stats["rows_not_alternating_from_black"],
                "rows_under_min_moves": self.stats["rows_under_min_moves"],
                "excluded_rows": self.stats["excluded_rows"],
                "sha256_sorted_key_moves_colour_level_player": hashlib.sha256(
                    "\n".join(lines).encode("ascii")).hexdigest()}


PIN_FIELDS = ("rows", "players", "rows_per_level", "distinct_full_keys", "keys_with_two_or_more_rows",
              "keys_with_two_or_more_players", "keys_with_two_or_more_levels", "rows_not_alternating_from_black",
              "rows_under_min_moves", "sha256_sorted_key_moves_colour_level_player")


def pin_differences(got, want):
    return {f: {"got": got.get(f), "pinned": want.get(f)} for f in PIN_FIELDS if got.get(f) != want.get(f)}


# ---------------------------------------------------------------- matching (PREREG_14 sections 2 and 4)
def read_questions(path):
    """[(question_id, [(slot, colour, sgf)])] in file order (slots 1..num_games, the T2 reader)."""
    t = t2.Table(guard_read(path))
    if t.error:
        raise Refuse(f"{path}: {t.error}")
    order, games = [], defaultdict(list)
    for qid, slot, colour, sgf in t2.question_games(t):
        if qid not in games:
            order.append(qid)
        games[qid].append((slot, colour, sgf))
    for qid in t.column("question_id") or []:
        if qid not in games:
            order.append(qid)
            games[qid] = []
    return [(q, games[q]) for q in order], t.sha256


def match_game(index, colour, sgf):
    """One test game -> {'status', 'rows'}: status no_colour / no_moves / short / illegal / none / same / opposite /
    mixed."""
    pts, cols = parse_game(sgf)
    n = len(pts)
    if colour not in ("B", "W"):
        return {"status": "no_colour", "moves": n, "same": [], "opposite": []}
    if n == 0:
        return {"status": "no_moves", "moves": 0, "same": [], "opposite": []}
    if n < MIN_MOVES:
        return {"status": "short", "moves": n, "same": [], "opposite": []}
    if not alternating_from_black(cols):
        return {"status": "illegal", "moves": n, "same": [], "opposite": []}
    hits = [i for i in index.lookup(full_key(pts)) if index.rows[i][4] == n]
    same = [i for i in hits if index.rows[i][3] == colour]
    opp = [i for i in hits if index.rows[i][3] != colour]
    status = "none" if not hits else "same" if same and not opp else "opposite" if opp and not same else "mixed"
    return {"status": status, "moves": n, "same": same, "opposite": opp}


def decide_question(index, games):
    """PREREG_14 4 for one question -> dict (override level or None, the copies, conflicts, record-only counts)."""
    copies, opposite, counts = [], 0, Counter()
    for slot, colour, sgf in games:
        m = match_game(index, colour, sgf)
        counts[m["status"]] += 1
        if m["same"]:
            copies.append((slot, m["same"]))
        if m["opposite"]:
            opposite += 1
    players = {index.rows[i][0] for _s, rows in copies for i in rows}
    levels = {index.rows[i][2] for _s, rows in copies for i in rows}
    out = {"games": len(games), "copies": len(copies), "opposite_side_copies_record_only": opposite,
           "status_counts": dict(sorted(counts.items())),
           "copy_slots": [s for s, _r in copies], "source_players": sorted(players), "source_levels": sorted(levels),
           "override": None, "conflict": False}
    if copies:
        if len(players) == 1:
            out["override"] = next(iter(levels))
        else:
            out["conflict"] = True
    return out


def scan_files(index, tests_dir):
    """Both rank question files -> {'files': {...}, 'questions': [(pool, qid, decision)]}."""
    files, questions = {}, []
    for pool, name in zip(POOLS, RANK_FILES):
        path = os.path.join(tests_dir, name)
        if not os.path.isfile(path):
            raise Refuse(f"{path} missing")
        qs, sha = read_questions(path)
        tot = Counter()
        for qid, games in qs:
            d = decide_question(index, games)
            questions.append((pool, qid, d))
            tot["questions"] += 1
            tot["games"] += d["games"]
            tot["copies"] += d["copies"]
            tot["opposite_side_copies_record_only"] += d["opposite_side_copies_record_only"]
            tot["questions_with_copy"] += d["copies"] > 0
            tot["questions_overridable"] += d["override"] is not None
            tot["questions_conflict"] += d["conflict"]
            for s, v in d["status_counts"].items():
                tot["games_" + s] += v
        files[name] = dict(sorted(tot.items()), pool=pool, sha256=sha)
    return {"files": files, "questions": questions}


# ---------------------------------------------------------------- base file and the 14 file
def read_base_lines(path):
    """(header line bytes, [(qid, level, line bytes, line ending bytes)]); refuses quoting or extra fields."""
    with open(guard_read(path), "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise Drift(f"{path}: starts with a BOM")
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise Drift(f"{path}: empty")
    out = []
    for ln in lines[1:]:
        body = ln.rstrip(b"\r\n")
        end = ln[len(body):]
        parts = body.decode("ascii").split(",")
        if len(parts) != 2 or '"' in body.decode("ascii") or parts[1] not in LEVELS:
            raise Drift(f"{path}: unexpected row {body[:60]!r}")
        out.append((parts[0], parts[1], body, end))
    if lines[0].rstrip(b"\r\n") != b"question_id,pred_rank":
        raise Drift(f"{path}: header {lines[0][:40]!r}")
    return lines[0], out


def build_14(base_path, overrides):
    """Bytes of the 14 file: the base file's bytes with only overridden rows whose level differs rewritten."""
    header, rows = read_base_lines(base_path)
    seen = {q for q, *_ in rows}
    missing = sorted(set(overrides) - seen)
    if missing:
        raise Drift(f"base file has no row for overridden questions {missing[:3]}")
    buf, changed = [header], []
    for qid, lv, body, end in rows:
        new = overrides.get(qid)
        if new is not None and new != lv:
            buf.append(f"{qid},{new}".encode("ascii") + end)
            changed.append({"question_id": qid, "base": lv, "new": new})
        else:
            buf.append(body + end)
    return b"".join(buf), changed


def write_overrides_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pool", "question_id", "base_level", "new_level", "changed", "copy_slots", "source_player"])
        for r in rows:
            w.writerow([r["pool"], r["question_id"], r["base_level"], r["new_level"], int(r["changed"]),
                        " ".join(map(str, r["copy_slots"])), r["source_player"]])


# ---------------------------------------------------------------- real-mode preconditions
def require_committed(rels):
    bad = [r for r in rels if not committed_clean(r)]
    if bad:
        raise Refuse(f"must be committed and clean first: {bad}")


def check_manifest(tests_dir, t1=None):
    """MANIFEST kind official, the rank question files equal to it, and (t1 given) each file's source_sha256 equal to
    the raw file the T1 checked (t02_checks.read_manifest)."""
    if t1 is not None:
        try:
            t2.read_manifest(tests_dir, "rank", t1)
        except SystemExit as e:
            raise Refuse(f"data/test does not match MANIFEST / the raw T1: {e}")
    mp = os.path.join(tests_dir, "MANIFEST.json")
    if not os.path.isfile(mp):
        raise Refuse(f"{mp} missing (docs/TESTDAY.md builds data/test)")
    man = read_json(mp)
    if man.get("kind") != "official":
        raise Refuse(f"{mp}: kind {man.get('kind')!r}; real mode needs the official test files")
    for name in RANK_FILES:
        p = os.path.join(tests_dir, name)
        if not os.path.isfile(p) or sha256_file(p) != ((man.get("files") or {}).get(name) or {}).get("sha256"):
            raise Refuse(f"{p} missing or differs from {mp}")
    return sha256_file(mp)


def git_first_commit_time(rel_path, row=None):
    """Committer time (datetime) of the first commit that added rel_path (or the text row to it); None if none."""
    args = ["log", "--reverse", "--format=%cI"] + (["-S" + row] if row else ["--diff-filter=A"]) + ["--", rel_path]
    rc, out = git(*args)
    if rc != 0 or not out:
        return None
    return datetime.fromisoformat(out.splitlines()[0].strip())


def check_k5_and_sim():
    """PREREG_14 6 and 8: sim.json committed with every pass condition true; the pins, the sim and the rev 2 row of
    the registration were first committed by K5-14 (2026-11-04 11:00 +08:00)."""
    r_sim, r_doc = rel(SIM), "docs/PREREG_14_RANK_EXACT_COPY.md"
    require_committed((r_sim, r_doc))
    sim = read_json(SIM)
    if not (sim.get("evaluation") or {}).get("passed"):
        raise NotUsed(f"{r_sim}: the simulation did not pass (PREREG_14 6): the layer is not used")
    late = {}
    for what, t in (("pins", git_first_commit_time(rel(PINS))), ("sim", git_first_commit_time(r_sim)),
                    ("rev 2 row", git_first_commit_time(r_doc, "| rev 2 |"))):
        if t is None or t > K5_14:
            late[what] = t.isoformat() if t else None
    if late:
        raise NotUsed(f"K5-14 {K5_14.isoformat()}: {late} committed late or not at all (PREREG_14 8): not used")
    return {"sim": r_sim, "sim_sha256_lf": sha256_lf(SIM), "k5_14": K5_14.isoformat(), "k5_ok": True}


def check_t1():
    r = rel(T1_JSON)
    if not os.path.isfile(T1_JSON) or not committed_clean(r):
        raise Refuse(f"{r} must exist and be committed (docs/TESTDAY.md, T1 on the raw files)")
    t1 = read_json(T1_JSON)
    probs = []
    if t1.get("complete") is not True:
        probs.append("not complete")
    if t1.get("stage") != "raw":
        probs.append(f"stage {t1.get('stage')!r}")
    if (t1.get("provenance") or {}).get("mode") != "real":
        probs.append("not real mode")
    if probs:
        raise Refuse(f"{r}: {probs}")
    v = t1.get("verdict")
    if v not in ("PASS", "NORMALIZED"):
        raise NotUsed(f"T1 verdict {v!r}: PREREG_14 9, the layer is not used")
    return {"file": r, "verdict": v, "sha256": sha256_file(T1_JSON), "_t1": t1}


def check_pins(index, part, pins_path):
    if not os.path.isfile(pins_path):
        raise Refuse(f"{pins_path} missing (PY tools/t14_copy.py pins, before 11/04)")
    want = (read_json(pins_path).get("parts") or {}).get(part)
    if not want:
        raise Refuse(f"{pins_path} has no pins for part {part!r}")
    got = index.pins()
    diff = pin_differences(got, want)
    if diff:
        raise Drift(f"index pins differ from {rel(pins_path)} (PREREG_14 3): {json.dumps(diff)[:900]}")
    return {"file": rel(pins_path), "part": part, "equal": True, "pins": {f: got[f] for f in PIN_FIELDS}}


def base_from_log(log_path, root, real):
    if real and not committed_clean(rel(log_path)):
        raise Refuse(f"{rel(log_path)} must be committed and clean")
    if not os.path.isfile(log_path):
        raise Refuse(f"{log_path} missing: no upload logged yet")
    lg = read_json(log_path)
    fin = (lg.get("final_choice") or {}).get("rank")
    if not fin:
        raise Refuse("final_choice.rank is not set: run official_log.py final --comp rank first (docs/TESTDAY_09.md "
                     "step 5); 14 is derived from the designated file (PREREG_14 7 S2)")
    e = next((x for x in lg.get("uploads") or [] if x.get("id") == fin.get("id")), None)
    if e is None or e.get("comp") != "rank" or e.get("status") != "uploaded":
        raise Refuse(f"final_choice.rank #{fin.get('id')} is not an uploaded competition-2 entry")
    if e.get("test14"):
        raise Refuse(f"final_choice.rank #{e['id']} is a 14 entry already; 14 is never built on 14")
    if fin.get("sha256") != e.get("sha256"):
        raise Drift("final_choice.rank and its entry record different SHA-256")
    frel = e.get("file") or ""
    if real and os.path.dirname(frel) not in BASE_DIRS:
        raise Refuse(f"base file {frel} is not in one of {BASE_DIRS}")
    path = os.path.join(root, *frel.split("/"))
    if not os.path.isfile(path) or sha256_file(path) != e["sha256"]:
        raise Drift(f"{frel} is missing or not the logged bytes ({e['sha256'][:12]}); restore with git -c "
                    f"core.autocrlf=false checkout -- {frel}")
    if real and not committed_clean(frel):
        raise Refuse(f"{frel} must be committed and clean")
    return {"entry_id": e["id"], "file": frel, "sha256": e["sha256"], "method": e.get("method"),
            "uploaded_at": e.get("uploaded_at"), "final_basis": fin.get("basis"), "final_set_at": fin.get("set_at"),
            "path": path}


# ---------------------------------------------------------------- commands
class Ctx:
    def __init__(self, args):
        self.real = not args.dry_run
        if self.real:
            if args.root or args.index_part or args.train_dir or args.exclude or args.split_csv or args.pins:
                raise Refuse("--root / --index-part / --train-dir / --exclude / --split-csv / --pins are for --dry-run "
                             "only")
            self.root, self.tests, self.log, self.out = REPO, DATA_TEST, LOG, OUT
            self.part, self.train_dir, self.exclude, self.pins_path = "all", TRAIN_DIR, None, PINS
            self.split_csv = SPLIT_CSV
        else:
            if not args.root:
                raise Refuse("--dry-run needs --root S")
            self.root = os.path.abspath(args.root)
            if inside(self.root, REPO):
                raise Refuse(f"--root {self.root} is inside the repository")
            self.tests = os.path.join(self.root, "data", "test")
            self.log = os.path.join(self.root, "results", "official_scores.json")
            self.out = os.path.join(self.root, "results", "test14")
            self.part = args.index_part or "fit"
            if self.part == "all":
                raise Refuse("--index-part all is for real mode only (PREREG_14 3: no 'val' players outside it)")
            self.train_dir = os.path.abspath(args.train_dir) if args.train_dir else TRAIN_DIR
            self.exclude = read_json(args.exclude) if args.exclude else None
            self.pins_path = os.path.abspath(args.pins) if args.pins else None
            self.split_csv = os.path.abspath(args.split_csv) if args.split_csv else SPLIT_CSV
        self.mode = "real" if self.real else "dry-run"

    def index(self):
        return Index(self.train_dir, self.part, self.exclude, split_csv=self.split_csv)


def guard_out(path, args):
    if os.path.exists(path) and not (args.rerun and (args.reason or "").strip()):
        raise Refuse(f"{path} exists; rewrite only with --rerun --reason TEXT")


def preflight(ctx, stage="scan", args=None):
    rec = {}
    if ctx.real:
        require_committed(OWN + (rel(PINS),))
        try:
            rec["k5_and_sim"] = check_k5_and_sim()
            rec["t1"] = check_t1()
        except NotUsed as e:
            write_not_used(ctx, str(e), {}, "preflight")
            raise
        rec["manifest_sha256"] = check_manifest(ctx.tests, rec["t1"].pop("_t1"))
    else:
        for name in RANK_FILES:
            if not os.path.isfile(os.path.join(ctx.tests, name)):
                raise Refuse(f"{ctx.tests}/{name} missing")
    nu = os.path.join(ctx.out, "not_used.json")
    if os.path.isfile(nu):
        prev = read_json(nu)
        # PREREG_14 7 / 9: after a registered 'not used' nothing is rerun, except an apply on a CHANGED base (an apply
        # whose overrides all equalled the old base's levels may change rows of a new base); cmd_apply checks the base
        if not (stage == "apply" and prev.get("stage") == "apply" and args is not None and args.rerun):
            raise NotUsed(f"{rel(nu)} exists: the layer is not used this season (PREREG_14 9)")
        rec["previous_not_used"] = prev
    return rec


def write_not_used(ctx, reason, extra, stage):
    p = os.path.join(ctx.out, "not_used.json")
    write_json(p, dict({"format": FORMAT_NOT_USED, "registration": "docs/PREREG_14_RANK_EXACT_COPY.md",
                        "stage": stage, "reason": reason}, **extra, provenance=provenance(ctx.mode)))
    log(f"not used: {reason} -> {p}")
    return p


def cmd_scan(args):
    ctx = Ctx(args)
    pre = preflight(ctx)
    out = os.path.join(ctx.out, "scan", "scan.json")
    guard_out(out, args)
    index = ctx.index()
    pins = check_pins(index, ctx.part, ctx.pins_path) if ctx.pins_path else None
    s = scan_files(index, ctx.tests)
    n_over = sum(1 for _p, _q, d in s["questions"] if d["override"] is not None)
    rep = {"format": FORMAT_SCAN, "registration": "docs/PREREG_14_RANK_EXACT_COPY.md (S1, record only)",
           "preflight": pre, "index_part": ctx.part, "index_seconds": index.seconds, "index_pins_check": pins,
           "files": s["files"], "questions_overridable": n_over,
           "overridable": [{"pool": p, "question_id": q, "level": d["override"], "copy_slots": d["copy_slots"],
                            "source_player": d["source_players"][0]} for p, q, d in s["questions"]
                           if d["override"] is not None],
           "conflicts": [{"pool": p, "question_id": q, "source_players": d["source_players"],
                          "source_levels": d["source_levels"]} for p, q, d in s["questions"] if d["conflict"]],
           "rerun_reason": args.reason if args.rerun else None, "provenance": provenance(ctx.mode)}
    write_json(out, rep)
    print(f"scan: {n_over} questions overridable; copies " +
          ", ".join(f"{f['pool']} {f.get('copies', 0)}" for f in s["files"].values()) + f" -> {out}")
    if n_over == 0:
        write_not_used(ctx, "S1 scan: no question has an exact same-side copy (PREREG_14 7 S1)",
                       {"scan": rel(out) if ctx.real else out}, "scan")
        raise NotUsed("no question can be overridden")
    return rep


def cmd_apply(args):
    ctx = Ctx(args)
    pre = preflight(ctx, "apply", args)
    rdir = os.path.join(ctx.out, "rank")
    rpath = os.path.join(rdir, "t14_copy_report.json")
    guard_out(rpath, args)
    base = base_from_log(ctx.log, ctx.root, ctx.real)
    prev = pre.pop("previous_not_used", None)
    if prev is not None:
        pb = prev.get("base") or {}
        if (pb.get("entry_id"), pb.get("sha256")) == (base["entry_id"], base["sha256"]):
            raise NotUsed(f"not_used.json was written for this same base #{base['entry_id']}; PREREG_14 7 reruns an "
                          f"apply only on a changed base")
        pre["superseded_not_used"] = {"base": pb, "reason": prev.get("reason")}
    index = ctx.index()
    pins = check_pins(index, ctx.part, ctx.pins_path) if ctx.pins_path else None
    s = scan_files(index, ctx.tests)
    _h, base_rows = read_base_lines(base["path"])
    base_level = {q: lv for q, lv, _b, _e in base_rows}
    overrides = {q: d["override"] for _p, q, d in s["questions"] if d["override"] is not None}
    data, changed = build_14(base["path"], overrides)
    rows = [{"pool": p, "question_id": q, "base_level": base_level.get(q), "new_level": d["override"],
             "changed": base_level.get(q) != d["override"], "copy_slots": d["copy_slots"],
             "source_player": d["source_players"][0]} for p, q, d in s["questions"] if d["override"] is not None]
    rep = {"format": FORMAT_REPORT, "registration": "docs/PREREG_14_RANK_EXACT_COPY.md", "mode": ctx.mode,
           "method": METHOD, "preflight": pre, "base": {k: v for k, v in base.items() if k != "path"},
           "index_part": ctx.part, "index_seconds": index.seconds, "index_pins_check": pins,
           "files": s["files"], "questions_overridden": len(rows), "rows_changed": len(changed),
           "changed_by_pool": dict(Counter(r["pool"] for r in rows if r["changed"])),
           "overridden_by_pool": dict(Counter(r["pool"] for r in rows)),
           "overrides": rows,
           "conflicts": [{"pool": p, "question_id": q, "source_players": d["source_players"],
                          "source_levels": d["source_levels"]} for p, q, d in s["questions"] if d["conflict"]],
           "k6_14": K6_14.isoformat(), "rerun_reason": args.reason if args.rerun else None}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    def supersede(names):
        """Move earlier outputs to results/test14/superseded/ (kept in git history; never deleted)."""
        moved = []
        for name in names:
            src = os.path.join(ctx.out, name)
            if os.path.isfile(src):
                dst = os.path.join(ctx.out, "superseded", f"{stamp}_" + name.replace("/", "_").replace(os.sep, "_"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(src, dst)
                moved.append(rel(dst) if ctx.real else dst)
        return moved
    if not rows or not changed:
        why = ("no question has an exact same-side copy" if not rows else
               "every overridden question already has that level in the base file (14 would equal the base)")
        rep["verdict"] = "not_used"
        rep["provenance"] = provenance(ctx.mode)
        rep["superseded"] = supersede([os.path.join("rank", "submission_rank.csv"), os.path.join("rank", "overrides.csv"),
                                       "not_used.json"])
        write_json(rpath, rep)
        write_not_used(ctx, f"S2 apply: {why} (PREREG_14 7 S2)",
                       {"report": rel(rpath) if ctx.real else rpath,
                        "base": {k: base[k] for k in ("entry_id", "file", "sha256")}}, "apply")
        raise NotUsed(why)
    sub = os.path.join(rdir, "submission_rank.csv")
    os.makedirs(rdir, exist_ok=True)
    with open(sub, "wb") as f:
        f.write(data)
    write_overrides_csv(os.path.join(rdir, "overrides.csv"), rows)
    qs = [os.path.join(ctx.tests, n) for n in RANK_FILES]
    errors = check_submission.check("rank", sub, qs, official=True)
    rep["check_submission"] = {"mode": "official", "errors": errors[:20]}
    rep["outputs_sha256"] = {"submission_rank.csv": sha256_file(sub),
                             "overrides.csv": sha256_file(os.path.join(rdir, "overrides.csv"))}
    rep["verdict"] = "error" if errors else "used"
    rep["superseded"] = [] if errors else supersede(["not_used.json"])
    rep["provenance"] = provenance(ctx.mode)
    write_json(rpath, rep)
    if errors:
        raise Drift(f"check_submission --official failed: {errors[:3]}")
    print(f"apply: 14 written, {len(rows)} questions overridden, {len(changed)} rows changed vs base #"
          f"{base['entry_id']} ({base['file']}) -> {sub}")
    return rep


def cmd_status(args):
    ctx = Ctx(args)
    lines = []
    for name in ("not_used.json", os.path.join("scan", "scan.json"), os.path.join("rank", "t14_copy_report.json")):
        p = os.path.join(ctx.out, name)
        if os.path.isfile(p):
            j = read_json(p)
            lines.append(f"{name}: " + (j.get("reason") or f"verdict {j.get('verdict')}, overridden "
                                        f"{j.get('questions_overridden', j.get('questions_overridable'))}"))
        else:
            lines.append(f"{name}: (none)")
    left = K6_14 - datetime.now(TW)
    lines.append(f"K6-14 {K6_14.isoformat()}: " + ("passed" if left.total_seconds() <= 0 else f"{left} left"))
    print("\n".join(lines))
    return lines


def cmd_pins(args):
    if args.dry_run:
        raise Refuse("pins is real mode only")
    out = PINS
    guard_out(out, args)
    parts = {}
    for part in ("all", "fit"):
        idx = Index(TRAIN_DIR, part)
        parts[part] = dict(idx.pins(), seconds=idx.seconds)
    rep = {"format": FORMAT_PINS, "registration": "docs/PREREG_14_RANK_EXACT_COPY.md section 3",
           "min_moves": MIN_MOVES, "key": t2.KEY_VERSION, "parts": parts,
           "rerun_reason": args.reason if args.rerun else None, "provenance": provenance("real")}
    write_json(out, rep)
    print(json.dumps({p: {k: v for k, v in d.items() if k != "rows_per_level"} for p, d in parts.items()}, indent=1))
    return rep


# ---------------------------------------------------------------- simulation (PREREG_14 section 6)
def _h(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def sym_sgf(sgf, s):
    """The game with every point mapped by symmetry s (p02_moves.SYM_TABLE), in the training SGF form."""
    full = t2.MOVE_RE.findall(sgf)
    return "(;SZ[19]" + "".join(f";{c}[{t2.SYM[s][p]}]" for c, p in full) + ")"


def drop_last_move(sgf):
    full = t2.MOVE_RE.findall(sgf)[:-1]
    return "(;SZ[19]" + "".join(f";{c}[{p}]" for c, p in full) + ")"


def load_fit_games(train_dir, split_csv=SPLIT_CSV):
    """{player: {'level', 'B': [(gid, sgf)], 'W': [...]}} for the fit players (each colour in the sim hash order)."""
    keep = players_of_part("fit", split_csv)
    by = {}
    csv.field_size_limit(10 ** 8)
    for lv in LEVELS:
        with open(guard_read(os.path.join(train_dir, f"train_{lv}.csv")), newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            next(rd)
            for pid, gid, rank, colour, sgf in rd:
                if pid in keep:
                    d = by.setdefault(pid, {"level": rank, "B": [], "W": []})
                    d[colour].append((gid, sgf))
    for pid, d in by.items():
        for c in "BW":
            d[c].sort(key=lambda g, pid=pid: _h(SIM_SALT_GAME, pid, g[0]))
    return by


def halves(games):
    k = (len(games) + 1) // 2
    return games[:k], games[k:]          # (held out, in the index)


def make_fake(train_dir, split_csv=SPLIT_CSV):
    """PREREG_14 6: two official-shaped question files from fit players. Returns (files {name: rows}, expected
    {qid: (kind, player, level)}, exclude [[pid, gid]] = every held-out game of every fit player, base rows)."""
    by = load_fit_games(train_dir, split_csv)
    exclude = [[pid, gid] for pid, d in by.items() for c in "BW" for gid, _s in halves(d[c])[0]]
    elig = sorted(((pid, c) for pid, d in by.items() for c in "BW" if len(d[c]) >= SIM_MIN_COLOUR_GAMES
                   and len(d["W" if c == "B" else "B"]) >= 2),
                  key=lambda x: _h(SIM_SALT_PLAYER, x[0], x[1]))
    per_level = defaultdict(list)
    for pid, c in elig:
        per_level[by[pid]["level"]].append((pid, c))
    used, queue = set(), []
    while any(per_level.values()):
        for lv in LEVELS:
            while per_level[lv]:
                pid, c = per_level[lv].pop(0)
                if pid not in used:
                    used.add(pid)
                    queue.append((pid, c))
                    break
    if len(queue) < 800:
        raise Refuse(f"only {len(queue)} eligible fit players for the simulation")
    kinds = [k for k, n in SIM_KINDS for _ in range(n)]
    files, expected, base = {}, {}, []
    qn = 0
    for fi, (pool, name) in enumerate(zip(POOLS, RANK_FILES)):
        prefix = ("pub_", "priv_")[fi]
        order = sorted(range(400), key=lambda j: _h(SIM_SALT_KIND, pool, j))
        kind_of = {j: kinds[r] for r, j in enumerate(order)}
        rows = []
        for j in range(400):
            pid, c = queue[qn]
            qn += 1
            k, size = kind_of[j], SIM_SIZES[j % 4]
            held, inidx = halves(by[pid][c])
            other = "W" if c == "B" else "B"
            if k == "N":
                games = [s for _g, s in held[:size]]
            elif k == "C1":
                games = [inidx[0][1]] + [s for _g, s in held[:size - 1]]
            elif k == "CH":
                m = max(1, size // 2)
                games = [s for _g, s in inidx[:m]] + [s for _g, s in held[:size - m]]
            elif k == "CA":
                games = [s for _g, s in inidx[:size]]
            elif k == "XS":
                games = [sym_sgf(inidx[0][1], 1 + j % 7)] + [s for _g, s in held[:size - 1]]
            elif k == "XO":
                games = [halves(by[pid][other])[1][0][1]] + [s for _g, s in held[:size - 1]]
            elif k == "XT":
                long_ = next(s for _g, s in inidx if len(t2.MOVE_RE.findall(s)) >= MIN_MOVES + 1)
                games = [drop_last_move(long_)] + [s for _g, s in held[:size - 1]]
            elif k == "XC":
                lv = by[pid]["level"]
                op = next(q for q, cc in elig if cc == c and q != pid and by[q]["level"] != lv
                          and _h("c10-p14|xc", pool, j, q) < "8")
                games = [inidx[0][1], halves(by[op][c])[1][0][1]] + [s for _g, s in held[:size - 2]]
            qid = f"{prefix}{j + 1:04d}"
            rows.append((qid, games, c))
            expected[qid] = {"kind": k, "player": pid, "level": by[pid]["level"], "size": size}
            base.append((qid, LEVELS[int(_h(SIM_SALT_BASE, qid), 16) % len(LEVELS)]))
        files[name] = rows
    return files, expected, exclude, base


def write_fake_tree(root, files, base):
    tdir = os.path.join(root, "data", "test")
    os.makedirs(tdir, exist_ok=True)
    man = {"kind": "t14-fake", "files": {}}
    for name, rows in files.items():
        p = os.path.join(tdir, name)
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(t2.Q_COLS)
            for qid, games, c in rows:
                w.writerow([qid, len(games)] + games + [""] * (20 - len(games)) + [c] * len(games)
                           + [""] * (20 - len(games)))
        man["files"][name] = {"sha256": sha256_file(p)}
    write_json(os.path.join(tdir, "MANIFEST.json"), man)
    bdir = os.path.join(root, "results", "test09", "rank")
    os.makedirs(bdir, exist_ok=True)
    bpath = os.path.join(bdir, "submission_rank.csv")
    with open(bpath, "wb") as f:
        f.write(b"question_id,pred_rank\n" + b"".join(f"{q},{lv}\n".encode("ascii") for q, lv in base))
    sha = sha256_file(bpath)
    lg = {"format": "c10 official_scores v1", "teams": {"rank": "TEAM_11248", "player": "TEAM_11249"},
          "baselines_public": {"rank": 0.4631, "player": 0.2845},
          "final_choice": {"rank": {"id": 1, "sha256": sha, "file": "results/test09/rank/submission_rank.csv",
                                    "method": "fake base", "basis": "t14 fake", "set_at": now_iso()},
                           "player": None},
          "uploads": [{"id": 1, "comp": "rank", "status": "uploaded", "file": "results/test09/rank/submission_rank.csv",
                       "sha256": sha, "method": "fake base", "uploaded_at": "2026-11-20T10:00:00+08:00"}]}
    write_json(os.path.join(root, "results", "official_scores.json"), lg)
    return bpath


def cmd_fake(args):
    if not args.root or inside(args.root, REPO):
        raise Refuse("fake needs --root S outside the repository")
    files, expected, exclude, base = make_fake(os.path.abspath(args.train_dir) if args.train_dir else TRAIN_DIR)
    root = os.path.abspath(args.root)
    write_fake_tree(root, files, base)
    write_json(os.path.join(root, "t14_expected.json"), expected)
    write_json(os.path.join(root, "t14_exclude.json"), exclude)
    print(f"fake tree in {root}: 800 questions, {len(exclude)} held-out games excluded from the index")
    return expected


def evaluate(expected, report, base_path, out_path):
    """PREREG_14 6 pass conditions 1-3."""
    over = {r["question_id"]: r["new_level"] for r in report.get("overrides") or []}
    want = {q for q, e in expected.items() if e["kind"] in SIM_EXPECT_OVERRIDE}
    by_kind = defaultdict(lambda: Counter())
    for q, e in expected.items():
        c = by_kind[e["kind"]]
        c["questions"] += 1
        c["overridden"] += q in over
        c["overridden_correct_level"] += over.get(q) == e["level"]
    missed = sorted(want - set(over))
    wrong_level = sorted(q for q in want & set(over) if over[q] != expected[q]["level"])
    spurious = sorted(set(over) - want)
    _h1, brow = read_base_lines(base_path)
    _h2, orow = read_base_lines(out_path)
    untouched_changed = sorted(q for (q, lv, *_r), (q2, lv2, *_r2) in zip(brow, orow) if q not in over
                               and (q != q2 or lv != lv2))
    with open(base_path, "rb") as f:
        bl = f.read().splitlines(keepends=True)
    with open(out_path, "rb") as f:
        ol = f.read().splitlines(keepends=True)
    byte_untouched = all(a == b for a, b, (q, *_r) in zip(bl[1:], ol[1:], brow) if q not in over) and \
        len(bl) == len(ol) and bl[0] == ol[0]
    out_level = {q: lv for q, lv, *_r in orow}
    file_wrong = sorted(q for q in want if out_level.get(q) != expected[q]["level"])
    cond = {"1_expected_overrides_all_correct": not missed and not wrong_level and not file_wrong,
            "2_no_override_where_not_expected": not spurious,
            "3_untouched_rows_byte_equal_and_format_ok": byte_untouched and not untouched_changed
            and not (report.get("check_submission") or {}).get("errors")}
    return {"conditions": cond, "passed": all(cond.values()),
            "expected_override_questions": len(want), "overridden": len(over), "missed": missed[:20],
            "wrong_level": wrong_level[:20], "output_file_wrong_level": file_wrong[:20], "spurious": spurious[:20],
            "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())}}


def loo_fit(train_dir, split_csv=SPLIT_CSV):
    """Record only (PREREG_14 6): for every fit game, other fit rows with the same full key and move count; and the
    K100 (first 100 moves, T2's copy key) pairs across players / levels."""
    keep = players_of_part("fit", split_csv)
    full, k100 = defaultdict(list), defaultdict(list)
    csv.field_size_limit(10 ** 8)
    n = 0
    for lv in LEVELS:
        with open(guard_read(os.path.join(train_dir, f"train_{lv}.csv")), newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            next(rd)
            for pid, gid, rank, colour, sgf in rd:
                if pid not in keep:
                    continue
                pts, _c = parse_game(sgf)
                kf, k1, _k6 = t2.game_keys(pts)
                full[(kf, len(pts))].append((pid, rank))
                if k1 is not None:
                    k100[k1].append((pid, rank))
                n += 1

    def summary(groups):
        g_multi = [v for v in groups.values() if len(v) > 1]
        rows_multi = sum(len(v) for v in g_multi)
        cross_p = [v for v in g_multi if len({p for p, _l in v}) > 1]
        cross_l = [v for v in g_multi if len({lv for _p, lv in v}) > 1]
        return {"groups_with_two_or_more_rows": len(g_multi), "rows_in_such_groups": rows_multi,
                "groups_across_players": len(cross_p), "rows_in_groups_across_players": sum(len(v) for v in cross_p),
                "groups_across_levels": len(cross_l), "rows_in_groups_across_levels": sum(len(v) for v in cross_l)}
    return {"fit_games": n, "full_game_same_moves": summary(full), "k100_record_only": summary(k100)}


def cmd_sim(args):
    if args.dry_run:
        raise Refuse("sim writes the committed design record; it builds its own temporary dry-run tree")
    out = SIM
    guard_out(out, args)
    t0 = time.time()
    files, expected, exclude, base = make_fake(TRAIN_DIR)
    with tempfile.TemporaryDirectory(prefix="c10_t14_sim_") as root:
        write_fake_tree(root, files, base)
        ex = os.path.join(root, "t14_exclude.json")
        write_json(ex, exclude)
        ns = argparse.Namespace(dry_run=True, root=root, index_part="fit", train_dir=None, exclude=ex, pins=None,
                                split_csv=None, rerun=False, reason=None)
        rep = cmd_apply(ns)
        ev = evaluate(expected, rep, os.path.join(root, "results", "test09", "rank", "submission_rank.csv"),
                      os.path.join(root, "results", "test14", "rank", "submission_rank.csv"))
        idx_info = {"part": rep["index_part"], "seconds": rep["index_seconds"]}
        outs = dict(rep["outputs_sha256"], base_submission_rank_csv=sha256_file(
            os.path.join(root, "results", "test09", "rank", "submission_rank.csv")))
        overrides = {"questions_overridden": rep["questions_overridden"], "rows_changed": rep["rows_changed"],
                     "overridden_by_pool": rep["overridden_by_pool"], "changed_by_pool": rep["changed_by_pool"],
                     "conflicts": len(rep["conflicts"])}
    loo = loo_fit(TRAIN_DIR)
    rec = {"format": FORMAT_SIM, "registration": "docs/PREREG_14_RANK_EXACT_COPY.md section 6",
           "players": "fit only (split v1); no validation player, no data/local",
           "held_out_games_excluded_from_index": len(exclude), "index": idx_info,
           "questions": len(expected), "kinds_per_file": dict(SIM_KINDS),
           "files": rep["files"], "apply": overrides, "outputs_sha256": outs, "evaluation": ev,
           "leave_one_out_fit_record_only": loo,
           "seconds": round(time.time() - t0, 1),
           "rerun_reason": args.reason if args.rerun else None, "provenance": provenance("design")}
    write_json(out, rec)
    print(json.dumps({"passed": ev["passed"], "conditions": ev["conditions"], "by_kind": ev["by_kind"],
                      "loo": loo}, indent=1))
    if not ev["passed"]:
        raise Drift("simulation pass conditions not met (PREREG_14 6): a program error to fix")
    return rec


# ---------------------------------------------------------------- CLI
def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="t14_copy.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pins", "sim", "scan", "apply", "status", "fake"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--root")
        p.add_argument("--index-part", choices=("fit", "train", "all"))
        p.add_argument("--train-dir")
        p.add_argument("--exclude")
        p.add_argument("--pins", help="dry-run: a pins JSON to check the index against")
        p.add_argument("--split-csv", help="dry-run: another split-v1 style player table (tests)")
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason")
    args = ap.parse_args(argv)
    if args.rerun and not (args.reason or "").strip():
        ap.error("--rerun needs --reason TEXT")
    fn = {"pins": cmd_pins, "sim": cmd_sim, "scan": cmd_scan, "apply": cmd_apply, "status": cmd_status,
          "fake": cmd_fake}[args.cmd]
    try:
        fn(args)
    except (Refuse, NotUsed, Drift) as e:
        log(f"{type(e).__name__} (exit {e.code}): {e}")
        return e.code
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
