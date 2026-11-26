"""C10 PREREG_21: the competition-1 exact-copy layer (docs/PREREG_21_PLAYER_EXACT_COPY.md). CPU only, no GPU.

A competition-1 question game that is an exact copy (PREREG_21 section 2) of a game in the SAME pool's candidate file --
the whole move sequence equal up to the 8 board symmetries (the T2 full key of tools/t02_checks.py), the same move
count, at least 100 moves, black first and strictly alternating, and the same colour -- names the candidate who played
that game. A question whose copies all point to one and the same candidate P gets P at top 1, followed by the base
row's other candidates in their order (section 4); every other question keeps the base file's row byte for byte. The
base file is the competition-1 upload that official_log.py final already designates (results/official_scores.json
final_choice.player, an uploaded entry: Q04, 06-OFF, 06-ON or 10).

    PY tools/t21_copy.py scan   [--rerun --reason TEXT]        -> results/test21/scan/scan.json (record only)
    PY tools/t21_copy.py apply  [--rerun --reason TEXT]        -> results/test21/player/{submission_player.csv,
                                                                  overrides.csv, t21_copy_report.json} or
                                                                  results/test21/not_used.json
    PY tools/t21_copy.py status
    PY tools/t21_copy.py sim    [--rerun --reason TEXT]        -> results/test21/design/sim.json (fit players only)
    PY tools/t21_copy.py fake   --root S [--no-copies]         (non-real: an official-shaped fake tree, section 6)
    scan / apply / status also take  --dry-run --root S

Modes
    real (default)  data/test (MANIFEST kind official, the four competition-1 files equal to it and to the raw T1 of
                    results/test02/checks/player/t1.json: committed, complete, stage raw, mode real, PASS or
                    NORMALIZED), the committed and passed results/test21/design/sim.json, K5-21 (the '| rev 2 |' row
                    of the registration reached main's first-parent history by 2026-11-03 23:59:59 +08:00), the base
                    from the committed results/official_scores.json. This tool, t02_checks.py, p02_moves.py,
                    check_submission.py and c10data.py must be committed and clean. Outputs only under results/test21/.
    --dry-run       any tree S (S/data/test, S/results/official_scores.json, S/results/test21); S must be outside the
                    repository. No T1, no K5-21, no committed-file checks; base folder not restricted.

Exit codes: 0 done (apply: 21 written; scan / sim: written), 2 refusal (a precondition; a traceback is 2 too),
3 the registered 'not used' (no override, the 21 file would equal the base, T1 not PASS / NORMALIZED, the simulation
not passed, K5-21 missed, an earlier not_used.json), 4 local drift (the base file differs from its log entry,
check_submission --official failed).

Reads at test time only the four competition-1 test files of data/test, their MANIFEST, the raw T1, the log and the
base file. Never opens data/local, never reads a competition-2 file, the training data (except `sim`, before the test
day, fit players only) or anything external; never imports tools/score.py, decide.py, a competition-2 tool, a 06 / 10
tool or a model tool. Candidate ids are the anonymised ids of the official files; nothing here links them to anything
outside them (PREREG_21 1).
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
import c10data  # noqa: E402  (sim only: LEVELS, COLUMNS, the split-v1 player sets)
import check_submission  # noqa: E402
import t02_checks as t2  # noqa: E402  (T2 game key, MOVE_RE, SYM, the question / candidate readers, read_manifest)

# ---------------------------------------------------------------- registered (PREREG_21)
MIN_MOVES = 100                       # section 2.4
FORMAT_REPORT = "c10-test21 copy apply v1"
FORMAT_SCAN = "c10-test21 copy scan v1"
FORMAT_SIM = "c10-test21 sim v1"
FORMAT_NOT_USED = "c10-test21 not used v1"
METHOD = "prereg21 exact-copy layer (PREREG_21 4) on the designated competition-1 file"
TW = timezone(timedelta(hours=8))
K5_21 = datetime(2026, 11, 3, 23, 59, 59, tzinfo=TW)      # PREREG_21 8: rev 2, tool, tests, sim merged into main by this
K6_21 = datetime(2026, 11, 23, 18, 0, 0, tzinfo=TW)       # PREREG_21 8: the 21 report's last commit
POOLS = ("public", "private")
Q_FILES = {pool: name for pool, kind, name in t2.FILES["player"] if kind == "questions"}
C_FILES = {pool: name for pool, kind, name in t2.FILES["player"] if kind == "candidates"}
PLAYER_FILES = tuple(name for _p, _k, name in t2.FILES["player"])
TOP = ["top_%d" % i for i in range(1, 6)]
HEADER = b"question_id," + ",".join(TOP).encode("ascii")
BASE_DIRS = ("results/test02/player", "results/test06/player/off", "results/test06/player/on", "results/test10/player")
REV2_ROW = "| rev 2 |"
SIM_SALT = "c10-p21"
SIM_KINDS = (("N", 200), ("C1", 50), ("CH", 50), ("CA", 40), ("XS", 12), ("XT", 12), ("XC", 12), ("XP", 12),
             ("XU", 12))
SIM_EXPECT_OVERRIDE = ("C1", "CH", "CA", "XS")
SIM_SIZES = (5, 10, 15, 20)
SIM_LIBRARY = 100
SIM_HELD = 20
SIM_SHORT = 80
SIM_PER_POOL = 400

# ---------------------------------------------------------------- paths
DATA_TEST = os.path.join(REPO, "data", "test")
LOG = os.path.join(REPO, "results", "official_scores.json")
OUT = os.path.join(REPO, "results", "test21")
T1_JSON = os.path.join(REPO, "results", "test02", "checks", "player", "t1.json")
SIM = os.path.join(OUT, "design", "sim.json")
DOC = "docs/PREREG_21_PLAYER_EXACT_COPY.md"
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
OWN = ("tools/t21_copy.py", "tools/t02_checks.py", "tools/p02_moves.py", "tools/check_submission.py",
       "tools/c10data.py")
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


def committed_clean(rel_path):
    rc_t, _ = git("ls-files", "--error-unmatch", "--", rel_path)
    rc_s, st = git("status", "--porcelain", "--", rel_path)
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
    d = {"tool": "tools/t21_copy.py", "tool_sha256_lf": sha256_lf(os.path.join(HERE, "t21_copy.py")),
         "t02_checks_sha256_lf": sha256_lf(os.path.join(HERE, "t02_checks.py")), "mode": mode,
         "git_head": git_head(), "python": platform.python_version(), "created_utc": now_iso()}
    d.update(extra or {})
    return d


# ---------------------------------------------------------------- moves and keys (PREREG_21 section 2)
def parse_game(sgf):
    """(points, colours) of every MOVE_RE match (the T2 parse, no move limit)."""
    full = t2.MOVE_RE.findall(sgf or "")
    return [p for _, p in full], "".join(c for c, _ in full)


def alternating_from_black(cols):
    return cols[:1] == "B" and cols == ("BW" * (len(cols) // 2 + 1))[:len(cols)]


def full_key(pts):
    """The T2 full key: SHA-1[:12] of the D4-minimal whole-game point string (t02_checks.game_key)."""
    return t2.game_key(pts)


# ---------------------------------------------------------------- candidate index (PREREG_21 section 3)
class PoolIndex:
    """One pool's candidate file: full key -> rows (player_id, game_id, colour, move count), file order."""

    def __init__(self, path):
        t0 = time.time()
        t = t2.Table(guard_read(path))
        if t.error:
            raise Refuse(f"{path}: {t.error}")
        if any(c not in t.col for c in t2.C_COLS):
            raise Refuse(f"{path}: header {t.header} lacks one of {t2.C_COLS}")
        self.sha256 = t.sha256
        self.rows = []
        self.by_key = defaultdict(list)
        self.stats = Counter()
        for pid, gid, colour, sgf in t2.candidate_games(t):
            pts, cols = parse_game(sgf)
            n = len(pts)
            if n < MIN_MOVES:
                self.stats["rows_under_min_moves"] += 1
            if not alternating_from_black(cols):
                self.stats["rows_not_alternating_from_black"] += 1
            i = len(self.rows)
            self.rows.append((pid, gid, colour, n))
            if n:
                self.by_key[full_key(pts)].append(i)
        self.seconds = round(time.time() - t0, 1)

    def lookup(self, k, n):
        return [i for i in self.by_key.get(k, ()) if self.rows[i][3] == n]

    def summary(self):
        multi = [v for v in self.by_key.values() if len(v) > 1]
        return {"sha256": self.sha256, "rows": len(self.rows), "candidates": len({r[0] for r in self.rows}),
                "distinct_full_keys": len(self.by_key), "keys_with_two_or_more_rows": len(multi),
                "keys_with_two_or_more_candidates": sum(1 for v in multi if len({self.rows[i][0] for i in v}) > 1),
                "rows_under_min_moves": self.stats["rows_under_min_moves"],
                "rows_not_alternating_from_black": self.stats["rows_not_alternating_from_black"],
                "seconds": self.seconds}


# ---------------------------------------------------------------- matching (PREREG_21 sections 2 and 4)
def read_questions(path):
    """[(question_id, [(slot, colour, sgf)])] in file order (slots 1..num_games, the T2 reader), file SHA-256."""
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


def match_game(own, other, colour, sgf):
    """One question game -> {'status', 'same', 'opposite', 'other_pool'}: status no_colour / no_moves / short /
    illegal / none / same / opposite / mixed (own pool only); other_pool = rows of the other pool (record only)."""
    pts, cols = parse_game(sgf)
    n = len(pts)
    out = {"moves": n, "same": [], "opposite": [], "other_pool": 0}
    if colour not in ("B", "W"):
        return dict(out, status="no_colour")
    if n == 0:
        return dict(out, status="no_moves")
    if n < MIN_MOVES:
        return dict(out, status="short")
    if not alternating_from_black(cols):
        return dict(out, status="illegal")
    k = full_key(pts)
    hits = own.lookup(k, n)
    same = [i for i in hits if own.rows[i][2] == colour]
    opp = [i for i in hits if own.rows[i][2] != colour]
    oth = len(other.lookup(k, n)) if other is not None else 0
    status = "none" if not hits else "same" if same and not opp else "opposite" if opp and not same else "mixed"
    return dict(out, status=status, same=same, opposite=opp, other_pool=oth)


def decide_question(own, other, games):
    """PREREG_21 4 for one question -> dict (override candidate or None, the copies, conflicts, record-only counts)."""
    copies, opposite, other_pool, counts = [], 0, 0, Counter()
    for slot, colour, sgf in games:
        m = match_game(own, other, colour, sgf)
        counts[m["status"]] += 1
        if m["same"]:
            copies.append((slot, m["same"]))
        opposite += bool(m["opposite"])
        other_pool += bool(m["other_pool"])
    cands = sorted({own.rows[i][0] for _s, rows in copies for i in rows})
    out = {"games": len(games), "copies": len(copies), "opposite_colour_copies_record_only": opposite,
           "other_pool_copies_record_only": other_pool, "status_counts": dict(sorted(counts.items())),
           "copy_slots": [s for s, _r in copies],
           "copy_games": [[own.rows[i][1] for i in rows] for _s, rows in copies],
           "source_candidates": cands, "override": None, "conflict": False}
    if copies:
        if len(cands) == 1:
            out["override"] = cands[0]
        else:
            out["conflict"] = True
    return out


def scan_files(tests_dir):
    """Both pools -> {'files': {...}, 'pools': {...}, 'questions': [(pool, qid, decision)]}."""
    for name in PLAYER_FILES:
        if not os.path.isfile(os.path.join(tests_dir, name)):
            raise Refuse(f"{os.path.join(tests_dir, name)} missing")
    idx = {pool: PoolIndex(os.path.join(tests_dir, C_FILES[pool])) for pool in POOLS}
    files, questions = {}, []
    for pool in POOLS:
        other = idx["private" if pool == "public" else "public"]
        name = Q_FILES[pool]
        qs, sha = read_questions(os.path.join(tests_dir, name))
        tot = Counter()
        for qid, games in qs:
            d = decide_question(idx[pool], other, games)
            questions.append((pool, qid, d))
            tot["questions"] += 1
            tot["games"] += d["games"]
            tot["copies"] += d["copies"]
            tot["opposite_colour_copies_record_only"] += d["opposite_colour_copies_record_only"]
            tot["other_pool_copies_record_only"] += d["other_pool_copies_record_only"]
            tot["questions_with_copy"] += d["copies"] > 0
            tot["questions_overridable"] += d["override"] is not None
            tot["questions_conflict"] += d["conflict"]
            for s, v in d["status_counts"].items():
                tot["games_" + s] += v
        files[name] = dict(sorted(tot.items()), pool=pool, sha256=sha)
    pools = {pool: dict(idx[pool].summary(), file=C_FILES[pool]) for pool in POOLS}
    return {"files": files, "pools": pools, "questions": questions}


# ---------------------------------------------------------------- base file and the 21 file
def read_base_lines(path):
    """(header line bytes, [(qid, [5 candidates], body bytes, line ending bytes)]); refuses quoting or other widths."""
    with open(guard_read(path), "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise Drift(f"{path}: starts with a BOM")
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise Drift(f"{path}: empty")
    if lines[0].rstrip(b"\r\n") != HEADER:
        raise Drift(f"{path}: header {lines[0][:60]!r}")
    out = []
    for ln in lines[1:]:
        body = ln.rstrip(b"\r\n")
        end = ln[len(body):]
        text = body.decode("ascii")
        parts = text.split(",")
        if len(parts) != 6 or '"' in text or len(set(parts[1:])) != 5 or any(not p for p in parts):
            raise Drift(f"{path}: unexpected row {body[:80]!r}")
        out.append((parts[0], parts[1:], body, end))
    return lines[0], out


def new_top5(picks, cand):
    """PREREG_21 4.3: the candidate first, then the base row without it, in its order (first 4)."""
    return [cand] + [c for c in picks if c != cand][:4]


def build_21(base_path, overrides):
    """Bytes of the 21 file: the base file's bytes with only overridden rows whose top-5 changes rewritten."""
    header, rows = read_base_lines(base_path)
    seen = {q for q, *_ in rows}
    missing = sorted(set(overrides) - seen)
    if missing:
        raise Drift(f"base file has no row for overridden questions {missing[:3]}")
    buf, changed = [header], []
    for qid, picks, body, end in rows:
        cand = overrides.get(qid)
        new = new_top5(picks, cand) if cand is not None else picks
        if new != picks:
            buf.append(",".join([qid] + new).encode("ascii") + end)
            changed.append({"question_id": qid, "base": picks, "new": new})
        else:
            buf.append(body + end)
    return b"".join(buf), changed


def write_overrides_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pool", "question_id", "candidate", "base_rank_of_candidate", "changed", "copy_slots",
                    "base_top5", "new_top5"])
        for r in rows:
            w.writerow([r["pool"], r["question_id"], r["candidate"], r["base_rank_of_candidate"] or "",
                        int(r["changed"]), " ".join(map(str, r["copy_slots"])), " ".join(r["base_top5"]),
                        " ".join(r["new_top5"])])


# ---------------------------------------------------------------- real-mode preconditions
def require_committed(rels):
    bad = [r for r in rels if not committed_clean(r)]
    if bad:
        raise Refuse(f"must be committed and clean first: {bad}")


def check_manifest(tests_dir, t1):
    """MANIFEST kind official, the four competition-1 files equal to it and each file's source_sha256 equal to the raw
    file the T1 checked (t02_checks.read_manifest)."""
    try:
        info = t2.read_manifest(tests_dir, "player", t1)
    except SystemExit as e:
        raise Refuse(f"data/test does not match MANIFEST / the raw T1: {e}")
    return info["manifest_sha256"]


def git_last_commit_time(rel_path):
    rc, out = git("log", "-1", "--format=%cI", "--", rel_path)
    if rc != 0 or not out:
        return None
    return datetime.fromisoformat(out.splitlines()[0].strip())


def git_row_on_main(rel_path, row):
    """(commit, time) of the first commit in main's first-parent history whose diff adds the row (a --no-ff merge
    counts at its own time); (None, None) when there is none."""
    rc, out = git("log", "main", "--first-parent", "--diff-merges=first-parent", "--reverse", "-S" + row,
                  "--format=%H %cI", "--", rel_path)
    if rc != 0 or not out:
        return None, None
    h, t = out.splitlines()[0].split()
    return h, datetime.fromisoformat(t)


def check_k5_and_sim():
    """PREREG_21 6 and 8: sim.json committed with every pass condition true, written by this very tool and last
    committed by K5-21; the rev 2 row of the registration reached main's first-parent history by K5-21."""
    r_sim = rel(SIM)
    require_committed((r_sim, DOC))
    sim = read_json(SIM)
    if not (sim.get("evaluation") or {}).get("passed"):
        raise NotUsed(f"{r_sim}: the simulation did not pass (PREREG_21 6): the layer is not used")
    if (sim.get("provenance") or {}).get("tool_sha256_lf") != sha256_lf(os.path.join(HERE, "t21_copy.py")):
        raise Refuse(f"{r_sim} was written by another tools/t21_copy.py (PREREG_21 6: the simulation checks this tool)")
    late = {}
    t_sim = git_last_commit_time(r_sim)
    if t_sim is None or t_sim > K5_21:
        late["sim"] = t_sim.isoformat() if t_sim else None
    h, t_rev2 = git_row_on_main(DOC, REV2_ROW)
    if t_rev2 is None or t_rev2 > K5_21:
        late["rev 2 row on main"] = t_rev2.isoformat() if t_rev2 else None
    if late:
        raise NotUsed(f"K5-21 {K5_21.isoformat()}: {late} committed / merged late or not at all (PREREG_21 8): not "
                      f"used")
    return {"sim": r_sim, "sim_sha256_lf": sha256_lf(SIM), "rev2_on_main": h, "rev2_on_main_at": t_rev2.isoformat(),
            "k5_21": K5_21.isoformat(), "k5_ok": True}


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
    return {"file": r, "verdict": t1.get("verdict"), "sha256": sha256_file(T1_JSON), "_t1": t1}


def base_from_log(log_path, root, real):
    if real and not committed_clean(rel(log_path)):
        raise Refuse(f"{rel(log_path)} must be committed and clean")
    if not os.path.isfile(log_path):
        raise Refuse(f"{log_path} missing: no upload logged yet")
    lg = read_json(log_path)
    fin = (lg.get("final_choice") or {}).get("player")
    if not fin:
        raise Refuse("final_choice.player is not set: run official_log.py final --comp player first "
                     "(docs/TESTDAY_10.md step 6 / docs/TESTDAY_06.md step 9); 21 is derived from the designated "
                     "file (PREREG_21 7 S2)")
    e = next((x for x in lg.get("uploads") or [] if x.get("id") == fin.get("id")), None)
    if e is None or e.get("comp") != "player" or e.get("status") != "uploaded":
        raise Refuse(f"final_choice.player #{fin.get('id')} is not an uploaded competition-1 entry")
    if e.get("test21"):
        raise Refuse(f"final_choice.player #{e['id']} is a 21 entry already; 21 is never built on 21")
    if fin.get("sha256") != e.get("sha256"):
        raise Drift("final_choice.player and its entry record different SHA-256")
    frel = e.get("file") or ""
    if real and os.path.dirname(frel) not in BASE_DIRS:
        raise Refuse(f"base file {frel} is not in one of {BASE_DIRS}")
    path = os.path.join(root, *frel.split("/"))
    if not os.path.isfile(path) or sha256_file(path) != e["sha256"]:
        raise Drift(f"{frel} is missing or not the logged bytes ({str(e.get('sha256'))[:12]}); restore with git -c "
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
            if args.root:
                raise Refuse("--root is for --dry-run only")
            self.root, self.tests, self.log, self.out = REPO, DATA_TEST, LOG, OUT
        else:
            if not args.root:
                raise Refuse("--dry-run needs --root S")
            self.root = os.path.abspath(args.root)
            if inside(self.root, REPO):
                raise Refuse(f"--root {self.root} is inside the repository")
            self.tests = os.path.join(self.root, "data", "test")
            self.log = os.path.join(self.root, "results", "official_scores.json")
            self.out = os.path.join(self.root, "results", "test21")
        self.mode = "real" if self.real else "dry-run"


def guard_out(path, args):
    if os.path.exists(path) and not (args.rerun and (args.reason or "").strip()):
        raise Refuse(f"{path} exists; rewrite only with --rerun --reason TEXT")


def write_not_used(ctx, reason, extra, stage):
    p = os.path.join(ctx.out, "not_used.json")
    if os.path.isfile(p):                          # review 10/20 MINOR 5: an earlier not_used.json is kept, not overwritten
        dst = os.path.join(ctx.out, "superseded", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_not_used.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(p, dst)
    write_json(p, dict({"format": FORMAT_NOT_USED, "registration": DOC, "stage": stage, "reason": reason}, **extra,
                       provenance=provenance(ctx.mode)))
    log(f"not used: {reason} -> {p}")
    return p


def preflight(ctx, stage="scan", args=None):
    rec = {}
    if ctx.real:
        require_committed(OWN)
        rec["t1"] = check_t1()                     # no T1 yet (before 11/04) is a refusal and writes nothing
        if not os.path.isfile(os.path.join(ctx.tests, "MANIFEST.json")):
            # review 10/20 I3: data/test is built (docs/TESTDAY.md step 2) only after a FAIL_OPENING was handled by
            # docs/TESTDAY_02_PASS.md; before it, a raw FAIL_OPENING must not switch the layer off for the season
            raise Refuse(f"{rel(os.path.join(ctx.tests, 'MANIFEST.json'))} missing: build data/test first "
                         f"(docs/TESTDAY.md step 2, after docs/TESTDAY_02_PASS.md when it applies)")
        try:
            if rec["t1"]["verdict"] not in ("PASS", "NORMALIZED"):
                raise NotUsed(f"T1 verdict {rec['t1']['verdict']!r}: PREREG_21 9, the layer is not used")
            rec["k5_and_sim"] = check_k5_and_sim()
        except NotUsed as e:
            write_not_used(ctx, str(e), {}, "preflight")
            raise
        rec["manifest_sha256"] = check_manifest(ctx.tests, rec["t1"].pop("_t1"))
    nu = os.path.join(ctx.out, "not_used.json")
    if os.path.isfile(nu):
        prev = read_json(nu)
        # PREREG_21 7 / 9: after a registered 'not used' nothing is rerun, except an apply on a CHANGED base (an apply
        # whose overrides all equalled the old base's rows may change rows of a new base); cmd_apply checks the base
        if not (stage == "apply" and prev.get("stage") == "apply" and args is not None and args.rerun):
            raise NotUsed(f"{rel(nu) if ctx.real else nu} exists: the layer is not used this season (PREREG_21 9)")
        rec["previous_not_used"] = prev
    return rec


def _q_rows(s):
    return [{"pool": p, "question_id": q, "candidate": d["override"], "copy_slots": d["copy_slots"],
             "copy_games": d["copy_games"]} for p, q, d in s["questions"] if d["override"] is not None]


def _conflicts(s):
    return [{"pool": p, "question_id": q, "source_candidates": d["source_candidates"], "copy_slots": d["copy_slots"]}
            for p, q, d in s["questions"] if d["conflict"]]


def cmd_scan(args):
    ctx = Ctx(args)
    pre = preflight(ctx)
    out = os.path.join(ctx.out, "scan", "scan.json")
    guard_out(out, args)
    s = scan_files(ctx.tests)
    over = _q_rows(s)
    rep = {"format": FORMAT_SCAN, "registration": DOC + " (S1, record only)", "mode": ctx.mode, "preflight": pre,
           "pools": s["pools"], "files": s["files"], "questions_overridable": len(over),
           "overridable_by_pool": dict(Counter(r["pool"] for r in over)), "overridable": over,
           "conflicts": _conflicts(s), "rerun_reason": args.reason if args.rerun else None,
           "provenance": provenance(ctx.mode)}
    write_json(out, rep)
    print(f"scan: {len(over)} questions overridable; copies " +
          ", ".join(f"{f['pool']} {f.get('copies', 0)}" for f in s["files"].values()) + f" -> {out}")
    if not over:
        write_not_used(ctx, "S1 scan: no question has an exact same-pool, same-colour copy (PREREG_21 7 S1)",
                       {"scan": rel(out) if ctx.real else out}, "scan")
        raise NotUsed("no question can be overridden")
    return rep


def cmd_apply(args):
    ctx = Ctx(args)
    pre = preflight(ctx, "apply", args)
    rdir = os.path.join(ctx.out, "player")
    rpath = os.path.join(rdir, "t21_copy_report.json")
    guard_out(rpath, args)
    if ctx.real and datetime.now(TW) > K6_21:
        raise Refuse(f"K6-21 {K6_21.isoformat()} has passed: no new apply (PREREG_21 8; the base stays final)")
    base = base_from_log(ctx.log, ctx.root, ctx.real)
    prev = pre.pop("previous_not_used", None)
    if prev is not None:
        pb = prev.get("base") or {}
        if (pb.get("entry_id"), pb.get("sha256")) == (base["entry_id"], base["sha256"]):
            raise NotUsed(f"not_used.json was written for this same base #{base['entry_id']}; PREREG_21 7 reruns an "
                          f"apply only on a changed base")
        pre["superseded_not_used"] = {"base": pb, "reason": prev.get("reason")}
    s = scan_files(ctx.tests)
    _h, base_rows = read_base_lines(base["path"])
    base_top = {q: picks for q, picks, _b, _e in base_rows}
    overrides = {r["question_id"]: r["candidate"] for r in _q_rows(s)}
    data, changed = build_21(base["path"], overrides)
    rows = []
    for r in _q_rows(s):
        picks = base_top.get(r["question_id"]) or []
        new = new_top5(picks, r["candidate"])
        rows.append(dict(r, base_top5=picks, new_top5=new, changed=new != picks,
                         base_rank_of_candidate=(picks.index(r["candidate"]) + 1) if r["candidate"] in picks
                         else None))
    rep = {"format": FORMAT_REPORT, "registration": DOC, "mode": ctx.mode, "method": METHOD, "preflight": pre,
           "base": {k: v for k, v in base.items() if k != "path"}, "pools": s["pools"], "files": s["files"],
           "questions_overridden": len(rows), "rows_changed": len(changed),
           "changed_by_pool": dict(Counter(r["pool"] for r in rows if r["changed"])),
           "overridden_by_pool": dict(Counter(r["pool"] for r in rows)),
           "overrides": rows, "conflicts": _conflicts(s), "k6_21": K6_21.isoformat(),
           "rerun_reason": args.reason if args.rerun else None}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    def supersede(names):
        """Move earlier outputs to results/test21/superseded/ (kept in git history; never deleted)."""
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
        why = ("no question has an exact same-pool, same-colour copy" if not rows else
               "every overridden question already has that candidate at top 1 in the base file (21 would equal the "
               "base)")
        rep["verdict"] = "not_used"
        rep["provenance"] = provenance(ctx.mode)
        rep["superseded"] = supersede([os.path.join("player", "submission_player.csv"),
                                       os.path.join("player", "overrides.csv"), "not_used.json"])
        write_json(rpath, rep)
        write_not_used(ctx, f"S2 apply: {why} (PREREG_21 7 S2)",
                       {"report": rel(rpath) if ctx.real else rpath,
                        "base": {k: base[k] for k in ("entry_id", "file", "sha256")}}, "apply")
        raise NotUsed(why)
    sub = os.path.join(rdir, "submission_player.csv")
    os.makedirs(rdir, exist_ok=True)
    earlier = supersede([os.path.join("player", "submission_player.csv"), os.path.join("player", "overrides.csv")])
    with open(sub, "wb") as f:
        f.write(data)
    write_overrides_csv(os.path.join(rdir, "overrides.csv"), rows)
    qs = [os.path.join(ctx.tests, Q_FILES[p]) for p in POOLS]
    cs = [os.path.join(ctx.tests, C_FILES[p]) for p in POOLS]
    errors = check_submission.check("player", sub, qs, cs, official=True)
    rep["check_submission"] = {"mode": "official", "errors": errors[:20]}
    rep["outputs_sha256"] = {"submission_player.csv": sha256_file(sub),
                             "overrides.csv": sha256_file(os.path.join(rdir, "overrides.csv"))}
    rep["verdict"] = "error" if errors else "used"
    rep["superseded"] = earlier + ([] if errors else supersede(["not_used.json"]))
    rep["provenance"] = provenance(ctx.mode)
    write_json(rpath, rep)
    if errors:
        raise Drift(f"check_submission --official failed: {errors[:3]}")
    print(f"apply: 21 written, {len(rows)} questions overridden, {len(changed)} rows changed vs base #"
          f"{base['entry_id']} ({base['file']}) -> {sub}")
    return rep


def cmd_status(args):
    ctx = Ctx(args)
    lines = []
    for name in ("not_used.json", os.path.join("scan", "scan.json"), os.path.join("player", "t21_copy_report.json")):
        p = os.path.join(ctx.out, name)
        if os.path.isfile(p):
            j = read_json(p)
            lines.append(f"{name}: " + (j.get("reason") or f"verdict {j.get('verdict')}, overridden "
                                        f"{j.get('questions_overridden', j.get('questions_overridable'))}, rows "
                                        f"changed {j.get('rows_changed')}"))
        else:
            lines.append(f"{name}: (none)")
    for what, k in (("K5-21", K5_21), ("K6-21", K6_21)):
        left = k - datetime.now(TW)
        lines.append(f"{what} {k.isoformat()}: " + ("passed" if left.total_seconds() <= 0 else f"{left} left"))
    print("\n".join(lines))
    return lines


# ---------------------------------------------------------------- simulation (PREREG_21 section 6)
def _h(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def sgf_of(pairs):
    return "(;SZ[19]" + "".join(f";{c}[{p}]" for c, p in pairs) + ")"


def sym_sgf(sgf, s):
    """The game with every point mapped by symmetry s (p02_moves.SYM_TABLE)."""
    return sgf_of((c, t2.SYM[s][p]) for c, p in t2.MOVE_RE.findall(sgf))


def drop_last_move(sgf):
    return sgf_of(t2.MOVE_RE.findall(sgf)[:-1])


def first_moves(sgf, n):
    return sgf_of(t2.MOVE_RE.findall(sgf)[:n])


def load_fit_black(train_dir, split_csv=SPLIT_CSV):
    """{player: [(gid, sgf)]} of the fit players' black games (split v1), each list in the sim hash order."""
    keep = c10data.players_for(c10data.read_split(split_csv), "fit")
    by = defaultdict(list)
    csv.field_size_limit(10 ** 8)
    for lv in c10data.LEVELS:
        with open(guard_read(os.path.join(train_dir, f"train_{lv}.csv")), newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            if next(rd) != list(c10data.COLUMNS):
                raise Drift(f"unexpected header in train_{lv}.csv")
            for pid, gid, _rank, colour, sgf in rd:
                if pid in keep and colour == "B":
                    by[pid].append((gid, sgf))
    for pid, games in by.items():
        games.sort(key=lambda g, pid=pid: _h(SIM_SALT, "game", pid, g[0]))
    return by


def make_fake(train_dir, no_copies=False, split_csv=SPLIT_CSV):
    """PREREG_21 6: two official-shaped pools from fit players. Returns (tree {'questions': {pool: rows}, 'candidates':
    {pool: rows}}, expected {qid: {...}}, base [(qid, [5 candidates])])."""
    by = load_fit_black(train_dir, split_csv)
    elig = sorted((p for p, g in by.items() if len(g) >= SIM_LIBRARY + SIM_HELD), key=lambda p: _h(SIM_SALT, "player", p))
    if len(elig) < 2 * SIM_PER_POOL:
        raise Refuse(f"only {len(elig)} fit players with >= {SIM_LIBRARY + SIM_HELD} black games")
    kinds = [k for k, n in SIM_KINDS for _ in range(n)]
    assert len(kinds) == SIM_PER_POOL
    prefix = {"public": "pub_", "private": "priv_"}
    pools = {}
    for pi, pool in enumerate(POOLS):
        players = elig[pi * SIM_PER_POOL:(pi + 1) * SIM_PER_POOL]
        cands = [f"{prefix[pool]}p_{i + 1:04d}" for i in range(SIM_PER_POOL)]
        lib = {c: [list(g) for g in by[p][:SIM_LIBRARY]] for c, p in zip(cands, players)}
        held = {c: [s for _g, s in by[p][SIM_LIBRARY:SIM_LIBRARY + SIM_HELD]] for c, p in zip(cands, players)}
        pools[pool] = {"players": dict(zip(cands, players)), "cands": cands, "lib": lib, "held": held}
    expected, base, questions = {}, [], {}
    for pool in POOLS:
        P = pools[pool]
        other = pools["private" if pool == "public" else "public"]
        qids = [f"{prefix[pool]}{j + 1:04d}" for j in range(SIM_PER_POOL)]
        truth = dict(zip(sorted(qids, key=lambda q: _h(SIM_SALT, "truth", pool, q)), P["cands"]))
        order = sorted(range(SIM_PER_POOL), key=lambda j: _h(SIM_SALT, "kind", pool, j))
        kind_of = {j: ("N" if no_copies else kinds[r]) for r, j in enumerate(order)}
        rows = []
        for j, qid in enumerate(qids):
            c = truth[qid]
            k, size = kind_of[j], SIM_SIZES[j % 4]
            held, lib = P["held"][c], P["lib"][c]
            if k == "N":
                games = held[:size]
            elif k == "C1":
                games = [lib[0][1]] + held[:size - 1]
            elif k == "CH":
                m = max(1, size // 2)
                games = [s for _g, s in lib[:m]] + held[:size - m]
            elif k == "CA":
                games = [s for _g, s in lib[:size]]
            elif k == "XS":
                games = [sym_sgf(lib[0][1], 1 + j % 7)] + held[:size - 1]
            elif k == "XT":
                long_ = next(s for _g, s in lib if len(t2.MOVE_RE.findall(s)) >= MIN_MOVES + 1)
                games = [drop_last_move(long_)] + held[:size - 1]
            elif k == "XC":
                oc = min((x for x in P["cands"] if x != c), key=lambda x: _h(SIM_SALT, "xc", pool, j, x))
                games = [lib[0][1], P["lib"][oc][0][1]] + held[:size - 2]
            elif k == "XP":
                oc = min(other["cands"], key=lambda x: _h(SIM_SALT, "xp", pool, j, x))
                games = [other["lib"][oc][0][1]] + held[:size - 1]
            elif k == "XU":
                short = first_moves(lib[1][1], SIM_SHORT)
                lib[1][1] = short                     # the candidate's library holds the 80-move game too
                games = [short] + held[:size - 1]
            else:
                raise AssertionError(k)
            rows.append((qid, games))
            expected[qid] = {"pool": pool, "kind": k, "truth": c, "size": size}
            ranked = sorted(P["cands"], key=lambda x: _h(SIM_SALT, "base", qid, x))
            top = [x for x in ranked if x != c][:5]
            r = int(_h(SIM_SALT, "base-r", qid), 16) % 7
            if r < 5:
                top[r] = c
            base.append((qid, top))
            expected[qid]["base_rank_of_truth"] = r + 1 if r < 5 else None
        questions[pool] = rows
    cand_rows = {}
    for pool in POOLS:
        P, rows, n = pools[pool], [], 0
        for c in P["cands"]:
            for _gid, sgf in P["lib"][c]:
                n += 1
                rows.append((f"{prefix[pool]}g{n:06d}", c, sgf, "B"))
        cand_rows[pool] = rows
    return {"questions": questions, "candidates": cand_rows}, expected, base


def write_fake_tree(root, tree, base):
    tdir = os.path.join(root, "data", "test")
    os.makedirs(tdir, exist_ok=True)
    man = {"kind": "t21-fake", "files": {}}
    for pool in POOLS:
        p = os.path.join(tdir, Q_FILES[pool])
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(t2.Q_COLS)
            for qid, games in tree["questions"][pool]:
                w.writerow([qid, len(games)] + games + [""] * (20 - len(games)) + ["B"] * len(games)
                           + [""] * (20 - len(games)))
        man["files"][Q_FILES[pool]] = {"sha256": sha256_file(p)}
        p = os.path.join(tdir, C_FILES[pool])
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(t2.C_COLS)
            for row in tree["candidates"][pool]:
                w.writerow(row)
        man["files"][C_FILES[pool]] = {"sha256": sha256_file(p)}
    write_json(os.path.join(tdir, "MANIFEST.json"), man)
    brel = "results/test10/player/submission_player.csv"
    bpath = os.path.join(root, *brel.split("/"))
    os.makedirs(os.path.dirname(bpath), exist_ok=True)
    with open(bpath, "wb") as f:
        f.write(HEADER + b"\n" + b"".join(",".join([q] + top).encode("ascii") + b"\n" for q, top in base))
    sha = sha256_file(bpath)
    lg = {"format": "c10 official_scores v1", "teams": {"rank": "TEAM_11248", "player": "TEAM_11249"},
          "baselines_public": {"rank": 0.4631, "player": 0.2845},
          "final_choice": {"rank": None, "player": {"id": 1, "sha256": sha, "file": brel, "method": "fake base",
                                                    "basis": "t21 fake", "set_at": now_iso()}},
          "uploads": [{"id": 1, "comp": "player", "status": "uploaded", "file": brel, "sha256": sha,
                       "method": "fake base", "uploaded_at": "2026-11-20T10:00:00+08:00"}]}
    write_json(os.path.join(root, "results", "official_scores.json"), lg)
    return bpath


def cmd_fake(args):
    if not args.root or inside(args.root, REPO):
        raise Refuse("fake needs --root S outside the repository")
    tree, expected, base = make_fake(TRAIN_DIR, no_copies=args.no_copies)
    root = os.path.abspath(args.root)
    write_fake_tree(root, tree, base)
    write_json(os.path.join(root, "t21_expected.json"), expected)
    print(f"fake tree in {root}: {len(expected)} questions" + (" (no copies)" if args.no_copies else ""))
    return expected


def evaluate(expected, report, base_path, out_path):
    """PREREG_21 6 pass conditions 1-3."""
    over = {r["question_id"]: r["candidate"] for r in report.get("overrides") or []}
    want = {q for q, e in expected.items() if e["kind"] in SIM_EXPECT_OVERRIDE}
    by_kind = defaultdict(Counter)
    for q, e in expected.items():
        c = by_kind[e["kind"]]
        c["questions"] += 1
        c["overridden"] += q in over
        c["overridden_correct"] += over.get(q) == e["truth"]
        c["truth_already_top1"] += e["base_rank_of_truth"] == 1
    missed = sorted(want - set(over))
    wrong = sorted(q for q in want & set(over) if over[q] != expected[q]["truth"])
    spurious = sorted(set(over) - want)
    _h1, brow = read_base_lines(base_path)
    _h2, orow = read_base_lines(out_path)
    bmap = {q: (picks, body, end) for q, picks, body, end in brow}
    omap = {q: (picks, body, end) for q, picks, body, end in orow}
    row_wrong = sorted(q for q in want if omap.get(q, ([],))[0] != new_top5(bmap[q][0], expected[q]["truth"]))
    with open(base_path, "rb") as f:
        bl = f.read().splitlines(keepends=True)
    with open(out_path, "rb") as f:
        ol = f.read().splitlines(keepends=True)
    same_shape = len(bl) == len(ol) and bl[0] == ol[0] and [q for q, *_r in brow] == [q for q, *_r in orow]
    untouched = [(a, b) for a, b, (q, *_r) in zip(bl[1:], ol[1:], brow) if q not in want]
    byte_untouched = same_shape and all(a == b for a, b in untouched)
    cond = {"1_expected_overrides_all_correct_rows": not missed and not wrong and not row_wrong,
            "2_no_override_where_not_expected": not spurious,
            "3_untouched_rows_byte_equal_and_format_ok": byte_untouched
            and not (report.get("check_submission") or {}).get("errors")}
    return {"conditions": cond, "expected_override_questions": len(want), "overridden": len(over),
            "untouched_rows_compared": len(untouched), "missed": missed[:20], "wrong_candidate": wrong[:20],
            "wrong_row": row_wrong[:20], "spurious": spurious[:20],
            "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())}}


def loo_library(tree):
    """Record only (PREREG_21 6): among the simulated library games, groups of rows with the same full key and move
    count, and such groups across candidates."""
    groups = defaultdict(list)
    for pool in POOLS:
        for _gid, cand, sgf, _c in tree["candidates"][pool]:
            pts, _cols = parse_game(sgf)
            groups[(full_key(pts), len(pts))].append(cand)
    multi = [v for v in groups.values() if len(v) > 1]
    return {"library_games": sum(len(v) for v in groups.values()), "groups_with_two_or_more_rows": len(multi),
            "groups_across_candidates": sum(1 for v in multi if len(set(v)) > 1)}


def _dry(root, rerun=False, reason=None):
    return argparse.Namespace(dry_run=True, root=root, rerun=rerun, reason=reason)


def cmd_sim(args):
    if args.dry_run:
        raise Refuse("sim writes the committed design record; it builds its own temporary dry-run trees")
    out = SIM
    guard_out(out, args)
    t0 = time.time()
    tree, expected, base = make_fake(TRAIN_DIR)
    with tempfile.TemporaryDirectory(prefix="c10_t21_sim_") as root:
        write_fake_tree(root, tree, base)
        bpath = os.path.join(root, "results", "test10", "player", "submission_player.csv")
        try:
            cmd_scan(_dry(root))
        except NotUsed as e:
            raise Drift(f"the simulated scan found nothing: {e}")
        rep = cmd_apply(_dry(root))
        ev = evaluate(expected, rep, bpath, os.path.join(root, "results", "test21", "player", "submission_player.csv"))
        outs = dict(rep["outputs_sha256"], base_submission_player_csv=sha256_file(bpath),
                    scan_questions_overridable=read_json(os.path.join(root, "results", "test21", "scan",
                                                                      "scan.json"))["questions_overridable"])
        summary = {"questions_overridden": rep["questions_overridden"], "rows_changed": rep["rows_changed"],
                   "overridden_by_pool": rep["overridden_by_pool"], "changed_by_pool": rep["changed_by_pool"],
                   "conflicts": len(rep["conflicts"]), "files": rep["files"], "pools": rep["pools"]}
    tree0, _exp0, base0 = make_fake(TRAIN_DIR, no_copies=True)
    with tempfile.TemporaryDirectory(prefix="c10_t21_sim0_") as root:
        write_fake_tree(root, tree0, base0)
        code, why = 0, None
        try:
            cmd_apply(_dry(root))
        except NotUsed as e:
            code, why = 3, str(e)
        nu = os.path.join(root, "results", "test21", "not_used.json")
        sub0 = os.path.join(root, "results", "test21", "player", "submission_player.csv")
        zero = {"exit": code, "reason": why, "not_used_written": os.path.isfile(nu),
                "submission_written": os.path.isfile(sub0)}
    ev["conditions"]["4_no_copy_control_not_used"] = (zero["exit"] == 3 and zero["not_used_written"]
                                                      and not zero["submission_written"])
    ev["passed"] = all(ev["conditions"].values())
    rec = {"format": FORMAT_SIM, "registration": DOC + " section 6",
           "players": "fit only (split v1), black games, >= 120 black games; no validation player, no data/local",
           "questions": len(expected), "kinds_per_pool": dict(SIM_KINDS), "apply": summary, "outputs_sha256": outs,
           "evaluation": ev, "no_copy_control": zero, "library_record_only": loo_library(tree),
           "seconds": round(time.time() - t0, 1), "rerun_reason": args.reason if args.rerun else None,
           "provenance": provenance("design")}
    write_json(out, rec)
    print(json.dumps({"passed": ev["passed"], "conditions": ev["conditions"], "by_kind": ev["by_kind"],
                      "no_copy_control": zero}, indent=1))
    if not ev["passed"]:
        raise Drift("simulation pass conditions not met (PREREG_21 6): a program error to fix")
    return rec


# ---------------------------------------------------------------- CLI
def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="t21_copy.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "apply", "status", "sim", "fake"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--root")
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason")
        if name == "fake":
            p.add_argument("--no-copies", action="store_true")
    args = ap.parse_args(argv)
    if args.rerun and not (args.reason or "").strip():
        ap.error("--rerun needs --reason TEXT")
    fn = {"scan": cmd_scan, "apply": cmd_apply, "status": cmd_status, "sim": cmd_sim, "fake": cmd_fake}[args.cmd]
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
