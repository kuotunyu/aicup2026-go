"""Test-day bundle (PREREG_02 rev 5 section 9; results/test02/design/spec.md 3.1). Stdlib only.

Official test files in, one Colab zip out, Colab result zips back into results/test02/.

    PY tools/t02_test_bundle.py ingest --src <zip-or-dir> [--src ...] [--dest data/test/raw]
    PY tools/t02_test_bundle.py build  --kind official|rehearsal|smoke [--tests-dir data/test/raw]
                                       [--out-dir data/test] [--zip-out data/colab_bundle/C10_test_v1.zip]
                                       [--checks-dir results/test02/checks] [--root S]
                                       [--allow-fail-opening --reason TEXT]
    PY tools/t02_test_bundle.py unpack --zip <test02_{rank,player}_results.zip> --comp rank|player
                                       [--results-dir results/test02] [--tests-dir data/test] [--allow-failed]
    Every command: [--rerun --reason TEXT] to replace outputs that already exist (never silently).

ingest  finds the six official file names (recursively in folders, and inside zip files found there or
        given directly), copies them byte for byte to data/test/raw/ and writes SOURCE.json. It refuses when
        a name is missing or when two copies of one name differ. The real ingest (into the repo's
        data/test/raw) also refuses sources that are this kit's own files: a MANIFEST.json of a test bundle
        (zip member or file; any kind), an ingest SOURCE.json or a t02_fake_tests FAKE_TREE.json. Those are
        rehearsal / smoke / fake trees built from validation games, never the download from AIdea (review
        finding 2026-09-26); a scratch ingest (--root / --dest outside the repo) only notes them.
build   requires results/test02/checks/{rank,player}/t1.json (T1 on the raw files, written by
        tools/t02_checks.py): verdict PASS or NORMALIZED (FAIL_OPENING only with --allow-fail-opening; the
        registered consequence then applies downstream), the recorded input SHA-256 equal to the raw files,
        and only whitelisted transforms in "normalize"; --kind official also needs the full T1 (complete,
        T1.1-T1.5) written in real mode on the raw files (stage 'raw'). It then writes data/test/<name>
        (normalized copies,
        otherwise byte copies), data/test/MANIFEST.json, re-runs T1.1-T1.3 on data/test through
        tools/t02_checks.py (must be PASS; written to <checks-dir>/built/<comp>/t1.json, in real mode
        results/test02/checks/built/<comp>/t1.json), writes <checks-dir>/bundle_build.json, runs its own quick structure
        check, writes the zip (members tests/<6 names> + tests/MANIFEST.json) and prints its SHA-256.
unpack  extracts a notebook result zip (members relative to results/test02/, all under <comp>/) after
        checking: no absolute, '..' or backslash member paths, no member outside <comp>/, no existing
        target file, run_log status 'completed' (else only with --allow-failed, and only outside the repo),
        config.test_bundle.manifest_sha256 equal to the local MANIFEST.json, and in the real results dir
        config.smoke == config.rehearsal == false. It prints every run's status in run_log.json and records
        them; in the repo's results dirs it refuses a zip whose run_log has an earlier run that failed
        (seam_failed, repro_failed, readonly_changed, final_check_failed): the notebook was run again in the
        same Drive folder without the registered rerun (PREREG_02 rev 5 9.2, rev 6 10.10: rename the Drive
        folder, new runtime), so this run's outputs are not used (review finding 2026-09-26).

Modes: --kind official is the real run and needs the exact repo paths (no --root). --kind rehearsal/smoke
and --root S (a scratch tree with the repo's layout: S/data/test, S/results/test02, ...) must stay outside
the repository. Nothing is ever written inside results/prereg01, results/prereg02 or data/local.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))

_NEEDLE = "ans" + "wers"   # validation label files are never opened (spelled in halves on purpose)

RANK_FILES = {"public": "rank_prediction_test_public.csv", "private": "rank_prediction_test_private.csv"}
PLAYER_Q_FILES = {"public": "player_identification_test_public.csv",
                  "private": "player_identification_test_private.csv"}
PLAYER_C_FILES = {"public": "player_identification_candidates_public.csv",
                  "private": "player_identification_candidates_private.csv"}
POOLS = ("public", "private")
COMP_FILES = {"rank": (RANK_FILES["public"], RANK_FILES["private"]),
              "player": (PLAYER_Q_FILES["public"], PLAYER_Q_FILES["private"],
                         PLAYER_C_FILES["public"], PLAYER_C_FILES["private"])}
ALL_NAMES = COMP_FILES["rank"] + COMP_FILES["player"]
QUESTION_NAMES = set(RANK_FILES.values()) | set(PLAYER_Q_FILES.values())
CANDIDATE_NAMES = set(PLAYER_C_FILES.values())

MANIFEST_NAME = "MANIFEST.json"
MANIFEST_FORMAT = "c10-test02 manifest v1"
SOURCE_NAME = "SOURCE.json"
ZIP_NAME = "C10_test_v1.zip"
ZIP_DIR = "tests"
ZIP_DATE = (2026, 1, 1, 0, 0, 0)   # fixed member time: the same files always give the same zip bytes
KINDS = ("official", "rehearsal", "smoke")
T1_OK = ("PASS", "NORMALIZED")
T1_ALL = ("PASS", "NORMALIZED", "FAIL_STRUCT", "FAIL_OPENING")
RESULT_ZIPS = {"rank": "test02_rank_results.zip", "player": "test02_player_results.zip"}
RUN_STATUSES = ("completed", "seam_failed", "repro_failed", "readonly_changed")
# statuses of a run whose outputs are never used; a later run in the same Drive folder is not the registered rerun
FAILED_RUN_STATUSES = ("seam_failed", "repro_failed", "readonly_changed", "final_check_failed")
# this kit's own records: a source holding one of them is a rehearsal / smoke / fake tree, not the AIdea download
OWN_RECORDS = {"MANIFEST.json": "c10-test02 manifest v1", "SOURCE.json": "c10-test02 source v1",
               "FAKE_TREE.json": "c10-test02 fake tree v1"}

# README (tutorial 359e06f) column order
QUESTION_COLUMNS = (["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, 21)]
                    + [f"color_{i}" for i in range(1, 21)])
CANDIDATE_COLUMNS = ["game_id", "player_id", "sgf_content", "color"]
N_SLOTS = 20
MIN_GAMES, MAX_GAMES = 5, 20
N_QUESTIONS = 400
N_PLAYERS = 400
GAMES_PER_PLAYER = 100
Q_PREFIX = {"public": "pub_", "private": "priv_"}
P_PREFIX = {"public": "pub_p_", "private": "priv_p_"}
G_PREFIX = {"public": "pub_", "private": "priv_"}

# the only transforms T1 may ask for (PREREG_02 rev 5 section 9 item 6; rev 8 adds the last one: the official data
# page says color_1..color_20 are always B, so the slots after num_games may carry a colour letter with no game)
NORMALIZE_WHITELIST = ("strip_bom", "crlf_to_lf", "upper_colour", "reorder_columns_to_readme",
                       "clear_slots_after_num_games_if_empty_whitespace",
                       "clear_colour_after_num_games_if_sgf_empty")
BYTE_TRANSFORMS = ("strip_bom", "crlf_to_lf")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class BundleError(SystemExit):
    """Refusal: message on stderr, exit code 1, no traceback."""

    def __init__(self, message):
        super().__init__(f"t02_test_bundle: {message}")


# ================================================================ small helpers
def refuse_forbidden(path):
    if path is not None and _NEEDLE in os.path.basename(str(path)).lower():
        raise BundleError(f"refusing to open {path!r}: test-day tools never read validation label files")
    return path


def norm(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def inside(path, root):
    p, r = norm(path), norm(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def same_path(a, b):
    return norm(a) == norm(b)


READONLY = (os.path.join(REPO, "results", "prereg01"), os.path.join(REPO, "results", "prereg02"),
            os.path.join(REPO, "data", "local"))


def check_writable(path):
    for ro in READONLY:
        if inside(path, ro):
            raise BundleError(f"refusing to write {path!r}: it is inside the read-only {ro!r}")
    return path


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(refuse_forbidden(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args):
    try:
        r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None, ""
    return r.returncode, r.stdout.strip()


def tool_record():
    rc, head = git("rev-parse", "HEAD")
    return {"tool": "tools/t02_test_bundle.py", "tool_sha256": sha256_text_file(os.path.abspath(__file__)),
            "sha256_rule": "tool file with CRLF folded to LF", "git_head": head if rc == 0 else None,
            "python": sys.version.split()[0]}


def write_json(path, obj):
    check_writable(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def write_bytes(path, data):
    check_writable(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def read_json(path):
    with open(refuse_forbidden(path), encoding="utf-8") as f:
        return json.load(f)


def require_fresh(paths, rerun, reason):
    """Existing outputs are replaced only with --rerun --reason (the reason goes into the record)."""
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not rerun:
        raise BundleError(f"outputs already exist: {existing[:4]}; pass --rerun --reason TEXT to replace them")
    if rerun and not (reason or "").strip():
        raise BundleError("--rerun needs --reason TEXT (it is stored in the record)")
    return existing


def run_status(run_log):
    """run_log.json status: a top-level 'status', else the last run's (prereg02 layout)."""
    if isinstance(run_log, dict):
        if isinstance(run_log.get("status"), str):
            return run_log["status"]
        runs = run_log.get("runs")
        if isinstance(runs, list) and runs and isinstance(runs[-1], dict):
            return runs[-1].get("status")
    return None


def run_history(run_log):
    """[(run number, status)] of every run in run_log.json (prereg02 layout: 'runs' list of dicts)."""
    runs = run_log.get("runs") if isinstance(run_log, dict) else None
    out = []
    for i, r in enumerate(runs if isinstance(runs, list) else [], start=1):
        out.append((r.get("run", i) if isinstance(r, dict) else i, r.get("status") if isinstance(r, dict) else None))
    return out


def earlier_failures(run_log):
    """Runs before the last one whose status says their outputs are never used (FAILED_RUN_STATUSES)."""
    return [(n, s) for n, s in run_history(run_log)[:-1] if s in FAILED_RUN_STATUSES]


def repo_paths(root=None):
    base = REPO if root is None else root
    return {"raw": os.path.join(base, "data", "test", "raw"), "tests": os.path.join(base, "data", "test"),
            "zip": os.path.join(base, "data", "colab_bundle", ZIP_NAME),
            "checks": os.path.join(base, "results", "test02", "checks"),
            "results": os.path.join(base, "results", "test02")}


def check_root(root):
    if root is not None and inside(root, REPO):
        raise BundleError(f"--root {root!r} is inside the repository; a scratch tree must be outside it")
    return root


# ================================================================ csv reading (structure only)
def read_table(path):
    """(header, rows as lists, raw bytes) of a CSV; BOM is kept in the bytes but not in the header."""
    csv.field_size_limit(10**9)
    with open(refuse_forbidden(path), "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        return [], [], raw
    return rows[0], rows[1:], raw


def quick_structure(tests_dir, comp):
    """Structure-only subset of T1.1-T1.3 (the notebooks' quick preflight). No game content is looked at
    beyond empty / non-empty cells, so it may run on validation-derived files. Returns a list of problems."""
    errs = []
    qfiles = RANK_FILES if comp == "rank" else PLAYER_Q_FILES
    seen_q = {}
    pools = {}
    for pool in POOLS:
        name = qfiles[pool]
        path = os.path.join(tests_dir, name)
        if not os.path.exists(path):
            errs.append(f"{name}: missing")
            continue
        header, rows, raw = read_table(path)
        if raw.startswith(b"\xef\xbb\xbf"):
            errs.append(f"{name}: starts with a BOM")
        if sorted(header) != sorted(QUESTION_COLUMNS) or len(header) != len(QUESTION_COLUMNS):
            errs.append(f"{name}: columns differ from question_id, num_games, sgf_1..20, color_1..20")
            continue
        col = {c: i for i, c in enumerate(header)}
        if len(rows) != N_QUESTIONS:
            errs.append(f"{name}: {len(rows)} rows, expected {N_QUESTIONS}")
        for r in rows:
            if len(r) != len(header):
                errs.append(f"{name}: a row with {len(r)} fields")
                break
            q = r[col["question_id"]]
            if not q.startswith(Q_PREFIX[pool]):
                errs.append(f"{name}: question id {q!r} does not start with {Q_PREFIX[pool]!r}")
            if q in seen_q:
                errs.append(f"{name}: question id {q!r} repeated (also in {seen_q[q]})")
            seen_q[q] = name
            ng = r[col["num_games"]]
            if not ng.isdigit() or not MIN_GAMES <= int(ng) <= MAX_GAMES:
                errs.append(f"{name}: {q} num_games {ng!r} is not an integer in {MIN_GAMES}..{MAX_GAMES}")
                continue
            n = int(ng)
            for i in range(1, N_SLOTS + 1):
                s, c = r[col[f"sgf_{i}"]], r[col[f"color_{i}"]]
                if i <= n and (not s.strip() or c not in ("B", "W")):
                    errs.append(f"{name}: {q} slot {i} <= num_games needs an SGF and colour B/W")
                    break
                if i > n and (s != "" or c != ""):
                    errs.append(f"{name}: {q} slot {i} > num_games is not empty")
                    break
        if len(errs) > 50:
            return errs
    if comp == "player":
        for pool in POOLS:
            name = PLAYER_C_FILES[pool]
            path = os.path.join(tests_dir, name)
            if not os.path.exists(path):
                errs.append(f"{name}: missing")
                continue
            header, rows, raw = read_table(path)
            if raw.startswith(b"\xef\xbb\xbf"):
                errs.append(f"{name}: starts with a BOM")
            if sorted(header) != sorted(CANDIDATE_COLUMNS) or len(header) != len(CANDIDATE_COLUMNS):
                errs.append(f"{name}: columns differ from game_id, player_id, sgf_content, color")
                continue
            col = {c: i for i, c in enumerate(header)}
            if len(rows) != N_PLAYERS * GAMES_PER_PLAYER:
                errs.append(f"{name}: {len(rows)} rows, expected {N_PLAYERS * GAMES_PER_PLAYER}")
            if any(len(r) != len(header) for r in rows):
                errs.append(f"{name}: rows with a wrong number of fields")
                continue
            per = Counter(r[col["player_id"]] for r in rows)
            games = Counter(r[col["game_id"]] for r in rows)
            if len(per) != N_PLAYERS:
                errs.append(f"{name}: {len(per)} distinct player_id, expected {N_PLAYERS}")
            off = [p for p, k in per.items() if k != GAMES_PER_PLAYER]
            if off:
                errs.append(f"{name}: {len(off)} candidates without exactly {GAMES_PER_PLAYER} games")
            if any(not p.startswith(P_PREFIX[pool]) for p in per):
                errs.append(f"{name}: player_id not starting with {P_PREFIX[pool]!r}")
            if any(k > 1 for k in games.values()):
                errs.append(f"{name}: repeated game_id")
            if any(not g.startswith(G_PREFIX[pool]) for g in games):
                errs.append(f"{name}: game_id not starting with {G_PREFIX[pool]!r}")
            if any(r[col["color"]] not in ("B", "W") for r in rows):
                errs.append(f"{name}: colour values other than B/W")
            if any(not r[col["sgf_content"]].strip() for r in rows):
                errs.append(f"{name}: empty sgf_content")
            pools[pool] = (set(per), set(games))
        if len(pools) == 2:
            if pools["public"][0] & pools["private"][0]:
                errs.append("candidate pools share player_id")
            if pools["public"][1] & pools["private"][1]:
                errs.append("candidate pools share game_id")
    return errs


def data_rows(path):
    return len(read_table(path)[1])


# ================================================================ normalization (T1 whitelist only)
def _terminator(text):
    first = text.find("\n")
    return "\r\n" if first > 0 and text[first - 1] == "\r" else "\n"


def _serialize(rows, term):
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator=term, quoting=csv.QUOTE_MINIMAL)
    w.writerows(rows)
    return buf.getvalue()


def normalize_bytes(name, raw, transforms):
    """Apply the requested whitelisted transforms to one file; returns (bytes, [transforms that changed it]).
    CSV-level transforms rewrite the file only after proving that re-writing the unchanged rows gives back
    the original bytes, so nothing but the requested cells, column order or line ends can change."""
    unknown = [t for t in transforms if t not in NORMALIZE_WHITELIST]
    if unknown:
        raise BundleError(f"{name}: normalization {unknown} is not in the T1 whitelist {list(NORMALIZE_WHITELIST)}")
    changed = []
    data = raw
    if "strip_bom" in transforms and data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
        changed.append("strip_bom")
    if "crlf_to_lf" in transforms and b"\r\n" in data:
        data = data.replace(b"\r\n", b"\n")
        changed.append("crlf_to_lf")
    csv_ts = [t for t in NORMALIZE_WHITELIST if t in transforms and t not in BYTE_TRANSFORMS]
    if not csv_ts:
        return data, changed
    bom = data.startswith(b"\xef\xbb\xbf")
    text = data[3:].decode("utf-8") if bom else data.decode("utf-8")
    term = _terminator(text)
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        raise BundleError(f"{name}: empty file")
    if _serialize(rows, term) != text:
        raise BundleError(f"{name}: re-writing the unchanged rows does not give back the same bytes (quoting "
                          f"or a missing final line end); cannot normalize without other changes")
    header, body = rows[0], [list(r) for r in rows[1:]]
    readme = QUESTION_COLUMNS if name in QUESTION_NAMES else CANDIDATE_COLUMNS
    if sorted(header) != sorted(readme) or len(header) != len(readme):
        raise BundleError(f"{name}: columns are not the README set; not a whitelisted normalization")
    col = {c: i for i, c in enumerate(header)}
    if any(len(r) != len(header) for r in body):
        raise BundleError(f"{name}: rows with a wrong number of fields; not a whitelisted normalization")
    new_header = header
    if "upper_colour" in csv_ts:
        cols = [col[c] for c in header if c == "color" or c.startswith("color_")]
        hit = False
        for r in body:
            for i in cols:
                if r[i] in ("b", "w"):
                    r[i] = r[i].upper()
                    hit = True
        if hit:
            changed.append("upper_colour")
    if "clear_slots_after_num_games_if_empty_whitespace" in csv_ts and name in QUESTION_NAMES:
        hit = False
        for r in body:
            ng = r[col["num_games"]].strip()
            if not ng.isdigit():
                continue          # left for the re-check to reject
            for i in range(int(ng) + 1, N_SLOTS + 1):
                for c in (f"sgf_{i}", f"color_{i}"):
                    v = r[col[c]]
                    if v != "" and v.strip() == "":
                        r[col[c]] = ""
                        hit = True
        if hit:
            changed.append("clear_slots_after_num_games_if_empty_whitespace")
    if "clear_colour_after_num_games_if_sgf_empty" in csv_ts and name in QUESTION_NAMES:
        hit = False
        for r in body:
            ng = r[col["num_games"]].strip()
            if not ng.isdigit():
                continue          # left for the re-check to reject
            for i in range(int(ng) + 1, N_SLOTS + 1):
                c = col[f"color_{i}"]
                if r[col[f"sgf_{i}"]].strip() == "" and r[c] in ("B", "W", "b", "w"):
                    r[c] = ""
                    hit = True
        if hit:
            changed.append("clear_colour_after_num_games_if_sgf_empty")
    if "reorder_columns_to_readme" in csv_ts and header != readme:
        idx = [col[c] for c in readme]
        body = [[r[i] for i in idx] for r in body]
        new_header = list(readme)
        changed.append("reorder_columns_to_readme")
    if not any(t in changed for t in csv_ts):
        return data, changed
    out = _serialize([new_header] + body, term).encode("utf-8")
    return (b"\xef\xbb\xbf" + out if bom else out), [t for t in NORMALIZE_WHITELIST if t in changed]


# ================================================================ t1.json (written by tools/t02_checks.py)
def parse_normalize(value, comp):
    """t1.json 'normalize' -> {official name: [transforms]} for this competition's files.
    Accepted shapes: ["strip_bom", ...] (all files of the competition); [{"transform"|"name": t,
    "file"|"files"|"path": ...}, ...]; {"<official name>": [t, ...]}; {"<transform>": [files]}."""
    names = COMP_FILES[comp]
    out = {n: [] for n in names}

    def add(t, files):
        if not isinstance(t, str):
            raise BundleError(f"t1.json normalize entry {t!r} is not a transform name")
        files = names if files is None else ([files] if isinstance(files, str) else list(files))
        for f in files:
            b = os.path.basename(str(f))
            if b not in out:
                raise BundleError(f"t1.json normalize names file {f!r}, which is not a {comp} test file")
            if t not in out[b]:
                out[b].append(t)

    if value is None:
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            if os.path.basename(k) in out:
                for t in ([v] if isinstance(v, str) else v):
                    add(t, [k])
            else:
                add(k, v)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                add(item, None)
            elif isinstance(item, dict):
                t = item.get("transform", item.get("name", item.get("op")))
                files = item.get("file", item.get("files", item.get("path")))
                add(t, files)
            else:
                raise BundleError(f"t1.json normalize entry {item!r} not understood")
    else:
        raise BundleError(f"t1.json normalize {value!r} not understood")
    for n, ts in out.items():
        bad = [t for t in ts if t not in NORMALIZE_WHITELIST]
        if bad:
            raise BundleError(f"t1.json asks for {bad} on {n}: not in the whitelist {list(NORMALIZE_WHITELIST)}")
    return out


def recorded_input_shas(obj, names):
    """{official name: {sha256, ...}} found anywhere in t1.json: a key equal to the name whose value is a
    SHA-256 or a dict with 'sha256', or a record {'file'|'path'|'name': <path of the name>, 'sha256': ...}."""
    found = defaultdict(set)

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                b = os.path.basename(str(k))
                if b in names:
                    if isinstance(v, str) and _HEX64.match(v):
                        found[b].add(v)
                    elif isinstance(v, dict) and isinstance(v.get("sha256"), str) and _HEX64.match(v["sha256"]):
                        found[b].add(v["sha256"])
            sha = o.get("sha256")
            if isinstance(sha, str) and _HEX64.match(sha):
                for key in ("file", "path", "name"):
                    if isinstance(o.get(key), str) and os.path.basename(o[key]) in names:
                        found[os.path.basename(o[key])].add(sha)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return found


def load_t1(checks_dir, comp, tests_dir, allow_fail_opening, require_full=False):
    path = os.path.join(checks_dir, comp, "t1.json")
    if not os.path.exists(path):
        raise BundleError(f"{path} is missing: run 'PY tools/t02_checks.py t1 --comp {comp} --tests-dir "
                          f"{tests_dir}' first")
    t1 = read_json(path)
    verdict = t1.get("verdict") if isinstance(t1, dict) else None
    if require_full and isinstance(t1, dict):
        # the official bundle is built only after the full T1 on the raw files (review finding 2026-09-26: a
        # structure-only t1.json, as the rehearsal commands write with --only, used to be accepted)
        mode = (t1.get("provenance") or {}).get("mode")
        if t1.get("complete") is not True or t1.get("stage") != "raw" or mode != "real":
            raise BundleError(f"{path}: complete={t1.get('complete')!r} (ran {t1.get('subchecks_run')}), stage "
                              f"{t1.get('stage')!r}, mode {mode!r}; --kind official needs the full T1 (T1.1-T1.5) "
                              f"on data/test/raw in real mode: PY tools/t02_checks.py t1 --comp {comp} --rerun "
                              f"--reason TEXT (without --only)")
    if verdict not in T1_ALL:
        raise BundleError(f"{path}: verdict {verdict!r} is not one of {list(T1_ALL)}")
    if verdict == "FAIL_STRUCT":
        raise BundleError(f"{path}: T1 verdict FAIL_STRUCT; stop and write a PREREG revision before any "
                          f"prediction (rev 5 section 9 item 6)")
    if verdict == "FAIL_OPENING" and not allow_fail_opening:
        raise BundleError(f"{path}: T1 verdict FAIL_OPENING. The registered consequence (rank R10 fallback, "
                          f"player per-pool procedure, T3 not evaluable) needs the bundle anyway: re-run build "
                          f"with --allow-fail-opening --reason TEXT")
    # t02_checks writes normalize = {official name: [transforms]} plus the flat normalize_transforms;
    # a per-file 'normalize_by_file' next to a flat 'normalize' list is accepted too
    by_file = t1.get("normalize_by_file")
    norm_map = parse_normalize(by_file if isinstance(by_file, dict) else t1.get("normalize"), comp)
    flat_ref = (t1.get("normalize_transforms") if isinstance(t1.get("normalize_transforms"), list)
                else t1.get("normalize") if isinstance(by_file, dict) and isinstance(t1.get("normalize"), list)
                else None)
    if flat_ref is not None and {t for ts in norm_map.values() for t in ts} != set(flat_ref):
        raise BundleError(f"{path}: the per-file normalize list {norm_map} and the flat list {flat_ref} disagree")
    n_requested = sum(len(v) for v in norm_map.values())
    if verdict == "PASS" and n_requested:
        raise BundleError(f"{path}: verdict PASS but normalize lists {norm_map}; inconsistent t1.json")
    if verdict == "NORMALIZED" and not n_requested:
        raise BundleError(f"{path}: verdict NORMALIZED but no normalize transforms are listed")
    shas = recorded_input_shas(t1, COMP_FILES[comp])
    for name in COMP_FILES[comp]:
        rec = shas.get(name, set())
        if not rec:
            raise BundleError(f"{path} does not record the SHA-256 of {name}; cannot confirm that T1 ran on "
                              f"these raw files")
        if len(rec) > 1:
            raise BundleError(f"{path} records several SHA-256 for {name}: {sorted(rec)}")
        got = sha256_file(os.path.join(tests_dir, name))
        if got not in rec:
            raise BundleError(f"{path} was written for another {name} (t1.json {sorted(rec)[0][:12]}, "
                              f"file {got[:12]}); re-run T1 on {tests_dir}")
    return {"path": path, "sha256": sha256_file(path), "verdict": verdict, "normalize": norm_map}


def recheck_target(comp, checks_dir, kind):
    """(--out, written file) of tools/t02_checks.py t1 for the re-check of the built files: <checks>/built and
    <checks>/built/<comp>/t1.json for every kind. In real mode (kind official) t02_checks.py accepts data/test only
    with the output dir results/test02/checks/built (its REAL_T1 pair (DATA_TEST, CHECKS_BUILT)) and always writes
    <out>/<comp>/t1.json, so the official build uses exactly that pair (review finding, 2026-09-26)."""
    out = os.path.join(checks_dir, "built")
    return out, os.path.join(out, comp, "t1.json")


def run_t1_recheck(comp, tests_dir, checks_dir, kind, reason=None):
    """T1.1-T1.3 on the built files through tools/t02_checks.py; returns (verdict, t1 json path)."""
    tool = os.path.join(HERE, "t02_checks.py")
    if not os.path.exists(tool):
        raise BundleError(f"{tool} is missing; the build re-check of T1.1-T1.3 needs it")
    out, target = recheck_target(comp, checks_dir, kind)
    cmd = [sys.executable, "-B", tool, "t1", "--comp", comp, "--tests-dir", tests_dir,
           "--only", "T1.1,T1.2,T1.3", "--out", out] + ([] if kind == "official" else [f"--{kind}"])
    before = sha256_file(target) if os.path.exists(target) else None
    if before is not None:
        cmd += ["--rerun", "--reason", f"t02_test_bundle build re-check ({reason or 'rebuild'})"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if not os.path.exists(target) or r.returncode not in (0, 2, 3, 4):
        raise BundleError(f"re-check of T1.1-T1.3 failed (exit {r.returncode}, {target}): "
                          f"{(r.stderr or r.stdout)[-800:]}")
    res = read_json(target)
    files = res.get("files") or {}
    stale = [n for n in COMP_FILES[comp]
             if (files.get(n) or {}).get("sha256") != sha256_file(os.path.join(tests_dir, n))]
    if stale or not same_path(res.get("tests_dir", ""), tests_dir):
        raise BundleError(f"{target} is not about the built files in {tests_dir} ({stale or res.get('tests_dir')})")
    return res.get("verdict"), target


# ================================================================ manifest and zip
def file_record(path, source_sha256=None, normalized=(), **extra):
    sha = sha256_file(path)
    rec = {"sha256": sha, "bytes": os.path.getsize(path), "rows": data_rows(path),
           "source_sha256": source_sha256 or sha, "normalized": list(normalized)}
    rec.update(extra)
    return rec


def write_manifest(tests_dir, kind, files, t1, **extra):
    if kind not in KINDS:
        raise BundleError(f"kind {kind!r} is not one of {list(KINDS)}")
    if set(files) != set(ALL_NAMES):
        raise BundleError(f"manifest needs exactly the six official names; got {sorted(files)}")
    for comp in ("rank", "player"):
        if t1.get(comp) not in T1_OK + ("FAIL_OPENING",):
            raise BundleError(f"manifest t1[{comp}] = {t1.get(comp)!r}")
    man = {"format": MANIFEST_FORMAT, "kind": kind, "created": now_iso(),
           "files": {n: files[n] for n in ALL_NAMES}, "t1": {"rank": t1["rank"], "player": t1["player"]}}
    man.update(extra)
    man["tool"] = tool_record()
    path = write_json(os.path.join(tests_dir, MANIFEST_NAME), man)
    return path, man


def verify_manifest(tests_dir, names=ALL_NAMES):
    """Local files equal MANIFEST.json; returns (manifest, its sha256)."""
    mpath = os.path.join(tests_dir, MANIFEST_NAME)
    if not os.path.exists(mpath):
        raise BundleError(f"{mpath} is missing")
    man = read_json(mpath)
    if man.get("format") != MANIFEST_FORMAT:
        raise BundleError(f"{mpath}: format {man.get('format')!r}")
    for n in names:
        rec = (man.get("files") or {}).get(n)
        p = os.path.join(tests_dir, n)
        if not rec or not os.path.exists(p) or sha256_file(p) != rec.get("sha256"):
            raise BundleError(f"{p} is missing or differs from {mpath}")
    return man, sha256_file(mpath)


def write_zip(zip_out, tests_dir):
    """Members tests/<6 official names> and tests/MANIFEST.json, fixed member times; verified after writing."""
    check_writable(zip_out)
    os.makedirs(os.path.dirname(os.path.abspath(zip_out)), exist_ok=True)
    tmp = zip_out + ".tmp"
    members = list(ALL_NAMES) + [MANIFEST_NAME]
    want = {}
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for n in members:
            p = os.path.join(tests_dir, n)
            info = zipfile.ZipInfo(f"{ZIP_DIR}/{n}", date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(refuse_forbidden(p), "rb") as f, z.open(info, "w", force_zip64=True) as out:
                shutil.copyfileobj(f, out, 1 << 20)
            want[f"{ZIP_DIR}/{n}"] = sha256_file(p)
    with zipfile.ZipFile(tmp) as z:
        got = {i.filename: sha256_bytes(z.read(i.filename)) for i in z.infolist()}
    if got != want:
        raise BundleError(f"{tmp}: members differ from the source files after writing")
    os.replace(tmp, zip_out)
    return sha256_file(zip_out), os.path.getsize(zip_out)


# ================================================================ ingest
def _near_misses(seen_names):
    keys = ("rank_prediction", "player_identification")
    return sorted({n for n in seen_names if n not in ALL_NAMES and (n.lower() in {a.lower() for a in ALL_NAMES}
                                                                     or any(k in n.lower() for k in keys))})


def _read_file(path):
    with open(refuse_forbidden(path), "rb") as f:
        return f.read()


def _read_member(zpath, member):
    with zipfile.ZipFile(zpath) as z:
        return z.read(member)


def _own_record(data):
    """The kind of this kit's record in data (bytes of a MANIFEST.json / SOURCE.json / FAKE_TREE.json), else None."""
    try:
        obj = json.loads(data.decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(obj, dict) and obj.get("format") in OWN_RECORDS.values():
        return "%s%s" % (obj["format"], (" kind %s" % obj["kind"]) if obj.get("kind") else "")
    return None


def find_sources(srcs, own=None):
    """{official name: [(location, bytes-reader)]} from folders (recursive; zip files inside are searched
    too) and zip files. Only the six official base names and this kit's own record names (OWN_RECORDS) are
    ever opened; own (a list) collects (location, record kind) of the latter."""
    found = defaultdict(list)
    seen = set()
    own = [] if own is None else own

    def scan_zip(zpath):
        try:
            with zipfile.ZipFile(zpath) as z:
                infos = z.infolist()
        except zipfile.BadZipFile as e:
            raise BundleError(f"{zpath}: not a readable zip ({e})")
        for info in infos:
            if info.is_dir():
                continue
            base = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
            seen.add(base)
            if base in ALL_NAMES:
                found[base].append((f"{zpath}!{info.filename}",
                                    lambda zp=zpath, m=info.filename: _read_member(zp, m)))
            elif base in OWN_RECORDS:
                kind = _own_record(_read_member(zpath, info.filename))
                if kind:
                    own.append((f"{zpath}!{info.filename}", kind))

    for src in srcs:
        if os.path.isdir(src):
            for root, dirs, files in os.walk(src):
                dirs.sort()
                for f in sorted(files):
                    p = os.path.join(root, f)
                    seen.add(f)
                    if f in ALL_NAMES:
                        found[f].append((p, lambda pp=p: _read_file(pp)))
                    elif f in OWN_RECORDS:
                        kind = _own_record(_read_file(p))
                        if kind:
                            own.append((p, kind))
                    elif f.lower().endswith(".zip"):
                        scan_zip(p)
        elif os.path.isfile(src) and src.lower().endswith(".zip"):
            scan_zip(src)
        elif os.path.isfile(src) and os.path.basename(src) in ALL_NAMES:
            found[os.path.basename(src)].append((src, lambda pp=src: _read_file(pp)))
        else:
            raise BundleError(f"--src {src!r} is not a folder, a zip file or an official test file")
    return found, seen


def cmd_ingest(args):
    dest = args.dest or repo_paths(args.root)["raw"]
    check_root(args.root)
    if inside(dest, REPO) and not inside(dest, os.path.join(REPO, "data", "test")):
        raise BundleError(f"--dest {dest!r}: inside the repository only data/test/** (gitignored) is allowed")
    check_writable(dest)
    own = []
    found, seen = find_sources(args.src, own)
    if own:
        msg = (f"these sources hold this kit's own records, so they are a rehearsal / smoke / fake tree built from "
               f"validation games (or a re-packed bundle), not the files downloaded from AIdea: "
               f"{[f'{loc} ({kind})' for loc, kind in own][:4]}")
        if same_path(dest, repo_paths(None)["raw"]):
            raise BundleError(msg + "; the real ingest takes only the AIdea download (review finding 2026-09-26)")
        print("note: " + msg + " (allowed outside the real data/test/raw)")
    missing = [n for n in ALL_NAMES if n not in found]
    if missing:
        near = _near_misses(seen)
        raise BundleError(f"official test files not found: {missing}"
                          + (f"; similar names seen: {near} (if the official names changed, follow the "
                             f"D5 path in docs/TESTDAY.md)" if near else ""))
    chosen, records = {}, {}
    for n in ALL_NAMES:
        blobs = [(loc, read()) for loc, read in found[n]]
        shas = {sha256_bytes(b) for _loc, b in blobs}
        if len(shas) > 1:
            raise BundleError(f"{n} was found {len(blobs)} times with different contents: "
                              f"{[loc for loc, _b in blobs]}; give --src a narrower folder")
        chosen[n] = blobs[0][1]
        records[n] = {"sha256": shas.pop(), "bytes": len(blobs[0][1]), "from": [loc for loc, _b in blobs]}
    outs = [os.path.join(dest, n) for n in ALL_NAMES] + [os.path.join(dest, SOURCE_NAME)]
    replaced = require_fresh(outs, args.rerun, args.reason)
    os.makedirs(dest, exist_ok=True)
    for n in ALL_NAMES:
        p = write_bytes(os.path.join(dest, n), chosen[n])
        if sha256_file(p) != records[n]["sha256"]:
            raise BundleError(f"{p}: SHA-256 changed while copying")
    rec = {"format": "c10-test02 source v1", "created": now_iso(),
           "sources": [os.path.abspath(s) for s in args.src], "dest": os.path.abspath(dest), "files": records,
           "own_records_in_sources": [{"location": loc, "kind": kind} for loc, kind in own],
           "rerun_reason": args.reason if replaced else None, "tool": tool_record()}
    write_json(os.path.join(dest, SOURCE_NAME), rec)
    for n in ALL_NAMES:
        print(f"  {n}  {records[n]['bytes']:>11,d} bytes  sha256 {records[n]['sha256'][:12]}")
    print(f"ingest: 6 official files copied to {dest}; next: PY tools/t02_checks.py t1 --comp rank|player "
          f"--tests-dir {dest}")
    return rec


# ================================================================ build
def resolve_build_paths(args):
    check_root(args.root)
    d = repo_paths(args.root)
    p = {"tests": args.tests_dir or d["raw"], "out": args.out_dir or d["tests"],
         "zip": args.zip_out or d["zip"], "checks": args.checks_dir or d["checks"]}
    if args.kind == "official":
        real = repo_paths(None)
        if args.root is not None:
            raise BundleError("--kind official is the real run: no --root")
        for key, want in (("tests", real["raw"]), ("out", real["tests"]), ("checks", real["checks"])):
            if not same_path(p[key], want):
                raise BundleError(f"--kind official needs the repo path {want!r} for {key}; got {p[key]!r}")
        if not inside(p["zip"], os.path.join(REPO, "data")):
            raise BundleError(f"--kind official writes the zip under data/ (gitignored); got {p['zip']!r}")
    else:
        for key in ("out", "zip", "checks"):
            if inside(p[key], REPO):
                raise BundleError(f"--kind {args.kind}: {key} {p[key]!r} is inside the repository; "
                                  f"use a scratch tree (--root S)")
    for key in ("out", "zip", "checks"):
        check_writable(p[key])
    return p


def cmd_build(args):
    p = resolve_build_paths(args)
    if args.allow_fail_opening and not (args.reason or "").strip():
        raise BundleError("--allow-fail-opening needs --reason TEXT")
    in_place = same_path(p["tests"], p["out"])
    missing = [n for n in ALL_NAMES if not os.path.exists(os.path.join(p["tests"], n))]
    if missing:
        raise BundleError(f"{p['tests']}: missing {missing}")
    t1 = {c: load_t1(p["checks"], c, p["tests"], args.allow_fail_opening, require_full=args.kind == "official")
          for c in ("rank", "player")}

    # 1-2: normalized copies (only the listed transforms) or byte copies
    outputs = {}
    for comp in ("rank", "player"):
        for n in COMP_FILES[comp]:
            with open(refuse_forbidden(os.path.join(p["tests"], n)), "rb") as f:
                raw = f.read()
            data, changed = normalize_bytes(n, raw, t1[comp]["normalize"][n])
            outputs[n] = (raw, data, changed, t1[comp]["normalize"][n])
    if in_place and any(o[0] != o[1] for o in outputs.values()):
        raise BundleError("--tests-dir equals --out-dir but T1 asks for normalization; give a separate --out-dir")
    build_rec = os.path.join(p["checks"], "bundle_build.json")
    targets = [recheck_target(c, p["checks"], args.kind)[1] for c in ("rank", "player")] + [
        build_rec, p["zip"], os.path.join(p["out"], MANIFEST_NAME)]
    if not in_place:
        targets += [os.path.join(p["out"], n) for n in ALL_NAMES]
    replaced = require_fresh(targets, args.rerun, args.reason)
    os.makedirs(p["out"], exist_ok=True)
    files = {}
    for n, (raw, data, changed, requested) in outputs.items():
        dst = os.path.join(p["out"], n)
        if not in_place:
            write_bytes(dst, data)
        files[n] = file_record(dst, source_sha256=sha256_bytes(raw), normalized=changed)
        if files[n]["sha256"] != sha256_bytes(data):
            raise BundleError(f"{dst}: SHA-256 changed while writing")
        if requested and set(requested) != set(changed):
            files[n]["normalize_requested"] = list(requested)

    # 3: MANIFEST
    verdicts = {c: t1[c]["verdict"] for c in ("rank", "player")}
    mpath, man = write_manifest(
        p["out"], args.kind, files, verdicts,
        t1_basis={c: {"t1_json": os.path.relpath(t1[c]["path"], REPO) if inside(t1[c]["path"], REPO)
                      else t1[c]["path"], "t1_json_sha256": t1[c]["sha256"]} for c in ("rank", "player")},
        fail_opening_reason=args.reason if args.allow_fail_opening else None)

    # 4: T1.1-T1.3 on the built files (tools/t02_checks.py) and the quick structure check
    recheck = {}
    for comp in ("rank", "player"):
        verdict, path = run_t1_recheck(comp, p["out"], p["checks"], args.kind, args.reason)
        recheck[comp] = {"verdict": verdict, "t1_json": path, "t1_json_sha256": sha256_file(path)}
        if verdict != "PASS":
            raise BundleError(f"T1.1-T1.3 on the built {comp} files gave {verdict!r} ({path}); the zip was not "
                              f"written")
        q = quick_structure(p["out"], comp)
        if q:
            raise BundleError(f"quick structure check of the built {comp} files failed although T1 passed "
                              f"(investigate): {q[:5]}")

    # 5: zip
    verify_manifest(p["out"])
    zsha, zbytes = write_zip(p["zip"], p["out"])
    rec = {"format": "c10-test02 bundle build v1", "created": now_iso(), "kind": args.kind,
           "zip": os.path.abspath(p["zip"]), "zip_sha256": zsha, "zip_bytes": zbytes,
           "manifest_sha256": sha256_file(mpath), "files": {n: files[n]["sha256"] for n in ALL_NAMES},
           "normalized": {n: files[n]["normalized"] for n in ALL_NAMES if files[n]["normalized"]},
           "t1_raw": {c: {"verdict": t1[c]["verdict"], "t1_json_sha256": t1[c]["sha256"]} for c in t1},
           "t1_built": recheck, "quick_structure": "PASS", "in_place": in_place,
           "rerun_reason": args.reason if replaced else None, "tool": tool_record()}
    write_json(build_rec, rec)
    for n in ALL_NAMES:
        extra = f"  normalized {files[n]['normalized']}" if files[n]["normalized"] else ""
        print(f"  {n}  rows {files[n]['rows']:>6d}  sha256 {files[n]['sha256'][:12]}{extra}")
    print(f"build ({args.kind}): {p['zip']}  {zbytes:,d} bytes  sha256 {zsha}")
    print(f"MANIFEST sha256 {rec['manifest_sha256']}")
    return rec


# ================================================================ unpack
def _member_ok(name, comp):
    if not name or name.startswith("/") or "\\" in name or ":" in name:
        return f"unsafe member path {name!r}"
    parts = name.rstrip("/").split("/")
    if any(x in ("", ".", "..") for x in parts):
        return f"unsafe member path {name!r}"
    if parts[0] != comp or len(parts) < 2:
        return f"member {name!r} is outside {comp}/"
    if _NEEDLE in parts[-1].lower():
        return f"member {name!r} looks like a validation label file"
    return None


def cmd_unpack(args):
    check_root(args.root)
    d = repo_paths(args.root)
    results_dir = args.results_dir or d["results"]
    tests_dir = args.tests_dir or d["tests"]
    real = same_path(results_dir, repo_paths(None)["results"])
    rehearsal_repo = same_path(results_dir, os.path.join(REPO, "results", "test02_rehearsal"))
    if inside(results_dir, REPO) and not (real or rehearsal_repo):
        raise BundleError(f"--results-dir {results_dir!r}: inside the repo only results/test02 (real) or "
                          f"results/test02_rehearsal (gitignored) are allowed")
    if args.allow_failed and inside(results_dir, REPO):
        raise BundleError("--allow-failed is for inspection only: unpack into a scratch folder outside the repo")
    if real and not same_path(tests_dir, repo_paths(None)["tests"]):
        raise BundleError(f"the real results dir needs the real tests dir {repo_paths(None)['tests']!r}")
    check_writable(results_dir)
    zpath = refuse_forbidden(args.zip)
    if not zipfile.is_zipfile(zpath):
        raise BundleError(f"{zpath} is not a zip file")
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        bad = [m for m in (_member_ok(n, args.comp) for n in names) if m]
        if bad:
            raise BundleError(f"{zpath}: {bad[:5]}")
        dup = sorted(n for n, k in Counter(names).items() if k > 1)
        if dup:
            raise BundleError(f"{zpath}: repeated members {dup[:5]}")
        need = [f"{args.comp}/run_log.json", f"{args.comp}/config.json"]
        absent = [n for n in need if n not in names]
        if absent:
            raise BundleError(f"{zpath}: missing {absent}")
        run_log = json.loads(z.read(need[0]).decode("utf-8"))
        config = json.loads(z.read(need[1]).decode("utf-8"))
        status = run_status(run_log)
        history = run_history(run_log)
        print(f"run_log.json runs: {', '.join('%s %s' % (n, s) for n, s in history) or 'none'} (status {status!r})")
        if status != "completed" and not args.allow_failed:
            raise BundleError(f"{zpath}: run_log status {status!r} (not 'completed'); nothing from this run may be "
                              f"used. For inspection only: --allow-failed with a scratch --results-dir")
        failed_before = earlier_failures(run_log)
        if failed_before and not args.allow_failed:
            msg = (f"{zpath}: run_log.json has earlier failed runs {failed_before} before the last run: the notebook "
                   f"was run again in the same Drive folder, which is not the registered rerun (PREREG_02 rev 5 9.2, "
                   f"rev 6 10.10: rename the Drive folder to <comp>_failed_N, new runtime). This run counts as another "
                   f"attempt and its outputs are not used; follow docs/TESTDAY.md step 4 '重跑一次'")
            if real or rehearsal_repo:
                raise BundleError(msg)
            print("WARNING: " + msg)
        tb = config.get("test_bundle") or {}
        man, man_sha = verify_manifest(tests_dir)
        if tb.get("manifest_sha256") != man_sha:
            raise BundleError(f"config.json test_bundle.manifest_sha256 {str(tb.get('manifest_sha256'))[:12]} differs "
                              f"from the local {os.path.join(tests_dir, MANIFEST_NAME)} ({man_sha[:12]})")
        files = tb.get("files") or {}
        for n in COMP_FILES[args.comp]:
            if files.get(n) != man["files"][n]["sha256"]:
                raise BundleError(f"config.json test_bundle.files[{n}] differs from the local MANIFEST")
        if real and (config.get("smoke") is not False or config.get("rehearsal") is not False):
            raise BundleError(f"config.json says smoke={config.get('smoke')!r}, rehearsal={config.get('rehearsal')!r}; "
                              f"only a real run (both false) goes into results/test02")
        if rehearsal_repo and config.get("rehearsal") is not True:
            raise BundleError("results/test02_rehearsal takes only rehearsal runs (config.rehearsal true)")
        record = os.path.join(results_dir, f"unpack_{args.comp}.json")
        targets = [os.path.join(results_dir, *n.split("/")) for n in names if not n.endswith("/")]
        existing = [t for t in targets + [record] if os.path.exists(t)]
        if existing:
            raise BundleError(f"target files already exist, nothing extracted: {existing[:4]}"
                              + (f" (+{len(existing) - 4} more)" if len(existing) > 4 else ""))
        for info, target in zip([i for i in infos if not i.filename.endswith("/")], targets):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            check_writable(target)
            with z.open(info) as src, open(target + ".tmp", "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(target + ".tmp", target)
    rec = {"format": "c10-test02 unpack v1", "created": now_iso(), "comp": args.comp,
           "zip": os.path.abspath(zpath), "zip_sha256": sha256_file(zpath), "zip_bytes": os.path.getsize(zpath),
           "members": len(targets), "run_status": status,
           "runs": [{"run": n, "status": s} for n, s in history],
           "earlier_failed_runs": [{"run": n, "status": s} for n, s in failed_before],
           "smoke": config.get("smoke"),
           "rehearsal": config.get("rehearsal"), "manifest_sha256": man_sha,
           "allow_failed": bool(args.allow_failed), "tool": tool_record()}
    write_json(record, rec)
    print(f"unpack: {len(targets)} files from {zpath} into {results_dir} (status {status}); record {record}")
    if real:
        print("next: commit results/test02/%s before any local step (docs/TESTDAY.md step 4)" % args.comp)
    return rec


# ================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="t02_test_bundle.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--root", help="scratch tree with the repo layout (outside the repo); default: the repo")
        p.add_argument("--rerun", action="store_true", help="replace existing outputs (needs --reason)")
        p.add_argument("--reason", help="why (stored in the record)")

    p = sub.add_parser("ingest", help="copy the six official test files to data/test/raw")
    p.add_argument("--src", action="append", required=True, help="folder or zip (repeatable)")
    p.add_argument("--dest")
    common(p)
    p = sub.add_parser("build", help="normalize per t1.json, MANIFEST, re-check, zip")
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--tests-dir")
    p.add_argument("--out-dir")
    p.add_argument("--zip-out")
    p.add_argument("--checks-dir")
    p.add_argument("--allow-fail-opening", action="store_true")
    common(p)
    p = sub.add_parser("unpack", help="extract a notebook result zip into results/test02")
    p.add_argument("--zip", required=True)
    p.add_argument("--comp", required=True, choices=("rank", "player"))
    p.add_argument("--results-dir")
    p.add_argument("--tests-dir")
    p.add_argument("--allow-failed", action="store_true")
    common(p)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"ingest": cmd_ingest, "build": cmd_build, "unpack": cmd_unpack}[args.cmd](args)


if __name__ == "__main__":
    main()
