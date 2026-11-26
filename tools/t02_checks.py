"""C10 test-day checks T1-T6 for the official test files (test02).

Registration: docs/PREREG_02_OPENING_FUSION.md section 9 (rev 5), items 6-9, and rev 8 (the official structure
of 2026-09-28: black-only competition-1 pools and questions, 5/10/15/20 games, one colour per rank question).
Spec: results/test02/design/spec.md section 4 (design record: results/test02/design/checks.md; audit of the
official structure: results/test02/design/official_structure/README.md).

    PY tools/t02_checks.py precompute [--part all|base|t3b] [--out results/test02/null] [--t2-index-cache [PATH]]
    PY tools/t02_checks.py t1  --comp rank|player [--tests-dir data/test/raw] [--out results/test02/checks]
                               [--only T1.1,T1.2,T1.3] [--require-pass]
    PY tools/t02_checks.py run --comp rank|player [--tests-dir data/test] [--only T2,T3,T4,T5]
                               [--index-part train|fit] [--index-cache [PATH]] [--out results/test02/checks]
    PY tools/t02_checks.py t6  --comp player [--submission results/test02/player/submission_player.csv]
                               [--player-dir results/test02/player]
    every command also takes [--dry-run|--smoke|--rehearsal] [--root S] [--rerun --reason TEXT]

Outputs: results/test02/checks/<comp>/{t1,t2,t3,t4,t5,t6,summary}.json, the bundle re-check of the built
copies in results/test02/checks/built/<comp>/t1.json, and (precompute) results/test02/null/{t3_null.json,
t3_null_raw.npz, t3_null_canon.npz, train_refs.json} (--part base, rev 5) and {t3b_null.json, t3b_null_raw.npz,
t3b_null_canon.npz} (--part t3b, rev 8: the T3 variant for pools whose candidates are all black; --part all
writes both sets). summary.json = {check: {verdict, consequence,
details_file, sha256}}. t1.json 'normalize' = {official file name: [whitelisted transforms]} for the files
that need one ({} when the verdict is PASS); 'normalize_transforms' = their union.

Modes (spec 3.0; this tool keeps its own copy of the guards):
    real (default)  t1: --tests-dir REPO/data/test/raw with --out REPO/results/test02/checks (always the
                    full T1.1-T1.5: --only is refused there, the raw t1.json gates the build and the uploads),
                    or --tests-dir REPO/data/test (the built copies) with --out REPO/results/test02/checks/built
                    (t02_test_bundle.py build calls it so, with --only T1.1,T1.2,T1.3). run / t6: --tests-dir is
                    REPO/data/test, with its MANIFEST.json (kind official) matching the files and the raw t1.json
                    (complete, stage raw). --out is REPO/results/test02/checks, the null is
                    REPO/results/test02/null. t1 / run / t6 need this file committed and clean (run also the
                    seven null files, T3b included). Content checks record how many games are local validation
                    games (the same SHA-1 membership as below) and refuse a file in which at least half are: that is a
                    validation-derived tree (rehearsal / smoke / fake) that got into data/test, not an
                    official file (an official file may hold a few training games; T2 looks for them).
    --dry-run       any tests dir; the output must be outside the repository. Content checks
                    (T1.4, T1.5, T2-T5) refuse a tree that holds any local validation game (SGF
                    SHA-1 membership against the validation question / candidate files, the same
                    hashes c10data.load_rows uses for its leak check).
    --smoke / --rehearsal  trees built from validation files: only t1 --only T1.1,T1.2,T1.3 and t6.
    --root S        (not real) default --tests-dir S/data/test and --out S/results/test02/checks;
                    without --root and --out the out dir is derived from a tests dir ending in
                    data/test or data/test/raw.
    Never writes into results/prereg01, results/prereg02 or data/local; never opens a file whose
    base name contains the validation label-file word; existing outputs are only replaced with
    --rerun --reason TEXT (the reason is stored in the report).

Exit codes: 0 done (t1: verdict PASS or NORMALIZED; with --require-pass only PASS), 2 t1 FAIL_OPENING,
3 t1 FAIL_STRUCT, 4 t1 NORMALIZED under --require-pass, 1 refusal / error / precompute pin
mismatch / T3 reference mismatch.

Python API (for t02_test_bundle.py and tests): t1_check(comp, tests_dir, only=None) -> dict
(the t1.json content without provenance; writes nothing).

Imports: c10data, p02_moves (SYM_TABLE, MOVE_RE) and p02_player_open (game_tokens, moves_of and the
T3 / T6 matrix helpers). Never a rank model tool, decide, t02_rank or t02_player. Training games are
read only through c10data.load_rows; the statistics use training players only.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import zipfile
from array import array
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c10data  # noqa: E402  (shared, leak-checked training access)
import p02_moves as pm  # noqa: E402  (shared preprocessing: SYM_TABLE, MOVE_RE)

# ---------------------------------------------------------------- paths
DATA_TEST = os.path.join(REPO, "data", "test")
DATA_TEST_RAW = os.path.join(DATA_TEST, "raw")
CHECKS_DIR = os.path.join(REPO, "results", "test02", "checks")
NULL_DIR = os.path.join(REPO, "results", "test02", "null")
PLAYER_DIR = os.path.join(REPO, "results", "test02", "player")
TRAIN_DIR = os.path.join(REPO, "data", "training")
SPLIT_CSV = os.path.join(REPO, "results", "split_v1", "players_split.csv")
LOCAL_DIR = os.path.join(REPO, "data", "local")
INDEX_CACHE = os.path.join(REPO, "data", "cache", "t2_train_index.npz")
READ_ONLY = (os.path.join(REPO, "results", "prereg01"), os.path.join(REPO, "results", "prereg02"),
             os.path.join(REPO, "data", "local"))
NULL_FILES = ("t3_null.json", "t3_null_raw.npz", "t3_null_canon.npz", "train_refs.json")

# The validation label files are never opened; the word is spelled in two halves so that this
# source never contains it (tests/test_t02_hygiene.py).
_FORBIDDEN = "ans" + "wers"

# ---------------------------------------------------------------- official format (README, tutorial 359e06f)
N_SLOTS = 20
Q_COLS = ["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, N_SLOTS + 1)] + \
         ["color_%d" % i for i in range(1, N_SLOTS + 1)]
C_COLS = ["game_id", "player_id", "sgf_content", "color"]
FILES = {
    "rank": [("public", "questions", "rank_prediction_test_public.csv"),
             ("private", "questions", "rank_prediction_test_private.csv")],
    "player": [("public", "questions", "player_identification_test_public.csv"),
               ("public", "candidates", "player_identification_candidates_public.csv"),
               ("private", "questions", "player_identification_test_private.csv"),
               ("private", "candidates", "player_identification_candidates_private.csv")],
}
POOLS = ("public", "private")
ID_PREFIX = {"public": "pub_", "private": "priv_"}
PLAYER_PREFIX = {"public": "pub_p_", "private": "priv_p_"}
N_QUESTIONS = 400
N_CANDIDATES = 400
GAMES_PER_CANDIDATE = 100
K_MIN, K_MAX = 5, 20

# ---------------------------------------------------------------- T1 (PREREG_02 9.6)
T1_SUBS = ("T1.1", "T1.2", "T1.3", "T1.4", "T1.5")
T1_STRUCT = ("T1.1", "T1.2", "T1.3")
NORMALIZE_WHITELIST = ("strip_bom", "crlf_to_lf", "upper_colour", "reorder_columns_to_readme",
                       "clear_slots_after_num_games_if_empty_whitespace",
                       # PREREG_02 rev 8 (official data description 2026-09-28: 'color_1..color_20 always B'): a
                       # colour letter in a slot after num_games whose sgf cell is empty or whitespace only
                       "clear_colour_after_num_games_if_sgf_empty")
COLOUR_LETTERS = ("B", "W", "b", "w")
SETUP_PROPS = ("AB", "AW", "AE", "HA", "PL")
S99 = ("cc", "cd", "ce", "cf", "dd", "de", "df", "jj")   # registered (9.6 T1.5)
T1_4_MAX_SHARE = 0.01        # FAIL_OPENING iff more than 1% of a file's games violate (a) or (b)
T1_5_MIN_SHARE = 0.95        # PASS iff share of first moves in S99 >= 0.95
MAX_EXAMPLES = 20
BOM = b"\xef\xbb\xbf"

# ---------------------------------------------------------------- T2 (9.7)
MOVE_RE = pm.MOVE_RE         # ;([BW])\[([a-s]{2})\], no move limit here
M_TIE = 12                   # tie search on the first 12 moves (exact, see canon_points)
KEY_BYTES = 12               # first 12 bytes of the SHA-1 digest (as in the design-record null)
K_COPY = 100                 # copy = first 100 moves identical (shorter games: the whole game)
K_RECORD = 60                # K60 matches are record-only
T2_MIN_COPIES = 10
T2_RATE = 0.001
KEY_VERSION = "c10-t02 game key v1: sha1(D4-min point string)[:12 bytes]; ties on the first 12 moves"

# ---------------------------------------------------------------- T3 (9.8)
T3_SALT = "c10-t3|cand"
T3_MIN_GAMES = 100           # null players: fit players with >= 100 games
T3_PROFILE = 100             # candidate profile = 100 games
T3_SHARE = 0.10              # overlap iff n_above >= ceil(0.10 * candidates) in raw OR canon
T3Q_MIN_GAMES = 25
ORIENTS = ("raw", "canon")
PINS = {
    "reference": {"players": 5767, "games": 720307,
                  "sha256_sorted_player_ids": "9940feed82f435a770030a17b32b5bbf7dc1002daa61616a989b24580bdf664f"},
    "raw": {"S2_q99": 0.1004267242871777, "vocab": 146421, "n_null": 1985, "n_holdout_inner": 226,
            "null_n_exceed": 19, "holdout_inner_n_exceed": 1},
    "canon": {"S2_q99": 0.08085695648061325, "vocab": 144097, "n_null": 1985, "n_holdout_inner": 226,
              "null_n_exceed": 19, "holdout_inner_n_exceed": 3},
}

# ---------------------------------------------------------------- T3b (PREREG_02 rev 8): black-only candidate pools
# The official competition-1 pools hold black games only (AIdea data page, DATA.md 2026-09-28). Against the
# registered mixed-colour reference such profiles lose most of T3's power (audit results/test02/design/
# official_structure: TPR 0.125 / 0.136 instead of 0.380 / 0.430). T3b is used for a pool whose candidate black
# share b_c is exactly 1: reference = each fit player's BLACK fit games; null = fit players with >= 100 black
# games, first 100 black games in the T3 hash order, own column left out; same tokens, weights, statistic, q99 rule
# and pool rule as T3.
T3B_MIN_BLACK = 100
# A pool left on the registered T3 whose candidates are almost all one colour (max(b_c, 1 - b_c) >= 0.9, b_c not
# exactly 1: a non-B game in an official pool, or an all-white pool) has little power there (black-only profiles:
# TPR 0.125 / 0.136, P(declare | pi = 0.5) = 0, t3b_null.json registered_T3_on_black_only_profiles): no overlap
# declared is then 'not determinable (low power)', never PASS (PREREG_02 rev 8, 12.2; set before any test data,
# not calibrated between 0.9 and 1)
T3_LOW_POWER_SHARE = 0.9
T3B_NULL_FILES = ("t3b_null.json", "t3b_null_raw.npz", "t3b_null_canon.npz")
T3B_PINS = {      # first computed 2026-09-28 (precompute --part t3b; equal to the scratch audit's q99)
    "reference": {"players": 5767, "black_games": 360581,
                  "sha256_sorted_player_ids": "9940feed82f435a770030a17b32b5bbf7dc1002daa61616a989b24580bdf664f"},
    "raw": {"S2_q99": 0.11054835109258787, "vocab": 146421, "n_null": 1133, "n_holdout_inner": 112,
            "null_n_exceed": 11, "holdout_inner_n_exceed": 0},
    "canon": {"S2_q99": 0.08269886422035472, "vocab": 144097, "n_null": 1133, "n_holdout_inner": 112,
              "null_n_exceed": 11, "holdout_inner_n_exceed": 0},
}
# the scratch audit's values (results/test02/design/official_structure/sim_t3b.json); recorded, not required
AUDIT_T3B_Q99 = {"raw": 0.11054835109258787, "canon": 0.08269886422035472}

# ---------------------------------------------------------------- T4 (9.9)
COLOUR_MAX_DIFF = 0.5
UPPER_RIGHT_MIN = 0.60

CONSEQUENCE = {
    ("T1", "PASS"): "none; the raw files are used as they are",
    ("T1", "NORMALIZED"): "normalized copies (only the whitelisted transforms listed in 'normalize') are "
                          "used everywhere, locally and on Colab (t02_test_bundle.py build)",
    ("T1", "FAIL_STRUCT"): "stop; write a revision before any prediction",
    ("T1", "FAIL_OPENING"): "rank: t02_rank.py apply --fallback R10; player: the per-pool procedure is kept "
                            "(self-retrieval and guard react on their own); T3 = not evaluable",
    ("T2", "PASS"): "none",
    ("T2", "FAIL"): "open split v2 and write a new registration; uploads of the frozen 甲 continue; no "
                    "prediction may use the copies before the new registration exists",
    ("T3", "PASS"): "none: no overlap declared (overlap of about 20% of the candidates or less is not "
                    "detectable)",
    ("T3", "OVERLAP"): "consider 乙 under a new registration; uploads of 甲 continue",
    ("T3", "NOT_EVALUABLE"): "T1 is FAIL_OPENING: T3 is 'not evaluable' (not 'no overlap')",
    ("T3", "LOW_POWER"): "no overlap declared, but not 'no overlap': a pool ran the registered T3 with candidates "
                         "almost all of one colour (max(b_c, 1 - b_c) >= 0.9, b_c not exactly 1), where T3 has little "
                         "power; record 'not determinable (low power)' (PREREG_02 rev 8); T2(a) still finds copies; "
                         "uploads of 甲 continue",
    ("T3", "RECORD_ONLY"): "rank: only T3q (question profiles) is recorded; it decides nothing",
    ("T3", "ERROR"): "the rebuilt reference differs from the committed null, so the registered rule cannot "
                     "be applied; find the cause (a changed null needs a revision)",
    ("T4", "PASS"): "none (the rest of T4 is record-only)",
    ("T4", "GUARD_LEVEL3"): "|b_c - b_q| > 0.5 in a pool: player guard level 3 (CNN only) for that pool; "
                            "enforced inside p02_player_fuse.guard_decision, T4 only reports it",
    ("T4", "RECORD_ONLY"): "none (rank: record-only)",
    ("T5", "RECORD_ONLY"): "none (record-only)",
    ("T6", "RECORD_ONLY"): "none (descriptive; without an explicit official statement the one-to-one "
                           "structure is not used)",
}


def log(*a):
    print(*a, flush=True)


def _po():
    import p02_player_open as po  # noqa: E402  (T3 tokens and the T6 simulation only)
    return po


def _sp():
    import scipy.sparse as sp
    return sp


# ---------------------------------------------------------------- small helpers
def refuse_forbidden(path):
    if path is not None and _FORBIDDEN in os.path.basename(str(path)).lower():
        raise SystemExit(f"refusing to open {path!r}: t02 tools never read validation label files")
    return path


def _real(p):
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def inside(path, root):
    a, b = _real(path), _real(root)
    return a == b or a.startswith(b.rstrip(os.sep) + os.sep)


def same_path(a, b):
    return _real(a) == _real(b)


def rel(path):
    try:
        r = os.path.relpath(os.path.realpath(path), os.path.realpath(REPO))
    except ValueError:        # another drive
        return os.path.abspath(path)
    return r.replace(os.sep, "/") if not r.startswith("..") else os.path.abspath(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(refuse_forbidden(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_file(path):
    """SHA-256 with CRLF folded to LF (git autocrlf may rewrite line ends of committed text)."""
    with open(refuse_forbidden(path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return [json_safe(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.str_):
        return str(o)
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(json_safe(obj), f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(refuse_forbidden(path), encoding="utf-8") as f:
        return json.load(f)


def save_npz(path, **arrays):
    """np.savez_compressed with fixed member timestamps, so equal arrays give equal bytes."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arr), allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            z.writestr(zi, buf.getvalue())
    os.replace(tmp, path)


def load_npz(path):
    with np.load(refuse_forbidden(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def git(*args):
    try:
        r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True)
    except OSError as e:
        return 127, str(e)
    return r.returncode, r.stdout.strip()


def git_head():
    rc, out = git("rev-parse", "HEAD")
    return out if rc == 0 else None


def not_committed_clean(paths):
    """Problems ([] when every path is tracked and has no change against HEAD)."""
    probs = []
    rels = [rel(p) for p in paths]
    for r in rels:
        rc, _ = git("ls-files", "--error-unmatch", "--", r)
        if rc != 0:
            probs.append(f"{r} is not tracked by git")
    rc, st = git("status", "--porcelain", "--", *rels)
    if rc != 0:
        probs.append("git status failed")
    elif st:
        probs.append("uncommitted changes: " + "; ".join(st.splitlines()))
    return probs


def ceil_share(share, n):
    return int(math.ceil(share * n - 1e-12))


def t2_threshold(n_games):
    """Copies needed for FAIL in one unit: max(10, ceil(0.001 x games in the unit's test file))."""
    return max(T2_MIN_COPIES, -(-int(n_games) // 1000))


def t3_needed(n_candidates):
    """Candidates above q99 needed to declare overlap: ceil(0.10 x pool size) (40 for 400)."""
    return ceil_share(T3_SHARE, n_candidates)


def t3_pool_overlap(n_raw, n_canon, n_candidates):
    """The registered T3 pool rule: overlap iff raw OR canon has >= ceil(0.10 x pool) above its q99."""
    need = t3_needed(n_candidates)
    return bool(n_raw >= need or n_canon >= need)


def t3_low_power(b_c, variant):
    """PREREG_02 rev 8: a pool on the registered T3 (not T3b) whose candidates are almost all one colour,
    max(b_c, 1 - b_c) >= T3_LOW_POWER_SHARE, is low-power (no declared overlap -> 'not determinable', not PASS)."""
    return bool(variant == "T3" and b_c is not None and max(b_c, 1.0 - b_c) >= T3_LOW_POWER_SHARE - 1e-12)


def q99(x):
    return float(np.quantile(x, 0.99, method="higher"))


def quantiles(x, qs=(0.0, 0.01, 0.5, 0.9, 0.99, 1.0)):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return None
    return {str(q): float(np.quantile(x, q)) for q in qs}


def provenance(mode, inputs=None, rerun_reason=None):
    try:
        import scipy
        scipy_v = scipy.__version__
    except ImportError:
        scipy_v = None
    return {"tool": "tools/t02_checks.py", "tool_sha256": sha256_text_file(os.path.abspath(__file__)),
            "tool_committed_clean": not not_committed_clean([os.path.abspath(__file__)]),
            "c10data_sha256": sha256_text_file(os.path.join(HERE, "c10data.py")),
            "p02_moves_sha256": sha256_text_file(os.path.join(HERE, "p02_moves.py")),
            "p02_player_open_sha256": sha256_text_file(os.path.join(HERE, "p02_player_open.py")),
            "git_head": git_head(), "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy_v, "mode": mode, "argv": sys.argv[1:],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "inputs": inputs or {},
            "rerun_reason": rerun_reason}


# ---------------------------------------------------------------- game keys (T2) and first-move class
SYM = pm.SYM_TABLE
FIRST_CLASS = {p: min(SYM[s][p] for s in range(8)) for p in SYM[0]}


def _tie_syms(head):
    keys = [tuple(SYM[s][p] for p in head) for s in range(8)]
    best = min(keys)
    return [s for s in range(8) if keys[s] == best]


def canon_points(pts, n=None):
    """The D4-minimal concatenated point string of the first n moves (all when None); '' for none.

    Exact: every point is two letters, so string order = order of the point tuples, and the minimum
    over all 8 images starts with the minimal first-12 prefix, i.e. it lies in the tie set."""
    seq = pts if n is None else pts[:n]
    if not seq:
        return ""
    return min("".join(SYM[s][p] for p in seq) for s in _tie_syms(seq[:M_TIE]))


def _h12(s):
    return hashlib.sha1(s.encode("ascii")).digest()[:KEY_BYTES]


def game_key(pts, n=None):
    s = canon_points(pts, n)
    return _h12(s) if s else None


def game_keys(pts):
    """(full, K100, K60) keys. For n >= 12 the tie set of the first n moves is that of the whole
    game, and the minimum of equal-length strings has the minimal prefix, so K_n = the key of the
    first 2n letters of the whole-game minimum (asserted against game_key in the tests)."""
    if not pts:
        return None, None, None
    s = canon_points(pts)
    n = len(pts)
    return (_h12(s), _h12(s[:2 * K_COPY]) if n >= K_COPY else None,
            _h12(s[:2 * K_RECORD]) if n >= K_RECORD else None)


def first_class(point):
    return FIRST_CLASS.get(point) if point else None


# ---------------------------------------------------------------- SGF grammar (T1.4)
FAST_SGF = re.compile(r"\(;SZ\[19\](?:;[BW]\[[a-s]{2}\])*\)\s*")
TOKEN_RE = re.compile(r"\s*(?:(?P<p>[;()])|(?P<id>[A-Za-z]+)\s*(?P<v>(?:\[(?:[^\]\\]|\\.)*\]\s*)+)|(?P<bad>\S))",
                      re.S)
VAL_RE = re.compile(r"\[((?:[^\]\\]|\\.)*)\]", re.S)
POINT_RE = re.compile(r"[a-s]{2}")
PASS_RE = re.compile(r";\s*[BW]\s*\[(?:tt)?\]")
SETUP_RE = re.compile(r"(?<![A-Za-z])(AB|AW|AE|HA|PL)\s*\[")
SZ_RE = re.compile(r"(?<![A-Za-z])SZ\s*\[([^\]]*)\]")


def sgf_props(sgf):
    """[(identifier, [values])] in order, and a parse error (None when the text parses)."""
    pos, n, props = 0, len(sgf), []
    while pos < n:
        m = TOKEN_RE.match(sgf, pos)
        if m is None:
            if sgf[pos:].strip():
                return props, "unparsed text at %d" % pos
            break
        if m.group("bad") is not None:
            return props, "unexpected %r at %d" % (m.group("bad"), m.start("bad"))
        if m.group("id") is not None:
            props.append((m.group("id"), VAL_RE.findall(m.group("v"))))
        pos = m.end()
    return props, None


def sgf_check(sgf):
    """Per-game T1.4 facts. moves = the MOVE_RE sequence (no limit)."""
    moves = MOVE_RE.findall(sgf)
    cols = "".join(c for c, _ in moves)
    alternating = cols == ("BW" * (len(cols) // 2 + 1))[:len(cols)]
    out = {"moves": moves, "setup": (), "sz": "19", "pass": False, "parse_error": None,
           "other_props": (), "regex_mismatch": False, "bad_move_values": 0}
    if not FAST_SGF.fullmatch(sgf):
        props, err = sgf_props(sgf)
        out["parse_error"] = err
        if err is None:
            ids = [k for k, _ in props]
            out["setup"] = tuple(sorted({k for k in ids if k in SETUP_PROPS}))
            sz = [v for k, vals in props if k == "SZ" for v in vals]
            out["sz"] = None if not sz else (sz[0].strip() if len(sz) == 1 else "|".join(sz))
            out["other_props"] = tuple(sorted({k for k in ids if k not in ("SZ", "B", "W")}))
            parsed = []
            for k, vals in props:
                if k in ("B", "W"):
                    for v in vals:
                        if v in ("", "tt"):
                            out["pass"] = True
                        elif POINT_RE.fullmatch(v):
                            parsed.append((k, v))
                        else:
                            out["bad_move_values"] += 1
            out["regex_mismatch"] = parsed != moves
        else:             # fall back to plain regex scans
            out["setup"] = tuple(sorted(set(SETUP_RE.findall(sgf))))
            sz = SZ_RE.findall(sgf)
            out["sz"] = None if not sz else (sz[0].strip() if len(sz) == 1 else "|".join(sz))
            out["pass"] = bool(PASS_RE.search(sgf))
    a_bad = bool(out["setup"]) or (out["sz"] is not None and out["sz"] != "19")
    reasons = []
    if out["setup"]:
        reasons.append("setup " + "/".join(out["setup"]))
    if out["sz"] is not None and out["sz"] != "19":
        reasons.append("SZ[%s]" % out["sz"])
    if not moves:
        reasons.append("no move")
    elif cols[0] != "B":
        reasons.append("first move W")
    elif not alternating:
        reasons.append("not alternating")
    if out["pass"]:
        reasons.append("pass")
    b_bad = (not moves) or cols[0] != "B" or not alternating or out["pass"]
    out.update(a_bad=a_bad, b_bad=b_bad, reasons=reasons, n_moves=len(moves),
               first=moves[0][1] if moves else None)
    return out


# ---------------------------------------------------------------- CSV tables
class Table:
    """One CSV file read byte-exactly: SHA-256, BOM, line ends, header and rows."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        with open(refuse_forbidden(path), "rb") as f:
            raw = f.read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.bytes = len(raw)
        self.bom = raw.startswith(BOM)
        body = raw[len(BOM):] if self.bom else raw
        crlf = body.count(b"\r\n")
        cr = body.count(b"\r") - crlf
        lf = body.count(b"\n") - crlf
        self.line_counts = {"crlf": crlf, "lf": lf, "cr": cr}
        self.line_ending = ("CR" if cr else "mixed" if (crlf and lf) else "CRLF" if crlf
                            else "LF" if lf else "none")
        self.error = None
        self.header, self.rows, self.blank_rows, self.bad_width = [], [], 0, []
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as e:
            self.error = "not UTF-8: %s" % e
            return
        del raw, body
        csv.field_size_limit(10 ** 8)
        rd = csv.reader(io.StringIO(text, newline=""))
        try:
            self.header = next(rd, [])
            w = len(self.header)
            for k, rec in enumerate(rd, start=2):
                if not rec:
                    self.blank_rows += 1
                elif len(rec) != w:
                    self.bad_width.append(k)
                else:
                    self.rows.append(rec)
        except csv.Error as e:
            self.error = "CSV error: %s" % e
        self.col = {}
        for j, c in enumerate(self.header):
            self.col.setdefault(c, j)

    def column(self, name):
        j = self.col.get(name)
        return None if j is None else [r[j] for r in self.rows]

    def info(self):
        return {"sha256": self.sha256, "bytes": self.bytes, "rows": len(self.rows), "bom": self.bom,
                "line_ending": self.line_ending, "line_counts": self.line_counts,
                "blank_rows": self.blank_rows, "rows_with_other_width": len(self.bad_width),
                "header": self.header, "error": self.error}


def question_games(t):
    """[(question_id, slot, colour, sgf)] for slots 1..num_games (all non-empty slots when num_games
    is not a valid integer); colours upper-cased."""
    out = []
    ci = {c: t.col.get(c) for c in Q_COLS}
    for r in t.rows:
        qid = r[ci["question_id"]] if ci["question_id"] is not None else ""
        ng = r[ci["num_games"]] if ci["num_games"] is not None else ""
        n = int(ng) if re.fullmatch(r"[0-9]+", ng or "") else None
        for i in range(1, N_SLOTS + 1):
            js, jc = ci["sgf_%d" % i], ci["color_%d" % i]
            if js is None:
                continue
            s = r[js]
            if (n is not None and i <= n) or (n is None and s.strip()):
                out.append((qid, i, (r[jc] if jc is not None else "").strip().upper(), s))
    return out


def candidate_games(t):
    """[(player_id, game_id, colour, sgf)] in file order; colours upper-cased."""
    j = {c: t.col.get(c) for c in C_COLS}
    if any(v is None for v in j.values()):
        return []
    return [(r[j["player_id"]], r[j["game_id"]], r[j["color"]].strip().upper(), r[j["sgf_content"]])
            for r in t.rows]


def file_games(t, kind):
    """Uniform [(owner, game_label, colour, sgf)]: owner = question id or candidate id."""
    if kind == "questions":
        return [(q, "%s/sgf_%d" % (q, i), c, s) for q, i, c, s in question_games(t)]
    return [(p, g, c, s) for p, g, c, s in candidate_games(t)]


# ---------------------------------------------------------------- T1
def _add(problems, sub, msg, cap=MAX_EXAMPLES):
    lst = problems[sub]
    if len(lst) < cap:
        lst.append(msg)
    elif len(lst) == cap:
        lst.append("... (more problems of this kind are not listed)")


def _counts_verdict(problems, norms):
    return "FAIL" if problems else ("NORMALIZED" if norms else "PASS")


def _colour_design_record(comp, kind, ro):
    """Record-only view of one file against the official colour design (AIdea data page, DATA.md 2026-09-28):
    competition 1 (player) questions and candidates black only; competition 2 (rank) one colour per question.
    Never changes a verdict (PREREG_02 rev 8)."""
    ro = dict(ro)
    out = {"counts": ro}
    if kind == "candidates":
        out["rows_not_B"] = ro.get("rows_colour_W", 0)
    else:
        after = {k[len("colour_after_num_games_"):]: v for k, v in ro.items()
                 if k.startswith("colour_after_num_games_") and len(k) == len("colour_after_num_games_") + 1}
        out["colour_letters_after_num_games"] = after
        out["questions_with_both_colours"] = ro.get("questions_with_both_colours", 0)
        if kind == "questions":
            out["used_slots_not_B"] = ro.get("used_slots_colour_W", 0)
            out["colour_after_num_games_not_the_question_colour"] = ro.get(
                "colour_after_num_games_not_the_question_colour", 0)
    if comp == "player":
        out["official_design"] = "competition 1: every colour (candidates' color, questions' color_1..color_20) is B"
        out["matches_official_design"] = (out.get("rows_not_B", 0) == 0 and out.get("used_slots_not_B", 0) == 0
                                          and not ro.get("colour_after_num_games_W", 0)
                                          and not ro.get("colour_after_num_games_w", 0))
    else:
        out["official_design"] = "competition 2: all games of a question have one colour (black or white)"
        out["matches_official_design"] = (out.get("questions_with_both_colours", 0) == 0 and
                                          out.get("colour_after_num_games_not_the_question_colour", 0) == 0)
    out["note"] = "record only (PREREG_02 rev 8): a difference from the official design never changes a verdict"
    return out


def t1_check(comp, tests_dir, only=None):
    """T1 on the comp's files in tests_dir. Returns the t1.json content (no provenance)."""
    if comp not in FILES:
        raise SystemExit(f"unknown competition {comp!r}")
    subs = tuple(only) if only else T1_SUBS
    if not set(T1_STRUCT) <= set(subs):
        raise SystemExit("T1.1, T1.2 and T1.3 always run (they are the structure the other checks read)")
    problems = defaultdict(list)
    norm_by_file = defaultdict(set)
    per = {s: {} for s in T1_SUBS}
    files, tables = {}, {}
    for pool, kind, name in FILES[comp]:
        path = os.path.join(tests_dir, name)
        if not os.path.isfile(path):
            files[name] = {"missing": True}
            _add(problems, "T1.1", f"{name}: file missing in {tests_dir}")
            continue
        t = Table(path)
        tables[name] = (pool, kind, t)
        files[name] = t.info()

    # T1.1 columns ------------------------------------------------------------
    col_ok = {}
    for name, (pool, kind, t) in tables.items():
        exp = Q_COLS if kind == "questions" else C_COLS
        hdr = t.header
        dup = sorted(c for c, k in Counter(hdr).items() if k > 1)
        missing = [c for c in exp if c not in hdr]
        extra = [c for c in hdr if c not in exp]
        d = {"expected_order": hdr == exp, "missing": missing, "extra": extra, "duplicated": dup}
        if t.error:
            _add(problems, "T1.1", f"{name}: {t.error}")
        if dup or missing or extra:
            _add(problems, "T1.1", f"{name}: columns differ from the README set (missing {missing}, "
                                   f"extra {extra}, duplicated {dup})")
        elif hdr != exp:
            norm_by_file[name].add("reorder_columns_to_readme")
        col_ok[name] = not (dup or missing or extra or t.error)
        per["T1.1"][name] = d

    # T1.2 counts and ids -------------------------------------------------------
    qids_by_pool, pids_by_pool, gids_by_pool = {}, {}, {}
    for name, (pool, kind, t) in tables.items():
        d = {"rows": len(t.rows), "blank_rows": t.blank_rows, "rows_with_other_width": len(t.bad_width)}
        if t.bad_width:
            _add(problems, "T1.2", f"{name}: {len(t.bad_width)} rows with a different number of fields "
                                   f"(first records {t.bad_width[:5]})")
        if not col_ok.get(name):
            per["T1.2"][name] = d
            continue
        pre = ID_PREFIX[pool]
        if kind == "questions":
            qids = t.column("question_id")
            dups = [q for q, k in Counter(qids).items() if k > 1]
            bad_pre = [q for q in qids if not q.startswith(pre)]
            empty = sum(1 for q in qids if not q)
            d.update(duplicated_ids=len(dups), wrong_prefix=len(bad_pre), empty_ids=empty)
            if len(qids) != N_QUESTIONS:
                _add(problems, "T1.2", f"{name}: {len(qids)} questions, expected {N_QUESTIONS}")
            if dups:
                _add(problems, "T1.2", f"{name}: duplicated question_id {dups[:5]}")
            if bad_pre:
                _add(problems, "T1.2", f"{name}: {len(bad_pre)} question_id without prefix {pre!r} "
                                       f"(e.g. {bad_pre[:3]})")
            if empty:
                _add(problems, "T1.2", f"{name}: {empty} empty question_id")
            qids_by_pool[pool] = set(qids)
        else:
            pids, gids = t.column("player_id"), t.column("game_id")
            per_pid = Counter(pids)
            not100 = {p: k for p, k in per_pid.items() if k != GAMES_PER_CANDIDATE}
            gdups = [g for g, k in Counter(gids).items() if k > 1]
            ppre = PLAYER_PREFIX[pool]
            bad_p = [p for p in per_pid if not p.startswith(ppre)]
            bad_g = [g for g in gids if not g.startswith(pre)]
            d.update(players=len(per_pid), games_per_player=dict(Counter(per_pid.values())),
                     duplicated_game_ids=len(gdups), wrong_player_prefix=len(bad_p),
                     wrong_game_prefix=len(bad_g))
            if len(gids) != N_CANDIDATES * GAMES_PER_CANDIDATE:
                _add(problems, "T1.2", f"{name}: {len(gids)} rows, expected {N_CANDIDATES * GAMES_PER_CANDIDATE}")
            if len(per_pid) != N_CANDIDATES:
                _add(problems, "T1.2", f"{name}: {len(per_pid)} distinct player_id, expected {N_CANDIDATES}")
            if not100:
                ex = sorted(not100.items())[:3]
                _add(problems, "T1.2", f"{name}: {len(not100)} candidates without exactly "
                                       f"{GAMES_PER_CANDIDATE} games (e.g. {ex})")
            if gdups:
                _add(problems, "T1.2", f"{name}: duplicated game_id {gdups[:5]}")
            if bad_p:
                _add(problems, "T1.2", f"{name}: {len(bad_p)} player_id without prefix {ppre!r} (e.g. {bad_p[:3]})")
            if bad_g:
                _add(problems, "T1.2", f"{name}: {len(bad_g)} game_id without prefix {pre!r} (e.g. {bad_g[:3]})")
            if any(not p for p in per_pid) or any(not g for g in gids):
                _add(problems, "T1.2", f"{name}: empty player_id or game_id")
            pids_by_pool[pool] = set(per_pid)
            gids_by_pool[pool] = set(gids)
    cross = {}
    if len(qids_by_pool) == 2:
        both = qids_by_pool["public"] & qids_by_pool["private"]
        cross["question_ids_in_both_files"] = len(both)
        if both:
            _add(problems, "T1.2", f"question_id in both files: {sorted(both)[:5]}")
    if len(pids_by_pool) == 2:
        sp_ = pids_by_pool["public"] & pids_by_pool["private"]
        sg_ = gids_by_pool["public"] & gids_by_pool["private"]
        cross.update(player_ids_in_both_pools=len(sp_), game_ids_in_both_pools=len(sg_))
        if sp_:
            _add(problems, "T1.2", f"the pools share {len(sp_)} player_id (e.g. {sorted(sp_)[:3]})")
        if sg_:
            _add(problems, "T1.2", f"the pools share {len(sg_)} game_id (e.g. {sorted(sg_)[:3]})")
    per["T1.2"]["cross"] = cross

    # T1.3 slots and encoding ---------------------------------------------------
    for name, (pool, kind, t) in tables.items():
        d = {"bom": t.bom, "line_ending": t.line_ending}
        if t.bom:
            norm_by_file[name].add("strip_bom")
        if t.line_ending == "mixed":
            norm_by_file[name].add("crlf_to_lf")
        elif t.line_ending == "CR":
            _add(problems, "T1.3", f"{name}: bare CR line ends")
        if not col_ok.get(name):
            per["T1.3"][name] = d
            continue
        c = Counter()
        ro = Counter()          # record-only facts about the official colour design (PREREG_02 rev 8)
        if kind == "questions":
            ci = {k: t.col[k] for k in Q_COLS}
            for r in t.rows:
                qid, ng = r[ci["question_id"]], r[ci["num_games"]]
                if not re.fullmatch(r"[0-9]+", ng) or not K_MIN <= int(ng) <= K_MAX:
                    c["bad_num_games"] += 1
                    _add(problems, "T1.3", f"{name}: {qid}: num_games {ng!r} is not an integer in {K_MIN}..{K_MAX}")
                    continue
                n = int(ng)
                used = set()           # upper-cased colours of slots 1..n
                for i in range(1, N_SLOTS + 1):
                    s, col = r[ci["sgf_%d" % i]], r[ci["color_%d" % i]]
                    if i <= n:
                        if not s.strip():
                            c["empty_game_within_num_games"] += 1
                            _add(problems, "T1.3", f"{name}: {qid}: sgf_{i} empty but num_games = {n}")
                        elif s != s.strip():
                            c["sgf_with_outer_whitespace_record_only"] += 1
                        if col in ("B", "W"):
                            pass
                        elif col in ("b", "w"):
                            c["lower_case_colour"] += 1
                            norm_by_file[name].add("upper_colour")
                        else:
                            c["bad_colour"] += 1
                            _add(problems, "T1.3", f"{name}: {qid}: color_{i} = {col!r}")
                        if col in COLOUR_LETTERS:
                            used.add(col.upper())
                            ro["used_slots_colour_" + col.upper()] += 1
                        continue
                    if s == "" and col == "":
                        continue
                    if s.strip():
                        # an sgf after num_games: never normalized (PREREG_02 rev 8 keeps this FAIL_STRUCT)
                        c["content_after_num_games"] += 1
                        _add(problems, "T1.3", f"{name}: {qid}: slot {i} > num_games = {n} is not empty")
                        continue
                    if s != "" or (col != "" and not col.strip()):
                        c["whitespace_cells_after_num_games"] += 1
                        norm_by_file[name].add("clear_slots_after_num_games_if_empty_whitespace")
                    if col == "" or not col.strip():
                        continue
                    if col in COLOUR_LETTERS:
                        c["colour_after_num_games_with_empty_sgf"] += 1
                        norm_by_file[name].add("clear_colour_after_num_games_if_sgf_empty")
                        ro["colour_after_num_games_%s" % col] += 1
                        if used and col.upper() not in used:
                            ro["colour_after_num_games_not_the_question_colour"] += 1
                    else:
                        c["content_after_num_games"] += 1
                        _add(problems, "T1.3", f"{name}: {qid}: slot {i} > num_games = {n}: color_{i} = {col!r}")
                ro["questions"] += 1
                if len(used) > 1:
                    ro["questions_with_both_colours"] += 1
                elif used:
                    ro["questions_single_colour_" + next(iter(used))] += 1
        else:
            ci = {k: t.col[k] for k in C_COLS}
            for r in t.rows:
                col, s = r[ci["color"]], r[ci["sgf_content"]]
                if col in ("B", "W"):
                    pass
                elif col in ("b", "w"):
                    c["lower_case_colour"] += 1
                    norm_by_file[name].add("upper_colour")
                else:
                    c["bad_colour"] += 1
                    _add(problems, "T1.3", f"{name}: {r[ci['game_id']]}: color = {col!r}")
                if col in COLOUR_LETTERS:
                    ro["rows_colour_" + col.upper()] += 1
                if not s.strip():
                    c["empty_sgf"] += 1
                    _add(problems, "T1.3", f"{name}: {r[ci['game_id']]}: empty sgf_content")
                elif s != s.strip():
                    c["sgf_with_outer_whitespace_record_only"] += 1
        d["counts"] = dict(c)
        d["official_colour_design_record_only"] = _colour_design_record(comp, kind, ro)
        per["T1.3"][name] = d

    # T1.4 / T1.5 (and the record-only move counts) --------------------------------
    move_counts = {}
    run_content = [s for s in ("T1.4", "T1.5") if s in subs]
    if run_content:
        for name, (pool, kind, t) in tables.items():
            if not col_ok.get(name):
                continue
            games = file_games(t, kind)
            n = len(games)
            c, other, ex = Counter(), Counter(), []
            n_bad, n_s99, nm = 0, 0, []
            for owner, label, colour, sgf in games:
                g = sgf_check(sgf)
                nm.append(g["n_moves"])
                if first_class(g["first"]) in S99:
                    n_s99 += 1
                for p in g["other_props"]:
                    other[p] += 1
                c["sz_missing_record_only"] += g["sz"] is None
                c["parse_error_record_only"] += g["parse_error"] is not None
                c["regex_vs_parse_mismatch_record_only"] += g["regex_mismatch"]
                c["bad_move_values_record_only"] += g["bad_move_values"] > 0
                if g["a_bad"] or g["b_bad"]:
                    n_bad += 1
                    c["a_violations"] += g["a_bad"]
                    c["b_violations"] += g["b_bad"]
                    for rsn in g["reasons"]:
                        c["setup" if rsn.startswith("setup") else "sz_not_19" if rsn.startswith("SZ[")
                          else rsn.replace(" ", "_")] += 1
                    if len(ex) < MAX_EXAMPLES:
                        ex.append({"game": label, "reasons": g["reasons"]})
            share_bad = n_bad / n if n else 0.0
            per["T1.4"][name] = {"games": n, "violating_games": n_bad, "share": share_bad,
                                 "max_share": T1_4_MAX_SHARE, "fails": share_bad > T1_4_MAX_SHARE,
                                 "counts": dict(c), "other_properties_record_only": dict(other),
                                 "examples": ex}
            share99 = n_s99 / n if n else 0.0
            per["T1.5"][name] = {"games": n, "first_move_in_S99": n_s99, "share": share99,
                                 "min_share": T1_5_MIN_SHARE, "fails": share99 < T1_5_MIN_SHARE}
            if nm:
                a = np.asarray(nm)
                move_counts[name] = {"min": int(a.min()), "p1": float(np.percentile(a, 1)),
                                     "median": float(np.median(a)), "max": int(a.max()), "games": n}
    for sub in ("T1.4", "T1.5"):
        for name, d in per[sub].items():
            if d.get("fails"):
                _add(problems, sub, f"{name}: share {d['share']:.4f} "
                                    f"{'>' if sub == 'T1.4' else '<'} {d.get('max_share', d.get('min_share'))}")

    # verdicts ------------------------------------------------------------------
    sub_norm = {"T1.1": {"reorder_columns_to_readme"},
                "T1.3": {"strip_bom", "crlf_to_lf", "upper_colour",
                         "clear_slots_after_num_games_if_empty_whitespace",
                         "clear_colour_after_num_games_if_sgf_empty"}}
    all_norm = set().union(*norm_by_file.values()) if norm_by_file else set()
    subchecks = {}
    for sub in T1_SUBS:
        if sub not in subs:
            subchecks[sub] = {"verdict": "NOT_RUN", "files": {}, "problems": []}
            continue
        v = _counts_verdict(problems.get(sub), all_norm & sub_norm.get(sub, set()))
        subchecks[sub] = {"verdict": v, "files": per[sub], "problems": problems.get(sub, [])}
    struct_fail = any(subchecks[s]["verdict"] == "FAIL" for s in T1_STRUCT)
    opening_fail = any(subchecks[s]["verdict"] == "FAIL" for s in ("T1.4", "T1.5"))
    transforms = [x for x in NORMALIZE_WHITELIST if x in all_norm]
    verdict = ("FAIL_STRUCT" if struct_fail else "FAIL_OPENING" if opening_fail
               else "NORMALIZED" if transforms else "PASS")
    return {"format": "c10-test02 t1 v1", "comp": comp, "verdict": verdict,
            "consequence": CONSEQUENCE[("T1", verdict)],
            "normalize": {k: [x for x in NORMALIZE_WHITELIST if x in v] for k, v in sorted(norm_by_file.items()) if v},
            "normalize_transforms": transforms,
            "normalize_note": "per official file name, only the files that need a transform (whitelist order); "
                              "empty when the verdict is PASS",
            "normalize_whitelist": list(NORMALIZE_WHITELIST),
            "subchecks_run": list(subs), "complete": set(subs) == set(T1_SUBS),
            "tests_dir": os.path.abspath(tests_dir), "files": files, "subchecks": subchecks,
            "record_only": {"move_counts": move_counts,
                            "line_ending_note": "uniform CRLF or LF is accepted as is (every reader parses both "
                                                "the same); only mixed line ends ask for crlf_to_lf"},
            "rules": {"T1.3": "slots 1..num_games: sgf non-empty and color B/W (b/w -> upper_colour); slots "
                              "after num_games: an sgf with content is FAIL_STRUCT; whitespace-only cells -> "
                              "clear_slots_after_num_games_if_empty_whitespace; a colour letter B/W/b/w with an "
                              "empty or whitespace-only sgf -> clear_colour_after_num_games_if_sgf_empty (PREREG_02 "
                              "rev 8); any other colour value there is FAIL_STRUCT. The official colour design "
                              "(competition 1 all B, competition 2 one colour per question) is recorded only",
                      "T1.4": "FAIL_OPENING iff more than %.0f%% of a file's games violate (a) no AB/AW/AE/HA/PL "
                              "and SZ = 19 (a missing SZ is recorded only) or (b) at least one move, first move "
                              "B, strictly alternating, no pass (B[] / B[tt])" % (100 * T1_4_MAX_SHARE),
                      "T1.5": "PASS iff the share of games whose first-move D4 class is in S99 = %s is >= %.2f"
                              % (list(S99), T1_5_MIN_SHARE)}}


# ---------------------------------------------------------------- modes, paths and guards
def mode_of(args):
    return ("dry-run" if getattr(args, "dry_run", False) else "smoke" if getattr(args, "smoke", False)
            else "rehearsal" if getattr(args, "rehearsal", False) else "real")


def guard_out(out, mode):
    for ro in READ_ONLY:
        if inside(out, ro):
            raise SystemExit(f"refusing to write into {out!r}: it is inside the read-only {ro!r}")
    if mode != "real" and inside(out, REPO):
        raise SystemExit(f"--{mode} refused: output dir {out!r} is inside the repository; use a scratch dir")


def _infer_root(tests_dir):
    parts = os.path.normpath(os.path.abspath(tests_dir)).split(os.sep)
    low = [p.lower() for p in parts]
    for tail in (["data", "test", "raw"], ["data", "test"]):
        if low[-len(tail):] == tail:
            return os.sep.join(parts[:-len(tail)]) or os.sep
    return None


CHECKS_BUILT = os.path.join(CHECKS_DIR, "built")
REAL_T1 = ((DATA_TEST_RAW, CHECKS_DIR), (DATA_TEST, CHECKS_BUILT))   # raw files; built copies (bundle re-check)
REAL_RUN = ((DATA_TEST, CHECKS_DIR),)


def resolve_paths(args, mode, real_pairs):
    """(tests_dir, out_dir) for t1 / run / t6 after the mode rules. real_pairs: allowed (tests, out)."""
    tests, out = args.tests_dir, args.out
    if mode == "real":
        if args.root:
            raise SystemExit("--root is only for --dry-run / --smoke / --rehearsal")
        tests = os.path.abspath(tests or real_pairs[0][0])
        pair = [p for p in real_pairs if same_path(tests, p[0])]
        if not pair:
            raise SystemExit(f"real mode reads {[rel(t) for t, _ in real_pairs]} only (got {tests}); "
                             f"use --dry-run for other trees")
        want = pair[0][1]
        out = os.path.abspath(out or want)
        if not same_path(out, want):
            raise SystemExit(f"real mode with --tests-dir {rel(tests)} writes only to {rel(want)} (got {out})")
    else:
        root = args.root
        if tests is None:
            if root is None:
                raise SystemExit("give --tests-dir or --root")
            tests = os.path.join(root, "data", "test")
        if out is None:
            base = root or _infer_root(tests)
            if base is None:
                raise SystemExit("give --out (or --root): cannot derive the output dir from --tests-dir")
            out = os.path.join(base, "results", "test02", "checks")
        tests, out = os.path.abspath(tests), os.path.abspath(out)
    if inside(tests, LOCAL_DIR):
        raise SystemExit(f"refusing to read test files from {tests!r}: it is inside data/local")
    guard_out(out, mode)
    return tests, out


def resolve_null(args, mode):
    nd = os.path.abspath(args.null_dir or NULL_DIR)
    if mode == "real" and not same_path(nd, NULL_DIR):
        raise SystemExit(f"real mode uses the committed null in {rel(NULL_DIR)} only")
    return nd


def check_rerun(args, paths):
    exist = [p for p in paths if os.path.exists(p)]
    if exist and not args.rerun:
        raise SystemExit(f"refusing to overwrite {[rel(p) for p in exist]}; use --rerun --reason TEXT")
    if args.rerun and not (args.reason and args.reason.strip()):
        raise SystemExit("--rerun needs --reason TEXT (stored in the report)")
    return args.reason.strip() if args.rerun else None


def _validation_sgf_hashes():
    """SHA-1 of every game in the local validation question / candidate files (c10data leak-check
    hashes). Used only as a membership guard: content checks never run on validation games."""
    return c10data._sgf_hashes_in_local(LOCAL_DIR)


def refuse_validation_games(sgfs):
    local = _validation_sgf_hashes()
    if not local:
        raise SystemExit("cannot load the validation game hashes (data/local); content checks refuse to run "
                         "outside real mode without that guard")
    n, hits = 0, 0
    for s in sgfs:
        n += 1
        hits += hashlib.sha1(s.encode("utf-8")).digest() in local
    if hits:
        raise SystemExit(f"{hits} of {n} games in this tree are local validation games: content checks (T1.4, "
                         f"T1.5, T2-T5) never run on validation-derived files (spec 0.11); use "
                         f"t1 --only T1.1,T1.2,T1.3")
    return {"games_checked": n, "validation_hashes": len(local), "hits": 0}


REAL_VALIDATION_REFUSE_SHARE = 0.5


def real_validation_guard(named_sgfs):
    """Real mode (review finding 2026-09-26; PREREG_02 rev 6 10.16). An official file may legitimately hold a few
    training games (T2 looks for copies), and some of them can be local validation games, so real mode records the
    hits per file instead of refusing on the first one. A file whose games are mostly local validation games is a
    validation-derived tree (rehearsal / smoke / t02_fake_tests) that got into data/test: content checks refuse it.
    named_sgfs: {file name: iterable of SGF strings}."""
    local = _validation_sgf_hashes()
    rule = ("real mode: hits are recorded; a file with share >= %.2f of local validation games is refused "
            "(validation-derived tree)" % REAL_VALIDATION_REFUSE_SHARE)
    if not local:
        log("warning: cannot load the validation game hashes (data/local); the real-mode guard is recorded as "
            "unavailable")
        return {"available": False, "rule": rule}
    files, total_n, total_hits = {}, 0, 0
    for name, sgfs in named_sgfs.items():
        n = hits = 0
        for s in sgfs:
            n += 1
            hits += hashlib.sha1(s.encode("utf-8")).digest() in local
        files[name] = {"games": n, "validation_hits": hits, "share": round(hits / n, 6) if n else 0.0}
        total_n, total_hits = total_n + n, total_hits + hits
    bad = sorted(k for k, v in files.items() if v["games"] and v["share"] >= REAL_VALIDATION_REFUSE_SHARE)
    if bad:
        raise SystemExit(f"{', '.join(bad)}: {max(files[k]['share'] for k in bad):.0%} of the games are local "
                         f"validation games. This is a validation-derived tree (rehearsal / smoke / fake), not the "
                         f"official files: content checks (T1.4, T1.5, T2-T5) never run on it (spec 0.11). Check "
                         f"data/test/raw/SOURCE.json and ingest the files downloaded from AIdea again")
    if total_hits:
        log(f"warning: {total_hits} of {total_n} games are local validation games (recorded in the provenance; "
            f"T2 reports copies of training games)")
    return {"available": True, "rule": rule, "games_checked": total_n, "validation_hashes": len(local),
            "hits": total_hits, "files": files}


def read_manifest(tests_dir, comp, t1):
    """Real-mode binding of the built test files: MANIFEST (kind official) and the raw t1.json."""
    mp = os.path.join(tests_dir, "MANIFEST.json")
    if not os.path.isfile(mp):
        raise SystemExit(f"{rel(mp)} missing: build the test bundle first (t02_test_bundle.py build)")
    man = read_json(mp)
    if man.get("kind") != "official":
        raise SystemExit(f"{rel(mp)} kind is {man.get('kind')!r}; real mode needs 'official'")
    info = {"manifest_sha256": sha256_file(mp), "files": {}}
    for _pool, _kind, name in FILES[comp]:
        ent = man.get("files", {}).get(name)
        if ent is None:
            raise SystemExit(f"{name} not in {rel(mp)}")
        cur = sha256_file(os.path.join(tests_dir, name))
        if cur != ent.get("sha256"):
            raise SystemExit(f"{name}: sha256 {cur} differs from MANIFEST {ent.get('sha256')}")
        raw = t1["files"].get(name, {}).get("sha256")
        if ent.get("source_sha256") != raw:
            raise SystemExit(f"{name}: MANIFEST source_sha256 {ent.get('source_sha256')} differs from the raw "
                             f"file checked by t1.json ({raw})")
        info["files"][name] = cur
    return info


def update_summary(out_comp, comp, entries):
    path = os.path.join(out_comp, "summary.json")
    s = read_json(path) if os.path.exists(path) else {"format": "c10-test02 checks summary v1", "comp": comp,
                                                       "checks": {}}
    for check, (verdict, details_file) in entries.items():
        dp = os.path.join(out_comp, details_file)
        s["checks"][check] = {"verdict": verdict,
                              "consequence": CONSEQUENCE.get((check.split("_")[0], verdict), ""),
                              "details_file": details_file,
                              "sha256": sha256_file(dp) if os.path.exists(dp) else None}
    s["checks"] = {k: s["checks"][k] for k in sorted(s["checks"])}
    s["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    s["git_head"] = git_head()
    write_json(path, s)
    return s


# ---------------------------------------------------------------- training data (compact, load_rows order)
class Train:
    """Training games of split-v1 part 'train' (fit + inner) in c10data.load_rows order.

    Per game: player, game id, colour, the first 60 moves (p02_player_open.moves_of, for tokens),
    the move count and first point, and (keys=True) the T2 keys of the whole MOVE_RE sequence."""

    def __init__(self, rows, inner, keys=True, extra_lengths=(), grammar=False):
        po = _po()
        n = len(rows)
        self.n = n
        self.inner_pids = set(inner)
        self.games_of = {}
        self.level_of = {}
        self.pids = []
        self.pid_idx = np.empty(n, np.int32)
        self.gid, self.colour, self.cols60, self.pts60 = [None] * n, [None] * n, [None] * n, [None] * n
        self.nmov = np.empty(n, np.int32)
        self.keys = {"full": [], "k100": [], "k60": []} if keys else None
        self.extra = {int(m): [] for m in extra_lengths} if keys else {}
        self.stats = Counter()
        pindex = {}
        for i, r in enumerate(rows):
            pid = r["player_id"]
            j = pindex.get(pid)
            if j is None:
                j = pindex[pid] = len(self.pids)
                self.pids.append(pid)
                self.games_of[pid] = []
                self.level_of[pid] = r["rank"]
            self.games_of[pid].append(i)
            self.pid_idx[i] = j
            self.gid[i] = r["game_id"]
            if r["color"] not in ("B", "W"):
                raise SystemExit(f"unexpected colour {r['color']!r} in training game {r['game_id']}")
            self.colour[i] = r["color"]
            sgf = r["sgf_content"]
            full = MOVE_RE.findall(sgf)
            mv = po.moves_of(sgf)
            self.cols60[i] = sys.intern("".join(c for c, _ in mv))
            self.pts60[i] = "".join(p for _, p in mv)
            self.nmov[i] = len(full)
            cols = "".join(c for c, _ in full)
            self.stats["alternating_from_B"] += cols == ("BW" * (len(cols) // 2 + 1))[:len(cols)] and cols[:1] == "B"
            if grammar:
                g = sgf_check(sgf)
                self.stats["t1_4_violations"] += g["a_bad"] or g["b_bad"]
                self.stats["passes"] += g["pass"]
                self.stats["setup"] += bool(g["setup"])
                self.stats["sz_not_19"] += g["sz"] != "19"
            if keys:
                pts = [p for _, p in full]
                kf, k1, k6 = game_keys(pts)
                self.keys["full"].append(kf)
                self.keys["k100"].append(k1)
                self.keys["k60"].append(k6)
                for m, lst in self.extra.items():
                    lst.append(game_key(pts, m) if len(pts) >= m else None)
        self.inner_game = np.fromiter((self.pids[j] in self.inner_pids for j in self.pid_idx), bool, count=n)
        self.fit = sorted(p for p in self.games_of if p not in self.inner_pids)
        self._order = None
        self._border = None

    def mv60(self, i):
        c, p = self.cols60[i], self.pts60[i]
        return [(c[j], p[2 * j:2 * j + 2]) for j in range(len(c))]

    def first_pt(self, i, k=1):
        s = self.pts60[i]
        return s[2 * (k - 1):2 * k]

    def t3_order(self):
        """Each player's games sorted by sha256('c10-t3|cand|{pid}|{gid}')."""
        if self._order is None:
            self._order = {p: sorted(g, key=lambda i, p=p: hashlib.sha256(
                ("%s|%s|%s" % (T3_SALT, p, self.gid[i])).encode()).hexdigest()) for p, g in self.games_of.items()}
        return self._order

    def black_order(self):
        """Each player's BLACK games in the T3 hash order (T3b, PREREG_02 rev 8)."""
        if self._border is None:
            self._border = {p: [i for i in g if self.colour[i] == "B"] for p, g in self.t3_order().items()}
        return self._border

    def reference_record(self):
        fit_games = sum(len(self.games_of[p]) for p in self.fit)
        return {"players": len(self.fit), "games": fit_games,
                "sha256_sorted_player_ids": hashlib.sha256("\n".join(self.fit).encode()).hexdigest()}

    def reference_record_black(self):
        blk = self.black_order()
        return {"players": len(self.fit), "black_games": sum(len(blk[p]) for p in self.fit),
                "sha256_sorted_player_ids": hashlib.sha256("\n".join(self.fit).encode()).hexdigest()}


def load_train(keys=True, extra_lengths=(), grammar=False):
    t0 = time.time()
    rows = c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part="train", local_dir=LOCAL_DIR)
    inner = c10data.inner_players(c10data.read_split(SPLIT_CSV))
    t1 = time.time()
    log("loaded %d training rows (part 'train' = fit + inner) in %.0fs" % (len(rows), t1 - t0))
    T = Train(rows, inner, keys=keys, extra_lengths=extra_lengths, grammar=grammar)
    del rows
    log("parsed%s in %.0fs" % (" and hashed" if keys else "", time.time() - t1))
    T.load_s = round(t1 - t0, 1)
    T.parse_s = round(time.time() - t1, 1)
    return T


# ---------------------------------------------------------------- T3 engine
def t3_token_matrix(T, orient):
    """Game x token count matrix of every training game (load_rows order), vocabulary in order of
    first appearance; tokens = p02_player_open.game_tokens (both sets, 10 moves)."""
    po, sp = _po(), _sp()
    vocab = {}
    indptr, cols, vals = array("q", [0]), array("i"), array("d")
    for i in range(T.n):
        for t, k in Counter(po.game_tokens(T.colour[i], T.mv60(i), orient)).items():
            cols.append(vocab.setdefault(t, len(vocab)))
            vals.append(k)
        indptr.append(len(cols))
    G = sp.csr_matrix((np.frombuffer(vals, np.float64).copy(), np.frombuffer(cols, np.int32).copy(),
                       np.frombuffer(indptr, np.int64).copy()), shape=(T.n, len(vocab)))
    return G, vocab


def t3_weigh(Mc, idf):
    po = _po()
    m = Mc.tocsr().copy()
    m.eliminate_zeros()
    m.data = 1.0 + np.log(m.data)
    return po._l2_rows(m.multiply(idf).tocsr()).tocsr()


def s2_rows(C):
    top = -np.partition(-C, 1, axis=1)[:, :2]
    return top[:, 0], top[:, 0] - top[:, 1]


def vocab_sha256(vocab):
    toks = sorted(vocab, key=vocab.get)
    return hashlib.sha256("\n".join(toks).encode("utf-8")).hexdigest()


def t3_reference(T, orient, min_games=T3_MIN_GAMES, profile=T3_PROFILE):
    """R (fit players, all their fit games), idf and the null S2 (fit players with >= min_games,
    first `profile` games in hash order, own column left out)."""
    po = _po()
    Gm, vocab = t3_token_matrix(T, orient)
    n, fit = T.n, T.fit
    fcol = {p: j for j, p in enumerate(fit)}
    players, order = T.games_of, T.t3_order()
    Rr = (po.indicator([players[p] for p in fit], n) @ Gm).tocsr()
    df = np.diff(Rr.tocsc().indptr)
    idf = np.log((1.0 + len(fit)) / (1.0 + df)) + 1.0     # tokens only in inner games: df = 0
    R = t3_weigh(Rr, idf)
    el = [p for p in fit if len(players[p]) >= min_games]
    Cf = t3_weigh(po.indicator([order[p][:profile] for p in el], n) @ Gm, idf)
    COS = np.asarray((Cf @ R.T).toarray())
    own = np.array([fcol[p] for p in el], dtype=np.int64)
    COSn = COS.copy()
    COSn[np.arange(len(el)), own] = -np.inf
    s1n, s2n = s2_rows(COSn)
    return {"Gm": Gm, "vocab": vocab, "idf": idf, "R": R, "fit": fit, "fcol": fcol, "el": el, "Cf": Cf,
            "own": own, "COSn": COSn, "s1n": s1n, "s2n": s2n, "q99": q99(s2n), "V": len(vocab)}


def t3_test_matrix(games, vocab, orient):
    """Count matrix of test games [(colour, mv)] on the reference vocabulary plus new columns for
    tokens unseen in training (they only enter the row norm)."""
    po, sp = _po(), _sp()
    V, ext = len(vocab), {}
    indptr, cols, vals = array("q", [0]), array("i"), array("d")
    for colour, mv in games:
        for t, k in Counter(po.game_tokens(colour, mv, orient)).items():
            j = vocab.get(t)
            if j is None:
                j = V + ext.setdefault(t, len(ext))
            cols.append(j)
            vals.append(k)
        indptr.append(len(cols))
    M = sp.csr_matrix((np.frombuffer(vals, np.float64).copy(), np.frombuffer(cols, np.int32).copy(),
                       np.frombuffer(indptr, np.int64).copy()), shape=(len(games), V + len(ext)))
    return M, len(ext)


def t3_profile_s2(ref, games, groups, orient):
    """(top1, S2) of each group profile against R, and the number of tokens unseen in training."""
    po = _po()
    M, n_ext = t3_test_matrix(games, ref["vocab"], orient)
    idf_ext = np.concatenate([ref["idf"], np.full(n_ext, np.log((1.0 + len(ref["fit"])) / 1.0) + 1.0)])
    W = t3_weigh(po.indicator(groups, M.shape[0]) @ M, idf_ext)
    C = np.asarray((W[:, :ref["V"]] @ ref["R"].T).toarray())
    top1, s2 = s2_rows(C)
    return top1, s2, n_ext


def t3_orient_precompute(T, orient, power_sims=5000, t3q_ks=range(K_MIN, K_MAX + 1),
                         min_games=T3_MIN_GAMES, profile=T3_PROFILE, t3q_min=T3Q_MIN_GAMES, ref=None):
    """Port of the design-record scratch t3final.py, per orientation (same order of operations).
    ref: a t3_reference(T, orient, min_games, profile) built by the caller (shared with T3b), else built here."""
    from scipy.stats import binom
    po = _po()
    to = time.time()
    if ref is None:
        ref = t3_reference(T, orient, min_games, profile)
    Gm, idf, R = ref["Gm"], ref["idf"], ref["R"]
    n, fit, players, order = T.n, ref["fit"], T.games_of, T.t3_order()
    el, own, COSn, Cf, s1n, s2n, thr = ref["el"], ref["own"], ref["COSn"], ref["Cf"], ref["s1n"], ref["s2n"], ref["q99"]
    thr1 = q99(s1n)
    lv = np.array([T.level_of[p] for p in el])
    t_null = time.time() - to
    ei = sorted(p for p in T.inner_pids if p in players and len(players[p]) >= min_games)
    Ci = t3_weigh(po.indicator([order[p][:profile] for p in ei], n) @ Gm, idf)
    _, s2i = s2_rows(np.asarray((Ci @ R.T).toarray()))
    lev = [lv_ for lv_ in c10data.LEVELS if (lv == lv_).any()]
    out = {"n_null": len(el), "n_holdout_inner": len(ei), "vocab": ref["V"],
           "vocab_sha256": vocab_sha256(ref["vocab"]), "idf_sha256": sha256_array(idf.astype("<f8")),
           "idf_sha256_rule": "sha256 of the idf vector as little-endian float64 bytes, vocabulary order = "
                              "first appearance over all training games in load_rows order",
           "S2_q99": thr, "S1_q99": thr1, "null_n_exceed": int((s2n > thr).sum()),
           "null_S2_quantiles": {q: float(np.quantile(s2n, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)},
           "null_S2_max": float(s2n.max()),
           "null_S2_exceed_by_level": {L: round(float((s2n[lv == L] > thr).mean()), 4) for L in lev},
           "null_worst_level_P_declare": max(float(binom.sf(39, 400, float((s2n[lv == L] > thr).mean())))
                                             for L in lev),
           "holdout_inner_exceed": round(float((s2i > thr).mean()), 4) if len(ei) else None,
           "holdout_inner_n_exceed": int((s2i > thr).sum())}
    pos = {}
    s2_pos_all, m_all = np.zeros(0), np.zeros(0, np.int64)
    for m in (20, 50, 100, None):
        sel = [j for j, p in enumerate(el) if len(order[p]) - profile >= (1 if m is None else m)]
        grp = [order[el[j]][profile:] if m is None else order[el[j]][profile:profile + m] for j in sel]
        key = "all" if m is None else str(m)
        if not sel:
            pos[key] = {"n": 0}
            continue
        Rp = t3_weigh(po.indicator(grp, n) @ Gm, idf)
        oc = np.asarray(Cf[sel].multiply(Rp).sum(1)).ravel()
        C2 = COSn[sel].copy()
        C2[np.arange(len(sel)), own[sel]] = oc
        _, s2p = s2_rows(C2)
        pos[key] = {"n": len(sel), "tpr": round(float((s2p > thr).mean()), 4),
                    "tpr_by_level": {L: round(float((s2p[lv[sel] == L] > thr).mean()), 3)
                                     for L in lev if (lv[sel] == L).any()}}
        if m is None:
            s2_pos_all = s2p
            m_all = np.array([len(order[el[j]]) - profile for j in sel])
    out["pos"] = pos
    bins = [(1, 20), (21, 50), (51, 100), (101, 200), (201, 10 ** 6)]
    out["pos_all_tpr_by_ref_games"] = {
        "%d-%d" % b: {"n": int(((m_all >= b[0]) & (m_all <= b[1])).sum()),
                      "tpr": round(float((s2_pos_all[(m_all >= b[0]) & (m_all <= b[1])] > thr).mean()), 4)
                      if ((m_all >= b[0]) & (m_all <= b[1])).any() else None} for b in bins}
    rng = np.random.default_rng(0)
    p_ref = 5767 / 8007
    power = {}
    for pi in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        fr = []
        for _ in range(power_sims):
            k_ref = rng.binomial(rng.binomial(400, pi), p_ref)
            v = np.concatenate([rng.choice(s2_pos_all, k_ref), rng.choice(s2n, 400 - k_ref)])
            fr.append((v > thr).mean())
        fr = np.array(fr)
        power[str(pi)] = {"P_declare": round(float((fr >= 0.10).mean()), 4),
                          "mean_share_above": round(float(fr.mean()), 4),
                          "p05_share": round(float(np.quantile(fr, 0.05)), 4)}
    out["power_pool_rule"] = power
    tq = time.time()
    eq = [p for p in fit if len(players[p]) >= t3q_min]
    ownq = np.array([ref["fcol"][p] for p in eq], dtype=np.int64)
    Rp = t3_weigh(po.indicator([order[p][K_MAX:] for p in eq], n) @ Gm, idf)
    t3q = {}
    for k in t3q_ks:
        Ck = t3_weigh(po.indicator([order[p][:k] for p in eq], n) @ Gm, idf)
        Ck_cos = np.asarray((Ck @ R.T).toarray())
        Ck_cos[np.arange(len(eq)), ownq] = -np.inf
        _, nk = s2_rows(Ck_cos)
        tk = q99(nk)
        oc = np.asarray(Ck.multiply(Rp).sum(1)).ravel()
        Ck_cos[np.arange(len(eq)), ownq] = oc
        _, pk = s2_rows(Ck_cos)
        t3q[k] = {"q99": tk, "tpr_rest_ge5": round(float((pk > tk).mean()), 4)}
    out["t3q"] = {"population": "fit players with >= %d games (%d); question = first k of the hash order; "
                                "positive reference = games %d.. of the hash order" % (t3q_min, len(eq), K_MAX + 1),
                  "by_k": t3q, "t_s": round(time.time() - tq, 1)}
    out["t_null_s"] = round(t_null, 1)
    out["t_orient_s"] = round(time.time() - to, 1)
    arrays = {"player_id": np.array(el), "level": lv, "null_S2": s2n, "null_S1": s1n,
              "holdout_player_id": np.array(ei), "holdout_S2": s2i, "pos_all_S2": s2_pos_all,
              "pos_all_m": m_all, "idf": idf}
    return out, arrays


# ---------------------------------------------------------------- T3b engine (PREREG_02 rev 8)
def t3b_reference(T, orient, Gm=None, vocab=None, min_black=T3B_MIN_BLACK, profile=T3_PROFILE):
    """R_b (fit players, their BLACK fit games), idf over R_b and the black-only null S2 (fit players with
    >= min_black black games, first `profile` black games in the T3 hash order, own column left out). Gm / vocab:
    the T3 token matrix of every training game (t3_token_matrix; shared with the registered reference)."""
    po = _po()
    if Gm is None:
        Gm, vocab = t3_token_matrix(T, orient)
    n, fit = T.n, T.fit
    fcol = {p: j for j, p in enumerate(fit)}
    blk = T.black_order()
    Rr = (po.indicator([blk[p] for p in fit], n) @ Gm).tocsr()
    df = np.diff(Rr.tocsc().indptr)
    idf = np.log((1.0 + len(fit)) / (1.0 + df)) + 1.0     # tokens in no black fit game: df = 0
    R = t3_weigh(Rr, idf)
    el = [p for p in fit if len(blk[p]) >= min_black]
    Cf = t3_weigh(po.indicator([blk[p][:profile] for p in el], n) @ Gm, idf)
    COSn = np.asarray((Cf @ R.T).toarray())
    own = np.array([fcol[p] for p in el], dtype=np.int64)
    COSn[np.arange(len(el)), own] = -np.inf
    s1n, s2n = s2_rows(COSn)
    return {"Gm": Gm, "vocab": vocab, "idf": idf, "R": R, "fit": fit, "fcol": fcol, "el": el, "Cf": Cf,
            "own": own, "COSn": COSn, "s1n": s1n, "s2n": s2n, "q99": q99(s2n), "V": len(vocab)}


def _sim_power(s2_pos, s2_null, thr, pis, sims, seed, p_ref=5767 / 8007):
    """P(declare) of one orientation for a 400-candidate pool with a share pi of training players (T3 layout)."""
    rng = np.random.default_rng(seed)
    power = {}
    for pi in pis:
        fr = []
        for _ in range(sims):
            k_ref = rng.binomial(rng.binomial(400, pi), p_ref)
            v = np.concatenate([rng.choice(s2_pos, k_ref), rng.choice(s2_null, 400 - k_ref)])
            fr.append((v > thr).mean())
        fr = np.array(fr)
        power[str(pi)] = {"P_declare": round(float((fr >= T3_SHARE).mean()), 4),
                          "mean_share_above": round(float(fr.mean()), 4),
                          "p05_share": round(float(np.quantile(fr, 0.05)), 4)}
    return power


def t3b_orient_precompute(T, orient, power_sims=5000, reg=None, min_black=T3B_MIN_BLACK, profile=T3_PROFILE):
    """T3b null, holdout (false positives), positives and power, per orientation; training players only.
    reg: the registered t3_reference of the same orientation (its token matrix is shared, and the registered T3
    is re-run on the same black-only profiles for the record)."""
    from scipy.stats import binom
    po = _po()
    to = time.time()
    if reg is None:
        reg = t3_reference(T, orient)
    ref = t3b_reference(T, orient, reg["Gm"], reg["vocab"], min_black, profile)
    Gm, idf, R = ref["Gm"], ref["idf"], ref["R"]
    n, fit, blk = T.n, ref["fit"], T.black_order()
    el, own, COSn, Cf, s1n, s2n, thr = ref["el"], ref["own"], ref["COSn"], ref["Cf"], ref["s1n"], ref["s2n"], ref["q99"]
    lv = np.array([T.level_of[p] for p in el])
    lev = [lv_ for lv_ in c10data.LEVELS if (lv == lv_).any()]
    t_null = time.time() - to
    # false positives: inner players (never in R) with >= min_black black games, the same profile rule
    ei = sorted(p for p in T.inner_pids if p in blk and len(blk[p]) >= min_black)
    if ei:
        Ci = t3_weigh(po.indicator([blk[p][:profile] for p in ei], n) @ Gm, idf)
        _, s2i = s2_rows(np.asarray((Ci @ R.T).toarray()))
    else:
        s2i = np.zeros(0)
    hold_rate = float((s2i > thr).mean()) if len(ei) else None
    out = {"n_null": len(el), "n_holdout_inner": len(ei), "vocab": ref["V"],
           "vocab_sha256": vocab_sha256(ref["vocab"]), "idf_sha256": sha256_array(idf.astype("<f8")),
           "idf_sha256_rule": "sha256 of the idf vector (df over the fit players' BLACK games) as little-endian "
                              "float64 bytes, vocabulary order = first appearance over all training games in "
                              "load_rows order (the T3 vocabulary)",
           "S2_q99": thr, "S1_q99": q99(s1n), "null_n_exceed": int((s2n > thr).sum()),
           "null_S2_quantiles": {q: float(np.quantile(s2n, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)},
           "null_S2_max": float(s2n.max()),
           "null_S2_exceed_by_level": {L: round(float((s2n[lv == L] > thr).mean()), 4) for L in lev},
           "null_worst_level_P_declare": max(float(binom.sf(t3_needed(400) - 1, 400,
                                                            float((s2n[lv == L] > thr).mean()))) for L in lev),
           "holdout_inner_exceed": round(hold_rate, 4) if hold_rate is not None else None,
           "holdout_inner_n_exceed": int((s2i > thr).sum()),
           "holdout_inner_P_declare_400_binomial": float(binom.sf(t3_needed(400) - 1, 400, hold_rate))
           if hold_rate is not None else None}
    # positives: the null players with their own column replaced by the profile of their OTHER black games
    pos = {}
    s2_pos_all, m_all = np.zeros(0), np.zeros(0, np.int64)
    for m in (20, 50, 100, None):
        sel = [j for j, p in enumerate(el) if len(blk[p]) - profile >= (1 if m is None else m)]
        key = "all" if m is None else str(m)
        if not sel:
            pos[key] = {"n": 0}
            continue
        grp = [blk[el[j]][profile:] if m is None else blk[el[j]][profile:profile + m] for j in sel]
        Rp = t3_weigh(po.indicator(grp, n) @ Gm, idf)
        oc = np.asarray(Cf[sel].multiply(Rp).sum(1)).ravel()
        C2 = COSn[sel].copy()
        C2[np.arange(len(sel)), own[sel]] = oc
        _, s2p = s2_rows(C2)
        pos[key] = {"n": len(sel), "tpr": round(float((s2p > thr).mean()), 4),
                    "tpr_by_level": {L: round(float((s2p[lv[sel] == L] > thr).mean()), 3)
                                     for L in lev if (lv[sel] == L).any()}}
        if m is None:
            s2_pos_all = s2p
            m_all = np.array([len(blk[el[j]]) - profile for j in sel])
    out["pos"] = pos
    bins = [(1, 20), (21, 50), (51, 100), (101, 200), (201, 10 ** 6)]
    out["pos_all_tpr_by_ref_games"] = {
        "%d-%d" % b: {"n": int(((m_all >= b[0]) & (m_all <= b[1])).sum()),
                      "tpr": round(float((s2_pos_all[(m_all >= b[0]) & (m_all <= b[1])] > thr).mean()), 4)
                      if ((m_all >= b[0]) & (m_all <= b[1])).any() else None} for b in bins}
    pis = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
    out["power_pool_rule"] = _sim_power(s2_pos_all, s2n, thr, pis, power_sims, 0) if len(s2_pos_all) else None
    # the registered T3 on the same black-only profiles (the reason for T3b; record only)
    rthr = reg["q99"]
    Cr = t3_weigh(po.indicator([blk[p][:profile] for p in el], n) @ Gm, reg["idf"])
    COSr = np.asarray((Cr @ reg["R"].T).toarray())
    COSr[np.arange(len(el)), own] = -np.inf
    _, s2r = s2_rows(COSr)
    prof = [set(blk[p][:profile]) for p in el]
    others = [[i for i in T.games_of[p] if i not in prof[j]] for j, p in enumerate(el)]
    Rpr = t3_weigh(po.indicator(others, n) @ Gm, reg["idf"])
    ocr = np.asarray(Cr.multiply(Rpr).sum(1)).ravel()
    COSr[np.arange(len(el)), own] = ocr
    _, s2rp = s2_rows(COSr)
    out["registered_T3_on_black_only_profiles"] = {
        "registered_q99": rthr, "null_players": len(el),
        "null_share_above": round(float((s2r > rthr).mean()), 4),
        "positive_tpr": round(float((s2rp > rthr).mean()), 4),
        "positive_definition": "own registered column replaced by the profile of all the player's other fit games "
                               "(both colours), as the registered R holds all fit games",
        "power_pool_rule": _sim_power(s2rp, s2r, rthr, (0.0, 0.2, 0.3, 0.5, 1.0), power_sims, 0),
        "note": "record only: the registered T3 (mixed-colour reference) applied to black-only 100-game profiles"}
    del COSr, Cr, Rpr
    out["t_null_s"] = round(t_null, 1)
    out["t_orient_s"] = round(time.time() - to, 1)
    arrays = {"player_id": np.array(el), "level": lv, "null_S2": s2n, "null_S1": s1n,
              "holdout_player_id": np.array(ei), "holdout_S2": s2i, "pos_all_S2": s2_pos_all,
              "pos_all_m": m_all, "idf": idf}
    return out, arrays


def t3b_false_positive(res, arrays):
    """False-positive table of T3b (training players only): per-candidate null / holdout exceedance per orientation
    and for the pool rule's raw OR canon, and P(declare) of a 400-candidate pool without training players."""
    from scipy.stats import binom
    a, b = arrays["raw"], arrays["canon"]
    tr, tc_ = res["raw"]["S2_q99"], res["canon"]["S2_q99"]
    need = t3_needed(400)
    out = {}
    for label, xa, xb in (("null_fit_leave_own_out", a["null_S2"], b["null_S2"]),
                          ("holdout_inner", a["holdout_S2"], b["holdout_S2"])):
        if not len(xa):
            out[label] = {"n": 0}
            continue
        ra, rb = float((xa > tr).mean()), float((xb > tc_).mean())
        either = float(((xa > tr) | (xb > tc_)).mean()) if len(xa) == len(xb) else None
        out[label] = {"n": int(len(xa)), "share_above_raw": round(ra, 4), "share_above_canon": round(rb, 4),
                      "share_above_raw_or_canon": round(either, 4) if either is not None else None,
                      "P_declare_400_binomial_raw": float(binom.sf(need - 1, 400, ra)),
                      "P_declare_400_binomial_canon": float(binom.sf(need - 1, 400, rb)),
                      "P_declare_400_binomial_raw_or_canon_bound": float(binom.sf(need - 1, 400, ra)) +
                      float(binom.sf(need - 1, 400, rb))}
    out["P_declare_simulated_pi_0"] = {o: (res[o]["power_pool_rule"] or {}).get("0.0") for o in ORIENTS}
    out["note"] = ("per candidate, about 1%% of null players exceed q99 by construction (method 'higher'); a pool "
                   "needs >= %d of 400 in one orientation" % need)
    return out


def t3_either(res, arrays, sims=5000):
    a, b = arrays["raw"], arrays["canon"]
    if not np.array_equal(a["player_id"], b["player_id"]):
        raise SystemExit("raw and canon null players differ")
    tr, tc = res["raw"]["S2_q99"], res["canon"]["S2_q99"]
    either = {"null_exceed_raw_or_canon_per_candidate": float(((a["null_S2"] > tr) | (b["null_S2"] > tc)).mean()),
              "null_exceed_raw_and_canon": float(((a["null_S2"] > tr) & (b["null_S2"] > tc)).mean()),
              "holdout_exceed_raw_or_canon": float(((a["holdout_S2"] > tr) | (b["holdout_S2"] > tc)).mean())
              if len(a["holdout_S2"]) else None,
              "note": "the pool rule is applied per orientation (>= 40 above that orientation's q99); "
                      "overlap if raw OR canon fires"}
    rng = np.random.default_rng(1)
    pr = {}
    na, nb, pa, pb = a["null_S2"], b["null_S2"], a["pos_all_S2"], b["pos_all_S2"]
    for pi in (0.0, 0.2, 0.3, 0.4, 0.5, 1.0):
        hits = 0
        for _ in range(sims):
            k_ref = rng.binomial(rng.binomial(400, pi), 5767 / 8007)
            ip = rng.integers(0, len(pa), k_ref)
            inl = rng.integers(0, len(na), 400 - k_ref)
            fr_a = ((pa[ip] > tr).sum() + (na[inl] > tr).sum()) / 400
            fr_b = ((pb[ip] > tc).sum() + (nb[inl] > tc).sum()) / 400
            hits += (fr_a >= 0.10) or (fr_b >= 0.10)
        pr[str(pi)] = hits / sims
    either["P_declare_raw_or_canon"] = pr
    return either


def check_pins(t3):
    """Differences between the computed T3 null and the registered pins ([] when all equal)."""
    bad = []
    for k, v in PINS["reference"].items():
        if t3["reference"].get(k) != v:
            bad.append("reference %s: got %r, pinned %r" % (k, t3["reference"].get(k), v))
    for o in ORIENTS:
        for k, v in PINS[o].items():
            got = t3["orient"][o].get(k)
            if got != v:
                bad.append("%s %s: got %r, pinned %r" % (o, k, got, v))
    return bad


T3_DEFINITIONS = {
    "reference_R": "all fit players (split v1, c10data.load_rows(part='fit') players), each profiled on all "
                   "their fit games",
    "tokens": "p02_player_open.game_tokens(colour, p02_player_open.moves_of(sgf), orient): colour-prefixed and "
              "colour-free sets together, own first 10 moves and whole-game first 10 moves; orient raw or "
              "canon (p02_moves.canon(mv, 10))",
    "vocabulary": "tokens of every training game (fit + inner) in load_rows order; tokens seen only in inner "
                  "games have df = 0",
    "weights": "profile counts summed over its games, tf = 1 + ln(count), idf = ln((1 + 5767) / (1 + df)) + 1 "
               "with df over R, rows L2-normalised (norm + 1e-12); a test token absent from the vocabulary "
               "enters only the row norm with idf = ln(5768) + 1",
    "statistic": "S2(c) = top1 - top2 of the cosine between the candidate profile (its 100 pool games) and "
                 "the 5,767 reference rows, per orientation",
    "null": "fit players with >= 100 games (1,985), profile = first 100 games in sha256('c10-t3|cand|{pid}|"
            "{gid}') order, own reference column left out; q99 = np.quantile(null, 0.99, method='higher')",
    "holdout": "inner players with >= 100 games (never in R), same profile rule",
    "positives": "the null players with their own column replaced by the profile of their other games "
                 "(m = 20 / 50 / 100 / all)",
    "t3q": "record-only: k-game profiles (k = 5..20) of fit players with >= 25 games, LOO null; test "
           "questions are compared with q99(k) of their own k",
}
T3_RULE = {"per_pool": "n_raw = #{S2_raw > q99_raw}, n_canon = #{S2_canon > q99_canon}; overlap iff n_raw >= "
                       "ceil(0.10 x candidates) OR n_canon >= ceil(0.10 x candidates) (40 of 400)",
           "test_day": "rebuild R; vocabulary size, vocab_sha256, idf_sha256, q99 and the null S2 vector must "
                       "equal the committed null; T1 FAIL_OPENING -> not evaluable; T2(a) FAIL proves the "
                       "overlap directly",
           "consequence": CONSEQUENCE[("T3", "OVERLAP")],
           "limit": "overlap of about 20% of the candidates or less is not detectable",
           "disclosure": "S2 was chosen after the inner holdout rates of 6 statistics had been seen",
           "black_only_pools": "a pool whose candidate black share b_c is exactly 1 uses T3b (PREREG_02 rev 8; "
                               "results/test02/null/t3b_null.json); the registered T3 numbers of that pool are "
                               "recorded only",
           "single_colour_low_power": "a pool on the registered T3 with max(b_c, 1 - b_c) >= %.1f (b_c not exactly "
                                      "1) is flagged low_power: without a declared overlap its conclusion is 'not "
                                      "determinable (low power)' and T3's verdict LOW_POWER instead of PASS "
                                      "(PREREG_02 rev 8); a declared overlap stays OVERLAP" % T3_LOW_POWER_SHARE}


def check_pins_t3b(t3b):
    """Differences between the computed T3b null and T3B_PINS ([] when all equal; an unset pin is a mismatch)."""
    bad = []
    for k, v in T3B_PINS["reference"].items():
        got = t3b["reference"].get(k)
        if v is None or got != v:
            bad.append("t3b reference %s: got %r, pinned %r" % (k, got, v))
    for o in ORIENTS:
        for k, v in T3B_PINS[o].items():
            got = t3b["orient"][o].get(k)
            if v is None or got != v:
                bad.append("t3b %s %s: got %r, pinned %r" % (o, k, got, v))
    return bad


T3B_DEFINITIONS = {
    "when": "competition 1, per candidate pool, when the pool's candidate black share b_c (T4) is exactly 1 (the "
            "official design: candidates black only); otherwise the registered T3 (flagged low_power when "
            "max(b_c, 1 - b_c) >= 0.9)",
    "reference_R_b": "all fit players (the T3 reference players), each profiled on their BLACK fit games only",
    "tokens": T3_DEFINITIONS["tokens"],
    "vocabulary": T3_DEFINITIONS["vocabulary"] + " (the same vocabulary as T3)",
    "weights": "as T3, with df over R_b: idf = ln((1 + 5767) / (1 + df_black)) + 1, df_black = the number of fit "
               "players (R_b rows) with the token in at least one of their BLACK fit games (players, as T3's df; not "
               "games); a token in no black fit game has df_black = 0",
    "statistic": "S2(c) = top1 - top2 of the cosine between the candidate profile (its 100 pool games) and the "
                 "5,767 R_b rows, per orientation",
    "null": "fit players with >= 100 black fit games, profile = their first 100 BLACK games in sha256('c10-t3|cand|"
            "{pid}|{gid}') order, own R_b column left out; q99 = np.quantile(null, 0.99, method='higher')",
    "holdout": "inner players with >= 100 black games (never in R_b), same profile rule (false positives)",
    "positives": "the null players with their own R_b column replaced by the profile of their other black games "
                 "(m = 20 / 50 / 100 / all)",
    "power": "a 400-candidate pool with a share pi of training players, each in R_b with probability 5767/8007 "
             "(as T3); 5,000 draws per pi",
}
T3B_RULE = {"per_pool": T3_RULE["per_pool"] + " (q99 and S2 of T3b)",
            "test_day": "rebuild R_b; vocabulary size, vocab_sha256, idf_sha256, q99, the null player ids and the null "
                        "S2 vector must equal this committed file; T1 FAIL_OPENING -> not evaluable",
            "consequence": CONSEQUENCE[("T3", "OVERLAP")],
            "pools_left_on_T3": T3_RULE["single_colour_low_power"],
            "registration": "PREREG_02 rev 8 (written before any test data)"}


# ---------------------------------------------------------------- precompute: T2 null, T4 / T6 simulations
def t2_null(T):
    """Port of the design-record scratch t2_null.py (training players only)."""
    n, inner = T.n, T.inner_game
    out = {}
    for name, kt in (("k_full", "full"), ("k100", "k100"), ("k60", "k60")):
        K = T.keys[kt]
        fit_idx, by_key, cnt = defaultdict(set), defaultdict(set), Counter()
        for i in range(n):
            k = K[i]
            if k is not None and not inner[i]:
                fit_idx[k].add(T.pid_idx[i])
                by_key[k].add(T.pid_idx[i])
                cnt[k] += 1
        hit = [i for i in range(n) if inner[i] and K[i] is not None and K[i] in fit_idx]
        n_inner = int(inner.sum())
        fit_cross = sum(1 for i in range(n) if not inner[i] and K[i] is not None and len(by_key[K[i]]) > 1)
        fit_same = sum(1 for i in range(n) if not inner[i] and K[i] is not None
                       and cnt[K[i]] > 1 and len(by_key[K[i]]) == 1)
        out[name] = {"inner_games": n_inner, "inner_games_with_copy_in_fit": len(hit),
                     "rate": len(hit) / n_inner if n_inner else None,
                     "inner_players_involved": len({int(T.pid_idx[i]) for i in hit}),
                     "fit_games_sharing_key_with_other_fit_player": fit_cross,
                     "fit_games_sharing_key_only_with_same_player": fit_same}
    return out


def first_move_refs(T):
    """Training first-move references (train and fit) and the S99 / S999 class sets."""
    n, inner = T.n, T.inner_game
    all_cls = Counter(FIRST_CLASS[T.first_pt(i)] for i in range(n))
    tot = sum(all_cls.values())
    cum, S999, S99_ = 0, set(), set()
    for c, v in all_cls.most_common():
        if cum / tot < 0.999:
            S999.add(c)
        if cum / tot < 0.99:
            S99_.add(c)
        cum += v

    def pop(mask):
        idx = np.nonzero(mask)[0]
        fm = Counter(T.first_pt(i) for i in idx)
        t = sum(fm.values())
        ur = sum(v for p, v in fm.items() if p[0] > "j" and p[1] < "j") / t
        col = Counter(T.colour[i] for i in idx)
        nm = T.nmov[idx]
        return {"games": int(len(idx)), "black_share": col["B"] / sum(col.values()), "upper_right_share": ur,
                "move_count": {"min": int(nm.min()), "p1": float(np.percentile(nm, 1)),
                               "median": float(np.median(nm)), "p99": float(np.percentile(nm, 99)),
                               "max": int(nm.max())},
                "first_move_point_dist": {p: v / t for p, v in fm.most_common()},
                "first_move_class_dist": {c: v / t for c, v in
                                          Counter(FIRST_CLASS[p] for p in fm.elements()).most_common()}}
    return {"S99": sorted(S99_), "S999": sorted(S999), "first_move_classes_total": len(all_cls),
            "train": pop(np.ones(n, bool)), "fit": pop(~inner)}


def t4_sim(T):
    """Port of the design-record scratch t4.py (training players only)."""
    n = T.n
    players = T.games_of
    pids = sorted(players)
    elig = [p for p in pids if len(players[p]) >= 100]
    elig5 = [p for p in pids if len(players[p]) >= 5]
    S99_ = set(S99)
    all_cls = Counter(FIRST_CLASS[T.first_pt(i)] for i in range(n))
    tot = sum(all_cls.values())
    cum, S999 = 0, set()
    for c, v in all_cls.most_common():
        if cum / tot < 0.999:
            S999.add(c)
        cum += v
    rng = np.random.default_rng(0)

    def pool(kind):
        if kind == "cand":
            pl = list(rng.choice(elig, 400, replace=False))
            games = []
            for p in pl:
                games += list(rng.choice(players[p], 100, replace=False))
        else:
            pl = list(rng.choice(elig5, 400, replace=False))
            games = []
            for p in pl:
                k = min(int(rng.integers(5, 21)), len(players[p]))
                games += list(rng.choice(players[p], k, replace=False))
        return set(pl), games

    def raw_dist(idx):
        c = Counter(T.first_pt(i) for i in idx)
        t = sum(c.values())
        return {k: v / t for k, v in c.items()}

    res = {}
    for kind in ("cand", "ques"):
        tv, share99, share999, bshare, rt = [], [], [], [], []
        for rep in range(100):
            pl, games = pool(kind)
            ref = [i for p in pids if p not in pl for i in players[p]] if rep < 20 else None
            if ref is not None:
                fr = raw_dist(ref)
                ft = raw_dist(games)
                tv.append(0.5 * sum(abs(ft.get(k, 0) - fr.get(k, 0)) for k in set(ft) | set(fr)))
            share99.append(np.mean([FIRST_CLASS[T.first_pt(i)] in S99_ for i in games]))
            share999.append(np.mean([FIRST_CLASS[T.first_pt(i)] in S999 for i in games]))
            bshare.append(np.mean([T.colour[i] == "B" for i in games]))
            rt.append(np.mean([T.first_pt(i)[0] > "j" and T.first_pt(i)[1] < "j" for i in games]))
        res[kind] = {"first_move_tvd_vs_rest": {"mean": float(np.mean(tv)), "max": float(np.max(tv)), "n": len(tv)},
                     "share_in_S99": {"min": float(np.min(share99)), "mean": float(np.mean(share99))},
                     "share_in_S999": {"min": float(np.min(share999)), "mean": float(np.mean(share999))},
                     "black_share": {"min": float(np.min(bshare)), "max": float(np.max(bshare)),
                                     "mean": float(np.mean(bshare))},
                     "upper_right_share": {"min": float(np.min(rt)), "max": float(np.max(rt)),
                                           "mean": float(np.mean(rt))}}
    alt = {}
    idx = list(rng.choice(n, 20000, replace=False))
    for k in (2, 3, 5, 11, 21, 31):
        alt[k] = {"share_in_S99": float(np.mean([FIRST_CLASS[T.first_pt(i, k)] in S99_ for i in idx])),
                  "share_in_S999": float(np.mean([FIRST_CLASS[T.first_pt(i, k)] in S999 for i in idx]))}
    res["not_from_move1_share"] = alt
    rot = [SYM[int(rng.integers(0, 8))][T.first_pt(i)] for i in idx]
    fr = raw_dist(range(n))
    c = Counter(rot)
    ft = {k: v / len(rot) for k, v in c.items()}
    res["randomly_rotated"] = {"tvd_vs_train": 0.5 * sum(abs(ft.get(k, 0) - fr.get(k, 0)) for k in set(ft) | set(fr)),
                               "upper_right_share": float(np.mean([p[0] > "j" and p[1] < "j" for p in rot]))}
    return res


def t6_sim(T, seeds=(0, 1, 2)):
    """Port of the design-record scratch t6.py: fit pools, opening-only raw scorer."""
    po = _po()
    by_player, level = {}, {}
    for p, g in T.games_of.items():
        if p in T.inner_pids:
            continue
        by_player[p] = [(T.gid[i], T.colour[i], None) for i in g]
        level[p] = T.level_of[p]
    elig = defaultdict(list)
    for p, g in by_player.items():
        if len(g) >= 105:
            elig[level[p]].append(p)
    out = []
    for seed in seeds:
        pool, cand, query = po.sim_pool(by_player, elig, seed, "random")
        pids = sorted(pool)
        rng = random.Random("t6|%d" % seed)
        owners = [rng.choice(pids) for _ in range(400)]
        q_wr = []
        for p in owners:
            cs = set(cand[p])
            rest = [g for g, _, _ in by_player[p] if g not in cs]
            q_wr.append((p, rng.sample(rest, min(rng.randint(5, 20), len(rest)))))
        keys = sorted({(p, g) for p in pids for g in cand[p]} | {(p, g) for p in pids for g in query[p]} |
                      {(p, g) for p, q in q_wr for g in q})
        row = {k: i for i, k in enumerate(keys)}
        gidx = {}
        for p in {k[0] for k in keys}:
            gidx.update({(p, T.gid[i]): i for i in T.games_of[p]})
        (Gm,) = po.token_matrices([[(T.colour[gidx[k]], T.mv60(gidx[k])) for k in keys]], "raw")
        cg = [[row[(p, g)] for g in cand[p]] for p in pids]
        cos1, _ = po.open_cos(Gm, cg, Gm, [[row[(p, g)] for g in query[p]] for p in pids])
        cos2, _ = po.open_cos(Gm, cg, Gm, [[row[(p, g)] for g in q] for p, q in q_wr])
        d1 = len(set(np.argmax(cos1, 1)))
        d2 = len(set(np.argmax(cos2, 1)))
        acc1 = float(np.mean(np.argmax(cos1, 1) == np.arange(400)))
        sr = []
        for items in po.sr_rounds({p: cand[p] for p in pids}):
            c, _ = po.open_cos(Gm, [[row[(p, g)] for g in refs] for p, _q, refs in items], Gm,
                               [[row[(p, g)] for g in q] for p, q, _r in items])
            sr.append(len(set(np.argmax(c, 1))))
        out.append({"seed": seed, "distinct_top1_one_to_one": d1, "top1_acc_one_to_one": acc1,
                    "distinct_owners_with_replacement": len(set(owners)), "distinct_top1_with_replacement": d2,
                    "self_retrieval_distinct_top1_by_round": sr})
        log("t6 sim seed %d: %s" % (seed, out[-1]))
    return out


def _range(vals):
    return [min(vals), max(vals)]


def _train_refs(T, refs_fm, t2n, t4, t6):
    """train_refs.json content (precompute --part all / base)."""
    stats = dict(T.stats)
    sims_ok = bool(stats.get("alternating_from_B") == T.n)
    refs = {"format": "c10-test02 train refs v1",
            "what": "training references for T1.5, T2, T4, T5 and T6 (PREREG_02 rev 5 9.6-9.9); training players "
                    "only (split-v1 part 'train' = fit + inner via c10data.load_rows)",
            "population": {"train": {"players": len(T.games_of), "games": T.n},
                           "fit": {"players": len(T.fit), "games": T.reference_record()["games"]}},
            "S99": list(S99), "S99_rule": "D4 classes of the first move (lexicographic minimum of the 8 images) "
                                          "covering 99% of the training games, taken in frequency order",
            "S999": refs_fm["S999"], "first_move_classes_total": refs_fm["first_move_classes_total"],
            "black_share": {"fit": refs_fm["fit"]["black_share"], "train": refs_fm["train"]["black_share"]},
            "upper_right_share": {"definition": "first move x > 'j' and y < 'j' (SGF letters)",
                                  "fit": refs_fm["fit"]["upper_right_share"],
                                  "train": refs_fm["train"]["upper_right_share"]},
            "move_count": {"fit": refs_fm["fit"]["move_count"], "train": refs_fm["train"]["move_count"]},
            "first_move_point_dist_train": refs_fm["train"]["first_move_point_dist"],
            "first_move_class_dist_train": refs_fm["train"]["first_move_class_dist"],
            "sgf_training": {"games": T.n, "alternating_from_B": stats.get("alternating_from_B", 0),
                             "all_alternating_from_B": sims_ok,
                             "t1_4_violations": stats.get("t1_4_violations", 0), "passes": stats.get("passes", 0),
                             "setup_properties": stats.get("setup", 0), "sz_not_19": stats.get("sz_not_19", 0)},
            "t2_null": dict(t2n, definitions={"key": KEY_VERSION, "index": "fit + inner (part 'train')",
                                              "copy": "K100 match; games under 100 moves: full key vs the "
                                                      "training K_n of the same length"}),
            "t4_sim": dict(t4, pool_shapes="candidate pool = 400 players x 100 games (players with >= 100 "
                                           "games); question file = 400 players x k ~ U{5..20} games; the "
                                           "TVD reference is the training players not in the pool"),
            "t4_null_ranges": {"black_share": {"cand": [t4["cand"]["black_share"]["min"], t4["cand"]["black_share"]["max"]],
                                               "ques": [t4["ques"]["black_share"]["min"], t4["ques"]["black_share"]["max"]]},
                               "upper_right_share": {"cand": [t4["cand"]["upper_right_share"]["min"],
                                                              t4["cand"]["upper_right_share"]["max"]],
                                                     "ques": [t4["ques"]["upper_right_share"]["min"],
                                                              t4["ques"]["upper_right_share"]["max"]]},
                               "first_move_point_tvd_max": {"cand": t4["cand"]["first_move_tvd_vs_rest"]["max"],
                                                            "ques": t4["ques"]["first_move_tvd_vs_rest"]["max"]},
                               "share_in_S99_min": {"cand": t4["cand"]["share_in_S99"]["min"],
                                                    "ques": t4["ques"]["share_in_S99"]["min"]}},
            "t5": {"uniform_5_20": {"mean": 12.5, "share_below_10": 5 / 16},
                   "chi2_df15_critical": {"0.05": 24.99579013972863, "0.01": 30.577914166892494},
                   "local_rank_validation_reference": {"mean": 12.06, "share_below_10": 0.364,
                                                       "source": "results/test02/design/checks.md (T5); copied, "
                                                                 "not recomputed here"}},
            "t6_sim": {"pools": t6,
                       "ranges": {"one_to_one_distinct_top1": _range([x["distinct_top1_one_to_one"] for x in t6]),
                                  "self_retrieval_rounds_distinct_top1":
                                      _range([v for x in t6 for v in x["self_retrieval_distinct_top1_by_round"]]),
                                  "with_replacement_distinct_top1": _range([x["distinct_top1_with_replacement"] for x in t6])},
                       "scorer": "opening-only raw (p02_player_open.open_cos), fit pools of p02_player_open.sim_pool"}}
    return refs


def cmd_precompute(args):
    t0 = time.time()
    mode = mode_of(args)
    out = os.path.abspath(args.out or NULL_DIR)
    guard_out(out, mode)
    if mode == "real" and not same_path(out, NULL_DIR):
        raise SystemExit(f"real mode writes the null only to {rel(NULL_DIR)}")
    if mode in ("smoke", "rehearsal"):
        raise SystemExit("precompute reads training players only; use it in real mode or with --dry-run")
    if args.allow_pin_mismatch and mode == "real":
        raise SystemExit("--allow-pin-mismatch is for --dry-run only")
    part = args.part
    do_base, do_b = part in ("all", "base"), part in ("all", "t3b")
    if args.t2_index_cache and not do_base:
        raise SystemExit("--t2-index-cache needs the T2 keys: use --part all or base")
    targets = ([os.path.join(out, f) for f in NULL_FILES] if do_base else []) + \
              ([os.path.join(out, f) for f in T3B_NULL_FILES] if do_b else [])
    reason = check_rerun(args, targets)
    T = load_train(keys=do_base, grammar=do_base)
    if do_base:
        refs_fm = first_move_refs(T)
        if tuple(refs_fm["S99"]) != S99:
            raise SystemExit(f"training S99 {refs_fm['S99']} differs from the registered {list(S99)}")
        t = time.time()
        t2n = t2_null(T)
        log("T2 null %.0fs: %s" % (time.time() - t, json.dumps(t2n)))
        t = time.time()
        t4 = t4_sim(T)
        log("T4 sims %.0fs" % (time.time() - t))
        t = time.time()
        t6 = t6_sim(T)
        log("T6 sims %.0fs" % (time.time() - t))
        if args.t2_index_cache:
            cp = os.path.abspath(args.t2_index_cache)
            guard_out(os.path.dirname(cp), mode)
            save_index_cache(T, cp)
            log("T2 index cached to %s" % cp)
    T.keys = None
    res, arrays, resb, arrb, regchk = {}, {}, {}, {}, {}
    for orient in ORIENTS:
        ref = t3_reference(T, orient)
        regchk[orient] = {"S2_q99": [ref["q99"], PINS[orient]["S2_q99"]], "vocab": [ref["V"], PINS[orient]["vocab"]],
                          "n_null": [len(ref["el"]), PINS[orient]["n_null"]]}
        if do_base:
            res[orient], arrays[orient] = t3_orient_precompute(T, orient, ref=ref)
            log(orient, json.dumps({k: res[orient][k] for k in ("S2_q99", "vocab", "n_null", "n_holdout_inner",
                                                                 "null_n_exceed", "holdout_inner_n_exceed",
                                                                 "idf_sha256", "t_orient_s")}))
        if do_b:
            resb[orient], arrb[orient] = t3b_orient_precompute(T, orient, reg=ref)
            log("t3b", orient, json.dumps({k: resb[orient][k] for k in (
                "S2_q99", "vocab", "n_null", "n_holdout_inner", "null_n_exceed", "holdout_inner_n_exceed",
                "idf_sha256", "t_orient_s")}))
        del ref
    bad = []
    if do_base:
        t3 = {"format": "c10-test02 t3 null v1",
              "what": "T3 null (PREREG_02 rev 5 9.8) from training players only: split-v1 part 'train' read through "
                      "c10data.load_rows; no validation game, no label file",
              "definitions": T3_DEFINITIONS, "reference": T.reference_record(), "orient": res,
              "either": t3_either(res, arrays), "rule": T3_RULE}
        bad = check_pins(t3)
        t3["pins"] = {"expected": PINS, "all_match": not bad, "mismatches": bad}
    if do_b:
        reg_bad = ["registered T3 %s %s: got %r, pinned %r" % (o, k, v[0], v[1])
                   for o in ORIENTS for k, v in regchk[o].items() if v[0] != v[1]]
        t3b = {"format": "c10-test02 t3b null v1",
               "what": "T3b null (PREREG_02 rev 8) for black-only candidate pools, from training players only: "
                       "split-v1 part 'train' read through c10data.load_rows; no validation game, no label file",
               "definitions": T3B_DEFINITIONS, "reference": T.reference_record_black(), "orient": resb,
               "either": t3_either(resb, arrb), "false_positive": t3b_false_positive(resb, arrb),
               "registered_reference_check": {"orient": regchk, "all_match": not reg_bad,
                                              "what": "the registered T3 reference rebuilt in the same run equals "
                                                      "its pins (the T3b and T3 references share the token matrix)"},
               "audit_scratch_q99": {"values": AUDIT_T3B_Q99, "source": "results/test02/design/official_structure/"
                                                                        "sim_t3b.json (scratch audit, 2026-09-28)",
                                     "reproduced": {o: resb[o]["S2_q99"] == AUDIT_T3B_Q99[o] for o in ORIENTS}},
               "rule": T3B_RULE}
        bad_b = check_pins_t3b(t3b) + reg_bad
        t3b["pins"] = {"expected": T3B_PINS, "all_match": not bad_b, "mismatches": bad_b}
        bad = bad + bad_b
    refs = _train_refs(T, refs_fm, t2n, t4, t6) if do_base else None
    prov = provenance(mode, {"split_csv": rel(SPLIT_CSV) + " " + sha256_text_file(SPLIT_CSV)}, reason)
    prov["runtime_s"] = {"load": getattr(T, "load_s", None), "parse_and_hash": getattr(T, "parse_s", None),
                         "total": round(time.time() - t0, 1)}
    try:
        import psutil
        prov["peak_wset_mb"] = round(getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2 ** 20)
    except ImportError:
        pass
    prov["part"] = part
    if bad and not args.allow_pin_mismatch:
        for b in bad:
            log("PIN MISMATCH:", b)
        raise SystemExit("the T3 / T3b null does not reproduce the registered pins; nothing written (fix the port or "
                         "write a revision before 11/04)")
    written = []
    if do_base:
        t3["provenance"] = prov
        refs["provenance"] = prov
        save_npz(os.path.join(out, "t3_null_raw.npz"), **arrays["raw"])
        save_npz(os.path.join(out, "t3_null_canon.npz"), **arrays["canon"])
        t3["npz_sha256"] = {f: sha256_file(os.path.join(out, f)) for f in ("t3_null_raw.npz", "t3_null_canon.npz")}
        write_json(os.path.join(out, "t3_null.json"), t3)
        write_json(os.path.join(out, "train_refs.json"), refs)
        written += list(NULL_FILES)
    if do_b:
        t3b["provenance"] = prov
        save_npz(os.path.join(out, "t3b_null_raw.npz"), **arrb["raw"])
        save_npz(os.path.join(out, "t3b_null_canon.npz"), **arrb["canon"])
        t3b["npz_sha256"] = {f: sha256_file(os.path.join(out, f)) for f in T3B_NULL_FILES[1:]}
        write_json(os.path.join(out, "t3b_null.json"), t3b)
        written += list(T3B_NULL_FILES)
    log("pins %s; wrote %s in %s (%.0fs)" % ("OK" if not bad else "MISMATCH (dry run kept)", written, out,
                                           time.time() - t0))
    return 1 if bad else 0


# ---------------------------------------------------------------- T2 index (test day)
class KeyIndex:
    def __init__(self, keys, games):
        """keys: list of bytes-or-None per game, games: the game indices they belong to."""
        sel = [(k, g) for k, g in zip(keys, games) if k is not None]
        ks = np.array([k for k, _ in sel], dtype="S%d" % KEY_BYTES) if sel else np.zeros(0, "S%d" % KEY_BYTES)
        gs = np.array([g for _, g in sel], dtype=np.int64)
        o = np.argsort(ks, kind="stable")
        self.k, self.g = ks[o], gs[o]

    def lookup(self, key):
        if key is None or not len(self.k):
            return self.g[:0]
        q = np.array([key], dtype=self.k.dtype)
        lo = int(np.searchsorted(self.k, q[0], "left"))
        hi = int(np.searchsorted(self.k, q[0], "right"))
        return self.g[lo:hi]


class TrainIndex:
    """Training key index for T2(a): full, K100, K60 and K_n (n < 100) keys -> training player index."""

    def __init__(self, keys, extra, pid_idx, pids, part_mask, part, black=None):
        """black: per training game, True when the training row's colour is B (T2(a) sides, PREREG_02 rev 8);
        None leaves the sides unavailable."""
        games = np.nonzero(part_mask)[0]
        self.part = part
        self.pid_idx, self.pids = pid_idx, pids
        self.black = None if black is None else np.asarray(black, dtype=bool)
        self.n_games = int(len(games))
        self.n_players = int(len(set(pid_idx[games].tolist())))
        self.idx = {kt: KeyIndex([keys[kt][i] for i in games], games) for kt in ("full", "k100", "k60")}
        self.extra = {m: KeyIndex([v[i] for i in games], games) for m, v in extra.items()}

    def players_of(self, games):
        return set(self.pid_idx[games].tolist())


INDEX_FORMAT = "c10-t02 index cache v2: keys, player index, inner flag, training row colour (black)"


def index_meta(part):
    return {"format": INDEX_FORMAT, "key_version": KEY_VERSION, "part": part,
            "split_csv_sha256": sha256_text_file(SPLIT_CSV),
            "c10data_sha256": sha256_text_file(os.path.join(HERE, "c10data.py")),
            "p02_moves_sha256": sha256_text_file(os.path.join(HERE, "p02_moves.py"))}


def save_index_cache(T, path):
    meta = dict(index_meta("train"), games=T.n, players=len(T.pids))
    arr = {"meta": np.array(json.dumps(meta, sort_keys=True)), "pid_idx": T.pid_idx,
           "pids": np.array(T.pids), "inner_game": T.inner_game, "black": train_black(T)}
    for kt in ("full", "k100", "k60"):
        K = T.keys[kt]
        arr["has_" + kt] = np.array([k is not None for k in K])
        arr["key_" + kt] = np.array([k or b"" for k in K], dtype="S%d" % KEY_BYTES)
    save_npz(path, **arr)
    return meta


def train_black(T):
    return np.fromiter((c == "B" for c in T.colour), dtype=bool, count=T.n)


def load_index_cache(path, expected_games=None):
    """(keys dict, pid_idx, pids, inner_game, black) or None when the cache is missing or stale."""
    if not path or not os.path.isfile(path):
        return None
    z = load_npz(path)
    meta = json.loads(str(z["meta"]))
    want = index_meta("train")
    if expected_games is None:
        expected_games = c10data.EXPECTED["train"]["games"]
    if any(meta.get(k) != v for k, v in want.items()) or meta.get("games") != expected_games or "black" not in z:
        log("index cache %s is stale; rebuilding" % path)
        return None
    keys = {}
    for kt in ("full", "k100", "k60"):
        raw = z["key_" + kt]
        has = z["has_" + kt]
        # numpy strips trailing NUL bytes of S12 items; pad back to 12 bytes
        keys[kt] = [bytes(k).ljust(KEY_BYTES, b"\0") if h else None for k, h in zip(raw, has)]
    return keys, z["pid_idx"], [str(p) for p in z["pids"]], z["inner_game"], z["black"]


# ---------------------------------------------------------------- test files for run / t6
class TestFile:
    def __init__(self, tests_dir, pool, kind, name):
        self.pool, self.kind, self.name = pool, kind, name
        self.path = os.path.join(tests_dir, name)
        if not os.path.isfile(self.path):
            raise SystemExit(f"{self.path} missing")
        self.table = Table(self.path)
        if self.table.error:
            raise SystemExit(f"{name}: {self.table.error}")
        self.sha256 = self.table.sha256
        self.games = []
        for owner, label, colour, sgf in file_games(self.table, kind):
            full = MOVE_RE.findall(sgf)
            self.games.append({"owner": owner, "label": label, "colour": colour, "sgf": sgf,
                               "pts": [p for _, p in full], "first": full[0][1] if full else None})
        self.owners = sorted({g["owner"] for g in self.games}) if kind == "candidates" else \
            list(dict.fromkeys(g["owner"] for g in self.games))
        if kind == "questions":
            ng = self.table.column("num_games") or []
            self.num_games = [int(x) if re.fullmatch(r"[0-9]+", x.strip()) else None for x in ng]
            self.question_ids = self.table.column("question_id") or []

    def n_games(self):
        return len(self.games)


def load_test_files(comp, tests_dir, allow_missing=False):
    """The comp's test files; allow_missing (not real mode) skips absent files, e.g. the one-pool NEG
    control tree of t02_fake_tests.py from-train."""
    return [TestFile(tests_dir, pool, kind, name) for pool, kind, name in FILES[comp]
            if not (allow_missing and not os.path.isfile(os.path.join(tests_dir, name)))]


def pools_with(files, *kinds):
    have = {(f.pool, f.kind) for f in files}
    return [p for p in POOLS if all((p, k) in have for k in kinds)]


# ---------------------------------------------------------------- T2 (test day)
def _keys_of(g):
    if "keys" not in g:
        g["keys"] = game_keys(g["pts"])
    return g["keys"]


def t2_run(comp, files, index):
    """T2 units (a)-(d) for one competition. files: TestFile list; index: TrainIndex."""
    units = []

    def unit(uid, label, games_in_file, copies, extra):
        thr = t2_threshold(games_in_file)
        u = {"unit": uid, "what": label, "games_in_file": games_in_file, "threshold": thr, "copies": copies,
             "verdict": "FAIL" if copies >= thr else "PASS"}
        u.update(extra)
        units.append(u)

    # (a) each test file vs the training index
    for f in files:
        copies, k60, short, nomove = 0, 0, 0, 0
        sides = Counter()          # record only (PREREG_02 rev 8): training row colour vs the test colour
        owner_players = defaultdict(set)
        owner_copies = Counter()
        for g in f.games:
            kf, k1, k6 = _keys_of(g)
            n = len(g["pts"])
            if n == 0:
                nomove += 1
                continue
            if n >= K_COPY:
                hit = index.idx["k100"].lookup(k1)
            else:
                short += 1
                if n not in index.extra:
                    raise SystemExit(f"index has no K_{n} keys for a {n}-move game")
                hit = index.extra[n].lookup(kf)
            if len(hit):
                copies += 1
                owner_copies[g["owner"]] += 1
                owner_players[g["owner"]] |= index.players_of(hit)
                sides[_copy_side(index, hit, g["colour"])] += 1
            if n >= K_RECORD and len(index.idx["k60"].lookup(k6)):
                k60 += 1
        who = "candidates" if f.kind == "candidates" else "questions"
        unit("a", f"{f.name} vs training players ({index.part})", f.n_games(), copies,
             {"file": f.name, "k60_matches_record_only": k60, "games_under_100_moves": short,
              "games_without_moves": nomove,
              "same_side_copies_record_only": sides["same"] + sides["both"],
              "opposite_side_copies_record_only": sides["opposite"],
              "copies_matching_both_sides_record_only": sides["both"],
              "copies_side_unknown_record_only": sides["unknown"],
              "%s_with_copy" % who: len(owner_copies),
              "%s_whose_copies_map_to_one_training_player" % who:
                  sum(1 for o in owner_copies if len(owner_players[o]) == 1),
              "copies_per_%s_histogram" % who[:-1]: dict(Counter(owner_copies.values()))})
    by = {(f.pool, f.kind): f for f in files}
    # (b) questions vs their own pool's candidates (player), full key
    if comp == "player":
        for pool in pools_with(files, "questions", "candidates"):
            q, c = by[(pool, "questions")], by[(pool, "candidates")]
            cfull = {_keys_of(g)[0] for g in c.games} - {None}
            c100 = {_keys_of(g)[1] for g in c.games} - {None}
            copies = sum(1 for g in q.games if _keys_of(g)[0] is not None and _keys_of(g)[0] in cfull)
            k100 = sum(1 for g in q.games if _keys_of(g)[1] is not None and _keys_of(g)[1] in c100
                       and _keys_of(g)[0] not in cfull)
            cand_dup = Counter(_keys_of(g)[0] for g in c.games if _keys_of(g)[0] is not None)
            cown = defaultdict(set)
            for g in c.games:
                if _keys_of(g)[0] is not None:
                    cown[_keys_of(g)[0]].add(g["owner"])
            unit("b", f"{q.name} vs {c.name} (full game)", q.n_games(), copies,
                 {"pool": pool, "k100_only_matches_record_only": k100,
                  "candidate_games_shared_by_two_candidates_record_only":
                      sum(v for k, v in cand_dup.items() if len(cown[k]) > 1),
                  "candidate_games_repeated_within_one_candidate_record_only":
                      sum(v for k, v in cand_dup.items() if v > 1 and len(cown[k]) == 1)})
    # (c) across questions of one file, full key
    for f in files:
        if f.kind != "questions":
            continue
        occ = defaultdict(list)
        for g in f.games:
            if _keys_of(g)[0] is not None:
                occ[_keys_of(g)[0]].append((g["owner"], g["colour"]))
        dup_games, same, opp, within = 0, 0, 0, 0
        for k, lst in occ.items():
            if len(lst) < 2:
                continue
            owners = {o for o, _ in lst}
            if len(owners) < len(lst):
                within += len(lst) - len(owners)
            for o, col in lst:
                others = [(o2, c2) for o2, c2 in lst if o2 != o]
                if not others:
                    continue
                dup_games += 1
                same += any(c2 == col for _, c2 in others)
                opp += any(c2 != col for _, c2 in others)
        unit("c", f"{f.name}: games shared by two or more questions (full game)", f.n_games(), dup_games,
             {"file": f.name, "games_shared_with_same_colour": same,
              "games_shared_with_opposite_colour": opp,
              "extra_copies_within_one_question_record_only": within})
    # (d) public vs private (per competition), full key
    pub = [g for f in files if f.pool == "public" for g in f.games]
    priv = [g for f in files if f.pool == "private" for g in f.games]
    kpub = {_keys_of(g)[0] for g in pub} - {None}
    kpriv = {_keys_of(g)[0] for g in priv} - {None}
    copies = sum(1 for g in priv if _keys_of(g)[0] is not None and _keys_of(g)[0] in kpub)
    if pub and priv:
        unit("d", "private games found among the public games (%s; full game)"
             % ("questions" if comp == "rank" else "questions + candidates"), len(priv), copies,
             {"public_games_found_among_private": sum(1 for g in pub if _keys_of(g)[0] is not None
                                                      and _keys_of(g)[0] in kpriv),
              "note": "threshold on the private side's game count"})
    verdict = "FAIL" if any(u["verdict"] == "FAIL" for u in units) else "PASS"
    return {"verdict": verdict, "consequence": CONSEQUENCE[("T2", verdict)], "units": units,
            "index": {"part": index.part, "games": index.n_games, "players": index.n_players,
                      "short_lengths": sorted(index.extra)},
            "definitions": {"key": KEY_VERSION + "; moves = all ;B[xy]/;W[xy] matches, colour letters left out",
                            "copy": "K100 keys equal; a test game under 100 moves: its full key equals the "
                                    "training key of the same length (K_n)",
                            "threshold": "FAIL iff copies >= max(10, ceil(0.001 x games in the unit's test file))",
                            "sides": "unit (a), record only (PREREG_02 rev 8): a copy is same_side when a matching "
                                     "training row has the test game's colour (the same player's side of that game "
                                     "can be in training), opposite_side when every matching row has the other colour "
                                     "(training holds the game from the opponent's side: not a leak of the same "
                                     "player); same_side + opposite_side + unknown = copies; the FAIL threshold stays "
                                     "on all copies",
                            "separation": "rank files are never compared with player files"}}


def _copy_side(index, hit, colour):
    """'same', 'both', 'opposite' or 'unknown' for one copied test game (T2(a) sides; record only)."""
    if index.black is None or colour not in ("B", "W"):
        return "unknown"
    hb = index.black[hit]
    tb = colour == "B"
    same, opp = bool((hb == tb).any()), bool((hb != tb).any())
    return "both" if same and opp else "same" if same else "opposite"


# ---------------------------------------------------------------- T3 (test day)
def _ref_check(ref, no, z):
    """Rebuilt reference vs the committed null of one orientation (T3 or T3b)."""
    chk = {"vocab": [ref["V"], no["vocab"]],
           "vocab_sha256": [vocab_sha256(ref["vocab"]), no["vocab_sha256"]],
           "idf_sha256": [sha256_array(ref["idf"].astype("<f8")), no["idf_sha256"]],
           "q99": [ref["q99"], no["S2_q99"]],
           "null_player_ids_equal": bool(np.array_equal(np.array(ref["el"]), z["player_id"])),
           "null_S2_equal": bool(len(ref["s2n"]) == len(z["null_S2"]) and np.array_equal(ref["s2n"], z["null_S2"])),
           "idf_equal_npz": bool(len(ref["idf"]) == len(z["idf"]) and np.array_equal(ref["idf"], z["idf"]))}
    pairs_equal = all(v[0] == v[1] for v in chk.values() if isinstance(v, list))
    chk["passed"] = bool(pairs_equal and chk["null_player_ids_equal"] and chk["null_S2_equal"] and chk["idf_equal_npz"])
    return chk


def _pool_profiles(c):
    """(candidate ids, [(colour, moves)], groups, black share b_c) of one candidate file."""
    pids = sorted({g["owner"] for g in c.games})
    pos = {p: [] for p in pids}
    games = []
    for g in c.games:
        pos[g["owner"]].append(len(games))
        games.append((g["colour"], _mv60(g)))
    b_c = sum(1 for g in c.games if g["colour"] == "B") / len(c.games) if c.games else None
    return pids, games, [pos[p] for p in pids], b_c


def t3_run(comp, files, T, null, npz, t1_verdict, t2_result, log_=log, null_b=None, npz_b=None):
    """T3 per candidate pool (player) and T3q (record only). A pool whose candidate black share b_c is exactly 1
    uses T3b (PREREG_02 rev 8; null_b / npz_b = the committed t3b null); its registered T3 numbers are recorded only.
    A pool left on T3 with max(b_c, 1 - b_c) >= 0.9 is low_power: without a declared overlap the verdict is LOW_POWER
    ('not determinable'), not PASS."""
    rec = {"reference_check": {}, "pools": {}, "t3q": {}}
    ok_all = True
    if null.get("reference") is not None:
        got = T.reference_record()
        same = all(got.get(k) == v for k, v in null["reference"].items())
        rec["reference_check"]["fit_reference"] = {"rebuilt": got, "committed": null["reference"], "passed": same}
        ok_all &= same
    q_files = [f for f in files if f.kind == "questions"]
    prof = {}
    if comp == "player":
        for pool in pools_with(files, "candidates"):
            prof[pool] = _pool_profiles(next(f for f in files if f.pool == pool and f.kind == "candidates"))
    black_pools = [pool for pool, v in prof.items() if v[3] == 1.0]
    use_b = bool(black_pools)
    if use_b:
        if null_b is None or npz_b is None:
            rec["reference_check"]["t3b"] = {
                "passed": False, "problem": "pools %s are black-only but the T3b null (results/test02/null/"
                                            "t3b_null.json, PREREG_02 rev 8) is missing" % black_pools}
            ok_all = False
            use_b = False
        elif null_b.get("reference") is not None:
            got = T.reference_record_black()
            same = all(got.get(k) == v for k, v in null_b["reference"].items())
            rec["reference_check"]["t3b_fit_reference"] = {"rebuilt": got, "committed": null_b["reference"],
                                                            "passed": same}
            ok_all &= same
    for orient in ORIENTS:
        t = time.time()
        ref = t3_reference(T, orient)
        no = null["orient"][orient]
        chk = _ref_check(ref, no, npz[orient])
        rec["reference_check"][orient] = chk
        ok = chk["passed"]
        refb = None
        if use_b:
            refb = t3b_reference(T, orient, ref["Gm"], ref["vocab"])
            chkb = _ref_check(refb, null_b["orient"][orient], npz_b[orient])
            rec["reference_check"]["t3b_" + orient] = chkb
            ok = ok and chkb["passed"]
            for k in ("Gm", "Cf", "COSn"):
                refb.pop(k, None)
        ok_all &= ok
        for k in ("Gm", "Cf", "COSn"):
            ref.pop(k, None)
        q99_o = no["S2_q99"]
        for pool, (pids, games, groups, b_c) in prof.items():
            d = rec["pools"].setdefault(pool, {"n_candidates": len(pids), "needed": t3_needed(len(pids)),
                                               "b_c": b_c, "variant": "T3b" if pool in black_pools else "T3"})
            _top1, s2, n_ext = t3_profile_s2(ref, games, groups, orient)
            n_above = int((s2 > q99_o).sum())
            reg = {"q99": q99_o, "n_above": n_above, "share_above": n_above / len(pids) if pids else None,
                   "S2_quantiles": quantiles(s2), "tokens_unseen_in_training": n_ext}
            if pool not in black_pools:
                d[orient] = reg
                continue
            d.setdefault("registered_T3_record_only", {})[orient] = reg
            if refb is not None:
                qb = null_b["orient"][orient]["S2_q99"]
                _top1, s2b, n_ext_b = t3_profile_s2(refb, games, groups, orient)
                nb = int((s2b > qb).sum())
                d[orient] = {"q99": qb, "n_above": nb, "share_above": nb / len(pids) if pids else None,
                             "S2_quantiles": quantiles(s2b), "tokens_unseen_in_training": n_ext_b}
        by_k = {int(k): v["q99"] for k, v in no["t3q"]["by_k"].items()}
        for f in q_files:
            owners = f.owners
            pos = {o: [] for o in owners}
            games = []
            for g in f.games:
                pos[g["owner"]].append(len(games))
                games.append((g["colour"], _mv60(g)))
            groups = [pos[o] for o in owners]
            _top1, s2, _ = t3_profile_s2(ref, games, groups, orient)
            ks = np.array([len(gr) for gr in groups])
            thr = np.array([by_k.get(int(k), np.inf) for k in ks])
            above = s2 > thr
            rec["t3q"].setdefault(f.name, {})[orient] = {
                "questions": len(owners), "n_above_q99_of_k": int(above.sum()),
                "share_above": float(above.mean()) if len(above) else None, "expected_null_share": 0.01,
                "questions_with_k_outside_5_20": int(np.isinf(thr).sum()), "S2_quantiles": quantiles(s2)}
        log_("T3 %s: reference %s%s (%.0fs)" % (orient, "OK" if ok else "MISMATCH",
                                               " (with T3b)" if refb is not None else "", time.time() - t))
        del ref, refb
    t2a = bool(t2_result and any(u["unit"] == "a" and u["verdict"] == "FAIL" for u in t2_result.get("units", [])))
    low = []
    if comp == "player":
        for pool, d in rec["pools"].items():
            d["overlap"] = (t3_pool_overlap(d["raw"]["n_above"], d["canon"]["n_above"], d["n_candidates"])
                            if "raw" in d and "canon" in d else None)
            d["low_power"] = t3_low_power(d["b_c"], d["variant"])
            d["conclusion"] = ("not computed" if d["overlap"] is None else "overlap" if d["overlap"] else
                               "not determinable (low power)" if d["low_power"] else "no overlap declared")
            if d["low_power"]:
                low.append(pool)
        declared = any(d["overlap"] for d in rec["pools"].values())
        if not ok_all:
            verdict = "ERROR"
        elif t1_verdict == "FAIL_OPENING":
            verdict = "NOT_EVALUABLE"
        elif declared:
            verdict = "OVERLAP"
        else:
            verdict = "LOW_POWER" if low else "PASS"
    else:
        verdict = "RECORD_ONLY" if ok_all else "ERROR"
    rec.update({"verdict": verdict, "consequence": CONSEQUENCE[("T3", verdict)], "t1_verdict": t1_verdict,
                "t2a_fired": t2a,
                "t2a_note": "T2(a) FAIL proves the overlap directly" if t2a else None,
                "rule": T3_RULE["per_pool"], "black_only_rule": T3_RULE["black_only_pools"],
                "low_power_rule": T3_RULE["single_colour_low_power"], "low_power_pools": low,
                "variants": {pool: d.get("variant") for pool, d in rec["pools"].items()},
                "reference": T.reference_record()})
    return rec, ok_all


def _mv60(g):
    if "mv60" not in g:
        g["mv60"] = _po().moves_of(g["sgf"])
    return g["mv60"]


# ---------------------------------------------------------------- T4 / T5 (test day)
def _tvd(counter, ref):
    tot = sum(counter.values())
    if not tot:
        return None
    return 0.5 * sum(abs(counter.get(k, 0) / tot - ref.get(k, 0.0)) for k in set(counter) | set(ref))


def t4_run(comp, files, refs):
    rng_ = refs["t4_null_ranges"]
    per_file = {}
    for f in files:
        kind = "cand" if f.kind == "candidates" else "ques"
        cols = [g["colour"] for g in f.games]
        b = sum(c == "B" for c in cols) / len(cols) if cols else None
        firsts = [g["first"] for g in f.games if g["first"]]
        cp = Counter(firsts)
        cc = Counter(FIRST_CLASS[p] for p in firsts)
        ur = sum(1 for p in firsts if p[0] > "j" and p[1] < "j") / len(firsts) if firsts else None
        tv = _tvd(cp, refs["first_move_point_dist_train"])
        per_file[f.name] = {
            "pool": f.pool, "kind": f.kind, "games": f.n_games(), "games_without_moves": f.n_games() - len(firsts),
            "black_share": b, "black_share_null_range": rng_["black_share"][kind],
            "black_share_inside_null_range": None if b is None else
            bool(rng_["black_share"][kind][0] <= b <= rng_["black_share"][kind][1]),
            "first_move_point_tvd_vs_train": tv,
            "first_move_point_tvd_null_max": rng_["first_move_point_tvd_max"][kind],
            "first_move_class_tvd_vs_train_record_only": _tvd(cc, refs["first_move_class_dist_train"]),
            "upper_right_share": ur, "upper_right_null_range": rng_["upper_right_share"][kind],
            "rotation_flag": None if ur is None else bool(ur < UPPER_RIGHT_MIN),
            "null_ranges_applicable": True}
        if b == 1.0:
            # PREREG_02 rev 8: the null ranges come from mixed-colour training pools; a black-only file (the official
            # competition-1 design) falls outside them by construction, which says nothing about the file
            d = per_file[f.name]
            d["mixed_colour_null_ranges_record_only"] = {
                k: d[k] for k in ("black_share_null_range", "black_share_inside_null_range",
                                  "first_move_point_tvd_null_max", "upper_right_null_range")}
            d.update(null_ranges_applicable=False, black_share_null_range=NOT_APPLICABLE,
                     black_share_inside_null_range=NOT_APPLICABLE, first_move_point_tvd_null_max=NOT_APPLICABLE,
                     upper_right_null_range=NOT_APPLICABLE,
                     null_ranges_note="not applicable: black-only file (official competition-1 design); the null "
                                      "ranges were simulated on mixed-colour pools (PREREG_02 rev 8)")
    pools = {}
    verdict = "RECORD_ONLY"
    if comp == "player":
        verdict = "PASS"
        for pool in pools_with(files, "questions", "candidates"):
            bc = per_file[FILES_BY[(comp, pool, "candidates")]]["black_share"]
            bq = per_file[FILES_BY[(comp, pool, "questions")]]["black_share"]
            diff = abs(bc - bq) if bc is not None and bq is not None else None
            lvl3 = diff is not None and diff > COLOUR_MAX_DIFF
            pools[pool] = {"b_c": bc, "b_q": bq, "abs_diff": diff, "guard_level3_colour": lvl3,
                           "black_only_pool": bool(bc == 1.0 and bq == 1.0)}
            if lvl3:
                verdict = "GUARD_LEVEL3"
    return {"verdict": verdict, "consequence": CONSEQUENCE[("T4", verdict)], "files": per_file, "pools": pools,
            "training": {"black_share": refs["black_share"], "upper_right_share": refs["upper_right_share"]},
            "rules": {"decision": "|b_c - b_q| > 0.5 -> player guard level 3 for that pool (enforced in "
                                  "p02_player_fuse.guard_decision)",
                      "record_only": "first-move point TVD vs training, class TVD, upper-right share (rotation "
                                     "flag < %.2f)" % UPPER_RIGHT_MIN,
                      "black_only_files": "a file whose black share is 1 has its mixed-colour null ranges marked "
                                          "'%s' (PREREG_02 rev 8); the values are kept under "
                                          "mixed_colour_null_ranges_record_only" % NOT_APPLICABLE}}


FILES_BY = {(c, pool, kind): name for c, lst in FILES.items() for pool, kind, name in lst}
NOT_APPLICABLE = "not applicable"
OFFICIAL_DESIGN = {5: 100, 10: 100, 15: 100, 20: 100}      # AIdea data page (DATA.md 2026-09-28)


def t5_run(files, refs):
    from scipy.stats import chi2
    out = {}
    for f in files:
        if f.kind != "questions":
            continue
        ng = [x for x in f.num_games if x is not None]
        hist = Counter(ng)
        n = len(ng)
        obs = np.array([hist.get(k, 0) for k in range(K_MIN, K_MAX + 1)], dtype=np.float64)
        exp = n / (K_MAX - K_MIN + 1)
        stat = float(((obs - exp) ** 2 / exp).sum()) if n else None
        out[f.name] = {"questions": n, "histogram": {str(k): hist.get(k, 0) for k in range(K_MIN, K_MAX + 1)},
                       "outside_5_20": sum(v for k, v in hist.items() if not K_MIN <= k <= K_MAX),
                       "mean": float(np.mean(ng)) if n else None,
                       "share_below_10": sum(1 for x in ng if x < 10) / n if n else None,
                       "chi2_vs_uniform": stat, "df": K_MAX - K_MIN,
                       "p_value": float(chi2.sf(stat, K_MAX - K_MIN)) if n else None,
                       "official_design": {str(k): v for k, v in OFFICIAL_DESIGN.items()},
                       "official_design_match": bool(dict(hist) == OFFICIAL_DESIGN and n == len(f.num_games)),
                       "chi2_note": "the official files use a fixed design (5/10/15/20 games x 100 questions), so the "
                                    "chi2 against uniform 5..20 is not a test of anything there (PREREG_02 rev 8; "
                                    "record only)"}
    return {"verdict": "RECORD_ONLY", "consequence": CONSEQUENCE[("T5", "RECORD_ONLY")], "files": out,
            "references": dict(refs["t5"], official_design={str(k): v for k, v in OFFICIAL_DESIGN.items()})}


# ---------------------------------------------------------------- run
def _read_t1(out_comp, comp, mode):
    p = os.path.join(out_comp, "t1.json")
    if not os.path.isfile(p):
        raise SystemExit(f"{p} missing: run t1 first")
    t1 = read_json(p)
    if t1.get("comp") != comp:
        raise SystemExit(f"{p} is for {t1.get('comp')!r}")
    return t1, p


def cmd_run(args):
    t0 = time.time()
    mode = mode_of(args)
    comp = args.comp
    if mode in ("smoke", "rehearsal"):
        raise SystemExit(f"--{mode} trees are built from validation files: content checks (T2-T5) never run "
                         f"on them (spec 0.11)")
    tests, out = resolve_paths(args, mode, REAL_RUN)
    null_dir = resolve_null(args, mode)
    only = [x.strip().upper() for x in args.only.split(",")] if args.only else ["T2", "T3", "T4", "T5"]
    if not set(only) <= {"T2", "T3", "T4", "T5"}:
        raise SystemExit(f"--only takes T2,T3,T4,T5 (got {args.only})")
    if args.index_part == "fit" and mode == "real":
        raise SystemExit("--index-part fit is only for the dry-run negative control")
    out_comp = os.path.join(out, comp)
    targets = [os.path.join(out_comp, "%s.json" % c.lower()) for c in only]
    reason = check_rerun(args, targets)
    t1, t1_path = _read_t1(out_comp, comp, mode)
    inputs = {rel(t1_path): sha256_file(t1_path)}
    if mode == "real":
        probs = not_committed_clean([os.path.abspath(__file__)] + [os.path.join(NULL_DIR, f)
                                                                   for f in NULL_FILES + T3B_NULL_FILES])
        if probs:
            raise SystemExit("real mode needs tools/t02_checks.py and the null (with the T3b null of PREREG_02 rev 8) "
                             "committed and clean: " + "; ".join(probs))
        if not t1.get("complete"):
            raise SystemExit("t1.json did not run T1.4 and T1.5; rerun t1 without --only")
        if t1.get("stage") != "raw":
            raise SystemExit(f"{rel(t1_path)} has stage {t1.get('stage')!r}; run needs the T1 of the raw files")
        if t1["verdict"] == "FAIL_STRUCT":
            raise SystemExit("T1 is FAIL_STRUCT: stop; write a revision before any prediction")
        inputs["manifest"] = read_manifest(tests, comp, t1)
    elif t1["verdict"] == "FAIL_STRUCT":
        log("warning: T1 is FAIL_STRUCT (dry run continues)")
    files = load_test_files(comp, tests, allow_missing=mode != "real")
    missing = [n for _p, _k, n in FILES[comp] if n not in {f.name for f in files}]
    if missing:
        log("warning: missing test files skipped (dry run): %s" % missing)
    for f in files:
        inputs[f.name] = f.sha256
    if mode != "real":
        guard = refuse_validation_games(g["sgf"] for f in files for g in f.games)
    else:
        guard = real_validation_guard({f.name: (g["sgf"] for g in f.games) for f in files})
    need_train = "T2" in only or "T3" in only
    null = npz = refs = null_b = npz_b = None
    if "T3" in only:
        null = read_json(os.path.join(null_dir, "t3_null.json"))
        npz = {o: load_npz(os.path.join(null_dir, "t3_null_%s.npz" % o)) for o in ORIENTS}
        for f in NULL_FILES[:3]:
            inputs[rel(os.path.join(null_dir, f))] = sha256_file(os.path.join(null_dir, f))
        if all(os.path.isfile(os.path.join(null_dir, f)) for f in T3B_NULL_FILES):
            null_b = read_json(os.path.join(null_dir, "t3b_null.json"))
            npz_b = {o: load_npz(os.path.join(null_dir, "t3b_null_%s.npz" % o)) for o in ORIENTS}
            for f in T3B_NULL_FILES:
                inputs[rel(os.path.join(null_dir, f))] = sha256_file(os.path.join(null_dir, f))
        elif mode == "real":
            raise SystemExit("the T3b null (results/test02/null/t3b_null*.{json,npz}, PREREG_02 rev 8) is missing")
    if "T4" in only or "T5" in only:
        refs = read_json(os.path.join(null_dir, "train_refs.json"))
        inputs[rel(os.path.join(null_dir, "train_refs.json"))] = sha256_file(os.path.join(null_dir, "train_refs.json"))
    results = {}
    if need_train:
        short = sorted({len(g["pts"]) for f in files for g in f.games if 0 < len(g["pts"]) < K_COPY})
        cache = None
        cache_path = os.path.abspath(args.index_cache) if args.index_cache else None
        if cache_path:
            guard_out(os.path.dirname(cache_path), mode)
        if "T2" in only and cache_path and "T3" not in only and not short:
            cache = load_index_cache(cache_path)
        if cache is not None:
            keys, pid_idx, pids, inner_game, black = cache
            T = None
            log("T2 index loaded from %s" % cache_path)
        else:
            T = load_train(keys="T2" in only, extra_lengths=short)
            if "T2" in only:
                keys, pid_idx, pids, inner_game, black = T.keys, T.pid_idx, T.pids, T.inner_game, train_black(T)
                if cache_path and not short:
                    save_index_cache(T, cache_path)
                    log("T2 index cached to %s" % cache_path)
        if "T2" in only:
            mask = np.ones(len(pid_idx), bool) if args.index_part == "train" else ~np.asarray(inner_game)
            extra = T.extra if T is not None else {}
            index = TrainIndex(keys, extra, np.asarray(pid_idx), pids, mask, args.index_part, black=black)
            t = time.time()
            results["T2"] = t2_run(comp, files, index)
            results["T2"]["seconds"] = round(time.time() - t, 1)
            log("T2 %s" % results["T2"]["verdict"])
            if T is not None:
                T.keys = None
            del index
        if "T3" in only:
            rec, ok = t3_run(comp, files, T, null, npz, t1["verdict"], results.get("T2") or _maybe(out_comp, "t2.json"),
                             null_b=null_b, npz_b=npz_b)
            results["T3"] = rec
    if "T4" in only:
        results["T4"] = t4_run(comp, files, refs)
    if "T5" in only:
        results["T5"] = t5_run(files, refs)
    prov = provenance(mode, inputs, reason)
    prov["validation_hash_guard"] = guard
    prov["missing_files_skipped"] = missing
    prov["runtime_s"] = round(time.time() - t0, 1)
    entries = {}
    for check in only:
        body = dict(results[check], format="c10-test02 %s v1" % check.lower(), comp=comp,
                    t1_verdict=t1["verdict"], provenance=prov)
        fn = "%s.json" % check.lower()
        write_json(os.path.join(out_comp, fn), body)
        entries[check] = (results[check]["verdict"], fn)
    s = update_summary(out_comp, comp, entries)
    for check in only:
        log("%s %s: %s" % (comp, check, s["checks"][check]["verdict"]))
    log("wrote %s (%.0fs)" % (out_comp, time.time() - t0))
    return 1 if results.get("T3", {}).get("verdict") == "ERROR" else 0


def _maybe(out_comp, name):
    p = os.path.join(out_comp, name)
    return read_json(p) if os.path.isfile(p) else None


# ---------------------------------------------------------------- t1 command
def cmd_t1(args):
    mode = mode_of(args)
    comp = args.comp
    tests, out = resolve_paths(args, mode, REAL_T1)
    only = [x.strip().upper() for x in args.only.split(",")] if args.only else list(T1_SUBS)
    if not set(only) <= set(T1_SUBS):
        raise SystemExit(f"--only takes {','.join(T1_SUBS)} (got {args.only})")
    if mode in ("smoke", "rehearsal") and set(only) - set(T1_STRUCT):
        raise SystemExit(f"--{mode} trees come from validation files: only --only T1.1,T1.2,T1.3 (spec 0.11)")
    if mode == "real" and same_path(tests, DATA_TEST_RAW) and set(only) != set(T1_SUBS):
        raise SystemExit(f"real mode on {rel(DATA_TEST_RAW)} runs the full T1 (T1.1-T1.5): its t1.json gates the build "
                         f"and the uploads, so --only is refused here (review finding 2026-09-26). --only is for the "
                         f"built copies in data/test (t02_test_bundle.py build) and for the other modes")
    name = "t1.json"
    out_comp = os.path.join(out, comp)
    target = os.path.join(out_comp, name)
    reason = check_rerun(args, [target])
    if mode == "real":
        probs = not_committed_clean([os.path.abspath(__file__)])
        if probs:
            raise SystemExit("real mode needs tools/t02_checks.py committed and clean: " + "; ".join(probs))
    guard = None
    if set(only) - set(T1_STRUCT):
        named = {}
        for _pool, kind, fn in FILES[comp]:
            p = os.path.join(tests, fn)
            if os.path.isfile(p):
                t = Table(p)
                named[fn] = [g[3] for g in file_games(t, kind)]
        if mode != "real":
            guard = refuse_validation_games(s for v in named.values() for s in v)
        else:
            guard = real_validation_guard(named)
    res = t1_check(comp, tests, only)
    inputs = {k: v.get("sha256") for k, v in res["files"].items()}
    src = os.path.join(tests, "SOURCE.json")
    if os.path.isfile(src):
        inputs["SOURCE.json"] = sha256_file(src)
    res["provenance"] = provenance(mode, inputs, reason)
    res["provenance"]["validation_hash_guard"] = guard
    res["stage"] = "built" if os.path.basename(os.path.normpath(out)).lower() == "built" else "raw"
    write_json(target, res)
    update_summary(out_comp, comp, {"T1": (res["verdict"], name)})
    log("T1 %s (%s): %s%s" % (comp, rel(tests), res["verdict"],
                              " normalize %s" % res["normalize"] if res["normalize"] else ""))
    for sub in T1_SUBS:
        for p in res["subchecks"][sub]["problems"][:5]:
            log("  %s %s" % (sub, p))
    log("wrote", target)
    v = res["verdict"]
    if v == "FAIL_STRUCT":
        return 3
    if v == "FAIL_OPENING":
        return 2
    if v == "NORMALIZED" and args.require_pass:
        return 4
    return 0


# ---------------------------------------------------------------- t6
def cmd_t6(args):
    mode = mode_of(args)
    comp = args.comp
    if comp != "player":
        raise SystemExit("T6 is for the player competition only")
    tests, out = resolve_paths(args, mode, REAL_RUN)
    null_dir = resolve_null(args, mode)
    if mode == "real":
        sub = os.path.abspath(args.submission or os.path.join(PLAYER_DIR, "submission_player.csv"))
        pdir = os.path.abspath(args.player_dir or PLAYER_DIR)
        if not same_path(sub, os.path.join(PLAYER_DIR, "submission_player.csv")) or not same_path(pdir, PLAYER_DIR):
            raise SystemExit(f"real mode reads {rel(PLAYER_DIR)} only")
        probs = not_committed_clean([os.path.abspath(__file__)])
        if probs:
            raise SystemExit("real mode needs tools/t02_checks.py committed and clean: " + "; ".join(probs))
    else:
        base = args.root or (os.path.dirname(os.path.dirname(os.path.dirname(out))))
        pdir = os.path.abspath(args.player_dir or os.path.join(base, "results", "test02", "player"))
        sub = os.path.abspath(args.submission or os.path.join(pdir, "submission_player.csv"))
    out_comp = os.path.join(out, comp)
    target = os.path.join(out_comp, "t6.json")
    reason = check_rerun(args, [target])
    res = t6_describe(tests, sub, pdir, out_comp, os.path.join(null_dir, "train_refs.json"))
    inputs = dict(res.pop("_inputs"))
    res["provenance"] = provenance(mode, inputs, reason)
    write_json(target, res)
    update_summary(out_comp, comp, {"T6": (res["verdict"], "t6.json")})
    for pool, d in res["pools"].items():
        log("T6 %s: questions %s candidates %s distinct top-1 %s sr rounds %s" % (
            pool, d["questions"], d["candidates"], d["distinct_top1_submission"], d["sr_top1_distinct_per_round"]))
    log("wrote", target)
    return 0


def t6_describe(tests, sub_path, player_dir, out_comp, refs_path):
    refuse_forbidden(sub_path)
    inputs = {}
    t = Table(sub_path)
    if t.error or t.header[:2] != ["question_id", "top_1"]:
        raise SystemExit(f"{sub_path}: not a player submission (header {t.header})")
    inputs[rel(sub_path)] = t.sha256
    top1 = {r[0]: r[1] for r in t.rows}
    t2 = _maybe(out_comp, "t2.json")
    refs = read_json(refs_path) if os.path.isfile(refs_path) else None
    pools = {}
    for pool in POOLS:
        qt = Table(os.path.join(tests, FILES_BY[("player", pool, "questions")]))
        ct = Table(os.path.join(tests, FILES_BY[("player", pool, "candidates")]))
        inputs[qt.name], inputs[ct.name] = qt.sha256, ct.sha256
        qids = qt.column("question_id") or []
        cands = set(ct.column("player_id") or [])
        picks = [top1.get(q) for q in qids]
        missing = sum(1 for p in picks if p is None)
        ap = os.path.join(player_dir, pool, "apply_report.json")
        sr = None
        if os.path.isfile(ap):
            inputs[rel(ap)] = sha256_file(ap)
            sr = read_json(ap).get("sr_top1_distinct_per_round")
        t2c = None
        if t2:
            for u in t2.get("units", []):
                if u["unit"] == "c" and u.get("file") == qt.name:
                    t2c = {"same_colour_games": u["games_shared_with_same_colour"],
                           "opposite_colour_games": u["games_shared_with_opposite_colour"]}
        pools[pool] = {"questions": len(qids), "candidates": len(cands),
                       "questions_equal_candidates_equal_400": len(qids) == len(cands) == N_QUESTIONS,
                       "distinct_top1_submission": len({p for p in picks if p is not None}),
                       "questions_missing_in_submission": missing,
                       "top1_outside_pool": sum(1 for p in picks if p is not None and p not in cands),
                       "sr_top1_distinct_per_round": sr, "t2c": t2c}
    ref = refs["t6_sim"]["ranges"] if refs else None
    return {"format": "c10-test02 t6 v1", "comp": "player", "verdict": "RECORD_ONLY",
            "consequence": CONSEQUENCE[("T6", "RECORD_ONLY")], "pools": pools, "training_simulation_ranges": ref,
            "registered": "without an explicit official statement, the one-to-one structure is not used",
            "_inputs": inputs}


# ---------------------------------------------------------------- CLI
def _mode_args(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="any tree; output outside the repo")
    g.add_argument("--smoke", action="store_true", help="notebook SMOKE tree (structure checks only)")
    g.add_argument("--rehearsal", action="store_true", help="rehearsal tree (structure checks only)")
    p.add_argument("--root", help="(not real) tree root: tests S/data/test, out S/results/test02/checks")
    p.add_argument("--rerun", action="store_true", help="replace existing outputs (needs --reason)")
    p.add_argument("--reason", help="why the outputs are replaced (stored in the report)")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precompute", help="T3 null and training references (training players only)")
    p.add_argument("--out", help="default results/test02/null")
    p.add_argument("--t2-index-cache", nargs="?", const=INDEX_CACHE, default=None,
                   help="also write the T2 training index cache (default path data/cache/t2_train_index.npz)")
    p.add_argument("--allow-pin-mismatch", action="store_true", help="(--dry-run) write even if the pins differ")
    p.add_argument("--part", choices=("all", "base", "t3b"), default="all",
                   help="all (default): the four T3 / reference files and the three T3b files; base: only the four "
                        "files of PREREG_02 rev 5; t3b: only t3b_null.json and t3b_null_{raw,canon}.npz (rev 8), "
                        "leaving the base files untouched")
    _mode_args(p)
    p = sub.add_parser("t1", help="T1 schema check (gates the pipeline)")
    p.add_argument("--comp", required=True, choices=sorted(FILES))
    p.add_argument("--tests-dir")
    p.add_argument("--out")
    p.add_argument("--only", help="subset of T1.1..T1.5 (T1.1-T1.3 always run)")
    p.add_argument("--require-pass", action="store_true", help="exit 4 on NORMALIZED (for the built copies)")
    _mode_args(p)
    p = sub.add_parser("run", help="T2-T5 (record, never gate the first upload)")
    p.add_argument("--comp", required=True, choices=sorted(FILES))
    p.add_argument("--tests-dir")
    p.add_argument("--out")
    p.add_argument("--null-dir", help="default results/test02/null")
    p.add_argument("--only", help="subset of T2,T3,T4,T5")
    p.add_argument("--index-part", choices=("train", "fit"), default="train")
    p.add_argument("--index-cache", nargs="?", const=INDEX_CACHE, default=None,
                   help="read / write the T2 training index cache (default path data/cache/t2_train_index.npz)")
    _mode_args(p)
    p = sub.add_parser("t6", help="T6 descriptive check after submit (player)")
    p.add_argument("--comp", required=True, choices=["player"])
    p.add_argument("--tests-dir")
    p.add_argument("--out")
    p.add_argument("--null-dir")
    p.add_argument("--submission")
    p.add_argument("--player-dir", help="t02_player.py out dir with <pool>/apply_report.json")
    _mode_args(p)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"precompute": cmd_precompute, "t1": cmd_t1, "run": cmd_run, "t6": cmd_t6}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
