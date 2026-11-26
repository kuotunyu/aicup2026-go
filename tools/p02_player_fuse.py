"""Competition 1 (player identification), PREREG_02 section 2.2: CNN terms (E0, E1), fusion with
the opening term (cosO), the Q00 reproduction check, self-retrieval selection, and apply with the
guard. Stdlib + numpy + scipy only; the opening part lives in tools/p02_player_open.py.

    python tools/p02_player_fuse.py repro  [--p01-dir results/prereg01/player]
                                           [--p02-dir results/prereg02/player]
                                           [--candidates data/local/player_val_candidates.csv]
                                           [--questions data/local/player_val_questions.csv]
                                           [--out-dir results/prereg02/player] [--smoke]
    python tools/p02_player_fuse.py select [--p02-dir ...] [--candidates ...] [--out-dir ...]
                                           [--guard OUT_DIR/guard_calibration.json]
                                           [--no-cnn --reason TEXT] [--smoke]
    python tools/p02_player_fuse.py apply  [--p02-dir ...] [--candidates ...] [--questions ...]
                                           [--out-dir ...] [--smoke]

Inputs: the candidate file and the question file (no labels in either), and the notebook dumps
P02_DIR/dumps/cand_games.npz (game_id, player_id, color, emb_sum [N,256], n_windows) and
P02_DIR/dumps/query_games.npz (question_id, slot, color, emb_sum, n_windows, parsed). repro also
reads P01_DIR/player_val_top5.csv and P01_DIR/player_val_embeddings.npz. Nothing else; no answer
file is ever opened (any path whose name contains that word is refused).

Terms, per question q and candidate c (candidates always in sorted id order):
    E0 = -||centroid_c - centroid_q||_2, centroid = sum of emb_sum / sum of n_windows over the games
         (window-weighted, exactly the official / prereg01 way)
    E1 = cosine of: per game emb_sum / n_windows, L2-normalised; mean over the games; L2-normalised
    O  = cosO of tools/p02_player_open.py, orientation raw or canon
    a game counts for E only when its n_windows > 0 (and parsed, for questions); a question with
    no such game has no E term; a candidate with none gets the worst E score of the question.
Fusion: s = z(E) + w * z(O), z over the candidates of the question (population std; std < 1e-12
or a missing term -> that term is 0). Single-term combinations use s = z(term). Ranking by s
descending, ties by candidate id; a question where every term is missing therefore gets the first
5 candidates in sorted id order (the official fallback).

Combinations (fixed order = tie priority): Q00 E0; Q01 E1; Q02-Q06 E0 + O_raw, Q07-Q11 E1 + O_raw,
Q12-Q16 E0 + O_canon, Q17-Q21 E1 + O_canon with w = 0.25, 0.5, 1, 2, 4; Q22 O_raw; Q23 O_canon.

repro  (plan.md 4.3): Q00 on the validation questions vs prereg01: identical top-5 on
       >= n_questions - 2 questions (398/400; --smoke: all) and max |centroid difference| <= 1e-4
       (plus: same question and candidate ids, and both dumps made with the same weights when they
       record weights_sha256). Without --smoke the notebook run must also have ended cleanly: its
       repro_check.json passed and the last run in run_log.json has status 'completed' (dumps,
       reproduction and prereg01 read-only fingerprint all fine); with --smoke this is recorded only.
       Writes OUT_DIR/repro_local.json (with its creation time: the failure date for the 7-day rule
       of PREREG_02 2.2); exit code 1 when it fails.
       Real mode needs the dumps to cover exactly the candidate / question files; --smoke uses the
       games present in the (subset) dumps.
select (self-retrieval on the candidate file + cand_games.npz only, see p02_player_open.py for
       the split; real mode needs exactly 100 games per candidate, committed p02 player tools and
       the unpacked Colab results committed and unchanged, PREREG_02 section 6 step 4):
       mean official score over 10 rounds x all candidates for every combination;
       the best mean, then the earliest combination within 0.003 of it; gate = selected - Q00
       >= 0.02 (recorded, not enforced here). Also the median of max_c cosO over the
       pseudo-queries per orientation (denominator of rho). Needs repro_local.json passed (unless
       --no-cnn) and guard_calibration.json (unless --smoke). Writes sr_report.json and
       frozen_config.json; refuses to overwrite an existing frozen_config.json (real mode).
       --no-cnn --reason TEXT: the registered fallback list {Q22, Q23}; no dump is read. Real mode
       accepts it only when a CNN dump is missing or OUT_DIR/repro_local.json says passed = false
       (PREREG_02 2.2); the basis is written to frozen_config.json (no_cnn_basis).
apply  refuses unless OUT_DIR/frozen_config.json is committed and identical to HEAD, the p02
       player tools are unchanged since select, and the inputs match (--smoke skips the git checks
       and is refused inside the repository). Full candidate games, all candidates, the question
       file; rho = median over the questions of max_c cosO / the frozen self-retrieval median,
       b_c / b_q = black share of the candidate / question games; three-level guard of
       PREREG_02 rev 2 (p02_player_open.guard_level; Q00 / Q01 exempt):
           level 3 (rho < theta2 or |b_c - b_q| > 0.5): CNN only, the E part (Q22 / Q23 -> Q00)
           level 2 (theta2 <= rho < theta1): w = min(w, 1), Q22 / Q23 -> Q04 / Q14
           level 1: the selected combination
       With --no-cnn (list {Q22, Q23}) levels 2 and 3 cannot switch to a CNN term; the selected
       combination is kept and the report says so. Writes player_val_top5.csv (question file
       order) and apply_report.json.
Output dirs inside results/prereg01 (or --p01-dir) are always refused.
--smoke runs only on SMOKE dumps (P02_DIR/config.json "smoke": true, whenever a CNN dump is read) and
never on the real candidate / question files (paths inside the repository, or files with the same
SHA-256 as data/local's, are refused); real runs refuse SMOKE dumps. Otherwise --smoke (which skips
the git checks) could run the real selection or apply in a scratch dir before frozen_config.json is
committed (PREREG_02 section 3).
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p02_player_open as po  # noqa: E402

REPO = po.REPO
TOP_N = po.TOP_N
Z_EPS = 1e-12
SELECT_TOL = 0.003
GATE_MIN = 0.02
FALLBACK_GATE_MIN = 0.50        # PREREG_02 rev 3, section 7 item 7: fallback list {Q22, Q23}, no Q00
REPRO_CENTROID_TOL = 1e-4
REPRO_MAX_DIFFERENT = 2          # 398 of 400
W_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
E_KINDS = ("E0", "E1")
TOP_COLS = ["top_%d" % i for i in range(1, TOP_N + 1)]


def _build_combos():
    out = [{"id": "Q00", "e": "E0", "o": None, "w": None},
           {"id": "Q01", "e": "E1", "o": None, "w": None}]
    n = 2
    for e, o in (("E0", "raw"), ("E1", "raw"), ("E0", "canon"), ("E1", "canon")):
        for w in W_GRID:
            out.append({"id": "Q%02d" % n, "e": e, "o": o, "w": w})
            n += 1
    out += [{"id": "Q22", "e": None, "o": "raw", "w": None},
            {"id": "Q23", "e": None, "o": "canon", "w": None}]
    return out


COMBOS = _build_combos()
COMBO_BY_ID = {c["id"]: c for c in COMBOS}
FALLBACK_IDS = ("Q22", "Q23")


def combo_id(e, o, w):
    for c in COMBOS:
        if c["e"] == e and c["o"] == o and c["w"] == w:
            return c["id"]
    raise KeyError((e, o, w))


# ---------------------------------------------------------------- reading inputs
def read_candidates(path):
    """[(game_id, player_id, colour, sgf)] in file order."""
    csv.field_size_limit(10**8)
    with open(po.refuse_forbidden(path), newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        need = {"game_id", "player_id", "sgf_content", "color"}
        if not need <= set(rd.fieldnames or []):
            raise SystemExit(f"{path}: candidate file needs columns {sorted(need)}")
        return [(r["game_id"], r["player_id"], r["color"], r["sgf_content"]) for r in rd]


def read_questions(path):
    """[(question_id, [(slot, colour, sgf), ...])] in file order; slots 1..num_games."""
    csv.field_size_limit(10**8)
    out = []
    with open(po.refuse_forbidden(path), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n = int(r["num_games"])
            out.append((r["question_id"], [(i, r["color_%d" % i], r["sgf_%d" % i])
                                           for i in range(1, n + 1)]))
    return out


def load_npz(path, keys):
    po.refuse_forbidden(path)
    if not os.path.exists(path):
        raise SystemExit(f"missing dump {path}; if the Colab player run failed, use "
                         f"'select --no-cnn --reason ...' (registered fallback Q22 / Q23)")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise SystemExit(f"{path}: missing arrays {missing}")
        return {k: z[k] for k in z.files}


def _strs(a):
    return [str(x) for x in a.tolist()]


class Data:
    """Candidate games (and optionally question games) with their moves and CNN sums."""

    def __init__(self, cand_path, p02_dir, use_cnn, smoke, q_path=None):
        self.shas = {"candidates_csv": po.sha256_file(cand_path)}
        self.weights = {}      # weights_sha256 stored in each dump (if present)
        self.paths = {"candidates_csv": cand_path}
        rows = read_candidates(cand_path)
        cd = None
        if use_cnn:
            p = os.path.join(p02_dir, "dumps", "cand_games.npz")
            cd = load_npz(p, ["game_id", "player_id", "color", "emb_sum", "n_windows"])
            self.shas["cand_games_npz"] = po.sha256_file(p)
            self.weights["cand_games_npz"] = str(cd["weights_sha256"]) if "weights_sha256" in cd else None
            self.paths["cand_games_npz"] = p
            gid = _strs(cd["game_id"])
            at = {g: i for i, g in enumerate(gid)}
            if len(at) != len(gid):
                raise SystemExit("cand_games.npz: duplicate game_id")
            if smoke:
                rows = [r for r in rows if r[0] in at]
            elif set(at) != {r[0] for r in rows}:
                raise SystemExit("cand_games.npz does not cover exactly the games of the candidate "
                                 "file (%d vs %d games)" % (len(at), len(rows)))
            pid, col = _strs(cd["player_id"]), _strs(cd["color"])
            bad = [r[0] for r in rows if pid[at[r[0]]] != r[1] or col[at[r[0]]] != r[2]]
            if bad:
                raise SystemExit(f"cand_games.npz: player/colour differ from the candidate file "
                                 f"for {len(bad)} games, e.g. {bad[:3]}")
        if not rows:
            raise SystemExit("no candidate games")
        gids = [r[0] for r in rows]
        if len(set(gids)) != len(gids):
            raise SystemExit("candidate file: duplicate game_id")
        for r in rows:
            if r[2] not in ("B", "W"):
                raise SystemExit(f"candidate game {r[0]}: colour {r[2]!r} is not B or W")
        self.cgames = [(r[0], r[1], r[2], po.moves_of(r[3])) for r in rows]
        self.cand_ids = sorted({r[1] for r in rows})
        self.c_by_player = {p: [] for p in self.cand_ids}
        for i, r in enumerate(rows):
            self.c_by_player[r[1]].append(i)
        self.use_cnn = use_cnn
        if use_cnn:
            order = [at[r[0]] for r in rows]
            self.cemb, self.cnw, self.cok = _cnn_arrays(cd["emb_sum"][order], cd["n_windows"][order],
                                                       np.ones(len(order), bool))
        self.qids, self.qgames, self.q_by_id = [], [], {}
        if q_path is not None:
            self._load_questions(q_path, p02_dir, use_cnn, smoke)
        self._G = {}

    def _load_questions(self, q_path, p02_dir, use_cnn, smoke):
        self.shas["questions_csv"] = po.sha256_file(q_path)
        self.paths["questions_csv"] = q_path
        qs = read_questions(q_path)
        qd = None
        if use_cnn:
            p = os.path.join(p02_dir, "dumps", "query_games.npz")
            qd = load_npz(p, ["question_id", "slot", "color", "emb_sum", "n_windows", "parsed"])
            self.shas["query_games_npz"] = po.sha256_file(p)
            self.weights["query_games_npz"] = str(qd["weights_sha256"]) if "weights_sha256" in qd else None
            self.paths["query_games_npz"] = p
            keys = list(zip(_strs(qd["question_id"]), [int(s) for s in qd["slot"].tolist()]))
            at = {k: i for i, k in enumerate(keys)}
            if len(at) != len(keys):
                raise SystemExit("query_games.npz: duplicate (question_id, slot)")
            want = {(q, s) for q, g in qs for s, _c, _x in g}
            if smoke:
                qs = [(q, [x for x in g if (q, x[0]) in at]) for q, g in qs]
                qs = [(q, g) for q, g in qs if g]
            elif set(at) != want:
                raise SystemExit("query_games.npz does not cover exactly the game slots of the "
                                 "question file (%d vs %d)" % (len(at), len(want)))
            qcol = _strs(qd["color"])
            bad = [(q, s) for q, g in qs for s, c, _x in g if qcol[at[(q, s)]] != c]
            if bad:
                raise SystemExit(f"query_games.npz: colour differs from the question file for "
                                 f"{len(bad)} slots, e.g. {bad[:3]}")
        order = []
        for q, g in qs:
            self.qids.append(q)
            self.q_by_id[q] = []
            for s, c, sgf in g:
                if c not in ("B", "W"):
                    raise SystemExit(f"question {q} slot {s}: colour {c!r} is not B or W")
                self.q_by_id[q].append(len(self.qgames))
                self.qgames.append((q, s, c, po.moves_of(sgf)))
                if use_cnn:
                    order.append(at[(q, s)])
        if len(set(self.qids)) != len(self.qids):
            raise SystemExit("question file: duplicate question_id")
        if use_cnn:
            self.qemb, self.qnw, self.qok = _cnn_arrays(
                qd["emb_sum"][order], qd["n_windows"][order], qd["parsed"][order].astype(bool))

    def G(self, orient):
        """(candidate game x token, question game x token) count matrices, shared columns."""
        if orient not in self._G:
            gc, gq = po.token_matrices([[(c, mv) for _g, _p, c, mv in self.cgames],
                                        [(c, mv) for _q, _s, c, mv in self.qgames]], orient)
            self._G[orient] = (gc, gq)
        return self._G[orient]


def _cnn_arrays(emb, nw, parsed):
    emb = np.asarray(emb, dtype=np.float64)
    nw = np.asarray(nw, dtype=np.float64).reshape(-1)
    if emb.ndim != 2 or emb.shape[0] != nw.shape[0]:
        raise SystemExit(f"emb_sum shape {emb.shape} does not match n_windows {nw.shape}")
    ok = parsed & (nw > 0) & np.isfinite(emb).all(1)
    emb = np.where(ok[:, None], emb, 0.0)
    nw = np.where(ok, nw, 0.0)
    return emb, nw, ok


# ---------------------------------------------------------------- CNN terms
def e_vectors(kind, emb, nw, ok, groups):
    """Per group: E0 centroid (window-weighted) or E1 vector; plus whether the group has any
    usable game."""
    A = po.indicator([[i for i in g if ok[i]] for g in groups], len(nw))
    if kind == "E0":
        S = np.asarray(A @ emb)
        n = np.asarray(A @ nw).reshape(-1)
        has = n > 0
        V = np.zeros_like(S)
        V[has] = S[has] / n[has, None]
        return V, has
    if kind == "E1":
        U = np.zeros_like(emb)
        U[ok] = emb[ok] / nw[ok, None]
        norm = np.linalg.norm(U, axis=1)
        U[norm > 0] /= norm[norm > 0, None]
        cnt = np.asarray(A @ ok.astype(np.float64)).reshape(-1)
        has = cnt > 0
        V = np.zeros((len(groups), emb.shape[1]))
        V[has] = np.asarray(A @ U)[has] / cnt[has, None]
        norm = np.linalg.norm(V, axis=1)
        V[norm > 0] /= norm[norm > 0, None]
        return V, has
    raise ValueError(kind)


def neg_l2(Qv, Cv, chunk=32):
    out = np.empty((len(Qv), len(Cv)))
    for s in range(0, len(Qv), chunk):
        d = Qv[s:s + chunk, None, :] - Cv[None, :, :]
        out[s:s + chunk] = -np.sqrt((d * d).sum(-1))
    return out


def e_scores(kind, c_arrays, c_groups, q_arrays, q_groups):
    """(score matrix n_q x n_c, query has-term mask, number of candidates without any window)."""
    Cv, c_has = e_vectors(kind, *c_arrays, c_groups)
    Qv, q_has = e_vectors(kind, *q_arrays, q_groups)
    M = neg_l2(Qv, Cv) if kind == "E0" else Qv @ Cv.T
    n_c_missing = int((~c_has).sum())
    if n_c_missing:
        fill = M[:, c_has].min(1, keepdims=True) if c_has.any() else np.zeros((len(Qv), 1))
        M[:, ~c_has] = fill
    return M, q_has, n_c_missing


# ---------------------------------------------------------------- fusion and ranking
def zscore_rows(M, row_ok):
    """z over the candidates of each question; 0 where the term is missing or std < 1e-12."""
    M = np.asarray(M, dtype=np.float64)
    Z = np.zeros(M.shape)
    if M.size == 0:
        return Z
    mu = M.mean(1)
    sd = M.std(1)
    good = np.asarray(row_ok, bool) & np.isfinite(sd) & (sd >= Z_EPS)
    Z[good] = (M[good] - mu[good, None]) / sd[good, None]
    return Z


def fuse(combo, terms):
    """terms: {'E0'|'E1'|'raw'|'canon': (score matrix, row_ok)} -> fused score matrix."""
    s = None
    if combo["e"] is not None:
        s = zscore_rows(*terms[combo["e"]][:2])
    if combo["o"] is not None:
        w = 1.0 if combo["e"] is None else float(combo["w"])
        z = zscore_rows(*terms[combo["o"]][:2])
        s = w * z if s is None else s + w * z
    return s


def all_missing(combo, terms):
    """Per question: every term of the combination is missing (-> official fallback)."""
    miss = None
    for key in (combo["e"], combo["o"]):
        if key is None:
            continue
        m = ~np.asarray(terms[key][1], bool)
        miss = m if miss is None else (miss & m)
    return miss


def top_ids(S, cand_ids, n=TOP_N):
    idx = np.arange(S.shape[1])
    out = []
    for row in S:
        order = np.lexsort((idx, -row + 0.0))
        out.append([cand_ids[i] for i in order[:n]])
    return out


# ---------------------------------------------------------------- git / provenance
def git_head():
    try:
        r = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    except OSError:
        return None


def _rel(path):
    return os.path.relpath(os.path.realpath(path), os.path.realpath(REPO)).replace(os.sep, "/")


def tools_dirty():
    names = ["tools/p02_player_open.py", "tools/p02_player_fuse.py", "tools/p02_moves.py",
             "tools/c10data.py"]
    r = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--"] + names,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("git status failed: " + r.stderr.strip())
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def check_committed(path):
    """frozen_config.json must be committed and identical to HEAD (line ends folded)."""
    if not po.inside_repo(path):
        raise SystemExit(f"{path} is outside the repository: apply needs a committed "
                         f"frozen_config.json (use --smoke for a scratch run)")
    rel = _rel(path)
    head = subprocess.run(["git", "-C", REPO, "show", "HEAD:" + rel], capture_output=True)
    if head.returncode != 0:
        raise SystemExit(f"{rel} is not committed; commit frozen_config.json before apply")
    st = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", rel],
                        capture_output=True, text=True)
    if st.stdout.strip():
        raise SystemExit(f"{rel} has uncommitted changes: {st.stdout.strip()}")
    with open(path, "rb") as f:
        cur = f.read()
    if head.stdout.replace(b"\r\n", b"\n") != cur.replace(b"\r\n", b"\n"):
        raise SystemExit(f"{rel} differs from HEAD")
    return git_head()


def _read_json(path):
    with open(po.refuse_forbidden(path), encoding="utf-8") as f:
        return json.load(f)


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- smoke / real input guards
NB_FILES = ("repro_check.json", "run_log.json", "config.json", "env.json", "timings.json",
            "p01_readonly_before.json", "p01_readonly_after.json", "player_val_top5_repro.csv")
DUMP_FILES = ("cand_games.npz", "query_games.npz")
REAL_INPUT_FILES = ("player_val_candidates.csv", "player_val_questions.csv")     # in data/local
_SHA_CACHE = {}


def _sha_cached(path):
    st = os.stat(path)
    key = (os.path.normcase(os.path.realpath(path)), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = po.sha256_file(path)
    return _SHA_CACHE[key]


def notebook_smoke_flag(p02_dir):
    """'smoke' in the notebook's own P02_DIR/config.json (None when the file or a boolean flag is missing)."""
    path = os.path.join(p02_dir, "config.json")
    if not os.path.exists(path):
        return None
    obj = _read_json(path)
    flag = obj.get("smoke") if isinstance(obj, dict) else None
    return flag if isinstance(flag, bool) else None


def check_dump_kind(p02_dir, smoke):
    """--smoke only on dumps the notebook wrote in SMOKE mode; a real run never on SMOKE dumps."""
    flag = notebook_smoke_flag(p02_dir)
    if smoke and flag is not True:
        raise SystemExit(f"--smoke needs dumps from a SMOKE notebook run: {os.path.join(p02_dir, 'config.json')} "
                         f"must say \"smoke\": true (got {flag!r})")
    if not smoke and flag is True:
        raise SystemExit(f"{p02_dir} holds SMOKE dumps (config.json \"smoke\": true); real runs refuse them")
    return flag


def check_smoke_inputs(smoke, *paths):
    """--smoke never reads the real validation candidate / question files: paths inside the repository
    and files with the same SHA-256 as data/local's are refused (use subset files in a scratch dir)."""
    if not smoke:
        return
    real = {}
    for name in REAL_INPUT_FILES:
        p = os.path.join(REPO, "data", "local", name)
        if os.path.exists(p):
            real[_sha_cached(p)] = name
    for p in paths:
        if p is None:
            continue
        if po.inside_repo(p):
            raise SystemExit(f"--smoke refused: {p!r} is inside the repository; smoke runs use subset "
                             f"files in a scratch directory")
        if os.path.exists(p) and _sha_cached(p) in real:
            raise SystemExit(f"--smoke refused: {p!r} is a copy of data/local/{real[_sha_cached(p)]}")


def no_cnn_basis(args):
    """PREREG_02 2.2: the fallback list {Q22, Q23} only when the Q00 reproduction failed (and was not
    fixed within 7 days; the main session records that) or the CNN dumps are missing. Real mode
    refuses --no-cnn otherwise; the basis is recorded in frozen_config.json."""
    dd = os.path.join(args.p02_dir, "dumps")
    basis = {k.replace(".", "_") + "_exists": os.path.exists(os.path.join(dd, k)) for k in DUMP_FILES}
    rp = os.path.join(args.out_dir, "repro_local.json")
    basis["repro_local_json"] = None
    if os.path.exists(rp):
        r = _read_json(rp)
        basis["repro_local_json"] = {"passed": r.get("passed"), "created": r.get("created"),
                                     "smoke": r.get("smoke"), "sha256": po.sha256_text_file(rp)}
    basis["dumps_missing"] = not all(os.path.exists(os.path.join(dd, k)) for k in DUMP_FILES)
    basis["repro_failed"] = bool(basis["repro_local_json"] and basis["repro_local_json"]["passed"] is False)
    basis["rule"] = ("real mode accepts --no-cnn only when a CNN dump is missing or OUT_DIR/repro_local.json "
                     "says passed = false; repro_local.json 'created' is the failure date for the 7-day rule")
    if not args.smoke and not (basis["dumps_missing"] or basis["repro_failed"]):
        raise SystemExit("--no-cnn refused: the registered fallback {Q22, Q23} applies only when the Colab "
                         "dumps are missing or the Q00 reproduction failed (run repro first); %s"
                         % json.dumps({k: basis[k] for k in ("dumps_missing", "repro_failed")}))
    return basis


def require_committed_results(p02_dir, need_dumps):
    """The unpacked Colab results in P02_DIR must be committed and unchanged before select (PREREG_02
    section 6 step 4). need_dumps: the CNN dumps must be there (not with --no-cnn)."""
    if not po.inside_repo(p02_dir):
        if need_dumps:
            raise SystemExit(f"{p02_dir} is outside the repository; select runs on the committed Colab "
                             f"results (use --smoke for smoke dumps)")
        return {"files": 0, "note": "P02_DIR outside the repository; no notebook result is read"}
    files = [os.path.join(p02_dir, "dumps", k) for k in DUMP_FILES]
    if need_dumps:
        missing = [f for f in files + [os.path.join(p02_dir, "repro_check.json"),
                                       os.path.join(p02_dir, "run_log.json")] if not os.path.exists(f)]
        if missing:
            raise SystemExit(f"notebook result files missing: {missing}")
    files = [f for f in files if os.path.exists(f)]
    files += [os.path.join(p02_dir, n) for n in NB_FILES if os.path.exists(os.path.join(p02_dir, n))]
    if not files:
        return {"files": 0, "note": "no notebook result file in P02_DIR"}
    rels = [_rel(f) for f in files]
    r = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", "--"] + rels,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"the Colab results in {p02_dir} are not all committed; commit the unpacked results "
                         f"before select ({r.stderr.strip()[:300]})")
    st = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--"] + rels, capture_output=True, text=True)
    if st.returncode != 0 or st.stdout.strip():
        raise SystemExit(f"the committed Colab results changed in the working tree: {st.stdout.strip()[:300]}")
    return {"files": len(rels), "git_head": git_head(),
            "rule": "dumps/* and the notebook run records tracked by git and unchanged"}


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x)) if len(x) else float("nan")


# ---------------------------------------------------------------- repro
def cmd_repro(args):
    t0 = time.time()
    po.check_out_dir(args.out_dir, args.smoke, extra_readonly=(args.p01_dir,))
    check_smoke_inputs(args.smoke, args.candidates, args.questions)
    check_dump_kind(args.p02_dir, args.smoke)
    d = Data(args.candidates, args.p02_dir, True, args.smoke, q_path=args.questions)
    c_groups = [d.c_by_player[p] for p in d.cand_ids]
    q_groups = [d.q_by_id[q] for q in d.qids]
    c_arr, q_arr = (d.cemb, d.cnw, d.cok), (d.qemb, d.qnw, d.qok)
    M, q_has, n_c_missing = e_scores("E0", c_arr, c_groups, q_arr, q_groups)
    ours = top_ids(fuse(COMBO_BY_ID["Q00"], {"E0": (M, q_has)}), d.cand_ids)
    # prereg01's own way: argsort of the L2 distance, fallback = first 5 sorted candidates
    direct = [[d.cand_ids[i] for i in np.argsort(-M[k], kind="stable")[:TOP_N]] if q_has[k]
              else d.cand_ids[:TOP_N] for k in range(len(d.qids))]

    p01_csv = os.path.join(args.p01_dir, "player_val_top5.csv")
    p01_npz = os.path.join(args.p01_dir, "player_val_embeddings.npz")
    with open(po.refuse_forbidden(p01_csv), newline="", encoding="utf-8-sig") as f:
        p01 = {r["question_id"]: [r[c] for c in TOP_COLS] for r in csv.DictReader(f)}
    z = load_npz(p01_npz, ["candidate_ids", "centroids", "question_ids", "queries"])
    same_q = set(p01) == set(d.qids)
    ident = [q for k, q in enumerate(d.qids) if p01.get(q) == ours[k]]
    ident_direct = [q for k, q in enumerate(d.qids) if p01.get(q) == direct[k]]
    differ = [q for q in d.qids if q not in set(ident)]

    Cv, c_has = e_vectors("E0", *c_arr, c_groups)
    Qv, _ = e_vectors("E0", *q_arr, q_groups)
    p01_c = dict(zip(_strs(z["candidate_ids"]), np.asarray(z["centroids"], np.float64)))
    p01_q = dict(zip(_strs(z["question_ids"]), np.asarray(z["queries"], np.float64)))
    ours_c = {p: Cv[i] for i, p in enumerate(d.cand_ids) if c_has[i]}
    ours_q = {q: Qv[k] for k, q in enumerate(d.qids) if q_has[k]}
    same_c_ids = set(p01_c) == set(ours_c)
    same_q_ids = set(p01_q) == set(ours_q)
    c_diff = max((float(np.abs(ours_c[p] - p01_c[p]).max()) for p in ours_c if p in p01_c), default=float("nan"))
    q_diff = max((float(np.abs(ours_q[q] - p01_q[q]).max()) for q in ours_q if q in p01_q), default=float("nan"))

    n_q = len(d.qids)
    need = n_q if args.smoke else n_q - REPRO_MAX_DIFFERENT
    w = [v for v in d.weights.values() if v is not None]
    same_weights = len(set(w)) <= 1
    notebook = notebook_status(args.p02_dir)
    notebook_ok = notebook["repro_check_passed"] is True and notebook["last_run_status"] == "completed"
    passed = bool(same_q and same_c_ids and len(ident) >= need and np.isfinite(c_diff)
                  and c_diff <= REPRO_CENTROID_TOL and same_weights and (notebook_ok or args.smoke))
    out = {
        "what": "PREREG_02 2.2 / plan.md 4.3 repro: Q00 (E0 = window-weighted centroid, -L2, fused "
                "path s = z(E0)) from the prereg02 per-game dumps vs prereg01's top-5 and saved "
                "centroids",
        "smoke": bool(args.smoke),
        "created": now_iso(),
        "passed": passed,
        "criteria": {"top5_identical_at_least": need, "of_questions": n_q,
                     "max_abs_centroid_diff_at_most": REPRO_CENTROID_TOL,
                     "question_ids_equal": True, "candidate_ids_equal": True,
                     "dump_weights_equal": True},
        "n_questions": n_q,
        "n_top5_identical": len(ident),
        "n_top5_identical_direct_argsort": len(ident_direct),
        "questions_differing": differ,
        "question_ids_equal": same_q,
        "candidate_ids_equal": same_c_ids,
        "query_ids_equal": same_q_ids,
        "max_abs_centroid_diff": c_diff,
        "max_abs_query_mean_diff": q_diff,
        "n_candidates": len(d.cand_ids),
        "n_candidate_games": len(d.cgames),
        "n_candidates_without_windows": n_c_missing,
        "n_question_games": len(d.qgames),
        "n_questions_without_windows": int((~q_has).sum()),
        "dump_weights_sha256": d.weights,
        "dump_weights_equal": same_weights,
        "notebook_run": notebook,
        "notebook_run_ok": notebook_ok,
        "notebook_run_required": not args.smoke,
        "inputs": {k: {"path": _disp(d.paths[k]), "sha256": v} for k, v in d.shas.items()}
        | {"p01_top5_csv": {"path": _disp(p01_csv), "sha256": po.sha256_file(p01_csv)},
           "p01_embeddings_npz": {"path": _disp(p01_npz), "sha256": po.sha256_file(p01_npz)}},
        "tools_sha256": po.source_shas(),
        "git_head": git_head(),
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "repro_local.json")
    po.write_json(path, out)
    print("repro %s: top-5 identical %d/%d (need %d; direct argsort %d), max centroid diff %.3g, "
          "max query diff %.3g, notebook run %s%s -> %s"
          % ("PASSED" if passed else "FAILED", len(ident), n_q, need, len(ident_direct), c_diff, q_diff,
             "ok" if notebook_ok else "NOT ok (%s)" % notebook,
             " (not required with --smoke)" if args.smoke else "", path))
    if not passed:
        sys.exit(1)


def notebook_status(p02_dir):
    """What the Colab run itself reported: repro_check.json 'passed' and the status of the last run in
    run_log.json ('completed' only when the dumps, the reproduction check and the prereg01 read-only
    fingerprint all passed). Missing files give None (and a real run then fails)."""
    out = {"repro_check_passed": None, "last_run_status": None, "last_run_p01_unchanged": None}
    rc = os.path.join(p02_dir, "repro_check.json")
    if os.path.exists(rc):
        obj = _read_json(rc)
        out["repro_check_passed"] = obj.get("passed") if isinstance(obj, dict) else None
    rl = os.path.join(p02_dir, "run_log.json")
    if os.path.exists(rl):
        runs = (_read_json(rl) or {}).get("runs") or []
        if runs:
            out["last_run_status"] = runs[-1].get("status")
            out["last_run_p01_unchanged"] = runs[-1].get("p01_unchanged")
    return out


def _disp(path):
    return _rel(path) if po.inside_repo(path) else os.path.abspath(path)


# ---------------------------------------------------------------- select
def self_retrieval(d, combos, strict, log=print):
    """Run the self-retrieval rounds. Returns per-combo point arrays per round, the max_c cosO
    of every pseudo-query per orientation, and counts."""
    rounds = po.sr_rounds({p: [d.cgames[i][0] for i in d.c_by_player[p]] for p in d.cand_ids},
                          strict=strict)
    row = {(g[1], g[0]): i for i, g in enumerate(d.cgames)}   # (player_id, game_id) -> row
    per_round = {c["id"]: [] for c in combos}
    max_cos = {o: [] for o in po.ORIENTS}
    n_queries, n_missing = [], {k: 0 for k in ("raw", "canon", "E0", "E1")}
    c_arr = (d.cemb, d.cnw, d.cok) if d.use_cnn else None
    for r, items in enumerate(rounds):
        t = time.time()
        c_groups = [[row[(p, g)] for g in refs] for p, _q, refs in items]
        qi = [i for i, (_p, q, _r) in enumerate(items) if q]
        n_queries.append(len(qi))
        if not qi:
            for c in combos:
                per_round[c["id"]].append([])
            continue
        q_groups = [[row[(items[i][0], g)] for g in items[i][1]] for i in qi]
        terms = {}
        for o in po.ORIENTS:
            G = d.G(o)[0]
            cos, has = po.open_cos(G, c_groups, G, q_groups)
            terms[o] = (cos, has)
            max_cos[o].extend(cos.max(1)[has].tolist())
            n_missing[o] += int((~has).sum())
        if d.use_cnn:
            for k in E_KINDS:
                M, has, _ = e_scores(k, c_arr, c_groups, c_arr, q_groups)
                terms[k] = (M, has)
                n_missing[k] += int((~has).sum())
        true = np.asarray(qi, dtype=np.int64)
        for c in combos:
            per_round[c["id"]].append(po.points(po.true_ranks(fuse(c, terms), true)).tolist())
        log("round %d: %d pseudo-queries (%.1fs)" % (r, len(qi), time.time() - t))
    return rounds, per_round, max_cos, n_queries, n_missing


def choose(means, order):
    """Best mean, then the earliest combination (list order) within SELECT_TOL of it."""
    ok = [c for c in order if np.isfinite(means[c])]
    if not ok:
        raise SystemExit("no combination has a self-retrieval score (no pseudo-query)")
    best_id = max(ok, key=lambda c: (means[c], -order.index(c)))
    best = means[best_id]
    chosen = next(c for c in ok if best - means[c] <= SELECT_TOL)
    return best_id, chosen


def cmd_select(args):
    t0 = time.time()
    po.check_out_dir(args.out_dir, args.smoke)
    check_smoke_inputs(args.smoke, args.candidates)
    frozen_path = os.path.join(args.out_dir, "frozen_config.json")
    if os.path.exists(frozen_path) and not args.smoke:
        raise SystemExit(f"{frozen_path} already exists: the selection is frozen. Remove it "
                         f"deliberately (and record why) before selecting again")
    use_cnn = not args.no_cnn
    if args.no_cnn and not (args.reason or "").strip():
        raise SystemExit("--no-cnn needs --reason (e.g. 'Colab player dumps missing' or "
                         "'Q00 reproduction failed and not fixed within 7 days')")
    no_cnn = no_cnn_basis(args) if args.no_cnn else None
    if use_cnn:
        check_dump_kind(args.p02_dir, args.smoke)
    results_commit = None
    if not args.smoke:
        dirty = tools_dirty()
        if dirty:
            raise SystemExit(f"commit the p02 player tools before select: {dirty}")
        results_commit = require_committed_results(args.p02_dir, need_dumps=use_cnn)

    repro = None
    if use_cnn:
        rp = os.path.join(args.out_dir, "repro_local.json")
        if not os.path.exists(rp):
            raise SystemExit(f"run 'repro' first ({rp} missing)")
        repro = _read_json(rp)
        if not repro.get("passed"):
            raise SystemExit("repro_local.json did not pass: the registered fallback is "
                             "'select --no-cnn --reason ...' (Q22, Q23)")
    guard_path = args.guard or os.path.join(args.out_dir, "guard_calibration.json")
    guard = None
    if os.path.exists(guard_path):
        g = _read_json(guard_path)
        if bool(g.get("smoke")) and not args.smoke:
            raise SystemExit(f"{guard_path} is a smoke calibration")
        if set(g.get("guard", {})) != set(po.ORIENTS):
            raise SystemExit(f"{guard_path}: no three-level thresholds for {list(po.ORIENTS)}")
        src = ("OUT_DIR/guard_calibration.json" if args.guard is None else _disp(guard_path))
        guard = {"source": src, "sha256": po.sha256_text_file(guard_path),
                 "smoke": bool(g.get("smoke")), "thresholds": g["guard"]}
    elif not args.smoke:
        raise SystemExit(f"run 'p02_player_open.py calibrate-guard' first ({guard_path} missing)")

    d = Data(args.candidates, args.p02_dir, use_cnn, args.smoke)
    if use_cnn:
        for k in ("cand_games_npz",):
            if repro["inputs"].get(k, {}).get("sha256") != d.shas[k]:
                raise SystemExit(f"{k} changed since repro_local.json; rerun repro")
        if repro["inputs"]["candidates_csv"]["sha256"] != d.shas["candidates_csv"]:
            raise SystemExit("candidate file changed since repro_local.json; rerun repro")
    combos = COMBOS if use_cnn else [COMBO_BY_ID[c] for c in FALLBACK_IDS]
    order = [c["id"] for c in combos]
    _rounds, per_round, max_cos, n_queries, n_missing = self_retrieval(
        d, combos, strict=not args.smoke, log=lambda s: print(s, flush=True))

    means, report_rows = {}, []
    for c in combos:
        allp = [x for rr in per_round[c["id"]] for x in rr]
        means[c["id"]] = _stats(allp)
        rmeans = [_stats(rr) for rr in per_round[c["id"]]]
        pts = np.asarray(allp)
        report_rows.append({"id": c["id"], "e": c["e"], "o": c["o"], "w": c["w"],
                            "mean": means[c["id"]],
                            "round_means": rmeans,
                            "round_sd": float(np.std([m for m in rmeans if np.isfinite(m)], ddof=1))
                            if sum(np.isfinite(rmeans)) > 1 else None,
                            "top1_share": float((pts == 1.0).mean()) if len(pts) else None,
                            "top5_share": float((pts > 0).mean()) if len(pts) else None})
    best_id, chosen = choose(means, order)
    sel = COMBO_BY_ID[chosen]
    q00 = means.get("Q00")
    gate = ({"q00_mean": q00, "selected_mean": means[chosen], "selected_minus_q00": means[chosen] - q00,
             "threshold": GATE_MIN, "passed": bool(means[chosen] - q00 >= GATE_MIN)}
            if q00 is not None else
            {"q00_mean": None, "selected_mean": means[chosen], "selected_minus_q00": None,
             "threshold": FALLBACK_GATE_MIN, "passed": bool(means[chosen] >= FALLBACK_GATE_MIN),
             "rule": "PREREG_02 rev 3 section 7 item 7: no Q00 (CNN dumps not used), so the gate is "
                     "selected self-retrieval mean >= %.2f" % FALLBACK_GATE_MIN})
    sr_med = {o: (float(np.median(max_cos[o])) if max_cos[o] else None) for o in po.ORIENTS}

    inputs = {k: {"path": _disp(d.paths[k]), "sha256": v} for k, v in d.shas.items()}
    if use_cnn:
        inputs["cand_games_npz"]["weights_sha256"] = d.weights.get("cand_games_npz")
    if repro is not None:
        rp = os.path.join(args.out_dir, "repro_local.json")
        inputs["repro_local_json"] = {"path": "OUT_DIR/repro_local.json", "passed": True,
                                      "n_top5_identical": repro["n_top5_identical"],
                                      "n_questions": repro["n_questions"],
                                      "max_abs_centroid_diff": repro["max_abs_centroid_diff"],
                                      "query_games_npz_sha256": repro["inputs"]
                                      .get("query_games_npz", {}).get("sha256")}
    split_text = ("per candidate: games sorted by sha256('%s|<player_id>|<game_id>'), 5 parts by "
                  "position (100 -> 5 x 20); round r = 0..%d uses part r mod 5, rng = "
                  "random.Random(int(sha256('%s|<r>').hexdigest()[:16], 16)), candidates in sorted "
                  "id order draw k = rng.randint(%d, %d) and query = rng.sample(part, k); references "
                  "= the other 4 parts (80 games); IDF and CNN centroids from the references only"
                  % (po.SR_SALT, po.SR_ROUNDS - 1, po.SR_K_SALT, po.K_MIN, po.K_MAX))
    frozen = {
        "what": "PREREG_02 2.2 competition 1: combination chosen by candidate-pool self-retrieval "
                "(candidate file + cand_games.npz only). apply uses exactly this file.",
        "smoke": bool(args.smoke),
        "cnn": use_cnn,
        "no_cnn_reason": None if use_cnn else args.reason,
        "no_cnn_basis": no_cnn,
        "notebook_results_commit_check": results_commit,
        "combination_list": order,
        "combinations": combos,
        "selected": sel,
        "best": {"id": best_id, "mean": means[best_id]},
        "selection_rule": "highest self-retrieval mean; the earliest combination in list order "
                          "within %.3f of it" % SELECT_TOL,
        "self_retrieval_mean": means,
        "gate": gate,
        "sr_max_cos_median": sr_med,
        "guard": guard,
        "guard_rule": "PREREG_02 rev 2 three-level guard. rho = median over the questions of "
                      "max_c cosO (full candidate games) / sr_max_cos_median, orientation of the "
                      "selected combination; b_c / b_q = black share of the candidate / question "
                      "games. Q00, Q01 exempt. level 3 (rho < theta2 or |b_c - b_q| > 0.5): CNN "
                      "only, the E part (Q22 / Q23 -> Q00); level 2 (theta2 <= rho < theta1): "
                      "w = min(w, 1), Q22 -> Q04, Q23 -> Q14; level 1: as selected. theta1 "
                      "disabled: no level 2; theta2 disabled: theta_RX is the only threshold "
                      "(below it level 3)",
        "self_retrieval": split_text,
        "inputs": inputs,
        "tools_sha256": po.source_shas(),
        "git_head_at_select": git_head(),
    }
    report = {
        "what": "PREREG_02 2.2 competition 1 self-retrieval report (official metric per "
                "pseudo-query: e^-(r-1) for rank r <= 5 among all candidates, ties by candidate id)",
        "smoke": bool(args.smoke),
        "split": split_text,
        "n_candidates": len(d.cand_ids),
        "n_candidate_games": len(d.cgames),
        "rounds": po.SR_ROUNDS,
        "pseudo_queries_per_round": n_queries,
        "pseudo_queries_missing_term": n_missing,
        "combinations": report_rows,
        "best": {"id": best_id, "mean": means[best_id]},
        "selected": chosen,
        "gate": gate,
        "sr_max_cos_median": sr_med,
        "n_pseudo_queries_with_tokens": {o: len(max_cos[o]) for o in po.ORIENTS},
        "guard": guard,
        "inputs": inputs,
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    po.write_json(os.path.join(args.out_dir, "sr_report.json"), report)
    po.write_json(frozen_path, frozen)
    print("selected %s (best %s %.4f; Q00 %s) gate %s -> %s" % (
        chosen, best_id, means[best_id], "%.4f" % q00 if q00 is not None else "n/a",
        gate["passed"], frozen_path))
    print("commit frozen_config.json before apply, together with competition 2's frozen_config.json "
          "(PREREG_02 rev 3, section 7 item 6: both frozen before any apply)")


# ---------------------------------------------------------------- apply
def guard_decision(sel, rho, guard, b_c, b_q, cnn):
    """(applied combination, level, reason) for the PREREG_02 rev 2 three-level guard.

    sel: selected combination; rho: {orient: rho}; guard: frozen guard entry (None = no
    calibration, smoke only); b_c / b_q: black share of candidate / question games; cnn: whether
    CNN terms exist (False for the fallback list {Q22, Q23}). Pure function (tested)."""
    if sel["o"] is None:
        return sel, 0, "exempt: CNN-only combination"
    o = sel["o"]
    if guard is None:
        return sel, 0, "no guard calibration (smoke run without calibrate-guard)"
    level, why = po.guard_level(rho.get(o), guard["thresholds"][o], b_c, b_q)
    if level == 1:
        return sel, 1, why
    if level == 3:
        if sel["e"] is not None:
            return COMBO_BY_ID[combo_id(sel["e"], None, None)], 3, why + ": CNN only (E part)"
        if cnn:
            return COMBO_BY_ID["Q00"], 3, why + ": CNN only (opening-only -> Q00)"
        return sel, 3, why + ": level 3 but no CNN term exists (fallback list); selection kept"
    if sel["e"] is not None:
        w = min(float(sel["w"]), 1.0)
        return COMBO_BY_ID[combo_id(sel["e"], o, w)], 2, why + ": w = min(w, 1) = %g" % w
    if cnn:
        return COMBO_BY_ID[combo_id("E0", o, 1.0)], 2, why + ": opening-only -> E0 + %s, w = 1" % o
    return sel, 2, why + ": level 2 but no CNN term exists (fallback list); selection kept"


def cmd_apply(args):
    t0 = time.time()
    po.check_out_dir(args.out_dir, args.smoke)
    check_smoke_inputs(args.smoke, args.candidates, args.questions)
    frozen_path = os.path.join(args.out_dir, "frozen_config.json")
    if not os.path.exists(frozen_path):
        raise SystemExit(f"{frozen_path} missing: run select and commit it first")
    head = None
    if not args.smoke:
        head = check_committed(frozen_path)
    fz = _read_json(frozen_path)
    if bool(fz.get("smoke")) != bool(args.smoke):
        raise SystemExit("frozen_config.json smoke flag %s does not match this run" % fz.get("smoke"))
    now = po.source_shas()
    changed = [k for k in ("p02_player_open.py", "p02_player_fuse.py", "p02_moves.py")
               if fz["tools_sha256"].get(k) != now.get(k)]
    if changed and not args.smoke:
        raise SystemExit(f"tools changed since select: {changed}; apply must run the frozen code")
    if not args.smoke:
        dirty = tools_dirty()
        if dirty:
            raise SystemExit(f"uncommitted changes in the p02 player tools: {dirty}")
    use_cnn = bool(fz["cnn"])
    if use_cnn:
        check_dump_kind(args.p02_dir, args.smoke)
    d = Data(args.candidates, args.p02_dir, use_cnn, args.smoke, q_path=args.questions)
    if len(d.cand_ids) < TOP_N:
        raise SystemExit(f"only {len(d.cand_ids)} candidates; need at least {TOP_N}")
    for k in ("candidates_csv", "cand_games_npz"):
        if k in fz["inputs"] and fz["inputs"][k]["sha256"] != d.shas.get(k):
            raise SystemExit(f"{k} differs from the one used by select")
    if use_cnn:
        want = fz["inputs"].get("repro_local_json", {}).get("query_games_npz_sha256")
        if want != d.shas.get("query_games_npz"):
            raise SystemExit("query_games.npz differs from the one checked by repro")

    c_groups = [d.c_by_player[p] for p in d.cand_ids]
    q_groups = [d.q_by_id[q] for q in d.qids]
    terms, rho, num = {}, {}, {}
    for o in po.ORIENTS:
        gc, gq = d.G(o)
        cos, has = po.open_cos(gc, c_groups, gq, q_groups)
        terms[o] = (cos, has)
        mx = cos.max(1)[has] if len(cos) else np.zeros(0)
        num[o] = float(np.median(mx)) if len(mx) else None
        den = fz["sr_max_cos_median"].get(o)
        rho[o] = num[o] / den if (num[o] is not None and den) else None
    n_c_missing = {}
    if use_cnn:
        c_arr, q_arr = (d.cemb, d.cnw, d.cok), (d.qemb, d.qnw, d.qok)
        for k in E_KINDS:
            M, has, n_c_missing[k] = e_scores(k, c_arr, c_groups, q_arr, q_groups)
            terms[k] = (M, has)
    sel = fz["selected"]
    b_c = po.colour_share(g[2] for g in d.cgames)
    b_q = po.colour_share(g[2] for g in d.qgames)
    applied, level, reason = guard_decision(sel, rho, fz["guard"], b_c, b_q, use_cnn)
    S = fuse(applied, terms)
    tops = top_ids(S, d.cand_ids)
    fb = all_missing(applied, terms)
    for k in np.flatnonzero(fb):
        if tops[k] != d.cand_ids[:TOP_N]:
            raise AssertionError("fallback question not filled with the first sorted candidates")

    out_csv = os.path.join(args.out_dir, "player_val_top5.csv")
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id"] + TOP_COLS)
        for q, t in zip(d.qids, tops):
            w.writerow([q] + t)
    os.replace(tmp, out_csv)
    report = {
        "what": "PREREG_02 2.2 competition 1 apply: frozen combination on the full candidate pool "
                "and the question file, with the guard",
        "smoke": bool(args.smoke),
        "frozen_config": {"path": _disp(frozen_path), "sha256": po.sha256_text_file(frozen_path),
                          "git_head": head},
        "selected": sel,
        "applied": applied,
        "guard_level": level,
        "guard_reason": reason,
        "guard_changed_combination": applied["id"] != sel["id"],
        "b_c_candidate_black_share": b_c,
        "b_q_question_black_share": b_q,
        "rho": rho,
        "rho_numerator_max_cos_median": num,
        "rho_denominator_sr_max_cos_median": fz["sr_max_cos_median"],
        "guard": fz["guard"],
        "n_questions": len(d.qids),
        "n_candidates": len(d.cand_ids),
        "n_candidate_games": len(d.cgames),
        "n_question_games": len(d.qgames),
        "n_questions_without_term": {k: int((~np.asarray(v[1], bool)).sum()) for k, v in terms.items()},
        "n_candidates_without_windows": n_c_missing,
        "n_fallback_questions": int(fb.sum()),
        "inputs": {k: {"path": _disp(d.paths[k]), "sha256": v} for k, v in d.shas.items()},
        "output": {"path": _disp(out_csv), "sha256": po.sha256_file(out_csv)},
        "tools_sha256": now,
        "tools_changed_since_select": changed,
        "runtime_s": round(time.time() - t0, 1),
    }
    blocked = []
    if fz.get("gate", {}).get("passed") is not True:
        blocked.append("gate not passed in select (PREREG_02 2.2 / rev 3 7.7): not submitted")
    if not use_cnn and level == 3:
        blocked.append("fallback list and guard level 3: no CNN term to switch to "
                       "(PREREG_02 rev 3 7.7): not submitted")
    report["score_allowed"] = not blocked
    report["score_blocked_reasons"] = blocked
    po.write_json(os.path.join(args.out_dir, "apply_report.json"), report)
    print("applied %s (selected %s; guard level %d: %s); %d questions, %d fallback -> %s"
          % (applied["id"], sel["id"], level, reason, len(d.qids), int(fb.sum()), out_csv))
    print("score allowed: %s%s" % (not blocked, (" (" + "; ".join(blocked) + ")") if blocked else ""))


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    res = os.path.join(REPO, "results")
    local = os.path.join(REPO, "data", "local")

    def common(p, questions=True, p01=False):
        p.add_argument("--p02-dir", default=os.path.join(res, "prereg02", "player"),
                       help="notebook result dir holding dumps/")
        p.add_argument("--candidates", default=os.path.join(local, "player_val_candidates.csv"))
        if questions:
            p.add_argument("--questions", default=os.path.join(local, "player_val_questions.csv"))
        if p01:
            p.add_argument("--p01-dir", default=os.path.join(res, "prereg01", "player"))
        p.add_argument("--out-dir", default=os.path.join(res, "prereg02", "player"))
        p.add_argument("--smoke", action="store_true",
                       help="smoke / scratch run: subset dumps allowed, no git checks; refused "
                            "when --out-dir is inside the repository")

    common(sub.add_parser("repro", help="Q00 vs prereg01"), questions=True, p01=True)
    s = sub.add_parser("select", help="self-retrieval selection -> frozen_config.json")
    common(s, questions=False)
    s.add_argument("--guard", default=None, help="guard_calibration.json (default: OUT_DIR/...)")
    s.add_argument("--no-cnn", action="store_true", help="registered fallback list {Q22, Q23}")
    s.add_argument("--reason", default=None, help="why --no-cnn (recorded)")
    common(sub.add_parser("apply", help="frozen combination -> player_val_top5.csv"), questions=True)
    args = ap.parse_args(argv)
    {"repro": cmd_repro, "select": cmd_select, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
