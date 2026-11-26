"""PREREG_10 (competition 1) part B: 10B = 06 (or 10A) + w* z(T_P1), the KataGo HumanSL neutral-probe (P1) LDA term
(docs/PREREG_10_PLAYER_GUARD_P1.md rev 1, sections 2.1-2.6).

    PY = .venv/Scripts/python.exe   (CPU; Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; no OMP / OPENBLAS / MKL
                                     thread variables)
    PY tools/p10_fuse.py selfcheck [--out-dir data/prereg10/tmp/<x>]   -> results/prereg10/player/selfcheck.json
    PY tools/p10_fuse.py mock      [--out-dir ... --sets S --chunk-dir D --shape SPEC --seeds-tune ... --seeds-eval ...]
                                                                       -> results/prereg10/player/mock_report.json
    PY tools/p10_fuse.py freeze    [--out-dir ...]                     -> results/prereg10/player/p1_lda.npz,
                                                                          frozen_config.json
    PY tools/p10_fuse.py apply     [--rerun --reason T]                -> results/prereg10/player/player_val_top5.csv,
                                                                          apply_report.json
    PY tools/p10_fuse.py apply --finalize   (after prescore_audit.json is committed: score_allowed only)

    (apply lives in tools/p10_apply.py since PREREG_10 rev 2; this CLI forwards to it)

Modes (as tools/p08_rank_fuse.py, without --precommit)
    real        the tools of the step and every earlier step's record committed and clean; writes
                results/prereg10/player.
    --out-dir   preview under data/prereg10/tmp/ (records say "preview": true; never accepted by a later step).
                Only a preview may take --sets / --chunk-dir / --shape / seed overrides (smoke runs on stand-in data).
apply has no preview: it needs the committed frozen_config.json, p1_lda.npz and results/prereg10/feat/val_report.json.

Tool binding (PREREG_10 rev 2): selfcheck, mock and freeze record and compare only MOCK_TOOL_FILES (this file and
the p06 / p02 / c10data modules it imports); the apply side (tools/p10_apply.py, tools/p10_p1.py, p06_player.py,
check_submission.py) can be fixed without touching the mock binding. A fix to a mock-path file after a committed
selfcheck needs a committed PREREG_10 revision N and `selfcheck --rerun --reason T --revision N`; mock then binds to
that selfcheck.json.

Steps
    selfcheck  port check (2.3): on the exploration pilot (data/prereg10/tmp/p1pilot) with the pilot's recipe (the P1
               LDA learned on PREREG_08's probe games of both colours, pilot players removed: P1ldaBW of
               results/prereg10/explore/x3_p1_eval.py), recompute x3's 'V2' and 'V2+P1ldaBW@w' (every grid w, every
               pilot scenario) with this file's functions; reference = x3_p1_eval.py itself run with its rounding
               switched off (module-level `round` replaced, nothing else). Passes when every |difference| <= 1e-9
               against that unrounded reference; equality with the committed 4-decimal x3_p1_eval.json is recorded
               only. A selfcheck whose only failure is a difference <= 1e-6 (NUMERIC_TOL: a numerical-path difference)
               may be followed by `mock --selfcheck-revision N` (a committed PREREG_10 revision N >= 2, later than
               selfcheck.json, that says why); larger differences stop. Competition 2's probe features are read only
               inside selfcheck, by PROBE_FUNCS: pilot_probe_lda (directly) and x3_reference (by running x3, which
               reads them); tests/test_p10_hygiene.py checks that no other function names or calls them.
    mock       2.2 / 2.3: the official-shape mock pools of the 926 fit players (seeds 0-39 x base / nov4 / wr / c20)
               from data/prereg10/mock/sets.json, the mock P1 chunks and the 06 fit features; per seed the P1 LDA is
               learned on the mock games of the 526 players outside the pool, the 06 LDA refit without the pool;
               w* on seeds 0-19 base. Gates G1 / G2 / G3 on seeds 20-39: point estimate = the per-question mean of
               10B(w*) - V2 pooled over the held-out pools; 95 % interval = player bootstrap (players resampled, the
               pooled per-question mean recomputed), 2,000 draws, a fresh default_rng(0) per scenario.
               Judged once: a real mock_report.json is never overwritten (a rerun needs --rerun --reason --revision N,
               N >= 2 a committed PREREG_10 revision later than the report).
               Memory: each worker holds the side-0 P1 matrix (memory-mapped) and up to ~2-3 GB while fitting the
               P1 LDA; the default --workers 2 suits this 32 GB machine with other work running.
    freeze     2.4: only when the committed mock_report.json passed. P1 LDA on all 926 players' usable mock games.
    apply      2.5 (tools/p10_apply.py): validation pool. Recomputes 06 with the 06 functions (byte-identical to
               results/prereg06/player/player_val_top5.csv, else stop), the sr06 pseudo pools with 10B (pool gate
               sr(10B) - sr(06 selected) >= 0), writes 10B's Top-5 and apply_report.json with score_allowed
               'pending_audit' (or false with reasons). apply --finalize binds results/prereg10/player/
               prescore_audit.json (another agent's audit: no labels, no scores; independent T_P1 and Top-5 equal) and
               sets score_allowed true.

Discipline
    * the p06 import-time guard (p06_style.install_guard) is on: data/local raises, except the two validation game
      files inside st.allow_local() in apply; any path containing the label-file marker raises. No torch, no p1_lib,
      no hs_lib: P1 chunks are read with numpy only. tools/score.py is never run or imported.
    * the LDA of 10B is learned only on competition 1's own mock-pool games (data/prereg10/mock); competition 2's
      feature folders (data/prereg08, data/prereg09) are read only inside selfcheck by PROBE_FUNCS.
Exit codes: 0 done; 1 the step ran and its registered outcome is a stop (selfcheck failed, gates not passed, 06 not
reproduced, ...); 2 refused or error.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p06_style as st  # noqa: E402

st.install_guard()

import numpy as np  # noqa: E402

import p06_sim as sim  # noqa: E402
from c10data import LEVELS  # noqa: E402


class Refused(RuntimeError):
    """Refused or error (exit 2)."""


class Stop(RuntimeError):
    """The step ran; its registered outcome is a stop (exit 1)."""


def path_of(r):
    return os.path.join(REPO, *r.split("/"))


# ================================================================ constants (PREREG_10 rev 1)
PREREG = path_of("docs/PREREG_10_PLAYER_GUARD_P1.md")
OUT = path_of("results/prereg10/player")
FEAT = path_of("results/prereg10/feat")
TMP = path_of("data/prereg10/tmp")
MOCK_DIR = path_of("data/prereg10/mock")
MOCK_WORK = path_of("data/prereg10/mock_work")
SETS = os.path.join(MOCK_DIR, "sets.json")
SETS_RECORD = os.path.join(FEAT, "mock_sets_record.json")
MANIFEST = os.path.join(FEAT, "mock_manifest.json")
FEATS06 = path_of("data/prereg06/feats")
CNN06 = path_of("data/prereg06/cnn_fit/black_ge105.npz")
P06_PLAYER = path_of("results/prereg06/player")
PILOT_DIR = path_of("data/prereg10/tmp/p1pilot")
X3_SCRIPT = path_of("results/prereg10/explore/x3_p1_eval.py")
X3_JSON = path_of("results/prereg10/explore/x3_p1_eval.json")
P06_KEYS6 = path_of("data/prereg10/tmp/open_keys.npz")
HS_LIB = path_of("results/prereg07/explore/pilot_humansl/hs_lib.py")
CKPT = path_of("data/katago/b18c384nbt-humanv0.ckpt")

WGRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 48.0)
TUNE_SEEDS = tuple(range(0, 20))
EVAL_SEEDS = tuple(range(20, 40))
SCENARIOS = ("base", "nov4", "wr", "c20")
W_TIE = 0.0005
G1_MIN, G2_MIN, G3_MIN = 0.003, 0.020, -0.002
BOOT_N, BOOT_SEED = 2000, 0
CSLS_K = 10
STD_EPS = 1e-6
MIN_GAMES = 2
FAIL_MAX = 0.005
SELFCHECK_TOL = 1e-9
NUMERIC_TOL = 1e-6           # a selfcheck failing only within this may continue with a revision (2.3)
THETA1, THETA2 = 0.9630063040189639, 0.8267520442154412   # results/prereg02/player/apply_report.json (raw); record only
P1_DIM = 770
N_PLAYERS = 926
N_MOCK_CHUNKS = 616
CHUNK = 200
CHUNK_FORMAT = "c10-prereg08 p1 chunk v1"
MOCK_SET = "p10mock"
PILOT_SCEN = ("c30", "c30k2", "c30nov")
PILOT_SEEDS = 40


class Shape:
    """Pool shape (2.2): per_level players per level, n_cand candidate games, c20 candidate games in c20, k values."""

    def __init__(self, per_level=sim.PER_LEVEL, n_cand=sim.N_CAND, n_c20=20, k_values=sim.K_VALUES):
        self.per_level, self.n_cand, self.n_c20, self.k_values = int(per_level), int(n_cand), int(n_c20), \
            tuple(int(k) for k in k_values)

    def as_dict(self):
        return {"per_level": self.per_level, "n_cand": self.n_cand, "n_c20": self.n_c20,
                "k_values": list(self.k_values)}

    def official(self):
        return self.as_dict() == Shape().as_dict()


def parse_shape(spec):
    """'per_level=10,n_cand=24,n_c20=8,k=2/4/6/8' -> Shape (previews only)."""
    kw = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        k = k.strip()
        if k == "k":
            kw["k_values"] = [int(x) for x in v.split("/")]
        elif k in ("per_level", "n_cand", "n_c20"):
            kw[k] = int(v)
        else:
            raise Refused(f"--shape: unknown key {k!r}")
    return Shape(**kw)


FORMATS = {"selfcheck": "c10-prereg10 player selfcheck v1", "mock": "c10-prereg10 player mock_report v1",
           "frozen": "c10-prereg10 player frozen_config v1", "lda": "c10-prereg10 p1_lda v1",
           "apply": "c10-prereg10 player apply_report v1", "audit": "c10-prereg10 player prescore_audit v1",
           "manifest": "c10-prereg10 mock_manifest v1", "val_report": "c10-prereg10 val_report v1",
           "sets_record": "c10-prereg10 mock sets record v1", "sets": "c10-prereg10 mock sets v1"}
# the mock path (selfcheck, mock, freeze) binds only these files (PREREG_10 rev 2); apply: p10_apply.APPLY_TOOL_FILES
MOCK_TOOL_FILES = ("tools/p10_fuse.py", "tools/p06_style.py", "tools/p06_sim.py", "tools/p02_player_fuse.py",
                   "tools/p02_player_open.py", "tools/p02_moves.py", "tools/c10data.py")
P06_FROZEN_FILES = ("frozen_config.json", "lda_B.npz", "lda_W.npz", "player_val_top5.csv", "apply_report.json")
P1_PINNED = ("results/prereg08/p1/p1_lib.py", "results/prereg08/p1/p1_extract.py")
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# the only functions that read competition 2's feature folders, both called only by cmd_selfcheck:
# pilot_probe_lda names them, x3_reference runs x3_p1_eval.py which reads them (tests/test_p10_hygiene.py checks)
PROBE_FUNCS = ("pilot_probe_lda", "x3_reference")


# ================================================================ small helpers
def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def rel(p):
    return st.rel(p)


def inside(p, root):
    a = os.path.normcase(os.path.abspath(p))
    r = os.path.normcase(os.path.abspath(root))
    return a == r or a.startswith(r + os.sep)


def sha_raw(path):
    return st.sha256_file(path)


def sha_lf(path):
    return st.sha256_lf(path)


def pin_sha(path):
    return sha_lf(path) if os.path.splitext(path)[1].lower() in (".json", ".csv", ".py", ".txt", ".md") \
        else sha_raw(path)


def read_json(path):
    return st.read_json(path)


def write_json(path, obj):
    if st.under_local(path):
        raise Refused("refusing to write under data/local")
    st.write_json(path, obj)


def git(args):
    r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True)
    return r.returncode, r.stdout


def tracked_clean(paths):
    """-> list of problems (untracked / modified) for the given paths."""
    probs = []
    for p in paths:
        r = rel(p)
        if git(["ls-files", "--error-unmatch", "--", r])[0] != 0:
            probs.append(f"{r} not committed")
        elif subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", r]).returncode != 0:
            probs.append(f"{r} differs from HEAD")
    return probs


def commit_time(path):
    rc, out = git(["log", "-1", "--format=%cI", "--", rel(path)])
    return (out.strip() or None) if rc == 0 else None


def require_committed(paths, what):
    missing = [rel(p) for p in paths if not os.path.exists(p)]
    if missing:
        raise Refused(f"{what}: missing {missing}")
    probs = tracked_clean(paths)
    if probs:
        raise Refused(f"{what}: {probs} (commit each step before the next)")
    return {"files": {rel(p): pin_sha(p) for p in paths}, "commit": commit_time(paths[0])}


def tool_state(files=MOCK_TOOL_FILES):
    files = list(files)
    rc, out = git(["status", "--porcelain", "--untracked-files=all", "--"] + files)
    dirty = [ln[3:] for ln in out.splitlines() if ln.strip()] if rc == 0 else None
    return {"git_head": git(["rev-parse", "HEAD"])[1].strip() or None, "dirty_or_untracked": dirty,
            "files_sha256_rule": "sha256, CRLF folded to LF",
            "files_sha256": {f: (sha_lf(path_of(f)) if os.path.exists(path_of(f)) else None) for f in files}}


def require_tools(ts):
    if ts["dirty_or_untracked"] is None:
        raise Refused("git is not available")
    if ts["dirty_or_untracked"]:
        raise Refused(f"commit the tools first (uncommitted: {ts['dirty_or_untracked']})")


def tools_differ(ts, recorded, files=MOCK_TOOL_FILES):
    rec = (recorded or {}).get("files_sha256") or {}
    return [f for f in files if rec.get(f) != ts["files_sha256"].get(f)]


def blas_env_check(preview):
    set_ = {k: os.environ[k] for k in BLAS_ENV if os.environ.get(k)}
    if set_ and not preview:
        raise Refused(f"BLAS thread variables are set ({set_}): unset them (formal runs use the default threads)")
    return set_


def revision_committed(n):
    """(commit, time) of the first commit whose PREREG_10 has a '| rev n |' row; raises when there is none."""
    rc, out = git(["log", "--format=%H %cI", "-S", f"| rev {n} |", "--", rel(PREREG)])
    lines = [x for x in out.splitlines() if x.strip()] if rc == 0 else []
    if not lines:
        raise Refused(f"PREREG_10 rev {n} is not in any commit")
    h, t = lines[-1].split()
    return h, t


class Ctx:
    pass


def resolve(a, cmd):
    """Real (results/prereg10/player) or a preview (--out-dir under data/prereg10/tmp/)."""
    c = Ctx()
    c.cmd = cmd
    od = getattr(a, "out_dir", None)
    if od:
        c.out_dir = os.path.abspath(od)
        if not inside(c.out_dir, TMP) or os.path.normcase(c.out_dir) == os.path.normcase(os.path.abspath(TMP)):
            raise Refused(f"--out-dir (preview) must be a folder under {rel(TMP)} (got {c.out_dir})")
        c.preview = True
    else:
        c.out_dir = OUT
        c.preview = False
    if st.under_local(c.out_dir):
        raise Refused("refusing to write under data/local")
    overrides = [k for k in ("sets", "chunk_dir", "shape", "seeds_tune", "seeds_eval", "pilot_dir", "x3_json")
                 if getattr(a, k, None)]
    if overrides and not c.preview:
        raise Refused(f"real runs take no overrides ({overrides}); they are for --out-dir previews only")
    return c


def guard_output(c, a, name):
    """None, or the rerun record. Real reports are never overwritten except by a registered rerun."""
    p = os.path.join(c.out_dir, name)
    if not os.path.exists(p):
        return None
    if not getattr(a, "rerun", False) or not getattr(a, "reason", None):
        raise Refused(f"{rel(p)} exists; rerun only with --rerun --reason TEXT")
    rec = {"reason": a.reason, "old_sha256": pin_sha(p)}
    if not c.preview and git(["ls-files", "--error-unmatch", "--", rel(p)])[0] == 0:
        rec["old_committed"] = True
    return rec


def prior(c, name, fmt):
    """An earlier step's record in this mode -> (record, sha, commit record)."""
    p = os.path.join(c.out_dir, name)
    if not os.path.exists(p):
        raise Refused(f"run the earlier step first: {name} missing in {rel(c.out_dir)}")
    r = read_json(p)
    if r.get("format") != fmt:
        raise Refused(f"{name}: not a {fmt} record")
    if bool(r.get("preview")) != c.preview:
        raise Refused(f"{name} is a {'preview' if r.get('preview') else 'real'} record; this run is "
                      f"{'a preview' if c.preview else 'real'} (a preview is never accepted by a later step)")
    commit = None if c.preview else require_committed([p], name)
    return r, pin_sha(p), commit


def meta(c, t0, ts, inputs=None):
    return {"preview": c.preview, "created": now_iso(), "tools": ts, "env": st.env_versions(),
            "blas_env": {k: os.environ.get(k) for k in BLAS_ENV}, "inputs": inputs or {},
            "seconds": round(time.time() - t0, 1)}


# ================================================================ P1 chunks (numpy only)
def chunk_files(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.startswith("chunk_") and f.endswith(".npz") and ".tmp" not in f)


def chunk_record(path):
    with np.load(path, allow_pickle=False) as z:
        ref = np.asarray(z["ref"], np.int64)
        failed = np.asarray(z["failed"], bool)
        counts = np.asarray(z["counts"])
        rec = {"file": os.path.basename(path), "sha256": sha_raw(path), "format": str(z["format"]),
               "set": str(z["set"]), "sides": [str(x) for x in z["sides"].tolist()],
               "precision": str(z["precision"]), "meta_sha256": str(z["meta_sha256"]),
               "ckpt_bytes": int(z["ckpt_bytes"]), "code_sha256": str(z["code_sha256"]),
               "games": int(len(ref)), "ref_first": int(ref[0]) if len(ref) else None,
               "ref_last": int(ref[-1]) if len(ref) else None, "failed": int(failed.sum()),
               "no_target_moves": int(((counts[:, 0] == 0) & ~failed).sum())}
        for k in ("sets_sha256", "keys_sha256"):
            if k in z.files:
                rec[k] = str(z[k])
    return rec


def load_side0(paths, n, want_set=None):
    """Chunks -> (X float32 [n, 770] side-0 means, usable [n], failed [n], have [n], counts0 [n]).
    usable = present, not failed, at least one target move (2.1)."""
    X = np.zeros((n, P1_DIM), np.float32)
    failed = np.zeros(n, bool)
    have = np.zeros(n, bool)
    cnt = np.zeros(n, np.int64)
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            if str(z["format"]) != CHUNK_FORMAT:
                raise Refused(f"{rel(p)}: chunk format {str(z['format'])!r}")
            if want_set is not None and str(z["set"]) != want_set:
                raise Refused(f"{rel(p)}: set {str(z['set'])!r} != {want_set!r}")
            ref = np.asarray(z["ref"], np.int64)
            if len(ref) and (ref.min() < 0 or ref.max() >= n):
                raise Refused(f"{rel(p)}: ref outside 0..{n - 1}")
            if have[ref].any():
                raise Refused(f"{rel(p)}: game refs seen in another chunk")
            X[ref] = z["means"][:, 0, :]
            failed[ref] = z["failed"]
            cnt[ref] = z["counts"][:, 0]
            have[ref] = True
    usable = have & ~failed & (cnt > 0)
    return X, usable, failed, have, cnt


# ================================================================ the P1 term (2.1)
def fit_p1_lda(X, labels, mask, min_games=MIN_GAMES):
    """LDA of 2.1 on the rows in mask: players with >= min_games such rows; standardise with their mean and sd
    (+1e-6), then p06_style.lda_fit (reg 1e-3, 64 dims). -> model dict (m, s, W, ev, mu, sd, n_games, n_players)."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    lab = np.asarray(labels)[idx]
    u, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_games
    idx, lab = idx[keep], lab[keep]
    if len(set(lab.tolist())) < 2:
        raise Refused("P1 LDA: fewer than 2 players with enough usable games")
    Xs = np.asarray(X[idx], np.float64)
    m = Xs.mean(0)
    s = Xs.std(0) + STD_EPS
    model = st.lda_fit((Xs - m) / s, lab)
    model.update({"m": m, "s": s, "n_games": int(len(idx)), "n_players": int(len(set(lab.tolist())))})
    return model


def project_p1(X, model, rows=None, block=20000):
    """Per-game 64-dim projection; rows = indices (default all); computed in row blocks (row-wise, same values)."""
    rows = np.arange(len(X)) if rows is None else np.asarray(rows, np.int64)
    out = np.zeros((len(rows), model["W"][:, :st.LDA_DIM].shape[1]))
    for s in range(0, len(rows), block):
        r = rows[s:s + block]
        out[s:s + len(r)] = st.lda_project((np.asarray(X[r], np.float64) - model["m"]) / model["s"], model)
    return out


def t_p1(Z, cand_groups, q_groups, usable=None):
    """T_P1 (2.1): group means of the projected usable games, -squared distance (p06_style.lda_term).
    -> (T [nq x nc], has_q). A question without a usable game: has False (its z row is 0). A candidate without a
    usable game: its column is the row minimum of the others (p06_style.colour_terms' fill; never in practice)."""
    if usable is not None:
        cand_groups = [[g for g in grp if usable[g]] for grp in cand_groups]
        q_groups = [[g for g in grp if usable[g]] for grp in q_groups]
    Zc, nc = st.group_means(Z, cand_groups)
    Zq, nq = st.group_means(Z, q_groups)
    T = st.lda_term(Zc, Zq)
    ch = nc > 0
    if not ch.all():
        T = np.array(T, np.float64, copy=True)
        T[:, ~ch] = T[:, ch].min(1, keepdims=True) if ch.any() else 0.0
    return T, nq > 0


def z_p1(Z, cand_groups, q_groups, usable=None):
    T, has = t_p1(Z, cand_groups, q_groups, usable)
    return st.zscore_rows(T, has)


def fuse_10b(F_sel, zT, w, k=CSLS_K):
    """C = CSLS_k(F_sel + w z(T_P1)) (2.1)."""
    return st.csls(np.asarray(F_sel, np.float64) + float(w) * zT, k)


def choose_w(tune_means, grid=WGRID, tie=W_TIE):
    """2.3: the w with the highest mean; any w within tie of it -> the largest such w."""
    best = max(tune_means[w] for w in grid)
    return max(w for w in grid if best - tune_means[w] <= tie + 1e-15)


def player_sums(diffs, owners):
    """diffs / owners: per-question arrays (concatenated over the held-out pools) -> (players sorted, per-player sum
    of the differences, per-player question count)."""
    owners = np.asarray(owners)
    u, inv = np.unique(owners, return_inverse=True)
    return u, np.bincount(inv, weights=np.asarray(diffs, np.float64)), np.bincount(inv).astype(np.float64)


def player_boot(sums, counts, n_boot=BOOT_N, seed=BOOT_SEED):
    """PREREG_10 rev 2 (M2): [pooled per-question mean, 2.5 %, 97.5 %]. Point = sum / count over all questions; each
    of n_boot draws resamples the players with replacement (rng default_rng(seed)) and recomputes the pooled
    per-question mean sum(sums[idx]) / sum(counts[idx])."""
    sums, counts = np.asarray(sums, np.float64), np.asarray(counts, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), (n_boot, len(sums)))
    m = sums[idx].sum(1) / counts[idx].sum(1)
    return [float(sums.sum() / counts.sum()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def gates(boot):
    """boot: {scenario: [mean, lo, hi]} on the held-out seeds -> gate record."""
    g1 = boot["base"][0] >= G1_MIN - 1e-15 and boot["base"][1] > 0
    g2 = boot["nov4"][0] >= G2_MIN - 1e-15 and boot["nov4"][1] > 0
    g3 = {s: boot[s][0] >= G3_MIN - 1e-15 for s in ("wr", "c20")}
    return {"G1_base": {"passed": bool(g1), "mean_diff": boot["base"][0], "ci95": boot["base"][1:],
                        "rule": "mean >= +0.003 and 95 % lower bound > 0"},
            "G2_nov4": {"passed": bool(g2), "mean_diff": boot["nov4"][0], "ci95": boot["nov4"][1:],
                        "rule": "mean >= +0.020 and 95 % lower bound > 0"},
            "G3_no_harm": {"passed": bool(all(g3.values())), "by_scenario": {s: {"passed": bool(v), "mean_diff":
                                                                                 boot[s][0], "ci95": boot[s][1:]}
                                                                             for s, v in g3.items()},
                           "rule": "wr and c20 each mean >= -0.002"},
            "passed": bool(g1 and g2 and all(g3.values()))}


# ================================================================ mock sets and pools (2.2)
class SetsIndex:
    """sets.json -> per player the candidate / base / nov4 game indices (sets order = k order), levels, store rows."""

    def __init__(self, sets):
        if sets.get("format") != FORMATS["sets"]:
            raise Refused(f"sets.json format {sets.get('format')!r}")
        self.players = list(sets["players"])
        games = sets["games"]
        self.n = len(games)
        self.pid = np.array([g["player_id"] for g in games])
        self.store_row = np.array([int(g["store_row"]) for g in games], np.int64)
        self.by = {p: {"cand": [], "base": [], "nov4": []} for p in self.players}
        self.level = {}
        last = {}
        for j, g in enumerate(games):
            p, role = g["player_id"], g["role"]
            if p not in self.by or role not in self.by[p]:
                raise Refused(f"sets.json game {j}: unknown player or role")
            if int(g["k"]) != len(self.by[p][role]) or last.get((p, role), -1) >= j:
                raise Refused(f"sets.json game {j}: k out of order")
            self.by[p][role].append(j)
            last[(p, role)] = j
            if g.get("color", "B") != "B":
                raise Refused(f"sets.json game {j}: not a black game")
            self.level.setdefault(p, str(g["rank"]))
            if self.level[p] != str(g["rank"]):
                raise Refused(f"player {p}: games of two levels")
        if len(set(self.store_row.tolist())) != self.n:
            raise Refused("sets.json: a store row appears twice")


def choose_players(ix, seed, per_level):
    """sim.choose_pool's rule on the sets players: per level sorted by sha256('{seed}|{pid}'), first per_level."""
    pool = []
    for lv in LEVELS:
        el = sorted((p for p in ix.players if ix.level[p] == lv),
                    key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        if len(el) < per_level:
            raise Refused(f"level {lv} has only {len(el)} mock players")
        pool += el[:per_level]
    return pool


def k_list(n, k_values):
    per = n // len(k_values)
    if per * len(k_values) != n:
        raise Refused(f"pool of {n} players does not split evenly over k values {k_values}")
    return [k for k in k_values for _ in range(per)]


def _hash(seed, p, tag=""):
    return hashlib.sha256(("%s%d|%s" % (tag, seed, p)).encode()).hexdigest()


def _nov4_questions(seed, pool, ks, nov):
    """x1_scenarios.py's nov rule on the nov4 reserves (lists of game indices): big questions first; a player without
    unused novel games gives the question to the pool player (hash order) with >= k (else >= 1) unused novel games."""
    n = len(pool)
    used = [0] * n
    q = [None] * n
    owner = list(range(n))
    k_eff = list(ks)
    n_replaced = 0
    for i in sorted(range(n), key=lambda i: -ks[i]):
        if len(nov[i]) - used[i] >= 1:
            take = min(ks[i], len(nov[i]) - used[i])
            q[i] = nov[i][used[i]:used[i] + take]
            used[i] += take
            k_eff[i] = take
            continue
        cands = sorted((j for j in range(n) if len(nov[j]) - used[j] >= ks[i]),
                       key=lambda j: _hash(seed, pool[j] + "|%d" % i, "repl|"))
        if not cands:
            cands = sorted((j for j in range(n) if len(nov[j]) - used[j] >= 1),
                           key=lambda j: _hash(seed, pool[j] + "|%d" % i, "repl|"))
        if not cands:
            raise Refused(f"seed {seed}: no novel games left for question {i}")
        j = cands[0]
        take = min(ks[i], len(nov[j]) - used[j])
        q[i] = nov[j][used[j]:used[j] + take]
        used[j] += take
        owner[i] = j
        k_eff[i] = take
        n_replaced += 1
    return q, owner, k_eff, n_replaced


def _wr_questions(seed, n, ks, cap):
    """p06_sim.make_pool's 'wr' owners and capacity repair (verbatim rule) with cap = the base reserve sizes."""
    from collections import Counter
    r2 = random.Random("scen|%s|%d" % ("wr", seed))
    owner = [r2.randrange(n) for _ in range(n)]
    ks = list(ks)
    n_rep = n_fallback = 0
    r3 = random.Random("repair|%s|%d" % ("wr", seed))
    for _ in range(10000):
        need = Counter()
        for q in range(n):
            need[owner[q]] += ks[q]
        bad = [i for i in need if need[i] > cap[i]]
        if not bad:
            break
        i = bad[0]
        qs = sorted([q for q in range(n) if owner[q] == i], key=lambda q: -ks[q])
        q = qs[0]
        cands = [q2 for q2 in range(n) if owner[q2] != i and ks[q2] < ks[q]
                 and need[owner[q2]] - ks[q2] + ks[q] <= cap[owner[q2]]]
        if cands:
            q2 = r3.choice(cands)
            ks[q], ks[q2] = ks[q2], ks[q]
        else:
            others = [j for j in set(owner) if j != i and need[j] + ks[q] <= cap[j]]
            if not others:      # PREREG_10 rev 2: make_pool would crash here (cap = the 20-game base reserve)
                others = [j for j in range(n) if j != i and need[j] + ks[q] <= cap[j]]
                n_fallback += 1
            if not others:
                raise Refused(f"seed {seed}: wr repair found no pool player with room for a {ks[q]}-game question")
            owner[q] = r3.choice(others)
        n_rep += 1
    else:
        raise Refused("could not repair the wr k assignment")
    return owner, ks, n_rep, n_fallback


def build_pool(ix, seed, scen, shape):
    """One mock pool (2.2) -> dict(pool, cand_idx, q_idx (sets game indices), owner, k (effective), info)."""
    pool = choose_players(ix, seed, shape.per_level)
    n = len(pool)
    ks = k_list(n, shape.k_values)
    random.Random("k|%d" % seed).shuffle(ks)
    ncand = shape.n_c20 if scen == "c20" else shape.n_cand
    cand = [ix.by[p]["cand"][:ncand] for p in pool]
    if any(len(c) < ncand for c in cand):
        raise Refused(f"seed {seed}: a player has fewer than {ncand} candidate games")
    info = {}
    if scen in ("base", "c20"):
        q = [ix.by[p]["base"][:ks[i]] for i, p in enumerate(pool)]
        if any(len(x) < k for x, k in zip(q, ks)):
            raise Refused(f"seed {seed}: base reserve shorter than k")
        owner = list(range(n))
    elif scen == "nov4":
        nov = [ix.by[p]["nov4"] for p in pool]
        q, owner, k_eff, n_rep = _nov4_questions(seed, pool, ks, nov)
        info = {"k_nominal": [int(k) for k in ks], "n_replaced_owner": n_rep, "n_distinct_owner": len(set(owner)),
                "mean_k_eff": float(np.mean(k_eff)), "short_questions": int(sum(1 for a, b in zip(k_eff, ks) if a < b)),
                "players_without_novel": int(sum(1 for x in nov if not x))}
        ks = k_eff
    elif scen == "wr":
        cap = [len(ix.by[p]["base"]) for p in pool]
        owner, ks, n_rep, n_fb = _wr_questions(seed, n, ks, cap)
        pos = [0] * n
        q = []
        for i in range(n):
            o = owner[i]
            q.append(ix.by[pool[o]]["base"][pos[o]:pos[o] + ks[i]])
            pos[o] += ks[i]
        info = {"n_repairs": n_rep, "n_fallback_owner_moves": n_fb, "n_distinct_owner": len(set(owner))}
    else:
        raise Refused(f"unknown scenario {scen!r}")
    return {"pool": pool, "cand_idx": cand, "q_idx": q, "owner": np.asarray(owner, np.int64),
            "k": np.asarray(ks, np.int64), "scenario": scen, "seed": seed, "info": info}


def to_store_rec(ix, pr):
    """The p06_sim view of a mock pool (store rows)."""
    sr = ix.store_row
    return {"pool": pr["pool"], "cand_rows": [[int(sr[j]) for j in g] for g in pr["cand_idx"]],
            "q_rows": [[int(sr[j]) for j in g] for g in pr["q_idx"]], "owner": pr["owner"], "k": pr["k"],
            "n_repairs": 0, "scenario": pr["scenario"], "seed": pr["seed"]}


# ================================================================ mock pool evaluation (worker side)
_W = {}


def _mock_init(store_dir, cnn_path, side0_path, usable_path, sets_path, shape_d, grid):
    sys.dont_write_bytecode = True
    st.install_guard()
    store = sim.Store(store_dir)
    _W["store"] = store
    _W["lda06"] = {"B": sim.LdaData(store_dir, "B", store)}
    _W["cnn"] = sim.Cnn(cnn_path, store)
    _W["X"] = np.load(side0_path, mmap_mode="r")
    _W["usable"] = np.load(usable_path)
    with open(sets_path, encoding="utf-8") as f:
        _W["ix"] = SetsIndex(json.load(f))
    _W["shape"] = Shape(**{k: v for k, v in shape_d.items()})
    _W["grid"] = tuple(grid)


def rho_raw(store, rec):
    """x1_scenarios.py's PREREG_02 rho approximation (record only)."""
    cos, has = sim.term_O(store, rec)
    num = float(np.median(cos.max(1)[has]))
    prs, _ids = sim.sr_pool(store, dict(rec, k=np.asarray(sim.k_list(len(rec["cand_rows"])))), rounds=(0, 1, 2, 3, 4))
    mx = []
    for pr in prs:
        c2, h2 = sim.term_O(store, pr)
        mx.append(c2.max(1)[h2])
    den = float(np.median(np.concatenate(mx)))
    rho = num / den
    return {"rho": rho, "level": 1 if rho >= THETA1 else (2 if rho >= THETA2 else 3)}


def eval_seed(seed, scenarios=SCENARIOS):
    """All scenarios of one seed: 06 terms (black), V2 / E0 / Q04, 10B at every grid w; per-question points."""
    t0 = time.time()
    ix, store, X, usable = _W["ix"], _W["store"], _W["X"], _W["usable"]
    shape, grid = _W["shape"], _W["grid"]
    pool = choose_players(ix, seed, shape.per_level)
    pset = set(pool)
    m06 = {"B": _W["lda06"]["B"].fit_excluding(store, pset)}
    train = usable & ~np.isin(ix.pid, np.asarray(sorted(pset)))
    mp1 = fit_p1_lda(X, ix.pid, train)
    rows = np.flatnonzero(usable)
    Z = np.zeros((ix.n, st.LDA_DIM))
    Z[rows] = project_p1(X, mp1, rows)
    out = []
    for scen in scenarios:
        pr = build_pool(ix, seed, scen, shape)
        rec = to_store_rec(ix, pr)
        T = sim.pool_terms(store, rec, _W["lda06"], m06, _W["cnn"], want=("O", "JO", "S2", "L", "E0"),
                           colours=("B",))
        v2nc = sim.combo_F(T, "E0", sim.V2_W)
        zT = z_p1(Z, pr["cand_idx"], pr["q_idx"], usable)
        own = pr["owner"]
        pts = {"E0": st.points(st.ranks(st.zscore_rows(*T["E0"]), own)),
               "Q04": st.points(st.ranks(sim.combo_F(T, "E0", None), own)),
               "V2": st.points(st.ranks(st.csls(v2nc, CSLS_K), own)),
               "P1": st.points(st.ranks(zT, own))}
        for w in grid:
            pts["10B@%g" % w] = st.points(st.ranks(fuse_10b(v2nc, zT, w), own))
        guard = None
        if shape.n_cand == sim.N_CAND and scen != "c20":
            guard = rho_raw(store, rec)
        out.append({"seed": seed, "scen": scen, "pts": {k: v.tolist() for k, v in pts.items()},
                    "owner_pid": [pr["pool"][o] for o in own.tolist()], "k": pr["k"].tolist(), "info": pr["info"],
                    "guard": guard, "lda_p1": [mp1["n_players"], mp1["n_games"]],
                    "lda06": [m06["B"]["n_players"], m06["B"]["n_games"]],
                    "q_without_usable_p1": int(sum(1 for g in pr["q_idx"] if not any(usable[j] for j in g)))})
    return {"seed": seed, "rows": out, "seconds": round(time.time() - t0, 1)}


def summarise(rows, tune_seeds, eval_seeds, grid):
    """rows (all pools) -> (w*, tune means, held-out bootstrap per scenario, record-only tables)."""
    by = {(r["seed"], r["scen"]): r for r in rows}

    def pool_mean(r, key):
        return float(np.mean(r["pts"][key]))
    tune = {w: float(np.mean([pool_mean(by[(s, "base")], "10B@%g" % w) for s in tune_seeds])) for w in grid}
    w_star = choose_w(tune, grid)
    key = "10B@%g" % w_star
    boot, q_mean = {}, {}
    for scen in SCENARIOS:
        R = [by[(s, scen)] for s in eval_seeds]
        diffs = np.concatenate([np.asarray(r["pts"][key]) - np.asarray(r["pts"]["V2"]) for r in R])
        owners = np.concatenate([np.asarray(r["owner_pid"]) for r in R])
        _u, sums, counts = player_sums(diffs, owners)
        boot[scen] = player_boot(sums, counts)
        q_mean[scen] = {"questions": int(len(diffs)), "players": int(len(sums)),
                        "per_player_mean_diff_record_only": float(np.mean(sums / counts))}
    rec_only = {}
    names = ["E0", "Q04", "V2", "P1"] + ["10B@%g" % w for w in grid]
    for scen in SCENARIOS:
        t = {}
        for part, seeds in (("tune", tune_seeds), ("eval", eval_seeds)):
            t[part] = {n: float(np.mean([pool_mean(by[(s, scen)], n) for s in seeds])) for n in names}
        rec_only[scen] = t
    return w_star, tune, boot, q_mean, rec_only


# ================================================================ inputs of mock / freeze
def verify_mock_inputs(c, a):
    """sets.json (SHA = the committed sets record), mock chunks (SHA = the committed manifest, complete). -> dict."""
    if c.preview:
        sets_path = os.path.abspath(a.sets) if a.sets else SETS
        chunk_dir = os.path.abspath(a.chunk_dir) if a.chunk_dir else MOCK_DIR
        files = chunk_files(chunk_dir)
        with open(sets_path, encoding="utf-8") as f:
            sets = json.load(f)
        return {"sets_path": sets_path, "sets": sets, "chunks": files, "sets_sha256": sha_raw(sets_path),
                "manifest": None, "chunk_dir": chunk_dir, "want_set": None}
    srec = require_committed([SETS_RECORD], "mock_sets_record.json")
    sr = read_json(SETS_RECORD)
    if sr.get("format") != FORMATS["sets_record"]:
        raise Refused("mock_sets_record.json: wrong format")
    got = sha_raw(SETS)
    if got != sr["sets_sha256"]:
        raise Refused(f"data/prereg10/mock/sets.json SHA-256 {got} != the committed record {sr['sets_sha256']}")
    mcommit = require_committed([MANIFEST], "mock_manifest.json")
    man = read_json(MANIFEST)
    if man.get("format") != FORMATS["manifest"] or man.get("set") != "mock" or not man.get("complete"):
        raise Refused("mock_manifest.json: not a complete mock manifest")
    if man.get("sets_sha256") != got:
        raise Refused("mock_manifest.json was made for another sets.json")
    files = chunk_files(MOCK_DIR)
    want = {r["file"]: r["sha256"] for r in man["chunks"]}
    if sorted(os.path.basename(f) for f in files) != sorted(want):
        raise Refused("the mock chunk files differ from mock_manifest.json")
    bad = [os.path.basename(f) for f in files if sha_raw(f) != want[os.path.basename(f)]]
    if bad:
        raise Refused(f"mock chunks changed since the manifest: {bad[:5]}")
    with open(SETS, encoding="utf-8") as f:
        sets = json.load(f)
    return {"sets_path": SETS, "sets": sets, "chunks": files, "sets_sha256": got, "manifest": man,
            "manifest_commit": mcommit, "sets_record_commit": srec, "chunk_dir": MOCK_DIR, "want_set": MOCK_SET}


# ================================================================ selfcheck (2.3)
def pilot_probe_lda(pilot_players):
    """THE ONLY READ OF COMPETITION 2'S FEATURES (selfcheck's port check, PREREG_10 2.3): the pilot recipe P1ldaBW =
    LDA on PREREG_08's P1 probe games (both colours, side 0), pilot players removed, players with >= 2 games. Uses this
    file's fit_p1_lda, so the port check covers it. Feeds no selection; nothing derived from it is saved."""
    probe_sets = path_of("data/prereg08/p1/sets.json")
    probe_dir = path_of("data/prereg08/p1/probe")
    with open(probe_sets, encoding="utf-8") as f:
        pr = json.load(f)["probe"]["games"]
    X, usable, failed, have, _cnt = load_side0(chunk_files(probe_dir), len(pr))
    pid = np.array([g["player_id"] for g in pr])
    x3_ok = have & ~failed
    keep = x3_ok & ~np.isin(pid, sorted(pilot_players))
    keep_formal = usable & ~np.isin(pid, sorted(pilot_players))
    model = fit_p1_lda(X, pid, keep_formal)
    return model, {"probe_games": len(pr), "probe_ok": int(x3_ok.sum()), "probe_usable": int(usable.sum()),
                   "x3_mask_equals_formal_mask": bool(np.array_equal(keep, keep_formal)),
                   "lda_players": model["n_players"], "lda_games": model["n_games"],
                   "probe_sets_sha256": sha_raw(probe_sets)}


def x3_reference(out_json):
    """Run results/prereg10/explore/x3_p1_eval.py (imported, bytecode off) with its module-level `round` replaced by
    the identity: the unrounded x3 numbers. Nothing in the explore folder is written."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("p10_x3_ref", X3_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.round = lambda x, n=None: x
    old = sys.argv
    sys.argv = [X3_SCRIPT, "--out", out_json]
    try:
        mod.main()
    finally:
        sys.argv = old
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def pilot_values(pilot_dir, model, seeds=PILOT_SEEDS):
    """The pilot pools of x3 (ported here, same rules) with this file's T_P1 / fusion. -> {scen: {'V2': v,
    'V2+P1ldaBW@w': v}} as x3's eval summaries (per-player means over the eval seeds, then the mean)."""
    with open(os.path.join(pilot_dir, "sets.json"), encoding="utf-8") as f:
        ps = json.load(f)
    games = ps["games"]
    X, usable, failed, have, _c = load_side0(chunk_files(pilot_dir), len(games))
    pok = have & ~failed
    Z = project_p1(X, model)
    store = sim.Store(FEATS06)
    lda06 = {"B": sim.LdaData(FEATS06, "B", store)}
    cnn = sim.Cnn(CNN06, store)
    with np.load(P06_KEYS6, allow_pickle=False) as z:
        keys6 = z["c6"]
    by_player = {}
    for j, g in enumerate(games):
        if pok[j]:
            by_player.setdefault(g["player_id"], []).append(j)
    players = sorted(p for p in ps["players"] if len(by_player.get(p, [])) >= 40)
    srow = np.array([g["store_row"] for g in games])
    models = {"B": lda06["B"].fit_excluding(store, set(players))}
    owner = np.arange(len(players))
    pts = {s: [] for s in PILOT_SCEN}
    for scen in PILOT_SCEN:
        for seed in range(seeds):
            cand, q = [], []
            for i, p in enumerate(players):
                g = list(by_player[p])
                random.Random("x3|%d|%s" % (seed, p)).shuffle(g)
                c = g[:30]
                rest = g[30:]
                k = 2 if scen == "c30k2" else (5 if i % 2 == (seed % 2) else 10)
                if scen == "c30nov":
                    ck = {keys6[srow[j]] for j in c}
                    rest = [j for j in rest if keys6[srow[j]] not in ck] or rest[:1]
                cand.append(c)
                q.append(rest[:k])
            rec = {"pool": players, "cand_rows": [[int(srow[j]) for j in c] for c in cand],
                   "q_rows": [[int(srow[j]) for j in g] for g in q]}
            T = sim.pool_terms(store, rec, lda06, models, cnn, want=("O", "JO", "S2", "L", "E0"), colours=("B",))
            v2nc = sim.combo_F(T, "E0", sim.V2_W)
            zT = z_p1(Z, cand, q, usable)
            row = {"V2": st.points(st.ranks(st.csls(v2nc, CSLS_K), owner))}
            for w in WGRID:
                row["V2+P1ldaBW@%g" % w] = st.points(st.ranks(fuse_10b(v2nc, zT, w), owner))
            pts[scen].append((seed, row))
    half = seeds // 2
    out = {}
    for scen in PILOT_SCEN:
        ev = [r for s, r in pts[scen] if s >= half]
        out[scen] = {k: float(np.mean(np.mean([r[k] for r in ev], 0))) for k in ev[0]}
    return out, {"pilot_games": len(games), "pilot_ok": int(pok.sum()), "pilot_usable": int(usable.sum()),
                 "players": len(players), "sets_sha256": sha_raw(os.path.join(pilot_dir, "sets.json"))}


def x3_numbers(x3):
    """x3 summary -> {scen: {'V2': v, 'V2+P1ldaBW@w': v}}."""
    out = {}
    for scen in PILOT_SCEN:
        s = x3["summary"][scen]
        d = {"V2": s["alone_eval"]["V2"]}
        for w, v in s["fusion"]["V2+P1ldaBW"]["grid_eval"].items():
            d["V2+P1ldaBW@%g" % float(w)] = v
        out[scen] = d
    return out


def compare_selfcheck(formal, ref_unrounded, ref_committed, tol=SELFCHECK_TOL, numeric_tol=NUMERIC_TOL):
    """-> (passed, only_numeric_path, rows). Passed: every |formal - unrounded x3| <= tol (and every value present).
    only_numeric_path: not passed, every value present, every difference <= numeric_tol. The 4-decimal equality
    with the committed x3_p1_eval.json is recorded only."""
    rows, ok, complete = [], True, True
    for scen in PILOT_SCEN:
        for k, u in ref_unrounded[scen].items():
            v = formal.get(scen, {}).get(k)
            cm = ref_committed[scen].get(k)
            if v is None:
                complete = ok = False
                rows.append({"scenario": scen, "name": k, "formal": None, "x3_unrounded": u, "abs_diff": None})
                continue
            d = abs(v - u)
            ok &= d <= tol
            rows.append({"scenario": scen, "name": k, "formal": v, "x3_unrounded": u, "abs_diff": d,
                         "x3_committed_4dp": cm, "formal_4dp_equal_record_only": round(v, 4) == cm})
    diffs = [r["abs_diff"] for r in rows if r["abs_diff"] is not None]
    numeric_only = (not ok) and complete and bool(diffs) and max(diffs) <= numeric_tol
    return bool(ok), bool(numeric_only), rows


def later_revision(n, after_iso, what):
    """A committed PREREG_10 revision N >= 2 whose commit is later than after_iso -> record; raises otherwise."""
    if not n or n < 2:
        raise Refused(f"{what}: needs --revision N >= 2 (a committed PREREG_10 revision that says why)")
    h, t = revision_committed(n)
    if not after_iso or datetime.datetime.fromisoformat(t) <= datetime.datetime.fromisoformat(after_iso):
        raise Refused(f"PREREG_10 rev {n} ({t}) is not later than {what} ({after_iso})")
    return {"rev": n, "commit": h, "commit_time": t, "later_than": after_iso}


def selfcheck_accept(c, sc, revision):
    """2.3: mock / freeze go on after a passed selfcheck, or after one that failed only on numerical-path
    differences (<= NUMERIC_TOL) when a committed PREREG_10 revision N, later than selfcheck.json, records why.
    -> None (passed) or the override record; raises otherwise."""
    if sc.get("passed") is True:
        if revision:
            raise Refused("--selfcheck-revision is only for a selfcheck that did not pass")
        return None
    if not sc.get("only_numeric_path"):
        raise Refused(f"selfcheck.json did not pass (max |diff| {sc.get('max_abs_diff')}): fix the program first")
    if c.preview:
        return {"revision": revision, "preview": True}
    return later_revision(revision, commit_time(os.path.join(c.out_dir, "selfcheck.json")),
                          "the committed selfcheck.json (numerical-path override)")


def cmd_selfcheck(a):
    t0 = time.time()
    c = resolve(a, "selfcheck")
    blas_env_check(c.preview)
    ts = tool_state()
    if not c.preview:
        require_tools(ts)
        require_committed([X3_SCRIPT, X3_JSON], "exploration x3 files")
    rerun = guard_output(c, a, "selfcheck.json")
    if rerun and not c.preview:
        old = os.path.join(c.out_dir, "selfcheck.json")
        rerun["old_commit"] = commit_time(old)
        rerun["revision"] = later_revision(getattr(a, "revision", None),
                                           rerun["old_commit"] or read_json(old).get("created"),
                                           "the selfcheck.json it replaces")
    pilot_dir = os.path.abspath(a.pilot_dir) if getattr(a, "pilot_dir", None) else PILOT_DIR
    with open(os.path.join(pilot_dir, "sets.json"), encoding="utf-8") as f:
        pilot_players = set(json.load(f)["players"])
    model, probe_rec = pilot_probe_lda(pilot_players)
    formal, pilot_rec = pilot_values(pilot_dir, model)
    os.makedirs(c.out_dir, exist_ok=True)
    unr_path = os.path.join(TMP, "selfcheck_x3_unrounded.json") if not c.preview else \
        os.path.join(c.out_dir, "selfcheck_x3_unrounded.json")
    x3u = x3_numbers(x3_reference(unr_path))
    x3c = x3_numbers(read_json(getattr(a, "x3_json", None) or X3_JSON))
    passed, numeric_only, rows = compare_selfcheck(formal, x3u, x3c)
    diffs = [r["abs_diff"] for r in rows if r["abs_diff"] is not None]
    rep = {"format": FORMATS["selfcheck"], "what": "PREREG_10 2.3 selfcheck: formal T_P1 / 10B fusion vs the x3 pilot "
           "(pilot recipe: LDA on the PREREG_08 probe, port check only, feeds no selection)",
           "passed": passed, "only_numeric_path": numeric_only, "tolerance": SELFCHECK_TOL,
           "numeric_path_tolerance": NUMERIC_TOL, "max_abs_diff": max(diffs) if diffs else None,
           "four_decimal_equal_record_only": all(r.get("formal_4dp_equal_record_only") for r in rows), "rows": rows, "probe": probe_rec, "pilot": pilot_rec,
           "x3_unrounded_json": {"path": rel(unr_path), "sha256": sha_raw(unr_path)},
           "x3_committed_json_sha256": sha_lf(X3_JSON), "x3_script_sha256": sha_lf(X3_SCRIPT), "rerun": rerun}
    rep.update(meta(c, t0, ts))
    write_json(os.path.join(c.out_dir, "selfcheck.json"), rep)
    print(f"selfcheck: passed {passed}, max |diff| {rep['max_abs_diff']}"
          f"{' (PREVIEW)' if c.preview else '; commit selfcheck.json before the mock extraction'}")
    if not passed:
        raise Stop("selfcheck did not pass: stop and find out why (PREREG_10 2.3)"
                   + ("; only numerical-path differences: commit selfcheck.json and a PREREG_10 revision that says "
                      "why, then mock --selfcheck-revision N" if numeric_only else ""))
    return 0


# ================================================================ mock (2.2-2.3)
def parse_seeds(s, default):
    if not s:
        return tuple(default)
    if "-" in s:
        x, y = s.split("-")
        return tuple(range(int(x), int(y) + 1))
    return tuple(int(x) for x in s.split(","))


def cmd_mock(a):
    t0 = time.time()
    c = resolve(a, "mock")
    blas_env_check(c.preview)
    ts = tool_state()
    rerun = guard_output(c, a, "mock_report.json")
    binding = {}
    if not c.preview:
        require_tools(ts)
        sc, sc_sha, sc_commit = prior(c, "selfcheck.json", FORMATS["selfcheck"])
        override = selfcheck_accept(c, sc, getattr(a, "selfcheck_revision", None))
        diff = tools_differ(ts, sc.get("tools"))
        if diff:
            raise Refused(f"mock-path tools changed since selfcheck.json: {diff} (a fix needs a PREREG_10 revision and "
                          f"selfcheck --rerun --reason T --revision N first)")
        binding["selfcheck"] = {"sha256": sc_sha, "commit": sc_commit, "override": override}
        binding["prereg_rev2"] = revision_committed(2)
        if rerun:
            old = os.path.join(c.out_dir, "mock_report.json")
            rerun["old_commit"] = commit_time(old)
            rerun["revision"] = later_revision(a.revision, rerun["old_commit"] or read_json(old).get("created"),
                                               "the mock_report.json it replaces (mock is judged once)")
    shape = parse_shape(a.shape) if a.shape else Shape()
    tune_seeds = parse_seeds(a.seeds_tune, TUNE_SEEDS)
    eval_seeds = parse_seeds(a.seeds_eval, EVAL_SEEDS)
    inp = verify_mock_inputs(c, a)
    ix = SetsIndex(inp["sets"])
    if not c.preview:
        if len(ix.players) != N_PLAYERS or len(inp["chunks"]) != N_MOCK_CHUNKS:
            raise Refused(f"{len(ix.players)} players / {len(inp['chunks'])} chunks, registered "
                          f"{N_PLAYERS} / {N_MOCK_CHUNKS}")
        store = sim.Store(FEATS06)
        for s in tune_seeds + eval_seeds:
            if choose_players(ix, s, shape.per_level) != sim.choose_pool(store.black_cache(), s):
                raise Refused(f"seed {s}: the pool differs from p06_sim.choose_pool")
        del store
    X, usable, failed, have, cnt = load_side0(inp["chunks"], ix.n, inp["want_set"])
    if not have.all():
        raise Refused(f"{int((~have).sum())} mock games have no chunk")
    work = c.out_dir if c.preview else MOCK_WORK
    os.makedirs(work, exist_ok=True)
    side0 = os.path.join(work, "p1_side0.npy")
    upath = os.path.join(work, "p1_usable.npy")
    np.save(side0, X)
    np.save(upath, usable)
    del X
    seeds = sorted(set(tune_seeds) | set(eval_seeds))
    init = (FEATS06, CNN06, side0, upath, inp["sets_path"], shape.as_dict(), WGRID)
    rows, secs = [], []
    if a.workers <= 1:
        _mock_init(*init)
        for s in seeds:
            r = eval_seed(s)
            rows += r["rows"]
            secs.append(r["seconds"])
            print(f"seed {s} done ({r['seconds']} s)", flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(a.workers, 6), initializer=_mock_init, initargs=init) as pool:
            for r in pool.imap_unordered(eval_seed, seeds):
                rows += r["rows"]
                secs.append(r["seconds"])
                print(f"seed {r['seed']} done ({r['seconds']} s), {len(secs)}/{len(seeds)}", flush=True)
    rows.sort(key=lambda r: (r["seed"], SCENARIOS.index(r["scen"])))
    w_star, tune, boot, q_mean, rec_only = summarise(rows, tune_seeds, eval_seeds, WGRID)
    g = gates(boot)
    pts_path = os.path.join(work, "mock_points.json")
    with open(pts_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"rows": rows}, f)
    nov = [r["info"] for r in rows if r["scen"] == "nov4"]
    rep = {"format": FORMATS["mock"], "what": "PREREG_10 2.2-2.3 mock pools: w* on the tune seeds' base pools, gates "
           "on the held-out seeds (10B(w*) - V2; player bootstrap)",
           "judged_once": not c.preview, "passed": g["passed"], "w_star": w_star, "w_grid": list(WGRID),
           "tune_seeds": list(tune_seeds), "eval_seeds": list(eval_seeds), "shape": shape.as_dict(),
           "official_shape": shape.official(),
           "tune_base_means": {"%g" % w: v for w, v in tune.items()}, "gates": g,
           "bootstrap": {"draws": BOOT_N, "rng": "numpy default_rng(%d), one fresh generator per scenario" % BOOT_SEED,
                         "point": "per-question mean difference pooled over the held-out pools",
                         "interval": "players resampled with replacement; pooled per-question mean recomputed"},
           "held_out_counts": q_mean,
           "record_only": {"scores": rec_only,
                           "k_eff_nov4": {"mean": float(np.mean([x["mean_k_eff"] for x in nov])) if nov else None,
                                          "short_questions_per_pool": [x["short_questions"] for x in nov],
                                          "replaced_owner_per_pool": [x["n_replaced_owner"] for x in nov]},
                           "rho_by_pool": [{"seed": r["seed"], "scen": r["scen"], **(r["guard"] or {})}
                                           for r in rows if r["guard"]],
                           "lda_p1_train": {str(r["seed"]): r["lda_p1"] for r in rows if r["scen"] == "base"},
                           "lda06_train": {str(r["seed"]): r["lda06"] for r in rows if r["scen"] == "base"},
                           "questions_without_usable_p1": int(sum(r["q_without_usable_p1"] for r in rows))},
           "p1_games": {"total": int(ix.n), "failed": int(failed.sum()), "no_target_moves":
                        int(((cnt == 0) & ~failed).sum()), "usable": int(usable.sum())},
           "inputs_bound": {"sets_sha256": inp["sets_sha256"], "chunk_dir": rel(inp["chunk_dir"]),
                            "chunks": len(inp["chunks"]),
                            "manifest_sha256": sha_lf(MANIFEST) if not c.preview else None,
                            "side0_npy_sha256": sha_raw(side0), "points_json": rel(pts_path),
                            "points_json_sha256": sha_raw(pts_path), **binding},
           "seconds_per_seed": secs, "rerun": rerun}
    rep.update(meta(c, t0, ts))
    write_json(os.path.join(c.out_dir, "mock_report.json"), rep)
    print(json.dumps({"w_star": w_star, "gates": {k: (v["passed"] if isinstance(v, dict) else v)
                                                  for k, v in g.items()}, "boot": boot}, indent=1))
    if not g["passed"]:
        print("B not adopted: write results/prereg10/RESULTS.md (B not adopted) and commit" if not c.preview
              else "(PREVIEW) gates not passed")
        return 1
    print("gates passed" + (" (PREVIEW)" if c.preview else "; commit mock_report.json, then freeze"))
    return 0


# ================================================================ freeze (2.4)
def save_lda(path, model, rec):
    np.savez(path, format=np.array(FORMATS["lda"]), m=model["m"], s=model["s"], W=model["W"], ev=model["ev"],
             mu=model["mu"], sd=model["sd"], n_games=np.array(model["n_games"]),
             n_players=np.array(model["n_players"]), record=np.array(json.dumps(rec, sort_keys=True)))


def load_lda(path):
    with np.load(path, allow_pickle=False) as z:
        if str(z["format"]) != FORMATS["lda"]:
            raise Refused(f"{rel(path)}: not a {FORMATS['lda']} file")
        return {k: np.asarray(z[k]) for k in ("m", "s", "W", "ev", "mu", "sd")} | \
            {"n_games": int(z["n_games"]), "n_players": int(z["n_players"])}


def katago_commit():
    with open(HS_LIB, encoding="utf-8") as f:
        m = re.search(r'KATAGO_COMMIT\s*=\s*"([0-9a-f]+)"', f.read())
    return m.group(1) if m else None


def cmd_freeze(a):
    t0 = time.time()
    c = resolve(a, "freeze")
    blas_env_check(c.preview)
    ts = tool_state()
    for n in ("frozen_config.json", "p1_lda.npz"):
        if os.path.exists(os.path.join(c.out_dir, n)):
            raise Refused(f"{n} exists in {rel(c.out_dir)}; the freeze is written once")
    mk, mk_sha, mk_commit = prior(c, "mock_report.json", FORMATS["mock"])
    if not mk.get("passed"):
        raise Stop("mock_report.json did not pass the gates: B is not adopted (no freeze)")
    if not c.preview:
        require_tools(ts)
        diff = tools_differ(ts, mk.get("tools"))
        if diff:
            raise Refused(f"mock-path tools changed since mock_report.json: {diff}")
        if not mk.get("official_shape"):
            raise Refused("mock_report.json was not made with the registered shape")
    inp = verify_mock_inputs(c, a)
    if inp["sets_sha256"] != mk["inputs_bound"]["sets_sha256"]:
        raise Refused("sets.json differs from the one mock used")
    ix = SetsIndex(inp["sets"])
    X, usable, failed, have, cnt = load_side0(inp["chunks"], ix.n, inp["want_set"])
    if not have.all():
        raise Refused("mock games without a chunk")
    model = fit_p1_lda(X, ix.pid, usable)
    os.makedirs(c.out_dir, exist_ok=True)
    lda_path = os.path.join(c.out_dir, "p1_lda.npz")
    lrec = {"source": "competition 1 mock-pool black games (cand, base, nov4) of all %d mock players" % len(ix.players),
            "usable_rule": "chunk present, not failed, >= 1 target move; players with >= 2 such games",
            "standardise": "mean, sd + 1e-6 of these games", "lda": "p06_style.lda_fit reg 1e-3, 64 dims"}
    save_lda(lda_path, model, lrec)
    p06 = {f: pin_sha(os.path.join(P06_PLAYER, f)) for f in P06_FROZEN_FILES}
    man = inp["manifest"] or {}
    fc = {"format": FORMATS["frozen"], "what": "PREREG_10 2.4 freeze of B (10B = 06 + w* z(T_P1))",
          "w_star": mk["w_star"], "csls_k": CSLS_K, "gates": mk["gates"], "mock_report_sha256": mk_sha,
          "mock_report_commit": mk_commit, "lda": {"file": rel(lda_path), "sha256": sha_raw(lda_path),
                                                   "n_games": model["n_games"], "n_players": model["n_players"],
                                                   **lrec},
          "p06_frozen_sha256": p06,
          "p1_code_sha256": {r: sha_lf(path_of(r)) for r in P1_PINNED},
          "katago_commit": katago_commit(),
          "checkpoint": {"path": rel(CKPT), "bytes": os.path.getsize(CKPT) if os.path.exists(CKPT) else None,
                         "sha256": sha_raw(CKPT) if os.path.exists(CKPT) and not c.preview else None},
          "neutral_metadata_sha256": sorted({r["meta_sha256"] for r in man.get("chunks", [])}) or None,
          "chunk_code_sha256": sorted({r["code_sha256"] for r in man.get("chunks", [])}) or None,
          "sets_sha256": inp["sets_sha256"], "fail_max": FAIL_MAX,
          "p1_games": {"total": int(ix.n), "failed": int(failed.sum()), "usable": int(usable.sum())},
          "b_passed": True, "score_pending_val": True}
    fc.update(meta(c, t0, ts))
    write_json(os.path.join(c.out_dir, "frozen_config.json"), fc)
    print(f"freeze: w* {mk['w_star']}, LDA {model['n_players']} players / {model['n_games']} games"
          f"{' (PREVIEW)' if c.preview else '; commit frozen_config.json + p1_lda.npz before extract --set val'}")
    return 0


# ================================================================ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selfcheck")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--pilot-dir", default=None)
    p.add_argument("--x3-json", default=None)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    p.add_argument("--revision", type=int, default=None)
    p = sub.add_parser("mock")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--sets", default=None)
    p.add_argument("--chunk-dir", default=None)
    p.add_argument("--shape", default=None)
    p.add_argument("--seeds-tune", default=None)
    p.add_argument("--seeds-eval", default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    p.add_argument("--revision", type=int, default=None)
    p.add_argument("--selfcheck-revision", type=int, default=None)
    p = sub.add_parser("freeze")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--sets", default=None)
    p.add_argument("--chunk-dir", default=None)
    p = sub.add_parser("apply")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    if getattr(a, "workers", 1) > 6:
        print("ERROR: at most 6 workers", file=sys.stderr)
        return 2
    if a.cmd == "apply":                       # the validation side lives in tools/p10_apply.py (rev 2)
        import p10_apply
        return p10_apply.run(p10_apply.cmd_apply, a)
    fn = {"selfcheck": cmd_selfcheck, "mock": cmd_mock, "freeze": cmd_freeze}[a.cmd]
    try:
        return fn(a)
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 1
    except (Refused, st.Refused) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
