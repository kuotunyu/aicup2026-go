"""PREREG_02 competition 2: per-game fusion and calibration on inner (PREREG_02 1.2, plan.md 4.2).

    python tools/p02_rank_fuse.py repro [--p01-dir results/prereg01/rank] [--p02-dir results/prereg02/rank]
                                        [--out-dir results/prereg02/rank] [--smoke]
    python tools/p02_rank_fuse.py tune  [--p02-dir ...] [--out-dir ...] [--smoke]
    python tools/p02_rank_fuse.py apply [--p02-dir ...] [--out-dir ...]
                                        [--questions data/local/rank_val_questions.csv] [--smoke]
    then: python tools/decide.py <out-dir>/rank_val_probs.csv <out-dir>

--p02-dir is the notebook's rank result dir (dumps/, ckpt/history.csv, env.json);
--p01-dir is prereg01's rank result dir (rank_val_probs.csv, pred_B.csv, env.json);
--out-dir holds the local files (pseudo_inner.json, opening_*.npz from the other two
tools; repro_local.json, tune_report.json, frozen_config.json, rank_val_probs.csv,
apply_report.json from this one). No validation label file is ever read.

repro  R00 (prereg01 epoch 5, original orientation, mean of the per-game softmax) on the
       validation dumps against prereg01's rank_val_probs.csv: largest difference
       <= 1e-4 and at most 4 questions whose rule-B level differs (PREREG_02 1.2 and
       plan.md 4.2 'repro'; --smoke: 4 x questions / 1589, rounded down, as the notebook).
       GPU rule (PREREG_02 revision, the same as the notebook's check 1.1 point 2, which is
       the same computation on the same dump): 1e-4 relaxes to 1e-3 when the dump was made on
       a GPU model other than prereg01's. The dump's GPU is the gpu_name in its completion
       marker dumps/_DONE_val_p01e5_d4.json (else repro_check.json same_gpu_as_p01); prereg01's
       is the last run in <p01-dir>/env.json; unknown -> 1e-4. Seam check: the local largest
       difference must equal the notebook's (repro_check.json val_c0_vs_p01.max_abs_diff)
       within 1e-6 (required without --smoke).
       Without --smoke the notebook run must also have ended cleanly: its repro_check.json
       passed and the last p01_readonly_after.json entry says prereg01 was unchanged.
       Writes repro_local.json; exit code 1 when it fails. tune refuses to run until
       it has passed.
tune   Reads inner files only (dumps/inner_*.npz, opening_inner.npz, pseudo_inner.json; of
       the validation dumps it only checks that val_p02e9_d4.npz exists). Needs a passed
       repro_local.json; without --smoke also committed tools, the notebook results committed
       and unchanged (dumps/ and the run records, PREREG_02 section 6 step 4), all 73,556 inner
       games, the registered thresholds in pseudo_inner.json, the p01e5 inner accuracy
       25.1482 +- 0.05 % (key-alignment check) and no frozen_config.json already tracked by git.
       Every combination R00..R20 (fixed order), 5-fold OOF on the eval rounds, selection,
       gate, refit on all fit rounds. Writes tune_report.json and frozen_config.json.
apply  Refuses unless frozen_config.json is committed and identical to HEAD, the notebook
       results are committed and unchanged, and the tool files are the ones tune recorded
       (except --smoke, whose output dir must be outside the repository). Reads the validation
       dumps, opening_val.npz and the question file; writes rank_val_probs.csv (prereg01 column
       layout, probabilities as repr, blank when n_parsed = 0) and apply_report.json (with the
       tune gate copied in and score_allowed = gate passed).
--smoke (notebook SMOKE dumps or synthetic data only): output outside the repository;
       <p02-dir>/config.json must say "smoke": true and the dumps must be smaller than the real
       ones (without --smoke, SMOKE dumps are refused); fewer games allowed, combinations whose
       dumps are missing are dropped, a fold without fit questions keeps the uncalibrated
       parameters (recorded as fit_empty_smoke).

Per-game sources (log-probabilities l_g over the 10 levels, LEVELS order):
    C1  p01e5, original orientation            log_softmax(z_s0)
    C2  p01e5, 8 symmetries                    log_softmax(mean_s log_softmax(z_s))
    C3  p01e3, p01e4, p01e5 (original)         log_softmax(mean_e log_softmax(z_e))
    C4  p02e9, original orientation
    C5  p02e9, 8 symmetries                    as C2
Per question (n = games the CNN parsed):
    prob pooling  m = log(mean_g exp l_g);   log pooling  m = mean_g l_g
    m_op = mean of the opening log_softmax over the question's games that have one (0 if none)
    p = softmax( e^a (n/12)^gamma (m + w m_op + b) ),  b = B beta,  B the fixed 10x9
        orthonormal (Helmert) basis of the sum-zero vectors, so sum(b) = 0.
Combinations (order = tie priority): R00 = C1 prob, no calibration (prereg01 B);
    R01..R10 = C1, C4, C3, C2, C5 x (prob, log) with (a, gamma, b);
    R11..R20 = R01..R10 plus the opening term (w).
    C4/C5 combinations (R03 R04 R09 R10 R13 R14 R19 R20) are dropped when the p02e9 dumps
    are missing or its inner per-game accuracy is < 24.15 % (accuracy of the dump's original
    orientation, the number tune uses; history.csv's val_acc is recorded only).
Fit: weighted NLL, weight 1/(L * questions of that level) (L = levels present, 10 in the
    real run) + 0.01 ||b||^2; L-BFGS-B, a in [-4, 4], gamma in [-1, 2], w in [0, 4], starts
    a = 0 and a = 1 (all else 0), maxiter 500, lower loss kept.
Selection: highest OOF rank-balanced mean (rule B); any combination within 0.003 of the best
    -> the earliest in the list. Gate (recorded): selected - R00 >= 0.010.
Decisions use rule B; rule_b_np() is tested equal to tools/decide.py rule_b.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy import optimize, sparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import decide  # noqa: E402  (reference rule B and the validated probability reader)
from c10data import EXPECTED, LEVELS, players_for, read_split  # noqa: E402
from p02_rank_opening import load_opening  # noqa: E402
from p02_rank_pseudo import (DEFAULTS, FIT_MIN_GAMES, MIN_GAMES, P02Error, check_dump_kind,  # noqa: E402
                             check_out_dir, check_smoke_size, is_inside, load_inner_keys, load_npz,
                             load_pseudo, now_iso, refuse_label_file, sha256_file, str_array,
                             write_json)

N_LEVELS = 10
E1 = math.exp(-1)
PROB_COLUMNS = ["question_id", "n_games", "n_parsed"] + ["p_" + lv for lv in LEVELS]
DUMP_S = {"p01e1": 1, "p01e2": 1, "p01e3": 1, "p01e4": 1, "p01e5_d4": 8, "p02e9_d4": 8}
SOURCES = {  # name -> list of (dump, which orientations)
    "C1": [("p01e5_d4", "s0")],
    "C2": [("p01e5_d4", "all")],
    "C3": [("p01e3", "s0"), ("p01e4", "s0"), ("p01e5_d4", "s0")],
    "C4": [("p02e9_d4", "s0")],
    "C5": [("p02e9_d4", "all")],
}
SOURCE_TEXT = {
    "C1": "prereg01 epoch 5, original orientation: log_softmax(z_s0)",
    "C2": "prereg01 epoch 5, 8 symmetries: log_softmax(mean_s log_softmax(z_s))",
    "C3": "prereg01 epochs 3, 4, 5 (original orientation): log_softmax(mean_e log_softmax(z_e))",
    "C4": "prereg02 epoch 9, original orientation: log_softmax(z_s0)",
    "C5": "prereg02 epoch 9, 8 symmetries: log_softmax(mean_s log_softmax(z_s))",
}
P02_SOURCES = ("C4", "C5")
P02E9_MIN_ACC = 24.15  # percent (prereg01 epoch 5: 25.15 - 1 point)
P01E5_ACC, P01E5_ACC_TOL = 25.1482, 0.05  # PREREG_02 1.1 check 1; repeated locally as a key-alignment check
PENALTY = 0.01
BOUNDS = {"a": (-4.0, 4.0), "gamma": (-1.0, 2.0), "w": (0.0, 4.0)}
STARTS = (0.0, 1.0)  # a at the two starts; gamma, w, beta start at 0
MAXITER = 500
SELECT_TOL = 0.003
GATE = 0.010
FLOAT_EPS = 1e-12  # tolerance and gate are compared with this slack for float rounding
N_BINS = ((5, 7), (8, 12), (13, 20))
N_BOOT = 1000
BOOT_SEED = 0
REPRO_TOL, REPRO_TOL_GPU = 1e-4, 1e-3
REPRO_MAX_B_DIFF = 4
SEAM_TOL = 1e-6  # local R00 largest difference vs the notebook's C0 largest difference (same dump)
VAL_QUESTIONS_FULL, VAL_GAMES_FULL = 1589, 19156  # the real local validation question file
# notebook result files (besides dumps/) that must be committed before tune / apply (PREREG_02 6.4)
NOTEBOOK_FILES = ("repro_check.json", "run_log.json", "config.json", "env.json", "timings.json",
                  "leak_check.json", "p01_readonly_before.json", "p01_readonly_after.json",
                  "ckpt/history.csv")
NOTEBOOK_REQUIRED = ("repro_check.json", "run_log.json", "config.json")
TOOL_FILES = ("p02_rank_fuse.py", "p02_rank_pseudo.py", "p02_rank_opening.py", "p02_moves.py",
              "c10data.py", "decide.py")
FROZEN_FORMAT = "c10-prereg02 rank frozen_config v1"


def combos():
    out = [{"id": "R00", "source": "C1", "pool": "prob", "calibrated": False, "opening": False}]
    pairs = [(s, p) for s in ("C1", "C4", "C3", "C2", "C5") for p in ("prob", "log")]
    for i, (s, p) in enumerate(pairs, 1):
        out.append({"id": "R%02d" % i, "source": s, "pool": p, "calibrated": True, "opening": False})
    for i, (s, p) in enumerate(pairs, 11):
        out.append({"id": "R%02d" % i, "source": s, "pool": p, "calibrated": True, "opening": True})
    return out


COMBOS = combos()


# ================================================================ numerics
def basis():
    """Fixed 10x9 orthonormal basis of the sum-zero vectors (normalised Helmert contrasts)."""
    B = np.zeros((N_LEVELS, N_LEVELS - 1))
    for j in range(1, N_LEVELS):
        B[:j, j - 1] = 1.0
        B[j, j - 1] = -float(j)
        B[:, j - 1] /= math.sqrt(j * (j + 1))
    return B


def log_softmax(x):
    mx = np.max(x, axis=-1, keepdims=True)
    z = x - mx
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def softmax(x):
    return np.exp(log_softmax(x))


def rule_b_np(P):
    """Row-wise rule B, identical to decide.rule_b: argmax p_k + e^-1 (p_{k-1} + p_{k+1}),
    neighbours past either end count as 0, ties to the smallest index."""
    P = np.asarray(P, dtype=np.float64)
    left = np.zeros_like(P)
    right = np.zeros_like(P)
    left[:, 1:] = P[:, :-1]
    right[:, :-1] = P[:, 1:]
    return np.argmax(P + E1 * (left + right), axis=1)


def official_score(pred, y):
    d = np.abs(np.asarray(pred) - np.asarray(y))
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def source_logprobs(source, dumps):
    """Per-game log-probabilities [N,10] (float64) of a source from {name: logits [N,S,10]}."""
    parts = []
    for name, which in SOURCES[source]:
        z = dumps[name].astype(np.float64)
        if which == "s0":
            parts.append(log_softmax(z[:, 0, :]))
        else:
            parts.extend(log_softmax(z[:, s, :]) for s in range(z.shape[1]))
    if len(parts) == 1:
        return parts[0]
    return log_softmax(np.mean(parts, axis=0))


def pool_matrix(groups, n_cols):
    """Sparse [Q, n_cols] averaging matrix; groups[q] = column indices (row of zeros if empty)."""
    rows, cols, vals = [], [], []
    for q, g in enumerate(groups):
        if len(g):
            rows.extend([q] * len(g))
            cols.extend(g)
            vals.extend([1.0 / len(g)] * len(g))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), n_cols))


def pooled(S, lp, pool):
    if pool == "log":
        return np.asarray(S @ lp)
    if pool == "prob":
        return np.log(np.maximum(np.asarray(S @ np.exp(lp)), 1e-300))
    raise ValueError(pool)


class QSet:
    """Question-level inputs of one set of questions (fit or eval, inner or val)."""

    def __init__(self, n, y=None, player=None, fold=None, mop=None):
        self.n = np.asarray(n, dtype=np.float64)
        self.y = None if y is None else np.asarray(y, dtype=np.int64)
        self.player = None if player is None else np.asarray(player, dtype=np.int64)
        self.fold = None if fold is None else np.asarray(fold, dtype=np.int64)
        self.mop = mop
        self.m = {}  # (source, pool) -> [Q,10]

    def subset(self, mask):
        out = QSet(self.n[mask], None if self.y is None else self.y[mask],
                   None if self.player is None else self.player[mask],
                   None if self.fold is None else self.fold[mask],
                   None if self.mop is None else self.mop[mask])
        out.m = {k: v[mask] for k, v in self.m.items()}
        return out

    def __len__(self):
        return len(self.n)


def unpack(theta, opening):
    a, g = theta[0], theta[1]
    if opening:
        return a, g, theta[2], np.asarray(theta[3:])
    return a, g, 0.0, np.asarray(theta[2:])


def question_logits(theta, m, mop, n, opening, B):
    a, g, w, beta = unpack(theta, opening)
    t = math.exp(a) * np.power(n / 12.0, g)
    u = m + (w * mop if opening else 0.0) + B @ beta
    return t[:, None] * u, t


def level_weights(y):
    present = np.unique(y)
    cnt = np.bincount(y, minlength=N_LEVELS).astype(np.float64)
    return 1.0 / (len(present) * cnt[y])


def objective(theta, m, mop, n, y, wq, opening, B):
    """Weighted NLL + PENALTY ||beta||^2 and its gradient."""
    z, t = question_logits(theta, m, mop, n, opening, B)
    ls = log_softmax(z)
    Q = len(y)
    nll = -np.sum(wq * ls[np.arange(Q), y])
    a, g, w, beta = unpack(theta, opening)
    loss = nll + PENALTY * float(beta @ beta)
    G = np.exp(ls)
    G[np.arange(Q), y] -= 1.0
    G *= wq[:, None]
    gz = np.sum(G * z, axis=1)
    grad = [np.sum(gz), np.sum(gz * np.log(n / 12.0))]
    if opening:
        grad.append(np.sum(t * np.sum(G * mop, axis=1)))
    gb = (t[:, None] * G).sum(0)
    grad = np.concatenate([grad, B.T @ gb + 2 * PENALTY * beta])
    return loss, grad


def n_params(opening):
    return 2 + (1 if opening else 0) + (N_LEVELS - 1)


def start(a, opening):
    x = np.zeros(n_params(opening))
    x[0] = a
    return x


def bounds(opening):
    b = [BOUNDS["a"], BOUNDS["gamma"]] + ([BOUNDS["w"]] if opening else [])
    return b + [(None, None)] * (N_LEVELS - 1)


def fit(X, source, pool, opening, B):
    """L-BFGS-B from the two starts; returns (theta, record)."""
    m = X.m[(source, pool)]
    mop = X.mop if opening else np.zeros_like(m)
    wq = level_weights(X.y)
    runs = []
    best = None
    for a0 in STARTS:
        r = optimize.minimize(objective, start(a0, opening), args=(m, mop, X.n, X.y, wq, opening, B),
                              jac=True, method="L-BFGS-B", bounds=bounds(opening),
                              options={"maxiter": MAXITER})
        runs.append({"start_a": a0, "loss": float(r.fun), "nit": int(r.nit), "success": bool(r.success),
                     "message": str(r.message)})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, {"loss": float(best.fun), "starts": runs, "params": params_dict(best.x, opening, B)}


def params_dict(theta, opening, B):
    a, g, w, beta = unpack(np.asarray(theta, dtype=np.float64), opening)
    return {"a": float(a), "gamma": float(g), "w": float(w),
            "beta": [float(x) for x in beta], "b": [float(x) for x in B @ beta]}


def predict(theta, X, source, pool, opening, B):
    m = X.m[(source, pool)]
    mop = X.mop if opening else np.zeros_like(m)
    z, _ = question_logits(theta, m, mop, X.n, opening, B)
    return softmax(z)


def uncalibrated_theta(opening=False):
    return np.zeros(n_params(opening))


# ================================================================ metrics
def metrics(pred, y, n, player):
    sc = official_score(pred, y)
    per_level = {}
    for i, lv in enumerate(LEVELS):
        mk = y == i
        if mk.any():
            per_level[lv] = float(sc[mk].mean())
    out = {"rank_balanced": float(np.mean(list(per_level.values()))),
           "question_mean": float(sc.mean()), "n_questions": int(len(y)),
           "levels_present": len(per_level), "per_level": per_level, "per_n_bin": {}}
    for lo, hi in N_BINS:
        mk = (n >= lo) & (n <= hi)
        key = f"{lo}-{hi}"
        if not mk.any():
            out["per_n_bin"][key] = {"n_questions": 0}
            continue
        pl = [float(sc[mk & (y == i)].mean()) for i in range(N_LEVELS) if (mk & (y == i)).any()]
        out["per_n_bin"][key] = {
            "n_questions": int(mk.sum()), "rank_balanced": float(np.mean(pl)), "levels_present": len(pl),
            "question_mean": float(sc[mk].mean()),
            "pred_distribution": dict(zip(LEVELS, np.bincount(pred[mk], minlength=N_LEVELS).tolist())),
            "true_distribution": dict(zip(LEVELS, np.bincount(y[mk], minlength=N_LEVELS).tolist()))}
    out["pred_distribution"] = dict(zip(LEVELS, np.bincount(pred, minlength=N_LEVELS).tolist()))
    return out, sc


class PlayerBootstrap:
    """Resample players within each level (N_BOOT times, seed BOOT_SEED); same draws for every
    combination, so differences are paired. Statistic: rank-balanced mean over the questions."""

    def __init__(self, player, y, n_players):
        self.player = player
        self.n_players = n_players
        level_of = np.full(n_players, -1)
        level_of[player] = y
        rng = np.random.default_rng(BOOT_SEED)
        self.draws = []
        for i in range(N_LEVELS):
            members = np.nonzero(level_of == i)[0]
            if len(members):
                self.draws.append(members[rng.integers(0, len(members), size=(N_BOOT, len(members)))])
        self.count = np.bincount(player, minlength=n_players).astype(np.float64)

    def stats(self, sc):
        s = np.bincount(self.player, weights=sc, minlength=self.n_players)
        per_level = [s[d].sum(1) / self.count[d].sum(1) for d in self.draws]
        return np.mean(per_level, axis=0)

    def summary(self, sc, sc_ref=None):
        st = self.stats(sc)
        out = {"se": float(np.std(st, ddof=1)) if N_BOOT > 1 else None}
        if sc_ref is not None:
            d = st - self.stats(sc_ref)
            out["diff_vs_R00_se"] = float(np.std(d, ddof=1))
            out["diff_vs_R00_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        return out


# ================================================================ inputs
def dumps_dir(p02_dir):
    return os.path.join(p02_dir, "dumps")


def load_dump(path, n_rows, expect_s):
    z = load_npz(path, ["logits", "weights_sha256", "syms"])
    lg = np.asarray(z["logits"])
    if lg.dtype != np.float32:
        raise P02Error(f"{path}: logits must be float32, got {lg.dtype}")
    if lg.ndim != 3 or lg.shape[0] != n_rows or lg.shape[2] != N_LEVELS:
        raise P02Error(f"{path}: logits shape {lg.shape}, expected [{n_rows}, S, 10]")
    s = int(z["syms"])
    if s != lg.shape[1] or s != expect_s:
        raise P02Error(f"{path}: syms {s}, logits S {lg.shape[1]}, expected {expect_s}")
    return {"logits": lg, "weights_sha256": str(z["weights_sha256"]), "syms": s, "path": path,
            "sha256": sha256_file(path), "marker": read_marker(path)}


def read_marker(path):
    """_DONE_<file stem>.json written by the notebook, if present (recorded, not required)."""
    d, f = os.path.split(path)
    mk = os.path.join(d, "_DONE_" + os.path.splitext(f)[0] + ".json")
    if not os.path.exists(mk):
        return None
    with open(mk, encoding="utf-8") as fh:
        obj = json.load(fh)
    rec = {"file": os.path.basename(mk)}
    if isinstance(obj, dict) and "sha256" in obj:
        rec["sha256_matches_file"] = obj["sha256"] == sha256_file(path)
    for k in ("rows", "gpu_name", "torch", "cudnn", "run"):  # gpu_name: GPU of the run that wrote the dump
        if isinstance(obj, dict) and k in obj:
            rec[k] = obj[k]
    return rec


def last_gpu_name(env_path):
    """GPU model of the last run in an env.json (first field of nvidia-smi), as the notebook's
    P01_GPU_NAME; None without a GPU; 'missing' when the file does not exist."""
    if not os.path.exists(env_path):
        return "missing"
    with open(env_path, encoding="utf-8") as f:
        obj = json.load(f)
    runs = obj.get("runs", [obj]) if isinstance(obj, dict) else []
    smi = ((runs[-1] if runs else {}) or {}).get("gpu") or {}
    smi = smi.get("nvidia_smi") or []
    return smi[0].split(",")[0].strip() if smi else None


def gpu_names(env_path):
    if not os.path.exists(env_path):
        return None
    with open(env_path, encoding="utf-8") as f:
        obj = json.load(f)
    runs = obj.get("runs", [obj]) if isinstance(obj, dict) else []
    names = set()
    for r in runs:
        smi = ((r or {}).get("gpu") or {}).get("nvidia_smi") or []
        names.update(s.split(",")[0].strip() for s in smi if s)
        if not smi:
            names.add("no GPU")
    return sorted(names)


def git(*args):
    try:
        r = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    return r.returncode, r.stdout.strip()


def sha256_text(path):
    """SHA-256 of a source file with CRLF folded to LF: the repo has core.autocrlf=true, so a later
    checkout may rewrite line ends without changing the code (same rule as the player tools)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def tool_state():
    rc, head = git("rev-parse", "HEAD")
    rels = ["tools/" + f for f in TOOL_FILES]
    rc2, st = git("status", "--porcelain", "--", *rels)
    dirty = [line[3:] for line in st.splitlines()] if rc2 == 0 else None
    return {"git_head": head if rc == 0 else None, "dirty_or_untracked": dirty,
            "files_sha256_rule": "sha256 of the file with CRLF folded to LF",
            "files_sha256": {f: sha256_text(os.path.join(HERE, f)) for f in TOOL_FILES}}


def require_committed(path):
    """frozen_config.json must be in HEAD and unchanged in the working tree."""
    if not is_inside(path, REPO):
        raise P02Error(f"{path} is outside the repository; a real apply needs the committed frozen config "
                       f"(use --smoke with a scratch dir for smoke runs)")
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(REPO)).replace(os.sep, "/")
    rc, _ = git("cat-file", "-e", f"HEAD:{rel}")
    if rc != 0:
        raise P02Error(f"{rel} is not committed; commit frozen_config.json before apply")
    rc, _ = git("diff", "--quiet", "HEAD", "--", rel)
    if rc != 0:
        raise P02Error(f"{rel} differs from HEAD; apply only runs on the committed frozen config")
    rc, head = git("rev-parse", "HEAD")
    return {"frozen_config": rel, "git_head": head}


def require_committed_results(p02_dir):
    """The unpacked Colab results (dumps/ and the run records) must be committed and unchanged before
    tune and apply (PREREG_02 section 6 step 4: results are fixed before any tuning)."""
    if not is_inside(p02_dir, REPO):
        raise P02Error(f"{p02_dir} is outside the repository; tune / apply run on the committed Colab "
                       f"results (use --smoke for smoke dumps in a scratch dir)")
    ddir = dumps_dir(p02_dir)
    files = [os.path.join(ddir, f) for f in sorted(os.listdir(ddir))] if os.path.isdir(ddir) else []
    if not files:
        raise P02Error(f"no dumps in {ddir}")
    missing = [r for r in NOTEBOOK_REQUIRED if not os.path.exists(os.path.join(p02_dir, r))]
    if missing:
        raise P02Error(f"notebook result files missing in {p02_dir}: {missing}")
    files += [os.path.join(p02_dir, *r.split("/")) for r in NOTEBOOK_FILES
              if os.path.exists(os.path.join(p02_dir, *r.split("/")))]
    rels = [os.path.relpath(os.path.realpath(f), os.path.realpath(REPO)).replace(os.sep, "/") for f in files]
    rc, _ = git("ls-files", "--error-unmatch", "--", *rels)
    if rc != 0:
        raise P02Error(f"the Colab results in {p02_dir} are not all committed; commit the unpacked results "
                       f"(dumps/ and the run records) before tune / apply")
    rc, st = git("status", "--porcelain", "--", *rels)
    if rc != 0 or st:
        raise P02Error(f"the committed Colab results changed in the working tree: {st[:300]}")
    rc, head = git("rev-parse", "HEAD")
    return {"files": len(rels), "git_head": head if rc == 0 else None,
            "rule": "dumps/* and the notebook run records tracked by git and unchanged"}


def read_json(path):
    refuse_label_file(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ================================================================ repro
def val_groups(vk):
    """val_keys -> ordered [(question_id, [row indices])], checking slots run 1..k in order."""
    order, groups = [], {}
    for i, (q, s) in enumerate(zip(vk["question_id"], vk["slot"])):
        if q not in groups:
            groups[q] = []
            order.append(q)
        elif order[-1] != q:
            raise P02Error(f"val_keys: question {q} is not contiguous")
        groups[q].append(i)
        if s != len(groups[q]):
            raise P02Error(f"val_keys: question {q} slot {s} out of order")
    return [(q, groups[q]) for q in order]


def load_val_keys(ddir):
    path = os.path.join(ddir, "val_keys.npz")
    z = load_npz(path, ["question_id", "slot", "color", "parsed"])
    vk = {"question_id": str_array(z["question_id"]), "slot": [int(x) for x in np.asarray(z["slot"]).tolist()],
          "color": str_array(z["color"]), "parsed": np.asarray(z["parsed"]).astype(bool),
          "path": path, "sha256": sha256_file(path)}
    n = len(vk["slot"])
    if any(len(vk[k]) != n for k in ("question_id", "color", "parsed")):
        raise P02Error(f"{path}: arrays of different lengths")
    bad = sorted({c for c in vk["color"] if c not in ("B", "W")})
    if bad:
        raise P02Error(f"{path}: colour values must be 'B'/'W', got {bad[:5]}")
    vk["n"] = n
    vk["groups"] = val_groups(vk)
    return vk


def check_val_logits(dump, parsed):
    lg = dump["logits"]
    fin = np.isfinite(lg).all(axis=(1, 2))
    nan = np.isnan(lg).all(axis=(1, 2))
    if not fin[parsed].all():
        raise P02Error(f"{dump['path']}: non-finite logits in a parsed row")
    if not nan[~parsed].all():
        raise P02Error(f"{dump['path']}: an unparsed row is not all NaN")


def repro_gpu(marker, nb_rc, p01_env, p02_env, smoke):
    """Did the run that wrote val_p01e5_d4.npz use a GPU model other than prereg01's?

    True / False, or None when unknown (then the strict 1e-4 applies). Source, in order: the dump's
    completion marker (gpu_name; None = no GPU, which only a SMOKE run on CPU may have, as prereg01's
    smoke run), repro_check.json same_gpu_as_p01, the GPU names in the two env.json."""
    p01_gpu = last_gpu_name(p01_env)
    out = {"dump_gpu_name": (marker or {}).get("gpu_name", "unrecorded"), "p01_gpu_name": p01_gpu}
    if marker and "gpu_name" in marker and p01_gpu != "missing":
        dump_gpu = marker["gpu_name"]
        if (dump_gpu is None or p01_gpu is None) and not smoke:
            out.update(differs=None, source="dump marker, but no GPU recorded (a real run needs one)")
        else:
            out.update(differs=dump_gpu != p01_gpu,
                       source="dumps/_DONE_val_p01e5_d4.json gpu_name vs prereg01 env.json (last run)")
    elif isinstance(nb_rc, dict) and isinstance(nb_rc.get("same_gpu_as_p01"), bool):
        out.update(differs=not nb_rc["same_gpu_as_p01"], source="repro_check.json same_gpu_as_p01")
    else:
        g1, g2 = gpu_names(p01_env), gpu_names(p02_env)
        out.update(differs=(set(g1) != set(g2)) if (g1 is not None and g2 is not None) else None,
                   source="env.json GPU names (all runs)")
    return out


def cmd_repro(a):
    check_out_dir(a.out_dir, a.smoke, extra_readonly=(a.p01_dir,))
    check_dump_kind(a.p02_dir, a.smoke)
    ddir = dumps_dir(a.p02_dir)
    vk = load_val_keys(ddir)
    check_smoke_size(vk["n"], VAL_GAMES_FULL, "val_keys.npz", a.smoke)
    d = load_dump(os.path.join(ddir, "val_p01e5_d4.npz"), vk["n"], DUMP_S["p01e5_d4"])
    check_val_logits(d, vk["parsed"])
    p01_csv = refuse_label_file(os.path.join(a.p01_dir, "rank_val_probs.csv"))
    try:
        p01 = decide.read_probs(p01_csv)
    except decide.DecideError as e:
        raise P02Error(f"{p01_csv}: {e}") from None
    p01_by = {r["question_id"]: r for r in p01}
    qids = [q for q, _ in vk["groups"]]
    if a.smoke:
        missing = [q for q in qids if q not in p01_by]
        if missing:
            raise P02Error(f"val dump questions not in prereg01's file: {missing[:3]}")
    elif qids != [r["question_id"] for r in p01]:
        raise P02Error("val_keys questions differ from prereg01's rank_val_probs.csv (set or order)")
    z = d["logits"][:, 0, :].astype(np.float64)
    max_diff, b_diff, count_bad = 0.0, [], []
    for q, rows in vk["groups"]:
        ref = p01_by[q]
        par = [i for i in rows if vk["parsed"][i]]
        if ref["n_games"] != len(rows) or ref["n_parsed"] != len(par):
            count_bad.append(q)
            continue
        if not par:
            continue
        p = softmax(z[par]).mean(0)
        max_diff = max(max_diff, float(np.max(np.abs(p - np.asarray(ref["probs"])))))
        if decide.rule_b([float(x) for x in p]) != decide.rule_b(ref["probs"]):
            b_diff.append(q)
    # GPU rule (PREREG_02 revision; the same rule as the notebook's check 1.1 point 2, which computes
    # the same number on the same dump): 1e-3 when the dump was made on another GPU model than
    # prereg01's, else (and when unknown) 1e-4.
    nb_rc_path = os.path.join(a.p02_dir, "repro_check.json")
    nb_rc = read_json(nb_rc_path) if os.path.exists(nb_rc_path) else None
    gpu = repro_gpu(d["marker"], nb_rc, os.path.join(a.p01_dir, "env.json"),
                    os.path.join(a.p02_dir, "env.json"), a.smoke)
    tol = REPRO_TOL_GPU if gpu["differs"] is True else REPRO_TOL
    nb_same = nb_rc.get("same_gpu_as_p01") if isinstance(nb_rc, dict) else None
    gpu_consistent = (None if not isinstance(nb_same, bool) or gpu["differs"] is None
                      else nb_same == (not gpu["differs"]))
    # seam: the notebook's C0 (float32 mean of the per-game softmax) and the local R00 (float64) on the
    # same dump must give the same largest difference
    nb_c0 = ((nb_rc or {}).get("val_c0_vs_p01") or {}).get("max_abs_diff") if isinstance(nb_rc, dict) else None
    seam = None if not isinstance(nb_c0, (int, float)) else abs(max_diff - float(nb_c0))
    seam_ok = None if seam is None else bool(seam <= SEAM_TOL)
    # rule-B allowance: 4 of 1,589; smoke scales it with the number of questions (as the notebook)
    b_allowed = REPRO_MAX_B_DIFF if not a.smoke else int(REPRO_MAX_B_DIFF * len(qids) / VAL_QUESTIONS_FULL)
    notebook = notebook_status(a.p02_dir)
    notebook_ok = notebook["repro_check_passed"] is True and notebook["p01_unchanged"] is True
    pred_b_check = None
    pb = os.path.join(a.p01_dir, "pred_B.csv")
    if os.path.exists(pb):
        with open(pb, newline="", encoding="utf-8") as f:
            saved = {r["question_id"]: r["pred_rank"] for r in csv.DictReader(f)}
        pred_b_check = sum(1 for r in p01 if r["probs"] is not None
                           and saved.get(r["question_id"]) != decide.rule_b(r["probs"]))
    if a.smoke:   # recorded; a seam or GPU mismatch that is present still fails
        seam_pass, gpu_pass = seam_ok is not False, gpu_consistent is not False
    else:
        seam_pass, gpu_pass = seam_ok is True, gpu_consistent is not False
    passed = (not count_bad and max_diff <= tol and len(b_diff) <= b_allowed
              and (notebook_ok or a.smoke) and seam_pass and gpu_pass)
    rep = {
        "what": "PREREG_02 1.2 local reproduction check: R00 on the validation dumps vs prereg01",
        "passed": passed, "smoke": bool(a.smoke), "created": now_iso(),
        "questions": len(qids), "games": vk["n"], "games_parsed": int(vk["parsed"].sum()),
        "max_abs_diff": max_diff, "tolerance": tol,
        "tolerance_rule": "PREREG_02 1.2 with the GPU rule of 1.1 (revision): <= 1e-4; <= 1e-3 when the "
                          "dump was made on a GPU model other than prereg01's; 1e-4 when unknown",
        "gpu_differs_from_prereg01": gpu["differs"], "gpu_decided_by": gpu["source"],
        "dump_gpu_name": gpu["dump_gpu_name"], "p01_gpu_name": gpu["p01_gpu_name"],
        "gpu_prereg01_env": gpu_names(os.path.join(a.p01_dir, "env.json")),
        "gpu_prereg02_env": gpu_names(os.path.join(a.p02_dir, "env.json")),
        "notebook_same_gpu_as_p01": nb_same, "gpu_consistent_with_notebook": gpu_consistent,
        "seam": {"notebook_c0_max_abs_diff": nb_c0, "abs_difference": seam, "tolerance": SEAM_TOL,
                 "ok": seam_ok, "required": not a.smoke},
        "notebook_run": notebook, "notebook_run_ok": notebook_ok,
        "notebook_run_required": not a.smoke,
        "rule_b_differences": len(b_diff), "rule_b_max": b_allowed,
        "rule_b_differing_questions": b_diff,
        "notebook_rule_b_differing_questions": (((nb_rc or {}).get("rule_b_diff") or {}).get("question_ids")
                                                if isinstance(nb_rc, dict) else None),
        "count_mismatch_questions": count_bad,
        "prereg01_pred_B_vs_rule_b_on_its_probs_mismatches": pred_b_check,
        "p02_dumps_dir": os.path.abspath(ddir),
        "inputs_sha256": {"val_keys.npz": vk["sha256"], "val_p01e5_d4.npz": d["sha256"],
                          "prereg01 rank_val_probs.csv": sha256_file(p01_csv)},
        "val_p01e5_d4_weights_sha256": d["weights_sha256"], "marker": d["marker"],
    }
    if nb_rc is not None:
        rep["notebook_repro_check"] = {"sha256": sha256_file(nb_rc_path),
                                       "pass": nb_rc.get("pass") if isinstance(nb_rc, dict) else None}
    os.makedirs(a.out_dir, exist_ok=True)
    write_json(os.path.join(a.out_dir, "repro_local.json"), rep)
    print(f"repro: {len(qids)} questions, max |diff| {max_diff:.3g} (tolerance {tol:g}; dump GPU "
          f"{gpu['dump_gpu_name']!r} vs prereg01 {gpu['p01_gpu_name']!r}), seam {seam if seam is None else f'{seam:.3g}'}"
          f", rule-B differences {len(b_diff)} (max {b_allowed}), count mismatches {len(count_bad)}, "
          f"notebook run {'ok' if notebook_ok else 'NOT ok'}{' (not required with --smoke)' if a.smoke else ''}"
          f" -> {'PASSED' if passed else 'FAILED'}")
    if not a.smoke and not notebook_ok:
        print(f"the notebook run did not end cleanly: {notebook}", file=sys.stderr)
    if gpu_consistent is False:
        print(f"the notebook's same_gpu_as_p01 ({nb_same}) disagrees with the dump marker", file=sys.stderr)
    return 0 if passed else 1


def notebook_status(p02_dir):
    """What the Colab run itself reported: its reproduction check and the prereg01 read-only check.

    repro_check.json is a flat dict with 'pass'; p01_readonly_after.json is {"runs": [...]} whose
    last entry has 'same_as_before'. Missing files give None (and a real run then fails)."""
    out = {"repro_check_passed": None, "p01_unchanged": None, "p01_changed_files": None}
    rc = os.path.join(p02_dir, "repro_check.json")
    if os.path.exists(rc):
        obj = read_json(rc)
        out["repro_check_passed"] = obj.get("pass") if isinstance(obj, dict) else None
    after = os.path.join(p02_dir, "p01_readonly_after.json")
    if os.path.exists(after):
        runs = (read_json(after) or {}).get("runs") or []
        if runs:
            out["p01_unchanged"] = runs[-1].get("same_as_before")
            out["p01_changed_files"] = runs[-1].get("changed_files")
    return out


# ================================================================ tune
def p02e9_status(p02_dir, ddir, keys, smoke):
    inner = os.path.join(ddir, "inner_p02e9_d4.npz")
    val = os.path.join(ddir, "val_p02e9_d4.npz")
    st = {"threshold_percent": P02E9_MIN_ACC, "inner_dump": os.path.exists(inner),
          "val_dump": os.path.exists(val), "inner_accuracy_percent": None, "history_val_acc_percent": None}
    st["definition"] = ("inner per-game accuracy of epoch 9 = mean(argmax of the original-orientation "
                        "logits in inner_p02e9_d4.npz == label) x 100 (the number tune uses) decides; "
                        "history.csv val_acc (the notebook's monitoring) is recorded only, and "
                        "history_same_side flags a disagreement about the threshold")
    hist = os.path.join(p02_dir, "ckpt", "history.csv")
    if os.path.exists(hist):
        with open(hist, newline="", encoding="utf-8") as f:
            st["history"] = list(csv.DictReader(f))
        for r in st["history"]:
            try:
                if int(float(r.get("epoch", ""))) == 9 and r.get("val_acc", "") != "":
                    st["history_val_acc_percent"] = float(r["val_acc"])
            except ValueError:
                pass
    dump = None
    if st["inner_dump"]:
        dump = load_dump(inner, keys["n"], DUMP_S["p02e9_d4"])
        acc = float(np.mean(np.argmax(dump["logits"][:, 0, :], axis=1) == keys["label"])) * 100
        st["inner_accuracy_percent"] = acc
        h = st["history_val_acc_percent"]
        if h is not None:
            st["history_minus_dump_points"] = h - acc
            st["history_same_side"] = (h < P02E9_MIN_ACC) == (acc < P02E9_MIN_ACC)
            if not st["history_same_side"]:
                print(f"note: epoch 9 inner accuracy from the dump {acc:.4f}% and from history.csv {h:.4f}% "
                      f"fall on different sides of {P02E9_MIN_ACC}%; the dump decides (recorded)", file=sys.stderr)
    if not (st["inner_dump"] and st["val_dump"]):
        st["usable"], st["reason"] = False, "epoch 9 dumps missing (epoch 9 not finished)"
    elif st["inner_accuracy_percent"] < P02E9_MIN_ACC:
        st["usable"], st["reason"] = False, f"inner accuracy < {P02E9_MIN_ACC}%"
    else:
        st["usable"], st["reason"] = True, "dumps present and inner accuracy >= threshold"
    return st, dump


def build_qset(rows_list, players, dump_row, fold_of, n_rows, op_o, op_has, player_bank):
    """rows_list: [(round, player index, positions)] -> QSet plus the game groups."""
    groups, n, y, pl, fd = [], [], [], [], []
    for _, pi, pos in rows_list:
        bank = player_bank[pi]
        g = [dump_row[bank[k]] for k in pos]
        groups.append(g)
        n.append(len(g))
        y.append(LEVELS.index(players[pi]["rank"]))
        pl.append(pi)
        fd.append(fold_of[pi])
    S = pool_matrix(groups, n_rows)
    Sop = pool_matrix([[r for r in g if op_has[r]] for g in groups], n_rows)
    X = QSet(n, y, pl, fd, np.asarray(Sop @ op_o))
    return X, S


def cmd_tune(a):
    t0 = time.time()
    check_out_dir(a.out_dir, a.smoke)
    frozen = os.path.join(a.out_dir, "frozen_config.json")
    if not a.smoke and os.path.exists(frozen) and is_inside(frozen, REPO):
        rel = os.path.relpath(os.path.realpath(frozen), os.path.realpath(REPO)).replace(os.sep, "/")
        rc, _ = git("ls-files", "--error-unmatch", "--", rel)
        if rc == 0:
            raise P02Error(f"{rel} is already tracked by git (frozen); tune will not overwrite it")
    check_dump_kind(a.p02_dir, a.smoke)
    tools = tool_state()
    if not a.smoke and (tools["git_head"] is None or tools["dirty_or_untracked"] is None
                        or tools["dirty_or_untracked"]):
        raise P02Error(f"commit the tools first (git state: head {tools['git_head']}, uncommitted "
                       f"{tools['dirty_or_untracked']})")
    results_commit = None if a.smoke else require_committed_results(a.p02_dir)
    ddir = dumps_dir(a.p02_dir)
    repro_path = os.path.join(a.out_dir, "repro_local.json")
    if not os.path.exists(repro_path):
        raise P02Error("run 'p02_rank_fuse.py repro' first (repro_local.json missing)")
    repro = read_json(repro_path)
    if not repro.get("passed"):
        raise P02Error("the reproduction check did not pass; tune does not run")
    if os.path.normcase(os.path.realpath(repro["p02_dumps_dir"])) != os.path.normcase(os.path.realpath(ddir)):
        raise P02Error("repro_local.json was made on a different dumps directory")
    if bool(repro.get("smoke")) != bool(a.smoke):
        raise P02Error("repro_local.json smoke flag differs from this run")
    for f in ("val_keys.npz", "val_p01e5_d4.npz"):  # hashes only; tune never reads validation contents
        if repro["inputs_sha256"].get(f) != sha256_file(os.path.join(ddir, f)):
            raise P02Error(f"{f} changed since the reproduction check; rerun repro")

    keys = load_inner_keys(ddir)
    check_smoke_size(keys["n"], EXPECTED["inner"]["games"], "inner_keys.npz", a.smoke)
    pseudo = load_pseudo(os.path.join(a.out_dir, "pseudo_inner.json"), verify=True)
    pobj = pseudo["obj"]
    if pobj["inputs"]["inner_keys_sha256"] != keys["sha256"]:
        raise P02Error("pseudo_inner.json was built from a different inner_keys.npz")
    if not a.smoke:
        if keys["n"] != EXPECTED["inner"]["games"]:
            raise P02Error(f"inner_keys has {keys['n']} games, expected {EXPECTED['inner']['games']}")
        if pobj["smoke"] or pobj["params"]["min_games"] != MIN_GAMES or \
                pobj["params"]["fit_min_games"] != FIT_MIN_GAMES:
            raise P02Error("pseudo_inner.json is a smoke file or has non-registered thresholds")
    split = read_split(a.split)
    inner_set = players_for(split, "inner")
    players = pseudo["players"]
    outside = sorted({p for p in keys["player_id"] if p not in inner_set}
                     | {p["player_id"] for p in players if p["player_id"] not in inner_set})
    if outside:
        raise P02Error(f"{len(outside)} player_ids are not split-v1 inner players, e.g. {outside[:3]}")
    assert all(p in inner_set for p in keys["player_id"])  # every player_id is an inner player

    dump_row = {}
    for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"])):
        dump_row[(r, g)] = i
    player_bank = []
    for p in players:
        rows = []
        for gid in p["bank"]:
            i = dump_row.get((p["rank"], gid))
            if i is None or keys["player_id"][i] != p["player_id"]:
                raise P02Error(f"bank game {gid} of {p['player_id']} is not in inner_keys for that player")
            rows.append((p["rank"], gid))
        player_bank.append(rows)
    fold_of = [p["fold"] for p in players]

    # opening scores aligned to the dump rows
    op = load_opening(os.path.join(a.out_dir, "opening_inner.npz"), "inner")
    op_row = {(r, g): i for i, (r, g) in enumerate(zip(str_array(op["rank"]), str_array(op["game_id"])))}
    op_pid = str_array(op["player_id"])
    idx = []
    for i, (r, g) in enumerate(zip(keys["rank"], keys["game_id"])):
        j = op_row.get((r, g))
        if j is None or op_pid[j] != keys["player_id"][i]:
            raise P02Error(f"inner game ({r}, {g}) missing from opening_inner.npz or player differs")
        idx.append(j)
    idx = np.asarray(idx)
    op_o = np.asarray(op["o"], dtype=np.float64)[idx]
    op_has = np.asarray(op["has"], dtype=bool)[idx]

    # CNN dumps
    loaded, avail_note = {}, {}
    for name in ("p01e1", "p01e2", "p01e3", "p01e4", "p01e5_d4"):
        path = os.path.join(ddir, f"inner_{name}.npz")
        if os.path.exists(path):
            loaded[name] = load_dump(path, keys["n"], DUMP_S[name])
        elif name in ("p01e3", "p01e4", "p01e5_d4"):
            if not a.smoke:
                raise P02Error(f"missing {path}")
            avail_note[name] = "missing (smoke)"
    for name, d in loaded.items():
        if not np.isfinite(d["logits"]).all():
            raise P02Error(f"{d['path']}: non-finite logits (every inner game is parsed)")
    if "p01e5_d4" not in loaded:
        raise P02Error("inner_p01e5_d4.npz is required (R00)")
    p02e9, d9 = p02e9_status(a.p02_dir, ddir, keys, a.smoke)
    if d9 is not None:
        if not np.isfinite(d9["logits"]).all():
            raise P02Error(f"{d9['path']}: non-finite logits")
        loaded["p02e9_d4"] = d9
    accuracies = {name: float(np.mean(np.argmax(d["logits"][:, 0, :], axis=1) == keys["label"])) * 100
                  for name, d in loaded.items()}
    align = {"p01e5_inner_accuracy_percent": accuracies["p01e5_d4"], "registered": P01E5_ACC,
             "tolerance_points": P01E5_ACC_TOL,
             "ok": abs(accuracies["p01e5_d4"] - P01E5_ACC) <= P01E5_ACC_TOL}
    if not a.smoke and not align["ok"]:
        raise P02Error(f"inner_p01e5_d4 accuracy {accuracies['p01e5_d4']:.4f}% is not {P01E5_ACC} +- "
                       f"{P01E5_ACC_TOL}: dumps misaligned with inner_keys or different weights")

    sources_ok = {}
    for s, parts in SOURCES.items():
        if s in P02_SOURCES and not p02e9["usable"]:
            sources_ok[s] = False
        else:
            sources_ok[s] = all(n in loaded for n, _ in parts)
    logits = {n: d["logits"] for n, d in loaded.items()}
    lp = {s: source_logprobs(s, logits) for s, ok in sources_ok.items() if ok}

    X_fit, S_fit = build_qset(pseudo["fit"], players, dump_row, fold_of, keys["n"], op_o, op_has,
                              player_bank)
    X_eval, S_eval = build_qset(pseudo["eval"], players, dump_row, fold_of, keys["n"], op_o, op_has,
                                player_bank)
    for s in lp:
        for pool in ("prob", "log"):
            X_fit.m[(s, pool)] = pooled(S_fit, lp[s], pool)
            X_eval.m[(s, pool)] = pooled(S_eval, lp[s], pool)
    B = basis()
    boot = PlayerBootstrap(X_eval.player, X_eval.y, len(players))
    print(f"inputs ready: fit {len(X_fit)} questions, eval {len(X_eval)} questions, sources "
          f"{sorted(lp)} ({time.time() - t0:.0f}s)", flush=True)

    results, sc_r00 = [], None
    for c in COMBOS:
        rec = dict(c)
        if not sources_ok[c["source"]]:
            rec["status"] = "dropped"
            rec["reason"] = (p02e9["reason"] if c["source"] in P02_SOURCES
                             else "source dumps missing (smoke)")
            results.append(rec)
            continue
        oof_pred = np.full(len(X_eval), -1, dtype=np.int64)
        folds = []
        for f in range(5):
            ev = X_eval.fold == f
            if not c["calibrated"]:
                theta, frec = uncalibrated_theta(), {"fit_questions": 0, "params": None}
            else:
                tr = X_fit.fold != f
                if not tr.any():
                    if not a.smoke:
                        raise P02Error(f"fold {f}: no fit questions")
                    theta = uncalibrated_theta(c["opening"])
                    frec = {"fit_questions": 0, "fit_empty_smoke": True,
                            "params": params_dict(theta, c["opening"], B)}
                else:
                    theta, frec = fit(X_fit.subset(tr), c["source"], c["pool"], c["opening"], B)
                    frec["fit_questions"] = int(tr.sum())
            if ev.any():
                P = predict(theta, X_eval.subset(ev), c["source"], c["pool"], c["opening"], B)
                oof_pred[ev] = rule_b_np(P)
            frec["fold"] = f
            frec["eval_questions"] = int(ev.sum())
            folds.append(frec)
        assert (oof_pred >= 0).all()
        m, sc = metrics(oof_pred, X_eval.y, X_eval.n, X_eval.player)
        if c["id"] == "R00":
            sc_r00 = sc
        m.update(boot.summary(sc, None if c["id"] == "R00" else sc_r00))
        rec.update({"status": "ok", "oof": m, "folds": folds if c["calibrated"] else None})
        results.append(rec)
        print(f"{c['id']} {c['source']} {c['pool']:4s} cal={int(c['calibrated'])} op={int(c['opening'])}: "
              f"OOF rank-balanced {m['rank_balanced']:.4f}, question mean {m['question_mean']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    r00 = next(r for r in ok if r["id"] == "R00")
    best = max(r["oof"]["rank_balanced"] for r in ok)
    chosen = next(r for r in ok if best - r["oof"]["rank_balanced"] <= SELECT_TOL + FLOAT_EPS)
    delta = chosen["oof"]["rank_balanced"] - r00["oof"]["rank_balanced"]
    gate = {"threshold": GATE, "selected_minus_R00": delta, "passed": bool(delta >= GATE - FLOAT_EPS),
            "meaning": "only when passed is the local validation set scored (PREREG_02 1.2); otherwise "
                       "record 'not submitted' and keep this scoring slot"}
    selection = {"rule": f"highest OOF rank-balanced mean (rule B); within {SELECT_TOL} of the best -> the "
                         f"earliest in the list (compared with {FLOAT_EPS} float slack)",
                 "best_id": next(r["id"] for r in ok if r["oof"]["rank_balanced"] == best),
                 "best_rank_balanced": best, "selected": chosen["id"],
                 "selected_rank_balanced": chosen["oof"]["rank_balanced"],
                 "R00_rank_balanced": r00["oof"]["rank_balanced"],
                 "within_tolerance": [r["id"] for r in ok
                                      if best - r["oof"]["rank_balanced"] <= SELECT_TOL + FLOAT_EPS]}
    # refit the selected combination on all fit questions
    if chosen["calibrated"]:
        if len(X_fit) == 0:
            if not a.smoke:
                raise P02Error("no fit questions for the refit")
            theta = uncalibrated_theta(chosen["opening"])
            refit = {"fit_questions": 0, "fit_empty_smoke": True}
        else:
            theta, refit = fit(X_fit, chosen["source"], chosen["pool"], chosen["opening"], B)
            refit["fit_questions"] = len(X_fit)
    else:
        theta, refit = uncalibrated_theta(), {"fit_questions": 0, "note": "R00 has no parameters"}
    final = params_dict(theta, chosen["opening"], B)

    used_dumps = sorted({n for n, _ in SOURCES[chosen["source"]]})
    inputs = {"inner_keys.npz": keys["sha256"], "pseudo_inner.json": pseudo["sha256"],
              "opening_inner.npz": op["sha256"], "repro_local.json": sha256_file(repro_path),
              "players_split.csv": sha256_file(a.split)}
    inputs.update({f"inner_{n}.npz": d["sha256"] for n, d in loaded.items()})
    fit_k = {lv: float(X_fit.n[X_fit.y == i].mean()) if (X_fit.y == i).any() else None
             for i, lv in enumerate(LEVELS)}
    eval_k = {lv: float(X_eval.n[X_eval.y == i].mean()) if (X_eval.y == i).any() else None
              for i, lv in enumerate(LEVELS)}
    report = {
        "what": "PREREG_02 competition 2 tune: combinations R00..R20 on inner pseudo-questions, 5-fold OOF",
        "created": now_iso(), "seconds": round(time.time() - t0, 1), "smoke": bool(a.smoke),
        "definitions": {"sources": SOURCE_TEXT, "formula": "p = softmax(e^a (n/12)^gamma (m + w m_op + b)), "
                        "b = B beta (fixed 10x9 orthonormal basis, sum b = 0)",
                        "pooling": {"prob": "m = log(mean_g exp l_g)", "log": "m = mean_g l_g"},
                        "opening": "m_op = mean over the question's games with an opening score of the "
                                   "opening log_softmax (0 if none)",
                        "objective": f"sum_q w_q NLL_q, w_q = 1/(L * questions of the level), + {PENALTY} "
                                     f"||b||^2; L-BFGS-B bounds {BOUNDS}, starts a in {STARTS}, "
                                     f"maxiter {MAXITER}",
                        "oof": "fold f: fit on fit questions (rounds 0-49) of players not in fold f, "
                               "predict eval questions (rounds 1000-1199) of players in fold f",
                        "metric": "official score with rule B; rank_balanced = mean over levels of the "
                                  "per-level mean (main metric); question_mean = plain mean",
                        "bootstrap": f"players resampled within each level, {N_BOOT} times, numpy "
                                     f"default_rng({BOOT_SEED}); same draws for every combination",
                        "n_bins": [f"{lo}-{hi}" for lo, hi in N_BINS]},
        "p02e9": p02e9,
        "inner_per_game_accuracy_percent": accuracies, "alignment_check": align,
        "questions": {"fit": len(X_fit), "eval": len(X_eval),
                      "fit_mean_games_by_level": fit_k, "eval_mean_games_by_level": eval_k,
                      "pseudo_counts": pobj["counts"]},
        "combinations": results,
        "selection": selection, "gate": gate,
        "refit": dict(refit, combo=chosen["id"]),
        "inputs_sha256": inputs, "tool": tools, "notebook_results_commit_check": results_commit,
        "notes": avail_note,
    }
    frozen_obj = {
        "format": FROZEN_FORMAT,
        "what": "PREREG_02 competition 2 frozen configuration (selected on inner; commit before apply)",
        "created": report["created"], "smoke": bool(a.smoke),
        "combo": {k: chosen[k] for k in ("id", "source", "pool", "calibrated", "opening")},
        "source_definition": SOURCE_TEXT[chosen["source"]],
        "dumps": {n: {"weights_sha256": loaded[n]["weights_sha256"], "syms": loaded[n]["syms"]}
                  for n in used_dumps},
        "params": final,
        "basis": [[float(x) for x in row] for row in B],
        "formula": report["definitions"]["formula"],
        "decision_rule": "B, tools/decide.py (argmax p_k + e^-1 (p_{k-1} + p_{k+1}))",
        "opening_build_id": op["meta"]["build_id"] if chosen["opening"] else None,
        "selection": selection, "gate": gate, "p02e9": p02e9,
        "inputs_sha256": inputs, "tool": tools,
    }
    write_json(os.path.join(a.out_dir, "tune_report.json"), report)
    write_json(frozen, frozen_obj)
    print(f"selected {chosen['id']} (OOF rank-balanced {chosen['oof']['rank_balanced']:.4f}; best "
          f"{selection['best_id']} {best:.4f}; R00 {r00['oof']['rank_balanced']:.4f}); gate "
          f"{'PASSED' if gate['passed'] else 'NOT passed'} (delta {delta:+.4f}, need {GATE})")
    print(f"wrote tune_report.json and frozen_config.json in {a.out_dir}; commit frozen_config.json "
          f"before apply, together with competition 1's frozen_config.json (PREREG_02 rev 3, section 7 "
          f"item 6: both frozen before any apply)")
    return 0


# ================================================================ apply
def read_question_meta(path):
    """[(question_id, num_games, [colours])] in file order (no game content kept)."""
    refuse_label_file(path)
    csv.field_size_limit(10**8)
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n = int(r["num_games"])
            out.append((r["question_id"], n, [r[f"color_{i}"].strip() for i in range(1, n + 1)]))
    return out


def cmd_apply(a):
    t0 = time.time()
    check_out_dir(a.out_dir, a.smoke)
    check_dump_kind(a.p02_dir, a.smoke)
    fpath = os.path.join(a.out_dir, "frozen_config.json")
    commit = None if a.smoke else require_committed(fpath)
    results_commit = None if a.smoke else require_committed_results(a.p02_dir)
    fc = read_json(fpath)
    if fc.get("format") != FROZEN_FORMAT:
        raise P02Error(f"{fpath}: not a {FROZEN_FORMAT} file")
    if bool(fc.get("smoke")) != bool(a.smoke):
        raise P02Error("frozen_config.json smoke flag differs from this run")
    B = np.asarray(fc["basis"], dtype=np.float64)
    if B.shape != (N_LEVELS, N_LEVELS - 1) or not np.array_equal(B, basis()):
        raise P02Error("frozen basis differs from the fixed basis")
    tools = tool_state()
    if not a.smoke:
        changed = [f for f, h in tools["files_sha256"].items() if fc["tool"]["files_sha256"].get(f) != h]
        if changed:
            raise P02Error(f"tool files changed since tune: {changed}; rerun tune (and revise) first")
    c = fc["combo"]
    par = fc["params"]
    theta = np.array([par["a"], par["gamma"]] + ([par["w"]] if c["opening"] else []) + par["beta"])
    if not np.allclose(B @ np.asarray(par["beta"]), par["b"], atol=1e-12):
        raise P02Error("frozen b is not B @ beta")

    ddir = dumps_dir(a.p02_dir)
    vk = load_val_keys(ddir)
    check_smoke_size(vk["n"], VAL_GAMES_FULL, "val_keys.npz", a.smoke)
    logits = {}
    dump_sha = {}
    for name, info in fc["dumps"].items():
        d = load_dump(os.path.join(ddir, f"val_{name}.npz"), vk["n"], DUMP_S[name])
        if d["weights_sha256"] != info["weights_sha256"]:
            raise P02Error(f"val_{name}.npz weights {d['weights_sha256']} differ from the inner dump's "
                           f"{info['weights_sha256']}")
        check_val_logits(d, vk["parsed"])
        logits[name] = np.where(vk["parsed"][:, None, None], d["logits"], 0.0).astype(np.float32)
        dump_sha[f"val_{name}.npz"] = d["sha256"]
    lp = source_logprobs(c["source"], logits)

    questions = read_question_meta(a.questions)
    by_q = {q: (n, cols) for q, n, cols in questions}
    groups = vk["groups"]
    qids = [q for q, _ in groups]
    if a.smoke:
        if qids != [q for q, _, _ in questions][:len(qids)]:
            raise P02Error("smoke val dump questions are not the first questions of the question file")
    elif qids != [q for q, _, _ in questions]:
        raise P02Error("val_keys questions differ from the question file (set or order)")
    for q, rows in groups:
        n, cols = by_q[q]
        if len(rows) != n or [vk["color"][i] for i in rows] != cols:
            raise P02Error(f"question {q}: val_keys slots/colours differ from the question file")

    cnn_groups = [[i for i in rows if vk["parsed"][i]] for _, rows in groups]
    S = pool_matrix(cnn_groups, vk["n"])
    m = pooled(S, lp, c["pool"])
    n = np.array([len(g) for g in cnn_groups], dtype=np.float64)
    if c["opening"]:
        op = load_opening(os.path.join(a.out_dir, "opening_val.npz"), "val")
        if op["meta"]["build_id"] != fc["opening_build_id"]:
            raise P02Error("opening_val.npz is from a different build than the frozen opening_inner.npz")
        pos = {(q, int(s)): i for i, (q, s) in enumerate(zip(str_array(op["question_id"]),
                                                           np.asarray(op["slot"]).tolist()))}
        oi = []
        for i, (q, s) in enumerate(zip(vk["question_id"], vk["slot"])):
            j = pos.get((q, s))
            if j is None or str(op["color"][j]) != vk["color"][i]:
                raise P02Error(f"opening_val.npz: no matching entry for question {q} slot {s}")
            oi.append(j)
        oi = np.asarray(oi)
        o = np.asarray(op["o"], dtype=np.float64)[oi]
        has = np.asarray(op["has"], dtype=bool)[oi]
        mop = np.asarray(pool_matrix([[i for i in rows if has[i]] for _, rows in groups], vk["n"]) @ o)
        opening_sha = op["sha256"]
    else:
        mop = np.zeros_like(m)
        opening_sha = None
    ok = n > 0
    P = np.full((len(groups), N_LEVELS), np.nan)
    if ok.any():
        z, _ = question_logits(theta, m[ok], mop[ok], n[ok], c["opening"], B)
        P[ok] = softmax(z)

    out_csv = os.path.join(a.out_dir, "rank_val_probs.csv")
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(PROB_COLUMNS)
        for k, (q, rows) in enumerate(groups):
            probs = [repr(float(x)) for x in P[k]] if ok[k] else [""] * N_LEVELS
            w.writerow([q, len(rows), int(n[k])] + probs)
    os.replace(tmp, out_csv)
    decide.read_probs(out_csv)  # the file must pass decide.py's strict reader
    # the gate is decided in tune (frozen_config.json); repeat it here so that the scoring step reads one file
    gate = fc.get("gate")
    blocked = []
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        blocked.append("gate not passed in tune (PREREG_02 1.2): do not score; record 'not submitted'")
    rep = {"what": "PREREG_02 competition 2 apply: frozen configuration on the validation questions",
           "created": now_iso(), "seconds": round(time.time() - t0, 1), "smoke": bool(a.smoke),
           "commit_check": commit, "notebook_results_commit_check": results_commit,
           "combo": c, "params": par,
           "questions": len(groups), "questions_without_parsed_game": int((~ok).sum()),
           "games": vk["n"], "games_parsed": int(vk["parsed"].sum()),
           "inputs_sha256": dict(dump_sha, **{"val_keys.npz": vk["sha256"],
                                              "frozen_config.json": sha256_file(fpath),
                                              "questions": sha256_file(a.questions),
                                              "opening_val.npz": opening_sha}),
           "output": os.path.abspath(out_csv), "output_sha256": sha256_file(out_csv),
           "tool": tools,
           "gate": gate, "score_allowed": not blocked, "score_blocked_reasons": blocked,
           "next": "python tools/decide.py <out-dir>/rank_val_probs.csv <out-dir>, then "
                   "tools/check_submission.py rank <out-dir>/pred_B.csv --questions <question file>; "
                   "score (PREREG_02 1.3) only when score_allowed is true"}
    write_json(os.path.join(a.out_dir, "apply_report.json"), rep)
    print(f"apply {c['id']}: {len(groups)} questions ({int((~ok).sum())} without a parsed game) -> {out_csv}")
    print("score allowed: %s%s" % (not blocked, (" (" + "; ".join(blocked) + ")") if blocked else ""))
    print(f"next: python tools/decide.py {out_csv} {a.out_dir}")
    return 0


# ================================================================ command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["repro", "tune", "apply"])
    ap.add_argument("--p01-dir", default=os.path.join(REPO, "results", "prereg01", "rank"))
    ap.add_argument("--p02-dir", default=DEFAULTS["p02_dir"])
    ap.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    ap.add_argument("--split", default=DEFAULTS["split"])
    ap.add_argument("--questions", default=os.path.join(REPO, "data", "local", "rank_val_questions.csv"))
    ap.add_argument("--smoke", action="store_true",
                    help="smoke dumps; output dir must be outside the repository")
    a = ap.parse_args(argv)
    try:
        return {"repro": cmd_repro, "tune": cmd_tune, "apply": cmd_apply}[a.command](a)
    except (P02Error, decide.DecideError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
