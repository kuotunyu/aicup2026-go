"""prereg05 exploration U1: shared helpers (inner pseudo-questions only; no validation data).

Rebuilds the PREREG_04 tune inputs (prereg02 inner pseudo-questions, C5 / OP / O4 / SWA / SWB) by importing the frozen
tools read-only, and reruns the OOF fits of Q00 (R20, prereg02's own path) and Q02 (R20 + O4 + SWA + SWB, generalised
path) exactly as tools/p04_rank_fuse.py cmd_tune does.

Safety: install_guard() makes every open / listdir / scandir / stat on a path under data/local raise, so nothing under
data/local (validation questions and answers) can be read by accident. No file is written by this module.
"""

import argparse
import builtins
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
E1 = float(np.exp(-1.0))
SCRATCH_CTX = os.environ.get(
    "C10_U1_CTX_DIR",
    os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "claude", "c10_u1_ctx"))   # never written to


# ---------------------------------------------------------------- guard: nothing under data/local is ever opened
_FORBIDDEN = os.path.normcase(os.path.abspath(os.path.join(REPO, "data", "local")))


def _forbidden(p):
    try:
        if isinstance(p, int):
            return False
        s = os.path.normcase(os.path.abspath(os.fspath(p)))
    except TypeError:
        return False
    return s == _FORBIDDEN or s.startswith(_FORBIDDEN + os.sep) or ("data" + os.sep + "local") in s


class LocalDataRefused(RuntimeError):
    pass


def install_guard():
    if getattr(builtins, "_c10_u1_guard", False):
        return
    real_open, real_listdir, real_scandir, real_stat = builtins.open, os.listdir, os.scandir, os.stat

    def g_open(file, *a, **k):
        if _forbidden(file):
            raise LocalDataRefused(f"U1 guard: refusing to open {file}")
        return real_open(file, *a, **k)

    def g_listdir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"U1 guard: refusing to list {path}")
        return real_listdir(path)

    def g_scandir(path="."):
        if _forbidden(path):
            raise LocalDataRefused(f"U1 guard: refusing to scan {path}")
        return real_scandir(path)

    def g_stat(path, *a, **k):
        if _forbidden(path):
            raise LocalDataRefused(f"U1 guard: refusing to stat {path}")
        return real_stat(path, *a, **k)

    builtins.open = g_open
    io.open = g_open
    os.listdir = g_listdir
    os.scandir = g_scandir
    os.stat = g_stat
    builtins._c10_u1_guard = True


install_guard()

import p02_rank_fuse as rf  # noqa: E402
import p03_rank_fuse as p3  # noqa: E402
import p04_rank_fuse as p4  # noqa: E402
from p02_rank_fuse import PlayerBootstrap, metrics, pooled, rule_b_np  # noqa: E402

assert rf.LEVELS == LEVELS


def context():
    """p04_rank_fuse's real-mode context in *preview* form (out-dir outside the repository; nothing is written)."""
    ns = argparse.Namespace(smoke=False, design_o4=False, p02_dir=None, p02_local_dir=None, p03_dir=None,
                            opp_dir=None, split=None, questions=None, out_dir=SCRATCH_CTX)
    c = p4.resolve(ns, "tune")
    assert c.preview and not c.smoke and not c.design
    return c


def load_inputs(verbose=True):
    """-> dict with the context, the inner question sets and the per-game log-probabilities (inner files only)."""
    c = context()
    frozen = p4.start(c)                         # frozen tools equal HEAD and the prereg02 / prereg03 records
    p02 = p3.p02_inputs(c, val=False)            # inner files only, SHA-256 equal to prereg02's records
    inner = p4.inner_setup(c)                    # prereg02 pseudo-questions + PREREG_03 single-colour set
    keys = inner["keys"]
    meta = {}
    lp, sha, probs = p4.inner_sources(c, keys, ["C5", "O4", "SWA", "SWB"], meta)
    if probs:
        raise RuntimeError(f"inner sources: {probs}")
    if verbose:
        print(f"inputs: fit {len(inner['X_fit'])}, eval {len(inner['X_eval'])}, single-colour "
              f"{len(inner['X_mono'])} questions; {keys['n']} inner games", flush=True)
    return {"c": c, "frozen": frozen, "p02": p02, "inner": inner, "lp": lp, "sha": sha, "meta": meta}


def run_q00(d):
    """Q00 = R20 through prereg02's own fit (as p04 cmd_tune run(..., path='p02'))."""
    inner, lp = d["inner"], d["lp"]
    Xm, Sm = inner["X_mono"], inner["S_mono"]
    lp5 = lp["C5"]
    p3.add_combo(inner, "Q00", lp5)
    Xm.m[("Q00", p3.POOL)] = pooled(Sm, lp5, "log")
    _m, _sc, folds, P = p3.run_oof(inner, "Q00", True, False, p4.B)
    if not np.array_equal(p3.fold_predict(inner["X_eval"], "Q00", folds, True, p4.B), P, equal_nan=True):
        raise RuntimeError("Q00 fold records do not reproduce the OOF probabilities")
    Pm = p3.fold_predict(Xm, "Q00", folds, True, p4.B)
    thetas = [[float(x) for x in p3.theta_of(fr["params"], True)] for fr in folds]
    return P, Pm, thetas


def run_gen(d, terms):
    """Generalised path (as p04 cmd_tune run(..., path='gen')) with base C5 and the given E terms."""
    inner, lp = d["inner"], d["lp"]
    Xf, Xe, Sf, Se = inner["X_fit"], inner["X_eval"], inner["S_fit"], inner["S_eval"]
    Xm, Sm = inner["X_mono"], inner["S_mono"]
    lp5 = lp["C5"]
    mf, me, mm = pooled(Sf, lp5, "log"), pooled(Se, lp5, "log"), pooled(Sm, lp5, "log")
    Ef, Ee, Em = (p4.question_terms(Xf, Sf, lp, terms), p4.question_terms(Xe, Se, lp, terms),
                  p4.question_terms(Xm, Sm, lp, terms))
    P, folds = p4.gen_oof_record(Xf, Xe, mf, me, Ef, Ee)
    if not np.array_equal(p4.gen_fold_predict(folds, me, Ee, Xe.n, Xe.fold), P):
        raise RuntimeError("fold records do not reproduce the OOF probabilities")
    Pm = p4.gen_fold_predict(folds, mm, Em, Xm.n, Xm.fold)
    thetas = [list(fr["theta"]) for fr in folds]
    return P, Pm, thetas


def score(pred, y):
    d = np.abs(np.asarray(pred) - np.asarray(y))
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def expected_scores(P):
    """[Q,10]: expected official score of answering level k when the level is distributed as P."""
    P = np.asarray(P, dtype=np.float64)
    left = np.zeros_like(P)
    right = np.zeros_like(P)
    left[:, 1:] = P[:, :-1]
    right[:, :-1] = P[:, 1:]
    return P + E1 * (left + right)


def rank_balanced(sc, y, mask=None):
    sc, y = np.asarray(sc, dtype=np.float64), np.asarray(y)
    if mask is not None:
        sc, y = sc[mask], y[mask]
    per = [sc[y == i].mean() for i in range(10) if (y == i).any()]
    return float(np.mean(per)) if per else float("nan")


def load_saved(path=None):
    """The npz written by u1_repro.py as a plain dict of arrays."""
    path = path or os.path.join(HERE, "u1_oof.npz")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
