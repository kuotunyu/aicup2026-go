"""prereg30 exploration (competition 2): P1 v2, pure numpy part (PLAN.md 1.2). No torch, no KataGo.

game_v2 turns one game's per-position arrays (P1's pos_logp plus the pos_lmax / pos_lead / pos_winp that
p30_scorer.V2Scorer adds, PLAN.md 1.1) into the 10 v2 features of PLAN.md 1.2; chunk_v2 does it for a whole chunk.
No file is written by this module.
"""

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

import numpy as np  # noqa: E402

V2_FORMAT = "c10-prereg30 v2 chunk v1"
EXTRA_KEYS = ("pos_lmax", "pos_lead", "pos_winp")
BIG = 5.0                 # PLAN.md 1.2: big loss = d_t >= 5 points (Kuboki 2024, n = 5)
T150 = 150                # moves with 0-based turn < 150
KEEP_IN_RUN = 2           # a run of consecutive big losses keeps its first two moves (the first of each colour)
V2_PER_SIDE = ("nlp_all", "nlp_150", "loss_150f", "big_150f", "wloss_150f")
V2_NAMES = tuple(f"{s}_{n}" for s in ("target", "opponent") for n in V2_PER_SIDE)
V2_DIM = len(V2_NAMES)
FORBIDDEN = ("/data/local", "rank_val_answers", "player_val_answers")
OWN_FILES = ("PLAN.md", "p30_lib.py", "p30_scorer.py", "p30_data.py", "p30_extract.py", "p30_eval.py", "test_p30.py",
             "run_pilot.sh")


class P30Error(RuntimeError):
    pass


def guard(p):
    q = os.path.abspath(p).replace("\\", "/").lower()
    if any(f in q for f in FORBIDDEN):
        raise SystemExit(f"refused path {p}")
    return p


def sha_lf(path):
    with open(guard(path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def sha_raw(path, block=1 << 20):
    h = hashlib.sha256()
    with open(guard(path), "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def own_code_shas():
    out = {}
    for name in OWN_FILES:
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            out[f"results/prereg30/explore/{name}"] = sha_lf(p)
    return out


# ================================================================ per-game v2 features (PLAN.md 1.2)
def keep_mask(d):
    """d: float64 [T] losses (NaN = no loss). -> (has, big, keep): in every run of consecutive turns that are all big,
    only the first KEEP_IN_RUN turns are kept (the first move of each colour); turns without a loss are not kept."""
    d = np.asarray(d, dtype=np.float64)
    has = np.isfinite(d)
    big = has & (d >= BIG)
    keep = has.copy()
    run = 0
    for t in range(len(d)):
        if big[t]:
            run += 1
            if run > KEEP_IN_RUN:
                keep[t] = False
        else:
            run = 0
    return has, big, keep


def game_v2(side, turn, logp, lmax, lead, winp):
    """One game's positions (in turn order) -> (v2 float64 [10], ok). ok False (all NaN) when the turns are not exactly
    0..T-1. Order: V2_NAMES (target side 0, then opponent side 1; each V2_PER_SIDE)."""
    side = np.asarray(side)
    turn = np.asarray(turn).astype(np.int64)
    T = len(turn)
    out = np.full(V2_DIM, np.nan)
    if T == 0 or not np.array_equal(turn, np.arange(T)):
        return out, False
    nlp = np.asarray(logp, dtype=np.float64) - np.asarray(lmax, dtype=np.float64)
    lead = np.asarray(lead, dtype=np.float64)
    winp = np.asarray(winp, dtype=np.float64)
    d = np.full(T, np.nan)
    w = np.full(T, np.nan)
    if T >= 2:
        d[:-1] = lead[:-1] + lead[1:]
        w[:-1] = winp[:-1] + winp[1:] - 1.0
    _has, big, keep = keep_mask(d)
    early = turn < T150
    dl = np.maximum(d, 0.0)
    wl = np.maximum(w, 0.0)
    for s in (0, 1):
        o = 5 * s
        ms = side == s
        if ms.any():
            out[o + 0] = nlp[ms].mean()
        m150 = ms & early
        if m150.any():
            out[o + 1] = nlp[m150].mean()
        C = m150 & keep
        if C.any():
            out[o + 2] = dl[C].mean()
            out[o + 3] = big[C].mean()
            out[o + 4] = wl[C].mean()
    return out, True


def chunk_v2(z):
    """A p30 chunk (np.load mapping) -> v2 float64 [G, 10], ok bool [G] (failed games: all NaN, ok False)."""
    if str(z["p30_format"]) != V2_FORMAT:
        raise P30Error("not a prereg30 v2 chunk")
    G = len(z["ref"])
    pg = np.asarray(z["pos_game"])
    cols = {k: np.asarray(z[k]) for k in ("pos_side", "pos_turn", "pos_logp") + EXTRA_KEYS}
    out = np.full((G, V2_DIM), np.nan)
    ok = np.zeros(G, dtype=bool)
    order = np.argsort(pg, kind="stable")
    bounds = np.searchsorted(pg[order], np.arange(G + 1))
    for gi in range(G):
        idx = order[bounds[gi]:bounds[gi + 1]]
        if len(idx) == 0:
            continue
        out[gi], ok[gi] = game_v2(cols["pos_side"][idx], cols["pos_turn"][idx], cols["pos_logp"][idx],
                                  cols["pos_lmax"][idx], cols["pos_lead"][idx], cols["pos_winp"][idx])
    return out, ok
