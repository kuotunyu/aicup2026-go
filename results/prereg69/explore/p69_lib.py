"""prereg69 (competition 2): M8 (prereg65's same-player merge) on top of 64. numpy only; prereg64 / prereg65 code is
imported, never edited (PLAN.md, NOTES.md, frozen.json).

  m8_rows   apply prereg65's m8 to one test pool's 64 log-probabilities and give the probability rows to write
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg58", "explore"),
          os.path.join(REPO, "results", "prereg59", "explore"), os.path.join(REPO, "results", "prereg64", "explore"),
          os.path.join(REPO, "results", "prereg65", "explore")):
    if p not in sys.path:
        sys.path.insert(0, p)
import p65_lib as K65  # noqa: E402

L = K65.L
W_A_64, W_O_64 = 0.7667121789025336, 0.4940900911252534      # results/prereg64/rank/frozen_config.json


def m8_rows(L0, P0, S, qcols, rule_b, softmax):
    """L0 [Q, 10]: 64's fused logits (log_softmax(w_A A + w_O OPP3)) of the ok questions; P0 [Q, 10]: the probability
    rows 64 writes (softmax(L0)); S [Q, Q] cc_whiten_sim; qcols: 'B' / 'W'; rule_b / softmax: the writer's functions.
    -> (P_out, pred_out, info). Unchanged rows are P0's rows (same objects / bytes); a changed row is softmax(L_new).
    Raises ValueError when M8's pred0 is not rule B of P0 or a changed row's rule B is not M8's new prediction."""
    pred_out, lp_new, edges, pred0 = K65.m8(L0, S, qcols)
    pr64 = rule_b(P0)
    if not np.array_equal(pred0, pr64):
        raise ValueError(f"M8 pred0 differs from rule B of 64 on {int((pred0 != pr64).sum())} questions")
    changed = np.flatnonzero(pred_out != pred0)
    P_out = np.array(P0, np.float64, copy=True)
    if len(changed):
        P_out[changed] = softmax(lp_new[changed])
        prc = rule_b(P_out[changed])
        if not np.array_equal(prc, pred_out[changed]):
            raise ValueError("a changed row's rule-B prediction is not M8's new prediction")
    return P_out, pred_out, {"edges": int(edges), "changed": changed.tolist(), "pred0": pred0}


def m8_adjacency(L0, S, qcols):
    """The M8 edge set (for the record of groups), recomputed exactly as p65_lib.m8 builds it."""
    lp_norm = K65.log_softmax(L0)
    E_rk = np.exp(lp_norm) @ np.arange(L, dtype=np.float64)
    same_c = (np.asarray(qcols)[:, None] == np.asarray(qcols)[None, :])
    rk_ok = (np.abs(E_rk[:, None] - E_rk[None, :]) <= K65.M8_GUARD)
    adj = (S > K65.M8_THR) & same_c & rk_ok
    np.fill_diagonal(adj, False)
    return adj
