"""prereg53 exploration: synthetic unit tests (no real data).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg53/explore/test_p53.py
"""

import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p53_lib as C  # noqa: E402

rf = C.rf


def synth(Q=4000, seed=0, temp=2.0, shift=None):
    """True probabilities from random logits; observed = tempered / shifted version of them."""
    rng = np.random.default_rng(seed)
    z = 1.5 * rng.standard_normal((Q, C.L))
    P_true = rf.softmax(z)
    y = np.array([rng.choice(C.L, p=p) for p in P_true])
    z_obs = temp * z + (0.0 if shift is None else shift)
    return rf.softmax(z_obs), P_true, y


def test_score_matrix_matches_rule_b():
    rng = np.random.default_rng(1)
    P = rf.softmax(rng.standard_normal((500, C.L)))
    assert np.array_equal(np.argmax(P @ C.R.T, axis=1), rf.rule_b_np(P))
    for y in range(C.L):
        for k in range(C.L):
            assert C.R[k, y] == rf.official_score(np.array([k]), np.array([y]))[0]


def test_identity_at_zero():
    P, _t, _y = synth(300)
    out = C.apply(np.zeros(C.N_PARAMS), P)
    assert np.max(np.abs(out - P)) <= 1e-12


def test_fit_recovers_temperature():
    P, P_true, y = synth(100000, seed=3, temp=2.0)
    th, rec = C.fit(P, y, "full")
    assert rec["tau"] < 0.85, rec["tau"]          # observed probabilities are too sharp: tau moves below 1
    assert rec["objective_H"] >= rec["objective_H_at_0"]
    th_t, rec_t = C.fit(P, y, "temp")
    assert np.all(th_t[:C.L] == 0.0)
    th_d, rec_d = C.fit(P, y, "delta")
    assert th_d[C.L] == 0.0


def test_fit_recovers_shift_direction():
    shift = np.zeros(C.L)
    shift[0] = -1.0          # observed probabilities understate level D
    P, P_true, y = synth(8000, seed=4, temp=1.0, shift=shift)
    th, rec = C.fit(P, y, "delta")
    d = np.asarray(rec["delta"])
    assert d[0] - d[1:].mean() > 0.3, d
    before = C.rank_balanced_score(rf.rule_b_np(P), y)
    after = C.rank_balanced_score(rf.rule_b_np(C.apply(th, P)), y)
    assert after >= before


def test_no_miscalibration_little_change():
    P, P_true, y = synth(8000, seed=5, temp=1.0)
    th, rec = C.fit(P, y, "full")
    assert abs(rec["t"]) <= 0.5 and np.max(np.abs(rec["delta"])) <= 0.5, rec


def test_stack_and_stage0b():
    rng = np.random.default_rng(6)
    Q = 1200
    y = rng.integers(0, C.L, Q)
    lp = rf.log_softmax(rng.standard_normal((Q, C.L)) + 1.5 * np.eye(C.L)[y])
    keys = [f"{y[i]}:p{i // 4}" for i in range(Q)]
    res = C.stage0b(lp, y, keys, ("full", "temp"))
    assert not np.isnan(res["P0"]).any()
    assert not np.isnan(res["cand"]["full"]).any()
    # player folds: all questions of one player in one fold
    fold = res["fold"]
    for k in set(keys):
        idx = [i for i, kk in enumerate(keys) if kk == k]
        assert len(set(fold[idx])) == 1
    assert C.player_bit("A:x", C.SALT_0B) == C.player_bit("A:x", C.SALT_0B)
