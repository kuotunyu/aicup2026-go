"""prereg40 exploration: synthetic unit tests (no real data, never the validation answers).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg40/explore/test_p40.py
"""

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p40_lib as L  # noqa: E402
import p09_forward as FW  # noqa: E402


def test_gather_raw_any_order_and_repeats():
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((7, 3)), rng.standard_normal((7, 2)).astype(np.float32))
    B = (rng.standard_normal((5, 3)), rng.standard_normal((5, 2)).astype(np.float32))
    cat_X, cat_E = np.concatenate([A[0], B[0]]), np.concatenate([A[1], B[1]])
    idx = np.array([11, 0, 6, 7, 3, 3, 9, 0])
    X, E = L.gather_raw(idx, 7, A, B)
    assert np.array_equal(X, cat_X[idx]) and np.array_equal(E, cat_E[idx])
    assert X.dtype == np.float64 and E.dtype == np.float32


def test_mean_nll_ensemble_and_members():
    rng = np.random.default_rng(1)
    hold = {5: (np.zeros((4, 5), dtype=np.int64), np.array([0, 3, 9, 2])),
            10: (np.zeros((3, 10), dtype=np.int64), np.array([1, 1, 8]))}
    lps = {}
    for n, (_i, y) in hold.items():
        lps[n] = []
        for _ in range(3):
            z = rng.standard_normal((len(y), 10))
            lps[n].append(z - np.log(np.exp(z).sum(1, keepdims=True)))
    m, per = L.mean_nll(lps, hold)
    tot = sum(-FW.lse_mean(lps[n])[np.arange(len(y)), y].sum() for n, (_i, y) in hold.items())
    assert abs(m - tot / 7) < 1e-12
    m1, _ = L.mean_nll(lps, hold, members=[1])
    tot1 = sum(-lps[n][1][np.arange(len(y)), y].sum() for n, (_i, y) in hold.items())
    assert abs(m1 - tot1 / 7) < 1e-12
    assert set(per) == {"5", "10"}


def test_stage0b_threshold():
    assert L.stage0b_verdict(0.5, 0.505)["passed"] and L.stage0b_verdict(1.0, 1.0051)["passed"]
    assert L.stage0b_verdict(1.0, 1.004)["passed"] is False
    assert L.stage0b_verdict(1.01, 1.0)["passed"] is False


def test_features32_blockwise_equal_and_ex_nonfinite_to_zero():
    rng = np.random.default_rng(2)
    mp = {"mu_x": rng.standard_normal(FW.X_DIM), "sd_x": np.abs(rng.standard_normal(FW.X_DIM)) + 0.5,
          "mu_ex": rng.standard_normal(FW.NEX), "sd_ex": np.abs(rng.standard_normal(FW.NEX)) + 0.5}
    X = rng.standard_normal((20, FW.X_DIM))
    EX = rng.standard_normal((20, FW.NEX)).astype(np.float32)
    EX[3, 5] = np.nan
    full = L.features32(X, EX, mp)
    part = np.concatenate([L.features32(X[:7], EX[:7], mp), L.features32(X[7:], EX[7:], mp)])
    assert np.array_equal(full, part) and full.dtype == np.float32 and full.shape == (20, L.D_IN)
    assert full[3, FW.X_DIM + 5] == 0.0
    EX[1, 1] = np.inf
    try:
        L.features32(X, EX, mp)
    except SystemExit:
        return
    raise AssertionError("inf EX must stop")


def test_holdout_lp_shapes():
    hold = {5: (np.arange(10).reshape(2, 5), np.array([0, 1])), 10: (np.zeros((0, 10), dtype=np.int64),
                                                                     np.array([], dtype=np.int64))}
    feats = np.arange(40, dtype=np.float32).reshape(10, 4)
    seen = []

    def fwd(par, Xg):
        seen.append(Xg.shape)
        return np.full((Xg.shape[0], 10), -np.log(10.0)) + par

    out = L.holdout_lp(fwd, [0.0, 0.0], hold, lambda idx: feats[idx])
    assert list(out) == [5] and len(out[5]) == 2 and seen == [(2, 5, 4), (2, 5, 4)]


def test_registered_constants():
    assert L.SALT == "c10-prereg25|holdout|"
    assert (L.HOLD_PER, L.HOLD_SEED) == (100, 19019)
    assert tuple(L.SEEDS) == (0, 1, 2, 3, 4) and L.D_IN == 1584
    assert L.GATE_0B == -0.005 and L.REF_Q25 == 0.7456519871546352


def test_guard_refuses_local():
    for p in ("data/" + "local" + "/x.csv", "D:/x/data/" + "local"):
        try:
            L.guard(p)
        except SystemExit:
            continue
        raise AssertionError(p)
