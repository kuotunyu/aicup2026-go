"""prereg26 exploration: synthetic unit tests (no real data, no data/local).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg26/explore/test_p26.py
"""

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p26_lib as L  # noqa: E402
import p09_forward as FW  # noqa: E402
import p09_rank_fuse as r  # noqa: E402


def _arrays(n, seed):
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, FW.X_DIM)).astype(np.float32)
    X[g.random(n) < 0.1, FW.F_DIM:] = np.nan            # games without an opponent move
    EX = g.standard_normal((n, FW.NEX)).astype(np.float32)
    EX[g.random((n, FW.NEX)) < 0.05] = np.nan
    return X, EX


def test_moments_are_09s_in_memory_formula():
    X, EX = _arrays(20000, 1)
    m = L.moments(r, X, EX, np.arange(len(X)))
    mu, sd = FW.fix_x_moments(np.nanmean(X.astype(np.float64), 0), np.nanstd(X.astype(np.float64), 0))
    assert np.allclose(m["mu_x"], mu, rtol=0, atol=1e-12) and np.allclose(m["sd_x"], sd, rtol=0, atol=1e-12)
    assert m["rows"] == 20000


def test_moments_of_a_prefix_ignore_the_appended_rows():
    X1, EX1 = _arrays(9000, 2)
    X2, EX2 = _arrays(3000, 3)
    a = L.moments(r, X1, EX1, np.arange(len(X1)))
    b = L.moments(r, np.concatenate([X1, X2]), np.concatenate([EX1, EX2]), np.arange(len(X1)))
    assert all(L.moments_equal(a, b).values())
    c = L.moments(r, np.concatenate([X1, X2]), np.concatenate([EX1, EX2]), np.arange(len(X1) + len(X2)))
    assert not L.moments_equal(a, c)["mu_x"]


def test_player_folds_cover_every_player_once():
    keys = np.array([f"A:p{i % 37}" for i in range(500)])
    f = FW.player_folds(keys)
    for k in set(keys.tolist()):
        assert len(set(f[keys == k].tolist())) == 1
    assert set(f.tolist()) == {0, 1, 2, 3, 4}


def test_player_ids_and_guard():
    assert L.player_ids(["A:x", "3D:y", "D:x"]) == {"x", "y"}
    try:
        L.guard(os.path.join("data", "lo" + "cal", "x.csv"))
    except SystemExit:
        return
    raise AssertionError("guard did not refuse")
