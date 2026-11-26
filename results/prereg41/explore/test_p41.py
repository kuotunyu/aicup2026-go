"""prereg41 exploration: synthetic unit tests (no real data).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg41/explore/test_p41.py
"""

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p41_lib as K  # noqa: E402

p4, rf = K.p4, K.rf
J = 3


def synth(Q=600, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(5, 21, Q).astype(np.float64)
    y = rng.integers(0, K.L, Q)
    m = rng.standard_normal((Q, K.L))
    Es = [rng.standard_normal((Q, K.L)) * (1 + j) for j in range(J)]
    # make the third term informative mostly for large n
    Es[2][np.arange(Q), y] += 0.3 * n / 12.0
    return m, Es, n, y


def test_logits_equal_p04_when_gammas_tied():
    m, Es, n, _y = synth()
    rng = np.random.default_rng(1)
    th = np.concatenate([[0.3, 0.7], rng.uniform(0.1, 2, J), rng.standard_normal(K.L - 1)])
    z0, _ = p4.gen_logits(th, m, Es, n)
    for free in ([True] * J, [False] * J, [False, False, True]):
        z = K.logits(K.from_base(th, J, free), m, Es, n, free)
        assert np.max(np.abs(z - z0)) <= 1e-12


def test_objective_equals_p04_when_tied():
    m, Es, n, y = synth()
    rng = np.random.default_rng(2)
    th = np.concatenate([[0.1, -0.3], rng.uniform(0.1, 2, J), rng.standard_normal(K.L - 1)])
    wq = rf.level_weights(y)
    l0, g0 = p4.gen_obj(th, m, Es, n, y, wq)
    free = [False] * J
    l1, g1 = K.objective(K.from_base(th, J, free), m, Es, n, y, wq, free)
    assert abs(l1 - l0) <= 1e-9 * abs(l0)
    assert np.allclose(g1[:2 + J], g0[:2 + J], rtol=1e-9, atol=1e-9)
    assert np.allclose(g1[2 + 2 * J:], g0[2 + J:], rtol=1e-9, atol=1e-9)
    assert np.all(g1[2 + J:2 + 2 * J] == 0)


def test_gradient_numeric():
    m, Es, n, y = synth(300, 3)
    wq = rf.level_weights(y)
    rng = np.random.default_rng(4)
    for free in ([True] * J, [False, False, True]):
        th = np.concatenate([[0.2, 0.4], rng.uniform(0.2, 1.5, J), rng.uniform(-0.5, 1.0, J) * np.array(free),
                             0.3 * rng.standard_normal(K.L - 1)])
        _l, g = K.objective(th, m, Es, n, y, wq, free)
        num = np.zeros_like(th)
        for i in range(len(th)):
            e = np.zeros_like(th)
            e[i] = 1e-6
            num[i] = (K.objective(th + e, m, Es, n, y, wq, free)[0]
                      - K.objective(th - e, m, Es, n, y, wq, free)[0]) / 2e-6
        assert np.max(np.abs(num - g) / (np.abs(num) + 1e-3)) < 1e-5


def test_fit_not_worse_than_base_and_recovers_gammas():
    rng = np.random.default_rng(5)
    Q = 6000
    n = rng.integers(5, 21, Q).astype(np.float64)
    s = n / 12.0
    E1, E2 = rng.standard_normal((Q, K.L)), rng.standard_normal((Q, K.L))
    z = (s ** 1.2)[:, None] * E1 + (s ** -0.5)[:, None] * E2          # true per-term scalings differ
    p = rf.softmax(z)
    y = np.array([rng.choice(K.L, p=row) for row in p])
    m = np.zeros((Q, K.L))
    Es = [E1, E2]
    th_b = p4.gen_fit(m, Es, n, y)
    free = [True, True]
    th, rec = K.fit_record(m, Es, n, y, free, K.from_base(th_b, 2, free))
    assert rec["loss"] < K.base_loss(th_b, m, Es, n, y)
    _a, _g0, _w, g, _beta = K.unpack(th, 2)
    assert abs(g[0] - 1.2) < 0.25 and abs(g[1] + 0.5) < 0.25, g


def test_bounds_pin_tied_gammas():
    b = K.bounds(J, [False, True, False])
    assert b[2 + J] == (0.0, 0.0) and b[2 + J + 1] == rf.BOUNDS["gamma"] and len(b) == 2 + 2 * J + K.L - 1
