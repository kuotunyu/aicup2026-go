"""prereg30 unit tests (CPU). Fast tests: pure numpy. Model tests (marked by the checkpoint fixture): CPU fp32 on a
few positions of one fit fixture game (PLAN.md 3, seam K).

    Git Bash, repository root:
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg30/explore/test_p30.py
"""

import json
import os
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (HERE, os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import p30_lib as V  # noqa: E402

FIXTURE = os.path.join(REPO, "data", "prereg07", "fixtures", "fit_sample.json")


def leads_for(d, first=0.0):
    """Lead series whose consecutive sums are the losses d (lead_t + lead_{t+1} = d_t)."""
    lead = [first]
    for x in d:
        lead.append(x - lead[-1])
    return np.array(lead)


# ---------------------------------------------------------------- v2 features (PLAN.md 1.2)
def test_filter_paper_table1():
    # Kuboki 2024 table 1 (n = 5): losses 1 2 5 6 5 7 0 1 -> moves 5 and 6 (turns 4, 5) are dropped
    d = np.array([1, 2, 5, 6, 5, 7, 0, 1], float)
    has, big, keep = V.keep_mask(np.concatenate([d, [np.nan]]))
    assert keep[:8].tolist() == [True, True, True, True, False, False, True, True]
    assert not keep[8] and not has[8]
    T = 9
    side = np.array([0, 1] * 5)[:T]          # target = black (even turns)
    v, ok = V.game_v2(side, np.arange(T), np.zeros(T), np.zeros(T), leads_for(d), np.full(T, 0.5))
    assert ok
    assert v[2] == pytest.approx(2.0) and v[7] == pytest.approx(3.0)       # paper: black 2, white 3
    assert v[3] == pytest.approx(1 / 3) and v[8] == pytest.approx(1 / 3)   # big kept: black t2, white t3
    assert v[4] == 0.0 and v[9] == 0.0                                      # winp 0.5 + 0.5 - 1 = 0


def test_nlp_and_150_limit():
    T = 160
    side = np.arange(T) % 2
    logp = np.full(T, -2.0)
    lmax = np.full(T, -1.0)
    lmax[150:] = -0.5                          # nlp -1 before 150, -1.5 after
    v, ok = V.game_v2(side, np.arange(T), logp, lmax, np.zeros(T), np.full(T, 0.5))
    assert ok
    assert v[1] == pytest.approx(-1.0) and v[6] == pytest.approx(-1.0)
    assert v[0] == pytest.approx((150 * -1.0 + 10 * -1.5) / 160)


def test_last_move_has_no_loss_and_negative_loss_clipped():
    d = np.array([-3.0, 2.0])
    v, ok = V.game_v2(np.array([0, 1, 0]), np.arange(3), np.zeros(3), np.zeros(3), leads_for(d), np.zeros(3) + 0.5)
    assert ok and v[2] == 0.0 and v[7] == 2.0
    # target side: only turn 0 counted (turn 2 is the last move)


def test_turns_not_contiguous_gives_nan():
    v, ok = V.game_v2(np.array([0, 1]), np.array([0, 2]), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    assert not ok and np.isnan(v).all()


def test_chunk_v2_groups_by_game():
    d = np.array([1, 2, 5, 6, 5, 7, 0, 1], float)
    lead = leads_for(d)
    z = {"p30_format": np.array(V.V2_FORMAT), "ref": np.array([10, 11, 12]),
         "pos_game": np.concatenate([np.full(9, 0), np.full(9, 2)]).astype(np.int32),
         "pos_side": np.tile(np.array([0, 1] * 5)[:9], 2).astype(np.int8),
         "pos_turn": np.tile(np.arange(9), 2).astype(np.int16), "pos_logp": np.zeros(18, np.float32),
         "pos_lmax": np.zeros(18, np.float32), "pos_lead": np.tile(lead, 2).astype(np.float32),
         "pos_winp": np.full(18, 0.5, np.float32)}
    v, ok = V.chunk_v2(z)
    assert ok.tolist() == [True, False, True]
    assert np.isnan(v[1]).all() and v[0, 2] == pytest.approx(2.0) and v[2, 7] == pytest.approx(3.0)


def test_guard():
    with pytest.raises(SystemExit):
        V.guard(os.path.join(REPO, "data", "local", "x"))


# ---------------------------------------------------------------- evaluation (PLAN.md 4)
@pytest.fixture(scope="module")
def ev():
    import p30_eval as E
    return E


def test_gradient(ev):
    rng = np.random.default_rng(1)
    Q, d = 40, 3
    LP = np.log(rng.dirichlet(np.ones(10), Q))
    Z = rng.standard_normal((Q, d))
    y = rng.integers(10, size=Q)
    th = rng.standard_normal(10 + 2 * d) * 0.3
    f, g = ev.objective(th, LP, Z, y, 0.01)
    num = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = 1e-6
        num[i] = (ev.objective(th + e, LP, Z, y, 0.01)[0] - ev.objective(th - e, LP, Z, y, 0.01)[0]) / 2e-6
    assert np.allclose(g, num, atol=1e-6)


def test_m0_is_m1_without_z(ev):
    rng = np.random.default_rng(2)
    LP = np.log(rng.dirichlet(np.ones(10), 60))
    y = rng.integers(10, size=60)
    a, _ = ev.fit(LP, np.zeros((60, 0)), y)
    b, _ = ev.fit(LP, np.zeros((60, 2)), y)
    assert np.allclose(a, b[:10], atol=1e-5)


def test_rule_b_equals_decide(ev):
    import decide
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(10), 50)
    got = ev.rule_b(np.log(p))
    want = [decide.argmax_first(decide.b_scores(list(r))) for r in p]
    assert got.tolist() == want


def test_bootstrap_constant_diff(ev):
    Q = {"player": np.array(["a", "a", "b", "c"]), "y": np.array([0, 0, 0, 1])}
    out = ev.bootstrap({"x": np.array([0.5, 0.5, 0.5, 0.5])}, Q, 20)
    assert np.allclose(out["x"], 0.5)


def test_fold_deterministic(ev):
    assert ev.fold_of("1D:abc") == ev.fold_of("1D:abc")
    assert {ev.fold_of(f"A:{i}") for i in range(200)} == set(range(5))


# ---------------------------------------------------------------- scorer (PLAN.md 3, seam K)
@pytest.fixture(scope="module")
def model():
    import p30_scorer as S
    if not os.path.exists(S.H.CKPT_PATH):
        pytest.skip("checkpoint missing")
    return S, S.P.load_model(device="cpu")


def fixture_game():
    if not os.path.exists(FIXTURE):
        pytest.skip("fixture missing")
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)["games"][0]


def test_scorer_p1_bitwise_and_extras(model):
    import torch
    S, m = model
    P = S.P
    g = fixture_game()
    f = P.featurize_game(g["sgf_content"], g["color"])
    sub = {k: (v[:9] if isinstance(v, np.ndarray) else v) for k, v in f.items()}
    a = P.ProbeScorer(m, device="cpu", precision="fp32", batch=4, pad=True)
    ra = a.position_features(sub)
    a.close()
    b = S.V2Scorer(m, device="cpu", precision="fp32", batch=4, pad=True)
    rb_ = b.position_features(sub)
    ex = b.last_extra
    b.close()
    assert np.array_equal(ra, rb_)                       # P1 part bit for bit
    assert ex.shape == (9, 3) and np.isfinite(ex).all()
    # lmax: max of the masked log-softmax, recomputed independently
    with torch.inference_mode():
        out = m(torch.as_tensor(sub["bin"]).float(), torch.as_tensor(sub["glob"]).float(),
                input_meta=torch.tensor(P.neutral_table())[torch.as_tensor(sub["pla"].astype(np.int64))])
        lg = out[0][0][:, 0, :].masked_fill(~torch.as_tensor(sub["legal"]), float("-inf"))
        lmax = torch.log_softmax(lg, -1).max(1).values.numpy()
    assert np.allclose(ex[:, 0], lmax, atol=1e-5)
    assert (ra[:, 768] - ex[:, 0] <= 1e-6).all()        # nlp <= 0
    # lead / value against KataGo's own GameState.get_model_outputs (postprocess_output)
    from katago.game.gamestate import GameState
    moves = S.H.parse_moves(g["sgf_content"])
    for i in range(0, 9, 4):
        gs = GameState(S.H.POS_LEN, S.H.RULES)
        for pla, x, y in moves[:int(sub["turn"][i])]:
            gs.play(pla, gs.board.loc(x, y))
        o = gs.get_model_outputs(m, sgfmeta=P.neutral_sgfmeta())
        assert abs(float(o["lead"]) - float(ex[i, 1])) <= 1e-4
        assert abs(float(o["value"][0]) - float(ex[i, 2])) <= 1e-4


def test_scorer_unpadded_batch_same(model):
    S, m = model
    P = S.P
    g = fixture_game()
    f = P.featurize_game(g["sgf_content"], g["color"])
    sub = {k: (v[:7] if isinstance(v, np.ndarray) else v) for k, v in f.items()}
    a = S.V2Scorer(m, device="cpu", precision="fp32", batch=4, pad=True)
    a.position_features(sub)
    ea = a.last_extra
    a.close()
    b = S.V2Scorer(m, device="cpu", precision="fp32", batch=64, pad=False)
    b.position_features(sub)
    eb = b.last_extra
    b.close()
    assert np.allclose(ea, eb, atol=1e-4)
