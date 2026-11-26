"""prereg44: synthetic unit tests (no real data, no GPU).

    .venv/Scripts/python.exe -m pytest -q results/prereg44/explore/test_p44.py
"""
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p44_lib as K  # noqa: E402


def fake_sgf(rng, n_moves=40):
    pts = rng.choice(361, size=n_moves, replace=False)
    out = "(;SZ[19]"
    for i, p in enumerate(pts.tolist()):
        r, c = divmod(p, 19)
        out += ";%s[%s%s]" % ("B" if i % 2 == 0 else "W", chr(97 + c), chr(97 + r))
    return out + ")"


def fake_rows(rng, players_per_level=3, games=6):
    rows = []
    for lv in K.LEVELS:
        for p in range(players_per_level):
            n = games if p else 3          # the first player of every level has < MIN_PLAYER_GAMES games
            for g in range(n):
                rows.append({"player_id": "p%s_%d" % (lv, p), "game_id": "g_%07d" % (len(rows) + 1), "rank": lv,
                             "color": "B" if g % 2 == 0 else "W", "sgf_content": fake_sgf(rng)})
    return rows


@pytest.fixture(scope="module")
def cache():
    d = tempfile.mkdtemp(prefix="p44test_")
    rows = fake_rows(np.random.default_rng(0))
    K.build_cache(iter(rows), os.path.join(d, "c"), "train", "synthetic")
    return K.open_cache(os.path.join(d, "c"))


def test_guard():
    for p in ("data/local/x.csv", "data/prereg36/cache/codes.npy", "data/prereg42/emb", "data/prereg06/feats/meta.npz",
              "x/rank_val_answers.csv"):
        with pytest.raises(SystemExit):
            K.guard(os.path.join(K.REPO, p))
    for p in ("data/prereg44/cache/train", "data/training/train_D.csv", "data/prereg25/v/sets.json",
              "data/prereg17/prep/probe_meta.npz", "data/prereg04/x"):
        assert K.guard(os.path.join(K.REPO, p))


def test_cache_and_level_index(cache):
    assert cache.n == 10 * (3 + 2 * 6)
    assert not cache.bad.any()
    idx = K.level_index(cache)
    assert idx["dropped_players"] == 10                        # every level's 3-game player
    assert [len(b) for b in idx["by_level"]] == [2] * 10
    for p in range(len(idx["p_off"]) - 1):
        g = idx["games"][idx["p_off"][p]:idx["p_off"][p + 1]]
        assert len(set(cache.pid[g].tolist())) == 1 and len(g) == 6


def test_sampler_deterministic_and_levels(cache):
    idx = K.level_index(cache)
    ds = K.RankBatches(cache.dir, cache.off, cache.color == "W", idx, batch_games=8, moves=5)
    a, b = ds.batch(3), ds.batch(3)
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert a["x"].shape[1:] == (17, 19, 19) and a["x"].dtype == np.uint8
    assert len(a["y"]) == 8 and a["seg"].max() == 7
    assert np.bincount(a["seg"]).max() <= 5
    ys = np.concatenate([ds.batch(s)["y"] for s in range(60)])
    assert set(ys.tolist()) == set(range(10))
    c = ds.batch(4)
    assert not np.array_equal(a["y"], c["y"]) or not np.array_equal(a["x"], c["x"])


def test_model_and_inference(cache):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = K.build().eval()
    idx = K.level_index(cache)
    ds = K.RankBatches(cache.dir, cache.off, cache.color == "W", idx, batch_games=3, moves=4)
    b = ds[0]
    with torch.no_grad():
        z = net(b["x"].float(), b["seg"], 3)
        e = net.trunk(b["x"].float())
        man = net.head(torch.stack([e[b["seg"] == i].mean(0) for i in range(3)]))
    assert z.shape == (3, 10)
    assert torch.allclose(z, man, atol=1e-5)

    class Sub:
        dir, off, n, meta = cache.dir, cache.off, 5, cache.meta
        color = cache.color[:5]
    lg, pos, bad = K.infer_logits(net, Sub, "cpu", workers=0, chunk=2, max_positions=50)
    assert lg.shape == (5, 10) and not bad.any()
    from p36_lib import planes_for_codes
    for g in range(5):
        x, _t = planes_for_codes(cache.game(g), "w" if cache.color[g] == "W" else "b")
        assert pos[g] == len(x)
        with torch.no_grad():
            one = net(torch.from_numpy(x).float(), torch.zeros(len(x), dtype=torch.long), 1)[0].numpy()
        assert np.allclose(lg[g], one, atol=1e-4)


def test_ema():
    torch = pytest.importorskip("torch")
    net = K.build()
    ema = K.Ema(net, decay=0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    k = "head.bias"
    before = ema.params[k].clone()
    ema.update(net)
    assert torch.allclose(ema.params[k], 0.5 * before + 0.5 * dict(net.named_parameters())[k].detach())
    sd = ema.state_dict()
    net2 = K.build()
    net2.load_state_dict(sd)


def test_stack_gradient_and_crossfit():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.integers(0, 10, n)
    lp25 = K.log_softmax(rng.standard_normal((n, 10)) + 1.5 * np.eye(10)[y])
    E = K.log_softmax(rng.standard_normal((n, 10)) + 1.0 * np.eye(10)[y])
    sn = rng.choice([5, 10, 15, 20], n)[:, None] / 12.0
    Fs = [lp25, E, sn * E]
    th = rng.standard_normal(3 + 9)
    w = K.stack_weights(y)
    f, g = K.stack_objective(th, Fs, y, w)
    num = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = 1e-6
        num[i] = (K.stack_objective(th + e, Fs, y, w)[0] - K.stack_objective(th - e, Fs, y, w)[0]) / 2e-6
    assert np.allclose(num, g, rtol=1e-4, atol=1e-7)
    fold = K.player_fold(["D:p%d" % i for i in range(n)])
    assert set(fold.tolist()) == {0, 1}
    base, _ = K.crossfit_stack([lp25], y, fold)
    cand, _ = K.crossfit_stack(Fs, y, fold)
    assert K.mean_nll(cand, y) < K.mean_nll(base, y) - 0.05
    noise, _ = K.crossfit_stack([lp25, rng.standard_normal((n, 10)), rng.standard_normal((n, 10))], y, fold)
    assert K.mean_nll(noise, y) > K.mean_nll(base, y) - 0.01


def test_balanced_game_nll():
    lp = np.log(np.full((4, 10), 0.1))
    y = np.array([0, 0, 0, 1])
    assert abs(K.balanced_game_nll(lp, y) - np.log(10)) < 1e-12
    lp2 = lp.copy()
    lp2[3] = K.log_softmax(np.eye(10)[1:2] * 50)
    assert K.balanced_game_nll(lp2, y) < np.log(10) / 2 + 1e-6
