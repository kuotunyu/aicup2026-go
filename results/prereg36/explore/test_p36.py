"""prereg36 explore: unit tests (CPU, fast).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -rs results/prereg36/explore/test_p36.py

Real data used: the first rows of data/training/train_1D.csv (planes against the tutorial implementation), the fit
feature store's meta.npz and the move cache (pool shapes, splits). No retrieval score on real data; the scoring
code is exercised on synthetic arrays only. The validation answers are never touched.
"""
import csv
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "third_party", "aicup2026_tutorial"))
import p36_lib as L  # noqa: E402

TRAIN_1D = os.path.join(REPO, "data", "training", "train_1D.csv")
HAVE_CACHE = os.path.exists(os.path.join(L.CACHE, "cache.json"))
HAVE_FEATS = os.path.exists(os.path.join(L.FEATS, "meta.npz"))
need_cache = pytest.mark.skipif(not (HAVE_CACHE and HAVE_FEATS), reason="move cache / feature store not built")


def tutorial(sgf, colour):
    from utils import SGFParsePlayerIdentification
    out = SGFParsePlayerIdentification().extract_features(sgf, colour.lower(), mode="inference")
    return np.asarray(out, np.uint8).reshape(len(out), 17, 19, 19)


def first_games(n):
    csv.field_size_limit(10 ** 8)
    rows = []
    with open(TRAIN_1D, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if len(rows) >= n:
                break
    return rows


# ---------------------------------------------------------------- planes
@pytest.mark.skipif(not os.path.exists(TRAIN_1D), reason="training data not present")
def test_planes_equal_tutorial_on_real_games():
    rows = first_games(40)
    assert len(rows) == 40
    with_captures = 0
    for r in rows:
        codes = L.moves_of(r["sgf_content"])
        S = L.replay(codes)
        stones = np.abs(S).sum(1)
        with_captures += int((np.diff(stones) < 1).any())
        for colour in ("B", "W"):
            mine = L.planes_for_game(r["sgf_content"], colour)
            ref = tutorial(r["sgf_content"], colour)
            assert mine.dtype == np.uint8 and mine.shape == ref.shape
            assert np.array_equal(mine, ref)
            assert len(mine) == len(L.target_moves(codes, colour)) > 0
    assert with_captures >= 30          # the comparison covers games with captures


@pytest.mark.skipif(not os.path.exists(TRAIN_1D), reason="training data not present")
def test_subset_is_rows_of_the_full_output():
    rng = np.random.default_rng(1)
    for r in first_games(8):
        for colour in ("B", "W"):
            full, t_all = L.planes_for_codes(L.moves_of(r["sgf_content"]), colour)
            sub, t = L.planes_for_game_subset(r["sgf_content"], colour, 24, rng)
            assert len(t) == min(24, len(t_all)) and len(set(t.tolist())) == len(t)
            assert np.array_equal(sub, full[np.searchsorted(t_all, t)])


CAPTURE = "(;SZ[19];B[aa];W[ba];B[cc];W[ab];B[dd];W[ee];B[aa];W[ff];B[gg];W[hh];B[ii];W[jj];B[kk];W[ll];B[mm])"
# suicide of one stone (sgfmill removes it) and a snapback-like multi-stone capture, then more moves
SUICIDE = ("(;SZ[19];B[ba];W[qq];B[ab];W[aa];B[cc];W[dd];B[ee];W[ff];B[gg];W[hh];B[ii];W[jj];B[kk];W[ll];B[mm]"
           ";W[aa];B[nn];W[oo])")


@pytest.mark.parametrize("sgf", [
    CAPTURE, SUICIDE,
    "(;GM[1]FF[4]SZ[19]HD[0]KM[6.5];B[pd];W[dp];B[];W[tt];B[pp];W[dd];B[fc];W[cf];B[jd];W[qf];B[nc];W[pj])",
    "(;SZ[19]\n;B[pd]\n;W[dp];B[pp];W[dd];B[fc];W[cf];B[jd];W[qf];B[nc];W[pj];B[qh]C[a [note\\]];W[ql])",
    "(;SZ[19];B[pd];W[dp];B[pp])",
])
def test_planes_equal_tutorial_on_special_sgf(sgf):
    for colour in ("B", "W"):
        assert np.array_equal(L.planes_for_game(sgf, colour), tutorial(sgf, colour))


def test_sample_is_the_position_before_the_targets_move():
    sgf = "(;SZ[19];B[aa];W[bb];B[cc];W[dd];B[ee];W[ff];B[gg];W[hh];B[ii])"
    x = L.planes_for_game(sgf, "W")           # the 8th non-pass move is W[hh]
    assert x.shape == (1, 17, 19, 19)
    own_now, opp_now = x[0, 7], x[0, 15]
    assert own_now.sum() == 3 and opp_now.sum() == 4      # W[hh] itself is not on the board yet
    assert x[0, 16].sum() == 0 and x[0, 0].sum() == 0 and x[0, 8].sum() == 0
    b = L.planes_for_game(sgf, "B")           # B[ii] is the 9th move
    assert b.shape[0] == 1 and b[0, 16].all() and b[0, 7].sum() == 4 and b[0, 15].sum() == 4


def test_occupied_point_raises_like_sgfmill():
    with pytest.raises(ValueError):
        L.replay(L.moves_of("(;SZ[19];B[aa];W[aa])"))


# ---------------------------------------------------------------- symmetries
def test_symmetries():
    a = np.arange(361).reshape(19, 19)
    outs = [L.sym(a, s) for s in range(8)]
    assert np.array_equal(outs[0], a)
    assert len({o.tobytes() for o in outs}) == 8
    for o in outs:
        assert o.shape == (19, 19) and sorted(o.ravel().tolist()) == list(range(361))
        assert o[9, 9] == a[9, 9]                                    # the centre stays
        assert {int(o[0, 0]), int(o[0, 18]), int(o[18, 0]), int(o[18, 18])} == {0, 18, 342, 360}
    # closed under composition (a group of 8)
    keys = {o.tobytes() for o in outs}
    for s in range(8):
        for t in range(8):
            assert L.sym(outs[s], t).tobytes() in keys
    x = np.random.default_rng(0).integers(0, 2, (5, 17, 19, 19)).astype(np.uint8)
    for s in range(8):
        y = L.sym(x, s)
        assert y.shape == x.shape
        assert np.array_equal(y[3, 11], L.sym(x[3, 11], s))


# ---------------------------------------------------------------- splits
def test_split_players_synthetic():
    from c10data import LEVELS
    store, fit = {}, {}
    for li, lv in enumerate(LEVELS):
        for i in range(80 if lv == "1D" else 170):
            store["s%02d_%03d" % (li, i)] = lv
    for p in store:
        fit[p] = 105
    for i in range(500):
        fit["small_%03d" % i] = 10 + i % 30
    E, D, T, per = L.split_players(store, fit)
    assert (E, D, T, per) == L.split_players(dict(reversed(list(store.items()))), fit)      # deterministic
    assert len(E) == 600 and all(per[lv]["E"] == 60 for lv in LEVELS)
    assert per["1D"]["D"] == 20 and all(per[lv]["D"] == 30 for lv in LEVELS if lv != "1D") and len(D) == 290
    assert not set(E) & set(D) and not set(T) & (set(E) | set(D))
    assert all(fit[p] >= 20 for p in T)
    assert set(T) == {p for p in fit if p not in set(E) | set(D) and fit[p] >= 20}


@need_cache
def test_real_splits_and_trainset():
    import json
    from c10data import LEVELS
    import p36_train as tr
    cache = L.Cache()
    E, D, T, per = L.splits(cache)
    assert (E, D, T, per) == L.splits(cache)
    assert len(E) == 600 and all(per[lv]["E"] == 60 for lv in LEVELS)
    assert all(per[lv]["D"] == min(30, per[lv]["store"] - 60) for lv in LEVELS) and len(D) <= 300
    assert not set(E) & set(D) and not set(T) & (set(E) | set(D))
    rec_path = os.path.join(HERE, "split_record.json")
    if os.path.exists(rec_path):
        with open(rec_path, encoding="utf-8") as f:
            rec = json.load(f)
        assert rec == L.split_record(None, cache)
    ds, T2, _sha = tr.make_trainset(cache, batch_games=6, moves=5)
    assert T2 == T and len(ds.t_off) == len(T) + 1 and (np.diff(ds.t_off) >= 20).all()
    held = set(E) | set(D)
    assert not held & set(np.unique(cache.pid[ds.t_games]).tolist())
    x, seg, y = ds.batch(7)
    x2, seg2, y2 = ds.batch(7)
    assert np.array_equal(x, x2) and np.array_equal(seg, seg2) and np.array_equal(y, y2)
    assert not np.array_equal(y, ds.batch(8)[2])
    assert x.dtype == np.uint8 and x.shape[1:] == (17, 19, 19) and len(seg) == len(x) <= 30
    assert sorted(set(seg.tolist())) == list(range(6)) and len(y) == 6 and (0 <= y).all() and (y < len(T)).all()
    assert np.bincount(seg).max() <= 5


# ---------------------------------------------------------------- pools
@need_cache
def test_pool_shapes():
    import p06_sim as sim
    import p35_sim as ps
    import p36_eval as ev
    store = sim.Store(L.FEATS, load=())
    E, D, _T, _per = L.splits()
    ref = ps.v2new_pool(store, 3)             # the same construction when every store player is allowed
    mine = ev.make_pool(store, 3, set(store.by_player()), ps.PER_LEVEL, ps.K_LIST)
    assert mine["pool"] == ref["pool"] and mine["cand_rows"] == ref["cand_rows"] and mine["q_rows"] == ref["q_rows"]
    assert np.array_equal(mine["owner"], ref["owner"]) and np.array_equal(mine["k"], ref["k"])
    for which, seed in (("D", 100), ("E", 0), ("E", 19)):
        _seeds, players, per_level, k_list, short = ev.pool_spec(which, E, D)
        rec = ev.make_pool(store, seed, players, per_level, k_list, short)
        assert set(rec["pool"]) <= set(players) and len(set(rec["pool"])) == len(rec["pool"])
        assert len(rec["pool"]) == (len(D) if which == "D" else 600)
        assert len(rec["q_rows"]) == (200 if which == "D" else 400)
        assert sorted(np.bincount(rec["k"])[[1, 2, 3, 5]].tolist()) == [len(rec["q_rows"]) // 4] * 4
        assert all(len(g) == 10 for g in rec["cand_rows"])
        assert [len(g) for g in rec["q_rows"]] == rec["k"].tolist()
        rows = [r for g in rec["cand_rows"] + rec["q_rows"] for r in g]
        assert len(set(rows)) == len(rows)
        for q, o in zip(rec["q_rows"], rec["owner"].tolist()):
            assert {str(store.pid[r]) for r in q} == {rec["pool"][o]}
        for g, p in zip(rec["cand_rows"], rec["pool"]):
            assert {str(store.pid[r]) for r in g} == {p}
    assert ev.D_SEEDS == list(range(100, 110)) and ev.E_SEEDS == list(range(20))


# ---------------------------------------------------------------- model
def test_model_forward_backward_and_segment_pooling():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net, head = L.build_model(7)
    assert len(net.layers) == 10 and net.conv_in.in_channels == 17 and net.conv_in.out_channels == 128
    assert net.fc.in_features == 128 and net.fc.out_features == 256 and head.weight.shape == (7, 256)
    x = torch.randint(0, 2, (10, 17, 19, 19)).float()
    seg = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    y = torch.tensor([1, 5, 0, 6])
    e = net(x)
    assert e.shape == (10, 256)
    m = L.segment_mean(e, seg, 4)
    for s in range(4):
        assert torch.allclose(m[s], e[seg == s].mean(0), atol=1e-5)
    assert torch.equal(L.segment_mean(e, seg, 5)[4], torch.zeros(256))
    emb = L.embed_segments(net, x, seg, 4)
    assert torch.allclose(emb.norm(dim=1), torch.ones(4), atol=1e-5)
    logits = head(emb)
    assert logits.shape == (4, 7) and float(logits.detach().abs().max()) <= L.SCALE + 1e-4
    w = F.normalize(head.weight, dim=1)
    assert torch.allclose(logits, 30.0 * emb @ w.t(), atol=1e-4)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=1e-3)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    before = net.fc.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, net.fc.weight) and torch.isfinite(loss)


# ---------------------------------------------------------------- scoring code on synthetic arrays
def synthetic_pool(rng, nc, nq, signal):
    import p35_sim as ps
    owner = rng.integers(0, nc, nq)
    ks = np.array([1, 2, 3, 5] * (nq // 4))
    T = {}
    for n in ps.TERMS:
        M = rng.normal(size=(nq, nc))
        M[np.arange(nq), owner] += 1.0
        T[n] = (M, np.ones(nq, bool))
    dim = 16
    cvec = rng.normal(size=(nc, dim))
    emb, cand_rows, q_rows = [], [], []
    for c in range(nc):
        cand_rows.append(list(range(len(emb), len(emb) + 3)))
        emb += [cvec[c] + rng.normal(size=dim) for _ in range(3)]
    for q in range(nq):
        q_rows.append(list(range(len(emb), len(emb) + int(ks[q]))))
        emb += [signal * cvec[owner[q]] + rng.normal(size=dim) for _ in range(int(ks[q]))]
    return {"seed": 0, "T": T, "owner": owner, "k": ks, "cand_rows": cand_rows, "q_rows": q_rows}, np.array(emb)


def test_e36_and_judgement_on_synthetic_pools():
    import p36_eval as ev
    rng = np.random.default_rng(5)
    pools, embs, cols = [], [], []
    for _ in range(6):
        pool, emb = synthetic_pool(rng, 30, 40, 2.0)
        count = np.ones(len(emb), int)
        count[pool["cand_rows"][0][0]] = 0                   # a game without a sample is left out
        colour = np.array(["B", "W"])[rng.integers(0, 2, len(emb))]
        e = ev.Emb(emb, count, colour=colour)
        M, ok = ev.e36(e, pool)
        assert M.shape == (40, 30) and ok.all() and np.abs(M).max() <= 1 + 1e-9
        v = emb[pool["cand_rows"][0][1:]].mean(0)
        q = emb[pool["q_rows"][0]].mean(0)
        assert np.isclose(M[0, 0], v @ q / np.linalg.norm(v) / np.linalg.norm(q))
        pools.append(pool)
        embs.append((M, ok))
        cols.append((ev.e36_colour(e, pool, False), ev.e36_colour(e, pool, True)))
    rows = ev.tune_grid(pools, embs)
    assert len(rows) == len(ev.WE_GRID) * len(ev.K_GRID) and [r[1:] for r in rows][:4] == [(0.5, 10), (0.5, 20),
                                                                                         (0.5, 40), (1, 10)]
    rec = ev.judge_numbers(pools, embs, 2, 10, cols)
    d = np.array(rec["per_pool"]["M"]) - np.array(rec["per_pool"]["G"])
    assert np.isclose(rec["M_minus_G"]["mean"], d.mean())
    lo, hi = rec["M_minus_G"]["ci95"]
    assert lo <= d.mean() <= hi and (lo, hi) == ev.bootstrap(d)
    assert rec["adopt"] == bool(d.mean() >= 0.02 and lo > 0 and (d > 0).sum() >= np.ceil(0.9 * len(d))) and rec["adopt"]        # the planted signal helps
    assert set(rec["by_k"]["M"]) == {"1", "2", "3", "5"}
    assert set(rec["E36_only_by_colour_csls10"]) == {"same_colour", "cross_colour"}
    # no signal in E36 -> not adopted
    pools0, embs0 = [], []
    for _ in range(6):
        pool, emb = synthetic_pool(rng, 30, 40, 0.0)
        pools0.append(pool)
        embs0.append(ev.e36(ev.Emb(emb, np.ones(len(emb), int)), pool))
    assert not ev.judge_numbers(pools0, embs0, 2, 10)["adopt"]
    # a question whose games all lack a sample: flagged, its E36 row contributes nothing
    pool, emb = synthetic_pool(rng, 30, 40, 2.0)
    count = np.ones(len(emb), int)
    count[pool["q_rows"][3]] = 0
    M, ok = ev.e36(ev.Emb(emb, count), pool)
    assert not ok[3] and ok.sum() == 39 and not M[3].any()


def test_judge_refuses_uncommitted_tune(tmp_path):
    import p36_eval as ev
    p = os.path.join(HERE, "_not_committed_tune_test.json")
    try:
        L.write_json(p, {"x": 1})
        assert not L.git_committed(p)
        with pytest.raises(SystemExit):
            ev.main(["judge", "--tune", p, "--out", str(tmp_path / "eval.json")])
    finally:
        os.remove(p)
    assert L.git_committed(os.path.join(HERE, "PLAN.md"))


def test_pool_items_order():
    cands = [{"player_id": "p_b", "sgf_content": "s1", "color": "B"},
             {"player_id": "p_a", "sgf_content": "s2", "color": "W"},
             {"player_id": "p_b", "sgf_content": "s3", "color": "W"}]
    qs = [{"question_id": "q1", "num_games": "2", "sgf_1": "s4", "color_1": "B", "sgf_2": "s5", "color_2": "W"}]
    items, ids, cr, qr, qids = L.pool_items(cands, qs)
    assert [s for s, _c in items] == ["s1", "s2", "s3", "s4", "s5"] and ids == ["p_a", "p_b"]
    assert cr == [[1], [0, 2]] and qr == [[3, 4]] and qids == ["q1"]
