"""prereg34 exploration: unit tests of the alignment, training and judgement code (synthetic data and committed
records only; never the answers folder).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg34/explore/test_p34_model.py
"""

import os
import shutil
import sys
import types

import numpy as np
import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p34_prep as PR  # noqa: E402
import p34_align as A  # noqa: E402
import p34_train as P  # noqa: E402
import p34_eval as EV  # noqa: E402

T, K, L25, L29, P25T, P29T = P.T, P.K, P.L25, P.L29, P.P25T, P.P29T


# ---------------------------------------------------------------- isotonic fit, labels (PLAN.md 3)
def test_pav_is_monotone_and_least_squares():
    rng = np.random.default_rng(0)
    x = rng.uniform(15, 36, 400)
    y = 0.4 * (x - 15) + rng.normal(0, 2, 400)
    ux, fu, f = A.pav(x, y)
    assert np.all(np.diff(fu) >= 0)
    order = np.argsort(x, kind="stable")
    assert np.all(np.diff(f[order]) >= -1e-12)
    # the mean is kept (least squares with an intercept-like block structure)
    assert abs(f.mean() - y.mean()) < 1e-9
    # another monotone fit (a small increasing tilt keeps monotonicity) has a larger squared error
    sse = ((f - y) ** 2).sum()
    g = fu + 1e-3 * np.linspace(0, 1, len(fu))
    assert ((g[np.searchsorted(ux, x)] - y) ** 2).sum() >= sse - 1e-9
    # = scipy's isotonic regression on the tie-pooled means (reference only; the program uses its own PAV)
    from scipy.optimize import isotonic_regression
    _u, inv = np.unique(x, return_inverse=True)
    w = np.bincount(inv)
    ref = isotonic_regression(np.bincount(inv, weights=y) / w, weights=w).x
    assert np.max(np.abs(ref - fu)) < 1e-12


def test_pav_known_example_and_ties():
    ux, fu, f = A.pav([1, 2, 3, 4], [1.0, 3.0, 2.0, 4.0])
    assert ux.tolist() == [1, 2, 3, 4] and fu.tolist() == [1.0, 2.5, 2.5, 4.0]
    # tied x pooled first: x = 2 twice with y 0 and 4 -> mean 2 before PAV
    ux, fu, f = A.pav([1, 2, 2, 3], [3.0, 0.0, 4.0, 1.0])
    assert ux.tolist() == [1, 2, 3]
    assert np.allclose(fu, [2.0, 2.0, 2.0]) and np.allclose(f, 2.0)
    # already monotone data are returned unchanged; decreasing data become one block
    ux, fu, f = A.pav([1, 2, 3], [0.0, 1.0, 5.0])
    assert fu.tolist() == [0.0, 1.0, 5.0]
    ux, fu, f = A.pav([1, 2, 3], [3.0, 2.0, 1.0])
    assert fu.tolist() == [2.0, 2.0, 2.0]


def test_labels_round_half_up_and_drop():
    f = np.array([-0.51, -0.5, 0.0, 0.49, 0.5, 1.4999, 1.5, 8.5, 9.0, 9.49, 9.5, 10.0])
    lab, keep = A.labels_of(f)
    assert keep.tolist() == [False, True, True, True, True, True, True, True, True, True, False, False]
    assert lab.tolist() == [-1, 0, 0, 0, 1, 1, 2, 9, 9, 9, -1, -1]
    assert set(lab[keep].tolist()) <= set(range(10))


def test_knots_are_the_constant_pieces():
    ux = np.array([1.0, 2.0, 3.0, 4.0])
    fu = np.array([0.5, 0.5, 2.0, 2.0])
    assert A.knots(ux, fu) == [[1.0, 2.0, 0.5], [3.0, 4.0, 2.0]]


def test_player_e_mean_of_logp_then_renormalised():
    p1 = np.full(10, 0.1)
    p2 = np.zeros(10)
    p2[9] = 1.0 - 9e-6
    p2[:9] = 1e-6
    p, e = A.player_e([np.log(p1)])
    assert np.allclose(p, 0.1) and abs(e - 4.5) < 1e-12
    p, e = A.player_e([np.log(p1), np.log(p2)])
    g = np.sqrt(p1 * p2)
    assert np.allclose(p, g / g.sum()) and abs(p.sum() - 1) < 1e-12
    assert abs(e - float(np.dot(np.arange(10), g / g.sum()))) < 1e-12


def _meta(rows):
    """rows: (player, colour, start_time, game number, median)."""
    return {"players": np.array([r[0] for r in rows]), "color": np.array([r[1] for r in rows]),
            "start_time": np.array([r[2] for r in rows], dtype=np.int64),
            "game_id": np.array([f"ogs_{r[3]}" for r in rows]),
            "player_median": np.array([r[4] for r in rows], dtype=np.float64)}


def test_questions_newest_20_per_colour():
    rows = [("ogs_1", "B", 100 + i, 1000 + i, 20.0) for i in range(25)]          # 25 black rows
    rows += [("ogs_1", "W", 50, 2000 + i, 20.0) for i in range(3)]               # 3 white rows, same time
    rows += [("ogs_2", "W", 10 * i, 3000 + i, 30.0) for i in range(22)]          # only white
    players, q = A.questions(_meta(rows))
    assert players == ["ogs_1", "ogs_2"]
    assert q["ogs_1"]["B"].tolist() == list(range(5, 25))                         # the newest 20 by time
    assert q["ogs_1"]["W"].tolist() == [25, 26, 27]
    assert set(q["ogs_2"]) == {"W"} and q["ogs_2"]["W"].tolist() == list(range(30, 50))
    # ties of start_time: the larger game number is newer
    rows = [("ogs_3", "B", 7, 5000 + i, 25.0) for i in range(21)]
    _p, q = A.questions(_meta(rows))
    assert q["ogs_3"]["B"].tolist() == list(range(1, 21))


def _random_pars(seed):
    rng = np.random.default_rng(seed)
    shapes = {"W1": (K.HIDDEN, K.D_IN), "b1": (K.HIDDEN,), "W2": (K.H_DIM, K.HIDDEN), "b2": (K.H_DIM,),
              "R1": (K.RHO, K.H_DIM), "c1": (K.RHO,), "R2": (1, K.RHO), "c2": (1,)}
    par = {k: rng.normal(0, 0.05, s) for k, s in shapes.items()}
    par["W1"][:, 0] = 0.5                              # feature 0 carries the level
    par["R1"][:] = np.abs(par["R1"])
    par["R2"][:] = np.abs(par["R2"]) + 0.1
    par["t0"] = np.array(-4.0)
    par["dt"] = np.full(K.L - 2, 0.0)
    return K.p64(par)


def test_align_end_to_end_synthetic():
    meta = PR.synth_meta(n_players=60, per_player=25, seed=7)
    n = len(meta["ref"])
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (n, K.D_IN))
    X[:, 0] += (meta["player_median"] - 25.0) / 3.0                           # monotone signal in the median
    pars = [_random_pars(s) for s in range(3)]
    H = [K.h_np(p, X) for p in pars]
    res = A.align(pars, H, meta)
    assert len(res["players"]) == 60
    order = np.argsort(res["median"], kind="stable")
    assert np.all(np.diff(res["f"][order]) >= -1e-12)
    assert np.all(np.diff(res["label"][order][res["keep"][order]]) >= 0)       # labels follow the OGS order
    assert res["keep"].all() and set(res["label"].tolist()) <= set(range(10))
    assert res["question_rows"]["max"] <= A.QUESTION_MAX
    # e of one player by hand: lse mean over seeds per colour, mean of the colours' lp, renormalised
    p0 = res["players"][0]
    _pl, q = A.questions(meta)
    lps = []
    for c in ("B", "W"):
        if c in q[p0]:
            g = q[p0][c]
            lps.append(K.lse_mean([K.head_np(par, Hs[g].mean(axis=0, keepdims=True))[0] for par, Hs in zip(pars, H)]))
    assert abs(A.player_e(lps)[1] - res["e"][0]) < 1e-10
    rb = A.per_bin(res)
    assert sum(v["players"] for v in rb.values()) == 60


# ---------------------------------------------------------------- prep join
def test_build_meta_join_and_refusals():
    games = [{"player_id": "ogs_1", "game_id": "ogs_10", "color": "B", "ogs_rank": "20.5"},
             {"player_id": "ogs_2", "game_id": "ogs_10", "color": "W"},
             {"player_id": "ogs_1", "game_id": "ogs_11", "color": "W"}]
    idx = {("ogs_1", "ogs_10", "B"): (20.5, 100, 21.0), ("ogs_2", "ogs_10", "W"): (25.0, 100, 24.0),
           ("ogs_1", "ogs_11", "W"): (21.5, 200, 21.0)}
    m = PR.build_meta(games, [0, 2], idx)
    assert m["players"].tolist() == ["ogs_1", "ogs_1"] and m["color"].tolist() == ["B", "W"]
    assert m["start_time"].tolist() == [100, 200] and m["player_median"].tolist() == [21.0, 21.0]
    with pytest.raises(SystemExit):
        PR.build_meta([{**games[0], "ogs_rank": "20.0"}], [0], idx)               # rank differs from the sample
    with pytest.raises(SystemExit):
        PR.build_meta([{"player_id": "ogs_1", "game_id": "ogs_99", "color": "B"}], [0], idx)   # not in the sample
    with pytest.raises(SystemExit):
        PR.build_meta([{"player_id": "1", "game_id": "ogs_10", "color": "B"}], [0], idx)       # not an OGS id
    bad = dict(idx)
    bad[("ogs_1", "ogs_11", "W")] = (21.5, 200, 22.0)
    with pytest.raises(SystemExit):
        PR.build_meta(games, [0, 2], bad)                                         # two medians for one player


# ---------------------------------------------------------------- cap per level (PLAN.md 4)
def test_eligible_players_need_5_rows_of_one_colour():
    pl = ["ogs_1"] * 5 + ["ogs_2"] * 4 + ["ogs_2"] * 4 + ["ogs_3"] * 6
    col = ["B"] * 5 + ["B"] * 4 + ["W"] * 4 + ["W"] * 6
    assert P.eligible_players(pl, col) == {"ogs_1", "ogs_3"}


def test_cap_players_respects_cap_and_is_fixed():
    labels = {f"ogs_{i}": (i % 3) for i in range(1, 61)}                         # 20 players at levels 0, 1, 2
    elig = set(labels) - {"ogs_3"}                                                  # one level-0 player not eligible
    caps = [5, 100, 0] + [10] * 7
    chosen, rec = P.cap_players(labels, elig, caps)
    lv = [labels[p] for p in chosen]
    assert lv.count(0) == 5 and lv.count(1) == 20 and lv.count(2) == 0
    assert "ogs_3" not in chosen
    assert rec[0] == {"level": 0, "eligible": 19, "cap": 5, "kept": 5}
    assert rec[2] == {"level": 2, "eligible": 20, "cap": 0, "kept": 0}
    again, _ = P.cap_players(labels, elig, caps)
    assert again == chosen
    other, _ = P.cap_players(labels, elig, caps, seed=35)
    assert other != chosen
    # the draw is default_rng(34): one permutation per level, levels ascending, players sorted by numeric id
    rng = np.random.default_rng(34)
    pl0 = sorted((p for p in elig if labels[p] == 0), key=lambda s: int(s[4:]))
    want0 = {pl0[i] for i in rng.permutation(len(pl0))[:5].tolist()}
    assert want0 == {p for p in chosen if labels[p] == 0}
    # cap not below the level size: everybody kept
    allc, _ = P.cap_players(labels, set(labels), [100] * 10)
    assert sorted(allc) == sorted(labels)


def test_ogs_keys_format():
    assert P.ogs_keys(["ogs_5", "ogs_7"], {"ogs_5": 0, "ogs_7": 9}).tolist() == ["D:ogs_5", "6D:ogs_7"]


# ---------------------------------------------------------------- stage 0b (PLAN.md 5)
def test_stage0b_rule_edges():
    c = {0: 1.0, 1: 1.0, 2: 1.0}
    d = P.stage0b_decision({0: 0.875, 1: 0.875, 2: 0.875}, c, go_mean=-0.125)
    assert d["decision"] == "GO" and d["mean_diff_s34_minus_s25"] == -0.125              # exactly at the threshold
    d = P.stage0b_decision({0: 0.5, 1: 0.875, 2: 1.0}, c, go_mean=-0.125)
    assert d["decision"] == "NO-GO" and d["condition_mean"] and d["seeds_s34_lower"] == 2   # one seed equal
    d = P.stage0b_decision({0: 0.5, 1: 0.875, 2: 1.25}, c, go_mean=-0.125)
    assert d["decision"] == "NO-GO" and d["condition_mean"]                              # one seed higher
    d = P.stage0b_decision({0: 0.9375, 1: 0.875, 2: 0.875}, c, go_mean=-0.125)
    assert d["decision"] == "NO-GO" and not d["condition_mean"] and d["condition_all_lower"]
    # registered threshold with the real S25 controls
    s25 = P.S25_NLL
    assert P.GO_MEAN == -0.010
    go = P.stage0b_decision({s: v - 0.02 for s, v in s25.items()}, s25)
    assert go["decision"] == "GO"
    nogo = P.stage0b_decision({s: v - 0.005 for s, v in s25.items()}, s25)
    assert nogo["decision"] == "NO-GO" and nogo["condition_all_lower"]
    with pytest.raises(SystemExit):
        P.stage0b_decision({0: 1.0, 1: 1.0}, c)
    with pytest.raises(SystemExit):
        P.stage0b_decision({0: float("nan"), 1: 1.0, 2: 1.0}, c)


def test_s25_controls_are_the_committed_record():
    tr25 = L29.read_json(P.P25_TRAIN)
    assert {int(s): v["best_holdout_nll"] for s, v in tr25["seeds"].items()} == P.S25_NLL
    assert tr25["players"]["A_players_per_level"] == [730, 678, 874, 1311, 360, 503, 504, 525, 544, 590]


# ---------------------------------------------------------------- once guards
def test_align_once(monkeypatch, tmp_path):
    out = tmp_path / "align_record.json"
    out.write_text("{}")
    monkeypatch.setattr(A, "SMOKE_RECORD", str(out))
    with pytest.raises(SystemExit, match="once"):
        A.cmd_align(types.SimpleNamespace(smoke=True))


def test_stage0b_once(monkeypatch, tmp_path):
    (tmp_path / "stage0b.json").write_text("{}")
    monkeypatch.setattr(P, "SMOKE_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="once"):
        P.cmd_stage0b(types.SimpleNamespace(smoke=True))


def test_eval_once_guard(tmp_path):
    rep = tmp_path / "eval_report.json"
    st = tmp_path / "eval_started.json"
    a = types.SimpleNamespace(rerun=False, reason=None)
    assert EV.once_guard(a, {"head": "h1"}, str(rep), str(st)) is None
    st.write_text('{"code": {"head": "h1"}}')
    with pytest.raises(SystemExit, match="judged once"):
        EV.once_guard(a, {"head": "h1"}, str(rep), str(st))
    a = types.SimpleNamespace(rerun=True, reason="bug X fixed")
    with pytest.raises(SystemExit, match="newer commit"):
        EV.once_guard(a, {"head": "h1"}, str(rep), str(st))
    assert EV.once_guard(a, {"head": "h2"}, str(rep), str(st)) == ["h1"]


def test_eval_needs_stage0b_go_for_this_train_record(tmp_path):
    tr = tmp_path / "train_record.json"
    tr.write_text('{"a": 1}\n')
    sha = A.sha_lf(str(tr))
    go = {"format": EV.S0B_FORMAT, "smoke": False, "train_record_sha256_lf": sha, "decision": "GO"}
    assert EV.stage0b_gate(go, str(tr), False)["ok"]
    assert not EV.stage0b_gate({**go, "decision": "NO-GO"}, str(tr), False)["ok"]
    assert not EV.stage0b_gate({**go, "train_record_sha256_lf": "0" * 64}, str(tr), False)["ok"]
    assert not EV.stage0b_gate({**go, "smoke": True}, str(tr), False)["ok"]
    assert EV.stage0b_gate({**go, "smoke": True, "decision": "NO-GO"}, str(tr), True)["ok"]   # smoke: pipeline only
    assert P.S0B_FORMAT == EV.S0B_FORMAT and P.FORMAT == EV.TRAIN_FORMAT


# ---------------------------------------------------------------- seams
def test_fill_s25_rows_equals_p25_rows_standardised():
    rng = np.random.default_rng(3)
    n_p, n_v = T.BLOCK + 300, T.BLOCK + 17
    probe = rng.normal(0, 2, (n_p, K.D_IN)).astype(np.float32)
    v = rng.normal(0, 2, (n_v, K.D_IN)).astype(np.float32)
    probe[5, 3] = np.nan
    fake = types.SimpleNamespace(prep=types.SimpleNamespace(probe_X=probe), vX=v, n_probe=n_p, n_v=n_v)
    mu = rng.normal(0, 1, K.D_IN)
    sd = rng.uniform(0.5, 2, K.D_IN)
    for smoke in (False, True):
        want = P25T.Rows.standardised(fake, mu, sd, smoke)
        X = np.empty((n_p + n_v + 11, K.D_IN), dtype=np.float32)
        P.fill_s25_rows(X, fake, mu, sd, smoke)
        assert np.array_equal(X[:n_p + n_v], want)


def test_ogs_standardised_into(tmp_path):
    meta = PR.synth_meta(n_players=4, per_player=5)
    n = len(meta["ref"])
    rng = np.random.default_rng(2)
    Xr = rng.normal(0, 1, (n, K.D_IN)).astype(np.float32)
    np.save(tmp_path / "ogs_X.npy", Xr)
    np.savez(tmp_path / "ogs_meta.npz", **meta)
    PR.write_json(str(tmp_path / "prep_record.json"), {"rows": n})
    o = PR.OgsRows(str(tmp_path))
    mu, sd = np.full(K.D_IN, 0.5), np.full(K.D_IN, 2.0)
    rows = np.array([1, 3, 7])
    got = o.standardised(mu, sd, False, rows)
    assert np.array_equal(got, ((Xr[rows].astype(np.float64) - 0.5) / 2.0).astype(np.float32))
    sm = o.standardised(mu, sd, True, rows)
    assert np.array_equal(sm, np.random.default_rng([PR.NOISE_SEED, PR.OGS_NOISE_STREAM, 0]).standard_normal(
        (3, K.D_IN)).astype(np.float32))


def test_pool_with_ogs_rows_appended_keeps_the_a_pool():
    keys = np.array(["D:p1"] * 6 + ["1D:p2"] * 6 + ["D:p3"] * 5)
    y = np.array([0] * 6 + [4] * 6 + [0] * 5)
    col = np.array((["B", "W"] * 3) * 2 + ["B"] * 5)
    rows_a = np.arange(12)
    ref = K.PlayerPool(keys, y, col, rows_a)
    k2 = np.concatenate([keys, P.ogs_keys(["ogs_9"] * 6, {"ogs_9": 0})])
    y2 = np.concatenate([y, np.zeros(6, dtype=np.int64)])
    c2 = np.concatenate([col, np.array(["B"] * 6)])
    assert P.same_pool(K.PlayerPool(k2, y2, c2, rows_a), ref)
    big = K.PlayerPool(k2, y2, c2, np.concatenate([rows_a, np.arange(17, 23)]))
    assert not P.same_pool(big, ref) and "D:ogs_9" in big.players
    assert len(big.eligible[5][0][0]) == 1        # level D black, n = 5: only the OGS player (p1 has 3 black rows)


def test_seam_r34_compares_bytes_with_frozen_s25_seed0():
    frozen = os.path.join(L29.S25_DIR, "s25_seed0.npz")
    tmp_path = __import__("pathlib").Path(PR.SMOKE_DIR) / "unit_seam_r"        # on the repository drive (relpath)
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True)
    same = tmp_path / "same.npz"
    shutil.copyfile(frozen, same)
    assert P29T.seam_r(str(same), False)["ok"]
    other = tmp_path / "other.npz"
    b = bytearray(open(frozen, "rb").read())
    b[-30] ^= 1
    other.write_bytes(bytes(b))
    assert not P29T.seam_r(str(other), False)["ok"]
    assert P29T.seam_r(str(same), True)["applicable"] is False
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_stop_and_formats():
    assert T.STOP == P.STOP and P.STOP.replace("\\", "/").endswith("data/prereg34/STOP")
    assert P.MODEL_FORMAT == L29.S25_FORMAT == "c10-prereg25 s25 model v1"
    assert EV.Q25_REF == (0.7456519871546352, 0.7482450052195216)
    assert all("lo" + "cal" not in f for f in P.IMPORTED + EV.IMPORTED + A.IMPORTED)
