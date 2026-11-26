"""prereg08 P1 tests (CPU only; kept out of tests/ like the pilot's test_hs_lib.py).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider results/prereg08/p1/test_p1.py
Model tests load the 324 MB checkpoint once (~10 s) and run a few positions in fp32 on the CPU.
"""

import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import p1_lib as P  # noqa: E402
import p1_extract as E  # noqa: E402
import p1_fuse as FU  # noqa: E402

H = P.H
REPO = P.REPO
FIXTURE = os.path.join(REPO, "data", "prereg07", "fixtures", "fit_sample.json")
PLAN = os.path.join(HERE, "p1_plan.json")


def fixture_games(n):
    if not os.path.exists(FIXTURE):
        pytest.skip("data/prereg07/fixtures/fit_sample.json missing")
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)["games"][:n]


# ---------------------------------------------------------------- pins, guard, plan
def test_pins_equal_p07_rank_fuse():
    src = open(os.path.join(REPO, "tools", "p07_rank_fuse.py"), encoding="utf-8").read()
    for rel, sha in P.PINNED.items():
        assert re.search(re.escape(f'"{rel}":') + r'\s*"' + sha + '"', src), rel
        assert P.check_pin(rel) == sha


def test_data_local_guard():
    with pytest.raises(H.LocalDataRefused):
        open(os.path.join(REPO, "data", "local", "anything_p1_test.csv"))
    with pytest.raises(H.LocalDataRefused):
        os.listdir(os.path.join(REPO, "data", "local"))


def test_plan_fixed_choices():
    with open(PLAN, encoding="utf-8") as f:
        plan = json.load(f)
    fx = plan["fixed_before_results"]
    assert fx["primary_variant"] == "TO" and fx["primary_combination"] == "Q02H+P1TO"
    assert fx["primary_combination"] in FU.COMBOS
    assert set(fx["record_only_combinations"]) | {fx["primary_combination"], "Q02H"} == set(FU.COMBOS)
    assert FU.COMBOS["Q02H"] == ["OP", "O4", "SWA", "SWB", "HS"]
    assert plan["mapping"]["lambdas"] == sorted(plan["mapping"]["lambdas"], reverse=True)
    assert plan["sets"]["probe"]["games"] == E.PROBE_PER_LEVEL * 10
    assert plan["sets"]["probe"]["seed"] == E.PROBE_SEED
    assert plan["costs_validation_scoring"] is False
    assert FU.GATE == 0.010 and FU.MONO_QUESTIONS == 120544 and FU.PILOT_Q02H == 0.6242752597994481


def test_sets_json_matches_plan():
    if not os.path.exists(E.SETS):
        pytest.skip("data/prereg08/p1/sets.json not built")
    with open(PLAN, encoding="utf-8") as f:
        plan = json.load(f)
    assert P.sha_raw(E.SETS) == plan["sets"]["sets_json_sha256"]
    s = E.load_sets()
    assert s["counts"]["probe"] == 30000 and s["counts"]["bank"] == 11813 and s["counts"]["mono"] == 61356
    lv = [g["rank"] for g in s["probe"]["games"]]
    assert all(lv.count(x) == 3000 for x in H.LEVELS)
    assert not set(s["bank"]["rows"]) & set(s["mono"]["rows"])


# ---------------------------------------------------------------- probe sampling
def _rows():
    rng = np.random.default_rng(5)
    rows = []
    for lv in H.LEVELS:
        for p in range(7):
            for g in range(int(rng.integers(1, 9))):
                rows.append({"rank": lv, "player_id": f"{lv}p{p}", "game_id": f"{lv}g{p}_{g}", "color": "B"})
    return rows


def test_build_probe_deterministic_and_spread():
    rows = _rows()
    a = E.build_probe(rows, per_level=12, seed=3)
    b = E.build_probe(list(reversed(rows)), per_level=12, seed=3)
    assert a == b
    assert len(a) == 120 and len({(g["rank"], g["game_id"]) for g in a}) == 120
    for lv in H.LEVELS:
        cnt = {}
        for g in a:
            if g["rank"] == lv:
                cnt[g["player_id"]] = cnt.get(g["player_id"], 0) + 1
        avail = {p: sum(1 for r in rows if r["player_id"] == p) for p in {r["player_id"] for r in rows if r["rank"] == lv}}
        # round robin: no player gets a second game while another available player has none
        if any(c >= 2 for c in cnt.values()):
            assert all(cnt.get(p, 0) >= 1 for p in avail)
    with pytest.raises(P.P1Error):
        E.build_probe(rows, per_level=100, seed=3)


def test_player_folds_grouped():
    players = np.array([f"p{i % 37}" for i in range(500)])
    f = FU.player_folds(players)
    for p in set(players):
        assert len(set(f[players == p])) == 1
    assert set(f) == set(range(5))


def test_standardise_fills_missing_with_mean():
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    Xs, mu, sd = FU.standardise(X)
    assert Xs[1, 1] == 0.0 and abs(mu[1] - 4.0) < 1e-12
    Y, _, _ = FU.standardise(np.array([[np.nan, 4.0]]), mu, sd)
    assert Y[0, 0] == 0.0 and Y[0, 1] == 0.0


def test_game_means():
    pf = np.arange(4 * P.F_DIM, dtype=np.float32).reshape(4, P.F_DIM)
    side = np.array([0, 1, 0, 0], dtype=np.int8)
    m, c = P.game_means(pf, side)
    assert list(c) == [3, 1]
    assert np.allclose(m[0], pf[[0, 2, 3]].mean(0)) and np.allclose(m[1], pf[1])
    m2, c2 = P.game_means(pf[:1], side[:1])
    assert list(c2) == [1, 0] and not m2[1].any()


# ---------------------------------------------------------------- metadata and featurisation
def test_neutral_metadata():
    t = P.neutral_table()
    ref = H.Profile("5k").row
    for pla in (1, 2):
        assert t[pla][4] == 1.0 and t[pla][5] == 1.0          # both ranks unknown
        assert not t[pla][6:74].any()                          # no rank bits
        r = ref(pla)
        assert np.array_equal(t[pla][:4], r[:4]) and np.array_equal(t[pla][74:], r[74:])
    assert not t[0].any()


def test_memo_featurizer_bitwise_equals_hs_lib():
    for g in fixture_games(4):
        a = H.featurize_game(g["sgf_content"], g["color"], sides=("target", "opponent"))
        b = P.featurize_game(g["sgf_content"], g["color"])
        assert a["n_moves"] == b["n_moves"]
        for k in ("bin", "glob", "legal", "move", "pla", "side", "turn"):
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k


# ---------------------------------------------------------------- model (CPU fp32)
@pytest.fixture(scope="module")
def model():
    if not os.path.exists(H.CKPT_PATH):
        pytest.skip("checkpoint missing")
    return P.load_model(device="cpu")


def test_probe_features_cpu(model):
    import torch
    from katago.train.model_pytorch import ExtraOutputs
    g = fixture_games(1)[0]
    f = P.featurize_game(g["sgf_content"], g["color"])
    sub = {k: (v[:9] if isinstance(v, np.ndarray) else v) for k, v in f.items()}
    sc = P.ProbeScorer(model, device="cpu", precision="fp32", batch=4, pad=True)
    a = sc.position_features(sub)
    sc.close()
    sc2 = P.ProbeScorer(model, device="cpu", precision="fp32", batch=64, pad=False)
    b = sc2.position_features(sub)
    sc2.close()
    assert a.shape == (9, P.F_DIM) and np.isfinite(a).all()
    assert np.max(np.abs(a - b)) < 1e-4                         # batch size / padding invariant
    # independent reference: KataGo's own ExtraOutputs 'trunkfinal' and an explicit masked softmax
    with torch.inference_mode():
        bt = torch.as_tensor(sub["bin"]).float()
        gt = torch.as_tensor(sub["glob"]).float()
        mt = torch.as_tensor(P.neutral_table())[torch.as_tensor(sub["pla"].astype(np.int64))]
        eo = ExtraOutputs(["trunkfinal"])
        out = model(bt, gt, input_meta=mt, extra_outputs=eo)
        tr = eo.returned["trunkfinal"].float().numpy()
        logits = out[0][0][:, 0, :].float().numpy().astype(np.float64)
    for i in range(9):
        mv = int(sub["move"][i])
        y, x = divmod(mv, 19)
        assert np.max(np.abs(a[i, :384] - tr[i, :, y, x])) < 1e-4
        assert np.max(np.abs(a[i, 384:768] - tr[i].mean((1, 2)))) < 1e-4
        z = np.where(sub["legal"][i], logits[i], -np.inf)
        lp = z - np.max(z) - np.log(np.sum(np.exp(z - np.max(z))))
        p = np.exp(lp)
        ent = -np.sum(np.where(sub["legal"][i], p * np.where(np.isfinite(lp), lp, 0.0), 0.0))
        assert abs(a[i, 768] - lp[mv]) < 1e-4 and abs(a[i, 769] - ent) < 1e-4


def test_chunk_ok_rejects_mismatch(tmp_path):
    root = str(tmp_path)
    path = E.chunk_path("probe", 0, root)
    os.makedirs(os.path.dirname(path))
    refs = np.arange(3, dtype=np.int64)
    np.savez(path, format=np.array(P.CHUNK_FORMAT), set=np.array("probe"), sides=np.array(E.SIDES), ref=refs,
             meta_sha256=np.array(P.neutral_table_sha()))
    assert E.chunk_ok(path, "probe", refs)
    assert not E.chunk_ok(path, "probe", refs + 1)
    assert not E.chunk_ok(path, "bank", refs)


def test_max_games_refused_for_real_folder():
    with pytest.raises(P.P1Error):
        E.main(["run", "--set", "probe", "--max-games", "5", "--device", "cpu"])


def test_night_launcher():
    p = os.path.join(REPO, "data", "prereg08", "p1", "night.sh")
    if not os.path.exists(p):
        pytest.skip("night.sh not present")
    s = open(p, encoding="utf-8").read()
    loop = s.index("for SET in probe bank mono; do")
    assert s.index("p1_extract.py run --set $SET") > loop and s.index("p1_fuse.py") > s.index("done")
    assert "STOP" in s and "score.py" not in s and "data/local" not in s
