"""prereg25 exploration: synthetic unit tests (no real data, no data/local).

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q results/prereg25/explore/test_p25.py
"""

import os
import sys
from collections import Counter

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p25_lib as L  # noqa: E402


def fake_rows(n_players, games_per, lv="A"):
    rows = []
    for p in range(n_players):
        for g in range(games_per[p]):
            rows.append({"rank": lv, "game_id": f"g{p}_{g}", "player_id": f"p{p}", "color": "BW"[g % 2]})
    return rows


def test_round_robin_quota_and_balance():
    per = [3, 50, 50, 50, 7]
    rows = fake_rows(5, per)
    out = L._build_one_level(rows, "A", 60, 1)
    assert len(out) == 60
    assert len({g["game_id"] for g in out}) == 60
    c = Counter(g["player_id"] for g in out)
    assert c["p0"] == 3 and c["p4"] == 7                       # exhausted players give everything
    big = [c[f"p{i}"] for i in (1, 2, 3)]
    assert max(big) - min(big) <= 1                            # round robin: the others differ by at most one
    assert out == L._build_one_level(rows, "A", 60, 1)         # deterministic
    assert out != L._build_one_level(rows, "A", 60, 2)


def test_round_robin_refuses_too_few_games():
    rows = fake_rows(2, [2, 2])
    try:
        L._build_one_level(rows, "A", 5, 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_batches_exclude_earlier_batches():
    rows = []
    for lv in L.LEVELS:
        rows += fake_rows(4, [40, 40, 40, 40], lv)
    quotas = L.scaled_quotas({lv: 4 for lv in L.LEVELS}, {lv: 16 for lv in L.LEVELS})
    assert quotas["v1"]["A"] == 750 and quotas["v2"]["A"] == 3000
    quotas = {b: {lv: q // 100 for lv, q in d.items()} for b, d in quotas.items()}
    v = L.build_v_sets(rows, quotas)
    allg = [(g["rank"], g["game_id"]) for b in v for g in v[b]]
    assert len(allg) == len(set(allg))
    assert sum(len(x) for x in v.values()) == sum(sum(d.values()) for d in quotas.values())


def test_symmetry_keys():
    g = ";B[ab];W[cd];B[];W[tt];B[se]"
    keys = L.sym_keys(g)
    assert keys[0] == L.move_key(g)[0] and L.move_key(g)[1] == 5
    assert len(set(keys)) == 8
    # transposed game (x, y) -> (y, x): its original key is one of the 8 keys of g
    t = ";B[ba];W[dc];B[];W[tt];B[es]"
    assert L.move_key(t)[0] in keys
    assert set(L.sym_keys(t)) == set(keys)


def test_holdout_bit_fraction():
    keys = [f"A:p{i}" for i in range(20000)]
    frac = sum(L.holdout_bit(k) for k in keys) / len(keys)
    assert 0.09 < frac < 0.11
    assert L.holdout_bit("A:p1") == L.holdout_bit("A:p1")


def test_guard_refuses_local():
    for p in ("data/local/x.csv", "D:/x/data/local"):
        try:
            L.guard(p)
        except SystemExit:
            continue
        raise AssertionError(p)
