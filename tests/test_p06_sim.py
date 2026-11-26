"""PREREG_06: tools/p06_sim.py pools against the exploration (c1h_lib.make_pool) on a synthetic cache, and the
D-mix pool rule (spec.md 9.1). Synthetic data only; nothing under data/local is touched."""
import os
import random
import sys

sys.dont_write_bytecode = True

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(REPO, "tools")
EXPLORE = os.path.join(REPO, "results", "prereg06", "explore", "c1_headroom")
for p in (TOOLS, EXPLORE):
    if p not in sys.path:
        sys.path.insert(0, p)

import p06_sim as sim  # noqa: E402
import c1h_lib as XL  # noqa: E402
from c10data import LEVELS  # noqa: E402


def synthetic_cache(seed=0, per_level=46):
    rng = random.Random(seed)
    n_all, n_black, level, by_player = {}, {}, {}, {}
    row = 0
    for lv in LEVELS:
        for i in range(per_level):
            p = "%s_%03d" % (lv, i)
            nb = rng.choice((125, 140, 200, 260))
            n_all[p] = nb + rng.randint(0, 50)
            n_black[p] = nb
            level[p] = lv
            if nb >= 105:
                by_player[p] = list(range(row, row + nb))
            row += nb
    return {"n_all": n_all, "n_black": n_black, "level": level, "by_player": by_player}


@pytest.mark.parametrize("scenario", ["o2o", "u390", "d80", "wr", "d1", "d5", "u300"])
def test_make_pool_equals_exploration(scenario):
    c = synthetic_cache()
    for seed in (0, 7):
        a = sim.make_pool(c, seed, scenario)
        b = XL.make_pool(c, seed, scenario)
        assert a["pool"] == b["pool"]
        assert a["cand_rows"] == b["cand_rows"] and a["q_rows"] == b["q_rows"]
        assert np.array_equal(a["owner"], b["owner"]) and np.array_equal(a["k"], b["k"])
        assert sorted(a["k"].tolist()) == sorted(sim.k_list(400))


def test_near_one_to_one_structures():
    c = synthetic_cache()
    for m in (1, 2, 3, 5):
        rec = sim.make_pool(c, 3, "d%d" % m)
        cnt = np.bincount(rec["owner"], minlength=400)
        assert (cnt == 0).sum() == m and (cnt == 2).sum() == m and (cnt == 1).sum() == 400 - 2 * m


class FakeStore:
    def __init__(self, games_per_player):
        pid, col, rk = [], [], []
        for i, (p, n) in enumerate(sorted(games_per_player.items())):
            for j in range(n):
                pid.append(p)
                col.append("B" if j % 3 else "W")
                rk.append(LEVELS[i % 10])
        self.pid, self.color, self.rank = np.array(pid), np.array(col), np.array(rk)
        self.gid = np.array(["g%d" % i for i in range(len(pid))])
        self.alt = np.ones(len(pid), bool)
        self.n = len(pid)
        self._by_player = None
        self._cache = None

    by_player = sim.Store.by_player
    black_cache = sim.Store.black_cache


def test_dmix_pool_rule():
    rng = random.Random(1)
    gp = {"p%04d" % i: rng.choice((105, 106, 110, 130, 300)) for i in range(10 * 45)}
    gp.update({"q%04d" % i: 80 for i in range(30)})           # < 105 games: never chosen
    s = FakeStore(gp)
    rec = sim.dmix_pool(s, 700)
    assert len(rec["pool"]) == 400 and not any(p.startswith("q") for p in rec["pool"])
    rng2 = random.Random("pool|700")
    bp = s.by_player()
    for i, p in enumerate(rec["pool"]):
        g = list(bp[p])
        rng2.shuffle(g)
        k = min(rng2.randint(5, 20), len(g) - 100)
        assert rec["cand_rows"][i] == g[:100] and rec["q_rows"][i] == g[100:100 + k]
    assert min(rec["k"]) >= 5 and max(rec["k"]) <= 20
    # both colours in the candidate games
    allc = s.color[[r for g in rec["cand_rows"] for r in g]]
    assert (allc == "B").any() and (allc == "W").any()


def test_sr_pool_one_to_one():
    gp = {"p%03d" % i: 150 for i in range(10 * 42)}
    s = FakeStore(gp)
    c = {"n_all": {p: 150 for p in gp}, "n_black": {p: 150 for p in gp},
         "level": {p: LEVELS[i % 10] for i, p in enumerate(sorted(gp))},
         "by_player": s.by_player()}
    rec = sim.make_pool(c, 0, "o2o")
    prs, ids = sim.sr_pool(s, rec)
    assert len(prs) == 2 and ids == sorted(ids)
    for pr in prs:
        assert len(pr["q_rows"]) == 400 and all(len(g) == 80 for g in pr["cand_rows"])
        for i, (cg, qg) in enumerate(zip(pr["cand_rows"], pr["q_rows"])):
            assert not set(cg) & set(qg)
            assert {s.pid[r] for r in cg + qg} == {pr["cand_ids"][i]}
