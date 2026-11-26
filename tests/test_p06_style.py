"""PREREG_06: tools/p06_style.py against the exploration functions (spec.md 9.1) and the white / alternation rules.

The exploration modules (results/prereg06/explore/c1_headroom) are imported with bytecode writing off; the test
checks that no __pycache__ appears there. Synthetic games only; nothing under data/local is touched.
"""
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

import p06_style as st  # noqa: E402

LET = "abcdefghijklmnopqrs"


def _pycache():
    return sorted(os.path.join(d, x) for d, dn, _f in os.walk(EXPLORE) for x in dn if x == "__pycache__")


_BEFORE = _pycache()
import c1h_eval as XE  # noqa: E402
import c1h_feats as XF  # noqa: E402
import c1h_lda as XL  # noqa: E402
import s8_switch as XS  # noqa: E402


def rand_pts(rng, n):
    return "".join(rng.choice(LET) + rng.choice(LET) for _ in range(n))


def moves(pts):
    return [("B" if i % 2 == 0 else "W", pts[2 * i:2 * i + 2]) for i in range(len(pts) // 2)]


def test_no_pycache_in_explore():
    assert _pycache() == _BEFORE


@pytest.mark.parametrize("seed", range(5))
def test_black_features_equal_exploration(seed):
    rng = random.Random(seed)
    for _ in range(60):
        pts = rand_pts(rng, rng.randint(1, 260))
        mv = moves(pts)
        assert st.alternates(mv)
        assert np.array_equal(st.style2(mv, "B"), XF.style2_vec(pts))
        assert np.array_equal(st.style34(mv, "B"), XF.style_vec(pts))
        assert st.resp_tokens(mv, "B") == XF.f_resp(pts)
        jo = st.jo_tokens(mv, "B")
        assert all(t.startswith("BK") for t in jo)
        assert [t[1:] for t in jo] == XF.f_joseki_own(pts)


def test_white_definitions_by_hand():
    mv = [("B", "dd"), ("W", "pp"), ("B", "dp"), ("W", "pd"), ("B", "jj")]
    # own white moves are i = 1, 3; at i = 1 there is no own previous move -> 'RS0:...|-'
    toks = st.resp_tokens(mv, "W")
    assert toks[:3] == ["L0:4", "R0:f", "RS0:f|-"]
    assert "S0:f" in toks and "RS0:f|f" in toks
    assert not any(t.startswith("RS0:") and t.endswith("|-") for t in st.resp_tokens(mv, "B"))
    s2 = st.style2(mv, "W")
    assert s2[0] == pytest.approx(0.05)
    # phase [0, 20): two own moves, both on line 4 (index 3), both in a corner
    assert s2[1 + 3] == 1.0 and s2[1 + 10] == 1.0
    jo = st.jo_tokens(mv, "W")
    assert all(t.startswith("WK") for t in jo) and len(jo) == 2
    # black's own moves are the even indices; the white JO tokens never contain black's points as own
    assert all(t.startswith("BK") for t in st.jo_tokens(mv, "B"))


def test_alternation_rule():
    assert st.alternates([("B", "aa"), ("W", "bb"), ("B", "cc")])
    assert not st.alternates([("B", "aa"), ("B", "bb")])             # two black moves in a row
    assert not st.alternates([("W", "aa"), ("B", "bb")])             # white first
    assert not st.alternates([])                                     # no move
    mv = st.parse_all("(;GM[1];B[aa];W[];B[bb];W[cc])")              # a pass is not matched by MOVE_RE
    assert mv == [("B", "aa"), ("B", "bb"), ("W", "cc")] and not st.alternates(mv)
    assert st.parse_all("(;B[aa];W[tt];B[bb])") == [("B", "aa"), ("B", "bb")]


def test_feature_chunk_skips_non_alternating():
    import p06_player as pl
    items = [("(;GM[1];B[aa];W[bb];B[cc])", "B"), ("(;GM[1];B[aa];W[];B[cc])", "B"), ("(;GM[1])", "W")]
    S2, S34, alt, nm, blocks = pl._feat_chunk(items)
    assert alt.tolist() == [True, False, False]
    assert nm.tolist() == [3, 2, 0]
    assert np.isfinite(S2[0]).all() and np.isnan(S2[1]).all() and np.isnan(S2[2]).all()
    voc, dat, ind, ptr = blocks["resp"]
    assert ptr.tolist()[1] > 0 and ptr.tolist()[1] == ptr.tolist()[2] == ptr.tolist()[3]
    voc, dat, ind, ptr = blocks["oraw"]
    assert ptr.tolist()[2] > ptr.tolist()[1]                         # O_raw is computed for every game


def test_lda_fit_equals_exploration(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_pl, per, D = 30, 25, 80
    centres = rng.normal(size=(n_pl, D))
    X = np.vstack([centres[i] + rng.normal(scale=2.0, size=(per, D)) for i in range(n_pl)]).astype(np.float32)
    X[:, 5] = 0.0                                                     # a constant column (sd < 1e-9 -> 1)
    labels = np.repeat(["p%02d" % i for i in range(n_pl)], per)
    path = tmp_path / "dense_ldain.npy"
    np.save(path, X)
    monkeypatch.setattr(XL, "LDAIN", str(path))
    W, ev, mu, sd = XL.fit(np.arange(len(X)), labels)
    m = st.lda_fit(np.asarray(X, np.float64), labels)
    assert np.max(np.abs(m["W"] - W[:, :64])) <= 1e-10
    assert np.max(np.abs(m["ev"] - ev[:64])) <= 1e-10
    assert np.array_equal(m["mu"], mu) and np.array_equal(m["sd"], sd)
    assert np.all(np.diff(m["ev"]) <= 0)


def test_csls_hungarian_with_top1_equal_exploration():
    rng = np.random.default_rng(1)
    S = rng.normal(size=(40, 40))
    for k in (1, 3, 10, 20):
        assert np.array_equal(st.csls(S, k), XE.csls(S, k))
    assert np.array_equal(st.csls(S, None), S)
    a = st.hungarian_top1(S)
    assert np.array_equal(a, XE.hungarian_top1(S))
    assert np.array_equal(st.with_top1(S, a), XE.with_top1(S, a))


def test_detector_equals_s8():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(50, 50)) + 3 * np.eye(50)
    P = [rng.normal(size=(50, 50)) + 2 * np.eye(50) for _ in range(2)]
    d = st.detector(F, P)
    _r, s = XS.methods_and_stats(F, np.arange(50), np.full(50, 5), with_methods=False)
    assert d["forced_cost"] == pytest.approx(s["forced_cost"], abs=1e-12)
    assert d["changed"] == pytest.approx(s["changed"], abs=1e-12)
    assert d["n0"] == s["n0"]
    fp = [XS.methods_and_stats(x, np.arange(50), np.full(50, 5), with_methods=False)[1]["forced_cost"] for x in P]
    assert d["r_forced_cost"] == pytest.approx(s["forced_cost"] / np.mean(fp), abs=1e-12)


def test_all_black_colour_merge_equals_z():
    rng = np.random.default_rng(3)
    T = rng.normal(size=(20, 30))
    out, has = st.colour_terms({"B": (T, np.ones(30, bool))}, {"B": [5] * 20})
    assert has.all()
    assert np.max(np.abs(out - st.zrows(T))) <= 1e-12
    # a fused z of it is unchanged
    assert np.max(np.abs(st.zrows(out) - st.zrows(T))) <= 1e-12


def test_colour_merge_mixed_and_missing():
    rng = np.random.default_rng(4)
    TB, TW = rng.normal(size=(3, 6)), rng.normal(size=(3, 6))
    chW = np.array([True, True, True, False, True, True])
    out, has = st.colour_terms({"B": (TB, np.ones(6, bool)), "W": (TW, chW)}, {"B": [2, 0, 0], "W": [2, 3, 0]})
    TWf = TW.copy()
    TWf[:, ~chW] = TW[:, chW].min(1, keepdims=True)
    exp0 = 0.5 * st.zrows(TB[[0]]) + 0.5 * st.zrows(TWf[[0]])
    assert np.allclose(out[0], exp0[0]) and np.allclose(out[1], st.zrows(TWf[[1]])[0])
    assert has.tolist() == [True, True, False] and np.all(out[2] == 0)


def test_sr06_rounds():
    rng = random.Random(5)
    cg = {"p%03d" % i: ["g%d_%d" % (i, j) for j in range(100)] for i in range(12)}
    ks = [rng.choice((5, 10, 15, 20, 25)) for _ in range(12)]
    a = st.sr06_rounds(cg, ks)
    b = st.sr06_rounds(cg, list(reversed(ks)))          # K is sorted before the shuffle: input order irrelevant
    assert a == b and len(a) == 10
    for p, games in cg.items():
        order = st.sr_order(p, games)
        parts = [order[j * 20:(j + 1) * 20] for j in range(5)]
        assert len(set(g for pp in parts for g in pp)) == 100
        for r, items in enumerate(a):
            it = dict((x[0], x[1:]) for x in items)
            q, refs = it[p]
            assert len(refs) == 80 and not set(q) & set(refs)
            assert q == parts[r % 5][:len(q)] and len(q) <= 20
    # K shuffle fixed by the salt: candidate i takes K[i mod |K|]
    K0 = sorted(min(k, 20) for k in ks)
    K = list(K0)
    random.Random(st._hash_int("c10-prereg06|sr-k|0")).shuffle(K)
    assert [len(x[1]) for x in a[0]] == [K[i % len(K)] for i in range(12)]
    with pytest.raises(ValueError):
        st.sr06_rounds({"p": ["g%d" % i for i in range(99)]}, [5])
    # query filter keeps the references
    f = st.sr06_rounds(cg, ks, query_filter=lambda p, g: int(g.split("_")[1]) % 2 == 0)
    assert all(int(g.split("_")[1]) % 2 == 0 for items in f for x in items for g in x[1])
    assert [x[2] for x in f[3]] == [x[2] for x in a[3]]


def test_switch_top5():
    rng = np.random.default_rng(6)
    F = rng.normal(size=(30, 30)) + 2 * np.eye(30)
    ids = ["c%02d" % i for i in range(30)]
    a = st.hungarian_top1(st.zrows(F))
    tops = st.switch_top5(F, ids)
    assert sorted(t[0] for t in tops) == ids                          # rank 1 is a permutation
    C3 = st.csls(F, 3)
    for i, t in enumerate(tops):
        assert t[0] == ids[a[i]] and ids[a[i]] not in t[1:]
        rest = [j for j in np.argsort(-C3[i], kind="stable") if j != a[i]][:4]
        assert t[1:] == [ids[j] for j in rest]


def test_fuse_missing_and_constant_rows():
    M = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [1.0, 0.0, 2.0]])
    ok = np.array([True, True, False])
    Z = st.zscore_rows(M, ok)
    assert np.all(Z[1] == 0) and np.all(Z[2] == 0)
    F = st.fuse([(M, ok), (M, np.ones(3, bool))], [0.0, 2.0])
    assert np.allclose(F, 2 * st.zscore_rows(M, np.ones(3, bool)))


def test_guard_refuses_local_and_marker():
    st.install_guard()
    with pytest.raises(RuntimeError):
        open(os.path.join(REPO, "data", "local", "player_val_questions.csv"))
    with pytest.raises(RuntimeError):
        os.listdir(os.path.join(REPO, "data", "local"))
    with pytest.raises(RuntimeError):
        open(os.path.join(REPO, "results", "x_" + "ans" + "wers.csv"))
    assert not st.under_local(os.path.join(os.environ.get("TEMP", "C:\\"), "AppData", "Local", "x"))
