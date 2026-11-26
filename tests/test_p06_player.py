"""PREREG_06: tools/p06_player.py refusals, selection rules, thresholds and the apply path on synthetic smoke data
(spec.md 9.1). Nothing under data/local is opened, listed or stat-ed; tools/score.py is never used."""
import csv
import os
import random
import sys
import types

sys.dont_write_bytecode = True

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(REPO, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import p02_player_fuse as pf  # noqa: E402
import p02_player_open as po  # noqa: E402
import p06_player as pl  # noqa: E402
import p06_style as st  # noqa: E402

LET = "abcdefghijklmnopqrs"


def args_for(**kw):
    ns = types.SimpleNamespace(smoke=True, train_dir=pl.TRAIN_DIR, split_csv=pl.SPLIT_CSV, local_dir=pl.LOCAL,
                               design_dir=None, player_dir=None, feat_dir=None, cnn_npz=None, cnn_report=None,
                               explore_cache=None, workers=1, threads=1, seeds_tune=None, seeds_eval=None,
                               seeds_sel=None, preview=None, candidates=None, questions=None, p02_dir=None,
                               p02_top5=None, rerun=False, reason=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------- refusals
def test_apply_refused_before_freeze(tmp_path):
    with pytest.raises(pl.Stop, match="frozen_config.json missing"):
        pl.cmd_apply(args_for(player_dir=str(tmp_path), design_dir=str(tmp_path)))


def test_design_step_cannot_read_local(tmp_path):
    a = args_for(train_dir=os.path.join(REPO, "data", "local"))
    with pytest.raises(RuntimeError):
        pl.fit_rows(a)


def test_marker_names_refused(tmp_path):
    p = tmp_path / ("x_" + "ans" + "wers.csv")
    with pytest.raises(st.Refused):
        st.sha256_file(str(p))
    with pytest.raises(RuntimeError):
        open(str(p), "w")


def test_no_overwrite_and_smoke_outside_repo(tmp_path):
    f = tmp_path / "frozen_config.json"
    f.write_text("{}")
    with pytest.raises(pl.Stop):
        pl.no_overwrite(str(f))
    with pytest.raises(pl.Stop, match="inside the repository"):
        pl.out_check(args_for(), os.path.join(REPO, "results", "prereg06", "design", "x.json"))
    pl.out_check(args_for(), str(tmp_path / "x.json"))


def test_build_refused_without_committed_leakcheck(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "check_tools", lambda args, allow_precommit=True: ({}, False))
    a = args_for(smoke=False, design_dir=str(tmp_path / "d"), feat_dir=str(tmp_path / "f"))
    with pytest.raises(pl.Stop, match="leakcheck"):
        pl.cmd_build(a)


def test_real_mode_refuses_uncommitted_tools(monkeypatch):
    monkeypatch.delenv("C10_P06_PRECOMMIT", raising=False)
    monkeypatch.setattr(st, "tools_state", lambda names=st.P06_TOOLS: {n: {"sha256_lf": "x",
                                                                           "committed_and_clean": False}
                                                                       for n in names})
    with pytest.raises(pl.Stop, match="not committed"):
        pl.check_tools(args_for(smoke=False))
    monkeypatch.setenv("C10_P06_PRECOMMIT", "1")
    assert pl.check_tools(args_for(smoke=False))[1] is True
    with pytest.raises(pl.Stop):
        pl.check_tools(args_for(smoke=False), allow_precommit=False)     # apply never runs precommit


def test_seed_overrides_only_in_smoke():
    with pytest.raises(pl.Stop, match="smoke only"):
        pl.main(["tune", "--seeds-tune", "1"])


def test_no_score_tool_used():
    for n in ("p06_player.py", "p06_sim.py", "p06_style.py", "p06_cnn_fit.py"):
        src = open(os.path.join(TOOLS, n), encoding="utf-8").read()
        assert "import score" not in src and "score.py compare" not in src
        if n != "p06_cnn_fit.py":           # (its refusal message names the marker; it never opens such a file)
            assert ("ans" + "wers") not in src


# ---------------------------------------------------------------- selection rules
def test_choose_combo_tolerance():
    order = ["V2", "V1", "V3", "V0"]
    assert pl.choose_combo({"V2": 0.900, "V1": 0.902, "V3": 0.899, "V0": 0.880}, order) == "V2"
    assert pl.choose_combo({"V2": 0.900, "V1": 0.904, "V3": 0.899, "V0": 0.880}, order) == "V1"
    assert pl.choose_combo({"V2": 0.900, "V3": 0.899, "V0": 0.9031}, ["V2", "V3", "V0"]) == "V0"
    assert pl.choose_combo({"V2": 0.900, "V3": 0.899, "V0": 0.9029}, ["V2", "V3", "V0"]) == "V2"


def _grid(v2, best_key, best):
    keys = ["%g|%g|%g" % (a, b, c) for a in pl.GRID_L for b in pl.GRID_J for c in pl.GRID_S]
    g = {k: 0.5 for k in keys}
    g["2.66|0.5|0.5"] = v2
    g[best_key] = best
    return g


def test_v1_decision():
    t = _grid(0.90, "4|1|1", 0.905)
    e = _grid(0.90, "4|1|1", 0.901)
    d = pl.v1_decision(t, e, 0.45, True)
    assert d["v1"] == [4.0, 1.0, 1.0] and not d["reasons"]
    d = pl.v1_decision(t, _grid(0.90, "4|1|1", 0.899), 0.45, True)       # evaluation condition fails
    assert d["v1"] is None and d["conditions"]["eval_not_below_v2"] is False
    d = pl.v1_decision(_grid(0.90, "4|1|1", 0.902), e, 0.45, True)       # gain < 0.003
    assert d["v1"] is None
    d = pl.v1_decision(t, e, 0.51, True)                                 # E0 cap
    assert d["v1"] is None and d["cnn_representative"] is False
    d = pl.v1_decision({}, {}, None, False)
    assert d["v1"] is None and d["cnn_representative"] is None


def _rows(spec):
    """spec: [(scen, r, on, off)] -> switch rows for key 'X|E0'."""
    return [{"scen": s, "seed": i, "res": {"X|E0": {"r_forced_cost": r, "on": on, "off_10": off, "off_none": off}}}
            for i, (s, r, on, off) in enumerate(spec)]


def test_select_threshold_rules():
    rows = _rows([("o2o", 1.0, 0.99, 0.98), ("o2o", 1.2, 0.99, 0.98), ("o2o", 1.4, 0.99, 0.98),
                  ("u390", 1.6, 0.97, 0.98), ("wr", 3.0, 0.5, 0.9)])
    thr, det = pl.select_threshold(rows, "X|E0", "off_10")
    assert thr == pytest.approx(1.4)                       # opening u390 (loss 0.01 > 0.005) is infeasible
    # ties: equal gain at 1.4 and 1.5 -> the smaller
    rows2 = rows + _rows([("d1", 1.5, 0.98, 0.98)])
    assert pl.select_threshold(rows2, "X|E0", "off_10")[0] == pytest.approx(1.4)
    # nothing gains -> None
    rows3 = _rows([("o2o", 1.0, 0.97, 0.98), ("wr", 2.0, 0.5, 0.9)])
    assert pl.select_threshold(rows3, "X|E0", "off_10")[0] is None


def test_r_star_and_k_star():
    assert pl.r_star_of([1.5, 1.4]) == 1.4
    assert pl.r_star_of([1.5, None]) is None and pl.r_star_of([]) is None
    rows = []
    for sc, (n, a10, a20) in {"o2o": (0.9, 0.91, 0.92), "wr": (0.5, 0.49, 0.5)}.items():
        rows.append({"scen": sc, "res": {"V2|E0": {"off_none": n, "off_10": a10, "off_20": a20, "off_40": n}}})
    assert pl.k_star(rows, "V2|E0")[0] == 20
    rows[1]["res"]["V2|E0"]["off_20"] = 0.49
    assert pl.k_star(rows, "V2|E0")[0] == 40
    rows[1]["res"]["V2|E0"]["off_40"] = 0.49
    assert pl.k_star(rows, "V2|E0")[0] is None


def test_usable_check():
    rows = []
    for i in range(10):
        rows.append({"scen": "o2o", "res": {"X|E0": {"r_forced_cost": 1.0 if i < 9 else 2.0, "on": 1, "off_10": 0.9},
                                            "X|open": {"r_forced_cost": 1.0}}})
    rows.append({"scen": "wr", "res": {"X|E0": {"r_forced_cost": 3.0, "on": 0.5, "off_10": 0.9},
                                       "X|open": {"r_forced_cost": 3.0}}})
    rows.append({"scen": "d1", "res": {"X|E0": {"r_forced_cost": 1.1, "on": 0.95, "off_10": 0.95},
                                       "X|open": {"r_forced_cost": 1.1}}})
    ok, det = pl.usable_check(rows, "X|E0", "X|open", "off_10", 1.5)
    assert ok and det["o2o_open_share"] == 0.9 and det["non_o2o_opens"] == 0
    rows[10]["res"]["X|open"]["r_forced_cost"] = 1.2             # opening+v2 alone would open wrongly
    assert not pl.usable_check(rows, "X|E0", "X|open", "off_10", 1.5)[0]
    assert not pl.usable_check(rows, "X|E0", "X|open", "off_10", None)[0]


# ---------------------------------------------------------------- apply on smoke data
def _sgf(rng, n):
    return "(;GM[1]SZ[19]" + "".join(";%s[%s]" % ("B" if i % 2 == 0 else "W", rng.choice(LET) + rng.choice(LET))
                                     for i in range(n)) + ")"


def make_smoke(tmp, n_pl=8, seed=0):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    cand, qs = [], []
    for i in range(n_pl):
        p = "p%02d" % i
        k = rng.randint(5, 20)
        for j in range(100 + k):
            col = "B" if rng.random() < 0.6 else "W"
            g = (("g%02d_%03d" % (i, j)), p, col, _sgf(rng, rng.randint(30, 160)))
            if j == 3:
                g = (g[0], p, col, "(;GM[1];B[aa];W[];B[cc];W[dd])")       # a pass: not alternating
            if j < 100:
                cand.append(g)
            else:
                qs.append((i, g))
    qrows = []
    for i in range(n_pl):
        games = [g for ii, g in qs if ii == i]
        qrows.append(("q%02d" % ((i * 5) % n_pl), games))
    qrows.sort()
    cp = os.path.join(tmp, "cands.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["game_id", "player_id", "color", "sgf_content"])
        for g in cand:
            w.writerow(g)
    qp = os.path.join(tmp, "questions.csv")
    with open(qp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)] + ["color_%d" % i for i in range(1, 21)])
        for q, games in qrows:
            sg = [g[3] for g in games] + [""] * (20 - len(games))
            cl = [g[2] for g in games] + [""] * (20 - len(games))
            w.writerow([q, len(games)] + sg + cl)
    p02 = os.path.join(tmp, "p02")
    os.makedirs(os.path.join(p02, "dumps"))
    np.savez(os.path.join(p02, "dumps", "cand_games.npz"), game_id=np.array([g[0] for g in cand]),
             player_id=np.array([g[1] for g in cand]), color=np.array([g[2] for g in cand]),
             emb_sum=nrng.normal(size=(len(cand), 256)).astype(np.float32),
             n_windows=np.full(len(cand), 50, np.int32))
    qid, slot, qc = [], [], []
    for q, games in qrows:
        for s, g in enumerate(games, 1):
            qid.append(q)
            slot.append(s)
            qc.append(g[2])
    np.savez(os.path.join(p02, "dumps", "query_games.npz"), question_id=np.array(qid), slot=np.array(slot, np.int32),
             color=np.array(qc), emb_sum=nrng.normal(size=(len(qid), 256)).astype(np.float32),
             n_windows=np.full(len(qid), 50, np.int32), parsed=np.ones(len(qid), bool))
    return cp, qp, p02, cand


def reference_q04(cp, qp, p02, out):
    d = pf.Data(cp, p02, True, True, q_path=qp)
    cg = [d.c_by_player[p] for p in d.cand_ids]
    qg = [d.q_by_id[q] for q in d.qids]
    gc, gq = d.G("raw")
    cos, has = po.open_cos(gc, cg, gq, qg)
    M, h, _ = pf.e_scores("E0", (d.cemb, d.cnw, d.cok), cg, (d.qemb, d.qnw, d.qok), qg)
    S = pf.fuse(pf.COMBO_BY_ID["Q04"], {"E0": (M, h), "raw": (cos, has)})
    pl.write_top5(out, d.qids, pf.top_ids(S, d.cand_ids))


def smoke_lda(cand, player):
    from collections import Counter
    for c in "BW":
        rows = [g for g in cand if g[2] == c and st.alternates(st.parse_all(g[3]))]
        toks = [Counter(st.resp_tokens(st.parse_all(g[3]), c)) for g in rows]
        vocab = sorted({t for x in toks for t in x})
        vi = {t: j for j, t in enumerate(vocab)}
        X = np.vstack([st.ldain_from(st.style2(st.parse_all(g[3]), c), x, vi) for g, x in zip(rows, toks)])
        m = st.lda_fit(X, [g[1] for g in rows])
        np.savez(os.path.join(player, "lda_%s.npz" % c), W=m["W"], ev=m["ev"], mu=m["mu"], sd=m["sd"],
                 vocab=np.array(vocab))


def test_apply_smoke(tmp_path):
    tmp = str(tmp_path)
    cp, qp, p02, cand = make_smoke(tmp)
    ref = os.path.join(tmp, "ref_top5.csv")
    reference_q04(cp, qp, p02, ref)
    player, design = os.path.join(tmp, "player"), os.path.join(tmp, "design")
    os.makedirs(player)
    os.makedirs(design)
    smoke_lda(cand, player)
    st.write_json(os.path.join(design, "dmix_report.json"), {"passed": True})
    st.write_json(os.path.join(player, "frozen_config.json"),
                  {"precommit_preview": False, "tools": {}, "combinations": {"V1": None}, "k_star": 10,
                   "switch": {"V2": {"r_star": 1.5, "usable": True}}, "prereg02_inputs": {},
                   "lda": {c: {"sha256": st.sha256_file(os.path.join(player, "lda_%s.npz" % c))} for c in "BW"}})
    a = args_for(player_dir=player, design_dir=design, candidates=cp, questions=qp, p02_dir=p02, p02_top5=ref)
    assert pl.cmd_apply(a) == 0
    rep = st.read_json(os.path.join(player, "apply_report.json"))
    assert rep["q04_reproduction"]["identical"] is True and rep["q04_reproduction"]["identical_06_path"] is True
    assert rep["inputs"]["candidates_csv"] == st.sha256_file(cp)
    assert rep["detector"]["would_open"] == (rep["detector"]["r_forced_cost"] <= 1.5)
    assert rep["selected"] in ("V2", "V3", "V0")
    assert all(rep["secondary_structure"].values())
    assert all(v["returncode"] == 0 for v in rep["format_checks"].values())
    assert rep["games"]["candidate"]["not_alternating_B"] + rep["games"]["candidate"]["not_alternating_W"] == 8
    with open(os.path.join(player, "player_val_top5_switch.csv"), encoding="utf-8") as f:
        r = list(csv.reader(f))
    assert r[0] == ["question_id"] + pf.TOP_COLS and len({x[1] for x in r[1:]}) == len(r) - 1
    # --rerun keeps the previous report
    a.rerun, a.reason = True, "test"
    assert pl.cmd_apply(a) == 0
    assert os.path.exists(os.path.join(player, "apply_report.prev1.json"))
    # a wrong LDA file is refused
    fz = st.read_json(os.path.join(player, "frozen_config.json"))
    fz2 = dict(fz, lda={"B": {"sha256": "0" * 64}, "W": fz["lda"]["W"]})
    st.write_json(os.path.join(player, "frozen_config.json"), fz2)
    with pytest.raises(pl.Stop, match="lda_B.npz"):
        pl.cmd_apply(a)
    st.write_json(os.path.join(player, "frozen_config.json"), fz)
    # a different reference Q04 file stops the run
    with open(ref, "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(pl.Stop, match="differs"):
        pl.cmd_apply(a)


def test_merge_blocks_equals_direct_counts():
    from collections import Counter
    rng = random.Random(3)
    items = [(_sgf(rng, rng.randint(1, 120)), rng.choice("BW")) for _ in range(40)]
    items.append(("(;GM[1];B[aa];B[bb])", "B"))
    parts = [pl._feat_chunk(items[:15]), pl._feat_chunk(items[15:])]
    for name in ("resp", "jo", "oraw"):
        M, vocab = pl.merge_blocks([p[4][name] for p in parts])
        assert vocab == sorted(vocab) and M.shape == (len(items), len(vocab))
        for i, (sgf, c) in enumerate(items):
            mv = st.parse_all(sgf)
            if name == "oraw":
                want = Counter(po.game_tokens(c, pl.p02_moves.parse_moves(sgf), "raw"))
            elif not st.alternates(mv):
                want = Counter()
            else:
                want = Counter(st.resp_tokens(mv, c) if name == "resp" else st.jo_tokens(mv, c))
            row = M.getrow(i)
            got = {vocab[j]: v for j, v in zip(row.indices, row.data)}
            assert got == dict(want)


# ---------------------------------------------------------------- review fixes: freeze, gate, apply refusals
def make_design(tmp):
    """A consistent set of smoke design reports, LDA files and feature-store files."""
    design, player, feats = (os.path.join(tmp, x) for x in ("design", "player", "feats"))
    for d in (design, player, feats):
        os.makedirs(d, exist_ok=True)
    tools = st.tools_state()
    base = {"mode": "smoke", "precommit": False, "tools": tools}
    for name, content in (("meta.npz", b"m"), ("ldain_B.npy", b"b"), ("ldain_B_rows.npy", b"br"),
                          ("ldain_B_cols.npy", b"bc"), ("ldain_W.npy", b"w"), ("ldain_W_rows.npy", b"wr"),
                          ("ldain_W_cols.npy", b"wc")):
        with open(os.path.join(feats, name), "wb") as f:
            f.write(content)
    for c in "BW":
        with open(os.path.join(player, "lda_%s.npz" % c), "wb") as f:
            f.write(c.encode())

    def w(name, obj):
        st.write_json(os.path.join(design, name), dict(base, **obj))
        return st.sha256_lf(os.path.join(design, name))

    def fsha(*p):
        return st.sha256_file(os.path.join(*p))
    lk = w("leakcheck_report.json", {"passed": True, "overlap": 0})
    b = w("build_report.json", {"passed": True, "files_sha256": {"meta.npz": fsha(feats, "meta.npz")},
                                "leakcheck_report": {"sha256": lk}})
    cols = {c: {"sha256": fsha(player, "lda_%s.npz" % c), "ldain_sha256": fsha(feats, "ldain_%s.npy" % c),
                "ldain_rows_sha256": fsha(feats, "ldain_%s_rows.npy" % c),
                "ldain_cols_sha256": fsha(feats, "ldain_%s_cols.npy" % c)} for c in "BW"}
    lda = w("lda_report.json", {"passed": True, "colours": cols, "inputs": {"build_report.json": b}})
    w("selfcheck_report.json", {"passed": True, "inputs": {"lda_report.json": lda}})
    w("dmix_report.json", {"passed": False, "inputs": {"lda_report.json": lda}})
    t = w("tune_report.json", {"inputs": {"build_report.json": b, "lda_report.json": lda,
                                          "cnn_fit_report.json": None}})
    w("switch_report.json", {"inputs": {"tune_report.json": t}})
    return design, player, feats, tools


def _edit(path, **kw):
    r = st.read_json(path)
    r.update(kw)
    st.write_json(path, r)


def test_freeze_consistent_set_has_no_problem(tmp_path):
    design, player, feats, tools = make_design(str(tmp_path))
    assert pl.freeze_problems(args_for(), design, player, feats, tools) == []     # a failing D-mix still freezes


@pytest.mark.parametrize("case", ["build_failed", "leak_overlap", "leak_missing", "tool_sha", "upstream",
                                  "lda_file", "feats", "selfcheck_failed", "build_leak_link", "mode"])
def test_freeze_refusals(tmp_path, case):
    design, player, feats, tools = make_design(str(tmp_path))

    def D(n):
        return os.path.join(design, n)
    if case == "build_failed":
        _edit(D("build_report.json"), passed=False)
    elif case == "leak_overlap":
        _edit(D("leakcheck_report.json"), overlap=3)
    elif case == "leak_missing":
        os.remove(D("leakcheck_report.json"))
    elif case == "tool_sha":
        r = st.read_json(D("switch_report.json"))
        r["tools"] = {k: dict(v, sha256_lf="0" * 64) for k, v in r["tools"].items()}
        st.write_json(D("switch_report.json"), r)
    elif case == "upstream":
        _edit(D("lda_report.json"), note="changed after tune")
    elif case == "lda_file":
        with open(os.path.join(player, "lda_W.npz"), "wb") as f:
            f.write(b"other")
    elif case == "feats":
        with open(os.path.join(feats, "ldain_B_cols.npy"), "wb") as f:
            f.write(b"other")
    elif case == "selfcheck_failed":
        _edit(D("selfcheck_report.json"), passed=False)
    elif case == "build_leak_link":
        _edit(D("leakcheck_report.json"), note="rewritten")
    elif case == "mode":
        _edit(D("dmix_report.json"), mode="real")
    assert pl.freeze_problems(args_for(), design, player, feats, tools)
    with pytest.raises(pl.Stop, match="refused"):
        pl.cmd_freeze(args_for(design_dir=design, player_dir=player, feat_dir=feats,
                               preview=str(tmp_path / "preview.json")))


def test_freeze_leakcheck_tool_sha_not_compared(tmp_path):
    design, player, feats, tools = make_design(str(tmp_path))
    r = st.read_json(os.path.join(design, "leakcheck_report.json"))
    r["tools"] = {k: dict(v, sha256_lf="1" * 64) for k, v in r["tools"].items()}
    st.write_json(os.path.join(design, "leakcheck_report.json"), r)
    # rewriting leakcheck changes its file SHA: re-point build at it, so only the tool SHAs differ
    _edit(os.path.join(design, "build_report.json"),
          leakcheck_report={"sha256": st.sha256_lf(os.path.join(design, "leakcheck_report.json"))})
    # build changed, so its downstream links are re-pointed too
    b = st.sha256_lf(os.path.join(design, "build_report.json"))
    _edit(os.path.join(design, "lda_report.json"), inputs={"build_report.json": b})
    lda = st.sha256_lf(os.path.join(design, "lda_report.json"))
    for n in ("selfcheck_report.json", "dmix_report.json"):
        _edit(os.path.join(design, n), inputs={"lda_report.json": lda})
    _edit(os.path.join(design, "tune_report.json"),
          inputs={"build_report.json": b, "lda_report.json": lda, "cnn_fit_report.json": None})
    _edit(os.path.join(design, "switch_report.json"),
          inputs={"tune_report.json": st.sha256_lf(os.path.join(design, "tune_report.json"))})
    assert pl.freeze_problems(args_for(), design, player, feats, tools) == []


def test_freeze_real_mode_needs_commits(tmp_path):
    design, player, feats, tools = make_design(str(tmp_path))
    for n in os.listdir(design):
        _edit(os.path.join(design, n), mode="real")
    probs = pl.freeze_problems(args_for(smoke=False), design, player, feats, tools)
    assert any("not committed" in p for p in probs)


def test_score_gate_branches():
    srm = {"V2": 0.95, "V3": 0.94, "V0": 0.94}
    ok = dict(chosen="V2", srm=srm, dmix_passed=True, q04_same=True, q04_06_same=True,
              struct={"a": True}, fmt={"x": {"returncode": 0}})
    assert pl.score_gate(**ok) == []
    assert pl.score_gate(**dict(ok, chosen="V0"))[0] == "selected V0"
    assert pl.score_gate(**dict(ok, srm={"V2": 0.944, "V0": 0.94}))                   # +0.004 < +0.005
    assert pl.score_gate(**dict(ok, srm={"V2": 0.945, "V0": 0.94})) == []             # exactly +0.005
    assert pl.score_gate(**dict(ok, dmix_passed=False)) == ["D-mix did not pass"]
    assert pl.score_gate(**dict(ok, dmix_passed=None)) == ["D-mix did not pass"]
    assert pl.score_gate(**dict(ok, q04_same=False))
    assert pl.score_gate(**dict(ok, q04_06_same=False))
    assert pl.score_gate(**dict(ok, struct={"a": False}))
    assert pl.score_gate(**dict(ok, fmt={"x": {"returncode": 1}}))
    assert pl.score_gate(**dict(ok, fmt={}))


@pytest.mark.parametrize("over", [{"candidates": "C:\\x\\c.csv"}, {"questions": "C:\\x\\q.csv"},
                                  {"p02_top5": "C:\\x\\t.csv"}, {"design_dir": "C:\\x"}, {"p02_dir": "C:\\x"}])
def test_apply_real_mode_refuses_overrides(over):
    a = args_for(smoke=False, candidates=pl.APPLY_DEFAULTS["candidates"], questions=pl.APPLY_DEFAULTS["questions"])
    for k, v in over.items():
        setattr(a, k, v)
    with pytest.raises(pl.Stop, match="overrides"):
        pl.cmd_apply(a)


def test_apply_real_mode_needs_frozen():
    if os.path.exists(os.path.join(pl.PLAYER, "frozen_config.json")):
        pytest.skip("frozen_config.json exists")
    a = args_for(smoke=False, candidates=pl.APPLY_DEFAULTS["candidates"], questions=pl.APPLY_DEFAULTS["questions"])
    with pytest.raises(pl.Stop, match="frozen_config.json missing"):
        pl.cmd_apply(a)


def test_rectangular_hungarian():
    rng = np.random.default_rng(7)
    a = st.hungarian_top1(rng.normal(size=(3, 5)))
    assert len(set(a.tolist())) == 3 and a.max() < 5
    with pytest.raises(ValueError):
        st.hungarian_top1(rng.normal(size=(5, 3)))


def test_guard_low_level_calls():
    import io
    st.install_guard()
    p = os.path.join(REPO, "data", "local", "player_val_questions.csv")
    for fn in (lambda: os.open(p, os.O_RDONLY), lambda: os.lstat(p), lambda: io.FileIO(p)):
        with pytest.raises(RuntimeError):
            fn()
