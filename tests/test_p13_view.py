"""tools/p13_view.py on synthetic smoke data (PREREG_13 sections 2.6, 3, 4, 5.2). Nothing under data/local is opened;
the scoring tool is never used.

1. The identity view (all games, all questions, strict=True) equals tools/p10_apply.py apply_core (sr06, sr06_10B,
   selection, pool gate, both Top-5 files byte for byte), and view_core's section 3 accepts it.
2. sr06_rounds / sr06_view with strict=False equal strict=True when every candidate has exactly 100 games.
3. "Groups hold only the black rows" equals "the white rows deleted from the pool" (a second synthetic pool written
   without any white game): the same n_view questions, the same sr06 / sr06_10B, and the Top-5 of 06/view and
   10B/view byte for byte.
4. view_core end to end: two 400-row files (here 8 rows), view rows from the view, the others verbatim from 06;
   check_submission passes; V1 refusals; a section 3 mismatch stops; the score_allowed decision; audit binding.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q tests/test_p13_view.py
"""
import csv
import os
import random
import shutil
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
import p06_player as pl  # noqa: E402
import p06_style as st  # noqa: E402
import p10_apply as A  # noqa: E402
import p10_fuse as F  # noqa: E402
import p13_view as V  # noqa: E402

LET = "abcdefghijklmnopqrs"
N_PL = 8
SET = "p10val_t"


def _sgf(rng, n):
    return "(;GM[1]SZ[19]" + "".join(";%s[%s]" % ("B" if i % 2 == 0 else "W", rng.choice(LET) + rng.choice(LET))
                                     for i in range(n)) + ")"


def make_games(n_pl=N_PL, seed=0):
    """The synthetic pool of tests/test_p10_apply.py make_smoke (copied): 100 candidate games per player (about 60 %
    black), one question of 5-20 games per player. -> (cand [(gid, pid, col, sgf)], qrows [(qid, games)], owner)."""
    rng = random.Random(seed)
    cand, qs = [], []
    for i in range(n_pl):
        p = "p%02d" % i
        k = rng.randint(5, 20)
        for j in range(100 + k):
            col = "B" if rng.random() < 0.6 else "W"
            g = (("g%02d_%03d" % (i, j)), p, col, _sgf(rng, rng.randint(30, 160)))
            if j < 100:
                cand.append(g)
            else:
                qs.append((i, g))
    qrows = []
    for i in range(n_pl):
        qrows.append(("q%02d" % ((i * 5) % n_pl), [g for ii, g in qs if ii == i]))
    qrows.sort()
    owner_of_q = {"q%02d" % ((i * 5) % n_pl): i for i in range(n_pl)}
    return cand, qrows, owner_of_q


def write_pool(tmp, cand, qrows, emb_c, emb_q):
    """Game files and prereg02-style dumps; emb_c: {gid: vector}, emb_q: {(qid, original game id): vector}."""
    os.makedirs(tmp, exist_ok=True)
    cp = os.path.join(tmp, "cands.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["game_id", "player_id", "color", "sgf_content"])
        for g in cand:
            w.writerow(g)
    qp = os.path.join(tmp, "questions.csv")
    with open(qp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)]
                   + ["color_%d" % i for i in range(1, 21)])
        for q, games in qrows:
            w.writerow([q, len(games)] + [g[3] for g in games] + [""] * (20 - len(games))
                       + [g[2] for g in games] + [""] * (20 - len(games)))
    p02 = os.path.join(tmp, "p02")
    os.makedirs(os.path.join(p02, "dumps"))
    np.savez(os.path.join(p02, "dumps", "cand_games.npz"), game_id=np.array([g[0] for g in cand]),
             player_id=np.array([g[1] for g in cand]), color=np.array([g[2] for g in cand]),
             emb_sum=np.vstack([emb_c[g[0]] for g in cand]).astype(np.float32),
             n_windows=np.full(len(cand), 50, np.int32))
    qid, slot, qc, qe = [], [], [], []
    for q, games in qrows:
        for s, g in enumerate(games, 1):
            qid.append(q)
            slot.append(s)
            qc.append(g[2])
            qe.append(emb_q[(q, g[0])])
    np.savez(os.path.join(p02, "dumps", "query_games.npz"), question_id=np.array(qid), slot=np.array(slot, np.int32),
             color=np.array(qc), emb_sum=np.vstack(qe).astype(np.float32),
             n_windows=np.full(len(qid), 50, np.int32), parsed=np.ones(len(qid), bool))
    return cp, qp, p02


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


def run_p06(tmp, cp, qp, p02, cand):
    """06 on the smoke pool through tools/p06_player.py (as tests/test_p10_apply.py)."""
    d = pf.Data(cp, p02, True, True, q_path=qp)
    import p02_player_open as po
    cg = [d.c_by_player[p] for p in d.cand_ids]
    qg = [d.q_by_id[q] for q in d.qids]
    gc, gq = d.G("raw")
    cos, has = po.open_cos(gc, cg, gq, qg)
    M, h, _ = pf.e_scores("E0", (d.cemb, d.cnw, d.cok), cg, (d.qemb, d.qnw, d.qok), qg)
    ref = os.path.join(tmp, "ref_q04.csv")
    pl.write_top5(ref, d.qids, pf.top_ids(pf.fuse(pf.COMBO_BY_ID["Q04"], {"E0": (M, h), "raw": (cos, has)}),
                                          d.cand_ids))
    player, design = os.path.join(tmp, "p06player"), os.path.join(tmp, "design")
    os.makedirs(player)
    os.makedirs(design)
    smoke_lda(cand, player)
    st.write_json(os.path.join(design, "dmix_report.json"), {"passed": True})
    st.write_json(os.path.join(player, "frozen_config.json"),
                  {"precommit_preview": False, "tools": {}, "combinations": {"V1": None}, "k_star": 10,
                   "switch": {"V2": {"r_star": 1.5, "usable": True}}, "prereg02_inputs": {},
                   "lda": {c: {"sha256": st.sha256_file(os.path.join(player, "lda_%s.npz" % c))} for c in "BW"}})
    a = types.SimpleNamespace(smoke=True, train_dir=pl.TRAIN_DIR, split_csv=pl.SPLIT_CSV, local_dir=pl.LOCAL,
                              design_dir=design, player_dir=player, feat_dir=None, cnn_npz=None, cnn_report=None,
                              explore_cache=None, workers=1, threads=1, seeds_tune=None, seeds_eval=None,
                              seeds_sel=None, preview=None, candidates=cp, questions=qp, p02_dir=p02, p02_top5=ref,
                              rerun=False, reason=None)
    assert pl.cmd_apply(a) == 0
    return player


def write_chunks(d, keys, M):
    os.makedirs(d)
    for ci, s in enumerate(range(0, len(keys), 200)):
        e = min(len(keys), s + 200)
        np.savez(os.path.join(d, "chunk_%05d.npz" % ci), format=np.array(F.CHUNK_FORMAT), set=np.array(SET),
                 sides=np.array(["target", "opponent"]), ref=np.arange(s, e), precision=np.array("fp16"),
                 meta_sha256=np.array("m"), ckpt_bytes=np.array(1), code_sha256=np.array("c"), means=M[s:e],
                 failed=np.zeros(e - s, bool), counts=np.full((e - s, 2), 7, np.int32))
    return F.chunk_files(d), {"keys_sha256": A.keys_sha(keys), "games": len(keys)}


def p1_vectors(cand, qrows, owner_of_q, signal=2.0, seed=1):
    """Synthetic side-0 P1 vectors addressed by game: {gid or (qid, gid): [2, 770]}."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, signal, (N_PL, 770))
    out = {}
    for g in cand:
        v = np.zeros((2, 770), np.float32)
        v[0] = centres[int(g[1][1:3])] + rng.normal(0, 1, 770)
        out[g[0]] = v
    for q, games in qrows:
        for g in games:
            v = np.zeros((2, 770), np.float32)
            v[0] = centres[owner_of_q[q]] + rng.normal(0, 1, 770)
            out[(q, g[0])] = v
    return out


def keyed_matrix(cand, qrows, vec):
    keys = [g[0] for g in cand] + ["%s#%d" % (q, s) for q, games in qrows for s in range(1, len(games) + 1)]
    M = np.stack([vec[g[0]] for g in cand] + [vec[(q, g[0])] for q, games in qrows for g in games])
    return keys, M


class Smoke:
    pass


def build(tmp, cand, qrows, emb_c, emb_q, vec, model=None, p06_from=None):
    s = Smoke()
    s.tmp = tmp
    s.cp, s.qp, s.p02 = write_pool(tmp, cand, qrows, emb_c, emb_q)
    if p06_from is None:
        s.player = run_p06(tmp, s.cp, s.qp, s.p02, cand)
    else:                                # the same frozen 06 (LDAs, k*), its input SHAs rebound to this pool
        s.player = os.path.join(tmp, "p06player")
        shutil.copytree(p06_from, s.player)
        rp = os.path.join(s.player, "apply_report.json")
        r = st.read_json(rp)
        d = pf.Data(s.cp, s.p02, True, True, q_path=s.qp)
        r["inputs"] = dict(d.shas)
        st.write_json(rp, r)
    keys, M = keyed_matrix(cand, qrows, vec)
    s.vfiles, s.vr = write_chunks(os.path.join(tmp, "val"), keys, M)
    if model is None:
        lab = np.array([k.split("_")[0].replace("g", "p") if not k.startswith("q") else "" for k in keys])
        model = F.fit_p1_lda(M[:, 0, :], lab, np.array([not k.startswith("q") for k in keys]))
    s.model = model
    s.pool = V.load_pool(s.cp, s.qp, s.p02, s.player, s.vfiles, SET, s.vr, model, smoke=True)
    return s


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("p13full"))
    cand, qrows, owner_of_q = make_games()
    nrng = np.random.default_rng(0)
    emb_c = {g[0]: nrng.normal(size=256) for g in cand}
    emb_q = {(q, g[0]): nrng.normal(size=256) for q, games in qrows for g in games}
    vec = p1_vectors(cand, qrows, owner_of_q)
    s = build(os.path.join(tmp, "full"), cand, qrows, emb_c, emb_q, vec)
    s.raw = types.SimpleNamespace(cand=cand, qrows=qrows, emb_c=emb_c, emb_q=emb_q, vec=vec, owner_of_q=owner_of_q,
                                  root=tmp)
    # PREREG_10's apply on the same pool: the section 3 reference
    s.p10_out = os.path.join(tmp, "p10out")
    s.p10_rep = A.apply_core(s.cp, s.qp, s.p02, s.player, s.vfiles, SET, s.vr, s.model, 4.0, s.p10_out, smoke=True)
    return s


class smoke_repro:
    """Section 3's registered constants (PREREG_10's pool gate and selection on the real pool) replaced by the smoke
    pool's own values for the duration of a test."""

    def __init__(self, full):
        self.vals = {"REPRO_GATE": full.p10_rep["pool_gate"]["sr_10B_minus_selected"],
                     "REPRO_SELECTED": full.p10_rep["selected"], "REPRO_SHA06_PREFIX": "", "REPRO_SHA10_PREFIX": ""}

    def __enter__(self):
        self.old = {k: getattr(V, k) for k in self.vals}
        for k, x in self.vals.items():
            setattr(V, k, x)

    def __exit__(self, *exc):
        for k, x in self.old.items():
            setattr(V, k, x)
        return False


def wb(path):
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------- 1. identity view = PREREG_10 apply
def test_identity_view_equals_p10_apply(full):
    res = V.run_view(full.pool, V.identity_view(full.pool), 4.0)
    assert {n: round(x, 6) for n, x in res["sr06"].items()} == full.p10_rep["sr06"]
    assert {n: round(x, 6) for n, x in res["sr06_10B"].items()} == full.p10_rep["sr06_10B"]
    assert res["selected"] == full.p10_rep["selected"]
    assert res["gate_v3"] == full.p10_rep["pool_gate"]["sr_10B_minus_selected"]
    assert V.top5_bytes(pl, res["qids"], res["top06"]) == wb(os.path.join(full.player, "player_val_top5.csv"))
    assert V.top5_bytes(pl, res["qids"], res["top10"]) == wb(os.path.join(full.p10_out, "player_val_top5.csv"))


def test_sr06_view_strict_is_p10_sr06_with_p1(full):
    p = full.pool
    v = V.identity_view(p)
    ks = [len(v.q[q]) for q in v.qids]
    a = V.sr06_view(p.P6, p.ms, p.d.cand_ids, v.cand, ks, p.earr, p.models, p.combos, p.kst, p.Z, p.usable, 4.0, True)
    b = A.sr06_with_p1(p.P6, p.ms, p.d.cand_ids, v.cand, ks, p.earr, p.models, p.combos, p.kst, p.Z, p.usable, 4.0)
    assert a == b


# ---------------------------------------------------------------- 2. strict=False == strict=True at 100 games
def test_strict_false_equals_strict_true_at_100_games(full):
    p = full.pool
    v = V.identity_view(p)
    assert all(len(v.cand[c]) == 100 for c in p.d.cand_ids)
    cg = {c: [str(p.ms.gid[r]) for r in v.cand[c]] for c in p.d.cand_ids}
    ks = [len(v.q[q]) for q in v.qids]
    assert st.sr06_rounds(cg, ks, strict=False) == st.sr06_rounds(cg, ks, strict=True)
    args = (p.P6, p.ms, p.d.cand_ids, v.cand, ks, p.earr, p.models, p.combos, p.kst, p.Z, p.usable, 4.0)
    assert V.sr06_view(*args, False) == V.sr06_view(*args, True)


def test_strict_false_short_candidates_run():
    cg = {"a": ["x%d" % i for i in range(2)], "b": ["y%d" % i for i in range(37)]}
    rr = st.sr06_rounds(cg, [5, 9], strict=False)
    with pytest.raises(ValueError):
        st.sr06_rounds(cg, [5, 9], strict=True)
    for items in rr:
        for p_, q, refs in items:
            assert not set(q) & set(refs)
            assert set(q) | set(refs) <= set(cg[p_])
    assert any(not q for items in rr for _p, q, _r in items)        # empty pseudo questions occur (not asked)


# ---------------------------------------------------------------- 3. black groups == white rows deleted
def check_black_equals_deleted(s, root):
    r = s.raw
    cand_b = [g for g in r.cand if g[2] == "B"]
    q_b = [(q, [g for g in games if g[2] == "B"]) for q, games in r.qrows]
    q_b = [(q, g) for q, g in q_b if g]          # a question without a black game cannot be written; never in view
    pv = V.black_view(s.pool)
    assert set(pv.qids) <= {q for q, _g in q_b}
    dele = build(os.path.join(root, "deleted"), cand_b, q_b, r.emb_c, r.emb_q, r.vec, model=s.model,
                 p06_from=s.player)
    dv = V.black_view(dele.pool)
    assert all(c == "B" for c in dele.pool.colour)
    assert dv.qids == pv.qids and 0 < len(pv.qids) < len(s.pool.d.qids)       # some questions drop out
    assert [len(dv.q[q]) for q in dv.qids] == [len(pv.q[q]) for q in pv.qids]
    assert {c: len(x) for c, x in dv.cand.items()} == {c: len(x) for c, x in pv.cand.items()}
    a = V.run_view(s.pool, pv, 4.0)
    b = V.run_view(dele.pool, dv, 4.0)
    assert a["selected"] == b["selected"]
    for k in ("sr06", "sr06_10B"):
        assert a[k].keys() == b[k].keys()
        for n in a[k]:
            assert abs(a[k][n] - b[k][n]) <= 1e-12, (k, n)
    for t in ("top06", "top10"):
        assert V.top5_bytes(pl, a["qids"], a[t]) == V.top5_bytes(pl, b["qids"], b[t]), t
    return pv


def test_black_groups_equal_white_rows_deleted(full):
    check_black_equals_deleted(full, full.raw.root)


@pytest.fixture(scope="module")
def short(tmp_path_factory):
    """The same pool with p00 left with 2 black candidate games and p01 with 3 (the view's short candidates: empty
    pseudo-question parts occur under strict=False)."""
    tmp = str(tmp_path_factory.mktemp("p13short"))
    cand, qrows, owner_of_q = make_games()
    keep = {"p00": 2, "p01": 3}
    seen = {}
    out = []
    for g in cand:
        if g[1] in keep:
            seen[g[1]] = seen.get(g[1], 0) + 1
            g = (g[0], g[1], "B" if seen[g[1]] <= keep[g[1]] else "W", g[3])
        out.append(g)
    cand = out
    nrng = np.random.default_rng(0)
    emb_c = {g[0]: nrng.normal(size=256) for g in cand}
    emb_q = {(q, g[0]): nrng.normal(size=256) for q, games in qrows for g in games}
    vec = p1_vectors(cand, qrows, owner_of_q)
    s = build(os.path.join(tmp, "full"), cand, qrows, emb_c, emb_q, vec)
    s.raw = types.SimpleNamespace(cand=cand, qrows=qrows, emb_c=emb_c, emb_q=emb_q, vec=vec, owner_of_q=owner_of_q,
                                  root=tmp)
    return s


def test_short_candidates_black_groups_equal_white_rows_deleted(short):
    pv = check_black_equals_deleted(short, short.raw.root)
    assert len(pv.cand["p00"]) == 2 and len(pv.cand["p01"]) == 3
    p = short.pool
    cg = {c: [str(p.ms.gid[r]) for r in pv.cand[c]] for c in p.d.cand_ids}
    rr = st.sr06_rounds(cg, [len(pv.q[q]) for q in pv.qids], strict=False)
    assert any(not q for items in rr for _p, q, _r in items)          # empty pseudo questions: not asked


# ---------------------------------------------------------------- 4. view_core end to end
def test_view_core_end_to_end(full, tmp_path):
    p = full.pool
    n_view = len(V.black_view(p).qids)
    out = str(tmp_path / "out")
    ref06 = os.path.join(full.player, "player_val_top5.csv")
    with smoke_repro(full):
        rep, a, b = V.view_core(p, 4.0, full.p10_rep, ref06, os.path.join(full.p10_out, "player_val_top5.csv"), out,
                                full.qp, full.cp, n_expected=n_view)
    assert rep["reproduction"]["reproduced"] is True
    assert rep["view"]["n_view"] == n_view and rep["gates"]["V1"]["passed"] is True
    assert wb(os.path.join(out, V.OUT_A)) == a and wb(os.path.join(out, V.OUT_B)) == b
    assert rep["format_check"][V.OUT_A]["returncode"] == 0 and rep["format_check"][V.OUT_B]["returncode"] == 0
    assert rep["nonview_rows_identical"] is True
    ref = wb(ref06).decode().split("\n")
    al, bl = a.decode().split("\n"), b.decode().split("\n")
    assert len(al) == len(bl) == len(ref) and al[0] == ref[0]
    vs = set(rep["view"]["view_question_ids"])
    for x, y, z in zip(al[1:-1], bl[1:-1], ref[1:-1]):
        assert x.split(",")[0] == y.split(",")[0] == z.split(",")[0]
        if x.split(",")[0] not in vs:
            assert x == y == z
    assert rep["margin"]["ci95_lower_must_exceed"] == pytest.approx(-0.010 * n_view / 400)
    g = rep["gates"]
    assert (rep["score_allowed"] == "pending_audit") == (g["V2"]["passed"] and g["V3"]["passed"])
    assert not any(f.endswith(".tmp.csv") for f in os.listdir(out))


def test_v1_refuses(full, tmp_path):
    p = full.pool
    n_view = len(V.black_view(p).qids)
    with smoke_repro(full), pytest.raises(F.Refused, match="n_view"):
        V.view_core(p, 4.0, full.p10_rep, os.path.join(full.player, "player_val_top5.csv"),
                    os.path.join(full.p10_out, "player_val_top5.csv"), str(tmp_path / "o"), full.qp, full.cp,
                    n_expected=n_view + 1, check=False)
    assert not os.path.exists(str(tmp_path / "o" / V.OUT_A))


def test_v1_problem_list():
    s = {"n_view": 263, "candidates_without_black": 0, "p1": {"failed_fraction": 0.005}}
    assert V.v1_problems(s) == []
    assert V.v1_problems(dict(s, n_view=262))
    assert V.v1_problems(dict(s, candidates_without_black=1))
    assert V.v1_problems(dict(s, p1={"failed_fraction": 0.0051}))


def test_reproduction_mismatch_stops(full, tmp_path):
    bad10 = str(tmp_path / "bad10.csv")
    with open(bad10, "wb") as f:
        f.write(wb(os.path.join(full.p10_out, "player_val_top5.csv")) + b"\n")
    with pytest.raises(F.Stop, match="section 3"):
        V.view_core(full.pool, 4.0, full.p10_rep, os.path.join(full.player, "player_val_top5.csv"), bad10,
                    str(tmp_path / "o"), full.qp, full.cp, check=False)
    # the pool gate 1e-9 away from the reference (tolerance 1e-12)
    with smoke_repro(full):
        V.REPRO_GATE += 1e-9
        with pytest.raises(F.Stop, match="pool_gate_equal"):
            V.view_core(full.pool, 4.0, full.p10_rep, os.path.join(full.player, "player_val_top5.csv"),
                        os.path.join(full.p10_out, "player_val_top5.csv"), str(tmp_path / "o2"), full.qp,
                        full.cp, check=False)


def test_colour_mismatch_refused(full, tmp_path):
    r = full.raw
    p02 = str(tmp_path / "p02")
    shutil.copytree(full.p02, p02)
    dp = os.path.join(p02, "dumps", "cand_games.npz")
    with np.load(dp, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    arr["color"] = arr["color"].copy()
    arr["color"][0] = "W" if arr["color"][0] == "B" else "B"
    np.savez(dp, **arr)
    with pytest.raises(F.Refused):
        V.load_pool(full.cp, full.qp, p02, full.player, full.vfiles, SET, full.vr, full.model, smoke=True)
    assert r.cand


def test_decision():
    assert V.view_decision(True, True, True, True, True, True)[0] == "pending_audit"
    for i in range(6):
        args = [True] * 6
        args[i] = False
        allowed, reasons = V.view_decision(*args)
        assert allowed is False and len(reasons) == 1


def test_constants():
    assert V.N_VIEW_EXPECTED == 263 and V.MIN_VIEW_GAMES == 5 and V.BLACK == "B"
    assert V.GATE_V2_MIN == 0.0 and V.GATE_V3_MIN == 0.0 and V.NONINF_PER_400 == -0.010
    assert V.REPRO_GATE == -0.019891447034757093 and V.REPRO_TOL == 1e-12 and V.REPRO_SELECTED == "V2"
    assert V.REPRO_SHA06_PREFIX == "094223f6" and V.REPRO_SHA10_PREFIX == "0efd160d"
    assert V.FORMATS13["audit"] == "c10-prereg13 player prescore_audit v1"
    with open(os.path.join(TOOLS, "t06_player.py"), encoding="utf-8") as f:
        assert "GATE06_MIN = 0.0 " in f.read()


def test_audit_binding():
    rep = {"outputs_sha256": {V.OUT_A: "a", V.OUT_B: "b"},
           "inputs_sha256": {"p1_lda.npz": "l", "frozen_config.json": "f", "val_report.json": "v"}}
    au = {"format": V.FORMATS13["audit"], "program_errors_affecting_view": [], "independent_recompute_identical": True,
          "outputs_sha256": {V.OUT_A: "a", V.OUT_B: "b", V.REPORT: "r"},
          "inputs_sha256": {"p1_lda.npz": "l", "frozen_config.json": "f", "val_report.json": "v"}}
    t0, t1 = "2026-10-14T10:00:00+08:00", "2026-10-14T11:00:00+08:00"
    assert V.audit_binding(au, rep, "r", t1, t0) == []
    assert V.audit_binding(au, rep, "r", t0, t1)                       # audit not later
    assert V.audit_binding(au, rep, "x", t1, t0)                       # report changed
    assert V.audit_binding(dict(au, independent_recompute_identical=False), rep, "r", t1, t0)
    assert V.audit_binding(dict(au, outputs_sha256={V.OUT_A: "a", V.OUT_B: "z", V.REPORT: "r"}), rep, "r", t1, t0)
    assert V.audit_binding(dict(au, program_errors_affecting_view=None), rep, "r", t1, t0)


def test_audit_state_and_finalize(tmp_path):
    d = str(tmp_path)
    assert V.audit_state(d)[0] == "missing"
    rep = {"score_allowed": "pending_audit", "score_blocked_reasons": ["prescore_audit.json of these outputs ..."]}
    with pytest.raises(F.Refused, match="not committed"):
        V.finalize_decision(rep, "missing", None)
    with pytest.raises(F.Refused, match="not an audit"):
        V.finalize_decision(rep, "unbound", {"binding_problems": ["x"]})
    assert V.finalize_decision(rep, "ok", {}) == (True, [])
    assert V.finalize_decision(rep, "bug", {})[0] is False
    with pytest.raises(F.Refused, match="other reasons"):
        V.finalize_decision({"score_allowed": False, "score_blocked_reasons": ["V3 not passed"]}, "ok", {})
