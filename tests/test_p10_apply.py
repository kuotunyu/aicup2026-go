"""tools/p10_apply.py apply_core end to end on synthetic smoke data (PREREG_10 2.5): 06 is first produced by
tools/p06_player.py apply --smoke (as tests/test_p06_player.py::test_apply_smoke), then apply_core must reproduce it
byte for byte, recompute the same sr06 means, and with w* = 0 write exactly 06's file. Synthetic P1 chunks; nothing
under data/local is opened; tools/score.py is never used.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q tests/test_p10_apply.py
"""
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
import p06_player as pl  # noqa: E402
import p06_style as st  # noqa: E402
import p10_apply as A  # noqa: E402
import p10_fuse as F  # noqa: E402

LET = "abcdefghijklmnopqrs"
N_PL = 8


def _sgf(rng, n):
    return "(;GM[1]SZ[19]" + "".join(";%s[%s]" % ("B" if i % 2 == 0 else "W", rng.choice(LET) + rng.choice(LET))
                                     for i in range(n)) + ")"


def make_smoke(tmp, n_pl=N_PL, seed=0):
    """Same synthetic pool as tests/test_p06_player.py make_smoke (copied: test modules do not import each other)."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
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
        w.writerow(["question_id", "num_games"] + ["sgf_%d" % i for i in range(1, 21)]
                   + ["color_%d" % i for i in range(1, 21)])
        for q, games in qrows:
            w.writerow([q, len(games)] + [g[3] for g in games] + [""] * (20 - len(games))
                       + [g[2] for g in games] + [""] * (20 - len(games)))
    p02 = os.path.join(tmp, "p02")
    os.makedirs(os.path.join(p02, "dumps"))
    np.savez(os.path.join(p02, "dumps", "cand_games.npz"), game_id=np.array([g[0] for g in cand]),
             player_id=np.array([g[1] for g in cand]), color=np.array([g[2] for g in cand]),
             emb_sum=nrng.normal(size=(len(cand), 256)).astype(np.float32), n_windows=np.full(len(cand), 50, np.int32))
    qid, slot, qc = [], [], []
    for q, games in qrows:
        for s, g in enumerate(games, 1):
            qid.append(q)
            slot.append(s)
            qc.append(g[2])
    np.savez(os.path.join(p02, "dumps", "query_games.npz"), question_id=np.array(qid), slot=np.array(slot, np.int32),
             color=np.array(qc), emb_sum=nrng.normal(size=(len(qid), 256)).astype(np.float32),
             n_windows=np.full(len(qid), 50, np.int32), parsed=np.ones(len(qid), bool))
    owner_of_q = {"q%02d" % ((i * 5) % n_pl): i for i in range(n_pl)}
    return cp, qp, p02, cand, owner_of_q


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
    """06 on the smoke pool through tools/p06_player.py (its own smoke mode): the reference of apply_core."""
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


def p1_chunks(tmp, cp, qp, owner_of_q, signal=2.0, seed=1):
    """Synthetic P1 chunks in the extraction order: candidate games, then question slots."""
    keys, cols = A.val_keys(pf.read_candidates(cp), pf.read_questions(qp))
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, signal, (N_PL, 770))
    owner = [int(k[1:3]) if not k.startswith("q") else owner_of_q[k.split("#")[0]] for k in keys]
    M = np.zeros((len(keys), 2, 770), np.float32)
    M[:, 0, :] = centres[owner] + rng.normal(0, 1, (len(keys), 770))
    d = os.path.join(tmp, "val")
    os.makedirs(d)
    for ci, s in enumerate(range(0, len(keys), 200)):
        e = min(len(keys), s + 200)
        np.savez(os.path.join(d, "chunk_%05d.npz" % ci), format=np.array(F.CHUNK_FORMAT), set=np.array("p10val_t"),
                 sides=np.array(["target", "opponent"]), ref=np.arange(s, e), precision=np.array("fp16"),
                 meta_sha256=np.array("m"), ckpt_bytes=np.array(1), code_sha256=np.array("c"), means=M[s:e],
                 failed=np.zeros(e - s, bool), counts=np.full((e - s, 2), 7, np.int32))
    vr = {"keys_sha256": A.keys_sha(keys), "games": len(keys)}
    lab = np.array(["p%02d" % o for o in owner])
    model = F.fit_p1_lda(M[:, 0, :], lab, np.array([not k.startswith("q") for k in keys]))
    return F.chunk_files(d), vr, model


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("p10apply"))
    cp, qp, p02, cand, owner_of_q = make_smoke(tmp)
    player = run_p06(tmp, cp, qp, p02, cand)
    vfiles, vr, model = p1_chunks(tmp, cp, qp, owner_of_q)
    return types.SimpleNamespace(tmp=tmp, cp=cp, qp=qp, p02=p02, player=player, vfiles=vfiles, vr=vr, model=model)


def test_apply_core_reproduces_06_and_writes_10b(smoke):
    out = os.path.join(smoke.tmp, "out1")
    rep = A.apply_core(smoke.cp, smoke.qp, smoke.p02, smoke.player, smoke.vfiles, "p10val_t", smoke.vr, smoke.model,
                       4.0, out, smoke=True)
    rep06 = st.read_json(os.path.join(smoke.player, "apply_report.json"))
    assert rep["p06_reproduction"]["identical"] is True
    assert rep["sr06_equals_06_report"] is True
    assert rep["selected"] == rep06["selected"]
    assert rep["format_check"]["returncode"] == 0
    assert rep["p1"]["failed"] == 0 and rep["p1"]["usable"] == rep["p1"]["games"]
    assert rep["score_allowed"] in (False, "pending_audit")
    assert (rep["score_allowed"] == "pending_audit") == rep["pool_gate"]["passed"]
    assert rep["outputs_sha256"]["player_val_top5.csv"] == st.sha256_lf(os.path.join(out, "player_val_top5.csv"))
    assert not os.path.exists(os.path.join(out, "p06_repro.tmp.csv"))
    # a strong synthetic P1 signal: 10B finds every owner first, and the pseudo-pool gate passes
    with open(os.path.join(out, "player_val_top5.csv"), encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    assert len(rows) == N_PL
    assert rep["pool_gate"]["passed"] is True


def test_w0_writes_exactly_06(smoke):
    out = os.path.join(smoke.tmp, "out0")
    rep = A.apply_core(smoke.cp, smoke.qp, smoke.p02, smoke.player, smoke.vfiles, "p10val_t", smoke.vr, smoke.model,
                       0.0, out, smoke=True)
    with open(os.path.join(out, "player_val_top5.csv"), "rb") as a, \
            open(os.path.join(smoke.player, "player_val_top5.csv"), "rb") as b:
        assert a.read() == b.read()
    assert rep["changed_top1_vs_06"] == 0
    assert abs(rep["pool_gate"]["sr_10B_minus_selected"]) < 1e-12


def test_refusals(smoke):
    with pytest.raises(F.Refused, match="game order"):
        A.apply_core(smoke.cp, smoke.qp, smoke.p02, smoke.player, smoke.vfiles, "p10val_t",
                     dict(smoke.vr, keys_sha256="0" * 64), smoke.model, 4.0, os.path.join(smoke.tmp, "o2"), smoke=True)
    with pytest.raises(F.Refused, match="set"):
        A.apply_core(smoke.cp, smoke.qp, smoke.p02, smoke.player, smoke.vfiles, "p10val", smoke.vr, smoke.model, 4.0,
                     os.path.join(smoke.tmp, "o3"), smoke=True)


def test_06_mismatch_stops(smoke, tmp_path):
    import shutil
    p2 = str(tmp_path / "p06copy")
    shutil.copytree(smoke.player, p2)
    with open(os.path.join(p2, "player_val_top5.csv"), "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(F.Stop, match="differs"):
        A.apply_core(smoke.cp, smoke.qp, smoke.p02, p2, smoke.vfiles, "p10val_t", smoke.vr, smoke.model, 4.0,
                     str(tmp_path / "o4"), smoke=True)
