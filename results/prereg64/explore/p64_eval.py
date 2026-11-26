"""prereg64 (competition 2): OPP3 (c49) + anchor fusion on Q25, judged once on inner against Q59; merge simulation.
CPU only, .venv (numpy / scipy), 4 cores, <= 3 GB (p58_lib.limits). Imports prereg58 / prereg59 / prereg23 / tools
unchanged.

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
    S=results/prereg64/explore;  PY=.venv/Scripts/python.exe
    (.venv-gpu) $S/p64_head.py      -> data/prereg64/c49_head.npz, $S/head.json
    $PY $S/p64_eval.py kernel       -> $S/kernel.json, data/prereg64/prototypes.npz  (D-only table; no inner game)
    $PY $S/p64_eval.py seam         -> $S/seam.json  (Q25, Q59 bit for bit, anchor-only = Q25; no OPP3 number)
    $PY $S/p64_eval.py eval         -> $S/eval.json  (once; all of the above committed and clean)
    $PY $S/p64_eval.py results      -> $S/RESULTS.md
    $PY $S/p64_eval.py mergesim     -> $S/merge_sim.json  (E n H simulation, no inner)
Never data/local, never tools/score.py, never a test-set file here (the test-day tool is tools/t64_rank.py).
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P58 = os.path.join(REPO, "results", "prereg58", "explore")
P59 = os.path.join(REPO, "results", "prereg59", "explore")
for p in (P58, P59, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p58_eval as E58  # noqa: E402
import p58_lib as K58  # noqa: E402
import p59_eval as E59  # noqa: E402
import p59_lib as K59  # noqa: E402
import p64_lib as K  # noqa: E402

TERM = "OPP3"
HARD_STOP = "2026-11-23T12:00:00"
REF_Q59 = 0.7539701874530542
REF_Q59_MONO = 0.7552848066878524
log, git, rel, committed_clean, peak_gb = E58.log, E58.git, E58.rel, E58.committed_clean, E58.peak_gb
FROZEN = os.path.join(HERE, "frozen.json")
NOTES = os.path.join(HERE, "NOTES.md")
HEAD_REC = os.path.join(HERE, "head.json")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def code_state():
    mods = E58.loaded_repo_modules()
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "prereg64_dirty": git("status", "--porcelain", "--", "results/prereg64"),
            "imported_dirty": git("status", "--porcelain", "--", *mods) if mods else "",
            "sha256_lf": {x: K58.sha_lf(os.path.join(REPO, *x.split("/"))) for x in mods}}


def check_vec(v, path, side=None):
    cfg = v["embed_cfg"]
    if v["ckpt_sha256"] != K.C49_CKPT_SHA or cfg["weights"] != "raw" or cfg["tta"] != 1 or not np.isfinite(v["emb"]).all():
        raise SystemExit(f"{path}: not c49 raw tta1 vectors")
    if side and (v["rows_cfg"] or {}).get("side") != side:
        raise SystemExit(f"{path}: not the {side} side")


# ================================================================ prototypes and kernel (no inner game)
def build_prototypes():
    """E (store_E: store_row = rows of the prereg06 feature store meta, p36_embed.store_games) + the 1,504 proto
    owners (rows_proto: store_row = rows of the prereg49 move cache, p58_rows.py)."""
    sys.path.insert(0, os.path.join(REPO, "results", "prereg36", "explore"))
    sys.path.insert(0, os.path.join(REPO, "results", "prereg49", "explore"))
    import p49_lib as L49
    cache = L49.Cache49()
    cpid, crk = np.asarray(cache.pid).astype(str), np.asarray(cache.rank).astype(str)
    meta = E59.feats_meta()
    mpid, mrk = meta["player_id"].astype(str), meta["rank"].astype(str)
    E_ids = set(L49.e_players())
    ids, P, rk, src, rec = [], [], [], [], {}
    for name, path in (("E", K.OWN_E), ("X", K.OWN_PROTO)):
        v = E58.load_vec(path)
        check_vec(v, path, "own" if name == "X" else None)
        rows = v["store_row"]
        if name == "X":
            if not np.array_equal(rows, np.load(K.PROTO_ROWS)):
                raise SystemExit("rows_proto.npz rows differ from proto_rows.npy")
            pid, prank = cpid[rows], K58.rank_index(crk[rows])
        else:
            pid, prank = mpid[rows], K58.rank_index(mrk[rows])
            if set(pid.tolist()) != E_ids:
                raise SystemExit("store_E is not exactly the E players' games")
        u, pr, n = K59.prototypes(v["emb"], v["count"], pid)
        first = {p: i for i, p in reversed(list(enumerate(pid.tolist())))}
        ids.append(u), P.append(pr), rk.append(prank[[first[p] for p in u.tolist()]]), src.append([name] * len(u))
        rec[name] = {"file": rel(path), "sha256": K58.sha256_file(path), "games": int(len(pid)),
                     "games_with_sample": int((v["count"] > 0).sum()), "prototypes": int(len(u)),
                     "games_per_prototype_min": int(n.min())}
    out = {"player_id": np.concatenate(ids), "proto": np.concatenate(P), "rank": np.concatenate(rk).astype(np.int64),
           "src": np.array(sum(src, []))}
    if len(set(out["player_id"].tolist())) != len(out["player_id"]):
        raise SystemExit("a player with two prototypes")
    rec["total"] = int(len(out["player_id"]))
    rec["per_rank"] = np.bincount(out["rank"], minlength=K.L).tolist()
    return out, rec


def load_prototypes():
    with np.load(K58.guard(K.PROTO), allow_pickle=False) as z:
        return {"player_id": np.asarray(z["player_id"]).astype(str), "proto": np.asarray(z["proto"], np.float64),
                "rank": np.asarray(z["rank"], np.int64), "src": np.asarray(z["src"]).astype(str)}


def inner_player_ids():
    _inner, _val, _L49 = E59.split_players()
    return set(_inner) | set(E58.inner_keys_arrays()["player_id"].astype(str).tolist())


def cmd_kernel(a):
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(K.KERNEL):
        raise SystemExit("kernel.json exists (once)")
    for p in (FROZEN, NOTES, HEAD_REC):
        if not committed_clean(p):
            raise SystemExit(f"{rel(p)} is not committed and clean")
    cs = code_state()
    if cs["prereg64_dirty"]:
        raise SystemExit(f"results/prereg64 not committed and clean: {cs['prereg64_dirty']}")
    Wn, hid, wr = K.load_head()
    pr, prec = build_prototypes()
    inner_ids = inner_player_ids()
    overlap = {"prototypes_and_inner": len(set(pr["player_id"].tolist()) & inner_ids),
               "prototypes_and_head": len(set(pr["player_id"].tolist()) & set(hid.tolist()))}
    if any(overlap.values()):
        raise SystemExit(f"overlap {overlap}")
    np.savez(K.PROTO + ".tmp.npz", **pr, ckpt_sha256=np.array(K.C49_CKPT_SHA))
    os.replace(K.PROTO + ".tmp.npz", K.PROTO)
    # the D players' games, opponent side
    v = E58.load_vec(K.OPP_D)
    check_vec(v, K.OPP_D)
    with np.load(K58.guard(K58.FEATS_META), allow_pickle=False) as z:
        meta = {k: np.asarray(z[k]).astype(str)[v["store_row"]] for k in ("rank", "game_id", "player_id", "color")}
    sys.path.insert(0, os.path.join(REPO, "results", "prereg36", "explore"))
    import p36_lib as L36
    _E, D, _T, _p = L36.splits(L36.Cache())
    players = sorted(set(meta["player_id"].tolist()))
    if set(players) != set(D) or set(players) & inner_ids:
        raise SystemExit("the table games are not exactly the D players' games")
    hidx = {p: i for i, p in enumerate(hid.tolist())}
    own = np.array([hidx.get(p, -1) for p in meta["player_id"].tolist()], np.int64)
    if (own < 0).any():
        raise SystemExit("a D player is not a head class")
    if not (wr[own] == K58.rank_index(meta["rank"])).all():
        raise SystemExit("a D player's head class has another rank")
    hd, jrec = E58.hd_join(meta["rank"], meta["game_id"], meta["player_id"], meta["color"])
    r_g, h_g, c_g = K58.rank_index(meta["rank"]), K58.hd_group(hd), K58.colour_index(meta["color"])
    ok = v["count"] > 0
    q = K.q_opp(v["emb"], Wn, wr, pr["proto"], pr["rank"], own)
    counts = K59.table_counts(q, r_g, h_g, c_g, ok)
    M = K59.table(counts)
    f_g = np.array([K58.player_fold(p) for p in meta["player_id"].tolist()], np.int64)
    ll, lln, by_fold = K59.heldout_ll(q, r_g, h_g, c_g, f_g, ok)
    rep = {"format": "c10-prereg64 kernel v1", "created": now(), "seconds": round(time.time() - t0, 1), "code": cs,
           "what": "frozen.json: c49 head rows (own row left out) + prototypes (E + 1,504), tau1 50 / tau2 30 / b 1; "
                   "M[h, c][r][s] = prereg59's construction on the D players' games only. No inner game is read.",
           "tau1": K.TAU1, "tau2": K.TAU2, "b": K.BIAS, "M": M.tolist(), "soft_counts": counts.tolist(),
           "index": {"h": ["HD 0", "HD 1", "HD 2+"], "c": ["player B", "player W"], "r": list(K58.RANKS),
                     "s": list(K58.RANKS)},
           "classes": {"head": int(len(wr)), "prototypes": prec, "union": int(len(wr) + len(pr["rank"]))},
           "prototypes_file": {"file": rel(K.PROTO), "sha256": K58.sha256_file(K.PROTO)},
           "overlap": overlap, "hd_join": jrec,
           "games": {"total": int(len(ok)), "with_sample": int(ok.sum()),
                     "hd_le1_record": int(((hd <= 1) & ok).sum()),
                     "per_h_c": [[int((ok & (h_g == h) & (c_g == c)).sum()) for c in range(2)] for h in range(3)]},
           "players": {"D": len(players), "per_rank": np.bincount(K58.rank_index(
               [meta["rank"][meta["player_id"] == p][0] for p in players]), minlength=K.L).tolist()},
           "record_heldout_mean_loglik": {"mean": ll, "normalised_over_r": lln, "by_fold": by_fold,
                                          "note": "tables from the other 4 D folds; record only, nothing chosen"},
           "record_mean_max_q": float(q[ok].max(axis=1).mean()),
           "vector_file": {"file": rel(K.OPP_D), "sha256": K58.sha256_file(K.OPP_D)},
           "peak_working_set_gb": peak_gb(st)}
    K58.write_json(K.KERNEL, rep)
    log(json.dumps({k: rep[k] for k in ("classes", "games", "record_heldout_mean_loglik", "record_mean_max_q",
                                        "seconds", "peak_working_set_gb")}))
    return 0


def load_kernel():
    k = K58.read_json(K.KERNEL)
    M = np.asarray(k["M"], np.float64)
    if M.shape != (3, 2, K.L, K.L) or not np.allclose(M.sum(axis=3), 1.0):
        raise SystemExit("kernel.json: not a [3, 2, 10, 10] table")
    if K58.sha256_file(K.PROTO) != k["prototypes_file"]["sha256"]:
        raise SystemExit("prototypes.npz is not the file of kernel.json")
    return M, load_prototypes(), k


# ================================================================ the inner pieces
def inner_sets(inner, terms, Emap):
    """(m, E list, X) per set, exactly as p08_rank_fuse.run_terms builds them."""
    r = E58.harness()[2]
    rf = r.rf
    di, lp = inner.d["inner"], inner.d["lp"]
    out = {}
    for which, X, S in (("fit", di["X_fit"], di["S_fit"]), ("eval", di["X_eval"], di["S_eval"]),
                        ("mono", di["X_mono"], di["S_mono"])):
        m = rf.pooled(S, lp["C5"], "log")
        Es = [Emap[t][which] if t in Emap else (X.mop if t == "OP" else rf.pooled(S, lp[t], "log")) for t in terms]
        out[which] = (m, Es, X)
    return out


def anchor(sets, theta, which, mask):
    r = E58.harness()[2]
    m, Es, X = sets[which]
    z, _t = r.p4.gen_logits(np.asarray(theta, np.float64), m[mask], [E[mask] for E in Es], X.n[mask])
    return z


def q59_build(inner, Emap, terms7):
    M, par, pr, _k = E59.load_kernel(K59.KERNEL, K59.PROTO)
    lg, _rec = E59.inner_term(inner, M, par, pr, False)
    di = inner.d["inner"]
    E = {w: K59.question_term(S, lg) for w, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"]))}
    return inner.run(terms7 + ["OPP2"], {**Emap, "OPP2": E})


def seam_checks(inner, terms7, Emap, q25, seam25):
    """Q25, Q59 bit for bit; Q25's own fold parameters reproduce its probabilities; the anchor alone decodes as Q25."""
    r = E58.harness()[2]
    rf = r.rf
    q59 = q59_build(inner, Emap, terms7)
    sets = inner_sets(inner, terms7, Emap)
    Xe, Xm = sets["eval"][2], sets["mono"][2]
    Pz = np.zeros((len(Xe), K.L))
    Pa = np.zeros((len(Xe), K.L))
    Pma = np.full((len(Xm), K.L), np.nan)
    for fr in q25["folds"]:
        f = fr["fold"]
        ev, mv = Xe.fold == f, Xm.fold == f
        z = anchor(sets, fr["theta"], "eval", ev)
        Pz[ev] = rf.softmax(z)
        Pa[ev] = rf.softmax(1.0 * K.log_softmax(z))
        Pma[mv] = rf.softmax(1.0 * K.log_softmax(anchor(sets, fr["theta"], "mono", mv)))
    E23 = E58.harness()[1]
    s_a = E23.scored(inner, Pa, Pma)
    rec = {"Q25": [q25["rb"], K58.REF_Q25], "Q25_mono": [q25["mono_rb"], K58.REF_Q25_MONO],
           "Q25_bitwise": seam25["Q25_bitwise"], "Q25_mono_bitwise": seam25["Q25_mono_bitwise"],
           "Q25_seam_ok": seam25["ok"],
           "Q59": [q59["rb"], REF_Q59], "Q59_mono": [q59["mono_rb"], REF_Q59_MONO],
           "Q59_bitwise": bool(q59["rb"] == REF_Q59), "Q59_mono_bitwise": bool(q59["mono_rb"] == REF_Q59_MONO),
           "Q25_fold_theta_reproduces_P": bool(np.array_equal(Pz, q25["P"])),
           "anchor_only_rule_b_equal_Q25": bool(np.array_equal(rf.rule_b_np(Pa), rf.rule_b_np(q25["P"]))
                                                and np.array_equal(rf.rule_b_np(Pma), rf.rule_b_np(q25["Pm"]))),
           "anchor_only_scores": [s_a["rb"], s_a["mono_rb"]],
           "anchor_only_scores_bitwise_Q25": bool(s_a["rb"] == q25["rb"] and s_a["mono_rb"] == q25["mono_rb"]),
           "anchor_only_max_abs_P_diff": float(np.abs(Pa - q25["P"]).max())}
    rec["ok"] = bool(rec["Q25_seam_ok"] and rec["Q59_bitwise"] and rec["Q59_mono_bitwise"]
                     and rec["Q25_fold_theta_reproduces_P"] and rec["anchor_only_rule_b_equal_Q25"]
                     and rec["anchor_only_scores_bitwise_Q25"])
    return rec, q59, sets


def cmd_seam(a):
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(K.SEAM):
        raise SystemExit("seam.json exists")
    _mods, inner, terms7, Emap, q25, seam25 = E58.build_q25()
    rec, _q59, _sets = seam_checks(inner, terms7, Emap, q25, seam25)
    rep = {"format": "c10-prereg64 seam v1", "created": now(), "code": code_state(),
           "what": "Q25 (p58_eval.build_q25) and Q59 (prereg59's committed kernel / prototypes / S0 inner vectors) on "
                   "inner bit for bit, Q25's fold parameters reproduce its probabilities, the anchor alone decodes as "
                   "Q25; no OPP3 number is computed", **rec, "seconds": round(time.time() - t0, 1),
           "peak_working_set_gb": peak_gb(st)}
    K58.write_json(K.SEAM, rep)
    log(json.dumps({k: v for k, v in rep.items() if k not in ("code",)}))
    return 0 if rec["ok"] else 1


def inner_opp3(inner, M, pr):
    Wn, hid, wr = K.load_head()
    v = E58.load_vec(K.OPP_INNER)
    check_vec(v, K.OPP_INNER, "opp")
    rows = np.load(K58.INNER_ROWS)
    if not np.array_equal(v["store_row"], rows) or v["rows_cfg"]["cache_dir"] != "data/prereg44/cache/inner":
        raise SystemExit("inner vectors are not of prereg58's inner rows")
    keys = inner.d["inner"]["keys"]
    kpid = np.asarray(keys["player_id"]).astype(str)
    if set(pr["player_id"].tolist()) & set(kpid.tolist()):
        raise SystemExit("an inner player has a prototype")
    hd, jrec = E58.hd_join(keys["rank"], keys["game_id"], keys["player_id"], keys["color"])
    h, c = K58.hd_group(hd), K58.colour_index(keys["color"])
    hidx = {p: i for i, p in enumerate(hid.tolist())}
    own = np.array([hidx.get(p, -1) for p in kpid[rows].tolist()], np.int64)
    q = K.q_opp(v["emb"], Wn, wr, pr["proto"], pr["rank"], own)
    vals = K.game_values(q, M, h[rows], c[rows], v["count"])
    lg = np.zeros((inner.n_inner, K.L))
    lg[rows] = vals
    if not np.isfinite(lg).all():
        raise SystemExit("not finite")
    rec = {"games_in_questions": int(len(rows)), "games_without_sample": int((v["count"] == 0).sum()),
           "own_head_row_left_out_games": int((own >= 0).sum()),
           "inner_question_players_that_are_head_classes": int(len(set(kpid[rows][own >= 0].tolist()))),
           "hd_join": jrec, "mean_max_q_record": float(q[v["count"] > 0].max(axis=1).mean()),
           "vector_file": {"file": rel(K.OPP_INNER), "sha256": K58.sha256_file(K.OPP_INNER)}}
    return lg, rec


def cmd_eval(a):
    st = K58.limits(log)
    t0 = time.time()
    if now() >= HARD_STOP:
        raise SystemExit("past the hard stop")
    for p in (K.KERNEL, K.SEAM, FROZEN, NOTES, HEAD_REC):
        if not committed_clean(p):
            raise SystemExit(f"{rel(p)} is not committed and clean")
    if not K58.read_json(K.SEAM)["ok"]:
        raise SystemExit("seam.json is not ok")
    if os.path.exists(K.EVAL) or os.path.exists(K.STARTED):
        raise SystemExit("eval.json / eval_started.json exist: judged once")
    M, pr, krec = load_kernel()
    mods, inner, terms7, Emap, q25, seam25 = E58.build_q25()
    E41, E23, r, K23 = mods
    rf = r.rf
    cs = code_state()
    if cs["prereg64_dirty"] or cs["imported_dirty"]:
        raise SystemExit(f"not committed and clean: {cs['prereg64_dirty']} {cs['imported_dirty']}")
    srec, q59, sets = seam_checks(inner, terms7, Emap, q25, seam25)
    log(f"seams {srec['ok']}: Q25 {q25['rb']!r}, Q59 {q59['rb']!r} ({time.time() - t0:.0f}s)")
    if not srec["ok"]:
        raise SystemExit("seam failed inside the judgement run")
    K58.write_json(K.STARTED, {"format": "c10-prereg64 eval started", "created": now(), "code": cs})
    # ---- OPP3 (first inner number of it below)
    lg, trec = inner_opp3(inner, M, pr)
    di = inner.d["inner"]
    O = {w: K.question_term(S, lg) for w, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"]))}
    Xf, Xe, Xm = sets["fit"][2], sets["eval"][2], sets["mono"][2]
    P = np.zeros((len(Xe), K.L))
    Pm = np.full((len(Xm), K.L), np.nan)
    folds = []
    for fr in q25["folds"]:
        f = fr["fold"]
        tr, ev, mv = Xf.fold != f, Xe.fold == f, Xm.fold == f
        A_tr = K.log_softmax(anchor(sets, fr["theta"], "fit", tr))
        ws, frec = K.fit_weights([A_tr, O["fit"][tr]], Xf.y[tr])
        P[ev] = rf.softmax(K.fused(ws, [K.log_softmax(anchor(sets, fr["theta"], "eval", ev)), O["eval"][ev]]))
        Pm[mv] = rf.softmax(K.fused(ws, [K.log_softmax(anchor(sets, fr["theta"], "mono", mv)), O["mono"][mv]]))
        folds.append({"fold": f, "w_A": float(ws[0]), "w_O": float(ws[1]), "fit_questions": int(tr.sum()), **frec,
                      "params": {"w": {"Q25": float(ws[0]), TERM: float(ws[1])}}})
    if np.isnan(Pm).any():
        raise SystemExit("a single-colour question without a fold")
    s = E23.scored(inner, P, Pm)
    cand = {"P": P, "Pm": Pm, **s, "folds": folds}
    vs59 = inner.compare(cand, q59)
    vs25 = inner.compare(cand, q25)
    gates = K23.gates(vs59["inner_diff"], vs59["inner_ci95"], vs59["mono_diff"], vs59["mono_ci95"],
                      vs59["mono_by_colour"])
    gates25 = K23.gates(vs25["inner_diff"], vs25["inner_ci95"], vs25["mono_diff"], vs25["mono_ci95"],
                        vs25["mono_by_colour"])
    adopt = bool(gates["passed"])
    log(f"Q64 {cand['rb']!r} vs Q59 {q59['rb']!r}: {json.dumps(gates)}")
    ones = {w: np.ones(len(O[w]), bool) for w in ("eval", "mono")}
    alone = E23.alone(inner, {w: O[w] for w in ("eval", "mono")}, ones)
    rep = {"format": "c10-prereg64 eval v1", "created": now(), "seconds": round(time.time() - t0, 1), "code": cs,
           "what": "PLAN.md steps 1-3, judged once on inner: Q64 = w_A * log_softmax(Q25 with its fold parameters) + "
                   "w_O * OPP3 (c49 opponent term, 10 x mean, flat), weights by Gemini 04's scalar fit per player fold; "
                   "rule B; vs Q59 (the current upload). Inner used for the third time (prereg58, 59, 64).",
           "adopt_steps_1_3": adopt,
           "status": ("steps 1-3 passed all three gates vs Q59" if adopt else "gates vs Q59 not passed: not adopted"),
           "seam": srec, "term": trec, "kernel_sha256_lf": K58.sha_lf(K.KERNEL),
           "candidate": {"name": "Q64", "inner_rank_balanced": cand["rb"], "single_colour_rank_balanced": cand["mono_rb"],
                         "single_colour_by_colour": {c: cand["mrec"]["by_colour"][c]["rank_balanced"]
                                                     for c in cand["mrec"]["by_colour"]},
                         "vs_Q59": vs59, "vs_Q25": vs25},
           "baselines": {"Q59": {"inner_rank_balanced": q59["rb"], "single_colour_rank_balanced": q59["mono_rb"]},
                         "Q25": {"inner_rank_balanced": q25["rb"], "single_colour_rank_balanced": q25["mono_rb"]}},
           "gates": gates, "record_gates_vs_Q25": gates25,
           "weights_by_fold": [{k: fr[k] for k in ("fold", "w_A", "w_O", "loss", "nit", "success", "fit_questions")}
                               for fr in folds],
           "record_only": {"OPP3_alone_rule_b": alone,
                           "by_game_count_vs_Q59": E23.by_count(inner, cand, q59),
                           "by_level_vs_Q59": E23.by_level(inner, cand, q59),
                           "by_game_count_vs_Q25": E23.by_count(inner, cand, q25),
                           "by_level_vs_Q25": E23.by_level(inner, cand, q25)},
           "peak_working_set_gb": peak_gb(st)}
    K58.write_json(K.EVAL, rep)
    log(f"-> {K.EVAL}: {rep['status']} ({time.time() - t0:.0f}s, peak {peak_gb(st)} GB)")
    return 0


# ================================================================ RESULTS.md
def results_text(e, k, ms=None):
    c, g = e["candidate"], e["gates"]
    v9, v5 = c["vs_Q59"], c["vs_Q25"]
    ro = e["record_only"]
    A = []
    a = A.append
    a("# prereg64 (competition 2): OPP3 (c49 opponent term) + anchor fusion on Q25 - steps 1-3 "
      + ("PASS" if e["adopt_steps_1_3"] else "NOT adopted"))
    a("")
    a(f"Written mechanically from `eval.json` ({e['created']}) and `kernel.json` ({k['created']}); numbers only. "
      "PLAN.md, NOTES.md, frozen.json. Inner used for the third time (prereg58, prereg59, prereg64).")
    a("")
    a("## Kernel (D players only, frozen settings from Gemini)")
    a("")
    pc = k["classes"]["prototypes"]
    a(f"- Classes {k['classes']['union']}: c49 head {k['classes']['head']} + prototypes {pc['total']} "
      f"(E {pc['E']['prototypes']}, others {pc['X']['prototypes']}); tau1 {k['tau1']}, tau2 {k['tau2']}, b {k['b']}.")
    a(f"- D games {k['games']['total']} ({k['games']['with_sample']} with a sample; HD <= 1: "
      f"{k['games']['hd_le1_record']}); record held-out mean log-likelihood {k['record_heldout_mean_loglik']['mean']:.4f}, "
      f"mean max q {k['record_mean_max_q']:.4f}.")
    a("")
    a("## Seams")
    a("")
    s = e["seam"]
    a(f"- Q25 {s['Q25'][0]!r} / single colour {s['Q25_mono'][0]!r}: bit for bit {s['Q25_bitwise']} / "
      f"{s['Q25_mono_bitwise']}; Q59 {s['Q59'][0]!r} / {s['Q59_mono'][0]!r}: bit for bit {s['Q59_bitwise']} / "
      f"{s['Q59_mono_bitwise']}; anchor alone = Q25 {s['anchor_only_scores_bitwise_Q25']}.")
    a("")
    a("## Judgement (inner, once)")
    a("")
    a("| | Q59 | Q64 | diff | 95% interval | gate |")
    a("|---|---|---|---|---|---|")
    a(f"| inner eval | {e['baselines']['Q59']['inner_rank_balanced']:.7f} | {c['inner_rank_balanced']:.7f} | "
      f"{v9['inner_diff']:+.5f} | [{v9['inner_ci95'][0]:+.5f}, {v9['inner_ci95'][1]:+.5f}] | "
      f"G1 {'pass' if g['G1']['passed'] else 'no'}; G2 {'pass' if g['G2']['passed'] else 'no'} |")
    bc = v9["mono_by_colour"]
    a(f"| single colour | {e['baselines']['Q59']['single_colour_rank_balanced']:.7f} | "
      f"{c['single_colour_rank_balanced']:.7f} | {v9['mono_diff']:+.5f} | [{v9['mono_ci95'][0]:+.5f}, "
      f"{v9['mono_ci95'][1]:+.5f}] | G4 {'pass' if g['G4']['passed'] else 'no'} (B {bc.get('B'):+.4f}, "
      f"W {bc.get('W'):+.4f}) |")
    a("")
    a(f"**steps 1-3: {e['status']}.**")
    a("")
    a("## Record only")
    a("")
    a(f"- vs Q25: inner {v5['inner_diff']:+.5f} [{v5['inner_ci95'][0]:+.5f}, {v5['inner_ci95'][1]:+.5f}], single "
      f"colour {v5['mono_diff']:+.5f} [{v5['mono_ci95'][0]:+.5f}, {v5['mono_ci95'][1]:+.5f}].")
    a("- Weights by fold (w_A, w_O): " + "; ".join(f"{w['w_A']:.4f}, {w['w_O']:.4f}" for w in e["weights_by_fold"]) + ".")
    al = ro["OPP3_alone_rule_b"]
    a(f"- OPP3 alone (rule B): {al['inner_rank_balanced']:.4f} (single colour {al['single_colour_rank_balanced']:.4f}).")
    a("- By rank class (eval, Q64 - Q59): " + ", ".join(f"{x['level']} {x['diff']:+.4f}"
                                                         for x in ro["by_level_vs_Q59"]["eval"]) + ".")
    b9 = ro["by_game_count_vs_Q59"]
    a("- Single colour by number of games (vs Q59): " + "; ".join(
        f"{n} {b9['mono_by_k'][n]['diff']:+.4f} [{b9['mono_by_k'][n]['ci95'][0]:+.4f}, {b9['mono_by_k'][n]['ci95'][1]:+.4f}]"
        for n in sorted(b9["mono_by_k"], key=int)) + ".")
    a("- Eval questions with exactly n games (vs Q59): " + "; ".join(
        f"{n} {b9['eval_by_n_exact'][n]['diff']:+.4f}" for n in sorted(b9["eval_by_n_exact"], key=int)) + ".")
    a(f"- Inner question games {e['term']['games_in_questions']}, own head row left out in "
      f"{e['term']['own_head_row_left_out_games']}; mean max q {e['term']['mean_max_q_record']:.4f}.")
    a("")
    if ms:
        a("## Merge simulation (step 4; E n H, threshold 0.65, n_j / 12, beta 1.0)")
        a("")
        for mode, x in ms["modes"].items():
            a(f"- {mode}: base S25 {x['base_S25_rb']:.5f}, merged - base mean {x['merged_minus_base_mean']:+.5f} "
              f"(min {x['merged_minus_base_min_max'][0]:+.5f}, max {x['merged_minus_base_min_max'][1]:+.5f}); "
              f"questions changed per pool {x['questions_changed_mean']:.2f}.")
        a("")
        a(f"**merge simulation: {'PASS' if ms['passed'] else 'NOT passed'}** ({ms['rule']}).")
        a("")
    return "\n".join(A)


def cmd_results(a):
    for p in (K.EVAL, K.KERNEL):
        if not committed_clean(p):
            raise SystemExit(f"{rel(p)} is not committed and clean")
    ms = None
    if os.path.exists(K.MERGE_SIM):
        if not committed_clean(K.MERGE_SIM):
            raise SystemExit("merge_sim.json is not committed and clean")
        ms = K58.read_json(K.MERGE_SIM)
    text = results_text(K58.read_json(K.EVAL), K58.read_json(K.KERNEL), ms)
    with open(K58.guard(K.RESULTS), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    log(f"-> {K.RESULTS}")
    return 0


# ================================================================ merge simulation (Gemini 03 lines 591-878, PLAN setting)
def cmd_mergesim(a):
    import gc
    import hashlib
    import p64_sim as SIM
    st = K58.limits(log)
    t0 = time.time()
    if os.path.exists(K.MERGE_SIM):
        raise SystemExit("merge_sim.json exists")
    cs = code_state()
    if cs["prereg64_dirty"]:
        raise SystemExit(f"results/prereg64 not committed and clean: {cs['prereg64_dirty']}")
    rec = SIM.run(log)
    rep = {"format": "c10-prereg64 merge simulation v1", "created": now(), "code": cs,
           "seconds": round(time.time() - t0, 1), **rec, "peak_working_set_gb": peak_gb(st)}
    K58.write_json(K.MERGE_SIM, rep)
    log(json.dumps({k: rep[k] for k in ("modes", "passed", "seconds", "peak_working_set_gb")}))
    del gc, hashlib
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("kernel", "seam", "eval", "results", "mergesim"):
        sub.add_parser(c)
    a = ap.parse_args(argv)
    return {"kernel": cmd_kernel, "seam": cmd_seam, "eval": cmd_eval, "results": cmd_results,
            "mergesim": cmd_mergesim}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
