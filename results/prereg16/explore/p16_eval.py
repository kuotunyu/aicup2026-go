"""prereg16 exploration (competition 2): consistency / tail features, inner-only judgement of PLAN.md (one candidate
Q09+CT), judged once. CPU only.

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg16/explore
    $S/p16_eval.py eval            -> $S/eval_report.json (judged once; the code must be committed and clean)
    $S/p16_eval.py eval --smoke    -> data/prereg16/smoke/eval_report.json (the 73 features replaced by
                                      default_rng(12345) normal noise before the model; pipeline and timing only,
                                      never a result)

Inputs (read only): HS per-move chunks data/prereg07/pilot/{bank,mono} (SHA-checked against
results/prereg07/hs/inner_manifest.json), P1 per-move chunks data/prereg08/p1/{bank,mono} (SHA-checked against
results/prereg08/feat/inner_manifest.json), data/prereg12/q09_inner/ (ft_eval.InnerPrep), Q09's frozen p1e_map.npz,
u1_lib's inner pseudo-questions through tools/p09_rank_fuse.Inner. Never data/local (u1_lib / hs_lib guards plus
guard_path here), never validation features, never tools/score.py.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
FT_DIR = os.path.join(REPO, "results", "prereg12", "explore", "finetune")
for p in (TOOLS, FT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p09_rank_fuse as r  # noqa: E402  (pins the p08 tools before importing them)
import p09_forward as FW  # noqa: E402
import ft_eval  # noqa: E402  (prereg12: InnerPrep, guard_path)
import p16_lib as K  # noqa: E402

R, rf, p3, p4 = r.R, r.rf, r.p3, r.p4
TOL = 1e-12
GATE = 0.010
HS_SUM_TOL = 1e-6
SMOKE_SEED = 12345
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_TUNE = os.path.join(REPO, "results", "prereg09", "rank", "tune_report.json")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
HS_MANIFEST = os.path.join(REPO, "results", "prereg07", "hs", "inner_manifest.json")
P1_MANIFEST = os.path.join(REPO, "results", "prereg08", "feat", "inner_manifest.json")
PREP_DIR = os.path.join(REPO, "data", "prereg12", "q09_inner")
DATA = os.path.join(REPO, "data", "prereg16")
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "eval_started.json")
REPORT = os.path.join(HERE, "eval_report.json")
OWN_FILES = ("PLAN.md", "p16_lib.py", "p16_eval.py", "test_p16.py")
IMPORTED = ("tools/p02_rank_fuse.py", "tools/p03_rank_fuse.py", "tools/p04_rank_fuse.py", "tools/p07_rank_fuse.py",
            "tools/p08_rank_fuse.py", "tools/p09_rank_fuse.py", "tools/p09_forward.py",
            "results/prereg12/explore/finetune/ft_eval.py", "results/prereg05/explore/u1/u1_lib.py",
            "results/prereg07/explore/pilot_humansl/pilot_fuse.py", "results/prereg07/explore/pilot_humansl/hs_lib.py",
            "results/prereg08/p1/p1_fuse.py", "results/prereg08/p1/p1_lib.py",
            "results/prereg07/hs/inner_manifest.json", "results/prereg08/feat/inner_manifest.json")
FORMAT = "c10-prereg16 explore eval v1"
guard_path = ft_eval.guard_path


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def code_state():
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN_FILES] + list(IMPORTED)
    dirty = git("status", "--porcelain", "--", *rel)
    return {"head": git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def once_guard(a, cs, out_path):
    """Judged once: a report or a started marker (an aborted run) needs --rerun --reason and a newer HEAD."""
    prior = [p for p in (out_path, STARTED) if os.path.exists(p)]
    if not prior:
        return None
    if not (a.rerun and a.reason):
        raise SystemExit(f"{prior} exist: judged once (--rerun --reason TEXT only after a commit that names a "
                         f"concrete program error)")
    heads = set()
    for p in prior:
        with open(p, encoding="utf-8") as f:
            heads.add(((json.load(f).get("code") or {}).get("head")))
    if cs["head"] in heads:
        raise SystemExit("--rerun needs a newer commit than the earlier run's (the fix of the named error)")
    return sorted(h for h in heads if h)


def tune_ref():
    with open(Q09_TUNE, encoding="utf-8") as f:
        c = json.load(f)["combinations"]["Q09"]
    return {"Q09": c["oof"]["rank_balanced"], "Q09_mono": c["single_colour"]["rank_balanced"]}


def set_terms(inner, Emap, terms):
    """m (pooled C5) and E terms of the fit / eval / mono sets exactly as p08_rank_fuse.run_terms builds them."""
    d = inner.d
    di, lp = d["inner"], d["lp"]
    out = {}
    for which, X, S in (("fit", di["X_fit"], di["S_fit"]), ("eval", di["X_eval"], di["S_eval"]),
                        ("mono", di["X_mono"], di["S_mono"])):
        m = rf.pooled(S, lp["C5"], "log")
        E = [Emap[t][which] if t in Emap else (X.mop if t == "OP" else rf.pooled(S, lp[t], "log")) for t in terms]
        out[which] = (m, E)
    return out


def scored(inner, P, Pm, folds, terms):
    """Inner.run's scoring of given eval / single-colour probabilities."""
    if np.isnan(Pm).any() or np.isnan(P).any():
        raise SystemExit("a question without a probability")
    m, sc = rf.metrics(rf.rule_b_np(P), inner.Xe.y, inner.Xe.n, inner.Xe.player)
    mrec, msc = p3.mono_score_record(rf.rule_b_np(Pm), inner.Xm, inner.ms, inner.mboot)
    for fr in folds:
        fr["params"] = p4.gen_params(fr["theta"], terms)
    return {"P": P, "Pm": Pm, "folds": folds, "terms": terms, "m": m, "sc": sc, "mrec": mrec, "msc": msc,
            "rb": m["rank_balanced"], "mono_rb": mrec["rank_balanced"]}


# ---------------------------------------------------------------- per-move inputs (PLAN.md 1, 2; seam 3)
def manifest_chunks(path):
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    return {s: [(x["file"], x["sha256"]) for x in man["sets"][s]["chunks"]] for s in ("bank", "mono")}


def load_moves(usable, log):
    """-> (h_moves, p_moves, seam3 record) for the usable inner rows."""
    n = len(usable)
    h_moves, p_moves = {}, {}
    rec = {"hs_chunks": 0, "p1_chunks": 0, "hs_sum_max_abs_diff": 0.0, "problems": []}
    seen_h = np.zeros(n, dtype=bool)
    seen_p = np.zeros(n, dtype=bool)
    t0 = time.time()
    for s, rows in manifest_chunks(HS_MANIFEST).items():
        for fn, sha in rows:
            p = guard_path(os.path.join(REPO, *fn.split("/")))
            if r.sha_raw(p) != sha:
                raise SystemExit(f"{fn}: SHA-256 differs from the HS manifest")
            with np.load(p, allow_pickle=False) as z:
                ref = np.asarray(z["ref"]).astype(np.int64)
                cnt = np.asarray(z["counts"])[:, 0]
                pg = np.asarray(z["pos_game"]).astype(np.int64)
                pt = np.asarray(z["pos_turn"]).astype(np.int64)
                lp = np.asarray(z["pos_logp"])
                sums = np.asarray(z["sums"])[:, 0, :]
                hside = np.asarray(z["pos_side"])
            if len(hside) and np.any(hside != 0):
                rec["problems"].append(f"{fn}: HS moves of a non-target side")
            if seen_h[ref].any() or len(np.unique(ref)) != len(ref):
                rec["problems"].append(f"{fn}: repeated refs")
            seen_h[ref] = True
            if len(pg) and not np.all(np.diff(pg) >= 0):
                rec["problems"].append(f"{fn}: pos_game not sorted")
            bc = np.bincount(pg, minlength=len(ref))
            if not np.array_equal(bc[:len(ref)], cnt) or len(bc) != len(ref):
                rec["problems"].append(f"{fn}: HS move counts differ from counts[:,0]")
            starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            for j, row in enumerate(ref):
                if cnt[j] == 0:
                    continue
                seg = slice(starts[j], starts[j] + cnt[j])
                dd = float(np.max(np.abs(lp[seg].astype(np.float64).sum(axis=0) - sums[j])))
                rec["hs_sum_max_abs_diff"] = max(rec["hs_sum_max_abs_diff"], dd)
                if usable[row]:
                    h_moves[row] = (lp[seg], pt[seg])
            rec["hs_chunks"] += 1
    log(f"HS per-move chunks read: {rec['hs_chunks']} ({time.time() - t0:.0f}s)")
    for s, rows in manifest_chunks(P1_MANIFEST).items():
        for fn, sha in rows:
            p = guard_path(os.path.join(REPO, *fn.split("/")))
            if r.sha_raw(p) != sha:
                raise SystemExit(f"{fn}: SHA-256 differs from the P1 manifest")
            with np.load(p, allow_pickle=False) as z:
                ref = np.asarray(z["ref"]).astype(np.int64)
                cnt = np.asarray(z["counts"])[:, 0]
                pg = np.asarray(z["pos_game"]).astype(np.int64)
                ps = np.asarray(z["pos_side"])
                pt = np.asarray(z["pos_turn"]).astype(np.int64)
                lp = np.asarray(z["pos_logp"])
                ent = np.asarray(z["pos_entropy"])
            if seen_p[ref].any() or len(np.unique(ref)) != len(ref):
                rec["problems"].append(f"{fn}: repeated refs")
            seen_p[ref] = True
            tgt = ps == 0
            bc = np.bincount(pg[tgt], minlength=len(ref))
            if not np.array_equal(bc[:len(ref)], cnt) or len(bc) != len(ref):
                rec["problems"].append(f"{fn}: P1 target move counts differ from counts[:,0]")
            order = np.argsort(pg[tgt], kind="stable")
            g_t, t_t, lp_t, e_t = pg[tgt][order], pt[tgt][order], lp[tgt][order], ent[tgt][order]
            starts = np.searchsorted(g_t, np.arange(len(ref)))
            ends = np.searchsorted(g_t, np.arange(len(ref)), side="right")
            for j, row in enumerate(ref):
                if usable[row]:
                    sl = slice(starts[j], ends[j])
                    p_moves[row] = (lp_t[sl], e_t[sl], t_t[sl])
            rec["p1_chunks"] += 1
    log(f"P1 per-move chunks read: {rec['p1_chunks']} ({time.time() - t0:.0f}s)")
    rec["hs_rows_seen"] = int(seen_h.sum())
    rec["p1_rows_seen"] = int(seen_p.sum())
    miss = [int(x) for x in np.nonzero(usable)[0] if x not in h_moves or x not in p_moves]
    rec["usable_without_moves"] = len(miss)
    rec["hs_vs_p1_target_moves_differ"] = int(sum(1 for x in h_moves if x in p_moves
                                                  and len(h_moves[x][0]) != len(p_moves[x][0])))
    rec["ok"] = (not rec["problems"]) and rec["hs_sum_max_abs_diff"] <= HS_SUM_TOL and not miss
    return h_moves, p_moves, rec


def build_features(inner, lp_q09, usable, log):
    t0 = time.time()
    h_moves, p_moves, seam3 = load_moves(usable, log)
    if not seam3["ok"]:
        return None, seam3, None
    lp_hs = np.asarray(inner.d["lp"]["HS"])
    hs_zero = np.all(lp_hs[usable] == 0.0, axis=1)
    seam3["usable_without_hs_baseline_lp"] = int(hs_zero.sum())
    if hs_zero.any():
        seam3["ok"] = False
        return None, seam3, None
    T = K.GameTable(inner.n_inner)
    T.e_hs = K.expected_level(lp_hs)
    T.e_p1e = K.expected_level(inner.lp_full(lp_q09))
    rows = np.nonzero(usable)[0]
    K.per_game_from_moves(h_moves, p_moves, rows, T)
    log(f"per-game statistics: {len(rows)} games ({time.time() - t0:.0f}s)")
    del h_moves, p_moves
    di = inner.d["inner"]
    F, info = {}, {}
    for which, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"])):
        F[which], nu = K.question_features(K.groups_of(S), usable, T)
        info[which] = {"questions": int(len(nu)), "without_usable_game": int((nu == 0).sum()),
                       "usable_games_min": int(nu.min()), "usable_games_max": int(nu.max())}
    log(f"question features built ({time.time() - t0:.0f}s): {info}")
    info["seconds"] = round(time.time() - t0, 1)
    return F, seam3, info


def players_of(X, fallback=None):
    p = getattr(X, "player", None)
    return np.asarray(p if p is not None else fallback, dtype=np.int64)


def run_candidate(inner, T9, F, interactions, log):
    di = inner.d["inner"]
    Xf, Xe, Xm = di["X_fit"], di["X_eval"], di["X_mono"]
    lw = np.asarray(Xf.y)
    if not np.array_equal(K.level_weights(lw), rf.level_weights(lw)):
        raise SystemExit("p16_lib.level_weights differs from p02_rank_fuse.level_weights")
    ct = K.nested_crossfit(F["fit"], Xf.y, Xf.fold, players_of(Xf), F["eval"], Xe.fold, players_of(Xe),
                           F["mono"], Xm.fold, players_of(Xm, di["ms"]["player"]), interactions=interactions, log=log)
    (mf, Ef), (me, Ee), (mm, Em) = T9["fit"], T9["eval"], T9["mono"]
    P, Pm, folds = K.fold_loop(p4, rf, Xf, Xe, Xm, mf, me, mm, Ef, Ee, Em, ct=ct)
    terms = list(r.Q08_TERMS) + ["CT"]
    return scored(inner, P, Pm, folds, terms), ct


def gates(cmp_):
    by = cmp_["mono_by_colour"]
    g1 = cmp_["inner_diff"] >= GATE
    g2 = cmp_["inner_ci95"][0] > 0
    g4 = (cmp_["mono_ci95"][0] is not None and cmp_["mono_ci95"][0] > 0
          and all(by.get(c) is not None and by[c] >= -TOL for c in ("B", "W")))
    return {"G1": {"threshold": GATE, "delta": cmp_["inner_diff"], "passed": bool(g1)},
            "G2": {"ci95": cmp_["inner_ci95"], "passed": bool(g2)},
            "G4": {"mono_diff": cmp_["mono_diff"], "mono_ci95": cmp_["mono_ci95"], "by_colour": by,
                   "passed": bool(g4)},
            "passed": bool(g1 and g2 and g4)}


def seam4_record(inner):
    """Player overlap between M training rows and predicted rows, per outer fold (all must be 0; nested_crossfit
    also asserts it)."""
    di = inner.d["inner"]
    Xf, Xe, Xm = di["X_fit"], di["X_eval"], di["X_mono"]
    pf, pe, pm = players_of(Xf), players_of(Xe), players_of(Xm, di["ms"]["player"])
    out = []
    for f in range(K.N_FOLDS):
        tr = set(pf[Xf.fold != f])
        nested = [len(set(pf[(Xf.fold != f) & (Xf.fold != g)]) & set(pf[(Xf.fold == g) | (Xf.fold == f)]))
                  for g in range(K.N_FOLDS) if g != f]
        out.append({"fold": f, "M_f_vs_eval_fold_f": len(tr & set(pe[Xe.fold == f])),
                    "M_f_vs_mono_fold_f": len(tr & set(pm[Xm.fold == f])), "M_fg_vs_folds_f_g": nested})
    ok = all(x["M_f_vs_eval_fold_f"] == 0 and x["M_f_vs_mono_fold_f"] == 0 and not any(x["M_fg_vs_folds_f_g"])
             for x in out)
    return {"ok": bool(ok), "folds": out}


def ct_record(ct):
    return [{k: v for k, v in fr.items()} for fr in ct["folds"]]


def w_ct_record(res):
    out = []
    for fr in res["folds"]:
        w = fr["params"]["w"]["CT"]
        out.append({"fold": fr["fold"], "w_CT": w, "at_bound": bool(w <= 1e-8 or w >= 4.0 - 1e-8)})
    return out


def cmd_eval(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(SMOKE_DIR, "eval_report.json") if smoke else REPORT
    cs = code_state()
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg16 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path)
        write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                             "rerun_reason": a.reason if a.rerun else None})
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    prep = ft_eval.InnerPrep(guard_path(os.path.abspath(a.prep_dir)))
    map_sha = r.sha_raw(Q09_MAP)
    with open(Q09_FROZEN, encoding="utf-8") as f:
        if map_sha != json.load(f)["map"]["sha256"]:
            raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    inner = r.Inner(prep, log=log)
    ref = tune_ref()
    lp_q09 = r.inner_map_lp(prep, FW.load_map(Q09_MAP))
    E_q09, _ = inner.E_p1(lp_q09)
    terms9 = list(r.Q08_TERMS)
    q09 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_q09})
    # ---- seam 1
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]]}
    seam1 = all(abs(x - y) <= TOL for x, y in seam.values())
    # ---- seam 2: own fold loop without CT = Q09 bit for bit
    T9 = set_terms(inner, {"HS": inner.E_hs, "P1TO": E_q09}, terms9)
    di = inner.d["inner"]
    P0, Pm0, f0 = K.fold_loop(p4, rf, di["X_fit"], di["X_eval"], di["X_mono"], T9["fit"][0], T9["eval"][0],
                              T9["mono"][0], T9["fit"][1], T9["eval"][1], T9["mono"][1], ct=None)
    seam2 = {"P_equal": bool(np.array_equal(P0, q09["P"])), "Pm_equal": bool(np.array_equal(Pm0, q09["Pm"])),
             "theta_equal": all(a_["theta"] == b_["theta"] for a_, b_ in zip(f0, q09["folds"]))}
    log(f"seam 1 {'ok' if seam1 else 'FAILED'}: {seam}; seam 2: {seam2}")
    if not (seam1 and all(seam2.values())):
        write_json(out_path, {"format": FORMAT, "status": "stop: seam failed", "seam": seam, "seam2": seam2,
                              "smoke": smoke, "code": cs})
        return 1
    # ---- features (seam 3 inside)
    hs_usable, _f, _c = R.pilot_hs_usable(inner.PF, inner.n_inner)
    usable = inner.covered & hs_usable
    use_rec = {"p1_usable": int(inner.covered.sum()), "hs_usable": int(hs_usable.sum()), "both": int(usable.sum()),
               "inner_rows": int(inner.n_inner)}
    F, seam3, finfo = build_features(inner, lp_q09, usable, log)
    if F is None:
        write_json(out_path, {"format": FORMAT, "status": "stop: per-move seam failed", "seam": seam,
                              "seam2": seam2, "seam3": seam3, "usable": use_rec, "smoke": smoke, "code": cs})
        return 1
    if smoke:
        rng = np.random.default_rng(SMOKE_SEED)
        F = {k: rng.normal(size=v.shape) for k, v in F.items()}
    # ---- candidate (seam 4 = the assertions inside nested_crossfit)
    seam4 = seam4_record(inner)
    if not seam4["ok"]:
        write_json(out_path, {"format": FORMAT, "status": "stop: seam 4 failed", "seam4": seam4, "code": cs})
        return 1
    tc = time.time()
    cand, ct = run_candidate(inner, T9, F, True, log)
    cmp_ = inner.compare(cand, q09)
    g = gates(cmp_)
    log(f"Q09+CT {cand['rb']!r} vs Q09 {q09['rb']!r}: {json.dumps(g)} ({time.time() - tc:.0f}s)")
    # ---- record only
    m_alone = scored(inner, np.exp(ct["ct_eval"]), np.exp(ct["ct_mono"]), [], ["CT"])
    noint, ct_noint = run_candidate(inner, T9, {k: v[:, :len(K.BASE_NAMES)] for k, v in F.items()}, False, log)
    rec_only = {"M_alone_rule_b": {"inner_rank_balanced": m_alone["rb"], "single_colour_rank_balanced":
                                   m_alone["mono_rb"]},
                "Q09+CT_without_interactions_62": {**inner.short(noint, q09), "M_folds": ct_record(ct_noint)},
                "w_CT_by_fold": w_ct_record(cand)}
    status = "gates passed (worth a PREREG_16; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {"format": FORMAT, "created": r.now_iso(), "seconds": round(time.time() - t0, 1), "smoke": smoke,
           "smoke_note": ("the 62 base features replaced by default_rng(12345) normal noise (the 11 interactions are built from "
                           "that noise): numbers are meaningless"
                          if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: inner-only judgement of Q09+CT (consistency / tail features -> multinomial LR M, nested "
                   "player-fold cross-fitting, fused as one extra term); training players only, no validation "
                   "scoring (competition 2 has none left)",
           "status": status, "seam": seam, "seam2": seam2, "seam3": seam3, "seam4": seam4, "usable": use_rec, "features": finfo,
           "feature_names": K.ALL_NAMES,
           "candidate": {"name": "Q09+CT", "terms": cand["terms"], **inner.short(cand, q09),
                         "M_folds": ct_record(ct)},
           "baseline": {"name": "Q09", **inner.short(q09)}, "gates": g, "record_only": rec_only,
           "q09_map_sha256": map_sha, "code": cs}
    write_json(out_path, rep)
    log(f"-> {out_path}: {status}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--prep-dir", default=PREP_DIR)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
