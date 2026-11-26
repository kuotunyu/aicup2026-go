"""prereg29 exploration (competition 2): inner-only judgement of PLAN.md (one candidate Q29 = Q25 with the SET term's
three S25 models replaced by the ten S29 models: S25's 3 frozen seeds + 7 new seeds), judged once. CPU only (.venv;
numpy forward, no torch).

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg29/explore
    $S/p29_eval.py eval            -> $S/eval_report.json (judged once; code and train_record.json committed and clean)
    $S/p29_eval.py eval --smoke    -> data/prereg29/smoke/eval_report.json (the 7 new seeds noise-trained: pipeline and
                                      timing only, never a result)

The Q09 / Q19 / Q25 rebuild, the SET term (per-game h -> mean over the question's P1-usable games -> head, lse mean over
the models, x 4), the scores and the record-only helpers are imported unchanged from
results/prereg23/explore/p23_eval.py (the code that reproduced Q19 / Q25 bit-exactly in prereg23 / 25 / 26); the fusion
is tools/p09_rank_fuse.Inner (unchanged). Never the answers folder, never validation features, never tools/score.py.
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
P23 = os.path.join(REPO, "results", "prereg23", "explore")
for p in (os.path.join(REPO, "tools"), P23, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p23_eval as E23  # noqa: E402  (Q19 rebuild, SET term, scores; puts prereg19 / prereg12 folders on sys.path)
import p23_lib as K23  # noqa: E402  (gates with the +0.005 threshold)
import p19_lib as K19  # noqa: E402
import p29_lib as L  # noqa: E402

r, ft_eval, FW = E23.r, E23.ft_eval, E23.FW
guard_path = E23.guard_path
TOL = 1e-12
STARTED = os.path.join(L.DATA, "eval_started.json")
SEAM_FAIL = os.path.join(L.DATA, "eval_seam_fail.json")
JUDGED = os.path.join(L.DATA, "eval_judged.json")          # the gates, written as soon as they exist     # a seam stop is written here (no candidate number exists)
REPORT = os.path.join(HERE, "eval_report.json")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p29_lib.py", "p29_train.py", "p29_eval.py", "test_p29.py", "train_record.json",
             "run_train.sh", "run_eval.sh")
IMPORTED = tuple(E23.IMPORTED) + ("results/prereg23/explore/p23_eval.py", "results/prereg23/explore/p23_lib.py",
                                  "results/prereg19/explore/p19_train.py", "results/prereg25/explore/p25_lib.py",
                                  "results/prereg25/explore/p25_train.py", "results/prereg25/explore/eval_report.json",
                                  "results/prereg25/explore/train_record.json", "results/prereg25/explore/audit.json",
                                  "results/prereg25/rank/s25_seed0.npz", "results/prereg25/rank/s25_seed1.npz",
                                  "results/prereg25/rank/s25_seed2.npz")
FORWARD_FILES = ("results/prereg19/explore/p19_lib.py", "results/prereg19/explore/p19_train.py",
                 "results/prereg25/explore/p25_train.py", "results/prereg25/explore/p25_lib.py", "tools/p09_forward.py",
                 "results/prereg29/explore/p29_train.py", "results/prereg29/explore/p29_lib.py")
FORMAT = "c10-prereg29 explore eval v1"


def log(*x):
    print(*x, flush=True)


def code_state(smoke):
    own = [f for f in OWN_FILES if not (smoke and f == "train_record.json")]
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in own] + list(IMPORTED)
    rel = sorted(set(rel) | L.loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = E23.git("status", "--porcelain", "--", *rel)
    return {"head": E23.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: E23.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def once_guard(a, cs, out_path):
    prior = [p for p in (out_path, STARTED) if os.path.exists(p)]
    if not prior:
        return None
    if not (a.rerun and a.reason):
        raise SystemExit(f"{prior} exist: judged once (--rerun --reason TEXT only after a commit that names a "
                         f"concrete program error)")
    heads = {(r.read_json(p).get("code") or {}).get("head") for p in prior}
    if cs["head"] in heads:
        raise SystemExit("--rerun needs a newer commit than the earlier run's (the fix of the named error)")
    return sorted(h for h in heads if h)


def seam_m(tr, smoke, cs):
    """The 7 new models = the train record's; settings registered; code clean; forward code unchanged; port checks;
    seam R (re-hashed here)."""
    tr25 = L.read_json(L.P25_TRAIN)
    want = L.registered_settings(K19)
    if smoke:
        want.update(steps=K19.SMOKE["steps"], eval_every=K19.SMOKE["eval_every"], hold_per=K19.SMOKE["hold_per"])
    s = tr.get("settings") or {}
    out = {"models": {},
           "train_settings_registered": bool(all(s.get(k) == v for k, v in want.items())
                                             and s.get("new_seeds") == list(L.NEW_SEEDS)
                                             and s.get("s25_seeds_reused") == list(L.S25_SEEDS)
                                             and s.get("seam_r_seed") == L.SEAM_R_SEED
                                             and s.get("model_format") == L.S25_FORMAT),
           "prereg25_settings_same": bool(all(tr25["settings"].get(k) == v
                                              for k, v in L.registered_settings(K19).items())),
           "train_code_clean": bool((tr.get("code") or {}).get("dirty") == "")}
    tr_sha = (tr.get("code") or {}).get("sha256_lf") or {}
    out["forward_code_same_as_train"] = {f: bool(tr_sha.get(f) is not None and tr_sha.get(f) == cs["sha256_lf"].get(f))
                                         for f in FORWARD_FILES}
    sr = tr.get("seam_r") or {}
    rpath = os.path.join(REPO, *sr["retrain_file"].split("/")) if sr.get("retrain_file") else None
    rsha = E23.sha_file(rpath) if rpath and os.path.exists(rpath) else None
    frozen0 = E23.sha_file(os.path.join(L.S25_DIR, "s25_seed0.npz"))
    out["seam_r"] = {"train_record_ok": bool(sr.get("ok")), "retrain_sha256_now": rsha,
                     "retrain_is_train_record": bool(rsha is not None and rsha == sr.get("retrain_sha256")),
                     "retrain_is_s25_seed0": bool(rsha is not None and rsha == frozen0 == tr25["models_sha256"]["0"])}
    out["seam_r"]["ok"] = bool(smoke or (out["seam_r"]["train_record_ok"] and out["seam_r"]["retrain_is_train_record"]
                                         and out["seam_r"]["retrain_is_s25_seed0"]))
    ok = bool(out["train_settings_registered"] and out["prereg25_settings_same"] and (smoke or out["train_code_clean"])
              and all(out["forward_code_same_as_train"].values()) and out["seam_r"]["ok"])
    pars = []
    for sd in L.NEW_SEEDS:
        rec = tr["seeds"][str(sd)]
        mpath = os.path.join(REPO, *rec["model_file"].split("/"))
        sha = E23.sha_file(mpath)
        par, seed, best_step, msmoke = E23.load_npz_model(mpath, L.S25_FORMAT)
        good = bool(sha == tr["models_sha256"][str(sd)] and sha == rec["model_sha256"] and seed == sd
                    and best_step == rec["best_step"] and msmoke == smoke and (rec.get("port_check") or {}).get("ok")
                    and L.finite_params(par))
        out["models"][str(sd)] = {"sha256": sha, "best_step": best_step, "ok": good}
        ok = ok and good
        pars.append(K19.p64(par))
    shas = [m["sha256"] for m in out["models"].values()] + list(tr25["models_sha256"].values())
    out["all_ten_distinct"] = bool(len(set(shas)) == len(shas))
    out["ok"] = bool(ok and out["all_ten_distinct"])
    return pars, out


def seam_p(tr, smoke):
    want = (r.read_json(E23.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {"probe_X.npy": E23.sha_file(os.path.join(E23.PROBE_PREP, "probe_X.npy")),
           "inner_X.npy": E23.sha_file(os.path.join(E23.INNER_PREP, "inner_X.npy"))}
    vrec = r.read_json(os.path.join(L.V_PREP, "prep_record.json"))
    v_sha = E23.sha_file(os.path.join(L.V_PREP, "v_X.npy"))
    m_sha = E23.sha_file(os.path.join(L.V_PREP, "v_meta.npz"))
    out = {"want": {k: want.get(k) for k in got}, "got": got, "v_X.npy": v_sha, "v_meta.npz": m_sha,
           "v_meta_is_prep_record": bool(m_sha == vrec["files_sha256"]["v_meta.npz"]),
           "v_X_is_prep_record": bool(v_sha == vrec["files_sha256"]["v_X.npy"]),
           "v_X_is_train_record": bool(v_sha == (tr.get("seam_p") or {}).get("v_X_sha256"))}
    out["ok"] = bool(all(want.get(k) == v for k, v in got.items()) and out["v_X_is_prep_record"]
                     and out["v_meta_is_prep_record"]
                     and (smoke or out["v_X_is_train_record"]))
    return out


def seam_d(inner, tr, smoke):
    with np.load(guard_path(os.path.join(E23.PROBE_PREP, "probe_meta.npz")), allow_pickle=False) as z:
        pkeys = set(np.asarray(z["players"]).astype(str).tolist())
    with np.load(guard_path(os.path.join(L.V_PREP, "v_meta.npz")), allow_pickle=False) as z:
        vkeys = set(np.asarray(z["players"]).astype(str).tolist())
    allk = pkeys | vkeys
    a_keys = {k for k in allk if L.L25.holdout_bit(k) == 0}
    h_keys = {k for k in allk if L.L25.holdout_bit(k) == 1}
    players = inner.d["inner"]["players"]
    i_keys = {f"{p['rank']}:{p['player_id']}" for p in players}
    i_ids = {p["player_id"] for p in players}
    pid = lambda ks: {k.split(":", 1)[1] for k in ks}  # noqa: E731
    aud = L.read_json(L.P25_AUDIT)
    out = {"A_players": len(a_keys), "H_players": len(h_keys), "I_players": len(i_keys), "V_players": len(vkeys),
           "A_and_H": len(a_keys & h_keys), "train_and_I": len(allk & i_keys),
           "train_and_I_player_id": len(pid(allk) & i_ids),
           "probe_and_V": len(pkeys & vkeys), "probe_and_V_player_id": len(pid(pkeys) & pid(vkeys)),
           "V_games_vs_inner_8_symmetries_audit": aud["duplicates"]["val_vs_inner_8_symmetries"],
           "train_record_players_match": bool(tr["players"]["A_players"] == len(a_keys)
                                              and tr["players"]["H_players"] == len(h_keys))}
    out["ok"] = bool(out["A_and_H"] == 0 and out["train_and_I"] == 0 and out["train_and_I_player_id"] == 0
                     and out["probe_and_V"] == 0 and out["probe_and_V_player_id"] == 0
                     and out["V_games_vs_inner_8_symmetries_audit"] == 0 and out["train_record_players_match"])
    return out


def cmd_eval(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(L.SMOKE_DIR, "eval_report.json") if smoke else REPORT
    train_path = os.path.join(L.SMOKE_DIR, "train_record.json") if smoke else TRAIN_RECORD
    cs = code_state(smoke)
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg29 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path)
    base = {"format": FORMAT, "smoke": smoke, "code": cs}
    tr = r.read_json(train_path)
    if bool(tr.get("smoke")) != smoke:
        raise SystemExit("train_record.json smoke flag differs from this run")
    new_pars, sm = seam_m(tr, smoke, cs)
    s19_pars, sf19 = E23.s19_models()
    s25_pars, sf25 = L.s25_models(K19, E23.load_npz_model)
    map_sha = r.sha_raw(E23.Q09_MAP)
    sf = {"S19": sf19, "S25": sf25, "q09_map_sha256": map_sha,
          "q09_map_is_frozen": bool(map_sha == r.read_json(E23.Q09_FROZEN)["map"]["sha256"])}
    sf["ok"] = bool(sf19["ok"] and sf25["ok"] and sf["q09_map_is_frozen"])
    # ---- Q09, Q19, Q25 and seam 1 (= prereg23 / 25 / 26's rebuild)
    prep = ft_eval.InnerPrep(guard_path(E23.INNER_PREP))
    inner = r.Inner(prep, log=log)
    ref = {**E23.refs(), **L.refs25()}
    mp = FW.load_map(E23.Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    terms9 = list(r.Q08_TERMS)
    q09 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_q09})
    E_s19, _lp, _h = E23.s19_term(prep, inner, mp, s19_pars)
    terms19 = terms9 + ["SET"]
    q19 = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s19})
    del E_s19
    E_s25, lp_s25, has_s25 = E23.s19_term(prep, inner, mp, s25_pars)
    q25 = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25})
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]], "Q19": [q19["rb"], ref["Q19"]],
            "Q19_mono": [q19["mono_rb"], ref["Q19_mono"]], "Q25": [q25["rb"], ref["Q25"]],
            "Q25_mono": [q25["mono_rb"], ref["Q25_mono"]]}
    seam1_tol = all(abs(x - y) <= TOL for x, y in seam.values())
    seam1_bitwise = all(x == y for x, y in seam.values())
    seam1 = bool(seam1_tol and seam1_bitwise)          # PLAN.md 3.1: every number bit-exact
    q25_short = inner.short(q25, q19)
    del q19
    sp = seam_p(tr, smoke)
    sd = seam_d(inner, tr, smoke)
    log(f"seam 1 {seam1} (bitwise {seam1_bitwise}) {seam}; seam P {sp['ok']}; seam D {sd}; seam M {sm['ok']} "
        f"(seam R {sm['seam_r']}); seam F {sf['ok']}")
    if not (seam1 and sp["ok"] and sd["ok"] and sm["ok"] and sf["ok"]):
        E23.write_json(os.path.join(L.SMOKE_DIR, "eval_seam_fail.json") if smoke else SEAM_FAIL,
                       {**base, "status": "stop: seam failed", "seam": seam, "seam1_bitwise": seam1_bitwise,
                        "seam_p": sp, "seam_d": sd, "seam_m": sm, "seam_f": sf})
        return 1
    # ---- the judged part starts here (after every seam, before any candidate number)
    if not smoke:
        E23.write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                 "rerun_reason": a.reason if a.rerun else None})
    tc = time.time()
    s29_pars = list(s25_pars) + list(new_pars)          # ensemble order = seeds 0..9 (L.ALL_SEEDS)
    E_s29, lp_s29, has_s29 = E23.s19_term(prep, inner, mp, s29_pars)
    cand = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s29})
    cmp_ = inner.compare(cand, q25)
    g = K23.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                  cmp_["mono_by_colour"])
    log(f"Q29 {cand['rb']!r} vs Q25 {q25['rb']!r}: {json.dumps(g)} ({time.time() - tc:.0f}s)")
    E23.write_json(os.path.join(L.SMOKE_DIR, "eval_judged.json") if smoke else JUDGED,
                   {**base, "created": r.now_iso(), "gates": g, "Q29": [cand["rb"], cand["mono_rb"]],
                    "Q25": [q25["rb"], q25["mono_rb"]], "note": "written before the record-only part"})
    # ---- record only
    chk = {"eval_rb_own": K19.rank_balanced(cand["sc"], inner.Xe.y),
           "mono_rb_own": K19.rank_balanced(cand["msc"], inner.ms["y"])}
    chk["ok"] = bool(abs(chk["eval_rb_own"] - cand["rb"]) <= TOL and abs(chk["mono_rb_own"] - cand["mono_rb"]) <= TOL)
    counts = E23.by_count(inner, cand, q25)
    levels = E23.by_level(inner, cand, q25)
    curve = {}
    for k in L.SUBSETS:
        if k == len(L.ALL_SEEDS):
            res = cand
        else:
            E_k, _lp, _h = E23.s19_term(prep, inner, mp, s29_pars[:k])
            res = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_k})
            del E_k
        curve[str(k)] = {"seeds": list(L.subset_seeds(k)), "inner_rank_balanced": res["rb"],
                         "single_colour_rank_balanced": res["mono_rb"], "vs_q25": inner.compare(res, q25),
                         "weights_by_fold": E23.w_record(res, ["P1TO", "SET"])}
        if k == 3:
            curve["3"]["bit_exact_q25"] = bool(res["rb"] == q25["rb"] and res["mono_rb"] == q25["mono_rb"])
        if res is not cand:
            del res
    per_seed = {}
    for i, sd_ in enumerate(L.NEW_SEEDS):
        E_i, _lp, _h = E23.s19_term(prep, inner, mp, [new_pars[i]])
        rs = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_i})
        per_seed[str(sd_)] = {"inner_rank_balanced": rs["rb"], "single_colour_rank_balanced": rs["mono_rb"],
                              "vs_q25": inner.compare(rs, q25)}
        del rs, E_i
    rec_only = {"seed_count_curve": curve, "per_new_seed_Q_s": per_seed,
                "S29_alone_rule_b": E23.alone(inner, lp_s29, has_s29),
                "S25_alone_rule_b": E23.alone(inner, lp_s25, has_s25),
                "by_game_count": counts, "by_level": levels,
                "weights_by_fold": E23.w_record(cand, ["P1TO", "SET"]),
                "q25_weights_by_fold": E23.w_record(q25, ["P1TO", "SET"]),
                "own_rank_balanced_check": chk, "seam1_bitwise": seam1_bitwise}
    status = "gates passed (worth a PREREG_29; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {**base, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("smoke: the 7 new seeds are noise-trained 300-step models; numbers are meaningless"
                          if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: inner-only judgement of Q29 (Q25 with the SET term's models S25 replaced by S29 = S25's "
                   "3 frozen seeds + 7 new seeds of the same code, lse mean of 10; Q09's player-fold OOF fusion); "
                   "no validation scoring",
           "status": status, "seam": seam, "seam_p": sp, "seam_d": sd, "seam_m": sm, "seam_f": sf,
           "train_record_sha256_lf": E23.sha_lf(train_path),
           "candidate": {"name": "Q29", "terms": terms19, "seeds": list(L.ALL_SEEDS), **inner.short(cand, q25)},
           "baseline": {"name": "Q25", "terms": terms19, "seeds": list(L.S25_SEEDS), **q25_short}, "gates": g,
           "record_only": rec_only, "q09_map_sha256": map_sha, "peak_memory_record": E23.peak_memory()}
    E23.write_json(out_path, rep)
    log(f"-> {out_path}: {status} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
