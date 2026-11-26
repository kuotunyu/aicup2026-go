"""prereg40 exploration (competition 2): stage 0b (held-out NLL, PLAN.md 4) and the inner-only judgement (PLAN.md 5)
of Q40 = Q25 with the SET term replaced by S40 (prereg23's SET2 trained on prereg25's rows). CPU only (.venv; numpy
forward, no torch).

    Git Bash, repository root on a subst drive; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP /
    OPENBLAS / MKL variable
    S=results/prereg40/explore
    $S/p40_eval.py stage0b          -> $S/stage0b.json (once; train_record.json committed, code clean; no inner data)
    $S/p40_eval.py eval             -> $S/eval_report.json (judged once; stage0b.json committed and passed)
    $S/p40_eval.py stage0b|eval --smoke -> data/prereg40/smoke/ (smoke models, noise inputs: pipeline and timing only)

Imported unchanged: p23_eval (Q19 rebuild, SET term of S19/S25 = s19_term, SET2 term = inner_h + set2_terms, scores,
records), p23_lib.gates (threshold 0.005), tools/p09_rank_fuse.Inner (fusion). Never the validation answers, never
validation features, never tools/score.py.
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

import p23_eval as E23  # noqa: E402
import p23_lib as K23  # noqa: E402
import p23_train as T23  # noqa: E402  (ProbeOnly, probe_colours: the same probe colours as training)
import p19_lib as K19  # noqa: E402
import p40_lib as L  # noqa: E402

r, ft_eval, FW = E23.r, E23.ft_eval, E23.FW
guard_path = E23.guard_path
TOL = 1e-12
DATA = L.DATA
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "eval_started.json")
STARTED_0B = os.path.join(DATA, "stage0b_started.json")
REPORT = os.path.join(HERE, "eval_report.json")
STAGE0B = os.path.join(HERE, "stage0b.json")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p40_lib.py", "p40_train.py", "p40_eval.py", "test_p40.py", "train_record.json")
IMPORTED = tuple(E23.IMPORTED) + ("results/prereg23/explore/p23_eval.py", "results/prereg23/explore/p23_lib.py",
                                  "results/prereg23/explore/p23_train.py", "results/prereg25/explore/p25_lib.py",
                                  "results/prereg25/explore/train_record.json",
                                  "results/prereg25/explore/eval_report.json", "results/prereg25/explore/audit.json",
                                  "results/prereg26/explore/v_ex_record.json")
FORWARD_FILES = ("results/prereg23/explore/p23_lib.py", "results/prereg23/explore/p23_train.py",
                 "results/prereg19/explore/p19_lib.py", "tools/p09_forward.py")
FORMAT = "c10-prereg40 explore eval v1"
FORMAT_0B = "c10-prereg40 explore stage0b v1"


def log(*x):
    print(*x, flush=True)


def code_state(smoke, own=OWN_FILES):
    own = [f for f in own if not (smoke and f in ("train_record.json", "stage0b.json"))]
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in own] + list(IMPORTED)
    rel = sorted(set(rel) | E23.loaded_repo_modules() | L.loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = E23.git("status", "--porcelain", "--", *rel)
    return {"head": E23.git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: E23.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def once_guard(a, cs, out_path, started):
    prior = [p for p in (out_path, started) if os.path.exists(p)]
    if not prior:
        return None
    if not (a.rerun and a.reason):
        raise SystemExit(f"{prior} exist: run once (--rerun --reason TEXT only after a commit that names a concrete "
                         f"program error)")
    heads = {(r.read_json(p).get("code") or {}).get("head") for p in prior}
    if cs["head"] in heads:
        raise SystemExit("--rerun needs a newer commit than the earlier run's (the fix of the named error)")
    return sorted(h for h in heads if h)


def seam_m(tr, smoke, cs):
    """The five S40 model files of train_record.json; registered SET2 settings; forward code as in training."""
    want = {"steps": K23.STEPS, "eval_every": K23.EVAL_EVERY, "hold_per": L.HOLD_PER, "seeds": list(K23.SEEDS),
            "batch": K23.BATCH, "ks": list(K23.KS), "lr": K23.LR, "wd": K23.WD, "pct_start": K23.PCT_START,
            "d_in": K23.D_IN, "hidden": K23.HIDDEN, "h_dim": K23.H_DIM, "pool_dim": K23.POOL_DIM, "rho": K23.RHO,
            "drop_in": K23.DROP_IN, "drop": K23.DROP, "threads": K23.THREADS, "hold_seed": L.HOLD_SEED,
            "sampler_base": K23.SAMPLER_BASE, "salt": L.SALT}
    if smoke:
        want.update(steps=K23.SMOKE["steps"], eval_every=K23.SMOKE["eval_every"], hold_per=K23.SMOKE["hold_per"])
    out = {"models": {}, "train_settings_registered": bool(all(tr["settings"].get(k) == v for k, v in want.items())),
           "train_code_clean": bool((tr.get("code") or {}).get("dirty") == ""),
           "train_seams": {k: (tr.get(k) or {}).get("ok") for k in ("seam_p", "seam_s25", "seam_d", "seam_h")}}
    tr_sha = (tr.get("code") or {}).get("sha256_lf") or {}
    out["forward_code_same_as_train"] = {f: bool(tr_sha.get(f) is not None and tr_sha.get(f) == cs["sha256_lf"].get(f))
                                         for f in FORWARD_FILES}
    ok = bool(out["train_settings_registered"] and all(out["forward_code_same_as_train"].values())
              and (smoke or (out["train_code_clean"] and all(v is True for v in out["train_seams"].values()))))
    pars = []
    for s in K23.SEEDS:
        sr = tr["seeds"][str(s)]
        mpath = os.path.join(REPO, *sr["model_file"].split("/"))
        sha = E23.sha_file(mpath)
        par, seed, best_step, msmoke = E23.load_npz_model(mpath, L.MODEL_FORMAT)
        good = bool(sha == tr["models_sha256"][str(s)] and seed == s and best_step == sr["best_step"]
                    and msmoke == smoke and (sr.get("port_check") or {}).get("ok"))
        out["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": good}
        ok = ok and good
        pars.append(K23.p64(par))
    out["ok"] = bool(ok)
    return pars, out


def s25_models():
    """The three frozen S25 models (prereg25 train_record.json SHAs) as float64 parameter dicts."""
    tr = r.read_json(L.S25_TRAIN_RECORD)
    pars, seam = [], {"models": {}, "ok": True}
    for s in L.S25_SEEDS:
        sr = tr["seeds"][str(s)]
        path = os.path.join(REPO, *sr["model_file"].split("/"))
        sha = E23.sha_file(path)
        par, seed, best_step, msmoke = E23.load_npz_model(path, L.S25_MODEL_FORMAT)
        ok = bool(sha == tr["models_sha256"][str(s)] and seed == s and not msmoke and best_step == sr["best_step"])
        seam["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": ok}
        seam["ok"] = bool(seam["ok"] and ok)
        pars.append(K19.p64(par))
    return pars, seam


def bound_train_record(smoke):
    path = os.path.join(SMOKE_DIR, "train_record.json") if smoke else TRAIN_RECORD
    tr = r.read_json(path)
    if bool(tr.get("smoke")) != smoke:
        raise SystemExit("train_record.json smoke flag differs from this run")
    return tr, path


# ================================================================ stage 0b (PLAN.md 4)
def cmd_stage0b(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(SMOKE_DIR, "stage0b.json") if smoke else STAGE0B
    cs = code_state(smoke)
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg40 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path, STARTED_0B)
        E23.write_json(STARTED_0B, {"format": FORMAT_0B + " started", "created": r.now_iso(), "code": cs})
    tr, tr_path = bound_train_record(smoke)
    s40, sm = seam_m(tr, smoke, cs)
    s25, s25seam = s25_models()
    if not (sm["ok"] and s25seam["ok"]):
        E23.write_json(out_path, {"format": FORMAT_0B, "smoke": smoke, "code": cs, "status": "stop: seam failed",
                                  "seam_m": sm, "seam_s25": s25seam})
        return 1
    mp = FW.load_map(E23.Q09_MAP)
    prep23 = T23.ProbeOnly(L.PROBE_PREP)
    rows = L.Rows(T23.probe_colours(prep23))
    del prep23
    _ra, _rh, _pa, pool_h = L.split_pools(rows)
    hold_per = K23.SMOKE["hold_per"] if smoke else L.HOLD_PER
    hold, hold_stats = K19.holdout_set(pool_h, hold_per, L.HOLD_SEED)
    seam_hold = {"holdout_questions_equal_train": bool({str(n): int(len(v[1])) for n, v in hold.items()}
                                                       == tr["players"]["holdout_questions"]),
                 "holdout_sampling_equal_train": bool(hold_stats == tr["players"]["holdout_sampling"])}
    seam_hold["ok"] = bool(all(seam_hold.values()))
    if not seam_hold["ok"]:
        E23.write_json(out_path, {"format": FORMAT_0B, "smoke": smoke, "code": cs, "status": "stop: seam failed",
                                  "seam_hold": seam_hold})
        return 1

    def feats(idx):
        if smoke:
            return np.random.default_rng([E23.NOISE_SEED, 7, int(idx[0]), len(idx)]).standard_normal(
                (len(idx), K23.D_IN)).astype(np.float32)
        Xr, EXr = rows.raw(idx)
        return L.features32(Xr, EXr, mp)

    lp40 = L.holdout_lp(K23.set_logp_np, s40, hold, feats)
    lp25 = L.holdout_lp(K19.set_logp_np, s25, hold, lambda idx: feats(idx)[:, :FW.X_DIM])
    n40, per_n40 = L.mean_nll(lp40, hold)
    n25, per_n25 = L.mean_nll(lp25, hold)
    v = L.stage0b_verdict(n40, n25)
    n40_3, per_n40_3 = L.mean_nll(lp40, hold, members=[0, 1, 2])
    rec = {"format": FORMAT_0B, "smoke": smoke, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "code": cs, "earlier_run_heads": earlier,
           "what": "PLAN.md 4: held-out NLL (S25's 4,000 held-out questions) of the S40 ensemble (5 seeds, lse mean) "
                   "vs the S25 ensemble (3 frozen seeds), numpy float64; no inner data",
           "train_record_sha256_lf": E23.sha_lf(tr_path), "seam_m": sm, "seam_s25": s25seam, "seam_hold": seam_hold,
           "nll": {"S40": n40, "S25": n25}, "verdict": v,
           "status": ("passed: run the judgement (PLAN.md 5)" if v["passed"] else "not passed: stop (no inner use)"),
           "record_only": {"per_n": {"S40": per_n40, "S25": per_n25,
                                     "delta": {k: per_n40[k] - per_n25[k] for k in per_n40}},
                           "S40_seeds_0_2": {"nll": n40_3, "delta_vs_S25": n40_3 - n25, "per_n": per_n40_3},
                           "per_seed_S40": {str(s): L.mean_nll(lp40, hold, members=[i])[0]
                                            for i, s in enumerate(K23.SEEDS)},
                           "per_seed_S25": {str(s): L.mean_nll(lp25, hold, members=[i])[0]
                                            for i, s in enumerate(L.S25_SEEDS)}},
           "peak_memory_record": E23.peak_memory()}
    E23.write_json(out_path, rec)
    log(f"stage 0b: S40 {n40:.5f} vs S25 {n25:.5f} -> {v} -> {out_path}")
    return 0


# ================================================================ judgement (PLAN.md 5)
def seam_d(inner, tr):
    with np.load(guard_path(os.path.join(L.PROBE_PREP, "probe_meta.npz")), allow_pickle=False) as z:
        pkeys = set(np.asarray(z["players"]).astype(str).tolist())
    with np.load(guard_path(os.path.join(L.V_PREP, "v_meta.npz")), allow_pickle=False) as z:
        vkeys = set(np.asarray(z["players"]).astype(str).tolist())
    allk = pkeys | vkeys
    a_keys = {k for k in allk if L.holdout_bit(k) == 0}
    h_keys = {k for k in allk if L.holdout_bit(k) == 1}
    players = inner.d["inner"]["players"]
    i_keys = {f"{p['rank']}:{p['player_id']}" for p in players}
    i_ids = {p["player_id"] for p in players}
    aud = r.read_json(L.S25_AUDIT)
    out = {"A_players": len(a_keys), "H_players": len(h_keys), "I_players": len(i_keys), "V_players": len(vkeys),
           "A_and_H": len(a_keys & h_keys), "train_and_I": len(allk & i_keys),
           "train_and_I_player_id": len(L.player_ids(allk) & i_ids), "probe_and_V": len(pkeys & vkeys),
           "probe_and_V_player_id": len(L.player_ids(pkeys) & L.player_ids(vkeys)),
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
    out_path = os.path.join(SMOKE_DIR, "eval_report.json") if smoke else REPORT
    cs = code_state(smoke, OWN_FILES + ("stage0b.json",))
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg40 files or the imported tools are not committed and clean: {cs['dirty']}")
        s0 = r.read_json(STAGE0B)
        if s0.get("smoke") or not (s0.get("verdict") or {}).get("passed"):
            raise SystemExit("stage0b.json is not a passed stage 0b: PLAN.md 4 says stop")
        earlier = once_guard(a, cs, out_path, STARTED)
    base = {"format": FORMAT, "smoke": smoke, "code": cs}
    tr, tr_path = bound_train_record(smoke)
    s40, sm = seam_m(tr, smoke, cs)
    s25, sf = s25_models()
    # ---- Q09, Q19, Q25 and seam 1
    prep = ft_eval.InnerPrep(guard_path(E23.INNER_PREP))
    map_sha = r.sha_raw(E23.Q09_MAP)
    if map_sha != r.read_json(E23.Q09_FROZEN)["map"]["sha256"]:
        raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    inner = r.Inner(prep, log=log)
    ref = E23.refs()
    mp = FW.load_map(E23.Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    terms9 = list(r.Q08_TERMS)
    q09 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_q09})
    s19_pars, _sf19 = E23.s19_models()
    E_s19, _lp19, _h19 = E23.s19_term(prep, inner, mp, s19_pars)
    terms19 = terms9 + ["SET"]
    q19 = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s19})
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]], "Q19": [q19["rb"], ref["Q19"]],
            "Q19_mono": [q19["mono_rb"], ref["Q19_mono"]]}
    del q09, q19, E_s19
    E_s25, lp_s25, has_s25 = E23.s19_term(prep, inner, mp, s25)
    q25 = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25})
    seam["Q25"] = [q25["rb"], L.REF_Q25]
    seam["Q25_mono"] = [q25["mono_rb"], L.REF_Q25_MONO]
    rep25 = r.read_json(L.S25_EVAL_REPORT)
    seam_ref_bound = bool(rep25["candidate"]["inner_rank_balanced"] == L.REF_Q25
                          and rep25["candidate"]["single_colour_rank_balanced"] == L.REF_Q25_MONO)
    seam1 = bool(seam_ref_bound and all(abs(x - y) <= TOL for x, y in seam.values()))
    seam1_bitwise = all(x == y for x, y in seam.values())
    # ---- seam P, D
    want = (r.read_json(E23.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {"probe_X.npy": E23.sha_file(os.path.join(L.PROBE_PREP, "probe_X.npy")),
           "probe_ex.npy": E23.sha_file(os.path.join(L.PROBE_PREP, "probe_ex.npy")),
           "inner_X.npy": E23.sha_file(os.path.join(E23.INNER_PREP, "inner_X.npy")),
           "inner_ex.npy": E23.sha_file(os.path.join(E23.INNER_PREP, "inner_ex.npy"))}
    sp = {"want": {k: want.get(k) for k in got}, "got": got,
          "train_record_seam_p": (tr.get("seam_p") or {}).get("ok")}
    sp["ok"] = bool(all(want.get(k) == v for k, v in got.items()) and (smoke or sp["train_record_seam_p"] is True))
    sd = seam_d(inner, tr)
    log(f"seam 1 {seam1} (bitwise {seam1_bitwise}) {seam}; seam P {sp['ok']}; seam D {sd['ok']}; seam M {sm['ok']}; "
        f"seam S25 {sf['ok']}")
    if not (seam1 and sp["ok"] and sd["ok"] and sm["ok"] and sf["ok"]):
        E23.write_json(out_path, {**base, "status": "stop: seam failed", "seam": seam, "seam_ref_bound": seam_ref_bound,
                                  "seam_p": sp, "seam_d": sd, "seam_m": sm, "seam_s25": sf})
        return 1
    # ---- the judged part starts here (after every seam, before any candidate number)
    if not smoke:
        E23.write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                 "rerun_reason": a.reason if a.rerun else None})
    tc = time.time()
    H, ex_info = E23.inner_h(prep, inner, mp, s40, smoke)
    E_s40, E_seed, lp_s40, has_s40, set_info = E23.set2_terms(inner, H, s40)
    cand = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s40})
    cmp_ = inner.compare(cand, q25)
    g = K23.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                  cmp_["mono_by_colour"])
    log(f"Q40 {cand['rb']!r} vs Q25 {q25['rb']!r}: {json.dumps(g)} ({time.time() - tc:.0f}s)")
    # ---- record only
    chk = {"eval_rb_own": K19.rank_balanced(cand["sc"], inner.Xe.y),
           "mono_rb_own": K19.rank_balanced(cand["msc"], inner.ms["y"])}
    chk["ok"] = bool(abs(chk["eval_rb_own"] - cand["rb"]) <= TOL and abs(chk["mono_rb_own"] - cand["mono_rb"]) <= TOL)
    counts = E23.by_count(inner, cand, q25)
    levels = E23.by_level(inner, cand, q25)
    weights = E23.w_record(cand, ["P1TO", "SET"])
    q25_weights = E23.w_record(q25, ["P1TO", "SET"])
    cand_short = inner.short(cand, q25)
    del cand
    add = inner.run(terms19 + ["SET2"], {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25, "SET2": E_s40})
    add_rec = {**inner.short(add, q25), "weights_by_fold": E23.w_record(add, ["P1TO", "SET", "SET2"])}
    del add
    E3, _s3, _lp3, _h3, _i3 = E23.set2_terms(inner, H[:3], s40[:3])
    r3 = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E3})
    seeds_0_2 = {"inner_rank_balanced": r3["rb"], "single_colour_rank_balanced": r3["mono_rb"],
                 "vs_q25": inner.compare(r3, q25)}
    del r3, E3
    per_seed = {}
    for i, s in enumerate(K23.SEEDS):
        rs = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_seed[i]})
        per_seed[str(s)] = {"inner_rank_balanced": rs["rb"], "single_colour_rank_balanced": rs["mono_rb"],
                            "vs_q25": inner.compare(rs, q25)}
        del rs
    rec_only = {"Q25+S40_as_8th_term": add_rec, "S40_seeds_0_2_only": seeds_0_2,
                "S40_alone_rule_b": E23.alone(inner, lp_s40, has_s40),
                "S25_alone_rule_b": E23.alone(inner, lp_s25, has_s25),
                "per_seed_Q40_s": per_seed, "by_game_count": counts, "by_level": levels,
                "weights_by_fold": weights, "q25_weights_by_fold": q25_weights, "set_term_info": set_info,
                "inner_ex_raw_nonfinite": ex_info, "own_rank_balanced_check": chk, "seam1_bitwise": seam1_bitwise}
    status = "gates passed (worth a PREREG_40; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {**base, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("smoke S40 models (noise-trained, 300 steps), noise inner inputs: numbers are meaningless"
                          if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md 5: inner-only judgement of Q40 (Q25 with the SET term's models S25 replaced by S40 = "
                   "prereg23's SET2 trained on prereg25's rows); Q09's player-fold OOF fusion; no validation scoring",
           "status": status, "seam": seam, "seam_ref_bound": seam_ref_bound, "seam_p": sp, "seam_d": sd,
           "seam_m": sm, "seam_s25": sf, "train_record_sha256_lf": E23.sha_lf(tr_path),
           "stage0b_sha256_lf": None if smoke else E23.sha_lf(STAGE0B),
           "candidate": {"name": "Q40", "terms": terms19, **cand_short},
           "baseline": {"name": "Q25", "terms": terms19, "inner_rank_balanced": q25["rb"],
                        "single_colour_rank_balanced": q25["mono_rb"]},
           "gates": g, "record_only": rec_only, "q09_map_sha256": map_sha, "peak_memory_record": E23.peak_memory()}
    E23.write_json(out_path, rep)
    log(f"-> {out_path}: {status} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage0b", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"stage0b": cmd_stage0b, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
