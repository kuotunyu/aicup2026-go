"""prereg25 exploration (competition 2): inner-only judgement of PLAN.md (one candidate Q25 = Q19 with the SET term's
three S19 models replaced by the three S25 models), judged once. CPU only (.venv; numpy forward, no torch).

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg25/explore
    $S/p25_eval.py eval            -> $S/eval_report.json (judged once; code and train_record.json committed and clean)
    $S/p25_eval.py eval --smoke    -> data/prereg25/smoke/eval_report.json (smoke S25 models trained on noise: pipeline
                                      and timing only, never a result)

The Q19 baseline, the SET term (per-game h -> mean over the question's P1-usable games -> head, x 4), the scores and the
record-only helpers are imported unchanged from results/prereg23/explore/p23_eval.py (the code that reproduced Q19
bit-exactly in prereg23); the fusion is tools/p09_rank_fuse.Inner (unchanged). Never data/local, never validation
features, never tools/score.py.
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
import p25_lib as L  # noqa: E402

r, ft_eval, FW = E23.r, E23.ft_eval, E23.FW
guard_path = E23.guard_path
TOL = 1e-12
DATA = os.path.join(REPO, "data", "prereg25")
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "eval_started.json")
REPORT = os.path.join(HERE, "eval_report.json")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
AUDIT = os.path.join(HERE, "audit.json")
V_PREP = os.path.join(DATA, "prep")
OWN_FILES = ("PLAN.md", "p25_lib.py", "p25_train.py", "p25_eval.py", "test_p25.py", "train_record.json", "audit.json")
IMPORTED = tuple(E23.IMPORTED) + ("results/prereg23/explore/p23_eval.py", "results/prereg23/explore/p23_lib.py",
                                  "results/prereg19/explore/p19_train.py")
FORWARD_FILES = ("results/prereg19/explore/p19_lib.py", "results/prereg19/explore/p19_train.py",
                 "results/prereg25/explore/p25_train.py", "tools/p09_forward.py")
FORMAT = "c10-prereg25 explore eval v1"
MODEL_FORMAT = "c10-prereg25 s25 model v1"


def log(*x):
    print(*x, flush=True)


def code_state(smoke):
    own = [f for f in OWN_FILES if not (smoke and f == "train_record.json")]
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in own] + list(IMPORTED)
    rel = sorted(set(rel) | E23.loaded_repo_modules() | L.loaded_repo_modules())
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
    want = {"steps": K19.STEPS, "eval_every": K19.EVAL_EVERY, "hold_per": K19.HOLD_PER, "seeds": list(K19.SEEDS),
            "batch": K19.BATCH, "ks": list(K19.KS), "lr": K19.LR, "wd": K19.WD, "pct_start": K19.PCT_START,
            "hidden": K19.HIDDEN, "h_dim": K19.H_DIM, "rho": K19.RHO, "drop_in": K19.DROP_IN, "drop": K19.DROP,
            "threads": K19.THREADS, "hold_seed": K19.HOLD_SEED, "sampler_base": K19.SAMPLER_BASE, "salt": L.SALT}
    if smoke:
        want.update(steps=K19.SMOKE["steps"], eval_every=K19.SMOKE["eval_every"], hold_per=K19.SMOKE["hold_per"])
    out = {"models": {}, "train_settings_registered": bool(all(tr["settings"].get(k) == v for k, v in want.items())),
           "train_code_clean": bool((tr.get("code") or {}).get("dirty") == "")}
    tr_sha = (tr.get("code") or {}).get("sha256_lf") or {}
    out["forward_code_same_as_train"] = {f: bool(tr_sha.get(f) is not None and tr_sha.get(f) == cs["sha256_lf"].get(f))
                                         for f in FORWARD_FILES}
    ok = bool(out["train_settings_registered"] and (smoke or out["train_code_clean"])
              and all(out["forward_code_same_as_train"].values()))
    pars = []
    for s in K19.SEEDS:
        sr = tr["seeds"][str(s)]
        mpath = os.path.join(REPO, *sr["model_file"].split("/"))
        sha = E23.sha_file(mpath)
        par, seed, best_step, msmoke = E23.load_npz_model(mpath, MODEL_FORMAT)
        good = bool(sha == tr["models_sha256"][str(s)] and seed == s and best_step == sr["best_step"]
                    and msmoke == smoke and (sr.get("port_check") or {}).get("ok"))
        out["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": good}
        ok = ok and good
        pars.append(K19.p64(par))
    out["ok"] = bool(ok)
    return pars, out


def seam_d(inner, tr):
    with np.load(guard_path(os.path.join(E23.PROBE_PREP, "probe_meta.npz")), allow_pickle=False) as z:
        pkeys = set(np.asarray(z["players"]).astype(str).tolist())
    with np.load(guard_path(os.path.join(V_PREP, "v_meta.npz")), allow_pickle=False) as z:
        vkeys = set(np.asarray(z["players"]).astype(str).tolist())
    allk = pkeys | vkeys
    a_keys = {k for k in allk if L.holdout_bit(k) == 0}
    h_keys = {k for k in allk if L.holdout_bit(k) == 1}
    players = inner.d["inner"]["players"]
    i_keys = {f"{p['rank']}:{p['player_id']}" for p in players}
    i_ids = {p["player_id"] for p in players}
    pid = lambda ks: {k.split(":", 1)[1] for k in ks}  # noqa: E731
    with open(AUDIT, encoding="utf-8") as f:
        aud = json.load(f)
    out = {"A_players": len(a_keys), "H_players": len(h_keys), "I_players": len(i_keys), "V_players": len(vkeys),
           "A_and_H": len(a_keys & h_keys), "train_and_I": len(allk & i_keys), "train_and_I_player_id": len(pid(allk) & i_ids),
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
    out_path = os.path.join(SMOKE_DIR, "eval_report.json") if smoke else REPORT
    train_path = os.path.join(SMOKE_DIR, "train_record.json") if smoke else TRAIN_RECORD
    cs = code_state(smoke)
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg25 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path)
    base = {"format": FORMAT, "smoke": smoke, "code": cs}
    tr = r.read_json(train_path)
    if bool(tr.get("smoke")) != smoke:
        raise SystemExit("train_record.json smoke flag differs from this run")
    s25_pars, sm = seam_m(tr, smoke, cs)
    s19_pars, sf = E23.s19_models()
    # ---- Q09, Q19 and seam 1 (= prereg23's rebuild)
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
    E_s19, lp_s19, has_s19 = E23.s19_term(prep, inner, mp, s19_pars)
    terms19 = terms9 + ["SET"]
    q19 = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s19})
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]], "Q19": [q19["rb"], ref["Q19"]],
            "Q19_mono": [q19["mono_rb"], ref["Q19_mono"]]}
    seam1 = all(abs(x - y) <= TOL for x, y in seam.values())
    seam1_bitwise = all(x == y for x, y in seam.values())
    # ---- seam P, D
    want = (r.read_json(E23.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {"probe_X.npy": E23.sha_file(os.path.join(E23.PROBE_PREP, "probe_X.npy")),
           "inner_X.npy": E23.sha_file(os.path.join(E23.INNER_PREP, "inner_X.npy"))}
    vrec = r.read_json(os.path.join(V_PREP, "prep_record.json"))
    v_sha = E23.sha_file(os.path.join(V_PREP, "v_X.npy"))
    sp = {"want": {k: want.get(k) for k in got}, "got": got, "v_X.npy": v_sha,
          "v_X_is_prep_record": bool(v_sha == vrec["files_sha256"]["v_X.npy"]),
          "v_X_is_train_record": bool(v_sha == (tr.get("seam_p") or {}).get("v_X_sha256"))}
    sp["ok"] = bool(all(want.get(k) == v for k, v in got.items()) and sp["v_X_is_prep_record"]
                    and (smoke or sp["v_X_is_train_record"]))
    sd = seam_d(inner, tr)
    log(f"seam 1 {seam1} (bitwise {seam1_bitwise}) {seam}; seam P {sp['ok']}; seam D {sd}; seam M {sm['ok']}; "
        f"seam F {sf['ok']}")
    if not (seam1 and sp["ok"] and sd["ok"] and sm["ok"] and sf["ok"]):
        T = {**base, "status": "stop: seam failed", "seam": seam, "seam_p": sp, "seam_d": sd, "seam_m": sm,
             "seam_f": sf}
        E23.write_json(out_path, T)
        return 1
    # ---- the judged part starts here (after every seam, before any candidate number)
    if not smoke:
        E23.write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                 "rerun_reason": a.reason if a.rerun else None})
    tc = time.time()
    E_s25, lp_s25, has_s25 = E23.s19_term(prep, inner, mp, s25_pars)
    cand = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25})
    cmp_ = inner.compare(cand, q19)
    g = K23.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                  cmp_["mono_by_colour"])
    log(f"Q25 {cand['rb']!r} vs Q19 {q19['rb']!r}: {json.dumps(g)} ({time.time() - tc:.0f}s)")
    # ---- record only
    chk = {"eval_rb_own": K19.rank_balanced(cand["sc"], inner.Xe.y),
           "mono_rb_own": K19.rank_balanced(cand["msc"], inner.ms["y"])}
    chk["ok"] = bool(abs(chk["eval_rb_own"] - cand["rb"]) <= TOL and abs(chk["mono_rb_own"] - cand["mono_rb"]) <= TOL)
    counts = E23.by_count(inner, cand, q19)
    levels = E23.by_level(inner, cand, q19)
    add = inner.run(terms19 + ["SET2"], {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s19, "SET2": E_s25})
    add_rec = {**inner.short(add, q19), "weights_by_fold": E23.w_record(add, ["P1TO", "SET", "SET2"])}
    del add
    per_seed = {}
    for i, s in enumerate(K19.SEEDS):
        E_i, _lp, _h = E23.s19_term(prep, inner, mp, [s25_pars[i]])
        rs = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_i})
        per_seed[str(s)] = {"inner_rank_balanced": rs["rb"], "single_colour_rank_balanced": rs["mono_rb"],
                            "vs_q19": inner.compare(rs, q19)}
        del rs, E_i
    rec_only = {"Q19+S25_as_8th_term": add_rec,
                "S25_alone_rule_b": E23.alone(inner, lp_s25, has_s25),
                "S19_alone_rule_b": E23.alone(inner, lp_s19, has_s19),
                "per_seed_Q25_s": per_seed, "by_game_count": counts, "by_level": levels,
                "weights_by_fold": E23.w_record(cand, ["P1TO", "SET"]),
                "q19_weights_by_fold": E23.w_record(q19, ["P1TO", "SET"]),
                "own_rank_balanced_check": chk, "seam1_bitwise": seam1_bitwise}
    status = "gates passed (worth a PREREG_25; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {**base, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("smoke S25 models (noise-trained, 300 steps): numbers are meaningless" if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: inner-only judgement of Q25 (Q19 with the SET term's models S19 replaced by S25 = S19 "
                   "retrained on probe + validation players; Q09's player-fold OOF fusion); no validation scoring",
           "status": status, "seam": seam, "seam_p": sp, "seam_d": sd, "seam_m": sm, "seam_f": sf,
           "train_record_sha256_lf": E23.sha_lf(train_path),
           "candidate": {"name": "Q25", "terms": terms19, **inner.short(cand, q19)},
           "baseline": {"name": "Q19", "terms": terms19, **inner.short(q19, q09)}, "gates": g,
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
