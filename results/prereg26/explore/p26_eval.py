"""prereg26 exploration (competition 2): inner-only judgement of PLAN.md (one candidate Q26 = Q25 with the P1TO term's
map P1TO_E replaced by P1TO_E26), judged once. CPU only (.venv; numpy forward, no torch).

    Git Bash, repository root (subst drive); PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS /
    MKL variable
    S=results/prereg26/explore
    $S/p26_eval.py eval            -> $S/eval_report.json (judged once; code and train_record.json committed and clean)
    $S/p26_eval.py eval --smoke    -> data/prereg26/smoke/eval_report.json (the smoke map trained on noise: pipeline
                                      and timing only, never a result)

The Q09 / Q19 / Q25 rebuild, the SET term (S19 / S25 through Q09's map mu / sd), the scores and the record-only helpers
are imported unchanged from results/prereg23/explore/p23_eval.py (the code that reproduced Q19 and Q25 bit-exactly in
prereg23 / prereg25); the fusion is tools/p09_rank_fuse.Inner (unchanged); the P1TO term of a map is
p09_rank_fuse.inner_map_lp -> Inner.E_p1 (unchanged, x 4). Never data/local, never validation features, never
tools/score.py.
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
import p26_lib as L  # noqa: E402
import p09_map as M  # noqa: E402  (BATCH, FOLDS: the registered training settings; nothing is trained here)

r, ft_eval, FW = E23.r, E23.ft_eval, E23.FW
guard_path = E23.guard_path
TOL = 1e-12
DATA = L.DATA
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "eval_started.json")
SEAM_FAIL = os.path.join(DATA, "eval_seam_fail.json")     # a seam stop is written here (no candidate number exists)
FULL_ROWS = {"rows": 344946, "probe_rows": 270000, "v_rows": 74946, "players": 7367}
REPORT = os.path.join(HERE, "eval_report.json")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p26_lib.py", "p26_prep.py", "p26_train.py", "p26_eval.py", "test_p26.py",
             "train_record.json", "v_ex_record.json", "run_train.sh", "run_eval.sh")
IMPORTED = tuple(E23.IMPORTED) + ("results/prereg23/explore/p23_eval.py", "results/prereg23/explore/p23_lib.py",
                                  "results/prereg25/explore/p25_lib.py", "results/prereg25/explore/eval_report.json",
                                  "results/prereg25/explore/train_record.json",
                                  "results/prereg25/rank/s25_seed0.npz", "results/prereg25/rank/s25_seed1.npz",
                                  "results/prereg25/rank/s25_seed2.npz")
FORWARD_FILES = ("tools/p09_forward.py", "tools/p09_map.py", "results/prereg26/explore/p26_train.py",
                 "results/prereg26/explore/p26_lib.py")
FORMAT = "c10-prereg26 explore eval v1"


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


def refs25():
    e = r.read_json(L.P25_EVAL)
    if e.get("smoke") or not e["gates"]["passed"] or e["candidate"]["name"] != "Q25":
        raise SystemExit("results/prereg25/explore/eval_report.json is not the judged Q25 report")
    return {"Q25": e["candidate"]["inner_rank_balanced"], "Q25_mono": e["candidate"]["single_colour_rank_balanced"]}


def seam_m(tr, smoke, cs):
    """map26 = the train record's; members as registered; settings registered; code clean; forward code unchanged;
    port check passed."""
    want_settings = {"members": [dict(m) for m in FW.MEMBERS], "configs": json.loads(json.dumps(FW.CONFIGS))}
    s = tr.get("settings") or {}
    out = {"settings_registered": bool(s.get("members") == want_settings["members"]
                                       and s.get("configs") == want_settings["configs"]
                                       and s.get("batch") == M.BATCH and s.get("folds") == M.FOLDS
                                       and s.get("fold_seed") == 0 and s.get("threads") == r.VENV["threads"]
                                       and (s.get("smoke_epochs") is None) != smoke),
           "train_code_clean": bool((tr.get("code") or {}).get("dirty") == "")}
    tr_sha = (tr.get("code") or {}).get("sha256_lf") or {}
    out["forward_code_same_as_train"] = {f: bool(tr_sha.get(f) is not None and tr_sha.get(f) == cs["sha256_lf"].get(f))
                                         for f in FORWARD_FILES}
    path = os.path.join(REPO, *tr["map_file"].split("/"))
    sha = E23.sha_file(path)
    mp = FW.load_map(path)
    out["map_sha256"] = sha
    out["map_is_train_record"] = bool(sha == tr["map_sha256"])
    out["members_registered"] = bool([(m["name"], m["kind"], m["seed"]) for m in mp["members"]]
                                     == [(m["name"], m["kind"], m["seed"]) for m in FW.MEMBERS])
    out["map_smoke_flag"] = bool(mp["record"].get("smoke"))
    rows = tr.get("rows") or {}
    out["rows_full"] = bool(smoke or all(rows.get(k) == v for k, v in FULL_ROWS.items()))
    sched = {}
    for spec in FW.MEMBERS:
        mr = (tr.get("members") or {}).get(spec["name"]) or {}
        ep = (mr.get("cv") or {}).get("epochs_schedule")
        want = FW.CONFIGS[spec["kind"]]["epochs"] if not smoke else s.get("smoke_epochs")
        sched[spec["name"]] = bool(ep == want and isinstance(mr.get("e_star"), int) and 1 <= mr["e_star"] <= ep
                                   and (spec["kind"] != "lowrank" or mr["e_star"] == ep))
    out["member_schedules_registered"] = sched
    out["map_e_star_is_train_record"] = bool([m["e_star"] for m in mp["members"]]
                                             == [((tr.get("members") or {}).get(m["name"]) or {}).get("e_star")
                                                 for m in FW.MEMBERS])
    out["port_check_ok"] = bool((tr.get("port_check") or {}).get("ok"))
    out["seam_t_ok"] = bool((tr.get("seam_t") or {}).get("ok"))
    out["ok"] = bool(out["settings_registered"] and (smoke or out["train_code_clean"])
                     and all(out["forward_code_same_as_train"].values()) and out["map_is_train_record"]
                     and out["members_registered"] and out["map_smoke_flag"] == smoke and out["port_check_ok"]
                     and out["rows_full"] and all(sched.values()) and out["map_e_star_is_train_record"]
                     and out["seam_t_ok"])
    return mp, out


def seam_p(tr):
    want = (r.read_json(E23.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {f: E23.sha_file(os.path.join(L.PROBE_PREP, f)) for f in L.PROBE_FILES}
    got.update({f: E23.sha_file(os.path.join(L.INNER_PREP, f)) for f in ("inner_X.npy", "inner_ex.npy")})
    vrec = r.read_json(L.V_EX_RECORD)
    rec25 = r.read_json(os.path.join(L.V_PREP, "prep_record.json"))
    v = {"v_X.npy": E23.sha_file(os.path.join(L.V_PREP, "v_X.npy")),
         "v_meta.npz": E23.sha_file(os.path.join(L.V_PREP, "v_meta.npz")), "v_ex.npy": E23.sha_file(L.V_EX)}
    out = {"want": {k: want.get(k) for k in got}, "got": got, "v": v,
           "v_X_meta_are_prereg25": bool(v["v_X.npy"] == rec25["files_sha256"]["v_X.npy"]
                                         and v["v_meta.npz"] == rec25["files_sha256"]["v_meta.npz"]),
           "v_ex_is_record": bool(v["v_ex.npy"] == vrec["v_ex_sha256"]),
           "train_record_same_v": bool(((tr.get("seam_p") or {}).get("v") or {}) == v)}
    out["ok"] = bool(all(want.get(k) == s for k, s in got.items()) and out["v_X_meta_are_prereg25"]
                     and out["v_ex_is_record"] and out["train_record_same_v"])
    return out


def seam_d(inner):
    with np.load(guard_path(os.path.join(L.PROBE_PREP, "probe_meta.npz")), allow_pickle=False) as z:
        pkeys = set(np.asarray(z["players"]).astype(str).tolist())
    with np.load(guard_path(os.path.join(L.V_PREP, "v_meta.npz")), allow_pickle=False) as z:
        vkeys = set(np.asarray(z["players"]).astype(str).tolist())
    allk = pkeys | vkeys
    players = inner.d["inner"]["players"]
    i_keys = {f"{p['rank']}:{p['player_id']}" for p in players}
    i_ids = {p["player_id"] for p in players}
    aud = r.read_json(os.path.join(REPO, "results", "prereg25", "explore", "audit.json"))
    out = {"train_players": len(allk), "probe_players": len(pkeys), "V_players": len(vkeys), "I_players": len(i_keys),
           "train_and_I": len(allk & i_keys), "train_and_I_player_id": len(L.player_ids(allk) & i_ids),
           "probe_and_V": len(pkeys & vkeys), "probe_and_V_player_id": len(L.player_ids(pkeys) & L.player_ids(vkeys)),
           "V_games_vs_inner_8_symmetries_audit": aud["duplicates"]["val_vs_inner_8_symmetries"]}
    out["ok"] = bool(out["train_and_I"] == 0 and out["train_and_I_player_id"] == 0 and out["probe_and_V"] == 0
                     and out["probe_and_V_player_id"] == 0 and out["V_games_vs_inner_8_symmetries_audit"] == 0)
    return out


def lp_record(lp_new, lp_old):
    d = np.abs(np.exp(lp_new) - np.exp(lp_old))
    el = lambda lp: (np.exp(lp) * np.arange(lp.shape[1])).sum(1)  # noqa: E731
    return {"rows": int(len(lp_new)), "mean_abs_prob_diff": float(d.mean()), "max_abs_prob_diff": float(d.max()),
            "mean_abs_expected_level_diff": float(np.abs(el(lp_new) - el(lp_old)).mean()),
            "argmax_agreement": float((lp_new.argmax(1) == lp_old.argmax(1)).mean())}


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
            raise SystemExit(f"the prereg26 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path)
    base = {"format": FORMAT, "smoke": smoke, "code": cs}
    tr = r.read_json(train_path)
    if bool(tr.get("smoke")) != smoke:
        raise SystemExit("train_record.json smoke flag differs from this run")
    mp26, sm = seam_m(tr, smoke, cs)
    s19_pars, sf19 = E23.s19_models()
    s25_pars, sf25 = L.s25_models(K19, E23.load_npz_model)
    map_sha = r.sha_raw(E23.Q09_MAP)
    sf = {"S19": sf19, "S25": sf25, "q09_map_sha256": map_sha,
          "q09_map_is_frozen": bool(map_sha == r.read_json(E23.Q09_FROZEN)["map"]["sha256"])}
    sf["ok"] = bool(sf19["ok"] and sf25["ok"] and sf["q09_map_is_frozen"])
    # ---- Q09, Q19, Q25 and seam 1 (= prereg23 / prereg25's rebuild)
    prep = ft_eval.InnerPrep(guard_path(L.INNER_PREP))
    inner = r.Inner(prep, log=log)
    ref = {**E23.refs(), **refs25()}
    mp = FW.load_map(E23.Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    terms9 = list(r.Q08_TERMS)
    q09 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_q09})
    E_s19, _lp, _h = E23.s19_term(prep, inner, mp, s19_pars)
    terms19 = terms9 + ["SET"]
    q19 = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s19})
    del E_s19
    E_s25, _lp, _h = E23.s19_term(prep, inner, mp, s25_pars)
    q25 = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25})
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]], "Q19": [q19["rb"], ref["Q19"]],
            "Q19_mono": [q19["mono_rb"], ref["Q19_mono"]], "Q25": [q25["rb"], ref["Q25"]],
            "Q25_mono": [q25["mono_rb"], ref["Q25_mono"]]}
    seam1 = all(abs(x - y) <= TOL for x, y in seam.values())
    seam1_bitwise = all(x == y for x, y in seam.values())
    del q19
    sp = seam_p(tr)
    sd = seam_d(inner)
    log(f"seam 1 {seam1} (bitwise {seam1_bitwise}) {seam}; seam P {sp['ok']}; seam D {sd}; seam M {sm['ok']}; "
        f"seam F {sf['ok']}")
    if not (seam1 and sp["ok"] and sd["ok"] and sm["ok"] and sf["ok"]):
        E23.write_json(os.path.join(SMOKE_DIR, "eval_seam_fail.json") if smoke else SEAM_FAIL,
                       {**base, "status": "stop: seam failed", "seam": seam, "seam_p": sp, "seam_d": sd,
                                  "seam_m": sm, "seam_f": sf})
        return 1
    # ---- the judged part starts here (after every seam, before any candidate number)
    if not smoke:
        E23.write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                                 "rerun_reason": a.reason if a.rerun else None})
    tc = time.time()
    lp26 = r.inner_map_lp(prep, mp26)
    E26, _ = inner.E_p1(lp26)
    cand = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E26, "SET": E_s25})
    cmp_ = inner.compare(cand, q25)
    g = K23.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                  cmp_["mono_by_colour"])
    log(f"Q26 {cand['rb']!r} vs Q25 {q25['rb']!r}: {json.dumps(g)} ({time.time() - tc:.0f}s)")
    # ---- record only
    chk = {"eval_rb_own": K19.rank_balanced(cand["sc"], inner.Xe.y),
           "mono_rb_own": K19.rank_balanced(cand["msc"], inner.ms["y"])}
    chk["ok"] = bool(abs(chk["eval_rb_own"] - cand["rb"]) <= TOL and abs(chk["mono_rb_own"] - cand["mono_rb"]) <= TOL)
    counts = E23.by_count(inner, cand, q25)
    levels = E23.by_level(inner, cand, q25)
    q09_26 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E26})
    q09_swap = {**inner.short(q09_26, q09), "by_game_count": E23.by_count(inner, q09_26, q09),
                "by_level": E23.by_level(inner, q09_26, q09)}
    del q09_26
    add = inner.run(terms19 + ["P1TO2"], {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25, "P1TO2": E26})
    add_rec = {**inner.short(add, q25), "weights_by_fold": E23.w_record(add, ["P1TO", "SET", "P1TO2"])}
    del add
    rec_only = {"Q09_with_P1TO_E26_vs_Q09": q09_swap, "Q25+P1TO_E26_as_8th_term": add_rec,
                "by_game_count": counts, "by_level": levels,
                "weights_by_fold": E23.w_record(cand, ["P1TO", "SET"]),
                "q25_weights_by_fold": E23.w_record(q25, ["P1TO", "SET"]),
                "lp_P1TO_E26_vs_P1TO_E_inner_rows": lp_record(lp26, lp_q09),
                "own_rank_balanced_check": chk, "seam1_bitwise": seam1_bitwise}
    status = "gates passed (worth a PREREG_26; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {**base, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("smoke map (noise-trained, 2-epoch schedules): numbers are meaningless" if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: inner-only judgement of Q26 (Q25 with the P1TO term's map P1TO_E replaced by P1TO_E26 = "
                   "09's ens3fam retrained on probe + validation players; the SET term (S25, Q09's mu / sd) unchanged; "
                   "Q09's player-fold OOF fusion); no validation scoring",
           "status": status, "seam": seam, "seam_p": sp, "seam_d": sd, "seam_m": sm, "seam_f": sf,
           "train_record_sha256_lf": E23.sha_lf(train_path),
           "candidate": {"name": "Q26", "terms": terms19, **inner.short(cand, q25)},
           "baseline": {"name": "Q25", "terms": terms19, **inner.short(q25, q09)}, "gates": g,
           "record_only": rec_only, "q09_map_sha256": map_sha, "map26_sha256": sm["map_sha256"],
           "peak_memory_record": E23.peak_memory()}
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
