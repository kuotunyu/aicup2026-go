"""prereg19 exploration (competition 2): inner-only judgement of PLAN.md (one candidate Q09+SET: the question-level
set model S19 as a 7th fusion term), judged once. CPU only (.venv; numpy forward, no torch).

    Git Bash, repository root (a subst drive when the worktree path contains 'AppData\\Local'); PYTHONDONTWRITEBYTECODE=1;
    .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg19/explore
    $S/p19_eval.py eval            -> $S/eval_report.json (judged once; code and train_record.json committed and clean)
    $S/p19_eval.py eval --smoke    -> data/prereg19/smoke/eval_report.json (smoke models, inner P1 features replaced by
                                      default_rng(12345) normal noise: pipeline and timing only, never a result)

Inputs (read only): data/prereg12/q09_inner (inner P1 features), data/prereg17/prep (probe player keys; seam P/D),
the S19 model files of train_record.json, Q09's frozen p1e_map.npz, u1_lib's inner pseudo-questions and per-game terms
through tools/p09_rank_fuse.Inner. Never data/local, never validation features, never tools/score.py.
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
import p19_lib as K  # noqa: E402

R, rf, p3, p4 = r.R, r.rf, r.p3, r.p4
TOL = K.TOL
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_TUNE = os.path.join(REPO, "results", "prereg09", "rank", "tune_report.json")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
Q09_CHECK = os.path.join(REPO, "results", "prereg09", "rank", "check_report.json")
INNER_PREP = os.path.join(REPO, "data", "prereg12", "q09_inner")
PROBE_PREP = os.path.join(REPO, "data", "prereg17", "prep")
DATA = os.path.join(REPO, "data", "prereg19")
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "eval_started.json")
REPORT = os.path.join(HERE, "eval_report.json")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p19_lib.py", "p19_train.py", "p19_eval.py", "test_p19.py", "train_record.json")
IMPORTED = ("tools/p02_rank_fuse.py", "tools/p03_rank_fuse.py", "tools/p04_rank_fuse.py", "tools/p07_rank_fuse.py",
            "tools/p08_rank_fuse.py", "tools/p09_rank_fuse.py", "tools/p09_forward.py", "tools/p09_map.py",
            "tools/p08_p1.py", "results/prereg12/explore/finetune/ft_eval.py", "results/prereg05/explore/u1/u1_lib.py",
            "results/prereg07/explore/pilot_humansl/pilot_fuse.py", "results/prereg07/explore/pilot_humansl/hs_lib.py",
            "results/prereg08/p1/p1_fuse.py", "results/prereg08/p1/p1_lib.py",
            "results/prereg09/rank/p1e_map.npz", "results/prereg09/rank/check_report.json",
            "results/prereg09/rank/tune_report.json", "results/prereg09/rank/frozen_config.json",
            "results/prereg08/rank/p1_map.npz")
FORMAT = "c10-prereg19 explore eval v1"
MODEL_FORMAT = "c10-prereg19 s19 model v1"
NOISE_SEED = 12345
BLOCK = 8192
guard_path = ft_eval.guard_path


def log(*x):
    print(*x, flush=True)


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


def sha_file(path):
    h = hashlib.sha256()
    with open(guard_path(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def loaded_repo_modules():
    """Repository files of every imported module (code_state coverage)."""
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).lower().startswith(REPO.lower() + os.sep):
            out.add(os.path.relpath(os.path.abspath(f), REPO).replace("\\", "/"))
    return out


def code_state(smoke):
    own = [f for f in OWN_FILES if not (smoke and f == "train_record.json")]
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in own] + list(IMPORTED)
    rel = sorted(set(rel) | loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    dirty = git("status", "--porcelain", "--", *rel)
    return {"head": git("rev-parse", "HEAD"), "dirty": dirty,
            "sha256_lf": {x: sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def once_guard(a, cs, out_path):
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


def load_model(path):
    with np.load(guard_path(path), allow_pickle=False) as z:
        if str(z["format"]) != MODEL_FORMAT:
            raise SystemExit(f"{path}: not an S19 model file")
        return {k: np.asarray(z[f"p_{k}"]) for k in K.PARAM_NAMES}, int(z["seed"]), int(z["best_step"]), \
            bool(z["smoke"])


# ---------------------------------------------------------------- the SET term (PLAN.md 3.6, 4)
def inner_h(prep, inner, mu, sd, par_list, smoke):
    """Per-game h of every seed on the full inner row space (rows without P1 features stay 0, never pooled)."""
    n = len(prep.inner_ref)
    H = [np.zeros((inner.n_inner, K.H_DIM)) for _ in par_list]
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        if smoke:
            X = np.random.default_rng([NOISE_SEED, 1, s]).standard_normal((e - s, K.D_IN)).astype(np.float32)
            X = X.astype(np.float64)
        else:
            X = FW.standardise_x(np.asarray(prep.inner_X[s:e], dtype=np.float64), mu, sd)
        rows = prep.inner_ref[s:e]
        for par, Hs in zip(par_list, H):
            Hs[rows] = K.h_np(par, X)
    return H


def set_terms(inner, H, par_list):
    """E_SET (x 4) and per-seed E for fit / eval / mono; usable = Inner.covered (= Q09's P1TO mask)."""
    di = inner.d["inner"]
    E, E_seed, lp, has, info = {}, [dict() for _ in par_list], {}, {}, {}
    for which, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"])):
        groups = K.usable_groups(S, inner.covered)
        E[which], per, lp[which], has[which] = K.set_term(par_list, H, groups, inner.n_inner)
        for i, x in enumerate(per):
            E_seed[i][which] = x
        info[which] = {"questions": int(len(groups)), "without_usable_game": int((~has[which]).sum())}
    return E, E_seed, lp, has, info


def seam_e(inner, lp_q09):
    """Own pooling (usable groups, averaging matrix) on 4 x lp_P1TO_E vs Inner.E_p1."""
    E_ref, _ = inner.E_p1(lp_q09)
    lp4 = K.SCALE * inner.lp_full(lp_q09)
    di = inner.d["inner"]
    out = {}
    for which, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"])):
        g = K.usable_groups(S, inner.covered)
        own = np.asarray(K.mean_matrix(g, inner.n_inner) @ lp4)
        out[which] = float(np.max(np.abs(own - E_ref[which])))
    out["ok"] = bool(all(v <= TOL for v in out.values()))
    return out


# ---------------------------------------------------------------- records
def scored(inner, P, Pm):
    m, sc = rf.metrics(rf.rule_b_np(P), inner.Xe.y, inner.Xe.n, inner.Xe.player)
    mrec, msc = p3.mono_score_record(rf.rule_b_np(Pm), inner.Xm, inner.ms, inner.mboot)
    return {"sc": sc, "msc": msc, "rb": m["rank_balanced"], "mono_rb": mrec["rank_balanced"], "m": m, "mrec": mrec}


def subset_diff(sc_a, sc_b, y, player, n_players, mask):
    a, b = K.rank_balanced(sc_a[mask], y[mask]), K.rank_balanced(sc_b[mask], y[mask])
    bt = rf.PlayerBootstrap(player[mask], y[mask], n_players)
    d = bt.stats(sc_a[mask]) - bt.stats(sc_b[mask])
    return {"n_questions": int(mask.sum()), "candidate": a, "q09": b, "diff": a - b,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def by_count(inner, cand, base):
    Xe, ms = inner.Xe, inner.ms
    n_players = int(ms["n_players"])
    ye, pe = np.asarray(Xe.y), np.asarray(Xe.player)
    ym, pm = np.asarray(ms["y"]), np.asarray(ms["player"])
    out = {"mono_by_k": {}, "eval_by_n_exact": {}, "eval_by_n_bin": {}}
    for k in K.KS:
        out["mono_by_k"][str(k)] = subset_diff(cand["msc"], base["msc"], ym, pm, n_players, np.asarray(ms["k"]) == k)
        out["eval_by_n_exact"][str(k)] = subset_diff(cand["sc"], base["sc"], ye, pe, n_players, Xe.n == k)
    for lo, hi in rf.N_BINS:
        out["eval_by_n_bin"][f"{lo}-{hi}"] = subset_diff(cand["sc"], base["sc"], ye, pe, n_players,
                                                        (Xe.n >= lo) & (Xe.n <= hi))
    return out


def by_level(inner, cand, base):
    ye, ym = np.asarray(inner.Xe.y), np.asarray(inner.ms["y"])
    a, b = K.per_level(cand["sc"], ye), K.per_level(base["sc"], ye)
    am, bm = K.per_level(cand["msc"], ym), K.per_level(base["msc"], ym)
    return {"eval": [{"level": r.LEVELS[i], "candidate": a[i], "q09": b[i], "diff": a[i] - b[i]} for i in range(K.L)],
            "mono": [{"level": r.LEVELS[i], "candidate": am[i], "q09": bm[i], "diff": am[i] - bm[i]}
                     for i in range(K.L)]}


def w_record(res, term):
    out = []
    for fr in res["folds"]:
        w = fr["params"]["w"][term]
        out.append({"fold": fr["fold"], f"w_{term}": w, "at_bound": bool(w <= 1e-8 or w >= 4.0 - 1e-8)})
    return out


# ---------------------------------------------------------------- eval
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
            raise SystemExit(f"the prereg19 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path)
    base = {"format": FORMAT, "smoke": smoke, "code": cs}
    # ---- seam M (model files = the committed train record; port check passed in training)
    tr = r.read_json(train_path)
    if bool(tr.get("smoke")) != smoke:
        raise SystemExit("train_record.json smoke flag differs from this run")
    par_list, seam_m = [], {"models": {}, "ok": True}
    want_settings = {"steps": K.STEPS, "eval_every": K.EVAL_EVERY, "hold_per": K.HOLD_PER, "seeds": list(K.SEEDS)}
    if smoke:
        want_settings.update(steps=K.SMOKE["steps"], eval_every=K.SMOKE["eval_every"], hold_per=K.SMOKE["hold_per"])
    seam_m["train_settings_registered"] = bool(all(tr["settings"].get(k) == v for k, v in want_settings.items()))
    seam_m["train_code_clean"] = bool((tr.get("code") or {}).get("dirty") == "")
    seam_m["ok"] = bool(seam_m["train_settings_registered"] and (smoke or seam_m["train_code_clean"]))
    for s in K.SEEDS:
        sr = tr["seeds"][str(s)]
        mpath = os.path.join(REPO, *sr["model_file"].split("/"))
        sha = sha_file(mpath)
        par, seed, best_step, msmoke = load_model(mpath)
        ok = (sha == tr["models_sha256"][str(s)] and seed == s and best_step == sr["best_step"] and msmoke == smoke
              and bool((sr.get("port_check") or {}).get("ok")))
        seam_m["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": bool(ok)}
        seam_m["ok"] = bool(seam_m["ok"] and ok)
        par_list.append(K.p64(par))
    # ---- Q09 and seam 1
    prep = ft_eval.InnerPrep(guard_path(INNER_PREP))
    map_sha = r.sha_raw(Q09_MAP)
    with open(Q09_FROZEN, encoding="utf-8") as f:
        if map_sha != json.load(f)["map"]["sha256"]:
            raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    inner = r.Inner(prep, log=log)
    ref = tune_ref()
    mp = FW.load_map(Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    terms9 = list(r.Q08_TERMS)
    q09 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_q09})
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]]}
    seam1 = all(abs(x - y) <= TOL for x, y in seam.values())
    # ---- seam E, P, D
    se = seam_e(inner, lp_q09)
    want = (r.read_json(Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {"probe_X.npy": sha_file(os.path.join(PROBE_PREP, "probe_X.npy")),
           "inner_X.npy": sha_file(os.path.join(INNER_PREP, "inner_X.npy"))}
    seam_p = {"want": {k: want.get(k) for k in got}, "got": got,
              "ok": bool(all(want.get(k) == v for k, v in got.items()))}
    with np.load(guard_path(os.path.join(PROBE_PREP, "probe_meta.npz")), allow_pickle=False) as z:
        pkeys = sorted(set(np.asarray(z["players"]).astype(str).tolist()))
    a_keys = {k for k in pkeys if K.holdout_bit(k) == 0}
    h_keys = {k for k in pkeys if K.holdout_bit(k) == 1}
    players = inner.d["inner"]["players"]
    i_keys = {f"{p['rank']}:{p['player_id']}" for p in players}
    seam_d = {"A_players": len(a_keys), "H_players": len(h_keys), "I_players": len(i_keys),
              "A_and_H": len(a_keys & h_keys), "probe_and_I": len((a_keys | h_keys) & i_keys),
              "probe_and_I_player_id_only_record": len({k.split(":", 1)[1] for k in pkeys}
                                                       & {p["player_id"] for p in players}),
              "train_record_players_match": bool(tr["players"]["A_players"] == len(a_keys)
                                                 and tr["players"]["H_players"] == len(h_keys))}
    seam_d["ok"] = bool(seam_d["A_and_H"] == 0 and seam_d["probe_and_I"] == 0 and seam_d["train_record_players_match"])
    log(f"seam 1 {seam1} {seam}; seam E {se}; seam P {seam_p['ok']}; seam D {seam_d}; seam M {seam_m}")
    if not (seam1 and se["ok"] and seam_p["ok"] and seam_d["ok"] and seam_m["ok"]):
        write_json(out_path, {**base, "status": "stop: seam failed", "seam": seam, "seam_e": se, "seam_p": seam_p,
                              "seam_d": seam_d, "seam_m": seam_m})
        return 1
    # ---- the judged part starts here (after every seam, before any candidate number; PLAN.md 9)
    if not smoke:
        write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                             "rerun_reason": a.reason if a.rerun else None})
    # ---- the SET term
    tc = time.time()
    H = inner_h(prep, inner, mp["mu_x"], mp["sd_x"], par_list, smoke)
    E_set, E_seed, lp_set, has, set_info = set_terms(inner, H, par_list)
    del H
    log(f"SET term built ({time.time() - tc:.0f}s): {set_info}")
    # ---- candidate Q09+SET (judged)
    terms_c = terms9 + ["SET"]
    cand = inner.run(terms_c, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_set})
    cmp_ = inner.compare(cand, q09)
    g = K.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                cmp_["mono_by_colour"])
    log(f"Q09+SET {cand['rb']!r} vs Q09 {q09['rb']!r}: {json.dumps(g)} ({time.time() - tc:.0f}s)")
    # ---- record only
    chk = {"eval_rb_own": K.rank_balanced(cand["sc"], inner.Xe.y), "mono_rb_own": K.rank_balanced(cand["msc"],
                                                                                               inner.ms["y"])}
    chk["ok"] = bool(abs(chk["eval_rb_own"] - cand["rb"]) <= TOL and abs(chk["mono_rb_own"] - cand["mono_rb"]) <= TOL)
    counts = by_count(inner, cand, q09)
    levels = by_level(inner, cand, q09)
    repl = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_set})
    alone_P = {w: rf.softmax(np.where(has[w][:, None], lp_set[w], 0.0)) for w in ("eval", "mono")}
    alone = scored(inner, alone_P["eval"], alone_P["mono"])
    per_seed = {}
    for i, s in enumerate(K.SEEDS):
        rs = inner.run(terms_c, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_seed[i]})
        per_seed[str(s)] = {"inner_rank_balanced": rs["rb"], "single_colour_rank_balanced": rs["mono_rb"],
                            "vs_q09": inner.compare(rs, q09)}
    rec_only = {"replacement_i_SET_in_P1TO_slot": inner.short(repl, q09),
                "SET_alone_rule_b": {"inner_rank_balanced": alone["rb"], "single_colour_rank_balanced": alone["mono_rb"]},
                "per_seed_Q09+SET_s": per_seed, "by_game_count": counts, "by_level": levels,
                "w_SET_by_fold": w_record(cand, "SET"), "own_rank_balanced_check": chk}
    status = "gates passed (worth a PREREG_19; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {**base, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("smoke models (noise-trained, 300 steps) and inner P1 features replaced by normal noise: "
                          "numbers are meaningless" if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: inner-only judgement of Q09+SET (S19 question-level set model trained on probe players, "
                   "4 x lp_SET as a 7th fusion term, Q09's player-fold OOF fusion); training players only, no "
                   "validation scoring (competition 2 has none left)",
           "status": status, "seam": seam, "seam_e": se, "seam_p": seam_p, "seam_d": seam_d, "seam_m": seam_m,
           "set_term": set_info, "train_record_sha256_lf": sha_lf(train_path),
           "candidate": {"name": "Q09+SET", "terms": terms_c, **inner.short(cand, q09)},
           "baseline": {"name": "Q09", **inner.short(q09)}, "gates": g, "record_only": rec_only,
           "q09_map_sha256": map_sha}
    write_json(out_path, rep)
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
