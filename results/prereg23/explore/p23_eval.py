"""prereg23 exploration (competition 2): inner-only judgement of PLAN.md (one candidate Q19+SET2: the second
question-level set model SET2 as an 8th fusion term after Q19 = Q09+SET), judged once. CPU only (.venv; numpy
forward, no torch).

    Git Bash, repository root (a subst drive when the worktree path contains 'AppData\\Local'); PYTHONDONTWRITEBYTECODE=1;
    .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg23/explore
    $S/p23_eval.py eval            -> $S/eval_report.json (judged once; code and train_record.json committed and clean)
    $S/p23_eval.py eval --smoke    -> data/prereg23/smoke/eval_report.json (smoke models, inner SET2 inputs replaced by
                                      default_rng(12345) normal noise: pipeline and timing only, never a result; the
                                      Q19 baseline uses the real features and the frozen S19 models)

Inputs (read only): data/prereg12/q09_inner (inner P1 X / EX features), data/prereg17/prep (probe player keys and
SHAs; seam P/D), the SET2 model files of train_record.json, the frozen S19 models results/prereg19/rank/s19_seed*.npz,
Q09's frozen p1e_map.npz, u1_lib's inner pseudo-questions and per-game terms through tools/p09_rank_fuse.Inner.
Never data/local, never validation features, never tools/score.py.
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
import p23_lib as K  # noqa: E402
import p19_lib as K19  # noqa: E402  (results/prereg19/explore, via p23_lib's sys.path)

R, rf, p3, p4 = r.R, r.rf, r.p3, r.p4
TOL = K.TOL
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_TUNE = os.path.join(REPO, "results", "prereg09", "rank", "tune_report.json")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
Q09_CHECK = os.path.join(REPO, "results", "prereg09", "rank", "check_report.json")
P19_EVAL = os.path.join(REPO, "results", "prereg19", "explore", "eval_report.json")
P19_TRAIN = os.path.join(REPO, "results", "prereg19", "explore", "train_record.json")
S19_DIR = os.path.join(REPO, "results", "prereg19", "rank")
S19_FORMAT = "c10-prereg19 s19 model v1"
S19_SEEDS = (0, 1, 2)
INNER_PREP = os.path.join(REPO, "data", "prereg12", "q09_inner")
PROBE_PREP = os.path.join(REPO, "data", "prereg17", "prep")
DATA = os.path.join(REPO, "data", "prereg23")
SMOKE_DIR = os.path.join(DATA, "smoke")
STARTED = os.path.join(DATA, "eval_started.json")
REPORT = os.path.join(HERE, "eval_report.json")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
OWN_FILES = ("PLAN.md", "p23_lib.py", "p23_train.py", "p23_eval.py", "test_p23.py", "train_record.json")
IMPORTED = ("tools/p02_rank_fuse.py", "tools/p03_rank_fuse.py", "tools/p04_rank_fuse.py", "tools/p07_rank_fuse.py",
            "tools/p08_rank_fuse.py", "tools/p09_rank_fuse.py", "tools/p09_forward.py", "tools/p09_map.py",
            "tools/p08_p1.py", "results/prereg12/explore/finetune/ft_eval.py", "results/prereg05/explore/u1/u1_lib.py",
            "results/prereg07/explore/pilot_humansl/pilot_fuse.py", "results/prereg07/explore/pilot_humansl/hs_lib.py",
            "results/prereg08/p1/p1_fuse.py", "results/prereg08/p1/p1_lib.py",
            "results/prereg09/rank/p1e_map.npz", "results/prereg09/rank/check_report.json",
            "results/prereg09/rank/tune_report.json", "results/prereg09/rank/frozen_config.json",
            "results/prereg08/rank/p1_map.npz", "results/prereg19/explore/p19_lib.py",
            "results/prereg19/explore/eval_report.json", "results/prereg19/explore/train_record.json",
            "results/prereg19/rank/s19_seed0.npz", "results/prereg19/rank/s19_seed1.npz",
            "results/prereg19/rank/s19_seed2.npz")
# the files of the SET2 forward / features: identical (LF SHA) in training (port check) and here (seam M)
FORWARD_FILES = ("results/prereg23/explore/p23_lib.py", "results/prereg23/explore/p23_train.py",
                 "results/prereg19/explore/p19_lib.py", "tools/p09_forward.py")
FORMAT = "c10-prereg23 explore eval v1"
MODEL_FORMAT = "c10-prereg23 set2 model v1"
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


def refs():
    with open(Q09_TUNE, encoding="utf-8") as f:
        c = json.load(f)["combinations"]["Q09"]
    with open(P19_EVAL, encoding="utf-8") as f:
        e = json.load(f)
    if e.get("smoke") or not e["gates"]["passed"] or e["candidate"]["name"] != "Q09+SET":
        raise SystemExit("results/prereg19/explore/eval_report.json is not the judged Q09+SET report")
    return {"Q09": c["oof"]["rank_balanced"], "Q09_mono": c["single_colour"]["rank_balanced"],
            "Q19": e["candidate"]["inner_rank_balanced"], "Q19_mono": e["candidate"]["single_colour_rank_balanced"]}


def load_npz_model(path, fmt):
    with np.load(guard_path(path), allow_pickle=False) as z:
        if str(z["format"]) != fmt:
            raise SystemExit(f"{path}: not a {fmt} file")
        return {k: np.asarray(z[f"p_{k}"]) for k in K.PARAM_NAMES}, int(z["seed"]), int(z["best_step"]), \
            bool(z["smoke"])


# ---------------------------------------------------------------- the frozen SET term of Q19 (prereg19, seam F)
def s19_models():
    tr = r.read_json(P19_TRAIN)
    pars, seam = [], {"models": {}, "ok": True}
    for s in S19_SEEDS:
        path = os.path.join(S19_DIR, f"s19_seed{s}.npz")
        sha = sha_file(path)
        par, seed, best_step, msmoke = load_npz_model(path, S19_FORMAT)
        ok = bool(sha == tr["models_sha256"][str(s)] and seed == s and not msmoke
                  and best_step == tr["seeds"][str(s)]["best_step"])
        seam["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": ok}
        seam["ok"] = bool(seam["ok"] and ok)
        pars.append(K19.p64(par))
    return pars, seam


def s19_term(prep, inner, mp, pars):
    """= p19_eval.inner_h + set_terms (real features, frozen S19 models): E_SET (x 4) for fit / eval / mono."""
    H = [np.zeros((inner.n_inner, K19.H_DIM)) for _ in pars]
    n = len(prep.inner_ref)
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        X = FW.standardise_x(np.asarray(prep.inner_X[s:e], dtype=np.float64), mp["mu_x"], mp["sd_x"])
        rows = prep.inner_ref[s:e]
        for par, Hs in zip(pars, H):
            Hs[rows] = K19.h_np(par, X)
    di = inner.d["inner"]
    E, lp, has = {}, {}, {}
    for which, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"])):
        groups = K19.usable_groups(S, inner.covered)
        E[which], _per, lp[which], has[which] = K19.set_term(pars, H, groups, inner.n_inner)
    return E, lp, has


# ---------------------------------------------------------------- the SET2 term (PLAN.md 3.6, 4)
def inner_h(prep, inner, mp, par_list, smoke):
    """Per-game h of every SET2 seed on the full inner row space (rows without P1 features stay 0, never pooled)."""
    n = len(prep.inner_ref)
    H = [np.zeros((inner.n_inner, K.H_DIM)) for _ in par_list]
    ex_info = {"raw_nan": 0, "raw_inf": 0}
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        raw_ex = np.asarray(prep.inner_ex[s:e])
        ex_info["raw_nan"] += int(np.isnan(raw_ex).sum())
        ex_info["raw_inf"] += int(np.isinf(raw_ex).sum())
        if smoke:
            X = np.random.default_rng([NOISE_SEED, 1, s]).standard_normal((e - s, K.D_IN)).astype(np.float32)
            X = X.astype(np.float64)
        else:
            X = K.features(prep.inner_X[s:e], raw_ex, mp)
        if ex_info["raw_inf"]:
            raise SystemExit(f"raw inner EX has inf values {ex_info}: standardise_ex would map them to +-1.8e308")
        if not np.isfinite(X).all():
            raise SystemExit("non-finite standardised inner features")
        rows = prep.inner_ref[s:e]
        for par, Hs in zip(par_list, H):
            Hs[rows] = K.h_np(par, X)
    return H, ex_info


def set2_terms(inner, H, par_list):
    """E_SET2 (x 4) and per-seed E for fit / eval / mono; usable = Inner.covered (= Q09's P1TO mask)."""
    di = inner.d["inner"]
    E, E_seed, lp, has, info = {}, [dict() for _ in par_list], {}, {}, {}
    for which, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"])):
        groups = K.usable_groups(S, inner.covered)
        E[which], per, lp[which], has[which] = K.set_term(par_list, H, groups)
        for i, x in enumerate(per):
            E_seed[i][which] = x
        sizes = np.array([len(g) for g in groups])
        info[which] = {"questions": int(len(groups)), "without_usable_game": int((sizes == 0).sum()),
                       "with_one_usable_game": int((sizes == 1).sum())}
    return E, E_seed, lp, has, info


SEAM_STD_QUESTIONS = 2000


def seam_e(inner, lp_q09):
    """Own [mean, std] pooling on 4 x lp_P1TO_E: the mean half vs Inner.E_p1 (all questions), the std half vs a direct
    per-question np.std(ddof=0) on the first SEAM_STD_QUESTIONS questions of each set."""
    E_ref, _ = inner.E_p1(lp_q09)
    lp4 = K.SCALE * inner.lp_full(lp_q09)
    di = inner.d["inner"]
    out = {}
    for which, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]), ("mono", di["S_mono"])):
        g = K.usable_groups(S, inner.covered)
        own = K.pool_mean_std(g, lp4)
        out[which] = float(np.max(np.abs(own[:, :K.L] - E_ref[which])))
        direct = np.array([lp4[gq].std(axis=0) if len(gq) else np.zeros(K.L) for gq in g[:SEAM_STD_QUESTIONS]])
        out[which + "_std"] = float(np.max(np.abs(own[:SEAM_STD_QUESTIONS, K.L:] - direct)))
    out["ok"] = bool(all(v <= TOL for v in out.values()))
    return out


# ---------------------------------------------------------------- records
def scored(inner, P, Pm):
    m, sc = rf.metrics(rf.rule_b_np(P), inner.Xe.y, inner.Xe.n, inner.Xe.player)
    mrec, msc = p3.mono_score_record(rf.rule_b_np(Pm), inner.Xm, inner.ms, inner.mboot)
    return {"sc": sc, "msc": msc, "rb": m["rank_balanced"], "mono_rb": mrec["rank_balanced"], "m": m, "mrec": mrec}


def alone(inner, lp, has):
    P = {w: rf.softmax(np.where(has[w][:, None], lp[w], 0.0)) for w in ("eval", "mono")}
    s = scored(inner, P["eval"], P["mono"])
    return {"inner_rank_balanced": s["rb"], "single_colour_rank_balanced": s["mono_rb"]}


def subset_diff(sc_a, sc_b, y, player, n_players, mask):
    a, b = K.rank_balanced(sc_a[mask], y[mask]), K.rank_balanced(sc_b[mask], y[mask])
    bt = rf.PlayerBootstrap(player[mask], y[mask], n_players)
    d = bt.stats(sc_a[mask]) - bt.stats(sc_b[mask])
    return {"n_questions": int(mask.sum()), "candidate": a, "baseline": b, "diff": a - b,
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
    return {"eval": [{"level": r.LEVELS[i], "candidate": a[i], "baseline": b[i], "diff": a[i] - b[i]}
                     for i in range(K.L)],
            "mono": [{"level": r.LEVELS[i], "candidate": am[i], "baseline": bm[i], "diff": am[i] - bm[i]}
                     for i in range(K.L)]}


def w_record(res, terms):
    out = []
    for fr in res["folds"]:
        row = {"fold": fr["fold"]}
        for t in terms:
            w = fr["params"]["w"][t]
            row[f"w_{t}"] = w
            row[f"{t}_at_bound"] = bool(w <= 1e-8 or w >= 4.0 - 1e-8)
        out.append(row)
    return out


def peak_memory():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"peak_working_set_gb": round(getattr(mi, "peak_wset", 0) / 2 ** 30, 2),
                "peak_pagefile_gb": round(getattr(mi, "peak_pagefile", 0) / 2 ** 30, 2)}
    except Exception as e:  # record only
        return {"error": repr(e)}


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
            raise SystemExit(f"the prereg23 files or the imported tools are not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, out_path)
    base = {"format": FORMAT, "smoke": smoke, "code": cs}
    # ---- seam M (SET2 model files = the committed train record; port check passed in training)
    tr = r.read_json(train_path)
    if bool(tr.get("smoke")) != smoke:
        raise SystemExit("train_record.json smoke flag differs from this run")
    par_list, seam_m = [], {"models": {}, "ok": True}
    want_settings = {"steps": K.STEPS, "eval_every": K.EVAL_EVERY, "hold_per": K.HOLD_PER, "seeds": list(K.SEEDS),
                     "batch": K.BATCH, "ks": list(K.KS), "lr": K.LR, "wd": K.WD, "pct_start": K.PCT_START,
                     "d_in": K.D_IN, "hidden": K.HIDDEN, "h_dim": K.H_DIM, "pool_dim": K.POOL_DIM, "rho": K.RHO,
                     "drop_in": K.DROP_IN, "drop": K.DROP, "threads": K.THREADS, "hold_seed": K.HOLD_SEED,
                     "sampler_base": K.SAMPLER_BASE, "salt": K.SALT}
    if smoke:
        want_settings.update(steps=K.SMOKE["steps"], eval_every=K.SMOKE["eval_every"], hold_per=K.SMOKE["hold_per"])
    seam_m["train_settings_registered"] = bool(all(tr["settings"].get(k) == v for k, v in want_settings.items()))
    seam_m["train_code_clean"] = bool((tr.get("code") or {}).get("dirty") == "")
    tr_sha = (tr.get("code") or {}).get("sha256_lf") or {}
    same = {f: bool(tr_sha.get(f) is not None and tr_sha.get(f) == cs["sha256_lf"].get(f)) for f in FORWARD_FILES}
    seam_m["forward_code_same_as_train"] = same
    seam_m["ok"] = bool(seam_m["train_settings_registered"] and (smoke or seam_m["train_code_clean"])
                        and all(same.values()))
    for s in K.SEEDS:
        sr = tr["seeds"][str(s)]
        mpath = os.path.join(REPO, *sr["model_file"].split("/"))
        sha = sha_file(mpath)
        par, seed, best_step, msmoke = load_npz_model(mpath, MODEL_FORMAT)
        ok = (sha == tr["models_sha256"][str(s)] and seed == s and best_step == sr["best_step"] and msmoke == smoke
              and bool((sr.get("port_check") or {}).get("ok")))
        seam_m["models"][str(s)] = {"sha256": sha, "best_step": best_step, "ok": bool(ok)}
        seam_m["ok"] = bool(seam_m["ok"] and ok)
        par_list.append(K.p64(par))
    s19_pars, seam_f = s19_models()
    # ---- Q09, Q19 and seam 1
    prep = ft_eval.InnerPrep(guard_path(INNER_PREP))
    map_sha = r.sha_raw(Q09_MAP)
    with open(Q09_FROZEN, encoding="utf-8") as f:
        if map_sha != json.load(f)["map"]["sha256"]:
            raise SystemExit("p1e_map.npz differs from results/prereg09/rank/frozen_config.json")
    inner = r.Inner(prep, log=log)
    ref = refs()
    mp = FW.load_map(Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    terms9 = list(r.Q08_TERMS)
    q09 = inner.run(terms9, {"HS": inner.E_hs, "P1TO": E_q09})
    tq = time.time()
    E_set, lp_set, has_set = s19_term(prep, inner, mp, s19_pars)
    terms19 = terms9 + ["SET"]
    q19 = inner.run(terms19, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_set})
    log(f"Q19 baseline rebuilt ({time.time() - tq:.0f}s)")
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]], "Q19": [q19["rb"], ref["Q19"]],
            "Q19_mono": [q19["mono_rb"], ref["Q19_mono"]]}
    seam1 = all(abs(x - y) <= TOL for x, y in seam.values())
    # ---- seam E, P, D
    se = seam_e(inner, lp_q09)
    want = (r.read_json(Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    got = {"probe_X.npy": sha_file(os.path.join(PROBE_PREP, "probe_X.npy")),
           "probe_ex.npy": sha_file(os.path.join(PROBE_PREP, "probe_ex.npy")),
           "inner_X.npy": sha_file(os.path.join(INNER_PREP, "inner_X.npy")),
           "inner_ex.npy": sha_file(os.path.join(INNER_PREP, "inner_ex.npy"))}
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
    log(f"seam 1 {seam1} {seam}; seam E {se}; seam P {seam_p['ok']}; seam D {seam_d}; seam M {seam_m}; "
        f"seam F {seam_f}")
    if not (seam1 and se["ok"] and seam_p["ok"] and seam_d["ok"] and seam_m["ok"] and seam_f["ok"]):
        write_json(out_path, {**base, "status": "stop: seam failed", "seam": seam, "seam_e": se, "seam_p": seam_p,
                              "seam_d": seam_d, "seam_m": seam_m, "seam_f": seam_f})
        return 1
    # ---- the judged part starts here (after every seam, before any candidate number)
    if not smoke:
        write_json(STARTED, {"format": FORMAT + " started", "created": r.now_iso(), "code": cs,
                             "rerun_reason": a.reason if a.rerun else None})
    # ---- the SET2 term
    tc = time.time()
    H, ex_info = inner_h(prep, inner, mp, par_list, smoke)
    E_set2, E_seed, lp_set2, has2, set2_info = set2_terms(inner, H, par_list)
    del H
    log(f"SET2 term built ({time.time() - tc:.0f}s): {set2_info}; EX raw {ex_info}")
    # ---- candidate Q19+SET2 (judged)
    terms_c = terms19 + ["SET2"]
    Emap_c = {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_set, "SET2": E_set2}
    cand = inner.run(terms_c, Emap_c)
    cmp_ = inner.compare(cand, q19)
    g = K.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                cmp_["mono_by_colour"])
    log(f"Q19+SET2 {cand['rb']!r} vs Q19 {q19['rb']!r}: {json.dumps(g)} ({time.time() - tc:.0f}s)")
    # ---- record only
    chk = {"eval_rb_own": K.rank_balanced(cand["sc"], inner.Xe.y),
           "mono_rb_own": K.rank_balanced(cand["msc"], inner.ms["y"])}
    chk["ok"] = bool(abs(chk["eval_rb_own"] - cand["rb"]) <= TOL and abs(chk["mono_rb_own"] - cand["mono_rb"]) <= TOL)
    counts = by_count(inner, cand, q19)
    levels = by_level(inner, cand, q19)
    repl = inner.run(terms9 + ["SET2"], {"HS": inner.E_hs, "P1TO": E_q09, "SET2": E_set2})
    repl_rec = {**inner.short(repl, q19), "vs_q09": inner.compare(repl, q09)}
    del repl
    per_seed = {}
    for i, s in enumerate(K.SEEDS):
        rs = inner.run(terms_c, {**Emap_c, "SET2": E_seed[i]})
        per_seed[str(s)] = {"inner_rank_balanced": rs["rb"], "single_colour_rank_balanced": rs["mono_rb"],
                            "vs_q19": inner.compare(rs, q19)}
        del rs  # memory: only the summary is kept (the machine is shared)
    rec_only = {"Q09+SET2_replacing_SET": repl_rec,
                "SET2_alone_rule_b": alone(inner, lp_set2, has2),
                "SET_alone_rule_b": alone(inner, lp_set, has_set),
                "per_seed_Q19+SET2_s": per_seed, "by_game_count": counts, "by_level": levels,
                "weights_by_fold": w_record(cand, ["P1TO", "SET", "SET2"]),
                "q19_weights_by_fold": w_record(q19, ["P1TO", "SET"]),
                "own_rank_balanced_check": chk, "ex_raw_nonfinite_inner": ex_info}
    status = "gates passed (worth a PREREG_23; not adoption)" if g["passed"] else "gates not passed: stop"
    rep = {**base, "created": r.now_iso(), "seconds": round(time.time() - t0, 1),
           "smoke_note": ("smoke SET2 models (noise-trained, 300 steps) and inner SET2 inputs replaced by normal noise: "
                          "numbers are meaningless" if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: inner-only judgement of Q19+SET2 (SET2 = second question-level set model on [X, EX] with "
                   "[mean, std] pooling and 5 seeds, trained on probe players; 4 x lp_SET2 as an 8th fusion term after "
                   "Q19 = Q09+SET; Q09's player-fold OOF fusion); training players only, no validation scoring",
           "status": status, "seam": seam, "seam_e": se, "seam_p": seam_p, "seam_d": seam_d, "seam_m": seam_m,
           "seam_f": seam_f, "set2_term": set2_info, "train_record_sha256_lf": sha_lf(train_path),
           "candidate": {"name": "Q19+SET2", "terms": terms_c, **inner.short(cand, q19)},
           "baseline": {"name": "Q19", "terms": terms19, **inner.short(q19, q09)}, "gates": g,
           "record_only": rec_only, "q09_map_sha256": map_sha, "peak_memory_record": peak_memory()}
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
