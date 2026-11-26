"""prereg44 explore (competition 2): select the CNN44 version on H (PLAN.md 4), stage 0b on S25's held-out questions
(PLAN.md 5), the inner judgement (PLAN.md 6). CPU only (.venv; numpy / scipy).

    Git Bash, repository root on the subst drive; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP /
    OPENBLAS / MKL variable;  S=results/prereg44/explore
    $S/p44_eval.py select     -> $S/select.json   (H game logits of the 8 candidates; train_record.json committed)
    $S/p44_eval.py stage0b    -> $S/stage0b.json  (select.json committed; judged once)
    $S/p44_eval.py eval       -> $S/eval_report.json (stage0b.json committed and passed; judged once)
    --smoke: select over whatever smoke logits exist (data/prereg44/smoke/logits), stage0b / eval with the CNN44
             term replaced by default_rng(12345) noise -> data/prereg44/smoke/*.json (pipeline and timing only)

Never the validation answers, never validation features, never tools/score.py; no inner logits are read before
select.json and stage0b.json are committed.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P23 = os.path.join(REPO, "results", "prereg23", "explore")
P25 = os.path.join(REPO, "results", "prereg25", "explore")
P40 = os.path.join(REPO, "results", "prereg40", "explore")
P41 = os.path.join(REPO, "results", "prereg41", "explore")
for p in (os.path.join(REPO, "tools"), P23, P25, P40, P41, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p44_lib as K  # noqa: E402

DATA = K.DATA
SMOKE_DIR = os.path.join(DATA, "smoke")
SELECT = os.path.join(HERE, "select.json")
STAGE0B = os.path.join(HERE, "stage0b.json")
REPORT = os.path.join(HERE, "eval_report.json")
TRAIN_RECORD = os.path.join(HERE, "train_record.json")
DATA_RECORD = os.path.join(HERE, "data_record.json")
STARTED_0B = os.path.join(DATA, "stage0b_started.json")
STARTED = os.path.join(DATA, "eval_started.json")
HOLD_NPZ = os.path.join(DATA, "hold", "hold.npz")
OWN = ("PLAN.md", "p44_lib.py", "p44_data.py", "p44_train.py", "p44_infer.py", "p44_eval.py", "test_p44.py",
       "data_record.json")
NOISE_SEED = 12345
TERM = "CNN44"


def log(*x):
    print(*x, flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def loaded_repo_modules():
    root = os.path.join(os.path.abspath(REPO), "").lower()
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f).lower().startswith(root):
            out.add(os.path.relpath(os.path.abspath(f), REPO).replace("\\", "/"))
    return out


def code_state(extra=()):
    rel = [os.path.relpath(os.path.join(HERE, f), REPO).replace("\\", "/") for f in OWN + tuple(extra)]
    rel = sorted(set(rel) | loaded_repo_modules())
    missing = [x for x in rel if not os.path.exists(os.path.join(REPO, *x.split("/")))]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    return {"head": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain", "--", *rel),
            "sha256_lf": {x: K.sha_lf(os.path.join(REPO, *x.split("/"))) for x in rel}}


def committed_clean(path):
    rel = os.path.relpath(path, REPO).replace("\\", "/")
    return bool(git("ls-files", "--", rel)) and not git("status", "--porcelain", "--", rel)


def once_guard(a, cs, paths):
    prior = [p for p in paths if os.path.exists(p)]
    if not prior:
        return None
    if not (a.rerun and a.reason):
        raise SystemExit(f"{prior} exist: judged once (--rerun --reason TEXT only after a commit that names a "
                         f"concrete program error)")
    heads = {(K.read_json(p).get("code") or {}).get("head") for p in prior}
    if cs["head"] in heads:
        raise SystemExit("--rerun needs a newer commit than the earlier run's (the fix of the named error)")
    return sorted(h for h in heads if h)


def logits_root(smoke):
    return os.path.join(SMOKE_DIR, "logits") if smoke else K.LOGITS


def load_logits(path, want_label, cache_dir):
    with np.load(K.guard(path), allow_pickle=False) as z:
        rec = {k: np.asarray(z[k]) for k in z.files}
    cm = K.read_json(os.path.join(cache_dir, "cache.json"))
    if str(rec["cache_label"]) != want_label or str(rec["cache_codes_sha256"]) != cm["codes_sha256"]:
        raise SystemExit(f"{path}: not the logits of the {want_label} cache {cm['codes_sha256'][:12]}")
    return rec


def tag_of(step, w):
    return "s%06d_%s" % (step, w)


# ================================================================ select (PLAN.md 4)
def cmd_select(a):
    smoke = bool(a.smoke)
    out = os.path.join(SMOKE_DIR, "select.json") if smoke else SELECT
    if not smoke:
        if not committed_clean(TRAIN_RECORD):
            raise SystemExit("train_record.json is not committed and clean")
        tr = K.read_json(TRAIN_RECORD)
        if tr.get("smoke") or not tr.get("finished"):
            raise SystemExit("train_record.json is not a finished real run")
        if os.path.exists(SELECT):
            raise SystemExit("select.json exists (chosen once)")
    cs = code_state()
    if not smoke and cs["dirty"]:
        raise SystemExit(f"prereg44 files not committed and clean: {cs['dirty']}")
    hold_dir = os.path.join(K.CACHE, "hold")
    cands = [(s, w) for s in K.CANDIDATE_STEPS for w in K.CANDIDATE_WEIGHTS]
    root = logits_root(smoke)
    if smoke:
        cands = []
        for t in sorted(os.listdir(root)):
            if os.path.exists(os.path.join(root, t, "hold.npz")):
                cands.append((int(t[1:7]), t.split("_", 1)[1]))
    res = {}
    for s, w in cands:
        rec = load_logits(os.path.join(root, tag_of(s, w), "hold.npz"), "hold", hold_dir)
        if not smoke and (int(rec["limit"]) != -1 or int(rec["step"]) != s or str(rec["weights"]) != w):
            raise SystemExit(f"{tag_of(s, w)}: wrong step / weights / limit")
        y = np.array([K.LEVELS.index(r) for r in rec["rank"].astype(str)], np.int64)
        ok = ~rec["bad"].astype(bool)
        lp = K.log_softmax(rec["logits"][ok])
        res[tag_of(s, w)] = {"balanced_game_nll": K.balanced_game_nll(lp, y[ok]), "games": int(ok.sum()),
                             "bad_games": int((~ok).sum()), "ckpt_sha256": str(rec["ckpt_sha256"])}
    if not res:
        raise SystemExit("no candidate logits")
    best = min(res, key=lambda t: (res[t]["balanced_game_nll"], list(res).index(t)))
    rep = {"format": "c10-prereg44 select v1", "smoke": smoke, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "code": cs, "what": "PLAN.md 4: level-balanced mean per-game NLL on the H held-out question games; "
                               "lowest of the 8 candidates (no inner logits read)",
           "candidates": res, "selected": best}
    K.write_json(out, rep)
    log(json.dumps({"selected": best, "nll": res[best]["balanced_game_nll"]}))
    return 0


# ================================================================ stage 0b (PLAN.md 5)
def hold_questions():
    with np.load(K.guard(HOLD_NPZ), allow_pickle=False) as z:
        return {int(n): {k: np.asarray(z[f"{k}_{n}"]) for k in ("idx", "y", "player", "colour")}
                for n in z["ks"].tolist()}


def s25_hold_lp(smoke):
    """S25's lp on the same held-out questions (prereg40's stage 0b code path), and the seams."""
    import p40_lib as L40
    import p23_train as T23
    import p19_lib as K19
    import p09_forward as FW
    import p41_eval as E41
    s25, sm = E41.s25_models()
    mp = FW.load_map(os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz"))
    prep = T23.ProbeOnly(L40.PROBE_PREP)
    rows = L40.Rows(T23.probe_colours(prep))
    del prep
    _ra, _rh, _pa, pool_h = L40.split_pools(rows)
    hold, stats = K19.holdout_set(pool_h, L40.HOLD_PER, L40.HOLD_SEED)

    def feats(idx):
        Xr, EXr = rows.raw(idx)
        return L40.features32(Xr, EXr, mp)[:, :FW.X_DIM]

    lp = L40.holdout_lp(K19.set_logp_np, s25, hold, feats)
    out = {}
    for n, (idx, y) in hold.items():
        out[n] = {"lp": FW.lse_mean(lp[n]), "y": np.asarray(y), "players": rows.keys[idx[:, 0]].astype(str),
                  "colours": rows.col[idx[:, 0]].astype(str)}
    return out, sm, stats


def cmd_stage0b(a):
    t0 = time.time()
    smoke = bool(a.smoke)
    out = os.path.join(SMOKE_DIR, "stage0b.json") if smoke else STAGE0B
    sel_path = os.path.join(SMOKE_DIR, "select.json") if smoke else SELECT
    if not smoke and not committed_clean(SELECT):
        raise SystemExit("select.json is not committed and clean")
    cs = code_state(() if smoke else ("select.json", "train_record.json"))
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"prereg44 files not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, [out, STARTED_0B])
    sel = K.read_json(sel_path)["selected"]
    hq = hold_questions()
    s25, sm, stats = s25_hold_lp(smoke)
    seam = {"s25_models": sm["ok"], "questions_equal": True, "sampling": stats}
    for n in hq:
        h, s = hq[n], s25[n]
        if not (np.array_equal(h["y"], s["y"]) and np.array_equal(h["player"].astype(str), s["players"])
                and np.array_equal(h["colour"].astype(str), s["colours"])):
            seam["questions_equal"] = False
    seam["ok"] = bool(seam["s25_models"] and seam["questions_equal"])
    if not seam["ok"]:
        K.write_json(out, {"format": "c10-prereg44 stage0b v1", "smoke": smoke, "code": cs,
                           "status": "stop: seam failed", "seam": seam})
        return 1
    if not smoke:
        K.write_json(STARTED_0B, {"format": "c10-prereg44 stage0b started", "code": cs, "selected": sel,
                                  "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    hold_dir = os.path.join(K.CACHE, "hold")
    if smoke:
        rng = np.random.default_rng(NOISE_SEED)
        lpg = K.log_softmax(rng.standard_normal((K.read_json(os.path.join(hold_dir, "cache.json"))["games"], K.L)))
    else:
        rec = load_logits(os.path.join(K.LOGITS, sel, "hold.npz"), "hold", hold_dir)
        if rec["bad"].any():
            raise SystemExit("held-out games without a logit")
        lpg = K.log_softmax(rec["logits"])
    ns = sorted(hq)
    y = np.concatenate([hq[n]["y"] for n in ns])
    nn = np.concatenate([np.full(len(hq[n]["y"]), n, np.float64) for n in ns])
    players = np.concatenate([hq[n]["player"].astype(str) for n in ns])
    lp25 = np.concatenate([s25[n]["lp"] for n in ns])
    E = np.concatenate([lpg[hq[n]["idx"]].mean(axis=1) for n in ns])
    sn = (nn / 12.0)[:, None]
    fold = K.player_fold(players)
    base_lp, base_par = K.crossfit_stack([lp25], y, fold)
    cand_lp, cand_par = K.crossfit_stack([lp25, E, sn * E], y, fold)
    alone_lp, alone_par = K.crossfit_stack([E, sn * E], y, fold)
    nb, nc = K.mean_nll(base_lp, y), K.mean_nll(cand_lp, y)
    d = nc - nb
    verdict = {"delta": d, "threshold": K.GATE_0B, "passed": bool(d <= K.GATE_0B)}

    import p02_rank_fuse as rf
    import p19_lib as K19

    def score(lp):
        return K19.rank_balanced(K19.official_score(rf.rule_b_np(np.exp(lp)), y), y)

    per_n = {str(n): {"base": K.mean_nll(base_lp[nn == n], y[nn == n]),
                      "cand": K.mean_nll(cand_lp[nn == n], y[nn == n])} for n in ns}
    for v in per_n.values():
        v["delta"] = v["cand"] - v["base"]
    rep = {"format": "c10-prereg44 stage0b v1", "smoke": smoke, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "seconds": round(time.time() - t0, 1), "code": cs, "earlier_run_heads": earlier,
           "rerun_reason": a.reason if a.rerun else None, "selected": sel,
           "smoke_note": "CNN44 game log-probs replaced by noise: numbers meaningless" if smoke else None,
           "what": "PLAN.md 5: 2-fold player cross-fitted stacks on S25's 4,000 held-out questions; base z = a lp_S25 "
                   "+ b, cand z = a lp_S25 + c E + d (n/12) E + b; no inner data",
           "seam": seam, "nll": {"base": nb, "cand": nc}, "verdict": verdict,
           "status": ("passed: run the inner judgement (PLAN.md 6)" if verdict["passed"]
                      else "not passed: stop (no inner use)"),
           "record_only": {"per_n": per_n, "CNN44_alone_nll": K.mean_nll(alone_lp, y),
                           "CNN44_alone_rule_b_score": score(alone_lp), "S25_base_rule_b_score": score(base_lp),
                           "cand_rule_b_score": score(cand_lp), "params": {"base": base_par, "cand": cand_par,
                                                                           "alone": alone_par},
                           "fold_sizes": [int((fold == f).sum()) for f in (0, 1)]}}
    K.write_json(out, rep)
    log(json.dumps({"nll": rep["nll"], "verdict": verdict}))
    return 0


# ================================================================ inner judgement (PLAN.md 6)
def cmd_eval(a):
    t0 = time.time()
    smoke = bool(a.smoke)
    out_path = os.path.join(SMOKE_DIR, "eval_report.json") if smoke else REPORT
    if not smoke:
        for p in (SELECT, STAGE0B):
            if not committed_clean(p):
                raise SystemExit(f"{os.path.basename(p)} is not committed and clean")
        s0 = K.read_json(STAGE0B)
        if s0.get("smoke") or not s0["verdict"]["passed"]:
            raise SystemExit("stage 0b did not pass: no inner judgement (PLAN.md 5)")
    import p41_eval as E41
    E23, r = E41.E23, E41.r
    import p23_lib as K23
    rf, ft_eval = r.rf, E41.ft_eval
    cs = code_state(() if smoke else ("select.json", "stage0b.json", "train_record.json"))
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"prereg44 files or imported tools not committed and clean: {cs['dirty']}")
        earlier = once_guard(a, cs, [out_path, STARTED])
    sel = K.read_json(os.path.join(SMOKE_DIR, "select.json") if smoke else SELECT)["selected"]
    s25, sm = E41.s25_models()
    prep = ft_eval.InnerPrep(E23.guard_path(E23.INNER_PREP))
    want = (r.read_json(E23.Q09_CHECK).get("prep") or {}).get("files_sha256") or {}
    sm["inner_X_sha_equal_check"] = bool(E23.sha_file(os.path.join(E23.INNER_PREP, "inner_X.npy"))
                                         == want.get("inner_X.npy"))
    sm["ok"] = bool(sm["ok"] and sm["inner_X_sha_equal_check"])
    inner = r.Inner(prep, log=log)
    ref = E23.refs()
    mp = E41.FW.load_map(E23.Q09_MAP)
    lp_q09 = r.inner_map_lp(prep, mp)
    E_q09, _ = inner.E_p1(lp_q09)
    terms7 = list(r.Q08_TERMS) + ["SET"]
    q09 = inner.run(list(r.Q08_TERMS), {"HS": inner.E_hs, "P1TO": E_q09})
    s19_pars, _ = E23.s19_models()
    E_s19, _a, _b = E23.s19_term(prep, inner, mp, s19_pars)
    q19 = inner.run(terms7, {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s19})
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]], "Q19": [q19["rb"], ref["Q19"]],
            "Q19_mono": [q19["mono_rb"], ref["Q19_mono"]]}
    del q09, q19, E_s19
    E_s25, _c, _d = E23.s19_term(prep, inner, mp, s25)
    Emap = {"HS": inner.E_hs, "P1TO": E_q09, "SET": E_s25}
    q25 = inner.run(terms7, Emap)
    seam["Q25"] = [q25["rb"], K.REF_Q25]
    seam["Q25_mono"] = [q25["mono_rb"], K.REF_Q25_MONO]
    rep25 = r.read_json(E41.S25_EVAL_REPORT)
    bound = bool(rep25["candidate"]["inner_rank_balanced"] == K.REF_Q25
                 and rep25["candidate"]["single_colour_rank_balanced"] == K.REF_Q25_MONO)
    seam1 = bool(bound and all(abs(x - y) <= E41.TOL for x, y in seam.values()))
    seam1_bitwise = all(x == y for x, y in seam.values())
    # seam K: the selected inner logits in inner_keys order
    seam_k = {"ok": True}
    n_inner = inner.n_inner
    if smoke:
        lp44 = K.log_softmax(np.random.default_rng(NOISE_SEED).standard_normal((n_inner, K.L)))
        seam_k["note"] = "smoke: noise"
    log(f"seam 1 {seam1} (bitwise {seam1_bitwise}) {seam}; seam M {sm['ok']}")
    if not smoke:
        if not (seam1 and sm["ok"]):
            E23.write_json(out_path, {"format": "c10-prereg44 eval v1", "smoke": smoke, "code": cs,
                                      "status": "stop: seam failed", "seam": seam, "seam_m": sm})
            return 1
        E23.write_json(STARTED, {"format": "c10-prereg44 eval started", "created": r.now_iso(), "code": cs,
                                 "selected": sel, "rerun_reason": a.reason if a.rerun else None})
        inner_dir = os.path.join(K.CACHE, "inner")
        rec = load_logits(os.path.join(K.LOGITS, sel, "inner.npz"), "inner", inner_dir)
        with np.load(K.guard(K.INNER_KEYS), allow_pickle=False) as z:
            ik = [np.asarray(z[k]).astype(str) for k in ("rank", "game_id", "player_id", "color")]
        same = all(np.array_equal(rec[f].astype(str), v) for f, v in zip(("rank", "game_id", "player_id", "color"), ik))
        seam_k = {"keys_equal_inner_keys": bool(same), "rows": int(len(rec["logits"])), "n_inner": int(n_inner),
                  "bad": int(rec["bad"].sum()), "step": int(rec["step"]), "weights": str(rec["weights"]),
                  "tag_equal_select": bool(tag_of(int(rec["step"]), str(rec["weights"])) == sel),
                  "limit": int(rec["limit"]), "ckpt_sha256": str(rec["ckpt_sha256"])}
        seam_k["ok"] = bool(same and seam_k["rows"] == n_inner and seam_k["bad"] == 0 and seam_k["tag_equal_select"]
                            and seam_k["limit"] == -1)
        if not seam_k["ok"]:
            E23.write_json(out_path, {"format": "c10-prereg44 eval v1", "smoke": smoke, "code": cs,
                                      "status": "stop: seam K failed", "seam_k": seam_k})
            return 1
        lp44 = K.log_softmax(rec["logits"])
    di = inner.d["inner"]
    E44 = {w: rf.pooled(S, lp44, "log") for w, S in (("fit", di["S_fit"]), ("eval", di["S_eval"]),
                                                       ("mono", di["S_mono"]))}
    terms8 = terms7 + [TERM]
    tc = time.time()
    cand = inner.run(terms8, {**Emap, TERM: E44})
    cmp_ = inner.compare(cand, q25)
    gates = K23.gates(cmp_["inner_diff"], cmp_["inner_ci95"], cmp_["mono_diff"], cmp_["mono_ci95"],
                      cmp_["mono_by_colour"])
    log(f"Q44 {cand['rb']!r} vs Q25 {q25['rb']!r}: {json.dumps(gates)} ({time.time() - tc:.0f}s)")
    has = {w: np.ones(len(E44[w]), bool) for w in ("eval", "mono")}
    alone = E23.alone(inner, {w: E44[w] for w in ("eval", "mono")}, has)
    status = "gates passed (worth a PREREG_44; not adoption)" if gates["passed"] else "gates not passed: stop"
    rep = {"format": "c10-prereg44 eval v1", "smoke": smoke, "created": r.now_iso(),
           "seconds": round(time.time() - t0, 1), "code": cs, "earlier_run_heads": earlier,
           "rerun_reason": a.reason if a.rerun else None, "selected": sel,
           "smoke_note": "CNN44 replaced by noise: numbers meaningless" if smoke else None,
           "what": "PLAN.md 6: inner-only judgement of Q44 = Q25's seven terms + CNN44 (pooled mean of per-game "
                   "log-probs), 5-fold player OOF; no validation scoring",
           "status": status, "seam": seam, "seam_ref_bound": bound, "seam1_bitwise": seam1_bitwise, "seam_m": sm,
           "seam_k": seam_k,
           "candidate": {"name": "Q44", "terms": terms8, **inner.short(cand, q25)},
           "baseline": {"name": "Q25", "terms": terms7, "inner_rank_balanced": q25["rb"],
                        "single_colour_rank_balanced": q25["mono_rb"]},
           "gates": gates,
           "record_only": {"CNN44_alone": alone, "by_game_count": E23.by_count(inner, cand, q25),
                           "by_level": E23.by_level(inner, cand, q25), "weights": E23.w_record(cand, terms8)},
           "peak_memory_record": E23.peak_memory()}
    E23.write_json(out_path, rep)
    log(f"-> {out_path}: {status} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "stage0b", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--rerun", action="store_true")
        p.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    return {"select": cmd_select, "stage0b": cmd_stage0b, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
