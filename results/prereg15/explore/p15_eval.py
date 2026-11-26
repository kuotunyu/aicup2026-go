"""prereg15 exploration (competition 2): the inner-only judgement of PLAN.md (EM prior shift arms A1 / A2 / A3,
per-colour fusion refit B1), judged once. CPU only.

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1; .venv/Scripts/python.exe; no OMP / OPENBLAS / MKL variable
    S=results/prereg15/explore
    $S/p15_eval.py eval            -> $S/eval_report.json (judged once; the code must be committed and clean)
    $S/p15_eval.py eval --smoke    -> data/prereg15/smoke/eval_report.json (every player's level permuted with
                                      default_rng(12345); pipeline and timing only, never a result)

Inputs (read only): data/prereg12/q09_inner/ (prereg12's prepared P1 arrays of the bank / mono games, through
ft_eval.InnerPrep), results/prereg09/rank/p1e_map.npz (Q09's frozen map, SHA-checked against frozen_config.json),
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
import ft_eval  # noqa: E402  (prereg12: InnerPrep only)
import p15_lib as K  # noqa: E402

R, rf, p3, p4 = r.R, r.rf, r.p3, r.p4
TOL = 1e-12
GATE_A_SKEW = 0.005
GATE_A_BAL_POINT = -0.001
GATE_A_BAL_LOWER = -0.005
GATE_A_EACH_SKEW = -0.002
GATE_B = 0.005
A_PRIORITY = ("A2", "A1", "A3")
SMOKE_SEED = 12345
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
Q09_TUNE = os.path.join(REPO, "results", "prereg09", "rank", "tune_report.json")
Q09_FROZEN = os.path.join(REPO, "results", "prereg09", "rank", "frozen_config.json")
PREP_DIR = os.path.join(REPO, "data", "prereg12", "q09_inner")
SMOKE_DIR = os.path.join(REPO, "data", "prereg15", "smoke")
REPORT = os.path.join(HERE, "eval_report.json")
OWN_FILES = ("PLAN.md", "p15_lib.py", "p15_eval.py", "test_p15.py")
IMPORTED = ("tools/p02_rank_fuse.py", "tools/p03_rank_fuse.py", "tools/p04_rank_fuse.py", "tools/p08_rank_fuse.py",
            "tools/p09_rank_fuse.py", "tools/p09_forward.py", "results/prereg12/explore/finetune/ft_eval.py",
            "results/prereg05/explore/u1/u1_lib.py")
STARTED = os.path.join(REPO, "data", "prereg15", "eval_started.json")
FORMAT = "c10-prereg15 explore eval v1"
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


def mono_terms(inner, E_q09, terms):
    """The single-colour m (pooled C5) and E terms exactly as p08_rank_fuse.run_terms builds them."""
    d = inner.d
    Xm, Sm, lp = d["inner"]["X_mono"], d["inner"]["S_mono"], d["lp"]
    Emap = {"HS": inner.E_hs, "P1TO": E_q09}
    mm = rf.pooled(Sm, lp["C5"], "log")
    Em = [Emap[t]["mono"] if t in Emap else (Xm.mop if t == "OP" else rf.pooled(Sm, lp[t], "log")) for t in terms]
    return mm, Em


# ---------------------------------------------------------------- (a) EM prior shift (PLAN.md 2)
def run_pools(Pm, y, player, n_players, level_of, bl, pi0, st, log):
    pred0 = rf.rule_b_np(Pm)
    sc0 = rf.official_score(pred0, y)
    uni = np.full(K.L, 1.0 / K.L)
    t0 = time.time()
    cal = K.sample_pools(K.SEED_TAU, st, K.proportions("S0"))
    tv_cal = np.array([K.tv(K.em_prior(Pm[ix], pi0)[0], pi0) for ix in cal])
    tau = float(np.percentile(tv_cal, K.TAU_PERCENTILE))
    log(f"tau {tau!r} from {len(cal)} calibration pools ({time.time() - t0:.0f}s)")
    arms = ("A1", "A2", "A3")
    rec_arms = ("A1_uniform_pi0", "oracle_prior", "A3_oracle_counts")
    out, pools_of, boot_store = {}, {}, {}
    for j, s in enumerate(K.SCENARIOS):
        t1 = time.time()
        props = K.proportions(s)
        pools = K.sample_pools(K.SEED_BASE + j, st, props)
        pools_of[s] = pools
        true_counts = K.largest_remainder(K.PER_K, props) * len(K.KS)
        sc = {a: np.empty(pools.shape) for a in arms + rec_arms}
        iters, nonconv, applied, err, tvs = [], 0, 0, [], []
        for i, ix in enumerate(pools):
            P, yy = Pm[ix], y[ix]
            pi_hat, Q, it, conv = K.em_prior(P, pi0)
            iters.append(it)
            nonconv += (not conv)
            t = K.tv(pi_hat, pi0)
            tvs.append(t)
            err.append(K.tv(pi_hat, props))
            a1 = rf.rule_b_np(Q)
            sc["A1"][i] = rf.official_score(a1, yy)
            use = t > tau
            applied += int(use)
            sc["A2"][i] = sc["A1"][i] if use else sc0[ix]
            sc["A3"][i] = rf.official_score(K.quota_assign(Q, K.largest_remainder(K.POOL, pi_hat)), yy)
            _pu, Qu, _iu, _cu = K.em_prior(P, uni)
            sc["A1_uniform_pi0"][i] = rf.official_score(rf.rule_b_np(Qu), yy)
            Qo = K.adjust(P, props, pi0)
            sc["oracle_prior"][i] = rf.official_score(rf.rule_b_np(Qo), yy)
            sc["A3_oracle_counts"][i] = rf.official_score(K.quota_assign(Qo, true_counts), yy)
        base = sc0[pools]
        flat_p = player[pools].ravel()
        res = {"proportions": [float(x) for x in props], "pools": int(len(pools)), "seed": K.SEED_BASE + j,
               "per_k_level_counts": [int(x) for x in K.largest_remainder(K.PER_K, props)],
               "q09_pool_mean": float(base.mean()), "q09_pool_sd": float(base.mean(1).std(ddof=1)),
               "em": {"iterations_mean": float(np.mean(iters)), "iterations_max": int(np.max(iters)),
                      "not_converged": int(nonconv), "tv_pi_hat_to_pi0_mean": float(np.mean(tvs)),
                      "tv_pi_hat_to_truth_mean": float(np.mean(err)),
                      "a2_pools_adjusted": int(applied)},
               "arms": {}}
        for a in arms + rec_arms:
            D, N = K.accumulate(flat_p, (sc[a] - base).ravel(), n_players)
            dp = K.delta_point(D, N, level_of, props)
            db = K.delta_boot(D, N, bl, props)
            boot_store[(s, a)] = db
            res["arms"][a] = {"delta": dp, "ci_adj": K.pct(db, K.CI_ADJ), "ci95": K.pct(db, K.CI_95),
                              "pool_mean": float(sc[a].mean()), "pool_mean_minus_q09": float(sc[a].mean() - base.mean()),
                              "per_k_delta": {str(k): float((sc[a] - base)[:, ki * K.PER_K:(ki + 1) * K.PER_K].mean())
                                              for ki, k in enumerate(K.KS)},
                              "record_only": a in rec_arms}
        out[s] = res
        log(f"{s}: {json.dumps({a: round(res['arms'][a]['delta'], 5) for a in arms + rec_arms})} "
            f"({time.time() - t1:.0f}s)")
    gates = {}
    for a in arms:
        sk = np.mean([boot_store[(s, a)] for s in K.SKEWED], axis=0)
        sk_point = float(np.mean([out[s]["arms"][a]["delta"] for s in K.SKEWED]))
        bal = out["S0"]["arms"][a]
        c1 = bal["delta"] >= GATE_A_BAL_POINT - TOL and bal["ci_adj"][0] > GATE_A_BAL_LOWER
        sk_ci = K.pct(sk, K.CI_ADJ)
        c2 = sk_point >= GATE_A_SKEW - TOL and sk_ci[0] > 0
        each = {s: out[s]["arms"][a]["delta"] for s in K.SKEWED}
        c3 = all(v >= GATE_A_EACH_SKEW - TOL for v in each.values())
        gates[a] = {"balanced": {"delta": bal["delta"], "ci_adj": bal["ci_adj"], "point_min": GATE_A_BAL_POINT,
                                 "lower_min_exclusive": GATE_A_BAL_LOWER, "passed": bool(c1)},
                    "skewed_mean": {"delta": sk_point, "ci_adj": sk_ci, "ci95": K.pct(sk, K.CI_95),
                                    "threshold": GATE_A_SKEW, "passed": bool(c2)},
                    "each_skewed": {"deltas": each, "min_allowed": GATE_A_EACH_SKEW, "passed": bool(c3)},
                    "passed": bool(c1 and c2 and c3)}
    chosen = next((a for a in A_PRIORITY if gates[a]["passed"]), None)
    return {"tau": tau, "tau_calibration": {"pools": int(len(cal)), "seed": K.SEED_TAU,
                                            "percentile": K.TAU_PERCENTILE,
                                            "tv_mean": float(tv_cal.mean())},
            "scenarios": out, "gates": gates, "priority": list(A_PRIORITY), "chosen": chosen}, pools_of


# ---------------------------------------------------------------- (b) per-colour refit (PLAN.md 3)
def run_percolour(q09, mm, Em, Xm, colour, y, log):
    folds = q09["folds"]
    P1 = np.full(q09["Pm"].shape, np.nan)
    P0 = np.full(q09["Pm"].shape, np.nan)
    fit_rec = []
    J = len(Em)
    for fr in folds:
        f = fr["fold"]
        th = np.asarray(fr["theta"], dtype=np.float64)
        w = th[2:2 + J]
        tr, ev = Xm.fold != f, Xm.fold == f
        for c in ("B", "W"):
            mt, me = tr & (colour == c), ev & (colour == c)
            thc, rec = K.fit_fixed_w(mm[mt], [E[mt] for E in Em], Xm.n[mt], y[mt], w)
            P1[me] = K.predict(thc, mm[me], [E[me] for E in Em], Xm.n[me])
            fit_rec.append({"fold": int(f), "colour": c, "fit_questions": int(mt.sum()), "eval_questions": int(me.sum()),
                            "a": float(thc[0]), "gamma": float(thc[1]), "beta": [float(x) for x in thc[2 + J:]],
                            "fold_a": float(th[0]), "fold_gamma": float(th[1]), **rec})
        th0, rec0 = K.fit_fixed_w(mm[tr], [E[tr] for E in Em], Xm.n[tr], y[tr], w)
        P0[ev] = K.predict(th0, mm[ev], [E[ev] for E in Em], Xm.n[ev])
        fit_rec.append({"fold": int(f), "colour": "both (B0, record only)", "a": float(th0[0]),
                        "gamma": float(th0[1]), **rec0})
        log(f"(b) fold {f} done")
    if np.isnan(P1).any() or np.isnan(P0).any():
        raise SystemExit("(b): a single-colour question without a prediction")
    return P1, P0, fit_rec


def b_record(P, sc_base, y, colour, ks, boot, ms):
    sc = rf.official_score(rf.rule_b_np(P), y)
    d = boot.stats(sc) - boot.stats(sc_base)
    allr = p3.score_summary(sc, y)["rank_balanced"] - p3.score_summary(sc_base, y)["rank_balanced"]
    by = {c: p3.score_summary(sc, y, colour == c)["rank_balanced"] - p3.score_summary(sc_base, y, colour == c)[
        "rank_balanced"] for c in ("B", "W")}
    byk = {str(k): p3.score_summary(sc, y, ks == k)["rank_balanced"] - p3.score_summary(sc_base, y, ks == k)[
        "rank_balanced"] for k in K.KS}
    md = p3.mono_diff_record(ms, boot, sc, sc_base)
    return sc, {"single_colour_rank_balanced": p3.score_summary(sc, y)["rank_balanced"], "delta": allr,
                "ci_adj": K.pct(d, K.CI_ADJ), "ci95": K.pct(d, K.CI_95), "by_colour_delta": by, "by_k_delta": byk,
                "by_colour_ci95_own_draws": {c: (md["by_colour"][c] or {}).get("ci95") for c in ("B", "W")},
                "question_mean_delta": float(sc.mean() - sc_base.mean())}


def cmd_eval(a):
    t0 = time.time()
    r.blas_env_check(False)
    smoke = bool(a.smoke)
    out_path = os.path.join(SMOKE_DIR, "eval_report.json") if smoke else REPORT
    cs = code_state()
    earlier = None
    if not smoke:
        if cs["dirty"]:
            raise SystemExit(f"the prereg15 files or the imported tools are not committed and clean: {cs['dirty']}")
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
    q09 = inner.run(list(r.Q08_TERMS), {"HS": inner.E_hs, "P1TO": E_q09})
    d = inner.d["inner"]
    Xm, ms = d["X_mono"], d["ms"]
    mm, Em = mono_terms(inner, E_q09, list(r.Q08_TERMS))
    Pm_re = p4.gen_fold_predict(q09["folds"], mm, Em, Xm.n, Xm.fold)
    y_real = np.asarray(ms["y"], dtype=np.int64)
    sc_re = rf.official_score(rf.rule_b_np(q09["Pm"]), y_real)
    seam = {"Q02H": [inner.h["rb"], r.REF["Q02H"]], "Q08": [inner.q08["rb"], r.REF["Q08"]],
            "Q08_mono": [inner.q08["mono_rb"], r.REF["Q08_mono"]], "Q09": [q09["rb"], ref["Q09"]],
            "Q09_mono": [q09["mono_rb"], ref["Q09_mono"]],
            "Q09_mono_rule_b_recomputed": [p3.score_summary(sc_re, y_real)["rank_balanced"], ref["Q09_mono"]]}
    seam_ok = all(abs(x - y) <= TOL for x, y in seam.values())
    pm_equal = bool(np.array_equal(Pm_re, q09["Pm"]))
    log(f"seam {'ok' if seam_ok and pm_equal else 'FAILED'}: {seam}; mono Pm rebuilt bit for bit: {pm_equal}")
    if not (seam_ok and pm_equal):
        write_json(out_path, {"format": FORMAT, "status": "stop: seam failed", "seam": seam,
                              "mono_pm_rebuilt_equal": pm_equal, "smoke": smoke, "code": cs})
        return 1
    Pm = np.asarray(q09["Pm"], dtype=np.float64)
    player = np.asarray(ms["player"], dtype=np.int64)
    n_players = int(ms["n_players"])
    colour = np.asarray(ms["colour"])
    ks = np.asarray(ms["k"], dtype=np.int64)
    level_of = np.full(n_players, -1, dtype=np.int64)
    level_of[player] = y_real
    if smoke:
        lv = level_of.copy()
        present = lv >= 0
        lv[present] = np.random.default_rng(SMOKE_SEED).permutation(lv[present])
        level_of = lv
        y = level_of[player]
        boot = rf.PlayerBootstrap(player, y, n_players)
    else:
        y = y_real
        boot = inner.mboot
    if not np.array_equal(level_of[player], y):
        raise SystemExit("a player with questions of two levels")
    ms_y = dict(ms, y=y)
    bl = K.boot_levels(boot, level_of)
    pi0 = K.level_balanced_mean(Pm, y)
    st = K.strata(y, ks)
    log(f"pi0 {pi0.tolist()}")
    # ---- (a)
    ta = time.time()
    rep_a, pools_of = run_pools(Pm, y, player, n_players, level_of, bl, pi0, st, log)
    rep_a["pi0"] = [float(x) for x in pi0]
    rep_a["seconds"] = round(time.time() - ta, 1)
    # ---- (b)
    tb = time.time()
    sc0 = rf.official_score(rf.rule_b_np(Pm), y)
    P1, P0, fit_rec = run_percolour(q09, mm, Em, Xm, colour, y, log)
    sc1, b1 = b_record(P1, sc0, y, colour, ks, boot, ms_y)
    _sc_b0, b0 = b_record(P0, sc0, y, colour, ks, boot, ms_y)
    gb = {"delta": b1["delta"], "threshold": GATE_B, "ci_adj": b1["ci_adj"], "by_colour": b1["by_colour_delta"]}
    gb["passed"] = bool(b1["delta"] >= GATE_B - TOL and b1["ci_adj"][0] > 0
                        and all(v >= -TOL for v in b1["by_colour_delta"].values()))
    b_pools = {}
    for s, pools in pools_of.items():
        props = K.proportions(s)
        D, N = K.accumulate(player[pools].ravel(), (sc1[pools] - sc0[pools]).ravel(), n_players)
        b_pools[s] = K.delta_point(D, N, level_of, props)
    rep_b = {"B1": b1, "B0_record_only": b0, "gate": gb, "B1_pool_delta_record_only": b_pools, "fits": fit_rec,
             "seconds": round(time.time() - tb, 1)}
    passed = {"a": rep_a["chosen"], "b": "B1" if gb["passed"] else None}
    status = ("gates not passed: stop" if not (passed["a"] or passed["b"])
              else f"passed: {', '.join(x for x in (passed['a'], passed['b']) if x)} (worth a PREREG_15; not adoption)")
    rep = {"format": FORMAT, "created": r.now_iso(), "seconds": round(time.time() - t0, 1), "smoke": smoke,
           "smoke_note": ("every inner player's level permuted with default_rng(12345): numbers are meaningless"
                          if smoke else None),
           "rerun_reason": a.reason if a.rerun else None, "earlier_run_heads": earlier,
           "what": "PLAN.md: inner-only judgement of (a) EM prior shift A1 / A2 / A3 on fake 400-question pools and "
                   "(b) the per-colour refit of a / gamma / beta (B1); vs Q09 + rule B; training players only, no "
                   "validation scoring (competition 2 has none left)",
           "status": status, "passed": passed, "seam": seam, "mono_pm_rebuilt_equal": pm_equal,
           "multiplicity": "4 arms (A1, A2, A3, B1): Bonferroni, every gate interval two-sided 98.75 % (percentiles "
                           "0.625 / 99.375); (a) takes the first passing arm in the order A2, A1, A3",
           "baseline": {"name": "Q09", "inner_rank_balanced": q09["rb"], "single_colour_rank_balanced": q09["mono_rb"]},
           "a": rep_a, "b": rep_b, "q09_map_sha256": map_sha, "code": cs,
           "single_colour_questions": int(len(y)), "players": n_players}
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
