"""PREREG_07 pilot, step 2 (CPU): fuse the HumanSL term into Q02 out of fold and report the pilot gate.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg07/explore/pilot_humansl/pilot_fuse.py
    -> results/prereg07/explore/pilot_humansl/pilot_fuse_report.json

Inputs: data/prereg07/pilot/plan.json and the chunk files written by pilot_run.py (sets fitmap, bank, and mono when
present); the PREREG_04 inner inputs through results/prereg05/explore/u1/u1_lib.load_inputs (inner pseudo-questions
only, frozen tools read-only). Validation files are never opened (hs_lib's and u1_lib's guards). No score.py.

1 Per-game HS term. For each game the target player's HS curve is x_g[j] = (sum of log p(played move | profile j)) /
  (number of target moves), j over the plan's profiles. A multinomial logistic regression (10 levels, L2 on the
  weights, features standardised with fitmap means / SDs) is fitted on the fitmap games ONLY (fit players, one game per
  player, 300 per level; none is an inner player), L2 strength chosen by 5-fold CV log-loss on fitmap. Its per-game
  log-probabilities over the 10 levels, lp_HS = log_softmax(W x_g + b), are computed for every covered inner game.
  So no inner label is used to build lp_HS; inner labels enter only through the fold-wise fusion fit below, exactly as
  for C5 / O4 / SWA / SWB.
2 Fusion. lp["HS"] is added to the PREREG_04 inputs and U.run_gen(d, terms) reruns the generalised calibration out of
  fold (5 player folds; fold f's parameters are fitted on the fit pseudo-questions of the other folds' players):
      Q02     = R20 + O4 + SWA + SWB             terms OP, O4, SWA, SWB   (04's adopted combination; seam-checked
                                                                            bit for bit against u1_oof.npz)
      Q02+HS  = Q02 + HS                         terms OP, O4, SWA, SWB, HS
  Record only: R20+HS (OP, HS); Q02+HS+HSO and Q02+HSO when the opponent side was scored (--sides both).
3 Gate (task text; pilot only, not a registration): on the inner eval pseudo-questions (prereg02 eval rounds),
  rank-balanced rule-B score  Q02+HS - Q02 >= +0.010  AND  paired PlayerBootstrap (players within level, 1000 draws,
  default_rng(0)) 95 % lower bound > 0.  Mono check (as PREREG_04 G4): on the PREREG_03 single-colour inner
  pseudo-questions whose games are all scored, paired PlayerBootstrap lower bound > 0 and the black and white point
  estimates > 0. Reported with its coverage (all 120,544 questions only when the mono set has been run).
"""

import argparse
import glob
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402

import hs_lib as H  # noqa: E402  (data/local guard)

U1_DIR = os.path.join(H.REPO, "results", "prereg05", "explore", "u1")
sys.path.insert(0, U1_DIR)
PILOT_DIR = os.path.join(H.REPO, "data", "prereg07", "pilot")
OUT = os.path.join(HERE, "pilot_fuse_report.json")
GATE = 0.010
Q02_TERMS = ["OP", "O4", "SWA", "SWB"]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_LEVELS = 10


# ================================================================ chunks
def load_set(name, n_profiles, pilot_dir=PILOT_DIR):
    """-> refs [G], sums [G, 2, N], counts [G, 2], failed [G] for the chunks present (sorted by chunk index)."""
    files = sorted(f for f in glob.glob(os.path.join(pilot_dir, name, "chunk_*.npz")) if ".tmp" not in f)
    refs, sums, counts, failed, meta = [], [], [], [], set()
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            if z["sums"].shape[2] != n_profiles:
                raise H.HSError(f"{f}: {z['sums'].shape[2]} profiles, plan has {n_profiles}")
            refs.append(z["ref"])
            sums.append(z["sums"])
            counts.append(z["counts"])
            failed.append(z["failed"])
            meta.add((str(z["precision"]), str(z["ckpt_sha256"]), str(z["hs_lib_sha256"]), tuple(z["sides"])))
    if not files:
        return None
    return {"files": len(files), "ref": np.concatenate(refs), "sums": np.concatenate(sums),
            "counts": np.concatenate(counts), "failed": np.concatenate(failed), "meta": sorted(meta)}


def curves(s, side=0):
    c = s["counts"][:, side].astype(np.float64)
    ok = (~s["failed"]) & (c > 0)
    x = np.zeros(s["sums"].shape[::2], dtype=np.float64)          # [G, N]
    x[ok] = s["sums"][ok, side, :] / c[ok, None]
    return x, ok


# ================================================================ per-game mapping: multinomial logistic regression
def mlr_obj(th, X, Y1, lam):
    n, d = X.shape
    W = th[:d * N_LEVELS].reshape(d, N_LEVELS)
    b = th[d * N_LEVELS:]
    Z = X @ W + b
    Z = Z - Z.max(1, keepdims=True)
    lse = np.log(np.exp(Z).sum(1))
    ll = (Z * Y1).sum(1) - lse
    loss = -ll.mean() + lam * float((W * W).sum())
    P = np.exp(Z - lse[:, None])
    G = (P - Y1) / n
    gW = X.T @ G + 2 * lam * W
    gb = G.sum(0)
    return loss, np.concatenate([gW.ravel(), gb])


def mlr_fit(X, y, lam):
    d = X.shape[1]
    Y1 = np.eye(N_LEVELS)[y]
    r = optimize.minimize(mlr_obj, np.zeros(d * N_LEVELS + N_LEVELS), args=(X, Y1, lam), jac=True,
                          method="L-BFGS-B", options={"maxiter": 2000})
    return {"W": r.x[:d * N_LEVELS].reshape(d, N_LEVELS), "b": r.x[d * N_LEVELS:], "loss": float(r.fun),
            "success": bool(r.success), "nit": int(r.nit)}


def mlr_logp(m, X):
    return H.log_softmax_np(X @ m["W"] + m["b"])


def fit_mapping(xf, yf, seed=0):
    """Standardise, choose lambda by 5-fold CV log-loss on the fitmap games, refit on all of them."""
    mu, sd = xf.mean(0), xf.std(0)
    sd[sd == 0] = 1.0
    Xs = (xf - mu) / sd
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(yf)) % 5
    cv = {}
    for lam in LAMBDAS:
        ll, acc, sc = [], [], []
        for f in range(5):
            m = mlr_fit(Xs[fold != f], yf[fold != f], lam)
            lp = mlr_logp(m, Xs[fold == f])
            ll.append(-lp[np.arange(len(lp)), yf[fold == f]])
            acc.append(np.argmax(lp, 1) == yf[fold == f])
            import u1_lib as U
            sc.append(U.score(U.rule_b_np(np.exp(lp)), yf[fold == f]))
        ll, acc, sc = np.concatenate(ll), np.concatenate(acc), np.concatenate(sc)
        cv[str(lam)] = {"logloss": float(ll.mean()), "top1": float(acc.mean()), "rule_b_score_per_game": float(sc.mean())}
    best = min(LAMBDAS, key=lambda l_: cv[str(l_)]["logloss"])
    m = mlr_fit(Xs, yf, best)
    return {"mu": mu, "sd": sd, "model": m, "lambda": best, "cv": cv,
            "logp": lambda X: mlr_logp(m, (X - mu) / sd)}


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pilot-dir", default=PILOT_DIR, help="chunk folder (default data/prereg07/pilot; a synthetic "
                                                           "folder for the mechanics smoke test)")
    a = ap.parse_args()
    t0 = time.time()
    import u1_lib as U
    with open(os.path.join(a.pilot_dir, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    N = len(plan["profiles"])
    sets = {k: load_set(k, N, a.pilot_dir) for k in ("fitmap", "bank", "mono")}
    if sets["fitmap"] is None or sets["bank"] is None:
        raise H.HSError("fitmap and bank must be run first (pilot_run.py run --set fitmap / bank)")
    if len(sets["bank"]["ref"]) != plan["counts"]["bank"]:
        raise H.HSError(f"bank: {len(sets['bank']['ref'])} games scored, plan has {plan['counts']['bank']}")
    rep = {"what": "PREREG_07 pilot: Q02 + HumanSL term, inner OOF (design phase, inner pseudo-questions only)",
           "created_utc": H.utc_now(), "plan_profiles": plan["profiles"], "pilot_dir": a.pilot_dir,
           "synthetic": bool(plan.get("synthetic", False)),
           "sets": {k: (None if v is None else {"games": int(len(v["ref"])), "chunks": v["files"],
                                                "failed": int(v["failed"].sum()), "meta": v["meta"]})
                    for k, v in sets.items()}}
    # ---- 1 mapping on fitmap
    fm = sets["fitmap"]
    lv_of = {lv: i for i, lv in enumerate(U.LEVELS)}
    yfit = np.array([lv_of[plan["sets"]["fitmap"]["games"][j]["rank"]] for j in fm["ref"]], dtype=np.int64)
    sides_scored = [s for s in (0, 1) if (fm["counts"][:, s] > 0).any()]
    maps = {}
    for s in sides_scored:
        x, ok = curves(fm, s)
        maps[s] = fit_mapping(x[ok], yfit[ok])
        rep[f"mapping_side{s}"] = {"games": int(ok.sum()), "lambda": maps[s]["lambda"], "cv": maps[s]["cv"],
                                   "fit_loss": maps[s]["model"]["loss"], "success": maps[s]["model"]["success"]}
        print(f"mapping side {s}: lambda {maps[s]['lambda']}, CV {maps[s]['cv'][str(maps[s]['lambda'])]}", flush=True)
    # ---- inner per-game lp_HS
    d = U.load_inputs()
    n_inner = d["inner"]["keys"]["n"]
    covered = np.zeros(n_inner, dtype=bool)
    lp_side = {s: np.zeros((n_inner, N_LEVELS)) for s in sides_scored}
    for name in ("bank", "mono"):
        st = sets[name]
        if st is None:
            continue
        for s in sides_scored:
            x, ok = curves(st, s)
            lp_side[s][st["ref"][ok]] = maps[s]["logp"](x[ok])
            if s == 0:
                covered[st["ref"][ok]] = True
    d["lp"]["HS"] = lp_side[0]
    if 1 in lp_side:
        d["lp"]["HSO"] = lp_side[1]
    inner = d["inner"]
    Xe, Xm, ms = inner["X_eval"], inner["X_mono"], inner["ms"]
    eval_cov = np.array([covered[np.asarray(g)].all() for g in inner["eval_groups"]])
    if not eval_cov.all():
        raise H.HSError(f"{int((~eval_cov).sum())} inner eval questions have an unscored game")
    mono_cov = np.array([covered[np.asarray(g)].all() for g in ms["groups"]])
    rep["coverage"] = {"inner_games_scored": int(covered.sum()), "eval_questions": int(len(eval_cov)),
                       "mono_questions_covered": int(mono_cov.sum()), "mono_questions": int(len(mono_cov))}
    # ---- 2 fusion, out of fold
    saved = U.load_saved()
    combos = {"Q02": Q02_TERMS, "Q02+HS": Q02_TERMS + ["HS"], "R20+HS": ["OP", "HS"]}
    if "HSO" in d["lp"]:
        combos.update({"Q02+HSO": Q02_TERMS + ["HSO"], "Q02+HS+HSO": Q02_TERMS + ["HS", "HSO"]})
    res = {}
    for cid, terms in combos.items():
        t = time.time()
        P, Pm, th = U.run_gen(d, terms)
        sc = U.score(U.rule_b_np(P), Xe.y)
        msc = U.score(U.rule_b_np(Pm), Xm.y)
        res[cid] = {"P": P, "Pm": Pm, "sc": sc, "msc": msc, "thetas": th, "terms": terms}
        print(f"{cid}: inner eval rank-balanced {U.rank_balanced(sc, Xe.y):.4f} ({time.time() - t:.0f}s)", flush=True)
    seam = {"eval_P_equal_u1": bool(np.array_equal(res["Q02"]["P"], saved["eval_P_Q02"])),
            "mono_P_equal_u1": bool(np.array_equal(res["Q02"]["Pm"], saved["mono_P_Q02"]))}
    rep["seam_Q02_vs_u1_oof"] = seam
    if not all(seam.values()):
        raise H.HSError(f"Q02 does not reproduce u1_oof.npz: {seam}")
    boot = inner["boot"]
    base = res["Q02"]
    rows = {}
    for cid, r in res.items():
        m = {"terms": r["terms"], "rank_balanced": U.rank_balanced(r["sc"], Xe.y),
             "question_mean": float(r["sc"].mean()),
             "per_level": {lv: float(r["sc"][Xe.y == i].mean()) for i, lv in enumerate(U.LEVELS)},
             "weights_by_fold": [list(map(float, th[2:2 + len(r["terms"])])) for th in r["thetas"]],
             "a_gamma_by_fold": [[float(th[0]), float(th[1])] for th in r["thetas"]]}
        if cid != "Q02":
            dd = boot.stats(r["sc"]) - boot.stats(base["sc"])
            m["diff_vs_Q02"] = m["rank_balanced"] - U.rank_balanced(base["sc"], Xe.y)
            m["diff_vs_Q02_ci95"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            m["question_mean_diff_vs_Q02"] = float(np.mean(r["sc"] - base["sc"]))
            sub = {"y": ms["y"][mono_cov], "player": ms["player"][mono_cov], "colour": ms["colour"][mono_cov],
                   "k": ms["k"][mono_cov], "n_players": ms["n_players"]}
            if mono_cov.any():
                sboot = U.PlayerBootstrap(sub["player"], sub["y"], sub["n_players"])
                m["mono_diff_vs_Q02"] = U.p3.mono_diff_record(sub, sboot, r["msc"][mono_cov], base["msc"][mono_cov])
        m["mono_rank_balanced_covered"] = U.rank_balanced(r["msc"], Xm.y, mono_cov) if mono_cov.any() else None
        rows[cid] = m
    rep["combinations"] = rows
    g = rows["Q02+HS"]
    mono = g.get("mono_diff_vs_Q02")
    gate = {"G1_delta": g["diff_vs_Q02"], "G1_passed": bool(g["diff_vs_Q02"] >= GATE - 1e-12),
            "G2_ci95": g["diff_vs_Q02_ci95"], "G2_passed": bool(g["diff_vs_Q02_ci95"][0] > 0),
            "question_mean_diff": g["question_mean_diff_vs_Q02"]}
    gate["passed"] = gate["G1_passed"] and gate["G2_passed"]
    if mono is not None:
        bw = {c: (mono["by_colour"].get(c) or {}).get("rank_balanced") for c in ("B", "W")}
        gate["mono"] = {"coverage": f"{int(mono_cov.sum())}/{len(mono_cov)} questions",
                        "rank_balanced_difference": mono["rank_balanced"], "ci95": mono["ci95"],
                        "by_colour": bw,
                        "passed": bool(mono["ci95"][0] > 0 and all(v is not None and v > 0 for v in bw.values())),
                        "complete": bool(mono_cov.all())}
    rep["gate"] = gate
    rep["seconds"] = time.time() - t0
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=float)
    print(json.dumps(gate, indent=1, default=float))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
