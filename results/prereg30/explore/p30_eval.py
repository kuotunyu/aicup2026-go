"""prereg30 (CPU): seam S and the incremental-information test, judged once (PLAN.md 3, 4).

    Git Bash, repository root; PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg30/explore/p30_eval.py
        seam            -> results/prereg30/explore/seam_s.json (P1 part of the pilot chunks vs the original probe)
        run             -> data/prereg30/eval_started.json, then results/prereg30/explore/eval_report.json (ONCE)
        run --smoke     -> data/prereg30/smoke/eval_report.json: v2 replaced by noise (default_rng(12345)), no chunk
                           read, 30 questions per cell, 2 placebos, 50 bootstrap draws; numbers are not recorded

Pure numpy / scipy: no torch, no KataGo. Reads only: the pilot chunks (data/prereg30/pilot/probe), the pilot map and
sets (data/prereg30/pilot), probe_X / probe_meta (data/prereg17/prep), the original probe chunks' counts, Q09's
p1e_map.npz and the frozen S25 models (results/prereg25/rank). No inner array, no validation player, nothing under
data/local (every path goes through p30_lib.guard). Does not run tools/score.py.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import warnings

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (HERE, os.path.join(REPO, "tools"), os.path.join(REPO, "results", "prereg19", "explore")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from scipy import optimize, stats  # noqa: E402

import p09_forward as FW  # noqa: E402
import p19_forward as SF  # noqa: E402
import p25_forward as F25  # noqa: E402
import p19_lib as PL  # noqa: E402
import p30_lib as V  # noqa: E402
import p30_data as D  # noqa: E402

guard = V.guard
L = 10
KS = PL.KS                                   # (5, 10, 15, 20)
PER_CELL = 300                               # PLAN.md 4.1
Q_SEED = 30030
FOLDS = 5
FOLD_SALT = "c10-prereg30|fold|"
LAMBDA = 1e-3                                # PLAN.md 4.2
MAXITER = 2000
TAU = (np.arange(L) - 4.5) / 4.5
N_BOOT = 1000
N_PLACEBO = 20
TOL_FEAT = 0.05                              # PLAN.md 3, seam S
GO_SCORE = 0.005                             # PLAN.md 4.4 GO-a
E1 = float(np.exp(-1.0))
NOISE_SEED = 12345
SMOKE = {"per_cell": 30, "placebo": 2, "boot": 50}
GROUPS = {"nlp_only": [0, 1, 5, 6], "loss_only": [2, 3, 4, 7, 8, 9]}

PILOT = os.path.join(REPO, "data", "prereg30", "pilot")
CHUNK_DIR = os.path.join(PILOT, "probe")
PROBE_X = os.path.join(REPO, "data", "prereg17", "prep", "probe_X.npy")
Q09_MAP = os.path.join(REPO, "results", "prereg09", "rank", "p1e_map.npz")
S25_DIR = os.path.join(REPO, "results", "prereg25", "rank")
STARTED = os.path.join(REPO, "data", "prereg30", "eval_started.json")
SMOKE_DIR = os.path.join(REPO, "data", "prereg30", "smoke")
SEAM_OUT = os.path.join(HERE, "seam_s.json")
REPORT = os.path.join(HERE, "eval_report.json")
FORMAT = "c10-prereg30 explore eval v1"
PROBE_X_SHA256 = "ee8ce70ec56b8ffb92952ca44ad271a420f1d145b8277f09ea6a611fb05701dc"  # results/prereg09/rank/check_report.json


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(guard(tmp), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# ================================================================ inputs
def load_pilot():
    with np.load(guard(D.MAP_OUT), allow_pickle=False) as z:
        m = {k: np.asarray(z[k]) for k in ("probe_row", "src", "ref", "players", "y")}
    with open(guard(D.SETS_OUT), encoding="utf-8") as f:
        games = json.load(f)["probe"]["games"]
    if len(games) != len(m["probe_row"]):
        raise SystemExit("pilot sets.json and pilot_map.npz disagree")
    for j, g in enumerate(games):
        if f"{g['rank']}:{g['player_id']}" != str(m["players"][j]):
            raise SystemExit(f"pilot game {j}: player differs between sets.json and pilot_map.npz")
    m["colour"] = np.array([g["color"] for g in games])
    m["players"] = m["players"].astype(str)
    m["src"] = m["src"].astype(str)
    return m


def read_chunks(n):
    """The pilot chunks -> dict of per-game arrays over the n pilot games (each exactly once) and the chunk SHA-256s."""
    files = sorted(glob.glob(os.path.join(guard(CHUNK_DIR), "chunk_*.npz")))
    files = [f for f in files if ".tmp" not in f]
    out = {"means": np.zeros((n, 2, 770), np.float32), "counts": np.zeros((n, 2), np.int64),
           "failed": np.zeros(n, bool), "n_moves": np.zeros(n, np.int64), "v2": np.full((n, V.V2_DIM), np.nan),
           "v2_ok": np.zeros(n, bool), "seen": np.zeros(n, np.int64)}
    shas, nlp_max = [], -np.inf
    for f in files:
        shas.append(V.sha_raw(f))
        with np.load(f, allow_pickle=False) as z:
            if "p30_format" not in z.files or str(z["p30_format"]) != V.V2_FORMAT:
                raise SystemExit(f"{f}: not a prereg30 chunk")
            ref = np.asarray(z["ref"]).astype(np.int64)
            out["means"][ref] = np.asarray(z["means"])
            out["counts"][ref] = np.asarray(z["counts"])
            out["failed"][ref] = np.asarray(z["failed"], bool)
            out["n_moves"][ref] = np.asarray(z["n_moves"])
            v2, ok = V.chunk_v2(z)
            out["v2"][ref], out["v2_ok"][ref] = v2, ok
            out["seen"][ref] += 1
            lp, lm = np.asarray(z["pos_logp"], np.float64), np.asarray(z["pos_lmax"], np.float64)
            if len(lp):
                nlp_max = max(nlp_max, float((lp - lm).max()))
    if not (out["seen"] == 1).all():
        raise SystemExit(f"pilot chunks cover {int((out['seen'] == 1).sum())} of {n} games exactly once")
    out["chunk_sha256"] = hashlib.sha256("\n".join(shas).encode()).hexdigest()
    out["chunks"] = len(files)
    out["nlp_max"] = nlp_max
    return out


# ================================================================ seam S (PLAN.md 3)
def seam(pm, ch):
    want = {}
    for s, r in zip(pm["src"], pm["ref"]):
        want.setdefault(str(s), set()).add(int(r))
    old = D.old_chunk_counts(want)
    n = len(pm["probe_row"])
    oc = np.array([old[(str(s), int(r))][0] for s, r in zip(pm["src"], pm["ref"])], dtype=np.int64)
    of = np.array([old[(str(s), int(r))][1] for s, r in zip(pm["src"], pm["ref"])], dtype=bool)
    on = np.array([old[(str(s), int(r))][2] for s, r in zip(pm["src"], pm["ref"])], dtype=np.int64)
    same_counts = bool(np.array_equal(oc, ch["counts"]))
    same_failed = bool(np.array_equal(of, ch["failed"]))
    same_nmoves = bool(np.array_equal(on, ch["n_moves"]))
    us = FW.usable_games(ch["failed"], ch["counts"])
    mp = FW.load_map(Q09_MAP)
    PX = np.load(guard(PROBE_X), mmap_mode="r")
    idx = np.flatnonzero(us)
    maxd, bitwise = 0.0, True
    for s in range(0, len(idx), 4096):
        b = idx[s:s + 4096]
        xn = FW.x_raw32(ch["means"][b], ch["counts"][b])
        xo = np.asarray(PX[pm["probe_row"][b]], dtype=np.float32)
        bitwise = bitwise and bool(np.array_equal(xn, xo, equal_nan=True))
        d = np.abs(FW.standardise_x(xn, mp["mu_x"], mp["sd_x"]) - FW.standardise_x(xo, mp["mu_x"], mp["sd_x"]))
        maxd = max(maxd, float(d.max()))
    passed = same_counts and same_failed and same_nmoves and maxd <= TOL_FEAT
    v2 = ch["v2"][us]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rec = {"format": "c10-prereg30 seam S v1", "games": n, "chunks": ch["chunks"],
               "chunk_list_sha256": ch["chunk_sha256"], "usable": int(us.sum()),
               "same_counts": same_counts, "same_failed": same_failed, "same_n_moves": same_nmoves,
               "max_abs_diff_standardised": maxd, "tolerance": TOL_FEAT, "bitwise_equal_raw_X": bitwise,
               "v2_turns_not_contiguous": int((~ch["v2_ok"] & us).sum()),
               "v2_nan_fraction": dict(zip(V.V2_NAMES, np.isnan(v2).mean(0).round(5).tolist())),
               "nlp_max_over_positions": ch["nlp_max"], "passed": bool(passed)}
    return rec


# ================================================================ questions (PLAN.md 4.1)
def fold_of(key):
    return int(int(hashlib.sha256(f"{FOLD_SALT}{key}".encode("utf-8")).hexdigest()[:8], 16) % FOLDS)


def make_questions(pm, usable, per_cell, seed=Q_SEED):
    rows = np.flatnonzero(usable)
    pool = PL.PlayerPool(pm["players"], pm["y"], pm["colour"], rows)
    rng = np.random.default_rng(seed)
    st = {"colour_switched": 0, "level_missing": 0}
    qs = []
    for n in KS:
        for k in range(L):
            for _ in range(per_cell):
                q = PL.sample_question(rng, pool, k, n, st)
                if q is None:
                    st["level_missing"] += 1
                    continue
                qs.append((q[0], k, n))
    key = np.array([pm["players"][q[0][0]] for q in qs])
    return {"rows": [q[0] for q in qs], "y": np.array([q[1] for q in qs], np.int64),
            "n": np.array([q[2] for q in qs], np.int64), "player": key,
            "fold": np.array([fold_of(k) for k in key], np.int64), "stats": st}


def s25_lp(pm, Q):
    """lp25 [Q, 10] = Q25's SET term / 4 for every question (PLAN.md 4.2)."""
    pars = F25.load_models(S25_DIR)
    mp = FW.load_map(Q09_MAP)
    PX = np.load(guard(PROBE_X), mmap_mode="r")
    Xraw = np.asarray(PX[pm["probe_row"]], dtype=np.float32)
    Hs = SF.game_h(pars, Xraw, mp["mu_x"], mp["sd_x"])
    out = np.zeros((len(Q["y"]), L))
    for n in KS:
        qi = np.flatnonzero(Q["n"] == n)
        if len(qi) == 0:
            continue
        idx = np.stack([Q["rows"][i] for i in qi])
        lps = [SF.head_np(par, H[idx].mean(axis=1)) for par, H in zip(pars, Hs)]
        out[qi] = FW.lse_mean(lps)
    return out


def question_z(v2, Q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmean(v2[r], axis=0) for r in Q["rows"]])


# ================================================================ models (PLAN.md 4.2)
def unpack(theta, d):
    a = theta[0]
    beta = np.concatenate([[0.0], theta[1:L]])
    U = theta[L:L + d]
    W = theta[L + d:L + 2 * d]
    return a, beta, U, W


def logits(theta, LP, Z):
    d = Z.shape[1]
    a, beta, U, W = unpack(theta, d)
    out = a * LP + beta[None, :]
    if d:
        out = out + np.outer(Z @ U, TAU) + np.outer(Z @ W, TAU ** 2)
    return out


def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def objective(theta, LP, Z, y, lam):
    d = Z.shape[1]
    lg = log_softmax(logits(theta, LP, Z))
    Qn = len(y)
    f = -lg[np.arange(Qn), y].mean()
    G = np.exp(lg)
    G[np.arange(Qn), y] -= 1.0
    G /= Qn
    g = np.zeros_like(theta)
    g[0] = (G * LP).sum()
    g[1:L] = G.sum(0)[1:]
    if d:
        _a, _b, U, W = unpack(theta, d)
        g[L:L + d] = Z.T @ (G @ TAU) + 2 * lam * U
        g[L + d:] = Z.T @ (G @ TAU ** 2) + 2 * lam * W
        f += lam * (U @ U + W @ W)
    return f, g


def fit(LP, Z, y, lam=LAMBDA):
    d = Z.shape[1]
    th0 = np.zeros(L + 2 * d)
    th0[0] = 1.0
    r = optimize.minimize(objective, th0, args=(LP, Z, y, lam), jac=True, method="L-BFGS-B",
                          options={"maxiter": MAXITER})
    return r.x, {"converged": bool(r.success), "iterations": int(r.nit), "f": float(r.fun)}


def standardise_z(Ztr, Zte):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(Ztr, axis=0)
        sd = np.nanstd(Ztr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    out = []
    for Z in (Ztr, Zte):
        x = (Z - mu) / sd
        x[~np.isfinite(x)] = 0.0
        out.append(x)
    return out


def cv(LP, Z, Q, lam=LAMBDA):
    """5-fold player OOF log-probabilities [Q, 10] and per-fold records. Z None or [Q, 0] -> M0."""
    lp = np.zeros_like(LP)
    recs = []
    for f in range(FOLDS):
        tr, te = Q["fold"] != f, Q["fold"] == f
        if Z is None or Z.shape[1] == 0:
            Ztr, Zte = np.zeros((tr.sum(), 0)), np.zeros((te.sum(), 0))
        else:
            Ztr, Zte = standardise_z(Z[tr], Z[te])
        th, info = fit(LP[tr], Ztr, Q["y"][tr], lam)
        lp[te] = log_softmax(logits(th, LP[te], Zte))
        d = Ztr.shape[1]
        a, _b, U, W = unpack(th, d)
        recs.append({"fold": f, "alpha": float(a), "norm_u": float(np.linalg.norm(U)),
                     "norm_v": float(np.linalg.norm(W)), **info})
    return lp, recs


# ================================================================ scores (PLAN.md 4.3)
def rule_b(lp):
    p = np.exp(lp)
    b = p.copy()
    b[:, 1:] += E1 * p[:, :-1]
    b[:, :-1] += E1 * p[:, 1:]
    return np.argmax(b, axis=1)                  # ties -> first, as tools/decide.argmax_first


def per_question(lp, y):
    sc = PL.official_score(rule_b(lp), y)
    nll = -lp[np.arange(len(y)), y]
    return sc, nll


def rb(x, y):
    return PL.rank_balanced(x, y)


def bootstrap(diffs, Q, reps, seed=0):
    """Paired PlayerBootstrap within level (PLAN.md 4.3): diffs {name: [Q]} -> {name: [reps] rank-balanced means}."""
    rng = np.random.default_rng(seed)
    players, pid = np.unique(Q["player"], return_inverse=True)
    lev = np.zeros(len(players), np.int64)
    lev[pid] = Q["y"]
    nq = np.bincount(pid, minlength=len(players)).astype(np.float64)
    sums = {k: np.bincount(pid, weights=v, minlength=len(players)) for k, v in diffs.items()}
    by_level = [np.flatnonzero(lev == k) for k in range(L)]
    out = {k: np.zeros(reps) for k in diffs}
    for r in range(reps):
        c = np.zeros(len(players))
        for pl in by_level:
            if len(pl):
                np.add.at(c, pl[rng.integers(len(pl), size=len(pl))], 1.0)
        for k in diffs:
            vals = [float((c[pl] * sums[k][pl]).sum() / (c[pl] * nq[pl]).sum()) for pl in by_level if len(pl)]
            out[k][r] = np.mean(vals)
    return out


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summary(lp0, lp1, Q):
    s0, n0 = per_question(lp0, Q["y"])
    s1, n1 = per_question(lp1, Q["y"])
    out = {"score_m0": rb(s0, Q["y"]), "score_m1": rb(s1, Q["y"]), "nll_m0": rb(n0, Q["y"]),
           "nll_m1": rb(n1, Q["y"])}
    out["d_score"] = out["score_m1"] - out["score_m0"]
    out["d_nll"] = out["nll_m1"] - out["nll_m0"]
    return out, s1 - s0, n1 - n0


# ================================================================ run
def cmd_seam(a):
    pm = load_pilot()
    ch = read_chunks(len(pm["probe_row"]))
    rec = seam(pm, ch)
    write_json(SEAM_OUT, rec)
    log(json.dumps(rec))
    return 0 if rec["passed"] else 3


def cmd_run(a):
    t0 = time.time()
    if any(os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise SystemExit("unset OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS")
    smoke = a.smoke
    per_cell = SMOKE["per_cell"] if smoke else PER_CELL
    n_plac = SMOKE["placebo"] if smoke else N_PLACEBO
    n_boot = SMOKE["boot"] if smoke else N_BOOT
    pm = load_pilot()
    n = len(pm["probe_row"])
    if V.sha_raw(PROBE_X) != PROBE_X_SHA256:
        raise SystemExit("probe_X.npy: SHA-256 differs from 09's check_report.json")
    if smoke:
        seam_rec = {"skipped": "smoke"}
        us = np.ones(n, bool)
        v2 = np.random.default_rng(NOISE_SEED).standard_normal((n, V.V2_DIM))
        out_path = os.path.join(SMOKE_DIR, "eval_report.json")
        os.makedirs(guard(SMOKE_DIR), exist_ok=True)
    else:
        if os.path.exists(STARTED):
            raise SystemExit(f"{STARTED} exists: the test was already started once (PLAN.md 7); not run again")
        ch = read_chunks(n)
        seam_rec = seam(pm, ch)
        write_json(SEAM_OUT, seam_rec)
        if not seam_rec["passed"]:
            log("seam S not passed: stop, not judged", json.dumps(seam_rec))
            return 3
        us = FW.usable_games(ch["failed"], ch["counts"])
        v2 = ch["v2"]
        out_path = REPORT
    log(f"pilot games {n}, usable {int(us.sum())}")
    Q = make_questions(pm, us, per_cell)
    log(f"questions {len(Q['y'])}, missing cells {Q['stats']['level_missing']}")
    LP = s25_lp(pm, Q)
    Z = question_z(v2, Q)
    lp0, rec0 = cv(LP, None, Q)
    if not smoke:
        write_json(STARTED, {"format": "c10-prereg30 eval started v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                           time.gmtime()),
                             "seam": seam_rec, "code": V.own_code_shas()})
    lp1, rec1 = cv(LP, Z, Q)
    main, ds, dn = summary(lp0, lp1, Q)
    boot = bootstrap({"score": ds, "nll": dn}, Q, n_boot)
    main["d_score_ci95"] = ci(boot["score"])
    main["d_nll_ci95"] = ci(boot["nll"])
    log("main", json.dumps(main))
    plac = []
    usable_idx = np.flatnonzero(us)
    for i in range(n_plac):
        rng = np.random.default_rng([30, i])
        vp = v2.copy()
        vp[usable_idx] = v2[usable_idx[rng.permutation(len(usable_idx))]]
        lpp, _ = cv(LP, question_z(vp, Q), Q)
        plac.append(summary(lp0, lpp, Q)[0]["d_score"])
    log("placebo", plac)
    go = {"GO_a": {"threshold": GO_SCORE, "d_score": main["d_score"], "passed": bool(main["d_score"] >= GO_SCORE)},
          "GO_b": {"d_nll_ci95": main["d_nll_ci95"], "passed": bool(main["d_nll_ci95"][1] < 0)},
          "GO_c": {"placebo_max": float(max(plac)), "passed": bool(main["d_score"] > max(plac))}}
    go["all_folds_converged"] = bool(all(r["converged"] for r in rec0 + rec1))   # record only
    go["GO"] = bool(go["GO_a"]["passed"] and go["GO_b"]["passed"] and go["GO_c"]["passed"])
    # ---- record only (PLAN.md 4.5)
    rec = {}
    by_n = {}
    for nn in KS:
        m = Q["n"] == nn
        s0, _ = per_question(lp0[m], Q["y"][m])
        s1, _ = per_question(lp1[m], Q["y"][m])
        by_n[str(nn)] = {"questions": int(m.sum()), "d_score": rb(s1, Q["y"][m]) - rb(s0, Q["y"][m])}
    rec["by_n"] = by_n
    s0, _ = per_question(lp0, Q["y"])
    s1, _ = per_question(lp1, Q["y"])
    rec["by_level"] = {"m0": PL.per_level(s0, Q["y"]), "m1": PL.per_level(s1, Q["y"])}
    for name, cols in GROUPS.items():
        lpg, _ = cv(LP, Z[:, cols], Q)
        rec[name] = summary(lp0, lpg, Q)[0]
    lpz, _ = cv(np.zeros_like(LP), Z, Q)
    rec["z_only"] = {"score": rb(per_question(lpz, Q["y"])[0], Q["y"]),
                     "nll": rb(per_question(lpz, Q["y"])[1], Q["y"])}
    sp = {}
    for j, nm in enumerate(V.V2_NAMES):
        m = us & np.isfinite(v2[:, j])
        sp[nm] = float(stats.spearmanr(v2[m, j], pm["y"][m]).correlation) if m.sum() > 2 else None
    rec["spearman_game_v2_vs_level"] = sp
    rec["folds_m0"], rec["folds_m1"] = rec0, rec1
    rec["placebo_d_score"] = plac
    report = {"format": FORMAT, "smoke": bool(smoke),
              "what": "PLAN.md 4: does v2 add information on top of S25 (fit players held out by S25)? judged once",
              "seam_s": seam_rec, "pilot_games": n, "usable": int(us.sum()),
              "questions": int(len(Q["y"])), "question_stats": Q["stats"],
              "settings": {"per_cell": per_cell, "q_seed": Q_SEED, "folds": FOLDS, "lambda": LAMBDA,
                           "placebo": n_plac, "bootstrap": n_boot, "go_score": GO_SCORE},
              "main": main, "gates": go, "record_only": rec, "code": V.own_code_shas(),
              "seconds": round(time.time() - t0, 1)}
    write_json(out_path, report)
    log("GO" if go["GO"] else "NO-GO", json.dumps(go))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seam")
    r = sub.add_parser("run")
    r.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    return {"seam": cmd_seam, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
