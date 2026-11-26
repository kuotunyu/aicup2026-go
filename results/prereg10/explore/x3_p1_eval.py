"""prereg10 explore x3 (competition 1): do KataGo HumanSL neutral-probe (P1) features identify players beyond 06?

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg10/explore/x3_p1_eval.py [--seeds 40]

Inputs (all fit players, git-ignored):
  data/prereg10/tmp/p1pilot/{sets.json, chunk_*.npz}  x2_p1_extract.py: 200 fit players (20 per level, >= 120 black
      games) x 40 black games; per game the P1 means (side 0 = the player, black; side 1 = the opponent).
  data/prereg08/p1/sets.json + probe/chunk_*.npz       PREREG_08 P1 probe: 30,000 fit games (~5 per player, both
      colours); used ONLY to learn the P1 projections, with every pilot player removed.
  data/prereg06/feats, cnn_fit                          the 06 terms (as x1_scenarios.py).
Pilot pools (one per seed): per player the 40 games shuffled (random.Random('x3|seed|pid')); candidates = first C
games, the question = next k (o2o, 200 questions). Scenarios:
  c30   C = 30, k = 5 or 10 (half each)
  c30k2 C = 30, k = 2
  c30nov C = 30, k = 5 or 10 but only games whose first 6 plies (canonical) are not among the candidate games (k_eff)
P1 representations of a game (side 0 = the player's own moves unless noted):
  P1cos   cosine between mean-pooled game vectors after standardising with the probe mean / sd (no learning)
  P1lda   LDA (tools/p06_style.lda_fit, reg 1e-3, 64 dims) learned on the probe's black games of non-pilot players with
          >= 2 black games; term = -squared distance of group means (= the 06 L term)
  P1lda2  same with both sides (player + opponent, 1540 dims), black probe games
  P1ldaBW same as P1lda but the LDA learned on the probe games of both colours (player's side)
Fusions: V2 (06 main, CSLS10) vs V2 + w * z(P1x) (+ CSLS10), w chosen on the tune seeds (first half) from a grid, reported
on the eval seeds (second half). Paired bootstrap over players (2,000 resamples) of the per-player mean point difference.
"""
import argparse
import json
import os
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import numpy as np  # noqa: E402

import p06_sim as sim  # noqa: E402
import p06_style as st  # noqa: E402

st.install_guard()
PILOT = os.path.join(REPO, "data", "prereg10", "tmp", "p1pilot")
PROBE_SETS = os.path.join(REPO, "data", "prereg08", "p1", "sets.json")
PROBE_DIR = os.path.join(REPO, "data", "prereg08", "p1", "probe")
FEATS = os.path.join(REPO, "data", "prereg06", "feats")
CNN = os.path.join(REPO, "data", "prereg06", "cnn_fit", "black_ge105.npz")
KEYS = os.path.join(REPO, "data", "prereg10", "tmp", "open_keys.npz")
WGRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 48.0)


def load_chunks(d, n):
    means = np.zeros((n, 2, 770), np.float32)
    failed = np.zeros(n, bool)
    have = np.zeros(n, bool)
    for f in sorted(os.listdir(d)):
        if not (f.startswith("chunk_") and f.endswith(".npz")) or ".tmp" in f:
            continue
        with np.load(os.path.join(d, f), allow_pickle=False) as z:
            ref = z["ref"]
            means[ref] = z["means"]
            failed[ref] = z["failed"]
            have[ref] = True
    return means, have & ~failed


def standardise(X, mu, sd):
    return (X - mu) / sd


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def gm(V, groups):
    return st.group_means(V, groups)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "x3_p1_eval.json"))
    a = ap.parse_args()
    t0 = time.time()
    with open(os.path.join(PILOT, "sets.json"), encoding="utf-8") as f:
        ps = json.load(f)
    games = ps["games"]
    pm, pok = load_chunks(PILOT, len(games))
    with open(PROBE_SETS, encoding="utf-8") as f:
        pr = json.load(f)["probe"]["games"]
    qm, qok = load_chunks(PROBE_DIR, len(pr))
    pilot_players = set(ps["players"])
    # ---------------- projections learned on the probe (pilot players removed)
    q_pid = np.array([g["player_id"] for g in pr])
    q_col = np.array([g["color"] for g in pr])
    keep = qok & ~np.isin(q_pid, sorted(pilot_players))
    proj = {}
    Xall = qm[keep][:, 0, :].astype(np.float64)
    mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
    proj["mu_sd"] = (mu, sd)

    def lda_on(mask, both_sides):
        X = qm[mask]
        X = X.reshape(len(X), -1) if both_sides else X[:, 0, :]
        X = X.astype(np.float64)
        lab = q_pid[mask]
        cnt = {p: c for p, c in zip(*np.unique(lab, return_counts=True))}
        m2 = np.array([cnt[p] >= 2 for p in lab])
        X, lab = X[m2], lab[m2]
        m = np.mean(X, 0)
        s = X.std(0) + 1e-6
        model = st.lda_fit((X - m) / s, lab)
        return {"model": model, "m": m, "s": s, "both": both_sides, "n_games": int(len(X)),
                "n_players": int(len(set(lab.tolist())))}
    proj["P1lda"] = lda_on(keep & (q_col == "B"), False)
    proj["P1lda2"] = lda_on(keep & (q_col == "B"), True)
    proj["P1ldaBW"] = lda_on(keep, False)
    print("projections", {k: (v["n_players"], v["n_games"]) for k, v in proj.items() if k != "mu_sd"},
          round(time.time() - t0, 1), flush=True)

    def proj_games(name, idx):
        X = pm[idx]
        if name == "P1cos":
            return l2(standardise(X[:, 0, :].astype(np.float64), mu, sd))
        p = proj[name]
        X = X.reshape(len(X), -1) if p["both"] else X[:, 0, :]
        return st.lda_project((X.astype(np.float64) - p["m"]) / p["s"], p["model"])
    P1NAMES = ("P1cos", "P1lda", "P1lda2", "P1ldaBW")
    PG = {n: proj_games(n, np.arange(len(games))) for n in P1NAMES}

    # ---------------- 06 terms on the same games
    store = sim.Store(FEATS)
    lda06 = {"B": sim.LdaData(FEATS, "B", store)}
    cnn = sim.Cnn(CNN, store)
    keys6 = np.load(KEYS)["c6"]
    by_player = {}
    for j, g in enumerate(games):
        if pok[j]:
            by_player.setdefault(g["player_id"], []).append(j)
    players = sorted(p for p in ps["players"] if len(by_player.get(p, [])) >= 40)
    srow = np.array([g["store_row"] for g in games])
    models = {"B": lda06["B"].fit_excluding(store, set(players))}
    print("players", len(players), "06 LDA", models["B"]["n_players"], round(time.time() - t0, 1), flush=True)

    def pool(seed, scen):
        cand, q, ks = [], [], []
        for i, p in enumerate(players):
            g = list(by_player[p])
            random.Random("x3|%d|%s" % (seed, p)).shuffle(g)
            c = g[:30]
            rest = g[30:]
            k = 2 if scen == "c30k2" else (5 if i % 2 == (seed % 2) else 10)
            if scen == "c30nov":
                ck = {keys6[srow[j]] for j in c}
                rest = [j for j in rest if keys6[srow[j]] not in ck] or rest[:1]
            qq = rest[:k]
            cand.append(c)
            q.append(qq)
            ks.append(len(qq))
        return cand, q, np.array(ks)

    def F_of(cand, q):
        rec = {"pool": players, "cand_rows": [[int(srow[j]) for j in c] for c in cand],
               "q_rows": [[int(srow[j]) for j in g] for g in q]}
        T = sim.pool_terms(store, rec, lda06, models, cnn, want=("O", "JO", "S2", "L", "E0"), colours=("B",))
        out = {k: st.zscore_rows(*T[k]) for k in ("E0", "O", "L", "JO", "S2")}
        out["V2nc"] = sim.combo_F(T, "E0", sim.V2_W)
        for n in P1NAMES:
            V = PG[n]
            Zc, Zq = gm(V, cand), gm(V, q)
            out[n] = st.zrows(Zq @ Zc.T if n == "P1cos" else st.lda_term(Zc, Zq))
        return out
    owner = np.arange(len(players))
    rows = []
    for scen in ("c30", "c30k2", "c30nov"):
        for seed in range(a.seeds):
            cand, q, ks = pool(seed, scen)
            F = F_of(cand, q)
            pts = {}
            for n in ("E0", "O", "L", "JO", "S2") + P1NAMES:
                pts[n] = st.points(st.ranks(F[n], owner))
            pts["V2"] = st.points(st.ranks(st.csls(F["V2nc"], 10), owner))
            pts["Q04"] = st.points(st.ranks(F["E0"] + F["O"], owner))
            for n in P1NAMES:
                for w in WGRID:
                    pts["V2+%s@%g" % (n, w)] = st.points(st.ranks(st.csls(F["V2nc"] + w * F[n], 10), owner))
                    pts["Q04+%s@%g" % (n, w)] = st.points(st.ranks(F["E0"] + F["O"] + w * F[n], owner))
                pts["E0+%s" % n] = st.points(st.ranks(F["E0"] + F[n], owner))
            rows.append({"scen": scen, "seed": seed, "mean_k": float(ks.mean()),
                         "pts": {k: v.tolist() for k, v in pts.items()}})
        print(scen, "done", round(time.time() - t0, 1), flush=True)

    # ---------------- summary
    rng = np.random.default_rng(20261010)
    B = 2000
    idx = rng.integers(0, len(players), (B, len(players)))

    def boot(diff_pp):
        m = diff_pp[idx].mean(1)
        return [round(float(diff_pp.mean()), 4), round(float(np.quantile(m, 0.025)), 4),
                round(float(np.quantile(m, 0.975)), 4)]
    summ = {}
    half = a.seeds // 2
    for scen in ("c30", "c30k2", "c30nov"):
        R = [r for r in rows if r["scen"] == scen]
        tune = [r for r in R if r["seed"] < half]
        ev = [r for r in R if r["seed"] >= half]
        pp = lambda rs, k: np.mean([r["pts"][k] for r in rs], 0)  # noqa: E731  per-player mean over seeds
        s = {"mean_k": round(float(np.mean([r["mean_k"] for r in R])), 3), "alone_eval": {}, "fusion": {}}
        for n in ("E0", "O", "L", "JO", "S2", "Q04", "V2") + P1NAMES + tuple("E0+%s" % n for n in P1NAMES):
            s["alone_eval"][n] = round(float(pp(ev, n).mean()), 4)
        for base in ("V2", "Q04"):
            for n in P1NAMES:
                best = max(WGRID, key=lambda w: pp(tune, "%s+%s@%g" % (base, n, w)).mean())
                k = "%s+%s@%g" % (base, n, best)
                s["fusion"]["%s+%s" % (base, n)] = {
                    "w_tuned": best, "eval": round(float(pp(ev, k).mean()), 4),
                    "minus_base_eval_boot_players": boot(pp(ev, k) - pp(ev, base)),
                    "grid_eval": {str(w): round(float(pp(ev, "%s+%s@%g" % (base, n, w)).mean()), 4) for w in WGRID}}
        s["E0+P1lda_minus_E0"] = boot(pp(ev, "E0+P1lda") - pp(ev, "E0"))
        summ[scen] = s
    out = {"what": "prereg10 x3: P1 (KataGo HumanSL neutral probe) features for competition 1, pilot of fit players",
           "n_players": len(players), "games_per_player": 40, "candidate_games": 30, "pool_size": len(players),
           "seeds": a.seeds, "tune_seeds": list(range(half)), "eval_seeds": list(range(half, a.seeds)),
           "projections": {k: {"n_players": v["n_players"], "n_games": v["n_games"]} for k, v in proj.items()
                           if k != "mu_sd"},
           "pilot_games_ok": int(pok.sum()), "probe_games_ok": int(qok.sum()),
           "summary": summ, "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
