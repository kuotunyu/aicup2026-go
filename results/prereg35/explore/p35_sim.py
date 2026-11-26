"""prereg35 explore (competition 1, v2 new pools): fit-player simulation of the new question shape (PLAN.md 1-3).

    PY results/prereg35/explore/p35_sim.py pool --seed 0            # shape stats of one pool (no score)
    PY results/prereg35/explore/p35_sim.py terms --seeds 100-119 --out data/prereg35/terms_tune
    PY results/prereg35/explore/p35_sim.py tune  --terms data/prereg35/terms_tune  -> results/prereg35/explore/tune.json
    PY results/prereg35/explore/p35_sim.py terms --seeds 0-29 --out data/prereg35/terms_eval
    PY results/prereg35/explore/p35_sim.py eval  --terms data/prereg35/terms_eval  -> results/prereg35/explore/eval.json

Only the fit-player feature store data/prereg06/feats (tools/p06_player.py build). No validation answers.
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import p06_sim as sim  # noqa: E402
import p06_style as st  # noqa: E402
from c10data import LEVELS  # noqa: E402

FEATS = os.path.join(REPO, "data", "prereg06", "feats")
N_CAND_GAMES = 10
PER_LEVEL = 60
K_LIST = [1] * 100 + [2] * 100 + [3] * 100 + [5] * 100
TERMS = ("O", "L", "JO", "S2", "PA")
FIXED = {  # name: (weights for O, L, JO, S2, PA), csls k (None = no CSLS)
    "A_k0": ((1, 0, 0, 0, 0), None), "A_k10": ((1, 0, 0, 0, 0), 10),
    "B_k0": ((1, 2.66, 0.5, 0.5, 0), None), "B_k10": ((1, 2.66, 0.5, 0.5, 0), 10),
    "B_k20": ((1, 2.66, 0.5, 0.5, 0), 20), "C_k10": ((1, 2.66, 0.5, 0.5, 1.0), 10),
}
GRID = {"L": (0, 1, 2, 2.66, 4, 6), "JO": (0, 0.5, 1), "S2": (0, 0.5, 1), "PA": (0, 0.5, 1, 2), "k": (None, 10, 20, 40)}
BOOT = 10000
BOOT_SEED = 35


def v2new_pool(store, seed):
    """PLAN.md 1: 60 fit players per level (sha256 order), 10 shuffled games each as candidates, 400 questions with
    k in K_LIST (shuffled), owners uniform with replacement, question games = the owner's next unused games."""
    bp = store.by_player()
    lvl = {p: str(store.rank[rows[0]]) for p, rows in bp.items()}
    pool = []
    for lv in LEVELS:
        pl = sorted((p for p in bp if lvl[p] == lv), key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        if len(pl) < PER_LEVEL:
            raise ValueError("level %s: %d players" % (lv, len(pl)))
        pool += pl[:PER_LEVEL]
    rng = random.Random("v2new|%d" % seed)
    games, cand_rows = [], []
    for p in pool:
        g = list(bp[p])
        rng.shuffle(g)
        games.append(g)
        cand_rows.append(g[:N_CAND_GAMES])
    ks = list(K_LIST)
    rng.shuffle(ks)
    nxt = [N_CAND_GAMES] * len(pool)
    owner, q_rows = [], []
    for k in ks:
        while True:
            o = rng.randrange(len(pool))
            if nxt[o] + k <= len(games[o]):
                break
        q_rows.append(games[o][nxt[o]:nxt[o] + k])
        nxt[o] += k
        owner.append(o)
    return {"pool": pool, "cand_rows": cand_rows, "q_rows": q_rows, "owner": np.asarray(owner), "k": np.asarray(ks),
            "seed": seed}


def shape_stats(store, rec):
    col = store.color
    cg = [r for g in rec["cand_rows"] for r in g]
    qg = [r for g in rec["q_rows"] for r in g]
    c = Counter(rec["owner"].tolist())
    return {"seed": rec["seed"], "candidates": len(rec["pool"]), "questions": len(rec["q_rows"]),
            "k": dict(sorted(Counter(rec["k"].tolist()).items())),
            "candidate_black_share": float(np.mean(col[cg] == "B")), "question_black_share": float(np.mean(col[qg] == "B")),
            "not_alternating": int((~store.alt[np.asarray(cg + qg)]).sum()),
            "candidates_without_question": int(len(rec["pool"]) - len(c)),
            "candidates_with_2plus": int(sum(1 for v in c.values() if v >= 2)),
            "overlap_free": len(set(cg + qg)) == len(cg) + len(qg)}


def pool_terms(store, ldas, seed):
    rec = v2new_pool(store, seed)
    models = {c: ldas[c].fit_excluding(store, set(rec["pool"])) for c in ("B", "W")}
    T = sim.pool_terms(store, rec, ldas, models, None, want=("O", "JO", "S2", "L", "PA"))
    return rec, T


def combo_score(T, owner, w, k):
    F = st.fuse([T[n] for n in TERMS], list(w))
    return st.score(st.csls(F, k), owner)


def by_k(T, owner, ks, w, k):
    F = st.csls(st.fuse([T[n] for n in TERMS], list(w)), k)
    pts = st.points(st.ranks(F, owner))
    return {str(kk): float(pts[ks == kk].mean()) for kk in sorted(set(ks.tolist()))}


def grid():
    for wl, wj, ws, wp, k in itertools.product(GRID["L"], GRID["JO"], GRID["S2"], GRID["PA"], GRID["k"]):
        yield (1, wl, wj, ws, wp), k


def cmd_pool(a):
    store = sim.Store(FEATS)
    print(json.dumps(shape_stats(store, v2new_pool(store, a.seed)), indent=1))
    return 0


def seeds_of(text):
    lo, hi = text.split("-")
    return list(range(int(lo), int(hi) + 1))


def cmd_terms(a):
    os.makedirs(a.out, exist_ok=True)
    store = sim.Store(FEATS)
    ldas = {c: sim.LdaData(FEATS, c, store) for c in ("B", "W")}
    for s in seeds_of(a.seeds):
        out = os.path.join(a.out, "pool_%03d.npz" % s)
        if os.path.exists(out):
            continue
        t = time.time()
        rec, T = pool_terms(store, ldas, s)
        arrs = {}
        for n in TERMS:
            arrs[n] = np.asarray(T[n][0], np.float32)
            arrs[n + "_ok"] = np.asarray(T[n][1], bool)
        np.savez(out + ".tmp.npz", owner=rec["owner"], k=rec["k"], **arrs)
        os.replace(out + ".tmp.npz", out)
        print(json.dumps({**shape_stats(store, rec), "seconds": round(time.time() - t, 1)}), flush=True)
    return 0


def load_terms(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.startswith("pool_") and f.endswith(".npz") and ".tmp" not in f:
            with np.load(os.path.join(d, f)) as z:
                T = {n: (np.asarray(z[n], np.float64), np.asarray(z[n + "_ok"])) for n in TERMS}
                out.append((int(f[5:8]), T, np.asarray(z["owner"]), np.asarray(z["k"])))
    return out


def cmd_tune(a):
    pools = load_terms(a.terms)
    rows = []
    for w, k in grid():
        rows.append((float(np.mean([combo_score(T, o, w, k) for _s, T, o, _k in pools])), w, k))
    best = max(rows, key=lambda r: r[0])  # first maximum in grid order
    fixed = {n: float(np.mean([combo_score(T, o, w, k) for _s, T, o, _k in pools])) for n, (w, k) in FIXED.items()}
    rec = {"format": "c10-prereg35 tune v1", "seeds": [s for s, *_ in pools], "grid_size": len(rows),
           "G": {"weights_O_L_JO_S2_PA": list(best[1]), "csls_k": best[2], "tune_mean": best[0]},
           "fixed_tune_means": fixed, "top10": [{"mean": r[0], "w": list(r[1]), "k": r[2]} for r in
                                                sorted(rows, key=lambda r: -r[0])[:10]]}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({"G": rec["G"], "fixed": fixed}, indent=1))
    return 0


def cmd_eval(a):
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    G = (tuple(tune["G"]["weights_O_L_JO_S2_PA"]), tune["G"]["csls_k"])
    pools = load_terms(a.terms)
    combos = dict(FIXED, G=G)
    per = {n: np.array([combo_score(T, o, w, k) for _s, T, o, _k in pools]) for n, (w, k) in combos.items()}
    d = per["G"] - per["B_k10"]
    rng = np.random.default_rng(BOOT_SEED)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    lo, hi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
    choose = "G" if lo > -0.005 else "B_k10"
    kk = {n: {s: v for s, v in zip(["1", "2", "3", "5"], np.mean([[by_k(T, o, ks, w, k)[x] for x in ("1", "2", "3", "5")]
                                                                  for _s, T, o, ks in pools], 0))}
          for n, (w, k) in combos.items()}
    rec = {"format": "c10-prereg35 eval v1", "seeds": [s for s, *_ in pools], "G": tune["G"],
           "means": {n: float(v.mean()) for n, v in per.items()}, "by_k": kk,
           "G_minus_B_k10": {"mean": float(d.mean()), "ci95": [lo, hi]}, "rule": "G if ci95 low > -0.005",
           "chosen": choose, "chosen_weights": list(combos[choose][0]), "chosen_csls_k": combos[choose][1]}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("means", "G_minus_B_k10", "chosen")}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("terms")
    t.add_argument("--seeds", required=True)
    t.add_argument("--out", required=True)
    u = sub.add_parser("tune")
    u.add_argument("--terms", required=True)
    u.add_argument("--out", default=os.path.join(HERE, "tune.json"))
    e = sub.add_parser("eval")
    e.add_argument("--terms", required=True)
    e.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    e.add_argument("--out", default=os.path.join(HERE, "eval.json"))
    a = ap.parse_args(argv)
    return {"pool": cmd_pool, "terms": cmd_terms, "tune": cmd_tune, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
