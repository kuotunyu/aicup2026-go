"""prereg10 explore x1 (competition 1): how 06 and its parts degrade in harder mock pools of fit players.

    Git Bash, repository root:
    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe results/prereg10/explore/x1_scenarios.py \
        [--seeds 1000-1019] [--scen base,wr,...] [--workers 6] [--out results/prereg10/explore/x1_scenarios.json]

Only fit players, through the PREREG_06 machinery (tools/p06_sim.py, p06_style.py; imported read-only, bytecode off):
the fit feature store data/prereg06/feats, the fit CNN sums data/prereg06/cnn_fit/black_ge105.npz (in-sample for fit
players: the CNN was trained on all training players), the LDA refit per pool on the black LDA set minus the pool's
players (as PREREG_06 1.6). Opening keys: data/prereg10/tmp/open_keys.npz (x0_open_keys.py). The p06 data/local guard
is installed; data/local is never touched; tools/score.py is never run.

Official structure of competition 1 (docs/DATA.md): 400 candidates x 100 black games, 400 questions of 5/10/15/20 black
games (100 each). Scenarios (all black only):
  base     official shape, random split, one question per candidate (o2o)           = sim.make_pool(seed, 'o2o')
  wr       question owners drawn with replacement (~250 distinct owners)            = sim.make_pool(seed, 'wr')
  ord      file-order split (candidates = first 100 black games, questions next)   = sim.make_pool(seed, 'o2o', 'ord')
  late     file order, candidates = first 100, questions = the LAST k games of the player's list
  nov6     question games restricted to games whose first 6 plies (8-symmetry canonical) never occur among the owner's
           100 candidate games' first 6 plies ("no shared opening"); nov4 / nov8 / nov10 = 4 / 8 / 10 plies
  k1,k2,k3,k5   every question has exactly k games (o2o)
  nov6k3   nov6 with k = 3 for every question
  c20,c50  only 20 / 50 candidate games per player
  big600, big800  600 / 800 candidates (fit players with >= 105 black games, chosen by hash), 400 question owners among
           those with >= 120 black games; LDA refit on the remaining players
For nov*: a question whose owner has fewer novel games than k uses all of them (k_eff < k); an owner with none is
replaced by the nearest (hash order) pool player that still has >= k unused novel games (structure then not o2o).

Methods (per pool, same terms): E0 (= Q00, CNN alone), O (opening tokens), L (LDA), JO, S2, Q04 = z(E0)+z(O),
V0c = Q04+CSLS10, V2 = 06 main = E0+O+2.66L+0.5JO+0.5S2 + CSLS10 (switch off), V2nc (no CSLS), V2noOpen = E0+2.66L+
0.5S2 + CSLS10 (no opening-token terms), V2noE = V2 without E0. Also the PREREG_02 split guard: rho_raw = median over
questions of max cosO_raw (full candidate games) / the same median on the sr06 pseudo pools (rounds 0-4) and the
guard level with the frozen prereg02 raw thresholds (theta1 0.9630, theta2 0.8268); '06eff' = V2 when level <= 2,
E0 (CNN only, the test02 fallback) when level 3.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TOOLS = os.path.join(REPO, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

FEATS = os.path.join(REPO, "data", "prereg06", "feats")
CNN = os.path.join(REPO, "data", "prereg06", "cnn_fit", "black_ge105.npz")
KEYS = os.path.join(REPO, "data", "prereg10", "tmp", "open_keys.npz")
THETA1, THETA2 = 0.9630063040189639, 0.8267520442154412       # results/prereg02/player/apply_report.json, raw
ALL_SCEN = ("base", "wr", "ord", "late", "nov4", "nov6", "nov8", "nov10", "k1", "k2", "k3", "k5", "nov6k3",
            "c20", "c50", "big600", "big800")
W = {}


def thread_env(n):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(n)


def _init(threads):
    thread_env(threads)
    sys.dont_write_bytecode = True
    import numpy as np
    import p06_sim as sim
    import p06_style as st
    st.install_guard()
    store = sim.Store(FEATS)
    W.update(np=np, sim=sim, st=st, store=store, lda={"B": sim.LdaData(FEATS, "B", store)},
             cnn=sim.Cnn(CNN, store))
    with np.load(KEYS, allow_pickle=False) as z:
        W["keys"] = {k: z[k] for k in z.files}


def _hash(seed, p, tag=""):
    return hashlib.sha256(("%s%d|%s" % (tag, seed, p)).encode()).hexdigest()


def build(seed, scen):
    np, sim = W["np"], W["sim"]
    store = W["store"]
    cache = store.black_cache()
    if scen == "base":
        return sim.make_pool(cache, seed, "o2o")
    if scen == "wr":
        return sim.make_pool(cache, seed, "wr")
    if scen == "ord":
        return sim.make_pool(cache, seed, "o2o", split="ord")
    if scen.startswith("big"):
        N = int(scen[3:])
        allp = sorted(cache["by_player"], key=lambda p: _hash(seed, p, "big|"))
        pool = allp[:N]
        owners = [p for p in pool if cache["n_black"][p] >= 120][:400]
        ks = sim.k_list(400)
        random.Random("k|%d" % seed).shuffle(ks)
        rng = random.Random("pool|%d" % seed)
        cand_rows, q_rows, owner = [], [], []
        idx = {p: i for i, p in enumerate(pool)}
        queue = {}
        for p in pool:
            g = list(cache["by_player"][p])
            rng.shuffle(g)
            cand_rows.append(g[:100])
            queue[p] = g[100:]
        for q, p in enumerate(owners):
            q_rows.append(queue[p][:ks[q]])
            owner.append(idx[p])
        return {"pool": pool, "cand_rows": cand_rows, "q_rows": q_rows, "owner": np.asarray(owner),
                "k": np.asarray(ks), "n_repairs": 0, "scenario": scen, "seed": seed}
    pool = sim.choose_pool(cache, seed)
    n = len(pool)
    ks = sim.k_list(n)
    random.Random("k|%d" % seed).shuffle(ks)
    kfix = None
    if scen in ("k1", "k2", "k3", "k5"):
        kfix = int(scen[1:])
    if scen == "nov6k3":
        kfix = 3
    if kfix is not None:
        ks = [kfix] * n
    ncand = 100
    if scen in ("c20", "c50"):
        ncand = int(scen[1:])
    rng = random.Random("pool|%d" % seed)
    cand_rows, queues = [], []
    for p in pool:
        g = list(cache["by_player"][p])
        rng.shuffle(g)
        if scen == "late":
            g = list(cache["by_player"][p])
        cand_rows.append(g[:ncand])
        queues.append(g[ncand:])
    owner = list(range(n))
    info = {}
    if scen == "late":
        q_rows = [queues[i][-ks[i]:] for i in range(n)]
    elif scen.startswith("nov"):
        plies = 6 if scen == "nov6k3" else int(scen[3:])
        key = W["keys"]["c%d" % plies]
        nov = []
        for i in range(n):
            ck = {key[r] for r in cand_rows[i]}
            nov.append([r for r in queues[i] if key[r] not in ck])
        used = [0] * n
        q_rows = [None] * n
        k_eff = list(ks)
        n_replaced = 0
        order = sorted(range(n), key=lambda i: -ks[i])       # big questions first
        for i in order:
            if len(nov[i]) - used[i] >= 1:
                take = min(ks[i], len(nov[i]) - used[i])
                q_rows[i] = nov[i][used[i]:used[i] + take]
                used[i] += take
                k_eff[i] = take
                continue
            # no novel game left: give the question to another pool player with >= k unused novel games
            cands = sorted((j for j in range(n) if len(nov[j]) - used[j] >= ks[i]),
                           key=lambda j: _hash(seed, pool[j] + "|%d" % i, "repl|"))
            if not cands:
                cands = sorted((j for j in range(n) if len(nov[j]) - used[j] >= 1),
                               key=lambda j: _hash(seed, pool[j] + "|%d" % i, "repl|"))
            j = cands[0]
            take = min(ks[i], len(nov[j]) - used[j])
            q_rows[i] = nov[j][used[j]:used[j] + take]
            used[j] += take
            owner[i] = j
            k_eff[i] = take
            n_replaced += 1
        info = {"n_novel_games_per_player": {"min": int(min(map(len, nov))), "median": float(np.median(
            [len(x) for x in nov])), "zero": int(sum(1 for x in nov if not x)),
            "lt5": int(sum(1 for x in nov if len(x) < 5))},
            "n_replaced_owner": n_replaced, "n_distinct_owner": len(set(owner)),
            "k_nominal": ks, "mean_k_eff": float(np.mean(k_eff))}
        ks = k_eff
    else:
        q_rows = [queues[i][:ks[i]] for i in range(n)]
    rec = {"pool": pool, "cand_rows": cand_rows, "q_rows": q_rows, "owner": np.asarray(owner),
           "k": np.asarray(ks), "n_repairs": 0, "scenario": scen, "seed": seed, "info": info}
    return rec


def metrics(S, owner, ks):
    np, st = W["np"], W["st"]
    r = st.ranks(S, owner)
    pts = st.points(r)
    out = {"score": float(pts.mean()), "top1": float((r == 1).mean()), "top5": float((r <= 5).mean())}
    by = {}
    for k in sorted(set(int(x) for x in ks)):
        m = np.asarray(ks) == k
        by[str(k)] = round(float(pts[m].mean()), 6)
    out["by_k"] = by
    return out, pts


def rho_raw(rec):
    """PREREG_02 rho (raw orientation), denominator from the sr06 pseudo pools (rounds 0-4) of the same pool."""
    np, sim, st = W["np"], W["sim"], W["st"]
    store = W["store"]
    cos, has = sim.term_O(store, rec)
    num = float(np.median(cos.max(1)[has]))
    # the test-day denominator does not see the questions' k: pseudo questions use the official 5/10/15/20 mix
    prs, _ids = sim.sr_pool(store, dict(rec, k=np.asarray(sim.k_list(400))), rounds=(0, 1, 2, 3, 4))
    mx = []
    for pr in prs:
        c2, h2 = sim.term_O(store, pr)
        mx.append(c2.max(1)[h2])
    den = float(np.median(np.concatenate(mx)))
    rho = num / den
    level = 1 if rho >= THETA1 else (2 if rho >= THETA2 else 3)
    return {"num": num, "den": den, "rho": rho, "level": level}


def job(args):
    seed, scen = args
    np, sim, st = W["np"], W["sim"], W["st"]
    t = time.time()
    store = W["store"]
    rec = build(seed, scen)
    models = {"B": W["lda"]["B"].fit_excluding(store, set(rec["pool"]))}
    T = sim.pool_terms(store, rec, W["lda"], models, W["cnn"], want=("O", "JO", "S2", "L", "E0"), colours=("B",))
    z = {k: st.zscore_rows(*T[k]) for k in ("E0", "O", "L", "JO", "S2")}
    F = dict(z)
    F["Q04"] = sim.combo_F(T, "E0", None)
    F["V0c"] = st.csls(F["Q04"], 10)
    v2 = sim.combo_F(T, "E0", sim.V2_W)
    F["V2nc"] = v2
    F["V2"] = st.csls(v2, 10)
    F["V2noOpen"] = st.csls(st.fuse([T["E0"], T["L"], T["S2"]], [1.0, 2.66, 0.5]), 10)
    F["V2noE"] = st.csls(sim.combo_F(T, None, sim.V2_W, open_only=True), 10)
    own, ks = rec["owner"], rec["k"]
    res, pts = {}, {}
    for name, S in F.items():
        res[name], pts[name] = metrics(S, own, ks)
    g = None
    if scen not in ("c20", "c50"):
        g = rho_raw(rec)
        res["06eff"] = res["V2"] if g["level"] <= 2 else res["E0"]
        pts["06eff"] = pts["V2"] if g["level"] <= 2 else pts["E0"]
    out = {"seed": seed, "scen": scen, "res": res, "guard": g, "info": rec.get("info", {}),
           "n_cand": len(rec["cand_rows"]), "n_q": len(rec["q_rows"]),
           "n_distinct_owner": int(len(set(own.tolist()))), "lda_train": [models["B"]["n_players"],
                                                                         models["B"]["n_games"]],
           "pts": {k: np.round(v, 6).tolist() for k, v in pts.items()}, "k": [int(x) for x in ks],
           "seconds": round(time.time() - t, 1)}
    return out


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1000-1019")
    ap.add_argument("--scen", default=",".join(ALL_SCEN))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "prereg10", "tmp", "x1_rows.json"))
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    scens = a.scen.split(",")
    jobs = [(s, c) for c in scens for s in seeds]
    t0 = time.time()
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    thread_env(a.threads)
    pool = Pool(min(a.workers, 6), initializer=_init, initargs=(a.threads,))
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    rows = []
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
    have = {(r["seed"], r["scen"]) for r in rows}
    jobs = [j for j in jobs if j not in have]
    with pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            rows.append(r)
            print("%s seed %d: V2 %.4f E0 %.4f O %.4f rho %s (%.0fs) %d/%d" % (
                r["scen"], r["seed"], r["res"]["V2"]["score"], r["res"]["E0"]["score"], r["res"]["O"]["score"],
                "%.3f" % r["guard"]["rho"] if r["guard"] else "-", r["seconds"], len(rows), len(jobs) + len(have)),
                flush=True)
            if len(rows) % 10 == 0:
                _save(a.out, rows)
    _save(a.out, rows)
    print("done", round(time.time() - t0, 1), "s")


def _save(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
