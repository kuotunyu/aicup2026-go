"""prereg36 explore: D tuning pools, E judge pools, checkpoint / weight selection, the one judgement, apply.

    PY results/prereg36/explore/p36_eval.py pools --which D      # seeds 100-109 -> data/prereg36/terms_D
    PY results/prereg36/explore/p36_eval.py pools --which E      # seeds 0-19    -> data/prereg36/terms_E
    PY results/prereg36/explore/p36_eval.py select               # -> tune.json   (commit it before judge)
    PY results/prereg36/explore/p36_eval.py judge                # -> eval.json   (once; needs tune.json committed)
    PY results/prereg36/explore/p36_eval.py apply --src <official tests dir> --pool public|private
    PY results/prereg36/explore/p36_eval.py assemble --v1 <v1 submission> --src <official tests dir>

Pools: the construction of p35_sim.v2new_pool restricted to a player set (test_p36.py checks that with all store
players it returns v2new_pool's pool). G terms (O, L, JO, S2, PA) by tools/p06_sim.pool_terms, LDA refitted per
pool without the pool's players. E36 = cosine(question vector, candidate vector), vector = mean of the game
vectors (p36_embed.py) then L2-normalised. M = G terms (weights of results/prereg35/explore/eval.json) + wE * z(E36).
No validation answers; fit players only.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p36_lib as L  # noqa: E402
import p06_sim as sim  # noqa: E402
import p06_style as st  # noqa: E402
import p35_sim as ps  # noqa: E402
from c10data import LEVELS  # noqa: E402

D_SEEDS = list(range(100, 110))
E_SEEDS = list(range(0, 20))
D_PER_LEVEL, E_PER_LEVEL = 30, 60
K_D = [1] * 50 + [2] * 50 + [3] * 50 + [5] * 50
K_E = list(ps.K_LIST)
WE_GRID = (0.5, 1, 2, 4, 8)
K_GRID = (10, 20, 40)
CNN_K = 10
BOOT = 10000
BOOT_SEED = 36
ADOPT_MEAN = 0.02
P35_EVAL = os.path.join(L.P35, "eval.json")
TERMS_DIR = {"D": os.path.join(L.DATA, "terms_D"), "E": os.path.join(L.DATA, "terms_E")}
OUT_APPLY = os.path.join(L.REPO, "results", "prereg36", "apply")


# ================================================================ pools
def make_pool(store, seed, players, per_level, k_list, allow_short=False):
    """p35_sim.v2new_pool with the players limited to `players`: per level the first `per_level` by
    sha256('{seed}|{pid}') (allow_short: all of them when the level has fewer), 10 shuffled games each as
    candidates, questions with k in k_list (shuffled), owners uniform with replacement, question games = the
    owner's next unused games; rng = random.Random('v2new|{seed}')."""
    bp = store.by_player()
    players = set(players)
    lvl = {p: str(store.rank[rows[0]]) for p, rows in bp.items() if p in players}
    pool = []
    for lv in LEVELS:
        pl = sorted((p for p in lvl if lvl[p] == lv),
                    key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        if len(pl) < per_level and not allow_short:
            raise ValueError("level %s: %d players" % (lv, len(pl)))
        pool += pl[:per_level]
    rng = random.Random("v2new|%d" % seed)
    games, cand_rows = [], []
    for p in pool:
        g = list(bp[p])
        rng.shuffle(g)
        games.append(g)
        cand_rows.append(g[:ps.N_CAND_GAMES])
    ks = list(k_list)
    rng.shuffle(ks)
    nxt = [ps.N_CAND_GAMES] * len(pool)
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


def pool_spec(which, E, D):
    if which == "D":
        return D_SEEDS, D, D_PER_LEVEL, K_D, True
    return E_SEEDS, E, E_PER_LEVEL, K_E, False


def cmd_pools(a):
    out_dir = a.out or TERMS_DIR[a.which]
    os.makedirs(out_dir, exist_ok=True)
    E, D, _T, _per = L.splits()
    seeds, players, per_level, k_list, short = pool_spec(a.which, E, D)
    if a.seeds:
        seeds = [s for s in ps.seeds_of(a.seeds) if s in seeds]
    store = sim.Store(L.FEATS)
    ldas = {c: sim.LdaData(L.FEATS, c, store) for c in ("B", "W")}
    for s in seeds:
        out = os.path.join(out_dir, "pool_%03d.npz" % s)
        if os.path.exists(out):
            continue
        t = time.time()
        rec = make_pool(store, s, players, per_level, k_list, short)
        assert set(rec["pool"]) <= set(players)
        models = {c: ldas[c].fit_excluding(store, set(rec["pool"])) for c in ("B", "W")}
        T = sim.pool_terms(store, rec, ldas, models, None, want=("O", "JO", "S2", "L", "PA"))
        arrs = {}
        for n in ps.TERMS:
            arrs[n] = np.asarray(T[n][0], np.float32)
            arrs[n + "_ok"] = np.asarray(T[n][1], bool)
        np.savez(out + ".tmp.npz", owner=rec["owner"], k=rec["k"], cand_rows=np.asarray(rec["cand_rows"], np.int64),
                 q_flat=np.asarray([r for g in rec["q_rows"] for r in g], np.int64), **arrs)
        os.replace(out + ".tmp.npz", out)
        print(json.dumps({**ps.shape_stats(store, rec), "which": a.which, "seconds": round(time.time() - t, 1)}),
              flush=True)
    return 0


def load_pools(d, seeds):
    out = []
    for s in seeds:
        f = os.path.join(d, "pool_%03d.npz" % s)
        if not os.path.exists(f):
            raise SystemExit("missing %s (run 'pools' first)" % f)
        with np.load(f) as z:
            T = {n: (np.asarray(z[n], np.float64), np.asarray(z[n + "_ok"])) for n in ps.TERMS}
            ks = np.asarray(z["k"])
            cut = np.cumsum(ks)[:-1]
            out.append({"seed": s, "T": T, "owner": np.asarray(z["owner"]), "k": ks,
                        "cand_rows": [r.tolist() for r in np.asarray(z["cand_rows"])],
                        "q_rows": [r.tolist() for r in np.split(np.asarray(z["q_flat"]), cut)]})
    return out


# ================================================================ E36
class Emb:
    """Per-game vectors of p36_embed.py addressed by feature-store row (or by item number for official pools)."""

    def __init__(self, emb, count, rows=None, colour=None, sha=None, name=None):
        self.emb = np.asarray(emb, np.float64)
        self.count = np.asarray(count)
        rows = np.arange(len(self.emb)) if rows is None else np.asarray(rows)
        self.pos = {int(r): i for i, r in enumerate(rows.tolist())}
        self.colour = colour          # {row: 'B' / 'W'} or an array indexed by row
        self.sha, self.name = sha, name

    @classmethod
    def load(cls, path, colour=None):
        with np.load(path, allow_pickle=False) as z:
            return cls(z["emb"], z["count"], z["store_row"] if "store_row" in z.files else None, colour,
                       str(z["ckpt_sha256"]), str(z["ckpt"]))

    def group_vecs(self, groups, colour=None):
        """L2-normalised mean of the game vectors of every group (games with a sample; optionally one colour
        only); has[i] False and a zero vector when the group has none."""
        V = np.zeros((len(groups), self.emb.shape[1]))
        has = np.zeros(len(groups), bool)
        for i, g in enumerate(groups):
            idx = [self.pos[int(r)] for r in g if colour is None or self.colour[int(r)] == colour]
            idx = [j for j in idx if self.count[j] > 0]
            if idx:
                v = self.emb[idx].mean(0)
                nv = np.linalg.norm(v)
                if nv > 0:
                    V[i] = v / nv
                    has[i] = True
        return V, has


def e36(emb, pool):
    """(questions x candidates cosine, question has a vector)."""
    C, _hc = emb.group_vecs(pool["cand_rows"])
    Q, hq = emb.group_vecs(pool["q_rows"])
    return Q @ C.T, hq


def e36_colour(emb, pool, cross):
    """Record only: per colour c the question's colour-c games against the candidates' games of the same colour
    (cross: of the other colour); the colours of a question are averaged with its numbers of games as weights."""
    nq, nc = len(pool["q_rows"]), len(pool["cand_rows"])
    M, wsum = np.zeros((nq, nc)), np.zeros(nq)
    for c in ("B", "W"):
        Q, hq = emb.group_vecs(pool["q_rows"], c)
        C, _hc = emb.group_vecs(pool["cand_rows"], ({"B": "W", "W": "B"}[c]) if cross else c)
        w = np.array([sum(1 for r in g if emb.colour[int(r)] == c) for g in pool["q_rows"]], float) * hq
        M += w[:, None] * (Q @ C.T)
        wsum += w
    ok = wsum > 0
    M[ok] /= wsum[ok, None]
    return M, ok


def fused(pool, E, w, wE):
    return st.fuse([pool["T"][n] for n in ps.TERMS] + [E], list(w) + [wE])


def score_by_k(S, pool):
    pts = st.points(st.ranks(S, pool["owner"]))
    return {str(kk): float(pts[pool["k"] == kk].mean()) for kk in sorted(set(pool["k"].tolist()))}


def mean_by_k(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def g_combo():
    with open(P35_EVAL, encoding="utf-8") as f:
        ev = json.load(f)
    return tuple(ev["chosen_weights"]), ev["chosen_csls_k"]


def bootstrap(d, n=BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def tune_grid(pools, embs):
    """[(mean M score on the pools, wE, k)] in grid order (wE outer, k inner)."""
    w, _k = g_combo()
    rows = []
    for wE in WE_GRID:
        Fs = [fused(p, e, w, wE) for p, e in zip(pools, embs)]
        for k in K_GRID:
            rows.append((float(np.mean([st.score(st.csls(F, k), p["owner"]) for F, p in zip(Fs, pools)])), wE, k))
    return rows


def judge_numbers(pools, embs, wE, kM, colour_terms=None):
    """Per-pool M and G scores and everything eval.json records. colour_terms: [(same, cross)] per pool."""
    w, kG = g_combo()
    M, G, C, bM, bG, bC = [], [], [], [], [], []
    for p, e in zip(pools, embs):
        SM = st.csls(fused(p, e, w, wE), kM)
        SG = st.csls(st.fuse([p["T"][n] for n in ps.TERMS], list(w)), kG)
        SC = st.csls(e[0], CNN_K)
        M.append(st.score(SM, p["owner"]))
        G.append(st.score(SG, p["owner"]))
        C.append(st.score(SC, p["owner"]))
        bM.append(score_by_k(SM, p))
        bG.append(score_by_k(SG, p))
        bC.append(score_by_k(SC, p))
    M, G, C = np.array(M), np.array(G), np.array(C)
    d = M - G
    lo, hi = bootstrap(d)
    rec = {"means": {"M": float(M.mean()), "G": float(G.mean()), "E36_only_csls%d" % CNN_K: float(C.mean())},
           "per_pool": {"M": M.tolist(), "G": G.tolist(), "E36_only": C.tolist()},
           "M_minus_G": {"mean": float(d.mean()), "ci95": [lo, hi], "pools_positive": int((d > 0).sum())},
           "by_k": {"M": mean_by_k(bM), "G": mean_by_k(bG), "E36_only": mean_by_k(bC)},
           "rule": "adopt M if mean(M - G) >= %.2f and ci95 low > 0 and M > G in >= 18 of 20 pools (pool bootstrap, %d samples, seed %d)"
                   % (ADOPT_MEAN, BOOT, BOOT_SEED),
           "adopt": bool(d.mean() >= ADOPT_MEAN and lo > 0 and int((d > 0).sum()) >= int(np.ceil(0.9 * len(d))))}
    if colour_terms is not None:
        rec["E36_only_by_colour_csls%d" % CNN_K] = {
            name: float(np.mean([st.score(st.csls(ct[i][0], CNN_K), p["owner"]) for ct, p in
                                 zip(colour_terms, pools)])) for i, name in enumerate(("same_colour", "cross_colour"))}
    return rec


# ================================================================ select / judge
def emb_path(stem, name):
    return os.path.join(L.EMB, stem, name + ".npz")


def cmd_select(a):
    with open(a.train_record, encoding="utf-8") as f:
        tr = json.load(f)
    if not tr.get("finished") or tr.get("smoke"):
        raise SystemExit("train_record.json is not a finished full training run")
    pools = load_pools(TERMS_DIR["D"], D_SEEDS)
    cnn = {}
    for name in tr["regular_checkpoints"]:
        stem = os.path.splitext(name)[0]
        f = emb_path(stem, "store_D")
        if not os.path.exists(f):
            raise SystemExit("missing %s (run p36_embed.py --source store --players D for every checkpoint)" % f)
        e = Emb.load(f)
        if e.sha != tr["checkpoints"][name]:
            raise SystemExit("%s was not made from the recorded checkpoint %s" % (f, name))
        cnn[stem] = float(np.mean([st.score(st.csls(e36(e, p)[0], CNN_K), p["owner"]) for p in pools]))
    best = max(sorted(cnn), key=lambda s: cnn[s])          # ties: the earliest checkpoint
    e = Emb.load(emb_path(best, "store_D"))
    embs = [e36(e, p) for p in pools]
    rows = tune_grid(pools, embs)
    top = max(rows, key=lambda r: r[0])                    # first maximum in grid order
    w, kG = g_combo()
    g_mean = float(np.mean([st.score(st.csls(st.fuse([p["T"][n] for n in ps.TERMS], list(w)), kG), p["owner"])
                            for p in pools]))
    rec = {"format": "c10-prereg36 tune v1", "d_seeds": D_SEEDS, "pool_shape": {
        "candidates": [len(p["cand_rows"]) for p in pools], "questions": [len(p["q_rows"]) for p in pools]},
        "cnn_only_csls%d_by_checkpoint" % CNN_K: cnn,
        "checkpoint": {"name": best + ".pt", "sha256": e.sha, "cnn_only_mean": cnn[best]},
        "G": {"weights_O_L_JO_S2_PA": list(w), "csls_k": kG, "p35_eval_sha256": L.sha256_file(P35_EVAL),
              "mean_on_D_pools": g_mean},
        "grid": [{"mean": r[0], "wE": r[1], "k": r[2]} for r in rows],
        "M": {"wE": top[1], "csls_k": top[2], "tune_mean": top[0]},
        "train_record_sha256": L.sha256_file(a.train_record)}
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("checkpoint", "M", "G")}, indent=1))
    return 0


def cmd_judge(a):
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the judgement is made once" % a.out)
    if not L.git_committed(a.tune):
        raise SystemExit("tune.json is not committed (or differs from HEAD); commit it before the judgement")
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    stem = os.path.splitext(tune["checkpoint"]["name"])[0]
    store = sim.Store(L.FEATS, load=())
    e = Emb.load(emb_path(stem, "store_E"), colour=store.color)
    if e.sha != tune["checkpoint"]["sha256"]:
        raise SystemExit("store_E embeddings were not made from the chosen checkpoint")
    pools = load_pools(TERMS_DIR["E"], E_SEEDS)
    E, D, T, _per = L.splits()
    Es = set(E)
    for p in pools:
        rows = {r for g in p["cand_rows"] + p["q_rows"] for r in g}
        if not {str(x) for x in store.pid[np.asarray(sorted(rows))].tolist()} <= Es:
            raise SystemExit("pool %d has players outside E" % p["seed"])
    embs = [e36(e, p) for p in pools]
    col = [(e36_colour(e, p, False), e36_colour(e, p, True)) for p in pools]
    rec = {"format": "c10-prereg36 eval v1", "e_seeds": E_SEEDS, "checkpoint": tune["checkpoint"],
           "M": tune["M"], "G": tune["G"], "tune_sha256": L.sha256_file(a.tune), "git_head": L.git_head()}
    rec.update(judge_numbers(pools, embs, tune["M"]["wE"], tune["M"]["csls_k"], col))
    rec["questions_without_E36"] = int(sum((~ok).sum() for _m, ok in embs))
    L.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("means", "M_minus_G", "by_k", "adopt")}, indent=1))
    return 0


# ================================================================ official pools
def cmd_apply(a):
    import p35_apply as pa
    t0 = time.time()
    with open(a.tune, encoding="utf-8") as f:
        tune = json.load(f)
    with open(a.eval, encoding="utf-8") as f:
        ev = json.load(f)
    if not ev["adopt"] or ev["tune_sha256"] != L.sha256_file(a.tune):
        raise SystemExit("eval.json does not adopt M (or belongs to another tune.json): keep prereg35's G")
    w, wE, k = tuple(tune["G"]["weights_O_L_JO_S2_PA"]), tune["M"]["wE"], tune["M"]["csls_k"]
    stem = os.path.splitext(tune["checkpoint"]["name"])[0]
    cands, qs = pa.read_pool(a.src, a.pool)
    items, cand_ids, cand_rows, q_rows, qids = L.pool_items(cands, qs)
    if not all(c.startswith(pa.PREFIX[a.pool] + "p_") for c in cand_ids):
        raise SystemExit("candidate id prefix")
    e = Emb.load(emb_path(stem, "pool_%s" % a.pool))
    if e.sha != tune["checkpoint"]["sha256"] or len(e.emb) != len(items):
        raise SystemExit("pool embeddings do not match the chosen checkpoint / the pool's games")
    fit = sim.Store(L.FEATS, load=())
    store = pa.PoolStore(items, fit.csr("resp").vocab)
    ldas, models = {}, {}
    for c in ("B", "W"):
        ld = sim.LdaData(L.FEATS, c, fit)
        models[c] = ld.fit_excluding(fit, set())
        ldas[c] = pa.PoolLda(store, ld.cols)
    rec = {"cand_rows": cand_rows, "q_rows": q_rows}
    T = sim.pool_terms(store, rec, ldas, models, None, want=("O", "JO", "S2", "L", "PA"))
    E = e36(e, rec)
    S = st.csls(fused({"T": T}, E, w, wE), k)
    tops = st.top5(S, cand_ids)
    od = os.path.join(a.out, a.pool)
    os.makedirs(od, exist_ok=True)
    path = os.path.join(od, "top5.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"])
        for q, t in zip(qids, tops):
            wr.writerow([q] + t)
    rep = {"format": "c10-prereg36 apply v1", "pool": a.pool, "weights_O_L_JO_S2_PA": list(w), "wE": wE, "csls_k": k,
           "checkpoint": tune["checkpoint"], "questions": len(qids), "candidates": len(cand_ids),
           "games": len(items), "games_without_E36": int((e.count == 0).sum()),
           "questions_without_E36": int((~E[1]).sum()),
           "terms_rows_without_signal": {n: int((~np.asarray(T[n][1])).sum()) for n in ps.TERMS},
           "tune_sha256": L.sha256_file(a.tune), "eval_sha256": L.sha256_file(a.eval),
           "top5_sha256": L.sha256_file(path), "seconds": round(time.time() - t0, 1)}
    L.write_json(os.path.join(od, "apply_report.json"), rep)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_assemble(a):
    import p35_apply as pa
    return pa.cmd_assemble(a)          # the same checks and file layout, under results/prereg36/apply


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pools")
    p.add_argument("--which", required=True, choices=["D", "E"])
    p.add_argument("--seeds", default=None, help="a sub-range such as 100-104 (to run parts in parallel)")
    p.add_argument("--out", default=None)
    s = sub.add_parser("select")
    s.add_argument("--train-record", default=os.path.join(HERE, "train_record.json"))
    s.add_argument("--out", default=os.path.join(HERE, "tune.json"))
    j = sub.add_parser("judge")
    j.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    j.add_argument("--out", default=os.path.join(HERE, "eval.json"))
    q = sub.add_parser("apply")
    q.add_argument("--src", required=True)
    q.add_argument("--pool", required=True, choices=["public", "private"])
    q.add_argument("--tune", default=os.path.join(HERE, "tune.json"))
    q.add_argument("--eval", default=os.path.join(HERE, "eval.json"))
    q.add_argument("--out", default=OUT_APPLY)
    b = sub.add_parser("assemble")
    b.add_argument("--v1", required=True)
    b.add_argument("--src", required=True)
    b.add_argument("--out", default=OUT_APPLY)
    a = ap.parse_args(argv)
    return {"pools": cmd_pools, "select": cmd_select, "judge": cmd_judge, "apply": cmd_apply,
            "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
