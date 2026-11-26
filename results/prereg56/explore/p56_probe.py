"""prereg56 probe (PLAN.md): do the same people appear across pools / in the training set?  CPU only.

    PY results/prereg56/explore/p56_probe.py [--out results/prereg56/explore/probe.json]

Vectors: c49 per-game embeddings (L2). A set's vector = L2(mean). Statistic = max cosine to a gallery of prototypes
(or to c49's class weights). Null and positive control are simulated with E players (never seen by c49) at the same
set sizes / colours; D players (seen by c49) are the positive control for the class head.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TAG = "V2_step_320000_raw_tta1"
FEATS = os.path.join(REPO, "data", "prereg06", "feats")
V2 = os.path.join(REPO, "data", "test_v2_download", "x", "tests")
RK = os.path.join(REPO, "data", "test_v2_download", "rank")
csv.field_size_limit(1 << 30)


def l2(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def proto(emb, rows):
    return l2(emb[np.asarray(rows, np.int64)].mean(0))


def load_emb(path):
    with np.load(path, allow_pickle=False) as z:
        return l2(z["emb"]), (z["store_row"] if "store_row" in z.files else None)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pool_layout(cand_csv, q_csv):
    """Row layout of p36_lib.pool_items: candidate rows in file order, then every question's games."""
    cands = read_csv(cand_csv) if cand_csv else []
    qs = read_csv(q_csv)
    cid = [r["player_id"] for r in cands]
    ccol = [r["color"].upper() for r in cands]
    n = len(cands)
    cand = {}
    for i, p in enumerate(cid):
        cand.setdefault(p, []).append(i)
    q = {}
    col = list(ccol)
    for r in qs:
        g = []
        for i in range(1, int(r["num_games"]) + 1):
            g.append(n)
            col.append(r["color_%d" % i].upper())
            n += 1
        q[r["question_id"]] = g
    return cand, q, np.array(col), n


def store(path):
    emb, rows = load_emb(path)
    with np.load(os.path.join(FEATS, "meta.npz"), allow_pickle=False) as z:
        pid, col = z["player_id"][rows], z["color"][rows]
    by = {}
    for i, (p, c) in enumerate(zip(pid.tolist(), col.tolist())):
        by.setdefault(p, {"B": [], "W": []})[str(c).upper()].append(i)
    return emb, by


def pick(rng, by_p, k, colour, avoid=()):
    """k game rows of one player: colour 'B' = black only, 'any' = both colours pooled."""
    rows = by_p["B"] if colour == "B" else by_p["B"] + by_p["W"]
    rows = [r for r in rows if r not in avoid]
    if len(rows) < k:
        return None
    return rng.choice(rows, size=k, replace=False).tolist()


def simulate(emb, by, rng, kq, cq, ng, kg, cg, reps=3):
    """Null (query player not in the gallery) and positive (in the gallery, disjoint games) max-cosine samples."""
    players = sorted(by)
    null, pos, hit = [], [], []
    for _ in range(reps):
        rng.shuffle(players)
        gal, gp, used = [], [], {}
        for p in players:
            if len(gal) >= ng:
                break
            r = pick(rng, by[p], kg, cg)
            if r is None:
                continue
            gal.append(proto(emb, r))
            gp.append(p)
            used[p] = set(r)
        G = np.stack(gal)
        ins = set(gp)
        for j, p in enumerate(gp):
            r = pick(rng, by[p], kq, cq, avoid=used[p])
            if r is None:
                continue
            s = G @ proto(emb, r)
            pos.append(float(s.max()))
            hit.append(int(int(s.argmax()) == j))
        for p in players:
            if p in ins:
                continue
            r = pick(rng, by[p], kq, cq)
            if r is None:
                continue
            null.append(float((G @ proto(emb, r)).max()))
    return np.array(null), np.array(pos), float(np.mean(hit)) if hit else None


def report(name, scores, null, pos, hit, extra=None):
    scores = np.asarray(scores)
    out = {"n_sets": int(len(scores)), "n_null": int(len(null)), "n_pos": int(len(pos))}
    if len(null) and len(scores):
        t99, t95 = float(np.quantile(null, 0.99)), float(np.quantile(null, 0.95))
        out.update({"null_p50": float(np.median(null)), "null_p95": t95, "null_p99": t99,
                    "sets_median": float(np.median(scores)),
                    "frac_sets_above_null_p99": float((scores > t99).mean()),
                    "frac_sets_above_null_p95": float((scores > t95).mean()),
                    "pos_median": float(np.median(pos)) if len(pos) else None,
                    "frac_pos_above_null_p99": float((pos > t99).mean()) if len(pos) else None,
                    "pos_top1": hit})
    if extra:
        out.update(extra)
    print(name, json.dumps(out), flush=True)
    return out


def gallery_test(name, emb_sets, sets, emb_gal, gal, E, by, rng, kq_of, cq, kg, cg, res):
    """sets: {id: rows in emb_sets}; gal: {id: rows in emb_gal}. Null simulated per distinct query size."""
    G = np.stack([proto(emb_gal, r) for r in gal.values()])
    sc, ks = [], []
    for sid, rows in sets.items():
        if not rows:
            continue
        sc.append(float((G @ proto(emb_sets, rows)).max()))
        ks.append(kq_of(len(rows)))
    sc, ks = np.array(sc), np.array(ks)
    for k in sorted(set(ks.tolist())):
        null, pos, hit = simulate(E, by, rng, k, cq, min(len(G), 400), kg, cg)
        res["%s|k=%d" % (name, k)] = report("%s k=%d" % (name, k), sc[ks == k], null, pos, hit,
                                            {"gallery": int(len(G)), "kg": kg, "cq": cq, "cg": cg})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "probe.json"))
    ap.add_argument("--seed", type=int, default=56)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    res = {}
    E, byE = store(os.path.join(REPO, "data", "prereg49", "emb", TAG, "store_E.npz"))
    D, byD = store(os.path.join(REPO, "data", "prereg56", "emb_D", TAG, "store_D.npz"))
    import torch
    ck = torch.load(os.path.join(REPO, "data", "prereg49", "S0", "ckpt", "step_320000.pt"), map_location="cpu",
                    weights_only=False)
    W = l2(ck["head"]["weight"].numpy())
    del ck

    def head_scores(emb, sets):
        return np.array([float((W @ proto(emb, r)).max()) for r in sets.values() if r])

    def head_ctrl(emb, by, k, colour, n=600):
        out = []
        for p in sorted(by)[:n]:
            r = pick(rng, by[p], k, colour)
            if r is not None:
                out.append(float((W @ proto(emb, r)).max()))
        return np.array(out)

    cap = lambda n: n  # noqa: E731
    for split in ("public", "private"):
        o_emb, _ = load_emb(os.path.join(REPO, "data", "prereg56", "emb_x1", TAG, "pool_%s.npz" % split))
        o_c, o_q, o_col, o_n = pool_layout(os.path.join(V2, "player_identification_candidates_%s.csv" % split),
                                           os.path.join(V2, "player_identification_test_%s.csv" % split))
        assert o_n == len(o_emb), (o_n, len(o_emb))
        n_emb, _ = load_emb(os.path.join(REPO, "data", "prereg49", "emb", TAG, "pool_%s.npz" % split))
        n_c, n_q, n_col, n_n = pool_layout(os.path.join(V2, "player_identification_candidates2_%s.csv" % split),
                                           os.path.join(V2, "player_identification_test2_%s.csv" % split))
        assert n_n == len(n_emb), (n_n, len(n_emb))
        r_emb, _ = load_emb(os.path.join(REPO, "data", "prereg56", "emb_rank", TAG, "pool_%s.npz" % split))
        _x, r_q, r_col, r_n = pool_layout(None, os.path.join(RK, "rank_prediction_test_%s.csv" % split))
        assert r_n == len(r_emb), (r_n, len(r_emb))
        black = lambda sets, col: {s: [r for r in rows if col[r] == "B"] for s, rows in sets.items()}  # noqa: E731
        T = split
        # H3 / H3q: new candidates / new questions (black games) against the original candidates (100 black games)
        gallery_test("H3 %s newcand(B)->origcand" % T, n_emb, black(n_c, n_col), o_emb, o_c, E, byE, rng, cap, "B",
                     100, "B", res)
        gallery_test("H3q %s newq(B)->origcand" % T, n_emb, black(n_q, n_col), o_emb, o_c, E, byE, rng, cap, "B",
                     100, "B", res)
        # sanity / positive control on real data: original questions against original candidates (known one-to-one)
        gallery_test("CTRL %s origq->origcand" % T, o_emb, o_q, o_emb, o_c, E, byE, rng, lambda n: 5, "B", 100, "B",
                     res)
        # new questions against new candidates (the actual task) as a reference level
        gallery_test("REF %s newq->newcand" % T, n_emb, n_q, n_emb, n_c, E, byE, rng, cap, "any", 10, "any", res)
        # H5 (record only): comp-2 questions against comp-1 galleries of the same split
        gallery_test("H5 %s rankq(B)->origcand" % T, r_emb, black(r_q, r_col), o_emb, o_c, E, byE, rng,
                     lambda n: min(n, 10), "B", 100, "B", res)
        gallery_test("H5b %s rankq->newcand" % T, r_emb, r_q, n_emb, n_c, E, byE, rng, lambda n: 5 * ((n + 4) // 5),
                     "any", 10, "any", res)
        # H1: class head (training players seen by c49). Null = E, positive = D.
        for nm, emb, sets, k, col in (("origcand", o_emb, o_c, 100, "B"), ("newcand", n_emb, n_c, 10, "any"),
                                      ("rankq", r_emb, r_q, 10, "any"), ("newq", n_emb, n_q, 3, "any"),
                                      ("origq", o_emb, o_q, 10, "B")):
            null, pos = head_ctrl(E, byE, k, col), head_ctrl(D, byD, k, col)
            res["H1 %s %s" % (T, nm)] = report("H1 %s %s->trainhead" % (T, nm), head_scores(emb, sets), null, pos,
                                               None, {"k_ctrl": k})
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
