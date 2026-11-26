"""Audit (read-only on the repo): how the registered test-day checks and the competition-1 guard
behave on the OFFICIAL structure (candidates and questions black-only; questions num_games in
{5,10,15,20}; rank questions single-colour). Training players only (c10data.load_rows part
'train' = fit + inner through t02_checks.load_train / Train); no validation game is used for any
statistic, no label file is opened. Outputs only in this scratch folder."""
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
import c10data  # noqa
import t02_checks as tc  # noqa
import p02_player_open as po  # noqa
import scipy.sparse as sp  # noqa
from scipy.stats import binom  # noqa

OUT = os.path.dirname(os.path.abspath(__file__))
KS = (5, 10, 15, 20)
t0 = time.time()
res = {"what": __doc__}


def log(*a):
    print("[%5.0fs]" % (time.time() - t0), *a, flush=True)


rows = c10data.load_rows(tc.TRAIN_DIR, tc.SPLIT_CSV, part="train", local_dir=tc.LOCAL_DIR)
inner = c10data.inner_players(c10data.read_split(tc.SPLIT_CSV))
T = tc.Train(rows, inner, keys=False)
log("train parsed", T.n)
fit = T.fit
fitset = set(fit)
blacks = {p: [i for i in T.games_of[p] if T.colour[i] == "B"] for p in T.games_of}
nb = {p: len(v) for p, v in blacks.items()}
res["fit_players_by_black_games"] = {str(k): sum(1 for p in fit if nb[p] >= k) for k in (100, 105, 120, 200)}
res["fit_players_ge120_black_by_level"] = dict(Counter(T.level_of[p] for p in fit if nb[p] >= 120))
log(res["fit_players_by_black_games"], res["fit_players_ge120_black_by_level"])

# ------------------------------------------------------------------ (0) T1 on official-shaped files
import csv  # noqa


def write_csv(path, header, body):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(body)


r0 = np.random.default_rng(7)
e120 = [p for p in fit if nb[p] >= 120]
inner_ok = {c: [p for p in T.games_of if p in inner and sum(T.colour[i] == c for i in T.games_of[p]) >= 20] for c in "BW"}
res["T1_official_shape"] = {}
perm = [str(p) for p in r0.permutation(e120)]
plan = {}
gno = 0
priv_pl = perm[400:800] if len(perm) >= 800 else [str(p) for p in r0.permutation(e120)][:400]
res["T1_official_shape_note"] = "fit players with >= 120 black games: %d (private pool %s)" % (
    len(perm), "disjoint" if len(perm) >= 800 else "re-drawn from the same players")
for pool, pre, pl in (("public", "pub_", perm[:400]), ("private", "priv_", priv_pl)):
    cand, ques = [], []
    for j, p in enumerate(pl):
        g = [int(x) for x in r0.permutation(blacks[p])]
        alias = "%sp_%s" % (pre, hashlib.sha256(p.encode()).hexdigest()[:8])
        for i in g[:100]:
            gno += 1
            cand.append(("%sg_%06d" % (pre, gno), alias, i))
        ques.append(("%sq_%04d" % (pre, j + 1), "B", g[100:100 + KS[j % 4]]))
    rq = []
    for j in range(400):
        c = "B" if j % 2 == 0 else "W"
        p = inner_ok[c][int(r0.integers(len(inner_ok[c])))]
        gg = [i for i in T.games_of[p] if T.colour[i] == c]
        rq.append(("%sq_%04d" % (pre, j + 1), c, [int(x) for x in r0.choice(gg, KS[j % 4], replace=False)]))
    plan[pool] = (cand, ques, rq)


def qrow(qid, c, idx, fill):
    k = len(idx)
    return ([qid, str(k)] + [rows[i]["sgf_content"] for i in idx] + [""] * (20 - k)
            + [c] * k + ([c] * (20 - k) if fill else [""] * (20 - k)))


for variant, fill in (("A_unused_colour_empty", False), ("B_unused_colour_filled", True)):
    d = os.path.join(OUT, "t1_shape", variant)
    os.makedirs(d, exist_ok=True)
    for pool, (cand, ques, rq) in plan.items():
        write_csv(os.path.join(d, "player_identification_candidates_%s.csv" % pool), tc.C_COLS,
                  [[g, a, rows[i]["sgf_content"], "B"] for g, a, i in cand])
        write_csv(os.path.join(d, "player_identification_test_%s.csv" % pool), tc.Q_COLS,
                  [qrow(q, c, idx, fill) for q, c, idx in ques])
        write_csv(os.path.join(d, "rank_prediction_test_%s.csv" % pool), tc.Q_COLS,
                  [qrow(q, c, idx, fill) for q, c, idx in rq])
    out = {}
    for comp in ("player", "rank"):
        t1 = tc.t1_check(comp, d)
        out[comp] = {"verdict": t1["verdict"],
                     "subverdicts": {s: t1["subchecks"][s]["verdict"] for s in tc.T1_SUBS},
                     "T1.3_problems_first": t1["subchecks"]["T1.3"]["problems"][:2],
                     "T1.3_counts": {k: v.get("counts") for k, v in t1["subchecks"]["T1.3"]["files"].items()},
                     "T1.5_share": {k: v["share"] for k, v in t1["subchecks"]["T1.5"]["files"].items()},
                     "T1.4_share": {k: v["share"] for k, v in t1["subchecks"]["T1.4"]["files"].items()},
                     "normalize": t1["normalize"]}
    import t02_test_bundle as tb  # noqa
    out["t02_test_bundle.quick_structure"] = {c: tb.quick_structure(d, c)[:2] for c in ("player", "rank")}
    res["T1_official_shape"][variant] = out
    log("T1", variant, json.dumps(out)[:1500])
del rows

# ------------------------------------------------------------------ (1) T1.5 / T4 on black-only pools
S99 = set(tc.S99)
refs = json.load(open(os.path.join(REPO, "results", "test02", "null", "train_refs.json"), encoding="utf-8"))
elig100 = [p for p in fit if nb[p] >= 100]
elig120 = [p for p in fit if nb[p] >= 120]
rng = np.random.default_rng(0)


def raw_dist(idx):
    c = Counter(T.first_pt(i) for i in idx)
    t = sum(c.values())
    return {k: v / t for k, v in c.items()}


t4 = {"cand": [], "ques": []}
for rep in range(40):
    pl = list(rng.choice(elig120, 400, replace=False))
    cand, ques = [], []
    for j, p in enumerate(pl):
        g = list(rng.permutation(blacks[p]))
        cand += g[:100]
        ques += g[100:100 + KS[j % 4]]
    rest = None
    if rep < 10:
        ps = set(pl)
        rest = raw_dist([i for p in T.games_of if p not in ps for i in T.games_of[p]])
    for kind, idx in (("cand", cand), ("ques", ques)):
        d = {"s99": float(np.mean([tc.FIRST_CLASS[T.first_pt(i)] in S99 for i in idx])),
             "ur": float(np.mean([T.first_pt(i)[0] > "j" and T.first_pt(i)[1] < "j" for i in idx]))}
        if rest is not None:
            ft = raw_dist(idx)
            d["tvd"] = 0.5 * sum(abs(ft.get(k, 0) - rest.get(k, 0)) for k in set(ft) | set(rest))
        t4[kind].append(d)
res["black_only_pools_T1_5_T4"] = {
    kind: {"reps": len(v), "share_in_S99_min": min(x["s99"] for x in v), "share_in_S99_mean": float(np.mean([x["s99"] for x in v])),
           "upper_right_min": min(x["ur"] for x in v), "upper_right_max": max(x["ur"] for x in v),
           "tvd_mean": float(np.mean([x["tvd"] for x in v if "tvd" in x])), "tvd_max": max(x["tvd"] for x in v if "tvd" in x),
           "registered_mixed_null": {"share_in_S99_min": refs["t4_null_ranges"]["share_in_S99_min"][kind],
                                     "tvd_max": refs["t4_null_ranges"]["first_move_point_tvd_max"][kind],
                                     "upper_right": refs["t4_null_ranges"]["upper_right_share"][kind]}}
    for kind, v in t4.items()}
# per-player worst case: share of players whose own black first moves are mostly outside S99
own = [np.mean([tc.FIRST_CLASS[T.first_pt(i)] in S99 for i in blacks[p]]) for p in elig100]
res["black_only_per_player_S99_share"] = {"players": len(own), "p01": float(np.quantile(own, 0.01)),
                                          "min": float(min(own)), "n_below_0.5": int(sum(x < 0.5 for x in own))}
log("T1.5/T4", json.dumps(res["black_only_pools_T1_5_T4"]))

# ------------------------------------------------------------------ (2) guard rho on black-only pools
def moves(i):
    return T.mv60(i)


def bo_pool(seed, min_black=120):
    elig = {}
    for p in fit:
        if nb[p] >= min_black:
            elig.setdefault(T.level_of[p], []).append(p)
    pool = []
    for lv in c10data.LEVELS:
        pl = sorted(elig.get(lv, []), key=lambda p: hashlib.sha256(("%d|%s" % (seed, p)).encode()).hexdigest())
        pool += pl[:40]
    r = random.Random("bo|%d" % seed)
    cand, query, games = {}, {}, {}
    for j, p in enumerate(pool):
        g = list(blacks[p])
        r.shuffle(g)
        cand[p] = [T.gid[i] for i in g[:100]]
        query[p] = [T.gid[i] for i in g[100:100 + KS[j % 4]]]
        for i in g[:100 + KS[j % 4]]:
            games[(p, T.gid[i])] = ("B", moves(i))
    return pool, cand, query, games


def mixed_pool(seed):
    by_player, elig = {}, {}
    for p in fit:
        by_player[p] = [(T.gid[i], T.colour[i], None) for i in T.games_of[p]]
        if len(by_player[p]) >= po.CAL_MIN_GAMES:
            elig.setdefault(T.level_of[p], []).append(p)
    pool, cand, query = po.sim_pool(by_player, elig, seed, "random")
    at = {(p, T.gid[i]): i for p in pool for i in T.games_of[p]}
    games = {k: (T.colour[i], moves(i)) for k, i in at.items()}
    return pool, cand, query, games


guard = json.load(open(os.path.join(REPO, "results", "prereg02", "player", "guard_calibration.json"), encoding="utf-8"))["guard"]
res["guard_thresholds_registered"] = {o: {k: guard[o][k] for k in ("theta1", "theta2", "rho_R", "rho_F", "rho_X")} for o in po.ORIENTS}
res["guard_rho"] = {}
for label, fn in (("mixed_random_calibration_repro", mixed_pool), ("black_only_random_official_k", bo_pool)):
    out = []
    for seed in (0, 1, 2):
        pool, cand, query, games = fn(seed)
        st = po.pool_open_stats(pool, cand, query, games, strict=True)
        out.append({"seed": seed, "b_c": st["b_c"], "b_q": st["b_q"],
                    **{o: {k: st[o][k] for k in ("rho", "real_max_cos_median", "sr_max_cos_median",
                                                  "real_open_score", "sr_open_score")} for o in po.ORIENTS}})
        log(label, seed, json.dumps(out[-1]))
    res["guard_rho"][label] = out

# col / nocol duplication on a black-only pool: cosine with both sets == cosine with one set?
pool, cand, query, games = bo_pool(0)
keys = sorted(games)
row = {k: i for i, k in enumerate(keys)}
cg = [[row[(p, g)] for g in cand[p]] for p in sorted(pool)]
qg = [[row[(p, g)] for g in query[p]] for p in sorted(pool)]
dup = {}
for orient in po.ORIENTS:
    (G,) = po.token_matrices([[games[k] for k in keys]], orient)
    cos_both, _ = po.open_cos(G, cg, G, qg)
    vocab = {}
    rr, cc, vv = [], [], []
    for i, k in enumerate(keys):
        c, mv = games[k]
        for t, n in Counter(po.nocol_tokens(c, po.orient_moves(mv, orient))).items():
            rr.append(i); cc.append(vocab.setdefault(t, len(vocab))); vv.append(n)
    G1 = sp.csr_matrix((np.asarray(vv, float), (np.asarray(rr), np.asarray(cc))), shape=(len(keys), len(vocab)))
    cos_one, _ = po.open_cos(G1, cg, G1, qg)
    dup[orient] = {"max_abs_diff_cos_both_vs_colour_free_only": float(np.abs(cos_both - cos_one).max()),
                   "vocab_both": G.shape[1], "vocab_colour_free": G1.shape[1]}
res["black_only_token_set_duplication"] = dup
log("dup", dup)

# ------------------------------------------------------------------ (3) T3 null on black-only candidate profiles
null = json.load(open(os.path.join(REPO, "results", "test02", "null", "t3_null.json"), encoding="utf-8"))
order = T.t3_order()
res["T3"] = {}
for orient in tc.ORIENTS:
    ref = tc.t3_reference(T, orient)
    q99 = null["orient"][orient]["S2_q99"]
    assert ref["q99"] == q99, (ref["q99"], q99)
    Gm, idf, R, fcol = ref["Gm"], ref["idf"], ref["R"], ref["fcol"]
    el_b = [p for p in fit if nb[p] >= 100]
    prof = [[i for i in order[p] if T.colour[i] == "B"][:100] for p in el_b]
    Cb = tc.t3_weigh(po.indicator(prof, T.n) @ Gm, idf)
    COS = np.asarray((Cb @ R.T).toarray())
    own_c = np.array([fcol[p] for p in el_b])
    COSn = COS.copy()
    COSn[np.arange(len(el_b)), own_c] = -np.inf
    s1, s2 = tc.s2_rows(COSn)
    ps = [set(x) for x in prof]
    others = [[i for i in T.games_of[p] if i not in ps[j]] for j, p in enumerate(el_b)]
    Rp = tc.t3_weigh(po.indicator(others, T.n) @ Gm, idf)
    oc = np.asarray(Cb.multiply(Rp).sum(1)).ravel()
    C2 = COSn.copy()
    C2[np.arange(len(el_b)), own_c] = oc
    _, s2p = tc.s2_rows(C2)
    sh = float((s2 > q99).mean())
    shp = float((s2p > q99).mean())
    # pool rule: 400 candidates, a share pi of them training players (fit reference; p_ref as registered)
    rngp = np.random.default_rng(0)
    power = {}
    for pi in (0.0, 0.2, 0.3, 0.5, 1.0):
        hits = 0
        for _ in range(5000):
            k = rngp.binomial(rngp.binomial(400, pi), 5767 / 8007)
            v = np.concatenate([rngp.choice(s2p, k), rngp.choice(s2, 400 - k)])
            hits += (v > q99).sum() >= 40
        power[str(pi)] = hits / 5000
    res["T3"][orient] = {
        "registered_q99": q99, "registered_null_share_above": float((ref["s2n"] > q99).mean()),
        "registered_null_S1_median": float(np.median(ref["s1n"])),
        "black_only_null_players": len(el_b), "black_only_null_share_above_q99": sh,
        "black_only_null_P_declare_400_binomial": float(binom.sf(39, 400, sh)),
        "black_only_null_S2_quantiles": tc.quantiles(s2), "black_only_null_S1_median": float(np.median(s1)),
        "registered_null_S2_quantiles": tc.quantiles(ref["s2n"]),
        "black_only_positive_tpr": shp, "registered_positive_tpr_all": null["orient"][orient]["pos"]["all"]["tpr"],
        "black_only_P_declare_by_pi_single_orientation": power}
    log("T3", orient, json.dumps(res["T3"][orient]))
    del ref, COS, COSn, C2

res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(OUT, "sim_c1.json"), "w"), indent=1)
log("done")
