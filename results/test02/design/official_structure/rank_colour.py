"""Audit (read-only): does R20 (frozen prereg02 params) behave differently on single-colour
questions (official rank test structure) than on mixed-colour questions? Inner players only,
committed inner dumps / opening_inner.npz; no validation data, no label file.
In-sample for inner (R20 params were fit on inner), identical for both conditions."""
import json, os, sys, random
import numpy as np

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
import p02_rank_fuse as rf  # noqa

OUT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(REPO, "results", "prereg02", "rank")
keys = rf.load_inner_keys(os.path.join(D, "dumps"))
N = keys["n"]
dump = rf.load_dump(os.path.join(D, "dumps", "inner_p02e9_d4.npz"), N, 8)
lg = dump["logits"].astype(np.float64)
ok = np.isfinite(lg).all(axis=(1, 2))
lg[~ok] = 0.0
lp = rf.source_logprobs("C5", {"p02e9_d4": lg})
op = np.load(os.path.join(D, "opening_inner.npz"), allow_pickle=False)
at = {(r, g): i for i, (r, g) in enumerate(zip(op["rank"].tolist(), op["game_id"].tolist()))}
perm = np.array([at[(r, g)] for r, g in zip(keys["rank"], keys["game_id"])])
assert [str(c) for c in op["color"][perm]] == list(keys["color"])
o, has = op["o"][perm], op["has"][perm]
fz = json.load(open(os.path.join(D, "frozen_config.json"), encoding="utf-8"))
P = fz["params"]
B = rf.basis()
b = B @ np.asarray(P["beta"])
assert np.allclose(b, P["b"])

col = np.asarray(keys["color"])
lab = np.asarray(keys["label"])
pid = np.asarray(keys["player_id"])
by = {}
for i in range(N):
    if ok[i]:
        by.setdefault(pid[i], {"B": [], "W": []})[col[i]].append(i)
KS = (5, 10, 15, 20)


def qprob(idx, w, a, g, bb):
    idx = np.asarray(idx)
    m = lp[idx].mean(0)
    hh = idx[has[idx]]
    mop = o[hh].mean(0) if len(hh) else np.zeros(10)
    n = len(idx)
    t = np.exp(a) * (n / 12.0) ** g
    return rf.softmax(t * (m + w * mop + bb))


def score(questions, w, a, g, bb):
    pred, y = [], []
    for idx, lv in questions:
        pred.append(rf.rule_b_np(qprob(idx, w, a, g, bb)[None, :])[0])
        y.append(lv)
    pred, y = np.array(pred), np.array(y)
    sc = rf.official_score(pred, y)
    per = [sc[y == L].mean() for L in range(10) if (y == L).any()]
    return float(np.mean(per)), float(sc.mean()), len(y)


rng = random.Random(0)
eligible = sorted(p for p, d in by.items() if len(d["B"]) >= 20 and len(d["W"]) >= 20)
res = {"eligible_inner_players_ge20_each_colour": len(eligible), "rounds": 20}
Qs = {"mixed": [], "single_B": [], "single_W": [], "single_random_colour": []}
for r in range(20):
    for j, p in enumerate(eligible):
        lv = int(lab[by[p]["B"][0]])
        k = KS[(r + j) % 4]
        allg = by[p]["B"] + by[p]["W"]
        Qs["mixed"].append((rng.sample(allg, k), lv))
        Qs["single_B"].append((rng.sample(by[p]["B"], k), lv))
        Qs["single_W"].append((rng.sample(by[p]["W"], k), lv))
        c = rng.choice("BW")
        Qs["single_random_colour"].append((rng.sample(by[p][c], k), lv))
for name, (w, a, g, bb) in {"R20": (P["w"], P["a"], P["gamma"], b),
                            "R20_without_opening_term(w=0)": (0.0, P["a"], P["gamma"], b)}.items():
    res[name] = {q: dict(zip(("rank_balanced", "question_mean", "n"), score(v, w, a, g, bb)))
                 for q, v in Qs.items()}
# per-colour mean opening log-softmax and CNN mean level shift (bias check)
res["mean_expected_level_by_colour"] = {}
for c in "BW":
    m = (col == c) & ok
    pr = np.exp(lp[m])
    res["mean_expected_level_by_colour"][c] = {
        "cnn_mean_expected_level_minus_true": float((pr @ np.arange(10) - lab[m]).mean()),
        "opening_mean_expected_level_minus_true": float((np.exp(o[m & has]) @ np.arange(10) - lab[m & has]).mean()),
        "opening_has_share": float(has[m].mean())}
json.dump(res, open(os.path.join(OUT, "rank_colour.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
