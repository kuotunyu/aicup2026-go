"""Audit (read-only): would a colour-matched T3 (reference = each fit player's BLACK fit games,
null / positives on black-only 100-game profiles) restore the power the registered T3 loses on
black-only candidate pools? Training players only (c10data.load_rows part 'train')."""
import json, os, sys, time
import numpy as np

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
import c10data  # noqa
import t02_checks as tc  # noqa
import p02_player_open as po  # noqa

OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
rows = c10data.load_rows(tc.TRAIN_DIR, tc.SPLIT_CSV, part="train", local_dir=tc.LOCAL_DIR)
inner = c10data.inner_players(c10data.read_split(tc.SPLIT_CSV))
T = tc.Train(rows, inner, keys=False)
del rows
fit = T.fit
order = T.t3_order()
blk = {p: [i for i in order[p] if T.colour[i] == "B"] for p in fit}
res = {}
for orient in tc.ORIENTS:
    Gm, vocab = tc.t3_token_matrix(T, orient)
    Rr = (po.indicator([blk[p] for p in fit], T.n) @ Gm).tocsr()
    df = np.diff(Rr.tocsc().indptr)
    idf = np.log((1.0 + len(fit)) / (1.0 + df)) + 1.0
    R = tc.t3_weigh(Rr, idf)
    fcol = {p: j for j, p in enumerate(fit)}
    el = [p for p in fit if len(blk[p]) >= 100]
    prof = [blk[p][:100] for p in el]
    C = tc.t3_weigh(po.indicator(prof, T.n) @ Gm, idf)
    COS = np.asarray((C @ R.T).toarray())
    own = np.array([fcol[p] for p in el])
    COSn = COS.copy()
    COSn[np.arange(len(el)), own] = -np.inf
    _, s2n = tc.s2_rows(COSn)
    q99 = tc.q99(s2n)
    out = {"null_players": len(el), "q99_black_ref": q99}
    for label, others in (("pos_other_black_games", [blk[p][100:] for p in el]),):
        sel = [j for j, g in enumerate(others) if len(g) >= 1]
        Rp = tc.t3_weigh(po.indicator([others[j] for j in sel], T.n) @ Gm, idf)
        oc = np.asarray(C[sel].multiply(Rp).sum(1)).ravel()
        C2 = COSn[sel].copy()
        C2[np.arange(len(sel)), own[sel]] = oc
        _, s2p = tc.s2_rows(C2)
        rng = np.random.default_rng(0)
        power = {}
        for pi in (0.2, 0.3, 0.5, 1.0):
            hits = 0
            for _ in range(5000):
                k = rng.binomial(rng.binomial(400, pi), 5767 / 8007)
                v = np.concatenate([rng.choice(s2p, k), rng.choice(s2n, 400 - k)])
                hits += (v > q99).sum() >= 40
            power[str(pi)] = hits / 5000
        out[label] = {"n": len(sel), "tpr": float((s2p > q99).mean()), "P_declare_by_pi": power}
    res[orient] = out
    print(orient, json.dumps(out), "%.0fs" % (time.time() - t0), flush=True)
json.dump(res, open(os.path.join(OUT, "sim_t3b.json"), "w"), indent=1)
