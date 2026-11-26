"""Task B extra: do 2+-level errors concentrate in a few inner players? (r20_oof.npz only)"""
import json, os
import numpy as np
from common import F, L, LEVELS, OUT, rule_b, score
R = np.load(os.path.join(OUT, "r20_oof.npz"))
y, pl, P = R["y"], R["player"], R["P20"]
pred = rule_b(P)
d = np.abs(pred - y)
out = {}
npl = pl.max() + 1
far2 = np.bincount(pl, weights=(d >= 2), minlength=npl)
far3 = np.bincount(pl, weights=(d >= 3), minlength=npl)
cnt = np.bincount(pl, minlength=npl)
lvl = np.zeros(npl, int); lvl[pl] = y
rate2 = far2 / np.maximum(cnt, 1)
for thr in (0.5, 0.8):
    mk = rate2 >= thr
    out[f"players_with_ge{int(thr*100)}pct_questions_off2plus"] = {
        "players": int(mk.sum()), "of": int((cnt > 0).sum()),
        "share_of_all_off2plus_questions": float(far2[mk].sum() / far2.sum()),
        "by_level": {LEVELS[i]: int((mk & (lvl == i)).sum()) for i in range(L)}}
# concentration: share of off2+ questions from the worst 10% players (within level-balanced view)
order = np.argsort(-far2)
k = int(0.1 * (cnt > 0).sum())
out["worst10pct_players_share_of_off2plus"] = float(far2[order[:k]].sum() / far2.sum())
out["worst10pct_players_share_of_off3plus"] = float(far3[order[:k]].sum() / far3.sum())
# rank-balanced if the players with >=80% far errors were removed (how much is 'player-level outlier')
keep = ~np.isin(pl, np.nonzero(rate2 >= 0.8)[0])
sc = score(pred, y)
out["rank_balanced_all"] = float(np.mean([sc[y == i].mean() for i in range(L)]))
out["rank_balanced_without_ge80pct_far_players"] = float(np.mean([sc[keep & (y == i)].mean() for i in range(L)]))
# per level: mean player-level predicted index (mode) direction for the far players
mode_pred = np.array([np.bincount(pred[pl == p], minlength=L).argmax() if cnt[p] else -1 for p in range(npl)])
far_players = np.nonzero(rate2 >= 0.8)[0]
out["far_players_true_to_modal_pred"] = sorted([f"{LEVELS[lvl[p]]}->{LEVELS[mode_pred[p]]}" for p in far_players])
json.dump(out, open(os.path.join(OUT, "far_errors.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
