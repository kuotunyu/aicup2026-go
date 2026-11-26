"""Power of the single validation comparison (inner OOF only; no validation data, no answers)."""
import sys, json, math
sys.dont_write_bytecode = True
sys.path.insert(0, r"D:\AI-Portfolio\competitions\aicup2026-go\tools")
import numpy as np
import p02_rank_fuse as F
z = np.load(sys.argv[1])
P20, P10, y, player = z["P20"], z["P10"], z["y"], z["player"]
s20 = F.official_score(F.rule_b_np(P20), y) if False else None
def sc(P):
    pred = F.rule_b_np(P)
    d = np.abs(pred - y)
    return np.where(d == 0, 1.0, np.where(d == 1, math.exp(-1), 0.0)), pred
a, pa = sc(P20); b, pb = sc(P10)
d = a - b
VAL_N = [178, 163, 207, 308, 87, 122, 124, 128, 131, 141]
var_rb = sum(np.var(d[y == L]) / VAL_N[L] for L in range(10)) / 100
se_rb = math.sqrt(var_rb)
se_mean = math.sqrt(np.var(d) / 1589)
rb_delta = np.mean([d[y == L].mean() for L in range(10)])
out = dict(inner_delta_rb=float(rb_delta), inner_delta_mean=float(d.mean()), disagree_rate=float(np.mean(pa != pb)),
           se_val_rb=se_rb, se_val_mean=se_mean, mde80_rb=2.8 * se_rb, mde50_rb=1.96 * se_rb,
           power_at={})
for delta in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
    # P(lower CI > 0) ~ P(Z > 1.96 - delta/se)
    zc = 1.96 - delta / se_rb
    out["power_at"][str(delta)] = float(0.5 * math.erfc(zc / math.sqrt(2)))
# simulation: one question per player, 636 players, analytic check by resampling
rng = np.random.default_rng(0)
players = np.unique(player)
idx_by_p = {p: np.nonzero(player == p)[0] for p in players}
sims = []
for _ in range(2000):
    pick = np.array([rng.choice(idx_by_p[p]) for p in players])
    dd, yy = d[pick], y[pick]
    sims.append(np.mean([dd[yy == L].mean() for L in range(10)]))
n_inner = [int(np.sum(np.isin(players, np.unique(player[y == L])))) for L in range(10)]
an_inner = math.sqrt(sum(np.var(d[y == L]) / n_inner[L] for L in range(10)) / 100)
out["sim_inner_1q_sd"] = float(np.std(sims)); out["analytic_inner_1q_se"] = an_inner; out["inner_players_per_level"] = n_inner
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=1)
