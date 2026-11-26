import sys, json
sys.path.insert(0, r"D:\AI-Portfolio\competitions\aicup2026-go\tools")
import c10data as c
split = c.read_split(r"D:\AI-Portfolio\competitions\aicup2026-go\results\split_v1\players_split.csv")
fit = c.players_for(split, "fit")
n = {p: split[p]["n_games"] for p in fit}
rk = {p: split[p]["rank"] for p in fit}
res = {}
for cap in [100, 150, 200, 300]:
    E = {L: sum(min(n[p], cap) for p in fit if rk[p] == L) for L in c.LEVELS}
    m = min(E.values())
    eff = {p: min(n[p], cap) * m / E[rk[p]] for p in fit}
    tot = sum(eff.values())
    s2 = sum((v / tot) ** 2 for v in eff.values())
    res[cap] = dict(per_level=E, min_level=m, epoch_games=tot, eff_players=1/s2)
    print(cap, {L: E[L] for L in c.LEVELS}, "min", m, "balanced epoch", round(tot), "effN", round(1/s2))
# players per level in fit, and games
for L in c.LEVELS:
    ps=[p for p in fit if rk[p]==L]
    print(L, len(ps), sum(n[p] for p in ps))
json.dump({str(k): v for k, v in res.items()}, open(sys.argv[1], "w"), indent=1)
