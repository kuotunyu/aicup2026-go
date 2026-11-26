import sys, json
sys.path.insert(0, r"D:\AI-Portfolio\competitions\aicup2026-go\tools")
import c10data as c
split = c.read_split(r"D:\AI-Portfolio\competitions\aicup2026-go\results\split_v1\players_split.csv")
fit = c.players_for(split, "fit")
n = {p: split[p]["n_games"] for p in fit}
rk = {p: split[p]["rank"] for p in fit}
tot = sum(n.values())
print("fit players", len(fit), "games", tot)
out = {}
for cap in [None, 50, 100, 150, 200, 300, 400]:
    eff = {p: (min(v, cap) if cap else v) for p, v in n.items()}
    E = sum(eff.values())
    per_lvl = {L: sum(eff[p] for p in fit if rk[p] == L) for L in c.LEVELS}
    # share of top 10% players (by n) per level, averaged
    shares = []
    for L in c.LEVELS:
        ps = sorted([p for p in fit if rk[p] == L], key=lambda p: -n[p])
        k = max(1, int(0.1 * len(ps)))
        shares.append(sum(eff[p] for p in ps[:k]) / per_lvl[L])
    # effective number of players (inverse Simpson of game share)
    s2 = sum((v / E) ** 2 for v in eff.values())
    out[str(cap)] = dict(games=E, frac=E / tot, per_level=per_lvl, top10_share_min=min(shares), top10_share_max=max(shares),
                         eff_players=1 / s2, lvl_ratio=max(per_lvl.values()) / min(per_lvl.values()))
    print(cap, E, round(E / tot, 3), "top10%", round(min(shares), 3), round(max(shares), 3), "effN", round(1 / s2), "lvl max/min", round(max(per_lvl.values()) / min(per_lvl.values()), 3))
json.dump(out, open(sys.argv[1], "w"), indent=1)
