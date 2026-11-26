"""Are inner games (bank games and all) also present in fit as the same move sequence (e.g. the same game seen from
the opponent's side, or a copy)? Exact match of the full move list (colour + point of every move)."""
import hashlib, json, sys
sys.dont_write_bytecode = True
import numpy as np
fi = np.load("moves_fit.npz"); inn = np.load("moves_inner.npz")
fm = np.load("moves_fit_moves.npy", mmap_mode="r"); fo = fi["off"]
fcol = fi["color"]; flab = fi["label"]
h = {}
for g in range(len(fo) - 1):
    k = hashlib.sha1(np.asarray(fm[fo[g]:fo[g + 1]]).tobytes()).digest()
    h.setdefault(k, []).append(g)
im = np.load("moves_inner_moves.npy", mmap_mode="r"); io = inn["off"]
icol = inn["color"]; ilab = inn["label"]
pj = json.load(open("D:/AI-Portfolio/competitions/aicup2026-go/results/prereg02/rank/pseudo_inner.json", encoding="utf-8"))
bank = {(p["rank"], gid) for p in pj["players"] for gid in p["bank"]}
ir, ig = inn["rank"].tolist(), inn["game_id"].tolist()
res = {"all": [0, 0, 0, 0], "bank": [0, 0, 0, 0]}   # games, matched, matched other colour, level diff sum
lvdiff = []
for g in range(len(io) - 1):
    k = hashlib.sha1(np.asarray(im[io[g]:io[g + 1]]).tobytes()).digest()
    hit = h.get(k, [])
    for key in (["all", "bank"] if (ir[g], ig[g]) in bank else ["all"]):
        res[key][0] += 1
        if hit:
            res[key][1] += 1
            if any(fcol[j] != icol[g] for j in hit):
                res[key][2] += 1
    if hit:
        lvdiff += [int(flab[j]) - int(ilab[g]) for j in hit if fcol[j] != icol[g]]
out = {k: {"games": v[0], "also_in_fit": v[1], "also_in_fit_other_colour": v[2]} for k, v in res.items()}
out["fit_unique_sequences"] = len(h)
out["fit_games"] = len(fo) - 1
if lvdiff:
    d = np.array(lvdiff)
    out["other_side_level_minus_inner_level"] = {"n": len(d), "mean": float(d.mean()), "abs_mean": float(np.abs(d).mean()),
                                                 "share_equal": float((d == 0).mean()), "share_within_1": float((np.abs(d) <= 1).mean())}
json.dump(out, open("dupcheck.json", "w"), indent=1)
print(json.dumps(out, indent=1))
