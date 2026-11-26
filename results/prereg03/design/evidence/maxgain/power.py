# Power of the single validation comparison, using inner OOF (R20, R10) as proxies. CPU only, no validation data.
import numpy as np, json, math
z = np.load(r"C:/Users/USER/AppData/Local/Temp/claude/D--AI-Portfolio-competitions-aicup2026-go/943dc84e-3a07-4c6f-b22a-052eec94e7d1/scratchpad/p03/B_error/r20_oof.npz")
P20, P10, y, pl = z["P20"], z["P10"], z["y"], z["player"]
E1 = math.exp(-1)
K = np.eye(10) + E1 * (np.abs(np.subtract.outer(np.arange(10), np.arange(10))) == 1)
def ruleb(P): return np.argmax(P @ K, 1)
def score(pred): 
    d = np.abs(pred - y); return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))
s20, s10 = score(ruleb(P20)), score(ruleb(P10))
# validation per-level question counts (docs/VALIDATION.md)
NV = np.array([178,163,207,308,87,122,124,128,131,141])
out = {}
def se_val(d):
    # SE of rank-balanced mean of per-question differences when each level has NV questions, one per player
    var = np.array([d[y == l].var() for l in range(10)])
    return float(math.sqrt(np.sum(var / NV) / 100.0)), float(math.sqrt(np.sum(np.array([d.var()]*10) / NV) / 100.0))
d = s20 - s10
flip = float(np.mean(ruleb(P20) != ruleb(P10)))
se, _ = se_val(d)
rb = float(np.mean([d[y == l].mean() for l in range(10)]))
out["R20_vs_R10"] = {"inner_rank_balanced_delta": rb, "decision_flip_rate": flip, "val_se_rank_balanced": se,
                     "sd_per_question": float(d.std())}
# synthetic: how SE scales with flip rate -> for a mix, flips with prob f produce per-question diff variance ~ f * v_flip
v_flip = float(d[ruleb(P20) != ruleb(P10)].var() + d[ruleb(P20) != ruleb(P10)].mean()**2)
out["mean_sq_diff_given_flip"] = v_flip
tab = []
for f in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
    se_f = math.sqrt(f * v_flip * np.sum(1.0 / NV) / 100.0)
    for delta in (0.005, 0.010, 0.015, 0.020, 0.030):
        power = 0.5 * math.erfc((1.959964 * se_f - delta) / (se_f * math.sqrt(2)))
        tab.append({"flip": f, "se": round(se_f, 5), "delta": delta, "P(CI low>0)": round(power, 3)})
out["power_table"] = tab
# per-question mean (secondary) SE for R20 vs R10 over validation composition
w = NV / NV.sum()
var = np.array([d[y == l].var() for l in range(10)])
out["R20_vs_R10"]["val_se_question_mean"] = float(math.sqrt(np.sum(w**2 * var / NV)))
print(json.dumps(out, indent=1))
json.dump(out, open(r"C:/Users/USER/AppData/Local/Temp/claude/D--AI-Portfolio-competitions-aicup2026-go/943dc84e-3a07-4c6f-b22a-052eec94e7d1/scratchpad/p03/maxgain/power.json", "w"), indent=1)
