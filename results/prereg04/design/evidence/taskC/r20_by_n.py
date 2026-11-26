# R20 inner OOF (p03/B_error/r20_oof.npz, inner players only) by number of games, rule B
import numpy as np, math, json
d = np.load(r'C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03\B_error\r20_oof.npz')
P, y, n = d['P20'], d['y'], d['n']
L = 10; e = math.exp(-1)
S = np.array([[1.0 if i == j else (e if abs(i - j) == 1 else 0.0) for j in range(L)] for i in range(L)])
pred = np.argmax(P @ S, axis=1); sc = S[pred, y]
rb = lambda m: float(np.mean([sc[m & (y == l)].mean() for l in range(L)]))
out = {'all': rb(np.ones_like(y, bool)), 'n_questions': int(len(y))}
for k in [5, 10, 15, 20]:
    m = n == k
    out[str(k)] = {'questions': int(m.sum()), 'rank_balanced': rb(m), 'acc': float((pred[m] == y[m]).mean()), 'acc_pm1': float((abs(pred[m] - y[m]) <= 1).mean())}
json.dump(out, open('r20_by_n.json', 'w'), indent=1); print(json.dumps(out, indent=1))
