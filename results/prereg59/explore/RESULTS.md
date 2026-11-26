# prereg59 (competition 2): opponent-identity term, second version - ADOPT (gates passed)

Written mechanically by `p59_eval.py results` from `eval.json` (2026-11-20T11:52:13+00:00) and `kernel.json` (2026-11-20T19:50:57); numbers only. Plan `PLAN.md`, narrow readings `NOTES.md`. This version was designed after prereg58's judgement on the same inner questions (inner reused).

## Kernel

- Classes: 3685 head rows + 2500 prototypes (D 300, E 600, V 1600) = 6185.
- Kernel players 900 (D 300, E 600), games 287751 (0 without a sample).
- Chosen tau1 = 50, tau2 = 30, b = 2 (held-out mean log-likelihood -1.8423; 60 grid points, range -1.8962 .. -1.8423).
- Standalone record on the kernel players (5 / 10 / 20 games): 0.5814 / 0.6367 / 0.6755 (prior only 0.1502).

## Seam

Q25 inner 0.7456519871546352 (reference 0.7456519871546352), single colour 0.7482450052195216; bit for bit: True / True.

## Judgement (inner, once)

| | Q25 | Q59 | diff | 95% interval | gate |
|---|---|---|---|---|---|
| inner eval | 0.7456520 | 0.7539702 | +0.00832 | [+0.00150, +0.01538] | G1 pass; G2 pass |
| single colour | 0.7482450 | 0.7552848 | +0.00704 | [+0.00293, +0.01082] | G4 pass (B +0.0054, W +0.0087) |

**adopt = true** (all three gates passed: freeze the 8-term fusion, write PREREG_59 (PLAN.md)).

## Record only

- Weight of the new term by fold: 0.8971, 0.6575, 1.0031, 1.3082, 0.9251 (limit 4.0; at the limit in 0 of 5 folds, at zero in 0).
- The term alone (rule B): scaled mean 0.6390 (single colour 0.6352); sum over games 0.6395 (0.6355).
- By rank class (eval, Q59 - Q25): D +0.0269, C -0.0033, B +0.0283, A +0.0133, 1D +0.0081, 2D -0.0019, 3D -0.0071, 4D +0.0042, 5D -0.0007, 6D +0.0154.
- By rank class (single colour): D +0.0309, C -0.0037, B +0.0040, A +0.0156, 1D +0.0116, 2D -0.0065, 3D -0.0080, 4D +0.0033, 5D -0.0017, 6D +0.0249.
- Single colour by number of games: 5 games +0.0084 [+0.0038, +0.0131]; 10 games +0.0060 [+0.0002, +0.0114]; 15 games +0.0061 [+0.0009, +0.0112]; 20 games +0.0069 [+0.0018, +0.0121].
- Eval questions with exactly n games: 5 games +0.0107 [+0.0045, +0.0171]; 10 games +0.0085 [+0.0021, +0.0153]; 15 games +0.0033 [-0.0046, +0.0108]; 20 games -0.0026 [-0.0150, +0.0111].
- Inner question games 73169, without a sample 0; mean of max_s q_g[s] 0.5803.
