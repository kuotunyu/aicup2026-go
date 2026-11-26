# prereg64 (competition 2): OPP3 (c49 opponent term) + anchor fusion on Q25 - steps 1-3 PASS

Written mechanically from `eval.json` (2026-11-21T05:12:06) and `kernel.json` (2026-11-21T05:07:41); numbers only. PLAN.md, NOTES.md, frozen.json. Inner used for the third time (prereg58, prereg59, prereg64).

## Kernel (D players only, frozen settings from Gemini)

- Classes 7875: c49 head 5771 + prototypes 2104 (E 600, others 1504); tau1 50, tau2 30, b 1.
- D games 94015 (94015 with a sample; HD <= 1: 91039); record held-out mean log-likelihood -1.7578, mean max q 0.6089.

## Seams

- Q25 0.7456519871546352 / single colour 0.7482450052195216: bit for bit True / True; Q59 0.7539701874530542 / 0.7552848066878524: bit for bit True / True; anchor alone = Q25 True.

## Judgement (inner, once)

| | Q59 | Q64 | diff | 95% interval | gate |
|---|---|---|---|---|---|
| inner eval | 0.7539702 | 0.7610187 | +0.00705 | [+0.00118, +0.01282] | G1 pass; G2 pass |
| single colour | 0.7552848 | 0.7599314 | +0.00465 | [+0.00110, +0.00788] | G4 pass (B +0.0023, W +0.0071) |

**steps 1-3: steps 1-3 passed all three gates vs Q59.**

## Record only

- vs Q25: inner +0.01537 [+0.00668, +0.02426], single colour +0.01169 [+0.00586, +0.01715].
- Weights by fold (w_A, w_O): 0.7763, 0.4593; 0.7669, 0.4755; 0.7753, 0.5211; 0.7698, 0.5133; 0.7461, 0.5015.
- OPP3 alone (rule B): 0.6856 (single colour 0.6723).
- By rank class (eval, Q64 - Q59): D +0.0145, C +0.0095, B +0.0121, A +0.0096, 1D +0.0378, 2D -0.0149, 3D -0.0177, 4D +0.0175, 5D -0.0151, 6D +0.0172.
- Single colour by number of games (vs Q59): 5 +0.0047 [-0.0001, +0.0093]; 10 +0.0043 [-0.0006, +0.0095]; 15 +0.0051 [+0.0012, +0.0091]; 20 +0.0043 [+0.0001, +0.0088].
- Eval questions with exactly n games (vs Q59): 5 +0.0135; 10 +0.0089; 15 +0.0081; 20 +0.0036.
- Inner question games 73169, own head row left out in 71516; mean max q 0.6149.

## Merge simulation (step 4; E n H, threshold 0.65, n_j / 12, beta 1.0)

- zero_repeat_safety: base S25 0.71615, merged - base mean +0.00023 (min +0.00000, max +0.00460); questions changed per pool 0.05.
- moderate_repeat_35pct: base S25 0.73618, merged - base mean +0.00732 (min -0.02107, max +0.03635); questions changed per pool 1.45.
- paired_repeat_100pct: base S25 0.74154, merged - base mean +0.00619 (min -0.01054, max +0.02826); questions changed per pool 3.10.

**merge simulation: NOT passed** (0% repeats: mean gain 0 with min = max = 0; 35% repeats: mean gain > 0).
