# prereg65 (competition 2): OPP4B (Neon's two-channel opponent term) on Q25 - steps 1-3 NOT adopted

Written mechanically from `eval.json` (2026-11-21T19:09:18); numbers only. PLAN.md, NOTES.md, frozen.json. Inner used for the fourth time (prereg58, 59, 64, 65).

## Seams

- In the judgement run: Q25 0.7456519871546352 / 0.7482450052195216 (seam ok True); Q64 0.7610187048035557 / 0.7599313807253795 bit for bit True. seam.json, seam_p.json, seam_n.json committed before.

## Judgement (inner, once)

| | Q64 | Q65 | diff | 95% interval | gate |
|---|---|---|---|---|---|
| inner eval | 0.7610187 | 0.7586215 | -0.00240 | [-0.00667, +0.00223] | G1 no; G2 no |
| single colour | 0.7599314 | 0.7593733 | -0.00056 | [-0.00333, +0.00221] | G4 no (B +0.0015, W -0.0026) |

**steps 1-3: gates vs Q64 not passed: not adopted.**

## Record only

- vs Q25: inner +0.01297 [+0.00396, +0.02207], single colour +0.01113 [+0.00512, +0.01667].
- Weights by fold (w_A, w_O): 0.7479, 0.4530; 0.7392, 0.4590; 0.7566, 0.4745; 0.7344, 0.5148; 0.7119, 0.5027.
- OPP4B alone (rule B): 0.6996 (single colour 0.6864).
- By rank class (eval, Q65 - Q64): D +0.0142, C +0.0038, B -0.0171, A -0.0328, 1D -0.0383, 2D +0.0198, 3D +0.0191, 4D -0.0052, 5D +0.0079, 6D +0.0047.
- Eval questions with exactly n games (vs Q64): 5 +0.0002; 10 -0.0039; 15 -0.0041; 20 -0.0046.
- By share of HD = 1 games (eval, vs Q64): hd1_eq_0 -0.0048 [-0.0110, +0.0021] (n 39445); hd1_gt_0 -0.0007 [-0.0059, +0.0046] (n 87755); hd1_gt_025 -0.0001 [-0.0102, +0.0101] (n 42681).
- Inner question games 73169; without c49 sample 0; without opponent P1 0; mean max q_c49 0.7353; mean alpha 0.6205.

## M8 merge simulation (Neon 04 section 3; E n H, 100 pools)

- Base weights (section 2 refit) [0.2162, 0.538] (Neon [0.2162, 0.538]); eligible players 46.
- zero_repeat_0pct: base 0.75324; gain mean +0.00000 (min +0.00000, max +0.00000); changed per pool 0.00 (max 0); pools with a change 0/100.
- moderate_repeat_35pct: base 0.73979; gain mean +0.00039 (min -0.01580, max +0.02167); changed per pool 0.52 (max 2); pools with a change 46/100.
- paired_repeat_100pct: base 0.75383; gain mean +0.00344 (min -0.01817, max +0.03108); changed per pool 2.18 (max 5); pools with a change 90/100.

**M8 simulation: PASS** (0% changed = 0 in 100/100 pools and 35% mean gain > 0).
