# PREREG_64 (competition 2): opponent term with c49 on a fixed Q25 anchor

Registered 2026-11-21 (results/prereg64/explore/PLAN.md, written before any code or number; readings NOTES.md,
settings frozen.json, both committed before any number). Source: the 11/21 Gemini review
(data/advice/gemini_1121/, C2-1 and C2-3). Inner used for the third time (prereg58, prereg59, prereg64).

## 1. Method

- Anchor: Q25's fused log-probabilities, its seven terms and parameters unchanged (results/prereg25/rank).
- OPP3: per game, the opponent's c49 vector (data/prereg49/S0/ckpt/step_320000.pt) against 5,771 c49 head rows
  (50 x cos) and 2,104 prototypes of the other labelled non-inner training players (E 600 + 1,504; 30 x cos + 1);
  q summed by rank; l_g(r) = log(sum_s q_g[s] M[h, c][r][s] + 1e-9) with M estimated on the D players' games only
  (prereg59's construction; prereg59 used D + E); question value 10 x mean over the games (no (n/12)^gamma).
- Fusion: Z = w_A * anchor + w_O * OPP3, (w_A, w_O) by Gemini 04's scalar fit (L-BFGS-B from 0.4, bounds [0, 4],
  mean NLL + 1e-3 sum w^2), rule B.
- Step 4 (same-player soft merge in each test pool, threshold 0.65, n_j / 12, beta 1.0): NOT applied (section 2).

## 2. Judgement (results/prereg64/explore/eval.json, merge_sim.json, RESULTS.md)

- Seams: Q25 0.7456519871546352 / 0.7482450052195216 and Q59 0.7539701874530542 / 0.7552848066878524 bit for bit.
- Q64 0.7610187 vs Q59 0.7539702: +0.0070 [+0.0012, +0.0128]; single colour +0.0046 [+0.0011, +0.0079]
  (B +0.0023, W +0.0071). G1, G2, G4 passed -> steps 1-3 adopted. vs Q25 +0.0154 [+0.0067, +0.0243].
- Merge simulation (E n H, 47 players, 20 pools): 0% repeats changed 1 question in 1 pool (mean +0.00023, not
  exactly 0) -> the registered zero-change condition failed; 35% +0.0073, 100% +0.0062. Not applied.

## 3. Frozen (results/prereg64/rank/frozen_config.json)

w_A 0.7667, w_O 0.4941, fitted once on all 25,400 inner fit questions with Q25's frozen theta (seam Z bit for bit);
c49_head.npz and prototypes.npz byte copies; merge.apply = false.

## 4. Test day

tools/t64_rank.py apply (CPU), docs/TESTDAY_64.md. Seams: A (Q25 recomputed = test25 probabilities byte for byte),
O / M (c49 vectors bound to the raw question files), B (w_A 1, w_O 0, no merge = 25's submission byte for byte),
check_submission --official. K6-64: 2026-11-24 23:00 +08:00. Upload and official_log.py: the main session decides.

## 5. Disclosure for the report

Inner reused (third time, designed after prereg58 / prereg59 on the same questions); c49 was trained on the
validation and inner players' games and the 322 validation players outside its head get prototypes; settings copied
from an external AI review (Gemini) and recorded with pointers.
