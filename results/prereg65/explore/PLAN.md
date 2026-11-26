# prereg65 (competition 2): two-channel opponent term (Gemini Neon 11/21, "Q65B")

Date 2026-11-21. Written before any code of this registration and before any inner number of it. Source: Gemini
Neon's review `data/advice/gemini_1121n/REPORT.md` section 4 (Q65B, OPP4B, M8) and its scripts/results in the same
folder (`02a_train_opp_p1_v2.py`, `opp_p1_v2.npz`, `02b_q64_audit_and_opp_fixes.*`, `04_final_candidate_stack.*`,
`01_merge_design_and_sim.*`). Its numbers are development evidence on held-out players (D ∩ H, (D ∪ E) ∩ H); inner
was not used by it. This is the 4th use of inner (prereg58, 59, 64 before); the report says so.

## Method = Neon's Q65B exactly as its recipe states (all numbers copied into `frozen.json` with file/line pointers)

1. Anchor A = Q25's fused log-probabilities (as prereg64).
2. OPP4B = 10 * mean over games of log(alpha_g R_c49(g, r) + (1 - alpha_g) R_P1(g, r) + 1e-9): R_c49 from prereg64's
   c49 opponent term in likelihood-ratio form (Neon's "LR-shrink", its M_c49_bayes table), R_P1 from Neon's frozen
   OPP_P1_v2 (4-member numpy model, `opp_p1_v2.npz`, trained on probe + V role-A HD <= 1 target-side P1; never H or
   inner) with its table M_P1_bayes; alpha_g = clip((max_s q_c49[g, s] - 0.1) / 0.9, 0, 1) ^ 1.5. Flat mean.
3. Fusion Z = w_A A + w_O OPP4B with prereg64's fit_weights (start 0.4, [0, 4], L2 1e-3, no offsets), rule B.
4. Merge M8 (test pools only, public / private separately): edges S_ij > 0.80 AND same colour AND
   |E[r_i] - E[r_j]| <= 2.5; L_new = L_norm + 0.85 * (A_adj @ L_norm); change i only if all its neighbours' new
   predictions agree and the new prediction is one of the group's original predictions. Applied only if Neon's
   E ∩ H simulation is reproduced here with exactly 0 changes at 0% repeats (100 pools) and gain > 0 at 35%.

## Seams before the judgement

Q25 inner 0.7456519871546352 and Q64 inner 0.7610187 (eval) / single colour as recorded in
results/prereg64/explore/eval.json, bit for bit through this path. OPP_P1_v2 numpy vs Neon's recorded outputs
within 1e-4. Neon's D ∩ H numbers for OPP4B reproduced (|diff| <= 1e-6) - checks the port.

## Judgement (inner, once)

Q65 (1-3) vs Q64: G1 >= +0.005, G2 lower bound > 0, G4 single colour (lower bound > 0, black and white each >= 0).
Recorded: vs Q25, by n, by class, by HD group, fitted weights. Hard stop 11/23 12:00.

## Apply

t64-style tool `tools/t65_rank.py` (CPU; P1 of the test games from results/test08/p1 as Neon used); seams: with
w_O = 0 it reproduces 25's file; with OPP4B replaced by prereg64's term it reproduces 64's file; check_submission;
upload as competition-2 #13 only if adopted.
