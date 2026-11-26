# prereg69 (competition 2): Neon's safe same-player merge M8 applied on top of 64

Date 2026-11-22 09:50. Written before any code of this registration and before any number of it.

prereg65 (Neon's Q65B) was not adopted, but its M8 merge passed its own registered simulation
(`results/prereg65/explore/merge_sim.json`: 0% repeats -> 0 changes over 100 E ∩ H pools; 35% -> +0.00039;
100% -> +0.00344). M8 was never applied because steps 1-3 of prereg65 failed. Here M8 is applied to 64 alone.

## Method

64's frozen fused log-probabilities on each test pool (`tools/t64_rank.py` / `results/test64/rank/`), public and
private separately. M8 exactly as frozen in prereg65 (`results/prereg65/explore/frozen.json`, NOTES): c49
player-side vectors of the questions (`data/prereg56/emb_rank/...`), cc_whiten_sim lam 0.30, edges S_ij > 0.80 AND
same colour AND |E[r_i] - E[r_j]| <= 2.5, L_new = L_norm + 0.85 * (A_adj @ L_norm), change i only if all its
neighbours' new predictions agree and the new prediction is one of the group's original predictions; rule B decision.

## Gate (no inner judgement is possible: inner questions are one player each)

Re-run prereg65's merge simulation with 64's fusion as the base (64's frozen weights on the same E ∩ H players and
the same 100 pools / seeds): adopt only if 0% repeats gives exactly 0 changed questions in all pools AND the mean
gain at 35% repeats is > 0. Seam: with the merge off the output equals 64's submission byte for byte.

## Apply

`tools/t69_rank.py` -> `results/test69/rank/submission_rank.csv`, list of changed questions per pool,
check_submission --official, upload as competition-2 #13 only if adopted.
