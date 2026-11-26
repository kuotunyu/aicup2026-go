# prereg64 (competition 2): opponent term v3 + same-player merge, found by the 11/21 Gemini review

Date 2026-11-21. Written before any code and any inner number of this registration. Source:
`data/advice/gemini_1121/REPORT.md` sections 1.3, 2.5, 3.2 (C2-1, C2-3), scripts `03_comp2_and_final_checks.py`,
`04_quick_verify_fusion.py`, results `03_comp2_and_final_checks.json`, `04_fusion_and_flips.json` (development on
D, E, H players; inner not used by them). Inner has been used by prereg58 and prereg59; this is its third reuse and
the report says so.

## Method (on top of Q25; Q59 = the current upload is the comparison)

1. Q25's 7-term fused log-probabilities are kept as one fixed anchor term (no refit of its internals: OP, O4, P1TO,
   SET keep their frozen weights).
2. OPP3 = prereg59's opponent term with the identity model c49 instead of S0: c49 head rows (5,771 classes) plus
   prototypes for every other labelled training player (c49 vectors, games of the prototype owner only, never an
   inner player; own prototype excluded for kernel players), same conditional table construction as prereg59
   (HD group x colour), per-game mean, NOT multiplied by (n/12)^gamma ("flat" as in Gemini's script). tau etc. as
   chosen by Gemini on kernel players (copy into `frozen.json` with pointers before any inner number).
3. Fusion: w_anchor * Q25 + w_opp * OPP3 (+ per-class offsets only if Gemini's script used them - copy exactly),
   fitted on inner fit questions with the same cross-fitting as prereg58/59, rule B decision unchanged.
4. Same-player soft merge inside each test pool separately (public / private never mixed), Gemini's C2-3 step A only:
   c49 player-side vectors of the questions, colour-centred and whitened with lam 0.30, threshold 0.65, the update
   L_i += beta * sum_j ((S_ij - 0.65) / 0.35) * (n_j / 12) * L_j with beta as in Gemini's script. Step B (opponent
   = another question's player) is not included.

## Judgement (inner, once)

Seam: Q25 inner 0.7456519871546352 and Q59 inner 0.7539702 (as recorded) reproduced. Q64 (1-3 without 4; inner
questions are one player each, so the merge cannot be judged there) vs Q59: G1 >= +0.005, G2 lower bound > 0,
G4 single colour (lower bound > 0, black and white each >= 0). Record also vs Q25.
Step 4 is judged separately on E ∩ H style simulation exactly as in Gemini's `03_...json` lines 245-273 (0% repeats
must change nothing; at 35% repeats gain > 0) and reported; it is applied only if 1-3 pass AND the simulation passes.
Hard stop 11/23 12:00.

## Apply

t59-style test-day tool (CPU): c49 opponent-side vectors of the test games already in
`data/prereg56/emb_opp_rank/V2_step_320000_raw_tta1/`, player-side in `data/prereg56/emb_rank/...`; seams: with
w_opp 0 the Q25 anchor reproduces 25's submission; check_submission; upload as competition-2 #12.
