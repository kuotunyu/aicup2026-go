# prereg64 - narrow readings of PLAN.md

Written 2026-11-21 after the c49 GPU vectors exist (`data/prereg64/embed_log.txt`: all done 04:58:59; inner opponent
side 73,169 games, prototype owners' player side 18,892 games, 0 without a sample) and before any code of the term,
any kernel number and any inner number of this registration. PLAN.md is not edited. Every setting is copied into
`frozen.json` with a file / line pointer to Gemini's scripts (`data/advice/gemini_1121/`). Decisions 1, 2, 3, 9 were
taken by the coordinating session on 2026-11-21 and are recorded here.

1. **Table on D only** (coordinator decision). The conditional table M[h, c][r][s] is estimated on the opponent-side
   c49 vectors of the D players' games only (`data/prereg56/emb_opp_D/.../store_D.npz`, 94,015 games), as Gemini did
   (03 line 548-553, 04 line 358-366). This differs from prereg59, whose table used D and E (900 players); no c49
   opponent-side vector of E exists and none is made.
2. **Table construction = prereg59's** (PLAN step 2): `p59_lib.table_counts` / `table` (soft counts q_g[s] at the
   player's rank r, add-0.5 per row), all three HD groups (0, 1, 2+) x player colour, every D game with a sample,
   estimated once on all D games (the frozen table). Gemini's experiment additionally restricted the table to HD <= 1
   games that also had a usable P1 vector and an S0 vector (03 line 398, 04 line 353): those were properties of his
   experimental question set, not of the term; this run follows the PLAN (prereg59's construction). Recorded only:
   the number of D games with HD <= 1.
3. **Classes** (coordinator decision; PLAN step 2): (a) the 5,771 c49 head rows (`data/prereg49/S0/ckpt/step_320000.pt`
   `head.weight`, class ids = `p49_lib.t49`, class rank = the player's rank in the cache), logit tau1 * cos; (b) one
   prototype per labelled training player who is neither a head class nor an inner player: E (600, c49 player-side
   vectors `data/prereg49/emb/.../store_E.npz`) and the 1,504 players of `data/prereg64/proto_rows.npy`
   (`data/prereg64/emb_proto/.../rows_proto.npz`); logit tau2 * cos + b. Gemini built prototypes for E only
   (03 line 517, 04 line 329); the PLAN gives every labelled non-inner player a prototype, so the 1,504 are added.
   Prototype = `p59_lib.prototypes` (L2 mean of L2 game vectors). Head rows are L2-normalised (03 line 510).
   2,104 prototypes + 5,771 head rows = 7,875 classes = all 8,007 cache players minus the 132 inner players that are
   not head classes (inner players that are head classes stay classes, see 4).
4. **Own class left out.** On a D game (table) the D player's own head row is removed from the soft-max (D players
   are all head classes; Gemini 03 line 527-532). On an inner game the inner question player's own head row is
   removed (508 of the 640 inner players are c49 head classes because c49 was trained on their games; the test
   players are not classes, so leaving the own row out keeps inner like the test set). Nothing else is left out.
   On the test set nothing is left out.
5. **(tau1, tau2, b) = (50, 30, 1)** copied from Gemini (03 line 532-535, 04 line 337-339); no grid, no choice on
   any data here. prereg59 had b = 2 (its own grid on D and E with S0); Gemini used 1 with c49.
6. **Term** = prereg59's per-game l_g(r) = log(sum_s q_g[s] M[h_g, c_g][r][s] + 1e-9), question value
   OPP3 = 10 x the mean over the question's games (the harness's mean-pooling matrices S_fit / S_eval / S_mono),
   NOT multiplied by (n/12)^gamma (04 line 412: `10.0 * lg_c49[g].mean(axis=0)`, "flat").
7. **Fusion** (PLAN steps 1 and 3; Gemini 04 `eval_scalar_oof`, line 416-436): the anchor A = Q25's fused
   log-probabilities = log_softmax of `p04_rank_fuse.gen_logits(theta_f, m, E7, n)` with Q25's own fold parameters
   theta_f (the seven terms, a, gamma, the class offsets beta: Q25's internals are not refitted), and
   Z = w_A * A + w_O * OPP3. Per player fold f (the harness's folds): theta_f is Q25's fold-f fit (fit questions of the
   other folds); (w_A, w_O) are fitted on the same fit questions (folds != f) with A computed by theta_f, by L-BFGS-B
   from (0.4, 0.4), bounds [0, 4], objective = mean NLL of the true rank + 1e-3 * sum w^2, no class offsets and no
   class weighting (Gemini 04 line 424-430 used none; his 03 version with offsets is not the one he recommended).
   Evaluated on eval fold f (and the single-colour questions of fold f). Rule B decode (`rf.rule_b_np`), scores
   through the harness (`rf.metrics`, `p03 mono_score_record`, the harness bootstrap), exactly as `Inner.run`.
   The anchor on the fit questions is in-sample for theta_f (stacking reading); on the test day theta = Q25's frozen
   all-inner fit and (w_A, w_O) are fitted on all inner fit questions with that anchor (same structure).
8. **Seams** (before the first number of OPP3): Q25 inner 0.7456519871546352 and single colour 0.7482450052195216
   bit for bit (`p58_eval.build_q25`); Q59 inner 0.7539701874530542 and single colour 0.7552848066878524 bit for bit
   (prereg59's committed kernel / prototypes / S0 inner vectors through `p59_eval.inner_term` and `Inner.run`);
   and the anchor alone (w_A = 1, w_O = 0 without fitting) reproduces Q25's eval / single-colour probabilities
   exactly. Run and committed as `seam.json` before `eval`; `eval` repeats them in the same process.
9. **Merge** (step 4; coordinator decision): threshold 0.65, (S_ij - 0.65) / 0.35, neighbour weight n_j / 12,
   beta 1.0 (Gemini 04 line 157-161) and lam 0.30 (04 line 109). Gemini's simulation numbers in 03 json 245-273 were
   made with a different setting (threshold = the null's 99.99th percentile, n_j / 15, beta 0.8; 03 line 838-858),
   so the simulation is re-run here with the PLAN's setting, otherwise copied from 03 `test_c2_3_...`
   (line 591-878: E players with c49 and P1, S25 holdout bit 1 (03 line 87-89), >= 20 games of each colour; 20 pools
   per mode, seeds 9000 + sd; modes 0 / 0.35 / 1.0 repeat fraction; base = the S25 set term `4 * lp`; rank-balanced
   official score; decode `p58_lib.decode`). Pass = mean gain exactly 0 with min = max = 0 at 0% AND mean gain > 0 at
   35%. The merge is applied after the step 1-3 log-probabilities, inside each test pool separately; only step A.
   The merge uses the c49 player-side vectors of the test questions' games (`data/prereg56/emb_rank/...`).
10. **Gates** = `p23_lib.gates` against Q59 (G1 >= +0.005, G2 lower bound > 0, G4 single colour lower bound > 0 and
    B, W each >= -1e-12). Recorded: vs Q25 (same gates as a record), by number of games, by rank class, fitted
    weights by fold, the term alone (rule B on 10 x mean).
11. **Hard stop**: `eval` refuses at or after 2026-11-23 12:00. Judged once: `eval` refuses when `eval.json` or
    `data/prereg64/eval_started.json` exists; kernel.json, seam.json, frozen.json, NOTES.md and the code must be
    committed and clean.
12. Inner is used for the third time (prereg58, prereg59, prereg64); the report says so. The validation players'
    training games and ranks are used (prototypes of the 322 validation players outside the c49 head, and c49
    itself was trained on validation players); the report says so.
