# prereg65 - narrow readings of PLAN.md

Written 2026-11-21 before any code of this registration and before any inner number of it. PLAN.md is not edited.
Every setting is copied into `frozen.json` with a file / line pointer into Gemini Neon's folder
`data/advice/gemini_1121n/` (not in git: `data/` is ignored; the SHA-256 of every file used is in `frozen.json`).
Base harness = prereg64 (`results/prereg64/explore/p64_lib.py`, `p64_eval.py`), imported, never edited.

1. **Q65 = Neon's Q65B recipe (REPORT.md section 4.1, lines 196-227)**, not its section-4 test-pool code. Neon's
   `04_final_candidate_stack.py` section 4 (lines 826-958) built its official-pool candidate with the exponent
   p = 1.0 (`lr_lg(qt_EX, R_EX_l25_t, R_p1_t, 1.0)`, line 917) and with w_O = prereg64's w_O scaled by a D n H weight
   ratio (lines 921-925). The PLAN and the recipe (REPORT line 217-221; 04 line 489 `OPP4B_EX_l25_p1.5`, the number
   the report quotes) say p = 1.5 and weights fitted by prereg64's `fit_weights`. This run uses p = 1.5 and fitted
   weights everywhere (inner and test). Neon's test-pool change counts (REPORT 1.3) are therefore not expected to
   be reproduced and are not a seam.
2. **c49 channel** (04 lines 340-382, `make_q_c49("splitEX")`): head rows (prereg64's `c49_head.npz`, 5,771) at
   tau1 = 70, prototypes split by `src` into E (600) at tau2 = 50 + b_E = -1 and X (1,504) at tau2 = 50 + b_X = -2;
   each part's rank sums divided by its class count per rank and multiplied by that count's mean (prior
   correction); stable sum with m = max(m_A, m_E - 1, m_X - 2); normalised. Own head row left out on D games
   (04 line 344, 351) and on inner games (prereg64 NOTES 4); nothing left out on the test pools (04 line 903).
   The prototypes are prereg64's frozen `prototypes.npz` (identical bytes in `data/prereg64/` and
   `results/prereg64/rank/`, SHA-256 6ffe6682...).
3. **c49 table** (04 lines 179-189, 389-402, 852-855): `K59.table_counts` on all 94,015 D games with a sample (all
   three HD groups, `ok_D_all`, 04 line 327), add-0.5 rows (`K59.table`); the HD = 1 rows shrunk 0.25 toward the
   row-normalised rank-difference kernel matrix `K58.kernel_matrix(K58.kernel(K58.kernel_counts(q, r, h, c, ok)))`
   of the same games; then divided by the mean over r (`M / max(mean_r M, 1e-12)`). On inner and the test pools the
   table is the all-D table (04 line 855, `M_EX_l25`); the 5-fold D tables are used only to reproduce Neon's
   D n H number (seam N).
4. **P1 channel**: Neon's frozen numpy model `opp_p1_v2.npz` (02a, 4 members mlp0, mlp1, clm0, clm1, probability
   mean; 04 lines 234-250) on the opponent-side 770 P1 means, standardised with the file's `mu_t`, `sd_t`
   (02a lines 95-96: = `p1e_map` mu_x[:770], sd_x[:770]; checked in the seam), non-finite -> 0. Table
   `M_P1_bayes` = `M_p1_tr` of `c2_2_scond_p1.npz` (03 line 293: A_tr HD <= 1 games, `M / mean_r M` without
   shrinkage), taken as frozen, not re-estimated (the PLAN says "its table M_P1_bayes"). R_P1 = 1 for a game without
   an opponent P1 vector (04 lines 405-406, 900-901). Inner: the opponent half of `data/prereg12/q09_inner/inner_X.npy`
   (rows -> inner games by `inner_meta.npz` `ref`, as p19 / p23 do); a game without a row or with a NaN opponent half
   (no opponent move, `p09_forward.x_raw`) has R_P1 = 1. Test: `data/test08/p1/<pool>/chunk_*.npz` means[:, 1]
   (04 lines 876-899); `counts[:, 1] == 0` -> R_P1 = 1.
5. **Term** (04 lines 464-466, 454-456): alpha_g = clip((max_s q_c49[g, s] - 0.1) / 0.9, 0, 1) ^ 1.5;
   l_g = log(alpha_g R_c49 + (1 - alpha_g) R_P1 + 1e-9); l_g = 0 for a game without a c49 sample (04 lines 915-917;
   prereg64's `game_values`). Question value OPP4B = 10 x the mean over the question's games with the harness's
   pooling matrices (exactly prereg64's `question_term`, games without a sample count as 0 in the mean as in Q64).
   Flat (no (n/12)^gamma), w_hd1 = 1.
6. **Fusion** = prereg64 NOTES 7 with OPP4B in place of OPP3 (`p64_lib.fit_weights`, start 0.4, [0, 4], L2 1e-3,
   per player fold with Q25's fold parameters; rule B). Test day: Q25's frozen theta and (w_A, w_O) fitted on all
   inner fit questions (prereg64 `p64_freeze.py` structure).
7. **Seams** before the judgement (`seam.json`, committed before `eval`):
   - Q25 inner 0.7456519871546352 / 0.7482450052195216 and Q64 inner 0.7610187048035557 / 0.7599313807253795
     (prereg64 `eval.json`) bit for bit through this code path (Q64 rebuilt with prereg64's kernel and OPP3).
   - P: the numpy OPP_P1_v2 ensemble on Neon's H rows (02a's selection: probe + V, HD <= 1, S25 holdout bit 1;
     31,772 games) gives the recorded torch H NLL 1.59235 and single-game rank-balanced 0.48575
     (02a_opp_p1_v2_record.json lines 39-42) within 1e-4. (Neon recorded only these outputs of the torch model.)
   - N: Neon's D n H numbers re-run with this code (04 section 1: questions with rng 20261122, S25 set term,
     5-fold D tables, Neon's `fit_scalar_oof`): Q64_proxy 0.74839 and OPP4B_EX_l25_p1.5 0.76584, weights
     [0.2294, 0.6102] (04 json lines 48-49, 576-577, 612-615). **Narrow reading**: Neon's json stores 5 decimals,
     so "|diff| <= 1e-6" is read as: our unrounded value rounded to 5 decimals equals the recorded value
     (|diff| <= 5e-6), and the weights rounded to 4 decimals are equal. If not, STOP.
8. **M8 simulation** = 04 section 3 (lines 666-824) exactly: base = S25 + OPP4B_S0 (S0 vectors of D and E, prereg59
   prototypes, tau 70 / 50, b -1, prior correction, 5-fold (D u E) tables lam 0.25, P1 channel, p 1.5) with the
   section-2 weights (fit on (D u E) n H questions, rng 20261123; recorded [0.2162, 0.538], json lines 1281-1283,
   reproduced here), E n H players with >= 20 games of each colour (46), 100 pools per mode, seeds 9000 + sd,
   modes 0 / 0.35 / 1.0; M8 = threshold 0.80, same colour, |E[r_i] - E[r_j]| <= 2.5, L_new = L_norm + 0.85 A L_norm,
   change i only if i and all its neighbours have the same new prediction and it is one of the group's original
   predictions (04 lines 691-748); Neon's decode (argmax of P @ score matrix) and rank-balanced score. Pass = in the
   0% mode exactly 0 questions changed in all 100 pools AND mean gain > 0 in the 35% mode. Recorded: Neon's numbers
   (json 1452-1517) next to ours. The merge is applied on the test pools only if this passes AND the judgement
   passes; c49 player-side vectors of the test games (`data/prereg56/emb_rank`), `p64_lib.cc_whiten_sim` lam 0.30.
   On the test pools, a question M8 changes gets the probability row softmax(L_new) and its rule-B prediction must
   equal M8's new prediction (checked); every other row keeps the step 1-3 probabilities.
9. **Gates** = `p23_lib.gates` vs Q64 (G1 >= +0.005, G2 lower bound > 0, G4 single colour lower bound > 0 and B, W
   each >= -1e-12). Recorded: vs Q25, by n, by class, by HD group (questions with a share of HD = 1 games 0 / > 0 /
   > 0.25), fitted weights by fold, the term alone. Hard stop 2026-11-23 12:00. Judged once: `eval` refuses when
   `eval.json` or `data/prereg65/eval_started.json` exists; NOTES, frozen.json, seam.json and the code committed
   and clean.
10. **Data touched** (report must say): inner used for the 4th time (prereg58, 59, 64, 65). OPP_P1_v2 and M_P1 were
    trained / estimated by Neon on probe + V role-A players (V = the validation players), never on H or inner
    (02a lines 129-145; checked here: no inner player among probe / V keys, recorded in seam.json). The seams use
    S25-holdout players H (D n H, E n H, H rows) - no inner, no test answer. c49 itself was trained on validation
    players (prereg49). Never data/local, never tools/score.py.
11. **Resources**: CPU only, one python process, <= 3 GB (p58_lib.limits: 4 cores, polite priority); no run between
    21:25 and 23:45 on 2026-11-21 (coordinator).
