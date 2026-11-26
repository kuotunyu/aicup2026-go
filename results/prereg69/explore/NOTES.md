# prereg69 - narrow readings of PLAN.md

Written 2026-11-22 before any code of this registration and before any number of it. PLAN.md is not edited.
Settings with pointers are in `frozen.json`. prereg64 / prereg65 code is imported, never edited.

1. **M8** = `results/prereg65/explore/p65_lib.py` `m8` unchanged (threshold 0.80, same colour, |E[r_i] - E[r_j]| <= 2.5,
   L_new = L_norm + 0.85 A L_norm, agree-and-support rule), similarity `p64_lib.cc_whiten_sim` lam 0.30 on the c49
   player-side vectors (prereg65 frozen.json `merge_M8`).
2. **Why the gate base is a proxy.** 64's anchor A = log_softmax(Q25) needs Q25's per-game inputs (the R20 CNN base
   m, OP, O4, P1TO, SET) and 64's OPP3 needs c49 opponent-side vectors. On the E n H players of the simulation only
   the S25 set model inputs (prereg39 P1 blocks), the S0 opponent-side vectors (`data/prereg58/emb_opp_D|E`) and the
   c49 player-side vectors (`data/prereg49/emb/.../store_E.npz`) exist (prereg64 NOTES 1: no c49 opponent-side
   vector of E exists; making one needs the GPU, which this run does not use). So "64's fusion" is read as 64's
   fusion form with its frozen weights on the closest available inputs:
   - A_proxy = log_softmax(lp_S25), lp_S25 = the S25 set model's question log-probabilities (`p19_forward.set_term`
     with the 3 frozen S25 seeds, the `lp` output; the same call prereg65's simulation makes);
   - O_proxy = 64's OPP3 form with S0 in place of c49: q = `p59_lib.combine` of `class_parts` (S0 head
     `data/prereg58/s0_head.npz`, prereg59 prototypes, own prototype left out, exactly the arrays prereg65 section 2
     builds) at 64's (tau1, tau2, b) = (50, 30, 1); table = prereg65's `oof_R` with bayes False, lam 0 (prereg59 /
     64 table: add-0.5 rows, no shrink, no / mean) estimated 5-fold by player fold on the same (D u E) games
     (`ok`) as prereg65 section 2, so the E n H players' own games never enter their table; l_g = log(R + 1e-9);
     question value 10 x the flat mean (64's `question_term`);
   - Z = w_A A_proxy + w_O O_proxy with 64's frozen (w_A, w_O) = (0.7667121789025336, 0.4940900911252534)
     (`results/prereg64/rank/frozen_config.json`); no refit. The M8 input is lp_base = log_softmax(Z).
3. **Pools** = prereg65's simulation exactly (E n H players with >= 20 games of each colour, 100 pools per mode,
   seeds 9000 + sd, modes 0 / 0.35 / 1.0, same rng consumption): the loop is copied from `p65_eval.neon_section2_3`
   and evaluates both bases on the same pools in the same pass.
4. **Seam S (before the gate is read)**: the same loop with prereg65's own base (S25 + OPP4B_S0, section-2 weights)
   reproduces `results/prereg65/explore/merge_sim.json` section 2 (rb, weights) and every section-3 number
   (base_rb_mean, gain mean / min / max, changed mean / max, pools with any change, edges mean) exactly (==). If not,
   STOP: no gate number is used.
5. **Gate** (PLAN): with the 64-proxy base, in the 0% mode 0 questions changed in all 100 pools AND the 35% mode
   mean gain > 0. Recorded only: 100% mode, base rb means, edges.
6. **Seam T (test tool)**: `tools/t69_rank.py` recomputes Q64 through `tools/t64_rank.py`'s functions; with the merge
   off, `rank_test_probs.csv` and `submission_rank.csv` equal `results/test64/rank/` byte for byte (else exit 4).
   M8's pred0 must equal rule B of the 64 probabilities on every question. A question M8 changes gets the row
   softmax(L_new) (prereg65 NOTES 8) and its rule-B prediction must equal M8's new prediction; every other row keeps
   64's bytes. Public and private are merged separately.
7. **Data touched**: no inner, no data/local, no tools/score.py, no GPU. The simulation uses S25-holdout E players
   (as prereg64 / 65). Test pools: the raw question files and the c49 player-side vectors 64 already reads.
8. **Resources**: CPU, one python process, p58_lib.limits (4 cores, <= 3 GB); available RAM >= 7000 MB checked before
   each run.
