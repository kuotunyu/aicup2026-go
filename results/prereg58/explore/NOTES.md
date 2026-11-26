# prereg58 - narrow readings of PLAN.md

Written 2026-11-20, before the kernel, before the seam and before any inner number of the new term. PLAN.md is not
edited; where it leaves something open, the reading below is the one the code implements.

1. **How the term enters the fusion.** PLAN: "l enters Q25's fusion as one more term exactly the way prereg44 added
   its 8th term". prereg44's term was `rf.pooled(S, per-game values, "log")`, i.e. the MEAN over the question's
   games, with no extra scale, as an 8th `E` of `Inner.run` (weight in [0, 4], 5-fold player OOF, rule B). So the
   term handed to the fusion is the mean over the question's games of l_g(r) = log(sum_s q_g[s] K[h_g, c_g][s - r]
   + 1e-9) = l(r) / n. Every term of this fusion is a per-game mean; the fusion's own factor exp(a) (n / 12)^gamma
   carries the dependence on n. The sum l(r) itself is only used for the "term alone" record.
2. **Kernel estimate.** The opponent's class rank is not observed, q_g is its posterior. Soft counts as in prereg56
   addition 3: cell (h, c, d) += q_g[r_g + d] for every kernel game g (r_g = the player's rank), d = -9 .. 9; then
   + 0.5 on each of the 19 cells of every (h, c) and normalised per (h, c). The frozen K is the one at the chosen
   tau on all kernel games.
3. **tau.** Folds: player p -> int(sha256("c10-prereg58|fold|" + p)[:8], 16) mod 5. For each tau of {20, 30, 50, 80,
   120}: K from the other four folds, statistic = mean over the held-out games (all five folds) of
   log(sum_s q_g[s] K[h_g, c_g][s - r_g] + 1e-9) (the same expression as l_g at the true rank, as prereg56
   addition 3 used on its training folds). Largest wins, ties -> the smaller tau. The same quantity normalised over
   r (log posterior of the true rank per game) is written next to it as a record and is not used for the choice.
4. **Kernel players.** "D and E minus inner": D (300) and E (600) of prereg36's split, their feature-store games =
   the rows of `store_D.npz` / `store_E.npz` of the launch chain. prereg36's move cache holds fit players only, so
   no inner player can be among them; the code checks that the overlaps kernel/inner, kernel/S0 classes and
   inner/S0 classes are all 0 and stops otherwise.
5. **Games without an opponent-side sample** (count 0: fewer than 8 non-pass moves before the opponent's last move,
   or an illegal move): left out of the kernel; in a question their l_g is 0 for every r (no information), and the
   mean is still over all n games of the question (prereg44 had no such game; this is the only difference).
6. **HD and colour.** HD from the official training data v1.1 through prereg31's table (`data/prereg31/hd_table.npz`,
   `p31_lib.HdTable.join`, every game must join); groups 0 / 1 / 2+. Colour = the player's colour label of the row
   as given (no relabelling for HD 2+, where the organiser says the labels are swapped: K is estimated and applied
   with the same convention).
7. **Gates.** "the gates of prereg29" = `p23_lib.gates` as prereg29 called it: G1 >= +0.005, G2 lower bound > 0, G4
   single-colour lower bound > 0 AND black and white each >= -1e-12. PLAN's short text names only the lower bound of
   G4; the stricter prereg29 definition decides, the lower-bound-only result is written next to it.
8. **Inner vectors.** The inner games are not in prereg36's move cache. They are embedded from
   `data/prereg44/cache/inner` (prereg36 cache format, built by prereg44 from the official training CSVs in
   `inner_keys.npz` order; `rows` checks rank / game_id / player_id / colour row by row and the recorded SHA of
   inner_keys.npz). Only the 73,169 of 73,556 inner games that occur in a fit / eval / single-colour question are
   embedded. Wrapper `p58_rows.py` -> `p43_run.py embed` (= `p42_embed.py`, raw weights, tta 1) with the colour
   flipped, as `p58_opp.py`.
9. **Machine rules.** (a) The earlier Q25 rebuilds peak at 3.8 GB working set because `Inner.__init__` fits Q08
   (`inner_lr_lp` holds the inner feature matrix in float64 several times). Q25 does not need Q02 / Q02H / Q08, so
   `lean_inner` builds the same attributes with the same calls and skips those three fits; a watchdog ends the
   process above 3 GB. (b) The harness refuses BLAS thread variables, so the 4-thread rule is met by CPU affinity
   (4 cores) set after numpy is loaded: the OpenBLAS thread count, hence the arithmetic, is unchanged. Both are
   covered by the seam (Q25 and its single-colour score bit for bit).
10. **Standalone record on the kernel players** (kernel.json): per held-out fold, K from the other four folds at
    the chosen tau, prior = rank frequencies (+1) of the other folds' players, n = 5 / 10 / 20 random games of one
    player (with a sample), 20 draws per player, default_rng(58), decoded by expected competition score.
11. **"The term alone"** (eval.json, record): rule B on softmax of the pooled mean (prereg44's record) and on
    softmax of the sum over games (PLAN's l(r), uniform prior).
