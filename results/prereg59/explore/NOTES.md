# prereg59 - narrow readings of PLAN.md

Written 2026-11-20, before the validation vectors exist, before the kernel and before any number of this version
(kernel players or inner). PLAN.md is not edited; where it leaves something open, the reading below is the one the
code implements. Everything not mentioned is as prereg58's NOTES.md.

1. **Prototype** ("L2 mean of S0's player-side game vectors of all his training games"): every game vector with a
   sample is L2-normalised, the mean over the player's games is taken and L2-normalised again (prereg56's set
   vector). "All his training games": D and E = the rows of S0's existing `store_D.npz` / `store_E.npz` (94,015 and
   193,736 games = every game of those players in prereg36's move cache; same store rows, row by row, as prereg58's
   opponent-side files - checked by the code); validation players = every row of theirs in the prereg49 move cache
   (206,137 games).
2. **Validation players** = `split == 'val'` in the committed `results/split_v1/players_split.csv`, read as prereg49
   read them (`p49_lib.extra_players` -> `tools/c10data.read_split`): 1,600 players. Their rank is the rank of
   their rows in the official training CSVs (the cache). `data/local` is never opened. Overlap with the inner
   players and with S0's 3,685 classes: 0 (checked in `rows` and again in `kernel` / `eval`, which stop otherwise).
   As in prereg25 / prereg49, the validation players' training games and rank labels are used; the report says so.
3. **Kernel players** stay D and E (900 players, prereg58's opponent-side vectors). The validation players only
   add prototypes: no opponent-side vector of their games is made, PLAN lists none under "Vectors still needed".
4. **Grid choice** ("5-fold (by player) likelihood ... as in prereg58"): prereg58's player folds
   (`p58_lib.player_fold`), statistic = mean over the held-out games of log(sum_s q_g[s] M[h_g, c_g][r_g][s] + 1e-9)
   with M from the other four folds; largest of the 60 grid points wins, ties -> the first in (tau1, tau2, b) grid
   order. The version normalised over r is written next to it as a record only.
5. **Leaving out the own prototype**: for a kernel player's game, only that player's own prototype is removed from
   the soft-max; the prototypes of all other players (other kernel players, held-out fold or not, and the
   validation players) stay - their rank is a class label, not the target. On inner (and on the test set) nothing
   is left out; the code stops if an inner player has a prototype.
6. **M**: soft counts cell (h, c, r_g, s) += q_g[s]; + 0.5 on each of the 10 cells of every row (h, c, r); each row
   normalised (a row without any game is uniform). The frozen M is the one at the chosen grid point on all kernel
   games.
7. **Term**: per game l_g(r) = log(sum_s q_g[s] M[h_g, c_g][r][s] + 1e-9) (0 for a game without an opponent-side
   sample); the fusion receives 10 x `S @ l_g` (S = the harness's mean-pooling matrix), as one more `E` of
   `Inner.run`, exactly where prereg58 put its term. q is computed exactly (the head part and the prototype part of
   the soft-max are summed by rank separately and combined; `test_p59.py` checks it against the direct soft-max).
8. **Gates**: `p23_lib.gates` (G1 >= +0.005, G2 lower bound > 0, G4 lower bound > 0 and black, white each
   >= -1e-12), as prereg58 / prereg29.
9. **Seam**: `p58_eval.build_q25` imported unchanged (the lean Inner, <= 3 GB). It only involves Q25, so it is run
   and committed before the kernel exists; the judgement run rebuilds Q25 and checks the two numbers bit for bit
   again before the first number of the new term.
10. **Hard stop**: `eval` refuses to start at or after 2026-11-23 12:00 local time.
11. **Standalone record** (kernel.json): as prereg58's NOTES 10 with M and the own prototype left out;
    default_rng(59). "The term alone" (eval.json): rule B on the scaled mean (what the fusion receives) and on the
    sum over games.
12. **Unattended chain** (`after_embed.sh`): kernel -> commit, seam (already committed: checked and skipped), judge
    once -> commit, `RESULTS.md` written mechanically from eval.json / kernel.json (numbers only) -> commit. A step
    whose committed output exists is skipped; `eval` refuses when `eval.json` or `data/prereg59/eval_started.json`
    exists, so a judgement that died half-way is never repeated by the chain.
13. **Checks that are not results**: `kernel --dry-de` (code path, time and memory with the D / E prototypes only,
    written outside results/), `eval --smoke` (the inner opponent vectors replaced by noise), `results --from-dir`.
    Their numbers are never reported or committed; no value of the real term on inner is computed by them.
14. **Addendum 2026-11-20 16:10 (disclosure, written before the validation vectors and the kernel exist).** The first
    `eval --smoke` run (dry kernel from the D / E prototypes only, inner opponent vectors replaced by noise) still
    passed the REAL HD group and colour of the inner games through the dry table, so its output was not pure noise:
    it showed what an "HD x colour only" version of the term does on inner with that dry table (inner eval +0.0006,
    single colour -0.0012, the term alone 0.18; nothing passed a gate). That is a number on inner that this plan
    did not intend to look at. It changes nothing: the grid, the selection rule, the scale, the gates and the chain
    were committed before it (2938065) and are not edited; the dry table is not the registered kernel. The smoke
    mode now also replaces HD group and colour by random draws. The report mentions this run.
