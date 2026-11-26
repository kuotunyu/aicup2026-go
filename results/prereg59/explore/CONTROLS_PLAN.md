# prereg59 - control checks on the kernel players (record only)

Written 2026-11-21, after prereg59's single inner judgement (eval.json: adopt) and before any number of these
controls. They are asked for by the review `data/advice/codex_1120.md` (sections 二, 四). They cannot change the
frozen kernel (`kernel.json`: tau1 50, tau2 30, b 2, M), the judgement, or the gates. They only say how far the
term's gain can be trusted, and feed one sentence of advice (upload / do not upload) to the main session.
No inner game, no test game, no `data/local`, no `tools/score.py`, no public score. CPU, one process, 4 cores,
<= 3 GB (`p58_lib.limits`). Code: `p59_controls.py` (new; imports p58 / p59 code unchanged).

## Data and questions (shared by all controls)

- The 900 kernel players (D 300, E 600; 287,751 opponent-side S0 vectors), prereg58's player folds, HD join and
  colour exactly as `p59_eval.py kernel`. Everything is cross-fitted by player fold: the table M used for a
  player's games is estimated on the other four folds, the rank prior is the other folds' rank frequencies (+1).
- Questions: for every kernel player, n = 5 / 10 / 15 / 20 games, 20 draws each, `default_rng(5901)`, two shapes:
  - **any**: n of his games without replacement;
  - **mono**: one colour (drawn at random among the colours in which he has >= n games with HD <= 1), n of his
    games of that colour with HD <= 1 - the shape of the official test questions.
  The same game sets are used by controls 1, 2, 3, 4a, 3b; only the per-game term changes.
- Statistic: competition score (1 / e^-1 / 0) of `decode(log prior + sum over the question's games of l_g)`
  (prereg58's decode = largest expected score), mean over questions, per n, per shape and per rank class.
  The players are balanced (90 per rank), so the mean is the rank-balanced mean. Reference: prior only.
- In every control the method is rerun on the control's data: q of the control -> cross-fitted M of that q -> l_g.

## Controls

0. **Seams** (stop if one fails): (a) the standalone record of kernel.json (n = 5 / 10 / 20, `default_rng(59)`,
   its own draw order) is reproduced bit for bit; (b) the held-out mean log-likelihood of all 60 grid points
   equals kernel.json bit for bit; (c) this script's direct soft-max over the 6,185 classes equals
   `p59_lib.combine` (max abs diff <= 1e-9).
1. **Real q** (frozen tau1, tau2, b; own prototype left out).
2. **Constant q**: every game gets the mean q of the training folds' games (test fold: the same vector).
   Reading: must be within 0.01 of prior only for every n and shape. If it is clearly above, the pipeline
   leaks rank through something other than q (HD group / colour cells) and the real number must be read net of it.
3. **Identities shuffled within the rank class**: each game's opponent vector is replaced by the opponent vector of
   a random game of ANOTHER kernel player of the same rank, same HD group and same colour (`default_rng(5903)`;
   a cell with no other player keeps its own vector, counted). The donor game's player's prototype is the one
   left out. Note what this can and cannot show: the term only uses the opponent's estimated rank distribution,
   so the per-game distribution of q given r is unchanged by construction; what is destroyed is anything specific
   to the player's own set of opponents. Reading: shuffled within 0.02 of real -> the information is "what
   opponents of a rank-r player look like" (rank level), not this player's particular opponents; real clearly
   higher -> part of the score needs the player's own opponents (person level, will not transfer to unseen
   opponents); real lower -> the per-player spread of opponents costs score.
3b. **Rank centroids instead of persons** (added, not in the request; record): q from 10 class-rank centroids
   (L2 mean of S0's unit head rows of each rank; no prototype), soft-max of tau * cos, tau from {30, 50, 80, 120}
   by the same cross-fitted held-out likelihood. Reading: about the real score -> the opponent's STRENGTH is read
   from the vector at rank level, no person needs to be recognised; much lower -> recognising persons matters.
4. **Where the information sits.**
   a. q recomputed with each game's top-1 class (after the own-prototype rule) removed, and with its top-10
      classes removed. Reading, with gain = score - prior only: top-1 removed keeps >= 80% of the real gain -> the
      single recognised person is not what carries it; <= 50% -> it is.
   b. Terciles of the kernel games by (i) max_s q_g[s] and (ii) the top-1 class probability (global thresholds,
      real q): questions of n = 5 / 10 games drawn only from a player's games in the lowest / middle / highest
      tercile ("any" shape, players with >= n such games, `default_rng(5904)`). Reported with the number of
      players per cell; the rank mix differs between terciles, so the per-tercile prior-only score is given too.
5. **Normalised objective** (record only): for each of the 60 grid points, held-out mean over players (each
   player = the mean over his games) of log softmax_r(log prior_fold(r) + l_g(r)) at the true rank. Reported:
   value and position of the chosen point among the 60, the best point, the difference; the same for the per-game
   version without prior already stored in kernel.json. "Near the top" = within 0.005 of the best or among the
   best 10. If the chosen point is not near the top, that is written down; nothing is re-chosen.

## How the whole is read (written before the numbers)

- "No information": control 1 on the mono shape is within 0.02 of control 2 for n = 5 and 10 -> recommend against
  uploading 59 (Part B is still finished).
- "Rank-level information" (expected from the design): control 1 well above control 2, control 3 about equal to
  control 1, control 4a top-1 removal keeps most of the gain. The gain is real but does not depend on
  recognising a person; transfer to the test set depends on the test opponents' q looking like the inner ones
  (the shift check of the test-day tool).
- "Person-level information": control 4a top-1 removal loses most of the gain and/or 3b is far below 1. Then the
  gain depends on the test opponents being labelled players the classes contain (prereg56 H8 estimates about
  60%), a risk to state.
- The standalone numbers are development estimates (the same folds chose tau1, tau2, b); none of them is a
  claim about the fused gain, which is the inner judgement only.

Outputs: `controls.json`, `CONTROLS.md`.
