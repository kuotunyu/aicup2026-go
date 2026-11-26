# prereg59 - control checks on the kernel players (record only)

Plan `CONTROLS_PLAN.md` (f972729, before any number); numbers `controls.json` (`p59_controls.py`, one run, 240 s,
peak 1.98 GB). 900 kernel players (D, E), cross-fitted by player fold, 20 questions per player and n; score = the
term alone + rank prior, competition score. Standard error by player about 0.006 - 0.010. Nothing here changes the
frozen kernel or the inner judgement; no inner game and no test game was read.

Seams: (a) kernel.json's standalone record 0.5814 / 0.6367 / 0.6755 reproduced bit for bit; (b) the 60 grid
likelihoods bit for bit; (c) direct soft-max vs `p59_lib.combine` max abs diff 1.8e-15.

| control | any n=5 | 10 | 15 | 20 | mono HD<=1 n=5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| prior only | 0.1502 | 0.1502 | 0.1502 | 0.1502 | 0.1485 | 0.1485 | 0.1485 | 0.1476 |
| 1 real q | 0.5842 | 0.6373 | 0.6592 | 0.6735 | 0.5899 | 0.6428 | 0.6618 | 0.6813 |
| 2 constant q | 0.1502 | 0.1502 | 0.1502 | 0.1502 | 0.1485 | 0.1485 | 0.1485 | 0.1476 |
| 3 shuffled within rank / HD / colour | 0.6273 | 0.7043 | 0.7423 | 0.7651 | 0.6345 | 0.7116 | 0.7527 | 0.7755 |
| 3b rank centroids (no person) | 0.5474 | 0.6155 | 0.6451 | 0.6642 | 0.5488 | 0.6188 | 0.6475 | 0.6692 |
| 4a top-1 class removed | 0.5266 | 0.5546 | 0.5699 | 0.5809 | 0.5311 | 0.5636 | 0.5744 | 0.5832 |
| 4a top-10 classes removed | 0.4949 | 0.5004 | 0.5078 | 0.5108 | 0.4966 | 0.5051 | 0.5074 | 0.5090 |

Per rank class (mono, n = 10; D C B A 1D 2D 3D 4D 5D 6D): real 0.93 0.66 0.68 0.63 0.58 0.46 0.47 0.50 0.54 0.96;
shuffled 0.99 0.69 0.74 0.74 0.67 0.55 0.57 0.58 0.60 0.99; centroids 0.81 0.68 0.67 0.56 0.48 0.50 0.49 0.54 0.58
0.88; top-1 removed 0.95 0.56 0.52 0.50 0.46 0.42 0.38 0.38 0.48 0.97 (all classes in controls.json).

One sentence per control:

1. Real q: the term alone scores 0.58 - 0.68 against 0.15 for the prior, the same on the official question shape
   (one colour, HD <= 1) as on any games.
2. Constant q gives exactly the prior-only score: the objective and the HD x colour cells do not manufacture
   information by themselves (Codex's worry about the un-normalised objective does not show at score level).
3. Shuffling the opponents among players of the same rank, HD group and colour does not lower the score, it raises
   it by 0.04 - 0.09: nothing depends on the player's own particular opponents; a real player's opponents are
   correlated (his games are not independent draws given his rank), which costs score. The term is a rank-level
   signal.
3b. Ten rank centroids instead of 6,185 persons keep 92% (n = 5) to 98% (n = 20) of the gain over the prior: most
   of the signal is "how strong does the opponent's play look", not "which labelled person is it". Persons help
   the two end classes (D 0.93 vs 0.81, 6D 0.96 vs 0.88) and hurt nothing much elsewhere.
4a. Removing each game's top-1 class keeps 87% (n = 5) to 82% (n = 20) of the gain, removing the top 10 keeps 79%
   to 69%: the single recognised person is not what carries it, but recognition adds a part that grows with n.
   The top-1 class has the player's own rank in 25% of the games and is within one rank in 59%.
4b. Questions drawn only from the highest tercile of max_s q (or of the top-1 class probability) score 0.63 / 0.67
   (0.63 / 0.70) for n = 5 / 10, from the lowest tercile 0.51 / 0.59 (0.54 / 0.57): confident games carry more,
   unconfident games still carry most of it (prior only 0.15 everywhere).
5. Normalised objective (held-out log posterior of the true rank, fold prior, players equally weighted): the chosen
   point (50, 30, 2) is 3rd of 60, 0.0014 below the best (80, 50, 2), range of the grid 0.053; per game without
   prior (kernel.json's record) also 3rd of 60, 0.0031 below the best. The registered objective and the normalised
   one agree; nothing is re-chosen.

Reading (the rule was written before the numbers): this is the "rank-level information" case. The term is not an
artefact (control 2) and its content is mostly the opponent's apparent strength read from S0's vector, combined
with P(opponent's rank | player's rank, HD, colour); recognising a specific labelled person contributes roughly a
tenth to a fifth of the standalone gain. So the name "opponent identity" overstates it, and the transfer to the
test set does not hinge on the test opponents being labelled players; it hinges on the test games' q looking like
the kernel / inner ones (the shift check of the test-day tool).

Limits. (i) These are standalone scores of the term; how much is new next to Q25's seven terms is only what the
inner judgement measured (+0.0083 [+0.0015, +0.0154]); a strength signal read from the same games overlaps with
what Q25 already reads. (ii) S0's input planes hold both players' stones, so the "opponent-side" vector may also
carry the player's own strength; the controls cannot separate that (it is not a label leak, the own prototype is
left out, but it would be overlap with Q25 rather than new information). (iii) The same folds chose (tau1, tau2, b);
the standalone numbers are development estimates. (iv) Control 3b's tau (50) was chosen on the same folds.
