# prereg59 (competition 2): opponent-identity term, second version

Date 2026-11-20. Written after prereg58's single inner judgement (not adopted: Q58 - Q25 +0.00059
[-0.0054, +0.0057]; single colour +0.0029 [-0.0002, +0.0058]) and before any number of this version.
This version was designed after seeing that judgement; the report must say so (inner has been reused).

## What prereg58 showed (results/prereg58/explore/eval.json, RESULTS.md)

1. The fitted weight of the term sat on the fusion's upper limit (4.0) in all 5 folds: the per-game mean of a
   nearly flat log-likelihood is too small in scale, the limit was binding.
2. The two end classes lost (D -0.031, 6D -0.025) while the middle classes gained (+0.006 .. +0.016): the
   shift kernel K[s - r] is wrong at the ends (a D player's opponents cannot be weaker than D).
3. Only S0's 3,685 training players could be recognised (46% of the labelled players).

## Changes (everything else as prereg58)

A. Classes = S0's 3,685 head rows + one prototype per labelled player that S0 never saw and that is not a comp-2
   inner player: D, E and the validation players (prototype = L2 mean of S0's player-side game vectors of all his
   training games; inner players get no prototype, neither in the judgement nor on the test set).
   Logit of a head class: tau1 * cos(o_g, w_j); of a prototype: tau2 * cos(o_g, p_k) + b. q_g[s] = soft-max over
   the union, summed by rank class. When the game belongs to a kernel player, his own prototype is left out.
   (tau1, tau2, b) from {30, 50, 80} x {30, 50, 80, 120} x {-2, -1, 0, 1, 2} by 5-fold (by player) likelihood on
   the kernel players (D and E minus inner), as in prereg58.
B. Full conditional table instead of the shift kernel: M[h, c][r][s] = P(opponent's class rank s | player's rank r,
   HD group h, player's colour c), soft counts on the kernel players, add-0.5 smoothing per row.
C. Term l(r) = 10 * mean_g log(sum_s q_g[s] M[h_g, c_g][r][s] + 1e-9): the same per-game mean as prereg58 with a
   fixed factor 10, so that the fusion's weight limit is not binding. Nothing else in the fusion changes.

Frozen before any inner number: `kernel.json` (M, tau1, tau2, b, counts, likelihoods, cross-fitted standalone score
on the kernel players for n = 5 / 10 / 20).

## Judgement (inner, once)

Seam: Q25 0.7456519871546352 bit for bit. Q59 = Q25 + this term, gates as prereg58 / prereg29: G1 >= +0.005,
G2 lower bound > 0, G4 single colour (lower bound > 0 and black, white each >= 0). Reported: by number of games,
by rank class, fitted weight (and whether it is at the limit), the term alone. Hard stop 11/23 12:00.
If adopted: freeze the 8-term fusion on all inner fit questions (as PREREG_25), test-day tool (CPU + S0 vectors
of the test games' opponent side, already in data/prereg58/emb_opp_rank), last competition-2 upload.

## Vectors still needed

S0 player-side game vectors of the validation players' training games (for their prototypes): one GPU job, run
when the 4090 is free (after prereg57's night chain, before prereg51-L resumes).

## Not allowed

As prereg58 (`data/local`, `tools/score.py`, public scores, competition-1 test data).
