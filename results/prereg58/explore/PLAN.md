# prereg58 (competition 2): who is the opponent? - an opponent-identity term added to Q25

Date 2026-11-20. Written before any inner number, any kernel and any S0 opponent-side vector exists.

## Finding that motivates it (prereg56, record only, commits cf43955 and before)

- The players of the test questions are not training players (1.4% of their games above the null's 99th percentile),
  but their OPPONENTS often are: 6.2% of the opponent-side games of the test questions and 6.1% of the opponent-side
  games of D players are above that threshold (null 1%; optimistic positive 9.3%).
- For D, recognised opponents have the same rank class as the player in 50% and within one class in 89% of the games.
  A predictor that uses nothing but the opponent-side class posterior scores 0.565 / 0.632 / 0.669 (5 / 10 / 20
  games) on D with a crude kernel, no HD, no colour.
- The current system (25) only estimates the opponent's strength from his moves in that one game.

## Method

- Identity model: S0 of prereg43 (`data/prereg43/S0/ckpt/step_320000.pt`; classes = its 3,685 training players T36;
  it never saw D, E, the comp-2 inner players or the validation players). Trained on official training games only.
  It is the same author's network from competition 1; the report says so.
- Per game g of a question: o_g = S0's game vector of the OPPONENT's moves (same vector definition as competition 1,
  tta 1). q_g[s] = sum over classes j with rank s of softmax_j(tau * cos(o_g, w_j)).
- Kernel K[h, c][d] = P(opponent's class rank - player's rank = d | h = HD group 0 / 1 / 2+, c = player's colour),
  estimated on the labelled players that S0 never saw and that are not comp-2 inner players (D and E minus inner),
  with add-0.5 smoothing; tau from {20, 30, 50, 80, 120} by 5-fold (by player) likelihood on the same players.
  Frozen (file + commit) before any inner number.
- Question term: l(r) = sum_g log(sum_s q_g[s] K[h_g, c_g][s - r] + 1e-9), r = 10 classes.
- Fusion: l enters Q25's fusion as one more term exactly the way prereg44 added its 8th term (same fitting
  procedure, same cross-fitting, rule B decision unchanged); nothing else in Q25 changes.

## Judgement (inner, once)

Seam first: Q25 0.7456519871546352 reproduced bit for bit through this code path. Then Q58 vs Q25 on inner with the
gates of prereg29: G1 mean gain >= +0.005, G2 bootstrap lower bound > 0, G4 single-colour gain lower bound > 0.
Also reported, not gates: gain by number of games (5 / 10 / 15 / 20) and by rank class; the term alone.
Adoption needs all three gates; then freeze, write PREREG_58, test-day tool (CPU + S0 vectors of the test games'
opponent side), upload as the last competition-2 file. Hard stop 11/23 12:00.

## Disclosure

Uses the validation-set-free inner protocol only (competition 2 has no scoring opportunity left); uses a network
whose training set contains no inner / validation player; the report states that the identity network comes from the
same author's competition-1 work and is trained on the official training data only.

## Not allowed

`data/local`, `tools/score.py`, public scores for any choice, competition-1 test data.
