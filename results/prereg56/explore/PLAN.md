# prereg56 (explore): structural probe - do the same people appear across the pools?

Date 2026-11-20. Written before any number below is computed. Nothing here touches E-pool judgements, data/local, or
tools/score.py; no submission is built from this step.

Motivation: on 11/20 the public leaderboards show a detached top group in both competitions (comp 1 top 0.93, comp 2
top five >= 0.82 vs 0.78 for #6) and the top teams of the two competitions carry adjacent team numbers (same people).
A structural property that our held-out simulations (E pools = players never seen anywhere else) cannot see would
explain this. User decision 11/20: "規則沒有明確說不行的話就是都可以" - probe first.

Hypotheses (each tested per pool; public and private are never mixed for anything that could become a method):
- H3: the 600 new candidates (candidates2, 10 games) include players of the original pool of the same split
  (400 candidates x 100 games, 400 questions x 5-20 games).
- H3q: the new questions' players are in the original pool of the same split.
- H1a: comp-1 pool players (old or new) are training players (c49 was trained on 5,771 of them: class head).
- H1b / H5 (record only, cross-competition; no method is built on it without a separate registration and the user's
  explicit go): comp-2 test questions' players are training players, or are in the comp-1 pools.

Vectors: c49 (data/prereg49/S0/ckpt/step_320000.pt, raw, tta1), per game, same code path as the official pools
(p49_run.py embed), for: the original comp-1 pools, the comp-2 test questions, and D (seen by c49) as the positive
control for H1a. E (store_E, never seen) is the null.

Statistic: a set's vector = L2(mean of its games' L2 vectors) (black games only where the other side is all black).
For set A against a gallery G: s(A) = max cosine to G's prototypes, and for H1 the max mean class probability of the
c49 head. Null: E players' sets of the same size against a gallery of the same size made of other E players
(100-game or 10-game prototypes) -> 99th percentile threshold; positive control: the same E player's disjoint games in
the gallery (H3) / D players' games (H1a). Report: fraction of sets above the null's 99th percentile, and the positive
control's fraction, per pool.

Decision: if a hypothesis shows >= 20% of sets above threshold (null 1%), write a registration for a method that uses
it (simulated on E by planting the same structure) before building anything. Otherwise record "not present".

## Addition 1 (2026-11-20, before any opponent-side number)

H7 (record only): a large share of the opponents are a few recurring identities (e.g. server bots of fixed level; the
data conventions point to a server where the organiser's bots play). Vectors: the same c49 game vectors computed for
the opponent's side (`p56_opp.py`), for the comp-2 test questions and for D. Statistic: for each opponent-side game
vector, the cosine to its 10th nearest neighbour among opponent-side vectors of games of OTHER players; compared with
the same statistic for player-side vectors (different players). If opponents were a few bots, the opponent-side
neighbours would be far closer. Also: size of the largest connected groups at a cosine threshold = the 99.9th
percentile of player-side different-player pairs. If present, the next step is a registration for a rank feature
"opponent identity cluster x HD x colour" trained on the training set (opponent side of training games).

## Addition 2 (2026-11-20, before any number of this kind)

H8 (record only): the OPPONENTS in the comp-2 test games (and in training players' games) are labelled training
players. If so, recognising the opponent (c49 class head on the opponent-side game vector) gives the opponent's known
rank, and with HD and colour an anchor for the player's rank. Statistic per single game: max cosine of the game vector
to c49's class weights. Null: E players' own single games (not in the class set). Optimistic positive: D players' own
single games (in the class set, and these very games were trained on). Tested sets: opponent side of the comp-2 test
games; opponent side of D players' games; player side of the comp-2 test games (expected = null by H1).
Also, for D (labels known): among opponent-side games above the null's 99th percentile, the distribution of
(rank of the recognised class - rank of the D player), against the same for games below the null's median.

## Addition 3 (2026-11-20, after h8.json, before any number of this kind)

Size of the H8 signal on D (labels known; D is in c49's class set as PLAYERS, so only the opponent side of D's games
is used and D's own classes are removed from the head before the soft-max - the opponent can never be "recognised" as
the D player himself). Per game: q_g = soft-max(tau * cos(opponent vector, class weights)) summed by class rank.
Per question (n = 5 / 10 / 20 random games of one D player, 20 draws per player): log-likelihood of rank r =
sum_g log(sum_s q_g[s] K[s - r]), K = rank-difference kernel estimated on the other folds (5 folds by player),
prior = rank frequencies of the training folds, decoded by expected competition score (1 / e^-1 / 0).
tau in {10, 20, 30, 50} chosen by training-fold likelihood. Report the competition score per n. Record only: this is a
standalone estimate, not a judgement; HD and colour are not used yet.
