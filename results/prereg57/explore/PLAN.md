# prereg57 (competition 1, new questions): closed-set adaptation to the candidate pool

Date 2026-11-20. Written and committed before any code exists and before any number is computed.

## Why

- Official Q&A Q15 (read 2026-11-20, `data/reference/qa_q15_20261120.md`): test data may be used to train or fine-tune
  ("可以。本競賽不限制資料的使用"), to be described in the report. Q10: candidate games may be used freely as long as
  public and private pools are not mixed and everything is automatic. Tutorial README: the four pools (`pub_`,
  `priv_`, `pub2_`, `priv2_`) must not be mixed.
- From v2-5 to v2-7 the held-out simulation (E pools) gained +0.044 but the official new questions (2 x public - 1)
  only about +0.009: making the generic model stronger barely transfers. The candidates of a pool carry identity
  labels (600 ids x 10 games); so far only the scoring back end (whitening / LDA) uses them.

## Method (one model per pool, pools never mixed)

For a pool P (candidates with ids, questions without labels) and a base model M (trunk + cosine head over its training
players):

1. New head = M's class weights followed by one row per candidate of P, initialised with the L2-normalised mean of
   M's game vectors of that candidate.
2. Fine-tune the whole network with the base model's loss (cosine softmax, same scale, same 24-move segments,
   96 segments per batch): a fraction `share` of every batch comes from P's candidates (uniform over candidates, then
   a uniform game of that candidate, the candidate's own moves), the rest from the base model's original training
   sampler. AdamW, weight decay as the base, constant learning rate `lr` after a 200-step linear warm-up, fp16, EMA
   0.999. Seed 0. Questions are not used for training.
3. Vectors of P's games (candidates and questions) come from the EMA weights at step `n`; scoring is the adopted
   back end unchanged (v2-7: `results/prereg43/explore/tune.json` structure and numbers, P1 term unchanged).

## Development (D pools; base S0 of prereg43, which never saw D)

- Pools: the first 2 of the 10 D pools used by prereg43's select.
- Grid: (share 0.25, lr 2e-4), (share 0.5, lr 2e-4), (share 0.5, lr 5e-4); checkpoints at n = 1,000 / 2,000 / 4,000 /
  6,000 -> 12 settings. Score = prereg43's D quick screen (CNN term only) and the v2-6 D score (with P1), both pools
  averaged; reference = the unadapted S0 on the same 2 pools.
- Choose the setting with the highest v2-6-style D mean; ties -> fewer steps, then smaller share. Commit `dev.json`
  with the choice before any E vector of this method exists. If no setting beats the reference by >= +0.01 on both
  pools, stop: not adopted, E is not touched.

## Judgement (E pools; base c49, which never saw E) - once

- Pools: the first 10 of the 20 E pools of the v2-7 judgement (`results/prereg49/explore/eval.json`), each with its
  own adapted copy of c49 (chosen share / lr / n).
- Seam first: the unadapted c49 through this code path reproduces v2-7's per-pool scores on those 10 pools bit for
  bit.
- Adopt if mean(adapted - v2-7) >= +0.01, the bootstrap lower bound (10,000 resamples of the 10 pools, seed 43) > 0
  and at least 9 of 10 pools are higher. This is E use number (7 + number of existing prereg51 / prereg54 E
  judgements at that time) + 1; the report states that E has been reused.
- If prereg51-L or a prereg54 ensemble is adopted in the meantime, this method is still judged against v2-7 with c49
  as its base (the adaptation starts from c49); combining is a separate registration.

## Apply

pub2 and priv2 separately: adapt c49 to that pool's candidates only, vectors for that pool's games, back end and P1
term as v2-7, top-5 per question; the original 800 rows are copied from v2-7 unchanged. Checks: 1,600 rows, ids, the
original 800 rows byte-identical to v2-7.

## Cost and schedule

4090: about 4.5 steps/s alone -> dev 3 runs x 2 pools x 6,000 steps ~ 2.3 h (+ 24 short embeddings); judgement
10 x n steps (n <= 6,000: <= 3.7 h); apply 2 runs. Starts when prereg51-L has released the 4090 (expected the night
of 11/20), judged by 11/21 evening; hard stop 11/23 12:00 (after that the last upload stays as it is).

## Not allowed

No reading of `data/local`, no `tools/score.py`, no manual labels, no mixing of pools, no use of questions' labels
(there are none) or of public scores to choose settings.

## Revision 1 (2026-11-20, before any adaptation has run and before any number of the method)

Method step 1: the new head rows are the L2-normalised candidate means multiplied by the median L2 length of the
base model's class rows (about 10.6 for S0 / c49). Reason: the cosine head ignores a row's length, but under AdamW a
row of length 1 would turn about ten times faster than the existing rows; the rows should move at the same rate.
Readings fixed by the implementation and accepted here (agent report, commit 035967d): the base part of each batch
continues the base sampler's stream after step 320,000; start from the raw weights with a fresh EMA and optimiser;
a candidate game without a sampled move is redrawn from the same candidate; the dev gate requires the chosen setting
to gain >= +0.01 on each of the two D pools; ties: fewer steps, smaller share, smaller lr; the quick screen is
reported only; E use number = (number of committed E judgements) + 1 = 8 if nothing else is judged first.
