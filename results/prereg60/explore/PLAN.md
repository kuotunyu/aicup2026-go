# prereg60 (competition 1, new questions): the network sees the move that was played

Date 2026-11-20. Written before any code and any number. Source of the idea: three independent reviews on 11/20
(`data/research/intel_1120_methods.md` item 5, `data/advice/antigravity_1120.md` H3, behavioural-stylometry papers
that feed state AND action).

## The gap (results/prereg36/explore/p36_lib.py, used by every CNN since prereg36)

- `planes_at`: the 17 input planes are the 8 positions BEFORE the target's move t. The move the player chose at t is
  not in the input (it only shows up in the history of his move t + 2), so a per-move vector describes the position
  the player was in, not what he did there.
- `target_moves`: the first 7 non-pass moves of a game give no sample at all (history not full) - the opening moves.

## Method

c49 (`data/prereg49/S0/ckpt/step_320000.pt`, raw) with two changes, then fine-tuned:
1. Input: 18 planes = the 17 old ones + one plane that is 1 at the point of the target's move t, 0 elsewhere.
   The first convolution gets one more input channel, initialised to 0 (so at step 0 the network equals c49).
2. Samples also at the target's moves with an incomplete history (missing history positions = empty boards), in
   training and in the game vectors.
Fine-tune on c49's own training set and sampler (T49, 96 segments x 24 moves, cosine head, same loss, weight decay),
30,000 steps, AdamW, learning rate 5e-4 with 500 warm-up steps then cosine to 0, fp16, seed 0; the raw weights at
step 30,000 are the model (checkpoints at 10,000 / 20,000 / 30,000 kept, only 30,000 is used).
Game vector, back end and P1 term exactly as v2-7 (`results/prereg43/explore/tune.json`), nothing retuned.

## Seams (before training)

- With change 1 only and the new channel at 0, the game vectors of 200 E games equal c49's stored vectors (fp16
  tolerance as in earlier seams: same `count`, max abs diff <= 0.05).
- v2-7's E score 0.7035676497986656 is reproduced bit for bit by the judging path with c49's stored vectors.

## Judgement (E, 20 pools, once)

A60 = the fine-tuned model's vectors in v2-7's pipeline. Adopt if mean(A60 - v2-7) >= +0.01, bootstrap lower bound
(10,000, seed 43) > 0 and >= 18 of 20 pools higher. E use number = committed E judgements + 1. Reported: by number of
games (1 / 2 / 3 / 5). No development set is used (the recipe is fixed here); the report says E has been reused.
If prereg57 is adopted first, this is still judged against v2-7; combining the two is a separate registration.
Hard stop: judged by 11/22 12:00, otherwise dropped.

## Apply (if adopted)

Vectors for pub2 / priv2 with the fine-tuned model, v2-7 pipeline, original 800 rows unchanged, usual checks.

## Cost

About 3 h on the 4090 alone (30,000 steps at ~3 steps/s) + E vectors (~17 min) + pool vectors. Runs after
prereg57's night chain and prereg59's short embedding; prereg51-L's remaining 60k steps move behind it (L will then
miss its own 11/21 12:00 deadline and is dropped unless the 4090 is free earlier).

## Not allowed

`data/local`, `tools/score.py`, public scores for choices, any retuning after seeing E.

## Revision 1 (2026-11-20, before any training and before any number of the method)

Seam (a): the tolerance of the comparison with c49's stored vectors is max abs diff <= 0.15 (not 0.05), together with
the recorded same-device check (widened net vs the frozen 17-plane code, both CPU fp32: max abs diff 0.0, commit
fb15f0e). Reason: the stored vectors were computed in fp16 on the GPU; fp32 against fp16 gives 0.0696 on 200 games
for vectors whose entries reach 80 (relative 0.0005, minimum cosine 0.999998), which says nothing about the change.
The readings in NOTES.md (commits 1a9ff21, 56bc2e6) are accepted.
