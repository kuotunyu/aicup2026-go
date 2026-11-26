# prereg61 (competition 1, new questions): prereg60's input, trained long enough

Date 2026-11-21. Written before any code and any number of this run.

prereg60 (18 planes = old 17 + the move played; samples at every move incl. the first 7; fine-tuned from c49 for
30,000 steps) scored E 0.7069 vs v2-7 0.7036: +0.0034 [+0.0009, +0.0057], 14 of 20 pools - positive but below the
gate. A zero-initialised input channel and new opening samples had only 30,000 steps (9% of c49's own training).
Earlier evidence in this project: the same network went 0.4241 -> 0.5868 on E when training went 40k -> 160k steps
(prereg37) and kept improving until the learning rate reached zero.

## Method

Exactly prereg60's code and recipe (`results/prereg60/explore/p60_lib.py`, `p60_run.py`; start from c49 raw, widened
first convolution, T49 sampler, 96 x 24, AdamW, wd as base, fp16, seed 0) with only the schedule changed:
200,000 steps, learning rate 1e-3 (half of c49's own peak), 2,000 warm-up steps, cosine to 0. Checkpoints at
50,000 / 100,000 / 150,000 / 200,000 kept; the raw weights at 200,000 are the model. Output `data/prereg61/`.
Back end and P1 term as v2-7, nothing retuned.

## Judgement (E, 20 pools, once)

A61 vs v2-7: adopt if mean difference >= +0.01, bootstrap lower bound (10,000, seed 43) > 0, >= 18 of 20 pools.
E use number = committed E judgements + 1. Seam: v2-7's E score bit for bit through the judging path.
If the run cannot finish, the latest kept checkpoint (100,000 or 150,000) may be judged instead ONLY if that is
decided and committed in a revision before any vector of this run is scored; otherwise dropped.
Hard stop: judged by 11/23 12:00.

## Apply (if adopted)

Vectors for pub2 / priv2, v2-7 pipeline, original 800 rows unchanged, usual checks, upload.

## Cost

About 18.5 h on the 4090 alone (3 steps/s) + 0.4 h vectors. Runs alone on the GPU; RAM guard as prereg60.

## Revision 1 (2026-11-21 14:50, before any vector or score of this run)

The 4090 run stopped at step 102,500 (`data/prereg61/S0/ckpt/step_102500.pt`, saved on a STOP for another
competition's job that needs the 4090 for several hours). The rest of the training (steps 102,500 -> 200,000) runs on
Colab G4 from that checkpoint, with the same code (prereg60/61 files of the p62 overlay on Drive, sha-checked), the
same config (the resume refuses any config difference), seed 0, the same sampler stream (batch 320,000 + s) and the
same schedule; only the device and the resume-checkpoint spacing (5,000 instead of 1,000 steps) change. The
200,000-step checkpoint and the train record come back from Drive; vectors and the single judgement run on the 4090
as registered. The 4090 run is not resumed (its watcher was stopped before the Colab run started).

## Revision 2 (2026-11-21 20:20, before any vector of the final model)

The standalone E judgement is dropped: the model becomes a member of prereg67's ensemble on top of v2-8 (adopted
after this registration). Only E + pool vectors are computed.
