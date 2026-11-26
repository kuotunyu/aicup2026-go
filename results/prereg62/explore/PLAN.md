# prereg62 (competition 1, new questions): prereg61's recipe on Colab G4 - seeds 1 and 2, and a run twice as long

Date 2026-11-21. Written and committed before any code and any number of these runs. prereg61's own training is
running on the 4090 and has no judgement yet; nothing of it is read here except its code.

## What is trained (three arms, one wrapper, three Colab G4 runtimes)

All three are exactly prereg61's code and recipe (`results/prereg61/explore/p61_run.py` over prereg60's
`p60_lib.py` / `p60_run.py`: start from c49's raw weights `data/prereg49/S0/ckpt/step_320000.pt` with the first
convolution widened to 18 planes, c49's head, T49 players and cache, 96 games x <= 24 moves, AdamW wd 1e-4, fp16,
peak learning rate 1e-3, 2,000 warm-up steps, cosine to 0 at the last step, raw weights of the last step = the
model). An arm fixes the seed and the number of steps; nothing else differs.

| arm | seed | steps | kept checkpoints | Drive folder | result zip |
|---|---|---|---|---|---|
| S1 | 1 | 200,000 | 50k / 100k / 150k / 200k | `C10/p62/S1/` | `C10/p62/C10_p62_S1_results.zip` |
| S2 | 2 | 200,000 | 50k / 100k / 150k / 200k | `C10/p62/S2/` | `C10/p62/C10_p62_S2_results.zip` |
| X  | 0 | 400,000 | 100k / 200k / 300k / 400k | `C10/p62/X/`  | `C10/p62/C10_p62_X_results.zip`  |

## What the seed controls in the code (p60_run.cmd_train, read line by line)

1. **The sampler stream**: `p60_lib.GameBatches60(..., seed, ...)` draws batch number b with
   `numpy.random.default_rng([seed, b])` - which players, which game of each player, which <= 24 moves of the game,
   which of the 8 board symmetries. Batch s of a run (s = 0, 1, ...) is batch 320,000 + s of that seed's stream
   (prereg60 note 1: the numbering continues after c49's 320,000 batches). This is the only thing that changes what
   the network sees.
2. `torch.manual_seed(seed)`: the V2 network and loss have no random operation (no dropout; classification head
   without sub-centres), every weight is loaded from c49 and the new input channel is set to zero, so this changes
   nothing in the arithmetic; it only seeds the DataLoader workers' base seeds, which the batches do not use.
3. The number written in the checkpoints' and the record's config (`"seed"`).
4. p60_run also compares the seed with the one stored in the c49 checkpoint (0) as part of "is this c49's recipe".
   For seeds 1 and 2 the wrapper itself checks that c49's stored seed is 0 and then lets that one comparison pass;
   every other comparison of that check (variant, spec, blocks, channels, emb_dim, scale, wd, EMA decay, batch
   shape, classes, cache hash, checkpoint sha256) is untouched.

So S1 / S2 differ from prereg61's 4090 run by the sampler stream and by nothing else (same start weights, same
schedule). **X uses seed 0**: its first 200,000 batches are the same batches, in the same order, as the 4090 run's;
its schedule is different from step 2,000 on (cosine over 400,000 instead of 200,000: at step 200,000 the learning
rate is still about half of the peak), so it is not a continuation of the 4090 run and its weights at 200,000 are
not prereg61's.

## Purpose - none of the three is judged on its own

* S1: second member for a score-level ensemble with prereg61's model (as prereg52 / prereg54: each model whitened
  on its own, scores averaged).
* S2: third member of the same ensemble.
* X: answers "does twice the training keep helping" (prereg37: 40k -> 160k steps helped until the learning rate
  reached zero).
* Insurance: if the 4090 run of prereg61 cannot finish, one of these may replace it.

Every use needs its own registration, written and committed **before any vector of the run concerned is scored**:
the ensemble (members, weights, gate) once prereg61's judgement exists; a replacement of prereg61's model only
through a revision of prereg61 committed before any vector of either run is scored; X against prereg61 as its own
registered comparison. Until then no score of any kind is computed from these runs. E-pool uses are counted by
those registrations, not here.

## Mechanics

* Colab G4 (RTX PRO 6000), runtime version 2026.07, one notebook per arm (`results/prereg62/colab/p62_<arm>.ipynb`),
  "run all"; after a disconnect "run all" again resumes from the checkpoint with the highest step number.
* Inputs on Drive: the prereg42 kit (`C10/p42/kit_parts`), prereg51's overlay with the T49 cache
  (`C10/p51/overlay_parts`), and a new overlay2 (`C10/p62/overlay2_parts`: p60_lib.py, p60_run.py, p61_run.py and
  the c49 checkpoint), each rebuilt and sha256-checked in the notebook. The wrapper `p62_run.py` is embedded in the
  notebook (hash-checked).
* Checkpoints every 5,000 steps to `C10/p62/<arm>/ckpt/` (and on a STOP file / free RAM < 2,048 MB, exit code 4).
  Pruning keeps the kept steps of the table **and the two most recent checkpoints**, so the newest resumable one is
  never the only copy while Drive is still syncing; a newest checkpoint that cannot be read is set aside (renamed
  `.bad`) and the one before is used.
* Result zip: the train record, the logs and the last checkpoint only (`step_200000.pt`; X: `step_400000.pt`).
  The other kept checkpoints stay in the Drive folder.
* Vectors (E, pub2, priv2) are computed later on the 4090 from the downloaded checkpoint
  (`p62_run.py embed --arm <arm> ...`, output `data/prereg62/<arm>/emb`), after the registration that uses them.
* Cost (not measured; the 17-plane network ran at about 5.4 steps/s on G4): about 10 h for S1 / S2, about 21 h
  for X. The speed is read from the training log after the first 1,000 steps.
