# prereg60 implementation notes (readings of PLAN.md, written with the code and before any number)

PLAN.md is not edited. Where it leaves a detail open, the narrowest reading was taken and is fixed here.

1. **Which batches.** "c49's own training set and sampler ... seed 0": batch s of the fine-tuning (s = 0 .. 29,999) is
   batch 320,000 + s of c49's sampler stream (`numpy.random.default_rng([0, 320000 + s])`, T49 classes, 96 games x at
   most 24 moves, one board symmetry per game). The stream continues where c49's training ended instead of replaying
   c49's first 30,000 batches (the reading prereg57 revision 1 accepted for the same question).
2. **Samples at early moves (PLAN change 2), exactly.** Every non-pass move of the target gives a sample (before: only
   from the 8th non-pass move of the game on). History slot j (0 = oldest .. 7 = the position the target plays in) of
   the sample at move t shows the position before non-pass move t - 7 + j; when that index is negative the slot is
   an empty board (both stone planes zero). The colour plane and the new action plane are as for any other move.
   Per game: training draws a uniform random subset of at most 24 of the target's moves (same call as before, on the
   larger set); the game vector is the mean over all of them. `count` = number of non-pass moves of the target
   (before + the target's moves among the first 7 non-pass moves: 4 for black / 3 for white in an ordinary game, at
   most 7). A game whose replay meets an illegal move still has count 0. Games shorter than 8 moves now get a vector.
3. **Action plane (PLAN change 1).** Plane 17 is 1 at the point of the target's move t (the move the sample belongs
   to) and 0 elsewhere; it turns with the board symmetry like the other planes.
4. **Start and optimiser.** c49's raw weights and c49's head (5,771 classes) are loaded; the first convolution gets an
   18th input channel of zeros; a new EMA (0.999, kept in the checkpoints, not used) and a new AdamW (lr 5e-4, weight
   decay 1e-4 on every parameter as in the base) start at step 0. Learning-rate factor at optimiser step s (0-based):
   (s + 1) / 500 for s < 500, then 0.5 (1 + cos(pi (s - 500) / 29,500)) - 1 at s = 500, 0 at s = 30,000.
5. **Checkpoints.** `step_010000.pt`, `step_020000.pt`, `step_030000.pt` are kept; every 1,000 steps (and on a STOP /
   low-RAM pause) a resume checkpoint is written and the previous resume checkpoint is deleted. Resume picks the
   highest step number. The model is the raw `net` of `step_030000.pt`.
6. **Seam (a).** Run as `p60_run.py embed --ckpt <c49> --source store --players E --limit 200 --no-early --device cpu`:
   the first 200 E games of c49's `store_E.npz`, the widened network with the new channel's weights at zero, the
   action plane really fed (so a non-zero result would show a wiring mistake), early moves excluded, CPU fp32,
   against the stored vectors (4090 fp16). `--zero-action` (plane 17 all zero) exists as a second debugging switch
   and is not needed for the seam.
7. **E use number.** 1 + the judgements on record: the folders through prereg49 that have an `eval*.json`
   (p49_eval's list) + every later `results/prereg5x|6x/explore/eval*.json` carrying `e_pools_use_number` (today:
   prereg51's `eval_B.json` -> prereg60 would be use 8; prereg57's or prereg51-L's judgement, if made first, adds one
   each). This is p57_eval's rule plus prereg57's own judgement.
8. **Checks of the judged vectors.** Same games / store rows as c49's file, raw, tta 1, tag
   `p60_step_030000_raw_tta1`, checkpoint sha = the training record's = the file's; `count` minus c49's `count` in
   0..7 for every game (it cannot be equal any more, see 2).
9. **Hard stop.** `judge` refuses after 2026-11-22 12:00 local time.
10. **Disclosure.** While writing the code, c49's checkpoint was opened once (CPU, `torch.load`, to print its `config`
    keys and tensor shapes) at a moment when the available RAM was 4.6 GB, below the 7 GB rule of this task; no number
    of any method came from it.

11. **Seam (a) on CPU fp32 does not meet 0.05; it is repeated on the 4090 in fp16 before training.** Measured on the
    CPU (fp32) against the stored vectors (4090, fp16 autocast): same 200 games, same `count`, max abs diff 0.0696
    (largest vector entry 80.6) - above the PLAN's 0.05, which is "the fp16 tolerance of earlier seams" (comparisons
    of fp16 runs with fp16 runs). The tolerance is NOT changed. Instead: (i) the same CPU run is also compared with
    the frozen 17-plane code (`p49_run.py embed`, same games, same device) - that difference is in `seam.json`
    (`a_same_device_vs_frozen_code`) and isolates the widening and the new game set from precision; (ii) the chain's
    step 0 repeats seam (a) on the 4090 with fp16 autocast (`p60_run.py embed ... --limit 200 --no-early`, then
    `p60_eval.py seam --device gpu`), commits `seam.json` and stops before training unless max abs diff <= 0.05 with
    equal counts and seam (b) bit for bit. `judge` needs the committed `seam.json` to be ok. Only c49 is involved.

12. **Resource disclosure.** The three CPU runs that embed E games (`--source store --players E --limit N`: the smoke
    embed of 32 games, seam (a) of 200 games, and the frozen-code reference of 200 games) each used about 3.6 GB,
    above this task's 2.5 GB per process: p42_embed's store loader builds the index of all 720,307 cached games
    before `--limit` applies. The machine coordinator flagged it after the third run; no further run of that kind
    was made (the stored c49 vectors are the reference, the three files already existed). Available RAM also fell
    below 7,000 MB during some runs although it was above it at every start after note 10.

## Launch (not done by the implementing agent; the GO file does not exist)

    cd D:\AI-Portfolio\competitions\aicup2026-go
    Start-Process -WindowStyle Hidden -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "data/prereg60/chain.sh"
    # when the 4090 may be taken:   New-Item -ItemType File data\prereg60\GO      (Git Bash: touch data/prereg60/GO)
    # stop:                         New-Item -ItemType File data\prereg60\STOP
