# prereg61 implementation notes (readings of PLAN.md, written with the code and before any number)

PLAN.md is not edited. prereg60's files are not edited: `p61_run.py` / `p61_eval.py` import `p60_run` / `p60_eval`
and replace module constants before calling them. Where PLAN.md leaves a detail open, the narrowest reading is here.

1. **What the wrapper changes in p60_run (checked line by line).** `cmd_train` / `cmd_embed` read `DATA`, `OUT`,
   `EMB`, `SEAM`, `SEAM_GPU`, `STOP_FILE`, `TRAIN_RECORD`, `TOTAL_STEPS`, `LR`, `WARMUP`, `KEEP`, `RESUME_EVERY`,
   `CODE_FILES` as module globals at call time, so replacing them is enough: the checkpoint names
   (`step_%06d.pt` -> `step_200000.pt`), the optimiser's lr, the LambdaLR (`p60_lib.lr_factor(s, 2000, 200000)`), the
   config stored in every checkpoint and in the record (`total_steps`, `lr`, `warmup`), the record's `finished`
   (`step == TOTAL_STEPS`) and `model` line, the smoke folder (`data/prereg61/smoke`) all follow. `p60_lib` holds no
   30,000 (only in a docstring). The one default argument bound at definition, `emb_tag(step=TOTAL_STEPS)`, is used
   nowhere in p60_run; the wrapper replaces it anyway. Labels that stay prereg60's because the code is prereg60's:
   the config key `"p60"`, the record's `"format": "c10-prereg60 train v1"`, and the vectors' tag
   **`p60_step_200000_raw_tta1`** (run name from `load_net60` + checkpoint stem) - under `data/prereg61/emb`.
   `code_sha256` of checkpoints and record gains `p61_run.py`.
2. **Which batches.** As prereg60 note 1: batch s of this run (s = 0 .. 199,999) is batch 320,000 + s of c49's sampler
   stream (seed 0, T49). The first 30,000 batches are therefore the ones prereg60 saw; nothing of prereg60's weights
   is used (start = c49 raw, widened).
3. **Schedule, exactly.** Peak lr 1e-3. Factor at optimiser step s (0-based): (s + 1) / 2,000 for s < 2,000
   (lr 5e-7 at s = 0), 1 at s = 2,000, then 0.5 (1 + cos(pi (s - 2,000) / 198,000)), 0 at s = 200,000.
   Kept checkpoints 50,000 / 100,000 / 150,000 / 200,000; a resume checkpoint every 1,000 steps (and on a STOP /
   low-RAM pause), the previous one deleted; resume by the highest step number. A new EMA is kept in the
   checkpoints and not used. The model is the raw `net` of `step_200000.pt`.
4. **Seam.** PLAN: "v2-7's E score bit for bit through the judging path" - `P51.base_seam` inside `judge` raises
   unless every pool is identical; recorded as `seam_base_E_identical`. No new widened-c49 seam is run (the code
   that widens and builds planes is prereg60's, unchanged): `judge` instead requires prereg60's committed
   `seam.json` to be ok and made with the present `p60_lib.py`, and records its sha. The chain has no step 0.
5. **E use number.** 1 + `p60_eval.e_uses_so_far()` (which leaves out prereg60's own file) + prereg60's `eval*.json`
   carrying `e_pools_use_number`; prereg61's own files never count. Today: 6 through prereg49 + prereg51 `eval_B.json`
   + prereg60 `eval.json` = 8 -> this judgement is use 9 (10 if another E judgement is committed first).
6. **Checks at the judgement.** PLAN.md, p61_run.py, p61_eval.py, train_record.json and prereg60's p60_lib.py /
   p60_run.py / p60_eval.py committed and unchanged; the record finished at 200,000 steps, not a smoke, with
   total_steps 200,000 / lr 1e-3 / warmup 2,000 / first_batch 320,000 and the code hashes of the present p61_run.py,
   p60_lib.py, p60_run.py; `step_200000.pt`'s sha256 = the record's = the vectors'; vectors raw, tta 1, the tag of
   note 1, same games / store rows as c49's file, count - c49's count in 0..7, written after the code's commit.
   Only the 200,000-step checkpoint can be judged by this code (PLAN: an earlier one needs a committed revision
   first). `judge` refuses after 2026-11-23 12:00 local time and when eval.json exists.
7. **apply / assemble** are p60_eval's functions pointed at prereg61 (their report says `"format": "c10-prereg60
   apply v1"`); output `results/prereg61/apply`.
8. **Chain.** `data/prereg61/chain.sh` = prereg60's without step 0. The done file
   `C:\AI-Portfolio-cache\war_room\4090_c10_p61_done.txt` (end time, rc, "4090 free") is written at the normal end
   and also whenever the chain exits after it had taken the GPU (STOP, error) - with a line saying it did not finish.
9. **Smoke (CPU, before the launch).** `C10_P61_SMOKE=1 GPUPY -m pytest test_p61.py -k Smoke`: 2 steps from c49
   (4 games x 6 moves), then `--auto-resume` to step 3, under `data/prereg61/smoke` (deleted by the test). The log's
   lr was 1.5e-06 after step 2 and 2e-06 after step 3 (= 1e-3 x 3 / 2000, 1e-3 x 4 / 2000); the smoke record carried
   total_steps 200,000, lr 1e-3, warmup 2,000, first_batch 320,000, 18 planes. Only the training loss was printed;
   no score.

## Launch / stop / resume

    cd D:\AI-Portfolio\competitions\aicup2026-go
    # allow it to take the 4090:   New-Item -ItemType File data\prereg61\GO      (Git Bash: touch data/prereg61/GO)
    Start-Process -WindowStyle Hidden -WorkingDirectory 'D:\AI-Portfolio\competitions\aicup2026-go' -FilePath 'C:\Program Files\Git\bin\bash.exe' -ArgumentList 'data/prereg61/chain.sh'
    # stop:     New-Item -ItemType File data\prereg61\STOP    (saves within 100 steps, the chain exits 4)
    # resume:   delete data\prereg61\STOP and start the chain again with the same Start-Process line
    # logs:     data\prereg61\chain_log.txt, train_log.txt, S0\train_log.jsonl, embed_log.txt, judge_log.txt
