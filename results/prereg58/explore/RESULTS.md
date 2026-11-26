# prereg58 (competition 2): opponent-identity term added to Q25 - judged once, NOT adopted

2026-11-20. Plan `PLAN.md` (frozen), narrow readings `NOTES.md` (committed before the kernel and before any inner
number). Every number below comes from `kernel.json`, `seam.json`, `eval.json` in this folder.

## What was run

- Identity model S0 (prereg43, 3,685 classes = T36). Opponent-side game vectors (colour flipped, raw weights, tta 1)
  through `p58_rows.py` -> `p43_run.py embed`. Check on 64 D games (CPU fp32 against the existing GPU fp16 files):
  cosine >= 0.999998 on the player side and on the opponent side (`vectors.json`).
- Inner: 73,169 of the 73,556 inner games occur in an inner question; all were embedded, none without a sample
  (`rows.json`, `vectors.json`).
- Kernel (`kernel.json`, commit de7c916): 900 players (D 300 + E 600, 90 per rank; no inner player, no S0 class),
  287,751 games (HD 0 / 1 / 2+: 236,938 / 42,395 / 8,418). tau = 50 (held-out mean log-likelihood -2.267 / -2.155 /
  -2.111 / -2.128 / -2.151 for tau 20 / 30 / 50 / 80 / 120; the normalised record agrees). K peaks at d = 0 for HD 0
  and HD 1; for HD 2+ the opponent is stronger when the player is labelled B and weaker when labelled W.
  Standalone record on the kernel players (kernel cross-fitted, term + rank prior only): 0.541 / 0.595 / 0.629 for
  5 / 10 / 20 games (prior only 0.150).
- Seam (`seam.json`, commit 264f8df): Q25 inner 0.7456519871546352 and single-colour 0.7482450052195216, both bit
  for bit, through `p58_eval.build_q25` (Inner without the Q02 / Q02H / Q08 fits; peak working set 2.25 GB).

## Judgement (inner, once; `eval.json`, commit e86c7a6)

Q58 = Q25's seven terms + OPP (mean over the question's games of l_g(r)), added as prereg44 added its 8th term.

| | Q25 | Q58 | diff | 95% interval | gate |
|---|---|---|---|---|---|
| inner eval (127,200 questions) | 0.7456520 | 0.7462456 | **+0.00059** | [-0.00539, +0.00570] | G1 (>= +0.005) no; G2 (lower > 0) no |
| single colour (120,544 questions) | 0.7482450 | 0.7511603 | +0.00292 | [-0.00016, +0.00576] | G4 no (lower bound not > 0; B +0.0041, W +0.0018) |

**adopt = false.** Competition 2 stays at 25. No test-set prediction was built.

Record only:

- Fitted weight of OPP: **4.0 in all five folds = the fusion's upper bound** (weights are bounded to [0, 4]); the
  other weights shrink (SET 0.36-0.49, P1TO 0-0.09, O4 ~0, a -0.46 .. -0.68). The registered way of adding the term
  (prereg44's: no scale) therefore capped it; this run says nothing about a rescaled term.
- The term alone (rule B): 0.5667 (pooled mean) / 0.6005 (sum over games); single colour 0.5637 / 0.5923.
- By rank class (eval, Q58 - Q25): D **-0.031**, C +0.006, B +0.014, A +0.016, 1D -0.001, 2D +0.009, 3D +0.004,
  4D +0.001, 5D +0.014, 6D **-0.025**. The two end classes lose, the middle gains; single colour shows the same
  pattern (D -0.023, 6D -0.017, A +0.021).
- By number of games, single colour: 5 games +0.0039 [+0.0005, +0.0072], 10 -0.0003 [-0.0046, +0.0039],
  15 +0.0041 [+0.0005, +0.0079], 20 +0.0040 [+0.0001, +0.0079]. Eval questions with exactly 5 / 10 / 15 / 20 games:
  -0.0055 / +0.0007 / -0.0032 / -0.0049 (intervals all contain 0).
- Inner question games: HD 0 / 1 / 2+ = 58,548 / 12,153 / 2,468; mean of max_s q_g[s] = 0.62.

## Reading

The opponent-identity term carries real rank information (alone 0.57-0.60 on inner, 0.54-0.63 on the kernel
players), yet added the registered way it moves the inner score by only +0.0006 with a wide interval: gains in the
middle classes are cancelled by losses in D and 6D, and its weight sits at the bound. Not checked here (hypotheses
only): that a rank-difference kernel shared by all ranks misfits the two end classes, where the opponent can only be
on one side; and that the [0, 4] weight bound without a scale holds the term back. Not adopted; the inner set has
now been used once for this term, so any variant needs its own registration.

## Files

`p58_lib.py` (q, kernel, l_g), `p58_rows.py` (embed wrapper, class head), `p58_eval.py` (rows / vectors / kernel /
seam / eval), `embed_inner.sh`, `test_p58.py`. Data (not in git) under `data/prereg58/`: `s0_head.npz`,
`inner_rows.npy`, `emb_opp_inner/`, `emb_opp_D/`, `emb_opp_E/`, `emb_opp_rank/` (test questions, launch chain; not
read by this code), `eval_started.json`, `eval_log.txt`, `smoke/` (noise run of the pipeline).
