# Test-readiness spec (Task B): the 11/04 test-day notebooks and local tools

## 0. Verdict

The existing code cannot run on the test data as it stands. The numerics are ready and can be reused without changing a single frozen file. Four quick experiments in scratch (CPU only; no answer file opened, no `score.py`) confirm this:

- **Rank apply is reproducible bit for bit.** I re-implemented `p02_rank_fuse.cmd_apply`'s numeric core by calling its own functions. I fed it `results/prereg02/rank/dumps/val_p02e9_d4.npz` with the question ids re-prefixed `val_`→`pub_`/`priv_`, both whole and cut into 400-question sets.
  - 1589/1589 rows were byte-identical to `results/prereg02/rank/rank_val_probs.csv`.
  - Rule B gave 0 differences from `pred_B.csv`.
  - Script: `…\scratchpad\testkit\specB\exp_rank_apply.py`.
- **The rank opening score does not depend on which games are in the scored set.** A build scored on the validation slots only, without the 73,556 inner games, gave `ll`, `o`, `has` and `n_terms` bit-identical to the committed `opening_val.npz` (27 s). So a test-day build over validation + public + private slots can be proven identical to the frozen build without the `build_id` check. Script: `exp_opening.py`.
- **Player, per pool, with the existing `p02_player_fuse` functions (`exp_player.py`):**
  - Replay on the original validation ids: all 24 self-retrieval means equal `sr_report.json` exactly, and `sr_max_cos_median` equals the frozen value. The result is Q04, gate +0.4991, guard level 1, and top-5 400/400 identical to `player_val_top5.csv`.
  - Fake `pub_` pool (full validation pool re-prefixed): Q04 selected (0.902147 against Q08 0.898223), gate passed, level 1 (rho raw 1.0126), top-5 400/400.
  - Fake `priv_` pool: Q04 selected (0.898212 against Q08 0.897055), level 1 (rho raw 1.0040), top-5 400/400.

**Correction to the requested dry run.** The self-retrieval split hashes `'{player_id}|{game_id}'` (`p02_player_open.py:312-314`). Re-prefixing the ids therefore changes the self-retrieval means, the denominator of rho, and possibly the selection. Exact reproduction of `player_val_top5.csv` holds for these particular fake ids, as shown above, but it is not guaranteed in general. The dry run pins these verified values, and adds a replay with the original validation ids that must match `sr_report.json` exactly.

## 1. Hard-coded assumptions that block test use

### Rank notebook (`colab/build_prereg02_rank_nb.py`)

- Output root is `prereg02` (`:265`). `P01_ROOT` is the only read-only root (`:271`, `:286-290`), so nothing protects `prereg02` on Drive.
- `FINISH_WITHOUT_TRAINING` (`:315-320`) is also referenced in the GPU gate (`:366`).
- The epoch assert (`:357-361`).
- The whole preflight needs prereg01's epochs 1–5, `state.pt`, 25.1482 % and 2,814 steps (`:435-526`).
- `DUMP_TOOLS` needs the training data:
  - `EXPECTED_INNER_GAMES` 73556 checked against `df_val` (`:544-547`)
  - `INNER_KEYS`, `KEY_LOADER`, `dump_inner` (`:685-744`)
  - the final print (`:841`)
- `parse_val` handles one question set only: it reads the global `RANK_QUESTIONS` and caches one global `VAL` (`:756-792`). `RANK_QUESTIONS` is the bundle file `local/rank_val_questions.csv` (`prereg02_rank.ipynb` cell 3, line 83).
- `dump_val` uses the stem `'val_'+name` (`:797`).
- 1589 is hard-coded in the P1 check (`:916`).
- The final cell expects the inner, P1 and P2 dumps (`:1267-1290`).
- The prepare cell unpacks the training tarball (cell 3, lines 54-74).

### Player notebook (`colab/build_prereg02_player_nb.py`)

- Output is `RESULT_DIR = prereg02/player` (`:252`), and only `P01_ROOT` is guarded (`:257-273`).
- The question and candidate files are the bundle's validation files (`:325-328`).
- The preflight requires `predict_summary` questions/candidates SHA to equal these files (`:468-471`), plus the counts 400/40000/4935 (`:116-119`, `:476-478`).
- The dump cell handles a single pool:
  - `DUMP_DIR` (`:541-545`)
  - the count assert against `P01_SUMMARY` (`:647-649`)
- The repro cell asserts 400 questions (`:724`).
- The zip is named from `RESULT_DIR` (the final cell).

### Rank tools

- `p02_rank_fuse.py`:
  - `VAL_QUESTIONS_FULL`/`VAL_GAMES_FULL` = 1589/19156 (`:134`, used at `:611`, `:658`, `:1105`)
  - `frozen_config.json` and `opening_val.npz` are read from `out_dir` (`:1081`, `:1137`)
  - the `build_id` equality check (`:1138-1139`) can never hold for a new build, because the id hashes `ll_i` and `ll_v` (`p02_rank_opening.py:212-216`)
  - `val_keys.npz`/`val_%s.npz` names (`:556`, `:1109`)
  - one question file that must match exactly (`:1118-1126`)
  - output `rank_val_probs.csv` (`:1162`)
  - defaults point to `results/prereg02/rank` and `data/local` (`:1204-1208`, `p02_rank_pseudo.py` `DEFAULTS`)
  - `check_out_dir` protects prereg01 only (`p02_rank_pseudo.py:~130`), so running with defaults writes into prereg02
- `p02_rank_opening.py`:
  - `build()` always loads inner and one question file (`:169-192`)
  - fixed output names (`:223-234`)
  - `load_opening` accepts only `part` in {inner, val} (`:261-272`)
- `decide.py` deletes, then rewrites, `pred_*` in `OUT_DIR`. It must never be given `results/prereg02/rank`.

### Player tools (`p02_player_fuse.py`)

- `cmd_select`:
  - refuses when a frozen config exists (`:721-724`)
  - requires `repro_local.json` against prereg01 (`:739-747`); no such reference exists for test pools
  - writes `frozen_config.json` into `out_dir` (`:816-869`)
- `cmd_apply`:
  - reads the frozen config from `out_dir` (`:910`, `:915`)
  - requires input SHAs equal to the frozen ones (`:934-940`), which is impossible for a new pool
  - writes `player_val_top5.csv` (`:970`)
- `REPRO_MAX_DIFFERENT` = 2 (`:102`), `REAL_INPUT_FILES` (`:438`), defaults point to prereg02 and `data/local` (`:1029-1048`).
- Strict self-retrieval needs exactly 100 games per candidate (`p02_player_open.py:94`, `:335-337`).
- `check_out_dir` protects prereg01 only (`:130-138`).

### Parts that work unchanged

- `Data` (`:176-267`) reads `<dir>/dumps/{cand_games,query_games}.npz` from any directory.
- `load_val_keys(ddir)` and `load_dump(path)` (`p02_rank_fuse.py:555`, `:405`).
- `tools/check_submission.py` already accepts `--questions PUB PRIV --candidates PUB PRIV`. It does not check the 400-per-set count or prefixes; the new tools add those.

### Why the frozen tools must not be edited

Apply re-hashes the tool files and refuses when they changed:

- rank `fc["tool"]["files_sha256"]` (`:1092-1096`)
- player `fz["tools_sha256"]` + `tools_dirty()` (`:919-927`)

Any edit to `tools/p02_*.py`, `c10data.py` or `decide.py` breaks the frozen-code guarantee.

## 2. Design

- **Frozen files stay byte-identical.** No change to `tools/p02_*.py`, `c10data.py`, `decide.py`, `colab/build_prereg02_*.py`, `colab/prereg02_*.ipynb` or `results/prereg02/**`. New files only.
- **New builders on the committed prereg02 notebooks.** Each is SHA-pinned: `prereg02_rank.ipynb` 9fb1f9d06ba5ee5fbebfb6dfe022246888c93e7762e9a89e1adc99c9f1c12b0c, `prereg02_player.ipynb` 738c1dd81aa8a93953e45019cf6563e9b2b44ffa7cf846763c51fcc8190a5c12 (working tree = HEAD, no CRLF). They follow the same discipline as today:
  - `rep()` with exactly-once targets
  - a change table
  - a per-cell diff at `results/test02/notebook_diff_{rank,player}.txt`
  - byte-identical rebuild
  - the refusal line with the forbidden token appears exactly once
- **Segment-identity asserts in the builders.** These back the "exactly the same" guarantee:
  - Rank: `d4` and its self-check, `load_net`, `sgf_to_features`, `dump_is_done`, `write_dump`, `write_keys`, the `parse_val` loop from `sgf_cols=` to `parsed=`, and the per-question loop in `dump_val` (`with torch.no_grad(): for _qid, start, k in V['spans']: …`) must be byte-equal to the base cell. Extract them with `ast.get_source_segment`.
  - Player: official cell 20 (`sgf_windows`, `embed`, `GPU_BATCH` 4096, `CHUNK` 256) stays identical. `game_embeddings`, the `CAND` loop and the `TEST` loop must be byte-equal to base.
- **One subdirectory per set, keeping the dump file names.** This lets the unchanged loaders read them:
  - Drive and local: `results/test02/rank/<set>/dumps/{val_keys,val_p02e9_d4}.npz` + `_DONE_*.json`
  - `results/test02/player/<set>/dumps/{cand_games,query_games}.npz`
  - `<set>` ∈ {`seam_val`, `public`, `private`}
  - Keeping the `val_` stems keeps `dump_val`/`write_keys` byte-identical and makes `load_val_keys` work. Document that inside test02 the `val_` prefix just means "question-set dump".
- **Seam set, run every time.** The notebooks recompute the validation dumps with the same code and weights, and the local tools run the seam set through the full pipeline. This replaces the missing `repro_local` check for test pools and catches Colab torch or cuDNN drift between 9/26 (torch 2.11.0+cu128, cudnn 91900, RTX PRO 6000) and 11/04.
- **Pinned SHAs, read from the repo at build time.**
  - Rank `epoch_9.pth`: 423581b4fddd5c2c160ebf691f4e86b9e00c4fe743d5cfb23df7891f4a2ca939. Assert that `frozen_config.json` `dumps.p02e9_d4.weights_sha256`, `_DONE_val_p02e9_d4.json` and the npz field all agree.
  - Player `best_model.pth`: 356eb4f65488925e2a552ea382979780c903ef515b85d329a3805db837b7773b (`build_prereg02_player_nb.py:99` = frozen `inputs.cand_games_npz.weights_sha256`).
  - Seam references on Drive (read-only): `prereg02/rank/dumps/val_keys.npz` 055d2941…, `val_p02e9_d4.npz` ec0b729d…, `prereg02/player/dumps/cand_games.npz` 56ed03d1…, `query_games.npz` 665d0888….
  - Assert at build time that `set(frozen["dumps"]) == {"p02e9_d4"}`.

## 3. Notebooks

### `colab/build_test02_rank_nb.py` → `colab/test02_rank.ipynb`

Base is `prereg02_rank.ipynb`. Cells in order:

1. **md header (new).** Purpose (inference only, frozen R20 dumps), steps, GPU choice, stop conditions, outputs, hand-over.
2. **setup (rep).**
   - `RESULT_ROOT = results/test02` (`test02_smoke` in SMOKE); `RESULT_DIR = …/rank`; no `RESULT_CKPT`.
   - Read-only roots: `P01_ROOT` and `P02_ROOT = results/prereg02` (`_smoke` in SMOKE). `guard_write` refuses both.
   - A fingerprint function for each root.
   - Remove `FINISH_WITHOUT_TRAINING` and `SMOKE_CFG_P02`. Replace `P02_CHANGES` with `T02_CHANGES`.
   - `TESTS_DIR = DRIVE_DIR/tests`; `TEST_FILES = {'public': 'rank_prediction_test_public.csv', 'private': 'rank_prediction_test_private.csv'}`.
3. **preflight (new).**
   - Weights: read `P02_ROOT/rank/ckpt/epoch_9.pth` (SMOKE: `epoch_2.pth`) and check the pinned SHA. In SMOKE, instead require equality with the `weights_sha256` in the prereg02_smoke markers.
   - Seam references: SHA-pinned.
   - Test files present under their exact official names; otherwise stop and list what is in `tests/`.
   - Quick format check (part of T1):
     - 400 rows each
     - unique `question_id` with prefix `pub_`/`priv_`
     - `num_games` 5–20
     - columns `sgf_1..20` and `color_1..20`
     - exactly the first `num_games` SGFs non-empty
     - colours B/W
     - no BOM
   - Record each test file's SHA-256 in `config.json` and in `<set>/dumps/_INPUT.json`.
   - **Stale-dump guard:** stop if an existing `_INPUT.json` SHA differs from the current file. Without it, `dump_is_done` could skip a replaced file that has the same row count.
   - Drive free space ≥ 0.5 GB; readonly-before fingerprints.
4. **prepare (rep).** Remove the training-tar block (cell 3, lines 54-74), as the player builder does. Keep the bundle SHA checks and the refusal line verbatim.
5. **official 1, environment:** unchanged.
6. **official 2 (rep).**
   - Remove the `EXTRA_EPOCHS`/`FINAL_EPOCH` assert (`:357-361`) and `guard_write(CKPT_DIR)`/`makedirs` (`:378-379`).
   - GPU gate without the `FINISH` clause: G4 (RTX PRO 6000) or A100, as prereg02.
7. **dump tools (base `dump_tools` cell + rep).**
   - Delete the `EXPECTED_INNER_GAMES` check, `INNER_*`, `decode_with_key`, `KEY_LOADER`, `dump_inner` and the final print.
   - Replace `record_p01_after` with a two-root version.
   - `parse_val()` → `parse_questions(path)`: a cache dict keyed by path; `pd.read_csv(path, dtype={'question_id': str})`; SMOKE `head()` kept.
   - `dump_val(net, …)` → `dump_questions(net, path, name, …)` calling `parse_questions(path)`.
   - `DUMP_DIR` is re-bound per set before the calls. The base functions read it as a global, so they stay byte-identical.
8. **Three set cells: `seam_val` (bundle validation file), `public`, `private`.** Each does `DUMP_DIR = RESULT_DIR/<set>/dumps`, `write_keys('val_keys', …)`, `dump_questions(net, path, 'p02e9_d4', 8, 'prereg02/rank/ckpt/epoch_9.pth', WSHA)`. The net is loaded once with `load_net(path, expect_sha=PIN)`.
9. **seam check (new).** Compare seam dumps with the Drive prereg02 validation dump.
   - Keys arrays must be equal. The parsed-mask must be identical.
   - Record: bitwise equality, max |Δlogit|, and the torch/cuDNN/GPU of both runs.
   - Pass rule (the registered 1.1 ② ③ criteria, applied against the prereg02 dump):
     - C0 per-question float32 mean softmax: max diff ≤ 1e-4 when the GPU model equals the marker's "NVIDIA RTX PRO 6000 Blackwell Server Edition", otherwise ≤ 1e-3
     - rule B (C0) differences ≤ 4 of 1589 (SMOKE scaled)
   - Write `seam_check.json`. On failure: pack `rank_results_seam_failed.zip`, flush Drive, stop.
10. **final (new).**
    - Per set: marker, SHA, rows, NaN pattern.
    - Readonly-after fingerprints for both roots.
    - `config`/`run_log`/`timings`.
    - `pack_results('rank_results.zip')` (unchanged; lands in `test02/`); `flush_drive()`.

**Estimate:** about 5–8 min on G4 (not measured).

### `colab/build_test02_player_nb.py` → `colab/test02_player.ipynb`

Base is `prereg02_player.ipynb`.

- **setup (rep).** `RESULT_DIR = test02/player`. Add `P02_ROOT`; `guard_write` refuses `P01_ROOT` and `P02_ROOT`. Add `TESTS_DIR`.
- **prepare, official 1, environment:** identical.
- **official 2 (rep).** Keep `QUESTIONS_CSV`/`CANDIDATES_CSV` as the bundle validation files; they are the seam pool. Add `TEST_POOLS = {'public': (tests/player_identification_test_public.csv, tests/player_identification_candidates_public.csv), 'private': (…)}`.
- **config record:** paths plus the test entries.
- **preflight (base + rep).**
  - Keep all prereg01 checks: the seam uses them.
  - Add: the prereg02 dump SHAs; the test files; the quick format check:
    - 400 questions
    - candidates 40,000 rows = 400 ids × exactly 100 games (strict self-retrieval needs this)
    - prefixes `pub_`/`priv_` for `game_id` and `question_id`, `pub_p_`/`priv_p_` for `player_id`
    - pools disjoint in both player and game ids
    - colours B/W
  - Record the test file SHAs. GPU must be G4, as prereg02.
- **official 20:** identical.
- **dump cell (base + rep), emitted three times.** Each copy follows a one-line preamble `POOL=…; QUESTIONS_CSV, CANDIDATES_CSV = …`. The builder asserts the three copies are identical text.
  - `DUMP_DIR = os.path.join(RESULT_DIR, POOL, 'dumps')`.
  - The count assert (`:647-649`) now checks `EXPECTED_COUNTS[POOL]` from the preflight.
  - Store per-pool `DUMP_INFO_BY_POOL[POOL]`.
  - Order: `public`, `private`, `seam_val`. The deletion loop (`:549-553`) removes `REPRO_*`, so the repro cell must run last.
- **repro cell:** base verbatim; it runs on the seam pool and is the registered 2.1 criterion (≥ 398/400, centroid ≤ 1e-4). Add a small cell that records bitwise equality with the prereg02 dumps.
- **final (base + rep).** Loop over pools, check both roots' fingerprints, write `test02/player_results.zip`, flush Drive, stop on problems.

**Estimate:** about 15–20 min on G4 (not measured).

## 4. Local tools (new; outputs only under `results/test02/`)

Common rules for both tools:

- `check_out_dir` gets `extra_readonly = (results/prereg01, results/prereg02)`.
- **Real mode refuses unless all of these hold:**
  - the frozen configs are committed and equal HEAD (the existing `p02_rank_fuse.require_committed`, `p02_player_fuse.check_committed`)
  - the frozen tool SHAs are unchanged (`p02_rank_fuse.tool_state()` against `fc["tool"]["files_sha256"]`; `po.source_shas()` against `fz["tools_sha256"]` + `tools_dirty()`)
  - the t02 files are committed and clean
  - `results/test02/<comp>/**` notebook outputs are committed and unchanged
  - the notebook seam passed
  - the local `data/test/<official names>` SHAs equal the notebook's `config.json` record
  - the notebook `config.json` says `"smoke": false`
- **`--dry-run`:** output must be outside the repo; `--root` points at the fake tree; only the git checks on test02 outputs are skipped. **`--smoke`:** for the notebooks' SMOKE dumps.
- Refuse to overwrite existing outputs unless `--rerun --reason` is given, and record the reason.

### `tools/t02_rank.py`

Imports `p02_rank_fuse`, `p02_rank_opening`, `decide`, `c10data`; never a player module.

**`opening`**
- Model: `OpeningModel(scored = seam_val + public + private slots)`, counted on `load_rows(part="fit")`, scored with `read_questions`, `game_string`, `log_softmax`.
- Check: the seam part's `ll`, `o`, `has` and `n_terms` must be bitwise equal to the committed `results/prereg02/rank/opening_val.npz` (SHA 75ffec64…). This replaces `p02_rank_fuse.py:1138-1139`.
- Write `opening_test.npz` (`set`, `question_id`, `slot`, `color`, `ll`, `o`, `has`, `n_terms`, `n_moves`, meta with `frozen_opening_build_id` c0cb86b4… and input SHAs) and `opening_test_report.json`.

**`apply`**
- Read `results/prereg02/rank/frozen_config.json` (44a16e09…). Check that `basis` equals `basis()`, `b = Bβ`, and each set's dump `weights_sha256` equals `fc["dumps"]["p02e9_d4"]`.
- For each set, replicate `cmd_apply` lines 1114, 1116, 1127-1135, 1149-1160 and 1164-1169 exactly, by calling `load_val_keys`, `load_dump`, `check_val_logits`, `source_logprobs`, `pool_matrix`, `pooled`, `question_logits`, `softmax`, `read_question_meta`.
- Outputs:
  - `seam_val/rank_val_probs.csv` + `seam_local.json`. The seam must match the committed `rank_val_probs.csv` (a8370043…) and `pred_B.csv` (863a6258…). Pass: rule-B differences ≤ 4; byte identity expected when the dumps are bitwise.
  - `rank_test_probs.csv`: public rows, then private (800 rows, `PROB_COLUMNS`).
  - `decide.run(…, results/test02/rank)` produces `pred_A.csv`, `pred_B.csv`, `decide_summary.json`.
  - `submission_rank.csv`: a byte copy of `pred_B.csv`.
  - `check_submission.check('rank', …, [pub, priv])` must return an empty list.
  - `apply_test_report.json`: gate copied, counts, questions without a parsed game.

### `tools/t02_player.py`

Imports `p02_player_fuse`, `p02_player_open`; no rank module and no `decide`.

**`run --pool {seam_val,public,private}`**
- **Select:** `Data(cand, pool_dir, True, False)` → `self_retrieval(strict=True)` → `_stats`/`choose` → gate = selected − Q00 ≥ 0.02 → `sr_med`. Same code as `cmd_select:762-797`, but without the `repro_local`/frozen-exists checks.
- **Guard:** from `results/prereg02/player/frozen_config.json` (d70726ab…). Check `po.sha256_text_file(guard_calibration.json)` = 793c16a7….
- **Apply:** `Data(…, q_path)` → as `cmd_apply:942-969` with the pool's own selection and denominator.
- **Outputs:** `<pool>/pool_select.json` (`sr_report` fields + selected/gate/`sr_max_cos_median`/guard), `<pool>/top5.csv`, `<pool>/apply_report.json`.
- **Seam pass rule:** `seam_val` must reproduce `sr_report` means exactly and give top-5 ≥ 398/400 against `player_val_top5.csv` (69ca74a6…); 400/400 is expected.

**`submit`**
- Public rows then private into `submission_player.csv`; `check_submission` player with both pools; `submit_report.json`.
- Optional `--no-cnn --reason`: the registered Q22/Q23 fallback per pool.

### `tools/t02_fake_tests.py`

Uses only csv/numpy/hashlib and imports no p02 model tool. Output must be outside the repo. It builds this tree:

- `<root>/tests/<6 official names>`:
  - rank public = validation questions 1–400 re-prefixed `pub_`; private = 401–800 re-prefixed `priv_` (disjoint, so the real 400-count checks stay live)
  - player questions: all 400, `pub_` / `priv_`
  - candidates: `game_id` and `player_id` `val_`→`pub_`/`priv_`
  - all other fields byte-identical
- `<root>/results/test02/{rank,player}/{seam_val,public,private}/dumps`:
  - `seam_val` = byte copies of the prereg02 dumps
  - test sets = row slices with re-keyed ids; logits, `emb_sum`, `n_windows` and `weights_sha256` byte-identical; markers regenerated
- Fake `config.json` (`smoke:false`, `dry_run:true`, test SHAs), `seam_check.json`/`repro_check.json` passed, `run_log` completed.

## 5. End-to-end dry run (no answers; `tests/test_t02_dryrun.py`, opt-in `C10_SLOW=1`, under 3 min)

1. `t02_fake_tests.py --root S`
2. `t02_rank.py opening --dry-run --root S`
   - Expect the seam part bitwise equal to `opening_val.npz` (verified).
3. `t02_rank.py apply --dry-run --root S`
   - Expect `seam_val/rank_val_probs.csv` byte-identical to prereg02.
   - Rows `pub_q_0001..0400` and `priv_q_0401..0800` equal the validation rows after prefix mapping (verified).
   - `submission_rank.csv` predictions equal `pred_B.csv` for those 800 questions; `check_submission` OK.
4. `t02_player.py run --dry-run --root S --pool seam_val`
   - Expect means equal `sr_report.json` exactly, `sr_med` equal to frozen, Q04, gate 0.4991241719699286, level 1.
   - `top5.csv` byte-identical to `player_val_top5.csv` (verified).
5. The same for `public` and `private`.
   - Expect Q04 and level 1 as in §0 (pin these numbers).
   - Top-5 400/400 after prefix mapping (verified).
6. `t02_player.py submit --dry-run`: 800 rows, `check_submission` OK.

**Negative cases** (fast unit tests):
- tampered `weights_sha256`
- test CSV SHA different from the notebook record
- out dir inside the repo or inside `results/prereg02`
- seam not passed
- frozen tool SHA mismatch (monkeypatched)
- a pool with 99 games for one candidate

## 6. CPU smoke for the new notebooks

`colab/run_local_smoke.py` needs no change. Steps:

1. Rebuild the prereg01 → prereg02 smoke chain in one scratch directory. It took about 15 min for rank and about 3 min for player (`build_review.md:23-26`, `:93-95`). Alternatively copy `…\scratchpad\p02final\smoke\{rank,player}\smoke\drive\results\prereg0{1,2}_smoke` if their `config.json` shows the committed notebooks.
2. `t02_fake_tests.py --tests-only --out <S>/drive/tests` (full-size files; the notebooks apply the SMOKE subsets).
3. `run_local_smoke.py colab/test02_{rank,player}.ipynb --scratch <S>`.
   - Expect the seam against the prereg02_smoke dumps to be bitwise equal on CPU.
   - Outputs in `test02_smoke`.
4. `t02_*.py --smoke` on those outputs; `check_submission` on subset question files, as the prereg02 smoke run did.

**Negative smoke cases:** missing test file; tampered weights copy; a write into `prereg02_smoke` is refused.

## 7. Also add

- `tests/test_t02_rank.py`, `tests/test_t02_player.py`: synthetic unit tests.
- `tests/test_t02_hygiene.py`:
  - no forbidden token except the inherited refusal line
  - team-import separation
  - **a pin test**: the current SHAs of `tools/p02_rank_*.py`, `p02_moves.py`, `c10data.py`, `decide.py` equal `results/prereg02/rank/frozen_config.json` `tool.files_sha256`, and the player tools equal `frozen_config.json` `tools_sha256`. This catches an accidental edit before 11/04.
- `.gitignore`: add `results/test02/*.zip`.
- `docs/COLAB_RUN.md`: a test02 section.
- PREREG_02 rev 5, before 11/04: the seam checks (operational), the output layout, and D1 below.

## 8. Decisions needed

- **D1 — per-pool gate at test time.** The registered gate governs local scoring only. Recommend registering: "if a pool's gate fails, that pool uses Q00".
- **D2 — player notebook GPU.** G4 only (as prereg02), or allow A100 when the seam passes.
- **D3 — seam failure.** Recommend: stop and submit nothing; likely cause is Colab torch drift.
- **D4 — `val_` stems inside test02.** Keep them for loader reuse, or rename and write new loaders.
- **D5 — official file names or layout differ on 11/04 (T1).** Change only the single name table in each builder and rebuild.
- **Clean-up:** the scratch copies of competition data can be deleted after review: `…\scratchpad\testkit\specB\fake_player\` (~110 MB).