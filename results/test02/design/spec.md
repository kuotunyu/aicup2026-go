# C10 test-day kit (`test02`): implementation spec

- Written 2026-09-26 against HEAD `eb9bc8d`. The repo was not changed.
- It combines Task A (formats), Task B (code readiness) and Task C (T1–T6 checks), and settles where they disagree.
- The goal: on and after 11/04, the frozen prereg02 method (甲, meaning the locally validated model used as is) produces `submission_rank.csv` for TEAM_11248 and `submission_player.csv` for TEAM_11249. The registered T1–T6 checks also run. No prereg02 file changes by a single byte.

## Global invariants (every implementer, every tool)

- **Never** open, read or hash `data/local/*answers*.csv`. Never run `tools/score.py`. No GPU.
- Python is `D:\AI-Portfolio\competitions\aicup2026-go\.venv\Scripts\python.exe` (3.12).
- Scratch lives only under `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\testkit\<label>\`.
- Training data is read only through `tools/c10data.load_rows(part=...)`. Games of validation players are never used for any statistic, including in dry runs (see 0.11).
- These files stay byte-identical: `tools/p02_*.py`, `tools/c10data.py`, `tools/decide.py`, `colab/build_prereg02_*.py`, `colab/prereg02_*.ipynb`, `results/prereg01/**`, `results/prereg02/**`. All existing tests stay green, including `tests/test_p02_hygiene.py` and `tests/test_tools.py::StrictCheck`.
- New repo outputs go only to `results/test02/**` (committed) and `results/official_scores.json`. Test data, caches and zips go to `data/test/**`, `data/cache/**` and `data/colab_bundle/C10_test_v1.zip`. All of these are gitignored through `data/`.
- Implementers do not commit. The integrator commits as `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`, with no Co-Authored-By line.

## 0. Decisions that settle the three investigations

| # | Topic | Decision | Reason |
|---|---|---|---|
| 0.1 | Frozen code | New files only, plus two edits to non-frozen files: `tools/check_submission.py` and `.gitignore`. | `apply` re-hashes the frozen tools. `check_submission.py` is in neither frozen list. Rank `tool.files_sha256` covers p02_rank_fuse/pseudo/opening, p02_moves, c10data and decide. Player `tools_sha256` covers p02_player_open/fuse, p02_moves and c10data. |
| 0.2 | Where outputs go | `results/test02/**`. The T3 null goes to `results/test02/null/`, not to Task C's `results/prereg02/testday/`. | The prereg02 tree stays untouched. The t02 out-dir guard treats `results/prereg01` and `results/prereg02` as read-only. |
| 0.3 | How test files reach Colab | One zip, `MyDrive/C10/C10_test_v1.zip` (members `tests/<6 official names>` and `tests/MANIFEST.json`), built locally after T1 passes. This replaces Task B's "6 files in `MyDrive/C10/tests/`". | The user uploads one file, and Colab provably uses the same bytes T1 checked. |
| 0.4 | `check_submission.py` | Edit it as Task A proposes:<br>• add `--official`<br>• a missing question file becomes an error<br>• bad argv prints usage and exits 1<br>The order check runs only under `--official`; default behaviour is otherwise unchanged. | Task B's plan to put the 400/800 checks in new tools would duplicate code. The default mode keeps checking historical validation files exactly as before. |
| 0.5 | Team separation | No shared t02 helper module; each of `t02_rank.py` and `t02_player.py` carries its own small guards. `t02_checks.py` runs per `--comp` and never reads the other competition's files in the same run. Rank test files are never compared with player test files. | PREREG_02 §3: the teams share only `c10data.py` and `p02_moves.py`. |
| 0.6 | Seam | Each Colab run recomputes, with the same code and weights: the validation-question dumps (rank) and the validation candidate pool (player). They are compared with prereg02. The local tools then push the seam through the whole local pipeline. If the seam fails, nothing from that run is uploaded. | This is the only way to catch Colab torch/cuDNN drift and t02 orchestration bugs on real output. |
| 0.7 | `val_` file stems | Kept inside `results/test02/**/dumps/`, where they mean "question-set dump". | The base notebook functions stay byte-identical, and the frozen loaders (`load_val_keys`, `load_dump`, `Data`) work unchanged. |
| 0.8 | Gate per pool at test time (D1) | If a pool's gate fails (selected minus Q00 < 0.02), that pool uses Q00. | Conservative, and a submission is mandatory at test time. |
| 0.9 | GPU (D2) | Rank: G4 or A100. Player: G4 only, the same as prereg02 rev 4 item 1. If no G4, wait. After one day, Claude decides (rev 3 item 8). | The player reproduction tolerance is never relaxed. |
| 0.10 | Gap in T1 | Rev 5 adds consequences for a T1.4/T1.5 failure:<br>• rank uses the registered R10 fallback (parameters fitted now from inner data, committed before 11/04)<br>• player keeps the per-pool procedure<br>• T3 becomes "not evaluable" | The registered "only change the reader/writer code" cannot repair opening features. |
| 0.11 | Validation games | The content statistics (T1.4–T5) never run on validation players, including in dry runs; checker dry runs use pools built from training players. The fake tree built from validation files is used only for:<br>• the apply pipeline (question and candidate files are the registered test-time exception)<br>• T1.1–T1.3 structure | Hard rule |
| 0.12 | Rehearsal (recommended) | One Colab rehearsal between 10/28 and 11/02, using a rehearsal zip built from re-prefixed validation files. The notebook detects `kind: rehearsal` from the manifest and writes to `results/test02_rehearsal`. | Catches Drive-path, GPU-gate and version-drift problems before test day. Costs the user about 1 h and about 5 compute units (not measured). |
| 0.13 | Result zip names | `test02_rank_results.zip` and `test02_player_results.zip`, always the same names. `run_log.json` `status` ∈ {`completed`, `seam_failed`, `repro_failed`, `readonly_changed`}. | No collision with the prereg02 zips in Downloads, and one name for the user to handle. |

## 1. 使用者 11/04 起要做的事

**11/04 以前**：不用做事。選做：10/28–11/02 做一次 Colab 彩排，約 1 小時、約 5 個運算單元（推估）。要做的話，步驟 Claude 屆時給。

**11/04 11:00 以後**

1. 登入 AIdea，競賽二（TEAM_11248）和競賽一（TEAM_11249）各在「資料」頁下載測試集，把下載檔的位置告訴 Claude。
2. Claude 檢查格式後，會產生 `C10_test_v1.zip`。
   - 把它上傳到 Google 雲端硬碟 `我的雲端硬碟/C10/`。
   - 先把兩份 prereg02 notebook 移到 `C10/old_prereg02/`，再放上 `test02_rank.ipynb`、`test02_player.ipynb`。
3. 照 `docs/COLAB_RUN.md` 的「測試集（test02）」一節，在 Colab 跑這兩份：
   - 兩份都選 G4。競賽二拿不到 G4 時可以用 A100；競賽一只用 G4，拿不到就晚點再試。
   - 兩份可以同時跑，各約 10–25 分鐘（推估）。
   - 跑完前不關分頁，電腦不睡眠。
4. 各下載 `test02_rank_results.zip`、`test02_player_results.zip`，把位置告訴 Claude。畫面出現「seam 沒過」或紅字時，截圖給 Claude，不要自己重跑。
5. Claude 算好、檢查好兩個上傳檔後，會分兩則訊息各給你一份，每則寫明：
   - 檔名
   - 要傳到哪一場
   - SHA-256 前 12 碼
   **你在對話裡說 OK 之後**，才到那一場的上傳頁上傳。
6. 上傳後把 public 分數（數字或截圖）告訴 Claude。
7. 之後只有 Claude 提出時才再上傳：修 bug、換成新登記的方法，或最後一次上傳。
   - **以最後一次上傳計分**。
   - Claude 在 11/23 前會告訴你最後要傳哪一份；11/24 23:59 截止。
8. 比賽結束後填後測問卷。

Claude-side rule (from memory `irreversible-step-checkpoints.md`): the upload instruction always gets its own message. All pre-checks go in earlier messages.

## 2. Colab notebooks

### 2.0 Common to both builders

- **Builders:**
  - `colab/build_test02_rank_nb.py` → `colab/test02_rank.ipynb`
  - `colab/build_test02_player_nb.py` → `colab/test02_player.ipynb`
- **Base notebooks, with SHA asserted at build start:**
  - `colab/prereg02_rank.ipynb` `9fb1f9d06ba5ee5fbebfb6dfe022246888c93e7762e9a89e1adc99c9f1c12b0c`
  - `colab/prereg02_player.ipynb` `738c1dd81aa8a93953e45019cf6563e9b2b44ffa7cf846763c51fcc8190a5c12`
- **Same discipline as the prereg02 builders:**
  - `rep(src, old, new)` requires exactly one match.
  - A `T02_CHANGES` table is shown in the header markdown cell and stored in `config.json` under `changes_from_prereg02`.
  - A per-cell diff goes to `results/test02/notebook_diff_{rank,player}.txt`.
  - Building twice gives byte-identical output.
  - The inherited prepare-cell refusal line (`assert not any(... '<needle>' in ...)`) appears exactly once per notebook. No other line in the notebook contains the needle.
- **Segment identity.** The builder extracts each of the following from the base cell and from the new cell with `ast.get_source_segment` and asserts they are byte-equal.
  - Rank:
    - `d4` and its self-check
    - `load_net`, `sgf_to_features`, `dump_is_done`, `write_dump`, `write_keys`
    - the `parse_val` loop body from `sgf_cols=` to `parsed=`
    - the per-question loop of `dump_val` (`with torch.no_grad(): for _qid, start, k in V['spans']: …`)
  - Player:
    - official cell 20 in full (`sgf_windows`, `embed`, `GPU_BATCH` 4096, `CHUNK` 256)
    - `game_embeddings`, the `CAND` loop and the `TEST` loop
- **Drive layout.**

  | Path | Role |
  |---|---|
  | `MyDrive/C10/C10_bundle_v1.zip` | Unchanged. The inherited SHA check stays. It provides the seam's validation files. |
  | `MyDrive/C10/C10_test_v1.zip` | New. The notebook unzips it to `WORK/tests/` and verifies every member against `tests/MANIFEST.json`. |
  | `MyDrive/C10/results/prereg01/**` | Read-only. `guard_write` refuses writes; fingerprinted before and after. |
  | `MyDrive/C10/results/prereg02/**` | Read-only, same as prereg01. |
  | `MyDrive/C10/results/test02/` | Output root. `test02_smoke` when `C10_SMOKE=1`; `test02_rehearsal` when the manifest says `kind == "rehearsal"`. |

- **MANIFEST contract** (written by `t02_test_bundle.py build`, §3.1):
  ```json
  {"format": "c10-test02 manifest v1", "kind": "official|rehearsal|smoke", "created": "...",
   "files": {"<official name>": {"sha256": "...", "bytes": 0, "rows": 0,
                                 "source_sha256": "...", "normalized": []}},
   "t1": {"rank": "PASS|NORMALIZED", "player": "PASS|NORMALIZED"}}
  ```
  The notebook stops if:
  - `kind` is not `official`, unless SMOKE (`smoke`) or a rehearsal
  - any member's SHA differs from the manifest
  - any of the six names is missing
- **Quick preflight in the notebook** (a subset of T1, per file):
  - no BOM
  - 400 rows in each question file
  - `question_id` unique, with prefix `pub_` or `priv_`
  - `num_games` is an int in 5..20
  - columns `sgf_1..20` and `color_1..20` exist
  - exactly the first `num_games` SGF cells are non-empty
  - colours are B/W
  - player only: each candidate file has 40,000 rows, 400 ids × exactly 100 games; prefixes `pub_p_`/`priv_p_` and `pub_`/`priv_`; the two pools share no player id and no game id

  Any failure stops the run and lists what failed.
- **Stale-dump guard.** Each `<set>/dumps/_INPUT.json` holds `{set, question_file, sha256, candidates_sha256?}`. If an existing `_INPUT.json` differs from the current input, stop; the base `dump_is_done` would otherwise skip a replaced file with the same row count.
- **The final cell** always:
  - writes `run_log.json` with a status
  - checks read-only fingerprints for both roots
  - packs `test02_<comp>_results.zip` into the output root (members relative to `results/test02/`, e.g. `rank/...`)
  - runs `drive.flush_and_unmount()`, then prints "Drive 已寫入完成並卸載"

### 2.1 `test02_rank.ipynb` (inference only)

1. **Header markdown (new).** Purpose, steps, GPU choice, stop conditions, outputs, hand-over.
2. **setup (rep).**
   - Paths:
     - `RESULT_ROOT = DRIVE_DIR/results/test02` (smoke or rehearsal variants as in §2.0)
     - `RESULT_DIR = RESULT_ROOT/rank`
     - no `RESULT_CKPT`
   - Read-only roots are `P01_ROOT` and `P02_ROOT` (`_smoke` variants in SMOKE); `guard_write` refuses both.
   - Remove `FINISH_WITHOUT_TRAINING` and `SMOKE_CFG_P02`.
   - `TEST_ZIP = DRIVE_DIR/C10_test_v1.zip`
   - `TEST_FILES = {'public': 'rank_prediction_test_public.csv', 'private': 'rank_prediction_test_private.csv'}`
3. **preflight (new).**
   - Weights: `P02_ROOT/rank/ckpt/epoch_9.pth` must have SHA `423581b4fddd5c2c160ebf691f4e86b9e00c4fe743d5cfb23df7891f4a2ca939`. In SMOKE, instead require equality with the `weights_sha256` in the prereg02_smoke marker.
   - Seam references on Drive:
     - `P02_ROOT/rank/dumps/val_keys.npz` `055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c`
     - `val_p02e9_d4.npz` `ec0b729d805eb9f3458f7a497444b099b8ec47e5cec73703ae4b8bf9cf0a11b2`
   - The builder reads all pins from the repo at build time. It asserts:
     - `frozen_config.json` `dumps == {"p02e9_d4": {weights_sha256: 423581b4…, syms: 8}}`
     - the weights SHA in `_DONE_val_p02e9_d4.json` equals the npz field
   - Also: unzip and verify the test zip (§2.0), the quick preflight, the stale-dump guard, Drive free space ≥ 0.5 GB, and the read-only fingerprints taken before the run.
4. **prepare (rep).** Delete the training-tar block (cell 3, lines 54–74). Keep the bundle SHA checks and the refusal line verbatim.
5. **Official cell 1 (environment).** Unchanged.
6. **Official cell 2 (rep).** Delete the `EXTRA_EPOCHS`/`FINAL_EPOCH` assert and `guard_write(CKPT_DIR)`/`makedirs`. The GPU gate accepts G4 ("RTX PRO 6000") or A100 and has no FINISH clause.
7. **dump tools (base cell + rep).**
   - Delete: `EXPECTED_INNER_GAMES`, `INNER_*`, `decode_with_key`, `KEY_LOADER`, `dump_inner` and the final print.
   - `record_p01_after` becomes a two-root version.
   - `parse_val()` becomes `parse_questions(path)`: a cache dict keyed by path, reading with `pd.read_csv(path, dtype={'question_id': str})`. The SMOKE `head()` stays.
   - `dump_val(net, …)` becomes `dump_questions(net, path, name, …)`.
   - `DUMP_DIR` is re-bound per set before each call; the base functions read it as a global and stay byte-identical.
8. **Three set cells: `seam_val`, `public`, `private`.**
   - `seam_val` uses the bundle's validation question file.
   - Each cell does:
     - `DUMP_DIR = RESULT_DIR/<set>/dumps`
     - `write_keys('val_keys', …)`
     - `dump_questions(net, path, 'p02e9_d4', 8, 'prereg02/rank/ckpt/epoch_9.pth', WSHA)`
   - `net` is loaded once with `load_net(path, expect_sha=PIN)`.
9. **seam check (new), writes `RESULT_DIR/seam_check.json`.**
   - The key arrays (`question_id`, `slot`, `color`, `parsed`) must equal the Drive prereg02 arrays exactly.
   - Recorded: `bitwise_equal`, `max_abs_logit_diff` (over all 8 symmetries), and GPU, torch and cuDNN for both runs.
   - Pass condition, by analogy with the registered 1.1②③: take C0 = the per-question mean softmax over parsed games, float32, original orientation. Then:
     - `max |ΔC0| ≤ tol`, where `tol = 1e-4` if the current GPU name equals the prereg02 marker's GPU ("NVIDIA RTX PRO 6000 Blackwell Server Edition"), otherwise `1e-3`
     - rule-B(C0) differences ≤ 4 of 1,589 (SMOKE: `≤ max(1, ceil(4·n/1589))`)
   - On failure: `status = seam_failed`, pack the zip, flush, stop.
10. **final (new).**
    - Per set: marker, SHA, rows, NaN pattern.
    - Read-only fingerprints after the run.
    - `config.json` (below), `run_log.json`, `timings.json`, `env.json`.
    - Pack and flush.

Stop conditions, in order: GPU model; weights SHA; seam-reference SHA; test zip or manifest; quick preflight; stale dump; free space; any write into prereg01/prereg02; seam; read-only fingerprint changed at the end (`status = readonly_changed`).

Runtime: about 5–8 min of compute plus about 3 min of setup on G4 (not measured).

### 2.2 `test02_player.ipynb` (inference only)

- **setup (rep).**
  - `RESULT_DIR = RESULT_ROOT/player`.
  - Add `P02_ROOT`; `guard_write` refuses `P01_ROOT` and `P02_ROOT`.
  - Add `TEST_ZIP`.
  - `TEST_POOLS = {'public': ('player_identification_test_public.csv', 'player_identification_candidates_public.csv'), 'private': (…_private…)}`.
- **prepare, official 1, environment.** Identical to base.
- **official 2 (rep).** `QUESTIONS_CSV`/`CANDIDATES_CSV` stay the bundle's validation files; they are the `seam_val` pool.
- **preflight (base + rep).**
  - Keep every prereg01 check; the seam uses them.
  - Add: the prereg02 dump SHAs `cand_games.npz` `56ed03d12e9a79871a21b5f69dc4bd4bb8c8aa69b1544cb8ec799e80227cfdea` and `query_games.npz` `665d0888a93f6632eb19edac76abf996d5499cd27c683d9844b97cb274b6f689`.
  - Add: `best_model.pth` SHA `356eb4f65488925e2a552ea382979780c903ef515b85d329a3805db837b7773b`, at the path prereg02 uses.
  - Add: the test zip check and the quick preflight.
  - GPU must be G4.
- **official cell 20.** Identical.
- **dump cell (base + rep), emitted three times.** Each copy starts with a one-line preamble: `POOL='public'; QUESTIONS_CSV, CANDIDATES_CSV = …`. The builder asserts the three bodies are identical text.
  - `DUMP_DIR = os.path.join(RESULT_DIR, POOL, 'dumps')`.
  - The count assert checks `EXPECTED_COUNTS[POOL]`, taken from the preflight.
  - Info goes to `DUMP_INFO_BY_POOL[POOL]`.
  - Order: `public`, `private`, `seam_val`. The base deletion loop removes `REPRO_*` files, so the repro cell must run last.
- **repro cell.** Base verbatim, run on `seam_val`. It is the registered 2.1 criterion: ≥ 398/400 top-5 identical to prereg01 and centroid difference ≤ 1e-4. If it fails, `status = repro_failed`; pack, flush, stop.
- **seam_p02 cell (new), writes `seam_p02.json`.** `bitwise_equal` for `cand_games.npz`/`query_games.npz` arrays against the Drive prereg02 dumps, and `max_abs_emb_diff`. Record only; the local seam decides (§3.3).
- **final (base + rep).** Loop over pools. Check both roots' fingerprints. Pack `test02_player_results.zip`, flush, stop on problems.

Runtime: about 15–20 min on G4 (not measured). Both notebooks can run at the same time.

### 2.3 Output contract (read by local tools and faked by `t02_fake_tests.py`)

```
results/test02/rank/{config.json, run_log.json, env.json, timings.json, seam_check.json,
                     readonly_before.json, readonly_after.json}
results/test02/rank/<set>/dumps/{val_keys.npz, val_p02e9_d4.npz, _DONE_val_keys.json,
                                 _DONE_val_p02e9_d4.json, _INPUT.json}      set in {seam_val, public, private}
results/test02/player/{config.json, run_log.json, env.json, timings.json, repro_check.json,
                       seam_p02.json, readonly_before.json, readonly_after.json}
results/test02/player/<pool>/dumps/{cand_games.npz, query_games.npz, _INPUT.json}
```

`config.json` must contain:
```
smoke, rehearsal (bool), notebook, notebook_sha256, gpu, torch, cudnn, weights {path, sha256},
test_bundle {zip_sha256, manifest_sha256, files {name: sha256}},
sets {<set>: {questions_sha256, candidates_sha256?, rows}}, changes_from_prereg02
```

The npz field names and dtypes are exactly those that `p02_rank_fuse.load_val_keys`/`load_dump` and `p02_player_fuse.Data` read.

### 2.4 CPU smoke of the notebooks (integration step)

1. Reuse the prereg02 smoke chain. Copy `…\scratchpad\p02final\smoke\{rank,player}\smoke\drive\results\prereg0{1,2}_smoke` into `<S>/drive/results/`, after checking that their `config.json` names the committed prereg02 notebooks. Otherwise rebuild it (about 15 min for rank, about 3 min for player; `results/prereg02/design/build_review.md`).
2. Build the smoke test zip into `<S>/drive/C10_test_v1.zip` (`kind: smoke`) with `t02_fake_tests.py from-val --root <S>/fake` and then `t02_test_bundle.py build --kind smoke --tests-dir <S>/fake/data/test --zip-out <S>/drive/C10_test_v1.zip`.
3. Run `colab/run_local_smoke.py colab/test02_rank.ipynb --scratch <S>`, and the same for the player notebook. The runner needs no change.
   - Expect the seam against the `prereg02_smoke` dumps to be bitwise on CPU.
   - Outputs land in `drive/results/test02_smoke`.
4. Run `t02_rank.py`/`t02_player.py` with `--smoke --results-dir <S>/drive/results/test02_smoke --tests-dir <S>/fake/data/test --out-dir <S>/local`.
5. Negative smoke cases:
   - a missing test file
   - a manifest SHA mismatch
   - a tampered weights copy
   - a write into `prereg02_smoke` (must be refused)
   - a stale `_INPUT.json`

## 3. Local tools

### 3.0 Rules shared by all t02 tools (each tool implements its own copy)

- **Modes:**
  - **real** (default): `--results-dir` must be exactly `REPO/results/test02` and `--tests-dir` must be `REPO/data/test`.
  - `--dry-run`: output must be outside the repo.
  - `--smoke`: for notebook SMOKE outputs.
  - `--rehearsal`: real Colab output from a rehearsal zip; output outside the repo.
- **Out-dir guard.** Refuse any output inside `results/prereg01`, `results/prereg02` or `data/local`. Refuse any input path whose basename contains the needle (reuse the `po.refuse_forbidden` pattern; in the rank tool, reimplement it locally).
- **Real mode refuses unless all of these hold:**
  - The frozen configs are committed and equal HEAD: `p02_rank_fuse.require_committed`, `p02_player_fuse.check_committed`.
  - The frozen tool SHAs are unchanged. Rank: `p02_rank_fuse.tool_state()["files_sha256"] == fc["tool"]["files_sha256"]`. Player: `po.source_shas()` matches `fz["tools_sha256"]` and `tools_dirty()` is empty.
  - The tool's own file and `tools/check_submission.py` are committed and clean.
  - The notebook outputs listed in §2.3 for that competition are tracked by git and unchanged against HEAD.
  - `run_log.status == "completed"`.
  - Rank `seam_check.passed`; player `repro_check.passed`.
  - `config.smoke == false` and `config.rehearsal == false`.
  - `sha256(data/test/<name>)` equals `config.test_bundle.files[name]`, and `sha256(data/test/MANIFEST.json)` equals `manifest_sha256`.
  - `results/test02/checks/<comp>/t1.json` verdict is `PASS` or `NORMALIZED` (rank: `FAIL_OPENING` only together with `--fallback R10`).
- Existing outputs are never overwritten without `--rerun --reason "<text>"`. The reason is stored in the report.
- Every report records: git HEAD, the SHA of every input and output, the tool's own SHA, and the Python version.

### 3.1 `tools/t02_test_bundle.py` (I4; stdlib and zipfile only)

```
PY tools/t02_test_bundle.py ingest --src <zip-or-dir> [--dest data/test/raw]
PY tools/t02_test_bundle.py build  --kind official|rehearsal|smoke [--tests-dir data/test/raw]
                                   [--out-dir data/test] [--zip-out data/colab_bundle/C10_test_v1.zip]
PY tools/t02_test_bundle.py unpack --zip <test02_{rank,player}_results.zip> --comp rank|player
                                   [--results-dir results/test02]
```

- **`ingest`** finds the six official names, recursively in a dir or zip. It copies them byte-for-byte to `data/test/raw/` and writes `SOURCE.json` (source path, per-file SHA, time). It refuses if any name is missing or duplicated.
- **`build`** requires `results/test02/checks/{rank,player}/t1.json` to exist for the raw files, with verdict `PASS` or `NORMALIZED`. It then:
  1. Applies only the transforms listed in `t1.json` `normalize`, the whitelist: `strip_bom`, `crlf_to_lf`, `upper_colour`, `reorder_columns_to_readme`, `clear_slots_after_num_games_if_empty_whitespace`, `clear_colour_after_num_games_if_sgf_empty` (PREREG_02 rev 8).
  2. Otherwise writes byte copies to `data/test/<name>`.
  3. Writes `data/test/MANIFEST.json` (§2.0 schema).
  4. Re-runs T1.1–T1.3 on `data/test` through `PY tools/t02_checks.py t1 --tests-dir data/test`; the result must be `PASS`.
  5. Writes the zip with members `tests/*`, and prints the zip SHA.
- **`unpack`** refuses:
  - absolute or `..` member paths
  - existing target files
  - `run_log.status != completed`, unless `--allow-failed` is given (used for inspection only)

  It also checks that `config.test_bundle.manifest_sha256` equals the local MANIFEST.

### 3.2 `tools/t02_rank.py` (I1)

Imports: `p02_rank_fuse`, `p02_rank_opening`, `p02_rank_pseudo` (only if needed), `decide`, `c10data`, `check_submission`. Never any `p02_player*` or `t02_player` module.

```
PY tools/t02_rank.py opening      [--dry-run|--smoke|--rehearsal] [--results-dir D] [--tests-dir T] [--out-dir O]
PY tools/t02_rank.py apply        [same flags] [--fallback R10 --reason TEXT] [--rerun --reason TEXT]
PY tools/t02_rank.py fallback-fit --combo R20|R10      (before 11/04; real repo only)
```

**`opening`**
- Build `OpeningModel` (M = 12, α = 20) with counts from `c10data.load_rows(part="fit")` only.
- Score the slots of `seam_val` (`data/local/rank_val_questions.csv`), `public` and `private`, using `p02_rank_opening.read_questions`, `game_string` and `log_softmax`. Follow `…\scratchpad\testkit\specB\exp_opening.py`.
- Check: the seam part's `ll`, `o`, `has` and `n_terms` must be bitwise equal to the committed `results/prereg02/rank/opening_val.npz` (SHA `75ffec64f9e949ac0be52ce9e72a4a19bb978a25de2b6ed9d355aac7a1a5ae85`). This replaces the `build_id` check in `p02_rank_fuse.py:1138-1139`, which can never pass for a new build.
- Write `O/opening_test.npz` with arrays `set, question_id, slot, color, ll, o, has, n_terms, n_moves` and meta holding `frozen_opening_build_id = c0cb86b47cd3f95c75940cc71b3b7ca24d43c3390c1a510bf96c6ae68e0f93a6`, the input SHAs and the fit-row count. Also write `O/opening_test_report.json`.
- Runtime: about 30–60 s.

**`apply`**
- Read `results/prereg02/rank/frozen_config.json` (SHA `44a16e09679cc6815d41d9a1255882f4a38599cd97cbf66996d447709b4f2efa`). Assert:
  - `basis == p02_rank_fuse.basis()`
  - `b == B @ beta` to 1e-12
  - combo R20 (C5, log, calibrated, opening)
  - each set's dump `weights_sha256 == fc["dumps"]["p02e9_d4"]["weights_sha256"]`
- For each set, in order `seam_val`, `public`, `private`, replicate the numeric core of `cmd_apply` (lines 1106–1169) by calling the frozen functions:
  - `load_val_keys(<set>/dumps)`
  - `load_dump(…/val_p02e9_d4.npz, n, 8)`
  - `check_val_logits`
  - the `np.where(parsed, logits, 0)` float32 cast
  - `source_logprobs("C5", …)`
  - `read_question_meta(question file)`, with the same qid/slot/colour equality checks
  - `pool_matrix`, `pooled(…, "log")`
  - opening lookup by `(set, question_id, slot)` in `opening_test.npz` with the colour check, then `mop`
  - `question_logits`, `softmax`
  - rows written with `repr(float)`, blank probabilities when `n = 0`
  - Reference pattern: `…\scratchpad\testkit\specB\exp_rank_apply.py`
- **Seam first.**
  - Write `O/seam_val/rank_val_probs.csv`, then `decide.run(O/seam_val/rank_val_probs.csv, O/seam_val)`.
  - `O/seam_val/seam_local.json` is `passed` iff:
    - max |ΔP| against `results/prereg02/rank/rank_val_probs.csv` (SHA `a83700432fce5d7d26653fdaee345b0c894729fdff0029a244ed94e06c803b63`) ≤ the `tol` in `seam_check.json`
    - rule-B differences against `results/prereg02/rank/pred_B.csv` (SHA `863a62582424adddc9c2e1821fcc056008bd651e103c19ec8ada98a14f87a2fd`) are ≤ 4
  - Record `byte_identical_probs` and `byte_identical_pred_B`; both are expected true when `seam_check.bitwise_equal`.
  - If the seam fails, stop with no test outputs.
- **Test sets.**
  - `O/rank_test_probs.csv`: public rows in question-file order, then private rows, 800 rows, `PROB_COLUMNS`.
  - `decide.run(O/rank_test_probs.csv, O)` writes `pred_A.csv`, `pred_B.csv` and `decide_summary.json`. Never point it at `results/prereg02/rank`.
  - `O/submission_rank.csv` is a byte copy of `O/pred_B.csv` (LF, `question_id,pred_rank`).
  - `check_submission.check("rank", O/submission_rank.csv, [pub_q, priv_q], official=True)` must return `[]`.
- **`O/apply_test_report.json`** records:
  - combo, parameters, the gate copied from frozen, counts per set
  - `questions_without_parsed_game` (filled with D, the official fallback)
  - `games_parsed`
  - the rule-B prediction distribution and mean probability per level for each set (record-only, for VALIDATION rev 2's check of official per-level counts)
  - all SHAs, and whether the fallback was used
- **`--fallback R10`** is allowed only when `checks/rank/t1.json` verdict is `FAIL_OPENING`. It uses `results/test02/rank/fallback_R10.json` with `opening=False` (`mop = 0`) and records the reason. Everything else is the same.

**`fallback-fit`** (run once before 11/04; output committed)
- Replicate `cmd_tune`'s fit-round preparation (about lines 826–995) by calling the frozen functions:
  - `load_inner_keys`
  - `load_pseudo(results/prereg02/rank/pseudo_inner.json, verify=True)`
  - `load_opening(opening_inner.npz, "inner")`
  - `build_qset` for the fit rounds
  - `source_logprobs`, `pooled`
  - `fit(X_fit, source, pool, opening, B)`
- No OOF, no eval, no validation input.
- `--combo R20` is a self-check: the result must equal the frozen parameters to ≤ 1e-12. The result is written to `O/fallback_selfcheck_R20.json`.
- `--combo R10` writes `results/test02/rank/fallback_R10.json`:
  - C5, log, calibrated, no opening
  - `a`, `gamma`, `beta`, `b`, the refit record
  - `inner_oof_rank_balanced`, copied from `tune_report.json` R10 (0.5687 in RESULTS)
  - input and tool SHAs
  - `selfcheck_R20_passed: true`
- Runtime: under 2 min.

### 3.3 `tools/t02_player.py` (I2)

Imports: `p02_player_fuse` (as `pf`), `p02_player_open` (as `po`), `check_submission`. No rank module, no `decide`.

```
PY tools/t02_player.py run --pool seam_val|public|private [--dry-run|--smoke|--rehearsal]
                           [--results-dir D] [--tests-dir T] [--out-dir O] [--no-cnn --reason TEXT]
                           [--rerun --reason TEXT]
PY tools/t02_player.py submit [same mode flags]
```

**Inputs per pool**

| Pool | Candidates / questions | Dumps |
|---|---|---|
| `seam_val` | `data/local/player_val_candidates.csv` and the validation question file | `D/player/seam_val/dumps/` |
| `public` / `private` | `T/player_identification_candidates_<pool>.csv` and `T/player_identification_test_<pool>.csv` | `D/player/<pool>/dumps/` |

`public` and `private` refuse to run unless `O/seam_val/seam_local.json` has `passed: true`.

**`run`**
1. **Select.** Same code as `cmd_select:755-797`, without the `repro_local` check and the frozen-file-exists check. Pattern: `…\scratchpad\testkit\specB\exp_player.py`.
   - `d = pf.Data(cand_csv, D/player/<pool>, use_cnn, smoke)`.
   - `pf.self_retrieval(d, pf.COMBOS, strict=True)`.
     - Strict mode needs exactly 100 games per candidate; T1.2 guarantees it, otherwise stop.
   - `_stats` gives the means.
   - `pf.choose(means, order)`.
   - Gate: `selected − Q00 ≥ 0.02`.
   - `sr_med[o] = median(max_cos[o])`.
2. **Test-time gate rule (rev 5, D1).** If the gate fails, `effective = Q00`. Record both `selected` and `effective`.
3. **Guard.**
   - Take thresholds from `results/prereg02/player/frozen_config.json` (SHA `d70726ab198643c749b8dadc78f37b52c69bd831513e36ec564a9d04e2c7f3de`) `["guard"]`.
   - Assert `po.sha256_text_file(results/prereg02/player/guard_calibration.json) == 793c16a79080e57d4c60941a1fc311f3c0d950e19c42bafaff16a54f7ad43a28` and equals `fz["guard"]["sha256"]`.
   - The ρ denominator is this pool's `sr_med`.
4. **Apply.** Same as `cmd_apply:936-969`.
   - `d = pf.Data(…, q_path=question_csv)`.
   - Build open cos, ρ and E terms.
   - `b_c`, `b_q` via `po.colour_share`.
   - `pf.guard_decision(effective, rho, guard, b_c, b_q, cnn)`.
   - `pf.fuse`, then `pf.top_ids`.
   - Fallback rows: the first 5 sorted candidates.
   - Rows must be written in **question-file order**: assert that `d.qids` equals the file order, or reorder explicitly.
5. **Outputs:**
   - `O/<pool>/pool_select.json`: sr_report-style rows, means, round means, `selected`, `best`, gate, `effective`, `sr_max_cos_median`, guard, input SHAs.
   - `O/<pool>/top5.csv`: LF, `question_id,top_1..top_5`.
   - `O/<pool>/apply_report.json`: `cmd_apply` report fields, plus `sr_top1_distinct_per_round` for the applied combination (record-only, for T6).
6. **Seam pass rule** (`O/seam_val/seam_local.json`). All of:
   - the self-retrieval means equal `results/prereg02/player/sr_report.json` (SHA `9c06f6ac852092adfe4d0c2824da011baf481006a32fcf0a2051971df28cfc2e`); record `exact` and `max_abs_diff`
   - `selected == Q04`, gate passed, guard level 1
   - top-5 identical to `results/prereg02/player/player_val_top5.csv` (SHA `69ca74a61ce34f8ffa8949df731df433d474347dce9aabb535156ab8197ead6c`) for ≥ 398/400 questions

   Byte identity is expected when the Colab dumps are bitwise.
7. **`--no-cnn --reason`.** The registered fallback list {Q22, Q23} (rev 3 item 7, rev 4 item 4). The gate becomes self-retrieval mean ≥ 0.50. If the guard lands at level 3, the selection is kept, and at test time the pool is still submitted.

**`submit`**
- Requires `public` and `private` reports, and T1 `PASS` or `NORMALIZED` for player.
- Writes `O/submission_player.csv`: a single header, then the public rows, then the private rows (LF).
- `check_submission.check("player", sub, [pub_q, priv_q], [pub_c, priv_c], official=True) == []`.
- Writes `O/submit_report.json`: per pool, the selected, effective and applied combination, guard level, ρ, `b_c`/`b_q`, `n_fallback`, SHAs.
- Runtime: under 1 min per pool (prereg02: select 11.9 s, apply 5.4 s).

### 3.4 `tools/check_submission.py` changes (I4)

```
PY tools/check_submission.py rank   SUB --questions PUB_Q PRIV_Q [--official]
PY tools/check_submission.py player SUB --questions PUB_Q PRIV_Q --candidates PUB_C PRIV_C [--official]
```

- API: `check(kind, sub_path, question_files, candidate_files=(), official=False) -> list[str]`. Default behaviour stays as today except:
  1. Zero question files → error `"no question files given"`.
  2. A duplicate `question_id` within one file → `"question X appears twice in <file>"`. Duplicates across files keep today's message.
  3. `main`:
     - kind not in {rank, player}, missing SUB, a positional argument before the first flag, or an unknown flag → print usage to stderr and `exit 1`, no traceback
     - `--candidates` given for `rank` → usage error
- `official=True` adds:
  - exactly 2 question files, each with exactly 400 rows and unique ids; all ids in file 1 start with `pub_` and all in file 2 with `priv_`
  - player: exactly 2 candidate files, each with 40,000 rows, 400 distinct `player_id` with exactly 100 rows each
    - prefixes `pub_p_`/`priv_p_`
    - `game_id` unique, with prefix `pub_`/`priv_`
    - the two pools share no player id and no game id
    - every pick's prefix matches its question's prefix, with a clear message naming the wrong pool
  - the body has exactly 800 rows
  - row order equals the question files' row orders concatenated in `--questions` order; report the first mismatching line
  - basename is exactly `submission_rank.csv` or `submission_player.csv`
  - OK message: `OK (official): … 800 rows in official order`
- Tests go in the new file `tests/test_check_submission_official.py`, using synthetic files at official size. Cases:
  - OK
  - reversed order
  - 399 rows
  - prefixes swapped between files
  - a candidate with 99 games
  - pools sharing a game id
  - wrong basename
  - zero question files
  - positional argument before a flag → exit code 1 and no `Traceback` on stderr
  - `tests/test_tools.py::StrictCheck` unchanged and passing

### 3.5 `tools/official_log.py` (I4; stdlib only; writes `results/official_scores.json`)

```
PY tools/official_log.py prepare  --comp rank|player --file PATH --reason TEXT [--method TEXT]
PY tools/official_log.py uploaded --id N --ok-at ISO --uploaded-at ISO
PY tools/official_log.py public   --id N --score FLOAT [--lb-rank INT] [--note TEXT]
PY tools/official_log.py final    --comp rank|player --id N --basis TEXT
PY tools/official_log.py status
```

- **`prepare`** requires:
  - the file is committed and clean
  - `check_submission --official` passes against `data/test`
  - the name matches `--comp`

  It appends an entry with status `prepared`, and prints the exact OK-request text: competition, team, file, rows and `sha256[:12]`.
- **`public`** computes the change from the previous uploaded entry of the same competition. It sets `investigate: true` when:
  - the change is ≤ −0.05, or
  - the score is below the official baseline (rank 0.4631, player 0.2845)
- **`status`** prints, per competition:
  - the last upload and whether its SHA equals the final choice
  - uploads per local date (maximum 5)
  - days left
- Schema:
  ```json
  {"format": "c10 official_scores v1", "teams": {"rank": "TEAM_11248", "player": "TEAM_11249"},
   "baselines_public": {"rank": 0.4631, "player": 0.2845}, "final_choice": {"rank": null, "player": null},
   "uploads": [{"id": 1, "comp": "rank", "status": "prepared|uploaded", "file": "results/test02/rank/submission_rank.csv",
                "sha256": "...", "rows": 800, "git_commit": "...", "method": "prereg02 R20 rule B (甲)",
                "reason": "...", "check_submission": "OK (official)", "prepared_at": "...", "user_ok_at": null,
                "uploaded_at": null, "public_score": null, "public_delta_vs_prev": null, "lb_rank": null,
                "investigate": false, "note": ""}]}
  ```

### 3.6 Test-day command sequence (`docs/TESTDAY.md`, written by I4)

`PY` = `.venv\Scripts\python.exe`, run from the repo root. Commit after each numbered block.

0. **T0.** `git clone` https://github.com/AILAB-NDHU/AICup-2026-Tutorial into scratch and diff the README and both notebooks against 359e06f, the commit recorded in `third_party/aicup2026_tutorial/SOURCE.txt`. Save the diff to `results/test02/checks/t0_tutorial_diff.txt`. If the file names, columns or submission format changed, take the D5 path: change only the name table in each builder or tool, rebuild, and the seam must still pass.
1. Ingest and check:
   - `PY tools/t02_test_bundle.py ingest --src "<download>"`
   - `PY tools/t02_checks.py t1 --comp rank --tests-dir data/test/raw`
   - `PY tools/t02_checks.py t1 --comp player --tests-dir data/test/raw`
2. `PY tools/t02_test_bundle.py build --kind official`, then hand the zip and the notebooks to the user.
3. While Colab runs: `PY tools/t02_checks.py run --comp rank`, and the same with `--comp player`.
4. Unpack the results and commit before any local step: `PY tools/t02_test_bundle.py unpack --zip … --comp rank`, the same for player, then commit.
5. Rank: `PY tools/t02_rank.py opening`, then `PY tools/t02_rank.py apply`.
6. Player: `PY tools/t02_player.py run --pool seam_val`, then `--pool public`, then `--pool private`, then `submit`.
7. `PY tools/t02_checks.py t6 --comp player`
8. Commit the outputs. Then, per competition, `official_log.py prepare` → OK request (its own message) → the user uploads → `uploaded` → `public` → commit.

Expected local wall time after the zips arrive: under 15 min.

## 4. T1–T6 checker: `tools/t02_checks.py` (I3)

Imports: `c10data`, `p02_moves` (SYM_TABLE), and `p02_player_open` (`game_tokens`, `moves_of`) for T3 only. Never `p02_rank_fuse`, `decide`, `t02_rank` or `t02_player`.

```
PY tools/t02_checks.py precompute [--out results/test02/null]                 # before 11/04, training only
PY tools/t02_checks.py t1  --comp rank|player [--tests-dir data/test] [--out results/test02/checks]
PY tools/t02_checks.py run --comp rank|player [--tests-dir data/test] [--only T2,T3,T4,T5]
                           [--index-part train|fit] [--out …]
PY tools/t02_checks.py t6  --comp player [--submission results/test02/player/submission_player.csv]
```

- Outputs go to `results/test02/checks/<comp>/{t1,t2,t3,t4,t5,t6,summary}.json`.
- `summary.json` holds `{check: {verdict, consequence, details_file}}`.
- `--dry-run` and `--root` behave as in §3.0.

**Common game key (T2 and T4).**
- Moves come from the regex `;([BW])\[([a-s]{2})\]`, with no 60-move limit.
- A game's key is `sha1(min over the 8 D4 images of the concatenated point string)[:12]`, with the colour letters left out.
- Speed-up: first find the symmetries that tie on the first 12 moves, then compare only those in full. The result equals the full 8-way minimum.
- Three keys per game: full, K100 (first 100 moves), K60 (first 60 moves).
- First-move D4 class = the lexicographic minimum of the 8 images of the 2-letter point.

### 4.1 Precompute (before 11/04; training players only; about 12 min, peak about 3 GB)

Port `…\scratchpad\testkit\C\t3final.py`, `t2_null.py`, `t4.py`, `t6.py` and `cache.py`. Read directly from `load_rows` with no pickle cache. Write:

- **`results/test02/null/t3_null.json`:**
  - the definitions below
  - reference: 5,767 fit players, 720,307 games, `sha256_sorted_player_ids = 9940feed82f435a770030a17b32b5bbf7dc1002daa61616a989b24580bdf664f`
  - per orientation: `q99`, `idf_sha256`, vocabulary size, null quantiles, exceedance per level, holdout result, true-positive tables, power table, the T3q table
  - the rule
- **`results/test02/null/t3_null_{raw,canon}.npz`** (about 270 KB each): `player_id`, `level`, `null_S2`, `holdout_S2`, `pos_all_S2`, `pos_all_m`, `idf`.
- **`results/test02/null/train_refs.json`:**
  - S99 = `{cc,cd,ce,cf,dd,de,df,jj}`
  - the training first-move class distribution
  - black share 0.5006
  - upper-right share 0.8166, where upper-right means `x > 'j' and y < 'j'`
  - move-count min/p1/median/max = 100/107/228/445
  - T2 null counts
  - T4, T5 and T6 simulation ranges

**Pinned values (must be reproduced exactly):**

| | raw | canon |
|---|---|---|
| q99 | 0.1004267242871777 | 0.08085695648061325 |
| Vocabulary size | 146,421 | 144,097 |
| Null players | 1,985 | 1,985 |
| Inner holdout players | 226 | 226 |
| Null values above q99 | 19 | 19 |
| Holdout above q99 | 1 | 3 |

If any value differs, stop. Before 11/04, fix the port or write a revision.

The optional pre-build of the T2 index goes to `data/cache/t2_train_index.npz`: 793,863 games, keys plus a player index, about 151 s. It is derived data, so it is gitignored.

### 4.2 T1: schema (both competitions; the only check that gates the pipeline)

`t1.json` has `verdict ∈ {PASS, NORMALIZED, FAIL_STRUCT, FAIL_OPENING}`, a list of normalization transforms, and per-subcheck details.

- **T1.1 Columns.**
  - Question files: the column set equals `{question_id, num_games, sgf_1..sgf_20, color_1..color_20}`.
  - Candidate files: the column set equals `{game_id, player_id, sgf_content, color}`.
  - A different column order only → `NORMALIZED` (`reorder_columns_to_readme`).
- **T1.2 Counts and IDs.**
  - Each question file has 400 rows.
  - `question_id` is unique across public and private, with `pub_`/`priv_` matching the file.
  - Player candidates: 40,000 rows per file, 400 distinct `player_id`, each exactly 100 rows.
    - prefixes `pub_p_`/`priv_p_`
    - `game_id` unique with prefix `pub_`/`priv_`
    - the pools share no player id and no game id
  - Any violation → `FAIL_STRUCT`.
- **T1.3 Slots and encoding.**
  - `num_games` is an int in 5..20.
  - For i ≤ n: `sgf_i` non-empty and `color_i` in {B, W}. For i > n: both empty.
  - These can be normalized: BOM, CRLF, lower-case b/w, whitespace-only cells after n. Anything else → `FAIL_STRUCT`.
  - rev 8: for i > n, a colour letter B/W/b/w with an empty or whitespace-only `sgf_i` is normalizable (`clear_colour_after_num_games_if_sgf_empty`); `sgf_i` with content after n stays `FAIL_STRUCT`.
- **T1.4 SGF grammar, per game.**
  - (a) No AB/AW/AE/HA/PL, and SZ = 19.
  - (b) At least one move, the first move is B, moves strictly alternate, and there is no pass (`B[]` or `B[tt]`).
  - Other property names are recorded only.
  - Verdict `FAIL_OPENING` iff more than 1% of a file's games violate (a) or (b). Any violation is recorded, with up to 20 game ids. Training shows 0 violations.
- **T1.5 Starts at move 1.** `share = #games whose first-move class ∈ S99 / #games`, per file. PASS iff share ≥ 0.95; otherwise `FAIL_OPENING`.
  - Training null minimum over 100 simulated files: 0.983 (candidate pools), 0.985 (question files).
  - Games starting at move 5, 11 or 21 give 0.766, 0.470 or 0.278.
  - A start at move 2 is caught by T1.4(b). A start at move 3 cannot be detected.
- **Record only:** move-count min/p1/median/max.
- **Consequences (rev 5):**
  - `NORMALIZED` → normalized copies are used everywhere (§3.1).
  - `FAIL_STRUCT` → stop; write a revision before any prediction.
  - `FAIL_OPENING` →
    - rank uses `--fallback R10`
    - player keeps the per-pool procedure (self-retrieval and guard react on their own)
    - T3 = "not evaluable"
- Runtime: seconds.

### 4.3 T2: exact duplicates

- **Index** from `load_rows(part="train")` (fit + inner): 6,407 players, 793,863 games, 79.4% of all training games. Validation players are not in the index; the stored hashes carry no answer or validation ids. `--index-part fit` exists only for the dry-run negative control.
- **Copy** = the K100 keys match; for games under 100 moves, the full keys match (index at that length built when needed). K60 matches are record-only.
- **Threshold per unit:** FAIL iff `copies ≥ max(10, ceil(0.001 × games in the unit's test file))`.
- **Units:**
  - (a) test file vs training, per file. Also record the number of candidates with a copied game, and whether each such candidate's copies all map to one training player.
  - (b) questions vs their own pool's candidates, full key, per pool. Same-player K100 repeats are record-only.
  - (c) across questions in one file, full key, counting games. Record same colour (same player in several questions) and opposite colour (the two target players played each other) separately.
  - (d) public vs private, full key, per competition.
  - Rank files are never compared with player files.
- **Null (training only; recorded in `train_refs.json`):**
  - inner vs fit (different people): 0 for full, K100 and K60
  - full-game duplicates within fit: 0
  - within fit, games sharing a K100 key only with the same player: 15; a K60 key only with the same player: 156; a K60 key with another player: 54
- **Consequence (registered):** open split v2 and write a new registration. Uploads of the frozen 甲 continue. No prediction may use the copies before the new registration exists.
- **Runtime:** 151 s to build the index (or load it from the cache), plus about 0.17 ms per test game.

### 4.4 T3: are the candidates training players? (player only)

- **Reference R.** All 5,767 fit players, each profiled on all their fit games.
  - Tokens: `p02_player_open.game_tokens`, both token sets, 10 moves.
  - Weights: `tf = 1 + ln(count)`, `idf = ln(5768/(1+df)) + 1` with df over R.
  - Rows are L2-normalized.
  - A test token missing from R's vocabulary enters only the row norm, with `idf = ln(5768) + 1`.
- **Candidate profile.** Its 100 pool games.
- **Statistic.** `S2(c) = top1 − top2` of the cosine over R, computed separately for raw and canon.
- **Null.** 1,985 fit players with ≥ 100 games, each using the first 100 games in `sha256('c10-t3|cand|{pid}|{gid}')` order, with their own column left out. `q99 = np.quantile(null, 0.99, method="higher")`.
- **Rule.** Per pool: `n_raw = #{S2_raw > q99_raw}`, `n_canon = #{S2_canon > q99_canon}`. "Overlap" iff `n_raw ≥ ceil(0.10·400) = 40` OR `n_canon ≥ 40`. This implements the registered "≥ 10%".
- **On test day:**
  - Rebuild R and assert that q99 and `idf_sha256` equal the committed null.
  - If T1 is `FAIL_OPENING`, T3 = "not evaluable".
  - If T2(a) fires, overlap is proven directly.
- **Properties (training only):**
  - The null exceedance per candidate for raw OR canon is 1.31%.
  - The worst single-level P(declare) is 3.5e-16 (raw) and 2.5e-10 (canon).
  - Power of raw OR canon at a given share of training players among the candidates:

    | Share | 0 | 0.2 | 0.3 | 0.4 | 0.5 | 1.0 |
    |---|---|---|---|---|---|---|
    | P(declare) | 0 | 0.020 | 0.539 | 0.974 | 0.9998 | 1.0 |

  - Overlap of about 20% of candidates or less is **not** detectable.
- **T3q** (record-only): k-game profiles of questions. q99(k) ranges from 0.070 to 0.077 (raw) and from 0.065 to 0.076 (canon). This is too weak to decide anything.
- **Consequence (registered):** consider 乙 under a new registration. Uploads of 甲 continue.
- **Runtime:** about 3 min for both orientations and both pools.

### 4.5 T4: colour and first move

- Black share per file: `b_c` (candidates) and `b_q` (questions), pooled over the whole pool.
  - Training reference: 0.5006.
  - Null ranges: candidate pools 0.490–0.521, question files 0.472–0.517.
  - The only decision is `|b_c − b_q| > 0.5`, which moves the player guard to level 3 (CNN only). It is enforced inside `guard_decision`; T4 only reports it.
- First-move class distribution: TVD against training (record-only). Null maximum: 0.048 (candidate pools), 0.073 (question files).
- Rotation flag (record-only): upper-right share < 0.60.
  - Training 0.8166; null 0.791–0.839 (candidates) and 0.749–0.806 (questions); randomly rotated games give 0.245.
  - Note: R20 does not depend on rotation (8-symmetry average and canonical n-grams). For the player, the per-pool self-retrieval chooses the orientation.

### 4.6 T5: `num_games` (record only)

- Per file: histogram over 5..20, mean, share below 10, and a χ² test against uniform 5..20 (df 15).
- References: uniform gives mean 12.5 and 0.3125 below 10; the local rank validation set has 12.06 and 36.4%.

### 4.7 T6: one question per candidate? (descriptive only; run after `submit`)

- Records:
  - whether questions = candidates = 400 per pool
  - T2(c) same-colour duplicates
  - distinct top-1 candidates per pool in the submission
  - the same count per self-retrieval round (`sr_top1_distinct_per_round`)
- Training simulation references: one-to-one 348–369; self-retrieval rounds 347–367; question owners drawn with replacement 244–254.
- **Registered:** without an explicit official statement, the one-to-one structure is not used.

## 5. End-to-end dry run without answers, CPU smoke, rehearsal

### 5.1 `tools/t02_fake_tests.py` (I4; csv, numpy, hashlib; imports no p02 model tool; `--root` must be outside the repo)

**`from-val --root S [--zip-kind rehearsal|smoke]`**: the pipeline dry-run tree.

- `S/data/test/<6 official names>`, prefix replacement only, all other bytes identical:
  - rank public = validation questions in rows 1–400 with `val_` → `pub_`
  - rank private = rows 401–800 with `val_` → `priv_`
  - player questions: all 400 in both sets (`val_pq_…` → `pub_pq_…`/`priv_pq_…`)
  - candidates: `val_p_<hex>` → `pub_p_<hex>`/`priv_p_<hex>` and `val_g_NNNNNN` → `pub_g_NNNNNN`/`priv_g_NNNNNN`
- `S/data/test/MANIFEST.json`, built with `t02_test_bundle.build` logic.
- `S/results/test02/rank/seam_val/dumps`: byte copies of `results/prereg02/rank/dumps/{val_keys,val_p02e9_d4}.npz` and their markers.
- `public`/`private` dumps: row slices of those arrays with only the id fields re-keyed. Logits and `weights_sha256` are byte-identical; markers and `_INPUT.json` are regenerated.
- The same for player: `seam_val` = byte copies of the prereg02 `cand_games.npz`/`query_games.npz`; pools = re-keyed copies with `emb_sum` and `n_windows` byte-identical.
- Fake notebook records: `config.json` (`smoke: false`, `rehearsal: true`, test SHAs), `seam_check.json`/`repro_check.json` `passed`, and `run_log` `completed`.
- Content statistics (T1.4–T5) are **never** run on this tree; only `t1 --only T1.1,T1.2,T1.3` is.

**`from-train --root S2`**: the checks dry-run tree. Pools come from `load_rows(part="fit")` and inner players, with synthesized ids `pub_p_<sha8>` and `pub_g_<n>`.

| Pool | Built from | Role |
|---|---|---|
| POS | 400 fit players × 100 games, questions from their other games | positive control |
| NEG | 226 inner players with ≥ 100 games; T3 threshold `ceil(0.1·226) = 23` | negative control |

A rank question file is built from inner players.

### 5.2 `tests/test_t02_dryrun.py` (integrator; `C10_SLOW=1`, about 3 min; `C10_SLOW=2` adds the checks tree, about 8 min)

The pinned values below were verified in scratch (`specB/exp_*.py`).

1. `t02_fake_tests.py from-val --root S`, then `t02_checks.py t1 --comp rank|player --tests-dir S/data/test --only T1.1,T1.2,T1.3 --dry-run` → `PASS`.
2. `t02_rank.py opening --dry-run …`: the seam part is bitwise equal to `opening_val.npz`.
3. `t02_rank.py apply --dry-run …`:
   - `seam_val/rank_val_probs.csv` byte-identical to prereg02 (a8370043…)
   - `seam_val/pred_B.csv` byte-identical (863a6258…)
   - `rank_test_probs.csv` rows equal the validation rows under prefix mapping (1–400 → `pub_`, 401–800 → `priv_`)
   - `submission_rank.csv` labels equal `pred_B.csv` for those 800 questions
   - `check --official` returns `[]`
4. `t02_player.py run --dry-run --pool seam_val`:
   - means exactly equal `sr_report.json`
   - `sr_med` equals the frozen value (raw 0.22665843779803582, canon 0.22122531121305591)
   - Q04, gate +0.4991241719699286, level 1
   - `top5.csv` byte-identical to `player_val_top5.csv`
5. `--pool public`: Q04 (0.902147 against Q08 0.898223), gate passed, level 1 (ρ raw 1.0126), top-5 400/400 equal under prefix mapping. `--pool private`: Q04 (0.898212 against Q08 0.897055), level 1 (ρ raw 1.0040), 400/400.
   - The first green run writes full-precision golden values into the test.
   - These values depend on the fake id scheme, because self-retrieval hashes `player_id|game_id`.
6. `t02_player.py submit --dry-run`: 800 rows; `check --official` returns `[]`.
7. (`C10_SLOW=2`) `from-train`:
   - T1 `PASS`
   - T2(a) FAIL on POS files (positive control)
   - with `--index-part fit`, NEG rank questions give 0 copies → `PASS`
   - T3: POS declared overlap; NEG not declared
   - T4 inside the null ranges

**Fast negative unit tests** (in each owner's test file):
- tampered `weights_sha256`
- test CSV SHA differs from `config.json`
- out-dir inside the repo or inside `results/prereg02`
- seam or repro not passed
- frozen-tool SHA mismatch (monkeypatched)
- a pool with 99 games for one candidate
- `--fallback R10` without `FAIL_OPENING`
- `public` run before `seam_val` passed
- `decide` out-dir pointing at prereg02

### 5.3 Colab rehearsal (recommended, 10/28–11/02; needs the user)

`from-val --zip-kind rehearsal` → `C10_test_v1.zip` (kind `rehearsal`) → the user runs both notebooks → outputs go to `results/test02_rehearsal`. Then run the local tools with `--rehearsal --results-dir <unpacked> --tests-dir S/data/test`. The expectations of 5.2 apply, with the seam on real G4.

Nothing from the rehearsal is uploaded. The records are kept in scratch, plus a one-line note in PREREG rev 6 or the RESULTS file.

## 6. `official_scores.json` and upload protocol (VALIDATION discipline 4 + `competitions/CLAUDE.md`)

- **U1 First upload per competition.** As soon as all of these hold:
  - T1 is `PASS`/`NORMALIZED`
  - the notebook and local seams passed
  - `check_submission --official` returns OK
  - the outputs are committed
  - T2–T5 have been run and recorded

  T2–T5 never block the first upload: the registered consequences still allow 甲 uploads. Target 11/04–11/05. Purpose: catch format errors early; this is not model selection.
- **U2 Before every upload.**
  - `official_log.py prepare` passes.
  - Claude sends one message per file: competition, team, path, rows, `sha256[:12]`.
  - The user replies OK in chat, for that file only; no earlier approval carries over.
  - The upload instruction is its own message.
- **U3 After an upload.** The user reports the public score, and `uploaded` and `public` are logged and committed.
  - Upload rejected or format error → fix the bug, re-run all checks, get a new OK.
  - Public score below the official baseline, or ≥ 0.05 below the previous upload in the same competition → `investigate`. Look for bugs in ID mapping, row order, pool pairing, seam and guard level before any further upload. Never change the method because of the score.
  - Anything else: no action. 400 public questions give a standard error of up to 0.025, so smaller differences are not evidence.
- **U4 Allowed later uploads:**
  - (a) a bug fix
  - (b) a method adopted by a later registration (for example 03) under the local validation rules
  - (c) re-uploading the final choice so that it is the last upload

  Never upload variants to test them on the public leaderboard.
- **U5 Final choice.** The latest method adopted under the local-validation primary metric and the registered adoption conditions; today that is prereg02 for both competitions.
  - `official_log.py final` records its SHA.
  - The final upload happens no later than 11/23 23:59; 11/24 is kept as a buffer.
  - After that, no further upload unless the final upload itself failed.
  - `status` must show that the last upload's SHA equals `final_choice`.
- **U6.** Any choice made after seeing official scores is written into the report (VALIDATION 4). Scores are quoted only from `official_scores.json`.

## 7. Registration to commit as PREREG_02 rev 5 before any test data exists (the exact text)

Add this revision-table row:

> `| rev 5 | (commit) | 2026-09-2x，官方測試集 11/04 11:00 才開放，**還沒有任何測試資料**。登記測試日的程序（第 9 節）：只推論的測試 notebook、接縫核對、每池程序的細節、交件組裝、T1–T6 的統計量與門檻、上傳規則。T3 的虛無分布、R10 備案參數在 11/04 前只用訓練玩家算好並 commit（`results/test02/null/`、`results/test02/rank/fallback_R10.json`） |`

Add this new section:

> ## 9. rev 5：測試日的程序（11/04 前登記，當時沒有任何測試資料）
>
> 1. **範圍**：交件模型是甲（第 3 節）。測試日只推論，不訓練、不調參。兩場的 `frozen_config.json`、prereg02 的權重和 `tools/p02_*.py`、`c10data.py`、`decide.py`、prereg02 notebook 一律不改。測試日的程式是新檔：`colab/build_test02_*`、`colab/test02_*.ipynb`、`tools/t02_*.py`。輸出只寫在 `results/test02/`。
> 2. **接縫核對**：每次測試日的 Colab 執行，都用同一份權重和同一段程式，把本機驗證資料重算一次。比對分兩層：
>    - notebook：
>      - 競賽二：驗證題 1,589 題，原方向逐盤 softmax 的每題平均和 prereg02 最大差 ≤ 1e-4（GPU 型號和 prereg02 相同）或 ≤ 1e-3（不同）；規則 B 不同 ≤ 4 題。
>      - 競賽一：驗證候選池，照 2.1 的重現核對（和 prereg01 至少 398/400 題 Top-5 相同、重心最大差 ≤ 1e-4），另記錄和 prereg02 的逐盤 embedding 是否逐位元組相同。
>    - 本機：
>      - 競賽二：R20 的每題機率和 `results/prereg02/rank/rank_val_probs.csv` 最大差 ≤ 同一容許差、規則 B 不同 ≤ 4 題。
>      - 競賽一：自我檢索選出 Q04、閘門通過、防護第 1 層，Top-5 和 `player_val_top5.csv` 至少 398/400 題相同。
>    - 逐盤輸出逐位元組相同時，預期本機結果完全相同。
>    - 任何一項沒過：那次執行的測試輸出一律不用、不上傳；換一個執行階段重跑一次。還是沒過就停下查清楚；要放寬容許差，照 rev 4 第 8 條另寫修訂，修訂前不上傳。
> 3. **競賽二的套用**：
>    - public、private 兩份題目各自照 R20 的凍結參數算每題機率，公式和合併方式同 1.2。
>    - 開局模型同 1.2（只用 fit 的棋計數，M＝12、α＝20）；接縫部分的開局分數必須和 `opening_val.npz` 逐位元組相同。
>    - 決策用規則 B（`tools/decide.py`）；一盤都解析不了的題填 D（官方 notebook 的做法）。
> 4. **競賽一的套用（2.2「每個候選池各自重跑同一個程序」的細節）**：
>    - 每一池各自做：自我檢索（相同的雜湊、10 輪、Q00–Q23 清單、0.003 容許）→ 選組合 → 閘門（比 Q00 高 ≥ 0.02）。
>    - **測試時閘門沒過，這一池改用 Q00**（測試時一定要交件；「未送分」只適用本機驗證）。
>    - 防護的 θ₁、θ₂ 用已 commit 的 `results/prereg02/player/guard_calibration.json`，不重算；ρ 的分母用這一池自己的自我檢索中位數；b_c、b_q 用這一池的候選和題目。
>    - 兩池可能選到不同組合、落在不同層級，都照實記錄。
>    - 候選不是每人剛好 100 盤時停下，另寫修訂，修訂前不交競賽一。
>    - 退回只用開局（rev 4 第 4 條）時，測試時落在第 3 層仍然交件，保留選到的組合。
> 5. **交件組裝**：
>    - `submission_rank.csv`：public 題目檔的列順序接 private 題目檔的列順序，共 800 列，欄位 `question_id,pred_rank`，UTF-8 無 BOM。
>    - `submission_player.csv`：順序相同，欄位 `question_id,top_1..top_5`，每題只填所屬那一池的玩家。
>    - 上傳前 `tools/check_submission.py --official` 必須通過。
> 6. **T1（欄位與棋譜）**：
>    - T1.1 欄位集合。
>    - T1.2 每份 400 題；每池 400 人 × 各 100 盤；前綴 `pub_`／`priv_`；兩池不共用玩家和棋。
>    - T1.3 前 `num_games` 格有棋與執色、其後為空。
>    - T1.4 每盤沒有 AB/AW/AE/HA/PL、SZ＝19、第一手是黑、黑白交替、沒有 pass；超過 1% 的棋不符才算不過。
>    - T1.5 第一手落在訓練集 99% 的 8 種對稱類別 {cc,cd,ce,cf,dd,de,df,jj} 的比例 ≥ 0.95。
>    - 後果：
>      - 只有欄位順序、BOM、行尾、執色大小寫、`num_games` 之後的空白格不同時，在本機產生正規化副本，只改這些並記錄前後 SHA-256；Colab 和本機都用副本。
>      - 其他結構不符時停下，另寫修訂。
>      - **T1.4 或 T1.5 不過**：開局特徵失準。競賽二改用 R10：同一 tune 程序在全部擬合資料重擬合；參數在 11/04 前算好並 commit；inner 交叉驗證 0.5687，沒有在本機驗證集打分。競賽一照常每池自我檢索和防護。T3 記為「無法判斷」。
> 7. **T2（同一盤棋）**：
>    - 定義：
>      - 全部落子取 8 種對稱中最小的座標序列，雜湊前 12 碼（不含執色字母）。
>      - 「複製」＝前 100 手相同（不到 100 手的棋比整盤）。
>    - 比對單位：
>      - (a) 每個測試檔 vs 訓練玩家的棋（fit＋inner）
>      - (b) 每池的題目 vs 同池候選（整盤）
>      - (c) 同一檔的題目之間（整盤，同執色、異執色分開記）
>      - (d) public vs private（整盤）
>      - 競賽二和競賽一的檔案不互相比對。
>    - 門檻：複製數 ≥ max(10, ⌈0.001 × 該檔盤數⌉) 就算不過。
>    - 後果照第 4 節：開 split v2 另寫登記；甲照常上傳；新登記前任何預測不得利用這些複製。
> 8. **T3（候選是不是訓練玩家）**：
>    - 參考集是 5,767 位 fit 玩家（各用全部 fit 棋）。
>    - token 同 2.2（兩套、10 手），權重 1＋ln(次數)，IDF＝ln(5768/(1＋df))＋1。
>    - 統計量 S2＝對參考集 cosine 的第 1 名減第 2 名。
>    - 虛無分布：1,985 位 ≥ 100 盤的 fit 玩家，各取雜湊順序前 100 盤、去掉自己。門檻取 99 百分位（raw 0.1004267242871777、canon 0.08085695648061325）。
>    - 判定：一池 400 人中 raw 或 canon 超過門檻的人數 ≥ 40（10%）就判定重疊。
>    - 測試日重建參考集，門檻和 IDF 的 SHA-256 要和 commit 的完全相同。
>    - 限制：候選中訓練玩家只占約 20% 以下時偵測不到。
>    - 後果照第 4 節（考慮乙，另寫登記；甲照常上傳）。
>    - 揭露：S2 是看過 6 種統計量在 inner 留出玩家上的結果後才選的。
> 9. **T4–T6**：
>    - T4：|b_c − b_q| > 0.5 時競賽一照 2.2 第 3 層只用 CNN。第一手分布（和訓練集的總變異距離）、右上角比例 < 0.60 只記錄。
>    - T5 只記錄。
>    - T6 只記錄（題數與人數是否相等、同一檔題目間同執色的重複、交件中不同的第 1 名人數）；沒有官方明文就不用一題對一人。
> 10. **GPU**：競賽二用 G4 或 A100；競賽一只用 G4，拿不到先等，等一天再由 Claude 決定（rev 3 第 8 條）。
> 11. **官方定義變動**：11/04 重抓官方教學 repo，和 `359e06f` 比對。檔名、欄位或交件格式有變時，只改各程式的檔名表並重建；接縫核對仍要通過。
> 12. **上傳規則**（VALIDATION 紀律 4 的做法）：
>     - 第一次上傳在檢查和接縫都過、交件檔已 commit 之後盡早進行，目的只是抓格式錯誤。
>     - 每次上傳前使用者要在對話裡對那一份檔案說 OK。
>     - public 分數低於官方 baseline，或比同一場上一次低 0.05 以上，只追查錯誤，不據此換方法。
>     - 最後一次上傳只依本機驗證的主要指標和事前登記的採用條件決定，在 11/23 23:59 前完成。
>     - 每次都記進 `results/official_scores.json`。
> 13. **揭露**：
>     - T1–T6 的統計量和門檻是在訓練玩家上設計的。
>     - 測試日的本機程式把 `cmd_apply` 的步驟用新檔重寫，靠接縫核對證明和 prereg02 相同。
>     - R10 備案沒有在本機驗證集打分。

If implementation or precompute cannot reproduce a pinned number, record it as rev 6 before 11/04 11:00.

## 8. Ownership for 4 parallel implementers (disjoint files) and integration

| Owner | Files, exclusive to this owner | Tests |
|---|---|---|
| Main session, **before** I1–I4 start | `docs/PREREG_02_OPENING_FUSION.md` (rev 5, §7) → commit | none |
| **I1 Rank** | `colab/build_test02_rank_nb.py`, `colab/test02_rank.ipynb`, `results/test02/notebook_diff_rank.txt`, `tools/t02_rank.py`, `results/test02/rank/fallback_R10.json` (+ `fallback_selfcheck_R20.json`) | `tests/test_t02_rank.py` (synthetic, plus the negatives in §5.2), `tests/test_t02_nb_rank.py` (byte-identical rebuild, segment identity, needle exactly once, pins consistent with repo files) |
| **I2 Player** | `colab/build_test02_player_nb.py`, `colab/test02_player.ipynb`, `results/test02/notebook_diff_player.txt`, `tools/t02_player.py` | `tests/test_t02_player.py`, `tests/test_t02_nb_player.py` |
| **I3 Checks** | `tools/t02_checks.py`, `results/test02/null/{t3_null.json, t3_null_raw.npz, t3_null_canon.npz, train_refs.json}` | `tests/test_t02_checks.py` (synthetic SGFs for each T1 subcheck, the key symmetry invariance, the threshold formula, the T3 rule on a synthetic null) |
| **I4 Infra** | `tools/check_submission.py` (edit), `tools/t02_test_bundle.py`, `tools/t02_fake_tests.py`, `tools/official_log.py`, `.gitignore` (+ `results/test02/*.zip`, `results/test02_rehearsal/`), `docs/COLAB_RUN.md` (new section "測試集（test02）" only), `docs/TESTDAY.md` | `tests/test_check_submission_official.py`, `tests/test_t02_infra.py`, `tests/test_t02_hygiene.py` (see below) |
| **Integrator**, after I1–I4 | `tests/test_t02_dryrun.py`, `results/test02/design/build_review.md`, all commits | runs §5 and §2.4 |
| Main session, after integration | `CLAUDE.md` status, `D:\AI-Portfolio\PORTFOLIO_MAP.md` line C10, `competitions/TIMELINE.md` if dates change | none |

**The contracts each implementer codes against:**
- the §2.3 layout and `config.json` fields (I1, I2 and I4)
- the §2.0 MANIFEST (I1, I2, I4)
- the §3.4 `check()` signature (I1, I2 call it; I4 implements it)
- `t1.json` `verdict` and `normalize` (I3 writes; I1, I2 and I4 read)
- the §5.1 fake tree (I4 writes; I1 and I2 tests may use small synthetic versions)

Cross-module calls are mocked in unit tests; the real calls are checked at integration.

**`tests/test_t02_hygiene.py` (I4) asserts:**
- The needle is absent from `tools/t02_*.py`, `tools/official_log.py`, `colab/build_test02_*.py` and `tests/test_t02_*.py`. The test builds the needle as `"ans" + "wers"`.
- `colab/test02_*.ipynb` has exactly the one inherited refusal line.
- Team separation:
  - `t02_rank` imports no `p02_player*` or `t02_player`
  - `t02_player` imports no `p02_rank*`, `decide` or `t02_rank`
  - `t02_checks` imports no `p02_rank_fuse`, `decide`, `t02_rank` or `t02_player`
- **Pin test:**
  - the current CRLF-folded SHAs of the rank tools equal `results/prereg02/rank/frozen_config.json` `tool.files_sha256`
  - `po.source_shas()` equals the player `frozen_config.json` `tools_sha256`
  - `colab/prereg02_*.ipynb` SHAs equal 9fb1f9d0… and 738c1dd8…
  - `git status --porcelain -- results/prereg01 results/prereg02` is empty

**Rules for implementers:**
- No commits.
- No answers, no `score.py`, no GPU.
- Keep scratch data under their own `testkit/<label>` directories.

**Integration steps, in order:**
1. Run the full `PY -m pytest tests -q`: prereg02 and the new tests must all be green.
2. Build each notebook twice and confirm the rebuilds are byte-identical.
3. Run `t02_checks.py precompute` and confirm the pinned values.
4. Run `t02_rank.py fallback-fit --combo R20` (self-check), then `--combo R10`.
5. `C10_SLOW=2` dry run.
6. CPU smoke of both notebooks, including the negative cases.
7. An independent read-only review agent writes `build_review.md`; fix what it finds and repeat steps 1–6.
8. Two commits:
   - code, tests and docs
   - `results/test02/null/*`, `fallback_R10.json` and the notebook diffs
9. Delete scratch data copies once they have been ported: `testkit/specB/fake_player` (about 110 MB), `testkit/C/cache_train.pkl` (171 MB), and the fake trees.

Everything must be done well before 11/04; target the first week of October, with the rehearsal 10/28–11/02.

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Colab changes torch/cuDNN before 11/04 | The seam fails and uploads are delayed | Seam check and rehearsal. Registered path: rerun once, then a revision (rev 4 item 8). There are 20 upload days. |
| No G4 available | Player notebook is blocked | Rank can use A100. Player waits one day, then Claude decides (rev 3 item 8). |
| Official format differs (names, column order, lower-case colours, `_q_` infix) | Readers fail or give wrong results | T0 diff, T1 whitelist normalization, name tables. The frozen `norm_colour` fails loudly on lower case rather than silently. |
| Official split is not random within each player (file order or colour split) | Player falls from 0.883 (local) towards CNN only (about 0.37 local prereg01) | The registered three-level guard per pool and the per-pool self-retrieval. Local and official scores are not comparable. |
| Test players are in training | 甲 may be below what 乙 could reach | T2(a) and T3 detect it, then a new registration. T3 cannot detect overlap of about 20% or less. |
| The t02 rewrite of the `cmd_apply` steps drifts from the frozen code | Wrong submission | Byte-identical seam dry run, the local seam on real output, and the pin test. |
| A frozen tool gets edited by accident | The frozen guarantee breaks and apply refuses | Hygiene pin test and the real-mode SHA checks. |
| Wrong file uploaded to the wrong competition, or the last upload is not the chosen file | Submission invalid or wrong final | `--official` basename, header and pool checks. OK messages name the competition. `official_log status`. Final upload by 11/23. |
| The T3 port does not reproduce the scratch null | Registered numbers not met | Pinned values; rev 6 before 11/04. The scratch scripts are session-scoped, so port them first. |
| Team-separation grey zone (one person, two teams) | Rule risk | No shared t02 module, per-competition check runs, no cross-competition file comparison; the reports disclose the shared preprocessing. |
| Repo growth | About 100 MB more for the committed test02 dumps (local git, no remote) | Accepted, same practice as prereg02. The zips are gitignored. |
| November workload (C1 crunch, E.SUN daily D-Plan) | Missed windows | First upload 11/04–05, then only the final-choice upload. The kit needs about 1 h from the user on 11/04. |
| Data licence | Test data on Drive/Colab | Same use as the training bundle already on Drive; never on HF or in public repos. |

Reference scratch files for porting: `…\scratchpad\testkit\C\{t3final.py, t2_null.py, t4.py, t6.py, cache.py, t3final.json, t4.json, t6.json}`, `…\scratchpad\testkit\specB\{exp_rank_apply.py, exp_opening.py, exp_player.py}`, `…\scratchpad\testkit\formatA\probe_check.py`. Here `…` = `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1`.