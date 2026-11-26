# test04 規格：競賽二 04（Q02）的測試日擴充

- 依據：`docs/PREREG_04_OPPONENT.md` 第 5 節（rev 1）、第 10 節（rev 2），以及本資料夾的 `rev3_draft.md`（K5 前 commit 成 rev 3）。
- 優先順序：登記 > 本規格 > 四份調查筆記（`understand_t02.md`、`understand_p04.md`、`understand_nb.md`、`understand_protocol.md`）。行號、dtype、實測數字以調查筆記為準，本規格只引用。
- 分支 `p04-testday`，從 main `6ec5e68` 開出。K5 11/03（二）23:59 前合併回 main，內部目標 10/30。
- 用字：Q02＝R20＋O4＋SWA＋SWB（`results/prereg04/rank/frozen_config.json`）。「04」指 Q02 在官方測試集上的那一份交件。R20 是 11/04 先交的那一份（test02）。
- 所有指令的 `PY` 是 repo 根目錄的 `.venv/Scripts/python.exe`（Python 3.12.13、numpy 2.5.3、scipy 1.18.1、zlib 1.3.1；zlib 是 9/30 設計階段用 `zlib.ZLIB_RUNTIME_VERSION` 實測，`val_report.json` 沒有記 zlib）。
- 本規格已照三個角度的審查修過一輪（第 12 節「critique log」逐條記錄接受或不接受）。和舊版不同的地方以本版為準。

## 0. 範圍、非目標、凍結檔

### 0.1 要做的

1. `tools/t04_rank.py`：`unpack`（收 test04 notebook 的 zip）、`opp`（CPU 重播 O4／T4）、`apply`（Q02 的本機接縫、Δκ、交件檔）、`status`（唯讀摘要）、`archive`（封存失敗的一次）。
2. `tools/t04_fake_tests.py`：用 prereg03 的驗證 dump 做假的 test04 notebook 輸出（演練用）。
3. `colab/build_test04_rank_nb.py` → `colab/test04_rank.ipynb`：只推論，03 兩臂在三組題目上的正確執色與對調 logits。
4. `tools/official_log.py` 的附加式修改：認得 04 的檔、5.3 的最後一版規則、K6。
5. 測試：`tests/test_t04_rank.py`、`test_t04_nb.py`、`test_t04_official_log.py`、`test_t04_hygiene.py`、`test_t04_dryrun.py`。
6. 文件：`docs/TESTDAY_04.md`（新）、`docs/COLAB_RUN.md` 檔尾新一節、PREREG_04 rev 3。
7. 演練：R20 的輸出逐位元組不變、04 對得回 prereg04、`C10_SLOW=2`、notebook 的 CPU 冒煙。

### 0.2 不做的

- 不重擬合、不新增來源或組合；Q02 的參數一律讀 `frozen_config.json`。
- 不做 MV、不做「test04 失敗就改交 Q01」（PREREG_04 第 7 節）。
- 04 的任何輸出都不進競賽一；不 import 任何 player 模組。
- 本機不用 GPU；不開答案檔；不跑 `tools/score.py`。
- 假樹裡的棋是驗證棋譜：只准跑重播和 apply，不跑任何內容統計（T1.4–T5、dupcheck 等）。
- 11/04 當天不做 04。

### 0.3 凍結檔與保護方式

| 檔案 | 規則 | 誰保護 |
|---|---|---|
| R20 路徑 39 個檔（`results/prereg04/design/r20_path_fingerprint.json`，含 `docs/TESTDAY.md`、`tools/t02_*.py`、`tools/check_submission.py`、`colab/*test02*`、`results/test02/**` 當時的檔） | 一個位元組都不能動 | `tests/test_p04_hygiene.py`（指紋 SHA 常數＋tag `r20-testday-pre04`） |
| `tools/official_log.py` | 只准插入新行；tag 版每一行照原順序出現在新檔 | `test_p04_hygiene.py`（子序列）＋`test_t02_infra.py` 的 OfficialLog 原封不動全過 |
| `tools/p02_*.py`、`p03_rank_fuse.py`、`p04_opp.py`、`p04_rank_fuse.py`、`c10data.py`、`decide.py` | 不改 | `t04_rank.py` 執行時核對 LF SHA＝`frozen_config.json` 的 `tool.files_sha256`；`test_t04_hygiene.py` 同樣核對 |
| `tools/score.py`、`tools/make_val.py`、`colab/build_prereg0*_*.py`、`colab/prereg0*_*.ipynb`、`results/score_ledger.csv`、`docs/PREREG_01..03*.md` | 和 `6ec5e68` 相同 | `test_t04_hygiene.py`（`git diff --quiet 6ec5e68 -- <檔>`） |
| `results/prereg01..04/**` | 沒有修改、沒有新檔、沒有 bytecode | `test_p04_hygiene.py` 管 01–03 和 `prereg04/design`；`test_t04_hygiene.py` 補 `prereg04/opp`、`prereg04/rank`、`prereg04/*.md` |
| `docs/PREREG_04_OPPONENT.md` | 只加 rev 3 的一列和第 11 節 | `test_t04_hygiene.py`：`6ec5e68` 版的每一行照順序出現在新檔 |
| `results/test02/**` 在測試日會多出的檔 | t04 一律不寫 | `t04_rank.py` 的輸出目錄檢查＋`test_t04_dryrun.py` 的前後快照 |
| `data/local` 的答案檔 | 永不開啟 | 路徑含 `"ans" + "wers"` 就拒絕（原始碼不寫出那個字）；`test_t04_hygiene.py` 掃描 |

## 1. 檔案與負責人（三位實作者的檔案互不重疊）

| 負責 | 檔案 | 說明 |
|---|---|---|
| I1 | `tools/t04_rank.py` | 第 3 節 |
| I1 | `tools/t04_fake_tests.py` | 第 3.8 節 |
| I1 | `tests/test_t04_rank.py` | 第 10.1 節 |
| I2 | `colab/build_test04_rank_nb.py` | 第 4 節 |
| I2 | `colab/test04_rank.ipynb` | 產生器的輸出，不手改 |
| I2 | `results/test04/notebook_diff_rank.txt` | 產生器的輸出（逐格 diff） |
| I2 | `colab/make_test04_smoke_tree.py` | CPU 冒煙用的 Drive 替身樹（第 8.5 節） |
| I2 | `tests/test_t04_nb.py` | 第 10.2 節 |
| I2 | `data/colab_upload/test04/test04_rank.ipynb` | 上傳副本，不進 git，只有這一個檔 |
| I3 | `results/test04/design/r20_fake_golden.json` | **第一件事**，見第 8.1 節 |
| I3 | `tools/official_log.py` | 第 9.2 節，只附加 |
| I3 | `tests/test_t04_official_log.py`、`tests/test_t04_hygiene.py`、`tests/test_t04_dryrun.py` | 第 10.3–10.5 節 |
| I3 | `docs/TESTDAY_04.md`（新）、`docs/COLAB_RUN.md`（只在檔尾加一節） | 第 9.4、9.5 節 |
| I3 | `.gitattributes`、`.gitignore`（只加行） | 第 2.4 節 |
| 主流程 | `docs/PREREG_04_OPPONENT.md`（rev 3）、`results/test04/design/build_review.md`、repo 的 `CLAUDE.md`（含把第 80 行的「`docs/TESTDAY.md` 的 04 步驟」改成 `docs/TESTDAY_04.md`，並寫明第 9.4 節開頭那段「U5 對競賽二由 5.3 取代」）、`PORTFOLIO_MAP.md` 的 C10 那一行 | 審查、修正之後 |

- 三者之間的介面：
  - 第 2.3 節「notebook 紀錄的格式」：I2 寫、I1 讀、I1 的假樹照著產生。
  - 第 3.4–3.6 節的報告欄位：I1 寫、I3 的 `official_log` 和演練測試讀。
  - 第 3.0 節的 CLI：I3 的 `test_t04_dryrun.py` 和 `TESTDAY_04.md` 照著呼叫。
- import 規則：
  - `t04_rank.py`、`t04_fake_tests.py`：一律 `import t02_rank as t2`、`import t02_test_bundle as tb`、`import t02_fake_tests as tf`、`import p02_rank_fuse as rf`、`import p03_rank_fuse as p3`、`import p04_opp as po`、`import p04_rank_fuse as pf`、`decide`、`check_submission`。不用 `from X import Y`（測試靠 `setattr` 換模組常數）。
  - `official_log.py`：只用標準函式庫，不 import `t04_rank`。
  - `build_test04_rank_nb.py`：只用標準函式庫；不 import 任何 `colab/build_*` 產生器（它們 import 時就會寫檔），也不 import `tools/p0*`。
  - 例外型別：`t2.T02Error`、`rf`／`p3`／`po` 的錯誤是 `RuntimeError` 系；`tb.BundleError`、`tf.FakeError`、`official_log.LogError` 繼承 `SystemExit`，`except Exception` 接不到，要明寫。
- 順序：I3 先做 golden（第 8.1 節）→ 三人平行 → 主流程整合、跑第 10.6 節全部驗證 → 三個角度獨立審查 → 修正 → rev 3 定稿、一起 commit → K5 前合併。

## 2. 資料版面

### 2.1 本機 `results/test04/`（real mode；只有 `t04_rank.py` 和 I2 的產生器寫這裡）

```
results/test04/
  design/                         設計紀錄（本檔、調查、rev3_draft.md、build_review.md、r20_fake_golden.json）
  notebook_diff_rank.txt          產生器輸出
  test04_rank_results.zip         使用者拖進來的 zip（gitignore；SHA 記在 unpack_sw.json；檔名可能帶 " (1)"）
  unpack_sw.json                  unpack 的紀錄
  sw/                             test04 notebook 的輸出（zip 解開，成員根目錄是 sw/）
    config.json run_log.json env.json timings.json seam_check.json readonly_before.json readonly_after.json
    seam_val/dumps/ public/dumps/ private/dumps/
      val_keys.npz _DONE_val_keys.json _INPUT.json
      val_p03a_d4.npz _DONE_val_p03a_d4.json val_p03a_s0_swap.npz _DONE_val_p03a_s0_swap.json
      val_p03b_d4.npz _DONE_val_p03b_d4.json val_p03b_s0_swap.npz _DONE_val_p03b_s0_swap.json
  opp/                            t04_rank.py opp
    o4_seam_val.npz o4_public.npz o4_private.npz opp_report.json
  rank/                           t04_rank.py apply（交件檔只在這裡）
    seam_val/rank_val_probs.csv pred_A.csv pred_B.csv decide_summary.json seam_local.json
    rank_test_probs.csv pred_A.csv pred_B.csv decide_summary.json submission_rank.csv t04_apply_report.json
    attempt_verdict.json          apply 以結束碼 1 或 4 結束時寫（第 3.3 節「attempt_verdict」）；有這個檔就沒有 t04_apply_report.json
  failed/attempt_N/               t04_rank.py archive：sw/（有的話）、unpack_sw.json（有的話）、rank/（有的話）、attempt.json
  RESULTS.md                      測試日之後
```

- notebook 輸出放 `sw/`、不放 `rank/`：
  - 三個產生者各有自己的資料夾（notebook、`opp`、`apply`），交件檔只在 `rank/`，`official_log` 以這個資料夾認 04。
  - zip 成員的根目錄是 `sw/`，萬一有人用 `t02_test_bundle.py unpack --comp rank` 解它，會因為成員不在 `rank/` 底下被拒絕，不會解進 `results/test02/rank/`。
- 彩排：`results/test04_rehearsal/`（gitignore），版面相同。
- `--dry-run`、`--smoke`：一律在 repo 外，版面相同。

### 2.2 Drive（AI Pro 帳號，`MyDrive/C10/`）

| 路徑 | 用途 |
|---|---|
| `C10_bundle_v1.zip` | 官方 `utils.py`、`network.py`；seam_val 的題目檔 `local/rank_val_questions.csv`（SHA `73e51bbf…`） |
| `C10_test_v1.zip` | 11/04 test02 用的同一份；public、private 題目檔＋`tests/MANIFEST.json` |
| `results/prereg03/rank_{a,b}/ckpt/final.pth` | 兩臂權重（唯讀；只有這一份） |
| `results/prereg03/rank_{a,b}/dumps/` | 接縫參照（唯讀） |
| `results/test02/rank/config.json`、`run_log.json` | 核對用（唯讀）：test02 已完成、用的是同一份測試包 |
| `results/test04/sw/` | 本 notebook 的輸出。接縫類失敗、登記的重跑前改名 `sw_failed_N`；中途斷線改名 `sw_interrupted_N`（第 5.4 節） |
| `results/test04/test04_rank_results.zip` | 給使用者下載的 zip。改名 `sw` 時同時改名成 `test04_rank_results_failed_N.zip`／`test04_rank_results_interrupted_N.zip`，讓「還沒看到新檔就等」的保護重新有效 |
| `results/test04_rehearsal/…`、`results/test04_smoke/…` | 彩排、冒煙（冒煙只在本機替身樹） |

### 2.3 notebook 紀錄的格式（I2 寫、I1 讀；假樹照這裡產生）

- **`sw/config.json`**（照 T02 的欄位，另加兩臂的東西）：
  - `format: "c10-test04 rank notebook config v1"`、`smoke`、`rehearsal`（bool）、`kind`（MANIFEST 的 kind）。
  - `gpu`、`torch`、`cudnn`、`device`。
  - `notebook: "colab/test04_rank.ipynb"`、`notebook_sha256`（T02 的規則，第 4.6 節）、`base_notebooks: {"colab/test02_rank.ipynb": sha, "colab/prereg03_rank_a.ipynb": sha}`。
  - `weights: {"a": {"label": "prereg03/rank_a/ckpt/final.pth", "sha256": …}, "b": {…}}`。
  - `seam_reference: {"a": {<檔名>: sha256}, "b": {…}, "gpu": <參照 GPU>}`。
  - `test_bundle`：和 T02 的 `config.test_bundle` 同一個形狀（`zip_sha256`、`manifest_sha256`、`kind`、`files: {<名稱>: sha256}`），由沿用的 T02 程式碼寫出。
  - `test02_record`：從 Drive 讀的 `results/test02/rank/config.json` 的 SHA-256、它的 `test_bundle.manifest_sha256`、`run_log.json` 的 `status`（smoke 時可以是 null）。
  - `sets: {<set>: {"questions": 路徑, "questions_sha256", "questions_count", "rows"}}`。
  - `tf32: {"recorded_03": {"cudnn": true, "matmul": false}, "default": {…}, "used": {…}, "benchmark_default", "benchmark_used"}`。
  - `dumps`：各組的執行紀錄（`SET_LOG`；每組記 `questions_path`（當時的 `QUESTIONS_PATH`）、`rows`、兩臂 `_DONE_` 的 `rows`）；`prereg: "docs/PREREG_04_OPPONENT.md rev 3 section 11"`。
- **`sw/run_log.json`**：`{"status": …, "runs": [{"run": n, "status": …, …}]}`，狀態值和 T02 相同（`completed`、`seam_failed`、`readonly_changed`、`final_check_failed`、`preflight_failed`、`interrupted`、`running`）。
- **`sw/seam_check.json`**：
  - `passed`（bool）、`tol`（1e-4 或 1e-3）、`seam_dump_gpu`、`reference_gpu`、`same_gpu`。
  - `keys_equal`、`weights_ok`、`env_same`、`env`（12 個 logits 標記加 3 個 key 標記的 `gpu_name`／`torch`／`cudnn`）。
  - `arms: {"a": {…}, "b": {…}}`，每臂：`c0_s0_max_abs_diff`、`c0_swap_max_abs_diff`、`passed`，以及只記錄的 `bitwise_equal_d4`、`bitwise_equal_swap`、`file_sha_equal_d4`、`file_sha_equal_swap`、`max_abs_logit_diff_per_sym`（8 個）、`max_abs_logit_diff_swap`、`max_abs_e_diff_s0`、`max_abs_e_diff_swap`、`rule_b_differences_s0_record_only`。
  - `bitwise_equal`：兩臂四個 logits 全都逐位元組相同（只記錄；本機用它決定要不要發警告）。
  - `created`、`earlier_runs`。
- **`<set>/dumps/`**：
  - `val_keys.npz`：`question_id <U*`、`slot int32`（1 起算）、`color <U1`、`parsed bool`；題目檔列序，題內依 slot。
  - `val_p03{a,b}_d4.npz`：`logits <f4 [n,8,10]`、`weights_sha256 <U64`、`syms int64 = 8`。
  - `val_p03{a,b}_s0_swap.npz`：`logits <f4 [n,10]`、`weights_sha256`、`view = "s0_colour_swapped"`。
  - 解析失敗的列整列 NaN；npz 以 `np.savez`（不壓縮）寫，zip 成員時間固定 1980-01-01（03 的 `write_dump` 原樣）。
  - `_DONE_<stem>.json`：`file`、`rows`、`syms`（8／1／null）、`bytes`、`sha256`、`weights`、`weights_sha256`、`seconds`、`time`、`run`、`gpu_name`、`torch`、`cudnn`、`device`（03 原樣）。
  - `_INPUT.json`：`{"set", "question_file", "sha256"}`，逐字照 T02 的 preflight ④（`build_test02_rank_nb.py:582`；`question_file` 是 `"bundle local/<檔名>"` 或 `"tests/<檔名>"`）。`t04_fake_tests` 照這個格式寫；`t04_rank` 只讀 `set` 和 `sha256`。

### 2.4 `.gitattributes`、`.gitignore`（I3，只加行）

- `.gitattributes` 加 `results/test04/** -text`（報告和交件檔的 SHA 要綁定；這台 `core.autocrlf=true`）。`colab/*.ipynb -text` 已涵蓋 `test04_rank.ipynb`。
- `.gitignore` 加：`results/test04/*.zip`、`results/test04_rehearsal/`。`data/` 已整個忽略（`data/test04_failed/`、`data/colab_upload/test04/` 不必另加）。

## 3. `tools/t04_rank.py`

### 3.0 共同規則

**指令**

```
PY tools/t04_rank.py unpack  --zip Z  [MODE] [DIRS] [--allow-failed]
PY tools/t04_rank.py opp             [MODE] [DIRS] [--workers N] [--rerun --reason TEXT [--bug-revision N]]
PY tools/t04_rank.py apply           [MODE] [DIRS] [--rerun --reason TEXT [--bug-revision N]]
PY tools/t04_rank.py status          [MODE] [DIRS]
PY tools/t04_rank.py archive --reason TEXT [--bug-revision N]   (real mode only)

MODE: 不給＝real；--rehearsal | --dry-run | --smoke 最多一個
DIRS: --results-dir R4      test04 的根（讀 R4/sw，寫 R4/opp、R4/rank、R4/unpack_sw.json）
      --t02-results-dir D2  test02 notebook 的根（讀 D2/rank/<set>/dumps、D2/rank/{config,run_log,seam_check}.json）
      --t02-out-dir O2      test02 apply 的輸出（讀 O2/opening_test.npz、O2/apply_test_report.json、O2/seam_val/seam_local.json）
      --tests-dir T         測試檔（正規化副本）＋MANIFEST.json
      --checks-dir C        T1（讀 C/rank/t1.json）；預設 D2/checks
```

- **real mode**：`R4 = results/test04`、`D2 = results/test02`、`O2 = results/test02/rank`、`T = data/test`、`C = results/test02/checks`，給別的值就拒絕。
- **其他模式**：R4、D2、O2、T 都要給；R4 必須在 repo 外，唯一例外是 `--rehearsal` 可以用 `results/test04_rehearsal`。
  - `--dry-run`：假樹（`t02_fake_tests.py from-val` 加 `t04_fake_tests.py from-val`），sw 的 config 要 `smoke false、rehearsal true`；notebook 身分只記錄。
  - `--rehearsal`：真正在 Colab 跑的彩排 zip，config 要 `smoke false、rehearsal true`；notebook 身分不符只警告。
  - `--smoke`：notebook 的 CPU 冒煙輸出，config 要 `smoke true`；所有接縫只記錄。
- **寫入的位置**：只寫 R4 底下（real 還有 `results/test04/failed/`）。R4／輸出目錄落在 `results/prereg0*`、`results/test02*`、`data/local`、`results/score_ledger.csv` 裡就拒絕。暫存（`score_games` 的 89 MB `tables.npy`）用 `tempfile.mkdtemp(prefix="c10_t04_")`，必須在 repo 外。
- **不覆蓋**：輸出已存在就拒絕，除非 `--rerun --reason TEXT`；被取代檔的 SHA-256 和原因記進報告（`t2.remove_outputs` 的做法）。
- **答案檔**：任何路徑轉小寫含 `"ans" + "wers"` 就拒絕（`t2.refuse_needle`）。
- **結束碼**：
  - `0`：完成、條件都成立。
  - `1`：接縫類失敗（第 5.4 節第一列；`unpack` 看到的也算）：這次 notebook 的輸出不用，走「重跑一次」（`archive`）。
  - `2`：拒絕或錯誤（前提不符、檔案不對、zip 不是重算的結果、notebook 在 R20 的 U1 之前跑）：照訊息修好前提再跑本機工具。
  - `3`：登記的「04 不用」：任一組（含 seam_val）重播失敗 > 0.5%、Δκ ≤ 0／NaN／沒有子題、T1 是 FAIL_OPENING、real mode 下已過 K6（2026-11-16 23:59:59 +08:00）。最終交 R20（FAIL_OPENING 時 R10），不重跑。
  - `4`：本機漂移（seam_val 的 O4 不逐位元組相同但失敗 ≤ 0.5%、seam_val 的 O4 Δκ 和 prereg04 不同、`apply` 的 O4 Δκ 和 `opp` 記的不同、`.venv` 版本不同）：停下查本機程式或 `.venv`，不動 Colab。
- **K6**：real mode 的 `opp`、`unpack`、`apply` 開頭先比現在時間和 K6，過了就結束碼 3（訊息：「K6 已過：這一季不用 04，最終 R20」）。`status`、`archive` 不受影響。操作上 `apply` 要在 11/15 前 commit（TESTDAY_04 寫明）。
- **登記修訂的核對**（`--bug-revision N`；`official_log` 的 `--not-04-reason` 用同一個定義，兩邊各寫一份、測試用同一組 fixture）：`N ≥ 4`；`docs/PREREG_04_OPPONENT.md` 已 commit 且乾淨；`git show HEAD:docs/PREREG_04_OPPONENT.md` 有一行以 `| rev N |` 開頭；加進這一行的 commit（`git log --reverse -S "| rev N |" --format=%cI -- docs/PREREG_04_OPPONENT.md` 的第一筆）時間晚於它要推翻的那份紀錄（`opp_report`／`t04_apply_report` 的 `created`、或 04 上傳紀錄的 `prepared_at`）。不符就結束碼 2。rev N 和它的 commit 記進報告或 `attempt.json`。
- **每份報告都記**：`format`、`mode`、`created`、`seconds`、git HEAD、`tool`（本檔和 import 的凍結工具的 LF SHA）、`environment`（`po.environment()` 的欄位，另加 `zlib: zlib.ZLIB_RUNTIME_VERSION`；`po.environment()` 本身沒有 zlib）、每個輸入和輸出的 SHA-256、`rerun`。
- **所有 CSV／JSON 的比對**用 `t2.pin_sha256`（CRLF 折回 LF）；npz 用原始位元組。

**real mode 的共同前提**（`opp`、`apply` 都檢查；`unpack` 檢查其中標 ★ 的）

1. ★ 本檔 `tools/t04_rank.py`、`tools/check_submission.py`、`tools/official_log.py` 已 commit 且乾淨。
2. `results/prereg04/rank/frozen_config.json`：已 commit、乾淨、LF SHA＝`ae728e9c936a55dfeced255e9b05b5ac6a4955a7101f108e78c53c679e162661`；`po.require_frozen_config()` 通過；`combo.id == "Q02"`、`combo.terms == ["OP","O4","SWA","SWB"]`。
3. 凍結工具：`po.tool_state()`、`pf.frozen_tools_check()` 通過；`p04_opp.py`、`p04_rank_fuse.py`、`p02_*.py`、`p03_rank_fuse.py`、`c10data.py`、`decide.py` 的 LF SHA 等於 `frozen_config.json` 的 `tool.files_sha256`，而且乾淨。`p04_opp.py` 另外要等於 `results/prereg04/opp/val_report.json` 記的值（`b594e0ef…`）。
4. **T1**：先讀 `C/rank/t1.json` 的 verdict。
   - `FAIL_OPENING` → 結束碼 3，訊息：「T1 是 FAIL_OPENING：04 不上場（PREREG_04 5.3），最終是 R10，照 docs/TESTDAY_04.md 第 1 步寫 RESULTS」。
   - `FAIL_STRUCT` → 結束碼 2：「照 test02 停下」。
   - 其餘交給下一條。
5. **test02 已完成**（`ctx2 = t2.resolve(Namespace(dry_run=…, smoke=…, rehearsal=…, results_dir=D2, tests_dir=T, out_dir=O2, checks_dir=C, rerun=False, reason=None))`）：
   - `t2.common_checks(ctx2, "apply", False)`：R20 的凍結設定、test02 notebook 的輸出已 commit、notebook 身分、測試檔對得上 notebook 的 `test_bundle`、各組 `_INPUT.json`、完整 raw T1 是 PASS 或 NORMALIZED、權重。
   - `t2.check_dump_env(ctx2)`：test02 三組 dump 同一個環境。
   - `O2/apply_test_report.json` 已 commit：`mode == "real"`、`fallback_used is False`、`seam_local.passed is True`、`check_submission.mode == "official"` 而且沒有錯誤。
   - `O2/opening_test.npz` 已 commit，`t2.load_opening_test(path, qfiles)` 通過。
6. **R20 的第一次上傳已記錄**（PREREG_04 5.3）：`results/official_scores.json` 已 commit 且乾淨；有一筆 `comp == "rank"`、`status == "uploaded"`、`file == "results/test02/rank/submission_rank.csv"`，`sha256` 等於 test02 報告的 `outputs_sha256["submission_rank.csv"]`。沒有就拒絕：「先照 docs/TESTDAY.md 做完 R20 的第一次上傳」。
7. `.venv`：Python（`sys.version`）、numpy、scipy 等於 `results/prereg04/opp/val_report.json` 的 `environment` 記的；zlib 等於常數 `ZLIB_PIN = "1.3.1"`（`val_report.json` 沒有記 zlib；這個值是 9/30 設計階段實測，註解引本規格開頭那一行）。不同就結束碼 4，訊息請還原 `.venv`。最後的把關仍是 `o4_seam_val.npz` 的檔案 SHA 等於 `30d312c0…`。

非 real 模式：第 1、2 條的「已 commit」、第 6 條略過（記 `not checked (<mode>)`）；第 5 條的 `apply_test_report.json`、`opening_test.npz` 改成「`mode` 等於本次的模式、不要求 commit」（彩排的 O2 是 `<S>\local_rank`）；其餘照做（`t2.common_checks` 自己會依模式放寬）。

### 3.1 `unpack`

- 只收 `--zip`（real：`results/test04/test04_rank_results.zip`，檔名帶 ` (1)` 時照 TESTDAY 第 4 步的做法處理，可以直接用 `--zip` 指定）。
- **成員檢查**（`tb._member_ok(name, "sw")` 或同等邏輯）：沒有絕對路徑、`..`、反斜線；全部在 `sw/` 底下；沒有重複；目標檔都不存在（含 `unpack_sw.json`）；必要成員齊全（第 4.5 節的清單）。
- **紀錄檢查**（依這個順序；標「→1」的是接縫類失敗，結束碼 1，訊息「這次的 04 輸出不用：照 TESTDAY_04『重跑一次』先 archive」；其餘結束碼 2）：
  - `run_log` 的最後狀態是 `seam_failed`、`readonly_changed`、`final_check_failed`，或最後一次之前有這三種（`t2.FAILED_RUN_STATUSES`；同一個 Drive 資料夾裡又跑了一次，不是登記的重跑）→1（real 與 repo 內彩排；repo 外只警告）。最後狀態是 `preflight_failed`、`interrupted`、`running` 或 null → 2（那次沒有產生接縫結果；照 COLAB_RUN 改名 `sw_interrupted_N` 後整份重跑）。只有 `preflight_failed`、`interrupted`、`running` 在前面時照常。
  - **不接續**：每個 `_DONE_*.json` 的 `run` 都等於最後一次執行的編號（斷線後一律整份重算，第 5.4 節）；不是 → 1。
  - `config.smoke`／`rehearsal` 符合模式（real 兩個都是 false）→ 2。
  - `config.test_bundle.manifest_sha256` 等於本機 `T/MANIFEST.json`（`tb.verify_manifest(T, names=tb.COMP_FILES["rank"])`）；`files` 裡兩個競賽二檔名的 SHA 等於 MANIFEST → 2。
  - `config.test_bundle` 的 `manifest_sha256` 和 `files` 等於 test02 已 commit 的 `D2/rank/config.json` 的同名欄位：「和 11/04 用的是同一份測試包」→ 2。
  - `config.notebook_sha256` 等於已 commit 的 `colab/test04_rank.ipynb` 用 `t2.notebook_source_sha256` 重算的值（real →1；彩排只警告）。
  - **R20 先**（real）：`results/official_scores.json` 裡 R20 U1（第 3.0 節第 6 條那一筆）的 `uploaded_at` 早於 `run_log.runs[0].start`，也早於每個 `_DONE_*.json` 的 `time`（都是有時區的 ISO 時間，照 aware datetime 比）。不是 → 2，訊息「5.3：R20 上傳前跑的 test04 不用；照『重跑一次』的做法 archive 後重跑，這次不算登記的那一次」（`archive` 記成 `before_r20_u1` 類，不占重跑次數）。
  - **是不是重算的結果**（real；`results/test04/failed/attempt_*/attempt.json` 存在時）：zip 的 SHA-256 不等於任何一次封存記的 `zip_sha256`；`run_log.runs[0].start` 晚於最近一次封存的 `archived_at`；每個 `seam_val/dumps/_DONE_*.json` 的 `time` 晚於最近一次封存記的同名標記時間（`attempt.json` 的 `done_times`，有記才比）。任一不成立 → 2，訊息「這不是重算的結果（Drive 上可能還是舊的 zip）：刪掉這個 zip，請使用者等新的 zip 出現再下載」。
  - **重跑次數**（real）：`failed/` 裡已有 1 個「占次數」的封存（第 3.7 節）時，這一次就是登記的重跑；印出「這是登記的重跑（第 2 次、也是最後一次）」。已有 2 個以上占次數的封存 → 2（不應該發生，`archive` 會擋）。
- 解壓照 `tb.cmd_unpack`（先寫 `.tmp` 再 `os.replace`），寫 `R4/unpack_sw.json`：`format "c10-test04 unpack v1"`、zip 的路徑、SHA-256、大小、成員數、`runs`（每次的狀態與 `start`）、`earlier_failed_runs`、smoke、rehearsal、manifest SHA、`notebook_sha256`、`r20_first_upload {id, uploaded_at}`、`attempt_number`（1 或 2）、`freshness`（上面三項比對的值）、`tool`。
- 接縫類（→1）或 R20 先（→2）被拒時不解壓；`archive` 直接讀 zip 裡的 `sw/run_log.json`、`sw/seam_check.json`、`sw/config.json` 判斷失敗類別。
- 印出每一次執行的狀態，最後印「下一步：commit results/test04/sw 和 unpack_sw.json，再跑 apply」。
- 已存在的目標一律拒絕（沒有 `--rerun`；要重來先 `archive`）。
- 只為了查看失敗的 zip：`--dry-run --results-dir <repo 外的 scratch> --allow-failed`（狀態不是 `completed` 也解開，但不寫進 repo；同 `tb.cmd_unpack` 的 `--allow-failed`）。

### 3.2 `opp`（CPU；不需要 notebook）

**輸入**

- 計數：`cpath, crec = po.registered_counts(None, None)`（`count_report.json` 已 commit、`counts_sha256 == f6c6debe90639585763b9812c235743a6b7f89d8db1c6422ece7bccd550a08a5`）。
- 表：`tabs, info = po.build_tables(po.load_counts(cpath))`；`po.tables_record(tabs)["tables_sha256"]` 必須等於 `frozen_config.json` 的 `source_sha256.O4.tables_sha256`（`f380e0d720edd4429e4858ab87a3e1a5efbbd9de84982c002ab214f19f120de5`），不同就結束碼 4。
- 題目檔：seam_val＝`data/local/rank_val_questions.csv`（SHA `73e51bbfd7b051e73cbb62af4b180c403d2422f122240c5e67e9c92303776db5`，等於 `val_report.json` 的 `questions_file.sha256`）；public、private＝`T/rank_prediction_test_{public,private}.csv`（正規化副本；絕不讀 `T/raw`）。
- key：`vk[s] = rf.load_val_keys(D2/rank/<s>/dumps)`，`po.slots_match_keys(po.read_question_slots(qfile), vk[s])`。

**計算**

1. 三組的 slot 串成一次 `po.val_scores(slots_all, tabs, workers=W, tmp_root=<repo 外暫存>)`（W 預設 6，範圍 1–8；每盤獨立，串接不影響數值，見 `understand_p04.md` 第 2 節）。腳本要有 `if __name__ == "__main__":`（spawn）。
2. 依組切開，照 `po.cmd_val` 的陣列順序和 dtype 存成 `R4/opp/o4_<set>.npz`（`po.save_npz`）：
   `question_id`（`np.array(vk["question_id"])`）、`slot`（`np.array(vk["slot"], dtype=np.int64)`）、`color`、`o4`、`t4`（float64 `[N,10]`）、`n_moves`（int64 `[N,2,3]`）、`anomalies`、`moves_parsed`（int64 `[N]`）、`replay_failed`（bool `[N]`）。
3. **seam_val 逐位元組相同**（第 5.1 節）：`sha256(o4_seam_val.npz) == "30d312c02c357f6761567099003885be5e08635b6e6c0b49d71baac137cd9cd4"`。另外逐陣列和 `results/prereg04/opp/val_o4.npz` 比（dtype、shape、`tobytes()`），結果記錄。
   - 檔案不同、9 個陣列全同：環境漂移（numpy 或 zlib）→ 結束碼 4，訊息「還原 .venv 後重跑 opp」。
   - 陣列不同：結束碼 4，查程式。
4. **重播失敗**（第 7 節）：每組 `failures = replay_failed.sum()`、`slots = len(vk["question_id"])`、`rate = failures / slots`。三組（含 seam_val）都照登記 5.2 的字面：`rate > 0.005` → 04 不用（結束碼 3）。seam_val 另外必須逐位元組相同（第 3 步），所以 seam_val 有 1 盤以上、但 ≤ 0.5% 失敗時，結果是結束碼 4（本機漂移），不是 3。
5. **O4 的 Δκ**（第 6 節）：每組 `mask = parsed & ~replay_failed`，`lpO4 = rf.log_softmax(np.where(mask[:, None], o4, 0.0))`、`lpT4` 同理，`pf.val_side_kappa(dict(vk[s], parsed=mask), lpT4, lpO4)`。
   - seam_val 的 `dkappa` 必須等於 `results/prereg04/rank/apply_report.json` 的 `val_colour_check.O4.dkappa`（`0.1050265798301427`），差 > 1e-12 就結束碼 4。
   - public 或 private 的 O4 Δκ ≤ 0、NaN 或沒有子題：**這就是登記的判定**（`apply` 用同一個定義只會得到逐位相同的值），結束碼 3，不開 Colab。
   - 兩個都 > 0：只是「可能可用」；最終判定在 `apply`，`apply` 算出的 O4 值必須和這裡記的逐位相同（不同 → 結束碼 4）。
6. **只記錄**：各組 `anomalies > 0` 的盤數、`moves_parsed == 0` 的盤數、沒有對手手數的盤數、含失敗盤的題數、目標方與對手方的總手數。

**輸出** `R4/opp/opp_report.json`（第 3.4 節）。

**判定與結束碼**

| 情況 | `passed` | `usable_so_far` | 結束碼 | 下一步 |
|---|---|---|---|---|
| seam_val 逐位元組相同、seam_val Δκ 相同、三組失敗率都在範圍內、public 和 private 的 O4 Δκ 都 > 0 | true | true | 0 | commit `results/test04/opp/`，請使用者跑 test04 notebook |
| 任一組（含 seam_val）失敗率 > 0.5%，或 public／private 的 O4 Δκ ≤ 0、NaN、沒有子題 | true／false（失敗率超過時 false） | false | 3 | commit；04 不用，**不開 Colab**，最終 R20；寫 RESULTS |
| seam_val 不逐位元組相同（失敗 ≤ 0.5%），或 seam_val Δκ 不同，或 `.venv` 版本不同 | false | false | 4 | 停下查本機，不請使用者做任何事 |

- 判定的先後：先看結束碼 3 的條件，再看 4（seam_val 失敗 > 0.5% 時一定也不逐位元組相同，照登記字面算 3）。
- `passed`＝seam_val 的兩項核對都過，而且三組失敗率都在範圍內（登記 5.2 對 `opp` 的要求）。
- **重跑 `opp`**（`--rerun --reason`）：
  - 已有的 `opp_report.json` 是結束碼 4（`passed false`、`usable_so_far false`、`unusable_reasons` 只有漂移類）：可以重跑（查好本機之後）。
  - 已有的報告是登記的判定（結束碼 3，或 `usable_so_far true`）：拒絕（結束碼 2），除非同時給 `--bug-revision N`（第 3.0 節的核對；PREREG_04 rev 3 第 20 條）。CPU 重播是確定的，同樣的程式和輸入只會得到同樣的結果；要翻盤只能先寫修訂指出程式錯誤。

### 3.3 `apply`

**額外前提**（real；全部通過才開始算）

1. `R4/opp/opp_report.json` 已 commit、乾淨、`mode == "real"`、`passed is True`、`usable_so_far is True`（否則拒絕：「opp 已判定 04 不用」，結束碼 3）。三個 npz 的 SHA 等於報告記的；報告記的 frozen_config、counts、tables、`p04_opp.py`、三組題目檔與 test02 key 檔的 SHA 等於現在的。
2. `R4/unpack_sw.json` 和 `R4/sw/**` 全部已 commit、乾淨。`unpack_sw.json` 記的 zip SHA、`attempt_number`、`freshness` 仍和 `failed/` 的現況一致（`archive` 之後沒有重新 unpack 就拒絕）。
3. **notebook 紀錄**（`sw/`）：
   - `run_log` 狀態 `completed`，前面沒有失敗的執行（real）；每個 `_DONE_` 的 `run` 等於最後一次（不接續）；`seam_check.passed is True`；`tol` 是數字。不成立 → 1（`unpack` 應該已經擋下；這裡是第二道）。
   - config 的 smoke／rehearsal 符合模式；`notebook_sha256` 等於已 commit 的 `colab/test04_rank.ipynb` 重算值（real 拒絕、彩排警告、演練只記錄）；`colab/test04_rank.ipynb` 已 commit 且乾淨（real）。
   - `config.test_bundle` 等於 test02 的 `D2/rank/config.json` 的 `test_bundle`（`manifest_sha256`、`files` 的兩個競賽二檔）；`T` 的測試檔與 MANIFEST 等於它（`t2.check_test_files` 的邏輯，對 sw 的 config 再跑一次）。
   - 各組 `_INPUT.json` 的 `set`、`sha256` 等於本機題目檔；`config.sets.<set>.questions_sha256` 同。
   - `config.weights.{a,b}.sha256` 等於 `pf.SW_WEIGHTS`（A `790b767c5425e262130d6c103c71e89238edb17f927fa48e2a2770db788e50ac`、B `8e88f6ec6cbaad5372e5d62607b99ece3d77dc7a20af86a9197553f1a6c59a42`）。
4. **dump**（每組、每臂；**依這個順序核對**，第一個不成立的決定結束碼）：
   1. **key 一致**：sw 各組的 `val_keys.npz` 四個陣列和 test02 同一組的 `val_keys.npz` 逐陣列相同（dtype、shape、值都相同；兩邊都由同一段 `write_keys` 寫出）。不同 → 1（接縫失敗，可以用登記的那一次重跑）。
   2. **同一個環境**：3 組 × 2 臂 × {d4, swap} 共 12 個標記，加上 3 個 key 標記，`gpu_name`／`torch`／`cudnn` 全部相同。不同 → 1。
   3. **完成標記**：每個標記的 `sha256`、`bytes` 等於檔案；`rows == n`。不同 → 1（代表這次 notebook 的輸出不可信，和 notebook 自己的收尾核對 `final_check_failed` 同類）。
   4. **內容**：`d4 = rf.load_dump(<dumps>/val_p03x_d4.npz, n, 8)`、`sw = p3.load_swap(<dumps>/val_p03x_s0_swap.npz, n)`；npz 的 `weights_sha256` 等於上面的釘死值（不同 → 1）；swap 的 `view == "s0_colour_swapped"`（不同 → 1）；NaN 列等於 `~parsed`（`rf.check_val_logits` 對 d4；swap 同樣檢查；不同 → 1）。
5. **notebook 接縫本機重算**（第 5.2 節）：用 sw 的 seam_val dump 和已 commit 的 `results/prereg03/rank_{a,b}/dumps/` 重算兩臂的 C0（s0、swap）最大差。**算法必須和 notebook 相同**：從已 commit 的 `colab/test04_rank.ipynb` 取出 `c0_of`（和 `_argmax_first` 等它用到的函式）的原始碼 exec 進一個只有 `np`、`torch` 的命名空間，用 torch CPU float32 算（`test_t04_rank` 用 ast 核對取出的原始碼和 T02 的 `c0_of` 逐字相同；來源 notebook 是模組常數 `C0_SOURCE_NB`，I2 的 notebook 完成前，I1 的單元測試可以 `setattr` 指到 `colab/test02_rank.ipynb`，兩者的 `c0_of` 由 `test_t04_nb` 核對逐字相同）。本機的通過與否必須和 `seam_check.json` 相同，tol 必須等於本機依標記算的 tol。不同 → 1。本機值和 notebook 值的差只記錄（`notebook_seam_local_recompute`）。
6. `results/prereg04/rank/{rank_val_probs.csv, pred_B.csv, pred_A.csv}` 的 LF SHA 等於 `3c7ab658dc558ce8c6aefdd7b0747de1a60de575d151e9eee96ca6775f442e5c`、`27b6d225d0afc8db9183ae970ee19946ff819ec5ccb2be6a1816ef4338efff79`、`6a6bb3c2e63761bc7196661c99999cec499fae1b38bbb3a7bb9d512e26487739`。

**凍結參數**（照 `pf.cmd_apply` :1712–1734 的檢查）：`fc["format"]` 正確；`smoke`、`preview`、`design_input` 都是 false；`np.array_equal(fc["basis"], pf.B)`；`combo` 是 Q02、`terms == list(pf.TERMS)`；`B @ beta` 和 `b` 差 ≤ 1e-12；`theta[6:] == beta`。`theta` 共 15 個數 `[a, γ, w_OP, w_O4, w_SWA, w_SWB, β1..β9]`。

**每一組的計算**（seam_val、public、private 同一段程式；逐步照 `understand_p04.md` 第 3.2 節，該節的配方已在 scratch 逐位元組重現 `3c7ab658…`）

1. 題目檔對齊、C5 載入、開局對齊三段，**逐行照抄** `t2.apply_set`（:698–704、:706–717、:725–738）；不能 import `apply_set`（它回傳 R20 的機率）。
2. C5：`rf.load_dump(<D2>/rank/<s>/dumps/val_p02e9_d4.npz, n, 8)`，`weights_sha256` 等於 `fc["source_sha256"]["C5"]["weights_sha256"]`（`423581b4…`）；`rf.check_val_logits`；`logits = np.where(parsed[:,None,None], d["logits"], 0.0).astype(np.float32)`；`lp5 = p3.combo_logprobs(("C5",), {"p02e9_d4": logits})`。
3. 題目層：`cnn = [[i for i in rows if parsed[i]] for _, rows in groups]`、`S = rf.pool_matrix(cnn, n_rows)`、`n = np.array([len(c) for c in cnn], float)`、`m = rf.pooled(S, lp5, "log")`、`ok = n > 0`。
4. OP：`opening_test.npz` 取 `set == s` 的列，依 `(question_id, slot)` 對齊、執色相同，得 `o`、`has`；`E_OP = rf.pool_matrix([[i for i in rows if has[i]] …], n_rows) @ o`。
5. O4：讀 `R4/opp/o4_<s>.npz`，`question_id`、`slot`、`color` 和 vk 逐列相同；`mask = parsed & ~replay_failed`；`lpO4 = rf.log_softmax(np.where(mask[:,None], o4, 0.0))`；`S_O4 = rf.pool_matrix([[i for i in rows if mask[i]] …], n_rows)`；`E_O4 = rf.pooled(S_O4, lpO4, "log")`。（`mask == parsed` 時 `S_O4` 就是 `S`，和驗證時相同。）
6. SWA／SWB：`lp = rf.log_softmax(np.where(parsed[:,None], sw, 0.0).astype(np.float64))`（先在 float32 上遮罩再轉 float64，順序同 `pf.val_arm` :965）；`E = rf.pooled(S, lp, "log")`。
7. 合併：`Es` 依 `fc["combo"]["terms"]` 的順序 `[E_OP, E_O4, E_SWA, E_SWB]`；`z, _ = pf.gen_logits(theta, m[ok], [E[ok] for E in Es], n[ok])`；`P[ok] = rf.softmax(z)`，其餘 NaN。
8. 寫檔：表頭 `rf.PROB_COLUMNS`；每列 `[q, len(rows), int(n[k])] + [repr(float(x)) for x in P[k]]`，沒有可解析棋的題填 10 格空字串；用 `t2.write_probs(path, rows)`（LF、先寫 tmp、寫完用 `decide.read_probs` 嚴格讀一次）。
9. κ 的輸入（第 6 節）：A0／B0＝`d4[:, 0, :]`，`lpA0 = rf.log_softmax(np.where(parsed[:,None], d4[:,0,:], 0.0).astype(np.float64))`；`lpT4` 同第 5 步的遮罩。

**順序與輸出**

1. **seam_val 先**：寫 `R4/rank/seam_val/rank_val_probs.csv`，`decide.run(seam_csv, seam_dir)`，再 `sc = t2.seam_compare(seam_csv, seam_dir/pred_B.csv, prereg04 rank_val_probs.csv, prereg04 pred_B.csv, tol, smoke)`（第 5.3 節）。
   - 兩個 notebook 的 seam_val dump 都逐位元組相同（test02 `seam_check.bitwise_equal`、test04 `seam_check.bitwise_equal`）而輸出不逐位元組相同：記警告（同 `t2` :948–951）。
   - 同時在 seam_val 上算三個來源的 Δκ：O4 必須等於 `opp_report` 記的 seam_val 值（逐位）；SWA、SWB 只記錄，並記和 `results/prereg04/rank/check_report.json` 的 `sources.SWA.colour.val_sub_questions`、`SWB` 同欄位的差。
   - 寫 `R4/rank/seam_val/seam_local.json`（第 3.5 節）。
   - 沒過：印「本機接縫沒過：這次的 04 不用、不上傳，走 docs/TESTDAY_04.md『重跑一次』」，**不寫任何測試集輸出、不寫 `t04_apply_report.json`**，寫 `attempt_verdict.json`，結束碼 1。
2. **public、private**：算每題機率和 6 個 Δκ（第 6 節），並核對 O4 的 Δκ 等於 `opp_report` 記的（逐位；不同 → 結束碼 4）。
3. **判定** `usable`＝本機接縫過、notebook 接縫過、三組失敗率在範圍內、seam_val 逐位元組相同、6 個 Δκ 都 > 0、`check_submission` 通過（最後一項在寫交件檔之後才知道）。
   - Δκ 或失敗率不成立：**只寫 `t04_apply_report.json`**（`usable false`、`unusable_reasons`、Δκ 表），不寫 `rank_test_probs.csv`、`pred_*.csv`、`submission_rank.csv`，結束碼 3。
   - 都成立：`rank_test_probs.csv`（public 列在前、private 在後，各依題目檔順序；800 列）→ `decide.run` 寫 `pred_A.csv`、`pred_B.csv`、`decide_summary.json` → `submission_rank.csv`＝`pred_B.csv` 的逐位元組複本（`t2.atomic_copy` 後比對位元組）→ `check_submission.check("rank", sub, [pub_q, priv_q], official=True)`（smoke：照 `t2` 的做法用題目檔的前幾題、非 official）→ `t04_apply_report.json`，結束碼 0。
   - n＝0 的題由 `decide` 填 `"D"`（`decide.FALLBACK`），題號記進報告。
   - 都成立時另外記 `rule_b_differences_vs_r20`：這份 `submission_rank.csv` 和 R20 的 `results/test02/rank/submission_rank.csv`（只讀）在 800 題裡等級不同的題數（`official_log` 的訊息 A 會印出來，讓使用者和 Claude 知道兩份確實不同）。

**`attempt_verdict.json`**（`R4/rank/`；`apply` 以結束碼 1 或 4 結束時一定寫，**前提階段就停下的也寫**，例如 key 不同、環境混用、完成標記不符、tol 不符）：`format "c10-test04 attempt verdict v1"`、`mode`、`created`、`exit_code`、`stage`（`prerequisites`／`notebook_seam_recompute`／`local_seam`／`o4_consistency`）、`reason`、`details`、`sw_sha256`（`unpack_sw.json` 記的 zip SHA）、`tool`。結束碼 2、3 不寫（2 沒有算任何東西；3 寫的是 `t04_apply_report.json`）。

**重跑 `apply`**（`--rerun --reason`）：
- 前一次是結束碼 4（`attempt_verdict.json` 的 `exit_code == 4`）：查好本機後可以重跑。
- 前一次是結束碼 1：拒絕（結束碼 2），訊息「接縫類失敗要 archive 後重跑 notebook，不是重跑 apply」；除非 `--bug-revision N`。
- 已有 `t04_apply_report.json`（結束碼 0 或 3，已是登記的判定）：拒絕，除非 `--bug-revision N`（第 3.0 節；rev 3 第 20 條）。

### 3.4 `R4/opp/opp_report.json`

```
format "c10-test04 opp v1", mode, created, seconds, workers, tool, environment, git_head, rerun
passed, usable_so_far, unusable_reasons [..], problems [..]
frozen_config {path, sha256 (ae728e9c…), combo_id}
counts {sha256 (f6c6debe…), count_report_sha256}, tables_sha256 (f380e0d7…)
p04_opp_sha256 (b594e0ef…), equal_to_val_report_tool
test02 {apply_test_report_sha256, t1_verdict, r20_first_upload_id (real) , keys_sha256 {set: ..}}
sets {
  seam_val|public|private: {questions: path, questions_sha256, questions_count, slots, failures, rate,
                            max_rate 0.005, rate_ok, anomalies, slots_without_moves,
                            slots_without_opponent_moves, questions_with_failed_slots, moves {target, opponent},
                            output: o4_<set>.npz, output_sha256}
}
seam_val {file_sha256_equal: bool, reference "results/prereg04/opp/val_o4.npz", reference_sha256 (30d312c0…),
          arrays_equal {name: bool}, dkappa_O4, dkappa_O4_reference 0.1050265798301427, dkappa_equal: bool}
o4_dkappa_early {public: {dkappa, sub_questions, by_colour, ...}, private: {...},
                 note: "the registered decision is apply's (PREREG_04 rev 3 11.x); apply must reproduce these bits"}
```

### 3.5 `R4/rank/seam_val/seam_local.json`

- `t2.seam_compare` 的全部欄位（`passed`、`questions`、`max_abs_diff`、`tol`、`rule_b_differences`、`rule_b_max`、`byte_identical_probs`、`byte_identical_pred_B`）。
- `mode`、`created`、`required`（smoke 為 false）、`warnings`、`reference_sha256`、`outputs_sha256`（seam 的四個檔）。
- `tol_check`：`{"tol", "c5": {"test02_seam_tol", "seam_dump_gpu", "prereg02_dump_gpu"}, "sw": {"notebook_tol", "recomputed_tol", "seam_dump_gpu", "prereg03_dump_gpu"}, "rule": "max of the two"}`。
- `dkappa_seam_val`：O4（等於 opp）、SWA、SWB（只記錄，含和 prereg04 記錄值的差）。
- `notebook_seam_local_recompute`：兩臂 C0 的本機重算值、和 notebook 記的值的差（只記錄）、通過與否（必須和 notebook 相同）。

### 3.6 `R4/rank/t04_apply_report.json`（`official_log` 依賴的欄位用粗體）

```
**format** "c10-test04 rank apply v1", **mode** "real"|"rehearsal"|"dry-run"|"smoke", created, seconds
**combo** {"id": "Q02", "sources": ["C5","OP","O4","SWA","SWB"], "terms": [...], "path": "gen"} (frozen_config 原樣)
params (frozen_config 原樣), theta
**notebook** {"seam_check_passed", "seam_check_tol", "seam_check_bitwise_equal", "run_log_status",
              "earlier_failed_runs": [], "gpu", "torch", "cudnn", "config_sha256", "notebook_sha256",
              "same_as_repo_notebook", "weights_sha256": {"a","b"}, "test_bundle_equal_to_test02": true,
              "local_recompute_passed", "keys_equal_to_test02": {set: bool}, "env_same": true}
**seam_local** {"passed", "tol", "max_abs_diff", "rule_b_differences", "byte_identical_probs",
                "byte_identical_pred_B", "warnings", "file": "seam_val/seam_local.json", "sha256"}
**replay** {"seam_val": {"slots", "failures", "equal_to_committed_val_o4": bool}, "public": {"slots","failures"},
            "private": {...}, "max_rate": 0.005, "opp_report_sha256"}
**colour_check** {"public": {"O4": {"dkappa", "sub_questions", "questions_without_sub_question",
                                    "by_colour", "per_k_record_only", "question_bootstrap_ci95_recorded_only"},
                             "SWA": {...}, "SWB": {...}},
                  "private": {...}, "seam_val": {...record / consistency...},
                  "definition": "PREREG_04 rev 3 section 11 (dkappa = kappa(reference) - kappa(source) ...)"}
**usable** bool, **unusable_reasons** [..]
**check_submission** {"mode": "official", "questions": [...], "errors": []}   (usable false 時是 null)
**outputs_sha256** {"rank_test_probs.csv", "pred_A.csv", "pred_B.csv", "decide_summary.json", "submission_rank.csv"}
questions_without_parsed_game {set: [...]}, missing_fill "D", games_parsed {set: n}, decide_summary
level_summary {set: {rule_b_distribution, mean_probability_per_level}}  (只記錄)
inputs_sha256 {"frozen_config.json", "opp_report.json", "o4_<set>.npz", "unpack_sw.json", "sw/config.json",
               "sw/seam_check.json", "<set>/val_p03x_*.npz", "test02 apply_test_report.json", "opening_test.npz",
               "t1.json", "<set> val_keys (test02)", "val_p02e9_d4 (test02)", "prereg04 rank_val_probs.csv", ...}
**test02** {"fallback_used": false, "t1_verdict", "r20_first_upload_id"}
**inputs_bound** {<repo 相對路徑>: sha256, ...}   (official_log 逐項重算；.csv/.json 用 CRLF→LF 的 SHA，.npz 用原始位元組)
    必含：results/test04/opp/opp_report.json、results/test04/unpack_sw.json、results/test04/sw/config.json、
          results/test04/sw/seam_check.json、results/test04/rank/seam_val/seam_local.json、
          results/test02/rank/apply_test_report.json、results/test02/rank/opening_test.npz、
          results/test02/checks/rank/t1.json、results/test02/rank/<set>/dumps/val_keys.npz（3 組）、
          results/test02/rank/<set>/dumps/val_p02e9_d4.npz（3 組）、results/prereg04/rank/frozen_config.json
**attempt** {"number": 1|2, "counted_archives": [...], "bug_revisions": [...]}
rule_b_differences_vs_r20 (只記錄；official_log 訊息 A 用)
checks, rerun, paths, tool, environment, git_head
```

- `submission_rank.csv` 只在 `usable` 為真時寫。`official_log` 不信任 `usable`，會用數字自己重算條件（第 9.2 節），也會核對 `inputs_bound` 的每個檔現在的 SHA 沒變、`notebook` 的布林欄位都是 true、`earlier_failed_runs` 是空的。

### 3.7 `status`、`archive`

- **`status`**（唯讀，real 或給 DIRS）：印出
  - R20：test02 的 T1、`apply_test_report` 是否已 commit、R20 第一次上傳的紀錄編號；
  - opp：有沒有、是否 commit、`passed`、`usable_so_far`、三組失敗率、O4 的提早 Δκ；
  - sw：有沒有、是否 commit、`run_log` 狀態、notebook 接縫、GPU；
  - apply：`usable`、Δκ 表、交件檔 SHA 前 12 碼；
  - `failed/` 裡有幾次嘗試、K6 倒數。
  不寫任何檔。
- **`archive --reason TEXT [--bug-revision N]`**（real）：只封存**接縫類**的失敗；登記的判定（04 不用）不能靠封存重來。
  - **拒絕**（結束碼 2）的情況，除非給了通過核對的 `--bug-revision N`（第 3.0 節）：
    - `results/test04/rank/t04_apply_report.json` 存在（`usable` 真或假都代表這次已經有登記的判定，不是接縫失敗）。
    - `results/test04/opp/opp_report.json` 的 `usable_so_far` 是 false（04 已判定不用，沒有東西要重跑）。
    - 找不到接縫類的證據（下一點）。
    - `failed/` 裡已經有 1 個「占次數」的封存：訊息「登記只准重跑一次：停下查清楚；要再跑只能先寫 PREREG_04 修訂」。
  - **失敗類別**（工具自己判斷，記進 `attempt.json` 的 `failure_class`；讀已解開的 `sw/`，沒有的話直接讀 zip 裡的 `sw/*.json`）：
    - 占次數（登記的「重跑一次」）：`notebook_run_log`（最後狀態或之前有 `seam_failed`／`readonly_changed`／`final_check_failed`）、`notebook_seam`（`seam_check.passed` 不是 true）、`not_committed_notebook`（real 下 `notebook_sha256` 和已 commit 的不同）、`resumed`（`_DONE_` 的 `run` 不全等於最後一次）、`apply_exit_1`（`rank/attempt_verdict.json` 的 `exit_code == 1`）。
    - 不占次數：`before_r20_u1`（`unpack` 的「R20 先」檢查不過）、`bug_revision`（給了 `--bug-revision N`；rev 3 第 20 條「這不算第 9 條那一次重跑，但要記錄」）。
  - `N` = `failed/` 裡已有的 `attempt_*` 數＋1（編號連續，和類別無關）。
  - 把 `results/test04/sw/`、`unpack_sw.json`、`rank/`（有的話）移到 `results/test04/failed/attempt_N/`；`opp/` 留在原處（CPU 的結果和 notebook 無關）。
  - `results/test04/` 底下**所有** `*.zip`（含 `test04_rank_results (1).zip` 這類）移到 `data/test04_failed/attempt_N/`（gitignore），檔名保留。
  - 寫 `failed/attempt_N/attempt.json`：`reason`、`failure_class`、`counts_toward_rerun`、`bug_revision`（有的話：N 和它的 commit）、`archived_at`、每個 zip 的 SHA-256、`run_log` 各次的狀態與 `start`、`seam_check.passed`、`seam_local.passed`、`done_times`（`seam_val/dumps/_DONE_*.json` 的 `time`，給下一次 `unpack` 比新舊）、`gpu`。
  - 印出要 commit 的路徑，和要給使用者的指示：「Drive 的 `C10/results/test04/sw` 改名成 `sw_failed_N`，`test04_rank_results.zip` 改名成 `test04_rank_results_failed_N.zip`；電腦『下載』裡的舊 zip 刪掉」。
  - 什麼都沒有可以封存時拒絕。

### 3.8 `tools/t04_fake_tests.py`

```
PY tools/t04_fake_tests.py from-val --root S [--zip-out Z] [--rerun]
```

- 前提：S 已經跑過 `t02_fake_tests.py from-val --root S`（有 `S/FAKE_TREE.json`、`S/data/test`、`S/results/test02/rank/config.json`）。
- 產生一個 zip（預設 `S/drive/test04_rank_results.zip`），成員從 `sw/` 起，內容照第 2.3 節：
  - seam_val：`results/prereg03/rank_a/dumps/{val_keys.npz, val_p03a_d4.npz, val_p03a_s0_swap.npz}`、`rank_b/dumps/{val_p03b_d4.npz, val_p03b_s0_swap.npz}` 和它們的 `_DONE_` 標記，**逐位元組複製**。
  - public、private：題目 `tf.RANK_SLICES`（第 1–400、401–800 題）對應的列切片，`question_id` 用 `tf._rekey_array` 換前綴（`tf.PREFIX`），用 `tf._save_npz` 寫；標記用 `tf._marker` 的做法（環境欄位照抄 prereg03 的標記）。
  - `_INPUT.json`：seam_val 用 `data/local/rank_val_questions.csv` 的 SHA，public、private 用 `S/data/test` 的檔。
  - `config.json`：`smoke false`、`rehearsal true`、`kind "rehearsal"`；`gpu`／`torch`／`cudnn` 取 prereg03 標記；`weights` 是釘死值；`test_bundle` 從 `S/results/test02/rank/config.json` 原樣複製；`notebook_sha256` 用已 commit 的 `colab/test04_rank.ipynb` 重算（檔案還不存在時填 64 個 0，演練模式只記錄）。
  - `run_log.json`：一次 `completed`。`seam_check.json`：`passed true`、`tol 1e-4`、`bitwise_equal true`、差都是 0（copy 本來就相同；`apply` 會自己重算）。
  - `S/FAKE_TREE_T04.json`：`{"rule": "validation games: only replay and apply; never run content statistics", ...}`。
- 不寫 repo 裡的任何東西；S 必須在 repo 外。

## 4. test04 notebook（`colab/build_test04_rank_nb.py` → `colab/test04_rank.ipynb`）

### 4.1 產生方式

- **兩個底稿**，都讀已 commit 的 notebook JSON，先核對 LF SHA-256：
  - `T02 = colab/test02_rank.ipynb`：`53337a7acc396bdc1f94b3cbdc7dd0ce859766511099e8004697ff0d6a487b05`（測試日骨架；在 R20 指紋裡，只讀）。
  - `P03 = colab/prereg03_rank_a.ipynb`：`85e0c2ae3269c68ad166310efb2525b2d48ed1279c9cf6eb6b428e59e8693557`（模型與 dump）。另外核對 `colab/prereg03_rank_b.ipynb`（`aa3669accceca23fc3ab2439bcab68204281bcec999cd1e00d43dd16825bd8b3`）和 A 只差 `ARM = ` 那一行。
- 依 cell metadata 的 `c10` key 挑格子（不看位置；P03 的 key 在 `c10.base` 那一層，照 `build_prereg03_rank_nb.py:1908-1911` 的 `cell_key()` 規則）。
- 修改只用 `rep(src, old, new)`（`old` 剛好出現一次）和 `cut(src, a, b)`（兩個標記各一次）；每個改過的格子都列進 `CHANGE_ROWS`，列出的格子都要真的改過。
- **逐字核對**（ast `get_source_segment`；不符就停）：
  - `view_d4` 整格等於 P03 的 `view_d4` 格。
  - P03 片段：`load_net`、`dump_val_p03`、`channels_last`、`_nchw_output`、`state_dict_sha256`。
  - T02 片段：`write_dump`、`dump_is_done`、`write_keys`、`require_gpu_for`、`pack_results`、`flush_drive`、`sgf_to_features`、`parse_questions`、`quick_check_questions`、`question_spans`、`c0_of`、`_argmax_first`、`rule_b`、`readonly_fingerprint`、`guard_write`、`_is_under`、`record_readonly_after`（註解可以不同，程式本體相同）。
  - `prepare` 格整格等於 T02 的；三組格子只差第一行 `SET = '…'`。
- **釘死值從 repo 讀出，再和產生器裡的常數比**（任一不符就停）：

| 值 | 來源檔 |
|---|---|
| `W_PIN = {'a': '790b767c…50ac', 'b': '8e88f6ec…9a42'}` | `results/prereg03/rank_{a,b}/config.json` 的 `final_sha256`＝`_DONE_val_p03x_*.json` 的 `weights_sha256`＝npz 的 `weights_sha256`＝`tools/p04_rank_fuse.py` 的 `SW_WEIGHTS`（文字解析，不 import） |
| 接縫參照 6 個 npz：`val_keys.npz` `055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c`（兩臂同一份）、`val_p03a_d4.npz` `1e4e5957442596358b087e5f4e69c1fcf9f22f2c7933cc2049066bfab3075ecd`、`val_p03a_s0_swap.npz` `b355c26bf68069feb0e48ee95c46bd211c2bdbc6b5a48416c8408a210aec3848`、`val_p03b_d4.npz` `bad2277dd20bf832c02213be781768ba9d8a304922693bee383a28aa3531a627`、`val_p03b_s0_swap.npz` `0cadff4bae36b2a41f02fddf4366a2081d9912ede4a3063c8b50fd28a2647527` | repo 檔當場算＝`_DONE_*.json` 的 `sha256`；swap 兩個另外等於 PREREG_04 1.3、`p04_rank_fuse.SW_SHA256` |
| 參照 GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`、torch `2.11.0+cu128`、cuDNN `91900` | `_DONE_val_p03{a,b}_d4.json` |
| TF32 `{'cudnn': True, 'matmul': False}` | `results/prereg03/rank_{a,b}/config.json` 的 `runs[0].training.tf32` |
| 打包檔 manifest（`BUNDLE_FILES_PIN`） | `results/prereg02/rank/env.json`（同 `build_prereg03_rank_nb.py:158`） |
| seam_val 題目檔 SHA `73e51bbf…` | 打包檔 manifest |

- **輸出**：`colab/test04_rank.ipynb`（metadata 沿用 T02；cell id `c10-rank-t04-NN`；cell metadata `c10 = {"base_nb": "test02"|"prereg03", "base_cell", "base_id", "base", "changed"}` 或 `{"new": key}`；`outputs == []`）、`results/test04/notebook_diff_rank.txt`（每格對自己的底稿格的 unified diff）。重跑產生器要逐位元組相同。CLI 同 T02：`[REPO] [--out-nb] [--out-diff]`。

### 4.2 格子

| # | key | 來源 | 內容 |
|---|---|---|---|
| 0 | header | 新 | 大標題「C10 測試日（test04）：競賽二 04（03 兩臂讀對手那一方）」；用途、三組、AI Pro、G4 優先 A100 備用、會停下的情況、輸出、接續、和底稿的差異表 |
| 1 | setup | T02 setup＋`rep` | `RESULT_ROOT = C10/results/test04`（彩排 `test04_rehearsal`、冒煙 `test04_smoke`），`RESULT_DIR = RESULT_ROOT/sw`；`READONLY_ROOTS` = `prereg01`、`prereg02`、`prereg03`，加上**只有** test02 的 `rank` 子資料夾：正式 `C10/results/test02/rank`、彩排 `C10/results/test02_rehearsal/rank`（冒煙用 `_smoke` 版）。**不放整個 `test02`**：競賽一的 `test02_player` 會一直寫 `test02/player/`（`build_test02_player_nb.py:270-272`），同時跑時會被誤判 `readonly_changed`；`NOTEBOOK_REL = 'colab/test04_rank.ipynb'` 和新的 `NOTEBOOK_SHA256` 行；config 的 `prereg`、`base_notebooks`；`run_log` 同 T02 |
| 2 | prepare | T02 逐字 | 不解開訓練資料；拒收答案檔的那一行是整份唯一含那個字的一行 |
| 3 | official_1 | 逐字 | import |
| 4 | environment | 逐字 | `env.json` |
| 5 | official_2 | T02＋`rep` | `IN_CHANNELS = 501`（正式執行 assert）；GPU 關卡照 T02 的 `GPU_MODELS_OK = ('RTX PRO 6000', 'A100')`，另補 03 的「查不到 GPU 型號」訊息（`GPU_MODEL_SHOWN`） |
| 6 | preflight_t04 | 新（段落照 T02 preflight） | 第 4.3 節 |
| 7 | view_d4 | P03 逐字 | `NUM_SYMS`、`d4`、`to_target_view`、`dump_input` 等和單元檢查 |
| 8 | model_tools | 新（片段逐字取自 P03） | 只有 `state_dict_sha256`、`_nchw_output`、`channels_last` 三個 def；不建模型、不建 optimizer |
| 9 | dump_tools | T02＋`rep` | 拿掉 T02 的 `NUM_SYMS`、`d4`（已在第 7 格）；T02 的 `load_net` 換成 P03 的 `load_net`；加入 P03 的 `dump_val_p03` **整個函式逐字**，再加轉接 `QUESTIONS_PATH = None` 和 `def parse_val(): return parse_questions(QUESTIONS_PATH)`；`NET = None` 改成 `NETS = {}`；`ARMS = ('a', 'b')`、`DUMP_NAMES = {'a': 'p03a', 'b': 'p03b'}`、`W_LABEL = {x: 'prereg03/rank_%s/ckpt/final.pth' % x}` |
| 10–12 | dump_set_{seam_val,public,private} | 新，三格只差第一行 | 第 4.4 節 |
| 13 | seam_check | 新（段落照 T02 seam） | 第 4.5 節 |
| 14 | final_check | T02 final 改成兩臂 | 第 4.5 節 |

### 4.3 preflight（全部在開始算之前；任一不過就 `SystemExit`，`run_log` 記 `preflight_failed`）

1. 兩臂 `final.pth` 存在，整檔 SHA 等於 `W_PIN`（冒煙：等於 `prereg03_smoke` 的 `_DONE_val_p03x_d4.json` 記的）。找不到或不符的訊息：「要用 AI Pro 帳號（Drive、Colab、授權三處都要是它），不要搬動、改名 `C10/results/prereg03` 的任何檔」。
2. 接縫參照：6 個 npz 的 SHA 釘死；npz 的 `weights_sha256` 等於 `W_PIN`；參照標記的 GPU 等於釘死值。
3. 打包檔 manifest 等於 `BUNDLE_FILES_PIN`。
4. 測試包：7 個成員的 SHA、大小等於 MANIFEST；`kind` 正式只收 `official`／`rehearsal`、冒煙只收 `smoke`；`quick_check_questions`（T02 逐字）。
5. **test02 已完成、用同一份測試包**：讀 `C10/results/test02/rank/config.json` 和 `run_log.json`（彩排讀 `test02_rehearsal`）。正式和彩排：兩個檔都要在、`run_log.status == 'completed'`、`test_bundle.manifest_sha256` 等於這次的 MANIFEST SHA，否則停：「先做完 test02；C10 裡只能有 11/04 那一份 `C10_test_v1.zip`」。冒煙：有就核對，沒有就記 null。
6. **上一次、重跑次數、不接續**：
   - `sw/seam_check.json` 沒過就停（T02 ④-0 的文字，路徑改成 `sw`、`sw_failed_N`，N 照 Claude 給的）；`sw/run_log` 前面有失敗狀態也停。
   - **不接續**：`sw/*/dumps/` 裡已經有任何 `_DONE_*.json`（上一次斷線或跑過）就停：「斷線後不接續：照 COLAB_RUN 把 `sw` 改名成 `sw_interrupted_N`（zip 同樣改名），開新的執行階段整份重跑」。
   - 掃 `RESULT_ROOT` 底下所有 `sw_*` 資料夾：自己的 `seam_check.json` 不是 passed，或 `run_log` 有 `seam_failed`／`readonly_changed`／`final_check_failed` 的，算一次「接縫類失敗」。改名成 `sw_interrupted_*` 的資料夾有這種紀錄就停：「這不是斷線，是接縫失敗；截圖給 Claude」。接縫類失敗的資料夾已有 2 個以上就停：「登記只准重跑一次」。（本機 `unpack`／`archive` 是最終把關，這裡只是提早擋。）
7. 各組 `_INPUT.json`（T02 ④）：已存在但不同就停。
8. Drive 剩 ≥ 0.5 GB。
9. **TF32**：記下這次的預設值（`torch.backends.cudnn.allow_tf32`、`torch.backends.cuda.matmul.allow_tf32`、`cudnn.benchmark`、`cudnn.deterministic`，有的話加 `torch.backends.cudnn.conv.fp32_precision`）。和 03 記的 `{'cudnn': True, 'matmul': False}` 相同就不動；不同才用舊 API 設成 03 的值（設不成就停）。`benchmark` 是 True 時設成 False。預設值和實際用的值都記進 `config.json`。
10. 只讀目錄的開頭指紋（T02 ⑦）；`config.json` 頂層記第 2.3 節的欄位。

### 4.4 各組的格子（第一行 `SET = '<set>'`，其餘三格相同）

```python
t0 = time.time()
_path = SET_QUESTIONS[SET]
DUMP_DIR = guard_write(os.path.join(RESULT_DIR, SET, 'dumps')); os.makedirs(DUMP_DIR, exist_ok=True)
if read_json(os.path.join(DUMP_DIR, '_INPUT.json')) != SET_INPUT[SET]: raise SystemExit(...)
QUESTIONS_PATH = _path
for _x in ARMS:
    if _x not in NETS:
        NETS[_x], _ = load_net(W_PATH[_x], expect_sha=WSHA[_x])     # 兩臂各載入一次
_V = parse_questions(_path)                                          # 一組解析一次，兩臂共用
assert parse_val() is _V, 'QUESTIONS_PATH 轉接錯了'                  # dump_val_p03 內部經 parse_val() 取的是同一份（QSETS 以路徑快取）
_st = {'keys': write_keys('val_keys', _V['keys'])}
for _x in ARMS:
    _st[_x] = dump_val_p03(NETS[_x], DUMP_NAMES[_x], W_LABEL[_x], WSHA[_x])   # 03 的函式，逐字
    # 呼叫後核對：這一臂兩個 _DONE_ 的 rows == len(_V['feats'])，不等就 SystemExit
SET_LOG[SET] = {'questions_path': QUESTIONS_PATH, 'rows': len(_V['feats']), ...}; del _V; QSETS.pop(_path, None); (cuda 時 torch.cuda.empty_cache()); save_config(dumps=SET_LOG); mark_time(...)
```

- 題目檔：seam_val＝打包檔的 `local/rank_val_questions.csv`（冒煙取前 12 題）；public、private＝`WORK/tests/rank_prediction_test_{public,private}.csv`。執色一律用題目檔的 `color_i`。
- **8 種對稱照算**（`dump_val_p03` 逐字的代價，G4 約多 1.8 分鐘）：本機只用 `d4[:, 0, :]`（A0／B0，只為了 κ）和 swap（SWA／SWB，合併項）。第 1–7 個對稱只用在接縫的「只記錄」欄位。
- 推論是 fp32、`torch.no_grad()`、沒有 autocast，存原始 logits（03 原樣）。
- `dump_is_done`（T02 逐字）保留，但 preflight 第 6 點不准在已有完成標記的資料夾裡開始，所以實際上不會跳過任何一組：斷線後一律整份重算（第 5.4 節）。

### 4.5 接縫、收尾、zip

- **接縫**（每臂 x ∈ {a, b}；新的＝`RESULT_DIR/seam_val/dumps/…`，參照＝`C10/results/prereg03/rank_x/dumps/…`）：

| 項目 | 定義 | 判定 |
|---|---|---|
| keys | `question_id`、`slot`、`color`、`parsed` 的 shape 相同且 `np.array_equal` | 必須相同 |
| C0（s0） | `c0_of(L_d4, spans, parsed)`（T02 逐字：每題可解析盤的 `softmax(L[ok, 0, :])` 的 float32 平均）新舊逐元素最大差 | ≤ tol |
| C0（swap） | `c0_of(L_swap[:, None, :], spans, parsed)` 的最大差 | ≤ tol |
| tol | 新的 `_DONE_val_p03x_d4.json` 的 `gpu_name` 等於參照標記的（而且不是 None）→ 1e-4，否則 1e-3；兩臂各算，兩臂一定相同（同一個環境） | — |
| 環境 | 3 組 × 2 臂 × {d4, swap} 的 12 個標記加 3 個 key 標記，`gpu_name`／`torch`／`cudnn` 全部相同 | 必須相同 |
| 權重 | 12 個標記和 npz 的 `weights_sha256` 依臂等於 `W_PIN` | 必須相同 |
| 只記錄 | logits 逐位元組相同（`tobytes()`）、檔案 SHA 相同、8 種對稱和 swap 的 logits 最大差、E（每題可解析盤 float64 log_softmax 平均）最大差、s0 C0 的規則 B 不同題數 | — |

  - `passed`＝兩臂的 keys、C0（s0）、C0（swap）、權重都成立，而且環境相同。冒煙：照 T02 的冒煙規則（參照是 `prereg03_smoke`）。
  - 沒過：記結尾指紋、`run_log` 狀態 `seam_failed`、打包、`flush_drive()`，`SystemExit`：「接縫核對沒過：這次執行的 04 測試輸出一律不用、不上傳；R20 不受影響。不要自己重跑，下載 zip、截圖給 Claude。」
- **收尾**（T02 final 改成兩臂）：每組核對 `val_keys` 和 4 個 logits 檔的完成標記、SHA、大小、形狀（d4 `float32 (rows, 8, 10)` 且 `syms == 8`；swap `float32 (rows, 10)` 且 `view == 's0_colour_swapped'`）、`weights_sha256`、NaN 列＝`~parsed`、`_INPUT.json`、題數；只讀目錄的結尾指紋（不同就 `readonly_changed`，照樣打包、停下）。
- **zip**：`RESULT_ROOT/test04_rank_results.zip`，成員從 `sw/` 起，跳過 `.pth`／`.pt`／`.tmp` 和 ≥ 100 MB 的檔。必要成員：7 個紀錄檔（`config`、`run_log`、`env`、`timings`、`seam_check`、`readonly_before`、`readonly_after`），每組 `val_keys.npz`、4 個 logits npz、5 個 `_DONE_`、`_INPUT.json`。最後 `flush_and_unmount`，印「Drive 已寫入完成並卸載」「完成。請下載 test04_rank_results.zip」。
- 大小：全部約 22 MB。

### 4.6 身分、GPU、帳號、時間

- `NOTEBOOK_SHA256`：T02 的規則（每格原始碼的 `json.dumps` 取 SHA-256，本身那一行換成 64 個 0；`t2.notebook_source_sha256` 可以重算）。
- GPU：只收 G4（`RTX PRO 6000`）或 A100（「大量 RAM」關）；其他型號、查不到型號、沒有 GPU，都在 official_2 停下。操作上 G4 優先。
- 帳號：只能在 AI Pro 跑，靠 preflight 第 1、2 點強制（權重只在那個帳號的 Drive）。
- 時間（推估，用 03 的實測）：G4 約 3.5–4.5 分鐘、約 0.6 單元；A100 約 5–6 分鐘。彩排時實測，寫進 COLAB_RUN。

## 5. 接縫與容許差（總表）

### 5.1 O4 的 seam_val（`opp`）

- 判定：`o4_seam_val.npz` 的**檔案** SHA-256＝`30d312c0…`（已實測做得到，zip 成員時間固定）。
- 逐陣列比對只當診斷。檔案不同、陣列相同＝環境漂移；陣列不同＝程式問題。兩種都是結束碼 4、停下查本機，不是 Colab 重跑的理由。
- 另外：seam_val 的 O4 Δκ＝`0.1050265798301427`（差 ≤ 1e-12）。

### 5.2 notebook 接縫（test04）

- 見第 4.5 節。本機 `apply` 用已 commit 的 prereg03 dump 重算一次，通過與否必須和 notebook 相同、tol 必須等於依標記算的值。

### 5.3 本機 Q02 接縫（`apply`）

- 比什麼：用 test02 的 seam_val C5、`opp` 的 seam_val O4、test04 的 seam_val SW、`opening_test.npz` 的 seam_val 開局，照第 3.3 節算出的 Q02 驗證題機率，和 `results/prereg04/rank/rank_val_probs.csv`、`pred_B.csv` 比。用 `t2.seam_compare`。
- 通過：題號和順序相同、`(n_games, n_parsed)` 相同、max|ΔP| ≤ tol、規則 B 最多差 4 題（`t2.SEAM_MAX_B_DIFF`）。
- tol：`max(tol_C5, tol_SW)`。
  - `tol_C5`：test02 已 commit 的 `apply_test_report.json` 的 `seam_local.tol`（1e-4 當且僅當 test02 seam_val C5 dump 的 GPU＝prereg02 dump 的 GPU）。
  - `tol_SW`：本機依 test04 標記重算的值（1e-4 當且僅當 test04 seam_val dump 的 GPU＝prereg03 dump 的 GPU），必須等於 notebook `seam_check.json` 的 `tol`。
  - 所以只有兩邊都在參照同型 GPU 上算時才是 1e-4。
- 只記錄：`byte_identical_probs`、`byte_identical_pred_B`；兩個 notebook 都逐位元組相同而本機不逐位元組相同時記警告。
- 冒煙：全部只記錄（`required false`）。

### 5.4 失敗後怎麼辦

| 失敗 | 類別 | 處理 |
|---|---|---|
| notebook `seam_check` 沒過；`apply` 的 notebook 接縫本機重算不一致；key 和 test02 不同；環境混用；完成標記不符；本機 Q02 接縫沒過；`run_log` 前面有失敗；接續過（`_DONE_` 的 `run` 不一致）；跑的不是已 commit 的 notebook | 接縫失敗（結束碼 1） | 登記的「重跑一次」：`archive`（工具核對失敗類別、次數上限 1）→ commit → 使用者把 Drive 的 `sw` 改名 `sw_failed_N`、zip 改名 `test04_rank_results_failed_N.zip` → 新的執行階段整份重跑。**重跑只在 G4**（第一次在 G4 或 A100 都一樣）：不換到容許差較寬的 A100，也避開 A100 沒實測過的差距；G4 拿不到就隔一段時間再試，最晚 11/15 中午，還拿不到就這一季不用 04（K6）。test04 自己有一次，和 test02 的分開算。`unpack` 核對新 zip 是重算的（SHA、執行開始時間、完成標記時間）。第二次還是沒過：停下查；`archive` 拒絕再封存；放寬容許差只能另寫 PREREG_04 修訂（rev 4），修訂前不上傳 04；K6 照樣適用 |
| seam_val O4 不逐位元組相同（失敗 ≤ 0.5%）；seam_val Δκ 不同；Δκ 和 opp 記的不同；`.venv` 版本不同 | 本機漂移（結束碼 4） | 停下查本機程式或 `.venv`；不動 Colab，不算重跑；查好後 `opp`／`apply --rerun --reason` |
| 前提不符、檔案缺、已 commit 的東西被改、zip 不是重算的結果、T1 FAIL_STRUCT | 拒絕（結束碼 2） | 照訊息修好再跑本機工具 |
| notebook 在 R20 的 U1 之前跑 | 拒絕（結束碼 2） | `archive`（類別 `before_r20_u1`，不占重跑次數）→ 照重跑的做法再跑一次 |
| 任一組（含 seam_val）重播失敗 > 0.5%；Δκ ≤ 0／NaN／沒有子題；T1 FAIL_OPENING；K6 已過 | 登記的「04 不用」（結束碼 3） | 最終 R20（FAIL_OPENING 時 R10），寫 RESULTS，不重跑；`archive`、`opp`／`apply --rerun` 都會拒絕（除非有指出程式錯誤的修訂） |
| notebook 在 preflight、GPU 關卡停下 | 不算一次執行 | 照 COLAB_RUN 處理後，在同一個資料夾再按「全部執行」（還沒有任何完成標記） |
| 中途斷線 | 不算一次執行（除非那個資料夾已有接縫類失敗的紀錄） | **不接續**：Drive 的 `sw` 改名 `sw_interrupted_N`（zip 有的話同樣改名），刪執行階段、開新的，整份重算（約 4–6 分鐘）；GPU 規則照被打斷的那一次（第一次：G4 優先、A100 備用；登記的重跑：只 G4）。TESTDAY_04 記下 N 和原因。notebook 的 preflight 第 6 點和本機 `unpack` 都會擋「在舊資料夾接續」 |

## 6. 執色檢查（Δκ）的定義與判定表

- **對象**：每組 s ∈ {public, private}，每個來源 src ∈ {O4, SWA, SWB}，共 6 個數。seam_val 上同樣算，只做一致性核對與記錄。
- **可用盤**：
  - SWA、SWB 和參照 A0、B0：`parsed` 為真的盤（test02 的 key；和 test04 的 key 已核對相同）。
  - O4 和參照 T4：`parsed & ~replay_failed`。
- **每盤向量**：float64 的 log_softmax。O4／T4 取 `o4`／`t4`；SWA／SWB 取 swap logits；A0／B0 取 `d4[:, 0, :]`。遮罩外的列先填 0 再算（不影響，因為不進子題）。
- **子題**：`p3.val_colour_subquestions(vk)`：每題的可用盤依 `color` 分成黑、白，≥ 2 盤的才算一個子題。官方每題單色，所以一題就是一個子題；萬一混了兩種執色也照這個規則拆開。
- **κ**：`p3.kappa_questions`（PREREG_03 2.2 第 4 條）：題內每一盤 g 算 `Σ_k softmax(mean_{h≠g} l_h)_k · l_{g,k}`，再取題內平均。
- **Δκ_s(src)**：`pf.val_side_kappa(vk_s, lp_ref, lp_src)["dkappa"]`，也就是所有子題 κ(參照) − κ(來源) 的簡單平均（點估計）；O4 用的 `vk_s` 把 `parsed` 換成遮罩。
- **判定**：

| 情況 | 結果 |
|---|---|
| 6 個 Δκ 都 > 0（嚴格大於） | 這一項通過 |
| 任一個 ≤ 0、是 NaN／None，或該組沒有任何子題 | 04 不用，最終 R20；不觸發重跑（這是題目內容的性質） |
| seam_val 的 O4 Δκ 和 `0.1050265798301427` 差 > 1e-12，或 `apply` 的 O4 Δκ 和 `opp` 記的不逐位相同 | 本機漂移：停下查（結束碼 4），不是 04 的判定 |

- **只記錄**：黑題、白題各自的 Δκ；各 k（5／10／15／20）的平均；`question_bootstrap_ci95_recorded_only`（`p3.question_boot_mean`）；子題數；可用盤不到 2 盤而沒算到的題數；seam_val 上 SWA、SWB 的值和 `check_report.json` 記的 `+0.03631175839656169`、`+0.034356617152967346` 的差。
- **雜訊**：每池只有 400 題，「其實是正、因抽樣而 ≤ 0」有可能發生；這只會造成改交 R20，不會讓錯的 04 上場。**登記和本規格都不引用數字**：`understand_p04.md` 4.3 的「≤ 2%」是用驗證題內容（沒有答案）算的，違反「驗證玩家的棋不做統計」，已撤回，只在 rev 3 第 22 條揭露（它沒有影響任何規則或門檻）。實作者不必另估；要估只能用 inner 的棋，而且只寫進 `build_review.md` 當揭露。

## 7. 重播失敗的定義

- **單位**：盤（slot）。
- **失敗**：`po.val_scores` 的 `replay_failed`＝`(moves_parsed == 0) | (anomalies > 0)`，也就是沒有解析到任何一手，或有一手下在已有子的點（PREREG_04 10.1 d）。
- **分母**：該組題目檔（正規化副本）的全部盤數，也就是 key 檔的列數（每題 slot 1..`num_games`）。官方每組 5,000 盤，0.5%＝25 盤。
- **分組**：seam_val、public、private 各自算。
- **判定**：
  - 三組（seam_val、public、private）一樣：`failures / slots > 0.005` 就不用 04（剛好 0.5% 算過；登記 5.2「任一組」的字面）。
  - seam_val 另外必須逐位元組相同（第 5.1 節），已 commit 的驗證 O4 是 0 盤失敗，所以 seam_val 有 1 盤以上、但 ≤ 0.5% 失敗時是本機漂移（結束碼 4）。
- **失敗盤的用法**：不進 E_O4（第 3.3 節第 5 步的遮罩），不進 O4／T4 的 κ。溫度項的 n 仍是 C5 的可解析盤數；SWA、SWB 的 E 仍用 C5 的可解析盤。一題的 O4 遮罩是空的：E_O4＝0 向量。
- **只記錄**：各組 `anomalies`、沒有任何一手的盤數、沒有對手手數的盤數、含失敗盤的題數（題為單位的比例也記，但不判定）。
- 注意：失敗盤的 `o4`／`t4` 是有限值（0 向量或停下前的部分和），一定要用 `replay_failed` 遮掉，不能靠 NaN。

## 8. 演練（K5 的「本機假測試演練通過」）

### 8.1 R20 逐位元組不變

1. **golden（I3 第一件事，在寫任何 t04 程式之前）**：分支目前的工作樹，程式碼仍等於 `6ec5e68`（只多了未追蹤的 `results/test04/design/`）。設 `C10_SLOW=1`、`C10_KEEP=1`、`C10_DRYRUN_DIR=<repo 外的 scratch>`，跑 `PY -m pytest -q tests/test_t02_dryrun.py -k FromValPipeline`；它依序跑
   `t02_fake_tests.py from-val` → `t02_checks.py t1 --dry-run --only T1.1,T1.2,T1.3` → `t02_rank.py opening --dry-run` → `t02_rank.py apply --dry-run`（加上競賽一和 R10 的演練）。
   跑完先確認 `git status` 只有 `results/test04/`，再把確定性的輸出記進 `results/test04/design/r20_fake_golden.json`：
   - 整檔 SHA-256：`out/rank/{rank_test_probs.csv, pred_A.csv, pred_B.csv, submission_rank.csv}`、`out/rank/seam_val/{rank_val_probs.csv, pred_A.csv, pred_B.csv}`。
   - 兩份 `decide_summary.json`（`out/rank/`、`out/rank/seam_val/`）**不比整檔**：`decide.run` 把輸入的絕對路徑寫進 `"input"`（`tools/decide.py:278`），樹放的位置不同 SHA 就不同。改記「刪掉 `input` 鍵後 `json.dumps(obj, sort_keys=True, ensure_ascii=False)` 的 SHA-256」，並另外記 `pred_A_sha256`、`pred_B_sha256`、`input_sha256`、`distribution_A`、`distribution_B`、`n_missing` 等欄位的值（有哪些欄位照實際檔案）。
   - `opening_test.npz` 逐陣列的 SHA-256（`meta` 有時間戳，不比檔案）。
   - `provenance`（commit、指令、時間、Python 與套件版本、當時用的 `C10_DRYRUN_DIR`）。
   已知值：`submission_rank.csv` `d9b9801b3df85c75e5932ce228eebbeb8a76c43b40d2efd02700ce8807490d97`、seam 的 `a8370043…`、`863a6258…`。
2. **每次演練**：`test_t04_dryrun.py` 用分支上的程式重跑同一段，逐項等於 golden（`decide_summary.json` 用上面的正規化比法）。
3. **合併前（主流程）**：兩個一次性 worktree（repo 外），一個是合併當下的 main，一個是合併結果，`data/` 用 junction 指到原本的 `data/`（工具只讀它）；兩邊**先後用同一個 `C10_DRYRUN_DIR` 路徑**（跑完一邊、把輸出搬到別處，再跑另一邊）各跑第 1 點，比對全部輸出逐位元組相同（`decide_summary.json` 仍用正規化比法，另外記整檔是否也相同），也等於 golden；並跑 `git diff r20-testday-pre04 -- <39 個檔>`，除了 `tools/official_log.py`（只能是新增行）都是空的。紀錄寫進 `build_review.md`。
4. **`official_log.py` 的差異測試**（`tests/test_t04_official_log.py`，R20 路徑上唯一會改的檔）：用 `git show r20-testday-pre04:tools/official_log.py` 把 tag 版載成另一個模組（`importlib.util.spec_from_loader` ＋ `exec`，不寫進 repo），和新版在同一份 fixture、同樣的 mock（`REPO`、`git_file_state`、時間）下各跑一遍 R20 的 `prepare`（U1）→ `uploaded` → `public` → `final` → `status`，斷言 log JSON 和 stdout 逐位元組相同；唯一例外是 `status` 多出的 K6 那一行（沒有 04 紀錄時新版不能多印別的）。結果記進 `build_review.md`，列進 rev 3 第 13 條的 K5 條件。

### 8.2 04 端到端 dry run（`test_t04_dryrun.py`，`C10_SLOW=1`）

- 在 8.1 的同一棵樹上：`t04_fake_tests.py from-val --root S` → `t04_rank.py unpack --dry-run --zip S/drive/test04_rank_results.zip --results-dir S/results/test04 …` → `opp --dry-run --workers 6` → `apply --dry-run`。
- 斷言：
  - `o4_seam_val.npz` 的 SHA＝`30d312c0…`；`o4_public.npz`、`o4_private.npz` 的 `o4`、`t4`、`n_moves`、`anomalies`、`moves_parsed`、`replay_failed` 等於 `val_o4.npz` 對應列（`question_id` 只差前綴）；三組失敗 0。
  - `rank/seam_val/` 的 `rank_val_probs.csv`、`pred_B.csv`、`pred_A.csv` 的 SHA 等於 `3c7ab658…`、`27b6d225…`、`6a6bb3c2…`；`seam_local.json` `passed`、`byte_identical_* true`、`warnings == []`。
  - `rank_test_probs.csv` 的 800 列等於 prereg04 `rank_val_probs.csv` 第 1–800 列換前綴；`submission_rank.csv` 等於 prereg04 `pred_B.csv` 前 800 列換前綴，並通過 `check_submission --official`。
  - seam_val 的 Δκ：O4 `0.1050265798301427`（逐位）；SWA、SWB 等於 `check_report.json` 記的值（假樹是逐位元組副本，應該逐位相同）。
  - 假 public、private 的 6 個 Δκ 都 > 0，值等於第一次綠燈寫進測試的完整值（四捨五入後要等於 U1 探針：public O4 0.0930、SWA 0.0321、SWB 0.0278；private 0.1000、0.0418、0.0364）。
  - `t04_apply_report.json` 的 `usable true`，第 3.6 節的粗體欄位齊全。
  - **快照**：t04 的每一步前後，`S/results/test02/**`、`S/out/rank/**`（t02 的輸出）、`S/data/test/**` 每個檔的 SHA 和檔案清單完全相同。
  - 假樹的規則：`S/FAKE_TREE_T04.json` 存在；沒有跑任何內容統計。

### 8.3 負面演練（同一個測試類別，複製假樹後改一處）

- 改 npz 的演練一律**重寫 npz**：成員名稱與其他成員不變（`logits`、`weights_sha256`、`view`／`syms`），只換要改的陣列；並更新對應 `_DONE_*.json` 的 `sha256` 和 `bytes`，讓它只在要測的那一關失敗（第 3.3 節第 4 點的核對順序）。改完重新打 zip（或直接改解開後的樹，照該列寫的階段）。

| 改哪裡 | 預期 |
|---|---|
| sw 的 public `_DONE_val_p03a_d4.json` 的 `gpu_name` 改掉 | `apply` 結束碼 1（環境混用），`attempt_verdict.json` 的 `stage == "prerequisites"`，沒有任何測試集輸出 |
| sw 的 public `val_p03a_d4.npz` 改一個值、標記**不**更新 | `apply` 結束碼 1（完成標記不符） |
| sw 的 `seam_check.json` `passed false`（`run_log` 同時是 `seam_failed`） | `unpack` 結束碼 1；`archive` 記 `notebook_seam`、占次數 |
| sw 的 `run_log` 前面加一次 `seam_failed` | `unpack` 結束碼 1（repo 內）或警告（repo 外），`apply` 拒絕 |
| 一個 `_DONE_` 的 `run` 改成前一次 | `unpack` 結束碼 1（接續過） |
| public 的 `val_p03a_s0_swap.npz` 重寫成同組 `d4[:, 0, :]`（`view` 仍是 `s0_colour_swapped`，標記更新） | SWA 在 public 的 Δκ 恰好是 `0.0`（兩邊輸入逐位元組相同）→ `apply` 結束碼 3，只有 `t04_apply_report.json`（`usable false`），沒有 `submission_rank.csv`；接著 `archive` 拒絕、`apply --rerun` 拒絕 |
| sw 的 `config.test_bundle.manifest_sha256` 改掉 | `unpack` 結束碼 2 |
| sw 的 public `val_keys.npz` 的 `parsed` 改一列（標記更新） | `apply` 結束碼 1（key 和 test02 不同，第一關） |
| `opp` 重跑沒帶 `--rerun --reason` | 拒絕 |
| `opp` 判 3 之後 `--rerun --reason` 不帶 `--bug-revision` | 拒絕 |
| 同一份 zip 在 `archive` 之後再 `unpack` | 結束碼 2（這不是重算的結果） |
| 新 zip 的 seam_val `_DONE_` 時間等於封存的那一份 | 結束碼 2 |
| `results/test04/` 裡有 `test04_rank_results (1).zip` 時 `archive` | 兩個 zip 都搬到 `data/test04_failed/attempt_N/` |
| 已有 1 個占次數的封存，再 `archive` | 拒絕（「登記只准重跑一次」） |
| `run_log.runs[0].start` 早於 R20 U1 的 `uploaded_at`（只能在 real 路徑的單元測試用 mock 做） | `unpack` 結束碼 2；`archive` 記 `before_r20_u1`、不占次數 |
| `t04_rank.py` 的 `--results-dir` 指到 repo 內、`results/test02`、`results/prereg04` | 拒絕 |

### 8.4 `C10_SLOW=2`

- 完整測試用 `C10_SLOW=2` 跑一次（含 `test_t02_dryrun.py` 的 FromTrainChecks、FromTrainOfficialShape 和 `test_t04_dryrun.py`），全部通過。
- 只准用 ≤ 6 個 worker；開跑前跑 `python D:\AI-Portfolio\_tools\handoff.py` 看有沒有其他線在吃 CPU。
- **單獨跑**（K5 閘門第 1 輪加）：同一時段不跑別的重工作，包括同一個 workflow 裡其他 gate 代理的 L2、L2b 冒煙。開跑前用 `(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory`（KB）確認可用虛擬記憶體至少約 12 GB。9/30 晚間第一次跑時機器記憶體吃緊，8 個測試以 MemoryError 失敗（`TestRealSelftest` 把 72 萬盤的 fit 全部讀進 pytest 行程，之後的測試跟著失敗）。
- 出現 MemoryError 時：先單獨重跑失敗的測試，再重跑一次完整的 `C10_SLOW=2`，兩次的結果都寫進 `build_review.md`。測試程式不改，也不要改成記憶體不足就 skip（會把問題藏起來）。

### 8.5 notebook 的 CPU 冒煙（本機，不用 GPU；`OMP_NUM_THREADS=6`、`MKL_NUM_THREADS=6`）

- **L1**：`PY -m pytest -q tests/test_t04_nb.py`。
- **L2b（開發時，約 12–15 分鐘，必做）**：`PY colab/make_test04_smoke_tree.py --scratch S` 建替身樹：
  - `S/drive/C10_bundle_v1.zip`（`run_local_smoke.py` 會自己複製）；
  - `S/drive/results/prereg03_smoke/rank_{a,b}/ckpt/final.pth`：`torch.manual_seed(1)`／`(2)` 的隨機 `GoRankResNet(501)`；
  - `…/dumps/`：exec P03 notebook 的 view_d4、model_tools、dump_tools 片段（`multiprocessing.dummy` 取代 `Pool`），算前 12 題的 `val_keys`、`val_p03x_d4`、`val_p03x_s0_swap` 和 `_DONE_`；
  - 冒煙測試包：`t02_fake_tests.py from-train --root F --shape official --slot-colours filled` → `t02_test_bundle.py build --kind smoke --root F …`，zip 複製成 `S/drive/C10_test_v1.zip`；
  - `S/drive/results/test02_smoke/rank/config.json`（有的話讓 preflight 第 5 點也走到）。
  然後 `PY colab/run_local_smoke.py colab/test04_rank.ipynb --scratch S`。預期：兩臂 4 個 dump `bitwise_equal`、接縫通過、zip 成員齊全。
  另外核對 public、private 的轉接（`QUESTIONS_PATH`）：用 P03 notebook 原始碼 exec 出來的 `dump_val_p03`，直接對冒煙的 public、private 題目檔各算一次（CPU 上結果確定，12 題），要求和 test04 notebook 那一組的 4 個 npz 逐位元組相同；`config.dumps.<set>.questions_path` 是那一組的題目檔。
- **L2（合併前跑一次，約 50–100 分鐘，視同時段負載而定，必做）**：`run_local_smoke.py` 依序跑 `prereg01_rank.ipynb --fresh` → `prereg02_rank.ipynb` → `prereg03_rank_a.ipynb` → `prereg03_rank_b.ipynb` → 用 03 冒煙真正的 dump 當參照跑 test04 冒煙；同一個 scratch 接著跑 `t04_rank.py unpack/apply --smoke`（test02 的冒煙也要先跑，給 C5 和開局）。
  - 開跑前確認可用記憶體 ≥ 8 GB、沒有別的重 CPU 工作（K5 閘門第 1 輪加）。
  - 某一步沒有 Python traceback 就結束（Git Bash 的 exit 139、Windows 的 0xC0000005）時：把那本 notebook 在 `S/drive/results/<prereg>_smoke/<臂>` 和 `S/work/*pass_ckpt*_<臂>_smoke` 的輸出移開，從那一步重跑（鏈腳本的第一個參數），重跑要記進紀錄。9/30 晚間的 gate 在 prereg03 A 臂遇過一次（cell 22 final dumps，訓練已完成），移開後重跑通過，權重相同。
  - 跨次數比什麼（K5 閘門第 2 輪加）：prereg01、prereg02 的 CPU 冒煙權重每次跑都不一樣（`build_review.md` 第 7 節列三次的 SHA），所以 test02 冒煙輸出、t02 本機接縫、test04 本機 Q02 接縫、test04 冒煙交件檔 SHA、「和 R20 規則 B 不同幾題」都會跟著變，只記錄、不當參照。跨次數要相同的只有 prereg03 兩臂的 `final.pth`、`opp` 的三個 O4 檔 SHA 和 public、private 的 O4 Δκ；notebook 接縫（`bitwise_equal`）在同一條鏈內比。
- 紀錄（時間、結果、SHA）寫進 `build_review.md`。

### 8.6 Colab 彩排（要使用者同意；建議提早到第 10.6 節驗證和審查通過後，約 10 月中下旬）

- **時間**：這是 11/05 前 test04 唯一一次真 GPU 執行，排得越早，發現問題後越有時間重建、冒煙（L2 約 50–100 分鐘，要挑沒有別的重 CPU 工作的時段）、審查。建議在第 10.6 節完整驗證通過、notebook 審查完就問使用者（目標 10/15–10/22）；10/28–11/02 留作第二次（有改動時）。
- **順序**（和 COLAB_RUN test02 第 21 步、`docs/TESTDAY.md` 的「彩排一結束就收拾」衝突，所以 TESTDAY_04 和 COLAB_RUN 的 test04 彩排小節要明寫）：
  1. test02 彩排照 COLAB_RUN「測試集（test02）」跑，**test02 兩份（rank、player）都跑完才開始 test04**（test04 的只讀根目錄只含 `test02_rehearsal/rank`，但仍不要和 player 同時跑，也省得兩份搶 Drive）。
  2. **test02 的第 21 步先不做**：test04 彩排要讀同一份彩排的 `C10_test_v1.zip`。Claude 在 test02 彩排開始前的訊息裡就先講。
  3. 用同一份彩排 zip 在 G4 跑 `test04_rank.ipynb`，結果在 `C10/results/test04_rehearsal/`。
  4. 三份都跑完，才把 `test02_rank.ipynb`、`test02_player.ipynb`、`test04_rank.ipynb`、`C10_test_v1.zip` 四個檔一起收進 `old_rehearsal`，並刪掉「下載」裡三個彩排的 zip。
- **本機**（D2、O2、T、C 照 `docs/TESTDAY.md` 彩排那一段的實際路徑）：
  ```
  PY tools/t04_rank.py unpack --rehearsal --zip "<S>\downloads\test04_rank_results.zip" --results-dir results/test04_rehearsal --t02-results-dir "<S>\rehearsal" --t02-out-dir "<S>\local_rank" --tests-dir "<S>\data\test" --checks-dir "<S>\results\test02\checks"
  PY tools/t04_rank.py opp   --rehearsal <同樣的 DIRS>
  PY tools/t04_rank.py apply --rehearsal <同樣的 DIRS>
  ```
  （彩排的第 3.0 節第 5 條是「`mode == "rehearsal"`、不要求 commit」。）
- **預期**：G4 上兩臂 `bitwise_equal`、本機 Q02 接縫逐位元組相同；A100 只要求接縫通過。
- **要記的**：Colab 的 torch、cuDNN 版本（和參照 `2.11.0+cu128`、`91900` 比），兩臂 C0（s0、swap）的最大差，實測時間與單元。G4 上不是逐位元組相同、而且版本和參照不同時，在 `build_review.md` 記下離 1e-4 還有多少空間；最大差超過 1e-5（離門檻不到 10 倍）時，先告訴使用者，在 11/04 前（還沒有任何測試資料）決定要不要寫 rev 4，不能等到測試日才處理。
- 彩排不上傳、不 commit 輸出；結果記進 `build_review.md`（K5 前）或 `results/test04/RESULTS.md`。彩排後改了產生器就重建、重審、commit，在 K5 前合併。

## 9. 上傳程序與 `official_log.py`

### 9.1 測試日順序（工具強制的部分標 ★）

1. 11/04：照 `docs/TESTDAY.md` 做完 test02，R20 的 U1 上傳、`official_log.py uploaded`、commit。當天不做 04。
2. ★ `t04_rank.py opp`（需要第 1 步的紀錄）→ commit `results/test04/opp/`。結束碼 3：04 不用，不開 Colab，寫 RESULTS，最終 R20。
3. 使用者跑 test04 notebook（COLAB_RUN 新一節）→ zip 放進 `results\test04\` → ★ `unpack` → commit。
4. ★ `apply` → commit `results/test04/rank/`。
5. ★ `official_log.py prepare --comp rank --file results/test04/rank/submission_rank.csv --reason "prereg04 Q02 first upload (PREREG_04 5.3)"` → commit → 訊息 A（檢查結果＋OK 請求）→ 使用者說 OK → 訊息 B（只有上傳步驟）→ `uploaded`、`public` → commit。
6. ★ `official_log.py final`（第 9.3 節），11/23 23:59 前。**競賽二在 04 判定之前（`opp`／`apply` 判定、或 K6 過了）不要執行 `final --comp rank`**；`docs/TESTDAY.md` U5 的「兩場都是 prereg02（甲）」對競賽二由 PREREG_04 5.3 和 TESTDAY_04 第 7 步取代（TESTDAY.md 在指紋裡不能改，所以寫在 TESTDAY_04 開頭和 repo 的 `CLAUDE.md`）。已經照 U5 設過 R20 也沒關係：第 7 步會重跑 `final`，`status` 會提示。11/22、11/23 各跑一次 `status`。

### 9.2 `tools/official_log.py` 的附加式修改

- **起點**：`understand_protocol.md` 第 3 節的插入塊 A–L（已在 scratch 套進副本並通過 `test_t02_infra` 的 OfficialLog 11 項；scratch 腳本 `…\scratchpad\p04testday\U4\apply_blocks.py`、`repo\tests\trial_t04_log.py`，不保證留存）。
- **和該節不同的地方**（因為本規格的版面）：
  - `opp_report.json` 的位置是 `results/test04/opp/opp_report.json`（不是 `rank/opp/`）。
  - 另外綁 `results/test04/sw/seam_check.json`：已 commit、`passed is True`，SHA 記進 `reports`。
  - `seam_local.json` 在 `results/test04/rank/seam_val/seam_local.json`（和該節相同）。
  - `opp_report.json` 的條件：`mode == "real"`、`passed is True`、`usable_so_far is True`。
- **規則**（違反任何一條測試就失敗）：
  - 只插入新行，不改、刪、移動任何舊行（含縮排、尾逗號）；插入點照 tag 版行號，在 docstring、常數、`check_reports` 開頭、`cmd_prepare`、`cmd_final`、`cmd_status`、parser。
  - 只用標準函式庫；不 import `t04_rank`；新程式、註解、字串都不能出現答案檔名的那個字。
  - 新路徑在函式裡用 `REPO` 組；新的 git 呼叫包成 `git_commit_time(rel)`（測試可以 mock）。
  - 不改 `RANK_COMBOS`、`FILE_NAMES`、`REPORT_OF`、`DEFAULT_METHOD`、`method_for`、`check_t2_t5`、`check_reports` 的 R20 部分。**不照 `results/prereg04/design/plan.md` 第 7 節的「`RANK_COMBOS` 加項」**：改那一行附加檢查就失敗，另寫一行重新定義又會改變 R20 的 `method_for` 行為；04 的方法名稱在 `method_for` 之後另外設定。這是 plan 的偏離，寫進 rev 3。
- **行為**：
  - 04 的檔以路徑認：`results/test04/rank/` 底下的 `submission_rank.csv`（`is_t04_file`），綁在 `t04_apply_report.json`（不叫 `apply_test_report.json`：萬一辨識失敗，R20 的路徑會因為報告不存在而拒絕，不會把 04 當成 R20）。
  - `check_reports_t04`：報告格式、real mode、combo Q02、notebook 接縫、本機接縫、`check_submission` official、交件檔 SHA；**用報告的數字重算** 6 個 Δκ > 0 和三組失敗率 ≤ 0.005、seam_val 逐位元組相同；`seam_local.json`、`opp_report.json`、`sw/seam_check.json` 已 commit 且通過；test02 的 `apply_test_report.json` 是 real、`fallback_used false`、接縫通過；完整 raw T1 是 PASS 或 NORMALIZED；報告最後一次 commit 的時間 ≤ K6（2026-11-16 23:59:59 +08:00）。另外：
    - **輸入沒變**：報告 `inputs_bound` 的每個檔，現在已 commit 的 SHA 都等於報告記的（R20 在 U1 之後依 U4(a) 修 bug、重跑 test02 的 opening 或 apply，04 就不再符合，理由「輸入已變更，重跑 apply」）。
    - **notebook 欄位**：`seam_check_passed`、`same_as_repo_notebook`、`test_bundle_equal_to_test02`、`local_recompute_passed`、`env_same` 都是 true，`keys_equal_to_test02` 每組都是 true，`earlier_failed_runs` 是空的。
    - **封存歷史**：讀 `results/test04/failed/attempt_*/attempt.json`：占次數的封存 ≥ 2 個，或有任何封存裡有 `t04_apply_report.json` 而 `attempt.json` 沒有記一個已 commit 的 `bug_revision`，就不符合。
  - `prepare`：
    - 04 的檔要先有 R20 的 U1（`status uploaded`，不是 04 的第一筆競賽二上傳）；方法預設 `T04_METHOD`，`--method` 必須寫到 `Q02`；上傳紀錄多一個 `test04` 欄（Δκ、失敗率、`k6`、`r20_first_upload_id`、報告 SHA、`rule_b_differences_vs_r20`）；OK 請求多印「這一份是 04，不是 11/04 那份 R20；兩份在 800 題裡有 N 題的規則 B 等級不同」。
    - **非 04 的競賽二檔**（R20、R10）：log 裡有任何一筆 04 紀錄**現在符合 5.3**（`t04_entry_eligible`）時拒絕，除非 `--not-04-reason`（和 `final` 同一套核對，見第 9.3 節）。log 裡沒有 04 紀錄時完全照舊（R20 的 U1 不受影響）。
  - `final`：log 裡沒有 04 紀錄時行為和現在逐位相同；有 04 紀錄時照第 9.3 節。指定的紀錄還是 `prepared`（沒上傳）時多印一行「這一筆還沒上傳」（新行，舊行不動）。
  - `public`：新增只附加的警告行：一筆 04 紀錄的 public 分數和任何一次 R20 上傳的分數完全相同（|Δ| < 1e-9）時，印「可能傳成 R20 的檔（兩份同名）：核對 AIdea 的上傳紀錄」；反方向也一樣。
  - `status`：一定多一行 K6 倒數。有 04 紀錄時另外：
    - 一行 04 的狀態（符合／不符合和理由）。
    - 有符合 5.3 的 04，但 `final_choice.rank` 不是 04 或沒有 `prereg04_5_3`：印「違反 PREREG_04 5.3：照 docs/TESTDAY_04.md 第 7 步重跑 final」。
    - 最後一次上傳是一筆現在不符合的 04，或最終選擇是 04 而現在不符合：印「照 PREREG_04 5.3：final 改 R20，再 prepare 並重傳 R20」。
- **R20 的行為要逐位相同**：沒有 04 紀錄時，`prepare`／`uploaded`／`public`／`final` 的輸出和 JSON 都不變。

### 9.3 最後一版（PREREG_04 5.3）

- 一筆 04 上傳紀錄「現在符合 5.3」（`t04_entry_eligible`）：H4R 採用（固定事實，引 `results/prereg04/RESULTS.md`、`e8b24f2`）；prepare 記的報告 commit 仍是報告的最後一次 commit，而且時間 ≤ K6；T1 PASS 或 NORMALIZED；6 個 Δκ 都 > 0；三組失敗率都 ≤ 0.005；報告和交件檔的 SHA 沒變。
- `final --id <04>`：現在不符合就拒絕。
- `final --id <R20 或 R10>`：有現在符合的 04 紀錄時拒絕，除非 `--not-04-reason TEXT`，而且：
  - 文字裡要寫出「PREREG_04 rev N」（N ≥ 4）；
  - 工具核對 HEAD 上已 commit、乾淨的 `docs/PREREG_04_OPPONENT.md` 有 `| rev N |` 那一列，加進那一列的 commit 晚於每一筆被推翻的 04 紀錄的 `prepared_at`（第 3.0 節「登記修訂的核對」同一個定義；`official_log` 自己寫一份，只用標準函式庫，git 呼叫包成可 mock 的函式）；
  - 那份修訂依 rev 3 第 20 條指出具體的程式錯誤和它破壞的是 5.3 的哪一條（內容由人審，工具只核對存在與時間）。
  - rev N、它的 commit 和時間、原因記進 `final_choice.rank.prereg04_5_3`，也寫進 RESULTS。只有自由文字、沒有修訂時一律拒絕。
- 04 符合 5.3 全部條件時，**沒有其他方法**改交 R20（public 分數較低不是理由；VALIDATION 紀律 4）。
- 已經傳過 04、之後才發現不能用：
  - 不符合是工具算出來的（例如輸入變了、報告被改）：這時已經沒有符合的 04，直接 `final --id <R20 的 U1>`，再 `prepare` 一次 test02 的交件檔、取得 OK、重傳，讓它成為最後一次上傳。
  - 是找到程式錯誤：先 commit PREREG_04 修訂（rev 3 第 20 條），再 `final --id <R20 的 U1> --not-04-reason "PREREG_04 rev N: …"`，同樣重傳。
- public 分數不進判定；比上一版低 0.05 以上時只查錯誤（`public` 會標 investigate），查不到 bug 就維持 5.3 的結果。
- 11/23 23:59 前完成；`status` 要顯示最後一次上傳的 SHA 等於最終選擇。競賽二每天最多 5 次，R20 和 04 合計。

### 9.4 `docs/TESTDAY_04.md`（Claude 的步驟；新檔）

- 大綱照 `understand_protocol.md` 4.1，改成本規格的版面和指令：第 1 步閘門（`official_log.py status`、T1、`git status`、`counts_fit.npz` SHA、`.venv` 版本、`PY -m pytest -q tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py`）→ 第 2 步 `t04_rank.py opp` 和結束碼的後果表 → 第 3 步給使用者跑 notebook（核對上傳副本 SHA 的 PowerShell 一行，先自己跑過）→ 第 4 步 `unpack` 與「(1)」檔名 → 「重跑一次」（`archive`、請使用者改 `sw` → `sw_failed_N`、確認新的完成標記時間較晚）→ 第 5 步 `apply` 和結束碼的後果表 → 第 6 步上傳（訊息 A／B，`explorer.exe "D:\AI-Portfolio\competitions\aicup2026-go\results\test04\rank"`，SHA 前 12 碼，TEAM_11248）→ 第 7 步最後一版 → 第 8 步 `results/test04/RESULTS.md`（另一個代理核對數字）、地圖、`CLAUDE.md`。
- 另有：K5 核對表（第 10.6 節）、K6 與日期（`apply` 要在 11/15 前 commit；重跑的 G4 最晚 11/15 中午）、D5（官方定義變動，rev 3 第 19 條）、彩排（第 8.6 節的順序，含「test02 第 21 步先不做」和四個檔一起收）。
- 開頭寫明：
  - test02 照 `docs/TESTDAY.md`（不改）；和登記不一致時以登記為準。
  - **競賽二的最終選擇**：TESTDAY.md U5 那一行「兩場都是 prereg02（甲）」對競賽二由 PREREG_04 5.3 和本檔第 7 步取代；04 判定前（`opp`／`apply` 判定或 K6 過了之前）不要執行 `final --comp rank`。
  - 每段做完就 commit，**不論結束碼**（`opp_report.json`、`attempt_verdict.json`、`t04_apply_report.json` 都要 commit；作者 kuotunyu，不加 Co-Authored-By）；`data/**` 和 zip 不 commit。
  - 結束碼 3（04 不用）之後不重跑任何步驟；要翻盤只能先寫修訂（rev 3 第 20 條）。
- 「重跑一次」一節：`archive` 會自己判斷失敗類別和次數；請使用者同時改名 `sw` 和 zip；只在 G4 重跑；新 zip 出現前不要下載（`unpack` 會擋舊 zip）。
- 「斷線」一節：不接續；`sw` → `sw_interrupted_N`；不算一次執行；記下 N 和原因。
- 第 7 步：11/22、11/23 各跑一次 `official_log.py status`，看到「違反 PREREG_04 5.3」或「final 改 R20」的提示就照做。

### 9.5 `docs/COLAB_RUN.md` 檔尾新一節「# 測試集 04（test04）（使用者操作步驟）」

- 其他節一個字都不改（只在檔尾加）。大綱照 `understand_protocol.md` 4.2，改成本規格的版面：
  - 開頭框：Claude 說「可以跑 test04 了」之後才用（R20 已上傳、`opp` 已確認 04 可能用得上，通常 11/05–11/07）；1 份 notebook、只推論、約 5 分鐘（推估）、約 0.7 單元。
  - 帳號只能 AI Pro；`C10` 裡該看到的東西；電腦不睡眠。
  - A. 放 notebook（`explorer.exe "D:\AI-Portfolio\competitions\aicup2026-go\data\colab_upload\test04"`，只有 1 個檔）。
  - 開跑前：**競賽一的 `test02_player` 還在跑、或還在等 G4 時，先不要跑 test04**；先刪掉 player 的執行階段（或等它跑完）。
  - B. 跑：確認頭像、檔名、第一格標題；可用量 ≥ 3；G4，灰色時 A100（大量 RAM 關）——**這只適用第一次；Claude 說「重跑」時只用 G4**，灰色就隔一段時間再試，最晚 11/15 中午，還拿不到就告訴 Claude；「全部執行」；確認 GPU 那一行和「種類 official」。
  - C. 等（只看不按）。D. 跑完：看最後兩行 → 先看左上角檔名再刪執行階段 → 先刪「下載」裡的舊同名 zip → 到 Drive `C10/results/test04` 下載 `test04_rank_results.zip`（還沒看到就等幾分鐘，不要下載別的檔名）→ 拖進 `explorer.exe "D:\AI-Portfolio\competitions\aicup2026-go\results\test04"` → 說「放好了」和拿到哪種 GPU。
  - 上傳交件檔（04）：訊息 A／B 的說明；這次的檔在 `results\test04\rank\`，和 11/04 那份同名，一定從 Claude 給的指令打開的資料夾拖。
  - 要重跑時（只有 Claude 請你做時）：Drive `C10/results/test04` 裡的 `sw` 資料夾改名成 `sw_failed_N`，**同一層的 `test04_rank_results.zip` 也改名成 `test04_rank_results_failed_N.zip`**（改名前後各截圖一張）；刪掉電腦「下載」裡的舊 zip；刪執行階段、開新的，**只選 G4**，從 B 重做。
  - 中途斷線時：不要接著按「全部執行」。把 `sw` 改名成 `sw_interrupted_N`（有 zip 的話同樣改名成 `test04_rank_results_interrupted_N.zip`，N 照 Claude 給的），刪執行階段、開新的，從 B 整份重做（約 5 分鐘）；GPU 照這一次原本的規則。notebook 看到舊資料夾裡有做到一半的輸出會自己停下。
  - 出狀況時的表格；不要做的事（不要動 `C10/results/prereg03`、不要改 notebook、同一份不要同時開兩個分頁、不要重傳 `C10_test_v1.zip`、不要和 `test02_player` 同時跑）。
  - 彩排小節（只有 Claude 安排時）：第 8.6 節的順序；**彩排那天 test02 的第 21 步先不做**，test02 兩份和 test04 都跑完，才把 `test02_rank.ipynb`、`test02_player.ipynb`、`test04_rank.ipynb`、`C10_test_v1.zip` 四個檔一起收進 `old_rehearsal`，再刪「下載」裡三個彩排的 zip。

## 10. 要寫的測試與整合時的完整驗證

### 10.1 `tests/test_t04_rank.py`（I1）

- **配方**（快，讀已 commit 的驗證輸入，不開答案）：用第 3.3 節的每組計算函式吃 prereg02 的 C5、prereg04 的 `val_o4.npz`、prereg03 的 SW／A0／B0、`opening_val.npz`（對齊成 `opening_test` 的形狀），重算的 CSV 逐位元組等於 `3c7ab658…`；`val_side_kappa` 三個值逐位等於 `0.1050265798301427`、`0.03631175839656169`、`0.034356617152967346`。
- **O4 seam_val**（慢，`C10_SLOW=1`，約 1.5 分鐘）：`opp` 的核心在 seam_val 上重算，檔案 SHA＝`30d312c0…`；另測「三組串接」和「單組」的陣列逐位相同（取一段 2,000 盤）。
- **重播規則**：合成 `replay_failed`：25/5000 過、26/5000 不過；seam_val 有任何失敗盤就 `passed false`（這時逐位元組比對也一定不過，結束碼 4）；失敗盤不進 E_O4 和 κ（和手算比）；遮罩等於 `parsed` 時結果和不遮罩逐位相同。
- **Δκ 判定表**：6 個值的每一種不過（0、負、NaN、沒有子題）各一個 subTest。
- **拒絕與結束碼**：每一條第 3.0、3.1、3.3 節的前提各一個 subTest（用 `setattr` 把模組常數指到暫存樹），並斷言結束碼：real mode 路徑不對、輸出目錄在凍結資料夾、答案字樣、沒有 R20 的 U1、T1 FAIL_OPENING（結束碼 3，訊息含「04 不上場」）、T1 FAIL_STRUCT（2）、K6 已過（mock 時間，3）、test02 `fallback_used true`、`run_log` 有前次失敗（1）、接續過（1）、smoke／rehearsal 旗標不符、notebook 身分不符（real 1）、test_bundle 和 test02 不同、權重不符、第 3.3 節第 4 點的四關各自單獨失敗（都是 1，而且是那一關的 `stage`／`reason`）、tol 不符、`opp` 判不用時 `apply` 拒絕、`--rerun` 沒帶 `--reason`、`.venv` 的 numpy／scipy／Python 和 `val_report.json` 不同或 zlib 不等於 `ZLIB_PIN`（4）、notebook 在 R20 U1 之前跑（2）。
- **重播失敗的字面**：seam_val 失敗 26/5000 → 3；1/5000 → 4。
- **重跑規則**：`opp` 判 3 後 `--rerun` 沒有 `--bug-revision` 拒絕、有通過核對的修訂則可以；判 4 後 `--rerun --reason` 可以；`apply` 前一次 1 → `--rerun` 拒絕；前一次 4 → 可以；已有 `t04_apply_report.json` → 拒絕（除非修訂）。`--bug-revision` 的核對：修訂列不存在、N < 4、檔案沒 commit、修訂 commit 早於被推翻的紀錄，各一個 subTest（git 呼叫 mock）。
- **`attempt_verdict.json`**：每一種結束碼 1、4 都寫（含前提階段）；結束碼 2、3 不寫。
- **notebook 接縫本機重算**：取出的 `c0_of` 原始碼和 T02 逐字相同（ast）；用 torch CPU float32；和 notebook 記的值相同的合成案例會得到相同的通過與否。
- **輸出**：接縫沒過時沒有任何測試集輸出也沒有 `t04_apply_report.json`；`usable false` 時只有報告；`submission_rank.csv` 是 `pred_B.csv` 的逐位元組複本；`inputs_bound` 的鍵齊全、值等於當場算的 SHA；`rule_b_differences_vs_r20` 等於手算。
- **`unpack` 的新舊核對**：同一份 zip 在 `archive` 之後再 unpack → 2；`completed` 之後又 `completed` 而且標記時間相同 → 2；`run_log.runs[0].start` 不晚於 `archived_at` → 2。
- **`archive`**：搬移正確、`opp/` 留著、編號遞增、沒東西時拒絕；`results/test04/` 裡帶 ` (1)` 的 zip 也被搬走；每一種失敗類別判對（`notebook_run_log`、`notebook_seam`、`not_committed_notebook`、`resumed`、`apply_exit_1`、`before_r20_u1`、`bug_revision`），前五種占次數；有 `t04_apply_report.json`、`opp` 判不用、找不到證據、已有 1 個占次數的封存，各自拒絕；`attempt.json` 記了 `done_times`、`archived_at`、每個 zip 的 SHA。
- **`t04_fake_tests.py`**：seam_val 是逐位元組副本；public、private 列數與前綴正確。

### 10.2 `tests/test_t04_nb.py`（I2；照 `test_p03_nb.py`、`test_t02_nb_rank.py`）

- **Build**：兩個底稿 SHA；重建逐位元組相同（寫到 tmp）；cell id、`compile()`、`outputs == []`；答案字樣只出現在沿用的 prepare 拒收行；沒改的底稿格逐字相同；`CHANGE_ROWS` 和實際改動一致；`notebook_diff_rank.txt` 和重建的相同。
- **SegmentIdentity**：第 4.1 節的全部片段，測試裡獨立用 ast 重算（不信任產生器）。
- **Pins**：第 4.1 節表格的每個值等於 repo 檔當場算的；重算 `NOTEBOOK_SHA256`；上傳副本存在時 SHA 等於 repo 的 notebook。
- **NoTraining**：AST 名稱沒有 `optimizer`、`df_train`、`df_val`、`T_TOTAL`、`make_train_loader`、`load_rows`、`CKPT_DIR`；沒有 `torch.autocast`、`bfloat16`、`tarfile.open`；每個被讀的名稱都有綁定（同 `test_t02_nb_rank.py:264-281`）。
- **CellFunctions**（CPU torch）：小的隨機 `GoRankResNet(501)`，3 題 × 5 盤合成特徵：test04 原始碼 exec 出的 `dump_val_p03` 和 P03 原始碼 exec 出的，輸出 npz 逐位元組相同；swap＝`net(dump_input(x, ~ib, 0))`、s0＝`net(dump_input(x, ib, 0))`；`guard_write` 擋下只讀目錄；接縫函式：逐位元組相同過、同型 GPU 差 5e-5 過、同型差 2e-4 不過、不同型差 2e-4 過、key 不同／環境混用／權重不符不過、swap 的 C0 是對 `[n, 10]` 算的。
- **Preflight**：用假的 Drive 樹 exec preflight：權重不符、test02 config 缺或測試包不同、上一次接縫沒過、`sw` 裡已有完成標記（不接續）、`sw_interrupted_*` 裡有接縫失敗紀錄、接縫失敗的 `sw_*` 資料夾已有 2 個，都停下；只有 `sw_failed_1`（接縫失敗）時照常開始。
- **只讀根目錄**：`READONLY_ROOTS` 只含 `prereg01`、`prereg02`、`prereg03` 和 test02 的 `rank` 子資料夾；執行期間在 `test02/player/` 新增或修改檔案，結尾狀態仍是 `completed`。
- **轉接**：各組格子裡有 `assert parse_val() is _V`，呼叫後核對兩臂 `_DONE_` 的 `rows`，`SET_LOG` 記 `questions_path`；exec 時把 `QUESTIONS_PATH` 故意指錯，會停下。

### 10.3 `tests/test_t04_official_log.py`（I3）

- 照 `understand_protocol.md` 3.5 的 7 組（04 的 prepare 成功、每一條拒絕、`--method`、R20 路徑不變、`final`、`status`、子序列），路徑改成本規格的版面（第 9.2 節）。
- setup 比照 `test_t02_infra.py:975-1041`：暫存 repo、mock `ol.REPO`、`ol.git_file_state`、`ol.git_commit_time`。
- 一定要有：log 裡沒有 04 時 `final_choice.rank` 沒有 `prereg04_5_3`，和現在逐位相同；R20 的 prepare 在 `results/test04/rank/` 也有檔時照舊、沒有 `test04` 欄。
- **差異測試**（第 8.1 節第 4 點）：tag 版和新版在同一份 fixture 下跑 R20 的完整流程，log JSON 和 stdout 逐位元組相同（除了 K6 那一行）。
- **5.3 的強制**：
  - `final --id <R20>` 在有符合的 04 時：只有自由文字的 `--not-04-reason` → 拒絕；引用的 rev 不存在、N < 4、修訂 commit 早於 04 的 `prepared_at` → 拒絕；合格的修訂 → 成功，`prereg04_5_3` 記了 rev N 和 commit。
  - 有符合的 04 時 `prepare` R20 的檔 → 拒絕；帶合格修訂 → 成功；沒有 04 紀錄時 R20 的 prepare 照舊（R20 U1 路徑）。
  - `status`：有符合的 04 但最終不是 04 → 印「違反 PREREG_04 5.3」那一行；最後一次上傳是現在不符合的 04、或最終是現在不符合的 04 → 印「final 改 R20」那一行。
  - **輸入變了**：換掉 test02 的 `apply_test_report.json`（或 `opening_test.npz`）之後，04 就不再符合，理由含「輸入已變更」；`notebook` 欄位有一個 false、`earlier_failed_runs` 不空，也不符合。
  - **封存歷史**：`failed/` 有 2 個占次數的封存、或有封存含 `t04_apply_report.json` 而沒有修訂 → 不符合。
  - `public`：04 和 R20 的分數完全相同時印警告行；`final` 指定 `prepared` 紀錄時印「還沒上傳」那一行。

### 10.4 `tests/test_t04_hygiene.py`（I3）

1. 凍結工具的 LF SHA 等於 `frozen_config.json` 的 `tool.files_sha256`；`p04_opp.py` 等於 `val_report.json` 記的；都已 commit 且乾淨。
2. `results/prereg04/{opp,rank}/**`、`results/prereg04/*.md`：沒有修改、沒有新檔、沒有 bytecode（`git status --porcelain --ignored`）。
3. 和 `6ec5e68` 相同：`tools/score.py`、`tools/make_val.py`、`colab/build_prereg0*_*.py`、`colab/prereg0*_*.ipynb`、`results/score_ledger.csv`、`docs/PREREG_01*.md`、`PREREG_02*.md`、`PREREG_03*.md`。
4. `docs/PREREG_04_OPPONENT.md`：`6ec5e68` 版的每一行照順序出現在現在的檔（只加）。
5. R20 路徑沒有多出程式或設計檔：`git ls-files` 裡符合指紋 `patterns` 而不在指紋 `files` 裡的檔，只能在 `results/test02/` 底下，而且不在 `results/test02/design/`、`results/test02/null/`，也不是 `results/test02/notebook_diff_*`、`results/test02/rank/fallback_*`（測試日的 test02 輸出本來就會新增）。
6. `tools/t04_*.py`：不 import player 模組、`score`；原始碼沒有答案字樣；`official_log.py` 不 import `t04_rank`。
7. `.gitattributes` 有 `results/test04/** -text`；`.gitignore` 有第 2.4 節的兩行。
8. `results/test04/**` 裡沒有 `.pyc`、`__pycache__`。

### 10.5 `tests/test_t04_dryrun.py`（I3）

- `C10_SLOW=1`：第 8.1 節第 2 點、第 8.2 節、第 8.3 節。只用 CLI（和測試日一樣），樹在 repo 外（或 `C10_DRYRUN_DIR`，必須在 repo 外）。
- `C10_SLOW=2`：同上（沒有另外的慢速類別；`test_t02_dryrun.py` 的 C10_SLOW=2 類別本身就要跑）。

### 10.6 整合時的完整驗證（主流程；全部記進 `results/test04/design/build_review.md`）

1. `PY -m pytest -q tests`：全部通過（記通過數、略過數）。
2. `C10_SLOW=2 PY -m pytest -q tests`：全部通過（單獨跑，見第 8.4 節）。
3. 第 8.1 節第 3 點：兩個 clone 的 R20 輸出逐位元組相同（`decide_summary.json` 用正規化比法），golden 也相同；`git diff r20-testday-pre04` 只有 `official_log.py` 的新增行；第 8.1 節第 4 點 `official_log` 的差異測試通過。
4. 第 8.5 節 L2b、L2 通過。
5. 三個角度的獨立審查（程式正確性、登記與規格一致、使用者步驟），沒有 blocker／major 才合併；修正後重跑 1–3。
6. rev 3 定稿：`rev3_draft.md` 依實作更新，數字、檔名、結束碼和程式一致（另一個代理核對），和程式、測試、文件一起 commit；再用一個 commit 填上 rev 3 的 commit 編號。
7. 上傳副本 `data/colab_upload/test04/test04_rank.ipynb` 的 SHA 等於已 commit 的 notebook；資料夾裡只有這一個檔。
8. `data/prereg04/counts_fit.npz` 的 SHA＝`f6c6debe…`。
9. 合併前 `git diff main...p04-testday` 檢查文件衝突（`CLAUDE.md`、`PORTFOLIO_MAP.md`）；合併後在 main 上重跑第 1 點。

## 11. 風險與需要使用者決定的事

### 11.1 風險（都已有對策，不需要決定）

| 風險 | 對策 |
|---|---|
| test04 notebook 在 11/05 前從沒在真 GPU 上跑過 | CPU 冒煙 L2b、L2；8 種對稱照算讓 seam_val 能逐位元組比；建議提早彩排（11.2 第 2 點）並記 torch／cuDNN 版本和 C0 差距；preflight／斷線不算一次執行 |
| 唯一一次登記的重跑被非接縫原因用掉（Drive 舊 zip、player 同時寫 test02、斷線接續、A100 重跑） | 只讀根目錄只含 `test02/rank`；`unpack` 核對 zip 是新算的；斷線一律整份重算；重跑只在 G4；`archive` 核對失敗類別、次數上限 1 |
| 04 判定不用之後靠重跑翻盤 | `archive`、`opp`／`apply --rerun` 在登記的判定之後都拒絕，除非有已 commit、指出程式錯誤的 PREREG_04 修訂 |
| 04 符合 5.3 卻改交 R20（例如看了 public 分數） | `final`／`prepare` 要求引用已 commit 的修訂；`status` 提示違反 5.3 |
| 03 的權重只有 AI Pro Drive 那一份 | COLAB_RUN、TESTDAY_04 寫「不要動 `C10/results/prereg03`」；備份見 11.2 第 1 點 |
| `counts_fit.npz` 不在 git 裡 | TESTDAY_04 第 1 步核對 SHA；本機另有 `data/prereg04/design_scratch/counts_fit.npz` 同 SHA；都不見時只能在 repo 外預覽重算（≥ 7 個 worker，約 34 分鐘），SHA 要等於 `f6c6debe…` |
| `.venv` 升級讓 O4 不再逐位元組相同 | 11/24 前不升級；`opp`／`apply` 核對版本 |
| 兩個都叫 `submission_rank.csv` | 訊息 A／B 寫完整路徑和 SHA 前 12 碼、用 `explorer.exe` 打開正確的資料夾、OK 請求多印「這一份是 04」 |
| A100 上 C0 的差（沒實測） | G4 優先；容許差 1e-3；本機規則 B ≤ 4 題把關 |
| test02 若在 11/04 重跑，04 順延 | K6 前有 12 天；04 只要約 5 分鐘 |
| 官方定義變動（D5）和 R20 指紋衝突 | rev 3 預先寫好：R20 照 PREREG_02 9.11 改，另寫 PREREG_04 修訂附 `git diff`，指紋和 tag 不動，修訂前不上傳 04 |
| 其他 session 同時改 `CLAUDE.md`、地圖 | 只有文件會衝突；合併前檢查 |
| 每池 400 題的 Δκ 抽樣雜訊 | 只會誤判成「不用 04」；rev 3 寫成已知限制，不引用數字（驗證題內容估的數字已撤回、揭露） |
| 重跑時 G4 一直拿不到 | 等到 11/15 中午；拿不到就這一季不用 04（K6），最終 R20 |

### 11.2 需要使用者決定的事（兩件，都不擋實作）

1. **要不要在本機留一份 03 權重的備份**：使用者本人從 AI Pro 的 Drive 下載 `C10/results/prereg03/rank_a/ckpt/final.pth`、`rank_b/ckpt/final.pth`（各約 105 MB）到 `data\prereg03_weights\`（不進 git），Claude 核對 SHA。好處：Drive 出事時 04 還救得回來。建議做，但不是 K5 的條件。
2. **Colab 彩排（test02＋test04）要不要做、什麼時候做**：這是 11/05 前 test04 唯一一次真 GPU 執行。**建議做，而且提早到建置審查通過後（目標 10/15–10/22）**，發現問題時還有時間重建、冒煙、審查、甚至在 11/04 前寫 rev 4；10/28–11/02 留作有改動時的第二次。test04 約 5 分鐘、約 0.7 單元，加上 test02 兩份彩排的用量。到建置完成時再問（repo 的 `CLAUDE.md` 原本排 10/28–11/02，主流程要改成新的建議）。

## 12. critique log（三個角度的審查，2026-09-30；每條先對照程式和登記核實）

- 核實方式：讀 `tools/decide.py:278`、`results/prereg04/opp/val_report.json`（`environment` 只有 python、platform、numpy、scipy、executable；`replay_failures == 0`）、`colab/build_test02_rank_nb.py`（`_INPUT.json` 在 :582 是 `{set, question_file, sha256}`；T02 的 `READONLY_ROOTS` 只有 prereg01、02；`parse_questions` 以路徑快取在 `QSETS`；`RUN['start']` 是有時區的 ISO）、`colab/build_test02_player_nb.py:270-272`（player 寫 `test02/player`）、`docs/COLAB_RUN.md:319`（第 21 步）、`docs/TESTDAY.md:229-236、255-278`、`tools/official_log.py`（`cmd_final`／`cmd_status`／`cmd_public`，沒有 `git_commit_time`）、`tests/test_p04_hygiene.py:101-114`、repo `CLAUDE.md:80`、`docs/PREREG_04_OPPONENT.md` 第 4、5 節、`frozen_config.json` 的 `single_colour.kappa_reference`。
- 沒有任何一條改動登記的科學內容（來源、格子、比率、Q02、凍結參數、閘門、H4R、截止點、三個門檻）。

| # | 角度 | 嚴重度 | 狀態 | 處理 |
|---|---|---|---|---|
| R1 | 登記 | major | accepted | `--not-04-reason` 必須引用已 commit、晚於 04 `prepared_at` 的 PREREG_04 rev N（N ≥ 4）；工具核對存在與時間；只有自由文字一律拒絕（第 3.0 節「登記修訂的核對」、第 9.3 節、rev 3 第 12 條、第 10.3 節測試） |
| R2 | 登記 | major | accepted | (a) 有符合 5.3 的 04 時，`prepare` 非 04 的競賽二檔要同樣的修訂；(b) `status` 印「違反 PREREG_04 5.3」；(c) TESTDAY_04 開頭與 repo `CLAUDE.md` 寫明 U5 對競賽二由 5.3 取代、04 判定前不跑 `final --comp rank`，`CLAUDE.md:80` 改指 TESTDAY_04（第 9.1、9.2、9.4 節、第 1 節主流程列）；各加測試 |
| R3 | 登記 | major | accepted | `archive` 只收接縫類失敗（工具判類別）、有 `t04_apply_report.json` 或 `opp` 判不用時拒絕、占次數的封存上限 1；`apply` 以 1／4 結束一律寫 `attempt_verdict.json`（含前提階段）；`unpack` 核對新 zip 是重算的；`official_log` 拒絕 ≥ 2 個占次數的封存或含 apply 報告而無修訂的封存；`opp`／`apply --rerun` 在登記的判定之後同樣要修訂（第 3.1、3.2、3.3、3.7、9.2 節、rev 3 第 9 條） |
| R4 | 登記 | minor | accepted | rev 3 第 19、20 條改列為「會影響測試日判定的（續）」，修訂紀錄列改成「第 1–13、19、20 條」；第 19 條補「04 的修訂不授權 R20 的任何變動、先告訴使用者、hygiene 失敗期間 t04 全部停下」 |
| R5 | 登記 | minor | accepted（選 a） | 照登記 5.2 字面：seam_val 失敗 > 0.5% 也是結束碼 3（04 不用）；0 < 失敗 ≤ 0.5% 或不逐位元組相同是 4。「只比第 5 節更嚴」因此成立（第 3.0、3.2、7 節、rev 3 第 2、3 條） |
| R6 | 登記 | minor | accepted | `opp` 的 O4 Δκ ≤ 0／NaN／沒有子題就是登記的判定；兩者都跑時以 `apply` 為準（第 3.2 節、rev 3 第 7 條） |
| R7 | 登記 | minor | accepted | 新增 `official_log` 的 tag 版對新版差異測試，列進 K5 條件（第 8.1 節第 4 點、第 10.3、10.6 節、rev 3 第 13 條） |
| R8 | 登記 | minor | accepted | 撤回用驗證題內容估的「≤ 2%」：登記與規格不引用數字，rev 3 第 22 條揭露設計階段做過這個估計、沒有影響任何規則（第 6 節、rev 3 第 7、22 條）。不另用 inner 重估：frozen_config 的 inner CI 是玩家層級的重抽，換算成每題標準差不正確，而這個數字本來就不進任何規則 |
| R9 | 登記 | minor | accepted | rev 3 第 14 條補時點說明：測試日部分由第 5 節「K5 前另寫修訂」規範；登記 5.1 的原文只要求新方法名稱、認得 04 的檔，不改 `RANK_COMBOS` 只偏離 plan、不偏離登記 |
| F1 | 忠實 | major | accepted | 核實：`val_report.json` 與 `po.environment()` 都沒有 zlib。改成 `ZLIB_PIN = "1.3.1"` 常數；Python、numpy、scipy 仍和 `val_report.json` 比；報告另加 zlib 欄位（第 3.0 節第 7 條、rev 3 第 18 條） |
| F2 | 忠實 | major | accepted | 核實：`decide.py:278` 寫絕對路徑。兩份 `decide_summary.json` 改用「刪掉 `input` 鍵的正規化 JSON」SHA 加關鍵欄位；兩個 worktree 用同一個 `C10_DRYRUN_DIR` 路徑先後跑（第 8.1 節） |
| F3 | 忠實 | minor | accepted | 各組格子加 `assert parse_val() is _V`、呼叫後核對 `_DONE_` 的 `rows`、`SET_LOG` 記 `questions_path`；L2b 冒煙加 P03 函式直接算 public／private 的逐位元組比對（第 2.3、4.4、8.5、10.2 節） |
| F4 | 忠實 | minor | accepted | `apply` 的 dump 核對定順序：key → 環境 → 完成標記 → 內容；完成標記不符定為結束碼 1；負面演練要重寫 npz 並更新標記；swap 演練寫明 Δκ 恰好 0.0（第 3.3、8.3 節） |
| F5 | 忠實 | minor | accepted | 本機重算 C0 直接 exec 已 commit notebook 的 `c0_of`（torch CPU float32），ast 核對和 T02 逐字相同；本機值和 notebook 值的差只記錄（第 3.3 節第 5 點） |
| F6 | 忠實 | minor | accepted | 核實：T02 的 `_INPUT.json` 是 `{set, question_file, sha256}`。第 2.3 節照改 |
| O1 | 操作 | major | accepted | 核實：player 寫 `test02/player`。`READONLY_ROOTS` 只放 `test02/rank`（彩排 `test02_rehearsal/rank`）；COLAB_RUN／TESTDAY_04 寫「player 還在跑或等 G4 時不跑 test04」；測試加「player 寫檔時仍 completed」（第 4.2、9.5、10.2 節） |
| O2 | 操作 | major | accepted | `unpack` 在有封存時核對 zip SHA、執行開始時間、seam_val 完成標記時間（不成立→2）；COLAB_RUN 請使用者同時改名 `sw` 和 zip；`archive` 搬走 `results/test04/*.zip` 全部（含 ` (1)`）；測試三項（第 3.1、3.7、8.3、9.5、10.1 節） |
| O3 | 操作 | major | accepted（修改） | 重跑一律只在 G4（第一次在 G4 或 A100 都一樣）：同時避免 A100 未實測的差距和「換到容許差較寬的 GPU 重試」；G4 最晚等到 11/15 中午，拿不到就這一季不用 04（K6）。第一次仍是 G4 優先、A100 備用（和 test02 一致），不採「第一次先等 G4 到 11/10」：A100 第一次失敗只會讓重跑落到 G4，不損失什麼（第 5.4、9.5 節、rev 3 第 4、9 條） |
| O4 | 操作 | major | accepted | 核實：COLAB_RUN 第 21 步和 TESTDAY.md 彩排收拾會先收走 `C10_test_v1.zip`。彩排那天 test02 第 21 步先不做，三份都跑完再四個檔一起收；Claude 在 test02 彩排前先講（第 8.6、9.4、9.5 節） |
| O5 | 操作 | minor | accepted | apply 報告新增 `inputs_bound`（repo 相對路徑→SHA），`official_log` 逐項重算；notebook 布林欄位都要 true、`earlier_failed_runs` 空；換掉 test02 報告的測試（第 3.6、9.2、10.3 節） |
| O6 | 操作 | minor | accepted | `status` 在「最後一次上傳是現在不符合的 04」「最終是現在不符合的 04」時印改交 R20 的提示；`final` 指定 `prepared` 紀錄時多印一行；TESTDAY_04 第 7 步 11/22、11/23 各跑一次 `status`（第 9.1、9.2、9.4 節） |
| O7 | 操作 | minor | accepted | `unpack`（real）核對 R20 U1 的 `uploaded_at` 早於 `run_log.runs[0].start` 和每個 `_DONE_` 的 `time`；不成立→2，`archive` 記 `before_r20_u1`、不占次數（第 3.1、3.7、5.4 節、rev 3 第 1 條） |
| O8 | 操作 | minor | accepted | 斷線一律不接續：`sw` → `sw_interrupted_N`、整份重算、不算一次執行；notebook preflight 第 6 點擋「已有完成標記」、擋「`sw_interrupted_*` 其實是接縫失敗」；本機 `unpack` 核對每個 `_DONE_` 的 `run` 等於最後一次（第 3.1、4.3、4.4、5.4、9.5 節、rev 3 第 9 條） |
| O9 | 操作 | minor | accepted | FAIL_OPENING 統一結束碼 3；`opp`／`unpack`／`apply` 在 real 下過了 K6 就結束碼 3；TESTDAY_04 寫「11/15 前完成 apply 並 commit」；rev 3 第 7 條同 R6（第 3.0、5.4、9.4 節） |
| O10 | 操作 | minor | accepted | 核實：TESTDAY.md 彩排用 `<S>\rehearsal`、`<S>\local_rank`、`<S>\results\test02\checks`。第 8.6 節改成這些路徑；第 3.0 節第 5 條在非 real 模式改成「`mode` 等於本次模式、不要求 commit」 |
| O11 | 操作 | minor | accepted | `public` 新增只附加的警告（04 和 R20 分數完全相同）；apply 報告記 `rule_b_differences_vs_r20`，訊息 A 印出（第 3.3、3.6、9.2 節） |
| O12 | 操作 | minor | accepted | 彩排建議提早到建置審查通過後（10/15–10/22），記 torch／cuDNN 和 C0 差距；接近門檻時在 11/04 前告訴使用者、決定要不要寫 rev 4；10/28–11/02 留作第二次（第 8.6、11.2 節）。要使用者同意，列為需要決定的事 |

- 沒有不接受的條目。O3 照審查的主要修法（重跑只在 G4），沒採它「也可以考慮」的第一次延後改 A100。R5 選了修法 (a)。
