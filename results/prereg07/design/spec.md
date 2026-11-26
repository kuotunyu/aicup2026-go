# PREREG_07 實作規格：競賽二 Q02H（Q02＋KataGo HumanSL）

- 依據：`docs/PREREG_07_HUMANSL.md`。本檔和登記不一致時，以登記為準。
- 本檔是設計文件，不是登記：實作偏離本檔不用寫修訂；偏離登記才要寫 rev 2，而且要在 `check` 之前 commit（登記第 9 節第 5 步）。
- 範圍：只做競賽二。GPU（本機 RTX 4090）只用在第 3 節（驗證題的 HS）和第 7.3 節（測試日的 HS）。不需要 Colab。
- 審查：`review_r1.md`（r1 的兩份審查和逐點回應）。
- 數字來源：
  - `results/prereg07/explore/pilot_humansl/` 的 README、`pilot_fuse_report.json`（bank 版，`56f46de`，SHA `a44703d4…`）、`hs_gpu_bench.json`、`pilot_smoke_check.json`。
  - `data/prereg07/pilot/run_log.jsonl`（10/3–10/4 的實際速度）：fitmap 3,000 盤、466.9 萬次評估、4,046 秒（1,154 evals/s）；bank 11,813 盤、1,843.6 萬次、11,064 秒（1,666 evals/s）；每盤平均約 1,561 次評估。
  - `results/prereg04/opp/val_report.json`：驗證題 1,589 題、19,156 盤。
  - 標「估計」的都是推算。

## 0. 紀律（每支新工具都要做到）

- **data/local**：只有這幾個地方准讀，而且只讀 `rank_val_questions.csv` 的 `question_id`、`num_games`、`color_i`、`sgf_i`；都只在正式模式（`t07_fake_tests.py from-val` 是演練工具，照 `t04_fake_tests.py from-val` 的做法）：
  - `p07_hs.py val`（凍結之後）；
  - `p07_rank_fuse.py apply`（沿用的 p04 讀檔函式需要題目檔時，讀法同 `p04_rank_fuse.py apply --questions`）；
  - 測試日 `t07_rank.py hs --set seam`（只用前 400 盤）；
  - 測試日 `t07_rank.py apply`（`t04_rank.fuse_set` 讀 seam_val 的題目 meta）；
  - `t07_fake_tests.py from-val`（演練用的假測試集）。
- **import 的順序**：讀完題目檔之後才 import `hs_lib`、`pilot_run`、`pilot_fuse`（`hs_lib` 在 import 時裝防護，同一個行程之後碰不到 `data/local`）。`pilot_fuse` 只在函式裡 import `u1_lib`，所以 import `pilot_fuse` 本身不會裝 `u1_lib` 的防護。
- **`u1_lib` 只給 `check`、`selfcheck`、`tune` 用**（延遲 import）。`p07_rank_fuse.py apply`、`p07_hs.py` 和所有 `t07_rank.py` 子指令都不能 import `u1_lib`，也不能 import 會帶進它的東西（例如呼叫 `pilot_fuse.fit_mapping`）。理由：`u1_lib._forbidden` 用子字串比對 `data\local`，`C:\Users\USER\AppData\Local\Temp` 也會被拒絕。
- **暫存**：所有工具的暫存寫在 `data/prereg07/tmp/`、`data/test07/tmp/`（測試日）、`data/test07_rehearsal/tmp/`（演練），用完刪掉；一律不寫 `%TEMP%`。逐位元組比對能在記憶體裡做的就在記憶體裡做。
- 路徑含標籤檔字樣一律拒絕（同 `t04_rank.refuse_label_path`）。路徑比對用絕對路徑和 realpath，逐段比對 `data`、`local`，不用子字串。
- **不跑 `tools/score.py`**。H7R 由主 session 照登記第 3 節手動執行。
- **import `results/` 底下的程式**：只准 import 四個釘住的檔，不複製、不改：
  - `results/prereg07/explore/pilot_humansl/hs_lib.py`（`d19b6a8615ef757891ff2d451fe5b96b19f31cc706a87862e49a7688d0bde97c`）；
  - 同資料夾的 `pilot_run.py`（`b0d127fd1d5a83af5b17ec3df1dccc85d94b11c3479d33cac07448a3796a3936`）、`pilot_fuse.py`（`37fb2dff0f37032bf08acc7f27213cafd30533ab024973f92b5b28f41c6ac0c6`）；
  - `results/prereg05/explore/u1/u1_lib.py`（`225dd51cff961ae3608e003432388e0b6d6a053fb4dac26add5425c002dc3ff7`）。
  - import 之前先設 `sys.dont_write_bytecode = True` 和 `os.environ['PYTHONDONTWRITEBYTECODE'] = '1'`（`pilot_run` 用 spawn 的 Pool，子行程只繼承環境變數，不繼承 `sys.dont_write_bytecode`），再讀檔算 SHA-256（CRLF 折成 LF；10/4 四個檔在工作樹裡都是 LF，兩種算法相同；repo 是 `core.autocrlf=true`，重新 checkout 可能變 CRLF），不等於釘住的值就拒絕。跑完確認 `results/prereg07/`、`results/prereg05/explore/u1/`、`third_party/katago/` 底下沒有新的 `__pycache__`。
  - 理由：inner 的 HS 就是這幾支算的，驗證題和測試日要用同一份程式；移植會多一個要核對的副本。這和 PREREG_06 spec 第 0 節「不 import results/」不同，是本線刻意的選擇（登記第 1.2 節）。
- **測試的隔離**：`tests/test_p07_*.py`、`tests/test_t07_*.py` 一律不在 pytest 的行程裡 import `hs_lib`、`u1_lib`、`pilot_*`。要用到它們的路徑，用子行程跑工具的 smoke 模式（同 t04 的 dryrun 測試）。每個 p07／t07 測試模組加一個模組內的 autouse fixture：每個測試結束時 `builtins` 不能有 `_c10_hs_guard`、`_c10_u1_guard`，有就失敗。`tests/conftest.py` 不改（06 的指紋要釘它的 SHA）。
- **每個指令寫一份 JSON 報告**：輸入檔的 SHA-256、工具和釘住檔的 SHA-256（LF）、git HEAD、mode、秒數、環境（`.venv`：Python 全文、numpy、scipy；GPU 指令另記 `.venv-gpu` 的 torch、cuDNN、GPU 型號、驅動版本、`nvidia-smi` 的 compute 程序清單）。real 模式要求工具已 commit 而且乾淨。
- **輸出已存在**時拒絕；要重跑加 `--rerun --reason`，理由和舊檔的 SHA 記進報告。GPU 指令可以續跑（每塊原子寫入，第 3.3 節），續跑不算重跑。
- **環境**：11/24 前不升級 `.venv`（Python 3.12.13、numpy 2.5.3、scipy 1.18.1）和 `.venv-gpu`（Python 3.12.13、torch 2.11.0+cu128、cuDNN 91900、numpy 2.5.3；GPU `NVIDIA GeForce RTX 4090`，出自 `hs_gpu_bench.json` 的 environment）。NVIDIA 驅動版本只記錄，不當前提；數值漂移由 GPU 接縫把關（第 7.4 節）。
- **GPU 工作用獨立的背景程序**（照 `data/prereg07/pilot/mono1.sh`：Git Bash 腳本，PowerShell `Start-Process -WindowStyle Hidden` 啟動），不受 Claude 用量上限和對話結束影響。腳本裡 `export PYTHONDONTWRITEBYTECODE=1`；**腳本只跑 GPU 那一步，不在後面自動接任何會算分數或判定的程式**（`mono1.sh` 在 STOP 後自動跑 `pilot_fuse.py`，造成了 10/4 19:00 的第二次中途結果）。

## 1. 檔案

### 1.1 新檔（main，在 PREREG_07 rev 1 之後）

| 檔 | 用途 |
|---|---|
| `tools/p07_hs.py` | `manifest`、`val`、`status`（`val` 用 `.venv-gpu`） |
| `tools/p07_rank_fuse.py` | `check`、`selfcheck`、`tune`、`apply`（CPU，`.venv`） |
| `tests/test_p07_hs.py`、`tests/test_p07_rank_fuse.py` | 單元測試（CPU；GPU 部分用假 Scorer；用到釘住檔的路徑以子行程跑） |
| `tests/test_p07_hygiene.py` | 防誤改（第 6 節） |
| `results/prereg07/hs/` | `inner_manifest.json`、`val_hs.npz`、`val_report.json`、`val_seam_moves.npz` |
| `results/prereg07/rank/` | `check_report.json`、`selfcheck.json`、`tune_report.json`、`frozen_config.json`、`hs_map.npz`、`apply_report.json`、`rank_val_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`check_submission.txt`、`prescore_audit.json`、`score_H7R.json` |
| `results/prereg07/explore/pilot_humansl/run_mono1.log` | mono 的 run log 副本（登記第 9 節第 3 步，照 `run_night1.log` 的做法） |
| `results/prereg07/RESULTS.md` | 主 session 寫，另一個代理核對 |
| `data/prereg07/val/chunk_*.npz` | 驗證題 HS 的 chunk（不進 git，SHA 在 `val_report.json`） |
| `data/prereg07/pilot/mono2.sh` | mono 續跑用的啟動腳本（只跑 `pilot_run.py run --set mono`，沒有 fuse；不進 git） |
| `data/katago_backup/b18c384nbt-humanv0.ckpt` | 權重的同 SHA 備份（不進 git） |

### 1.2 新檔（分支 `p07-testday`，H7R 採用後）

- `tools/t07_rank.py`、`tools/t07_fake_tests.py`、`tests/test_t07_*.py`（含 `test_t07_hygiene.py`、`test_t07_official_log.py`、`test_t07_dryrun.py`）、`docs/TESTDAY_07.md`、`results/test07/design/`（`build_review.md`、golden 檔）。
- `tools/official_log.py` 只加行；`.gitattributes` 只加一行 `results/test07/** -text`；`.gitignore` 只加一行 `results/test07_rehearsal/`。
- 輸出 `results/test07/`，大檔 `data/test07/`（`data/` 本來就在 gitignore）；演練只寫 `results/test07_rehearsal/`、`data/test07_rehearsal/`。

### 1.3 不能動的

- R20 的 39 個檔（tag `r20-testday-pre04`）、04 的 21 個檔（tag `p04-testday-pre06`）、06 的測試日路徑（tag `p06-testday-pre07`，07 測試日動工前照登記第 5.1 節的順序建立）。
- `results/prereg01`–`prereg06/**`、`results/prereg05/explore/u1/**`、`tools/p02_*.py`、`p03_rank_fuse.py`、`p04_*.py`、`p06_*.py`、`t02_*.py`、`t04_*.py`、`t06_*.py`、`c10data.py`、`decide.py`、`score.py`、`make_val.py`、`tests/conftest.py`。
- `third_party/katago/**`、`results/prereg07/explore/**`（試跑；登記第 9 節第 3 步 commit 的 `pilot_fuse_report.json` 和 run log 副本是最後一次更動）。
- 新檔名先對照 `tests/test_t04_hygiene.py`、`test_p04_hygiene.py`、`test_p06_hygiene.py` 的 pattern：不能含 `test02`、`test04`，不能放進 `results/test02`、`results/test04`、`results/prereg04`，不能新增 `tools/p02_*`。06 的 kit 不准 import `t07_rank`（`test_t06_hygiene.py`），07 的程式也不 import 06 的工具。

## 2. 共用常數（`p07_hs.py`、`p07_rank_fuse.py`、`t07_rank.py` 一致）

- KataGo commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`；`third_party/katago/SOURCE.txt` `039d5a1cbc815b4d69b68dc8c01163354820ac00b5070ef136c14a040b5ba20b`。
- 權重 `data/katago/b18c384nbt-humanv0.ckpt`：323,752,318 bytes、`84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27`。每個 GPU 指令和 `manifest` 都完整重算（`hs_lib.check_checkpoint(full_hash=True)`）。原檔不對時照登記第 1.2 節：備份 → 使用者同意的重新下載 → 停下。
- profile：`pilot_run.PILOT_PROFILES` 的 14 個 `rank_*`，等於 plan.json 的 `profiles`，順序固定。
- 跑法：fp16、batch 1,024、channels_last 開、`group_batches` 16、8 個特徵化 worker、`cudnn.benchmark` 開、sides 只有 target。
- plan.json `8494e9afd79c06249bd81545263b86caef7b9e77f3cd631a1e354368cfe33946`；chunk 200 盤；格式 `c10-prereg07 pilot chunk v1`（驗證題和測試日沿用，多幾個欄位，第 3.3 節）。
- 驗證題：1,589 題、19,156 盤；`results/prereg02/rank/dumps/val_keys.npz` `055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c`；題目檔 `73e51bbfd7b051e73cbb62af4b180c403d2422f122240c5e67e9c92303776db5`。
- prereg04：`frozen_config.json` `ae728e9c936a55dfeced255e9b05b5ac6a4955a7101f108e78c53c679e162661`、`rank_val_probs.csv` `3c7ab658…`、`pred_B.csv` `27b6d225…`、`val_o4.npz` `30d312c02c357f6761567099003885be5e08635b6e6c0b49d71baac137cd9cd4`、`inner_o4.npz`（SHA 由 `manifest` 讀當時的檔記下並核對 04 的紀錄）（全文照 `tools/t04_rank.py` 的 `P04_SEAM_PINS` 等常數）。
- 試跑的數字（`selfcheck` 的目標）：Q02 0.6101211982700248；Q02H 0.6242752597994481；差 0.014154061529423334；CI [0.004729748170068845, 0.023937551131530375]；每題差 0.012494963827849034；λ 1e-4；CV log-loss 1.7571755879185869；重擬合 loss 1.7425066323027925；評估假題 127,200 題。
- 單色假題：120,544 題（`45d0f14a…`），mono 失敗盤上限 306（0.5%）。新題子集：第二次中途結果沒涵蓋的 108,719 題（第 4.3 節第 5 點）。
- HS 方向檢查的下限：驗證題 0.078（n＝1,589）、測試日每組 0.155（n＝400），都是單側 p＝0.001 的 Spearman 臨界值（≈ 3.09／√(n−1)）。
- GPU 接縫容許差：每手 log p 0.05、每盤每個 profile 的平均 log p 0.002、每盤 lp_HS 0.02。

## 3. `tools/p07_hs.py`

### 3.1 `manifest`（CPU）

- 讀 plan.json 和 `data/prereg07/pilot/{fitmap,bank,mono}/chunk_*.npz`（不讀 `.tmp`）。
- 核對：塊數 15／60／307；每塊 `ref` 等於 plan 的那一段（`pilot_run.set_games`）；`profiles`、`sides`、`precision`、`batch`、`channels_last`、`katago_commit`、`hs_lib_sha256` 等於第 2 節。
- 完整重算權重 SHA（補上試跑 chunk 的 `not hashed in this run`）。
- **手數交叉核對**（登記第 2.2 節第 5 條）：bank、mono 每一盤，依 chunk 的 `ref`（＝inner_keys 的列）對到 `results/prereg04/opp/inner_o4.npz` 的同一列（先核對兩邊的 `game_id`、`player_id`、`color`），比 HS 的 `n_moves` 和 O4 `n_moves` 的合計、HS 的 `counts[:,0]` 和 O4 `n_moves[:,0,:]` 的合計。只在兩邊都沒失敗（HS `failed` 為否；O4 `anomalies` 0）的盤上比，記不符的盤數和前 20 筆。10/4 在 bank 前 1,000 盤試過，0 不符。
- 寫 `results/prereg07/hs/inner_manifest.json`：每塊的檔名、SHA-256、盤數、失敗盤數、沒有目標手的盤數、`created_utc`、`torch`；三組合計；權重 SHA；plan SHA；`inner_o4.npz` 的 SHA；手數交叉核對的結果；`complete`（三組塊數都齊）。
- mono 還沒跑完時照樣可以跑（`complete: false`），但 `check` 要求 `complete: true`。
- 這一步只讀 chunk 的 meta、手數和 SHA，**不算任何 mono 的分數**。

### 3.2 `val`（GPU）

- 前提（任一不成立拒絕）：
  - `results/prereg07/rank/frozen_config.json` 已 commit、乾淨，裡面 `score_pending_val_hs` 為真（`tune` 過了閘門）；
  - `p07_hs.py`、`p07_rank_fuse.py`、四個釘住檔都已 commit、乾淨；
  - `results/prereg07/hs/val_report.json` 不存在（登記的 val 只跑一次；中斷時續跑，不重跑）；
  - `nvidia-smi` 沒有別的 compute 程序（同 `pilot_run`）；
  - `.venv-gpu` 的 Python、torch、cuDNN、GPU 型號等於 `frozen_config.json` 記的值。
- **只有 real 模式讀 `data/local`**。smoke 模式只接受放在 `data/local` 以外、用 inner 棋做成的假題目檔，`--ignore-other-gpu-jobs` 也只准在 smoke 模式。任何模式只要題目檔路徑在 `data/local` 底下，就要求已 commit 的 `frozen_config.json` 和 real 模式。
- 順序：
  1. 先讀 `data/local/rank_val_questions.csv`（`p04_opp.read_question_slots`），用 `p03_rank_fuse.load_val_keys` 讀 `val_keys.npz`，`p04_opp.slots_match_keys` 核對題號、slot、執色；要是 1,589 題、19,156 盤。
  2. 之後才 import `hs_lib`、`pilot_run`（裝防護）。
  3. 盤的順序＝`val_keys` 的列順序；`ref`＝列號；`color`＝`val_keys` 的 `color`（等於題目檔的 `color_i`）。
  4. 用 `pilot_run.run_chunk` 每 200 盤一塊，寫 `data/prereg07/val/chunk_*.npz`（tmp 再 `os.replace`）。中斷後重下同一行：`chunk_ok` 通過的塊跳過。STOP 檔、`--stop-after-min` 同 `pilot_run`。
  5. 全部塊都在之後，合成 `results/prereg07/hs/val_hs.npz`：`question_id`、`slot`、`color`、`sums`[S,14]（float64）、`counts`[S]、`n_moves`[S]、`failed`[S]、`no_target_moves`[S]。另外合成 `val_seam_moves.npz`：前 2 塊（前 400 盤）的 `pos_game`、`pos_turn`、`pos_logp`（float32）、兩塊的 SHA。寫 `val_report.json`：每塊 SHA、環境（含驅動）、開始和結束時間、失敗盤數、沒有目標手的盤數、評估數、秒數、evals/s、`frozen_config.json` 的 SHA 和它的 commit 時間。
- 失敗盤 > 0：報告 `status: stop`，結束碼 1，主 session 照登記第 2.7 節處理。
- 用獨立背景程序啟動（第 0 節），估計 5.0–7.2 小時。開跑前在對話裡告知，排夜間。

### 3.3 chunk 格式（驗證題、測試日）

- 沿用 `pilot_run` 的欄位：`format`、`profiles`、`sides`、`ref`、`set`、`precision`、`batch`、`channels_last`、`torch`、`ckpt_bytes`、`ckpt_sha256`、`katago_commit`、`hs_lib_sha256`、`created_utc`、`seconds`、`sums`、`counts`、`n_moves`、`failed`、`errors`、`pos_game`、`pos_turn`、`pos_side`、`pos_logp`。
- 另加：`cudnn`、`gpu_name`、`driver`、`keys_sha256`（這一組 key 檔或題目檔的 SHA）、`tool_sha256`。
- `ckpt_sha256` 一律是完整重算的值，不准 `not hashed in this run`。
- `chunk_ok` 另外核對 `keys_sha256` 和環境欄位（torch、cuDNN、GPU 型號；驅動只記錄）；環境不同的舊塊不跳過、也不覆寫，直接拒絕（測試日是結束碼 4：環境變了，先查）。
- 測試日另外：`created_utc` 早於 R20 第一次上傳 `uploaded_at` 的舊塊一律拒絕（結束碼 2，第 7.3 節）。
- `pos_logp` 要留著：測試日的 GPU 接縫要逐手比（第 7.4 節）。

### 3.4 `status`

只讀：每組已完成幾塊、還差幾塊、估計剩餘時間、STOP 檔在不在。不算任何分數。

## 4. `tools/p07_rank_fuse.py`（CPU，`.venv`）

共同：inner 的計算一律經由 `u1_lib.load_inputs`、`u1_lib.run_gen`、`u1_lib.load_saved`、`pilot_fuse.load_set`、`curves`、`fit_mapping`、`mlr_logp`，不重寫。`check`、`selfcheck`、`tune` 在同一個行程裡不碰 `%TEMP%`（第 0 節）。

### 4.1 `check`

- `inner_manifest.json` 已 commit、`complete` 為真；每塊現在的 SHA 等於 manifest；手數交叉核對 0 不符（或 rev 2 寫死的定義下 0 不符）。
- 每塊的 meta 等於第 2 節（`pilot_fuse.load_set` 的 meta 只能有一組值）。
- fitmap、bank 的失敗盤和沒有目標手的盤都是 0；mono 的記錄。
- plan 記的 4 個輸入 SHA 現在不變；`u1_lib.load_inputs` 成功（它核對 p02／p03／p04 的工具和紀錄）。
- `third_party/katago` 的每個檔等於 `SOURCE.txt`（同 `test_hs_lib.py` 的核對）。
- 寫 `check_report.json`（`status: ok|stop`）。不算任何分數。

### 4.2 `selfcheck`

- 照 `pilot_fuse.main` 的流程在記憶體裡重算（不寫試跑的報告檔）：
  - Q02：`run_gen(d, Q02_TERMS)` 的 `P`、`Pm` 和 `u1_oof.npz` 的 `eval_P_Q02`、`mono_P_Q02` 逐位元相同（`np.array_equal`）。
  - 對照模型：λ、各 λ 的 CV log-loss、重擬合 loss 和第 2 節的數字差 ≤ 1e-12。
  - Q02H：分等級平均、差值、CI、每題差，和第 2 節的數字差 ≤ 1e-12。
  - mono：用試跑的邏輯（`pilot_fuse` 只算盤都有 HS 的題），在試跑涵蓋的那組題上，和登記第 9 節第 3 步 commit 的全覆蓋 `pilot_fuse_report.json` 裡 Q02+HS 的 `mono_diff_vs_Q02`（`rank_balanced`、`ci95`、`by_colour`、`by_k`）差 ≤ 1e-12。報告的 `coverage` 不是 `120544/120544` 而 mono 也沒有失敗盤時，停下查（表示那不是全覆蓋的那一次）。
- 另外用第 4.5 節的遮罩版 E_HS 再算一次 Q02H：inner 沒有不可用盤時，必須和 `pooled` 版逐位元相同。
- 寫 `selfcheck.json`。

### 4.3 `tune`

1. 前提：`check_report.json`、`selfcheck.json` 已 commit、`ok`，而且是現行工具產生的（報告記的工具 SHA 等於現在）。`tune_report.json` 已存在時拒絕；`--rerun` 只准在一個指出具體程式錯誤的 PREREG_07 修訂 commit 之後，並在報告記那個修訂的編號和 commit（登記第 2.5 節「閘門只判一次」）。
2. 對照模型：`pilot_fuse.fit_mapping(xf, yf, seed=0)`（fitmap，side 0）。存 `hs_map.npz`：`mu`[14]、`sd`[14]、`W`[14,10]、`b`[10]、`lambda`、`profiles`、`cv`（JSON 字串）、`fitmap_games`。存完再讀回來，用 `pilot_fuse.mlr_logp` 重算 inner 的 lp_HS，要和記憶體裡的逐位元相同（apply 和測試日只讀這個檔）。
3. inner 的 lp_HS：bank＋mono 每盤 `mlr_logp`；不可用的盤照第 4.5 節處理。
4. OOF：Q02、Q02H、R20＋HS（只記錄）。閘門 G1–G4 照登記第 2.5 節。G4 用 `p3.mono_diff_record` 和 `PlayerBootstrap`（同 `pilot_fuse`），在全部 120,544 題上用遮罩版判；mono 失敗盤 > 306 時 G4 不過。
5. 只記錄：
   - 每級、每 k、單色黑白；新題子集（第二次中途結果沒涵蓋的題：至少有一盤不在 bank 或 mono 第 0、1 塊裡的單色假題；應為 108,719 題，題數不符就記下並停下查）的差、CI、黑白。
   - 每折參數和打到界限的權重（|w − 4| ≤ 1e-9 或 w ≤ 1e-9）；重擬合的參數、打到的界限、重擬合在 inner 擬合假題上的樣本內分數和 OOF 分數並列。
   - HS 方向檢查的兩個參照值：ρ_eval（inner 評估假題：每題「只用 HS 的期望等級」對 Q02 OOF 的期望等級）、ρ_mono（單色假題的同一個值），定義照第 4.6 節。
   - `se_val_record`（`p3.se_val_record`，同 p04 `tune`）、推估採用機率（`p04_rank_fuse.adoption`）。
6. 閘門都過：`p4.gen_fit_record(mf, Ef, Xf.n, Xf.y)` 用全部 inner 擬合假題重擬合（項順序 OP、O4、SWA、SWB、HS），寫 `frozen_config.json`。沒過：只寫 `tune_report.json`（`score_allowed: false` 和原因），不寫 `frozen_config.json`、`hs_map.npz`。
7. `frozen_config.json` 的欄位：
   - `format`＝`c10-prereg07 rank frozen_config v1`；`combo`（id `Q02H`、sources、terms、path gen、pool log）；`params`（theta 全文和 `gen_params` 的分解）；`folds`（5 折 theta 和優化紀錄）；`basis`；`formula`；`decision_rule`＝B。
   - `hs`：profiles、metadata 描述、規則、精度、批次、channels_last、權重 SHA、KataGo commit、四個釘住檔的 SHA、`hs_map.npz` 的 SHA、`inner_manifest.json` 的 SHA。
   - `prereg04`：`frozen_config.json` 的 SHA 和 `source_sha256` 全文。
   - `gates`、`hs_direction_ref`（ρ_eval、ρ_mono、兩個下限 0.078、0.155）、`se_val`、`score_pending_val_hs: true`、`tool`。
   - `environment`：`.venv` 的版本（實測）；`.venv-gpu` 的參照值（torch 取 chunk 的 `torch`，cuDNN 和 GPU 型號取已 commit 的 `hs_gpu_bench.json` environment；驅動取當時 `nvidia-smi` 的值，只記錄）。`val` 和測試日的 `hs` 都和這組比。

### 4.4 `apply`

不 import `u1_lib`；`pilot_fuse` 在讀完題目檔之後才 import（用它的 `curves`、`mlr_logp`）。

1. 前提：`frozen_config.json` 已 commit、乾淨；`val_hs.npz`、`val_report.json`、`val_seam_moves.npz` 已 commit；`val_report.json` 的開始時間晚於 `git log -1 --format=%cI -- results/prereg07/rank/frozen_config.json`；`val_hs.npz` 的 key 等於 `val_keys.npz`；各塊 SHA 等於報告。
2. Q02 重現：照 `p04_rank_fuse.cmd_apply` 的算法（唯讀 import `val_c5`、`val_arm`、`load_o4_val`、`val_question_inputs`、`gen_logits` 等），用 prereg04 的 `frozen_config.json` 算驗證題機率，在記憶體裡（或 `data/prereg07/tmp/`）用 p04 的 writer 產生，和 `results/prereg04/rank/rank_val_probs.csv` 逐位元組相同；不同就停下（`status: stop`）。p04 的讀檔路徑需要題目檔時，在 import `pilot_fuse` 之前照 p04 的讀法讀完。
3. HS：`val_hs.npz` → 曲線 → `hs_map.npz` → lp_HS → 遮罩版 E_HS（第 4.5 節）。
4. Q02H：`gen_logits(theta_07, m, [E_OP, E_O4, E_SWA, E_SWB, E_HS], n)` → softmax → `rank_val_probs.csv`（格式和欄位同 p04）。
5. 手數交叉核對（第 4.7 節）對 `val_o4.npz`。
6. HS 方向檢查（第 4.6 節）：ρ ≥ max(0.5 × ρ_eval, 0.078)。
7. 接著依序：`decide.py results/prereg07/rank/rank_val_probs.csv results/prereg07/rank`；`check_submission.py rank results/prereg07/rank/pred_B.csv --questions data/local/rank_val_questions.csv`（`pred_A.csv` 也跑），輸出寫 `check_submission.txt`。可以由工具用子程序呼叫，或印出指令讓主 session 照跑。
8. `apply_report.json`：`score_allowed`、`score_blocked_reasons`，條件照登記第 2.7 節（驗證題 HS 失敗 0，或第 2.7 節允許的 ≤ 95 盤；Q02 重現；手數交叉核對 0 不符；HS 方向檢查；`prescore_audit.json` 已 commit 且沒有找到程式錯誤；兩份格式檢查通過）。`prescore_audit.json` 還沒有時，`score_allowed` 寫 `pending_audit`，核對 commit 後用 `apply --finalize` 只更新這一欄（不重算預測，核對預測檔 SHA 不變）。另記：Q02H 和 Q02 規則 B 不同的題數、沒有可解析盤的題數、HS 不可用的盤數、各級預測分布、`se_val_record`、lp_HS 和 E_HS 的摘要。
9. 不讀任何標籤檔，不打分。

### 4.5 遮罩版 E_HS（inner、驗證題、測試日共用一個函式）

- 每盤「可用」＝C5 可解析、`failed` 為否、而且 `counts > 0`。
- E_HS(q)＝這一題可用盤的 lp_HS 的平均；一盤都沒有就是 0 向量。
- 所有 C5 可解析的盤都可用時，直接用 `rf.pooled(S, lp_HS, log)`（S 是 C5 可解析盤的平均矩陣），所以 inner 和試跑逐位元相同；有不可用的盤時，才用遮罩後重新正規化的矩陣（同 `t04_rank.fuse_set` 對 O4 的做法）。
- 溫度的 n 仍是 C5 可解析的盤數。
- 「失敗」（算 0.5% 的分子）＝`failed` 或整盤沒有任何一手；有手但沒有目標手的盤不算失敗，只是不可用，件數另記。
- 這個函式放在 `tools/p07_rank_fuse.py`，不 import `u1_lib`（`t07_rank.py` 唯讀 import 它）。

### 4.6 HS 方向檢查

- 只用至少有 1 盤可用的題。ē_HS(q)＝Σ_k k·softmax(E_HS(q))_k；ē_Q02(q)＝Σ_k k·P_Q02(q)_k（k＝0…9，D…6D）。
- `scipy.stats.spearmanr(ē_HS, ē_Q02)`（平均名次）。
- 通過＝ρ ≥ max(0.5 × 參照, 下限)：驗證題用 ρ_eval 和 0.078；測試日 public、private 各用 ρ_mono 和 0.155。參照值由 `tune` 寫進 `frozen_config.json`。
- 只抓大錯（讀錯檔、整批題和盤對錯）：完全對錯時 ρ 約 0。槽位錯開和執色顛倒由第 4.7 節的手數交叉核對抓（04 的 Δκ 只檢查 O4／SW 的執色語意，不檢查 07 的程式把哪個顏色傳給 `featurize_game`）。
- `official_log.py` 不重算 ρ（它只用標準函式庫），只核對報告裡的數字和門檻。

### 4.7 手數交叉核對（驗證題、測試日）

- 對照檔：驗證題 `results/prereg04/opp/val_o4.npz`；測試日 `results/test04/opp/o4_<組>.npz`（test04 已 commit）。兩者都有 `question_id`、`slot`、`color`、`n_moves`[S,2,3]、`moves_parsed`[S]、`replay_failed`[S]。
- 先核對兩邊的 `question_id`、`slot`、`color` 逐列相同。
- 在兩邊都沒失敗（HS `failed` 為否；O4 `replay_failed` 為否）的盤上：HS `n_moves` ＝ O4 `moves_parsed`，HS 目標手數 `counts` ＝ O4 `n_moves[:,0,:].sum()`。不符的盤數要是 0。
- 只有一邊失敗的盤：記件數，不當不符（失敗另有 0.5% 的規則）。
- 結果寫進報告（驗證題 `apply_report.json`，測試日 `t07_apply_report.json`）：比過的盤數、不符的盤數和前 20 筆、只有一邊失敗的件數。

## 5. H7R 和 RESULTS

- 指令、採用條件、對照組重讀、合理性門檻、預算、查 bug 重跑：照登記第 3 節，主 session 執行。
- **打分前核對**（`prescore_audit.json`，另一個代理，不看分數）：val HS 每塊 SHA 和 `val_report.json`；`hs_map.npz` 套用（獨立重算 lp_HS，和 apply 的相同）；遮罩版 E_HS；Q02 重現；`decide` 的輸入輸出；手數交叉核對；方向檢查的數字。只寫事實和找到的程式錯誤，不寫建議。
- `RESULTS.md` 要有：
  - H7R 的兩個差值和 CI、兩個指標的判定、對照組重讀的差、打分用的 Python 版本；
  - 各級分數和各級平均盤數（VALIDATION「風險」那一段）；Q02H 和 Q02 規則 B 不同的題數；
  - G1–G4 的值、mono 的黑白和各 k、新題子集、兩次中途結果（登記第 0 節）、R20＋HS（inner，只記錄）、打到界限的權重、λ；
  - rev 1 的 commit 時間和全覆蓋 fuse 的時間（登記第 9 節第 1 步）；
  - HS 失敗和不可用的盤數、手數交叉核對、HS 方向檢查（inner 參照、驗證）；
  - 消融表（登記第 8 節）；
  - GPU 使用紀錄（哪一晚、多久、`nvidia-smi` 的結果、驅動版本、有沒有讓出給 C4）；
  - 揭露。
- 另一個代理獨立核對（不論採不採用；固定範圍＝打分前核對的範圍加 RESULTS 的每個數字）。找到影響 Q02H 預測的程式錯誤時，照登記第 3 節的重跑規則。

## 6. 防誤改（main）

- `tests/test_p07_hygiene.py`：
  - 四個釘住檔、`third_party/katago/**` 的 SHA（LF）；`results/prereg07/explore/**` 在登記第 9 節第 3 步之後不再變（以那個 commit 為基準）。
  - PREREG_07 只准加行（對 rev 1 的 commit 做子序列比對，同 `test_p04_hygiene.py`）。
  - H7R 之後：`results/prereg07/rank/`、`results/prereg07/hs/` 的檔釘住 SHA；打分紀錄帳在 H6P 的 4 列之後：正好 2 列 `prereg07 H7R`；或 4 列，後 2 列的 note 以 `prereg07 H7R rerun (PREREG_07 rev N)` 開頭，rev N 那一列已 commit、而且早於那兩列被加進帳的 commit。沒有其他 `prereg07` 列（列數的核對只在 `C10_T04_GATE=1` 或 `C10_K5=1` 時跑）。
  - `results/prereg05/explore/u1/`、`results/prereg07/`、`third_party/katago/` 底下沒有 `__pycache__`。
  - `tests/test_p07_*.py`、`tests/test_t07_*.py` 的模組層級不 import `hs_lib`、`u1_lib`、`pilot_run`、`pilot_fuse`（靜態檢查 import 敘述）；`tools/p07_*.py`、`tools/t07_rank.py` 的模組層級也不 import 它們。
  - 不 import `u1_lib` 的工具：`p07_rank_fuse.py` 的 `apply` 路徑、`p07_hs.py`、`t07_rank.py`（靜態檢查函式內的 import）。
  - `tools/p07_*.py`、`tests/test_p07_*.py` 不出現標籤檔的字樣。
- 防誤改閘門指令加上這一支：
  `C10_T04_GATE=1 .venv/Scripts/python.exe -m pytest -q -rs tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py tests/test_p06_hygiene.py tests/test_p06_player_hygiene.py tests/test_t06_hygiene.py tests/test_p07_hygiene.py`
- H7R 之後已知的失敗只有 `test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`（帳多了 H6P 4 列和 H7R 2 列（或 4 列），PREREG_04 rev 3 第 19 條允許）。`test_p06_player_hygiene.py::LedgerRows::test_later_rows_only_prereg07` 要通過（note 以 `prereg07` 開頭）。`CLAUDE.md`、`AGENTS.md` 的閘門指令和已知例外跟著改。
- 第 2 步 commit 前：main 上用單一行程跑 `PY -m pytest -q tests`（`docs/TESTDAY.md` 的就緒清單）全綠。跑在 mono 跑完、CPU 空出來之後。

## 7. 測試日：`tools/t07_rank.py`（分支 `p07-testday`）

### 7.1 結束碼

| 碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成，條件都成立 | commit，做下一步 |
| 2 | 拒絕或錯誤：前提不符、檔案不對、舊塊（工具內部錯誤也是 2，印 traceback） | 照訊息修好前提，再跑同一行 |
| 3 | 登記的「07 不用」：寫 `results/test07/not_used.json` | commit；最終照 PREREG_04 5.3；不重跑 |
| 4 | 本機漂移或對不上（`.venv`、`.venv-gpu`、GPU 數值、程式、輸入綁定）：寫 `attempt_verdict.json` | commit 紀錄，停下查；查好後 `--rerun --reason`。手數交叉核對不符要先 commit 指出程式錯誤的 PREREG_07 修訂 |

- 沒有 1：07 沒有 notebook，也就沒有「重跑一次」的次數規則。
- 結束碼 3、或 `apply` 寫出 `t07_apply_report.json` 之後，不重跑任何步驟；要翻盤只能先 commit 測試日修訂之後的 PREREG_07 修訂（指出具體的程式錯誤），工具核對方法同 PREREG_04 第 11.3 節第 20 條。

### 7.2 共同前提（real 模式；順序固定，登記的 3 先於「輸出已存在」）

1. git 沒有未 commit 的追蹤檔變更；工具和釘住檔已 commit（2）。
2. `.venv` 版本（2）。GPU 指令另外核對：`.venv-gpu` 的 torch、cuDNN 等於 `frozen_config.json`（不同是 4）；GPU 型號是 `NVIDIA GeForce RTX 4090`（不同是 2）；驅動版本只記錄。
3. `results/prereg07/rank/frozen_config.json`、`hs_map.npz`、`results/prereg07/hs/val_hs.npz`、`val_seam_moves.npz` 等於釘住的 SHA（2）。
4. H7R 採用：`score_H7R.json` 的分等級平均差下界 > 0、每題平均差 ≥ 0，而且 `results/prereg07/RESULTS.md` 寫「**採用**」（否則 3）。
5. K6-07（2026-11-19 23:59:59 +08:00）未過。過了、之前沒有成功的正式執行 → 3；之前已有 → 2（那一次照常有效）。
6. T1（`results/test02/checks/rank/t1.json`，完整的 raw T1）：PASS、NORMALIZED 往下；FAIL_OPENING → 3；FAIL_STRUCT → 2。
7. test02 競賽二是 R20 的正式執行（不是 R10 備案）、跑完、本機接縫通過、`check_submission` official 通過，報告、dumps、`opening_test.npz` 已 commit（2）。
8. `results/official_scores.json` 已 commit，有 R20 的第一次上傳、`uploaded`、SHA 等於 test02 報告記的交件檔（2）。
9. 只有 `apply`：04 的 `t04_apply_report.json` 正式模式、結束碼 0、已 commit（04 的 `opp` 或 `apply` 判不用 → 3；還沒做完 → 2）。不要求 04 已上傳。

### 7.3 `hs --set seam|public|private`（4090）

- seam：`val_keys` 順序的前 400 盤。讀題目檔的方式同第 3.2 節（只取那 400 盤的題目內容和執色），寫 `data/test07/hs/seam/`。
- public、private：讀 test02 第 2 步打包的那一份測試檔（`data/test` 加 `MANIFEST.json`；拒絕 `data/test/raw` 和 `data/local` 當測試檔目錄），槽位和 test02 同一組的 key 逐列核對（題號、slot、執色、parsed），寫 `data/test07/hs/<組>/`。
- **舊塊**：開跑前掃 `data/test07/hs/<組>/`，有 `created_utc` 早於 R20 第一次上傳 `uploaded_at` 的塊就拒絕（2），請操作者把那個資料夾改名封存（`data/test07/hs/<組>_stale_<時間>/`），不自動刪。
- 順序：seam 先完成；public、private 每一塊的環境欄位要和 seam 相同（不同是 4）。
- 可以續跑；STOP、`--stop-after-min` 同 `pilot_run`。用獨立背景程序啟動（第 0 節），腳本只跑這一個子指令。
- 合成 `results/test07/hs/<組>_hs.npz`（seam 另含逐手 log p）和 `hs_<組>_report.json`（每塊 SHA、`created_utc`、環境、失敗盤、沒有目標手的盤、秒數、evals/s），寫好就 commit。
- `--max-games N` 只准在演練模式（第 9 節），real 模式拒絕。
- 估計：seam 約 62 萬次評估（6–9 分鐘，含 autotune）；public＋private 約 1,560 萬次（2.6–3.8 小時）。

### 7.4 `apply`（CPU）

照下面的順序，第一個不成立的那一條決定結束碼。讀 seam_val 題目 meta 的 `t04_rank.fuse_set` 呼叫要在 import `pilot_fuse` 之前做完（第 0 節）。

1. 三組 `<組>_hs.npz` 和報告已 commit；每塊 SHA 相符；三組環境相同（不同是 4）；每一塊的 `created_utc` 都晚於 R20 第一次上傳的時間（不是的話 2）。
2. **GPU 接縫**（`results/test07/hs/gpu_seam.json`）：seam 的 400 盤對已 commit 的參照（`val_hs.npz` 前 400 列、`val_seam_moves.npz`）：`failed`、`counts`、`n_moves` 完全相同；每手 log p（依 `pos_game`、`pos_turn` 對齊）最大差 ≤ 0.05；每盤每個 profile 的平均 log p 最大差 ≤ 0.002；每盤 lp_HS（用 `hs_map.npz`）最大差 ≤ 0.02；是否逐位元相同只記錄。沒過 → 4。
3. **Q02 對得上 04**：唯讀 import `t04_rank.fuse_set`，在 public、private 上重算 Q02，用 t04 的 `write_probs` 寫到 `data/test07/tmp/`（或記憶體），和 `results/test04/rank/rank_test_probs.csv` 逐位元組相同（不同是 4）。`t04_apply_report.json` 的 `inputs_bound` 每個檔現在的 SHA 不變（變了是 2：04 的輸入已變，先照 TESTDAY_04 處理）。`fuse_set` 不回傳 m（C5 的 pooled），所以 C5 那三行逐字移植，並用測試核對移植後的 P 和 `fuse_set` 的 P 逐位元相同。
4. **手數交叉核對**（第 4.7 節）對 `o4_public.npz`、`o4_private.npz`：不符 > 0 → 4。
5. **本機接縫**（`results/test07/rank/seam_val/seam_local.json`）：seam_val（1,589 題）用 `fuse_set` 的 seam_val 輸入加 `val_hs.npz` 算 Q02H，和 `results/prereg07/rank/rank_val_probs.csv` 比：
   - 題號、順序、每題盤數和可解析盤數不同 → 4（結構問題）。
   - 04 的 seam_val 輸入和 prereg04 逐位元組相同時，要求逐位元組相同；不同 → 4（程式漂移）。
   - 否則：最大差 ≤ tol（＝`results/test04/rank/seam_val/seam_local.json` 記的 tol），規則 B 最多差 4 題。沒過，而且 04 的本機接縫通過、第 3 條也通過 → 3（登記的「07 不用」：只是 07 的權重放大了 04 輸入裡的 Colab 差異，本機修不好）。
6. **HS 失敗**：public、private 各自 `200 × 失敗盤 ≤ 盤數` 才過（每組 5,000 盤時上限 25 盤）。沒過 → 3。
7. **HS 方向檢查**：public、private 各自 ρ ≥ max(0.5 × ρ_mono, 0.155)（第 4.6 節）；seam_val 的值記錄，輸入逐位元相同時要等於 `apply_report.json` 的驗證值。沒過 → 3。
8. 寫 `results/test07/rank/rank_test_probs.csv`（public 在前、private 在後，各依題目檔順序）、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`submission_rank.csv`（`pred_B.csv` 的逐位元組複本），對兩份題目檔跑 `check_submission --official`（沒過是 4，刪掉這五個檔）。
9. `t07_apply_report.json`：每一條的值；`inputs_bound`（登記第 5.4 節列的檔和各自 SHA：test02 的 `apply_test_report.json`、`opening_test.npz`、三組 C5 dumps 和 key、T1；test04 的 `opp_report.json`、`o4_<組>.npz`、`unpack_sw.json`、sw 的 dumps、`seam_check.json`、`seam_local.json`、`t04_apply_report.json`、`rank_test_probs.csv`；prereg04、prereg07 的凍結檔、`val_hs.npz`、`val_seam_moves.npz`；三組 HS 和 `gpu_seam.json`）；R20 第一次上傳的 `uploaded_at` 和三組塊最早的 `created_utc`；07 和 04、07 和 R20 在 **public 題**上規則 B 不同的題數（`official_log.py public` 的警告用）和全部 800 題的不同題數（只記錄）；一盤都解析不了、填 D 的題。
- 結束碼 4 都寫 `attempt_verdict.json`（階段、原因），不留可上傳的檔。

### 7.5 `status`

只讀：三組 HS 的進度、各資料夾裡殘留的 STOP 檔和舊塊、各前提、04 的狀態、K6-07 倒數、還差哪一步。

## 8. `tools/official_log.py`（只加行）

- 規則：只加行（每一行 tag `r20-testday-pre04`、`p04-testday-pre06`、`p06-testday-pre07` 的內容都要依序還在）；只用標準函式庫和 `check_submission`；不出現標籤檔的字樣；所有判定用報告裡的數字和檔的 SHA 重算，不重算 numpy／scipy 的統計量。
- 常數：`T07_DIR`＝`results/test07/rank`、`T07_REPORT`＝`t07_apply_report.json`、`T07_METHOD`＝`prereg07 Q02H = Q02 + HumanSL, rule B (甲)`、`K6_07`＝2026-11-19 23:59:59 +08:00、`P07_DOC`＝`docs/PREREG_07_HUMANSL.md`、`P07_MIN_REV`＝測試日修訂的編號＋1（測試日修訂寫死這個數，預定 4）。
- `is_t07_file`：只認 `results/test07/rank/submission_rank.csv`，綁同資料夾的 `t07_apply_report.json`。在 `check_reports` 開頭、`is_t04_file` 之前加一行分流到 `check_reports_t07`，紀錄帶 `test07`；不分流的話 07 的檔會被當成 R20（`integration/REPORT.md` 第 3 點對 06 實測過同樣的問題）。
- `t07_entry_eligible`：用報告裡的數字和檔重算登記第 5.4 節的每一條。「04 仍符合 PREREG_04 5.3」用 `t04_entry_eligible` 對 log 裡那一筆 04 重算，並核對 07 綁的 04 檔 SHA 和那一筆 04 綁的相同；「塊晚於 R20 上傳」用報告記的時間和 log 裡 R20 那一筆的 `uploaded_at` 比。
- 插入的行（每一處都是新的行，不改既有的行）：
  - `prepare_rule_t04` 開頭：`if bound.get("test07"):` 帶了 `--not-04-reason` 就拒絕，否則 `return None`（07 不是 R20／R10）。
  - `final_rule_t04` 的 docstring 之後：`if e.get("test07"): return None`（07 的判定另記；否則 04 的規則會把 07 記成 chosen R20，並在有符合的 04 時要求 `--not-04-reason`，`final` 07 永遠失敗）。
  - 新函式 `prepare_rule_t07`：07 的檔要 R20、04 的第一次上傳都已 `uploaded`、報告最後一次 commit ≤ K6-07；有符合的 07 時，04／R20／R10 的檔要合格的 `--not-07-reason`。`cmd_prepare` 加一行呼叫；argparse 加 `--not-07-reason`。
  - 新函式 `final_rule_t07`：選 07 時現在不符合就拒絕，記 `chosen: "07"`；有符合的 07 時選 04／R20／R10，要 `--not-07-reason "PREREG_07 rev N: …"`：N ≥ `P07_MIN_REV`、那一列已 commit、加入那一列的 commit 晚於每一筆被推翻的 07 紀錄的 `prepared_at`（核對法同 `p04_revision`）。紀錄寫在 `final_choice.rank.prereg07_5_4`。`cmd_final` 加兩行（呼叫、寫入）和印出的一行。07 和 04 都符合時選 R20，兩個 reason 都要。
  - `t04_status`：在 `last_bad` 那一行之前插一行 `final_ok = final_ok or t07_final_holds(log, fin)`（最終是 07、`prereg07_5_4.chosen` 是 07、而且現在符合）；回傳的 dict 裡插一行，只在 log 有 07 紀錄時加 `t07_eligible` 鍵。
  - `t04_status_lines`：在 `redo_r20` 那段之前插行：有符合的 07 而最終還沒設時，把「最終選擇要設成 04」那一行換成「有符合 PREREG_07 5.4 的 07：最終選擇要設成 07（docs/TESTDAY_07.md）」。
  - `t04_public_lines`：開頭插 `if e.get("test07"): return []`；`same = [...]` 之後插一行濾掉 07 的紀錄（避免把 07 標成「R20／R10」）。`cmd_public` 加一行印 `t07_public_lines`。
  - `cmd_status`：log 有 07 紀錄時才加 `out["rank"]["prereg07"]` 和幾行文字：有符合的 07、最終卻不是 07 → 「違反 PREREG_07 5.4」；現在沒有符合的 07、最後一次上傳或最終選擇卻是 07 → 「final 改照 PREREG_04 5.3，再 prepare 並重傳」；K6-07 倒數。
- `t07_public_lines`：07、04、R20 任兩份 public 分數完全相同（差 < 1e-9）、而報告記的 **public 題**規則 B 不同題數 > 0 → 警告（07 和 04 可能只在 private 題不同，所以不數全部 800 題）。
- log 裡沒有 07 紀錄時：`prepare`、`uploaded`、`public`、`final`、`status` 的輸出、紀錄檔、回傳值都和 tag `p06-testday-pre07` 版逐位元組相同。`tests/test_t07_official_log.py` 的 TagVersusNew 要涵蓋：競賽二 R20、R10、04 三種流程，各接競賽一 Q04、06-OFF、06-ON 的流程。只用標準函式庫，不 import `t07_rank`。

## 9. 演練（K5-07 前；紀錄在 `results/test07/design/build_review.md`）

- **06 的指紋先做**（登記第 5.1 節的順序）：06 的 K5' 待辦在 main 做完 → `results/test06/design/p06_path_fingerprint.json`（06 的檔釘 SHA；`tools/official_log.py`、`.gitattributes`、`.gitignore` 放 `insert_only`；`tests/conftest.py` 釘 SHA）→ tag `p06-testday-pre07` → 釘住它的測試 → 分支 `p07-testday` 從那裡開。
- **先記基準**（寫 t07 程式之前）：R20（照 `r20_fake_golden.json` 的做法）、04（t04 dry run 的輸出）、Q04／06（`q04_fake_golden.json`）的假測試演練輸出 → `results/test07/design/*_golden.json`。合併前逐位元組相同。
- **演練的輸出位置**：一律 `results/test07_rehearsal/`、`data/test07_rehearsal/`，不寫 `results/test07/`、`data/test07/`（避免留下會被正式那天沿用的塊）。
- **07 對得回 prereg07**：`t07_fake_tests.py from-val` 在 t04 的假樹上加假的 HS。seam＝`val_seam_moves.npz`、`val_hs.npz` 前 400 列的副本；public、private＝驗證題第 1–400、401–800 題的 `val_hs` 列切片。跑 `apply` 的演練模式，要求：
  - 本機接縫逐位元組相同；
  - 手數交叉核對對假樹的 `o4_<組>.npz` 0 不符；
  - 交件檔 800 列等於 prereg07 `pred_B.csv` 的前 800 列（只換題號前綴），通過 `check_submission --official`；
  - 每一步前後，test02、test04 的輸出都沒有變。
- **4090 實測**（排在沒有別的 GPU 工作時，各約 10 分鐘）：
  1. 真的跑 `hs --set seam`（寫演練資料夾），和已 commit 的參照比，記下每手、每盤、lp_HS 的實際最大差和速度。實測的 lp_HS 最大差 > 0.005（容許差的 1／4）時，在測試日的修訂裡重新檢討容許差（拿到任何測試資料之前）。
  2. 真的跑 `hs --set public --max-games 400`（演練模式）在 from-val 假樹上，走一遍真的讀 `data/test`、對 test02 key、取執色和 SGF 欄位的路徑；結果和對應驗證盤的 chunk（依題號、slot 對到原驗證盤）比：`failed`、`counts`、`n_moves` 相同，每盤平均 log p 差 ≤ 0.002。
  3. 在 T1 判 NORMALIZED 的假樹（正規化副本）上再跑一次第 2 項。
- **負面演練**（每一項斷言結束碼）：`.venv-gpu` 版本不同（4）；GPU 型號不同（2）；權重 SHA 不同（2）；seam 的一塊被改（4）；seam 每手差 0.051（4）；seam lp_HS 差 0.021（4）；資料夾裡有早於 R20 上傳的舊 seam 塊（2）；apply 時有一塊早於 R20 上傳（2）；key 和 test02 不同（2）；Q02 和 04 差 1 位元（4）；手數交叉核對 1 盤不符（4）；本機接縫結構不符（4）；本機接縫超出 tol、04 接縫和 Q02 都過（3）；04 判不用（3）；04 的 apply 還沒做（2）；沒有 R20 的上傳（2）；K6-07 已過（3 或 2）；FAIL_OPENING（3）；失敗 25／5,000 和 26／5,000（0 和 3）；方向檢查低於門檻（3）；`--max-games` 在 real 模式（2）；`prepare` 之後輸入被改（不符合）；`prepare` 07 時沒有 04 的上傳（拒絕）；`--not-07-reason` 引用測試日修訂或更早的 rev、或 HEAD 沒有那一列（拒絕）；輸出已存在沒有 `--rerun`（2）；把 `data/local`、`data/test/raw` 當測試檔目錄（2）。
- **正面演練（`official_log.py`）**：(a) R20 → 04 → 07 都 `uploaded`，`final` 選 07 成功、`status` 沒有「違反 PREREG_04 5.3」、04 的提示改成 07；(b) 07 符合時 `final` 選 04，只帶 `--not-07-reason` 成功；(c) 07 和 04 都符合時 `final` 選 R20，要兩個 reason；(d) 07 和 04 public 同分但 public 題規則 B 全同 → 不警告，有不同 → 警告。
- **完整測試**：`PY -m pytest -q tests`（單一行程，全綠）、`C10_SLOW=2`、防誤改閘門（再加 `tests/test_t07_hygiene.py`）、TagVersusNew。
- **正式模式演練**：一次性 clone、已 commit 的程式、from-val 的假樹：`hs`（seam 用 GPU；public、private 用假 HS 的替代入口，只准在演練模式）→ `apply` → `official_log.py prepare`、`uploaded`、`final`、`status`。clone 的 `data` 不能整個接到 repo 的 `data`。
- 三個角度的獨立審查（正確性、程序、操作者），回應寫進 `build_review.md`。

## 10. `docs/TESTDAY_07.md` 大綱（在分支上寫）

1. 每一步都適用的規則（同 TESTDAY_04：不開標籤檔、不跑 `score.py`；凍結檔不改；每次開工跑防誤改閘門；每次上傳都要使用者對那一份說 OK，上傳指示自成一則訊息）。
2. 結束碼表（第 7.1 節）。
3. K5-07 核對表。
4. 日期與 K6-07：11/04 只做 test02；R20 記成 `uploaded` 並 commit 後，夜間跑 `hs`；04 的 `apply` 結束碼 0 後跑 07 的 `apply`；04 上傳（最晚 11/18）後才 `prepare` 07；K6-07 是 11/19；11/22、11/23 各跑一次 `status`。
5. 第 1 步：閘門（`official_log.py status`、T1、test02 的報告）；確認 `PORTFOLIO_MAP.md` 已登記 11/04–11/07 夜間的 4090。
6. 第 2 步：`hs --set seam` → `public` → `private`（`nvidia-smi`、獨立背景程序的啟動指令、排夜間、STOP 的用法、`status` 看殘留的 STOP 和舊塊）→ 每組寫好就 commit。
7. 第 3 步：等 04 的 `apply`（TESTDAY_04 第 5 步做完：`t04_apply_report.json` 結束碼 0）。
8. 第 4 步：`apply` → commit。
9. 第 5 步：04 已上傳後，上傳 07（兩則訊息：檢查結果一則，上傳指示一則）。
10. 第 6 步：最後一版（`final --comp rank` 照 PREREG_07 5.4；11/22、11/23 跑 `status`）。
11. 第 7 步：`results/test07/RESULTS.md`（防誤改閘門的結果、HS 速度和 GPU 使用、驅動版本、GPU 接縫、手數交叉核對、本機接縫、失敗率、方向檢查、07 和 04 不同的題數、上傳紀錄、5.4 的判定、看過 public 才做的選擇（應該是沒有））。

## 11. 時程和 GPU 預算（估計）

| 工作 | 量 | 時間 |
|---|---|---|
| mono（試跑，進行中） | 61,356 盤、約 9,580 萬次評估 | 10/4 19:45 時 15／307 塊，估 10/5 10:40 前後跑完；以 `run_log.jsonl` 的 `eta_min` 為準 |
| 驗證題 HS | 19,156 盤、約 2,990 萬次 | 5.0 小時（1,666/s）到 7.2 小時（1,154/s） |
| 演練的 4090 實測 | seam 400 盤＋public 400 盤＋正規化 400 盤、約 190 萬次 | 約 30 分鐘（含 autotune） |
| 測試日 HS | seam 400 盤＋約 10,000 盤、約 1,620 萬次 | 2.7–3.9 小時 |
| CPU：`manifest`、`check`、`selfcheck`、`tune`、`apply` | — | 各幾分鐘（試跑的 `pilot_fuse` 35 秒，另加 `u1_lib.load_inputs`） |

- GPU 都在本機 4090：開跑前跑 `nvidia-smi`；10/24 前 C4 優先（mono 讓出時照登記第 2.5 節先結束啟動腳本、再放 STOP；其他 GPU 工作直接放 STOP）；超過一小時排夜間；11/04–11/07 夜間先在 `PORTFOLIO_MAP.md` 登記。
- Claude 的工作量：main 的工具和測試約 1–1.5 天；06 的 K5' 收尾和指紋約半天；測試日擴充約 3–4 天（04 的測試日擴充實際 32 小時，07 沒有 notebook，估計比較少）。

## 12. 風險

| 風險 | 處理 |
|---|---|
| mono 再被中途看一次（STOP 或異常結束時 `mono1.sh` 自動 fuse） | 登記第 2.5 節：先結束啟動腳本再放 STOP；續跑用沒有 fuse 的 `mono2.sh`；萬一看到寫進 rev 2 |
| rev 1 來不及在 mono 全覆蓋結果之前 commit | 先結束 `mono1.sh` 的 bash 行程，fuse 改手動 |
| C4 要用 4090，驗證題 HS 延後 | K3-07 定 10/16，留 9 個晚上；STOP 讓出、隔夜續跑 |
| mono 跑不完或有失敗盤 | G4 要求全部覆蓋；失敗 > 306 盤時 G4 不過 |
| G4 真的不過（兩次中途結果的下界都 < 0） | 未送分，打分機會保留 |
| H7R 沒過（估計採用機會約 2–6 成，看增益打幾折） | 照舊用 04；驗證集只打一次（查 bug 重跑照登記第 3 節的對稱規則） |
| fp16 不能逐位元重現 | 接縫用容許差；演練實測一次 |
| 測試日沿用了 10 月或演練留下的 seam 塊 | 早於 R20 上傳的塊一律拒絕；演練寫在另一個資料夾 |
| GPU 接縫的參照或權重不見 | 參照已 commit（`val_hs.npz`、`val_seam_moves.npz`）；權重有同 SHA 備份 |
| 驅動被 Windows 更新 | 驅動只記錄、由 GPU 接縫判；建議使用者暫停驅動自動更新 |
| `core.autocrlf=true` 讓 SHA 對不上 | 一律用 LF 折疊後的 SHA |
| `u1_lib` 的子字串防護擋住 `%TEMP%`；import 時的防護在單一 pytest 行程裡互相干擾 | 暫存不寫 `%TEMP%`；只有 `check`、`selfcheck`、`tune` import `u1_lib`；測試以子行程跑用到釘住檔的路徑，`conftest.py` 不改 |
| `official_log.py` 改錯，R20／04／06 的行為跟著變 | 只加行；TagVersusNew；三組假測試基準；正面、負面演練 |
| 06 的指紋和 07 對 `official_log.py` 加行衝突 | 06 的指紋把 `official_log.py`、`.gitattributes`、`.gitignore` 放 `insert_only`，在 06 的 K5' 收尾之後、07 分支之前做 |
| 測試日 04 不能用 | 07 跟著不用（設計如此） |
| 測試棋有 pass 等 `hs_lib` 不收的格式 | 0.5% 規則；不改 `hs_lib` |
| 驗證題 HS 有失敗盤 | 登記第 2.7 節 |
| 報告審查的「程式原創性」 | 揭露、消融、可重現（登記第 8 節） |
