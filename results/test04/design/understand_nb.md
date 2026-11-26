# U3：test04 notebook（prereg03 兩臂模型在測試集上）的調查與設計

範圍：`colab/build_test04_rank_nb.py` → `colab/test04_rank.ipynb`（PREREG_04 第 5.2 節第 2 點）。只讀 repo，沒有 commit、沒跑 `score.py`、沒開任何答案檔、沒用 GPU。行號以分支 `p04-testday`（= main `6ec5e68`）為準。

## 0. 結論（先看這裡）

1. **seam_val 的題目不在 `C10_test_v1.zip` 裡**：測試集 zip 只有 `tests/<6 個官方檔>` 和 `tests/MANIFEST.json`（`tools/t02_test_bundle.py:674` 的 docstring；notebook 端 `build_test02_rank_nb.py:486-497` 只收這 7 個成員）。seam_val 用 `C10_bundle_v1.zip` 的 `local/rank_val_questions.csv`（打包檔 manifest 釘 SHA `73e51bbf…`，`build_prereg03_rank_nb.py:84`），和 test02 一樣（`build_test02_rank_nb.py:574`）。所以 Drive 上兩個 zip 都要有。
2. **完全沿用 03 的算法**：`dump_val_p03`（`build_prereg03_rank_nb.py:1217-1250`）整個函式逐字搬進 test04，每題一個 batch、8 種對稱加一次整題執色對調、fp32、沒有 autocast、存原始 logits。在同一種 GPU、同一版 torch／cuDNN 上，seam_val 的輸出應該和已 commit 的 `val_p03{a,b}_d4.npz`、`val_p03{a,b}_s0_swap.npz` **逐位元組相同**：npz 的 zip 成員時間固定是 1980-01-01、不壓縮（實測），logits 相同時檔案 SHA 就相同。
3. **輸出檔名、格式和 03 相同**：每組 `val_keys.npz`、`val_p03a_d4.npz`、`val_p03a_s0_swap.npz`、`val_p03b_d4.npz`、`val_p03b_s0_swap.npz`，加上 `_DONE_` 標記和 `_INPUT.json`。本機可以直接用凍結的 `p02_rank_fuse.load_dump(path, n, 8)` 和 `p03_rank_fuse.load_swap(path, n)` 讀。第 1–7 個對稱用不到，但照算：這樣逐題迴圈才能和 03 逐字相同，接縫也能比整份 dump。代價約多 1.8 分鐘 G4。
4. **接縫（建議寫進修訂）**：key 陣列完全相同；兩臂各自在「原方向正確執色」（d4 的第 0 個對稱）和「整題執色對調」兩種輸出上，每題 float32 平均 softmax（test02 的 `c0_of`，`build_test02_rank_nb.py:768-774`）最大差都 ≤ tol。tol：算 seam_val dump 的 GPU 和 03 dump 的 GPU（`NVIDIA RTX PRO 6000 Blackwell Server Edition`）相同時 1e-4，不同時 1e-3。另外 12 個 dump 的執行環境（GPU、torch、cuDNN）要相同。以下只記錄：是否逐位元組相同、logits 最大差（8 種對稱都記）、E（每題平均 log_softmax，float64，也就是合併公式實際用的量）最大差。規則 B 的「≤ 4 題」放在本機 `t04_rank.py apply` 的合併接縫，因為單一來源的 SW 沒有規則 B 的意義。
5. **GPU 規則**：和 test02 競賽二、03 一樣，只收 G4（RTX PRO 6000）或 A100。操作上 G4 優先，G4 灰色時才用 A100。T4、L4 和其他型號一律擋下。理由見 3.7。
6. **帳號**：只能在 AI Pro 帳號跑。notebook 用「Drive 上有 `C10/results/prereg03/rank_{a,b}/ckpt/final.pth`，而且 SHA 等於釘死的值」來強制：權重只在 AI Pro 的 Drive（`results/prereg03/RESULTS.md:34-37`）。換成別的帳號，preflight 找不到權重就會停下。
7. **時間**：用 03 的實測推估，G4 約 3.5–4.5 分鐘、約 0.6 單元；A100 約 5–6 分鐘。符合登記的「約 5 分鐘、0.7 單元」。明細在 3.9。
8. **產生器不 import 凍結的產生器**：`build_prereg03_rank_nb.py`、`build_test02_rank_nb.py` 一 import 就會解析參數、寫檔（前者 `:43-46`、`:1958-1961`，後者 `:41-46`、`:1076-1078`）。改成讀**已 commit 的 notebook**（LF 的 SHA 釘死）：`colab/test02_rank.ipynb` `53337a7a…` 當測試日骨架，`colab/prereg03_rank_a.ipynb` `85e0c2ae…` 當模型與 dump 的來源。依 cell metadata `c10` 挑格子，用 `rep()`／`cut()` 修改，並用 ast 核對搬過來的片段逐字相同。
9. **最大的營運風險是權重只有一份**（AI Pro Drive）。建議請使用者同意後，把兩個 `final.pth`（各約 105 MB）下載一份到本機 `data/`（不進 git）當備份，順便可以做第 5 節的 L3 CPU 核對。另外建議 10/28–11/02 用彩排 zip 在 G4 跑一次 test04 彩排，預期逐位元組相同。

---

## 1. 03 notebook 怎麼算出已 commit 的驗證 dump（事實）

### 1.1 模型與兩臂

- 架構是官方的 `GoRankResNet(in_channels=501, num_channels=256, num_blocks=20, num_classes=10)`（`third_party/aicup2026_tutorial/network.py:19-48`），共 26,352,110 個參數（`results/prereg03/rank_a/config.json` → `runs[0].model.parameters`）：
  - `conv_in` 3×3 → BN → ReLU
  - 20 個 `ResidualBlock`（`network.py:4-16`）
  - `conv_in2` 1×1 降到 6 層 → BN → ReLU → `view` 攤平
  - `fc` 2166→722 → ReLU → `Dropout(0.5)`（eval 時不作用）→ `fc_2` 722→10
- **channels_last**：`build_prereg03_rank_nb.py:1093-1102`。`channels_last(net)` 把模型轉成 channels_last，並在 `bn_in2` 掛 forward hook `_nchw_output`（`output.contiguous()`），讓官方的 `view` 能用。數值不變，`network.py` 不改。
- **兩臂**：架構、初始化（`init_weights_sha256` `ed6e8c08…`，兩臂相同）、T＝33,768 步、AdamW（lr 1e-3、wd 0.01）、1,000 步 warmup 之後 cosine、訓練用 bf16 autocast、D4 增強，全部相同，只差取樣：
  - A 臂每一輪用全部 fit 棋，跑 12 輪；
  - B 臂每人最多 200 盤、各級等量，跑 24 輪（`build_prereg03_rank_nb.py:839-909`、`:1326-1699`；`RESULTS.md:48`）。
  - notebook 兩份只差 `ARM = 'A'/'B'` 那一行（`:1963-1970`）。
- **指定的權重**：第 T 步的 `ckpt/final.pth`（`:1666-1670`），Drive 路徑是 `MyDrive/C10/results/prereg03/rank_{a,b}/ckpt/final.pth`（`RESULT_ROOT`／`RESULT_DIR`／`RESULT_CKPT`：`:367-369`）。

### 1.2 輸入編碼

- **SGF 解析**：`SGFParseRankPrediction(history_length=250)`（`third_party/aicup2026_tutorial/utils.py:8-84`）。
  - 從根節點（空盤）起，每個節點記一個盤面；pass 也記一次，盤面不變。
  - 盤面數不到 250 時，前面補空盤；超過時取**最後 250 個**（`:53-64`）。
  - 每個盤面兩層（黑子、白子），共 500 層，uint8 `[500, 19, 19]`。
  - 解析失敗傳回 `None`，那個 slot 整列 NaN。
- **題目檔解析**：`parse_val()`（prereg02 dump 工具的原版，03 逐字沿用，`:1947-1951` 核對過）。
  - 讀題目檔時 `question_id` 當字串，取非空的 `sgf_1..sgf_num_games`（assert 剛好是前 num_games 格），執色取 `color_j`（assert 只有 B/W）。
  - 用 `Pool(NPROC).map(sgf_to_features, sgfs, chunksize=8)` 平行解析。
  - 傳回 `{'spans': [(question_id, 開始列, 盤數)], 'feats': [...], 'keys': {...}}`。
- **目標玩家視角**：`to_target_view(x, is_black)`（`:720-726`）。
  - 目標玩家執白時，每個盤面的兩層對調，變成「目標的子、對手的子」。
  - 再接第 500 層常數「目標執黑＝1」，得到 `[B, 501, 19, 19]` uint8。
- **最後輸出的輸入**：`dump_input(x, is_black, s)` = `d4(to_target_view(x, is_black).float(), s).contiguous(memory_format=torch.channels_last)`（`:737-739`）。
  - 順序：先視角轉換，再轉 float32，再做 D4。
  - `d4(x, s)`（`:693` 從 prereg02 逐字搬來）：s=0 傳回原張量；s≥4 先在 dim 3 鏡射，再在 (2, 3) 旋轉 s%4 次。
  - **對稱順序就是 s＝0..7**。

### 1.3 `dump_val_p03`：本機驗證題的兩種 dump

原始碼在 `build_prereg03_rank_nb.py:1217-1250`，生成後是 `prereg03_rank_a.ipynb` 的 cell `c10-rank03-19`。

- 執色：`colour_b = V['keys']['color'] == 'B'`，也就是題目檔的 `color_i`。
- **batch**：`for _qid, start, k in V['spans']`，一題的可解析棋疊成一個 uint8 張量，每批最多 `GPU_BATCH = 64` 盤。題目最多 20 盤，所以**每題剛好一個 batch**，batch 形狀 `[5..20, 501, 19, 19]`。
- 每一批依序：
  - `for s in range(NUM_SYMS): logits[rows, s] = net(dump_input(xb, ib, s)).float().cpu().numpy()`（8 次 forward）；
  - 然後 `swap[rows] = net(dump_input(xb, ~ib, 0)).float().cpu().numpy()`（原方向、每盤的執色都反過來，也就是整題執色對調）。
- **精度**：fp32、`torch.no_grad()`、**沒有 autocast**（`AMP` 只用在訓練的 forward／loss，`:1583`；最後輸出的註解寫一律 fp32，`:1705-1706`）。存的是**原始 logits**（`fc_2` 的輸出），沒有 softmax、也沒有 log_softmax。log_softmax 在本機 `p04_rank_fuse.py:874-876`、`:965-966` 用 float64 算。
- **確定性旗標**：03 沒有設 `cudnn.benchmark`／`deterministic`（都是預設 False），TF32 也是 torch 預設：`config.json` → `runs[0].training.tf32 = {'cudnn': True, 'matmul': False}`。所以卷積走 cuDNN 的 TF32，全連接層是純 fp32。
- **權重載入**：`load_net(path, expect_sha)`。整個檔讀進記憶體、算 SHA-256，再用 `torch.load(io.BytesIO(raw), map_location='cpu')` 從同一份位元組載入（torch 2.11 的 `weights_only` 預設是 True，state_dict 可以讀），然後 `GoRankResNet(in_channels=IN_CHANNELS, …)`、`channels_last(net.eval().to(DEVICE))`（`:1259-1261`）。
  - 最後輸出用的是這個剛從 `final.pth` 載入的網路（`:1712`），不是訓練中的 `model`。

### 1.4 檔案格式（實測已 commit 的檔）

| 檔案 | 鍵（npz 內的順序）：dtype、形狀 |
|---|---|
| `val_keys.npz` | `question_id` `<U10` (19156,)；`slot` `<i4` (19156,)，從 1 起算；`color` `<U1` (19156,)；`parsed` bool (19156,)。1,589 題，每題 5–20 盤，全部解析成功 |
| `val_p03a_d4.npz` | `logits` `<f4` (19156, 8, 10)；`weights_sha256` `<U64` ()；`syms` `<i8` () ＝ 8 |
| `val_p03a_s0_swap.npz` | `logits` `<f4` (19156, 10)；`weights_sha256` `<U64` ()；`view` `<U17` () ＝ `'s0_colour_swapped'` |

- **列的順序**：題目檔的列順序，同一題內依 slot 1..n。解析失敗的 slot 整列是 NaN（本機驗證題沒有這種列）。
- **npz 的 zip 成員**：`date_time (1980,1,1,0,0,0)`、`compress_type 0`，所以內容相同時檔案逐位元組相同。實例：prereg02 和 03 兩臂的 `val_keys.npz` 都是 `055d2941…`。
- **`_DONE_<stem>.json`**：由 `write_dump` 寫（prereg02 dump 工具，03 逐字沿用），欄位有：
  - `file`、`rows`、`syms`（d4 是 8、swap 是 1、key 檔是 null）、`bytes`、`sha256`
  - `weights`（標籤 `prereg03/rank_a/ckpt/final.pth`）、`weights_sha256`
  - `seconds`（同一次呼叫的 d4 和 swap 記同一個秒數）、`time`、`run`
  - `gpu_name`（nvidia-smi 第一欄）、`torch`、`cudnn`（整數）、`device`
- **跳過規則**：重跑時，`dump_is_done(stem, rows, syms, weights_sha)` 的完成標記、SHA、列數、權重都對得上就跳過。

### 1.5 環境與 SHA（釘死用的值與來源）

- **GPU／環境**（兩臂都是 G4）：
  - `NVIDIA RTX PRO 6000 Blackwell Server Edition`，97,887 MiB，driver 580.82.07，48 vCPU，RAM 190 GB
  - torch `2.11.0+cu128`、CUDA 12.8、cuDNN `91900`、Python 3.13.15、numpy 2.1.3
  - 來源：`rank_{a,b}/env.json`；`RESULTS.md:44`
- **權重 SHA-256**：

| | SHA-256 | 來源（全部一致） |
|---|---|---|
| A 臂 `final.pth` | `790b767c5425e262130d6c103c71e89238edb17f927fa48e2a2770db788e50ac` | `RESULTS.md:35`；`rank_a/config.json` 頂層 `final_sha256`；`_DONE_val_p03a_*.json` 的 `weights_sha256`；npz 的 `weights_sha256`；`tools/p04_rank_fuse.py:174` `SW_WEIGHTS['a']`；PREREG_04 1.3 |
| B 臂 `final.pth` | `8e88f6ec6cbaad5372e5d62607b99ece3d77dc7a20af86a9197553f1a6c59a42` | 同上（`:36`、`:175`） |

- **接縫參照**（repo 裡已 commit 的檔，Drive 上應該是同一份）：

| 檔案 | SHA-256 |
|---|---|
| `rank_a/dumps/val_keys.npz`、`rank_b/dumps/val_keys.npz` | `055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c` |
| `rank_a/dumps/val_p03a_d4.npz` | `1e4e5957442596358b087e5f4e69c1fcf9f22f2c7933cc2049066bfab3075ecd` |
| `rank_a/dumps/val_p03a_s0_swap.npz` | `b355c26bf68069feb0e48ee95c46bd211c2bdbc6b5a48416c8408a210aec3848`（PREREG_04 1.3 釘死） |
| `rank_b/dumps/val_p03b_d4.npz` | `bad2277dd20bf832c02213be781768ba9d8a304922693bee383a28aa3531a627` |
| `rank_b/dumps/val_p03b_s0_swap.npz` | `0cadff4bae36b2a41f02fddf4366a2081d9912ede4a3063c8b50fd28a2647527`（PREREG_04 1.3 釘死） |

- **產生 03 notebook 的底稿**：
  - `colab/prereg03_rank_a.ipynb` `85e0c2ae3269c68ad166310efb2525b2d48ed1279c9cf6eb6b428e59e8693557`
  - `colab/prereg03_rank_b.ipynb` `aa3669ac…`
  - 這兩個值和 Colab 上傳副本相同（`RESULTS.md:56-59`）。
  - 檔案是 LF：`.gitattributes` 設了 `colab/*.ipynb -text`。
- **其他 03 的數字**：logits 範圍 −19.0 到 9.5（d4）、−16.9 到 9.8（swap）。inner 原方向逐盤準確率：A 27.39%、B 27.30%；執色對調後 26.72%、26.68%（`config.json` 的 `dumps_final`）。

### 1.6 03 的實測時間（`rank_{a,b}/timings.json`、`_DONE_` 標記）

| 階段 | A 臂 | B 臂 |
|---|---|---|
| `prepare_bundle`（含解開訓練資料） | 16.6 秒 | 15.9 秒 |
| `preflight` | 3.8 秒 | 4.1 秒 |
| `parse_val`（19,156 盤，48 個進程） | 7.8 秒 | 7.7 秒 |
| val dump（8 種對稱＋對調，共 9 次 forward） | 44.4 秒 | 44.2 秒 |
| inner dump（73,556 盤 × 9 次 forward） | 115.6 秒 | 115.7 秒 |
| `final_dumps` 合計 | 160.3 秒 | 160.2 秒 |
| 整次執行 | 2,575 秒 | 2,742 秒 |

- G4 上每盤每次 forward ≈ 44.4 / (19,156 × 9) ≈ **0.26 毫秒**（每題一個 batch，含搬到 GPU、搬回 CPU）。

---

## 2. test02 notebook 的慣例（test04 照用）

來源 `colab/build_test02_rank_nb.py`，生成後是 `colab/test02_rank.ipynb`（在 R20 指紋裡，只讀）。

- **Drive 路徑**：
  - `DRIVE_DIR = /content/drive/MyDrive/C10`，`WORK = /content/c10`。
  - 測試集 zip 是 `C10/C10_test_v1.zip`。第一格只讀它的 `tests/MANIFEST.json`：`format` 要是 `'c10-test02 manifest v1'`；`kind` 正式執行只收 `official`／`rehearsal`，SMOKE 只收 `smoke`（`:287-311`）。
  - 結果目錄 `results/test02`（彩排 `test02_rehearsal`，SMOKE `test02_smoke`）底下的 `rank/`（`:313-316`）。
- **只讀**：`READONLY_ROOTS`、`guard_write()`、`readonly_fingerprint()`，開頭記 `readonly_before.json`、結尾記 `readonly_after.json`（`:318-362`、`:608-620`）。
- **notebook 身分**：`NOTEBOOK_SHA256`＝把每格原始碼的 `json.dumps` 取 SHA-256，本身那一行換成 64 個 0 再算（`:58-59`、`:93-94`、`:367-377`、`:978-982`）。`tools/t02_rank.py` 會重算並核對（`t02_rank.py:250-283`）。
- **preflight**（`:428-638`）：
  - 權重和接縫參照的 SHA 釘死。
  - 測試集 zip 只收 7 個成員，每個檔的 SHA、大小要等於 MANIFEST。
  - `quick_check_questions`（T1 的一部分）。
  - **④-0 上一次執行的 `seam_check.json` 沒過就停**，要先把資料夾改名成 `rank_failed_1`（`:563-571`）。
  - 每組的 `_INPUT.json`：已存在但不同就停（`:573-596`）。
  - Drive 剩 ≥ 0.5 GB（`:440`、`:598-606`）。
  - 只讀目錄的指紋。
- **dump 工具**：
  - `DUMP_DIR` 由各組的格子指定。
  - `parse_questions(path)` 依題目檔路徑快取，用完就 `QSETS.pop`。
  - `write_keys`、`dump_is_done`、`write_dump`、`pack_results`、`flush_drive` 和底稿逐字相同（`:643-712`；ast 核對在 `:1010-1057`）。
- **三組格子**：seam_val、public、private 內容相同，只差第一行 `SET = '…'`（`:717-738`、`:1059-1062`）。
- **接縫**（`:743-859`）：
  - key 陣列相同。
  - C0（每題 float32 平均 softmax）最大差 ≤ tol：用算 seam_val dump 的完成標記上的 GPU 決定，和參照同型是 1e-4，否則 1e-3。
  - 規則 B 最多差 4 題。
  - 三組 dump 的 `gpu_name`／`torch`／`cudnn` 要相同（`:815-824`）。
  - 沒過：狀態記 `seam_failed`，打包、`flush_drive`，然後 `SystemExit`。
- **收尾**（`:864-933`）：
  - 核對每組的完成標記、SHA、形狀、權重、NaN 位置和 `_INPUT.json`，比對只讀目錄的指紋（不同時狀態記 `readonly_changed`）。
  - 打包 `test02/test02_rank_results.zip`，zip 內路徑從 `rank/` 起，並檢查必要的檔都在 zip 裡。
  - 最後 `flush_and_unmount`。
- **run_log 狀態**：`completed`／`seam_failed`／`readonly_changed`。本機 `t02_rank.py:122` 的 `FAILED_RUN_STATUSES` 會拒收。

---

## 3. test04 notebook 設計

### 3.1 輸入（都在 AI Pro 帳號的 `MyDrive/C10/`）

| 輸入 | 用途 | 核對 |
|---|---|---|
| `C10_bundle_v1.zip` | 官方 `utils.py`、`network.py`；seam_val 的題目檔 `local/rank_val_questions.csv` | 沿用 prepare 格的逐檔 SHA；preflight 另外要求 `MANIFEST['files'] == BUNDLE_FILES_PIN`（值從 `results/prereg02/rank/env.json` 讀出，同 `build_prereg03_rank_nb.py:158`、`:597-600`） |
| `C10_test_v1.zip` | public、private 題目檔 | 和 test02 同一份，照 test02 的 MANIFEST 核對。「和 test02 用的是同一份」由本機 `t04_rank.py` 比對兩邊 `config.json` 的 `test_bundle` |
| `results/prereg03/rank_a/ckpt/final.pth`、`rank_b/…`（只讀） | 兩臂權重 | SHA 等於 1.5 的釘死值（SMOKE：等於 `prereg03_smoke` 的 `_DONE_val_p03x_d4.json` 記的 `weights_sha256`） |
| `results/prereg03/rank_{a,b}/dumps/{val_keys,val_p03x_d4,val_p03x_s0_swap}.npz` 和 `_DONE_*.json`（只讀） | 接縫參照 | SHA 等於 1.5 的釘死值；npz 裡的 `weights_sha256` 等於權重的釘死值；`_DONE_val_p03x_d4.json` 的 `gpu_name` 等於釘死的 `NVIDIA RTX PRO 6000 Blackwell Server Edition` |

### 3.2 輸出（`MyDrive/C10/results/test04/rank/`；彩排 `test04_rehearsal/`，SMOKE `test04_smoke/`）

```
rank/<set>/dumps/val_keys.npz, _DONE_val_keys.json, _INPUT.json
rank/<set>/dumps/val_p03a_d4.npz  [n,8,10] f4, weights_sha256, syms=8   + _DONE_
rank/<set>/dumps/val_p03a_s0_swap.npz [n,10] f4, weights_sha256, view   + _DONE_
rank/<set>/dumps/val_p03b_d4.npz, val_p03b_s0_swap.npz                  + _DONE_
rank/seam_check.json, config.json, env.json, run_log.json, timings.json, readonly_before.json, readonly_after.json
results/test04/test04_rank_results.zip   （zip 內路徑從 rank/ 起；跳過 .pth/.pt/.tmp 和 ≥100 MB 的檔）
```

- `<set>` ∈ {seam_val, public, private}。
- **大小**：
  - seam_val 每臂 6.1 MB＋0.77 MB。
  - public、private 各約 5,000 盤（官方 5／10／15／20 盤各 100 題），每臂約 1.6 MB＋0.2 MB。
  - 全部約 22 MB，zip 約 20 MB。
- **本機**：解到 `results/test04/`，`t04_rank.py apply` 的輸出也寫在 `results/test04/rank/`（同 test02 的做法）。建議在 `.gitattributes` 加一行 `results/test04/** -text`（只附加；test02 已有同樣的一行）。

### 3.3 格子（id `c10-rank-t04-NN`；「T02」「P03」指兩個底稿 notebook 的同名格）

| # | key | 來源 | 內容 |
|---|---|---|---|
| 0 | header | 新 | 用途、兩臂、三組、步驟（AI Pro、G4 優先、A100 備用）、會停下的情況、輸出、接續、和底稿的差異表 |
| 1 | setup | T02 setup＋`rep` | 結果目錄改成 `test04`／`test04_rehearsal`／`test04_smoke` 的 `rank/`。`READONLY_ROOTS` = `prereg01`、`prereg02`、`prereg03`、`test02`、`test02_rehearsal`（SMOKE 是 `_smoke` 版）。`T04_CHANGES`。`NOTEBOOK_REL = 'colab/test04_rank.ipynb'` 和新的 hash 那一行。`config.json` 的 `prereg` 指 PREREG_04 的新修訂，`base_notebooks` 記兩個底稿的 SHA。`run_log['status']` 同 T02 |
| 2 | prepare | T02 prepare **逐字** | 不解開訓練資料；拒收答案檔的那一行是整份唯一含那個字的一行 |
| 3 | official_1 | T02／P03 逐字 | import |
| 4 | environment | 逐字 | `env.json`（torch、cuDNN、GPU） |
| 5 | official_2 | T02 official_2＋`rep` | 加 `IN_CHANNELS = 501`（正式執行 assert）。GPU 關卡照 T02 的 `GPU_MODELS_OK = ('RTX PRO 6000', 'A100')`，另外補 03 的「查不到 GPU 型號」訊息（`GPU_MODEL_SHOWN`，`build_prereg03_rank_nb.py:533-546`） |
| 6 | preflight_t04 | 新（段落照 T02 preflight） | 見 3.3.1 |
| 7 | view_d4 | **P03 view_d4 逐字** | `NUM_SYMS`、`d4` 和它的自我檢查、`D4_PERM`、`d4_gather`、`to_target_view`、`model_input`、`dump_input`、單元檢查（在 DEVICE 上）。會用到 `torch`、`DEVICE`、`IN_CHANNELS`、`save_config` |
| 8 | model_tools | 新，片段逐字取自 P03 official_14 | 只取 `state_dict_sha256`、`_nchw_output`、`channels_last` 三個 def（ast 核對）；不建模型、不建 optimizer |
| 9 | dump_tools | T02 dump_tools＋`rep` | 見 3.3.2 |
| 10–12 | dump_set_{seam_val,public,private} | 新，三格只差第一行 | 見 3.3.3 |
| 13 | seam_check | 新（段落照 T02 seam） | 見 3.5 |
| 14 | final_check | T02 final＋改寫成兩臂 | 見 3.6 |

**3.3.1 preflight_t04**（順序照 T02，全部在開始算之前）：

1. 兩臂權重存在，SHA 等於 `W_PIN = {'a': '790b…', 'b': '8e88…'}`。找不到時的訊息寫「要用 AI Pro 帳號，不要搬動 `prereg03` 的任何檔」（帳號規則）。
2. 接縫參照：兩臂共 6 個 npz 的 SHA 釘死；npz 的 `weights_sha256` 等於 W_PIN；參照 GPU `P03_REF_GPU_PIN` 釘死。
3. 打包檔的 manifest 等於釘死值。
4. 測試集 zip：7 個成員的 SHA、大小等於 MANIFEST；跑 `quick_check_questions`（片段逐字取自 T02）。
5. 上一次的 `seam_check.json` 沒過就停（T02 ④-0 的文字，改寫路徑）。
6. 每組的 `_INPUT.json`（T02 ④）。
7. Drive 剩 ≥ 0.5 GB。
8. **TF32 釘住**：
   - 先記下這次的預設值：`torch.backends.cudnn.allow_tf32`、`torch.backends.cuda.matmul.allow_tf32`、`cudnn.benchmark`、`deterministic`，以及新版 API `torch.backends.cudnn.conv.fp32_precision`（有的話）。
   - 再設成 03 記下的 `{'cudnn': True, 'matmul': False}`，`benchmark = False`。
   - 這組值由產生器從 `rank_{a,b}/config.json` 的 `runs[0].training.tf32` 讀出並 assert。
   - 兩組值（預設的、實際用的）都記進 `config.json`。
   - 本機 torch 2.14 的預設就是這組值（實測：`cudnn.conv.fp32_precision='tf32'`、matmul `none`），所以這一步不改變 03 的算法，只是防 Colab 之後改預設。
9. 只讀目錄的開頭指紋（T02 ⑦）。
10. `config.json` 頂層記：GPU、torch、cuDNN、兩臂權重、接縫參照、測試集 zip 和 MANIFEST 的 SHA、各組題目檔的 SHA 和題數。
11. 選配：讀 `results/test02/rank/run_log.json` 的 `status`（只讀），沒有 completed 就印提醒，只記錄。真正的前提由本機 apply 強制。

**3.3.2 dump_tools**（以 T02 那一格為底）：

- 拿掉 `NUM_SYMS = 8` 和 `d4` 連同它的自我檢查（已在 view_d4 那一格，做法同 `build_prereg03_rank_nb.py:1256-1258`）。
- 把 T02 的 `load_net`（prereg02 的 500 層版）換成 **P03 的 `load_net`**（ast 逐字）。
- 加入 **P03 的 `dump_val_p03` 整個函式逐字**（ast 的 `get_source_segment` 和 P03 dump_tools 的那一段相同），再加一個轉接用的 `parse_val()`：

```python
QUESTIONS_PATH = None   # 本版：各組的格子在呼叫前指定（同 DUMP_DIR）；03 的 dump_val_p03 一字不改地呼叫 parse_val()
def parse_val():
    return parse_questions(QUESTIONS_PATH)
```

- 這樣 03 產生已 commit dump 的那個函式，在 test04 裡逐位元組相同，只差它讀的題目檔。T02 自己已經用同樣的模式處理 `DUMP_DIR`。
- 另一種做法是照 T02 把函式改名成 `dump_questions_p03(net, path, …)`，逐字核對第一個敘述以外的部分。這個比較弱，不建議。
- `write_dump`、`dump_is_done`、`write_keys`、`require_gpu_for`、`pack_results`、`flush_drive`、`sgf_to_features`、`parse_questions`、`_argmax_first`、`rule_b` 和 T02 逐字相同。
- `record_readonly_after` 照 T02 的版本，只改註解；它對 `READONLY_ROOTS` 通用。
- 最後把 T02 的 `NET = None` 改成 `NETS = {}`，並加上：
  - `ARMS = ('a', 'b')`
  - `DUMP_NAMES = {'a': 'p03a', 'b': 'p03b'}`
  - `W_LABEL = {x: 'prereg03/rank_%s/ckpt/final.pth' % x}`，和 03 的標籤相同（`:1709`），所以 npz 和完成標記的權重欄位和 03 一模一樣。

**3.3.3 各組的格子**（第一行 `SET = '<set>'`，其餘三格相同）：

```python
t0 = time.time()
_path = SET_QUESTIONS[SET]
DUMP_DIR = guard_write(os.path.join(RESULT_DIR, SET, 'dumps')); os.makedirs(DUMP_DIR, exist_ok=True)
if read_json(os.path.join(DUMP_DIR, '_INPUT.json')) != SET_INPUT[SET]: raise SystemExit(...)
QUESTIONS_PATH = _path
for _x in ARMS:
    if _x not in NETS:
        NETS[_x], _ = load_net(W_PATH[_x], expect_sha=WSHA[_x])   # 兩臂的權重各只載入一次（各 ~105 MB）
_V = parse_questions(_path)                                        # 一組只解析一次，兩臂共用
_st = {'keys': write_keys('val_keys', _V['keys'])}
for _x in ARMS:
    _st[_x] = dump_val_p03(NETS[_x], DUMP_NAMES[_x], W_LABEL[_x], WSHA[_x])   # 03 的函式，逐字
SET_LOG[SET] = {...}; del _V; QSETS.pop(_path, None); torch.cuda.empty_cache() if cuda; save_config(dumps=SET_LOG); mark_time(...)
```

- **記憶體**：seam_val 的特徵是 19,156 × 180,500 bytes ≈ 3.46 GB uint8，兩個網路合計約 0.21 GB。G4 的 190 GB、A100 標準 RAM 都夠。
- **順序**：同一組內先 A 臂、再 B 臂。每次 forward 彼此獨立，所以順序不影響數值。

### 3.4 三組題目

| 組 | 題目檔 | 題數／盤數 |
|---|---|---|
| seam_val | 打包檔 `local/rank_val_questions.csv`（prepare 格的 `RANK_QUESTIONS`） | 1,589 題／19,156 盤；SMOKE 取前 12 題（142 盤，等於 03 冒煙的 val dump） |
| public | `WORK/tests/rank_prediction_test_public.csv` | 400 題（官方：5／10／15／20 盤各 100 題，每題單一執色）≈ 5,000 盤 |
| private | `WORK/tests/rank_prediction_test_private.csv` | 同上 |

- 執色一律用題目檔的 `color_i`。T1 判 NORMALIZED 時，zip 裡已經是正規化後的副本，和 test02 相同。
- 官方題目每題只有一種執色，所以「整題執色對調」就是把整題改成另一種顏色。

### 3.5 接縫：定義與失敗時怎麼做

每一臂 x ∈ {a, b}：
- 新的：`RESULT_DIR/seam_val/dumps/{val_keys, val_p03x_d4, val_p03x_s0_swap}.npz`
- 參照：`P03_ROOT/rank_x/dumps/` 的同名檔（Drive，SHA 已在 preflight 核對）

| 項目 | 定義 | 判定 |
|---|---|---|
| keys | `question_id`、`slot`、`color`、`parsed` 四個陣列 `shape` 相同且 `np.array_equal`（T02 `seam_compare` 的前半段） | 必須全部相同 |
| C0，s0（正確執色） | `c0_of(L_d4, spans, parsed)`（T02 逐字：每題可解析的盤取 `softmax(L[ok,0,:])` 的 float32 平均），新舊逐元素最大差 | ≤ tol |
| C0，swap | `c0_of(L_swap[:, None, :], spans, parsed)` 的最大差 | ≤ tol |
| tol | 新 seam_val `_DONE_val_p03x_d4.json` 的 `gpu_name` 等於參照標記的 `gpu_name`（而且不是 None；SMOKE 例外同 T02）→ 1e-4，否則 1e-3 | 兩臂分開算；本機工具要重算，而且必須等於 notebook 記的值（照 test02 build_review U4 的做法） |
| 執行環境 | 3 組 × 2 臂 × {d4, swap} 共 12 個標記的 (`gpu_name`, `torch`, `cudnn`) 全部相同 | 不同就算沒過（接續時換了環境） |
| 權重 | 12 個標記和 npz 的 `weights_sha256` 都等於 W_PIN | 必須相同 |
| 只記錄 | `bitwise_equal`（logits 的 `tobytes()` 相同；檔案 SHA 相同另記一欄，因為 numpy 版本不同時 .npy 表頭可能不同）；`max_abs_logit_diff`（s0、swap、8 種對稱全部）；`max_abs_e_diff`（每題可解析盤的 float64 log_softmax 平均，E_j 的定義，PREREG_04 2.1）；s0 C0 的規則 B 不同題數 | — |

- **passed** = 兩臂的 keys、C0（s0）、C0（swap）、權重都成立，而且執行環境相同。
- 寫 `seam_check.json`，格式同 T02，另外按臂分開記 `arms: {a: {...}, b: {...}}`。`earlier_runs` 把前幾次的紀錄接在後面。
- **失敗時**（同 T02 `:848-859`）：
  1. 記結尾指紋、`RUN.status = run_log['status'] = 'seam_failed'`；
  2. 打包 `test04_rank_results.zip`、`flush_drive()`；
  3. `SystemExit`，訊息寫：「接縫核對沒過：這次執行的 04 測試輸出一律不用、不上傳；R20 不受影響。不要自己重跑，下載 zip、截圖給 Claude。」
- **重跑規則**（建議寫進修訂，比照 PREREG_02 9.2）：
  - 由 Claude 決定才重跑：先把 Drive 的 `C10/results/test04/rank` 改名成 `rank_failed_1`，換一個新的執行階段、同一種 GPU（最好是 G4），「全部執行」，**只重跑一次**。
  - 還是沒過就這一季不用 04，最後一版交 R20。
  - K6（11/16）之後不再重跑。
  - preflight 的 ④-0 保證已完成的 dump 不會被沿用去「再比一次」。
- **規則 B 的「≤ 4 題」**（PREREG_04 5.2 的本機接縫）交給 `t04_rank.py apply`：
  - 用 test02 的 seam_val C5、t04 opp 的 seam_val O4、本 notebook 的 seam_val SW，重建 Q02 的驗證機率；
  - 和 `results/prereg04/rank/rank_val_probs.csv` 比：max|ΔP| ≤ tol，規則 B 最多差 4 題。
  - 不放在 notebook 裡，因為單一臂 SW 的 C0 做規則 B 沒有意義。

### 3.6 收尾（T02 final 改成兩臂）

- **每組核對**：
  - `val_keys` 和 4 個 logits 檔：完成標記、SHA、`< DUMP_MAX_BYTES`；
  - d4 是 `float32 (rows, 8, 10)` 而且 `syms == 8`；swap 是 `float32 (rows, 10)` 而且 `view == 's0_colour_swapped'`；
  - `weights_sha256` 依臂等於 W_PIN；NaN 的列等於 `~parsed`；
  - `_INPUT.json` 等於 preflight 記的；題數等於 `SET_ROWS`。
- 只讀目錄的結尾指紋：不同就 `status = readonly_changed`，照樣打包，但會停下。
- 打包 `test04_rank_results.zip`。必要成員：7 個紀錄檔（config、run_log、env、timings、seam_check、readonly_before／after），加上每組的 `val_keys.npz`、4 個 logits npz、5 個 `_DONE_`、`_INPUT.json`。
- 最後 `flush_drive()`，印「完成。請下載…」。

### 3.7 GPU 規則（決定與理由）

- **決定**：只收 G4 或 A100，沿用 T02 official_2 的 `GPU_MODELS_OK` 關卡。使用者步驟：G4 優先；G4 灰色或「無法指派後端」時選 A100（「大量 RAM」關掉）。其他型號、查不到型號、沒有 GPU，都在 official_2 停下。
- **理由**：
  1. 03 的參照是在 G4 上算的。同一型號、同版 torch／cuDNN、同一段程式，預期逐位元組相同，這是最強的接縫。
  2. 登記（PREREG_04 5.2）本來就預想了不同型號的 GPU（容許 1e-3）。04 不是 11/04 的關鍵路徑（先交 R20），但 K6（11/16）前要有一次成功的正式執行，允許 A100 可以避免被 G4 的供給卡住。
  3. A100 是 Ampere，卷積一樣走 TF32，和 G4 的差別只在 TF32 kernel 的累加順序，C0 的差預期遠小於 1e-3（**還沒實測**，見風險 5）。
  4. T4 沒有 TF32、bf16，算的是純 fp32，和 03 的 TF32 路徑差比較大；L4 沒測過。沿用 test02、03 已經給使用者的同一套 GPU 步驟，比較不會出錯。
- **接續**：要選中斷前的同一種 GPU，否則會因為執行環境不同而判定接縫沒過。

### 3.8 帳號規則

- PREREG_04 1.3、5.2：「只在 AI Pro 帳號跑」。notebook 沒辦法直接讀帳號，用以下三點強制：
  1. `final.pth` 在 `C10/results/prereg03/rank_{a,b}/ckpt/`，而且 SHA 釘死；
  2. 03 的 dump 在 Drive 上，而且 SHA 釘死；
  3. 找不到時的訊息照 T02 的 `P02_HELP` 寫：「確認用的是 AI Pro 帳號；Drive、Colab、授權三處都要是它；不要搬動、改名 prereg03 的任何檔」。
- test02 也在 AI Pro 跑（`docs/COLAB_RUN.md:244`），所以兩次用同一個帳號、同一份 `C10_test_v1.zip`。

### 3.9 時間與運算單元（推估，用 1.6 的實測）

| 步驟 | G4 | A100（12 vCPU） |
|---|---|---|
| 掛載 Drive、授權、pip（缺套件才裝） | 0.5–1 分 | 同左 |
| prepare（不解開訓練資料）、env、preflight（讀 2 × 105 MB 權重並算 SHA、6 個參照、測試集 zip） | 約 20–30 秒 | 約 30–40 秒 |
| view_d4 單元檢查、載入權重 | < 5 秒 | < 5 秒 |
| seam_val：解析 7.8 秒＋兩臂 × 44.3 秒 | 約 1.6 分 | 解析約 30 秒（12 個進程）、GPU 推估 1–1.5 倍 → 約 2–2.5 分 |
| public、private：解析各約 2 秒＋兩臂 × 5,000 盤 × 9 次 × 0.26 毫秒 ≈ 2 × 11.6 秒 | 約 50 秒 | 約 1.2–1.5 分 |
| 接縫、收尾、zip（約 20 MB）、flush | 約 15 秒 | 約 15 秒 |
| **合計** | **約 3.5–4.5 分**（G4 約 8.7 單元／小時 → 約 0.6 單元） | **約 5–6 分**（約 5.4 單元／小時 → 約 0.5 單元） |

- 如果只算 s0 和 swap（不照 03 逐字），dump 的時間是 2/9，G4 合計約 2–2.5 分；但就失去逐字相同的核對（見 0.3）。
- 不含使用者開檔、確認 GPU 的時間。時間是推估，彩排時實測。

### 3.10 程式共用：`colab/build_test04_rank_nb.py`

- **兩個底稿**，都讀已 commit 的 notebook JSON，先 assert LF 的 SHA：
  - `T02 = colab/test02_rank.ipynb` `53337a7acc396bdc1f94b3cbdc7dd0ce859766511099e8004697ff0d6a487b05`（在 R20 指紋裡，只讀）
  - `P03 = colab/prereg03_rank_a.ipynb` `85e0c2ae3269c68ad166310efb2525b2d48ed1279c9cf6eb6b428e59e8693557`
  - 另外 assert P03 A、B 兩份只差 ARM 那一行（和 03 產生器的核對相同），所以用 A 當來源和用 B 等價。
- **挑格子**：依 cell metadata `c10` 的 key（T02：`header, setup, prepare, official_1, environment, official_2, preflight_t02, dump_tools, dump_set_*, seam_check, final_check`；P03：`view_d4`、`official_14`、`dump_tools`），不看位置。
  - 函式 `base_key()` 照 `build_test02_rank_nb.py:103-108` 的定義，但要能處理 P03 metadata 多一層 `base`，也就是 `cell_key()`（`build_prereg03_rank_nb.py:1908-1911`）。
- **修改規則**：只用 `rep()`（目標剛好一次）和 `cut()`（兩個標記各一次），照兩個舊產生器的紀律。每個改過的格子都要列進 `CHANGE_ROWS`，而且列出的格子都要真的改過。
- **不 import** `build_prereg03_rank_nb.py`、`build_test02_rank_nb.py`：兩個都在 import 時執行、寫檔。也**不**從凍結的 `tools/p0*` import 任何東西進 notebook。
- **逐字核對**（ast `get_source_segment`；不符就讓產生器停下）：
  - view_d4 整格等於 P03 的 view_d4 格；
  - `load_net`、`dump_val_p03`、`channels_last`、`_nchw_output`、`state_dict_sha256` 等於 P03 的片段；
  - `write_dump`、`dump_is_done`、`write_keys`、`require_gpu_for`、`pack_results`、`flush_drive`、`sgf_to_features`、`parse_questions`（含 `sgf_cols=`…`parsed=` 解析段）、`quick_check_questions`、`question_spans`、`c0_of`、`_argmax_first`、`rule_b`、`readonly_fingerprint`、`guard_write`、`_is_under` 等於 T02 的片段；
  - 三組格子只差第一行；
  - prepare 格等於 T02 的 prepare 格。
- **釘死值從 repo 讀出**，再和常數比對（同 `build_test02_rank_nb.py:122-138`）：
  - W_PIN：`rank_{a,b}/config.json` 的 `final_sha256` ＝ 完成標記 ＝ npz。
  - 接縫參照：repo 檔的 SHA ＝ 完成標記的 `sha256`；參照 GPU 取自完成標記。
  - TF32：`config.json` 的 `runs[0].training.tf32`。
  - 打包檔 manifest：`results/prereg02/rank/env.json`。
  - 另外和 `tools/p04_rank_fuse.py:169-175` 的 `SW_SHA256`、`SW_WEIGHTS` 交叉比對：用文字解析，不 import，因為那個模組 import 時會算 basis 等東西。
- **輸出**：
  - `colab/test04_rank.ipynb`：metadata 沿用 T02；cell id `c10-rank-t04-NN`；cell metadata `c10 = {'base_nb': 'test02'|'prereg03', 'base_cell', 'base_id', 'base', 'changed'}` 或 `{'new': key}`。
  - `results/test04/notebook_diff_rank.txt`：逐格 unified diff，對它自己的底稿格。
  - `NOTEBOOK_SHA256` 用 T02 的規則，路徑是 `colab/test04_rank.ipynb`。
  - 重跑產生器要逐位元組相同。CLI 同 T02：`[REPO] [--out-nb] [--out-diff]`。
- **上傳用的副本**：`data/colab_upload/test04/test04_rank.ipynb`，只有這一個檔。`C10_test_v1.zip` 沿用 test02 放在 Drive `C10` 的那一份。

---

## 4. 測試（`tests/test_t04_nb_rank.py`，照 `test_p03_nb.py`、`test_t02_nb_rank.py` 的做法）

- **Build**：
  - 兩個底稿的 SHA；重建逐位元組相同（寫到 tmp）；
  - cell id、編譯、`outputs == []`；
  - 答案檔的那個字只出現在沿用的 prepare 拒收行；
  - 沒改的底稿格和底稿逐字相同。
- **SegmentIdentity**：3.10 的全部片段，測試裡獨立用 ast 重算，不信任產生器。
- **Pins**：
  - W_PIN、6 個參照 SHA、參照 GPU、TF32、打包檔 manifest，都等於 repo 檔當場算的值；
  - `SW_SHA256` 和 PREREG_04 1.3 的表一致；
  - 重算 `NOTEBOOK_SHA256`。
- **NoTraining**：
  - AST 名稱裡沒有 `optimizer`、`df_train`、`df_val`、`T_TOTAL`、`make_train_loader`、`load_rows`、`CKPT_DIR`；
  - 程式碼裡沒有 `torch.autocast`、`bfloat16`、`tarfile.open`；
  - 每個被讀的名稱都有綁定（同 `test_t02_nb_rank.py:264-281`）。
- **CellFunctions**（CPU torch 2.14，先例是 `test_p03_nb.py:225-238` 用 exec 跑 view_d4）：
  - 用小的隨機 `GoRankResNet(501)`，對 3 題 × 5 盤的合成特徵：
    - 從 test04 notebook 原始碼 exec 出來的 `dump_val_p03`，和從 P03 notebook 原始碼 exec 出來的 `dump_val_p03`，在同一個網路、同一份輸入下，輸出的 npz **逐位元組相同**；
    - swap 等於 `net(dump_input(x, ~ib, 0))`，s0 等於 `net(dump_input(x, ib, 0))`；
    - 只讀目錄的寫入被 `guard_write` 擋下。
  - 接縫函式用合成陣列測：
    - 逐位元組相同 → 通過；
    - 同型 GPU、差 5e-5 → 通過；同型、差 2e-4 → 沒過；不同型、差 2e-4 → 通過；
    - key 不同、執行環境混用、權重 SHA 不符 → 沒過；
    - swap 的 C0 真的是對 `[n, 10]` 算。
- **Hygiene**：
  - `test_p04_hygiene.py` 照跑，R20 指紋裡的檔一個都不能動；
  - 產生器和測試都不 import `build_*` 產生器。

## 5. CPU 冒煙計畫（本機，不用 GPU，最多 6 個 worker）

先設 `OMP_NUM_THREADS=6`、`MKL_NUM_THREADS=6`（`run_local_smoke.py` 本身不限制執行緒）。

- **L1**：`.venv/Scripts/python.exe -m pytest -q tests/test_t04_nb_rank.py`，幾分鐘內跑完。
- **L2b，開發時用，約 12–15 分鐘**：做一個假的 `prereg03_smoke` 樹（scratch 腳本，或 tests 裡的 helper），**用凍結的 03 程式算參照**：
  1. `S/drive/C10_bundle_v1.zip`：`run_local_smoke.py` 會自己複製。
  2. `S/drive/results/prereg03_smoke/rank_{a,b}/ckpt/final.pth`：`torch.manual_seed(1)`／`(2)` 產生的隨機 `GoRankResNet(501)` state_dict。
  3. 用 exec 跑 P03 notebook 的 view_d4、model_tools、dump_tools 片段（`load_net`、`parse_val`、`dump_val_p03`、`write_dump`、`write_keys` 等），SMOKE 的全域照 03（`SMOKE_CFG['questions'] = 12`，Windows 上 `Pool` 換成 `multiprocessing.dummy`），算出前 12 題（142 盤）的 `dumps/val_keys.npz`、`val_p03x_d4.npz`、`val_p03x_s0_swap.npz` 和 `_DONE_`。
  4. 冒煙用的測試集 zip：
     - `tools/t02_fake_tests.py from-train --root F --shape official --slot-colours filled`
     - `tools/t02_test_bundle.py build --kind smoke --root F --tests-dir F/data/test --out-dir F/built/data/test`（同 `test_t02_dryrun.py:550-561`，只用 fit 玩家的棋），把 zip 複製成 `S/drive/C10_test_v1.zip`。
  5. `.venv/Scripts/python.exe colab/run_local_smoke.py colab/test04_rank.ipynb --scratch S`。
  - 預期：兩臂 4 個 dump 都 `bitwise_equal` 為真（同一台 CPU、torch 2.14、同一個函式），接縫通過，zip 裡的成員齊全。
  - CPU 時間：每次 forward 約 56 毫秒（03 冒煙 `final_dumps` 275 秒 ÷ 4,878 次 forward）。seam 142 盤＋public、private 冒煙各前 12 題約 150 盤，× 9 次 × 2 臂 ≈ 7,800 次 forward ≈ 7–8 分鐘，加上解析。
- **L2，合併前跑一次完整的鏈，約 65 分鐘**：
  1. `run_local_smoke.py` 依序跑 `prereg01_rank.ipynb --fresh`（548 秒）→ `prereg02_rank.ipynb`（1,012 秒）→ `prereg03_rank_a.ipynb`（813 秒）→ `prereg03_rank_b.ipynb`（809 秒），秒數取自 `results/prereg03/design/build_review.md:41-44`。
  2. 跑 test04 的冒煙。
  - 參照是真正由 03 notebook 冒煙產生的 dump，接縫預期逐位元組相同。
  - 同一個 scratch 也可以接著跑 `t04_rank.py … --smoke` 的整條鏈。
- **L3，選配，要使用者先同意下載**：請使用者從 AI Pro 的 Drive 下載兩個 `final.pth`（各約 105 MB）到 `data/prereg03_weights/`（不進 git，SHA 要等於 W_PIN）。
  - 本機 CPU 用 notebook 的函式，算前 100 題（約 1,200 盤）的 s0 和 swap（約 4,800 次 forward，約 5 分鐘），和已 commit 的 dump 對應列比。
  - 只記錄 max|Δlogit|、max|ΔC0|、max|ΔE|，這是 CPU fp32 和 GPU TF32 的差，**還沒有實測值**。
  - 用一個寬鬆的閘門抓接線錯誤：C0 ≤ 1e-2。接錯執色或視角時，C0 的差會是 0.05 以上的量級。
  - 這一步同時當權重的備份（風險 9）。
- **L4，建議在 K5 前做，Colab 彩排**：10/28–11/02 的 test02 彩排做完後、把彩排檔收進 `old_rehearsal`（COLAB_RUN 第 21 步）**之前**，在 G4 用同一份彩排 zip（kind rehearsal）跑 `test04_rank.ipynb`。
  - 結果寫到 `results/test04_rehearsal`，預期兩臂 `bitwise_equal` 為真。
  - 約 5 分鐘、約 0.7 單元，要使用者操作。
  - 實測時間寫進 COLAB_RUN。

## 6. 風險

| # | 風險 | 影響 | 對策 |
|---|---|---|---|
| 1 | bf16 不確定性 | 只影響訓練，權重已經用 SHA 固定。推論是 fp32、沒有 autocast | 測試禁止 notebook 出現 `autocast`、`bfloat16` |
| 2 | Colab 更新 torch 後改了 TF32 預設（新 API `fp32_precision`） | 同一型號 GPU 上接縫可能超過 1e-4 | preflight 把旗標設成 03 的值，預設值和實際用的值都記下（3.3.1 第 8 點） |
| 3 | Colab 的 torch／cuDNN／numpy 版本和 9/28 不同（03 是 2.11.0+cu128／91900／numpy 2.1.3） | 不再逐位元組相同（cuDNN 換了演算法，累加順序不同）；numpy 的 .npy 表頭可能不同，檔案 SHA 就不同 | 容許差 1e-4 應該綽綽有餘（預期 ~1e-6）；逐位元組相同比的是 logits 的位元組，不是檔案 SHA；env.json 會記版本 |
| 4 | cuDNN 的啟發式選演算法時受可用顯存影響（`benchmark=False`） | 03 算 dump 時 GPU 上還有訓練的狀態，test04 沒有 | 96 GB 顯存下不太可能影響；彩排會實測 |
| 5 | 退到 A100 時，TF32 kernel 的累加順序不同 | C0 預期遠小於 1e-3，但**沒有實測**；logits 的差放大到 E 或合併機率，可能讓本機規則 B 多差幾題 | G4 優先；本機合併接縫最多差 4 題把關；沒過就走重跑規則 |
| 6 | 中途斷線後換了 GPU 或執行階段接續 | 環境混用，接縫判定沒過 | 照 T02 的指示選同一種 GPU；沒過就把資料夾改名、整份重跑 |
| 7 | Drive 空間 | 輸出只有約 45 MB（dump 加 zip）；門檻 0.5 GB；9/28 時 AI Pro 還剩 190 GB | preflight 查空間；查不到就只提醒（同 T02） |
| 8 | 用錯帳號（Pro+ 或其他） | 找不到權重，還沒開始算就停 | 訊息寫明要用 AI Pro；授權時也要選 AI Pro（COLAB_RUN 照 test02 第 10 步） |
| 9 | **權重只有 AI Pro Drive 那一份**（zip 裡沒有、本機也沒有） | 被誤刪或 Drive 出問題時，04 整季都不能用 | 請使用者同意後下載備份（L3）；COLAB_RUN 和 TESTDAY_04 寫「`C10/results` 不要動」 |
| 10 | test02 同時在跑，或 test02 的結果目錄在執行中被改 | 只讀指紋不同，狀態記 `readonly_changed` | 使用者步驟：test02 跑完、刪掉執行階段之後才開 test04 |
| 11 | 放錯測試集 zip（彩排和正式混用） | MANIFEST 的 kind 把關；本機再比對 test02 和 test04 的 `test_bundle` SHA | 正式那天 `C10` 裡只能有一份 `C10_test_v1.zip` |
| 12 | 第 1–7 個對稱用不到卻照算 | 多 1.8 分鐘 G4；本機若誤用 8 種對稱會偏離登記 | 修訂寫明 apply 只讀 d4 的 `[:, 0, :]` 和 swap；`t04_rank.py` 的測試把關 |
| 13 | Colab 預裝了不同版本的 sgfmill | 解析結果不同 | `parsed` 和 key 的摘要、logits 的接縫會抓到；env.json 記版本 |
| 14 | RAM | seam_val 的特徵約 3.5 GB | 兩種 GPU 都夠；用完就 `QSETS.pop` |

## 7. 給其他設計者的介面

- **`t04_rank.py apply`**（見 3.2 的檔案清單）。讀 dump 時：
  - 讀：`p02_rank_fuse.load_dump(path, n, 8)` 取 `[:, 0, :]` 當 A0／B0；`p03_rank_fuse.load_swap(path, n)` 當 SWA／SWB。
  - log_softmax 用 float64，未解析的列遮成 0（同 `p04_rank_fuse.py:965-966`）。
  - 核對：`notebook_sha256`（用 T02 的重算規則）；`run_log.status == completed`，而且沒有任何一次失敗的狀態；`seam_check.passed`；tol 自己重算，而且必須等於 notebook 記的；`config.test_bundle` 等於 test02 已 commit 的 `config.json` 裡的值；12 個標記的環境相同。
- **本機假測試演練的 fake tree**（PREREG_04 5.1）：
  - 要照 3.2 的版面產生 `rank/<set>/dumps/…`：seam_val 直接複製已 commit 的 03 val dump；public、private 取和 `t02_fake_tests.py from-val` 相同的列切片；加上 `seam_check.json`、`config.json`（`smoke false`、`rehearsal true`、測試集 SHA、`notebook_sha256`）。
  - `t02_fake_tests.py` 是凍結的，要另寫（例如 `tools/t04_fake_tests.py`）。
- **`docs/COLAB_RUN.md` 新的一節「測試集（test04）」**，要點：
  - 帳號 AI Pro；前提是 test02 已跑完、下載好、執行階段已刪；
  - 只上傳 `test04_rank.ipynb` 一個檔（zip 沿用 `C10` 裡那一份）；
  - 確認三件事：頭像、檔名、第一格標題；
  - GPU 選 G4，灰色時選 A100；`device` 那一行的樣子；
  - 約 5 分鐘；下載 `results/test04/test04_rank_results.zip`，放到 `results\test04\`；
  - 出狀況時的表格照 test02 那一節；
  - 彩排的時機見 L4；
  - 「不要動 `C10/results/prereg03`」。
- **`docs/TESTDAY_04.md`（Claude 的步驟）**：解壓、`t04_rank.py opp`／`apply`、`check_submission --official`、上傳前要使用者說 OK（PREREG_04 5.3）。

## 8. 要在修訂裡定的事（建議值）

1. **接縫的量**：C0（每題 float32 平均 softmax），兩臂 × {s0, swap}，≤ 1e-4 或 1e-3；其餘只記錄（3.5）。另一個選項是用 E（log_softmax 平均）當閘門：比較直接，但在 A100 上比較容易誤判沒過，**不建議**。
2. **GPU**：只收 G4 或 A100，G4 優先（3.7）。
3. **重跑**：接縫沒過時，由 Claude 決定才重跑，最多一次（資料夾改名、新的執行階段），K6 前做完；再沒過就交 R20。
4. **8 種對稱照算**，本機只用 s0 和 swap（寫明是為了和 03 逐字相同，不是新的來源）。
5. **TF32 旗標**固定成 03 記下的值，並記錄 Colab 的預設。
6. **彩排**（L4）要不要列成 K5 的條件：登記的 K5 只要求本機假測試演練。建議把 Colab 彩排列為強烈建議，但不當硬性條件。
