# C10 下一輪實驗設計用的技術地圖（2026-09-26）

以下都是讀檔、跑本機 CPU 小計算得到的。只用 repo 內的 `score_ledger.csv` 和分數 JSON，沒開 `data/local/*answers*.csv`，沒跑 `score.py`，沒改 repo。

**引用格式**：repo 根目錄是 `D:\AI-Portfolio\competitions\aicup2026-go\`。`檔案:行` 指 repo 裡的檔案。官方 notebook 寫成 `rank-nb cN:Lm`（`rank-prediction.ipynb` 第 N 格第 m 行）和 `player-nb cN:Lm`（`player-identification.ipynb`），格號從 0 起算，和兩個 builder 的 `OFF[i]` 相同。標「推算」的數字是我用實測值算出來的，不是實測。

## 0. 最重要的八件事

1. **競賽二模型不知道目標玩家執哪一方。** 特徵只有「黑子／白子」兩層盤面，是絕對顏色，沒有執色層（`utils.py:53-64`）。抽特徵的 worker 也沒寫 `color`（`rank-nb c8:L12-18`），推論只讀 `sgf_` 欄。但官方測試檔明定有 `color_1..20`（`third_party/aicup2026_tutorial/README.md:94`），本機題目檔也有（`tools/make_val.py:63-72`）。
2. **每盤只取最後 250 個盤面**（`utils.py:57`）。每一級的手數 p75 都 ≥258（`results/data_inventory.json`），所以超過 25% 的棋 ≥250 手，開局看不到。
3. **競賽二是 fp32 訓練、學習率固定、沒有資料增強。** 內部驗證每盤 accuracy 依序是 21.47 / 19.72 / 23.22 / 21.76 / 25.15。train acc 18.4→26.3 持續上升，train loss 2.114→1.902，內部驗證 loss 沒有持續變差，看起來是訓練不足，不是過擬合。best＝最後一個 epoch。
4. **競賽二的平均機率堆在兩端。** A 規則的預測分布是 D 373、6D 264、3D 只有 20。對照各級題數 D 178、6D 141、3D 124（`rank/decide_summary.json`、`split_v1/manifest.json`）。校準或序數的空間很大。
5. **競賽二可以只重跑推論、不重訓，就拿到每盤 logits。** `best_model.pth` 是第 5 個 epoch，也就是最後一個，所以「用內部驗證選 checkpoint」幾乎沒有造成偏差。權重在使用者的 Google AI Pro 帳號 Drive 上。推算：內部驗證 73,556 盤約 2.5 分鐘推論。
6. **競賽一的 prereg01 用全部 6,407 位訓練玩家訓練，包含 640 位內部驗證玩家。** 所以它的權重不能拿來做「留出內部驗證」的實驗，也不能拿來 warm start。
7. **內部驗證只有 215 人有 ≥105 盤。** 湊不出 400 人×100 盤的模擬候選庫。
8. **G4 上的實測**：競賽二一個 epoch 436 秒，競賽一一個 epoch 8.4 分鐘。整份各約 45、51 分鐘，照 G4 每小時 8.71 單元推算約 6.5 和 7.4 單元。

---

## 1. 競賽二（棋力）模型

### 輸入特徵
`SGFParseRankPrediction(history_length=250)`（`utils.py:8-84`）：

- `_extract_game_states` 記錄每個節點「落子後」的盤面，含根節點的空盤，所以有 手數+1 個盤面（`utils.py:31-51`）。
- 取最後 250 個盤面。不足 250 的在前面補空盤（`utils.py:55-57`），所以 ≤249 手的棋，模型可以從補了幾個空盤間接看出手數。
- 每個盤面 2 層：黑子＝1、白子＝1（`utils.py:59-61`），共 500×19×19，存成 uint8（`utils.py:84`）。
- 棋盤在 `_prepare_board` 裡是轉置存的（`utils.py:26-28`），訓練和推論一致，沒有影響。
- 沒有執色層、輪誰下的層、最後一手的標記、氣。
- 一盤棋＝一筆樣本。docstring 寫「每一手一個特徵」是錯的。

### 架構
`GoRankResNet`（`network.py:19-48`），26,349,806 個參數（`rank/config.json`）：

- 500→256 的 3×3 conv、BN
- 20 個 residual block（256 通道）
- 1×1 conv 到 6 通道、BN
- Linear 2166→722 → ReLU → Dropout 0.5 → Linear 722→10
- 前向每筆 17.87 GFLOPs（`bench_colab_20260925.json`）

### 損失、最佳化、訓練設定
- `CrossEntropyLoss`，把 10 級當成獨立類別，沒有利用序數結構。
- `AdamW(lr=1e-3)`，其他用預設值（weight_decay 0.01）（`rank-nb c14:L1-3`，`config.json` 也有記錄）。
- 沒有 LR schedule、沒有 AMP（`build_rank_nb.py:921-965` 的迴圈裡沒有 autocast）。
- batch 256（`build_rank_nb.py:725`），5 個 epoch（`:727`），每個 epoch 2,814 步＝720,307 盤（`history.csv`）。
- webdataset：`shardshuffle=1` 加 `shuffle(1000)`（`rank-nb c11:L7-13`）。寫分片前 fit 和 inner 都先各自打散（`build_rank_nb.py:782-783`）。
- DataLoader 4 個 worker（`build_rank_nb.py:875`）。沒有 8 種對稱的資料增強。

### 選 checkpoint
- 內部驗證 640 人、73,556 盤的「逐盤」accuracy 最高者（`rank-nb c15`、`build_rank_nb.py:954-959`），同分取較早的 epoch。
- 這是逐盤正確率，不是比賽計分方式。
- `validate()` 只回傳 loss 和 accuracy，不保留輸出（`rank-nb c15:L5-13`）。

### 推論時一題怎麼聚合
- 每盤各自 softmax，同一題取算術平均（`build_rank_nb.py:1042-1044`，對應 `rank-nb c20:L40`）。
- A＝argmax；B＝使 `p_k + e^-1(p_{k-1}+p_{k+1})` 最大的 k（`tools/decide.py`，已採用 B）。
- 解析失敗的盤會被丟掉，丟掉後就對不回是第幾個欄位（`build_rank_nb.py:1031`）。prereg01 的 19,156 盤全部解析成功。

---

## 2. 競賽一（辨識玩家）模型

### 特徵
`SGFParsePlayerIdentification(history_length=8)`（`utils.py:87-202`）：

- 從目標玩家的角度組 17 層：8 層「自己的子」加 8 層「對手的子」，取最近 8 個盤面；再加 1 層執色（黑＝全 1，白＝全 0）（`utils.py:91-116`）。
- 盤面在落子「之前」加進歷史（`utils.py:181`），落子在後（`:196`）。所以每個窗口只有目標玩家面對的局面，**不含他這一手下在哪裡**。
- 只在輪到目標玩家、且歷史滿 8 個盤面時才產生窗口（`utils.py:184`），平均每盤 108 個窗口（bench）。

### Embedding
`GoPlayerResNet`（`network.py:50-80`），4,775,744 個參數：

- 17→128 conv、16 個 block、global average pooling
- MLP 128→64→256，再 L2 正規化（`network.py:79`）
- 輸出 256 維。docstring 寫 128 是錯的。

### Triplet 取樣
`TripletDataset`（`player-nb c11:L8-35`）：

- 種子＝`SEED + 1000·epoch + worker_id`。
- 每一筆：均勻抽兩位不同玩家（正例、負例）→ 正例玩家抽兩盤不同的棋（anchor、positive）→ 負例玩家抽一盤 → 每盤隨機取一個窗口。
- 每次取窗口都要把整盤解壓，一盤 647 KB。
- 玩家等權，不管他有幾盤。沒有 hard negative mining，負例常是不同等級的人，很容易分。

### 損失與訓練
- `TripletMarginLoss(margin 0.5, p=2)`，`AdamW(lr=1e-4)`，學習率固定（`player-nb c14`）。
- 三路串成一次 forward，用 bf16 autocast（`player-nb c15`）。
- batch 100 個 triplet（300 個窗口）。
- 官方設定是每個 epoch 15,000 步 × 5 個 epoch。實際是 14,976 步：48 個 worker 各分到 31,250 筆，`drop_last` 每個 worker 丟掉 50 筆。

### 選 checkpoint
- 訓練 loss 最低的 epoch（`player-nb c16:L26-29`），每個 epoch 是 0.4712→0.3566，還在下降，所以選到最後一個。

### 推論
- 候選重心＝這位候選「所有盤的所有窗口」embedding 的平均。這是依窗口加權，而且**平均後沒有重新正規化**（`player-nb c20:L26-42`）。題目向量同樣是所有窗口的平均。
- 用 L2 距離比較 `np.linalg.norm(C - q)`（`build_player_nb.py:944`），取 argsort 前 5 名。
- 候選清單照玩家代號排序。查詢沒有任何可解析棋譜的題，照官方填排序後的前 5 位。
- 另外存了 `player_val_embeddings.npz`：400 個重心和 400 個題目平均，完整 float32（`build_player_nb.py:955-959`）。

---

## 3. 各 builder 要改哪裡

### 兩份共通
- 路徑和名稱寫死：
  - 結果目錄 `prereg01`：rank `build_rank_nb.py:259`（特徵備份 `:262`）、player `build_player_nb.py:170`。
  - 打包檔 `C10_bundle_v1`：rank `:385`、player `:171-172` 和 `:239`。
  - 輸出 notebook 路徑：rank `:24-25`、player `:15`。player 的 diff 路徑在 `make_player_diff.py:11-14`。
  - player 的 `ROOT` 寫死 Windows 路徑（`build_player_nb.py:13`）；rank 從命令列參數讀（`:22`）。
- 改動要登記：
  - rank：每一格有改的官方格，都要登記在 `DEV_ROWS`，否則 assert 失敗（`:1126-1135`）。**官方 cell 14（optimizer）目前沒改**，要加 `MOD[14]`。
  - player：用 `rep()` 做字串替換，目標字串必須剛好出現一次（`:39-43`）。
  - 每個 code 格都必須能 compile，而且不能出現 `answers` 字串（`:1055-1058`）。
- **結果 zip 會跳過 `.pth`、`.pt` 和任何大於 100 MB 的檔**（rank `:697`、player `:1010`）。每盤的輸出檔要小於 100 MB，否則會被悄悄漏掉。
- 冒煙測試（`run_local_smoke.py`）：
  - 只能用 CPU 版 torch（`:73-74`）。
  - 在 Windows 上，每一格跑完後才把 `Process`、`Pool` 換成執行緒版（`:99-103`）。新增的格不要自己再 `import Pool`，否則在 Windows 會失敗。
- 打包檔：
  - 新加的輔助 `.py` 要加進 `make_colab_bundle.py:24-33`。
  - 改了 `c10data.py` 要三件事一起做：重打包、重新產生兩份 notebook（player 在產生時就把 SHA-256 寫進 notebook，`build_player_nb.py:28-36`）、使用者重新上傳 247 MB。
  - 打包檔內容一變，rank 特徵備份的指紋就對不上（`build_rank_nb.py:829` 含 `MANIFEST['files']`），會重抽特徵，約多 6 分鐘。

### (a) 訓練更久／加 LR schedule

**競賽二**
- 常數：`build_rank_nb.py:725-727`。
- 建 scheduler：新增 `MOD[14]`。
- 每步或每個 epoch 呼叫 `scheduler.step()`：放在 `:925-943`。總步數用 `train_counts['written']/BATCH_SIZE` 算，一個 epoch 2,814 步。
- scheduler 狀態要存進 `save_state`（`:618-624`）、從 `load_state` 讀回（`:627-648`），`STATE_CONFIG`（`:589-590`）也要加上 schedule 參數。
- AMP 加在同一段迴圈，要另外登記。
- **warm start**：prereg01 的 `ckpt/state.pt`（epoch 5，含 AdamW 狀態，只用 fit 訓練，所以內部驗證仍然乾淨）可以接著訓練。但 `load_state` 遇到 `NUM_EPOCHS` 不同會拒絕，要另寫一條「從 prereg01 初始化、寫到新 ckpt 目錄」的路徑。
- 特徵可以從 `MyDrive/C10/results/prereg01/features_rank/` 還原，前提是打包檔沒變、`FEAT_BACKUP` 指向那裡（還原邏輯在 `:542-583`）。

**競賽一**
- 常數在官方 cell 2 的替換段（`build_player_nb.py:445-481`）。
- optimizer 和 scheduler：`:806-815`；迴圈：`:820-865`。
- scheduler 狀態存進 `save_train_state`（`:743-753`），參數加進 `TRAIN_FINGERPRINT`（`:666-668`）。
- 要恢復成剛好 15,000 步：讓 `1,500,000 / worker 數` 能被 100 整除，例如 40 個 worker；或改 `player-nb c11:L24`。
- 有 schedule 時，「訓練 loss 最低」幾乎一定選到最後一個 epoch。要用有意義的方式選 checkpoint，需要 (d)。

### (b) 存每盤的輸出

**競賽二，內部驗證**
- 不用改分片。val loader 會回傳 `(x, label, game_name)`，`game_name` 就是 `game_id`（`rank-nb c8:L13-17`）。用 `(LEVELS[label], game_name)` 當 key，接回記憶體裡 `df_val` 的 `player_id`、`color`（`build_rank_nb.py:772`）。
- 做法：在訓練後新增一格，對 `val_loader` 跑模型、寫出每盤 10 維 logits，約 73,556×10 個數。
- 內部驗證本來就是拿來調參的，可以每個 epoch 都存。

**競賽二，驗證題**
- 在 `MOD[20]` 保留 `out`（`:1041-1042`），寫 `(question_id, slot, color, logits)`。
- 解析失敗時要保留欄位索引，不能用 `:1031` 那種過濾。
- 只能用登記指定的 `best_model.pth`（紀律 8）。

**競賽一**
- `mean_embeddings`（`player-nb c20:L26-42`）照 key 加總。把 key 改成每盤一個（候選用 `game_id`，題目用 `(question_id, slot)`），並存窗口數，才能還原官方「依窗口加權」的平均。
- 大小：4.49 萬盤×256×4 B ≈ 46 MB；內部驗證 7.36 萬盤 ≈ 75 MB。都小於 100 MB。
- 每個窗口都存會到 GB 級，會被 zip 規則跳過。
- 現在的 npz 只存平均（`:955-959`）。

### (c) 內部驗證的模擬題
- 照 `make_val.py:126-133` 出題：每人 `k=min(randint(5,20), n)`、`rng.sample`，欄位格式照 `:63-72`。
- **最便宜的做法**：Colab 只存內部驗證 73,556 盤的每盤 logits，模擬題在本機 CPU 反覆抽，可以抽很多輪，不用再跑 Colab。
- 可出題的人數：640 人中 636 人 ≥5 盤。各級：D 71、C 65、B 83、A 123、1D 34、2D 49、3D 49、4D 52、5D 53、6D 57。一輪平均約 7,644 盤（推算）。
- 限制：各級只有 34–123 人。多抽幾輪只降低「抽到哪幾盤」的變異，降不了「是哪些人」的變異。
- 如果 checkpoint 也用內部驗證挑，拿同一份資料調校準會有樂觀偏差。可以固定用最後一個 epoch，或把內部驗證對半分。

### (d) 競賽一留出內部驗證玩家
- `build_player_nb.py:540` 的 `part='train'` 改成 `'fit'`，`:543` 的 EXPECTED 跟著改；另外讀 `part='inner'`。
- 外洩檢查格（`:626-653`）要加上「索引裡沒有內部驗證玩家」的 assert。
- 內部驗證的棋不進 `player_index.json`；它們的 embedding 可以照候選庫的方式邊解析邊算（`player-nb c20` 那條路）。
- fit 有 3 位只有 1 盤的玩家會被 `MIN_GAMES` 濾掉，索引會是 5,764 人（推算）。
- 模擬候選庫受人數限制：內部驗證 ≥105 盤的只有 215 人（各級 16–26），≥100 盤 226 人。池子只有 215 人，比 400 人容易，分數不能和本機驗證的絕對值比。
  - 替代方案：縮小每位候選的盤數（≥50 盤 341 人、≥60 盤 305 人、≥80 盤 255 人），或另外開一份更大的競賽一專用留出集。後者會讓兩場的 fit 不一樣。

---

## 4. 執行時間、成本與限制

### G4 實測（RTX PRO 6000 Blackwell 96 GB、48 vCPU、190 GB RAM）

| 階段 | 競賽二 | 競賽一 |
|---|---|---|
| 準備 | 18 秒 | 18.9 秒 |
| `load_rows` | 18 秒 | 9.9 秒 |
| 抽特徵 | 355 秒 | 283 秒 |
| 特徵備份到 Drive | 5.7 秒 | 不備份 |
| 數 worker | 19.8 秒 | — |
| 每個 epoch | 436–438 秒 | 8.38–8.41 分鐘（訓練合計 2,518 秒） |
| 推論 | 36.6 秒（1,589 題、19,156 盤） | 258 秒（候選 4 萬盤＋題目 4,935 盤） |
| 合計 | 2,678 秒 | 3,074 秒 |

推算的速度：

- 競賽二：每步 155 ms、每秒約 1,650 盤、實際約 88.6 TFLOP/s（fp32/TF32）。DataLoader 4 個 worker 的上限約每秒 4,900 筆（0.81 ms/筆），所以瓶頸比較像在 GPU，但沒量過。
- 競賽一：每步 33.6 ms（約 3.1 TFLOP/步）。CPU 解壓約 473 ms/步（單核，bench），48 核分攤後 ≥9.9 ms/步。瓶頸在哪沒量過。

### Bench（本機單核，當時 CPU 很忙，數字偏保守）
- 競賽二：每盤解析 29.7 ms，gzip6 2.47 ms，原始 176 KB／壓縮後 1.57 KB，每筆解碼 0.81 ms。
- 競賽一：每盤解析 22.1 ms，gzip6 11.9 ms，原始 647 KB／壓縮後 13.0 KB，每盤 108 個窗口，每步 473 ms CPU，每個窗口 3.42 GFLOPs。

### 成本（每小時單元數是二手實測：G4 8.71、A100 5.40、A100 80GB 7.52、L4 1.71、H100 18.05）
- 競賽二多一個 epoch 約 1.06 單元，競賽一約 1.22 單元，全部重跑約 14 單元。
- 只重跑競賽二推論（內部驗證加驗證題）約 1 單元以下（推算）。
- **prereg01 是用 Google AI Pro 帳號跑的**，這個方案每月給多少單元沒有公開資料。權重和特徵備份都在那個帳號的 Drive。換到 Pro+ 帳號的話，warm start 和特徵還原都要先把檔案搬過去。

### 限制
- 其他 GPU 只有 12 vCPU，RAM 是 A100 83.5 GB、A100 高記憶體版 167 GB、L4 53 GB。
- G4 本機磁碟剩 200 GB，inode 剩 1,526 萬。競賽一的特徵約 79 萬個小檔、約 10 GB（推算），不備份，斷線要重抽約 5 分鐘。
- 競賽二的特徵備份約 1.2–1.35 GB（推算）。
- 競賽二每個 `.pth` 約 105 MB，`state.pt` 約 316 MB，每次跑約 0.95 GB，每多一個 epoch 多 105 MB。競賽一每個 `.pth` 約 19 MB。
- `COLAB_RUN.md` 要求 Drive 至少剩 4 GB。
- Pro+ 一次執行最長 24 小時，但有人回報 1.5–3 小時就被斷線；閒置約 90 分鐘會被回收（非官方說法）。同時能開幾個執行階段沒有公開。

---

## 5. 官方 baseline 沒用到的便宜訊號

**可以用：**
1. **目標玩家執黑還是執白**：訓練檔的 `color` 欄、測試檔的 `color_1..20` 都有，競賽二完全沒用。每一級剛好 5 萬黑、5 萬白，所以執色本身不透露等級，但可以讓模型「只看目標玩家下的棋」。本機規則只禁 `num_games`（`VALIDATION.md:46`）、`game_id`（`DATA.md:49`）和檔案列順序（`VALIDATION.md:45`）。
2. **開局**：超過 25% 的棋看不到開局。可以改成取前段加後段，或跳著取盤面。
3. **8 種對稱**：兩個 baseline 都沒做資料增強，推論時也沒做對稱平均。
4. **手數**：訊號很弱。平均從 D 227.8 到 6D 213.5，中位數 235→213，各級的分布大量重疊。模型已經從補空盤間接看得到（只限 <250 手的棋）。
5. **最後一手是誰下的**：這是假設，沒驗證。資料裡沒有 pass、沒有勝負，所以最後一手的奇偶配上執色，可能是「目標玩家有沒有贏」的代理。可以只用 fit／inner 驗證。
6. **聚合方式和校準**：可以換成平均 log 機率、溫度、序數模型；競賽一可以換成 cosine、重心重新正規化、每盤等權。改聚合要另寫事前登記（`PREREG_01:82`）。
7. **每人盤數不平均**：訓練被多盤玩家主導，前 10% 的玩家占 44% 的盤。`DATA.md:51` 建議限制每人最多幾盤。
8. **跨題任務**：候選各有 100 盤，可以用競賽二模型估每位候選的等級，拿來重排競賽一的 Top-5。本機候選庫跨等級（每級 32–48 人），官方是否跨等級未知。要登記。
9. **候選庫本身**：測試時可以拿到、不含答案，可以當無標註資料用，要登記（`VALIDATION.md:24`）。

**不存在：**
- SGF 只有 SZ、B、W 三種屬性（`DATA.md:39`）：沒有貼目、讓子、勝負、時間設定、名稱、日期。黑棋永遠先下，沒有讓子擺子。

**要登記或不確定：**
- 一題對一人的指派（`VALIDATION.md:47`）。
- 測試玩家是否也在訓練集裡：11/04 才知道。如果重疊，「認出這個人」就等於直接知道等級（`DATA.md:75-78`）。
- 外部資料或預訓練圍棋引擎（例如 KataGo）：C10 規則原文只查到「要揭露生成式 AI，並說明外部資料的貢獻」（`_research-2026-09-23/evidence.md:45`）。「允許使用外部資料和預訓練模型」只查核過 C1，C10 沒確認。

主要檔案：
- D:\AI-Portfolio\competitions\aicup2026-go\colab\build_rank_nb.py
- D:\AI-Portfolio\competitions\aicup2026-go\colab\build_player_nb.py
- D:\AI-Portfolio\competitions\aicup2026-go\colab\run_local_smoke.py
- D:\AI-Portfolio\competitions\aicup2026-go\tools\c10data.py
- D:\AI-Portfolio\competitions\aicup2026-go\tools\make_colab_bundle.py
- D:\AI-Portfolio\competitions\aicup2026-go\tools\make_val.py
- D:\AI-Portfolio\competitions\aicup2026-go\third_party\aicup2026_tutorial\utils.py
- D:\AI-Portfolio\competitions\aicup2026-go\third_party\aicup2026_tutorial\network.py
- D:\AI-Portfolio\competitions\aicup2026-go\third_party\aicup2026_tutorial\README.md
- D:\AI-Portfolio\competitions\aicup2026-go\results\prereg01\RESULTS.md
- D:\AI-Portfolio\competitions\aicup2026-go\results\prereg01\rank\timings.json
- D:\AI-Portfolio\competitions\aicup2026-go\results\prereg01\player\timings.json
- D:\AI-Portfolio\competitions\aicup2026-go\results\prereg01\rank\decide_summary.json
- D:\AI-Portfolio\competitions\aicup2026-go\results\bench_colab_20260925.json
- D:\AI-Portfolio\competitions\aicup2026-go\results\colab_facts_20260925.json