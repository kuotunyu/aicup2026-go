# 事前登記 01：官方 baseline 在本機驗證集的分數，以及「期望得分」決策規則

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `3f0fb50` | 第一版，2026-09-24 |
| rev 2 | `2e2ad8a` | 9/24 稽核後修正，當時**還沒開始訓練、沒有任何方法打過分**：寫明和官方 notebook 的所有差異、內部驗證的固定切法、H1 的平手與缺值處理、合理性門檻、競賽一的 checkpoint |
| rev 3 | `b0af8ab` | 9/25，**仍未開始訓練、沒有任何方法打過分**：只對齊實際檔名與路徑（diff 分成 rank／player 兩份；結果放 `rank/`、`player/` 子資料夾）、寫明 `tools/decide.py` 的機率和防呆容許值、notebook 結尾另存不含權重的結果 zip。實驗內容與採用條件不變 |

## 要做什麼

### 1. 競賽二 baseline

照官方教學 `rank-prediction.ipynb`：模型 `GoRankResNet`、最後 250 個盤面、AdamW lr 1e-3、batch 256、5 個 epoch、seed 42。

**和官方 notebook 的差異只有下面幾處**。改好的 notebook（`colab/prereg01_rank.ipynb`，由 `colab/build_rank_nb.py` 產生）和逐格 diff（`results/prereg01/notebook_diff_rank.txt`）要先 commit，才能開始跑。

- ① **只讀訓練玩家**：cell 4 改用 `tools/c10data.py` 的 `load_rows(..., part="fit")` 讀訓練資料（5,767 人、720,307 盤），內部驗證用 `part="inner"`（640 人、73,556 盤），再各自轉成 DataFrame。`load_rows` 會 assert 人數、盤數，並確認沒有任何一盤出現在本機驗證檔裡。assert 的輸出要存進結果檔。
- ② **打散後才寫分片**：DataFrame 讀進來時是照等級、再照黑白排好的，官方的分片又是照列順序寫。所以取代 `df.iloc[:VAL_SIZE]` 之後，`df_train`、`df_val` 都要各自 `sample(frac=1.0, random_state=42)`。不打散的話，每個 batch 會只有一個等級。
- ③ **改 `VAL_SIZE`**：cell 2 的 `VAL_SIZE` 改成內部驗證實際寫入的盤數。官方會拿它比對讀出的筆數，數字不對會默默退回單執行緒。
- ④ **改推論**：cell 20 `predict()` 改讀本機的 `rank_val_questions.csv`，拿掉 public／private 迴圈。每題另外存 10 維平均機率（完整 float，不四捨五入）。

模型、特徵、超參數、seed 都不動。checkpoint 照官方做法，依內部驗證的逐盤 accuracy（官方的 `validate()`）選 `best_model.pth`。

### 2. 推論

每題每盤各自算 softmax，同一題取平均，得到一個 10 維機率向量，存成 `rank_val_probs.csv`（Colab 寫在 `MyDrive/C10/results/prereg01/rank/`，帶回本機放 `results/prereg01/rank/`）。

### 3. 兩種決策規則，都從同一份機率向量計算

- **A（官方）**：取機率最大的等級。
- **B（期望得分）**：取 k 使 `p_k + e^-1 × (p_{k-1} + p_{k+1})` 最大，超出兩端的鄰居算 0。機率如果準，B 就是期望得分最高的選法；但模型的機率不一定準，所以要實測。
- **平手**：取等級索引最小（較弱）的那一級，和 `numpy.argmax` 相同。
- **整題沒有可解析的棋譜**：A、B 都填 `D`（照官方做法），並記錄題數。
- **先 commit 再打分**：算 A、B 的腳本（`tools/decide.py`）在打分之前 commit。
- **防呆（rev 3）**：`decide.py` 在任何一題的機率和偏離 1 超過 1e-5、有負值、或 `n_parsed>0` 卻缺機率時整份報錯、不寫任何檔；它不會修正或重新正規化機率，所以不會改變任何一題的答案。

### 4. 競賽一 baseline

照官方教學 `player-identification.ipynb`：模型 `GoPlayerResNet`、triplet loss、超參數與 seed 都不動。notebook 是 `colab/prereg01_player.ipynb`（由 `colab/build_player_nb.py` 產生），逐格 diff 在 `results/prereg01/notebook_diff_player.txt`。差異只有兩處：

- ⑤ **只讀訓練玩家**：cell 4 在 `MIN_GAMES` 過濾之前，改用 `load_rows(..., part="train")` 只讀訓練玩家（6,407 人），並 assert `player_index.json` 裡沒有任何驗證玩家。
- ⑥ **改用本機題目**：cell 20 的候選庫和題目改讀本機的 `player_val_candidates.csv`、`player_val_questions.csv`，拿掉 public／private 迴圈，候選清單照玩家代號排序。

checkpoint 照官方做法，用訓練 loss 最低的 `best_model.pth`，不在本機驗證集比較各 epoch。

## 怎樣算成功

### H1：B 比 A 好

- 指令固定為 `python tools/score.py compare-rank pred_A.csv pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg01 H1"`，報告的差值是 B−A。
- **採用 B 的條件**：分等級平均差值的 95% 信賴區間下界 > 0。bootstrap 在每級內分層重抽 10,000 次、seed 0。
- 每題等權平均的差值也要報告；兩者結論相反時照 VALIDATION 的規則，不採用 B。
- 不成立的話，之後都用 A。之後的模型要再試 B，必須另寫事前登記，並寫明這是 B 的第幾次檢定。

### 合理性檢查（跑之前寫死）

- 競賽二 A 的分等級平均 < 0.25，視為異常，先查 bug。
  - 要能指出具體的程式錯誤才算 bug，分數低本身不算。
  - 修正後重跑，兩次分數、bug 內容、修正的 commit 都寫進結果檔和打分紀錄帳。
- 外洩不靠分數判斷，靠上面 ①、⑤ 的 assert。

### baseline 分數本身

沒有門檻，只當之後實驗的比較基準。

本機分數和官方 public（競賽二 0.4631、競賽一 0.2845）不需要一樣，原因：

- 官方 baseline 用全部 8,007 位玩家訓練；官方測試玩家如果在訓練集裡，官方的 baseline 會占便宜。
- 題目組成不同：各級題數、每題盤數、候選庫是否跨等級。
- 本機只用 72% 的玩家訓練（fit 5,767 人）。
- 官方按盤切的驗證只影響它選哪個 epoch。

## 各打幾次分

- 競賽二：A、B 各一次，用同一個模型、同一份機率向量。
- 競賽一：一次。
- 每次打分都會記進 `results/score_ledger.csv`。
- 這份登記沒寫的做法（改聚合方式、改溫度、融合模型等），一律另寫事前登記。

## 算力與偏離

- 在 Colab 跑，不用本機 4090。
- Colab 只上傳訓練 CSV、`players_split.csv`、`c10data.py`、題目檔和候選庫，**不上傳答案檔**。
- 如果 GPU 等級跑不完 5 個 epoch，可以減少 epoch 或抽樣訓練資料，但要在結果檔寫明實際用了多少，並標記「偏離官方設定」。
- 結果寫進 `results/prereg01/rank/` 與 `results/prereg01/player/`（Colab 結尾各自打包成不含權重的 `rank_results.zip`、`player_results.zip`，下載後解開到 `results/prereg01/`），內容包括：
  - 分數 JSON、每題機率向量
  - 實際設定、執行環境、訓練時間
  - 外洩 assert 的輸出
