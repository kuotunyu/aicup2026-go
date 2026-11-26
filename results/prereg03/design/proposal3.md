# 事前登記 03 提案：排程與風險優先

## 摘要

- **範圍**
  - 03 只登記競賽二的一個新方法：recipe v2 從零重訓，再和 prereg02 的 C5 合併。
  - 競賽一在 03 不登記新方法，也不打分。這場的預算維持用了 1 次、共 4 次。理由見第 2 節。
- **最壞情況**：任何一步出問題，都照截止點退回 prereg02 R20，本機驗證 0.5891。test02 測試日工具一個位元組都不動，11/04 仍然照 `docs/TESTDAY.md` 先上傳 R20／Q04，03 是之後才加上去的一次上傳。
- **截止點**
  - C1 10/16：Colab 結果 commit。
  - C2 10/19：H3R 打分。
  - C3 11/03：test03 工具和 rev N 都要 commit，dry run 也要過。
  - C4 11/16：03 在測試集要有一次成功的正式執行。
  - 任何一個沒趕上，03 這一季就不用，R20 維持最終上傳。
- **使用者要做的事**
  - 10/06–10/08 跑一次 Colab，約 1–1.6 小時，約 7–14 個運算單元。跑兩個 seed 就是 14–28 單元。
  - 選做：10/28–11/02 彩排，比原本多約 20 分鐘。
  - 11/04 當天多跑一份 notebook，約 10–15 分鐘、約 1.5 單元。
- **打分預算**：H3R 打完後，競賽二用了 2 次，共 4 次。inner 閘門沒過就不打分，這次機會保留。

---

## PREREG_03 草稿內容

檔名：`docs/PREREG_03_RECIPE_V2.md`

### 修訂紀錄（格式同 PREREG_02）

| 版本 | 內容 |
|---|---|
| rev 1 | 登記本文，當時沒有任何 prereg03 的 notebook、輸出或分數。設計紀錄放在 `results/prereg03/design/`，內容是三份調查報告，加上只用 inner 的診斷腳本和 JSON：`colour_diag`、`length_diag`、`r20_diag`、`far_errors`、`feat_signal`，由 scratch 複製進來 commit；npz 不進 git。 |
| rev 2 | 程式寫完、冒煙測試和審查之後、使用者跑 Colab 之前，記錄實作時的偏離。 |
| rev N | 只在 H3R 採用時才寫：登記 test03 測試日程序，必須在 11/04 11:00 前 commit。 |

### 0. 為什麼做（都不看答案）

- 校準和決策層已經榨乾。以下各項在 inner 交叉驗證都拿不到 +0.002 以上：重新校準、完整矩陣校準（交叉驗證下 +0.0008）、既有權重的對稱或 epoch 組合、SGF 特徵。
- 神諭上限：從規則 B 的前兩名裡永遠選對是 0.795，表示剩下的差距在「分辨相鄰兩級」。
- 過擬合：prereg02 從 epoch 5 到 9，訓練 accuracy 26.3% → 33.9%，inner 只有 25.15% → 25.58%，inner loss 從 1.930 升到 1.980。
- 網路沒有學到旋轉不變：s0 的逐盤 accuracy 25.6%，另外 7 種對稱只有 21.4–23.6%。
- 模型不知道目標玩家執哪一色。
- 所以 03 要改的是每盤棋的表示：目標玩家視角、訓練時隨機套 D4、cosine 排程。
- 以下診斷只用 inner，只當背景，不拿來調參：
  - 依執色分開的輸出偏移沒有用：NLL 1.9395，共用一組是 1.9362。
  - 長棋沒有比較差：≥ 300 手的 accuracy 是 .276。
  - 所以不加長盤面歷史，不重抽特徵。

### 1. 競賽二

#### 1.1 Colab：`colab/prereg03_rank_s{1,2}.ipynb`（寫死，rev 1 凍結）

- **產生方式**：`colab/build_prereg03_rank_nb.py` 同時產生 s1、s2 兩份。builder 會 assert 兩份只差 `SEED = 1`／`SEED = 2` 這一行。使用者不需要改任何格子。
- **資料**
  - 只用 fit 玩家（5,767 人、720,307 盤），在 notebook 裡用 `load_rows(part="fit")` 讀。
  - inner（640 人、73,556 盤）只拿來監看和產生逐盤輸出。
  - 本機驗證題只做推論，沒有答案。
- **特徵**：用 prereg01 現有的 500 層 uint8 分片，不重抽，特徵指紋要和 prereg01 相同。
  - AI Pro 帳號：從 prereg01 的 Drive 備份還原。
  - Pro+ 帳號：重抽，指紋仍然要相同。
- **目標玩家視角**
  - 執色一律用 `(rank, game_id)` 去 join `df_train` 的 `color` 欄。不用 game_id 的規則推，因為 DATA.md 第 49 行禁止拿 game_id 當特徵。
  - 在 GPU 上翻色：第 2t 層是目標玩家的子，第 2t+1 層是對手的子，再加 1 層「目標玩家執黑＝1」，共 501 層。
  - 模型：`GoRankResNet(in_channels=501)`，其餘和官方相同（20 個 block × 256 通道）。`network.py` 不改。
- **訓練時的 D4**：每筆樣本從 8 種對稱等機率抽一種。用 [8, 361] 的座標排列表加 `gather` 實作，並 assert 結果和 `d4(x, s)` 相同。抽樣用 GPU generator，種子是 `SEED*1000 + epoch`。
- **最佳化**
  - AdamW，lr 1e-3，wd 0.01，batch 256，loss 是 CE（不加 label smoothing）。
  - 前 1,000 步線性 warmup，接 cosine 降到 0。
  - 總步數 T ＝ 12 × `expected_train_steps()`，依實際的 worker 數算出來，存進 state。
- **epoch 與 checkpoint**：12 個 epoch，**指定 checkpoint 是 `epoch_12.pth`**，不另選 best。inner 只當監看。
- **精度**
  - 訓練用 bf16 autocast 加 channels_last，權重維持 fp32，不用 GradScaler。
  - 逐盤輸出一律用 fp32、不開 autocast。
- **loader**
  - worker 把 uint8 傳到 GPU 上再轉型。
  - worker 數 ＝ min(16, vCPU − 2)。
  - shard 和樣本的洗牌都傳固定的 seed。確切 API 以 webdataset 1.0.2 為準，記進 rev 2。
- **接續**
  - 每個 epoch 存一次 `state.pt`：模型、optimizer、scheduler、global_step、RNG 狀態，以及 recipe 設定（worker 數、T、AMP、增強方式）。
  - 設定不同就停下，沿用 prereg02 的做法。
  - `.pth` 只留最後兩個。
- **Drive**
  - 輸出寫到 `C10/results/prereg03/rank_s{1,2}/`。
  - prereg01、prereg02、test02 的目錄一律只讀：擋所有寫入，開頭和結尾比對指紋。
- **GPU 關卡**：只接受 G4 或 A100，型號不符就在計算前停下。拿到 A100 時 worker 最多 10 個。

#### 1.2 notebook 內的檢查（寫死）

| 檢查 | 不過時 |
|---|---|
| 執色 join：每一筆訓練樣本都查得到、執色是 B 或 W；訓練集執黑比例在 0.50 ± 0.01（訓練集是 0.5006） | 停止，不訓練 |
| 翻色的單元測試：執黑的樣本不變，執白的兩層對調；D4 用 gather 的結果和 `d4()` 相同 | 停止 |
| 開頭 300 步量每步時間，印出推估總時間 | 只記錄 |
| loss 出現非有限值 | 停止，狀態記 `nan_stop`。唯一的備案：rev 2 以後再寫一版修訂，把 `AMP='fp32'` 整份從頭重跑。不接續，也不混用兩種精度 |
| 最後一個 epoch 的 inner 逐盤 accuracy（原方向）≥ 24.15%（和 PREREG_02 的門檻相同） | 這個 seed 不能用 |
| 把執色對調後的 inner accuracy | 只記錄：對手的等級通常和目標玩家相近，差距可能很小，不拿來判定 |
| 12 個 epoch 都跑完 | 沒跑完，這個 seed 不能用 |

- **哪些 seed 能用**：s1 能用就是 C6。s1 不能用而 s2 能用時，改由 s2 當 C6，所有要兩個 p03 模型的組合都拿掉。兩個都不能用：03 記為「未送分」。

#### 1.3 逐盤輸出（結果 zip `prereg03_rank_s{k}_results.zip`）

- `inner_keys.npz`：必須和 prereg02 的逐陣列相同。
- `inner_p03s{k}_d4.npz`：logits 是 float32 [73556, 8, 10]。
- `val_keys.npz`：必須和 prereg02 的逐陣列相同。
- `val_p03s{k}_d4.npz`：[19156, 8, 10]，解析失敗的那盤整列都是 NaN。
- 其他：`_DONE_*.json`、`history.csv`、`config.json`、`env.json`、`timings.json`、`run_log.json`、唯讀指紋、權重 SHA。
- 輸出前先做視角轉換，再做 D4。格式要讓 `p02_rank_fuse.load_dump`、`load_val_keys` 可以原樣讀。

#### 1.4 本機合併與選擇（`tools/p03_rank_fuse.py`；只用 inner）

- **程式**
  - 只 import p02 的數值函式：`basis`、`fit`、`predict`、`pooled`、`PlayerBootstrap`、`load_dump`、`build_qset` 等。
  - `p02_rank_fuse.py` 被 PREREG_02 第 9 節凍結，不改。
  - 假題沿用 `pseudo_inner.json`，開局沿用 `opening_inner.npz`、`opening_val.npz`，都已 commit，不重算。
- **來源**（都是各方向 log_softmax 的等權平均，不加新參數）
  - C5：p02e9，8 種對稱。
  - C6：p03 s1，8 種對稱。
  - C6′：p03 s2，8 種對稱。
- **組合**：一律用 log 合併、校準（a、γ、b）、開局項 w、規則 B。順序固定，也是平手時的優先順序。

| 組合 | 來源 | 需要 |
|---|---|---|
| S00 | C5（＝R20 重算） | 自我檢查 |
| S01 | C6 | — |
| S02 | C5 ⊕ C6（16 個方向） | — |
| S03 | C6 ⊕ C6′ | s2 能用 |
| S04 | C5 ⊕ C6 ⊕ C6′ | s2 能用 |

- **自我檢查**（不過就停下修程式，修好前不 tune）：
  - S00 的 inner 5 折交叉驗證分等級平均 ＝ 0.5854702521055817，容許差 ≤ 1e-12。
  - S00 用全部擬合資料重擬合的參數，和 `results/prereg02/rank/frozen_config.json` 相差 ≤ 1e-12。
- **選法**
  - 5 折交叉驗證的分等級平均最高者；和最高分差距在 0.003 以內的，取清單中最前面的。
  - 因為 S00 排第一，新組合要比 R20 高超過 0.003 才會被選到。
- **送分閘門**：以下兩個都成立才在本機驗證集打分，否則記為「未送分」，打分機會保留，R20 照舊。
  - (i) 選到的組合 − S00 ≥ **+0.010**。
  - (ii) 同一個差值在 inner 的 PlayerBootstrap 95% CI 下界 > 0（每級內重抽玩家 1,000 次、`default_rng(0)`、配對）。
- **盤數不能變成等級特徵**：沿用 PREREG_02 1.2 的溫度形式；報告各盤數分箱的預測分布。
- **備案參數（沒有開局項）**：選到的組合拿掉開局項，照 R10 的做法在全部擬合資料上擬合，和 `frozen_config.json` 一起 commit。它的 inner 交叉驗證分數也要記錄，第 5 節要用。
- **順序由程式強制**：
  1. Colab 結果 commit。
  2. `tune`。
  3. `frozen_config.json` commit。
  4. `apply`、`decide.py`、`check_submission.py`，然後 commit。
  5. 閘門過了才打分。
  - `tune` 不讀任何 `val_*` 檔。

#### 1.5 H3R：本機驗證集只打一次分

```
.venv\Scripts\python.exe tools/score.py compare-rank results/prereg02/rank/pred_B.csv results/prereg03/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg03 H3R (A = prereg02 R20, reference re-read)" --json results/prereg03/rank/score_H3R.json
```

- **採用條件**：以下兩個都要成立。
  - 分等級平均差值（03 − R20）的 95% CI 下界 > 0：每級內分層重抽 10,000 次、seed 0。
  - 每題平均差值的點估計不 < 0。
  - 兩個指標結論相反，就記為「結論依指標而定」，不採用。
- **對照組重讀**：A 側要和帳上 prereg02 R20 那一列比。
  - 條件：完全猜中率、差一級率相同，而且分數相差 ≤ 1e-12。帳上的值是 0.5890831292289003 和 0.591969684316557。
  - 打分用 `.venv`（Python 3.12），版本寫進 RESULTS。
- **合理性門檻**：新方法 < 0.55 時先查 bug。R20 是 0.5891，被選中的組合在 inner 已經 ≥ 0.595。要指出具體的程式錯誤才算 bug；找不到就照結果判定。
- **檢定力**（推估）：
  - 驗證集配對 CI 的半寬大約 0.01–0.02。依據是 inner 上 R20−R10 的 CI 半寬約 0.012，以及 02 在驗證集的 CI 半寬約 0.020。
  - 所以真實增益在 +0.01 左右時，很可能判「沒有改善」。這是刻意保守，不因此再打分。
- **不採用時**：R20 維持最終方法；不改組合重打；p03 的權重留在 Drive，給以後的登記用。

### 2. 競賽一：03 不登記新方法

- **原因**
  1. 本機 0.8830 的主力是「同一位玩家的開局習慣」加上隨機切。主要風險是官方怎麼切，這一點本機驗證集量不到。
     - fit 模擬池上，執色分開切時，連不分執色的 token 也只剩 0.248。
     - 能補救的只有 CNN，但 CNN 是用全部訓練玩家訓練的，沒有乾淨的集合可以評估。
  2. test02 的競賽一工具已經實跑過正式模式。重新打開它就要整套重新驗證，會占掉 03 在 10 月的時間。
  3. 團隊分開：p03 的棋力模型和它的任何輸出都不進競賽一。
- **評估過、決定不做的便宜選項**

| 選項 | 為什麼不做 |
|---|---|
| 第 3 層依每池自我檢索在 Q00／Q01 之間選 | 本機自我檢索只差 +0.018（隨機切），而且只在最壞情況用得到；要改已驗證的 `t02_player.py` |
| 競賽一的 8 種對稱 TTA | 只改善 CNN 項，要新的測試 notebook，GPU 時間 8 倍 |
| 重訓 embedding | 沒有乾淨的調參集 |

- **不花成本的應變**：11/04 如果 T4（|b_c − b_q| > 0.5）、T2 或 T3 觸發，照 PREREG_02 已登記的程序先交件。之後才考慮 PREREG_04-player，用官方候選池做無標籤的自我檢索（例如依執色切），只在既有組合裡選。到時再決定。
- **營運面**：彩排照舊；11/20 還拿不到 G4 就照 rev 7 第 2 條處理。

### 3. 兩場共通

- **預算**：競賽二 H3R 一次（第 2 次，共 4 次）；競賽一這次不打分。
- **交件模型**：選甲。
- **禁止事項**
  - 不用外部資料、預訓練模型或 KataGo，從零訓練。
  - 不用「一題對一人」的結構。
  - `num_games` 只能進入共用的溫度。
  - 不拿 game_id 推執色。
- **凍結檔一個位元組都不動**：`tools/p02_*.py`、`c10data.py`、`decide.py`、`colab/*prereg02*`、`colab/test02_*`、`tools/t02_*.py`、`results/prereg01/**`、`results/prereg02/**`、`results/test02/**`。
  - 唯一的例外是第 5 節 `official_log.py` 的方法標籤，改了要補測試。
- **揭露**
  - recipe 的選擇參考了只用 inner 的診斷和文獻，沒有在 inner 上調任何訓練超參數。
  - 開局的 M、α 是在 inner 上選的，所以 inner 分數略為樂觀。
  - bf16 加 cuDNN 不能逐位元組重現。
  - 生成式 AI 協助的部分照規則揭露。

### 4. 截止點與退路（寫死）

| 截止點 | 條件沒達成時 |
|---|---|
| **C1 10/16（五）23:59** | 還沒有一個能用的 seed 的 Colab 結果 commit：03 記「未完成」並關閉，R20 照舊，不打分 |
| **C2 10/19（一）23:59** | H3R 還沒打分，而且閘門還沒判定：同上 |
| **C3 11/03（二）23:59**（目標 10/27） | 採用後，test03 工具、rev N、測試、假測試集 dry run 還沒全部 commit 通過：03 這一季不上傳，最終上傳是 R20 |
| **C4 11/16（一）23:59** | 03 在官方測試集還沒有一次成功的執行（test03 接縫、本機接縫、T1、`check_submission --official` 都過）：最終上傳是 R20 |

- 任何截止點沒趕上，都不拿額外的打分機會去補。

### 5. 測試日擴充（只有 H3R 採用才做；rev N 在 11/04 11:00 前登記）

- **設計**：test02 完全不動，而且先跑。test03 只推論 p03，C5 的逐盤輸出直接讀 test02 已 commit 的結果。這樣 test03 失敗也不影響 R20 的上傳。
- **`colab/build_test03_rank_nb.py` → `test03_rank.ipynb`**
  - 底稿是 prereg03 的 notebook（SHA 釘死）。
  - 權重 SHA 釘死。只載入選中組合用到的 p03 權重（一或兩個）。
  - 三組題目：`seam_val`、`public`、`private`，8 種對稱，fp32。
  - 接縫和 prereg03 的驗證題輸出比：每題平均 softmax 的最大差，同型 GPU ≤ 1e-4、不同型 ≤ 1e-3；規則 B 不同 ≤ 4／1,589 題。
  - 讀同一個 `C10_test_v1.zip`：MANIFEST 的 SHA 要和 test02 那次相同。
- **`tools/t03_rank.py`**
  - `apply` 要求：
    - test02 rank 的狀態是 completed、接縫過、已 commit。
    - test03 也是 completed、接縫過。
    - 兩邊的 `manifest_sha256` 相同，每組題目的 key 陣列相同。
    - 開局沿用已 commit 的 `results/test02/rank/opening_test.npz`。
  - 本機接縫：`seam_val` 要重現 `results/prereg03/rank/rank_val_probs.csv`（同一個容許差）和 `pred_B.csv`（不同 ≤ 4 題）。
  - 輸出：`results/test03/rank/submission_rank.csv`，並跑 `check_submission --official`。
  - `fallback-fit`：選中的組合拿掉開局項，並自我核對選中組合重擬合的參數 ≤ 1e-12，11/04 前 commit。
- **T1 是 FAIL_OPENING 時**：選中組合的無開局版本和 R10（0.5687）比 inner 交叉驗證，取高的。這個判定在 11/04 前就寫進 rev N。
- **`tools/official_log.py`**：加上方法標籤 `prereg03 S0x rule B (甲)` 和它的無開局備案；`check_reports` 要接受 `results/test03/rank/`。改完重跑整套 test02 測試和慢速 dry run，證明 test02 不受影響。
- **使用者 11/04 的順序**
  1. 照 test02 做，先上傳 R20 和 Q04。
  2. 再跑 `test03_rank.ipynb`，約 10–15 分鐘。
  3. Claude 核對後，把 S0x 當成競賽二的第二次上傳。上傳指示自成一則訊息，使用者要先說 OK。
- **最後一次上傳（11/23 前）**：同時符合以下三項才交 S0x，否則交 R20（或照 test02 規則交 R10）。
  - H3R 採用。
  - C4 之前有一次成功的正式執行。
  - T1 不是 FAIL_STRUCT。
- **public 分數**：照 VALIDATION 紀律 4，比上一版低 0.05 以上只追查錯誤，不因此換方法。

### 6. Colab 執行與成本

- **帳號（營運上的選擇，不影響判定）**
  - 開跑前，使用者在 AI Pro 帳號的 Colab「資源」面板看剩多少運算單元：
    - ≥ 35：兩個 seed 都在 AI Pro 跑。Drive 已經有備份，測試日也全部在同一個帳號。
    - 15–35：只跑 s1。
    - 看不到或 < 15：改用 Pro+。要先上傳 `C10_bundle_v1.zip`（247 MB）一次；測試日要把測試 zip 也傳到 Pro+ 的 Drive，test03 在 Pro+ 跑。
- **每份的時間**（依據：`results/prereg0{1,2}/rank/timings.json`、每小時 8.71 單元）

| 情況 | 時間 | 運算單元 |
|---|---|---|
| bf16 快 2 倍 | 約 50 分 | 約 7.2 |
| bf16 快 1.5 倍 | 約 64 分 | 約 9.3 |
| bf16 沒加速 | 約 94 分 | 約 13.7 |

  - bf16 沒實測，開頭 300 步會印出推估。
  - 在 Pro+ 重抽特徵每份再加約 5 分鐘、0.7 單元。
- **兩份同時跑**：AI Pro 在 01／02 時已經同時跑過兩份。
- **斷線**：每斷一次約多 10 分鐘、1.5 單元。
- **預留**：總共 40 單元，包括兩個 seed、一次重跑、彩排的額外部分、測試日的 test03。
- **AI Pro 沒有背景執行**：跑完前分頁要開著、電腦不能睡眠。01 跑 45／51 分鐘、02 跑約 40 分鐘，都沒有斷過。

### 7. 工程清單（Claude；估時）

| 項目 | 估時 | 什麼時候做 |
|---|---|---|
| E1 PREREG_03 rev 1、`results/prereg03/design/` | 0.5 天 | 10/01 前 |
| E2 prereg03 builder 和兩份 notebook、CPU 冒煙（極小設定、關 autocast）、逐格 diff、獨立審查 | 1.5–2 天 | 10/05 前 |
| E3 `p03_rank_fuse.py` 和測試（S00 等於 0.58547…，參數自我檢查，順序強制），rev 2 | 1 天 | 10/05 前 |
| E4 收到 zip 後：commit、tune、凍結、apply、打分、RESULTS（獨立核對） | 1–1.5 天 | 收到 zip 後 |
| E5 採用才做：test03 builder 和 notebook、`t03_rank.py`、`official_log`、假測試集 dry run、TESTDAY／COLAB_RUN、審查、rev N | 3–4 天 | 10/12–10/27 |

### 8. 風險與防護

| 風險 | 防護 |
|---|---|
| bf16 沒加速或不穩定 | 開頭 300 步量時間；非有限值就停，fp32 備案要先寫修訂、整份重跑 |
| 拿不到 G4 | 接受 A100（worker ≤ 10）；T4、L4、H100 在關卡就擋下 |
| 斷線 | 每個 epoch 存 state 並核對 recipe；最多損失一個 epoch |
| 運算單元不夠（AI Pro 不明） | 開跑前看剩餘單元；Pro+ 當備案 |
| 執色接錯 | join 檢查、執黑比例、翻色單元測試都是硬性停止；對調執色的 accuracy 只記錄 |
| 資料外洩 | 只用 `load_rows(part="fit")` 訓練；`tune` 不讀 `val_*`；答案不上 Colab；打分要記帳 |
| 在 inner 上選組合造成的樂觀 | 只有 4–5 個組合；閘門要 +0.010 且 CI 下界 > 0；最後由驗證集判定 |
| 有小增益但打分沒過 | 接受，不重打；打分機會照登記扣 |
| test03 工具趕不上 | C3、C4 截止點；test02 不動，R20 永遠先上傳 |
| 測試日兩個帳號搞混 | 優先單一帳號；SHA 釘死；test03 核對 MANIFEST |
| 兩隊混用 | p03 只用在競賽二；競賽一的工具不變 |
| 盤數變成等級特徵 | 同一種溫度形式；檢查各盤數分箱 |
| 不能逐位元組重現 | 逐盤輸出用 fp32；接縫容許差照 test02 |
| 使用者 10 月很忙（C4 10/16 口述、C4 用 4090） | 使用者只需要一次 Colab，彩排選做 |

### 9. 時程

| 日期 | 事項 | 誰 |
|---|---|---|
| 9/27–10/01（四） | E1，rev 1 commit | Claude |
| 10/01–10/05（一） | E2、E3，rev 2；使用者看 AI Pro 剩餘單元（1 分鐘） | Claude／使用者 |
| 10/06（二）–10/08（四） | 使用者跑 Colab（s1，可同時跑 s2） | 使用者 |
| 收到 zip 後 1–2 天（目標 10/09–10/10） | E4：閘門 → H3R → RESULTS | Claude |
| 10/16、10/19 | C1、C2 截止點 | — |
| 10/12–10/27（採用才做） | E5，rev N | Claude |
| 10/28（三）–11/02（一） | 選做的彩排：test02 加 test03 | 使用者約 1.5 小時 |
| 11/01（日） | 11 月投入的決定（這時 03 的結果已經知道） | 使用者 |
| 11/03 | C3 截止點 | — |
| 11/04（三） | test02 先上傳 R20／Q04，接著 test03，上傳 S0x | 兩人 |
| 11/16（一） | C4 截止點 | — |
| 11/23（一） | 最後一次上傳 | 使用者 |

### 10. 刻意不放進 03 的

| 做法 | 為什麼 |
|---|---|
| 逐手、依等級條件化的落子模型 | 我們的規模沒有前例；工程 1–2 週，G4 2–2.5 小時，還要自己的測試日工具，會把 11/04 拖下水。要做的話另寫 04：10/14 決定要不要做，11/06 前打分，11/13 前工具完成，否則放棄 |
| label smoothing（R3B） | 證據弱，而且多一條程式路徑；第二個平行名額改給第二個 seed，不用多寫任何程式 |
| 每人盤數上限或按人加權 | 多一個超參數，沒有證據 |
| 氣和提子的特徵層、加長盤面歷史 | 要重抽特徵；AlphaGo 的表顯示加氣沒有增益；長棋並沒有比較差 |
| 只在 CPU 上細化合併方式、SGF 特徵 | inner 上都拿不到 +0.002 以上 |
| 競賽一的任何變更 | 見第 2 節 |

---

整個過程只讀 repo，沒有 commit。沒有打開或雜湊任何 `*answers*`，沒有跑 `score.py`，沒有用 GPU。讀過的主要檔案：`D:\AI-Portfolio\competitions\aicup2026-go\CLAUDE.md`、`docs\PREREG_02_OPENING_FUSION.md`、`docs\VALIDATION.md`、`docs\TESTDAY.md`、`docs\COLAB_RUN.md`、`results\prereg02\RESULTS.md`、`results\prereg02\rank\tune_report.json`、`results\prereg02\explore\opening_player_sim.json`、`results\test02\design\spec.md`、`results\score_ledger.csv`、`tools\official_log.py`、`tools\t02_rank.py`、`tools\p02_rank_fuse.py`。