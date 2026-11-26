# PREREG_02 提案：主攻競賽二，分三階逐步推進

角度：以 11/24 前競賽二的期望分數最大為目標。文中的「預期增益」都是判斷，不是實測。實測只看兩處：inner 假題目，以及每個凍結方法在本機驗證集的那一次打分。本提案沒有開過 `*answers*.csv`，沒有跑 `score.py`，也沒有改 repo。

## 0. 結論

1. **競賽二分三階推進，每完成一階就多一個驗證過、可以上傳的版本：**
   - **S1：聚合方式和校準，不重訓。** 用 prereg01 的權重重跑推論，存下每盤的 logits。在 inner 假題目上調四件事：log-pool、依盤數調整的溫度 τ(n)、等級偏移、D4 TTA。決策規則維持已採用的 B。
   - **S2：換一套訓練設定（recipe v2）重訓。** 改動有：
     - 改成目標玩家視角，另加一層執色
     - 訓練時做 D4 資料增強
     - 盤面歷史從 250 加長到 360
     - cosine LR，訓練 12 個 epoch
     - bf16
     
     checkpoint 固定用最後一個 epoch。inner 不拿來挑 checkpoint，全部留給校準。
   - **S3：KataGo HumanSL 逐手特徵，加上和 CNN 融合，有閘門。** 先跑 pilot 量速度和訊號，過了閘門才跑全量。stacker 只用 fit 玩家訓練，和 CNN 的融合權重在 inner 上調。這一階的上限最高：出題者自己參與的論文，在「一題全是同一位玩家」的設定下，換算成本題指標是 0.72–0.75。
2. **競賽一只做 P1：不重訓，只重新排序。** 方式是 cosine 或 CSLS，加上開局相似度。參數用候選庫的「對半自我檢索」來選，這是 VALIDATION 允許的「測試時使用候選庫」例外，必須寫進登記。重訓延到 11/04 看過官方題目結構後，另寫 PREREG_03。
3. **本機驗證集最多打 4 次分：**
   - 競賽二：S1、S2、S3 各一次，都和 prereg01 B 比。
   - 競賽一：P1 一次，和 prereg01 比。
   - 每次打分前都要先過 inner 閘門。最後交哪一版由 inner 分數決定，驗證集只負責確認。
4. **Colab 成本：** 使用者要操作 4 次（Run 1–3，加上 11/04 的 Run T）。G4 合計約 3–7 小時，約 22–62 個運算單元。Pro+ 每月有 600 單元（`results/colab_facts_20260925.json`）。乙版（用全部 8,007 人重訓）不在這份登記裡。

## 1. 預期增益（判斷，不是實測）

| 候選 | 相對 prereg01 B（分等級平均 0.4909） | 依據 | 能成的機會（判斷） |
|---|---|---|---|
| S1 | +0.00 ～ +0.04 | 平均機率不會隨盤數變尖（Spearman −0.03）；62% 的題目第一、二名相差不到 0.02。前人多用幾何平均。反例：Kittler 1998 發現相加規則比較穩。出題者估計 set 聚合能加 2–4 分，沒有實測 | 中 |
| S2（含它自己的聚合） | +0.02 ～ +0.08 | 固定 LR，第 5 個 epoch 還在學（train 26.3%，inner 25.2%）。出題者估計 D4 能加 3–5%，沒有實測。執色和開局兩項沒有直接證據 | 中高 |
| S3（相對 S1、S2 中較好的一個） | 過閘門的話 +0.05 ～ +0.15 | Kuboki 等（2505.00279，出題者是共同作者），同一玩家設定：SE＋HumanSL 在 n=5 到 15 盤時，換算本題指標 0.718–0.755。消融顯示拿掉 HumanSL 的影響最大。但他們用野狐資料、分 11 組，不能直接當成預測值 | 約一半（工程、速度、資料差異三項風險） |
| P1（競賽一，相對 0.3668） | +0.01 ～ +0.06 | CSLS 在其他領域 P@1 加 3–7 分。圍棋只看前 10 手、查詢 10 盤、200 人時，辨識率 45%（Lin 等）。本機的 hubness 屬輕微 | 中 |

- 社會組拿獎狀要進前 25%。9/26 競賽二有 69 隊，前 25% 約是前 17 名。只做 S1、S2 可能不夠，這是把 S3 納入的主要理由。

## 2. 不放進 PREREG_02 的做法與理由

- **序數 loss（CORN、SORD）**
  - CE 是 proper scoring rule，機率校準後再套規則 B，本來就是在最大化本題的期望得分。
  - SORD 的已知增益只有 +2%，而且是別的領域。
  - 多一個變數，S2 的效果就無法歸因到單一改動。
- **attention MIL 或 set 模型**
  - 可以調參的乾淨資料只有 636 位 inner 玩家，參數多就會過擬合。
  - 改用 fit 玩家訓練的話，吃到的是 CNN 在訓練資料上過度自信的輸出。
  - 所以改用低參數的 log-pool 加 τ(n)。
- **氣、提子特徵**：720k 盤 × 360 個盤面都要算，抽特徵時間會變成好幾倍。出題者估計 +4%，但沒有實測。
- **每人盤數上限**：沒有證據，而且會減少訓練資料。
- **競賽一重訓**（留出 inner、拿掉 64 維瓶頸、換 loss）：約一週工程加約 10 單元，而且要先知道官方結構。放 PREREG_03。
- **用棋力模型幫競賽一重排（P4）**：兩場是不同隊伍，規則寫「各隊伍間不可私下共享程式及特徵值」。使用者接受的是「共用前處理」，把一隊的模型輸出拿給另一隊當特徵，已經超出這個範圍。不做，除非使用者另外決定。
- **Hungarian 或一題對一人**：VALIDATION 規定要等官方確認。
- **搜尋出來的輸棋分數損失**：成本高，Kuboki 的消融顯示沒有貢獻。
- **比對外部野狐棋譜庫**：違反資料 README「不得用於識別真實使用者」。

## 3. 共用：inner 假題目（所有候選共用同一批題目）

- **玩家**：inner 640 人中，至少 5 盤的有 636 人。各級人數：D 71、C 65、B 83、A 123、1D 34、2D 49、3D 49、4D 52、5D 53、6D 57。
- **bank**：每人依 `sha256("c10-prereg02|bank|"+rank+"|"+game_id)` 排序，取前 min(20, n) 盤，合計 **11,813 盤**（本機依 `players_split.csv` 加上 inner 規則重算）。
  - S1、S2、S3 的 inner 評估都只用 bank 裡的棋，所以三者的題目完全相同，可以直接比。
  - HSL 也只需要算這 11,813 盤。
- **出題**：
  - 照 `make_val.py:126-133`：每一輪、每一位玩家，k = min(randint(5,20), n_bank)，從 bank 抽 k 盤。
  - 亂數用 `random.Random(f"c10-prereg02|pq|{r}")`。
  - 擬合用第 0–49 輪，評估用第 1000–1199 輪。
- **分折**：依玩家分 5 折。每一級內依 `sha256("c10-prereg02|fold|"+player_id)` 排序後輪流分配。
- **評估指標**：分等級平均（rank_balanced_mean）、規則 B，取 200 輪的平均。
  - OOF 的意思：用另外 4 折擬合參數，在這一折上評估。
  - 另外記錄每一級的分數，以及依盤數分箱（5–7、8–12、13–20）的分數。
- **擬合目標**：class-balanced NLL，每一級的總權重相同，和主要指標一致。用 L-BFGS-B，不直接最佳化得分。
- **盤數 n 的限制**：n 只能透過 τ(n) 這種「對 10 個等級都一樣」的銳利度參數進入，不能有隨等級不同的 n 項。VALIDATION 禁止把 num_games 當特徵，這一點要請獨立審查特別看。
- **不能用的數字**：T≈2.85 是對著已打分結果反推出來的，任何地方都不准用。

## 4. 競賽二

### S1：聚合方式和校準，不重訓

- **輸入**：Run 1 輸出的 prereg01 `best_model.pth`（第 5 個 epoch，就是登記指定的 checkpoint）的每盤 logits，範圍如下：
  - inner：全部 73,556 盤
  - 本機驗證：19,156 盤
  - 每盤存 identity 和 D4 共 8 種對稱
- **可選的組合，登記時就固定，共 2 × 3 = 6 種**：
  - TTA：{不做，8 種對稱的 logits 取平均}
  - F0：每盤 softmax 後取平均（就是 prereg01 的做法）
  - F1：log-pool，logits_q = Σ_g log_softmax(z_g) / (τ0·n^α)，τ0 ∈ [0.05, 50]、α ∈ [0, 1]
  - F2：F1 再加等級偏移 b（10 維、總和為 0），在平均 NLL 上加 L2 懲罰 λ=0.01
- **挑選**：OOF 最高者勝出。差距在 0.002 以內時選簡單的，順序是 F0 < F1 < F2、不做 TTA < TTA。
- **打分閘門**：OOF(S1*) − OOF(F0、不做 TTA) ≥ +0.005。沒過就不打分，只記錄。
- **打分前要 commit 的東西**：
  - 在全部 inner 上重新擬合的參數 JSON
  - OOF 表
  - 驗證集的 `rank_val_probs.csv` 格式檔，接著照舊用 `tools/decide.py` 產生 `pred_B.csv`，沿用它的防呆

### S2：recipe v2 重訓

| 項目 | prereg01 | S2 |
|---|---|---|
| 資料 | fit 5,767 人／720,307 盤 | 相同 |
| 輸入 | 最後 250 個盤面 × 黑／白 = 500 層 | 最後 360 個盤面 × 目標玩家／對手 = 720 層，加 1 層執色（執黑填 1）= **721 層**。可以涵蓋 359 手以內的棋（各級 p90 都 ≤289 手，最長 445 手）。執色來自訓練檔的 `color` 和題目檔的 `color_k`，兩者都合法（`VALIDATION.md` 只禁 num_games、game_id、列順序）|
| 模型 | GoRankResNet | 相同，只改第一層 conv 的輸入通道數，計算量推估約多 7% |
| 資料增強 | 無 | 訓練時每筆隨機套一種 D4 對稱，在 GPU 上做 |
| loss | CE | CE，不改 |
| 最佳化 | AdamW，lr 固定 1e-3 | AdamW 峰值 1e-3、wd 0.01。前 1,000 步線性暖身，之後 cosine 降到 1e-5 |
| epoch | 5 | **12**，每個 epoch 2,814 步，合計 33,768 步 |
| 精度 | fp32 | bf16 autocast（權重維持 fp32）、channels_last |
| DataLoader worker | 4 | 16（在 12 vCPU 的機器上用 10）|
| checkpoint | inner 逐盤 accuracy 最高的 | **固定用第 12 個 epoch**，inner 只記錄、不拿來挑 |
| 存檔 | 每題的平均機率 | 見下方清單 |

- **存檔內容**（每個檔都小於 100 MB，避免被結果 zip 的規則漏掉）：
  - 每個 epoch 的 inner 每盤 logits，只存 identity，每個 epoch 2.9 MB。
  - 第 12 個 epoch 的 inner 和驗證題每盤 logits，8 種對稱都存，分別約 23.5 MB 和 6.1 MB。
  - 權重只留最後的 `.pth` 和 `state.pt`（含 scheduler 和 RNG 狀態，可以接續），省 Drive 空間。
- **合理性檢查**：第 12 個 epoch 的 inner 逐盤 accuracy（identity）如果低於 25.15%，就視為疑似 bug。要能指出具體的 bug 才可以重跑；找不到 bug，S2 就不打分。
- **可選的組合**：S1 的 6 種 × 和 prereg01 做 ensemble 的權重 w ∈ {1, 0.75, 0.5}，也就是 z = w·z_v2 + (1−w)·z_p01，共 18 種。挑選和平手規則同 S1。
- **打分閘門**：OOF(S2*) ≥ OOF(S1*) + 0.01。

### S3：KataGo HumanSL 特徵加融合（有閘門）

- **規則**：
  - 正式規則沒有禁止外部資料或預訓練模型，要求在報告中揭露來源和貢獻。
  - 出題者的簡報第 80 張把預訓練權重列為改進方向；第 15 張寫「所有人都從零訓練」，是反向訊號。
  - 兩份報告都要揭露。如果主辦方明文禁止，S3 就整個取消，改用 S1 或 S2。
- **程式和權重**：
  - 程式用 KataGo repo（MIT）固定在某一個 commit 的 Python／PyTorch 程式碼。
  - 權重用 v1.15.0 釋出的 `b18c384nbt-humanv0.ckpt`（MIT 式授權），builder 裡寫死 SHA-256。
  - Claude 會先在本機下載，做 CPU 冒煙測試，並核對 SHA。下載前先問使用者，說明檔名、來源、大小。
- **profile**：
  - 等級：{14k, 11k, 8k, 5k, 2k, 1d, 2d, 3d, 4d, 5d, 6d, 8d}，共 12 個。這組涵蓋我們的 D（10–12 級）到 6D。
  - 系列：pilot 同時試 `rank_` 和 `preaz_` 兩個系列，全量只用 pilot 選出的那一個。
  - 兩位玩家的 metadata 都設成該 profile 的等級。
- **每盤要存的特徵**（每盤 112 個數）：
  - 只看目標玩家下的棋。
  - 依手數分成 3 段：1–60、61–160、161 以後。
  - 每個 profile、每一段，各存三個加總：實際落子的 log p、策略的 entropy、實際落子是否等於 top-1。
  - 另外存每一段的手數。
- **pilot（Run 2b，和 S2 並行）**：
  - 資料：fit 每級 40 人 × 最多 10 盤（約 4,000 盤），加上 inner 636 人 × bank 的前 5 盤（3,180 盤）。
  - 每 3 手目標落子取 1 手，兩個系列都跑，約 640 萬次網路運算。
  - 要量的：端到端每秒運算數，以及 CPU 抽特徵每個盤面要幾 ms。
- **閘門 G3**（三項都要過；在本機 CPU 上算，commit 後才能開 Run 3）：
  1. **正確性**：
     - 在 fit pilot 上，每位玩家「平均 log p 最高的 profile」和真實等級的 Spearman ≥ 0.5。
     - 最佳 profile 的 top-1 命中率 ≥ 25%。
     - 目的是抓出特徵編碼寫錯的情況：寫錯的話，等級趨勢會是平的。
  2. **價值**：
     - 用 inner 的 5 盤題（每人一題，共 636 題），算 5 折 OOF。
     - 比較「HSL stacker 加 prereg01 S1* 的融合」和「只有 S1*」，差距要 ≥ +0.02。
     - stacker 只用 fit pilot 訓練。
     - 選哪個 profile 系列也看這一項。
  3. **成本**：用實測速度算出全量的 GPU 時間，在 stride s ∈ {1, 2, 3, 4} 中取最小、且推估 ≤ 2.0 小時的 s。s=4 仍超過就不通過。
     - 全量約 7,170 萬次運算。s=1 時每秒要 1 萬次，s=2 要 5 千次，s=3 要 3,320 次，s=4 要 2,490 次。
- **全量（Run 3）**：
  - 資料合計 **53,852 盤**：
    - fit：每級 120 人（至少 5 盤，salt `c10-prereg02|hslfit`）× 最多 20 盤，共 22,883 盤
    - inner bank：11,813 盤
    - 驗證題：19,156 盤
  - 每 500 盤寫一個檔到 Drive，斷線後跳過已完成的部分接著跑。
- **stacker**：
  - 題目特徵：每個 profile × 段落，把上面三個量依手數加權取平均，共 108 維，用 fit 的統計值標準化。**不放任何盤數或手數的特徵。**
  - 模型從兩種中選：multinomial LR（C 用 fit 玩家 5 折 CV 選），或參數固定的 LightGBM（num_leaves 15、300 棵樹、lr 0.05、min_data 50）。
  - 訓練資料和挑選都只用 fit 玩家的假題目（50 輪）。
- **融合（在 inner 上）**：
  - 公式：logits = a·n^β·u + c·v + b。u 是 stacker 的 log 機率；v 是 S1*、S2* 中 OOF 較高者的 CNN 聚合 logits。
  - 兩種選擇：只用 HSL（c=0），或融合。
  - 同樣用 balanced NLL 擬合、5 折 OOF。
- **打分閘門**：OOF(S3*) ≥ max(OOF(S1*), OOF(S2*)) + 0.02。

## 5. 競賽一 P1：不重訓，只重新排序

- **輸入**：
  - Run 1 輸出的 prereg01 player `best_model.pth`，對每盤的窗口 embedding 加總，另存窗口數。範圍是候選 40,000 盤加題目 4,935 盤，約 46 MB。
  - 開局特徵直接讀本機的題目檔和候選檔，只用 CPU。
- **可選的組合，共 3 × 4 = 12 種**：
  - 距離 d，三種：
    - 官方做法：依窗口加權平均後算 L2
    - 每盤等權平均、重新正規化後算 cosine
    - CSLS（k=10）加 cosine
  - 開局相似度的權重 β ∈ {0, 0.5, 1, 2}
  - 分數 = z(−d 或 CSLS) + β·z(O)，z 是每一題在候選之間做標準化。
- **開局相似度 O(q, c)**：
  - 每盤取前 30 手，在 8 種對稱中選字典序最小的那一種作為標準形。
  - 只和同一執色的候選棋比較。
  - O = 題目每一盤對候選各盤「共同前綴長度 / 30」取最大值，再對題目的各盤取平均。
- **挑選方式：對半自我檢索**：
  - 把每位候選的 100 盤依雜湊分成 A、B 兩半。從 A 抽 5–20 盤當假題目，拿去比所有候選的 B 半。
  - 抽 20 輪，照官方公式計分，選平均最高的組合。
  - 只用候選檔，不用答案。這是 VALIDATION「測試時可以用候選庫」的例外，要寫進登記。
  - 凍結的是「程序」：本機驗證池和官方 public、private 各池都照同一個程序重新選。
  - 如果 11/04 發現官方題數不等於候選人數，或不是一題對一人的結構，就把 CSLS 從可選組合中移除。
- **打分閘門**：自我檢索上，選出的組合比官方做法高 ≥ +0.01。

## 6. 驗證集打分、採用規則、最後交哪一版

| 編號 | 指令（差值是 新−舊） | 對照組 |
|---|---|---|
| S1 | `python tools/score.py compare-rank results/prereg01/rank/pred_B.csv results/prereg02/S1/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg02 S1" --json results/prereg02/S1/score_S1.json` | prereg01 B |
| S2 | 同上，換成 `S2/`，`--note "prereg02 S2"` | prereg01 B |
| S3 | 同上，換成 `S3/`，`--note "prereg02 S3"` | prereg01 B |
| P1 | `python tools/score.py compare-player results/prereg01/player/player_val_top5.csv results/prereg02/P1/player_val_top5.csv data/local/player_val_answers.csv data/local/player_val_candidates.csv --ledger ... --note "prereg02 P1" --json ...` | prereg01 player |

- **採用條件**：照 VALIDATION，主要指標差值的 95% CI 下界 > 0（競賽二每一級內分層重抽；bootstrap 10,000 次、seed 0）。
  - 競賽二另有衝突規則：次要指標的點估計 < 0，就不採用。
  - 競賽一上傳前要先過 `check_submission.py`。
- **對照組要寫清楚**：RESULTS.md 寫過「A、B 各打這一次分，之後不再用這兩份預測打分」。但 `score.json` 沒有每題的分數，配對 bootstrap 只能把舊檔重新讀一次。
  - PREREG_02 要明文說明：prereg01 的 `pred_B.csv`（SHA `d42a0b11…`）和 `player_val_top5.csv`（`0f5761a1…`）只當作固定的對照組，不是重新評估那個方法，也不會依它做任何選擇。
  - 打分紀錄帳上照常會多出這幾列。
- **打分前要 commit 的東西**：可選組合、salt、閘門門檻（都寫在本登記裡）、工具的 commit、選中的組合、擬合好的參數、inner OOF 表，以及驗證預測檔的 SHA。
- **最後交哪一版（競賽二）**：
  - 在已採用的 {S1, S2, S3} 中，選「打分前就已 commit 的 inner OOF」最高的那一個。驗證集的點估計不參與選擇。
  - 都沒採用的話，交 prereg01 B。
- **競賽一**：P1 採用就交 P1，否則交 prereg01。
- **沒採用時**：寫進 RESULTS，下一階照樣進行。例如 S1 沒採用，S2 的可選組合裡仍保留 log-pool，但在報告中註明這是第 2 次檢定。
- **多重比較**：
  - 競賽二最多檢定 3 次。如果三者全都沒有真實效果，至少誤採用一次的機率上限約 14%（Bonferroni 上限）。
  - 緩解方式有三：只有過了 inner 閘門的候選才打分；最後的選擇依 inner；所有結果都要報告，包括失敗的。
- **規則 B**：PREREG_01 的 H1 已成立，B 是標準決策規則，這次不重測 A 對 B。

## 7. 交件模型（甲／乙）和 11/04

- 本登記選**甲**：直接上傳本機驗證過的模型，從第一次上傳就用。
- 乙（用全部 8,007 人、固定 epoch 數重訓）驗證集驗不到。值不值得，要看官方測試玩家是否也在訓練集裡：
  - 有重疊的話，「認出這個人」就等於知道他的等級，另外也可以考慮用「最近的訓練玩家」這類做法。
  - 11/04 照 `DATA.md` 最後一節逐項檢查後，才決定要不要寫 PREREG_03，並且在上傳乙之前 commit。
- **Run T（11/04–06）**：對官方題目跑選中的整套流程，也順便算出退路版本的預測，以備需要切換。上傳前跑 `check_submission.py`。
  - public 分數只用來抓格式錯誤和大幅退步（掉 0.05 以上才追查）。
  - 最後一次上傳必須是依登記選出的那一版。

## 8. Colab 成本和使用者要做的事

| 次 | 內容 | 約何時 | G4 時間 | 單元（每小時 8.71）|
|---|---|---|---|---|
| Run 1 | rank dump、player dump，兩份並行，各一份 notebook（兩隊分開）| 10/01 | 各 0.2–0.3 小時 | 3–5 |
| Run 2a | rank v2 訓練 | 10/09 | 1.1–1.9 小時（最壞情況：fp32 每 epoch 436–470 秒 × 12，加抽特徵約 9 分鐘和 dump）| 10–17 |
| Run 2b | HSL pilot，和 2a 並行 | 10/09 | 0.3–1.0 小時 | 3–9 |
| Run 3 | HSL 全量（過閘門才跑）| 10/19 | 設計上限 2.5 小時 | ≤22 |
| Run T | 測試集推論 | 11/04–06 | 0.3–1.0 小時 | 3–9 |
| 合計 | | | 約 3–7 小時 | **約 22–62** |

- **使用者要做的事**：
  0. 現在先在 Colab 的「查看資源」看兩個帳號各剩多少單元，決定用哪個帳號。
     - prereg01 的權重和特徵備份都在 AI Pro 帳號的 Drive。改用 Pro+ 帳號的話，Claude 會寫搬移約 2.5 GB 的步驟。
     - Drive 要剩至少 6 GB。
  1. Run 1、2、3、T 每一次的操作：
     - 開 notebook，選 G4，按「全部執行」。
     - 被斷線就再按一次「全部執行」，會接著跑。
     - 跑完下載結果 zip，把路徑告訴 Claude。
  2. 11/01 決定 11 月要投入多少。PREREG_02 在 10 月底前就會做完，11 月最少只需要 Run T 和上傳。
  3. 11/04 本人登入 AIdea 下載測試檔。每次上傳都要先在對話裡說 OK。
- 打包檔 v1 不改：新程式直接寫在 notebook 裡，使用者不用重新上傳 247 MB。

## 9. Claude 的工程量（約 8–10 個 session 工作天）

1. PREREG_02 本文和 commit：0.5 天。
2. `tools/` 的 inner bank、假題目、聚合擬合、OOF 工具，加測試：1 天。測試包括假題目的可重現性、合成資料上的 NLL 擬合、和 `decide.py` 的 B 一致。
3. Run 1 的兩個 dump builder：1 天。沿用 prereg01 的 builder 片段，含 CPU 冒煙測試和獨立審查。
4. `tools/` 的 P1 工具，加測試：1 天。包括對稱標準形、共同前綴長度、CSLS、對半自我檢索。
5. `build_rank_nb.py` 參數化出 prereg02 v2：1.5 天。
   - 要改的部分：特徵子類別、第一層 conv、D4、scheduler 存進 state、bf16、dump。
   - 要新增 `MOD[14]`，並登記進 `DEV_ROWS`。
6. `build_hsl_nb.py`（pilot 和全量共用，靠 MODE 切換）：2–3 天，是最大的風險點。
   - 內容：接上 KataGo 的 Python 程式、48 核抽特徵、GPU 批次、分塊接續。
   - 本機用 CPU 跑幾十個盤面做冒煙測試。
7. Run T 的 builder：1 天。先搭骨架，11/04 確認格式後補完。
- 每份 notebook 都照同一套流程：builder 產生 → 冒煙測試 → diff → 獨立審查 → commit。notebook 一律不手改。

## 10. 風險與防護

| 風險 | 防護 |
|---|---|
| 外洩 | 每份 notebook 都只透過 `load_rows` 讀 fit／inner，並把 `leak_check.json` 放進結果 zip。HSL 的 fit 樣本只來自 fit，stacker 標準化只用 fit 的統計值。答案檔不上 Colab。P1 只用候選檔（登記過的例外）|
| 校準過擬合 | S1、S2 最多 12 個參數，S3 融合約 13 個，對 636 人。用 NLL 擬合、依玩家做 OOF、b 加 L2。可選組合事先固定。checkpoint 固定用最後一個 epoch，所以 inner 沒有被挑 checkpoint 用過 |
| 多重比較 | 最多打 4 次分；有 inner 閘門；最後選哪一版依 inner；全部結果都要報告 |
| 盤數被當成等級訊號 | n 只准出現在對所有等級相同的 τ(n) 或 n^β，stacker 不放盤數或手數特徵，請獨立審查特別確認 |
| HSL 程式寫錯，或 metadata 跟 C++ 的 profile 不一樣 | G3 的正確性檢查（等級趨勢、top-1 命中率）。stacker 會學對應關係，不需要完全照 C++ 的語意 |
| HSL 太慢 | 成本閘門自動決定 stride；最壞的情況是不跑 |
| HumanSL 可能看過同一批來源的棋 | 如果有，驗證集和測試集是同樣受益，不會造成兩者之間的落差。報告要寫明 |
| 規則或觀感（簡報第 15 張）| 在報告中揭露；主辦方明文禁止就取消 S3，改用 S2 或 S1 |
| Colab 斷線、拿不到 G4 | v2 可以從 `state.pt` 接續，HSL 分塊接續。每份輸出都小於 100 MB。拿不到 G4 時，v2 改用 A100 |
| 官方先驗和分等級平均不一致 | b 以均勻先驗擬合。11/04 看模型對測試題的預測分布，但不依測試分數重調 |
| 官方不是一題對一人 | 競賽二不受影響；競賽一在那種情況下移除 CSLS |
| 時程延誤 | 每一階都留下可上傳的版本；最差情況交 prereg01 B |

## 11. 時程

- **9/27–29**：commit PREREG_02；做 Run 1 的 builder 和 `tools/`。
- **10/01**：使用者跑 Run 1。
- **10/02–05**：S1 和 P1 做 inner 調參、過閘門、打分。
- **10/02–08**：做 v2 和 HSL pilot 的 builder。
- **10/09**：使用者跑 Run 2（兩份並行）。
- **10/11–15**：S2 打分；判斷 G3。
- **10/19**：使用者跑 Run 3。
- **10/27 前**：S3 打分，寫 `results/prereg02/RESULTS.md`。
- **11/01**：使用者決定 11 月投入多少。
- **11/04–06**：跑 DATA.md 的檢查和 Run T，第一次上傳。
- **11/24**：最後一次上傳，必須是依登記選出的那一版。

參考的檔案（都只讀）：
- `D:\AI-Portfolio\competitions\aicup2026-go\docs\VALIDATION.md`
- `D:\AI-Portfolio\competitions\aicup2026-go\docs\PREREG_01_BASELINE.md`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg01\RESULTS.md`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\score_ledger.csv`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\split_v1\players_split.csv`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\data_inventory.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\colab_facts_20260925.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\tools\score.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\tools\decide.py`