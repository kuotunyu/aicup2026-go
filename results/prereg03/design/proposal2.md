# 事前登記 03（草案）：競賽二 recipe v2 重訓兩個模型，和 R20 在 log 空間集成

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。前一份：`docs/PREREG_02_OPENING_FUSION.md`（rev 7），結果在 `results/prereg02/RESULTS.md`。
- 設計依據：9/26 的三份診斷，放在 scratchpad 的 `p03/taskA`、`p03/B_error`、`p03/priorart`。rev 1 commit 前要搬進 `results/prereg03/design/`。本草案另外算了兩個數字，腳本和輸出一起搬：`p03/prereg_draft/cap_stats2.py`／`.json`、`power.py`／`.json`。這些分析都只用 fit／inner 和 split 清單。
- 本登記寫「要比什麼、怎麼選、怎樣算成功」。notebook 格子和檔名的細節，以之後的 `results/prereg03/design/plan.md` 為規格。實作時要偏離，先在修訂紀錄記一筆，而且要在任何 prereg03 輸出產生之前。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | （待填） | 第一版，2026-09-2x。當時**還沒有任何 prereg03 的程式、notebook、輸出或分數**。打分紀錄帳上，競賽二已用 2 次（01、02）；整季預算從 02 起算，已用 1 次 |

## 結論（給使用者）

- **做什麼**：競賽二用 recipe v2 從零訓練兩個新模型，一次 Colab 同時跑兩份。
  - 新做法：模型知道「哪一方是要猜的玩家」、訓練時隨機旋轉或鏡射棋盤、學習率先升後降、訓練到 12 個 epoch 的步數。
  - B 模型另外做一件事：限制棋譜特別多的玩家，每人最多算 200 盤。
  - 回本機後，在訓練玩家（inner）上從「R20 原樣、R20 加新模型、只用新模型」7 種組合裡選一種。過了閘門，才在本機驗證集打一次分。
- **競賽一這次不動**：不打分，預算維持用了 1 次。
- **成本**：Colab 一次，兩份同時跑約 1–2 小時，約 16–31 個運算單元（推估，bf16 的加速還沒實測）。之後如果採用，測試日多跑一份約 10–20 分鐘的 notebook。
- **能不能證明有效**：本機驗證集能分辨的差距有限，用 inner 推算如下（推估）：

  | 新方法真正好多少 | 驗證集證明得出來的機率 |
  |---|---|
  | +0.010 | 約 3 成 |
  | +0.015 | 約 6 成 |
  | +0.020 | 約 8–9 成 |

  沒證明出來就維持 R20，測試日照原計畫。
- **時程**：rev 1 在 9/27 commit；程式到 10/03；使用者 10/04 前後跑 Colab；10/10 前打分。採用的話，10/27 前擴充測試日工具並登記，10/28–11/02 彩排。**10/24 還沒打分的話，03 這一季就不用了。**

## 0. 為什麼做這些（都不看答案、不碰驗證集）

1. **校準和決策這一層已經榨乾。** 重新校準、既有權重的對稱或 epoch 組合、便宜的 SGF 特徵，在 inner 交叉驗證上都拿不到 +0.002 以上。完美校準的上限也只有 +0.002 以內（`B_error/r20_diag.json`，U2–U4）。
2. **瓶頸在每一盤棋的表示。**
   - 同一個模型給無限多盤時：≥20 盤的玩家 0.654、≥50 盤 0.663；同一批題目在 n=20 時是 0.622。
   - 相鄰兩級（1D–4D）玩家平均「期望等級」只差 0.34–0.43，同一級內的標準差卻有 0.45–0.59。
   - 如果每題都能從規則 B 的前兩名裡選對，可到 0.795。剩下的失分主要是分不開相鄰兩級。
3. **是跨玩家泛化的問題，不是訓練不夠。**
   - prereg02 第 6→9 個 epoch：訓練 accuracy 27.7%→33.9%，inner 約 25%，inner loss 1.950→1.980（`results/prereg02/rank/ckpt/history.csv`）。
   - fit 裡每一級前 10% 的玩家貢獻了該級 39.7–43.5% 的棋。以 inverse Simpson 算，有效玩家數只有 1,907 人（實際 5,767 人）（`cap_stats2.json`）。
4. **模型沒學到旋轉不變。** epoch 9 在原方向的逐盤 accuracy 是 25.6%，其他 7 種對稱只有 21.4–23.6%。8 種對稱平均仍然最好（`B_error`，第 4 節）。
5. **執色**：現行模型不知道哪一方是目標玩家。
   - 只在輸出端加「依執色的偏移」沒有改善：分開擬合 NLL 1.9395，共用 1.9362（`taskA/colour_diag.json`）。
   - 所以視角能帶來多少，只有實際訓練才知道。
   - 每一級黑白各半（`docs/DATA.md`），所以執色這一層本身不帶等級訊號。
6. **不加長盤面歷史**：手數 ≥300 的棋 accuracy 反而最高（.276），長棋沒有比較差（`taskA/length_diag.json`）。

## 1. 範圍

### 1.1 比什麼

| 編號 | 新方法 | 對照組（固定） | 在驗證集打分 |
|---|---|---|---|
| **H3R**（競賽二） | 在 inner 上從 P00–P06 選出的組合（第 3 節），規則 B | prereg02 R20：`results/prereg02/rank/pred_B.csv`（SHA `863a6258…`），帳上分等級平均 0.5890831292289003、每題平均 0.591969684316557（Python 3.12） | 最多 1 次，要先過閘門 |

- 競賽一不在 03 裡：不改方法、不打分。
  - 競賽一的不確定性在官方切法，只有 11/04 的 T1–T6 能回答。
  - 11/04 後如果防護落在第 3 層（只用 CNN），另寫登記。

### 1.2 刻意不放進 03 的

| 做法 | 不做的理由（本機證據） | 去向 |
|---|---|---|
| 在 fit 假題上訓練的整題聚合器（attention／Set Transformer） | ① fit 的逐盤輸出是訓練樣本內的（訓練 accuracy 33.9%，inner 25.6%），聚合器會學到信任過度自信的輸入。<br>② 要避開就得交叉擬合 K 個 CNN，GPU 成本乘 K 倍。<br>③ 改在 inner 上訓練：只有 636 人，連 10×10 矩陣校準都過擬合（樣本內 0.6105，照 tune 流程 0.5817）。<br>④ 盤數無限多時的上限只有 0.654–0.663，聚合方式不是瓶頸 | 不做 |
| neighbour label smoothing、SORD | 瓶頸是分不開相鄰兩級，把目標往鄰級抹平只會更分不開。Paplhám 等人（CVPR 2024）：資料夠多時 CE 就夠了。校準加規則 B 已經針對期望得分 | 不做 |
| 逐手、依等級條件化的落子模型 | 工程量大（新資料管線、逐手推論、新接縫）；在我們的規模沒有前例；和 03 綁在一起會拖到 11/04 的準備 | 04 候選（第 10 節），10/10 決定要不要登記 |
| 加長歷史，或加氣、提子特徵 | 見第 0 節第 6 條；AlphaGo 的表：有落子歷史後，加氣沒有增益 | 不做 |
| 改架構（加寬、加深、GAP 頭、SE block） | 證據是過擬合，不支持加容量；沒有本題證據 | 不做 |
| 用全部 8,007 人重訓（乙） | 本機驗證不了 | 看 11/04 的 T3 |
| 外部資料、預訓練模型、KataGo | 登記禁止；出題者寫「所有人都從零訓練」 | 永不做 |

## 2. Colab：兩份訓練 notebook（一次執行、同時跑）

- 產生器：`colab/build_prereg03_rank_nb.py`，產生 `colab/prereg03_rank_a.ipynb`、`colab/prereg03_rank_b.ipynb`。
  - 兩份只差 `ARM = 'A'／'B'` 一個常數和結果目錄，使用者不用改任何格子。
  - 底稿是已 commit 的 `prereg02_rank.ipynb`（先 assert SHA），用 `rep()` 修改。
- prereg01／prereg02 的檔案一律唯讀：`guard_write` 擋住兩個根目錄，開頭和結尾比對指紋。

### 2.1 兩份共同（寫死，不調）

| 項目 | 設定 |
|---|---|
| 訓練資料 | `load_rows(part="fit")`（5,767 人、720,307 盤），加外洩 assert；inner 只用來監看和產生逐盤輸出 |
| 特徵 | prereg01 的 500 層 uint8 分片（最後 250 個盤面 × 黑白），指紋 `FEAT_FINGERPRINT` 釘死。AI Pro 帳號從 prereg01 備份還原（實測 80 秒）；Pro+ 重抽（實測 355 秒），只有 A 寫備份 |
| 執色 | 只用 `load_rows` 的 `color` 欄，以 `(rank, game_id)` 對回分片 key。**不從 game_id 推**（`DATA.md` 第 3 條）。assert 每個分片 key 都查得到，黑白筆數等於 `df_train` 算出的值 |
| 目標玩家視角 | 在 GPU 上：目標玩家執白時，把每個盤面的兩層對調（第 2t 層＝目標玩家、2t+1 層＝對手），另加第 500 層（1＝目標玩家執黑），共 501 層。模型 `GoRankResNet(in_channels=501)`，其餘和官方相同（256×20、官方的頭、dropout 0.5）；`network.py` 不改 |
| D4 增強（只在訓練時） | 每筆均勻抽一種 s∈0..7，用預先算好的 [8,361] 座標排列表 `gather`。assert 結果等於既有的 `d4(x,s)`。執色層在 D4 下不變 |
| 最佳化 | AdamW lr 1e-3、weight decay 0.01（官方預設）；batch 256；CE，不做 label smoothing |
| 學習率排程 | 前 1,000 步線性 warmup 到 1e-3，之後 cosine 降到 0。**總步數 T＝33,768（＝12×2,814），和 worker 數無關**，取代 prereg02 寫死 2,814 的 assert |
| 精度 | 訓練用 bf16 autocast 加 channels_last，權重維持 fp32，不用 GradScaler。**所有逐盤輸出和 inner 監看都用 fp32**（TF32 設定和 prereg02 相同）。loss 不是有限值就停 |
| loader | worker 傳 uint8，到 GPU 再轉型；A 用 16 個 worker、B 用 32 個（上限 = vCPU 數 − 4）。每一輪（pass）的分片順序用 `Random(42·1000+pass)` 洗，shuffle buffer 1,000，rng 固定種子。解決 PREREG_02 rev 4 第 7 條「不能重現」的問題，但 cuDNN 本身仍不保證逐位元組相同 |
| 隨機種子 | 模型初始化 SEED＝42，兩份相同；D4 抽樣用 GPU generator，種子 42＋pass |
| 存檔與接續 | 每一輪結束存 `state.pt`：model、opt、scheduler、global_step、pass、history、RNG、recipe 設定（ARM、T、worker、AMP、增強、CAP）。設定不同就停。中斷後整輪從頭重跑，步數從已存的 global_step 接續，資料順序和沒中斷時相同 |
| **指定的 checkpoint** | 第 T 步的 `final.pth`，不另選 best。每一輪的 inner 監看（原方向 loss／accuracy，約 13 秒）只記錄 |
| 時間提示 | 前 300 步印出每秒步數和推估總時間，只提示、不影響判定。推估 > 3 小時就停下告訴 Claude，不算結果 |

### 2.2 A 和 B 只差一件事

- **A**：每一輪用全部 fit 棋。約 2,814 步一輪，共 12 輪。
- **B**：每位玩家最多算 200 盤，而且各級等量。每一輪用 hash 決定每盤收不收，比例寫死。
  - 收下的機率：q_g ＝ min(1, 200 / n_p) × 35,845 / E_L。
    - n_p＝這位玩家在 fit 的盤數。
    - E_L＝該級加上 200 盤上限後的總盤數：D 49,949、C 49,717、B 56,575、A 64,188、1D 35,845、2D 45,137、3D 45,024、4D 43,441、5D 46,263、6D 47,407。
  - 判定方式：u ＝ int(sha256(f"c10-prereg03|acc|{pass}|{rank}|{game_id}")[:16], 16) / 2^64，u < q_g 就收下。
  - 預期每一輪 358,450 盤（約 1,400 步、約 24 輪），每一級 35,845 盤。有效玩家數從 1,907 升到 3,184（`cap_stats2.json`）。
  - notebook 用 `df_train` 重算上面的 E_L，必須完全相同；每一輪實際收下的筆數記錄下來。
  - **200 和各級等量都是先寫死的，沒有調。**
- 兩份 T 相同（算力相同），所以差別只來自取樣方式。無論哪一份比較好，兩者的集成（P06）都有多樣性。

### 2.3 輸出（每份都 < 100 MB，才能進結果 zip）

| 檔案 | 內容 | 要和什麼一致 |
|---|---|---|
| `dumps/inner_keys.npz` | rank、game_id、player_id、color、label | 和 `results/prereg02/rank/dumps/inner_keys.npz` 逐陣列相同 |
| `dumps/inner_p03{a,b}_d4.npz` | fp32 logits [73,556, 8, 10]、`weights_sha256`、`syms` | `p02_rank_fuse.load_dump` 的格式 |
| `dumps/inner_p03{a,b}_s0_swap.npz` | 原方向、執色故意對調後的 logits [73,556, 10] | 只給第 3.1 節的接線檢查用 |
| `dumps/val_keys.npz` | question_id、slot、color、parsed | 和 prereg02 的逐陣列相同 |
| `dumps/val_p03{a,b}_d4.npz` | [19,156, 8, 10]，解析失敗的整列 NaN | `check_val_logits`；和 prereg02 一樣每題一個 batch |
| `_DONE_*.json`、`env.json`、`timings.json`、`run_log.json`、`ckpt/history.csv`、`config.json`、`leak_check.json` | GPU、torch、cuDNN、權重 SHA、秒數、每一輪的監看 | `write_dump` 的格式 |

- `final.pth` 和 `state.pt` 留在 Drive，不進 zip；只保留最後兩輪的 `.pth`。

### 2.4 notebook 的停止條件（依序；任一不過就說明原因後停下，不花 GPU 時間或不輸出）

1. GPU 型號：只接受 G4（RTX PRO 6000）。拿不到就等；等一天還拿不到，由 Claude 決定改用 A100（worker 上限 10），並先寫修訂。
2. 打包檔 SHA；特徵指紋；`load_rows` 外洩檢查；Drive 剩餘空間 ≥ 3 GB。
3. 執色查表：涵蓋率 100%，黑白筆數相符。視角轉換用一個人工棋盤做單元檢查。D4 gather 等於 `d4()`。B 的 E_L 必須相同。
4. 接續時設定必須相同；loss 必須是有限值。
5. 收尾核對：各逐盤輸出的列數和完成標記；prereg01／02 前後指紋相同。

### 2.5 帳號、GPU、時間、運算單元（推估，bf16 沒實測）

| | A | B |
|---|---|---|
| 準備（preflight、打包檔、load_rows、還原、pick_workers、解析驗證題） | 約 3 分 | 約 3 分 |
| 訓練 33,768 步（fp32 實測每步 151–154 ms → 85–87 分；bf16 快 1.5–2 倍 → 43–58 分） | 43–87 分 | 同左；loader 多讀一倍，最多再慢約 20% |
| 監看＋最後的 8 種對稱輸出（實測 145 秒）＋執色對調 | 約 5–6 分 | 約 7–8 分 |
| **合計** | **約 52–96 分，7.5–13.9 單元** | **約 60–115 分，8.7–16.7 單元** |

- G4 每小時 8.71 單元（二手實測，`COLAB_RUN.md:33`）。在 Pro+ 重抽特徵，每份再加約 6 分鐘、0.9 單元。每斷線一次，最多損失一輪（A 約 7 分、B 約 4 分）。**預留 40 單元。**
- **帳號規則（第 9 節第 0 步）**：
  - AI Pro 可用 ≥ 40 單元：就在 AI Pro 跑。特徵從 prereg01 備份還原，權重和 test02 在同一個 Drive，測試日只用一個帳號。
  - 不到 40：改在 Pro+ 跑。使用者先上傳 `C10_bundle_v1.zip`（247 MB）一次。採用的話，10/27 前把用到的 `final.pth`（每個約 105 MB）複製到 AI Pro 的 Drive，test03 用 SHA 核對。
- AI Pro 沒有背景執行：跑完前分頁要一直開著，電腦不能睡眠。

## 3. 本機：來源、組合、在 inner 上選（`tools/p03_rank_fuse.py`，只用 CPU）

- 只 import `p02_rank_fuse` 的數值函式：`basis`、`log_softmax`、`softmax`、`rule_b_np`、`pool_matrix`、`pooled`、`QSet`、`question_logits`、`fit`、`predict`、`metrics`、`PlayerBootstrap`、`load_dump`、`load_val_keys`、`check_val_logits`、`build_qset`、`read_question_meta`。
- `tools/p02_*.py`、`c10data.py`、`decide.py`、`results/prereg0{1,2}/**`、`results/test02/**` 一律不改。

### 3.1 `check`：新來源能不能用（在 `tune` 之前）

- 兩份的 key 檔和 prereg02 逐陣列相同；權重 SHA 和完成標記一致；T 步跑完。任一不符 → 那個來源不能用。
- **地板**：`final.pth` 在 inner 原方向的逐盤 accuracy < 24.15%（和 prereg02 的門檻相同）→ 拿掉用到它的組合。
- **執色接線**：inner 原方向的 accuracy，執色對調後減正確執色，差距 > 0.2 個百分點就判定為 bug。
  - 停下查；要重新輸出或重訓，先記修訂。
  - 差距（正確減對調）照實記錄，當作「模型用了多少視角資訊」的發現。
- 只記錄、不影響判定：每一輪監看的曲線；8 種對稱逐盤 accuracy 的分散程度（預期比 C5 的 21.4–25.6% 小）；訓練和 inner 的差距。
- 輸出 `results/prereg03/rank/check_report.json`。

### 3.2 `selfcheck`：R20 重現（過了才能 `tune`）

- 用 prereg02 已 commit 的 `pseudo_inner.json`、`opening_inner.npz`、`inner_p02e9_d4.npz` 重算 R20。
- 5 折交叉驗證的分等級平均必須等於 `tune_report.json` 的 0.5854702521（差距 ≤ 1e-12）；全部擬合資料重擬合的參數和 `frozen_config.json` 相差 ≤ 1e-12。

### 3.3 來源與組合（順序固定，也是平手時的優先順序）

- **來源**：每個來源 ℓ_s ＝ log_softmax(mean over 8 種對稱 log_softmax(z))，和 C5 的定義相同。
  - C5：p02e9，已 commit。
  - C6A：p03a。
  - C6B：p03b。
- **集成**：ℓ ＝ log_softmax(mean_s ℓ_s)，各來源等權，不加新參數。
- **每一題**：一律 log 合併、校準（a、γ、b），加開局項 w；公式和 R20 完全相同：p ＝ softmax(e^a (n/12)^γ (m + w·m_op + b))。
  - 盤數只進入 10 個等級共用的溫度。

| 組合 | 來源 |
|---|---|
| P00 | C5（＝R20，閘門的基準） |
| P01 | C5＋C6A |
| P02 | C5＋C6B |
| P03 | C5＋C6A＋C6B |
| P04 | C6A |
| P05 | C6B |
| P06 | C6A＋C6B |

- 帶 C5 的集成排在前面：它保留 C5 對方向敏感的訊號，萬一 D4 增強抹掉了某種真實的方向習慣，還能補回來；測試日也可以退回 test02。

### 3.4 在 inner 上怎麼選（本機驗證集完全不參與）

- **假題、分折、擬合**：原樣沿用 prereg02 的 `pseudo_inner.json`。
  - 擬合用第 0–49 輪、只收盤數 ≥20 的 508 人；評估用第 1000–1199 輪、636 人全收。
  - 在每一級內依玩家分 5 折。
  - 目標：分等級加權的 NLL 加 0.01‖b‖²；用 L-BFGS-B，界限、起點都和 prereg02 相同。
- **選法**：交叉驗證的分等級平均（規則 B）最高者；和最高分差距在 0.003 以內的，取清單中最前面的。
- **送分閘門**（兩條都要過）：
  - ① 選到的組合比 P00 高 ≥ +0.010。
  - ② 選到的組合的每題平均不 < P00。
  - 沒過 → 不打分，記為「未送分」，打分機會保留，R20 不變。
- **只記錄、不影響判定**：
  - 選到的組合減 P00 的配對差，分等級平均的 95% CI（每級內重抽玩家 1,000 次、`default_rng(0)`）。
  - 各級分數，特別看 A、1D、2D、3D。
  - 盤數分箱（5–7／8–12／13–20）的預測分布：檢查盤數沒有變成等級特徵，交給獨立審查。
- 選定後用全部擬合資料重擬合，寫出 `tune_report.json`、`frozen_config.json`（組合、參數、基底、各輸入的 SHA、工具的 commit）並 commit。之後才能跑 `apply`。

### 3.5 `apply`

- 要求 `frozen_config.json` 已 commit 而且和 HEAD 相同。
- 讀 `val_*` 的逐盤輸出、`results/prereg02/rank/opening_val.npz`、`rank_val_questions.csv`，寫出 `results/prereg03/rank/rank_val_probs.csv`。
- 接著跑 `decide.py`，再跑 `check_submission.py rank … --questions …`，commit。
- **本機接縫**：P00 在驗證題上的機率和 prereg02 的 `rank_val_probs.csv` 最大差距 ≤ 1e-12，而且規則 B 預測完全相同。證明新工具沒有改到 R20。

## 4. H3R：在本機驗證集打一次分

- 指令（一律用 `.venv`，Python 3.12）：
  ```
  .venv\Scripts\python.exe tools/score.py compare-rank results/prereg02/rank/pred_B.csv results/prereg03/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg03 H3R (A = prereg02 R20, reference re-read)" --json results/prereg03/rank/score_H3R.json
  ```
- **採用條件**（兩個都成立才採用）：
  - 分等級平均差值（prereg03 − R20）的 95% CI 下界 > 0（每級內分層重抽 10,000 次、seed 0）。
  - 每題平均差值的點估計不 < 0。
  - 一個成立、一個不成立 → 記為「結論依指標而定」，不採用。
- **合理性門檻**：新方法的分等級平均 < 0.54（R20 − 0.05）時先查 bug，要指出具體的程式錯誤才算。
- **對照組重讀**：A 側必須是 1,589 題，而且兩個指標和帳上的差距都 ≤ 1e-12；記錄打分用的 Python 版本。不符就停下來查，查清前不採用。
- **檢定力（推估，只用 inner）**：
  - 以 R10→R20 當替身：inner 上差 +0.0168，22% 的題換了答案。
  - 在驗證集各級題數下，差值的標準誤約 0.0062–0.0073。解析值是 0.0073；inner 一人一題的模擬顯示解析值高估約 15%。
  - 真正的差距 +0.010／+0.015／+0.020 時，CI 下界 > 0 的機率約 28–36%／54–68%／79–90%（`power.json`）。
  - 閘門仍然設 +0.010，理由：打一次分只用掉預算，打完之後還剩 2 次（04 和備用）。
- **只記錄**：各級差值、各盤數分箱。它們不影響採用。

## 5. 共通紀律

- **預算**：競賽二這次最多打 1 次（打完剩 2 次：04 和 split v2／查 bug 備用）；競賽一 0 次。查 bug 後重跑也算進預算，並記帳。
- **資料**：
  - Colab 沒有答案檔。訓練資料只經 `load_rows(fit)`。
  - `tune` 只接受 inner 的檔，並 assert player_id 都屬於 inner。
  - 開局 n-gram 沿用 prereg02 只用 fit 算的計數。所有 p03 工具和 notebook 都不含 `answers` 字串。
- **沿用的限制**：盤數只進入共用的溫度；不用一題對一人的結構；兩隊不交換模型輸出（03 只碰競賽二）；不用外部資料或預訓練模型。
- **交件模型**：甲（直接用本機驗證過的做法）。
- **揭露**（寫進 RESULTS 和報告）：
  - B 的 200 盤上限和各級等量、12 個 epoch 的步數、warmup 1,000 步，都是在看過 fit 的中繼資料之後、沒有調參就寫死的。
  - 開局的 M、α 是在 inner 上選的，再加上從 7 個組合裡選，inner 交叉驗證會略為樂觀。
  - 訓練不能逐位元組重現（cuDNN）。
  - 生成式 AI 協助的部分照規則揭露。

## 6. 採用之後：測試日工具擴充（test03；H3R 不採用就不做）

1. **`colab/build_test03_rank_nb.py` → `test03_rank.ipynb`**（只推論）：
   - 底稿是 prereg03 notebook 的輸出工具，只載入選定組合用到的 C6 權重（SHA 釘死）。
   - 三組題目：接縫、public、private。
   - 接縫：每個 C6 來源重算驗證題，原方向每題平均和 prereg03 的最大差 ≤ 1e-4（GPU 型號相同）或 ≤ 1e-3（不同），規則 B 不同 ≤ 4 題。
   - C5 仍由已驗證過的 `test02_rank.ipynb` 產生，不改。
2. **`tools/t03_rank.py`**：
   - `apply`：讀 test02 的 C5 輸出和 test03 的 C6 輸出。
     - 核對兩邊的 key 陣列和題目檔 SHA 相同（同一份 `C10_test_v1.zip`）。
     - 開局分數沿用 `t02_rank.py opening` 產生的 `opening_test.npz`（核對 SHA）。
     - 本機接縫：選定組合在驗證題上的結果和 `results/prereg03/rank/rank_val_probs.csv` 比，最大差 ≤ 同一容許差、規則 B 不同 ≤ 4 題。
     - 每個來源各自要求三組輸出來自同一個執行環境，來源之間可以不同。
   - `fallback-fit`：選定組合拿掉開局項的版本（R10 的對應），給 T1.4／T1.5 不過時用。在 inner 擬合輪上重擬合，做參數自我檢查後 commit。沒有在本機驗證集打分。
3. **`tools/official_log.py`**：加 prereg03 的方法標籤（選定組合和它的無開局備案）。改完重跑它的測試。
4. **測試與文件**：用 `t02_fake_tests.py` 做演練（慢速測試）；更新 `docs/TESTDAY.md`、`COLAB_RUN.md`。
5. **登記**：在 PREREG_03 另寫一個修訂版，寫明測試日的程序，11/04 前 commit。
6. **測試日順序**：
   - 第一次上傳照原計畫，用 test02 的 R20（抓格式錯誤）。
   - test03 的接縫都過之後，才上傳 03 的組合。
   - test03 兩次都沒過 → 維持 R20 並記錄。
   - 最後一次上傳只依 H3R 的結論決定，在 11/23 前完成。
7. **成本**：Claude 約 3–4 天；測試日多跑一份約 10–20 分鐘、1.5–3 單元（推估）。

## 7. 順序與時程

| 日期 | 事項 | 誰 |
|---|---|---|
| 9/27 | PREREG_03 rev 1、`design/`（診斷腳本和輸出）commit | Claude |
| 9/28–10/03 | 產生器和兩份 notebook、差異檔、`p03_rank_fuse.py`、`tests/test_p03_*.py`、CPU 冒煙測試（先重建 prereg01／02 的冒煙樹，約 15 分鐘）、獨立審查、rev 2（實作偏離）、`COLAB_RUN.md` 的 prereg03 一節 | Claude |
| 10/03 | 使用者查 AI Pro 的可用運算單元，決定用哪個帳號（第 2.5 節） | 使用者 |
| 10/04–10/06 | 在 Colab 同時跑兩份，下載兩個 zip（第 9 節） | 使用者 |
| 收到後 1–3 天 | ① 解開到 `results/prereg03/rank_{a,b}/`，先 commit；② `check` → `selfcheck` → `tune`，commit `frozen_config.json`；③ `apply` → `decide` → `check_submission`，commit；④ 過閘門才跑第 4 節的指令；⑤ 帳本、分數 JSON、`RESULTS.md`（獨立核對）、PORTFOLIO_MAP 一起 commit | Claude |
| 10/10 前後 | H3R 的結論出來；決定要不要登記 04（第 10 節） | 使用者決定 |
| 10/11–10/27 | 採用的話：做 test03 工具，登記修訂版 | Claude |
| **10/24** | **03 最晚的打分日**；過了就放棄 03，維持 R20 | — |
| 10/28–11/02 | Colab 彩排（test02＋test03），約 1–1.5 小時 | 使用者（選做） |
| 11/01 | 11 月投入多少，由使用者決定 | 使用者 |
| 11/04 起 | 照 `TESTDAY.md` | 兩者 |

- **中斷時怎麼接**：
  - 各步驟的結果在 commit 之後才算完成。
  - Colab 的接續靠 `state.pt`；沒跑完 T 步的權重一律不用。
  - 任何一份在開跑後 7 天內沒有跑完，就拿掉那個來源（清單縮小，不另寫修訂），兩份都沒跑完就是「未送分」。

## 8. 風險與防護

| 風險 | 防護 |
|---|---|
| 執色接反或對錯盤 | 只用 `color` 欄；查表涵蓋率和黑白筆數 assert；人工棋盤單元檢查；key 陣列和 prereg02 相同；執色對調檢查（> 0.2 個百分點就停） |
| D4 增強抹掉真實的方向習慣 | 清單裡有帶 C5 的集成，而且排在前面 |
| B 的上限反而變差 | P01、P04 只用 A；選法會自己排除 |
| bf16 沒加速，或數值不穩 | 前 300 步印出推估時間；loss 不是有限值就停；逐盤輸出一律 fp32；Claude 可以指示改 fp32（T 不變，先記修訂） |
| 拿不到 G4、斷線 | G4 優先；每一輪存檔，接續最多損失一輪；等一天後 Claude 決定 A100 |
| AI Pro 的運算單元不明 | 先查，≥40 才用；否則用 Pro+，事後複製權重 |
| 驗證集的檢定力不足 | 第 4 節的推估寫明；CI 含 0 就不採用，R20 照舊 |
| inner 略為樂觀、多重比較 | 只有 7 個組合；平手取前面的；閘門 +0.010；每場只打 1 次；揭露 |
| 盤數變成等級特徵 | 校準公式不變；擬合只用盤數 ≥20 的人；報告各分箱的預測分布；獨立審查 |
| 測試日步驟變多 | 先用 test02 的 R20 上傳；03 等接縫過才用；彩排；test03 失敗就退回 R20 |
| Colab 的 torch／cuDNN 版本漂移 | 接縫容許差和 PREREG_02 rev 4 第 8 條相同；放寬要另寫修訂 |
| 寫壞 prereg01／02 或 test02 | `guard_write`、前後指紋比對；p02／t02 的工具不改（`official_log.py` 例外，只在採用後、記修訂才改） |

## 9. 使用者要做的事

0. **10/03 前**：用 AI Pro 帳號開 Colab，「執行階段」→「查看資源」，把可用單元的數字告訴 Claude。≥ 40 就用 AI Pro，否則用 Pro+；用 Pro+ 的話，先上傳 `C10_bundle_v1.zip` 一次。
1. 把兩份 notebook 放進 `MyDrive/C10/`（舊的移到 `old_*`），兩個分頁各開一份，都選 **G4**，各按一次「全部執行」。
2. 跑完前分頁一直開著、電腦不睡眠（AI Pro 沒有背景執行），約 1–2 小時。出現紅字就截圖給 Claude，不要自己重跑。
3. 下載兩個結果 zip，把路徑告訴 Claude；按「中斷連線並刪除執行階段」。
4. 採用而且是在 Pro+ 訓練的：10/27 前照 Claude 的步驟把權重檔複製到 AI Pro 的 Drive。
5. 10/28–11/02 選做彩排；11/01 決定 11 月的投入；11/04 起照 `COLAB_RUN.md`「測試集」，每次上傳前在對話裡說 OK。

## 10. 04 候選（不在 03；10/10 前後決定要不要登記）

- **逐手、依等級條件化的落子模型**：共用主幹接 10 個 policy 頭，只訓練目標玩家的手；每盤取 10 級的 log 似然加總，當融合的第三項（用似然，不用逐局面分類）。
- **成本**：先試跑約 0.5 小時，正式跑 2–2.5 小時、18–22 單元（推估）。還要新的逐手推論和接縫。
- **條件**（都要成立）：
  - 03 在 10/10 前有結論。
  - 使用者同意再跑一次 Colab。
  - 登記和測試日工具能在 11/03 前做完；否則延到 11 月，由 11/01 的決定處理。
- 預算：04 用剩下 2 次裡的 1 次。

---

**本草案產生的檔案**（都只用 split 清單和 inner 的 OOF；沒有讀任何 `*answers*`、沒有跑 `score.py`、沒有用 GPU、repo `git status` 乾淨）：
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03\prereg_draft\cap_stats.py`、`cap_stats.json`、`cap_stats2.py`、`cap_stats2.json`：每人盤數上限和各級等量的數字。
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03\prereg_draft\power.py`、`power.json`：用 inner OOF 的 R10／R20 推估 H3R 的檢定力。