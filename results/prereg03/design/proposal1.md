# 事前登記 03（提案）：競賽二用 recipe v2 從頭重訓，把新模型接進 prereg02 的合併與校準

建議檔名：`docs/PREREG_03_RECIPE_V2.md`。這份是提案內容，repo 沒有改動。

## 結論

- **只做競賽二，只跑一次 Colab。** 用目前的 AI Pro 帳號和 G4 跑 notebook A（必跑）。運算單元夠的話，同時跑 notebook B（同一套 recipe，換 seed）。
  - 牆鐘時間約 50–94 分鐘。A 約 7–14 單元，A＋B 約 14–28 單元。
  - 從 prereg01 的特徵備份還原，不重抽特徵。
- **本機沿用 prereg02 的整條管線。** 新寫 `tools/p03_rank_fuse.py`，只 import p02 的數值函式，只多一個參數 v（新模型的權重）。候選只有 3 個組合：
  - T00：R20 本身，當自我檢查。
  - T01：R20 的 C5 加上 v 倍的新模型（主假設）。
  - T02：只用新模型。
- **打分前多一道閘門，確認這次打分大概率驗得出來。**
  - 我用 inner 的 R20 對 R10 當代理估算過：一個「小改動」在本機驗證集上，差值的標準誤約 0.0073。
  - 所以真實進步只有 +0.010 時，驗證集能判定採用的機率只有約 27%。+0.015 約 53%，+0.020 約 77%。
  - 因此閘門定成：inner OOF 的進步 ≥ max(0.010, 1.96 × 推估的驗證集標準誤)，實際門檻大約是 +0.015。沒過就不打分，這次機會留給 04。
  - 新模型的逐盤輸出照樣保留，04 可以直接當合併來源，不算白跑。
- **測試日工具只在採用之後才擴充。** test02 一個字都不改，當作退路。測試日先照原計畫上傳 R20，03 的組合最後才上傳。
- **時程**：
  - 登記加程式在 10/02 前完成。
  - 使用者 10/03（六）或 10/04（日）跑 Colab。
  - 10/05 前打分。
  - 採用的話，測試日擴充在 10/16 前完成。
  - 10/20 還沒打分，03 就不進測試日。

---

## 0. 為什麼做（都沒看答案）

- **瓶頸在每盤棋的判讀，和跨玩家的泛化。**
  - inner 上，校準、序數決策、既有權重的對稱或 epoch 組合、便宜的 SGF 特徵，各種 CPU 改法都拿不到 +0.002 以上。
  - 規則 B 的「完美校準」上限是 +0.002。
  - 神諭（相鄰兩級永遠選對）是 0.795，所以剩下的差距在「分辨相鄰兩級」。
  - 以上都是任務 B 的結果，在 `scratchpad/p03/B_error/`。
- **prereg02 的接著訓練在記訓練玩家。**
  - epoch 5→9：訓練 accuracy 26.3→33.9%，inner accuracy 25.15→25.58%，inner loss 1.930→1.980。
  - 來源：`results/prereg02/rank/ckpt/history.csv`。
  - 訓練時的 D4 增強和學習率排程，正好對準這個問題。
- **不用重抽特徵。**
  - 目標玩家視角：翻色再加 1 層執色。
  - D4 增強、cosine LR、bf16 都可以在 GPU 上從現有的 500 層 uint8 分片算出來。
  - 執色用 `(rank, game_id)` 對回 `load_rows` 的 `color` 欄。
  - 以上見任務 A 的可行性評估。
- **為什麼選「加一個來源」，不選「換模型」。** C5（epoch_9）已經驗證過，測試日工具也都做好了。新模型的輸入編碼不同，最有機會帶來集成的多樣性。另外，驗證集上比的是 C5＋新模型對 C5，兩者高度相關，差值的變異比較小，打分的檢定力比較高。

## 1. 要比什麼

- **H3R**：inner 上選出的 03 組合，在本機驗證集的分等級平均高於 prereg02 R20（`results/prereg02/rank/pred_B.csv`）。
  - R20 帳上的數字：分等級平均 0.5890831292289003，每題平均 0.591969684316557，SHA 是 `863a6258…`。
- 競賽一不在 03 的範圍：不打分，預算不動（見第 5 節）。

## 2. Colab：`colab/prereg03_rank_s0.ipynb`（A）和 `prereg03_rank_s1.ipynb`（B）

兩份由 `colab/build_prereg03_rank_nb.py` 產生，只差 SEED 常數，使用者不用改任何東西。

### 2.1 Recipe（寫死）

| 項目 | 設定 |
|---|---|
| 訓練資料 | 只用 fit 玩家（5,767 人、720,307 盤）。特徵從 prereg01 的 Drive 備份還原（約 80 秒），特徵指紋要和 prereg01 相同。inner（640 人）只拿來監看和產生逐盤輸出，驗證玩家完全不碰 |
| 輸入 | 501 層：第 2t 層是目標玩家的子，第 2t+1 層是對手的子，第 500 層是 1（目標玩家執黑）。在 GPU 上從 uint8 轉。執色一律查 `color` 欄，不用 game_id 推 |
| 模型 | 官方 `GoRankResNet(in_channels=501)`，256 通道 × 20 個 block，不改 `network.py`。從頭訓練，不接 prereg01/02 的權重 |
| 增強 | 每筆抽一種 D4，用 [8,361] 的座標排列表加 `gather` 做，並 assert 結果和 `d4()` 相同。generator 的種子是 SEED＋epoch |
| 最佳化 | AdamW，lr 1e-3，wd 0.01，batch 256。前 1,000 步線性 warmup，接 cosine 降到 0。總步數 T 用 `expected_train_steps()` 依實際 worker 數算，寫進 state |
| epoch | 12 個。**指定 checkpoint 是 `epoch_12.pth`**，不挑 best，inner 只當監看 |
| 精度 | 訓練用 bf16 autocast 加 channels_last，權重維持 fp32，不需要 GradScaler。逐盤輸出一律用 fp32。CPU 冒煙測試時關掉 autocast |
| loader | 傳 uint8，16 個 worker（A100 時 10 個）。用 `detshuffle` 並傳 seed，解決 PREREG_02 rev 4 第 7 條「shuffle 不能重現」的問題 |
| seed | A＝0，B＝1。其他完全相同 |

### 2.2 自我檢查與停止條件（任一不過就停，先打包紀錄）

1. **GPU**：只接受 G4 或 A100（標準版）。型號不符時，還沒開始算就停。
2. **只讀**：prereg01、prereg02 目錄的指紋在開頭、結尾各比對一次。
3. **執色接線**：
   - fit、inner、驗證題三邊的執色查表覆蓋率都要是 100%。
   - 翻色的單元檢查：執黑的樣本不變，執白的樣本兩層對調，第 500 層是常數。
   - D4 gather 的結果和 `d4()` 相同。
4. **速度**：前 300 步計時，印出推估的總時間。推估超過 150 分鐘（G4）或 180 分鐘（A100）就停下回報，不空燒單元。
5. **數值**：loss 出現 NaN 或 inf 就停。
6. **接續**：每個 epoch 存 `state.pt`，內容照可行性評估第 6 節。recipe 設定不同就停。`.pth` 只留最後兩個。
7. **epoch_12 能不能用**：兩項都成立，這個來源才能用，不成立就拿掉。
   - inner 逐盤 argmax accuracy（原方向，fp32）≥ 24.15%，和 prereg02 用的下限相同。
   - 用正確執色算的 inner accuracy，嚴格高於把執色對調後算的。兩個數都記錄。

### 2.3 輸出（格式照可行性評估第 7 節，逐陣列和 prereg02 相同）

- `inner_keys.npz`、`val_keys.npz`：必須和 prereg02 的逐陣列相同。
- `inner_p03s{0,1}_d4.npz`：[73,556, 8, 10]。
- `val_p03s{0,1}_d4.npz`：[19,156, 8, 10]，解析失敗的整列是 NaN。
- `_DONE_*.json`，記 GPU、torch、cuDNN、權重 SHA-256。
- `ckpt/history.csv`、`timings.json`（含 bf16 實測速度）、`env.json`、`config.json`、`run_log.json`、`leak_check.json`、只讀指紋。
- 選配：每個 epoch 一份原方向的 inner 輸出，每份 2.9 MB。**只拿來畫學習曲線，任何組合都不用。**
- 權重留在 Drive，SHA 寫進結果檔，zip 不含權重。

### 2.4 帳號、GPU、成本

| | 只跑 A | A＋B 同時跑 |
|---|---|---|
| 牆鐘時間（G4） | 約 50–94 分鐘 | 同左 |
| 運算單元 | 約 7–14 | 約 14–28 |
| 開跑前「可用量」門檻 | ≥ 35 | ≥ 50 |

- **門檻怎麼來**：最壞情況的成本，加上 1 次斷線（約 1.5 單元），再留約 20 單元給測試日的 test02、test03、重跑和彩排。
- **B 跑不跑，只看開跑前的可用量**，和任何結果無關。B 沒跑、中斷沒補完，或第 2.2 節第 7 條不過，就從組合裡拿掉，這不算偏離。
- **AI Pro 的可用量連 A 的門檻都不到**：改用 Pro+ 帳號。這會多出三件事：上傳打包檔 247 MB、重抽特徵約 6 分鐘、把權重約 105 MB 複製到 AI Pro 的 Drive（SHA 釘死）。這條路要先記一筆修訂，再做程式，而且必須在任何 03 輸出產生之前。
- **AI Pro 沒有背景執行**：跑完前分頁不能關，電腦不能睡眠。B 要等 A 印出 `device cuda` 之後才開，避免 Drive 建出兩個同名資料夾。

## 3. 本機：`tools/p03_rank_fuse.py`（只用 CPU）

### 3.1 來源

- **import p02 的函式**：`basis`、`log_softmax`、`softmax`、`rule_b_np`、`pool_matrix`、`pooled`、`QSet`、`metrics`、`PlayerBootstrap`、`load_dump`、`load_val_keys`、`check_val_logits`、`build_qset`、`read_question_meta`。
- **不改 p02 的檔**：`p02_*.py`、`c10data.py`、`decide.py` 都不動，工具開頭核對它們的 SHA-256，和 HEAD 相同才執行。

| 代號 | 內容 |
|---|---|
| C5 | prereg02 epoch 9，8 種對稱（沿用） |
| C8 | 03 新模型：B 可用時，是 s0 和 s1 共 16 個方向的 log_softmax 等權平均；B 不可用時，只用 s0 的 8 個方向 |

- n 是「本組合用到的每個來源都有有限 logits 的盤數」。驗證題上 C5 和 C8 的解析集合必須相同，不同就停下查 bug。

### 3.2 校準形式

只比 R20 多一個 v：

p = softmax( e^a · (n/12)^γ · ( m₅ + v·m₈ ＋ w·m_op ＋ b ) )

- v ∈ [0, 4]，其餘的界限、罰項 0.01·‖b‖²、L-BFGS-B、兩個起點都和 p02 相同。
- 盤數 n 仍然只進入 10 個等級共用的溫度，符合登記的規則。
- 報告照 prereg02 列出各盤數分箱（5–7／8–12／13–20）的預測分布。
- 任務 B 的診斷已經用過同樣的寫法：讓 p01 當第二項、權重自己擬合，5 折都擬合成 0。所以這個寫法可行。

### 3.3 組合清單（順序固定，也是平手時的優先順序）

| 組合 | 分數來源 | 參數 | 角色 |
|---|---|---|---|
| T00 | C5 | a, γ, b, w | ＝R20，自我檢查 |
| T01 | C5 ＋ v·C8 | a, γ, b, w, v | 主假設：舊模型加新模型 |
| T02 | C8 | a, γ, b, w | 只用新模型 |

- 全部用 log 合併、加校準、加開局項。prob 合併在 02 每一對都輸，不再列入。
- 假題、5 折、擬合輪和評估輪、開局 n-gram（`opening_inner.npz`、`opening_val.npz`、`pseudo_inner.json`）全部沿用 prereg02 已 commit 的檔。

### 3.4 在 inner 上選

- 在 T01、T02 之間取交叉驗證的分等級平均（規則 B）較高者。兩者差距在 0.003 以內時取 T01。
- 選定後用全部擬合資料重擬合。
- **同時算好退路參數**：選定組合拿掉開局項的版本（w＝0，同一套 tune 程序），寫進同一份 `results/prereg03/rank/frozen_config.json`，一起 commit。

### 3.5 送分閘門（三項都成立才打分）

- **G1**：Δ ≥ max(0.010, 1.96 × SE_val)。
  - Δ 是選定組合減 T00 的 inner OOF 分等級平均。
  - SE_val ＝ √(Σ_l Var_l(d) / N_l) / 10，d 是 inner 評估輪上每題的得分差，N_l 是驗證集各級題數（VALIDATION.md 公開的 178、163、207、308、87、122、124、128、131、141）。
  - 這條的意思是：把 inner 的進步當真，驗證集判定採用的機率要 ≥ 50%。
- **G2**：inner 依玩家 bootstrap（每級內重抽 1,000 次，seed 0）的 95% 下界 > 0。
- **G3**：inner 每題平均的差值 ≥ 0。
- **G1 門檻的依據**（inner 的 R20 對 R10 當代理，`scratchpad/p03/maxgain/power.json`）：
  - 決策不同的題 21.6%，驗證集標準誤約 0.0073。
  - 決策不同 25% 時，採用機率：進步 +0.010 是 27%，+0.015 是 53%，+0.020 是 77%，+0.030 是 98%。
- 沒過閘門就記為「未送分」，保留這次打分機會。C8 的逐盤輸出留給 04 當來源（04 要另寫登記）。
- `frozen_config.json` 另外記：Δ、SE_val、推估的採用機率、T01 的 v、各級 OOF。

### 3.6 本機重現核對（在 tune 之前，任一不過就停）

1. T00 的 inner OOF 分等級平均必須正好是 0.5854702521，而且每一列都和 `tune_report.json` 的 R20 相同。
2. T00 套到驗證題的每題機率，和 `results/prereg02/rank/rank_val_probs.csv` 最大差 ≤ 1e-12，規則 B 不同 0 題。
3. `inner_keys.npz`、`val_keys.npz` 和 prereg02 逐陣列相同。
4. 新的驗證題逐盤輸出，在凍結前只做形狀、NaN 樣式和雜湊檢查，不讀內容。

### 3.7 順序由程式強制（沿用 p02 的做法）

- `tune` 要求 Colab 結果和工具都已 commit。
- `apply` 要求 `frozen_config.json` 已 commit、和 HEAD 相同，而且工具沒改過。

## 4. H3R：本機驗證集只打一次分

```
.venv\Scripts\python.exe tools/score.py compare-rank results/prereg02/rank/pred_B.csv results/prereg03/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg03 H3R (A = prereg02 R20, reference re-read)" --json results/prereg03/rank/score_H3R.json
```

- **採用條件**：
  - 分等級平均差值的 95% CI 下界 > 0（每級內分層重抽 10,000 次，seed 0）。
  - 每題平均差值的點估計不 < 0。
  - 兩者都成立才採用。結論和指標有關時，照 VALIDATION 記錄。
- **合理性門檻**：新方法的分等級平均 < 0.52 時，先查 bug，要指出具體的程式錯誤才算。R20 在驗證集是 0.589，選定組合在 inner 至少要 ≥ 0.600 才過得了閘門。
- **對照組重讀**：A 側要和帳上 prereg02 R20 那一列題數相同，差距 ≤ 1e-12，並記錄打分用的 Python 版本（3.12）。
- **打分前**：先過 `decide.py` 和 `check_submission.py rank`。
- **帳本**：競賽二整季 4 次，已用 1 次（02）。03 最多用 1 次，還剩 2 次（04 和備用）。
- **交件模型**：用甲，直接用本機驗證過的模型，不拿全部玩家重訓。

## 5. 競賽一

- 03 不動競賽一，不打分。
- 競賽二的模型輸出不進競賽一，反之亦然。
- 官方切法 11/04 才知道，原本的三層防護和 T4 檢查照 PREREG_02 做。

## 6. 測試日工具擴充（只在 H3R 採用後才做；test02 完全不改）

1. **`colab/build_test03_rank_nb.py` → `test03_rank.ipynb`**：
   - 只算 C8 用到的權重（SHA 釘死），產生接縫（驗證題）、public、private 三組逐盤輸出。
   - notebook 的接縫：驗證題的每題平均機率和 prereg03 的驗證題輸出比，同型號 GPU ≤ 1e-4、不同型號 ≤ 1e-3；規則 B 不同 ≤ 4 題。
   - GPU 用 G4 或 A100。每次約 1.5–2 單元。
2. **`tools/t03_rank.py apply`**：
   - 讀 test02 的 C5 輸出、test03 的 C8 輸出，和 `t02_rank.py opening` 產生的 `opening_test.npz`。
   - 核對兩邊的 key 相同、題目檔 SHA 相同。
   - 本機接縫：選定組合在驗證題上重算，和 prereg03 的 `rank_val_probs.csv` 比，容許差同上。
3. **退路**（照順序）：
   - T1.4 或 T1.5 沒過（開局不可靠）：用 03 組合的無開局版（第 3.4 節已 commit 的參數）。
   - test03 重跑一次仍然沒過：用 R20。開局也不可靠時用 R10。這兩條都走現成的 test02 流程。
4. **`official_log.py`**：新增方法標籤 `P03`、`P03-noopen`。
5. **上傳順序**：
   - 第一次上傳照原計畫用 R20，只為了抓格式錯誤。
   - 03 組合在兩份接縫都過之後才上傳，當最後一版。
   - public 分數低 ≥ 0.05 只追查錯誤，不換方法。
6. **登記與驗證**：
   - 測試日的程序寫成 PREREG_03 修訂，在 11/04 前 commit。
   - 用 `t02_fake_tests.py` 的假測試包 dry-run，更新 `TESTDAY.md` 和 `COLAB_RUN.md`，並做獨立審查。
7. **工作量**：約 3–4 天（可行性評估第 8 節）。

## 7. 風險與防護

| 風險 | 防護 |
|---|---|
| 執色接反或對不上 | 覆蓋率 100%；翻色單元檢查；真執色的 accuracy 要高於對調執色，不成立就拿掉該來源 |
| bf16 沒加速，或數值出問題 | 前 300 步實測並有時間上限；NaN 就停；逐盤輸出用 fp32 |
| 盤數變成等級特徵 | 校準形式不變（n 只進共用溫度）；報各盤數分箱的預測分布；獨立審查專門檢查 |
| 打分沒有檢定力 | G1 的門檻是依推估的標準誤定的；沒過就留給 04 |
| inner 上的「贏家詛咒」 | 只有 2 個候選；開局的 M、α 是在 inner 定的，報告揭露 |
| AI Pro 運算單元不夠 | 開跑前看可用量；B 可以不跑；Pro+ 路線要先修訂 |
| 拿不到 G4 | 可以用 A100（10 個 worker），速度門檻放寬到 180 分鐘，實際用哪種記下來 |
| 斷線 | 每個 epoch 存 state，最多損失 1 個 epoch；出現紅字後使用者不自己重跑 |
| 結果不能逐位元組重現（cuDNN、bf16） | 報告寫明；測試日靠 SHA 釘住權重，加接縫容許差 |
| 測試日變複雜 | 只有採用才擴充；test02 不動、當退路；R20 先上傳 |
| 時間不夠 | 10/20 前沒打分，03 就不進測試日（仍可寫進報告） |

## 8. 順序與時程

| 日期 | 事項 |
|---|---|
| 9/27（日） | PREREG_03 rev 1 commit，登記全文、這份設計和 `power.py` 的依據。使用者回報 AI Pro 的「可用量」（截圖一張） |
| 9/28–10/01 | builder 和兩份 notebook、`p03_rank_fuse.py`、`tests/test_p03_*.py`、CPU 冒煙測試（小型假分片） |
| 10/01–10/02 | 獨立審查與修正；rev 2 記實作偏離；`COLAB_RUN.md`「事前登記 03」一節；更新上傳副本 |
| 10/03（六）或 10/04（日） | 使用者跑 Colab，約 1–1.5 小時，分頁開著、電腦不睡眠 |
| 收到 zip 後 1–2 天內（目標 10/05） | 解開後 commit → repro → tune → 凍結 commit → apply、decide、check → 過閘門才打 H3R → RESULTS、地圖 |
| 10/10–10/11 | 保留的重跑窗口：Colab 出錯要修 builder 的話，照「Colab 出錯時」流程走 |
| 採用：10/06–10/16 | 做 test03 工具、測試日程序修訂、假測試 dry-run、審查 |
| 10/20 | 截止：H3R 沒打分，03 不進測試日 |
| 10/28–11/02 | 可選的 Colab 彩排：test02 加 test03 用假測試包，約 1.5 小時、約 7 單元 |
| 11/01 | 使用者決定 11 月投入多少 |
| 11/04 | 測試日照 `TESTDAY.md` 做（採用的話已含 test03） |

不採用或沒送分時，測試日工具不變，10/06 起評估 04（逐手、依等級條件化的落子模型，C8 可以當來源）。

## 9. 使用者要做的

1. 在 AI Pro 帳號的 Colab 按「執行階段」→「查看資源」，回報「可用量」。
2. 照 `COLAB_RUN.md` 新增的一節跑 A（可用量夠就同時跑 B），把 zip 的路徑給 Claude。
3. 採用的話，可以選擇在 10/28–11/02 做彩排。

## 10. 揭露（寫進報告）

- 開局的 M、α 在 inner 定，inner 交叉驗證略為樂觀。
- v 和校準參數在 inner 的擬合輪擬合。
- 退路（無開局版）沒有在驗證集打分。
- bf16 和 cuDNN 讓結果不能逐位元組重現。
- 生成式 AI 協助的部分照規則揭露。

## 11. 刻意不放進 03 的

| 做法 | 理由 |
|---|---|
| 逐手、依等級條件化的落子模型 | 上限最高，但工程是新的，測試日也要新的逐手推論 → 04 |
| label smoothing（R3B） | 證據弱；B 的名額給 seed 複本，比較能確定有集成增益 |
| 加寬、加深、GAP 頭 | 已經過擬合，沒有本題的證據 |
| 更長的盤面歷史、氣或提子層 | 長棋沒有比較差；AlphaGo 的表顯示加氣是 +0.0 |
| 用全部 8,007 人重訓（乙） | 本機驗不到 |
| 校準細化、SGF 特徵 | inner 上都 ≤ +0.002 |

---

依據檔案（只用 CPU、只讀 inner 的 OOF，沒碰驗證資料）：
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03\maxgain\power.py`
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03\maxgain\power.json`