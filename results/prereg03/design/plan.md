# PREREG_03 建議定稿：競賽二 recipe v2 雙臂重訓，和 R20 在 log 空間集成

建議檔名 `docs/PREREG_03_RECIPE_V2.md`。主體採用 P2（gain）的雙臂設計，截止點、測試日順序和運算單元分級取自 P3，R20 重現核對、工具 SHA 核對和檢定力紀錄取自 P1。兩位評審點出的缺陷都已修正，另外新增一項不讀答案的執色語意檢查。

這份定稿是我在 9/26 對 repo 做唯讀核對後寫的。沒有改動 repo，沒有讀任何 `*answers*`，沒有跑 `score.py`，也沒有用 GPU。核對結果：
- 帳上 R20 的數字無誤：0.5890831292289003／0.591969684316557，SHA `863a6258…`。
- PREREG_02 的預算在 :159，凍結規定在 §9.1。
- p02 的常數：`GATE=0.010`、`SELECT_TOL=0.003`、`P02E9_MIN_ACC=24.15`、`N_BOOT=1000`、`BOOT_SEED=0`，`SOURCES` 的寫法也核對過。
- 本機驗證題的 `color_i` 由 `tools/make_val.py` 直接取訓練 CSV 的 `color` 欄。
- 官方 README:94 對 `color_i` 只寫「對應棋譜的執色」，沒有說是目標玩家的顏色。第 4.4 節的 κ 檢查就是為這一點補的。

---

## 1. 一句話結論

**10 月只做競賽二的一件事**：用新配方在 Colab 從零訓練兩個棋力模型，再回本機和現在的 R20 合併。在訓練玩家（inner）上過了閘門，才在本機驗證集打這季唯一的一次分。確定比 R20 好，11/04 才用；任何一步出錯或趕不上截止點，就維持 R20，測試日工具 test02 照舊。

- **新配方有四點**：
  - 模型知道哪一方是要猜的玩家。
  - 訓練時隨機旋轉或鏡射棋盤。
  - 學習率先升後降。
  - 訓練步數固定。
- **兩個模型（兩臂）只差一件事**：B 臂每位玩家最多算 200 盤，而且各級等量。
- **使用者跑 Colab 一次**：兩個分頁同時跑，約 1–2 小時。
  - 合計約 16–31 個運算單元，開跑前帳上要有 50 個以上。
  - 採用的話，測試日多跑一份約 10–20 分鐘、1.5–3 單元的 notebook。
- **能不能證明有效（推估）**：新方法真正好 +0.010／+0.015／+0.020 時，本機驗證集證明得出來的機率約 3 成／5–7 成／8–9 成。證明不出來就維持 R20。
- **使用者要做的**：
  - 10/03 回報兩個帳號的「可用量」。
  - 10/04 跑 Colab。
  - 採用的話，10/28–11/02 可以選做彩排。
  - 11/04 照 `TESTDAY.md` 做。
- **最晚時點**：10/19 打分；11/03 測試日工具完成；11/16 03 在測試集要成功執行過一次。任一沒趕上，這一季就用 R20。

---

## 2. 比什麼、在哪裡選、怎樣算成功

### 2.1 唯一的比較 H3R（競賽二，本機驗證集打 1 次分）

| | 內容 |
|---|---|
| 新方法 | inner 上從 P00–P06 選出的組合（第 4.3 節），規則 B |
| 對照組 | prereg02 R20：`results/prereg02/rank/pred_B.csv`（SHA `863a6258…`），帳上分等級平均 0.5890831292289003、每題平均 0.591969684316557 |
| 指令 | 見第 4.5 節第 8 步 |

**採用條件**（兩項都成立才採用）：
- 分等級平均差值（03 − R20）的 95% CI 下界 > 0：每級內分層重抽 10,000 次、seed 0。
- 每題平均差值的點估計不 < 0。
- 只有一項成立，記為「結論依指標而定」，不採用。

**合理性門檻**：新方法的分等級平均 < 0.54（R20 − 0.05）時，先查 bug。要指出具體的程式錯誤才算 bug；找不到就照結果判定。

**對照組重讀**：A 側要是 1,589 題，兩個指標和帳上的差距都 ≤ 1e-12，並記錄打分用的 Python 版本（`.venv`，3.12）。不符就停下，查清之前不採用。

**預算**：
- 競賽二整季 4 次，已用 1 次（02）。03 最多用 1 次，打完剩 2 次（04 和備用）。
- 查 bug 後重跑也算進預算，並記帳。
- 閘門沒過就記為「未送分」，這次打分機會保留。

**交件模型**：甲，也就是直接用本機驗證過的做法，不拿全部 8,007 人重訓。

### 2.2 只在 inner 上決定的東西（本機驗證集完全不參與）

- **要調的只有**：
  - 從 P00–P06 選哪一個組合。
  - 每個組合的校準參數 a、γ、b（10 維）和 w：原樣沿用 prereg02 的 tune 程序，公式 p = softmax(e^a (n/12)^γ (m + w·m_op + b))。
  - 選定組合的無開局版參數（w＝0）。
- **沿用 prereg02 的設定**：
  - 假題 `pseudo_inner.json`：擬合用第 0–49 輪、只收 ≥20 盤的玩家；評估用第 1000–1199 輪、636 人全收。
  - 每級內依玩家分 5 折。
  - 目標函數：分等級加權 NLL 加 0.01·‖b‖²；L-BFGS-B，界限和兩個起點都和 p02 相同。
- **訓練超參數都在 rev 1 寫死，沒有在 inner 上調**：lr、wd、warmup、T、200 盤上限、各級等量、worker 數。

### 2.3 送分閘門（三項都成立才打 H3R，只看選到的那一個組合）

- **G1**：選定組合減 P00 的 inner 5 折交叉驗證分等級平均 ≥ +0.010（沿用 `GATE`，浮點容許 1e-12）。
- **G2**：同一個差值的 inner 配對 PlayerBootstrap 95% 下界 > 0。每級內重抽玩家 1,000 次，`default_rng(0)`，用 p02 的類別。
- **G3**：inner 評估輪的每題平均差值 ≥ 0。
- **只記錄、不當閘門**，寫進 `frozen_config.json`：
  - SE_val ＝ √(Σ_l Var_l(d) / N_l) / 10。d 是 inner 評估輪上每題的得分差；N_l 是驗證集各級題數：178、163、207、308、87、122、124、128、131、141。
  - 推估的採用機率 Φ(Δ/SE_val − 1.96)。
  - 各級差值；盤數分箱（5–7／8–12／13–20）的預測分布。
- **選到 P00 本身**，或閘門沒過：「未送分」，R20 不變。C6 的逐盤輸出留給 04 當來源。
- **閘門的取捨**：G2、G3 等於把驗證集的採用條件先在 inner 上套一次。P1 以檢定力定門檻（約 +0.015），會在真實增益 +0.010～0.015、採用機率仍有 3–6 成時放棄打分，所以這個門檻只記錄、不採用。

### 2.4 競賽一：03 不登記新方法，不打分，預算維持用了 1 次

- 本機 0.8830 的主要風險在官方怎麼切，這一點本機驗證量不到，只有 11/04 的 T1–T6 能回答。
- test02 的競賽一工具已經實跑過正式模式，重新打開要整套重驗。
- 兩隊不交換模型輸出：p03 的模型和任何輸出都不進競賽一。
- 11/04 如果 T2、T3 或 T4 觸發，照 PREREG_02 已登記的程序處理；之後要改再另寫 04-player。

---

## 3. Colab notebook 規格

### 3.1 檔案

- **產生器**：`colab/build_prereg03_rank_nb.py`，產生 `colab/prereg03_rank_a.ipynb`、`colab/prereg03_rank_b.ipynb`。
  - 底稿是已 commit 的 `colab/prereg02_rank.ipynb`，先 assert SHA，再用 `rep()` 修改，每處替換 assert 剛好命中一次。
  - 兩份只差 `ARM = 'A'／'B'` 一行，builder 會 assert 這一點。使用者不改任何格子。
  - 另外輸出逐格差異 `results/prereg03/design/nb_diff_{a,b}.txt`。
  - notebook 不可手改。
- **模型**：官方 `GoRankResNet(in_channels=501)`，256 通道 × 20 個 block，官方的頭、dropout 0.5。`network.py`、打包檔 `C10_bundle_v1.zip` 都不改。
- **Drive 輸出**：`C10/results/prereg03/rank_{a,b}/`。
  - `guard_write` 擋住寫入 prereg01、prereg02、test02 三個根目錄，開頭和結尾各比對一次指紋。
  - 結果 zip：`prereg03_rank_{a,b}_results.zip`，每份 < 100 MB，不含權重。

### 3.2 格子順序

1. **Preflight**
   - GPU 關卡：只接受 G4（RTX PRO 6000）或 A100（標準版）。型號不符就在計算前停下。
   - 印出 vCPU 數、RAM；Drive 剩餘空間要 ≥ 3 GB。
   - 印出 `device … | GPU … | ARM … | workers …`。
2. **掛 Drive**：`guard_write`、唯讀指紋（前）。
3. **打包檔**：核對打包檔 SHA、解開。
   - `load_rows(part="fit")` 得 5,767 人、720,307 盤；inner 得 640 人、73,556 盤。
   - 做外洩 assert：fit、inner、驗證玩家三者不相交。
4. **特徵**：`FEAT_FINGERPRINT` 要和 prereg01 相同。
   - 有 prereg01 的 Drive 備份時還原（實測 80 秒）。
   - 沒有時重抽（實測 355 秒）；只有 A 寫備份，B 抽到本機磁碟。
5. **執色表**：鍵是 `(rank, game_id)`，對回 `df_train`／`INNER_KEYS` 的 `color` 欄。**不從 game_id 推**（`DATA.md` 第 3 條）。
   - 每個 fit 和 inner 分片 key 都要查得到（涵蓋率 100%）。
   - fit 執黑筆數等於 `df_train` 算出的值，執黑比例在 0.50 ± 0.01。
   - 用人工棋盤做視角轉換的單元檢查：執黑樣本不變；執白樣本每個盤面兩層對調；第 500 層是常數。
   - 對 8 種對稱，assert D4 `gather` 的結果等於 `d4(x, s)`。
6. **只有 B**：用 `df_train` 重算加上 200 盤上限後各級的總盤數 E_L（第 3.4 節），必須和登記值完全相同。
7. **Loader、模型、optimizer、scheduler 建立**：有 `state.pt` 就接續（第 3.6 節）。
8. **訓練迴圈**：
   - 以 global step 計，到 T 步就停，不管是不是一輪的中間。
   - 每一輪結束：inner 監看（原方向、fp32，約 13 秒）、存 `state.pt`、寫 `history.csv`。
   - 第 300 步：計時並推估總時間（第 3.5 節）。
9. **最後輸出**：全部用 fp32，關 autocast；先做視角轉換，再做 D4。
   - `final.pth` 的 SHA-256。
   - inner 的 8 種對稱輸出，和原方向、執色對調的輸出。
   - `parse_val` → `val_keys`；驗證題的 8 種對稱輸出（和 prereg02 一樣每題一個 batch），和原方向、執色對調的輸出。
   - `_DONE_*.json`。
10. **收尾**：核對列數和完成標記、唯讀指紋（後）；打包 zip；印出「Drive 已寫入完成並卸載」。

### 3.3 兩臂共同的配方（寫死）

| 項目 | 設定 |
|---|---|
| 輸入 | 501 層。第 2t 層是目標玩家的子，第 2t+1 層是對手的子（最後 250 個盤面），第 500 層＝1 代表目標玩家執黑。在 GPU 上從 uint8 轉：`x=x.view(B,250,2,19,19); x=torch.where(isB[:,None,None,None,None], x, x.flip(2))`，再接上執色層 |
| D4（只在訓練時） | 每筆均勻抽 s∈0..7，用 [8,361] 排列表 `gather`。GPU generator 的種子是 42＋pass。執色層在 D4 下不變 |
| 最佳化 | AdamW，lr 1e-3，wd 0.01，batch 256，CE，不做 label smoothing |
| 學習率 | 前 1,000 步線性 warmup 到 1e-3，之後 cosine 降到 0。**T＝33,768 步（＝12 × 2,814），和 worker 數、GPU 型號無關**，取代 prereg02 寫死 2,814 的 assert |
| 精度 | 訓練用 bf16 autocast 加 channels_last，權重 fp32，不用 GradScaler。監看和所有逐盤輸出都用 fp32，TF32 設定和 prereg02 相同。CPU 冒煙測試時關 autocast |
| 初始化 | SEED＝42，兩臂相同（random、numpy、torch） |
| Loader | webdataset 1.0.2。worker 傳 uint8、到 GPU 才轉型；每一輪重建 loader。分片順序用 `random.Random(42·1000+pass)` 洗；worker 內的 shuffle buffer 1,000，rng 種子由 (42, pass, worker) 決定；批次在 worker 內組成，最後一批不滿也算一步（同官方）。這解決 PREREG_02 rev 4 第 7 條的問題；cuDNN 和 bf16 仍不保證逐位元組相同 |
| worker 數 | G4：A 16 個、B 32 個。A100：兩臂各 10 個。寫進 recipe 設定 |
| 指定 checkpoint | 第 T 步的 `final.pth`，不挑 best。inner 監看只記錄 |

### 3.4 A、B 只差取樣方式

- **A**：每一輪用全部 fit 棋，約 2,814 步一輪，共約 12 輪。
- **B**：每位玩家最多算 200 盤，而且各級等量。
  - 收下的機率：q_g ＝ min(1, 200/n_p) × 35,845 / E_L。n_p 是這位玩家在 fit 的盤數。
  - E_L（該級加上 200 盤上限後的總盤數）：D 49,949、C 49,717、B 56,575、A 64,188、1D 35,845、2D 45,137、3D 45,024、4D 43,441、5D 46,263、6D 47,407。
  - 判定：u ＝ int(sha256(`c10-prereg03|acc|{pass}|{rank}|{game_id}`)[:16], 16) / 2^64，u < q_g 就收下。先看 key 決定收不收，不收的不解碼。
  - 預期每一輪 358,450 盤，每級 35,845 盤；T 步約是 24.1 輪。有效玩家數從 1,907 升到 3,184（`cap_stats2.json`）。
  - 每一輪實際收下的筆數寫進 `history.csv`。
- 兩臂的 T 相同，算力也相同，差別只來自取樣。200 盤和各級等量是看過 fit 的中繼資料後寫死的，沒有調過。

### 3.5 停止條件（依序；任一不過就印出原因，請使用者截圖給 Claude）

1. GPU 不是 G4 或 A100 → 計算前停。
2. 打包檔 SHA、特徵指紋、外洩檢查、Drive 空間、第 3.2 節第 5 格的執色和 D4 檢查、B 的 E_L → 任一不符就停，不訓練。
3. 第 300 步推估的訓練時間 > 150 分鐘（G4）或 > 180 分鐘（A100）→ 停。那一臂不算結果，由 Claude 決定、先寫修訂再處理。
4. loss 出現非有限值 → 停，狀態記 `nan_stop`。唯一的備案是先寫修訂，再用 fp32（T 不變）整份從頭重跑；不接續，也不混用精度。
5. 接續時 recipe 設定不同（第 3.6 節）→ 停。
6. 收尾：輸出列數、完成標記、唯讀指紋前後相同。不符就打包成 `*_failed.zip`，那一臂不用。

### 3.6 接續

- 每一輪結束時存 `state.pt`（先寫暫存檔再改名）。內容：model、optimizer、scheduler、global_step、pass、history、各 RNG 狀態，以及 recipe 設定（ARM、T、WARMUP、LR、WD、AMP、增強、CAP、E_L、worker 數、GPU 型號）。
- 接續時從下一輪的開頭重跑，資料順序和沒中斷時相同。最多損失一輪：A 約 4–7 分鐘，B 約 2–3.5 分鐘。
- `.pth` 只留最後兩輪的，加上 `final.pth`，每個約 105 MB。沒跑完 T 步的權重一律不用。

### 3.7 輸出（每臂一個 zip）

| 檔案 | 內容 | 要和什麼一致 |
|---|---|---|
| `dumps/inner_keys.npz` | rank、game_id、player_id、color、label | 和 `results/prereg02/rank/dumps/inner_keys.npz` 逐陣列相同 |
| `dumps/inner_p03{a,b}_d4.npz` | fp32 logits [73,556, 8, 10]、`weights_sha256`、`syms` | `p02_rank_fuse.load_dump` 的格式 |
| `dumps/inner_p03{a,b}_s0_swap.npz` | 原方向、執色對調的 logits [73,556, 10] | 只給第 4.4 節的接線檢查用 |
| `dumps/val_keys.npz` | question_id、slot、color、parsed | 和 prereg02 逐陣列相同 |
| `dumps/val_p03{a,b}_d4.npz` | [19,156, 8, 10]，解析失敗的整列 NaN | `check_val_logits` |
| `dumps/val_p03{a,b}_s0_swap.npz` | [19,156, 10] | 只給第 4.4 節的 κ 檢查用 |
| `_DONE_*.json`、`config.json`、`env.json`、`timings.json`（含 bf16 實測每步 ms）、`run_log.json`、`leak_check.json`、`ckpt/history.csv`、唯讀指紋前後 | GPU、torch、cuDNN、權重 SHA、秒數、每一輪的監看 | `write_dump` 的格式 |

### 3.8 帳號、時間、運算單元（推估；G4 每小時 8.71 單元，bf16 沒實測）

| | A | B |
|---|---|---|
| 準備 | 約 3 分（Pro+ 重抽再加約 6 分） | 同左 |
| 訓練 33,768 步 | 43–87 分（fp32 實測每步 151–154 ms；bf16 快 1.5–2 倍） | 同左，loader 最多再慢約 20% |
| 監看＋最後輸出 | 約 5–6 分 | 約 7–8 分（監看 24 次） |
| **合計** | **52–96 分，7.5–13.9 單元** | **60–115 分，8.7–16.7 單元** |

**帳號規則**：運作上的選擇，不影響判定；在任何 03 輸出產生前記進 `run_log`。
- 優先用 AI Pro：有 prereg01 的特徵備份，權重和 test02 在同一個 Drive。
- 開跑帳號的可用量：
  - **≥ 50**：A、B 同時跑。
  - **25–49**：只跑 A，用到 C6B 的組合從清單拿掉，不算偏離。
  - **< 25**：不在這個帳號跑。
- AI Pro < 50 而 Pro+ ≥ 50：改在 Pro+ 跑兩臂。使用者先上傳 `C10_bundle_v1.zip`（247 MB）一次；採用的話，test03 也在 Pro+ 跑。
- 兩個帳號都 < 25：等額度，到 C1 截止點（10/16）還不夠就關閉 03。
- B 要等 A 印出 `device` 那一行之後才開，避免 Drive 建出兩個同名資料夾。
- AI Pro 沒有背景執行：跑完前分頁要一直開著，電腦不能睡眠。

---

## 4. 本機工具：`tools/p03_rank_fuse.py`（只用 CPU、`.venv`）

### 4.1 沿用和凍結

- **import p02 的函式**：`basis`、`log_softmax`、`softmax`、`rule_b_np`、`pool_matrix`、`pooled`、`QSet`、`question_logits`、`fit`、`predict`、`metrics`、`PlayerBootstrap`、`load_dump`、`load_val_keys`、`check_val_logits`、`build_qset`、`read_question_meta`。
- **沿用 p02 的常數**：`GATE`、`SELECT_TOL`、`PENALTY`、`BOUNDS`、`STARTS`、`N_BOOT`、`BOOT_SEED`、`P02E9_MIN_ACC`。
- **一律不改**：`tools/p02_*.py`、`c10data.py`、`decide.py`、`score.py`、`colab/*prereg0{1,2}*`、`colab/test02_*`、`tools/t02_*.py`、`results/prereg0{1,2}/**`、`results/test02/**`。
  - 工具開頭核對 `p02_rank_fuse.py`、`p02_rank_pseudo.py`、`p02_rank_opening.py`、`p02_moves.py`、`c10data.py`、`decide.py` 的 SHA-256，和 HEAD 的 blob 相同才執行。
- **沿用已 commit 的檔**：`pseudo_inner.json`、`opening_inner.npz`、`opening_val.npz`（M＝12、α＝20，只用 fit 計數），不重算。
- p03 的工具和 notebook 都不含 `answers` 字串；`tune` 會 assert 所有 player_id 都屬於 inner。

### 4.2 來源

| 代號 | 定義（格式同 p02 的 `SOURCES`） |
|---|---|
| C5 | `[("p02e9_d4","all")]`，已 commit |
| C6A | `[("p03a_d4","all")]` |
| C6B | `[("p03b_d4","all")]` |

- 集成：各來源的方向清單串起來，照 p02 的 `log_softmax(mean(parts))` 算。每個來源都是 8 個方向，所以各來源等權，不加新參數。
- n 是這一題可解析的盤數。驗證題上各來源的解析集合必須相同，不同就停下查 bug。
- 盤數只進入 10 個等級共用的溫度。

### 4.3 組合（順序固定，也是平手時的優先順序；全部用 log 合併、校準、開局項、規則 B）

| 組合 | 來源 |
|---|---|
| P00 | C5（＝R20，自我檢查和閘門的基準） |
| P01 | C5 ⊕ C6A |
| P02 | C5 ⊕ C6B |
| P03 | C5 ⊕ C6A ⊕ C6B |
| P04 | C6A |
| P05 | C6B |
| P06 | C6A ⊕ C6B |

- **選法**：5 折交叉驗證的分等級平均最高者；和最高分差距在 0.003 以內的，取清單中最前面的。帶 C5 的集成排在前面，萬一 D4 增強抹掉了真實的方向習慣，還能補回。
- 不能用的來源，連同用到它的組合，從清單拿掉。

### 4.4 `check`：新來源能不能用（在 `tune` 之前）

1. **結構**：
   - key 檔和 prereg02 逐陣列相同。
   - 權重 SHA 和 `_DONE` 一致；`history.csv` 記錄跑完 T 步。
   - 驗證題輸出只查形狀、NaN 樣式和雜湊，NaN 樣式要和 `val_p02e9_d4` 相同。
2. **地板**：`final.pth` 在 inner 原方向的逐盤 accuracy < 24.15% → 這個來源不能用。
3. **執色接線（inner）**：執色對調後的 accuracy 減正確執色的 accuracy > 0.2 個百分點 → 判定為 bug，停下查；要重新輸出或重訓，先寫修訂。正確減對調的差值 Δacc 照實記錄。
4. **執色語意 κ 檢查**（新增，不讀答案）：
   - 定義：每一題中，每一盤的原方向 log 機率 ℓ_g，和「其他盤合併後的分布」softmax(mean_{h≠g} ℓ_h) 的期望對數似然；κ 是題內平均、再取各題平均。
   - Δκ ＝ κ(正確執色) − κ(整題執色對調)。
   - 在 inner 評估輪上算 Δκ_inner 和它的 PlayerBootstrap 95% CI。
   - **Δacc ≥ 0.5 個百分點，而且 Δκ_inner 的下界 > 0**（也就是模型確實在用執色）時，要求驗證題的 Δκ_val > 0，否則停下查 bug。
   - 條件不成立時只記錄：模型幾乎不用執色，接錯的代價也很小。
   - 這一步只輸出「通過／停止」，不影響組合的選擇。
5. **只記錄**：每一輪的學習曲線；8 種對稱逐盤 accuracy 的分散程度（對照 C5 的 21.4–25.6%）；訓練和 inner 的差距；B 臂每一輪收下的筆數。
6. **輸出**：`results/prereg03/rank/check_report.json`，列出能用的來源。

### 4.5 指令與順序（程式強制；每一步 commit 之後才算完成）

1. `.venv\Scripts\python.exe tools/p03_rank_fuse.py unpack --arm a "<zip 路徑>"`（B 同理）
   - 解開到 `results/prereg03/rank_{a,b}/`，核對檔案清單、不含權重。
   - commit。
2. `… p03_rank_fuse.py check` → commit。
3. `… p03_rank_fuse.py selfcheck` → `selfcheck.json`，commit。
   - P00 的 inner 5 折交叉驗證是 0.5854702521，和 `tune_report.json` 的 R20 相差 ≤ 1e-12，每一列都相同。
   - 全部擬合資料重擬合的參數，和 `results/prereg02/rank/frozen_config.json` 相差 ≤ 1e-12。
   - P00 套到驗證題（只用 prereg02 已 commit 的 C5 輸出），和 `results/prereg02/rank/rank_val_probs.csv` 最大差 ≤ 1e-12，規則 B 完全相同。
4. `… p03_rank_fuse.py tune` → `tune_report.json`、`frozen_config.json`，commit。
   - 要求 Colab 結果和工具都已 commit；不讀任何 `val_*`。
5. `… p03_rank_fuse.py apply` → `rank_val_probs.csv`、`apply_report.json`（記閘門結果和 `score_allowed`）。
   - 要求 `frozen_config.json` 已 commit、和 HEAD 相同，而且工具沒改過。
6. `.venv\Scripts\python.exe tools/decide.py results/prereg03/rank/rank_val_probs.csv results/prereg03/rank`
7. `.venv\Scripts\python.exe tools/check_submission.py rank results/prereg03/rank/pred_B.csv --questions data/local/rank_val_questions.csv` → commit 第 5–7 步的產物。
8. 只有 `score_allowed` 才跑：
   ```
   .venv\Scripts\python.exe tools/score.py compare-rank results/prereg02/rank/pred_B.csv results/prereg03/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg03 H3R (A = prereg02 R20, reference re-read)" --json results/prereg03/rank/score_H3R.json
   ```
9. 帳本、`score_H3R.json`、`RESULTS.md`（由另一個代理獨立核對數字）、PORTFOLIO_MAP 的 C10 那一行，一起 commit。

### 4.6 `frozen_config.json` 的內容（format 字串 `c10-prereg03 rank frozen_config v1`）

- 選定的組合、各來源和 dump 的 SHA-256、參數（a、γ、b、w）、basis、開局參數。
- 閘門數值：Δ、G2 的 CI、G3 的差值、SE_val、推估的採用機率。
- 各級 OOF、盤數分箱的預測分布。
- **無開局備案**：選定組合拿掉開局項的參數（w＝0，同一套 tune 程序），加上它的 inner 交叉驗證分數。
  - **FAIL_OPENING 時用哪一個**：取這個備案和 R10 之中 inner 交叉驗證較高者（R10 是 0.5687，來自 `results/test02/rank/fallback_R10.json`）。在這裡先決定好，測試日不再判斷。
- 工具的 commit、git HEAD，Python、numpy、scipy 的版本。

### 4.7 測試

- `tests/test_p03_rank_fuse.py` 用合成的逐盤輸出測：
  - 來源的串接和等權；
  - 順序強制；
  - 閘門；
  - κ 和 Δacc 的計算；
  - 工具 SHA 核對會擋下被改過的 p02。
- 真實資料的 P00 重現，由 `selfcheck` 負責。

---

## 5. 測試日工具擴充（只在 H3R 採用後才做）與截止點

### 5.1 設計原則

- test02 的 notebook 和 `t02_*.py` 不動，而且先跑。先照 `TESTDAY.md` 上傳 R20（或 R10）和 Q04。03 的組合只是之後的另一次上傳。
- test03 只推論 C6；C5 的輸出直接讀 test02 已 commit 的結果。

### 5.2 要新增或修改的

1. **`colab/build_test03_rank_nb.py` → `colab/test03_rank.ipynb`**
   - 結構照 `test02_rank`：三組題目 `seam_val`、`public`、`private`，MANIFEST，完成狀態。
   - 模型和視角轉換的程式碼逐字取自 prereg03 notebook，builder 會 assert 文字相同。
   - 只載入選定組合用到的 `final.pth`，SHA 釘死。
   - 輸出 8 種對稱 fp32 的逐盤結果，和原方向、執色對調的結果。
   - MANIFEST 的 SHA 要和 test02 那次相同（同一份 `C10_test_v1.zip`）。
   - 接縫：原方向每題平均 softmax 和 prereg03 的驗證題輸出比，同型 GPU ≤ 1e-4、不同型 ≤ 1e-3；規則 B 不同 ≤ 4／1,589 題。
   - GPU 用 G4 或 A100；在放權重的那個帳號跑。
2. **`tools/t03_rank.py`**
   - `apply` 的前提：
     - test02 rank 的狀態是 completed、接縫過、已 commit；test03 也是 completed、接縫過。
     - 兩邊的 `manifest_sha256` 相同，每組題目的 key 陣列相同。
     - 開局沿用 `results/test02/rank/opening_test.npz`，核對 SHA。
     - 每個來源各自的三組輸出要來自同一個執行環境；不同來源之間可以不同。
   - 本機接縫：`seam_val` 上的選定組合，要重現 `results/prereg03/rank/rank_val_probs.csv`（同一個容許差），規則 B 不同 ≤ 4 題。
   - **測試集的 κ 檢查**：第 4.4 節第 4 條的條件成立時，public、private 各自的 Δκ 都要 > 0；不成立就判定官方 `color_i` 的語意和訓練不同，03 不用，維持 R20，並記錄。
   - 輸出 `results/test03/rank/submission_rank.csv`，並跑 `check_submission --official`。
   - `fallback-fit`：自我核對選定組合重擬合的參數 ≤ 1e-12，把無開局版參數寫成 `results/test03/rank/fallback_noopen.json`，在 11/04 前 commit。
3. **`tools/official_log.py`**：test02 這一側唯一要改的檔。
   - 加方法標籤 `prereg03 P0x rule B (甲)` 和 `prereg03 P0x no-opening rule B (甲)`。
   - `check_reports` 要接受 `results/test03/rank/`。
   - 改完重跑整套 `tests/test_t02_*.py`、`test_p02_*.py`，以及慢速的 `test_t02_dryrun.py`，證明 test02 不受影響。
4. **演練與文件**：用 `t02_fake_tests.py` 做 test02 加 test03 的 dry run；更新 `docs/TESTDAY.md`、`docs/COLAB_RUN.md`；做獨立審查。
5. **PREREG_03 rev N**：寫明測試日的程序（本節），在 11/04 11:00 前 commit。

### 5.3 11/04 的順序

1. 照 test02 做，先上傳 R20／Q04。
2. 跑 `test03_rank.ipynb`，約 10–20 分鐘。
3. Claude 跑 `t03_rank.py apply`，核對後把 03 的組合當成競賽二的下一次上傳。上傳指示自成一則訊息，使用者在對話裡說 OK 才傳。
- test03 接縫沒過：換一個執行階段重跑一次；還是沒過，維持 R20 並記錄。
- T1 是 FAIL_OPENING：用 `frozen_config` 已決定的那個（03 無開局版或 R10）。
- T1 是 FAIL_STRUCT：照現有規則停下。

### 5.4 最後一版（11/23 前上傳）

同時符合以下四項才交 03 的組合，否則交 R20（FAIL_OPENING 時照 test02 規則交 R10）：
1. H3R 採用。
2. C4 之前有一次成功的正式執行：test03 接縫、本機接縫、κ、`check_submission --official` 都過，而且已 commit。
3. T1 不是 FAIL_STRUCT。
4. 測試集的 κ 檢查通過。

public 分數比上一版低 0.05 以上，只追查錯誤，不因此換方法（VALIDATION 紀律 4）。

### 5.5 截止點（寫死；任一沒趕上，這一季就不用 03，也不拿額外的打分機會去補）

| 截止點 | 條件 |
|---|---|
| **C1 10/16（五）23:59** | 至少一臂能用的 Colab 結果已 commit |
| **C2 10/19（一）23:59** | 閘門已判定；過了的話，H3R 已打分 |
| **C3 11/03（二）23:59**（目標 10/27） | 第 5.2 節全部 commit，演練也過 |
| **C4 11/16（一）23:59** | 03 在官方測試集至少有一次成功的正式執行 |

---

## 6. 時程

| 日期 | 事項 | 誰 |
|---|---|---|
| 9/27（日） | PREREG_03 rev 1 commit；`results/prereg03/design/` 收進三份調查報告、診斷腳本和 JSON（`colour_diag`、`length_diag`、`r20_diag`、`far_errors`、`feat_signal`、`cap_stats2`、`power`），npz 不進 git | Claude |
| 9/28（一）–10/02（五） | builder 和兩份 notebook、逐格差異、`p03_rank_fuse.py`、`tests/test_p03_*.py` | Claude |
| 9/28（一）–10/02（五） | CPU 冒煙：先重建 prereg01／02 的冒煙樹，約 15 分；用極小的假分片、關 autocast，兩臂都跑；中斷接續後資料順序相同；輸出能被 `load_dump` 讀 | Claude |
| 10/02–10/03（六） | 獨立審查與修正；rev 2 記實作偏離；`COLAB_RUN.md` 加「事前登記 03」一節；更新上傳副本 | Claude |
| 10/03（六） | 回報 AI Pro 和 Pro+ 的「可用量」，各截一張圖 | 使用者 |
| 10/04（日） | 兩個分頁同時跑 A、B（第 3.8 節），約 1–2 小時；下載兩個 zip，把路徑告訴 Claude | 使用者 |
| 10/05–10/11 | 保留的重跑窗口。出錯時照「Colab 出錯時」流程：改 builder、冒煙、審查、記修訂 | 兩者 |
| 收到後 1–3 天（目標 10/07） | 第 4.5 節第 1–9 步 | Claude |
| 10/12（一）–10/27（二） | 採用的話：test03 工具、rev N、演練 | Claude |
| 10/14（三） | 04（逐手、依等級條件化的落子模型）要不要登記：看 03 的結論，由使用者決定 | 使用者 |
| 10/16（五）、10/19（一） | C1、C2 截止點 | — |
| 10/28（三）–11/02（一） | 選做的 Colab 彩排：test02 加 test03，用假測試包，約 1.5 小時、約 7 單元 | 使用者 |
| 11/01（日） | 使用者決定 11 月投入多少，這時 03 的結果已經知道 | 使用者 |
| 11/03（二） | C3 截止點 | — |
| 11/04（三） | 照 `TESTDAY.md`（採用的話已含 test03） | 兩者 |
| 11/16（一）、11/23（一） | C4 截止點；最後一次上傳 | — |

**中斷時怎麼接**：commit 過的步驟才算完成。Colab 靠 `state.pt` 接續。任一臂在開跑後 7 天內沒跑完，就拿掉那個來源；兩臂都沒跑完，記為「未送分」。每一次 commit 後更新 PORTFOLIO_MAP 第 0 節的 C10 那一行。

---

## 7. 刻意不放進 03 的

| 做法 | 理由 | 去向 |
|---|---|---|
| 逐手、依等級條件化的落子模型 | 上限最高，但工程是新的，在我們的規模沒有前例，約 18–22 單元，還要新的逐手推論和接縫，會拖到 11/04 的準備 | 04 候選，10/14 決定，要能在 11/03 前完成；可以把 C6 當來源 |
| 換 seed 的複本當第二臂 | B 臂（盤數上限、各級等量）對準「有效玩家只有 1,907 人」，而且評分是每位玩家同權，多樣性比換 seed 大 | 不做 |
| label smoothing、SORD | 瓶頸是分不開相鄰兩級，把目標往鄰級抹平只會更分不開；資料夠多時 CE 就夠了 | 不做 |
| 加寬、加深、GAP 頭、SE block | 已經過擬合，沒有本題的證據 | 不做 |
| 加長盤面歷史、氣或提子層 | 長棋沒有比較差（≥300 手的 accuracy .276）；AlphaGo 的表顯示加氣沒有增益；要重抽特徵 | 不做 |
| 訓練出來的整題聚合器 | fit 的逐盤輸出是訓練樣本內的；inner 只有 636 人，10×10 矩陣校準都過擬合；盤數無限多時的上限只有 0.654–0.663 | 不做 |
| 校準細化、SGF 特徵 | inner 上都 ≤ +0.002 | 不做 |
| 用全部 8,007 人重訓（乙） | 本機驗證不了 | 看 11/04 的 T3 |
| 競賽一的任何變更 | 第 2.4 節 | 11/04 後再看 |
| 外部資料、預訓練模型、KataGo | 登記禁止；出題者寫所有人都從零訓練 | 永不做 |

---

## 8. 風險與防護

| 風險 | 防護 |
|---|---|
| 執色接反或對錯盤 | 只用 `color` 欄；涵蓋率 100%、執黑比例 0.50±0.01；人工棋盤單元檢查；key 陣列和 prereg02 相同；對調檢查在差距 > 0.2 個百分點時停（不用「嚴格大於」，因為對手多半同級，會誤殺正常的模型） |
| 驗證題或官方測試的 `color_i` 語意和訓練不同（README 沒寫清楚） | κ 檢查：在 inner 確認模型確實在用執色之後，驗證題和 public、private 的 Δκ 都要 > 0，否則停下或退回 R20 |
| D4 增強抹掉真實的方向習慣 | 帶 C5 的集成排在清單前面 |
| B 臂變差（1D 在 B 臂每盤約看 24 次，A 臂 12 次；上限和等量兩件事分不開） | inner 選組合會排除；揭露。這是取樣方式整包的檢驗，不需要拆開歸因 |
| bf16 沒加速或不穩定 | 第 300 步推估，超過 150／180 分鐘就停；非有限值就停，fp32 備案要先寫修訂、整份重跑；逐盤輸出一律 fp32 |
| 拿不到 G4 | 接受 A100（worker 10 個，T 不變）；其他型號在關卡擋下；實際型號記下來 |
| B 臂 32 個 worker 的記憶體或 loader 太慢 | 冒煙時量每個 worker 的記憶體；第 300 步的時間上限；B 失敗只拿掉 C6B |
| 斷線 | 每一輪存 state，最多損失一輪；出現紅字後使用者不自己重跑，截圖給 Claude |
| 運算單元不夠或不明 | 開跑前看兩個帳號的可用量，照 50／25 分級；Pro+ 當備案 |
| inner 略為樂觀、從 7 個組合裡選 | 平手取前面的；G1、G2、G3 三道閘門；只打 1 次分；揭露（開局的 M、α 在 inner 定；在 inner 上選組合） |
| 驗證集的檢定力不足 | SE_val 和推估的採用機率寫進 `frozen_config`；CI 含 0 就不採用，R20 照舊 |
| 盤數變成等級特徵 | 校準公式不變；擬合只用 ≥20 盤的玩家；報告各分箱的預測分布；獨立審查專門檢查 |
| 資料外洩 | 只經 `load_rows(fit)` 訓練；`tune` 不讀 `val_*`、assert inner 成員；答案不上 Colab；打分記帳 |
| 寫壞 prereg01／02 或 test02 | `guard_write`、前後指紋比對；p02 和 t02 的工具在執行時做 SHA 核對；`official_log.py` 改完跑整套測試和 dry run |
| 測試日變複雜或趕不上 | test02 先跑、R20 先上傳；C1–C4 截止點；test03 失敗就退回 R20；彩排選做 |
| 結果不能逐位元組重現（cuDNN、bf16） | 報告寫明；測試日靠 SHA 釘住權重，加上接縫容許差（放寬要另寫修訂） |
| 使用者 10 月很忙（C4 在 10/16 前要口述） | 只需要一次 Colab（10/04）；彩排選做；不看結果的工作都由 Claude 做 |

**揭露（寫進 RESULTS 和報告）**：
- 配方的選擇參考了只用 inner 的診斷和文獻，沒有在 inner 上調任何訓練超參數。
- B 臂的 200 盤上限、各級等量、T、warmup，都是看過 fit 的中繼資料後寫死的。
- 開局的 M、α 和組合都是在 inner 上選的。
- 無開局備案沒有在驗證集打分。
- 訓練不能逐位元組重現。
- 生成式 AI 協助的部分照規則揭露。

---

依據檔都在 scratchpad `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03\`：`taskA\`、`B_error\`、`priorart\`、`prereg_draft\cap_stats2.json`、`maxgain\power.json`、`judge\`。rev 1 commit 前要搬進 `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg03\design\`。