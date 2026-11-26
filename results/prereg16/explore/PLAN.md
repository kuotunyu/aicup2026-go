# prereg16 探索：一致性／尾端特徵（競賽二，inner-only）

2026-10-16 寫成，**在算出任何候選數字之前 commit**（分支 `p16-explore`）。只用 inner 訓練玩家已存在的產物；不碰 `data/local`、不讀任何驗證題特徵、不跑 `tools/score.py`、不用 GPU、不跑新的 KataGo。本檔的閘門在正式執行前寫死，正式執行後只判一次（`eval_report.json` 原樣 commit）。

## 0. 白話版

- 現在競賽二的首選 Q09：每一盤先算「平均」（HumanSL 14 個棋力設定的平均似然、KataGo 探針的平均內部特徵），再把一題 5～20 盤的結果平均，最後融合成 10 級的機率。
- 這次問：**除了平均，「穩不穩定」和「最差的幾手有多差」能不能多分辨出等級？** 靈感是 Regan 的 Intrinsic Performance Ratings：棋力＝平常下得多準，加上犯大錯的傾向。具體是每一題多算三類統計：
  - 一盤之內：逐手指標的標準差、10／90 分位、最差 5 手的平均、開局／中盤／終盤分開的平均、大失誤的比例；
  - 整題所有手合起來：最差 10 手的平均、整體標準差；
  - 盤與盤之間：每盤平均的標準差、每盤預測等級的標準差和全距。
- 用一個小的多類 logistic regression（只看這些特徵）翻成 10 級的對數機率，當作 Q09 融合的一個新項（權重由融合自己擬合），用玩家分折的**巢狀交叉擬合**，評估的題從來沒有參與擬合。
- 先講預期：類似的東西試過兩次都沒用（10/5 `hs_rich_fitmap_cv.py`：每盤加 SD／分位數／分階段，只用 3,000 盤 fitmap 學，都沒贏過平均；09 的 `lowrank3_ex` 已吃了 P1 逐手 log p／熵的分位數和分階段平均，單獨只有 +0.0076）。這次新的是「題目層級」（跨盤離散度、整題最差手）和改用 inner 的 25,400 道擬合假題來學。過關的機會不高；閘門照舊 +0.010。
- 競賽二已經沒有驗證集打分機會：就算過關，採用也只能像 09 一樣靠 inner，而且要另寫 PREREG_16 和測試日路徑（本探索**不做**，由主 session 決定）。

## 1. 資料盤點（10/16 讀過，CPU 就夠，不需要 GPU）

| 來源 | 位置 | 內容 | inner 覆蓋 |
|---|---|---|---|
| HS 逐手 | `data/prereg07/pilot/{bank,mono}/chunk_*.npz`（60＋307 塊） | `pos_game`、`pos_turn`、`pos_logp [手, 14]`（目標方每一手在 14 個棋力設定下的 log p），另有每盤 `sums`、`counts`、`failed` | bank 11,813＋mono 61,356 盤；SHA 對 `results/prereg07/hs/inner_manifest.json` |
| P1 逐手 | `data/prereg08/p1/{bank,mono}/chunk_*.npz` | `pos_game`、`pos_side`、`pos_turn`、`pos_logp`（中性 metadata 下實際落點的 log p）、`pos_entropy`；每盤 `means [2, 770]` | 同上；SHA 對 `results/prereg08/feat/inner_manifest.json` |
| 每盤 lp_HS | `Inner.d["lp"]["HS"]`（`p1_fuse.hs_baseline`，fitmap 凍結對照模型） | 每盤 10 維 | 不用 inner 標籤 |
| 每盤 lp_P1E | `data/prereg12/q09_inner/`＋09 凍結 `p1e_map.npz`（`p09_rank_fuse.inner_map_lp`） | 每盤 10 維（Q09 的 P1TO_E，乘 4 之前） | 73,169 盤；不用 inner 標籤 |

- 10/16 只做了結構檢查（沒有算任何候選數字）：HS 塊的 `pos_game` 是塊內盤號、依序排列，`bincount` 等於 `counts[:,0]`，逐手 log p 的和等於 `sums[:,0]`；`pos_turn` 是整盤的手數（從 0 起）；HS 只有目標方，P1 兩方都有（`pos_side` 0＝目標方）；兩邊的 `ref` 都是 inner 列號。
- inner 題數：擬合假題 25,400（混色，5 折玩家折 5,250／5,050／5,100／4,950／5,050）、評估假題 127,200、單色假題 120,544；inner 盤 73,556（其中 P1 可用 73,169）。

## 2. 特徵（寫死，共 73 維）

### 2.1 逐手的 5 條序列（只用目標方的手）

令 ℓ_t ∈ R^14 是 HS 第 t 手在 14 個設定（20k…9d，r＝0…13）的 log p；P1 的 log p、熵記 lp_t、H_t（`pos_side`＝0）。

1. `h_rank`＝Σ_r r·softmax(ℓ_t)_r（均勻先驗下這一手「像幾段」，0～13）；
2. `h_slope`＝ℓ_t[13] − ℓ_t[0]（9d 和 20k 的似然差，正＝像強者的手）；
3. `h_acc`＝ℓ_t[13]（9d 設定下實際落點的 log p；越低＝越像大錯）；
4. `p_logp`＝lp_t；
5. `p_surp`＝−lp_t − H_t（比局面平均更意外多少；越高＝越像大錯）。

「差的方向」：`p_surp` 是越大越差，其他四條越小越差。

### 2.2 每盤的統計（每一盤的目標方手）

- 每條序列：平均、標準差（ddof＝0）、10／90 分位（`np.percentile` 預設內插）、最差 5 手的平均（不足 5 手就全部）、開局（`pos_turn` < 60）／中盤（60–149）／終盤（≥ 150）三段的平均（該段沒有手就是 NaN）。
- 大失誤比例：`h_acc` < log 0.01 的比例、`p_logp` < log 0.01 的比例。
- 每盤預測等級：e_HS＝Σ_k k·softmax(lp_HS)_k、e_P1E＝Σ_k k·softmax(lp_P1E)_k（k＝0…9）。
- 另存每盤的最差 10 手（每條序列）、Σx、Σx²、手數，給 2.3 B 用。
- **可用盤**＝P1 可用（`Inner.covered`：沒失敗、有目標手）而且 HS 可用（`pilot_hs_usable`）。HS 和 P1 的目標手數不同的盤記數（只記錄；手數用各自的）。

### 2.3 題目層級的 73 維（只用該題的可用盤）

- **A 盤內（37）**：上面每盤的 5×(標準差、q10、q90、最差 5 手、三段平均)＝35 維＋2 個大失誤比例，在該題的可用盤上取平均（三段平均用 nanmean，全部 NaN 就是 NaN）。
- **B 整題（10）**：每條序列，把該題所有手合起來的最差 10 手平均（由每盤的最差 10 手合併後取最差 10，等於直接合併全部手）；整題所有手的標準差（由 Σx、Σx²、手數算）。
- **C 盤間（11）**：5 條序列每盤平均的標準差（ddof＝0）；e_HS、e_P1E 的標準差；e_HS、e_P1E 的全距（max − min）；e_P1E 的最小值、最大值。只有 1 盤可用時標準差、全距是 0。
- **D 錨點（4）**：e_HS、e_P1E 的平均；log(可用盤數)；可用盤目標手數的 log 平均。
- **E 交互（11）**：C 的 11 維各自乘上 D 的「e_P1E 平均」（兩者都先標準化）。理由：同樣的不穩定程度，對強者和弱者的意義不同；線性模型要靠交互項才看得到。
- 沒有可用盤的題：全部記 NaN（標準化後設成 0），數目記錄。
- **標準化**：每次擬合只用該次的訓練列算 nanmean／nanstd（標準差非有限或 0 時用 1），套到訓練列和預測列，之後 NaN 設 0；E 是標準化後的 C 乘標準化後的錨點，不再標準化。

## 3. 模型和巢狀交叉擬合

### 3.1 模型 M

- 多類 logistic regression：lp_M＝log_softmax(z·W＋b)，W 是 73×10、b 是 10。
- 損失：等級平衡的加權 NLL（權重 `rf.level_weights(y)`，和融合相同），除以權重總和，加 λ‖W‖²（b 不罰）。L-BFGS-B（`scipy.optimize.minimize`，解析梯度），起點全 0，maxiter 1,000。
- λ 從 {1e-4, 1e-3, 1e-2, 1e-1} 選（見 3.2）；同分取較大的 λ。

### 3.2 巢狀交叉擬合（用 inner 原本的 5 折玩家折；擬合和評估的玩家不重疊）

- M 只用**擬合假題**（`X_fit`）訓練。
- 對每個外層折 f（0…4）：
  1. 對每個 λ、每個 g ≠ f：用折 ∉ {f, g} 的擬合假題訓練 M_{f,g,λ}，預測折 g 的擬合假題 → 4 折合起來的等級平衡平均 NLL；取最小的 λ_f。
  2. 融合的訓練輸入：折 ≠ f 的擬合假題，CT 項用 M_{f,g,λ_f} 的預測（每一題的 CT 都來自沒看過該題玩家的模型）。
  3. 融合參數 θ_f：`p4.gen_fit_record`（照 Q09：OP、O4、SWA、SWB、HS、P1TO＝4·lp_P1TO_E，加 CT；所有 w 界限 (0, 4)、a (−4, 4)、γ (−1, 2)、起點 a＝0、1、PENALTY 0.01‖β‖²、MAXITER 500）。
  4. M_f＝用全部折 ≠ f 的擬合假題、λ_f 訓練；預測折 f 的評估假題和折 f 玩家的單色假題，CT＝lp_M_f；再用 θ_f（`p4.gen_logits`）算機率，規則 B。
- 共 5×4×4＋5＝85 次 M 擬合；隨機性只有 bootstrap（沒有其他隨機）。
- **CT 項的尺度**：E_CT＝lp_M（題目層級、不再乘倍數）。

### 3.3 對照、接縫

- **對照**：Q09（inner 0.7114420614163589、單色 0.7145431684624712，`results/prereg09/rank/tune_report.json`），由 `Inner.evaluate(lp_P1E)` 重算。
- **機器（只 import，不改）**：`tools/p09_rank_fuse.py`（`Inner`、`inner_map_lp`、`REF`）、`tools/p09_forward.py`、`tools/p08_rank_fuse.py`（`run_terms`、`masked_inner`、`pilot_hs_usable`）、`tools/p04_rank_fuse.py`（`gen_fit_record`、`gen_logits`、`gen_fold_predict`、`gen_params`）、`tools/p02_rank_fuse.py`（`rule_b_np`、`level_weights`、`softmax`、`log_softmax`）、`tools/p03_rank_fuse.py`、`results/prereg12/explore/finetune/ft_eval.py`（`InnerPrep`、`guard_path`）、`results/prereg05/explore/u1/u1_lib.py`。
- **接縫（先過，否則停，不算任何候選）**：
  1. Q02H、Q08、Q08 單色等於 `p09_rank_fuse.REF`；Q09、Q09 單色等於 `tune_report.json`；相差都 ≤ 1e-12。
  2. 本探索自己的分折迴圈在**不加 CT**（只有 Q09 的 6 項）時，評估假題和單色假題的機率和 `Inner.evaluate` 的 P、Pm **逐位元相同**，每折 θ 也相同。
  3. 逐手資料：每塊 SHA 等於兩份 manifest；HS 每盤 `bincount(pos_game)`＝`counts[:,0]`、逐手和與 `sums[:,0]` 相差 ≤ 1e-6；P1 每盤目標方手數＝`counts[:,0]`；每塊 `ref` 不重複。
  4. 外層折 f 的任何 M 的訓練列都不含折 f 的玩家；M_{f,g} 也不含折 g（程式內斷言，報告記錄）。

## 4. 閘門（單一候選 Q09＋CT，正式執行後只判一次；全部成立才算過）

- **G1**：inner 評估假題的分等級平均差（Q09＋CT − Q09）≥ **+0.010**（inner 已被 P1、08、09、prereg12、prereg15 重複使用，選擇偏差約 0.003；門檻留餘裕）。
- **G2**：同一差值的配對 PlayerBootstrap（`Inner.compare`：每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
- **G4**：單色假題（120,544 題）的分等級平均差，配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 ≥ 0（容許 1e-12）。
- 只有一個候選，不做多重比較校正。
- **只記錄、不能救候選**：M 單獨（規則 B 用 softmax(lp_M) 的 OOF 分數，評估和單色）；每折 λ_f、每折融合權重（w_CT 有沒有打到界限）；不含 E 交互的版本（同一套巢狀流程，66 維）的差；沒有可用盤的題數；HS／P1 手數不同的盤數。

## 5. 判定和之後

- **只判一次**：`p16_eval.py eval` 正式跑一次，寫 `results/prereg16/explore/eval_report.json`，原樣 commit。開始時先寫 `data/prereg16/eval_started.json`，中途中斷也算跑過；要重跑只准在指出具體程式錯誤、比上一次更新的 commit 之後（`--rerun --reason`），報告要記。正式執行要求本探索的檔和 import 的工具都已 commit 且乾淨，SHA 記進報告。
- **CPU smoke**（正式執行前，輸出在 `data/prereg16/smoke/`，不是結果）：特徵照常抽（確認讀檔、接縫 3、時間），但進模型前把 73 維換成 `default_rng(12345)` 的常態雜訊，跑完整的巢狀流程和融合；只確認程式能跑完、接縫 1／2 過、時間多長。數字沒有意義、不記進結果。
- **都沒過**：記錄後收掉，Q09 照舊。
- **過了**：只代表「值得寫 PREREG_16」，不是採用。採用需要（本探索都不做，主 session 決定）：
  - PREREG_16：登記唯一候選 Q09＋CT、凍結 M（用全部 inner 擬合假題、照 3.2 的規則選 λ 後重訓）和融合參數（全部擬合假題重擬合），inner-only 的採用規則（同 PREREG_09；偏離「不交沒在驗證集打過分的組合」要在報告寫明）；
  - 驗證題只做無標籤檢查（特徵抽取、方向、格式），照 09 的 apply／獨立稽核；
  - 測試日路徑：特徵只要 07 的 HS 塊和 08 的 P1 塊裡的逐手陣列（測試日本來就會產生，不用多跑 GPU）；要新工具 `t16`、假資料演練、測試、指紋和 tag、官方紀錄只加行，全部要在 11/04 前完成，否則不採用。
- **審查**：正式執行前，程式由另一個代理（子代理）獨立審查；修完才跑。

## 6. 檔案

- `p16_lib.py`（逐手→每盤統計、題目層級特徵、LR、巢狀交叉擬合的分折邏輯；純 numpy／scipy）、`p16_eval.py`（`eval [--smoke]`）、`test_p16.py`（合成資料的單元測試）、`eval_report.json`（正式執行後 commit）、`RESULTS.md`（判定和筆記）。
- 大檔、smoke 輸出在 `data/prereg16/`（不進 git）。只 import、不改：`tools/*`、`results/prereg05/explore/u1/u1_lib.py`、`results/prereg12/explore/finetune/ft_eval.py`、`results/prereg07/explore/pilot_humansl/*`、`results/prereg08/p1/*`。
- 環境：`.venv`（CPU），`PYTHONDONTWRITEBYTECODE=1`，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。worktree 在 scratchpad，用 `subst W:` 避開 u1_lib 的 `data\local` 防護誤擋（同 prereg15，防護本身不改）。

## 7. 實作補記（正式執行前，閘門和數字都沒改）

- 程式：`p16_lib.py`、`p16_eval.py`、`test_p16.py`（合成資料 12 個測試全過）。
- 更正筆誤：第 4 節「不含 E 交互的版本」是 **62 維**（37＋10＋11＋4），不是 66 維。
- 定義寫明：D 的「目標手數的 log 平均」＝可用盤 log(HS 目標手數) 的平均。
- CPU smoke（`data/prereg16/smoke/`，62 維基本特徵換成雜訊、交互項由雜訊算）：接縫 1 五個數逐位元相同、接縫 2（自己的分折迴圈＝Q09 的 P、Pm、θ）逐位元相同、接縫 3（367＋367 塊 SHA、手數、逐手和差 0）、接縫 4（玩家不重疊）都過；P1 可用＝HS 可用＝73,169 盤，每題至少 5 盤可用；全程約 4 分鐘。數字無意義，不記。
- 獨立審查（另一個子代理，只讀程式和跑單元測試，沒跑正式執行）：沒有 blocker、沒有 major。minor 都已修：① G1 改成嚴格 ≥ +0.010（原本多了 1e-12 容許）；② 正式執行時核對 `p16_lib.level_weights` 和 `rf.level_weights` 相同（另加測試）；③ D 的手數定義寫明（上一點）；④ smoke 說明改成 62 維；⑤ 報告加 `seam4`（每折玩家重疊數，全部要是 0）；⑥ 可用盤的 lp_HS 不能是全 0（`hs_baseline` 覆蓋核對），HS 塊的 `pos_side` 必須全是目標方。
