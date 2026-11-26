# prereg40 探索計畫：S40＝SET2 的輸入與聚合 × S25 的玩家（競賽二，inner-only）

2026-11-12 寫成，**在任何 S40 訓練、任何 S40 數字之前 commit**。只用訓練玩家（probe＋split v1 的驗證玩家 V，和 S25 相同）與 inner 訓練玩家；不碰驗證集答案資料夾、不讀驗證題特徵、不跑 `tools/score.py`、不用 GPU。分支 `p40-plan`，不合併到 main（合併由主 session 決定）。

## 0. 白話

- 25 的 SET 項（S25）＝每盤 1,540 維 P1 → 64 維 → 一題取平均 → 序數頭，3 個種子，訓練玩家 6,619 位（含 1,440 位驗證玩家）。
- prereg23 的 SET2 在**舊的玩家集合**（S19 的 5,138 位）上，把輸入加上 EX 44 維、聚合改成「平均＋標準差」、5 個種子：取代 S19 時 inner +0.0064、單色 +0.0028（G4 下界 −0.0002，差一點）。
- prereg25 的 S25 在**同一個輸入**上，把玩家加到 6,619 位：inner +0.0111、單色 +0.0052。
- 兩件事（更多輸入、更多玩家）各自量過、方向都是正的，**合起來還沒試過**。S40＝SET2 的模型與訓練設定（一字不改），用 S25 的資料列和 S25 的 A／H 分組訓練；Q40＝Q25 的 SET 項換成 S40。只判一次。
- 預期（推估，不是結果）：inner +0.003～+0.007、單色 +0.002～+0.004；G4 可能又差一點。過了也只代表「值得寫 PREREG_40」，不是採用。

## 1. 為什麼是這個（診斷摘要；數字來源都是已 commit 的檔）

| 項目 | 數字 | 來源 |
|---|---|---|
| 官方測試題型 | 每題單色、5／10／15／20 盤各 100 題（public、private 各 400） | `results/test02/checks/rank/t5.json`、`docs/DATA.md` |
| Q25 單色（＝測試題型）依盤數 | 5 盤 0.6889、10 盤 0.7540、15 盤 0.7806、20 盤 0.7952 | `results/prereg25/explore/eval_report.json` record_only.by_game_count.mono_by_k |
| 依盤數的失分占比（1 − 分數，四組相加） | 5 盤 32%、10 盤 25%、15 盤 22%、20 盤 21% | 同上（本計畫計算） |
| Q25 單色依等級 | D 0.862、C 0.770、B 0.780、A 0.710、1D 0.664、2D 0.650、3D 0.673、4D 0.714、5D 0.781、6D 0.879 | 同上 by_level.mono |
| 完全猜中率（Q19，評估假題） | D 0.796、C 0.640、B 0.608、A 0.525、1D 0.504、2D 0.460、3D 0.497、4D 0.583、5D 0.572、6D 0.827 | `results/prereg19/explore/audit.json` confusion_diag_eval |
| 25 測試題的模型自估期望分數 vs public | 自估 public 0.7582（private 0.7419）；實際 0.7238，差 −0.034 | `results/test25/rank/rank_test_probs.csv`（無標籤）、`results/official_scores.json` |
| 同上，7 個上傳版本 | 差 −0.029～−0.034，R20（官方 CNN）到 25（KataGo）都一樣 | `results/test0{2,4,7,8,9}/`、`test19/`、`test25/` 的 `rank_test_probs.csv` |
| inner 增益到 public 的轉移 | 09→19：inner +0.023、public +0.023；19→25：inner +0.011（單色 +0.005）、public +0.0095 | 各 RESULTS、`official_scores.json` |

讀法：
- 丟分最多的是 5 盤題和中段等級（A、1D–4D）。25 的增益集中在多盤題，5 盤題幾乎沒有。
- public 比模型自估低約 0.03，而且和方法無關（R20 也一樣）→ 是這 400 題本身（或所有特徵共有的分布差），不是 25 特有的毛病；inner 的增益大致 1:1 反映到 public，所以 inner 判定仍是對的工具。
- 決策層已經到頂：規則 B 本來就是期望分數最大化（prereg01）；黑白分開重擬合 +0.0003、等級分布調整全負（prereg15）。
- 補玩家（prereg24／33／34）、長訓練（prereg32）、HD（prereg31）、P1 v2（prereg30）、多種子（prereg29）、微調 KataGo（prereg12）都沒過。剩下有正面證據、又還沒組合過的，就是「S25 的玩家 × SET2 的輸入」。

## 2. 候選（唯一）

### 2.1 S40（模型與訓練：prereg23 的 SET2，一字不改）

- 匯入、不改：`results/prereg23/explore/p23_train.py` 的 `torch_model`、`train_seed`、`port_check`、`export_params`、`holdout_nll`，`p23_lib.py` 的 `features`、`h_np`、`head_np`、`pool_mean_std`、`set_term`、所有超參數。
- 每盤輸入 [standardise_x(X) 1,540, standardise_ex(EX) 44（非有限值→0）]＝1,584；標準化用 09 凍結的 `results/prereg09/rank/p1e_map.npz`。
- 聚合：一題 64 維 h 的 [平均, 母體標準差]＝128 → rho → 累積連結（序數）頭。
- 種子 0–4（5 個）、`SAMPLER_BASE` 23、15,000 步、每 500 步在留出題算 NLL、取最低那步；lr 1e-3、wd 1e-2、batch 128、KS 5／10／15／20、8 threads。都和 prereg23 相同。

### 2.2 資料列與分組（照 S25，一字不改）

- 列＝probe（`data/prereg17/prep`：`probe_X.npy`、`probe_ex.npy`、`probe_meta.npz`；270,000 列）接著 V（`data/prereg25/prep/v_X.npy`、`v_meta.npz`；74,946 列）；V 的 EX＝`data/prereg26/prep/v_ex.npy`（prereg26 從 prereg25 的 V 塊重讀，`v_ex_record.json` 的 SHA 與接縫）。
- A／H：`p25_lib.holdout_bit`（鹽 `c10-prereg25|holdout|`）→ A 6,619、H 748，和 S25 相同。
- 留出題：`p19_lib.holdout_set(pool_h, 100, 19019)`（S25 的設定）→ 和 S25 **同一批 4,000 題**（每盤數 1,000 題），所以 S40 和 S25 的留出 NLL 可以直接比（第 4 節）。
- 不讀任何 inner 列。

### 2.3 Q40（融合：照 Q25）

- 7 項 OP、O4、SWA、SWB、HS、P1TO、SET；只把 SET 項換成 E_SET40＝4 × lse_mean（5 個種子）of head([mean, std] of 可用盤的 h)（`p23_lib.set_term`；可用盤＝`Inner.covered`，同 Q25）。
- 融合 `tools/p09_rank_fuse.Inner.run`（p04 的廣義校準、權重 (0, 4)、5 折玩家 OOF、規則 B），不改。

## 3. 接縫（全部要過才判定，否則停下查）

- **接縫 1**：Q02H、Q08、Q08 單色、Q09、Q09 單色、Q19、Q19 單色等於 prereg25 的參照；**Q25 inner 0.7456519871546352、Q25 單色 0.7482450052195216**（`results/prereg25/explore/eval_report.json`）用凍結的 S25 模型重建，相差 ≤ 1e-12（預期逐位元相同）。
- **接縫 P**：`probe_X.npy`、`probe_ex.npy`、`inner_X.npy`、`inner_ex.npy` 的 SHA 等於 09 的 `check_report.json`；`v_X.npy` 等於 prereg25 `prep_record.json` 與 `train_record.json`；`v_ex.npy` 等於 prereg26 `v_ex_record.json`。
- **接縫 D**：A 6,619、H 748（各級人數和 S25 `train_record.json` 的 players 相同）；A∩H＝0；訓練玩家∩inner＝0（含只比 player_id）；probe∩V＝0。
- **接縫 H**（訓練時）：用 S25 的 3 個凍結模型（numpy float64）在本計畫建的留出題上算 NLL，和 S25 `train_record.json` 的 `best_holdout_nll` 相差 ≤ 1e-5（證明留出題相同）。
- **接縫 M**：5 個 S40 模型檔 SHA、seed、best_step 等於 `train_record.json`；訓練時程式乾淨；前向程式（`p23_lib.py`、`p23_train.py`、`p19_lib.py`、`tools/p09_forward.py`）的 LF SHA 和訓練時相同；torch／numpy 移植核對 ≤ `TORCH_TOL`。
- **接縫 S25**：凍結的 S25 模型檔 SHA 等於 prereg25 `train_record.json`。

## 4. 階段 0b：留出 NLL 先篩（訓練完就算，不碰 inner）

- 在第 2.2 節的 4,000 題留出題上，比 S40 集成（5 種子 lse 平均）和 S25 集成（3 種子 lse 平均）的平均 NLL（numpy float64）。
- **過**：Δ＝NLL(S40) − NLL(S25) ≤ **−0.005**。沒過就停：不跑 inner、不寫 PREREG_40，inner 少用一次。
- 依據：prereg31 的 0b 是 −0.0102 時 inner 只有 +0.0047（差一點沒過 +0.005）；−0.005 只是便宜的篩子，真正的判定仍是第 5 節。
- 只記錄：每個盤數各自的 Δ、S40 前 3 個種子（0–2）的集成 Δ（和 S25 同種子數）、每個種子單獨的 NLL。
- 0b 的數字寫進 `stage0b.json`，commit 之後才可以跑第 5 節。

## 5. 閘門（判一次，差＝Q40 − Q25）

| 閘門 | 條件 |
|---|---|
| G1 | inner 評估假題（127,200 題）分等級平均差 ≥ **+0.005** |
| G2 | 同一差值的配對 PlayerBootstrap 95% 下界 > 0 |
| G4 | 單色假題（120,544 題）差的 95% 下界 > 0，且黑題、白題的差各 ≥ 0 |

- `p23_lib.gates`（門檻 0.005）原樣使用。
- **只記錄、不能救候選**：依盤數（單色 5／10／15／20、評估假題）、依等級、S40 單獨（規則 B）、每個種子單獨換進去、S40 只用種子 0–2、Q25＋S40 當第 8 項、每折權重和是否打到界限、峰值記憶體。
- 判定只做一次。程式錯誤要重跑，必須先 commit 寫明錯誤的修正（`--rerun --reason`）。

## 6. 已知偏差（要寫進結果和報告）

- inner 已被 P1、08、09、prereg12、15–34 重複使用，選擇偏差至少約 0.003，而且每多判一次就再累積。prereg23 寫過「不建議用同一個 inner 再試相近的變體」；這次仍做，理由是它不是 SET2 的新超參數，而是兩個已分別判過的因素（輸入、玩家）第一次組合，門檻不降。
- prereg29 顯示 S25 的 3 個種子在 inner 上剛好偏好（7 個新種子單獨換進去都比較低）→ 對照 Q25 的 inner 可能偏高約 0.001～0.003，對候選是保守的。
- 訓練用了驗證玩家（同 25）；競賽二的驗證集打分已用完，報告要寫明。

## 7. 資源與執行（本代理不開跑；要排程 session 同意）

| 步驟 | 指令（Git Bash、repo 根目錄；`S=results/prereg40/explore`） | 估計 |
|---|---|---|
| smoke 訓練 | `$S/run_train.sh --smoke` | 幾分鐘；雜訊輸入、每種子 300 步 |
| 訓練 | `$S/run_train.sh` | CPU 5 種子 × 約 12–20 分鐘＝約 1–1.7 小時（prereg23 實測 84 分鐘／5 種子，資料多 28%）；記憶體峰值估 3–3.5 GB（X32＝344,946 × 1,584 float32＝2.2 GB） |
| 0b | `$S/p40_eval.py stage0b` | 1–2 分鐘、< 3 GB |
| 判定 | `$S/run_eval.sh` | 約 5–10 分鐘；記憶體峰值估約 4 GB（prereg25／29／31／33 的判定都是 3.8 GB） |

- CPU 單一程序、BELOW_NORMAL、8 threads；可用 RAM < 4 GB 就停（建 `data/prereg40/STOP`）。不用 4090。
- 中斷：`train --resume`（同一 commit、同一程式；已完成的種子保留）。
- 截止：11/18 23:59 前要判完並（若過）由主 session 決定是否寫 PREREG_40；過了這天就不做（11/20 凍結）。

## 8. 過了之後（本探索不做）

- 寫 PREREG_40（唯一候選 Q40，凍結 5 個 S40 模型檔，用全部 inner 擬合假題重擬合 7 項）。
- 測試日只用 CPU：測試題的 P1 X 已在 `results/test08/p1/{public,private}_p1.npz`（t25 讀的同一份），EX 由同一批 P1 塊讀（09 的測試日已算過 EX，要核對同一份）；以 `tools/t25_rank.py` 為底做 t40。任何一步出問題就維持 25。
- 沒過：記錄後收掉「再換 set 模型」這條線（不換種子數、不換超參數重試）。

## 9. 檔案

- `PLAN.md`（本檔）、`p40_lib.py`（常數、資料列、0b 的 numpy 計算）、`p40_train.py`（train）、`p40_eval.py`（stage0b／eval）、`test_p40.py`、`run_train.sh`、`run_eval.sh`；之後 `train_record.json`、`stage0b.json`、`eval_report.json`、`RESULTS.md`。
- 大檔在 `data/prereg40/`（不進 git）：`models/s40_seed{0..4}.npz`、log、smoke。

## 10. 修訂 1（2026-11-12，寫程式時；仍在任何 S40 訓練與數字之前）

- **接縫 H 的容許差 1e-5 → 1e-3，並加兩個逐字相等條件**：S25 的 `best_holdout_nll` 是 torch float32 算的，numpy float64 重算時，真類機率接近 1e-12 下限的題會差到數個 nat（prereg19／25 的移植核對記過 log 機率最大差約 11），平均到 4,000 題可能到 1e-3 量級。所以改成：(a) 3 個種子各自 |numpy − 紀錄| ≤ 1e-3；(b) 留出題數（每盤數 1,000）等於 S25 `train_record.json` 的 `holdout_questions`；(c) 留出題抽樣紀錄（`holdout_sampling`）和 S25 的逐字相等。(b)(c) 才是「同一批題」的主要證據。
- 接縫 D 在訓練時也做：A／H 的人數、盤數、V 人數、各級人數都要和 S25 `train_record.json` 的 players 相同，不同就停（smoke 只記錄）。
- 0b（`p40_eval.py stage0b`）重建同一批留出題時，也要求 (b)(c) 和 S40 自己的 `train_record.json` 相等；S40 用 [X, EX] 的 float32 特徵、S25 用其中的 X 部分，兩者都以 numpy float64 前向。
- 判定時，Q19 的接縫算完就釋放（省記憶體），再用凍結 S25 重建 Q25；Q25 參照值綁 `results/prereg25/explore/eval_report.json` 的 candidate 欄（兩個數要逐字相等才算接縫 1）。
- 程式：`p40_lib.py`、`p40_train.py`、`p40_eval.py`、`test_p40.py`（合成資料 7 項）、`run_train.sh`、`run_eval.sh`。本代理只跑了單元測試和 `--help`（匯入檢查），沒有 smoke、沒有訓練、沒有讀任何 inner 或留出特徵。
