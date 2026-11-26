# prereg19 探索：題目層級（set）模型當 Q09 的新項（競賽二，inner-only）

2026-10-20 寫成，**在算出任何候選數字之前 commit**（分支 `p19-explore`）。使用者 10/20 同意的點子：`analysis/codex_ideas_1020.md` 第 (b)1 節。只用 CPU（`.venv` 的 torch CPU）；不碰 `data/local`、不讀任何驗證題特徵、不跑 `tools/score.py`、不跑新的 KataGo。本檔的閘門在正式執行前寫死，正式執行後只判一次（`eval_report.json` 原樣 commit）。

## 0. 白話版

- Q09 現在的做法：6 個小模型**每一盤各自**猜等級（每盤 1,540 個 KataGo 探針數字 → 10 級機率），一題 5～20 盤的對數機率取平均，乘 4 放進融合，再由融合依盤數調信心。模型訓練時從沒看過「一整題」。
- 這次問：**直接用「一整題」來訓練**會不會更好？做法：每一盤先經過一個共用的小網路變成 64 個數字，同一題的盤把這 64 個數字取平均，再經過一個小 head 算出一個棋力分數，切成 10 級（順序模型，同 09 的 mlpclm）。訓練時每一筆樣本就是一道「假題」：同一位玩家、同一執色的 5／10／15／20 盤，損失是整題的 NLL。這樣模型會直接學「跨盤穩定的訊號」，不用每一盤單獨猜對。
- 用法（**只登記一個候選**）：當作 Q09 融合的**第 7 項** SET（Q09＋SET），權重由融合自己擬合。
- 訓練資料只用 probe（fit 玩家，和 inner、驗證玩家都不重疊），和 Q09 的 6 個成員同一批；inner 只用來評估，融合參數照 Q09 的 inner 協定以玩家分 5 折交叉擬合。
- 先講預期：Codex 的規劃估計 0～+0.010，「沒有增益」是主要可能。閘門照舊 +0.010。
- 競賽二已經沒有驗證集打分機會：就算過關，採用也只能像 09 一樣靠 inner，而且要另寫 PREREG_19 和測試日路徑（本探索**不做**，只寫筆記，由主 session 決定）。

## 1. 為什麼選「加一項」（ii），不選「取代 P1TO」（i）

- (ii) 包含 Q09：融合把 w_SET 擬合成 0 就回到 Q09 的項組，傷害有限；(i) 會丟掉 6 成員集成（含唯一吃 EX 的 `lowrank3_ex`），而 09 探索顯示單一模型的 inner 種子雜訊約 ±0.003、單一模型都輸給集成。
- 點子的假設是 set 訓練抓到「跨盤聚合」的**互補**訊號，不是同一份訊號換一個更好的翻譯；互補訊號用加一項來檢驗最直接（同 prereg16 的 CT、prereg17 的第 7 成員的做法）。
- (i) 只記錄，不能救候選（第 6 節）。

## 2. 資料和玩家（全部是訓練玩家，沒有驗證玩家）

| 角色 | 玩家 | 用途 |
|---|---|---|
| A（訓練） | probe（P1＋P2＋P3，270,000 盤，5,767 位 fit 玩家）中 `int(sha256("c10-prereg19|holdout|<rank>:<player_id>")[:8], 16) % 10 != 0` | 訓練 set 模型 |
| H（留出） | 上式 `== 0` 的 probe 玩家（約 10%） | 只用來定「停在哪一步」（第 3.4 節） |
| I（評估） | inner 訓練玩家（split v1 inner，636 位，5 折玩家折） | 正式評估；融合參數 OOF |

- 特徵：每盤 1,540 維 P1 TO 特徵（probe：`data/prereg17/prep/probe_X.npy`，接縫 P 已在 prereg17／18 證明等於 09 的 prep；inner：`data/prereg12/q09_inner/inner_X.npy`），用 Q09 凍結的 `p1e_map.npz` 的 μ、σ 以 `p09_forward.standardise_x` 標準化（＝6 個成員看到的完全相同的輸入）。EX 不用（只用 1,540 維）。
- 等級：probe 由 `probe_meta.npz` 的 `y`；執色：`data/prereg08/p1/sets.json`、`data/prereg08/p2/sets.json`、`data/prereg09/p3/sets.json` 的 `color`（玩家鍵交叉核對，同 prereg18 `probe_players`）。
- 盤點（10/20，只數盤數，不含任何模型或分數）：probe 每級 27,000 盤；玩家每級 312～1,140 位；某執色有 ≥ 20 盤的玩家每級 265～399 位。
- inner 的可用盤：P1 可用（`Inner.covered`，73,169 盤），和 Q09 的 P1TO 項相同。

## 3. set 模型 S19（寫死）

### 3.1 架構

- 每盤 h(x)：Dropout(0.1) → Linear(1540, 256) → GELU（erf 版）→ Dropout(0.3) → Linear(256, 64) → GELU，得到 64 維。
- 題目內聚合：v＝題內各盤 h 的平均（只對該題的盤，沒有補零）。
- head ρ(v)：Linear(64, 32) → GELU → Linear(32, 1)＝棋力分數 f。
- 順序（累積 link）輸出，同 09 的 mlpclm：θ₁＝t0，θ_k＝t0＋Σ softplus(dt)（t0 起點 −4，dt 起點 log(e−1)），P(y ≤ k)＝σ(θ_k − f)，相鄰相減、下限 1e-12 再取 log → 10 維 lp。
- 盤數不進模型（不提供類別偏移）；信心依盤數的校準交給融合原有的 t＝e^a (n/12)^γ。

### 3.2 抽題（每一步一個 batch）

- batch 128 題，同一個 batch 的盤數 n 相同：n 從 {5, 10, 15, 20} 均勻抽（和等級獨立）。
- 每一題：等級 k 從 0…9 均勻抽 → 執色從 {B, W} 均勻抽 → 在 A 裡等級 k、該執色可用盤 ≥ n 的玩家中均勻抽一位（該執色沒有合格玩家就改另一個執色；兩個都沒有就重抽等級；次數記錄）→ 從該玩家該執色的盤中不放回抽 n 盤（同一題沒有重複的盤；不同題、不同步之間可以重複用到同一盤）。
- 亂數：`numpy.random.default_rng([19, 種子])`。

### 3.3 最佳化

- 損失：batch 內每題 NLL 的平均（等級已均勻抽，不再加權）。
- AdamW lr 1e-3、weight decay 1e-2；OneCycleLR（max_lr 1e-3、pct_start 0.1、總步數 15,000）；`torch.manual_seed(種子)` 後建模型；`torch.set_num_threads(8)`；輸入 float32。
- 總共 15,000 步（每步約 0.09 秒，10/20 用隨機張量量的；一個種子約 25 分鐘）。

### 3.4 停在哪一步（唯一的選擇，事前寫死）

- 留出題組：用 H 的玩家、同 3.2 的抽題規則，每級 × 每種盤數各 100 題（共 4,000 題，執色同 3.2 抽），`default_rng(19019)`，訓練前產生一次、所有種子共用。
- 每 500 步（500、1,000、…、15,000，共 30 次）在留出題組上算平均 NLL（dropout 關閉）；**平均 NLL 最小的那一步的權重**就是這個種子的模型（同分取較早）。不另外用全部 probe 重訓。

### 3.5 種子和集成

- 3 個種子（0、1、2），各自照 3.2–3.4。
- 集成：lp_SET＝log( max( 3 個種子機率的平均, 1e-300 ) )（同 `p09_forward` 的 `lse_mean`）。

### 3.6 前向（評估用）

- inner 的 lp_SET 一律用 **numpy float64 前向**（權重存 float32、讀進來轉 float64；GELU 用 `scipy.special.erf`），不用 torch。
- 每盤先算 h（73,169 盤 × 64），每題對該題的 **P1 可用盤**取平均（`rf.pool_matrix` 只放可用盤，和 `P7.masked_E` 同一個遮罩），再算 ρ 和 lp。沒有可用盤的題：E_SET＝0 向量（同 `masked_E`），題數記錄。
- 移植核對（訓練時）：同一份權重，torch float32 和 numpy float64 在留出題組上的機率，每題最大差 ≤ 1e-4，否則停下查。

## 4. 候選 Q09＋SET（唯一一個）和交叉擬合

- 融合：`Inner.run(Q08_TERMS + ["SET"], {"HS": E_hs, "P1TO": E_P1TO_E, "SET": E_SET})`，E_SET＝**4**·lp_SET（題目層級，乘 4 和 P1TO 相同：Q09 的 P1TO 權重頂在界限 4，題目層級的 lp 已經比每盤 lp 的平均銳利，不乘的話權重很可能打到界限）。其餘照 Q09：項的順序 OP、O4、SWA、SWB、HS、P1TO、SET；所有 w 界限 (0, 4)、a (−4, 4)、γ (−1, 2)、起點 a＝0、1、PENALTY、MAXITER、等級加權、規則 B、PlayerBootstrap 都不變（`p08_rank_fuse.run_terms`，不改）。
- **為什麼不會洩漏**：set 模型只用 probe 玩家（A、H）訓練和選步，inner 玩家從來不在裡面（接縫 D）；融合參數照 Q09 的 inner 協定，第 f 折的參數只用其他 4 折玩家的擬合假題擬合，評估第 f 折的評估假題和單色假題（`Inner.run`，不改）。所以每一題的機率都來自沒看過該題玩家的 set 模型和融合參數，和 Q09 的評估方式一模一樣。

## 5. 接縫（先過，否則停，不算任何候選）

1. Q02H、Q08、Q08 單色等於 `p09_rank_fuse.REF`；Q09（inner 0.7114420614163589）、Q09 單色（0.7145431684624712）等於 `results/prereg09/rank/tune_report.json`；相差都 ≤ 1e-12。
2. **接縫 E**：本探索自己的「每盤 → 題目平均（只用可用盤）」程式，用在 4·lp_P1TO_E 上，擬合／評估／單色三組的 E 和 `Inner.E_p1` 相差 ≤ 1e-12（驗證題目分組、可用遮罩、列號對應）。
3. **接縫 D**：A ∩ H＝∅；（A ∪ H）∩ I＝∅（以 `<rank>:<player_id>` 比）；只比 player_id 的交集只記錄。
4. **接縫 P**：`probe_X.npy`、`inner_X.npy` 的 SHA-256 等於 `results/prereg09/rank/check_report.json` 的 `prep.files_sha256`。
5. **接縫 M**：3 個模型檔的 SHA 等於已 commit 的 `train_record.json`；訓練時的移植核對（3.6）過。

## 6. 閘門（單一候選 Q09＋SET，正式執行後只判一次；全部成立才算過）

- **G1**：inner 評估假題（127,200 題）的分等級平均差（Q09＋SET − Q09）≥ **+0.010**（嚴格，不加容許；inner 已被 P1、08、09、prereg12、15、16、17、18 重複使用）。
- **G2**：同一差值的配對 PlayerBootstrap（`Inner.compare`：每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
- **G4**：單色假題（120,544 題）的分等級平均差，配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 ≥ 0（容許 1e-12）。
- 只有一個候選，不做多重比較校正。
- **只記錄、不能救候選**：
  - (i) 取代版：SET 放在 P1TO 的位置（`Inner.run(Q08_TERMS, {"HS": E_hs, "P1TO": E_SET})`）的 inner／單色分數和差；
  - SET 單獨（規則 B 用 softmax(lp_SET)）的 inner／單色分數；
  - 每個種子單獨的 Q09＋SET_s（3 個）的差；
  - **依盤數**：單色假題 n＝5／10／15／20 各自的差（分等級平均，配對 PlayerBootstrap 95% 區間）；評估假題依 `rf.N_BINS`（5–7、8–12、13–20）和 n 正好是 5／10／15／20 的子集的差；
  - **依等級**：評估、單色每一級的平均分數差；
  - 每折融合權重（w_SET 有沒有打到界限）；
  - 訓練紀錄：每個種子留出 NLL 曲線、選到的步數、訓練秒數、抽題時改執色／重抽等級的次數、各級各盤數的合格玩家數、移植核對的最大差。
- 不論判定結果，報告和 RESULTS.md 都寫出依盤數的差。

## 7. 順序、判定和之後

1. 本檔 commit（任何候選數字之前）。
2. 程式、合成資料的單元測試、smoke。**smoke**（輸出全部在 `data/prereg19/smoke/`，不是結果）：probe 和 inner 的 P1 特徵換成 `default_rng(12345)` 的常態雜訊（形狀不變），訓練每個種子只跑 300 步、每 100 步評一次留出，留出題組每級 × 每種盤數 5 題；train 和 eval 的整條流程（含接縫、融合、閘門、只記錄的項目）跑完。只確認程式能跑、接縫過、時間多長；數字沒有意義、不記進結果。
3. **審查**：正式執行前，程式由另一個子代理獨立審查（只讀程式、跑單元測試）；做不到就由本代理做一輪對抗式自審並寫明。修完、commit 才跑。審查結果寫進第 9 節。
4. **train 只跑一次**：`p19_train.py train`（`.venv`，CPU，獨立背景程序；約 75 分鐘），寫 `data/prereg19/models/`（3 個模型檔）和 `results/prereg19/explore/train_record.json`（commit）。train 不讀任何 inner 資料。開始時先寫 `data/prereg19/train_started.json`；中斷或要重跑，只准在指出具體程式錯誤、比上一次更新的 commit 之後（`--rerun --reason`）。
5. **eval 只判一次**：`p19_eval.py eval` 正式跑一次，寫 `results/prereg19/explore/eval_report.json`，原樣 commit。開始時先寫 `data/prereg19/eval_started.json`，中途中斷也算跑過；要重跑只准在指出具體程式錯誤、比上一次更新的 commit 之後（`--rerun --reason`），報告要記。正式執行要求本探索的檔和 import 的工具都已 commit 且乾淨，SHA 記進報告。
- **沒過**：記錄後收掉，Q09 照舊。
- **過了**：只代表「值得寫 PREREG_19」，不是採用。只在 RESULTS.md 寫筆記說明 PREREG_19 和測試日路徑需要什麼（本探索**不做**）。

## 8. 檔案和環境

- `p19_lib.py`（純 numpy／scipy：玩家分割、抽題、留出題組、numpy 前向、題目平均、分等級平均、閘門）、`p19_train.py`（`.venv` torch CPU：`train [--smoke]`）、`p19_eval.py`（`.venv`：接縫、融合、閘門、只記錄項目、報告；`eval [--smoke]`）、`test_p19.py`（合成資料的單元測試）、`train_record.json`、`eval_report.json`、`RESULTS.md`。
- 大檔、smoke 輸出在 `data/prereg19/`（不進 git）。只 import、不改：`tools/*`、`results/prereg05/explore/u1/u1_lib.py`、`results/prereg12/explore/finetune/ft_eval.py` 等（同 prereg16／17／18）。
- 環境：`.venv`（Python 3.12.13、torch 2.14.0+cpu、numpy 2.5.3、scipy 1.18.1），`PYTHONDONTWRITEBYTECODE=1`，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。不用 GPU。worktree 在 scratchpad，用 `subst Q:` 避開 u1_lib 的 `data\local` 防護誤擋（scratchpad 路徑含 `AppData\Local`），`core.autocrlf=false` checkout。

## 9. 實作補記（正式執行前）

閘門、第 3 節的架構／抽題／步數／種子／停步規則、乘 4、候選都沒改。

- 程式：`p19_lib.py`、`p19_train.py`、`p19_eval.py`、`test_p19.py`（合成資料 13 個測試全過，含抽題的不變量、torch 和 numpy 前向一致、題目平均等於 `rf.pool_matrix`、閘門）。
- 盤點（smoke 時讀到，不是結果）：A 5,186 位玩家（242,942 盤）、H 581 位（27,058 盤）；`p19_train.py` 只開 probe 的 `probe_X.npy`、`probe_meta.npz`，不開 inner 陣列。
- smoke（`data/prereg19/smoke/`，特徵換成雜訊、每個種子 300 步）：train 約 4 分鐘（每步約 0.06 秒，正式 15,000 步估計每個種子約 15 分鐘）；移植核對最大機率差約 1.5e-7；eval 的接縫 1（五個數逐位元相同）、接縫 E（三組差 0）、接縫 P、接縫 D（A∩H、probe∩I 都是 0，只比 player_id 也是 0）、接縫 M 全過；eval 約 4～8 分鐘。同一份 smoke 重跑兩次，留出 NLL 和 eval 數字逐位元相同（CPU torch 在這台機器可重現，只記錄）。數字無意義，不記。
- 寫明：3.4 的留出題組在某級、某盤數兩種執色都沒有合格的 H 玩家時**略過那一題**（記 `level_missing`），不重抽等級，以保持各級題數平衡（3.2 的重抽只用在訓練 batch）。
- `eval_started.json` 改在**所有接縫都過之後、算 SET 項之前**寫（第 7 節原寫「開始時」）：接縫失敗不會用掉唯一一次判定，仍在任何候選數字之前。
- 獨立審查（另一個子代理，只讀程式並跑單元測試，沒跑 train／eval）：沒有 blocker。正式執行前都已修：
  - major：`--resume` 可能混用不同版本的程式 → 續跑必須和 `train_started.json` 同一個 commit、同一份程式 SHA，而且保留的種子要有通過的移植核對。
  - minor：① `eval_started.json` 寫在讀輸入之前 → 移到接縫之後（上一點），缺任何列出的檔就停；② train 原本經 `Prep` 開了 inner 陣列（沒用到、沒有標籤）→ 改成只開 probe；③ `code_state` 涵蓋不全 → 加 `tools/p09_map.py`、`tools/p08_p1.py`，並自動加入所有已載入的 repo 模組；④ eval 加核對 train 紀錄的設定等於登記值、train 時程式是乾淨的（接縫 M）；⑤ 留出題組略過而不重抽（上面寫明）。
