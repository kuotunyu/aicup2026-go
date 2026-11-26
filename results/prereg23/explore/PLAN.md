# prereg23 探索：第二個 set 模型 SET2 當 Q19 的第 8 項（競賽二，inner-only）

2026-11-05 寫成，**在算出任何候選數字之前 commit**（分支 `p23-explore`）。主 session 交辦（使用者 10/2、10/16 授權 AI 決定方法）。只用 CPU（`.venv` 的 torch CPU）；不碰 `data/local`、不讀任何驗證題特徵、不跑 `tools/score.py`、不跑 KataGo、不用 GPU。閘門在正式執行前寫死，正式執行後只判一次（`eval_report.json` 原樣 commit）。這是探索，不寫 PREREG_23、不做測試日工具、不上傳。

## 0. 白話版

- 現在交的競賽二是 19＝Q09＋SET。SET（S19）是「整題訓練」的 set 模型：每盤 1,540 個 KataGo 探針數字 → 64 維 → 同一題取**平均** → 棋力分數 → 10 級。prereg19 的結果：+0.0231，而且盤數越多增益越大。
- 這次問：再訓練一個**不一樣的** set 模型 SET2，當第 8 項加進去（Q19＋SET2），會不會再多一點？SET2 和 S19 只差三件事（事前寫死）：
  1. 每盤的輸入多加 09 的 44 個 EX 特徵（逐手機率的分布形狀：標準差、分位數、分段平均等；09 只有 `lowrank3_ex` 一個成員用到，S19 沒用）。
  2. 同一題的聚合從「平均」改成「平均＋標準差」（每盤 64 維 → 題目 128 維）：讓模型看得到「這位玩家各盤穩不穩定」。
  3. 種子 5 個（S19 是 3 個）。
- 其他全部照抄 prereg19（架構大小、抽題、batch、步數、OneCycle、留出選步、numpy float64 前向、乘 4、融合程式）。用新的留出鹽和亂數種子，所以是另一組獨立訓練的模型。
- 預期：沒有增益是主要可能（SET2 和 SET 很像，融合可能只是把權重分給兩個）。閘門 +0.005。

## 1. 為什麼這樣設計

- 只登記一個候選（Q19＋SET2，加一項）：Q19 是現行最終版，「加一項」在融合把 w_SET2 擬合成 0 時就回到 Q19。取代 SET 的版本（Q09＋SET2）只記錄、不能救。
- 三個差異是同一個假設的三個面：「整題訓練還有空間」（Codex 的教訓：表示方式到頂，訓練方式還有空間）。EX 是 09 已在用的特徵（不是新表示），std 聚合是 set 模型最常見的延伸，5 個種子降低 set 模型的種子雜訊（prereg19 單一種子 +0.0185～+0.0223）。三者一起測，不拆開（不做消融；拆開要多個候選，多重比較不划算）。

## 2. 資料和玩家（全部是訓練玩家，沒有驗證玩家）

| 角色 | 玩家 | 用途 |
|---|---|---|
| A（訓練） | probe（P1＋P2＋P3，270,000 盤，5,767 位 fit 玩家）中 `int(sha256("c10-prereg23|holdout|<rank>:<player_id>")[:8], 16) % 10 != 0` | 訓練 SET2 |
| H（留出） | 上式 `== 0` 的 probe 玩家（約 10%） | 只用來定「停在哪一步」（第 3.4 節） |
| I（評估） | inner 訓練玩家（split v1 inner，636 位，5 折玩家折） | 正式評估；融合參數 OOF |

- 鹽和 prereg19 不同（`c10-prereg23|holdout|`），所以 A／H 和 prereg19 的 A／H 不同；H 只用於 SET2 的選步。S19（凍結）是用 prereg19 的 A 訓練的，可能含本次 H 的玩家，不影響：H 不參與評估，inner 玩家從來不在任何訓練裡（接縫 D）。
- 每盤特徵（1,584 維）：
  - X 1,540 維：`data/prereg17/prep/probe_X.npy`（probe）、`data/prereg12/q09_inner/inner_X.npy`（inner），用 Q09 凍結 `results/prereg09/rank/p1e_map.npz` 的 `mu_x`、`sd_x` 以 `p09_forward.standardise_x` 標準化（＝S19 和 09 六個成員看到的輸入）。
  - EX 44 維：`data/prereg17/prep/probe_ex.npy`、`data/prereg12/q09_inner/inner_ex.npy`，用同一個 map 的 `mu_ex`、`sd_ex` 以 `p09_forward.standardise_ex` 標準化（＝09 `lowrank3_ex` 成員的程式路徑：float32 原值 → (ex − μ)/σ → `nan_to_num`），之後再把任何非有限值設成 0（11/5 盤點：兩個陣列沒有 inf，NaN 由 `nan_to_num` 變 0，這一步預期是空操作）。原始 EX 若出現 inf（`nan_to_num` 會把它變成 ±1.8e308 而不是 0），train／eval 直接停下，不自行處理。
  - **和交辦的差異**：交辦寫「42 個 EX」；實際登記的 EX 是 44 維（`p09_forward.NEX`＝44，`ex_names()`，`lowrank3_ex` 的輸入 1,584＝1,540＋44）。照實際的 44 維做。
- 等級、執色：同 prereg19（`probe_meta.npz` 的 `y`；`data/prereg08/p1|p2/sets.json`、`data/prereg09/p3/sets.json` 的 `color`，玩家鍵交叉核對）。

## 3. set 模型 SET2（寫死）

### 3.1 架構

- 每盤 h(x)：Dropout(0.1) → Linear(1584, 256) → GELU（erf 版）→ Dropout(0.3) → Linear(256, 64) → GELU，得到 64 維。
- 題目內聚合：v＝[ mean_j h_j , std_j h_j ]（128 維）。std 是**母體**標準差：先算平均 m，再 sqrt( mean_j (h_j − m)² )（兩遍算法）；只有 1 盤時 std＝0。torch 裡 sqrt 用「變異數 > 0 才開根號、否則 0」的寫法，避免變異數剛好為 0 時梯度變 NaN（數值上等於母體標準差）。
- head ρ(v)：Linear(128, 32) → GELU → Linear(32, 1)＝棋力分數 f。
- 順序（累積 link）輸出，同 S19：θ₁＝t0，θ_k＝t0＋Σ softplus(dt)（t0 起點 −4，dt 起點 log(e−1)），P(y ≤ k)＝σ(θ_k − f)，相鄰相減、下限 1e-12 再取 log。

### 3.2 抽題

- 同 prereg19 3.2（直接 import `results/prereg19/explore/p19_lib.py` 的 `PlayerPool`、`sample_batch`）：batch 128 題，batch 內盤數 n 相同，n 從 {5, 10, 15, 20} 均勻抽；等級均勻 → 執色均勻 → 合格玩家均勻 → 該玩家該執色不放回抽 n 盤。
- 亂數：`numpy.random.default_rng([23, 種子])`。

### 3.3 最佳化

- 同 prereg19：batch 內每題 NLL 平均；AdamW lr 1e-3、wd 1e-2；OneCycleLR（max_lr 1e-3、pct_start 0.1、15,000 步）；`torch.manual_seed(種子)` 後建模型；`torch.set_num_threads(8)`；輸入 float32。

### 3.4 停在哪一步

- 留出題組：H 玩家、同 3.2 規則，每級 × 每種盤數 100 題（共最多 4,000 題），`default_rng(23023)`，訓練前產生一次、所有種子共用；兩種執色都沒有合格 H 玩家的那一題略過（記 `level_missing`，同 prereg19 第 9 節）。
- 每 500 步在留出題組上算平均 NLL（dropout 關閉）；**最小的那一步的權重**就是這個種子的模型（同分取較早）。

### 3.5 種子和集成

- 5 個種子（0、1、2、3、4）。lp_SET2＝log( max( 5 個種子機率的平均, 1e-300 ) )（`p09_forward.lse_mean`）。

### 3.6 前向（評估用）

- inner 的 lp_SET2 一律 numpy float64 前向（權重存 float32、讀進來轉 float64；GELU 用 erf）。每盤先算 h，每題對該題的 **P1 可用盤**（`Inner.covered`，和 Q09 的 P1TO、S19 同一個遮罩）算平均和母體標準差，再算 ρ 和 lp。沒有可用盤的題：E_SET2＝0 向量；題數記錄。
- 移植核對（訓練時）：同一份權重，torch float32 和 numpy float64 在留出題組上的機率，每題最大差 ≤ 1e-4，否則停下查。

## 4. 基準 Q19、候選 Q19＋SET2 和交叉擬合

- **Q19（基準）**：用凍結的 `results/prereg19/rank/s19_seed{0,1,2}.npz`（SHA 須等於 `results/prereg19/explore/train_record.json` 的 `models_sha256`），以 prereg19 的 numpy 前向（import `p19_lib`）算 E_SET＝4·lp_SET，`Inner.run(Q08_TERMS + ["SET"], {"HS", "P1TO", "SET"})`，和 prereg19 判定時完全相同。
- **候選 Q19＋SET2（唯一一個）**：`Inner.run(Q08_TERMS + ["SET", "SET2"], {"HS": E_hs, "P1TO": E_P1TO_E, "SET": E_SET, "SET2": E_SET2})`，E_SET2＝**4**·lp_SET2。項的順序 OP、O4、SWA、SWB、HS、P1TO、SET、SET2；其餘（界限、起點、PENALTY、等級加權、規則 B、PlayerBootstrap）都不變（`p08_rank_fuse.run_terms`，不改）。
- 不會洩漏的理由同 prereg19 第 4 節：SET2 只用 probe 玩家（A、H）訓練和選步；融合參數照 Q09 的 inner 協定以玩家分 5 折 OOF。

## 5. 接縫（先過，否則停，不算任何候選）

1. Q02H、Q08、Q08 單色等於 `p09_rank_fuse.REF`；Q09（0.7114420614163589）、Q09 單色（0.7145431684624712）等於 `results/prereg09/rank/tune_report.json`；**Q19（0.7345843051712626）、Q19 單色（0.7430172362615272）等於 `results/prereg19/explore/eval_report.json` 的候選**；相差都 ≤ 1e-12。
2. **接縫 E**：本探索的「平均＋標準差」聚合程式的平均那一半，用在 4·lp_P1TO_E 上，擬合／評估／單色三組等於 `Inner.E_p1`（≤ 1e-12）。
3. **接縫 D**：A ∩ H＝∅；（A ∪ H）∩ I＝∅（以 `<rank>:<player_id>` 比）；只比 player_id 的交集只記錄。
4. **接縫 P**：`probe_X.npy`、`probe_ex.npy`、`inner_X.npy`、`inner_ex.npy` 的 SHA-256 等於 `results/prereg09/rank/check_report.json` 的 `prep.files_sha256`。
5. **接縫 M**：5 個 SET2 模型檔的 SHA 等於已 commit 的 `train_record.json`；訓練設定等於登記值；訓練時程式乾淨；移植核對過。
6. **接縫 F**：3 個凍結 S19 模型檔的 SHA 等於 prereg19 `train_record.json`。

## 6. 閘門（單一候選 Q19＋SET2，正式執行後只判一次；全部成立才算過）

- **G1**：inner 評估假題（127,200 題）的分等級平均差（Q19＋SET2 − Q19）≥ **+0.005**（嚴格，不加容許）。
- **G2**：同一差值的配對 PlayerBootstrap（`Inner.compare`：每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
- **G4**：單色假題（120,544 題）的分等級平均差，配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 ≥ 0（容許 1e-12）。
- 只有一個候選，不做多重比較校正。
- **只記錄、不能救候選**：
  - SET2 取代 SET：Q09＋SET2（`Q08_TERMS + ["SET2"]`）的 inner／單色分數，和 Q19、Q09 的差；
  - SET2 單獨（規則 B 用 softmax(lp_SET2)）、SET 單獨的 inner／單色分數；
  - 每個種子單獨的 Q19＋SET2_s（5 個）的差；
  - **依盤數**：單色假題 n＝5／10／15／20 各自的差，評估假題依 `rf.N_BINS` 和 n 正好是 5／10／15／20 的差（配對 PlayerBootstrap 95% 區間）；
  - **依等級**：評估、單色每一級的平均分數差；
  - 每折融合權重（w_P1TO、w_SET、w_SET2，有沒有打到界限）；
  - 訓練紀錄：每個種子留出 NLL 曲線、選到的步數、訓練秒數、抽題時改執色／重抽等級的次數、合格玩家數、移植核對的最大差、EX 非有限值個數。
- 不論判定結果，報告和 RESULTS.md 都寫出依盤數的差和 SET2 單獨的分數。

## 7. 順序、判定和之後

1. 本檔 commit（任何候選數字之前）。
2. 程式、合成資料的單元測試、smoke。**smoke**（輸出在 `data/prereg23/smoke/`，不是結果）：probe 和 inner 的 SET2 輸入（1,584 維）換成 `default_rng(12345)` 的常態雜訊，每個種子 300 步、每 100 步評一次留出、留出題組每級 × 每種盤數 5 題；train 和 eval 整條流程跑完（Q19 基準用真特徵和凍結 S19，只是重現已知數字，不是候選數字）。數字沒有意義、不記進結果。
3. **審查**：正式執行前做一輪對抗式自審（或派一個子代理獨立審查），修完、commit 才跑；結果寫進第 9 節。
4. **train 只跑一次**：`p23_train.py train`（`.venv`，CPU 8 執行緒，獨立背景程序；估計 80～100 分鐘），寫 `data/prereg23/models/`（5 個模型檔）和 `results/prereg23/explore/train_record.json`（commit）。train 不讀任何 inner 資料。開始時寫 `data/prereg23/train_started.json`；中斷可 `--resume`（同一 commit、同一程式），重跑只准在指出具體程式錯誤、更新的 commit 之後（`--rerun --reason`）。
5. **eval 只判一次**：`p23_eval.py eval`，寫 `results/prereg23/explore/eval_report.json`，原樣 commit。所有接縫過之後、算 SET2 項之前寫 `data/prereg23/eval_started.json`，之後中斷也算跑過；重跑規則同上。
- **沒過**：記錄後收掉，19 照舊。
- **過了**：只代表「值得寫 PREREG_23」，不是採用；由主 session 決定（11/24 前要完成登記、凍結、測試日工具、獨立審查才可能上傳）。本探索不做。

## 8. 檔案和環境

- `p23_lib.py`（純 numpy／scipy：玩家分割、特徵標準化、numpy 前向、平均＋標準差聚合、閘門；抽題和分數函式 import `p19_lib`）、`p23_train.py`（torch CPU）、`p23_eval.py`、`test_p23.py`、`train_record.json`、`eval_report.json`、`RESULTS.md`。大檔在 `data/prereg23/`（不進 git）。只 import、不改：`tools/*`、`results/prereg19/explore/p19_lib.py`、`results/prereg12/explore/finetune/ft_eval.py` 等。
- 環境：`.venv`（Python 3.12.13、torch 2.14.0+cpu），`PYTHONDONTWRITEBYTECODE=1`，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。worktree 在 session scratchpad，`subst Q:`（路徑含 `AppData\Local`），`data` 是指向主 repo `data` 的 junction（移除 worktree 前先拆）。worktree 先用 `core.autocrlf=false` checkout，但 `tools/p03_rank_fuse.py` 的 prereg02 輸入 SHA（`players_split.csv`）在 LF 版不符；所以把和主 repo 工作區位元組不同的 383 個已追蹤檔改成主 repo 的位元組（git 內容相同，只差換行），凍結檔的 SHA 檢查看到的和 main 一樣。其中 `results/prereg07/explore/pilot_humansl/` 9 個檔在 worktree 顯示已修改（只差換行），不 commit；本探索的 commit 只明確加 `results/prereg23/explore/` 的檔。

## 9. 實作補記（正式執行前）

閘門、第 3 節的架構／抽題／步數／種子／停步規則、乘 4、候選都沒改。

- 程式：`p23_lib.py`（抽題、留出題組、分數函式 import `p19_lib`）、`p23_train.py`、`p23_eval.py`、`test_p23.py`（合成資料 11 個測試全過：登記常數、特徵路徑等於 `standardise_x`／`standardise_ex`、平均＋母體標準差（1 盤為 0）、torch 和 numpy 前向一致（n＝1、5、20）、變異數為 0 時梯度有限、題目聚合等於逐題直接算、閘門）。
- 盤點（smoke 時讀到，不是結果）：A 5,138 位玩家（241,197 盤）、H 629 位（28,803 盤）；probe EX 原始 NaN 123,558 個、inf 0；inner EX NaN 33,928、inf 0；inner 擬合／評估／單色題沒有「0 盤可用」或「只有 1 盤可用」的題。
- smoke（`data/prereg23/smoke/`，SET2 的輸入換成雜訊、每個種子 300 步）：train 約 2 分鐘（每步約 0.05 秒，正式 15,000 步估計每個種子約 13～15 分鐘，5 個種子約 70～80 分鐘）；移植核對最大機率差約 1.5e-7。eval 約 5～8 分鐘（記憶體峰值約 5.7 GB）：接縫 1 七個數（Q02H、Q08、Q08 單色、Q09、Q09 單色、Q19、Q19 單色）逐位元相同，接縫 E 平均那一半差 2.8e-14，P、D（A∩H、probe∩I 都是 0）、M、F 全過。雜訊項的數字無意義，不記。
- 審查：派一個子代理獨立審查（只讀程式、跑單元測試，沒跑 train／eval）。沒有 blocker／major；minor 已在正式執行前修：
  1. 原始 EX 若有 inf，`nan_to_num` 會變成 ±1.8e308 而不是 0 → train／eval 遇到 inf 直接停（第 2 節已改寫；實際 0 個）。
  2. 接縫 M 沒綁「訓練時的前向程式＝評估時的」→ 加 `forward_code_same_as_train`：`p23_lib.py`、`p23_train.py`、`p19_lib.py`、`tools/p09_forward.py` 的 LF SHA 要等於 train 紀錄。
  3. 接縫 M 只核部分訓練設定 → 補齊 batch、ks、lr、wd、pct_start、hidden、h_dim、rho、drop、threads。
  4. 只檢查 loss 有限 → 每步更新後也檢查參數全有限。
  5. worktree 換行的實際做法和第 8 節原文不符 → 第 8 節改寫；commit 只加本資料夾的檔。
  6. 標準差那一半沒有真資料接縫 → 接縫 E 加一項：每組前 2,000 題的標準差那一半等於逐題 `np.std(ddof=0)`（≤ 1e-12）。
  7. （本代理）修完後重跑 smoke eval 時一次在「每個種子單獨」的融合中途記憶體不足（同時有另一個 session 的程序佔用約 19.5 GB）：每個種子和取代版的融合結果只留摘要、算完即釋放，報告加記憶體峰值；重跑 smoke 全程通過。正式 eval 若因記憶體不足中斷，屬於環境問題而不是程式錯誤，照第 7 節以 `--rerun --reason` 重跑（要另一個 commit，報告記原因）。
  - 沒改的：沒有加 once／resume 防護的單元測試（同 prereg19）；母體標準差隨盤數偏小（(n−1)/n）是登記的定義，訓練和 inner 的盤數都在 5～20，不改。
