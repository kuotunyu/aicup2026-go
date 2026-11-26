# prereg31 探索：把官方棋譜的 HD（讓子欄位）當成 set 模型的每盤輸入（S31），在 Q25 融合裡取代 SET 項（競賽二，inner-only）

2026-11-10 寫成，**在算出任何統計或模型數字之前 commit**（分支 `p31-explore`）。主 session 交辦（使用者 10/2、10/16 授權 AI 決定方法；11/10 同意「YOLOv4 式逐模組整合」的路線，並同意下載官方訓練資料 v1.1）。規則和門檻在執行前寫死，每一階段只判一次（紀錄檔原樣 commit）。這是探索：不寫 `docs/PREREG_31`、不做測試日工具、不上傳。不碰驗證集答案的資料夾（不開、不讀、不列、不雜湊，指令裡也不寫那個路徑）、不跑 `tools/score.py`。**硬截止：11/16 23:59 前合併**；趕不上就停下、記錄原因。11/11 起測試集 v2 的工作（PREREG_27）優先，本探索只在排程 session 同意的空檔跑。

## 0. 白話版

- 競賽二現在的最終是 25（Q25＝Q19 的 SET 項換成 S25）。到目前為止所有模型看的都是「棋怎麼下」，沒有任何一個用到官方 11/02 加進棋譜開頭的 **HD 欄位**（HD[0] 對等局、HD[1] 讓先、HD[2–9] 讓子）。測試檔每一盤也有 HD（只有 0 和 1），現行流程在 ingest 時把它拿掉（PREREG_22），所以這是一項還沒用過的官方資訊。
- 為什麼可能有用：HD[1] 表示執黑的一方比執白的一方弱大約一級。set 模型本來就同時看得到目標玩家和對手的 P1 特徵，多知道「這盤是讓先、目標執哪一色」，就多一條「目標和對手差一級」的線索；而且各等級下讓先棋的比例可能不同（最高、最低等級找不到同級對手時讓先棋會變多或變少）。競賽二測試題 HD[1] 佔 23.1%／22.5%（`results/test02/checks/root_props/prepare.json`），比官方 Q&A 說的整體 16% 高，也暗示比例會隨等級變。
- 做法：S25 的程式、資料、留出鹽、訓練設定**一字不改**，只在每盤的 1,540 維 P1 特徵後面多接 2 個數：「HD 旗標 × 目標執黑」「HD 旗標 × 目標執白」，訓練 10 個種子＝**S31**。Q31＝Q25 只把 SET 項換成 S31。
- 分三段，前一段沒過就停，不往下做：
  - **0a（只統計、不訓練）**：核對 v1.1 和 v1.0 除了 HD 以外逐位元組相同；統計 HD × 等級 × 執色；用寫死的規則決定 HD≥2 的棋怎麼編碼；確認測試題的 HD 比例落在訓練資料解釋得了的範圍內。
  - **0b（只看留出玩家的 NLL，不碰 inner）**：S31 的 10 個種子和「沒有 HD、同種子」的 10 個模型比留出 NLL。
  - **1（inner 判一次）**：Q31 對 Q29（同樣 10 個種子、沒有 HD）和對現任 Q25。
- **先講預期：不確定，偏保守。** HD[1] 只佔約 16–23% 的棋，而且對手強弱 set 模型已經看得到一部分。0a 的統計幾分鐘就能看出各等級的 HD 比例差多少；0b 沒有 −0.010 的 NLL 改善就不碰 inner。全部只用 CPU（約 2 小時訓練），不用 4090。

## 1. HD 資料和編碼

### 1.1 來源

- 官方訓練資料 v1.1：GitHub `AILAB-NDHU/aicup2026-go-dataset` release v1.1 的 `aicup2026_go_training_v1.1.tar.xz`（220,710,452 bytes，SHA-256 `e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e`）。下載到 `data/training_v1.1/`（新的空資料夾，不進 git），下載後先比 SHA-256，不同就停。
- **只取 HD 一欄**。`data/training`（v1.0）不換；P1、開局等所有既有特徵照舊來自 v1.0。PREREG_22 在 ingest 拿掉測試檔 HD 的流程不變（現行各版看到的格式不變）。
- 下載的檔是不受信任的資料：解壓用 `tarfile` 的 `filter="data"`；讀它的程式 `p31_hd.py` 只用標準函式庫和 numpy、不 import repo 內其他模組，用 `python -I` 執行；程式放在 `results/prereg31/explore/`，資料路徑一律用參數傳入。

### 1.2 HD 表（`p31_hd.py build`）

- 10 個等級檔各 100,000 列，主鍵 `(rank, game_id)`。每列：v1.1 的 `sgf_content` 開頭讀出 `HD[n]`；把 `HD[n]` 這個欄位拿掉之後必須和 v1.0 同一列的 `sgf_content` **逐位元組相同**，`player_id`、`game_id`、`rank`、`color` 也相同（接縫 A；有任何一列不同就停）。
- 另外算每盤的指紋 F（1.3 用）。輸出 `data/prereg31/hd_table.npz`（不進 git；SHA 記在紀錄裡）和 `results/prereg31/explore/hd_stats.json`（commit）。

### 1.3 HD≥2 的棋怎麼編碼：兩選一，由寫死的規則決定

- 背景：官方 Q&A 說訓練資料 HD[2–9] 約 3%，黑白標記相反、讓子石沒記錄；但測試檔只有 HD[0] 和 HD[1]。所以測試集裡 HD≥2 的棋不是被排除、就是被改標成 0 或 1。
- 兩種編碼（只差訓練和 inner 的 HD≥2 列）：
  - **Z**：旗標＝［HD＝1］。HD≥2 的列兩個輸入都是 0（和對等局一樣）。對應「測試集排除了 HD≥2」或「改標成 HD[0]」。
  - **C1**：旗標＝［HD≥1］。HD≥2 的列和 HD[1] 一樣處理（執色用該列記錄的 `color`）。對應「測試集把 HD≥2 改標成 HD[1]」。
- **指紋** F(盤)＝整盤沒有任何一手下在 `pd`，也沒有任何一手下在 `dp`（SGF 座標；讓 2 子以上時這兩個星位一開始就放了讓子石，而讓子石沒記錄，所以整盤都不會有人下在那裡）。
- 訓練資料（v1.1，100 萬列）：f0、f1、f2＝HD[0]、HD[1]、HD≥2 各組的 F 比例；π0、π1、π2＝各組佔比。測試（競賽二 public＋private 原始檔合計 10,000 盤，`data/test_rootprops/original/`，路徑用參數傳、SHA 要等於 `SOURCE.json`）：g0、g1＝HD[0]、HD[1] 各組的 F 比例。
- **健全性**：f2 ≥ 0.90 而且 f0 ≤ 0.50、f1 ≤ 0.50，否則指紋不成立 → 停下回報（要先修訂本檔才能繼續）。
- **規則**：s1＝π2 (f2 − f1) / (π1 + π2)（HD≥2 全部混進測試的 HD[1] 時，g1 預期升高的量）；D＝(g1 − g0) − (f1 − f0)（用差的差，抵銷測試和訓練等級組成不同的影響）。**D ≥ s1 / 2 → C1；否則 Z。**
- 只記錄：d1＝g1 − f1、d0＝g0 − f0、s0＝π2 (f2 − f0) / (π0 + π2)（HD≥2 混進 HD[0] 時的預期量；Z 已經涵蓋這個情形）、各等級的 f0／f1。
- 這條規則寫的時候已經知道測試檔的 HD 佔比和 Q&A 的比例表（第 7 節揭露），但還沒算過任何指紋統計。事前猜測是 Z（競賽一測試檔 HD[1] 佔 15.5–17.7%，和 Q&A 的 16% 相近）。

### 1.4 每盤的 2 個輸入

- x_B＝旗標 ×［目標執黑］、x_W＝旗標 ×［目標執白］（0／1）。目標執色＝訓練列的 `color`、測試題的 `color_k`。
- 標準化：用 A 角色訓練列（1.5 的 Rows 裡 A 玩家的 probe＋V 列）算 μ、σ（各 2 個數，σ 必須 > 0），存在每個 S31 模型檔和 `train_record.json`；留出題、inner、之後的測試題都用同一組 μ、σ。
- 不加單獨的執色輸入、不加 HD≥2 專用旗標、不加其他欄位。**只有這一個候選。**

### 1.5 S31 怎麼訓練（只差這 2 個輸入）

- 其他全部照 S25／S29，直接 import 不複製：資料列（`p25_train.Rows`：probe 270,000 列＋V 74,946 列；Q09 map 的 μ、σ 標準化前 1,540 維）、玩家角色（`p25_lib.holdout_bit`，鹽 `c10-prereg25|holdout|`，A 6,619／H 748 位）、留出題組（`p19_lib.holdout_set`，每級 × 每種盤數 100 題、`default_rng(19019)`）、訓練（`p19_train.train_seed`：AdamW lr 1e-3、wd 1e-2、OneCycle pct 0.1、15,000 步、batch 128、n ∈ {5,10,15,20}、每 500 步算留出 NLL、取最小那一步；8 個 torch 執行緒）、移植核對（`p19_train.port_check`）。
- 種子 0～9，規則同 S25／S29：`torch.manual_seed(s)`、抽題亂數 `np.random.default_rng([19, s])`。抽題只看玩家、等級、執色、盤數，和特徵無關，所以 S31 種子 s 和沒有 HD 的種子 s 抽到的題目序列相同（0b 的配對就靠這點）。
- 實作：輸入寬度 1,540 → 1,542，做法是在建模、`train_seed`、留出 NLL、移植核對這幾個呼叫的外面，用 context manager 暫時把 `p19_lib.D_IN` 設成 1542（`p19_train` 是在呼叫當下才讀 `K.D_IN`）；先呼叫 `Rows.standardised`（1,540 維）再接上 2 欄。**HD 關掉時不做任何替換**，走的就是 S25 的原路（接縫 R 驗證）。
- HD 值靠 `(rank, game_id)` 從 HD 表接到每一列：probe 列經 `probe_meta.npz` 的 `ref` → 三份 `sets.json` 的 `games`；V 列經 `v_meta.npz` 和 prereg25 的棋局清單；每一列都要接得到，而且 `player_id`、`color` 和該列原有的紀錄相同（接縫 A）。
- 模型檔 `data/prereg31/models/s31_seed{0..9}.npz`（不進 git；格式 `c10-prereg31 s31 model v1`，含 10 個參數陣列、`hd_mu`、`hd_sd`、編碼名稱），每個種子的紀錄 `data/prereg31/seed{s}_record.json`，總紀錄 `results/prereg31/explore/train_record.json`（commit）。train 不讀任何 inner 資料。

## 2. 第 0a 階段：統計和停損（不訓練）

`p31_hd.py stats` 一次算完，寫 `hd_stats.json`（原樣 commit）：

1. 接縫 A 的逐列比對結果（1,000,000 列）。
2. 表 N[等級][執色][HD 0／1／≥2] 和各等級、各執色的 HD 佔比。
3. 測試（競賽二兩池、依題目執色）HD[1] 佔比；競賽一題目和候選庫的佔比只記錄。
4. 1.3 的指紋統計和 **C1／Z 決定**。
5. **分布防護（要過，否則停，不訓練）**：在決定的編碼下，p(r, c)＝訓練資料等級 r、執色 c 的旗標比例；t(c)＝測試題（兩池合計）執色 c 的 HD[1] 比例。對黑、白都要 min_r p(r, c) − 0.03 ≤ t(c) ≤ max_r p(r, c) + 0.03。不成立表示測試的 HD 比例無法用任何等級組成解釋（訓練和測試的 HD 分布不同），HD 當輸入有風險 → NO-GO。
6. 只記錄、不設門檻：10 個等級等權重時的預期測試佔比 mean_r p(r, c)；只用 HD 的樸素貝氏（用 A 玩家估 p(r, c)，在 H 玩家的留出題上算 NLL，對照均勻分布）；每個等級相鄰兩級的 p 差。

## 3. 第 0b 階段：留出 NLL（不碰 inner）

- 先做接縫 R，再訓練 S31 種子 0～9。
- 對照組＝沒有 HD 的 10 個同種子模型已 commit 的留出 NLL：種子 0～2 取 prereg25 `train_record.json`，種子 3～9 取 prereg29 `train_record.json`（同一份程式、同一批留出題）。
- **GO 條件（兩項都要）**：① 10 個種子「最佳留出 NLL」的平均，S31 − 對照 ≤ **−0.010**；② 同種子的 10 對裡至少 8 對 S31 較低。
- 門檻的來由：種子之間留出 NLL 的標準差約 0.003–0.005，10 對平均的雜訊約 0.002；prereg24 的對照顯示留出 NLL −0.022 只對應 S 項分數 +0.0034，所以改善不到 0.010 就不可能撐起 inner 的 +0.005。這是「沒希望就不碰 inner」的門檻，不是採用門檻。
- 判定是已 commit 的 `train_record.json` 的純函式，寫 `stage0b.json`（commit）。NO-GO → 寫 RESULTS、收掉，不跑 eval。

## 4. 第 1 階段：候選、接縫、閘門（inner 判一次）

### 4.1 候選和融合

- **Q31**：`Inner.run(Q08_TERMS + ["SET"], {"HS", "P1TO": E_P1TO_E(Q09 map), "SET": 4·lp_S31})`，10 個 S31 模型照種子順序 lse 平均（`p19_lib.set_term`）。項的順序、界限、起點、PENALTY、等級加權、規則 B、5 折玩家 OOF 都不變。
- inner 每盤的 2 個 HD 輸入：`(rank, game_id)` 接 HD 表，用決定的編碼和訓練存下的 μ、σ（也就是照測試集會呈現的方式）。SET 項的計算用 `p23_eval.s19_term` 的同一個流程，只在標準化後、`h_np` 前接上 2 欄（本次 `p31_eval.py` 自己的版本；HD 關時必須和 `s19_term` 逐位元相同，見接縫 1）。
- **對照 Q29**（同樣 10 個種子、沒有 HD；prereg29 的 S29：凍結的 S25 三個檔＋`data/prereg29/models/s29_seed{3..9}.npz`）和**現任 Q25**。
- 不會洩漏：S31 只用 probe 玩家和驗證玩家訓練、選步，inner 玩家不在裡面（接縫 D）；HD 是每盤棋自己的欄位，不含等級標籤。

### 4.2 接縫（先過，否則停，不算候選）

1. **接縫 1**（全部**逐位元相同**）：Q02H、Q08、Q08 單色＝`p09_rank_fuse.REF`；Q09 0.7114420614163589、單色 0.7145431684624712；Q19 0.7345843051712626、單色 0.7430172362615272；Q25 0.7456519871546352、單色 0.7482450052195216；**Q29 0.7445162616450341、單色 0.7494986278839273**（prereg29 `eval_report.json` 的 candidate）。Q25、Q29 要用本次 eval 的 SET 項程式在「HD 關」下算出。
2. **接縫 F**：凍結 S19、S25 各三個檔 SHA＝各自的 `train_record.json`；S29 種子 3～9 的檔 SHA＝prereg29 `train_record.json`；Q09 map SHA＝09 `frozen_config.json`。
3. **接縫 P**：`probe_X.npy`、`inner_X.npy` 的 SHA＝09 `check_report.json`；`v_X.npy`、`v_meta.npz` 的 SHA＝prereg25 `prep_record.json`。
4. **接縫 D**：同 prereg29（A ∩ H＝∅；（A ∪ H）∩ inner＝∅；probe ∩ V＝∅；A、H 人數＝train 紀錄）。
5. **接縫 R（HD 關、重訓種子 0）**：`p31_train.py` 在 HD 關的模式下重訓種子 0，用 `p25_train.save_model` 存檔，SHA 必須**逐位元組等於** S25 種子 0（凍結檔和 prereg25 `train_record.json` 兩邊都比）。不相同就停，不訓練 S31。
6. **接縫 A（HD 表）**：v1.1 壓縮檔 SHA＝1.1；100 萬列逐列比對全過；HD 表 SHA＝`hd_stats.json`＝`train_record.json`；probe 270,000 列、V 74,946 列、inner 全部列都接得到 HD，`player_id`、`color` 相符。
7. **接縫 M**：10 個 S31 模型檔 SHA＝已 commit 的 `train_record.json`；種子、選步、smoke 旗標、`hd_mu`／`hd_sd`／編碼和紀錄相同；訓練設定＝登記值（和 prereg25 的 settings 逐項相同，只差 `seeds` 和輸入寬度）；訓練時程式乾淨；前向程式（`p19_lib.py`、`p19_train.py`、`p25_lib.py`、`p25_train.py`、`tools/p09_forward.py`、本次 `p31_lib.py`、`p31_train.py`）的 LF SHA 和訓練時相同；每個種子的移植核對都過；0a 的防護過、0b 是 GO。

### 4.3 閘門（單一候選 Q31；正式執行後只判一次；全部成立才算過）

定義同 prereg23／25／26／29（`p23_lib.gates`）：

- **G1**：inner 評估假題（127,200 題）的分等級平均差 **Q31 − Q29** ≥ **+0.005**。
- **G2**：同一差值的配對 PlayerBootstrap（每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
- **G4**：單色假題（120,544 題）的差（Q31 − Q29），配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 ≥ 0（容許 1e-12）。
- **G5（對現任）**：Q31 − **Q25** 的配對 PlayerBootstrap 95% 下界 > 0。
- 為什麼 G1／G2／G4 對 Q29 而不是 Q25：prereg29 顯示 Q25 的 3 個種子在 inner 上剛好偏好（Q29 − Q25＝−0.0011，區間 [−0.0041, +0.0020]），要量「HD 這個模組」的效果，對照必須是同樣 10 個種子、只差 HD 的 Q29；G5 再要求它確實贏過現在交出去的那一份。
- 只有一個候選，不做多重比較校正。
- **只記錄、不能救候選**：種子數曲線 k＝3／5／10（S31）；S31 單獨（規則 B）；依盤數、依等級、依執色；題內有／沒有 HD[1] 棋的題各自的差；不含 HD≥2 棋的題的差（測試集若排除 HD≥2，這一組最接近）；**把 S31 的 HD 輸入全部設成對等局的值再算一次**（看模型實際用了多少 HD）；每折融合權重；訓練紀錄。

## 5. 凍結、不能動的東西

- 不改：`results/prereg01–30/**`、`docs/PREREG_*`、`tools/*`（只 import）、測試日工具與指紋、驗證集 split v1、19／25 的模型和融合參數、`results/test*`、`data/training`（v1.0）。
- 新檔只在 `results/prereg31/explore/`（commit）和 `data/prereg31/`、`data/training_v1.1/`（不進 git）。
- 競賽一不受影響。

## 6. 順序、判定和檔案

1. 本檔 commit（任何數字之前）。
2. `p31_hd.py`＋單元測試 commit → 下載 v1.1、比 SHA、解壓 → `build`、`stats` → `hd_stats.json` commit。0a 沒過就停。
3. `p31_lib.py`、`p31_train.py`、`p31_eval.py`、`test_p31.py`（合成資料單元測試，含「HD 關時不替換任何東西」）、`run_train.sh`、`run_eval.sh` commit；smoke（特徵換成 `default_rng(12345)` 雜訊、每個種子 300 步、留出每格 5 題，輸出 `data/prereg31/smoke/`，數字不記）。
4. **訓練前另派一個代理獨立審查程式**（看程式和 smoke 的接縫紀錄，不看任何結果），修完 commit。
5. **train 只跑一次**（CPU 8 執行緒，獨立背景程序；先接縫 R 再種子 0～9，共 11 次訓練 × 約 9 分鐘＝約 1.7 小時；`--resume` 只用於中斷，同一 commit；重跑只准在指出具體程式錯誤、更新的 commit 之後 `--rerun --reason`）→ `train_record.json` commit → `stage0b.json` commit。
6. 0b 是 GO 才做：**eval 只判一次**（所有接縫過之後、算 S31 項之前寫 `data/prereg31/eval_started.json`；接縫沒過寫 `eval_seam_fail.json`，不佔用判定；閘門一算出先寫 `eval_judged.json`）→ `eval_report.json` 原樣 commit。
7. `RESULTS.md` commit，`--no-ff` 合併。

- **沒過**（任何一階段）：記錄後收掉，25 照舊。**過了**：只代表「值得寫 PREREG_31」，不是採用。要採用還要：寫登記、凍結、測試日工具（從**沒拿掉 HD 的官方原始測試檔**讀 HD，路徑用參數；競賽二測試檔若在 11/11 之後換新，用 1.3 同一條規則在新檔重算，結論和訓練時不同就停下回報、不自動換編碼）、獨立稽核，11/20 前凍結、11/21 前上傳；public 分數只能否決（比 25 低 0.010 以上就退回 25）。
- 時間界限：train 若 11/11 08:00 還沒完成就停下（`data/prereg31/STOP`），等排程 session 同意後 `--resume`；11/16 23:59 前沒合併就停、記錄原因。
- 共用電腦：每一個本機程序（下載、python、pytest）都要排程 session（戰情室）同意才開；全程不用 4090。別的工作需要 CPU 時，用**降低程序優先權、限制 CPU 親和性或暫停**（STOP＋`--resume`）讓出，不改 torch 執行緒數（8 是登記值，改了接縫 R 可能不過）。
- 環境：CPU `.venv`（Python 3.12.13、torch CPU）；`PYTHONDONTWRITEBYTECODE=1`，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。worktree 在 session scratchpad，`subst Q:`，`data` 是指向主 repo `data` 的 junction（移除 worktree 前先拆）；`core.autocrlf=false` checkout，再把和主 repo 工作區位元組不同的已追蹤檔換成主 repo 的位元組。本探索的 commit 只明確加 `results/prereg31/explore/` 的檔。

## 7. 揭露

- inner 已被 P1、08、09、prereg12、15–26、29 重複使用（選擇偏差約 0.003）。
- S31 和 S25 一樣用了驗證玩家的棋訓練；競賽二的 4 次驗證集打分已經用完。
- HD 是官方資料本身的欄位（訓練資料 v1.1、測試檔原始檔），不是外部資料。
- 這個點子是看過官方 11/02 的 v1.1 公告、Q&A 的讓子比例表、測試檔的 HD 佔比（競賽二 23.1%／22.5%，競賽一 15.5–17.7%）之後才提的；編碼規則（1.3）和分布防護（第 2 節第 5 項）寫的時候已知這些佔比，但還沒算過指紋、分等級佔比或任何模型數字。
- G1／G2／G4 的對照是 Q29 而不是現任 Q25，原因見 4.3；G5 對 Q25。

## 修訂 1（2026-11-10，算出任何統計之前）

1. 第 2 節第 6 項「只用 HD 的樸素貝氏」需要 A／H 角色和留出題組（`p25_lib.holdout_bit`、`p19_lib.holdout_set`），而 `p31_hd.py` 依 1.1 不能 import repo 模組。改成：`p31_hd.py stats` 算第 1–5 項和第 6 項的另外兩個數；樸素貝氏由 `p31_train.py hdnb` 在訓練前算一次，寫 `results/prereg31/explore/hd_nb.json`（commit；只記錄、不設門檻、不影響任何判定）。定義：P(旗標 | 等級 r, 執色 c) 用 A 角色列估計（加 1 平滑）；留出題（和 0b 同一批）的後驗 ∝ Π_k P(z_k | r, c_k)，等級先驗均勻；記錄平均 NLL，對照均勻分布的 ln 10。
2. 1.2 的「各 100,000 列」是預期值：工具記錄實際列數；v1.0 和 v1.1 每個檔的列數必須相同、`(rank, game_id)` 在全部列裡必須唯一，否則停。
3. 排程 session 11/10 的條件（記錄用，不改登記值）：可用 RAM < 4 GB 時用 STOP 停下再 `--resume`；別的工作回報變慢時，先降優先權、限制 CPU 親和性，仍不改 8 個 torch 執行緒。
