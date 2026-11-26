# prereg26 探索：09 的翻譯模型 P1TO_E 加上驗證玩家重訓（P1TO_E26），在 Q25 融合裡取代 P1TO_E（競賽二，inner-only）

2026-11-06 寫成，**在算出任何模型數字之前 commit**（分支 `p26-explore`）。主 session 交辦（使用者 10/2、10/16 授權 AI 決定方法）。閘門在正式執行前寫死，正式執行後只判一次（`eval_report.json` 原樣 commit）。這是探索：不寫 `docs/PREREG_26`、不做測試日工具、不上傳。不碰 `data/local`（不開、不讀、不列、不雜湊，指令裡也不寫這個路徑）、不跑 `tools/score.py`。

## 0. 白話版

- 現在競賽二的最終是 25（Q25＝Q19 把 set 模型 S19 換成加了驗證玩家重訓的 S25；public 0.7237578）。prereg24／25 顯示 set 模型「多一些不同的玩家」進步很多（+0.0111）。
- Q25 裡另一個吃 P1 特徵的項是 09 的翻譯模型 **P1TO_E**（6 個成員：mlpclm256 ×3、mlp512 ×2、lowrank3_ex），它只用 probe 的 270,000 盤（5,767 位 fit 玩家）訓練。prereg25 已經在 4090 算好 1,600 位驗證玩家 74,946 盤的 P1 塊（`data/prereg25/v/probe/`）。
- 這次：P1TO_E 的 6 個成員照 09 **一字不改**的設定和 epoch 規則，在 probe＋V（344,946 盤、7,367 位玩家）上重訓，得到 **P1TO_E26**；Q26＝Q25 只把 P1TO 項換成 P1TO_E26，用 prereg19／25 同一套 inner 5 折玩家 OOF 融合判一次。
- **先講預期：多半不會過。** 09 已經量過翻譯模型的資料曲線：P1＋P2（15 萬盤）→ 加 P3（27 萬盤）inner 只 +0.0001、單色反而 −0.0009（`results/prereg09/RESULTS.md`「已知事項」）。不同的是 P3 是同一批玩家的更多盤，這次是新玩家（玩家數 ×1.28）；但翻譯模型是逐盤模型、不是題目層級的模型，而且 Q25 裡 P1TO 的權重已經被 S25 壓到 0.25～0.49（Q19 是 0.77～4.0）。花費只有約 2.5 小時 CPU、不用 GPU，所以值得判一次。

## 1. 可不可行（唯讀盤點；沒有模型數字）

翻譯模型的訓練需要每盤的 X（1,540 維）、EX（44 維，只給 lowrank3_ex）、等級 y、玩家鍵（分 5 折用）。逐項看 V 有沒有：

| 需要 | probe（09 用的） | V（prereg25 抽的） |
|---|---|---|
| X | `data/prereg17/prep/probe_X.npy`（SHA＝09 `check_report.json`，已核對） | `data/prereg25/prep/v_X.npy`（74,946 列，0 盤不可用；`prep_record.json`） |
| EX | `probe_ex.npy`（SHA＝09 check_report，已核對） | **沒有現成的陣列**，但 V 的塊和 probe 的塊是同一支 `p1_extract.py`（一字不改）產生的同一種格式：每塊都有 `pos_game`、`pos_side`、`pos_turn`、`pos_logp`、`pos_entropy`（已打開 `chunk_00000.npz` 看過欄位名稱）。EX 就是 `tools/p09_map.read_chunk_rows` 從這 5 個陣列算的（`p09_forward.ex_raw32`），所以可以用同一個函式補算 |
| y、玩家鍵 | `probe_meta.npz`（`y`、`players`＝`<等級>:<player_id>`、`src`） | `v_meta.npz`（`y`、`players` 同格式、`ref`、`color`）；等級來自官方訓練 CSV 的檔名，和 split v1 檔核對過（prereg25 `audit.json`） |
| 和 inner 不重疊 | 09 manifest 核對過 | prereg25 盤點：驗證玩家 ∩ inner 玩家 0；驗證棋 vs inner 棋 8 種對稱重複 0 |

所以這個想法**不是不成立**：V 缺的只有 EX 陣列，而算 EX 要的東西都在塊裡。補算用 `read_chunk_rows`（不改），而且同一次讀出來的 X、ref、y、玩家鍵必須和 prereg25 的 `v_X.npy`、`v_meta.npz` 逐位元相同（接縫 V），才確定 EX 和 X 是同一盤、同一順序。

## 2. P1TO_E 影響哪些項（範圍）

Q25 的 7 項：OP、O4、SWA、SWB（04 的五項裡用到的四項）、HS（07）、P1TO（09 的 P1TO_E ×4）、SET（S25 ×4）。逐項看哪些用到 09 的 map 檔 `results/prereg09/rank/p1e_map.npz`：

| 項 | 用到 map 的什麼 | 這次怎麼處理 |
|---|---|---|
| OP、O4、SWA、SWB、HS | 都不用（04／07 的特徵和 C5 基底） | 不變 |
| **P1TO** | map 的全部：6 個成員權重、X 的 μ／σ、EX 的 μ／σ（`p09_rank_fuse.inner_map_lp` → `Inner.E_p1`，×4，P1 可用遮罩） | **換成 P1TO_E26**（新的成員權重和新的 μ／σ） |
| **SET**（S25） | 只用 map 的 **X 的 μ、σ** 標準化 X（`p23_eval.s19_term` → `FW.standardise_x(X, mp["mu_x"], mp["sd_x"])`）；S25 就是用 Q09 的 μ、σ 訓練的 | **不變：繼續用 Q09 map 的 μ、σ**（S25 是凍結模型，換 μ、σ 等於換了輸入分布）。也就是 Q26 的 SET 項和 Q25 逐位元相同 |
| P1 可用遮罩 | 不用 map（`Inner.covered` 來自 inner prep 的列，`has` 來自同一個遮罩） | 不變 |

- **EX 不是融合項**，只是 lowrank3_ex 成員的輸入；它的 μ、σ 存在 map 裡，跟著 P1TO_E26 一起換。
- 結論：**Q26 和 Q25 只差 `Emap["P1TO"]` 一項**；融合參數照 5 折玩家 OOF 重新擬合（每折各自擬合，同 prereg19／25）。
- 測試日的含意（本探索不做）：過了的話，t25 的路徑要多一份 map26 只給 P1TO 項；SET 項的標準化仍要讀 Q09 的 map。兩份 map 不能搞混。

## 3. 設計：P1TO_E26（照 PREREG_09 第 1.3 節，只差訓練列）

- **訓練列**：probe 的 270,000 列（P1＋P2＋P3，09 凍結 map 用的同一份，順序不變）接著 V 的 74,946 列（`v_meta.npz` 的順序）＝ **344,946 列、7,367 位玩家**（5,767＋1,600，鍵和 player_id 都不重疊，prereg25 已核對，這裡再核對一次）。
- **標準化**：PREREG_09 1.3 第 1 點「用實際用到的 probe 可用盤標準化」→ μ、σ（X 和 EX）用全部 344,946 列算（`p09_rank_fuse.Prep.moments` 原函式：`nan_moments` 兩趟串流、`fix_x_moments`／`fix_ex_moments`）。
- **成員、架構、超參數、種子、epoch 規則**：全部 import `tools/p09_map.py`、`tools/p09_forward.py` 不改：`fit_member`（以玩家分 5 折 CV，`player_folds` seed 0，折用 7,367 位玩家重算；MLP 每折跑滿 24 個 epoch、以所有 OOF 的平均 log-loss 選 e*；再用全部列重訓、排程 24 跑到 e* 停；lowrank3_ex 固定 30）、`model_params`、`Rows`（float32 輸入）、`set_threads(8)`。6 個成員的順序和種子照 `FW.MEMBERS`。
- **集成和前向**：`FW.save_map` 存成 `p1e_map` 同一格式（`data/prereg26/map/p1e26_map.npz`，不進 git），numpy float64 前向（`FW.map_logp`），等權 lse 平均，×4。
- **移植核對**：`p09_map.torch_numpy_max_diff`（float64 torch 對 numpy，集成 lp 最大差 ≤ **1e-4**，PREREG_09 rev 2 第 1 點的判法）在訓練列的前 8,192 列 probe＋前 8,192 列 V 上做；沒過就停。
- **不讀 inner**：train 只開 probe 和 V 的陣列，不開任何 inner 特徵或標籤。
- 環境：`.venv`（Python 3.12.13、torch 2.14.0+cpu、numpy 2.5.3、scipy 1.18.1；和 09 登記相同，不同就拒絕），8 個 torch 執行緒，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。CPU torch 不保證逐位元重現，凍結的是權重。時間：09 在 270,000 列上 6,923 秒，這次約 ×1.28 ≈ 2.5 小時；獨立背景程序（`Start-Process` 跑 `run_train.sh`），每個成員做完就存檔，`--resume` 只保留同一 commit、同一程式做完的成員。記憶體：特徵陣列用 memmap 讀（約 2.2 GB），OOF 陣列約 0.33 GB；開跑前看可用記憶體。

## 4. 候選和融合（唯一一個候選）

- **基準 Q25**：`Inner.run(Q08_TERMS + ["SET"], {"HS", "P1TO": E_P1TO_E(Q09 map), "SET": 4·lp_S25})`，S25 讀已 commit 的凍結檔 `results/prereg25/rank/s25_seed{0,1,2}.npz`（SHA＝prereg25 `train_record.json`），SET 項用 `p23_eval.s19_term`（Q09 map 的 μ、σ），和 prereg25 判定時同一段程式。
- **候選 Q26**：同一個呼叫，只把 P1TO 換成 `Inner.E_p1(inner_map_lp(prep, map26))`。項的順序、界限、起點、PENALTY、等級加權、規則 B、5 折玩家 OOF 都不變。
- 不會洩漏：P1TO_E26 只用 probe 玩家和驗證玩家訓練（CV 也只在這些玩家內分折），inner 玩家從來不在裡面（接縫 D）；融合照 Q09 的 inner 協定以玩家分 5 折 OOF。

## 5. 接縫（先過，否則停，不算候選）

1. **接縫 1**：Q02H、Q08、Q08 單色等於 `p09_rank_fuse.REF`；Q09 0.7114420614163589、單色 0.7145431684624712；Q19 0.7345843051712626、單色 0.7430172362615272；**Q25 0.7456519871546352、單色 0.7482450052195216**（prereg25 `eval_report.json` 的 candidate）；相差 ≤ 1e-12（實際要逐位元相同，prereg23／25 已經做到）。
2. **接縫 F**：凍結 S19 三個模型檔 SHA＝prereg19 `train_record.json`；凍結 S25 三個模型檔 SHA＝prereg25 `train_record.json`；Q09 map SHA＝09 `frozen_config.json`。
3. **接縫 P**：`probe_X`、`probe_ex`、`probe_meta`、`inner_X`、`inner_ex` 的 SHA＝09 `check_report.json`；`v_X.npy`、`v_meta.npz` 的 SHA＝prereg25 `prep_record.json`；`v_ex.npy` 的 SHA＝本次 `v_ex_record.json`，它的塊 SHA 清單＝prereg25 `prep_record.json` 的 `chunk_sha256`。
4. **接縫 V**（補算 EX 時）：`read_chunk_rows` 重讀 375 塊得到的 X、ref、y、玩家鍵和 `v_X.npy`、`v_meta.npz` 逐位元相同；EX 沒有 ±inf。
5. **接縫 T**（訓練開始前，在 train 裡）：只用 probe 270,000 列，以同一段程式算的 μ、σ（X、EX 共 4 個向量）和 Q09 map 存的**逐位元相同**；只用 probe 玩家算的 5 折大小等於 09 `train_report.json` 的 `folds`（53,295／54,353／54,409／53,073／54,870）。證明「拼接列＋原函式」的路徑在 probe 子集上就是 09 本身。
6. **接縫 D**：訓練玩家（probe ∪ V）∩ inner 玩家＝∅（以 `<rank>:<player_id>` 和以 player_id 各比一次）；probe ∩ V＝∅；V 棋 vs inner 棋 8 種對稱重複＝0（prereg25 `audit.json`）。
7. **接縫 M**：map26 的 SHA＝已 commit 的 `train_record.json`；6 個成員的名稱、種類、種子照 `FW.MEMBERS`；訓練設定（`FW.CONFIGS`、`BATCH`、`FOLDS`、執行緒）等於登記值；訓練時程式乾淨；`tools/p09_forward.py`、`tools/p09_map.py` 的 LF SHA 和訓練時相同；移植核對 ≤ 1e-4 過。

## 6. 閘門（單一候選 Q26，和 Q25 比；正式執行後只判一次；全部成立才算過）

定義同 prereg23／25（`p23_lib.gates`）：

- **G1**：inner 評估假題（127,200 題）的分等級平均差（Q26 − Q25）≥ **+0.005**。
- **G2**：同一差值的配對 PlayerBootstrap（`Inner.compare`：每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
- **G4**：單色假題（120,544 題）的分等級平均差，配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 ≥ 0（容許 1e-12）。
- 只有一個候選，不做多重比較校正。
- **只記錄、不能救候選**：
  - Q09 層級的換法：Q08_TERMS 裡 P1TO_E → P1TO_E26（對 Q09），看翻譯模型本身多了多少（不含 SET 的重疊）。
  - Q25＋P1TO_E26 當第 8 項（`P1TO2`）。
  - 依盤數（單色 n＝5／10／15／20、評估假題 `rf.N_BINS` 和 n 正好 5／10／15／20，配對區間）；依等級；每折融合權重（P1TO、SET）。
  - 訓練紀錄：每個成員的 CV log-loss、top1、e*、秒數（和 09 的 `train_report.json` 並列）；移植核對；新舊 μ、σ 的差；inner 上 lp_P1TO_E26 和 lp_P1TO_E 的平均絕對差。
- 不論結果，RESULTS.md 都寫出依盤數、依等級的差。

## 7. 凍結、不能動的東西

- 不改：`results/prereg01–25/**`、`docs/PREREG_*`、`tools/*`（只 import）、測試日工具與指紋、驗證集 split v1、09 的 `p1e_map.npz`、19／25 的模型和融合參數、`results/test*`。
- 本探索的新檔只在 `results/prereg26/explore/`（commit）和 `data/prereg26/`（不進 git）。
- 競賽一不受影響。

## 8. 順序、判定和檔案

1. 本檔 commit（任何模型數字之前）。
2. `p26_lib.py`（路徑防護、拼接列、`loaded_repo_modules`）、`p26_prep.py`（補算 V 的 EX → `data/prereg26/prep/v_ex.npy`＋`results/prereg26/explore/v_ex_record.json`，接縫 V）、`p26_train.py`、`p26_eval.py`、`test_p26.py`（合成資料單元測試）、`run_train.sh`、`run_eval.sh` commit；smoke（特徵換成 `default_rng(12345)` 雜訊、只用 probe 前 6,000 列＋V 前 2,000 列、每個成員的排程縮成 2 個 epoch，輸出 `data/prereg26/smoke/`，數字不記）。
3. **訓練前另派一個代理獨立審查程式**（看程式、不看任何結果），修完 commit。
4. **train 只跑一次**（`--resume` 只用於中斷，同一 commit、同一程式；重跑只准在指出具體程式錯誤、更新的 commit 之後 `--rerun --reason`）→ `train_record.json` commit。
5. **eval 只判一次**：所有接縫過之後、算 P1TO_E26 項之前寫 `data/prereg26/eval_started.json`，之後中斷也算跑過 → `eval_report.json` 原樣 commit → `RESULTS.md` commit。
- **沒過**：記錄後收掉，25 照舊。**過了**：只代表「值得寫 PREREG_26」，不是採用。
- 環境：CPU `.venv`；`PYTHONDONTWRITEBYTECODE=1`。worktree 在 session scratchpad，`subst Q:`，`data` 是指向主 repo `data` 的 junction（移除 worktree 前先拆）；worktree 用 `core.autocrlf=false` checkout，再把和主 repo 工作區位元組不同的已追蹤檔換成主 repo 的位元組（431 個，git 內容相同），用 `git add` 重新整理這些檔的索引狀態（blob 不變、沒有暫存任何內容差異）；剩 9 個 `results/prereg07/explore/pilot_humansl/` 只差換行的檔顯示已修改，不 commit。本探索的 commit 只明確加 `results/prereg26/explore/` 的檔。

## 9. 揭露

- inner 已被 P1、08、09、prereg12、15–25 重複使用（選擇偏差約 0.003）。
- 用驗證玩家的棋訓練翻譯模型：競賽二的 4 次驗證集打分已經用完；25 已經用了驗證玩家，這次再用一次不改變「驗證集不再獨立」的狀態。
- 翻譯模型的成員、架構、epoch 規則是 09 看過 inner 才挑的（09 第 8 節）；這次一個都不改。

## 10. 實作補記（正式訓練和判定之前；閘門、候選、P1TO_E26 的設定都沒改）

- **接縫 V 過**（`v_ex_record.json`，`e27cf5e`）：375 塊重讀，X、ref、y、玩家鍵和 prereg25 的 `v_X.npy`／`v_meta.npz` 逐位元相同，塊 SHA 清單相同，EX 沒有 inf；74,946 列，其中 8,249 列 EX 有 NaN（對手方沒有手等，標準化時照 09 設成 0；比例和 probe 相近）。38 秒。
- smoke（`data/prereg26/smoke/`，特徵換成雜訊、probe 前 6,000 列＋V 前 2,000 列、排程 2 個 epoch）：train 約 35 秒，**接縫 T 過**（只用 probe 270,000 列算的 μ、σ 4 個向量和 Q09 map 逐位元相同；5 折大小 53,295／54,353／54,409／53,073／54,870 和 09 相同）；移植核對 2.2e-15。eval 約 2 分鐘，**接縫 1 的 9 個數逐位元相同（含 Q25 0.7456519871546352、單色 0.7482450052195216）**，P、D（訓練玩家 7,367、inner 636、交集 0）、M、F 全過。雜訊 map 的數字無意義，不記。
- 程式修正（smoke 時發現）：`p25_lib.loaded_repo_modules` 會把 torch 的 `torch._classes`／`torch._ops`（`__file__` 是相對路徑 `_classes.py`、`_ops.py`）算成 repo 根目錄的檔，`code_state` 因此找不到檔而停。`p26_lib.loaded_repo_modules` 只收絕對路徑（p25_lib 是凍結檔，不改）。
- **獨立審查**（另一個代理，只看程式和 smoke 的接縫紀錄，沒有看任何結果）：沒有 blocker；6 個 minor 已處理 5 個：① eval 的接縫 M 另外核對訓練列數（344,946／270,000／74,946／7,367 位）、每個成員的排程等於登記值且 1 ≤ e* ≤ 排程（lowrank e*＝30）、map 的 e* 等於訓練紀錄；② 訓練做完後拒絕 `--resume`；③ train／eval 的 `code_state` 加上 `p26_prep.py`（eval 也加 `run_*.sh`）；④ 接縫沒過時寫 `data/prereg26/eval_seam_fail.json`，不佔用「只判一次」的 `eval_report.json`；⑤ train 開跑前先讀 prereg25 `audit.json`，驗證玩家 ∩ inner 和驗證棋 vs inner 棋 8 種對稱重複都要是 0（train 仍不開任何 inner 檔）。沒改的 1 個：`run_train.sh` 依賴 `Q:`（同一個登入工作階段啟動，可見）。
