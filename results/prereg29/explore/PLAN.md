# prereg29 探索：S25 從 3 個種子加到 10 個種子（S29），在 Q25 融合裡取代 S25（競賽二，inner-only）

2026-11-08 夜寫成，**在算出任何模型數字之前 commit**（分支 `p29-explore`）。主 session 交辦（使用者 10/2、10/16 授權 AI 決定方法）。閘門在正式執行前寫死，正式執行後只判一次（`eval_report.json` 原樣 commit）。這是探索：不寫 `docs/PREREG_29`、不做測試日工具、不上傳。不碰驗證集答案的資料夾（不開、不讀、不列、不雜湊，指令裡也不寫那個路徑）、不跑 `tools/score.py`。**硬截止：11/10 23:59 前合併**；趕不上就停下、記錄原因，11/11 測試集 v2 那天不受打擾。

## 0. 白話版

- 競賽二現在的最終是 25（Q25＝Q19 的 SET 項換成 S25；S25 是 3 個種子的 set 模型，lse 平均）。
- prereg25 的紀錄顯示種子之間差很多：單一種子換進 Q19 各只多 +0.0071／+0.0058／+0.0066，3 個種子的集成多 +0.0111。prereg23 的 SET2 也一樣（5 個種子的集成比任一個種子都好）。這表示 set 模型的**變異（雜訊）還很大**，多平均幾個種子可能再降一些。
- 這次：S25 的 3 個模型檔**原樣沿用**（種子 0、1、2），用**一字不改**的程式、設定、資料、留出鹽再訓練 7 個新種子（3～9），10 個一起 lse 平均＝**S29**；Q29＝Q25 只把 SET 項換成 S29，用 prereg19／25 同一套 inner 5 折玩家 OOF 融合判一次。
- **先講預期：多半不會過 G1。** 單一種子 → 3 種子多了約 +0.0045（+0.0111 對單種子平均 +0.0065）；集成的變異大約照 1/k 縮小，3 → 10 種子剩下的可降空間大約只有 1 → 3 的三分之一左右，照這樣外推約 +0.001～+0.002，離 +0.005 很遠。花費只有約 1.5～2 小時 CPU、不用 GPU，而且只要 CPU 幾分鐘就能在測試日套用，所以值得判一次。

## 1. S29 怎麼做（只差種子數）

- **種子 0、1、2**：`results/prereg25/rank/s25_seed{0,1,2}.npz`（已 commit 的凍結檔，SHA＝prereg25 `train_record.json` 的 `models_sha256`），逐位元組沿用，不重訓。
- **種子 3～9（7 個新種子）**：照 S25 的種子規則延伸——S25 的種子 s 用 `torch.manual_seed(s)`（`p19_train.torch_model`）和抽題亂數 `np.random.default_rng([19, s])`（`p19_lib.SAMPLER_BASE`＝19），新種子就是 s＝3、4、…、9，同一個規則。
- **其他全部照 S25，直接 import 不複製**：資料列（`p25_train.Rows`：probe 270,000 列＋V 74,946 列；Q09 map 的 μ、σ 標準化）、玩家角色（`p25_lib.holdout_bit`，鹽 `c10-prereg25|holdout|`，A 6,619／H 748 位）、留出題組（`p19_lib.holdout_set`，每級 × 每種盤數 100 題、`default_rng(19019)`）、模型和訓練（`p19_train.train_seed`：AdamW lr 1e-3、wd 1e-2、OneCycle pct 0.1、15,000 步、batch 128、n ∈ {5,10,15,20}、每 500 步算留出 NLL、取最小那一步；8 個 torch 執行緒）、移植核對（`p19_train.port_check`）、存檔（`p25_train.save_model`，格式字串照 S25 的 `c10-prereg25 s25 model v1`，這樣重訓的種子 0 才能和 S25 逐位元組比）。
- 新種子存 `data/prereg29/models/s29_seed{3..9}.npz`（不進 git），每個種子的紀錄 `data/prereg29/seed{s}_record.json`，總紀錄 `results/prereg29/explore/train_record.json`（commit）。train 不讀任何 inner 資料。
- **集成**：10 個模型照種子順序（0～9）傳給 `p23_eval.s19_term`（prereg25 判 Q25 的同一個函式：每盤 h → 題內可用盤平均 → head → `p19_lib.set_term` 的 lse 平均，×4）。

## 2. 候選和融合（唯一一個候選）

- **基準 Q25**：`Inner.run(Q08_TERMS + ["SET"], {"HS", "P1TO": E_P1TO_E(Q09 map), "SET": 4·lp_S25})`，S25＝凍結的 3 個檔，和 prereg25／26 判定時同一段程式（`p23_eval.s19_term`）。
- **候選 Q29**：同一個呼叫，SET 項換成 4·lp_S29。項的順序、界限、起點、PENALTY、等級加權、規則 B、5 折玩家 OOF 都不變（每折各自重新擬合融合，同 prereg19／25）。
- 不會洩漏：S29 的 10 個模型都只用 probe 玩家和驗證玩家訓練、選步，inner 玩家從來不在裡面（接縫 D）。

## 3. 接縫（先過，否則停，不算候選）

1. **接縫 1**（全部要**逐位元相同**）：Q02H、Q08、Q08 單色＝`p09_rank_fuse.REF`；Q09 0.7114420614163589、單色 0.7145431684624712；Q19 0.7345843051712626、單色 0.7430172362615272；**Q25 0.7456519871546352、單色 0.7482450052195216**（prereg25 `eval_report.json` 的 candidate）。
2. **接縫 F**：凍結 S19 三個檔 SHA＝prereg19 `train_record.json`；凍結 S25 三個檔 SHA＝prereg25 `train_record.json`；Q09 map SHA＝09 `frozen_config.json`。
3. **接縫 P**：`probe_X.npy`、`inner_X.npy` 的 SHA＝09 `check_report.json`；`v_X.npy` 的 SHA＝prereg25 `prep_record.json`＝本次 `train_record.json`。
4. **接縫 D**：A ∩ H＝∅；（A ∪ H）∩ inner＝∅（以 `<rank>:<player_id>` 和以 player_id 各比一次）；probe ∩ V＝∅；V 棋 vs inner 棋 8 種對稱重複＝0（prereg25 `audit.json`）；A、H 人數＝train 紀錄。
5. **接縫 R（重訓種子 0）**：train 先用本次程式重訓種子 0（存 `data/prereg29/models/s29_seed0_retrain.npz`），模型檔 SHA 必須**逐位元組等於** S25 種子 0（`4a01d6a6…`，凍結檔和 prereg25 `train_record.json` 兩邊都比）。不相同就停：不訓練新種子（表示環境或程式和 S25 不同，新種子就不是「同一個程式」）。eval 再重新雜湊一次。
6. **接縫 M**：7 個新模型檔 SHA＝已 commit 的 `train_record.json`；每個檔的種子、選步、smoke 旗標和紀錄相同；訓練設定＝第 1 節的登記值（和 prereg25 `train_record.json` 的 settings 逐項相同，只差 `seeds`）；訓練時程式乾淨；前向程式（`p19_lib.py`、`p19_train.py`、`p25_train.py`、`tools/p09_forward.py`、本次 `p29_train.py`）的 LF SHA 和訓練時相同；每個種子的移植核對都過；接縫 R 過。

## 4. 閘門（單一候選 Q29，和 Q25 比；正式執行後只判一次；全部成立才算過）

定義同 prereg23／25／26（`p23_lib.gates`）：

- **G1**：inner 評估假題（127,200 題）的分等級平均差（Q29 − Q25）≥ **+0.005**。
- **G2**：同一差值的配對 PlayerBootstrap（`Inner.compare`：每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
- **G4**：單色假題（120,544 題）的分等級平均差，配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 ≥ 0（容許 1e-12）。
- 只有一個候選，不做多重比較校正。
- **只記錄、不能救候選**：
  - **種子數曲線 k＝3／5／10**：k 個種子＝種子 0～k−1 的 lse 平均放進同一個融合。k＝3 就是 Q25 本身（S29 的前 3 個就是 S25 的 3 個檔），另外用 S29 的前 3 個模型重算一次核對和 Q25 逐位元相同；k＝5 對 Q25 的配對差；k＝10 就是 Q29。
  - 新種子 3～9 各自單獨的 Q25_s（對 Q25 的差；種子 0～2 單獨的數字在 prereg25 紀錄裡）。
  - S29 單獨（規則 B，不融合）；依盤數（單色 n＝5／10／15／20、評估假題 `rf.N_BINS` 和 n 正好 5／10／15／20）；依等級；每折融合權重（P1TO、SET）；訓練紀錄（每個種子的選步、留出 NLL、秒數、移植核對）。
- 不論結果，RESULTS.md 都寫出 k 曲線、依盤數、依等級的差。

## 5. 凍結、不能動的東西

- 不改：`results/prereg01–28/**`、`docs/PREREG_*`、`tools/*`（只 import）、測試日工具與指紋、驗證集 split v1、19／25 的模型和融合參數、`results/test*`。
- 本探索的新檔只在 `results/prereg29/explore/`（commit）和 `data/prereg29/`（不進 git）。
- 競賽一不受影響。

## 6. 順序、判定和檔案

1. 本檔 commit（任何模型數字之前）。
2. `p29_lib.py`、`p29_train.py`、`p29_eval.py`、`test_p29.py`（合成資料單元測試）、`run_train.sh`、`run_eval.sh` commit；smoke（特徵換成 `default_rng(12345)` 雜訊、每個種子 300 步、留出每格 5 題，輸出 `data/prereg29/smoke/`，數字不記）。
3. **訓練前另派一個代理獨立審查程式**（看程式和 smoke 的接縫紀錄，不看任何結果），修完 commit。
4. **train 只跑一次**（CPU 8 執行緒，獨立背景程序；`--resume` 只用於中斷，同一 commit、同一程式；重跑只准在指出具體程式錯誤、更新的 commit 之後 `--rerun --reason`）→ `train_record.json` commit。預估 8 個種子（重訓 0＋新 3～9）× 約 9～15 分鐘＝約 1.2～2 小時；開跑前看可用記憶體（約 3～4 GB 峰值）。
5. **eval 只判一次**：所有接縫過之後、算 S29 項之前寫 `data/prereg29/eval_started.json`，之後中斷也算跑過；接縫沒過寫 `data/prereg29/eval_seam_fail.json`（不佔用只判一次的報告）→ `eval_report.json` 原樣 commit → `RESULTS.md` commit。
- **沒過**：記錄後收掉，25 照舊。**過了**：只代表「值得寫 PREREG_29」，不是採用；要不要採用由主 session 決定，而且 11/11 起是測試集 v2，任何新交件都要排在 PREREG_27 的 v2 流程之後。
- 時間界限：11/10 23:59 前要 `--no-ff` 合併；train 若 11/10 18:00 還沒完成就停下（`data/prereg29/STOP`），記錄原因、不判定。
- 環境：CPU `.venv`（Python 3.12.13、torch CPU）；`PYTHONDONTWRITEBYTECODE=1`，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。worktree 在 session scratchpad，`subst Q:`，`data` 是指向主 repo `data` 的 junction（移除 worktree 前先拆）；worktree 用 `core.autocrlf=false` checkout，再把和主 repo 工作區位元組不同的已追蹤檔換成主 repo 的位元組（460 個，git 內容相同、索引已重新整理），剩 9 個 `results/prereg07/explore/pilot_humansl/` 只差換行的檔顯示已修改，不 commit。本探索的 commit 只明確加 `results/prereg29/explore/` 的檔。

## 7. 揭露

- inner 已被 P1、08、09、prereg12、15–26 重複使用（選擇偏差約 0.003）。
- S29 和 S25 一樣用了驗證玩家的棋訓練；競賽二的 4 次驗證集打分已經用完。
- 這個點子是看過 prereg25 的每種子分數（inner）之後才提的，報告要寫明。

## 8. 實作補記（正式訓練和判定之前；閘門、候選、S29 設定都沒改）

- smoke（`data/prereg29/smoke/`，特徵換成雜訊、每個種子 300 步）：train 約 3 分鐘（8 個種子，移植核對全過；接縫 R 在 smoke 不適用）；eval 約 4.5 分鐘、記憶體峰值約 3.8 GB；**接縫 1 的 9 個數逐位元相同（含 Q25 0.7456519871546352、單色 0.7482450052195216）**，P、D（A 6,619、H 748、inner 636、交集 0）、M、F 全過；k＝3 的子集和 Q25 逐位元相同。雜訊項的數字無意義，不記。
- **獨立審查**（另一個代理，只看程式和 smoke 的接縫紀錄，沒看任何結果）：沒有 blocker、沒有 major；5 個 minor 都已處理：① 每個種子紀錄加 commit `head`，`--resume` 只保留同一 commit 的種子；② train 的 code_state 加上凍結的 S25 三個模型檔；③ eval 的 FORWARD_FILES 加 `p25_lib.py`（A／H 的鹽）；④ train 和 eval 的接縫 P 另外核對 `v_meta.npz` 的 SHA＝prereg25 `prep_record.json`；⑤ 閘門一算出就先寫 `data/prereg29/eval_judged.json`，再做只記錄的部分（中斷時判定結果不會遺失）。
- 審查提醒（寫明即可）：k＝3 的重算用的是同一組 S25 模型物件，只驗證重算穩定，不是獨立重現。
