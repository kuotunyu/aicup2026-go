# prereg25 探索：S19 加上驗證玩家重訓（S25），在 Q19 融合裡取代 S19（競賽二，inner-only）

2026-11-06 凌晨寫成，**在算出任何模型數字之前 commit**（分支 `p25-explore`）。主 session 交辦（使用者 10/2、10/16 授權 AI 決定方法）。閘門在正式執行前寫死，正式執行後只判一次（`eval_report.json` 原樣 commit）。這是探索：不寫 `docs/PREREG_25`、不做測試日工具、不上傳。不碰 `data/local`（不開、不讀、不列、不雜湊，指令裡也不寫這個路徑）、不跑 `tools/score.py`。

## 0. 白話版

- prereg24 的小實驗發現：S19（19 的 set 模型）**多一倍玩家**時進步約 +0.0101，**每人多一些盤**只進步 +0.0034。fit 玩家（5,767 位）已經全部在 probe 裡，沒有新玩家可加。
- 還沒用過的玩家只剩 split v1 的 **1,600 位驗證玩家**。競賽二的 4 次驗證集打分已經全部用完，驗證集對競賽二已經沒有別的用處；這些玩家的棋和等級就在官方訓練 CSV 裡（驗證題是從他們的棋做出來的）。
- 這次：把驗證玩家的棋照 probe 的抽法（每級輪流各抽一盤、分三批）抽約 75,000 盤，在 4090 算 P1 特徵（約 3.7 小時），和原本 270,000 盤 probe 一起訓練 **S25**（架構、設定、種子全部照 S19，只多了玩家），再把 S25 放進 Q19 的融合**取代 S19**，用 prereg19 同一套 inner 評估判一次。
- inner（640 位訓練玩家，評估用）從來不進任何訓練，所以 G1／G2／G4 和 Q19 的接縫照樣有效。
- 先講預期：prereg24 的 P 軸是 S19 單獨、在留出題上的分數，而且 P25 → P50 沒有單調變好（雜訊大）。這次 A 玩家從 5,186 位變成約 6,630 位（×1.28，約 0.36 次「加倍」），照 +0.0101／加倍外推，S25 單獨約 +0.004；融合後要過 G1 的 +0.005 **並不容易，「沒過」是很可能的結果**。花費只有約 4 小時 GPU 和 1 小時 CPU，所以值得判一次。

## 1. 資料盤點（唯讀；`p25_data.py audit` → `audit.json`，和本檔同一個 commit）

驗證玩家**只**從已 commit 的 split v1 檔（`results/split_v1/players_split.csv` 的 `split == val`，用 `tools/c10data.read_split` 讀，和 prereg24 盤點同一個機制）決定；他們的棋和等級**只**從官方訓練 CSV（`data/training/train_<等級>.csv`，等級＝檔名）讀。程式每一列核對「檔名等級＝split 檔等級」。

| 項目 | 數字 |
|---|---|
| split v1 玩家 | 訓練 6,407（fit 5,767＋inner 640）、驗證 1,600 |
| 驗證玩家的棋 | **206,137 盤**、45,711,248 手；每人 2～1,442 盤，中位數 53、p90 340 |
| 驗證玩家 ∩ inner／fit 玩家 | 0／0 |
| 重複棋局 | 驗證棋 vs inner 棋（8 種對稱）**0**；vs fit、vs probe、驗證內部（原方向）都 0 |
| **V（本次要抽的棋）** | **74,946 盤、1,600 位玩家、16,668,238 手** |
| 4090 時間（同一支程式實測速度） | P3 的 1,239.8 局面／秒 → **約 3.7 小時**；P2 的 1,113 → 約 4.2 小時 |

每級（驗證玩家／fit 玩家；驗證玩家的棋；V 的盤數）：D 178／641、22,105、7,497；C 163／589、17,445、7,472；B 209／755、20,305、7,474；A 316／1,140、21,155、7,484；1D 87／312、21,429、7,529；2D 122／439、20,499、7,504；3D 124／445、19,084、7,524；4D 128／460、22,264、7,513；5D 131／474、19,214、7,461；6D 142／512、22,637、7,488。

## 2. 設計

### 2.1 加什麼、不加什麼

- **只加驗證玩家（V）**。不加 fit 玩家沒進 probe 的 450,307 盤（prereg24 的 extra）：prereg24 已判那條路 NO-GO（每人多盤 +0.0034、約 20 小時 GPU），而且同時加兩種資料會讓這次的判定分不出是哪一個起作用。
- **09 的 6 成員 P1TO_E 集成不動**（預設）。理由：① 範圍小：只改 SET 一項，G1 量到的就是「S19 多了玩家」；② P1TO_E 的 map 是 09 凍結的，重訓要重做 map、重擬合融合、測試日要換兩個凍結物，風險和工作量都翻倍；③ prereg24 顯示「吃資料」的是 set 模型（題目層級的訓練），逐盤模型在 08／09 時已經用 probe 的 270,000 盤做過資料量曲線。
- S25 只用 1,540 維 X（同 S19），不用 EX。

### 2.2 V 怎麼抽（照 probe 的抽法，讓新列和 probe 列分布一樣）

- probe 是三批（P1 每級 3,000、P2 每級 12,000、P3 每級 12,000），每批都是 `p1_extract.build_probe` 的規則：該級玩家排序後用固定種子洗牌，**輪流每人抽一盤**（該玩家還沒被抽過的盤中隨機一盤），抽滿該級配額為止；後一批排除前面批已抽的盤。
- V 照抄三批，**每級配額依玩家數等比例縮小**：配額＝round(probe 配額 × 該級驗證玩家數 ÷ 該級 fit 玩家數)（例如 D：833／3,332／3,332）。輪流抽樣的深度（每人盤數）取決於「配額 ÷ 玩家數」，所以等比例縮小就讓每位驗證玩家得到和同級 fit 玩家一樣多的盤。
- 種子：三批 20261125／20261126／20261127，每級用自己的 `random.Random("<種子>|<等級>")`（`p25_lib._build_one_level`，和 `build_probe` 逐行相同，只差配額分級、每級一個產生器）。
- 抽之前先排除和任何 inner 棋在 8 種對稱下相同的驗證棋（實際 0 盤）。
- **盤點證實分布相同**（`audit.json` `depth_games_per_player`）：每人盤數的平均每級都和 probe 相同到小數第 2 位（例如 A 級 23.68 對 23.68、1D 86.54 對 86.54），最大值也接近（A 37 對 37、6D 69 對 72）。抽題單位「某執色 ≥ 20 盤的玩家」每級多 41～99 位（probe 165～348 位）。
- `p25_data.py sets` 產生 `data/prereg25/v/sets.json`（`p1_extract` 的 sets 格式，probe 的 `source` 是 `val`，bank／mono 照抄 P1、不跑），遊戲清單的 SHA 必須等於 `audit.json` 的 `games_sha256`（`0607d974…`）。

### 2.3 P1 特徵（4090）

- 程式：`results/prereg08/p1/p1_extract.py` **一字不改**，經 `p25_extract.py` 執行：它只把 `p1_extract.set_games` 換成「`source == val` 時從 `p25_lib.load_val_rows` 取棋（核對玩家、執色和 sets.json 相同）、其他來源照原函式」，然後呼叫 `p1_extract.main`。設定和 P3 probe 相同：`run --set probe --workers 12`，fp16、batch 1024、group 4、channels-last（預設值）、每 200 盤一個 chunk、原子寫入、已完成的 chunk 自動跳過（逐 chunk 續跑）。
- 開跑前 `nvidia-smi`：驅動程式必須是 616.92、沒有別的運算程序（Type C；`p1_extract` 自己也會拒絕），否則不跑、回報。
- 獨立背景程序：`Start-Process` 跑 `run_v.sh`（記錄到 `data/prereg25/v/run.log`、`run_log.jsonl`）；`data/prereg25/v/STOP` 讓它在一個 chunk 後停下讓出 GPU，刪掉後重跑同一個指令續跑。
- **接縫 G（先跑，過了才抽 V）**：用同一個 `p25_extract.py`（`source == fit` 走原函式）把 P3 probe 的前 400 盤（chunk 0、1）重算到 `data/prereg25/seam/`，和 `probe_X.npy` 的對應列（`probe_meta` 的 `src == p3_probe`、`ref` 0～399）比：failed、counts、n_moves 和原 P3 chunk 相同；以 Q09 `p1e_map` 的 μ、σ 標準化後每個特徵的最大差 **≤ 0.05**（PREREG_08 5.2 的 GPU 接縫容許差 `TOL_FEAT`）。只記錄：是否逐位元相同、09 map lp_P1TO 的最大差。沒過就停，查原因（多半是驅動程式），不抽 V。

### 2.4 V 的特徵陣列（CPU）

- `p25_prep.py`：讀 V 的 chunk（`tools/p09_map.read_chunk_rows`，不改；和 09 prep 同一個函式：拿掉失敗或沒有目標手的盤，X＝[目標平均, 對手平均] float32 原值），寫 `data/prereg25/prep/v_X.npy`（N×1,540 float32）和 `v_meta.npz`（ref、等級 y、玩家鍵 `<等級>:<player_id>`、執色）；記錄 chunk SHA、不可用盤數。

### 2.5 S25（照 S19，只差訓練玩家）

- 列：probe 270,000 列（`data/prereg17/prep/probe_X.npy`，SHA＝09 check_report）＋V 的列。標準化：Q09 凍結 `p1e_map.npz` 的 μ、σ（`p09_forward.standardise_x`，同 S19）。
- 玩家角色（新鹽）：`int(sha256("c10-prereg25|holdout|<rank>:<player_id>")[:8], 16) % 10 == 0` → H（留出，只用來選步），否則 A（訓練）。玩家池＝probe 的 5,767 位＋V 的 1,600 位＝7,367 位，H 約 10%（同 S19 的比例；S19 是 581／5,767）。
- **其他全部照 S19，直接 import 不複製**：`p19_train.torch_model`、`train_seed`（AdamW lr 1e-3、wd 1e-2、OneCycle pct 0.1、15,000 步、batch 128、n ∈ {5,10,15,20}、每 500 步在留出題組算 NLL、取最小那一步）、`port_check`、`export_params`；`p19_lib.PlayerPool`、`sample_batch`、`holdout_set`（每級 × 每種盤數 100 題、`default_rng(19019)`）、numpy float64 前向。種子 0、1、2；抽題亂數 `default_rng([19, 種子])`；`torch.set_num_threads(8)`。
- `p25_train.py train`：CPU 獨立背景程序，`--resume` 保留已完成的種子（同一 commit、同一程式，同 prereg19／23 的規則），寫 `data/prereg25/models/s25_seed{0,1,2}.npz`（格式 `c10-prereg25 s25 model v1`）和 `train_record.json`（commit）。train 不讀任何 inner 資料。記憶體：特徵約 2.1 GB（345,000 列 × 1,540 × 4 位元組），另一個程序可能佔 20～28 GB，開跑前看可用記憶體。

### 2.6 候選和融合（唯一一個候選）

- **基準 Q19**：凍結 `results/prereg19/rank/s19_seed{0,1,2}.npz`（SHA＝prereg19 `train_record.json`），用 prereg23 判定時的同一段程式（import `results/prereg23/explore/p23_eval.py` 的 `s19_models`、`s19_term`、`refs`）重建：`Inner.run(Q08_TERMS + ["SET"], {"HS", "P1TO": E_P1TO_E, "SET": 4·lp_S19})`。
- **候選 Q25**：同一個呼叫，SET 項換成 4·lp_S25（同一個 `s19_term` 函式、同一個 P1 可用遮罩，項的順序、界限、起點、PENALTY、等級加權、規則 B、5 折玩家 OOF 都不變）。也就是說 Q25 和 Q19 只差 SET 項的 3 個模型檔。
- 不會洩漏：S25 只用 probe 玩家和驗證玩家訓練和選步，inner 玩家從來不在裡面（接縫 D）；融合參數照 Q09 的 inner 協定以玩家分 5 折 OOF。

## 3. 接縫（先過，否則停，不算候選）

1. **接縫 1**：Q02H、Q08、Q08 單色等於 `p09_rank_fuse.REF`；Q09（0.7114420614163589）、Q09 單色（0.7145431684624712）；**Q19 inner 0.7345843051712626、Q19 單色 0.7430172362615272**（prereg19 `eval_report.json`）；相差都 ≤ 1e-12（實際要逐位元相同，prereg23 已經做到）。
2. **接縫 F**：3 個凍結 S19 模型檔 SHA 等於 prereg19 `train_record.json`。
3. **接縫 P**：`probe_X.npy`、`inner_X.npy` 的 SHA 等於 09 `check_report.json`；`v_X.npy` 的 SHA 等於 `prep_record.json`，它的 chunk SHA 清單等於 V 的 chunk 檔。
4. **接縫 D**：A ∩ H＝∅；（A ∪ H）∩ inner＝∅（以 `<rank>:<player_id>` 比，也以 player_id 比）；V 玩家 ∩ probe 玩家＝∅；V 棋 vs inner 棋 8 種對稱重複＝0（`audit.json`）。
5. **接縫 M**：3 個 S25 模型檔 SHA 等於已 commit 的 `train_record.json`；訓練設定等於登記值（第 2.5 節）；訓練時程式乾淨；前向程式（`p19_lib.py`、`p19_train.py`、`p25_train.py`、`tools/p09_forward.py`）的 LF SHA 等於訓練時；移植核對過。
6. **接縫 G**（第 2.3 節，抽 V 之前）。

## 4. 閘門（單一候選 Q25，正式執行後只判一次；全部成立才算過）

- **G1**：inner 評估假題（127,200 題）的分等級平均差（Q25 − Q19）≥ **+0.005**（嚴格）。
- **G2**：同一差值的配對 PlayerBootstrap（`Inner.compare`：每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
- **G4**：單色假題（120,544 題）的分等級平均差，配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 ≥ 0（容許 1e-12）。
- 只有一個候選，不做多重比較校正。
- **只記錄、不能救候選**：Q19＋S25（加成第 8 項）；S25 單獨、S19 單獨（規則 B）；每個種子單獨的 Q25_s；依盤數（單色 n＝5／10／15／20、評估假題 `rf.N_BINS` 和 n 正好 5／10／15／20，配對區間）；依等級；每折融合權重；訓練紀錄（留出 NLL 曲線、選到的步數、秒數、抽題統計、合格玩家數、移植核對）。
- 不論結果，RESULTS.md 都寫出依盤數的差和 S25 單獨的分數。

## 5. 凍結、不能動的東西

- 不改：`results/prereg01–24/**`、`docs/PREREG_*`、`tools/*`（只 import）、測試日工具與指紋、驗證集 split v1、`results/prereg19/rank/` 的 S19 模型和 19 的融合參數、`results/test*`。
- 本探索的新檔只在 `results/prereg25/explore/`（commit）和 `data/prereg25/`（不進 git）。
- 競賽一不受影響：S25 只用於競賽二；競賽一剩 1 次的驗證集打分照舊保留（競賽一的驗證題來自同一批驗證玩家，但這次的訓練不碰競賽一的任何東西）。

## 6. 測試日路徑（本探索**不做**，過了才由主 session 做）

- 測試題的 P1 特徵 **11/04 已經算好**：`results/test08/p1/{public,private}_p1.npz`（19 apply 用的同一份，t19 讀的就是它），S25 **不用任何 GPU**，只要 CPU 幾分鐘。
- 要做的：`docs/PREREG_25`（凍結 3 個 S25 模型檔、融合參數用全部 inner 擬合假題以 `p4.gen_fit_record` 重擬合 7 項，同 19）、以 `tools/t19_rank.py` 為底的 t25（只換 3 個模型檔和融合參數；numpy 前向、只讀凍結檔）、假資料演練、單元和 hygiene 測試、指紋和 tag、`official_log.py` 只加行、`docs/TESTDAY_25.md`、獨立審查。上傳順序：19 之後（25 成為競賽二最後一份）；任何一步出問題就維持 19（現在已上傳的最後一份）。
- 揭露：25 和 19 一樣**沒在驗證集打過分**（競賽二 4 次已用完），而且用了驗證玩家的棋訓練——驗證集從此不再是獨立的檢查，報告要寫明。

## 7. 時程（今天 11/06；上傳目標約 11/18；硬截止 11/24 23:59）

| 時間 | 事 |
|---|---|
| 11/06 凌晨 | 本檔＋盤點 commit；`p25_extract.py`、`run_v.sh` commit；接縫 G（約 5 分鐘 GPU）；V 抽取（4090 約 3.7～4.2 小時，獨立背景程序） |
| 11/06 上午 | V prep（CPU 幾分鐘）→ S25 訓練（CPU 8 執行緒約 50～70 分鐘）→ 判定一次（約 5 分鐘）→ RESULTS.md → `--no-ff` 合併、閘門 |
| 11/07–11/14 | （過了）主 session：PREREG_25、凍結、t25、演練、獨立審查、合併 |
| 11/15–11/18 | apply 到測試題特徵、`check_submission`、上傳（K6-25 建議 11/20） |
| 11/24 23:59 | 硬截止；沒趕上就維持 19 |

- GPU 中斷：已完成的 chunk 保留，重跑同一個指令續跑。V 是三批依序、每批逐級排列，跑到一半的前綴不是各級平均，所以**不用部分結果**，一律等全部跑完才訓練。

## 8. 順序、判定和檔案

1. 本檔＋`p25_lib.py`＋`p25_data.py`＋`audit.json` commit（任何模型數字之前）。
2. `p25_extract.py`、`run_v.sh` commit → `p25_data.py sets` → 接縫 G → V 抽取 → `p25_prep.py`。
3. `p25_train.py`、`p25_eval.py`、`test_p25.py`（合成資料單元測試）commit；smoke（特徵換成 `default_rng(12345)` 雜訊、每個種子 300 步、留出每格 5 題，輸出 `data/prereg25/smoke/`，數字不記）；一輪對抗式自審，修完 commit。
4. **train 只跑一次**（`--resume` 只用於中斷，同一 commit、同一程式；重跑只准在指出具體程式錯誤、更新的 commit 之後 `--rerun --reason`）→ `train_record.json` commit。
5. **eval 只判一次**：所有接縫過之後、算 S25 項之前寫 `data/prereg25/eval_started.json`，之後中斷也算跑過 → `eval_report.json` 原樣 commit → `RESULTS.md` commit。
- **沒過**：記錄後收掉，19 照舊。**過了**：只代表「值得寫 PREREG_25」，不是採用。
- 環境：CPU `.venv`（Python 3.12.13、torch CPU）、GPU `.venv-gpu`；`PYTHONDONTWRITEBYTECODE=1`，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。worktree 在 session scratchpad，`subst Q:`（路徑含 `AppData\Local`，防護會誤擋），`data` 是指向主 repo `data` 的 junction（移除 worktree 前先拆）；worktree 的已追蹤檔換成和主 repo 工作區相同的位元組（只差換行；git 會顯示已修改，不 commit），本探索的 commit 只明確加 `results/prereg25/explore/` 的檔。

## 9. 實作補記（正式訓練和判定之前；閘門、候選、S25 設定都沒改）

- **接縫 G 過**（`seam_g.json`，`daa0c3c`）：P3 probe 前 400 盤重算，failed／counts／n_moves／ref 全同；標準化特徵最大差 0.0302 ≤ 0.05；09 map 的 lp_P1TO 最大差 0.0077（只記錄）；不是逐位元相同（fp16、cudnn benchmark，同 PREREG_08 rev 4 的經驗）。
- **V 抽取完成**：375 個 chunk、74,946 盤、16,668,238 局面、0 盤失敗；4090 實際 13,635 秒（約 3.79 小時；另接縫 G 108 秒）。驅動程式 616.92，開跑前沒有別的運算程序。`p25_prep.py prep`：74,946 列、0 盤不可用（`data/prereg25/prep/prep_record.json`，`v_X.npy` SHA `d70d151f…`）。
- 盤點（smoke 時讀到，不是結果）：A 6,619 位（311,086 列，其中 V 1,440 位）、H 748 位（33,860 列，其中 V 160 位）。
- **程式修正（smoke 時發現）**：`p19_train.loaded_repo_modules`／`p23_eval.loaded_repo_modules` 用 `REPO + os.sep` 當前綴，repo 根目錄是磁碟根（`Q:\`）時一個模組都收不到，所以 `code_state` 沒涵蓋 `p19_lib.py`、`p19_train.py`，接縫 M 的「前向程式和訓練時相同」會失敗。`p25_lib.loaded_repo_modules` 改用正規化的前綴，train／eval 的 `code_state` 兩者取聯集（prereg23／24 的紀錄也有同樣的漏洞，只記錄，不改它們的檔）。
- **worktree 換行**：一開始用 `core.autocrlf=true` checkout 再換成主 repo 的位元組，git 把約 490 個只差換行的檔都顯示成已修改，`tools/p03_rank_fuse.py` 的 prereg02 輸入檢查因此擋下 smoke eval。改成 `git -c core.autocrlf=false checkout -- .` 後再換成主 repo 的位元組（同 prereg23 第 8 節），只剩 16 個只差換行的舊紀錄檔顯示已修改（prereg02 checks／log、prereg03／04 check_submission、prereg07 pilot、prereg08 log），都不 commit。
- smoke（`data/prereg25/smoke/`，特徵換成雜訊、每個種子 300 步）：train 約 46 秒；eval 約 160 秒、記憶體峰值約 3.8 GB；接縫 1 七個數逐位元相同（Q19 0.7345843051712626、單色 0.7430172362615272），P、D（A∩H、訓練玩家∩inner、probe∩V 都是 0）、M、F 全過。雜訊項的數字無意義，不記。
- 單元測試 `test_p25.py`（合成資料 6 個：輪流抽樣的配額和平衡、配額不足時拒絕、分批不重複、8 種對稱鍵、留出比例、路徑防護）全過。
- 自審（本代理，對抗式）：① S25 和 S19 的差只有資料列和鹽：模型、訓練、抽題、選步、前向都 import 不改（`T.STOP` 改指 `data/prereg25/STOP` 是唯一的 monkeypatch）；② inner 不進訓練：train 只開 probe 和 V 的陣列，接縫 D 以 `<rank>:<player_id>` 和 player_id 兩種方式核對；③ 驗證玩家的身分只來自 split v1 檔，棋只來自訓練 CSV，程式路徑防護拒絕任何含 `data/local` 的路徑；④ 判定只一次：`eval_started.json` 在接縫全過、算 S25 項之前寫。沒有發現要再改的地方。
