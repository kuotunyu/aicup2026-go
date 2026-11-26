# prereg44 探索計畫：直接學棋力等級的逐手 CNN（長訓練）＝CNN44，加進 Q25 當第 8 項（競賽二，inner 判一次）

2026-11-12 寫成，**在任何 CNN44 的快取統計、訓練、分數之前 commit**（分支 `p44-rankcnn`）。主 session 交辦（使用者 10/2、10/16、11/10 授權 AI 決定方法；11/10、11/11 要求 YOLOv4 式「逐模組拼裝＋消融＋長訓練」）。這是探索：不寫 `docs/PREREG_44`、不做測試日工具、不上傳。驗證集答案資料夾不開、不讀、不列、不雜湊（指令裡也不寫它的路徑），不跑 `tools/score.py`，不讀驗證題特徵。

## 0. 白話

- 競賽二的 25（Q25）在現有特徵（官方 CNN R20＝C5、開局、對手 O4、兩臂 SW、HumanSL HS、KataGo 探針 P1TO、set 模型 S25）上，11/06 後 10 輪改良都沒過：**融合和 set 模型已到頂，要新的「資訊來源」**。
- 競賽一這兩天的經驗：自己從零訓練的逐手 CNN（prereg36：官方 17 平面、ResNet 10×128、每步 96 盤×24 手）把步數從 4 萬拉到 16 萬，留出模擬 0.42 → 0.59；prereg42 V2（SE＋全域池化偏置）在同步數再 +0.018。
- 競賽二的 CNN（R20）只是官方範例多訓練幾輪，**從沒做過「長訓練＋直接學等級＋更好的主幹」**。
- 這次：用 prereg42 V2 的主幹，從競賽二官方訓練資料**重新建快取**，訓練玩家＝S25 的 A 玩家（6,619 位，inner 和留出玩家都不在裡面），每盤 24 手 → 盤向量 → 10 級等級的 softmax，Colab G4 約 12 小時（20 萬步）。每盤得到 10 級的 log 機率，一題取平均＝新項 CNN44，加在 Q25 的 7 項之外（不取代任何項），用 inner 5 折 OOF 融合判一次。
- 先講預期（不是結果）：R20 本身在 inner 已經是融合的底（C5），CNN44 和它、和 HS／P1TO 一定有重疊；過 G1 的 +0.005 **不容易**。值得做的理由：這是 11/06 後唯一一條「新訊號」的路，成本主要是 Colab（使用者 11/12 起沒有運算上限）。

## 1. 規則界線（兩場是不同隊伍，不可共享特徵值）

- **不用競賽一的任何權重、向量、輸出、特徵或快取**：prereg35–43 的 `data/prereg3x|4x/` 底下的 ckpt、emb、cache、P1 塊一律不讀（程式的路徑防護直接拒絕這些資料夾，見第 9 節）。
- **可以 import 競賽一的程式碼**（同一作者；報告寫「前處理共用」）：`results/prereg36/explore/p36_lib.py` 的棋盤重播、官方 17 平面編碼（和官方教學 `extract_features` 逐值相同，prereg36 的測試已核對）、8 種對稱、`segment_mean`。prereg42 V2 的主幹（`GPSENet`：殘差塊＋SE＋第 2／5／8 塊的全域池化偏置，10×128）在分支 `p42-recipe`（未合併到 main），**照抄**到 `p44_lib.py` 並註明來源 commit（`249d1f5` 的 `results/prereg42/explore/p42_lib.py`）；EMA 類別同樣照抄。
- 資料與快取全部從競賽二官方訓練 CSV（`data/training/train_<等級>.csv`）重新建，放 `data/prereg44/`。

## 2. 玩家角色（怎麼保證 inner 乾淨）

照 S25（prereg25）一字不改，所以 CNN44 和 S25 的留出玩家是同一批：

| 角色 | 定義 | 用途 |
|---|---|---|
| inner | `c10data.players_for(split, "inner")`（640 位，split v1 訓練玩家的 10%） | **只在第 6 節判定時用**；CNN44 從來不訓練、不選步 |
| 候選池 | fit 玩家（`players_for(split, "fit")`，5,767 位）∪ 驗證玩家（split v1 `split == val`，1,600 位；`p25_lib.val_players`） | — |
| A（訓練） | 候選池中 `p25_lib.holdout_bit("<等級>:<player_id>") == 0`（鹽 `c10-prereg25|holdout|`） | CNN44 訓練 |
| H（留出） | 同上 `== 1` | 選存檔（第 4 節）、0b（第 5 節） |

- 驗證玩家的棋和等級只從官方訓練 CSV 讀（`p25_lib.load_val_rows` 同樣的核對：檔名等級＝split 檔等級）；驗證集答案資料夾不碰。
- **接縫 D（建快取時）**：A、H 的人數與各級人數等於 S25 `train_record.json` 的 `A_players`／`H_players`／`*_per_level`；A∩H＝∅；(A∪H)∩inner＝∅（以 `<等級>:<player_id>` 和只比 player_id 兩種）。不等就停。
- 訓練集裡沒有任何重複棋局（整盤、含 8 種對稱，`docs/DATA.md`），所以 A 的棋不會以複本形式出現在 inner 或 H。
- 競賽二的驗證集打分已用完；用驗證玩家訓練（同 25）要在報告寫明。

## 3. 模型與訓練（CNN44）

### 3.1 資料與快取（CPU，`p44_data.py cache`）

- 三份棋步快取（`p36_lib.moves_of` 的 uint16 碼，格式同 prereg36 的 cache：`codes.npy`、`off.npy`、`meta.npz`、`cache.json` 含 SHA）：
  - `data/prereg44/cache/train/`：A 玩家的**全部**官方棋（黑白都收；不抽樣、不設每人上限——抽樣改在取樣器做，見 3.3）。
  - `data/prereg44/cache/hold/`：H 留出題（第 5 節的 4,000 題）用到的棋（不重複）。
  - `data/prereg44/cache/inner/`：inner 的 73,556 盤，順序＝`results/prereg02/rank/dumps/inner_keys.npz`（逐列核對 rank、game_id、player_id、color）。
- 逐列串流讀 CSV（不一次載入全部 SGF；記憶體峰值目標 < 2 GB）。快取的盤數、手數、大小、秒數寫進 `results/prereg44/explore/data_record.json`（commit）；這些是資料統計，不是模型數字。
- 訓練程式只開 `cache/train`；它的 meta 必須只有 A 玩家（程式斷言）。

### 3.2 網路

- 輸入：官方 17 平面（目標玩家最近 8 個局面的子、對手的子、執黑全 1／執白全 0），`p36_lib.planes_for_codes`，只取目標玩家第 8 個非虛手之後的手（同官方教學）。
- 主幹：prereg42 V2 的 `GPSENet`（10 塊×128、SE ratio 8、全域池化偏置在第 2／5／8 塊、輸出 [平均, 最大] 池化 → Linear 256），照抄。
- 每手 256 維 → 一盤的 24 手取平均（`segment_mean`）→ **Linear(256 → 10)** → 10 級 softmax。
- **頭選 10 類 softmax＋交叉熵（不用 CORAL／累積 logit、不用 soft label）**。理由：① 下游融合吃的是「每盤 log 機率、一題取平均」（同 C5／HS），需要校準好的機率，交叉熵是 proper scoring rule；② 訓練盤有 80 萬盤級的數量，序數限制省下的參數沒有意義，反而限制「6D 和 D 都像某種風格」這類非單調訊號；③ soft label（鄰級 1/e）會把機率往鄰級推，融合時重複計入（規則 B 已經用期望分數做決策）。序數結構留給融合層和規則 B。

### 3.3 取樣與訓練設定

- 每步 96 盤：每盤先**等級均勻**抽一級，再在該級的 A 玩家（盤數 ≥ 5，黑白合計）中**玩家均勻**抽一人，再在該人的棋中均勻抽一盤（黑白都可），取最多 24 個目標手（隨機、不重複），整盤一個隨機對稱（8 種）。亂數 `numpy.random.default_rng([44, 步數])`，在 worker 內，與 worker 數無關（`--auto-resume` 續跑同一串）。
  - 理由：等級均勻＝評分的「分等級平均」；玩家均勻＝每位玩家同權（prereg24：缺的是「更多人」，不是每人更多盤；盤數最多的 10% 玩家占該級約 40% 的棋，按盤均勻會讓模型記住少數人的風格）。
- AdamW lr 2e-3、wd 1e-4、OneCycle pct 0.05（prereg36／42 的設定）、fp16 autocast、種子 0、**總步數 200,000**、每 20,000 步存檔（含 optimizer、scheduler、scaler、EMA、步數、設定；`--auto-resume`）。EMA decay 0.999（同 prereg42）。
- 步數的依據：prereg42 V2 在 Colab G4（RTX PRO 6000）實測 4.47 步／秒（同樣 96 盤×24 手）→ 200,000 步約 12.4 小時。notebook 先 bench 300 步並記錄；**步數不因 bench 改**（OneCycle 依總步數排程）。bench 低於 3.5 步／秒（> 16 小時）也照跑，只是靠 `--auto-resume` 跨斷線。
- Colab：結果每次存檔都寫到 Drive `MyDrive/C10/p44/run/`；斷線後重按「全部執行」接續。

### 3.4 推論（每盤 10 級 log 機率）

- 一盤用**全部**目標手（不是 24 手）的每手向量平均 → 頭 → logits（float32 存檔），恆等對稱（不做 TTA；第 8 節只記錄）。
- 候選版本（只有這 8 個）：第 140,000、160,000、180,000、200,000 步的存檔 × {原始權重, EMA}。
- 對每個候選在 Colab 算 `hold` 和 `inner` 兩份快取的每盤 logits（`data/prereg44/logits/<版本>/{hold,inner}.npz`，含 game key 和快取 SHA）。**算 inner 的 logits 不是打分**：在第 4 節的選擇 commit 之前，本地程式不讀任何 inner 的 logits（`p44_eval.py` 在 `select.json` commit 之前拒絕 `eval`）。

## 4. 選版本（只用 H，不碰 inner）

- 指標：H 留出題用到的盤（`cache/hold`）的**分等級平均每盤 NLL**（每級盤的平均 −log p(真等級)，10 級再平均）。8 個候選取最低者；寫 `select.json`（commit）之後才可以跑第 5、6 節。
- 0b 和選版本用同一批 H 玩家（不同指標：盤 NLL 對題目的堆疊 NLL），有小的選擇偏差；8 個候選、差異預期很小，記錄並揭露。

## 5. 階段 0b：留出題的堆疊 NLL（不碰 inner）

- 留出題：S25 的同一批 4,000 題（`p40_lib.Rows`＋`split_pools` → `p19_lib.holdout_set(pool_h, 100, 19019)`；每個盤數 5／10／15／20 各 1,000 題、單色）。接縫 H：題數、抽樣紀錄和 S25 `train_record.json` 的 `holdout_questions`、`holdout_sampling` 逐字相等。
- 題目的棋從列號對回 game key：probe 列經 `p23_train.probe_colours` 用的 sets.json（`src`、`ref`）、V 列經 `data/prereg25/v/sets.json`；核對 player key 相同。
- S25 的題目 log 機率：凍結 S25 三個模型（SHA＝prereg25 `train_record.json`）以 numpy float64 前向、lse 平均（同 prereg40 的 0b）。
- CNN44 的題目項：E＝該題各盤 log_softmax(logits) 的平均（同融合裡的 `pooled(S, lp, "log")`）。
- 兩個堆疊模型，**以玩家分 2 折交叉擬合**（玩家鍵的 sha256 奇偶；第 f 折的參數只用另一折的題擬合），L-BFGS、等級加權（每級總權重相同）：
  - base：z＝a·lp_S25＋b（b 為 10 維、b_0 固定 0）
  - cand：z＝a·lp_S25＋c·E＋d·(n/12)·E＋b
- **過**：Δ＝NLL(cand) − NLL(base)（4,000 題的平均 NLL，交叉擬合的預測）≤ **−0.010**。沒過就停：不跑 inner、不寫 PREREG_44。
- 依據：prereg31 的 0b −0.0102 → inner +0.0047；prereg40 的 0b −0.0072 → inner −0.0014；這次是「加項」不是「換項」，用 −0.010。
- 只記錄：每個盤數的 Δ、CNN44 單獨（z＝c·E＋d·(n/12)·E＋b，同一交叉擬合）的 NLL 和規則 B 分數、S25 單獨的分數、擬合參數。
- 0b 的結果寫 `stage0b.json`，commit 之後才可以跑第 6 節。

## 6. 判定（inner，只判一次）

- 基準 **Q25**：OP、O4、SWA、SWB、HS、P1TO、SET（凍結 S25），用 prereg41 判定時同一段程式重建（`p23_eval.s19_term`、Q09 的 `p1e_map`、`tools/p09_rank_fuse.Inner.run`）。
- 候選 **Q44**：同一個 `Inner.run`，項＝Q25 的 7 項＋`CNN44`，E_CNN44＝`rf.pooled(S, lp44, "log")`（lp44＝選定版本在 inner 每盤的 log_softmax(logits)，float64；inner 每盤都有，所以不需要遮罩），fit／eval／mono 三組。權重界限、起點、PENALTY、等級加權、規則 B、5 折玩家 OOF 都不改（`p04_rank_fuse.gen_*`）。
- **接縫 1**：Q02H、Q08、Q08 單色、Q09、Q09 單色、Q19、Q19 單色等於既有參照；**Q25 inner 0.7456519871546352、單色 0.7482450052195216**（綁 `results/prereg25/explore/eval_report.json` 的 candidate 欄），差 ≤ 1e-12（預期逐位元相同）。關掉 CNN44（就是 Q25 本身）逐位元重現。
- **接縫 K**：inner logits 檔的 game key 順序和 `inner_keys.npz` 逐列相同；logits 檔和 `cache/inner` 的 SHA 等於 Colab 的紀錄；版本等於 `select.json`。
- **接縫 M**：凍結 S25 三個模型檔 SHA 等於 prereg25 `train_record.json`；`inner_X.npy` SHA 等於 09 的 `check_report.json`。

| 閘門 | 條件（差＝Q44 − Q25） |
|---|---|
| G1 | inner 評估假題（127,200 題）分等級平均差 ≥ **+0.005** |
| G2 | 同一差值的配對 PlayerBootstrap 95% 下界 > 0 |
| G4 | 單色假題（120,544 題）差的 95% 下界 > 0，且黑題、白題的差各 ≥ 0 |

- `p23_lib.gates`（門檻 0.005）原樣使用；只有一個候選。
- 判定前寫 `data/prereg44/eval_started.json`（接縫全過、算 Q44 之前），之後中斷也算判過；程式錯誤要重跑，先 commit 寫明錯誤的修正（`--rerun --reason`）。
- **只記錄、不能救候選**：CNN44 單獨（規則 B）；依盤數（單色 5／10／15／20、評估假題）；依等級；每折融合權重和是否打到界限 (0, 4)；其他 7 個候選版本**不算**（只判選定的那一個）。

## 7. 已知偏差（要寫進結果和報告）

- inner 已被 P1、08、09、prereg12、15–41 重複使用，選擇偏差至少約 0.003。門檻不降。
- prereg29：S25 的 3 個種子在 inner 上剛好偏好 → 對照 Q25 可能偏高，對候選保守。
- 訓練用了驗證玩家（同 25）。
- R20（C5）是融合的底，和 CNN44 都是「看棋盤的 CNN」，重疊預期很大。

## 8. 只記錄、之後可做（本輪不做）

- TTA-8、多種子、更大主幹、序數頭：只有 CNN44 判定過了才考慮，而且要新登記。
- 測試日（過了才做，由主 session）：測試題（`data/test` 的競賽二 public／private）建 cache → 選定版本的 logits（4090 或 Colab，約數分鐘）→ 用全部 inner 擬合假題重擬合 8 項的融合（同 PREREG_25 的做法）→ 以 `tools/t25_rank.py` 為底做 t44；寫 PREREG_44、凍結、假資料演練、獨立審查。任何一步出問題就維持 25。

## 9. 執行與檔案

- 本輪（本代理）只做：本檔 → 程式 → 單元測試 → CPU 冒煙（`--device cpu`、極小規模）→ 快取 → Colab 套件與 notebook。**不做正式訓練、不算任何 H 或 inner 分數、不上傳**。4090 被競賽一占用，不用。
- 路徑防護：程式拒絕任何含驗證答案資料夾、`rank_val_answers`、`player_val_answers` 的路徑；也拒絕讀競賽一的資料夾 `data/prereg35`–`data/prereg43`、`data/prereg06`（只准讀 `data/training`、`data/prereg17/prep`、`data/prereg25`、`data/prereg26/prep`、`data/prereg08/p1`、`data/prereg08/p2`、`data/prereg09/p3`、`data/prereg12/q09_inner`、`data/prereg44`，以及既有 Inner 工具自己讀的 prereg02–09 檔）。
- 環境：CPU `.venv`、GPU `.venv-gpu`（本地只做 CPU 冒煙）；`PYTHONDONTWRITEBYTECODE=1`，不設 OMP／OPENBLAS／MKL 執行緒變數；torch CPU 執行緒 ≤ 8；可用 RAM < 4 GB 就停。worktree 在 session scratchpad、`subst S:`、`data` 是指向主 repo `data` 的 junction。
- 檔案（`results/prereg44/explore/`）：`PLAN.md`、`p44_lib.py`（角色、快取、取樣、模型、EMA、推論）、`p44_data.py`（cache／hold）、`p44_train.py`、`p44_infer.py`、`p44_eval.py`（select／stage0b／eval）、`p44_colab.py`（套件、分割、notebook）、`test_p44.py`；之後 `data_record.json`、`train_record.json`、`select.json`、`stage0b.json`、`eval_report.json`、`RESULTS.md`。Colab 檔在 `results/prereg44/colab/`；大檔在 `data/prereg44/`（不進 git）。
- 時程：11/12 套件 → Colab 訓練約 12–13 小時 → 下載 → select → 0b → inner 判定；**硬截止 11/18 23:59 判完**，過了就不做（測試日工具要在 11/24 前完成）。

## 10. 修訂 1（2026-11-12，寫程式與冒煙之後；仍在任何 CNN44 訓練、H 或 inner 分數之前）

只補實作細節，候選、閘門、門檻都沒改。

- **資料統計**（`data_record.json`，`p44_data.py record`，接縫 D、H 都過）：訓練快取 832,593 盤、6,619 位 A 玩家、184,829,436 手、0 盤解析失敗、470 MB；取樣器用盤數 ≥ 5 的 6,570 位（每級 730／678／865／1,277／359／503／503／525／542／588；丟掉 49 位），可抽 832,425 盤。inner 快取 73,556 盤（`inner_keys.npz` 順序逐列核對）。hold 快取＝S25 的 4,000 題用到的 24,827 盤、669 位 H 玩家（題數、抽樣紀錄和 S25 `train_record.json` 逐字相同）。建快取 CPU 共約 1.5 分鐘。
- **0b 的玩家分折**：`sha256("c10-prereg44|0b|<等級>:<player_id>")` 前 8 位 mod 2；堆疊的等級加權＝每級總權重相同；L-BFGS-B maxiter 1000、無正則化；起點 a＝1、其他 0。
- **存檔檔名** `step_NNNNNN.pt`（6 位數）；候選版本名 `sNNNNNN_raw|ema`。推論每次 32 盤、每批最多 4,096 個局面；壞盤（沒有目標手）logits 記 0 並標 `bad`，0b 和判定遇到壞盤就停。
- **冒煙（都不是結果、數字不記）**：CPU 3 步訓練＋`--auto-resume` 續跑；CPU 推論（hold、inner 各取前幾十盤）；`select --smoke`、`stage0b --smoke`（CNN44 換成雜訊；接縫 S25 模型、題目逐題相同都過）、`eval --smoke`（CNN44 換成雜訊；接縫 1 九個參照值**逐位元相同**，Q25 0.7456519871546352／單色 0.7482450052195216；約 2 分鐘）。CPU 上滿批次的 bench 只確認能跑（共用電腦、和 G4 無關，不記）。
- **worktree 換行**：照 prereg25 第 9 節，`core.autocrlf=false` 重新 checkout 後把已追蹤檔換成主 repo 工作區的位元組；`results/prereg02/rank/pseudo_inner.json` 不同時 `p03_rank_fuse.p02_inputs` 會擋。剩 19 個只差換行的舊紀錄檔顯示已修改，不 commit。
- **Colab 套件**：`C10_p44_kit.zip`（deflate）sha256 `ff20262c3a4a…`、265,884,140 bytes，17 個檔（程式 5 個＋三份快取各 4 個）；切成 38 份 7,000,000 bytes（`results/prereg44/colab/kit_parts_sha.txt`）；notebook `results/prereg44/colab/p44_train.ipynb` 第 0 格逐份核雜湊再接回。已在本機把分割檔接回、解開、逐檔核對，並從解開的資料夾（`python -I`）跑通訓練與推論的 CPU 冒煙，確認套件自足。
