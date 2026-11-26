> **勘誤（2026-09-26，PREREG_02 rev 2）**：2.2 節「照檔案順序切時兩套 0.766 對只用含執色版 0.661」的 0.661 其實是只看本人落子的 own 版本；5 個 seed 下正確的比較是 both 0.7436、只含執色 0.7232、不分執色 0.7713（`results/prereg02/explore/opening_player_sim.json`）。4.3 節的切法防護已在 PREREG_02 rev 2 改成三層（加上執色分開的模擬與執色比例條件）。本檔其餘內容保留設計當時的原文。

# PREREG_02 建議方案（2026-09-26，可以直接照著實作）

以 Judge 2 選的提案 3 當骨架：只推論的 Colab、重現核對、候選池自我檢索、整季打分預算、11/04 檢查。再嫁接三處：
- 提案 1：原始方向的開局 token、接著訓練、prereg01 唯讀保護。
- Judge 1：黑白兩套 token 一起用。
- 提案 2：固定的 inner bank、5 折 OOF、兩隊不交換模型輸出。

兩位評審指出的缺陷都修了。盤數可能變成隱藏特徵的問題，也在 4.2 節處理掉。

---

## 1. 一句話結論

PREREG_02 只要你用跑 prereg01 的那個帳號，在 Colab 各按一次「全部執行」，跑兩份 notebook（約 50 分鐘、約 8 個運算單元，推估）：
- **競賽一**：加上「比對玩家開局前 10 手」。在訓練玩家的模擬上約 0.89，現行 CNN 在本機驗證是 0.37。兩者題目不同，不能直接相減。
- **競賽二**：同一次執行把模型再訓練 4 個 epoch、存下每盤輸出，回本機再加開局模型和機率校準。

兩場各只在本機驗證集打一次分，過了事先寫好的門檻才換版。

---

## 2. PREREG_02 要比什麼

### 2.1 本機驗證集的打分（每場只打一次）

| 編號 | 新方法 | 對照組（固定） | 採用條件 | 合理性門檻（打分後） |
|---|---|---|---|---|
| **H2R**（競賽二） | 在 inner 上從 R00–R20 選出的組合（4.2 節），決策一律用規則 B（`tools/decide.py`） | prereg01 B：`results/prereg01/rank/pred_B.csv`（SHA `d42a0b11…`），帳上分等級平均 0.4908937784188385、每題平均 0.4919948709488189 | ① `B_minus_A_rank_balanced` 的 95% CI 下界 > 0（每級內分層重抽 10,000 次，seed 0）<br>② `B_minus_A_mean` 的點估計不 < 0<br>①② 都成立才採用，否則記為「結論依指標而定」、不採用 | 新方法分等級平均 < 0.40 時先查 bug |
| **H2P**（競賽一） | 用候選池自我檢索從 Q00–Q23 選出的組合，加上測試時防護（4.3 節） | prereg01 player：`player_val_top5.csv`（SHA `0f5761a1…`），帳上 0.3668476608218301 | `B_minus_A` 的 95% CI 下界 > 0 | 新方法 < 0.50 時先查 bug |

- **送分閘門**：沒過就不打分，這次的打分機會保留下來，RESULTS 記「未送分」。
  - 競賽二：inner OOF 分等級平均，選到的組合 − R00 ≥ **+0.010**。
  - 競賽一：自我檢索平均分，選到的組合 − Q00 ≥ **+0.02**。
- **查 bug 的規則**：分數低本身不算 bug，要指出具體的程式錯誤。修正後重跑要記帳，也算進整季預算。
- **兩場是不同的檢定家族**，不做合併校正。候選組合都在 inner 或自我檢索上選，本機驗證集上沒有多重比較。

### 2.2 只在 inner、fit 或候選池上決定的東西（本機驗證集完全不參與）

| 項目 | 用什麼資料決定 | 狀態 |
|---|---|---|
| 競賽二開局 n-gram 的 M=12、α=20 | inner 探索（提案 3，0.4228）。**要揭露**：之後的校準也用同一批 inner，所以 inner OOF 會略為樂觀 | 寫死 |
| 競賽二 R00–R20 的選擇和參數（a、γ、b、w） | inner 假題，5 折 OOF（4.2 節） | 程序寫死，數值由 `tune` 產生後 commit |
| 要不要用 p02e9（接續訓練）的組合 | p02e9 的 inner 逐盤 accuracy < 24.15%（即 25.15% − 1 個百分點），或第 9 個 epoch 沒跑完，就從清單拿掉所有用到 C4、C5 的組合 | 規則寫死 |
| 競賽一 token 設計（N=10、原始方向、黑白兩套 token） | 只用 fit 玩家的模擬池（Judge 1 重跑 0.889；只用含執色版 0.9018；照檔案順序切時兩套 0.766 對只用含執色版 0.661） | 寫死 |
| 競賽一 Q00–Q23 的選擇 | 候選池自我檢索。這是 VALIDATION 允許的「測試時使用候選庫」例外，只讀候選檔，**要寫進登記** | 程序寫死，官方每個池各自重跑同一個程序 |
| 競賽一防護門檻 θ | fit 玩家模擬池（4.3 節） | 在本機打分前算好並 commit |

- **競賽一不用 inner**：prereg01 player 用全部 6,407 位訓練玩家訓練過。

### 2.3 預算、對照組重讀、交件模型

- **整季預算**：每場到 11/24 最多打 4 次新方法。02、03、04 各 1 次，另留 1 次給 split v2 或查 bug 重跑。
- **對照組重讀**：`compare-*` 會把 prereg01 的兩份預測再算一次分，帳上會多出幾列。
  - PREREG_02 要明文寫：這是固定對照組的重讀，不是重新評估那個方法，也不依它做任何選擇。
  - 在 `results/prereg01/RESULTS.md`「之後不再用這兩份預測打分」那句後面，補一行有日期的註記。
  - A 側分數必須和帳上**完全相同**；不同就停下來查，B 側結果在查清前不採用。
- **交件模型選甲**：直接用本機驗證過的模型。02 不考慮乙。
- **登記的其他事項**：
  - 不用一題對一人的結構。
  - 兩隊不交換模型輸出。只共用前處理，也就是 `c10data.py` 和 `tools/p02_moves.py`。

---

## 3. Builder 規格

### 3.0 共通

- **prereg01 的兩個 builder 和 notebook 一律不改**。新 builder 讀已 commit 的 prereg01 notebook 當底稿：
  - 先 assert SHA：`prereg01_rank.ipynb` = `5f97d519…`，`prereg01_player.ipynb` = `54c96f03…`。
  - 再依 cell metadata `c10`（`new`／`official_cell`）挑格子，用 `rep()` 改。`rep()` 的目標字串必須剛好出現一次。
  - 所有新格都是新增的 code 格。
- **新檔**：
  - `colab/build_prereg02_rank_nb.py` → `colab/prereg02_rank.ipynb`，差異檔 `results/prereg02/notebook_diff_rank.txt`（對 prereg01_rank 逐格比）。
  - `colab/build_prereg02_player_nb.py` → `colab/prereg02_player.ipynb`，差異檔 `results/prereg02/notebook_diff_player.txt`。
  - `ROOT` 從命令列參數讀，不寫死路徑。
- **每格都要檢查**：
  - `compile()` 過得了。
  - 不含 `answers` 字串。
  - cell id 唯一（`c10-rank02-NN`／`c10-player02-NN`）。
  - 重跑 builder，輸出要逐位元組相同。
- **打包檔 `C10_bundle_v1` 不改**：不用重傳，特徵指紋也對得上。
- **prereg01 唯讀保護**：
  - 在 setup 格定義 `P01_ROOT = MyDrive/C10/results/prereg01`（SMOKE 時是 `prereg01_smoke`）。
  - 另外定義 `guard_write(path)`：realpath 落在 `P01_ROOT` 底下就 raise。
  - 所有寫檔、`torch.save`、`np.savez`、`shutil.copyfile` 的目的地、`rmtree`、zip，一律先呼叫它。
  - 開頭和結尾各記一次 `P01_ROOT` 的檔名、大小、mtime（`p01_readonly_before.json`／`_after.json`），結尾 assert 兩者相同。
- **結果目錄**：`MyDrive/C10/results/prereg02/{rank,player}/`。
  - 每個輸出檔都 < 100 MB，因為結果 zip 會跳過 `.pth`、`.pt` 和大於 100 MB 的檔。
  - zip 是 `prereg02/rank_results.zip`、`prereg02/player_results.zip`。

### 3.1 `prereg02_rank.ipynb`（競賽二，TEAM_11248）

| # | 來源 | 內容 |
|---|---|---|
| 1 | 新增 md | 說明、使用者步驟、輸出清單 |
| 2 | 底稿 `setup` 加 rep | `RESULT_ROOT` 改成 `prereg02`（SMOKE 是 `prereg02_smoke`）；`FEAT_BACKUP = prereg02/features_rank`；新增 `P01_RANK`、`P01_CKPT`、`P01_FEAT = prereg01/features_rank`；定義 `guard_write`，並接到 `write_json` |
| 3 | 新增 preflight | 見表下說明 |
| 4 | 底稿 `prepare` | 不改，逐檔核對打包檔 SHA |
| 5 | 官方 1 | imports |
| 6 | 底稿 `environment` | 不改 |
| 7 | 底稿官方 2 加 rep | `CKPT_DIR = prereg02/rank/ckpt`。`NUM_EPOCHS` 維持 5（P01 的值，比對 state 用）。新增 `EXTRA_EPOCHS = 4`（SMOKE 是 1）、`PEAK_LR = 1e-3` |
| 8 | 底稿官方 4 | `load_rows` 的 fit／inner 加外洩 assert，`leak_check.json` 寫到 prereg02 |
| 9 | 底稿官方 6、8 | 抽特徵的 worker，只在還原失敗時用到 |
| 10 | 底稿 `feature_backup_tools` 加 rep | 見表下說明 |
| 11 | 底稿官方 9 加 rep | 照上述順序還原或重抽 |
| 12 | 底稿官方 11、12 | loader、`pick_workers`（4 個 worker），`VAL_SIZE` 要等於 73,556 |
| 13 | 官方 14、15 | 模型、`CrossEntropyLoss`、`AdamW(lr=1e-3)`、`validate()` |
| 14 | 新增 dump 工具 | 見表下說明 |
| 15 | 新增 P1 輸出與重現核對 | 見表下說明 |
| 16 | 新增接續訓練 | 見表下說明 |
| 17 | 新增 P2 輸出 | 用 `epoch_9.pth` 輸出 `inner_p02e9_d4.npz`、`val_p02e9_d4.npz`。第 9 個 epoch 沒完成就跳過並記錄 |
| 18 | 改寫底稿 `final_check` | 核對所有 dump 的列數和完成標記；比對 P01 前後指紋；寫 config、timings、run_log；打包 `rank_results.zip`（約 80 MB） |

**格 3 preflight**
- 確認這些檔存在：`P01_CKPT/epoch_{1..5}.pth`（SMOKE 時是 `1..NUM_EPOCHS`）、`best_model.pth`、`state.pt`，以及 `P01_RANK/rank_val_probs.csv`。
- 記下各檔的 SHA-256，assert `sha(best_model.pth) == sha(epoch_5.pth)`。
- `P01_FEAT/_COMPLETE.json` 不存在時只警告：會重抽特徵，多約 6 分鐘。
- Drive 剩餘空間要 ≥ 3 GB。
- 任何一項不符，就用中文說明處理方式後停下，不花 GPU 時間。

**格 10 特徵工具（`feature_backup_tools` 加 rep）**
- `restore_features(fp, src)` 改成可以指定來源；還原時依序試 `[prereg02/features_rank, prereg01/features_rank]`，對來源唯讀。
- `backup_features` 只寫到 prereg02，`rmtree` 前先過 `guard_write`。

**格 14 dump 工具**
- **對稱變換**：`d4(x, s)`：`y = flip(x, dims=(3,)) if s >= 4 else x`，再 `rot90(y, k=s % 4, dims=(2, 3))`，s=0 是原方向。
- **`dump_inner(net, name, syms)`**：
  - 對 `val_loader` 跑一遍，每個 batch 在 GPU 上做 syms 種變換，存 float32 logits。
  - key 是 `(LEVELS[label], game_name)`，接回 `df_val` 的 `player_id`、`color`。
  - assert 列數 = 73,556、key 唯一、key 集合 = `df_val` 的 `(rank, game_id)`。
- **`parse_val()`**：照 `rank_val_questions.csv` 的列順序和 slot 1..num_games，用 `Pool(NPROC)` 解析。失敗的保留位置，logits 填 NaN。
- **`dump_val(net, name, syms)`**：**每題一個 batch**，和 prereg01 一樣，這樣重現核對才會一致。
- 每個檔先寫成 `.tmp` 再改名，並附 `_DONE_<name>.json`（列數、SHA、權重 SHA、秒數）。重跑時完成標記和 SHA 都對得上就跳過。

**格 15 P1 輸出與重現核對**
- **輸出**：
  - `epoch_1..4.pth` 各輸出原方向：`inner_p01e{1..4}.npz`（各 2.9 MB）、`val_p01e{1..4}.npz`（各 0.8 MB）。
  - `epoch_5.pth` 輸出 8 種對稱：`inner_p01e5_d4.npz`（[N,8,10]，23.5 MB）、`val_p01e5_d4.npz`（6.1 MB）。
  - key 檔：`inner_keys.npz`（rank、game_id、player_id、color、label）、`val_keys.npz`（question_id、slot、color、parsed）。
- **重現核對**（寫進 `repro_check.json`，任一項不過就停下，**不進入訓練**）：
  - ① p01e5 原方向在 inner 的逐盤 accuracy = 25.1482% ± 0.05 個百分點。
  - ② C0（每盤 softmax 在 float32 下平均）和 `P01_RANK/rank_val_probs.csv` 相比，最大差距 ≤ 1e-4。和 P01 的 GPU 型號不同時放寬到 1e-3，並記下 GPU 名稱。
  - ③ 規則 B 選出的等級不同的題 ≤ 4 題。

**格 16 接續訓練**
- **初始化**：
  - 還沒有 `prereg02/rank/ckpt/state.pt` 時，唯讀 `torch.load(P01_CKPT/state.pt)`。
  - assert：`epoch == NUM_EPOCHS`、`config['features'] == FEAT_FINGERPRINT`、LR 1e-3、batch 256、seed 42；模型權重和 `epoch_5.pth` 逐張量 `torch.equal`。
  - 載入 model 和 AdamW 狀態。
- **學習率**：`LambdaLR(opt, λ(t) = max(0, 1 − t/T))`，`T = EXTRA_EPOCHS × P01_STEPS`。`P01_STEPS` 取 state 裡 history 最後一列的 steps，正式執行時 assert = 2814，所以 T = 11,256。每次 `optimizer.step()` 後呼叫 `scheduler.step()`。
- **其他設定**：
  - epoch 編號 6–9；每個 epoch 開始時 `random`、`np`、`torch` 的 seed 設為 `42 + epoch`。
  - fp32、沒有資料增強，其餘和 prereg01 相同。
- **每個 epoch 結束**：
  - 跑 `validate()`，只當監看、記進 `ckpt/history.csv`。
  - 存 `epoch_N.pth`，最後存 `state.pt`（model、opt、scheduler、global_step、epoch、history、`STATE_CONFIG_P02`）。
  - `STATE_CONFIG_P02` = {SEED、BATCH_SIZE、PEAK_LR、SCHEDULE、EXTRA_EPOCHS、T、VAL_SIZE、features、`init_state_sha256`}。
- **接續**：有 P02 的 `state.pt` 就讀它，設定不同就停下。斷在 epoch 中間時，那個 epoch 從頭重跑，學習率排程從已存的 global_step 接續。
- **不產生 `best_model.pth`**。登記指定的 checkpoint 是 **`prereg02/rank/ckpt/epoch_9.pth`**。

**Drive 用量**：4 個 `.pth` 加 `state.pt` 約 0.75 GB，dump 約 80 MB，zip 約 80 MB。重抽特徵的話再多約 1.3 GB。

### 3.2 `prereg02_player.ipynb`（競賽一，TEAM_11249，只推論）

- **沿用的底稿格**：
  - `setup`（rep：`RESULT_DIR = prereg02/player`；加上 `P01_PLAYER_CKPT` 和 `guard_write`）
  - `prepare`、官方 1、`environment`
  - 官方 2（rep：checkpoint 路徑改讀 `P01_PLAYER_CKPT`，不 `makedirs`）
  - 設定格（底稿 cell 8）
  - 官方 20 裡的 `sgf_windows`、`embed`
- **刪掉**：官方 4–18、21，以及外洩索引檢查和接續相關的新格。這份不讀任何訓練資料。
- **新增 preflight**：確認 `prereg01/player/ckpt/best_model.pth`、`player_val_top5.csv`、`player_val_embeddings.npz` 存在，記下 SHA。
- **新增 `game_embeddings()`**：key 是每盤一個，候選是 `game_id`，題目是 `(question_id, slot)`；存每盤窗口 embedding 的**加總**（float32）和窗口數。
  - `player/dumps/cand_games.npz`：game_id、player_id、color、emb_sum [40000,256]、n_windows，約 41 MB。
  - `player/dumps/query_games.npz`：question_id、slot、color、emb_sum [4935,256]、n_windows、parsed，約 5 MB。
- **重現核對**（寫進 `repro_check.json`，不過就停下）：
  - 依窗口數加權算重心，再用 L2 取 top-5，寫成 `player_val_top5_repro.csv`。
  - 和 P01 的 top-5 比：400 題中至少 398 題完全相同。
  - 和 P01 npz 的重心比：最大差距 ≤ 1e-4。
- **結尾**：`player_results.zip` 加上 P01 前後指紋比對。

### 3.3 inner 假題（本機產生；寫在這裡是因為它決定 dump 的用法）

- **bank**：inner 裡 ≥5 盤的 636 人。每人的棋依 `sha256("c10-prereg02|bank|{rank}|{game_id}")` 排序，取前 min(20, n) 盤，合計 11,813 盤。只從 CNN dump 裡有的棋取。
- **玩家順序**：依 `(LEVELS.index(rank), player_id)`。
- **第 r 輪**：
  - `rng = random.Random(int(sha256(f"c10-prereg02|pq|{r}").hexdigest()[:16], 16))`。
  - 每位玩家 `k = min(rng.randint(5,20), len(bank))`，`rng.sample(bank, k)`，和 `make_val.py` 相同。
  - 擬合用 r = 0–49，評估用 r = 1000–1199。
- **分折**：每一級內依 `sha256(f"c10-prereg02|fold|{player_id}")` 排序，依序輪流分到 5 折。
- **擬合集只收 inner 盤數 ≥20 的 508 人**（各級 58、48、58、70、33、46、44、47、51、53）。這樣每題的 k 都從 20 盤的 bank 均勻抽，和等級無關（4.2 節）。評估時 636 人全收，包括會被截斷的人，和驗證集的情形一樣。

### 3.4 斷線與重跑

- 使用者只要再按一次「全部執行」：
  - 已完成的 dump 靠完成標記和 SHA 跳過。
  - 特徵從 prereg02 或 prereg01 的備份還原。
  - 訓練從 P02 的 `state.pt` 接續。
- player 的 dump 大約 5 分鐘，斷了就整份重跑。

### 3.5 G4 時間（沿用 prereg01 實測推算，還沒實測）

| 階段 | rank | player |
|---|---|---|
| setup／prepare／env／load_rows | 約 1 分 | 約 0.7 分 |
| 還原特徵、`pick_workers` | 1.5–3.5 分（重抽的話 +6 分） | — |
| P1 輸出與核對 | 5–7 分 | — |
| 接續訓練 4 × 437 秒，加監看 | 約 30.5 分 | — |
| P2 輸出 | 約 4 分 | — |
| 每盤 embedding 與核對 | — | 約 4.5–5 分（P01 實測 258 秒） |
| **合計** | **約 42–48 分，約 6.1–7.0 單元** | **約 5–7 分，約 0.9 單元** |

### 3.6 交給使用者之前

1. **冒煙測試**：在同一個 `--scratch` 資料夾裡，先跑 prereg01 兩份的冒煙測試，產生 `prereg01_smoke`，再跑 prereg02 兩份。只用 CPU 版 torch。重現核對在冒煙測試裡也要通過。
2. **產生差異檔**。
3. **請一個代理獨立審查**，逐項確認：
   - prereg01 唯讀
   - key 對得回去
   - 8 種對稱正確
   - 學習率排程和接續
   - 沒有答案檔
   - 每個輸出 < 100 MB
4. **commit**，再更新 `docs/COLAB_RUN.md` 的 prereg02 一節。

---

## 4. 本機後處理（只用 CPU，全部經 `c10data.load_rows` 或題目檔、候選檔讀資料，**不開任何 `*answers*`**）

### 4.1 新檔

| 檔案 | 用途 |
|---|---|
| `tools/p02_moves.py`（兩隊共用的前處理） | 用 regex `;([BW])\[([a-s]{2})\]` 取前 60 手；座標的 8 種對稱變換；`canon(mv, M)`：選「前 M 手座標 tuple 最小」的那個 s（同分取 s 最小），整盤套用 |
| `tools/p02_rank_pseudo.py` | 3.3 節的 bank、各輪題目、分折 → `results/prereg02/rank/pseudo_inner.json` |
| `tools/p02_rank_opening.py` | 開局等級 n-gram → `rank/opening_inner.npz`、`rank/opening_val.npz` |
| `tools/p02_rank_fuse.py` | `repro`、`tune`、`apply` |
| `tools/p02_player_open.py` | token、TF-IDF；`calibrate-guard` |
| `tools/p02_player_fuse.py` | `repro`、`select`、`apply` |
| `tests/test_p02.py` | 見表下說明 |
| `results/prereg02/explore/` | 見表下說明 |

**`tests/test_p02.py` 要測的**
- 假題可重現，換 salt 結果會不同。
- 分折在各級內平均。
- 擬合集每題的 bank 都是 20 盤。
- 本機用的規則 B 和 `decide.py` 的 `rule_b` 一致。
- R00 在合成資料上等於 softmax 平均。
- 在合成資料上擬合，能找回事先設定的溫度。
- 8 種對稱組成封閉的群；`canon` 在對稱變換下不變。
- 兩套 token：換執色後，不分執色的 token 不變。
- 自我檢索切出的題目和參考沒有重疊，參考剛好 80 盤。
- 所有 p02 工具的原始碼都不含 `answers`。

**`results/prereg02/explore/`（和 PREREG_02 rev 1 一起 commit）**
- 把 scratchpad 的 `judge/opening_both.py`、`opening_colour_robust.py`、`judge_meth/opening_sim_verify.py`、`p02/opening_rank_nb*.py` 搬進來，重新跑出 json。原本那份 `opening_sim.json` 已經被覆蓋。
- 另外補一張表：3 種切法（隨機、檔案順序接下 k 盤、執色完全分開的「最後 k 盤」）× 3 種 token（含執色、不分執色、兩套），只當揭露用。

### 4.2 競賽二

**開局 n-gram（寫死）**
- 每盤取前 12 手，用 `canon(mv, 12)` 轉成標準形。
- context 是（目標玩家執色, 前綴），兩色的每一手都要預測。
- 只用 fit 的棋計數。
- 每一級的 log 似然：
  - ℓ_k += log((N_k(c,pre,mv) + α·pg) / (N_k(c,pre) + α))
  - α = 20，pg = (N(c,pre,mv) + 1) / (N(c,pre) + 361)
  - 遇到沒見過的 context 就停止累加。
- 每盤取 õ_g = log_softmax(ℓ)；一手都沒算到的盤，標記為不參與。

**CNN 輸入**（每盤 ℓ_g 是 log 機率）

| 代號 | 內容 |
|---|---|
| C1 | p01e5，原方向 |
| C2 | p01e5，log_softmax(mean_s log_softmax(z_s))，s 是 8 種對稱 |
| C3 | p01e3、e4、e5 三個 epoch 平均：log_softmax(mean_e log_softmax(z_e)) |
| C4 | p02e9，原方向 |
| C5 | p02e9，8 種對稱，算法同 C2 |

**每題**（n = CNN 解析成功的盤數）
- 兩種合併方式：
  - prob：m = log(mean_g exp ℓ_g)
  - log：m = mean_g ℓ_g
- 開局項：m_op = mean_g õ_g，平均只算有開局分數的盤；一盤都沒有就是 0。
- p = softmax( e^a · (n/12)^γ · (m + w·m_op + b) )，其中 Σb = 0。
  - b 用 10×9 的固定正交基底參數化，基底存進設定檔。
  - **b 放在溫度裡面**，所以 n 只改變機率的尖銳程度，對 10 個等級作用相同。

**防止 num_games 變成特徵的三道防線**
1. b 放在溫度裡面。
2. 擬合集只用盤數 ≥20 的 inner 玩家，所以擬合資料裡 k 和等級無關。
3. 請獨立審查專門檢查這一點。另外報告各盤數分箱的預測等級分布。

**組合清單**（順序固定，也是平手時的優先順序）

| 組合 | 內容 |
|---|---|
| R00 | C1、prob、不校準（a=γ=w=0、b=0），也就是 prereg01 B 的重現 |
| R01–R10 | C1、C4、C3、C2、C5 依序，各配 prob、log，加校準（a、γ、b） |
| R11–R20 | 依序等於 R01–R10 再加開局項（多一個參數 w） |

- p02 不能用時（2.2 節的規則），拿掉 R03、R04、R09、R10、R13、R14、R19、R20。

**擬合**
- 目標：分等級加權的 NLL，每題權重 1/(10·該級題數)，加上 0.01·‖b‖²。
- 用 L-BFGS-B，範圍：a ∈ [−4,4]、γ ∈ [−1,2]、w ∈ [0,4]。
- 兩個起點（a=0、a=1，其他全 0），取 loss 較低的；maxiter 500。

**`tune`**
- **只接受** `inner_*.npz`、`opening_inner.npz`、`pseudo_inner.json`，並 assert 每個 player_id 都屬於 `players_for(split,'inner')`。
- 每折用其他 4 折的擬合集（第 0–49 輪）擬合，在本折所有玩家的第 1000–1199 輪上評估。
- 報告：OOF 分等級平均、每題平均、各級分數、盤數分箱（5–7、8–12、13–20），以及按玩家重抽 1,000 次的標準誤。
- **選法**：OOF 最高者；和最高分差距在 0.003 以內的，取清單中最前面的。
- 選定後在全部擬合集重新擬合，寫出 `tune_report.json`、`frozen_config.json`（組合代號、參數、基底、輸入檔 SHA、工具的 commit）。

**`apply`**
- 設定檔沒有 commit，或和 HEAD 不同時，拒絕執行。
- 讀 `val_*.npz`、`opening_val.npz`、`data/local/rank_val_questions.csv`。
- 寫出 `rank/rank_val_probs.csv`，欄位和 prereg01 相同，機率用 repr 寫出；n_parsed = 0 的題機率欄留空。
- 接著跑 `python tools/decide.py results/prereg02/rank/rank_val_probs.csv results/prereg02/rank`。

**`repro`**
- 用 R00 在驗證題上的結果，比對 `results/prereg01/rank/rank_val_probs.csv`：最大差距 ≤ 1e-4，規則 B 不同的題 ≤ 4 題。
- 過了才能跑 `tune`。

### 4.3 競賽一

**開局 token（寫死）**
- 每盤取目標玩家自己的前 10 手 `own`，和整盤前 10 手 `seq`。
- 含執色的一套：`"{c}{i}:{m}"`、`"{c}*:{m}"`、`"{c}q:{col1}{m1}>{col2}{m2}"`。
- 不分執色的一套：`"o{i}:{m}"`、`"o*:{m}"`、`"q:{m1}>{m2}"`。
- 兩套一起用。
- 方向有兩種：raw（照 SGF 原樣），或 canon（`canon(mv, 10)`）。
- 每個候選或每題：把 token 次數加總，取 1+ln(次數)，乘上 IDF = ln((1+P)/(1+df)) + 1（df 只在池內的候選上算），L2 正規化後算 cosine，記作 cosO。

**CNN**
- E0：依窗口加權的重心，距離是 −L2（官方做法）。
- E1：每盤先平均並 L2 正規化，再跨盤平均、再正規化，用 cosine。

**融合**
- s = z_q(E) + w·z_q(O)。z_q 是對該題所有候選做標準化；標準差 < 1e-12 時取 0；缺哪一項就把那一項當 0。
- 同分時依候選代號排序；整題沒有可解析的棋，照官方填排序後的前 5 位。

**組合清單**（順序固定）

| 組合 | 內容 |
|---|---|
| Q00 | 只用 E0，也是 prereg01 的重現和閘門的基準 |
| Q01 | 只用 E1 |
| Q02–Q06 | E0 + O_raw，w = 0.25、0.5、1、2、4 |
| Q07–Q11 | E1 + O_raw，w 同上 |
| Q12–Q16 | E0 + O_canon，w 同上 |
| Q17–Q21 | E1 + O_canon，w 同上 |
| Q22 | 只用 O_raw |
| Q23 | 只用 O_canon |

**`select`（自我檢索，只讀候選檔和 `cand_games.npz`）**
- 每位候選的 100 盤依 `sha256(f"c10-prereg02|sr|{player_id}|{game_id}")` 排序，依位置分成 5 份，每份 20 盤。
- 第 r 輪（r = 0–9）用第 r mod 5 份：
  - `rng = Random(int(sha256(f"c10-prereg02|sr-k|{r}")[:16],16))`。
  - 候選依代號排序，每人 `k = rng.randint(5,20)`、`rng.sample(那 20 盤, k)` 當題目。
  - 其餘 80 盤當參考，IDF 和 E 都只用參考的棋算。
- 照官方公式計分，選平均最高的組合；差距在 0.003 以內的，取清單中最前面的。
- 寫出 `sr_report.json`、`frozen_config.json`。

**防護（測試時的程序，現在一起登記）**
- ρ = 「真題的 max_c cosO 中位數」÷「自我檢索假題的 max_c cosO 中位數」，用選定的方向算。
- **θ 怎麼定**：`calibrate-guard` 用 fit 玩家模擬 3 個隨機切的池和 3 個照檔案順序切的池，θ = 兩組 ρ 平均的中點。在本機打分前 commit。兩組差距 < 0.05 時停用防護。
- **ρ < θ 時**：
  - w 改成 min(w, 1)。
  - 選到 Q22 或 Q23（只用開局）的話，改用「E0＋同一方向、w=1」。
- 選到只用 CNN 的組合時，不套用防護。

**`apply`**
- 用完整的 100 盤和 400 位候選重算，記下 ρ 和最後用了哪個組合。
- 寫出 `player/player_val_top5.csv`、`apply_report.json`。

**`repro`**
- Q00 的 top-5 要和 prereg01 至少 398/400 題相同，重心最大差距 ≤ 1e-4。
- **不過又修不好（7 天內）時**，照登記把清單縮成 Q22、Q23，不用另寫修訂。

### 4.4 commit 順序和打分指令

> **rev 3 勘誤**：依 PREREG_02 第 7 節第 6 條，第 5、6 步的順序改成：兩邊的 `tune`、`calibrate-guard`、`select` 先跑完，兩份 `frozen_config.json` 一起 commit，之後才跑任何一邊的 `apply` 和打分。下面的列法是 rev 3 之前的順序。

1. **PREREG_02 rev 1 和 explore**：commit 後才開始寫程式。
2. **程式本體**：builder、notebook、差異檔、tools、tests、冒煙紀錄、審查結果。
3. **Colab 結果**：解壓到 `results/prereg02/`，先 commit，這時還沒調參。
4. **重現核對**：`python tools/p02_rank_fuse.py repro`、`python tools/p02_player_fuse.py repro`。
5. **競賽二**：
   - 依序跑 `p02_rank_pseudo.py`、`p02_rank_opening.py build`、`p02_rank_fuse.py tune`，commit。
   - 跑 `p02_rank_fuse.py apply` 和 `decide.py`，再跑 `python tools/check_submission.py rank results/prereg02/rank/pred_B.csv --questions data/local/rank_val_questions.csv`，commit。
   - 過了閘門才打分：
     ```
     python tools/score.py compare-rank results/prereg01/rank/pred_B.csv results/prereg02/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg02 H2R (A = prereg01 B, reference re-read)" --json results/prereg02/rank/score_H2R.json
     ```
6. **競賽一**：
   - 跑 `p02_player_open.py calibrate-guard` 和 `p02_player_fuse.py select`，commit。
   - 跑 `apply`，再跑 `python tools/check_submission.py player results/prereg02/player/player_val_top5.csv --questions data/local/player_val_questions.csv --candidates data/local/player_val_candidates.csv`，commit。
   - 過了閘門才打分：
     ```
     python tools/score.py compare-player results/prereg01/player/player_val_top5.csv results/prereg02/player/player_val_top5.csv data/local/player_val_answers.csv data/local/player_val_candidates.csv --ledger results/score_ledger.csv --note "prereg02 H2P (A = prereg01 player, reference re-read)" --json results/prereg02/player/score_H2P.json
     ```
7. **收尾**：打分紀錄帳、分數 JSON、`results/prereg02/RESULTS.md`、PORTFOLIO_MAP 第 0 節，一起 commit。

---

## 5. 使用者要做的事和運算單元

> **rev 3 之後的註記**：本節的使用者步驟已被 PREREG_02 第 7 節第 8 條和 `docs/COLAB_RUN.md` 的「事前登記 02」一節取代，本節只留作設計紀錄。不同的地方包括：競賽一只用 G4、拿不到時等，不換 GPU（下面「player 用 L4」不再適用）；每份跑完先刪除執行階段再下載；交件方式、斷線後怎麼重跑，都以 COLAB_RUN 為準。

**Colab：一次，兩份 notebook**

0. **先確認帳號**：
   - 用**跑 prereg01 的同一個帳號**（AI Pro）。權重、`state.pt`、特徵備份都在它的 Drive。
   - 在「執行階段 → 查看資源」確認剩 **≥10 單元**。
   - Drive 剩 ≥3 GB。
   - 不夠的話告訴 Claude，Claude 會寫搬到 Pro+ 帳號的步驟，約要搬 2.5 GB。
1. 把 Claude 給的 `prereg02_rank.ipynb`、`prereg02_player.ipynb` 放進 `MyDrive/C10/`。打包檔不用重傳。
2. **開 `prereg02_rank.ipynb`**：
   - 「執行階段 → 變更執行階段類型」選 **G4**。
   - 按「全部執行」，允許 Drive 存取。約 45 分鐘。
3. **在另一個分頁開 `prereg02_player.ipynb`**：同樣選 G4、按「全部執行」，約 6 分鐘。Colab 不讓你同時開第二個 GPU，就等 rank 跑完再跑。
4. **跑完之後**：
   - 下載 `MyDrive/C10/results/prereg02/rank_results.zip` 和 `player_results.zip`，把下載路徑告訴 Claude。
   - 按「執行階段 → 中斷連線並刪除執行階段」。
5. **斷線或出錯時**：再按一次「全部執行」就會接著跑。出現錯誤訊息就截圖給 Claude。

**運算單元**（推估，G4 每小時 8.71 單元）
- rank 約 6.1–7.0 單元，player 約 0.9 單元，合計 **約 7–8 單元**。
- 要重抽特徵的話再多約 0.9 單元。
- 拿不到 G4 時：
  - rank 用 A100 標準版，約 1–1.5 小時、5.4–8.1 單元，這個速度沒量過。
  - player 用 L4，約 15–20 分鐘、約 0.5 單元。

**在那之後**
- 11/04 前不用再跑 Colab。
- 11/04 以後：你下載測試檔；Claude 產生測試版 notebook，你跑一次「只推論」（兩份合計約 15 分鐘、約 2–3 單元）；每次上傳前你在對話裡說 OK。

**Claude 的工作量**：約 3–4 個工作天。
- 9/27：PREREG_02 和 explore。
- 9/28–30：builder、tools、tests、冒煙測試、審查。
- 你跑 Colab 之後 2–3 天：打分、寫 RESULTS。

---

## 6. 刻意不放進 02 的做法（03 的候選）

| 做法 | 為什麼現在不做 | 去向 |
|---|---|---|
| 競賽二 recipe v2：目標玩家視角加執色層、D4 資料增強、盤面歷史 360、cosine LR、bf16、固定最後一個 epoch | 特徵要重抽、改動大，G4 約 1–2 小時。先看 02 的接續訓練和校準拿到多少 | **PREREG_03 主要候選** |
| 逐手、依等級條件化的落子模型 | 工程大，文獻換算約 0.67–0.75，但資料不同 | 03 或 04，要重用 02 的 inner bank 和融合工具 |
| 競賽一重訓 embedding：留出 inner、拿掉 64 維瓶頸、PK 取樣加 contrastive | 開局已經拿到大部分訊號；CNN 沒有乾淨的調參集 | 03 |
| CSLS、hubness 修正 | hubness 輕微；需要池的結構 | 03 用自我檢索評估 |
| player 的 8 種對稱 TTA | GPU 時間是 8 倍，開局才是主力 | 03 |
| KataGo HumanSL | 規則和觀感有疑慮（簡報第 15 張寫「所有人都從零訓練」）；可能看過同一批來源的棋；Blackwell 上的工程風險 | 你先登入 AIdea 看討論區或詢問紀錄，而且約 2 單元的 pilot 過了閘門才考慮 |
| Hungarian 指派、一題對一人 | 官方是不是一題對一人還不知道 | 11/04 確認後另寫登記 |
| 用棋力模型重排競賽一（跨題任務） | 等於兩隊交換特徵值，超出「共用前處理」 | 不做，除非你明確擴大共用範圍 |
| 乙（用全部 8,007 人重訓） | 本機驗不到；要看 11/04 測試玩家和訓練玩家有沒有重疊 | 04，第一次上傳前決定 |
| 序數 loss（CORN、SORD）、限制每人盤數 | 證據弱；CE 加校準再套規則 B 已經針對期望得分 | 03 的消融再看 |
| 輸棋分數損失、比對外部棋譜庫 | 前者成本高又沒貢獻；後者違反資料 README 的「不得用於識別真實使用者」 | 永不做 |

---

## 7. 風險與防護

| 風險 | 防護 |
|---|---|
| 驗證集外洩 | Colab 上沒有答案檔。訓練資料只經 `load_rows`（會 assert 人數、盤數和外洩）。`tune` 只接受 inner 的檔，並 assert player_id 都屬於 inner。開局 n-gram 只用 fit 計數。競賽一只讀候選檔（登記過的例外）。所有工具不含 `answers`。`apply` 要等設定 commit 之後才能跑 |
| 盤數變成等級訊號（兩位評審都指出） | b 放在溫度裡面；擬合只用盤數 ≥20 的 508 人；報告各盤數分箱的預測分布；獨立審查專門檢查這點 |
| 寫壞 prereg01 的檔 | `guard_write` 擋住所有寫入；開頭和結尾比對指紋；讀到的權重記 SHA；從 prereg01 的 `state.pt` 初始化前逐張量比對 |
| 輸出對錯了盤 | key 集合 assert；競賽二三項、競賽一兩項重現核對，任一不過就不訓練、不打分 |
| 挑 checkpoint 或校準過擬合 | p02 固定用第 9 個 epoch；校準最多 13 個參數；依玩家 5 折 OOF；b 有 ridge；閘門 +0.010／+0.02；平手選清單前面的 |
| 官方的題目和候選不是隨機切（例如依執色切） | 黑白兩套 token 都用（照檔案順序切的模擬 0.766，只用含執色 token 是 0.661）；ρ 防護和 fallback 先登記；11/04 檢查黑白比例 |
| 官方棋譜被隨機旋轉或鏡射 | 清單有 O_canon，自我檢索在官方池上會自己選到 |
| Colab 斷線、拿不到 G4 | 每個 dump 有完成標記；接續訓練靠 `state.pt`；第 9 個 epoch 沒完成就照規則只用 P1 的組合；每個輸出都 < 100 MB |
| 多重比較 | 每場這次只打 1 次，整季最多 4 次；組合全在 inner 或自我檢索上選；失敗也要報告 |
| 兩隊的規則界線 | 兩份 notebook、兩組工具、兩份報告分開；只共用 `c10data.py`、`p02_moves.py`；模型輸出不跨隊 |
| 本機和官方不一致 | 分等級平均和每題平均都報告；選甲；public 分數比上一版低 0.05 以上才追查 |

**11/04 檢查**（登記但不打分；虛無分布在 11/04 前用 fit 算好）

| # | 檢查 | 結果不符時 |
|---|---|---|
| T1 | 欄位、題數、每池 400×100、`color_1..20`、每盤是否從第 1 手開始 | 只改讀寫程式 |
| T2 | 用落子序列加 8 種對稱雜湊比對：測試 vs 訓練、題目 vs 候選、不同題目之間 | 開 split v2，另寫登記 |
| T3 | 候選和訓練玩家有沒有重疊：開局 TF-IDF 相似度對照虛無分布，≥10% 的候選超過門檻就判定重疊 | 乙優先，另寫登記 |
| T4 | 候選庫和題目的黑白比例、第一手位置的分布 | 只記錄；防護已經事先登記 |
| T5 | `num_games` 的分布 | 只記錄 |
| T6 | 是否一題對一人 | 沒有明文就不用 |

**這次沒有做的事**：沒有開任何 `*answers*.csv`、沒有跑 `score.py`、沒有改 repo、沒有用 GPU。本機只讀 `players_split.csv`，算出 inner 盤數 ≥20 的有 508 人。

**要搬進 repo 的 scratchpad 腳本**：
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\judge\opening_both.py`
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\judge\opening_colour_robust.py`
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\judge_meth\opening_sim_verify.py`
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p02\opening_rank_nb.py`
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p02\opening_rank_nb_m20.py`