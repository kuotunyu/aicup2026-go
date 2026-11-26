# 事前登記 01 結果

登記：`docs/PREREG_01_BASELINE.md` rev 3。驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。
各規則的分數引用 `results/score_ledger.csv`；差值、信賴區間引用同一次打分寫出的 JSON（`rank/score_H1.json`、`player/score_player.json`）。完整精度看原檔，本檔四捨五入到小數第 4 位。

## 競賽二（棋力）

### 結論

- **H1 成立，之後競賽二用決策規則 B（期望得分）。**
  - 主要指標（分等級平均）：B−A ＝ +0.0171，95% 信賴區間 [0.0027, 0.0311]。下界 > 0，符合採用條件。
  - 次要指標（每題等權平均）：B−A ＝ +0.0120，95% 信賴區間 [−0.0052, 0.0293]。區間含 0，但點估計 > 0。
  - VALIDATION rev 2 的規則是：兩個指標中「一個的 CI 下界 > 0，另一個的點估計 < 0」才算結論相反、不採用。這次次要指標的點估計是正的，不算相反。
- 合理性檢查：A 的分等級平均 0.4738，高於 0.25 的門檻，沒有觸發查 bug。
- 本機分數和官方 public baseline（0.4631）不能直接比，原因見 PREREG_01「baseline 分數本身」。

### 分數（各規則：`results/score_ledger.csv`，2026-09-25T19:28:32Z，git `07b569d`，note `prereg01 H1`；差值：`rank/score_H1.json`）

| 規則 | 分等級平均（主要） | 每題等權平均 | 完全猜中率 | 差一級率 |
|---|---|---|---|---|
| A：取機率最大的等級（官方） | 0.4738 | 0.4800 | 0.3499 | 0.3537 |
| B：取期望得分最高的等級 | 0.4909 | 0.4920 | 0.3304 | 0.4393 |

| 差值 B−A | 點估計 | 95% 信賴區間 | 重抽中 ≤0 的比例 |
|---|---|---|---|
| 分等級平均（每級內分層重抽） | +0.0171 | [0.0027, 0.0311] | 0.0108 |
| 每題等權平均 | +0.0120 | [−0.0052, 0.0293] | 0.0851 |

- 配對 bootstrap 10,000 次、seed 0（`tools/score.py` 的預設）；1,589 題、10 個等級。
- 小註：`score.py` 取第 250 和第 9,749 個（從 0 起算）重抽值當 95% 區間的兩端，比常見取法可能窄一個次序統計量。對結論沒有影響：10,000 次重抽中只有 108 次 ≤ 0。
- 打分指令（PREREG_01 的固定指令，只多了兩處：預測檔寫出所在路徑 `results/prereg01/rank/`，另加 `--json` 存輸出；打分紀錄帳的 command 欄只記 `compare-rank`）：
  `python tools/score.py compare-rank results/prereg01/rank/pred_A.csv results/prereg01/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg01 H1" --json results/prereg01/rank/score_H1.json`
- A、B 各打這一次分，之後不再用這兩份預測打分。
  - 註（2026-09-26）：事前登記 02（`docs/PREREG_02_OPENING_FUSION.md` 1.3、2.3）用 `compare-*` 和 prereg01 比較時，指令會把 prereg01 的 `pred_B.csv`、`player_val_top5.csv` 當固定對照組再讀一次分，帳上會多出列。這不是重新評估 prereg01，也不依它做任何選擇；分數必須和上面的紀錄完全相同。

### B 改了哪些題（不看答案，`rank/decide_summary.json`）

- 1,589 題中 656 題 A、B 不同。
- 最大宗是兩端往內移：6D→5D 221 題、D→C 256 題。兩端的等級只有一個鄰居，差一級的部分分數只能從一邊來，所以 B 很少選 D 和 6D（A 選 D 373 題、6D 264 題；B 選 D 115 題、6D 42 題）。
- 結果是 B 完全猜中的比例少了 1.95 個百分點（0.3499→0.3304），差一級的比例多了 8.56 個百分點（0.3537→0.4393），總分較高。

### 這次執行的實際情況

| 項目 | 內容 | 來源 |
|---|---|---|
| 執行環境 | Colab，**使用者的 Google AI Pro 方案帳號**（不是原本規劃的 Pro+ 帳號） | 使用者 9/26 告知 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition（97,887 MiB）＝ G4 | `rank/env.json` |
| CPU／RAM | 48 vCPU，約 190 GB；抽特徵 NPROC 48（官方寫死 8，營運性差異，run 前已 commit）、訓練 DataLoader worker 4 | `rank/env.json`、`rank/config.json` |
| 軟體 | Python 3.13.15、torch 2.11.0+cu128、CUDA 12.8 | `rank/env.json` |
| 執行時間 | 2026-09-25 18:37:50 – 19:22:01 UTC（`run_log` 的起訖，相差 44 分 11 秒）；`timings` 從設定格開頭算起共 2,678 秒，比 `run_log` 的起點早約 27 秒 | `rank/run_log.json`、`rank/timings.json` |
| 各段秒數 | 抽特徵 355、每個 epoch 約 436–438、推論 37 | `rank/timings.json` |
| 接續 | 沒有中斷、沒有接續（resume_count 0），5 個 epoch 都在第 1 次執行完成 | `rank/run_log.json` |
| 訓練量 | 每個 epoch 2,814 步、720,307 盤，5 個 epoch，沒有偏離官方設定 | `rank/ckpt/history.csv` |
| 外洩檢查 | fit 5,767 人／720,307 盤、inner 640 人／73,556 盤，人數盤數 assert 通過，0 盤出現在本機驗證檔，fit 與 inner 玩家重疊 0 | `rank/leak_check.json` |
| 選用的權重 | `best_model.pth`＝第 5 個 epoch（內部驗證逐盤 accuracy 25.15%，官方 `validate()`） | `rank/config.json` |
| 推論 | 1,589 題、19,156 盤全部解析成功，沒有缺題；每題 10 個機率加總與 1 的差距在 1.2e-7 以內 | `rank/config.json`（runs[0].outputs）、`rank/decide_summary.json` |
| A 的交叉核對 | `decide.py` 的 A 和 notebook 自己算的 argmax（`pred_official_argmax.csv`）1,589 題全部相同 | 本機比對 |
| 結果 zip | `rank_results.zip` SHA-256 `e65f03e9b75315dc2ba0b9b772f1208a1031c4f8d75c3c40286018821d0bae95`（zip 本身不進 git，內容已解開 commit） | 本機計算 |

內部驗證（inner 640 人，逐盤；不是本機驗證集）每個 epoch 的 accuracy：21.47%、19.72%、23.22%、21.76%、25.15%。不是逐 epoch 上升，但最後一個 epoch 最高。

### 注意事項

- `rank/config.json` 的 `prereg` 欄寫「rev 2」：notebook 裡的標籤沒跟著 rev 3 更新（notebook 本身已含 rev 3 的內容，例如結尾打包結果 zip）。rev 3 對齊檔名與路徑、寫明 `decide.py` 的防呆、加上結果 zip，實驗內容與採用條件和 rev 2 相同，不影響結果。
- 執行帳號和規劃不同（見上表），不影響程式和資料；結果和權重存在那個帳號的 Google 雲端硬碟 `C10/results/prereg01/`。

## 競賽一（辨識玩家）

### 結論

- 官方 baseline 只用訓練玩家重訓後，本機驗證集（400 題）分數 **0.3668**：第 1 名就猜中 30.0%，正確玩家在 Top-5 內 60.25%。
- 這是之後競賽一實驗的比較基準，沒有門檻（PREREG_01「baseline 分數本身」）。
- 和官方 public baseline（0.2845）不能直接比，原因見 PREREG_01。

### 分數（`results/score_ledger.csv`，2026-09-25T19:32:41Z，git `e63814b`，note `prereg01 player baseline`）

| 指標 | 值 |
|---|---|
| 分數（正確玩家在第 r 名得 e^-(r-1)，400 題平均） | 0.3668 |
| 第 1 名猜中率 | 0.3000 |
| Top-5 命中率 | 0.6025 |

- 打分前先過嚴格格式檢查：`python tools/check_submission.py player results/prereg01/player/player_val_top5.csv --questions data/local/player_val_questions.csv --candidates data/local/player_val_candidates.csv` → 400 題全部通過。
- 打分指令：`python tools/score.py player results/prereg01/player/player_val_top5.csv data/local/player_val_answers.csv data/local/player_val_candidates.csv --ledger results/score_ledger.csv --note "prereg01 player baseline" --json results/prereg01/player/score_player.json`
- 只打這一次分。

### 這次執行的實際情況

| 項目 | 內容 | 來源 |
|---|---|---|
| 執行環境 | Colab，使用者的 Google AI Pro 方案帳號（同競賽二） | 使用者 9/26 告知 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition ＝ G4 | `player/env.json`、`player/run_log.json` |
| CPU／RAM | 48 vCPU，約 190 GB；NPROC 48、DataLoader worker 48 | `player/env.json`、`player/run_log.json` |
| 軟體 | Python 3.13.15、torch 2.11.0+cu128、CUDA 12.8 | `player/env.json` |
| 執行時間 | 2026-09-25 18:38:32 – 19:29:46 UTC，共 3,074 秒（約 51 分鐘） | `player/run_log.json`、`player/timings.json` |
| 各段秒數 | 抽特徵 283、每個 epoch 約 8.4 分鐘、推論 258 | `player/timings.json` |
| 接續 | 沒有中斷、沒有接續（1 次執行完成 5 個 epoch） | `player/run_log.json` |
| 訓練量 | **偏離官方設定（輕微）：每個 epoch 14,976 步，不是官方的 15,000 步**，5 個 epoch 共 74,880 步／7,488,000 個 triplet（官方 75,000 步／7,500,000 個，少 0.16%）。原因：官方每個 epoch 1,500,000 筆，平均分給每個 DataLoader worker（`self.samples // nworkers`），48 個 worker 各 31,250 筆，沒有餘數；但每個 worker 各自組 batch 且 `drop_last=True`，31,250 筆只能組成 312 個滿的 batch（100 筆），每個 worker 丟掉 50 筆，312×48＝14,976 步。官方的 NPROC 8（或 12 個 worker）剛好是 15,000 步。這是 NPROC 從官方的 8 改成 CPU 核數（營運性修改，run 前已 commit、已預告）的副作用，不是算力不夠；每個 worker 的抽樣也因此和官方不同。超參數、batch 100、seed 都沒改 | `player/run_log.json`、`third_party/aicup2026_tutorial/player-identification.ipynb`（cell 11 `self.samples // nworkers`、cell 12 `drop_last=True`） |
| 外洩檢查 | 訓練玩家 6,407 人／793,863 盤，人數盤數 assert 通過，沒有任何一盤出現在本機題目檔與候選庫；索引裡驗證玩家 0 人、不明玩家 0 人 | `player/leak_check.json` |
| 索引 | 6,404 人／793,860 盤：官方 `MIN_GAMES=2` 過濾掉 3 位只有 1 盤的玩家 | `player/leak_check.json` |
| 選用的權重 | `best_model.pth`＝第 5 個 epoch（訓練 loss 最低，照官方做法；訓練結束時 SHA-256 核對通過） | `player/run_log.json` |
| 訓練 loss | 0.4712、0.3947、0.3717、0.3636、0.3566（每個 epoch 平均） | `player/ckpt/history.csv` |
| 推論 | 400 題、查詢 4,935 盤、候選 40,000 盤（400 人），沒有退回預設答案的題目 | `player/predict_summary.json` |
| 結果檔 | `player_val_top5.csv` SHA-256 `0f5761a1…3ec7`，和 Colab 端記錄的一致 | `player/predict_summary.json`、本機計算 |
| 結果 zip | `player_results.zip` SHA-256 `c12071e67e2bb84bb4d5ada1ae3e405a9039d1071cc38ee8dd3c4db9ac332543`（zip 本身不進 git，內容已解開 commit） | 本機計算 |

### 注意事項

- `player/config.json` 的 `prereg` 欄同樣寫「rev 2」，理由同競賽二。
- 訓練 loss 到第 5 個 epoch 還在下降。
