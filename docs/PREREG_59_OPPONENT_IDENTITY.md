# 事前登記 59：競賽二在 Q25 上加第 8 項「對手是誰／對手多強」（Q59）

- 競賽二（棋力預測，TEAM_11248）。只用 CPU；不碰 `data/local`、不跑 `tools/score.py`、不用競賽一測試資料、不用任何 public 分數做選擇。
- 探索紀錄（都已 commit，本登記不改它們）：`results/prereg59/explore/` 的 `PLAN.md`（`21a9668`）、`NOTES.md`＋程式（`2938065`、`3cddff7`）、`rows.json`（`6f32507`）、`seam.json`（`dd6a318`）、`kernel.json`（`aed895b`）、`eval.json`（`0419d54`，**inner 判一次**）、`RESULTS.md`（`5eaf905`）、對照檢查 `CONTROLS_PLAN.md`（`f972729`）→ `controls.json`／`CONTROLS.md`（`2ccf433`）。
- 本登記寫在判定和對照檢查之後、凍結融合參數和任何測試題輸出之前。

## 修訂紀錄

| 版 | commit | 內容 |
|---|---|---|
| rev 1 | （本檔第一次 commit） | 第一版，2026-11-21。commit 時：inner 已判過一次（過）；對照檢查已完成；`results/prereg59/rank/`、`results/test59/`、`tools/t59_rank.py` 都不存在；沒有任何 8 項融合的凍結參數，沒有任何測試題的第 8 項數值。 |

## 1. 方法（照探索原樣，不再調）

- 基底＝已上傳的 25（Q25：`results/prereg25/rank/frozen_config.json` 的 7 項 OP、O4、SWA、SWB、HS、P1TO、SET 和它們的模型、對映）。這 7 項的算法、模型檔一律不變。
- 第 8 項 OPP2（`p59_lib.py`）：
  1. 每盤取**對手那一方**的 S0 向量（S0＝同一作者競賽一的認人網路 prereg43 種子 0，320,000 步，raw、不做 TTA；只用官方訓練棋譜訓練，3,685 位訓練玩家，沒有 inner 玩家、沒有驗證玩家）。
  2. 類別＝S0 分類頭的 3,685 列（logit＝50 × cos）＋2,500 位「S0 沒看過的有標籤玩家」的原型（D 300、E 600、split v1 驗證玩家 1,600；原型＝該玩家全部訓練棋局的 S0 己方向量平均；logit＝30 × cos＋2）。inner 玩家和測試玩家沒有原型。對全部 6,185 類做 softmax，依類別的等級加總成 q_g[s]（10 級）。
  3. 條件表 M[h, c][r][s]＝P(對手類別等級 s｜玩家等級 r、HD 組 h（0／1／2 以上）、玩家執色 c)，在 900 位核玩家（D、E）上用 q 做軟計數、每列加 0.5 平滑。τ1＝50、τ2＝30、b＝2 是 60 個格點裡留出概似最大的（`kernel.json`，在任何 inner 數字之前凍結）。
  4. 每盤 l_g(r)＝log(Σ_s q_g[s]·M[h_g, c_g][r][s]＋1e-9)；每題的項＝10 × 該題各盤 l_g 的平均，當作融合的第 8 個 E 項。
- 決策規則同 25：規則 B（`tools/decide.py`）。

## 2. 凍結（`results/prereg59/explore/p59_freeze.py`，只跑一次）

- 做法比照 PREREG_25 2.1：在**全部 inner fit 題**上對 8 項的 θ 做一次擬合（`p04_rank_fuse.gen_fit_record`，和 25 同一個函式、同樣的起點和上下限）。
- 接縫（任何一個不過就停）：
  - Q25 的 inner OOF 經 prereg58 的路徑逐位元重現（0.7456519871546352／單色 0.7482450052195216）；
  - **Z**：同一個呼叫只放 7 項時，得到的 θ 和 `results/prereg25/rank/frozen_config.json` 的 θ 逐位元相同（＝新項權重為 0 的情形就是凍結的 Q25）；
  - **T**：inner 題棋局的 max_s q 平均和 `eval.json` 記的相同（0.5803…），題內棋局數相同。
- **不計算任何新的 inner 分數**：不重跑含新項的 OOF、不算擬合後的指標；判定只有 `eval.json` 那一次。
- 輸出 `results/prereg59/rank/`：`frozen_config.json`（θ、接縫、釘住的 SHA）、`prototypes.npz`、`s0_head.npz`（判定時用的兩個檔的逐位元組副本，讓測試日工具只讀 git 裡的檔）。commit 之後一律不改。

## 3. 判定（已完成，只此一次）和對照檢查

| | Q25 | Q59 | 差 | 95% 區間 | 閘門 |
|---|---|---|---|---|---|
| inner eval | 0.7456520 | 0.7539702 | +0.00832 | [+0.00150, +0.01538] | G1（≥ +0.005）過、G2（下界 > 0）過 |
| 單色 | 0.7482450 | 0.7552848 | +0.00704 | [+0.00293, +0.01082] | G4 過（黑 +0.0054、白 +0.0087） |

- 新項的權重 5 折 0.66–1.31（上限 4.0，沒有一折卡上限或為 0）。依等級：D +0.027、B +0.028、6D +0.015，3D −0.007。剛好 20 盤的題 −0.0026 [−0.0150, +0.0111]，5 盤題 +0.0107。
- 對照檢查（只在 900 位核玩家上，交叉擬合；只記錄，`CONTROLS.md`）：這一項單獨 0.58–0.68（先驗 0.15）；每盤給同一個 q 時正好等於先驗（不是選參數目標造成的假訊號）；同級內打亂對手不降反升（+0.04～+0.09）；把 6,185 個人換成 10 個等級中心仍保留 92–98% 的增益；拿掉每盤最像的那一類保留 82–87%。**結論：這一項主要讀的是「對手看起來多強」，認出特定的人只占一到兩成**；選到的格點在正規化目標下是 60 個裡的第 3。
- 這些單獨分數不是融合增益；融合增益只有上表那一次 inner 判定。

## 4. 測試日（`tools/t59_rank.py apply`；CPU、不碰 `data/local`）

- 輸入：
  - 25 的正式執行和它綁定的一切（經 `tools/t25_rank.py` 的函式重算 public、private 的 7 項；**接縫 A**：重算的 Q25 機率檔和 `results/test25/rank/rank_test_probs.csv` 逐位元組相同）；
  - 測試棋局對手方的 S0 向量 `data/prereg58/emb_opp_rank/V2_step_320000_raw_tta1/pool_{public,private}.npz`（11/20 在 4090 上算好；來源檔 `data/prereg56/src_rank/` 的題目檔和官方 `data/test_v2_download/rank/rank_prediction_test_{public,private}.csv` SHA-256 相同、候選檔 0 列，所以向量順序＝題目照檔案順序、題內照 sgf_1…sgf_n；工具會重驗 SHA、盤數、checkpoint SHA）；
  - HD 和執色：官方原始 CSV（每盤開頭的 `HD[n]`、`color_i`）；執色要和 test02 的 `val_keys` 相同。
  - `results/prereg59/rank/` 的凍結檔、`results/prereg59/explore/kernel.json`（都釘 SHA）。
- 第 8 項：每盤 l_g 照第 1 節（測試玩家沒有原型，不排除任何類別）；每題＝10 × 該題**已解析棋局**的 l_g 平均（和 Q25 的 m 用同一個平均矩陣）。
- **接縫 B**：把凍結的 Q25 θ 補上「新項權重 0」走 8 項的路徑，產生的交件檔和 `results/test25/rank/submission_rank.csv` 逐位元組相同。
- 輸出 `results/test59/rank/`：`rank_test_probs.csv`、`pred_A.csv`、`pred_B.csv`、`submission_rank.csv`（＝`pred_B.csv`，800 列）、`decide_summary.json`、`t59_apply_report.json`；`check_submission --official` 要過。
- `decide_summary.json` 另外記錄（只記錄）：每池和 25 不同的題數、25 和 59 的預測等級分布、測試棋局 max_s q 的平均（對照 inner 的 0.5803）、每池各 HD×執色組的題數比例、新項單獨的期望等級和 Q25 期望等級的 Spearman ρ。
- **檢視線（寫在看到任何測試題數值之前；不自動判定）**：任何一池的 max_s q 平均和 0.5803 相差超過 0.05，或任何一池和 25 不同的題超過 25%（100 題），主 session 上傳前要先查原因。
- 結束碼：0 完成；2 拒絕（前提不符、輸出已存在）；4 接縫或檢查不過。K6-59＝**2026-11-23 12:00**（PLAN 的硬截止）：過了不再產生或重跑。

## 5. 上傳和最後一版（競賽二）

- 59 只上傳一次（競賽二 #11），由主 session 照 AGENTS.md 的上傳規則做；本登記的工具不上傳、不呼叫 `official_log.py`。
- **最後一版**：59 上傳後，檢查都沒問題就留作最後一版；public 分數只用來抓「格式錯誤或大幅退步」這類明顯的問題，不用來在 25 和 59 之間挑選。判斷由主 session 做並寫進紀錄；判定 59 有問題時，重傳 25（`results/test25/rank/submission_rank.csv`）作為最後一次上傳。
- 沒有在 11/23 12:00 前完成第 4 節，或接縫不過：競賽二最後一版維持 25（PREREG_25 4.3）。
- `tools/official_log.py` 的 25 規則（PREREG_25 4.3 的「有符合的 25 → 25」）不改；59 成為最後一版時，紀錄由主 session 依本節手寫補充並註明本登記。

## 6. 不改的東西

- `results/prereg01`–`prereg58/**`、`results/test02`–`test25/**`、`tools/t*`、`tools/p*`、既有的 `docs/PREREG_*`、`docs/TESTDAY*`：一律不改（只 import）。`.gitattributes` 只加一行（`results/test59/** -text`，讓交件檔逐位元組保存）。
- 競賽一不讀、不動。

## 7. 揭露（寫進 RESULTS 和競賽二報告）

1. **設計在看過 prereg58 的 inner 判定之後**（prereg58：+0.0006，沒過；兩端等級退、權重卡上限）；59 針對這三點改。**inner 被重複使用**（58 一次、59 一次，之前 19–44 也用過），數字偏樂觀。
2. NOTES 第 14 點：第一次 `eval --smoke` 讓真實的 HD×執色經過試跑用的表，看到了一個不該看的 inner 數字（inner +0.0006、單色 −0.0012）；格點、選擇規則、尺度、閘門都在那之前 commit（`2938065`），沒有因此更動。
3. **認人網路 S0 是同一作者競賽一的網路**（prereg43），只用官方訓練棋譜訓練，沒有 inner、驗證玩家；競賽一的測試資料完全沒有用在競賽二。兩場報告註明同一作者、前處理和這個網路共用。
4. **驗證玩家**（split v1 的 1,600 位）的原型用了他們的訓練棋譜和訓練標籤（和 25、prereg49 相同的用法）；競賽二沒有再在驗證集打分。
5. 官方 Q&A Q15（`data/reference/qa_q15_20261120.md`）：不限制資料的使用，報告詳述來源及作法。本項沒有用測試資料訓練；只在推論時讀測試棋局。
6. 對照檢查的結論（第 3 節）：這一項主要是「對手看起來多強」，名稱裡的「身分」言過其實；S0 的輸入盤面含雙方棋子，對手方向量也可能帶有玩家自己的棋力，和 Q25 既有的項重疊，融合增益以 inner 那一次為準。
7. 59 沒有在驗證集打過分（競賽二的打分次數已用完）；τ1、τ2、b 和單獨分數用同一批折。
