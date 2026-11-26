# 測試日操作手冊：競賽二 59（test59，25 已上傳之後）

- 這份是 Claude 的指令順序，只管競賽二的 59（Q59＝25 的 7 項＋第 8 項 OPP2「對手多強／是誰」）。依據 `docs/PREREG_59_OPPONENT_IDENTITY.md`；兩者不一致時以登記為準。
- **只用 CPU、約 1 分鐘**。不跑 GPU、不開 Colab、不碰 `data/local`、不跑 `tools/score.py`、不讀競賽一。
- 59 用到的東西都已經在本機：25 的正式執行和它綁定的檔（`results/test02|04|07|08|09|19|25/`、`data/test08/p1/`、`data/test/`）、測試棋局對手方的 S0 向量（`data/prereg58/emb_opp_rank/V2_step_320000_raw_tta1/pool_{public,private}.npz`，不進 git，**不要刪**）、官方原始題目檔（`data/test_v2_download/rank/`）、凍結檔 `results/prereg59/rank/`。
- 執行環境：`PY`＝`.venv/Scripts/python.exe`；repo 根目錄、Git Bash，指令前加 `PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8`；**不要設** `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`。
- **K6-59＝11/23 12:00**：過了工具就拒絕執行；沒做完就維持 25。

## 第 1 步：產生交件檔（已在 11/21 做完）

```
PY tools/t59_rank.py apply
```

- 先決條件：git 沒有未 commit 的追蹤檔變更；`results/test59/rank/` 還沒有輸出（要重跑加 `--rerun --reason "原因"`）。
- 演練（不寫進 repo）：`PY tools/t59_rank.py apply --out <repo 外的資料夾>`。
- 結束碼：0 完成；2 拒絕（照訊息修好前提再跑）；4 接縫或檢查不過（停下查，不上傳）。
- 過程會印三個接縫，都要過：
  - 接縫 A：重算的 Q25 機率檔和 `results/test25/rank/rank_test_probs.csv` 逐位元組相同；
  - 接縫 O：對手方向量確實是這份題目檔的棋（來源檔 SHA、5,000 盤、每盤取樣數＝對手落子數減掉前 7 手、S0 的 checkpoint）；
  - 接縫 B：新項權重設 0 時，交件檔和 `results/test25/rank/submission_rank.csv` 逐位元組相同。
- 輸出在 `results/test59/rank/`：`submission_rank.csv`（800 列）、`rank_test_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`（多一個 `prereg59_record`）、`t59_apply_report.json`。做完只 `git add results/test59` 再 commit。

## 第 2 步：上傳前看的東西（主 session）

1. `results/test59/rank/decide_summary.json` 的 `prereg59_record`：
   - `review_needed` 要是 false（檢視線：任何一池 max_s q 平均和 inner 的 0.5803 差超過 0.05，或和 25 不同的題超過 100 題）；是 true 就先查原因。
   - 每池和 25 不同的題數、25／59 的預測等級分布、HD×執色比例。
2. `results/prereg59/explore/CONTROLS.md`（這一項能信到什麼程度）和 `docs/PREREG_59_OPPONENT_IDENTITY.md` 第 7 節（揭露）。
3. `PY tools/check_submission.py rank results/test59/rank/submission_rank.csv --questions data/test/rank_prediction_test_public.csv data/test/rank_prediction_test_private.csv --official`（工具已跑過一次；上傳前可再跑）。
4. 交件檔 SHA-256 前 12 碼和 `t59_apply_report.json` 的 `outputs_sha256.submission_rank.csv` 相同。

## 第 3 步：上傳和最後一版（主 session；本工具不做）

- 照 AGENTS.md 的上傳規則上傳 `results\test59\rank\submission_rank.csv`（競賽二 #11；當天上傳次數 < 5）。`tools/official_log.py` 不認得 59：紀錄由主 session 手寫補充，註明 PREREG_59 第 5 節。
- 最後一版照 PREREG_59 第 5 節：59 的檢查都沒問題就留作最後一版；public 只用來抓格式錯誤或大幅退步，不用來在 25 和 59 之間挑。判定 59 有問題就重傳 25（`results\test25\rank\submission_rank.csv`）當最後一次上傳。
- 同名檔注意：25 和 59 的交件檔都叫 `submission_rank.csv`，上傳前核對資料夾和 SHA。
