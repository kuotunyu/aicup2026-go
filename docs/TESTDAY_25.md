# 測試日操作手冊：競賽二 25（test25，19 的正式執行之後）

- 這份是 Claude 的指令順序，只管競賽二的 25（Q25＝Q19 的 SET 項換成 S25：S19 的 set 模型加上驗證玩家重訓）。
  - test02、04、07、08、09、19 照各自的 `docs/TESTDAY*.md`，那幾份都不改；14 已記錄不用（`results/test14/not_used.json`），不跟著 25。
  - **25 只用 CPU、只要十幾秒**：讀 08 在 4090 上算好、09 和 19 也讀過的 P1 塊（`data/test08/p1/`），不跑 GPU、不開 Colab。
- 依據：`docs/PREREG_25_SET_MORE_PLAYERS.md` 第 3、4、5 節、rev 3（測試日程序）；凍結和採用紀錄 `results/prereg25/rank/`（`frozen_config.json`、`freeze_audit.json`、`adoption.json` usable）。兩者不一致時，以登記為準。
- **25 用了驗證玩家訓練、沒在驗證集打過分**（PREREG_25 第 3 節）。不影響測試日的做法，報告要寫明。
- **競賽二的最終選擇**：PREREG_25 4.3 取代 PREREG_19 4.3 的「有符合的 19 → 19」。
  - 有符合的 25 → 交 25。
  - 沒有 → 照 PREREG_19 4.3（`docs/TESTDAY_19.md` 第 5 步：19 → 09 → 08 → 07 → 04 → R20，T1 FAIL_OPENING 時 R10）。
- 執行環境：`PY`＝`.venv/Scripts/python.exe`；repo 根目錄、Git Bash，指令前加 `PYTHONDONTWRITEBYTECODE=1`；**不要設** `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`。
- **每一步做完就 commit，不論結束碼**：只 `git add results/test25 results/official_scores.json`，不要 `git add -A`。`results/test0*/`、`results/test19/`、`data/test08/` 都**不要動**（25 會重算它們的 SHA）。
- 出現結束碼 3 之後，不重跑任何步驟。要翻盤，只能先寫並 commit PREREG_25 rev 4 起的修訂（指出具體的程式錯誤），再用 `--bug-revision N`。

## 每一步都適用的規則

- **不碰 `data/local`**：25 的工具不開、不讀、不列、不雜湊 `data/local` 底下的任何東西（不跑 seam_val、不跑 09／08／07／04／test02 的前提鏈；改成綁定 19 已 commit 的正式執行，並在 public／private 上重算 Q09、Q19）。不跑 `tools/score.py`。
- **凍結檔一律不改**：`results/prereg25/rank/**`、`results/prereg*`、`results/test02|04|07|08|09|19/**`、各測試日路徑、`tools/t0*`、`tools/t19_rank.py`、`tools/p*`、`tools/t25_rank.py`。`tools/official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_25_SET_MORE_PLAYERS.md` 只准加行。
- **每次開工先跑防誤改閘門**（AGENTS.md 的閘門指令，含 `tests/test_p25_hygiene.py`、`tests/test_t25_hygiene.py`）；接著跑 `PY -m pytest -q tests/test_t25_official_log.py -k TagVersusNew`。
- **上傳**：照 AGENTS.md「和使用者合作的規則」當時的規定（11/04 起是常設同意：檢查全過後由 Claude 上傳；主 session 另有指示時照指示）。
- **七份交件檔同名 `submission_rank.csv`**：R20 `results\test02\rank\`、04、07、08、09、19 各自的 `results\test0X\rank\`／`results\test19\rank\`、**25 `results\test25\rank\`**。上傳時只用 25 的資料夾，先核對 SHA-256 前 12 碼。
- 上傳前跑 `PY tools/official_log.py status`，確認競賽二當天上傳次數 < 5。

## `t25_rank.py` 的結束碼

| 碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成 | commit，做下一步 |
| 2 | 拒絕：前提不符。例如 git 不乾淨、prereg25 或 prereg19 的釘住檔變了、19 的正式執行沒做完、沒 commit、判不用，或它綁定的輸入、它的輸出被改過；08 的 P1 報告或塊和 19 記的不同、輸出已存在、設了 BLAS 執行緒變數、K6-25 過後已有成功的那一次。工具內部錯誤（有 traceback）也是 2 | 照訊息修好前提，再跑同一行；要重寫輸出加 `--rerun --reason` |
| 3 | 登記的「25 不用」：25 沒有採用紀錄、K6-25 已過而且之前沒有成功的執行、SET25 方向檢查沒過 | commit `results/test25/not_used.json`；不重跑；25 不 prepare、不上傳；最終照 PREREG_19 4.3（`docs/TESTDAY_19.md` 第 5 步） |
| 4 | 本機漂移或對不上：`.venv` 版本、Q09 重算對不上 `results/test09/rank/`、Q19 重算對不上 `results/test19/rank/`（`rank_test_probs.csv` 或 `test_lp_set.npz`）、check_submission | commit `rank/attempt_verdict.json`，停下查，查好後 `--rerun --reason` |

## 日期

- **K5-25＝11/14 23:59**：套件以 `--no-ff` 合併回 main，閘門只有 AGENTS.md 列的允許失敗，`C10_SLOW=1` 的 `tests/test_t25_dryrun.py` 全過，PREREG_25 rev 3 已 commit。
- **K6-25＝11/18 23:59:59**：`t25_apply_report.json` 最後一次 commit 要在這之前；上傳目標也是 11/18 前。
- 11/22、11/23 各跑一次 `status`。

## 第 1 步：閘門

1. 跑防誤改閘門（見上）。
2. `PY tools/official_log.py status`：R20 的第一次上傳、09、19 都已記成 uploaded 並 commit。
3. T1（`results/test02/checks/rank/t1.json`）要是 PASS 或 NORMALIZED。

## 第 2 步：確認 19 的正式執行

- `results/test19/rank/t19_apply_report.json` 結束碼 0、usable、已 commit；19 已上傳（`official_scores.json` 的 19 那一筆 status uploaded）。
- 19 判不用或沒做完 → 25 不能跑（2）。

## 第 3 步：`apply`（CPU）

- 跑 `PY tools/t25_rank.py apply`，然後 commit `results/test25/`。
- 看 `results/test25/rank/t25_apply_report.json`：
  - `q09_vs_09.equal`、`q19_vs_19.equal`（預期 `byte_equal` 都是 true）：Q09、Q19 由 25 重算，和 09、19 已 commit 的檔相同；`q19_vs_19.lp_set19.equal`；
  - `set_direction`：兩組的 ρ（門檻 = max(0.5 × ρ_mono25, 0.155)）；
  - `set_seam_record_only`：SET25 在 seam 400 盤的最大差（**只記錄**）；
  - `rule_b_differences`：25 和 19／09／08／07／04／R20 在 public、private 題、規則 B 下不同的題數（只記錄）。
- **Q09 或 Q19 對不上（結束碼 4）**：本機問題（環境、檔案、程式），不是 25 的判定；查好後 `--rerun --reason`。

## 第 4 步：prepare 和上傳

1. 跑：
   `PY tools/official_log.py prepare --comp rank --file results/test25/rank/submission_rank.csv --reason "prereg25 Q25 (PREREG_25 4.3)"`
   prepare 會重算 4.2／4.3 的每一條（25、19、09 的採用紀錄、H8R、Q09／Q19 對得上、SET25 方向、T1、綁定的輸入、K6-25、R20 和綁定的 19 已上傳而且 19 仍符合 PREREG_19 4.3）。被拒絕就照訊息處理，不繞過。commit `results/official_scores.json`。
2. 上傳 `results\test25\rank\submission_rank.csv`（AIdea 競賽二 TEAM_11248），核對 SHA 前 12 碼。
3. 上傳後：`uploaded --id N --ok-at … --uploaded-at …`、`public --id N --score …`，commit。
- public 分數不進判定。25 比 19 低 0.05 以上時只查錯（ID、列順序、輸入綁定），查不到就維持 4.3 的結果並寫進報告；`public` 印出同檔警告時，核對 AIdea 的上傳紀錄。

## 第 5 步：最後一版（PREREG_25 4.3；11/23 23:59 前）

- 有符合的 25：`PY tools/official_log.py final --comp rank --id <25> --basis "PREREG_25 4.3"`。最後一次上傳要是 25。
- `status` 說 25 不再符合，或 25 判不用（結束碼 3）、從沒 prepare：照 PREREG_19 4.3 改 final（`docs/TESTDAY_19.md` 第 5 步）；最後一次上傳不是那一份時，再 prepare 並重傳。
- 要改交 19／09／08／07／04／R20，只能靠 PREREG_25 rev 4 起的修訂，加 `--not-25-reason "PREREG_25 rev N: …"`（19 也符合而要交更舊的，`--not-19-reason` 也要給，依此類推）。

## 第 6 步：結果

寫 `results/test25/RESULTS.md`：閘門結果；Q09／Q19 對得上、SET25 方向 ρ、SET25 的 GPU 接縫紀錄；25 和 19／09 不同的題數（public／private）；上傳紀錄、4.3 的判定；看過 public 分數之後做的選擇（應該沒有）；「用了驗證玩家訓練、沒在驗證集打過分、沒有驗證題 apply」的揭露（PREREG_25 第 3、8 節）。

## 19 或更早的測試日輸出重跑了

- 25 綁住了 19 的報告、輸出和它綁定的所有輸入（inputs_bound）。它們一變，25 的紀錄就不再符合。照各自的手冊處理完之後，25 只能先寫修訂，再 `apply --rerun --reason … --bug-revision N`。

## 演練（審查用；不寫進 repo）

- `C10_SLOW=1 PY -m pytest -q -p no:cacheprovider tests/test_t25_dryrun.py`：`--dry-run` 讀 repo 裡真正的 test02–test19 輸出（唯讀），輸出寫到 repo 外的暫存資料夾；另有負面演練（把 test19 的部分檔複製出來改壞）。
- 暫存資料夾不要放在路徑含 `AppData\Local` 的地方（`data\local` 防護會誤擋）：用 `C10_DRYRUN_DIR=<subst 的磁碟>`，例如 scratchpad 裡的資料夾 `subst R:` 後用 `R:\`。
