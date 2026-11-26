# 測試日操作手冊：競賽二 07（test07，11/04 夜起）

- 這份是 Claude 的指令順序，只管競賽二的 07（Q02H＝Q02＋KataGo HumanSL）。
  - test02 照 `docs/TESTDAY.md`，04 照 `docs/TESTDAY_04.md`，兩份都不改。
  - 07 不用 Colab：HS 在本機 RTX 4090 上算，其他步驟在 CPU。
- 依據：
  - 登記：`docs/PREREG_07_HUMANSL.md` 第 4、5 節、rev 3（採用條件）、rev 4（第 11 節，測試日程序）。
  - 規格：`results/prereg07/design/spec.md` 第 7–9 節。
  - 兩者不一致時，以登記為準。
- **競賽二的最終選擇**：PREREG_07 5.4 取代 PREREG_04 5.3 和 `TESTDAY_04.md` 第 7 步。
  - 有符合的 07 → 交 07。
  - 沒有 → 照 PREREG_04 5.3：交 04；04 不符合就交 R20；T1 判 FAIL_OPENING 時交 R10。
  - 07 判定之前不執行 `official_log.py final --comp rank`。已經設過也沒關係，第 6 步會重設。
- 執行環境：
  - `PY`＝`.venv/Scripts/python.exe`，`GPY`＝`.venv-gpu/Scripts/python.exe`。
  - 一律在 repo 根目錄、用 Git Bash 執行，指令前面加 `PYTHONDONTWRITEBYTECODE=1`。
- **每一步做完就 commit，不論結束碼。**
  - 要 commit 的檔：`hs_<組>_report.json`、`<組>_hs.npz`、`gpu_seam.json`、`seam_val/`、`t07_apply_report.json`、`not_used.json`、`attempt_verdict.json`。
  - 只 `git add results/test07 results/official_scores.json`，不要 `git add -A`。
  - `data/test07/**` 不進 git，它的 SHA 記在報告裡。
- 出現結束碼 3 之後，不重跑任何步驟。要翻盤，只能先寫並 commit PREREG_07 rev 5 起的修訂（指出具體的程式錯誤），再用 `--bug-revision N`。

## 每一步都適用的規則

- **不碰答案**：不開、不讀、不雜湊驗證標籤檔。只有 `hs --set seam` 和 `apply` 會讀 `data/local/rank_val_questions.csv`（題號、slot、執色、SGF），工具自己會擋其他路徑。不跑 `tools/score.py`。
- **凍結檔一律不改**：
  - R20 的 39 個檔、04 的 21 個檔、06 的測試日路徑；
  - `results/prereg01`–`prereg07/**`、`tools/p0*_*.py`、`t02_*`、`t04_*`、`t06_*`、`third_party/katago/**`；
  - 四個釘住檔、PREREG_01–06、打分紀錄帳。
  - `tools/official_log.py`、`docs/PREREG_07_HUMANSL.md` 只准加行。
- **每次開工先跑防誤改閘門**（CLAUDE.md 的指令，加上 `tests/test_t07_hygiene.py`）。
  - 唯一允許的失敗是 `test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`。它失敗的原因是帳上多了 H6P 的 4 列和 H7R 的 2 列，都對得上已 commit 的登記。
  - 接著跑 `PY -m pytest -q tests/test_t07_official_log.py -k TagVersusNew`。
- **上傳**：
  - 每一份、每一次，都要使用者在對話裡對那一份說 OK。
  - 訊息 A 放檢查結果和 OK 請求；訊息 B 自成一則，只放上傳步驟。
  - 同一時間只能有一則等 OK 的訊息，不分競賽。
- **三份交件檔同名 `submission_rank.csv`**：
  - R20 在 `results\test02\rank\`，04 在 `results\test04\rank\`，07 在 `results\test07\rank\`。
  - 訊息 B 只給一個資料夾，寫上 SHA-256 前 12 碼和「不要從另外兩個資料夾拖」。
  - 發訊息 B 之前，先用 `Get-FileHash` 核對。
- 每則訊息 A 之前跑 `PY tools/official_log.py status`，確認競賽二當天上傳次數 < 5。

## `t07_rank.py` 的結束碼

| 碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成 | commit，做下一步 |
| 2 | 拒絕：前提不符。例如 git 不乾淨、R20 的第一次上傳還沒記錄、test02／04 沒做完或沒 commit、舊塊、HS 檔和報告不符、輸出已存在、GPU 型號不對、權重 SHA 不對、GPU 上有別的工作。工具內部錯誤（有 traceback）也是 2 | 照訊息修好前提，再跑同一行。要重寫輸出就加 `--rerun --reason` |
| 3 | 登記的「07 不用」：rev 3 的 H7R 條件不成立、K6-07 已過而且之前沒有成功的執行、T1 FAIL_OPENING、04 判不用、本機接縫超出容許差（04 的接縫和 Q02 都對）、HS 失敗 > 0.5%、方向檢查沒過 | commit `results/test07/not_used.json`；最終照 PREREG_04 5.3；不重跑 |
| 4 | 本機漂移或對不上：`.venv`／`.venv-gpu` 版本、塊的環境、三組環境不同、GPU 接縫、Q02 對不上 04、手數交叉核對、接縫結構或位元、check_submission | commit `rank/attempt_verdict.json`，停下查，查好後 `--rerun --reason`。手數交叉核對不符時，要先 commit 指出程式錯誤的 PREREG_07 修訂，再加 `--bug-revision N` |

## K5-07 核對表（11/03 23:59，目標 10/25）

- 分支 `p07-testday` 已合併回 main。
- 閘門（含 `test_t07_hygiene.py`）只有已知的那一項失敗；`C10_SLOW=1` 的 `tests/test_t07_dryrun.py` 全過。
- PREREG_07 rev 4 已 commit。
- 4090 實測已在 10/5 做過，結果在 `results/test07/design/gpu_rehearsal_*.json`。
- 06 的指紋和 tag `p06-testday-pre07` 已在 10/6 做好。
- 權重 `data/katago/b18c384nbt-humanv0.ckpt` 和備份 `data/katago_backup/` 的 SHA 都是 `84b23476…`。
- 選做：請使用者在 10/6–11/19 暫停 Windows 自動更新顯示卡驅動。

## 日期與 K6-07

- **11/04**：只做 test02。R20 記成 `uploaded` 並 commit 之後，夜間跑 `hs`（約 3–4 小時）。
- **11/05–11/07**：04 照 TESTDAY_04 做。04 的 `apply` 結束碼 0 之後，跑 07 的 `apply`。04 上傳之後（最晚 11/18），才 `prepare` 07。
- **K6-07＝11/19 23:59:59**：`t07_apply_report.json` 最後一次 commit 要在這之前。
- 11/22、11/23 各跑一次 `status`。

## 第 1 步：閘門

1. 跑防誤改閘門（見上）。
2. 跑 `PY tools/official_log.py status`：R20 的第一次上傳要已經記成 uploaded，而且 commit 了。
3. 看 T1（`results/test02/checks/rank/t1.json`），要是 PASS 或 NORMALIZED。
   - FAIL_OPENING：07 不用，`hs` 會寫 `not_used.json`。
   - FAIL_STRUCT：照 test02 停下。
4. 跑 `nvidia-smi`：不能有別的 compute 程序（C+G 的桌面程式不算）。有的話先問使用者，不要自己關掉。

## 第 2 步：`hs`（4090，獨立背景程序，排夜間）

- 先寫 `data/test07/hs_night.sh`（不進 git）。內容只有三行 GPU 指令；前一行結束碼不是 0 就停，後面不接任何算分或判定的程式：

```
#!/bin/bash
set -e; export PYTHONDONTWRITEBYTECODE=1; cd /d/AI-Portfolio/competitions/aicup2026-go
.venv-gpu/Scripts/python.exe tools/t07_rank.py hs --set seam    >> data/test07/hs_night.log 2>&1
.venv-gpu/Scripts/python.exe tools/t07_rank.py hs --set public  >> data/test07/hs_night.log 2>&1
.venv-gpu/Scripts/python.exe tools/t07_rank.py hs --set private >> data/test07/hs_night.log 2>&1
```

- 用 PowerShell 啟動：`Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/d/AI-Portfolio/competitions/aicup2026-go/data/test07/hs_night.sh" -WindowStyle Hidden`
- 看進度：`PY tools/t07_rank.py status`。
- 讓出 GPU：在 `data/test07/hs/<組>/` 放一個空檔 `STOP`，跑完手上那一塊就會停。要續跑時，先刪 STOP，再跑同一行（這是續跑，不是重跑）。
- 中途斷電或重新開機：重下同一行，會從沒寫完的那一塊接著跑。
- 結束碼 2「written before R20's first upload」：那個資料夾是 10 月或演練留下的。改名成 `<組>_stale_<時間>` 再跑，不要刪。
- 每一組寫好 `results/test07/hs/<組>_hs.npz` 和 `hs_<組>_report.json` 就 commit。三組也可以早上一起 commit。

## 第 3 步：等 04 的 `apply`

- 要等 TESTDAY_04 第 5 步的 `t04_apply_report.json` 結束碼 0，而且已 commit。
- 04 判不用（opp 或 apply 結束碼 3）→ 07 也不用，07 的 `apply` 會回 3。

## 第 4 步：`apply`（CPU）

- 跑 `PY tools/t07_rank.py apply`，然後 commit `results/test07/`。
- 看報告裡的這幾項：
  - GPU 接縫的三個最大差；
  - Q02 對得上 04；
  - 手數交叉核對 0；
  - 本機接縫的判定；
  - 兩組的失敗數和 ρ；
  - 07 和 04 在 public 題、規則 B 下不同的題數。

## 第 5 步：上傳 07（兩則訊息）

1. 04 已記成 uploaded 之後，跑：
   `PY tools/official_log.py prepare --comp rank --file results/test07/rank/submission_rank.csv --reason "prereg07 Q02H (PREREG_07 5.4)"`
   prepare 會重算 5.4 的每一條。被拒絕就照訊息處理，不繞過。
2. **訊息 A**：放檢查結果（照 prepare 印出的 OK 請求），並請使用者對這一份說 OK。
3. 使用者說 OK 之後，**訊息 B** 自成一則：在 AIdea 競賽二（TEAM_11248）上傳 `results\test07\rank\submission_rank.csv`，寫上 SHA 前 12 碼，並說明不要從 test02 或 test04 的資料夾拖。
4. 使用者回報後，跑 `uploaded --id N --ok-at … --uploaded-at …` 和 `public --id N --score …`，然後 commit。
- public 分數不進判定。
  - 07 比 04 低 0.05 以上時，只查錯（ID、列順序、輸入綁定）。查不到錯就維持 5.4 的結果，並寫進報告。
  - `public` 印出同檔警告時，請使用者核對 AIdea 的上傳紀錄。

## 第 6 步：最後一版（PREREG_07 5.4；11/23 23:59 前）

- 有符合的 07：`final --comp rank --id <07> --basis "PREREG_07 5.4"`。
- `status` 說 07 不再符合：照 PREREG_04 5.3 改 final（交 04，不行就交 R20），再 prepare 並重傳那一份，讓它成為最後一次上傳。
- 要改交 04／R20，只能靠 PREREG_07 rev 5 起的修訂，加 `--not-07-reason "PREREG_07 rev N: …"`。要交 R20、而 04 也符合時，兩個 reason 都要給。

## 第 7 步：結果

寫 `results/test07/RESULTS.md`，內容包括：
- 閘門結果；
- HS 的速度、GPU 使用時間、驅動版本；
- GPU 接縫的三個值、Q02 對得上 04、手數交叉核對、本機接縫；
- 失敗率、方向 ρ；
- 07 和 04 不同的題數（public／全部）；
- 上傳紀錄、5.4 的判定；
- 看過 public 分數之後做的選擇（應該沒有）。

## 4090 不能用時

不登記 CPU 或 Colab 的備案（PREREG_07 5.3）。K6-07 前修不好，這一季就不用 07，最終照 PREREG_04 5.3。

## test02 或 04 重跑了

- 07 綁住了 test02 和 04 的輸入（inputs_bound）。它們一變，07 的紀錄就不再符合。
- 照 TESTDAY／TESTDAY_04 處理完之後，07 只能先寫修訂，再 `apply --rerun --reason … --bug-revision N`。
