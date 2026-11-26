# 測試日操作手冊：競賽二 08（test08，07 的 apply 之後）

- 這份是 Claude 的指令順序，只管競賽二的 08（Q08＝Q02H＋4·P1TO）。
  - test02 照 `docs/TESTDAY.md`，04 照 `docs/TESTDAY_04.md`，07 照 `docs/TESTDAY_07.md`，三份都不改。
  - 08 不用 Colab：P1 特徵在本機 RTX 4090 上算，其他步驟在 CPU。
- 依據：
  - 登記：`docs/PREREG_08_STRENGTH_PROBE.md` 第 3、4、5 節和 rev 3（第 13 節，測試日程序）。
  - 兩者不一致時，以登記為準。
- **競賽二的最終選擇**：PREREG_08 5.4 取代 PREREG_07 5.4 的「有符合的 07 時最終是 07」。
  - 有符合的 08 → 交 08。
  - 沒有 → 照 PREREG_07 5.4：交 07；07 不符合就照 PREREG_04 5.3 交 04，再不行交 R20；T1 判 FAIL_OPENING 時交 R10。
  - 08 判定之前不執行 `official_log.py final --comp rank`。已經設過也沒關係，第 6 步會重設。
- 執行環境：
  - `PY`＝`.venv/Scripts/python.exe`，`GPY`＝`.venv-gpu/Scripts/python.exe`。
  - 一律在 repo 根目錄、用 Git Bash 執行，指令前面加 `PYTHONDONTWRITEBYTECODE=1`。
  - **不要設** `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`：`apply` 會拒絕（PREREG_08 rev 2 第 12 節第 2 條；位元組接縫要用預設執行緒）。
- **每一步做完就 commit，不論結束碼。**
  - 要 commit 的檔：`results/test08/p1/<組>_p1.npz`、`p1_<組>_report.json`、`gpu_seam.json`、`results/test08/rank/`（含 `seam_val/`、`t08_apply_report.json`、`test_lp_p1.npz`）、`not_used.json`、`attempt_verdict.json`。
  - 只 `git add results/test08 results/official_scores.json`，不要 `git add -A`。
  - `data/test08/**`（P1 的塊，含每盤 770×2 維的平均）不進 git，每塊的 SHA 記在 `p1_<組>_report.json`，`apply` 會逐塊重算。塊**不要刪**，到 11/25 以後都要留著。
- 出現結束碼 3 之後，不重跑任何步驟。要翻盤，只能先寫並 commit PREREG_08 rev 4 起的修訂（指出具體的程式錯誤），再用 `--bug-revision N`。

## 每一步都適用的規則

- **不碰答案**：不開、不讀、不雜湊驗證標籤檔。只有 `p1 --set seam` 和 `apply` 會讀 `data/local/rank_val_questions.csv`（題號、slot、執色、SGF），工具自己會擋其他路徑。不跑 `tools/score.py`。
- **凍結檔一律不改**：
  - R20 的 39 個檔、04 的 21 個檔、06 的測試日路徑、07 的測試日路徑（`results/test07/design/p07_path_fingerprint.json`，tag `p07-testday-pre08`）；
  - `results/prereg01`–`prereg08/**`、`tools/p0*_*.py`、`t02_*`、`t04_*`、`t06_*`、`t07_*`、`third_party/katago/**`；
  - 四個釘住檔、PREREG_01–06、打分紀錄帳。
  - `tools/official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_07_HUMANSL.md`、`docs/PREREG_08_STRENGTH_PROBE.md` 只准加行。
- **每次開工先跑防誤改閘門**：合併之後是十二個檔＝main 的 CLAUDE.md 閘門指令（已含 `tests/test_t07_path_hygiene.py`）再加 `tests/test_t08_hygiene.py`。
  - 唯一允許的失敗是 `test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`（帳上多了 H6P 4 列、H7R 2 列、H8R 2 列，都對得上已 commit 的登記）。
  - 接著跑 `PY -m pytest -q tests/test_t08_official_log.py -k TagVersusNew`。
- **上傳**：
  - 每一份、每一次，都要使用者在對話裡對那一份說 OK。
  - 訊息 A 放檢查結果和 OK 請求；訊息 B 自成一則，只放上傳步驟。
  - 同一時間只能有一則等 OK 的訊息，不分競賽。
- **四份交件檔同名 `submission_rank.csv`**：
  - R20 在 `results\test02\rank\`，04 在 `results\test04\rank\`，07 在 `results\test07\rank\`，08 在 `results\test08\rank\`。
  - 訊息 B 只給一個資料夾，寫上 SHA-256 前 12 碼和「不要從另外三個資料夾拖」。
  - 發訊息 B 之前，先用 `Get-FileHash` 核對。
- 每則訊息 A 之前跑 `PY tools/official_log.py status`，確認競賽二當天上傳次數 < 5。

## `t08_rank.py` 的結束碼

| 碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成 | commit，做下一步 |
| 2 | 拒絕：前提不符。例如 git 不乾淨、R20 的第一次上傳還沒記錄、07 的 apply 沒做完或沒 commit、舊塊、P1 檔和報告不符、塊被改過、輸出已存在、GPU 型號不對、權重 SHA 不對、GPU 上有別的工作、07 綁的輸入被改過、設了 BLAS 執行緒變數。工具內部錯誤（有 traceback）也是 2 | 照訊息修好前提，再跑同一行。要重寫輸出就加 `--rerun --reason` |
| 3 | 登記的「08 不用」：H8R 不成立、K6-08 已過而且之前沒有成功的執行、T1 FAIL_OPENING、07 判不用（連帶 04 判不用）、本機接縫超出容許差（07 的接縫成立時）、P1 失敗 > 0.5%、方向檢查沒過 | commit `results/test08/not_used.json`；最終照 PREREG_07 5.4；不重跑 |
| 5 | 只有 `p1`：被要求停下（STOP 檔或 `--stop-after-min`），那一組還沒算完；不判定任何事，已寫好的塊留著 | 不 commit 什麼。要續跑：刪 STOP，再跑同一行（續跑，不是重跑）。第 3 步的 `p1_run.sh` 遇到非 0 就停，不會接著開下一組 |
| 4 | 本機漂移或對不上：`.venv`／`.venv-gpu` 版本、塊的環境、三組環境不同、GPU 接縫、Q02H 對不上 07、手數交叉核對、接縫結構或位元、check_submission | commit `rank/attempt_verdict.json`，停下查，查好後 `--rerun --reason`。手數交叉核對不符時，要先 commit 指出程式錯誤的 PREREG_08 修訂，再加 `--bug-revision N` |

## K5-08 核對表（10/31 23:59）

- 分支 `p08-testday` 已合併回 main。
- 閘門（十二個檔：main 的 CLAUDE.md 那十一個，已含 `test_t07_path_hygiene.py`，加 `test_t08_hygiene.py`）只有已知的那一項失敗；`C10_SLOW=1` 的 `tests/test_t08_dryrun.py` 全過（R20／04／07 的輸出等於 `results/test07/design/t07_fake_golden.json`）。
- PREREG_08 rev 3 已 commit。
- 4090 實測做過，結果在 `results/test08/design/gpu_rehearsal_*.json`（主 session 跑 `data/prereg08/gpu_rehearsal_t08.sh`，見 `results/test08/design/build_review.md`）。
- 權重 `data/katago/b18c384nbt-humanv0.ckpt` 和備份 `data/katago_backup/` 的 SHA 都是 `84b23476…`。

## 日期與 K6-08

- **11/04**：只做 test02。之後照 TESTDAY_07 跑 07 的 HS（夜間）。
- **11/05–11/07**：04 照 TESTDAY_04；04 的 `apply` 結束碼 0 之後跑 07 的 `apply`。
- **07 的 `apply` 結束碼 0 並 commit 之後**，才跑 08 的 `p1`（PREREG_08 5.2 的前提）。約 0.55–0.65 小時，不到一小時，白天也可以跑，但開跑前照樣 `nvidia-smi`，並告訴使用者。PREREG_08 第 10 節寫的「07 的 HS 之後同一晚」因為這個前提做不到，以 5.2 為準（rev 3 第 13 節）。
- 07 上傳之後（最晚 11/18），才 `prepare` 08。
- **K6-08＝11/19 23:59:59**：`t08_apply_report.json` 最後一次 commit 要在這之前。
- 11/22、11/23 各跑一次 `status`。

## 第 1 步：閘門

1. 跑防誤改閘門（見上）。
2. 跑 `PY tools/official_log.py status`：R20 的第一次上傳要已經記成 uploaded，而且 commit 了。
3. 看 T1（`results/test02/checks/rank/t1.json`），要是 PASS 或 NORMALIZED。FAIL_OPENING：08 不用（07 也不用）。
4. 跑 `nvidia-smi`：不能有別的 compute 程序（C+G 的桌面程式不算）。有的話先問使用者，不要自己關掉。

## 第 2 步：等 07 的 `apply`

- 要等 TESTDAY_07 第 4 步的 `results/test07/rank/t07_apply_report.json` 結束碼 0，而且已 commit。
- 07 判不用（`results/test07/not_used.json`）→ 08 也不用，`p1` 和 `apply` 都會回 3。

## 第 3 步：`p1`（4090，獨立背景程序）

- 先寫 `data/test08/p1_run.sh`（不進 git）。內容只有三組的 GPU 指令；已經寫好報告的組跳過（所以中斷後重跑同一支就是續跑），任何一組結束碼不是 0（含 STOP 的 5）就整支停，不開下一組；後面不接任何算分或判定的程式：

```
#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1; unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
cd /d/AI-Portfolio/competitions/aicup2026-go
for s in seam public private; do
  [ -f results/test08/p1/p1_${s}_report.json ] && continue
  .venv-gpu/Scripts/python.exe tools/t08_rank.py p1 --set $s >> data/test08/p1_run.log 2>&1 || { echo "p1 $s exit $? $(date -u +%FT%TZ); p1_run.sh stopped" >> data/test08/p1_run.log; exit 1; }
done
echo "p1_run.sh done $(date -u +%FT%TZ)" >> data/test08/p1_run.log
```

- 用 PowerShell 啟動：`Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/d/AI-Portfolio/competitions/aicup2026-go/data/test08/p1_run.sh" -WindowStyle Hidden`
- 估計：seam 400 盤＋public／private 約 10,000 盤（每組 400 題、5／10／15／20 盤各 100 題＝5,000 盤；PREREG_08 5.2），共約 10,400 盤，每盤約 228 個局面，用 val 實測的 1,050–1,200 positions/s 算，約 0.55–0.65 小時。
- **跑的時候不要改任何已追蹤的檔**（不改文件、不 commit 一半的修改、不切分支）：正式模式的 `p1` 每一組開頭都檢查 git 乾淨，seam 跑完後若有已追蹤的檔被改，public 會結束碼 2、整支停下。新產生的未追蹤檔（例如 `results/test08/p1/seam_p1.npz`）不算。真的停了：把修改 commit 或還原，再用同一行 `Start-Process` 重跑（跳過已完成的組、從沒寫完的塊接著跑）。
- 看進度：`PY tools/t08_rank.py status`（每塊一行 JSON 也寫在 `data/test08/p1_run.log`）。
- 讓出 GPU：在**正在跑的那一組**的 `data/test08/p1/<組>/` 放一個空檔 `STOP`（不確定是哪一組，就三個資料夾都放），跑完手上那一塊就會以結束碼 5 停下，`p1_run.sh` 跟著停、不開下一組，GPU 就空出來。用 `nvidia-smi` 確認沒有 python 程序了。要續跑時，先刪掉所有 STOP，再用同一行 `Start-Process` 重跑 `p1_run.sh`（這是續跑，不是重跑）。
- 中途斷電或重新開機：重下同一行，已完成的組跳過，從沒寫完的那一塊接著跑。
- 結束碼 2「written before R20's first upload」：那個資料夾是 10 月或演練留下的。改名成 `<組>_stale_<時間>` 再跑，不要刪。
- 每一組寫好 `results/test08/p1/<組>_p1.npz` 和 `p1_<組>_report.json` 就 commit。三組也可以跑完一起 commit。

## 第 4 步：`apply`（CPU）

- 跑 `PY tools/t08_rank.py apply`，然後 commit `results/test08/`。
- 看報告（`results/test08/rank/t08_apply_report.json`）裡的這幾項：
  - GPU 接縫的兩個最大差（標準化特徵 ≤ 0.05、lp_P1TO ≤ 0.04，PREREG_08 rev 4）和是否逐位元相同；
  - Q02H 對得上 07（`q02h_vs_07.equal`）；
  - 手數交叉核對（對 07 的 HS、對 04 的 O4）0；
  - 本機接縫的判定（07 的 seam_val 逐位元組相同時，08 也要逐位元組相同）；
  - 兩組的 P1 失敗數和方向 ρ（門檻 0.4837）；
  - 08 和 07／04／R20 在 public 題、規則 B 下不同的題數（只記錄）。
- **GPU 接縫沒過（結束碼 4）**：停下查（PREREG_08 5.3）。
  - 先看 `results/test08/p1/gpu_seam.json` 的 `counts_equal`：不相同是程式或輸入問題，不是 GPU。
  - 再看 `.venv-gpu`、驅動有沒有變（`p1_seam_report.json` 的 `environment` 和 10 月演練的紀錄比）。
  - 只是本機環境的問題（例如 `.venv-gpu` 被改過）而 P1 不必重算：修好環境後 `apply --rerun --reason …`。
  - 要重算 P1：已 commit 的 `<組>_p1.npz` 工具不會覆寫（同 07），所以要先寫並 commit PREREG_08 的修訂（rev 4 起），寫明原因和重算的做法，再照修訂做；舊的塊資料夾改名成 `<組>_stale_<時間>`，不刪。
  - K6-08 前修不好：這一季不用 08，最終照 PREREG_07 5.4（07 本身也要 4090，實際上就是 07 的規則）。

## 第 5 步：上傳 08（兩則訊息）

1. 07 已記成 uploaded 之後，跑：
   `PY tools/official_log.py prepare --comp rank --file results/test08/rank/submission_rank.csv --reason "prereg08 Q08 (PREREG_08 5.4)"`
   prepare 會重算 5.2／5.4 的每一條（H8R、GPU 接縫、Q02H 對得上 07、手數交叉核對、本機接縫、失敗率、方向、T1、綁定的輸入、K6-08、R20／04／07 的上傳順序）。被拒絕就照訊息處理，不繞過。
2. **訊息 A**：放檢查結果（照 prepare 印出的 OK 請求），並請使用者對這一份說 OK。
3. 使用者說 OK 之後，**訊息 B** 自成一則：在 AIdea 競賽二（TEAM_11248）上傳 `results\test08\rank\submission_rank.csv`，寫上 SHA 前 12 碼，並說明不要從 test02、test04 或 test07 的資料夾拖。
4. 使用者回報後，跑 `uploaded --id N --ok-at … --uploaded-at …` 和 `public --id N --score …`，然後 commit。
- public 分數不進判定。
  - 08 比 07 低 0.05 以上時，只查錯（ID、列順序、輸入綁定）。查不到錯就維持 5.4 的結果，並寫進報告。
  - `public` 印出同檔警告時，請使用者核對 AIdea 的上傳紀錄。

## 第 6 步：最後一版（PREREG_08 5.4；11/23 23:59 前）

- 有符合的 08：`final --comp rank --id <08> --basis "PREREG_08 5.4"`。
- `status` 說 08 不再符合：照 PREREG_07 5.4 改 final（交 07，不行交 04，再不行交 R20），再 prepare 並重傳那一份，讓它成為最後一次上傳。
- 要改交 07／04／R20，只能靠 PREREG_08 rev 4 起的修訂，加 `--not-08-reason "PREREG_08 rev N: …"`。要交 04 或 R20、而 07 也符合時，`--not-07-reason` 也要給（04 也符合而交 R20 時再加 `--not-04-reason`）。

## 第 7 步：結果

寫 `results/test08/RESULTS.md`，內容包括：
- 閘門結果；
- P1 的速度、GPU 使用時間、驅動版本；
- GPU 接縫的兩個值、Q02H 對得上 07、手數交叉核對、本機接縫；
- 失敗率、方向 ρ；
- 08 和 07／04／R20 不同的題數（public／全部）；
- 上傳紀錄、5.4 的判定；
- 看過 public 分數之後做的選擇（應該沒有）。

## 4090 不能用時

不登記 CPU 或 Colab 的備案（PREREG_08 5.3）。K6-08 前修不好，這一季就不用 08，最終照 PREREG_07 5.4。

## test02、04 或 07 重跑了

- 08 綁住了 test02、04、07 的輸入（inputs_bound）。它們一變，08 的紀錄就不再符合。
- 照 TESTDAY／TESTDAY_04／TESTDAY_07 處理完之後，08 只能先寫修訂，再 `apply --rerun --reason … --bug-revision N`。
