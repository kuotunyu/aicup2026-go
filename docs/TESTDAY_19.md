# 測試日操作手冊：競賽二 19（test19，09 的 apply 之後）

- 這份是 Claude 的指令順序，只管競賽二的 19（Q19＝Q09＋4·SET，題目層級 set 模型 S19 當第 7 項）。
  - test02 照 `docs/TESTDAY.md`，04、07、08、09 照 `docs/TESTDAY_04.md`、`TESTDAY_07.md`、`TESTDAY_08.md`、`TESTDAY_09.md`，14 照 `docs/TESTDAY_14.md`；那幾份都不改。
  - **19 只用 CPU**：讀 08 在 4090 上算好、09 也讀過的 P1 塊（`data/test08/p1/`），不跑 GPU、不開 Colab。幾分鐘就跑完，白天可以跑。
- 依據：`docs/PREREG_19_SET_MODEL.md` 第 3、4、5 節、rev 3（測試日程序）；凍結和採用紀錄 `results/prereg19/rank/`（`frozen_config.json`、`apply_report.json` usable、`apply_audit.json`）。兩者不一致時，以登記為準。
- **19 和 09 一樣是沒在驗證集打過分的組合**（PREREG_19 第 3 節）。不影響測試日的做法，報告要寫明。
- **競賽二的最終選擇**：PREREG_19 4.3 取代 PREREG_09 5.3 的「有符合的 09 → 09」。
  - 有符合的 19 → 交 19。
  - 沒有 → 照 PREREG_09 5.3（`docs/TESTDAY_09.md` 第 5 步：09 → 08 → 07 → 04 → R20，T1 FAIL_OPENING 時 R10）。
  - 14（完全複製層）一律套在最後選定的那一份上：19 是最終時，14 的底檔就是 19。
- 執行環境：`PY`＝`.venv/Scripts/python.exe`；repo 根目錄、Git Bash，指令前加 `PYTHONDONTWRITEBYTECODE=1`；**不要設** `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`。
- **每一步做完就 commit，不論結束碼**：只 `git add results/test19 results/official_scores.json`，不要 `git add -A`。19 自己不寫大檔；08 的 P1 塊和 `results/test09/` 都**不要動**（19 會重算它們的 SHA）。
- 出現結束碼 3 之後，不重跑任何步驟。要翻盤，只能先寫並 commit PREREG_19 rev 4 起的修訂（指出具體的程式錯誤），再用 `--bug-revision N`。

## 每一步都適用的規則

- **不碰答案**：不開、不讀、不雜湊驗證標籤檔。只有 `apply` 經由 test02／04 的已登記程式讀 `data/local/rank_val_questions.csv`（seam_val 的題目結構），工具自己會擋其他路徑。不跑 `tools/score.py`。
- **凍結檔一律不改**：`results/prereg19/rank/**`、`results/prereg0*`／`prereg1[0-8]`、`results/test02|04|07|08|09/**`、R20／04／06／07／08／09 的測試日路徑、`tools/t0*`、`tools/p*`、`tools/t19_rank.py`。`tools/official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_19_SET_MODEL.md` 只准加行。
- **每次開工先跑防誤改閘門**（AGENTS.md 的閘門指令，含 `tests/test_p19_hygiene.py`、`tests/test_t19_hygiene.py`）；接著跑 `PY -m pytest -q tests/test_t19_official_log.py -k TagVersusNew`。
- **上傳**：每一份、每一次，都要使用者在對話裡對那一份說 OK；訊息 A 放檢查結果和 OK 請求，訊息 B 自成一則只放上傳步驟；同一時間只能有一則等 OK 的訊息。
- **六份交件檔同名 `submission_rank.csv`**：R20 `results\test02\rank\`、04 `results\test04\rank\`、07 `results\test07\rank\`、08 `results\test08\rank\`、09 `results\test09\rank\`、**19 `results\test19\rank\`**（14 另在 `results\test14\rank\`）。訊息 B 只給一個資料夾，寫 SHA-256 前 12 碼（先 `Get-FileHash` 核對），並說明不要從其他資料夾拖。
- 每則訊息 A 之前跑 `PY tools/official_log.py status`，確認競賽二當天上傳次數 < 5（19 讓測試日多 1 次上傳）。

## `t19_rank.py` 的結束碼

| 碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成 | commit，做下一步 |
| 2 | 拒絕：前提不符。例如 git 不乾淨、prereg19 的釘住檔變了、09 的 apply 沒做完或沒 commit、09 報告綁定的輸入或 09 的輸出被改過、08 的 P1 報告或塊和紀錄不同、輸出已存在、設了 BLAS 執行緒變數、K6-19 過後已有成功的那一次。工具內部錯誤（有 traceback）也是 2 | 照訊息修好前提，再跑同一行；要重寫輸出加 `--rerun --reason` |
| 3 | 登記的「19 不用」：19 沒有採用紀錄、09 判不用（連帶 08／07／04）、K6-19 已過而且之前沒有成功的執行、T1 FAIL_OPENING、本機接縫超出容許差（09 的接縫是容許差判定時）、SET 方向檢查沒過 | commit `results/test19/not_used.json`；不重跑；19 不 prepare、不上傳；最終照 PREREG_09 5.3（`docs/TESTDAY_09.md` 第 5 步） |
| 4 | 本機漂移或對不上：`.venv` 版本、Q09 重算對不上 09 的 `rank_test_probs.csv` 或 `test_lp_p1e.npz`、本機接縫的結構或位元（09 的接縫逐位元組相同時 19 也要）、check_submission | commit `rank/attempt_verdict.json`，停下查，查好後 `--rerun --reason` |

## K5-19 核對表（10/31 23:59）

- 分支已用 `--no-ff` 合併回 main；閘門只有 AGENTS.md 列的允許失敗；`C10_SLOW=1` 的 `tests/test_t19_dryrun.py` 全過（R20／04／07／08 的輸出等於 `results/test09/design/baseline_pre09.json`，09 的輸出等於 `t09_fake_golden.json`，19 的輸出等於 `results/test19/design/t19_fake_golden.json`）；`tests/test_t14_dryrun.py` 也全過。
- PREREG_19 rev 3 已 commit。

## 日期與 K6-19

- **09 的 `apply` 結束碼 0 並 commit 之後**，才跑 19 的 `apply`（CPU 幾分鐘）。
- 09 上傳之後（最晚 11/18），才 `prepare` 19。
- **K6-19＝11/19 23:59:59**（和 K6-09 同一刻）：`t19_apply_report.json` 最後一次 commit 要在這之前。
- 11/22、11/23 各跑一次 `status`。

## 第 1 步：閘門

1. 跑防誤改閘門（見上）。
2. `PY tools/official_log.py status`：R20 的第一次上傳要已經記成 uploaded 並 commit。
3. T1（`results/test02/checks/rank/t1.json`）要是 PASS 或 NORMALIZED；FAIL_OPENING 時 19 不用（09 以下也不用）。

## 第 2 步：等 09 的 `apply`

- 要等 `docs/TESTDAY_09.md` 第 3 步的 `results/test09/rank/t09_apply_report.json` 結束碼 0、已 commit。
- 09 判不用（`results/test09/not_used.json`）→ 19 也不用，`apply` 會回 3。
- 09 的 apply 之後，不要再動 `results/test08/`、`results/test09/`、`data/test08/` 的任何檔。

## 第 3 步：`apply`（CPU）

- 跑 `PY tools/t19_rank.py apply`，然後 commit `results/test19/`。
- 看 `results/test19/rank/t19_apply_report.json`：
  - `q09_vs_09.equal`（預期 `byte_equal` 也是 true）：Q09 由 19 重算，和 09 已 commit 的檔相同；
  - `seam_local`：09 的 seam_val 逐位元組相同時，19 要和 `results/prereg19/rank/` 逐位元組相同；
  - `set_direction`：兩組的 ρ（門檻 0.4793）；
  - `set_seam_record_only`：SET 在 seam 400 盤的最大差（**只記錄**；09 的 GPU 接縫已判過同一批塊）；
  - `rule_b_differences`：19 和 09／08／07／04／R20 在 public 題、規則 B 下不同的題數（只記錄）。
- **Q09 對不上 09（結束碼 4）**：本機問題（環境、09 的檔、程式），不是 19 的判定；查好後 `--rerun --reason`。

## 第 4 步：上傳 19（兩則訊息）

1. 09 已記成 uploaded 之後，跑：
   `PY tools/official_log.py prepare --comp rank --file results/test19/rank/submission_rank.csv --reason "prereg19 Q19 (PREREG_19 4.3)"`
   prepare 會重算 4.2／4.3 的每一條（19 和 09 的採用紀錄、H8R、Q09 對得上 09、本機接縫、SET 方向、T1、綁定的輸入、K6-19、R20 和綁定的 09 已上傳而且 09 仍符合 PREREG_09 5.3）。被拒絕就照訊息處理，不繞過。
2. **訊息 A**：放檢查結果（照 prepare 印出的 OK 請求，含「19 是沒在驗證集打過分的組合」那一行），請使用者對這一份說 OK。
3. 使用者說 OK 之後，**訊息 B** 自成一則：在 AIdea 競賽二（TEAM_11248）上傳 `results\test19\rank\submission_rank.csv`，寫 SHA 前 12 碼，說明不要從 test02／04／07／08／09 的資料夾拖。
4. 使用者回報後：`uploaded --id N --ok-at … --uploaded-at …`、`public --id N --score …`，commit。
- public 分數不進判定。19 比 09 低 0.05 以上時只查錯（ID、列順序、輸入綁定），查不到就維持 4.3 的結果並寫進報告；`public` 印出同檔警告時，請使用者核對 AIdea 的上傳紀錄。

## 第 5 步：最後一版（PREREG_19 4.3；11/23 23:59 前）

- 有符合的 19：`PY tools/official_log.py final --comp rank --id <19> --basis "PREREG_19 4.3"`。
- `status` 說 19 不再符合，或 19 判不用（結束碼 3）、從沒 prepare：照 PREREG_09 5.3 改 final，做法照 `docs/TESTDAY_09.md` 第 5 步；最後一次上傳不是那一份時，再 prepare 並重傳。
- 要改交 09／08／07／04／R20，只能靠 PREREG_19 rev 4 起的修訂，加 `--not-19-reason "PREREG_19 rev N: …"`（09 也符合而要交更舊的，`--not-09-reason` 也要給，依此類推）。
- **之後才做 14**（`docs/TESTDAY_14.md`）：final 是 19 時，14 的底檔就是 19。14 成為最終之後，11/22–11/23 的確認改成重跑 `final --id <14>`（TESTDAY_14 第 4 步），不要對 19 重跑 final。

## 第 6 步：結果

寫 `results/test19/RESULTS.md`：閘門結果；Q09 對得上 09、本機接縫、SET 方向 ρ、SET 的 GPU 接縫紀錄；19 和 09／08／07／04／R20 不同的題數（public／全部）；上傳紀錄、4.3 的判定；看過 public 分數之後做的選擇（應該沒有）；「沒在驗證集打過分」和「點子來自外部 AI 審閱者（Codex）」的揭露（PREREG_19 第 3、8 節）。

## test02、04、07、08 或 09 重跑了

- 19 綁住了它們的輸入和 09 的輸出（inputs_bound）。它們一變，19 的紀錄就不再符合。照各自的手冊處理完之後，19 只能先寫修訂，再 `apply --rerun --reason … --bug-revision N`。

## 演練（主 session 或之後的審查用；不寫進 repo）

- `C10_SLOW=1 C10_KEEP=1 PY -m pytest -q -p no:cacheprovider tests/test_t19_dryrun.py`：先照 `tests/test_t09_dryrun.py` 建 R20／04／07／08／09 的假樹（約 15 分鐘），再跑 19 的 `apply`／`status --dry-run` 和負面演練。
- 重用已有 09 演練的樹 `<S>`（`<S>/results/test09/rank/t09_apply_report.json` 存在；只重建 `<S>/results/test19`）：加 `C10_T19_TREE=<S>`。
- 樹不要放在路徑含 `AppData\Local` 的地方（工具的 `data\local` 防護會誤擋），也不要放 `D:\` 根目錄：用 `C10_DRYRUN_DIR=<subst 的磁碟>`，例如 scratchpad 裡的資料夾 `subst S:` 後用 `S:\`。
