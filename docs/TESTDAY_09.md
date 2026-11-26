# 測試日操作手冊：競賽二 09（test09，08 的 apply 之後）

- 這份是 Claude 的指令順序，只管競賽二的 09（Q09＝Q02H＋4·P1TO_E，ens3fam）。
  - test02 照 `docs/TESTDAY.md`，04 照 `docs/TESTDAY_04.md`，07 照 `docs/TESTDAY_07.md`，08 照 `docs/TESTDAY_08.md`，四份都不改。
  - **09 只用 CPU**：直接讀 08 在 4090 上算好的 P1 塊（`data/test08/p1/`），不再跑 GPU、不開 Colab。幾分鐘就跑完，白天可以跑。
- 依據：
  - 登記：`docs/PREREG_09_MAPPING_ENSEMBLE.md` 第 3、4、5 節、rev 3（第 13 節，測試日程序）和 rev 4（第 14 節，EX 的判定只看 42 維）；採用紀錄 `results/prereg09/RESULTS.md`。
  - 兩者不一致時，以登記為準。
- **09 是沒在驗證集打過分的組合**（PREREG_09 第 3 節的偏離）。這件事不影響測試日的做法，但報告要寫明。
- **競賽二的最終選擇**：PREREG_09 5.3 取代 PREREG_08 5.4 的「有符合的 08 時最終是 08」。
  - 有符合的 09 → 交 09。
  - 沒有 → 照 PREREG_08 5.4：交 08；08 不符合就照 PREREG_07 5.4 交 07，再不行照 PREREG_04 5.3 交 04，再不行交 R20；T1 判 FAIL_OPENING 時交 R10。
  - 09 判定之前不執行 `official_log.py final --comp rank`。已經設過也沒關係，第 5 步會重設。
- 執行環境：
  - `PY`＝`.venv/Scripts/python.exe`（09 不用 `.venv-gpu`）。
  - 一律在 repo 根目錄、用 Git Bash 執行，指令前面加 `PYTHONDONTWRITEBYTECODE=1`。
  - **不要設** `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`：`apply` 會拒絕（PREREG_09 1.3 第 7 點、2.6；位元組接縫要用預設執行緒）。
- **每一步做完就 commit，不論結束碼。**
  - 要 commit 的檔：`results/test09/gpu_seam.json`、`results/test09/rank/`（含 `seam_val/`、`t09_apply_report.json`、`test_lp_p1e.npz`）、`not_used.json`、`attempt_verdict.json`。
  - 只 `git add results/test09 results/official_scores.json`，不要 `git add -A`。
  - 09 自己不寫任何大檔；08 的 P1 塊（`data/test08/p1/**`）**不要刪、不要動**，09 會逐塊重算 SHA，到 11/25 以後都要留著。
- 出現結束碼 3 之後，不重跑任何步驟。要翻盤，只能先寫並 commit PREREG_09 rev 5 起的修訂（指出具體的程式錯誤），再用 `--bug-revision N`。

## 每一步都適用的規則

- **不碰答案**：不開、不讀、不雜湊驗證標籤檔。只有 `apply` 會讀 `data/local/rank_val_questions.csv`（題號、slot、執色、SGF；seam_val 用），工具自己會擋其他路徑。不跑 `tools/score.py`。
- **凍結檔一律不改**：
  - R20 的 39 個檔、04 的 21 個檔、06、07、08 的測試日路徑（`results/test08/design/p08_path_fingerprint.json`，tag `p08-testday-pre09`）；
  - `results/prereg01`–`prereg09/**`、`tools/p0[1-9]*.py`、`t02_*`、`t04_*`、`t06_*`、`t07_*`、`t08_*`、`third_party/katago/**`；
  - 四個釘住檔、PREREG_01–06、打分紀錄帳。
  - `tools/official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_07_HUMANSL.md`、`docs/PREREG_08_STRENGTH_PROBE.md`、`docs/PREREG_09_MAPPING_ENSEMBLE.md` 只准加行。
- **每次開工先跑防誤改閘門**：合併之後是十五個檔＝main 的 CLAUDE.md 閘門指令（十四個檔，含 `tests/test_t08_path_hygiene.py`）再加 `tests/test_t09_hygiene.py`（Git Bash，repo 根目錄）：
  `C10_T04_GATE=1 PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -rs tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py tests/test_p06_hygiene.py tests/test_p06_player_hygiene.py tests/test_t06_hygiene.py tests/test_p07_hygiene.py tests/test_t06_path_hygiene.py tests/test_t07_hygiene.py tests/test_p08_hygiene.py tests/test_t07_path_hygiene.py tests/test_t08_hygiene.py tests/test_p09_hygiene.py tests/test_t08_path_hygiene.py tests/test_t09_hygiene.py`
  - 唯一允許的失敗是 `test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`（帳上多了 H6P 4 列、H7R 2 列、H8R 2 列，都對得上已 commit 的登記；09 沒有打分列）。
  - 接著跑 `PY -m pytest -q tests/test_t09_official_log.py -k TagVersusNew`。
- **上傳**：
  - 每一份、每一次，都要使用者在對話裡對那一份說 OK。
  - 訊息 A 放檢查結果和 OK 請求；訊息 B 自成一則，只放上傳步驟。
  - 同一時間只能有一則等 OK 的訊息，不分競賽。
- **五份交件檔同名 `submission_rank.csv`**：
  - R20 在 `results\test02\rank\`，04 在 `results\test04\rank\`，07 在 `results\test07\rank\`，08 在 `results\test08\rank\`，09 在 `results\test09\rank\`。
  - 訊息 B 只給一個資料夾，寫上 SHA-256 前 12 碼和「不要從另外四個資料夾拖」。
  - 發訊息 B 之前，先用 `Get-FileHash` 核對。
- 每則訊息 A 之前跑 `PY tools/official_log.py status`，確認競賽二當天上傳次數 < 5（09 讓測試日多 1 次上傳）。

## `t09_rank.py` 的結束碼

| 碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成 | commit，做下一步 |
| 2 | 拒絕：前提不符。例如 git 不乾淨、prereg09 的釘住檔變了、08 的 apply 沒做完或沒 commit、08 報告綁定的輸入或 08 的輸出被改過、08 的 P1 報告或塊和紀錄的 SHA 不同、塊裡沒有逐局面陣列、輸出已存在、設了 BLAS 執行緒變數、K6-09 過後已有成功的那一次。工具內部錯誤（有 traceback）也是 2 | 照訊息修好前提，再跑同一行。要重寫輸出就加 `--rerun --reason` |
| 3 | 登記的「09 不用」：08 判不用（連帶 07／04）、09 沒有採用紀錄、H8R 不成立、K6-09 已過而且之前沒有成功的執行、T1 FAIL_OPENING、本機接縫超出容許差（08 的接縫成立時）、P1 失敗 > 0.5%、方向檢查沒過 | commit `results/test09/not_used.json`；不重跑；09 不 prepare、不上傳；最終照 PREREG_08 5.4，做法照 `docs/TESTDAY_08.md` 第 6 步（有符合的 08 就 `final` 08） |
| 4 | 本機漂移或對不上：`.venv` 版本、09 的 GPU 接縫（標準化 EX、lp_P1TO_E）、Q08 重算對不上 08 的 `rank_test_probs.csv`、可用盤和 08 不同、接縫結構或位元、check_submission | commit `rank/attempt_verdict.json`，停下查，查好後 `--rerun --reason` |

## K5-09 核對表（10/31 23:59）

- 分支 `p09-testday` 已合併回 main。
- 閘門（十五個檔：main 的 CLAUDE.md 那十四個，加 `test_t09_hygiene.py`）只有已知的那一項失敗；`C10_SLOW=1` 的 `tests/test_t09_dryrun.py` 全過（R20／04／07／08 的輸出等於 `results/test09/design/baseline_pre09.json`，09 的輸出等於 `t09_fake_golden.json`）。
- PREREG_09 rev 3、rev 4 已 commit；tol_E 的實測在 `results/test09/design/tol_e_measurement.json`。
- 08 的 K5-08 核對表也成立（09 讀的是 08 的塊）。

## 日期與 K6-09

- **11/04–11/07**：照 TESTDAY、TESTDAY_04、TESTDAY_07、TESTDAY_08。
- **08 的 `apply` 結束碼 0 並 commit 之後**，才跑 09 的 `apply`（PREREG_09 5.2 的前提）。CPU 幾分鐘，不用 GPU，不用排夜間。
- 08 上傳之後（最晚 11/18），才 `prepare` 09。
- **K6-09＝11/19 23:59:59**（和 K6-08 同一刻）：`t09_apply_report.json` 最後一次 commit 要在這之前。
- 11/22、11/23 各跑一次 `status`。

## 第 1 步：閘門

1. 跑防誤改閘門（見上）。
2. 跑 `PY tools/official_log.py status`：R20 的第一次上傳要已經記成 uploaded 而且 commit 了（`apply` 經由 `t08_rank.py` 的前提會核對它；正式模式沒有就是 2）。04、07、08 的上傳是 09 的 `prepare` 才要求的，`apply` 不看。
3. 看 T1（`results/test02/checks/rank/t1.json`），要是 PASS 或 NORMALIZED。FAIL_OPENING：09 不用（08、07、04 也不用）。

## 第 2 步：等 08 的 `apply`

- 要等 TESTDAY_08 第 4 步的 `results/test08/rank/t08_apply_report.json` 結束碼 0，而且已 commit。
- 08 判不用（`results/test08/not_used.json`）→ 09 也不用，`apply` 會回 3。
- 08 的 apply 之後，不要再動 `results/test08/`、`data/test08/` 的任何檔：09 會核對 08 綁定的每一個輸入、08 的輸出和每一塊 P1 的 SHA。

## 第 3 步：`apply`（CPU）

- 跑 `PY tools/t09_rank.py apply`，然後 commit `results/test09/`。
- 看報告（`results/test09/rank/t09_apply_report.json`）裡的這幾項：
  - 09 的 GPU 接縫（`gpu_seam`）：seam 400 盤（08 的 seam 塊）對驗證塊第 0、1 塊（同一批盤、同樣分塊），標準化 EX ≤ 0.05（只判兩維 frac_lowp 以外的 42 維，PREREG_09 rev 4；兩維 frac_lowp 的最大差和跳動盤數在 `frac_lowp_record_only`，只記錄）、lp_P1TO_E ≤ 0.04（tol_E，PREREG_09 rev 3），是否逐位元相同；
  - Q08 對得上 08（`q08_vs_08.equal`，預期 `byte_equal` 也是 true）；
  - 本機接縫的判定（08 的 seam_val 逐位元組相同時，09 也要和 `results/prereg09/rank/` 逐位元組相同）；
  - 兩組的 P1 失敗數和方向 ρ（門檻 0.4843）；
  - 09 和 08／07／04／R20 在 public 題、規則 B 下不同的題數（只記錄；演練時 public 和 08 不同 74 題）。
- **GPU 接縫沒過（結束碼 4）**：停下查。
  - 先看 `results/test09/gpu_seam.json` 的 `counts_equal`：不相同是程式或輸入問題。
  - 再看 `max_abs_std_ex`（42 維）和 `max_abs_lp_p1e`：不同批次組合下，42 維實測最大 0.016、lp_P1TO_E 0.011，離容許差都很遠；超過就是真的不一樣，照 PREREG_09 第 5 節停下查。兩維 frac_lowp 跳一格（實測最大 0.254）是 fp16 捨入，rev 4 起只記錄、不判定，不要因為它停下。
  - K6-09 前修不好：這一季不用 09，最終照 PREREG_08 5.4。
- **Q08 對不上 08（結束碼 4）**：本機問題（環境、08 的檔、程式）。不是 09 的判定；查好後 `--rerun --reason`。

## 第 4 步：上傳 09（兩則訊息）

1. 08 已記成 uploaded 之後，跑：
   `PY tools/official_log.py prepare --comp rank --file results/test09/rank/submission_rank.csv --reason "prereg09 Q09 (PREREG_09 5.3)"`
   prepare 會重算 5.2／5.3 的每一條（09 的採用紀錄、H8R、09 的 GPU 接縫、Q08 對得上 08、本機接縫、失敗率、方向、T1、綁定的輸入、K6-09、R20／04／07／08 的上傳順序，綁定的 08 仍符合 PREREG_08 5.4）。被拒絕就照訊息處理，不繞過。
2. **訊息 A**：放檢查結果（照 prepare 印出的 OK 請求，含「09 是沒在驗證集打過分的組合」那一行），並請使用者對這一份說 OK。
3. 使用者說 OK 之後，**訊息 B** 自成一則：在 AIdea 競賽二（TEAM_11248）上傳 `results\test09\rank\submission_rank.csv`，寫上 SHA 前 12 碼，並說明不要從 test02、test04、test07 或 test08 的資料夾拖。
4. 使用者回報後，跑 `uploaded --id N --ok-at … --uploaded-at …` 和 `public --id N --score …`，然後 commit。
- public 分數不進判定（PREREG_09 5.3）。
  - 09 比 08 低 0.05 以上時，只查錯（ID、列順序、輸入綁定）。查不到錯就維持 5.3 的結果，並寫進報告。
  - `public` 印出同檔警告時，請使用者核對 AIdea 的上傳紀錄。

## 第 5 步：最後一版（PREREG_09 5.3；11/23 23:59 前）

- 有符合的 09：`final --comp rank --id <09> --basis "PREREG_09 5.3"`。
- `status` 說 09 不再符合，或 09 判不用（結束碼 3）、從沒 prepare：照 PREREG_08 5.4 改 final，做法照 `docs/TESTDAY_08.md` 第 6 步（交 08，不行交 07，再不行交 04，再不行交 R20）；最後一次上傳不是那一份時，再 prepare 並重傳，讓它成為最後一次上傳。
- 要改交 08／07／04／R20，只能靠 PREREG_09 rev 5 起的修訂，加 `--not-09-reason "PREREG_09 rev N: …"`。要交 07／04／R20、而 08 也符合時，`--not-08-reason` 也要給（照 TESTDAY_08 第 6 步，依此類推）。

## 第 6 步：結果

寫 `results/test09/RESULTS.md`，內容包括：
- 閘門結果；
- 09 的 GPU 接縫兩個值、Q08 對得上 08、本機接縫；
- 失敗率、方向 ρ；
- 09 和 08／07／04／R20 不同的題數（public／全部）；
- 上傳紀錄、5.3 的判定；
- 看過 public 分數之後做的選擇（應該沒有）；
- 「沒在驗證集打過分」的揭露（PREREG_09 第 3、8 節）。

## test02、04、07 或 08 重跑了

- 09 綁住了 test02、04、07、08 的輸入（inputs_bound），也核對 08 的每一塊 P1。它們一變，09 的紀錄就不再符合。
- 照 TESTDAY／TESTDAY_04／TESTDAY_07／TESTDAY_08 處理完之後，09 只能先寫修訂，再 `apply --rerun --reason … --bug-revision N`。

## 演練（主 session 或之後的審查用；不寫進 repo）

- CPU 從頭建樹：`C10_SLOW=1 C10_KEEP=1 PY -m pytest -q -p no:cacheprovider tests/test_t09_dryrun.py`（先建 R20／04／07 的樹，約 10 分鐘）。
- 重用已有 07 演練的樹 `<S>`（會重建 `<S>/results/test08`、`<S>/results/test09`）：加 `C10_T09_TREE=<S>`。10/9 用的是 `D:\c10_t09_reh`（`D:\c10_t08_reh` 去掉 test08 的副本；`D:\c10_t08_reh` 留給 08 的 4090 演練，不要在那裡跑）。
- 08 的 from-val 假塊沒有逐局面陣列，09 演練用 `tools/t09_fake_tests.py from-val`（＝`t08_fake_tests.py from-val` 再補上同一批驗證盤的逐局面陣列），要在 08 的 `apply` 之前做。
