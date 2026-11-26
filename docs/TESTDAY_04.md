# 測試日操作手冊：競賽二 04（test04，11/05 起）

- 這份是 Claude 的指令順序，只管競賽二的 04（Q02＝R20＋O4＋SWA＋SWB）。**test02（R20 和競賽一）照 `docs/TESTDAY.md` 做，那份不改**（它在 R20 的指紋裡）。使用者在 Colab 上的步驟在 `docs/COLAB_RUN.md` 最後一節「測試集 04（test04）」。
- 依據：`docs/PREREG_04_OPPONENT.md` 第 5 節和 rev 3 第 11 節（登記）；實作規格 `results/test04/design/spec.md`。兩者不一致時以登記為準。
- **競賽二的最終選擇**：`docs/TESTDAY.md`「之後的上傳」那一段寫「最終選擇（U5）：現在兩場都是 prereg02（甲）」。對競賽二，這一行由 PREREG_04 第 5.3 節和本檔第 7 步取代。**04 判定之前**（`opp` 或 `apply` 已經判定，或 K6 已過之前），不要執行 `official_log.py final --comp rank`。已經照 U5 設過也沒關係：第 7 步會重跑 `final`，`status` 會提示。競賽一照 U5 不變。
- `PY` = `.venv\Scripts\python.exe`（Git Bash 寫 `.venv/Scripts/python.exe`），一律在 repo 根目錄 `D:\AI-Portfolio\competitions\aicup2026-go` 執行。
- **每一段做完就 commit，不論結束碼**：`opp_report.json`、`attempt_verdict.json`、`t04_apply_report.json` 都要 commit。作者 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，不加 Co-Authored-By。`data/**` 和 zip 在 gitignore 裡，不 commit；它們的 SHA-256 記在會 commit 的 JSON 裡。
- **結束碼 3（04 不用）之後，不重跑任何步驟**。要翻盤只能先寫並 commit PREREG_04 的修訂（rev 3 第 20 條：指出具體的程式錯誤），工具會核對。
- 11/04 當天只做 test02，不做 04。

## 每一步都適用的規則

- 不開、不讀、不雜湊 `data/local/` 裡的驗證標籤檔；不跑 `tools/score.py`；不用 GPU。
- 凍結檔一律不改：R20 的 39 個檔（`results/prereg04/design/r20_path_fingerprint.json`，含 `docs/TESTDAY.md`、`tools/t02_*.py`、`tools/check_submission.py`）、`results/prereg01`–`prereg04/**`、`tools/p02_*.py`、`p03_rank_fuse.py`、`p04_opp.py`、`p04_rank_fuse.py`、`c10data.py`、`decide.py`、`score.py`、`make_val.py`。`tools/official_log.py` 只准加行。
- 每次開工先跑一次防誤改測試，全過才往下做（唯一的例外見第 1 步「防誤改測試」那一列：失敗的正好是一份已 commit 的 PREREG_02 修訂改的 R20 側檔案）。在 Git Bash 跑，前面一定要有 `C10_T04_GATE=1`：少了它，打分紀錄帳和 PREREG_01–03 的列數核對（`test_ledger_and_prereg_rows_pinned`）會被略過（review round 3：一般的 `PY -m pytest -q tests` 刻意略過它，免得別的登記在 11/04 前依規打分時，R20 自己的「全綠」就緒閘門被 04 的測試弄紅）：

```
C10_T04_GATE=1 PY -m pytest -q tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py
```

- 工具拒絕時照訊息處理，不繞過。要改程式或放寬門檻，先寫 PREREG_04 修訂（下一版是 rev 4）。
- 上傳：每一份檔、每一次上傳，都要使用者在對話裡對那一份說 OK，不沿用先前的同意。檢查結果放在較早的訊息，**上傳指示自成一則訊息**。
- 使用者的不可逆操作（AIdea 上傳、Drive 改名或刪除、刪電腦「下載」裡的檔）**一則訊息只給一件**，做完等使用者回報、Claude 核對後，才給下一件。

## `t04_rank.py` 的結束碼（每一步都一樣）

| 結束碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成，條件都成立 | commit，做下一步 |
| 1 | 接縫類失敗：這次 notebook 的輸出不用 | commit 寫出的紀錄，照「重跑一次」 |
| 2 | 拒絕或錯誤：前提不符 | 照訊息修好前提，再跑同一個本機指令。訊息是「unexpected internal error」（前面有 traceback）時：那是工具本身出錯，**不是接縫失敗**，不 archive、不請使用者重跑；停下查本機工具 |
| 3 | 登記的「04 不用」 | commit 寫出的報告（有寫出的話；K6、FAIL_OPENING 時工具不寫檔，把 stderr 的訊息記進 RESULTS）；最終 R20（T1 是 FAIL_OPENING 時 R10）→ 第 7、8 步（第 7 步要設 `final`、確認最後一次上傳、11/22–23 跑 `status`）；不重跑 |
| 4 | 本機漂移（`.venv` 或程式） | commit 寫出的紀錄；停下查本機，不請使用者做任何事，不動 Colab；查好後 `--rerun --reason "..."` |

## 11/03 以前要就緒（K5 核對表，11/03 23:59）

- [ ] PREREG_04 rev 3 已 commit，分支 `p04-testday` 已合併回 main。
- [ ] `results/test04/design/build_review.md` 記了：R20 逐位元組演練（兩個一次性 worktree，也等於 `r20_fake_golden.json`）、`official_log.py` 的 tag 版對新版差異測試、04 的 dry run（`tests/test_t04_dryrun.py`）、完整測試、`C10_SLOW=2`、notebook 的 CPU 冒煙（L2b、L2），和三個角度的審查。
- [ ] 計數檔 SHA 正確（第 1 步那一行）。
- [ ] `colab/test04_rank.ipynb` 已 commit；`data\colab_upload\test04\` 裡只有 `test04_rank.ipynb`，SHA 等於已 commit 的那一份（第 3 步那一行）。
- [ ] `.gitattributes` 有 `results/test04/** -text`（`tests/test_t04_hygiene.py` 會檢查）。
- [ ] 合併前在分支上跑 `C10_K5=1 PY -m pytest -q tests/test_t04_hygiene.py`，全過（打分紀錄帳、PREREG_01–03 和 6ec5e68 逐位元組相同）。
- [ ] 選做：Colab 彩排（本文最後一節）做完、收拾完，結果記進 `build_review.md`。

## 日期與 K6

- 11/04：只做 test02（`docs/TESTDAY.md`）。04 的第一步要等 R20 的第一次上傳在 `results/official_scores.json` 記成 `uploaded` 並 commit。
- 11/05–11/07：04 的目標。R20 的第一次上傳延誤（例如 test02 重跑）時，04 順延，K6 不順延。
- **11/15 前**：`apply` 做完並 commit（K6 看的是報告 commit 的時間，留一天緩衝）。登記的重跑只在 G4，最晚等到 11/15 中午。
- **K6 11/16（一）23:59:59**：過了之後，正式模式的 `opp`、`unpack`、`apply` 一律結束碼 3，最終 R20。
- 11/22、11/23 各跑一次 `PY tools/official_log.py status`（第 7 步）。最後一次上傳在 11/23 23:59 前（截止 11/24 23:59）。
- 競賽二每天最多上傳 5 次，R20 和 04 合計。

## 第 1 步：閘門（test02 做完了嗎）

先全部核對，任何一項不對就停下，不做第 2 步。

```
PY tools/official_log.py status
PY -c "import json; t=json.load(open('results/test02/checks/rank/t1.json', encoding='utf-8')); print(t['verdict'], t.get('complete'), t.get('stage'))"
git status --porcelain
PY -c "import hashlib; print(hashlib.sha256(open('data/prereg04/counts_fit.npz','rb').read()).hexdigest())"
PY -c "import sys, numpy, scipy, zlib; print(sys.version.split()[0], numpy.__version__, scipy.__version__, zlib.ZLIB_RUNTIME_VERSION)"
C10_T04_GATE=1 PY -m pytest -q -rs tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py
PY -c "import json, hashlib; fp=json.load(open('results/prereg04/design/r20_path_fingerprint.json', encoding='utf-8')); print([r for r, h in sorted(fp['files'].items()) if hashlib.sha256(open(r, 'rb').read().replace(b'\r\n', b'\n')).hexdigest() != h])"
```

最後一行列出 R20 的 39 個指紋檔裡內容變了的檔。正常只有 `tools/official_log.py`（04 只准對它加行，`test_p04_hygiene.py` 另外核對）。

| 項目 | 要看到 | 不是的話 |
|---|---|---|
| `status` | 競賽二有一筆 `results/test02/rank/submission_rank.csv` 已上傳（R20 的第一次上傳），`results/official_scores.json` 已 commit | 先照 `docs/TESTDAY.md` 第 8 步做完 |
| T1 | `PASS` 或 `NORMALIZED`，`True raw` | `FAIL_OPENING`：04 整條不做，RESULTS 記「04 不上場（T1 FAIL_OPENING）」，最終 R10（照 TESTDAY.md），本檔到此結束。`FAIL_STRUCT`：照 test02 停下 |
| `git status` | 空的 | 先 commit 或查清楚 |
| 計數檔 | `f6c6debe90639585763b9812c235743a6b7f89d8db1c6422ece7bccd550a08a5` | 本機另有一份 `data/prereg04/design_scratch/counts_fit.npz`（同一個 SHA）可以複製回來。兩份都不見：只能在 repo 外照 `p04_opp.py count` 重算（至少 7 個 worker，約 34 分鐘，排在沒有其他線吃 CPU 的時段），SHA 必須等於左邊的值 |
| `.venv` | `3.12.13 2.5.3 1.18.1 1.3.1` | 停下，還原 `.venv`（11/24 前不升級） |
| 防誤改測試 | 全過，而且 `-rs` 沒有列出 `test_ledger_and_prereg_rows_pinned` 被略過（被略過＝少了 `C10_T04_GATE=1`，重跑）；指紋檔清單只有 `tools/official_log.py` | 停下查是誰改了凍結檔。**唯一的例外**（rev 3 第 19 條擴大，見下面「R20 側的已登記變動」一節）：失敗的只有 `test_p04_hygiene.py` 的 `test_pinned_files_unchanged`，和／或 `test_t04_hygiene.py` 的 `test_ledger_and_prereg_rows_pinned`（它把打分紀錄帳多出來的資料列、PREREG_01–03 多出來的 `\| rev N \|` 列印在失敗訊息裡），而且：① 清單裡多出來的檔，正好是一份**已 commit** 的 PREREG_02 修訂（rev 9 起，或 rev 7 第 2 條競賽一改 A100）列出要改的檔；② 多出來的每一列都已 commit（這個測試也核對乾淨），而且各自對得上一份已 commit 的登記（打分紀錄帳的列：哪一份事前登記的哪一次打分；修訂列：那份修訂本身）。用 `git diff --stat r20-testday-pre04 -- <那些檔>`、`git log --oneline -- results/score_ledger.csv docs/PREREG_0[123]*.md` 核對，把修訂編號、檔名、改前後的 SHA、多出來的每一列和它的 commit 記進 RESULTS。多出來的只有競賽一的檔（檔名含 `player`）或只有打分紀錄帳的列：04 照常往下做。多出來的有其他檔（R20 的路徑）：照那一節，04 停下直到 PREREG_04 修訂 commit。另一種不是凍結檔被改的失敗（K5 閘門第 1 輪）：只有 `test_t02_hygiene.py` 的 `Pins.test_pinned_files` 失敗、而且 `git status --porcelain` 是空的，多半是重新 checkout（新 clone、換電腦）時 `core.autocrlf=true` 把檔案寫成 CRLF。`.gitattributes` 對那 7 個文字檔設了 `-text`，正常不會再發生；真遇到時先 `git -c core.autocrlf=false checkout -- results/prereg02` 寫回 LF，再跑一次閘門，並把這件事記進 RESULTS |

## 第 2 步：O4 重播（CPU，不用 Colab）

開跑前跑 `python D:\AI-Portfolio\_tools\handoff.py`（加 `PYTHONIOENCODING=utf-8`），看有沒有別的線在吃 CPU。worker 最多 6，結果和 worker 數無關。

```
PY tools/t04_rank.py opp --workers 6
```

- 讀 `data/test` 的題目檔（MANIFEST 核對過的那一份，絕不讀 `data/test/raw`）、`data/local/rank_val_questions.csv`（只讀題目）、test02 的 key。寫 `results/test04/opp/`：`o4_seam_val.npz`、`o4_public.npz`、`o4_private.npz`、`opp_report.json`。約 2–3 分鐘（推估）。
- 看 `opp_report.json`：seam_val 是否逐位元組相同（`seam_val.file_sha256_equal`）、三組的重播失敗（`sets.<組>.failures`／`slots`）、public 和 private 的 O4 Δκ（`o4_dkappa_early`）。

| 結束碼 | 情況 | 下一步 |
|---|---|---|
| 0 | seam_val 逐位元組相同、seam_val 的 Δκ 相同、三組失敗率都 ≤ 0.5%、public 和 private 的 O4 Δκ 都 > 0 | commit `results/test04/opp/` → 第 3 步 |
| 3 | 任一組（含 seam_val）重播失敗 > 0.5%；或 seam_val 逐位元組相同、Δκ 也相同，而 public／private 的 O4 Δκ ≤ 0、NaN、沒有子題；或 T1 FAIL_OPENING；或 K6 已過 | commit `results/test04/opp/`（有寫出的話）。04 不用，**不開 Colab**，最終 R20（FAIL_OPENING 時 R10）→ 第 7、8 步 |
| 4 | seam_val 不逐位元組相同（失敗 ≤ 0.5%）、seam_val 的 Δκ 不同、比率表的 SHA 不同、`.venv` 版本不同 | commit 報告（有寫出的話：seam_val 的兩種會寫；比率表、`.venv` 不同時在寫檔之前就停下，沒有報告）；停下查本機，不請使用者做任何事；查好後 `PY tools/t04_rank.py opp --workers 6 --rerun --reason "..."`。這時 public／private 的 O4 Δκ 就算 ≤ 0 也只記錄（報告的 `o4_dkappa_not_judged_while_drift`）：它是用沒過接縫核對的重播算的，查好重跑後才判定 |
| 2 | 前提不符（例如 R20 的第一次上傳還沒記錄、test02 的輸出沒 commit；計數檔不見或 SHA 不同） | 照訊息修好再跑（計數檔照第 1 步那一列還原） |

- 判定是 0 或 3 之後，`opp --rerun` 會拒絕（CPU 重播是確定的）；只有寫了指出程式錯誤的修訂，才能加 `--bug-revision N` 重跑。
- commit 訊息例：`test04 opp: usable so far (public / private O4 dkappa > 0, 0 replay failures)`。

## 第 3 步：請使用者跑 test04 notebook

**先由 Claude 自己核對（第 4 點是另外一則請使用者做的訊息），全部對了才給使用者步驟：**

1. 上傳副本和已 commit 的 notebook 相同。在 PowerShell 貼這一行（先 `cd D:\AI-Portfolio\competitions\aicup2026-go`），兩行 `Hash` 要一樣，而且資料夾裡只有這一個檔：

   ```powershell
   Get-FileHash -Algorithm SHA256 "colab\test04_rank.ipynb", "data\colab_upload\test04\test04_rank.ipynb" | Format-List Hash, Path; Get-ChildItem "data\colab_upload\test04" | Select-Object Name
   ```

   不一樣或多了別的檔：重新複製 `colab/test04_rank.ipynb` 過去（或刪掉多的檔），再核對一次。
   然後取出已 commit 的 notebook 的版本 12 碼（notebook 第一個程式格最後一行會印同一個值），寫進給使用者的訊息：

   ```
   PY -c "import json; nb=json.load(open('colab/test04_rank.ipynb', encoding='utf-8')); print([l for c in nb['cells'] for l in c['source'] if l.startswith('NOTEBOOK_SHA256 = ')][0][19:31])"
   ```

2. 第 2 步是結束碼 0，而且 `results/test04/opp/` 已 commit。
3. 競賽一的 `test02_player` 沒有開著的執行階段（在跑、或已經連上 GPU）。有的話先等它跑完：兩份同時跑會搶 Drive，也違反 COLAB_RUN 的規則。只是在等 G4、目前沒有開執行階段，**不算在跑**，可以先跑 test04（K6 是 11/16，重跑最晚 11/15 中午；競賽一可以等 G4 到 11/20）。G4 難拿時，先讓 test04 的第一次跑完（第一次可以用 A100），再讓競賽一去試 G4；test04 在跑的時候，請使用者不要同時去試競賽一。
4. **電腦「下載」裡的舊 zip**（不可逆，自成一則訊息）：請使用者看電腦的「下載」資料夾裡有沒有舊的 `test04_rank_results.zip`（彩排或上一次留下的），有的話只刪這一個，回報「刪好了」或「沒有」。等回報後才給下面的步驟。

**再給使用者步驟**：把 `docs/COLAB_RUN.md`「測試集 04（test04）」一節的步驟給使用者，並寫明：

- 這是第一次（G4 優先，灰色才改 A100），還是登記的重跑（**只用 G4**）。前面只有 `sw_exempt_N`（不占次數的封存）時，照第一次的規則。
- notebook 版本的 12 碼（上面第 1 點那一行印的）。review round 3 起，notebook 在寫任何紀錄、算任何東西之前，會跳出輸入框請使用者對照這 12 碼（COLAB_RUN 第 9–10 步）：
  - 一樣：使用者輸入 `y`，照常跑。
  - 不一樣：使用者輸入 `n`，notebook 停在第一個程式格，**還沒寫任何紀錄，不算一次執行**（同 GPU 關卡停下），同一個 Drive 資料夾照常可用、不改名。請使用者刪執行階段，從 COLAB_RUN 第 6 步重開（一定從 Drive 的 `C10` 開）；`C10` 裡的 `test04_rank.ipynb` 本身不對時（上傳副本核對過，不應該發生），先截圖給 Claude，不重傳。
  - 按下「全部執行」1 分鐘內沒跳出輸入框：開到的是沒有這個輸入框的舊版，請使用者馬上刪執行階段、截圖。這一版還沒給過使用者（彩排也會用有輸入框的版本），正常不會發生。發生時：Drive 的 `sw` 裡還沒有完成標記，就照「notebook 在 GPU 關卡停下」處理（同一個資料夾照常可用，notebook 會把那一次記成 `interrupted`，不算一次執行）；已經有完成標記，照下面「斷線」處理。
  - 使用者對不一樣的 12 碼仍輸入了 `y`，notebook 跑完：那份 zip 照常下載、交給 `unpack`，工具判 `not_committed_notebook`（接縫類、占次數，rev 3 第 9 條），照「重跑一次」做。不另外當成斷線。
- 做完第 15 步（zip 放好）就停，跟 Claude 說「放好了」和拿到哪種 GPU。

使用者在跑的時候，本機沒有要做的事。

## 第 4 步：收 notebook 的結果（先 commit，再做任何本機步驟）

使用者把 zip 放進 `results\test04\` 後：

```
PY tools/t04_rank.py unpack --zip results/test04/test04_rank_results.zip
```

- 檔名帶 ` (1)`（「下載」資料夾裡還有同名的舊檔）：先看 `unpack` 印的執行紀錄和開始時間，確認是這一次的，再用 `--zip "results/test04/test04_rank_results (1).zip"` 直接指定。不確定就請使用者刪掉「下載」裡的舊檔（一則訊息只做這件事），從 Drive 重新下載。另一個舊 zip 不要刪，它在 gitignore 裡；要重跑時 `archive` 會一起搬走。
- **每次都看 `unpack` 印的執行紀錄**（`run 1: completed (start …)`）。
- `unpack` 印「WARNING: Drive's sw_failed_* / sw_exempt_* folders … differ from the local archives」：notebook 在 Drive 上看到的封存資料夾（占次數的 `sw_failed_N`、不占次數的 `sw_exempt_N`）和本機封存對不上（只記錄，不擋）。先請使用者截圖 Drive 的 `C10/results/test04`，核對改名是否照 `archive` 印的名稱做；對不上就停下查清楚，再做第 5 步。

| 結束碼 | 情況 | 下一步 |
|---|---|---|
| 0 | 解開到 `results/test04/sw/`，寫 `results/test04/unpack_sw.json` | commit 兩者 → 第 5 步 |
| 1 | 接縫類失敗：notebook 的接縫沒過、`run_log` 有 `seam_failed`／`readonly_changed`／`final_check_failed`、完成標記不是最後一次執行的（接續過）、跑的不是已 commit 的 notebook | 不解開。照下面「重跑一次」。**這一次是登記的重跑**（`unpack` 印了「這是登記的重跑」，訊息寫「沒有第三次」）：不 archive，照「重跑一次」第 6 點停下（K5 閘門第 1 輪）。那一次如果也在 R20 上傳前跑（「R20 先」的核對排在這些之後，所以印的是接縫類的訊息），`archive` 會先判成 `before_r20_u1`、不占次數，下一次照下面「R20 上傳前跑的」那一列的 GPU 規則 |
| 2 | 「這不是重算的結果」 | Drive 上可能還是舊的 zip（包括已封存的那份接縫失敗的 zip：這個核對排在接縫核對之前，所以是 2 不是 1）：**不是接縫失敗、不 archive**。刪掉 `results\test04\` 裡這個 zip，請使用者等 Drive 上新的 zip 出現再下載（COLAB_RUN 第 14 步）。比的是算出輸出的那幾次執行（最後一次，和它之前不是停在版本確認、GPU 關卡、preflight 的執行）；同一個資料夾在封存前只停在 preflight 的那一次不算（K5 閘門第 1 輪修正，以前比 `runs[0]`） |
| 2 | 「登記的重跑只在 G4」 | 登記的重跑用了別的 GPU（已 commit 的 notebook 會在 preflight 擋下，所以多半是跑了別的 notebook）：輸出不用、不 archive，停下查清楚，先不給使用者任何步驟 |
| 2 | 「R20 上傳前跑的 test04 不用」（同一個 zip 先查到接縫類紀錄時是上面的 1；`archive` 兩種都先判 `before_r20_u1`，做法相同） | `PY tools/t04_rank.py archive --reason "run before R20's first upload"`（類別 `before_r20_u1`，不占重跑次數；`archive` 印的 Drive 名稱是 `sw_exempt_N`，notebook 不把它算進重跑次數），commit，再照「重跑一次」第 2–5 點請使用者重跑。GPU 照 `archive` 印的「下一次執行」那一行：前面沒有占次數的封存時照第一次的規則（G4 優先、A100 備用），`unpack` 也不會印「這是登記的重跑」；前面已有 `sw_failed_N` 時仍只用 G4（K5 閘門第 1 輪）。比的是算出輸出的那一次執行（和它之前沒有停在 preflight／GPU 關卡的執行）的開始時間和每個完成標記；同一個資料夾在 R20 上傳前只停在 preflight 或 GPU 關卡的那一次不算（review round 2） |
| 2 | `colab/test04_rank.ipynb` 沒 commit、不乾淨，或「is not self-consistent」 | 本機問題，不是 Colab 的問題：`git status`、`git diff colab/test04_rank.ipynb` 查清楚，還原成已 commit 的那一份再跑同一個指令；不 archive |
| 2 | 那次沒有產生接縫結果（`preflight_failed`、`interrupted`、`running`） | 照下面「斷線」 |
| 2 | 其他（測試包和 11/04 的不同、旗標不符…） | 照訊息處理；不確定就停下查 |
| 3 | K6 已過 | 最終 R20 → 第 7、8 步 |

- 只為了查看一個失敗的 zip（不寫進 repo）：
  `PY tools/t04_rank.py unpack --dry-run --allow-failed --zip "<zip>" --results-dir "<scratch>\look04" --t02-results-dir results/test02 --t02-out-dir results/test02/rank --tests-dir data/test`
  - `--allow-failed` 把接縫類的紀錄和正式 zip 的旗標（`rehearsal false`）都改成警告，只在 `--dry-run`、repo 外的資料夾可用。它不算一次 `unpack`，不影響重跑次數。

## 重跑一次（接縫類失敗，結束碼 1）

PREREG_04 rev 3 第 9 條：那次的 04 輸出一律不用、不上傳；換一個執行階段重跑一次。test04 自己有一次，和 test02 的分開算。

1. **封存**（工具自己判斷失敗類別和次數）：

   ```
   PY tools/t04_rank.py archive --reason "<原因，例如 notebook seam_failed>"
   git add -A results/test04
   ```

   - 它把 `results/test04/sw/`、`unpack_sw.json`、`rank/`（有的話）搬到 `results/test04/failed/attempt_N/`，把 `results/test04/` 裡所有 zip（含 ` (1)` 的）搬到 `data/test04_failed/attempt_N/`，`opp/` 留在原處。印出 N、要給使用者的改名（`Drive: sw -> …` 那一行和「給使用者」那一行，也記在 `attempt.json` 的 `drive_rename`），和下一次的 GPU 規則。
   - Drive 名稱依類別（review round 3）：占次數的封存 → `sw_failed_N`／`test04_rank_results_failed_N.zip`，下一次是登記的重跑（只 G4）；不占次數的（`before_r20_u1`、`bug_revision`）→ `sw_exempt_N`／`test04_rank_results_exempt_N.zip`，notebook 不把它算進重跑次數；前面沒有占次數的封存時，下一次照第一次的 GPU 規則，**已有 `sw_failed_N` 時下一次仍是登記的重跑、只用 G4**。以 `archive` 印的「下一次執行」那一行為準（也記在 `attempt.json` 的 `next_run`；K5 閘門第 1 輪以前那一行只看這一次算不算次數）。
   - 拒絕時：已經有 1 個占次數的封存（「登記只准重跑一次」）→ 停下查清楚，要再跑只能先寫 PREREG_04 修訂；已有 `t04_apply_report.json`、或 `opp` 登記判定 04 不用 → 那是登記的判定，不是接縫失敗，不能封存；「archive 要先核對 R20 的第一次上傳」→ `results/official_scores.json` 或 test02 的 `apply_test_report.json` 沒 commit 或不乾淨：先 commit 再 archive（核對不了就分不出這次是不是在 R20 上傳前跑的）；「opp_report.json 是本機漂移」→ 先照第 2 步結束碼 4 那一列查好本機、重跑 `opp`，再 archive。
   - 最後三種（R20 核對不了、repo 的 notebook 沒 commit 或不一致、`opp` 是本機漂移）是本機要先修好的前提，`--bug-revision N` 不解除（第 7 點）；修訂只解除前面三種。
   - commit，訊息「test04 attempt N archived: <原因>」。
2. **使用者改 Drive 的名稱**（不可逆；這一則訊息只做這件事）：Drive →`C10`→`results`→`test04`：資料夾和同一層的 zip **照 `archive` 印的「給使用者」那一行改名**（通常是 `sw` → `sw_failed_N`、`test04_rank_results.zip` → `test04_rank_results_failed_N.zip`；不占次數時是 `sw_exempt_N`、`test04_rank_results_exempt_N.zip`）。把兩個完整的新名稱寫在訊息裡，不要只給 N。改名前後各截一張圖。**等使用者回報、Claude 看過截圖，才給下一點。**
3. **使用者刪電腦「下載」裡的舊 `test04_rank_results.zip`**（只刪這一個；這一則訊息只做這件事）。等使用者回報。
4. 使用者照 COLAB_RUN「測試集 04」的「要重跑時」R3 重跑：先確認競賽一沒有開著的執行階段、電腦設成不睡眠（R3 會提醒），刪執行階段、開新的。占次數的封存之後**只選 G4**（不換到容許差較寬的 A100），G4 灰色就隔一段時間再試，最晚 11/15 中午；還拿不到，這一季不用 04，最終 R20。只有不占次數的封存（`sw_exempt_N`）時，照第一次的規則（G4 優先、A100 備用）。**這則訊息要附上第 3 步第 1 點那一行印的 12 碼**（notebook 開頭的輸入框要對照），並先照第 3 步第 1、3 點再核對一次上傳副本和競賽一。
5. 新 zip 放好後回第 4 步 `unpack`。它會核對真的重算了（zip 的 SHA-256 和封存的都不同、算出輸出的那幾次執行的開始時間都晚於封存時間、seam_val 完成標記的時間晚於封存的那一份；同一個資料夾在封存前只停在 preflight 的那一次不算），並印「這是登記的重跑（第 2 次、也是最後一次）」。
6. 第二次還是接縫類失敗（`unpack`／`apply` 的訊息寫「這是登記的重跑…沒有第三次」）：停下查清楚，不 archive。`archive` 會拒絕再封存；要放寬容許差只能另寫 PREREG_04 修訂（rev 4），修訂前不上傳 04。K6 照樣適用。
7. **`archive --bug-revision N`**（失敗是本機程式錯誤造成的，不占次數）：PREREG_04 rev N 要在**這一次失敗之後**才 commit（工具核對：晚於這次執行的開始、每個完成標記、`unpack_sw.json`、`attempt_verdict.json` 的時間）；一份修訂只換一次不占次數的重跑，較早的封存已經引用同一個 N 就拒絕，新的錯誤要寫 rev N+1。`official_log` 在 `prepare`／`final`／`status` 時也重算這兩條（review round 2）。Drive 名稱是 `sw_exempt_N`；下一次的 GPU：前面沒有占次數的封存時照第一次的規則（review round 3），已有 `sw_failed_N` 時仍只用 G4（K5 閘門第 1 輪；照 `archive` 印的「下一次執行」那一行）。工具讀 PREREG_04 用 UTF-8（review round 3 修正：以前在這台電腦一定以「unexpected internal error」結束）。

## 斷線（notebook 跑到一半停掉）

- **不接續**，不算一次執行（除非那個資料夾裡已經有接縫類失敗的紀錄：那時照下面「notebook 說 sw_interrupted_K 不是斷線」）。
- **最後一次執行是 `completed` 的資料夾不是斷線**（使用者跑完後又按了一次「全部執行」）：notebook 會停下說「這個資料夾已經跑完了…不要改名」（review round 3）。不改名、不重跑：請使用者照 COLAB_RUN 第 12–15 步下載同一份 `test04_rank_results.zip`，照常回第 4 步。
- 一則訊息一件事，依序請使用者（順序是先刪執行階段、再改名：「執行階段已中斷連線」出現時，VM 可能還在寫 Drive，先改名會讓輸出分散在兩個資料夾）：
  1. 截圖、說「斷線了」。Claude 回覆 N（從 1 起，每次斷線加 1），把 N 和原因記下來（寫進 RESULTS）。
  2. 「執行階段」→「中斷連線並刪除執行階段」，**等 1–2 分鐘**（讓還在寫的檔寫完、同步到 Drive）。回報做好了。
  3. Drive 的 `sw` 改名 `sw_interrupted_N`；有 `test04_rank_results.zip` 的話同樣改成 `test04_rank_results_interrupted_N.zip`。前後截圖，等 Claude 看過。
  4. 開新的執行階段，整份重跑（約 5 分鐘）。GPU 照被打斷的那一次的規則（第一次：G4 優先、A100 備用；登記的重跑：只 G4，notebook 看到 `sw_failed_N` 資料夾就只收 G4；只有 `sw_exempt_N` 不算）。訊息附上 12 碼（第 3 步第 1 點）。
- notebook 在 GPU 關卡、版本確認或 preflight 停下（還沒有任何完成標記）：不算一次執行，照 COLAB_RUN 處理後，在同一個資料夾再按「全部執行」。
- notebook 會自己擋「在舊資料夾接續」；本機 `unpack` 也會擋（完成標記不是最後一次執行的）。

### notebook 說「sw_interrupted_K 不是斷線，是接縫失敗」（review round 3）

會發生在：接縫格已經寫了 `seam_failed` 之後才斷線，使用者看到「執行階段已中斷連線」就照上面 D1–D3 改名了。之後每次執行都會在 preflight 停在這一則紅字（不算一次執行）。這個資料夾其實是一次接縫失敗，照「重跑一次」算次數。一則訊息一件事：

1. 請使用者先不要再執行、不要改名，下載 Drive `C10/results/test04/test04_rank_results_interrupted_K.zip`（**只有這一次**例外於 COLAB_RUN 第 14 步「不要下載別的檔名」；把完整檔名寫在訊息裡），放進 `results\test04\`，回報「放好了」。
   - 沒有這個 zip（在打包前就斷線）：改請使用者從 Drive 的 `C10/results/test04/sw_interrupted_K/` 下載三個檔 `run_log.json`、`seam_check.json`、`config.json`，Claude 放進 `results/test04/sw/`（這三個檔就是 `archive` 的證據）。
2. Claude 封存：

   ```
   PY tools/t04_rank.py archive --reason "sw_interrupted_K was a seam failure" --drive-folder sw_interrupted_K
   git add -A results/test04
   ```

   `--drive-folder` 讓印出的改名指向這個資料夾（`sw_interrupted_K` → `sw_failed_N`，`test04_rank_results_interrupted_K.zip` → `test04_rank_results_failed_N.zip`）。拒絕時照「重跑一次」第 1 點的說明（例如已經有 1 個占次數的封存：停下）。commit。
3. 請使用者照 `archive` 印的「給使用者」那一行改名（兩個完整的新名稱寫在訊息裡），改名前後各截一張圖。Claude 看過截圖（`sw_interrupted_K` 不在了），再照「重跑一次」第 3–5 點。重跑就在現在的 `sw` 裡（裡面已有封存前停在 preflight 的那幾次，不用改名）：`unpack` 只比算出輸出的那一次執行的開始時間，封存前停在 preflight 的那幾次不算（K5 閘門第 1 輪修正；以前這裡的重跑一定被判「不是重算的結果」）。

## 第 5 步：`apply`（本機算 04 的交件檔）

```
PY tools/t04_rank.py apply
```

- 前提（工具會檢查）：第 2 步是 0 而且已 commit；`results/test04/sw/`、`unpack_sw.json` 已 commit；notebook 的紀錄、dump、key 和 test02 一致；`colab/test04_rank.ipynb` 已 commit 且乾淨。
- 輸出在 `results/test04/rank/`：`seam_val/`（`rank_val_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`seam_local.json`）、`rank_test_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`submission_rank.csv`（`pred_B.csv` 的逐位元組複本）、`t04_apply_report.json`。以結束碼 1 或 4 結束時只留 `attempt_verdict.json`（和 `seam_val/`，只當紀錄）：`check_submission` 沒過時，已寫出的五個測試集輸出會被刪掉。
- 看報告：本機接縫（`seam_local`：通過、最大差、規則 B 差幾題、是否逐位元組相同）、6 個 Δκ（`colour_check` 的 public／private × O4、SWA、SWB）、三組重播失敗（`replay`）、`check_submission`（official）、和 R20 規則 B 不同的題數（`rule_b_differences_vs_r20`）。

| 結束碼 | 情況 | 下一步 |
|---|---|---|
| 0 | 04 可以用（`usable` true） | commit `results/test04/rank/` → 第 6 步 |
| 1 | 接縫類失敗（key 和 test02 不同、環境混用、完成標記不符、notebook 接縫重算不一致、本機接縫沒過） | commit `attempt_verdict.json` → 「重跑一次」。這一次是登記的重跑（訊息寫「沒有第三次」）時，照「重跑一次」第 6 點停下，不 archive |
| 3 | 登記的「04 不用」：任一個 Δκ ≤ 0／NaN／沒有子題、重播失敗 > 0.5% | commit 寫出的 `t04_apply_report.json`（沒有交件檔）；最終 R20 → 第 7、8 步；不重跑 |
| 3 | 登記的「04 不用」：T1 FAIL_OPENING、K6 已過（`opp` 已判 04 不用時也一樣） | 工具不寫任何檔：把 stderr 的訊息記進 RESULTS；最終 R20（FAIL_OPENING 時 R10）→ 第 7、8 步；不重跑 |
| 4 | 本機漂移（O4 的 Δκ 和 `opp` 記的不同、`.venv`、`check_submission` 沒過：刪掉這次寫出的五個測試集輸出，`attempt_verdict.json` 的 `stage` 是 `check_submission`） | commit `attempt_verdict.json`；停下查本機；查好後 `apply --rerun --reason "..."` |
| 2 | 前提不符（例如計數檔不見或 SHA 不同：照第 1 步那一列還原，再跑同一個 `apply`；不是重跑 `opp`） | 照訊息 |

- **11/15 前 commit**：K6 看的是 `t04_apply_report.json` 最後一次 commit 的時間。

## 第 6 步：上傳 04（兩則訊息）

1. 準備上傳紀錄：

   ```
   PY tools/official_log.py prepare --comp rank --file results/test04/rank/submission_rank.csv --reason "prereg04 Q02 first upload (PREREG_04 5.3)"
   ```

   - `official_log` 不信任報告的結論，會自己重算：R20 的第一次上傳已 `uploaded`；報告是正式模式、Q02；notebook 的各項核對都通過、前面沒有失敗的執行；本機接縫通過；6 個 Δκ 都 > 0；三組重播失敗都 ≤ 0.5%，seam_val 逐位元組相同；報告記的每個輸入現在已 commit 的 SHA 都沒變；封存紀錄沒有超過一次重跑；報告在 K6 前 commit。
   - 拒絕時照訊息。不能用 04 就是最終 R20（第 7 步）。
2. commit `results/official_scores.json`。
3. **訊息 A**（檢查結果＋請使用者確認；這則不放上傳步驟）：貼 `prepare` 印的「請確認這一份檔案」整段：比賽、隊伍、檔案、列數、SHA-256 前 12 碼、登記編號，以及「這一份是 04，不是 11/04 那份 R20；兩份在 800 題裡有 N 題的規則 B 等級不同」。請使用者只對這一份說 OK。
4. 使用者說 OK 之後，**訊息 B 只放上傳步驟**：
   - 比賽「競賽二：從圍棋棋譜預測玩家棋力等級」（TEAM_11248），比賽頁 https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403 （出處 `docs/REGISTRATION.md`）。同一個 AIdea 帳號還有競賽一和 C1，一定寫完整名稱和網址。
   - 在 PowerShell 貼這一行，從打開的資料夾拖 `submission_rank.csv`（SHA-256 前 12 碼 …）：

     ```powershell
     explorer.exe "D:\AI-Portfolio\competitions\aicup2026-go\results\test04\rank"
     ```

     **不要從 `results\test02\rank` 拖**：那是 11/04 的 R20，檔名一樣。
   - 傳完回報時間和 public 分數（數字或截圖）。
   - 上傳頁要求的檔名或格式和 `submission_rank.csv` 不同：先停下，不改檔、不上傳（D5）。
5. 使用者回報後：

   ```
   PY tools/official_log.py uploaded --id N --ok-at 2026-11-05T15:00 --uploaded-at 2026-11-05T15:05
   PY tools/official_log.py public --id N --score 0.xxxx
   ```

   時間沒寫時區就當台灣時間；`--ok-at` 是使用者在對話裡說 OK 的時間。然後 commit。
   - `public` 印 `INVESTIGATE`：只查錯誤（ID 對應、列順序、接縫），**不因為分數換方法**。
   - `public` 印「可能傳成 R20 的檔（兩份同名）」：請使用者到 AIdea 的上傳紀錄截圖（檔名、時間），Claude 核對。傳錯了就照第 7 步重傳正確的那一份（要重新取得 OK）。

## 第 7 步：最後一版（PREREG_04 5.3；11/23 23:59 前）

一筆 04 上傳紀錄「現在符合 5.3」＝以下全部成立（`official_log` 每次重算）：H4R 採用（`results/prereg04/RESULTS.md`、`e8b24f2`）；報告最後一次 commit 就是 `prepare` 記的那一個，而且在 K6 前；T1 是 PASS 或 NORMALIZED；6 個 Δκ 都 > 0；三組重播失敗都 ≤ 0.5%；報告和交件檔從 `prepare` 之後沒變；報告記的輸入都沒變；notebook 的各項核對都通過；封存紀錄沒有超過一次重跑。

- **有符合的 04**：最終是 04。最後一次上傳要是它。

  ```
  PY tools/official_log.py final --comp rank --id <04 那一筆> --basis "PREREG_04 5.3: H4R adopted; the 04 run meets every condition"
  ```

- **沒有符合的 04**（`opp` 或 `apply` 判 04 不用、K6 過了、或之後輸入變了）：最終 R20（T1 FAIL_OPENING 時 R10）。

  ```
  PY tools/official_log.py final --comp rank --id <R20 的第一次上傳> --basis "PREREG_04 5.3: no 04 entry meets the conditions (<哪一條>)"
  ```

  最後一次上傳不是 R20 時（例如已經傳過 04，之後才發現不符合），再 `prepare` 一次 `results/test02/rank/submission_rank.csv`（`--reason "final: R20 per PREREG_04 5.3 (<哪一條>)"`），照第 6 步的訊息 A、B 取得 OK 後重傳，讓它成為最後一次上傳。
- **04 符合，但找到 04 路徑的程式錯誤**：先寫並 commit PREREG_04 修訂（rev 4 起，指出具體錯誤和它破壞 5.3 的哪一條），再：

  ```
  PY tools/official_log.py final --comp rank --id <R20 的第一次上傳> --basis "..." --not-04-reason "PREREG_04 rev N: <錯誤和條件>"
  ```

  工具核對修訂那一列已 commit、而且是在 04 那一筆 `prepare` 之後才加的；只寫文字、沒有修訂，一律拒絕。重傳 R20 的 `prepare` 也要帶同一個 `--not-04-reason`。修訂編號、commit 和原因會記進 `official_scores.json`，也寫進 RESULTS。
- 04 符合全部條件時，沒有別的方法改交 R20：public 分數較低不是理由（VALIDATION 紀律 4）。
- **11/22、11/23 各跑一次** `PY tools/official_log.py status`：
  - 印「違反 PREREG_04 5.3」：有符合的 04，最終卻不是它（沒有引用修訂，或引用的修訂是在那一筆 04 之前寫的、沒有涵蓋它）→ 照本步重跑 `final`。
  - 印「最終選擇要設成 04」→ 照本步設。
  - 印「final 改 R20，再 prepare 並重傳 R20」→ 照本步做。
  - 「最後一次上傳就是它」要成立。11/23 前告訴使用者最後交的是哪一份。

## 第 8 步：結果

- `results/test04/RESULTS.md`：各步驟的數字（三組重播失敗、6 個 Δκ、seam_val 的 Δκ、本機接縫、GPU、時間）、斷線和重跑的次數與原因、上傳紀錄（分數只引 `results/official_scores.json`）、5.3 的判定、看過官方分數才做的任何選擇。由另一個代理獨立核對數字。
- 更新 `D:\AI-Portfolio\PORTFOLIO_MAP.md` 第 0 節 C10 那一行、repo 的 `CLAUDE.md`。commit。

## R20 側的已登記變動（D5、PREREG_02 rev 9 起、競賽一改 A100；rev 3 第 19 條，review round 1 擴大）

- 會發生的情況：PREREG_02 第 9 節第 11 條的 D5（官方定義變動）；`docs/TESTDAY.md` 開頭的 PREREG_02 rev 9（FAIL_STRUCT、候選盤數異常、修 bug）；PREREG_02 rev 7 第 2 條競賽一改 A100（要改 `colab/build_test02_player_nb.py`、`colab/test02_player.ipynb`，TESTDAY.md 允許 11/20 前提早改）；11 月若再打分，`results/score_ledger.csv` 多一列。
- R20 的檔怎麼改，由 test02 依 PREREG_02 決定（R20 的登記優先），改之前先告訴使用者。PREREG_04 的任何修訂都**不授權** R20 的任何變動。指紋、tag `r20-testday-pre04` 和 `tests/test_p04_hygiene.py` 都不動。
- 之後 `test_p04_hygiene.py` 的 `test_pinned_files_unchanged` 會一直失敗（指紋是固定的），所以第 1 步的閘門改成：**失敗的只有這一項（和下面這一項），而且第 1 步清單多出來的檔，正好是那份已 commit 的 PREREG_02 修訂列出的檔**。`test_t04_hygiene.py` 對 PREREG_01–03、打分紀錄帳、`docs/COLAB_RUN.md` 要求 6ec5e68 的每一行照順序還在（改到某一行時同樣照這個規則）；另外 review round 2 起，打分紀錄帳和 PREREG_01–03 要已 commit、乾淨，資料列數和 `| rev N |` 列都要和 6ec5e68 相同。PREREG_02 多一列修訂、打分紀錄帳多一列時，`test_ledger_and_prereg_rows_pinned` 會失敗並印出多出來的列：**不會自動算過**，照第 1 步那一列逐列核對、記進 RESULTS。這個測試只在設了 `C10_T04_GATE=1`（本檔的閘門指令）或 `C10_K5=1` 時跑（review round 3）：一般的 `PY -m pytest -q tests`（`docs/TESTDAY.md` 11/04 前的「全綠」就緒清單）會略過它，所以別的登記依規打分多出來的列，不會讓 R20 的就緒閘門變紅；04 的閘門照樣抓得到。
- K5 合併前（在分支上）另跑一次 `C10_K5=1 PY -m pytest -q tests/test_t04_hygiene.py`：這時這些檔要和 6ec5e68 逐位元組相同（CRLF 折成 LF 後）。不同就先查清楚是哪一份登記改的，再合併。
  - 多出來的只有競賽一的檔（檔名含 `player`）：不影響 04，不需要 PREREG_04 修訂。在 RESULTS 記修訂編號、檔名、改前後的 SHA（`git diff --stat r20-testday-pre04 -- <檔>`），04 照常往下做。
  - 多出來的有 R20 路徑的檔（例如 D5 改 `tools/t02_*.py`、`colab/test02_rank.ipynb`、`docs/TESTDAY.md`）：04 的所有步驟停下，直到 PREREG_04 的修訂 commit：列出改了哪些檔、改前後的 SHA，附 `git diff r20-testday-pre04`，證明只改了那份 PREREG_02 修訂寫的內容（D5 是檔名表）。04 的工具和 notebook 需要跟著改時同樣只改那一項，重建、冒煙、審查。修訂前不上傳 04；R20 照 PREREG_02 的規則上傳，不等 04。
  - R20 修 bug 後重跑了 test02（報告、開局分數或 key 變了）：已做的 04 不再符合 5.3（`official_log` 會印「輸入已變更」），最終 R20（第 7 步）。`opp`、`apply` 的重跑會拒絕，要重做 04 須先寫 PREREG_04 修訂（rev 4 起）說明為什麼可以重做。

## 彩排（選做；要使用者同意；建議 10/15–10/22，10/28–11/02 留作第二次）

- **9/30 使用者決定：越快越好，不刻意等日期。** test02 彩排 9/30 已交給使用者（假測試集 zip SHA-256 `90f5baa276af…`，`<S>` 是對話 `b3b12887` 的 scratch `…\scratchpad\rehearsal`，第 21 步先不做）；test04 的彩排在 K5 前的最後驗證通過、notebook 定版後接著做。

這是 11/05 前 test04 唯一一次在真的 GPU 上跑。`<S>` = scratch 底下一個新資料夾（同 `docs/TESTDAY.md` 彩排那一段）。彩排不上傳，輸出不 commit（`results/test04_rehearsal/` 在 gitignore 裡）。

- **順序**（和 TESTDAY.md 的「彩排一結束就收拾」、COLAB_RUN test02 第 21 步不同，要在 test02 彩排開始前的訊息裡就告訴使用者）：
  1. test02 彩排照 TESTDAY.md 彩排那一段做，**test02 兩份（rank、player）都跑完才開始 test04**。
  2. **test02 的第 21 步先不做**：test04 要讀同一份彩排的 `C10_test_v1.zip`。
  3. 使用者用同一份彩排 zip 在 G4 跑 `test04_rank.ipynb`（COLAB_RUN「測試集 04」的彩排小節），結果在 Drive 的 `C10/results/test04_rehearsal/`。給使用者步驟之前，Claude 先照第 3 步第 1 點核對上傳副本的 SHA、取出 12 碼，**把 12 碼寫進給使用者的步驟**（notebook 開頭的輸入框要對照；彩排也會問），並照第 3 步第 3 點確認競賽一沒有開著的執行階段。zip 放進 `results\test04\` 後，Claude 立刻移到 `<S>\downloads\`。
  4. 三份都跑完，才請使用者把 `test02_rank.ipynb`、`test02_player.ipynb`、`test04_rank.ipynb`、`C10_test_v1.zip` 四個檔一起收進 `old_rehearsal`（一則訊息），回報後再請使用者刪掉「下載」裡三個彩排的 zip（另一則訊息）。Claude 刪掉 `data\colab_upload\test02\` 裡的彩排檔。
  5. **第二次彩排之前**（第一次做過以後，不論下一次是 test02 或 test04 的彩排，包括 TESTDAY.md 自己排在 10/28–11/02 的 test02 彩排）：
     - 一則訊息只做這件事：請使用者在 Drive 的 `C10/results` 裡，把 `test02_rehearsal` 改名成 `test02_rehearsal_1`、`test04_rehearsal` 改名成 `test04_rehearsal_1`，改名前後各截一張圖。Claude 看過截圖（兩個舊名字都不在了）才給彩排步驟。
     - 不改名的話：test04 的 preflight 會看到上一次的完成標記而停下（訊息寫「斷線後不接續」，容易誤會）；test02 的彩排會沿用上一次的逐盤輸出，不是在 GPU 上重算，或因為 `_INPUT.json` 不同而停下。
     - 本機：Claude 先把 `results/test04_rehearsal/`（在 gitignore 裡）移到上一次的 `<S>` 底下，第二次用新的 `<S>`。
     - 收拾時 Drive 的 `C10` 已經有 `old_rehearsal`（裡面有同名檔）：第二次改收進新建的 `old_rehearsal_2`，不要取代舊檔。
- 本機：

  ```
  PY tools/t04_rank.py unpack --rehearsal --zip "<S>\downloads\test04_rank_results.zip" --results-dir results/test04_rehearsal --t02-results-dir "<S>\rehearsal" --t02-out-dir "<S>\local_rank" --tests-dir "<S>\data\test" --checks-dir "<S>\results\test02\checks"
  PY tools/t04_rank.py opp --rehearsal --workers 6 --results-dir results/test04_rehearsal --t02-results-dir "<S>\rehearsal" --t02-out-dir "<S>\local_rank" --tests-dir "<S>\data\test" --checks-dir "<S>\results\test02\checks"
  PY tools/t04_rank.py apply --rehearsal --results-dir results/test04_rehearsal --t02-results-dir "<S>\rehearsal" --t02-out-dir "<S>\local_rank" --tests-dir "<S>\data\test" --checks-dir "<S>\results\test02\checks"
  ```

- 預期：G4 上兩臂 `bitwise_equal`、本機 Q02 接縫逐位元組相同；A100 只要求接縫通過。
- 記進 `build_review.md`：Colab 的 torch、cuDNN 版本（和參照 `2.11.0+cu128`、`91900` 比）、兩臂 C0（s0、swap）的最大差、實際時間和運算單元。G4 上不是逐位元組相同、而且版本和參照不同時，記下離 1e-4 還有多少空間；最大差超過 1e-5 時先告訴使用者，在 11/04 前（還沒有任何測試資料）決定要不要寫 rev 4。
- 彩排後改了產生器：重建、重新冒煙和審查、commit，K5 前合併；上傳副本 `data\colab_upload\test04\` 也要換掉並重新核對 SHA。另發一則訊息（只做這件事）請使用者在 Drive 把 `old_rehearsal`（或 `old_rehearsal_2`）裡的 `test04_rank.ipynb` 改名成 `test04_rank_OLD.ipynb`，改名前後各截一張圖：Colab 的「最近」清單不顯示檔案在哪個資料夾，同名的舊版可能被誤開（第 3 步的 12 碼對照是第二道防線）。
