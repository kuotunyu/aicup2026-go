# 測試日操作手冊：競賽一 06（test06，11/05 起）

- 這份是 Claude 的指令順序，只管競賽一的 06（06-OFF＝Q04＋v2 風格特徵＋CSLS；06-ON＝06-OFF＋一對一開關）。**test02（R20 和競賽一的 Q04）照 `docs/TESTDAY.md` 做，那份不改**（它在 R20 的指紋裡）；競賽二的 04 照 `docs/TESTDAY_04.md`。06 不用 Colab，使用者不必多跑 notebook，`docs/COLAB_RUN.md` 不動。
- 依據：`docs/PREREG_06_PLAYER_STYLE.md` 第 2.7、4、5 節和 rev 3（第 11 節，測試日程序）；實作規格 `results/prereg06/design/spec.md` 第 7、8 節。兩者不一致時以登記為準。
- **競賽一的最終選擇**：`docs/TESTDAY.md` U5「兩場都是 prereg02（甲）」和 `docs/TESTDAY_04.md` 開頭「競賽一照 U5 不變」，對競賽一由 PREREG_06 第 5.4 節和本檔第 9 步取代。**06 判定之前不要執行 `official_log.py final --comp player`**；已經照 U5 設過也沒關係，第 9 步會重設，`status` 會提示。
- **TESTDAY.md 另外兩條，只對 06-ON 由 PREREG_06 第 5.5 節取代**（兩份手冊看起來矛盾時以這裡為準）：
  - TESTDAY.md「之後的上傳」只准 (a) 修 bug、(b) 之後的登記依本機驗證規則採用的方法、(c) 重傳最終選擇，「不為了在 public 排行榜試變體而上傳」→ **E1** 新增 (d)：06-ON，一季最多一次，條件和順序照 5.3。06-OFF 屬於 (b)，否決後重傳 06-OFF 屬於 (c)。
  - TESTDAY.md 第 7 步（T6）、PREREG_02 第 9 節第 9 條和 DATA.md「拿到測試集後」第 7 條「沒有官方明文就不用一題對一人」→ **E4**：對 06-ON，「官方明文」由 5.3 的條件（偵測器、T1、T2、防護層、AIdea 頁面）加上 public 否決取代。Q04 和 06-OFF 照舊不用一一對應。
  - **E2、E3**：public 分數只用在「06-ON 還是 06-OFF」這一個決定，而且只能把開關關掉；06 和 Q04 之間、V0–V3 之間都不看 public。
- `PY` = `.venv\Scripts\python.exe`（Git Bash 寫 `.venv/Scripts/python.exe`），一律在 repo 根目錄 `D:\AI-Portfolio\competitions\aicup2026-go` 執行，前面加 `PYTHONDONTWRITEBYTECODE=1`。
- **每一步做完就 commit，不論結束碼**（`feats_report.json`、`t06_seam.json`、`t06_pool.json`、`t06_submit_report.json`、`veto.json`、`not_used.json` 都要 commit）。用 `git add results/test06 results/official_scores.json` 指定路徑，**不要 `git add -A`**（04 同一天在跑時會把 04 的輸出混進來）。作者 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，不加 Co-Authored-By。`data/test06/**` 在 gitignore 裡，它的 SHA-256 記在會 commit 的報告裡。工具在正式模式要求 git 乾淨（沒有未 commit 的追蹤檔變更），04 的輸出沒 commit 時 06 也會以結束碼 2 拒絕：先把 04 那一步 commit 完。
- **結束碼 3（06 不用）之後，不重跑任何步驟**，最終照 U5 交 Q04（第 9 步）。唯一的例外是 test02 競賽一本身依 `docs/TESTDAY.md` 重跑（修 bug、notebook 重跑）：這時照本檔「test02 重跑了怎麼辦」從 `feats` 重做（PREREG_06 rev 3 第 26 條）。除此之外要翻盤，只能先寫並 commit PREREG_06 的修訂（rev 4 起，指出具體的程式錯誤和它破壞的是哪一條）。
- 11/04 當天只做 test02，不做 06。
- **06 和 04 同一天時**：同一時間只能有一則等 OK 的訊息 A，不分競賽。一條線的 prepare → 訊息 A → OK → 訊息 B → `uploaded`／`public` → commit 整輪做完，才開始另一條線的下一次上傳。

## 每一步都適用的規則

- 不開、不讀、不雜湊 `data/local/` 裡的驗證標籤檔；`t06_player.py` 只在 `feats --pool seam_val` 和 `seam` 讀 `data/local` 的兩份棋譜檔（`player_val_candidates.csv`、`player_val_questions.csv`），工具自己會擋其他路徑（也擋經由 junction 的實體路徑）。不跑 `tools/score.py`；不用 GPU；CPU 一個行程就夠。
- 凍結檔一律不改：R20 的 39 個檔（`results/prereg04/design/r20_path_fingerprint.json`）、04 的 21 個檔（`results/prereg06/design/p04_path_fingerprint.json`）、`results/prereg01`–`prereg06/**`（含 `results/prereg06/player/` 的 8 個 H6P 輸出）、`tools/p02_*.py`、`p06_*.py`、`c10data.py`、`score.py`、PREREG_01–05、`docs/VALIDATION.md`、打分紀錄帳。`tools/official_log.py`、`docs/PREREG_06_PLAYER_STYLE.md` 只准加行。
- 每次開工先跑一次防誤改測試（Git Bash，前面一定要有 `C10_T04_GATE=1`），再跑 `official_log.py` 的 tag 對新版測試（約 10 秒）：

```
C10_T04_GATE=1 PY -m pytest -q -rs tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py tests/test_p06_hygiene.py tests/test_p06_player_hygiene.py tests/test_t06_hygiene.py
PY -m pytest -q tests/test_t06_official_log.py -k TagVersusNew
```

  - **永遠在的已知失敗**：`tests/test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`。它印出打分紀錄帳在 6ec5e68 之後多出來的列；其中 4 列是 H6P（note 開頭 `prereg06 H6P`，2026-10-03，commit `190478c`），PREREG_04 rev 3 第 19 條允許（「只多了打分紀錄帳的列」）。`tests/test_p06_player_hygiene.py` 另外把這 4 列釘死（不准有第 5 列 H6P）。
  - **其他登記依規產生的失敗**（照 `docs/TESTDAY_04.md` 第 1 步那一列和「R20 側的已登記變動」的寫法；PREREG_06 rev 3 第 28 條）：失敗的只限下面這幾項，而且每一個多出來的檔、每一列都對得上一份**已 commit** 的登記：
    1. `test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned` 印出 H6P 以外的列：打分紀錄帳的列各自對得上一份已 commit 事前登記的那一次打分（例如 PREREG_07 的 H7R、PREREG_02 第 2 節的 split v2），PREREG_01–03 多出的 `| rev N |` 列就是那份修訂本身。
    2. `test_p06_player_hygiene.py::LedgerRows::test_later_rows_only_prereg07`：H6P 之後多出不是 PREREG_07 的列，核對方法同上。
    3. `test_p04_hygiene.py::R20TestDayPath::test_pinned_files_unchanged`（和 `test_t02_hygiene.py` 的 `Pins`）：多出來的檔正好是一份已 commit 的 PREREG_02 修訂列出的檔（rev 7 第 2 條競賽一改 A100：`colab/build_test02_player_nb.py`、`colab/test02_player.ipynb`；rev 9 起；D5）。
    4. `test_p06_hygiene.py::P04TestDayPath::test_pinned_files_unchanged`：多出來的只有 `results/test04/design/build_review.md` 加了第二次 test04 彩排的紀錄（`docs/TESTDAY_04.md` 彩排那一節要求寫在那裡），或一份已 commit 的 PREREG_04 修訂列出的檔。
    - 用 `git diff --stat r20-testday-pre04 -- <檔>`、`git diff --stat p04-testday-pre06 -- <檔>`、`git log --oneline -- results/score_ledger.csv docs/PREREG_0[1234]*.md` 核對，把修訂編號、檔名、改前後的 SHA、多出來的每一列和它的 commit 記進 `results/test06/RESULTS.md`。對得上：06 照常往下做。06 不讀競賽二的任何東西（PREREG_06 第 6 節），所以只有競賽二的檔變了不影響 06。**競賽一改 A100**（第 3 項）時 06 照常做，但 test02 的 seam_val dumps 多半不再和 prereg02 逐位元組相同，`seam` 會走 398/400 那條路，可能是結束碼 3（06 不用，最終 Q04）。
    - 對不上、或失敗的是別的測試：停下查是誰改了凍結檔。
  - `-rs` 不能列出 `test_ledger_and_prereg_rows_pinned` 或 `test_later_rows_only_prereg07` 被略過（被略過＝少了 `C10_T04_GATE=1`，重跑）。
- 工具拒絕時照訊息處理，不繞過。要改程式或放寬門檻，先寫 PREREG_06 修訂（下一版是 rev 4）。
- **`--rerun --reason` 只用在三種情況**：結束碼 2 的原因查好之後（例如輸出已存在、要重寫）、結束碼 4 的原因查好之後（理由寫查到的原因），以及本檔「test02 重跑了怎麼辦」。結束碼 3 之後一律不用。這一次 06 的任何一筆進了上傳紀錄之後，`feats`、`seam`、`apply`、`submit` 的 `--rerun` 都會被拒絕（結束碼 2：改寫會讓已記錄的 06 不再符合 5.4）；只有 test02 重跑過（test02 的報告和這一次 06 綁的不同）時，`feats`、`seam`、`apply` 才放行。
- 上傳：每一份檔、每一次上傳，都要使用者在對話裡對那一份說 OK，不沿用先前的同意。**訊息 A** 放檢查結果和 OK 請求；使用者說 OK 之後，**訊息 B** 自成一則、只放上傳步驟。
- 每則訊息 A 之前先跑 `PY tools/official_log.py status`，確認競賽一當天（台灣日期）的上傳次數 < 5（每場比賽各自算，`docs/REGISTRATION.md`）；已經 5 次就排到隔天。
- 三份交件檔同名（`submission_player.csv`）：Q04 在 `results\test02\player\`，06-OFF 在 `results\test06\player\off\`，06-ON 在 `results\test06\player\on\`。每次的訊息 B 都只給一個資料夾，並寫明 SHA-256 前 12 碼和「不要從另外兩個資料夾拖」。發訊息 B 之前，Claude 先在 PowerShell 核對那個資料夾裡的檔等於 `prepare` 記的 SHA（照 `docs/TESTDAY_04.md` 第 3 步第 1 點的做法）：

  ```powershell
  Get-FileHash -Algorithm SHA256 "results\test06\player\off\submission_player.csv" | Format-List Hash, Path
  ```

- 使用者的不可逆操作（AIdea 上傳）**一則訊息只給一件**，做完等使用者回報、Claude 核對後，才給下一件。
- **AIdea 上傳沒成功**（頁面錯誤、上傳紀錄裡沒有這一筆）：不記 `uploaded`。被拒的原因是格式：當成 D5 或 bug，停下查。網路或頁面錯誤：沿用同一筆 `prepare`，不另開一筆（06-ON 的第二次 `prepare` 會被拒絕，訊息寫「already prepared as entry N」，意思是已經 prepare 過，不是真的傳過），重發訊息 A 取得新的 OK、再發訊息 B；成功後用同一個 `--id` 記 `uploaded`，`--ok-at` 用新的 OK 時間。

## `t06_player.py` 的結束碼（每一步都一樣）

| 結束碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成 | commit，做下一步 |
| 2 | 拒絕：前提不符（test02 沒做完或沒 commit、Q04 的第一次上傳還沒記錄、git 不乾淨、`.venv` 版本不同、輸出已存在、T2 檔格式不對、seam_val 的 dumps 不是 test02 判過的那一份、這一次 06 的紀錄已經在上傳紀錄裡還要重跑 `submit`（`feats`、`seam`、`apply` 的 `--rerun` 也一樣，test02 重跑過的除外）、K6' 前已 commit 過一次 06 而 K6' 已過…）。凍結檔被改（`results/prereg06/player/` 的 8 個檔和釘住的 SHA 不同、打分紀錄帳不是正好 4 列 `prereg06 H6P`）也是 2：停下查是誰改的 | 照訊息修好前提，再跑同一個指令（要重寫輸出時加 `--rerun --reason`）。訊息是「unexpected internal error」（前面有 traceback）：那是工具本身出錯，停下查本機工具，**不是登記的判定** |
| 3 | 登記的「06 不用」：T1 FAIL_OPENING、K6' 已過而且 K6' 前沒有 commit 過一次 06、H6P 沒採用（95% 下界不 > 0，或 `results/prereg06/RESULTS.md` 沒寫「**採用**」）、兩池都退回 test02 的列、seam 沒過而且 test02 的 seam_val dumps 和 prereg02 不是逐位元組相同 | commit `results/test06/player/not_used.json`（和寫出的報告）；最終照 U5 交 Q04 → 第 9 步；不重跑（例外見開頭） |
| 4 | 本機漂移：seam 對不上，而 dumps 和 H6P 逐位元組相同（或和 dumps 無關的盤數、詞彙外 token 數不同） | commit `t06_seam.json`；停下查本機（`.venv`、程式、`data/test06` 的特徵檔），不請使用者做任何事；查好後 `--rerun --reason "<查到的原因>"`；要放寬只能寫 PREREG_06 修訂 |

- 工具先核對所有前提和登記的「06 不用」，才看自己的輸出存不存在，所以「輸出已存在」不會蓋掉結束碼 3。

## 日期與 K6'、K7'

- 11/04：只做 test02（`docs/TESTDAY.md`）。06 的第一步要等 Q04 的第一次上傳在 `results/official_scores.json` 記成 `uploaded` 並 commit。
- 11/05–11/06：06 的目標（CPU 約 15 分鐘，上傳 1–3 次）。test02 競賽一延誤（例如拿不到 G4）時 06 順延，K6'、K7' 不順延。
- **K6' 11/21（六）23:59:59**：`feats`（三池）、`seam`、兩池 `apply`、`submit`（至少 06-OFF）都要以結束碼 0 跑完，報告已 commit，而且 `t06_submit_report.json` 最後一次 commit 在 K6' 之前。過了之後 `feats`、`seam`、`apply`、`submit` 不再執行：K6' 前沒有 commit 過一次 06 是結束碼 3（這一季不用 06，最終 Q04）；K6' 前已 commit 過（`off/t06_submit_report.json` 的 commit 時間 ≤ K6'）是結束碼 2，那一次 06 照常有效。`veto`、`status` 和 `official_log.py` 照常運作，最終選擇照 `PY tools/official_log.py status`（第 9 步）。
- **K7' 11/22（日）23:59:59**：要用開關時，06-ON 已上傳、兩份 public 分數已記錄、`veto.json` 已 commit。過了還沒判定：不用開關，最終 06-OFF；06-ON 已經上傳的話，再上傳一次 06-OFF。
- 11/22、11/23 各跑一次 `PY tools/official_log.py status` 和 `PY tools/t06_player.py status`。最後一次上傳在 11/23 23:59 前（截止 11/24 23:59）。
- 競賽一每天最多上傳 5 次（Q04 和 06 合計）。06 通常最多 3 次：06-OFF、06-ON、（否決或 K7' 時）再一次 06-OFF；第 6 步核對出 06-OFF 傳錯時多一次。

## 第 1 步：閘門（test02 做完了嗎）

先全部核對，任何一項不對就停下，不做第 2 步。

```
PY tools/official_log.py status
PY -c "import json; t=json.load(open('results/test02/checks/player/t1.json', encoding='utf-8')); print(t['verdict'], t.get('complete'), t.get('stage'))"
git status --porcelain
PY -c "import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"
PY -c "import json; [print(p, json.load(open(f'results/test02/player/{p}/apply_report.json', encoding='utf-8'))['guard_level'], json.load(open(f'results/test02/player/{p}/apply_report.json', encoding='utf-8'))['gate_passed']) for p in ('public', 'private')]"
PY tools/t06_player.py status
```

然後跑上面的防誤改測試和 TagVersusNew。

| 項目 | 要看到 | 不是的話 |
|---|---|---|
| `status` | 競賽一有一筆 `results/test02/player/submission_player.csv` 已上傳（Q04 的第一次上傳），`results/official_scores.json` 已 commit | 先照 `docs/TESTDAY.md` 第 8 步做完 |
| T1 | `PASS` 或 `NORMALIZED`，`True raw` | `FAIL_OPENING`：06 整條不做（PREREG_06 2.7）。跑一次 `PY tools/t06_player.py feats --pool seam_val` 讓工具寫下 `not_used.json`（結束碼 3；`feats` 已經跑過也一樣是 3），commit，→ 第 9 步（`final` 設成 Q04，照 TESTDAY.md U5；11/22、11/23 照常跑 `status`）。`FAIL_STRUCT`：照 test02 停下 |
| `git status` | 空的 | 先 commit 或查清楚 |
| `.venv` | `3.12.13 2.5.3 1.18.1` | 停下，還原 `.venv`（11/24 前不升級） |
| 兩池的防護層級、閘門 | 記下來（第 1 層、`True` 是常態） | 不是停下的條件：第 3 層、第 0 層或閘門沒過的那一池，06 自動交 test02 的那 400 列（PREREG_06 2.7、rev 3 第 7 條）；第 2 層照常用 06，但整份沒有 06-ON |
| T2 | `results/test02/checks/player/t2.json` 已 commit（test02 第 3 步） | 沒有 T2 時 06-OFF 照做，06-ON 的條件不成立 |
| 防誤改測試 | 全過，只有上面「已知失敗」，和對得上已 commit 登記的「其他登記依規產生的失敗」 | 停下查 |
| TagVersusNew | 全過 | 停下查 `tools/official_log.py` 是誰改的 |

## 第 2 步：`feats`（三池的 v2 特徵）

```
PY tools/t06_player.py feats --pool seam_val
PY tools/t06_player.py feats --pool public
PY tools/t06_player.py feats --pool private
```

- 每池約 25–75 秒（10/3 演練的 `feats_report.json` 記的秒數：第一輪 24.3–49.4 秒；審查修正後那一輪 32.3–73.2 秒，當時同一台機器另有全套測試和 GPU 試跑在跑）。讀 `data/test` 裡 test02 用過的那兩份檔（工具核對 SHA-256 等於 test02 的 `apply_report.json`；T1 是 NORMALIZED 時就是正規化後的副本），seam_val 讀 `data/local` 的兩份棋譜檔。
- 輸出：`data/test06/player/<池>/feats.npz`（不進 git）和 `results/test06/player/<池>/feats_report.json`（盤數、不交替的盤數、詞彙外 token 數、特徵檔的 SHA）。
- 看報告：官方池應該全黑（`games.candidate.W`、`games.question.W` 都是 0）；不交替的盤數記進 RESULTS（官方 T1.4 容許最多 1%）。
- commit：`test06 feats: seam_val / public / private`。

## 第 3 步：`seam`（在本機驗證池把 H6P 重做一次）

```
PY tools/t06_player.py seam
```

- 只讀 test02 自己的 seam_val 判過的那一份 dumps（SHA-256 要等於 test02 `seam_val/apply_report.json` 記的輸入，不等是結束碼 2）。用它把 H6P 的 apply 整套重跑，和 `results/prereg06/player/` 比：選中的組合、兩個閘門的判定、偵測器開或關都要相同；dumps 和 prereg02 逐位元組相同時（正常情況），`top5_off.csv`、`top5_on.csv` 要和 `player_val_top5.csv`、`player_val_top5_switch.csv` 逐位元組相同，sr06、閘門的值、r_forced_cost 也要完全相同。
- 輸出：`results/test06/player/seam_val/{t06_seam.json, top5_off.csv, top5_on.csv}`。

| 結束碼 | 情況 | 下一步 |
|---|---|---|
| 0 | 通過 | commit → 第 4 步 |
| 3 | 沒過，而且 dumps 和 prereg02 不是逐位元組相同（test02 的 notebook 這次算出的 CNN 不同；這時規則是 398/400） | commit `t06_seam.json`、`not_used.json`；06 不用，最終 Q04 → 第 9 步 |
| 4 | 沒過，而 dumps 逐位元組相同 | commit `t06_seam.json`；停下查本機。不請使用者做任何事 |

## 第 4 步：`apply`（每池各自判斷）

```
PY tools/t06_player.py apply --pool public
PY tools/t06_player.py apply --pool private
```

- 每池照 PREREG_06 2.7 判斷：test02 那一池在第 3 層、第 0 層（選中只用 CNN 的組合）或閘門沒過（Q00）→ 那一池就是 test02 的 400 列（逐位元組相同）；否則跑 sr06、照 2.4 選組合、06 的閘門 sr(選中) − sr(R02) ≥ 0（R02＝test02 那一池實際交出的組合），沒過也交 test02 的列。
- 輸出：`results/test06/player/<池>/t06_pool.json`（判斷用的表、每個組合的 sr、選中的、閘門、偵測器和它用的 `r*_X`、輸入的 SHA）、`top5_off.csv`，需要時 `top5_on.csv`：public 只要用了 06 就一律寫（強制探測）；private 只在第 1 層、選中的組合「能用」、偵測器打開時寫。
- 第二池也退回 test02 的列時，工具結束碼 3（兩池都退回 test02：06 不用）→ commit 報告和 `not_used.json`，最終 Q04。另一池的報告只有在它是用現在這份 test02 報告和 seam 做的時候才算數（test02 重跑前留下的舊報告不算，交給 `submit` 判斷）。
- 看報告並記進 RESULTS：兩池選中的組合、sr06、06 閘門、`detector.r_forced_cost` 和 `r_star`、`opens`、Hungarian 改了幾題的第 1 名（`detector.changed_top1`，T6 的只記錄項）。
- commit：`test06 apply: public <選中> / private <選中>`。

## 第 5 步：`submit`

Claude 先核對 AIdea 競賽一的頁面（比賽頁 https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688 的說明、資料頁；需要登入才看得到時，請使用者截圖那兩頁，一則訊息只做這件事）：有沒有寫題目和候選不是一一對應，或同一位玩家可以出多題。`--aidea-page-checked` 裡直接引用看到的原文（或寫「沒有提到」）和網址：

```
PY tools/t06_player.py submit --aidea-page-checked "<11/0X 讀了哪幾頁（網址）、原文寫了什麼（例如：資料頁寫每池 400 位候選、400 題，沒有提到一題對一人與否）>"
```

頁面寫了同一位玩家可以出多題、或題目和候選不是一一對應（不論用什麼字），或 Claude 沒把握時，都改加 `--aidea-not-one-to-one`（06-ON 就不產生；這樣只會把開關關掉）。

- **`--aidea-page-checked` 一定要在第 6 步之前給**：這一次 06 的任何一筆（`prepared` 或 `uploaded`）進了上傳紀錄之後，`submit --rerun` 會被拒絕（結束碼 2：重寫報告會讓已記錄的 06-OFF 不再符合 5.4）。忘了給就是這一次 06 沒有 06-ON。
- 輸出 `results/test06/player/off/submission_player.csv`（public 400 列＋private 400 列），旁邊 `t06_submit_report.json`；PREREG_06 5.3 的條件全部成立時，另有 `results/test06/player/on/` 的同名兩個檔。
- 06-ON 的條件（報告的 `on_conditions`，每一條都有值）：private 選中的組合能用而且偵測器打開；H6P 的次要指令點估計 ≥ 0；兩池都是 400 題、400 位候選；T2(c) 兩份題目檔都是 0，而且 T2(a)、(b)、(d) 沒有不過；兩池都用了 06 而且都在第 1 層；AIdea 頁面已核對、沒有寫不是一題對一人；K7' 還沒過。
- 報告記下：`check_submission --official` 的結果、和 Q04 第 1 名不同的題數（分池）、06-OFF 和 06-ON 第 1 名不同的題數（分池）、`inputs_bound`（所有輸入的 SHA-256；`official_log` 之後會逐一核對）。06-OFF 的 `inputs_bound` 不含 T2 和 `top5_on`（那些只是 06-ON 的輸入），所以之後 T2 重寫不會讓已上傳的 06-OFF 失效；06-ON 的含。
- commit：`test06 submit: 06-OFF <sha12> [06-ON <sha12>]`。這一個 commit 的時間就是 K6' 看的時間。

## 第 6 步：上傳 06-OFF（兩則訊息）

1. 準備上傳紀錄：

   ```
   PY tools/official_log.py prepare --comp player --file results/test06/player/off/submission_player.csv --reason "prereg06 06-OFF first upload (PREREG_06 5.3 step 2; TESTDAY U4 (b))"
   ```

   `official_log` 不信任報告的結論，會自己核對：Q04 的第一次上傳已 `uploaded`；報告是正式模式、06-OFF、`check_submission` official 通過、記的就是這個檔；06 的 seam 和兩池報告已 commit；test02 競賽一的報告、本機接縫、完整的 raw T1（PASS／NORMALIZED）都已 commit；`inputs_bound` 的每個檔現在已 commit 而且 SHA 沒變；報告在 K6' 前 commit。拒絕時照訊息；不能用就是最終 Q04（第 9 步）。
2. commit `results/official_scores.json`。
3. 跑 `PY tools/official_log.py status`，確認競賽一今天的上傳 < 5 次、沒有另一則等 OK 的訊息 A（04 也算）。
4. **訊息 A**（檢查結果＋OK 請求；這則不放上傳步驟）：貼 `prepare` 印的「請確認這一份檔案」整段（比賽、隊伍、檔案、列數、SHA-256 前 12 碼、登記編號），以及「這一份是 06-OFF，不是 11/04 那份 Q04；和 Q04 第 1 名不同的題數 public N、private M」。請使用者只對這一份說 OK。
5. 使用者說 OK 之後，Claude 先用上面的 `Get-FileHash` 核對 `results\test06\player\off\submission_player.csv` 等於 `prepare` 記的 SHA，再發**訊息 B（只放上傳步驟）**：
   - 比賽「競賽一：從圍棋棋譜集辨識玩家」（TEAM_11249），比賽頁 https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688 。同一個 AIdea 帳號還有競賽二和 C1，一定寫完整名稱和網址。
   - 在 PowerShell 貼這一行，從打開的資料夾**用拖的**把 `submission_player.csv`（SHA-256 前 12 碼 …）拖到上傳頁：

     ```powershell
     explorer.exe "D:\AI-Portfolio\competitions\aicup2026-go\results\test06\player\off"
     ```

     **不要按上傳頁的「選擇檔案」再從跳出的視窗挑**（它會停在上一次用的資料夾），也**不要從 `results\test02\player` 或 `results\test06\player\on` 拖**：三個檔名一樣。
   - 傳完回報上傳時間，並**截 public 分數那一格**（Claude 照截圖把畫面上的全部位數抄進 `public --score`，不四捨五入：d 和同名檔警告都看這個數）。
   - 上傳頁要求的檔名或格式和 `submission_player.csv` 不同：先停下，不改檔、不上傳（D5）。
6. 使用者回報後：

   ```
   PY tools/official_log.py uploaded --id N --ok-at 2026-11-05T15:00 --uploaded-at 2026-11-05T15:05
   PY tools/official_log.py public --id N --score 0.xxxxxx
   ```

   時間沒寫時區就當台灣時間；`--ok-at` 是使用者在對話裡說 OK 的時間。然後 commit。
   - `public` 印 `INVESTIGATE`（例如 06-OFF 比 Q04 低 0.05 以上）：只查錯（ID 對應、列順序、池的配對、接縫），查不到錯就維持（PREREG_06 5.4 最後一段），寫進報告。**不因為分數換方法。**
   - **`public` 印 `WARNING (PREREG_06 5.3)`：Q04 和 06-OFF 的 public 分數完全相同**（差 < 1e-9），而第 1 名不同的題數 > 0：可能把 Q04 的檔當成 06-OFF 傳了。這時 `prepare` 06-ON 一律拒絕（PREREG_06 rev 3 第 19 條）。一則訊息只做這件事：請使用者截 AIdea 競賽一的「上傳紀錄」（每一筆的檔名、時間），Claude 核對那一次傳的是哪一份。
     - 核對出傳錯（那一次其實是 Q04 的檔）：照本步從 1 再做一次（同一個 06-OFF 檔重新 `prepare`、新的 OK、訊息 A／B、`uploaded`、`public`）。新的那一筆和前一筆是同一個 SHA、分數不同，`public` 會印 `WARNING (PREREG_06 rev 3) … 是同一份檔`：這是預期的，記進 RESULTS。新的 06-OFF 分數和 Q04 不同之後，06-ON 才能做，而且配對的是這一筆（最新一筆）06-OFF。
     - 核對出沒傳錯（分數真的相同），或核對不出：這一季不做 06-ON，最終 06-OFF（第 9 步）。原因記進 RESULTS。
     - **核對不出時，再上傳一次 06-OFF**（PREREG_06 5.3「要的話再上傳一次 06-OFF」；第二輪審查 minor 4）：AIdea 的上傳紀錄多半只看得到同一個檔名和時間，所以那一次實際計分的可能是 Q04 的檔，而紀錄寫的是 06-OFF。照本步從 1 再傳一次，讓最後一次上傳確定是 06-OFF。新分數和前一筆不同時 `public` 會印 rev 3 的「是同一份檔」警告：這是預期的，等於看出前一次傳錯了，記進 RESULTS。**不論新分數是多少，這一季都不做 06-ON**（PREREG_06 rev 3 第 19 條：核對不出就不做）。新分數和 Q04 不同之後，`official_log` 不會再擋 06-ON 的 `prepare`，所以要靠本條自己守住，不要做第 7 步。
   - `public` 印 `WARNING (PREREG_06 rev 3) … 是同一份檔`，而不是上面那一種情況：同一個 SHA 的檔分數一定相同，所以其中一次傳的不是這一份；照上面的方式截上傳紀錄核對，核對出來的結果記進 RESULTS，最後一次上傳要是規則指定的那一份（第 9 步）。

## 第 7 步：上傳 06-ON（只有第 5 步產生了 06-ON 才做；兩則訊息）

1. 先確認：06-OFF 已記成 `uploaded`、public 分數已記錄、`public` 沒有印 WARNING（同一次 06 執行，`official_log` 會核對 `run_id`，也會核對 Q04 和最新一筆 06-OFF 沒有同名檔警告）；K7' 還沒過；log 裡沒有 06-ON 的上傳（一季最多一次，PREREG_06 5.5 E1）。
   - **唯一可以不管的 WARNING**（第二輪審查 minor 3）：第 6 步核對出傳錯、重傳正確的 06-OFF 之後，`public` 印的 `WARNING (PREREG_06 rev 3) … 是同一份檔`。條件是三項都成立：它指的另一筆，就是第 6 步核對出傳錯、已記進 RESULTS 的那一筆；這一次（最新一筆）06-OFF 的分數和 Q04 不同；沒有印 `WARNING (PREREG_06 5.3)`（Q04 和最新一筆 06-OFF 分數完全相同）。其他任何 WARNING 都不做 06-ON。
   - 第 6 步的結論是「沒傳錯」或「核對不出」時，這一季不做 06-ON（就算之後再傳的 06-OFF 分數和 Q04 不同、`official_log` 不再擋）。
2. 準備：

   ```
   PY tools/official_log.py prepare --comp player --file results/test06/player/on/submission_player.csv --reason "prereg06 06-ON (PREREG_06 5.3 step 3, 5.5 E1: once a season)"
   ```

   commit `results/official_scores.json`。跑 `status` 確認今天的次數 < 5、沒有另一則等 OK 的訊息 A。
3. **訊息 A**：貼 `prepare` 印的整段，並用白話寫明：
   > 這一份是 06-ON。「一對一」＝假設每一題剛好對應不同的一位候選玩家。最終名次只看 private 那 400 題；這一份的 public 那一半一律打開一對一，除此之外和已經傳過的 06-OFF 一樣，所以兩份 public 分數的差 d 就是一對一的效果：d < 0 就不用（之後再交一次 06-OFF），d ≥ 0 就用。private 那一半要它自己的檢查同意才打開。傳完後 public 排行榜的分數可能下降，這是預期的。和 06-OFF 第 1 名不同的題數 public N、private M。

   請使用者只對這一份說 OK。
4. **訊息 B**：同第 6 步（先 `Get-FileHash` 核對），資料夾改成 `results\test06\player\on`，並寫「不要從 `results\test02\player` 或 `results\test06\player\off` 拖」。
5. 使用者回報後：`uploaded`、`public`（同第 6 步，分數照截圖的全部位數），commit，然後：

   ```
   PY tools/t06_player.py veto
   ```

   `veto` 把 06-ON 和「同一次 06 執行、比它早上傳的最新一筆 06-OFF」配對（`official_log` 用同一條規則）。

   | 結束碼 | 情況 | 下一步 |
   |---|---|---|
   | 0 | `veto.json` 寫好（`outcome`）：d＝public(06-ON) − public(06-OFF)。d < 0：`vetoed`，最終 06-OFF；d ≥ 0（含 0）：`kept`，最終 06-ON（它必須仍是最後一次上傳） | commit `veto.json`（K7' 前）→ d < 0 時第 8 步，否則第 9 步 |
   | 0 | K7' 已過：`k7_passed`，不判 d，最終 06-OFF | commit → 第 8 步 |
   | 2 | 同名檔警告：Q04、06-OFF、06-ON 任兩份 public 分數完全相同（差 < 1e-9），而報告記的 public 池第 1 名不同題數 > 0；`veto.json` 寫 `paused: true` | commit `veto.json`；下面「同名檔警告」 |
   | 2 | 其他（06-ON 比 06-OFF 早、沒有同一次執行的 06-OFF、缺 public 分數、報告在 prepare 之後變了） | 照訊息查 |

   - **同名檔警告**（PREREG_06 5.3、rev 3 第 19 條）：暫停時規則**什麼都不指定**：`official_log.py status` 印「否決暫停…核對前不重傳、不設最終選擇」，`final` 一律拒絕。**核對前不要重傳任何檔。** 一則訊息只做這件事：請使用者截 AIdea 競賽一的「上傳紀錄」（每一筆的檔名、時間），Claude 核對各是哪一份。然後只照下面三種之一重跑（都在 K7' 前，commit `veto.json`）：
     - 核對出沒傳錯（分數真的相同）：

       ```
       PY tools/t06_player.py veto --rerun --reason "AIdea upload record checked" --confirmed-upload-check "<核對了哪幾筆、各是哪一份>"
       ```

       d 照常判。
     - 核對出 06-OFF／06-ON 這一對裡有傳錯（包括 06-ON 那一次其實傳成 Q04 或 06-OFF 的檔）：**絕不用 `--confirmed-upload-check`**（d 會拿錯的分數算）。06-ON 不能再傳（E1），重傳的 06-OFF 又會在 06-ON 之後，所以這一季不用開關：

       ```
       PY tools/t06_player.py veto --rerun --reason "AIdea upload record checked" --wrong-upload "<哪一筆其實是哪一份>"
       ```

       最終 06-OFF → 第 8 步。
     - 核對不出、或 K7' 前來不及：

       ```
       PY tools/t06_player.py veto --rerun --reason "upload record inconclusive" --upload-check-inconclusive "<原因>"
       ```

       最終 06-OFF → 第 8 步。K7' 已過還沒判定時，`veto`（加 `--rerun`）會直接寫 `k7_passed`。
   - `official_log public` 對 06-ON 也會印 d 和否決的判定（只是印出來）；正式判定以 `veto.json` 為準。06-ON 比 06-OFF 低 0.05 以上時 `public` 會標 `INVESTIGATE`：照常快速查一次 ID、列順序、池的配對，但不管查到什麼，結果都是 06-OFF（PREREG_06 5.3 的 0.05 警示）。

## 第 8 步：再上傳一次 06-OFF（否決、上傳紀錄核對出傳錯或核對不出、K7' 已過時）

- `PY tools/official_log.py prepare --comp player --file results/test06/player/off/submission_player.csv --reason "final 06-OFF once more (PREREG_06 5.3 step 5 / rev 3 / K7')"`，commit，訊息 A／B 照第 6 步（同一個檔、同一個 SHA），`uploaded`、`public`，commit。目的只是讓 06-OFF 成為最後一次上傳。
- 這一次的 public 分數應該等於之前**沒有傳錯**的那筆 06-OFF。`public` 印 `WARNING (PREREG_06 rev 3) … 是同一份檔` 時，逐一看警告裡的「#N 和 #M」（第二輪審查 minor 1；`official_log` 會拿每一筆同 SHA 的紀錄來比，包括已知傳錯的那幾筆）：
  - 警告指的那一筆，**全部**都是之前已經核對出傳錯、記進 RESULTS 的（第 6 步核對出的，或第 7 步 `--wrong-upload` 寫進 `veto.json` 的那一筆，例如那一次其實傳了 Q04 的檔）：這是預期的。記進 RESULTS，不用再傳；之後每傳對一次都會再印同樣的警告，一樣處理。
  - 警告裡有任何一筆**不是**已知傳錯的（這一次和一筆沒傳錯的 06-OFF 分數不同）：這一次又傳錯了。照第 6 步最後一點截上傳紀錄核對，再傳一次正確的 06-OFF。
  - 之前沒有任何一筆沒傳錯的 06-OFF 可以比（例如唯一的 06-OFF 在第 7 步被核對出傳錯）：這一次的分數就是第一個正確的 06-OFF 分數，警告只會指向那幾筆已知傳錯的，照第一點處理。核對不出（`upload_check_inconclusive`）的那一筆不算已知傳錯：警告指向它時，照第 6 步最後一點截上傳紀錄核對這一次。
- **否決判定之後還印的 `WARNING (PREREG_06 5.3)`**（第二輪審查 minor 2）：只要 Q04、06-OFF、06-ON 裡有兩份分數完全相同，之後每次記競賽一的 `public` 都會重印這個警告，最後還寫「否決暫停」。如果 `veto.json` 已經 commit、`outcome` 不是 `paused`（例如 `wrong_upload`、`upload_check_inconclusive`、`confirmed_upload_check` 之後的 `kept`／`vetoed`），這只是在重複舊的分數比較，不是新的暫停。以 `veto.json` 和 `PY tools/official_log.py status` 為準：`status` 沒有印「否決暫停」，就照本步和第 9 步做，不要重跑 `veto`。把這個警告記進 RESULTS。
- K7' 已過、06-ON 已上傳、否決還沒判定：`t06_player.py status` 和 `official_log.py status` 都會印「再上傳一次 06-OFF」。

## 第 9 步：最後一版（PREREG_06 5.4；11/23 23:59 前）

`official_log` 每次重算「哪一份是規則指定的」：

- 一筆 06 上傳「符合」＝T1 是 PASS 或 NORMALIZED；報告最後一次 commit 就是 `prepare` 記的那一個，而且在 K6' 前；報告和交件檔從 `prepare` 之後沒變；報告記的每個輸入（test02 的 dumps 和報告、測試檔的 SHA、`frozen_config.json`、06 的 seam 和兩池報告；06-ON 另有 T2 和 `top5_on`）都沒變；06-ON 另外要 5.3 的條件都成立。
- 指定的那一份：
  - 沒有符合的 06-OFF → Q04（U5）。
  - 有符合的 06-OFF、沒有 06-ON → 06-OFF。
  - 06-ON 符合、`veto.json`（已 commit、在 K7' 前、配對的是同一次執行比 06-ON 早的最新一筆 06-OFF、d 和 log 的分數對得上）說 d ≥ 0，**而且 06-ON 仍是競賽一最後一次上傳** → 06-ON。06-ON 已經不是最後一次上傳（之後又傳了別的）→ 06-OFF（06-ON 不能再傳，E1）。
  - 否決、`wrong_upload`、`upload_check_inconclusive`、06-ON 不符合、K7' 前沒判定 → 06-OFF。
  - 06-ON 已上傳、K7' 未過，而否決還沒跑或暫停（同名檔警告）→ 還沒定（`final` 拒絕，先照第 7 步把 `veto` 做完）。

```
PY tools/official_log.py final --comp player --id <指定的那一筆> --basis "PREREG_06 5.4: <06-OFF / 06-ON (veto d = ...)/ Q04 (哪一條不成立)>"
PY tools/official_log.py status
PY tools/t06_player.py status
```

- 最後一次上傳要是指定的那一份（`status` 的「最後一次上傳就是它」）。不是的話：
  - 指定的是 06-OFF（例如最後一次是被否決的 06-ON）：照第 8 步再傳一次。
  - 指定的是 Q04（例如 06 沒用上、test02 重跑後 06 不再符合），而最後一次上傳是 06 的檔：再傳一次 Q04，訊息 A／B 照第 6 步，資料夾是 `results\test02\player`：

    ```
    PY tools/official_log.py prepare --comp player --file results/test02/player/submission_player.csv --reason "final: Q04 per PREREG_06 5.4 (<哪一條不成立>)"
    ```

- **06 的路徑找到程式錯誤**：先寫並 commit PREREG_06 修訂（rev 4 起，指出具體錯誤和它破壞的是哪一條），再 `final ... --not-06-reason "PREREG_06 rev N: <錯誤和條件>"`；要重傳 Q04 時，`prepare` 也要帶同一個 `--not-06-reason`（有符合 5.4 的 06-OFF 時，沒有它 `prepare` Q04 會被拒絕）。工具核對修訂那一列已 commit、而且是在被推翻的紀錄 `prepare` 之後才加的；只寫文字一律拒絕。
- public 分數較低不是改選的理由（VALIDATION 紀律 4；PREREG_06 5.5 的例外只有「06-ON 還是 06-OFF」這一個決定）。
- **11/22、11/23 各跑一次** `status`：印「違反 PREREG_06 5.4」→ 照本步重跑 `final`；印「最終選擇要設成 …」→ 照本步設；印「再上傳一次 06-OFF」→ 第 8 步；印「否決暫停」→ 第 7 步的同名檔警告。11/23 前告訴使用者最後交的是哪一份。

## test02 重跑了怎麼辦（PREREG_06 rev 3 第 26 條）

- test02 競賽一在 06 之後重跑（修 bug、notebook 重跑）：06 的輸入變了。`seam`、`apply`、`submit` 會拒絕（「test02 的輸入在這一步之後變了」），`official_log` 會判已上傳的 06 不再符合（「輸入已變更」），指定的那一份回到 Q04。
- 要重做 06：照本檔從第 2 步重跑，三池 `feats`、`seam`、兩池 `apply`、`submit` 都加 `--rerun --reason "test02 rerun <原因>"`，commit；新的 06-OFF 要重新上傳（新的 `run_id`）。之前因為 test02 的狀態得到的結束碼 3（兩池退回、seam 的 dumps 不同）也照這一節重做。K6' 照樣適用；06-ON 一季仍最多一次；06-ON 只和同一次執行（同一個 `run_id`）的 06-OFF 配對。

## 第 10 步：結果

- `results/test06/RESULTS.md`：防誤改閘門的結果（含「其他登記依規產生的失敗」逐項核對）；seam 的判定；兩池的選擇、sr06、06 閘門、偵測器（r_forced_cost、r*、開或關、改了幾題第 1 名）；不交替的盤數、詞彙外 token；06-ON 每一條條件的值和 AIdea 頁面的原文；上傳紀錄（分數只引 `results/official_scores.json`）；同名檔警告和上傳紀錄核對的結果；d 和否決；5.4 的判定。
- 「看了官方 public 分數才做的選擇」一節：寫出一對一開關的規則、兩份檔的 public 分數、d、有沒有否決，說明 public 池是強制打開的探測，以及 PREREG_06 5.5 的例外 E1–E4；06-OFF 比 Q04 低 0.05 以上時查了什麼。
- 由另一個代理獨立核對數字。更新 `D:\AI-Portfolio\PORTFOLIO_MAP.md` 第 0 節 C10 那一行、repo 的 `CLAUDE.md`。commit。

## 11/03 以前要就緒（K5' 核對表，11/03 23:59；目標 10/20）

- [ ] PREREG_06 rev 3（修訂紀錄那一列和第 11 節）已 commit，分支 `p06-testday` 已合併回 main。
- [ ] `results/test06/design/build_review.md` 記了：spec 9.3 的演練（Q04 逐位元組不變、防誤改閘門、TagVersusNew、06 對得回 H6P、官方形狀、負面演練）、完整測試、三個角度的審查和回應。
- [ ] 合併當下重跑 `C10_SLOW=1 PY -m pytest -q tests/test_t06_dryrun.py -k Q04Golden`，和 `q04_fake_golden.json` 相同。
- [ ] 合併後在 main（沒有 junction）跑 `PY -m pytest -q tests`（不設 `C10_SLOW`，單一行程）全綠，滿足 `docs/TESTDAY.md` 的就緒清單。`tests/conftest.py` 讓 `p06_player`、`c1h_lib` 在 import 時裝的 `data/local` 防護只留在用到它們的測試檔裡。在 worktree 跑全套時一律 `--deselect tests/test_t02_infra.py::Build::test_mode_guards --deselect tests/test_t02_infra.py::FromVal::test_guards`（junction 會讓它們寫進 main 的 `data/`）。
- [ ] 合併後、PREREG_07 的測試日程式動工前：替 06 的路徑做指紋 `results/prereg06/design/p06_path_fingerprint.json`（放在 `results/test06/design/` 會被自己的樣式涵蓋，所以和 04 的指紋放同一個資料夾；`tools/t06_player.py`、`tools/t06_fake_tests.py`、`docs/TESTDAY_06.md`、`tests/test_t06_*.py`、`results/test06/design/*`、`tests/conftest.py` 的 SHA；`tools/official_log.py`、`.gitattributes`、`.gitignore`、PREREG_06 只准加行）、打 tag `p06-testday-pre07`，並加一支釘住它的防誤改測試 `tests/test_t06_path_hygiene.py`（照 04 的 `p04_path_fingerprint.json` 做法）。
- [ ] `.gitattributes` 有 `results/test06/** -text`（對 main 只加這一行）。
- [ ] `CLAUDE.md`、`AGENTS.md` 的防誤改閘門指令加上 `tests/test_p06_hygiene.py`、`tests/test_p06_player_hygiene.py`、`tests/test_t06_hygiene.py` 和 TagVersusNew 那一行，「全過」改成本檔第 1 步的已知例外和「其他登記依規產生的失敗」（spec 9.2）。
- [ ] 第二次 test04 彩排（選做）的紀錄寫進 `results/test04/design/build_review.md` 時，06 的閘門會因 `test_p06_hygiene.py::P04TestDayPath` 失敗；那是第 1 步「其他登記依規產生的失敗」第 4 項，核對只多了彩排紀錄。
