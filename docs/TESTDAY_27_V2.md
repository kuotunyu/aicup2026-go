# 測試日 v2（11/11 起）：Claude 的順序

- 登記：`docs/PREREG_27_TESTSET_V2.md`（rev 1、rev 2）。和其他 `TESTDAY_*.md` 不一致時，**只有**本頁寫到的地方（封存、分支、ingest 的來源、上傳紀錄）以本頁為準；其他每一步照原頁，一字不改。
- `PY` = `.venv/Scripts/python.exe`（Git Bash，repo 根目錄，`PYTHONDONTWRITEBYTECODE=1`）。作者 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，不加 Co-Authored-By。**每一段做完就 commit**，並更新 AGENTS.md 現況和地圖 C10 那一行。
- 絕不開、讀、列出或雜湊 `data/local`；不跑 `tools/score.py`。
- 上傳：常設同意（11/04 15:29），檢查全過後 Claude 用使用者的 Chrome 上傳；`official_log.py uploaded` 的 note 寫「PREREG_27 v2; standing approval」。每場每天 5 次，**先讀 AIdea「今日已上傳 N 次」**。

## 0. 11/11 11:00 之前

1. `git status` 乾淨；tag `testset-v1-final` 存在（`git rev-parse testset-v1-final`）。
2. `nvidia-smi`：沒有別的工作、驅動程式 **616.92**（不是就停，請使用者裝回；Claude 不改系統設定）。
3. 用 Chrome 看兩場 AIdea 頁面的公告（競賽二有沒有公告、排行榜有沒有清空），截圖或抄下原文，存 `data/reference/`。

## 1. 下載、probe、決定分支（11:00 後，CPU 幾秒）

使用者照 `docs/TESTDAY_USER_CHECKLIST.md`「v2」第 1 步下載**兩場**的資料，告訴 Claude 資料夾（下面寫 `<DL>`）。

```
PY tools/testset_v2.py probe --src "<DL>"
```

- 結束碼 0＝結構和 v1 相同（題數、人數、盤數、檔名、id 前綴）。結束碼 4＝`needed_changes` 有東西 → **停**：PREREG_27 第 7 節，先寫 rev 3（只改結構常數），commit 後才往下。`missing_files` 有東西＝下載不完整，請使用者補下載。
- 競賽二分支（PREREG_27 第 2 節）：`rank_bytes_identical_to_v1` 是 false → **RR**；true 而且頁面沒公告、排行榜沒清 → **K**；true 但頁面要求重交 → **R0**。
- commit：`results/testset_v2/probe.json`（訊息寫分支和理由）。

## 2. 封存 v1（CPU 幾分鐘：要雜湊 `data/test*`）

```
PY tools/testset_v2.py archive --branch K      # 或 R0、RR
git commit -m "PREREG_27 archive v1 (K)"       # 工具已經 stage 好，只 commit
```

- 結束碼 1 或印 WARNING（index 裡有封存以外的東西）：先看清楚，不 commit 別的東西。
- **失敗時**：工具會把這次搬過的搬回去、`ARCHIVE.json` 的那一階段標 `failed`。查原因（多半是某個 `data/test*` 檔被別的程式開著），然後 `git checkout -- results/archive_testset_v1/ARCHIVE.json`（第一次就失敗時改成刪掉 `results/archive_testset_v1/` 裡未追蹤的兩個檔），再跑一次。
- 之後跑閘門（AGENTS.md），允許失敗照 PREREG_27 10.3（原本 6 項＋`test_t25_hygiene.py::Frozen::test_no_change_since_tag`；RR 剛封存時 p09／t19 那兩項會暫時通過）。
- K 之後競賽二才被清：再跑 `archive --branch R0` 或 `--branch RR`（只做剩下的）。

## 3. 使用者：Drive 改名（一則訊息一件事）

- K／R0：`C10/results/test02/player` → `player_v1`。RR：`C10/results/test02` → `test02_v1`、`C10/results/test04` → `test04_v1`。
- `C10/C10_test_v1.zip` 移進新資料夾 `C10/old_v1/`。
- 請使用者截 `C10` 和 `C10/results`，確認沒有 ` (1)`。

## 4. ingest、T1、HD、build（照 `docs/TESTDAY.md` 第 0–2 步＋`docs/TESTDAY_22_ROOTPROPS.md`）

- **K／R0**：先做乾淨的來源，再 ingest：

```
PY tools/testset_v2.py rank-src --src "<DL>" --dest data/test_v2_src
PY tools/t02_test_bundle.py ingest --src data/test_v2_src
```

  `rank-src` 會確認兩個 rank 檔和 v1 原檔逐位元組相同，把 v1 的正規化 rank 檔和下載的 4 個競賽一檔放進 `data/test_v2_src`，並寫 `results/testset_v2/rank_src.json`（commit）。之後**只跑競賽一的 T1**（`t1 --comp player`）；`results/test02/checks/rank/` 不動（25 綁著它）。`root_props.py prepare` 會說 rank `not_needed`、player `apply`；TESTDAY_22 第 2 步只重跑 `t1 --comp player`。
  - 任何工具要求重跑競賽二的 T1：**停**，告訴使用者，先請他決定競賽二最後一版（25）再說（PREREG_27 第 2 節）。
- **RR**：`ingest --src "<DL>"`，兩場都照原頁跑。
- 接著照原頁：T0、T1（FAIL_OPENING 時 `TESTDAY_02_PASS.md`）、`root_props`、`build --kind official`、複製 3 個檔到 `data/colab_upload/test02/` 並核對 SHA-256。

## 5. Colab（使用者）與 T2–T5

- 使用者照清單上傳 Drive、跑 notebook：K／R0 只跑 `test02_player.ipynb`；RR 兩份都跑。執行階段 **2026.07**、終端機確認 cu128。
- 同時 Claude 跑 `t02_checks.py run --comp player`（RR 另加 `--comp rank`）。

## 6. 本機與上傳（照原頁）

- 競賽一：`TESTDAY.md` 第 4、6、7 步 → 上傳 Q04 → `TESTDAY_06.md`（06-OFF → 條件成立才 06-ON → 否決）→ `TESTDAY_10.md`（B′ 的 P1 照 PREREG_27 10.5：build 一好就在 4090 上先跑）→ `TESTDAY_21.md`。
  - **PREREG_28（11/06 加行）**：`TESTDAY_06.md` 第 5 步的 submit 在 v2 上一律改用 `PY tools/t28_n.py submit --aidea-page-checked "<v2 頁面原文>"`（參數相同；400×400 時結果和 `t06_player.py submit` 相同；每池 N 題＝N 位候選、400 ≤ N ≤ 800 時尺寸條件照 PREREG_28 成立）。v2 上**不要**再用 `t06_player.py submit`（含 `--rerun`）：它會照 400×400 的字面把 `on/` 的 06-ON 刪掉。submit 後把 `off/t06_submit_report.json` 的 `on_conditions.pools_400_by_400.detail` 每池的 N 寫進 RESULTS。
- 競賽二 RR：`TESTDAY.md` 第 4、5 步 → 上傳 R20（**記錄好之後**才請使用者跑 test04）→ `TESTDAY_04.md` → `TESTDAY_07.md`（4090）→ `TESTDAY_08.md`（4090）→ `TESTDAY_09.md` → `TESTDAY_19.md` → `TESTDAY_25.md` → `TESTDAY_14.md`。
- 競賽二 R0：build 好之後（`data/test` 已重建）`official_log.py prepare --comp rank --file results/test25/rank/submission_rank.csv --reason "PREREG_27 R0: re-upload 25"`，上傳一次。
- 競賽二 K：什麼都不做。
- 每份上傳後（只記錄）：`PY tools/testset_v2.py compare --v1 results/archive_testset_v1/<v1 同一版的檔> --v2 <新檔>`，數字寫進 RESULTS。

## 7. 資源與時間（v1 實測 × 假設的 v2 大小 1.5 倍／2 倍）

| 步驟 | 在哪 | v1 實測 | ×1.5 | ×2 |
|---|---|---|---|---|
| test02 player notebook | Colab G4 | 771 秒 | 約 19 分 | 約 26 分 |
| test02 rank notebook（RR） | Colab G4 | 175 秒 | 約 4 分 | 約 6 分 |
| test04 notebook（RR） | Colab G4 | 359 秒 | 約 9 分 | 約 12 分 |
| B′ P1 兩池（10） | 4090 | 約 5.4 小時（演練估） | 約 8 小時 | 約 11 小時 |
| 07 HS（RR） | 4090 | 2.6 小時（接縫 7 分＋兩池 2.5 小時） | 約 3.9 小時 | 約 5.1 小時 |
| 08 P1（RR） | 4090 | 0.57 小時 | 約 0.8 小時 | 約 1.1 小時 |
| root_props prepare／verify | CPU | 2 分多／1 分 40 秒（演練） | — | — |
| t02 opening／apply；06；09／19／25；14、21 | CPU | 27 秒／1 秒；約 15 分；各 10 秒內；100 秒、幾秒 | 分鐘級 | 分鐘級 |

- 接縫（seam_val）只跑驗證集，和 v2 大小無關（v1 的時間就是 v2 的時間）。
- 4090 一次一個工作；超過一小時的排夜間（使用者同意）；每次開跑前 `nvidia-smi` 和 616.92。
