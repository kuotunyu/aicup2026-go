# 測試日操作手冊：競賽二 14（完全複製應變層，PREREG_14）

- 這份是 Claude 的指令順序，只管競賽二的 14。R20／04／07／08／09 照 `docs/TESTDAY.md`、`TESTDAY_04/07/08/09.md`，那幾份都不改。
- 依據：`docs/PREREG_14_RANK_EXACT_COPY.md`（第 2 節複製的定義、第 4 節覆寫規則、第 7 節程序、第 8 節截止、第 9 節失敗處理）。兩者不一致時，以登記為準。
- **一句話**：競賽二測試題裡有和官方訓練集「整盤一模一樣、同一邊、≥ 100 手」的棋時，那一題改成那位訓練玩家的等級；其他題照最終選定的那一份（底檔）不動。沒有這種棋（很可能）就什麼都不做。
- **只用 CPU**，幾分鐘。不用 GPU、不開 Colab。不讀 `data/local/`，不跑 `tools/score.py`。
- 執行環境：`PY`＝`.venv/Scripts/python.exe`；repo 根目錄、Git Bash，指令前加 `PYTHONDONTWRITEBYTECODE=1`。
- 每一步做完就 commit：只 `git add results/test14 results/official_scores.json`，不要 `git add -A`。

## `t14_copy.py` 的結束碼

| 碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成（scan 寫好；apply 產生 14） | commit，做下一步 |
| 2 | 拒絕：前提不符（工具或釘住值沒 commit、`data/test` 不是 official 或和 MANIFEST 不同、T1 沒 commit、`final_choice.rank` 沒設或不是已上傳的競賽二紀錄、輸出已存在）。有 traceback 也是 2 | 照訊息修好前提，再跑同一行；要重寫輸出加 `--rerun --reason "…"` |
| 3 | 登記的「不用」：沒有任何題能覆寫、覆寫的題等級都和底檔相同、T1 不是 PASS／NORMALIZED、模擬紀錄沒過或 K5-14 沒趕上、已經有 `not_used.json`（工具都會寫 `not_used.json`，`stage` 記是哪一步） | commit `results/test14/`；本層這一季不用，競賽二照原路徑，後面都不做（唯一例外：apply 判「覆寫的題等級都和底檔相同」之後底檔換了，見下面「底檔換了」） |
| 4 | 本機漂移：索引釘住值不同、訓練玩家跨等級、底檔的位元組和上傳紀錄不同、`check_submission --official` 失敗 | 停下查；K6-14 前修不好就不用（PREREG_14 第 9 節） |

## 第 1 步：scan（只記錄；11/04 test02 的 `data/test` 建好、T1 commit 之後）

1. 前提：`results/test02/checks/rank/t1.json` 已 commit（`docs/TESTDAY.md` 第 1、3 步）。
2. 跑 `PY tools/t14_copy.py scan`（建全部 100 萬盤的索引約 2 分鐘，核對 `results/test14/design/index_pins.json`）。
3. commit `results/test14/`。
4. 看 `results/test14/scan/scan.json`：
   - `questions_overridable` 是 0 → 結束碼 3，工具已寫 `not_used.json`。用白話告訴使用者「測試題裡沒有和訓練集一模一樣的棋，14 這一季不用」，之後都不做。
   - 大於 0 → 告訴使用者有幾題、各在 public／private 幾題（只說數字，不說來源玩家代號以外的任何東西），等第 2 步。
   - `files` 裡的 `opposite_side_copies_record_only`（同一盤棋但訓練集記在對手名下）只記錄，不用。
5. 這一步和 T2（`results/test02/checks/rank/t2.json`）不衝突：T2 照 PREREG_02 記錄，本層不開 split v2。

## 第 2 步：apply（底檔定了之後）

1. 前提：競賽二的最後一版已經照現行規則設好（`docs/TESTDAY_09.md` 第 5 步，或它指向的 08／07／04／R20 的步驟），也就是 `official_log.py final --comp rank` 已指向一筆**已上傳**的紀錄，而且 `PY tools/official_log.py status` 的競賽二區塊沒有「違反」。
2. 跑 `PY tools/t14_copy.py apply`，commit `results/test14/`。
3. 結束碼 3（覆寫 0 題，或覆寫的題本來就是那個等級）：14 不用，底檔維持最終，結束。
4. 結束碼 0：看 `results/test14/rank/t14_copy_report.json`：`questions_overridden`、`rows_changed`、`changed_by_pool`、`overrides`（每題的底檔等級和新等級）、`conflicts`（指向兩位以上訓練玩家的題，照登記不改）。
- **K6-14＝11/23 18:00**：`t14_copy_report.json` 最後一次 commit 要在這之前。

## 第 3 步：上傳 14（兩則訊息）

1. `PY tools/official_log.py status`：競賽二當天上傳次數 < 5（14 多用一次；滿了就不傳 14，底檔維持最終）。
2. `PY tools/official_log.py prepare --comp rank --file results/test14/rank/submission_rank.csv --reason "prereg14 exact-copy layer (PREREG_14 7)"`
   - prepare 會核對：報告是正式模式、`check_submission --official`、這個檔的 SHA、索引釘住值、T1、K6-14、底檔仍是 `final_choice.rank` 那一筆而且已上傳、現行規則（04／07／08／09）仍指定底檔、14 和底檔只有報告列出的題不同。被拒絕就照訊息處理，不繞過。
3. **訊息 A**：放檢查結果（prepare 印出的 OK 請求：覆寫幾題、public／private 各幾題、底檔是哪一份），請使用者對這一份說 OK。
4. 使用者說 OK 之後，**訊息 B** 自成一則：在 AIdea 競賽二（TEAM_11248）上傳 `results\test14\rank\submission_rank.csv`，寫 SHA-256 前 12 碼（先用 `Get-FileHash` 核對），並說明不要從 test02／04／07／08／09 的資料夾拖（六份同名）。
5. 使用者回報後：`uploaded --id N --ok-at … --uploaded-at …`、`public --id N --score …`，commit。
   - public 分數不進判定。`public` 印出 PREREG_14 的同檔警告時，請使用者核對 AIdea 的上傳紀錄。

## 第 4 步：最後一版

- `PY tools/official_log.py final --comp rank --id <14 的編號> --basis "PREREG_14 7"`
  - 底檔當初 final 時若用了 `--not-0x-reason`（例如底檔是 08、旁邊有 09），這裡要給同一組理由：04–09 的規則是套在底檔上判的。
- 底檔當初 final 時記下的 `--not-0x-reason` 會自動沿用（存在 14 紀錄的 `base_rule_reasons`）；命令列另外給的優先。
- 之後 `status`：競賽二「最終選擇」是 14、最後一次上傳就是它；04–09 的區塊照底檔判，不應出現「違反」；14 的區塊顯示「最終選擇是 14」。
- **14 成為最終之後，11/22–11/23「確認最後一版」**（AGENTS.md 測試日表、`docs/TESTDAY_09.md` 第 5 步）改成：跑 `status`；14 仍符合就重跑 `final --comp rank --id <14 的編號> --basis "PREREG_14 7"`，**不要**對底檔重跑 final（會被要求 `--not-14-reason`）。`status` 說底檔不再符合（例如 09 失效）時，照現行規則 final 到新的檔（不需要 `--not-14-reason`：14 跟著舊底檔一起失效），再照「底檔換了」處理。

## 底檔換了、或 14 出錯

- **底檔換了**（例如 11/22 `status` 說 09 不再符合，final 改成 08）：舊的 14 不再符合（`status` 的 14 區塊會說「final choice moved」），final 到新檔不需要 `--not-14-reason`。K6-14 之前：先照現行規則把 final 設成新的底檔，再 `PY tools/t14_copy.py apply --rerun --reason "base changed to #N (PREREG_14 7)"`，回到第 3 步。過了 K6-14：不用 14，底檔要是最後一次上傳（必要時重傳）。
- **14 本身有錯**（程式錯誤）：先寫並 commit PREREG_14 rev 3 起的修訂，`final --comp rank --id <底檔> --not-14-reason "PREREG_14 rev N: …"`，再 prepare（同樣要 `--not-14-reason`）並重傳底檔，讓它成為最後一次上傳。
- 重跑 apply 時，舊的 `not_used.json`、交件檔和 `overrides.csv` 會移到 `results/test14/superseded/`（不刪），一起 commit。
- 結束碼 3 之後不重跑（底檔換了的情形除外）。

## 第 5 步：結果

寫 `results/test14/RESULTS.md`：scan 的數字（複製、對手那一方、可覆寫、衝突）、apply 的數字（覆寫、改變、public／private）、上傳紀錄和 public 分數、最後一版；揭露（PREREG_14 第 10 節：複製的機率事前未知、沒在驗證集打分、只比對官方訓練集、不辨認真實的人）。

## 演練（不寫進 repo）

- `C10_SLOW=1 PY -m pytest -q -p no:cacheprovider tests/test_t14_dryrun.py`：用 fit 玩家重建第 6 節的假題（800 題），以 `--dry-run` 跑 scan 和 apply，結果要等於已 commit 的 `results/test14/design/sim.json`（約 5 分鐘）。
- 正式模式的 `scan`／`apply` 在 11/04 前跑不了（沒有 `data/test`、沒有上傳紀錄），會回結束碼 2，這是預期的。
