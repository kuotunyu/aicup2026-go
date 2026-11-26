# 測試日操作手冊：競賽一 21（完全複製應變層，PREREG_21）

- 這份是 Claude 的指令順序，只管競賽一的 21。Q04／06／10 照 `docs/TESTDAY.md`、`TESTDAY_06.md`、`TESTDAY_10.md`，那幾份都不改。
- 依據：`docs/PREREG_21_PLAYER_EXACT_COPY.md`（第 2 節複製的定義、第 4 節覆寫規則、第 5 節和 06-ON／10／T2 的關係、第 7 節程序、第 8 節截止、第 9 節失敗處理）。兩者不一致時，以登記為準。
- **一句話**：競賽一某一題裡有一盤棋，和**同一池**候選庫裡某位候選的一盤棋「整盤一模一樣（含旋轉鏡射）、同手數、≥ 100 手、黑先交替、執色相同」，而且這一題所有這種棋都指向同一位候選時，那位候選放第 1 名，其他四名照最終選定的那一份（底檔）的順序往後挪；其他題照底檔不動。沒有這種棋（很可能）就什麼都不做。
- **只用 CPU**，幾秒到一兩分鐘。不用 GPU、不開 Colab。測試日只讀 `data/test` 的四份競賽一檔（不讀訓練集、競賽二或外部資料），不讀 `data/local/`，不跑 `tools/score.py`。
- 執行環境：`PY`＝`.venv/Scripts/python.exe`；repo 根目錄、Git Bash，指令前加 `PYTHONDONTWRITEBYTECODE=1`。
- 每一步做完就 commit：只 `git add results/test21 results/official_scores.json`，不要 `git add -A`。作者 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，不加 Co-Authored-By。

## `t21_copy.py` 的結束碼

| 碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成（scan 寫好；apply 產生 21） | commit，做下一步 |
| 2 | 拒絕：前提不符（工具沒 commit、`sim.json` 不是這一版工具寫的、T1 沒 commit、`data/test` 還沒 build、apply 時 K6-21 已過、`data/test` 不是 official 或和 MANIFEST／raw T1 不同、`final_choice.player` 沒設或不是已上傳的競賽一紀錄、底檔不在 `results/test02/player`、`test06/player/off`、`test06/player/on`、`test10/player` 之一、輸出已存在）。有 traceback 也是 2 | 照訊息修好前提，再跑同一行；要重寫輸出加 `--rerun --reason "…"` |
| 3 | 登記的「不用」：沒有任何題能覆寫、覆寫的題都和底檔逐列相同、T1 不是 PASS／NORMALIZED、模擬紀錄沒過、K5-21 沒趕上、已經有 `not_used.json`（工具都會寫 `not_used.json`，`stage` 記是哪一步） | commit `results/test21/`；本層這一季不用，競賽一照原路徑，後面都不做（唯一例外：apply 判「都和底檔相同」之後底檔換了，見下面「底檔換了」） |
| 4 | 本機漂移：底檔的位元組和上傳紀錄不同、`check_submission --official` 失敗 | 停下查；K6-21 前修不好就不用（PREREG_21 第 9 節） |

## 第 1 步：scan（只記錄；11/04 test02 的 `data/test` 建好、競賽一的 T1 commit 之後）

1. 前提：`results/test02/checks/player/t1.json` 已 commit，而且 `data/test` 已經 build（`docs/TESTDAY.md` 第 1、2 步；T1 判 FAIL_OPENING 時先照 `docs/TESTDAY_02_PASS.md` 做完再 build）。`data/test/MANIFEST.json` 還不存在時工具回 2、不寫任何檔（原始 T1 的 FAIL_OPENING 不會在虛手修補前就把本層關掉）。
2. 跑 `PY tools/t21_copy.py scan`（兩池各 40,000 盤建索引，約 1 分鐘）。
3. commit `results/test21/`。
4. 看 `results/test21/scan/scan.json`：
   - `questions_overridable` 是 0 → 結束碼 3，工具已寫 `not_used.json`。用白話告訴使用者「競賽一的題目裡沒有和同池候選庫一模一樣的棋，21 這一季不用」，之後都不做。
   - 大於 0 → 告訴使用者有幾題、各在 public／private 幾題（`overridable_by_pool`；只說數字，不說候選代號以外的任何東西），等第 2 步。
   - `conflicts`（指向兩位以上候選的題，照登記不改）、`files` 裡的 `opposite_colour_copies_record_only`、`other_pool_copies_record_only`（執色不同、另一池）只記錄，不用。
5. 這一步和 T2（`results/test02/checks/player/t2.json`）不衝突：T2 照 PREREG_02 記錄，本層不開 split v2。T2(c) > 0 或 T2(b) FAIL 時 06-ON 照原規則不產生（`t2_c_zero_and_abd_not_failed`）；21 不會讓 06-ON 重新成立。

## 第 2 步：apply（底檔定了之後）

1. 前提：競賽一的最後一版已經照現行規則設好（`docs/TESTDAY_10.md` 第 6 步；沒有 10 時 `docs/TESTDAY_06.md` 第 9 步），也就是 `official_log.py final --comp player` 已指向一筆**已上傳**的紀錄（06-ON 有產生時否決已判定），而且 `PY tools/official_log.py status` 的競賽一區塊沒有「違反」。不必等到 11/22：規則指定的那一份上傳之後就可以先設 final。
2. 跑 `PY tools/t21_copy.py apply`，commit `results/test21/`。
3. 結束碼 3（沒有題被覆寫，或覆寫的題本來就是那位候選第 1）：21 不用，底檔維持最終，結束。
4. 結束碼 0：看 `results/test21/player/t21_copy_report.json`：`questions_overridden`、`rows_changed`、`changed_by_pool`、`overrides`（每題的候選、底檔的 Top-5、新的 Top-5、候選在底檔的名次）、`conflicts`。
- **K6-21＝11/23 18:00**：`t21_copy_report.json` 最後一次 commit 要在這之前。

## 第 3 步：上傳 21（兩則訊息；21 是競賽一最後一次上傳）

1. `PY tools/official_log.py status`：競賽一當天上傳次數 < 5（21 多用一次；滿了就不傳 21，底檔維持最終）。
2. `PY tools/official_log.py prepare --comp player --file results/test21/player/submission_player.csv --reason "prereg21 exact-copy layer (PREREG_21 7)"`
   - prepare 會核對：報告是正式模式、`check_submission --official`、這個檔的 SHA、四份測試檔和報告讀的相同、T1、模擬紀錄、K5-21、K6-21、底檔仍是 `final_choice.player` 那一筆而且已上傳、現行規則（PREREG_06 5.4／PREREG_10 4.4，隱去 21 的紀錄）仍指定底檔、每一題覆寫照「候選第 1，其餘照底檔」、21 和底檔只有報告列出的題不同。被拒絕就照訊息處理，不繞過。
3. **訊息 A**：放檢查結果（prepare 印出的 OK 請求：覆寫幾題、public／private 各幾題、底檔是哪一份），請使用者對這一份說 OK。
4. 使用者說 OK 之後，**訊息 B** 自成一則：在 AIdea 競賽一（TEAM_11249）上傳 `results\test21\player\submission_player.csv`，寫 SHA-256 前 12 碼（先用 `Get-FileHash` 核對），並說明不要從 test02／test06 off／test06 on／test10 的資料夾拖（五份同名）。
5. 使用者回報後：`uploaded --id N --ok-at … --uploaded-at …`、`public --id N --score …`，commit。
   - public 分數不進判定。`public` 印出 PREREG_21 的同檔警告時，請使用者核對 AIdea 的上傳紀錄。

## 第 4 步：最後一版

- `PY tools/official_log.py final --comp player --id <21 的編號> --basis "PREREG_21 7"`
  - 06／10 的規則套在底檔上判（隱去 21 的紀錄）；底檔當初 final 時記下的 `--not-06-reason`／`--not-10-reason` 會自動沿用（存在 21 紀錄的 `base_rule_reasons`），命令列另外給的優先。
- 之後 `status`：競賽一「最終選擇」是 21、最後一次上傳就是它；06／10 的區塊照底檔判，不應出現「違反」或「再上傳一次」；21 的區塊顯示「最終選擇是 21」。
- 21 已上傳、還沒 final 的這段時間，`status` 的 06／10 區塊也照底檔判（不會叫你再傳底檔），只有 21 區塊說「違反 PREREG_21 7」：照本步 final 到 21 就好。
- **21 成為最終之後，11/22–11/23「確認最後一版」**（`docs/TESTDAY_10.md` 第 6 步、`docs/TESTDAY_06.md` 第 9 步）改成：跑 `status`；21 仍符合就重跑 `final --comp player --id <21 的編號> --basis "PREREG_21 7"`，**不要**對底檔重跑 final（會被要求 `--not-21-reason`）。

## 底檔換了、或 21 出錯

- **底檔換了**（例如 11/22 `status` 說 10 不再符合，規則改指定 06-OFF）：舊的 21 不再符合（`status` 的 21 區塊會說不符合的原因）。K6-21 之前：先照現行規則把 final 設成新的底檔（需要時照該手冊重傳，讓它成為最後一次上傳），再 `PY tools/t21_copy.py apply --rerun --reason "base changed to #N (PREREG_21 7)"`，回到第 3 步。過了 K6-21：不用 21，底檔要是最後一次上傳（必要時重傳）。
- **21 本身有錯**（程式錯誤）：先寫並 commit PREREG_21 rev 3 起的修訂，`final --comp player --id <現行規則指定的那一筆> --not-21-reason "PREREG_21 rev N: …"`，再 prepare（同樣要 `--not-21-reason`）並重傳它，讓它成為最後一次上傳。**底檔是 06-ON 時**：21 上傳後 06-ON 已不是最後一次上傳、而且不能再傳（PREREG_06 5.5 E1），所以照 PREREG_06 rev 3 改成 06-OFF（`docs/TESTDAY_06.md` 第 8 步重傳 06-OFF）。
- 重跑 apply 時，舊的 `not_used.json`、交件檔和 `overrides.csv` 會移到 `results/test21/superseded/`（不刪），一起 commit；前提判「不用」時，舊的 `not_used.json` 也先移過去。
- 結束碼 3 之後不重跑（底檔換了的情形除外）。

## 第 5 步：結果

寫 `results/test21/RESULTS.md`：scan 的數字（複製、衝突、執色不同、另一池、可覆寫）、apply 的數字（覆寫、改變、public／private）、底檔、上傳紀錄和 public 分數、最後一版；揭露（PREREG_21 第 10 節：複製的機率事前未知、沒在驗證集打分、只比對同池的官方競賽一測試檔、不辨認真實的人）。

## 演練（不寫進 repo）

- `C10_SLOW=1 PY -m pytest -q -p no:cacheprovider tests/test_t21_dryrun.py`：用 fit 玩家重建第 6 節的假池（800 題），以 `--dry-run` 跑 scan 和 apply，結果要等於已 commit 的 `results/test21/design/sim.json`；另有「沒有任何複製」（結束碼 3）和「底檔被改過」（結束碼 4）兩個演練（約 2 分鐘）。樹放 `C10_DRYRUN_DIR`（repo 以外）或暫存資料夾。
- 正式模式的 `scan`／`apply` 在 11/04 前跑不了（沒有競賽一的 T1、沒有上傳紀錄），會回結束碼 2、不寫任何檔，這是預期的。
