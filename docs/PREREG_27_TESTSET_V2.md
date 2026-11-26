# 事前登記 27：官方測試集 v2（11/11 重新開放）的重跑規則

**白話版**：11/06 官方在競賽一頁面公告：11/06 11:00 起暫停上傳、清空排行榜，11/11（三）11:00 重新開放，公布「原有＋新增」題目的測試集（格式當天說明），之前的上傳全部要重交、重新計分。競賽二頁面 11/06 11:10 沒有公告、排行榜沒清。本登記只寫「怎麼把已登記的流程原樣搬到 v2」：**方法、門檻、選擇規則一個都不改**；v1 的輸出怎麼保存、v2 的輸出寫在哪、上傳紀錄怎麼記、上傳順序、截止點，以及 11/11 的格式和 v1 不同時怎麼辦。

- 決定者：C10 主 session（使用者 10/2、10/9、10/16 把對排名有幫助的方法決定交給 AI；11/04 15:29 起 AIdea 上傳是常設同意）。選最後一版、審報告仍是使用者本人。
- 適用：競賽一（TEAM_11249）一定適用；競賽二（TEAM_11248）照第 2 節分支。
- 名詞：**v1**＝11/04 開放的測試集（各場 public 400＋private 400 題，MD5 和官網相同，`data/test_rootprops/original/`）；**v2**＝11/11 公布的測試集。**tag `testset-v1-final`**＝v1 所有測試日輸出和上傳紀錄都在、還沒動任何東西的 commit。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | （本檔第一次 commit） | 2026-11-06，分支 `v2-prep`。commit 時：v2 測試集還沒公布；repo 裡**沒有**任何 v2 檔、沒有 `tools/testset_v2.py`、沒有 `results/archive_testset_v1/`；v1 的輸出原樣在 `results/test02…test25`、上傳紀錄 `results/official_scores.json` 共 10 筆（競賽一 #2、#4、#5，競賽二 #1、#3、#6–#10）。先 commit 本檔，之後才寫程式。 |
| rev 2 | （本列所在的 commit） | 2026-11-06，合併前：工具的實作差異、演練結果、確定的允許失敗（第 10 節）。commit 時：工具（`ab9a75c` 起的 4 個 commit）已在分支 `v2-prep`；v2 還沒公布；只在 repo 外的複本和 scratch 樹演練過，main 的 v1 輸出和上傳紀錄都沒動。 |

之後的修訂只准加行（不改本表已有的列和第 0–9 節的原文）。預定：rev 2＝工具的實作差異、演練結果、確定的允許失敗清單（合併前）；rev 3 起＝11/11 看到格式之後的結構修訂（第 7 節），一律在任何 v2 預測之前 commit。

## 0. 原則

1. 已登記、已凍結的流程和選擇規則**原樣**套用在 v2：PREREG_02（含 rev 9／rev 11＝PREREG_22）、06、10、13、14、19、21、25，以及它們引用的 04、07、08、09、11。工具、模型檔、門檻、接縫標準、上傳條件、最後一版的規則都不改。
2. 不因 v1 或 v2 的 public 分數改任何決定；登記本來就看 public 的只有 PREREG_06 5.3 的 06-ON 否決（d），在 v2 上用 v2 的兩個 public 分數重新判。
3. 接縫（seam_val）只用本機驗證集，和測試集無關；v2 照樣要重跑、照樣要過。
4. v1 的任何預測、分數、接縫都**不**拿來代替 v2 的。v2 和 v1 相同 id 的題目，只記錄（第 6 節），不重用、不比分。
5. 本登記之後才寫 `tools/testset_v2.py` 和它的測試；11/11 之前不碰 AIdea、不碰 Drive、不上傳。

## 1. v1 怎麼保存（不刪、可重現）

- **tag `testset-v1-final`**：打在 v2-prep 合併前的 main（含 v1 全部測試日輸出、`results/official_scores.json` 10 筆）。之後永遠不移動、不刪。
- **封存位置**：`results/archive_testset_v1/`，底下保留原本的相對路徑（例：`results/test02/player/submission_player.csv` → `results/archive_testset_v1/results/test02/player/submission_player.csv`）。用 `git mv`，內容逐位元組不變；工具在搬之前和之後都比對 tag 裡的 blob。`.gitattributes` 加一行 `results/archive_testset_v1/** -text`（只准加行）。
- **要搬的檔**＝v1 測試日才產生的檔：`git diff --diff-filter=A --name-only f21ff1e testset-v1-final -- results/test*`（`f21ff1e`＝11/04 11:00 前 main 的最後一個 commit），依第 2 節的分支取其中幾組：
  - **P 組（競賽一）**：`results/test02/player/**`、`results/test02/unpack_player.json`、`results/test02/checks/player/**`、`results/test06/**`、`results/test10/**`、`results/test21/**`；
  - **S 組（兩場共用）**：`results/test02/checks/built/**`、`results/test02/checks/bundle_build.json`、`results/test02/checks/root_props/**`、`results/test02/checks/t0_tutorial_diff.txt`、`results/test02/checks/trailing_pass/**`（若有）；
  - **R 組（競賽二）**：其餘（`results/test02/rank/**`、`results/test02/unpack_rank.json`、`results/test02/checks/rank/**`、`results/test02/failed/**`、`results/test04|07|08|09|14|19|25/**`）。
  - 11/04 之前就 commit 的檔（`results/test02/null/**`、`fallback_R10.json`、`fallback_selfcheck_R20.json`、各 `design/**`、`results/test10/` 的演練紀錄等）**不搬**：它們不是 v1 的測試輸出，v2 照樣要用。
- **`data/`（不進 git）**：`data/test`、`data/test06`、`data/test07`、`data/test08`、`data/test_rootprops`、`data/test_pass`（若有）、`data/colab_bundle/C10_test_v1.zip`、`data/colab_upload/test02`、`data/colab_upload/test04` 搬到 `data/archive_testset_v1/`（同一顆磁碟上改名，不複製、不刪）；搬前逐檔 SHA-256 寫進 `results/archive_testset_v1/data_manifest.json`（commit）。`data/test10`（10/13 的 B′ 演練）不是 v1 輸出，不搬。依分支只搬該分支會重建的資料夾（第 2 節）。
- **上傳紀錄**：`results/official_scores.json` 原檔先逐位元組複製到 `results/archive_testset_v1/official_scores.json`。重跑那一場的 v1 紀錄從 `uploads` 移到同一檔的新鍵 `voided_uploads`，每筆原樣保留，只多一個 `void` 欄位（`by: organizer`、公告名稱、時間、本登記）。**不刪任何一筆**。新的上傳編號接在最大的舊編號後面（#11 起）：`tools/official_log.py` 只加一行（`voided_uploads` 也算進編號）。
- 搬完的那個 commit 之後，凍結測試的允許失敗只增加「因為 v1 輸出被搬走」的那幾項（rev 2 列出確切清單，條件是 `git diff --name-status testset-v1-final` 在 tools／docs／tests／colab／prereg 底下是空的、`results/test*` 底下只有搬走或新的 v2 輸出）。

## 2. 分支（11/11 由位元組決定，不由分數決定）

- **競賽一**：一定重跑（公告寫明）。v1 競賽一的紀錄 #2、#4、#5 一律移到 `voided_uploads`。例外：v2 競賽一的 4 個檔和 v1 原檔逐位元組相同 → 停下，寫 rev 3 再說（極不可能）。
- **競賽二**：先比位元組。`tools/testset_v2.py probe` 比 AIdea 新下載的兩個 rank 檔和 v1 原檔（`data/archive_testset_v1/test_rootprops/original/`，也就是官網 MD5 相同的那兩個檔）。
  - **K（不動）**：兩個 rank 檔逐位元組相同，而且競賽二頁面沒有要求重交（沒有公告、排行榜沒清）。競賽二**什麼都不重跑**，紀錄不動，**25（#10）照舊是最後一次上傳**；R 組不搬。
  - **R0（只重傳）**：兩個 rank 檔逐位元組相同，但頁面要求重交或排行榜被清。不重跑；v1 的 rank 紀錄留在 `uploads`（它們是 25 的依據，PREREG_25 4.3 要綁 #9），每筆加 `void` 欄位；照 PREREG_25 4.3 的最後一版（25）對 `results/test25/rank/submission_rank.csv` 原檔重新 `prepare` 一次、上傳一次（TESTDAY「之後的上傳」(c)）。
  - **RR（全部重跑）**：任一 rank 檔不同。R 組搬走、rank 紀錄移到 `voided_uploads`，競賽二整條鏈照登記重跑（第 4 節）。
- **K／R0 的共用步驟**：test02 的 ingest／T1／root_props／build 是兩場一起做的，而競賽二的 25 綁了 `results/test02/checks/rank/t1.json`（v1）。為了不動它：ingest 時競賽二那兩個檔用 v1 的正規化副本（`data/archive_testset_v1/test_rootprops/normalised/rank_prediction_test_{public,private}.csv`，先由 `probe` 確認新下載的原檔和 v1 原檔逐位元組相同，再複製到一個只有這兩個檔的乾淨資料夾當 `--src`）；這樣 `data/test/raw` 的 rank 檔和 v1 的 `t1.json` 記的 SHA 相同，**不重跑競賽二的 T1／T2–T5**，`root_props` 只會套用到競賽一。這是 PREREG_02「`--src` 只給 AIdea 下載」的唯一例外，`SOURCE.json` 和 RESULTS 會記兩個來源。若演練或當天顯示某個工具仍要重跑競賽二的 T1，就停下：先請使用者決定競賽二的最後一版（25，`official_log.py final --comp rank --id 10`），再重跑。
- K／R0 只搬 P 組＋S 組；RR 三組都搬。`data/` 同理（K／R0 不搬 `data/test07`、`data/test08`）。

## 3. v2 的輸出寫在哪

- **和 v1 相同的正式路徑**（`results/test02/…`、`results/test06/…`、`data/test/…` 等）。各工具的 real mode 把路徑寫死；v1 搬走後它們就是乾淨的，工具照 v1 的方式執行，不加任何旗標、不用 `--rerun`。
- Colab：Drive `C10/results/test02`（RR）或 `C10/results/test02/player`（K／R0），以及 RR 時的 `C10/results/test04`，先由使用者改名成 `…_v1`；`C10/C10_test_v1.zip` 先移進 `C10/old_v1/`。新的打包檔**仍叫 `C10_test_v1.zip`**（notebook 寫死這個名字；「v1」是打包格式版本，不是測試集版本）。
- 執行階段一律選版本 **2026.07**，連線後在終端機確認 torch cu128（11/04 的規則）。

## 4. 上傳順序和條件（照原登記；每場每天 5 次）

- **競賽一**：Q04 → 06-OFF →（PREREG_06 5.3 的條件全部成立才做）06-ON → 用 **v2 的** public(06-ON) − public(06-OFF) 判否決（d < 0 → 再傳一次 06-OFF）→（06-ON 沒產生、被否決、傳錯、核對不出或 K7' 已過時）10（PREREG_10 4.4，A＋B′，B′ 照 PREREG_13）→（21 有可改的題時，一律最後）21。06-ON 的條件照原文，包括「兩池都是 400 題、400 位候選」：v2 不是 400×400 時 06-ON 不做，這是登記的後果，不是看分數的選擇。
- **競賽二 RR**：R20 → 04 → 07 → 08 → 09 → 19 → 25（→ 14 有可改的題時一律最後）。04 的 Colab 要在 v2 的 R20 上傳並記錄之後才按「全部執行」（t04 綁「R20 第一次上傳」）。
- **競賽二 K**：不上傳。**R0**：只傳 25 一次。
- 最後一版：照 PREREG_06 5.4／PREREG_10 4.4／PREREG_21（競賽一）、PREREG_25 4.3／PREREG_14（競賽二）。這些規則只看 `uploads` 裡的紀錄；v1 被移到 `voided_uploads` 的紀錄不參與。
- 每次上傳照 11/04 的常設同意（檢查全過後 Claude 直接上傳、傳完回報），`--ok-at` 的 note 寫「PREREG_27 v2」。

## 5. 截止點（從 11/11 算；都在 11/24 23:59 之前）

工具裡寫死的截止日期**不改**（改要動凍結檔），它們都落在 11/11 之後，所以 v2 直接沿用；另外加本登記自己的目標點。

| 名稱 | 時間（台灣） | 距 11/11 | 來源 |
|---|---|---|---|
| K27-A：v1 封存＋ingest＋build | 11/11 15:00 目標 | D+0 | 本登記 |
| K27-B：兩場第一次上傳（Q04；RR 時 R20） | 11/11 23:59 目標 | D+0 | 本登記 |
| K6B-10（B′ 兩池 P1 commit） | 11/14 23:59:59 | D+3 | PREREG_10／13（程式） |
| K6（04 的報告 commit） | 11/16 23:59:59 | D+5 | PREREG_04（程式） |
| K6-25 | 11/18 23:59:59 | D+7 | PREREG_25（程式） |
| K6-07／08／09／19 | 11/19 23:59:59 | D+8 | 各登記（程式） |
| K6'（06 正式執行 commit）、K6-10 | 11/21 23:59:59 | D+10 | PREREG_06／10（程式） |
| K7'（06-ON 上傳＋否決） | 11/22 23:59:59 | D+11 | PREREG_06（程式） |
| K6-21、K6-14 | 11/23 18:00 | D+12 | PREREG_21／14（程式） |
| 最後一次上傳目標／截止 | 11/23 23:59／11/24 23:59 | D+12／D+13 | TESTDAY |

- 沒趕上任何一個 K，照該登記「沒趕上就照表做」（退回保底），不延期。
- 11/04 11:00 這個時間點在程式裡出現的地方（例如「`fallback_R10.json` 要在 11/04 11:00 前 commit」「B′ 演練要在測試集開放前 commit」）指的是**模型和備案在看到任何測試資料之前就凍結**；v1 已經是測試資料，所以這些條件對 v2 只會更嚴，不放寬、不改。

## 6. 只記錄、不進判定

- `probe` 寫 `results/testset_v2/probe.json`：每個檔的列數、id 格式、每題盤數分布、候選每人盤數、和 v1 相同 id 的題目有幾題逐列相同、v2 新題目和 v1 候選／題目有沒有同一盤棋。
- v2 的預測在 v1 相同 id 的題目上和 v1 有幾題不同（兩場各自），寫進 RESULTS。差很多只代表要查 bug（id 對應、列順序），不改方法。

## 7. 11/11 的格式和 v1 不同時（不看分數決定）

- 判定照原登記：T1 判 FAIL_STRUCT（例如每池不是 400 題、候選不是 400 人×100 盤、id 前綴不同、檔名不同或多了檔）→ PREREG_02 第 9 節第 6 條：**停**，先寫修訂，修訂前不做任何預測。
- 那份修訂（PREREG_27 rev 3 起）**只准改結構常數**，讓工具認得官方的新格式：題數、候選人數、每人盤數（只在官方明寫時）、id 前綴、檔名、檔案個數、列順序，以及這些常數在 `t02_test_bundle.py`、`t02_checks.py`（T1.2）、`check_submission.py`、`official_log.py`、兩份 test02／test04 notebook 的 preflight 和下游工具裡的對應處。每個被改的凍結檔，`git diff testset-v1-final` 只能出現那幾行常數；因此失敗的凍結測試在修訂裡逐項列為允許失敗。
- **不准改**：模型檔、特徵、自我檢索與閘門（PREREG_02、06）、CSLS、06-ON 的條件（包括「400 題、400 位候選」的字面）、否決規則、A／B′ 的條件、21／14 的規則、競賽二的融合和規則 B、接縫標準、任何門檻、上傳順序。
- 每題盤數或每位候選盤數超出模型能處理的範圍（例如題目超過 20 盤、候選不是 100 盤而自我檢索要 100 盤）：照各登記的原文處理（PREREG_02 第 9 節：候選不是每人 100 盤 → 停、修訂前不交競賽一）。這種情況由 rev 3 寫明怎麼退回保底，不另造方法。
- `probe` 會列出新格式和每一個寫死常數的差異（`needed_changes`），rev 3 照那份清單寫。

## 8. 揭露

- 報告兩場都寫：官方 11/06 更新測試集、排行榜清空；v1 的上傳紀錄保留在 `voided_uploads`（不刪）、輸出在 tag `testset-v1-final` 和 `results/archive_testset_v1/`；v2 照原登記重跑，沒有改方法。
- v1 的 public 分數（例如競賽一 06-ON 1.0）只寫成「v1、已被官方作廢」，不和 v2 混著比。

## 9. 防誤改

- `tests/test_testset_v2.py`：工具的單元測試（合成檔、暫存 git repo），以及本檔在 `tools/testset_v2.py` 之前 commit。
- 閘門照 AGENTS.md；搬檔之後的允許失敗以 rev 2 的清單為準。

## 10. rev 2：實作差異、演練、允許失敗（2026-11-06，合併前）

### 10.1 實作（`tools/testset_v2.py`，`tests/test_testset_v2.py`）

- `plan`／`archive --branch K|R0|RR`：要搬的檔＝`git diff --diff-filter=A f21ff1e testset-v1-final -- results/test[0-9]*`，P／S／R 組照第 1 節的前綴分；11/04 之後有任何 `results/test*` 的檔是「被修改」而不是「新增」就停（沒有發生）。搬之前：tag 是 HEAD 的祖先；每個檔在 HEAD 和 tag 的 blob 相同；**要搬的檔和上傳紀錄**沒有未 commit 的修改、那些資料夾沒有未追蹤的檔（只檢查這些，不看整個 `results/`，因為 worktree 常有只差換行的舊紀錄）。搬完把變空的資料夾刪掉（只在 `results/` 底下）。紀錄只有一份 `ARCHIVE.json`（和第 1 節寫的 `data_manifest.json` 不同：資料清單併在裡面，每一階段一筆 `stages`：分支、tag commit、每個檔的 blob、上傳紀錄動了哪幾筆、`data_rows` 逐檔 SHA-256），加上 `official_scores.json` 原檔副本（只在第一階段寫）。順序（審查第 3 點）：所有檢查 → 寫紀錄（`in_progress`）→ `data/` 改名 → `git mv` → 上傳紀錄；改名或 `git mv` 失敗時，這一次搬過的全部搬回、紀錄標 `failed`，之後要人工清掉那一筆才能再跑。成功後只 stage 封存相關的檔（搬移、`results/official_scores.json`、兩個紀錄檔），並檢查 index 裡沒有別的東西（有就印 WARNING、結束碼 1）；**不 commit**。
- **可以往後加寬**（審查第 7 點）：K 之後官方才清競賽二 → 再跑 `archive --branch R0` 或 `--branch RR`，只做前一階段沒做的（R 組、rank 紀錄、競賽二的資料）；已標 `void` 的紀錄搬進 `voided_uploads` 時保留第一次的 `void`。不能往回（RR → K）。
- **Colab 結果 zip**（審查第 4 點）：`results/test02/test02_player_results.zip`（每個分支）、`results/test02/test02_rank_results.zip`、`results/test04/test04_rank_results.zip`（RR）不在 git 裡，跟 `data/` 一起搬到 `data/archive_testset_v1/results/…`，逐檔 SHA-256 記在 `data_rows`。
- P／S／R 三組的前綴都列明；v1 的輸出不屬於任何一組就停（審查第 8 點）。
- 上傳紀錄：`void_log`。`move`：整筆原樣搬到 `voided_uploads`，只多 `void`；那一場已有 `final_choice` 時搬到 `voided_final_choice`、原處設成 null。`mark`（R0）：留在 `uploads`、只加 `void`。`tools/official_log.py` 只加一行：新編號＝max(`uploads` 的編號＋1，`voided_uploads` 的編號＋1)。
- `probe --src …`：資料夾（含裡面的 zip；略過 `__MACOSX`、`._*`）、zip 或 csv；只少了檔（例如只下載了競賽一）列在 `missing_files`，競賽二的分支寫「not decidable」，**不會**誤判成 RR（審查第 1 點）；重複的 id 列進 `needed_changes`；每個檔的列數、id 樣式、每題盤數、候選每人盤數、開頭屬性（HD 等）、和 v1 原檔逐列比對（相同 id 的列相同幾題、只差開頭屬性幾題、v1 的 id 是否在最前面）；`needed_changes`＝和寫死常數不同的地方；結束碼 0＝結構和 v1 相同，4＝要第 7 節的結構修訂。寫 `results/testset_v2/probe.json`（11/11 commit）。
- `rank-src --src <下載> --dest <新資料夾>`（K／R0）：兩個 rank 檔和 v1 原檔逐位元組相同才做；`--dest` 放 v1 的正規化 rank 檔＋下載的 4 個競賽一檔（原樣），當 `ingest` 唯一的 `--src`。不同就拒絕並寫「branch RR」；`--src` 裡有 6 個名稱以外的檔也拒絕；兩個來源寫進 `results/testset_v2/rank_src.json`（審查第 5 點，第 2 節的「`SOURCE.json` 和 RESULTS 記兩個來源」由這個檔做到：`SOURCE.json` 只會記 `--dest`）。
- `compare --v1 FILE --v2 FILE`（第 6 節，只記錄）：相同題號的列有幾題完全相同、第一欄（等級或 top_1）不同幾題，分 public／private。用 v1 的 19 對 25 試跑：800 題共同、第一欄不同 89 題（public 42，和 25 報告記的規則 B 差異相同）。
- `fake --root S --scenario same|grow|grow-player`（只供演練）：v1 原檔＋fit 玩家的棋（`c10data.load_rows(part="fit", local_dir=None)`，不開 `data/local`），新 id 接在 v1 後面，開頭加 `HD[0]`、CRLF、和官方相同的欄位排法。`same`＝每池換掉最後 50 位候選和 50 題（數量不變），`grow`／`grow-player`＝每池多 100 位候選和 100 題（`grow` 兩個 rank 檔也各多 100 題）。

### 10.2 演練（scratch 樹和 repo 外的複本；數字都是實際執行）

- `probe`：`same` 結構和 v1 相同、rank 逐位元組相同（結束碼 0）；`grow` 列出 6 項（每池題數 500≠400、候選 500≠400，rank 兩檔 500≠400；結束碼 4，分支 RR）；`grow-player` 4 項（只有競賽一，分支 K／R0）。v1 的題目在三種假 v2 都逐列相同、排在最前面（`same` 是 350／400）。
- **K 的共用步驟**（`same`，工具照原樣，`--root` scratch 模式）：`rank-src` → `ingest` → 競賽一 `t1`（只能跑 T1.1–T1.3：T1.4／T1.5 要 `data/local` 的雜湊防護，演練不碰；為了往下走，在 scratch 把 `complete` 改成 true）PASS → `root_props prepare`：**競賽二 not_needed（5,000／5,000 盤已是純格式）、競賽一 apply（89,945 盤拿掉 HD）**，v1 的競賽二 `t1.json` 直接被接受 → `ingest --rerun`、競賽一 `t1 --rerun` → `verify`：只因 T1.4／T1.5 沒跑而 NOT PASSED（演練的限制），**沒有任何競賽二的問題** → `build`（rehearsal 種類）接受 v1 的競賽二 T1、沒重跑；`data/test` 的 rank 檔 SHA 和 v1 的 `data/test` 相同（`75baefaded49` 等）。用時：prepare 超過 2 分鐘、verify 1 分 40 秒、build 12 秒。
- **題數變了**（`grow-player`）：競賽一 `t1` → FAIL_STRUCT（結束碼 3），問題只有 T1.2 的 6 條（題數、列數、人數），T1.1／T1.3 沒有問題。這就是第 7 節的情形：停、寫 rev 3。
- **封存**（新 clone、`core.autocrlf=true`、`--skip-data`，各自 commit 後跑閘門＋`tests/test_testset_v2.py`）：
  - 封存前：只有 6 項已知的允許失敗。
  - K：搬 67 個檔（P＋S），紀錄 #2、#4、#5 移到 `voided_uploads`；閘門多 1 項：`test_t25_hygiene.py::Frozen::test_no_change_since_tag`（`p19-testday-pre25` 之後 `results/test0*` 的檔被搬走）。在紀錄副本上 `official_log.py final --comp rank --id 10`：結束碼 0，「PREREG_25 4.3：最終選擇是 25；符合條件的 25 紀錄 [10]」——K 之後 25 仍然符合。
  - R0：搬 67 個檔；rank 的 #1、#3、#6–#10 加 `void`、留在 `uploads`；在紀錄副本上對 `results/test25/rank/submission_rank.csv` 重新 `prepare`：結束碼 0，所有檢查通過，編號 **#11**。
  - RR：搬 239 個檔（P＋S＋R），10 筆都移到 `voided_uploads`；閘門：`test_p09_hygiene.py::FrozenPaths`、`test_t19_hygiene.py::Frozen` 反而通過（v1 的輸出不在了），`test_t25_hygiene.py::Frozen` 失敗。
- 沒演練到、11/11 才知道的：T1.4／T1.5、T2–T5、兩份 notebook、所有接縫（都要 `data/local` 或 GPU）；這些工具不改，v2 和 v1 用同一套。

### 10.3 允許失敗（封存之後）

- 原本的 6 項照舊（AGENTS.md 閘門一節）。p09／t19 的條件改成：`git diff --name-only <該 tag> -- <凍結路徑>` 只列出 `results/test0*`／`results/test1*` 底下的檔（v1 搬走的或 v2 新寫的），沒有 tools／docs／tests／colab／prereg 檔。
- **新增 1 項**：`tests/test_t25_hygiene.py::Frozen::test_no_change_since_tag`，條件相同：`git diff --name-only p19-testday-pre25 -- <它的 FROZEN_PATHS>` 只列出 `results/test0*`、`results/test1*`、`results/test2[0-4]` 底下的檔。
- `tests/test_g11_hygiene.py::InsertOnly::test_official_log_untouched` 拿 `tools/official_log.py` 和 `main` 比：分支上會失敗，合併進 main 之後通過，不算允許失敗。
- 其他任何失敗都要停下查。

### 10.4 11/11 的操作頁

`docs/TESTDAY_27_V2.md`（Claude 的順序）；使用者的步驟在 `docs/TESTDAY_USER_CHECKLIST.md` 最後一節。

### 10.5 B′ 的 P1 什麼時候跑（審查第 6 點；排程，不是選擇）

- PREREG_10 4.4：06-ON 沒產生、被否決、傳錯、核對不出或 K7' 已過時傳 10；10 的 B′ 要兩池 P1 在 K6B-10（11/14 23:59:59）前 commit。v2 很可能不是 400×400（06-ON 照原文不做），所以 10 很可能是登記的路徑。
- 規則：競賽一的 `data/test` 一 build 好（測試檔定了），就照 `docs/TESTDAY_10.md` 跑兩池 B′ 的 P1，**4090 上排第一**（RR 的 07 HS、08 P1 排在後面，它們的期限是 11/19）；只有在 P1 開跑前 06-ON 已上傳、而且否決判 `kept` 時才不跑。06-ON 判定之前 P1 已經跑完也照樣 commit（沒用到就只是紀錄）。
- 每次 4090 步驟前 `nvidia-smi`：沒有別的工作、驅動程式 616.92。

### 10.6 獨立審查（2026-11-06，另一個子代理，唯讀）

沒有 blocker；6 個 major、6 個 minor。major 1–5 和 minor 7、8、10、11、12 已在本節和 `tools/testset_v2.py` 修好並加測試；major 6 由 10.5 處理；minor 9（R0 的重新 `prepare` 要在 build 之後，因為 `data/test` 每個分支都會搬走重建）寫進 `docs/TESTDAY_27_V2.md`。審查也確認：P 組 57、S 組 10、共 67 個檔，和演練相同；`official_log` 的規則（Q04 第一次上傳、R20 第一次上傳、06-ON 一季一次、否決、t10、t14／t21 的 `final_choice`）都只讀 `uploads`；K 的共用步驟在 real mode 沒有額外的時間順序或 `SOURCE.json` 檢查。審查提醒的「06-ON 的 400×400 字面」照第 7 節保留（不看格式先決定，是登記的後果）。

## 11. 和 PREREG_28 的關係（2026-11-06 加行，v2 公布前）

- `docs/PREREG_28_06ON_N_BY_N.md`（v2 公布前登記、fit 玩家模擬通過）只修訂 06-ON 的尺寸條件：第 4 節「v2 不是 400×400 時 06-ON 不做」和第 7 節「不准改 06-ON 的條件（包括 400 題、400 位候選的字面）」，對**尺寸條件**改成「每池 N 題＝N 位候選（06 和 test02 數的都是）、400 ≤ N ≤ 800」；N 在範圍外或題數 ≠ 候選數時照舊不做。第 7 節其他禁止照舊，rev 3 仍然不准碰 06-ON 的任何條件。測試日用 `tools/t28_n.py submit`（`docs/TESTDAY_27_V2.md` 第 6 節加的那一行）。`tools/testset_v2.py` 的 probe 說明文字（「PREREG_27 7 keeps it」）不改，以本節為準。

## 12. rev 3（2026-11-11 加行，v2 公布後、任何 v2 上傳前）：實際的 v2 結構和處理方式

- **實際結構**（`results/testset_v2/probe.json`，`1a40129`）：原測試集四個檔和 v1 逐位元組相同；另外新增 `player_identification_test2_{public,private}.csv`（各 400 題，1／2／3／5 盤各 100，黑白都有）和 `player_identification_candidates2_{public,private}.csv`（各 600 位候選 × 10 盤，黑白都有，非一對一）。交件 1,600 列（`pub_`／`priv_`／`pub2_`／`priv2_`）；Public／Private 分數＝原題和新題平均。競賽二 rank 兩檔逐位元組相同、頁面無公告、排行榜未清 → **K**（25 照舊，不重跑、不重傳）。
- **這個情況本登記沒預想到**（原設想是整包換新、全鏈重跑）。處理方式：
  1. 原題 800 列：v1 的測試日輸出直接沿用（輸入檔逐位元組相同，重跑只會得到相同結果）。最後一版照 v1 的規則是 06-ON（`results/test06/player/on/submission_player.csv`，sha256 94cc82650dd4…）。**不跑 `testset_v2.py archive`**，v1 的輸出留在原位當輸入；tag `testset-v1-final` 照舊保存。
  2. 新題 800 列：06／10／21／28 的登記都以「每池 400 題、候選 100 盤黑棋」為前提，不適用；06-ON 的一對一開關在新題不開（PREREG_28 的尺寸條件是題數＝候選數，新題 400≠600）。新題的方法另寫登記；第一版是 `results/prereg35/explore/`（PLAN `44a2d04` 先於分數，評估池判一次 `a45a4b0`）。
  3. 上傳紀錄：`tools/official_log.py` 和 `tools/check_submission.py --official` 綁 800 列，v2 的 1,600 列檔不走它們；格式檢查用 `results/prereg35/explore/p35_apply.py assemble`（1,600 列、四種前綴各 400、無重複、每列 5 個不重複且前綴相同、ID 都在對應候選檔），上傳紀錄手寫在 `results/official_scores_v2.json`（每筆：時間、檔案 sha256、方法、public 分數、名次、當日第幾次）。v1 的 `results/official_scores.json` 不動。
  4. 上傳次數每天 5 次照舊；每次上傳前先讀 AIdea「今日已上傳 N 次」。public 分數只用來抓格式錯誤和大幅退步，不拿來挑版本；新題的方法選擇一律在 fit 玩家的模擬池判。
- 本節只加行，前面各節原文不動；和本節不一致的地方（封存、全鏈重跑、06-ON 在 v2 的條件）以本節為準。
