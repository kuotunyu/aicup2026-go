# test06 建置紀錄（分支 `p06-testday`；PREREG_06 第 5 節、spec 第 7–9 節）

- 範圍：競賽一 06 的測試日擴充。`tools/t06_player.py`、`tools/t06_fake_tests.py`、`tools/official_log.py`（只加行）、`tests/test_t06_player.py`、`test_t06_official_log.py`、`test_t06_hygiene.py`、`test_t06_dryrun.py`、`test_p06_player_hygiene.py`、`tests/conftest.py`、`docs/TESTDAY_06.md`、PREREG_06 rev 3（修訂紀錄一列＋第 11 節）。
- 工作地點：worktree `D:\AI-Portfolio\competitions\c10-p06-testday`（分支 `p06-testday`，從 main `952180c` 切出），`data` 是指向 main 的 `data` 的 junction。所有指令在 worktree 根目錄、用 main 的 `.venv`（Python 3.12.13、numpy 2.5.3、scipy 1.18.1）、`PYTHONDONTWRITEBYTECODE=1` 跑；演練樹都在 session 暫存區 `…\scratchpad\p06td\`。只用 CPU，沒有跑 `tools/score.py`，沒有打開任何標籤檔。
- 本檔的數字都出自 2026-10-03 實際執行的輸出（演練樹裡的 JSON 報告、pytest 的結果）。第 2.1–2.6 節是第一輪建置（HEAD `777e38c`）的紀錄；第 2.7 節是三個角度的審查之後、修正版（HEAD `d7b5b2a`）重跑的紀錄；第 5、6 節是審查和逐條回應。

## 1. commit

| commit | 內容 |
|---|---|
| `3b0414a` | 寫任何 t06 程式之前，先記 Q04 假測試基準 `results/test06/design/q04_fake_golden.json`（spec 9.3 第 1 項）和 `tests/test_t06_dryrun.py` 的 `Q04Golden`；`.gitattributes` 加 `results/test06/** -text`（當時另有一行註解，`e50145f` 刪掉，對 main 只加一行） |
| `7929cb8` | `t06_player.py`、`t06_fake_tests.py`、`official_log.py` 的 06 行、`test_t06_player.py`、`test_t06_official_log.py` |
| `f987001` | `tests/test_p06_player_hygiene.py`（spec 9.2）、`tests/test_t06_hygiene.py` |
| `777e38c` | `tests/test_t06_dryrun.py` 的 spec 9.3 演練（`FromValDrills`、`OfficialShapeDrills`）；`apply` 沒有一對一指派時不寫強制探測檔 |
| `e50145f` | 審查修正（第 6 節）：否決的配對、同名檔警告的三種核對結果、暫停時不指定、06-ON 不再是最後一次上傳時指定 06-OFF、06-ON 的 `prepare` 先排除 Q04／06-OFF 同名檔警告、同 SHA 重傳的警告、登記的 3 先於「輸出已存在」、K6' 後已 commit 的 06 是 2、seam 綁 test02 的 seam_val dumps、另一池的舊報告不算、已記錄後拒絕 `submit --rerun`、06-OFF 不綁 T2；對應的單元測試和演練 |
| `d7b5b2a` | `tests/conftest.py`（單一行程跑全套測試）；`test_p06_player_hygiene.py` 的「H6P 之後只准 PREREG_07 的列」只在 06 的閘門（`C10_T04_GATE=1` 或 `C10_K5=1`）跑 |
| （之後） | 本檔、`docs/TESTDAY_06.md`、PREREG_06 rev 3（由主 session 寫入並 commit） |

## 2. spec 9.3 的演練

第 2.1–2.6 節是第一輪（HEAD `777e38c`）的紀錄，原樣保留；修正後的重跑在第 2.7 節。

### 2.1 第 1 項：Q04 逐位元組不變

- 基準在 `3b0414a` 記下（那時沒有 `tools/t06_player.py`；`tests/test_t06_hygiene.py` 用 git 核對這個先後）。基準是 test02 競賽一在 from-val 假樹上的 dry run：`submission_player.csv`、三池 `top5.csv` 的 SHA-256，八份報告去掉時間類欄位後的 SHA-256，以及三池的選擇、閘門、防護層級。記錄時在同一路徑跑了兩次，檔案和正規化後的報告都相同。
- 寫完 t06 之後重跑（`C10_SLOW=1 … pytest tests/test_t06_dryrun.py`，`Q04Golden`，HEAD 在 `7929cb8` 之後）：`test_equal_to_golden`、`test_golden_rule_unchanged` 都過。`FromValDrills.test_q04_path_unchanged_on_this_tree` 在另一棵樹上再核對一次，也相同。
- 合併當下要再跑一次（K5' 核對表）。

### 2.2 第 2 項：R20 和 04 不變

- 防誤改閘門（`C10_T04_GATE=1 … tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py tests/test_p06_hygiene.py`）：開工前、`7929cb8` 之前、之後都是 **1 failed, 50 passed, 35 subtests passed**，失敗的固定是 `tests/test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`（打分紀錄帳在 6ec5e68 之後多了 H6P 的 4 列，PREREG_04 rev 3 第 19 條允許）。
- `official_log.py` 只加行：和 tag `p04-testday-pre06`、`r20-testday-pre04` 比都是子序列（`test_p06_hygiene.InsertOnlySinceTag`、`test_p04_hygiene`、`test_t06_hygiene` 都核對）；`git diff` 沒有任何刪除行。
- `tests/test_t06_official_log.py::TagVersusNew`：tag `p04-testday-pre06` 的 `official_log.py`（`git show` 出來、在記憶體裡執行）和新版跑同一套流程、同一組 mock：競賽二 R20、R10、04 三種流程，各接競賽一 Q04 的 prepare／uploaded／public／final／status 和拒絕；有和沒有 `results/test06` 樹各一次。log 位元組、回傳值、stdout、拒絕訊息全部相同（3 個測試、128 個 subtest）。新版在沒有 06 紀錄時從不呼叫 `t06_now`（mock 成一叫就失敗）。

### 2.3 第 3 項：06 對得回 H6P（from-val 樹）

from-val 樹的 public、private 都是本機驗證池換了代號（`val_` → `pub_`／`priv_`）。`t06_player.py` 跑完整套（`--dry-run`）：

| 項目 | 結果 |
|---|---|
| `feats` 三池 | 各 44,935 盤（候選 40,000、題目 4,935），不交替 0 盤，詞彙外 token 0；每池 36–40 秒 |
| `seam` | `passed`；dumps 逐陣列相同；`top5_off.csv`、`top5_on.csv` 和 `player_val_top5.csv`、`player_val_top5_switch.csv` 逐位元組相同（400/400）；sr06 V2 0.9758877068932452、V3 0.9697336206689693、V0 0.9221355664136796、R02 0.8998720491985943（取 6 位和 H6P 記的相同，也等於 `p06_player.sr06_apply`）；送分閘門 +0.05375、測試日閘門 +0.07602；r_forced_cost 1.1457767848959581（和 H6P 完全相同），偵測器開；Q04 用 p02 的函式和 06 的路徑重算都逐位元組相同；和 dumps 無關的盤數、詞彙外 token、LDA SHA 都相同；24.0 秒 |
| `apply public` | 選 V2（sr06 V2 0.975462、V3 0.970126、V0 0.919266、R02 0.898035；sr06 的分份依代號的雜湊，所以數字和 seam 不同），06 閘門 +0.07743 通過；r_forced_cost 1.0224 ≤ r* 2.2516，偵測器開；Hungarian 改了 17 題的第 1 名；和 test02 第 1 名不同 51 題；13.6 秒 |
| `apply private` | 選 V2（V2 0.976223、V3 0.968977、V0 0.923703、R02 0.905485），06 閘門 +0.07074；r_forced_cost 1.3778，偵測器開；改 17 題；和 test02 不同 51 題；16.7 秒 |
| 換回代號 | 兩池的 `top5_off.csv` 和 `player_val_top5.csv`、`top5_on.csv` 和 `player_val_top5_switch.csv` 逐列相同（401 列含標頭） |
| `submit` | 06-OFF 800 列，SHA-256 `62c88e217cb1…`，`check_submission --official` 通過；這棵樹沒有 T2（驗證玩家的棋不跑內容檢查），所以 06-ON 的 T2 條件不成立、不產生 06-ON（其他六條都成立） |

### 2.4 第 4 項：官方形狀（from-train 樹）

`t06_fake_tests.py official`：用 `t02_fake_tests.py` 的 from-train 官方形狀產生器（`from_train(shape="official")`，fit／inner 的列由 `c10data.load_rows(local_dir=None)` 讀，不開 `data/local`），每池 400 位 fit 玩家（黑棋 ≥120 盤）、各 100 盤黑棋候選、5／10／15／20 盤的題目各 100 題；`add-dumps` 用第 4 節的真實 CNN embedding（`data/prereg06/cnn_fit/black_ge105.npz`，prereg01 的玩家 CNN）：90,000 盤全部找到，0 盤用固定 seed 的替身。T1（結構）PASS；test02 兩池都選 Q04、防護第 1 層。

| 樹 | 結果 |
|---|---|
| 一題對一人 | seam 通過（逐位元組）。public：選 V2（sr06 V2 0.9912、V3 0.98851、V0 0.952826、R02 0.932627），06 閘門 +0.058573；r_forced_cost 1.6084 ≤ r* 2.2516，偵測器開，Hungarian 改 6 題第 1 名。private：選 V2（V2 0.988542、V3 0.985733、V0 0.94941、R02 0.935271），閘門 +0.053271；r_forced_cost 0.8836，開，改 2 題。兩池全黑（候選和題目的白棋都是 0 盤）、不交替 0 盤。替身 T2 的 (b)(c)(d) 實算都是 0。**06-ON 產生**，七條條件都成立；06-OFF `c3cabb1f1c4c…`、06-ON `bb9dd7588b15…`，`check_submission --official` 都過；和 Q04 第 1 名不同：06-OFF public 22／private 27，06-ON 26／29；06-OFF 對 06-ON 6／2。手動跑一次和 pytest 的那一次，兩份交件檔的 SHA 相同 |
| `--dup 80`（每池 80 人出兩題、80 人沒題；400 題、320 位不同的出題者） | 題目沒有共用的棋，替身 T2(c) 實算 0。public r_forced_cost 101.01、private 93.23（r* 2.2516），兩池偵測器都關；**沒有 06-ON**（不成立的只有 `private_switch_usable_and_detector_open`）；06-OFF 照常產生（V2，閘門 +0.058731／+0.053429） |
| T2(b) 不過（`fake-t2 --fail b`，同一棵一題對一人的樹） | `t2_c_zero_and_abd_not_failed` 不成立（`abd_failed` = b:public、b:private），**沒有 06-ON** |

### 2.5 第 5 項：負面演練

| 演練 | 怎麼做 | 結果 |
|---|---|---|
| T1 FAIL_OPENING | checks 副本的 `t1.json` 改成 FAIL_OPENING | 結束碼 3，`not_used.json` 記 `t1_fail_opening` |
| 某池防護第 3 層 | test02 輸出副本，private 的 `guard_level` 改 3 | private 的 `top5_off.csv` 和 test02 的 `top5.csv` 逐位元組相同、沒有 `top5_on`；交件檔 private 的 400 列等於 Q04 的；沒有 06-ON |
| 兩池都第 3 層 | 兩池都改 3 | 第二池的 `apply` 結束碼 3（兩池都退回 test02） |
| 防護第 2 層 | public 改 2 | public 照常用 06、`switch_allowed_at_this_level` 假；`submit` 的 `both_pools_06_at_guard_level_1` 不成立，沒有 06-ON |
| 防護第 0 層、test02 閘門沒過 | public 改 0；或 `gate_passed` 假＋第 0 層 | 那一池交 test02 的列（逐位元組相同） |
| T2(b) 不過 | 官方形狀樹，`fake-t2 --fail b` | 沒有 06-ON（見 2.4） |
| 選中的組合不能用 | 在同一個行程裡把凍結設定的 `switch.V2.usable` 改成假（只有記憶體） | private 選 V2、`usable` 假、不寫 `top5_on`；public 照樣寫強制探測檔；沒有 06-ON |
| H6P 次要指令 < 0 | `results/prereg06/player` 的暫存副本，`score_H6P_switch.json` 的點估計改 −0.0001（`--prereg06-dir`） | `h6p_secondary_not_negative` 不成立，沒有 06-ON |
| 06 閘門沒過 | 同一個行程把 `GATE06_MIN` 設成 1.0 | public 交 test02 的列（逐位元組相同）、沒有 `top5_on` |
| K6' 已過 | `--now 2026-11-22T00:00:00+08:00` | 結束碼 3 |
| H6P 沒採用 | 副本的 `score_H6P.json` 下界改 −0.001 | 結束碼 3 |
| test02 沒 commit | 正式模式，路徑常數指到暫存樹，git mock（`test_t06_player.RealMode`） | 結束碼 2；同一組另驗：`.venv` 版本不同 2、git 不乾淨 2、T1 不是完整的 raw 2、FAIL_STRUCT 2、FAIL_OPENING 3、K6' 3、H6P 標記不在 3 |
| Q04 還沒上傳 | log 裡 Q04 只有 `prepared` | 結束碼 2；有 `uploaded` 時同一步照常 |
| 06-OFF 還沒上傳就 prepare 06-ON | `test_t06_official_log.T06Prepare.test_order_rules` | 拒絕；同一個 06-ON 檔第二次 prepare 拒絕；06-ON 已上傳後再 prepare 拒絕（一季一次）；不同 `run_id` 拒絕；K7' 之後拒絕 |
| 同名檔警告 | 06-ON 和 06-OFF、06-ON 和 Q04、06-OFF 和 Q04 的 public 相同而第 1 名不同題數 > 0 | `veto` 結束碼 2、`paused: true`；`official_log public` 印 WARNING；`--confirmed-upload-check` 之後判定；第 1 名不同題數是 0 時不警告 |
| d < 0 時最終設 06-ON | `T06Final.test_d_negative_vetoes` | 拒絕；06-OFF 通過；`status` 要求再傳一次 06-OFF，再傳後不再要求 |
| d ≥ 0（含 0）時最終設 06-OFF | `T06Final.test_d_zero_and_positive_keep_on` | 拒絕；06-ON 通過 |
| K7' 已過、06-ON 已上傳、否決沒判定 | `T06Final.test_undecided_before_k7_and_off_after`、`test_t06_player.Veto.test_k7_passed`、`test_status_k7_line` | K7' 前三種 final 都拒絕（還沒定）；K7' 之後只准 06-OFF；兩個 `status` 都要求再傳 06-OFF；`veto` 寫最終 06-OFF |
| 帶合格修訂的翻盤 | `T06Final.test_revision_override` | 只寫文字、rev 3、沒有那一列、修訂早於紀錄：都拒絕；rev 4 在紀錄之後加入：通過，記進 `prereg06_5_4.revision` |

### 2.6 第 6 項：完整測試、`C10_SLOW`、閘門、審查

- **06 的演練**（`C10_SLOW=1 … pytest tests/test_t06_dryrun.py`，樹在暫存區 `dry`、`dry2`）：第一次 21 項 18 過、3 項失敗，失敗的三項是演練本身寫錯（用 `feats` 演練，但共用的特徵資料夾已有輸出，工具正確地以「輸出已存在」結束碼 2 拒絕，沒走到要演練的條件）；改成 `apply`、新的輸出資料夾後，`FromValDrills` 重跑 16 項全過（7 分 58 秒）。`Q04Golden` 2 項、`OfficialShapeDrills` 3 項在第一次就過（全部 22 分）。
- **完整測試（每個測試檔各自一個 pytest 行程）**：912 passed、3 failed、88 skipped、2 deselected、576 subtests passed。
  - deselect 的 2 項：`tests/test_t02_infra.py::Build::test_mode_guards`、`FromVal::test_guards`（第 4 節：在 junction 的 worktree 裡會寫進 main 的 `data/`）。
  - 失敗的 3 項，所在的測試檔和它們測的程式都和 main 的 `952180c` 逐位元組相同（`git diff --stat 952180c` 沒有這些檔），不是這個分支造成的：`test_p03_nb.py::Built::test_rebuild_is_byte_identical`（重建的 prereg03 notebook 和已 commit 的 SHA 不同）、`test_p04_opp.py::TestGates::test_out_dir_rules`（`REPO/data/prereg04` 經 junction 後被當成 repo 外，`P04Error` 沒有丟出）、`test_p04_rank_fuse.py::FrozenTools::test_interface_constants_equal_p04_opp`（單獨跑會過，和同檔其他測試一起跑才失敗）。要在 main（沒有 junction）上確認。
  - 同一套測試放在**同一個** pytest 行程跑時是 12 failed、102 errors：`tests/test_p06_player.py`（和 `test_t06_player.py`）在收集階段 import `p06_player`，它的 import 時防護會擋整個行程讀 `data/local`；`tests/test_p06_style.py` import 探索的 `c1h_lib`，它也裝了自己的防護。之後的 `test_p02_player`、`test_t02_player`、`test_t04_rank`、`test_score` 等讀驗證棋譜檔或合成的標籤檔名時就被擋。這是測試隔離的問題，不是 06 的程式錯誤：裝防護的 `tests/test_p06_player.py`、`test_p06_style.py` 和被擋的測試檔都和 main 相同，所以在 main 上用單一行程跑全套應該也會一樣（推論，沒有在 main 上實跑）。`test_p02_player.py` 和 `test_p06_player.py` 一起跑就重現得出來（被擋在 `data/local/player_val_candidates.csv` 的 stat）。建議另外處理（例如把防護改成只在 CLI 的 `main()` 安裝）。**已修正（`tests/conftest.py`，第 2.7 節、第 6 節）。**
- **`C10_SLOW=2`**（每個檔各自一個行程，樹在暫存區 `dry3`）：
  - `tests/test_t02_dryrun.py -k FromValPipeline`：15 passed（5 分 40 秒）。
  - `tests/test_t04_dryrun.py`：32 passed、1 skipped、63 subtests（4 分 35 秒）。
  - `tests/test_t02_rank.py`：35 passed、1 failed。失敗的是 `Committed::test_real_selfcheck_and_r10_reproduce`：`t02_rank.py fallback-fit --combo R20` 拒絕，因為 worktree 裡的 `results/prereg02/rank/pseudo_inner.json` 是 CRLF（新 checkout，`core.autocrlf=true`，這個檔沒有 `-text`），原始 SHA `34bd41a2e674…`；折成 LF 後是 `3a51b3dd4edf…`，正好是凍結紀錄的值。是 checkout 的換行問題，不是程式；這個檔和測試都和 main 相同。`test_p03_nb` 的重建差異多半是同一個原因（沒有再追）。
  - 06 的演練見本節第一點。
  - **沒有在 worktree 跑的**：`test_t02_dryrun.py` 的 `FromTrainChecks`、`FromTrainOfficialShape`（`C10_SLOW=2` 才跑）、`test_p04_opp.py` 的 `TestRealSelftest`、`test_t04_rank.py` 的 `SlowO4SeamVal`。它們經由 `c10data` 的外洩檢查或 04 的 seam_val 讀 `data/local` 的棋譜檔，超出這次工作准許讀 `data/local` 的範圍（只有 06 的 seam_val 和 from-val 假樹）；它們測的 `t02_checks`、`t02_fake_tests`、`p04_opp`、`t04_rank` 這個分支都沒有改。合併前由 main 的 session 決定要不要在 main 上補跑。
- **防誤改閘門**（最後一次，HEAD `777e38c`）：`C10_T04_GATE=1 … tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py tests/test_p06_hygiene.py` → 1 failed, 50 passed, 35 subtests passed；失敗的只有 `tests/test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`（H6P 的 4 列）。加上 `tests/test_p06_player_hygiene.py tests/test_t06_hygiene.py`：1 failed, 67 passed, 44 subtests passed（同一項）。

### 2.7 審查修正之後重跑（HEAD `d7b5b2a`；10/3 21:4x–22:3x）

修正內容見第 6 節。重跑的是受影響的 9.3 項目；樹在暫存區 `…\scratchpad\p06td\fix\`。這段時間 main 的 KataGo GPU 試跑從 22:05 起在跑，這邊同時最多 2 個 pytest 行程（04 的演練是 6 個 worker，單獨跑），沒有用 GPU。

| 項目 | 指令 | 結果 |
|---|---|---|
| 單元測試 | `PY -m pytest -q tests/test_t06_player.py tests/test_t06_official_log.py` | 71 passed、172 subtests（第一輪 59 passed；新增的 12 項是第 6 節列的審查情境，含 TagVersusNew 3 項照舊通過） |
| 06 的演練（9.3 第 1、3、4、5 項） | `C10_SLOW=1 C10_KEEP=1 C10_DRYRUN_DIR=…\fix\dry PY -m pytest -q tests/test_t06_dryrun.py` | **26 passed**（34 分 4 秒）：`Q04Golden` 2、`FromValDrills` 20（第一輪 16＋審查的 4 項）、`OfficialShapeDrills` 4（第一輪 3＋報告交給 `official_log` 檢查的 1 項） |
| 第 1 項 Q04 | 同上 `Q04Golden`、`FromValDrills.test_q04_path_unchanged_on_this_tree` | 和 `q04_fake_golden.json` 相同（交件檔、三池 top5、八份正規化報告、選擇和閘門） |
| 第 3 項 from-val | 同上 | seam `passed`，sr06 V2 0.9758877068932452、V3 0.9697336206689693、V0 0.9221355664136796、R02 0.8998720491985943，r_forced_cost 1.1457767848959581（都和第一輪、H6P 相同）；public／private 選 V2，06 閘門 +0.077427744751423／+0.07073714495735872，r_forced_cost 1.0224／1.3778（r* 2.2516），各改 17 題第 1 名；06-OFF `62c88e217cb1…`、800 列、`check_submission --official` 通過；06-OFF 的 `inputs_bound` 沒有 `t2.json`；沒有 T2，所以沒有 06-ON |
| 第 4 項官方形狀 | 同上 | 一題對一人：public V2（閘門 +0.058572655079063396，r 1.6084，改 6 題）、private V2（+0.053271278101106434，r 0.8836，改 2 題），七條條件都成立，06-OFF `c3cabb1f1c4c…`、06-ON `bb9dd7588b15…`（和第一輪相同），和 Q04 第 1 名不同 22／27 和 26／29，06-OFF 對 06-ON 6／2；06-OFF 的 `inputs_bound` 不含 `t2.json`、06-ON 含。`--dup 80`：r 101.01／93.23，偵測器關，沒有 06-ON（只差 `private_switch_usable_and_detector_open`），06-OFF `8cd8b06cd82d…`。T2(b)：`abd_failed` b:public、b:private，沒有 06-ON。兩份報告換成正式模式、repo 相對路徑後交給 `official_log.t06_report_problems`、`t06_inputs_problems`：都回傳空清單 |
| 第 5 項新增的演練 | 同上 `FromValDrills` | K6' 已過、這棵樹已有一次 06：`apply` 結束碼 2、訊息「K6' 前已 commit 的 06 照常有效」、沒有 `not_used.json`（另一棵沒有 06 的樹照舊是 3，`not_used.json` 記 `k6p`）；seam_val 的 dumps 打亂（test02 的 seam_val 報告不變）：`seam` 結束碼 2（「differ from test02's seam_val apply_report.json inputs」），沒有 `not_used.json`；另一池的舊報告（舊的 test02 下 private 第 3 層）＋新的 test02（public 第 3 層、private 第 1 層）：`apply --pool public` 不再判兩池都退回（沒有 `not_used.json`），private 重跑後用 06；這一次 06-OFF 已 `prepared` 在上傳紀錄裡時 `submit --rerun`：結束碼 2，報告的 SHA 沒變 |
| 第 5 項 official_log 和否決 | 單元測試 `T06Rev3`、`T06Final`、`T06Prepare`、`T06Public`、`Veto`、`RealMode` | 兩次 06 執行時 06-ON 配自己那一次的 06-OFF；06-ON 之前重傳的 06-OFF 是配對的那一筆；`--wrong-upload`、`--upload-check-inconclusive` → 最終 06-OFF、要再傳一次；一次給兩種核對選項拒絕；暫停時 K7' 前什麼都不指定（三種 `final` 都拒絕、不要求重傳），K7' 後 06-OFF；d ≥ 0 而 06-ON 已不是最後一次上傳 → 06-OFF；`wrong_upload` 的 `veto.json` 寫 06-ON 不算數；06-ON 的 `prepare` 要 Q04 和 06-OFF 的分數、兩者同名檔警告時拒絕；同一 SHA 分數不同的警告，d 只對 06-ON 和配對的 06-OFF 印；FAIL_OPENING 在輸出已存在時仍是 3 |
| 第 2 項 R20、04 | 防誤改閘門（下面）；`PY -m pytest -q tests/test_t06_official_log.py -k TagVersusNew`（含在上面的 71 項）；`C10_SLOW=2 PY -m pytest -q tests/test_t04_dryrun.py` | 閘門見下；TagVersusNew 3 passed；04 的演練 32 passed、1 skipped、63 subtests（6 分 46 秒，`--workers 6`），和第一輪相同 |
| 第 6 項完整測試（單一行程） | `PY -m pytest -q -rfs tests --deselect tests/test_t02_infra.py::Build::test_mode_guards --deselect tests/test_t02_infra.py::FromVal::test_guards`（不設 `C10_SLOW`，一個 pytest 行程） | **2 failed、925 passed、94 skipped、2 deselected、580 subtests passed**（21 分 44 秒）。第一輪同一個指令是 12 failed、102 errors。失敗的 2 項都是 worktree 的環境，測試檔和它測的程式都和 main 相同：`test_p03_nb.py::Built::test_rebuild_is_byte_identical`（worktree 裡 `results/prereg03/design/nb_diff_a.txt` 是 CRLF，SHA `a0fcd845a560…`；折成 LF 是 `e4aa08270ce5…`，正好是重建出來的值）、`test_p04_opp.py::TestGates::test_out_dir_rules`（`REPO/data/prereg04` 經 junction 被當成 repo 外）。第一輪只在同一個檔裡失敗的 `test_p04_rank_fuse.py::FrozenTools::test_interface_constants_equal_p04_opp` 這次通過。要在 main（沒有 junction、LF checkout）合併後再跑一次（K5' 核對表） |
| `tests/conftest.py` 單獨驗證 | `PY -m pytest -q tests/test_p02_player.py tests/test_p06_player.py tests/test_p06_style.py`，有和沒有 `--noconftest` | 有：94 passed、1 skipped；沒有：7 failed、87 passed、1 skipped（`test_p02_player` 被 `p06_player` 的防護擋住，就是第一輪的情況） |
| `feats` 的時間 | 全部 `feats_report.json` 的 `seconds` | 第一輪（135 份）24.3–49.4 秒；這一輪（45 份）32.3–73.2 秒（同時有全套測試和 GPU 試跑） |

- **防誤改閘門**（`C10_T04_GATE=1 … tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py tests/test_p06_hygiene.py`）：修正前（承接的未 commit 修改，21:4x）和最後（HEAD `d7b5b2a`）都是 **1 failed, 50 passed, 35 subtests passed**，失敗的只有 `tests/test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`（打分紀錄帳多了 H6P 的 4 列，第一列 `2026-10-03T03:38:13Z`、`190478c`、`prereg06 H6P (A = prereg02 Q04, reference re-read)`；PREREG_04 rev 3 第 19 條允許）。加上 `tests/test_p06_player_hygiene.py tests/test_t06_hygiene.py`：18 passed、9 subtests（第一輪 17 項；多的是 `LedgerRows.test_later_rows_only_prereg07`，閘門下才跑）。
- `tools/official_log.py` 對 main 和 tag `p04-testday-pre06` 仍然只加行（`git diff main --numstat`：622 行新增、0 行刪除）；`.gitattributes` 對 main 只加 `results/test06/** -text` 一行。

## 3. 實作和登記、規格不同或補上的地方（寫進 rev 3，第 11 節的條號）

1. 防護第 0 層（test02 選中只用 CNN 的組合）比照第 3 層：交 test02 的列（第 7 條）。
2. seam 只讀 test02 的 seam_val 判過的 dumps（第 11 條）；失敗的結束碼：dumps 逐位元組相同時 4（本機漂移）、不同時照 398/400，沒過 3（第 13–14 條）。spec 7.1 沒寫後者。
3. `.venv` 版本不同、凍結檔被改（8 個 H6P 輸出、打分紀錄帳不是 4 列 H6P）是 2，不是 4 或 3（第 4 條）。前提和登記的 3 先於「輸出已存在」核對（第 5 條）。K6' 已過而 K6' 前已 commit 過一次 06：2，不寫 `not_used.json`（第 4、27 條）。
4. `official_log.py` 另加了 `prepare` 的規則：有符合 5.4 的 06-OFF 上傳時，重傳 Q04 要帶合格的 `--not-06-reason`（比照 04 的 `prepare_rule_t04`；spec 第 8 節只寫了 `final`）。
5. 規則指定的最終選擇多了一種「還沒定」：06-ON 已上傳、K7' 未過，沒有 `veto.json` 或否決暫停時 `final` 一律拒絕（第 24 條）。
6. `veto.json` 要在 K7' 前 commit，`official_log` 才認 06-ON 是最終（K7' 的「否決已判定」用 commit 時間核對）；d ≥ 0 而 06-ON 已經不是最後一次上傳時指定 06-OFF（第 24 條，比 5.3 嚴）。
7. 同名檔警告的核對結果寫死成三種（第 19 條，比 5.3 嚴，只會把開關關掉）：沒傳錯 → 照 d；06-OFF／06-ON 這一對傳錯、或核對不出 → 最終 06-OFF。06-ON 之前的 Q04／06-OFF 警告由 `prepare` 06-ON 擋住，傳錯的 06-OFF 在 06-ON 之前重傳，06-ON 配最新一筆 06-OFF（第 19、20、22 條）。
8. 06-OFF 的 `inputs_bound` 不含 T2 和 `top5_on`；這一次 06 進了上傳紀錄之後 `submit --rerun` 拒絕（第 18 條）。
9. `official_log public` 另印同 SHA 分數不同的警告，d 只對 06-ON 和它配對的 06-OFF 印（第 23 條）。
10. 演練樹的 T2 用 `t06_fake_tests.py fake-t2` 的替身（`fake: true`，正式模式拒絕）：from-train 的棋都是訓練棋，真的 T2(a) 一定判 FAIL，06-ON 就演練不到。替身的 (b)(c)(d) 用 `t02_checks.game_keys` 在樹上實算，(a) 設 0。
11. `tools/t06_fake_tests.py official` 呼叫 `t02_fake_tests.from_train` 時自己傳入 fit／inner 的列（`local_dir=None`），所以等級的順序是排序後的等級名稱，和 CLI 版（`c10data.LEVELS`）挑到的玩家不同；不影響演練的用途。
12. `tests/conftest.py`（第 29 條）：凍結的 `p06_player.py`／`c1h_lib.py` 在 import 時裝的 `data/local` 防護只留在用到它們的測試檔裡。

## 4. 本機環境的注意事項

- worktree 的 `data` 是 junction：`tests/test_t02_infra.py` 的 `Build::test_mode_guards`、`FromVal::test_guards` 會試著寫進 `REPO/data/...`，期待被「在 repo 裡」的防護擋下；但 junction 的實體路徑在 main 底下，防護用實體路徑判斷，所以沒擋，寫進了 main 的 `data/`。10/3 16:38 跑全套測試時發生一次，產生 `data\test_should_not_exist\` 和 `data\fake_should_not_exist\`（合成的假樹，不含答案，在 gitignore 裡）；主 session 在 10/3 21:45 前已刪掉。之後在 worktree 跑全套測試都 `--deselect` 這兩個測試（第 2.7 節那一次也是）。
- 新開的 worktree 用 `core.autocrlf=true` checkout，沒有標 `-text` 的凍結檔會變成 CRLF（例如 `results/prereg02/rank/pseudo_inner.json`：原始 SHA `34bd41a2e674…`，折成 LF 後 `3a51b3dd4edf…`＝凍結紀錄的值），所以有幾個測試在 worktree 裡失敗、在 main 上不會（第 2.6、2.7 節）。

## 5. 審查（三個角度，獨立代理，10/3 17:4x–18:0x，對 HEAD `777e38c`）

第一輪建置之後，由三個代理各自獨立審查（spec 9.3 第 6 項），都沒有 commit、沒有改追蹤檔，演練樹都在各自的暫存資料夾：

- **正確性**：重跑 9.3 第 1–5 項，建置者的數字全部逐位對得上（seam、兩池的 sr06、閘門、r_forced_cost、四份交件檔的 SHA、負面演練的結束碼）。另拿 10/1 真實 Colab 彩排的 seam_val dumps（`data\rehearsal_backup\results_20261001\test02_player_results.zip`）和 prereg02 比，`dumps_bitwise` 判為真：正式那天 seam 會走逐位元組的路徑。3 個 major、6 個 minor。
- **程序與防護**：R20、04、Q04 的路徑沒變，凍結檔沒動，`official_log.py` 只加行，沒有 06 紀錄時輸出和 tag 逐位元組相同（TagVersusNew 3 passed、128 subtests；空 log 上 tag 版和新版 `status` 的輸出 `cmp` 相同）。用測試的 fixture 重現了否決暫停和傳錯檔兩類漏洞（S1–S5）。3 個 major、9 個 minor。
- **操作者**（冷讀 `docs/TESTDAY_06.md`）：主流程照得做，兩則訊息的模式正確，三個同名檔的資料夾寫得清楚，每天 5 次是每場各自算（`docs/REGISTRATION.md`）。3 個 major、10 個 minor。

三份審查的防誤改閘門都是 1 failed（已知的 H6P 4 列）、50 passed、35 subtests；加上兩支 06 的 hygiene 測試是 1 failed、67 passed、44 subtests。

## 6. 審查意見的處理（逐條；修正在 `e50145f`、`d7b5b2a`，文件在本次 rev 3 和 `docs/TESTDAY_06.md`）

| 來源 | 意見 | 處理 |
|---|---|---|
| 正確性 major 1 | `veto` 一律拿最早一筆 06-OFF；test02 重跑、06 重做後有兩次執行時 run_id 對不上，06-ON 永遠判不了 | 修正：`veto` 和 `official_log`（`prepare`、`t06_state`、`public`）都配「同一個 run_id、比 06-ON 早上傳的最新一筆 06-OFF」（rev 3 第 20 條）。測試 `Veto.test_two_runs_pair_the_same_run`、`T06Rev3.test_off_uploaded_again_before_on_is_paired` |
| 正確性 major 2／程序 major 2／操作者 major 2 | 5.3「核對出傳錯 → 重傳 → 再判」工具做不到：06-ON 不能再 prepare（E1），重傳的 06-OFF 在 06-ON 之後、`veto` 卻拿傳錯的那筆算 d；只有 06-OFF 時的警告指到第 7 步，`veto` 又會拒絕 | 選最安全、只會關掉開關的做法，寫進 rev 3 第 19 條（明寫比 5.3 嚴）：06-ON 之前，`prepare` 06-ON 要求 Q04 和最新一筆 06-OFF 的分數都記了、沒有同名檔警告；傳錯的 06-OFF 在 06-ON 之前重傳，06-ON 配最新一筆（這是 5.3 的「重傳再判」能做到的部分）；核對出沒傳錯或核對不出 → 這一季不做 06-ON。06-ON 之後：`veto --wrong-upload`、`--upload-check-inconclusive` → 不用開關、最終 06-OFF、再傳一次 06-OFF；只有核對出沒傳錯才用 `--confirmed-upload-check` 判 d。`docs/TESTDAY_06.md` 第 6、7、8 步照改。測試 `Veto.test_wrong_upload_or_inconclusive_gives_off`、`test_one_check_option_only`、`T06Rev3.test_wrong_upload_or_inconclusive_gives_off`、`test_upload_check_veto_cannot_say_on`、`T06Prepare.test_order_rules`、`T06Public` |
| 正確性 major 3／程序 minor | rev 3 第 11.1 節第 4 條寫「8 個檔或打分紀錄帳不符是 3」，程式是 2 | 改文字（程式的 2 比較安全）：rev 3 第 4 條分成兩點，8 個檔或 4 列 H6P 不符是 2（凍結檔被改），下界或 RESULTS 的「採用」不成立是 3 |
| 正確性 minor 1 | 「兩池都退回」直接相信另一池的舊報告，test02 重跑後可能錯判 3（不可逆） | 修正：只採用以現在的 test02 那一池報告和現在的 seam 做出的另一池報告（rev 3 第 8 條）。演練 `FromValDrills.test_stale_other_pool_report_ignored` |
| 正確性 minor 2 | `seam` 讀的 dumps 沒和 test02 的 seam_val 綁在一起 | 修正：兩個 dumps 的 SHA 要等於 test02 `seam_val/apply_report.json` 記的輸入（已 commit），不等是 2（rev 3 第 11 條）。演練 `test_seam_dumps_must_be_test02s` |
| 正確性 minor 3 | 「輸出已存在」先於前提，蓋掉登記的 3 | 修正：`feats`、`seam`、`apply`、`submit` 先跑全部前提（rev 3 第 5 條）。測試 `RealMode.test_registered_3_before_outputs_exist` |
| 正確性 minor 4 | 06-OFF 傳出後再 `submit --rerun` 會改寫報告，讓已上傳的 06-OFF 不符合 5.4 | 修正：上傳紀錄裡有這一次 06 的任何一筆時拒絕（2）（rev 3 第 18 條；TESTDAY 第 5 步寫明要先給 `--aidea-page-checked`）。演練 `test_submit_rerun_refused_once_logged` |
| 正確性 minor 5 | 06-OFF 也綁了 `t2.json`，T2 重寫會讓已上傳的 06-OFF 失效 | 修正：06-OFF 不綁 T2 和 `top5_on`，06-ON 綁（rev 3 第 18 條）。`OfficialShapeDrills.test_reports_pass_official_log_checks` 核對 |
| 正確性 minor 6／程序 minor | `feats` 的秒數文件不一致 | 改成實測範圍並寫明出處（第 2.7 節最後一列；TESTDAY 第 2 步） |
| 程序 major 1 | 否決暫停、K7' 未過時 `status` 已叫人重傳 06-OFF；照做後核對沒傳錯會指定 06-ON，但最後一次上傳是 06-OFF | 修正：暫停時指定「還沒定」（`final` 拒絕、不要求重傳），K7' 後才 06-OFF；d ≥ 0 只有 06-ON 仍是最後一次上傳才指定 06-ON，否則 06-OFF；新增 `--upload-check-inconclusive`；兩支 `status` 都印「核對前不重傳、不設最終選擇」（rev 3 第 19、24 條）。測試 `T06Final.test_paused_veto_is_undecided_before_k7`、`T06Rev3.test_on_kept_but_not_last_gives_off`、`Veto.test_status_paused_line` |
| 程序 major 3 | `PY -m pytest -q tests`（TESTDAY.md 就緒清單）在單一行程跑不過 | 修正：`tests/conftest.py`（凍結的 `p06_player.py`、`p06_style.py` 不改），rev 3 第 29 條。單一行程全套 2 failed（都是 worktree 的 CRLF／junction）、925 passed；第一輪 12 failed、102 errors（第 2.7 節）。合併後在 main 再跑一次列進 K5' 核對表 |
| 程序 minor 1 | TESTDAY「K6' 過了之後每個指令都是 3」寫錯；K6' 後誤跑會寫 `not_used.json`，和 `official_log` 的指定矛盾 | 修正：K6' 前已 commit 過一次 06 時 K6' 後是 2、不寫 `not_used.json`（rev 3 第 4、27 條；TESTDAY「日期」一節）。測試 `RealMode.test_k6p_passed_after_a_committed_run_is_2`、演練 `test_k6p_passed_after_the_run_is_2` |
| 程序 minor 2 | rev 3 文字和實作三處不一致（`.gitattributes` 兩行、「每個指令寫報告」、「2.7 的表補一列」） | `.gitattributes` 刪掉註解行（對 main 只加一行）；第 2 條改「`status` 只讀不寫」；修訂紀錄那一列改成「表不改，補充在第 7–10 條」 |
| 程序 minor 3 | 「結束碼 3 之後不重跑」和 test02 重跑那一節打架 | TESTDAY 開頭和 rev 3 第 26 條：只有 test02 本身依 TESTDAY.md 重跑時，照「test02 重跑了」從 `feats` 重做；`--rerun` 的三種用法寫明 |
| 程序 minor 4 | 第 8 步重傳的 06-OFF 沒有傳錯檔檢查，還會再印一次 d | 修正：同 SHA 分數不同時印警告；d 只對 06-ON 和配對的 06-OFF 印（rev 3 第 23 條）。測試 `T06Rev3.test_reupload_scored_like_another_file_is_warned` |
| 程序 minor 5 | 合併後沒有 06 路徑的指紋；閘門沒跑 TagVersusNew | TagVersusNew 加進 TESTDAY 第 1 步和 rev 3 第 28 條的閘門；06 路徑的指紋和 tag 要等合併後的檔定版才做，列進 K5' 核對表（未做） |
| 程序 minor 6 | 沒有測試把 `t06_player` 真的報告交給 `official_log` 檢查 | 新增 `OfficialShapeDrills.test_reports_pass_official_log_checks`（通過） |
| 程序 minor 7 | main 的 `data/` 留著兩個假樹資料夾 | 主 session 已刪；第 4 節和 K5' 核對表寫明 worktree 裡一律 `--deselect` 那兩個測試 |
| 操作者 major 1 | 閘門只准 H6P 那一項失敗；A100（rev 7 第 2 條）、D5、rev 9、第二次 test04 彩排紀錄、其他依規打分的列都會讓 06 停下 | TESTDAY 第 1 步加「其他登記依規產生的失敗」四項和核對方法（照 TESTDAY_04 第 1 步、「R20 側的已登記變動」），rev 3 第 28 條同步；`test_p06_player_hygiene` 的「H6P 之後只准 PREREG_07」只在 06 的閘門跑（一般的全套測試不因別的登記變紅）；A100 時 seam 可能走 398/400 而結束碼 3 也寫明 |
| 操作者 major 3 | 開頭只寫 U5 被取代，沒寫 TESTDAY「不為 public 試變體上傳」和 T6「沒有官方明文不用一題對一人」對 06-ON 由 E1、E4 取代 | TESTDAY 開頭加一點，列出 E1、E4（和 E2、E3） |
| 操作者 minor 1 | public 分數要照畫面全部位數 | 訊息 B 改成截分數那一格，Claude 照截圖抄全部位數 |
| 操作者 minor 2 | 強制探測的說明太多術語 | 第 7 步訊息 A 改成白話（一對一的意思、為什麼不傷名次、public 可能下降） |
| 操作者 minor 3 | 06 和 04 同一天時可能有兩則等 OK 的訊息；`git add -A` 會混進 04 的輸出 | 開頭加「同一時間只有一則等 OK 的訊息 A」；commit 用指定路徑 |
| 操作者 minor 4 | 檔案選擇視窗會停在上一次的資料夾；訊息 B 前沒有核對 SHA | 訊息 B 寫「用拖的，不要按選擇檔案」；發訊息 B 前 `Get-FileHash` 核對 |
| 操作者 minor 5 | FAIL_OPENING 那一列寫「本檔到此結束」，和結束碼表的「→ 第 9 步」不一致 | 改成 → 第 9 步（`final` 設 Q04、照常跑 `status`） |
| 操作者 minor 6 | 結束碼 4 之後怎麼接續沒寫；`--rerun` 的用法 | 結束碼表和「每一步都適用的規則」寫明 `--rerun --reason` 的三種用法 |
| 操作者 minor 7 | 指定 Q04 而最後一次上傳是 06 的檔時，要重傳的是 Q04 | 第 9 步補 Q04 的 `prepare` 指令和資料夾 |
| 操作者 minor 8 | AIdea 頁面的判定措辭 | 寫了同一人可出多題、不是一一對應（不論用什麼字）或沒把握，都加 `--aidea-not-one-to-one`；引用原文和網址（rev 3 第 17 條 (f)） |
| 操作者 minor 9 | AIdea 上傳失敗怎麼辦 | 「每一步都適用的規則」加一點：不記 `uploaded`，沿用同一筆 `prepare`、重取 OK |
| 操作者 minor 10 | 每天次數要在訊息 A 之前查 | 每則訊息 A 之前跑 `official_log.py status` 看當天次數 |
