# test09 建置紀錄（10/9，分支 p09-testday）

## 建置
- 分支從 main `0b65255`（＝tag `p08-testday-pre09`，PREREG_09 5.1 第 1–3 步做完）開；worktree `D:\AI-Portfolio\competitions\c10-p09-testday`，`data` 是指向主 repo 的 junction。Python 用主 repo 的 `.venv`（絕對路徑）；沒有用 `.venv-gpu`、沒有用 GPU。
- commit：`b4ee6aa`（`tools/t09_rank.py`、`tools/t09_fake_tests.py`、`.gitattributes`／`.gitignore` 加行）、`767210d`（`tools/official_log.py` 只加行 615 行＋`tests/test_t09_official_log.py`）、`d9bdf34`（`results/test09/design/tol_e_measurement.json`、`t09_fake_golden.json`、`tests/test_t09_rank.py`、`test_t09_hygiene.py`、`test_t09_dryrun.py`）、之後一個 commit 加 `docs/TESTDAY_09.md`、PREREG_09 rev 3（修訂表一列＋第 13 節）和本檔。
- 09 只讀 08 的 P1 塊（`data/test08/p1/<組>/`），不跑 GPU；X、EX、lp_P1TO_E 都用 `tools/p09_forward.py`（numpy float64，和驗證題的 apply 同一套函式）。`t09_rank.py` 不 import torch、`p09_map`、u1_lib（`tests/test_t09_hygiene.py` 用 ast 檢查；單元測試確認 pytest 行程裡沒有載入 torch）。

## tol_E 的實測（任何測試資料之前）
- `PY tools/t09_fake_tests.py measure-tol --tree D:/c10_t08_reh --json results/test09/design/tol_e_measurement.json`（08 的 4090 演練樹：`gpu/test08` 是 GPU 算的 seam 400 盤，`gpu_pub/test08` 是 GPU 算的 private 前 1,000 盤；10/9 02:22–02:31 的演練，torch 2.11.0+cu128、cuDNN 91900、驅動 616.92）。只讀那棵樹，沒有改它。
- seam（和驗證塊同樣分塊）：EX、X、lp_P1TO_E 的差都是 0，逐位元相同。
- private 前 1,000 盤（驗證列 4,787–5,786，每塊都在驗證塊格線之外）：lp_P1TO_E 每盤最大差 **m＝0.011267**（中位數 0.0011、99% 分位 0.0046；6 盤 > 0.005；最大等級 999／1,000 相同）→ **tol_E＝max(0.04, 2.5 × m)＝0.04**。
- **（10/9 主 session 已決定，見下面「rev 4」）** 原本要主 session 決定的事：同一批 private 盤的**標準化 EX 最大差 0.254**，超過 5.2 寫死的 0.05。來源幾乎只有兩維「log p < log 0.05 的比例」（t 0.254、o 0.239；1,000 盤中 24 盤有變），是有門檻的計數，fp16 捨入讓一兩手跨過門檻就跳一格；其他 42 維最大 0.016。lp 只差 0.011，對預測幾乎沒影響。測試日的 seam 和驗證塊同樣分塊（兩次 4090 實測都是 0），所以照登記維持 0.05；但若測試日 cuDNN 換了演算法，09 的接縫會判 4、09 用不上。可選的 rev 4（要在任何測試資料之前）：EX 容許差改成 2.5 × 0.254 ≈ 0.64，或 frac_lowp 兩維只記錄、其他 42 維維持 0.05。PREREG_09 rev 3 第 13 節第 5 點已寫明。

## rev 4：EX 只判 42 維（10/9，主 session 決定，在任何測試資料之前）
- 規則：09 的 GPU 接縫的標準化 EX ≤ 0.05 只判兩維 frac_lowp 以外的 42 維；兩維 frac_lowp 只記錄（最大差、跳動盤數）。lp_P1TO_E ≤ 0.04 和其他條件不變。放寬整個 EX 到 0.64 的選項不採用（會讓其他 42 維幾乎不被檢查）。推翻 09 的修訂從 rev 5 起（`P09_MIN_REV = 5`）。PREREG_09 修訂表一列＋第 14 節。
- 程式：`tools/t09_rank.py`（`EX_RECORD_ONLY`、`gpu_seam.json` 多記 `max_abs_std_ex_all_44`、`frac_lowp_record_only`）、`tools/t09_fake_tests.py` 的 measure-tol 輸出、`tools/official_log.py` 只改本分支自己加的 09 行（對 tag `p08-testday-pre09` 仍是只加行，`test_t08_path_hygiene`／`test_t09_hygiene` 的 InsertOnly 都過）。
- 同一次 4090 實測在新規則下：42 維最大 0.0161、lp_P1TO_E 0.0113，不同批次組合的 1,000 盤也判過 → `results/test09/design/ex_rule_rev4_check.json`（rev 3 的 `tol_e_measurement.json` 釘住不改）。
- 新測試：`test_t09_rank.py::Seam::test_frac_lowp_jump_recorded_only`（只有兩維 frac_lowp 跳 0.25：判過、記 2 盤；`t_logp_q50`、`o_ent_std`、`t_log1p_count` 任一移動 0.06：判失敗）；`test_ex_tolerance` 只在 42 維裡挑；hygiene 加 `test_ex_rule_in_tools`、rev 4 那一列、`P09_MIN_REV = 5`；log 測試的推翻改用 rev 5。
- 演練輸出不變（同樣分塊的 seam 兩邊都是 0），`t09_fake_golden.json` 的 8 個 SHA 照舊相同。

## CPU 演練（from-val 假樹）
- 樹：`D:\c10_t09_reh`（`D:\c10_t08_reh` 的 07 演練樹的副本，去掉 test08、gpu、gpu_pub；`D:\c10_t08_reh` 本身沒動）。
- 08 的 from-val 假塊沒有逐局面陣列（`pos_side` 長度 0），09 的 EX 算不出來。`t09_fake_tests.py from-val` 先呼叫 `t08_fake_tests.from_val`（不改），再把同一批驗證盤的逐局面陣列抄進每一塊，重寫 `p1_<組>_report.json` 的塊列；之後跑 08 的 `apply --dry-run`（交件檔 `457c8542…`，等於 08 的基準），再跑 09。
- `t09_rank.py apply --dry-run` 結束碼 0（約 8 秒）：
  - 09 的 GPU 接縫：counts 全同、標準化 EX 0、lp_P1TO_E 0，EX 和平均逐位元相同；
  - Q08 重算等於 08 的 `rank_test_probs.csv` **逐位元組相同**，規則 A／B 差 0，可用盤等於 08 的 `test_lp_p1.npz`；
  - 本機接縫和 prereg09 **逐位元組相同**（`rank_val_probs.csv` `3f8d32f4…`、`pred_B.csv` `d1a2f1d6…`、`pred_A.csv` `5c412993…`），seam_val 的 ρ 等於 prereg09 apply 的 0.96985；
  - 800 題測試列：機率和 prereg09 驗證前 800 列最大差 4.4e-16，規則 A、B 等級全同；lp_P1TO_E 有 132／9,602 盤差 ≤ 1.9e-14（每組一次前向 vs 驗證題 19,156 列，BLAS 捨入；rev 3 第 13 節第 9 點）；
  - `check_submission --official` 通過；交件檔 `eabba882…`；規則 B 和 08／07／04／R20 在 public 題不同 74／134／151／169 題（只記錄）。
- `C10_SLOW=1 C10_KEEP=1 C10_T09_TREE=D:/c10_t09_reh` 的 `tests/test_t09_dryrun.py`：21 過、1 略過（09 基準還沒寫時）；寫了 `t09_fake_golden.json` 之後見下面「測試」。R20／04／07／08 的 38 個輸出等於 `baseline_pre09.json`；09 不改 test02／04／07／08 的任何檔。負面演練的結束碼都照登記：09 的 GPU 接縫（EX）4、08 的塊在 08 apply 後被改 2、08 的 P1 報告被改 2、Q08 差一個值 4、08 的輸出被改 2、08 判不用 3、08 沒做完 2、K6-09 3 和「成功的那次仍有效」2、FAIL_OPENING 3、輸出已存在 2（`--rerun` 沒有修訂 2、`--bug-revision 3` 2）、BLAS 變數 2、raw 資料夾 2、正式模式用演練旗標 2。

## 測試
- `C10_SLOW=1` 的 `tests/test_t09_dryrun.py` 從頭建樹（`test_t07_dryrun` 的 R20 → 04 → 07 樹，再 08、09；暫存資料夾在 repo 外，3.5 分）：**22 過、0 略過、46 個子測試過**，含 R20／04／07／08 的 38 個輸出等於 `baseline_pre09.json`、09 的 8 個輸出等於 `t09_fake_golden.json`（第一次寫基準用的是 `D:\c10_t09_reh`，從頭建的樹逐位元組相同）。
- 閘門（十五個檔：main 的 CLAUDE.md 十四個加 `test_t09_hygiene.py`，`C10_T04_GATE=1`）：174 過、1 略過（PREREG_09 rev 3 那一列在這個 commit 之前還沒有），只有已知的帳列那一項失敗。
- `tests/test_t09_rank.py` 27 過；`tests/test_t09_official_log.py` 26 過、312 個子測試（TagVersusNew：tag `p08-testday-pre09` 對新版，R20／R10／04／06 流程、07 流程、08 流程，都在磁碟上有 test09 樹時逐位元組相同）。既有的 `test_t04/t06/t07/t08_official_log.py`：127 過（只加行之後）。
- 全套一個行程（`C10_T04_GATE=1`，CPU，12.7 分）：1325 過、10 失敗、171 略過。10 個失敗：
  - 已知的帳列 1 個；
  - 我自己的 `test_t09_rank.py::Helpers::test_no_guarded_module_loaded`：全套裡別的測試檔先載入了 torch（`sys.modules` 是整個行程共用的）。已改成用子行程只 import 兩支 09 工具再看（改後和 `test_p09_map.py` 一起跑 42 過）；
  - 其餘 8 個都是 worktree 的環境造成，和 09 的程式無關（同 08 的 build_review）：CRLF checkout 的 `results/prereg04/rank/frozen_config.json` 讓 `p07_rank_fuse` 的 raw SHA 核對失敗 → `test_p07_rank_fuse.py` 2 個、`test_p08_rank_fuse.py` 1 個、`test_p09_rank_fuse.py` 1 個；`nb_diff` 同理 → `test_p03_nb.py` 1 個；`data` 是 junction → `test_p04_opp.py` 1 個、`test_t02_infra.py` 2 個（後兩個在主 repo 的 `data/` 寫了 `test_should_not_exist`、`fake_should_not_exist` 假資料夾，已刪）。
  - 合併後在 main 上要再跑一次全套，確認這 8 個會過。

## 獨立審查（10/9，兩份，主 session 派出）與修正
- 兩份都沒有 blocker／major。照審查修的 minor（PREREG_09 rev 4 補記第 7–10 點）：
  1. `read_set_chunks` 原本只檢查五個逐局面陣列在不在；空的或截短的陣列會算出錯的 EX 而不被擋。加 `pos_problems`：等長、範圍、每盤每方局面數＝`counts`（bincount），不符拒絕（2）。154 塊（08 驗證 96、4090 演練的 GPU 塊、add-positions 的演練塊）全部成立。測試 `test_incomplete_positions_refused`（5 種情況）。
  2. `check_bug_revision` 說明的「N >= 4」、`test_t09_hygiene.py` 略過訊息的「P09_MIN_REV = 4」改成 5。
  3. `docs/TESTDAY_09.md`：第 1 步寫明 `apply` 經由 08 的前提也要求 R20 的第一次上傳；結束碼 3 和第 5 步寫明 09 不用／從沒 prepare 時，最終照 `docs/TESTDAY_08.md` 第 6 步；寫出十五個檔的閘門指令全文。
  4. PREREG_09 加「rev 4 補記」（只加行）：審查、修正、逐局面陣列的不變式、rev 3 那一列的「rev 4 起」由第 14 節第 6 點取代（推翻要 rev 5 起）。修訂表的 commit 編號留給合併時回填。
  5. （選做，做了）`test_t09_hygiene.py`：`docs/COLAB_RUN.md` 列入凍結；`NoNewDesignFiles` 檢查 `results/test02/design`、`test02/null`、`test04/design`、`test07/design`、`test08/design` 自 tag `p08-testday-pre09` 起沒有新增檔案（已追蹤或未追蹤）。沒有對整個 `results/test02|04|07|08/**` 做這個檢查：合併後真正的測試日會在 `rank`、`opp`、`hs`、`p1` 等資料夾新增輸出，整個樹的檢查會在測試日誤報。
- 沒修的：無。

## 還沒做（合併前要補）
1. ~~獨立審查~~：兩份做完（見上）。
2. ~~EX 容許差~~：rev 4 已定（見上）。
3. 合併回 main 之後：在 main 重新 checkout `results/test09/design/baseline_pre09.json`（`.gitattributes` 改成 `-text` 之後工作樹的 CRLF 會被看成已修改；rev 3 第 13 節第 14 點）；回填 PREREG_09 rev 3、rev 4 兩列的 commit 編號；CLAUDE.md 的閘門指令（十四個檔）加 `tests/test_t09_hygiene.py`（十五個檔）。
4. 合併前在 main 上再跑一次 `C10_SLOW=1` 的 `tests/test_t08_dryrun.py`（基準）和 `tests/test_t09_dryrun.py`。
