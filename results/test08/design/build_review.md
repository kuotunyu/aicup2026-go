# test08 建置紀錄（10/9，分支 p08-testday）

## 建置
- 分支從 main `dd336d3`（＝tag `p07-testday-pre08`，PREREG_08 5.1 第 1–3 步做完）開；worktree `D:\AI-Portfolio\competitions\c10-p08-testday`，`data` 是指向主 repo 的 junction。Python 用主 repo 的 `.venv`／`.venv-gpu`（絕對路徑）。
- commit：`0e78e6d`（`tools/t08_rank.py`、`tools/t08_fake_tests.py`、`.gitattributes`／`.gitignore` 加行）、`6a7db9c`（`tools/official_log.py` 只加行＋`tests/test_t08_official_log.py`）、`0a8dffd`（`tests/test_t08_rank.py`、`tests/test_t08_hygiene.py`、`tests/test_t08_dryrun.py`、`t08_fake_golden.json`）、之後一個 commit 加 `docs/TESTDAY_08.md`、PREREG_08 rev 3（第 13 節）和本檔。
- 沒有用 GPU；`p1` 只用 `--fake-scorer` 在 CPU 上跑過幾盤。

## CPU 演練（from-val 假樹）
- 假樹：R20／04／07 的 dry run（`tests/test_t07_dryrun.py` 的樹），`t08_fake_tests.py from-val` 從 `data/prereg08/val` 的塊（SHA 照 `val_report.json` 核過）複製 seam（驗證 0–399 盤，等於 `val_seam_feats.npz`）、public（驗證題 1–400）、private（401–800）。
- `t08_rank.py apply --dry-run` 結束碼 0：
  - GPU 接縫 0／0（逐位元相同，因為是同一份資料）；
  - Q02H 重算等於 07 的 `rank_test_probs.csv`（`ede52634…`）；07 綁的 54 個輸入都沒變；
  - 手數交叉核對（對 07 HS、對 04 O4）0 不符，4,787／4,815 盤比過；P1 失敗 0；
  - 方向 ρ public 0.9723、private 0.9647（門檻 0.4837）；seam_val 0.96832…，等於 prereg08 apply 的值；
  - 本機接縫和 prereg08 **逐位元組相同**（`rank_val_probs.csv` `7a513367…`、`pred_B.csv` `b02d4f94…`、`pred_A.csv` `e24699e9…`）；
  - 800 題測試列：機率和 prereg08 驗證前 800 列最大差 2e-24 級（≤ 1e-12），規則 A、B 等級全部相同；lp_P1TO 有 48／9,602 盤差 ≤ 4e-15（每組一次矩陣乘法 vs 驗證題 19,156 列一次乘，BLAS 捨入；rev 3 第 13 節第 9 點）；
  - `check_submission --official` 通過；規則 B 和 07／04／R20 在 public 題不同 126／145／167 題（只記錄）。
- `tests/test_t08_dryrun.py`（`C10_SLOW=1`，用現成的 07 假樹 `C10_T08_TREE`）：26 過、1 略過（08 基準還沒寫時）；寫了 `t08_fake_golden.json` 之後 T08DryRun 7 過。R20／04／07 的 27 個輸出等於 `t07_fake_golden.json`。負面演練的結束碼都照登記（GPU 接縫 4、P1 檔和報告不符 2、塊被改 2、環境不同 4、P1 早於 R20 上傳 2、Q02H 差一個位元組 4、07 綁的輸入被改 2、手數交叉核對 4 且重跑要 `--bug-revision`、失敗 23／24 盤 0／3、方向 3、07 不用 3、07 沒做完 2、K6-08 3 和「成功的那次仍有效」2、FAIL_OPENING 3、輸出已存在 2、raw 資料夾 2；`p1` 用 fake scorer：舊塊 2 且不刪、續跑 todo 0、鍵和 test02 不同 2、鍵和 07 不同 2、seam 之前跑 public 2、正式模式用演練旗標 2）。

## 測試
- 閘門（12 個檔：CLAUDE.md 的十個加 `test_t07_path_hygiene.py`、`test_t08_hygiene.py`，`C10_T04_GATE=1`）：134 過，只有已知的帳列那一項失敗。
- 全套一個行程（`C10_T04_GATE=1`，CPU）：1178 過、8 失敗、145 略過（14.7 分）。8 個失敗：已知的帳列 1 個；其餘 7 個都是 worktree 的環境造成，和 08 的程式無關：
  - CRLF checkout：worktree 的 `results/prereg04/rank/frozen_config.json` 是 CRLF（main 是 LF），`p07_rank_fuse` 用 raw SHA 比 → `test_p07_rank_fuse.py` 2 個、`test_p08_rank_fuse.py` 1 個（Q02 接縫的子行程）；`results/prereg03/design/nb_diff_*.txt` 同理 → `test_p03_nb.py` 1 個。
  - `data` 是 junction：`data/...` 的 realpath 在主 repo，「在 repo 內」的防護判斷不到 → `test_p04_opp.py` 1 個、`test_t02_infra.py` 2 個。後兩個因此在主 repo 的 `data/` 寫了 `test_should_not_exist`、`fake_should_not_exist`（假資料，gitignore），已刪。
  - 合併後在 main 上要再跑一次全套，確認這 7 個會過（07 合併時同類的 4 個在 main 上都過）。
- `C10_SLOW=1` 的 `tests/test_t08_dryrun.py` 從頭建樹（R20 → 04 → 07 → 08，4.4 分）：27 過、40 個子測試過，含 R20／04／07 的 27 個輸出等於 `t07_fake_golden.json`、08 的 11 個輸出等於 `t08_fake_golden.json`。
- `data/prereg08/build_t07dry_t08.sh /d/c10_t08_reh` 在 CPU 上跑過一次（`BUILD_DONE`），08 的 CPU 步驟在這棵樹上重跑，交件檔 SHA 和基準相同（`457c8542…`）。這棵樹 `D:\c10_t08_reh` 留給主 session 做 4090 實測（`gpu_rehearsal_t08.sh` 會先清掉 `<S>/gpu`、`<S>/gpu_pub`）。
- `test_t08_official_log.py`：TagVersusNew（tag `p07-testday-pre08` 對新版：R20／R10／04／06 流程加上 test08 樹在磁碟上，另加 R20 → 04 → 07 的流程）全部逐位元組相同；08 的路徑 19 個測試。
- **審查修正之後（10/9）**：閘門十二個檔 134 過、只有已知的帳列 1 失敗；`tests/test_t08_rank.py`＋`test_t08_official_log.py` 53 過；`C10_SLOW=1 C10_T08_TREE=/d/c10_t08_reh` 的 `tests/test_t08_dryrun.py` 31 過（原 27＋新 4 個演練：STOP 結束碼 5、塊的題目檔／程式碼不同不重用、apply 塊的 RUN 設定／題目檔不同、三組程式碼不同），R20／04／07 的輸出仍等於 `t07_fake_golden.json`，08 的 11 個輸出仍等於 `t08_fake_golden.json`（修正只加核對，不改任何輸出，所以基準不變）。
- 既有的 `test_t04_official_log.py`、`test_t06_official_log.py`、`test_t07_official_log.py`：102 過（只加行之後）。

## 獨立審查（10/9，兩份）與修正
- 兩份審查都沒有 blocker／major。照審查修的（PREREG_08 rev 3 第 13 節補記第 18–24 點）：
  1. STOP：原本 `p1` 被 STOP 停下時結束碼 0，串接的 `p1_run.sh` 會接著開下一組、GPU 讓不出來。改成結束碼 5（`EXIT_STOPPED`），TESTDAY_08 的 `p1_run.sh` 改成：已有報告的組跳過、任何非 0 整支停；文件寫明 STOP 放在正在跑的那一組（不確定就三個都放）。測試 `test_stop_exit_code`、`test_stop_file_exits_5`（STOP 時 5、沒有塊、刪 STOP 後同一行續跑 0）。
  2. 塊的重用（`chunk_check`）加查題目檔 SHA、P1 程式碼 SHA、權重 SHA（fake scorer 為 `fake`）、precision／batch／channels_last／device；`apply` 的 `p1_sets` 逐塊核對鍵檔／題目檔（public／private 也和現在 tests 資料夾的題目檔比）、權重、RUN 設定，三組 P1 程式碼紀錄相同且等於磁碟上的 LF SHA。測試 `test_chunk_check_questions_code_and_run_settings`、`test_chunk_of_other_question_file_or_code_not_reused`、`test_chunk_run_settings_differ`、`test_p1_code_differs_between_sets`。
  3. `hs_keys_match` 改成整組相等（含列數），在 `--max-games` 截斷之前比。測試 `test_hs_keys_match_is_equality`。
  4. `t08_fake_tests.py from-val` 的 seam 核對加 n_moves、failed、slot、執色。
  5. TESTDAY_08：盤數改約 10,400；`p1_run.sh` 跑時不要改已追蹤的檔；閘門寫法（main 的 CLAUDE.md 已含 `test_t07_path_hygiene.py`，合併後加 `test_t08_hygiene.py` 為十二個檔）；結束碼表加 5。本機接縫容許差的來源更正為 `results/test04/rank/t04_apply_report.json` 的 `seam_local.tol`（同 07），報告欄位 `tol_source` 的文字跟著改。
  6. 4090 實測第 2 步原本是 public 前 400 盤＝seam 的同一批盤、同樣分塊（局面數 46,175／44,138 相同），差 0 沒有意義。改成 private 前 1,000 盤（驗證列 4,787 起，每塊都不在驗證塊格線上）；`compare` 多記驗證列範圍、每塊起點、是否和驗證分塊不同、局面數和 GPU 秒數。
- 沒修的：無。

## 4090 實測（10/9 02:22–02:31，主 repo 的 `data/prereg08/gpu_rehearsal_t08.sh /d/c10_t08_reh`，從 worktree 跑）
- 環境：RTX 4090、驅動 616.92、torch 2.11.0+cu128、cuDNN 91900；開跑前 nvidia-smi 沒有 compute 程序（只有 C+G 桌面程式）。
- 第 1 步 seam 400 盤：0 失敗，90,313 個局面 115 秒（786 positions/s；第一塊暖機 586、第二塊 1,224）。GPU 接縫：counts 等全同、標準化特徵差 0、lp 差 0、平均逐位元相同 → `results/test08/design/gpu_rehearsal_seam.json`。
- 第 1b 步 from-val 假 public／private＋`apply --dry-run`：結束碼 0，本機接縫逐位元組相同，交件檔 `457c8542…`（同 CPU 演練），規則 B 和 07／04／R20 不同 126／145／167 題。
- 第 2 步 private 前 1,000 盤（不同批次組合）：0 失敗，221,085 個局面 225 秒（983 positions/s；暖機後 1,152–1,179）。failed／counts／n_moves／執色全同；標準化特徵最大差 **0.0186**（≤ 0.05），lp_P1TO 最大差 **0.0160**（≤ 0.02；中位數 0.0024、99% 分位 0.0086、115／1,000 盤 > 0.005），平均不逐位元相同，P1 單獨的最大等級全同 → `results/test08/design/gpu_rehearsal_compare_private.json`。紀錄 `results/test08/design/gpu_rehearsal.log`（去掉 nvidia-smi 的桌面程式列）。
- 判斷：lp 差超過 0.005 檢討線（07 是 0.0059），但接縫比的是同一批分塊的 seam（兩次都 0），容許差 0.05／0.02 維持，不寫 rev 4；在 rev 3 補記第 24 點揭露。**要不要另外收緊或放寬由主 session 決定**；要改就要在任何測試資料之前寫 rev 4。

## 還沒做（合併前要補）
1. ~~4090 實測~~：10/9 做完（見上）。若之後要重做：CPU 建樹 `bash /d/AI-Portfolio/competitions/aicup2026-go/data/prereg08/build_t07dry_t08.sh <S>`（合併後加第二個參數 `/d/AI-Portfolio/competitions/aicup2026-go`），GPU `bash .../gpu_rehearsal_t08.sh <S> > <S>_gpu.log 2>&1`（第 2 步現在是 private 前 1,000 盤，比較檔是 `<S>/gpu_pub/compare_private.json`）。
2. ~~獨立審查~~：兩份做完（見上）。
3. 合併回 main 之後：在 main 重新 checkout `results/test08/design/baseline_pre08.json`（`.gitattributes` 改成 `-text` 之後工作樹的 CRLF 會被看成已修改；rev 3 第 13 節第 13 點）；回填 rev 3 那一列的 commit 編號；CLAUDE.md 的閘門指令（main 已有十一個檔，含 `tests/test_t07_path_hygiene.py`）加 `tests/test_t08_hygiene.py`（十二個檔）。
4. 合併前在 main 上再跑一次 `C10_SLOW=1` 的 `tests/test_t07_dryrun.py`（R20／04／07 基準）和 `tests/test_t08_dryrun.py`。
