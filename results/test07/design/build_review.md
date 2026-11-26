# test07 建置紀錄（10/5，分支 p07-testday）

## 建置
- commit：`fc73cb6`（official_log 只加行＋`test_t07_official_log.py`）、`3a651ca`（`t07_rank.py`、`t07_fake_tests.py`、測試、`t07_fake_golden.json`）、`30b6c5f`（`test_t07_dryrun.py`）、`5b5c098`（審查修正）。從 main `2d81c80` 開分支；worktree `D:\AI-Portfolio\competitions\c10-p07-testday`，`data` 是指向主 repo 的 junction。
- 審查第 1 輪（一個代理，三個角度合併）：沒有 blocker，3 個 major 都修了（hs 先查 nvidia-smi 再碰 CUDA、手數交叉核對不符後重跑要修訂、登記的結束碼 3 先於「輸出已存在」）；minor 1、2、3、5、7 修了；4、6 和 hs 的 GPU 型號檢查順序留著。spec 第 9 節要三份獨立審查，這次只做了一份合併的。

## CPU 演練（from-val 假樹）
- `apply --dry-run` 結束碼 0：本機接縫和 prereg07 逐位元組相同（rank_val_probs `6ae2db60…`、pred_B `51d7dbf8…`、pred_A `b229be2c…`）；800 題測試列等於 prereg07 驗證前 800 列加前綴；`check_submission --official` 通過；Q02 等於 04（`0324e141…`）；手數交叉核對 0 不符；失敗 0/4787、0/4815；ρ public 0.9438、private 0.9532、seam_val 0.94825。
- `test_t07_dryrun.py`（`C10_SLOW=1`）：25 過；負面演練的結束碼全部照登記。
- NORMALIZED：填了 3,213 格執色後 T1 判 NORMALIZED，HS 讀的 4,787 槽完全相同。
- lp_HS 靈敏度：單一 profile 平均移 0.0019（在 0.002 內）時 lp_HS 最多移 0.0214，所以由 lp_HS 那條把關。

## 測試
- 閘門（7 個檔）：只有已知的帳列那一項失敗；`test_t07_official_log.py` 27 過；`test_t07_rank.py` 24 過；既有 t04／t06 official_log 測試 75 過。
- 全套一個行程：1027 過、4 失敗，4 個都是 worktree 的 CRLF 造成（main 上同樣的測試會過）。

## 還沒做（合併前要補）
1. **4090 實測**：建置時 GPU 被別的程式（ComfyUI，`main.py --port 8188`）占用，沒跑。腳本：先 `data/prereg07/build_t04dry_t07.sh <S>`（CPU 約 5 分），再 `data/prereg07/gpu_rehearsal_t07.sh <S>`（約 15 分）。lp_HS 實測 > 0.005 時，在 rev 4 檢討容許差。
2. **06 的指紋和 tag `p06-testday-pre07`**：照 PREREG_07 5.1 應該先做，分支開出時還沒做；合併前在 main 補做，或在 rev 4 寫明理由。
3. 正式模式的一次性 clone 演練、`C10_SLOW=2` 全跑、三份獨立審查。
4. PREREG_07 rev 4（測試日程序，第 11 節）和 `docs/TESTDAY_07.md`：草稿在建置代理 10/5 的回報裡（主 session 對話紀錄），合併前寫進來。
5. 閘門指令加 `tests/test_t07_hygiene.py`。
