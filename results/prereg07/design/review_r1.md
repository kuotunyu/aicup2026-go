# PREREG_07 rev 1 草稿：r1 審查摘要和逐點回應

- 對象：`docs/PREREG_07_HUMANSL.md` rev 1 草稿和 `results/prereg07/design/spec.md`（10/4 起草）。
- 兩份審查：審查 1（科學與外洩）、審查 2（程序與測試日）。兩份都沒有 blocker；major 共 10 個（4＋6），minor 共 17 個（8＋9）。
- 處理原則：major 全部修；minor 只要便宜就修（這次全部修了，做法和審查建議不同的寫在最後一節）。
- 修訂時另外核對過的事實（只讀檔、輕量 CPU，沒改追蹤檔、沒 git 寫入、沒開 `data/local`、沒跑 `score.py`、沒用 GPU）：
  - `data/prereg07/pilot/mono1.log`：10/4 19:00:04 STOP、19:00:07 `mono rc=0`、19:00:52 `pilot_fuse.py` 跑完，印出 mono 覆蓋 11,825／120,544 題、差 +0.01387、CI [−0.00175, +0.03099]、黑 +0.0102、白 +0.0186。`pilot_fuse_report.json` 現在和 `56f46de` 相同（git 乾淨）。
  - `mono1.sh` 不管 `pilot_run` 的結束碼，一律接著跑 `pilot_run.py status` 和 `pilot_fuse.py`。10/4 19:04:46 重新啟動，19:45 時 15／307 塊、0 盤失敗，`eta_min` 892 → 約 10/5 10:40 跑完並自動 fuse。
  - `pilot_fuse.py` 開頭（802eaea 起沒改）寫「Gate (task text; pilot only, not a registration)」；README「還沒決定的事」第 268 行寫 mono 全跑或部分未定；`katago/REPORT.md` 也列為開放問題。
  - 試跑 Q02＋HS 的每折權重（順序 OP、O4、SWA、SWB、HS）：第 1、2、5 折 w_SWB＝0.0；第 3 折 w_HS＝4.0、第 4 折 w_O4＝4.0，a＝−0.945／−1.210。
  - `tools/official_log.py` 第 943–970 行：`final_rule_t04` 把任何不是 04 的競賽二紀錄記成 chosen R20，有符合的 04 時要求 `--not-04-reason`。`t04_status` 在最終不是 04 又沒有涵蓋的修訂時判違反；`t04_public_lines` 把非 04 的紀錄都叫「R20／R10」。
  - `results/prereg05/explore/u1/u1_lib.py` 的 `_forbidden` 用子字串 `"data" + os.sep + "local"` 比對 normcase 後的路徑，`...\appdata\local\temp` 也會命中。`hs_lib` 的是逐段比對，沒有這個問題。
  - `tests/conftest.py`：每個測試模組收集完就把 8 個函式還原、清掉 `_FLAGS` 裡的旗標；`_c10_hs_guard`、`_c10_u1_guard` 不在 `_FLAGS`，所以一個在模組層級 import `hs_lib` 的測試模組，收集後自己的防護就悄悄失效（旗標還在，`install_guard()` 變成不做事），測防護的測試會在單一行程裡失敗。結論和審查 2 一樣，機制略有不同。
  - `pilot_run.py` 第 318 行用 `mp.get_context("spawn").Pool`；`mono1.sh` 靠 `export PYTHONDONTWRITEBYTECODE=1` 才沒有寫出 `__pycache__`。
  - `t04_rank.fuse_set` 用 `rf.read_question_meta(refuse_label_path(qfile))` 讀題目檔；seam_val 的題目檔就是 `data/local/rank_val_questions.csv`。
  - `val_o4.npz` 有 `n_moves`[19156,2,3]、`moves_parsed`、`replay_failed`；test04 的 `o4_<組>.npz` 也有（`t04_rank.py` 第 1187 行）；`inner_o4.npz` 有 `n_moves`[73556,2,3]、`anomalies`。
  - **手數交叉核對試算**：bank 前 5 塊（1,000 盤）的 HS `counts[:,0]` 對 `inner_o4` `n_moves[ref,0,:]` 合計、HS `n_moves` 對 `n_moves[ref]` 合計，0 不符（`inner_o4` 的 anomalies 合計 0）。

## 審查 1：科學與外洩

### [major] 1. 「G4 的規則試跑前就寫死、照抄沒改」說得太滿

- **回應：接受。** 第 0 節改寫成「試跑前寫下的，和它的限度」：引用「not a registration」和 README 的開放問題；寫明 G1、G2 過了、第一次部分結果下界 −0.0025 之後，10/4 18:53 才決定全跑；照 802eaea 的部分覆蓋走法，兩次中途結果都不會過；全覆蓋是本登記第一次寫成登記。第 2.5 節「G4 的規則和由來」刪掉「不因看過的部分數字改」，理由 4 不再用中途結果的標準誤。第 8 節同步。
- 新題子集（108,719 題）：加成**只記錄**（第 2.4、2.5 節；spec 4.3 第 5 點）。不當閘門的理由見最後一節。

### [major] 2. 第二次中途看 mono 沒揭露，而且之後每次 STOP 都會再看一次

- **回應：接受。** 已核對 `mono1.log`。修訂紀錄、第 0 節（新表列兩次中途結果）、第 1.6 節、第 8 節都寫進去，並寫明報告檔已還原、git 裡維持 `56f46de` 版到第 9 節第 3 步。
- 第 2.5 節新增「不再中途看 mono」：fuse 只在 307／307 而且 rev 1 已 commit 後跑；要讓出 GPU 或重開機時，先結束 `mono1.sh` 的 bash 行程、確認 `run_log.jsonl` 還在增加，才放 STOP；續跑用沒有 fuse 的 `mono2.sh`；萬一又看到寫進 rev 2。spec 第 0 節另規定：GPU 工作的啟動腳本只跑 GPU 那一步，不在後面接任何算分數或判定的程式。

### [major] 3. rev 1 在 mono 跑完前 commit 只寫「最好」

- **回應：接受。** 第 9 節第 1 步改成硬規則：rev 1 的 commit 要早於 mono 全覆蓋的結果第一次被算出。兩種做法：在最後一塊之前 commit（估 10/5 10:40），或先結束 `mono1.sh` 的 bash 行程讓 fuse 改成手動。RESULTS 記 rev 1 的 commit 時間和 fuse 的時間（spec 第 5 節）。時程表改成「10/04 夜」。

### [major] 4. 查 bug 重跑的條款不對稱

- **回應：接受，照審查的版本。** 第 3 節：只在 (1) Q02H < 0.57 而且指出具體程式錯誤，或 (2) 固定範圍、不論採不採用都做的獨立核對找到影響 Q02H 預測的程式錯誤時重跑；採用之後才找到的也要重跑；最多一次，判定就是最終；用掉第 4 次。
- 另外把核對往前移一步：apply 之後、打分之前先做一次同範圍的「打分前核對」（第 2.7 節，`prescore_audit.json`，不看分數），它是 `score_allowed` 的條件之一，讓程式錯誤盡量在打分前抓到、不占打分機會。
- `tune`：「閘門只判一次」，`--rerun` 只准在指出程式錯誤的修訂 commit 之後，每次結果都寫進 RESULTS（第 2.5 節，spec 4.3 第 1 點）。

### [minor] 5. HS 方向檢查幾乎沒有作用，執色的理由也寫錯

- **回應：接受，兩項都做。**
  - (a) 門檻改成 ρ ≥ max(0.5 × 參照, 單側 p＝0.001 的臨界值)：驗證題用 inner 評估假題的 ρ_eval 和 0.078，測試日用單色假題的 ρ_mono 和 0.155；參照值由 `tune` 寫進 `frozen_config.json`（第 2.6、2.7、5.2 節，spec 4.6）。
  - (b) 新增逐盤、不看標籤的手數交叉核對（HS 對 04 的 O4 重播：總手數＝`moves_parsed`、目標手數＝`n_moves[:,0,:]` 合計），用在驗證題 `score_allowed`、測試日 `apply`，inner 在 `manifest` 先做一次。已在 bank 1,000 盤試算 0 不符。
  - 「執色由 04 的 Δκ 把關」的錯誤說法刪掉，改成由手數交叉核對把關（spec 4.6）。

### [minor] 6. 標準誤和採用機率的表偏樂觀

- **回應：接受。** 第 3 節改用 0.0062–0.0065，表改成 63／59%、40／37%、21／19%（重算過）；第 0 節改成「約 6 成；打折約 2–4 成」；spec 風險表同步。

### [minor] 7. 打到界限的權重沒寫全

- **回應：接受。** 第 2.4 節寫明第 1、2、5 折 w_SWB＝0、第 3、4 折各有一個＝4；只記錄重擬合的參數、打到的界限、樣本內和 OOF 分數並列（spec 4.3 第 5 點）。第 7、8 節同步。

### [minor] 8. 預算重分配沒寫明

- **回應：接受。** 第 3 節「預算」和第 8 節寫明 07 用的是 PREREG_02 第 3 節分給 03、03 沒用到的那一次（10/2、試跑前決定）。

### [minor] 9. `val` 的非正式模式可能在凍結前讀驗證題

- **回應：接受。** 第 1.7 節和 spec 3.2：只有 real 模式讀 `data/local`；smoke 只接受 `data/local` 以外、用 inner 棋做的假題目檔；任何讀 `data/local` 的模式都要求已 commit 的 `frozen_config.json`。

### [minor] 10. mono 有失敗盤時，selfcheck 和 G4 的定義互相矛盾

- **回應：接受。** selfcheck 的 mono 用試跑的邏輯、在試跑涵蓋的題上比（第 2.3 節，spec 4.2）；G4 用遮罩版、在全部題上判；「全部的題」改成「307 塊都在、manifest 已 commit」（第 2.5 節）。

### [minor] 11. GPU 接縫只核對曲線，沒核對 lp_HS

- **回應：接受。** GPU 接縫加「每盤 lp_HS 最大差 ≤ 0.02」（第 5.2 節，spec 2、7.4）。演練的 4090 實測若 lp_HS 最大差 > 0.005，在測試日的修訂裡（拿到測試資料之前）重新檢討容許差（spec 第 9 節）。

### [minor] 12. 其他揭露

- **回應：接受。** 第 0、2.5 節寫明 G1–G3 已知、G4 是唯一的未知；第 8 節加「同一個驗證集的第三次依序使用（H2R、H4R、H7R），A 側的 Q02 也是在這個驗證集上採用的」。

## 審查 2：程序與測試日

### [major] 1. `official_log`：只繞過了 prepare，`final` 07 永遠失敗

- **回應：接受。** 已核對第 943–970 行。spec 第 8 節列出插入的行：`final_rule_t04`、`prepare_rule_t04` 開頭各加一行讓 07 跳過；新增 `final_rule_t07`、`prepare_rule_t07`，判定記在 `final_choice.rank.prereg07_5_4`；`t04_status` 插一行讓最終是符合的 07 時不判違反；07 和 04 都符合時選 R20 要兩個 reason。登記第 5.2、5.4 節同步。正面演練 (a)–(c) 加進 spec 第 9 節。

### [major] 2. 06 的指紋和 07 自己要改的檔衝突，順序也不對

- **回應：接受，做法稍改。** 登記第 5.1 節寫死順序：06 的 K5' 其他待辦先做完 → 做 06 的指紋（`official_log.py`、`.gitattributes`、`.gitignore` 放 `insert_only`，照 `p04_path_fingerprint.json`）→ tag `p06-testday-pre07` → 分支 `p07-testday` 從那個 commit 開。
- 和審查建議不同：`tests/conftest.py` 照 TESTDAY_06 釘 SHA，因為 07 改成完全不動它（見下一點）。所以 07 在 main 的第 2 步不碰任何 06 的檔，06 的指紋可以等到第 10 步之前再做，不用趕在 10/5。

### [major] 3. import 時的防護會讓單一行程的測試和 TEMP 出問題

- **回應：接受問題，做法不同。** 已核對 `u1_lib._forbidden` 的子字串比對和 conftest 的還原機制（見開頭）。改成：
  - 只有 `check`、`selfcheck`、`tune` import `u1_lib`（延遲 import）；`apply`、`p07_hs.py`、所有 `t07_rank.py` 子指令都不 import 它，也不呼叫會帶進它的 `pilot_fuse.fit_mapping`。
  - 所有暫存寫在 `data/prereg07/tmp/`、`data/test07/tmp/`、`data/test07_rehearsal/tmp/`，一律不寫 `%TEMP%`；能在記憶體比的就在記憶體比。
  - 測試一律不在 pytest 的行程裡 import `hs_lib`、`u1_lib`、`pilot_*`，用子行程跑工具的 smoke 模式；每個 p07／t07 測試模組有自己的 autouse fixture 檢查兩個旗標沒有出現；hygiene 用靜態檢查擋模組層級的 import。
  - 第 2 步 commit 前，main 上用單一行程跑 `PY -m pytest -q tests` 全綠（排在 mono 跑完之後）。
- 沒照審查改 `conftest.py` 的理由見最後一節。

### [major] 4. 測試日的 GPU 接縫可能沿用 10 月的舊塊

- **回應：接受。** (a) real 模式 `hs` 拒絕資料夾裡 `created_utc` 早於 R20 第一次上傳的塊（2），請操作者改名封存；(b) `apply` 核對三組每一塊都晚於那個時間、環境相同；(c) 演練一律寫 `results/test07_rehearsal/`、`data/test07_rehearsal/`（`.gitignore` 加一行）；(d) 負面演練放一個舊 seam 塊，要被拒絕。登記第 5.2、5.4 節，spec 3.3、7.3、7.4、9。

### [major] 5. GPU 接縫的參照只有一份沒進 git 的檔

- **回應：接受。** `p07_hs.py val` 另外 commit `results/prereg07/hs/val_seam_moves.npz`（前 400 盤的逐手 log p，float32，約 2–3 MB）；接縫改成對 `val_hs.npz` 前 400 列和這個檔比，不再需要 `data/prereg07/val/` 的 chunk。權重做同 SHA 備份 `data/katago_backup/`，順序寫死：原檔 → 備份 → 使用者同意的重新下載（SHA 要相同）→ 停下（登記第 1.2、1.7、5.2 節）。

### [major] 6. 測試集的 HS 路徑在 11/04 前從沒跑過

- **回應：接受。** 演練加兩項 4090 實測：在 from-val 假樹上真的跑 `hs --set public --max-games 400`（演練限定），對應的驗證盤比 `failed`、`counts`、`n_moves` 相同、每盤平均 log p 差 ≤ 0.002；在 NORMALIZED 的假樹上再跑一次（spec 第 9 節）。

### [minor] 7. 「不寫 PREREG_04 rev 4」的理由寫錯

- **回應：接受。** 第 5.5 節改成：照 PREREG_06 5.4 的前例；`docs/TESTDAY_04.md` 寫「下一版是 rev 4」，現在寫會讓那句過時、04 真要修 bug 時得用 rev 5；並註明 10 月的修訂本來就因為時間規則不能當翻盤依據。

### [minor] 8. 有 07 紀錄後，04 的幾行會出錯；同名警告會誤報

- **回應：接受。** spec 第 8 節：`t04_public_lines` 插行跳過 07；有符合的 07 時把 04 的提示換成 07；07 的同名警告只數 public 題的規則 B 不同（`t07_apply_report.json` 記 public 題和全部的不同題數）。登記第 5.4 節同步。

### [minor] 9. spec 第 0 節的 data/local 清單漏了 `t07 apply`

- **回應：接受。** 清單加上 `t07_rank.py apply`（`fuse_set` 讀 seam_val 題目 meta）和 `t07_fake_tests.py from-val`，並規定讀完才 import `pilot_fuse`／`hs_lib`（spec 0、7.4）。

### [minor] 10. `sys.dont_write_bytecode` 不會傳到 spawn 的子行程

- **回應：接受。** spec 第 0 節：import 前同時設 `os.environ['PYTHONDONTWRITEBYTECODE']='1'`；背景腳本 `export`；登記第 1.7、5.2 節同步。

### [minor] 11. `official_log` 加行的衛生條件沒寫

- **回應：接受。** spec 第 8 節開頭：只用標準函式庫和 `check_submission`、不出現標籤檔字樣、不重算 numpy／scipy 統計量（方向檢查只核對報告裡的數字和門檻）。spec 本身也不寫標籤檔的檔名。

### [minor] 12. 打分紀錄帳只准 2 列，和允許查 bug 重跑矛盾

- **回應：接受。** spec 第 6 節：2 列，或 4 列（後 2 列 note `prereg07 H7R rerun (PREREG_07 rev N)`，rev N 已 commit 且早於那兩列）。登記第 3 節同步。

### [minor] 13. `apply` 綁 04 的上傳，可能拖過 K6-07

- **回應：接受。** `t07 apply` 只要求 04 的 `t04_apply_report.json` 結束碼 0、已 commit；「R20 和 04 都已上傳」留在 `official_log.py prepare` 07 的前提；時程寫 04 最晚 11/18 上傳（登記第 5.2、5.4 節，spec 7.2 第 9 條）。

### [minor] 14. 本機接縫超出容許差被當成漂移

- **回應：接受。** 結構不符、或 04 的輸入逐位元相同而 07 不同 → 4；只有容許差超出、而 04 的接縫和 Q02 對 04 都過 → 3（登記的「07 不用」）（登記第 5.2 節，spec 7.4 第 5 條）。

### [minor] 15. GPU 工作的操作細節沒寫

- **回應：接受。** (a) 驗證題和測試日的 HS 都用獨立背景程序（spec 0、3.2、7.3）；(b) 驅動版本記進 `frozen_config.json` 和每塊，只記錄、由 GPU 接縫判；暫停 Windows 驅動自動更新寫成給使用者的建議（登記第 5.3 節），交主 session 決定要不要請使用者做；(c) `status` 印殘留的 STOP 和舊塊；(d) 每組 `<組>_hs.npz` 寫好就 commit；(e) 11/04–11/07 夜間先在 `PORTFOLIO_MAP.md` 登記 4090（登記第 5.3 節，TESTDAY_07 第 1 步）。

## 沒照審查原樣做的地方

- **conftest**（審查 2 major 3 建議在 `tests/conftest.py` 加 `hs_lib`、`u1_lib`）：改成「測試不在 pytest 行程裡 import 它們」。理由：(1) TESTDAY_06 K5' 要 06 的指紋釘 `conftest.py` 的 SHA，07 不動它就不用改 06 的待辦；(2) 07 在 main 的第 2 步因此不碰任何 06 的檔，不用先趕完 06 的 K5'（那要在 mono 占著 CPU 時跑 `C10_SLOW` 的重算）；(3) 加進 conftest 之後，模組層級 import `u1_lib` 的測試仍會被它的子字串防護擋住 pytest 的 tmp_path，還是要另外處理暫存。代價：用到釘住檔的測試要寫成子行程，跑得慢一點。
- **新題子集的 G4**（審查 1 major 1 建議可選做閘門）：只記錄。新題子集占全部的 90%，和全體 G4 幾乎同一件事；當閘門只多一條失敗路徑，對防止「看結果決定」幾乎沒有幫助（全跑的決定本來就比部分覆蓋更嚴）。
- **HS 方向檢查的門檻**（審查 1 minor 5 給兩個選項）：兩個一起用，取較嚴的那個；真正抓槽位和執色的是手數交叉核對。
- **查 bug 重跑**（審查 1 major 4 和替代方案「07 不重跑」）：採用審查的對稱版本，另加打分前核對。PREREG_02 第 3 節本來就把「查 bug 重跑」列為第 4 次的用途；完全不准重跑時，一個打分後才發現的程式錯誤會讓已採用或未採用的判定都建立在錯的預測上。

## 留給主 session 決定的

- 現在就要決定：要不要立刻結束 `mono1.sh` 的 bash 行程（讓 `pilot_run` 照常跑完、不自動 fuse），還是只靠在 10/5 10:40 前 commit rev 1。
- 06 的指紋改成 `official_log.py`、`.gitattributes`、`.gitignore` 放 `insert_only`（TESTDAY_06 K5' 原本寫釘 SHA），而且排在 06 的 K5' 收尾之後、07 分支之前。
- 查 bug 重跑採對稱版本（用掉第 4 次），還是 07 完全不重跑。
- 要不要請使用者 10/6–11/19 暫停 Windows 自動更新顯示卡驅動（系統設定，只能使用者本人做；選做）。
