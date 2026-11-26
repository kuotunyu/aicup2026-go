# test04 設計調查 U4：登記、程序、上傳紀錄、文件

- 角色：U4（登記、程序、上傳紀錄、文件）。只讀 repo；本檔是唯一寫入的檔。
- 分支：`p04-testday`（從 main `6ec5e68` 開出），工作樹乾淨。
- 讀過：`docs/PREREG_04_OPPONENT.md`（全文）、`docs/PREREG_03_RECIPE_V2.md` 2.2／5／11、`docs/PREREG_02_OPENING_FUSION.md` 第 7–12 節、`docs/TESTDAY.md`、`docs/COLAB_RUN.md`「測試集（test02）」與「事前登記 03」、`docs/VALIDATION.md`、`docs/DATA.md` 官方資料說明、`tools/official_log.py`、`tests/test_p04_hygiene.py`、`tests/test_t02_hygiene.py`、`tests/test_t02_infra.py`（OfficialLog）、`tools/t02_rank.py`（apply 與接縫）、`tools/p04_rank_fuse.py`（κ、apply）、`tools/p03_rank_fuse.py`（κ）、`tools/p04_opp.py`（val）、`results/prereg04/{RESULTS.md, design/plan.md 第 7–8 節, design/r20_path_fingerprint.json}`、`results/test02/design/spec.md` 第 6 節、repo 的 `CLAUDE.md`。
- 沒做的事：沒開、沒讀、沒雜湊任何答案檔；沒跑 `tools/score.py`；沒用 GPU；沒 commit／stash／checkout。
- `results/official_scores.json` 目前**不存在**（還沒有任何上傳）；它的格式照 `tools/official_log.py:156-158`（`empty_log`）和 `:400-407`（每筆上傳）。
- `tools/official_log.py` 現在和 tag `r20-testday-pre04` 的版本相同（LF 摺疊後 SHA-256 `1c4bcbe9…`＝指紋），所以下文的行號就是 tag 版的行號。

## 0. 結論（先看這裡）

1. 已登記的 04 測試日要求共 34 條（第 1 節）。最硬的是：R20 路徑 39 個檔一個位元組都不能動（`official_log.py` 只准附加）、K5 11/03 23:59 前合併並演練通過、K6 11/16 23:59 前有一次成功的正式執行已 commit、最後一版只依 5.3 的五個條件決定。
2. rev 3 要補 26 個缺口（第 2 節）。最影響判定的六個：
   - 官方單色題上的 Δκ 怎麼算（每組、每個來源、用哪些盤）。
   - 「重播失敗 > 0.5%」的單位：盤（slot）、分組算、分母是題目檔的全部盤。另外要定失敗盤在 E_O4 和 κ 裡怎麼處理。
   - 順序：test02 全部做完、R20 的 U1 記成 uploaded 之後，才跑任何 04。先跑 CPU 的 `t04 opp`，O4 不能用就不開 Colab。
   - test04 的重跑一次，重跑預算和 test02 分開算。Drive 改名 `rank_failed_N`，本機封存 `results/test04/failed/rank_N/`。
   - K6 的「成功的正式執行」定義成「全部條件都過、報告 commit 時間 ≤ K6」。
   - 最後一版由 `official_log.py final` 依 5.3 強制判定。
3. `official_log.py` 可以只靠插入新行完成，不改任何舊行（第 3 節），共 12 個插入點（A–L，其中 J、K 分成兩三段）。
   - 04 的檔用路徑認：`results/test04/rank/` 底下的 `submission_rank.csv`，綁在自己的報告上。
   - R20 的邏輯一行都不動，只有 `status` 多印一行 K6 倒數。
   - `plan.md` 第 7 節寫的「`RANK_COMBOS` 加項」**不要照做**：改那一行會讓附加檢查失敗；用另一行重新定義，又會改變 R20 的 `method_for` 行為。
4. 使用者步驟另起 `docs/COLAB_RUN.md`「測試集 04（test04）」一節；Claude 步驟另起 `docs/TESTDAY_04.md`，因為 `docs/TESTDAY.md` 在指紋裡。兩份的大綱在第 4 節。
5. 最大的風險（第 5 節）：
   - test04 notebook 從沒在真 GPU 上跑過。
   - 03 的權重只存在 AI Pro 帳號的 Drive。
   - `counts_fit.npz` 不在 git 裡；遺失後重算至少要 7 個 worker。
   - 官方定義變動（D5）和 R20 指紋互相衝突。
   - 兩個都叫 `submission_rank.csv` 的檔，上傳時容易拿錯。

---

## 1. 已登記、適用於 04 測試日路徑的要求（逐條附出處）

「強制」欄：T＝工具會拒絕、H＝hygiene 測試、P＝靠程序（文件）、U＝使用者。

| # | 要求 | 出處 | 強制 |
|---|---|---|---|
| 1 | **K5 11/03（二）23:59**：測試日擴充（第 5 節）和它的修訂已 commit、已合併回 main、本機假測試演練通過；沒趕上這一季不用 04，也不拿額外的打分機會補 | PREREG_04:192-197；RESULTS.md:108 | P |
| 2 | **K6 11/16（一）23:59**：04 在官方測試集至少有一次成功的正式執行，已 commit | PREREG_04:198 | P（建議 T，見第 3 節） |
| 3 | 測試日細節要在 K5 前另寫修訂（＝rev 3），只能加新的一列和新的一節 | PREREG_04:200, 281；任務規則 | P |
| 4 | 指紋：`results/prereg04/design/r20_path_fingerprint.json` 記 39 個檔的 SHA-256（CRLF 摺成 LF），tag `r20-testday-pre04` | PREREG_04:204-205 | H |
| 5 | 防誤改測試 `tests/test_p04_hygiene.py`：指紋本身釘兩次（常數 `72fa6c12…` 和 tag 的 blob），除了 `official_log.py` 以外任何一個檔變了就失敗 | PREREG_04:206, 356-358；test_p04_hygiene.py:29, 66-99 | H |
| 6 | `official_log.py` 只准附加：tag 版的每一行都要照原順序出現在新檔（LF 比對）；「新方法名稱、認得 04 的檔」，R20 的分支邏輯不動 | PREREG_04:211, 359-361；test_p04_hygiene.py:60-63, 101-114 | H |
| 7 | 凍結的結果資料夾（prereg01、02、03、`prereg04/design`）不能有改動、新檔或 bytecode | PREREG_04:362-364；test_p04_hygiene.py:32, 123-140 | H |
| 8 | 分支隔離：04 的測試日程式全部在 `p04-testday`；合併前要全部通過並經獨立審查：test02 的全部測試、`test_p02_*`、`C10_SLOW=2` 慢速演練、用 `t02_fake_tests.py` 做的假測試演練（R20 的 `submission_rank.csv` 和合併前逐位元組相同） | PREREG_04:207-210 | P |
| 9 | `t04_rank.py opp`：在 CPU 重播 `seam_val`、public、private 三組的 SGF；`seam_val` 的 O4 要和已 commit 的驗證 O4 逐位元組相同 | PREREG_04:215-216 | T |
| 10 | 任一組重播失敗 > 0.5%：04 不用 | PREREG_04:217, 237 | T |
| 11 | 重播失敗的定義：沒有解析到任何一手，或有一手下在已有子的點（重播在那裡停下） | PREREG_04:307；p04_opp.py:1294-1306 | T |
| 12 | 因為選中含 SW 的 Q02，要做 `colab/build_test04_rank_nb.py` → `test04_rank.ipynb` | PREREG_04:218；RESULTS.md:109 | P |
| 13 | notebook 樣式照 test02：同一份 `C10_test_v1.zip`、核對 MANIFEST、上一次接縫沒過就停 | PREREG_04:219；plan.md:376-379 | T（notebook） |
| 14 | 用 03 兩臂的 `final.pth`（SHA 釘死：A `790b767c5425…50ac`、B `8e88f6ec6cba…9a42`），算正確執色（只為了 κ）和執色對調兩種 logits | PREREG_04:90, 220；prereg03 RESULTS.md:34-36 | T（notebook） |
| 15 | notebook 接縫：`seam_val` 要重現已 commit 的驗證 dump，同型 GPU 差 ≤ 1e-4、不同型 ≤ 1e-3 | PREREG_04:221 | T（notebook） |
| 16 | 只在 AI Pro 帳號跑；推估約 5 分鐘、0.7 單元 | PREREG_04:222 | U／P |
| 17 | `t04_rank.py apply` 的前提：test02 競賽二已完成並 commit（C5 dump、`opening_test.npz`）；T1 是 PASS 或 NORMALIZED；test04 已完成、接縫通過 | PREREG_04:224 | T |
| 18 | 本機接縫：要重現 `results/prereg04/rank/rank_val_probs.csv`，規則 B 的判定最多差 4 題（GPU 容許差同 15） | PREREG_04:225；plan.md:399 | T |
| 19 | 執色檢查：每個對手方來源（O4、SWA、SWB）在 public、private **各自**的 Δκ 都要 > 0，否則 04 不用 | PREREG_04:226, 236；RESULTS.md:114 | T |
| 20 | 輸出 `results/test04/rank/submission_rank.csv`，跑 `check_submission --official`；只寫入 `results/test04/` | PREREG_04:227 | T |
| 21 | 11/04 照 `docs/TESTDAY.md` 先做完 test02，R20 的第一次上傳記進 `official_scores.json`，之後才開始 04；04 的上傳目標 11/05–11/07 | PREREG_04:231 | P（建議 T） |
| 22 | 最後一次上傳（11/23 前）交 04 的五個條件，全部要成立：H4R 採用；K6 前有一次成功的正式執行且已 commit；T1 是 PASS 或 NORMALIZED；所有 Δκ > 0；重播失敗 ≤ 0.5% | PREREG_04:232-237 | P（建議 T：`final`） |
| 23 | 任一條不成立就交 R20，不交沒在驗證集打過分的組合（Q01 等）。已經傳過 04、之後才發現不能用：再傳一次 R20，讓它成為最後一次 | PREREG_04:238, 255 | P |
| 24 | FAIL_OPENING：04 不上場，照 test02 用 R10；FAIL_STRUCT：照 test02 停下 | PREREG_04:165, 239；frozen_config.json `fail_opening` | T |
| 25 | public 分數比上一版低 0.05 以上時，只追查錯誤，不據此換方法 | PREREG_04:240；VALIDATION:98-100 | T（`public` 標 investigate）＋P |
| 26 | 每次上傳前，使用者都要在對話裡對那一份檔明確說 OK | PREREG_04:241；TESTDAY:15；CLAUDE.md:58 | P／U |
| 27 | 合併公式：E_j＝每盤可解析的棋先取 log_softmax(10 維)，再取平均；一盤都沒有就記 0；盤數只經由 (n/12)^γ 進來 | PREREG_04:98-102 | T |
| 28 | SW 的語意：原方向、整題執色對調；A0／B0＝同一模型、原方向、正確執色（d4 的第 0 個對稱），只當 κ 的參照 | PREREG_04:80, 91；PREREG_03:48 | T |
| 29 | κ、Δκ 的定義同 PREREG_03 2.2 第 4 條；04 的 Δκ＝κ(目標方參照) − κ(對手方來源)（O4 對 T4、SWA 對 A0、SWB 對 B0） | PREREG_04:119-121；PREREG_03:70 | T |
| 30 | 官方題目上的 κ：每題單色、整題執色對調；另外分黑、白題記錄（只記錄） | PREREG_03:209-215 | T |
| 31 | 測試日的接縫與重跑：沒過的那次輸出一律不用、不上傳；換一個執行階段重跑一次；還是沒過就停下查，要放寬容許差另寫修訂 | PREREG_02:244-252, 332-333, 351；rev 4:235 | P＋T |
| 32 | 上傳規則：第一次上傳在檢查和接縫都過之後；每次記進 `official_scores.json`；最後一次上傳只依本機驗證主要指標和登記的採用條件決定，11/23 23:59 前完成；之後的上傳只有三種：(a) 修 bug、(b) 之後的登記採用的方法、(c) 重傳最終選擇 | PREREG_02:306-311；spec.md:758-786；TESTDAY:223-237 | T／P |
| 33 | 驗證玩家的棋不得用於統計或調參；測試集預測不人工修改；答案檔不上 Colab | VALIDATION:22, 97, 110 | T／P |
| 34 | 釘死的數字：fit 計數 SHA `f6c6debe…`；比率表 `tables_sha256` `f380e0d7…`（`frozen_config.json` 的 `source_sha256.O4.tables_sha256`）；SW 驗證 dump 的 SHA（PREREG_04:83-88） | PREREG_04:71-74, 345-347 | T |

---

## 2. rev 3 在 K5 前要補的缺口，以及提案

原則：每條提案只把已登記的內容具體化，或比登記更嚴。來源、格子、比率、組合、參數、閘門、H4R、截止點都不變。凍結的 `frozen_config.json` 參數照用，不重擬合。

### G1. 官方單色題上的 Δκ（每組、每個來源）

- **缺口**：5.2 只寫「public、private 各自的 Δκ > 0」。沒寫的有：用哪些盤、單位是題還是子題、失敗盤怎麼處理、要點估計還是信賴區間、題目裡萬一混了兩種執色怎麼辦。
- **提案**（每組 s ∈ {public, private}、每個來源 src ∈ {O4, SWA, SWB} 各算一次）：
  1. **可用盤**：
     - SWA、SWB 及參照 A0、B0：`val_keys.npz` 的 `parsed` 為真的盤。test02 各組的 `val_keys.npz` 要和 test04 各組的 key 陣列（`question_id`、`slot`、`color`、`parsed`）逐列相同，不同就停。
     - O4 及參照 T4：`parsed` 為真，而且 `replay_failed` 為否。
  2. **子題**：沿用 `p3.val_colour_subquestions(vk)`（`tools/p03_rank_fuse.py:732-741`）。每題的可用盤依題目檔（正規化副本）的 `color_i` 分成黑、白，≥ 2 盤的才算一個子題。官方每題單色，所以一題就是一個子題；萬一混了兩種執色，也照這個規則拆開，不另外處理（T1.3 的 `official_colour_design_record_only` 照記）。
  3. **κ**：用 `p3.kappa_questions(lp, groups)`（`p03_rank_fuse.py:541-558`）。lp 是每盤 10 維向量的 log_softmax（float64）：
     - O4／T4：`log_softmax(o4)`、`log_softmax(t4)`，同 `p04_rank_fuse.py:1767-1768`。
     - SW：`log_softmax(對調的 logits)`。
     - A0／B0：`log_softmax(正確執色、第 0 個對稱的 logits)`。
     - 直接呼叫 `p04_rank_fuse.val_side_kappa(vk_s, lp_ref, lp_src)`（`:916-934`）即可，vk_s 是那一組的 key dict；O4 用的那份要把 `parsed` 換成 `parsed & ~replay_failed`。
  4. **Δκ_s(src)**：所有子題上 κ(參照) − κ(來源) 的**簡單平均**（點估計）。
  5. **判定**：6 個值（2 組 × 3 個來源）都要 **> 0**（嚴格大於；None 或 NaN 算不過），不看信賴區間，同 PREREG_03 11 (c)。任一個不過，04 不用。這是題目內容本身的性質，重跑不會改變，所以**不觸發重跑**（見 G6）。
  6. **只記錄**：
     - 黑題、白題各自的 Δκ；各 k（5／10／15／20）的平均。
     - `question_bootstrap_ci95_recorded_only`（`p3.question_boot_mean`，`:744-752`）。
     - 子題數，以及因可用盤 < 2 而沒算到的題數。
     - `seam_val` 上的同一組 Δκ（應該等於 check／apply 已記的 SWA +0.0363、SWB +0.0344、O4 +0.1050；O4 要逐位相同，SW 在 GPU 容許差內），不符時停下查。
  7. 某一組沒有任何子題（官方結構下不會發生）：Δκ＝None，判不過，04 不用。
- **不超出科學的理由**：定義就是 PREREG_03 2.2 第 4 條和 11 (b)(c) 用在驗證子題上的那一套，只是換了題目檔；不讀任何標籤。

### G2. 「重播失敗 > 0.5%」的單位、分母、分組，以及失敗盤的處理

- **缺口**：沒寫是盤還是題、分母是什麼、三組分開算還是合計、失敗的盤在 E_O4 和 κ 裡怎麼處理。驗證集失敗是 0，所以程式從沒碰過這條路。
- **提案**：
  1. **單位**：盤（slot）。失敗的定義照 rev 2 10.1 d（`p04_opp.val_scores`，`p04_opp.py:1294-1306`）。
  2. **分母**：該組題目檔（正規化副本）的全部盤數（每題 slot 1..`num_games`）。官方每組 100 × (5＋10＋15＋20)＝5,000 盤，0.5%＝25 盤。
  3. **分組**：`seam_val`、public、private 各自算。判定條件是 failures / slots **> 0.005**：剛好 0.5% 算過，嚴格大於才不過。
  4. **`seam_val`**：一定要 0，而且和已 commit 的驗證 O4 逐位元組相同（G3）。
  5. **失敗盤在 E_O4 裡**：照 2.1「每盤可解析的棋」，E_O4 只平均「C5 有解析、而且 O4 沒有重播失敗」的盤；一題一盤都沒有就記 0。盤數 n（溫度用）仍是 C5 的可解析盤數，SWA／SWB 的 E 仍用 C5 的可解析盤。受影響的題數記進報告。
     - 驗證集失敗是 0，兩種寫法在驗證集上完全相同，所以 `seam_val` 的接縫照樣逐位相同。
     - 這是 2.1 原文的直接讀法，不是新的選擇。
  6. **κ**：失敗的盤不進 O4、T4 的 κ（G1 第 1 點）。
  7. **只記錄**：各組 `anomalies`、`slots_without_moves`、`slots_without_opponent_moves`、含失敗盤的題數。
- **理由**：
  - rev 2 10.1 d 已定義什麼是失敗，而且以盤為單位（`val_report.json` 的 `replay_failures` 就是盤數）；這裡只補分母和分組。
  - 注意：以盤為單位比以題為單位**寬鬆**。以題為單位時，官方 400 題的 0.5% 只能容許 2 題有失敗盤；以盤為單位則可容許 25 盤，可能分散在 25 題。
  - 仍建議以盤為單位，因為這是登記原文最直接的讀法；而且 E_O4 只平均沒失敗的盤，一題裡少數幾盤失敗的影響有限。
  - 審查時如果認為應該用較嚴的讀法，可以改成「盤和題兩個比例都 ≤ 0.5%」。這只會讓 04 更少上場，不超出登記。

### G3. 「`seam_val` 的 O4 逐位元組相同」是比什麼

- **缺口**：比 npz 檔的位元組，還是比陣列內容？`save_npz` 是 `np.savez_compressed`（`p04_opp.py:824-828`），zlib 版本不同時，陣列一樣、檔案位元組也可能不同。
- **提案**：
  - 判定用陣列：`question_id`、`slot`、`color`、`o4`、`t4`、`n_moves`、`anomalies`、`moves_parsed`、`replay_failed` 這 9 個陣列，dtype、shape 和 `tobytes()` 都要和 `results/prereg04/opp/val_o4.npz`（SHA-256 `30d312c0…`）相同。
  - 重算出的比率表 `tables_sha256` 要等於 `f380e0d7…`。
  - npz 容器的 SHA-256 只記錄（預期相同）。
  - 不符就停下查（本機程式或 `.venv` 的 numpy／scipy 版本），**不是**請使用者重跑 Colab 的理由。
- **理由**：「逐位元組」要保證的是 O4 的數值；陣列逐位元組相同就是這件事。

### G4. 順序（test02 → t04 opp → test04 notebook → t04 apply → 上傳）

- **缺口**：5.3 寫「test02 做完、R20 第一次上傳記進 `official_scores.json` 之後才開始 04」。沒寫的有：`opp` 和 notebook 誰先；「記進」是 prepared 還是 uploaded；工具要不要強制。
- **提案**（全部由工具強制）：
  1. **test02 全部做完**：`docs/TESTDAY.md` 第 0–8 步，競賽二的 U1 已上傳，`official_log.py uploaded` 記成 `status: uploaded` 並 commit。
     - 競賽一的進度不影響 04（兩隊分開）。
     - T1 是 FAIL_OPENING：04 整條不做，記進 RESULTS，最終＝R10。
     - T1 是 FAIL_STRUCT：全部停。
  2. **`t04_rank.py opp`**（CPU，約 1–2 分鐘）→ commit。
     - 任一組重播失敗 > 0.5%，或 O4 在 public／private 的 Δκ(T4−O4) ≤ 0：04 不用，**不開 Colab**，最終＝R20。
     - O4 的 Δκ 在這裡先算、先記；`apply` 會照同一個定義再算一次，那一次才是判定。
  3. **test04 notebook**（使用者，AI Pro）→ 收 zip → `t04_rank.py unpack` → commit（任何本機步驟之前）。
  4. **`t04_rank.py apply`** → commit。
  5. **`official_log.py prepare`**（04）→ 上傳 → `uploaded`／`public` → commit。
  6. **`official_log.py final`**（第 G10 條）。
- **工具強制**：`t04_rank.py opp`／`apply` 在 real mode 讀已 commit、乾淨的 `results/official_scores.json`，要有一筆 `comp: rank`、`status: uploaded`、`file: results/test02/rank/submission_rank.csv` 的紀錄；`official_log.py prepare` 對 04 的檔也檢查同一件事（第 3 節 G 插入點）。
- **理由**：先跑 `opp` 只是把登記的 CPU 步驟提前。O4 不能用時 04 一定不上場，這時不開 Colab 可以省下運算單元，也少一次使用者操作；選擇規則沒有變。

### G5. notebook 接縫比什麼、容許差怎麼定

- **缺口**：5.2 只寫「重現已 commit 的驗證 dump」。沒寫比哪個量；本機 Q02 接縫同時用到 C5（test02 的 notebook）和 SW（test04 的 notebook），「同型 GPU」要和誰比也沒寫。
- **提案**：
  - **notebook 接縫**：
    - 每臂（a、b）、每種 logits（正確執色第 0 個對稱、對調）：原方向逐盤 softmax 的每題平均，和已 commit 的 `results/prereg03/rank_{a,b}/dumps/val_p03{a,b}_d4.npz`（取 `[:, 0]`）、`val_p03{a,b}_s0_swap.npz` 算出的同一個量相比，最大差 ≤ tol。這和 test02 的量相同（PREREG_02:246）。
    - 另外逐盤 logits 最大差、是否逐位元組相同只記錄。
    - key 陣列要和 `val_keys.npz`（`055d2941…`）相同。
    - tol：算這次 `seam_val` dump 的 GPU 和 prereg03 dump 的 GPU（兩臂都是 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，見 `_DONE_val_p03{a,b}_*.json`）相同時是 1e-4，否則 1e-3。
  - **同一執行環境**：test04 的 `seam_val`、public、private 所有 dump 的完成標記，`gpu_name`／`torch`／`cudnn` 都要相同（同 test02，`t02_rank.py:812-833`，PREREG_02 rev 6 10.10）。不同就判接縫沒過。
  - **本機 Q02 接縫**：
    - Q02 在 `seam_val` 的每題機率和 `results/prereg04/rank/rank_val_probs.csv`（`3c7ab658…`）比，最大差 ≤ tol；規則 B 和 `results/prereg04/rank/pred_B.csv`（`27b6d225…`）最多差 4 題。比對時 CRLF 摺成 LF，可直接沿用 `t02_rank.seam_compare`（`:780-809`）。
    - tol＝1e-4 的條件：test02 的 `seam_val` C5 dump 的 GPU 等於 prereg02 的 GPU，**而且** test04 的 SW dump 的 GPU 等於 prereg03 的 GPU。其他情況 1e-3。
    - tol 要等於 test04 notebook 在 `seam_check.json` 記的值（同 `t02_rank.seam_tol`，`:836-853`）。
    - O4 是 CPU 算的，接縫只接受逐位相同；開局 OP 由 test02 的 `opening` 保證和 `opening_val.npz` 逐位相同。
  - 全部都逐位元組相同時，預期輸出檔也逐位元組相同；不同就記警告（同 `t02_rank.py:948-951`）。
- **理由**：數值和規則照 5.2，比較的量照 test02 已登記的做法。

### G6. test04 的「重跑一次」與資料夾改名

- **缺口**：PREREG_02 的重跑規則只涵蓋 test02。04 要不要照用、預算是不是分開、哪些失敗算、第二次再失敗怎麼辦，都沒寫。
- **提案**：
  1. **算接縫失敗、可以重跑一次的**：
     - notebook 的 `seam_check.json` 沒過。
     - 本機 Q02 接縫沒過。
     - 三組 dump 的執行環境不同。
     - `run_log` 前面有 `seam_failed`／`repro_failed`／`readonly_changed`／`final_check_failed`（同 `t02_rank.py:122, 386-391`）。
     - 本機跑 notebook 的不是已 commit 的 `colab/test04_rank.ipynb`。
  2. **不算接縫失敗、不重跑、04 直接不用的**：Δκ ≤ 0、重播失敗 > 0.5%、T1 不是 PASS／NORMALIZED、K6 已過。這些是資料或程序的事實，重跑不會改變。
  3. **`seam_val` 的 O4 不逐位相同**：屬於本機程式或環境的問題，不是 Colab 的問題。停下查，不請使用者重跑。
  4. **預算**：test04 自己有一次登記的重跑，和 test02 的重跑分開算（兩份 notebook、兩個結果資料夾）。
  5. **做法**：
     - 本機：把 `results/test04/rank/`（`opp/` 留在原處）和 `results/test04/unpack_rank.json` 移到 `results/test04/failed/rank_N/`；失敗那次的 zip 移到 `data/test04_failed/`（gitignore）；寫 `attempt.json`（原因、zip 的 SHA-256）；commit，訊息寫「test04 attempt N archived: <原因>」。
     - 使用者：把 Drive 的 `C10/results/test04/rank` 改名成 `rank_failed_N`，「中斷連線並刪除執行階段」後開新的一個，整份重跑。
     - 確認真的重算了：新 `seam_val` dump 完成標記的 `time` 要晚於封存那份。
  6. **第二次還是沒過**：停下查清楚。要放寬容許差，只能照 PREREG_02 rev 4 第 8 條另寫 PREREG_04 的修訂（rev 4），修訂前不上傳 04。K6 照樣適用：K6 前沒有成功的執行，最終就是 R20。
  7. **同一個資料夾裡的重跑**（使用者在紅字後自己按了「全部執行」）：不算登記的那一次，照 TESTDAY:116 的做法算成第 N 次嘗試並封存。
- **理由**：完全照 PREREG_02 rev 5 9.2、rev 6 10.10–10.11、rev 7 11.1 的規則，換成 test04 的資料夾。

### G7. test04 拿不到登記的 GPU、帳號與運算單元

- **缺口**：5.2 只寫容許差「同型／不同型」，沒寫接受哪些 GPU、拿不到怎麼辦。
- **提案**：
  - **GPU**：只接受 G4（`NVIDIA RTX PRO 6000 Blackwell Server Edition`）和 A100（「大量 RAM」關），同 PREREG_03 1.3 和 PREREG_02 rev 4 8.1。其他型號由 notebook 的 GPU 關卡在計算前擋下。
    - 先選 G4；G4 是灰色或無法連線時選 A100（tol 自動變 1e-3）。
    - 兩種都拿不到：隔一段時間再試，每次的時間和畫面記進 RESULTS。
    - K6 前沒有一次成功的執行，這一季就不用 04（R20）。
    - 不為了趕 K6 換其他 GPU，也不請使用者手改 notebook。
  - **帳號**：只用 AI Pro 帳號（03 的權重只在它的 Drive）。notebook 開頭核對兩臂 `final.pth` 的 SHA，不符就停下（可能是帳號錯了）。
  - **運算單元**：開跑前「可用量」≥ 3（推估 0.7，留一次重跑的量）。
- **理由**：運作上的選擇，不影響判定；容許差照 5.2。

### G8. T1 的後果（FAIL_OPENING、FAIL_STRUCT）

- **提案**：
  - 04 的所有工具都讀 `results/test02/checks/rank/t1.json`，要求是完整的 T1（complete、stage raw、mode real），同 `official_log.t1_problems`（`:241-254`）、`t02_rank.t1_status`（`:450-475`）。
  - verdict 只接受 PASS、NORMALIZED。
  - FAIL_OPENING：`t04 opp`／`apply`／`prepare`（04）一律拒絕，錯誤訊息寫明「04 不上場、照 test02 用 R10」。
  - 不擬合、也不使用 Q02 的無開局版（它只記錄，PREREG_04 2.4、2.5）。
  - FAIL_STRUCT：全部停，照 test02。
- **理由**：照 PREREG_04 2.5、5.3。

### G9. T1 NORMALIZED 時讀哪一份

- **缺口**：沒寫 04 讀原始檔還是正規化副本。
- **提案**：
  - 04 的所有步驟都讀 **test02 第 2 步打包的那一份**：本機是 `data/test/<檔名>`＋`data/test/MANIFEST.json`（NORMALIZED 時是正規化副本），Colab 是同一個 `C10_test_v1.zip`。絕不讀 `data/test/raw`。
  - 執色（O4 的目標方、κ 的分色）取副本裡的 `color_i`。rev 8 的轉換只清 `num_games` 之後的格子，前面的格子不變。
  - 核對：
    - test04 notebook 的 `config.json` 記的 `test_bundle.zip_sha256`、`manifest_sha256`，要等於 test02 的 `results/test02/rank/config.json`、`results/test02/checks/bundle_build.json`，以及本機 `data/test/MANIFEST.json`。
    - notebook 自己也讀 Drive 上 `C10/results/test02/rank/config.json`（唯讀），zip 的 SHA 不同就在 preflight 停下。
- **理由**：5.2「同一份 `C10_test_v1.zip`、MANIFEST 核對」的具體化。

### G10. 最後一版怎麼決定（5.3），以及 `official_log` 怎麼記

- **提案**：
  1. **04 的一筆上傳紀錄「現在符合 5.3」**的條件：
     - H4R 採用：固定事實，引 `results/prereg04/RESULTS.md`、打分紀錄帳第 25–26 行、`e8b24f2`。
     - 報告的最後一次 commit 時間 ≤ K6，而且就是當初 prepare 記的那個 commit。
     - T1 是 PASS／NORMALIZED。
     - 6 個 Δκ 都 > 0。
     - 三組重播失敗率都 ≤ 0.005，`seam_val` 逐位相同。
     - 報告和交件檔的 SHA 都沒變。
  2. **判定**：有任何一筆 04 符合，最終就是 04；否則最終是 R20（FAIL_OPENING 時 R10）。
  3. **工具強制**（第 3 節 J）：
     - `final --id <04 那筆>`：現在不符合就拒絕。
     - `final --id <R20 那筆>`：有符合的 04 時拒絕，除非加 `--not-04-reason TEXT`，寫明發現哪一條其實不成立（例如找到 04 路徑的具體 bug）。原因記進 `final_choice.rank.prereg04_5_3` 和 RESULTS。
     - log 裡完全沒有 04 的紀錄時，`final` 的行為和現在逐位相同（不加任何欄位）。
  4. **流程**：
     - 04 上傳、記下 public 分數後，符合就 `final --id <04>`。
     - 04 已上傳、之後才發現不能用：`final --id <R20 的 U1> --not-04-reason …`，再 `prepare` 一次 `results/test02/rank/submission_rank.csv`（原因寫「final: R20 per PREREG_04 5.3（哪一條）」），取得 OK 後重傳，讓它成為最後一次上傳。
     - `status` 要顯示最後一次上傳的 SHA 等於最終選擇。
     - 11/23 23:59 前完成，11/24 留作緩衝。
  5. **public 分數**：不進入判定。低 0.05 以上時只追查錯誤（`public` 會標 investigate）；查不到 bug 就維持 5.3 的結果，而且要寫進報告（VALIDATION 紀律 4）。
  6. **上傳的合法性**：04 的上傳屬於 U4 (b)「之後的登記依本機驗證規則採用的方法」（spec.md:775-779；TESTDAY:227）。
- **理由**：5.3 原文的程式化；「看分數換方法」被結構性擋掉。

### G11. K6「成功的正式執行」的定義

- **缺口**：沒定義「成功」，也沒說 commit 時間怎麼算。
- **提案**：以下全部成立，才算一次成功的正式執行：
  - test04 notebook 在 official 的 zip 上跑完（`run_log` completed、前面沒有失敗的執行、notebook 接縫過），`unpack` 後已 commit。
  - `t04 opp` 是 real mode：`seam_val` 逐位相同，三組重播失敗都 ≤ 0.5%，已 commit。
  - `t04 apply` 是 real mode：本機接縫過，6 個 Δκ 都 > 0，`check_submission --official` 過，`submission_rank.csv` 和報告已 commit。
  - **commit 時間**：報告那個 commit 的 committer 時間（`git log -1 --format=%cI -- results/test04/rank/<報告>`）≤ 2026-11-16 23:59:59 +08:00。
  - 上傳可以在 K6 之後，但要在 11/23 前。
  - Δκ 或重播不過的那次「跑完但 04 不能用」，**不算**成功。
  - `official_log prepare` 記下 `k6.committed_at`；K6 之後才 commit 的報告，prepare 直接拒絕（上傳也沒有意義，而且違反 U4）。
- **理由**：K6 原文「至少一次成功的正式執行，已 commit」；把「成功」定成 5.3 的全部條件，只會讓 04 更少上場。

### G12. 演練政策

- **K5 前必做**（K5 的「本機假測試演練通過」，由獨立代理做，紀錄寫進 `results/test04/design/build_review.md`）：
  1. **R20 逐位元組不變**：兩個一次性 scratch clone，一個是合併前的 main（`6ec5e68`，或合併當下的 main HEAD），一個是合併結果。
     - 兩邊都跑 `t02_fake_tests.py from-val --root <S>`，再跑 `t02_rank.py opening --dry-run …` 和 `apply --dry-run …`。
     - 比對 `submission_rank.csv`、`rank_test_probs.csv`、`seam_val/*`：逐位元組相同。
     - 另外 `git diff r20-testday-pre04 -- <39 個檔>` 除了 `official_log.py` 以外都是空的。
     - 競賽一的 dry run 選做（04 不碰它）。
  2. **04 端到端 dry run**：在同一棵 from-val 樹上，用假的 test04 notebook 輸出跑：
     - 假輸出的做法：`seam_val` 用 prereg03 驗證 dump 的逐位元組副本；public／private 是列切片，只改 `question_id`。
     - 依序跑 `t04 opp --dry-run`、`t04 apply --dry-run`。
     - 預期：
       - `seam_val` 的 O4 逐位相同，本機接縫逐位相同。
       - public、private 的 O4 等於 `val_o4.npz` 的對應列。
       - `rank_test_probs.csv` 在前綴對應下等於 `results/prereg04/rank/rank_val_probs.csv` 的前 800 列。
       - `seam_val` 的 Δκ 等於 check／apply 記的值。
     - 假樹是驗證棋譜：**只准跑重播和 apply**，不跑任何內容統計（同 TESTDAY:249）。
  3. 完整測試：`PY -m pytest -q tests` 全綠（含 `test_p04_hygiene.py`、`test_t02_*`、`test_p02_*`、新的 `test_t04_*`），以及 `C10_SLOW=2` 慢速演練。
  4. 負面案例：用 `official_log` 的新單元測試逐條觸發拒絕（第 3.5 節）。
- **10/28–11/02 選做（要使用者，約 20 分鐘，約 0.7 單元）**：Colab 彩排 test04。
  - 和 test02 的彩排共用同一個 rehearsal zip。
  - 結果寫到 Drive 的 `C10/results/test04_rehearsal/rank`。
  - 本機用 `--rehearsal` 跑 `unpack`／`opp`／`apply`，預期同上一點（G4 上逐位相同；A100 只要求 ≤ 1e-3、規則 B ≤ 4）。
  - 彩排不上傳、不 commit，結果記進 RESULTS 或 rev 4（不在已 commit 的 rev 3 後面補字）。
  - 彩排結束立刻收拾：`test04_rank.ipynb` 收進 `old_rehearsal`，刪掉「下載」裡的彩排 zip。
  - 彩排前 notebook 要已 commit、已審查。彩排後若改了產生器，要重建、重新審查、commit，在 K5 前合併。
  - **強烈建議做**：這是 test04 notebook 在 11/04 前唯一一次真 GPU 執行。
- **理由**：5.1 和 K5 的演練要求加上 test02 的彩排做法；彩排本身不影響判定。

### G13. 官方定義變動（D5）和 R20 指紋的衝突

- **缺口**：PREREG_02 9.11（`:305`）允許 D5 時改 test02 的檔名表並重建，但這些檔都在指紋裡，`test_p04_hygiene` 會失敗。指紋又規定「永不更新」。
- **提案**：D5 時 R20 路徑照 PREREG_02 rev 5 第 11 條改（R20 的登記優先）。
  - `test_p04_hygiene` 因此失敗是預期的。
  - 另寫 PREREG_04 修訂，列出改了哪些檔、改前後的 SHA，並附 `git diff r20-testday-pre04 -- <檔>`，證明只改檔名表。
  - 04 的工具和 notebook 同樣只改檔名表並重建、冒煙、審查；K6 照樣適用。
  - 修訂前不上傳 04。R20 的上傳照 PREREG_02 的規則，不等 04。
  - 不更新指紋 JSON，也不移動 tag。
- **理由**：避免測試日兩份登記互相卡住；科學不變。

### G14. 測試日 04 的結果寫在哪

- **提案**：
  - `results/prereg04/**` 是凍結的，所以測試日 04 的結果寫在 `results/test04/RESULTS.md`（新檔）。
  - 內容：各步驟的數字、Δκ、重播率、接縫、GPU、上傳紀錄（引 `official_scores.json`）、5.3 的判定。
  - 由另一個代理核對數字；地圖和 repo 的 `CLAUDE.md` 同步更新。

### G15. 看過測試資料後才發現 04 的 bug

- **提案**：
  - 修 bug 屬於 U4 (a)。要寫 PREREG_04 修訂（rev 4），指出具體的程式錯誤（同 H4R「要指出具體的程式錯誤才算」的標準）。修正不得改變來源、參數或組合。
  - 修正後照 G11 重跑受影響的步驟：notebook 有 bug 才重跑 notebook；這不算登記的「接縫重跑一次」，但要記錄。
  - K6 之後修好的，這一季不用 04。
  - 已上傳的 04 有 bug：照 G10 以 R20 為最終並重傳。

### G16. `counts_fit.npz` 不在 git 裡

- **提案**：
  - `t04 opp` 開頭核對 `data/prereg04/counts_fit.npz` 的 SHA＝`f6c6debe…`（`count_report.json`），並重算 `tables_sha256`＝`f380e0d7…`。
  - 遺失時只能照 `p04_opp.py count` 重算：至少 7 個 worker，登記的是 8 個（10.1 b），約 34 分鐘。這要排在沒有其他線吃 CPU 的時段。
  - 11/03 前把這條寫進 `docs/TESTDAY_04.md` 第 1 步的檢查。
- 重播和打分本身可以用 ≤ 6 個 worker（沒有 int32 防護的限制）。

### G17. 執色語意

- **提案**：
  - 「對調」＝整題每一盤的目標執色都換成另一色，原方向（對稱 0），同 prereg03 dump 的 `view: 's0_colour_swapped'`。
  - 正確執色只用在 κ，`apply` 要斷言合併項只有 `OP、O4、SWA、SWB`（`p04_rank_fuse.TERMS`，`:183`）。
  - O4 的目標方＝`color_i`，同驗證（`p04_opp.py:1295`）。

### G18. 測試棋和 fit 重複（對 O4 的意義）

- **提案**：
  - 不新增規則。T2(a) 已經抓複製；它的後果（開 split v2、新登記前不得利用這些複製、甲照常上傳）同樣適用 04，因為 04 也是甲，計數只用 fit。
  - `t04 opp` 可以用 `p04_opp` 的 dupcheck 函式記錄「測試盤在 fit 裡的比例」（原色、對調分開），只記錄，寫進 RESULTS。

### G19. 時間與日期

- **提案**：
  - 04 的目標：11/05–11/07 上傳。
  - R20 的 U1 延誤（例如 test02 重跑）時，04 順延。
  - K6 11/16 23:59 是硬期限。
  - 最後一次上傳 11/23 23:59 前，截止 11/24 23:59。
  - 競賽二每天最多 5 次，R20 和 04 合計。
  - 11/04 當天只做 test02，不做 04。

### G20. `official_log.py` 怎麼「認得 04」（和 plan.md 的差異）

- **提案**：
  - 04 的檔以路徑認：`results/test04/rank/submission_rank.csv`，那個資料夾只有 `t04_rank.py apply` 會寫。
  - 04 的檔綁在它自己的報告上，報告**不叫** `apply_test_report.json`：萬一辨識失敗，舊的 R20 路徑會因「報告不存在」拒絕，不會被當成 R20。
  - 方法名稱在新程式裡設。
  - 不做 `plan.md:372` 的「`RANK_COMBOS` 加項」，原因見第 3.2 節。這是 plan 的偏離，要寫進 rev 3；登記的 5.1 只要求「新方法名稱、認得 04 的檔」。

### G21. 換行

- **提案**：`.gitattributes`（不在指紋裡）加一行 `results/test04/** -text`，理由同 PREREG_02 rev 7 第 7 條：報告和交件檔的 SHA 要綁定，重新 checkout 不能改位元組。`colab/*.ipynb -text` 已經涵蓋 `test04_rank.ipynb`。

### G22. 使用者操作的不可逆步驟都要先檢查

- **提案**：寫進 COLAB_RUN 和 TESTDAY_04（第 4 節）：
  - 上傳 AIdea：先發訊息 A（檢查結果＋請使用者對那一份說 OK），OK 之後另一則訊息 B 只放上傳步驟。
  - 刪執行階段：先看分頁左上角的檔名。
  - 改 Drive 資料夾名稱：Claude 給完整路徑和新名稱，先截圖確認。
  - 刪「下載」裡的舊 zip：先確認檔名和日期。

### G23. 一盤都解析不了的題

- **提案**：照 test02（PREREG_02 9.3），交件填 D（`decide.FALLBACK`）；04 不另外處理，題號記進報告。

### G24. notebook 的 GPU 型號記錄

- **提案**：notebook 印 `device cuda | GPU … | ARM A/B …` 那一行；`config.json` 記 GPU、torch、cuDNN、兩臂權重的 SHA。本機把它記進 `t04_apply_report.json` 的 `notebook` 欄，`official_log` 再記進上傳紀錄。

### G25. 04 的報告被 rerun 覆蓋

- **提案**：`t04 apply` 在 real mode 不覆蓋已存在的輸出，除非加 `--rerun --reason`，同 `t02_rank`。
  - 覆蓋前舊輸出一定要先照 G6 封存。
  - 覆蓋後，舊的 `official_scores.json` 紀錄因報告 SHA 不同，自動失去 5.3 資格（第 3 節 J 的重新核對）。

### G26. 本機環境版本

- **提案**：
  - 11/24 前不升級 `.venv`（numpy 2.5.3、scipy 1.18.1、Python 3.12.13，見 `val_report.json`）。
  - `t04 opp`／`apply` 的報告記版本。
  - O4 不逐位相同時，先比版本。

### rev 3 的骨架（給寫登記的代理）

- **修訂紀錄新增一列**（不改舊列）：
  `| rev 3 | (commit) | 2026-10-xx，測試日擴充（第 5 節的細節，第 11 節）。**官方測試集 11/04 11:00 才開放，當時沒有任何測試資料**。t04_rank.py opp／apply、test04 notebook、official_log.py 的附加式修改、Δκ 與重播失敗的算法、順序、重跑一次、GPU、K6 的定義、最後一版的判定、演練。來源、格子、比率、組合、參數、閘門、H4R、截止點都不變 |`
- **新一節**「## 11. rev 3：測試日擴充（K5 前登記，沒有任何測試資料）」：
  - 11.1 範圍與檔案：新檔清單、只寫 `results/test04/`、指紋不動。
  - 11.2 順序（G4）。
  - 11.3 O4 重播（G2、G3、G16、G18）。
  - 11.4 test04 notebook（G5 前半、G7、G9、G17、G24）。
  - 11.5 apply（G5 後半、G2 第 5 點、G23、G25）。
  - 11.6 Δκ（G1）。
  - 11.7 重跑一次（G6）與 bug（G15）。
  - 11.8 T1（G8）。
  - 11.9 上傳、最後一版與 `official_log`（G10、G19、G20）。
  - 11.10 K6（G11）。
  - 11.11 演練（G12）。
  - 11.12 D5（G13）。
  - 11.13 換行與環境（G21、G26）。
  - 11.14 揭露：rev 3 是 H4R 之後、看過驗證集分數才寫的；但只補程序，不改任何會影響驗證分數的東西。O4 的 Δκ 提前在 `opp` 算，只是為了省 Colab。

---

## 3. `tools/official_log.py` 的附加式修改（精確）

### 3.1 限制（違反任何一條，測試就失敗）

- **只准附加**（`tests/test_p04_hygiene.py:101-114`）：tag 版的每一行（LF 摺疊後）要照原順序出現在新檔。可以在任何地方插入新行；不能改、刪、移動舊行，**包括縮排和尾端逗號**。
- **只用標準函式庫**（`tests/test_t02_hygiene.py:141-142`）：不 import `p02_*`、`t02_*`、`t02_rank`／`t02_player`／`t02_checks`／`decide`、`numpy`。也建議不 import `t04_rank`（它會帶 numpy 進來）。
- **禁用字**（`test_t02_hygiene.py:71-86`）：新程式、註解、字串都不能出現答案檔名的那個字（`"ans"+"wers"`）。
- **路徑在呼叫時才算**：`test_t02_infra.py:975-1000` 會 mock `ol.REPO` 和 `ol.git_file_state`，所以新路徑一律在函式裡用 `REPO` 組，不在 import 時先算好。新的 git 呼叫包成可被 mock 的函式（`git_commit_time`）。
- **R20 的行為要逐位相同**：沒有 04 紀錄時，`prepare`／`uploaded`／`public`／`final` 的輸出和 JSON 都不能變；`test_t02_infra.py` 的 OfficialLog 11 項測試原封不動要全過。唯一多的輸出是 `status` 的一行 K6 倒數（只增加文字，不改判斷）。

### 3.2 不做的事

- **不改 `RANK_COMBOS = ("R20", "R10")`（`:66`）**：改那一行，附加檢查就失敗。
- **也不另外寫一行 `RANK_COMBOS = RANK_COMBOS + ("Q02",)`**：
  - `method_for`（`:297-307`）會把 Q02 當成「另一個組合」。一份 04 的檔就不能帶含「Q02」的 `--method`，因為它執行時 `used` 會是 R20。
  - R20 的檔帶含 Q02 的 `--method` 會多出一個新的拒絕，等於改了 R20 的分支邏輯。
  - 所以 04 的方法由新程式在 `method_for` 之後設定。
- **不動 `FILE_NAMES`、`REPORT_OF`、`DEFAULT_METHOD`、`check_reports` 的 R20 部分、`method_for`、`check_t2_t5`**。04 的檔名一樣是 `submission_rank.csv`，所以 `cmd_prepare` 的檔名檢查（`:373-374`）照舊適用。

### 3.3 插入點（行號是 tag 版；每一塊都是「在第 N 行之後插入」）

**A. 在 `:7` 之後（docstring 的用法）**

```
                                      [--not-04-reason TEXT]    (rank only; PREREG_04 5.3, below)
```

**B. 在 `:38` 之後（docstring 最後，`"""` 之前；`argparse` 的 description 只取第一段，不受影響）**

```
PREREG_04 rev 3 (competition 2, test04; additive only, tests/test_p04_hygiene.py; R20's lines above are the
          tagged r20-testday-pre04 version): a rank file in results/test04/rank/ is a 04 file (Q02 = R20 + O4 + SWA
          + SWB). prepare binds it to the committed tools/t04_rank.py apply report next to it (real mode, combo Q02,
          notebook and local seams passed, O4 replay failures <= 0.5 % of the slots of each set with seam_val
          byte-identical to the committed validation O4, dkappa > 0 for O4 / SWA / SWB on public and private
          separately, check_submission official, this file's SHA-256), its seam and O4 replay records, the committed
          real R20 run of test02 (not the R10 fallback), the full raw T1 (PASS or NORMALIZED; FAIL_OPENING: 04 never
          plays), R20's first upload (logged as uploaded: PREREG_04 5.3) and K6 (the report committed by 2026-11-16
          23:59 +08:00); the method is the 04 one. final applies PREREG_04 5.3: a 04 entry only while every condition
          still holds; an R20 / R10 entry while some 04 entry meets them only with --not-04-reason TEXT (recorded).
          A log without 04 entries is handled exactly as before. status adds the 04 state and the K6 countdown.
```

**C. 在 `:77`（`_NEEDLE = …`）之後：常數**

```python
# ---- PREREG_04 rev 3 (test04, competition 2 only); additive, R20's constants above are unchanged
T04_DIR = ("results", "test04", "rank")                   # only tools/t04_rank.py apply writes a 04 file here
T04_REPORT = "t04_apply_report.json"                      # never apply_test_report.json (that is R20's)
T04_FORMAT = "c10-test04 rank apply v1"
T04_COMBO_ID = "Q02"
T04_COMBO_SOURCES = ("C5", "OP", "O4", "SWA", "SWB")
T04_METHOD = "prereg04 Q02 = R20 + O4 + SWA + SWB, rule B (甲)"
T04_OPPONENT_SOURCES = ("O4", "SWA", "SWB")               # PREREG_04 5.2: dkappa > 0 on each test set
T04_TEST_SETS = ("public", "private")
T04_REPLAY_MAX = 0.005                                    # PREREG_04 5.2: more than 0.5 % of a set's slots -> no 04
P04_H4R = "adopted (results/prereg04/RESULTS.md; score_ledger.csv rows 25-26; commit e8b24f2)"
K6 = datetime(2026, 11, 16, 23, 59, 59, tzinfo=TW)        # PREREG_04 4: a successful official 04 run committed
```

**D. 在 `:312`（`check_reports` 的 docstring 結尾）之後**

```python
    if is_t04_file(comp, path):          # PREREG_04 rev 3: a results/test04/rank file is bound to its own reports
        return check_reports_t04(comp, path, sha)
```

**E. 在 `:357`（`check_reports` 的 `return`）之後：新函式**

```python
# ---------------------------------------------------------------- PREREG_04 rev 3: test04 (competition 2)
def t04_dir():
    return os.path.join(REPO, *T04_DIR)


def is_t04_file(comp, path):
    """A competition-2 file in results/test04/rank/ is a 04 file (only tools/t04_rank.py apply writes there)."""
    if comp != "rank":
        return False
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return os.path.normcase(here) == os.path.normcase(os.path.realpath(t04_dir()))


def git_commit_time(rel):
    """(commit, committer time ISO) of the last commit that touched rel; (None, None) when there is none."""
    rc, out = git("log", "-1", "--format=%H %cI", "--", rel)
    if rc != 0 or not out:
        return None, None
    h, t = out.split()
    return h, t


def r20_first_upload(log):
    """PREREG_04 5.3: R20's first upload = the first uploaded competition-2 entry that is not a 04 file."""
    ups = [x for x in uploads_of(log, "rank") if not x.get("test04")]
    return ups[0] if ups else None


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def t04_conditions(rep):
    """PREREG_04 5.2 / 5.3 recomputed from the numbers of the apply report (never from its own verdicts).
    -> ({'dkappa': {'public.O4': ..}, 'replay_rate': {'seam_val': ..}}, problems)."""
    probs, dk, rates = [], {}, {}
    col = rep.get("colour_check") or {}
    for s in T04_TEST_SETS:
        for src in T04_OPPONENT_SOURCES:
            v = ((col.get(s) or {}).get(src) or {}).get("dkappa")
            dk[f"{s}.{src}"] = v
            if not _is_num(v) or not v > 0:
                probs.append(f"dkappa of {src} on {s} is {v!r}; PREREG_04 5.2 needs > 0 (04 is not used)")
    rp = rep.get("replay") or {}
    for s in ("seam_val",) + T04_TEST_SETS:
        r = rp.get(s) or {}
        n, f = r.get("slots"), r.get("failures")
        if not (isinstance(n, int) and n > 0 and isinstance(f, int) and 0 <= f <= n):
            rates[s] = None
            probs.append(f"O4 replay of {s}: slots {n!r}, failures {f!r}")
            continue
        rates[s] = f / n
        if f / n > T04_REPLAY_MAX:
            probs.append(f"O4 replay of {s}: {f} of {n} slots failed ({f / n:.2%} > {T04_REPLAY_MAX:.1%}; no 04)")
    if (rp.get("seam_val") or {}).get("equal_to_committed_val_o4") is not True:
        probs.append("O4 of seam_val is not byte-identical to results/prereg04/opp/val_o4.npz (PREREG_04 5.2)")
    return {"dkappa": dk, "replay_rate": rates}, probs


def check_reports_t04(comp, path, sha):
    """PREREG_04 5.2 / 5.3 and rev 3: bind a 04 file to the committed t04_rank.py apply report next to it, its local
    seam and O4 replay records, the committed real R20 run of test02 (not the R10 fallback), the full raw T1 (PASS or
    NORMALIZED) and K6. Returns check_reports' shape plus 'test04'."""
    d = t04_dir()
    frel = repo_rel(path)
    out, problems = {}, []
    rpath = os.path.join(d, T04_REPORT)
    rel, rep = _committed_json(rpath, "the test04 apply report")
    out[rel] = sha256_file(rpath)
    if rep.get("format") != T04_FORMAT or rep.get("mode") != "real":
        problems.append(f"{rel}: format {rep.get('format')!r}, mode {rep.get('mode')!r} (a real-mode {T04_FORMAT})")
    combo = rep.get("combo") or {}
    if combo.get("id") != T04_COMBO_ID or tuple(combo.get("sources") or ()) != T04_COMBO_SOURCES:
        problems.append(f"{rel}: combination {combo.get('id')!r} {combo.get('sources')!r} is not the adopted Q02")
    if (rep.get("notebook") or {}).get("seam_check_passed") is not True:
        problems.append(f"{rel}: the test04 notebook seam did not pass")
    if (rep.get("seam_local") or {}).get("passed") is not True:
        problems.append(f"{rel}: the local Q02 seam did not pass")
    cs = rep.get("check_submission") or {}
    if cs.get("mode") != "official" or cs.get("errors"):
        problems.append(f"{rel}: check_submission {cs.get('mode')!r} with errors {cs.get('errors')}")
    want = (rep.get("outputs_sha256") or {}).get(FILE_NAMES[comp])
    if want != sha:
        problems.append(f"{rel}: records another {FILE_NAMES[comp]} than this file ({sha[:12]})"
                        + lf_hint(path, want, frel))
    cond, cprobs = t04_conditions(rep)
    problems += cprobs
    for sub, what in ((("seam_val", "seam_local.json"), "the test04 local seam record"),
                      (("opp", "opp_report.json"), "the test04 O4 replay report")):
        p = os.path.join(d, *sub)
        srel, obj = _committed_json(p, what)
        out[srel] = sha256_file(p)
        if obj.get("passed") is not True or obj.get("mode") != "real":
            problems.append(f"{srel}: passed={obj.get('passed')!r}, mode={obj.get('mode')!r}")
    t2 = os.path.join(REPO, "results", "test02", "rank", REPORT_OF["rank"])
    t2rel, t2rep = _committed_json(t2, "the test02 rank apply report (R20 goes first, PREREG_04 5.3)")
    out[t2rel] = sha256_file(t2)
    if t2rep.get("mode") != "real" or t2rep.get("fallback_used") is not False \
            or (t2rep.get("seam_local") or {}).get("passed") is not True:
        problems.append(f"{t2rel}: mode {t2rep.get('mode')!r}, fallback_used {t2rep.get('fallback_used')!r}: 04 needs "
                        f"test02's passed real-mode R20 run (FAIL_OPENING: 04 never plays)")
    t1_path = os.path.join(REPO, "results", "test02", "checks", comp, "t1.json")
    trel, t1 = _committed_json(t1_path, "the T1 result")
    out[trel] = sha256_file(t1_path)
    problems += t1_problems(t1, trel)
    v = t1.get("verdict")
    if v not in T1_UPLOAD_OK:
        problems.append(f"{trel}: verdict {v!r}; 04 plays only with T1 PASS or NORMALIZED (PREREG_04 5.3: "
                        f"FAIL_OPENING -> R10, FAIL_STRUCT -> stop)")
    h, t = git_commit_time(rel)
    k6_ok = t is not None and parse_time(t, "commit time") <= K6
    if not k6_ok:
        problems.append(f"{rel} was last committed at {t} ({h}), not by K6 {K6.isoformat()}: no 04 this season "
                        f"(PREREG_04 4); the final choice is R20")
    if problems:
        raise LogError("the file is not backed by a usable 04 run (PREREG_04 5.2, 5.3, rev 3):\n - "
                       + "\n - ".join(problems))
    rec = dict(cond, combo=T04_COMBO_ID, report=rel, report_sha256=out[rel],
               k6={"commit": h, "committed_at": t, "deadline": K6.isoformat(), "ok": k6_ok},
               seam_local_tol=(rep.get("seam_local") or {}).get("tol"), notebook_gpu=(rep.get("notebook") or {}).get("gpu"))
    return {"reports": out, "t1_verdict": v, "fallback_used": False, "test04": rec}


def method_for_t04(given):
    """A 04 file is recorded as Q02, never with R20's default label (method_for has already refused a name of R10)."""
    if given and not re.search(r"(?<![A-Za-z0-9])%s(?![0-9])" % T04_COMBO_ID, given):
        raise LogError(f"--method {given!r} does not name {T04_COMBO_ID}; a results/test04/rank file is the 04 "
                       f"combination (default {T04_METHOD!r})")
    return given or T04_METHOD


def print_t04_ok_lines(t4):
    print(f"- 這一份是 04（{T04_METHOD}），不是 R20；R20 的第一次上傳是 #{t4.get('r20_first_upload_id')}")
    print("- 04 的檢查：test04 notebook 和本機（Q02）的接縫都通過；重播失敗 "
          + "、".join(f"{s} {r:.2%}" for s, r in t4["replay_rate"].items())
          + f"（每組上限 {T04_REPLAY_MAX:.1%}）；Δκ 都 > 0："
          + "、".join(f"{k} {v:+.4f}" for k, v in t4["dkappa"].items()))
    print(f"- K6：04 的報告在 {t4['k6']['committed_at']} commit（期限 11/16 23:59）")


def t04_entry_eligible(x):
    """PREREG_04 5.3 for one logged 04 entry, re-checked now: H4R adopted (P04_H4R), the report commit recorded at
    prepare is still its last commit and is by K6, T1 PASS / NORMALIZED, every dkappa > 0, replay <= 0.5 %, and the
    report and the uploaded file are unchanged. -> (ok, reasons)."""
    t4, why = x.get("test04") or {}, []
    if x.get("t1_verdict") not in T1_UPLOAD_OK:
        why.append(f"T1 {x.get('t1_verdict')!r}")
    k6 = t4.get("k6") or {}
    h, t = git_commit_time(t4.get("report") or "")
    if not k6.get("ok") or h is None or h != k6.get("commit") or parse_time(t, "commit time") > K6:
        why.append(f"K6: the report's last commit is {h} at {t} (prepare recorded {k6.get('commit')})")
    dk = t4.get("dkappa") or {}
    if len(dk) != len(T04_TEST_SETS) * len(T04_OPPONENT_SOURCES) or not all(_is_num(v) and v > 0 for v in dk.values()):
        why.append(f"dkappa {dk}")
    rr = t4.get("replay_rate") or {}
    if set(rr) != {"seam_val", *T04_TEST_SETS} or not all(_is_num(v) and v <= T04_REPLAY_MAX for v in rr.values()):
        why.append(f"replay rates {rr}")
    for rel_, want in ((t4.get("report"), t4.get("report_sha256")), (x.get("file"), x.get("sha256"))):
        p = os.path.join(REPO, *str(rel_ or "").split("/"))
        if not rel_ or not os.path.exists(p) or sha256_file(p) != want:
            why.append(f"{rel_} changed since prepare")
    return not why, why


def final_rule_t04(log, e, not_04_reason):
    """PREREG_04 5.3 for competition 2. None when no 04 entry is logged (the R20-only log is handled exactly as
    before). A 04 entry is final only while t04_entry_eligible holds; while some 04 entry holds, an R20 / R10 final
    needs --not-04-reason TEXT (the condition found broken, e.g. a bug in the 04 path), which is recorded."""
    if e["comp"] != "rank":
        return None
    t04 = [x for x in log["uploads"] if x["comp"] == "rank" and x.get("test04")]
    if not t04:
        return None
    elig = {x["id"]: t04_entry_eligible(x) for x in t04}
    ok_ids = [i for i, (ok, _w) in elig.items() if ok]
    reason = (not_04_reason or "").strip()
    rec = {"rule": "PREREG_04 5.3", "h4r": P04_H4R, "eligible_04_entries": ok_ids,
           "not_eligible_04_entries": {str(i): w for i, (ok, w) in elig.items() if not ok}, "checked_at": now_iso()}
    if e.get("test04"):
        ok, why = elig[e["id"]]
        if not ok:
            raise LogError(f"entry #{e['id']} (04) does not meet PREREG_04 5.3 now ({'; '.join(why)}): the final choice "
                           f"is R20 (R10 when T1 is FAIL_OPENING); upload it again so that it is the last upload")
        return dict(rec, chosen="04")
    if ok_ids and not reason:
        raise LogError(f"04 entries {ok_ids} meet every condition of PREREG_04 5.3, so the final choice is 04; an R20 "
                       f"final needs --not-04-reason TEXT naming the broken condition (recorded, and written into the "
                       f"report)")
    return dict(rec, chosen="R10" if e.get("fallback_used") else "R20", not_04_reason=reason or None)


def t04_status(log, fin):
    ents = [x for x in log["uploads"] if x["comp"] == "rank" and x.get("test04")]
    return {"entries": [x["id"] for x in ents], "eligible": [x["id"] for x in ents if t04_entry_eligible(x)[0]],
            "final_is_04": bool(fin and any(x["id"] == fin.get("id") for x in ents))}


def t04_status_lines(st):
    return [f"  04（PREREG_04 5.3）：紀錄 {st['entries']}；現在符合條件的 {st['eligible'] or '沒有'}；最終選擇"
            f"{'是 04' if st['final_is_04'] else '不是 04'}"]
```

**F. 在 `:394`（`method = method_for(…)`）之後**

```python
    if bound.get("test04"):              # PREREG_04 rev 3: a 04 file never carries R20's label
        method = method_for_t04(args.method)
```

**G. 在 `:397`（`log = load_log(log_path)`）之後**

```python
    if bound.get("test04"):              # PREREG_04 5.3: R20's first upload is logged before any 04 upload
        r20 = r20_first_upload(log)
        if r20 is None:
            raise LogError("PREREG_04 5.3: log R20's first upload (results/test02/rank/submission_rank.csv, status "
                           "uploaded) before preparing a 04 upload (docs/TESTDAY_04.md)")
        bound["test04"]["r20_first_upload_id"] = r20["id"]
```

**H. 在 `:407`（`e = {…}` 的最後一行）之後**

```python
    if bound.get("test04"):
        e["test04"] = bound["test04"]
```

**I. 在 `:423`（T2–T5 那一行 print 的結尾）之後**

```python
    if bound.get("test04"):
        print_t04_ok_lines(bound["test04"])
```

**J. `cmd_final`：在 `:502` 之後插入第一段，在 `:507` 之後插入第二段**

第一段：

```python
    rule04 = final_rule_t04(log, e, getattr(args, "not_04_reason", None))      # PREREG_04 5.3
```

第二段：

```python
    if rule04:
        log["final_choice"][args.comp]["prereg04_5_3"] = rule04
```

**K. `cmd_status`：在 `:533` 之後、`:548` 之後、`:550` 之後各插一段**

`:533` 之後：

```python
        if comp == "rank":
            out[comp]["prereg04"] = t04_status(log, fin)
```

`:548` 之後：

```python
        if comp == "rank" and out[comp]["prereg04"]["entries"]:
            lines += t04_status_lines(out[comp]["prereg04"])
```

`:550` 之後：

```python
    lines.append(f"距 11/16 23:59（K6：04 的正式執行要在這之前 commit）：{_left(now, K6)}")
```

**L. parser：在 `:583`（`p.add_argument("--basis", required=True)`）之後**

```python
    p.add_argument("--not-04-reason", help="rank: an R20 / R10 final choice although a 04 entry meets PREREG_04 5.3 "
                                           "(name the broken condition; recorded)")
```

- **scratch 試套（2026-09-30，U4）**：用 `apply_blocks.py` 把本節 A–L 照行號插進 `official_log.py` 的 scratch 副本，結果如下（repo 沒有改動）：
  - `py_compile` 通過；和 tag 版比，子序列檢查為真；沒有禁用字。
  - `test_t02_infra.py` 的 OfficialLog 11 項，加上 `test_t02_hygiene` 的 `NoLabelFileName::test_sources`、`TeamSeparation::test_infra`，全部通過。
  - 試寫的 `trial_t04_log.py`（繼承 OfficialLog，所以 R20 的 11 項在 `results/test04/rank` 也有檔的情況下再跑一次）共 14 項、18 個 subtest 通過。涵蓋：04 的 prepare、R20 的 U1 還沒上傳時拒絕、final 選 04、R20 沒帶原因時拒絕、帶原因時成功、K6 之後拒絕、重播 26/5000 拒絕（25/5000 過）、Δκ 等於 0 拒絕、combo、notebook 接縫、mode、`--method` 沒寫 Q02、FAIL_OPENING、只有 R20 時 final 沒有新欄位。
  - 兩支腳本都在 `…\scratchpad\p04testday\U4\`（`apply_blocks.py`、`repo\tests\trial_t04_log.py`），可以當 U2 寫 `tests/test_t04_official_log.py` 的起點。
- 檢查：以上每一塊都只是新增行；舊行一字不改。
- 插入 E 的位置在 `# --- commands` 註解（`:360`）之前、兩個空行之後都可以。
- `check_reports` 在 D 呼叫的 `is_t04_file`、`check_reports_t04` 是 E 定義的模組層級函式，執行時已存在，Python 不需要先宣告。

### 3.4 `t04_rank.py apply` 要寫出的報告（`official_log` 依賴的欄位契約，給 U1／U2）

`results/test04/rank/t04_apply_report.json`：

```
format            "c10-test04 rank apply v1"
mode              "real" | "dry-run" | "rehearsal" | "smoke"
combo             {"id": "Q02", "sources": ["C5","OP","O4","SWA","SWB"], ...}   (frozen_config.json combo)
notebook          {"seam_check_passed": bool, "run_log_status": "completed", "earlier_failed_runs": [],
                   "gpu": str, "torch": str, "cudnn": int, "config_sha256": str, "notebook_sha256": str,
                   "same_as_repo_notebook": bool, "weights_sha256": {"a": .., "b": ..}}
seam_local        {"passed": bool, "tol": 1e-4|1e-3, "max_abs_diff": float, "rule_b_differences": int,
                   "reference": "results/prereg04/rank/rank_val_probs.csv", "byte_identical_probs": bool, ...}
replay            {"seam_val": {"slots": 19156, "failures": 0, "equal_to_committed_val_o4": true, ...},
                   "public": {"slots": int, "failures": int, ...}, "private": {...}, "max_rate": 0.005}
colour_check      {"public": {"O4": {"dkappa": float, "sub_questions": int, "by_colour": {...}, ...},
                              "SWA": {...}, "SWB": {...}},
                   "private": {...}, "seam_val": {... record / consistency ...}, "definition": "...PREREG_04 rev 3 11.6"}
usable            bool   (all registered conditions of this run; official_log recomputes them, never trusts this)
unusable_reasons  [str]
check_submission  {"mode": "official", "questions": [...], "errors": []}
outputs_sha256    {"rank_test_probs.csv": .., "pred_A.csv": .., "pred_B.csv": .., "decide_summary.json": ..,
                   "submission_rank.csv": ..}
inputs_sha256     {"opp_report.json": .., "test02 apply_test_report.json": .., "opening_test.npz": ..,
                   "frozen_config.json": "ae728e9c…", "t1.json": .., ...}
test02            {"fallback_used": false, "t1_verdict": "PASS"|"NORMALIZED", "r20_first_upload_id": int}
tool, environment, created, seconds, rerun
```

- `results/test04/rank/seam_val/seam_local.json`：`{"passed": bool, "mode": ..., …}`。
- `results/test04/rank/opp/opp_report.json`：`{"passed": bool, "mode": ..., "sets": {...}, "tables_sha256": "f380e0d7…", "counts_sha256": "f6c6debe…", …}`。其中 `passed`＝`seam_val` 逐位相同、三組都 ≤ 0.5%，另外記錄 O4 的早期 Δκ。
- 陣列檔 `results/test04/rank/opp/test_o4.npz`：比照 `val_o4.npz` 的 9 個陣列，多一個 `set`（`<U8`）。
- `submission_rank.csv` 只在 `usable` 為真時才寫。不能用時只寫報告（`usable: false` 和原因），不留可以上傳的檔。

### 3.5 新測試 `tests/test_t04_official_log.py`（不改 `test_t02_infra.py`）

setup 比照 `test_t02_infra.py:975-1041`：在暫存 repo 裡放 test02 的報告和 `t04_apply_report.json`、`seam_local.json`、`opp_report.json`，mock `ol.REPO`、`ol.git_file_state`、`ol.git_commit_time`。

1. R20 的 U1 已記 uploaded：04 的 prepare 成功；`method`＝`T04_METHOD`；`e["test04"]` 有 6 個 Δκ、3 個重播率、`k6`、`r20_first_upload_id`；`reports_sha256` 含 test02 的報告和 `t1.json`；OK 請求的文字含「這一份是 04」。
2. 拒絕，每一條各一個 subTest：
   - 沒有 R20 的 U1（只有 prepared 的不算）。
   - T1 是 FAIL_OPENING 或 FAIL_STRUCT。
   - test02 的 `fallback_used` 為真。
   - 6 個 Δκ 任一個 ≤ 0、None 或 NaN。
   - public 26/5000 失敗（25/5000 過，測邊界）。
   - `seam_val` 失敗 1 盤，或 `equal_to_committed_val_o4` 為否。
   - notebook 接縫沒過；本機接縫沒過。
   - combo 不是 Q02；mode 不是 real；format 不對。
   - `outputs_sha256` 和檔案不符（附 CRLF 提示）。
   - 報告沒 commit。
   - `git_commit_time` 在 K6 之後（2026-11-17T00:00:00+08:00）。
   - `seam_local.json` 或 `opp_report.json` 沒過或沒 commit。
3. `--method "prereg04 04"`（沒寫 Q02）被拒；`--method "… R10 …"` 被 `method_for` 拒；`--method "prereg04 Q02 (04)"` 可以。
4. R20 路徑不變：
   - 在 `results/test04/rank/` 也有檔的情況下，prepare `results/test02/rank/submission_rank.csv` 的 entry 和原來逐欄相同，沒有 `test04` 欄。
   - `test_t02_infra.OfficialLog` 全部原封不動通過（在完整測試裡跑）。
5. `final`：
   - 04 符合 → final 04 成功，記 `prereg04_5_3.chosen == "04"`。
   - R20 在沒有 `--not-04-reason` 時被拒，有的話成功，原因記錄在案。
   - 04 報告的 SHA 變了，或 `git_commit_time` 的 commit 不同 → final 04 被拒。
   - log 裡沒有 04 時，`final_choice.rank` 沒有 `prereg04_5_3` 欄，和現在逐位相同。
6. `status`：有 04 紀錄時多一行「04（PREREG_04 5.3）」；K6 倒數一行一定在；沒有 04 時 `out["rank"]["prereg04"]["entries"] == []`。
7. `is_subsequence`：tag 版 → 新版（`test_p04_hygiene` 已涵蓋）；禁用字、只 import 標準函式庫（`test_t02_hygiene` 已涵蓋，確認新程式也過）。

### 3.6 `results/official_scores.json` 會長什麼樣子（04 的一筆）

- 格式照舊（`"c10 official_scores v1"`）。
- 每筆上傳的欄位和 R20 相同（`:400-407`），另外多一個 `"test04"`：`{"dkappa": {"public.O4": …, …}, "replay_rate": {"seam_val": 0.0, "public": …, "private": …}, "combo": "Q02", "report": "results/test04/rank/t04_apply_report.json", "report_sha256": …, "k6": {"commit": …, "committed_at": …, "deadline": "2026-11-16T23:59:59+08:00", "ok": true}, "seam_local_tol": …, "notebook_gpu": …, "r20_first_upload_id": 1}`。
- 04 那筆的 `fallback_used` 是 false，`method` 是 `T04_METHOD`。
- `final_choice.rank` 在有 04 紀錄時多一個 `prereg04_5_3`（chosen、eligible、not_eligible 和原因、`not_04_reason`、checked_at）。

---

## 4. 文件大綱

### 4.1 `docs/TESTDAY_04.md`（Claude 的指令順序；新檔，不在指紋裡）

開頭：
- 這份是 04 的測試日步驟，test02 照 `docs/TESTDAY.md`（不改）。
- 依據：PREREG_04 第 5 節和 rev 3 第 11 節；兩者不一致時以登記為準。
- `PY`、repo 根目錄、每段做完就 commit（作者 kuotunyu，不加 Co-Authored-By）；`data/**` 和 zip 不 commit，SHA 記在會 commit 的 JSON 裡。

**每一步都適用的規則**
- 不開答案檔；不跑 `score.py`；本機不用 GPU。
- 凍結檔清單：R20 的 39 個檔、`results/prereg0{1..4}/**`、p02／p03／p04 工具、`frozen_config.json`。每次開工先跑 `PY -m pytest -q tests/test_p04_hygiene.py tests/test_t02_hygiene.py`。
- 工具拒絕時照訊息處理，不繞過；要改程式或放寬門檻，先寫 PREREG_04 修訂（下一版 rev 4）。
- 上傳：每一份、每一次都要 OK；檢查結果放在較早的訊息，上傳指示自成一則。
- 04 絕不在 11/04 當天做。

**11/03 以前要就緒（K5 核對表）**
- rev 3 已 commit、分支已合併回 main。
- R20 逐位元組演練、04 dry run、完整測試、`C10_SLOW=2` 都過，紀錄在 `results/test04/design/build_review.md`。
- `data/prereg04/counts_fit.npz` 的 SHA＝`f6c6debe…`。
- `colab/test04_rank.ipynb` 已 commit，上傳副本在 `data\colab_upload\test04\`（只有 1 個檔）且 SHA 相同。
- （選做）彩排已做並收拾。
- `.gitattributes` 有 `results/test04/** -text`。

**第 1 步：閘門（test02 做完了嗎）**
- `PY tools/official_log.py status`：競賽二要有 U1 且已 uploaded。
- `results/test02/checks/rank/t1.json`：PASS 或 NORMALIZED。FAIL_OPENING：04 整條不做，RESULTS 記「04 不上場（T1 FAIL_OPENING）」，最終＝R10。FAIL_STRUCT：停。
- `git status` 乾淨；`counts_fit.npz` 的 SHA 正確；`.venv` 的版本和 `val_report.json` 相同。

**第 2 步：`PY tools/t04_rank.py opp`**
- 讀 `data/test`（MANIFEST），不讀 raw。
- 看 `seam_val` 是否逐位相同、三組的重播率、O4 的早期 Δκ。
- 不過的後果表：
  - 重播 > 0.5%，或 O4 的 Δκ ≤ 0：04 不用，不開 Colab，最終＝R20，寫 RESULTS。
  - `seam_val` 不同：停下查本機環境，不請使用者做任何事。
- commit `results/test04/rank/opp/`。

**第 3 步：給使用者跑 test04**
- 核對 `data\colab_upload\test04\test04_rank.ipynb` 的 SHA（PowerShell 一行，先自己跑過）。
- 核對 Drive 狀態：`C10_test_v1.zip` 應該還在 `C10`，是 test02 那一份。
- 把 COLAB_RUN「測試集 04」一節的步驟給使用者。

**第 4 步：收結果**
- `PY tools/t04_rank.py unpack --zip results/test04/test04_rank_results.zip` → 看印出的執行紀錄 → commit。
- 拒絕的情況和「(1)」檔名的處理，照 TESTDAY 第 4 步的寫法。

**重跑一次（test04）**
- 封存指令：`rank` 改為 test04 的路徑、`opp/` 留在原處。
- 請使用者改 Drive 資料夾名稱（逐步寫，附完整路徑）。
- 確認真的重算了。
- 第二次還是不過：停，G6 第 6 點。

**第 5 步：`PY tools/t04_rank.py apply`**
- 輸出清單；本機接縫；Δκ 表；重播；`check_submission`。
- 不過的後果表：
  - 本機接縫沒過：重跑一次。
  - Δκ／重播不過：04 不用，最終 R20。
  - 其他拒絕：照訊息處理。
- commit。

**第 6 步：上傳 04**
- 指令：`PY tools/official_log.py prepare --comp rank --file results/test04/rank/submission_rank.csv --reason "prereg04 Q02 first upload (PREREG_04 5.3)"`。
- commit `official_scores.json`。
- 訊息 A：貼 prepare 印的 OK 請求。要特別寫「這一份是 04，不是 11/04 那份 R20；路徑在 `results\test04\rank\`」。
- 使用者說 OK 之後，訊息 B 只放上傳步驟：
  - 比賽全名、網址（`docs/REGISTRATION.md:91`）、TEAM_11248。
  - 用 `explorer.exe "D:\AI-Portfolio\competitions\aicup2026-go\results\test04\rank"` 打開資料夾再拖檔。
  - SHA 前 12 碼。
  - 傳完回報時間和 public 分數。
- 使用者回報後：`uploaded`、`public`，然後 commit。
- investigate 的處理：只查錯誤，不換方法。

**第 7 步：最後一版（U5，照 5.3）**
- 符合條件表（G10 第 1 點）。
- 指令：
  - 04 符合：`final --id <04> --basis "PREREG_04 5.3: …"`。
  - 04 不符合：`final --id <R20 U1> --basis …`，必要時照 G10 重傳 R20。
  - 已傳 04 之後才發現不能用：`--not-04-reason`，再重傳 R20。
- 11/23 前 `status` 要顯示最後一次上傳就是最終選擇；11/23 前告訴使用者最後交哪一份。

**第 8 步：RESULTS**
- `results/test04/RESULTS.md`：由另一個代理核對數字。
- 更新地圖和 `CLAUDE.md`。
- 看過官方分數才做的選擇要寫明。

**K6 與日期**：11/16 23:59 前沒有成功的執行，這一季就不用 04（寫 RESULTS）。

**D5（官方定義變動）**：照 G13。

**彩排（選做，10/28–11/02）**：指令都帶 `--rehearsal`；預期結果；收拾清單；彩排不上傳、不 commit。

### 4.2 `docs/COLAB_RUN.md` 新一節「# 測試集 04（test04）（使用者操作步驟）」

放在檔尾「事前登記 03」之後，其他節不改。寫法照 test02 那一節（`COLAB_RUN.md:226-345`），給不寫程式的人看。

- **開頭框**：
  - Claude 說「可以跑 test04 了」之後才用這一節：11/04 那份 R20 已經傳上 AIdea、Claude 也在本機檢查過 04 可以用之後，通常是 11/05–11/07。
  - notebook 第一格的說明和這一節不一致時，以這一節為準。
  - 只有 1 份 notebook、只推論、約 5 分鐘（推估，還沒實測）、約 0.7 個運算單元。
  - 不用寫也不用貼任何程式碼；Colab 上沒有答案檔。
- **小表**：做什麼、GPU（G4；拿不到改 A100、「大量 RAM」關；其他都不行）、時間、運算單元、跑完要下載的檔 `test04_rank_results.zip`。
- **開始前**：
  1. 帳號一定要用 **AI Pro 帳號**：03 的兩個模型只在它的 Drive。
  2. Drive 的 `C10` 裡應該看得到：
     - `results` 資料夾，以及 `old_` 開頭的資料夾。
     - `C10_bundle_v1.zip`、`C10_test_v1.zip`（11/04 那份，**不要動、不要重傳**）、`test02_rank.ipynb`、`test02_player.ipynb`。
     - 看不到 `C10_test_v1.zip`：先停，截圖給 Claude。
  3. 電腦不要睡眠（同 test02 第 3 步）。
- **A. 放上 notebook**：
  4. 貼 `explorer.exe "D:\AI-Portfolio\competitions\aicup2026-go\data\colab_upload\test04"`（裡面只有 `test04_rank.ipynb`），拖進 `C10`。
     - 跳出「上傳選項」就按「取消」，截圖。
     - 做完後核對 `C10` 裡的清單。
- **B. 跑**：
  5. 從 Drive 的 `C10` 開（右鍵 →「開啟工具」→「Google Colaboratory」），確認三件事：
     - 頭像是 AI Pro 帳號。
     - 檔名是 `test04_rank.ipynb`。
     - 第一格大標題是「C10 測試日（test04）：競賽二 04…」。
     - 任何一個不對：關掉這個分頁，不要按「全部執行」。
  6. 「查看資源」：可用量至少 3。不夠就停，告訴 Claude。
  7. 選 G4；G4 是灰色就選 A100，「大量 RAM」關。
  8. 「全部執行」，處理對話框（同 test02 第 10 步）。
  9. 確認 GPU 那一行，以及第一格印的「種類 official」。
- **C. 等它跑完**：分頁不要關；回來看時只看，不按任何按鈕。
- **D. 跑完**：
  10. 跑完的樣子：最後兩行是「Drive 已寫入完成並卸載」和「完成。請下載 `test04_rank_results.zip`」。有紅字時照常下載、截圖給 Claude，**不要自己重跑**。
  11. **先看左上角的檔名**，再「中斷連線並刪除執行階段」。
  12. 先刪掉「下載」資料夾裡舊的同名 zip（先看日期），再到 Drive →`C10`→`results`→`test04` 下載。
  13. 貼 `explorer.exe "D:\AI-Portfolio\competitions\aicup2026-go\results\test04"`，把 zip 拖進去，說「放好了」，並說拿到哪種 GPU。睡眠設定改回來。
- **上傳交件檔（04）**：
  - Claude 會先發一則訊息 A：檢查結果，加上「這一份是 04」、檔案路徑、SHA 前 12 碼。
  - 你在對話裡說 OK 之後，Claude 才另外給上傳步驟（訊息 B）。
  - 每一次上傳都要重新說 OK。
  - 注意：這次的檔在 `results\test04\rank\`，和 11/04 那份（`results\test02\rank\`）同名，一定要從 Claude 給的那一行指令打開的資料夾拖。
- **要重跑時（只有 Claude 請你做時）**：
  - Drive →`C10`→`results`→`test04`，在 `rank` 資料夾上按右鍵 →「重新命名」→ 改成 `rank_failed_1`（數字照 Claude 給的）。
  - 改名前先截圖給 Claude 確認路徑。
  - 刪除執行階段後開新的一個，從第 5 步重做。
- **出狀況時**（表格，照 test02 的寫法）：
  - GPU 不對：改 A100，或等一下再試；不要換別種 GPU。
  - 紅字提到權重、`final.pth`、帳號：不要搬動或上傳任何檔；刪除執行階段、確認頭像後再「全部執行」一次；還是不行就截圖。
  - 紅字提到 `C10_test_v1.zip`、MANIFEST、「和 test02 用的不同」：不要重傳、不要解壓，截圖。
  - 「接縫核對沒過」：照常下載，截圖，不要重跑。
  - 中途斷線：選同一種 GPU 接著跑；那一種拿不到就先停，告訴 Claude。
- **不要做的事**：
  - 不要開 `old_` 資料夾裡的 notebook。
  - **不要動 `C10/results/prereg03` 裡的任何檔**：兩個 `final.pth` 刪掉就再也做不出 04。
  - 不要改 notebook。
  - 同一份不要在兩個分頁同時跑。
  - 結果 zip 只放到 `results\test04\`。
  - 不要把 AIdea 的測試檔直接傳到 Drive。
- **彩排（只有 Claude 安排時）**：
  - 步驟相同，只有兩處不同：結果在 `results`→`test04_rehearsal`；跑完把 `test04_rank.ipynb` 拖進 `old_rehearsal`。
  - 彩排的 zip 下載後由 Claude 立刻移走。

---

## 5. 風險

| # | 風險 | 影響 | 對策 |
|---|---|---|---|
| 1 | test04 notebook 從沒在真 GPU 上跑過（5 分鐘只是推估） | 11/05 第一次正式執行就出錯，吃掉登記的重跑，K6 期限變緊 | 強烈建議 10/28–11/02 做 Colab 彩排（G12）；CPU 冒煙；把 notebook 的崩潰（preflight／例外）和接縫失敗分開，前者不算登記的重跑 |
| 2 | 03 的權重只在 AI Pro 的 Drive（`C10/results/prereg03/rank_{a,b}/ckpt/final.pth`） | 被刪就做不出 04（重訓不能逐位重現） | COLAB_RUN 的「不要做的事」、CLAUDE.md「在等使用者」已寫；可以選擇請使用者本人另外下載一份到本機 `data/prereg03/ckpt/`（gitignore），由使用者決定 |
| 3 | `data/prereg04/counts_fit.npz` 不在 git 裡 | 遺失要重算約 34 分鐘，而且至少要 7 個 worker（int32 防護），這台電腦同時有其他線在跑 | TESTDAY_04 第 1 步先核對 SHA；重算排在沒人用 CPU 的時段 |
| 4 | D5（官方改格式）和 R20 指紋衝突 | `test_p04_hygiene` 必然失敗，兩份登記互相卡住 | G13：R20 照 PREREG_02 9.11 改，另寫 PREREG_04 修訂附 diff；不動指紋 |
| 5 | 兩個都叫 `submission_rank.csv` | 上傳時拖錯檔（AIdea 看不出差別） | 訊息 A、B 都寫完整路徑和 SHA 前 12 碼；用 `explorer.exe` 打開正確的資料夾；`prepare` 的 OK 請求多印「這一份是 04」 |
| 6 | 附加式修改不小心動到舊行（縮排、尾逗號、docstring） | hygiene 測試失敗，或更糟：語意變了但測試沒抓到 | 第 3 節只用「在第 N 行之後插入」；新測試逐條測 R20 的行為不變；`test_t02_infra` 原封不動要全過 |
| 7 | 04 的辨識靠路徑 | 有人把 R20 的檔放進 `results/test04/rank/`，或反過來 | 04 的報告另外命名（`t04_apply_report.json`），格式、combo 都要對；R20 的路徑找不到 `apply_test_report.json` 就拒絕 |
| 8 | 官方執色語意和訓練不同（Δκ ≤ 0） | 04 不上場 | 這是設計上的保護，照 5.3 交 R20；O4 的 Δκ 在 `opp` 就先看到，可以省下一次 Colab |
| 9 | public 分數 04 比 R20 低 ≥ 0.05 | 有換回 R20 的壓力（看分數選方法） | `public` 標 investigate，只查 bug；`final` 在 04 符合時拒絕 R20，除非寫出原因 |
| 10 | K6 太緊：11/04 的 test02 要是重跑，04 順延 | 11/16 前沒有成功的執行 → 交 R20 | 12 天緩衝；test04 只要約 5 分鐘；彩排先把 notebook 的 bug 找出來 |
| 11 | 環境飄移（numpy／scipy 升級、`core.autocrlf=true`） | O4 不再逐位相同，或報告 SHA 綁定失效 | 11/24 前不升級 `.venv`；`.gitattributes` 加 `results/test04/** -text`；比對時 CRLF 摺成 LF |
| 12 | git 的 commit 時間可以手動設定 | K6 的判定只是自我申報 | 照實做；`official_scores.json` 另記 `prepared_at`（UTC 現在時間）當旁證 |
| 13 | A100 和 G4 混用（test02 在 A100、test04 在 G4，或反過來） | 本機接縫 tol 變 1e-3，機率不會逐位相同 | 登記已允許；照 G5 定 tol，並記錄 |
| 14 | 使用者在紅字後自己重按「全部執行」 | 那次不算登記的重跑，輸出不能用 | notebook 看到上一次接縫沒過就停；`unpack` 看 `run_log` 拒絕；COLAB_RUN 寫明不要自己重跑 |
| 15 | 用 from-val 假樹做演練時，誤把內容統計（T1.4–T5）跑在驗證棋譜上 | 違反規則 | `t04` 的 dry run 只准跑重播和 apply；照 TESTDAY:249 的警語 |
| 16 | 平行的其他 session 同時修改 `CLAUDE.md` 或地圖 | 合併時衝突 | 只有文件會衝突；合併前 `git diff main...p04-testday` 檢查；工具檔不會衝突 |
| 17 | 合併回 main 太晚（K5 23:59） | 這一季不用 04 | 內部目標 10/30 合併（plan.md:419）；演練、審查、修正都排在 10/28–11/02 |
