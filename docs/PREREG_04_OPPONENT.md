# 事前登記 04：競賽二加上「對手的落子」兩項證據（O4 計數表、03 模型讀對手那一方）

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。前幾份：
  - `docs/PREREG_02_OPENING_FUSION.md`：競賽二 R20、競賽一 Q04，都採用。
  - `docs/PREREG_03_RECIPE_V2.md`：沒過閘門，未送分，結果在 `results/prereg03/RESULTS.md`。
- 設計過程：`results/prereg04/design/`，都只用 fit／inner 訓練玩家，沒有開任何答案檔。
  - 三份調查：`understand_eng.md`、`understand_probe.md`、`understand_prior.md`。
  - 三份提案、兩位評審（`judges.json`）、綜合方案 `plan.md`。
  - 腳本、JSON、log 在 `evidence/`。
  - 大檔在 `data/prereg04/design_scratch/`（不進 git），SHA 記在 `scratch_manifest.json`。
- 本登記寫四件事：比什麼、怎麼選、怎樣算成功、什麼時候放棄。
  - 實作規格以 `plan.md` 第 2、5–7 節為準，但**不含逐手 CNN（MV）**，見下面「範圍」。
  - 實作時要偏離規格，在跑 `check` 之前記進修訂紀錄。
- **只做競賽二。** 競賽一不登記新方法、不打分；04 的任何輸出都不進競賽一。
- **範圍**（使用者 2026-09-29 的決定）：
  - (A) 同意範圍改變。原本 04 指的是「逐手、依等級條件化的 CNN」。設計階段發現，主要的新訊號在**對手的落子**，而且有兩個來源不需要新的訓練，04 改成登記這兩項。
  - (B) 這次**不做**逐手 CNN（MV）的試跑和訓練，10 月不開 Colab。
    - 理由：設計推估 CNN 在 O4＋SW 之上的增量不到 +0.003。第 2.4 節的選法要求比前一個組合高 0.003 以上才會被選上，所以就算做了，很可能也用不上。
    - 之後若要做，另寫一份登記（05），用它自己的打分預算。
  - `plan.md` 第 1、3、4 節的試跑、正式訓練、K0–K2 截止點都不適用；第 2.4 節的 MV 規格和第 2.5 節的 Q03 不在本登記。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `cdec913` | 第一版，2026-09-29。當時**沒有任何 04 的正式輸出或分數**：`results/prereg04/` 只有 `design/`，沒有 `check`／`tune`／`apply` 的輸出。驗證集的 O4 沒算過；本機驗證集自 prereg02 之後沒有再打過分。設計階段在 fit／inner 上算過的 O4 計數和 inner 分數都已揭露，數字和來源見第 0 節和 `design/` |
| rev 2 | `82027fa` | 實作偏離與補強，見第 10 節。2026-09-29 和 `tools/p04_*.py`、`tests/test_p04_*.py` 一起 commit，**在正式 `selftest`／`count` 之前**，所以也在 `check` 之前。當時仍然沒有任何 04 的正式輸出或分數，驗證題的 O4 沒算過。scratch clone 的正式模式演練結果：inner 輸出和 `proxy_inner_all.npz` 逐位元相同；Q00／Q01／Q02 的 OOF 和第 2.3 節的數字差 0。來源、格子和比率的定義、組合、選法、閘門、H4R、截止點都不變 |
| rev 3 | `62ca557`（第 11 節在合併前、任何測試資料之前，依 K5 驗證修正於 `62794a3`、`5bc8774`） | 測試日擴充的程序（第 11 節）。2026-09-30 建置，經三輪、每輪三個角度的獨立審查（`results/test04/design/build_review.md`），和 `tools/t04_rank.py`、`tools/t04_fake_tests.py`、`colab/build_test04_rank_nb.py` → `colab/test04_rank.ipynb`（和 `results/test04/notebook_diff_rank.txt`）、`colab/make_test04_smoke_tree.py`、`tools/official_log.py` 的附加式修改（對 tag `r20-testday-pre04` 只有加行）、`tests/test_t04_*.py`、`docs/TESTDAY_04.md`、`docs/COLAB_RUN.md` 新一節、`.gitattributes`／`.gitignore` 的新行、`results/test04/design/` 一起 commit，在第 11 節第 13 條的正式模式演練和合併回 main 之前，也在 K5 之前。**官方測試集 11/04 11:00 才開放，當時沒有任何測試資料**，也沒有任何 04 的測試日輸出。H4R 已經打過分（`e8b24f2`），本修訂在那之後才寫，但只寫測試日的程序：來源、格子和比率的定義、組合 Q02、凍結參數、選法、閘門、H4R、截止點都不變。會影響測試日判定的是第 11 節第 1–13、19、20 條；11.6–11.9 記實作和規格（`spec.md`）的差異，其中會影響判定的都已寫進那幾條 |

## 0. 為什麼做這個（都不看答案；看過 inner 才定，偏樂觀）

- R20 在 inner 5 折交叉驗證是 0.5855，本機驗證集 0.5891（唯一打過分的數字）。
- 03 的教訓：新模型的逐盤準確率從 25.6% 提高到 27.4%，但整題分數只多 +0.0014 到 +0.0036。
- 設計階段（`understand_probe.md`、`evidence/B_proxy/`）：用 CPU 計數表模擬「依等級的逐手落子似然」。
  - **原本打算用的「目標玩家自己的手」**：加在 R20 上，5 折擬合出的權重全是 0，差值 +0.0000。另外 7 種目標方的變體，最好的是 +0.0033，CI 含 0。
  - **對手那一方的手**（用目標玩家的等級去評對手每一手）：R20＋它 0.5855 → 0.6042，+0.0188 [+0.0081, +0.0288]。
    - 10 個等級全部是正的。
    - 在單色假題（官方結構）上 +0.0182 [+0.0130, +0.0234]。
    - 雜訊對照組最多 +0.0006。
  - **03 的兩個模型改成讀對手那一方**（整題執色對調的輸出，03 已經算好並 commit）：再疊上去後，R20＋O4＋SW 是 0.6101（比 R20 高 +0.0247）。只看增量是 +0.0059 [−0.0016, +0.0139]；在單色假題上是 +0.0092 [+0.0056, +0.0136]。
  - 同一個 03 模型用**正確執色**，疊上去是 −0.0014。「看對手有用、看自己沒用」的不對稱，在計數表和整盤 CNN 這兩種獨立的來源上都出現了。
- **可能的機制**（沒辦法直接驗證）：
  - 網站配對的對手實力和目標玩家相近，所以對手的水準是棋力的旁證。
  - 目標玩家只有一個人，他的個人習慣會蓋過等級的特徵；對手是很多不同的人，個人習慣互相抵消。
- **挑選的自由度**（第 8 節揭露）：
  - O4 是事前主變體（只看目標方，失敗了）的鏡像；對手方實際只比了 3 種，全部變體加起來 9 個以上。
  - SW 是從 12 個組合裡看出來的。
  - 所以第 2.5 節的 inner 閘門對這兩項幾乎沒有保護力，**真正的判定只有第 3 節那一次驗證集打分**。

## 1. 來源

### 1.1 沿用（不重算）

- **C5**：prereg02 epoch 9、8 種對稱的逐盤 logits，也就是 R20 的整盤 CNN。
- **OP**：開局項，用已 commit 的 `results/prereg02/rank/opening_{inner,val}.npz`（M＝12、α＝20，只用 fit 計數）。

### 1.2 O4：對手落子的計數表（本機 CPU；定義照設計階段的程式原樣凍結）

- **格子**：每一手都從輪到下的那一方看棋盤，每個空點編成一個格子，共 557,928 種。格子由以下組成：
  - 3×3 鄰域，8 種對稱正規化後 1,107 類。
  - 戰術碼：能提子、救被叫吃的子、叫吃。
  - 和對方上一手的距離（7 級）、和自己上一手的距離（3 級）。
  - 手數階段：< 30、< 100、其餘。
- **比率**：r_k(c)＝(C_k(c)＋20)／(A_k(c)＋20／r(c))，α＝20。
  - C_k(c)：等級 k 的表裡，格子 c 被下的次數。
  - A_k(c)：格子 c 出現為空點的次數。
  - r(c)：10 級合併的比率。
- **每一手**：log P_k＝log r_k(下的格子) − log Σ_空點 r_k。
- **每盤**：把**對手**的每一手加總，得到 10 維向量。重播整份 SGF，不受 250 個盤面的窗口限制。
- **計數**：只用 `tools/c10data.load_rows(part="fit")`。表上的等級是目標玩家的等級。交件模型是甲（只用 fit 計數，不用全部 8,007 人）。
- **程式**：`tools/p04_opp.py` 從 `design/evidence/B_proxy/{feat.py,count.py,score_inner.py}` 逐字移植。
- **釘死的數字**，任一不符就停下查：
  - fit 720,307 盤；目標方 79,985,811 手、對手方 79,985,352 手；0 盤異常；557,928 個格子。
  - 計數陣列要和 `data/prereg04/design_scratch/counts_fit.npz`（SHA-256 `f6c6debe…`，全文在 `scratch_manifest.json`）逐陣列相同。
  - inner 73,556 盤的輸出要和 `proxy_inner_all.npz`（`3e6e3efd…`）相差 ≤ 1e-12。
- **驗證題的 O4**：只讀 `data/local/rank_val_questions.csv` 的題目內容和 `color_i`，不讀答案，而且**要在 `frozen_config.json` commit 之後才准算**（工具強制）。重播失敗必須是 0 盤，否則停下查。
- **只當參照、不能選的 T4**：同一套程式改成評目標方自己的手。只用在第 2.2 節的執色檢查。

### 1.3 SWA、SWB：03 的模型讀對手那一方（已 commit，不需要 Colab）

- **內容**：`results/prereg03/rank_{a,b}/dumps/{inner,val}_p03{a,b}_s0_swap.npz`，也就是原方向、整題執色對調後的逐盤 logits，每盤取 log_softmax。
- **本登記把 SHA-256 釘死**：

| 檔案 | SHA-256 |
|---|---|
| `rank_a/dumps/inner_p03a_s0_swap.npz` | `743825209569aeb922b0246587ec5801f3c2e0a10083bfb6111a327dd5ff1071` |
| `rank_a/dumps/val_p03a_s0_swap.npz` | `b355c26bf68069feb0e48ee95c46bd211c2bdbc6b5a48416c8408a210aec3848` |
| `rank_b/dumps/inner_p03b_s0_swap.npz` | `106887321cc9d54316e0d7d96e776c2060dc309c54720a8c7345f280056d2056` |
| `rank_b/dumps/val_p03b_s0_swap.npz` | `0cadff4bae36b2a41f02fddf4366a2081d9912ede4a3063c8b50fd28a2647527` |

- 權重只在 AI Pro 帳號 Drive 的 `C10/results/prereg03/rank_{a,b}/ckpt/final.pth`：A 臂 `790b767c…`、B 臂 `8e88f6ec…`，全文在 `results/prereg03/RESULTS.md`。
- **執色參照 A0、B0**：同一個模型、原方向、正確執色，也就是 `rank_{a,b}/dumps/inner_p03{a,b}_d4.npz` 的第 0 個對稱。只用在第 2.2 節的執色檢查，不能選。
- **新用途要說明**：PREREG_03 1.3 寫對調的 logits「只供第 2.2 節的檢查」，那是 03 自己的規則。本登記另外登記它當合併來源的新用途，並在第 8 節揭露。

## 2. 本機的合併與選擇（只用 CPU，`tools/p04_rank_fuse.py`）

### 2.1 合併公式（沿用 02／03 的校準，只多加幾項）

p＝softmax( e^a · (n/12)^γ · ( m_C5 ＋ w_op·m_OP ＋ Σ_j w_j·E_j ＋ b ) )

- **m_C5、m_OP**：和 R20 相同。
- **E_j**：新來源 j（O4、SWA、SWB）在這一題上，每盤可解析的棋先算 log_softmax(該盤 10 維向量)，再取平均；一盤都沒有就記 0。
- **盤數**：只經由共用溫度 (n/12)^γ 進來。E_j 是平均，不是加總。
- **參數**：
  - 界限：a (−4, 4)、γ (−1, 2)、所有 w (0, 4)。
  - 起點：a 取 0 和 1 兩個，其他從 0 開始。
  - PENALTY 0.01‖b‖²，MAXITER 500。
- **沿用 p02 的設定**：5 折、假題輪次（`results/prereg02/rank/pseudo_inner.json`）、規則 B、PlayerBootstrap，都和 `tools/p02_rank_fuse.py` 相同。
  - import p02／p03 的函式，**不改任何 p02、p03 檔**。
  - 工具執行時核對 `p02_*.py`、`p03_rank_fuse.py`、`c10data.py`、`decide.py` 的 SHA-256 和 HEAD 相同。
  - 多項目標函數從設計階段的 `gen_obj`／`gen_fit`（`evidence/B_proxy/fuse.py`）移植，梯度用有限差分驗算（寫成測試）。

### 2.2 check：來源能不能用（在選組合之前）

1. **結構**：
   - 各來源的 key 陣列和 prereg02 的 `inner_keys.npz`、`val_keys.npz` 相同。
   - SW 的 SHA 等於第 1.3 節。
   - NaN 樣式和 C5 相同。
   - O4 的釘死數字相符。
2. **執色語意**：每個對手方來源都要和它的目標方參照比較：O4 對 T4、SWA 對 A0、SWB 對 B0。
   - Δκ＝κ(目標方參照) − κ(對手方來源)，κ 的定義同 PREREG_03 2.2 第 4 條。
   - 要求：
     - inner 單色假題上 Δκ 的 95% 下界 > 0。單色假題用 PREREG_03 rev 3 (a) 的同一組，120,544 題，SHA `45d0f14a…`。
     - 驗證單色子題的 Δκ > 0。
   - O4 的驗證 Δκ 要等驗證題 O4 算出來才有，所以延到 `apply`（第 2.6 節）。
   - 不成立的來源，連同用到它的組合，在 tune 之前就從清單拿掉。
3. **重複棋檢查**（`p04_opp.py dupcheck`）：
   - 驗證題的落子序列和 fit 比對，原色、執色對調兩種都比；只讀題目檔。
   - 有重複就記錄並揭露；達 slot 的 1% 以上就停下，先寫修訂才能 tune。
   - inner 對 fit 在設計階段實測是 0 盤。

### 2.3 selfcheck

- **Q00 要重現 R20**：inner OOF 0.5854702521055817、重擬合參數、驗證題機率，差距都 ≤ 1e-12，規則 B 完全相同。
- **Q01、Q02 要重現設計階段的數字**：0.6042484782462081、0.6101211982700248，差距 ≤ 1e-12。這只核對程式有沒有移植對，不是選擇。
- 不符就停下查。查明是數值路徑的差異、不是邏輯錯誤的話，寫修訂、以 repo 的數字為準並揭露。

### 2.4 組合與選法（順序固定，也是平手時的優先順序）

| 組合 | 內容 |
|---|---|
| Q00 | R20（C5＋OP） |
| Q01 | R20＋O4 |
| Q02 | R20＋O4＋SWA＋SWB |

- **選法**：inner 5 折交叉驗證的分等級平均（規則 B）最高者；和最高分差 0.003 以內的，取清單中最前面的。
- 選定後用全部擬合資料重擬合，寫進 `results/prereg04/rank/frozen_config.json` 並 commit，之後才能產生驗證題的預測，也才能算驗證題的 O4。
- **只記錄、不可選**：
  - O4 單獨。
  - R20＋SWA＋SWB。
  - Q02＋C6A。
  - 各組合的無開局版。
  - 各級、各盤數分箱、單色（全體、黑、白、各 k）的分數。
  - 驗證集的推估標準誤與採用機率（`se_val_record`）。

### 2.5 送分閘門（只看選中的組合對 Q00，全部成立才打分）

| 閘門 | 條件 |
|---|---|
| G1 | inner 混色 OOF 的分等級平均差 ≥ +0.010 |
| G2 | 同一差值的配對 PlayerBootstrap（每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0 |
| G3 | inner 評估輪的每題平均差 ≥ 0 |
| G4 | 單色假題（120,544 題）上的分等級平均差，PlayerBootstrap 95% 下界 > 0，而且黑題、白題各自的點估計都 > 0 |

- 選到 Q00，或任一閘門、第 2.2 節的執色條件沒過：記為「未送分」，打分機會保留，R20 照舊。
- **FAIL_OPENING**（測試日 T1）：04 不上場，照 test02 的規則用 R10。所以不另外擬合無開局版的備案，只記錄。

### 2.6 apply

- 要求 `frozen_config.json` 已 commit。
- 選到含 O4 的組合時，先跑 `p04_opp.py val` 算驗證題的 O4，再算 O4 的驗證單色子題 Δκ（第 2.2 節第 2 條）。不 > 0 就把 `score_allowed` 設為否。
- 輸出 `rank_val_probs.csv`、`apply_report.json`（`score_allowed`、`score_blocked_reasons`）。
- 接著 `decide.py`、`check_submission.py`（**要帶題目檔** `--questions data/local/rank_val_questions.csv`）。

## 3. H4R：在本機驗證集打一次分

- **指令**（寫死）：`.venv\Scripts\python.exe tools/score.py compare-rank results/prereg02/rank/pred_B.csv results/prereg04/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg04 H4R (A = prereg02 R20, reference re-read)" --json results/prereg04/rank/score_H4R.json`
- **只在 `apply_report.json` 的 `score_allowed` 為真時才跑。**
- **採用條件**（兩者都要成立）：
  - 分等級平均差（04 − R20）的 95% 信賴區間下界 > 0（每級內分層重抽 10,000 次、seed 0）。
  - 每題平均差的點估計不 < 0。
  - 否則不採用，照 VALIDATION 記為「結論依指標而定」或「沒有改善」。
- **對照組重讀**：A 側 1,589 題，兩個指標和帳上 prereg02 R20 的紀錄差距都 ≤ 1e-12，題數相同，並記錄打分用的 Python 版本（`.venv`，3.12）。不符就停下，查清前不採用。
- **合理性門檻**：新方法的分等級平均 < 0.54 時先查 bug，要指出具體的程式錯誤才算。
- **預算**：競賽二整季 4 次，已用 1 次（02）；04 最多用 1 次，查 bug 重跑也算進預算並記帳。
- **推估的採用機率**（常態近似，用各組合自己的 se_val；`plan.md` 第 6 節）：

| 組合 | inner 增益全部成立 | 打 0.75 折 | 打五折 |
|---|---|---|---|
| Q02 | 92% | 71% | 39% |
| Q01 | 77% | 53% | 27% |

## 4. 截止點（寫死；任一沒趕上，這一季不用 04，也不拿額外的打分機會補）

| 截止點 | 條件 |
|---|---|
| K3 10/14（三）23:59 | 閘門已判定；過了的話 H4R 已打分 |
| K5 11/03（二）23:59 | 採用時：測試日擴充（第 5 節）和它的修訂已 commit、已合併回 main、本機假測試演練通過 |
| K6 11/16（一）23:59 | 採用時：04 在官方測試集至少有一次成功的正式執行，已 commit |

## 5. 採用後的測試日（細節在 K5 前另寫修訂）

### 5.1 保護 R20 的路徑（硬規則，rev 1 就生效）

- **指紋**：`results/prereg04/design/r20_path_fingerprint.json`（本 rev 1 commit）記下測試日 R20 路徑上 39 個檔的 SHA-256（CRLF 折回 LF），並打本機 tag `r20-testday-pre04`。
  - 範圍：`tools/t02_*.py`、`check_submission.py`、`official_log.py`、`colab/*test02*`、`results/test02/**`、`docs/TESTDAY.md`。
- **防誤改測試**：`tests/test_p04_hygiene.py`。除了 `official_log.py`，指紋裡任何一個檔變了，測試就失敗。
- **分支隔離**：04 的測試日程式全部放在分支 `p04-testday`。以下全部通過、並經獨立審查後才合併回 main：
  - test02 的全部測試和 `test_p02_*`。
  - `C10_SLOW=2` 慢速演練。
  - 用 `t02_fake_tests.py` 做的假測試集演練：R20 的 `submission_rank.csv` 要和合併前逐位元組相同。
- **`official_log.py`** 只做附加式的修改：新方法名稱、認得 04 的檔。R20 的分支邏輯不動。

### 5.2 要做的東西

- **`tools/t04_rank.py opp`**：
  - 在 CPU 上重播三組題目（`seam_val`、public、private）的 SGF。`seam_val` 的 O4 要和已 commit 的驗證題 O4 逐位元組相同。
  - 任一組重播失敗 > 0.5% 時，04 不用。
- **只有選中含 SW 的組合時**，才需要 `colab/build_test04_rank_nb.py` → `test04_rank.ipynb`：
  - 樣式照 test02：同一份 `C10_test_v1.zip`、MANIFEST 核對、上一次接縫沒過就停。
  - 用 03 兩臂的 `final.pth`（SHA 釘死）算正確執色（只為了 κ）和對調兩種 logits。
  - 接縫：`seam_val` 要重現已 commit 的驗證 dump，同型 GPU 差 ≤ 1e-4、不同型 ≤ 1e-3。
  - 只在 AI Pro 帳號跑；推估約 5 分鐘、0.7 單元。
- **`tools/t04_rank.py apply`**：
  - 前提：test02 競賽二已完成並 commit（C5 的 dump、`opening_test.npz`）；T1 是 PASS 或 NORMALIZED；test04（需要時）已完成、接縫通過。
  - 本機接縫：要重現 `results/prereg04/rank/rank_val_probs.csv`，規則 B 的判定最多差 4 題（GPU 容許差同上）。
  - 執色檢查：每個對手方來源在 public、private 各自的 Δκ 都要 > 0，否則 04 不用。
  - 輸出 `results/test04/rank/submission_rank.csv`，跑 `check_submission --official`；只寫入 `results/test04/`。

### 5.3 上傳規則

- **11/04**：照 `docs/TESTDAY.md` 先把 test02 做完，R20 的第一次上傳記進 `official_scores.json`，之後才開始 04。04 的上傳目標是 11/05–11/07。
- **最後一次上傳（11/23 前）交 04 的條件**，全部都要成立：
  - H4R 採用。
  - K6 前有一次成功的正式執行，已 commit。
  - T1 是 PASS 或 NORMALIZED。
  - 所有 Δκ 都 > 0。
  - 重播失敗 ≤ 0.5%。
- 任一條不成立就交 **R20**，不交沒在驗證集打過分的其他組合。已經傳過 04、之後才發現不能用的話，再傳一次 R20，讓它成為最後一次。
- FAIL_OPENING：04 不上場，照 test02 用 R10。FAIL_STRUCT：照 test02 停下。
- public 分數比上一版低 0.05 以上時，只追查錯誤，不據此換方法（VALIDATION 紀律 4）。
- 每次上傳前，使用者都要在對話裡對那一份檔案明確說 OK。

## 6. 競賽一：這次不做

- 04 的任何輸出都不進競賽一；競賽一的整季預算維持「已用 1 次」。

## 7. 刻意不放進 04 的

| 做法 | 理由 |
|---|---|
| 逐手、依等級條件化的 CNN（MV） | 使用者 9/29 決定這次不做（見開頭的範圍）；要做另寫 05 |
| SE 式強度頭 | 證據顯示增益來自「對手的誤差互相抵消」，不是序數結構；小算力下沒有前例 |
| 只看目標方的計數（T4） | 疊在 R20 上 +0.0000，權重是 0；只當執色參照 |
| 03 模型的正確執色、8 種對稱的 SW | 疊在 Q01 上 −0.0014 到 +0.0005；現有的對調 dump 只有原方向，補算要多跑一次 Colab |
| 測試日 test04 失敗時改交 Q01 | Q01 沒在驗證集打過分，所以交 R20 |
| 乙（用全部 8,007 人重建計數表） | 本機驗證不了 |
| KataGo、HumanSL、預訓練模型、外部資料 | 規則禁止 |

## 8. 揭露（寫進 RESULTS 和報告）

- O4、SW 和組合清單都是看過 inner 才定的：
  - O4 是事前主變體（只看目標方，失敗了）的鏡像，全部變體 9 個以上。
  - SW 是從 12 個組合裡看出來的。
  - Q01、Q02 在 inner 上過閘門幾乎是已知結果；只有 H4R 算數。
- α＝20、格子的定義、E_j 取平均的形式，都是設計階段在 fit／inner 上定的，沒有在驗證集上試過。
- 設計階段已經在 fit／inner 上算過 O4 的計數和 inner 分數；本登記要求正式工具逐位重現那些數字，不重新選擇。
- PREREG_03 寫對調的 logits「只供檢查」；本登記另外登記它當合併來源的新用途。
- 「對手實力和目標玩家相近」這個機制沒辦法直接驗證；官方測試集上靠 public、private 的 Δκ 檢查把關。
- 03 的訓練不能逐位元組重現（cuDNN、bf16）；O4 在 CPU 上可以逐位重現。
- 生成式 AI 協助的部分照規則揭露。

## 9. 順序

1. 本登記、`results/prereg04/design/`（含 R20 路徑指紋）一起 commit，打 tag `r20-testday-pre04`；再用一個 commit 填上 rev 1 的 commit 編號。
2. 程式：`tools/p04_opp.py`（selftest、count、inner、dupcheck、val）、`tools/p04_rank_fuse.py`（check、selfcheck、tune、apply）、`tests/test_p04_*.py`（含 hygiene），獨立審查後 commit。
   - 實作偏離寫成 rev 2，在跑 `check` 之前 commit。
   - `count` 在本機約 72 分鐘（8 個 worker，設計階段實測 4,305 秒），產出的大檔放 `data/prereg04/`（不進 git），SHA 記進要 commit 的 `results/prereg04/opp/count_report.json`。
3. `count`、`inner`、`dupcheck` → commit。
4. `check` → commit；`selfcheck` → commit；`tune` → commit `frozen_config.json`；`apply`（含驗證題 O4）、`decide.py`、`check_submission.py`（帶題目檔）→ commit；`score_allowed` 為真才跑第 3 節的指令。
5. 打分紀錄帳、分數 JSON、`results/prereg04/RESULTS.md`（由另一個代理獨立核對數字）、地圖一起 commit。
6. 採用的話，照第 5 節在分支上做測試日擴充，寫修訂，K5 前合併。

## 10. rev 2：實作和登記的差異（在正式 `selftest`、`count`、`check` 之前 commit）

- 下面各項**都不改科學內容**：來源、格子和比率的定義、組合 Q00–Q02、選法、閘門 G1–G4、H4R、截止點都照 rev 1。
- 分成兩類：
  - 照原規格做不到、改成最接近的做法：10.2 的 g、j。
  - 比 rev 1 更嚴的補強，或把 `plan.md` 第 5 節的要求落實成工具強制：其餘各項。

### 10.1 `tools/p04_opp.py`

- **a. selftest**：
  - 用 fit 盤，目標方和對手方都測。
  - 提子邏輯：150 盤和 sgfmill 比。
  - 特徵：12 盤、每 7 個局面取一個，和逐點的慢速參考實作比。實測 379 個局面，比 plan 要求的 ≥ 89 個嚴。
  - `selftest.json` 和 `count_report.json` 一起 commit；`check` 會檢查它（見 l）。
- **b. count**：
  - `--stride`／`--limit` 只准預覽用。worker 數是參數，正式執行用 8 個。
  - 整數加總和 worker 數無關：8 個和 12 個 worker 算出的計數檔 SHA 都是 `f6c6debe…`。
  - worker 照設計版用 int32 累加，另外加兩道防護，任一不成立就停下，改用更多 worker：
    1. **啟動 worker 之前**：用編碼後的手數算每個 shard 的「361 × 較多那一方的手數」，要 < 2^32。
       - fit 上 6 個 worker 會被擋（4,814,925,584）。7 個以上才過（4,128,758,805）。
       - 登記的 8 個是 3,612,396,318，約 2^32 的 84%。
    2. **合併時**：用每個 shard 實際計入的手數再檢查一次同一個上界，而且任何部分和都不能是負數。
  - 兩道防護一起保證沒有 int32 溢位。
- **c. 只算登記的兩個變體**：o_choice_a20（O4）和 t_choice_a20（T4）。
- **d. 驗證題的「重播失敗」**：沒有解析到任何一手，或有一手下在已有子的點（重播在那裡停下）。
- **e. dupcheck**（照設計版 `evidence/B_proxy/dupcheck.py` 的語意）：
  - 比對整串落子：每一手的顏色和位置都相同。
  - 「原色」：fit 裡有同一串，而且那一列的目標執色和題目 slot 的 `color_i` 相同。
  - 「執色對調」：同一串，但 fit 那一列的目標執色不同，也就是同一盤從對手那邊看。
  - 1% 門檻算「任一種」的 slot 數。空的落子序列不算重複。
  - inner 對 fit 用同樣的分法，只記錄。
  - 題目檔的 slot 要和 `val_keys.npz` 逐列相同。
- **f. 工具核對**：涵蓋全部 `tools/p02_*.py`。
- **g′. val 的前提**（第 1.2、2.6 節「工具強制」的做法）：
  - `results/prereg04/rank/frozen_config.json` 已 commit，工作區沒有改動。只 stage 不算 commit。
  - 內容要是 tune 產生的正式檔：格式 `c10-prereg04 rank frozen_config v1`，smoke、preview、design_input 都是 false。
  - 選中的組合要含 O4。沒含 O4 時不需要、也不准算驗證題的 O4（第 2.6 節）。
- **h′. 比率表的 SHA**：inner、val 的報告都記 `tables_sha256`，也就是兩張表的位元組 SHA-256（[2, 557,928, 10]、float64、C 順序）。

### 10.2 `tools/p04_rank_fuse.py`

- **g. Q00 的路徑**（依第 2.3 節「數值路徑差異」修訂）：
  - Q00 走 prereg02 自己的擬合路徑（p03 的 `run_oof`、`refit_all`、`val_probs`），所以和 R20 逐位元相同。Q01、Q02 走移植的 `gen_fit`。
  - 原因：gen 路徑的 Q00 重擬合參數和 R20 差 1.39e-12，超過登記的 1e-12。差異來自加總順序：(m＋Bβ)＋w·m_OP 對 (m＋w·m_OP)＋Bβ。
  - selfcheck 另外要求 gen 路徑的 Q00 和 prereg02 路徑一致：
    - 每一題 inner 評估題的規則 B 判定都相同；
    - OOF 分等級平均差 ≤ 1e-12。
  - 以下只記錄：機率差（實測 2.4e-14）、各折參數差（2.9e-14）、重擬合參數差（1.39e-12）。
- **h. 額外的嚴格檢查**：
  - SW 的權重 SHA 要等於 03 的 `final.pth`。
  - A0／B0 和 SW 的 dump 要等於 03 check 的紀錄。
  - p04_opp 的報告 status 不是 ok、或 preview 為真，都當停止。
- **i. 兩個 design 大檔不在本機時**：check 改用 p04_opp 報告裡自己記錄的比對結果。
- **j. 驗證題 O4 的 NaN 樣式**：
  - 對驗證題的 O4，這條**取代第 2.2 節第 1 條的「NaN 樣式和 C5 相同」**。
  - 要求：`replay_failed` 全為否；C5 有解析的 slot，O4／T4 都是有限值。
  - C5 沒解析的 slot 不要求是 NaN：遮成 0，而且 E_j 只平均有解析的盤，所以不會用到。
  - 本機 C5 在 19,156 個 slot 都有解析（`val_keys.npz` 的 parsed 全為真），實際上沒有差別。
- **k. `--design-o4`**：只供預覽，正式的 tune 和 apply 都拒絕。保留下來當日後重驗的工具。
- **l. selftest 是 check 的前提**（`plan.md` 第 5 節「提子 0 盤不符、特徵 0 個不符」）：
  - 正式 check 要求 `results/prereg04/opp/selftest.json` 已 commit，和其他 p04_opp 輸出一起核對。
  - 內容要求：passed 為真、沒有 problems、不是預覽、提子和特徵都是 0 個不符。不符就停下。
- **m. 比率表 SHA 凍結**：
  - tune 把 inner 報告的 `tables_sha256` 寫進 `frozen_config.json` 的 `source_sha256.O4.tables_sha256`。
  - apply 要求 val 報告的值和它相同。
- **n. apply 的輸入綁定**（正式模式，任一不符就停下）：
  - `check_report.json` 已 commit、沒有改動、SHA 等於 `frozen_config.json` 的 `check.check_report_sha256`，而且是通過的正式 check。
  - 選中含 O4 時，`val_report.json` 記的 frozen_config SHA 要等於現在已 commit 的那一份。
  - `val_report.json` 記的 `p04_opp.py` SHA 要等於 tune 記錄的那一份。
- **o. 順序防護**：`val_o4.npz` 存在時，`frozen_config.json` 必須已追蹤，而且工作區乾淨；只 stage 不算 commit。不符就停下（check、tune 都會檢查）。

### 10.3 第 5.1 節的防誤改測試（`tests/test_p04_hygiene.py`）

- **p. 指紋本身釘死**：`r20_path_fingerprint.json`（CRLF 折成 LF）的 SHA-256 要同時等於：
  - 測試裡的常數 `72fa6c1248f24a6c3d9fb9d8f9f699237c769f9a968faf7eba31d42fd8b348da`；
  - 本機 tag `r20-testday-pre04` 裡的版本。tag 不在時測試失敗，不略過。
- **q. `official_log.py` 改成實際檢查「只准附加」**：
  - tag 版的每一行都要照原順序出現在新檔裡（用 LF 比對）。
  - 這取代 rev 1 寫的「除了 `official_log.py`」整檔豁免。第 5.1 節的規則本身不變。
- **r. 凍結的結果資料夾**：`results/prereg01`、`prereg02`、`prereg03`、`results/prereg04/design` 以下情況測試就失敗：
  - 有任何修改或新檔；
  - 有 bytecode（`__pycache__`、`.pyc`）。這些檔被 gitignore，`git status` 看不到，所以另外檢查。

### 10.4 第 9 節第 3 步改成

3. `selftest`、`count`、`inner`、`dupcheck` → commit。`selftest.json` 和 `count_report.json` 一起 commit。`count` 用 8 個 worker（至少 7 個，見 10.1 b）。

### 10.5 已知、在 04 範圍外的問題（不改登記）

- `tests/test_p03_rank_fuse.py` 有兩個測試，是在 03 的真實結果 commit 之前寫的：
  - `Unpack::test_refusals`：斷言 `results/prereg03/rank_a` 不存在，但這個資料夾在 `e55b019` 已經 commit。
  - `EndToEnd::test_real_mode_apply_needs_committed_frozen_config`：frozen_config 在 `1218ba8` 已經 commit，所以這個測試會真的執行 prereg03 的正式 apply，改寫 `results/prereg03/rank/apply_report.json`。
- 在 scratch clone 實測：`28f5ac8`（沒有 p04 檔）和 p04 檔已 commit 的狀態下，都是同樣兩個測試失敗，和 04 無關。repo 裡的 `apply_report.json` 自 `93ce679` 之後沒有被改過（git 核對）。
- 已在 `3eab110` 另外修正，只改測試檔、不動 03 的工具和結果：
  - 第一個改成斷言「被拒絕的 unpack 沒有動到 `results/prereg03`」（`git status` 前後相同）。
  - 第二個在真實的 frozen_config 已追蹤時略過，不再重跑正式 apply。
- 所以第 5.1 節和第 9 節的完整測試不需要 `--deselect`。萬一又有測試改寫凍結的結果，10.3 r 的檢查會失敗，不會悄悄過關。

## 11. rev 3：測試日擴充的程序（K5 前，沒有任何測試資料）

- 設計與實作紀錄：`results/test04/design/`（四份調查 `understand_*.md`、綜合規格 `spec.md`、建置與審查紀錄 `build_review.md`、R20 演練的基準值 `r20_fake_golden.json`）。
- 程式：`tools/t04_rank.py`（`unpack`、`opp`、`apply`、`status`、`archive`）、`tools/t04_fake_tests.py`（演練用的假 test04 輸出）、`colab/build_test04_rank_nb.py` → `colab/test04_rank.ipynb`、`colab/make_test04_smoke_tree.py`（CPU 冒煙用的 Drive 替身）、`tools/official_log.py`（只加行）、`tests/test_t04_*.py`。
- 使用者步驟在 `docs/COLAB_RUN.md`「測試集 04（test04）」；Claude 的步驟在 `docs/TESTDAY_04.md`（`docs/TESTDAY.md` 在 R20 的指紋裡，不改）。
- 下面各條是把第 5 節寫具體、比第 5 節更嚴，或在第 22 條揭露的讀法：原文有兩種讀法、或原文沒寫到的地方，第 22 條寫明採用哪一種。來源、格子和比率的定義、組合、凍結參數（`results/prereg04/rank/frozen_config.json`，SHA-256 `ae728e9c936a55dfeced255e9b05b5ac6a4955a7101f108e78c53c679e162661`）、閘門、H4R、截止點都照 rev 1、rev 2。
- 測試日只推論，不擬合、不調參。04 的所有輸出只寫在 `results/test04/`（封存的 zip 放 `data/test04_failed/`，不進 git）；R20 的 39 個檔（`tools/official_log.py` 只加行）、`results/prereg0{1,2,3,4}/**`、p02／p03／p04 的工具、`c10data.py`、`decide.py` 一律不改。
- **結束碼**（`tools/t04_rank.py` 每個指令都一樣，下面各條用這五個數）：
  - 0：完成，條件都成立。
  - 1：接縫類失敗：這次 notebook 的輸出不用，走第 9 條的重跑。
  - 2：拒絕或錯誤：前提不符、檔案不對、不是重算的結果。工具內部未預期的錯誤也是 2（印 traceback），**不是** 1。
  - 3：登記的「04 不用」：最終交 R20（T1 FAIL_OPENING 時 R10），不重跑。
  - 4：本機漂移：停下查本機的程式或 `.venv`，不是請使用者重跑 Colab 的理由；查好後 `--rerun --reason`。

### 11.1 會影響測試日判定的

1. **順序**（第 5.3 節「R20 先」的做法，由工具強制）：
   - 11/04 只照 `docs/TESTDAY.md` 做 test02，當天不做 04。
   - 04 的第一步要等 R20 的第一次上傳記成 `uploaded` 並 commit：`results/official_scores.json` 已 commit、乾淨，裡面有一筆競賽二、`status` 是 `uploaded`、檔案是 `results/test02/rank/submission_rank.csv`、SHA-256 等於 test02 報告記的交件檔。只有 `prepared` 不算。正式模式的 `opp`、`apply` 沒有這一筆就給 2。
   - 之後依序：`t04_rank.py opp`（CPU）→ commit → test04 notebook（使用者，AI Pro）→ `unpack` → commit → `apply` → commit → `official_log.py prepare` → 上傳。
   - `opp` 先做，是把登記的 CPU 步驟提前：它判定 04 不用時（第 3、7 條），**不開 Colab**，最終交 R20。選擇規則沒有改變。
   - test04 notebook 也要在 R20 的第一次上傳之後才跑。正式模式的 `unpack` 核對「算出輸出的執行」的開始時間和每個完成標記的時間，都晚於 R20 第一次上傳的時間，不成立給 2，那次輸出不用。這個核對排在第 9 條接縫類的核對之後：同一個 zip 先查到接縫類失敗時給 1，一樣不用，`archive` 一律先判成 `before_r20_u1`，所以兩種結束碼的做法相同（下一點）。「算出輸出的執行」＝最後一次執行，加上它之前狀態不是 `preflight_failed`、`interrupted`、`running` 的執行；同一個 Drive 資料夾在 R20 上傳前只停在版本確認、GPU 關卡或 preflight 的那一次不算。
   - 那一次照第 9 條的做法封存（類別 `before_r20_u1`，不占重跑次數，Drive 改名 `sw_exempt_N`）後再跑一次，不算第 9 條那一次重跑。下一次的 GPU：前面沒有占次數的封存時照第一次的規則（G4 優先、A100 備用）；前面已有 1 個占次數的封存（Drive 上有 `sw_failed_*`）時，下一次仍是登記的重跑，只用 G4（第 9 條；讀法見第 22 條）。`archive` 核對不了 R20 的第一次上傳時（紀錄或 test02 的報告沒 commit）拒絕封存（2），不當成「R20 上傳前跑的」。
   - R20 的第一次上傳延誤（例如 test02 重跑）時，04 順延；K6 不順延。K6（2026-11-16 23:59:59 +08:00）過了之後，正式模式的 `opp`、`unpack`、`apply` 一律給 3（這一季不用 04）；`status`、`archive` 不受影響。

2. **O4 的重播與 seam_val**（第 5.2 節第 1 點）：
   - O4、T4 用 `tools/p04_opp.py` 的函式計算（import，不改檔）：計數 `data/prereg04/counts_fit.npz`（SHA-256 `f6c6debe…`，`count_report.json`），比率表 `tables_sha256` 必須等於 `frozen_config.json` 記的 `f380e0d7…`（不同給 4）。計數檔不見、或 SHA 和 `count_report.json` 記的不同：`opp`、`apply` 都給 2，照第 17 條還原後再跑同一個指令（不是重跑 `opp`）。目標執色取題目檔的 `color_i`。
   - 三組題目：seam_val＝`data/local/rank_val_questions.csv`（SHA-256 `73e51bbf…`，只讀題目內容與 `color_i`）；public、private＝第 6 條的那一份測試檔。每組的 slot 和 test02 同一組的 key 逐列核對（題號、slot、執色）。
   - 「seam_val 的 O4 和已 commit 的驗證題 O4 逐位元組相同」照字面判定：`results/test04/opp/o4_seam_val.npz` 的**檔案** SHA-256 等於 `results/prereg04/opp/val_o4.npz` 的 `30d312c02c357f6761567099003885be5e08635b6e6c0b49d71baac137cd9cd4`（9/30 設計階段實測做得到，npz 成員時間固定；9/30 的演練也逐位元組相同）。9 個陣列的逐陣列比對只當診斷。
   - seam_val 的重播失敗 > 0.5%：照第 5.2 節「任一組」的字面，04 不用（3）。
   - 其餘不相同的情況（檔案不同，或 seam_val 的 O4 Δκ 和登記值差 > 1e-12，第 7 條）：4，停下查本機的程式或 `.venv`（numpy、zlib 版本），查清楚前不用 04。11/24 前不升級 `.venv`。
   - `--workers` 預設 6、範圍 1–8（這台電腦的規則是最多 6）；結果和 worker 數無關。

3. **重播失敗**（第 5.2 節「任一組重播失敗 > 0.5%」）：
   - 單位是盤（slot）；失敗的定義照 rev 2 第 10.1 節 d（沒有解析到任何一手，或有一手下在已有子的點），也就是 `p04_opp.val_scores` 的 `replay_failed`＝`(moves_parsed == 0) | (anomalies > 0)`。
   - 分母是該組題目檔的全部盤數（每題 slot 1..`num_games`，等於 test02 key 檔的列數）；seam_val、public、private 各自算。
   - 三組一樣：失敗盤數 ÷ 盤數 **> 0.005** 就不用 04（3），剛好 0.5% 算過。工具用整數算：`200 × 失敗盤數 ≤ 盤數` 才算過。官方每組 5,000 盤時上限是 25 盤。
   - seam_val 另外必須和已 commit 的驗證題 O4 逐位元組相同（第 2 條；那一份是 0 盤失敗），所以 seam_val 有失敗盤但 ≤ 0.5% 時是本機漂移（4），不是 04 的判定。`official_log` 也要求 seam_val 0 盤失敗。
   - 失敗盤不進 E_O4，也不進 O4／T4 的 κ：遮罩是「C5 有解析、而且 O4 沒有重播失敗」（`parsed & ~replay_failed`），E_O4 只平均遮罩內的盤，一盤都沒有就記 0 向量（第 2.1 節「每盤可解析的棋」的讀法）。溫度項的 n 仍是 C5 的可解析盤數；SWA、SWB 的 E 仍平均 C5 的可解析盤。驗證題失敗是 0 盤，所以這條在 seam_val 上和 prereg04 的做法完全相同。
   - 失敗盤的 `o4`、`t4` 是有限值（0 或停下前的部分和），一律用 `replay_failed` 遮掉，不靠 NaN。
   - 含失敗盤的題數、以題為單位的比例、有異常的盤數、沒有任何一手的盤數、沒有對手手數的盤數只記錄。

4. **test04 notebook**（第 5.2 節第 2 點）：
   - 用 03 兩臂的 `final.pth`（A `790b767c5425e262130d6c103c71e89238edb17f927fa48e2a2770db788e50ac`、B `8e88f6ec6cbaad5372e5d62607b99ece3d77dc7a20af86a9197553f1a6c59a42`），在 seam_val、public、private 上算原方向正確執色和整題執色對調兩種 logits。
   - 產生 03 驗證 dump 的函式（`dump_val_p03`）**逐字沿用**：8 種對稱加一次對調，fp32、沒有 autocast，存原始 logits。本機只用第 0 個對稱（A0／B0，只為了 κ）和對調（SWA／SWB，合併項）；第 1–7 個對稱只為了讓 seam_val 能和 03 的 dump 整份比對，不是新的來源。
   - 題目檔：seam_val 用 `C10_bundle_v1.zip` 裡的本機驗證題目檔（1,589 題）；public、private 用 11/04 test02 的同一份 `C10_test_v1.zip`（7 個成員的 SHA-256 和大小核對 MANIFEST）。開始算之前讀 Drive 上 test02 競賽二的 `config.json`、`run_log.json`：test02 已完成（`completed`），而且測試包的 MANIFEST 相同，否則停下。
   - **版本確認**：第一個程式格在寫任何紀錄、算任何東西之前，印出 notebook 版本（`NOTEBOOK_SHA256` 前 12 碼），請使用者和 Claude 在訊息裡給的 12 碼對照，輸入 `y` 才往下（只在 Colab 上問，正式和彩排都問）。其他輸入都停下，不算一次執行（第 9 條）。
   - **preflight**（在任何計算之前；任一不過就記 `preflight_failed` 並停下，不算一次執行）：兩臂 `final.pth` 的 SHA-256；03 的 6 個接縫參照 npz 的 SHA-256、npz 記的權重、03 完成標記記的執行環境（`NVIDIA RTX PRO 6000 Blackwell Server Edition`、torch `2.11.0+cu128`、cuDNN `91900`）；打包檔的檔案清單；測試包；test02 已完成；上一次的接縫沒過、這個資料夾之前有接縫類失敗的執行、已有完成標記（不接續）、已經跑完、改名成 `sw_interrupted_*` 的資料夾其實是接縫失敗、占次數的封存已有 2 個、登記的重跑不是 G4（第 9 條）；各組 `_INPUT.json`；Drive 剩 ≥ 0.5 GB（查不到剩餘空間時只印提醒、照常往下，同 test02）；TF32；只讀目錄的開頭指紋。
   - **GPU**：GPU 關卡照 test02 的 notebook，型號名稱含 `RTX PRO 6000`（G4）或 `A100` 才往下；其他型號、查不到型號、沒有 GPU，都在計算前停下。先選 G4、A100 的「大量 RAM」關，寫在使用者步驟。兩種都拿不到就隔一段時間再試，不換別的 GPU、不手改 notebook。第 9 條的重跑只在 G4，用完整名稱判定。
   - **帳號**：只在 AI Pro 帳號跑。權重只在那個帳號的 Drive，preflight 核對兩臂 `final.pth` 和 03 驗證 dump 的 SHA-256，找不到或不符就停下。
   - **TF32**：和 03 記錄的設定（卷積 TF32 開、矩陣乘法關）相同就不動，不同才設成 03 的值；`cudnn.benchmark` 是開的就關掉；設不成就停下。預設值和實際用的值都記進 `config.json`。
   - **只讀目錄**：`prereg01`、`prereg02`、`prereg03`，以及 test02 的 `rank` 子資料夾（不放整個 `test02`：競賽一的 notebook 會寫 `test02/player/`）。開頭和結尾的指紋不同就記 `readonly_changed`（接縫類失敗）。
   - **已知限制**：G4 的完整名稱只在 03 的完成標記看過一次。Colab 回報的名稱和 `NVIDIA RTX PRO 6000 Blackwell Server Edition` 不同時，第一次執行照常，但接縫的 tol 變成「不同型」的 1e-3（第 5 條）；登記的重跑會在 preflight 停下，本機也拒收（第 9 條）。真遇到時停下查，要改規則只能另寫本登記的修訂（rev 4 起），修訂前不上傳 04。彩排時記下 `device` 那一行的完整名稱。

5. **接縫**（第 5.2 節的兩個接縫；容許差「同型 GPU ≤ 1e-4、不同型 ≤ 1e-3」不變）：
   - **notebook 接縫**（每一臂都要成立）：
     - seam_val 的 key 陣列（`question_id`、`slot`、`color`、`parsed`）和 03 的 `val_keys.npz`（`055d2941…`）完全相同；
     - 原方向正確執色（d4 的第 0 個對稱）和對調兩種 logits，各算「每題可解析盤 softmax 的 float32 平均」（C0，和 test02 同一個量，PREREG_02 第 9 節第 2 條），和已 commit 的 `results/prereg03/rank_{a,b}/dumps/val_p03{a,b}_d4.npz`、`val_p03{a,b}_s0_swap.npz` 算出的同一個量相比，最大差 ≤ tol；
     - 12 個 logits 完成標記和 npz 記的權重，依臂等於登記的權重；
     - 三組 × 兩臂 × {d4、對調} 的 12 個 logits 標記加 3 個 key 標記，GPU 型號、torch、cuDNN 全部相同（同一個執行環境）。
     - tol：這一臂 seam_val 的 d4 完成標記記的 GPU 名稱和 03 參照標記的完全相同時 1e-4，否則 1e-3。同一個執行環境下兩臂一定相同；`seam_check.json` 記兩臂的較大者。
     - 逐位元組是否相同、8 種對稱和對調的 logits 最大差、E 的最大差、規則 B 不同題數只記錄。
   - **本機重算 notebook 接縫**：`apply` 用已 commit 的 03 dump 再算一次兩臂的 C0（原方向、對調），算法是已 commit 的 `colab/test04_rank.ipynb` 裡的 `c0_of`、`question_spans`（和 test02 逐字相同），在 CPU 上用 torch float32 算。通過與否（整體和每一臂）要和 `seam_check.json` 相同，tol 要等於依標記重算的值（兩臂算出的 tol 不同也算不一致），不同給 1。本機值和 notebook 值的差只記錄。
   - **本機接縫**：用 test02 的 seam_val C5、`opp` 的 seam_val O4、test04 的 seam_val SW、test02 的 seam_val 開局分數，照 `frozen_config.json` 算 Q02 的驗證題機率，和 `results/prereg04/rank/rank_val_probs.csv`（`3c7ab658…`）、`pred_B.csv`（`27b6d225…`）比（`t02_rank.seam_compare`）：題號、順序、每題的盤數和可解析盤數相同，最大差 ≤ tol，規則 B 最多差 4 題；沒過給 1。`pred_A.csv`（`6a6bb3c2…`）只記錄是否逐位元組相同。
     - tol 取兩邊的較大者：test02 已 commit 的報告記的本機接縫 tol（C5 的 GPU 對 prereg02），和 test04 依標記重算的 tol（SW 的 GPU 對 03）；兩邊都在參照同型 GPU 上算才是 1e-4。
     - 先後：seam_val 上先核對 O4 Δκ 和 `opp` 記的逐位相同（不同給 4，第 7 條），再判本機接縫（沒過給 1），免得本機的問題用掉登記的那一次重跑。
   - 兩個 notebook 的 seam_val dump 都和參照逐位元組相同時，預期機率檔也逐位元組相同；不同時記警告、照接縫判定走。

6. **讀哪一份測試檔、key 一致**：
   - 04 的所有步驟都讀 test02 第 2 步打包的那一份（T1 判 NORMALIZED 時是正規化副本；本機 `data/test` 加 `MANIFEST.json`，Colab 是同一個 `C10_test_v1.zip`），絕不讀 `data/test/raw`（工具拒絕把 `data/local` 或 `raw` 資料夾當測試檔目錄）。
   - test04 notebook 記錄的測試包、本機的 MANIFEST、test02 notebook 記錄的測試包三者必須相同（`manifest_sha256` 和兩個競賽二題目檔的 SHA-256），不同給 2（不是 11/04 的測試包）。各組 `_INPUT.json` 記的題目檔 SHA-256 也要等於本機的題目檔。
   - test04 每一組的 key 陣列（題號、slot、執色、是否解析成功）必須和 test02 同一組的相同（dtype、形狀、值）。不同算接縫失敗（1，第 9 條）。
   - `apply` 核對 dump 的順序：key → 同一個執行環境 → 完成標記（SHA-256、大小、列數）和檔案相符 → 內容（權重、`view`、NaN 列等於沒解析的盤）。第一個不成立的那一關決定原因，都是 1。

7. **執色檢查**（第 5.2 節「每個對手方來源在 public、private 各自的 Δκ 都要 > 0」）：
   - 6 個數：public、private 兩組 × O4、SWA、SWB 三個來源。
   - 可用盤：SWA、SWB 和參照 A0、B0 用解析成功的盤；O4 和參照 T4 用解析成功、而且沒有重播失敗的盤。
   - 每盤向量取 float64 的 log_softmax（遮罩外的列先填 0，不進子題）。子題照 PREREG_03 rev 3 第 11 節：每題的可用盤依執色分成黑、白，≥ 2 盤的才算一個子題（官方每題單色，所以一題就是一個子題；萬一混了兩種執色也照這個規則拆）。κ 的定義同 PREREG_03 2.2 第 4 條。
   - Δκ＝所有子題上 κ(目標方參照) − κ(對手方來源) 的簡單平均（點估計；`p04_rank_fuse.val_side_kappa` 的 `dkappa`）。參照：O4 對 T4、SWA 對 A0、SWB 對 B0。
   - 6 個數都**嚴格 > 0**、而且各有至少 1 個子題才算通過；任一個 ≤ 0、是 NaN，或那一組沒有任何子題，04 不用（3），最終交 R20。這是題目內容的性質，**不觸發重跑**。不看信賴區間。
   - O4 的兩個數在 `opp` 就算出來並記錄（第 1 條）。`opp` 算出 public 或 private 的 O4 Δκ ≤ 0、NaN 或沒有子題時，**只在 seam_val 的 O4 檔案逐位元組相同、而且 seam_val 的 O4 Δκ 等於登記值時**，才是登記的判定（3；`apply` 用同一個定義只會得到逐位相同的值），不開 Colab。seam_val 漂移時，這兩個數是用沒過核對的重播算的，只記錄（`opp_report.json` 的 `o4_dkappa_not_judged_while_drift`），結束碼是 4；查好本機、重跑 `opp` 之後才判定。重播失敗 > 0.5% 不管有沒有漂移都是 3。
   - `opp` 判定兩個都 > 0 時只是「可能可用」：`apply` 再算一次，public、private 的 O4 Δκ 必須和 `opp` 記的逐位相同（不同給 4），6 個數的最終判定以 `apply` 為準。
   - seam_val 上同樣算：`opp` 的 O4 必須等於 prereg04 記的 `+0.1050265798301427`（差 ≤ 1e-12，不同給 4）；`apply` 的 seam_val O4 必須和 `opp` 記的逐位相同（不同給 4）。SWA、SWB 只記錄，並記和 prereg04 記錄值（`+0.03631175839656169`、`+0.034356617152967346`）的差。
   - 只記錄：黑題、白題各自的 Δκ，各盤數（5／10／15／20）的平均，每題重抽的 95% 區間，子題數，沒有子題的題數。
   - 已知限制：每組只有 400 題，「其實是正、因為抽樣才 ≤ 0」有可能發生。這只會造成改交 R20，不會讓錯的 04 上場。本修訂不引用機率的數字（見第 22 條）。

8. **輸出**（第 5.2 節第 3 點）：
   - 本機接縫過、第 3、7 條都成立時，才寫 `results/test04/rank/rank_test_probs.csv`（public 在前、private 在後，各依題目檔順序）、`pred_A.csv`、`pred_B.csv`、`decide_summary.json` 和 `submission_rank.csv`（`pred_B.csv` 的逐位元組複本），並對兩份題目檔跑 `check_submission --official`；報告 `t04_apply_report.json`，結束碼 0。報告另記這份交件檔和 R20 的 `results/test02/rank/submission_rank.csv` 規則 B 等級不同的題數（只記錄）。
   - 第 3 條或第 7 條不成立：只寫報告（`usable false` 和原因），結束碼 3，不留任何可以上傳的檔。
   - 本機接縫沒過、其他接縫類失敗（1）或本機漂移（4）：不留任何測試集輸出（`check_submission` 沒過的那一種是寫出後刪掉，見下一點），也不寫 `t04_apply_report.json`，改寫 `attempt_verdict.json`（結束碼、階段、原因；前提階段就停下的也寫），讓第 9 條的封存能核對原因。seam_val 的輸出（`rank/seam_val/`）照寫，只當紀錄。
   - 交件檔寫出後 `check_submission` 沒過：4，刪掉這次寫出的五個測試集輸出（`rank_test_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`submission_rank.csv`；`pred_B.csv` 和被拒的交件檔逐位元組相同），`attempt_verdict.json` 的階段是 `check_submission`，並記下刪了哪些檔。
   - 一盤都解析不了的題填 D（同 test02，PREREG_02 9.3），題號記進報告。
   - K6 已過、T1 FAIL_OPENING、`opp` 已判不用：`apply` 在寫任何檔之前給 3。

9. **重跑一次**（PREREG_02 第 9 節第 2 條、rev 6 第 10 條第 10、11 條、rev 7 第 11 條第 1 條的規則，套到 test04）：
   - **算接縫失敗、可以重跑的**（1；封存時都「占次數」）：
     - notebook 的接縫沒過（類別 `notebook_seam`）；
     - notebook 的執行紀錄最後一次，或在它之前，有 `seam_failed`、`readonly_changed`、`final_check_failed`（`notebook_run_log`；同一個 Drive 資料夾裡又跑了一次也在這裡）；
     - dump 不是全部在最後一次執行算的（接續過，`resumed`）；
     - 跑的不是已 commit 的 `colab/test04_rank.ipynb`（`not_committed_notebook`）；
     - `apply` 以 1 結束（`apply_exit_1`）：本機重算的 notebook 接縫和 notebook 的結論不同、test04 三組 dump 的執行環境不同、完成標記和檔案不符、dump 的內容或權重不對、key 和 test02 不同、本機接縫沒過。
   - 那次的測試輸出一律不用、不上傳；換一個執行階段重跑一次。test04 有自己的一次，和 test02 的分開算。
   - 「不是已 commit 的 notebook」只對照 repo 裡已 commit、乾淨、而且自身一致（`NOTEBOOK_SHA256` 那一行等於重算值）的那一份。repo 這一份不符合時是本機問題：`unpack`、`archive`、`apply` 都拒絕（2），不判成 1。
   - **由工具強制**：
     - `tools/t04_rank.py archive --reason TEXT` 自己從執行紀錄判斷失敗類別（同時有幾種時依 `bug_revision` > `before_r20_u1` > `notebook_seam` > `notebook_run_log` > `not_committed_notebook` > `resumed` > `apply_exit_1`，找到的都記進 `classes_found`），只封存上面這些接縫類的失敗和第 1 條「R20 上傳前跑的」。
     - 拒絕封存（2），有依第 20 條的修訂時例外：已經有登記的判定（`t04_apply_report.json` 已寫出，或 `opp` 登記判定 04 不用）；找不到接縫類的證據；已有 1 個占次數的封存（這一次同時是 R20 上傳前跑的也一樣）。
     - 另外三種也拒絕（2），修訂**不**解除（都是本機要先修好的前提）：正式模式下核對不了 R20 的第一次上傳；repo 的 notebook 沒 commit、不一致；`opp` 的報告是本機漂移（4，不是登記的判定：先查好本機、重跑 `opp`，再封存）。
     - 沒有 `attempt.json`、或沒寫 `counts_toward_rerun: false` 的封存，一律算占次數（保守的一邊）。
     - `apply` 每次以 1 或 4 結束都寫 `attempt_verdict.json`（第 8 條）。
   - **做法**：
     - 本機：`archive` 把 `results/test04/sw/`、`unpack_sw.json`、`rank/`（有的話）移到 `results/test04/failed/attempt_N/`（`opp/` 留著），`results/test04/` 裡的 zip 全部（含檔名帶 ` (1)` 的）移到 `data/test04_failed/attempt_N/`，寫 `attempt.json`（類別、是否占次數、每個 zip 和 `unpack` 讀的那一份 zip 的 SHA-256、各次執行的開始時間、seam_val 完成標記的時間、Drive 要改的名稱），並 commit。N 是封存的流水號，和類別無關。
     - Drive（使用者照 `archive` 印的完整名稱改）：占次數的封存 `sw` → `sw_failed_N`、`test04_rank_results.zip` → `test04_rank_results_failed_N.zip`；不占次數的（`before_r20_u1`、`bug_revision`）改成 `sw_exempt_N`、`test04_rank_results_exempt_N.zip`。然後開新的執行階段，整份重算。
     - **登記的重跑**（已有 1 個占次數的封存）**只在 G4**（第一次在 G4 或 A100 都一樣；不換到容許差較寬的 GPU，也避開 A100 沒實測過的差距）。三處用同一個判定、同一個完整名稱 `NVIDIA RTX PRO 6000 Blackwell Server Edition`：notebook 看到 Drive 上有 `sw_failed_*` 資料夾，這次的 GPU 名稱不是它就在 preflight 停下（不算一次執行）；`unpack` 在已有 1 個占次數的封存時，要求 15 個完成標記都是它，否則 2；`official_log` 對 `attempt.number` ≥ 2 的報告，要求 notebook 的 GPU 和完成標記的環境都是它（沒有環境紀錄也不算）。只有 `sw_exempt_*` 時照第一次的規則。G4 拿不到就隔一段時間再試，最晚 11/15 中午；還拿不到，這一季不用 04（K6）。
     - notebook 的「只准重跑一次」上限：Drive 上的 `sw_failed_*` 一律算一次（包括 Drive 上接縫本身通過、本機 `apply` 才判失敗的那種），`sw_exempt_*` 不算，其他改過名、自己有接縫類失敗紀錄的也算；已有 2 個就在 preflight 停下。最後以本機 `results/test04/failed/` 為準：Drive 上的名稱和本機封存對不上時，`unpack` 只印警告（`unpack_sw.json` 的 `drive_archives`），Claude 請使用者截圖核對。
   - `unpack` 核對新結果是重算的，排在所有紀錄核對之前：zip 的 SHA-256 和每一次封存記的都不同（包括 `unpack` 讀過、放在別處的那一份），算出輸出的執行（第 1 條的定義）每一次的開始時間都晚於最近一次封存的時間（同一個 Drive 資料夾在封存之前只停在版本確認、GPU 關卡或 preflight 的那幾次不算，例如下面「不是斷線」那種資料夾封存前，新的 `sw` 在 preflight 停下的那一次），新的 seam_val 完成標記的時間晚於封存記的那一份。任一不成立給 2，不收；所以已封存的 zip（包括接縫失敗的那一份）再解一次一律是 2，不是 1。已有 2 個占次數的封存時也給 2。
   - 還是沒過：停下查清楚（`archive` 拒絕再封存）；要放寬容許差只能另寫本登記的修訂，修訂前不上傳 04。K6 照樣適用。
   - **不算一次執行**：
     - notebook 在版本確認（沒有輸入 `y`）、GPU 關卡或 preflight 停下：還沒寫任何完成標記，同一個 Drive 資料夾照常可用。對不同的 12 碼仍輸入 `y` 而跑完的，照上面判「不是已 commit 的 notebook」（占次數）。
     - 中途斷線：**不接續**。先刪執行階段、等 1–2 分鐘，再把 Drive 的 `sw` 改名成 `sw_interrupted_N`（有 zip 的話同樣改名成 `test04_rank_results_interrupted_N.zip`），開新的執行階段整份重算，GPU 規則照被打斷的那一次。被打斷的資料夾裡已有接縫類失敗的紀錄時（notebook 在 preflight 停下說「不是斷線」），照接縫失敗處理：證據是 `test04_rank_results_interrupted_N.zip`（沒有 zip 時是那個資料夾的 `run_log.json`、`seam_check.json`、`config.json`），用 `archive --drive-folder sw_interrupted_N` 封存，Drive 上把那個資料夾改成 `archive` 印的名稱。
     - 最後一次有結果的執行是 `completed`、接縫通過、結果 zip 還在的資料夾（跑完後又按了一次全部執行）：notebook 用另一則訊息停下；不是斷線，不改名，直接用那份 zip。
     - notebook 在 R20 的第一次上傳之前跑（第 1 條）。
   - **不重跑、04 直接不用**：第 3、7 條不成立、T1 FAIL_OPENING、K6 已過（後兩種 `opp`、`apply` 每次都在寫檔之前給 3）。之後除非依第 20 條先 commit 指出程式錯誤的修訂，下面這些都拒絕（2）：
     - `opp` 已判 04 不用、或 `t04_apply_report.json` 已寫出：`archive` 拒絕封存；
     - `opp` 已有登記的判定（0 或 3）：`opp` 的重跑拒絕（CPU 重播是確定的）；
     - `t04_apply_report.json` 已寫出（0 或 3）：`apply` 的重跑拒絕。上一次 `apply` 是 1 時，`apply` 的重跑也拒絕：要照上面封存、重跑 notebook；那一次已經是登記的重跑（已有 1 個占次數的封存）時沒有第三次，照上面「還是沒過」停下，不封存（K5 閘門第 2 輪）。
   - **停下查本機、不重跑 Colab**（4）：第 2 條的 seam_val O4 不逐位元組相同（失敗 ≤ 0.5%）或 seam_val 的 Δκ 不同、比率表的 SHA 不同、第 7 條 `apply` 和 `opp` 的 O4 Δκ 不同、第 18 條的 `.venv` 版本不同、第 8 條的 `check_submission` 沒過。查好後 `opp` 或 `apply` 加 `--rerun --reason` 重跑；`opp` 判 4 時 `apply` 拒絕（2），要先重跑 `opp`。比率表的 4 和 `.venv` 的 4 在寫檔之前就停下，`opp` 不留報告。計數檔不見或 SHA 不同是 2，不是 4：照第 17 條還原後再跑同一個指令（`opp` 已有登記的判定時，`apply` 也不必、也不能重跑 `opp`）。

10. **T1**：
    - 04 的工具先讀 test02 的 `results/test02/checks/rank/t1.json`：FAIL_OPENING 給 3（04 不上場，照 test02 用 R10，第 2.5、5.3 節；不擬合、也不使用 Q02 的無開局版，它只記錄）；FAIL_STRUCT 給 2（照 test02 停下）。
    - 其餘照 test02 的核對：完整的原始檔 T1（正式模式）只接受 PASS 和 NORMALIZED。
    - 04 另外要求 test02 的競賽二是 R20 的正式執行（`fallback_used` 為否，不是 R10 備案）、test02 的本機接縫通過、`check_submission` 是 official 而且沒有錯誤，`apply_test_report.json` 和 `opening_test.npz` 已 commit。

11. **K6「一次成功的正式執行，已 commit」的定義**（第 4 節）：以下全部成立才算。
    - test04 notebook 在正式的測試包上跑完（`config.json` 的 `smoke`、`rehearsal` 都是 false，已 commit 的 notebook 只在測試包 MANIFEST 是 `kind official` 時這樣記；而且測試包和本機的 MANIFEST、test02 記的相同，第 6 條。工具不另外讀 `kind`）：執行紀錄 `completed`、前面沒有失敗的執行、所有 dump 都在最後一次執行算的（沒有接續）、在 R20 第一次上傳之後才跑、notebook 接縫通過；`unpack` 的結果已 commit。
    - `opp` 是正式模式、結束碼 0（seam_val 逐位元組相同、seam_val 的 Δκ 相同、三組重播失敗都在第 3 條的範圍內、public 和 private 的 O4 Δκ 都 > 0），已 commit。
    - `apply` 是正式模式、結束碼 0：本機接縫通過，6 個 Δκ 都 > 0，`check_submission --official` 通過，交件檔和報告已 commit。
    - 報告最後一次 commit 的 committer 時間（`git log -1 --format=%cI -- results/test04/rank/t04_apply_report.json`）≤ 2026-11-16 23:59:59 +08:00。上傳可以在 K6 之後，但要在 11/23 之前。
    - 跑完但第 3 條或第 7 條不成立的那一次不算成功。

12. **最後一版**（第 5.3 節，由 `tools/official_log.py` 的 `prepare`、`final`、`status` 強制）：
    - `prepare` 一份 04 的檔之前，log 裡要先有 R20 的第一次上傳（`results/test02/rank/submission_rank.csv`，`uploaded`）。
    - 一筆 04 的上傳紀錄「符合第 5.3 節」＝以下全部成立：H4R 採用（`results/prereg04/RESULTS.md`、`e8b24f2`）；第 11 條成立；T1 PASS 或 NORMALIZED；6 個 Δκ 都 > 0；三組重播失敗都在範圍內（seam_val 0 盤，而且逐位元組相同）；報告最後一次 commit 就是 `prepare` 記的那一個，而且 ≤ K6；報告和交件檔從 `prepare` 之後沒有變；報告記下的輸入（`inputs_bound`：test02 的報告、開局分數、三組的 C5 dump 和 key、T1、`frozen_config.json`，以及 04 自己的 `opp` 報告和 O4、`unpack_sw.json`、notebook 的紀錄和 dump、本機接縫紀錄）現在已 commit 的內容都沒有變；notebook 的各項核對都是真、`earlier_failed_runs` 是空的；登記的重跑只在 G4；封存紀錄符合第 9、20 條（最多 1 個占次數的封存；不占次數的只能是 `before_r20_u1` 或 `bug_revision`；存有 `t04_apply_report.json` 的封存要有修訂）。條件都用報告裡的數字和檔案重算，不信任報告自己的結論，`prepare`、`final`、`status` 每次重算。
    - 有任何一筆符合，最終選擇是 04；否則是 R20（T1 FAIL_OPENING 時 R10）。
    - `final` 選 04 時，現在不符合就拒絕。
    - 選 R20／R10（`final`，或在 04 符合時 `prepare` 一份 R20／R10 的檔）：只要有一筆 04 符合就拒絕，**除非先依第 20 條 commit 一份本登記的修訂（rev 4 起）**，指出具體的程式錯誤和它破壞的是第 5.3 節哪一條，並用 `--not-04-reason "PREREG_04 rev N: …"` 引用。工具核對：理由寫出「PREREG_04 rev N」、N ≥ 4、`docs/PREREG_04_OPPONENT.md` 已 commit 且乾淨、HEAD 的本檔有以 `| rev N |` 開頭的那一列、加進那一列的 commit 晚於每一筆被推翻的 04 紀錄的 `prepared_at`。rev N、它的 commit 和時間、理由記進 `official_scores.json`，也寫進結果。只寫一段文字、沒有修訂，一律拒絕。04 符合全部條件時沒有其他方法改交 R20。
    - `status`：有符合的 04、最終卻不是 04 時，印「違反 PREREG_04 5.3」。最終是 R20／R10 而且帶著修訂時，要現在每一筆符合的 04 都在那份修訂涵蓋的清單裡、修訂的 commit 晚於那一筆的 `prepared_at`，才不算違反（和 `final` 同一個時間規則）。還沒設最終選擇時印「最終選擇要設成 04」。現在沒有任何一筆 04 符合、而且最後一次上傳或最終選擇是一筆不符合的 04 時，印「final 改 R20，再 prepare 並重傳 R20」。還有別的 04 符合時不印這一行：最終選擇是不符合的那一筆時印「違反 PREREG_04 5.3」；最終選擇符合、最後一次上傳卻是不符合的那一筆時，印一般的「最後一次上傳不是它」。
    - 已經傳過 04、之後才發現不能用：工具算出不符合的（例如輸入變了），最終選 R20，並再傳一次 R20，讓它成為最後一次上傳；是找到程式錯誤的，先照上一點寫修訂，再同樣重傳。
    - `docs/TESTDAY.md`（在 R20 的指紋裡，不能改）U5 寫「兩場都是 prereg02（甲）」；對競賽二，這一行由本條取代。04 判定之前不執行 `final --comp rank`；已經設過也沒關係，照 `docs/TESTDAY_04.md` 第 7 步重跑 `final`，11/22、11/23 各跑一次 `status`。
    - public 分數不進判定；比上一版低 0.05 以上時只追查錯誤，查不到錯誤就維持本條的結果，並寫進報告（VALIDATION 紀律 4）。04 和 R20 的交件檔同名：一筆 04 和一筆 R20／R10 的 public 分數完全相同時，`public` 印警告，請使用者核對上傳紀錄。
    - 最後一次上傳在 11/23 23:59 前完成。04 的上傳屬於 PREREG_02 第 9 節第 12 條的「事前登記依本機驗證規則採用的方法」；競賽二每天最多 5 次，R20 和 04 合計。每一次上傳前，使用者都要在對話裡對那一份檔明確說 OK。

13. **本機假測試演練**（K5 的「本機假測試演練通過」；紀錄在 `results/test04/design/build_review.md`）：以下全部通過、並經獨立審查，才合併回 main。
    - **R20 逐位元組不變**：用 `t02_fake_tests.py from-val` 的假樹跑 R20 的 `opening`／`apply` 演練（`tests/test_t02_dryrun.py -k FromValPipeline`）。7 個 CSV（`submission_rank.csv` 是 `d9b9801b…`）、刪掉 `input` 鍵後的兩份 `decide_summary.json`（它記了輸入的絕對路徑）、`opening_test.npz` 逐陣列的 SHA-256，要和寫 t04 程式之前記下的 `results/test04/design/r20_fake_golden.json`、和合併當下的 main 都相同（兩個一次性 clone 先後用同一個暫存路徑跑）。`git diff r20-testday-pre04` 在 39 個檔上只有 `tools/official_log.py` 的新增行，0 行刪除。
    - **`official_log.py` 的行為不變**：tag `r20-testday-pre04` 版和新版在同一份假紀錄上各跑一遍 R20 的 `prepare`（第一次上傳）→ `uploaded` → `public` → `final` → `status`，也各跑一遍 T1 FAIL_OPENING 時 R10 的同樣流程（旁邊有、沒有 04 的樹各一次）：紀錄檔、輸出逐位元組相同，回傳值相同。例外只有 `status`：輸出多一行 K6 倒數，回傳的 dict 多一個 `rank.prereg04`（`entries` 是空的；測試比對前先拿掉這個鍵）。
    - **04 對得回 prereg04**：同一棵假樹加上 `t04_fake_tests.py from-val` 的假 test04 輸出（seam_val 是 03 驗證 dump 的逐位元組副本，public、private 是驗證題第 1–400、401–800 題的列切片），跑 `unpack`、`opp`、`apply` 的演練模式：seam_val 的 O4 檔案逐位元組相同；本機接縫的三個檔等於 prereg04 的 `rank_val_probs.csv`、`pred_B.csv`、`pred_A.csv`；seam_val 的三個 Δκ 等於 prereg04 的記錄值；交件檔 800 列等於 prereg04 `pred_B.csv` 前 800 列（只換題號前綴），通過 `check_submission --official`；t04 的每一步前後，test02 的輸出、測試檔都沒有變。
    - **負面演練**（每一項都斷言結束碼）：環境混用、npz 改了但標記沒改、key 和 test02 不同、notebook 接縫沒過、前面有失敗的執行、接續過、本機重算的 notebook 接縫不一致（原方向、對調各一次）、tol 和標記不符、第一次在 A100（tol 1e-3）、本機接縫沒過、tol 取兩者較大、Δκ＝0（對調換成 A0）、`opp` 記的 O4 Δκ 差 1e-15、`o4_<組>.npz` 在 `opp` 之後被改、重播失敗 25／5,000 和 26／5,000 的邊界、seam_val 1 盤失敗、小量失敗盤的遮罩、測試包不同、舊 zip 再解一次（包括封存過的接縫失敗 zip）、封存第 2 次、已有 2 個占次數的封存、重跑不是 G4、`archive` 時 R20 的紀錄沒 commit、R20 上傳前只停在 preflight 的那一次、repo 的 notebook 沒 commit 或不一致、`--bug-revision` 早於這次失敗或重複引用、`--allow-failed` 用在正式模式或 repo 裡、`unpack_sw.json` 和 `failed/` 對不上、zip 的 CRC 壞了、內部錯誤、`archive --drive-folder` 寫錯：都照第 5–9、20 條被拒絕、判不用或判漂移。
    - **完整測試**：`PY -m pytest -q tests`、`C10_SLOW=2` 的完整測試、`C10_T04_GATE=1` 的三個防誤改測試（`test_p04_hygiene.py`、`test_t02_hygiene.py`、`test_t04_hygiene.py`）、合併前在分支上的 `C10_K5=1 PY -m pytest -q tests/test_t04_hygiene.py`，全部通過。
    - **test04 notebook 的本機 CPU 冒煙**：L2b（隨機權重的替身）和 L2（prereg01 → 02 → 03 兩臂 → test02 → test04 整條冒煙，接本機 `unpack` → `opp` → `apply --smoke`）：兩臂的 dump 和 03 的函式逐位元組相同、接縫通過。冒煙模式下接縫和 Δκ 的正負只記錄，重播失敗率照判。
    - **正式模式演練**：rev 3 和程式 commit 之後、合併之前，在一次性 clone 裡用 from-val 的假樹、已 commit 的程式實跑正式模式的 `opp` → `unpack` → `apply` → `official_log.py prepare`／`uploaded`／`final`／`status`，另外演練：沒有 R20 的第一次上傳時 `opp` 給 2；`archive` 時紀錄沒 commit 給 2；A100 的重跑 `unpack` 給 2；只在工作樹改 notebook 時 `unpack` 給 2；`archive --bug-revision` 引用早於這次失敗的修訂給 2、引用晚於的修訂給 0 並印 `sw_exempt_N`。clone 的 `data` 不能整個接到 repo 的 `data`（正式模式會寫 `data/test`、`data/test04_failed`）。
    - 假樹是驗證棋譜：只跑重播和 apply，不跑任何內容統計。
    - Colab 彩排是選做，不是 K5 的條件；做的話結果記進 `build_review.md` 或測試日的結果。彩排發現需要改規則的事，只能在 11/04 前（還沒有任何測試資料）另寫修訂。

### 11.2 只是實作說明

14. **`official_log.py` 怎麼認得 04**（第 5.1 節「新方法名稱、認得 04 的檔」）：
    - 04 的檔以資料夾認：`results/test04/rank/` 裡的競賽二檔，綁在同一個資料夾的 `t04_apply_report.json`（不叫 `apply_test_report.json`：萬一辨識失敗，R20 的路徑會因為報告不存在而拒絕，不會把 04 當成 R20）。另外要已 commit 而且通過的 `results/test04/rank/seam_val/seam_local.json`、`results/test04/opp/opp_report.json`（正式模式、`passed`、`usable_so_far`）、`results/test04/sw/seam_check.json`，以及 test02 的報告和 T1。條件用報告裡的數字重算，不信任報告自己的結論。
    - 方法名稱「prereg04 Q02 = R20 + O4 + SWA + SWB, rule B (甲)」在新程式裡設定；給 `--method` 時要寫到 `Q02`。上傳紀錄多一個 `test04` 欄；OK 請求多印「這一份是 04，不是 11/04 那份 R20；兩份在 800 題裡有 N 題的規則 B 等級不同」（N 由 `official_log` 自己算）。
    - **偏離 `results/prereg04/design/plan.md` 第 7 節**：不在 `RANK_COMBOS` 加項。改那一行會違反「只准附加」；另寫一行重新定義又會改變 R20 的 `method_for` 行為。這個偏離在 `check`（`00acde5`）之後才記，開頭「偏離規格要在 `check` 之前記進修訂紀錄」的時點規則不適用：那條管的是 `check` 之前的工具；測試日的部分由第 5 節「細節在 K5 前另寫修訂」規範，本修訂就在 K5 前。而且第 5.1 節的原文只要求「新方法名稱、認得 04 的檔」，不改 `RANK_COMBOS` 和登記本身一致，只偏離 plan。
    - log 裡沒有 04 的紀錄時，R20 的 `prepare`／`uploaded`／`public`／`final` 輸出和 JSON 都和原本逐位相同；`status` 多一行 K6 倒數。對 tag 版只有加行（652 行新增、0 行刪除），只用標準函式庫，不 import `t04_rank`。

15. **notebook 的輸出位置和 Drive 名稱**：Drive `C10/results/test04/sw/`（彩排 `test04_rehearsal`），zip `C10/results/test04/test04_rank_results.zip` 的成員從 `sw/` 起，本機解到 `results/test04/sw/`。用 `sw` 而不是 `rank`，是為了讓 test02 的解壓工具一定拒收這個 zip。改過名的資料夾有三種：`sw_failed_N`（占次數的封存）、`sw_exempt_N`（不占次數的封存）、`sw_interrupted_N`（斷線）；zip 同樣改成 `test04_rank_results_failed_N.zip`、`test04_rank_results_exempt_N.zip`、`test04_rank_results_interrupted_N.zip`，讓「還沒看到新檔就等」的保護重新有效。

16. **產生器**：`colab/build_test04_rank_nb.py` 讀已 commit 的 `colab/test02_rank.ipynb`（`53337a7a…`）和 `colab/prereg03_rank_a.ipynb`（`85e0c2ae…`；另核對 `colab/prereg03_rank_b.ipynb`（`aa3669ac…`）和它只差 `ARM` 那一行），依格子的 `c10` metadata 挑格子、只做有紀錄的替換，並用 ast 核對沿用的函式逐字相同；不 import 任何凍結的產生器，也不 import `tools/p0*`。釘死的 SHA 和 03 的執行環境都從 repo 檔讀出再和常數比；另外用 ast 讀 `tools/t04_rank.py` 的 `G4_GPU`、`tools/official_log.py` 的 `T04_RERUN_GPU`，核對三處的 G4 名稱相同。重跑產生器逐位元組相同；`--upload-copy` 另寫上傳副本 `data/colab_upload/test04/test04_rank.ipynb`（資料夾裡只有這一個檔，SHA 要等於已 commit 的 notebook）。

17. **計數檔**：`data/prereg04/counts_fit.npz` 不在 git 裡。測試日第一步先核對 SHA（`f6c6debe…`；`opp`、`apply` 也再核對一次，不見或不同時給 2，照這一條還原後再跑同一個指令）；遺失時先用同 SHA 的 `data/prereg04/design_scratch/counts_fit.npz`，兩份都不見才照 `p04_opp.py count` 在 repo 外重算（至少 7 個 worker，登記是 8 個，約 34 分鐘），SHA 必須等於 `f6c6debe…`。

18. **換行與環境**：`.gitattributes` 加 `results/test04/** -text`（理由同 PREREG_02 rev 7 第 7 條），K5 閘門第 1 輪另加 `tests/test_t02_hygiene.py` 以原始 SHA 釘的 7 個 `results/prereg02` 文字檔（逐檔，不整個資料夾：工作樹裡其他 `results/prereg0*` 文字檔是 CRLF，整個資料夾設 `-text` 會讓它們變成已修改）；`.gitignore` 加 `results/test04/*.zip`、`results/test04_rehearsal/`。`opp`／`apply` 的報告記 Python、numpy、scipy、zlib 的版本。Python（`sys.version` 全文）、numpy、scipy 要和 `results/prereg04/opp/val_report.json` 記的相同；zlib 要等於 `1.3.1`（`val_report.json` 沒有記 zlib，這個值是 9/30 設計階段用 `zlib.ZLIB_RUNTIME_VERSION` 實測的）。不同給 4，停下查本機。最後的把關仍是第 2 條的 `o4_seam_val.npz` 檔案 SHA。

### 11.3 會影響測試日判定的（續）

19. **R20 側的已登記變動**（擴大原本的 D5）：包括 PREREG_02 第 9 節第 11 條的 D5（官方定義變動）、PREREG_02 rev 9 起的修訂、rev 7 第 2 條競賽一改 A100，也包括打分紀錄帳依別的登記新增的列。
    - R20 的檔怎麼改，由 test02 依 PREREG_02 決定（R20 的登記優先），改之前先告訴使用者。本登記的任何修訂都**不授權** R20 的任何變動。指紋、tag `r20-testday-pre04`、`tests/test_p04_hygiene.py` 都不動。
    - 04 的閘門（`docs/TESTDAY_04.md` 第 1 步，`C10_T04_GATE=1` 跑三個防誤改測試）：失敗的只能是 `test_p04_hygiene.py` 的指紋核對，和／或 `test_t04_hygiene.py` 的 `test_ledger_and_prereg_rows_pinned`（打分紀錄帳和 PREREG_01–03 要已 commit、乾淨，打分紀錄帳的資料列是 `6ec5e68` 的 25 列、`| rev N |` 列和 `6ec5e68` 相同，多出來的列印在失敗訊息裡）；而且多出來的檔正好是一份已 commit 的 PREREG_02 修訂列出的檔，多出來的每一列都已 commit、對得上一份已 commit 的登記。逐檔、逐列記進 RESULTS，不會自動算過。
    - 這個列數核對只在 `C10_T04_GATE=1` 或 `C10_K5=1` 時跑：一般的完整測試（`docs/TESTDAY.md` 11/04 前的「全綠」就緒清單）略過它，所以別的登記依規打分多出來的列，不會讓 R20 的就緒閘門變紅。
    - `test_t04_hygiene.py` 另外對 PREREG_01–03、打分紀錄帳、`docs/COLAB_RUN.md` 要求 `6ec5e68` 的每一行照順序還在；`results/prereg01`–`prereg04` 對 `6ec5e68` 不能有任何差異（已 commit 的也算），也不能有未追蹤的檔。
    - 只動到競賽一的檔（檔名含 `player`）或只多了打分紀錄帳的列：不需要本登記的修訂，04 照常做。
    - 動到 R20 路徑的檔：04 的所有步驟停下，直到本登記的修訂 commit：列出改了哪些檔、改前後的 SHA，附 `git diff r20-testday-pre04`，證明只改了那份 PREREG_02 修訂寫的內容（D5 是檔名表）。04 的工具和 notebook 需要跟著改時同樣只改那一項，並重建、冒煙、審查。修訂 commit 之後照上面的閘門判定，不是「全過」。修訂前不上傳 04；R20 照 PREREG_02 的規則上傳，不等 04。

20. **找到 04 的程式錯誤**：屬於 PREREG_02 第 9 節第 12 條的「修 bug」。
    - 先另寫本登記的修訂（rev 4 起），指出具體的程式錯誤和它影響第 5.3 節、本節哪一條；修正不得改變來源、參數或組合。
    - 這份修訂是工具唯一接受的翻盤依據：`archive`、`opp`／`apply` 的重跑在登記的判定之後、`final`／`prepare` 在 04 符合時改選 R20，都要引用它（`--bug-revision N`／`--not-04-reason "PREREG_04 rev N: …"`）。工具核對：N ≥ 4；`docs/PREREG_04_OPPONENT.md` 已 commit 且乾淨；HEAD 有以 `| rev N |` 開頭的那一列；加進那一列的 commit 晚於它推翻的紀錄（`opp_report.json`、`t04_apply_report.json`、`attempt_verdict.json` 的 `created`，已有的占次數封存的 `archived_at`，或 04 上傳紀錄的 `prepared_at`）。
    - `archive --bug-revision N` 另外要求：rev N 在**這一次失敗之後**才 commit（晚於算出輸出的執行的開始、每個完成標記、`unpack_sw.json`、`attempt_verdict.json` 的時間；一個時間都找不到就拒絕），而且一份修訂只換一次不占次數的重跑（較早的封存已經引用同一個 N 就拒絕，新的錯誤要寫下一個 rev）。這樣封存的類別是 `bug_revision`，不占次數，Drive 改名 `sw_exempt_N`。`official_log` 在 `prepare`／`final`／`status` 重算：有修訂的封存，那一列在 HEAD、commit 晚於封存記的每個時間，沒有兩個封存引用同一個 N。
    - 修好後照第 11 條重跑受影響的步驟（notebook 有錯才重跑 notebook；這不算第 9 條那一次重跑，但要記錄；GPU 照第 9 條：沒有占次數的封存時照第一次的規則，已有 1 個占次數的封存（Drive 上有 `sw_failed_*`）時只用 G4）。K6 之後才修好的，這一季不用 04；已上傳的 04 有錯，照第 12 條以 R20 為最終並重傳。
    - R20 依 PREREG_02 修 bug、重跑 test02 之後，已做的 04 因為輸入變了（第 12 條的 `inputs_bound`）不再符合第 5.3 節，最終 R20。`opp`／`apply` 的重跑只接受本登記的修訂（rev 4 起），由那份修訂自己說明為什麼可以重做；不能直接引用 PREREG_02 的修訂。

### 11.4 只是實作說明（續）

21. **結果寫在哪**：`results/prereg04/**` 凍結，測試日 04 的結果寫在 `results/test04/RESULTS.md`（各步驟的數字、Δκ、重播率、接縫、GPU、斷線和重跑的次數與原因、上傳紀錄、第 12 條的判定、看過官方分數才做的任何選擇），由另一個代理核對數字。

### 11.5 揭露

22. 本修訂在 H4R 之後、看過驗證集分數才寫。它只補程序，不改任何會影響驗證分數或選擇的東西。
    - O4 的 Δκ 提早在 `opp` 算，只是為了在 04 一定不上場時省下一次 Colab。
    - 原文的讀法：
      - 第 3 條把失敗盤排除在 E_O4 之外，也排除在 O4／T4 的 κ 子題之外（同一個遮罩 `parsed & ~replay_failed`）；第 5 條本機接縫的 tol 取兩邊較大者。這是第 2.1、5.2 節原文的讀法；驗證集上都和 prereg04 的做法完全相同（驗證題 0 盤失敗），不改任何驗證集的數字。
      - 第 5 條：第 5.2 節「`seam_val` 要重現已 commit 的驗證 dump」以 C0（每題可解析盤 softmax 的 float32 平均）判定，照 test02（PREREG_02 第 9 節第 2 條；第 5.2 節「樣式照 test02」）；原始 logits 和 E（進 Q02 的 float64 log_softmax 平均）的最大差只記錄。不改任何驗證集的數字。
      - 第 2、3 條把「任一組重播失敗 > 0.5%」照字面套到 seam_val（04 不用），seam_val 較小的差異另外當本機漂移處理。
      - 第 7 條：seam_val 漂移時不判 public、private 的 O4 Δκ（只記錄，查好本機再判），因為那時的 O4 沒過第 5.2 節要求的 seam_val 核對。
      - 第 1、9、20 條：R20 上傳前跑的那一次、依修訂推翻的那一次，都「不算第 9 條那一次重跑」。讀成：封存和改名照第 9 條，那一次不占次數；前面沒有占次數的封存時，下一次的 GPU 照第一次的規則（G4 優先、A100 備用），前面已有 1 個占次數的封存時，下一次仍是登記的重跑，只用 G4。「只在 G4」只看有沒有占次數的封存（Drive 上的 `sw_failed_*`），三處同一個判定。較嚴的另一種讀法（不占次數的封存之後也只用 G4，可以擋住「接縫在 G4 失敗、因為在 R20 上傳前跑而不占次數、下一次改用容許差較寬的 A100」）沒有採用；要採用只能另寫修訂，notebook、`unpack`、`official_log` 三處一起改。這個讀法也表示：不占次數的封存（每一次 R20 上傳前跑的、每一份修訂換的一次）之後的那一次執行，接縫照樣判定，所以除了第 9 條那一次重跑，還可能多出判接縫的執行。第 5.2 節只寫「樣式照 test02」，test02 沒有這兩種情況；這是本修訂的讀法，不是更嚴。
      - 第 19 條：R20 側依已 commit 的 PREREG_02 修訂改檔時，第 5.1 節的防誤改測試（`test_p04_hygiene.py` 的指紋核對）會一直失敗。本條讀成：失敗的只能是那份修訂列出的檔，逐檔核對、記進 RESULTS 後 04 照常做；只動競賽一的檔時不需要本登記的修訂，動到 R20 路徑的檔時先寫本登記的修訂。這是第 5.1 節的硬規則遇到 R20 依自己的登記改檔時的處理，不是更嚴。
      - 以上讀法都不動來源、組合、參數、閘門、H4R、截止點或門檻。
    - 設計階段的調查筆記（`results/test04/design/understand_p04.md` 4.3）曾用本機驗證題的**題目內容**（沒有答案）估過「每組 400 題時 6 個 Δκ 檢查因抽樣而誤判」的機率。這違反「驗證玩家的棋不做統計」；那個數字沒有用在任何規則或門檻上，本修訂不引用，在此揭露。

### 11.6 實作和規格的差異：`tools/t04_rank.py`

- 規格（`results/test04/design/spec.md`）是設計文件，不是登記。11.6–11.9 記實作和它不同的地方，都不改科學內容；會影響判定的已寫進第 1–13、19、20 條，這裡只記來由和細節。
- **a. `opp` 判定的先後**（規格 3.2 寫「先看 3 再看 4」）：重播失敗 > 0.5% 照舊先判 3；public、private 的 O4 Δκ ≤ 0 只在 seam_val 可信時判 3，漂移時判 4、只記錄（第 7 條）。報告多 `exit_code`、`verdict_kind`、`o4_dkappa_not_judged_while_drift`，重跑規則靠前兩個判斷。
- **b. `apply` 遇到 `opp` 判漂移**（規格 3.3 寫一律 3）：給 2，要先查好再 `opp --rerun`；只有登記的「04 不用」給 3。這樣本機漂移不會被誤當成登記的判定。
- **c. seam_val 上的先後**（規格沒定）：先核對 O4 Δκ 和 `opp` 相同（4），再判本機接縫（1）。
- **d. 交件檔的 `check_submission` 沒過**（規格沒定義）：4，刪掉這次寫出的五個測試集輸出（`build_review.md` 第 5 節以前只刪 `submission_rank.csv`，`pred_B.csv` 等四個檔會留下）；`attempt_verdict.json` 因此多一個階段 `check_submission`，`details.removed` 記刪了哪些檔。
- **e. 前面有失敗的執行**：`apply` 在所有模式都給 1（規格 3.3 寫「real」、8.3 寫「拒絕」，照較嚴的 8.3）。`unpack` 對最後一次執行是接縫類失敗的，repo 外也給 1（規格寫 repo 外只警告）；最後一次之前有失敗的執行，repo 外仍然只警告。本機把 test02 的 `repro_failed` 也當接縫類狀態（沿用 `t02_rank.FAILED_RUN_STATUSES`；test04 notebook 不會寫這個狀態）。
- **f. `unpack` 的核對順序和多的核對**：新舊核對、占次數的封存數、重跑的 G4 排在所有紀錄核對之前；讀紀錄之前先查 zip 的 CRC（壞了給 2）；成員副檔名只收 `.json`、`.npz`、`.csv`、`.txt`、`.log`；`completed` 但 `seam_check.passed` 不是真給 1；`config.format` 必須是 `c10-test04 rank notebook config v1`，否則 2（`apply` 也核對）；缺成員的檢查在紀錄檢查之後，所以缺成員的接縫失敗 zip 仍然給 1；解壓到一半出錯時刪掉已經解出的檔。接縫類失敗（1）的訊息：第一次寫「先 archive」；已有 1 個占次數的封存時（登記的重跑）改寫「沒有第三次」，指向第 9 條的「還是沒過：停下」（`archive` 一定拒絕；K5 閘門第 1 輪，`apply` 同）。`apply --rerun` 在上一次 `apply` 是 1 時的拒絕（2）也照同一個規則分兩種訊息：第一次寫「archive 後重跑 notebook」，登記的重跑指向同一則「沒有第三次」（K5 閘門第 2 輪）。
- **g. 「R20 先」的時間**（規格寫 `run_log.runs[0].start`）：改成「算出輸出的執行」（第 1 條），`unpack` 和 `archive` 共用同一個函式。`unpack` 的新舊核對（第 9 條）也改用同一個定義（K5 閘門第 1 輪；規格 3.1 寫 `runs[0].start`：「不是斷線」的流程之後，登記的重跑在同一個 `sw` 裡跑，`runs[0]` 是封存前 preflight 停下的那一次，重跑一定被判「不是重算的結果」）；`unpack_sw.json` 的 `freshness` 記 `producing_runs`，`runs0_start` 只記錄。
- **h. `archive`**：
  - 類別的優先順序和 `classes_found`（第 9 條）。
  - 核對不了 R20 的第一次上傳時拒絕（規格寫記成 `before_r20_u1`）；已有 1 個占次數的封存時，`before_r20_u1` 也不跳過拒絕。
  - `opp_report.json` 的 `usable_so_far` 是 false 時（規格一律拒絕），分成兩種：登記的判定照規格拒絕（修訂可以解除）；本機漂移（4）另一則訊息拒絕，修訂不解除，要先重跑 `opp`（`build_review.md` 第 5 節以前兩種用同一則「04 已經不用」的訊息，漂移也能被修訂解除）。
  - 另記 `unpack` 讀的那一份 zip 的 SHA-256（`zips` 裡的 `unpacked:<檔名>` 和 `unpacked_zip`），不管它放在哪裡。
  - Drive 名稱依類別（`sw_failed_N`／`sw_exempt_N`；規格一律 `sw_failed_N`），記在 `attempt.json` 的 `drive_rename`，印出給使用者的完整名稱和下一次的 GPU 規則（K5 閘門第 1 輪起看這次封存之後有沒有占次數的封存，不只看這一次算不算次數；記在 `attempt.json` 的 `next_run`）；新選項 `--drive-folder sw|sw_interrupted_K`（只給 `archive`）。
  - `--bug-revision` 的時間規則和一份修訂只用一次（第 20 條）。
  - 也能在非正式模式跑（要給 `--results-dir`，zip 搬到 `R4/failed_zips/attempt_N/`），規格寫只有正式模式；`R4/failed/` 裡有封存時，`unpack` 在任何模式都做新舊核對和次數核對，「R20 先」仍然只在正式模式。
- **i. 「不是已 commit 的 notebook」**（規格只寫比 SHA）：只對照已 commit、乾淨、自身一致的 notebook（第 9 條）。
- **j. 重跑只在 G4**（規格只寫在使用者步驟）：`unpack` 核對 15 個完成標記（第 9 條），`unpack_sw.json` 記 `rerun_gpu_check`。`apply` 本身不再核對，靠 `unpack` 的紀錄和 `official_log` 的重算：`sw/` 在 `unpack` 之後被改，要到 `prepare` 才擋。
- **k. 錯誤處理**：未預期的內部錯誤給 2 並印 traceback（Python 預設的 1 會和接縫類失敗撞碼）。`git` 的輸出一律用 UTF-8 解碼：原本借用的 `t02_rank.git` 用系統編碼（這台電腦是 cp950），讀中文的 PREREG_04 會失敗，`--bug-revision` 一定以內部錯誤結束；改成 `t04_rank.py` 自己的 `git()`，凍結的 `t02_rank.py` 不動。計數檔不見或和 `count_report.json` 不同（第 2、17 條）：`opp`、`apply` 都給 2，訊息寫照第 17 條還原（`build_review.md` 第 5 節以前：`opp` 不見時是內部錯誤、不同時只印 `p04_opp` 的訊息；`apply` 請使用者「重跑 `opp`」，而 `opp` 有登記的判定之後重跑要修訂）。
- **l. 本機重算接縫的來源**：`c0_of`、`question_spans` 從 `colab/test04_rank.ipynb` 取出（兩者在整本 notebook 裡各只有一個定義）；非正式模式下檔案不存在時改用 `colab/test02_rank.ipynb`（`c0_of` 相同），記在 `source_notebook`；正式模式缺檔就拒絕。notebook 的 tol 記兩臂的較大者，本機要求兩臂 tol 相同（不同判不一致，1）；兩臂的 03 參照 dump 在同一型 GPU 算，實際上不會發生。
- **m. 答案檔字樣**：用 `"ans" + "wer"` 比對整個絕對路徑，範圍比規格的 `"ans" + "wers"` 大。
- **n. 報告欄位**：`sets.<組>.anomalies` 是有異常的盤數；失敗率用整數算（第 3 條）；`inputs_bound` 的鍵在所有模式都用正式模式的 repo 路徑，另有 `inputs_bound_paths` 對到實際檔案，共 33 項；`questions_without_parsed_game` 每組列出題號；`unpack_sw.json` 多 `rerun_gpu_check` 和 `drive_archives`（Drive 上的 `sw_failed_*`、`sw_exempt_*` 對本機封存換算的名稱，只記錄，對不上印 WARNING：Drive 改名打錯一個數字時拒收，會浪費一次正確算完的執行）。
- **o. 冒煙模式**（`--smoke`；規格 5.3「冒煙：全部只記錄」）：C5 的權重改用 test02 冒煙 config 記的 SHA，兩臂權重用 sw 的 config；notebook 接縫、本機 Q02 接縫、seam_val 逐位元組核對、6 個 Δκ 的正負都只記錄（報告的 `smoke_record_only`）；重播失敗率照判；`opp` 三組都截到冒煙 key 的長度；`check_submission` 用題目檔的冒煙子集、非 official。規格 8.5 L2 的順序寫成 `unpack` → `opp` → `apply --smoke`。
- **p. `--allow-failed`**（只准 `--dry-run`、repo 外的資料夾）：也把正式 zip 的模式旗標改成警告，只供檢視。

### 11.7 實作和規格的差異：test04 notebook 和產生器

- **q. 執行紀錄的狀態**：開始時記 `running`；下一次執行時，還是 `running`、沒有 `end` 的舊執行改記 `interrupted`；preflight 停下記 `preflight_failed`；GPU 關卡停下時維持 `running`，之後被改成 `interrupted`；結尾格失敗記 `final_check_failed`，照樣打包後停下。
- **r. 比規格嚴的檢查**：preflight ② 在正式執行時也核對 03 參照標記的 torch（`2.11.0+cu128`）和 cuDNN（`91900`）；各組格子要求 dump 狀態是 `written`（不接續）；記 `net_state_sha256`（只記錄）。
- **s. 版本確認**（第 4 條；規格沒有）：輸入框在 `NOTEBOOK_SHA256` 那一行之後、任何寫檔之前；最後一行再印一次版本，只記錄。
- **t. preflight ⑥ 的次數和 G4**（規格寫任何接縫失敗的 `sw_*` 都算）：照第 9 條，`sw_failed_*` 一律算、`sw_exempt_*` 不算；有 `sw_failed_*` 時只收 G4 的完整名稱（函式 `rerun_gpu_problem`，SMOKE 不查）；名稱含 `RTX PRO 6000` 但不是完整名稱時用另一則訊息（開頭「G4 的名稱和 03 記的不同」）。新增 ⑥-3a：已經跑完的資料夾另一則訊息停下，不當斷線。
- **u. 產生器**：`bfloat16` 只准出現在沿用的 `state_dict_sha256` 裡（只用來判斷 dtype），整本沒有 autocast（規格 4.2 和 10.2 在這點上互相衝突，產生器和測試都用 ast 核對）；`ARMS`、`DUMP_NAMES`、`W_LABEL` 改在 preflight 定義，dump 工具格只 assert 相同；拿掉 test02 的 `dump_questions`（test04 用不到）；`guard_write`、`record_readonly_after` 只改註解，用去掉註解後的 token 核對；diff 檔另附對照 test02 原格的 diff；多一個 `--upload-copy`。
- **v. `config.json` 的欄位**：`sets.<組>.questions` 是標籤（`bundle local/<檔名>`、`tests/<檔名>`），不是絕對路徑，絕對路徑在 `dumps.<組>.questions_path`；`rows` 是處理的題數（冒煙 12），`questions_count` 是題目檔總題數，盤數在 `dumps.<組>.rows`。另記 `attempt`、`prior_sw_folders`、`changes_from_bases`。
- **w. 冒煙替身 `colab/make_test04_smoke_tree.py`**：建之前先跑 `t02_checks.py t1 --dry-run --only T1.1,T1.2,T1.3`（兩場）；另寫一份標 `stand_in` 的 test02 冒煙紀錄，讓 preflight ⑤ 在 L2b 也有東西核對；核對功能是同一個檔的 `--check`。

### 11.8 實作和規格的差異：`tools/official_log.py`（只加行）

- **x. 比規格 9.2 多的**：seam_val 必須 0 盤重播失敗；缺 `attempt.json` 或沒寫 `counts_toward_rerun: false` 的封存算占次數；不占次數的封存只能是 `before_r20_u1`、`bug_revision`；有修訂的封存重算時間和只用一次（第 20 條）；`attempt.number` ≥ 2 只收 G4（讀 `t04_rank.py` 實際寫的扁平 `notebook.env`，巢狀的也讀，沒有 env 不算）；`final` 的 PREREG_04 5.3 那兩行（判定結果和「這一筆還沒上傳」）只在 log 有 04 紀錄、或給了 `--not-04-reason` 時印，所以 log 沒有 04 紀錄、也不帶這個選項時，R20 的輸出逐位元組不變（沒有 04 紀錄卻給了 `--not-04-reason` 時，判定結果那一行的最終選擇印成 `None`，只是顯示）；還沒設最終選擇時 `status` 印「最終選擇要設成 04」，不印違反；帶著涵蓋它的修訂的 R20／R10 最終選擇不算違反；OK 請求自己算規則 B 差異題數，也記報告裡的值；`--not-04-reason` 在 `prepare`、`final` 都有；輸入變更的提示（`T04_REDO_HINT`）寫明「最終 R20；要重做 04 須先寫並 commit PREREG_04 修訂」。
- **y. 已知限制**：`official_log` 不能自己驗證 `before_r20_u1` 的封存真的在 R20 上傳前跑；它只核對不占次數的封存類別。這個判定只在正式模式的 `t04_rank.py archive`（已 commit 的紀錄）。

### 11.9 實作和規格的差異：測試、演練和文件

- **z. 防誤改測試**（規格 0.3、10.4 寫打分紀錄帳、PREREG_01–03、`docs/COLAB_RUN.md` 和 `6ec5e68` 相同）：放寬成「每一行照順序還在」，另加閘門時才跑的列數核對（第 19 條）；`docs/COLAB_RUN.md` 恰好有一節 `# 測試集 04（test04）`，`6ec5e68` 沒有；`results/prereg01`–`04` 另用 `git diff 6ec5e68` 核對，已 commit 的改動也抓得到。原因：R20 側依已登記的修訂改檔時，逐位元組相同的規則會讓 04 的閘門永遠過不了。
- **aa. golden 的算法**：`opening_test.npz` 逐陣列比 `numpy.ascontiguousarray(a).tobytes()` 的 SHA-256，另記 dtype、shape；npz 成員的 zip 時間固定，所以整檔不同只會來自 `meta`（建立時間、絕對路徑）。指令多了 `-p no:cacheprovider` 和執行緒上限。規格 8.1 第 1 點寫 `results/test04/design/` 還沒追蹤，實際上已在 `9f63acd` commit，所以 golden 跑之前、之後的 `git status` 都是全空。
- **ab. 時間**（規格 8.5 的推估；9/30 實測）：L2b 9.7–23 分鐘（同時段有別的工作時較慢）；L2 約 50–100 分鐘，視同時段負載而定（安靜時 notebook 鏈約 47 分鐘、連本機工具約 50 分鐘；9/30 晚間 K5 閘門實測：步驟時間合計 84 分鐘，含一次崩潰重跑的牆鐘 98 分鐘），要挑沒有別的重 CPU 工作的時段跑；`C10_SLOW=2` 完整測試安靜時約 27 分半，同時段有別的工作時 56–72 分鐘，要單獨跑（規格 8.4：可用虛擬記憶體至少約 12 GB，記憶體不足時會出現 MemoryError 連鎖失敗）。test04 notebook 在 G4、A100 上的時間還沒實測。
- **ac. 使用者步驟（只改文件和 notebook 的文字）**：斷線時先刪執行階段、等 1–2 分鐘，再改名；競賽一只有「開著的執行階段」才擋 test04，只是在等 G4 不算；刪電腦「下載」裡的舊 zip、Drive 改名，都自成一則訊息；不要從 Colab 的「最近」清單開 notebook；跑完後不要再按全部執行；第二次彩排前先把 Drive 上的彩排資料夾改名成 `…_1`，收拾改用 `old_rehearsal_2`；彩排後改了產生器，舊的 `test04_rank.ipynb` 改名成 `test04_rank_OLD.ipynb`；Drive 空間不足時不自己刪檔或清垃圾桶；「出狀況時」各列用紅字的開頭分辨。
