# 事前登記 10：競賽一的防護修正（全黑池第 3 層改交 06-OFF）和選做的 KataGo P1 融合

**白話版**：官方競賽一的候選和題目都只有黑棋。PREREG_02 的切法防護裡，第 3 層（ρ 低於 θ₂ 就「只用 CNN」）原本是為「候選多半黑、題目多半白」那種執色分開的切法設計的，全黑資料不可能出現那種切法。10/10 的探索（只用 fit 玩家）發現：如果題目和候選不共用開局（nov4），06 本身還有 0.9251，但 ρ 會掉到 θ₂ 以下、20 池全落第 3 層，照現行登記要改交「只用 CNN」，實際只剩 0.2042。

本登記做兩件事：

- **A（主要，一定做，不打分）**：全黑的池子落在第 3 層時，改交 06-OFF，不交「只用 CNN」。本機驗證池在第 1 層，A 在驗證集的輸出和 06 逐位元組相同，所以不用打分。
- **B（選做，有閘門）**：在 06 上再加一項 KataGo HumanSL 中性探針（P1）的 LDA 項。先在官方形狀的模擬池（只用 fit 玩家）定權重 w、過閘門（4090 約 7–8 小時）；過了再在驗證集打一次分（競賽一第 3 次，共 4 次），採用條件是「不比 06 差」；測試日兩池再用 4090 約 5.5 小時。B 主要是保險：題目開局和候選不同、或盤數少的時候，它能加分。**LDA 只用競賽一自己的模擬池棋局學，不用競賽二算的任何特徵值。**

使用者在 10 月不用做任何事。測試日最多多 1 次上傳（10）。

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。競賽一的前幾份：
  - PREREG_02：Q04，0.8830（H2P）。
  - **PREREG_06：06，0.9681（H6P，+0.0851 [+0.0609, +0.1097]，採用）**，`results/prereg06/RESULTS.md`。本登記的對照組。
  - 競賽一 4 次打分已用 2 次（H2P、H6P）。
- 探索（只用 fit 玩家、沒打分、沒碰 `data/local`）：`results/prereg10/explore/`，程式和 JSON `03b067e`，報告 `REPORT.md` `fc39967`。
- **只做競賽一（TEAM_11249）。** 10 的任何輸出都不進競賽二，也不用競賽二的特徵值、測試資料或輸出。
- 代號：
  - 06-OFF、06-ON、R02、V0–V3、CSLSk、`k*`：同 PREREG_06。
  - θ₁、θ₂、ρ、b_c、b_q、第 0–3 層：同 PREREG_02 第 2.2 節、rev 5–8。
  - **T_P1**：P1 的 LDA 項（第 2.1 節）。
  - **10A**：06-OFF，但 A 成立的池子改用「06 照第 1 層計算」的列。
  - **10B**：10A 再加 w*·z(T_P1)（第 2 節）。
  - **10**：測試日交的那一份檔，每一池是 06-OFF、10A 或 10B 的列（第 4.2 節）。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `b9b954d` | 第一版，2026-10-10。commit 時的狀態：① 探索的程式、JSON 和報告已 commit（`03b067e`、`fc39967`），A 的動機（nov4）和 B 的設計（P1 用 LDA、w 很大）都是**看過探索才定的**（第 7 節）。② 模擬池的 P1 抽取還沒開始，驗證玩家的 P1 特徵沒算過。③ 打分紀錄帳沒有 `prereg10` 列，競賽一在 H6P 之後沒打過分。④ 沒有任何測試資料，test02、test06 都沒有正式執行過。⑤ 主 session 定案：LDA 改用競賽一自己的模擬池棋局學（起草時原本用競賽二的 probe 特徵；為避開「各隊不可共享特徵值」的疑慮而改）。 |
| rev 2 | `4c2c15a` | B 的程式（`tools/p10_fuse.py`、`tools/p10_apply.py`、`tools/p10_p1.py` 的新增部分、`tests/test_p10_*.py`）和登記的實作差異，寫在第 11 節。commit 時：selfcheck、模擬池 P1 抽取、`mock` 都還沒跑；驗證玩家的 P1 沒算過；打分紀錄帳沒有 `prereg10` 列。 |
| rev 3 | `1f43c01`（merge） | 測試日的修訂（第 12 節），2026-10-10，分支 `p10-testday`：只做 A；A 的判定細節（T4 核對、guard 重算、t06_pool.json 的時效）、seam 和 submit 的定義、結束碼、K5-10 的判法（本列的 commit 時間）、`official_log.py` 加的規則；B 的掛鉤回「B 沒有採用」。 |
| rev 4 | `66b2ed4`（修正 `398b7c1`、`aa359eb`、`cb678ba`；合併 `6993f90`） | B 的測試日修訂（第 13 節），2026-10-11，分支 `p10b-testday`：`t10_player.py p1`（`seam_val`＝GPU 接縫、`public`／`private`）、`p1-rehearse`（量 m）、seam 重算驗證集的 10B、apply 在用了 06 的池加 w*·z(T_P1)、K6B-10、K5-10（B）、`official_log.py` 加的規則。commit 時：B 還沒採用（驗證玩家的 P1、apply、稽核、H10P 都還沒做，`results/prereg10/RESULTS.md` 不存在），所以 B 的路徑全部照 rev 3 回結束碼 3；**tol_P【待演練】**（m 還沒量）；沒有任何測試資料、沒有打分、沒有用 GPU。 |
| rev 5 | 分支 `p13-testday` 的第一個 commit（以 `git merge --no-ff` 合併進 main；K5-13 看 main first-parent 上第一次出現本列的 commit） | 「B 採用」也可以由 PREREG_13 判定（第 16 節），2026-10-13。commit 時：PREREG_13 的 H13P 已打分、判定 B′ **採用**（`results/prereg13/RESULTS.md`、`player/score_H13P.json`，`f761454`）；PREREG_10 的 H10P 沒跑、`apply_report.json` 仍是 `score_allowed: false`（不改）；t10 的程式還沒改（本列先 commit）；tol_P 仍【待演練】；沒有任何測試資料、沒有用 GPU。 |

## 0. 為什麼做這個（都沒看答案；探索在 fit 玩家上看過很多次，數字偏樂觀）

- **第 3 層的原意**：PREREG_02 rev 2 加第 3 層，是因為「執色分開」的模擬（候選多半黑、題目幾乎全白）只用開局只剩 0.2082。rev 8 第 9 條已經寫明官方全黑時這種切法和執色條件都不會出現，但「ρ < θ₂ 時第 3 層」照舊保留。
- **探索**：全黑、官方形狀、20 池（`REPORT.md` 第 1 節）。

| 情境 | E0（只用 CNN） | Q04 | V2（06） | ρ／防護層 | 06 照登記實際交的 |
|---|---|---|---|---|---|
| base | 0.4189 | 0.9411 | 0.9887 | 1.009／第 1 層 | 0.9887 |
| nov6 | 0.3790 | 0.8904 | 0.9806 | 0.872／第 2 層 | 0.9806 |
| nov4 | 0.2042 | 0.7129 | **0.9251** | **0.679 < θ₂ 0.8268／20 池全第 3 層** | **0.2042** |
| c20、big800、late、wr | 0.32–0.43 | 0.81–0.94 | 0.94–0.99 | 第 1 層（c20 沒記） | — |

- 全黑時，每一種情境的 E0 都遠低於 V2；E0 對 fit 玩家又是樣本內（CNN 訓練時看過他們），真實分數只會更低。在全黑資料上，第 3 層把 0.93 換成 0.20，沒有任何情境支持這個換法。
- **B 的先導**（200 位 fit 玩家，`REPORT.md` 第 2 節）：P1 用 LDA 投影後單用 0.935；疊在 06 上，每題 5／10 盤 +0.033 [+0.023, +0.045]，每題 2 盤 +0.118，開局新穎 +0.050。先導的 LDA 學在競賽二的 probe 上；本登記改學在競賽一自己的模擬池棋局（第 2.1 節），效果要在模擬上重新判（第 2.3 節）。官方形狀下 06 已 0.9887，最多只剩約 +0.011。

## 1. A：全黑池的第 3 層改交 06-OFF

### 1.1 什麼時候用（每一池各自判斷，以下全部成立）

1. T1 是 PASS 或 NORMALIZED（FAIL_OPENING 時 06 本來就整個不用，A 也不用）。
2. **全黑**：test02 那一池 `apply_report.json` 記的 `b_c_candidate_black_share` 和 `b_q_question_black_share` 都**剛好是 1**（登記的算法 `p02_player_open.colour_share`，也就是 PREREG_02 rev 8 判定用 T3b 的同一個條件）；另外核對 T4 記這兩個檔是全黑。
3. **第 3 層是 ρ 造成的**：`guard_level` 是 3；test02 的閘門通過；`effective` 組合有開局項；那個方向的 ρ 算得出來，而且小於 `results/prereg02/player/frozen_config.json` 凍結的 θ₂。t10 用報告裡的 ρ 和凍結門檻重算一次層級，要和 `guard_reason` 一致，不一致是結束碼 2。ρ 一律用 test02 已算好、已 commit 的值，不重算（探索的 ρ 是近似值，只當依據）。
4. 06 在這一季可用：H6P 採用；06 的 `feats`（三池）、`seam` 以結束碼 0 跑完並 commit；這一池的 `t06_pool.json` 是用現在這份 test02 報告和 seam 做的（同 PREREG_06 rev 3 第 8 條的判斷）。

### 1.2 那一池怎麼算

- 照 06 的第 1 層計算：sr06 在 V2／V3／V0（有 V1 時也算）裡選，照 PREREG_06 2.4 節；權重不變；CSLS`k*`（10）。
- 06 閘門：sr(選中) − sr(R02′) ≥ 0。R02′ 是 test02 那一池的 `effective`（防護改之前的組合，通常是 Q04），不是 `applied`（第 3 層換成的只用 CNN；對它比，閘門會白白通過）。沒過：這一池維持 test02 的列。
- 程式只 import `tools/t06_player.py`、`p06_*.py` 的函式（`load_pool_data`、`pool_store`、`compute_pool` 等），讀 t06 已算好的 `feats.npz`（三池的 `feats` 不看防護層級都會算），不改任何一支。
- 這一池不准開一對一（PREREG_06 5.3 的 06-ON 條件 (e)「兩池都在第 1 層」不改），所以有 A 的時候不會有 06-ON。

### 1.3 不用 A 的情況（照原登記）

| 狀況 | 怎麼做 |
|---|---|
| 第 1、2 層 | 不變（06 照常） |
| 第 0 層（test02 選中只用 CNN 的組合，或 test02 閘門沒過） | 不變（test02 的列）。這兩種是候選池自我檢索的結果，不是切法造成的，探索沒有看過 |
| 第 3 層，但不是全黑（b_c 或 b_q 不是 1），或執色條件觸發 | 不變（只用 CNN）。混色池的第 3 層正是原本設計要擋的情況 |
| T1 FAIL_OPENING／FAIL_STRUCT | 照 PREREG_06 2.7（06 不用／停下） |
| 06 本身不能用（t06 結束碼 3；結束碼 2、4 沒解決） | A 不用，最終照 PREREG_06 5.4 |
| 06 閘門對 R02′ 沒過 | 這一池維持 test02 的列 |

### 1.4 為什麼不用打分

- 本機驗證池是隨機切，防護在第 1 層（H6P 的 `apply_report.json`）。A 只改第 3 層的行為，所以 10A 在驗證集的輸出和 06 **逐位元組相同**，打分的差一定是 0。這一點由 t10 的 seam 檢查強制（第 4.2 節）。
- A 真正改變的情況（全黑且 ρ < θ₂），驗證集不可能出現，只能靠 fit 玩家的模擬（第 0 節）。這是揭露的限制（第 7 節）。

## 2. B：10A＋w*·z(T_P1)（選做；過閘門才往下）

### 2.1 特徵和項（寫死）

- **P1 特徵**：照 PREREG_08 第 1.3 節，用 `results/prereg08/p1/p1_lib.py`、`p1_extract.run_chunk`（import、不改；KataGo 程式和權重同 PREREG_08 第 1.2 節）。每盤取目標方（side 0）的 770 維平均：候選棋是那位候選，題目棋是被問的那位；官方全黑，所以都是黑方；驗證池是那一盤的 `color`／`color_i`。抽取兩方都算（`p1_lib` 不改），只用 side 0。
- **LDA 投影（只用競賽一自己的資料）**：學在第 2.2 節模擬池抽出來的 fit 玩家黑棋上（候選、base、nov4 三種盤都可以用）；不用競賽二的 P1／P2／P3 probe 特徵。可用盤：沒失敗、有目標手；只用 ≥ 2 盤的玩家。用這些盤的平均、標準差（＋1e-6）標準化，再 `p06_style.lda_fit`（reg 1e-3、64 維）。模擬時每池排除池內 400 人重學；凍結時（第 2.4 節）用全部 926 人。
- **T_P1**：每位候選、每一題，把投影後的各盤取群組平均，算 `p06_style.lda_term`（負的平方距離，同 06 的 L 項），再每題對全部候選做 z 分數。失敗的盤不進平均；一盤都沒有的列，z 記 0（同 p02）。
- **合併**：F_B＝F_sel＋w*·z(T_P1)，再 CSLS`k*`。F_sel 是那一池 06（或 10A）選中組合在做 CSLS 之前的分數。依 C 由大到小取 Top-5，同分時照候選代號。

### 2.2 模擬池（官方形狀；只用 fit 玩家；4090）

- **玩家**：黑棋 ≥ 120 盤的 fit 玩家，共 926 人（`docs/DATA.md`）。
- **每人的盤**（`tools/p10_p1.py sets`）：黑棋依 sha256('c10-prereg10|mock|{player_id}|{game_id}') 排序：前 100 盤是候選，接下來 20 盤是 base 題目庫；剩下的盤裡，前 4 手（8 種對稱取最小，同 `x0_open_keys.py` 的鍵）沒出現在自己 100 盤候選裡的，依同一順序取前 ≤ 20 盤，當 nov4 題目庫。最多 140 盤／人，約 11.1–13.0 萬盤，4090 約 6.8–7.9 小時（以先導的 1,012 局面/秒、每盤約 222 個局面估計）。`sets.json` 的 SHA 在抽取前 commit。
- **池**（seed 0–39，每個 seed 每種情境一池）：400 人照 seed 抽；400 題，盤數 5／10／15／20 各 100 題，照 seed 洗牌分給每人，一題對一人。
  - **base**：題目取 base 題目庫的前 k 盤。
  - **nov4**：題目取 nov4 題目庫的前 k 盤；不夠就全用（k_eff < k）；沒有新穎棋的人，照 `x1_scenarios.py` 的規則把題目給池裡還有新穎棋的人。
  - 只記錄、用在 G3：**wr**（題目有放回抽）、**c20**（候選只用前 20 盤）。
- **06 的項**：照 `x1_scenarios.py`（`p06_sim`、`p06_style`，import）。CNN 用 `data/prereg06/cnn_fit/black_ge105.npz`（樣本內），06 的 LDA 每池排除池內玩家重學。

### 2.3 w 的選法和閘門（`tools/p10_fuse.py mock`；只判一次）

- **selfcheck**（移植核對，在 `mock` 之前）：正式工具在先導資料上、用先導的配方（競賽二 probe 學的 LDA，只當移植核對、不進任何選擇），重算 `x3_p1_eval.json` 的 `V2`、`V2+P1ldaBW@w`（全部格點），差 ≤ 1e-9。不符就停下查；查明是數值路徑差異、不是邏輯錯誤時寫 rev 2，以 repo 算出的數字為準。
- **格點**：w ∈ {0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 48}。
- **選 w\***：seed 0–19 的 base 池平均分最高的 w；和最高差 0.0005 以內的，取最大的 w（保險偏向 P1）。
- **閘門**（seed 20–39 留出；差＝10B(w*) − V2；95% 區間用玩家 bootstrap：每位玩家在留出池的平均差，2,000 次，`default_rng(0)`）：

| 閘門 | 條件 |
|---|---|
| G1 base | 平均差 ≥ +0.003，而且 95% 下界 > 0 |
| G2 nov4 | 平均差 ≥ +0.020，而且 95% 下界 > 0 |
| G3 不傷 | wr、c20 各自的平均差 ≥ −0.002 |

- G1 的 +0.003：官方形狀下 06 已約 0.989，剩約 0.011；B 的主要價值在 G2（開局改變時的保險），base 只要求確實有正的增益。
- **只記錄、不可選**：每個 w 在每種情境的分數；E0、Q04、V2；k_eff 的分布；每池的 ρ 近似值。
- 沒過：`results/prereg10/RESULTS.md` 記「B 未採用」，commit；10 只有 A。

### 2.4 凍結（`p10_fuse.py freeze`）

- 用模擬池全部 926 人的可用盤學 LDA，寫 `results/prereg10/player/p1_lda.npz`（μ、σ、投影），和 `frozen_config.json` 一起 commit。
- `frozen_config.json` 記：w*、閘門紀錄、LDA 用的盤數和人數、06 凍結檔、`p1_lib`／`p1_extract` 釘住檔的 SHA、KataGo commit、權重 SHA、中性 metadata SHA、工具 SHA、`.venv` 版本。
- 從這之後才准在驗證玩家的棋上算 P1。

### 2.5 驗證集的 apply（`p10_p1.py extract --set val`、`p10_fuse.py apply`）

- **抽取**：只讀 `data/local/player_val_candidates.csv`、`player_val_questions.csv` 兩份棋譜檔（同 PREREG_06 rev 2 ④），在 import `p1_lib`（會關上 `data/local`）之前讀完，同 `tools/p08_p1.py` 的做法。約 44,935 盤，4090 約 2.7 小時。塊的 SHA 寫進 `results/prereg10/feat/val_report.json`，commit。
- **先重現 06**：用 06 的程式重算驗證池的 06 主檔，要和 `results/prereg06/player/player_val_top5.csv` 逐位元組相同，不同就停下。
- **池閘門**（不看答案）：在驗證池的 sr06 同一批假題上算 sr(10B) − sr(06 選中)，要 ≥ 0。沒過：B 這一季不用，記為「未送分」，保留打分機會。
- 寫 `results/prereg10/player/player_val_top5.csv`、`apply_report.json`。`score_allowed` 全部成立才為真：06 逐位元組重現；P1 失敗盤 ≤ 0.5%；池閘門過；`check_submission.py player … --questions … --candidates …` 過；另一個代理的稽核（看不到答案、不看分數；獨立重算 T_P1 和 Top-5，相同）已 commit。

### 2.6 H10P：打一次分，和採用條件（不劣性）

- **指令**（寫死；`score_allowed` 為真才跑）：

```
.venv/Scripts/python.exe tools/score.py compare-player results/prereg06/player/player_val_top5.csv results/prereg10/player/player_val_top5.csv data/local/player_val_answers.csv data/local/player_val_candidates.csv --ledger results/score_ledger.csv --note "prereg10 H10P (A = prereg06 06 main, reference re-read)" --json results/prereg10/player/score_H10P.json
```

- **對照組重讀**：A 側平均要等於 `score_H6P.json` 的 B 側平均（差 ≤ 1e-12），不符就停下。
- **採用條件**：B−A（10B − 06）的點估計 ≥ 0，**而且** 95% 下界 > −0.010（`score.py` 的配對 bootstrap，10,000 次、seed 0）。
- **為什麼用不劣性**：① B 的增益已在官方形狀模擬的留出 seed 判過（G1、G2），驗證集的任務是抓「P1 在沒看過的玩家、真實流程上會不會壞掉」；② 驗證池是隨機切、一題對一人、黑白混合，06 已 0.9681，B 的主要價值（開局改變、盤數少）在這種池子量不到；③ 400 題的檢定力有限，用「下界 > 0」幾乎一定放棄一個在模擬上證明的保險；④ 點估計 ≥ 0 加下界 −0.010，代表最多容許約 1% 的真實損失，換取開局改變時約 +0.05 的保險（模擬）。
- **合理性門檻**：10B 低於 0.95 時先查 bug，要指出具體的程式錯誤才算。
- **預算**：競賽一第 3 次（剩 1 次）；查 bug 重跑也算，並記帳。帳上多 2 列；這 2 列會讓哪幾個防誤改測試變紅、怎麼允許，在打分前的 rev 2 寫死（照 PREREG_06 rev 3 第 28 條的寫法）。

## 3. 截止點（台灣時間；沒趕上就照表做）

| 截止點 | 條件 | 沒趕上 |
|---|---|---|
| K1-10（無日期） | rev 1 的 commit 早於模擬池的任何 P1 抽取，也早於驗證玩家的任何 P1 抽取 | 這一季不做 B |
| K2-10 10/21（三）23:59 | `sets`、selfcheck、模擬抽取、`mock`（閘門判定）都 commit | 這一季不用 B |
| K3-10 10/25（日）23:59 | B 過閘門時：`freeze`、驗證抽取、`apply`、稽核、H10P、`RESULTS.md` 都 commit | 這一季不用 B |
| K5-10 10/31（六）23:59 | 測試日擴充（A 一定、B 採用時）、測試日的修訂、演練、合併回 main | A 沒趕上：這一季不用 A，照 PREREG_06；B 沒趕上：這一季不用 B |
| K6B-10 11/14（六）23:59 | B 採用時：兩池的 P1 抽取和 GPU 接縫已 commit | 10 只有 A |
| K6-10 11/21（六）23:59 | 10 在官方測試集有一次成功的正式執行（結束碼 0、報告已 commit） | 最終照 PREREG_06 5.4 |

## 4. 測試日

### 4.1 保護舊路徑（rev 1 就生效）

- 一律不改：R20、04、06、07、08、09 的測試日路徑（各自的指紋和 tag）；`tools/p0*_*.py`、`t02_*`、`t04_*`、`t06_*`、`t07_*`、`t08_*`、`t09_*`；`results/prereg01`–`prereg09/**`、`results/test02|04|06|07|08|09/**`；`results/prereg08/p1/**`；`third_party/katago/**`；`results/prereg10/explore/**`（只准加檔）。
- 只准加行：`tools/official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_02/06`（本登記不改它們的原文）、本登記。
- **t10 動工前，順序寫死**：① 做 09 路徑的指紋 `results/test09/design/p09_path_fingerprint.json`，打 tag `p09-testday-pre10`，加 `tests/test_t09_path_hygiene.py`；② 記下 Q04、06-OFF、06-ON 的假測試演練輸出，和競賽二各路徑的演練輸出當基準（`results/test10/design/baseline_pre10.json`），合併前要逐位元組相同；③ 開分支 `p10-testday`（worktree 放 `competitions\` 以外）。

### 4.2 要做的東西

- **`tools/t10_player.py`**：
  - **`seam`**：在 seam_val 上用 t06 的 seam 輸入重跑。A 不觸發（第 1 層），10A 的 Top-5 要和 t06 的 `top5_off.csv` 逐位元組相同。B 採用時，用已 commit 的驗證 P1 塊重算 10B，要和 `results/prereg10/player/player_val_top5.csv` 逐位元組相同。不符：結束碼 4。
  - **`apply --pool public|private`**：讀 test02、t06 那一池的報告，照第 1.1 節判 A，寫 `t10_pool.json`（每一條件的值）和 `top5.csv`。A 成立用第 1.2 節；否則用 t06 的 `top5_off.csv`（06 退回 test02 的池就是 test02 的列）。B 採用、K6B-10 前已有這一池的 P1 塊和接縫時，加上 w*·z(T_P1)；池閘門 sr(10B) − sr(選中) ≥ 0、P1 失敗盤 ≤ 0.5% 都成立才用，否則這一池不加 B。B 只加在「用了 06（含 A）」的池。
  - **`p1 --pool public|private`**（只在 B 採用時；`.venv-gpu`、4090）：抽兩池的候選和題目，每池約 4.5 萬盤、約 2.7 小時。開跑前核對 `nvidia-smi` 沒有別的計算程序。STOP 檔讓出 GPU，分段存檔、可以接續。**GPU 接縫**：重抽驗證池前 400 盤，和已 commit 的驗證塊比，T_P1 投影後的每盤最大差 ≤ tol_P；tol_P＝max(0.05, 2.5 × m)，m 在拿到測試資料前用 4090 演練量好，寫進測試日的修訂。沒過：B 不用（結束碼 3 只對 B），10 只有 A。
  - **`submit`**：兩池的列（public 接 private，列順序照 test02）組成 `results/test10/player/submission_player.csv` 和 `t10_submit_report.json`，要過 `check_submission.py --official`。兩池的列都和這一次 06-OFF（沒有 06-OFF 時和 Q04）逐列相同時，不產生交件檔（結束碼 3「10 不需要」）。報告記每池用的是 06-OFF／10A／10B／test02，和 06-OFF、Q04 第 1 名不同的題數，`inputs_bound`。
  - **`status`**：只讀，列出還差哪一步和 K6B-10、K6-10 的倒數。
  - **結束碼**同 t06（0 完成、2 拒絕、3 登記的「不用」、4 本機漂移）。**前提**同 t06 rev 3 第 4 條，另加：t06 的 `feats`、`seam` 已以 0 跑完並 commit；test02 重跑後 06 從 `feats` 重做時，10 也從頭重做。
- **`tools/t10_fake_tests.py`**：做全黑、官方形狀、nov4 式的假池（讓 test02 落第 3 層），以及負面演練：b_q ≠ 1；第 0 層；T1 FAIL_OPENING；t06 seam 沒過；B 的 GPU 接縫沒過。
- **`tests/test_p10_*.py`、`tests/test_t10_*.py`（含 `test_p10_hygiene.py`）、`docs/TESTDAY_10.md`**。輸出只寫 `results/test10/`，大檔放 `data/test10/`；演練寫到 repo 外。
- **`tools/official_log.py` 只加行**：認得 `results/test10/player/`，方法名稱「prereg10 10: 06-OFF + all-black level-3 lift (A) [+ P1 LDA (B)] (甲)」；`prepare` 的前提：Q04 已上傳；有 06-OFF 時 06-OFF 已上傳；有 06-ON 時 `veto.json` 已 commit，結果不是 `kept`；log 裡沒有 10 紀錄時，輸出和 tag `p09-testday-pre10` 的版本逐位元組相同。

### 4.3 GPU（B 採用時）

- 測試日 4090 的順序：競賽二的 07 HS（11/04 夜）和 08 P1（11/05）優先（沒有備案、截止較早）；之後才跑 t10 `p1`（約 5.5 小時，排夜間）。
- 10 月的模擬和驗證抽取：開跑前 `nvidia-smi`；10/24 前 C4 要用就建 STOP 檔讓出，等空了再續跑；超過一小時排夜間，告訴使用者。10/10 另一個專案（`C:\rsna_env`）用過 4090，開跑前一律檢查。

### 4.4 上傳順序和最後一版（競賽一）

- **順序**：Q04（11/04）→ 06-OFF →（06-ON → 否決）→ 10。每一份都要使用者在對話裡對那一份說 OK。兩池都退回 test02 的列時沒有 06 檔，順序是 Q04 → 10。
- **10 什麼時候傳**：06-ON 沒有產生；或 06-ON 被否決、傳錯、核對不出、K7' 已過。**06-ON 判定 `kept` 時不傳 10**（06-ON 要求兩池都在第 1 層，A 不會出現；B 沒有和開關一起驗證過）。
- **一筆 10 的上傳「符合」**：A 有效（K5-10 成立）；K6-10 成立；T1 是 PASS 或 NORMALIZED；10 的 seam 通過；報告和交件檔從 `prepare` 之後沒有變；報告綁的輸入沒有變。
- **最後一版**：06-ON 照 PREREG_06 5.3／5.4 是最終 → 06-ON；否則有符合的 10 → 10；都沒有 → 照 PREREG_06 5.4（06-OFF → Q04）。改選其他的，只能用 `--not-10-reason "PREREG_10 rev N: …"`（N ≥ 2，核對方法同 PREREG_06 rev 3 第 24 條）。
- public 分數不進判定。10 比上一份低 0.05 以上時只查錯，查不到就維持本節的結果，寫進報告。

### 4.5 對既有規則的例外（不改原文；範圍只限競賽一、全黑池）

| 編號 | 被取代的條文 | 10 的例外 |
|---|---|---|
| F1 | PREREG_02 2.2「套用」第 3 條、rev 8 第 9 條（θ₂ 仍是第 3 層門檻） | 第 1.1 節成立的池，06／10 這條線改交 06 的第 1 層計算。test02 自己的輸出和 Q04 檔不變 |
| F2 | PREREG_06 2.7「防護在第 3 層 → test02 的列」、rev 3 第 8 條 | 同 F1，由 t10 做；t06 的判斷和輸出不變 |
| F3 | PREREG_06 5.4 的最終選擇、`docs/TESTDAY.md` U5 | 第 4.4 節 |
| F4 | PREREG_06 第 7 節「競賽一不用 KataGo HumanSL」 | B 過閘門和 H10P 才用。當時的理由已被先導推翻：學出來的 LDA 單用 0.935（第 7 節揭露） |

## 5. 另一場

- 10 的任何輸出都不進競賽二；不用競賽二的特徵值（P1／P2／P3 probe）、測試資料或輸出。
- 共用的只有程式：KataGo 的抽取程式 `p1_lib`／`p1_extract`（PREREG_08）。這屬於「同一作者、前處理共用」（`docs/REGISTRATION.md` 第 4 點），兩份報告都寫明。

## 6. 刻意不放進 10 的

| 做法 | 理由 |
|---|---|
| A 也套在第 0 層或混色池的第 3 層 | 探索沒看過；混色的第 3 層正是原設計要擋的情況 |
| A 的池子開一對一 | 06-ON 的條件不改 |
| 10B 加一對一（10-ON） | 沒驗證過開關和 P1 一起用；06-ON `kept` 時不傳 10 |
| 用驗證集或 public 選 w | w 只在模擬的 tune seed 上選 |
| 用競賽二的 probe 特徵學 LDA（先導的配方） | 避開「各隊不可共享特徵值」的疑慮；改用競賽一自己的模擬池棋局 |
| P1 原始 cosine | 先導很弱（0.266） |
| 只抽目標方的局面（約省一半 GPU） | 要改 `p1_lib` 或另寫抽取 |
| 用 KataGo 補「棋力」項 | 競賽一要分辨的是人，不是等級 |

## 7. 揭露（寫進 RESULTS 和報告）

- **事前與事後的界線**：A 的動機（nov4 下第 3 層從 0.93 掉到 0.20）、B 的配方（LDA、w 很大）都是看過 10/10 探索才定的。本登記在模擬池抽取、驗證玩家的 P1 之前 commit。先導的 200 人都在 926 人的模擬母體裡；w 只在 tune seed 選、在留出 seed 判，但玩家母體相同。
- **LDA 的資料來源改了**：先導用競賽二的 probe（兩色，2.9 萬盤）學；正式改用競賽一模擬池的黑棋（約 11–13 萬盤）。先導的數字不能直接當正式的預期，正式的效果以 `mock` 的閘門為準。
- **nov4 是人造情境**：fit 檔案沒有日期，「開局改變」用「前 4 手不出現在自己的候選裡」近似；官方實際落在哪一層要看測試日的 ρ。
- **E0 是樣本內**：CNN 看過 fit 玩家，模擬的「只用 CNN」和 V2 都偏樂觀。
- **ρ**：探索的 ρ 是近似值，A 的判定一律用 test02 已 commit 的 ρ 和凍結的 θ₂。
- **A 沒有打分**：在驗證集輸出相同（第 1.4 節）；A 改變的情況只有模擬證據。
- **B 的打分**：競賽一第 3 次；不劣性條件（第 2.6 節）在打分前寫死。驗證池黑白混合，官方全黑；T_P1 對兩色用同一個投影。
- **跨競賽**：只共用 KataGo 的抽取程式；不用競賽二的任何特徵值、測試資料或輸出。KataGo 程式和 HumanSL 權重是外部開源資源，來源和授權同 PREREG_08 第 8 節。
- **推翻 PREREG_06 第 7 節**（第 4.5 節 F4），理由和證據照實寫。
- 生成式 AI 協助的部分照規則揭露（`competitions/_drafts/disclosure-template.md`）。

## 8. 順序（`PY`＝`.venv/Scripts/python.exe`，Git Bash，加 `PYTHONDONTWRITEBYTECODE=1`）

1. 本登記 rev 1 commit，再用一個 commit 填上 rev 1 的編號。**硬規則：早於任何模擬池或驗證玩家的 P1 抽取。**
2. **A（不等 B）**：照 4.1 做 09 的指紋和 tag、記基準，開分支 `p10-testday`；實作 `t10_player.py`（seam、apply、submit、status）、`t10_fake_tests.py`、`official_log.py` 只加行、測試、`docs/TESTDAY_10.md`；獨立審查 → 測試日的修訂（rev 2 起）→ 演練 → 防誤改閘門 → K5-10 前合併。
3. **B**：寫 `tools/p10_p1.py`（`sets`、`extract --set mock|val`、`status`）、`tools/p10_fuse.py`（`selfcheck`、`mock`、`freeze`、`apply`、`apply --finalize`）、`tests/test_p10_*.py`，經獨立審查、全套測試綠了才 commit；實作和登記的差異寫成 rev 2，在 `mock` 之前 commit。`PY tools/p10_p1.py sets` → commit `sets.json` → `selfcheck` → commit → `.venv-gpu/Scripts/python.exe tools/p10_p1.py extract --set mock`（4090，夜間，可分兩晚）→ commit manifest → `PY tools/p10_fuse.py mock`（只判一次）→ commit。沒過：寫 `RESULTS.md`（B 未採用）→ commit，結束 B。
4. 過了：`freeze` → commit；`extract --set val`（約 2.7 小時）→ commit `val_report.json`；`apply` → 稽核（另一個代理）→ `apply --finalize` → commit；`score_allowed` 為真才跑第 2.6 節的指令；打分紀錄帳、`score_H10P.json`、`RESULTS.md`（數字由另一個代理核對）、地圖、`AGENTS.md` 一起 commit。
5. B 採用：在 `p10-testday` 加 `p1`、GPU 接縫、量 tol_P、演練，K5-10 前合併。

## 9. 時程（估計）

| 日期 | 事 | 誰 |
|---|---|---|
| 10/10–10/11 | rev 1；B 的程式；模擬抽取開跑（夜間） | Claude、背景程序 |
| 10/12–10/21 | 模擬抽取跑完、`mock`（K2-10）；A 的測試日工具 | Claude |
| 10/22–10/25 | 過閘門時：凍結、驗證抽取、H10P（K3-10） | Claude |
| 10/26–10/31 | 測試日擴充、演練、合併（K5-10） | Claude |
| 11/05–11/07 | t06 之後跑 t10 的 seam、apply（CPU，幾分鐘） | Claude |
| 11/06–11/14 | B：兩池 `p1`（4090 約 5.5 小時，競賽二的 07／08 之後） | Claude、背景程序 |
| 11/22–11/23 | `official_log.py status`；需要時上傳 10 | 使用者（每次說 OK） |

## 10. 防誤改閘門

- `AGENTS.md` 的閘門指令加 `tests/test_p10_hygiene.py`；4.1 之後加 `tests/test_t09_path_hygiene.py`，合併 10 的測試日後加 `tests/test_t10_hygiene.py`。
- 唯一允許的失敗不變（`test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`）。H10P 的 2 列照第 2.6 節的 rev 2 處理。H10P 之前，打分紀錄帳不得出現 `prereg10` 列。

## 11. rev 2：實作和登記的差異（在 selfcheck、mock 之前）

以下是 B 的程式實際怎麼做、和第 2–3、8、10 節寫的不同或沒寫到的地方。以 repo 的程式為準；登記原文不改。

1. **`p10_p1.py` 先 commit**：`sets`、`extract --set mock`、`status` 在 `15b3e98` 先 commit（趁 4090 空著），早於整套工具和獨立審查；`sets.json`（SHA `167c5a51…`）在任何模擬抽取前 commit。之後只加了 `manifest`、`extract --set val` 和 `main` 的參數檢查；`cmd_sets`、`load_games`、`cmd_status`、`cmd_extract` 的模擬部分逐字不變（`tests/test_p10_hygiene.py` 用 AST 比 `15b3e98`）。`mock_sets_record.json` 的 `tool_sha256` 是 `15b3e98` 那一版的 SHA，檔案現在的 SHA 不同。`extract --set mock` 帶 `--smoke`／`--candidates`／`--questions`／`--out-dir`／`--fake-scorer` 一律拒絕（在 `main`，不動凍結的函式）。
2. **模擬塊的清單**：第 8 節的「commit manifest」由新指令 `p10_p1.py manifest --set mock` 做：616 塊全到、ref、set、`sets.json` SHA 都對才寫 `results/prereg10/feat/mock_manifest.json`（只寫一次）。`mock`、`freeze` 要它已 commit，並逐塊重算 SHA 比對；`sets.json` 的 SHA 要等於已 commit 的 `mock_sets_record.json`。
3. **wr 池**：題目只取 base 題目庫（每人 20 盤）。`p06_sim.make_pool` 的 wr 修補照抄，但容量只有 20 時 40 個 seed 有 39 個會當掉（沒有可移的人）。改成：照抄的規則找不到人時，才改在全池找容量夠的人（同一個 `r3.choice`）；全池也沒有就拒絕（結束碼 2）。每池約 13 次這種移動，不同的出題人約 267 位（修補前 254）。
4. **池的其他細節**：
   - 每個 seed 的 400 人照 `p06_sim.choose_pool` 的規則（每級 40 人，sha256('{seed}|{pid}') 排序）在 926 人裡選，real 模式逐 seed 核對和 `choose_pool` 相同。
   - 盤數：`k_list(400)` 用 `random.Random('k|seed')` 洗牌（同 `x1_scenarios.py`）。
   - c20：候選取每人 100 盤候選的前 20 盤（sets 順序），題目同 base。
   - 同一個 seed 的四種情境用同一批 400 人，所以每個 seed 只學一次 P1 LDA、一次 06 LDA。
   - ρ 近似值只記 base、nov4、wr（c20 不是 100 盤，不算）。
5. **閘門的統計量**（主 session 定案）：
   - 點估計：留出 seed 20–39 的全部題目合起來，對每一題的差（10B(w*) − V2）取平均，和官方計分的單位一致。
   - 95% 區間：玩家 bootstrap。每次有放回地重抽玩家，把抽到的玩家的全部題目合起來，重算每題平均；2,000 次，每種情境各用一個新的 `default_rng(0)`，取 2.5%／97.5%。
   - 「每位玩家的平均差」只記錄，不進判定。
6. **可用盤和缺值**：
   - 可用盤：塊裡有這盤、沒失敗、目標方至少 1 手（`counts[:,0] > 0`）。
   - LDA 只用可用盤，只用有 ≥ 2 盤可用盤的玩家。
   - 題目沒有可用盤：z 記 0（同登記）。
   - 候選沒有可用盤（登記沒寫）：那一欄填該題其他候選的最小值（同 `p06_style.colour_terms`），實際上不會發生。
7. **selfcheck 的比法**：
   - 參考值：已 commit 的 `x3_p1_eval.json` 只到小數 4 位，所以 ≤ 1e-9 的參考值改成「把 `x3_p1_eval.py` 原樣 import、只把它模組裡的 `round` 換成不取整」重跑的結果。只比這個，差 ≤ 1e-9 才算過；和 4 位小數 JSON 相等只記錄。
   - 比對範圍：先導三種情境，各比 V2 和 V2+P1ldaBW@w（全部 11 個格點），共 36 個數。
   - 數值路徑的例外：沒過、但每個差都 ≤ 1e-6、而且沒有缺值時，記 `only_numeric_path`（判為數值路徑差異）。這時可以寫一版 PREREG_10 修訂（N ≥ 2，commit 晚於 `selfcheck.json`），說明為什麼，再用 `mock --selfcheck-revision N` 往下做。差更大就停下修程式。
   - 競賽二 probe 特徵只在 selfcheck 裡讀，由 `PROBE_FUNCS` 兩支負責：`pilot_probe_lda` 直接讀，`x3_reference` 跑 x3 時間接讀。兩支都只有 `cmd_selfcheck` 呼叫，結果不存、不進任何選擇；防誤改測試核對這一點。
8. **工具綁定和修 bug 的路**：
   - 檔案分開：驗證那一側（`apply`、`apply --finalize`、稽核、驗證抽取用的函式）拆到 `tools/p10_apply.py`。`p10_fuse.py apply` 仍可用，會轉過去。
   - `selfcheck`、`mock`、`freeze` 只綁 `MOCK_TOOL_FILES`：`p10_fuse.py` 和它 import 的 `p06_style`、`p06_sim`、`p02_player_fuse`、`p02_player_open`、`p02_moves`、`c10data`。所以修 `p10_apply.py`、`p10_p1.py` 不影響模擬的綁定。
   - selfcheck 已 commit 後要修模擬路徑的程式：先 commit 一版修訂 N（晚於舊的 `selfcheck.json`），再跑 `selfcheck --rerun --reason T --revision N`；`mock` 一律對照當時已 commit 的 `selfcheck.json` 檢查工具 SHA。
   - `mock` 只判一次；要重跑需 `--rerun --reason T --revision N`，修訂 N 晚於舊的 `mock_report.json`。
   - `freeze` 綁 `mock_report.json` 記的模擬路徑工具；`apply` 綁全部工具（`APPLY_TOOL_FILES`），而且模擬路徑的檔案要和 `frozen_config.json` 記的相同（T_P1 和融合一定用凍結時的程式）。
9. **模式和執行**：
   - 沒有 `--precommit`：工具 commit 後才跑 selfcheck。
   - real 的 `mock` 要本修訂 rev 2 已 commit，`selfcheck.json` 已 commit。
   - `--out-dir` 只能在 `data/prereg10/tmp/` 底下，只有它能換 sets／塊／池形狀／seed；這種預覽紀錄標 `preview`，之後的步驟一律不收。
   - 大檔寫到 `data/prereg10/mock_work/`（`p1_side0.npy`、`p1_usable.npy`、`mock_points.json`，不進 git），SHA 記在報告裡。
   - 平行用 spawn 的子程序（預設 2 個、最多 6 個），每個約 2–3 GB；BLAS 執行緒用預設，real 模式設了 `OMP_NUM_THREADS` 等變數就拒絕。
10. **凍結記的東西**：
    - `p1_lda.npz`：m、s（先標準化用）、W、ev、mu、sd 和盤數、人數。
    - `frozen_config.json`：
      - w*、閘門紀錄、`mock_report.json` 的 SHA 和 commit；
      - 06 的凍結檔（`frozen_config.json`、`lda_B/W.npz`、`player_val_top5.csv`、`apply_report.json`）的 SHA；
      - `p1_lib`／`p1_extract` 的 SHA，KataGo commit（從 `hs_lib.py` 讀）；
      - 權重的大小和完整 SHA，manifest 裡的中性 metadata 和程式 SHA；
      - 工具 SHA，`.venv` 版本。
11. **驗證抽取（`extract --set val`）**：
    - 前提：`frozen_config.json`、`p1_lda.npz` 已 commit，B 過閘門，不是預覽。
    - 先裝 06 的 `data/local` 防護，只在 `allow_local()` 裡讀那兩份棋譜檔；兩檔的 SHA 要等於 06 `apply_report.json` 記的。讀完才 import `p1_lib`。
    - 盤的順序：候選檔順序，接題目檔的題目、槽位順序；目標方是那一列的 `color`／`color_i`。
    - 塊的 set 記 `p10val` 和順序的 SHA。
    - 全部做完寫 `results/prereg10/feat/val_report.json` 和 `val_seam_feats.npz`（前 400 盤的 side 0 平均、手數、失敗旗標），兩個一起 commit，給第 4.2 節的 GPU 接縫用。
12. **apply 比登記嚴**：
    - 06 用 06 的函式在程式內重算，不跑 `p06_player.py apply`。
    - 除了逐位元組相同，還要：sr06 各組合到小數 6 位等於 06 的 `apply_report.json`、選中的組合相同；四個輸入（兩份棋譜檔、p02 兩個 dump）的 SHA 和 06 相同。
    - 池閘門用 sr06 全部 10 輪。
    - 「P1 失敗盤 ≤ 0.5%」只算重播失敗；沒有目標手的盤另外記。
    - `score_allowed` 先寫 `pending_audit`，`--finalize` 才變成真。
13. **稽核紀錄的格式**：
    - 檔案：`results/prereg10/player/prescore_audit.json`，format `c10-prereg10 player prescore_audit v1`。
    - 欄位：`program_errors_affecting_10B`（清單）、`independent_recompute_identical: true`。
    - 綁定：`player_val_top5.csv`、`p1_lda.npz`、`frozen_config.json`、`val_report.json` 的 SHA。
    - commit 要晚於 `apply_report.json`。
14. **防誤改測試**：
    - `tests/test_p10_hygiene.py` 以 `559311f` 為基準。
    - 例外：第 4.1 節加的 09 指紋 `results/test09/design/p09_path_fingerprint.json`（由 `test_t09_path_hygiene.py` 釘住）。
    - 本登記只准加行，但修訂表的 `| rev N |` 列可以填 commit 編號。
    - H10P 之前打分紀錄帳不得有 `prereg10` 列。
    - 本檔 commit 時 `AGENTS.md` 的閘門要加這支測試。
15. **結束碼**：0 完成；1 登記的停下（selfcheck 沒過、閘門沒過、06 沒重現、apply 被擋）；2 拒絕或錯誤。

## 12. 測試日的修訂（rev 3，2026-10-10；只做 A）

**白話版**：A 的測試日工具（`tools/t10_player.py`、`tools/t10_fake_tests.py`、`tools/official_log.py` 加的行、`tests/test_t10_*.py`、`docs/TESTDAY_10.md`）在分支 `p10-testday` 做好。B 還沒過閘門（模擬池抽取都還沒跑），所以這一版只有 A：B 的掛鉤一律回「B 沒有採用」。下面是實作和第 1、4 節寫法不同或寫得不夠細、這裡寫死的地方。看過的東西：只有 fit 玩家做的假池（`t10_fake_tests.py`）和既有的演練樹；沒有測試資料、沒有打分、沒碰 `data/local` 的答案。

1. **範圍**：rev 3 只實作 A。`t10_player.py p1` 和 apply 裡的 B 分支在 B 沒採用時一律回結束碼 3（`b_not_adopted`），不寫任何檔。B 過閘門（第 2.3 節）而且 H10P 採用時，另寫 rev 4 起的修訂把 `p1`、GPU 接縫、tol_P 和 10B 加進來，同樣要在 K5-10 前合併；在那之前 10 只有 A（10A／06-OFF／test02 的列）。
2. **1.1 第 2 條（全黑）**：用 test02 那一池 `apply_report.json` 的 `b_c_candidate_black_share`、`b_q_question_black_share` 是否都剛好等於 1；T4 的核對是 `checks/player/t4.json` 的 `pools[池].black_only_pool` 為真、`b_c`、`b_q` 為 1。報告說全黑但 T4 沒有這一池的紀錄或不一致（或反過來）是**結束碼 2**（停下查），不是「A 不用」。
3. **1.1 第 3 條（第 3 層是 ρ 造成的）**：t10 用 `p02_player_fuse.guard_decision`（test02 自己用的同一支函式），拿報告的 `effective`、ρ、b_c、b_q 和 `results/prereg02/player/frozen_config.json` 凍結的防護重算，`applied` 的代號、`guard_level`、`guard_reason` 三樣都要和報告相同；報告裡記的門檻也要等於凍結檔。不同是結束碼 2。A 成立還要：層級 3、test02 閘門通過、`effective` 有開局項、那個方向的 ρ 是有限數、`theta2_enabled` 為真、ρ < θ₂。
4. **1.1 第 4 條（06 可用）**：H6P 採用；06 的 `feats`（三池）報告存在（real mode 已 commit）；06 的 seam 通過（`t06_player.load_seam`，dumps 不同而沒過時是結束碼 3）；這一池的 `t06_pool.json` 存在、是用現在的 test02 報告（`apply_report.json` 和 `top5.csv` 的 SHA）和現在的 06 seam 做的，`top5_off.csv` 是它記的那一份。`t06_pool.json` 不存在或過期是**結束碼 2**（先跑或重跑 `t06_player.py apply`），不是 3。t06 對第 3 層的池子不算 06，但照樣寫 `t06_pool.json`（兩池都退回時，第二池在結束碼 3 之前已寫好）。**t06 `apply` 因「兩池都退回 test02」（`both_pools_test02`）回的結束碼 3、和 `submit` 因此不產生 06-OFF，都不算「06 不能用」**：那是第 3 層的結果，不是 06 本身壞掉（nov4 的情況就是這樣），06 的 feats、seam 照常可用，A 照判；只有 T1 FAIL_OPENING、H6P 沒採用、06 seam 因 dumps 不同沒過、K6-10 已過才是 06 不能用。
5. **1.2 的計算**：import `tools/t06_player.py` 的 `load_pool_data`、`t02_binding`、`load_feats`、`pool_store`、`compute_pool`（不改任何一支），`compute_pool` 的 R02 參數傳 test02 那一池的 `effective`（R02′，通常是 Q04），06 閘門 sr(選中) − sr(R02′) ≥ `t06_player.GATE06_MIN`（0）。過了這一池是 10A（`compute_pool` 的 06-OFF 列）；沒過是 test02 的列。一對一不進輸出。
6. **1.3（A 不用的池）**：10 的那一池就是 06 的 `top5_off.csv`（06 用了時是 06-OFF 的列；06 退回 test02 時是 test02 的列）。
7. **seam**：在 seam_val 上用 06 seam 的輸入（test02 的 seam_val dumps、06 的 feats，SHA 都要和 `t06_seam.json` 記的相同），先對 test02 seam_val 的 `apply_report.json` 判 A，必須**不成立**；再用第 5 條的同一段計算（R02 = Q04，同 06 的 seam）重跑，Top-5 要和 06 seam 的 `top5_off.csv` 逐位元組相同；06 seam 的 dumps 是逐位元組相同時，也要和 `results/prereg06/player/player_val_top5.csv` 逐位元組相同；選中組合、06 閘門的判斷要和 06 seam 相同。不符是結束碼 4（`seam_drift`）。B 沒採用，所以不重算 10B。
8. **submit**：參照組是這一季的 06-OFF（`results/test06/player/off/` 有交件檔和報告，而且報告記的兩池 `t06_pool.json` 是現在的那兩份；不是的話結束碼 2，先重做 06 的 submit）；沒有 06-OFF（兩池都退回 test02）時是 Q04。兩池的列都和參照組逐列相同：不產生交件檔，結束碼 3（`10_not_needed`）。報告記每池的來源（06-OFF／10A／test02）、A 的每一條件、和 06-OFF、Q04 第 1 名不同的題數、`inputs_bound`（test02 的交件檔、報告、seam、T1、T4、dumps，`prereg02` 的凍結檔，06 的凍結檔、seam、feats、pool 報告、`top5_off.csv`，有 06-OFF 時它的交件檔和報告，10 的 seam、pool 報告、`top5.csv`）。
9. **結束碼 3 的情況**（記在 `results/test10/player/not_used.json`，commit）：T1 FAIL_OPENING；H6P 沒有採用；06 的 seam 因 dumps 不同沒過；K6-10 已過、而且 K6-10 前沒有 commit 過 10 的 submit 報告（有的話是 2，那一份照常有效）；K5-10 不成立（submit，real mode）；10 不需要；B 沒採用（`p1`）。`t06_player` 回的結束碼 3 由 t10 轉成 10 自己的紀錄，**不寫進 `results/test06/`**。
10. **K5-10 怎麼判**：本登記修訂表的 `| rev 3 |` 這一列在 HEAD，而且加入它的 commit 時間 ≤ 2026-10-31 23:59:59（台灣時間）。`t10_player.py submit`（real mode）和 `official_log.py prepare`／`final` 都查。也就是本修訂（rev 3）要在 10/31 前 commit；`p10-testday` 在 K5-10 前合併回 main。之後的修訂（例如 B 採用時）另起 rev 4 起。
11. **`official_log.py`（只加行）**：
    - `results/test10/player/` 的 `submission_player.csv` 是 10 的檔（在 06 的規則之前認，不會被當成 Q04）；方法名稱「prereg10 10: 06-OFF + all-black level-3 lift (A) [+ P1 LDA (B)] (甲)」。
    - `prepare`（10）：綁定 `t10_submit_report.json`（real mode、方法、check_submission official、這個檔的 SHA、至少一池 10A、沒有 10B、B 記為沒採用、10 和 06 的 seam 通過、H6P 採用、T1 PASS／NORMALIZED），10 的 seam 和兩池報告、06 的 seam、test02 的 Q04 報告和完整 T1，`inputs_bound`（有一池 A 成立時要含 T4），K6-10、K5-10。順序：Q04 第一次上傳已記；這一季有 06-OFF 時，10 報告比對的那份 06-OFF（run_id 和 SHA 相同）已上傳；06-ON 產生時，要 06-ON 不是 PREREG_06 5.3／5.4 指定的最終（否決、傳錯、核對不出、K7' 已過、06-ON 不符合條件），06-ON 產生但沒上傳時要等 K7' 過。
    - **最後一版**（有 10 紀錄時取代 PREREG_06 5.4 的 final 規則）：06-ON 照 PREREG_06 判定 kept → 06-ON；06-ON 還在等否決 → 還沒定；有符合條件的 10 上傳 → 10；都沒有 → 照 PREREG_06 5.4（06-OFF → Q04）。改選其他的（`final`，或之後再 `prepare` Q04／06-OFF／06-ON）要 `--not-10-reason "PREREG_10 rev N: …"`（N ≥ 2，加入那一列的 commit 晚於被推翻的紀錄；rev 2、rev 3 早於測試日的任何紀錄，所以實際上要 rev 4 起）。有 10 紀錄時 `--not-06-reason` 在 `final` 拒絕（改用 `--not-10-reason`）；`status` 的 06 那段不再報違規，10 被指定時也不再要求再傳一次 06-OFF。
    - `public`：10 和 Q04／06-OFF 的 public 分數完全相同、而 10 報告記的 public 第 1 名不同題數 > 0 → 傳錯檔警告；10 比上一份低 0.05 以上 → 提醒只查錯（第 4.4 節）。
    - log 裡沒有 10 紀錄時，輸出、log 位元組、回傳值和 tag `p09-testday-pre10` 的版本相同（`tests/test_t10_official_log.py` TagVersusNew，含 06 的流程）。
12. **防誤改閘門**：合併後 `AGENTS.md` 的閘門加 `tests/test_t10_hygiene.py`（第 10 節）。它凍結 tag `p09-testday-pre10` 以來的舊工具、舊登記、各測試日的 design 紀錄和 `results/prereg01–09`，`results/prereg10/explore/` 只准加檔，`official_log.py`、`.gitattributes`、`.gitignore`、`PREREG_02／06／07／08／09／10` 只准加行；`baseline_pre10.json` 釘住。`.gitattributes` 加 `results/test10/** -text`。
13. **演練**：`tests/test_t10_dryrun.py`（`C10_SLOW=1`）在三棵假樹上跑完 Q04 → 06 → 10，結果和負面演練記在 `results/test10/design/build_review.md`。官方形狀樹的 Q04、06-OFF、06-ON 和 `baseline_pre10.json` 逐位元組相同。
14. **`tools/p10_p1.py` 先 commit 的揭露**：已寫在 rev 2（第 11 節第 1 條），這裡不重複。

## 13. B 的測試日修訂（rev 4，2026-10-11；B 採用時才生效）

**白話版**：B（P1 的 LDA 項）的測試日工具做好了（`tools/t10_b.py`，由 `tools/t10_player.py` 呼叫），但**只有 B 採用之後才會動**。現在 B 還沒採用（驗證玩家的 P1 抽取、apply、稽核、H10P 都還沒做），所以 `p1` 和其他 B 的路徑一律回結束碼 3「B 沒有採用」、不寫任何檔，10 只有 A，和 rev 3 完全一樣。GPU 接縫的容許差 tol_P 要在拿到測試資料前用 4090 演練量出來，**本修訂 commit 時還沒量（tol_P【待演練】）**；沒有已 commit 的演練紀錄，正式模式的 `p1` 一律拒絕。下面是實作和第 2、3、4 節寫法不同或寫得不夠細、這裡寫死的地方。看過的東西：只有假樹（fit 玩家的模擬池、合成的 P1）；沒有測試資料、沒有打分、沒碰 `data/local` 的答案、沒用 GPU。

1. **「B 採用」怎麼判**（`t10_b.b_state`）：`results/prereg10/RESULTS.md` 有一行剛好是 `H10P：B **採用**`（前後空白不算）。沒有這一行＝B 沒採用：所有 B 的路徑結束碼 3、不寫檔，其他都不查（rev 3 的行為）。有這一行時，下面全部要成立，否則是**結束碼 2**（不一致，停下查）：
   - `score_H10P.json` 符合第 2.6 節：B−A 點估計 ≥ 0、95% 下界 > −0.010，而且 A 側平均等於 `score_H6P.json` 的 B 側平均（差 ≤ 1e-12）；
   - `frozen_config.json` 是凍結紀錄、`b_passed`、不是預覽；`p1_lda.npz` 的 SHA 等於它記的；
   - `apply_report.json` 的 `score_allowed` 是 `true`（已 `--finalize`），記的 `player_val_top5.csv` SHA 和 w* 都對；
   - `val_report.json` 完整、正式模式、記的 `val_seam_feats.npz` SHA 對、用的是現在這份 `frozen_config.json`；
   - 正式模式另外要：上面這些檔和打分紀錄帳都已 commit 而且乾淨；帳上 `prereg10 H10P` 剛好 2 列；模擬路徑的工具（`p10_fuse.MOCK_TOOL_FILES`）和 `p1_lib`／`p1_extract` 跟凍結時相同。
   - H10P 之後寫 `RESULTS.md` 的人照這一行的寫法；不採用寫 `H10P：B **不採用**`。
2. **`p1`**（`.venv-gpu/Scripts/python.exe tools/t10_player.py p1 --pool seam_val|public|private`；4090）：
   - 順序：先 `seam_val`（GPU 接縫），通過後才跑 `public`、`private`。前提同 10 的其他步驟（test02、06 的 feats／seam 已 commit 等；第 4.2 節），另加 B 採用、K5-10（B）、K6B-10 還沒過、演練紀錄已 commit（第 4 條）。
   - 開跑前 `nvidia-smi` 不能有別的計算程序（Type C），之後才碰 CUDA；GPU 要是 `NVIDIA GeForce RTX 4090`；checkpoint 完整 SHA、中性 metadata SHA、P1 程式 SHA 要等於 `frozen_config.json` 記的（不同：結束碼 2）。
   - 設定同驗證抽取（`p10_p1.py extract --set val`）：fp16、batch 1024、channels_last、cuDNN benchmark、每 4,096 個局面送一次；`p1_extract.run_chunk` 和 `p1_lib` 原樣 import（不改）；遊戲檔讀完才 import 它們。
   - 盤的順序：候選檔順序，接題目檔的題目、槽位順序；目標方是那一列的 `color`／`color_i`（官方全黑）。測試池的兩個檔要和 test02 用的相同（SHA）。
   - 每 200 盤一塊，寫到 `data/test10/p1/<池>/`，原子寫入；已有的塊只有和這一次完全相同（格式、set、ref、鍵的 SHA、模式、scorer、torch／cuDNN／GPU、metadata、程式、checkpoint）才沿用，否則結束碼 2，絕不覆寫。`STOP` 檔：做完手上那一塊就停，**結束碼 5**（沒有判定；刪掉 STOP、同一行重跑就接續）。
   - 公開／私人池的環境（scorer、torch、cuDNN、GPU）要和 GPU 接縫的相同，不同是結束碼 2（在這個環境重跑接縫）。
   - 報告：`results/test10/p1/<池>_p1_report.json`（塊的 SHA、失敗盤、沒有目標手的盤、鍵、輸入檔 SHA、環境、tol）；接縫另寫 `gpu_seam.json`。每一步做完就 commit。
3. **GPU 接縫的定義**（第 4.2 節寫「每盤最大差」，這裡寫死）：驗證池前 400 盤（`extract --set val` 的順序，也就是候選檔前 400 盤）在 4090 重抽，和已 commit 的 `val_seam_feats.npz` 比：失敗旗標和目標方手數要逐盤相同；在可用盤（沒失敗、目標方 ≥ 1 手）上，用凍結的 `p1_lda.npz` 把兩邊的目標方平均投影到 64 維，每盤取 64 維的最大絕對差，再取所有可用盤的最大值，要 ≤ tol_P。沒過：`b_not_used.json` 記一筆、結束碼 3（只對 B），這一季 B 不用，10 只有 A；之後的 `p1 --pool public|private` 也回 3。
4. **tol_P 和演練**：`p1-rehearse`（`.venv-gpu`、4090）做第 3 條同一段抽取和比較，m＝那個最大差；tol_P＝max(0.05, 2.5 × m)，寫進 `results/test10/p1/p1_rehearsal.json`。
   - 只能在 2026-11-04 11:00（測試集開放）之前、`data/test` 還沒有任何官方測試檔時跑；紀錄的 commit 時間要早於 11/04 11:00；要是 scorer＝real、通過（失敗旗標和手數相同）、綁定現在的 `frozen_config.json` 和 `val_seam_feats.npz`。
   - 正式模式的 `p1` 沒有這份已 commit 的紀錄一律拒絕（結束碼 2）；11/04 11:00 前沒演練，這一季就沒有 tol_P，B 不用。
   - **tol_P【待演練】**：本修訂 commit 時 B 還沒採用、`val_seam_feats.npz` 還不存在、4090 給其他專案用，所以 m 沒量。演練後只 commit 紀錄檔，數字以紀錄檔為準，不另改登記。
   - 演練的啟動器：`data/prereg10/t10_rehearsal/run.sh`（不進 git；會等 GPU 空），寫法見 `docs/TESTDAY_10.md` 的 B 一節。
5. **seam 擴充**：B 採用時，`seam` 在 seam_val 上用已 commit 的驗證 P1 塊（`data/prereg10/val/`，SHA 照 `val_report.json`）和凍結的 LDA，在 06 的第 1 層計算（R02＝Q04）上重算 10B。06 seam 的 dumps 和 prereg02 逐位元組相同時，要和 `results/prereg10/player/player_val_top5.csv` 逐位元組相同；dumps 不同時（Colab 重算），≥ 398／400 列相同就算過（**比第 4.2 節寫的「逐位元組相同」寬**，理由同 PREREG_06 5.2：dumps 不同時 06 本身也只要求 398／400）。不符：結束碼 4。B 採用後，沒有 B 紀錄的舊 seam 不收（要 `--rerun`）。
6. **apply 加 B**（只在來源是 06-OFF 或 10A 的池；test02 的列不加）：
   - 06-OFF 的池：用 06 的程式（`t06_player.compute_pool`，R02＝test02 的 `applied`）重算，要和 06 的 `top5_off.csv` 逐列相同，不同是結束碼 2；10A 的池用第 1.2 節已算好的那一份。F_sel 是選中組合做 CSLS 之前的分數。
   - 這一池的 P1 報告或 GPU 接縫還沒有：K6B-10 前是結束碼 2（先跑 `p1`）；K6B-10 已過是「這一池不加 B」。接縫沒過：不加 B。
   - 池閘門：在這一池的 sr06 假題（全部 10 輪，同 rev 2 第 12 條）上，sr(10B) − sr(選中) ≥ 0；而且 P1 失敗盤（只算重播失敗）≤ 0.5%。都成立：這一池的來源是 **10B**；否則維持原來的來源，報告記理由。重算的 sr(選中) 要等於 06 的計算（差 ≤ 1e-12），不同是結束碼 2。
   - 06 的 `top5_off.csv`、test02 的檔都不改；B 的列只寫在 `results/test10/player/<池>/top5.csv`。
7. **K6B-10**（11/14 23:59:59）：正式模式下，這一池的 P1 報告和 `gpu_seam.json` 最後一次 commit 都要在 K6B-10 前；`p1` 在 K6B-10 之後跑是結束碼 3（B 不用）。
8. **K5-10（B）**：本修訂表的 `| rev 4 |` 列在 HEAD，而且加入它的 commit 時間 ≤ 2026-10-31 23:59:59（台灣時間）；不成立時 B 不用（`p1` 結束碼 3、apply 不加 B、`official_log.py` 不收 10B）。A 的 K5-10 仍照 rev 3 第 10 條（`| rev 3 |` 列）。
9. **submit**：來源可以是 10B；報告的 `b` 記採用、GPU 接縫、tol、哪幾池是 10B、K5-10（B）；`inputs_bound` 加上 B 的檔（`results/prereg10/RESULTS.md`、`player/` 的凍結檔、LDA、`score_H10P.json`、`apply_report.json`、`player_val_top5.csv`、`feat/val_report.json`、`val_seam_feats.npz`、演練紀錄、`gpu_seam.json`、三份 P1 報告）。和 06-OFF（或 Q04）逐列相同時照舊是「10 不需要」。方法名稱不變（「… [+ P1 LDA (B)] (甲)」本來就含 B）。
10. **`official_log.py`（只加行）**：rev 3 的 `t10_report_problems` 原文不動，後面加一個同名的包裝：報告的 B 是「沒採用」時，原樣交給 rev 3 的函式（行為逐位元組相同）；B 是「採用」時，先把 10B 當成「有提升的池」交給 rev 3 的規則，再加 B 的規則：K5-10（B）；GPU 接縫通過、有 tol_P 和 w*；每一個 10B 池的 B 紀錄是「已加」、池閘門通過、P1 失敗 ≤ 0.5%；那一池的 P1 報告和 `gpu_seam.json` 最後一次 commit 在 K6B-10 前；`inputs_bound` 含 B 的必要檔（`T10_B_INPUTS_REQUIRED`＋10B 池的 P1 報告）。第 4.4 節「一筆 10 的上傳符合」因此對 10B 多了這些條件。
11. **結束碼**：`p1`、`p1-rehearse` 加 **5**（STOP，沒有判定）。B 的「不用」（沒採用、接縫沒過、K6B-10 已過、K5-10（B）不成立）是結束碼 3，但**只對 B**：沒採用時不寫檔；其他記在 `results/test10/p1/b_not_used.json`（不是 10 的 `not_used.json`），10 照 A 繼續。
12. **假測試和演練**（只用 CPU）：
    - `--synthetic-p1`（只限 `--dry-run`）：不跑 KataGo，用 SGF 的落點統計當 770 維特徵，給大的假樹用；`--fake-scorer`（只限 `--dry-run`）：走 `p1_extract.run_chunk`，探針換成 CPU 的確定性替身（同 `p10_p1.py` 的 `FakeProbeScorer`）。
    - `tools/t10_b_fake.py`：做假的 `results/prereg10`（凍結檔照抄、驗證池合成 P1、`p10_apply.apply_core` 算的 10B、**假的** `score_H10P.json`、有採用那一行的 `RESULTS.md`），和負面演練的拷貝（接縫被改、沒採用）。
    - `tests/test_t10_b.py`（快）和 `tests/test_t10_dryrun.py` 的 B 樹（`C10_SLOW=1`）；結果記在 `results/test10/design/build_review.md`。
13. **防誤改閘門**：`tests/test_t10_hygiene.py` 改成：`t10_player.py` 不再有 `B_ADOPTED` 常數（改由 `b_state` 讀已 commit 的檔）；`tools/t10_b.py`、`tools/t10_b_fake.py`、`tests/test_t10_b.py` 列入 10 的工具；`t10_b.py` 只有在函式裡才 import `torch`／`p1_lib`／`p1_extract`，不 import 競賽二的工具、`tools/score.py`；`official_log.py` 有 `T10_SOURCES_B`；本登記有剛好一列 `| rev 4 |`。

## 14. rev 4 的補充（獨立審查之後，2026-10-11；仍在任何測試資料、任何打分之前）

**白話版**：審查要求一條總原則：**B 的任何問題都只讓 B 這一季不用，10 照 A 繼續**，不能因為 B 而讓 seam、apply、submit 停下（結束碼 2、4）。下面取代第 13 節裡和這條原則衝突的寫法（第 13 節原文不改，衝突處以本節為準）。

1. **總原則**：B 沒採用、B 的檔對不上、K3-10／K5-10（B）／K6B-10 不成立、11/04 11:00 之後沒有可用的演練紀錄、GPU 接縫沒過、seam 重算的驗證集 10B 對不上、某一池的 P1 有問題（塊或鍵對不上、重算 06 對不上、sr06 對不上、任何例外）→ B 在那一步（或那一池）不用：`p1` 回 3、`seam`／`apply`／`submit` 照 A 做完（結束碼 0），理由記在 `results/test10/p1/b_not_used.json` 和報告的 `b`。只有 registered「B 沒有採用」（下面第 2 條的 `not_marked`、`h10p_not_adopted`）不寫檔，同 rev 3。取代第 13 節第 1 條「結束碼 2」、第 5 條「結束碼 4」、第 6 條「結束碼 2」的寫法。
2. **「B 採用」的判定（取代第 13 節第 1 條）**：`b_state` 不會拒絕，只回「採用」或理由：
   - 沒有 `score_H10P.json`（正式模式：沒有 commit）而且 `RESULTS.md` 沒有判定行 → `not_marked`。
   - 有 `score_H10P.json`：`RESULTS.md` 要剛好有一行判定，而且和 `h10p_decision`（第 2.6 節）一致；不採用而且一致 → `h10p_not_adopted`；不一致（沒有行、兩行、和分數相反）或有行但沒有分數 → `inconsistent`。判定行接受前面的項目符號（`-`、`*`、`+`）、全形或半形冒號、後面接其他文字，例如 `- H10P: B **採用**（…）`。
   - 採用而且一致：正式模式下 `score_H10P.json` 最後一次 commit、和 `RESULTS.md` 第一次出現「**採用**」的 commit 都要 ≤ **K3-10（2026-10-25 23:59:59）**，否則 `k3_10`（第 3 節：K3-10 沒趕上，這一季不用 B）。
   - 其他檔（凍結檔、LDA、apply、驗證 P1 報告、工具 SHA）對不上 → `inconsistent`。打分紀錄帳的 `prereg10 H10P` 列數接受 **2 或 4**（第 2.6 節：查 bug 重跑也算、要記帳）。
   - `inconsistent` 在測試日之前（`p1-rehearse`，11/04 11:00 前）是結束碼 2（要修好檔）；測試日是「B 不用」（`p1` 結束碼 3，記錄），A 照常。
3. **seam 的驗證集 10B（取代第 13 節第 5 條的「結束碼 4」）**：不再是 seam 的通過條件；對不上時 seam 照 A 判定，`b_seam.passed` 記假、寫 `b_not_used.json`，apply 在每一池以 `val_10b_seam_failed` 不加 B。seam 沒有 B 紀錄（例如 B 在 seam 之後才採用）→ `seam_without_b`，同樣不加 B。
4. **apply 之前 P1 還沒做（取代第 13 節第 6 條的「結束碼 2」）**：那一池先維持 A（`p1_not_done`），提示在 K6B-10 前做完 `p1` 後用 `apply --rerun` 再跑一次才會加 B；`submit` 只提醒、不擋。已有 10B 的池在之後 B 的狀態變了（例如檔對不上）→ `submit` 要求先重跑 apply（重跑會照第 1 條退回 A）。
5. **11/04 11:00 之後沒有可用的演練紀錄** → B 不用（`no_rehearsal`，結束碼 3，記錄）；之前是結束碼 2（先做演練）。**和第 4.2 節的差異（揭露）**：第 4.2 節寫「m … 寫進測試日的修訂」（也就是 10/31 前寫進登記）；改成 tol_P 由 **11/04 11:00 前 commit** 的演練紀錄 `results/test10/p1/p1_rehearsal.json` 決定，不另寫進登記。理由：m 要等 H10P 採用、驗證 P1 抽完才能量，可能晚於 10/31；而紀錄檔在任何測試資料之前 commit、工具查 commit 時間，事前性不變。
6. **K5-10（B）（取代第 13 節第 8 條）**：`| rev 4 |` 這一列要在 HEAD，而且在 **main 的 first-parent 歷史**裡第一次出現的那個 commit（合併進 main 的 merge commit，或直接 commit 在 main）時間 ≤ 2026-10-31 23:59:59，也就是「10/31 前合併回 main」。`t10_b.k5_b_state` 和 `official_log.t10_k5_b_state` 都用 `git log main --first-parent --diff-merges=first-parent -S`。
7. **`official_log.py`（取代第 13 節第 10 條的觸發條件）**：B 的規則只在**至少一池是 10B**時才查；B 記為採用但兩池都不是 10B 的報告就是 A 的報告，照 rev 3 的規則（不要求 B 的檔、K6B-10）。10B 而 B 記為沒採用 → 拒絕。
8. **假樹不碰 repo 的 B**：`--dry-run` 不給 `--prereg10-dir` 時讀一個不存在的資料夾（`<O10>_no_prereg10`），所以 A 的假樹在 B 採用之後仍只跑 A；B 的假樹一律明確帶 `--prereg10-dir`。
9. **B 採用之後會變紅的測試（tripwire，預期中）**：`tests/test_t10_player.py::Modes::test_p1_is_b_not_adopted`、`tests/test_t10_b.py::BState::test_real_repository_is_not_adopted_now`、`tests/test_t10_b.py::Cli::test_not_adopted_is_3_and_writes_nothing`（正式模式的部分）、`tests/test_t10_dryrun.py::Nov4Tree::test_b_gpu_seam_is_b_not_adopted`。它們假設 repo 的 B 還沒採用；H10P 採用後照修訂改寫，不是程式錯誤。
10. **其他**：
    - `p1-rehearse` 在正式模式不收 `--now`（只看時鐘）。
    - `t10_player.TOOLS`（報告的工具紀錄）加 `tools/t10_b.py`、`tools/p10_fuse.py`、`tools/p10_apply.py`；正式模式也要求 `tools/t10_b.py` 已 commit。
    - 啟動器進 git：`tools/t10_p1_launch.sh rehearse|seam_val|public|private`（取代第 13 節第 4 條的 `data/prereg10/t10_rehearsal/run.sh`，那個檔已刪）。
    - `.gitattributes` 加一行 `results/prereg06/player/player_val_top5.csv -text`：`p10_apply.apply_core` 逐位元組比這個檔，`core.autocrlf=true` 的新 worktree 會把它取成 CRLF 而誤判「06 沒重現」。

## 15. rev 4 的補充之二（第二次審查，2026-10-12；仍在任何測試資料之前）

1. **合併方式**：`p10b-testday`（和之後任何帶 PREREG_10 修訂列的分支）合併回 main 一律用 `git merge --no-ff`。第 14 節第 6 條的 K5-10（B）看 main first-parent 歷史上第一次出現 `| rev 4 |` 的 commit 時間；fast-forward 時那個 commit 是分支上的 commit，時間不代表「合併進 main」。
2. **`--now`（更正第 14 節第 10 條的寫法）**：正式模式下，`status` 以外的所有指令（seam、apply、submit、p1、p1-rehearse）都不收 `--now`，只看時鐘。
3. **`inputs_bound` 只在有 10B 池時綁 B 的檔**：B 採用但兩池都不是 10B 時，10 的報告不綁 `results/prereg10/**`、演練紀錄、`gpu_seam.json`、P1 報告；所以沒用到的 B 檔（例如沒過、還沒 commit 的 `gpu_seam.json`）不會讓 A 的上傳不符合第 4.4 節。`submit` 讀 B 紀錄（`gpu_seam.json`、演練紀錄）出錯（例如 JSON 壞掉）只記成 B 的錯誤，不中斷 submit。
4. **K3-10 的「判定行第一次 commit」**：找的是 `RESULTS.md` 第一次出現**符合判定行格式**（第 14 節第 2 條）而且是「採用」的那個 commit（逐一讀改過這個檔的 commit），不是任何地方出現「**採用**」字樣的 commit。

## 16. rev 5：B 採用也可以由 PREREG_13 判定（2026-10-13；只加行，在 t10 的程式改動之前 commit）

**白話版**：B 在混色驗證池的池閘門沒過、H10P 沒跑，但 PREREG_13 在全黑視圖上重判、H13P 判定 B′ 採用。本修訂只改「B 採用」從哪裡讀：PREREG_10 的路（H10P）原樣保留，另加 PREREG_13 的路；兩條路都不成立才是「B 沒採用」。B 本身（凍結檔、LDA、w*、T_P1、測試日的 GPU 接縫、tol_P、每池的池閘門、P1 失敗 ≤ 0.5%、只加在用了 06 的池、K6B-10、B 的任何問題都只停 B）一行都不改，照第 13–15 節。

1. **PREREG_13 的路**（`t10_b.b_state` 在 PREREG_10 的路回 `not_marked` 時才查；PREREG_10 的路回其他理由時照第 14 節，不改查 PREREG_13）。下面全部成立才是「採用」，記 `source: "PREREG_13"`：
   - `results/prereg13/RESULTS.md` 剛好一行判定 `H13P：B′ **採用**`（寫法同第 14 節第 2 條：前面可以有項目符號、全形或半形冒號、後面可以接文字；`H13P：B′ **不採用**` 也算判定行）；
   - `results/prereg13/player/score_H13P.json` 符合 PREREG_13 第 6.3 節：B−A 點估計 ≥ 0，而且 95% 下界 > −0.010 × n_view／400（n_view 取 `view_report.json`）；判定行要和分數一致；
   - `results/prereg13/player/view_report.json` 的 `score_allowed` 為真、V1／V2／V3 都過、w* 等於 `frozen_config.json`、它綁定的 `p1_lda.npz`、`frozen_config.json`、`val_report.json` 的 SHA 等於現在的檔，兩個 Top-5 檔的 SHA 等於檔案；
   - 稽核的綁定：`apply --finalize` 會改寫 `view_report.json`，所以不和現在的檔比，改成 `view_report.json` 的 `audit.view_report_sha256_at_audit` 要等於 `prescore_audit.json` 的 `outputs_sha256["view_report.json"]`；正式模式另外要等於稽核 commit 時的 `view_report.json`（`git show <最後一次改 prescore_audit.json 的 commit>:results/prereg13/player/view_report.json`，CRLF 折成 LF 的 SHA）；`audit.sha256` 等於 `prescore_audit.json`；稽核寫 `independent_recompute_identical: true`、`program_errors_affecting_view` 是空的，綁的兩個 Top-5 SHA 和報告相同；
   - 正式模式：上面這些檔和打分紀錄帳都已 commit 而且乾淨；帳上 `prereg13 H13P` 開頭剛好 2 或 4 列，最後一對（A、B）的 `pred_sha256` 等於 `view_report.json` 記的兩個檔；`score_H13P.json` 的最後一次 commit 和 `RESULTS.md` 第一次出現「採用」判定行的 commit 都 ≤ **K3-13（2026-10-25 23:59:59）**，否則 `k3_13`（B 不用）；
   - PREREG_10 這一側照第 13 節第 1 條、第 14 節第 2 條原樣核對：`frozen_config.json` 是凍結紀錄、`p1_lda.npz` 的 SHA、`val_report.json` 完整而且記的 `val_seam_feats.npz` SHA 對、模擬路徑的工具和 `p1_lib`／`p1_extract` 跟凍結時相同、`apply_report.json` 記的 `player_val_top5.csv` SHA 和 w* 對；**只有一條不同**：不要求 `apply_report.json` 的 `score_allowed: true`（改查 `view_report.json`，`apply_report.json` 照 PREREG_13 第 1 節不改）。
   - 對不上 → `inconsistent`（同第 14 節第 2 條：測試日之前 `p1-rehearse` 結束碼 2，測試日 B 不用、記錄）；`H13P：B′ **不採用**` 而且和分數一致 → `h13p_not_adopted`（和 `not_marked` 一樣是登記的「B 沒採用」，不寫檔）。
2. **K5-13**（取代第 14 節第 6 條在 PREREG_13 這條路上的用法）：`source` 是 PREREG_13 時，K5（B）要 `| rev 4 |` 照第 14 節第 6 條成立，**而且**本修訂表的 `| rev 5 |` 列在 HEAD、在 main 的 first-parent 歷史裡第一次出現的 commit 時間 ≤ **K5-13（2026-10-31 23:59:59）**（PREREG_13 第 7 節）。合併方式照第 15 節第 1 條（`--no-ff`）。
3. **seam 不改**：`seam` 照第 13 節第 5 條、第 14 節第 3 條在 seam_val 上重算混色驗證池的 10B，和 `results/prereg10/player/player_val_top5.csv` 比。B 沒變，這個檔仍是 B 在驗證池上的正確輸出（PREREG_13 第 8.2 節）。
4. **`official_log.py`（只加行）**：照 rev 4 的寫法在後面加一個同名的包裝（`t10_report_problems`）：報告的 `b.source` 不是 `PREREG_13` 時，原樣交給 rev 4 的函式（行為逐位元組相同）；是 `PREREG_13` 而且至少一池是 10B 時，`inputs_bound` 要求的 B 檔把 `results/prereg10/player/score_H10P.json` 換成 `results/prereg13/RESULTS.md`、`results/prereg13/player/score_H13P.json`、`results/prereg13/player/view_report.json`，另加第 2 條的 K5-13（`| rev 5 |`）；其他規則（K5-10（B）、K6B-10、GPU 接縫、池閘門、P1 失敗）不變。方法名稱不變（「… [+ P1 LDA (B)] (甲)」）。
5. **報告**：B 的紀錄（`b_record`）加 `source`（`PREREG_10` 或 `PREREG_13`），PREREG_13 時記 `h13p`（點估計、下界、n_view、門檻）取代 `h10p`；`inputs_bound` 在有 10B 池時另綁 `results/prereg13/` 的 `RESULTS.md`、`score_H13P.json`、`view_report.json`、`prescore_audit.json`。
6. **假樹**：dry-run 的 PREREG_13 資料夾是 `--prereg10-dir` 的路徑後面加 `_p13`（正式模式固定是 `results/prereg13`），所以不帶 `--prereg10-dir` 的 A 假樹和既有的 B 假樹（H10P 的路）都看不到 PREREG_13。`tools/t10_b_fake.py` 加 `fake-p13`：做一棵 H10P 沒跑、`apply_report.json` 是 `score_allowed: false` 的假 PREREG_10 樹，旁邊放假的 `results/prereg13`（假的 `score_H13P.json`、`view_report.json`、`prescore_audit.json`、有判定行的 `RESULTS.md`），B 的假樹走這條路演練一次（`tests/test_t10_dryrun.py`）。
7. **會變紅的測試**（第 14 節第 9 條）照本修訂改寫成「repo 的 B 由 PREREG_13 採用」；`tests/test_t10_hygiene.py` 加本列和新常數（`K3_13`、`K5_13`、`B13_ADOPTED_MARK`、`H13P_LEDGER_PREFIX`、`H13P_LEDGER_ROWS`、`B13_REV_ROW`）的檢查。
8. **還沒做的**（不在本修訂）：4090 的演練（`tools/t10_p1_launch.sh rehearse`，K4-13：11/04 11:00 前 commit `results/test10/p1/p1_rehearsal.json`），由主 session 和使用者排時間。

