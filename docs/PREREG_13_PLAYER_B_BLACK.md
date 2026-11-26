# 事前登記 13：凍結的 B 在「全黑視圖」上重判（B′，競賽一）

**白話版**：PREREG_10 的 B（KataGo P1 的 LDA 項，`7a69737` 凍結：`frozen_config.json`、`p1_lda.npz`、w*＝16）在本機驗證池的池閘門沒過（sr(10B) − sr(06 選中 V2) = −0.0199），這一季照登記不用。但本機驗證池黑白混合，官方測試池全黑，而 B 的 LDA 只用黑棋學。本登記**不改 B 的任何東西**，只把評估的池子換成「驗證池只留黑棋」的全黑視圖，在上面重跑一次 06 自己的選法、再加 B，先過不看答案的池閘門，過了才在驗證集打一次分（競賽一第 3 次，共 4 次）。**這個想法是看到 B 沒過之後才有的**，本登記照實寫明，報告也要揭露。

- 只做競賽一（TEAM_11249）；不用競賽二的任何特徵值、測試資料或輸出（同 PREREG_10 第 5 節）。
- 代號：
  - **B**：PREREG_10 第 2 節的 10B 配方（T_P1、w*＝16、CSLS`k*`＝10），程式和凍結檔一律不改。
  - **全黑視圖（視圖）**：第 2 節定義的驗證池子集。
  - **06/視圖**：06 照 PREREG_06 2.3–2.4 的規則在視圖上自己選組合、算 Top-5。
  - **10B/視圖**：06/視圖選中組合的 F_sel 加 w*·z(T_P1)，再 CSLS`k*`。
  - **B′**：「B 在視圖上判定、採用後照 PREREG_10 rev 4 的測試日路徑使用」這件事。B′ 不是新方法。
  - **H13P**：本登記的那一次打分。
- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。競賽一 4 次打分已用 2 次（H2P、H6P）；PREREG_10 的 H10P 沒跑（未送分）。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `37f7dc8` | 第一版，2026-10-13（草稿）。commit 時的狀態：① PREREG_10 B 已在混色驗證池的池閘門停下（`results/prereg10/player/apply_report.json`，`score_allowed: false`），`RESULTS.md` 記「未送分」；② 視圖上的 sr、Top-5、任何分數都沒算過；③ 打分紀錄帳沒有 `prereg13` 列；④ 沒有任何測試資料。起草時看過的東西見第 13 節。 |
| rev 2 | `e5b54dc` | 第 2 階段的實作（`tools/p13_view.py`、`tests/test_p13_view.py`、`tests/test_p13_hygiene.py`）和登記的差異，寫在第 15 節。commit 時：視圖上的 sr、Top-5、閘門都還沒算過（`apply` 沒跑過，`results/prereg13/` 不存在）；第 3 節的恆等視圖（只產生已 commit 的數字）在施工的 worktree 裡單獨跑過一次、全部相同；打分紀錄帳沒有 `prereg13` 列。 |

## 0. 為什麼做這個（事後的動機，照實寫）

- **經過**：PREREG_10 的 B 在官方形狀的模擬池（只用 fit 玩家、全黑）過了閘門（G1 +0.0086、G2 +0.0656），但在本機驗證池的池閘門是 −0.0199，照第 2.5 節「B 這一季不用」，沒打分（`results/prereg10/RESULTS.md`）。
- **事後的推測**（RESULTS.md 已記「只記錄」）：本機驗證池的候選和題目黑白各半（候選 20,109 黑／19,891 白、題目 2,514 黑／2,421 白，`results/prereg06/player/apply_report.json` 的 `games`），B 的 LDA 只在模擬池的黑棋上學（PREREG_10 rev 1 定的配方），對白棋的投影沒有學過。官方測試的候選和題目**只有黑棋**（`docs/DATA.md`「官方資料說明」）。所以混色池上的池閘門量的有一半是官方不會出現的情況。
- **為什麼還值得一試**：B 的主要價值是保險（題目開局和候選不重疊時，模擬 +0.0656）。如果 B 在全黑的真實驗證玩家上不比 06 差，測試日有它比沒有好；如果在全黑上也輸，就確定這一季不用。
- **紀律上的代價**：這是看到 B 失敗之後才改的評估方式，屬於分岔路徑（forking path）。本登記用三件事限制它：B 一個數字都不改；視圖只有一種、在任何視圖上的數字算出來之前寫死；先過不看答案的池閘門，過了才用一次打分機會，而且採用條件在打分前寫死。看了 H13P 的分數之後不再有任何其他視圖、門檻或重跑（第 6.5 節的查錯除外）。

## 1. 不變的東西

- B 的一切照 PREREG_10 凍結：`results/prereg10/player/frozen_config.json`（SHA-256 `8e2bd5a3…`）、`p1_lda.npz`（`92435787…`）、w*＝16、`k*`＝10、`p10_fuse.MOCK_TOOL_FILES` 的程式（T_P1、`z_p1`、`fuse_10b`）。不重學 LDA、不重選 w、不加任何新項或新權重。
- 06 的一切照 PREREG_06 凍結：`results/prereg06/player/frozen_config.json`、`lda_B.npz`／`lda_W.npz`、組合清單（V2、V3、V0；V1 照凍結檔）、選法（2.4：最高 sr、0.003 以內取清單最前面）、`k*`。
- 驗證玩家的 P1 特徵：沿用 `bb007d7` 已抽好、已 commit 的 225 塊（`data/prereg10/val/`，SHA 照 `results/prereg10/feat/val_report.json`；44,935 盤＝候選 40,000＋題目 4,935，0 失敗、0 沒有目標手）。每盤的目標方是那一列的 `color`／`color_i`，所以黑棋盤的 side 0 就是黑方，正是 B 的 LDA 學的那一種。**不需要任何新的 GPU 工作。**
- p02 的 CNN embedding 和開局 token：沿用 `results/prereg02/player/dumps/`（`cand_games.npz`、`query_games.npz`，SHA 等於 06 `apply_report.json` 記的）。
- `results/prereg06/**`、`results/prereg10/**`（含 `apply_report.json`、`player_val_top5.csv`、`RESULTS.md`）都不改。B′ 的輸出只寫在 `results/prereg13/`。

## 2. 全黑視圖（寫死；不看答案）

### 2.1 用什麼判顏色

- 每盤的顏色一律取已 commit 的 p02 dumps：候選盤用 `cand_games.npz` 的 `color`，題目盤用 `query_games.npz` 的 `color`（以 `question_id`、`slot` 對應）。工具另外核對它和兩份棋譜檔的 `color`／`color_i` 逐盤相同，不同就停下（結束碼 2）。
- 只用「這盤目標玩家執黑」（`B`）這一個條件。不看手數、勝負、開局或任何答案。

### 2.2 候選

- 400 位候選**全部保留**（正解一定在候選裡；拿掉候選要先知道答案，所以不做）。
- 每位候選只留自己執黑的盤（`color == B`），白棋盤整盤不進任何計算（IDF、CNN 重心、S2、JO、LDA 項、T_P1、CSLS 的鄰居統計、sr06 的假題和參考都一樣）。
- 起草時數過的結構（第 13 節）：每位候選的黑棋盤數最少 2、第 5 百分位 36、中位數 49、最多 100；不到 10 盤的 3 人、不到 25 盤的 9 人。官方每人 100 盤黑棋，所以視圖的候選盤數大約是官方的一半（揭露，第 12 節）。

### 2.3 題目

- 每一題只留執黑的盤，盤的順序照原題的 slot 順序。
- **進視圖的題**：留下的黑棋盤 ≥ 5 盤（官方每題最少 5 盤）。這樣的題數記為 n_view。起草時數過是 **263 題**（黑棋盤數 5–18；0 盤 5 題、1–4 盤 132 題），工具重數，和 263 不同就停下查（結束碼 2），不是登記的判定。
- 為什麼不收 1–4 盤的題：官方題目最少 5 盤；PREREG_10 的先導顯示盤數越少 B 的增益越大（每題 2 盤 +0.118），收進來會讓視圖偏向官方不會出現、而且對 B 有利的情況。這個門檻取自官方規格，不是調出來的。
- 不進視圖的 137 題：不參加視圖上的任何計算（不進 CSLS、不進 sr06 的盤數清單），只在交件檔裡補列（第 5.2 節）。

### 2.4 06/視圖（06 在全黑池上會怎麼做）

- 視圖就是一個「400 位候選 × 各自的黑棋盤、263 題全黑」的池子。06 在這個池子上**照登記重新自己選**，不沿用混色池選中的 V2：
  1. **sr06**（PREREG_06 2.3）：在視圖的候選盤上做 10 輪假題。盤數清單 K＝視圖 263 題的黑棋盤數（超過 20 的當 20）。每位候選的黑棋盤照同一個雜湊（sha256('c10-prereg06|sr|{player_id}|{game_id}')）排序、依位置切 5 份。
     - **和 2.3 的差異（不得不改）**：2.3 要求每位候選剛好 100 盤、每份 20 盤；視圖的候選只有 2–100 盤，所以用 `p06_style.sr06_rounds(..., strict=False)`（已有的參數，不改函式）：每份是 n/5 盤（依位置切），假題是那一份的前 min(k, 那一份盤數) 盤，參考是其他 4 份；假題是空的那一輪那位候選不出題（同 `sr06_apply` 的 `if q`）。
  2. **選組合**（2.4）：V2、V3、V0（V1 照凍結檔）中 sr 最高、0.003 以內取清單最前面。
  3. **06 的測試日閘門**（PREREG_06 2.7、`t06_player.GATE06_MIN`）：sr(選中) − sr(R02) ≥ 0，R02 是同一批假題上的 Q04 組合（`sim.combo_F(T, "E0", None)`，同 06 的 seam：R02＝Q04）。
  4. 06/視圖的 Top-5：選中組合的 F_sel 做 CSLS`k*`，依候選代號排序取 Top-5（同 06）。
- **為什麼這樣比**：測試日在全黑池上，06 會在那一池自己選組合、過自己的閘門，B 加在它選中的 F_sel 上（PREREG_10 rev 4 第 13 節第 6 條）。視圖照同一個順序做，比的就是「06 在全黑池上實際會交的」對「同一池再加 B」。
- 不做的：PREREG_02 的切法防護（ρ、層級）不在視圖上重算（A 之後，全黑池在第 1–3 層都用 06 的第 1 層計算，所以 06 一定會被算）；一對一開關（06-ON）不進視圖（B 本來就不和開關一起用）。

### 2.5 10B/視圖

- T_P1：已抽好的 P1 塊，只取視圖裡的盤（候選的黑棋盤、263 題的黑棋盤），用凍結的 `p1_lda.npz` 投影（`p10_fuse.project_p1`），每位候選、每一題各自取群組平均，`p10_fuse.z_p1` 做每題 z 分數。
- 10B/視圖＝`p10_fuse.fuse_10b(F_sel, z(T_P1), w*=16, k*=10)`，F_sel 是第 2.4 節選中組合在 CSLS 之前的分數。依候選代號排序取 Top-5。

### 2.6 怎麼實作（寫死，給第 2 階段）

- 新工具 `tools/p13_view.py`，只 import、不改 `p10_apply.py`、`p10_fuse.py`、`p06_player.py`、`p06_style.py`、`p06_sim.py`、`p02_*.py` 的函式。
- 視圖用「群組只收黑棋列」做：每盤的陣列（embedding、O_raw、MemStore）照原樣建，候選群組和題目群組只放 `color == B` 的列、題目只放 n_view 那 263 題。所有池級統計（IDF、z、CSLS）都由群組算，所以和「把白棋列整列刪掉」相同；`tests/test_p13_view.py` 要在假池上核對兩種做法的 Top-5 逐位元組相同。
- sr06 的視圖版要另寫一支和 `p10_apply.sr06_with_p1` 同樣邏輯、只把 `sr06_rounds` 改成 `strict=False` 的函式（不改 `p10_apply.py`）；`strict=False` 在每人剛好 100 盤時要和 `strict=True` 完全相同（測試核對）。
- 讀資料：只在 `p06_style.allow_local()` 裡讀 `data/local/player_val_candidates.csv`、`player_val_questions.csv` 兩份棋譜檔（同 PREREG_06 rev 2 ④、PREREG_10 2.5），SHA 要等於 06 `apply_report.json` 記的。**不讀答案檔**；不 import、不跑 `tools/score.py`（H13P 那一行指令除外，第 6 節）。

## 3. 先重現（視圖的程式沒寫錯）

`p13_view.py apply` 在算視圖之前，用同一支程式、把視圖換成「全部盤、全部 400 題」（恆等視圖，`strict=True`）跑一次，下面全部成立才往下，否則停下修程式（結束碼 1，不是登記的判定）：

- 06 的 Top-5 和 `results/prereg06/player/player_val_top5.csv` 逐位元組相同（SHA `094223f6…`）。
- sr06 各組合到小數 6 位等於 `results/prereg10/player/apply_report.json` 的 `sr06`，`sr06_10B` 也相同，選中 V2。
- 10B 的 Top-5 和 `results/prereg10/player/player_val_top5.csv` 逐位元組相同（`apply_report.json` 記的 SHA `0efd160d…`）；池閘門值等於 −0.019891447034757093（差 ≤ 1e-12）。

這些都是已 commit 的數字，重現不產生新的資訊。

## 4. 池閘門（不看答案；在任何打分之前）

在視圖上，以下全部成立才算過：

| 閘門 | 條件 | 沒過 |
|---|---|---|
| V1 結構 | n_view＝263；400 位候選每人至少 1 盤黑棋；P1 失敗盤（只算重播失敗，視圖內）≤ 0.5% | 停下查（結束碼 2） |
| V2 06 自己 | 06/視圖的 sr(選中) − sr(R02) ≥ 0（PREREG_06 2.7 的測試日閘門） | B′ 不用：全黑池上 06 自己就不會被用，B 也不會加 |
| V3 B 的池閘門 | sr(10B/視圖) − sr(06/視圖選中) ≥ 0，全部 10 輪（同 PREREG_10 2.5、rev 2 第 12 條） | B′ 不用，記為「未送分」，保留打分機會 |

- 門檻和 PREREG_10 2.5 相同（≥ 0），只是池換成視圖。
- 只記錄、不進判定：每個組合在視圖上的 sr06 和 sr06_10B；06/視圖和 10B/視圖第 1 名不同的題數；視圖的候選和題目盤數分布。

## 5. 輸出、稽核、score_allowed

### 5.1 檔案（`results/prereg13/player/`）

- `view_report.json`：視圖的定義和每一條的值（n_view、每位候選的黑棋盤數、題目的黑棋盤數）、第 3 節的重現、第 4 節的閘門、選中組合、`margin`（第 6.3 節）、輸入和工具的 SHA、`score_allowed`。
- `view_06_top5.csv`（H13P 的 A 側）和 `view_10b_top5.csv`（B 側）：

### 5.2 兩個交件檔怎麼組（400 列，`score.py` 才吃得下）

- `score.py` 的 `compare-player` 要求預測檔剛好涵蓋答案檔的 400 題（缺題、多題都是 `SubmissionError`），而且答案檔不能拆（不讀答案）。所以兩個檔都是 400 列、照驗證題目檔的題目順序：
  - 視圖的 263 題：A 是 06/視圖的列，B 是 10B/視圖的列。
  - 不在視圖的 137 題：A、B **都**照抄 `results/prereg06/player/player_val_top5.csv` 那一題的列（逐字相同）。
- 這 137 題在 A、B 完全相同，每題差＝0，所以 `score.py` 的 B−A 平均＝（263／400）×（視圖 263 題的平均差）；配對 bootstrap 也在 400 題上做，那 137 題的差恆為 0。採用條件照這個比例換算（第 6.3 節）。
- 兩個檔都要過 `check_submission.py player … --questions … --candidates …`。

### 5.3 稽核和 score_allowed

- `score_allowed` 全部成立才為真：第 3 節重現；第 4 節三個閘門都過；兩個檔過格式檢查；137 題的列在 A、B 和 06 逐字相同（工具核對）；另一個代理的稽核已 commit。
- **稽核**（`prescore_audit.json`，format `c10-prereg13 player prescore_audit v1`）：看不到答案、不看分數；照本登記獨立重算視圖（顏色、n_view）、06/視圖的選擇和 Top-5、10B/視圖的 Top-5，和兩個檔逐位元組相同；欄位 `program_errors_affecting_view`（清單）、`independent_recompute_identical: true`；綁定兩個檔、`view_report.json`、`p1_lda.npz`、`frozen_config.json`、`val_report.json` 的 SHA；commit 晚於 `view_report.json`。
- 流程同 PREREG_10：`apply` 先寫 `pending_audit`，稽核 commit 後 `apply --finalize` 才變成真。

## 6. H13P：打一次分，和採用條件

### 6.1 指令（寫死；`score_allowed` 為真才跑）

```
.venv/Scripts/python.exe tools/score.py compare-player results/prereg13/player/view_06_top5.csv results/prereg13/player/view_10b_top5.csv data/local/player_val_answers.csv data/local/player_val_candidates.csv --ledger results/score_ledger.csv --note "prereg13 H13P (A = 06 on the black-only view, B = frozen B on the same view; 137 non-view rows identical)" --json results/prereg13/player/score_H13P.json
```

- note 不得含 `prereg10`、`H10P` 字樣（`tests/test_p10_hygiene.py` 在 `score_H10P.json` 不存在時，帳上任何欄位含這兩個字就紅）。
- 帳上多 2 列（A、B 各一列，note 開頭 `prereg13 H13P`），和 `score_H13P.json`、`RESULTS.md` 一起 commit。

### 6.2 對照組核對

- 沒有可以重讀的舊分數（06/視圖是新的對照）。改成：帳上 A、B 兩列的 `pred_sha256` 要等於稽核綁定的兩個檔（CRLF 折成 LF 後），不符就停下查。

### 6.3 採用條件（不劣性，同 PREREG_10 2.6 的形式）

- 記 n_view（`view_report.json`，應為 263）。`score.py` 給的 `B_minus_A`：
  - **點估計 `mean_diff` ≥ 0，而且**
  - **95% 下界 `ci95[0]` > −0.010 × n_view／400**（n_view＝263 時是 −0.006575）。
- 換算的理由：137 題的差恆為 0，400 題上的平均和區間大約是視圖平均的 n_view／400 倍；PREREG_10 的「容許約 1% 的真實損失」是對交件的那些題說的，所以換算到視圖的 263 題。點估計 ≥ 0 不受比例影響。
- bootstrap 照 `score.py`（配對、10,000 次、seed 0）。
- **合理性門檻**：A 或 B 的 400 題平均低於 0.80，或 `mean_diff` < −0.03 時，先查 bug，要指出具體的程式錯誤才算（同 PREREG_10 2.6）。

### 6.4 結果怎麼寫

- `results/prereg13/RESULTS.md` 剛好一行判定：`H13P：B′ **採用**` 或 `H13P：B′ **不採用**`（格式同 PREREG_10 rev 4 第 14 節第 2 條的判定行；數字由另一個代理核對）。

### 6.5 預算

- H13P 是競賽一第 3 次打分（剩 1 次）。查 bug 重跑也算（第 4 次、用完），要先 commit 本登記的修訂（指出具體的程式錯誤），帳上 note 開頭 `prereg13 H13P rerun (PREREG_13 rev N)`；帳上 `prereg13 H13P` 開頭的列只能是 2 或 4。

## 7. 截止點（台灣時間；沒趕上就照表做）

| 截止點 | 條件 | 沒趕上 |
|---|---|---|
| K1-13（無日期） | 本登記 rev 1 的 commit 早於視圖上的任何計算（sr、Top-5、閘門） | 這一季不做 B′ |
| K2-13 10/20（二）23:59 | `p13_view.py`、測試、第 3 節重現、第 4 節閘門、`view_report.json`、稽核、`--finalize` 都 commit | 這一季不用 B′ |
| K3-13 10/25（日）23:59 | H13P、`score_H13P.json`、帳、`RESULTS.md` 判定行都 commit（同 K3-10） | 這一季不用 B′ |
| K5-13 10/31（六）23:59 | B′ 採用時：PREREG_10 rev 5 和 t10 的改動（第 8 節）以 `git merge --no-ff` 合併進 main（看 main first-parent 上第一次出現 `\| rev 5 \|` 的 commit 時間） | 這一季不用 B′ |
| K4-13 11/04（三）11:00 | B′ 採用時：`p1-rehearse` 做完，`results/test10/p1/p1_rehearsal.json` commit（tol_P） | 這一季沒有 tol_P，B 不用 |
| K6B-10 11/14（六）23:59 | 不變（PREREG_10 rev 4）：兩池 P1 和 GPU 接縫已 commit | 10 只有 A |

## 8. 採用後怎麼接上測試日（第 3 階段；只在 H13P 採用時做）

**原則**：測試日的 B 就是 PREREG_10 rev 4／第 14、15 節的 B，一行邏輯都不改（GPU 接縫、tol_P、每池的池閘門 sr(10B) − sr(選中) ≥ 0、P1 失敗 ≤ 0.5%、只加在用了 06 的池、K6B-10、B 的任何問題都只停 B）。唯一要改的是「B 採用」從哪裡讀。

### 8.1 現在為什麼一定回「B 沒採用」

- `tools/t10_b.py` 的 `b_state` 只認 PREREG_10 的路：要 `results/prereg10/player/score_H10P.json`、`RESULTS.md` 的 `H10P：B **採用**`、`apply_report.json` 的 `score_allowed: true`、帳上 `prereg10 H10P` 2 或 4 列、K3-10。這些都不會出現（H10P 沒跑，`apply_report.json` 是 `false` 而且不改），所以現在一律 `not_marked`。
- `tools/official_log.py` 的 `T10_B_INPUTS_REQUIRED` 也要求 `score_H10P.json`。

### 8.2 最小的改法：PREREG_10 rev 5（只加行）＋ t10 的程式

- **PREREG_10 加 `| rev 5 |` 一列和第 16 節**（只准加行，允許）：「B 採用」也可以由 PREREG_13 判定：`results/prereg13/RESULTS.md` 剛好一行 `H13P：B′ **採用**`、`score_H13P.json` 符合第 6.3 節（含 n_view 換算的下界）、`view_report.json` 的 `score_allowed` 為真、帳上 `prereg13 H13P` 2 或 4 列、兩者都在 K3-13 前 commit；凍結檔、LDA、`val_report.json`、`val_seam_feats.npz`、模擬路徑工具的核對照 rev 4 原樣。PREREG_10 的路（H10P）不變，兩條路都不成立才是「沒採用」。
- **`tools/t10_b.py`**（10 的工具，不在凍結清單；`test_t10_hygiene.py` 釘住的常數不能動）：加 `K3_13`、`K5_13`、`B13_ADOPTED_MARK = "H13P：B′ **採用**"`、`H13P_LEDGER_PREFIX = "prereg13 H13P"`、`H13P_LEDGER_ROWS = (2, 4)`；`b_state` 在 PREREG_10 的路回 `not_marked` 時再查 PREREG_13 的路，成立就回 `adopted`（記 `source: "PREREG_13"`），只有 PREREG_13 的路時不要求 `apply_report.json` 的 `score_allowed: true`（改查 `view_report.json`），但仍要求 `apply_report.json` 記的 `player_val_top5.csv` SHA 和檔案相同；`k5_b_state` 對 PREREG_13 的路看 `| rev 5 |`。
- **seam 不用改**：`seam_10b` 在 seam_val 上重算混色驗證池的 10B，比 `results/prereg10/player/player_val_top5.csv`（`0efd160d…`）。B 沒變，這個檔仍是 B 在驗證池上的正確輸出，當 CPU 接縫照用。
- **`tools/official_log.py`**（只加行）：照 rev 4 的寫法在後面加同名包裝：B 的來源是 PREREG_13 時，`inputs_bound` 要求的 B 檔把 `score_H10P.json` 換成 `results/prereg13/RESULTS.md`、`player/score_H13P.json`、`view_report.json`；其他規則（K5、K6B-10、接縫、池閘門）不變。方法名稱不變（「… [+ P1 LDA (B)] (甲)」）。
- **`tools/t10_b_fake.py`**：加做假的 `results/prereg13`（假的 `score_H13P.json`、有判定行的 `RESULTS.md`），B 的假樹改走這條路也要演練一次。
- **`docs/TESTDAY_10.md`**（10 的手冊，不在凍結清單）：B 一節的「時機」改成「H10P 或 H13P 採用」。
- **會變紅的測試（PREREG_10 rev 4 第 14 節第 9 條，預期中）**：`tests/test_t10_player.py::Modes::test_p1_is_b_not_adopted`、`tests/test_t10_b.py::BState::test_real_repository_is_not_adopted_now`、`tests/test_t10_b.py::Cli::test_not_adopted_is_3_and_writes_nothing`（正式模式那幾行）、`tests/test_t10_dryrun.py::Nov4Tree::test_b_gpu_seam_is_b_not_adopted`。照 rev 5 改寫成「repo 的 B 由 PREREG_13 採用」；`tests/test_t10_hygiene.py` 加 rev 5 一列和新常數的檢查。
- 經獨立審查、全套測試和閘門過、`C10_SLOW=1` 的 `test_t10_dryrun.py` 過之後，K5-13 前 `--no-ff` 合併。

### 8.3 GPU（4090）

- **演練**（K4-13，11/04 11:00 前；約 2 分鐘）：合併之後 `tools/t10_p1_launch.sh rehearse`，量 m，tol_P＝max(0.05, 2.5 × m)，紀錄 commit 時間要早於 11/04 11:00（PREREG_10 rev 4 第 13 節第 4 條、第 14 節第 5 條）。開跑前 `nvidia-smi`；10/24 前 C4 優先。
- **測試日**（約 5.5 小時）：競賽二的 07 HS（11/04 夜，3–4 小時）、08 P1（11/05，約 0.6 小時）先跑；之後 `t10_p1_launch.sh seam_val`（約 2 分鐘）→ `public`（約 2.7 小時）→ `private`（約 2.7 小時），排夜間；K6B-10（11/14）前 commit。4090 不能用時沒有備援（PREREG_11 只涵蓋 07／08），B 這一季不用、10 只有 A。

## 9. 防誤改閘門和會變紅的測試

- **新加** `tests/test_p13_hygiene.py`，進 `AGENTS.md` 的閘門清單：本登記只准加行（修訂表的 `| rev N |` 列可以填 commit 編號）；`results/prereg06/**`、`results/prereg10/**`、`tools/p10_*.py`、`tools/p06_*.py`、`tools/p02_*.py`、`tools/score.py` 從本登記 commit 起不改；H13P 之前帳上沒有 `prereg13` 列；H13P 之後 `prereg13 H13P` 開頭剛好 2 或 4 列、A／B 的 `pred_sha256` 等於稽核綁定的檔；`tools/p13_view.py` 不 import `score`、`torch`、`p1_lib`、`p1_extract`、競賽二的工具，不寫出答案檔的名字。
- **H13P 的 2 列會讓這幾個凍結的測試變紅**（測試檔在 `tests/test_p0[1-9]*.py`，凍結，不能改）：
  1. `tests/test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`（本來就是唯一允許的失敗，多印 2 列）。
  2. `tests/test_p06_player_hygiene.py::LedgerRows::test_later_rows_only_prereg07`（H6P 之後只准 `prereg07`、`prereg08 H8R`）。
  3. `tests/test_p07_hygiene.py::LedgerRows::test_prereg07_rows`（07 之後只准 H8R）。
  4. `tests/test_p08_hygiene.py::LedgerRows::test_h8r_rows`（H7R 之後只准 H8R）。
  - 第 2–4 項都只在 `C10_T04_GATE=1`（閘門）時跑。H13P 之後允許的失敗擴成這 4 項，條件是：失敗訊息裡多出的列正好是帳上已 commit 的 `prereg13 H13P` 列（2 或 4），`test_p13_hygiene.py` 過（它把這幾列釘死）。這一條在打分前寫進 `AGENTS.md`「唯一允許的失敗」那一段。
  - **和競賽二測試日手冊的關係（主 session 10/13 決定：接受）**：`docs/TESTDAY_07.md`、`TESTDAY_08.md`、`TESTDAY_09.md`（競賽二，凍結）寫「唯一允許的失敗是 `test_t04_hygiene…`」。H13P 之後競賽二的測試日閘門會多第 2–4 項預期中的紅；這三份不能改原文，以本節、`AGENTS.md`「唯一允許的失敗」那一段（打分前更新）和 `test_p13_hygiene.py`（把 `prereg13 H13P` 列釘死）為準。前例：PREREG_10 2.6 對 H10P 的 2 列也預定照 PREREG_06 rev 3 第 28 條的寫法處理。測試日跑閘門時，紅燈只要正好是這 4 項、多出的列正好是帳上已 commit 的 `prereg13 H13P` 列，就算通過；其他任何紅燈照舊停下查。

## 10. 檔案清單（第 2、3 階段）

| 檔 | 動作 | 允不允許 |
|---|---|---|
| `docs/PREREG_13_PLAYER_B_BLACK.md` | 加（本檔）；之後只准加行 | 允許（新檔） |
| `tools/p13_view.py`、`tests/test_p13_view.py`、`tests/test_p13_hygiene.py` | 加 | 允許（新檔） |
| `results/prereg13/player/view_report.json`、`view_06_top5.csv`、`view_10b_top5.csv`、`prescore_audit.json`、`score_H13P.json`；`results/prereg13/RESULTS.md` | 加 | 允許（新檔） |
| `results/score_ledger.csv` | `score.py` 加 2 列（或 4） | 允許（只增不刪，VALIDATION 第 6 條）；第 9 節的 4 個測試變紅，打分前寫進 `AGENTS.md` |
| `.gitattributes` | 加一行 `results/prereg13/** -text` | 允許（只准加行） |
| `AGENTS.md` | 閘門加 `test_p13_hygiene.py`；允許的失敗照第 9 節；現況 | 允許（規則檔，照規則更新） |
| `PORTFOLIO_MAP.md` 第 0 節 C10 那一行 | 改 | 允許（只改既有那一行） |
| `docs/PREREG_10_PLAYER_GUARD_P1.md` | 第 3 階段：加 `\| rev 5 \|` 一列和第 16 節 | 允許（只准加行） |
| `tools/t10_b.py`、`tools/t10_b_fake.py`、`tests/test_t10_b.py`、`tests/test_t10_player.py`、`tests/test_t10_dryrun.py`、`tests/test_t10_hygiene.py`、`tests/test_t10_official_log.py`、`docs/TESTDAY_10.md` | 第 3 階段：改（第 8.2 節） | 允許，要先 commit PREREG_10 rev 5；`test_t10_hygiene.py` 釘住的常數（`B_REV_ROW`、`B_ADOPTED_MARK`、`K3_10`、`K6B_10`、`TOL_P_*`、`H10P_LEDGER_ROWS`）不改 |
| `tools/official_log.py` | 第 3 階段：只加行（同名包裝） | 允許（只准加行） |
| `tools/p10_fuse.py`、`p10_apply.py`、`p10_p1.py`、`p06_*.py`、`p02_*.py`、`score.py`、`check_submission.py` | 只 import | **不准改**（凍結或 b_state 核對 SHA） |
| `results/prereg06/**`、`results/prereg10/**`（含 `apply_report.json`、`RESULTS.md`）、`docs/PREREG_06` 原文、`docs/VALIDATION.md`、`docs/TESTDAY*.md`（10 以外） | 不動 | **不准改** |
| `data/local/` | 工具只讀兩份棋譜檔；答案檔只有第 6.1 節那一行讀 | 其他一律不碰 |

## 11. 任何一步失敗時

- 第 3 節重現不符：修程式（程式錯誤），不是判定；K2-13 前修不好 → B′ 不用。
- 第 4 節 V2 或 V3 沒過：B′ 不用，`RESULTS.md` 記「未送分」，競賽一打分仍只用 2 次。
- 稽核找到錯誤、K2-13／K3-13／K5-13／K4-13 沒趕上、H13P 不採用：B′ 不用。
- B′ 不用時：06 和 A 完全照現在（PREREG_06、PREREG_10 rev 3），t10 的 B 路徑照舊回「B 沒採用」，不需要演練和測試日的 4090。
- 採用之後測試日的任何 B 問題：照 PREREG_10 rev 4 第 14 節，只停 B，10 照 A 交。

## 12. 風險與揭露（寫進 RESULTS 和報告）

- **事後的動機**：視圖是看到 B 在混色池失敗之後才提的（第 0 節）。報告寫明：「B 在事前登記的混色驗證池沒過池閘門；事後改用全黑視圖重判，視圖和門檻在任何視圖數字算出來之前寫死，用掉競賽一第 3 次打分。」
- **分岔路徑**：可以選的地方（候選要不要只留黑、題目要不要只留黑、題目最少幾盤、06 要不要重選、比例換算）都在本登記一次寫死，只有一種視圖；不再試別的視圖、門檻或 w。仍然是「第二次機會」，所以就算採用，證據強度也比一次就過的弱，報告要這樣寫。
- **為什麼全黑視圖是對官方測試的原則性模擬**：官方候選和題目只收執黑棋譜；B 的 LDA 只學黑棋；視圖的唯一條件「目標玩家執黑」直接來自官方規格，不碰答案、不看分數，06 和 B 用的都是同一套凍結程式和同一個重選規則。
- **視圖和官方的差異**：候選每人只有 2–100 盤黑棋（中位數 49，官方 100）；題目 5–18 盤（官方 5／10／15／20 各 100 題）；n_view 263（官方 400）；仍是一題對一人；sr06 的假題變短（`strict=False`）。候選盤數少、題目盤數不同，B 和 06 的差在官方可能不同。
- **檢定力**：263 題的不劣性只能抓「明顯變差」，不能證明 B 加分；B 的增益證據仍是模擬（G1、G2）。
- **已經知道的東西**：混色池上的 sr06、sr06_10B、池閘門 −0.0199、10B 和 06 第 1 名不同 30 題、06 的單色檢查（假題只用黑棋、參考黑白混合：V2 0.934273、R02 0.720885）都已 commit、看過。視圖上的 10B 沒算過。
- **跨競賽**：只共用 KataGo 的抽取程式（同 PREREG_10 第 5 節）。H13P 的帳列會讓競賽二的測試日閘門多 3 項預期中的紅（第 9 節）。
- 生成式 AI 協助的部分照規則揭露（`competitions/_drafts/disclosure-template.md`）。

## 13. 起草時看過的東西

- 已 commit 的檔：PREREG_06、PREREG_10（含 rev 2–4、第 11–15 節）、`results/prereg10/RESULTS.md`、`apply_report.json`、`val_report.json` 的欄位、`results/prereg06/player/apply_report.json`、`tools/score.py`、`p06_player.py`、`p06_style.sr06_rounds`、`p10_apply.py`、`t10_b.py`、相關的防誤改測試。
- **結構計數**（不含任何分數或特徵）：從已 commit 的 `results/prereg02/player/dumps/cand_games.npz`、`query_games.npz` 只讀 `player_id`／`question_id`／`color`，數每位候選的黑棋盤數和每題的黑棋盤數（第 2.2、2.3 節的數字）。沒有讀 `data/local`、沒有算 sr、沒有算任何 Top-5 或分數。

## 14. 順序（`PY`＝`.venv/Scripts/python.exe`，Git Bash，加 `PYTHONDONTWRITEBYTECODE=1`）

1. 本登記 rev 1 commit，再用一個 commit 填上編號（K1-13）。
2. 寫 `tools/p13_view.py`（`apply`、`apply --finalize`、`status`）、`tests/test_p13_*.py`；獨立審查；閘門（加 `test_p13_hygiene.py`）過才 commit；實作和登記的差異寫成 rev 2，在 `apply` 之前 commit。
3. `PY tools/p13_view.py apply`（CPU，幾分鐘）→ commit `view_report.json`、兩個檔。閘門沒過：寫 `RESULTS.md`（未送分）→ commit，結束。
4. 稽核（另一個代理）→ commit → `apply --finalize` → commit（K2-13）。
5. `AGENTS.md` 的允許失敗照第 9 節更新 → commit；`score_allowed` 為真才跑第 6.1 節的指令；帳、`score_H13P.json`、`RESULTS.md`（數字由另一個代理核對）、地圖、`AGENTS.md` 現況一起 commit（K3-13）。
6. 採用：照第 8 節做 PREREG_10 rev 5 和 t10 的改動（分支、worktree 放 `competitions\` 以外），審查、演練、`--no-ff` 合併（K5-13）→ `t10_p1_launch.sh rehearse` → commit 演練紀錄（K4-13）。

## 15. rev 2：實作和登記的差異（在 `apply` 之前，2026-10-13）

- 實作：`tools/p13_view.py`（`apply`、`apply --finalize`、`status`）、`tests/test_p13_view.py`、`tests/test_p13_hygiene.py`（進 `AGENTS.md` 的閘門，二十個檔）、`.gitattributes` 加 `results/prereg13/** -text`。分支 `p13-view`，worktree 在 `competitions\` 以外。
- **獨立審查**：另開一個代理（Claude 子代理，沒有參與撰寫）對照本登記和凍結程式逐條審查，沒有找到會改變判定的錯誤；它另外在假池上把兩位候選壓到只剩 2、3 盤黑棋，「群組只收黑棋」和「刪白棋列」的 sr、選中組合、兩份 Top-5 都相同。它提出的修正都已照做：第 3 節的暫存檔改放系統暫存資料夾（不在 `results/prereg13/` 留空資料夾）；先核對盤序再核對顏色；`apply` 要求本登記 rev 2 已 commit；參考檔 SHA 加前綴核對；測試加「短候選」假池、finalize 的判定；`test_p13_hygiene.py` 的帳列照 A、B 順序比、排除 `tools/p10_p1.py`；刪掉沒用到的 import。

和登記原文不同或登記沒寫到的地方（逐條）：

1. **第 3 節的比對方式**：兩份參考檔（06、10B 的 `player_val_top5.csv`）讀進來先把 CRLF 折成 LF 再和工具寫出的位元組比（main 上兩份都是 LF，等於逐位元組；新 checkout 的 worktree 裡 `results/prereg10/player/player_val_top5.csv` 沒有 `-text`，可能是 CRLF）。另外加三條核對：參考檔的 SHA 要等於 `results/prereg10/player/apply_report.json` 記的（06 那份對 `p06_reproduction.reference_sha256`，10B 那份對 `outputs_sha256`），而且前綴是 `094223f6`／`0efd160d`；選中組合要等於 p10 報告的而且是 V2（常數 `REPRO_SELECTED`）；池閘門同時和 −0.019891447034757093 及 p10 報告的值比（差都 ≤ 1e-12）。恆等視圖的 Top-5 只寫在系統暫存資料夾、讀回後刪掉，不寫進 `results/prereg13/`。
2. **第 3 節之前已經單獨跑過一次恆等視圖**：施工時在 worktree 裡只呼叫 `reproduce`（不算視圖）確認程式沒寫錯，結果第 3 節每一條都相同（06 Top-5 `094223f6…`、sr06 V2 0.975888／V3 0.969734／V0 0.922136／R02 0.899872、sr06_10B 相同、選 V2、10B Top-5 `0efd160d…`、池閘門 −0.019891447034757093）。這些都是已 commit 的數字，沒有產生視圖上的任何數字。`apply` 照登記會再跑一次。
3. **沒寫檔的停法**：第 3 節不符（結束碼 1）、V1 不過、顏色或盤序和 `val_report.json`／棋譜檔不符（結束碼 2）時，`results/prereg13/player/` 不寫任何檔。p02 的 `Data` 自己也核 dumps 和棋譜檔的顏色，它的 `SystemExit` 轉成結束碼 2。
4. **閘門沒過也寫檔**：V2 或 V3 沒過、格式檢查沒過、137 題不一致時，照樣寫出兩個 Top-5 檔和 `view_report.json`（`score_allowed: false`、列出原因），結束碼 1（同 `p10_apply.py`）。V2 沒過時 V3 仍照算、照記。這兩個檔在 `score_allowed` 為真之前不會被打分。
5. **finalize**：先核對稽核綁的 `view_report.json` SHA（CRLF 折成 LF）等於當時 commit 的那份、兩個 Top-5 的 SHA 等於報告記的，再改寫 `view_report.json`（加 `audit`、`finalized`、`finalized_by_tool`，`audit.view_report_sha256_at_audit` 記稽核綁的那個 SHA）。稽核判有程式錯誤時 `score_allowed: false`、結束碼 1（`p10_apply.py` 一律 0）；其他拒絕結束碼 2。判定本身沿用 `p10_apply.finalize_decision`。`status` 只印狀態、不寫檔、結束碼 0。
6. **SHA 的算法**：`view_report.json` 的 `outputs_sha256` 是兩個 Top-5 檔原始位元組的 SHA；`.gitattributes` 的 `-text` 讓它們永遠是 LF，所以和 CRLF 折成 LF 的 SHA 相同。稽核綁 `view_report.json` 用 CRLF 折成 LF 的 SHA。
7. **多加的前置核對**：`apply` 要求本登記已 commit 且沒有未 commit 的修改、rev 1 和 rev 2 都在 commit 裡；`results/prereg10/player/apply_report.json` 和 `player_val_top5.csv` 已 commit；工具綁定 `p10_apply.APPLY_TOOL_FILES` 加 `tools/p13_view.py`，全部要 commit 且乾淨；模擬路徑的工具要等於 `frozen_config.json` 記的（同 `p10_apply.py`）；`frozen_config.json`、`p1_lda.npz`、`val_report.json`、225 塊的 SHA 照 `p10_apply.apply_prerequisites` 核對。
8. **V1 的 P1 失敗比例**：分母是視圖裡的盤（400 位候選的黑棋盤＋n_view 題的黑棋盤），分子是其中 `failed` 的盤（重播失敗；沒有目標手的盤另記 `no_target_moves`，不算失敗）。
9. **2.6「和刪白棋列相同」**：數學上相同（所有池級統計都由群組算），正式路徑只用群組的做法。測試在兩個假池上核對（每人 100 盤的假池；同一池把兩位候選壓到 2、3 盤黑棋）：題目、每人盤數、選中組合相同，06/視圖和 10B/視圖的 Top-5 逐位元組相同，sr06 和 sr06_10B 的差 ≤ 1e-12。sr 用 1e-12 容差而不要求位元組相同，是因為刪掉白棋列時 MemStore 的 JO 詞彙表欄位順序會變，稀疏內積的加總順序可能讓最後一位浮點數不同。`strict=False` 和 `strict=True` 在每人剛好 100 盤時，`sr06_rounds` 的輸出和 sr06（含 10B）的平均都完全相同（測試核對）；另外核對 `sr06_view(strict=True)` 和 `p10_apply.sr06_with_p1` 完全相同。
10. **`view_report.json` 多記的欄位**（不進判定）：`reproduction`（第 3 節每一條的值）、`view.definition`、每位候選和每題的黑棋盤數與摘要、題目依黑棋盤數的分組（0／1–4／≥5）、`changed_top1_06_vs_10B_in_view`、`nonview_rows_identical`、`results/prereg10/player/apply_report.json` 的 SHA、各前置檔的 commit。
11. **`test_p13_hygiene.py`**：凍結清單另加 `tools/check_submission.py`（第 10 節「不准改」）；`docs/PREREG_06`、`docs/VALIDATION.md`、`docs/TESTDAY*.md` 由既有的閘門管，不重複。H13P 之後帳上 A、B 兩列照順序（A＝`view_06_top5.csv`、B＝`view_10b_top5.csv`）比最後一對的 `pred_sha256`。工具只准 import 標準函式庫、numpy、scipy 和 `tools/p10_*`（`p10_p1` 除外）、`p06_*`、`p02_player_*`。
12. 第 9 節「允許的失敗擴成 4 項」照登記在打分前才寫進 `AGENTS.md`，本次沒寫；本次 `AGENTS.md` 只在閘門指令加 `tests/test_p13_hygiene.py`。
