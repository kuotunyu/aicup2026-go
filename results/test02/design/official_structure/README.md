# 官方結構的模擬與程式檢查（workflow `wf_86f416aa-695`，2026-09-28）

## 1. 競賽一全黑結構的開局檢索模擬（只用 fit 玩家）

競賽一的開局檢索換成官方結構（全黑棋）後，**比原本混合執色的模擬更好也更穩**。raw、both、N=10 的官方分數約 0.924（隨機切）到 0.929（檔案順序切），混合執色隨機切是 0.8953。原本「照檔案順序切會大掉」的風險不見了，col、nocol、both 三種 token 幾乎沒有差別。已登記的切法防護在所有黑棋池、兩種方向都停在 level 1。

## 做法

- **池子怎麼抽**：官方結構要每人至少 100＋k 盤黑棋。fit 玩家中黑棋盤數達標的人數如下：

  | 黑棋盤數 | ≥105 | ≥110 | ≥115 | ≥120 |
  |---|---|---|---|---|
  | 人數 | 1,082 | 1,029 | 975 | 926 |

  - 每一級都至少有 40 人，最少的是 A 級，依序為 79／70／65／57。
  - 這 1,082 人在檔案裡全部是黑棋排在白棋前面。
- **主設計 P120**：從黑棋 ≥120 盤的人裡每級抽 40 人，挑法同原本的 sim（sha256('seed|pid')）。k 取 5／10／15／20 各 100 題，隨機打散，跑 5 個 seed。
- **變體 P105**：從黑棋 ≥105 盤的人裡抽，k 上限設為黑棋盤數減 100，每種 k 仍各 100 題，跑 3 個 seed。
- **兩種切法**（都只在該玩家的黑棋裡切，候選取前 100 盤，題目取接下來 k 盤）：
  - (a) 先隨機打亂再切。
  - (b) 照檔案順序切。
- **對照組**：同一批 P120 玩家、同樣的 k，但黑白棋混在一起隨機切。
- **核對**：用改寫後的程式重跑原本混合執色 sim 的 12 格（random、seed 0–1、raw/canon × col/nocol/both），和 `opening_player_sim.json` 到小數第 4 位全部相同。
- canon（前 6 手）和 p02_moves.canon(mv,10) 在 10 萬盤 fit 棋譜上沒有任何一盤結果不同，所以下表的 canon 同時適用兩種定義。

## 主表：P120、N=10、5 個 seed 平均

每格是官方分數／top1／top5。

| 切法 | 方向 | col | nocol | both |
|---|---|---|---|---|
| 黑棋隨機 | raw | 0.9240/0.9055/0.9735 | 0.9186/0.9010/0.9715 | **0.9242**/0.9065/0.9730 |
| 黑棋隨機 | canon | 0.8931/0.8725/0.9600 | 0.8864/0.8625/0.9545 | 0.8912/0.8690/0.9590 |
| 黑棋檔案順序 | raw | 0.9285/0.9130/0.9770 | 0.9265/0.9100/0.9770 | **0.9291**/0.9125/0.9785 |
| 黑棋檔案順序 | canon | 0.8995/0.8785/0.9615 | 0.8986/0.8770/0.9645 | 0.8982/0.8760/0.9645 |
| 對照：混合執色隨機 | raw | 0.8963 | 0.8451 | 0.8913 |
| 對照：混合執色隨機 | canon | 0.8761 | 0.8215 | 0.8724 |

- both、raw 在黑棋隨機切下，5 個 seed 分別是 0.9184、0.9213、0.9263、0.9232、0.9319。

**依 k 分開看**（both，每格是官方分數／top1／top5）：

| 切法、方向 | k=5 | k=10 | k=15 | k=20 |
|---|---|---|---|---|
| 黑棋隨機 raw | 0.8112/0.772/0.922 | 0.9349/0.920/0.978 | 0.9710/0.962/0.994 | 0.9797/0.972/0.998 |
| 黑棋隨機 canon | 0.7459/0.710/0.876 | 0.9065/0.880/0.974 | 0.9483/0.932/0.992 | 0.9642/0.954/0.994 |
| 黑棋檔案順序 raw | 0.8148/0.780/0.932 | 0.9321/0.914/0.982 | 0.9789/0.970/1.000 | 0.9905/0.986/1.000 |

**N=20 的紀錄**（col／nocol／both）：

| 切法 | raw | canon |
|---|---|---|
| 黑棋隨機 | 0.9290 / 0.9232 / 0.9267 | 0.9028 / 0.8926 / 0.9014 |
| 黑棋檔案順序 | 0.9305 / 0.9266 / 0.9315 | 0.9003 / 0.8921 / 0.9024 |
| 對照：混合執色 | both 0.8794 | — |

**P105 變體**（3 個 seed，N=10，col／nocol／both）：

| 切法 | raw | canon |
|---|---|---|
| 黑棋隨機 | 0.9258 / 0.9204 / 0.9267 | 0.9017 / 0.8919 / 0.8963 |
| 黑棋檔案順序 | 0.9290 / 0.9288 / 0.9294 | 0.8987 / 0.8933 / 0.8988 |

## 和混合執色的舊數字比（`opening_player_sim.json`，raw，col／nocol／both）

- 隨機切：0.901 / 0.8394 / 0.8953；canon 是 0.8706 / 0.8061 / 0.8663。
- 檔案順序切：0.7232 / 0.7713 / 0.7436。
- 執色分開：0.1973 / 0.2479 / 0.2082。

**怎麼解讀**

- **全黑棋比混合執色高約 0.03**，原因是同一種執色。同一批 P120 玩家混合執色時是 0.8913，和舊的 0.8953 差不多，所以提升不是來自「只挑棋譜多的玩家」。
- **舊的檔案順序切會掉分，是執色造成的**，不是時間先後造成的。全黑棋時照檔案順序切是 0.9291，還略高於隨機切的 0.9242。
- **三種 token 的差距縮到 ≤0.006**：全黑棋時 col 和 nocol 幾乎是一一對應的。
- **canon 仍比 raw 低約 0.03**，和混合執色時一樣。
- **官方沒有執色分開的情形**，0.2 那種最壞情況在官方結構下不會出現。

## ρ 和切法防護

ρ 直接用 `tools/p02_player_open.py` 的 `pool_open_stats` 和 `guard_level` 算，門檻用 `guard_calibration.json` 登記的值：

| | θ₁ | θ₂ |
|---|---|---|
| raw | 0.9630 | 0.8268 |
| canon | 0.9706 | 0.8637 |

各種池子的 ρ（平均，括號內是各 seed 的範圍）：

| 池子 | raw | canon |
|---|---|---|
| P120 黑棋隨機 | 1.0019（0.9967–1.0137） | 1.0038（0.9839–1.0156） |
| P120 黑棋檔案順序 | 1.0123（0.9941–1.0273） | 1.0022（0.9941–1.0189） |
| P105 黑棋隨機 | 1.0030 | 1.0043 |
| P105 黑棋檔案順序 | 1.0157 | 1.0053 |
| 對照：混合執色 | 1.0175 | 1.0143 |

- **所有池子、兩種方向都是 level 1**（黑棋隨機切下，raw 5 個池、canon 5 個池全部如此）。
- 離門檻最近的一格是 canon seed 4：0.9839 對 θ₁ 0.9706，只高 0.013。raw 最低 0.9967，比 θ₁ 高 0.034。
- 全黑棋時 b_c＝b_q＝1.0，執色比例條件不會觸發。
- 全黑棋時 max cosO 的中位數約 0.30–0.31（raw），比混合執色的約 0.235 高，但真題和自我檢索假題一起升高，比值還是約 1。
- 真題 max cosO 中位數依 k 分（黑棋隨機、raw）：k=5 是 0.2415，k=10 是 0.2886，k=15 是 0.3357，k=20 是 0.3642。
- 用 p02 程式本身算的 real_open_score（both）是 raw 0.9242、canon 0.8912，和上表完全一致。
- **限制**：自我檢索假題的 k 用 randint(5,20)，和官方的 5／10／15／20 不同，但平均都是 12.5。官方結構下 X 情境不存在，所以 θ₂ 那一層實際上不會用到。

## 資料紀律

- 只透過 `load_rows(part="fit")` 讀訓練資料，沒開任何 answers 檔，沒跑 `score.py`，只用 CPU。
- 已用 `sys.dont_write_bytecode` 避免寫 .pyc 進 repo，沒有改 repo 裡任何檔案。
- `git status` 顯示 `docs/DATA.md` 有未 commit 的修改，這在我開始前就已經存在，不是這次造成的。
- 完整執行 1,123 秒，其中讀資料 301 秒。

檔案都在 `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\official\c1_blackonly\`：
- `official_c1_blackonly_sim.py`：主要腳本
- `official_c1_blackonly_sim.json`：完整結果，含每個 seed、每個 k、guard 理由
- `run.log`：執行紀錄
- `count_black.py`：黑棋盤數統計
- `check_canon.py`：canon6 和 canon10 的比對


## 2. 程式與登記的結構假設檢查

唯讀審查完成（repo 沒改、沒 commit；沒開答案檔、沒跑 score.py、沒用 GPU）。模擬只用訓練玩家（c10data.load_rows 的 train／fit／inner）和已 commit 的 inner dumps，結果只寫在 scratch `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\official\audit_c1\`：`sim_c1.py/.json/.log`、`sim_t3b.py/.json`、`rank_colour.py/.json`、`t1_shape/{A_unused_colour_empty,B_unused_colour_filled}/`。這些數字都沒 commit，而且是 scratch 模擬。

結論：
(1) 有一個可能擋路的問題。官方說明寫 color_1…color_20「一律是 B」。如果官方檔真的把 num_games 之後的空 slot 也填上執色，T1.3 會判 FAIL_STRUCT，TESTDAY 規定「停、先寫修訂、修訂前不做任何預測」，兩場的第一次上傳都會卡住。我用官方形狀的假檔實測過：空 slot 的執色留空 → PASS，填 B → 兩場都 FAIL_STRUCT。修法是在 11/04 前登記一個新的白名單正規化（rev 8）。
(2) 有一個會悄悄失準的問題：T3。候選全黑時，登記的 T3（參考集是黑白混合）幾乎失去偵測力。TPR 從 0.38／0.43 掉到 0.125／0.136。就算一半候選是訓練玩家，判定「重疊」的機率也是 0；可是報告仍會寫「PASS：沒有重疊」。改用只含黑棋的參考集可以恢復偵測力（π=0.5 時判定重疊的機率 0.995／0.999），但要重算虛無分布和 pins，並在 11/04 前寫成修訂。沒有會誤報重疊的問題：假陽性 0.7%／0.8%。
(3) T2(a) 會把「同一盤棋、訓練集裡記的是對手那一方」也算成複製。後果只是開 split v2，不擋上傳，建議加一個只記錄的分色欄位。
(4) 其餘多半只是「不同」，不是錯：
- 執色防護在全黑結構下永遠不會觸發。全黑、隨機切、官方盤數 {5,10,15,20} 的模擬中 ρ 在 1.004–1.036，都高於 θ₁，會落在第 1 層。
- 只用開局的得分在全黑池反而較高（raw 0.919–0.932，混色是 0.880–0.925）。
- T1.5 在全黑池最低 0.984，門檻 0.95，不會誤觸 FAIL_OPENING。
- T4、T5 只記錄的欄位會顯示「落在虛無範圍外」、chi2≈1200。這是結構造成的，不代表異常。
- 競賽二的 R20 在單色題：隨機執色的單色題 0.6050，混色 0.6040（inner、樣本內），平均沒有退步；但黑題 0.590、白題 0.614，有執色不對稱。
- PREREG_03 的 κ 檢查和閘門都是在混色題上算的，要在還沒有任何 prereg03 輸出的現在，補登記單色題的參照。
- 假測試集和 dry-run 測試都是混色、5–20 均勻抽，完全沒測到官方結構。
- 更正一個原本的假設：全黑時「含執色」和「不分執色」兩套 token 並不等價（詞彙 47,122 對 19,620，cos 最大差 0.046），但不影響表現。

所有會改判定或改測試日程式的修正（第 1、2、3 條，以及 PREREG_03 的補登記）都要在 11/04 11:00 前寫成修訂：PREREG_02 rev 8、PREREG_03 rev 3。另外注意：工作樹裡 docs/DATA.md 有一份還沒 commit 的修改，指向 `results/prereg02/checks/official_structure/`，但 results/prereg02/** 依 TESTDAY 是凍結區，建議改放 results/test02/design/ 底下。


- **[blocker]** `tools/t02_checks.py` t1_check T1.3 題目檔分支（約 761–769 行 content_after_num_games）；NORMALIZE_WHITELIST（125–126 行）；tools/t02_test_bundle.py normalize_bytes（429–442 行）與 quick_structure（316–318 行）；PREREG_02 第 9 節第 6 條；results/test02/design/checks.md 第 28 行「For i > n, both are empty」：T1.3 要求 num_games 之後的 slot 裡 sgf 和 color 都是空的。官方說明「color_1…color_20 一律是 B」如果代表空 slot 也填了 B（競賽二則可能填該題的執色），raw 的 T1 就是 FAIL_STRUCT。依登記要停下並寫修訂，修訂前不做任何預測，11/04 兩場都不能上傳。現有白名單轉換 clear_slots_after_num_games_if_empty_whitespace 只清除純空白，不處理 'B'。
  - 證據：sim_c1.py 用官方形狀的檔案實跑 t02_checks.t1_check（400 題、盤數 5/10/15/20、候選全黑 400×100，競賽二每題單色）：空 slot 的 color 留空 → player PASS、rank PASS；空 slot 填 B（或填該題執色）→ player FAIL_STRUCT、rank FAIL_STRUCT，每個題目檔 content_after_num_games = 3000，t02_test_bundle.quick_structure 也報「slot > num_games is not empty」。notebook 的快速檢查（build_test02_player_nb.py 579–594、build_test02_rank_nb.py 516–543）和所有讀檔程式（p02_player_fuse.read_questions、p02_rank_opening.read_questions、t02_checks.question_games）都只讀 1..num_games，不受影響，所以只有 T1 和打包這兩道關會擋。官方教學 README 只寫「只有前 num_games 個 sgf 欄位有值」，沒寫空 slot 的 color 怎麼填。
  - 建議：11/04 前登記 PREREG_02 rev 8，新增白名單轉換 clear_colour_after_num_games_if_sgf_empty：i > num_games 且 sgf_i 是空的（或純空白）、color_i 是 B/W/b/w 時，把 color_i 清成空字串，並記錄件數。另外只記錄：競賽一這類 color 不是 B 的件數；競賽二和同題已用 slot 執色不同的件數。sgf 在 num_games 之後有內容仍判 FAIL_STRUCT。程式要改三處：t02_checks 的 NORMALIZE_WHITELIST 和 T1.3、sub_norm['T1.3']，t02_test_bundle 的 NORMALIZE_WHITELIST 和 normalize_bytes（quick_structure 跑在正規化後的副本上，維持嚴格即可），以及 tests。raw 的 verdict 會變成 NORMALIZED，之後照原流程用正規化副本。checks.md 第 28 行和 TESTDAY 第 1 步的表格同步更新。
- **[major]** `tools/t02_checks.py` T3：t3_reference（1184–1204 行，參考集 R＝fit 玩家的全部 fit 棋，黑白混合）、t3_run 的 player 分支（1967–1980 行）、CONSEQUENCE[('T3','PASS')]（174–175 行）；PREREG_02 第 9 節第 8 條與第 4 節 T3：官方候選全黑時，候選的 100 盤 profile 只有 B 前綴的含執色 token，參考集卻是黑白混合，T3 幾乎失去偵測力。假陽性沒有升高，但判讀會悄悄變成「PASS：沒有重疊（約 20% 以下偵測不到）」，實際上大半候選是訓練玩家也偵測不到。
  - 證據：sim_c1.json（1,133 位黑棋 ≥100 盤的 fit 玩家，全黑 profile 取雜湊順序前 100 盤黑棋，對照登記的 q99）：虛無超門檻比例 raw 0.71%、canon 0.79%（登記的混色是 0.96%），不會誤判重疊（400 人達 ≥40 的機率約 1e-30）。但 TPR raw 0.125（登記 0.380）、canon 0.136（登記 0.430）。單一方向的判定機率：π=0.5 時 raw 0.0、canon 0.0（登記的混色是 0.995／1.0）；π=1.0 時 raw 0.32、canon 0.53。改用只含黑棋的參考集（sim_t3b.json：R＝每位 fit 玩家的黑棋、虛無和正例都用全黑 profile）：q99 raw 0.1105、canon 0.0827，TPR 0.372／0.414，π=0.3 時 0.22／0.45，π=0.5 時 0.995／0.999，和登記設計的偵測力相當。
  - 建議：兩種做法都要在 11/04 前寫成 PREREG_02 rev 8。(A) 建議：登記 T3b，候選池 b_c＝1 時使用。參考集是每位 fit 玩家的黑棋 fit 棋，虛無分布用 fit 玩家（黑棋 ≥100 盤）的全黑 100 盤 profile，門檻仍取 99 百分位、判定規則仍是 ≥40 人；用 `t02_checks.py precompute` 的擴充在 11/04 前算好並 commit 新的 pins。本 scratch 的數字只是參考，必須由 commit 的工具重算。(B) 至少要做：把 T3 在全黑池的 PASS 改寫成「無法判斷（偵測力不足）」，不寫「沒有重疊」，並在 RESULTS 揭露。兩種做法的後果都仍只是「考慮乙」，不擋上傳。
- **[minor]** `tools/t02_checks.py` t2_run 單元 (a)（1836–1866 行）；KEY_VERSION 與 results/test02/design/checks.md 第 13 行（game key 刻意不含執色字母）；PREREG_02 第 9 節第 7 條：官方題目和候選只給一方（競賽一全黑、競賽二每題單色）。測試棋如果以「對手那一方」出現在訓練集，也就是訓練列是白方、那位玩家其實是對手，T2(a) 一樣算一次複製，達門檻就 FAIL，觸發「開 split v2、另寫登記、不得利用這些複製」。這種複製不代表同一位玩家洩漏，是誤報。T2(c) 已經把同色、異色分開記，(a) 沒有。
  - 證據：game_keys 不含執色，checks.md 第 13 行明寫「the same game seen from the other side also matches」。TrainIndex 和 index cache 只存 key、pid_idx、pids、inner_game，沒有訓練列的執色，所以無法分辨。訓練集內同一盤從兩方各出現一次是 0（DATA.md），但測試集和訓練集之間的關係不知道。
  - 建議：只記錄的欄位：(a) 的每盤複製依「訓練列執色是否等於測試執色」分成 same_side 和 opposite_side（TrainIndex 要帶執色，index cache 加一個 colour 陣列或改成不用 cache）。rev 8 同時寫明判讀：只有 same_side 的複製才可能是同一位玩家；要把 FAIL 門檻改成只算 same_side，也要在同一個修訂裡寫。
- **[minor]** `docs/PREREG_03_RECIPE_V2.md` 2.2 節第 4 條（κ／Δκ 用 inner 評估輪與驗證題，混色題）、2.4 節閘門 G1–G3（inner 混色假題）、第 5 節測試日的 κ 檢查；tools/p03_rank_fuse.py kappa_questions（506 行）與 colour_decision（534 行）：prereg03 的兩臂有「目標玩家執黑」輸入層。官方競賽二每題單一執色，模型的執色偏差在題內不會互相抵消。κ 的參照（inner Δκ 的信賴區間、驗證題 Δκ>0）和閘門都是在混色題上算的，測試日卻要求 public／private 的單色題 Δκ>0。兩者的參照分布不同，可能誤判「color_i 語意和訓練不同」，也可能在單色題上悄悄退步。
  - 證據：用 R20（沒有執色層）在 inner 樣本內做同一個檢查（rank_colour.json，322 位雙色各 ≥20 盤的 inner 玩家、20 輪、k∈{5,10,15,20}）：混色 0.6040、隨機單色 0.6050、全黑 0.5902、全白 0.6136。CNN 預測的期望等級減真實等級：黑 −0.051、白 −0.201。R20 平均沒有退步，但執色不對稱確實存在；有執色層的模型可能更明顯。prereg03 目前還沒有任何 Colab 輸出。
  - 建議：趁還沒有任何 prereg03 輸出，現在就寫 PREREG_03 rev 3：(1) 在 inner 另外做單色假題（每位玩家各色分開抽，k∈{5,10,15,20}），只記錄選定組合和 P00 在單色題上的分數，以及 Δκ 在單色題上的 inner 信賴區間；(2) 測試日的 κ 門檻改用單色題的 inner 參照，驗證題也用單色子集算。不改選法和閘門。C3（11/03）的修訂照這個寫。
- **[minor]** `tools/t02_checks.py` t4_run（2034–2073 行）的 black_share_inside_null_range、first_move_point_tvd_null_max、upper_right_null_range；train_refs.json 的 t4_null_ranges；tests/test_t02_dryrun.py 495–503 行：T4 只記錄的虛無範圍是用混色池算的。官方競賽一全黑時，每份檔的 black_share＝1.0，一定落在範圍外；第一手 TVD 也會系統性超過混色虛無的最大值。判定本身（|b_c−b_q|>0.5）不受影響，但報告會出現大量看起來異常的欄位，容易被誤讀。
  - 證據：sim_c1.json，40 個全黑池（fit 玩家、黑棋 ≥120 盤）：候選的第一手 TVD 平均 0.0575、最大 0.0714，登記的混色虛無最大值 0.0477；題目的 TVD 平均 0.0622、最大 0.0761，虛無最大值 0.0730。右上角比例 0.8086–0.8533，虛無範圍 [0.791, 0.839]。競賽一的 b_c＝b_q＝1.0，T4 verdict 是 PASS。test_t02_dryrun 斷言 black_share_inside_null_range 為 True、TVD ≤ 虛無最大值，只在混色假資料上成立。
  - 建議：不影響判定，不需要改判定規則。rev 8 寫明：官方全黑時這些範圍不適用。選做：precompute 另外算只記錄的全黑虛無範圍，或讓 t4_run 在 b＝1.0 時標「不適用」。dry-run 的斷言跟著官方形狀的假資料更新。
- **[minor]** `tools/t02_fake_tests.py` from-train（docstring 36–48 行：候選和題目取自混色的 fit 棋，題目 5–20 盤均勻抽）；from-val；tests/test_t02_dryrun.py、test_t02_checks.py 的 T4／T5 案例：所有演練和測試用的假資料都是混色、每題 5–20 盤均勻抽，沒有一個重現官方結構：競賽一全黑、盤數只有 5/10/15/20、競賽二每題單色、空 slot 可能填了執色。上面的 FAIL_STRUCT 和 T3 偵測力問題，現有測試都抓不到。
  - 證據：test_t02_nb_player.py 291 行的假題目執色 'BW' 交替；test_t02_player.py 98 行 rng.randint(5, 20)；test_t02_dryrun.py 的 T4 斷言預設混色。
  - 建議：from-train 加一個官方形狀的選項：候選 400×100 全黑，題目 5/10/15/20 各 100 題且全黑，競賽二每題單色；空 slot 的執色做「留空」和「填入」兩種版本。在 C10_SLOW=2 的 dry run 各跑一次，預期結果是：留空 PASS；填入 NORMALIZED（第 1 條修正後），修正前是 FAIL_STRUCT。這部分只改測試，不影響判定，實作說明記進 rev 8。
- **[info]** `docs/PREREG_02_OPENING_FUSION.md` 2.2 節切法防護（執色比例條件、X 情境、「兩套一起用」的理由）、第 4 節 T4、第 9 節第 9 條；docs/TESTDAY.md 第 3 步（「T4 的 |b_c−b_q|>0.5 由 t02_player.py 自動套第 3 層」）；tools/p02_player_open.guard_level、t02_player.apply（608–610 行）：官方全黑時 b_c＝b_q＝1，執色條件永遠不會觸發，T4 的 GUARD_LEVEL3 也不可能出現。θ₂ 是用官方結構下不可能發生的「執色分開」情境 X 校準的，但它仍然是 ρ 下降時的第 3 層門檻，程式行為不變。
  - 證據：sim_c1.json，全黑、黑棋內隨機切、題目 k∈{5,10,15,20}：ρ raw 1.005／1.023／1.036、canon 1.004／1.016／1.021，都高於 θ₁（raw 0.963、canon 0.971），落在第 1 層。同一段程式重現混色校準池，raw 得到 1.0150／1.0301／1.0141，和 guard_calibration.json 的 random 池一致。只用開局的真題得分：全黑 raw 0.932／0.926／0.919，混色 0.880／0.898／0.925，全黑沒有變差。
  - 建議：不用改程式。rev 8 補一段說明：官方結構下執色條件和 X 情境不適用；另外，如果官方是依時間切題目和候選，ρ 下降會照登記由 θ₁／θ₂ 處理。TESTDAY 第 3 步同步改字。
- **[info]** `tools/p02_player_open.py` col_tokens／nocol_tokens／game_tokens；PREREG_02 2.2 節「兩套一起用」：原本的假設「全黑時含執色和不分執色兩套 token 完全重複」不成立。含執色那套的整盤相鄰兩手 token 帶了落子方的顏色（Bq:Bab>Wcd 和 Bq:Wab>Bcd 是不同的 token），所以不是單純的複本；自己落子的 token 則等於在不分執色那套上多算一倍權重。這不是錯誤，只是全黑時兩套的比重和混色時不同。
  - 證據：sim_c1.json，一個全黑池：兩套合用的詞彙 47,122，只用不分執色的 19,620；cosO 最大差 raw 0.046、canon 0.045。只用開局的表現見上一條（全黑沒有變差）。
  - 建議：不用改。rev 8 更正說明文字，不要寫成「重複」。
- **[info]** `tools/p02_player_open.py` sr_rounds（k = rng.randint(5, 20)）；PREREG_02 2.2 節與第 9 節第 4 條（每池自我檢索）：自我檢索抽的假題盤數是 5–20 均勻，官方是 {5,10,15,20} 各 100 題，兩者平均都是 12.5。這只影響選組合時的權重分布，不影響 ρ（見上條）。改成官方盤數會破壞本機接縫（驗證池必須照 randint 選出 Q04），不值得。
  - 證據：彩排預期 public 選 Q04（0.902147，Q08 是 0.898223），兩者差距小於 0.003 的容許範圍以外不多；全黑、官方盤數下 ρ 仍在 1.004–1.036。
  - 建議：維持不變。選做：只記錄的診斷，另外用 k∈{5,10,15,20} 算各組合的自我檢索平均，不用來選。
- **[info]** `tools/t02_checks.py` t5_run（2079–2098 行）的 chi2_vs_uniform；train_refs.json 的 t5 參照（uniform_5_20、本機 12.06／36.4%）：官方的盤數是設計好的固定分布（4 種各 100 題），拿來和 5–20 均勻分布做卡方檢定沒有意義，會算出 p≈0 的「差異」。T5 只記錄，不會崩潰，也不影響判定。
  - 證據：官方直方圖代入：觀察值是 4 格各 100、12 格 0，期望值每格 25，chi2＝4×75²/25＋12×25＝1200，df 15，p≈0。T3q 的 by_k 有 5–20 每個 k，所以 k∈{5,10,15,20} 都找得到門檻。
  - 建議：不需要修訂。選做：t5_run 加一個只記錄的欄位 official_design_match（每份檔 5/10/15/20 是否各 100 題），報告說明不看 chi2。
- **[info]** `tools/p02_rank_fuse.py` question_logits 的 (n/12)^γ；N_BINS＝((5,7),(8,12),(13,20))；R20 frozen_config：官方 n∈{5,10,15,20} 都在擬合時的盤數範圍 5–20 內（γ＝0.4387），溫度項不需要外推。盤數分箱的記錄會變成 5、10、15＋20 三組，只影響記錄。單色題見 PREREG_03 那條：R20 平均沒有退步。
  - 證據：rank_colour.json（樣本內）：隨機單色 0.6050、混色 0.6040；拿掉開局項（w=0）時是 0.5960 和 0.5875。
  - 建議：不用改。選做：之後的登記可以考慮依執色加偏移項（黑、白的 CNN 偏差差 0.15 級）。這是新方法，要另外登記，不算修正。
- **[info]** `colab/build_test02_player_nb.py` quick_questions／quick_candidates（579–625 行）；colab/build_test02_rank_nb.py 516–543 行：notebook 的快速格式檢查只看 1..num_games 的執色，以及「有棋的 slot 剛好是 1..n」，不管空 slot 有沒有執色，也不假設黑白混合，所以官方結構不會讓它停下。notebook 讀的是打包後的正規化副本，第 1 條修正後照常運作。
  - 證據：filled 只看 sgf_i.strip()；執色檢查的範圍是 range(1, n+1)；候選只檢查 color ∈ {B,W}。
  - 建議：不用改，notebook 和產生器都不用重建。
- **[minor]** `docs/DATA.md` 新增的「官方資料說明（AIdea…）」一節（工作樹裡還沒 commit 的修改）：新段落說影響評估放在 `results/prereg02/checks/official_structure/`。依 TESTDAY「凍結檔一律不改：… results/prereg02/**」，而且 t02 工具把 results/prereg02 列為唯讀，放進去會和凍結規則衝突，也容易讓 hygiene 的判讀變複雜。另外還有幾件事該補進 DATA.md。
  - 證據：results/prereg02/checks/ 目前只有 reference_reread_sumcheck.*，是 9694bba 打分時留下的；TESTDAY.md 第 12 行列出凍結範圍。
  - 建議：位置改成 results/test02/design/official_structure/（或 results/official_structure/）。DATA.md 建議補記：(1) color_k 在 k>num_games 時是空的還是填了執色，目前未知，11/04 看 raw 檔確認，這一點決定 T1 的結果；(2) 每份題目檔是 5/10/15/20 各 100 題，T5 的卡方不適用；(3) 競賽一全黑，所以「自己的前 10 手」＝整盤第 1、3、…、19 手，執色防護不會觸發；(4) 競賽二每題單色，黑、白題的比例 11/04 記錄，R20 在 inner 的黑題 0.590、白題 0.614（樣本內、scratch 數字，沒 commit）；(5) 本機驗證集（混色、5–20 均勻抽）和官方結構不同，本機分數 0.5891、0.8830 是在不同結構上量的；(6) fit 玩家黑棋 ≥100 盤 1,133 人、≥120 盤 926 人、≥200 盤 424 人，A 級黑棋 ≥120 盤只有 57 人。官方候選每人要有 ≥100 盤黑棋加題目棋，所以如果候選是訓練玩家，只可能是下棋很多的那群；(7) T2 的 game key 不分執色，同一盤從對手那一方出現也算複製；(8) T3 在全黑池的偵測力（見 T3 那條）。