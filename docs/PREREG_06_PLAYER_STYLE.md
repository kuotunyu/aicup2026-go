# 事前登記 06：競賽一加上下棋風格特徵（v2）、CSLS 修正，和有條件的一對一開關

**白話版**：Q04 是「CNN＋開局 token」。這次在它上面再加三項「下棋風格」特徵，並一律做一個叫 CSLS 的小修正（預設 k＝10，凍結前重驗一次）。CSLS 的作用是：如果某位候選對很多題都很像，就把他的分數扣一點。另外加一個開關：只有資料本身看起來是「一題對一位候選」，而且 public 分數沒有反對時，最後一版才改用一對一配對。這個開關會用到 public 分數，和既有的上傳規則不同，所以在第 5.5 節正式登記成例外。本機驗證集只打一次分，這一次就是判定。

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。前幾份：
  - `docs/PREREG_02_OPENING_FUSION.md`：競賽一 Q04 採用，本機驗證集 0.8830（`results/score_ledger.csv`、`results/prereg02/RESULTS.md`）。測試日的每池程序、三層防護、T1–T6 在它的第 4、9–12 節。
  - PREREG_03、04 只做競賽二，05 在 10/2 收掉。競賽一的打分機會在 02 之後沒有再用：已用 1 次，剩 3 次。
- 探索：`results/prereg06/explore/`。競賽一的主體是 `c1_headroom/REPORT.md`，審查在 `critic/REPORT.md`，怎麼接進測試日在 `integration/REPORT.md`。都只用 fit／inner 訓練玩家、只用 CPU、沒有打分。
- 設計：`results/prereg06/design/`。
  - 實作規格：`spec.md`。
  - 兩支輔助程式：`c1d_veto_check.py` 用探索存下的模擬結果（`s8_rows.json`）檢查否決規則，輸出 `c1d_veto_check.json`；`c1d_cnn_bench.py` 量 4090 上算 CNN 的速度，輸出 `c1d_cnn_bench.json`。
  - 第 1 輪審查：`review_r1.md`（補跑的兩個檢查在 `review_r1/`），回應在 `review_r1_response.md`。本版已照審查改過。
- **只做競賽一（TEAM_11249）。** 競賽二的新方法是 PREREG_07（KataGo HumanSL），另外寫、預算分開。06 的輸出都不進競賽二，反之亦然。
- **交件模型選甲**（VALIDATION「交件模型」、PREREG_02 第 3 節）：直接用本機驗證過的做法，每個官方候選池各自計算，不重訓任何模型。
- 本登記寫四件事：比什麼、怎麼選、怎樣算成功、什麼時候放棄。實作細節以 `spec.md` 為準。實作要偏離本登記時，在第 9 節第 7 步 `freeze` 之前寫進修訂紀錄（rev 2）。
- 代號：
  - Q04：PREREG_02 選中的組合，z(E0)＋1·z(O_raw)。
  - v2：本登記新加的三項風格特徵（第 1 節）。
  - V0–V3：本登記的組合（第 2.2 節）。
  - CSLSk：k＝`k*` 的 CSLS 修正。`k*` 在第 2.6 節重驗，預設 10；重驗後寫進 `frozen_config.json`，之後不改。
  - R02：PREREG_02 在某一池實際交出的組合。本機驗證池是 Q04；官方每一池是 test02 用自我檢索選出、再經閘門和防護後的組合（`results/test02/player/<池>/apply_report.json`）。
  - 06-OFF、06-ON：測試日的兩份交件檔（第 5.3 節）。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `802eaea` | 第一版，2026-10-0X，已納入第 1 輪審查（`results/prereg06/design/review_r1.md`）。當時**沒有任何 06 的程式、正式輸出或分數**，`results/prereg06/` 只有 `explore/` 和 `design/`。本機驗證集自 H2P（9/26）之後沒有在競賽一打過分；v2 特徵沒有在任何驗證玩家的棋上算過。探索和審查看過的數字都寫在第 0 節 |
| rev 2 | `9d8245d` | 2026-10-03，在 `freeze` 之前；當時沒有任何驗證玩家的 v2 特徵、沒有 06 的分數。實作和 rev 1 的差異：① 2.6 選門檻：可行的門檻裡，最大的「11 種結構平均增益」≤ 0 時，那種 F 來源沒有門檻；② 2.6 `r*_X`：兩種 F 來源都有門檻時才取較小者，任一種沒有，組合 X 的開關判定「不能用」；「能用」的四條在每種 F 來源上都要成立；`k*` 先定，選門檻和判定時的「關」都是 CSLS`k*`；③ 2.3 sr06：盤數清單 K 先由小到大排序再用鹽洗牌（rev 1 沒寫洗牌前的順序）；單色檢查的假題是那一份裡依雜湊順序的前 k 盤黑棋；④ 2.8 leakcheck：不經 `load_rows(local_dir=data/local)`（它還會讀 `rank_val_questions.csv`），改由 `p06_player.py leakcheck` 只讀 `player_val_candidates.csv`、`player_val_questions.csv` 兩份棋譜檔，用同樣的 SHA-1 算法自己比對；fit 的人數和盤數仍由 `load_rows(part="fit", local_dir=None)` 核對；⑤ 程序加嚴（實作前獨立審查的 5 個 major）：`freeze` 要求 leakcheck 通過且重疊 0（只要求它當時的工具已 commit，不和現行工具比 SHA）、build／lda／selfcheck 通過、其他設計報告都已 commit 而且由現行工具產生、報告之間記下的 SHA 互相對得上、`lda_{B,W}.npz` 和特徵快取的 SHA 等於報告所記；`apply` 要求 `lda_{B,W}.npz` 的 SHA 等於 `frozen_config.json` 而且已 commit，正式模式不接受改輸入路徑，prereg02 的 Q04 檔 SHA 要等於凍結值，Q04 用 p02 的函式和 06 的路徑各重現一次、都要逐位元組相同，D-mix 的判定讀 `frozen_config.json`。其他照 rev 1 |
| rev 3 | `86c54e9` | 2026-10-03，K5'（11/03 23:59）之前、拿到任何測試資料之前：測試日程序（第 5 節「細節在 K5' 前另寫 rev 3」），全文在第 11 節，已納入 10/3 三個角度的審查（正確性、程序、操作者；`results/test06/design/build_review.md`）。要點：① 工具 `tools/t06_player.py`（`feats`／`seam`／`apply`／`submit`／`veto`／`status`，結束碼 0 完成、2 拒絕、3 登記的「06 不用」、4 本機漂移；前提和登記的 3 先於「輸出已存在」核對；凍結檔被改是 2）、`tools/t06_fake_tests.py`（演練用）、`tools/official_log.py` 只加行、`tests/conftest.py`（單一行程跑全套測試）；步驟 `docs/TESTDAY_06.md`；② 第 2.7 節的表不改，補充寫在第 11.2 節第 7–10 條：test02 那一池選中只用 CNN 的組合（防護第 0 層）時，和第 3 層一樣交 test02 的列；兩池都退回 test02 的列時 06 不用；③ seam 只讀 test02 判過的 dumps，判定和失敗時的結束碼寫死（dumps 和 prereg02 逐位元組相同時沒過＝本機漂移；不同時照 398/400，沒過＝06 不用）；④ 各池的 R02＝test02 那一池 `apply_report.json` 的 `applied`，06 閘門 sr(選中) − sr(R02) ≥ 0 算通過；⑤ 06-ON 的七條條件、AIdea 頁面的核對方式（沒把握就當成不是一題對一人）、06-OFF 不綁 T2；⑥ 同名檔警告（「完全相同」＝差 < 1e-9）：06-ON 之前要先排除 Q04 和 06-OFF 分數相同，傳錯的 06-OFF 在 06-ON 之前重傳；06-ON 之後核對出 06-OFF／06-ON 這一對傳錯、或核對不出，不用開關、最終 06-OFF（比第 5.3 節嚴，只會把開關關掉，因為 06-ON 不能再傳）；暫停時什麼都不指定；⑦ 否決只讀已 commit 的 `official_scores.json`，06-ON 配同一次執行比它早的最新一筆 06-OFF，`veto.json` 要在 K7' 前 commit；d ≥ 0 而 06-ON 已不是最後一次上傳時最終 06-OFF；⑧ `official_log.py` 依 5.4 指定最終選擇，推翻只能引用 rev ≥ 4；log 裡沒有 06 紀錄時輸出和 tag `p04-testday-pre06` 逐位元組相同；⑨ test02 在 06 之後重跑，06 從 `feats` 重做（照 PREREG_04 第 11.3 節第 20 條的前例）；K6' 之後不再執行 06 的步驟，K6' 前已 commit 的照常有效；⑩ 防誤改閘門加上 06 的測試和 TagVersusNew，已知例外是 H6P 的 4 列（PREREG_04 rev 3 第 19 條），其他登記依規產生的失敗照 `docs/TESTDAY_04.md` 的寫法逐項核對。當時沒有任何測試資料，test02 也還沒有正式執行；這些規則只在分支 `p06-testday` 的假測試樹上演練過（`results/test06/design/build_review.md`） |

## 0. 為什麼做這個（都沒看答案；但看過探索的評估池才定，所以偏樂觀）

- **Q04 輸在哪**：見 `c1_headroom/REPORT.md` (a)。用的是官方形狀的模擬池：400 位 fit 玩家各 100 盤黑棋，題目 5／10／15／20 盤各 100 題。
  - 只用開局那一半是 0.9253。
  - 失分的 63.7% 在 5 盤題，56.5% 來自「自己開局前後不一致」的那 1/5 玩家。
- **新訊號**：見同一份報告 (c)、`s5b_combine_v2.json`。權重在調參池 seed 100–109 選，數字取自評估池 seed 0–19。
  - 各項單獨加在開局上：
    - lda64（用池外 fit 玩家學的「分辨玩家」投影）+0.0452 [0.0405, 0.0499]
    - joseki_own +0.0348
    - style2 +0.0243
  - v2＝開局 1＋lda64 2.66＋joseki_own 0.5＋style2 0.5（`fused_v2_nopost.json`）。不修正 0.9819，比只用開局 +0.0566；加 CSLS10 是 0.9869，+0.0616。
  - 疊在 Q04 上的估計：本機算不出 fit 玩家的 CNN，所以用替身代替。
    - 替身 A（style NB）：v2 照原權重加上去再做 CSLS10，+0.0300 [0.026, 0.034]。
    - 替身 B（lda32，偏悲觀）：+0.0234 [0.0204, 0.0265]。
    - 最壞的情況是 CNN 已經含有全部新資訊，那時只剩 CSLS10 的 +0.0128。
    - 合理範圍 +0.01～+0.04（估計）。
- **CSLS10**（hubness 修正）在 7 種題目結構裡都不為負（`s8_switch.json` 評估池）。
  - v2：一題對一人 +0.0050；有放回抽 +0.0000 [−0.0013, +0.0014]。
  - 替身 A：+0.0128 到 +0.0053。
  - 審查補跑（`review_r1.md` 3.2）：有放回抽時，增益隨特徵變強而變小（base +0.0139、替身 A +0.0053、v2 +0.00002）；替身 A＋v2 是 +0.0008 [−0.0008, +0.0024]。所以 k 不寫死，凍結前在最終的 F 上重驗（第 2.6 節）。
  - CSLS3 在一題對一人時比較好，但有放回抽時是 −0.0070，所以只在一對一開關打開時用。
- **一對一（Hungarian）**：
  - 一題對一人時，比 CSLS10 再多約 +0.011：v2 評估池是 0.9980 對 0.9869。
  - 不是一題對一人時，每池虧 −0.009 到 −0.23。數字來自 `c1d_veto_check.py` 的 v2：u390 平均 −0.0090，wr 平均 −0.2276。
  - 自我校準偵測器 r_forced_cost ≤ 1.58233：在一題對一人的評估池打開 19/20；非一題對一人的 120 池誤開 0 次。
  - 審查補跑「只有 1–12 位候選重複」（`review_r1.md` 3.1，v2）：d1 打開 11/20、d2 打開 3/20，d3 以上 0/20；有打開的池最差 −0.0016。偵測器停止打開的位置，比開關由賺轉虧的位置早。
- **CNN 是樣本內的**：prereg01 的玩家 CNN 用全部 6,407 位訓練玩家訓練（fit＋inner，PREREG_01 第 4 節 ⑤ `part="train"`）。所以拿 fit 玩家做的模擬裡，CNN 都比它碰到新玩家時強。因此：
  - 在官方池上用每池自我檢索選組合（第 2.4 節），那一池的 CNN 是真實強度。
  - 用真實 CNN 調出的 V1 排在 V2 之後，並有額外條件（第 2.5 節）。
  - CNN 在模擬裡強得不合理時，整個改用替身（第 2.5 節的 E0 上限）。
- **挑選的自由度**（第 8 節揭露）：
  - 探索看過 14 種以上的特徵族、4 種 LDA 維度、5 種 CSLS k、dual softmax 和 Sinkhorn 的參數、8 種偵測統計量。
  - v2 是在調參池上用貪婪前向選擇挑出來的。
  - 評估池 seed 0–19 在探索中被看過很多次，所以探索的數字偏樂觀。
  - 本登記新的調參用沒用過的 seed（第 2.5、2.6 節），但候選仍是探索用過的同一批 fit 玩家（官方形狀池是同一群 926 人，D-mix 是 1,937 人）。**新 seed 不等於新資料。真正的判定只有第 3 節那一次驗證集打分。**

## 1. 特徵（定義寫死）

### 1.1 沿用（不重算、不改）

- E0、O_raw、它們的 z 分數、缺值規則和同分規則，都照 `tools/p02_player_fuse.py`、`tools/p02_player_open.py` 的函式（import，不改檔）。Q04＝z(E0)＋1·z(O_raw)。
- CNN 的逐盤 embedding：
  - 本機驗證池：`results/prereg02/player/dumps/{cand_games,query_games}.npz`（SHA `56ed03d1…`、`665d0888…`，已 commit）。
  - 官方池：test02 notebook 寫出、已 commit 的 `results/test02/player/<池>/dumps/`。
  - 不另外跑 Colab。

### 1.2 目標方與手的編號

- 每盤棋有一位目標玩家，看候選檔的 `color` 或題目檔的 `color_i`。
- 落子：用 `p02_moves.MOVE_RE` 依序取出整盤的手，不設 60 手上限。MOVE_RE 只取盤上的落子，pass 不會被取到。手的編號 i 從 0 起。
- **交替檢查**：取到的序列必須從黑開始嚴格黑白交替（第 i 手的顏色字母，i 偶數是 B、奇數是 W）。不交替的盤（例如中間有 pass，或連下兩手同色）**不進任何 v2 項**，盤數依池、依顏色記錄。官方 T1.4 容許最多 1% 這種棋。
- 交替的盤上：目標方的手＝顏色字母是目標方的那些手，等於黑是 i＝0、2、4…，白是 i＝1、3、5…。「對手的上一手」是第 i−1 手，「自己的上一手」是第 i−2 手。
- 一手都取不到的盤，也不進任何 v2 項，盤數記錄。
- 候選或題目在某一項沒有任何可用盤時，那一項依 1.8、2.1 節的缺值規則處理。
- 官方的棋全部執黑，所以在官方池的交替盤上，1.3–1.6 節的定義和探索的黑棋版本（`c1_headroom/c1h_feats.py`）完全相同。探索沒有排除不交替的盤，所以 selfcheck 會有極少數差異（第 2.8 節）。白棋只出現在本機驗證集。

### 1.3 JO：自己的角落定式 token（joseki_own）

- 取整盤前 80 手，兩方都算。x 或 y 等於 9（第 10 路）的手不屬於任何角，其餘依 x>9、y>9 分成 4 個角。
- 每個角的手照順序排：
  - 座標轉到左上角：右邊的 x→18−x，下面的 y→18−y。
  - 這個角裡第一手不在對角線上（x≠y）的落點如果 x>y，整個角的 x、y 對調。
  - 每一手記成 o（目標方下的）或 p（對手下的），再加兩個字母。
- token：每個角只留 o 的手、保持順序，取前 n 手（n＝1…min(4, o 的手數)），寫成 `{C}K{n}:{前 n 手串起來}`。C 是目標方的顏色。
- 向量和相似度同 PREREG_02 的 cosO（`p02_player_open.open_cos`）：
  - token 次數加總，權重 1＋ln(次數)；
  - 乘上 IDF＝ln((1＋P)/(1＋df))＋1，P 是候選人數，df 只數候選向量；
  - L2 正規化後算 cosine。
- 全黑時 C 一律是 B，結果和探索的 `f_joseki_own` 相同。

### 1.4 S2：每盤 66 個風格比例（style2）

- 第 1 個值：整盤手數 ÷ 100。
- 依手的編號 i 分 5 段：[0,20)、[20,50)、[50,100)、[100,150)、150 以後。每段只看目標方的手；設 m＝max(這段目標方的手數, 1)，各記 13 個比例（都除以 m）：
  - 線位 5 個：ln＝min(x, y, 18−x, 18−y)，依 min(ln, 4) 分 5 格。
  - 和對手上一手的切比雪夫距離 4 個：≤1、2、3–4、≥5（i≥1 才算）。
  - 和自己上一手的切比雪夫距離 ≤2 的比例 1 個（i≥2 才算）。
  - 區域 3 個：角（min(x,18−x)<6 而且 min(y,18−y)<6）、中央（ln≥6）、邊（其他）。
- 全黑時和探索的 `style2_vec` 相同。

### 1.5 每盤的 LDA 輸入

- 應手 token（resp），對目標方的每一手 i：
  - 階段 ph：i<60 是 0，i<150 是 1，其餘是 2。
  - rel(a, b)：兩點的 dx、dy 取絕對值，d＝max。d≤2 寫成「小的數字接大的數字」（例如 `12`），d 是 3 或 4 寫 `m`，d≥5 寫 `f`。
  - token：
    - `L{ph}:{min(ln+1, 5)}`；
    - i≥1 時 `R{ph}:{rel(第 i 手, 第 i−1 手)}`；
    - i≥2 時 `S{ph}:{rel(第 i 手, 第 i−2 手)}`；
    - i≥1 時 `RS{ph}:{前一個 rel}` 接直線符號再接 `{後一個 rel，i<2 時寫 -}`。
- 詞彙：每個顏色一份，是那個顏色的 LDA 訓練棋（1.6 節）出現過的 token，排序後凍結。探索的黑棋詞彙是 220 個。詞彙外的 token 丟掉，件數記錄。
- 每盤的 LDA 輸入＝S2 的 66 個值，接上每個詞彙 token 的 sqrt(次數 ÷ 這盤詞彙內 token 總數)；總數是 0 時全填 0。黑棋是 66＋220＝286 維。

### 1.6 LDA64：學一個「分辨玩家」的投影

- **訓練資料**：只用 fit 玩家，透過 `tools/c10data.py` 的 `load_rows(part="fit")` 讀（人數和盤數核對 `EXPECTED["fit"]` 照常做；和本機驗證檔的雜湊比對在第 9 節第 3 步的 `leakcheck` 單獨做一次，見 `spec.md` 第 0 節）。兩個顏色分開：
  - 黑：黑棋 ≥105 盤的 fit 玩家的全部黑棋，1,082 人、232,165 盤。
  - 白：白棋 ≥105 盤的 fit 玩家的全部白棋，1,077 人、229,761 盤。
  - 這些人數和盤數是 10/2 設計時只讀 fit 資料數出來的（`spec.md` 第 3 節），審查用探索快取重算相同。
  - 「≥105 盤」和訓練棋都依交替檢查前的盤數算（和探索相同）；不交替的盤不進 LDA 訓練，件數記錄。
  - 不用 inner，也不用任何驗證玩家：驗證玩家本來就不在 fit 裡，`load_rows` 也會核對。
- **為什麼這樣定**：
  - fit、≥105 盤、只看黑棋，就是探索驗證過的設定（`c1h_lib.build_cache`）。官方全黑，所以官方池只會用到黑棋的投影，和探索一樣。
  - 白棋的投影只為了黑白混合的本機驗證集。它不另外調參，定義和黑棋完全對稱。
  - 不加 inner：人數只多約一成，增益不明，又多一個沒驗證過的選擇。
  - ≥105 盤：每位玩家的平均要夠穩，LDA 才學得到「人和人之間」的差別；探索也只驗證過這個門檻。
- **做法**（照探索的 `c1h_lda.fit`）：
  - 每一維用訓練資料的平均和標準差標準化；標準差 <1e-9 的當成 1。
  - Sw＝玩家內的共變異；Sb＝玩家平均之間的共變異，以盤數加權。兩者都除以總盤數。
  - 在 Sw 上加 1e-3·trace(Sw)/D 倍的單位矩陣，解 Sb w＝λ Sw w（`scipy.linalg.eigh`），依 λ 由大到小取前 64 個方向。
- **用法**：
  - 每盤投影成 64 維。
  - 一位候選（或一題）在某個顏色的輪廓，是它那個顏色的棋的投影平均。
  - T_L,色(q, c)＝−（題目輪廓和候選輪廓的平方歐氏距離）。
- **凍結**：用上面的訓練資料各學一次，存成 `results/prereg06/player/lda_{B,W}.npz`（64 個方向、λ、平均、標準差、詞彙），SHA-256 寫進 `frozen_config.json`。驗證池和官方池都用這兩份，不重學。
- **模擬裡**（第 2.5、2.6、2.8 節）：照探索，每一池重學一次，排除池裡那 400 人，讓投影對池裡的人是樣本外的。

### 1.7 S2 的項：每色各做一次 Gaussian naive Bayes（照探索的 `style_ll`）

- 對每個顏色：
  - 候選的平均 μ_c＝它那個顏色的棋的 S2 平均。
  - 池內的玩家內變異 s2w＝每位候選逐盤變異數（母體）的平均，再加 1e-6；只算那個顏色至少有 2 盤的候選。
- 題目在那個顏色有 k 盤時：x̄＝那 k 盤的平均，var＝s2w/k＋s2w/n_c（n_c 是候選那個顏色的盤數）。
- T_S2,色(q, c)＝−½·Σ_維 (x̄−μ_c)²/var。

### 1.8 兩個顏色怎麼合（只用在 T_L 和 T_S2）

- 對每一題、每個它有棋的顏色：先把那個顏色的那一列對全部候選做 z 分數（母體標準差；<1e-12 時整列當 0）。候選在那個顏色沒有任何棋時，先填那一列的最小值，再做 z 分數。
- 合併：T(q, ·)＝Σ_色 (題目那個顏色的可用盤數 ÷ 題目的可用盤數)·z(T_色(q, ·))。題目沒有任何可用盤時，這一項缺值、記 0。
- 全黑時 T＝z(T_B)。融合時會再做一次 z 分數，不會改變它，所以結果和探索相同。
- JO 不用這一步：它的 token 本身帶顏色，不同顏色的 token 不會互相比到。

## 2. 合併、選擇、CSLS 和一對一開關

### 2.1 合併公式

- F_V(q, c)＝z(E0)＋1·z(O_raw)＋w_L·z(T_L)＋w_J·z(JO)＋w_S·z(T_S2)。
  - z 是同一題對全部候選的 z 分數。
  - 缺值或標準差 <1e-12 的項記 0（同 p02）。
- CSLSk：C＝2F−r_c−r_q（探索的 `c1h_eval.csls`）。
  - r_c：這位候選在全部題目裡最高的 k 個 F 的平均。
  - r_q：這一題在全部候選裡最高的 k 個 F 的平均。
  - k＝`k*`（第 2.6 節重驗，預設 10）。`k*` 是「不修正」時，C＝F。
- 依 C 由大到小取 Top-5。同分時照候選代號；候選一律依代號排序。

### 2.2 組合清單（順序固定，也是平手時的優先順序）

| 組合 | 內容 | 什麼時候在清單裡 |
|---|---|---|
| V2 | Q04＋v2，w_L 2.66、w_J 0.5、w_S 0.5（探索的 `fused_v2_nopost.json`）＋CSLSk | 一定 |
| V1 | Q04＋v2，權重 w_tuned（第 2.5 節）＋CSLSk | 只有第 2.5 節的條件全部成立時 |
| V3 | Q04＋v2，權重是 V2 的一半（1.33、0.25、0.25）＋CSLSk | 一定 |
| V0 | Q04＋CSLSk（不加 v2） | 一定 |

- V2 排第一：它的權重來自探索，和 fit 玩家的 CNN 無關。V1 用樣本內的 CNN 調出來，偏差有系統性（第 0 節），所以排在 V2 後面，平手時讓給 V2。
- V3 是保險：CNN 和 v2 的資訊可能重疊，真正最好的 v2 權重可能比探索的小。
- v2 的權重都是相對於 O_raw 的 1 來說的。E0 和 O_raw 的 1:1 不動，也就是 Q04。

### 2.3 06 的自我檢索 sr06（只用候選檔和候選的逐盤 embedding，不讀任何答案）

- 每位候選的 100 盤依 sha256('c10-prereg06|sr|{player_id}|{game_id}') 排序，依位置切成 5 份，每份 20 盤。
- 盤數清單 K：這一池全部題目的 `num_games`，超過 20 的當 20。官方池是 5／10／15／20 各 100 個；本機驗證池是它自己的 5–20。
- 第 r 輪（r＝0–9）：
  - 用第 r mod 5 份。
  - K 用 random.Random(int(sha256('c10-prereg06|sr-k|{r}') 的前 16 個十六進位字, 16)) 洗牌。候選依代號排序後，第 i 位拿第 i 個 k；K 和候選人數不同時，拿第 i mod |K| 個。
  - 假題＝那一份裡（依上面的雜湊順序）的前 k 盤；參考＝其他 4 份，共 80 盤。
- 每一輪的 IDF、CNN 重心、S2 的 s2w 和所有輪廓都只用參考棋算。LDA 用凍結的投影，不重學。
- 每一輪每位候選剛好一題，所以假題池一定是一題對一人。照官方公式計分；sr(X)＝組合 X 在 10 輪 × 全部候選上的平均分。
- 第 0、1 輪同時是一對一開關的假題池（第 2.6 節）。
- R02 的 sr 用同一批假題算，它的 E0／E1、raw／canon 都用 p02 的函式算。

### 2.4 每一池怎麼選（本機驗證池和官方的兩池都在那一池上各自做）

- 算清單裡每個組合的 sr，取最高的。和最高分差 0.003 以內的，取清單中最前面的。所以 V0 要比每個有 v2 的組合都高出 0.003 以上，才會被選中；V1 要比 V2 高出 0.003 以上，才會被選中。
- 選中的寫進那一池的報告。本機驗證池在 `apply`（第 3 節）選，官方池在測試日選（第 5 節）。

### 2.5 V1 的權重（要 fit 玩家的真實 CNN embedding，用本機 4090 算）

- **前提**：使用者把 prereg01 的玩家權重從 AI Pro 帳號的 Drive 下載到 `data/prereg01_weights/player/best_model.pth`，SHA-256 必須是 `356eb4f65488925e2a552ea382979780c903ef515b85d329a3805db837b7773b`。這就是 CLAUDE.md 裡本來就排在 10 月的「權重備份」。
- **算 embedding**（`spec.md` 第 4 節）：
  - 用和 test02 notebook 相同的函式：官方教學 `utils.py` 的 `SGFParsePlayerIdentification`、`network.py` 的 `GoPlayerResNet()`，fp32、不用 autocast、一盤一次前向。
  - 算 1.6 節黑棋訓練資料的 232,165 盤。
  - 估計約 50 分鐘：`c1d_cnn_bench.py` 量到每盤 12.67 毫秒，推算約 0.82 小時。照規則排夜間，開跑前看 `nvidia-smi`，C4 要用就讓出。
  - 結果放 `data/prereg06/cnn_fit/`（不進 git），SHA 寫進要 commit 的 `results/prereg06/design/cnn_fit_report.json`。
- **模擬池**：官方形狀、一題對一人。調參池 seed 300–319（20 池），評估池 seed 400–419（20 池）。探索用過的 0–19、100–109、200–219 都不用。
- **CNN 代表性檢查（E0 上限）**：
  - 在調參池上算 E0 單用（Q00，不做 CSLS）的平均分。
  - 超過 **0.50** 就判定「fit 模擬裡的 CNN 不具代表性」：沒有 V1，第 2.6 節的門檻和 `k*` 全部改用替身 A 選。
  - 0.50 是在看到任何 fit 玩家的 CNN 分數之前定的。參考：本機驗證候選池上，CNN 單用的自我檢索是 0.4006（`results/prereg02/player/sr_report.json`，驗證玩家對 CNN 是樣本外的）；上限留了約 0.10 的空間給題目形狀的差異。
- **格子**：w_L ∈ {0.5, 1, 1.33, 2, 2.66, 4}、w_J ∈ {0, 0.25, 0.5, 1}、w_S ∈ {0, 0.25, 0.5, 1}，共 96 組。每組都加 CSLS10（調參時 k 固定 10，`k*` 在 2.6 節才重驗），目標是調參池的平均分。
- **V1 的條件**（三條都要成立）：
  - CNN 代表性檢查通過；
  - 最高分那一組在調參池上比 V2 的權重（2.66、0.5、0.5）高 ≥0.003；
  - 同一組在評估池 seed 400–419 上的平均分不低於 V2（這一條修正從 96 組挑最大的挑選偏差）。
  - 任一條不成立：沒有 V1。
- **只記錄**：
  - 評估池上 Q04（真實 CNN）、V0–V3 的分數，以及它們對 Q04 的差（20 池的 t 區間）。
  - CNN 單用（Q00）在調參池、評估池的分數。
  - CNN 是樣本內的，所以評估池的數字裡 CNN 那一半偏強，不能當成 06 的增益估計。
- **K-CNN（10/07 23:59）前沒算完**（權重沒到、SHA 不符或跑失敗）：沒有 V1，第 2.6 節改用替身 A，其他照常。之後權重才到也不改，因為已經凍結。

### 2.6 一對一開關和 `k*`

- **偵測統計量**（照探索的 `s8_switch.py`）：
  - F＝某個組合在做 CSLS 之前的分數；Z＝F 對每一題做 z 分數。
  - Hungarian：在 Z 上解「每題配一位不同的候選、總和最大」的指派（`scipy.optimize.linear_sum_assignment`）。
  - forced_cost＝每一題「自己的第 1 名」和「Hungarian 指派的候選」的 Z 分數差，取平均。
  - r_forced_cost＝這一池真題的 forced_cost ÷ 同一池第 0、1 輪假題池（2.3 節）forced_cost 的平均。
- **打開時怎麼排**：第 1 名是 Hungarian 指派的候選；第 2–5 名依 CSLS3（k＝3）的分數排，跳過已經排第 1 的那位。
- **每個組合各有自己的門檻**：清單裡的每個組合 X（V2、V1（有的話）、V3、V0）各自校準一個門檻 `r*_X` 和「能用」判定。在某一池，偵測器用的 F、Hungarian 用的 F、「關」用的 F，都是那一池選中的組合的 F，門檻用那個組合的 `r*_X`。所以校準時的 F 和使用時的 F 一定相同。
- **F 的來源**：有 fit 玩家的真實 CNN、而且 2.5 節的代表性檢查通過時，每個組合都分別用「真實 E0」和「替身 A 代替 z(E0)」各校準一次；否則只用替身 A。
- **池**：
  - 題目結構 11 種：o2o、u390、u375、u350、d80、u300、wr（定義同探索的 `make_pool`），加上 d1、d2、d3、d5（同一個 `make_pool` 的 `d<m>`：m 位候選沒題、另外 m 位各兩題，其餘一人一題）。
  - 選門檻：seed 500–529，每種結構 30 池。評估：seed 600–619，每種結構 20 池。
- **選法**（每個組合 X、每種 F 來源各做一次；「關」是 CSLSk，「開」是 Hungarian＋CSLS3）：
  - 候選門檻是選門檻池上出現過的 r_forced_cost 值。
  - 「可行」＝每種結構的平均（開 − 關）≥ −0.0005，而且沒有任何一個不是一題對一人的池虧超過 0.005。
  - 可行的門檻裡，取 11 種結構平均增益最大的那個；平手時取最小的門檻。
  - 有兩種 F 來源時，`r*_X` 取兩個門檻中較小的那個。
- **能不能用**（在評估池上判斷，用 `r*_X`；寫進 `frozen_config.json`），四條都要成立：
  - o2o 池打開的比例 ≥0.8（用來校準的每種 F 來源都要）；
  - u390、u375、u350、d80、u300、wr 這 120 池打開 0 次（每種 F 來源都要）；
  - d1、d2、d3、d5 每種結構照偵測器的平均（開 − 關）≥ −0.0005，沒有任何一池虧超過 0.005；
  - 同一個門檻用在「只有開局＋v2」（沒有 CNN 也沒有替身）時，那 120 池也是 0 次誤開。
  - 不成立：組合 X 的開關「不能用」。選到 X 的池不准打開。所有組合都不能用時：這一季不用開關，06 只有 CSLSk。
- **`k*` 的重驗**（在選門檻池上，所有 11 種結構）：
  - F 用清單第一個在場的組合：V2（真實 E0；沒有就替身 A）。
  - 依序試 k＝10、20、40，取第一個「每種結構的平均（CSLSk − 不修正）≥ 0」的 k。
  - 三個都不行：`k*`＝不修正。
  - `k*` 用在所有組合；這一點寫進揭露。
- **評估池上另外記錄**：每個組合的否決統計（同 `c1d_veto_check`：每種結構 d＝開 − 關的分布、d<0 的比例）。
- **本機驗證集**本來就是一題對一人（VALIDATION.md），所以開關不算進採用（第 3 節）。

### 2.7 和 PREREG_02 的防護、T1–T6 怎麼互動（官方池，每一池各自判斷）

06 疊在 test02 的結果上，不改 test02 的任何判斷。

| 狀況 | 06 在這一池怎麼做 |
|---|---|
| T1 是 FAIL_STRUCT | 照 test02 停下 |
| T1 是 FAIL_OPENING | 06 整個不用（兩池都不用），最終照 U5（test02） |
| test02 這一池的防護在第 3 層（開局不可靠、只用 CNN），或 test02 這一池的閘門沒過（改用 Q00） | 這一池不用 06，交件檔裡這一池就是 test02 的那 400 列（逐位元組相同）。v2 的三項跟著開局一起不用：JO 本身就是開局 token；S2 和 LDA 是在「題目和候選隨機切」的模擬裡驗證的，防護判定切法不像隨機時，它們也不可信 |
| 防護在第 2 層 | 照常用 06。PREREG_02 第 2 層的規則（開局權重改成 min(w,1)）照套在開局類的兩項：O_raw 的 1 和 JO 的 0.5 本來就 ≤1。LDA 和 S2 不是開局項，權重不變；依據是探索的 `s9_split_check.json`：照檔案順序切時，v2＋CSLS3 對開局的增益只比隨機切差 −0.0014 [−0.0070, +0.0042]。這一池不准開一對一，因此整份交件檔都沒有 06-ON |
| 一對一開關 | 條件見第 5.3 節，其中一條是兩池都在第 1 層 |
| 06 的閘門 sr(選中) − sr(R02) < 0 | 這一池不用 06，交 test02 的那些列 |
| T2(a)／(d) 不過 | 照 PREREG_02 第 4 節：開 split v2、另寫登記。06 照常做，不利用這些複製。沒有 06-ON |
| T2(b) 不過（題目的棋出現在同池候選裡） | 照 PREREG_02 第 4 節：開 split v2、另寫登記。06 照常做，不刻意利用這些複製。沒有 06-ON：題目和候選的抽法和模擬不同，開關沒有在這種結構上驗證過 |
| T2(c)：同一份題目檔裡有棋出現在兩題以上（複製數 > 0，不看門檻） | 一對一開關不准開，兩池都不准（DATA.md 第 6 條：這是同一人出多題的跡象） |
| T3／T3b 判定重疊（候選是訓練玩家） | 只記錄，06 照常。凍結的 LDA 可能看過這些人的棋，對他們只會更準 |
| T4、T5 | 只記錄。盤數清單 K 本來就取自題目的 `num_games` |
| T6 | PREREG_02 第 9 節第 9 條（T6 只記錄、**沒有官方明文就不用一題對一人**）和 DATA.md 第 7 條，對 test02（Q04）照舊。對 06-ON，這一條由第 5.3 節的統計條件取代，見第 5.5 節的例外 E4。06 另外記兩池的 r_forced_cost，以及 Hungarian 改了幾題的第 1 名 |

### 2.8 凍結前的檢查

- **leakcheck**（只做一次，第 9 節第 3 步）：照 VALIDATION rev 2 的讀法，用 `load_rows(part="fit", local_dir=data/local)` 比對 fit 的棋和本機驗證檔的棋（只算 SHA-1 雜湊，只讀有棋譜的那兩份檔，不讀答案）。重疊要是 0。其他設計步驟用 `local_dir=None`（人數和盤數的核對照常做），不碰 `data/local`。
- **selfcheck**（程式有沒有移植對）：在探索的評估池 seed 0–4（只有黑棋）上比較新工具和探索的函式：
  - 只比交替的盤：O_raw、JO、T_S2 逐值相同，差 ≤1e-9。
  - LDA64 項的 z 分數差 ≤1e-6（詞彙排序不同，數值路徑會有些微差異）。
  - 不交替的盤數記錄。
  - v2 不修正、v2 加 CSLS10 在 seed 0–19 的平均分，四捨五入到小數第 4 位要等於 0.9819、0.9869。
  - 不符就停下查。查明是數值路徑或不交替盤的差異、不是邏輯錯誤時，寫 rev 2，以 repo 算出的數字為準並揭露。
  - 探索的快取（`cache_black.pkl`、`fam_*.npz`、`dense_*.npy`）原本在 session 暫存區；selfcheck 用的那一份放在 `data/prereg06/explore_cache/`（重建或複製），SHA 記進報告。
- **D-mix**（白棋那一半能不能用；只記錄，但有停止條件）：
  - 用 fit 玩家做 10 個像本機驗證集的黑白混合池（seed 700–709）：兩色合計 ≥105 盤的 fit 玩家每級 40 人，候選取洗牌後的前 100 盤（黑白混合），題目取接下來的 k 盤，k＝min(randint(5,20), 剩下的盤數)（照 `tools/make_val.py`）。沒有 CNN。
  - 主比較：「O_raw＋v2（每色分開、權重同 V2）＋CSLS10」對「O_raw＋CSLS10」。
  - 只記錄：「O_raw＋v2（只用黑棋算 v2，白棋不進 v2）＋CSLS10」對同一個基準。
  - 主比較的平均差 ≤0 時，06 不在本機驗證集打分（記為「未送分」）。例外：在 `freeze` 之前用 rev 2 改了顏色的處理，而且重跑 D-mix 通過。

## 3. H6P：在本機驗證集打一次分

順序由工具強制：

1. **先凍結**：`frozen_config.json` commit 之後，才准在驗證玩家的棋上算任何 v2 特徵。它記錄：
   - 兩份 LDA 的 SHA 和詞彙；
   - V1 的權重，或「沒有 V1」和原因；
   - 每個組合的 `r*_X` 和「能用」判定，或「不用開關」；
   - `k*`；
   - 所有 seed 和雜湊鹽；
   - CNN embedding 報告的 SHA，以及代表性檢查的結果；
   - 工具的 SHA。
2. **`apply`**（`spec.md` 第 6 節）：
   - 讀 `data/local/player_val_candidates.csv`、`player_val_questions.csv`（只讀棋譜、`color`、`color_i`、`num_games`，不讀答案）和 prereg02 的 dumps。
   - 先用 06 的程式算 Q04（不加 v2、不做 CSLS），必須和 `results/prereg02/player/player_val_top5.csv`（SHA `69ca74a6…`）逐位元組相同；不同就停下。
   - 在驗證候選池上跑 sr06，照 2.4 節選組合，寫出：
     - `results/prereg06/player/player_val_top5.csv`：主檔，選中的組合＋CSLSk。
     - `results/prereg06/player/player_val_top5_switch.csv`：次要檔，同一個組合的 F，**強制**打開一對一（第 1 名用 Hungarian，第 2–5 名用 CSLS3）。
     - `apply_report.json`：每個組合的 sr、選中的組合、閘門、驗證池的 r_forced_cost 和偵測器會不會打開（只記錄）、次要檔的結構檢查（第 1 名是 400 位不同的候選；第 2–5 名依 CSLS3 排序而且不含第 1 名）、`score_allowed`。
     - 只記錄的單色檢查：在驗證候選池上只用黑棋做假題（其他同 sr06），算 sr(選中) − sr(V0)，看官方那種全黑形狀下 v2 是否還在加分。
   - 兩份都跑 `tools/check_submission.py player <檔> --questions data/local/player_val_questions.csv --candidates data/local/player_val_candidates.csv`。
3. **送分閘門**：以下全部成立，`score_allowed` 才是真：
   - 選中的不是 V0；
   - sr(選中) − sr(V0) ≥ +0.005；
   - D-mix 通過；
   - Q04 逐位元組重現通過；
   - 次要檔的結構檢查通過；
   - 兩份檔都通過格式檢查。
   - 不成立就記為「未送分」：打分機會保留，競賽一照 Q04。
4. **指令**（寫死；只在 `score_allowed` 為真時跑，兩行接著跑）：
   - 主要：`.venv/Scripts/python.exe tools/score.py compare-player results/prereg02/player/player_val_top5.csv results/prereg06/player/player_val_top5.csv data/local/player_val_answers.csv data/local/player_val_candidates.csv --ledger results/score_ledger.csv --note "prereg06 H6P (A = prereg02 Q04, reference re-read)" --json results/prereg06/player/score_H6P.json`
   - 次要：`.venv/Scripts/python.exe tools/score.py compare-player results/prereg06/player/player_val_top5.csv results/prereg06/player/player_val_top5_switch.csv data/local/player_val_answers.csv data/local/player_val_candidates.csv --ledger results/score_ledger.csv --note "prereg06 H6P secondary (A = 06 main re-read, B = one-to-one switch forced on; one-way check)" --json results/prereg06/player/score_H6P_switch.json`

- **採用條件**：主要指令的 B−A（06 − Q04）平均分差，95% 信賴區間下界 > 0（配對 bootstrap 10,000 次、seed 0，就是 `score.py` 的做法）。
- **次要指令只能關掉東西（單向）**：
  - 次要指令的 B−A（開 − 關）點估計 < 0 時：這一季不做 06-ON。本機驗證池是一題對一人，模擬預期開關會加分（v2 約 +0.011），所以點估計是負的，代表 Hungarian 那段程式有錯、或真實 CNN 下開關不成立。之後只有 rev ≥4 的修訂指出並修好具體的程式錯誤時，才能恢復。
  - 點估計 ≥ 0 時：什麼都不打開。測試日要不要做 06-ON，仍然只看第 5.3 節。
  - 次要指令不影響 06（06-OFF）的採用。
- **對照組重讀**：
  - 主要指令的 A 側 400 題，平均分要和帳上的 prereg02 H2P（0.8829546403931485）差 ≤1e-12，並記錄打分用的 Python（`.venv`，3.12）。不符就停下，查清楚之前不採用。
  - 次要指令的 A 側（06 主檔）要和主要指令的 B 側差 ≤1e-12。
- **合理性門檻**：06 主檔低於 0.85 時，先查 bug；要指出具體的程式錯誤才算。
- **預算**：競賽一整季可以打 4 次新方法，已用 1 次（02）。06 最多用 1 次，主要和次要指令算同一次；查 bug 重跑也算，並記帳。打分紀錄帳會多 4 列。
- **檢定力（估計）**：探索裡配對差的每池標準差約 0.0085–0.0100（`critic/cr_derived.json`），所以 400 題的 H6P 只有在真實增益約 +0.02 以上時才有較大機會通過；真實 +0.02 時通過的機會估計約一半到三分之二。沒通過不代表 v2 無效，只代表這一季不用。
- **CSLS 的偏差**：CSLS 在一題對一人的池子增益比較大（替身 A：+0.0128，有放回抽時 +0.0053），所以 H6P 的差值裡，CSLS 貢獻的那一份偏樂觀。報告要寫明。

## 4. 截止點（寫死；沒趕上就照表做，不拿額外的打分機會補）

| 截止點 | 條件 | 沒趕上 |
|---|---|---|
| K-CNN 10/07（三）23:59 | fit 玩家的 CNN embedding 算完，報告已 commit | 沒有 V1；開關的門檻和 `k*` 用替身 A 選 |
| K3' 10/16（五）23:59 | 送分閘門已判定；過了的話，H6P 已打分、RESULTS 已 commit | 這一季不用 06 |
| K5' 11/03（二）23:59（目標 10/20） | 採用時：測試日擴充和 rev 3 已 commit、合併回 main，本機演練通過 | 這一季不用 06 |
| K6' 11/21（六）23:59 | 採用時：06 在官方測試集有一次成功的正式執行（第 5.2 節），已 commit | 最終照 U5（Q04） |
| K7' 11/22（日）23:59 | 要用開關時：06-ON 已上傳、兩份 public 分數已記錄、否決已判定 | 不用開關，最終交 06-OFF。06-ON 已經上傳、但否決沒在 K7' 前判定時，再上傳一次 06-OFF，讓它成為最後一次上傳 |

## 5. 採用後的測試日（細節在 K5' 前另寫 rev 3）

### 5.1 保護舊路徑（rev 1 就生效）

- 這些一律不改，`tools/official_log.py` 只准加行：
  - R20／test02 的 39 個檔（`results/prereg04/design/r20_path_fingerprint.json`，tag `r20-testday-pre04`）；
  - 04 的 21 個檔（`results/prereg06/design/p04_path_fingerprint.json`，tag `p04-testday-pre06`）。
- 06 的測試日程式全部是新檔，放在分支 `p06-testday`：`tools/t06_player.py`、`tools/t06_fake_tests.py`、`tests/test_t06_*.py`、`docs/TESTDAY_06.md`。輸出只寫 `results/test06/`，大檔放 `data/test06/`。
- 寫 06 的程式之前，先記下 test02 競賽一假測試演練的輸出當基準（`results/test06/design/q04_fake_golden.json`）；合併前要和它逐位元組相同。
- 不需要新的 Colab notebook：06 讀 test02 notebook 已經寫出、已 commit 的每池 dumps，其他都在本機 CPU 上算。

### 5.2 要做的東西

- `tools/t06_player.py`：
  - `feats`：算三組（seam_val、public、private）的 v2 特徵。
  - `seam`：在 seam_val（本機驗證池）上整套重跑，要對得上 H6P 的輸出：
    - 選中的組合、閘門判定、偵測器開或關都相同；
    - test02 的 seam_val dumps 和 prereg02 的逐位元組相同時，兩份 Top-5 檔也要逐位元組相同；不同時，至少 398/400 題的 Top-5 相同（同 PREREG_02 第 9 節第 2 條）。
  - `apply --pool public|private`：照 2.7 節判斷，選組合、過閘門、跑偵測器，寫出這一池的 Top-5（關）和需要時的 Top-5（開）。
  - `submit`：組出 `results/test06/player/off/submission_player.csv`，需要時再組 `results/test06/player/on/submission_player.csv`。兩份旁邊各有一份 `t06_submit_report.json`，都要通過 `check_submission.py --official`。
  - `veto`：讀 `results/official_scores.json`，照 5.3 節判定，寫出 `results/test06/player/veto.json`。
  - `status`：列出還差哪一步，以及 K6'／K7' 的倒數。
- `tools/official_log.py` 只加行：
  - 認得 `results/test06/player/{off,on}/` 的檔。報告名稱叫 `t06_submit_report.json`，不叫 `submit_report.json`：萬一沒認出來，Q04 的路徑會因為找不到報告而拒絕。
  - 方法名稱：「prereg06 06-OFF: Q04 + v2 style + CSLS (甲)」和「prereg06 06-ON: 06-OFF + one-to-one switch (甲)」。
  - 檢查上傳順序；`public` 之後印出同名檔警告（5.3 節）和否決的判定；`final --comp player` 照 5.4 節。
  - `status` 只在 log 裡有 06 的紀錄時才多印幾行；沒有 06 紀錄時，輸出和現在逐位元組相同。
- `docs/TESTDAY_06.md`：Claude 的步驟。`docs/COLAB_RUN.md` 不用動，因為使用者不用多跑 Colab。
- **一次成功的正式執行**（K6'）：`feats`、`seam`、兩池的 `apply`、`submit`（至少 06-OFF）都在正式模式下以結束碼 0 跑完，報告已 commit，而且最後一次 commit 的時間 ≤ K6'。

### 5.3 上傳順序和 public 否決（一對一開關的保險）

- **先後**：
  1. 11/04 照 `docs/TESTDAY.md` 交 Q04（test02），記成 `uploaded` 並 commit。06 的任何上傳都要等這一筆。
  2. 上傳 06-OFF：兩池都只做 CSLSk，不用一對一。這屬於 TESTDAY「之後的上傳」(b)：之後的登記依本機驗證規則採用的方法。
  3. 上傳 06-ON：06-OFF 已上傳、而且下面的條件全部成立才做。private 池用一對一；**public 池一律強制打開一對一**，當成探測。這一次上傳是第 5.5 節登記的例外。
  4. 兩份的 public 分數都記進 `results/official_scores.json` 之後，算 d＝public(06-ON) − public(06-OFF)，用 `official_scores.json` 裡記的值（照 AIdea 顯示的全部位數）。
  5. **d < 0 就否決**：最終交 06-OFF，再上傳一次 06-OFF，讓它成為最後一次上傳（TESTDAY「之後的上傳」(c)）。**d ≥ 0**（包括 d＝0）：最終交 06-ON，它已經是最後一次上傳。
- **同名檔警告**（Q04、06-OFF、06-ON 三份都叫 `submission_player.csv`，傳錯檔會讓 d 判錯）：
  - 三份裡任兩份的 public 分數完全相同，而 `t06_submit_report.json` 記的「這兩份在 public 池第 1 名不同的題數」> 0 時：否決暫停，先請使用者截 AIdea 的上傳紀錄核對是哪一份。
  - 核對出傳錯：照 TESTDAY 重傳正確的那份（屬於「修 bug」），重記 public 分數再判。
  - 核對不出、或 K7' 前來不及：不用開關，最終交 06-OFF（要的話再上傳一次 06-OFF）。
  - 這條規則比照 `official_log.py` 的 `t04_public_lines`。
- **06-ON 的條件**（全部成立才做；這些條件在 06-ON 上取代 PREREG_02 T6 的「官方明文」和 DATA.md 第 7 條的「確認」，見 5.5 節例外 E4）：
  - private 池選中的組合在 2.6 節判定「能用」，而且 private 的偵測器打開（r_forced_cost ≤ 那個組合的 `r*_X`）。
  - H6P 的次要指令沒有關掉開關（第 3 節）。
  - 兩池都是 400 題、400 位候選（T1）。
  - 兩份題目檔 T2(c) 的複製數都是 0（DATA.md 第 6 條的檢查：Claude 讀 `t2.json`，寫進 RESULTS），而且 T2(a)、(b)、(d) 都沒有不過。DATA.md 第 5 條（題數和人數、題目和候選有沒有同一盤棋）也照常記錄。
  - 兩池都用了 06（沒有退回 test02 的列），而且防護都在第 1 層。
  - AIdea 的頁面沒有寫「不是一題對一人」。
  - K7' 還沒過。
- **為什麼 public 要強制打開**：最終名次只看 private，public 只是比賽過程中的排行，所以 public 那 400 題可以拿來做這個實驗。public 池自己的偵測器只記錄，不參與判斷。public 池在 06-ON 裡用的是 public 自己選中的組合的 F（開和關只差在一對一，d 才乾淨）。
- **為什麼這是保險，不是挑版本**：
  - 只比兩份事前定好、差別只在「開不開一對一」的檔；
  - 規則和門檻（d < 0）在拿到任何測試資料之前就寫死；
  - public 分數只能把開關關掉，不能打開；private 要開，還得它自己的偵測器先說是；
  - 其他地方都不看 public 分數：V0–V3 的選擇、06 和 Q04 之間的選擇，都照 PREREG_02 第 9 節第 12 條、VALIDATION 紀律 4。只有 5.5 節列出的那一個決定例外。
- **依據**（`c1d_veto_check.py`，用探索的 350 個模擬池，每種結構 50 池；這裡的 d 是那一池「開 − 關」的分差）：
  - v2 特徵：
    - 一題對一人的 50 池全部 d > 0（最小 +0.0006，平均 +0.0114）。
    - u390 的 50 池有 49 池 d < 0。
    - u375、u350、d80、u300、wr 全部 d < 0，而且都 ≤ −0.026。
  - 替身 A：
    - 一對一的 50 池全部 d > 0（最小 +0.0091）。
    - u390 只有 58% 的池 d < 0，但平均只差 −0.0003，就算沒否決也幾乎不虧。
    - 其他結構全部 d < 0。
  - 審查補跑的 d1–d12（`review_r1.md` 3.1）：近一對一、偵測器會打開的結構，平均都賺。
  - 所以：真的是一題對一人時，幾乎不會誤否決；不是的時候，會造成大虧損的結構都會被否決。
- **0.05 警示**：06-ON 的 public 比 06-OFF 低 0.05 以上時，`official_log.py` 照樣標 `investigate`。這種情況本來就由否決處理（最終交 06-OFF）：Claude 照常快速查一次 ID、列順序、池的配對，但不管查到什麼，結果都是 06-OFF。
- **上傳次數**：最多 06-OFF、06-ON、再一次 06-OFF，共 3 次，加上 11/04 的 Q04。每天上限 5 次。每一次都要使用者在對話裡對那一份檔說 OK。
- **報告怎麼寫**：在「看了官方 public 分數才做的選擇」一節寫出一對一開關的規則、兩份檔的 public 分數、d、有沒有否決，並說明 public 池是當成探測、強制打開的，以及第 5.5 節的例外。

### 5.4 最後一版（競賽一）

照 PREREG_04 第 11.1 節第 12 條「取代 U5」的前例，不改 `docs/TESTDAY.md`。

- 一筆 06 的上傳「符合」＝以下全部成立：
  - H6P 採用；
  - K6' 成立；
  - T1 是 PASS 或 NORMALIZED；
  - 06 的 seam 通過；
  - 報告和交件檔從 `prepare` 之後沒有變；
  - 報告記下的輸入（test02 的 dumps 和報告、測試檔、`frozen_config.json`）沒有變；
  - 06-ON 另外要 5.3 節的條件成立、而且沒有被否決。
- 有符合的 06-OFF：最終交 06，交 06-ON 還是 06-OFF 照 5.3 節決定。沒有符合的：照 U5，交 Q04（test02 的每池程序）。
- `docs/TESTDAY.md` 的 U5（「兩場都是 prereg02（甲）」）和 `docs/TESTDAY_04.md` 開頭的「競賽一照 U5 不變」，對競賽一由本節取代；兩份都在指紋裡，所以不改。
- 06 判定之前，不執行 `official_log.py final --comp player`。已經設過也沒關係，照 `docs/TESTDAY_06.md` 重設。
- 要推翻這個結果，只能靠本登記的修訂（rev 4 起）：指出具體的程式錯誤，以及它破壞的是哪一條；修訂的 commit 時間要晚於被推翻的紀錄（同 PREREG_04 第 11.3 節第 20 條）。
- public 分數比上一版低 0.05 以上（例如 06-OFF 比 Q04 低）時，只查錯：ID、列順序、池的配對、接縫。查不到錯就維持本節的結果，並寫進報告。06-ON 和 06-OFF 之間的情況照 5.3 節的 0.05 警示。

### 5.5 對既有規則的例外（範圍只限競賽一的 06-ON 和它的否決）

下面這些條文都在指紋裡或已凍結，不改原文；照 PREREG_04 第 11.1 節第 12 條取代 U5 的前例，在這裡登記例外。例外以外的情況，原條文照舊。

| 編號 | 被取代的條文 | 06 的例外 |
|---|---|---|
| E1 | `docs/TESTDAY.md`「之後的上傳（U3–U6）」：只有 (a) 修 bug、(b) 之後的登記依本機驗證規則採用的方法、(c) 重傳最終選擇三種情況能再上傳；「不為了在 public 排行榜試變體而上傳」 | 新增一類 (d)：06-ON，一季最多一次，條件和順序照 5.3 節。06-OFF 屬於 (b)，否決後重傳 06-OFF 屬於 (c) |
| E2 | `docs/VALIDATION.md` 紀律 4（rev 2）：最後一次上傳交哪一版只依本機驗證集的主要指標和事前登記的採用條件決定；public 分數只用來抓格式錯誤和大幅退步 | 只有「06-ON 還是 06-OFF」這一個決定，依 5.3 節的 d（public 分數）單向判定：只能把開關關掉，不能打開，也不能用來在 06 和 Q04、或 V0–V3 之間選 |
| E3 | PREREG_02 第 9 節第 12 條：public 低 0.05 以上只追查錯誤、不據此換方法；最後一次上傳只依本機驗證的主要指標和事前登記的採用條件決定 | 同 E2；06-ON 對 06-OFF 低 0.05 以上的情況由否決處理（5.3 節「0.05 警示」） |
| E4 | PREREG_02 第 3 節「不用『一題對一人』的結構」、第 9 節第 9 條「T6 只記錄；沒有官方明文就不用一題對一人」、DATA.md「拿到測試集後」第 7 條「確認官方是一題對一人之前，不採用任何利用一一對應的做法」 | 對 06-ON，「官方明文」和「確認」由 5.3 節的統計條件（偵測器、T1、T2、防護層、AIdea 頁面）加上 public 否決取代。test02（Q04）和 06-OFF 照舊不用一一對應 |

- **為什麼可以例外**：例外的規則、門檻、條件都在拿到任何測試資料前寫死；只比兩份事前定好的檔；public 只能關掉開關；private 要開，還得它自己的偵測器同意。10/2 的決定（CLAUDE.md「06 探索」第 1 條）已經選了「用 public 分數當這個開關的否決保險」，本節是把它正式登記。
- **不做這個例外的選項**：不做 06-ON、只交 06-OFF。這樣就沒有任何衝突，代價是一題對一人時放棄約 +0.011（模擬，估計）。
- 寫進第 8 節的揭露，也寫進報告。

## 6. 另一場：不交換輸出

- 兩場只共用前處理（`tools/c10data.py`、`tools/p02_moves.py`）。競賽二（R20／04／PREREG_07）的任何輸出不進 06，06 的任何輸出也不進競賽二。
- 不用競賽二的測試資料。critic 第 1 節第 2 項提到的跨競賽比對不做，因為規則第 8 條有疑義。

## 7. 刻意不放進 06 的

| 做法 | 理由 |
|---|---|
| GPU 風格模型（GE2E＋Transformer），或重訓、微調玩家 CNN | 模擬中 v2＋CSLS3 之後最多只剩約 +0.008（`c1_headroom` (e)），11/04 前不值得做 |
| 競賽一用 KataGo HumanSL | 剩下的錯不是棋力資訊造成的；兩場共用特徵在規則上也有疑慮（critic 第 1 節第 10 項） |
| CSLS3 永遠開著 | 有放回抽時是 −0.0070 |
| dual softmax、unbalanced Sinkhorn | 沒有任何一組參數在 7 種結構都不為負（`s8_switch.json`） |
| 在官方候選池上學 LDA | 預期只有 +0.000～+0.005，又多一套沒驗證過的流程（critic 第 1 節第 4 項） |
| 用訓練集補候選的參考棋 | 只有候選剛好是訓練玩家時才有用，要等 T3 判斷；要做就另寫登記 |
| 用 inner 或全部 8,007 人學 LDA | 理由見 1.6 節；驗證玩家一律不能用 |
| 依 public 分數挑 V0–V3 或任何其他版本 | public 分數只用在 5.3 節的否決（5.5 節的例外） |

## 8. 揭露（寫進 RESULTS 和報告）

- v2 的特徵、權重、CSLS、開關的規則，都是在 fit 玩家的模擬上看過很多次評估池才定的（第 0 節的自由度）。只有 H6P 算數。
- 本登記用的新 seed（300–709）抽的仍是探索用過的同一批 fit 玩家，不是新資料。
- 玩家 CNN 對 fit 玩家是樣本內的，第 2.5 節的調參和第 2.6 節的門檻因此偏向 CNN；用 V2 優先、E0 上限、評估池條件、門檻取真實 CNN 和替身 A 較小者來補。
- `k*` 只在 V2 的 F 上重驗，用在所有組合。
- LDA 只用 fit 玩家學。白棋的投影沒有在模擬裡驗證過，D-mix 只看了混色池上的開局＋v2。
- 不交替的盤（含 pass 等）不進 v2，件數記錄；探索沒有排除它們。
- H6P 的差值裡包含 CSLS 在一題對一人池較大的增益；H6P 的檢定力有限（第 3 節）。
- H6P 的次要指令（強制打開一對一）只能把開關關掉。
- 測試日用 public 分數當一對一開關的否決（5.3 節），public 池被當成探測；這是第 5.5 節登記的、對 TESTDAY、VALIDATION 紀律 4、PREREG_02 第 3 節和第 9 節第 9、12 條、DATA.md 第 7 條的例外。
- 探索時的一次越界：`c2_headroom` 曾經從 repo 根目錄做過一次遞迴 grep，可能走過 `data/local` 的資料夾名稱，但沒有打開任何檔（`critic/REPORT.md` 2.5 第 4 點）。這件事和競賽一無關，照實揭露。
- 生成式 AI 協助的部分照規則揭露（`competitions/_drafts/disclosure-template.md`）。

## 9. 順序

1. 把暫存區的 `c1d_veto_check.json`、`c1d_cnn_bench.json`、`s8_rows.json` 和審查的檢查程式（`review_r1/`）放進 `results/prereg06/design/`，連同本登記的 rev 1、`spec.md`、兩支輔助程式、`review_r1.md`、`review_r1_response.md` 一起 commit，再用一個 commit 填上 rev 1 的編號。04 路徑的指紋和 tag `p04-testday-pre06` 已在 `d90b9be` 完成。
2. 寫程式：`tools/p06_style.py`、`p06_sim.py`、`p06_player.py`、`p06_cnn_fit.py`、`tests/test_p06_*.py`，經獨立審查後 commit。實作和登記不同的地方寫成 rev 2，在第 7 步之前 commit。
3. `leakcheck`、`build`（fit 的特徵快取）、`lda`（兩份凍結的投影）、`selfcheck` → commit 報告。
4. 權重到了就跑 `cnn-fit`（4090，排夜間）→ commit 報告。K-CNN 前沒到就跳過。
5. `tune`（2.5 節）→ commit；`switch`（2.6 節，含 `k*`）→ commit；`dmix`（2.8 節）→ commit。
6. 確認第 3–5 步的報告都 commit 了、工具都乾淨。
7. `freeze` → commit `results/prereg06/player/frozen_config.json`。從這之後才准碰驗證玩家的棋。
8. `apply`、`check_submission` → commit。`score_allowed` 為真，才跑第 3 節的兩行指令。
9. 打分紀錄帳、分數 JSON、`results/prereg06/RESULTS.md`（數字由另一個代理獨立核對）、地圖一起 commit。
10. 採用的話，在分支 `p06-testday` 照第 5 節做測試日擴充：先記 Q04 的假測試基準 → 實作 → 從三個角度審查 → rev 3 → 演練 → 防誤改閘門 → 在 K5' 前合併。

## 10. 時程（台灣時間）

| 日期 | 事 | 誰 |
|---|---|---|
| 10/03–10/05 | commit rev 1 | Claude |
| 10/03 起 | 使用者有空時下載 `best_model.pth`（K-CNN 10/07 前要到），Claude 核對 SHA 後放進 `data/prereg01_weights/player/` | 使用者做一件事 |
| 10/05–10/07 | 第 2–3 步 | Claude |
| 10/06–10/07 夜 | 第 4 步（4090，約 50 分鐘，估計）。和 PREREG_07 共用夜間的 4090，先排這一步，因為它決定 K-CNN 能不能趕上；10/24 前 C4 優先 | Claude |
| 10/08–10/09 | 第 5–7 步（`switch` 只用 CPU，估計 1.5–3 小時，排夜間） | Claude |
| 10/10 前後 | 第 8–9 步（H6P） | Claude |
| K3' 10/16 | 最晚的判定日 | — |
| 10/11–10/20 | 測試日擴充、rev 3、演練、合併（目標 10/20，硬截止 11/03 23:59） | Claude |
| 11/01 | 使用者決定 11 月投入多少；選項裡列出 06 測試日多出的 1–3 次上傳 | 使用者 |
| 11/04 | test02 照舊，先交 Q04 | 使用者上傳 |
| 11/05–11/06 | 06-OFF、（需要時）06-ON、否決判定 | 每次上傳前使用者說 OK |
| 11/22–11/23 | 跑 `status`；最後一次上傳在 11/23 23:59 前 | 使用者 |

使用者在 10 月只多一件事：下載權重，這本來就在待辦裡。測試日多 1–3 次上傳。10/19–10/27（玉山最忙的那段）不需要使用者做任何事。

## 11. rev 3：測試日程序（K5' 前登記；拿到任何測試資料之前）

本節是第 5 節「細節在 K5' 前另寫 rev 3」的內容。和第 1–10 節不一致的地方以第 1–10 節為準，只有第 19、20、24 條明寫比第 5.3 節更嚴的地方例外（那些只會把一對一開關關掉，不會打開，符合第 5.3 節「public 分數只能把開關關掉」）；本節其他條文只補上第 1–10 節沒有寫死、測試日一定要有答案的地方。第 2.7 節的表本身不改（本檔只准加行），它的補充寫在第 7–10 條。Claude 的步驟在 `docs/TESTDAY_06.md`。

### 11.1 工具和檔案

1. 測試日的程式全部是新檔：`tools/t06_player.py`、`tools/t06_fake_tests.py`、`tests/test_t06_*.py`、`tests/test_p06_player_hygiene.py`、`tests/conftest.py`（第 29 條）。`tools/official_log.py` 只加行。輸出只寫 `results/test06/player/`，特徵快取在 `data/test06/player/`（不進 git，SHA-256 記在報告裡）。`.gitattributes` 加一行 `results/test06/** -text`。
2. `t06_player.py` 的指令：`feats --pool seam_val|public|private`、`seam`、`apply --pool public|private`、`submit`、`veto`、`status`。`status` 只讀不寫；其他每個指令寫一份 JSON 報告（輸入的 SHA-256、工具的 SHA-256、git HEAD、版本、秒數、mode）。輸出已存在時要 `--rerun --reason`，理由和舊檔的 SHA 記進報告。
3. 結束碼：0 完成；2 拒絕（前提不符；工具內部錯誤也是 2，附 traceback，那不是登記的判定）；3 登記的「06 不用」（寫 `results/test06/player/not_used.json`，最終照 U5 交 Q04）；4 本機漂移（停下查，放寬只能寫修訂）。
4. 正式模式的共同前提（順序固定）：
   - `.venv` 是 Python 3.12.13、numpy 2.5.3、scipy 1.18.1（`frozen_config.json` 記的版本；不同是 2）；
   - git 沒有未 commit 的追蹤檔變更（2）；
   - `frozen_config.json` 的 SHA-256（CRLF 折成 LF）是 `5e5c3b82…`，兩份 LDA 等於它記的 SHA，p06 的三支工具和 p02 的相依檔等於它記的 SHA（2）；
   - K6' 未過。過了：K6' 前已 commit 過一次 06（`off/t06_submit_report.json` 的 commit 時間 ≤ K6'）是 2，那一次照常有效、不寫 `not_used.json`；否則是 3；
   - `results/prereg06/player/` 的 8 個檔等於釘住的 SHA，打分紀錄帳正好有 4 列 `prereg06 H6P`：不成立是 2（凍結檔被改，停下查）；
   - H6P 採用＝`score_H6P.json` 的 95% 下界 > 0，而且 `results/prereg06/RESULTS.md` 寫「**採用**」：不成立是 3；
   - T1 是完整的 raw T1（PASS 或 NORMALIZED；FAIL_OPENING 是 3、FAIL_STRUCT 是 2）；
   - test02 競賽一是正式模式、跑完、本機接縫通過、`check_submission` official 通過，報告、top-5 和 dumps 都已 commit（2）；
   - `results/official_scores.json` 已 commit，裡面有 Q04（`results/test02/player/submission_player.csv`）的第一次上傳、狀態 `uploaded`（2）。
5. `feats`、`seam`、`apply`、`submit` 先核對第 4 條的全部前提（含登記的結束碼 3），才看自己的輸出存不存在：「輸出已存在」（2）不會蓋掉 3。
6. `data/local` 只在 `feats --pool seam_val` 和 `seam` 讀兩份棋譜檔 `player_val_candidates.csv`、`player_val_questions.csv`；檔名含標籤字樣的路徑一律拒絕；比對同時看絕對路徑和實體路徑（經由 junction 也擋）。不跑 `tools/score.py`。

### 11.2 第 2.7 節的表：補上和寫死的地方

7. test02 那一池的 `apply_report.json` 是防護第 0 層（選中只用 CNN 的組合，開局沒有被判定）時，和第 3 層一樣：那一池交 test02 的 400 列（逐位元組相同）。理由同 2.7 第 3 列：JO 是開局 token；S2 和 LDA 只在題目、候選隨機切的模擬裡驗證過，而第 0 層代表開局在這一池不可用或沒有被核對。test02 閘門沒過（Q00）時，test02 的防護本來就是第 0 層，結果相同。
8. 兩池都退回 test02 的列（第 3 層、第 0 層、閘門沒過，或 06 閘門沒過，任兩種的組合）：06 不用，結束碼 3，最終 Q04（不產生 06 的交件檔）。`apply` 判斷「另一池也退回」時，只採用以現在這份 test02 那一池的 `apply_report.json` 和現在這份 seam 做出的另一池報告；test02 重跑前留下的舊報告不算，交給 `submit` 判斷。
9. 各池的 R02＝test02 那一池 `apply_report.json` 的 `applied`（防護改過之後的組合）。sr(R02) 用 p02 的函式、在 sr06 同一批假題上算（第 2.3 節最後一點）。06 閘門：sr(選中) − sr(R02) ≥ 0 算通過（＝0 也通過），< 0 那一池交 test02 的列。
10. 第 2 層：照常用 06（權重不變，O_raw 的 1 和 JO 的 0.5 本來就 ≤ 1），這一池不准開一對一，整份交件檔沒有 06-ON。public 池的強制探測檔照樣寫（它只用在 06-ON；第 2 層時 06-ON 的條件不成立）。

### 11.3 seam（第 5.2 節）

11. `seam` 只讀 test02 自己的 seam_val 判過的 dumps：兩個 dumps 檔的 SHA-256 要等於 test02 `seam_val/apply_report.json` 記的輸入（它也要已 commit），不等是 2。登記的判定不會從 test02 沒判過的輸入得出。
12. 判定（`t06_seam.json`）：選中的組合、兩個閘門的判定（送分閘門 sr(選中) − sr(V0) ≥ +0.005；測試日閘門 sr(選中) − sr(R02) ≥ 0）、偵測器開或關，都要和 H6P 的 `apply_report.json` 相同；和 dumps 無關的數字（每色盤數、不交替的盤數、詞彙外 token 數、兩份 LDA 的 SHA）一律要相同。
13. test02 的 seam_val dumps 和 prereg02 的逐陣列相同（`np.array_equal`，同 dtype、同形狀）時：`top5_off.csv`、`top5_on.csv` 要和 `player_val_top5.csv`、`player_val_top5_switch.csv`（CRLF 折成 LF）逐位元組相同，另外 sr06（取到小數第 6 位和 H6P 記的比）、兩個閘門的值、r_forced_cost 要完全相同，Q04 用 p02 的函式和 06 的路徑各重算一次都要和 `results/prereg02/player/player_val_top5.csv` 逐位元組相同，sr06 要等於 `p06_player.sr06_apply` 算的。這時任何一條不成立是結束碼 4（本機漂移）。
14. dumps 不同時：兩份 Top-5 各至少 398/400 題相同（第 5.2 節）。不成立是結束碼 3（06 不用）；和 dumps 無關的數字不同仍是 4。

### 11.4 官方池的 06-ON（第 5.3 節）

15. 偵測器：在那一池，用選中組合做 CSLS 之前的 F，假題池是那一池 sr06 的第 0、1 輪，門檻是 `frozen_config.json` 裡那個組合的 `r*_X`；打開＝那個組合「能用」、`switch_used_at_all` 為真、r_forced_cost ≤ `r*_X`。題數多於候選人數時沒有一對一指派，偵測器不適用（關），也不寫強制探測檔。
16. `top5_on.csv`：public 池只要用了 06 就一律寫（強制探測，用 public 選中組合的 F）；private 池只在第 1 層、選中組合能用、偵測器打開時寫。06-ON 的交件檔＝public 的 `top5_on` 接 private 的 `top5_on`，兩份都要有。
17. 06-ON 的條件（`t06_submit_report.json` 的 `on_conditions`，全部為真才產生）：(a) private 選中的組合能用、偵測器打開、而且在第 1 層；(b) `score_H6P_switch.json` 的點估計 ≥ 0；(c) 兩池都是 400 題、400 位候選（06 自己數的和 test02 記的都要是）；(d) `results/test02/checks/player/t2.json` 存在，兩份題目檔 T2(c) 的複製數都是 0，(a)(b)(d) 沒有任何一個單元是 FAIL，四個單元都在；(e) 兩池都用了 06、都在第 1 層；(f) Claude 核對過 AIdea 競賽一的說明和資料頁（`--aidea-page-checked "<網址和原文>"`，記進報告）；頁面寫了同一位玩家可以出多題、或題目和候選不是一一對應（不論用什麼字），或 Claude 沒把握時，加 `--aidea-not-one-to-one`，06-ON 不產生；(g) 現在 ≤ K7'。DATA.md 第 5 條（題數、人數、題目和候選有沒有同一盤棋）照常記錄。
18. `inputs_bound`：06-OFF 的報告只綁 06-OFF 的輸入（test02 的交件檔和報告、每池的 `apply_report.json`／選擇／top-5、三組 dumps、T1、`frozen_config.json`、兩份 LDA、H6P 的兩份分數檔、06 的 seam、三組 `feats_report.json`、兩池的 `t06_pool.json` 和 `top5_off.csv`），不綁 T2 和 `top5_on.csv`；06-ON 的報告另外綁這兩樣。上傳紀錄裡已經有這一次 06（同一個 `run_id`）的任何一筆（`prepared` 或 `uploaded`）之後，`submit --rerun` 一律拒絕（2）：重寫的報告會讓那一筆不再符合 5.4。
19. 同名檔警告（第 5.3 節；「完全相同」＝兩個 public 分數差 < 1e-9；第 1 名不同的題數取 `t06_submit_report.json` 記的 public 欄）。第 5.3 節的「核對出傳錯：照 TESTDAY 重傳正確的那份……重記 public 分數再判」，工具只能在 06-ON 上傳之前做到：06-ON 一季只能傳一次（5.5 E1），而 06-ON 之後才重傳的 06-OFF 會讓 06-ON 不再是最後一次上傳。所以寫死如下（比第 5.3 節嚴，只會把開關關掉）：
    - **06-ON 之前**（Q04 和 06-OFF 的分數完全相同、第 1 名不同題數 > 0）：`official_log.py prepare` 06-ON 要求 Q04 和同一次執行最新一筆 06-OFF 的 public 分數都已記錄，而且不是「完全相同、而那一份報告記的 public 池第 1 名不同題數 > 0」；不成立就拒絕。請使用者截 AIdea 的上傳紀錄核對：06-OFF 那一次傳錯了 → 照 TESTDAY 再上傳一次 06-OFF（新的 OK），記 public 分數；06-ON 和這一筆最新的 06-OFF 配對（這就是第 5.3 節的「重傳正確的那份再判」）。核對出沒傳錯（分數真的相同）或核對不出：這一季不做 06-ON，最終 06-OFF。
    - **06-ON 之後**（Q04、06-OFF、06-ON 任兩份）：`veto` 結束碼 2、`veto.json` 寫 `paused: true`、`outcome: paused` 和要核對的項目。暫停時規則什麼都不指定：`official_log.py` 的指定是「還沒定」，`final` 一律拒絕，`status` 不要求重傳；K7' 過了還沒判定才是 06-OFF（第 4 節）。使用者截上傳紀錄後，Claude 只用下面三種之一重跑 `veto --rerun --reason`：核對出沒傳錯 → `--confirmed-upload-check "<說明>"`，照 d 判；核對出 06-OFF／06-ON 這一對有傳錯（包括 06-ON 那一次其實傳成 Q04 或 06-OFF 的檔）→ `--wrong-upload "<說明>"`，`outcome: wrong_upload`；核對不出或 K7' 前來不及 → `--upload-check-inconclusive "<原因>"`，`outcome: upload_check_inconclusive`。後兩種：不用開關，最終 06-OFF，再上傳一次 06-OFF。一次只准給一種。說明都寫進 `veto.json`。
20. 否決只讀已 commit 的 `results/official_scores.json`：06-ON 恰好一筆 `uploaded`；它和「同一次 06 執行（`run_id` 相同）、比它早上傳的最新一筆 06-OFF」配對（`official_log.py` 用同一條規則）；沒有這樣的 06-OFF 是 2；兩筆都有 public 分數；兩份報告等於 `prepare` 時記的 SHA。d＝public(06-ON) − public(06-OFF)；d < 0 否決（`outcome: vetoed`），d ≥ 0（含 0）不否決（`outcome: kept`）。`veto` 在 K7' 之後才跑：不判 d（`outcome: k7_passed`），最終 06-OFF、要再上傳一次。`veto.json` 要在 K7' 前 commit，06-ON 才能是最終。

### 11.5 `official_log.py`（第 5.2 節最後一段、第 5.4 節）

21. 06 的檔＝`results/test06/player/off/` 或 `on/` 裡的 `submission_player.csv`，報告叫 `t06_submit_report.json`。方法名稱：「prereg06 06-OFF: Q04 + v2 style + CSLS (甲)」、「prereg06 06-ON: 06-OFF + one-to-one switch (甲)」。
22. `prepare`：06 的檔要 Q04 的第一次上傳已 `uploaded`；06-ON 另外要同一次執行的 06-OFF 已 `uploaded`、第 19 條第一點的分數條件、log 裡沒有已上傳的 06-ON、同一個 06-ON 檔沒有還沒上傳的紀錄、K7' 未過。報告、06 的 seam、兩池報告、test02 的報告和 T1 都要已 commit 並通過；`inputs_bound` 的每個檔都已 commit 而且 SHA 沒變；報告最後一次 commit 在 K6' 前。Q04 的檔在有符合 5.4 的 06 上傳時，要帶合格的 `--not-06-reason`。
23. `public`（只印，不判定）：同名檔警告（Q04、配對的 06-OFF、06-ON 任兩份）；一筆 06 上傳的 SHA 之前傳過、public 分數卻不同（差 ≥ 1e-9；同一份檔分數一定相同，所以其中一次傳的不是這一份）時另印警告；沒有警告時，只在這一筆是 06-ON 或和它配對的 06-OFF 時印 d 和否決的判定（正式判定是 `veto.json`）。
24. 規則指定的最終選擇（`final`、`status` 每次重算）：
    - 沒有符合 5.4 的 06-OFF 上傳 → Q04；有、沒有 06-ON → 06-OFF；06-ON 不符合 → 06-OFF。
    - 06-ON 符合、`veto.json` 已 commit、配對的是第 20 條那一筆 06-OFF、d 和 log 的分數對得上：`kept`（d ≥ 0）、`veto.json` 在 K7' 前 commit，**而且 06-ON 仍是競賽一最後一次上傳** → 06-ON；06-ON 已經不是最後一次上傳 → 06-OFF（比第 5.3 節嚴：06-ON 不能再傳，E1）。`vetoed`、`wrong_upload`、`upload_check_inconclusive`、`k7_passed` → 06-OFF；`wrong_upload`、`upload_check_inconclusive` 的 `veto.json` 寫的不是 06-OFF 時不算數。
    - 06-ON 已上傳、K7' 未過，還沒有 `veto.json` 或否決暫停 → 還沒定，`final` 拒絕；K7' 已過還沒判定 → 06-OFF。
    - 改選其他的，只能用 `--not-06-reason "PREREG_06 rev N: …"`，N ≥ 4、修訂那一列已 commit、而且加入的時間晚於被推翻的 06 紀錄（核對方法同 PREREG_04 第 11.3 節第 20 條）；只寫文字一律拒絕。
25. log 裡沒有 06 紀錄時，`prepare`、`uploaded`、`public`、`final`、`status` 的輸出、紀錄檔和回傳值都和 tag `p04-testday-pre06` 的版本逐位元組相同（`tests/test_t06_official_log.py` TagVersusNew：競賽二 R20、R10、04 三種流程，各接競賽一 Q04 的流程）。

### 11.6 test02 重跑、K6'、防誤改閘門、演練

26. test02 競賽一在 06 的任一步之後重跑（修 bug、notebook 重跑，照 TESTDAY.md）：06 的輸入變了。`seam`、`apply`、`submit` 會拒絕；已上傳的 06 不再符合 5.4（輸入變了），規則指定的回到 Q04。要用 06 必須從 `feats` 重做（三池 `feats`、`seam`、兩池 `apply`、`submit` 都加 `--rerun --reason`），新的 06-OFF 重新上傳。因為 test02 的狀態得到的結束碼 3（兩池退回 test02、seam 的 dumps 不同）也照這一條重做；除此之外結束碼 3 之後不重跑。06-ON 一季仍最多一次，只和同一次執行的 06-OFF 配對。照 PREREG_04 第 11.3 節第 20 條的前例。
27. K6' 的「一次成功的正式執行」＝三池 `feats`、`seam`、兩池 `apply`、`submit`（至少 06-OFF）都在正式模式以結束碼 0 跑完、報告已 commit，而且 `t06_submit_report.json` 最後一次 commit 在 K6' 之前（`official_log` 核對的就是這個時間）。K6' 之後 `feats`、`seam`、`apply`、`submit` 不再執行（第 4 條）；`veto`、`status`、`official_log.py` 照常。
28. 防誤改閘門：`C10_T04_GATE=1 PY -m pytest -q -rs tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py tests/test_p06_hygiene.py tests/test_p06_player_hygiene.py tests/test_t06_hygiene.py`，加上 `PY -m pytest -q tests/test_t06_official_log.py -k TagVersusNew`。
    - 永遠在的已知失敗：`test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`，原因是 H6P 的 4 列（PREREG_04 rev 3 第 19 條「只多了打分紀錄帳的列」）；`test_p06_player_hygiene.py` 把這 4 列和 `results/prereg06/player/` 的 8 個檔釘死。
    - 其他登記依規產生的失敗，照 PREREG_04 rev 3 第 19 條和 `docs/TESTDAY_04.md`「R20 側的已登記變動」的寫法允許：失敗的只限 `test_ledger_and_prereg_rows_pinned`、`test_p06_player_hygiene.py::LedgerRows::test_later_rows_only_prereg07`（H6P 之後不是 PREREG_07 的列；這一項只在 `C10_T04_GATE=1` 或 `C10_K5=1` 時跑，所以一般的 `PY -m pytest -q tests` 不會因別的登記依規打分而變紅）、`test_p04_hygiene.py::R20TestDayPath::test_pinned_files_unchanged`（和 `test_t02_hygiene.py` 的 `Pins`）、`test_p06_hygiene.py::P04TestDayPath::test_pinned_files_unchanged`，而且多出來的每個檔、每一列都對得上一份已 commit 的登記（PREREG_02 rev 7 第 2 條競賽一改 A100、rev 9 起、D5；PREREG_04 的修訂；`docs/TESTDAY_04.md` 彩排那一節要求寫進 `results/test04/design/build_review.md` 的第二次彩排紀錄；PREREG_07 等登記的打分）。逐項記進 `results/test06/RESULTS.md`。對得上時 06 照常；競賽一改 A100 時 seam 多半走第 14 條。其他任何失敗：停下查。
29. `docs/TESTDAY.md` 的就緒清單要求 `PY -m pytest -q tests` 全綠（單一行程）。`tools/p06_player.py`（經 `p06_style.install_guard`）和探索的 `c1h_lib.py` 在 import 時就在整個行程裝 `data/local` 防護、不還原；這兩支凍結的程式不改。`tests/conftest.py` 在每個測試檔收集後還原原本的函式，只在一個測試檔自己 import 得到這兩支程式時，替它重新呼叫工具自己的 `install_guard()`，那個檔跑完就還原：和「每個測試檔各自一個行程」的行為相同。
30. 演練（spec 9.3）在 K5' 前做完，紀錄在 `results/test06/design/build_review.md`：Q04 逐位元組不變（`q04_fake_golden.json`）、R20 和 04 不變、06 對得回 H6P、官方形狀（一題對一人會產生 06-ON；「同一人出兩題」和 T2(b) 不過都不會）、負面演練，以及 10/3 三個角度的審查和回應。演練樹的 T2 是 `t06_fake_tests.py fake-t2` 寫的替身（`fake: true`，正式模式拒絕它），因為訓練玩家的棋在真正的 T2(a) 一定會判 FAIL。
