# prereg39 PLAN：競賽一新題加「KataGo P1 的 LDA 項」（先於任何 P1 抽取與分數）

日期 2026-11-11 晚（v2-2 上傳後、prereg37 訓練中）。這份和名單統計（`lists_record.json`，只有盤數，沒有任何分數）一起 commit，
早於 prereg39 的任何 P1 抽取、任何分數。分支 `p39-p1`。只做競賽一；不碰驗證集答案資料夾（不開、不讀、不列、不雜湊），
不讀 `data/prereg10/val`（驗證玩家的 P1），不跑 `tools/score.py`，不用打分機會。

## 0. 白話版

現在的新題系統（M36，或 prereg37 過了之後的 M37）只看盤面，沒有「這一手好不好」的資訊。KataGo 的 HumanSL 中性探針（P1）
在 PREREG_10 的 200 人先導裡，每題 2 盤時疊在 06 上 +0.118 [+0.098, +0.137]（`results/prereg10/explore/REPORT.md`）。
這一輪把同一套 P1 抽取用在新題形狀：黑白都抽，在訓練玩家上學 LDA 投影，當成一個新的分項加進融合。

## 1. 特徵（不改任何既有程式）

- 照 PREREG_10 第 2.1 節：`results/prereg08/p1/p1_lib.py`、`p1_extract.run_chunk`（import、不改）；KataGo 程式與
  HumanSL 權重 `data/katago/b18c384nbt-humanv0.ckpt` 同 PREREG_08 第 1.2 節。每盤取目標方（side 0）的 770 維平均；
  **目標方＝該玩家在那一盤的執色，黑白都抽**。失敗的盤、沒有目標手的盤不用。
- 只用競賽一自己抽的 P1。**不用競賽二的任何 P1／P2／P3／HS／S25 特徵或輸出**；共用的只有抽取程式（報告揭露）。
- **重用 `data/prereg10/mock`**（競賽一自己在 10 月抽的，926 位 fit 玩家的黑棋，616 塊都在）：名單裡的棋若是同一個特徵庫列
  （同玩家、同 game_id、黑棋）就直接用那一塊的 side 0，不重抽。查核結果：L 1,742 盤、D 5,148 盤、E 11,719 盤、E2 8,284 盤
  可重用（共 26,893 盤）；白棋和不在 mock 的棋都要新抽。載入時檢查：mock 塊與新塊的 `format`、`meta_sha256`、
  `ckpt_bytes`、`precision`、`code_sha256` 必須相同，不同就拒絕（那時改成全部重抽，寫修訂）。

## 2. 要抽的棋（`p39_lists.py build`，確定性的 hash 規則；統計在 `lists_record.json`）

| 名單 | 內容 | 盤數 | 新抽 | 估計 4090 小時 |
|---|---|---|---|---|
| L（學 LDA） | prereg36 切分的 T 玩家（不含 E、D）：每級依 `sha256("p39|L|pid")` 取前 120 人（共 1,200 人）；每人的棋（排除快取標記 bad 的）依 `sha256("p39|L|pid|game_id")` 排序，黑白各取前 15 盤，不足 30 盤就依同一順序補到 30（不夠就全取） | 34,630 | 32,888 | 2.00 |
| D | D 池種子 100–109 用到的全部棋（`data/prereg36/terms_D` 的 cand_rows＋q_flat 聯集） | 28,212 | 23,064 | 1.41 |
| pub2／priv2 | 官方新題兩池的候選與題目全部棋（`p36_lib.pool_items` 的順序） | 7,100＋7,100 | 14,200 | 0.86 |
| E | E 池種子 0–9 用到的全部棋 | 56,934 | 45,215 | 2.76 |
| E2 | E 池種子 10–19 多出來的棋 | 35,769 | 27,485 | 1.67 |
| 合計 | | 169,745 | 142,852 | **8.70** |

- 時間估計用 PREREG_10 先導的實測（每盤約 222 個局面、約 1,012 局面／秒）；是估計，實際以抽取紀錄為準。
- **E 選 0–19 全部 20 池**：合計 8.70 小時 ≤ 10 小時的上限，判定維持和 prereg36／37 相同的 20 池。
- 抽取順序固定：L → D → pub2 → priv2 → E → E2。
- **預先寫好的退路**：若到 **11/14 23:59** E2 還沒抽完，就只用 E（種子 0–9 的 10 池）判定，條件改成 10 池至少 9 池；
  這個選擇只看「E2 抽完了沒」，不看任何分數。
- 檢查（`p39_lists.py` 內建，不過就停）：L 的玩家都在 T、不含任何 E／D 玩家；D、E 玩家互斥；同一盤不在兩份名單。

## 3. 分項 T_P1

- LDA（`tools/p10_fuse.fit_p1_lda`：至少 2 盤可用的玩家、標準化後 `p06_style.lda_fit`，64 維）只學在 **L**，類別＝玩家。
  兩種變體在 D 池上選：
  - `joint`：黑白合在一起學一個投影；每位候選／每題取全部可用盤投影後的平均 → `p06_style.lda_term`（負的平方距離）。
  - `split`：黑、白各學一個投影；每個執色各算一個 T（題目該色的盤對候選該色的盤），各自對候選做 z 分數後，
    依題目該色的盤數加權平均；題目沒有某色的盤就不算那一色。
- 每題對全部候選做 z 分數（`p06_style.zscore_rows`）；題目一盤可用的都沒有 → 該列 z 記 0。
  候選沒有可用盤 → 該欄填該列其他候選的最小值（同 `p10_fuse.t_p1`）。

## 4. 融合與挑選（D 池，10 池，種子 100–109）

- 基底＝**判定當時已採用的系統**：M36（`results/prereg36/explore/tune.json`：step_40000、wE 4、CSLS 20），
  或 prereg37 採用後的 M37（它的 tune.json）。基底在做 CSLS 之前的融合分數 F_base 不動。
- **M39**＝CSLS_k(F_base ＋ wP·z(T_P1))。格點：變體 ∈ {joint, split} × wP ∈ {0.25, 0.5, 1, 2, 4, 8} × k ∈ {10, 20, 40}，
  取 D 池平均最高的（格點順序的第一個最大值）。`tune.json`（記明基底是哪一個）先 commit 才判定。
- 接縫：wP＝0、k＝基底的 k 時，D 池平均要和基底 tune.json 的 `M.tune_mean` 逐位元相同，否則停。

## 5. 判定（主 session 做；E 池，只判一次）

- 接縫：重算基底在 E 池的每池分數，要和它的 eval.json 逐位元相同。
- **採用 M39 的條件（全部成立）**：mean(M39 − 基底) ≥ **+0.01**；池 bootstrap（10,000 次、種子 39）95% 下界 > 0；
  至少 90% 的池 M39 較高（20 池至少 18 池；走第 2 節退路時 10 池至少 9 池）。
- 沒過：維持基底，只記錄。只記錄的還有：依盤數 1／2／3／5 的分數、只用 T_P1 的分數。
- 揭露：E 池在 prereg36（第 1 次）、prereg37（第 2 次）用過；prereg38 若先判定，這次是第 4 次，否則第 3 次——
  判定時把實際是第幾次寫進 `eval.json` 和 RESULTS。E 沒參與任何挑選（存檔、權重、變體都只在 D 上挑）。

## 6. 過了之後

官方兩池用同一條程式路徑套用 M39（pub2／priv2 的 P1 已在名單裡）→ 組 1,600 列（原題 800 列沿用 v1 06-ON）→ 格式檢查
→ 上傳，記在 `results/official_scores_v2.json`。公開分數不用來選版本。

## 7. 揭露（報告要寫）

- 外部資源：KataGo 程式與 HumanSL 權重（來源、版本、授權照 PREREG_08／PREREG_10 已寫的）；用法＝凍結的特徵抽取器，
  不微調。報告附消融（有／沒有 T_P1）。交出去的程式要能在 Google Colab 跑：KataGo 抽取的 Colab 路徑在 `colab_gpu/`。
- 抽取程式和競賽二共用（同一作者的前處理），特徵值不共用。

## 8. 資源與中斷

- 4090、一次一個 GPU 工作（排在 prereg37 的訓練和向量之後）。開跑前 `nvidia-smi`：驅動 616.92、沒有別的計算程序
  （`p39_extract.py` 自己也會檢查，有就不啟動）。向戰情室報開始和結束。
- 每 200 盤一塊（`data/prereg39/p1/<名單>/chunk_*.npz`），已完成的塊跳過；建 `data/prereg39/p1/STOP` 會在下一塊之前停。
  接續：刪 STOP 後重跑同一行指令。
- 硬截止：11/16 23:59 前判定，否則不用。

## 修訂 1（2026-11-11，抽取前、沒有任何分數）

- 官方新題檔每盤開頭是 `SZ[19]HD[0|1]`（兩個候選檔實數：HD[0] 9,173 盤、HD[1] 2,827 盤）；KataGo 的解析遇到 HD 會整盤失敗
  （PREREG_22 記過）。所以 `p39_extract.py` 對官方棋先用 `tools/root_props.classify_game`（PREREG_22 的同一個函式，import、不改）
  拿掉開頭訓練集沒有的欄位，再抽 P1；訓練棋（L、D、E、E2）本來就沒有 HD，不處理。HD 的值不當特徵用。
- 每個名單的失敗盤數記在抽取紀錄；官方任一池失敗盤 > 0.5% 就停下查，不套用。

## 修訂 2（2026-11-12，任何 P1 抽取、任何 prereg39 分數之前；基底改成 prereg38 後端）

這份修訂先 commit，之後才寫 `--base p38` 的程式。到 commit 為止，prereg39 沒有抽過任何 P1、沒有算過任何分數。

- **背景**：prereg38 的計分後端 11/11 採用並上傳 v2-3（E 池 0.4970 對 M36 0.4241）。第 4、5 節的「基底」從 M36 改成
  **已採用、已上傳的「M36＋prereg38 後端」**。p36／p37 基底的程式留著，但這一輪的挑選與判定不用（M36 單獨已被 v2-3 取代）。
- **基底的三個檔（預設）**：設定 `results/prereg38/explore/tune_D_B4.json`（`recommended.cfg`：每盤先 L2、候選庫類內白化
  λ 0.02＋LDA 32、G 各項＋wE 4、CSLS 10；存檔 step_40000，sha256 a2c8e03a…）；判定 `results/prereg38/explore/eval.json`
  （`adopt` true，它記的 `tune_sha256` 要等於設定檔的 LF 雜湊）；向量 `data/prereg36/emb/step_40000/`。
  F_base＝prereg38 的 `fuse_cfg(pool, e_term(pool, emb, cfg), cfg)`，也就是後端在 CSLS 之前的分數（import prereg38 的程式，不改）。
- **融合方式（選定）**：P1 項是後端融合分數上的**另一個加項**：M39＝CSLS_k(F_base＋wP·z(T_P1))，**不放進 CNN 向量的白化區塊**。理由：
  1. wP＝0 時程式上就是 F_base，逐位元回到已採用的系統，接縫乾淨、消融（有／沒有 T_P1）清楚；
  2. P1（770 維 KataGo 探針平均）和 CNN 向量的來源、尺度、維度都不同，串在一起白化會改掉已經在 E 判定過的 CNN 項，等於重做 prereg38 的結構；
  3. 和 PREREG_10 的 B′、原 PLAN 第 4 節同一種形式（在 CSLS 之前加權相加）。
- **P1 項本身多一個結構選項 pw（在 D 上選）**：
  - `none`：第 3 節原樣（L 學的 LDA 64 維 → 群組平均 → 負平方距離）。
  - `b4`：L 學的 LDA 64 維投影之後，照 B4 的做法在**該池內**處理：每盤先 L2 → 用該池候選的棋（標籤只有候選 id）估類內共變異，
    白化 λ 0.02 → 候選間共變異的前 32 個方向 → 候選／題目原型（L2 正規化平均）的 cosine。用 prereg38 的 `transform`、`proto`
    （import，不改）；λ 0.02、32 照 prereg38 選定值固定，不再調。`split` 變體每個執色各做一次，再照第 3 節依題目該色盤數加權。
  - 理由：prereg38 的 D 消融裡，候選庫白化是對 CNN 向量唯一有大增益的模組；P1 投影有沒有同樣的效果事先不知道，所以兩種都放進 D 格點，由 D 決定。
  - 兩種 pw 都一樣：題目沒有可用的 P1 盤 → 該列 z 記 0；候選沒有 → 該欄填該列其他候選的最小值。
- **D 格點**：變體 {joint, split} × pw {none, b4} × wP {0.25, 0.5, 1, 2, 4, 8} × CSLS k {10, 20, 40}＝72 組，
  取 D 池（種子 100–109）平均最高的（格點順序變體 → pw → wP → k 的第一個最大值）。`tune.json` 記明基底三個檔的路徑與雜湊，先 commit 才判定。
- **D 接縫**：選之前先重算基底在 10 個 D 池的每池分數，要和 `tune_D_B4.json` 的 `recommended.per_pool` **逐位元相同**
  （平均 0.5805158972862898），否則停。
- **E 判定（主 session，只判一次）**：對照的是 v2-3 那版（＝基底）。接縫：重算基底在 E 池的每池分數，要和 `eval.json` 的
  `per_pool.backend` 逐位元相同，平均＝**0.4969569920097353**，否則停。採用條件不變：mean(M39 − 基底) ≥ +0.01；池 bootstrap
  （10,000 次、種子 39）95% 下界 > 0；20 池至少 18 池較高；E2 沒抽完、走第 2 節退路時用 10 池（種子 0–9），至少 9 池較高。
- **E 池第幾次使用**：prereg36 第 1 次、prereg38 第 2 次（prereg38 PLAN 修訂 6）。prereg37 的判定若在這次之前，它是第 3 次、這次是
  **第 4 次**；若這次先判，這次是**第 3 次**。`judge --e-use N` 照實際順序填，寫進 `eval.json` 和 RESULTS（第 5 節「prereg38 若先判定…」那句以這裡為準）。
  E 不參與任何挑選。
- **若 prereg37 先採用 M37**：基底換成「M37＋後端」＝prereg38 `retune` 產生的設定檔＋它對 v2-3 的那次判定（format `c10-prereg38 eval v1`、
  `adopt` true）＋`data/prereg37/emb/<選定存檔>/`。select 用 `--base p38 --base-tune … --base-eval … --base-emb-dir …`；
  D 接縫值換成那份設定檔的 `recommended.per_pool`，E 接縫值換成那份判定的 `per_pool.backend`／`means.backend`。M37 沒採用就用上面的預設。
  judge／apply 只照 `tune.json` 記的基底讀，不接受另外指定。
- **官方池**：基底走 prereg38 `apply` 同一條路徑（同一 cfg、`pool_<pool>.npz`）。用預設基底時多一道檢查：基底（不加 P1）的 top5
  要和 `results/prereg38/apply/<pool>/top5.csv`（v2-3 的新題部分）逐位元組相同，否則停。原題 800 列照舊沿用 v1 的 06-ON。
- 這一輪（修訂 2 的施工）只寫程式和測試：相容性測試只用 `data/prereg10/mock` 的黑棋（形狀、z 列、接縫、格式檢查），
  **不印、不記任何含 P1 的 D 池分數**；不載入 E 池；不用 GPU。

## 修訂 3（2026-11-12 08:4x，P1 抽取進行中、任何 prereg39 分數之前）：基底＝v2-4

- prereg37 11/12 採用並上傳 v2-4（M37＋prereg38 後端在 D 重調）。照修訂 2「若 prereg37 先採用 M37」那段，基底換成：
  `--base p38 --base-tune results/prereg37/explore/tune_D_B4_M37.json --base-eval results/prereg37/explore/eval_backend.json
  --base-emb-dir data/prereg37/emb/step_160000`。
- 判定檔格式是 `c10-prereg37 backend eval v1`（不是修訂 2 寫的 `c10-prereg38 eval v1`），但程式只看 `adopt`、`tune_sha256`
  （等於設定檔的 LF 雜湊）和 `per_pool.backend`，三個都符合；本分支已合併 main（`826576e`）才讀得到這些檔。
  載入檢查（不算分數）：基底 k＝20；D 接縫值＝`tune_D_B4_M37.json` 的 `recommended.per_pool`（平均 0.6655350985215651）；
  E 接縫值＝`eval_backend.json` 的 `per_pool.backend`（平均 **0.5856450890595781**）。
- 官方池檢查：修訂 2 的「基底 top5 要和 prereg38 apply 逐位元組相同」只在預設基底時做；換成 v2-4 後改成要和
  `results/prereg37/apply/<pool>/top5.csv` 逐位元組相同（程式若只支援預設那一條，判定後、apply 前補上，先 commit 再跑）。
- E 池第幾次使用：prereg36、prereg38、prereg37 之後，這次是**第 4 次**（`--e-use 4`）；prereg43 若先判，順延。
