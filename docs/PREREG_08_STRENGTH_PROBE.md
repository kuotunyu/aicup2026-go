# 事前登記 08：競賽二加上 KataGo HumanSL 中性棋力探針（Q08＝Q02H＋P1TO）

**白話版**：07 用 HumanSL 在 14 個「棋力設定」下各算一次「這手棋有多像那個等級的人」。08 改成只用一個「不告訴網路棋力」的中性設定，每個局面跑一次，直接讀網路內部對這手棋、這個盤面的表示（主幹 384 維），再加上這手的機率和整體的猶豫程度（熵）。目標方、對手方各取全盤平均，每盤得到 1,540 個數字，用一個 logistic regression 翻成 10 個等級的機率（P1TO 項），照 04、07 的同一套校準加進 07。這個翻譯模型只用 fit 玩家的棋學（P1 的 3 萬盤，加上 P2 的 12 萬盤）。P1 試跑只用訓練玩家：inner 上比 07 多 +0.0508（95% 下界 +0.0370）；最像官方題型的單色題全部 120,544 題 +0.0496，黑白都正。本機驗證集只打一次分，這一次就是判定，而且是競賽二整季最後一次。測試日在本機 4090 上多算約 0.6 小時，順序 08 → 07 → 04 → R20 → R10。

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。競賽二的前幾份：
  - PREREG_02：R20，本機驗證集 0.5891。
  - PREREG_04：Q02＝R20＋O4＋SWA＋SWB，0.6236（H4R）。
  - PREREG_07：Q02H＝Q02＋HS，0.6327（H7R +0.0091 [−0.0017, +0.0200]）。照規則不採用，rev 3 由 Claude 偏離規則，定為競賽二首選（`results/prereg07/RESULTS.md`）。
  - 03 沒過閘門；05 收掉；06 只做競賽一。
- 探索（都只用 fit／inner 訓練玩家、沒打分）：
  - `results/prereg08/explore/REPORT.md`（10/5：微調和對手方 HS 暫不做，先做 P1）。
  - `results/prereg08/p1/`（計畫 `p1_plan.json`、程式、`p1_fuse_report.json`、`p1_explore_report.json`）。
  - `results/prereg08/p2/p2_sets.py`。
- **只做競賽二（TEAM_11248）。** 08 的任何輸出都不進競賽一（第 6 節）。
- **交件模型選甲**：直接用本機驗證過的做法。翻譯模型只用 probe（fit 玩家）學，合併參數只用 inner 擬合；不重訓。
- 代號：
  - P1 特徵：每盤每方 770 維，第 1.3 節。
  - P1TO：目標方＋對手方 1,540 維經翻譯模型得到的 10 維對數機率。
  - HS：同 PREREG_07。
  - Q02：04；**Q02H**：07（本登記的對照組）；**Q08**：Q02H＋P1TO（唯一的候選）。
  - probe：學翻譯模型的 fit 玩家棋，P1 部分 30,000 盤、P2 部分 120,000 盤。bank、mono：同 PREREG_07 試跑。
- 本登記寫四件事：比什麼、怎麼選、怎樣算成功、什麼時候放棄。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `543315a` | 第一版，2026-10-06。commit 時的狀態如下。① P1 的計畫和閘門已在結果之前 commit（`4d6a853`，10/6 00:53），P1 已過關（`75b6684`）。② 閘門之後的探索已 commit（`0ca8316`：倍數、學習曲線、重複盤檢查）。③ P2 的 probe 名單已 commit（`a709408`）。④ P2 抽取 10/6 10:12 起在 4090 上跑，**還沒有任何 P2 的特徵被讀進翻譯模型、也沒有任何用 P2 算出的數字**。⑤ 驗證題沒有算過 P1 特徵。⑥ 競賽二的本機驗證集自 H7R（10/5）之後沒有再打過分。 |
| rev 2 | `13a6db9` | 2026-10-06，在 manifest、check 之前：實作和登記的差異（第 12 節）。工具 `1ec64de`（經獨立審查，沒有 blocker／major）。此時仍沒有任何 P2 特徵被讀進翻譯模型，驗證題沒有 P1 特徵。 |
| rev 3 | `5473a35`（補記 `22cc0b9`；合併 `6a5e828`） | 2026-10-09，測試日的修訂（第 13 節），在任何測試資料之前：`tools/t08_rank.py`、`tools/t08_fake_tests.py`、`tools/official_log.py` 只加行、`docs/TESTDAY_08.md`、`tests/test_t08_*.py`；實作和第 5.2 節的差異、容許差（照 5.2 不變，4090 實測待主 session 補）、K5-08 的條件。之後要推翻 08 的登記結果，要 rev 4 起的修訂。 |
| rev 4 | `2c07c54`（合併 `6a5e828`） | 2026-10-09，在任何測試資料之前、rev 3 補記第 24 點的 4090 實測之後：GPU 接縫 lp_P1TO 的容許差 0.02 → 0.04（第 14 節）。標準化特徵的 0.05 不變。之後要推翻 08 的登記結果，要 rev 5 起的修訂。 |

## 0. 為什麼做這個（都不看答案；P1 看過 inner，所以 inner 的數字偏樂觀）

- **07 的限度**：HS 只看「這手在 14 個等級下的機率」，每盤 14 個數字。10/5 的探索（`hs_rich_fitmap_cv.py`）試過從同一批逐手 log p 做更多特徵，都沒有比原本的平均曲線好，所以 HS 的極限在網路的輸出端。
- **P1 的想法**：不讓網路「扮演」某個等級，改讀它內部的表示（主幹在實際落點的 384 維、全盤平均 384 維）和這手的 log p、熵，讓翻譯模型自己學哪些方向和棋力有關。成本是每局面 1 次前向，HS 是 14 次。
- **P1 試跑**（`p1_fuse_report.json`，10/6 08:57 UTC 寫成，`75b6684`）：
  - inner 評估假題 127,200 題，5 折玩家 OOF：Q02H 0.6243 → Q02H＋P1TO **0.6750**，**+0.0508** [+0.0370, +0.0654]；每題平均 +0.0523。
  - 單色假題全部 120,544 題 **+0.0496** [+0.0406, +0.0599]，黑 +0.0559、白 +0.0431。
  - 0 盤失敗。
  - 每折 P1TO 的權重都打到上限 4.0；HS 的權重有 4 折是 0。
- **P1 事前寫死的部分**（`4d6a853`，在 bank、mono 的任何 P1 特徵存在之前）：
  - 首選變體 TO、首選組合 Q02H＋P1TO。
  - 中性 metadata、特徵版面、probe 的抽法。
  - LR 和 λ 格點、以玩家分折的 CV。
  - 閘門 G1 ≥ +0.010、G2 下界 > 0、G4（全部單色題、黑白各 > 0）。
  - 只記錄的組合（T、T＋O、Q02＋P1TO、R20＋P1TO）不能救首選。
- **閘門之後的探索**（`0ca8316`，只記錄，都是看過 P1 結果才做的）：
  - **E1 倍數**：lp_P1TO 乘 s 再合併（等於把 P1TO 的權重上限從 4 放寬到 4s）。×1 +0.0508、×2 +0.0519、×4 **+0.0525**、×8 +0.0525，到 ×4 飽和。×4 的單色 +0.0508 [+0.0414, +0.0614]。
  - **E2 學習曲線**：TO 翻譯模型以玩家分折的 CV log-loss，在 10k／20k／30k 盤時是 1.763／1.709／1.679，還在下降。
  - **重複盤檢查**：probe 和 bank／mono 用落子序列比對，0 盤重複。
- **所以 08 在 P1 之上改了兩件事，都是看過 inner 才定的**（第 8 節揭露）：
  1. 乘 4。
  2. probe 加上 P2 的 12 萬盤（fit 玩家，和 P1 probe 不重疊）。
  - P2 會改變翻譯模型，所以**第 2.5 節的 inner 閘門在本登記 commit 時是未知的**（用最終翻譯模型重判）。
- **預期**：驗證集差值的標準誤估 0.0093–0.0098（第 3 節）。inner 增益打五折時，H8R 採用的機會仍約 8 成（估計）。

## 1. 來源（全部釘死）

### 1.1 沿用 07 的五項和 HS（不重算、不改）

- C5、OP、O4、SWA、SWB：照 PREREG_04 第 1 節、`results/prereg04/rank/frozen_config.json`（`ae728e9c…`）。
- HS：照 PREREG_07。
  - 翻譯模型 `results/prereg07/rank/hs_map.npz`（raw SHA-256 `a4334e79c490e4d8102dd5136ac08c0de1060315981f4ccc527ebffa1b84a3e4`）。
  - inner 的 lp_HS 用試跑的路徑（`p1_fuse.hs_baseline`，同 `tools/p07_rank_fuse.py selfcheck`）。
  - 驗證題的 HS 讀已 commit 的 `results/prereg07/hs/val_hs.npz`（`ef2067e8…`），不重算。
- 07 的凍結參數 `results/prereg07/rank/frozen_config.json`（`93a0ac29615260865d554ee61aec258c633820a87aeb7dfcf8047a5dfd200c0b`）。
- 07 的驗證題機率 `results/prereg07/rank/rank_val_probs.csv`（`6ae2db6025a903eb5c58af9418dc203c0b6e84dd664049f85dfa850e57dc28c1`）。
- inner 輸入照 `u1_lib.load_inputs`（`results/prereg05/explore/u1/u1_lib.py`，`225dd51c…`）。

### 1.2 KataGo 的程式和權重

- 和 PREREG_07 第 1.2 節相同：`third_party/katago/`（commit `d91ea855…`，`SOURCE.txt` `039d5a1c…`）。
- 權重 `data/katago/b18c384nbt-humanv0.ckpt`：323,752,318 bytes，SHA-256 `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27`，用 `model`。
- P1、P2 的塊記的是 `not hashed in this run`，所以第 9 節第 3 步的 `manifest` 完整重算一次。之後每一步都完整重算。
- 釘住的檔，只 import、不複製、不改；SHA-256 都是 CRLF 折成 LF：
  - `hs_lib.py` `d19b6a86…`
  - `pilot_run.py` `b0d127fd…`
  - `pilot_fuse.py` `37fb2dff…`
  - `u1_lib.py` `225dd51c…`
  - `results/prereg08/p1/p1_lib.py` `062600168326f9eecd4ec1073e54881c475f8f276e376164eab20d373bbff511`
  - `p1_extract.py` `2ee8eb127150e672b62bcb30b70c4fac46b8cd6e32c7e4eb19d54f746a6e3f9c`
  - `p1_fuse.py` `ad356c8ca9f26e1e761677afa07b5aa369ca31e5318f61eb6a6cbda33b970953`
  - `p1_plan.json` `0cd46bf90c764cae3824f5906e72e13336ab7276ef5fa7edb2ec6831911b228d`
  - `results/prereg08/p2/p2_sets.py` `98bad3d85a32e80c065226b0cc8cd1065b423cdb39dce89829a54b0cbf381c8f`

### 1.3 每一個局面怎麼算（寫死，照 `p1_lib.py`）

- **棋譜、重播、規則**：同 PREREG_07 第 1.3 節（只接受黑先、黑白交替的 SGF；KataGo `Board` 從空棋盤重播；日本規則、貼 6.5）。
- 特徵化用 `p1_lib.featurize_game`：和 `hs_lib.featurize_game` 逐位元相同，只把 ladder 搜尋做記憶化。
- **metadata（中性）**：PREREG_07 的 `rank_<r>` 預設，但雙方等級都設成未知（rank-unknown 旗標開、沒有 rank bit）。每局面只做 1 次前向。
- **每個局面讀 770 維**：
  - `model.act_trunkfinal` 輸出在實際落點的 384 維、全盤平均 384 維；
  - 主 policy head 第 0 通道的 log p(這一手)（非法點先遮掉再 log-softmax）；
  - 同一個 policy 的熵。
- **每盤**：目標方、對手方各自對自己的手取平均，得到 `means [2, 770]`，另記 `counts [2]`、`n_moves`、`failed`。目標方＝題目的執色。
- **精度**：GPU fp16 autocast、channels_last、每批 1,024 個局面、每批補齊到 1,024 列、`cudnn.benchmark` 開。不能逐位元重現，容許差見第 5.2 節。

### 1.4 probe（學翻譯模型的棋，只有 fit 玩家）

- **P1 probe**：`data/prereg08/p1/sets.json`（SHA-256 `f0c494962788615d76080b5481de5b762d71f3ad37f8add55a09efef82acb7c7`）。
  - 30,000 盤，每級 3,000 盤，5,767 人，每人最多 10 盤，seed 20261006。
  - 塊在 `data/prereg08/p1/probe/`（150 塊），0 盤失敗。
- **P2 probe**：`data/prereg08/p2/sets.json`（SHA-256 `2b68f7ae27b2b6bbe00a3920065495d2a5d6b73fd7df33854f4bf99ae5390049`）。
  - 120,000 盤，每級 12,000 盤，5,740 人，每人最多 45 盤，seed 20261007，P1 probe 的盤一律排除。
  - 用沒有改過的 `p1_extract.py run --set probe --sets data/prereg08/p2/sets.json --out-dir data/prereg08/p2` 抽取，共 600 塊。
- 兩者都只取 `c10data.load_rows(part="fit")`，沒有 inner 玩家、沒有驗證玩家（split v1）。
- **P2 用不用（固定規則，沒有選擇）**：
  - `manifest`（第 9 節第 3 步）只准在 P2 的 600 塊都寫完之後、或 **10/8 23:59 之後**跑。
  - P2 用上的條件：`manifest` 那一刻 600／600 塊都在、通過塊檢查，而且失敗盤 ≤ 600（0.5%）。
  - 否則只用 P1 probe，`frozen_config.json` 記 `probe_used: "P1"` 和原因。
  - 這樣定的理由：凍結截止是 10/9 23:59，tune 要 2–4 小時 CPU，所以留一天。
- **重複盤**：`manifest` 用落子序列（原方向）比對 probe（P1＋P2）和 bank／mono。重複的 probe 盤一律不進翻譯模型（固定規則），盤數記錄。P1 的部分 10/6 查過是 0。
- bank、mono 的 P1 特徵沿用 `data/prereg08/p1/bank`（60 塊）、`mono`（307 塊），不重算。

### 1.5 翻譯模型（P1TO 項）

1. 特徵：目標方 770 維接對手方 770 維，共 1,540 維（`p1_fuse.variant_X(…, "TO")`）。
   - 失敗或沒有目標手的盤不用。
   - 對手方沒有手時，那 770 維設成 probe 平均（標準化後是 0）。
2. 標準化：用 probe 可用盤的平均和標準差（nan 不算；標準差是 0 時用 1）。
3. 模型：10 類 multinomial LR，損失＝平均負對數似然＋λ‖W‖²（`pilot_fuse.mlr_obj`），L-BFGS-B，maxiter 2,000。
   - 用 `p1_fuse.fit_mapping_grouped` 原樣。
4. λ 從 `p1_plan.json` 的格點 {1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001} 選。
   - 選法：以玩家（`等級:player_id`）分 5 折的 CV log-loss 最小者；折號＝玩家排序後 `default_rng(0).permutation`、位置 % 5；由大到小 warm start。
   - 再用全部 probe 可用盤重擬合。
   - 選到格點端點只記錄，不改格點。
5. lp_P1TO(g)＝log_softmax(W·z(x_g)＋b)。建 lp_P1TO 不用任何 inner 或驗證玩家的標籤。
6. **乘 4**：合併時用 4·lp_P1TO（第 2.1 節）。
7. 凍結時把 μ、σ、W、b、λ、各 λ 的 CV 紀錄、`probe_used`、盤數和人數存進 `results/prereg08/rank/p1_map.npz`，SHA 寫進 `frozen_config.json`。之後驗證題和測試日都只讀這一份。

### 1.6 驗證題的 P1 特徵（凍結後才算）

- 和 07 的驗證題 HS 是**同一份盤的清單**：
  - 只讀 `data/local/rank_val_questions.csv` 的 `question_id`、`num_games`、`color_i`、`sgf_i`，不讀答案。
  - 槽位要和 `results/prereg02/rank/dumps/val_keys.npz`（`055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c`）逐列相同：1,589 題、19,156 盤，順序同 val_keys。
- **要在 `frozen_config.json`、`p1_map.npz` commit 之後才准算**（工具強制）。讀完題目檔才 import `p1_lib`；它的防護會讓同一個行程之後碰不到 `data/local`。
- 精度、批次、metadata、環境都和 P1 相同。
  - 塊放 `data/prereg08/val/`（不進 git，200 盤一塊，可以續跑；STOP 檔 `data/prereg08/val/STOP`）。
  - 要 commit 的：`results/prereg08/feat/val_p1.npz`（每盤的 counts、n_moves、failed、兩方的 log p 和熵平均）、`val_seam_feats.npz`（前 400 盤的完整 `means`，測試日 GPU 接縫的參照）、`val_report.json`（每塊 SHA）。
- 用獨立的背景程序跑，設 `PYTHONDONTWRITEBYTECODE=1`；腳本只跑這一步。
- 估計 4090 上 1.0–1.2 小時：19,156 盤 × 每盤約 228 個局面，用 P1 實測的 1,050–1,190 positions/s 估。
- 只記錄：驗證題的盤和 probe 的落子序列重複數。驗證玩家和 fit 玩家是分開的，所以預期 0。

## 2. 合併、檢查和閘門（只用 CPU，`tools/p08_rank_fuse.py`）

### 2.1 合併公式（PREREG_04 的廣義校準，多加一項）

p＝softmax( e^a · (n/12)^γ · ( m_C5 ＋ w_OP·m_OP ＋ w_O4·E_O4 ＋ w_SWA·E_SWA ＋ w_SWB·E_SWB ＋ w_HS·E_HS ＋ w_P1·E_P1 ＋ Bβ ) )

- 項的順序：OP、O4、SWA、SWB、HS、P1TO。
- 其他全部照 PREREG_04 第 2.1 節（`p04_rank_fuse.gen_*`）：參數界限（a (−4, 4)、γ (−1, 2)、所有 w (0, 4)，**不改**）、起點、PENALTY、MAXITER、等級加權。
- **E_P1**：這一題每一盤「C5 可解析、P1 沒有失敗、而且有目標手」的 **4·lp_P1TO** 取平均；一盤都沒有就記 0 向量。
  - inner 沒有這種盤時，就是 `pooled(S, 4·lp_P1TO, log)`（selfcheck 核對）。
  - 因為乘 4，w_P1 的上限 4 等於 lp_P1TO 的權重上限 16。
- **E_HS**：照 PREREG_07 第 2.1 節（遮罩平均）。
- inner 用 5 折玩家 OOF（`u1_lib.run_gen`／`p04_rank_fuse.gen_oof_record`，照 `tools/p07_rank_fuse.py run_terms` 的寫法）；單色假題用那位玩家所在的折。規則 B、PlayerBootstrap 照 02／04／07。

### 2.2 check：輸入能不能用

全部成立才往下，任一不成立就停下查：

1. `results/prereg08/feat/inner_manifest.json` 已 commit。
   - 每塊的 SHA、塊數和盤數相符：P1 probe 150／30,000、P2 probe 600／120,000（用 P2 時）、bank 60／11,813、mono 307／61,356。
   - 每塊的 `ref` 等於 sets.json 的那一段。
2. 每塊記的 format（`c10-prereg08 p1 chunk v1`）、sides（target、opponent）、中性 metadata 的 SHA（`p1_lib.neutral_table_sha()`）、精度 fp16、cuda、batch 1,024、channels_last、KataGo commit、`p1_lib.py`／`p1_extract.py` 的 SHA，都等於第 1.2–1.3 節。
3. 權重完整重算的 SHA-256 等於第 1.2 節；`third_party/katago` 每個檔等於 `SOURCE.txt`。
4. bank、mono 的失敗盤和沒有目標手的盤都是 0（P1 實測 0）。probe 的失敗盤記錄，P2 的照第 1.4 節。
5. **手數交叉核對**：bank、mono 每一盤，P1 的 `n_moves` 等於 07 試跑 HS 塊的總手數，也等於 `results/prereg04/opp/inner_o4.npz` 的 `n_moves` 合計；P1 的目標方手數 `counts[:,0]` 等於 HS 的目標手數和 `inner_o4` 的 `n_moves[:,0,:]` 合計。兩邊都沒失敗的盤，不符要是 0。
6. probe 的玩家和 inner、驗證玩家的交集是空的；P2 和 P1 probe 的盤不重疊；重複盤照第 1.4 節。
7. `u1_lib.load_inputs` 的前提照常；07 試跑的 `plan.json` SHA 仍是 `8494e9af…`。

### 2.3 selfcheck：正式工具要重現 P1（只核對移植，不是選擇）

- 相差都要 ≤ 1e-12，不符就停下查。查明是數值路徑的差異、不是邏輯錯誤時，寫 rev 2，以 repo 算出的數字為準並揭露。
- **接縫**：Q02 的 inner OOF（評估和單色）和 `u1_oof.npz` 逐位元相同；Q02H 的 inner 評估分等級平均 0.6242752597994481。
- **只用 P1 probe 的翻譯模型**：
  - λ＝0.01；
  - 該 λ 的 CV log-loss 1.6790734173594255；
  - 重擬合 loss 1.585088334325938。
- **Q02H＋P1TO ×1**：
  - 0.675034627957959，差 0.0507593681585109，CI [0.03701489044315354, 0.06537945687999477]；
  - 單色差 0.04964601127852308，黑 0.055909493750024186、白 0.04309679963025359。
- **Q02H＋P1TO ×4**：
  - 0.6767709161878595，差 0.05249565638841136，CI [0.0386137095020332, 0.06756280199569697]；
  - 單色差 0.05075085423026038。
- 遮罩版 E_P1、E_HS 在 inner 沒有不可用盤時，和 `pooled` 版逐位元相同。

### 2.4 組合（只有一個候選）

| 組合 | 內容 | 用途 |
|---|---|---|
| Q02H | OP、O4、SWA、SWB、HS | 對照（就是 07） |
| Q08 | Q02H＋4·P1TO（最終翻譯模型） | 唯一的候選 |

- 沒有選擇這一步：閘門都過了就是 Q08，沒過就未送分。
- **只記錄、不可選**：
  - Q08 ×1；Q02＋P1TO ×4；R20＋P1TO ×4；
  - 只用 P1 probe 翻譯模型的 Q08（看 P2 的效果）；
  - 各級、各盤數、單色（全體、黑、白）的分數；每折參數和打到界限的權重；
  - 凍結時重擬合的參數和樣本內分數；
  - 翻譯模型各 λ 的 CV 指標；`se_val_record`。

### 2.5 送分閘門（Q08 對 Q02H，全部成立才凍結並打分）

| 閘門 | 條件 | 寫本登記時 |
|---|---|---|
| 接縫 | 第 2.3 節的 Q02、Q02H 兩項 | 已知會過 |
| G1 | inner 評估假題的分等級平均差 ≥ +0.010（容許 1e-12） | 最終翻譯模型下未知（只用 P1 probe 時是 +0.0525） |
| G2 | 同一差值的配對 PlayerBootstrap（每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0 | 未知（只用 P1 probe 時是 +0.0386） |
| G4 | 單色假題（PREREG_03 rev 3 (a) 的同一組：120,544 題，SHA `45d0f14a…`）的分等級平均差：配對 PlayerBootstrap 95% 下界 > 0，黑題、白題的差各自 > 0；在全部題上判 | 未知（只用 P1 probe 的 ×4 是 +0.0508 [+0.0414, +0.0614]） |

- 每題平均差只記錄。
- 閘門只判一次：`tune` 正式跑一次；要 `--rerun`，只准在指出具體程式錯誤的修訂 commit 之後。
- 任一閘門沒過：記為「未送分」，`results/prereg08/RESULTS.md` commit 後結束，不算驗證題的 P1 特徵。競賽二照舊（07 → 04 → R20 → R10）。第 4 次打分機會沒用到，這一季也不再用。

### 2.6 tune：凍結

- 閘門都過了才凍結：用全部 inner 擬合假題重擬合 Q08（`p04_rank_fuse.gen_fit_record`，同 07 的 tune），寫 `results/prereg08/rank/frozen_config.json` 和 `p1_map.npz`，一起 commit。
- `frozen_config.json` 記：
  - 組合、項的順序、倍數 4、參數、5 折紀錄；
  - `p1_map.npz` 的 SHA、`probe_used` 和兩份 sets.json 的 SHA；
  - 07 的 `hs_map.npz` 和 `frozen_config.json` 的 SHA；
  - 釘住檔的 SHA、`inner_manifest.json` 的 SHA、閘門紀錄、`se_val_record`；
  - P1 方向檢查的參照值 ρ_eval、ρ_mono（第 2.7 節）；
  - 工具的 SHA；
  - 兩個環境的版本（`.venv`：Python、numpy、scipy；`.venv-gpu`：Python、torch、cuDNN、GPU 型號；驅動版本只記錄）。
- 從這之後才准算驗證題的 P1 特徵。

### 2.7 apply（驗證題）

- **前提**：
  - `frozen_config.json`、`p1_map.npz`、`val_p1.npz`、`val_seam_feats.npz`、`val_report.json` 已 commit；
  - `val_report.json` 記的開始時間晚於 `frozen_config.json` 最後一次 commit 的時間；
  - 每塊的 SHA 等於 `val_report.json`。
- **先重現 07**：
  - 用 `tools/p07_rank_fuse.py` 的 `q02_validation`、`apply_core`（import，不改）算驗證題的 Q02 和 Q02H。
  - Q02 要和 `results/prereg04/rank/rank_val_probs.csv` 逐位元組相同；Q02H 要和 `results/prereg07/rank/rank_val_probs.csv`（`6ae2db60…`）逐位元組相同。
  - 不同就停下。
- 再照 `frozen_config.json` 算 Q08，寫 `results/prereg08/rank/rank_val_probs.csv`、`val_lp_p1.npz`；接著跑 `decide.py`、`check_submission.py rank … --questions data/local/rank_val_questions.csv`。
- `apply_report.json` 的 `score_allowed` 要以下全部成立才為真：
  - **P1 失敗盤**（`failed` 或整盤沒有手）是 0。
    - 不是 0 時停下查。查到程式錯誤，先寫修訂、修好重算。
    - 確定是資料本身的性質時，照遮罩算，失敗 ≤ 0.5%（≤ 95 盤）、加 `--accept-data-failures` 才往下；> 0.5% 記為「未送分」。
  - Q02、Q02H 逐位元組重現通過。
  - **手數交叉核對**：每一盤，P1 的 `n_moves` 等於 `val_hs.npz` 的 `n_moves` 和 `val_o4.npz` 的 `moves_parsed`；P1 的目標手數等於 `val_hs.npz` 的 `counts` 和 `val_o4.npz` 的 `n_moves[:,0,:]` 合計。兩邊都沒失敗的盤，不符要是 0。
  - **P1 方向檢查**：每題「只用 E_P1 的期望等級」和「Q02 的期望等級」的 Spearman ρ ≥ max(0.5 × ρ_eval, 0.078)。
  - 打分前核對（下一點）沒有找到影響 Q08 預測的程式錯誤，而且已 commit 在 apply 之後，再由 `apply --finalize` 寫入。
  - 兩份預測都通過格式檢查。
- **打分前核對**（固定範圍，由另一個看不到答案、不看任何分數的代理做）：
  - val 塊的 SHA；用 `p1_map.npz` 重算的 lp_P1TO 和 apply 的相同；
  - 遮罩版 E_P1；Q02、Q02H 重現；
  - `decide` 的輸入輸出；乘 4 只做一次。
  - 結果寫 `results/prereg08/rank/prescore_audit.json` 並 commit。

## 3. H8R：在本機驗證集打一次分

- **指令**（寫死；只在 `apply_report.json` 的 `score_allowed` 為真時由主 session 跑）：
  `.venv/Scripts/python.exe tools/score.py compare-rank results/prereg07/rank/pred_B.csv results/prereg08/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg08 H8R (A = prereg07 Q02H, reference re-read)" --json results/prereg08/rank/score_H8R.json`
- A 側是 Q02H：08 要贏的是現在的首選。
- **採用條件**：分等級平均差（Q08 − Q02H）的 95% 信賴區間下界 > 0（每級內分層重抽 10,000 次、seed 0）。成立才採用；每題平均差只記錄。
- **這次不偏離**：不成立就不採用，07 維持首選，**不寫偏離條款**。
  - 07 的 inner 增益只有 +0.0142，驗證集偵測力本來就不夠，所以 rev 3 偏離過。
  - 08 的 inner 增益約 +0.05（07 的 3.7 倍），驗證集標準誤約 0.0095；就算增益打五折，採用機會也約 8 成。
  - 所以驗證集下界不 > 0，代表 inner 和驗證之間有真正的問題（翻譯模型過擬合 fit 玩家、或驗證題的特徵有錯），不是運氣。
- **對照組重讀**：
  - A 側 1,589 題，分等級平均和每題平均要和帳上 prereg07 H7R 的 B 列（0.632702198843505、0.6345453305055525）差 ≤ 1e-12，題數相同；`pred_B.csv` 的 SHA 是 `51d7dbf8…`。
  - 記錄打分用的 Python（`.venv`，3.12.13）。不符就停下，查清前不採用。
- **合理性門檻**：Q08 的分等級平均 < 0.57 時先查 bug，要指出具體的程式錯誤才算。
- **預算（寫明重新分配）**：
  - PREREG_02 第 3 節「整季預算」：每場最多 4 次（02、03、04 各一次，另留一次給 split v2 或查 bug 重跑）。
  - 實際用掉：02、04、07（07 用的是 03 沒用到的那一次）。
  - **08 用掉第 4 次、最後一次。之後這一季競賽二沒有任何驗證集打分機會，也沒有保留給 split v2 或查 bug 重跑的那一次**（PREREG_07 第 3 節「第 4 次照舊保留」由本條取代）。
  - 測試日 T2 若要求開 split v2：照 PREREG_02 第 4 節甲照常上傳，split v2 的登記這一季不能打分。
- **之後找到 bug（不重新打分）**：
  - 在 H8R 之後（含 RESULTS 的獨立核對）找到影響 Q08 預測的程式錯誤時：先 commit 指出錯誤的修訂，修好後只在 inner 重跑（check → selfcheck → tune `--rerun --revision N`），再重跑驗證題的 apply 產生修好的預測，但**不打分**。
  - 修好的 Q08 在 inner 閘門仍過、而且 H8R 原本採用：測試日用修好的版本。規則 B 和打分版本不同的題數記錄並揭露。
  - inner 閘門沒過，或 H8R 原本就沒採用：07 維持首選。修 bug 不能讓沒採用的 08 變成採用。
- 打分紀錄帳多 2 列，note 以 `prereg08 H8R` 開頭。
- **推估的採用機率**（估計，常態近似）。驗證集標準誤用 04 的比例（1.26–1.32）乘 P1 ×4 的 inner 標準誤約 0.0074；`tune` 另照 PREREG_04 算 `se_val_record`。

| inner 增益 | 標準誤 0.0093 | 標準誤 0.0098 |
|---|---|---|
| 全部成立（+0.0525） | ≈100% | ≈100% |
| 打 0.75 折（+0.0394） | 99% | 98% |
| 打五折（+0.0263） | 80% | 76% |
| 打 0.25 折（+0.0131） | 29% | 27% |

## 4. 截止點（寫死；沒趕上就照表做，不拿額外的打分機會補）

| 截止點 | 條件 | 沒趕上 |
|---|---|---|
| K1-08 10/7（三）23:59 | rev 1 已 commit（要早於任何 P2 結果被讀進翻譯模型）；P2 抽取的目標完成日 | rev 1 沒 commit：這一季不做 08。P2 照第 1.4 節的規則 |
| K3-08 10/9（五）23:59 | 第 9 節第 3–5 步：manifest、check、selfcheck、tune 都 commit；過了的話 `frozen_config.json` 已 commit | 這一季不用 08 |
| K4-08 10/10（六）23:59 | 驗證題 P1 特徵、apply、打分前核對、H8R、RESULTS 都 commit | 這一季不用 08（不打分） |
| K5-08 10/31（六）23:59 | 採用時：測試日擴充和測試日的修訂已 commit、合併回 main、本機演練（含 4090 實測）通過 | 這一季不用 08 |
| K6-08 11/19（四）23:59:59 | 採用時：08 在官方測試集有一次成功的正式執行（第 5.2 節），報告已 commit | 最終照 PREREG_07 5.4（07 → 04 → R20 → R10） |

- 4090 在 10/24 前 C4 優先；被讓出導致截止點沒趕上，照表做。

## 5. 採用後的測試日（細節在 K5-08 前另寫測試日的修訂）

### 5.1 保護舊路徑（rev 1 就生效）

- 這些一律不改：
  - R20 的 39 個檔（tag `r20-testday-pre04`）、04 的 21 個檔（`p04-testday-pre06`）、06 的測試日路徑（`p06-testday-pre07`）；
  - `results/prereg01`–`prereg07/**`；`results/prereg08/p1/**`、`results/prereg08/p2/**`；
  - `tools/p0*_*.py`、`t02_*`、`t04_*`、`t06_*`、`t07_*`；`third_party/katago/**`；釘住檔；
  - H8R 之後的 `results/prereg08/rank/**`、`results/prereg08/feat/**`。
  - `tools/official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_07_HUMANSL.md`、本登記只准加行。
- **07 的指紋，順序寫死**（照 PREREG_07 第 5.1 節替 06 做的方式；08 的測試日程式動工前做完）：
  1. 做 `results/test07/design/p07_path_fingerprint.json`：`tools/t07_*.py`、`tests/test_t07_*.py`、`results/test07/design/*`、`docs/TESTDAY_07.md`、`tests/conftest.py` 釘 SHA；`official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_07_HUMANSL.md` 放 `insert_only`。
  2. 在那個 commit 打 tag `p07-testday-pre08`，加一支釘住它的 `tests/test_t07_path_hygiene.py`。
  3. 記下 R20、04、07 的假測試演練輸出當基準；合併前要逐位元組相同。
  4. 分支 `p08-testday` 從那個 commit 開。
- 08 的測試日程式全部是新檔：`tools/t08_rank.py`、`tools/t08_fake_tests.py`、`tests/test_t08_*.py`、`docs/TESTDAY_08.md`。輸出只寫 `results/test08/`，大檔放 `data/test08/`；演練寫 `results/test08_rehearsal/`、`data/test08_rehearsal/`。
- 不需要新的 Colab notebook；使用者不用多跑 Colab。

### 5.2 要做的東西

- **`tools/t08_rank.py p1 --set seam|public|private`**（4090）：
  - 前提：07 的 `t07_apply_report.json` 是正式模式、結束碼 0、已 commit（07 判「不用」時 08 也不用）；H8R 採用；K6-08 未過；`nvidia-smi` 沒有別的 compute 程序；`.venv-gpu` 等於 `frozen_config.json`；權重完整 SHA 正確。
  - seam：驗證題 val_keys 順序的前 400 盤。public、private：和 test02／07 同一組的 key 逐列相同。
  - 塊的 `created_utc` 早於 R20 第一次上傳的，一律拒絕。可以續跑；用獨立背景程序。
  - 估計 0.55–0.65 小時：約 10,400 盤 × 228 個局面，用 1,050–1,190 positions/s。接在 07 的 HS（2.7–3.9 小時）之後，同一晚做完。
- **`tools/t08_rank.py apply`**（CPU）：
  - **GPU 接縫**：seam 400 盤和 `val_seam_feats.npz` 比。
    - `failed`、`counts`、`n_moves` 完全相同；
    - 標準化後的特徵最大差 ≤ 0.05；lp_P1TO 每盤最大差 ≤ 0.02；
    - 逐位元是否相同只記錄。
    - 07 的 seam 在同一批組合下重算是逐位元相同；演練實測後，測試日的修訂可以在拿到任何測試資料前調整容許差。
  - **Q02H 對得上 07**：在 public、private 上用 07 的程式重算的 Q02H，要和 07 已 commit 的 `rank_test_probs.csv` 逐位元組相同。
  - **手數交叉核對**：和 07 的 `<組>_hs.npz`、04 的 `o4_<組>.npz` 比，不符 0 盤。
  - **本機接縫**：seam_val（1,589 題）用 test02／04／07 的 seam_val 輸入加已 commit 的 `val_lp_p1.npz` 算 Q08，和 `results/prereg08/rank/rank_val_probs.csv` 比：容許差同 07；07 的 seam_val 逐位元組相同時要求逐位元組相同。
  - **P1 失敗率**每組 ≤ 0.5%（`200 × 失敗 ≤ 盤數`）。
  - **方向檢查**：ρ ≥ max(0.5 × ρ_mono, 0.155)。
  - 都過了才寫 `results/test08/rank/submission_rank.csv` 等。結束碼同 t07：0／2／3／4。
- **`tools/official_log.py` 只加行**：
  - 認得 `results/test08/rank/`（在 07、04 的分流之前），方法名稱「prereg08 Q08 = Q02H + 4 x P1TO, rule B (甲)」。
  - `prepare` 08 之前，log 要有 R20、04、07 的第一次上傳。
  - 有符合的 08 而交別的，要 `--not-08-reason "PREREG_08 rev N: …"`（修訂的 commit 晚於被推翻的紀錄）。
  - log 沒有 08 紀錄時，輸出和 tag `p07-testday-pre08` 的版本逐位元組相同。
- `docs/TESTDAY_08.md`：Claude 的步驟。

### 5.3 4090 不能用時

- 08 的 P1 特徵只在本機 RTX 4090 上算。不登記 CPU 或 Colab 備案（理由同 PREREG_07 第 5.3 節）。
- 4090 不能用、或 GPU 接縫過不了、K6-08 前修不好：這一季不用 08，最終照 PREREG_07 5.4。07 本身也要 4090，所以實際上就是 07 的規則。

### 5.4 上傳順序和最後一版（競賽二）

- **順序**：R20（11/04）→ 04 → 07 → 08。每一份都要使用者在對話裡說 OK。
- **最後一版**：有符合的 08 → 08；沒有 → 照 PREREG_07 5.4（07 → 04 → R20，T1 FAIL_OPENING 時 R10）。不交沒在驗證集打過分的組合。
- public 分數不進判定；08 比 07 低 0.05 以上時只查錯，查不到就維持本節的結果。
- 取代的條文：PREREG_07 第 5.4 節「有符合的 07 時最終是 07」→ 有符合的 08 時最終是 08。不改原文（照 PREREG_07 第 5.5 節的前例）。

## 6. 另一場：不交換輸出

- 競賽一不用 P1；08 的任何輸出都不進競賽一。
- 不用競賽一的測試資料。

## 7. 刻意不放進 08 的

| 做法 | 理由 |
|---|---|
| MLP 等非線性翻譯模型 | P1 只登記了 LR；看過 inner 再換模型等於新設計 |
| 微調 KataGo（只學條件、全網路、加 head） | 10/5 探索：成本 6–12 GPU 小時起、誤差可能變得和 R20 一樣；沒有實測 |
| 對手方 HS（14 個 profile） | P1 的對手方特徵已用 1/14 的成本回答同一個問題 |
| 用競賽一的測試資料 | 規則第 8 條有疑義 |
| 再搜尋倍數或權重上限 | E1 到 ×4 已飽和；再搜就是看 inner 挑參數 |
| 其他 λ 格點、T 或 T＋O 變體、拿掉 HS 項 | P1 事前寫死 TO 和格點；只記錄的組合不能救首選 |
| Q02＋P1TO、R20＋P1TO 當候選或備案 | 沒在驗證集打分，不能上場 |
| 8 種對稱平均、TensorRT | 計算量或新的數值路徑；沒驗證過 |
| 測試日的 CPU 或 Colab 備案 | 第 5.3 節 |
| probe 再加大（P3） | 第 4 次打分是最後一次；P2 之後不再擴充 |

## 8. 揭露（寫進 RESULTS 和報告）

- **事前與事後的界線**：
  - P1 的設計、首選、閘門在任何 bank／mono 特徵之前 commit（`4d6a853`）。
  - P1 過關（`75b6684`）。
  - 閘門之後的探索（`0ca8316`）：倍數、學習曲線、0 盤重複的檢查。
  - **乘 4 和加 P2 都是看過 inner 結果才定的**，所以 inner 的數字偏樂觀。真正的判定只有第 3 節那一次驗證集打分。
- **P2 的時序**：名單 `a709408`；抽取 10/6 10:12 開始；本登記 commit 時沒有任何 P2 的結果被看過。P2 用或不用照第 1.4 節的固定規則。
- **權重上限**：w_P1 的上限 4 照 PREREG_04 不改，但乘 4 等於把 lp_P1TO 的上限放寬到 16。
- **HS 的權重**：P1 試跑時 HS 的權重有 4 折是 0（Q02＋P1TO ≈ Q02H＋P1TO）。Q08 保留 HS 項是為了直接接在 07 之上；凍結時 w_HS 的值照實記。
- **外部資源**（規則第 3 條）：
  - KataGo 的程式（commit `d91ea855`）和 HumanSL 網路 `b18c384nbt-humanv0`（v1.15.0）。授權同 PREREG_07 第 8 節（程式 MIT 類、網路權重是 KataGo 的網路授權，保留聲明、不轉發權重）。
  - 08 多用了網路的中間層（主幹輸出），仍是同一個網路。
  - 我們的部分：中性 metadata 的設定、特徵的選擇和平均、翻譯模型、合併與校準、評估和測試日工具。
- **消融**（報告審查的「程式原創性」）：
  - 驗證集：H8R 本身就是有／沒有 P1 的消融（Q08 對 Q02H）。加上 H7R（Q02H 對 Q02），得到「沒有 KataGo → HS → HS＋P1」三層。
  - inner：Q02、Q02H、Q08、Q08 ×1、只用 P1 probe 的 Q08、Q02＋P1TO、R20＋P1TO（只記錄）。
- **同一個驗證集的第四次依序使用**：H2R、H4R、H7R、H8R；A 側的 Q02H 本身是在 rev 3 偏離規則下定為首選的。
- **預算的重新分配**：08 用掉第 4 次，取代 PREREG_07 第 3 節的保留；之後的 bug 只能不打分處理（第 3 節）。
- **看過測試棋的疑慮**：同 PREREG_07 第 8 節（HumanSL 的訓練來源沒有幽玄之間）。
- **數值**：GPU fp16 不能逐位元重現，測試日靠容許差核對。
- **生成式 AI** 協助的部分照規則揭露（`competitions/_drafts/disclosure-template.md`）；報告草稿 `report/` 競賽二第 7 節要依 H8R 結果改寫。

## 9. 順序

1. 本登記 rev 1 commit，再用一個 commit 填上 rev 1 的編號。**硬規則：rev 1 的 commit 要早於任何 P2 特徵被讀進翻譯模型**（P2 抽取本身可以繼續跑）。
2. 寫程式：
   - `tools/p08_p1.py`（`manifest`、`val`、`status`）、`tools/p08_rank_fuse.py`（`check`、`selfcheck`、`tune`、`apply`、`apply --finalize`）；
   - `tests/test_p08_*.py`（含 `test_p08_hygiene.py`）；
   - 對 `tests/test_p06_player_hygiene.py`、`tests/test_p07_hygiene.py` 的最小修改（讓 07 的列之後可以接 0 或 2 列 `prereg08 H8R`）。
   - 經獨立審查、全套測試綠了才 commit。
3. P2 600／600，或 10/8 23:59 已過：`p08_p1.py manifest` → commit `results/prereg08/feat/inner_manifest.json`（含 P2 用不用）。
4. 實作和登記的差異寫成 rev 2，在 `check` 之前 commit。`check` → commit；`selfcheck` → commit。
5. `tune`（只判一次）：閘門都過才寫 `frozen_config.json`、`p1_map.npz` → commit。沒過：寫 `results/prereg08/RESULTS.md`（未送分）→ commit，結束。
6. `p08_p1.py val`（4090，獨立背景程序，約 1.0–1.2 小時）→ commit `val_p1.npz`、`val_seam_feats.npz`、`val_report.json`。
7. `p08_rank_fuse.py apply` → commit；打分前核對 → commit `prescore_audit.json`；`apply --finalize` → commit。
8. `score_allowed` 為真才跑第 3 節的 H8R 指令。打分紀錄帳、`score_H8R.json`、`results/prereg08/RESULTS.md`（另一個代理照第 2.7 節的範圍獨立核對）、地圖、`CLAUDE.md` 一起 commit。
9. 採用的話：照第 5.1 節做 07 的指紋和 tag `p07-testday-pre08`、記基準，再從那個 commit 開分支 `p08-testday` 做測試日擴充：實作 → 審查 → 測試日的修訂 → 演練（含 4090 實測）→ 防誤改閘門 → K5-08 前合併。

## 10. 時程（台灣時間；都是估計）

| 日期 | 事 | 誰 |
|---|---|---|
| 10/06 | P2 抽取（4090，背景，約 7–8 小時）；本登記 rev 1、第 2 步 | 背景程序、Claude |
| 10/07 | 第 3–5 步（CPU：check／selfcheck 約 1 小時，tune 2–4 小時） | Claude |
| 10/07 夜–10/08 | 第 6 步（4090 約 1 小時）；開跑前 `nvidia-smi`，C4 要用就讓出 | 背景程序 |
| 10/08–10/10 | 第 7–8 步（打分前核對、H8R、RESULTS） | Claude |
| 10/11–10/31 | 07 的指紋和 tag、08 測試日擴充、修訂、演練、合併 | Claude |
| 11/04–11/07 | R20 → 04 → 07 照舊；07 的 HS 之後同一晚跑 `t08_rank.py p1`；07 上傳後上傳 08 | 使用者上傳（每次說 OK） |
| K6-08 11/19 | 08 成功正式執行的最晚 commit | — |
| 11/22–11/23 | `official_log.py status`；最後一次上傳 11/23 23:59 前 | 使用者 |

使用者在 10 月不用做任何事。測試日多 1 次上傳（08），最壞再多 1 次重傳。

## 11. 防誤改閘門

- CLAUDE.md 的閘門指令加 `tests/test_p08_hygiene.py`，變成十個檔。
- 唯一允許的失敗仍是 `test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`：帳多了 H6P 4 列、H7R 2 列，H8R 之後再加 2 列，每列都要對到已 commit 的登記。
- 第 9 節第 2 步修改 `test_p06_player_hygiene.py` 和 `test_p07_hygiene.py` 之後，這兩支在 H8R 之後仍要全過；`test_p08_hygiene.py` 釘住 H8R 的 2 列。

## 12. rev 2：實作和登記的差異（在 check 之前）

1. **`tests/test_t07_hygiene.py` 也改了**（第 9 節第 2 步的清單漏了它）：它從 tag `p06-testday-pre07` 起凍結 `tools/p0*.py`、`tests/test_p0*.py`，新的 p08 檔一定會讓它失敗。改法：範圍縮成 `p0[1-7]*`（涵蓋的還是原本那 15＋17 個檔）；`test_p06_player_hygiene.py`、`test_p07_hygiene.py` 這兩支照 spec E 改過的檔從比對裡排除，改在 `FROZEN_EDITED` 釘 LF SHA，另加 `test_edited_files_pinned`。
2. **BLAS 執行緒**：設了 `OMP_NUM_THREADS=4` 時，07 的 Q02H 驗證題 csv 不再逐位元組等於 `6ae2db60…`，L-BFGS 的迭代次數也會變。所以 `check`、`selfcheck`、`tune`、`apply` 在 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 任一有設時拒絕；正式步驟用預設執行緒。只有預覽可以用 `C10_P08_PREVIEW_THREADS=1` 放行。
3. **`frozen_config.json` 的 GPU 參照**（第 2.6 節沒寫來源）：torch 取自 P1、P2 塊記的值（經 manifest；不只一種時 tune 在擬合前就停）；Python、cuDNN、GPU、numpy 沿用 07 `frozen_config.json` 的 `venv_gpu_reference`（同一個 `.venv-gpu`）。
4. **manifest 的塊格式檢查**：項目和 `p1_fuse.load_p1_set` 相同（format、set、metadata SHA），另核對 precision、device、batch、channels_last、KataGo commit、`p1_lib`／`p1_extract` 的 SHA；不直接呼叫 `load_p1_set`，因為它會把 P2 整份特徵（約 1.5 GB）讀進記憶體。`selfcheck`、`tune` 載入塊時仍走 `load_p1_set`。
5. **第 1.4 節 P2 規則的「失敗盤」**：只算 `failed` 旗標；沒有目標手的盤只記錄（tune 照第 1.5 節不用它們）。
6. **G4**：要剛好 120,544 題、95% 下界 > 0、黑白點估計都 > 0。沒有 07 那條「單色失敗盤 ≤ 306」，因為 `check` 已要求 bank、mono 失敗盤是 0。
7. **`val` 的 probe 重複比對（只記錄）**：讀完題目檔之後、import `p1_lib` 之前，用 `c10data` 讀 fit 訓練棋譜（約 1–2 GB、幾分鐘），比對的 probe 照 frozen 的 `probe_used`。`val` 的 worker 數預設 20、每批 1,024 個局面 × `group_batches` 4，和 P1、P2 相同。
8. **審查的 minor，已修**：apply 讀驗證塊時另核對 `mode == "real"` 和 `keys_sha256`；tune 擬合前丟掉多餘的 probe 副本（記憶體尖峰）。
9. **審查的 minor，不修、照操作規則避免**：
   - `apply` 拒絕覆寫已 commit 的輸出，所以第 3 節「H8R 之後找到 bug、修好重跑 apply、不打分」那條路，到時要先寫修訂、輸出改寫到另一個位置。
   - `val` 沒有防兩個程序同時跑的鎖：只用一個背景程序跑。
   - `apply --finalize` 只核對輸出 SHA 和稽核，不重核 frozen 前提（同 07）。

## 13. rev 3：測試日的修訂（在任何測試資料之前；分支 `p08-testday`）

第 5.1 節第 1–3 步（07 的指紋、tag `p07-testday-pre08`、R20／04／07 的假測試基準 `results/test08/design/baseline_pre08.json`）已在 main 的 `dd336d3` 做完，分支從那個 commit 開。本節寫實作和第 5.2 節的差異，以及測試日照什麼做；Claude 的步驟在 `docs/TESTDAY_08.md`。

1. **程式**：分支 `p08-testday` 的 `0e78e6d`（`tools/t08_rank.py`、`tools/t08_fake_tests.py`、`.gitattributes`／`.gitignore` 各加行）、`6a7db9c`（`tools/official_log.py` 只加行＋`tests/test_t08_official_log.py`）、`0a8dffd`（`tests/test_t08_rank.py`、`tests/test_t08_hygiene.py`、`tests/test_t08_dryrun.py`、`results/test08/design/t08_fake_golden.json`）。輸出只寫 `results/test08/`，大檔放 `data/test08/`；演練只寫 `results/test08_rehearsal/`、`data/test08_rehearsal/` 或 repo 外。07 的工具（`tools/t07_rank.py`）只 import、不改：Q02、Q02H、seam_val 的輸入和 test02／04 的前提都走它的函式。
2. **`p1` 的前提，照 5.2 的順序**：git 乾淨（正式）；07、08 的已 commit 檔 SHA 不變（08 釘 11 個檔：`frozen_config.json`、`p1_map.npz`、`rank_val_probs.csv`、`pred_A/B.csv`、`apply_report.json`、`score_H8R.json`、`val_lp_p1.npz`、`val_seam_feats.npz`、`val_p1.npz`、`val_report.json`）；H7R（rev 3 的那條）、**H8R 採用**；K6-08；T1；test02；R20 的第一次上傳；**07 的 `t07_apply_report.json` 是同一模式、結束碼 0、usable、已 commit**（07 判不用時 08 結束碼 3）；`nvidia-smi` 沒有別的 compute 程序；`.venv-gpu` 的 Python、torch、cuDNN 等於 `frozen_config.json` 的 `venv_gpu_reference`（不等是 4），GPU 型號（不對是 2）；權重完整重算 SHA。
   - 「H8R 採用」＝`score_H8R.json` 的 1,589 題、分等級平均差的 95% 下界 > 0、A 側兩個值和 H7R 的 B 列差 ≤ 1e-12、Q08 ≥ 0.57（第 3 節）。不成立是結束碼 3。
   - 第 10 節時程寫「07 的 HS 之後同一晚跑 `t08_rank.py p1`」。但 5.2 把 07 的 apply（要等 04 的 apply，11/05–11/07）列為前提，所以 `p1` 在 07 的 apply 之後才跑；約 0.6 小時，不到一小時，白天可以跑。以 5.2 為準。
3. **`p1` 的輸出**：每 200 盤一塊，寫 `data/test08/p1/<組>/chunk_*.npz`（格式 `c10-test08 p1 chunk v1`，欄位同 val 的塊，多記 mode）。全部寫完才寫 `results/test08/p1/<組>_p1.npz`（題號、slot、執色、counts、n_moves、failed、log p／熵的平均；**seam 另存 400 盤完整的 770×2 維平均**）和 `p1_<組>_report.json`（每塊的 SHA）。public／private 的完整平均（每組約 30 MB）不進 git，留在塊裡；`apply` 逐塊重算 SHA、核對 counts／n_moves／failed 和 `<組>_p1.npz` 相同，才讀平均。塊的 `created_utc` 早於 R20 第一次上傳的一律拒絕（2，資料夾改名、不刪）；正式模式拒絕演練（fake）的塊。鍵：slot 要和 test02 的 `val_keys.npz`、07 的 `<組>_hs.npz` 逐列相同。
4. **GPU 接縫的定義**（容許差照 5.2 不變）：seam 400 盤和 `val_seam_feats.npz` 比。
   - `failed`、`counts`（兩方）、`n_moves`、題號、slot、執色完全相同。
   - 「標準化後的特徵」＝`p1_fuse.variant_X(…, "TO")` 的 1,540 維，用 `p1_map.npz` 的 μ、σ 標準化，對手方沒有手時是 0（同 `p1_fuse.standardise`）；在兩邊都可用的盤（沒失敗、有目標手）上取最大絕對差，≤ 0.05。
   - 「lp_P1TO 每盤最大差」＝乘 4 之前的 log_softmax(W·z＋b)，每盤 10 級取最大，再對盤取最大，≤ 0.02。
   - 逐位元是否相同（`bitwise_equal_means`）只記錄；lp 差 > 0.005 另標「檢討線」（同 07），只記錄。
   - 4090 實測：主 session 照 `results/test08/design/build_review.md` 跑 `data/prereg08/gpu_rehearsal_t08.sh`，結果記在 `results/test08/design/gpu_rehearsal_*.json`。實測後若要調整容許差，要在拿到任何測試資料之前另寫修訂（rev 4 起）；沒有修訂就維持 0.05／0.02。
5. **Q02H 對得上 07**：public、private 用 `t07_rank.q02h_set`（07 的程式、07 已 commit 的 `<組>_hs.npz`）重算 Q02H，必須和 07 已 commit 的 `rank_test_probs.csv` 逐位元組相同（不同是 4）。之前先核對 07 報告 `inputs_bound` 的每一個檔都沒變（變了是 2，先照 TESTDAY_07 處理）。seam_val 上的 Q02H 和 07 的 `seam_val/rank_val_probs.csv` 是否相同只記錄。
6. **手數交叉核對**：public、private 每一盤，P1 的 `n_moves`、目標手數 `counts[:,0]`，和 07 的 `<組>_hs.npz`（`n_moves`、`counts`）、04 的 `o4_<組>.npz`（`moves_parsed`、`n_moves[:,0,:]` 合計）比；兩邊都沒失敗的盤不符要是 0（不符是 4；之後的重跑要 `--bug-revision N`，N ≥ 4）。
7. **本機接縫**：seam_val 用 07 的 seam_val 輸入（test02／04 的 seam_val、`val_hs.npz`）加已 commit 的 `val_lp_p1.npz`（不從塊重算）算 Q08，和 `results/prereg08/rank/rank_val_probs.csv`、`pred_B.csv` 比。容許差同 07（`results/test04/rank/seam_val/seam_local.json` 的 tol，規則 B 最多差 4 題）。**07 的 `seam_local.json` 逐位元組相同（`byte_identical_probs`、`byte_identical_pred_B`）時，08 要求逐位元組相同**，而且 seam_val 的方向 ρ 要等於 `results/prereg08/rank/apply_report.json` 的值（不等是 4）。題號、順序、盤數不同是 4；07 的接縫成立、08 卻超出容許差是登記的「08 不用」（3）。
8. **失敗和方向**：P1 失敗＝`failed` 或整盤沒有手；每組 `200 × 失敗 ≤ 盤數`，否則 3。方向 ρ：每組「只用 E_P1 的期望等級」和「Q02 的期望等級」的 Spearman ρ ≥ max(0.5 × ρ_mono, 0.155)＝0.4837（ρ_mono＝0.9673017067776464，`frozen_config.json`），否則 3。
9. **數值路徑**（演練實測）：測試題的 lp_P1TO 是每組一次矩陣乘法（`p08_rank_fuse.p1_lp_from_map`，和驗證題同一個函式），驗證題當時是 19,156 列一次乘。用驗證題假扮測試題的演練裡，9,602 盤有 48 盤的 lp_P1TO 差 ≤ 4e-15（BLAS 依列數分塊的捨入），機率差 ≤ 1e-12，規則 A、B 的等級全部相同。所以 800 題測試列「等於 prereg08 驗證前 800 列」用 1e-12 和等級相同判，不要求逐位元組相同（07 的演練是逐位元組相同）。seam_val 讀已 commit 的 lp，逐位元組相同。
10. **結束碼**同 t07：0／2／3／4；前提裡登記的 3 先判，再看「輸出已存在」；出現 3 或寫好 `t08_apply_report.json` 之後，只有 `--bug-revision N`（N ≥ 4、那一列的 commit 晚於被推翻的紀錄）才能重跑。
11. **official_log**（只加行）：`results/test08/rank/` 的檔在 07、04 的分流之前認出；方法名稱「prereg08 Q08 = Q02H + 4 x P1TO, rule B (甲)」；`prepare` 08 要 R20 的第一次上傳、08 報告綁定的那一筆 07 已 uploaded 而且仍符合 PREREG_07 5.4（它又要求綁定的 04 已上傳），P1 的塊都晚於 R20 上傳；`prepare`／`final`／`status` 每次用報告裡的數字和檔重算 5.2 的條件和 H8R（`score_H8R.json` 釘 LF SHA `8ecd2425…`）。有符合的 08 而交 07／04／R20／R10，要 `--not-08-reason "PREREG_08 rev N: …"`（N ≥ 4）。為了不把 08 當成 R20 或 04，04 和 07 的規則函式各加了「08 跳過」的行；`status` 在最終是符合的 08 時不印 04／07 的違反，04／07 的提示改成 08。log 沒有 08 紀錄時，所有輸出、紀錄檔、回傳值和 tag `p07-testday-pre08` 的版本逐位元組相同（`tests/test_t08_official_log.py` TagVersusNew，含一段 R20 → 04 → 07 的流程）。
12. **基準**：`C10_SLOW=1` 的 `tests/test_t08_dryrun.py` 在同一棵假樹上跑 R20、04、07、08，R20／04／07 的 27 個輸出等於 `results/test07/design/t07_fake_golden.json`（第 5.1 節第 3 步）。08 自己的基準 `results/test08/design/t08_fake_golden.json` 是程式寫好之後第一次通過的演練記下的（同 07 的做法），照實揭露。
13. **`.gitattributes` 加 `results/test08/** -text`**：之前 commit 的 `results/test08/design/baseline_pre08.json` 在 CRLF 的工作樹會被看成已修改；分支上已重新 checkout 成 LF（內容和 commit 的相同）。合併回 main 之後在 main 也要做一次（刪掉工作樹的那份再 `git checkout --`）。
14. **防誤改閘門**：CLAUDE.md 的閘門指令加 `tests/test_t07_path_hygiene.py`（第 5.1 節加的，還沒進閘門）和 `tests/test_t08_hygiene.py`，共十二個檔；唯一允許的失敗不變。
15. **K5-08（10/31 23:59）的條件**：分支合併回 main；閘門只有已知的那一項失敗；`C10_SLOW=1` 的 t08 演練全過；本 rev 3 已 commit；4090 實測做過並 commit 紀錄。
16. **取代的條文**：PREREG_07 第 5.4 節「有符合的 07 時最終是 07」→ 有符合的 08 時最終是 08（第 5.4 節已寫，照 PREREG_07 第 5.5 節的前例不改原文）；第 10 節「同一晚」見第 2 點。
17. **審查**：建置代理只做了自我核對和測試，沒有獨立審查；合併前由主 session 決定要不要另派審查，照實揭露。

### rev 3 補記（2026-10-09，兩份獨立審查之後、合併之前；仍在任何測試資料之前）

審查沒有 blocker／major。下面是照審查修的地方和 4090 實測；第 1–17 點的內容不變，這裡只補充。

18. **STOP 的結束碼**：`p1` 因為 STOP 檔或 `--stop-after-min` 停下、那一組還沒算完時，改回 **結束碼 5**（原本是 0）。5 不判定任何事、不寫 not_used／attempt_verdict，已寫好的塊留著；刪 STOP 後跑同一行就是續跑。第 10 點的 0／2／3／4 對 `apply` 和 `p1` 的其他情況不變，5 只出現在 `p1`。改的原因：結束碼 0 會讓串接的啟動腳本接著開下一組，GPU 讓不出來。`docs/TESTDAY_08.md` 第 3 步的 `p1_run.sh` 改成：已有報告的組跳過、任何一組非 0（含 5）整支停下。測試：`tests/test_t08_rank.py::test_stop_exit_code`、`tests/test_t08_dryrun.py::test_stop_file_exits_5`。
19. **塊的重用和 `apply` 的塊核對加嚴**：`p1` 續跑時，已存在的塊除了原本的格式、方、refs、鍵檔、中性 metadata、模式、環境之外，還要題目檔 SHA（`questions_sha256`）、P1 程式碼（`code_sha256`）、權重 SHA（fake scorer 是 `fake`）、precision／batch／channels_last／device 都等於這一次，不等是 2（不覆寫）。`apply` 逐塊核對同樣幾項對得上那一組報告的鍵檔和題目檔（public／private 的題目檔也要等於現在 tests 資料夾裡的那份；seam 等於登記的驗證題目檔）、權重和 `RUN` 設定；三組的 P1 程式碼紀錄要相同，而且等於磁碟上那些檔現在的 LF SHA，否則 2。測試：`test_chunk_check_questions_code_and_run_settings`、`test_chunk_of_other_question_file_or_code_not_reused`、`test_chunk_run_settings_differ`、`test_p1_code_differs_between_sets`。
20. **鍵和 07 的比對改成相等**：`hs_keys_match` 用整組（在演練的 `--max-games` 截斷之前）和 07 的 `<組>_hs.npz` 比，列數要相等（原本是 `>=`）。
21. **`t08_fake_tests.py from-val`** 的 seam 核對加上 n_moves、failed、slot、執色（原本只比平均、counts、題號）。只影響演練。
22. **本機接縫容許差的來源（更正第 7 點的寫法）**：程式讀的是 `results/test04/rank/t04_apply_report.json` 的 `seam_local.tol`（和 07 的 `t07_rank.py` 相同），不是 `results/test04/rank/seam_val/seam_local.json`。值和判定方式不變；報告欄位 `tol_source` 的文字跟著改。
23. **TESTDAY_08 的盤數**：public／private 各 400 題、5／10／15／20 盤各 100 題＝各 5,000 盤（第 5.2 節），連 seam 共約 10,400 盤（原寫「約 9,600」）；時間估計 0.55–0.65 小時不變。另寫明 `p1_run.sh` 跑的時候不要改已追蹤的檔（正式模式每一組開頭都檢查 git 乾淨，改了下一組會結束碼 2；重跑同一支就續跑）。合併後閘門是十二個檔：main 的 CLAUDE.md 閘門（已含 `test_t07_path_hygiene.py`）加 `test_t08_hygiene.py`。
24. **4090 實測**（2026-10-09 02:22–02:31 台灣時間，`.venv-gpu` torch 2.11.0+cu128、cuDNN 91900、驅動 616.92；紀錄 `results/test08/design/gpu_rehearsal_seam.json`、`gpu_rehearsal_compare_private.json`、`gpu_rehearsal.log`；演練樹 `D:\c10_t08_reh`）：
    - **GPU 接縫**（seam 400 盤，和驗證題的前兩塊同一批盤、同樣的 200 盤分塊）：counts／n_moves／failed 全同，標準化特徵最大差 0、lp_P1TO 最大差 0，**平均逐位元相同**；之後的 `apply --dry-run` 結束碼 0，本機接縫逐位元組相同，交件檔 `457c8542…` 和 CPU 演練相同。前一次（10/8）的同一步也是 0。
    - **不同的批次組合**：原本第 2 步（public 前 400 盤）其實就是 seam 的同一批盤、同樣分塊（局面數 46,175／44,138 完全相同），量不到新東西，改成 private 前 1,000 盤：驗證列 4,787–5,786，每塊從驗證塊格線之外開始（4,787 mod 200 = 187），所以 GPU 的批次組合和驗證時不同。結果 failed／counts／n_moves／執色全同；**標準化特徵最大差 0.0186**（中位數 0.0050、99% 分位 0.0092），**lp_P1TO 每盤最大差 0.0160**（中位數 0.0024、90% 分位 0.0053、99% 分位 0.0086；1,000 盤中 115 盤 > 0.005），平均不逐位元相同，P1 單獨的最大等級 1,000 盤全同。
    - **速度**：seam 786 positions/s（第一塊有 cuDNN benchmark 暖機，586；第二塊 1,224），private 1,000 盤 983 positions/s（暖機後各塊 1,152–1,179）。照 10,400 盤、每盤約 228 個局面，約 0.55–0.65 小時的估計維持。
    - **容許差的檢討**：lp 差 0.0160 超過 0.005 的檢討線（07 同類實測 0.0059），是 0.02 的 80%；特徵差 0.0186 是 0.05 的 37%。這是 fp16 在不同批次組合下的捨入，不是環境漂移。測試日的 GPU 接縫比的是 seam 400 盤，它和驗證題是同一批盤、同樣分塊，兩次實測都是 0，所以 0.05／0.02 對接縫判定的餘裕足夠；**維持 0.05／0.02，不寫 rev 4**。要揭露的是：測試題的 P1 特徵和「假如在驗證時一起算」相比，每盤 lp_P1TO 會有這個量級（最大約 0.016、乘 4 之前）的數值差，這是方法本身的數值雜訊，不進任何判定。若測試日 seam 出現非 0 的差（例如 cuDNN 選了別的演算法），只要在容許差內就照登記判過；接近 0.02 時照第 5.3 節停下查。

## 14. rev 4：GPU 接縫 lp_P1TO 的容許差（2026-10-09，在任何測試資料之前）

決定者：Claude（使用者 10/2 授權「只要對最終排名有幫助」，10/9 授權這一輪一路做到底）。依據是 rev 3 補記第 24 點的 4090 實測；第 5.2 節原文寫明「演練實測後，測試日的修訂可以在拿到任何測試資料前調整容許差」。

1. **改什麼**：GPU 接縫（第 5.2 節、第 13 節第 4 點）的 lp_P1TO 每盤最大差容許值 **0.02 → 0.04**（乘 4 之前）。標準化特徵 0.05、counts／n_moves／failed 完全相同、其他所有判定都不變。
2. **為什麼**：
   - 不同批次組合下，lp_P1TO 的 fp16 捨入差實測最大 0.0160（1,000 盤；99% 分位 0.0086），是舊容許差的 80%。測試日的 seam 和驗證題同一批盤、同樣分塊，兩次實測都是 0；但若測試日 cuDNN 選了不同演算法，seam 的雜訊會接近「不同批次組合」的量級，0.02 的餘裕只有 1.25 倍。
   - 接縫誤判的代價是這一季不用 08（退回 07，驗證集差 0.0612），對名次是實際損失。0.04 對實測最大雜訊有 2.5 倍餘裕。
   - 抓程式錯誤不靠這一條：Q02H 逐位元組對 07、手數交叉核對 0 不符、本機接縫逐位元組相同、方向 ρ、失敗率都不變；程式錯誤造成的 lp 差通常遠大於 0.04。07 的 lp_HS 容許差也是 0.02 而實測 0.0059，比例上和這次的 0.04 對 0.016 相近。
3. **取代**：rev 3 補記第 24 點「維持 0.05／0.02，不寫 rev 4」和「接近 0.02 時照第 5.3 節停下查」由本節取代：接縫在 0.04 以內照登記判過；超過 0.04 照第 5.3 節（這一季不用 08，最終照 PREREG_07 5.4）。
4. **程式**：`tools/t08_rank.py` 的 `TOL_LP`、`tools/official_log.py` 的 `T08_TOL["lp_p1"]`（這一行是 rev 3 在本分支加的，對 tag `p07-testday-pre08` 仍是只加行）、`tools/t08_fake_tests.py` 的說明和判定、`docs/TESTDAY_08.md`，以及兩個測試的數值（`tests/test_t08_rank.py` 的常數檢查、`tests/test_t08_official_log.py` 的「gpu lp」0.021 → 0.041）。10/9 的 4090 演練在新容許差下判定不變（seam 0／0，不同批次 0.0160 ≤ 0.04）。
5. **之後的修訂編號**：測試日要推翻 08 的紀錄或 `--bug-revision`／`--not-08-reason` 時，用 rev 5 起；程式的「N ≥ 4、而且那一列的 commit 晚於被推翻的紀錄」照舊，本 rev 4 的 commit 早於任何測試日紀錄，不能拿來推翻。


> 2026-10-10 追加（只加這一行，不是修訂列）：4090 不能用時的 Colab GPU 備援另寫在 `docs/PREREG_11_GPU_FALLBACK.md`；它的條件成立時，取代本登記第 5.2 節 `p1` 前提中的 `.venv-gpu`／RTX 4090 一句和第 5.3 節，其他判定、容許差和最後一版的規則都不變。
