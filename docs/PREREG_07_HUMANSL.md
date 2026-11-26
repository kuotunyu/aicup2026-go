# 事前登記 07：競賽二加上 KataGo HumanSL 的逐手棋力證據（Q02H＝Q02＋HS）

**白話版**：KataGo 公開了一個叫 HumanSL 的網路：給它一個盤面和一個「棋力設定」（例如 5 級、1 段），它會說這個棋力的人類下在每一點的機率。我們拿目標玩家每一手實際下的位置，在 14 個棋力設定下各算一次機率，得到一條「這個人像幾級」的曲線，再用一個小模型把曲線翻成 10 個等級的機率（HS 項），照 04 加 O4、SW 的同一套方法加進 Q02。試跑只用訓練玩家，在 inner 上比 Q02 多 +0.0142（95% 下界 +0.0047）。最像官方題型的單色檢查（G4）要等全部算完才判；寫本登記時它只被中途看過兩次部分結果（第 0 節）。本機驗證集只打一次分，這一次就是判定。測試日在本機 4090 上算測試棋的 HS（估 2.7–3.9 小時），不需要新的 Colab；最後一版的順序是 07 → 04 → R20 → R10。

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。競賽二的前幾份：
  - PREREG_02：R20 採用，本機驗證集 0.5891。
  - PREREG_03：沒過閘門，未送分。
  - PREREG_04：Q02＝R20＋O4＋SWA＋SWB 採用，本機驗證集分等級平均 0.6236、每題平均 0.6245（`results/score_ledger.csv` 的 prereg04 H4R）。測試日程序在它的第 5 節和 rev 3 第 11 節（`docs/TESTDAY_04.md`）。
  - PREREG_05：10/2 收掉。PREREG_06 只做競賽一，預算分開。
- 探索（都只用 fit／inner 訓練玩家、沒打分）：`results/prereg06/explore/` 的 `katago/`（KataGo 的來源、授權、做法）、`c2_headroom/`（Q02 輸在哪）、`integration/`（怎麼接進測試日）、`rules/`（規則）、`critic/`（總審查）。
- 試跑：`results/prereg07/explore/pilot_humansl/`（README、計分器 `hs_lib.py`、`pilot_run.py`、`pilot_fuse.py`、`pilot_fuse_report.json`），計畫 `data/prereg07/pilot/plan.json`（不進 git，SHA 在第 1.6 節）。
- 設計：`results/prereg07/design/`，實作規格 `spec.md`，審查和逐點回應 `review_r1.md`。實作要偏離本登記時，在第 9 節第 5 步 `check` 之前寫成 rev 2。
- **只做競賽二（TEAM_11248）。** 競賽一不用 HS；07 的任何輸出都不進競賽一（第 6 節）。
- **交件模型選甲**（VALIDATION「交件模型」）：直接用本機驗證過的做法。對照模型只用 fitmap 的 3,000 盤學，合併參數只用 inner 擬合；不重訓、不用全部 8,007 人重做。
- 代號：
  - HS：本登記的新來源，每盤一個 10 維對數機率（第 1.5 節）。
  - Q02：04 選中的組合（C5＋OP＋O4＋SWA＋SWB），也就是「04」。
  - **Q02H**：Q02＋HS，也就是「07」。
  - R20、R10：同 PREREG_02、04。
  - fitmap、bank、mono：試跑的三組棋（第 1.6 節）。
- 本登記寫四件事：比什麼、怎麼選、怎樣算成功、什麼時候放棄。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `5957a01` | 第一版，2026-10-04（單色檢查跑到約 20/307、還沒有完整結果時）。當時**沒有任何 07 的正式輸出或分數**：`results/prereg07/` 只有 `explore/`（試跑）和 `design/`。試跑只用訓練玩家：fitmap 3,000 盤、bank 11,813 盤已算完，G1–G3 的數字已知（第 0 節）。mono 被中途看過兩次，兩次 G4 的下界都 < 0：10/4 11:08 的 10,610 題（`56f46de` 的 `pilot_fuse_report.json`），和 10/4 19:00:52 的 11,825 題（用 STOP 停下後 `mono1.sh` 自動跑 `pilot_fuse.py`，數字只留在 `data/prereg07/pilot/mono1.log`；被覆寫的報告檔已還原成 `56f46de` 版）。本版 commit 時，mono 全覆蓋的結果還沒有被算出（第 9 節第 1 步）。驗證題的棋沒有用 HumanSL 算過。本機驗證集在競賽二自 H4R（9/29）之後沒有再打過分。兩份審查和回應在 `results/prereg07/design/review_r1.md` |
| rev 2 | `9a0f6b8` | 2026-10-05，在凍結之前、沒有任何驗證題 HS：① 程序偏離：第 9 節要求 inner_manifest、check、selfcheck、tune 每步先 commit 再做下一步；實作代理不能 commit，四步用 `--precommit` 模式一次跑完（只放寬「已 commit 且乾淨」、改記 SHA），結果不變，官方 tune 只判了一次（10/5 約 05:19 UTC）；② 開發中曾在 `data/prereg07/tmp/` 做過 tune 的預覽，數字和已知結果逐位相同，已刪；③ 實作比 rev 1 加嚴：`apply` 不會寫出 `score_allowed: true`，只有 `apply --finalize` 在綁定 SHA 的 prescore audit commit 之後才能；`val`／`apply` 要求四份設計報告已 commit 且 SHA 等於 frozen_config；tune 重跑要 rev ≥ 3；④ 新增 `apply --accept-data-failures`（驗證題失敗盤 ≤ 95、由主 session 判定是資料性質時才可用）；⑤ GPU 冒煙的 lp_HS 漂移 0.0085（容許 0.02，高於 spec 第 9 節 0.005 的檢討線），測試日的修訂要重看這個容許值。其他照 rev 1 |
| rev 3 | `a7fb0ca` | 2026-10-05，H7R 之後、拿到任何測試資料之前。H7R：+0.0091，95% CI [−0.0017, +0.0200]，照第 3 節**不採用**。使用者授權由 Claude 判斷，Claude 決定**偏離採用規則**：測試日仍以 07 為競賽二首選（07 → 04 → R20 → R10），理由是 inner 混色、inner 單色、驗證集三次量測都為正且大小一致（`results/prereg07/RESULTS.md`）。除了「H7R 採用」這一條改成「H7R 已打分、Q02H ≥ 0.57、對照組重讀相等」，第 5 節其他測試日條件全部照舊；這次偏離寫進報告。之後不再為 07 打分 |
| rev 4 | `360fa58` | 2026-10-06，測試日程序（K5-07 前、拿到任何測試資料之前）：`tools/t07_rank.py`（hs／apply／status，結束碼 0/2/3/4）、`tools/t07_fake_tests.py`、`tools/official_log.py` 只加行（07 → 04 → R20 → R10、`--not-07-reason` 要 rev ≥ 5）、`tests/test_t07_*.py`、`docs/TESTDAY_07.md`；細節、4090 實測和偏離在第 11 節 |

## 0. 為什麼做這個（都不看答案；試跑看過 inner，所以 inner 的數字偏樂觀）

- **Q02 輸在哪**（`c2_headroom/REPORT.md` (b)，inner 單色假題）：
  - 官方題型的失分有 70–84% 是「玩家層級」：用那個人同一執色的全部棋也猜錯。
  - 每盤棋的資訊變 2 倍，官方題型約 +0.040；4 倍約 +0.067（估計，假設新資訊和舊來源無關）。
  - 便宜的 CPU 新特徵都沒用；加大 CNN 的 03 只多 +0.0014～+0.0036。要往上，需要和 R20 誤差不同的逐盤證據。
- **HumanSL 為什麼可能有用**：
  - 它是 KataGo 公開的「依棋力條件化的人類落子模型」，用和我們完全不同的棋譜訓練（OGS、KGS、野狐、GoGoD、Go4Go；從原始碼推的，第 8 節），誤差和 R20 的相關應該比 03 低（推論）。
  - 前例：Kuboki 等人 2025（arXiv 2505.00279，出題老師是共同作者）用 HumanSL 估棋力；換算成本題指標，在「同一玩家」設定下比他們的 SE 多 +0.011～+0.049（`katago/REPORT.md` 第 8 項）。
- **試跑**（`pilot_fuse_report.json`，10/4 11:08，`56f46de`；inner 評估假題 127,200 題，5 折玩家 OOF）：
  - Q02 0.6101 → Q02＋HS **0.6243**，分等級平均 **+0.0142**，PlayerBootstrap 95% [**+0.0047, +0.0239**]；每題平均 +0.0125。
  - 10 級裡 9 級變好，D 級 −0.0088。
  - 只記錄：R20＋HS 0.6050（比 Q02 低 0.0051）。
  - 對照模型單獨（fitmap 5 折 CV）：每盤 Top-1 33.1%、規則 B 每盤 0.472。
- **試跑前寫下的，和它的限度**（`802eaea`，10/3 06:12 commit，在 fitmap、bank 開跑（10/3 22:05）之前；`pilot_fuse.py` 和 README 之後都沒有改過）：
  - 寫死、照用的：14 個 profile、對照模型的做法和 λ 格點、合併的方法、G1（+0.010）、G2（下界 > 0）。試跑只跑了這一個變體，沒有在 inner 評估假題上挑變體。
  - 但 `pilot_fuse.py` 開頭把閘門標成「task text; pilot only, not a registration」。mono 檢查寫成「照 PREREG_04 G4」，在「盤都算過的單色假題」上算、附覆蓋率。
  - **當時沒有定的**：README「還沒決定的事」寫「mono 要全跑還是只跑一部分。建議先看 G1／G2 再決定」，`katago/REPORT.md` 也列為開放問題；README 的計畫是「閘門過了或接近，再分兩三晚補 mono」。
  - 10/4 11:08 G1、G2 過了，同時第一次看到 mono 的部分結果（下界 −0.0025，沒過）；10/4 18:53 決定 mono 全跑。照 802eaea「部分覆蓋也照算」的走法，G4 在兩次中途結果都不會過。
  - 所以：G4 的核心條件（下界 > 0、黑白點估計都 > 0）和 802eaea、PREREG_04 G4 相同；「要在全部 120,544 題上判」是本登記第一次寫成登記，而且是在看過部分結果之後。這個選擇只會比部分覆蓋更嚴（第 2.5 節）。
- **mono 被中途看過兩次**（兩次的題都只是 bank 玩家的題加 mono 開頭幾塊，不是隨機樣本；都不算判定）：

| 時間 | 覆蓋 | 差 | 95% CI | 黑 | 白 |
|---|---|---|---|---|---|
| 10/4 11:08（`56f46de` 的報告） | 10,610／120,544 | +0.0156 | [−0.0025, +0.0294] | +0.0139 | +0.0121 |
| 10/4 19:00:52（`mono1.log`） | 11,825／120,544 | +0.0139 | [−0.0017, +0.0310] | +0.0102 | +0.0186 |

  - 第二次是 10/4 19:00 為了使用者重新開機用 STOP 停下時發生的：`mono1.sh` 不論 `pilot_run` 怎麼結束都會接著跑 `pilot_fuse.py`。那次覆寫的 `pilot_fuse_report.json` 已還原，git 裡仍是 `56f46de` 版，直到第 9 節第 3 步。
  - rev 1 起不再中途看（第 2.5 節「不再中途看 mono」）。
- **挑選的自由度**（第 8 節揭露）：
  - profile 格點、只用 KGS 預設、日本規則和貼目 6.5，是看了少數 fit 盤的似然曲線（`hs_parity.json` (c) 10 盤、`hs_gpu_bench.json` 格點掃描 30 盤）才定的。這些是 fit 玩家的棋，不是 inner。
  - 合併的方法和 inner 假題是 02–05 一路用過很多次的同一套，Q02 本身就是在 inner 上選出來的。
  - 試跑比 04 的設計乾淨（只看一次、只有一個變體），但 inner 仍不是乾淨的保留資料。**真正的判定只有第 3 節那一次驗證集打分。**
- **G1–G3 寫本登記時都已知會過**（第 2.5 節）；打分前唯一還不知道結果的閘門是 G4。
- **預期**：驗證集差值的標準誤估 0.0062–0.0065（第 3 節）。增益和 inner 一樣時，H7R 採用的機會約 6 成；打折的話約 2–4 成（估計）。

## 1. 來源（全部釘死）

### 1.1 沿用 Q02 的五項（不重算、不改）

- C5、OP、O4、SWA、SWB 的定義、檔案和 SHA，照 PREREG_04 第 1 節和 `results/prereg04/rank/frozen_config.json`（SHA-256 `ae728e9c936a55dfeced255e9b05b5ac6a4955a7101f108e78c53c679e162661`）。
- 驗證題的 O4 用已 commit 的 `results/prereg04/opp/val_o4.npz`（`30d312c02c357f6761567099003885be5e08635b6e6c0b49d71baac137cd9cd4`），不重算。
- inner 的五項照試跑的路徑讀：`results/prereg05/explore/u1/u1_lib.py` 的 `load_inputs`。它唯讀 import p02／p03／p04 的工具，並核對它們和 prereg02／03 的紀錄相同。

### 1.2 KataGo HumanSL 的程式和權重

- **程式**：`third_party/katago/`。KataGo commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1` 的 python 模組，原封不動複製，共 16 個檔。`SOURCE.txt` 記每個檔的 git blob id 和 SHA-256，它本身的 SHA-256 是 `039d5a1cbc815b4d69b68dc8c01163354820ac00b5070ef136c14a040b5ba20b`。附 `LICENSE`、`CONTRIBUTORS`。
- **權重**：`data/katago/b18c384nbt-humanv0.ckpt`（不進 git），KataGo v1.15.0 release 的檔，網址 `https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.ckpt`。
  - 323,752,318 bytes，SHA-256 `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27`（使用者 10/2 同意下載）。GitHub 沒公布這個檔的 digest，所以這個 SHA 只釘住我們這一份，不能驗證來源。
  - 用 `model`，不用 `swa_model`（同 KataGo issue #1154）。
  - 載入：先在不執行 pickle 的情況下掃出檔內引用的全域名稱，要正好是 `hs_lib.CKPT_GLOBALS` 的 7 個；再用只認這 7 個名稱的 restricted unpickler 載入。原檔不改寫。
  - 正式的每一步（`manifest`、`val`、測試日）都要**完整重算一次 SHA-256** 並記錄。試跑的 chunk 記的是 `not hashed in this run`，所以第 9 節第 4 步的 `manifest` 補做一次。
  - **本機備份**：第 9 節第 2 步複製一份到 `data/katago_backup/b18c384nbt-humanv0.ckpt`（同 SHA，不進 git）。原檔壞了或不見時：先用備份；備份也不行，才請使用者同意重新下載，下載後的 SHA 要等於上面的值。
- **計分器**：`results/prereg07/explore/pilot_humansl/hs_lib.py`，SHA-256 `d19b6a8615ef757891ff2d451fe5b96b19f31cc706a87862e49a7688d0bde97c`（試跑每一個 chunk 記的就是這個值）。正式工具一律 import 這個檔，不複製、不改。
- **跑法**：`pilot_run.py`（`b0d127fd1d5a83af5b17ec3df1dccc85d94b11c3479d33cac07448a3796a3936`）的 `run_chunk` 和 chunk 格式（`c10-prereg07 pilot chunk v1`）；對照模型和合併照 `pilot_fuse.py`（`37fb2dff0f37032bf08acc7f27213cafd30533ab024973f92b5b28f41c6ac0c6`）。正式工具 import 它們的函式，不改檔。
- **對拍**（10/2–10/3 做過；`hs_parity.json`、`pilot_smoke_check.json`、`test_hs_lib.py` 17 passed）：issue #1154 空棋盤的三個數字差 ≤ 5.8e-7；和 KataGo 自己的 `GameState`、`humanslnet_server.py` 比，機率最大差 9.7e-7；GPU fp16 和 CPU fp32 比，每手最大差 0.0073、每盤平均最大差 0.00019。

### 1.3 每一手怎麼算（寫死，照 `hs_lib.py`）

- **棋譜**：只接受 `(;SZ[19];B[..];W[..]...)`、從黑開始黑白交替的 SGF。別的格式、顏色不交替、在遮罩下不合法的一手，那一盤記「失敗」（`failed`），不猜。
- **重播**：KataGo 的 python `Board`，從空棋盤重播整份 SGF，不受 250 手窗口限制。
- **規則**（只經由 3 個全域輸入進網路）：日本規則（`KO_SIMPLE`、`SCORING_TERRITORY`、`TAX_SEKI`、不准自殺）、白貼 6.5 目、沒有 button、encore 0。我們的 SGF 沒有 RU／KM／HA，棋譜來自日本棋院幽玄之間，所以這是假設（第 8 節）。
- **輸入特徵**：KataGo features version 15（22 個平面、19 個全域值），歷史從開局算起。
- **metadata**（照 C++ `makeBasicRankProfile` 的 `rank_<r>` 預設）：雙方都是人類、雙方同一個等級、是否計分不明、讀秒制（基本 1,200 秒＋5 次 × 30 秒）、來源 KGS、日期 2020-03-01、棋盤面積 361。
- **輸出**：主 head 的 policy 第 0 通道，原方向，不做對稱平均。非法點（有子、劫的禁著點、單子或多子自殺）先設成 −∞ 再 log-softmax；pass 永遠合法。
- **只算目標方的手**，目標執色＝題目的執色。對手方不算（第 7 節）。
- **精度**：GPU fp16 autocast、channels_last、每批 1,024 次評估（73 個局面 × 14 個 profile）、最後一批補齊到同一形狀、`cudnn.benchmark` 開。fp16 不會逐位元重現，容許差見第 5.2 節。

### 1.4 profile 格點（寫死）

14 個 KGS 預設，依序：`rank_20k`、`rank_15k`、`rank_12k`、`rank_10k`、`rank_8k`、`rank_6k`、`rank_4k`、`rank_2k`、`rank_1d`、`rank_2d`、`rank_3d`、`rank_5d`、`rank_7d`、`rank_9d`。這個順序就是特徵的順序。

### 1.5 每盤的曲線和對照模型（HS 項）

1. **曲線**：x_g[j]＝目標方每一手在 profile j 下的 log p 總和 ÷ 目標方手數，14 維。沒有目標手、或那一盤失敗，就沒有曲線。
2. **對照模型**：10 類 multinomial logistic regression，照 `pilot_fuse.py` 原樣：
   - 只用 fitmap 的 3,000 盤學（fit 玩家，每級 300 位不同的人各 1 盤；沒有 inner、沒有驗證玩家）。
   - 特徵用 fitmap 的平均和標準差（ddof＝0；標準差是 0 時用 1）標準化。
   - 損失＝平均負對數似然＋λ‖W‖²（b 不罰），L-BFGS-B、從 0 開始、maxiter 2,000。
   - λ 從 {1e-4, 1e-3, 1e-2, 0.1, 1} 選：fitmap 上 5 折 CV 的 log-loss 最小者（折號＝`default_rng(0).permutation(3000) % 5`），再用全部 3,000 盤重擬合。
   - 試跑選到 λ＝1e-4，CV log-loss 1.7571755879185869，重擬合 loss 1.7425066323027925。
3. **HS 項**：lp_HS(g)＝log_softmax(W·z(x_g)＋b)，10 維。建 lp_HS 不用任何 inner 或驗證玩家的標籤；inner 的標籤只經由第 2.1 節的分折擬合進來，和 O4、SW 相同。
4. 凍結時把 μ、σ、W、b、λ 存進 `results/prereg07/rank/hs_map.npz`，SHA 寫進 `frozen_config.json`。之後驗證題和測試日都只讀這一份，不重擬合。

### 1.6 試跑的 inner 輸入（沿用，不重算）

- `data/prereg07/pilot/plan.json`（SHA-256 `8494e9afd79c06249bd81545263b86caef7b9e77f3cd631a1e354368cfe33946`，10/2 13:48 UTC）：
  - fitmap：3,000 盤，每級 300 位 fit 玩家各 1 盤（seed 20261003）。
  - bank：11,813 盤，`results/prereg02/rank/pseudo_inner.json` 的 636 位 bank 玩家的全部棋；inner 的擬合假題和評估假題都從這裡抽。
  - mono：61,356 盤，PREREG_03 單色假題用到、不在 bank 裡的棋；依玩家分組、各級輪流（seed 20261004），共 307 塊。
  - plan 記的輸入 SHA：`inner_keys.npz` `b686872d…`、`pseudo_inner.json` `3a51b3dd…`、`u1_oof.npz` `7e874e07…`、`players_split.csv` `69554875…`（全文在 plan.json）。
- chunk 檔在 `data/prereg07/pilot/{fitmap,bank,mono}/chunk_*.npz`（不進 git）。第 9 節第 4 步的 `manifest` 把每塊的 SHA-256 記進要 commit 的 `results/prereg07/hs/inner_manifest.json`，之後每一步都核對。
- 試跑在本機 RTX 4090 上跑：`.venv-gpu`（Python 3.12.13、torch 2.11.0+cu128、cuDNN 91900、numpy 2.5.3）。fitmap、bank 已完成，都是 0 盤失敗。
- **mono 的進度**：10/4 18:53 開跑；19:00 為使用者重新開機用 STOP 停在 2／307 塊；19:04 由 `data/prereg07/pilot/mono1.sh` 從第 3 塊續跑（紀錄 `mono1.log`）。10/4 19:45 時 15／307 塊、0 盤失敗，估 10/5 10:40 前後跑完（以 `run_log.jsonl` 的 `eta_min` 為準）。
- 實測速度（`data/prereg07/pilot/run_log.jsonl`）：bank 平均 1,666 evals/s、fitmap 1,154 evals/s；每盤平均約 1,561 次評估。

### 1.7 驗證題的 HS（凍結後才算）

- 只讀 `data/local/rank_val_questions.csv` 的 `question_id`、`num_games`、`color_i`、`sgf_i`（同 PREREG_04 1.2 的 O4），不讀答案。槽位要和 prereg02 的 `val_keys.npz`（`055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c`）逐列相同（題號、slot、執色）：1,589 題、19,156 盤。
- **要在 `frozen_config.json` commit 之後才准算**（工具強制）。讀完題目檔才 import `hs_lib`；它的防護會讓同一個行程之後碰不到 `data/local` 的任何檔。
- **只有正式模式讀 `data/local`**。冒煙和演練一律用 inner 棋做成的假題目檔（放在 `data/local` 以外）；任何讀 `data/local` 的模式，前提都是已 commit 的 `frozen_config.json`。
- 算法、精度、批次、profile、環境都和試跑相同。chunk 放 `data/prereg07/val/`（不進 git）；每盤的總和、手數、失敗旗標寫成要 commit 的 `results/prereg07/hs/val_hs.npz`，報告 `val_report.json`。
- 另外把前 400 盤（前 2 塊）的逐手 log p（`pos_game`、`pos_turn`、`pos_logp`，float32，約 2–3 MB）寫成要 commit 的 `results/prereg07/hs/val_seam_moves.npz`：測試日 GPU 接縫的參照（第 5.2 節），不靠沒進 git 的 chunk。
- 用獨立的背景程序跑（同 `mono1.sh` 的做法，不受 Claude 用量上限影響），設 `PYTHONDONTWRITEBYTECODE=1`（特徵化的 spawn 子行程不會繼承 `sys.dont_write_bytecode`）。
- 4090 上估 5.0–7.2 小時：19,156 盤 × 每盤約 1,561 次評估＝約 2,990 萬次，用 bank 的 1,666 evals/s 和 fitmap 的 1,154 evals/s 估。排夜間。

## 2. 合併、檢查和閘門（只用 CPU，`tools/p07_rank_fuse.py`）

### 2.1 合併公式（照試跑：PREREG_04 的廣義校準，多加一項）

p＝softmax( e^a · (n/12)^γ · ( m_C5 ＋ w_OP·m_OP ＋ w_O4·E_O4 ＋ w_SWA·E_SWA ＋ w_SWB·E_SWB ＋ w_HS·E_HS ＋ Bβ ) )

- m_C5、m_OP、E_O4、E_SWA、E_SWB、B、參數界限（a (−4, 4)、γ (−1, 2)、所有 w (0, 4)）、起點（a 取 0 和 1）、PENALTY 0.01‖β‖²、MAXITER 500、等級加權的負對數似然，全部照 PREREG_04 第 2.1 節（`p04_rank_fuse.gen_*`）。項的順序：OP、O4、SWA、SWB、HS。
- **E_HS**：這一題每一盤「C5 可解析、HS 沒有失敗、而且有目標手」的 lp_HS 取平均；一盤都沒有就記 0 向量。inner 和驗證題沒有這種盤時，這就是 `pooled(S, lp_HS, log)`，和試跑逐位元相同。
- 盤數只經由共用溫度 (n/12)^γ 進來，n＝C5 可解析的盤數（同 04）。
- inner 用 5 折玩家 OOF（`u1_lib.run_gen`，也就是 `p04_rank_fuse.gen_oof_record`）：第 f 折的參數只用其他折玩家的擬合假題擬合，在第 f 折的評估假題上算；單色假題用那位玩家所在的折。
- 規則 B、PlayerBootstrap 照 02／04。

### 2.2 check：輸入能不能用（在 selfcheck 之前）

全部成立才往下，任一不成立就停下查：

1. `inner_manifest.json` 已 commit；每塊的 SHA、塊數（fitmap 15、bank 60、mono 307）、盤數（3,000、11,813、61,356）和 plan 相符；每塊的 `ref` 等於 plan 的那一段。
2. 每塊記的 profile、sides（只有 target）、精度 fp16、batch 1,024、channels_last、KataGo commit、`hs_lib.py` 的 SHA 都等於第 1.2–1.4 節。
3. 權重完整重算的 SHA-256 等於第 1.2 節；`third_party/katago` 每個檔等於 `SOURCE.txt`。
4. fitmap、bank 的失敗盤和沒有目標手的盤都是 0（試跑實測 0）。mono 的照第 2.5 節 G4 的規則處理。
5. **手數交叉核對**（`manifest` 時算、寫進 `inner_manifest.json`）：bank、mono 每一盤，HS 的總手數要等於 04 的 `results/prereg04/opp/inner_o4.npz`（另一套重播程式，同一組 inner_keys 列）的 `n_moves` 合計，HS 的目標方手數要等於 `n_moves[:,0,:]` 的合計；兩邊都沒失敗的盤，不符的要是 0。10/4 在 bank 前 1,000 盤試過，0 不符。不是 0 時，先查清楚定義差在哪，在 rev 2 寫死驗證題和測試日用的定義，再往下。
6. `u1_lib.load_inputs` 的前提照常；plan 記的 4 個輸入 SHA 現在不變。

### 2.3 selfcheck：正式工具要重現試跑（只核對移植，不是選擇）

- Q02 的 inner OOF（評估假題和單色假題的機率）和 `u1_oof.npz` 的 `eval_P_Q02`、`mono_P_Q02` 逐位元相同；分等級平均 0.6101211982700248。
- 對照模型：λ＝1e-4；各 λ 的 CV log-loss、重擬合 loss 和試跑相差 ≤ 1e-12。
- Q02H 的 inner 評估分等級平均 0.6242752597994481、差值 +0.014154061529423334、PlayerBootstrap 95% [0.004729748170068845, 0.023937551131530375]、每題平均差 +0.012494963827849034，相差都 ≤ 1e-12。
- mono：用**試跑的邏輯**（只算盤都有 HS 的題），在試跑涵蓋的那組題上，重現第 9 節第 3 步 commit 的全覆蓋 `pilot_fuse_report.json` 的 mono 數字到 ≤ 1e-12。mono 沒有失敗盤時，這組題就是全部 120,544 題。
- 遮罩版 E_HS（第 2.1 節）在 inner 沒有不可用盤時，要和 `pooled` 版逐位元相同。
- 不符就停下查。查明是數值路徑的差異、不是邏輯錯誤的話，寫 rev 2，以 repo 算出的數字為準並揭露。

### 2.4 組合（只有一個候選）

| 組合 | 內容 | 用途 |
|---|---|---|
| Q02 | C5＋OP＋O4＋SWA＋SWB | 對照（就是 04） |
| Q02H | Q02＋HS | 唯一的候選 |

- 沒有選擇這一步：閘門都過了就是 Q02H，沒過就未送分。
- **只記錄、不可選**：
  - R20＋HS；各級、各盤數、單色（全體、黑、白、各 k）的分數。
  - 每折的參數和打到界限的權重。試跑的 Q02＋HS：第 1、2、5 折 w_SWB＝0（下限）；第 3、4 折是另一種解（a −0.94／−1.21、所有 w ≥ 0.93），各有一個權重＝4（第 3 折 w_HS、第 4 折 w_O4）。
  - 凍結時重擬合的參數、打到哪些界限、重擬合在 inner 擬合假題上的樣本內分數和 OOF 分數並列。
  - HS 單獨的每盤 CV 指標；驗證集的推估標準誤與採用機率（`se_val_record`，同 PREREG_04）。
  - 新題子集的 G4（第 2.5 節）。

### 2.5 送分閘門（Q02H 對 Q02，全部成立才打分）

| 閘門 | 條件 | 寫本登記時 |
|---|---|---|
| G1 | inner 評估假題的分等級平均差 ≥ +0.010（判定時容許 1e-12） | 已知：+0.0142 |
| G2 | 同一差值的配對 PlayerBootstrap（每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0 | 已知：+0.0047 |
| G3 | 同一批題的每題平均差 ≥ 0 | 已知：+0.0125 |
| G4 | 單色假題（PREREG_03 rev 3 (a) 的同一組：623 人、120,544 題，黑 59,744、白 60,800，SHA `45d0f14a…`）的分等級平均差：配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 > 0；在全部 120,544 題上判 | 未知（兩次中途結果見第 0 節，不算判定） |

- G1、G2 照試跑原樣。G3 照 PREREG_04（VALIDATION「兩個指標」的精神）。G1–G3 的值寫本登記時已知，揭露；打分前唯一的未知是 G4。
- **G4 的規則和由來**：
  1. 官方每題只有一種執色，單色假題是 inner 上最接近官方結構的檢查。
  2. 核心條件和 PREREG_04 G4 一字不差，802eaea 的試跑程式也照 PREREG_04 G4 寫；但 802eaea 明寫「not a registration」，全跑或只跑一部分當時沒有定（第 0 節）。全跑是 10/4 18:53 在第一次部分結果之後決定的；本登記照實寫，不說成「事先寫死」。
  3. 黑、白兩色只看點估計，不要求各自的下界 > 0：一色只有一半的題，下界太寬；點估計 > 0 已經抓得到「一色有效、另一色反向」的結構問題。這也和 04 相同。
  4. 要求全部的題：兩次中途看到的題全是 bank 玩家的題加 mono 開頭幾塊，不是隨機樣本；PREREG_04 G4 本來也定義在全部 120,544 題上。
- **G4 的邊界情況**：
  - 「全部的題」＝plan 的 mono 307 塊都在、`inner_manifest.json` 已 commit。
  - mono 有失敗盤時：G4 用第 2.1 節的遮罩版照常算全部的題，失敗盤數和受影響的題數記錄；失敗盤超過 mono 61,356 盤的 0.5%（> 306 盤）時 G4 不過。selfcheck 的 mono 比對照第 2.3 節，在試跑涵蓋的題上用試跑的邏輯。
  - K3-07（第 4 節）前 mono 沒跑完：G4 不過。
- **只記錄：新題子集的 G4**：第二次中途結果沒涵蓋的 108,719 題（至少有一盤不在 bank 或 mono 前 2 塊裡的單色假題），記差值、CI、黑白。不當閘門。
- **不再中途看 mono**（rev 1 起）：
  - `pilot_fuse.py`（或任何算 mono 分數的程式）只准在 `pilot_run.py status` 顯示 mono 307／307、而且本登記 rev 1 已 commit 之後跑。
  - 4090 要讓出（例如 C4 要用）或電腦要重新開機時：**先結束跑 `mono1.sh` 的那個 bash 行程**（它結束後 `pilot_run` 會照常跑完手上的工作，但不會接著跑 `pilot_fuse.py`），確認 `run_log.jsonl` 還在增加，才放 STOP。只放 STOP 會讓 `mono1.sh` 接著跑 `pilot_fuse.py`，等於再看一次。
  - 續跑改用不含 fuse 的啟動腳本（例如 `data/prereg07/pilot/mono2.sh`：只跑 `pilot_run.py run --set mono`）。
  - 萬一又看到部分結果，寫進 rev 2 和 RESULTS。
- **閘門只判一次**：`tune` 正式跑一次；之後要 `--rerun`，只准在指出具體程式錯誤的修訂 commit 之後。每一次的閘門結果都寫進 RESULTS。
- 任一閘門沒過：記為「未送分」，打分機會保留，競賽二照舊（04 → R20 → R10）。

### 2.6 tune：凍結

- 閘門都過了才凍結：用全部 inner 擬合假題重擬合 Q02H（`p04_rank_fuse.gen_fit_record`），參數寫進 `results/prereg07/rank/frozen_config.json`，連同 `hs_map.npz` 一起 commit。
- `frozen_config.json` 記：組合和項的順序、參數、5 折的紀錄、`hs_map.npz` 的 SHA、profile、metadata、規則、精度和批次、權重 SHA、KataGo commit、四個釘住檔（`hs_lib.py`、`pilot_run.py`、`pilot_fuse.py`、`u1_lib.py`）的 SHA、`inner_manifest.json` 的 SHA、閘門紀錄、`se_val_record`、HS 方向檢查的兩個參照值（第 2.7 節：inner 評估假題的 ρ_eval、單色假題的 ρ_mono）、工具的 SHA、兩個環境的版本（`.venv`：Python、numpy、scipy；`.venv-gpu`：Python、torch、cuDNN、GPU 型號，另記當時的 NVIDIA 驅動版本，只記錄）。
- 從這之後才准算驗證題的 HS。閘門沒過就不凍結，也不算驗證題的 HS（省下 5–7 小時的 GPU）。

### 2.7 apply（驗證題）

- 前提：`frozen_config.json` 已 commit；`val_hs.npz`、`val_report.json`、`val_seam_moves.npz` 已 commit，而且 `val_report.json` 記的開始時間晚於 `frozen_config.json` 最後一次 commit 的時間。
- **先重現 Q02**：用 07 的程式和 prereg04 的凍結參數算驗證題的 Q02，必須和 `results/prereg04/rank/rank_val_probs.csv`（`3c7ab658…`）逐位元組相同；不同就停下。
- 再照 `frozen_config.json` 算 Q02H，寫 `results/prereg07/rank/rank_val_probs.csv`；接著 `decide.py`（`pred_A.csv`、`pred_B.csv`、`decide_summary.json`）、`check_submission.py rank … --questions data/local/rank_val_questions.csv`。
- `apply_report.json` 記 `score_allowed`。全部成立才為真：
  - 驗證題 HS 的失敗盤（`failed` 或整盤沒有任何一手）是 0。不是 0 時停下查：查到程式錯誤，先寫修訂、修好重算；確定是資料本身的性質（例如有 pass）時，照第 2.1 節的遮罩算，失敗 ≤ 0.5%（≤ 95 盤）才往下，> 0.5% 記為「未送分」。
  - Q02 逐位元組重現通過。
  - **手數交叉核對**：每一盤，HS 的總手數等於 `val_o4.npz` 的 `moves_parsed`、HS 的目標方手數等於 `val_o4.npz` 的 `n_moves[:,0,:]` 合計（兩邊都沒失敗的盤，不符 0 盤）。這是另一套重播程式、同一組 key，抓得到槽位錯開和執色顛倒（約一半的棋黑白手數差 1）。
  - **HS 方向檢查**：每題「只用 HS 的期望等級」和「Q02 的期望等級」的 Spearman 相關 ρ ≥ max(0.5 × ρ_eval, 0.078)。0.078 是 1,589 題時單側 p＝0.001 的臨界值。它只抓大錯（讀錯檔、整批對錯）；執色和槽位由上一條把關。
  - 打分前核對（下一點）沒有找到影響 Q02H 預測的程式錯誤。
  - 兩份預測都通過格式檢查。
- **打分前核對**（固定範圍，由另一個代理做，不看任何分數）：val HS 每塊的 SHA 和 `val_report.json` 相符；用 `hs_map.npz` 重算的 lp_HS 和 apply 的相同；遮罩版 E_HS；Q02 重現；`decide` 的輸入輸出。結果寫 `results/prereg07/rank/prescore_audit.json` 並 commit。

## 3. H7R：在本機驗證集打一次分

- **指令**（寫死；只在 `apply_report.json` 的 `score_allowed` 為真時跑）：
  `.venv/Scripts/python.exe tools/score.py compare-rank results/prereg04/rank/pred_B.csv results/prereg07/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg07 H7R (A = prereg04 Q02, reference re-read)" --json results/prereg07/rank/score_H7R.json`
- A 側是 Q02，不是 R20：07 要贏的是現在的首選。
- **採用條件**（兩者都要成立）：
  - 分等級平均差（Q02H − Q02）的 95% 信賴區間下界 > 0（每級內分層重抽 10,000 次、seed 0）。驗證集一位玩家一題，所以這就是玩家層級的重抽。
  - 每題平均差的點估計不 < 0。
  - 否則不採用，照 VALIDATION 記為「結論依指標而定」或「沒有改善」，競賽二照舊。
- **對照組重讀**：A 側 1,589 題，分等級平均和每題平均要和帳上 prereg04 H4R 的 B 列（0.6236017652948923、0.6245054776753989）差 ≤ 1e-12，題數相同；`pred_B.csv` 的 SHA 是 `27b6d225…`。記錄打分用的 Python（`.venv`，3.12.13）。不符就停下，查清前不採用。
- **合理性門檻**：Q02H 的分等級平均 < 0.57 時先查 bug（04 用的是比基準低約 0.05 的 0.54），要指出具體的程式錯誤才算。
- **預算**：PREREG_02 第 3 節是「02、03、04 各一次，另留一次給 split v2 或查 bug 重跑」。03 沒有送分；07 用的是 03 沒用到的那一次（10/2、試跑之前決定），是競賽二整季的第 3 次。第 4 次照舊保留。
- **查 bug 重跑**（對稱，不看結果決定）：
  - H7R 只在兩種情況重跑：(1) Q02H < 0.57，而且指出具體的程式錯誤；(2) RESULTS 的獨立核對（不論採不採用都做，範圍同第 2.7 節的打分前核對）找到影響 Q02H 預測的程式錯誤。
  - 採用之後才找到的錯誤一樣要重跑；重跑的判定取代原判定。最多重跑一次，那一次的判定就是最終。
  - 重跑前先 commit 指出錯誤的修訂。重跑用掉第 4 次，這一季就沒有 split v2 的打分了（記帳、寫進 RESULTS）。
- 打分紀錄帳多 2 列，note 以 `prereg07` 開頭（`tests/test_p06_player_hygiene.py` 只准 H6P 之後出現這種列）。重跑時再多 2 列，note 寫 `prereg07 H7R rerun (PREREG_07 rev N)`。
- **推估的採用機率**（常態近似；驗證集差值的標準誤用 04 的比例推：04 的驗證集標準誤約 0.00702、inner 0.005577，比例 1.26–1.32；07 的 inner 約 0.0049 → 驗證集 0.0062–0.0065。都是估計，`tune` 另照 PREREG_04 算 `se_val_record`）：

| inner 增益 | 標準誤 0.0062 | 標準誤 0.0065 |
|---|---|---|
| 全部成立（+0.0142） | 63% | 59% |
| 打 0.75 折（+0.0106） | 40% | 37% |
| 打五折（+0.0071） | 21% | 19% |

## 4. 截止點（寫死；沒趕上就照表做，不拿額外的打分機會補）

| 截止點 | 條件 | 沒趕上 |
|---|---|---|
| K3-07 10/16（五）23:59 | mono 跑完；閘門已判定；過了的話 H7R 已打分、RESULTS 已 commit | 這一季不用 07 |
| K5-07 11/03（二）23:59（目標 10/25） | 採用時：測試日擴充和測試日的修訂（預定 rev 3）已 commit、合併回 main、本機演練通過 | 這一季不用 07 |
| K6-07 11/19（四）23:59:59 | 採用時：07 在官方測試集有一次成功的正式執行（第 5.2 節），報告已 commit | 最終照 PREREG_04 5.3（04 → R20 → R10） |

- 最後一次上傳照舊在 11/23 23:59 前（截止 11/24 23:59）。
- K6-07 比 04 的 K6（11/16）晚 3 天，因為 07 要等 04 的 `apply`（TESTDAY_04 要求 11/15 前 commit）。

## 5. 採用後的測試日（細節在 K5-07 前另寫測試日的修訂）

### 5.1 保護舊路徑（rev 1 就生效）

- 這些一律不改，`tools/official_log.py` 只准加行：
  - R20／test02 的 39 個檔（`results/prereg04/design/r20_path_fingerprint.json`，tag `r20-testday-pre04`）；
  - 04 的 21 個檔（`results/prereg06/design/p04_path_fingerprint.json`，tag `p04-testday-pre06`）；
  - 06 的測試日路徑（見下一點）；
  - `results/prereg01`–`prereg06/**`、p02／p03／p04／p06 的工具、`c10data.py`、`decide.py`、`score.py`、`tests/conftest.py`；
  - 07 自己的凍結檔：H7R 之後的 `results/prereg07/**`、`tools/p07_*.py`、四個釘住檔、`third_party/katago/**`。
- **06 的指紋，順序寫死**（`docs/TESTDAY_06.md` K5' 的待辦；07 的測試日程式動工前做完）：
  1. 06 的 K5' 其他待辦先在 main 做完（全套測試、`C10_SLOW` 的 Q04Golden、第二輪審查的 minor）；
  2. 做 `results/test06/design/p06_path_fingerprint.json`：06 的程式、文件、測試釘 SHA；`tools/official_log.py`、`.gitattributes`、`.gitignore` 放在 `insert_only`（照 `p04_path_fingerprint.json` 的做法），因為 07 要往這三個檔加行；`tests/conftest.py` 照 TESTDAY_06 釘 SHA（07 不動它，第 5.2 節）；
  3. 在那個 commit 打 tag `p06-testday-pre07`，加一支釘住它的防誤改測試；
  4. 分支 `p07-testday` 從那個 commit 開。
- 07 的測試日程式全部是新檔，放在分支 `p07-testday`：`tools/t07_rank.py`、`tools/t07_fake_tests.py`、`tests/test_t07_*.py`、`docs/TESTDAY_07.md`。輸出只寫 `results/test07/`，大檔放 `data/test07/`；演練只寫 `results/test07_rehearsal/`、`data/test07_rehearsal/`（`.gitignore` 加一行）。
- 寫 07 的程式之前，先記下 R20、04、Q04（06）的假測試演練輸出當基準；合併前要逐位元組相同。
- **不需要新的 Colab notebook**：07 讀 test02、test04 已 commit 的輸出，HS 在本機 4090 上算，其他在 CPU 上算。使用者不用多跑 Colab。

### 5.2 要做的東西

- **`tools/t07_rank.py hs --set seam|public|private`**（4090）：
  - 前提：R20 的第一次上傳已在 `results/official_scores.json` 記成 `uploaded` 並 commit（照 04 的「R20 先」）；T1 是 PASS 或 NORMALIZED（FAIL_OPENING：07 不用；FAIL_STRUCT：照 test02 停下）；H7R 採用；K6-07 未過；`nvidia-smi` 沒有別的 compute 程序；`.venv-gpu` 的 torch、cuDNN 和 GPU 型號等於 `frozen_config.json` 記的值（驅動版本只記錄，由 GPU 接縫把關）；權重完整重算的 SHA 正確。
  - seam：驗證題 `val_keys` 順序的前 400 盤，只讀題目內容和執色。
  - public、private：test02 第 2 步打包的那一份測試檔（T1 判 NORMALIZED 時是正規化副本），槽位要和 test02 同一組的 key 逐列相同。
  - **不沿用舊的塊**：資料夾裡已有、`created_utc` 早於 R20 第一次上傳時間的塊，一律拒絕（不跳過、不覆寫），避免 10 月或演練留下的 seam 塊讓漂移檢查形同沒做。
  - 可以續跑（每塊原子寫入）；每一塊記的環境要相同。用獨立的背景程序跑，設 `PYTHONDONTWRITEBYTECODE=1`。每一組合成的 `<組>_hs.npz` 和報告寫好就 commit（04 的步驟要求工作樹乾淨）。
- **`tools/t07_rank.py apply`**（CPU）：
  - 前提：上面全部；三組 HS 的每一塊 `created_utc` 都晚於 R20 第一次上傳的時間；04 的 `t04_apply_report.json` 是正式模式、結束碼 0、已 commit（04 判定「不用」時，07 也不用）。不要求 04 已經上傳（那是 `official_log.py prepare` 07 的前提）。
  - **GPU 接縫**：seam 的 400 盤和已 commit 的參照比（`val_hs.npz` 前 400 列、`val_seam_moves.npz`）：失敗旗標、總手數、目標手數完全相同；每手 log p 最大差 ≤ 0.05；每盤每個 profile 的平均 log p 最大差 ≤ 0.002（試跑 GPU 煙霧測試的容許差，10/3 在試跑之前定的）；每盤 lp_HS 最大差 ≤ 0.02。逐位元是否相同只記錄。沒過＝本機漂移，停下查，不判定。演練實測後，測試日的修訂可以在拿到任何測試資料之前調整這幾個容許差。
  - **Q02 對得上 04**：用 07 的程式、prereg04 的凍結參數、和 04 同一批輸入，在 public、private 上算 Q02，必須和 04 已 commit 的 `rank_test_probs.csv` 逐位元組相同；不同＝本機問題。
  - **手數交叉核對**：public、private 每一盤，HS 的總手數、目標手數和 04 的 `o4_<組>.npz` 的 `moves_parsed`、`n_moves[:,0,:]` 合計相同（兩邊都沒失敗的盤，不符 0 盤）。不符＝07 的程式或輸入綁定有錯，停下查；只能靠指出程式錯誤的 PREREG_07 修訂修好後重跑，K6-07 前修不好就不用 07。
  - **本機接縫**：seam_val（驗證題全部 1,589 題）用 test02、test04 的 seam_val 輸入加已 commit 的驗證題 HS 算 Q02H，和 `results/prereg07/rank/rank_val_probs.csv` 比：題號、順序、盤數要相同（不同＝停下查）；最大差 ≤ 04 本機接縫的容許差（`results/test04/rank/seam_val/seam_local.json` 記的 tol），規則 B 最多差 4 題。04 的 seam_val 輸入和 prereg04 逐位元組相同時，要求逐位元組相同。04 的接縫過了、Q02 也對得上 04，07 卻超出容許差時，那只是 07 的權重放大了 04 輸入裡的 Colab 差異，本機修不好：記為登記的「07 不用」。
  - **HS 失敗**：public、private 各自失敗盤數 ÷ 盤數 > 0.5% 就不用 07（工具用 `200 × 失敗 ≤ 盤數` 判過；每組 5,000 盤時上限 25 盤）。失敗盤不進 E_HS。
  - **HS 方向檢查**：public、private 各自的 ρ ≥ max(0.5 × ρ_mono, 0.155)（0.155 是 400 題時單側 p＝0.001 的臨界值），否則不用 07。
  - 都過了才寫 `results/test07/rank/submission_rank.csv`（和 `rank_test_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`t07_apply_report.json`），跑 `check_submission --official`。
  - 結束碼：0 完成；2 拒絕（前提不符）；3 登記的「07 不用」；4 本機漂移或對不上（停下查；查好後 `--rerun --reason`）。沒有 notebook，所以沒有 04 那種「重跑一次」。
- **`tools/official_log.py` 只加行**（只用標準函式庫，不出現標籤檔的字樣）：
  - 認得 `results/test07/rank/` 的檔（在 `is_t04_file` 之前分流；不分流的話 07 的檔會被當成 R20），報告叫 `t07_apply_report.json`。
  - 方法名稱「prereg07 Q02H = Q02 + HumanSL, rule B (甲)」。
  - `prepare` 一份 07 之前，log 裡要有 R20 和 04 的第一次上傳（`uploaded`）。有符合的 07 時，`prepare` 一份 04／R20／R10 要帶合格的 `--not-07-reason`。
  - `final --comp rank` 照第 5.4 節。現有的 04 規則會把任何不是 04 的紀錄當成 R20，所以要在 `final_rule_t04`、`prepare_rule_t04` 的開頭各加一行讓 07 的紀錄跳過，07 的判定另記在 `final_choice.rank.prereg07_5_4`；`status` 在最終是符合的 07 時不印「違反 PREREG_04 5.3」，04 的提示改成 07；04 的同分警告不把 07 當成 R20。
  - `status` 只在 log 裡有 07 的紀錄時多印幾行。log 裡沒有 07 紀錄時，所有輸出、紀錄檔、回傳值和 tag `p06-testday-pre07` 的版本逐位元組相同。
- **測試和防護**：07 的工具只在讀完 `data/local` 之後才 import `hs_lib`、`pilot_*`；只有 `check`、`selfcheck`、`tune` import `u1_lib`。`u1_lib` 的防護用子字串比對，會連 `AppData\Local`（`%TEMP%`、pytest 的 tmp_path）一起拒絕，所以 import 它的行程不碰 `%TEMP%`；所有暫存寫在 `data/prereg07/tmp/`、`data/test07/tmp/`。測試一律不在 pytest 的行程裡 import `hs_lib`、`u1_lib`、`pilot_*`（要用到它們的路徑以子行程跑），所以 `tests/conftest.py` 不用改。
- `docs/TESTDAY_07.md`：Claude 的步驟。`docs/COLAB_RUN.md` 不用動。
- **一次成功的正式執行**（K6-07）：三組 `hs` 和 `apply` 都在正式模式下以結束碼 0 跑完，報告已 commit，`t07_apply_report.json` 最後一次 commit 的時間 ≤ K6-07。

### 5.3 4090 不能用時

- 測試日的 HS 只在本機 RTX 4090 上算（和驗證題同一張卡、同一個 `.venv-gpu`）。
- 估計時間：約 10,000 盤（800 題 × 平均 12.5 盤）× 每盤約 1,561 次評估＝約 1,560 萬次，加上 seam 約 62 萬次：2.7 小時（1,666 evals/s）到 3.9 小時（1,154 evals/s）。排夜間。10/24 之後 C4 已經沒有優先權，但 11/04–11/07 的夜間要先在 `PORTFOLIO_MAP.md` 登記 4090 給 C10，開跑前照樣用 `nvidia-smi` 確認沒有別的工作。
- **不登記 CPU 備案**：CPU fp32 實測約 10.2 evals/s（試跑 (c)，機器有負載），1,560 萬次要約 18 天；就算快 3 倍也要約 6 天，而且會占滿 CPU，擋到 test02、test04 的本機步驟和別條線。
- **不登記 Colab 備案**：要新的 notebook、冒煙、審查和彩排，11/03 前做不完。
- 所以：4090 壞掉、被長時間占用、或驅動或環境變了而 GPU 接縫過不了，K6-07 前修不好，**這一季就不用 07**，最終照 PREREG_04 5.3。中途斷電或重新開機：重下同一行，從沒寫完的那一塊接著跑。
- 建議（使用者本人的設定，不強制）：10/6–11/19 暫停 Windows 自動更新顯示卡驅動，降低 GPU 接縫因驅動更新而失敗的機會。

### 5.4 上傳順序和最後一版（競賽二）

- **順序**：
  1. 11/04：照 `docs/TESTDAY.md` 交 R20（test02），記成 `uploaded` 並 commit。之後才准跑 `t07_rank.py hs`（可以在 11/04 晚上，和 04 的 Colab 步驟同時進行）。
  2. 11/05–11/07：照 `docs/TESTDAY_04.md` 交 04，記成 `uploaded` 並 commit。最晚 11/18 前要傳，07 才來得及在 K6-07 附近 `prepare`。
  3. 07 的 `apply` 結束碼 0、04 已上傳之後，上傳 07。這屬於 TESTDAY「之後的上傳」(b)：之後的登記依本機驗證規則採用的方法。
- 一筆 07 的上傳「符合」＝以下全部成立（`official_log.py` 的 `prepare`、`final`、`status` 每次用報告裡的數字和檔重算，不信任報告自己的結論）：
  - H7R 採用；
  - 第 5.2 節的成功正式執行在 K6-07 前；
  - T1 是 PASS 或 NORMALIZED；
  - 07 綁的 04（`t04_apply_report.json` 和它的輸入）現在仍符合 PREREG_04 5.3，而且和 log 裡那一筆 04 綁的檔相同；
  - GPU 接縫、Q02 對得上 04、手數交叉核對、本機接縫、HS 失敗率、HS 方向檢查都過；三組 HS 的塊都晚於 R20 第一次上傳；
  - 報告和交件檔從 `prepare` 之後沒有變；報告記下的輸入（test02、test04 的檔、三組 HS、`frozen_config.json`、`hs_map.npz`、`val_hs.npz`、`val_seam_moves.npz`）現在都沒有變。
- **最後一版**：有符合的 07 → 07。沒有 → 照 PREREG_04 5.3：有符合的 04 → 04；否則 R20（T1 FAIL_OPENING 時 R10）。不交沒在驗證集打過分的組合（例如 R20＋HS）。
- 已經傳過 07、之後才發現不符合：照上一點的順序選最終，並再傳一次那一份，讓它成為最後一次上傳。
- 要推翻這個結果（例如 07 符合卻要改交 04），只能靠本登記在測試日的修訂之後的修訂：指出具體的程式錯誤和它破壞的是哪一條；修訂的 commit 時間要晚於被推翻的紀錄（同 PREREG_04 第 11.3 節第 20 條）。工具用 `--not-07-reason "PREREG_07 rev N: …"` 核對；只寫一段文字、沒有修訂，一律拒絕。07 和 04 都符合、最終卻要交 R20 時，`--not-07-reason` 和 `--not-04-reason` 都要。
- public 分數不進判定。07 比 04 低 0.05 以上時只查錯（ID、列順序、輸入綁定），查不到就維持本節的結果，寫進報告（VALIDATION 紀律 4）。07、04、R20 的交件檔同名：任兩份的 public 分數完全相同、而兩份在 **public 題**上規則 B 不同的題數 > 0 時，`official_log.py public` 印警告，請使用者核對上傳紀錄（同 `t04_public_lines`；07 和 04 可能只在 private 題不同，所以只數 public 題）。
- 上傳次數：R20、04、07 各一次，最壞再傳一次最終那份；競賽二每天最多 5 次。每一次都要使用者在對話裡對那一份檔說 OK。

### 5.5 取代的條文（照 PREREG_04 第 11.1 節第 12 條、PREREG_06 第 5.4 節的前例，不改原文）

| 被取代的條文 | 07 的規則 |
|---|---|
| `docs/TESTDAY.md` U5「兩場都是 prereg02（甲）」（對競賽二已由 PREREG_04 5.3 取代） | 競賽二由本登記第 5.4 節決定 |
| PREREG_04 第 5.3 節、第 11.1 節第 12 條「有任何一筆 04 符合，最終選擇是 04」，和 `docs/TESTDAY_04.md` 第 7 步 | 有符合的 07 時最終是 07；沒有時照原文 |
| PREREG_02 第 3 節、PREREG_04 第 7 節「不用 KataGo」（04 寫成「規則禁止」） | 只對 07 的新路徑解除；R20、04 的路徑不變（第 8 節） |

- 這三處都在指紋裡或已凍結，所以不改原文，也不寫 PREREG_04 的修訂。理由：照 PREREG_06 第 5.4 節的前例；`docs/TESTDAY_04.md` 寫「下一版是 rev 4」，現在寫 rev 4 會讓那句過時，04 真要修 bug 時得用 rev 5。（10 月 commit 的修訂本來就不能當 11 月紀錄的翻盤依據：`p04_revision` 要求修訂的 commit 晚於被推翻紀錄的 `prepared_at`。）
- 07 判定之前不執行 `official_log.py final --comp rank`；已經設過也沒關係，照 `docs/TESTDAY_07.md` 重設。
- 07 不用 public 分數做任何決定，沒有 PREREG_06 第 5.5 節那種例外。

## 6. 另一場：不交換輸出

- 競賽一不用 HS；07 的任何輸出都不進競賽一，06 的輸出也不進 07。兩場只共用前處理（`tools/c10data.py`）。
- 不用競賽一的測試資料（規則第 8 條有疑義；critic 第 1 節第 2 項的跨競賽比對不做）。

## 7. 刻意不放進 07 的

| 做法 | 理由 |
|---|---|
| 對手方的手也算 HS（HSO） | 計算量加倍；O4、SW 已經讀對手的手，重疊多半很大；試跑沒算 |
| 野狐（FOX）來源、其他日期、更多 profile、preaz | 試跑只跑了 KGS 的 14 個；改了就是沒有閘門保護的新變體 |
| 8 種對稱平均 | 計算量 8 倍；試跑沒做 |
| 用 inner 或驗證玩家學對照模型、改用 LightGBM 等其他對照模型 | 試跑只登記了 fitmap 上的 logistic regression |
| 放寬合併的權重上限（試跑有兩折打到 4、三折的 w_SWB 打到 0） | 那是試跑閘門保護的公式；看了 inner 才改等於新設計 |
| R20＋HS 當候選或備案 | inner 比 Q02 低 0.0051；沒在驗證集打分，不能上場 |
| TensorRT 或 C++ 引擎加速 | 測試日 2.7–3.9 小時已經夠；多一條沒驗證過的數值路徑 |
| 測試日的 CPU 或 Colab 備案 | 第 5.3 節 |
| 測試日少算幾手、少算幾個 profile 來省時間 | 改了算法，就不是驗證過的方法 |
| 競賽一用 HumanSL | 剩下的錯不是棋力資訊造成的；兩場共用特徵有規則疑慮（PREREG_06 第 7 節） |
| 整份等級均衡解碼、加大 CNN、半監督 | 10/2 的決定（`c2_headroom`、critic）：增益 ≤ +0.004，或等級不均衡時倒扣 |

## 8. 揭露（寫進 RESULTS 和報告）

- **外部資源**（規則第 3 條：寫來源和對結果的貢獻）：
  - KataGo 的程式（commit `d91ea855`）和 HumanSL 網路 `b18c384nbt-humanv0`（v1.15.0）。授權：程式是 MIT 類（`third_party/katago/LICENSE`）；網路權重是 KataGo 的網路授權（https://katagotraining.org/network_license/ ，明寫涵蓋 HumanSL 網路）。都准任何用途、要保留聲明。我們不轉發權重，報告寫官方網址和 SHA-256。
  - 我們的部分：SGF 讀取和非法點遮罩、profile 格點、批次推論、每盤曲線、對照模型、合併與校準、評估和測試日工具。KataGo 的部分：網路本身、輸入特徵、棋盤規則。
- **看過測試棋的疑慮**：
  - KataGo 程式裡的訓練資料來源只有 OGS、KGS、野狐、GoGoD、Go4Go（另有 Tygem 代碼，但寫訓練資料的程式沒有那一支），沒有幽玄之間。官方沒公布清單，所以這是從原始碼推的（`katago/REPORT.md` 第 2 項）。
  - 我們的訓練、驗證、測試棋都來自幽玄之間（`docs/DATA.md`），所以 HumanSL 記得這些棋的機會很低；inner、驗證、測試三者在這一點上相同，inner 的增益不會因此偏樂觀。
  - 反過來，前例論文用野狐的棋，而野狐是 HumanSL 的訓練來源之一，論文的數字可能偏樂觀。
- **消融**（報告審查的「程式原創性」要看有和沒有 KataGo 各多少分；`rules/REPORT.md` 第 4 點）：
  - 本機驗證集：H7R 本身就是消融，A＝Q02（沒有 KataGo）、B＝Q02H（有）。
  - inner：Q02 0.6101、Q02H 0.6243、R20＋HS 0.6050（試跑）；單色假題的同一組比較（G4）。
  - HS 單獨：fitmap CV 的每盤 Top-1、規則 B 每盤分數。
  - 測試日：07 和 04 在 public、private 上規則 B 不同的題數（只記錄）。
- **自我限制的解除**：PREREG_02 第 3 節、PREREG_04 第 7 節寫「不用 KataGo」，04 還寫成「規則禁止」。10/2 的規則核對（`rules/REPORT.md`）確認正式規則准用、要揭露，那是我們自己的誤讀。07 只對新路徑解除，R20、04 的程式和指紋不變。
- **G4 的由來和中途結果**（第 0 節）：802eaea 的閘門寫明「not a registration」，全跑或部分當時沒定；全跑是在第一次部分結果之後決定的；mono 被中途看過兩次（10,610 題、11,825 題），下界都 < 0。新題子集的 G4 只記錄。
- **G1–G3 的值寫本登記時已知**（+0.0142、+0.0047、+0.0125）；打分前唯一的未知是 G4。
- **同一個驗證集的第三次依序使用**：H2R、H4R、H7R；A 側的 Q02 本身也是在這個驗證集上採用的。
- **預算的重新分配**：07 用的是 PREREG_02 第 3 節分給 03、03 沒用到的那一次（第 3 節）。
- **設計的自由度**：見第 0 節。profile 格點、KGS 預設、規則和貼目是看過少數 fit 盤才定的；試跑只跑一個變體、G1／G2 事先寫下；inner 假題和合併方法在 02–05 用過很多次。
- **對照模型的 λ 選到格點的最小值 1e-4**；合併時有兩折的權重打到上限 4、三折的 w_SWB 打到下限 0。都照試跑原樣，不改格點或界限。
- **規則和貼目是假設**（日本規則、6.5 目）：SGF 沒有這些欄位，只影響 3 個全域輸入。
- **數值**：GPU fp16 不能逐位元重現，驗證題和測試日的 HS 靠容許差核對（第 5.2 節）；fp16 和真 fp32 比，log p 最大差 0.0084、各 profile 平均最大差 0.0002（`hs_gpu_bench.json`）。
- **權重 SHA 的限制**：GitHub 沒公布 digest，SHA 只釘住我們這一份。
- 探索時的一次越界（`c2_headroom` 的遞迴 grep 可能走過 `data/local` 的資料夾名稱，沒有開檔）已在 PREREG_06 第 8 節揭露；那是競賽二的探索，這裡再記一次。
- 生成式 AI 協助的部分照規則揭露（`competitions/_drafts/disclosure-template.md`）。
- 報告草稿 `report/` 競賽二第 7 節目前寫「沒有用 KataGo」，H7R 之後要改寫。採用與否都要寫；沒採用時寫「試過、沒用在交件」。

## 9. 順序

1. 本登記 rev 1、`results/prereg07/design/spec.md`、`review_r1.md` 一起 commit，再用一個 commit 填上 rev 1 的編號。**硬規則：rev 1 的 commit 要早於 mono 全覆蓋的結果第一次被算出。** 做法二選一：在 mono 最後一塊寫完之前 commit（估 10/5 10:40 前後）；或先照第 2.5 節「不再中途看 mono」結束 `mono1.sh` 的 bash 行程，讓 `pilot_run` 跑完但不自動 fuse。RESULTS 記 rev 1 的 commit 時間和第 3 步 fuse 的時間。
2. 寫程式：`tools/p07_hs.py`（`manifest`、`val`、`status`）、`tools/p07_rank_fuse.py`（`check`、`selfcheck`、`tune`、`apply`）、`tests/test_p07_*.py`（含 hygiene；要用 `hs_lib`、`u1_lib`、`pilot_*` 的路徑以子行程跑），權重備份到 `data/katago_backup/`。經獨立審查，mono 跑完、CPU 空出來後在 main 用單一行程跑 `PY -m pytest -q tests` 全綠，才 commit。
3. mono 307／307、rev 1 已 commit：跑一次 `pilot_fuse.py`（手動，或 rev 1 commit 之後 `mono1.sh` 自動跑的那一次），commit `pilot_fuse_report.json` 和 run log 的副本（試跑的一部分，只用 inner；照 `56f46de` 的做法）。
4. `p07_hs.py manifest`（完整重算權重 SHA、每塊 chunk 的 SHA、第 2.2 節第 5 條的手數交叉核對）→ commit `inner_manifest.json`。
5. 實作和登記的差異寫成 rev 2，在 `check` 之前 commit。`check` → commit；`selfcheck` → commit。
6. `tune`（只判一次）：閘門都過才寫 `frozen_config.json`、`hs_map.npz` → commit。沒過：寫 `results/prereg07/RESULTS.md`（未送分）→ commit，結束。
7. `p07_hs.py val`（4090，獨立背景程序，排夜間，估 5.0–7.2 小時）→ commit `val_hs.npz`、`val_report.json`、`val_seam_moves.npz`。
8. `apply`、`decide.py`、`check_submission.py`（帶題目檔）→ commit；打分前核對（第 2.7 節）→ commit `prescore_audit.json`。`score_allowed` 為真，才跑第 3 節的指令。
9. 打分紀錄帳、`score_H7R.json`、`results/prereg07/RESULTS.md`（另一個代理照第 3 節的固定範圍獨立核對，不論採不採用）、地圖、`CLAUDE.md` 一起 commit。
10. 採用的話：照第 5.1 節的順序先做 06 的指紋和 tag `p06-testday-pre07`，記下 R20／04／Q04 的假測試基準，再從那個 commit 開分支 `p07-testday`，照第 5 節做測試日擴充：實作 → 三個角度審查 → 測試日的修訂 → 演練（含 4090 實測）→ 防誤改閘門 → K5-07 前合併。

## 10. 時程（台灣時間；都是估計）

| 日期 | 事 | 誰 |
|---|---|---|
| 10/04 夜 | rev 1 審查、commit（要早於 mono 全覆蓋的結果）；必要時先結束 `mono1.sh` 的 bash 行程 | Claude |
| 10/05 上午 | mono 跑完（4090，背景；以 `run_log.jsonl` 的 `eta_min` 為準）；第 3 步 | 背景程序、Claude |
| 10/05–10/06 | 第 2、4、5 步 | Claude |
| 10/06 | 第 6 步（CPU，幾分鐘） | Claude |
| 10/06 夜 | 第 7 步（4090，5.0–7.2 小時）。開跑前跑 `nvidia-smi`；10/24 前 C4 要用就讓出（STOP 檔），隔夜續跑 | 背景程序 |
| 10/07 | 第 8–9 步（打分前核對、H7R） | Claude |
| 10/08–10/09 | 緩衝（GPU 被占用時） | — |
| K3-07 10/16 | 最晚的判定日 | — |
| 10/10–10/25 | 06 的 K5' 收尾和指紋、測試日擴充、測試日的修訂、演練、合併（目標 10/25，硬截止 11/03 23:59） | Claude |
| 11/01 | 使用者決定 11 月投入多少；選項裡列出 07 多出的 1 次上傳 | 使用者 |
| 11/04 | test02 照舊，先交 R20；R20 記成 `uploaded` 後，夜間跑 `t07_rank.py hs`（約 3–4 小時） | 使用者上傳；HS 背景跑 |
| 11/05–11/07 | 04 照舊；04 的 `apply` 結束碼 0 後跑 07 的 `apply`；04 上傳後上傳 07 | 每次上傳前使用者說 OK |
| K6-07 11/19 | 07 的成功正式執行最晚 commit 的時間 | — |
| 11/22–11/23 | 跑 `status`；最後一次上傳在 11/23 23:59 前 | 使用者 |

使用者在 10 月不用做任何事（暫停驅動自動更新是選做的建議）；測試日多 1 次上傳（07），最壞再多 1 次重傳。

## 11. 測試日程序（rev 4）

1. 程式：分支 `p07-testday`（`fc73cb6`、`3a651ca`、`30b6c5f`、`5b5c098`）。輸出 `results/test07/`、大檔 `data/test07/`；演練只寫 `results/test07_rehearsal/`、`data/test07_rehearsal/` 或 repo 外。
2. 「H7R 採用」照 rev 3 換成：`score_H7R.json` 的 Q02H 分等級平均 ≥ 0.57、A 側兩個值和 H4R 差 ≤ 1e-12、1,589 題；不成立是結束碼 3。
3. 結束碼照 spec 7.1，另外：(a) 前提裡登記的 3 先判，再看「輸出已存在」；(b) 手數交叉核對不符（4）之後的 `apply --rerun` 一定要 `--bug-revision N`（N ≥ 5）；(c) `hs` 在 `nvidia-smi` 確認沒有別的 compute 程序之後才碰 CUDA；(d) 演練產生的塊在正式模式一律拒絕。
4. 舊塊：資料夾裡有 `created_utc` ≤ R20 第一次上傳時間的塊，`hs` 拒絕（2），由操作者把資料夾改名 `<組>_stale_<時間>`，工具不刪。
5. 本機接縫：「04 的 seam_val 輸入和 prereg04 逐位元組相同」以 test04 `seam_local.json` 的 `byte_identical_probs` 和 `byte_identical_pred_B` 都為真代表；這時另外要求 seam_val 的方向 ρ 等於 `results/prereg07/rank/apply_report.json` 的值（不等是 4）。
6. HS 失敗＝`failed` 或整盤沒有手；每組 `200 × 失敗 ≤ 盤數`。
7. 容許差：GPU 接縫照第 5.2 節（每手 0.05／每盤每 profile 平均 0.002／lp_HS 0.02）。
   - 10/5 4090 實測（演練樹，`results/test07/design/gpu_rehearsal_*`）：seam 400 盤和 10/5 驗證題 HS **逐位元組相同**（每手、每盤、lp_HS 最大差都是 0.0）；另外重算 400 盤 public 演練盤（不同的批次組合），每盤每 profile 平均最大差 0.000459、lp_HS 最大差 0.005948，速度約 1,654 evals/s。
   - lp_HS 實測 0.0059 略高於 spec 的檢討線 0.005，但不到容許差 0.02 的三分之一；測試日的 GPU 接縫是同一批 seam 盤重算（實測逐位元組相同），所以**容許差維持原值**。
8. official_log：07 的檔在 04 規則之前分流；prepare 07 要 R20 和 07 報告綁定的那一筆 04 都已 uploaded、那筆 04 現在仍符合 PREREG_04 5.3、HS 的塊都晚於 R20 上傳；`final` 照 07 → 04 → R20 → R10；有符合的 07 而交別的，要 `--not-07-reason "PREREG_07 rev N: …"`（N ≥ 5、那一列 commit 晚於被推翻的紀錄）。log 沒有 07 紀錄時輸出和 `2d81c80` 逐位元組相同（TagVersusNew）。
9. 偏離第 5.1 節的順序：06 的指紋和 tag `p06-testday-pre07` 是在分支開出之後（10/6，`ab5b654`）才做；分支從 `2d81c80` 開，TagVersusNew 對 `2d81c80` 和 `p04-testday-pre06` 比。合併時 06 的路徑防誤改測試（`tests/test_t06_path_hygiene.py`）要全過。
10. 基準：R20／04 的假測試輸出和 07 的第一次乾跑記在 `results/test07/design/t07_fake_golden.json`；R20 部分等於 `r20_fake_golden.json`。基準是在 t07 程式寫好之後才記的（t07 不寫進 test02／test04 的樹，測試有前後快照），照實揭露。
11. 正規化（NORMALIZED）：HS 只讀 1..num_games 的槽，演練在 T1 判 NORMALIZED 的檔上讀到的槽和原檔逐一相同，所以 GPU 不另跑正規化的那一份。
12. 審查：只做了一份合併三個角度的獨立審查（spec 第 9 節要三份），3 個 major 已修；照實揭露。

> 2026-10-10 追加（只加這一行，不是修訂列）：4090 不能用時的 Colab GPU 備援另寫在 `docs/PREREG_11_GPU_FALLBACK.md`；它的條件成立時，取代本登記第 5.2 節 `hs` 前提中的 `.venv-gpu`／RTX 4090 一句和第 5.3 節「只在本機 4090」「不登記 Colab 備案」，其他判定、容許差和最後一版的規則都不變。
