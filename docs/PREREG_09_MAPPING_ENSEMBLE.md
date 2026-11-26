# 事前登記 09：競賽二把 P1TO 的翻譯模型換成集成（Q09＝Q02H＋4·P1TO_E）

**白話版**：08 把 KataGo 中性探針的 1,540 個數字，用一個 logistic regression（LR）翻成 10 個等級的機率。09 探索（只用訓練玩家）發現：這些數字裡的棋力幾乎是一維的，用「順序 MLP」（先算出一個棋力分數，再切成 10 級）加上普通 MLP 和低秩 LR 的 6 個模型集成來翻，inner 上比 08 多 +0.0127（95% 下界 +0.0028），全部 120,544 題單色題多 +0.0207，黑白都正。本登記把這個集成的做法一字不改地寫死，再加上 P3 探針的 12 萬盤（都是 fit 玩家）重新訓練，用 inner 判一次。**競賽二已經沒有驗證集打分機會**（H8R 是第 4 次、最後一次），所以 09 只靠 inner 決定，交件時會是「沒在驗證集打過分的組合」，這是偏離 PREREG_08 5.4，報告要寫明。測試日不用多跑 GPU：直接用 08 算好的 P1 特徵，在 CPU 上多跑一步、多上傳 1 次。順序 R20 → 04 → 07 → 08 → 09。

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。競賽二的前幾份：
  - PREREG_04：Q02，0.6236（H4R）。PREREG_07：Q02H，0.6327（H7R，rev 3 偏離採用）。
  - **PREREG_08：Q08＝Q02H＋4·P1TO，0.6939（H8R +0.0612 [+0.0436, +0.0793]，採用）**，`results/prereg08/RESULTS.md`。本登記的對照組。
- 探索（都只用 fit／inner 訓練玩家、沒打分）：`results/prereg09/explore/mapping/`（程式和 JSON `7c18b0a`，報告 `REPORT.md` `18145df`）。
- **只做競賽二（TEAM_11248）**，09 的任何輸出都不進競賽一。
- 代號：
  - P1 特徵、P1TO、HS、Q02、Q02H、Q08：同 PREREG_08。
  - **P1TO_E**：同一份 1,540 維（加 44 維額外特徵）經第 1.3 節集成得到的 10 維對數機率。
  - **Q09**＝Q02H＋4·P1TO_E（首選）；**Q08P**＝Q02H＋4·P1TO，LR 用 P1＋P2＋P3 重擬合（唯一的備案）。
  - probe：P1（30,000）、P2（120,000）、P3（120,000）盤，都是 fit 玩家。
- 本登記寫四件事：比什麼、怎麼選、怎樣算成功、什麼時候放棄。另寫一個偏離：沒有驗證集打分（第 3 節）。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `73b4deb` | 第一版，2026-10-09。commit 時的狀態：① 探索的程式和數字已 commit（`7c18b0a`、`18145df`），ens3fam 的成員是**看過 inner 才挑的**（第 8 節）。② P3 名單已 commit（`73bd6e3`，`data/prereg09/p3/sets.json` SHA-256 `164d8bd7b0f212110e222f0a92aeb20079d91ae538c2f0414d563b8478d92eba`）。③ P3 抽取 10/9 08:20 起在 4090 跑；**沒有任何 P3 特徵被讀過**，只看過抽取紀錄（`run.log` 的塊數、失敗盤數、局面數）。④ 驗證題沒有算過任何 09 的 lp；`data/prereg08/val/` 的塊是 08 留下的，09 沒讀過。⑤ 打分紀錄帳自 H8R 之後沒有新列。 |
| rev 2 | `e0ea7fe` | 2026-10-09，在 manifest、check 之前：實作和登記的差異（第 12 節）。工具 `a7fe325`（經獨立審查，沒有 blocker；3 個 major 和 minor 已修）。此時仍沒有任何 P3 特徵被讀過。 |
| rev 3 | `ffcc81c`（合併 `12817db`） | 2026-10-09，測試日的修訂（第 13 節），在任何測試資料之前：`tools/t09_rank.py`、`tools/t09_fake_tests.py`、`tools/official_log.py` 只加行、`docs/TESTDAY_09.md`、`tests/test_t09_*.py`；實作和第 5.2 節的差異、實測的 tol_E＝0.04（08 的 4090 演練樹，m＝0.0113）、K5-09 的條件；標準化 EX 的容許差 0.05 照 5.2 不變（不同批次組合下實測 0.254，要不要改由主 session 決定）。之後要推翻 09 的登記結果，要 rev 4 起的修訂。 |
| rev 4 | `4e1a12f`（補記 `265711d`；合併 `12817db`） | 2026-10-09，在任何測試資料之前、rev 3 第 13 節第 5 點的實測之後：09 GPU 接縫的標準化 EX 只判兩維 frac_lowp 以外的 42 維（≤ 0.05），兩維 frac_lowp 只記錄（第 14 節）。lp_P1TO_E ≤ tol_E＝0.04 和其他接縫條件不變。之後要推翻 09 的登記結果，要 rev 5 起的修訂。 |

## 0. 為什麼做這個（都不看答案；inner 已被多次使用，數字偏樂觀）

- **08 的限度**：P1TO 是 10 類 multinomial LR。探索發現它在 probe 上過擬合（CV log-loss 1.6149），而且沒有用到「等級有順序」這件事。
- **探索結果**（`REPORT.md`，×4，和 Q08 比）：
  - 單一方向的累積 link 在 CV 上就贏過 LR（1.5875 對 1.6149）：棋力在這些特徵上接近一維。
  - **ens3fam**：CV 1.5358；inner 0.6986 → **0.7114**，**+0.0127 [+0.0028, +0.0234]**；單色 **+0.0207 [+0.0136, +0.0282]**，黑 +0.0285、白 +0.0132。
  - 每個候選的 P1 融合權重都頂到 4，HS 權重都是 0（同 Q08）。
  - 單一模型在 inner 上的種子雜訊約 ±0.003。
  - **誠實的增益估計 +0.010～+0.013**：只用 CV 挑的集成在 ×4 下是 +0.0098，下界 −0.0003。
- **學習曲線**（估計）：盤數每加倍，inner 約多 +0.0107（集成）、+0.0094（LR）。加上 P3（共 27 萬盤），兩者都會再升，差約維持 +0.010～+0.015。所以 P3 照固定規則加進來（第 1.2 節），備案 Q08P 也用同樣的 probe。
- **為什麼還做**：08 的驗證集增益和 inner 一致（+0.0612 對 +0.0743），P1 特徵在驗證題上不偏；09 只換翻譯模型，特徵、融合、校準、規則 B 都不變。

## 1. 來源（全部釘死）

### 1.1 沿用 08（不重算、不改）

- 08 的全部凍結檔：`results/prereg08/rank/frozen_config.json`、`p1_map.npz`、`rank_val_probs.csv`、`pred_B.csv`、`val_lp_p1.npz`、`apply_report.json`、`score_H8R.json`；`results/prereg08/feat/*`。
- 07 的 HS、04 的五項、KataGo 程式和權重、中性 metadata、P1 的特徵化：同 PREREG_08 第 1.1–1.3 節。
- inner 的融合和評估一律走 `tools/p08_rank_fuse.py` 的函式（`run_terms`、`masked_inner`、`Q08_TERMS`、`P1REF`；import，不改），同探索的 `ev_lib.Inner`。
- 探索的程式釘 SHA、只當規格，正式工具另寫：`results/prereg09/explore/mapping/{prep,train,combos,ev_lib}.py`（LF SHA 由 `p09_map.py manifest` 記下）。

### 1.2 probe

- P1、P2：同 PREREG_08 第 1.4 節（`p1_map.npz` 記 `probe_used: "P1+P2"`、150,000 盤）。
- **P3**：`data/prereg09/p3/sets.json`（SHA-256 `164d8bd7…`，上表全碼）。120,000 盤，每級 12,000 盤，seed 20261009，只取 fit 玩家，P1、P2 的盤一律排除。用沒有改過的 `p1_extract.py run --set probe` 抽取，共 600 塊。
- **P3 用不用（固定規則，沒有選擇）**：
  - `p09_map.py manifest` 只准在 P3 的 600 塊都寫完之後、或 **10/10（六）23:59 之後**跑。
  - P3 用上的條件：`manifest` 那一刻 600／600 塊都在、通過塊檢查（格式、中性 metadata SHA、精度、batch、KataGo commit、`p1_lib`／`p1_extract` 的 SHA，同 PREREG_08 第 2.2 節第 2 條），而且 `failed` 盤 ≤ 600（0.5%）。
  - 否則只用 P1＋P2，`frozen_config.json` 記 `probe_used: "P1+P2"` 和原因。Q09 和 Q08P 用同一份 probe。
- **重複盤**：`manifest` 用落子序列（原方向）比對 P3 和 bank／mono、P1／P2 probe；重複的 P3 盤一律不用，盤數記錄。和驗證題的重複只在 apply 記錄（第 2.7 節）。
- 玩家：P3 的玩家和 inner、驗證玩家的交集要是空的（split v1）。

### 1.3 翻譯模型 P1TO_E（ens3fam，照探索原樣，不再調）

1. **輸入**：
   - X：目標方 770 維接對手方 770 維（`p1_fuse.variant_X(…, "TO")`），用實際用到的 probe 可用盤以 `p1_fuse.standardise` 標準化（只用 P1＋P2 時就等於 `p1_map.npz` 的 μ、σ）；對手方沒有手時是 0。存成 float32。
   - EX（44 維，只給 lowrank3_ex）：照 `prep.py` 的 `extra_features`，從塊的逐局面 `pos_game`、`pos_side`、`pos_turn`、`pos_logp`、`pos_entropy` 算，每方 22 維：log p、熵各自的標準差、10／25／50／75／90 分位、手數區間 [0,30)／[30,80)／[80,150)／[150,∞) 的平均；log p < log 0.05 的比例；log1p(手數)。用 probe 的 nanmean／nanstd 標準化（標準差非有限或 0 時用 1），之後 nan 設成 0。存成 float32。
   - 可用盤：沒失敗、而且有目標手（同 08）。
2. **成員**（6 個，種子寫死）：

| 成員 | 架構 | 損失、最佳化 | epoch |
|---|---|---|---|
| mlpclm256 ×3（種子 0、1、2） | Dropout 0.1 → Linear(1540,256) → GELU（erf 版）→ Dropout 0.3 → Linear(256,1)＝棋力分數 f；累積 link：θ₁＝t0，θ_k＝t0＋Σ softplus(dt)（t0 起點 −4，dt 起點 log(e−1)），P(y≤k)＝σ(θ_k−f)，相鄰相減、下限 1e-12 再取 log | NLL；AdamW lr 1e-3、weight decay 1e-2 | 排程 24，CV 選停在哪一個 epoch |
| mlp512 ×2（種子 0、1） | Dropout 0.2 → Linear(1540,512) → GELU → Dropout 0.4 → Linear(512,10) → log_softmax | NLL；AdamW lr 1e-3、weight decay 1e-2 | 排程 24，CV 選 |
| lowrank3_ex ×1（種子 0） | 輸入 1540＋44；Linear(1584,3,無 bias，初始 N(0,0.01)) → Linear(3,10) → log_softmax | NLL＋1e-4·(‖A‖²＋‖B‖²)（不罰 bias）；Adam lr 2e-3 | 固定 30 |

   - 共同設定：batch 512；OneCycleLR（max_lr＝lr、pct_start 0.1、總步數＝排程 epoch × ⌈列數/512⌉）；每個 epoch 用 `default_rng(種子)` 打亂列序，batch 內排序；`torch.manual_seed(種子)` 後建模型；`torch.set_num_threads(8)`。
3. **epoch 規則（MLP 兩種）**：每個成員各自做以玩家分 5 折的 CV（`p1_fuse.player_folds`，seed 0，折用實際的 probe 玩家重算），每折跑滿 24 個 epoch、每個 epoch 結束時存 OOF 機率；所有 OOF 的平均 log-loss 最小的那個 epoch 就是 e*。再用全部 probe 重訓：排程照 24 個 epoch 設，跑到 e* 停。探索時選到的 e*（24、24、24、24、15）只記錄，P3 進來後照規則重選。lowrank3_ex 不選 epoch（CV 只記錄）。
4. **集成**：lp_P1TO_E＝log( 6 個成員機率的平均 )（等權，下限 1e-300），同 `combos.lse_mean`。
5. **前向（寫死）**：inner、驗證題、測試日的 lp_P1TO_E 一律由 **numpy float64 前向**計算（權重存 float32、讀進來轉 float64；GELU 用 `scipy.special.erf`），不用 torch。torch 只用在訓練和移植核對。
6. **乘 4**：合併時用 4·lp_P1TO_E（第 2.1 節），同 08。
7. **環境**：訓練在 `.venv`（Python 3.12.13、torch 2.14.0+cpu、numpy 2.5.3、scipy 1.18.1），8 個執行緒，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。版本、執行緒、種子、每個成員的 e*、CV 紀錄、訓練秒數都記進 `frozen_config.json`。CPU torch 是否逐位元重現只記錄；凍結的是權重本身。
8. **凍結**：μ、σ（X、EX）、6 個成員的全部權重、e*、CV 紀錄、`probe_used`、盤數和人數存進 `results/prereg09/rank/p1e_map.npz`，SHA 寫進 `frozen_config.json`。之後驗證題和測試日都只讀這一份。

### 1.4 備案 Q08P 的翻譯模型

- 照 PREREG_08 第 1.5 節原樣（`p1_fuse.fit_mapping_grouped`，同一組 λ 格點、以玩家分折的 CV、L-BFGS-B maxiter 2,000），只是 probe 換成第 1.2 節實際用到的那一份。
- **只在 Q09 沒過閘門時才擬合、才評估**（第 2.4 節）。凍結時存 `results/prereg09/rank/p1p_map.npz`（格式同 `p1_map.npz`）。

### 1.5 驗證題的 lp（凍結後才算，不打分）

- 從 08 留下的 `data/prereg08/val/chunk_*.npz`（96 塊、19,156 盤）讀 `means` 和逐局面資料，**逐塊重算 SHA，要等於已 commit 的 `results/prereg08/feat/val_report.json`**，`ref` 要等於 val_keys 的順序。已 commit 的 `val_p1.npz` 沒有完整平均和逐局面資料，不能單獨用。
- 只讀 `data/local/rank_val_questions.csv` 的題號、盤數、執色、SGF，不讀答案（同 08）。
- 寫 `results/prereg09/rank/val_lp_p1e.npz`（19,156 盤的 lp_P1TO_E、可用遮罩、塊 SHA），commit。

## 2. 合併、檢查和閘門（CPU；`tools/p09_map.py`、`tools/p09_rank_fuse.py`）

### 2.1 合併公式

- 同 PREREG_08 第 2.1 節，只把 E_P1 的來源換掉：Q09 用 4·lp_P1TO_E，Q08P 用 4·lp_P1TO（新 LR）。
- 項的順序 OP、O4、SWA、SWB、HS、P1TO；界限、起點、PENALTY、MAXITER、等級加權、5 折玩家 OOF、PlayerBootstrap 都不變。HS 項照 Q08 保留。
- 不改倍數：×8 在探索裡多約 +0.001～+0.003，但沒登記，不用。

### 2.2 check：輸入能不能用

全部成立才往下：

1. `results/prereg09/feat/p3_manifest.json` 已 commit：P3 每塊的 SHA、塊數、盤數、`ref` 和 sets.json 相符；P3 用不用的判定和原因。
2. P1、P2 塊和 bank／mono 塊的 SHA 等於 08 的 `inner_manifest.json`。
3. 釘住檔的 SHA（第 1.1 節、探索的四支）不變。
4. probe（P1＋P2＋P3）的玩家和 inner、驗證玩家交集是空的；P3 和 P1／P2 不重疊。
5. bank／mono 的可用盤數是 73,169（探索 `prep_record.json` 的值）。

### 2.3 selfcheck：正式工具要重現探索（只核對移植，不是選擇）

1. **融合接縫**：
   - Q02H inner 0.6242752597994481、Q08（凍結 `p1_map.npz`）inner 0.6986195853659671、單色 0.6946737091777063，差 ≤ 1e-12。
   - 把探索存的 6 個成員的 lp_inner（`data/prereg09/tmp/mapping/v_*.npz`）照 1.3 第 4 點平均、乘 4 放進融合：inner 要是 0.7113584160645174、單色 0.7154116187609454，差 ≤ 1e-12。
2. **訓練移植**：用 `p09_map.py` 只在 P1＋P2 上重訓 6 個成員：
   - 每個成員的 CV log-loss、e* 和探索紀錄比較；用 numpy 前向算的 inner lp 和探索的 lp_inner 比，最大差記錄。
   - 集成的 inner 分等級平均和 0.7113584160645174 的差要 ≤ 0.002（小於種子雜訊 0.003），否則停下查。查明是數值路徑差異、不是邏輯錯誤時，寫 rev 2 揭露，以 repo 算出的數字為準。
   - 同一份權重的 torch 前向和 numpy 前向，每盤 lp 最大差 ≤ 1e-4（float32 和 float64 的差）。
   - 這一份「只用 P1＋P2 的 Q09」只記錄，也是「換模型、不加 P3」的消融。

### 2.4 組合（固定順序，第一個過的就停）

| 順序 | 組合 | 內容 |
|---|---|---|
| 對照 | Q08 | 凍結的 `p1_map.npz`（就是 08） |
| 1 | **Q09** | Q02H＋4·P1TO_E（第 1.3 節，用第 1.2 節的 probe 訓練） |
| 2 | Q08P | Q02H＋4·P1TO（第 1.4 節）；**只在 Q09 沒過時才擬合和評估** |

- 沒有其他選擇。Q09 過了就凍結 Q09，不擬合 Q08P。
- **只記錄、不可選**：只用 P1＋P2 的 Q09（selfcheck）；各成員單獨 ×4；每折融合權重；各級、單色（全體、黑、白）分數；凍結時的樣本內分數；`se_val_record`。

### 2.5 閘門（對 Q08，全部成立才凍結）

| 閘門 | 條件 |
|---|---|
| 接縫 | 第 2.3 節第 1 點 |
| G1 | inner 評估假題的分等級平均差（候選 − Q08）≥ +0.010（容許 1e-12） |
| G2 | 同一差值的配對 PlayerBootstrap（每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0 |
| G4 | 單色假題全部 120,544 題（SHA `45d0f14a…`）的分等級平均差：配對 PlayerBootstrap 95% 下界 > 0，黑題、白題的差各自 > 0 |

- 每題平均差只記錄。
- **只判一次**：`tune` 正式跑一次；要 `--rerun`，只准在指出具體程式錯誤的修訂 commit 之後。Q08P 那一輪也只判一次。
- 兩個都沒過：記為「未採用」，`results/prereg09/RESULTS.md` commit 後結束。競賽二照 PREREG_08 5.4（08 → 07 → 04 → R20 → R10）。

### 2.6 tune：凍結

- 過了的那個候選，用全部 inner 擬合假題重擬合融合參數（`p04_rank_fuse.gen_fit_record`，同 08），寫 `results/prereg09/rank/frozen_config.json` 和 `p1e_map.npz`（或 `p1p_map.npz`），一起 commit。
- `frozen_config.json` 記：組合、倍數 4、參數、5 折紀錄、閘門紀錄、`se_val_record`；map 的 SHA、`probe_used`、三份 sets.json 的 SHA；08 凍結檔的 SHA；釘住檔、`p3_manifest.json` 的 SHA；方向檢查的參照值 ρ_eval、ρ_mono（只用 E_P1 的期望等級對 Q02 的期望等級，同 08）；工具的 SHA；`.venv` 的 Python、numpy、scipy、torch 版本和執行緒數。
- 凍結時再核對一次：同一份權重的 torch 前向和 numpy 前向，在 inner 每盤 lp 最大差 ≤ 1e-4。
- `tune`、`apply` 在 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 任一有設時拒絕（同 PREREG_08 rev 2 第 2 條）；apply 不 import torch。

### 2.7 apply（驗證題，**不打分**）

- **前提**：`frozen_config.json` 和 map 已 commit；val 塊的 SHA 等於 08 的 `val_report.json`。
- **先重現 08**：用 08 的程式算驗證題的 Q02、Q02H、Q08，要和 `results/prereg04|07|08/rank/rank_val_probs.csv` 逐位元組相同，不同就停下。
- 照 1.5 寫 `val_lp_p1e.npz`，照 `frozen_config.json` 算 Q09（或 Q08P），寫 `results/prereg09/rank/rank_val_probs.csv`；跑 `decide.py`、`check_submission.py rank … --questions data/local/rank_val_questions.csv`。
- `apply_report.json` 的 `usable` 要以下全部成立才為真（**都不看答案**）：
  - 可用遮罩、失敗盤和 08 的 `val_p1.npz` 完全相同（P1 失敗 0 盤）；
  - Q02、Q02H、Q08 逐位元組重現；
  - **方向檢查**：只用 E_P1E 的期望等級對 Q02 的期望等級，Spearman ρ ≥ max(0.5 × ρ_eval, 0.078)；
  - 格式檢查過；
  - 稽核（下一點）沒有找到影響預測的程式錯誤，由 `apply --finalize` 寫入。
- **只記錄**：Q09 和 Q08 在規則 B 下預測不同的題數、等級差的分布；只用 E_P1E 和只用 E_P1 的期望等級的 ρ；驗證題盤和 probe（含 P3）的落子序列重複數（預期 0）。
- **稽核**（另一個代理，看不到答案、不看任何分數）：val 塊 SHA；用 `p1e_map.npz` 獨立重算的 lp_P1TO_E 和 apply 的相同（≤ 1e-12）；集成平均、乘 4 只做一次；Q02／Q02H／Q08 重現；`decide` 的輸入輸出。結果寫 `results/prereg09/rank/apply_audit.json` 並 commit。
- **不跑 `tools/score.py`，不讀 `rank_val_answers.csv`**。`tests/test_p09_hygiene.py` 要檢查打分紀錄帳沒有任何 `prereg09` 列。

## 3. 沒有驗證集打分（偏離，寫明）

- PREREG_08 第 3 節：H8R 是競賽二整季第 4 次、最後一次打分，之後沒有任何驗證集打分機會。PREREG_08 5.4 寫「不交沒在驗證集打過分的組合」。
- **本登記偏離這一條**：Q09 只靠 inner 閘門（第 2.5 節）和驗證題的無標籤檢查（第 2.7 節）就上場。決定者：Claude（使用者 10/2 授權「只要對最終排名有幫助」，10/9 再說「目標始終是 Private 分數和排名」）。
- 理由：
  1. 特徵、融合、校準、規則 B、測試日的 GPU 計算都和已在驗證集證明過的 Q08 相同，只換翻譯模型；翻譯模型只用 fit 玩家學。
  2. 08 的 inner 增益和驗證集增益一致（+0.0743 對 +0.0612），P1 特徵在驗證玩家上沒有失真。
  3. 閘門比 08 嚴：除了 inner 下界 > 0，還要求單色題黑白都正。
- 風險（揭露）：inner 已被重複使用（第 8 節），可能有約 0.003 的選擇偏差；真實增益可能比 +0.010 小，甚至接近 0。最壞情況是 Q09 和 Q08 差不多，因為 P1 融合權重頂到上限、預測的方向一致（第 2.7 節記錄不同題數）。
- 取代的條文：PREREG_08 5.4「不交沒在驗證集打過分的組合」，只對 Q09／Q08P 例外；不改原文（照 PREREG_07 第 5.5 節的前例）。

## 4. 截止點（台灣時間；沒趕上就照表做）

| 截止點 | 條件 | 沒趕上 |
|---|---|---|
| K1-09（無日期） | rev 1 的 commit 早於任何 P3 特徵被讀 | 這一季不做 09 |
| K3-09 10/12（一）23:59 | manifest、check、selfcheck、train、tune 都 commit；過了的話 `frozen_config.json` 已 commit | 這一季不用 09 |
| K4-09 10/14（三）23:59 | 驗證題 apply、稽核、`apply --finalize`、`RESULTS.md` 都 commit | 這一季不用 09 |
| K5-09 10/31（六）23:59 | 採用時：08 路徑指紋和 tag、09 測試日擴充、測試日的修訂已 commit、合併回 main、演練通過 | 這一季不用 09 |
| K6-09 11/19（四）23:59:59 | 09 在官方測試集有一次成功的正式執行並 commit | 最終照 PREREG_08 5.4 |

## 5. 採用後的測試日

### 5.1 保護舊路徑（rev 1 就生效）

- 一律不改：R20、04、06、07 的測試日路徑（各自的 tag）；`results/prereg01`–`prereg08/**`、`results/test02|04|07|08/**`；`tools/p0*_*.py`、`t02_*`、`t04_*`、`t06_*`、`t07_*`、`t08_*`；`third_party/katago/**`；釘住檔；`results/prereg09/explore/**` 只准加檔。
- 只准加行：`tools/official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_07_HUMANSL.md`、`docs/PREREG_08_STRENGTH_PROBE.md`、本登記。
- **08 的指紋，順序寫死**（09 的測試日程式動工前做完）：
  1. 做 `results/test08/design/p08_path_fingerprint.json`：`tools/t08_*.py`、`tests/test_t08_*.py`、`results/test08/design/*`、`docs/TESTDAY_08.md` 釘 SHA；只准加行的那幾個放 `insert_only`。
  2. 在那個 commit 打 tag `p08-testday-pre09`，加一支 `tests/test_t08_path_hygiene.py`。
  3. 記下 R20、04、07、08 的假測試演練輸出當基準（`results/test09/design/baseline_pre09.json`），合併前要逐位元組相同。
  4. 從那個 commit 開分支 `p09-testday`。

### 5.2 要做的東西（全部 CPU，不多跑 GPU）

- **`tools/t09_rank.py apply`**：
  - **前提**：08 的 `t08_apply_report.json` 是同一模式、結束碼 0、usable、已 commit（08 判不用時 09 也不用，結束碼 3）；09 已採用（`results/prereg09/RESULTS.md`）；K6-09 未過；`.venv` 版本等於 `frozen_config.json`；不設 BLAS 執行緒變數。
  - **P1 輸入**：seam、public、private 都讀 08 的塊 `data/test08/p1/<組>/`，逐塊 SHA 要等於 08 的 `p1_<組>_report.json`；X 和 EX 照 1.3 從塊算，lp_P1TO_E 用 numpy 前向。
  - **Q08 對得上 08**：用 `p1_map.npz` 和同一份輸入重算的 Q08，要和 08 已 commit 的 `rank_test_probs.csv` 相同（容許差照 PREREG_08 rev 3 第 9 點：機率差 ≤ 1e-12、等級全同）；之前先核對 08 報告綁定的檔都沒變。
  - **GPU 接縫（09 的部分）**：seam 400 盤的塊和 08 的 `data/prereg08/val/` 前 2 塊（SHA 照 `val_report.json`）比：標準化 EX 最大差 ≤ 0.05；lp_P1TO_E 每盤最大差（乘 4 之前）≤ tol_E。
    - **tol_E 在拿到測試資料前定**：用 10/9 的 4090 演練樹（private 前 1,000 盤，不同批次組合，`results/test08/design/gpu_rehearsal_*`）量 lp_P1TO_E 對驗證塊的最大差 m，tol_E＝max(0.04, 2.5 × m)，寫進測試日的修訂。
  - **本機接縫**：seam_val 用 08 的 seam_val 輸入加已 commit 的 `val_lp_p1e.npz` 算 Q09，和 `results/prereg09/rank/rank_val_probs.csv` 比；08 的本機接縫逐位元組相同時，09 也要逐位元組相同。
  - **方向檢查**：每組 ρ ≥ max(0.5 × ρ_mono, 0.155)（ρ_mono 取 `frozen_config.json`）。
  - 失敗率和手數交叉核對沿用 08 的判定（08 結束碼 0 已涵蓋）。
  - 都過了才寫 `results/test09/rank/submission_rank.csv`、`test_lp_p1e.npz`、`t09_apply_report.json`。結束碼同 t08：0／2／3／4。
- **`tools/official_log.py` 只加行**：認得 `results/test09/rank/`（在 08 之前分流），方法名稱「prereg09 Q09 = Q02H + 4 x P1TO_E (ens3fam), rule B (甲)」；`prepare` 09 要 R20、04、07、08 都已上傳，而且 08 仍符合 PREREG_08 5.4；有符合的 09 而交別的，要 `--not-09-reason "PREREG_09 rev N: …"`；log 沒有 09 紀錄時，輸出和 tag `p08-testday-pre09` 的版本逐位元組相同。
- `tools/t09_fake_tests.py`、`tests/test_t09_*.py`、`docs/TESTDAY_09.md`。輸出只寫 `results/test09/`，演練寫 `results/test09_rehearsal/` 或 repo 外。

### 5.3 上傳順序和最後一版（競賽二）

- **順序**：R20（11/04）→ 04 → 07 → 08 → 09。每一份都要使用者在對話裡說 OK。09 在 08 上傳之後才 `prepare`。
- **最後一版**：有符合的 09 → 09；沒有 → 照 PREREG_08 5.4（08 → 07 → 04 → R20，T1 FAIL_OPENING 時 R10）。
- public 分數不進判定；09 比 08 低 0.05 以上時只查錯，查不到就維持本節的結果。
- 取代的條文：PREREG_08 5.4「有符合的 08 時最終是 08」→ 有符合的 09 時最終是 09（不改原文）。

## 6. 另一場

- 競賽一不用 P1／P1TO_E；09 的任何輸出都不進競賽一。不用競賽一的測試資料。

## 7. 刻意不放進 09 的

| 做法 | 理由 |
|---|---|
| ×8 或放寬權重上限 | 探索多約 +0.002，屬於看 inner 挑參數 |
| 改成員、種子、超參數、epoch 規則 | 寫死成探索的原樣；再調等於第二次挑選 |
| 集成和凍結的 LR 並列成兩項 | 探索 +0.0134，沒有比單獨用集成好 |
| 梯度提升 | `.venv` 沒有 sklearn，要下載；沒探索 |
| 再加 probe（P4） | 沒有驗證集打分機會，只靠 inner 的東西不再擴充 |
| 拿掉 HS 項 | Q08 的結構保留，接在 08 之上 |
| 測試日的 GPU 重算 | 不需要，08 的 P1 塊就是輸入 |

## 8. 揭露（寫進 RESULTS 和報告）

- **事前與事後的界線**：
  - ens3fam 的成員、架構、種子、epoch 規則、倍數，都是**看過 inner 才挑的**（探索約 40 個變體，`summary.json`）。誠實的增益估計 +0.010～+0.013；只用 CV 挑的集成下界 −0.0003。
  - 本登記在讀任何 P3 特徵、做任何 P3 重訓之前 commit，寫死做法；P3 用或不用照第 1.2 節的固定規則。
- **inner 的重複使用**：P1 試跑、08 tune、09 探索、09 tune 用的是同一批 inner 玩家，可能有約 0.003 的選擇偏差（單一模型的種子雜訊量級）。
- **沒有驗證集打分**（第 3 節）：09 是沒在驗證集打過分的組合，偏離 PREREG_08 5.4；驗證題只做無標籤檢查。
- **備案的多重比較**：兩個候選依固定順序各判一次，第一個過的就停。
- **權重上限**：同 08（×4 等於 lp 的權重上限 16）。
- **外部資源**：同 PREREG_08 第 8 節（KataGo 程式和 HumanSL 權重）；09 只換本隊自己的翻譯模型（PyTorch CPU 訓練、numpy 推論）。
- **消融**：驗證集三層（Q02 → Q02H → Q08）照 08；inner：Q08、只用 P1＋P2 的 Q09（換模型的效果）、Q09（加 P3）、各成員單獨。
- **數值**：CPU torch 訓練不保證逐位元重現，凍結的是權重；推論用 numpy float64。
- 生成式 AI 協助的部分照規則揭露；報告草稿 `report/` 競賽二第 7 節依 09 結果改寫。

## 9. 順序（`PY`＝`.venv/Scripts/python.exe`，Git Bash，加 `PYTHONDONTWRITEBYTECODE=1`）

1. 本登記 rev 1 commit，再用一個 commit 填上 rev 1 的編號。**硬規則：早於任何 P3 特徵被讀。**
2. 寫程式：`tools/p09_map.py`（`manifest`、`prep`、`selfcheck`、`train [--fallback]`、`status`）、`tools/p09_rank_fuse.py`（`check`、`tune [--fallback]`、`apply`、`apply --finalize`）、`tests/test_p09_*.py`（含 `test_p09_hygiene.py`）。經獨立審查、全套測試綠了才 commit。
3. P3 600／600，或 10/10 23:59 已過：`PY tools/p09_map.py manifest` → commit `results/prereg09/feat/p3_manifest.json`。
4. 實作和登記的差異寫成 rev 2，在 `check` 之前 commit。`PY tools/p09_map.py prep`（memmap 放 `data/prereg09/map/`，不進 git）→ `PY tools/p09_rank_fuse.py check` → commit → `PY tools/p09_map.py selfcheck`（P1＋P2 重訓，約 1 CPU 小時）→ commit。
5. `PY tools/p09_map.py train`（實際 probe，約 2 CPU 小時，權重放 `data/prereg09/map/`）→ `PY tools/p09_rank_fuse.py tune`（只判一次）：過了寫 `frozen_config.json`、`p1e_map.npz` → commit。沒過：`train --fallback` → `tune --fallback`（只判一次）→ 過了凍結 Q08P；也沒過就寫 `results/prereg09/RESULTS.md`（未採用）→ commit，結束。
6. `PY tools/p09_rank_fuse.py apply` → commit；另一個代理做稽核 → commit `apply_audit.json`；`apply --finalize` → commit。不跑 `score.py`。
7. `results/prereg09/RESULTS.md`、地圖、`CLAUDE.md` 一起 commit。
8. 採用的話：照 5.1 做 08 的指紋和 tag `p08-testday-pre09`、記基準，再開分支 `p09-testday`：實作 → 量 tol_E → 審查 → 測試日的修訂 → 演練 → 防誤改閘門 → K5-09 前合併。

## 10. 時程（估計）

| 日期 | 事 | 誰 |
|---|---|---|
| 10/09 | rev 1；P3 抽取跑完（4090） | Claude、背景程序 |
| 10/10–10/12 | 第 2–5 步（CPU） | Claude |
| 10/13–10/14 | 第 6–7 步 | Claude |
| 10/15–10/31 | 第 8 步 | Claude |
| 11/05–11/18 | 08 的 apply 之後跑 `t09_rank.py apply`（CPU，幾分鐘）；08 上傳後上傳 09 | 使用者上傳（每次說 OK） |
| 11/22–11/23 | `official_log.py status`；最後一次上傳 | 使用者 |

使用者在 10 月不用做任何事。測試日多 1 次上傳（09）。

## 11. 防誤改閘門

- CLAUDE.md 的閘門指令加 `tests/test_p09_hygiene.py`；5.1 之後再加 `tests/test_t08_path_hygiene.py`，合併 09 測試日後加 `tests/test_t09_hygiene.py`。
- 唯一允許的失敗不變（`test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`）；打分紀錄帳不得出現 `prereg09` 列。

## 12. rev 2：實作和登記的差異（在 manifest、check 之前）

工具 `a7fe325`（`tools/p09_forward.py`、`tools/p09_map.py`、`tools/p09_rank_fuse.py`、`tests/test_p09_*.py`）。獨立審查沒有 blocker；三個 major（重用舊權重沒核對程式版本、登記的 2.3 第 2 點路徑走不到、紀錄綁「現在的工具」導致修程式後永遠過不了 check）和幾個 minor 都已修。下面是和 rev 1 不同或 rev 1 沒寫清楚的地方。

1. **torch 對 numpy 的核對（取代 2.3 第 2 點和 2.6 的寫法）**：判定的是「存成 float32 的權重轉成 float64、輸入 float64 的 torch 前向」對「numpy float64 前向」，集成對數機率每盤最大差 ≤ 1e-4。torch 用 float32 原樣前向只記錄。原因：累積 link 的上尾（1 − sigmoid 接近 1）在 float32 下失去精度，預覽時單一成員的 log 機率最多差 11.6、合成測試 3.2e-4；float32 的集成差 7.7e-6。推論一律用 numpy float64（rev 1 第 1.3 第 5 點不變），所以這只是更嚴格的移植核對。
2. **準備好的陣列**存原始 float32（塊的平均，對手方沒有手時是 NaN；原始 EX），用的時候才標準化。訓練輸入和探索 `prep.py` 存的逐位元相同。推論輸入是原始值的 float64 標準化，不再四捨五入成 float32。
3. **動差**：X 的 μ、σ 用兩趟串流算，和 `np.nanmean`／`np.nanstd` 逐位元相同；正式 selfcheck 要求 P1＋P2 的 μ、σ 和 `p1_map.npz` 逐位元相同。EX 的動差在 float32 化的原始 EX 上算，只修 σ（同 `train.py`）。
4. **P3**：落子序列和 bank／mono 或 P1／P2 probe 相同的 P3 盤不用；「有效」還要求資料夾沒有多餘的檔、計畫盤數 120,000。manifest 記內容（塊 SHA、判定）和 `manifest_code_sha256`（`p09_map.py`、`p1_lib`、`p1_extract`），check 比內容、不比「現在的工具是不是當時那一份」。
5. **接縫**：Q08 用準備好的 inner 陣列、照 08 tune 分 bank、mono 兩次呼叫的方式算；探索集成的接縫用探索的 `inner_meta.npz` 對齊 `v_*.npz` 的列。`data/prereg09/tmp/mapping/{inner_meta.npz, v_<6 個成員>.npz}` 要保留到 `selfcheck.json` commit 之後。
6. **selfcheck 容許差沒過時（落實 2.3 第 2 點）**：集成 inner 和 0.7113584160645174 差 > 0.002 時，selfcheck 寫 `passed: false` 和 `only_ensemble_tolerance_failed`。查明是數值路徑差異、不是邏輯錯誤時，先 commit `selfcheck.json`，再 commit 一個說明原因的修訂（rev N，N ≥ 2，commit 晚於 selfcheck），才能 `train --revision N`；這個覆寫記進 `train_report.json`、`tune_report.json`、`frozen_config.json`。接縫或 torch 核對失敗不能覆寫。
7. **紀錄的綁定**：已 commit 的 `check_report.json` 在輸入不變時一直有效；selfcheck → train → tune 只綁訓練程式（`p09_map.py`、`p09_forward.py`），每一步都重新核對 check 的輸入（釘住檔、manifest SHA、用到的 P3 塊、準備好的陣列）；`apply` 要求全部工具等於 tune 那一份。已存的成員權重只有在準備資料、列、`p09_map.py`／`p09_forward.py` 的 LF SHA、torch 版本、執行緒數都相同時才重用。
8. **環境**：正式 selfcheck、train、tune 在 `.venv` 和第 1.3 第 7 點不同時拒絕；apply 只核對 python、numpy、scipy，而且不 import torch。正式執行一律 8 個執行緒；`--threads`、`--members` 只准預覽用。
9. **新增的 commit 紀錄**：`train_report.json`／`train_fallback_report.json`（綁權重 SHA）。權重放 `data/prereg09/map/`（不進 git）；`p1e_map.npz` 約 11 MB，進 git。`frozen_config.json` 加 `members`、每個成員的 `training_seconds`、`selfcheck_override`。
10. **備案**：只在 Q09 的 `tune_report.json`（閘門沒過、沒有程式問題）已 commit 之後才准 `train --fallback`、`tune --fallback`，`--precommit` 也一樣。LR 在 27 萬列上重擬合，記憶體峰值約 12–14 GB（`fit_mapping_grouped` 內部用 float64，為了和 PREREG_08 1.5 一致不改），跑的時候不要有其他大程式。
11. **tune 的失敗**：接縫失敗在閘門之前就停，不寫報告、不算判過；torch 核對在閘門算完之後失敗時，照樣寫 `tune_report.json`（`gates_passed: false`、記問題），結束碼 1，之後要重跑須 `--rerun --revision N`。
12. **tune 只記錄的項目**：各成員單獨 ×4、selfcheck 的 P1＋P2 集成 ×4。
13. **apply**：報告欄位是 `usable`（False／`pending_audit`／True）；稽核格式 `c10-prereg09 apply_audit v1`、欄位 `program_errors_affecting_predictions`；綁定輸出 `rank_val_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`val_lp_p1e.npz`，輸入 `frozen_config.json`、map、08 的 `val_report.json`、`train_report.json`。「和 08 相同」是指 counts／failed 等於 `val_p1.npz`，p1_ok／usable 等於 08 的 `val_lp_p1.npz`。驗證題和 probe 的重複由訓練 SGF 重算。`frozen_usable` 拒絕預覽和 precommit 寫的凍結檔。
14. **G4**：「黑題、白題的差各自 > 0」看點估計，同 08。
15. **防誤改**：`tests/test_t08_hygiene.py` 的 `FROZEN_GLOBS` 從 `p0*` 縮成 `p0[1-8]*`（否則新的 p09 檔會被當成凍結路徑下的新檔），由 `tests/test_p09_hygiene.py` 釘 LF SHA `081104659f56…`。第 5.1 節的 `tools/p0*_*.py` 指 p01–p08。閘門加 `tests/test_p09_hygiene.py`，共十三個檔。

## 13. rev 3：測試日的修訂（在任何測試資料之前；分支 `p09-testday`）

第 5.1 節第 1–3 步（08 的指紋、tag `p08-testday-pre09`、R20／04／07／08 的假測試基準 `results/test09/design/baseline_pre09.json`）已在 main 的 `0b65255` 做完，分支從那個 commit 開。本節寫實作和第 5.2 節的差異、實測的 tol_E，以及測試日照什麼做；Claude 的步驟在 `docs/TESTDAY_09.md`。

1. **程式**：分支 `p09-testday` 的 `b4ee6aa`（`tools/t09_rank.py`、`tools/t09_fake_tests.py`、`.gitattributes`／`.gitignore` 各加行）、`767210d`（`tools/official_log.py` 只加行＋`tests/test_t09_official_log.py`）、`d9bdf34`（`results/test09/design/tol_e_measurement.json`、`t09_fake_golden.json`、`tests/test_t09_rank.py`、`test_t09_hygiene.py`、`test_t09_dryrun.py`）。輸出只寫 `results/test09/`；演練只寫 `results/test09_rehearsal/` 或 repo 外。08 的工具（`tools/t08_rank.py`，經由它 07、04 的工具）和 `tools/p09_rank_fuse.py`、`tools/p09_forward.py` 只 import、不改。`t09_rank.py` 不 import torch、`p09_map`、u1_lib。
2. **`apply` 的前提，照 5.2 的順序**：git 乾淨（正式）；不設 BLAS 執行緒變數；prereg09 的 8 個已 commit 檔 SHA 不變（`frozen_config.json`、`p1e_map.npz`、`rank_val_probs.csv`、`pred_A/B.csv`、`val_lp_p1e.npz`、`apply_report.json`、`apply_audit.json`），`frozen_config.json` 是閘門通過的 Q09，`apply_report.json` 是 `usable: true` 而且 finalized（「09 已採用」，不成立是 3），`RESULTS.md` 存在；`.venv` 的 python、numpy、scipy 等於 `frozen_config.json` 記的版本（不等是 4）；K6-09；接著跑 `t08_rank.py` 的全部前提（07、08 的釘住檔、H7R、**H8R**、T1、test02、R20 的第一次上傳、07、04 的 apply；任何一條登記的「不用」都是 09 的 3）；最後 **08 的 `t08_apply_report.json` 是同一模式、結束碼 0、usable、已 commit**（08 判不用時 09 結束碼 3），08 報告綁定的每一個輸入、08 的 6 個輸出都沒變（變了是 2，先照 TESTDAY_08 處理）。
3. **P1 輸入**：seam、public、private 都讀 08 的塊 `data/test08/p1/<組>/`。`p1_<組>_report.json` 和 `<組>_p1.npz` 的 SHA 要等於 08 的 `t08_apply_report.json` 記的值；報告列的每一塊重算 SHA；refs 依序、counts／n_moves／failed 等於 `<組>_p1.npz`。X 用 `p09_forward.x_raw`（塊的 float32 平均轉 float64，對手方沒有手時 NaN），EX 用 `p09_forward.ex_raw32`（逐塊的 `pos_game`／`pos_side`／`pos_turn`／`pos_logp`／`pos_entropy`，和驗證題的 apply 同一個函式）；塊裡沒有逐局面陣列時拒絕（2）。lp_P1TO_E 用凍結的 `p1e_map.npz`、numpy float64 前向（`p09_forward.map_logp`）。可用盤＝沒失敗而且有目標手（同 08）。
4. **09 的 GPU 接縫**（5.2；容許差 tol_E 在本 rev 定）：seam 400 盤（08 的 seam 塊）和 08 的驗證塊 `data/prereg08/val/chunk_00000.npz`、`chunk_00001.npz`（同一批盤、同樣的 200 盤分塊；SHA 照 `results/prereg08/feat/val_report.json`）比。`val_seam_feats.npz` 沒有逐局面陣列，所以改比驗證塊本身；兩者的平均逐位元相同（08 的 `t08_fake_tests` 已核對）。
   - failed、counts（兩方）、n_moves 完全相同；
   - 標準化 EX（用 `p1e_map.npz` 的 μ_EX、σ_EX，`standardise_ex`）在兩邊都可用的盤上的最大絕對差 ≤ 0.05（5.2，不變）；
   - lp_P1TO_E 每盤最大差（乘 4 之前，10 級取最大再對盤取最大）≤ **tol_E＝0.04**；
   - 標準化 X 的最大差、是否逐位元相同、lp 的分位數只記錄；lp 差 > 0.005 另標檢討線（同 07、08）。
   - 不通過是 4（停下查，照第 5 節）。
5. **tol_E 的實測**（10/9，任何測試資料之前；`tools/t09_fake_tests.py measure-tol --tree D:/c10_t08_reh`，紀錄 `results/test09/design/tol_e_measurement.json`）：用 08 的 4090 演練樹（`.venv-gpu` torch 2.11.0+cu128、cuDNN 91900、驅動 616.92）。
   - seam 400 盤（GPU，和驗證塊同樣分塊，`gpu/test08`）：標準化 EX、標準化 X、lp_P1TO_E 的差都是 0，平均和 EX 逐位元相同。
   - private 前 1,000 盤（GPU，驗證列 4,787–5,786，每塊都在驗證塊格線之外，`gpu_pub/test08`）：failed／counts／n_moves 全同；**lp_P1TO_E 每盤最大差 m＝0.01127**（中位數 0.0011、90% 分位 0.0026、99% 分位 0.0046；1,000 盤中 6 盤 > 0.005），最大等級 999／1,000 盤相同；標準化 X 最大差 0.0186（同 08 的實測）。
   - **tol_E＝max(0.04, 2.5 × m)＝max(0.04, 0.0282)＝0.04**。寫進 `tools/t09_rank.py` 的 `TOL_E`、`tools/official_log.py` 的 `T09_TOL["lp_p1e"]`，`tests/test_t09_hygiene.py` 核對兩者等於紀錄。
   - **標準化 EX 在不同批次組合下最大差 0.254，超過 0.05**（只記錄，不改 5.2 的值）：幾乎都來自「log p < log 0.05 的比例」兩維（`t_frac_lowp` 0.254、`o_frac_lowp` 0.239；1,000 盤中 24 盤有一方的這一維變了）。它是有門檻的計數，fp16 的捨入讓一兩手跨過門檻就跳一格（σ 約 0.062）。去掉這兩維，其他 42 維最大差 0.016。lp_P1TO_E 只差 0.011，所以這個跳動對預測幾乎沒有影響。測試日的接縫比的是同樣分塊的 seam，兩次 4090 實測（08 的 10/8、10/9 演練）都是 0，所以照登記維持 0.05；但如果測試日 cuDNN 選了不同演算法，seam 的 EX 可能出現這個量級的差而判 4。**要不要在測試資料之前另寫 rev 4（例如 EX 的容許差改成 2.5 × 0.254 ≈ 0.64，或兩維 frac_lowp 只記錄、其他 42 維 0.05）由主 session 決定**；沒有修訂就維持 0.05。
6. **Q08 對得上 08**：public、private 用 08 的程式（`t08_rank.q08_set`，test02／04／07 的輸入、08 凍結的 `p1_map.npz`、08 的塊）重算 Q08，和 08 已 commit 的 `rank_test_probs.csv` 比：題號、順序、盤數相同，機率差 ≤ 1e-12，規則 A、規則 B 的等級全同（PREREG_08 rev 3 第 9 點的判法）；逐位元組是否相同只記錄（演練是相同）。另外 09 的可用盤（`p1_ok`、`usable`）要等於 08 的 `test_lp_p1.npz`。不成立是 4。這一條取代 5.2 寫的「之前先核對 08 報告綁定的檔都沒變」以外的 Q02H 核對：Q08 重算時 Q02H 也一起重算，對得上 08 就涵蓋了 07 那一層。
7. **本機接縫**：seam_val 用 07 的 seam_val 輸入（test02／04 的 seam_val、`val_hs.npz`）加已 commit 的 `val_lp_p1e.npz`（不從塊重算）算 Q09，和 `results/prereg09/rank/rank_val_probs.csv`、`pred_B.csv` 比。容許差同 07、08（`results/test04/rank/t04_apply_report.json` 的 `seam_local.tol`，規則 B 最多差 4 題）。**08 的 `seam_local.json` 逐位元組相同時，09 要求逐位元組相同**，而且 seam_val 的方向 ρ 要等於 `results/prereg09/rank/apply_report.json` 的值（不等是 4）。題號、順序、盤數不同是 4；08 的接縫成立、09 卻超出容許差是登記的「09 不用」（3）。
8. **失敗和方向**：P1 失敗＝`failed` 或整盤沒有手；每組 `200 × 失敗 ≤ 盤數`，否則 3（08 結束碼 0 已涵蓋，同一批資料）。方向 ρ：每組「只用 E_P1E 的期望等級」和「Q02 的期望等級」的 Spearman ρ ≥ max(0.5 × ρ_mono, 0.155)＝**0.48426**（ρ_mono＝0.9685160520465519，`frozen_config.json`），否則 3。seam_val 的 ρ（門檻 max(0.5 × ρ_eval, 0.078)）只記錄。
9. **數值路徑**（演練實測）：測試題的 lp_P1TO_E 是每組一次前向（每 8,192 列一塊），驗證題當時是 19,156 列。用驗證題假扮測試題的演練裡，9,602 盤有 132 盤的 lp_P1TO_E 差 ≤ 1.9e-14（BLAS 依列數分塊的捨入），800 題的機率差 ≤ 4.4e-16，規則 A、B 的等級全部相同。所以「等於 prereg09 驗證前 800 列」用 1e-12 和等級相同判，不要求逐位元組相同。seam_val 讀已 commit 的 lp，逐位元組相同。
10. **結束碼**同 t08：0／2／3／4（09 沒有 GPU 步驟，沒有 5）；前提裡登記的 3 先判，再看「輸出已存在」；出現 3 或寫好 `t09_apply_report.json` 之後，只有 `--bug-revision N`（N ≥ 4、那一列的 commit 晚於被推翻的紀錄）才能重跑。K6-09 過後已有成功的那一次：2，那一次仍有效。
11. **official_log**（只加行）：`results/test09/rank/` 的檔在 08、07、04 的分流之前認出；方法名稱「prereg09 Q09 = Q02H + 4 x P1TO_E (ens3fam), rule B (甲)」；`prepare` 09 要 R20 的第一次上傳、09 報告綁定的那一筆 08 已 uploaded 而且仍符合 PREREG_08 5.4（它又要求綁定的 07、04 已上傳），P1 的塊都晚於 R20 上傳；`prepare`／`final`／`status` 每次用報告裡的數字和檔重算 5.2 的條件、H8R 和 09 的採用紀錄（`results/prereg09/rank/apply_report.json` 釘 LF SHA `691cd3a7…`，usable、finalized）。有符合的 09 而交 08／07／04／R20／R10，要 `--not-09-reason "PREREG_09 rev N: …"`（N ≥ 4）。為了不把 09 當成 08、07、04 或 R20，04、07、08 的規則函式各加了「09 跳過」的行；`status` 在最終是符合的 09 時不印 04／07／08 的違反，提示改成 09；`prepare` 印「09 是沒在驗證集打過分的組合」。log 沒有 09 紀錄時，所有輸出、紀錄檔、回傳值和 tag `p08-testday-pre09` 的版本逐位元組相同（`tests/test_t09_official_log.py` TagVersusNew，含 07、08 的流程）。
12. **演練的輸入**：08 的 `t08_fake_tests.py from-val` 寫的假塊沒有逐局面陣列（`pos_side` 長度 0），09 的 EX 算不出來。`tools/t09_fake_tests.py from-val` 先呼叫它（不改），再把同一批驗證盤的逐局面陣列照原順序抄進每一塊（`pos_game` 改成塊內編號），重寫 `p1_<組>_report.json` 的塊列；`<組>_p1.npz` 和 08 讀的陣列都不變，所以 08 的輸出仍等於基準。只限 fake_from_val 的紀錄、只限 08 的 apply 之前。
13. **基準**：`C10_SLOW=1` 的 `tests/test_t09_dryrun.py` 在同一棵假樹上跑 R20、04、07、08、09，R20／04／07／08 的 38 個輸出等於 `results/test09/design/baseline_pre09.json`（第 5.1 節第 3 步）。09 自己的基準 `results/test09/design/t09_fake_golden.json` 是程式寫好之後第一次通過的演練記下的（同 07、08 的做法），照實揭露。
14. **`.gitattributes` 加 `results/test09/** -text`**：之前 commit 的 `results/test09/design/baseline_pre09.json` 在 CRLF 的工作樹會被看成已修改；分支上已重新 checkout（內容和 commit 的相同）。合併回 main 之後在 main 也要做一次（刪掉工作樹的那份再 `git checkout --`）。
15. **防誤改閘門**：CLAUDE.md 的閘門指令（十四個檔）加 `tests/test_t09_hygiene.py`，共十五個檔；唯一允許的失敗不變。
16. **K5-09（10/31 23:59）的條件**：分支合併回 main；閘門只有已知的那一項失敗；`C10_SLOW=1` 的 t09 演練全過；本 rev 3 已 commit；tol_E 的實測已 commit（不需要另跑 4090：09 不碰 GPU，tol_E 用 08 的演練樹量）。
17. **取代的條文**：PREREG_08 第 5.4 節「有符合的 08 時最終是 08」→ 有符合的 09 時最終是 09（第 5.3 節已寫，照 PREREG_07 第 5.5 節的前例不改原文）；第 5.2 節「GPU 接縫…和 08 的 `data/prereg08/val/` 前 2 塊比」照第 4 點（比的就是那兩塊）；第 6 點的 Q02H 核對見上。
18. **審查**：建置代理只做了自我核對和測試，沒有獨立審查；合併前由主 session 決定要不要另派審查，照實揭露。

## 14. rev 4：09 GPU 接縫的標準化 EX 只判 42 維（2026-10-09，在任何測試資料之前）

決定者：Claude（使用者 10/2 授權「只要對最終排名有幫助」，10/9 再說「目標始終是 Private 分數和排名」），由主 session 在建置代理回報第 13 節第 5 點的實測之後決定。依據是那一次 4090 實測；第 5.2 節寫明 tol_E 在拿到測試資料前定，本節同樣在任何測試資料之前。

1. **改什麼**：09 的 GPU 接縫（第 5.2 節、第 13 節第 4 點）的「標準化 EX 最大差 ≤ 0.05」，只對 44 維 EX 中**兩維 frac_lowp 以外的 42 維**判定。兩維 frac_lowp（目標方、對手方「log p < log 0.05 的手數比例」，`t_frac_lowp`、`o_frac_lowp`）**只記錄**：最大差、有跳動的盤數，寫在 `gpu_seam.json` 的 `frac_lowp_record_only`，44 維全部的最大差另記 `max_abs_std_ex_all_44`。
2. **不變的**：lp_P1TO_E 每盤最大差 ≤ tol_E＝0.04（這一條才真正保護預測）；failed／counts／n_moves 完全相同；標準化 X 只記錄；其他所有接縫條件、Q08 對得上 08、本機接縫、失敗率、方向檢查都不變。
3. **為什麼**：
   - 第 13 節第 5 點的 4090 實測（private 前 1,000 盤，不同批次組合）：兩維 frac_lowp 在 fp16 捨入下跳一格，最大差 0.254，1,000 盤中 24 盤有跳動；其他 42 維最大 0.016；lp_P1TO_E 最大 0.011。這兩維是有門檻的計數，一兩手跨過門檻就跳一格，不是環境漂移，也幾乎不影響預測。
   - 測試日的 seam 和驗證塊同樣分塊，兩次實測都是 0；但若 cuDNN 換了演算法，seam 會出現這個量級的跳動。照 0.05 判全部 44 維，接縫會誤判失敗，09 這一季就不用。誤判的代價是名次的實際損失。
   - **不採用**「整個 EX 容許差放寬到 2.5 × 0.254 ≈ 0.64」：那樣會讓其他 42 維幾乎不被檢查（它們的實測雜訊只有 0.016），等於關掉 EX 的接縫。
4. **程式**：`tools/t09_rank.py` 的 `EX_RECORD_ONLY`、`compare_sets`／`gpu_seam_record`（判定用 `max_abs_std_ex`＝42 維；另記 `max_abs_std_ex_all_44`、`frac_lowp_record_only`）；`tools/t09_fake_tests.py` 的 `measure-tol` 輸出（42 維、44 維、frac_lowp 分開記）；`tools/official_log.py` 的 09 行（本分支自己加的行；對 tag `p08-testday-pre09` 仍是只加行）把 `max_abs_std_ex` 讀成 42 維的值，並把 `P09_MIN_REV` 改成 5。在新規則下重跑同一次 4090 實測：42 維最大差 0.0161（≤ 0.05）、lp_P1TO_E 0.0113（≤ 0.04），也就是不同批次組合的 1,000 盤也會判過（紀錄 `results/test09/design/ex_rule_rev4_check.json`；rev 3 的 `tol_e_measurement.json` 不改）。
5. **測試**：`tests/test_t09_rank.py::Seam::test_frac_lowp_jump_recorded_only`（只有兩維 frac_lowp 跳 0.25：判過、記下 2 盤；其他任一 EX 維移動 0.06：判失敗）、`test_ex_tolerance`（只在 42 維裡挑）、`tests/test_t09_hygiene.py`（`EX_RECORD_ONLY` 和 `P09_MIN_REV = 5`）、`tests/test_t09_official_log.py`（推翻要 rev 5 起）。
6. **取代**：rev 3 第 13 節第 4 點「標準化 EX … 最大絕對差 ≤ 0.05」改成本節第 1 點；第 13 節第 5 點最後的「要不要另寫 rev 4 由主 session 決定；沒有修訂就維持 0.05」由本節定案；第 13 節第 10 點和 `--bug-revision`／`--not-09-reason` 的「N ≥ 4」改成 **N ≥ 5**（本 rev 4 的 commit 早於任何測試日紀錄，不能拿來推翻）。

### rev 4 補記（2026-10-09，兩份獨立審查之後、合併之前；仍在任何測試資料之前）

兩份獨立審查（主 session 派出）都沒有 blocker／major。照審查修的 minor（第 13、14 節的判定和數值都不變，這裡只補充）：

7. **逐局面陣列的完整性（加強第 13 節第 3 點）**：`t09_rank.py` 讀 08 的每一塊時，除了檢查五個逐局面陣列都在，還要求它們等長、`pos_game` 在 [0, 盤數)、`pos_side` 是 0 或 1，而且每盤每方的局面數等於 `counts`（`bincount(pos_game × 2 + pos_side, minlength = 2 × 盤數)` 排成 [盤數, 2] 要和 `counts` 完全相同），否則拒絕（2）。這個等式在 08 的 96 塊驗證塊、4090 演練的 GPU 塊、`t09_fake_tests.py add-positions` 寫的演練塊上都成立（154 塊，0 不符）。測試 `tests/test_t09_rank.py::Chunks::test_incomplete_positions_refused`（空、截短、只少一個陣列、翻一手的執色、pos_game 超出範圍）。
8. **說明文字**：`check_bug_revision` 的說明和 `tests/test_t09_hygiene.py` 的略過訊息改成 N ≥ 5、`P09_MIN_REV = 5`；`docs/TESTDAY_09.md` 寫明 `apply` 經由 08 的前提也要求 R20 的第一次上傳已記錄、09 判不用或從沒 prepare 時最終照 `docs/TESTDAY_08.md` 第 6 步，並寫出十五個檔的閘門指令全文。
9. **防誤改**：`tests/test_t09_hygiene.py` 把 `docs/COLAB_RUN.md` 列入凍結（對 tag `p08-testday-pre09` 不改、不新增），並檢查 test02／04／07／08 的設計紀錄資料夾（`results/test02/design`、`results/test02/null`、`results/test04/design`、`results/test07/design`、`results/test08/design`）自 tag 起沒有新增檔案；測試日真正的輸出寫在其他資料夾，不受影響。
10. **修訂編號**：修訂表 rev 3 那一列寫的「之後要推翻 09 的登記結果，要 rev 4 起的修訂」由第 14 節第 6 點取代：推翻要 **rev 5 起**（rev 3、rev 4 都是測試日程序本身）。不改那一列的原文。

> 2026-10-10 追加（只加這一行，不是修訂列）：08 的 P1 在 4090 不能用時可能改在 Colab 上算（`docs/PREREG_11_GPU_FALLBACK.md`）；09 讀 08 的塊，GPU 接縫和所有判定照本登記不變。
