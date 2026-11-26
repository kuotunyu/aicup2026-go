# 事前登記 19：競賽二在 Q09 上加「題目層級（set）模型」項（Q19＝Q09＋4·SET）

**白話版**：09 的 6 個小模型是「每一盤各自猜等級，再把一題 5～20 盤的猜測平均」。prereg19 探索（只用訓練玩家）另外訓練了一個「一次看一整題」的小網路（S19）：每一盤先變成 64 個數字，同一題的盤把這 64 個數字取平均，再判斷棋力。把它當成 09 融合的第 7 項，inner 上比 09 多 **+0.0231**（95% 區間 [+0.0104, +0.0352]），全部 120,544 題單色題多 +0.0285，黑白都正；盤數越多增益越大（5 盤約 +0.01，15～20 盤約 +0.04）。另一個代理的獨立稽核（`85cbe04`）確認沒有洩漏、數字由已 commit 的模型逐位元重現。本登記把那一個候選**一字不改**地寫死（不重訓、不再挑），用全部 inner 擬合一次融合參數凍結，驗證題只做無標籤檢查，測試日接在 09 之後多跑一步 CPU、多上傳 1 次。**競賽二已經沒有驗證集打分機會**，19 和 09 一樣只靠 inner 採用，交件時是「沒在驗證集打過分的組合」，報告要寫明。順序 R20 → 04 → 07 → 08 → 09 → 19（之後才是 14）。

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。對照組：**PREREG_09 的 Q09**（`results/prereg09/rank/frozen_config.json`，inner 0.7114420614163589、單色 0.7145431684624712）。
- 探索（只用 fit／inner 訓練玩家、沒打分、沒碰 `data/local`）：`results/prereg19/explore/`。
  - `PLAN.md`（`937f48c`，在任何候選數字之前 commit；第 9 節是審查修正）；程式 `p19_lib.py`、`p19_train.py`、`p19_eval.py`、`test_p19.py`；
  - `train_record.json`（`310f39c`，3 個種子只訓練一次）、`eval_report.json`（`17f1974`，只判一次，閘門全過）、`RESULTS.md`、`audit.json`（`85cbe04`，獨立稽核：沒有洩漏，eval 由已 commit 的模型逐位元重現）。
- 點子的來源：外部 AI 審閱者（Codex）10/20 提出的 `analysis/codex_ideas_1020.md` 第 (b)1 節，使用者 10/20 同意探索；採用由主 session 決定、使用者同意（10/20）。
- **只做競賽二（TEAM_11248）**；19 的任何輸出都不進競賽一，不讀競賽一的任何檔案。
- 代號：
  - **S19**：題目層級 set 模型（架構見第 1.2 節），3 個種子 0、1、2。
  - **lp_SET**：一題的 10 維對數機率＝3 個種子機率平均再取 log（下限 1e-300）。**E_SET**＝4·lp_SET（該題沒有可用盤時是 0）。
  - **Q09**：PREREG_09 的首選（OP、O4、SWA、SWB、HS、P1TO_E 六項）。**Q19**＝Q09 的六項加第 7 項 SET（項的順序 OP、O4、SWA、SWB、HS、P1TO、SET），融合參數重新擬合。
- 本登記寫四件事：做什麼（第 1 節）、怎麼凍結和檢查（第 2 節）、測試日怎麼用、什麼時候放棄（第 4、5 節）。偏離寫在第 3 節。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `faac6e8` | 第一版，2026-10-20。commit 時的狀態：① 探索已完成並 commit（上列），**閘門已判過一次**（G1 +0.0231、G2 下界 +0.0104、G4 單色 +0.0285、黑 +0.0306、白 +0.0264）；這些數字是本登記的依據，不是本登記之後的結果。② 3 個 S19 模型檔在 `data/prereg19/models/`（SHA 在 `train_record.json`），本登記**原樣凍結、不重訓**。③ 還沒有：任何凍結的融合參數、任何驗證題的 lp_SET 或 Q19 機率、任何 19 的工具程式（`tools/p19_*.py`、`tools/t19_*.py`）、`results/prereg19/rank/`、`results/test19/`。④ 打分紀錄帳自 H8R 之後沒有新列；本登記也不加任何列。 |
| rev 2 | `768f7d2`（工具） | 2026-10-20，在 `freeze` 之前：凍結和驗證題檢查的實作差異（第 9 節）。commit 時：`results/prereg19/rank/` 不存在；還沒有任何凍結參數、驗證題的 lp_SET 或 Q19 機率。工具只有單元測試（合成資料），獨立稽核在 `freeze`、`apply` 之後做（第 2.3 節）。 |
| rev 3 | `b4d32e3` | 2026-10-20，測試日套件的實作差異（第 10 節），在任何測試資料之前、合併之前。commit 時：凍結和驗證題檢查已完成（`c950816` freeze、`fd50642` apply、`7108916` 獨立稽核、`1e19a0e` finalize，usable true）；`results/test19/` 只有 `design/t19_fake_golden.json`（假樹演練）；沒有任何測試資料。之後要推翻 19 的登記結果，要 rev 4 起的修訂。 |

之後的修訂只准加行（不改本表已有的列和第 0–8 節的原文）。預定：rev 2＝凍結工具和驗證題檢查的實作差異（在 `freeze` 之前 commit）；rev 3＝測試日工具的實作差異（合併前 commit）。要推翻 19 的登記結果（`--bug-revision`、`--not-19-reason`），要 **rev 4 起**、而且晚於被推翻的紀錄。

## 0. 為什麼做這個（都不看答案；inner 已被多次使用，數字偏樂觀）

- **09 的做法**：6 個成員每一盤各自把 1,540 個 KataGo 探針數字翻成 10 級機率，一題的 log 機率取平均乘 4 放進融合；模型訓練時從沒看過「一整題」。
- **探索的假設**：直接用「一整題」訓練（同一位玩家、同一執色的 5／10／15／20 盤當一筆樣本，損失是整題的 NLL），模型會直接學「跨盤穩定的訊號」。結果（`eval_report.json`，判一次）：
  - G1：inner 評估假題（127,200 題）0.7114 → **0.7346（+0.0231）**；G2：配對 PlayerBootstrap 95% [+0.0104, +0.0352]；G4：單色 0.7145 → 0.7430（+0.0285 [+0.0199, +0.0361]），黑 +0.0306、白 +0.0264。
  - 依盤數（只記錄）：單色 5 盤 +0.0113、10 盤 +0.0297、15 盤 +0.0393、20 盤 +0.0411；評估假題 5–7 盤 +0.0104 [−0.0024, +0.0229]（區間跨 0）。
  - 每折 w_SET：2.71、0.62、3.50、3.82、2.52（沒有打到界限；第 1 折偏向舊項）。
  - 只記錄的消融：SET 取代 P1TO（不要 6 成員）+0.0215；SET 單獨（不融合）0.7286；每個種子單獨 +0.0185／+0.0223／+0.0196，三個的集成最好。
- **誠實的增益估計**：inner 已被 P1、08、09、prereg12、15、16、17、18 重複使用，選擇偏差約 0.003；這次的下界 +0.0104 仍在偏差之外。測試題的結構（盤數分布、執色）和 inner 假題不完全相同，真實增益可能比 +0.023 小。5 盤題的增益最小。
- **為什麼還做**：
  1. 特徵（08 的 P1 TO 每盤平均）、09 的 6 成員、融合形式、校準、規則 B、測試日的 GPU 計算都和 09 相同，只多一項；S19 只用 fit 玩家（probe）訓練和選步，inner 和驗證玩家都不在裡面（探索的接縫 D）。
  2. 融合可以把 w_SET 擬合成 0：最壞情況接近 Q09。
  3. 測試日不用多跑 GPU：SET 讀的就是 08 已算好、09 也讀的那份 P1 塊。

## 1. 來源（全部釘死）

### 1.1 沿用（不重算、不改）

- 09 的全部凍結檔和測試日路徑：`results/prereg09/rank/*`（`frozen_config.json`、`p1e_map.npz`、`rank_val_probs.csv`、`pred_A.csv`、`pred_B.csv`、`val_lp_p1e.npz`、`apply_report.json`、`apply_audit.json`）、`tools/t09_rank.py`（指紋 `results/test09/design/p09_path_fingerprint.json`、tag `p09-testday-pre10`）。
- 08、07、04、R20 的凍結檔和測試日路徑（各自的指紋和 tag）；KataGo 程式和權重；P1 的特徵化。
- inner 的融合和評估一律走 `tools/p09_rank_fuse.py` 的 `Inner`（import，不改；它再用 `tools/p08_rank_fuse.py` 的 `run_terms`、`masked_inner`）、`tools/p04_rank_fuse.py` 的 `gen_fit_record`、`gen_params`、`gen_logits`；inner 的 P1 特徵 `data/prereg12/q09_inner/`（`results/prereg12/explore/finetune/ft_eval.py` 的 `InnerPrep`，import，不改），和探索相同。
- 探索的程式只當規格（釘 LF SHA，不 import）：`results/prereg19/explore/{p19_lib.py, p19_eval.py, p19_train.py}`，以 `85cbe04` 的版本為準。

### 1.2 S19 模型（照探索原樣，不重訓、不再調）

- 架構（`PLAN.md` 第 3.1 節）：每盤 h(x)：Linear(1540, 256) → GELU（erf 版）→ Linear(256, 64) → GELU（推論時 dropout 關閉）；題目內聚合 v＝該題可用盤 h 的平均；head：Linear(64, 32) → GELU → Linear(32, 1)＝棋力分數 f；累積 link（同 09 的 mlpclm：θ₁＝t0，θ_k＝t0＋Σ softplus(dt)，P(y ≤ k)＝σ(θ_k − f)，相鄰相減、下限 1e-12 再取 log）。
- 輸入：每盤 1,540 維 P1 TO 特徵（目標方 770 維接對手方 770 維，對手方沒有手時是 NaN），用 **09 凍結的 `p1e_map.npz` 的 μ_X、σ_X** 以 `p09_forward.standardise_x` 標準化（非有限值 → 0）。不用 EX。
- 可用盤：沒失敗、至少 1 手目標手（`p09_forward.usable_games`），而且棋譜解析成功（同 09 的 P1TO 項的遮罩）。題目沒有可用盤時 E_SET＝0 向量。
- 3 個種子的模型檔＝探索訓練的那 3 個檔，**位元組不變**複製到 `results/prereg19/rank/s19_seed{0,1,2}.npz`（每個約 1.66 MB，進 git），SHA-256 必須等於 `train_record.json` 的 `models_sha256`：
  - seed 0：`b0d7ea70e77d4e2ab145e42415b13a5c987c68d676c0c2a07760fb91d6859949`（第 10,000 步）
  - seed 1：`8b8729ea5a0fed3948134cc1a8c8bc54302e9e3b0a8bda7d3a3945b5e70b1ffd`（第 8,500 步）
  - seed 2：`b3bc704a4ba19329d99dacf0724c7a9d70f325f5dbb8d08ee6c08c122b7c9519`（第 7,000 步）
- 集成：lp_SET＝log( max( 3 個種子機率的平均, 1e-300 ) )（`p09_forward.lse_mean`）。
- **前向（寫死）**：inner、驗證題、測試日一律用 **numpy float64 前向**（權重存 float32、讀進來轉 float64；GELU 用 `scipy.special.erf`），不用 torch；每盤的 h 以每 8,192 列一塊計算，題目平均用稀疏平均矩陣（同探索 `p19_lib.mean_matrix`）。這段前向寫成共用的純 numpy 模組 `tools/p19_forward.py`（不 import torch），由凍結工具和測試日工具共用。
- 乘 4：融合時用 E_SET＝4·lp_SET，同探索。

### 1.3 融合（唯一一組參數）

- 公式同 PREREG_09 第 2.1 節，只多第 7 項：p＝softmax(e^a (n/12)^γ (m_C5 ＋ w_OP m_OP ＋ w_O4 E_O4 ＋ w_SWA E_SWA ＋ w_SWB E_SWB ＋ w_HS E_HS ＋ w_P1 E_P1E ＋ w_SET E_SET ＋ Bβ))，規則 B（`tools/decide.py`）。
- 界限、起點、PENALTY、MAXITER、等級加權都不變（`p04_rank_fuse.gen_fit_record`：所有 w 在 (0, 4)、a 在 (−4, 4)、γ 在 (−1, 2)，起點 `rf.STARTS`，L-BFGS-B；是確定性的計算，沒有亂數種子）。
- **凍結的測試日參數只擬合一次**：用**全部** inner 擬合假題（25,400 題，同 09 凍結時用的那一批）重擬合 7 項的 θ（`p4.gen_fit_record(mf, Ef, n, y)`，Ef＝七項在擬合假題上的 E，順序同上），同 09 的做法（PREREG_09 2.6）。不看 OOF、不看任何驗證題，再也不改。

## 2. 凍結、無標籤檢查、稽核（CPU；`tools/p19_set.py`）

### 2.1 `freeze`（只跑一次）

全部成立才寫凍結檔；任一接縫不過就停（不寫凍結檔、不算判過，查明是程式錯誤後在 rev 2 起的修訂寫明再跑）：

1. **接縫 1**：Q02H、Q08、Q08 單色等於 `p09_rank_fuse.REF`；Q09 inner 0.7114420614163589、Q09 單色 0.7145431684624712（`results/prereg09/rank/tune_report.json`）；差都 ≤ 1e-12。
2. **接縫 M**：3 個模型檔（`results/prereg19/rank/` 的副本）的 SHA 等於第 1.2 節，而且等於 `data/prereg19/models/` 的原檔；`train_record.json`、`eval_report.json`、`audit.json` 的 LF SHA 等於 `85cbe04` 的版本。
3. **接縫 F（重現探索的判定）**：用 `tools/p19_forward.py` 和複製的模型檔算出的 Q19 OOF，inner 分等級平均等於 `eval_report.json` 的 0.7345843051712626、單色 0.7430172362615272，每折的 w_SET 等於 `eval_report.json`（差都 ≤ 1e-12）。這證明正式工具和探索是同一個東西；沒有任何新的選擇。
4. 自己的題目平均（只用可用盤）用在 4·lp_P1TO_E 上，擬合／評估／單色三組等於 `Inner.E_p1`（差 ≤ 1e-12，同探索的接縫 E）。

過了之後：

- 第 1.3 節擬合 θ，寫 `results/prereg19/rank/frozen_config.json`：組合、項、倍數 4、參數（`p4.gen_params`）、界限有沒有打到、擬合紀錄（各起點的 loss）、樣本內分數（只記錄）、5 折 OOF 紀錄（＝探索的判定）、模型檔 SHA 和步數、方向檢查的參照值 ρ_eval、ρ_mono（只用 E_SET 的期望等級對 Q02 OOF 的期望等級，有可用盤的題，同 PREREG_09 2.6 的定義）、釘住檔的 SHA、工具 SHA、`.venv` 的 python／numpy／scipy 版本。
- `freeze` 在 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 任一有設時拒絕；不 import torch。
- 凍結檔、3 個模型檔、`freeze_report.json` 一起 commit。

### 2.2 `apply`（驗證題，**不打分**）

- **前提**：凍結檔和模型檔已 commit；08 的驗證塊 SHA 等於 `results/prereg08/feat/val_report.json`。
- **先重現**：Q02、Q02H、Q08、**Q09** 在驗證題上的機率，要和 `results/prereg04|07|08|09/rank/rank_val_probs.csv` 逐位元組相同（09 的那份用 09 的程式 `p09_rank_fuse.q09_core` 和凍結的 `p1e_map.npz`），不同就停。
- 只讀 `data/local/rank_val_questions.csv` 的題號、盤數、執色、SGF（經由已登記工具 `p07_rank_fuse.q02_validation` 和 `check_submission --questions`，同 PREREG_08／09），**不讀答案**。
- 算驗證題的 lp_SET（08 的 96 塊驗證塊的 `means`、`counts`、`failed`，可用遮罩同 Q09 的 P1TO 項），用凍結的 θ 算 Q19，寫 `results/prereg19/rank/rank_val_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`check_submission.txt`、`val_set.npz`（每題的 lp_SET、有沒有可用盤、題號順序；給測試日的本機接縫用）。
- `apply_report.json` 的 `usable` 要以下全部成立才為真（**都不看答案**）：
  - Q02、Q02H、Q08、Q09 逐位元組重現；
  - 可用遮罩等於 09 的 `val_lp_p1e.npz` 的 `usable`；
  - **方向檢查**：只用 E_SET 的期望等級對 Q02 的期望等級，Spearman ρ ≥ max(0.5 × ρ_eval, 0.078)（ρ_eval 取 `frozen_config.json`）；
  - 格式檢查過（`decide.py`、`check_submission.py`）；
  - 稽核（下一節）沒有找到影響預測的程式錯誤，由 `apply --finalize` 寫入。
- **只記錄**：Q19 和 Q09 在規則 B 下不同的題數、等級差的分布（依盤數分開）；E_SET 和 E_P1E 的期望等級的 ρ。
- **不跑 `tools/score.py`，不讀 `rank_val_answers.csv`**。打分紀錄帳不得出現 `prereg19` 列。

### 2.3 獨立稽核（凍結之後、`finalize` 之前）

- 另一個代理（看不到答案、不看任何分數；做不到就由建置代理做一輪對抗式自審，並寫明是哪一種）：
  - 模型檔 SHA；用自己寫的 numpy 前向重算驗證題的 lp_SET，和 `val_set.npz` 相同（≤ 1e-9）；
  - 用 `frozen_config.json` 的 θ 獨立重算 Q19 的驗證機率，和 `rank_val_probs.csv` 相同（≤ 1e-12）；
  - 乘 4 只做一次、項的順序、可用遮罩；Q09 的重現；`decide` 的輸入輸出；`freeze` 的 θ 由同一個確定性擬合重現（獨立重跑 `gen_fit_record`，θ 相同 ≤ 1e-9）；
  - 沒有讀答案、沒有打分、沒有 inner 以外的玩家進入擬合。
- 結果寫 `results/prereg19/rank/apply_audit.json`（欄位 `program_errors_affecting_predictions`）並 commit，再 `apply --finalize`。

### 2.4 不做的

- 不重訓 S19、不改種子、步數、架構、倍數、項的組合；不加 EX；不試「SET 取代 P1TO」（探索只記錄，比加一項略低）。
- 不用 inner OOF 以外的任何東西挑參數；凍結的 θ 只擬合一次。

## 3. 沒有驗證集打分（偏離，寫明）

- 競賽二的 4 次驗證集打分已用完（H8R 是最後一次）；PREREG_08 5.4 寫「不交沒在驗證集打過分的組合」，PREREG_09 第 3 節已對 Q09 偏離一次。
- **本登記再偏離一次**：Q19 只靠 inner 閘門（探索判一次）、驗證題的無標籤檢查（第 2.2 節）和獨立稽核就上場。決定者：主 session（使用者 10/2 授權「只要對最終排名有幫助」、10/9 再說「目標始終是 Private 分數和排名」、10/20 同意採用）。
- 理由：第 0 節「為什麼還做」三點；閘門比 09 嚴（G1 不加容許、單色黑白都 ≥ 0），而且獨立稽核確認探索的判定是真的。
- 風險（揭露）：inner 被重複使用（選擇偏差約 0.003）；權重在折之間不穩定（第 1 折 w_SET 0.62）；5 盤題區間跨 0；S19 沒有用 EX。最壞情況是 Q19 和 Q09 差不多。
- 取代的條文（不改原文，照 PREREG_07 第 5.5 節的前例）：PREREG_08 5.4「不交沒在驗證集打過分的組合」對 Q19 例外；PREREG_09 5.3「有符合的 09 → 09」改為第 4.3 節。

## 4. 測試日（全部 CPU，不多跑 GPU）

### 4.1 保護舊路徑

- 一律不改：R20、04、06、07、08、09、10 的測試日路徑（各自的指紋和 tag）；`results/prereg01`–`prereg18/**`、`results/test02|04|06|07|08|09|10/**`；`tools/p0*`、`p1[0-8]*`、`t0*`、`t10_*`；`third_party/katago/**`。
- `results/prereg19/explore/**` 只准加檔；`results/prereg19/rank/` 的凍結檔在 `finalize` 之後一律不改。
- 只准加行：`tools/official_log.py`、`.gitattributes`、`.gitignore`、本登記。
- 動工前在 main 打 tag **`p14-testday-pre19`**（競賽二最後一個測試日套件 14 合併之後、19 之前的狀態）：19 的測試要求 `official_log.py` 對這個 tag 只加行；R20／04／07／08／09 的假測試演練輸出要等於已 commit 的 `results/test09/design/baseline_pre09.json` 和 `t09_fake_golden.json`。
- **14 要跟著認得 19**：`tools/t14_copy.py` 和 `official_log.py` 的底檔資料夾清單只加 `results/test19/rank`，14 的底檔規則加上本登記的 4.3 節；14 的定義、規則、釘住值內容不變（細節在 rev 3 寫明）。

### 4.2 `tools/t19_rank.py apply`（09 的 apply 之後；幾分鐘）

- **前提**（照順序，第一個不成立的決定結束碼）：git 乾淨（正式模式）；不設 BLAS 執行緒變數；prereg19 的凍結檔 SHA 不變、`apply_report.json` 是 `usable: true` 而且 finalized（不成立是 3：19 沒採用）；`.venv` 的 python、numpy、scipy 等於 `frozen_config.json`（不等是 4）；K6-19；`t09_rank.py apply` 的全部前提（經由它，08、07、04、test02、R20 的第一次上傳、T1、H7R、H8R）；最後 **09 的 `t09_apply_report.json` 是同一模式、結束碼 0、usable、已 commit**（09 判不用時 19 也不用，結束碼 3），09 報告綁定的每一個輸入、09 的輸出都沒變。
- **輸入**：08 的 P1 塊（`data/test08/p1/<組>/`，逐塊 SHA 等於 08 的報告，同 09）。
- **步驟**：
  1. **Q09 對得上 09**：用 09 的程式和同一份輸入重算 Q09，和 09 已 commit 的 `rank_test_probs.csv` 比：題號、順序、盤數相同，機率差 ≤ 1e-12，規則 A、規則 B 的等級全同；可用盤等於 09 的 `test_lp_p1e.npz`。不成立是 4。
  2. **SET 的 GPU 接縫（只記錄）**：seam 400 盤（08 的 seam 塊）和 08 的驗證塊第 0、1 塊，每盤當成一題算 lp_SET，記最大差。**不判定**：09 的 GPU 接縫（同一批塊，lp_P1TO_E ≤ 0.04）已判過；09 的 4090 演練樹已刪，沒辦法在測試資料之前為 SET 量一個有根據的容許差，猜一個太緊的值會讓 fp16 的捨入誤殺 19（理由同 PREREG_09 rev 4）。
  3. **本機接縫**：seam_val 用 09 的 seam_val 輸入（test02／04／07 的 seam_val 檔、`val_hs.npz`）加已 commit 的 `results/prereg09/rank/val_lp_p1e.npz` 和 `results/prereg19/rank/val_set.npz` 算 Q19，和 `results/prereg19/rank/rank_val_probs.csv`、`pred_B.csv` 比。**09 的 `seam_local.json` 逐位元組相同時，19 要逐位元組相同**（不同是 4）；否則用 test04 的容許差、規則 B 最多差 4 題（超出是 3）。題號、順序、盤數不同是 4。
  4. **方向檢查**：每組（public、private）只用 E_SET 的期望等級和 Q02 的期望等級的 Spearman ρ ≥ max(0.5 × ρ_mono, 0.155)（ρ_mono 取 `frozen_config.json`）；不成立是 3。
  5. **輸出**：`results/test19/rank/` 的 `rank_test_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`submission_rank.csv`（＝`pred_B.csv`）、`test_lp_set.npz`、`t19_apply_report.json`；`check_submission --official` 失敗是 4（輸出刪掉）。
- P1 失敗率沿用 09 的判定（09 結束碼 0 已涵蓋，同一批資料）。
- **結束碼**同 t09：0 完成；2 拒絕（前提不符、輸出已存在、工具錯誤）；3 登記的「19 不用」（寫 `results/test19/not_used.json`，不重跑；最終照 PREREG_09 5.3）；4 本機漂移或對不上（寫 `rank/attempt_verdict.json`，停下查，查好 `--rerun --reason`）。出現 3 或寫好 `t19_apply_report.json` 之後，只有 `--bug-revision N`（N ≥ 4、那一列的 commit 晚於被推翻的紀錄）才能重跑。K6-19 過後已有成功的那一次：2，那一次仍有效。
- 輸出只寫 `results/test19/`；演練只寫 `results/test19_rehearsal/` 或 repo 外。不讀競賽一；不讀驗證答案；不跑 `score.py`；不 import torch。

### 4.3 上傳順序和最後一版（競賽二）

- **順序**：R20（11/04）→ 04 → 07 → 08 → 09 → **19**；14（若用）一律最後。每一份都要使用者在對話裡說 OK。19 在 09 上傳之後才 `prepare`。
- **最後一版**：有符合的 19 → **19**；沒有 → 照 PREREG_09 5.3（09 → 08 → 07 → 04 → R20，T1 FAIL_OPENING 時 R10）。
- 「符合」＝`official_log.py` 每次用 19 報告裡的數字重算第 4.2 節的條件，加上：19 的採用紀錄（`apply_report.json` 釘 LF SHA、usable、finalized）、19 報告綁定的那一筆 09 已上傳而且仍符合 PREREG_09 5.3、輸入綁定沒變、K6-19。
- 有符合的 19 而要交別的，要 `--not-19-reason "PREREG_19 rev N: …"`（N ≥ 4，晚於那幾筆 19 的 prepare）。
- public 分數不進判定；19 比 09 低 0.05 以上時只查錯，查不到就維持本節的結果。
- **和 PREREG_14 的關係**：14 一律套在「現行規則選定的最後一版」上；19 是最後一版時，14 的底檔就是 19（`results/test19/rank/submission_rank.csv`），14 的底檔規則包含本節。
- 取代的條文（不改原文）：PREREG_09 5.3「有符合的 09 時最終是 09」→ 有符合的 19 時最終是 19。

## 5. 截止點（台灣時間；沒趕上就照表做）

| 截止點 | 條件 | 沒趕上 |
|---|---|---|
| K1-19（無日期） | 本登記 rev 1 的 commit 早於任何 19 的工具程式、凍結參數和驗證題 lp_SET | 這一季不做 19 |
| K4-19 10/27（二）23:59 | `freeze`、`apply`、稽核、`apply --finalize` 都 commit | 這一季不用 19 |
| **K5-19 10/31（六）23:59** | 測試日工具、`official_log.py` 的加行、測試、演練（`C10_SLOW=1`）通過、rev 3 已 commit，以 `--no-ff` **合併回 main** | 這一季不用 19 |
| **K6-19 11/19（四）23:59:59** | 19 在官方測試集有一次成功的正式執行並 commit（同 K6-09） | 最終照 PREREG_09 5.3 |

## 6. 另一場

- 競賽一不用 SET、P1 或任何 19 的輸出；不讀競賽一的測試資料。

## 7. 防誤改閘門

- AGENTS.md 的閘門指令加 `tests/test_p19_hygiene.py`（prereg19 凍結檔、打分紀錄帳沒有 `prereg19` 列、工具的 import 限制、本登記只准加行）和 `tests/test_t19_hygiene.py`（測試日工具的登記值、`official_log.py` 對 tag `p14-testday-pre19` 只加行、舊路徑不變）。
- 允許的失敗不變（AGENTS.md 那 4 項）。

## 8. 揭露（寫進 RESULTS 和競賽二報告）

- **事前與事後的界線**：S19 的架構、抽題、步數規則、種子、倍數 4、「加一項」的候選都在 `PLAN.md`（任何候選數字之前）寫死；閘門只判一次；本登記在判定和稽核**之後**寫，所以登記本身不是盲的——它只是把已判過的那一個候選凍結起來，沒有再挑任何東西（第 2.4 節）。
- **inner 的重複使用**和**沒有驗證集打分**（第 3 節）。
- **點子的來源**：外部 AI 審閱者（Codex）10/20 的提案；探索、凍結、測試日程式由 Claude 寫，獨立稽核由另一個代理做。
- **外部資源**：同 PREREG_08／09（KataGo 程式和 HumanSL 權重）；S19 是本隊自己的模型（PyTorch CPU 訓練、numpy 推論）。
- **消融**（報告用，都來自已 commit 的檔）：inner 上 Q09 → Q19（+0.0231）；依盤數；SET 單獨；SET 取代 P1TO；每個種子單獨。
- 生成式 AI 協助的部分照規則揭露；報告草稿 `report/` 競賽二的方法和消融依 19 結果改寫（由主 session 做）。

## 9. rev 2：凍結和驗證題檢查的實作差異（在 `freeze` 之前）

工具 `768f7d2`：`tools/p19_forward.py`（純 numpy 的 S19 前向，`p19_lib.py` 的移植）、`tools/p19_set.py`（`freeze`、`apply`、`apply --finalize`）、`tests/test_p19_set.py`（合成資料：前向、題目平均、抽成 1 題的結果都和探索的 `p19_lib.py` 逐位元相同；決策和稽核綁定的單元測試）。第 1、2 節的方法、模型、融合和閘門都沒改；下面是第 2 節沒寫清楚或做法不同的地方。

1. **inner 的特徵**：`freeze` 自己讀 `data/prereg12/q09_inner/`（`inner_X.npy`、`inner_meta.npz`，和 `ft_eval.InnerPrep` 相同的內容），不 import `ft_eval`（它會 import `tools/p09_map.py`）；另加**接縫 P**：`inner_X.npy` 的 SHA 等於 `results/prereg09/rank/check_report.json` 綁定的那一份（同探索）。
2. **接縫 F 比登記嚴**：除了 Q19 的 inner 和單色分等級平均，5 折的 7 個融合權重全部要等於 `eval_report.json`（差 ≤ 1e-12）。
3. **版本**：`freeze` 要求 `.venv` 的 python、numpy、scipy 等於 09 的 `frozen_config.json`（3.12.13、2.5.3、1.18.1）；`apply` 要求等於 19 的 `frozen_config.json`。兩者都不 import torch。
4. **模型檔**：`freeze` 第一次執行時把 `data/prereg19/models/` 的 3 個檔逐位元組複製到 `results/prereg19/rank/`，SHA 核對第 1.2 節。另寫 `freeze_report.json`（接縫、OOF、擬合紀錄），和 `frozen_config.json`、3 個模型檔一起 commit。
5. **只跑一次**：`freeze_report.json` 或 `frozen_config.json` 已存在時拒絕；要重跑須 `--rerun --reason --revision N`（本登記 rev N ≥ 2，那一列的 commit 晚於舊紀錄的 commit）。`apply` 在 `finalize` 之前可以 `--rerun --reason`（不打分、確定性的計算），`finalize` 之後拒絕。
6. **apply 的「和 09 相同」**：可用盤和 P1 可用盤等於 `results/prereg09/rank/val_lp_p1e.npz` 的 `usable`、`p1_ok`；Q02、Q02H、Q08、Q09 四份機率都要逐位元組重現（Q09 用 `p09_rank_fuse.q09_core`）。方向檢查只算有 SET 可用盤、而且有解析成功的盤的題。
7. **`val_set.npz`**：每題（驗證題順序）的 `question_id`、`lp`（lp_SET，沒有可用盤的題是第 1.2 節定義的值、不會被使用）、`has`；每盤的 `usable`；模型 SHA、μ/σ 的來源（09 的 `p1e_map.npz` SHA）、08 驗證塊 SHA。測試日的本機接縫由它重建 E_SET（`has` 的題 4·lp，其他 0）。
8. **apply 讀 `data/local` 的地方**：只經由已登記的 `p07_rank_fuse.q02_validation`（題目結構）、`check_question_file`（題號、盤數、執色）和 `check_submission.py --questions data/local/rank_val_questions.csv`（格式），都在 `pilot_fuse`／`p1_lib` 被 import 之前；同 PREREG_09 2.7。不讀答案。
9. **稽核的綁定**：`apply_audit.json`（格式 `c10-prereg19 apply_audit v1`）要記 `rank_val_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`val_set.npz` 的輸出 SHA 和 `frozen_config.json`、3 個模型檔、08 `val_report.json` 的輸入 SHA，和 `apply_report.json` 相同，而且 commit 晚於 `apply_report.json`；`finalize` 才把 `usable` 設成 true。
10. **修正（`460689e`，仍在 `freeze` 之前）**：`768f7d2` 的釘住檢查用 git 輸出比對探索紀錄時把結尾換行去掉，`eval_report.json` 被誤判為改過（`freeze` 第一次執行在任何計算之前就拒絕，結束碼 2，沒有寫任何檔）；改成讀 commit 裡的原始位元組。方法和判定不變。
11. **修正（`6b09c59`，仍在 `freeze` 之前）**：自己的 `InnerPrep` 少了 `inner_ex`、`inner_rows`（`Inner` 和 Q09 的前向要用），`freeze` 在 `Inner` 建立時停下（沒有寫任何凍結檔；之前一次因為本機工作樹的換行和 prereg02 紀錄不同而在同一處拒絕，改用和 main 相同位元組的工作樹）。補上這兩個；接縫 P 改成核對 `inner_X.npy`、`inner_ex.npy`、`inner_meta.npz` 三個檔都等於 09 的 `check_report.json`。方法和判定不變。

## 10. rev 3：測試日套件的實作差異（在任何測試資料之前；分支 `p19-kit`）

**凍結的數字**（都來自已 commit 的檔）：`results/prereg19/rank/frozen_config.json`：w_OP 4.0、w_O4 3.594、w_SWA 3.667、w_SWB 0、w_HS 0、w_P1TO 3.842、**w_SET 3.314**，a −2.793、γ 0.379（25,400 題擬合假題，L-BFGS-B 兩個起點的 loss 0.895077／0.895077）；方向參照 ρ_eval 0.9543、ρ_mono 0.9586（測試日門檻 0.4793）。接縫 1、M、P、E、F 全部逐位元相同（Q19 OOF 0.7345843051712626＝探索）。驗證題無標籤檢查（`apply_report.json`）：Q02／Q02H／Q08／Q09 逐位元組重現，可用盤＝09，SET 方向 ρ 0.9593（門檻 0.4772），格式過；只記錄：1,589 題中 292 題規則 B 和 Q09 不同（等級差 −1：166、＋1：117、其他 9）。獨立稽核（另一個子代理，`apply_audit.json`）：自己的前向重算 lp_SET 最大差 1.5e-13、自己的融合重算 Q19 機率最大差 7.2e-16、θ 由同一個確定性擬合重現（差 0），沒有影響預測的程式錯誤；另記一個敏感度註記：擬合的最佳點很平，題目平均改用普通平均（只差捨入）θ 最多動 0.0074，驗證題機率最多動 6.8e-5、規則 B 一題都沒變；測試日用同一份平均程式（`tools/p19_forward.py`），所以逐位元相同。

1. **程式**：`tools/t19_rank.py`（`apply`、`status`；import `tools/t09_rank.py`，經由它 08／07／04，不改）、`tools/official_log.py` 只加行、`docs/TESTDAY_19.md`、`tests/test_t19_rank.py`、`tests/test_t19_official_log.py`、`tests/test_t19_dryrun.py`（`C10_SLOW=1`）、`tests/test_p19_hygiene.py`、`tests/test_t19_hygiene.py`（兩個加進 AGENTS.md 的閘門指令）、`.gitattributes`（`results/test19/** -text`）、`.gitignore`（`results/test19_rehearsal/`、`data/test19_rehearsal/`）各加行。
2. **tag `p14-testday-pre19`** 打在 main 的 `85cbe04`（19 的任何套件程式之前）。`tests/test_t19_hygiene.py` 要求 `official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_14_RANK_EXACT_COPY.md` 對它只加行，R20／04／06／07／08／09／10 的程式、測試、測試日手冊和 `results/prereg0*`–`prereg18`、`results/test0*`、`test10`、`test11` 自它起不變。09 的路徑另有 tag `p09-testday-pre10` 和 `tests/test_t09_path_hygiene.py`（照舊）。本套件沒有替 19 自己做路徑指紋（09 的前例是由下一個套件做，例如 `p19-testday-pre20`）。
3. **第 4.2 節的實作**：
   - P1 塊用 `t09_rank.Apply.p1_sets` 讀（同一套檢查），而且三組的報告和 `<組>_p1.npz` 的 SHA 要等於 09 的報告記的；
   - 第 1 步另要求 lp_P1TO_E 和 09 的 `test_lp_p1e.npz` 差 ≤ 1e-12（登記只寫「可用盤相同」）；
   - 本機接縫另要求 seam_val 的可用盤等於 `val_set.npz` 的 `usable`（不同是 4，結構問題）；09 的 `seam_local.json` 逐位元組相同時，SET 方向 ρ 也要等於 `results/prereg19/rank/apply_report.json` 的值（不等是 4）；
   - SET 的 GPU 接縫紀錄寫在 `results/test19/set_seam.json`（不判定，第 4.2 節第 2 步）；
   - 方向檢查只算「有解析成功的盤、而且有 SET 可用盤」的題；P1 失敗率不另判（09 結束碼 0 已判），報告抄 09 的數字；
   - 輸出另有 `test_lp_set.npz`（每題 lp_SET、`has`、每盤 `usable`、模型 SHA）。
4. **`official_log.py`（只加行）**：19 檔在 09／08／07／04 之前認出；最終選擇的紀錄鍵是 `prereg19_4_3`；09、08、07、04 的 prepare／final 規則跳過 19（各加「19 跳過」的行），status 的違反判定把「成立中的 19 最終選擇」當成合規，有符合的 19 時提示改成「最終選擇要設成 19」；public 的同檔警告 04–09 跳過 19，19 自己另有一個；`prepare`／`final` 加 `--not-19-reason`（rev 4 起、晚於那幾筆 19 的 prepare）；`status` 有 19 紀錄時印 K6-19。**沒有 19 紀錄時，所有輸出、紀錄檔、回傳值和 tag `p14-testday-pre19` 的版本逐位元組相同**（`tests/test_t19_official_log.py` TagVersusNew：R20／R10／04／06、07、08、09 的流程，和一條 14 的流程）。
5. **14 跟著認得 19（第 4.1 節）**：`tools/t14_copy.py` 只加一行（`BASE_DIRS` 加 `results/test19/rank`），`official_log.py` 的 `T14_BASE_DIRS` 同樣加一行；14 的底檔規則加上本登記 4.3（final 時記下的 `--not-19-reason` 會沿用）。因為 14 的設計紀錄綁「現在的工具」，`results/test14/design/index_pins.json`、`sim.json` 用新工具 `--rerun --reason` 重算：釘住值、模擬的題目、覆寫和三個通過條件**全部相同**，只有 provenance（工具 SHA、時間）不同；`tests/test_t14_hygiene.py` 的 `PINS_SHA_LF` 改成新檔的 SHA（唯一改到的 14 測試行）。PREREG_14 文末加一行說明（不是修訂列）。測試：`tests/test_t19_official_log.py` 的 `T19With14`（19 是最終時 14 以 19 為底檔，04–19 都沒有違反）。
6. **演練**（`C10_SLOW=1`，假樹＝驗證題前 800 題；樹放在 scratchpad 的 subst 磁碟，因為路徑含 `AppData\Local` 會被 `data\local` 防護誤擋）：R20／04／07／08 的輸出等於 `baseline_pre09.json`、09 的輸出等於 `t09_fake_golden.json`；Q09 重算和 09 逐位元組相同；SET 的 seam 差 0；本機接縫和 `results/prereg19/rank/` 逐位元組相同；800 題的機率和 prereg19 驗證題前 800 題差 ≤ 1e-12、規則 A／B 等級全同；lp_SET 和 `val_set.npz` 差 ≤ 1e-12；check_submission 過；10 個負面演練（09 不用／沒做完／輸出改過、Q09 差一個值、K6-19、FAIL_OPENING、輸出已存在、BLAS、raw、正式模式的演練旗標）都得到登記的結束碼。19 自己的基準 `results/test19/design/t19_fake_golden.json` 是程式寫好之後第一次通過的演練記下的（同 09 的做法）。
7. **容許差**：沒有新的容許差。09 的 4090 演練樹已經刪掉，沒辦法為 SET 量 GPU 的容許差，所以 SET 的 GPU 接縫只記錄（第 4.2 節）。
8. **閘門**：AGENTS.md 的閘門指令加 `tests/test_p19_hygiene.py`、`tests/test_t19_hygiene.py`。`tests/test_g11_hygiene.py::InsertOnly::test_official_log_untouched` 要求 `official_log.py` 和 main 相同，所以在分支上是紅的，合併後是綠的（同 PREREG_14 13.2 第 16 條）。
9. **工作樹的換行**：凍結時發現從 git 新取出的工作樹（`core.autocrlf=true`）有些檔和 main 工作樹的位元組不同（例如 `results/split_v1/players_split.csv`、`results/prereg02/rank/pseudo_inner.json`），prereg02 的 SHA 紀錄是 main 工作樹的那一份；分支的工作樹改用和 main 相同的位元組。測試日在 main 上跑，不受影響。
10. **獨立審查**（另一個子代理，讀程式、跑 19 的單元測試、`C10_SLOW=1` 演練和閘門；沒有碰 `data/local`）：沒有 blocker、沒有 major；4 個 minor 都在合併前處理：
    - ① 重跑時若前提判了 3 或 4，`not_used.json`／`attempt_verdict.json` 可能蓋掉或擺在已 commit 的決定旁邊 → 輸出已存在而沒有 `--rerun --reason` 時，在前提之前就拒絕（2）；已有 `t19_apply_report.json` 或 `not_used.json` 而這一次還沒通過重跑閘門時，判定只印出、不寫任何檔（2）。09 的工具有同樣的寫法，但它已凍結，不改，只在這裡記下。
    - ② 採用紀錄的 `usable`／`finalized` 改成在 prereg19 釘住檔的 SHA 核對**之後**才讀。
    - ③ 結束碼和第 4.2 節的寫法不同：prereg19 的釘住檔被改過是 **2**（拒絕，同 09），不是 3；`apply_report.json` 不是 usable／finalized（SHA 相同時不會發生）才是 3。`docs/TESTDAY_19.md` 照 2 寫。
    - ④ 演練補三個：本機接縫走容許差路徑判「不用」（3）、SET 方向沒過（3）、在已記錄的 `not_used.json` 上用 `--bug-revision` 重跑（沒有修訂時拒絕、前提失敗時不覆寫、有修訂時跑完並記下被取代的檔）；後三個用子程序把一個模組值換掉來到達（假樹改檔做不到）。修完重跑：19 的單元、hygiene、`C10_SLOW=1` 演練 21 項全過，19 的輸出仍等於 `t19_fake_golden.json`。

## 附註（2026-10-20，主 session；只加行，不是修訂，不推翻任何登記結果）

- `tests/test_t19_hygiene.py` 的 FROZEN_PATHS 拿掉 `docs/TESTDAY_USER_CHECKLIST.md`：那是給使用者看的一頁清單，明寫以各份 TESTDAY 手冊為準，上傳順序改變時（例如加上 19）要跟著更新，不屬於測試日的程式或手冊。其他凍結路徑不變。
