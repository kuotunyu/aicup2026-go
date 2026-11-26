# 事前登記 25：競賽二把 Q19 的 set 模型換成「多了驗證玩家」重訓的 S25（Q25）

**白話版**：19 的 set 模型 S19 只用 fit 玩家（5,767 位）訓練。prereg25 探索把 split v1 的 1,600 位驗證玩家的棋（照 probe 的抽法抽 74,946 盤，P1 特徵在 4090 上算）加進去，架構、設定、種子全部照 S19 重訓成 **S25**，在 19 的融合裡取代 S19（其他 6 項不動），inner 從 0.7346 到 **0.7457（+0.0111，95% 區間 [+0.0050, +0.0183]）**，單色 +0.0052 [+0.0014, +0.0091]，三個閘門都過；獨立稽核（`1965276`）由已存的模型逐位元重現、沒有洩漏。本登記把那一個候選**原樣**凍結（不重訓、不再挑），用全部 inner 擬合一次融合參數，測試日在 19 已算好的東西上只多跑幾分鐘 CPU。代價：**驗證玩家進了訓練，驗證集從此不能當 25 的獨立檢查**（競賽二的 4 次驗證集打分本來就用完了），報告要寫明。上傳在 19 之後（目標 11/18 前）；任何一步不成立就維持 19。

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。對照組：**PREREG_19 的 Q19**（`results/prereg19/rank/frozen_config.json`；探索 inner 0.7345843051712626、單色 0.7430172362615272）。
- 探索（只用 inner 評估、沒打分、沒碰 `data/local`）：`results/prereg25/explore/`。
  - `PLAN.md`（`6e4b022`，任何模型數字之前；第 9 節實作補記 `a839f3d` 也在訓練之前）、`audit.json`（資料盤點，同一個 commit）；程式 `p25_lib.py`、`p25_data.py`、`p25_extract.py`、`p25_prep.py`、`p25_train.py`、`p25_eval.py`、`test_p25.py`；
  - `seam_g.json`（`daa0c3c`）、`train_record.json`（`5769330`，只訓練一次）、`eval_report.json`（`4aa42f5`，只判一次，閘門全過）、`RESULTS.md`（`392eafe`）、`independent_audit.json`（`1965276`，獨立稽核：verdict real）。
- 決定者：主 session（使用者 10/2、10/9、10/16 把對最終排名有幫助的方法決定交給 AI）。
- **只做競賽二（TEAM_11248）**；25 的任何輸出都不進競賽一，不讀競賽一的任何檔案。
- 代號：
  - **S25**：S19 的架構和訓練（`results/prereg19/explore/p19_lib.py`、`p19_train.py`，import 不改），訓練列＝probe 270,000 列＋驗證玩家 74,946 列，A／H 用新鹽 `c10-prereg25|holdout|`；3 個種子 0、1、2。
  - **lp_SET25**：一題的 10 維對數機率＝3 個 S25 種子機率平均再取 log（下限 1e-300）。**E_SET**＝4·lp_SET25（該題沒有可用盤時是 0）。
  - **Q25**＝Q19 的 7 項（OP、O4、SWA、SWB、HS、P1TO、SET），SET 項的模型換成 S25，融合參數重新擬合。
- 本登記寫四件事：做什麼（第 1 節）、怎麼凍結和檢查（第 2 節）、測試日怎麼用、什麼時候放棄（第 4、5 節）。偏離寫在第 3 節。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `535a5ae` | 第一版，2026-11-06。commit 時的狀態：① 探索已完成並 commit（上列），**閘門已判過一次**（G1 +0.0111、G2 下界 +0.0050、G4 單色 +0.0052、黑 +0.0010、白 +0.0092），獨立稽核 `1965276` 判 real；這些數字是本登記的依據，不是本登記之後的結果。② 3 個 S25 模型檔在 `data/prereg25/models/`（SHA 在 `train_record.json`），本登記**原樣凍結、不重訓**。③ 還沒有：任何凍結的融合參數、任何 25 的工具程式（`tools/p25_*.py`、`tools/t25_*.py`）、`results/prereg25/rank/`、`results/test25/`、任何測試題的 lp_SET25 或 Q25 機率。④ 打分紀錄帳自 H8R 之後沒有競賽二的新列；本登記也不加任何列。⑤ main 已打 tag `p19-testday-pre25`（`163d413`，25 的任何程式之前）。 |
| rev 2 | `ff41c37`（工具） | 2026-11-06，在 `freeze` 之前：凍結和採用工具的實作差異（第 9 節）。commit 時：`results/prereg25/rank/` 不存在；還沒有任何凍結參數或測試題的 lp_SET25。工具只有單元測試（合成資料和登記常數）。 |
| rev 3 | `7dbb3b4` | 2026-11-06，測試日套件的實作差異（第 10 節），在任何正式測試日執行之前、合併之前。commit 時：凍結和採用已完成（`af79fc3` freeze、`d583848` 獨立稽核、`639d9ec` finalize，usable true）；`results/test25/` 不存在；套件只在 repo 外跑過 `--dry-run`（讀已 commit 的 test02–test19 輸出，唯讀）。之後要推翻 25 的登記結果，要 rev 4 起的修訂。 |

之後的修訂只准加行（不改本表已有的列和第 0–8 節的原文）。預定：rev 2＝凍結工具的實作差異（在 `freeze` 之前 commit）；rev 3＝測試日套件的實作差異（合併前 commit）。要推翻 25 的登記結果（`--bug-revision`、`--not-25-reason`），要 **rev 4 起**、而且晚於被推翻的紀錄。

## 0. 為什麼做這個（都不看答案；inner 已被多次使用，數字偏樂觀）

- **prereg24 的線索**：S19 多一倍玩家約 +0.0101，每人多一些盤只 +0.0034；fit 玩家已經全部在 probe 裡，唯一還沒用過的玩家是 1,600 位驗證玩家。競賽二的 4 次驗證集打分已全部用完（H8R 是最後一次），驗證集對競賽二已沒有別的用途。
- **探索的結果**（`eval_report.json`，判一次）：
  - G1：inner 評估假題（127,200 題）0.7345843 → **0.7456520（+0.0111）**；G2：配對 PlayerBootstrap 95% [+0.0050, +0.0183]；G4：單色 0.7430 → 0.7482（+0.0052 [+0.0014, +0.0091]），黑 +0.0010、白 +0.0092。
  - 依盤數（只記錄）：單色 5 盤 +0.0031 [−0.0007, +0.0072]、10 盤 +0.0048、15 盤 +0.0062、20 盤 +0.0082；評估假題 5 盤 −0.0030（區間跨 0）。
  - 依等級（只記錄）：D 級兩種題都退約 0.02；3D–6D 進步最多。
  - 只記錄：S25 單獨（規則 B）inner 0.7439（S19 單獨 0.7286）；Q19＋S25（加成第 8 項）+0.0108，和「取代」差不多；每個種子單獨 +0.0071／+0.0058／+0.0066，三個的集成最好。
  - 每折融合權重整體變小（Q25 的 w_SET 0.42～0.93、w_P1TO 0.25～0.49；Q19 是 0.62～3.82／0.77～4.0），沒有打到界限。
- **獨立稽核**（另一個子代理，`independent_audit.json`）：eval 由已存的模型和同一套函式逐位元重現（最大差 0）；自寫 numpy 前向在 2,000 題上 lp_SET 最大差 2.1e-13；inner 玩家不在 S25 的訓練組或留出組（以 `<rank>:<player_id>` 和只比 player_id 都是 0）；V 的玩家正好是 split v1 的 1,600 位驗證玩家，74,946 盤逐盤在官方訓練 CSV 找到；V 棋對 inner 棋 8 種對稱重複 0；沒有影響結果的程式錯誤。4 個 minor 都是揭露事項（權重尺度、留出 NLL 不可比、inner 重複使用、稽核腳本的路徑）。
- **誠實的增益估計**：inner 已被 P1、08、09、prereg12、15–24 重複使用，選擇偏差約 0.003；這次下界 +0.0050 在偏差之外，但接近。測試題的結構（盤數、執色）和 inner 假題不同，真實增益可能比 +0.011 小；5 盤題沒有增益，D 級退步。
- **為什麼還做**：
  1. 只換 SET 項的 3 個模型檔，其他 6 項、特徵、校準、規則 B、測試日的 P1 計算都和 19 相同；inner 玩家不在 S25 的任何列（探索接縫 D、獨立稽核 C）。
  2. 測試日不用 GPU：SET 讀的就是 08 已算好、09 和 19 也讀的那份 P1 塊；19 的測試日輸出已 commit，25 可以先逐值重現 19 再換模型，路徑風險很小。
  3. 最壞情況：25 的測試日任何一步不成立，就維持 19（已上傳的最後一份）。

## 1. 來源（全部釘死）

### 1.1 沿用（不重算、不改）

- 19 的全部凍結檔和測試日輸出：`results/prereg19/rank/*`、`tools/p19_forward.py`、`tools/p19_set.py`、`tools/t19_rank.py`；`results/test19/**`（11/05 的正式執行，已上傳 #9）。
- 09、08、07、04、R20 的凍結檔和測試日路徑、輸出（各自的指紋和 tag）；KataGo 程式和權重；P1 的特徵化。
- inner 的融合和評估一律走 `tools/p09_rank_fuse.py` 的 `Inner`（import，不改）和 `tools/p04_rank_fuse.py` 的 `gen_fit_record`、`gen_params`；inner 的 P1 特徵 `data/prereg12/q09_inner/`，和探索、19 的凍結相同。
- 探索的程式只當規格（釘 LF SHA，不 import）：`results/prereg25/explore/{PLAN.md, p25_lib.py, p25_train.py, p25_eval.py}`，以 `1965276` 的版本為準。

### 1.2 S25 模型（照探索原樣，不重訓、不再調）

- 架構和 S19 完全相同（PREREG_19 第 1.2 節）：每盤 h(x)：Linear(1540, 256) → GELU（erf）→ Linear(256, 64) → GELU；題目內取可用盤 h 的平均；head：Linear(64, 32) → GELU → Linear(32, 1)；累積 link 同 09 的 mlpclm。
- 輸入：每盤 1,540 維 P1 TO 特徵，以 **09 凍結的 `p1e_map.npz` 的 μ_X、σ_X** 標準化（同 S19）。可用盤的定義同 19（沒失敗、至少 1 手目標手、棋譜解析成功）。
- 3 個模型檔＝探索訓練的那 3 個檔，**位元組不變**複製到 `results/prereg25/rank/s25_seed{0,1,2}.npz`（進 git），格式 `c10-prereg25 s25 model v1`，SHA-256 必須等於 `train_record.json` 的 `models_sha256`：
  - seed 0：`4a01d6a60355cd514e9cc4988c08424a7952fc33302b47a63542a8caf67163bb`（第 11,000 步）
  - seed 1：`ab53d69b2762a7c01bed354ac6c218cf167fda44bbd3d88eed06a6f4adf3cba4`（第 7,500 步）
  - seed 2：`20e190f56866c7a28b4eeef0c0d4d656867b79a908e066ba91a47570601c4fbb`（第 10,500 步）
- **前向（寫死）**：numpy float64，和 19 同一段程式。新模組 `tools/p25_forward.py` 只負責讀 S25 的檔（登記的 SHA、格式、種子、步數），計算一律呼叫 `tools/p19_forward.py` 的函式（`game_h`、`set_term`、`question_set_term`、`single_game_logp`；import，不改）。不 import torch。
- 乘 4：E_SET＝4·lp_SET25，同 19。

### 1.3 融合（唯一一組參數）

- 公式同 PREREG_19 第 1.3 節：p＝softmax(e^a (n/12)^γ (m_C5 ＋ w_OP m_OP ＋ w_O4 E_O4 ＋ w_SWA E_SWA ＋ w_SWB E_SWB ＋ w_HS E_HS ＋ w_P1 E_P1E ＋ w_SET E_SET ＋ Bβ))，規則 B（`tools/decide.py`）；E_SET 用 S25。
- 界限、起點、PENALTY、MAXITER、等級加權都不變（`p04_rank_fuse.gen_fit_record`，確定性的計算）。
- **凍結的測試日參數只擬合一次**：用**全部** inner 擬合假題（25,400 題，同 09／19 凍結時用的那一批）重擬合 7 項的 θ（`p4.gen_fit_record(mf, Ef, n, y)`），同 PREREG_19 1.3。不看 OOF、不看任何驗證題，再也不改。權重打到界限只記錄、不重挑（探索的每折權重都沒打到）。

## 2. 凍結、獨立稽核、採用（CPU；`tools/p25_set.py`）

### 2.1 `freeze`（只跑一次）

全部成立才寫凍結檔；任一接縫不過就停（不寫凍結檔、不算判過，查明是程式錯誤後在 rev 2 起的修訂寫明再跑）：

1. **接縫 1**：Q02H、Q08、Q08 單色等於 `p09_rank_fuse.REF`；Q09 inner 0.7114420614163589、Q09 單色 0.7145431684624712；差都 ≤ 1e-12。
2. **接縫 M**：3 個 S25 模型檔（`results/prereg25/rank/` 的副本）的 SHA 等於第 1.2 節，而且等於 `data/prereg25/models/` 的原檔；探索的 `PLAN.md`、`p25_lib.py`、`p25_train.py`、`p25_eval.py`、`train_record.json`、`eval_report.json`、`audit.json`、`independent_audit.json` 的 LF SHA 等於 `1965276` 的版本；19 的 3 個凍結模型檔 SHA 等於 PREREG_19 1.2。
3. **接縫 P**：`inner_X.npy`、`inner_ex.npy`、`inner_meta.npz` 等於 09 的 `check_report.json`（同 19）。
4. **接縫 E**：自己的題目平均用在 4·lp_P1TO_E 上等於 `Inner.E_p1`（≤ 1e-12，同 19）。
5. **接縫 F19**：同一個工具、凍結的 S19 算出的 Q19 OOF 等於 0.7345843051712626、單色 0.7430172362615272（≤ 1e-12）。證明工具和 19 的凍結、探索的基準是同一個東西。
6. **接縫 F（重現探索的判定）**：S25 算出的 Q25 OOF，inner 分等級平均等於 `eval_report.json` 的 0.7456519871546352、單色 0.7482450052195216，5 折的 7 個融合權重全部等於 `eval_report.json`（差都 ≤ 1e-12）。

過了之後：第 1.3 節擬合 θ，寫 `results/prereg25/rank/frozen_config.json`（組合、項、倍數、參數、界限有沒有打到、擬合紀錄、樣本內分數（只記錄）、5 折 OOF 紀錄（＝探索的判定）、模型檔 SHA 和步數、方向檢查的參照值 ρ_eval、ρ_mono（只用 E_SET 的期望等級對 Q02 OOF 的期望等級，定義同 PREREG_19 2.1）、釘住檔的 SHA、工具 SHA、`.venv` 的 python／numpy／scipy 版本（必須等於 19 的 `frozen_config.json`））和 `freeze_report.json`。凍結檔、3 個模型檔、`freeze_report.json` 一起 commit。`freeze` 在 BLAS 執行緒變數有設時拒絕；不 import torch。

### 2.2 驗證題的無標籤檢查：**不做**（決定和理由）

PREREG_19 2.2 在驗證題上無標籤地套用 Q19（重現 Q02／Q02H／Q08／Q09 的位元組、SET 方向檢查、格式檢查），產生測試日本機接縫的參照檔。25 **不做這一步**，理由：

1. **不再獨立**：S25 用驗證玩家的棋訓練。驗證題是從同一批玩家的棋做出來的，S25 在驗證題上的方向檢查（ρ）、和 19 的差異題數都會被訓練資料抬高，沒有檢查價值；做了反而可能被誤讀成「驗證集上的證據」。
2. **它的管線檢查角色由更強的接縫取代**：
   - 凍結時的接縫 F19、F：同一個工具在 inner 上逐值重現 19 的凍結結果和探索的判定。
   - 測試日的接縫（第 4.2 節第 2、3 步）：在**真正的 public／private 題**上重算 Q09 和 Q19，要和已 commit 的 `results/test09/`、`results/test19/` 輸出一致（機率差 ≤ 1e-12、規則 A／B 等級全同）。這直接證明 25 用的 6 項、題目結構、可用盤、融合程式和 19 上傳的那一份完全相同，差別只在 SET 的 3 個模型和 θ。這比 19 的「seam_val 本機接縫」更直接。
3. **不碰 `data/local`**：19 那一步經由登記工具讀 `data/local` 的驗證題目檔（題號、盤數、執色、SGF）和用它做格式檢查；主 session 這一輪的規則是任何程式都不開、不讀、不列、不雜湊 `data/local` 底下的任何東西（11/06 又發生一次列目錄事件）。
- 取代的條文（不改原文）：PREREG_19 2.2／2.3 的驗證題 apply、`apply_audit.json` 和 `apply --finalize` 對 25 改成第 2.3、2.4 節。

### 2.3 獨立稽核（凍結之後、採用之前）

- 另一個代理（看不到答案、不看任何分數、不碰 `data/local`；做不到就由建置代理做一輪對抗式自審，並寫明是哪一種）：
  - 3 個模型檔 SHA 和原檔、`train_record.json` 相同；
  - 用自己寫的 numpy 前向（不 import `p19_forward`／`p25_forward`）在 inner 擬合假題上重算 lp_SET25（抽樣 ≥ 2,000 題），和凍結工具的 E_SET 相同（≤ 1e-9）；
  - 用同一個確定性擬合獨立重跑 `gen_fit_record`，θ 和 `frozen_config.json` 相同（≤ 1e-9）；乘 4 只做一次、項的順序、可用遮罩；
  - `freeze_report.json` 的接縫 1、F19、F 數字和 `eval_report.json`、19 的紀錄相同；方向參照值 ρ 可由凍結檔重算；
  - 沒有讀答案、沒有打分、沒有 inner 以外的玩家進入擬合、沒有任何 `data/local` 存取。
- 結果寫 `results/prereg25/rank/freeze_audit.json`（格式 `c10-prereg25 freeze_audit v1`，欄位 `program_errors_affecting_predictions`、`inputs_sha256`：`frozen_config.json`、`freeze_report.json`、3 個模型檔），commit 晚於凍結檔。

### 2.4 採用（`p25_set.py finalize`）

- 前提：凍結檔、模型檔、`freeze_report.json`、`freeze_audit.json` 都已 commit；稽核綁定的 SHA 等於現在的檔；稽核的 commit 晚於凍結檔。
- 寫 `results/prereg25/rank/adoption.json`（格式 `c10-prereg25 adoption v1`）：`combo` Q25、`usable`（稽核沒有找到影響預測的程式錯誤才是 true）、`finalized` 時間、稽核的 SHA 和 commit。commit 之後是 25 的採用紀錄（相當於 19 的 `apply_report.json` usable／finalized），之後一律不改。

### 2.5 不做的

- 不重訓 S25、不改種子、步數、架構、倍數、項的組合；不試「加成第 8 項」（探索只記錄，和取代差不多）；不加 fit 玩家沒進 probe 的盤。
- 不用 inner OOF 以外的任何東西挑參數；凍結的 θ 只擬合一次。

## 3. 偏離（寫明）

- **沒有驗證集打分**（同 PREREG_19 第 3 節）：Q25 只靠 inner 閘門（探索判一次）、獨立稽核和第 2.1 節的接縫上場。PREREG_08 5.4「不交沒在驗證集打過分的組合」對 Q25 例外（照 PREREG_19 3 的前例）。
- **用了驗證玩家訓練**：驗證集從此對任何含 S25 的組合都不是獨立檢查；競賽一沒有用到 S25（競賽一剩 1 次的驗證集打分不受影響）。
- **沒有驗證題的無標籤 apply**（第 2.2 節）。
- 風險（揭露）：inner 重複使用（選擇偏差約 0.003）；D 級退約 0.02；5 盤題沒有增益；黑題的單色增益只有 +0.0010；權重尺度和 Q19 差很多。最壞情況是 25 在測試題上不如 19。
- 決定者：主 session（使用者授權方法決定）。

## 4. 測試日（全部 CPU，不跑 GPU，不碰 `data/local`）

### 4.1 保護舊路徑

- 一律不改：R20、04、06、07、08、09、10、14、19、21 的測試日路徑和輸出；`results/prereg01`–`prereg24/**`、`results/test02|04|06|07|08|09|10|11|14|19|21/**`；`tools/p0*`、`p1*`、`t0*`、`t10_*`、`t14_*`、`t19_*`、`t21_*`；`third_party/katago/**`。
- `results/prereg25/explore/**` 只准加檔；`results/prereg25/rank/` 的凍結檔在 `finalize` 之後一律不改。
- 只准加行：`tools/official_log.py`、`.gitattributes`、`.gitignore`、本登記。
- tag **`p19-testday-pre25`**（main `163d413`，25 的任何程式之前）：25 的測試要求 `official_log.py` 對這個 tag 只加行，而且沒有 25 紀錄時輸出、紀錄檔、回傳值都和 tag 的版本逐位元組相同。
- **14 不跟著改**：14 在 11/04 的掃描是 0 題、已記錄不用（`results/test14/not_used.json`），所以 25 不會成為 14 的底檔；`tools/t14_copy.py` 不動。

### 4.2 `tools/t25_rank.py apply`（19 已正式執行並 commit 之後；幾分鐘）

- **不讀 `data/local`**：t25 不跑 09／08／07／04／test02 的前提鏈（那條鏈會雜湊 seam_val 的題目檔，它在 `data/local`），也不跑 seam_val。改成：綁定已 commit 的 19 正式執行，並在 public／private 上重算 Q09、Q19 來證明輸入和程式沒有變。
- **前提**（照順序，第一個不成立的決定結束碼）：
  1. git 乾淨（正式模式）；不設 BLAS 執行緒變數；
  2. prereg25 的凍結檔 SHA 不變（被改過是 2）；`adoption.json` 是 `usable: true` 而且 finalized（不成立是 3：25 沒採用）；
  3. `.venv` 的 python、numpy、scipy 等於 `frozen_config.json`（不等是 4）；
  4. K6-25；
  5. **19 的正式執行**：`results/test19/rank/t19_apply_report.json` 是正式模式、結束碼 0、usable、已 commit；它 `inputs_bound` 的每一個檔都已 commit 而且 SHA 沒變；它的 6 個輸出 SHA 沒變；T1 是 PASS 或 NORMALIZED、沒有用 R10 備援（從它綁定的 `t1.json`、test02 報告重讀）。19 沒有成功的正式執行是 2（25 等 19）。
- **輸入**：08 的 P1 塊（`data/test08/p1/<組>/`，逐塊 SHA 等於 08 的報告，三組的報告和 `<組>_p1.npz` 的 SHA 等於 19 的報告記的）；test02／test04／test07 的 public、private 檔（`data/test` 的題目檔、test02 的 dumps、test04 的 sw／opp、test07 的 hs，都是 19 綁定過的檔）。
- **步驟**：
  1. **P1 塊**：同 19（`t09_rank` 的讀法和檢查）。
  2. **Q09 對得上 09**：用 09 的程式重算 public、private 的 Q09，和 `results/test09/rank/rank_test_probs.csv` 比：題號、順序、盤數相同，機率差 ≤ 1e-12，規則 A、B 的等級全同；lp_P1TO_E、可用盤等於 09 的 `test_lp_p1e.npz`。不成立是 4。
  3. **Q19 對得上 19**：用凍結的 S19 和 19 的 θ 重算 Q19，和 `results/test19/rank/rank_test_probs.csv` 比（同上的條件），lp_SET19 和 `test_lp_set.npz` 差 ≤ 1e-12、`has`／`usable` 相同。不成立是 4。位元組是否相同只記錄。
  4. **SET25 的 GPU 接縫（只記錄）**：同 PREREG_19 4.2 第 2 步，把模型換成 S25，記最大差，不判定。
  5. **方向檢查**：每組（public、private）只用 E_SET（S25）的期望等級和 Q02 的期望等級的 Spearman ρ ≥ max(0.5 × ρ_mono25, 0.155)（ρ_mono25 取 25 的 `frozen_config.json`）；不成立是 3。
  6. **輸出**：用凍結的 S25 和 θ25 算 Q25，`results/test25/rank/` 的 `rank_test_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`submission_rank.csv`（＝`pred_B.csv`）、`test_lp_set.npz`、`t25_apply_report.json`；`check_submission --official` 失敗是 4（輸出刪掉）。報告記錄規則 B 和 19、09 不同的題數（public、private 分開）。
- **結束碼**同 t19：0 完成；2 拒絕；3 登記的「25 不用」（寫 `results/test25/not_used.json`，不重跑；最終照 PREREG_19 4.3）；4 本機漂移或對不上（寫 `rank/attempt_verdict.json`，停下查，查好 `--rerun --reason`）。出現 3 或寫好 `t25_apply_report.json` 之後，只有 `--bug-revision N`（N ≥ 4、那一列的 commit 晚於被推翻的紀錄）才能重跑。K6-25 過後已有成功的那一次：2，那一次仍有效。
- 輸出只寫 `results/test25/`；演練只寫 repo 外或 `results/test25_rehearsal/`。不讀競賽一；不讀驗證答案；不跑 `score.py`；不 import torch；不開、不讀、不列、不雜湊 `data/local`。

### 4.3 上傳順序和最後一版（競賽二）

- **順序**：R20 → 04 → 07 → 08 → 09 → 19（都已在 11/04–11/05 上傳）→ **25**（目標 11/18 前）。25 在它綁定的 19 已上傳之後才 `prepare`。
- **最後一版**：有符合的 25 → **25**；沒有 → 照 PREREG_19 4.3（19，否則 PREREG_09 5.3：09 → 08 → 07 → 04 → R20，T1 FAIL_OPENING 時 R10）。
- 「符合」＝`official_log.py` 每次用 25 報告裡的數字重算第 4.2 節的條件，加上：25 的採用紀錄（`adoption.json` 釘 LF SHA、usable、finalized）、25 報告綁定的那一筆 19 已上傳而且仍符合 PREREG_19 4.3、輸入綁定沒變、K6-25。
- 有符合的 25 而要交別的，要 `--not-25-reason "PREREG_25 rev N: …"`（N ≥ 4，晚於那幾筆 25 的 prepare）。
- public 分數不進判定；25 比 19 低 0.05 以上時只查錯，查不到就維持本節的結果。
- 取代的條文（不改原文）：PREREG_19 4.3「有符合的 19 時最終是 19」→ 有符合的 25 時最終是 25。
- `official_log.py` 的規則（只加行）：25 檔在 19／09／08／07／04 之前認出；最終選擇的紀錄鍵 `prereg25_4_3`；19、09、08、07、04 的 prepare／final／status／public 規則跳過 25；`prepare`／`final` 加 `--not-25-reason`；沒有 25 紀錄時，一切和 tag `p19-testday-pre25` 的版本逐位元組相同。

## 5. 截止點（台灣時間；沒趕上就照表做）

| 截止點 | 條件 | 沒趕上 |
|---|---|---|
| K1-25（無日期） | 本登記 rev 1 的 commit 早於任何 25 的工具程式、凍結參數和測試題的 lp_SET25 | 這一季不做 25 |
| K4-25 11/10（二）23:59 | `freeze`、獨立稽核、`finalize` 都 commit | 這一季不用 25 |
| **K5-25 11/14（六）23:59** | 測試日工具、`official_log.py` 的加行、測試、演練（`C10_SLOW=1`）通過、rev 3 已 commit，以 `--no-ff` **合併回 main** | 這一季不用 25 |
| **K6-25 11/18（三）23:59:59** | 25 在官方測試集有一次成功的正式執行並 commit | 最終照 PREREG_19 4.3 |
| （上傳） | 使用者在 11/18 前上傳 25（AIdea 每天 5 次、以最後一次計分；11/24 23:59 截止） | 沒上傳就維持 19（最後一次上傳） |

## 6. 另一場

- 競賽一不用 S25、SET 或任何 25 的輸出；不讀競賽一的測試資料。

## 7. 防誤改閘門

- AGENTS.md 的閘門指令加 `tests/test_p25_hygiene.py`（prereg25 凍結檔、打分紀錄帳沒有 `prereg25` 列、工具的 import 限制、本登記只准加行）和 `tests/test_t25_hygiene.py`（測試日工具的登記值、`official_log.py` 對 tag `p19-testday-pre25` 只加行、舊路徑和 test19 不變、t25 程式不含 `data/local` 存取）。
- 允許的失敗不變（AGENTS.md 那 6 項）；新增的任何失敗都要在 rev 2／rev 3 寫明理由並經允許。

## 8. 揭露（寫進 RESULTS 和競賽二報告）

- **事前與事後的界線**：S25 的資料（只加驗證玩家、照 probe 抽法）、架構、步數規則、種子、「取代 SET 項」的候選都在 `PLAN.md`（任何模型數字之前）寫死；閘門只判一次；本登記在判定和稽核**之後**寫，只是把已判過的那一個候選凍結起來，沒有再挑任何東西（第 2.5 節）。
- **inner 的重複使用**、**沒有驗證集打分**、**驗證玩家進了訓練**、**沒有驗證題 apply**（第 2.2、3 節）。
- **外部資源**：同 PREREG_08／09／19（KataGo 程式和 HumanSL 權重；驗證玩家的 P1 特徵也是 KataGo 在本機 4090 上算的，約 3.8 小時）；S25 是本隊自己的模型（PyTorch CPU 訓練、numpy 推論）。
- **消融**（報告用，都來自已 commit 的檔）：inner 上 Q19 → Q25（+0.0111）；依盤數、依等級；S25 單獨；加成 vs 取代；每個種子單獨。
- 生成式 AI 協助的部分照規則揭露；探索、凍結、測試日程式由 Claude 寫，獨立稽核和審查由另外的代理做。報告草稿 `report/` 競賽二的方法和消融在 25 採用後依結果改寫（由主 session 做）。

## 9. rev 2：凍結和採用工具的實作差異（在 `freeze` 之前）

工具 `ff41c37`：`tools/p25_forward.py`（只有 S25 的讀檔：登記的 SHA、格式、種子、步數；計算的函式就是 `tools/p19_forward.py` 的同一個函式物件）、`tools/p25_set.py`（`freeze`、`finalize`）、`tests/test_p25_set.py`。第 1、2 節的方法、模型、融合和接縫都沒改；下面是第 2 節沒寫清楚或做法不同的地方。

1. **重用 19 的凍結工具**：inner 的特徵讀取（`InnerPrep`）、接縫 P、接縫 E、題目平均和 SET 項（`inner_set_terms`）、19 的釘住檢查（`check_pins`：19 的探索紀錄和 prereg09 的檔）都直接 import `tools/p19_set.py`，不複製、不改。
2. **接縫 F19 比登記嚴**：除了 Q19 的 inner 和單色分等級平均，5 折的 7 個融合權重也要等於 19 的 `freeze_report.json`（差 ≤ 1e-12）。
3. **接縫 F**：另要求 `eval_report.json` 的閘門判定是 passed。
4. **版本**：`freeze` 要求 `.venv` 的 python、numpy、scipy 等於 19 的 `frozen_config.json`（3.12.13、2.5.3、1.18.1）。
5. **只跑一次**：`freeze_report.json` 或 `frozen_config.json` 已存在時拒絕；要重跑須 `--rerun --reason --revision N`（本登記 rev N ≥ 2，那一列的 commit 晚於舊紀錄）。
6. **採用**：命令是 `p25_set.py finalize`；`freeze_audit.json` 的 `inputs_sha256` 要綁 `frozen_config.json`、`freeze_report.json`、3 個模型檔（`freeze_report.json` 是登記沒列、這裡加的），稽核的 commit 要晚於 `frozen_config.json`；`adoption.json` 只寫一次。
7. 兩個工具都不 import torch，原始碼裡沒有驗證題目檔、答案檔的名字，也沒有 `data/local` 的路徑（單元測試檢查）。

## 10. rev 3：測試日套件的實作差異（在正式測試日執行之前；分支 `p25-kit`）

**凍結的數字**（都來自已 commit 的檔）：`results/prereg25/rank/frozen_config.json`：w_OP 0.5849、w_O4 0.5062、w_SWA 0、w_SWB 0、w_HS 0、w_P1TO 0.3478、**w_SET 0.5784**，a −0.9284、γ 0.4005（25,400 題擬合假題，L-BFGS-B 兩個起點的 loss 0.876186098／0.876186094；SWA、SWB、HS 打到下界，只記錄，登記允許）；方向參照 ρ_eval 0.9538、ρ_mono 0.9576（測試日門檻 0.4788）。接縫 1、M、P、E、F19、F 全部逐位元相同（Q19 OOF 0.7345843051712626、Q25 OOF 0.7456519871546352＝探索，5 折權重差 0）。獨立稽核（另一個子代理，`freeze_audit.json`）：自寫前向在 inner 全部假題上 E_SET 最大差 9.2e-11、θ 由同一個確定性擬合重現（差 0）、方向參照重現，沒有影響預測的程式錯誤。

1. **程式**：`tools/t25_rank.py`（`apply`、`status`；import `tools/t19_rank.py`、`t09_rank.py`，經由它們 08／07／04／02 的函式，不改）、`tools/official_log.py` 只加行、`docs/TESTDAY_25.md`、`tests/test_t25_rank.py`、`tests/test_t25_official_log.py`、`tests/test_t25_dryrun.py`（`C10_SLOW=1`）、`tests/test_p25_hygiene.py`、`tests/test_t25_hygiene.py`（兩個加進 AGENTS.md 的閘門指令）、`.gitattributes`（`results/test25/** -text`）加行。
2. **tag `p19-testday-pre25`** 打在 main 的 `163d413`（25 的任何程式之前）。`tests/test_t25_hygiene.py` 要求 `official_log.py`、`.gitattributes`、`.gitignore` 對它只加行；R20–21 的測試日工具、它們的測試、`docs/TESTDAY*.md`（25 除外）、`docs/PREREG_0*`–`PREREG_24`、`results/prereg01`–`prereg24`、`results/test02`–`test21`、`third_party`、`colab*` 自它起不變。
3. **第 4.2 節的實作**：
   - **不讀 `data/local`**：t25 不 import／不呼叫 t09／t08／t04／t02 的前提函式（`t9.prereqs`、`check_test02`、`load_opening_test`：它們會雜湊 seam_val 的題目檔）。test02 的 `opening_test.npz` 用 t25 自己的讀法，只核對 public、private 的題目檔 SHA（和 `t02_rank.load_opening_test` 其餘的格式檢查相同）。其他用到的讀法：`t9.check_p09_pins`、`t7.check_p07_pins`、`t4.load_frozen_config`（釘住檔和格式，不碰 `data/local`）、`t9.Apply.p1_sets`、`t7.load_sw`、`t4.load_c5`、`t4.fuse_set`、`t7.port_c5`、`t7.q02h_set`、`t8.q08_core`、`t19.set_term_of`、`t19.q19_core`、`t19.set_seam_record`、`t9.probs_rows_compare`；兩組都只跑 public、private。
   - **19 的正式執行**：報告的格式、模式、結束碼 0、usable、Q19；報告、6 個輸出、`set_seam.json`、`seam_val/seam_local.json` 已 commit（正式模式）；報告 `inputs_bound` 的每一個檔重算 SHA（`--dry-run` 依樹的路徑對應）；輸出 SHA 等於報告；T1 從它綁定的 `t1.json` 重讀（正式模式要 PASS／NORMALIZED）、test02 的 `fallback_used` 要是 false。
   - **`--dry-run`**：可以讀 19 的**正式**執行（報告 `mode` 是 real 或 dry-run）；08 的 P1 塊照 19 綁定那一次的模式讀（`p1_sets` 的模式檢查）。正式模式只接受 real。
   - 第 2、3 步用 `t9.probs_rows_compare`（機率差 ≤ 1e-12、規則 A／B 等級全同；位元組相同只記錄）；lp_SET19 只在 `has` 的題比較（沒有可用盤的題 lp 不會被使用）。
   - 被 import 的工具丟出的錯誤：拒絕是 2、漂移（4）照樣是 4、它們的「不用」（3）在 t25 是 2（25 不替別的套件做登記判定）。
   - 報告另記：25 和 19／09／08／07／04／R20 的規則 B 差異題數（public、private、全部分開）、每組和 Q19 不同的題數、`data_local` 欄（從不開、不列、不雜湊）。
4. **`official_log.py`（只加行）**：25 檔在 19／09／08／07／04 之前認出；最終選擇的紀錄鍵是 `prereg25_4_3`；04、07、08、09、19 的 prepare／final 規則跳過 25（各加「25 跳過」的行），status 的違反判定把「成立中的 25 最終選擇」當成合規，有符合的 25 時提示改成「最終選擇要設成 25」（蓋過 19 以下的提示）；public 的同檔警告 04–19 跳過 25，25 自己另有一個；`prepare`／`final` 加 `--not-25-reason`（rev 4 起、晚於那幾筆 25 的 prepare）；`status` 有 25 紀錄時印 K6-25。**沒有 25 紀錄時，所有輸出、紀錄檔、回傳值和 tag `p19-testday-pre25` 的版本逐位元組相同**（`tests/test_t25_official_log.py` TagVersusNew：R20／R10／04／06、07、08、09、14 的流程，加一條 R20→…→19、final 19 的流程）；真的紀錄檔 `results/official_scores.json` 的 `status` 輸出也和 tag 的版本逐位元組相同。
5. **14**：沒有改（第 4.1 節）；`official_log.py` 的 `T14_BASE_DIRS` 不加 25。
6. **演練**（`C10_SLOW=1`，`--dry-run` 讀 repo 裡已 commit 的 test02–test19 輸出和 `data/test08/p1`、`data/test`，輸出寫在 repo 外的 subst 磁碟）：正常執行結束碼 0，Q09、Q19 重算和 test09、test19 **逐位元組相同**、lp_SET19 差 0，check_submission（官方模式）過；負面演練：輸出已存在（2）、K6-25 已過而之前沒有成功（3，`not_used.json`）以及在它上面重跑（沒有 `--bug-revision` 是 2、沒有 rev 4 也是 2）、19 的報告被改（2）、Q19 漂移（複製的 test19 樹裡一個機率差 1e-9：4，`attempt_verdict.json` stage `q19_vs_19`，沒有交件檔）、SET25 方向沒過（子程序把下限提高：3）、BLAS 變數（2）。正式執行之後，演練的輸出要和 `results/test25/rank/` 逐位元組相同（同一個測試會檢查）。只記錄：演練時 25 和 19 的規則 B 差異 public 42 題、private 47 題；SET25 的 GPU 接縫最大差 0.0084（只記錄）。這些數字不進任何判定。
7. **容許差**：沒有新的容許差（第 2、3 步沿用 1e-12）。
8. **閘門**：AGENTS.md 的閘門指令加 `tests/test_p25_hygiene.py`、`tests/test_t25_hygiene.py`。`tests/test_g11_hygiene.py::InsertOnly::test_official_log_untouched` 要求 `official_log.py` 和 main 相同，所以在分支上是紅的，合併後是綠的（同 PREREG_19 10 第 8 條）。分支的工作樹有 9 個只差換行的 `results/prereg07/explore/pilot_humansl/` 檔（worktree 的換行問題，同 prereg23／25 探索），`test_t25_hygiene.py::Frozen` 在分支上看得到它們，在 main 上是綠的。
9. **獨立審查**（另一個子代理，讀程式和登記、跑 25 的單元、official_log、hygiene 測試和 `C10_SLOW=1` 演練、另用 `sys.addaudithook` 跑一次 `--dry-run` 記錄所有開檔／列目錄；沒有碰 `data/local`）：沒有 blocker、沒有 major；audit hook 實測 t25 對 `data/local` 的存取 0 次、所有寫入都在 R25；真紀錄檔的 `status` 和 tag 版逐位元組相同；把演練報告餵給 `official_log` 的 `t25_conditions` 沒有問題、`inputs_bound` 涵蓋 `T25_INPUTS_REQUIRED`。3 個 minor 都在合併前處理：
    - ① `call()` 只轉換 t19／t09 的錯誤，t07／t08／t04／t02 的漂移（4）會變成內部錯誤 2、不寫 `attempt_verdict.json` → `call()` 改成轉換 t19／t09／t08／t07／t04／t02 六種錯誤（4 照樣是 4），`set_term_of`、`q19_core`、`set_seam_record` 也改經 `call()`。
    - ② check_submission 失敗時刪輸出沒有經過寫入防護 → 刪之前先 `guard_write`。
    - ③ 「不碰 `data/local`」只靠禁字串檢查 → `tests/test_t25_dryrun.py` 的正常演練改在 `sys.addaudithook` 底下跑（open／listdir／scandir 的路徑含 `data/local` 就失敗），實測 0 次。
    修完重跑：25 的單元、hygiene、`C10_SLOW=1` 演練全過。

