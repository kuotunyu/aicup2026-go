# AI CUP 2026 圍棋棋力與風格預測競賽：競賽一＋競賽二的程式碼

**English summary.** Code, pre-registrations and result records for the two tracks of AI CUP 2026 "Go rank and style prediction" by one author entering as two teams: Track 1 (identify the player from a set of games, TEAM_11249, Private 0.8697239, rank 27) and Track 2 (predict the rank of a player from 5–20 games, TEAM_11248, Private 0.7267759, rank 30). The repository holds the data processing, training and inference code, the frozen parameters and the judgement records. It does **not** hold the official game records, anything computed per game or per question from them, or model weights; weights and intermediate features are provided through the cloud-drive link below, and the KataGo network is downloaded from its official release. Reproduction is described in three layers (features → submission, test files → features, training). Program code and documents were produced with generative AI tools (see "生成式 AI 使用聲明").

---

## 1. 專案概述

| | 競賽一：從圍棋棋譜集辨識玩家 | 競賽二：從圍棋棋譜預測玩家棋力等級 |
|---|---|---|
| 隊伍 | TEAM_11249 | TEAM_11248 |
| 題目 | 每題給一位玩家的數盤棋，從候選庫排出最可能的 5 位玩家 | 每題給一位玩家的 5–20 盤棋，預測 10 個等級之一 |
| 計分 | 正確玩家在 Top-5 第 r 名得 e^-(r-1)，取平均 | 完全猜中 1 分、差一級 1/e、其他 0，取平均 |
| 官方 baseline | 0.2845 | 0.4631 |
| 最終交件 | v2-9（`results/prereg67/`） | 69（`tools/t69_rank.py`） |
| Public | 0.8417387 | 0.7493017 |
| **Private** | **0.8697239，第 27 名** | **0.7267759，第 30 名** |

兩支隊伍是同一位作者；兩場的前處理程式共用，報告分開撰寫。官方分數的紀錄在 `results/official_scores.json`（v1 測試集）與 `results/official_scores_v2.json`（競賽一 v2 測試集）。

出題單位：國立東華大學智慧科技中心；平台 AIdea。

## 2. 這個 repo 有什麼、沒有什麼

有：

| 目錄 | 內容 |
|---|---|
| `tools/` | 共用前處理（`c10data.py`、`make_val.py`、`root_props.py`…）、各事前登記的正式工具（`p0x_*.py`）、測試日工具（`t*_*.py`）、交件檔檢查（`check_submission.py`）、上傳紀錄（`official_log.py`） |
| `results/prereg*/` | 每個方法一個資料夾：`explore/` 內是該輪的 `PLAN.md`（先於任何數字 commit）、程式、判定 JSON、`RESULTS.md`／`NOTES.md`；`rank/`、`player/` 內是凍結設定 |
| `results/test*/` | 測試日各層的設計文件、演練紀錄與報告 JSON |
| `docs/` | 事前登記（`PREREG_*.md`）、測試日步驟（`TESTDAY_*.md`）、資料說明（`DATA.md`）、驗證規則（`VALIDATION.md`）；導覽見 `docs/README.md` |
| `repro/` | 兩場最終交件的重現入口與說明（`repro/README.md` 是最完整的重現文件） |
| `colab/`、`colab_gpu/` | Colab notebook 與產生器（早期 CNN 訓練／推論；KataGo 抽取的 Colab 備援） |
| `tests/` | 單元測試、測試日演練（防誤改檢查 `test_*_hygiene.py` 不在這份副本，見下方說明） |
| `third_party/` | 官方教學程式與 KataGo 的 python 模組（原樣副本，各附來源與授權，見第 9 節） |

沒有（刻意不放）：

- **官方棋譜與測試檔**：僅供本競賽使用，不轉載。取得方式見第 4 節。
- **由官方棋譜逐盤／逐題算出的檔**：逐盤特徵與模型輸出（`.npz`）、逐題機率與預測、交件檔與 Top-5（`.csv`）、玩家名單、含題號或玩家代號清單的報告 JSON。原本的工作用 repo 有這些檔，這裡拿掉了；需要的部分由第 5 節的雲端硬碟提供。
- **模型權重**（`.pt`／`.pth`／模型用的 `.npz`）：見第 5 節。
- 工作日誌、報告草稿、報名與帳號相關紀錄。

本 repo 是私有開發 repo 在 commit `6a43ae2`（全碼 `6a43ae20e99d002b4c09de7ce6e251d55017e2c0`）的單一快照；文件中引用的其他 commit 編號（例如「`dbfe55e` 先於數字」）屬於私有歷史，在這裡查不到。文件裡的本機絕對路徑（`D:\AI-Portfolio\…`、`C:\Users\USER\…`）是當時的工作環境，使用者名稱已換成 `USER`；程式本身用相對於 repo 根目錄的路徑。

**和私有開發 repo 的差別**：

- 防誤改檢查 `tests/test_*_hygiene.py`（29 個檔）沒有放進來：它們比對私有 repo 的 git tag、歷史與上面拿掉的檔，在單一 commit 的快照上必然失敗。文件裡提到的「閘門」指的就是這批測試。其餘測試都保留；其中需要逐盤特徵或驗證集的測試，在沒有那些檔的環境會略過或失敗。
- 第 1 層重現要先用 `repro/stage_inputs.py` 把你自己下載的官方測試檔（第 4 節）和雲端硬碟的三個包（第 5.2 節）放回原路徑（第 6.1 節）。

## 3. 環境

- 作業系統：開發與最終推論在 Windows 11（Git Bash）；Linux／Google Colab 可跑第 1 層（見 6.1）。
- Python **3.12.13**。競賽二的凍結工具會檢查 python 3.12.13、numpy 2.5.3、scipy 1.18.1 三個版本完全相同，不同就停下。
- 兩個環境：CPU（`repro/requirements_cpu.txt`，torch 2.14.0+cpu）與 GPU（`repro/requirements_gpu.txt`，torch 2.11.0+cu128；開發機是 RTX 4090）。

Windows（Git Bash，在 repo 根目錄）：

```
uv venv --python 3.12.13 .venv
uv pip install --python .venv -r repro/requirements_cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv venv --python 3.12.13 .venv-gpu
uv pip install --python .venv-gpu -r repro/requirements_gpu.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Linux／Colab：同樣兩行，直譯器是 `.venv/bin/python`；`colorama` 只是 Windows 上 pytest 的相依，可以略過。

執行時的共同規定：

- 指令前加 `PYTHONDONTWRITEBYTECODE=1`。
- **不要設** `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`（競賽二的工具會拒絕執行）。
- 資料放在 repo 根目錄下的 `data/`（不進 git），結果在 `results/`；程式用自己檔案的位置推算根目錄。

## 4. 資料取得

- 官方訓練棋譜：<https://github.com/AILAB-NDHU/aicup2026-go-dataset>（release v1.0 `aicup2026_go_training.tar.xz`，解壓到 `data/training/`；v1.1 只多了每盤開頭的 `HD`，本專案只在 prereg31 的探索用過）。**僅供 2026 AI CUP 競賽使用，本 repo 不轉載。**
- **官方測試檔（不得轉載，本 repo 與雲端硬碟都沒有，請自行下載）**：從 AIdea 兩場競賽頁面的「資料」分頁下載（需參賽帳號）。第 1 層重現需要下面兩個下載檔裡的 10 個 CSV：
  - 競賽一 v2 測試集（2026-11-11 公布）：`aicup2026_player_identification_v2_tests_20261108_fix_readme.zip`（35,831,860 bytes，MD5 `90cda19fc750f0c99ab75c0706ba9314`，SHA-256 `11054312b9f24a175dbe2e6eebc86b56ba1442299e74bec38a3c14c15e593a58`），內有 8 個 `player_identification_*.csv`。
  - 競賽二：`aicup2026_rank_prediction_tests_20261030.zip`（3,447,261 bytes，MD5 `355c0b40be6af0413859991ebbcf31f3`，SHA-256 `8e1020d5e2b86dae2e4978ee6fe790b05e6f8191279b6dbc15e32fabc02f05c7`），內有 2 個 `rank_prediction_test_*.csv`。
  - 兩個 zip（或解開的 CSV）放進同一個資料夾，交給 `repro/stage_inputs.py --official-dir`（第 6.1 節）：它依檔名尋找、核對下表的 bytes 與 SHA-256，相符才複製到 repo 路徑。zip 的雜湊是 2026-11-11 下載時的值，主辦方重新打包的話可能不同；以下表 CSV 的雜湊為準。

| 檔名 | 放到 repo 的路徑（`stage_inputs.py` 會自動放） | bytes | SHA-256 | MD5 |
|---|---|---|---|---|
| `rank_prediction_test_private.csv` | `data/test_v2_download/rank/` | 6,752,165 | `d59865e24fe2f5ad742db68108a7f59c2e41138f18c4dc2eea1aa364f3e73c73` | `0580468d339696b6aa9251497d0c5584` |
| `rank_prediction_test_public.csv` | `data/test_v2_download/rank/` | 6,775,591 | `4691803a8306ccd65d93a90bd6d7ba767d57abe96c0660b262f1a2b756b84fef` | `26d73aa37987fc3f67e55dd7ab7fa4ba` |
| `player_identification_candidates2_private.csv` | `data/test_v2_download/x/tests/` | 8,210,305 | `8e80f94689284722bd21a3f3306c476ee78aead60a4061e74a0f8aaaede36812` | `d790e9ead87001c1edae191aa659a5d4` |
| `player_identification_candidates2_public.csv` | `data/test_v2_download/x/tests/` | 8,114,857 | `ac3d7d20d1a19ee7e670638b556f5c3cc021904ca4e1644bc0d0bcec4ff6616d` | `2529fd99ff1f7f4f9b76c293a4d7e203` |
| `player_identification_candidates_private.csv` | `data/test_v2_download/x/tests/` | 55,008,481 | `db2cfe33be9c1e36355fac5664b758909ef80bce1be257e55bae056fba40d963` | `b1a90af9a883ada9faa4e82723634ad3` |
| `player_identification_candidates_public.csv` | `data/test_v2_download/x/tests/` | 54,956,081 | `e662242cb0f0b2be2009c129abd291c9c852af7fae25d8f1e1c00208785ee24d` | `44544cc3c7f4f01821fc11eb835a5a2c` |
| `player_identification_test2_private.csv` | `data/test_v2_download/x/tests/` | 1,481,715 | `52e01bcc508e3e350f47b6430a7c841531976ac4a9d294fc2328b4570ca8a527` | `f0e975e3e12ec26eeadd6b606fc8504e` |
| `player_identification_test2_public.csv` | `data/test_v2_download/x/tests/` | 1,468,253 | `33485b58da29a43d53526f9f83d92ebdc4610948cb0f5249e07bba5cbdd53d6a` | `f87a039954a1e7e47105ef192054ecdf` |
| `player_identification_test_private.csv` | `data/test_v2_download/x/tests/` | 6,756,005 | `eedd2630731070633dfa264897ac739aec03879f90ac57149bba033d9bd53dee` | `172bf82ac324103ad0228b93f4729c03` |
| `player_identification_test_public.csv` | `data/test_v2_download/x/tests/` | 6,705,235 | `fbf6f14de57bc37bbec6e3d732143a2561887d37b1f36271223ad1a14810ac74` | `d5351b45751dbef7f284e6164ab8f77a` |

  - 另有 6 個檔是原本執行時由上面的官方檔做出來的，同樣不散布，由 `stage_inputs.py` 從你下載的檔重建並核對 SHA-256：`data/prereg56/src_rank/player_identification_test2_{public,private}.csv`（競賽二題目檔的改名副本）、`data/prereg56/src_rank/player_identification_candidates2_{public,private}.csv`（只有表頭一行）、`data/test/rank_prediction_test_{public,private}.csv`（競賽二題目檔拿掉每盤開頭的 `HD[n]`，即測試日 `tools/root_props.py` 的處理）。
- 官方教學程式：<https://github.com/AILAB-NDHU/AICup-2026-Tutorial>（本 repo 的 `third_party/aicup2026_tutorial/` 是原樣副本）。
- 資料格式與已知的坑：`docs/DATA.md`。本機驗證集的切法：`docs/VALIDATION.md`、`tools/make_val.py`（按玩家切，split v1）。

## 5. 權重與中間特徵

### 5.1 外部權重：KataGo HumanSL（請自行下載，本 repo 不轉發）

| 項目 | 值 |
|---|---|
| 來源 | <https://github.com/lightvector/KataGo/releases/tag/v1.15.0> 的 release asset |
| 檔名 | `b18c384nbt-humanv0.ckpt` |
| 大小 | 323,752,318 bytes |
| SHA-256 | `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27` |
| 授權 | <https://katagotraining.org/network_license/> |
| 放哪 | `data/katago/b18c384nbt-humanv0.ckpt` |
| 用法 | 只做推論、不微調：讀中性設定下的內部表示（稱 P1 特徵）與各棋力設定的逐手機率（稱 HS） |

### 5.2 本隊的權重與特徵：`【雲端硬碟連結】`

下載後照表中的路徑放到 repo 根目錄下。SHA-256 是 2026-11-25 在開發機實測的值。

兩場最終交件用到的權重：

| 路徑 | bytes | SHA-256 | 用在 |
|---|---|---|---|
| `data/prereg49/S0/ckpt/step_320000.pt`（c49） | 69,578,397 | `c091d3af1797a19ce98dc0a670eea211ceca2de4b58ee4a3c62b73b854748e12` | 競賽一新題；競賽二的對手項 |
| `data/prereg55/S/ckpt/step_320000.pt`（S） | 69,576,733 | `23c91566f5aa52daa6fc4aba32f6fdf7b9defdcff1653a25a2d294b8c21d7a3d` | 競賽一新題集成 |
| `data/prereg51/B/ckpt/step_320000.pt`（B） | 95,177,401 | `e61a4feb6ee4438587622c16e79a427b54622fc8de2caf5af9c4a1ab6199c192` | 同上 |
| `data/prereg61/S0/ckpt/step_200000.pt`（p61） | 69,597,213 | `046336c9618f6fc47dba2e9c607e9aa0fbd24914734c17fbf605bf2c5b4c5a64` | 同上 |
| `data/prereg62/S1/ckpt/step_200000.pt` | 69,595,549 | `2ec6f2714a1db4ec40dbace9275bfa48afee9e7cb3a76fa3574d47e70376450e` | 同上 |
| `data/prereg62/S2/ckpt/step_200000.pt` | 69,595,549 | `5d3e4165e9ea70c9a42a8da08fe767fbf6a7927930334e1e03a011f694ea1bff` | 同上 |
| `data/prereg62/X/ckpt/step_400000.pt` | 69,595,549 | `2c6f6796fbca3a480a807017cbd10d4415d523b9b3059c0593bef0ed348957b4` | 同上 |
| `data/prereg01_weights/player/best_model.pth` | 19,212,521 | `356eb4f65488925e2a552ea382979780c903ef515b85d329a3805db837b7773b` | 競賽一原題（06 路徑的 CNN） |
| `data/prereg02_weights/rank/epoch_9.pth` | 105,581,163 | `423581b4fddd5c2c160ebf691f4e86b9e00c4fe743d5cfb23df7891f4a2ca939` | 競賽二 C5 |
| `data/prereg03_weights/rank_a/final.pth` | 105,589,835 | `790b767c5425e262130d6c103c71e89238edb17f927fa48e2a2770db788e50ac` | 競賽二 SWA |
| `data/prereg03_weights/rank_b/final.pth` | 105,589,835 | `8e88f6ec6cbaad5372e5d62607b99ece3d77dc7a20af86a9197553f1a6c59a42` | 競賽二 SWB |
| `data/prereg04/counts_fit.npz` | 178,537,934 | `f6c6debe90639585763b9812c235743a6b7f89d8db1c6422ece7bccd550a08a5` | 競賽二 O4 計數表 |
| `results/prereg09/rank/p1e_map.npz` | 11,167,184 | `ecc9c9b1747b7b7d40cf53869b6c67c8e6f776ba7498498737fe3eede8e2edba` | 競賽二 P1TO_E；競賽一段位相容項 |
| `results/prereg25/rank/s25_seed0.npz` | 1,655,779 | `4a01d6a60355cd514e9cc4988c08424a7952fc33302b47a63542a8caf67163bb` | 競賽二 S25；競賽一段位相容項 |
| `results/prereg25/rank/s25_seed1.npz` | 1,655,779 | `ab53d69b2762a7c01bed354ac6c218cf167fda44bbd3d88eed06a6f4adf3cba4` | 同上 |
| `results/prereg25/rank/s25_seed2.npz` | 1,655,779 | `20e190f56866c7a28b4eeef0c0d4d656867b79a908e066ba91a47570601c4fbb` | 同上 |
| `data/prereg64/c49_head.npz` | 6,326,550 | `ea7408d96181e5f6012bb7aafd1f4c2c31a66be77bf72b8220d688a41d0f0a6a` | 競賽二對手項（c49 分類頭） |
| `data/prereg64/prototypes.npz` | 4,470,398 | `6ffe6682b6584ee08c544247ef2f6ec9371ab08c59fb726b741b6fe5ff5565f7` | 競賽二對手項（2,104 個原型） |

第 1 層重現要的中間特徵（CNN 向量、KataGo P1 特徵塊、各層的逐盤輸出與凍結檔）打包成三個包（zip，不壓縮，內部是 repo 相對路徑），放在同一個雲端資料夾：

| 包 | 檔數 | bytes | SHA-256 |
|---|---|---|---|
| `C10_public_features_comp1.zip` | 897（`data/` 下） | 4,598,622,322 | `2ecd3ffee9fb8c53319556d65b908efdc84ae693ca569522b7ed33953afaeed5` |
| `C10_public_features_comp2.zip` | 56（`data/` 下） | 120,212,732 | `be67d7bc212a7ed2fa3046d253ba62b41262ce0dcb7e377f9c707fc84d8194d9` |
| `C10_features_layer1_ingit.zip` | 67（`results/` 下 66 個＋`repro/ref_lda_L.npz`） | 71,507,868 | `ccff1b16c80af5e4eef88943223efa677d232b1c0c1cff2640f927786b8baa64` |

- **三個包都不含官方棋譜或官方測試檔**；官方測試檔請照第 4 節自行從 AIdea 下載。
- 前兩個是開發時就不進 git 的特徵檔；第三個是私有 repo 有追蹤、這份公開副本拿掉的第 1 層輸入檔（凍結模型檔、各層的逐題機率與交件檔、逐盤輸出）。
- 每個檔的路徑、bytes、SHA-256 與來源列在 `repro/inputs_public.json`（共 1,090 個：三個包 897＋56＋67，本 repo 內 54，官方測試檔 10，由官方測試檔重建的 6）；兩支入口各自讀哪些檔見 `repro/inputs_comp1.json`（922 個）與 `repro/inputs_comp2.json`（169 個）。
- 不用手動解壓：`repro/stage_inputs.py` 會核對 SHA-256 並放到原路徑（第 6.1 節）。

## 6. 三層重現

| 層 | 從哪裡到哪裡 | 硬體 | 狀態 |
|---|---|---|---|
| 第 1 層 | 已存的中間特徵＋官方測試檔 → 最終交件 CSV，比 SHA-256 | CPU | 有單一入口；開發機（Windows）上兩場都逐位元組相同 |
| 第 2 層 | 官方測試檔 → 中間特徵 | GPU | 列出實際用過的指令，沒有單一入口；抽樣比對過 4 項 |
| 第 3 層 | 訓練（權重、LDA、融合參數） | GPU／CPU | 列出程式與指令，沒有重跑 |

細節、實測耗時與已知限制都在 `repro/README.md`，下面只列入口。

### 6.1 第 1 層：從特徵重算交件檔

最終交件檔的 SHA-256：

| 交件 | 列數 | SHA-256 |
|---|---|---|
| 競賽一 v2-9 `submission_player.csv` | 1,600 | `36b222d866267abffda03d126d94db9ed5bea3406b2f8b47323622fd5329a026` |
| 競賽二 69 `submission_rank.csv` | 800 | `c7595f2eb43c185973c55c7b4b03875de41c5ce92327f7042fc3769dbbae36c8` |

步驟：

1. clone 時保留 git 的原始位元組：`git clone -c core.autocrlf=false https://github.com/kuotunyu/aicup2026-go.git`（Windows 上若讓 git 把換行轉成 CRLF，以 SHA-256 釘住的輸入檔會對不上）。
2. 照第 3 節建 `.venv`。
3. 照第 4 節從 AIdea 下載官方測試檔（兩個 zip，放進一個資料夾，不必解開），把第 5.2 節的三個包下載到另一個資料夾，在 repo 根目錄執行（只用標準函式庫，任何 Python 3 都可以）：

```
python repro/stage_inputs.py --zips-dir <放三個包的資料夾> --official-dir <放官方測試檔的資料夾>
```

   它會：核本 repo 內的 54 個輸入檔 → 依檔名找出 10 個官方測試 CSV、核 SHA-256、複製到原路徑，並重建 6 個由它們做出的檔（第 4 節）→ 核三個包的大小與 SHA-256 → 解壓到原路徑並逐檔核 bytes 與 SHA-256 → 把 `C10_features_layer1_ingit.zip` 的 67 個檔 `git add -f` 並 commit → 確認工作樹乾淨。可以重複執行；`--check-only` 只檢查。
   **為什麼要 commit**：競賽二的凍結工具要求輸入檔被 git 追蹤而且和 HEAD 相同（`tools/t25_rank.py` 的 `require_committed`）。私有 repo 本來就追蹤這些檔；這份副本拿掉了，所以改在你自己的 clone 裡 commit。**這個 commit 只留在本機，不要推到任何地方**（內容是由官方棋譜算出的檔）。官方測試檔放在 `data/` 下，不會進 git。凍結的工具本身沒有改。
4. 在 repo 根目錄執行（`--out` 必須是 repo 之外的資料夾）：

```
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe repro/comp1_from_features.py --out <repo 外的資料夾>
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe repro/comp2_from_features.py --out <repo 外的資料夾>
```

Linux／Colab 把直譯器換成 `.venv/bin/python`。兩支入口最後會比對上表的 SHA-256 並印出 PASS／FAIL。

- 開發機實測（2026-11-25，Windows，私有 repo）：兩支都 PASS；競賽一約 2–4 分鐘、峰值記憶體約 2 GB，競賽二約 10–30 秒。
- 這份公開副本的實測（2026-11-26，Windows）：在空資料夾 `git clone` 本 repo 的 commit → `stage_inputs.py`（官方測試檔用 AIdea 下載的兩個 zip）→ 兩支入口，**兩場都逐位元組 PASS**。
- Linux 上浮點數最後一位會和 Windows 不同，工具內「逐位元組相同」的中間檢查會停下。這時加 `--tolerant`：中間的數值檢查改用容許差（絕對差 ≤ 1e-9 + 1e-9 × 參考值），實際差異寫進 `<out>/tolerant_report.json`；PASS 的判定不變，仍然只看最終 CSV 的 SHA-256。容許模式另需 `repro/ref_lda_L.npz`（在 `C10_features_layer1_ingit.zip` 裡，`stage_inputs.py` 會放好）。Linux／Colab 上的實測結果見 `repro/README.md` 第 2.4、5 節。
- Colab notebook：**`repro/colab_public.ipynb`**（CPU 執行階段）。第 0 格的參數：`REPO_URL`（本 repo）、`DRIVE_DIR`（你的 Google Drive 上放三個包的資料夾；也接受 `parts/` 的 7 MB 分塊＋`C10_public_parts_manifest.json`）、`OFFICIAL_DIR`（你放官方測試檔的資料夾）、`MOUNT_DRIVE`、`SAVE_TO_DRIVE` 等。它依序 clone 本 repo → 取三個包 → `stage_inputs.py` → 用 `uv` 建 Python 3.12.13 環境 → 兩支入口先跑嚴格模式、停在接縫就加 `--tolerant` 再跑 → 比 SHA-256。這份 notebook 在公開 repo 建立前寫成，還沒有在 Colab 上執行過。
- `repro/colab_layer1.ipynb`、`repro/wsl_drill.sh` 與 `repro/make_delivery.py` 的 `code`／`bundle`／`features`／`sums` 是開發時用私有 repo 的 git bundle 交付的版本，留作紀錄；公開 repo 不提供 bundle，請用上面的流程（`repro/README.md` 第 5.6 節）。
- 競賽一原題的 800 列沿用 v1 測試日 06 路徑已產生的交件檔，第 1 層不重算（原因見 `repro/README.md` 第 2.2 節）。

### 6.2 第 2 層：從測試檔抽特徵（GPU）

`GPY`＝`.venv-gpu/Scripts/python.exe`，在 repo 根目錄執行。完整表格在 `repro/README.md` 第 3 節。

| 特徵 | 指令 |
|---|---|
| 競賽一新題：7 個 CNN 成員的兩池向量 | `GPY -u results/prereg49/explore/p49_run.py embed --ckpt <ckpt> --weights raw --tta 1 --workers 4 --emb-root data/prereg49/emb --source pool --src data/test_v2_download/x/tests --pool public\|private`；其他成員用 `results/prereg55/explore/p55_run.py`、`prereg51/explore/p51_run.py`、`prereg61/explore/p61_run.py`、`prereg62/explore/p62_run.py` 的 `embed` |
| 競賽一新題：KataGo P1 特徵 | `bash results/prereg39/explore/p39_extract.sh --sets pub2,priv2 --workers 4` |
| 競賽二：KataGo P1 特徵 | `GPY tools/t08_rank.py p1 …`（步驟 `docs/TESTDAY_08.md`） |
| 競賽二：c49 對手方向量 | `GPY -u results/prereg56/explore/p56_opp.py embed --ckpt data/prereg49/S0/ckpt/step_320000.pt --weights raw --tta 1 --workers 2 --emb-root data/prereg56/emb_opp_rank --source pool --src data/prereg56/src_rank --pool public\|private` |
| 競賽二：官方 CNN 的逐盤輸出（C5、SWA／SWB） | `colab/test02_rank.ipynb`、`colab/test04_rank.ipynb`（Colab GPU） |
| 競賽二：開局、對手 O4 | `tools/t02_rank.py`、`tools/t04_rank.py`（CPU） |

CNN 向量用 fp16 推論，重抽不是逐位元組相同（抽樣比對最大絕對差約 7.6e-06）；KataGo P1 抽樣 20 盤逐位元組相同。所以「逐位元組重現」只對第 1 層成立。

### 6.3 第 3 層：訓練

| 產物 | 程式 | 紀錄的硬體與時間 |
|---|---|---|
| c49（競賽一主幹 CNN，320,000 步） | `GPY -u results/prereg49/explore/p49_run.py train --workers 4 --auto-resume --log-every 1000` | RTX 4090，107,434 秒 |
| S、B、p61、S1、S2、X（其他 6 個成員） | `results/prereg55/explore/p55_run.py`、`prereg51/explore/p51_run.py`、`prereg61/explore/p61_run.py`、`prereg62/explore/p62_run.py`；Colab 版的產生器是同資料夾的 `p*_colab*.py` | Colab GPU，各 11–27 小時 |
| KataGo P1 的 LDA、多子空間投影 | `results/prereg39/explore/p39_eval.py`（apply 時現場重學）、`results/prereg63/explore/p63_eval.py fit` | CPU（P1 抽取用 GPU） |
| 競賽一原題的特徵庫、LDA、CNN | `tools/p06_player.py`、`tools/p06_cnn_fit.py`、`colab/prereg01_player.ipynb`、`colab/prereg02_player.ipynb` | Colab GPU＋CPU |
| 競賽二 C5、SWA／SWB | `colab/prereg01_rank.ipynb`、`prereg02_rank.ipynb`、`prereg03_rank_{a,b}.ipynb` | Colab GPU |
| 競賽二開局、O4 計數 | `tools/p02_rank_opening.py`、`tools/p04_opp.py count` | CPU |
| 競賽二 P1TO_E（6 個翻譯模型）、S25（set 模型） | `results/prereg08/p1/p1_extract.py`、`tools/p09_map.py train`、`results/prereg25/explore/p25_extract.py`／`p25_prep.py`／`p25_train.py` | GPU（抽取）＋CPU（訓練） |
| 競賽二融合參數、對手項的頭／原型／條件表 | `tools/p25_set.py freeze`、`results/prereg64/explore/p64_freeze.py`、`p64_head.py`、`p64_rows.py`、`p64_eval.py kernel` | CPU |

每個訓練程式的完整參數寫在各自的檔頭說明與同資料夾的 `PLAN.md`、`train_record.json`。

## 7. 兩場的最終推論流程與重要模組

### 7.1 競賽一（v2-9）

交件檔 1,600 列＝原題 800 列＋新題 800 列（v2 測試集，2026-11-11 公布）。

- **原題 800 列**：沿用 v1 的 06-ON。官方教學 CNN 的特徵＋開局習慣＋風格特徵（LDA、角落定式等）＋CSLS，另有「一題對一人」的自動開關（`docs/PREREG_06_PLAYER_STYLE.md`、`PREREG_28_06ON_N_BY_N.md`）。
- **新題 800 列**（每題 1／2／3／5 盤、黑白都有、候選 600 人各 10 盤）：

```
分數 = csls_20( G + 14·E_ens + (5.5·z_p1 + 3·z_multi + 3·z_sp128) + 1.8·z_rk ) + w_post(k)·z_rk
```

  - `G`：手工項（開局、LDA、定式等，prereg35）。
  - `E_ens`：7 個逐手 CNN 的分數層集成（c49、S、B 吃 17 個輸入平面；p61、S1、S2、X 另加「這一手下在哪」共 18 個平面；各自在該池候選庫上白化後取 z 分數，權重 1/1/1/2/2/2/2）。
  - `z_p1`、`z_multi`、`z_sp128`：KataGo P1 特徵的三種投影。
  - `z_rk`：段位相容項（用競賽二的模型估題目與候選人的段位是否相容），`w_post(k)` 依題目盤數加權。
  - 每池分開計算、全自動，候選庫只用來做正規化。

| 模組 | 輸入 | 輸出 |
|---|---|---|
| `results/prereg36/explore/p36_lib.py` | SGF 文字 | 逐手輸入平面（官方教學的 17 平面）、8 種對稱、CNN 模型定義（ResNet、cosine-softmax 頭） |
| `results/prereg42/explore/p42_lib.py`、`p42_train.py`、`p42_embed.py` | 訓練玩家的棋（`tools/c10data.load_rows`）；或官方測試檔資料夾 | 訓練：`data/prereg*/…/ckpt/step_*.pt`；embed：每盤一個向量的 `pool_<池>.npz`／`store_*.npz` |
| `results/prereg49/explore/p49_run.py`（及 p51／p55／p61／p62 的 `*_run.py`） | 同上，各成員的設定 | 各成員的權重與向量 |
| `results/prereg39/explore/p39_extract.py`、`results/prereg08/p1/p1_lib.py` | 棋局名單＋KataGo 權重 | KataGo P1 特徵塊 `data/prereg39/p1/<集合>/chunk_*.npz`（每塊 200 盤） |
| `results/prereg38/explore/p38_backend.py` | 池（題目列、候選列、各項分數矩陣）＋向量 | 題目 × 候選的分數矩陣（池內白化、LDA、CSLS） |
| `results/prereg66/explore/p66_eval.py`、`results/prereg67/explore/p67_eval.py` | `apply --src <官方測試檔資料夾> --pool public\|private`；`assemble --src …` | 每池 `top5.csv`、`apply_report.json`；`assemble` 產生 `submission_player.csv` |
| `tools/t06_player.py`、`tools/p06_player.py`、`tools/p06_style.py` | 原題測試檔＋CNN 逐盤輸出 | 原題的 Top-5 與交件檔 |
| `repro/comp1_from_features.py` | `--out <資料夾>`、已存的特徵 | 重建的 `submission_player.csv` 與 SHA-256 比對結果 |

### 7.2 競賽二（69）

```
Q25  = softmax( e^a·(n/12)^γ·( m_C5 + w_OP·m_OP + w_O4·E_O4 + w_SWA·E_SWA + w_SWB·E_SWB + w_HS·E_HS + w_P1·E_P1E + w_SET·E_SET + Bβ ) )
Q64  = w_A·log_softmax(Q25) + w_O·OPP3          （w_A 0.7667、w_O 0.4941）
69   = Q64 → 池內同一玩家的安全合併 M8 → 規則 B（選期望得分最高的等級）
```

- `C5`：官方教學 CNN 多訓練後的逐盤輸出；`OP`：開局；`O4`：對手相關的計數特徵；`SWA`／`SWB`：另一組訓練配方的 CNN；`HS`：KataGo HumanSL 各棋力設定的逐手機率；`P1TO_E`：KataGo P1 特徵經 6 個小模型翻成等級；`SET`：把同一題的棋一起看的 set 模型（S25）。
- `OPP3`：對手項。用 c49 取每盤「對手那一方」的向量，和有等級標籤的訓練玩家比對，推估對手強度，再經條件表換成題目玩家的等級分布。
- `M8`：同一測試池內判定為同一玩家的題目才合併；最終只改了 private 3 題。

| 模組 | 輸入 | 輸出 |
|---|---|---|
| `tools/t02_checks.py`、`tools/root_props.py`、`tools/trailing_pass.py` | 官方測試檔 | 結構檢查報告；拿掉訓練集沒有的開頭欄位（HD）後的測試檔 |
| `colab/build_*_nb.py` → `colab/*.ipynb` | 訓練棋譜或測試檔（Drive） | CNN 權重、逐盤 logits |
| `tools/t02_rank.py`、`tools/t04_rank.py`、`tools/p04_opp.py` | 逐盤 logits、測試檔、O4 計數表 | 開局項、O4 項、04 層的機率檔 |
| `tools/t07_rank.py`、`tools/p07_hs.py` | 測試檔＋KataGo 權重 | HS 特徵與 07 層機率 |
| `tools/t08_rank.py`、`tools/p08_p1.py` | 測試檔＋KataGo 權重 | P1 特徵塊 `data/test08/p1/`、08 層機率 |
| `tools/t09_rank.py`、`tools/p09_map.py`、`tools/p09_forward.py` | P1 特徵、`p1e_map.npz` | P1TO_E 項、09 層機率 |
| `tools/t19_rank.py`、`tools/t25_rank.py`、`tools/p19_set.py`、`tools/p25_set.py` | P1 特徵、`s25_seed*.npz`、凍結設定 | SET 項、Q25 的 `rank_test_probs.csv`、`submission_rank.csv` |
| `tools/t59_rank.py`、`tools/t64_rank.py`、`results/prereg64/explore/p64_lib.py` | Q25 機率、c49 對手方向量、`c49_head.npz`、`prototypes.npz` | Q64 的機率檔與交件檔 |
| `tools/t69_rank.py`、`results/prereg69/explore/p69_lib.py` | `apply [--out 資料夾]` | 最終 `submission_rank.csv`、`t69_apply_report.json` |
| `tools/decide.py` | 每題 10 級的機率 | 等級（規則 A＝argmax、規則 B＝期望得分最高） |
| `tools/check_submission.py` | 交件檔＋題目檔（＋候選檔） | 格式檢查，結束碼 0／1 |
| `repro/comp2_from_features.py` | `--out <資料夾>`、已存的特徵 | 重建的 `submission_rank.csv` 與 SHA-256 比對結果 |

測試日工具是一條鏈（t02 → t04 → t07 → t08 → t09 → t19 → t25 → t59 → t64 → t69）：每層綁定前一層報告的 SHA-256，並有截止時間、git 乾淨等檢查，所以正式模式在比賽結束後不能直接重跑；重現請用 `repro/` 的入口。

## 8. 事前登記制度

每個方法在看到任何分數之前先寫登記並 commit：要比什麼、門檻多少、怎樣算成功。

- 正式登記在 `docs/PREREG_*.md`；探索輪的登記是 `results/prereg*/explore/PLAN.md`，判定結果在同資料夾的 `eval*.json` 與 `RESULTS.md`。
- 設定凍結並 commit 之後，在判定用的資料上**只判一次**；沒過門檻的方法不採用，結果照樣留在 repo（`results/README.md` 有每個資料夾的結論）。
- 本機驗證集按玩家切分（split v1），每場最多打分 4 次，紀錄在 `results/score_ledger.csv`。驗證集不逐題手調，測試集預測不人工修改。
- 同一批判定資料重複使用、訓練用到驗證玩家的棋、看過官方分數才做的選擇，都寫在對應的登記與 `NOTES.md`，並在書面報告揭露。

## 9. 外部資源

| 資源 | 來源 | 授權 | 在本 repo |
|---|---|---|---|
| KataGo 程式（python 模組） | <https://github.com/lightvector/KataGo> commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1` | `third_party/katago/LICENSE`（MIT 式）、`python/katago/train/LICENSE_AND_AUTHORS` | `third_party/katago/`：15 個上游檔原樣複製、未修改，逐檔雜湊在 `SOURCE.txt` |
| KataGo HumanSL 權重 | 見 5.1 | <https://katagotraining.org/network_license/> | 不含 |
| 官方教學程式 | <https://github.com/AILAB-NDHU/AICup-2026-Tutorial> commit `359e06fcaf811f9e87179f12f71425dd18dbabd1` | 其 README：參賽者可自由使用、修改與散布 | `third_party/aicup2026_tutorial/` 原樣副本；本隊的改動在 `colab/` |
| 官方棋譜 | <https://github.com/AILAB-NDHU/aicup2026-go-dataset> | 僅供本競賽使用 | 不含 |
| OGS 公開棋譜 | <https://za3k.com/ogs/> | 該頁註明經 OGS 同意散布 | 不含；只在競賽二 prereg34 試過，結果為負、未採用 |
| Python 套件 | `repro/requirements_*.txt` | 各套件自己的授權 | 不含 |

## 10. 生成式 AI 使用聲明

- 本 repo 的程式、測試、事前登記、結果紀錄與文件由 **Claude Code**（Anthropic）產生，並由它在本機與 Colab 執行。
- **OpenAI Codex**、**Google Antigravity（Gemini）**、**GitHub Copilot** 用於獨立審查、稽核與提出方法建議；其中被採用的建議（例如競賽二的 set 模型、對手項的作法、競賽一新題的計分後端與集成配方）在對應的登記與 `NOTES.md` 註明來源。
- 參賽者負責決定參賽與目標、同意各項資源的使用、確認最終版本；方法的選擇自 2026-10-02 起交由 AI 工具判斷。
- 細節見兩場的書面報告。

## 11. 授權

`【授權待定】`

`third_party/` 內的檔案依各自的授權（見第 9 節），不在本 repo 的授權範圍內。
