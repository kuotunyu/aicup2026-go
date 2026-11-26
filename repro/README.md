# repro/：兩場最終交件的可重現交付包（第 1 層已實作並驗收）

AI CUP 2026 圍棋，競賽一（辨識玩家，TEAM_11249）與競賽二（預測棋力，TEAM_11248）。
本資料夾只新增檔案，不改 repo 內任何既有程式、登記或結果。數字都是 2026-11-25 在本機實測，或引自 repo 內已 commit 的紀錄（各處註明來源）。

| 最終交件 | 檔案 | SHA-256 |
|---|---|---|
| 競賽一 v2-9 | `results/prereg67/apply/submission_player.csv`（1,600 列） | `36b222d866267abffda03d126d94db9ed5bea3406b2f8b47323622fd5329a026` |
| 競賽二 69 | `results/test69/rank/submission_rank.csv`（800 列） | `c7595f2eb43c185973c55c7b4b03875de41c5ce92327f7042fc3769dbbae36c8` |

## 0. 三層的意思

| 層 | 從哪裡到哪裡 | 硬體 | 本資料夾的狀態 |
|---|---|---|---|
| 第 1 層 | 已存的中間特徵（CNN 向量、KataGo P1 特徵、各模型輸出）＋官方測試 CSV → 最終 CSV | CPU | **已實作，兩條都逐位元組 PASS**（第 2 節） |
| 第 2 層 | 官方測試 SGF → 中間特徵（CNN 向量、KataGo P1／HumanSL、官方 CNN 的逐盤輸出） | GPU | 指令與相依寫在第 3 節；抽樣比對做了 4 項（第 3.4 節）；沒有單一入口 |
| 第 3 層 | 訓練（各模型的權重、LDA、融合參數） | GPU／CPU | 只列程式與指令（第 4 節），沒有重跑 |

## 1. 環境

- 本機：Windows 11、Git Bash、Python 3.12.13。CPU 環境 `.venv`（`requirements_cpu.txt`），GPU 環境 `.venv-gpu`（`requirements_gpu.txt`，torch 2.11.0+cu128，RTX 4090）。兩份清單是 11/25 從兩個環境的套件 metadata 列出的實際版本（這兩個 venv 是 uv 建的、沒有 pip，`pip freeze` 會是空的）。
- 第 1 層在 import 階段只用到 numpy 和 scipy（實測 `sys.modules`）；不 import torch。
- **版本要完全相同**：競賽二的凍結工具會比對 python 3.12.13、numpy 2.5.3、scipy 1.18.1（`tools/t25_rank.py` `check_venv`，對 `results/prereg25/rank/frozen_config.json`），不同就以結束碼 4 停下。
- 不要設 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`（競賽二的工具會拒絕；位元組接縫用預設執行緒）。
- 安裝（新機器；本機已裝好，這段沒有在乾淨機器上實測）：

```
uv venv --python 3.12.13 .venv
uv pip install --python .venv -r repro/requirements_cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv venv --python 3.12.13 .venv-gpu
uv pip install --python .venv-gpu -r repro/requirements_gpu.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

- 根目錄：兩支入口都用 `--root`，沒給就讀環境變數 `C10_ROOT`，再沒有就用 `repro/` 的上一層。既有模組一律用自己檔案的位置推 repo 根目錄、路徑用 `os.path.join`，所以 `data/`、`results/` 要照原本的相對位置放在根目錄下。
- 輸出資料夾 `--out` **必須在 repo 之外**（凍結工具的演練模式規定）；不給就開一個新的暫存資料夾。

## 2. 第 1 層（CPU）：單一入口與驗收

在 repo 根目錄、Git Bash：

```
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe repro/comp1_from_features.py --out <repo 外的資料夾>
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe repro/comp2_from_features.py --out <repo 外的資料夾>
```

Linux／Colab 把 `.venv/Scripts/python.exe` 換成該環境的 `bin/python`，逐位元組接縫過不了時加 `--tolerant`（第 2.4 節；Colab 用 `colab_layer1.ipynb`，見第 5 節）。
兩支都可加 `--list-inputs <檔.json>`：把這次執行實際讀過的 repo 內檔案（路徑、bytes、SHA-256）寫出來；本資料夾的 `inputs_comp1.json`、`inputs_comp2.json` 就是 11/25 這樣產生的。

### 2.1 驗收結果（2026-11-25，本機）

| 入口 | 結果 | 耗時（實測） | 既有紀錄 |
|---|---|---|---|
| `comp1_from_features.py` | **PASS**：`submission_player.csv` SHA-256 `36b222d8…a026`；兩池 `top5.csv` 也和 `results/prereg67/apply/<池>/top5.csv` 相同（public `f62c8986…b269`、private `2a82f4e4…0343`） | 共 243.9 秒（public 158.9、private 82.2；含 `--list-inputs`） | 11/22 原執行 public 56.5 秒、private 74.6 秒，峰值記憶體 1,999.4／1,934.9 MB（`results/prereg67/apply/*/apply_report.json`） |
| `comp2_from_features.py` | **PASS**：`submission_rank.csv` SHA-256 `c7595f2e…36c8` | 第一次 24.1 秒、第二次 9.8 秒 | 11/22 原執行 15.1 秒（`results/test69/rank/t69_apply_report.json`） |

- 兩次執行的讀檔紀錄裡 `data/local` 都是 0 個檔（`inputs_comp*.json` 的 `data_local_opened_not_hashed` 為空）。
- 執行後 `git status` 只多了 `repro/`。

### 2.2 兩支入口各做什麼（都只呼叫既有模組，不重寫演算法）

**`comp1_from_features.py`** → `results/prereg67/explore/p67_eval.py`：
1. `apply --pool public`、`apply --pool private`：讀官方 v2 測試檔（新題 `candidates2`／`test2`），從 SGF 重算手工項 G；載入已存的 P1 特徵、7 個成員的 CNN 向量；重學 L 玩家的 P1 LDA（雜湊要等於 `results/prereg43/explore/tune.json`）；**接縫**：先重建 v2-8 的 `top5.csv`，要和 `results/prereg66/apply/<池>/top5.csv` 相同，才套 v2-9。
2. `assemble`：原題 800 列取自已存的 v1 06-ON 交件檔 `results/test06/player/on/submission_player.csv`（SHA-256 `94cc8265…2cbf`，`assemble_report.json` `v1_sha256`），後面接新題 800 列。
3. 比 SHA-256。

**競賽一原題 800 列（06-ON）沒有重算**：登記的 assemble 步驟本來就是沿用 v1 的輸出（PREREG_27 rev 3）。06 的測試日工具 `tools/t06_player.py` 的 `seam` 步驟會讀驗證集答案資料夾裡的兩個檔，而且正式模式已過截止時間不再產生，所以這裡不跑；路徑寫在第 3.3 節。

**`comp2_from_features.py`** → `tools/t69_rank.py apply --out <資料夾>`（演練模式）：
1. 經 t64／t59／t25 的函式，從已存的逐盤來源重算 Q25 的 7 項；**接縫 A**：Q25 機率檔和 `results/test25/rank/rank_test_probs.csv` 逐位元組相同。
2. 用已存的 c49 對手方向量算 OPP3（接縫 O）、玩家方向量（接縫 M）；**接縫 T**：合併關閉時和 `results/test64/rank/` 的機率檔、交件檔逐位元組相同。
3. 每池做 M8、規則 B、`check_submission --official`。
4. 比 SHA-256。

**唯一的偏離**：`tools/t64_rank.py` 的 `K6_64`（登記的截止 2026-11-24 23:00，過了就拒絕產生）在這支入口的程序內被改成很久以後——重現一定發生在截止之後。沒有改任何檔案；其他檢查（釘住的 SHA、接縫、版本、BLAS 變數）都照跑。

### 2.3 輸入檔

完整清單（每個檔的路徑、bytes、SHA-256）：`inputs_comp1.json`（922 個檔、4,759,129,811 bytes）、`inputs_comp2.json`（169 個檔、231,411,895 bytes）。不含 `.py` 程式檔。下面是分組摘要。

**競賽一**

| 資料夾 | 檔數 | bytes | 內容 | 在 git？ |
|---|---|---|---|---|
| `data/test_v2_download/x/tests/` | 8 | 142,700,932 | 官方 v2 測試檔（新題 4 檔算分；原題 4 檔只被讀來核對） | 否（官方棋譜） |
| `data/prereg06/feats/`（含 `resp/`） | 11 | 1,355,001,420 | 06 的訓練集特徵庫（G 項的 LDA、開局詞表） | 否 |
| `data/prereg39/p1/{pub2,priv2}/` | 72 | 134,980,715 | 兩池的 KataGo P1 特徵塊 | 否 |
| `data/prereg39/p1/L/`、`data/prereg10/mock/` | 782 | 1,505,835,049 | L 玩家的 P1 特徵（學 LDA 用；黑棋部分沿用 PREREG_10 的塊） | 否 |
| `data/prereg39/lists/` | 3 | 4,665,824 | P1 的棋局名單 | 否 |
| 7 個成員的向量資料夾（`data/prereg49/emb/…`、`prereg55/emb_S/…`、`prereg51/emb_B/…`、`prereg61/emb/…`、`prereg62/{S1,S2,X}/emb/…`） | 21 | 1,507,225,824 | 每成員 `pool_public.npz`、`pool_private.npz`、`store_E.npz`（`store_E` 只被拿來核對 SHA，不進計算） | 否 |
| `data/prereg43/emb_S0/…` | 2 | 14,603,648 | v2-6 的兩池向量（基底檢查） | 否 |
| `data/prereg49/S0/ckpt/step_320000.pt` | 1 | 69,578,397 | c49 權重（核對 checkpoint SHA） | 否 |
| `data/prereg63/p1multi.npz`、`data/prereg66/sp128.npz` | 2 | 1,609,704 | P1 多投影、sp128 投影 | 否 |
| `data/prereg25/models/` | 3 | 4,967,337 | 競賽二 S25（段位相容項） | 否（同檔在 `results/prereg25/rank/`，在 git） |
| `results/` 內 | 17 | 17,960,961 | `p1e_map.npz`、`c49_head.npz`、各版 `eval.json`／`tune.json`／`frozen.json`、v2-8 的 `top5.csv`、v1 06-ON 交件檔 | 是 |

新題 4 個官方檔的 SHA-256：`candidates2_public` `ac3d7d20…616d`、`candidates2_private` `8e80f946…6812`、`test2_public` `33485b58…3d6a`、`test2_private` `52e01bcc…a527`（全碼在 `inputs_comp1.json`）。

**競賽二**

| 資料夾 | 檔數 | bytes | 內容 | 在 git？ |
|---|---|---|---|---|
| `data/test_v2_download/rank/`、`data/test/` | 4 | 27,005,512 | 官方題目檔（原始檔，和拿掉 HD 的檢查後副本） | 否（官方棋譜） |
| `data/prereg56/src_rank/` | 4 | 13,527,828 | 題目檔改名成向量程式吃的檔名＋只有表頭的候選檔 | 否 |
| `data/prereg56/emb_opp_rank/…`、`emb_rank/…` | 4 | 20,572,096 | c49 對手方／玩家方向量 | 否 |
| `data/test08/p1/` | 52 | 99,630,832 | 測試題的 KataGo P1 特徵塊 | 否 |
| `results/test02`…`test25`、`test64`、`test69` | 59 | 27,874,601 | 11/04 起各層測試日工具的正式輸出（C5、SW 逐盤 logits、開局、O4、HS、P1 彙整、各層機率檔與報告） | 是 |
| `results/prereg04`…`prereg69` | 42 | 42,787,387 | 凍結設定與模型檔（`p1e_map.npz`、`s19_*`／`s25_*`、`c49_head.npz`、`prototypes.npz`、`kernel.json`、`merge_sim.json`…） | 是 |

官方題目檔 SHA-256：public `4691803a…4fef`、private `d59865e2…3c73`。

### 2.4 容許模式 `--tolerant`（給 Linux／Colab；2026-11-25 加）

**為什麼**：11/25 在 Colab（Linux、Python 3.12.13、numpy 2.5.3、scipy 1.18.1、Xeon 8 核）實跑，兩支入口都在接縫停下、沒有產生最終 CSV：

- 競賽二：`seam A (Q25 recomputed vs test25): byte equal False` → `DRIFT (exit 4)`。同一行紀錄是 `max_abs_diff` 7.77e-16、`tol` 1e-12、`equal` true、`rule_a_differences` 0、`rule_b_differences` 0：只有浮點數最後一位不同，判定完全相同，但 `tools/t69_rank.py` 另外要求逐位元組相同。
- 競賽一：BLAS 已限 1 執行緒，仍在 `the L LDA refit differs from prereg43 tune.json` 停下（重學的 LDA 陣列的 SHA-256 要等於已存的雜湊）。

**做法**：兩支入口加 `--tolerant`。只在入口的程序內（monkeypatch，**不改 `repro/` 以外任何檔**）把下表的精確比較換掉，並把每個接縫**實際的差異**寫進 `<out>/tolerant_report.json`（工具以錯誤停下時也會寫）。程式在 `_tolerant.py`。

| 鏈 | 接縫（原本的要求） | 容許模式怎麼比 | 容許值 |
|---|---|---|---|
| 競賽二 | 接縫 A：`t25.rows_compare`，Q25 機率 vs `results/test25/rank/rank_test_probs.csv`（工具自己的 `equal`＝差 ≤ 1e-12 且規則 A／B 等級相同，**另外**要求 `byte_equal`） | 只要求工具自己的 `equal` | 工具原有的 1e-12（PREREG_25 4.2） |
| 競賽二 | 接縫 T（機率）：合併關閉的 Q64 機率檔 vs `results/test64/rank/rank_test_probs.csv`（逐位元組） | 逐格比：文字相同，或兩邊都是數字且在容許值內 | 絕對差 ≤ 1e-9 + 1e-9 × 參考值的絕對值 |
| 競賽二 | 接縫 T（交件）：合併關閉的規則 B 預測 vs `results/test64/rank/submission_rank.csv`（逐位元組） | 離散接縫：記錄不同的列數與前 10 列，繼續跑 | — |
| 競賽一 | L 玩家 P1 LDA 重學（joint／B／W 三個模型的 `m`、`s`、`mu`、`sd`、`W` 的 SHA-256 要等於 `results/prereg43/explore/tune.json` 的 `lda.sha256`；每池一次） | 和 `ref_lda_L.npz`（Windows 上擬合的同一組陣列，SHA-256 就是 tune.json 記的那個）逐陣列比；另記 `W` 每欄對齊正負號後的差（特徵向量的正負號本來就不定），但容許值套在原樣的 `W` 上 | 絕對差 ≤ 1e-9 + 1e-9 × 參考值的絕對值 |
| 競賽一 | v2-8 的 `top5.csv` 重建 vs `results/prereg66/apply/<池>/top5.csv`（每池一次，逐位元組） | 離散接縫：記錄不同的列數與前 10 列，繼續跑 | — |

- **數值接縫超過容許值仍然停下**（工具照原本的訊息以非 0 結束，報告裡該接縫 `within_tolerance` 為 false）。
- **離散接縫**（比的是預測檔，不是浮點數）不設容許值：有不同的列就記下來、繼續跑到最終 CSV，好讓最終檔能逐列比較。這不會把不同變成 PASS。
- **PASS 的判定不變**：只看最終 CSV 的 SHA-256 是否等於已上傳檔。不同時（兩種模式都是）印出不同的列數與前 10 列；容許模式另寫進 `tolerant_report.json` 的 `final_csv`。
- **不加 `--tolerant` 時行為不變**：`_tolerant.py` 不會被 import。
- 沒放寬、照原樣跑的檢查：所有檔案 SHA 釘選、版本（`check_venv`）、BLAS 環境變數、接縫 O／M（整數與結構檢查）、RSS 看門狗，以及同一程序內兩條計算路徑互比的檢查（`port_equal`、`tools/t69_rank.py` 的 `np.array_equal(rf.softmax(L0), P0)`、競賽一 `Carbon's P1 loader` 的 `array_equal`）。後面這幾個比的是同一次執行算出的兩個值，不涉及跨平台；如果在 Linux 上仍然擋住，會以原訊息停下，要再補。
- `ref_lda_L.npz`（1,260,284 bytes）由 `make_tolerant_refs.py` 在 Windows 產生：呼叫既有的 `p39_eval.load_set("L")`／`fit_models`，確認 `models_sha` 等於 tune.json 的 `ccc95351…419f` 才寫檔。

`tolerant_report.json` 的欄位：每個接縫一筆（`seam`、`context`（競賽一是哪一池）、`kind` numeric／discrete、`strict_equal`、`within_tolerance`、`relaxed`、`max_abs_diff`、`max_rel_diff`、`rows_different`、`first_differences`…），加總欄 `seams_checked`、`seams_relaxed`、`numeric_all_within_tolerance`、`numeric_max_abs_diff`、`discrete_rows_different`，和 `final_csv`（`sha256`、`pass`、不同列數）。

**驗收（2026-11-25 深夜，Windows，本機 `.venv`）**

| 執行 | 結果 |
|---|---|
| 兩支入口不加旗標 | 都 **PASS**（競賽二 `c7595f2e…36c8` 28.7 秒、競賽一 `36b222d8…a026` 138.8 秒），和加旗標之前相同 |
| 兩支入口加 `--tolerant` | 都 **PASS**（18.5 秒、148.1 秒）。競賽二的報告：接縫 A `strict_equal` true、最大差 0、放寬 0 個（這次執行時接縫 T 相同的情形還不會記進報告，之後已改成也記）。競賽一的報告：記了 6 筆（每池的 LDA、v2-8 top5、v2-9 top5），全部 `strict_equal`、放寬 0 個、不同列 0 |
| 模擬漂移（不在 repo 的測試腳本，在程序內擾動中間值）：競賽二把接縫 A 的紀錄改成 Colab 看到的 `byte_equal` false／7.77e-16，並把接縫 T 的機率乘 (1+1e-13) | 不加旗標：停在接縫 A（和 Colab 相同的訊息）。加 `--tolerant`：接縫 A、接縫 T 機率各放寬 1 個（最大絕對差 7.77e-16、9.99e-14），接縫 T 交件 0 列不同，最終 CSV **PASS** |
| 同上但接縫 T 的機率乘 (1+1e-6) | 加 `--tolerant` 仍停在接縫 T（`BEYOND TOLERANCE`，最大絕對差 1e-6），沒有最終 CSV，報告有寫出 |
| 模擬漂移：競賽一把 LDA 的 `W` 乘 (1+1e-13) | 不加旗標：以 `the L LDA refit differs from prereg43 tune.json` 停下（和 Colab 相同的訊息）。加 `--tolerant` 的那一次**沒跑完**（共用電腦夜間排程，程序在走到 LDA 之前被停掉）→ 這條路徑 11/26 在 Colab 實際走過（見第 5 節開頭），PASS |

**Linux 上還沒有實測容許模式**：本機 WSL 模擬（`wsl_drill.sh`：clone bundle、解壓特徵 zip、uv 建 3.12.13，兩支入口各跑不加旗標與 `--tolerant`）排在 11/26 07:50 之後（共用電腦排程），Colab 也要重跑。所以「Linux 上各接縫實際差多少、容許模式下後面還有沒有別的檢查會擋、最終 CSV 是否相同」當時還不知道。**11/26 Colab 實測結果見第 5 節開頭**（WSL 模擬沒有跑）。

## 3. 第 2 層（GPU）：從官方測試 SGF 抽特徵

沒有單一入口；下面是實際用過的指令（出處是當時的腳本與 log）。`GPY`＝`.venv-gpu/Scripts/python.exe`，都在 repo 根目錄執行，開跑前 `nvidia-smi` 確認沒有別的工作。

### 3.1 競賽一新題

| 中間檔 | 指令 | 權重 | 耗時紀錄（4090） |
|---|---|---|---|
| 7 成員的兩池向量 | 17 平面：`GPY -u results/prereg49/explore/p49_run.py embed --ckpt <ckpt> --weights raw --tta 1 --workers 4 --emb-root data/prereg49/emb --source pool --src data/test_v2_download/x/tests --pool public\|private`；S 用 `p55_run.py embed --arm S … --emb-root data/prereg55/emb_S`；B 用 `p51_run.py embed --arm B --size B15C128 … --emb-root data/prereg51/emb_B`。18 平面：`p61_run.py embed --ckpt <ckpt> --workers 3 --source pool …`；`p62_run.py embed --arm S1\|S2\|X --ckpt <ckpt> --workers 2 --source pool …` | 7 個 ckpt（路徑、SHA 在 `report/facts/shared.md` 7.4） | 每池秒數 public／private：c49 55.3／62.2、S 52.8／41.9、B 172.1／152.9、p61 71.9／47.9、S1 67.9／47.7、S2 43.7／43.9、X 65.4／43.2（各 `data/prereg*/embed*_log.txt`） |
| 兩池 P1 特徵 | `bash results/prereg39/explore/p39_extract.sh --sets pub2,priv2 --workers 4`（名單 `results/prereg39/explore/p39_lists.py`；官方棋先用 `tools/root_props.py` 的函式拿掉 HD） | `data/katago/b18c384nbt-humanv0.ckpt`（KataGo v1.15.0 release asset，323,752,318 bytes，SHA-256 `84b23476…8b27`；不轉發） | pub2 7,100 盤 1,216.9 秒、priv2 7,100 盤 1,231.9 秒（各塊秒數加總，`data/prereg39/p1/run_log.jsonl`） |
| G 項 | 不用 GPU；第 1 層的 apply 直接從 SGF 算 | — | 含在第 1 層 |

L 玩家的 P1（學 LDA 用，32,888 盤 5,596.6 秒）和 `store_E.npz` 來自訓練棋譜，屬於第 3 層。

### 3.2 競賽二

| 中間檔 | 指令／程式 | 硬體 | 耗時紀錄 |
|---|---|---|---|
| 檢查、拿掉 HD | `tools/t02_test_bundle.py`、`tools/t02_checks.py`、`tools/root_props.py`（`docs/TESTDAY.md`、`TESTDAY_22_ROOTPROPS.md`） | CPU | 沒查 |
| C5 逐盤 logits（`results/test02/rank/<池>/dumps/`） | `colab/test02_rank.ipynb`（產生器 `colab/build_test02_rank_nb.py`）；權重 `epoch_9.pth` | Colab G4，執行階段 2026.07（torch cu128） | 沒查 |
| SWA／SWB（`results/test04/sw/`；融合權重 0，但工具鏈會讀） | `colab/test04_rank.ipynb` | Colab G4 | 沒查 |
| 開局、O4 | `tools/t02_rank.py`、`tools/t04_rank.py`（`tools/p04_opp.py`；計數表 `data/prereg04/counts_fit.npz`） | CPU | 沒查 |
| HS（`results/test07/hs/`；融合權重 0，但工具鏈會讀） | `tools/t07_rank.py`（`docs/TESTDAY_07.md`） | 4090 | 每個設定約 170–190 秒（`results/test07/hs/hs_*_report.json`），總和沒算 |
| P1（`data/test08/p1/`、`results/test08/p1/`） | `tools/t08_rank.py p1 …`（`docs/TESTDAY_08.md`） | 4090 | public 1,007.4、private 855.2 GPU 秒（`results/test08/p1/p1_*_report.json`） |
| c49 對手方向量 | `GPY -u results/prereg56/explore/p56_opp.py embed --ckpt data/prereg49/S0/ckpt/step_320000.pt --weights raw --tta 1 --workers 2 --emb-root data/prereg56/emb_opp_rank --source pool --src data/prereg56/src_rank --pool public\|private` | 4090 | 每池約 35 秒（`data/prereg56/embed_opp_log.txt` 起訖時間） |
| c49 玩家方向量 | 同上但用 `results/prereg49/explore/p49_run.py embed … --emb-root data/prereg56/emb_rank` | 4090 | 120 秒／66 秒（`data/prereg56/embed_log.txt` 起訖時間；當時和另一個訓練同時跑） |

- `data/prereg56/src_rank/`：題目檔和官方 `rank_prediction_test_{public,private}.csv` SHA-256 相同、改成 `player_identification_test2_<池>.csv` 的檔名，再加只有表頭 `player_id,game_id,sgf_content,color` 的候選檔（`docs/PREREG_59_OPPONENT_IDENTITY.md` 第 48 行）。**建立這個資料夾的腳本不在 repo 裡**。
- 各層測試日工具（t02→t04→t07→t08→t09→t19→t25→t59→t64→t69）是正式模式的鏈：每層綁前一層報告的 SHA，有截止時間、git 乾淨、驗證集接縫等檢查。第 1 層入口吃的是這條鏈**已存的輸出**；要從 SGF 整條重跑，需要另寫一支不經這些檢查的串接入口（還沒做）。

### 3.3 競賽一原題 800 列（06-ON）

CNN（E0）逐盤輸出：`colab/test02_player.ipynb`（Colab G4；權重 `best_model.pth`）→ `tools/t02_player.py` → `tools/t06_player.py feats／seam／apply／submit`（CPU 約 15 分鐘，`AGENTS.md` 測試日表）。`seam` 會讀驗證集的兩個檔，交付時不能附；這一段目前只能靠已存的 `results/test06/player/on/submission_player.csv`。

### 3.4 抽樣比對（2026-11-25，4090；GPU 合計約 2 分鐘；開跑前 GPU 上沒有其他 python 程序）

| 項目 | 範圍 | 結果 |
|---|---|---|
| c49 向量，競賽一新題 public | 前 500 盤（53,703 個局面），42.4 秒 | 取樣數全部相同；和已存 `pool_public.npz` 的最大絕對差 **7.63e-06**（不是逐位元組相同） |
| c49 對手方向量，競賽二 public | 前 500 盤，9.1 秒 | 取樣數相同；最大絕對差 7.63e-06 |
| c49 玩家方向量，競賽二 public | 前 500 盤，9.2 秒 | 取樣數相同；最大絕對差 5.72e-06 |
| KataGo P1，競賽一 pub2 | 前 20 盤（4,688 個局面），30.5 秒（整個程序 69.6 秒） | `means`、`pos_logp`、`pos_entropy`、各計數欄**逐位元組相同** |

- 用的是既有程式的 `--limit 500 --compare <已存檔>`（輸出寫在 repo 外）和 `p39_extract.py run --smoke … --max-games 20`（輸出在 `data/prereg39/tmp/` 下，比完已刪）。
- CNN 向量在同一台機器重抽也不是逐位元（fp16；這次只抽 500 盤，批次組成和原本不同）。向量有 1e-5 量級的差時，最終 CSV 是否仍逐位元組相同**沒有驗證**——所以「逐位元組重現」只對第 1 層成立；第 2 層的驗收要用「容許差＋最終 CSV 有幾列不同」。CNN 向量沒有查到事前登記的容許差；P1 的登記容許差是 0.05（PREREG_10／13 的 tol_P）。
- 18 平面成員（p61、S1、S2、X）、B、S，以及競賽二的 C5／SW／HS 沒有抽樣（18 平面的 `--limit` 固定寫進 `data/prereg60/seam_gpu`；C5／SW 在 Colab）。

## 4. 第 3 層：訓練（只列程式，沒有重跑）

詳細設定、硬體與秒數在 `report/facts/comp1_final.md` 第 4.1、8.2 節和 `report/facts/comp2_final.md` 第 7.2 節；權重 SHA 在 `report/facts/shared.md` 7.4。

| 產物 | 程式 | 硬體與時間（紀錄） |
|---|---|---|
| c49 | `GPY -u results/prereg49/explore/p49_run.py train --workers 4 --auto-resume --log-every 1000`（`data/prereg49/chain_p49b.sh`） | 4090，107,434 秒 |
| S、B、p61、S1、S2、X | `results/prereg55/explore/p55_run.py`、`prereg51/…/p51_run.py`、`prereg61/…/p61_run.py`、`prereg62/…/p62_run.py`；Colab notebook 產生器 `p55_colab.py`、`p51_colab.py`、`p61_colab_nb.py`、`p62_colab.py` | Colab G4，各 11–27 小時（train_record） |
| P1 的 L-LDA、p1multi | `results/prereg39/explore/p39_eval.py`（apply 時現場重學 L-LDA）；`results/prereg45/explore/p45_extract.py`、`results/prereg63/explore/p63_eval.py fit` | 4090（P1 抽取）＋CPU |
| sp128 | 陣列取自顧問檔 `data/advice/gemini_1121b/`，repo 內沒有自己的擬合程式（`results/prereg66/explore/NOTES.md` 4） | — |
| 06 的特徵庫、LDA、E0 CNN | `tools/p06_player.py`、`tools/p06_cnn_fit.py`；`colab/prereg01_player.ipynb`、`prereg02_player.ipynb` | Colab G4＋CPU |
| 競賽二 C5、SWA／SWB | `colab/prereg01_rank.ipynb`、`prereg02_rank.ipynb`、`prereg03_rank_{a,b}.ipynb` | Colab G4 |
| 開局、O4 計數 | `tools/p02_rank_opening.py`、`tools/p04_opp.py count` | CPU |
| P1 probe、P1TO_E、S25 | `results/prereg08/p1/p1_extract.py`、`tools/p09_map.py train`、`results/prereg25/explore/p25_extract.py`／`p25_prep.py`／`p25_train.py` | 4090＋CPU |
| Q25、Q64 融合參數；c49 頭、原型、條件表 | `tools/p25_set.py freeze`、`results/prereg64/explore/p64_freeze.py`、`p64_head.py`、`p64_rows.py`、`p64_eval.py kernel` | CPU（原型向量 4090） |

除 c49 那一行外，上表沒有逐條核對完整的命令列參數；交付前要從各登記的 PLAN／腳本補齊。

## 5. Colab（第 1 層）

2026-11-26 Colab 實測（公開 notebook，clone 本 repo commit `7df09c0`；CPU 執行階段，Python 3.12.13、NumPy 2.5.3、SciPy 1.18.1；notebook 少了一個 `sha256` 小函式，執行中補上，已改進本 repo）：兩場的嚴格模式都在接縫停下、沒有最終 CSV；加 `--tolerant` 後**兩場的最終 CSV 都和上傳檔 SHA-256 相同（PASS）**。競賽二放寬 2 個數值接縫，最大絕對差 1.89e-15，規則 A／B 與交件列 0 列不同；競賽一放寬 2 個（兩池的 L-LDA 重擬合，和 `ref_lda_L.npz` 的最大絕對差 5.10e-12），top5 接縫 0 列不同。耗時：接回分塊 297 秒、stage_inputs 56 秒、建環境 12 秒、競賽二 21＋27 秒、競賽一 36＋84 秒，全程約 16.5 分鐘。

以下 5.1–5.3 是 11/25 用私有 git bundle 交付的舊流程，留作紀錄；公開 repo 請用 `repro/colab_public.ipynb`（第 5.6 節）。

第 2、3 層的盤點在 `colab_plan.md`。這一節只講第 1 層。

### 5.1 交付包（`data/delivery/`，不進 git；用 `repro/make_delivery.py` 產生）

| 檔 | 內容 | 大小（bytes） |
|---|---|---|
| `C10_code.bundle` | `git bundle`（分支 `report-final` 的 HEAD 與歷史）：程式、登記、git 內的結果檔。**notebook 用這個** | 見 `C10_code_bundle_manifest.txt` |
| `C10_code.zip` | `git -c core.autocrlf=false archive --format=zip HEAD`：同一個 commit 的檔案（git 原始位元組，LF），給人看程式用；**不能直接拿來跑**（見 5.3） | 見 `C10_code_manifest.txt` |
| `C10_features_layer1_comp1.zip` | `inputs_comp1.json` 裡不在 git 的 905 個檔（都在 `data/` 下），ZIP_STORED | 4,741,325,254（SHA-256 `2e06da0276aa2e9f3ace09d46b64bcdd0d99dba5c86c2438db420ec440c76159`） |
| `C10_features_layer1_comp2.zip` | `inputs_comp2.json` 裡不在 git 的 64 個檔，ZIP_STORED | 160,747,756（SHA-256 `eb05d5808c5a6b70d5227d5e46b578fd32103e8ba7671bf71120158617228273`） |
| `C10_features_manifest.json` | 兩份清單聯集 1,089 個檔（4,973,038,015 bytes）的 bytes、SHA-256（交付時的位元組）和所在的檔 | — |
| `C10_code_manifest.txt`、`C10_code_bundle_manifest.txt` | commit、檔數、大小、SHA-256、檢查結果 | — |
| `C10_delivery_SHA256SUMS.txt` | 以上各檔的 SHA-256 與大小（notebook 用它核對上傳是否完整） | — |
| `colab_layer1.ipynb` | `repro/colab_layer1.ipynb` 的副本 | — |

- 兩份清單共 1,089 個不同的檔：969 個不在 git（放特徵 zip），120 個在 git（隨 bundle／code zip，用 git 的位元組）。其中 14 個 json／csv 在這台 Windows 的工作副本是 CRLF，`inputs_comp*.json` 記的是 CRLF 版的 SHA；交付的是 git 的 LF 版，manifest 兩個 SHA 都記（`windows_working_copy_sha256`）。
- 特徵 zip 分成兩個是因為合起來超過 4 GB；競賽一那個仍是 4.7 GB（ZIP64，Python `zipfile` 和 Info-ZIP 6 都能解）。
- 清單裡沒有任何 `data/local` 的檔（`make_delivery.py` 遇到就停）；notebook 解壓後也檢查樹裡沒有 `data/local`。
- 檢查過 `C10_code.zip`：1,907 個檔每個都和 HEAD 的 git blob 相同；沒有 `.sgf`、權重、`data/training`／`data/test*`／`data/prereg*`、帶 sgf 欄的 csv。分支歷史裡的所有文字 blob 也沒有連續 30 手以上的 SGF 片段。**注意**：`results/` 內有由棋譜逐盤算出的 npz／csv（逐盤輸出、逐題機率），本來就在 git 裡；特徵 zip 裡有官方測試 CSV（競賽一 v2 8 檔、競賽二 4 檔＋`data/prereg56/src_rank/` 的改名副本）。這些能不能交給主辦方以外的人，見 `colab_plan.md` 第 3 節。

### 5.2 步驟

1. 把下列 6 個檔上傳到 Drive 的同一個資料夾（預設 `MyDrive/C10/report_repro`）：`C10_code.bundle`、`C10_features_layer1_comp1.zip`、`C10_features_layer1_comp2.zip`、`C10_features_manifest.json`、`C10_code_bundle_manifest.txt`、`C10_delivery_SHA256SUMS.txt`。
2. 在 Colab 開 `colab_layer1.ipynb`（一般 CPU 執行階段，建議高 RAM），資料夾不同就改第 0 格的 `DRIVE_DIR`，全部執行。
3. notebook 依序做：掛載 Drive → 複製到本機磁碟並核 SHA-256 → `git clone` bundle 到 `/content/c10`（`core.autocrlf=false`）→ 兩個特徵 zip 解壓進同一棵樹 → 照 manifest 核 1,089 個檔 → 用 `uv` 建 Python 3.12.13 環境（在樹外 `/content/venv312`）並裝 `requirements_cpu.txt`（排除 `colorama`：它只是 Windows 上 pytest 的相依；清單裡沒有其他 Windows 專用套件。整份裝不起來時退回只裝 numpy、scipy、psutil 並印出說明）→ 跑 `comp2_from_features.py`、`comp1_from_features.py`：每支**先不加旗標跑一次**（輸出 `/content/out/<comp>`，記錄是否逐位元組 PASS）；若在接縫停下、沒有產生最終 CSV，**再加 `--tolerant` 跑一次**（輸出 `/content/out/<comp>_tolerant`，印出每個接縫的實際差異）→ 兩次都印 PASS／FAIL、SHA-256、停在哪裡；有產生 CSV 但 SHA 不同時印出和已上傳檔不同的列數與前 10 列 → 最後印 python／numpy／scipy 版本、CPU、各段耗時，並把紀錄（summary 內含兩次的結果與 `tolerant_report.json` 的內容、兩種模式的 log）和產生的 CSV 存回 Drive 的 `colab_run_<時間>/`。
4. 啟動入口時，子程序的環境拿掉 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`（入口的要求）和 `PYTHONPATH`／`PYTHONHOME`（不讓 Colab 內建 Python 的套件混進來），不另外設執行緒變數。

預期耗時：見本節開頭的 11/26 實測（本機參考：競賽二約 25 秒、競賽一約 135 秒；Colab 另加 Drive 複製約 5.4 GB、核 SHA、建環境）。
Colab 上的結果：11/25 第一次實跑（還沒有 `--tolerant`）兩支都停在接縫、沒有最終 CSV（第 2.4 節開頭）；加了 `--tolerant` 的公開 notebook 11/26 實測見本節開頭。

### 5.3 為了能在 Linux／乾淨機器上跑所做的事

- **程式用 git bundle 交付，不用 zip**：競賽二的凍結工具（`tools/t25_rank.py` 的 `require_committed`）要求輸入檔在 git 裡已 commit 且沒改過。11/25 實測：把 `C10_code.zip`＋特徵 zip 解壓到沒有 `.git` 的資料夾，`comp2_from_features.py` 以 `REFUSED (exit 2): (t25_rank) results/test19: commit first` 停下。改成 clone bundle 後通過。bundle 帶完整歷史（一千多個 commit），因為有些工具會查檔案的 commit 時間。
- **`_common.blas_one_thread_posix()`（新增）**：競賽一的既有程式在 Windows 會在程序內把 numpy 的 OpenBLAS 限成 1 執行緒（`p38_backend.polite` → `blas_threads(1)`，只找 Windows 的 DLL，其他系統是空動作）。`comp1_from_features.py` 在非 Windows 上改由這個函式對 `numpy.libs` 的 `.so` 做同一個呼叫，讓執行緒數和原執行一致；Windows 上不做任何事。**Linux 上沒實測**。
- 兩支入口其他地方本來就用 `os.path.join`／`--root`，沒有寫死 Windows 路徑；`repro/` 以外的檔都沒改。

### 5.4 本機乾淨樹演練（2026-11-25，Windows，`.venv`）

照 notebook 的順序在 repo 外的空資料夾做：`git -c core.autocrlf=false clone C10_code.bundle`（commit `48cc26d`，文字檔是 LF，和 Linux checkout 相同）→ 解壓兩個特徵 zip → manifest 1,089 個檔 0 個不符、樹裡沒有 `data/local` → 用主 repo 的 `.venv` 跑兩支入口（`--root` 指向該樹）：

| 入口 | 結果 | 耗時 |
|---|---|---|
| `comp2_from_features.py` | **PASS**，`c7595f2e…36c8`；接縫 A、T 逐位元組相同 | 25.7 秒 |
| `comp1_from_features.py` | **PASS**，`36b222d8…a026` | 134.9 秒 |

所以交付包的內容是完整的，而且文字檔用 LF 版也會過（先前 `colab_plan.md` 第 2.1 節列的疑慮）。跑完後該樹 `git status` 乾淨。演練資料夾已刪。

### 5.5 Colab 上還可能出問題的地方（都沒實測）

1. **BLAS 不同**：Linux 的 numpy／scipy 輪子用的 OpenBLAS 建置和 Windows 不同，Colab 的 CPU 核心數也不同（競賽二用預設執行緒數）。**11/25 Colab 實測確實如此**：兩支都在第一個數值接縫停下、沒有最終 CSV。現在 notebook 會自動改用 `--tolerant` 再跑一次（第 2.4 節）；容許模式下後面還有沒有別的檢查會擋、最終 CSV 是否相同，還沒在 Linux 上測。
2. **版本**：`uv python install 3.12.13` 要抓得到這個小版本；`torch==2.14.0+cpu` 等版本要有 Linux 輪子（裝不起來時退回最小安裝）。
3. **RSS 看門狗**：`p63_eval.rss_watchdog` 在程序 RSS 超過 3.0 GB 時直接結束（結束碼 97）。Windows 原執行峰值約 2.0 GB；Linux 的 RSS 算法不同，可能較高。
4. **大小寫**：Windows 檔案系統不分大小寫，Linux 分。程式裡若有和實際檔名大小寫不同的路徑，只會在 Linux 上出現「找不到檔」。
5. **Drive**：4.7 GB 的 zip 上傳要完整（notebook 會核 SHA-256）；Colab 本機磁碟要有約 11 GB 空間（複製 zip＋解壓，逐個刪除）。

### 5.6 公開 repo 版的第 1 層（2026-11-26 加）

公開 repo（`https://github.com/kuotunyu/aicup2026-go`，單一 commit 的快照）不放官方棋譜、由官方棋譜逐盤／逐題算出的檔、模型檔，也不帶歷史；不提供 `C10_code.bundle`。**官方競賽檔不得轉載，所以給公開連結的包一律不含官方檔**；5.1 的 `C10_features_layer1_comp{1,2}.zip` 內有官方測試 CSV，只供私下使用，不放公開連結。第 1 層改成「clone 公開 repo＋使用者自己從 AIdea 下載的官方測試檔＋三個公開包」：

| 包（雲端硬碟，公開連結） | 內容 | bytes | SHA-256 |
|---|---|---|---|
| `C10_public_features_comp1.zip` | 5.1 競賽一特徵 zip 的 905 個檔扣掉 8 個官方測試 CSV＝897 個（`data/` 下） | 4,598,622,322 | `2ecd3ffee9fb8c53319556d65b908efdc84ae693ca569522b7ed33953afaeed5` |
| `C10_public_features_comp2.zip` | 5.1 競賽二特徵 zip 的 64 個檔扣掉 2 個官方測試 CSV 與 6 個由它們做出的檔＝56 個（`data/` 下） | 120,212,732 | `be67d7bc212a7ed2fa3046d253ba62b41262ce0dcb7e377f9c707fc84d8194d9` |
| `C10_features_layer1_ingit.zip` | 補充包：兩份輸入清單中在開發用 repo 有追蹤、公開 repo 沒放的 66 個檔（37 npz＋29 csv，都在 `results/` 下）＋`repro/ref_lda_L.npz`（`--tolerant` 用），共 67 個；repo 相對路徑、git 的位元組 | 71,507,868 | `ccff1b16c80af5e4eef88943223efa677d232b1c0c1cff2640f927786b8baa64` |

三個都是 ZIP_STORED。產生時每個裝進去的檔都掃過「連續三手 SGF 落子文字」（0 個命中），並檢查沒有任何成員在 `data/test_v2_download/`、`data/test/`、`data/training`、`data/local` 下。

**不在任何包裡、要使用者自己準備的 16 個檔**（`repro/inputs_public.json` 裡 `"from": "official"`／`"derived"`）：

| 類別 | repo 內路徑 | 怎麼來 |
|---|---|---|
| 官方（8） | `data/test_v2_download/x/tests/player_identification_{candidates,candidates2,test,test2}_{public,private}.csv` | AIdea 競賽一「資料」分頁的 `aicup2026_player_identification_v2_tests_20261108_fix_readme.zip`（35,831,860 bytes，MD5 `90cda19fc750f0c99ab75c0706ba9314`） |
| 官方（2） | `data/test_v2_download/rank/rank_prediction_test_{public,private}.csv` | AIdea 競賽二「資料」分頁的 `aicup2026_rank_prediction_tests_20261030.zip`（3,447,261 bytes，MD5 `355c0b40be6af0413859991ebbcf31f3`） |
| 衍生（2） | `data/prereg56/src_rank/player_identification_test2_{public,private}.csv` | 競賽二題目檔的逐位元組副本（改名） |
| 衍生（2） | `data/prereg56/src_rank/player_identification_candidates2_{public,private}.csv` | 只有表頭一行（36 bytes） |
| 衍生（2） | `data/test/rank_prediction_test_{public,private}.csv` | 競賽二題目檔拿掉每盤開頭的 `HD[n]`（測試日 `tools/root_props.py` 做的事，PREREG_22） |

10 個官方 CSV 各自的 bytes、SHA-256、MD5 在 `repro/inputs_public.json`（公開 repo 的頂層 README 第 4 節有表）。

- 三個包、`data/delivery/C10_features_layer1_ingit_manifest.json`、`data/delivery/C10_public_SHA256SUMS.txt` 和 `repro/inputs_public.json` 由 `python repro/make_delivery.py public <公開副本資料夾>` 產生。
- `repro/inputs_public.json`：第 1 層全部 1,090 個輸入檔（兩份清單的 1,089 個＋`ref_lda_L.npz`）的路徑、bytes、SHA-256 和來源：comp1 包 897、comp2 包 56、補充包 67、公開 repo 本身 54、官方 10、衍生 6。
- `repro/stage_inputs.py --zips-dir <放三個包的資料夾> --official-dir <放官方測試檔的資料夾>`（只用標準函式庫）：核公開 repo 內的 54 個輸入檔 → 在 `--official-dir`（含子資料夾與裡面的 zip）依檔名找 10 個官方 CSV，bytes 與 SHA-256 相符才接受，複製到原路徑，再重建 6 個衍生檔並核 SHA-256 → 核三個包的大小與 SHA-256 → 解壓到 repo 原路徑、每個檔核 bytes 與 SHA-256（清單外的檔或 `data/local` 路徑就停）→ 把補充包的 67 個檔 `git add -f` 並在**該 clone 裡** commit（作者固定填 `layer1-repro`，訊息 `restore layer-1 inputs (local only; do not push)`；官方檔在 `data/` 下、不進 git）→ 確認 git 工作樹乾淨。可重複執行；`--check-only` 只檢查不寫入。
- 為什麼要 commit：`tools/t69_rank.py` 的演練模式（`--out`）不檢查 git 是否乾淨，但它呼叫的 `tools/t25_rank.py` 仍會對 `results/test19` 等輸入檔做 `require_committed`（檔案要被 git 追蹤、而且和 HEAD 相同）。11/26 實測：只需要「已追蹤且未改動」，兩支入口都不查 commit 時間或歷史——單一 commit 的公開 repo 加一個本機 commit 就通過，凍結的工具與入口都沒有改。
- Colab：`repro/colab_public.ipynb`（第 0 格的參數：`REPO_URL`、`REPO_REF`、`MOUNT_DRIVE`、`DRIVE_DIR`、`OFFICIAL_DIR`、`SAVE_TO_DRIVE`、`WORK`、`OUT`、`VENV`、`ZIP_LOCAL`、`ZIPS`、`PARTS_MANIFEST`）。流程：clone → 從 `DRIVE_DIR` 取三個包（整檔，或 `parts/` 的 7 MB 分塊＋`C10_public_parts_manifest.json` 接回；分塊用 `python repro/make_delivery.py split <檔名>…` 產生在 `data/delivery/public_parts/`）→ `stage_inputs.py`（官方檔取自 `OFFICIAL_DIR`）→ 建 Python 3.12.13 環境 → 兩支入口先嚴格模式、停了再 `--tolerant` → 比 SHA-256。環境、執行、總結三格沿用 `colab_layer1.ipynb`。**這份 notebook 還沒有在 Colab 上執行過**（公開 repo 還沒建立）；只做過每格的語法檢查。
- 本機乾淨 clone 演練的結果記在開發用 repo 的 `repro/public_manifest.md`（該檔不在公開 repo）。

## 6. 檔案

| 檔 | 用途 |
|---|---|
| `comp1_from_features.py`、`comp2_from_features.py` | 第 1 層單一入口 |
| `_common.py` | 根目錄、SHA、輸出資料夾檢查、最終 CSV 逐列比較、讀檔紀錄（不含任何演算法） |
| `_tolerant.py` | `--tolerant`：接縫的容許比較與 `tolerant_report.json`（第 2.4 節；不含任何演算法） |
| `ref_lda_L.npz`、`make_tolerant_refs.py` | 競賽一 L-LDA 在 Windows 上的擬合結果（容許模式的比較基準）與產生它的腳本 |
| `wsl_drill.sh` | 本機 WSL 的 Linux 演練（和 notebook 同樣的步驟；用法在檔頭）。**還沒執行過**，只做過 `bash -n` 語法檢查 |
| `inputs_comp1.json`、`inputs_comp2.json` | 11/25 實跑時讀過的輸入檔清單（路徑、bytes、SHA-256） |
| `requirements_cpu.txt`、`requirements_gpu.txt` | 兩個環境的實際套件版本 |
| `colab_plan.md` | Colab 化的盤點 |
| `colab_layer1.ipynb` | 第 1 層的 Colab notebook（第 5 節） |
| `make_delivery.py` | 產生 `data/delivery/` 的交付包（bundle、code zip、特徵 zip、manifest、SHA256SUMS；`public`／`split` 是公開 repo 版的補充包與分塊） |
| `stage_inputs.py`、`inputs_public.json` | 公開 repo 版：把使用者下載的官方測試檔與三個公開包核對、放到原路徑、在本機 clone 裡 commit（第 5.6 節）；全部輸入檔的清單 |
| `colab_public.ipynb` | 公開 repo 版的第 1 層 Colab notebook（第 5.6 節） |
