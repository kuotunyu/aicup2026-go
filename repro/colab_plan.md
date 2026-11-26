# Colab 化盤點（第 1 層的 notebook 與交付包已做，見 README 第 5 節；以下是 11/25 做之前的盤點，第 2、3 層仍適用）

背景：官方 Q&A 要求交出去的程式能在 Google Colab 跑（含下載外部權重）。這份只盤點要準備什麼、已知的障礙和要決定的事。大小都是 2026-11-25 實測或引自 `report/facts/shared.md` 7.4；沒量的寫「沒量」。

## 1. 現況

| 項目 | 狀態 |
|---|---|
| 第 1 層（特徵 → 最終 CSV）在本機 Windows | 兩場都逐位元組 PASS（`README.md` 2.1） |
| 第 1 層在 Colab／Linux | **沒跑過** |
| KataGo P1／HS 抽取的 Colab 版 | 有：`colab_gpu/gpu_fallback.ipynb`、`gpu_fallback_core.py`（PREREG_11，10/10 演練過） |
| 官方 CNN（競賽二 C5／SW、競賽一 E0）推論 | 有：`colab/test02_rank.ipynb`、`test04_rank.ipynb`、`test02_player.ipynb`（11/04 實際在 Colab 跑過，執行階段 2026.07） |
| v2 CNN 成員訓練 | 有 Colab 產生器（`p51_colab.py`、`p55_colab.py`、`p61_colab_nb.py`、`p62_colab.py`） |
| v2 CNN 成員「抽向量」、競賽二 c49 向量 | **沒有 Colab 版**（都在本機 4090 算） |
| 兩場最終後端（prereg67、t69）的 Colab notebook | **沒有** |

## 2. 要放 Drive 的東西

### 2.1 程式

- repo 追蹤的檔案共 675,458,683 bytes，其中 `results/` 665,070,080（含逐盤輸出的 npz／csv）；`tools/` 3,450,034、`tests/` 2,351,792、`colab/` 1,860,990、`third_party/` 490,793、`docs/` 1,303,233、`colab_gpu/` 56,732。
- 程式不能只取 `tools/`：兩條鏈都 import `results/prereg*/explore/` 下的模組，也讀 `results/` 下的凍結設定與模型檔。最省事是整個 repo 打包。
- 打包要用 git 的原始位元組（`git archive` 或 bundle 後在 Linux clone）。這台機器 `core.autocrlf=true`，Windows 工作副本裡部分文字檔是 CRLF；工具對 `.json` 多半用「CRLF 折成 LF」的雜湊，但有些是原始位元組比對（例如接縫 T 比 `results/test64/rank/` 的檔）。Linux checkout 下這些比對會不會過，沒驗證。

### 2.2 第 1 層的輸入（清單與 SHA 在 `inputs_comp1.json`、`inputs_comp2.json`）

| | 檔數 | bytes | 其中不在 git（要另外放 Drive） |
|---|---|---|---|
| 競賽一 | 922 | 4,759,129,811 | 905 個檔、4,741,168,850 bytes（`data/` 下） |
| 競賽二 | 169 | 231,411,895 | 64 個檔、160,736,268 bytes（`data/` 下） |

競賽一可以瘦身的地方（都要另寫入口或改登記的流程，現在的 apply 會讀）：7 個 `store_E.npz` 共 1,405,000,256 bytes 只被拿來核對 SHA；L 玩家的 P1 塊（1,505,835,049 bytes）只為了現場重學一個 LDA；06 特徵庫 1,355,001,420 bytes 只為了 G 項的 LDA 和詞表。

### 2.3 第 2 層另外需要的

| 檔 | bytes | 備註 |
|---|---|---|
| 7 個 CNN 成員權重（c49、S、B、p61、S1、S2、X） | 512,716,391（合計） | 路徑與 SHA：`shared.md` 7.4 |
| KataGo `b18c384nbt-humanv0.ckpt` | 323,752,318 | 不放 Drive：在 notebook 裡從 KataGo v1.15.0 release 下載並核 SHA-256 `84b23476…8b27` |
| 競賽二 C5 `epoch_9.pth` | 105,581,163 | 已在 Drive `C10/results`（11/04 用過） |
| 競賽二 SWA／SWB `final.pth` | 各 105,589,835 | 融合權重 0，但工具鏈會讀；已在 Drive |
| 競賽二 O4 計數表 `counts_fit.npz` | 178,537,934 | 由訓練棋譜算出 |
| 競賽一原題 E0 `best_model.pth` | 19,212,521 | 已在 Drive |
| 官方測試檔 | 競賽一 v2 8 檔 142,700,932；競賽二 2 檔 13,527,756 | 官方棋譜，不轉載；請評審放自己的副本或由主辦方提供 |

### 2.4 第 3 層（訓練）另外需要的

官方訓練棋譜（`data/training/`、v1.1 壓縮檔 220,710,452 bytes）、各訓練快取（沒量）。訓練不打算在交付時重跑，只給程式與指令。

## 3. 不能公開／不能轉載的

| 類別 | 例子 |
|---|---|
| 官方棋譜 | `data/test*`、`data/test_v2_download/`、`data/training*`、`data/prereg56/src_rank/` |
| 由棋譜逐盤算出的檔 | CNN 向量（`pool_*.npz`、`store_E.npz`）、P1 塊（`data/prereg39/p1/`、`data/test08/p1/`、`data/prereg10/mock/`）、06 特徵庫（`data/prereg06/feats/`）、名單（`data/prereg39/lists/`）、`results/test0*/**` 的逐盤 dumps 與逐題機率檔、`results/` 內其他 npz／逐題 csv |
| 驗證集答案 | `data/local/`：任何情況不交付、不上 Drive |
| 外部權重 | KataGo 權重不轉發，只給官方網址與 SHA |
| 模型權重（自己訓練的） | 不是逐盤資料，但由官方棋譜訓練；能不能公開、交給主辦方的方式，待使用者決定 |

「放在自己 Drive、只把連結給主辦方」算不算轉載，規則原文沒有寫到這個情形（沒查到）；要使用者決定。

## 4. 已知的障礙（要處理才可能在 Colab 逐位元組 PASS）

1. **版本鎖**：`tools/t25_rank.py` `check_venv` 要 python 3.12.13、numpy 2.5.3、scipy 1.18.1 完全相同，否則結束碼 4。Colab 預設 Python 不是這個小版本，要在 notebook 裡用 uv 建 3.12.13 的 venv 再跑（沒試過）。
2. **BLAS 與執行緒**：競賽一的後端在 Windows 會把 numpy 的 OpenBLAS 限成 1 執行緒（`p38_backend.polite`／`blas_threads`，只對 Windows 的 DLL 有效，Linux 上是空動作）；競賽二的位元組接縫用預設執行緒。Linux 的 numpy 輪子用不同的 OpenBLAS 建置，浮點結果是否逐位元相同沒有保證、沒驗證。這是最大的不確定。
3. **截止時間鎖**：競賽二入口已在程序內繞過 `K6_64`（`README.md` 2.2）。其他層的工具（t25、t19、t09…）在正式模式也有各自的截止，從頭重跑整條鏈時都會拒絕。
4. **git**：演練模式不檢查 git 是否乾淨，但會呼叫 `git rev-parse HEAD` 記錄；沒有 `.git` 時的行為沒試過。
5. **Windows 專用程式碼**：`p38_backend.py` 的 `polite`、`wait_for_ram`、可用記憶體查詢用 `ctypes.windll`（非 Windows 直接跳過）；`p63_eval.py` 的記憶體監看在 Linux 的行為沒試過。腳本（`.sh`）裡寫死 `/d/AI-Portfolio/...` 和 `.venv-gpu/Scripts/python.exe`，Colab 不能直接用，要改用 python 指令列。
6. **第 2 層不是逐位元**：同一台 4090 重抽 c49 向量，500 盤的最大絕對差 7.6e-06（`README.md` 3.4）；11/04 Colab 換執行階段時競賽二 C5 的接縫差過 1.29e-4。Colab 上重抽特徵後，最終 CSV 可能有少數列不同。驗收標準要改成「特徵在容許差內＋最終 CSV 不同的列數」，容許差要先定。
7. **驗證集接縫**：競賽二 t08／t09／t19 的 apply 和競賽一 t06 的 seam 會讀 `data/local` 的題目檔做接縫。交付包不能附 `data/local`，所以「從 SGF 整條重跑」不能直接用這些測試日工具，要另寫只做測試題那一半的串接入口（呼叫同樣的函式、跳過驗證集接縫）。
8. **`data/prereg56/src_rank/` 的建立腳本不在 repo**（題目檔改名＋只有表頭的候選檔），要補一支小工具。
9. **sp128 的擬合程式不在 repo**（陣列取自顧問檔）；只能交陣列檔，訓練重現要另外說明。

## 5. 工作項目（不估時數）

A. 第 1 層搬上 Colab（兩場）：
  1. 打包：repo（git 原始位元組）＋ `inputs_comp*.json` 列的 `data/` 檔，照原相對路徑放 Drive。
  2. notebook：掛 Drive → 建 3.12.13 venv、裝 `requirements_cpu.txt` → 核對輸入 SHA → 跑兩支入口。
  3. 實跑一次，看是 PASS 還是哪個接縫先掛；掛在 BLAS 差異的話，改成回報「和已上傳檔不同的列數」並寫明。

B. 第 2 層搬上 Colab：
  1. CNN 向量：把 `p49_run.py embed`（及 p51／p55／p61／p62／p56_opp）的指令包成 notebook 格子；輸出到新資料夾，和已存向量比（取樣數相同、最大絕對差）。
  2. KataGo P1：沿用 `colab_gpu/`，補上競賽一 pub2／priv2 與競賽二測試題的名單；HS 視是否要重現權重 0 的項而定。
  3. 競賽二 C5／SW、競賽一 E0：沿用 `colab/test02_*.ipynb`、`test04_rank.ipynb`。
  4. 另寫「不經測試日鎖」的串接入口，把重抽的特徵接到第 1 層（第 4 節第 3、7 點）。

C. 第 3 層：整理每個模型的訓練指令（`README.md` 第 4 節表格補齊命令列），不重跑。

D. 文件：權重與資料怎麼給主辦方、哪些檔不附，寫進報告的「程式碼」一節。

## 6. 要使用者決定的事

1. 交付範圍：只保證第 1 層逐位元組重現（特徵檔一起交），還是也要第 2 層（從官方 SGF 重抽特徵，結果是「容許差內」而不是逐位元組）？
2. 由官方棋譜逐盤算出的特徵檔（競賽一約 4.7 GB）能不能放 Drive 給主辦方；不能的話，第 1 層對主辦方就無法獨立執行。
3. 自己訓練的權重怎麼交（Drive 連結、附檔、或只給 SHA）。
4. Colab 上若因 BLAS 差異不是逐位元組，接受「列出不同的列數」的寫法嗎？
5. 競賽一原題 800 列（06-ON）和競賽二 SW／HS（權重 0）要不要也做成可在 Colab 重跑，還是沿用已存輸出並寫明。
