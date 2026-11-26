# PREREG_07 試跑（競賽二）：KataGo HumanSL 計分器

最後更新：2026-10-03 凌晨（對話 `a8d08794` 的 workflow 子代理，接續 10/2 晚因用量上限中斷的那一輪）。

範圍：只用訓練玩家（fit／inner）的棋譜。沒打分，沒碰 `data/local`，沒跑 `tools/score.py`。

## 結論

- **計分器做好了，和 KataGo 自己的程式對得上。**
  - issue #1154 空棋盤的三個數字，和 issue 印出的值相差 ≤ 5.8e-7。
  - 25 個真實 fit 局面 × 4 個 profile：和 KataGo 的 `GameState`、`humanslnet_server.py` 比，機率最大差 9.7e-7。
  - 5 個等級的似然曲線峰值照等級排序。
- **4090 實測**（PyTorch，一批 1,022 次評估＝73 個局面 × 14 個 profile）：
  - fp16 是 1,451 evals/s，加 channels_last 是 1,805 evals/s，最快。
  - fp16 和真 fp32 比：log p 最大差 0.0084；每個 profile 對 600 個局面取平均後，差 ≤ 0.0002。
- **GPU 煙霧測試通過**：用 `pilot_run.py` 在 GPU 跑 80 局，再用 CPU fp32 重算核對，每手最大差 0.0073，每局平均差 ≤ 0.00019。
- **試跑還沒跑。**
  - 全量（fitmap 3,000＋bank 11,813＋mono 61,356 局，14 個 profile）約 1.18 億次評估，估 18–23 小時（估計值，見「成本」）。
  - 建議第一晚只跑 fitmap＋bank（估 3.5–4.4 小時），接著跑 `pilot_fuse.py` 看閘門。
  - 閘門過了或接近，再分兩三晚補 mono。

## 檔案

| 檔案 | 用途 |
|---|---|
| `third_party/katago/` | KataGo python 模組，原封不動複製。commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`；LICENSE、CONTRIBUTORS、SOURCE.txt 記了每個檔的 git blob id 和 SHA-256 |
| `hs_lib.py` | 計分器本體，見下一節 |
| `hs_fixtures.py` | 抽固定的 fit 樣本 `data/prereg07/fixtures/fit_sample.json`：每級 12 局，共 120 局，seed 20261002 |
| `hs_parity.py`、`hs_parity.json` | 核對 (a)(b)(c) 的程式和 10/2 的結果 |
| `test_hs_lib.py` | pytest，刻意不放在 `tests/`。內容：vendor 檔逐一和上游 blob 相同、data/local 防護、metadata 列、棋譜重播、權重 SHA、(a)(b)(c)，以及補齊批次前後數值相同 |
| `hs_gpu_bench.py`、`hs_gpu_bench.json`、`hs_cpu_feat.json` | 4090 測速、精度比對、29 級格點掃描、CPU 特徵化速度 |
| `pilot_run.py` | 試跑第 1 步：GPU、可續跑，輸出到 `data/prereg07/pilot/<set>/chunk_*.npz` |
| `pilot_fuse.py` | 試跑第 2 步：CPU 上做 OOF 融合和閘門判定，輸出 `pilot_fuse_report.json` |
| `pilot_smoke_check.py`、`pilot_smoke_check.json` | GPU 煙霧 chunk 和 CPU fp32 的核對 |
| `hs_synth.py` | 合成 chunk（null／planted），只用來測 `pilot_fuse` 的機制 |

權重 `data/katago/b18c384nbt-humanv0.ckpt`：

- 323,752,318 bytes，SHA-256 `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27`，zip 內 1,245 個 CRC 全對。
- GitHub 沒公布這個檔的 digest，所以這個 SHA 只是釘住我們自己的這份，不是驗證來源。

## `hs_lib.py` 的固定選擇

- **權重**：用 `model`，不用 `swa_model`，和 issue #1154 一樣。
- **怎麼載入**：
  - 先試 `torch.load(weights_only=True)`。KataGo 的 `load_model` 註冊過 safe globals，但這個檔的 running metrics 是 defaultdict，所以還是失敗。
  - 於是先在不執行 pickle 的情況下掃出它引用的全域名稱，確認正好是釘住的 7 個。
  - 再用只認這 7 個名稱的 restricted unpickler 載入。原檔不改寫。
- **規則**：日本規則、貼目 6.5。
  - 我們的 SGF 只有 SZ 和落子，沒有 RU、KM、HA。棋譜來自日本棋院幽玄之間（docs/DATA.md），所以這是假設。
  - 規則只影響 3 個全域輸入：貼目、地域計分旗標、seki tax。
- **metadata**：照 C++ 的 humanSLProfile 預設（`makeBasicRankProfile`）。
  - 雙方都是人類、雙方同等級、是否計分不明。
  - 時限：基本 1,200 秒＋讀秒 5 次 × 30 秒。
  - 來源 KGS，日期 2020-03-01（`rank_*`）或 2016-09-01（`preaz_*`）。
  - 可選：來源 FOX、任意日期、對手等級（same／unknown／指定一級）。
- **非法點遮罩**：先遮罩非法點再做 log-softmax（C++ 引擎也這樣做）。
  - 遮掉的點：有子的點、劫的禁著點、自殺（單子或多子，日本規則都禁止）。
  - pass 永遠合法。
  - KataGo 的 python `get_model_outputs` 不遮罩，所以 (b) 遮罩前、遮罩後兩種都比。
- **輸出**：主 head 的 policy 第 0 通道（policy0）。不做對稱平均（symmetry 0）。
- **輸出內容**：每局、每個 profile 的 log p(實際落子) 總和與手數。目標方一定算；對手方可選（`--sides both`）。
- **CUDA 上補齊批次（10/3 新增）**：每次 `logp_moves` 呼叫的最後一批補到固定形狀。理由見「這輪修的 bug」。

## 驗證結果

數字出自本資料夾的 json。

### (a) issue #1154 空棋盤

- 設定：中國規則、貼目 7.5、proyear_2020 metadata。
- C16／D16／E16 的機率：0.036621131、0.156813085、5.0567211e-05。
- 和 issue 的 python 數字最大差 1.04e-7，和 engine 數字最大差 5.81e-7。
- hs_lib 直接前向和 GameState 的差是 0。

### (b) 和 KataGo 自己的推論路徑比

- 樣本：4 局 fit（D、A、3D、6D），共 25 個局面 × 4 個 profile（rank_10k、rank_1d、rank_5d、fox20220601_3d_oppunknown），合計 100 次評估。
- 和 `GameState.get_model_outputs` 比：
  - 機率最大差：遮罩前 9.72e-7，遮罩後 9.13e-7。
  - log p(實際落子) 最大差 2.71e-6。
  - 合法點遮罩完全一致。
- 和 `humanslnet_server.py` 比：
  - 原封不動當子程序跑，CPU，batch 1。
  - 最大差 9.72e-7。

### (c) 似然曲線的峰值順序

- 樣本：D、A、1D、4D、6D 各 2 局 × 11 個 profile。
- 每手平均 log p 的峰值：D→6k、A→3k、1D→1k、4D→1k、6D→2d。
- 事前寫下的標準是「照等級不遞減，且 6D 高於 D」，通過。
- CPU fp32 只有 10.2 evals/s。

### pytest

- 指令：`PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -rs -p no:cacheprovider results/prereg07/explore/pilot_humansl/test_hs_lib.py`
- 10/3 跑完：17 passed、135 subtests passed，花 2,772 秒。
- 其中 (c) 佔 2,186 秒，因為同時有別的 session 在吃 CPU。

### GPU 精度（`hs_gpu_bench.json` agreement）

- 樣本：600 個局面 × 14 個 profile。
- 對照組：GPU fp32，cuDNN 和 matmul 的 TF32 都關。

| 精度 | log p 最大差 | 平均差 | p99 | 各 profile 平均的最大差 | 每局面最佳 profile 相同的比例 |
|---|---|---|---|---|---|
| fp16 autocast | 0.0084 | 0.00062 | 0.0036 | 0.00021 | 98.2% |
| bf16 autocast | 0.062 | 0.0049 | 0.028 | 0.0011 | 92.2% |
| fp32＋TF32（PyTorch 預設） | 0.0043 | 0.00030 | 0.0017 | 0.00007 | 98.5% |

- GPU fp32 和 CPU fp32 比：48 個局面 × 2 個 profile，最大差 7.2e-6。
- 結論：試跑用 fp16，不用 bf16。

### GPU 煙霧測試（`pilot_smoke_check.json`）

- 指令：`pilot_run.py run --set fitmap|bank --max-games 40 --out-dir data/prereg07/pilot_smoke`，fp16、channels_last、batch 1024。
- 0 局失敗。
- refs 等於 plan 的前 40 列。
- 每局目標手數等於重新特徵化的結果。
- 各局總和等於逐手 log p 加總，差 0。
- 抽 2＋2 局、3 個 profile（rank_20k、rank_1d、rank_9d）用 CPU fp32 重算：
  - 每手最大差：fitmap 0.0067，bank 0.0073（容許 0.05）。
  - 每局平均最大差 0.00019（容許 0.002）。

### pilot_fuse 機制測試

- 合成資料用 `hs_synth.py` 照現行 plan 重建，放在 `data/prereg07/smoke_synth/*_v2`。
- Q02 的 inner eval 分等級平均是 0.6101，和 `u1_oof.npz` 逐位元組相同（seam 檢查）。
- null（純雜訊）：Q02+HS − Q02 = 0.0，CI [0, 0]，閘門不過，符合預期。
- planted（植入訊號）：
  - Δ = +0.3887，CI [0.3674, 0.4106]，閘門過。
  - mono Δ = +0.3883，CI [0.3720, 0.4059]，黑白兩色都 > 0。
- 只放前 60 塊 mono 時：覆蓋 27,814 / 120,544 題，報告寫 `complete: false`。

## 測速（`hs_gpu_bench.json`，2026-10-03，RTX 4090）

- 環境：torch 2.11.0+cu128，cuDNN 91900，`cudnn.benchmark` 開。
- 開跑前 nvidia-smi 沒有別的 compute 程序。結束時列出的那一個是測速程式自己。
- 每格先暖機，再量 6 秒。速度包含 H2D 複製、遮罩、log-softmax、gather。

| 精度 | 每批評估數 | evals/s | 暖機（含 autotune）秒數 |
|---|---|---|---|
| fp16 | 252 | 1,626 | 9.0 |
| fp16 | 504 | 1,513 | 8.6 |
| fp16 | 1,022 | 1,451 | 9.2 |
| fp16 | 2,044 | 1,441 | 11.3 |
| bf16 | 252 | 1,597 | 4.5 |
| bf16 | 504 | 1,469 | 4.5 |
| bf16 | 1,022 | 1,455 | 5.3 |
| bf16 | 2,044 | 1,545 | 6.9 |
| **fp16＋channels_last** | **1,022** | **1,805** | 8.0 |
| fp32（TF32） | 1,022 | 1,264 | 6.8 |

- 顯存最多 10.5 GB（batch 2,044）。
- 速度瓶頸在 PyTorch eager，加大 batch 沒有幫助。另一次只量裸前向的診斷（scratchpad 腳本）也只有 1,802–1,814 evals/s。
- **CPU 特徵化**（`hs_cpu_feat.json`，24 個邏輯核心）：
  - 每秒局面數：1 個 process 155、4 個 418、8 個 672、12 個 794、16 個 982、23 個 1,094。
  - GPU 每秒只需要 1,805 / 14 ≈ 129 個局面，所以 4 個 worker 就夠。預設用 8 個。
- **煙霧測試實際速度**：每塊含 process 啟動和 autotune，fitmap 40 局 945 evals/s，bank 40 局 1,149 evals/s。長時間的穩定速度沒有量到；正式跑時 `run_log.jsonl` 的 `eta_min` 會給實際值。
- **格點掃描**（每級 3 局、29 個等級，只當設計參考）：
  - KGS 各級峰值：D 5k、C 5k、B 2k、A 3k、1D 1k、2D 2k、3D 2k、4D 1k、5D 1d、6D 1d。
  - FOX 各級峰值：D 5k、C 4k、B 2d、A 1k、1D 3d、2D 2d、3D 2d、4D 3d、5D 4d、6D 4d。
  - FOX 的峰值分得比較開。哪一種比較能分辨等級，沒有量。

## 試跑設計

### 要算哪些局

`data/prereg07/pilot/plan.json` 已建好，10/2 13:48 UTC，SHA-256 `8494e9af…`。

- **bank**：11,813 局，是 `results/prereg02/rank/pseudo_inner.json` 的 636 位 bank 玩家的全部棋局。prereg02 的 inner fit 和 eval 擬題都從這裡抽，所以閘門 G1／G2 只需要這一組。
- **mono**：PREREG_03 單色擬題用到、但不在 bank 裡的 61,356 局（單色擬題共用 72,908 局）。
  - 依玩家分組，各級玩家輪流排序（seed 20261004）。
  - 中途停下來，已算完的玩家仍然完整，`pilot_fuse` 會報實際覆蓋了幾題。
  - bank 和 mono 合計 73,169 局，inner 共 73,556 局。
- **fitmap**：每級 300 位 fit 玩家各 1 局，共 3,000 局（seed 20261003）。只用來學「HS 曲線 → 10 級」的對照，不是 inner 玩家。

### HS 項怎麼接到 10 級

1. **每局的曲線**：x_g[j] ＝ 目標方在 profile j 下的 log p 總和 ÷ 目標手數，共 14 維。
2. **對照模型**：10 類 multinomial logistic regression。
   - 用 fitmap 的平均和標準差把特徵標準化，加 L2。
   - L2 強度從 {1e-4, 1e-3, 1e-2, 0.1, 1} 裡挑，用 fitmap 上 5 折 CV 的 log-loss 選。
   - 輸出每局的 lp_HS ＝ log_softmax(Wx + b)。
   - 建 lp_HS 的過程不用任何 inner 標籤。
3. **融合**：lp_HS 照 O4、SW 的方式，在 p04 的 `pooled(…, "log")` 裡合併成一題的對數項，用 `u1_lib.run_gen` 做 5 折玩家 OOF 的廣義校準。
   - 比較 Q02（OP、O4、SWA、SWB）和 Q02+HS（再加 HS）。
   - 只記錄不判定：R20+HS；用 `--sides both` 時另記 Q02+HSO、Q02+HS+HSO。
4. **閘門**（試跑用，不是事前登記）：
   - **G1**：inner eval 的分等級平均，Q02+HS − Q02 ≥ +0.010。
   - **G2**：成對的 PlayerBootstrap（1,000 次，`default_rng(0)`）95% 下界 > 0。
   - **mono 檢查**：照 PREREG_04 G4。單色擬題的下界 > 0，黑白兩色的點估計都 > 0，並附覆蓋率。

### profile 格點

14 個 KGS 預設：rank_20k、15k、12k、10k、8k、6k、4k、2k、1d、2d、3d、5d、7d、9d。

- (c) 的峰值落在 6k–2d，格點掃描每局的峰值落在 7k–6d，所以中段排密。
- 兩端保留，用來固定曲線的形狀。

### 成本

- 計算基礎：每局平均 111.04 個目標手（全部 720,307 局 fit 的平均；煙霧測試是 105.4 和 115.4），14 個 profile，等於每局 1,554.6 次評估。
- 以下時數都是估計值。

| 組 | 評估數 | 1,805 evals/s（量到的最快值） | 1,451 evals/s（不開 channels_last） | 1,149 evals/s（煙霧測試，含啟動） |
|---|---|---|---|---|
| fitmap | 4.66M | 0.72 h | 0.89 h | 1.13 h |
| bank | 18.36M | 2.83 h | 3.52 h | 4.44 h |
| mono | 95.38M | 14.7 h | 18.3 h | 23.1 h |
| 合計 | 118.4M | 18.2 h | 22.7 h | 28.6 h |

- 一塊 200 局約 311k 次評估，約 2.9–3.6 分鐘。用 STOP 讓出 GPU 時，最多等這麼久。
- 對手方也算（`--sides both`）的話，成本加倍。預設不算。

### 指令

在 Git Bash，從 repo 根目錄 `D:\AI-Portfolio\competitions\aicup2026-go` 執行。開跑前先跑 `nvidia-smi`，確認沒有別的 compute 程序。程式自己也會檢查，有的話拒絕開跑。

```
PY=.venv-gpu/Scripts/python.exe; S=results/prereg07/explore/pilot_humansl; export PYTHONDONTWRITEBYTECODE=1
# 第一晚：fitmap + bank（估 3.5-4.4 h）
$PY $S/pilot_run.py run --set fitmap && $PY $S/pilot_run.py run --set bank
$PY $S/pilot_run.py status
# CPU，約 1 分鐘：閘門
.venv/Scripts/python.exe $S/pilot_fuse.py          # -> $S/pilot_fuse_report.json
# G1/G2 過了或接近才補 mono，每晚限時、可續跑：
$PY $S/pilot_run.py run --set mono --stop-after-min 420
.venv/Scripts/python.exe $S/pilot_fuse.py          # 每晚跑完後重跑一次，報告會寫覆蓋率
```

- **中斷與續跑**：每塊寫完才換名（tmp→正式）。中斷最多丟掉正在跑的那一塊；重下同一行就從沒寫完的那塊接著跑。
- **讓出 GPU**（例如 10/24 前 C4 要用）：在 `data/prereg07/pilot/` 建一個空檔 `STOP`，跑完手上那一塊就停。刪掉 `STOP` 再重下同一行就繼續。
- **不要重跑 `pilot_run.py plan`**：結果相同，但會改寫 `created_utc`。

## 這輪修的 bug 和改動（10/3）

1. **每遇到新形狀就重新 autotune，速度大降**。
   - 原因：開了 `cudnn.benchmark`，每個新的批次形狀要 autotune 8.5–10.6 秒（診斷時實測）。原本每局、每塊的最後一批形狀都不一樣。
   - 影響：第一次 GPU 測速的格點掃描每局都撞到新形狀，超過 10 分鐘上限。約 10.5 分鐘時手動停掉，沒有產出，停掉時 GPU 使用率只剩 2%。
   - 修法：`hs_lib.Scorer` 新增 `pad`，CUDA 上預設開，把最後一批用最後一個局面補齊，算完丟掉補的列。
   - 驗證：新增測試 `Padding`，CPU 上補齊前後相同（容許 1e-5），通過。
2. **`hs_gpu_bench.py`**：
   - 加時間預算：480 秒後不再開新的段落；每段跑完就存檔。
   - 每格量 6 秒，格點掃描每級 3 局，並改成每個來源一次補齊呼叫。
   - 記錄每格的暖機秒數。
3. **`pilot_run.py`**：
   - nvidia-smi 有別的 compute 程序就拒絕開跑。
   - 新增 STOP 檔和 `--stop-after-min`。
   - 新增 `--max-games`，只限煙霧測試；指向正式資料夾時會拒絕。
   - channels_last 預設開，有 `--no-channels-last` 可關。
   - 每組的前向次數改成 16，減少補齊浪費。
   - chunk 內記錄 batch 和 channels_last。
   - `status` 可以帶 `--out-dir`。
4. 新增 `pilot_smoke_check.py`，並把合成資料產生器移成 `hs_synth.py`（10/2 的版本在 scratchpad）。
   - 10/2 的 `smoke_synth/null`、`planted` 是用較早的 plan 建的（mono 順序不同），各自帶著自己的 `plan.json`，結果自洽。
   - 10/3 另外用現行 plan 建了 `*_v2`。
5. **GPU 使用紀錄**：這次接續合計約 22 分鐘。
   - 超時那次約 10.5 分鐘。
   - 兩次診斷各約 1 分鐘。
   - 正式測速 386 秒。
   - 煙霧測試 3 分鐘。
   - 每次開跑前 nvidia-smi 都沒有別的 compute 程序。

## 還沒決定的事

- 試跑什麼時候跑。要排夜間；10/24 前 C4 優先，可以用 STOP 讓出。
- mono 要全跑（估 15–18 小時，兩三晚），還是只跑一部分。建議先看 G1／G2 再決定。
- profile 用 KGS 預設，還是 FOX 或混合。這次用 KGS。FOX 的峰值分得比較開，但分辨力沒有量過。
- 速度只有 PyTorch eager 的約 1.8k evals/s。
  - REPORT 估計 TensorRT 版的 b18 約 7k/s，沒有實測。
  - 只有試跑過關、而且測試日的成本變重要時才值得做。
  - 測試日約 10,000 局，以 1,805/s 估約 2.4 小時（估計值）。
