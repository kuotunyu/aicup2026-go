# PREREG_06 實作規格：競賽一 v2 風格特徵、CSLS、一對一開關

- 依據：`docs/PREREG_06_PLAYER_STYLE.md`（已納入第 1 輪審查 `review_r1.md`）。本檔和登記不一致時，以登記為準。
- 本檔是設計文件，不是登記：
  - 實作偏離本檔，不用寫修訂。
  - 偏離登記，才要寫 rev 2，而且要在 `freeze` 之前。
- 範圍：只做競賽一，不需要 Colab。GPU 只用在第 4 節（fit 玩家的 CNN embedding，本機 4090）。
- 數字來源：本檔引用的數字，都出自已存在的檔，或 10/2–10/3 實際跑過的程式。
  - `results/prereg06/explore/c1_headroom/*.json`：探索的結果。
  - `results/prereg06/design/c1d_veto_check.py` 的輸出 `c1d_veto_check.json`：讀探索存下的 `s8_rows.json`，350 個模擬池。
  - `results/prereg06/design/c1d_cnn_bench.py` 的輸出 `c1d_cnn_bench.json`：在 4090 上量 768 盤 fit 黑棋。
  - `results/prereg06/design/review_r1/`：審查補跑的 `fewdup_sim.py`、`csls_strength.py` 和它們的輸出。
  - 第 3 節的人數和盤數：10/2 用 `c1h_lib`（裝了 data/local 的防護）加上 `load_rows(part="fit", local_dir=None)` 數出來的；審查用探索快取重算相同。
  - `c1d_veto_check.json`、`c1d_cnn_bench.json`、`s8_rows.json`（1.6 MB）和審查的檔目前在 session 的暫存區，PREREG_06 rev 1 commit 前要複製進本資料夾（登記第 9 節第 1 步）。
  - 標「估計」的都是推算，不是實測。

## 0. 紀律（每支新工具都要做到）

- **data/local**：
  - 只有三個地方准讀，而且只讀有棋譜的兩份檔 `player_val_candidates.csv`、`player_val_questions.csv`：
    - `p06_player.py leakcheck`（凍結前，只算 SHA-1 雜湊，透過 `load_rows(part="fit", local_dir=...)`；VALIDATION rev 2 規定的外洩檢查）；
    - 凍結之後的 `p06_player.py apply`；
    - `t06_player.py feats/seam --pool seam_val`。
  - 檔名含 "ans"+"wers" 的路徑一律拒絕，同 p02 的 `refuse_forbidden`。
- **其他設計步驟只讀 fit 玩家**（build、lda、selfcheck、cnn-fit、tune、switch、dmix）：`load_rows(part="fit", local_dir=None)`，`EXPECTED["fit"]` 的人數和盤數核對照常做；和驗證檔的雜湊比對由 `leakcheck` 單獨做一次，結果寫進 `leakcheck_report.json`，`build` 要求它已 commit 而且重疊是 0。
  - 讀任何路徑之前，先確認它不在 `data/local` 底下：比對絕對路徑和 realpath 的前綴。
  - 不用子字串比對：`AppData\Local` 會被誤判成 data/local，這是探索時 `c1h_lib` 遇到的問題。
- **不跑 `tools/score.py`**。H6P 的兩行指令由主 session 照登記第 3 節手動執行。
- **不 import `results/` 底下的程式**：會寫出 `__pycache__`，凍結的資料夾就髒了。
  - 需要的探索函式照原樣移植進 `tools/p06_*.py`，再用測試核對結果和探索相同。
  - 例外：`selfcheck` 和測試要 import 探索程式時，先設 `sys.dont_write_bytecode = True`，跑完確認 `results/prereg06/explore/` 底下沒有新的 `__pycache__`。
  - 探索程式找快取的環境變數 `C10_C1H_SCRATCH` 設成 `data/prereg06/explore_cache`（重建或從暫存區複製），檔案 SHA 記進 `selfcheck_report.json`。
- **每個指令都寫一份 JSON 報告**，內容包括：
  - 輸入檔的 SHA-256；
  - 工具的 SHA-256（CRLF 折成 LF 再算）；
  - git HEAD；
  - Python、numpy、scipy 的版本；
  - 秒數；
  - mode（real 或 smoke）。
  - real 模式要求工具已 commit 而且乾淨。
- **大檔**放 `data/prereg06/`、`data/test06/`（整個 `data/` 在 gitignore 裡），它們的 SHA 寫進要 commit 的報告。
- **CPU 平行最多 6 個 worker**，這是這台電腦的規則。每個 worker 設 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`（審查實測：不限執行緒、5 個 worker 時 8 分鐘跑不完 1 個 seed；限 4 條後每個 seed 11–13 秒），值寫進報告。

## 1. 檔案

### 1.1 新檔（main，在 PREREG_06 rev 1 之後）

| 檔 | 內容 |
|---|---|
| `tools/p06_style.py` | 每盤特徵和各項分數（純函式，只用 stdlib、numpy、scipy） |
| `tools/p06_sim.py` | fit 玩家的模擬池、假題池、每池重學 LDA |
| `tools/p06_player.py` | 設計步驟和驗證集用的 CLI |
| `tools/p06_cnn_fit.py` | 4090 上的 CNN embedding（用 `.venv-gpu` 跑） |
| `tests/test_p06_style.py`、`test_p06_sim.py`、`test_p06_player.py` | 單元測試和流程測試 |
| `tests/test_p06_player_hygiene.py` | H6P 之後才加：釘住 `results/prereg06/player/**`、打分紀錄帳的列數，並檢查 PREREG_06 只准加行 |
| `results/prereg06/design/*.json` | 各步驟的報告 |
| `results/prereg06/player/` | `lda_B.npz`、`lda_W.npz`、`frozen_config.json`、`sr_report.json`、`apply_report.json`、`player_val_top5.csv`、`player_val_top5_switch.csv`、`score_H6P.json`、`score_H6P_switch.json` |
| `results/prereg06/RESULTS.md` | 結果，由另一個代理核對數字 |

### 1.2 新檔（分支 `p06-testday`）

| 檔 | 內容 |
|---|---|
| `tools/t06_player.py` | 測試日工具（第 7 節） |
| `tools/t06_fake_tests.py` | 演練用的假 dumps：官方形狀、fit 玩家；第 4 節的 embedding 存在時用它，不存在時用固定 seed 產生的向量 |
| `tests/test_t06_player.py`、`test_t06_official_log.py`、`test_t06_hygiene.py` | 測試日的測試 |
| `docs/TESTDAY_06.md` | Claude 的測試日步驟 |
| `results/test06/design/q04_fake_golden.json`、`build_review.md` | 寫 t06 之前記下的 Q04 演練基準；建置和審查紀錄 |
| `tools/official_log.py` | 只加行（第 8 節） |
| `.gitattributes` | 只加一行：`results/test06/** -text` |

### 1.3 不能動的

- R20 路徑的 39 個檔（`results/prereg04/design/r20_path_fingerprint.json`）。
- 04 路徑的 21 個檔（`results/prereg06/design/p04_path_fingerprint.json`）。
- `results/prereg01`–`prereg05/**`、`results/prereg06/explore/**`。
- `tools/p02_*.py`、`p03_rank_fuse.py`、`p04_*.py`、`c10data.py`、`decide.py`、`score.py`、`make_val.py`。
- 所有產生器和 notebook；`docs/PREREG_01`–`04`、`docs/TESTDAY.md`、`docs/TESTDAY_04.md`、`docs/VALIDATION.md`、`docs/DATA.md`。
- 打分紀錄帳只准由 `score.py` 加列。

### 1.4 檔名

- 不能用的名稱：
  - 含 `test02` 或 `t02_`；
  - 放進 `results/test02/**` 或 `results/prereg0[1-5]/**`；
  - 新增 `tools/p02_*`；
  - 符合 04 指紋的樣式：`tools/t04_*.py`、`colab/*test04*`、`tests/test_t04_*.py`、`results/test04/*`、`docs/TESTDAY_04.md`。
  - 用了這些名稱，`test_p04_hygiene`、`test_t04_hygiene`、`test_p06_hygiene` 會失敗。
- PREREG_07（競賽二）用 `p07_*`、`t07_*`，不會和本線撞名。

## 2. `tools/p06_style.py`（函式和約定）

定義照登記第 1 節。這裡寫的是介面。

| 函式 | 約定 |
|---|---|
| `parse_all(sgf)` | 用 `p02_moves.MOVE_RE` 取出全部的手，回傳 `[(色, 'xy')]`，不設上限 |
| `alternates(mv)` | 登記 1.2 的交替檢查：第 i 手的顏色字母是 B（i 偶數）或 W（i 奇數）。不交替的盤，下面所有特徵函式都不呼叫，盤數記錄 |
| `jo_tokens(mv, target)` | 登記 1.3。token 前面加目標方的顏色 |
| `style2(mv, target)` | 登記 1.4，回傳 66 維 float64 |
| `resp_tokens(mv, target)` | 登記 1.5 |
| `ldain(mv, target, vocab_index)` | 回傳 66＋詞彙數 維；詞彙內 token 總數是 0 時那一段全 0 |
| `lda_fit(X, labels, reg=1e-3)`、`lda_project(X, model, d=64)` | 照 `c1h_lda.fit`；方向依 λ 由大到小 |
| `colour_terms(...)` | 每個顏色的 T_L、T_S2，以及登記 1.8 的合併；全黑時等於 z(T_B) |
| `jo_cos(...)` | 直接用 `p02_player_open.open_cos` |
| `zrows`、`fuse(terms, weights)` | 和 p02 相同：母體標準差，<1e-12 或缺值時記 0 |
| `csls(F, k)` | 照 `c1h_eval.csls`；k 是 None 時回傳 F |
| `hungarian_top1(Z)`、`with_top1(S, a)` | 照 `c1h_eval` |
| `detector(F_real, F_pseudo_list)` | 回傳 forced_cost、r_forced_cost、changed、n0 |
| `sr06_rounds(cand_games, ks, n_rounds=10)` | 登記 2.3；鹽是 `c10-prereg06|sr`、`c10-prereg06|sr-k` |
| `top5(F_or_C, cand_ids)` | 同分照候選代號 |
| `switch_top5(F, a, k_rest=3)` | 第 1 名＝Hungarian 指派 a，第 2–5 名依 CSLS3 排序、跳過第 1 名 |

每盤特徵的快取（`data/prereg06/feats/*.npz`）存這些欄位：

- `key`：`(rank, game_id)` 或測試檔的 `game_id`。
- `player_id`、`target`、`alternating`（布林）。
- `n_moves`、`s2[66]`、`ldain_B` 或 `ldain_W`（不交替的盤填 NaN，不進任何項）。
- JO token 的 CSR 矩陣，以及它自己的詞彙。

## 3. 資料（fit 玩家；10/2 實際數出來的）

| 集合 | 人數 | 盤數 | 用途 |
|---|---|---|---|
| fit 全部 | 5,767 | 720,307 | — |
| 黑棋 ≥105 盤 | 1,082 | 232,165（黑棋） | 學 LDA_B；CNN embedding（第 4 節） |
| 黑棋 ≥120 盤 | 926 | 214,767 | 官方形狀模擬池的候選，每級 40 人（各級人數見 `s0_cache_counts.json`：A 級 57 人最少） |
| 白棋 ≥105 盤 | 1,077 | 229,761（白棋） | 學 LDA_W；各級人數 D 108、C 115、B 116、A 82、1D 101、2D 115、3D 114、4D 107、5D 117、6D 102 |
| 兩色合計 ≥105 盤 | 1,937 | — | D-mix 的混色池（審查：每級 146–232 人，夠每級抽 40 人） |

- 門檻（≥105、≥120）照交替檢查前的盤數算，和探索相同；不交替的盤數在 `build_report.json` 依顏色記錄。
- 黑棋詞彙：探索算出 220 個。白棋詞彙在 build 時才算出來並記錄；會多出 i＝1 時 `RS0:…` 接 `-` 那幾種 token，估計不到 10 個。

## 4. 4090：fit 玩家的 CNN embedding（`tools/p06_cnn_fit.py`）

- **前提**：`data/prereg01_weights/player/best_model.pth` 的 SHA-256 是 `356eb4f6…7b773b`。不對就停下；K-CNN 之前修不好就改走備案。10/3 時這個資料夾還不存在。
- **環境**：`.venv-gpu`。10/2 實測的版本：
  - torch `2.11.0+cu128`、cuDNN `91900`，和 10/1 Colab 彩排記錄的版本相同。
  - TF32 用預設：卷積開、矩陣乘法關，兩者都寫進報告。
  - `cudnn.benchmark` 關。
- **程式**：和 test02 notebook 用同一套。
  - `sgf_windows`：官方教學 `utils.py` 的 `SGFParsePlayerIdentification().extract_features(sgf, 'b', mode='inference')`，轉成 uint8。
  - `embed`：`GoPlayerResNet()` 的預設值，fp32，不用 autocast，`GPU_BATCH` 4096，一盤做一次前向。
  - `emb_sum = e.sum(0)`（float32），`n_windows = len(e)`。
  - `utils.py`、`network.py` 的 SHA 核對 `third_party/aicup2026_tutorial/` 的原檔（和 `colab/build_player_nb.py` 的 `SOURCE_SHA256` 相同）。
  - 權重先整份讀進記憶體、核對 SHA 後才載入。
- **資料**：黑棋 ≥105 盤的 1,082 人的 232,165 盤，依檔案順序（和 `c1h_lib.build_cache` 一樣）。來源是 `load_rows(part="fit", local_dir=None)` 的 `sgf_content`；目標色一律是 b。CNN 不受交替檢查影響（照 test02 的做法，每盤都算）。
- **平行**：CPU 抽窗口用 6 個 worker，每次平行處理 256 盤（同 notebook 的 CHUNK）。
- **輸出**：
  - 每 8,192 盤寫一個分段 `data/prereg06/cnn_fit/shard_NNNN.npz`，欄位是 `rank`、`game_id`、`player_id`、`emb_sum[N,256]`、`n_windows`。分段的 SHA 寫進進度檔，中斷後從下一個分段接著跑。
  - 最後合成 `black_ge105.npz`，約 232,165×256×4 bytes＝238 MB。
  - 報告 `results/prereg06/design/cnn_fit_report.json`，記錄：
    - 最終檔的 SHA；
    - 盤數、沒有窗口的盤數、窗口總數；
    - 前 256 盤再算一次的最大差（預期是 0）；
    - 開跑前 `nvidia-smi` 的快照；
    - 秒數和環境。
- **時間**（`c1d_cnn_bench.json`：768 盤、6 個 worker、隨機權重；速度和權重無關）：
  - 每盤平均 106.82 個窗口。
  - CPU 抽窗口 2.18 秒，GPU 前向 7.55 秒，合計 9.73 秒，平均每盤 12.67 毫秒。
  - 推算 232,165 盤約 0.82 小時，也就是約 49 分鐘（估計），另外讀 fit 資料約 1 分鐘（`s0` 實測 45.1 秒）。
  - 只算黑棋 ≥120 盤那 926 人的 214,767 盤，約 0.76 小時。差別不大，所以全部算，讓 LDA 訓練集和快取對齊。
  - 一次塞好幾盤做前向沒有比較快：6,822 個窗口 0.844 秒，一盤一次推算約 0.628 秒。所以照 notebook 的做法，一盤一次。
  - 時間接近 1 小時，照規則排夜間。開跑前看 `nvidia-smi`，有別的工作（Codex 也算）就不跑；10/24 前 C4 優先。和 PREREG_07 共用夜間時段時，先排這一步（它決定 K-CNN）。
- **使用者**：只需要下載權重（CLAUDE.md 的「權重備份」，Drive 上那份不要刪）。

## 5. 設計步驟（照順序；每一步都 commit 報告）

指令裡的 `PY` 指 `.venv/Scripts/python.exe`；在 repo 根目錄、用 Git Bash 跑，並設 `PYTHONDONTWRITEBYTECODE=1`。

| 步驟 | 指令 | 輸出 | 通過條件 | 時間（估計） |
|---|---|---|---|---|
| S0 | `PY tools/p06_player.py leakcheck` | `results/prereg06/design/leakcheck_report.json` | `load_rows(part="fit", local_dir=data/local)` 不丟例外：fit 的人數、盤數等於 `EXPECTED`，和驗證檔的雜湊重疊 0 | 約 1–2 分鐘 |
| S1 | `PY tools/p06_player.py build` | `data/prereg06/feats/`、`build_report.json` | S0 已 commit；人數和盤數等於第 3 節；不交替的盤數依顏色記錄 | 讀 fit 約 1 分鐘；特徵約 0.9 毫秒／盤（探索 `s1_families.json`：resp 137.7 秒、style2 27.5 秒、joseki_own 21.6 秒、base 20.0 秒，都是 232,165 盤），6 個 worker 約 2–5 分鐘 |
| S2 | `PY tools/p06_player.py lda` | `results/prereg06/player/lda_{B,W}.npz`、`lda_report.json` | 兩份都有 64 個有限的方向，λ 由大到小 | 1 分鐘內 |
| S3 | `PY tools/p06_player.py selfcheck` | `selfcheck_report.json` | 登記 2.8：在交替的盤上逐值核對探索的函式；seed 0–19 的 v2 平均分四捨五入是 0.9819、0.9869；探索快取的 SHA 和不交替的盤數記錄 | 約 5–10 分鐘 |
| S4 | `.venv-gpu/Scripts/python.exe tools/p06_cnn_fit.py`（夜間） | `cnn_fit_report.json` | 第 4 節 | 約 50 分鐘 |
| S5 | `PY tools/p06_player.py tune` | `tune_report.json` | 登記 2.5：E0 單用在調參池的平均分和上限 0.50 的判定；V1 的三條條件逐條記錄；產生 V1，或寫明沒有 V1 和原因；評估池只記錄 | 40 池，每池重學 LDA 約 4 秒（`s4_terms_timing_lda.json`），加上 96 組各項，約 10–20 分鐘 |
| S6 | `PY tools/p06_player.py switch` | `switch_report.json` | 登記 2.6：每個組合 X 的 `r*_X`（每種 F 來源各一個、取較小者）和「能用」的四條判定；`k*`；評估池上每個組合的否決統計（同 `c1d_veto_check`） | 11 種結構 × 50 個 seed × 最多 4 個組合 × 2 種 F 來源，加上「只有開局＋v2」。探索的 s8（3 個特徵等級、7 種結構、50 個 seed、8 個 worker）是 1,534.6 秒，約每個（等級、結構、seed）11.7 worker 秒；照此推算約 1.5–3 小時，CPU，排夜間 |
| S7 | `PY tools/p06_player.py dmix` | `dmix_report.json` | 登記 2.8：主比較的平均差和 t 區間；只用黑棋算 v2 的版本只記錄 | 約 5 分鐘 |
| S8 | `PY tools/p06_player.py freeze` | `results/prereg06/player/frozen_config.json` | S0–S7 的報告都已 commit；工具乾淨；不准覆寫已有的檔 | 秒 |
| S9 | `PY tools/p06_player.py apply` | 第 6 節 | 第 6 節 | 驗證池 44,935 盤的特徵約 1 分鐘；sr06（4 個組合加 Q04 參考，加上只用黑棋的假題）約 2–4 分鐘 |
| S10 | 主 session 跑 H6P 的兩行指令 | `score_H6P*.json`、打分紀錄帳多 4 列 | 登記第 3 節 | 秒 |
| S11 | `results/prereg06/RESULTS.md` | — | 另一個代理獨立核對 | — |

- 模擬用的 seed：
  - 調參 300–319、評估 400–419（S5）；
  - 選門檻 500–529、評估 600–619（S6）；
  - D-mix 700–709（S7）。
  - `p06_sim.make_pool` 是探索 `c1h_lib.choose_pool`／`make_pool` 的逐字移植：題目結構 o2o、u390、u375、u350、d80、u300、wr，加上 `d<m>`（m＝1、2、3、5）；P120、每級 40 人。
  - D-mix 的題目盤數：k＝min(randint(5,20), 剩下的盤數)，照 `tools/make_val.py`。
- 有 CNN 時，E0 的候選重心用池內候選棋的 `emb_sum`／`n_windows`，算法同 p02 的 E0。
- 替身 A＝探索的 style NB（`s5b_combine_v2.py` 的 proxyA）代替 z(E0)。
- `frozen_config.json` 的內容：
  - 組合清單和順序（V2、V1（或「沒有 V1」和原因）、V3、V0）；
  - 兩份 LDA 和詞彙的 SHA；
  - 每個組合的 `r*_X`、每種 F 來源的門檻、「能用」判定，或「不用開關」；
  - `k*`（10、20、40 或「不修正」），以及開關打開時第 2–5 名用的 k＝3；
  - 所有 seed、鹽、選法的容許差 0.003；
  - 閘門的 +0.005；
  - E0 上限 0.50 和實際值；
  - S0–S7 報告的 SHA；
  - 工具的 SHA；
  - prereg02 的 dumps 和 `player_val_top5.csv` 的 SHA。

## 6. `p06_player.py apply`（本機驗證集）

- **拒絕執行的情況**：`frozen_config.json` 還沒 commit、和 HEAD 不同、工具在 freeze 之後改過、`results/prereg06/player/apply_report.json` 已經存在（要重跑得加 `--rerun --reason`，而且重跑要記帳）。
- **步驟**：
  1. 讀候選檔和題目檔：棋譜、`color`、`color_i`、`num_games`，不讀答案。
  2. 讀 prereg02 的 dumps，核對 SHA。
  3. 用 p02 的函式算 Q04（不加 v2、不做 CSLS），寫到暫存檔，和 `results/prereg02/player/player_val_top5.csv` 比，必須逐位元組相同。
  4. 算 v2 特徵。投影用凍結的 LDA_B、LDA_W，依每盤的目標色選用。不交替的盤數依顏色記錄。
  5. 跑 sr06：V2／V1／V3／V0，另外算 Q04 當作參考。照登記 2.4 選組合，寫 `sr_report.json`。
  6. 用選中的組合算全部題目的 F，寫出兩個檔：
     - `player_val_top5.csv`：做 CSLSk；
     - `player_val_top5_switch.csv`：同一個 F，Hungarian 定第 1 名，第 2–5 名照 CSLS3。
  7. 次要檔的結構檢查（不用答案）：第 1 名是 400 位不同的候選；第 1 名等於獨立重算一次的 `linear_sum_assignment`；第 2–5 名依 CSLS3 排序、不含第 1 名。
  8. 偵測器用第 0、1 輪的假題池、選中組合的 `r*_X`，結果只記錄。
  9. 只記錄的單色檢查：只用候選的黑棋做假題（其他同 sr06），算 sr(選中) − sr(V0)。
  10. 兩份都跑 `check_submission`。
  11. 寫 `apply_report.json`，記錄：
      - 選中的組合、每個組合的 sr、閘門；
      - `score_allowed` 和 `score_blocked_reasons`；
      - r_forced_cost，以及偵測器會不會打開；
      - 結構檢查和單色檢查；
      - 每色的盤數、不交替的盤數；
      - 詞彙外 token 的件數；
      - 兩份輸出的 SHA。
- **打分**：只有主 session 能做，而且要 `score_allowed` 為真、上面這些都已 commit。
- **打分後**：次要指令的 B−A 點估計 < 0 時，`RESULTS.md` 寫明「這一季不做 06-ON」，t06 的 `submit` 讀 `score_H6P_switch.json` 強制不產生 06-ON。

## 7. 測試日：`tools/t06_player.py`（分支 `p06-testday`）

### 7.1 結束碼

| 碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成 | commit 後做下一步 |
| 2 | 拒絕：前提不符 | 照訊息修好，再跑同一個指令 |
| 3 | 登記的「06 不用」：T1 FAIL_OPENING、K6' 已過、H6P 沒採用、兩池都退回 test02 | commit 報告；最終照 U5 |
| 4 | 本機漂移：seam 對不上，而且不是 dumps 不同造成的 | commit 報告後停下查；要放寬只能寫 PREREG_06 的修訂 |

06 不用 Colab，所以沒有「notebook 接縫失敗、請使用者重跑」這一類。test02 自己的重跑照 TESTDAY.md 做；test02 重跑之後，06 的輸入就變了，要從 `feats` 重做（這一點照 PREREG_04 第 11.3 節第 20 條的前例寫進 rev 3）。

### 7.2 指令

- **real 模式的共同前提**：
  - `frozen_config.json` 的 SHA 等於登記值；
  - H6P 已採用（`results/prereg06/RESULTS.md`，以及打分紀錄帳那 4 列）；
  - test02 競賽一是 real、completed、seam 通過、已 commit；
  - `results/test02/checks/player/t1.json` 是 PASS 或 NORMALIZED；
  - `results/official_scores.json` 裡有 Q04 的第一次上傳（`results/test02/player/submission_player.csv`，`uploaded`），而且已 commit；
  - K6' 還沒過；git 乾淨；`.venv` 的版本等於登記值。
- `feats --pool seam_val|public|private`：
  - 讀 `data/test`（T1 是 NORMALIZED 時讀正規化副本，MANIFEST 的 SHA 要等於 test02 記錄的）。seam_val 讀 `data/local` 的兩份檔。
  - 寫 `data/test06/player/<池>/feats.npz` 和 `results/test06/player/<池>/feats_report.json`（含不交替的盤數）。
  - 每池約 45,000 盤，約 1–2 分鐘（估計）。
- `seam`：寫 `results/test06/player/seam_val/t06_seam.json`；判定方法照登記 5.2。
- `apply --pool public|private`：照登記 2.7 的表判斷。
  - 讀 test02 那一池的 `apply_report.json`、`pool_select.json`、dumps 和 `results/test02/checks/player/t2.json`（看 unit a–d 的判定和 unit c 的 `copies`）。
  - 寫 `results/test06/player/<池>/t06_pool.json`（判斷用的表、每個組合的 sr、選中的組合、閘門、偵測器和它用的 `r*_X`、輸入的 SHA）、`top5_off.csv`，需要時再寫 `top5_on.csv`。public 池的 `top5_on.csv` 一律寫（強制探測），用 public 選中的組合的 F。
- `submit`：
  - 寫 `results/test06/player/off/submission_player.csv`。
  - 06-ON 的條件（登記 5.3）全部成立時，另寫 `results/test06/player/on/submission_player.csv`。條件之一是 `score_H6P_switch.json` 的點估計 ≥ 0。
  - 每份旁邊有 `t06_submit_report.json`，記錄：
    - 版本（off 或 on）；
    - 06-ON 每一條條件的結果；
    - `check_submission --official` 的結果；
    - 輸入的 SHA（`inputs_bound`）；
    - 和 Q04 交件檔第 1 名不同的題數，分池列出；
    - off 和 on 之間第 1 名不同的題數，分池列出（同名檔警告要用 public 那一欄）。
- `veto`：
  - 讀已 commit 的 `official_scores.json`。
  - 要有 06-OFF 和 06-ON 各一筆 `uploaded`、都有 public 分數，而且 06-OFF 在前。
  - 先跑同名檔警告（登記 5.3）：Q04、06-OFF、06-ON 任兩份 public 分數完全相同、而報告記的 public 第 1 名不同題數 > 0 時，結束碼 2，寫 `veto.json` 的 `paused: true` 和要使用者核對的項目。使用者核對後，用 `--confirmed-upload-check "<說明>"` 重跑（說明寫進報告）；核對出傳錯時照 TESTDAY 重傳，重記分數再跑。
  - 算 d，寫 `results/test06/player/veto.json`：兩筆的編號和分數、d、`vetoed`（d < 0）、最終要交的版本、還要不要再上傳一次 06-OFF。d＝0 判不否決。
- `status`：列出還差哪一步，以及 K6'（2026-11-21 23:59:59 +08:00）、K7'（2026-11-22 23:59:59 +08:00）的倒數。K7' 已過而 06-ON 已上傳、否決還沒判定時，印出「再上傳一次 06-OFF」。

### 7.3 `docs/TESTDAY_06.md` 的步驟

1. 閘門：test02 做完了沒；Q04 已上傳；T1；git 乾淨；防誤改測試（已知的例外照實寫）。
2. `feats`（三池）→ commit。
3. `seam` → commit。
4. `apply` public、private → commit。
5. `submit` → commit。
6. 上傳 06-OFF，分成兩則訊息：訊息 A 是檢查結果和 OK 請求，訊息 B 是上傳步驟。接著 `prepare`、`uploaded`、`public` → commit。
7. 有 06-ON 時，同樣用兩則訊息上傳。OK 請求寫明「這一份的 public 池是強制打開的探測」。接著 `public`、`veto` → commit。
8. 否決、K7' 已過、或同名檔警告核對不出時：再上傳一次 06-OFF（兩則訊息）。
9. `final --comp player`；11/22、11/23 各跑一次 `status`。
10. `results/test06/RESULTS.md`：由另一個代理核對；看過 public 分數才做的選擇和登記 5.5 節的例外，要寫進報告。

## 8. `tools/official_log.py`（只加行）

- **新常數**：
  - `T06_DIR = ("results", "test06", "player")`、`T06_VARIANTS = ("off", "on")`、`T06_REPORT = "t06_submit_report.json"`。
  - 兩個方法名稱。
  - `K6P`、`K7P`。
- `is_t06_file(comp, path)`：comp 是 player，而且檔案在 `T06_DIR/off` 或 `T06_DIR/on`。
- `check_reports` 最前面加一行，同 t04 的寫法：是 06 的檔就交給 `check_reports_t06` 處理。它核對：
  - 報告已 commit、是 real 模式；
  - `check_submission` 是 official 而且通過；
  - 報告記的輸出 SHA 等於這份檔；
  - `t06_seam.json` 通過；
  - 兩池的 `t06_pool.json` 都在；
  - test02 競賽一的 `submit_report.json`、`seam_local.json`、`t1.json` 已 commit 而且通過；
  - `inputs_bound` 沒有變。
- `cmd_prepare`：
  - 06 的檔要求 log 裡已有 Q04 的第一次上傳（`uploaded`）。
  - 06-ON 要求同一次 06 執行的 06-OFF 已經 `uploaded`，而且 log 裡還沒有 06-ON 的 `uploaded`（一季最多一次，登記 5.5 E1）。
  - 把 `test06` 欄位寫進這一筆紀錄。
  - OK 請求多印：這一份是 06-OFF 或 06-ON；和 Q04 第 1 名不同的題數；06-ON 另外印「public 池是強制打開的探測」。
- `cmd_public`：
  - `t06_public_lines`（比照 `t04_public_lines`）：Q04、06-OFF、06-ON 任兩份 public 分數相差 < 1e-9，而 06 報告記的 public 第 1 名不同題數 > 0 時，印 WARNING：「可能傳錯檔（三份同名）：核對 AIdea 的上傳紀錄；否決暫停」。
  - 06-OFF 和 06-ON 都有 public 分數、而且沒有警告時，印出 d 和否決的判定。這只是印出來；正式判定由 `t06_player.py veto` 寫檔。
- `cmd_final`（player）：
  - 有符合登記 5.4 的 06 紀錄時，最終只准是規則指定的那一份（`veto.json` 決定 06-ON 或 06-OFF；沒有 06-ON 或否決暫停沒解決時是 06-OFF）。
  - 要改選，必須帶 `--not-06-reason "PREREG_06 rev N: …"`，而且 N ≥ 4、修訂已 commit、修訂的時間晚於被推翻的紀錄（核對方法同 `p04_revision`）。
- `cmd_status`：log 裡有 06 紀錄時，才多印：
  - 符合條件的紀錄；
  - 否決的結果或「暫停」；
  - 最終應該是哪一筆；
  - K6'、K7' 的倒數，以及 K7' 已過時要不要再傳 06-OFF。
  - 回傳的 dict 也只在這時多一個 `player.prereg06`。
- **不變的保證**：log 裡沒有 06 紀錄時，`prepare`、`uploaded`、`public`、`final`、`status` 的輸出、紀錄檔和回傳值，都要和 tag `p04-testday-pre06` 的版本逐位元組相同。由 `tests/test_t06_official_log.py` 的 TagVersusNew 核對：競賽一和競賽二的 R20／R10／04 流程各跑一遍。
- 只用標準函式庫，不 import `t06_player`。

## 9. 測試和演練

### 9.1 main（freeze 之前）

- `test_p06_style.py`：
  - 用合成的黑棋，核對和探索的函式逐值相同：`f_joseki_own`（去掉顏色前綴後）、`style2_vec`、`f_resp`、`c1h_lda.fit`（≤1e-10）、`csls`、`with_top1`、`hungarian_top1`、`s8` 的偵測統計量。
  - 用手算的小棋盤核對白棋的定義：自己的手是奇數編號，`RS0:…|-` 這類 token 會出現。
  - 交替檢查：連下兩手同色、第一手是白、中間有 pass（MOVE_RE 取不到）的盤，`alternates` 是假，不進任何 v2 項，計數正確。
  - 全黑時，每色合併等於 z(T_B)。
  - sr06：結果固定、5 份互不重疊、參考是 80 盤、K 的洗牌固定。
  - `switch_top5`：第 1 名是排列；第 2–5 名不含第 1 名、依 CSLS3 排序。
- `test_p06_sim.py`：在合成的快取上，`make_pool` 和探索的版本逐池相同（o2o、u390、d80、wr、d1、d5）；D-mix 的池照 make_val 的 k＝min(randint(5,20), 剩下的盤數)。
- `test_p06_player.py`：
  - 拒絕的情況：freeze 之前跑 apply；leakcheck 以外的設計步驟碰到 data/local；檔名含答案字樣；覆寫凍結檔；smoke 模式寫進 repo；`build` 在 leakcheck 沒 commit 時。
  - 選法：V1 和 V2 差 0.003 以內時選 V2；V1 的評估池條件不過時沒有 V1；E0 單用超過 0.50 時沒有 V1、門檻改用替身。
  - 門檻：每個組合各自校準；平手取最小；兩種 F 來源取較小者；`k*` 依序試 10、20、40。
  - 在 smoke dumps 上，Q04 的重現路徑正確。
  - 輸出格式和 `check_submission` 相容。

### 9.2 H6P 之後

- 新增 `test_p06_player_hygiene.py`：
  - 釘住 `results/prereg06/player/**` 的 SHA。
  - 打分紀錄帳的資料列＝`6ec5e68` 的 25 列，加上 note 開頭是 `prereg06 H6P` 的 4 列；PREREG_07 的列由它自己的測試釘住，兩邊要協調。
  - PREREG_06 只准加行。
- H6P 之後，現有的 `C10_T04_GATE=1` 閘門裡，`test_ledger_and_prereg_rows_pinned` 會固定失敗，因為打分紀錄帳多了列。這是 PREREG_04 rev 3 第 19 條允許的情況（「只多了打分紀錄帳的列」）。CLAUDE.md 和 AGENTS.md 的閘門指令要加上 `tests/test_p06_hygiene.py`，「全過」的說法兩份一起改成例外清單（H6P、H7R 之後各自列入）。

### 9.3 分支 `p06-testday`：合併前的演練（K5'）

1. **Q04 逐位元組不變**：寫 t06 之前，先用 `t02_fake_tests.py from-val` 的假樹跑 test02 競賽一的 dry-run，把 `submission_player.csv`、三池的 `top5.csv`、報告（去掉絕對路徑）的 SHA 記進 `q04_fake_golden.json`。合併當下重跑，要相同。
2. **R20 和 04 不變**：三支防誤改測試加上 `test_p06_hygiene.py` 全過；TagVersusNew 通過。
3. **06 對得回 H6P**：在 from-val 的假樹上，public 和 private 都是驗證池換了代號。`t06_player.py` 跑完整套：
   - 兩池選中的組合和 H6P 相同；
   - `top5_off.csv` 換回代號後和 `player_val_top5.csv` 逐列相同；
   - `top5_on.csv` 和 `player_val_top5_switch.csv` 逐列相同；
   - seam 是逐位元組相同。
4. **官方形狀**：
   - 用 `t02_fake_tests.py from-train --shape official` 的假樹，加上 `t06_fake_tests.py add-dumps` 的假 dumps（第 4 節的 embedding 存在時用它）。
   - 確認全黑、5／10／15／20 的處理正確；這是一題對一人的池，偵測器應該打開，會產生 06-ON。
   - 再做一個「同一人出兩題」的變體：T2(c) > 0 或偵測器關著，不會產生 06-ON。
5. **負面演練**（每一項都檢查結束碼）：
   - T1 FAIL_OPENING → 3；
   - 某一池防護在第 3 層 → 那一池的列等於 test02 的；
   - 防護在第 2 層 → 不產生 06-ON；
   - T2(b) 不過 → 不產生 06-ON；
   - 選中的組合「不能用」→ 不產生 06-ON；
   - `score_H6P_switch.json` 點估計 < 0 → 不產生 06-ON；
   - 06 的閘門沒過 → 那一池等於 test02 的；
   - K6' 已過 → 3；
   - H6P 沒採用 → 3；
   - test02 沒 commit、Q04 還沒上傳 → 2；
   - 06-OFF 還沒上傳就 prepare 06-ON → official_log 拒絕；第二次 prepare 06-ON → 拒絕；
   - 06-ON 和 06-OFF 的 public 分數相同、public 第 1 名不同題數 > 0 → `veto` 結束碼 2、official_log 印警告；06-ON 和 Q04 相同時也一樣；
   - d<0 時把最終設成 06-ON → 拒絕；
   - d≥0（含 d＝0）時把最終設成 06-OFF → 拒絕；
   - K7' 已過、06-ON 已上傳、否決沒判定 → `status` 要求再傳 06-OFF，`final` 只准 06-OFF；
   - 帶了合格修訂的翻盤 → 通過。
6. 完整測試、`C10_SLOW=2`、防誤改閘門都過；再從三個角度做獨立審查（正確性、紀律、使用者步驟），紀錄寫進 `results/test06/design/build_review.md`；rev 3 commit；合併。

## 10. 時程和工作量

| 日期 | 內容 | 工作量（估計） |
|---|---|---|
| 10/03–10/05 | commit PREREG_06 rev 1（04 指紋已在 `d90b9be` 完成） | — |
| 10/05–10/07 | 寫 `p06_*` 和測試、審查、rev 2；S0–S3 | 約 1–1.5 天 |
| 10/06 或 10/07 夜 | S4（4090，先於 PREREG_07 的 GPU 工作） | 約 50 分鐘 |
| 10/08–10/09 | S5–S8（S6 排夜間，CPU） | S5 約 10–20 分鐘、S6 約 1.5–3 小時，加上檢查 |
| 10/10 前後 | S9–S11（H6P） | 半天 |
| 10/11–10/20 | 分支 `p06-testday`：t06、official_log 只加行、TESTDAY_06、測試、演練、審查、rev 3、合併 | 約 2–3 天（04 的測試日擴充實際花了 32.13 小時） |
| 11/04–11/06 | test02 → 06 | CPU 約 15 分鐘；上傳 2–3 次 |

## 11. 風險

- **CNN 樣本內**：fit 模擬裡 CNN 偏強，V1 的權重會偏小、門檻可能偏寬。補救：V2 排在 V1 前面、E0 上限、V1 的評估池條件、門檻取真實 CNN 和替身 A 較小者；官方池用每池自我檢索選組合。
- **白棋**：LDA_W 只在本機驗證集用到，沒有在模擬裡驗證過，D-mix 只是粗略檢查。H6P 因此可能比官方情境難過；那是保守的方向。
- **H6P 檢定力**：真實 +0.02 時估計只有約一半到三分之二的機會通過（登記第 3 節）。
- **public 否決的前提**：public 和 private 的出題方式相同。如果不同，否決可能錯：開關該開卻被關掉，或該關卻開了。private 的偵測器仍然要先同意才會開，所以兩道關卡都要錯，才會造成損失。
- **傳錯檔**：三份交件檔同名。同名檔警告（第 7.2、8 節）只抓得到分數完全相同的情況；不同檔傳錯但分數不同時抓不到，所以每次上傳前的 OK 請求要寫明檔案路徑和 SHA，使用者截圖核對檔名和時間。
- **時間**：K7' 只比最後上傳日早一天。test02 競賽一如果拖到 11/20（拿不到 G4，改用 A100），06 還有 1–2 天可以做，但很緊。
- **打分紀錄帳**：PREREG_07 的 H7R 和 06 的 H6P 都會加列，防誤改測試的列數要一起協調。
- **登記外的規則例外**：06-ON 依賴登記 5.5 節對 TESTDAY、VALIDATION、PREREG_02、DATA.md 的例外。報告審查時要能說清楚「為什麼不是挑版本」。
