# prereg17 探索：TabICLv2 當 Q09 集成的第 7 個成員（競賽二，inner-only）

2026-10-18 寫成，**在算出任何候選數字之前 commit**（分支 `p17-explore`）。只用 fit 玩家（probe）當 TabICL 的上下文、只用 inner 訓練玩家評估；不碰 `data/local`、不讀任何驗證題特徵、不跑 `tools/score.py`、不跑新的 KataGo。本檔的閘門在正式執行前寫死，正式執行後只判一次（`eval_report.json` 原樣 commit）。

## 0. 白話版

- Q09 的翻譯模型是 6 個小神經網路／LR 的等權集成（PREREG_09 第 1.3 節）：每一盤的 1,540 個 KataGo 探針數字 → 10 級的機率，6 個機率平均再取 log，乘 4 放進融合。
- 這次問：**再加一個完全不同類型的模型 TabICLv2 當第 7 個成員，能不能多分辨出等級？** TabICL 是預先訓練好的表格「基礎模型」：不用訓練，直接把一批「已知等級的盤」當範例（in-context）餵進去，它就對新的盤輸出 10 級機率。和現有 6 個成員的錯誤型態可能不同，集成才有機會變好。
- 做法：1,540 維先用 PCA 壓到 128 維（PCA 只用 probe 的 27 萬盤擬合），從 probe 抽 30,000 盤（每級 3,000）當範例，對 inner 的 73,169 盤各算一次 10 級機率；和 6 個成員的機率等權平均（7 個），其餘照 Q09 原樣融合、規則 B。
- 先講預期：Q09 的 6 個成員吃的是同一份特徵、用的是 27 萬盤全部；TabICL 只看 3 萬盤的 128 維壓縮版。過關機會不高；閘門照舊 +0.010。
- 競賽二已經沒有驗證集打分機會：就算過關，採用也只能像 09 一樣靠 inner，而且要另寫 PREREG_17 和測試日路徑（本探索**不做**，只寫筆記，由主 session 決定）。

## 1. 下載和環境（使用者 10/17 核准的項目，只有這些）

| 項目 | 來源 | 位置 | SHA-256 |
|---|---|---|---|
| `tabicl-2.2.0-py3-none-any.whl`（310,448 bytes，BSD-3） | PyPI | `data/prereg17/downloads/pypi/` | `c80b8ec2719be26bc0bf1dd758a2361f58019d8f7877038b785529a425789ce4` |
| `tabicl-classifier-v2-20260212.ckpt`（110,368,038 bytes，BSD-3） | Hugging Face `jingang/TabICL` | `data/prereg17/downloads/hf/` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |

- 環境：repo 外的 `C:\Users\USER\venvs\c10-tabicl`（uv，Python 3.12.13），**不動 `.venv`／`.venv-gpu`**。torch `2.11.0+cu128`（官方 PyTorch 索引，和 `.venv-gpu` 同版本，uv 快取）、numpy 2.5.3、scipy 1.18.1（同 `.venv`）、tabicl 2.2.0 和 PyPI 上的必要依賴（scikit-learn 1.9.1、einops 0.8.2、huggingface-hub 2.2.0、psutil 7.2.2、tqdm 4.70.1 等；完整清單由 `p17_tabicl.py` 寫進 `run_record.json`）。
- checkpoint 用 `torch.load(weights_only=True)`（tabicl 自己的載入方式）；`allow_auto_download=False`、`model_path` 指定上表的檔，執行前核對 SHA。TabICL 部分一律用 `python -I` 執行（不從工作目錄或腳本目錄載入其他模組）。
- 10/17 只做了**合成資料**的可行性測試（隨機常態特徵、合成等級，沒有用任何比賽資料）：30,000 筆上下文、128 維，4090 上每 2,000 筆預測約 20～100 秒、峰值約 21 GB；50,000 筆上下文在沒有磁碟卸載時記憶體不夠。同一個擬合、同樣的分塊，重複預測逐位元相同；分塊大小不同時機率差到約 2e-3（混合精度）→ 分塊大小寫死。

## 2. 候選 Q09＋T（唯一一個）

### 2.1 特徵（寫死）

1. 原始輸入：`tools/p09_map.py` 的 `build_prep` 重建 09 的 prep 陣列（P1＋P2＋P3 probe 270,000 盤、inner 73,169 盤）到 `data/prereg17/prep/`；做法和 09 `prep` 相同（chunk SHA 對 08 的 `inner_manifest.json` 和 09 的 `p3_manifest.json`、重複盤排除照 manifest）。**接縫 P**：六個檔的 SHA 等於已 commit 的 `results/prereg09/rank/check_report.json` 的 `prep.files_sha256`。
2. 標準化：`p09_forward.standardise_x`，μ、σ 用 09 凍結的 `p1e_map.npz`（＝270,000 盤 probe 的矩，和 6 個成員看到的完全相同）。EX 不用（只用 1,540 維）。
3. PCA（k＝**128**）：只用 270,000 盤 probe 的標準化 X（float64）。平均 μ_P、共變異 C＝(ΣxxT − n μ_P μ_PT)/n（分塊累加，float64）、`numpy.linalg.eigh`，取最大的 128 個特徵值的向量（由大到小）。符號：每個向量絕對值最大的那一個分量取正。投影 Z＝(X_std − μ_P) V，存 float32（TabICL 本來就用 float32）。解釋變異比例只記錄。
4. 上下文：probe 列，每級 **3,000** 盤（共 **30,000**），`numpy.random.default_rng(20261017)`，依 0…9 級依序對該級的 probe 列（儲存順序）`rng.choice(…, 3000, replace=False)`，合起來排序。

### 2.2 TabICL（寫死）

- `TabICLClassifier(n_estimators=8, norm_methods=None, feat_shuffle_method="latin", class_shuffle_method="shift", outlier_threshold=4.0, softmax_temperature=0.9, average_logits=True, support_many_classes=True, batch_size=8, kv_cache=False, model_path=<上表 ckpt>, allow_auto_download=False, checkpoint_version="tabicl-classifier-v2-20260212.ckpt", device="cuda", use_amp="auto", use_fa3="auto", offload_mode="auto", random_state=0)`（除了 `model_path`、`allow_auto_download`、`device`、`random_state` 以外都是 2.2.0 的預設值）。
- `fit(Z_ctx, y_ctx)` 一次；`classes_` 必須是 0…9。
- 預測：inner 73,169 盤依 prep 的 inner 列順序，每 **2,000** 列一塊（37 塊，最後一塊 1,169 列），`predict_proba`；每塊存 `data/prereg17/tabicl/chunk_XXXXX.npz`（原始 float32 機率、列範圍、輸入 SHA）。中斷後重開：已存在且核對通過的塊保留，從下一塊接著算（每次開程序都重新 `fit`；同一擬合同一分塊逐位元相同，跨程序是否逐位元相同只記錄）。
- 機率轉 float64、每列除以列和（列和偏離 1 的最大值記錄）→ p_T。
- GPU：4090。每次開跑前 `nvidia-smi`：驅動必須是 616.92；有其他計算程序（任何 python 在 GPU 上）就等、重試，不搶。預計 20～90 分鐘，一律用獨立背景程序（`Start-Process` 跑 `run_p17_tabicl.sh`，有 STOP 檔、可分塊接續），不綁在對話上。

### 2.3 集成和融合（照 Q09，只多一個成員）

- 6 個成員的 inner lp：`p09_rank_fuse.inner_map_lp(prep, p1e_map, with_members=True)`（numpy float64 前向）。
- **lp_E7＝log( max( (Σ_{6 成員} exp(lp_m) ＋ p_T) / 7, 1e-300 ) )**（等權，下限同 `p09_forward.lse_mean`）。
- 融合：`Inner.run(Q08_TERMS, {"HS": E_hs, "P1TO": E_p1(lp_E7)})`，也就是 Q09 的同一條路徑：×4、項的順序 OP、O4、SWA、SWB、HS、P1TO、同一組界限／起點／PENALTY／MAXITER、等級加權、5 折玩家 OOF（inner 的玩家折，和 Q09 完全相同）、規則 B、PlayerBootstrap。

### 2.4 為什麼不會洩漏（交叉擬合）

- TabICL 的上下文和 PCA 只來自 probe（fit 玩家），和 6 個成員的訓練資料同一批；inner 玩家（所有評估假題、單色假題的玩家）從來不在上下文或 PCA 裡。**接縫 C**：上下文的玩家（`等級:player_id`）和 inner 所有盤的玩家交集是 0（程式內斷言，報告記錄；只比 player_id 的交集只記錄）。
- 融合參數照 Q09 的 inner 協定：每一折的參數只用其他 4 折玩家的擬合假題擬合，評估那一折的題（`Inner.run`，不改）。
- 所以每一題的機率都來自「沒看過該題玩家」的上下文、PCA 和融合參數，和 Q09 的評估方式一模一樣。

## 3. 接縫（先過，否則停，不算任何候選）

1. Q02H、Q08、Q08 單色等於 `p09_rank_fuse.REF`；Q09（inner 0.7114420614163589）、Q09 單色（0.7145431684624712）等於 `results/prereg09/rank/tune_report.json`；相差都 ≤ 1e-12。
2. **接縫 P**（2.1 第 1 點）；重建的 `inner_X.npy` 和 `data/prereg12/q09_inner/inner_X.npy`（`Inner` 用的那份）逐位元組相同。
3. **接縫 M**：6 個成員的 lp 照 `lse_mean` 平均，和 Q09 的 lp_P1TO_E 逐位元相同。
4. **接縫 C**（2.4）。
5. **接縫 T**：37 塊都在、列範圍連續涵蓋 0…73,168；每塊的輸入 SHA（Z_ctx、y_ctx、Z_inner、ckpt）等於 `feat_record.json`／第 1 節；機率全部有限、非負；`classes_`＝0…9；`run_record.json` 的 TabICL 參數等於 2.2；tabicl、torch 版本等於第 1 節。

## 4. 閘門（單一候選 Q09＋T，正式執行後只判一次；全部成立才算過）

- **G1**：inner 評估假題的分等級平均差（Q09＋T − Q09）≥ **+0.010**（嚴格，不加容許；inner 已被 P1、08、09、prereg12、15、16 重複使用）。
- **G2**：同一差值的配對 PlayerBootstrap（`Inner.compare`：每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
- **G4**：單色假題（120,544 題）的分等級平均差，配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 ≥ 0（容許 1e-12）。
- 只有一個候選，不做多重比較校正。
- **只記錄、不能救候選**：TabICL 單獨放在 P1TO 位置（×4，`Inner.evaluate(log p_T)`）的 inner／單色分數；每折融合權重；PCA 前 128 維的解釋變異比例；上下文的玩家數；GPU 秒數；p_T 列和偏離 1 的最大值。

## 5. 判定和之後

- **只判一次**：`p17_eval.py eval` 正式跑一次，寫 `results/prereg17/explore/eval_report.json`，原樣 commit。開始時先寫 `data/prereg17/eval_started.json`，中途中斷也算跑過；要重跑只准在指出具體程式錯誤、比上一次更新的 commit 之後（`--rerun --reason`），報告要記。TabICL 的 GPU 執行也只做一次（中斷接續不算重跑；要重算必須同樣先 commit 指出錯誤）。正式執行要求本探索的檔和 import 的工具都已 commit 且乾淨，SHA 記進報告。
- **smoke**（正式執行前，輸出全部在 `data/prereg17/smoke/`，不是結果）：PCA 特徵換成 `default_rng(12345)` 的常態雜訊（上下文、inner 都換，形狀不變），上下文只取登記樣本每級前 100 盤（1,000 盤），TabICL、分塊、接續、融合、閘門的整條流程跑完；只確認程式能跑、接縫 1～5 過、時間多長。數字沒有意義、不記進結果。
- **都沒過**：記錄後收掉，Q09 照舊。
- **過了**：只代表「值得寫 PREREG_17」，不是採用。只寫一份筆記，說明 PREREG_17 和測試日路徑需要什麼（本探索**不做**）：凍結 PCA、上下文、ckpt 和環境；測試日要在 GPU 上對測試盤跑 TabICL（同分塊）、GPU 接縫容許差、測試日工具 `t17`、假資料演練、指紋和 tag、官方紀錄只加行，11/04 前做不完就不採用。
- **審查**：正式執行前，程式由另一個子代理獨立審查（只讀程式、跑單元測試）；修完才跑。審查結果寫進第 7 節。

## 6. 檔案

- `p17_lib.py`（純 numpy：PCA、上下文抽樣、7 成員平均、分塊、閘門）、`p17_prep.py`（`.venv`：重建 prep、接縫 P、標準化、PCA、上下文、寫 `data/prereg17/feat/`；`--smoke` 寫雜訊版）、`p17_tabicl.py`（`c10-tabicl` 環境、`python -I`：TabICL 擬合和分塊預測）、`run_p17_tabicl.sh`（背景啟動器：`nvidia-smi` 檢查、STOP 檔、重試）、`p17_eval.py`（`.venv`：接縫、融合、閘門、報告）、`test_p17.py`（合成資料的單元測試）、`eval_report.json`（正式執行後 commit）、`RESULTS.md`（判定和筆記）。
- 大檔、下載、smoke 輸出在 `data/prereg17/`（不進 git）；`data/prereg17/prep/` 的 probe 陣列（約 1.7 GB）判定後可刪。只 import、不改：`tools/*`、`results/prereg05/explore/u1/u1_lib.py`、`results/prereg12/explore/finetune/ft_eval.py` 等（同 prereg16）。
- 環境：CPU 部分 `.venv`、`PYTHONDONTWRITEBYTECODE=1`，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。worktree 在 scratchpad，用 `subst V:` 避開 u1_lib 的 `data\local` 防護誤擋（同 prereg15／16；`W:` 被另一個代理用著，不碰）。

## 7. 實作補記（正式執行前）

- 程式：`p17_lib.py`、`p17_prep.py`、`p17_tabicl.py`、`run_p17_tabicl.sh`、`p17_eval.py`、`test_p17.py`（合成資料 15 個測試全過）。閘門、PCA 維度、上下文、TabICL 參數、分塊都沒改。
- prep：09 的 prep 陣列重建（硬碟讀取慢，約 28 分鐘），**接縫 P 過**（六個檔的 SHA 等於 `check_report.json`，`inner_X.npy` 和 prereg12 那份相同）。PCA 前 128 維解釋 85.3% 的變異（只記錄）。上下文 30,000 盤（P1 3,407／P2 13,256／P3 13,337 盤，5,397 位玩家）。
- smoke（`data/prereg17/smoke/`，特徵換成雜訊、上下文 1,000 盤）：TabICL 37 塊約 1 分鐘；STOP 檔、分塊接續都正常；刪掉兩塊後在新的程序重算，機率逐位元相同（跨程序可重現，只記錄）；eval 的接縫 1、M、P、C、T 全過；全程約 3 分鐘。數字無意義，不記。
- 獨立審查（另一個子代理，只讀程式並跑單元測試，沒跑 prep／TabICL／eval）：沒有 blocker。正式執行前都已修：
  - major：GPU 忙碌判斷只看 python → 改成 python 或 katago 的計算程序，或 GPU 已用記憶體 > 4,000 MiB 就等；每塊記錄 `torch.cuda.mem_get_info()`（TabICL 依可用記憶體決定內部批次，可能影響低位數，只記錄）。
  - minor：驅動不同改成結束碼 6、啟動器直接停（不重試 4 小時）；docstring 的結束碼；`pca.npz` 原子寫入、讀回時核對形狀和有限值（smoke 也會用到真的 PCA，PCA 是確定性的前處理、不含標籤）；eval 核對 feat 的 smoke 旗標、重新推導上下文列和 y；`code_state` 多列 import 到的 p02 工具和讀到的凍結檔；接縫 C 加防呆（玩家鍵為空或等級前綴對不上就停）；STOP 檔要手動刪才能重開（寫進啟動器說明）。
