# prereg30 探索（第 0 階段）：P1 v2 逐手特徵有沒有 P1 以外的新資訊（競賽二）

2026-11-09 寫成，**在算出任何特徵值或模型數字之前 commit**（分支 `p30-explore`）。主 session 交辦（使用者 10/2、10/9、10/16 把方法決定交給 AI）。這是第 0 階段：只判「值不值得做全量」，不寫 `docs/PREREG_30`、不做測試日工具、不上傳。不碰 `data/local`（不開、不讀、不列、不 stat、不雜湊，指令裡也不寫這個路徑）、不跑 `tools/score.py`、不用 inner 和驗證玩家的任何資料。

## 0. 白話版

- 來源：`data/research/reports/2026-11-09 新資源調查.md` 第 1 項，Kuboki、Hsueh、池田（JAIST）GPW-2024〈ゲームの展開に過度に影響されないプレイヤ強さの推定方法〉（https://dspace.jaist.ac.jp/dspace/bitstream/10119/20018/1/T-IKEDA-K-0930-7.pdf ）。本代理 11/09 讀了原文（第 3–6 節、表 1–4），三個技巧照原文是：
  1. **正規化 prior**（原文「幾何平均 2」）：這一手的 prior 除以該局面所有手的最大 prior，再對玩家的手取幾何平均。原文表 3：7 個 HumanSL 段位設定全部比不正規化好（1d：0.983 → 0.903）；prior 類用整盤比只看前 150 手好。
  2. **損失只看前 150 手**：原文表 2：不過濾時整盤 1.759、前 150 手 1.609。
  3. **雙方連續大損失只算第一手**（原文「フィルタ 2」）：雙方連續下出損失 ≥ n 目的手時，這段連續區間裡**每一方只留第一手**，其餘從平均裡拿掉（原文表 1 的例子：第 3–6 手連續 ≥ 5 目，拿掉第 5、6 手）。n＝3、5、10，最好是 n＝5、前 150 手（表 4：0.618）。原文另有「依手數加權」的フィルタ 1，和フィルタ 2 合用時不再需要（表 4），**本計畫不用**。
- 我們的 P1（1,540 維）每手只存「這一手的 log p」和「熵」（另有 09 的 EX 44 維：分手數段的 log p 統計），**沒有**除以最大 prior，也**沒有**任何損失特徵。
- 這一階段只問一件事：**在 S25 已經看過 P1 之後，v2 還能不能讓預測變好？** 用 S25 留出玩家（只用來選步、沒進訓練的 fit 玩家）的 probe 棋，在 4090 上用「P1 同一次前向」多存幾個數（約 1.3 小時），再用 CPU 幾分鐘判一次。過了才規劃全量（約 19 小時 4090）；沒過就停，25 照舊。
- 先講預期：調查報告推估 +0.002～+0.008，而且我們已有熵、EX、07 的 HumanSL 逐段位機率；**「沒過」是很可能的結果**。

## 1. 特徵定義（寫死）

### 1.1 每個局面多存的量（和 P1 同一次前向，不加搜尋、不加 profile）

P1 的查詢不變：`results/prereg08/p1/p1_lib.py` 的中性 metadata（雙方段位不明、KGS、2020-03-01、byo-yomi 1200 s＋5×30 s）、日本規則貼 6.5、HumanSL `b18c384nbt-humanv0.ckpt` 的 `model` 權重、每個局面（雙方的每一手）前向一次、不對稱平均。這等於 KataGo 的 maxVisits 1（只看網路輸出）。同一次前向的輸出裡多讀三個量（都已經算出來，不多花前向）：

| 鍵 | 定義 |
|---|---|
| `pos_lmax` | 主 policy 頭第 0 通道、非法手遮成 −∞ 後 log-softmax 的**最大值**（＝log max prior）。`pos_logp`（P1 原有）－`pos_lmax` ＝ log(prior／max prior) ≤ 0 |
| `pos_lead` | 主頭 `miscvalue[:, 2] × lead_multiplier`（KataGo 的 scoreLead，**輪到下的一方**的觀點，單位：目） |
| `pos_winp` | 主頭 `softmax(value_logits)[:, 0]`（輪到下的一方的勝率） |

- 實作：`p30_lib.V2Scorer` 是 `p1_lib.ProbeScorer` 的子類，P1 的 770 維算法一行不改（CPU fp32 單元測試要和 `ProbeScorer` 逐位元相同）；value／lead 的讀法要和 KataGo 自己的 `GameState.get_model_outputs`（`model.postprocess_output`）在 CPU fp32 上相同（≤ 1e-4）。
- 和原文的差別（揭露）：原文用 KataGo **搜尋後**的 scoreLead 算「最佳手 − 實戰手」；我們只有網路一次前向，損失用「這一手前後兩個局面的 lead 差」近似。原文的 prior 用 `preaz_<段位>` 的 10 個段位設定；我們只用 P1 的中性設定（多一個段位設定就多一倍 4090 時間；段位方向的資訊 07 的 HS 項已經有逐段位的 log p）。

### 1.2 每盤的 v2 特徵（CPU，從逐局面的量算；10 維）

一盤的局面依手數 t＝0, 1, …, T−1（0 起算的第幾手，`pos_turn`）排列。P1 兩方都查，所以每一手都有局面；**要求** `pos_turn` 正好是 0…T−1（不是就整盤 v2 記為缺值）。

- 正規化 prior：nlp_t ＝ logp_t − lmax_t（≤ 0）。
- 損失（t ≤ T−2，第 t＋1 個局面存在）：d_t ＝ lead_t ＋ lead_{t+1}（下這手的人估計丟掉的目數；lead_{t+1} 是對手觀點，所以用加的）；勝率損失 w_t ＝ winp_t ＋ winp_{t+1} − 1。最後一手沒有損失。
- 大損失：big_t ＝ (d_t ≥ 5.0)（原文 n＝5 目）。
- **連續大損失過濾**：在有損失的手裡，找出手數連續、全部 big 的區間；每個區間只留前兩手（＝雙方各自的第一手），其餘標成「不計」。先在整盤上判，再限制 t < 150。
- 計入集合 C_s：side s 的手、t < 150、有損失、沒被過濾掉。

每一方（s＝0 目標方、1 對手方，同 P1 的 side），5 個特徵：

| # | 名稱 | 定義 |
|---|---|---|
| 0 | `nlp_all` | side s 所有手的 nlp_t 平均（＝log 幾何平均 2，整盤） |
| 1 | `nlp_150` | side s 中 t < 150 的手的 nlp_t 平均 |
| 2 | `loss_150f` | C_s 上 max(0, d_t) 的平均 |
| 3 | `big_150f` | C_s 上 big_t 的比例 |
| 4 | `wloss_150f` | C_s 上 max(0, w_t) 的平均 |

v2 ＝ [目標方 0–4, 對手方 0–4]，float64；集合是空的就是 NaN（標準化後當 0，同 P1 的 NaN 處理）。

## 2. 試跑資料（4090 約 1.3 小時，等主 session 明確同意才跑）

- **只用 fit 玩家**：S25 的留出玩家 H（`results/prereg25/explore/p25_lib.holdout_bit`，鹽 `c10-prereg25|holdout|`）中屬於 probe 的 **588 位**，他們在 probe（`data/prereg17/prep/probe_meta.npz` 的 270,000 列）裡的**全部 26,779 盤**。盤點（唯讀，舊 chunk 的 counts）：5,924,986 個局面（每盤 221.3），照 P1 實測 1,234 局面／秒約 **1.33 小時**。
  - 為什麼用 H：S25 只用 H 選步（30 個檢查點挑留出 NLL 最低的一個），H 的棋從沒進過權重更新，所以 S25 在 H 上的輸出近似樣本外，可以直接問「S25 看過 P1 之後，v2 還多什麼」。選步的輕微偏差對 M0、M1 一樣（兩者都含 S25）。
  - 不用：調查報告建議的 bank（11,813 盤）是 **inner** 棋，本階段不用；驗證玩家（V）不用；`data/local` 不碰。
  - 每級玩家／盤數：D 69／2,978、C 57／2,414、B 71／2,363、A 107／2,262、1D 30／2,506、2D 43／2,587、3D 58／3,272、4D 52／2,798、5D 48／2,815、6D 53／2,784。
- `p30_data.py sets`：照 probe_meta 的列序，從原 sets.json（`data/prereg08/p1/sets.json`、`data/prereg08/p2/sets.json`、`data/prereg09/p3/sets.json`，依 `src`）取 {rank, game_id, player_id, color}，核對玩家鍵；寫 `data/prereg30/pilot/sets.json`（p1_extract 的 sets 格式，probe 的 `source` 是 `fit`）和 `pilot_map.npz`（試跑第 j 盤 → probe 列號、src、ref）。
- 抽取：`p30_extract.py`（包 `p1_extract.main`，`p1_extract.py` 一字不改；只換 scorer 和 `run_chunk`，chunk 多存 `pos_lmax`、`pos_lead`、`pos_winp` 和格式標記 `p30_format`），設定同 P1／prereg25：fp16、batch 1024、group 4、channels-last、workers 12、每 200 盤一個 chunk（134 個）、原子寫入、已完成的 chunk 自動跳過（可續跑）、`STOP` 檔讓出 GPU。獨立背景程序 `run_pilot.sh`（先 `nvidia-smi`：驅動 616.92、沒有別的運算程序，否則不跑）。

## 3. 接縫（先過，否則停，不判）

- **接縫 S（P1 重算）**：試跑 chunk 的 P1 部分（means、counts、failed、n_moves）和原 probe chunk 的同一盤相同：failed、counts、n_moves 完全相同；以 09 `p1e_map.npz` 的 μ_X、σ_X 標準化後，和 `probe_X.npy` 對應列的每個特徵最大差 ≤ **0.05**（PREREG_08 5.2 的 `TOL_FEAT`，同 prereg25 接縫 G）。只記錄：是否逐位元相同。
- **接縫 C（成本）**：只記錄試跑的局面／秒，和 P1 的參考值（prereg25 V 抽取平均 1,234）比；全量時數用試跑的實測速度重算。
- **接縫 K（KataGo 讀法）**：單元測試（CPU fp32，幾個局面）：V2Scorer 的 P1 部分和 `ProbeScorer` 逐位元相同；`pos_lead`、`pos_winp` 和 KataGo `get_model_outputs` 的 lead、value[0] 差 ≤ 1e-4；`pos_lmax` 等於遮罩後 log-softmax 的最大值。

## 4. 增量資訊檢定（CPU，幾分鐘；正式執行只判一次）

### 4.1 假題

- 可用盤：P1 可用（沒失敗、至少 1 手目標手）。v2 缺值照 1.2 處理（不影響可用與否）。
- 用 `p19_lib.PlayerPool` 和 `sample_question`（import 不改）在試跑的可用盤上出單色假題：n ∈ {5, 10, 15, 20}、10 級，每格 300 題，`np.random.default_rng(30030)`；沒有合格玩家的格子照 `holdout_set` 的規則跳過並記錄。
- 5 折以玩家分：fold ＝ int(sha256("c10-prereg30|fold|<rank>:<player_id>")[:8], 16) mod 5。

### 4.2 兩個模型（同一套設定，只差有沒有 v2）

- **S25 項**：凍結的 3 個 S25 模型（`results/prereg25/rank/s25_seed{0,1,2}.npz`，`tools/p25_forward.load_models` 核對登記的 SHA），輸入 probe_X 的原列（S25 看到的就是它），以 09 `p1e_map` 的 μ_X、σ_X 標準化；每題 lp25 ＝ 3 個種子「題內 h 平均 → head」的機率平均取 log（`tools/p19_forward` 的 `game_h`、`head_np`、`FW.lse_mean`，同 Q25 的 SET 項）。
- **z**：題內各盤 v2 的平均（忽略 NaN；全 NaN 是 NaN），用**訓練折**題目的平均和標準差標準化，NaN → 0。
- **M0**：logit_k ＝ α·lp25_k ＋ β_k（β_0 ＝ 0）。
- **M1**：logit_k ＝ α·lp25_k ＋ β_k ＋ Σ_j z_j·(u_j·τ_k ＋ v_j·τ_k²)，τ_k ＝ (k − 4.5)／4.5。
- 擬合：最小化訓練折題目的平均 NLL ＋ λ(‖u‖² ＋ ‖v‖²)，λ ＝ 1e-3；scipy L-BFGS-B（float64，起點 α＝1、其他 0，maxiter 2000，解析梯度）；確定性。
- 每折擬合、預測測試折，拼成 OOF 機率；等級用規則 B（`tools/decide.b_scores` 的同一公式，取最大、平手取第一個）。

### 4.3 指標

- Score ＝ 官方分數（猜中 1、差一級 1/e、其他 0），**分等級平均**（`p19_lib.rank_balanced`）。NLL ＝ −log p(y)，分等級平均。
- Δ ＝ M1 − M0（Score 越大越好；NLL 越小越好）。
- 區間：配對 PlayerBootstrap，每級內重抽玩家 1,000 次（`np.random.default_rng(0)`），每題權重＝其玩家被抽中的次數；95% 百分位區間。
- **安慰劑**：20 次，`np.random.default_rng([30, i])` 把每盤 v2 整列在全部可用試跑盤之間隨機重排（打斷 v2 和玩家的對應，保留分布和維度），重算 z 和整個 M1 的 5 折 OOF，記 Δscore_i。

### 4.4 GO 規則（寫死；全部成立才 GO）

- **GO-a**：Δscore ≥ **+0.005**。
- **GO-b**：ΔNLL 的 95% 上界 < 0。
- **GO-c**：Δscore > 20 次安慰劑 Δscore 的最大值。
- 接縫 S 沒過：不判（停下查，多半是驅動程式）。

**門檻的理由**：
1. 全量的閘門是 Q25 → Q30 的 G1 ≥ +0.005。過去 SET 層級的增益進融合後只剩一部分：prereg25 S25 單獨 +0.0153 → 融合 +0.0111（73%）；prereg26 Q09 層級 +0.0030 → Q25 +0.0011（37%）。本檢定量的是「S25 單獨之上」的增益，而 Q25 另外有 HS（07 的逐段位 log p）、P1TO（含 EX 的分段 log p）這些和 v2 重疊的項，所以試跑低於 +0.005 時，全量過 G1 的機會很小。S30 端到端訓練也許比線性疊加多拿一點，所以不再往上加碼。
2. 雜訊：Δscore 是離散分數，題目共用玩家，所以另外要求 GO-b（連續的 NLL、以玩家為單位的區間不含 0）和 GO-c（贏過「同樣維度但沒有資訊」的 20 次安慰劑，約 p < 0.05）。這兩條防止「多 20 個參數在 588 位玩家上碰巧拿到 +0.005」。
3. 不對稱的代價：GO 會花約 19 小時 4090（和競賽一 v2 搶時段）和 4–5 天；NO-GO 的代價是放棄調查推估的 +0.002～+0.008。所以寧可嚴。

### 4.5 只記錄、不能救

- Δ 依 n（5／10／15／20）、依等級；M0 和 M1 的分數本身；只有 nlp 兩項（0、1 與對手方）的 M1、只有損失三項（2–4 與對手方）的 M1；只用 z（不含 lp25）的模型分數；每個 v2 特徵單盤和等級的 Spearman ρ；每折的 α 和 ‖u‖、‖v‖；安慰劑 Δ 的分布；v2 缺值比例；試跑速度（接縫 C）。
- 不論結果，`RESULTS.md` 都寫出 Δscore、ΔNLL、區間、安慰劑最大值、依 n 的差。

## 5. 全量計畫（只有 GO 才做；先寫死，避免看到試跑數字後才改）

### 5.1 要算的棋（4090；時數用歷史實測，GO 後以試跑速度比例重算）

| 組 | 盤數 | 局面 | P1 實測秒數 | 時數 |
|---|---|---|---|---|
| probe P1／P2／P3（fit） | 30,000／120,000／120,000 | 6.65M／13.83M／26.60M | 5,993／12,419／21,458 | 11.1 |
| V（驗證玩家的訓練棋，prereg25） | 74,946 | 16.67M | 13,635 | 3.8 |
| inner（bank＋mono，只為了 G1/G2/G4 評估） | 11,813＋61,356 | 2.63M＋13.50M | 2,366＋11,943 | 4.0 |
| 測試題（public＋private） | 5,000＋5,000 | 1.11M＋1.11M | 1,007＋855 | 0.5 |
| **合計** | | **82.1M** | | **約 19.4 小時** |

- 抽取都用同一支 `p30_extract.py`（V 走 prereg25 的 `val` 來源）。每組的 P1 部分和既有 chunk 比（接縫 S），S30 的 P1 輸入仍用凍結的 `probe_X`／`v_X`／`inner_X`，v2 只當多出來的 10 維。
- 順序：inner → probe P3 → P2 → P1 → V（測試題在 PREREG_30 之後）；每個 chunk 可續跑，可以拆成幾個夜間時段。
- **資源停損**：11/16 23:59 前 inner＋probe＋V 沒算完，就停，25 照舊（不用部分資料訓練）。

### 5.2 S30 和閘門（CPU）

- S30 ＝ S25 的架構、訓練、種子、鹽、步數、選步完全照抄（`p19_train` / `p25_train`，import 不改），只把每盤輸入從 1,540 維改成 1,550 維（P1 以 09 `p1e_map` 標準化；v2 以 probe 列的平均和標準差標準化（寫進模型檔），NaN → 0）。訓練列＝probe 270,000＋V 74,946（同 S25）。CPU 約 30 分鐘。
- Q30 ＝ Q25 的 7 項，SET 項換成 S30；評估、閘門、接縫照 prereg25 PLAN 第 3、4 節原樣：
  - 接縫 1：Q25 inner 0.7456519871546352、單色 0.7482450052195216 逐位元重現（相差 ≤ 1e-12）；
  - **G1** inner 評估假題（127,200 題）Q30 − Q25 ≥ +0.005；**G2** 配對 PlayerBootstrap 95% 下界 > 0；**G4** 單色（120,544 題）95% 下界 > 0 且黑、白各自 ≥ 0。只判一次。
- 過了才寫 `docs/PREREG_30`（凍結、獨立稽核、t30：測試題 v2 抽取約 0.5 小時 4090＋CPU 幾分鐘），照 PREREG_25 的流程。

### 5.3 時程（今天 11/09；上傳截止 11/24 23:59）

| 時間 | 事 |
|---|---|
| 11/09 | 本檔 commit → 程式、CPU smoke、獨立審查 → **等主 session 給 4090 時段** |
| 11/09 夜–11/10 | 試跑（約 1.3 小時 4090，可分段續跑）→ CPU 檢定（約 5 分鐘）→ GO／NO-GO |
| 11/10 夜–11/11 11:30 | （GO）inner＋probe 先算（約 15 小時內能做的部分） |
| 11/11 12:00–11/13 | 4090 給競賽一 v2，本線不用 |
| 11/13 夜–11/15 | 剩下的 probe＋V（約 5–8 小時）；遇到競賽二 v2 測試集重跑（R0／RR）時那條鏈優先 |
| 11/15–11/16 | S30 訓練、判一次 |
| 11/16–11/20 | （過了）PREREG_30、凍結、稽核、t30、測試題抽取、演練 |
| 11/21–11/23 | 上傳（最晚 11/23，留一天緩衝） |

- 任何一步晚於上表一天以上、或 4090 排不到，就停，25 照舊。

## 6. 凍結、不能動的東西

- 不改：`results/prereg01–29/**`、`docs/PREREG_*`、`tools/*`、`results/prereg08/p1/*`、`results/prereg07/**`（只 import）、測試日工具和指紋、split v1、`results/test*`。
- 本探索的新檔只在 `results/prereg30/explore/`（commit）和 `data/prereg30/`（不進 git）。
- 不碰 `p29-explore` worktree 和 Q: 磁碟機。競賽一不受影響。

## 7. 順序和檔案

1. 本檔 commit（任何特徵值或模型數字之前）。
2. `p30_lib.py`、`p30_data.py`、`p30_extract.py`、`p30_eval.py`、`test_p30.py`、`run_pilot.sh` commit；CPU smoke（幾盤，輸出 `data/prereg30/smoke/`，數字不記、不判）；獨立審查（子代理），修完 commit。
3. `p30_data.py sets` → 等主 session 同意 → `run_pilot.sh`（4090）。
4. 接縫 S → `p30_eval.py run` 只判一次（接縫過後、算 M1 之前寫 `data/prereg30/eval_started.json`，之後中斷也算判過）→ `eval_report.json` 原樣 commit → `RESULTS.md`。
- 環境：CPU `.venv`、GPU `.venv-gpu`；`PYTHONDONTWRITEBYTECODE=1`，不設 `OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`。worktree 在本 session 的 scratchpad，`core.autocrlf=false` checkout 後換成主 repo 的位元組，`data` 是指向主 repo `data` 的 junction（移除前先拆）。

## 8. 實作補記（試跑和判定之前；特徵定義、GO 規則、試跑資料都沒改）

- **程式分檔**：第 1.1 節寫的 `p30_lib.V2Scorer` 實際放在 `p30_scorer.py`（它 import `p1_lib`，會載入 torch／KataGo）；`p30_lib.py` 只留純 numpy 的 v2 計算（`keep_mask`、`game_v2`、`chunk_v2`），讓 `p30_eval.py` 不碰 torch。V2Scorer 不複製 P1 的算法：父類別的 `position_features` 原樣執行，只把模型包一層、留住同一次前向的輸出；lmax 用和父類別相同的列（含補齊）重算 masked log-softmax 取最大。
- **試跑名單已建**（`p30_data.py sets`，CPU 約 5 秒，只讀 probe_meta、probe 的 sets.json、舊 chunk 的 counts）：`pilot_inventory.json`（和本節同一個 commit）＝26,779 盤、588 位、5,924,986 局面、P1 速度約 1.334 小時；`data/prereg30/pilot/sets.json` SHA `55b4bb14…`、`pilot_map.npz` SHA `bbb38508…`。
- **一次未經排程同意的 CPU smoke（事件）**：11/09 00:51 跑了 `p30_extract.py run --device cpu --max-games 4 --out-dir data/prereg30/smoke` 約 24 秒，沒寫出任何 chunk（`data/prereg30/smoke/` 只有 `run_log.jsonl`），沒用 GPU，只讀 fit 玩家的訓練 CSV 和 KataGo 權重。之後所有執行（含單元測試、smoke）都等主 session 的排程時段。
- **判定的實作**（`p30_eval.py`）：接縫 S 在 `run` 裡先算（過了才往下；`seam_s.json` 也單獨寫）；M0 算完、M1 之前寫 `data/prereg30/eval_started.json`，存在就拒絕再跑。M0／M1 用解析梯度（單元測試和數值梯度比）；規則 B 和 `tools/decide.py` 相同（單元測試）。題目的玩家＝題目第一盤的玩家（同一題都是同一人）。依盤數、依等級、nlp／損失分組、只用 z 的模型、單盤 Spearman 都只記錄。
- `--smoke`：v2 換成 `default_rng(12345)` 雜訊、不讀 chunk、每格 30 題、2 次安慰劑、50 次 bootstrap，輸出 `data/prereg30/smoke/`，數字不記、不判。
