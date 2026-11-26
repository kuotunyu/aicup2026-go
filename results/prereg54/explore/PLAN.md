# prereg54 PLAN：競賽一新題「強模型的分數層集成」（c49 ＋ prereg51 L／B 臂）

日期 2026-11-18。分支 `p54-ens2`（從 `p52-ens` `648221e`，沿用 prereg52 的偽池程式）。**這份在任何 prereg54 數字之前
commit。** 只做競賽一新題；不用競賽二的任何特徵或輸出；不碰驗證集答案資料夾（不開、不讀、不列、不雜湊，指令裡不寫那個
路徑）；不跑 `tools/score.py`；不用打分機會。

## 0. 白話版與依據

- prereg52：同強度兩個種子（S0＋S1）的分數層集成在 D 上 +0.0110（過閘門）；但「強的 c49 ＋ 弱一級的 S0」在候選庫偽池只有
  +0.0033（沒過），弱成員會稀釋。所以值得試的是**強度相近的強模型**互相集成：c49（320k、B10C128、T49）、prereg51 的
  L 臂（480k、B10C128）、B 臂（320k、B15C128，或照 prereg51 第 3 節的 B12C128）。三個都用 T49 訓練，看過 D 的玩家，
  所以開發集用 prereg52 修訂 1 的**候選庫自檢索偽池**（不靠 D），判定用 E。
- prereg51 先照它自己的登記各臂單獨判 E；這一輪排在 prereg51 的判定**之後**，基底＝那時已採用的單一模型。

## 1. 成員與向量

| 成員 | 存檔 | 向量資料夾（`store_E.npz`、`pool_public.npz`、`pool_private.npz`） |
|---|---|---|
| c49 | `data/prereg49/S0/ckpt/step_320000.pt` | `data/prereg49/emb/V2_step_320000_raw_tta1/` |
| L | `data/prereg51/L/ckpt/step_480000.pt` | `data/prereg51/emb_L/*_step_480000_raw_tta1/`（4090 的 `chain_p51L.sh` 產生） |
| B | `data/prereg51/B/ckpt/step_320000.pt`（Colab 傳回） | `data/prereg51/emb_B/*_step_320000_raw_tta1/`：4090 上 `p51_run.py embed --arm B --size <實際大小> --ckpt … --weights raw --tta 1 --emb-root data/prereg51/emb_B`（E 與兩個官方池） |

- 每個向量檔要符合：`ckpt_sha256`＝該存檔的 SHA-256；raw、tta1；同一批盤（`store_row`、玩家雜湊、`count` 和 c49 的檔相同；
  官方池檔列數相同）。不符就停。
- 某成員的向量不存在：含它的候選刪掉（第 2 節），寫進紀錄。

## 2. 基底與候選（固定）

- **基底**＝prereg51 判定後已採用的單一模型（`--base c49|L|B`）。c49 時判定紀錄＝`results/prereg49/explore/eval.json`
  （E 0.7035676497986656、官方池 `results/prereg49/apply/`）；L 或 B 被採用時，用 prereg51 已 commit 的判定檔
  （`--base-eval`，`adopt` 為真）與它的 apply 資料夾（`--base-apply`），基底的 E 每池要和判定檔該臂的 `per_pool` 逐位元相同。
- **候選**（順序即優先序，見第 4 節）：

| 代號 | 成員 |
|---|---|
| ENS_c49_L_B | c49＋L＋B |
| ENS_c49_L | c49＋L |
| ENS_c49_B | c49＋B |

- 每個成員各自走 v2-6 後端（`results/prereg43/explore/tune.json` 的 `c_star.backend.cfg`：gnorm、λ 0.7、不降維、原型、wE 8），
  得到各自的 E 項；集成＝prereg52 第 2.1 節（逐列 z、等權平均、ok 取交集，只有一個成員時用原矩陣）；之後 `fuse_cfg`
  ＋ 8 × z(P1 joint／b4) → CSLS 20。權重、設定全部固定，不在任何資料上調。

## 3. 開發集與接縫

- 偽池＝prereg52 修訂 1 的 10 個偽池，**同樣的種子與切法**（`p52_pseudo.split`；public `52000 + s`、private `52100 + s`，s＝0–4），
  只讀官方候選庫檔，向量與 P1 清單只用前 6,000 列。偏差同 prereg52 修訂 1（只比差，不比絕對分數）。
- **接縫（逐位元，否則停）**：
  1. 偽池：N 個成員的程式只放 c49 → 每個偽池和 `results/prereg52/explore/pseudo.json` 的 `per_pool.A49` 相同（平均 0.6188134125030416）。
  2. E：同一程式只放 c49 → 每池和 `results/prereg49/explore/eval.json` 的 `per_pool.c49` 相同（平均 0.7035676497986656）。
     這只重算已判過的 v2-7，不算任何候選的 E。基底不是 c49 時，另外要重現基底判定檔的每池分數。

## 4. 開發閘門（偽池；每個候選對基底）

- 平均差 ≥ **+0.01**、偽池 bootstrap（10,000 次、種子 43）95% 下界 **> 0**、10 個偽池中 **≥ 8 個**較高。
- 多個過關時：**成員數多的優先；成員數相同取偽池平均較高者；再平手取第 2 節表中較前者**。只有這一個進 E。
- 都沒過：只記錄，不用 E，競賽一維持基底。
- `dev.json` commit 之後才可以跑 E 判定。

## 5. E 判定（20 池，只判一次）

- 選出的候選 對 基底；採用條件同 prereg51：平均差 ≥ +0.01、池 bootstrap（10,000 次、種子 43）95% 下界 > 0、20 池至少 18 池較高。
- E 池使用次數＝下一個（到 prereg49 是第 6 次；prereg51 每判一臂加一次；程式依 `results/prereg51/explore/eval*.json` 的檔數核對
  `--e-use`）。
- L、B 的 E 向量是 prereg51 單臂判定已經用過的檔，不另要求晚於本 PLAN；c49 同。
- **期限：2026-11-22 12:00（本機時間）前完成 E 判定**；過了程式不判，維持基底。

## 6. 採用後（同 prereg49）

兩個官方池 apply（基底接縫：基底在該池重算的 top5 要和基底 apply 的 `top5.csv` 逐位元組相同；注意 CRLF，必要時用
`git -c core.autocrlf=false checkout` 重取）→ assemble 1,600 列（原題 800 列沿用 v1）→ 檢查 → 上傳（常設同意）→
`results/official_scores_v2.json`。

## 7. 資源

只用 CPU：`BK.polite()`、可用 RAM ≥ 9,000 MB 才開始、跑的時候 < 7,000 MB 就中止重來、共用鎖 `data/c10_cpu.lock`、
單程序峰值 ≤ 2.5 GB：成員一個一個載入，算完 E 項就釋放；E 的向量以 float32 保存（後端在 `pool_games` 才轉 float64，
數值相同，由接縫 2 驗證）；G 項的 LDA 資料每色用完就釋放。

## 修訂 1（2026-11-18 17:xx，任何 prereg54 候選數字之前；只改格式與路徑，不改候選、閘門、判定規則）

- prereg51 的判定程式 `results/prereg51/explore/p51_eval.py`（分支 `p51-eval`）寫的判定檔是 `results/prereg51/explore/eval_<臂>.json`
  （`adopt`、`arm`、`means.<臂>`、`per_pool.<臂>`、`by_k.<臂>`、`e_pools_use_number`、`base`），apply 輸出在
  `results/prereg51/apply/<臂>/<pool>/top5.csv`。第 2 節的 `--base-eval`／`--base-key`／`--base-apply` 固定為這三者
  （預設即是，給別的值就停）。
- `--base` 必須等於 prereg51 紀錄裡最後被採用的那一臂（`eval_L.json`／`eval_B.json` 裡 `adopt` 為真、`e_pools_use_number`
  最大者；都沒有＝c49），不一致就停。
- 第 5 節的 E 使用次數核對改為數 `results/prereg51/explore/eval_*.json`（和 prereg51 自己的規則「7 ＋ 已有 eval_*.json 數」一致）。


## 修訂 2（2026-11-18 17:xx，在任何 prereg54 候選分數之前）

新增成員 **S**＝prereg55（c49 配方只換種子 1，`data/prereg55/S/ckpt/step_320000.pt`，向量 `data/prereg55/emb_S/*_step_320000_raw_tta1`）。理由：prereg52 顯示集成要同等強的成員。候選與優先順序（成員多者優先，同數依此表）：{c49,S,L,B}、{c49,L,B}、{c49,S,L}、{c49,S,B}、{c49,L}、{c49,B}、{c49,S}；只用向量都存在的。其他（閘門、假題目池、E 判一次、截止）不變。
