# prereg20 探索：B′-ON（在 10B 的分數上做一對一）能不能贏現行完整策略（競賽一，第 A 階段，fit 玩家模擬）

2026-10-20 寫成，**在算出任何候選數字之前 commit**（分支 `p20-sim`）。只用 CPU；不碰 `data/local`、不跑 `tools/score.py`、不用 GPU、不抽新的 KataGo 特徵（只讀 10/11 已抽好的模擬池 P1 塊）。不改任何凍結檔、事前登記或測試日路徑。本檔的閘門在正式執行前寫死，正式執行後只判一次（`eval_report.json` 原樣 commit）。點子來源：`analysis/codex_ideas_1020.md` (a)3、(b)3（使用者 10/20 同意做）。

## 0. 白話版

- 現在競賽一的登記是「二選一」：06-ON（在 06 的分數上做一題配一人）被 public 否決保留時，就不傳 10（A＋B′）；否則交 10。B′ 和一對一沒有一起用過（PREREG_10 第 4.4 節）。
- 這次要量的是第三種：**B′-ON**：在 10B 的分數上做 Hungarian 一對一（第 1 名），第 2–5 名照 10B；偵測器的門檻用 10B 自己的分數重新校準；public 一樣強制打開當探測、d < 0 就否決。
- 比的對象是**現行完整策略**（照登記：06-OFF／06-ON＋偵測器＋public 否決／10 的 A＋B′＋池閘門），不是只比 06-OFF。
- 只在 fit 玩家的官方形狀模擬池上比。過了也只代表「值得寫登記和改測試日工具」，本探索**不做**那些（第 9 節只寫要做什麼和截止）。

## 1. 玩家和資料（只用競賽一的 fit 玩家；沒有驗證玩家）

- 母體：PREREG_10 模擬池的 926 位 fit 玩家（`data/prereg10/mock/sets.json`，SHA `167c5a51…`，每人 100 盤候選黑棋＋20 盤 base 題目庫＋≤ 20 盤 nov4 題目庫）；P1 特徵是 10/11 抽好的 616 塊（`results/prereg10/feat/mock_manifest.json`），讀 `data/prereg10/mock_work/p1_side0.npy`、`p1_usable.npy`（mock 當時由那 616 塊寫出；程式開頭核對 `p1_side0.npy` 的 SHA 等於 `mock_report.json` 記的 `side0_npy_sha256`、`mock_points.json` 的 SHA 等於 `points_json_sha256`，`p1_usable.npy` 要 123,114 盤全為真（mock 報告記 usable 123,114），不符就停）。
- 06 的項：照 PREREG_10 的 mock（`tools/p10_fuse.py`：`p06_sim.Store(data/prereg06/feats)`、CNN `data/prereg06/cnn_fit/black_ge105.npz`，樣本內）；06 的 LDA 每池排除池內玩家重學（`LdaData.fit_excluding`）。
- **P1 的 LDA**：照 PREREG_10 的 mock，每池用「池外玩家」的可用盤重學（`p10_fuse.fit_p1_lda`），w*＝16（凍結值），CSLS k＝10。模擬裡不用凍結的 `p1_lda.npz`（它學過全部 926 人）。
- **玩家分成 dev／eval 兩半（不重疊）**：每個等級內依 `sha256("c10-prereg20|split|<player_id>")` 排序；eval 取前 max(40, ⌈n/2⌉) 人，其餘是 dev。A 級只有 57 人，所以 eval 40、dev 17；其他等級各約一半。人數寫進報告。
  - dev：只用來校準 B′-ON 的偵測器門檻 r*_B（第 3 節）。
  - eval：只用來組正式比較的池（第 4 節）。
  - 兩半都會出現在對方池的「池外玩家」裡，用來學 LDA（池內玩家一律不進 LDA）。

## 2. 三個要比的策略（每一個 trial＝一個 private 池＋一個另外抽的 public 池；只計 private 分數）

每個池先算（和測試日的 t06／t10 同一套程式邏輯，用 fit 的模擬資料）：

1. **06 的項**：O、JO、S2、L、E0（只用黑棋），組合 V2／V3／V0（`p06_sim.combo_F`，E 來源＝真實 E0）。
2. **sr06**（PREREG_06 2.3，10 輪假題，`p06_sim.sr_pool` 給的候選盤）：V2、V3、V0、R02（＝Q04，近似 test02 選中的組合）各自的 sr；選組合照 PREREG_06 2.4（容差 0.003，順序 V2、V3、V0）。06 閘門：sr(選中) − sr(Q04) ≥ 0，沒過這一池交 test02 的列（第 1 層＝Q04；第 3 層＝只用 CNN）。
3. **防護層**：`p10_fuse.rho_raw`（PREREG_10 mock 用的 ρ 近似）→ 第 1／2／3 層。
4. **10B**：F_B＝F_sel＋16·z(T_P1)，交 CSLS10(F_B)；**B 的池閘門**：同一批 sr 假題上 sr(10B 選中) − sr(選中) ≥ 0（`p10_apply.sr06_with_p1` 的算法）；沒過這一池不加 B。B 只加在用了 06（含 A）的池。
5. **06 偵測器**（PREREG_06 2.6／t06）：r＝forced_cost(F_sel) ÷ 平均(forced_cost(第 0、1 輪假題的 F_sel))，打開＝選中組合能用、r ≤ 凍結的 r*_X（V2 2.2515761…、V3 2.2678004…；V0 不能用）。06-ON 的排法：第 1 名 Hungarian（z(F_sel)），第 2–5 名 CSLS3。

**(a) 現行完整策略（照登記）**

- 06-ON 的條件（PREREG_06 5.3、t06 rev 3 第 17 條在模擬裡能判的部分）：兩池都用了 06 而且都在第 1 層；private 選中組合能用而且 private 偵測器打開；public 強制打開；d06＝public(06-ON) − public(06-OFF)，**d06 ≥ 0 保留 06-ON**（public 分數只用 public 池自己的答案算）。模擬裡 T2(c) 複製數一定是 0、題數和候選數一定是 400、AIdea 頁面當成沒寫。
- 06-ON 保留 → 最終交 06-ON。否則 → 最終交 10：每池＝10B（B 池閘門過）／06-OFF（第 1、2 層，或第 3 層經 A）／test02 的列（06 閘門沒過）。

**(b) B′-OFF**：上面的「否則」那一支（永遠不做 06-ON）。只記錄。

**(c) 候選：B′-ON／OFF 開關**

- 兩池都用了 10B（B 池閘門都過）而且都在第 1 層時：
  - B′-ON 的排法：第 1 名＝在 z(F_B) 上的 Hungarian 指派，第 2–5 名＝依 CSLS10(F_B)（10B）排，跳過第 1 名那位。
  - 偵測器 r_B＝forced_cost(F_B) ÷ 平均(forced_cost(第 0、1 輪假題的 F_B))，假題的 F_B＝那一輪的 F_sel＋16·z(T_P1)（P1 用那一輪的 80 盤參考棋和假題棋）。打開＝r_B ≤ r*_B（第 3 節）。
  - public 強制打開；d_B＝public(B′-ON) − public(B′-OFF)；private 偵測器打開而且 d_B ≥ 0 → 最終 B′-ON；否則 B′-OFF（＝10）。
- 條件不成立（任一池沒有 10B，或不在第 1 層）→ 照 (a)。
- r*_B 沒有可行門檻（第 3 節）→ 候選永遠不開，等於 (b) 加上 (a) 的退路；照樣判閘門。

## 3. dev：校準 r*_B（照 PREREG_06 2.6 的選法；只用 dev 玩家）

- **池**：seed 1000–1029（30 個），每個 seed 一組 dev 玩家：每個等級依 `sha256("<seed>|<player_id>")` 取前 min(40, 該級 dev 人數) 人，不足 400 人時從剩下的 dev 玩家依同一雜湊順序補滿 400 人（A 級只有 17 人，所以 dev 池的等級比例和官方不同，揭露）。
- **結構**：PREREG_06 的 11 種：o2o、u390、u375、u350、d80、u300、wr、d1、d2、d3、d5（owner 的抽法照 `p06_sim.make_pool` 同名結構，`random.Random("scen|<結構>|<seed>")`）。題目從 base 題目庫（每人 20 盤）依序取；一位玩家題數加總超過 20 盤時，用 `p10_fuse._wr_questions` 的修補規則（換 k、必要時把題移給還有空間的玩家；`random.Random("repair|<結構>|<seed>")`）。盤數 5／10／15／20 各 100，`random.Random("k|<seed>")` 洗牌。候選＝每人 sets 的前 100 盤。
- **F**：F_sel 固定用 V2（dev 不做 sr 選組合，省算力；eval 才照登記選）。兩種 E 來源各校準一次（照 PREREG_06 2.6）：真實 E0（樣本內 CNN），和替身 A（`p06_sim.combo_F(T, "PA", …)`，s34 的 NB 代替 E0）；F_B＝F_V2(來源)＋16·z(T_P1)。
- **開／關**：開＝B′-ON 的排法；關＝CSLS10(F_B)。假題 forced_cost 用該 seed 的 o2o 池的第 0、1 輪（候選相同，所有結構共用）。
- **選法**（`p06_player.select_threshold`，原樣）：候選門檻＝dev 池上出現過的 r_B；可行＝每種結構的平均（開 − 關）≥ −0.0005，而且沒有任何非 o2o 池虧超過 0.005；可行的門檻裡取 11 種結構平均增益最大的，平手取最小；最大平均增益 ≤ 0 視為沒有門檻。**r*_B＝兩種來源門檻的較小者**；任一來源沒有門檻 → 候選永遠不開。
- dev 的數字是校準用，可以看；r*_B 寫進 `dev_report.json` 並 commit 之後，才跑 eval。

## 4. eval：正式比較（只用 eval 玩家；判一次）

- **池**：每個 trial i＝0–39：private 池 seed 2000＋i、public 池 seed 3000＋i；兩池各自從 eval 玩家抽，每個等級依 `sha256("<seed>|<player_id>")` 取前 40 人（兩池可以有相同玩家，候選和題目各自獨立組；不共用答案）。
- **情境**（public 和 private 同一種情境）：

| 情境 | 定義 | 用途 |
|---|---|---|
| o2o | 一題對一人（base 題目庫） | **G1** |
| d40 | `make_pool` 的 d40：40 位候選沒題、另 40 位各兩題（10% 玩家問 2 題），容量修補同第 3 節 | **G2** |
| d120 | d120（30% 玩家問 2 題） | **G2** |
| nov4 | PREREG_10 的 nov4（題目的前 4 手沒出現在自己的候選棋；沒有新穎棋的人把題讓給別人） | **G3**（開局不重疊壓力） |
| wr | PREREG_10 的 wr（有放回抽題） | 只記錄 |

- **nov4 的防護層**：照登記 nov4 多半在第 3 層，(a)、(c) 都退回 10（差一定是 0）。所以 **G3 用「忽略防護層」的版本判**（兩個策略都當作第 1 層：06-ON 和 B′-ON 都可能打開，A 不觸發）；照登記（有防護層）的版本只記錄。
- **點估計和區間**：每題的差＝(c) − (a) 的 private 得分（官方公式 e^-(r−1)，Top-5），40 個 private 池合併的每題平均；95% 區間用玩家 bootstrap（以題目的真正玩家為單位有放回抽，2,000 次，每個情境一個新的 `numpy.random.default_rng(0)`；同 `p10_fuse.player_boot`）。

## 5. 閘門（寫死；只判一次）

| 閘門 | 條件 |
|---|---|
| **G1** o2o | 平均差 (c) − (a) ≥ **+0.003**，而且 95% 下界 > 0 |
| **G2** d40、d120 | 各自平均差 ≥ **−0.002**；而且候選的 private 偵測器（r_B ≤ r*_B，否決前）在 d40＋d120 的 80 個 private 池裡打開的比例 ≤ **1%**（也就是 0 池） |
| **G3** nov4（忽略防護層） | 平均差 ≥ **−0.002** |

- 三個都過＝「通過」：只寫第 9 節的筆記，不做登記和測試日工具。任一沒過＝「不用」。
- 只記錄、不可拿來選：(b) − (a)、(c) − (b)；沒有否決的候選（偵測器打開就用）；各策略各情境的平均分；06 偵測器和 B′ 偵測器各情境的打開率、否決率；B 池閘門、06 閘門、防護層的分布；d06、d_B 的分布；wr 的全部數字；照登記的 nov4。

## 6. 接縫（在 dev 和 eval 之前；不過就停下查，不跑 dev／eval）

- **S1（06 和 10B 的 mock 數字）**：用本探索的程式（同一個 pool 計算函式）在 PREREG_10 mock 的 seed 20–39 base 池（926 人、池外 LDA，和 mock 相同）重算 V2、10B@16 每題得分，要和 `data/prereg10/mock_work/mock_points.json` 的 `pts["V2"]`、`pts["10B@16"]` **逐值相同**；20 池的平均等於 `mock_report.json` 的 `record_only.scores.base.eval.V2`（0.9875688662686579）和 `10B@16`（0.9961770495166119）（差 ≤ 1e-12）。
- **S2（06 偵測器）**：用本探索的偵測器函式在 PREREG_06 switch 的 seed 500 o2o 池（`p06_sim.make_pool`，06 的 store 池）重算 V2|E0、V2|PA 的 r_forced_cost，要和 `results/prereg06/design/switch_report.json` 記的 1.5027003071519183、1.425846984044762 差 ≤ 1e-9。
- **S3（sr06）**：S1 的 seed 20 base 池上，sr06 選中的組合和四個 sr 平均只記錄（沒有既有數字可比）；另在單元測試裡用小池核對 `sr06_with_p1` 的算法和本程式相同。
- 接縫只比既有的 06／10B 數字，不產生任何候選數字。

## 7. 實作、審查和執行順序

1. 本 PLAN commit（分支 `p20-sim`，worktree 在 scratchpad）。
2. 程式 `results/prereg20/explore/p20_sim.py`（`seam`、`dev`、`eval` 三個子指令）和 `test_p20.py`；只 import 既有工具（`p10_fuse`、`p10_apply` 的算法、`p06_sim`、`p06_style`、`p06_player.select_threshold`），不改任何既有檔。大檔放 `data/prereg20/`（不進 git）。
3. smoke：只印時間和形狀，不印任何分數（不能用 smoke 偷看效果）；用 seed 9000 以外的不計數 seed。
4. 接縫 S1、S2 → `seam_report.json`，commit。
5. dev → `dev_report.json`（r*_B），commit。
6. **正式執行前的獨立審查**（另一個代理；它看程式和 PLAN，不看 eval 數字）：發現的問題修好、寫進第 10 節，commit。
7. eval（判一次）→ `eval_report.json`，原樣 commit；`RESULTS.md` 寫判定。
8. `--no-ff` 合併回 main；拆 junction、移除 worktree、刪分支；檢查 main 的 `data/` 沒有多出 `*should_not_exist*` 資料夾（只看 `data/` 第一層和 `data/prereg20`，不進 `data/local`）。
- 程式錯誤造成的重跑：只有在 eval 之前發現才可以；eval 跑完之後發現錯誤，寫明錯誤、在本檔加修訂再重跑一次，兩次的報告都留著。

## 8. 已知限制（寫進 RESULTS）

- CNN 對 fit 玩家是樣本內（偏強）；兩個策略都含同一個 CNN，偏差大致相消，但會讓 06 偵測器、06-ON 比真實情況好看。P1 的 LDA 是池外學的。
- 06 偵測器的凍結門檻 r*_X 是在同一批 fit 玩家（含 eval 玩家）上校準的；B′ 的 r*_B 只用 dev 玩家。兩者的樂觀程度不同，對現行策略有利。
- dev 池的 A 級只有 17 人；eval 池每池的 A 級是同一批 40 人。
- 雙題玩家的盤數受 base 題目庫 20 盤限制，修補後雙題玩家的 k 偏小（報告記錄修補次數）。
- R02 用 Q04 近似（test02 的自我檢索多半選 Q04）；防護層用 ρ 的近似值。
- public 和 private 是同一種結構；官方兩池是否同結構不知道。
- 新 seed 不等於新資料：dev／eval 仍是 PREREG_06／10 探索看過很多次的同一批 926 人。

## 9. 通過時才寫的筆記（不做）

通過時在 `RESULTS.md` 寫：要上場需要的登記修訂（PREREG_06／10／13 的修訂或新的 PREREG_20：B′-ON 的條件、r*_B、上傳順序、public 否決對 10 的例外、和 06-ON 的先後）、`t06`／`t10` 要改的地方、K 截止點（11/04 11:00 測試集開放之前要 commit 登記和工具；K5-13 10/31）。不寫就不上場。

## 10. 修訂和審查紀錄

### rev 1（2026-10-20，dev 和 eval 之前；當時沒有任何候選數字）

**審查**：另一個代理（Claude 子代理，獨立 context，只讀程式和登記、跑單元測試，不跑 seam／dev／eval）在 `7785e08` 上審查：沒有 blocker，4 個 major、7 個 minor。它核對過正確的部分：(a) 的組合選擇、06 閘門、06-ON 排法、06 偵測器和凍結 r*_X、public 探測和 d ≥ 0；10 的退路（第 3 層經 A、閘門沒過交只用 CNN 的列）、B 池閘門；(c) 的排法、假題 F_B、r*_B 取兩來源較小者；dev／eval 分開、兩個 LDA 都排除池內玩家、public 只用 public 自己的答案；閘門算術和 bootstrap；沒有碰 `data/local`。

舊版（`7785e08`）的接縫 S1（只有 base、用 p10 的 build_pool）和 S2 已經跑過而且都過（S2 差 2.7e-15、2.2e-16）；那一份報告沒有 commit，改用下面加嚴的接縫重跑。

照審查改的地方（都在 dev／eval 之前）：

1. **（major）** 工作行程裡的錯誤改成 `RuntimeError`（`SystemExit` 會讓 multiprocessing 卡住不結束）。
2. **（major）第 6 節 S1 加嚴、S3 改寫**：S1 改用本程式自己的 `build_pool`，在 mock 的 seed 20–39 重建 **base、nov4、wr** 三種池（926 人、和 mock 同一批池內玩家），每池的 owner、k、V2 和 10B@16 每題得分要和 `mock_points.json` **逐值相同**，ρ 差 ≤ 1e-12、防護層相同；三種情境的 20 池平均要等於 `mock_report.json` 記的 eval 平均（差 ≤ 1e-12）。S3 原本的「單元測試核對 `sr06_with_p1`」不做（那支函式走 t06 的 MemStore，沒法在模擬池上逐值比）；改成 sr06 迴圈只有一份（`pseudo_rounds`），S2 也走它（S2 另比兩輪假題的 forced_cost，差 ≤ 1e-9）；seed 20 base 池的 sr06 選擇和平均只記錄。審查代理已逐行核對 sr06 迴圈和 `p10_apply.sr06_with_p1`、`p06_player.sr06_apply` 的算法相同。
3. **（major）B′ 只在 private 池選中 V2 時才能打開**：r*_B 只用 V2 的 F_B 校準（第 3 節），照 PREREG_06 2.6「每個組合各有自己的門檻」的精神，選中 V3／V0 的池不開 B′（G2 的誤開也照這個算）。
4. **（major，同 2）** S1 現在涵蓋 eval 用的建池程式（o2o、nov4、wr 的 owner／k／ρ／層）。d40、d120 的建池規則（`owners_for`＋`repair`）由單元測試核對：d 結構的 owner 和 `p06_sim.make_pool` 的規則相同、修補後容量 ≤ 20、`repair("wr")` 和 `p10_fuse._wr_questions` 逐值相同。
5. （minor）eval 每個 job 寫一個檔（`data/prereg20/eval_jobs/`），中斷後重跑會接續，同一次執行的結果不變。
6. （minor）eval 開跑前檢查：seam 通過而且已 commit；`dev_report.json`、`p20_sim.py`、`PLAN.md` 已 commit 而且乾淨；`eval_report.json` 已存在時，只有 `--rerun --reason` 才能再跑，而且寫成 `eval_report_rerunN.json`（第 7 節「兩次都留著」）；eval 之後不准重跑 dev。
7. （minor，不改規則，寫進第 8 節）兩個池都有 10B 時，(c) 只在 B′-ON 和 B′-OFF 之間選，不會退回 06-ON。所以「(a) 保留了 06-ON、但 B′ 的偵測器沒開或被否決」的 trial，對 (c) − (a) 是扣分；r*_B 沒有門檻時 (c) 等於 (b)。報告另記這種 trial 的個數（`a_06on_but_c_not_BON`，只記錄）。
8. （minor）寫死的常數（k*＝10、CSLS3、V1 無、組合順序、w*＝16）開跑前和兩份凍結檔核對。
9. （minor，同 2）S2 走和 eval 相同的假題函式。
10. （minor）單元測試加 `judge()`（閘門、誤開、nov4 兩種版本）、第 2 層、選中 V3 時 B′ 不開、`repair("wr")` 等於 p10（18 項全過）。
11. （minor）第 7 節第 3 條的 smoke 用的是 seed 9000（不在任何計分的 seed 裡），只印時間和形狀。smoke 已在 `7785e08` 跑過，沒有印任何分數。
12. （minor，第二輪審查：同一個代理在 `f25f893`＋seam 報告上覆核，前面 1–11 都正確、沒有新錯；剩這一項）eval 的每個 job 檔記下 `p20_sim.py` 的 SHA，重跑接續時 SHA 或 (seed, role) 不同就拒絕，避免新舊程式的結果混在一起。

**dev 結果**（`dev_report.json`，eval 之前 commit）：r*_B＝3.6519820570685297（E0 來源 4.0135、替身 A 3.6520，取較小者）。
