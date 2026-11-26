# 事前登記 03：競賽二 recipe v2 雙臂重訓，和 R20 在 log 空間集成

- 驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。前一份：`docs/PREREG_02_OPENING_FUSION.md`（競賽二、競賽一都採用 prereg02，結果 `results/prereg02/RESULTS.md`）。
- 設計過程：`results/prereg03/design/`（三份調查、三份提案、兩位評審、綜合方案 `plan.md`；診斷腳本與 JSON 在 `evidence/`，全部只用 fit／inner 訓練玩家）。
- 本登記寫「比什麼、怎麼選、怎樣算成功、什麼時候放棄」。notebook 格子、工具指令等實作規格以 `results/prereg03/design/plan.md` 第 3–5 節為準；實作時要偏離規格，在任何 prereg03 輸出產生之前記進修訂紀錄。
- **只做競賽二。** 競賽一這次不登記新方法、不打分（第 6 節）。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | `9e4f8da` | 第一版，2026-09-26。當時**還沒有任何 prereg03 的 notebook、輸出或分數** |
| rev 2 | `39ebccb` | 2026-09-27，程式寫完、CPU 冒煙、三角度審查與修正後，**使用者還沒跑 Colab、沒有任何 prereg03 的真實輸出或分數**。實作時的偏離與補充（第 10 節）；影響判定的是第 10 節第 1–4 條（停止分三類、新的每一輪核對、B 臂的還原來源與分片排列、工具怎麼處理停下過的臂），第 5–14 條是實作說明 |
| rev 3 | `af739a5` | 2026-09-28，AIdea 官方資料說明公布後補寫（競賽二每題單一執色，5／10／15／20 盤各 100 題）；**使用者還沒跑 Colab，沒有任何 prereg03 的輸出或分數**。新增只記錄的單色紀錄，測試日 κ 檢查的參照改用單色題（第 11 節）；選法、閘門 G1–G3、check 的停止條件都不變。會影響判定的只有第 11 節 (c)：單色條件成立而驗證單色子題的 Δκ 不是 > 0 時，03 不在測試日使用，也不跑 H3R（打分機會保留） |

## 0. 為什麼做這個（都不看答案）

- R20 在 inner 5 折交叉驗證是 0.5855，本機驗證集 0.5891。各級：D .742、C .618、B .577、A .536、1D .467、2D .488、3D .472、4D .576、5D .631、6D .747，中段（1D–3D）最弱。
- 官方模型只看最後 250 個盤面、**不知道目標玩家執哪一色**；prereg02 的訓練準確率（33.9%）明顯高於 inner（25.6%），有過擬合跡象；訓練資料的有效玩家只有約 1,907 人（少數玩家的盤數很多）。
- 這三點各自對應 recipe v2 的一項改動：目標玩家視角、D4 資料增強、每人盤數上限與各級等量（B 臂）。診斷細節：`results/prereg03/design/understand_errors.md`、`evidence/`。
- 其他候選（校準細化、SGF 特徵、整題聚合器、label smoothing）在 inner 上都 ≤ +0.002 或有明確反證，這次不做（第 7 節）。

## 1. Colab：兩臂各一份 notebook，一次執行

`colab/build_prereg03_rank_nb.py` → `colab/prereg03_rank_a.ipynb`、`colab/prereg03_rank_b.ipynb`（兩份只差 `ARM` 一行）。

### 1.1 兩臂共同的配方（寫死）

| 項目 | 設定 |
|---|---|
| 模型 | 官方 `GoRankResNet`，輸入 501 層，其他不變；從零訓練（SEED 42） |
| 輸入 | 最後 250 個盤面，每個盤面兩層改成「目標玩家的子／對手的子」，加 1 層「目標玩家執黑＝1」。執色一律取訓練資料的 `color` 欄，不從 game_id 推 |
| D4 增強 | 只在訓練時，每筆均勻抽 8 種對稱之一 |
| 最佳化 | AdamW，lr 1e-3，weight decay 0.01，batch 256，交叉熵，不做 label smoothing |
| 學習率 | 前 1,000 步線性 warmup，之後 cosine 降到 0；**總步數 T＝33,768**（＝12 × 2,814），和 worker 數、GPU 型號無關 |
| 精度 | 訓練用 bf16 autocast；監看和所有逐盤輸出用 fp32 |
| 資料順序 | 每一輪的分片順序和 shuffle 由 (42, 輪次, worker) 決定（修正 PREREG_02 rev 4 第 7 條「用時間當種子」的問題）；cuDNN 與 bf16 仍不保證逐位元組相同 |
| 指定 checkpoint | 第 T 步的 `final.pth`，不另選 best；inner 監看只記錄 |

### 1.2 兩臂只差取樣

- **A 臂**：每一輪用全部 fit 棋（720,307 盤），約 12 輪。
- **B 臂**：每位玩家最多算 200 盤，而且各級等量：盤 g 收下的機率 q_g＝min(1, 200/n_p)×35,845/E_L，E_L 寫死在 `plan.md` 3.4 節；用 `sha256("c10-prereg03|acc|{pass}|{rank}|{game_id}")` 決定收不收。每一輪約 358,450 盤、每級 35,845 盤，T 步約 24.1 輪。
- 兩臂的 T 相同；200 盤上限、各級等量、T、warmup 是看過 fit 的中繼資料後寫死的，沒有在 inner 上調。

### 1.3 輸出與停止條件

- 每臂輸出 inner（73,556 盤）和本機驗證題（19,156 盤）的 fp32 逐盤 logits，8 種對稱；另存「原方向、執色對調」的 logits 只供第 2.2 節的檢查。格式同 prereg02 的 dump，key 陣列必須和 prereg02 逐陣列相同。
- 停止：GPU 不是 G4 或 A100；打包檔、特徵指紋、外洩、執色與 D4 的單元檢查、B 臂的 E_L 任一不符；第 300 步推估的訓練時間超過 150 分鐘（G4）或 180 分鐘（A100）；loss 出現非有限值（備案要先寫修訂、改 fp32 整份重跑）；接續時配方設定不同；收尾核對不符。任一停止，那一臂不算結果。
- 只讀 prereg01、prereg02、test02 的 Drive 資料夾（寫入一律擋下、前後比對指紋）。

### 1.4 帳號與運算單元（運作上的選擇，不影響判定）

- 推估：A 臂 52–96 分鐘、7.5–13.9 單元；B 臂 60–115 分鐘、8.7–16.7 單元（bf16 加速沒實測）。
- 開跑前看帳號的「可用量」：**≥ 50** 兩臂同時跑；**25–49** 只跑 A（用到 B 的組合從清單拿掉，不算偏離）；**< 25** 不在這個帳號跑。優先用 AI Pro 帳號（有 prereg01 的特徵備份）；AI Pro 不足而 Pro+ ≥ 50 時改在 Pro+ 跑（使用者要先上傳打包檔一次）。實際用了哪個帳號、哪種 GPU 記進 RESULTS。

## 2. 本機的合併與選擇（只用 CPU，`tools/p03_rank_fuse.py`）

### 2.1 沿用 prereg02

- import `tools/p02_rank_fuse.py` 的函式與常數（校準公式、擬合、5 折、PlayerBootstrap、規則 B 等），**不改任何 p02 檔**；工具執行時核對 p02 工具、`c10data.py`、`decide.py` 的 SHA-256 和 HEAD 相同。
- 沿用已 commit 的 `pseudo_inner.json`、`opening_inner.npz`、`opening_val.npz`（M＝12、α＝20，只用 fit 計數），不重算。
- 校準公式、擬合目標、界限、起點、5 折、假題輪次都和 prereg02 相同：p＝softmax(e^a·(n/12)^γ·(m＋w·m_op＋b))，盤數只進入 10 個等級共用的溫度。

### 2.2 新來源能不能用（`check`，在選組合之前）

1. 結構：key 陣列和 prereg02 相同、權重 SHA 一致、`history.csv` 記錄跑完 T 步、驗證題輸出的 NaN 樣式和 `val_p02e9_d4` 相同。
2. 地板：`final.pth` 在 inner 原方向的逐盤 accuracy < 24.15% → 這個來源不能用。
3. 執色接線：執色對調後的 accuracy 比正確執色高 0.2 個百分點以上 → 判為 bug，停下查。
4. 執色語意（κ，不讀答案）：κ＝每盤 log 機率和「同題其他盤合併分布」的期望對數似然，題內平均再各題平均；Δκ＝κ(正確執色)−κ(整題執色對調)。當 Δacc ≥ 0.5 個百分點且 inner 的 Δκ 95% 下界 > 0（模型確實在用執色）時，要求驗證題的 Δκ > 0，否則停下查。這一步只決定「能不能用」，不影響組合的選擇。
5. `selfcheck`：只用 C5 的 P00 在 inner 的交叉驗證、全部擬合資料重擬合的參數、套到驗證題的機率，都要重現 prereg02 R20（差距 ≤ 1e-12，規則 B 完全相同），才往下做。

### 2.3 組合（順序固定，也是平手時的優先順序；全部用 log 合併、校準、開局項、規則 B）

| 組合 | 來源 |
|---|---|
| P00 | C5（prereg02 epoch 9，8 種對稱）＝R20 |
| P01 | C5 ⊕ C6A |
| P02 | C5 ⊕ C6B |
| P03 | C5 ⊕ C6A ⊕ C6B |
| P04 | C6A |
| P05 | C6B |
| P06 | C6A ⊕ C6B |

- 集成：各來源 8 個方向的 log_softmax 串起來再平均（各來源等權，不加新參數）。驗證題上各來源的可解析盤必須相同。
- 不能用的來源，連同用到它的組合，從清單拿掉。
- 選法：inner 5 折交叉驗證的分等級平均（規則 B）最高者；和最高分差距在 0.003 以內的，取清單中最前面的。選定後用全部擬合資料重擬合，寫進 `results/prereg03/rank/frozen_config.json` 並 commit，之後才能產生驗證題的預測。
- 同時用同一套程序擬合選定組合的**無開局版**（w＝0），記下 inner 分數；測試日 T1 是 FAIL_OPENING 時，用「這個無開局版」和 R10（`results/test02/rank/fallback_R10.json`，inner 0.5687）中 inner 較高者，現在就寫進 `frozen_config.json`。

### 2.4 送分閘門（三項都成立才打分，只看選到的組合）

- **G1**：選定組合減 P00 的 inner 交叉驗證分等級平均 ≥ +0.010。
- **G2**：同一差值的 inner 配對 PlayerBootstrap（每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
- **G3**：inner 評估輪的每題平均差值 ≥ 0。
- 選到 P00 本身或閘門沒過：記為「未送分」，打分機會保留，R20 照舊；新來源的逐盤輸出留給之後的登記。
- 只記錄：驗證集的推估標準誤與採用機率（用 inner 的每題得分差和驗證集各級題數算）、各級差值、盤數分箱的預測分布。

## 3. H3R：在本機驗證集打一次分

- 指令：`.venv\Scripts\python.exe tools/score.py compare-rank results/prereg02/rank/pred_B.csv results/prereg03/rank/pred_B.csv data/local/rank_val_answers.csv --ledger results/score_ledger.csv --note "prereg03 H3R (A = prereg02 R20, reference re-read)" --json results/prereg03/rank/score_H3R.json`
- **採用條件**：分等級平均差值（03 − R20）的 95% 信賴區間下界 > 0（每級內分層重抽 10,000 次、seed 0），而且每題平均差值的點估計不 < 0。兩者都成立才採用；否則不採用，照 VALIDATION 記為「結論依指標而定」或「沒有改善」。
- **合理性門檻**：新方法的分等級平均 < 0.54 時先查 bug（要指出具體的程式錯誤才算）。
- **對照組重讀**：A 側 1,589 題，兩個指標和帳上 prereg02 R20 的紀錄差距都 ≤ 1e-12（PREREG_02 rev 7 之後的定義），並記錄打分用的 Python 版本（`.venv`，3.12）；不符就停下，查清前不採用。
- 預算：競賽二整季 4 次，已用 1 次（02）；03 最多用 1 次。查 bug 重跑也算進預算並記帳。
- 交件模型：甲（本機驗證過的做法，不用全部 8,007 人重訓）。

## 4. 截止點（寫死；任一沒趕上，這一季不用 03，也不拿額外的打分機會補）

| 截止點 | 條件 |
|---|---|
| C1 10/16（五）23:59 | 至少一臂能用的 Colab 結果已 commit |
| C2 10/19（一）23:59 | 閘門已判定；過了的話 H3R 已打分 |
| C3 11/03（二）23:59 | 採用時：測試日工具的 03 擴充（`plan.md` 第 5 節）已全部 commit、演練通過，並另寫本登記的修訂 |
| C4 11/16（一）23:59 | 採用時：03 在官方測試集至少有一次成功的正式執行 |

## 5. 採用後的測試日（細節在 C3 前另寫修訂）

- test02 照 `docs/TESTDAY.md` 先跑、先上傳 R20（或 R10）和競賽一 Q04；03 是之後的另一次上傳。
- test03 只推論新來源（C6），C5 讀 test02 已 commit 的結果；接縫、κ 檢查（public、private 各自的 Δκ > 0，不成立代表官方 `color_i` 語意和訓練不同，03 不用、維持 R20）都要過。
- **最後一次上傳**（11/23 前）同時符合才交 03：H3R 採用；C4 前有一次成功的正式執行且已 commit；T1 不是 FAIL_STRUCT；測試集的 κ 檢查通過。否則交 R20（FAIL_OPENING 時照 test02 規則）。
- public 分數比上一版低 0.05 以上只追查錯誤，不據此換方法（VALIDATION 紀律 4）。

## 6. 競賽一：這次不做

- 本機 0.8830 的主要風險是官方怎麼切題目和候選，本機驗證量不到，只有 11/04 的 T1–T6 能回答。
- 兩隊不交換模型輸出：prereg03 的模型和任何輸出都不進競賽一。
- 競賽一的整季預算維持「已用 1 次」；11/04 之後需要改再另寫登記。

## 7. 刻意不放進 03 的

| 做法 | 理由 |
|---|---|
| 逐手、依等級條件化的落子模型 | 上限最高但工程新、在我們的規模沒有前例，會拖到 11/04 的準備；10/14 看 03 結論再決定要不要當 04 |
| 換 seed 的複本當第二臂 | B 臂對準「有效玩家少」的問題，多樣性比換 seed 大 |
| label smoothing、SORD | 瓶頸是分不開相鄰兩級，把目標往鄰級抹平只會更分不開 |
| 加寬、加深、換頭 | 已經過擬合，沒有本題的證據 |
| 加長盤面歷史、氣或提子層 | 長棋沒有比較差；要重抽特徵 |
| 訓練出來的整題聚合器 | fit 的逐盤輸出在訓練樣本內；inner 只有 636 人 |
| 校準細化、SGF 特徵 | inner 上都 ≤ +0.002 |
| 用全部 8,007 人重訓（乙） | 本機驗證不了；看 11/04 的 T3 |
| 外部資料、預訓練模型、KataGo | 登記禁止；出題者寫所有人都從零訓練 |

## 8. 揭露（寫進 RESULTS 和報告）

- 配方的選擇參考了只用 inner 的診斷和文獻，沒有在 inner 上調任何訓練超參數。
- B 臂的 200 盤上限、各級等量、T、warmup 是看過 fit 的中繼資料後寫死的。
- 開局的 M、α 和組合都是在 inner 上選的，inner 交叉驗證略為樂觀。
- 無開局版沒有在驗證集打分。
- 訓練不能逐位元組重現（cuDNN、bf16）。
- 生成式 AI 協助的部分照規則揭露。

## 9. 順序

1. 本登記和 `results/prereg03/design/` commit。
2. 程式：notebook 產生器與兩份 notebook、逐格差異、`tools/p03_rank_fuse.py`、`tests/test_p03_*.py`、CPU 冒煙紀錄、獨立審查結果，commit；`docs/COLAB_RUN.md` 加「事前登記 03」一節。
3. 使用者回報帳號的可用量，照 1.4 決定怎麼跑；跑兩份 notebook，下載兩個 zip。
4. 結果解開到 `results/prereg03/`，先 commit（調參前）。
5. `check` → commit；`selfcheck` → commit；`tune` → commit `frozen_config.json`；`apply`、`decide.py`、`check_submission.py` → commit；過閘門才跑第 3 節的指令。
6. 打分紀錄帳、分數 JSON、`results/prereg03/RESULTS.md`（由另一個代理獨立核對數字）、地圖一起 commit。

## 10. rev 2：實作時的偏離與補充

實作紀錄：`results/prereg03/design/build_review.md`（兩個實作代理、整合、三角度審查、修正）。程式：`colab/build_prereg03_rank_nb.py` → `colab/prereg03_rank_{a,b}.ipynb`（逐格差異 `results/prereg03/design/nb_diff_{a,b}.txt`）、`tools/p03_rank_fuse.py`、`tests/test_p03_*.py`；使用者步驟 `docs/COLAB_RUN.md`「事前登記 03」一節。

**會影響判定的**

1. **停止分三類**（取代 1.3 的停止條件寫法）：
   - (a) 登記的停止，這一臂不算結果：第 300 步推估時間超過上限（`time_stop`）、loss 非有限值（`nan_stop`）、每一輪的盤數／key 集合／執黑數／學習率核對不符、或那一輪沒有讀到任何資料（`pass_check_failed`）、收尾核對不符（`final_check_failed`；收尾時另外核對 Drive 上的 `ckpt/history.csv` 要和記憶體中的紀錄相同、最後到第 T 步，不同就重寫一次再核對，仍不同才算）。這些會留下 `STOPPED.json`，之後重按「全部執行」一律在開頭停下。
   - (b) 計算前的拒絕，可以修好後重跑：GPU 關卡、preflight（包括 Drive 剩餘空間查不到，或少於 3 GB；A 臂要重抽特徵並寫備份時少於 5 GB）、各項單元檢查、`state.pt` 讀不進來、接續時配方設定或分片排列不同。一臂絕不以不同設定接續。
   - (c) 不算停止，從最後存好的一輪接續：Drive 空間不足的暫停（`drive_space_pause`，每一輪開始前剩不到 3 GB，和 preflight 同一個門檻）、斷線、執行中的其他例外。
2. **每一輪的盤數核對**（登記沒寫的新停止條件，屬 (a)）：每跑完一輪核對讀到的盤數、key 集合、執黑數；訓練前把 fit、inner 分片的 key 全讀一遍。
3. **B 臂只從 prereg01／prereg02 的唯讀特徵備份還原**，不用 A 臂寫的備份；分片排列的 SHA-256 寫進配方設定，接續時不同就不接續。
4. **本機工具怎麼處理停下過的臂**：`run_log` 任何一次執行記過 (a) 類狀態、`config.json` 有 `stopped`、或有 `STOPPED.json` 時，`p03_rank_fuse.py unpack` 拒收，不解開；`check` 把沒解開的臂視為沒跑，組合清單拿掉用到它的組合（等於 1.2「那一臂不算結果」）。萬一這樣的臂已經被解開，`check` 判整體停止，要先寫修訂才能往下做。

**只是實作說明**

5. **channels_last**：官方 `forward` 在 `bn_in2` 之後用 `view` 攤平，channels_last 的張量不能 `view`，所以加一個 forward hook 把 `bn_in2` 的輸出轉回 NCHW；`network.py` 不改，和官方 NCHW 輸出相差 < 1e-5，`final.pth` 可直接載入官方類別。
6. **種子細節**（1.1 的「由 (42, 輪次, worker) 決定」的具體寫法）：輪次從 1 起算；每一輪開始時 random／numpy／torch 的種子是 42·1000＋輪次；worker 的 shuffle 種子是 (42·1000＋輪次)·1000＋worker；warmup 是 λ(t)＝(t＋1)/1000，之後 cosine 在第 T 步剛好為 0；建模型前重設 `torch.manual_seed(42)` 並記錄初始權重 SHA-256。
7. **中間存檔**：每一輪的 `pass_N.pth` 只存在 Colab 本機磁碟，不寫進 Drive；Drive 只放 `state.pt` 和 `final.pth`。
8. **key 檔比對**：npz 位元組含時間戳，所以正式執行改比 key 陣列的內容摘要（釘住值見產生器）；打包檔的檔案清單必須等於 prereg01／02 `env.json` 記的那一份，特徵指紋也釘 SHA-256。
9. **帳號紀錄**：notebook 無法得知是哪個帳號，改由使用者回報帳號和 GPU、Claude 記進 RESULTS（1.4 原寫「記進 run_log」）。10/03 另外收 Pro+ 的 Drive 空間截圖，帳號分級不變。
10. **第 300 步的時間推估**從第 1 輪建 loader 就開始計時，含啟動時間，偏保守。
11. **GPU 關卡**位在沿用的準備格之後，仍在抽特徵和訓練之前。
12. 沿用的準備格保留 prereg02 那一行拒收答案檔的 assert（字面上含那個字），notebook 不讀答案檔。
13. `history.csv` 在第 T 步停下的那一輪，`samples_expected` 記為 NaN。
14. 冒煙測試實測：B 臂 32 個 worker 每個約 0.23 GB（G4 有 190 GB RAM）；本機工具 `tune` 在真實規模約 1 分鐘。

## 11. rev 3：官方單色結構（單色紀錄；測試日 κ 參照）

依據：`docs/DATA.md`「官方資料說明」、`results/test02/design/official_structure/README.md` 第 2 節。官方競賽二每題的棋全部同一執色，盤數只有 5／10／15／20；本登記的 inner 假題（2.3、2.4）、本機驗證集、2.2 第 4 條的 κ 檢查都是黑白混合。建置、審查、修正紀錄：`results/prereg03/design/build_review.md` 第 5 節。

**不變的**：2.3 的組合與選法、2.4 的閘門 G1–G3、第 3 節 H3R 的指令、採用條件和預算、2.2 第 1–5 條（第 4 條仍在混色的 inner 評估輪和驗證題上判定，停止規則不變）、`frozen_config.json` 的參數與 FAIL_OPENING 的選擇、第 4 節截止點。(a)(b) 不進入選法、閘門或 check 的停止；唯一會影響判定的是 (c)。

**(a) 單色 inner 假題（只記錄）**
1. 玩家：`pseudo_inner.json` 的 bank 玩家，各自沿用原本的折。
2. 每位玩家、每種執色的池：這位玩家該執色的全部 inner 棋，依 `sha256("c10-prereg03|mono|order|{rank}|{game_id}")` 排序。
3. 第 0–31 輪：每位玩家（照玩家順序）、先黑後白，k∈{5,10,15,20} 且 k 不超過池的盤數，各出一題，位置是 `random.Random(int(sha256("c10-prereg03|mono|q|{輪}|{player_id}|{執色}|{k}")[:16], 16)).sample(range(池), k)`。每題一個種子，和其他玩家無關。
4. 規模（真實資料預覽）：623 位玩家、1,170 個（玩家、執色）組、120,544 題（黑 59,744、白 60,800；k＝5／10／15／20 各 37,440／32,288／27,296／23,520），題目集 SHA-256 `45d0f14acd98…`。
5. `tune`：每個組合用 5 折交叉驗證各折的參數（仍由登記的混色 fit 假題擬合）打單色假題，第 f 折玩家的題用第 f 折的參數。打分前先核對「各折參數重算混色評估輪的機率，和 OOF 逐位相同」，不同就停下。記錄全體、只黑、只白、各 k 的分等級平均與每題平均，以及選定組合減 P00 的配對 PlayerBootstrap 95% CI（全體；黑、白各自重抽玩家）。`tune_report.json` 記全部組合，`frozen_config.json` 記選定組合、P00 和差值。
6. 參考（scratch 預跑，不是登記數字；RESULTS 改引 commit 後 `tune_report.json` 的數字）：P00（＝R20）在單色假題的 OOF 分等級平均 0.5833（混色 0.5855）；黑 0.5647、白 0.6017；k＝5／10／15／20 為 0.534／0.583／0.615／0.624。

**(b) 單色 κ 參照（測試日用；(c) 第 2 種情況在 tune 就生效）**
1. κ、Δκ 的定義同 2.2 第 4 條（整題執色對調）。
2. inner：在 (a) 的單色假題上算 Δκ 和 PlayerBootstrap 95% CI（每級內重抽玩家 1,000 次，`default_rng(0)`）；黑、白各自再算一次（各自重抽玩家），各 k 只記平均。
3. 本機驗證集：每題可解析的棋依題目檔的 `color_i`（經 `val_keys.npz`；只讀題目內容，不讀答案）分成黑、白兩個子題，≥2 盤的子題才算（2,986 個子題：黑 1,492、白 1,494；1,589 題都至少有一個）。Δκ 取子題平均，黑、白各記一次，另記題目層級 bootstrap CI（只記錄）。
4. **單色條件**：Δacc ≥ 0.5 個百分點，而且 inner 單色 Δκ 的 95% 下界 > 0。
5. 寫在 `check_report.json` 各臂的 `colour.kappa_single_colour`（`test_day_reference` 記單色條件和驗證子題 Δκ 是否 > 0）；`frozen_config.json` 另存選定組合用到的 C6 來源那一份（`single_colour.kappa_reference`）。`check` 不因這些新增任何停止，也不改可用來源。

**(c) 測試日的 κ 檢查**（取代第 5 節第 2 點和 `plan.md` 5.2 第 2 點的 κ 寫法；test03 的實作在 C3 前另寫修訂）

對選定組合用到的每個 C6 來源：
- **單色條件成立，而且驗證集單色子題的 Δκ > 0**：public、private 各自在官方題目上（每題單色、整題執色對調）的 Δκ 都要 > 0。不成立就判定官方 `color_i` 的語意和訓練不同：03 不用，維持 R20，並記錄。
- **單色條件成立，但驗證集單色子題的 Δκ 不是 > 0**（≤ 0，或沒有可算的子題）：單色題上的 κ 在本機就判讀不了執色，測試日的檢查沒有參照：03 不在測試日使用，維持 R20，並記錄。這在 `check`／`tune` 時就知道，所以**不跑第 3 節的 H3R，打分機會保留**（記為「未送分」，同 2.4 最後一點）：`tune` 把 `score_allowed` 設為否，原因寫在 `tune_report.json`、`frozen_config.json` 的 `score_blocked_single_colour`；`apply` 從 `frozen_config.json` 的參照重算同一件事，寫進 `apply_report.json` 的 `score_blocked_reasons`。選定的 C6 來源沒有單色參照時也一樣處理（正式資料不會發生）。
- **單色條件不成立**：模型在單色題上看不出有用執色，接錯的代價也小；public、private 的 Δκ 只記錄。
- public、private 另外分黑題、白題記錄 Δκ（只記錄），對照 inner 和驗證子題的黑、白參照。
- 第 5 節「最後一次上傳」的「測試集的 κ 檢查通過」指這一條。
- 第 9 節第 5 步「過閘門才跑第 3 節的指令」改為「`apply_report.json` 的 `score_allowed` 為真才跑」（閘門沒過、選到 P00、或上面第 2 種情況，都是否）。

**(d) 實作說明**
- `tools/p03_rank_fuse.py`：`MONO_*` 常數、`mono_questions`／`mono_set`／`mono_qset`、`kappa_single_colour`、`val_colour_subquestions`、`question_boot_mean`、`fold_predict`、`single_colour_tune_records`、`single_colour_score_block`。新的拒絕：正式模式沒有任何單色假題；`tune` 讀到的 `check_report.json` 沒有單色紀錄或題目集 SHA 不同；各折參數重算不出 OOF 機率。
- `tool_state` 記錄的未 commit 檔名修正（原本第一個檔名少一個字；只影響紀錄，「有沒有未 commit」的判斷不變）。
- `selfcheck` 在真實資料照舊重現 R20（P00 inner OOF 0.5854702521055817），輸出和改前只差時間與工具狀態。
- 測試：`tests/test_p03_rank_fuse.py` 共 45 項，這次新增 12 項，包括「關掉單色假題時，選擇、閘門、參數、`score_allowed`、驗證題機率都逐位相同」，以及「(c) 第 2 種情況：選法、閘門、參數、驗證題機率不變，`tune` 和 `apply` 的 `score_allowed` 變否，`apply` 從 frozen 重算」。

**(e) 揭露（併入第 8 節）**
- 單色紀錄和 (c) 是看過官方資料說明後補的，都在任何 prereg03 輸出之前。單色 inner 分數只記錄，不影響選擇；(c) 只可能讓 03 不上場、不打分，不會讓 03 多打分。
- inner 單色假題的分數仍是 OOF，但開局的 M、α 和組合是在 inner 選的，所以略為樂觀；單色假題和混色假題抽自同一批 inner 玩家，兩者不獨立。
