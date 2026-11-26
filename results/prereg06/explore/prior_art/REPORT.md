# prereg06 探索：prior_art（2026-10-02）

> 這份由 workflow `wf_85ecc07b-a64` 的探索代理寫成。子代理不能直接寫 .md，主 session 照它回傳的結構化結果（同資料夾 `result.json`）原文排版存檔，沒有改內容。數字都是設計階段、只用訓練玩家（fit／inner），不是驗證集結果。

## 摘要

競賽二：文獻最強的做法是逐手計算「各等級的人下這手的機率」（KataGo HumanSL 多等級）＋強度分數，再交給 LightGBM（Kuboki 2025，出題者顏士淨是共同作者）。
同一玩家、同盤數換算成本題分數：論文 5／10／15 盤是 0.718／0.740／0.755，只用 HumanSL 是 0.678／0.705／0.715；我們的 04 在 inner 單色假題是 0.554／0.611／0.647（20 盤 0.655）。
論文用野狐，我們是幽玄，等級分組也不同，HumanSL 的訓練資料又含野狐，所以不能直接比。實際增益估計 +0.02～+0.06（沒有實測），要先在 inner 試跑；前提是規則允許用 KataGo。
差距在「差兩級以上」：我們差一級以內 82～91%，論文 93～97%。
標籤比例配對和 EM 不做：自己模擬，剛好平衡時只 +0.001，不平衡時 −0.006～−0.032，EM −0.02～−0.03。目數損失特徵也不做（論文消融顯示幾乎沒用）。
私榜 400 題的平均分，1 個標準誤約 ±0.019：小於 0.01 的改動在私榜上看不出來。
競賽一：我們的開局檢索在官方結構的模擬是 0.924，已遠超公開的圍棋研究（Lin 2024：200 人、5 盤 Top-1 48.5～57.5%）。剩下的空間主要在 5 盤題（目前 0.81）。
候選池內的 hubness 修正（同一池一起看，壓低「跟誰都像」的候選人）在兩種情境下都加分：CSLS10 +0.018／+0.020，dual softmax τ=0.02 +0.031／+0.020。值得登記試。
一對一配對（匈牙利演算法）：一題對一人時 +0.054（5 盤題 0.81→0.94）；不是時 −0.097。只能在 11/04 確認測試集結構後用。
2023 屆的題目不同，也沒有公開分數。今年主辦方簡報只教一般技巧，沒提開局、HumanSL 或一對一，一般隊伍大多會走那條路。
10/2 報名數：競賽二 98 隊、競賽一 80 隊。
REPORT.md 被 harness 擋下（子代理不能寫報告 .md），完整內容在本結果的 verdicts 和 opportunities 裡。

## 判定與證據

### (a) Chen, Shih, Wu『Strength Estimator（SE）』ICLR 2025

- 判定：「逐手打強度分數，再對整題取平均」比「逐局面分類再加總」強很多。但論文主打的數字來自隨機抽樣（同一等級、不同玩家），會高估；同一玩家的情形換算約 0.669／0.729／0.743（5／10／15 盤）。需要大網路和大算力。
- 證據：https://arxiv.org/abs/2502.17109 （PDF 已讀）：野狐 11 組（3–5 級到 9 段），每組 4.5 萬盤；20b256c；13 萬步，約 242 GPU 小時；隨機抽樣 15 盤 >80%、100 盤 97.5%；前人 SL_sum 100 盤只有 49%。同一玩家的數字來自 arXiv 2505.00279 的表 4（51.6/93.1、59.4/96.0、61.1/96.9）。換算在 convert_literature.json。

### (a) Kuboki, Ogawa, Hsueh, 顏士淨, Ikeda 2025（HumanSL 多等級機率＋SE＋LightGBM）

- 判定：和我們題型最接近、證據最強的做法，出題者是共同作者。同一玩家時換算成本題分數：5／10／15 盤 0.718／0.740／0.755；只用 HumanSL 和目數損失（不用 SE）也有 0.678／0.705／0.715。我們 Q02 是 0.554／0.611／0.647。帳面差 +0.07～+0.16，但因為資料和分組不同，不能直接比。
- 證據：https://arxiv.org/abs/2505.00279 （PDF 已讀）。表 4，同一玩家：58.1%(95.3)、61.0%(96.2)、63.2%(96.6)；隨機抽樣：70.4、82.8、89.4。表 7 消融（同一玩家、15 盤）：全部 63.2、拿掉 SE 57.8、拿掉 HumanSL 61.1、拿掉目數損失 62.8。meta-model 是用隨機抽樣的資料訓練的。我們這邊的數字：results/prereg06/explore/prior_art/convert_literature.json，由 u1_oof.npz 的 mono_P_Q02 以規則 B、分等級平均重算，和 u1_analysis.json 的 by_n 逐位相同。

### (a) KataGo HumanSL（b18c384nbt-humanv0）

- 判定：可以依等級條件化輸出落子機率：rank_20k～rank_9d，也有 preaz_ 和 proyear_。訓練來源的代碼有 OGS、KGS、FOX、TYGEM、GOGOD、GO4GO，沒有�幽玄。所以它應該沒看過我們這批棋，但段位的對應不同，要由我們的堆疊模型去學。我們的棋是 2021–2023 年，應該用 rank_，不是論文用的 preaz_。
- 證據：https://github.com/lightvector/KataGo/releases/tag/v1.15.0 ；https://raw.githubusercontent.com/lightvector/KataGo/master/cpp/configs/gtp_human5k_example.cfg ；https://raw.githubusercontent.com/lightvector/KataGo/master/python/katago/game/sgfmetadata.py （SOURCE_OGS…SOURCE_GO4GO）。授權與速度見 9/26 的 results/prereg02/design/understand_priorArt.md 第 4 節。能不能用由規則主題判斷。

### (a) Moudřík/Baudiš/Neruda 2015（GoStyle）；2016 三類 CNN

- 判定：舊做法，精度遠不如 SE 和 HumanSL，不值得照做。
- 證據：https://arxiv.org/abs/1512.08969 （PDF 已讀）：KGS，26 個等級，每位玩家 10–50 盤，RMSE 2.659 級（猜平均 7.503）。https://researchgate.net/publication/309559342 （只讀到摘要）：三類，單一局面 71.5%、整盤 77.9%。

### (a) Maia／Maia-2（西洋棋）

- 判定：Maia-2 不做「從落子反推等級」，只預測落子。值得參考的是架構（依棋力調整的注意力，同時輸入自己和對手的等級），以及「對手等級影響決策較小」：換自己的等級最多改變 22% 的決策，換對手的只有 6%。這和我們 O4（看對手落子）有幫助但幅度小的結果方向一致。
- 證據：https://arxiv.org/abs/2409.20553 （PDF 已讀）：落子預測平均 53.25%（Maia-1500 51.39%、Stockfish 約 38–40%）；訓練 1.69 億盤、91 億局面。

### (a) Kaggle Finding Elo（2014–15）

- 判定：從單盤預測 FIDE Elo，評分用 MAE。前幾名都用 Stockfish 的每手評估差和局面特徵。和我們的關聯低：有人類等級模型時，損失類特徵幾乎沒有加分（見 Kuboki 消融）。冠軍分數未查核。
- 證據：https://www.kaggle.com/competitions/finding-elo （討論頁抓不到內容；做法只看到搜尋摘要；https://github.com/jwnorman/Chess_ELO 說明了資料格式）。

### (b) McIlroy-Young 等人 2021 Behavioral Stylometry（NeurIPS）

- 判定：架構：逐手 ResNet（6 block、64 channel、SE）→ 12 層 Transformer 聚合成一盤的向量 → 512 維，玩家向量取 centroid，loss 用 GE2E。每人 100 盤參考、100 盤查詢時：2,844 人去掉前 15 手 0.860，保留開局 0.979；400 人（去掉前 15 手）0.953；41,184 人 0.54。約 50 盤開始飽和。沒見過的玩家和見過的玩家一樣準。
- 證據：https://proceedings.neurips.cc/paper/2021/hash/ccf8111910291ba472b385e9c5f59099-Abstract.html （PDF 已讀，表 2、表 3、圖 3a）。只看 1900 分附近的人時 92.6%。開局 5 手的基準（從第 15 手起）是 0.244。

### (b) Maia 個人化模型（KDD 2022）、MHR-Maia 2025、Maia4All 2025

- 判定：開局最能認出人；盤數少時差最多。400 人時 Top-1：10 盤 0.86（全部手數）、0.47（第 10 手起）、0.11（第 30 手起）；100 盤 0.98／0.95／0.55。MHR：400 人、30 盤 98.8%；47,864 人、100 盤 94.4%（見過）／87.6%（沒見過）。Maia4All 只要 20 盤就能個人化，但摘要沒給辨識準確率。
- 證據：https://arxiv.org/abs/2008.10086 （PDF 已讀，表 3）；https://arxiv.org/abs/2502.14998 （網頁版結果段落）；https://arxiv.org/abs/2507.21488 （摘要）。

### (b) 圍棋辨識玩家：Lin, Chiu, Wu（TMLR 2024）

- 判定：公開的圍棋辨識數字遠低於我們的開局檢索。他們是 200 人、每人 100 盤候選，只看前 10 手：查詢 5 盤 48.5～57.5%、10 盤 66.5～72.5%、25 盤 81～85%、100 盤 94.5～97%。我們在官方結構的模擬（400 人）Top-1：5 盤 0.772、10 盤 0.920、15 盤 0.962、20 盤 0.972。
- 證據：https://arxiv.org/abs/2408.06051 （PDF 已讀，附錄表 21–24，候選 100 盤那一列）；我們的數字：results/test02/design/official_structure/README.md（P120、raw、N10、both，5 個 seed）。

### (c) AI CUP 2023 圍棋（東華，508 隊）

- 判定：題目不同：模仿 10 級和 1 段的落子，加上三類棋風。找不到公開的排行榜分數或冠軍報告，對今年的分數上限沒有參考價值。
- 證據：T-Brain 舊網址 https://tbrain.trendmicro.com.tw/Competitions/Details/29 現在轉址後 404；https://github.com/AILAB-NDHU/AIcupTutorial ；https://stat.fcu.edu.tw/news-detail/?id=41484 （508 隊，佳作）；https://github.com/yiyunrin/AI-CUP-Team_3839 （沒有分數）。計分權重（10 級 Top-1 0.25…棋風 0.3）只在搜尋摘要看到，未查核。

### (c) 今年主辦方的教學與簡報

- 判定：EP5 列的進階方向都是一般技巧：D4 對稱、氣／提子特徵、SE／Attention、ordinal loss、label smoothing、集成（「不集成幾乎拿不到前段名次」）、hard mining、ArcFace、Set Transformer、EMA、Mixup。第 80 張寫「兩個子題目前都沒用 pretrained 權重，這是現成的改進方向」；第 15 張寫教學版拿掉預訓練模型，「所有人都從零訓練」。簡報沒提開局、HumanSL、逐手機率、一對一。一般隊伍最可能走「baseline＋這些技巧＋集成」。
- 證據：簡報 https://www.aicup.tw/_files/ugd/7fbdbf_dee0824b36cb4dd2a4af2497347a04f7.pptx （10/2 重新下載，SHA-256 ff2b6033…42ae4，和 9/26 相同）；用 results/prereg06/explore/prior_art/extract_slides.py 抽出文字（第 15、80–95 張）。baseline：競賽二 0.4631、競賽一 0.2845。

### (c) 參賽人數與頂端分數

- 判定：10/2 時，競賽二 144 人／98 隊，競賽一 119 人／80 隊（9/26 是 69 隊和 63 隊），報名到 11/03。頂端分數沒有直接證據。估計：競賽二 0.60 以上應已在前段，0.65 以上可能要靠逐手機率類的做法；競賽一只要有人想到開局就能到 0.9 左右，前段的差距會在 5 盤題和有沒有用上一對一。
- 證據：https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403 ；https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688 （10/2 讀取）。頂端分數是估計，沒有實測。

### (d) 競賽二：標籤比例配對（每級強制 40 題）和 EM 重估比例

- 判定：不做。真實比例剛好平衡時，強制配對只 +0.0010（2.5～97.5% 區間 [−0.016, +0.019]）；隨機抽人時 −0.0057；比例偏斜時 −0.0317。EM 在三種情境是 −0.0265／−0.0229／−0.0036。原因：Q02 本來就在平衡的假題上校準過；400 題估比例的 L1 誤差約 0.32～0.37。
- 證據：results/prereg06/explore/prior_art/sim_c2_label_matching.py → sim_c2_label_matching.json：inner 單色假題、Q02 的 OOF 機率，1,000 份假測試檔，每份 400 題、5／10／15／20 盤各 100 題，約 2 分鐘 CPU。文獻：Saerens 等人 2002（Neural Computation）；Alexandari 等人 2020 https://proceedings.mlr.press/v119/alexandari20a.html 。

### (d) 競賽一：一對一配對（匈牙利演算法）和池內 hubness 修正

- 判定：匈牙利配對：一題對一人時 0.924→0.978（+0.054，5 個 seed 都在 +0.047～+0.072），5 盤題 0.811→0.935；不是一對一時（80 位候選沒有題、80 位各兩題）0.930→0.833（−0.097）。所以只能在確認結構後用。hubness 修正兩種情境都是正的：CSLS10 +0.018／+0.020（最差的 seed +0.011），dual softmax τ=0.02 +0.031／+0.020，τ=0.05 +0.010／+0.014。τ 沒有在樣本外挑；以上只用開局檢索，沒有 CNN。
- 證據：results/prereg06/explore/prior_art/sim_c1_assignment.py → sim_c1_assignment.json：fit 玩家、官方結構 P120、black_random、seed 0–4；基準逐 seed 重現 9/28 的 0.9184／0.9213／…（reproduced=true）。前例：Kaggle Recursion 第 1 名的 make_submission.py 用 linear_sum_assignment（https://github.com/maciej-sypetkowski/kaggle-rcic-1st ），加分多少未查核；CSLS https://arxiv.org/abs/1710.04087 。PREREG_02 當時判「hubness 輕微」而沒做；兄弟主題 results/prereg06/explore/c1_headroom/ 正在用融合後的分數做更完整的版本。

### 私榜雜訊

- 判定：一份 400 題的平均分，1 個標準誤：競賽二約 ±0.019（每題分數的標準差約 0.375）。差距小於 0.01 的改動在私榜上幾乎看不出來；+0.05 以上才會穩定改變名次。
- 證據：用 u1_oof.npz 的 mono_P_Q02 以規則 B 算：每題平均 0.6168、標準差 0.3751，除以 √400 得 0.0188（指令在本次 session 執行，沒有另存檔）。

## 機會

### KataGo HumanSL 多等級落子機率特徵＋小模型堆疊，疊在 Q02 上（comp2，信心 low）

- 預期增益：+0.02～+0.06（估計，沒有實測；論文換算的帳面差 +0.07～+0.16 不能直接比）
- 依據：Kuboki 2025 同一玩家換算 0.718／0.740／0.755，只用 HumanSL 0.678／0.705／0.715；我們 Q02 在 inner 單色假題 0.554／0.611／0.647／0.655。HumanSL 在 SE 之上，同一玩家時還多 +2.1～+5.6 個百分點的完全猜中率。
- 工作量：約 5 天；Colab 3 次
- 使用者要做：在 Colab 跑 notebook：先在 inner 試跑一次推論，之後是正式版，測試日再跑一份。上傳 Drive、截圖確認檔名、回報結果。
- 主要風險：規則是否允許 KataGo（規則主題判斷）；HumanSL 的段位對應幽玄不同；10 個等級 × 每手的推論算力；測試日工具要在 11/03 前另開分支完成，不能動 R20／04 的凍結路徑；整季打分機會只剩 2 次。

### 池內 hubness 修正（CSLS k=10 或 dual softmax），用在 Q04 的融合分數上（comp1，信心 medium）

- 預期增益：+0.01～+0.03（只用開局檢索的模擬：CSLS +0.018／+0.020，DSM τ=0.02 +0.031／+0.020；疊在 Q04 上可能比較小）
- 依據：自己模擬 sim_c1_assignment.json，5 個 seed、一對一和非一對一兩種情境都是正的；CSLS 在其他領域 P@1 +4～+6。
- 工作量：約 2 天；Colab 0 次
- 使用者要做：沒有。全部在本機 CPU，測試日流程不變。
- 主要風險：τ 是在同一批模擬裡看的；Q04 融合後的分數分布不同；要寫事前登記、另開分支改測試日工具；c1_headroom 主題正在量融合後的效果，要以它為準。

### 一對一配對（匈牙利演算法），測試日確認結構後才開（comp1，信心 medium）

- 預期增益：一題對一人時 +0.05（模擬 0.924→0.978）；不是時 −0.10
- 依據：自己模擬：5 盤題 0.811→0.935；dup80 情境 0.930→0.833。
- 工作量：約 2 天；Colab 0 次
- 使用者要做：沒有（11/04 拿到測試集後由 Claude 檢查結構）。
- 主要風險：官方沒寫一位候選只對一題；結構判斷錯了會扣約 0.10；要先寫好偵測規則並登記（c1_headroom 有偵測器的雛形）。

### 從零訓練 SE 式強度網路（KataGo 被判不能用時的備案）（comp2，信心 low）

- 預期增益：不確定（論文同一玩家換算 0.67～0.74，但用了 242 GPU 小時；小網路的效果沒有人驗證過）
- 依據：Chen SE（ICLR 2025）；9/29 估計 20b256c 在 G4 約 21 小時（results/prereg04/design/understand_prior.md 第 5 節）
- 工作量：約 6 天；Colab 3 次
- 使用者要做：多次長時間的 Colab 訓練（可能要跨夜），上傳與回報。
- 主要風險：算力和 11/03 的期限；03 的經驗是新模型逐盤更準，但整題只多 +0.001～+0.004。

### 5 盤題專用的逐手模型（GE2E＋Transformer 聚合）（comp1，信心 low）

- 預期增益：上限約 +0.03～+0.05（估計：只有 5 盤題的 100 題有空間）
- 依據：NeurIPS 2021 stylometry；我們 10 盤以上已經 0.93～0.98
- 工作量：約 6 天；Colab 2 次
- 使用者要做：Colab 訓練和推論，上傳 Drive 並回報。
- 主要風險：工程量大；和 hubness 修正、一對一配對的增益會重疊；期限。

## 未解問題

- REPORT.md 沒有寫成：harness 擋下了子代理寫報告 .md（錯誤訊息：Subagents should return findings as text）。完整內容在本結果的 verdicts 和 opportunities 裡；要的話請上層代為寫入 results/prereg06/explore/prior_art/REPORT.md。
- KataGo HumanSL 能不能用，由規則主題判斷。簡報第 15 張（從零訓練）和第 80 張（pretrained 是現成的改進方向）兩句都有。
- HumanSL 的 rank_ 等級和幽玄 D～6D 怎麼對應，要在 inner 上試跑才知道。這是決定增益的關鍵實驗。
- 官方測試集的結構要 11/04 才知道：競賽一是不是一題對一人；競賽二每級幾題；同一人會不會出多題。
- CSLS 和 dual softmax 疊在 Q04 融合分數上還剩多少：以兄弟主題 c1_headroom 的結果為準。
- 沒查到的：Finding Elo 冠軍分數、Kaggle Recursion 一對一配對的加分、2023 屆的排行榜分數。

## 寫出的檔案

- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\prior_art\extract_slides.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\prior_art\convert_literature.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\prior_art\convert_literature.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\prior_art\sim_c2_label_matching.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\prior_art\sim_c2_label_matching.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\prior_art\sim_c1_assignment.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\prior_art\sim_c1_assignment.json`
