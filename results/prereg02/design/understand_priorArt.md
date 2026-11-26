# C10 研究報告：棋力／玩家辨識的前人做法、規則原文、KataGo 成本（2026-09-26）

## 結論

**競賽二（棋力）**
- 證據最強的做法是「逐手看落子，算每個等級的人下這手的機率，再把整題累加」。官方 baseline 不是這樣做。
- 公開論文在「一題的棋都來自同一位玩家」的設定下，換算成本題指標約 **0.67–0.75**。我們本機 baseline 是 0.480（取機率最大）／0.492（取期望得分）。
- 但論文用的是野狐資料、11 級分組，也沒寫玩家有沒有和訓練集重疊，所以這只能當上限參考，不是預測值。
- 做法有兩條：
  - (a) 用 KataGo HumanSL 當特徵，外加一個小的堆疊模型。規則允許，要揭露。
  - (b) 自己從頭訓練一個「依等級條件化」的落子模型。
- 最便宜的第一步：跑一次只做推論的 Colab（約 2 個運算單元），把逐盤的 logits 存下來，才能做聚合方式和校準。目前只存了每題平均後的機率，這兩件事做不了。

**競賽一（辨識玩家）**
- 證據較好又便宜的有三件：
  1. 訓練更久、換 loss。
  2. 加開局特徵：前人研究顯示開局最能認出人，本機 CPU 就能先測。
  3. 在同一個候選池內做 CSLS 相似度正規化：只用 CPU，而且可以直接用已存的 embedding，事前登記後打一次分。
- 「一題對一人」的指派要等 11/04 確認測試集結構後才能用。

**規則**
- 正式規則沒有禁止外部資料、預訓練模型或 KataGo，只要求揭露來源和貢獻、遵守著作權。
- 出題者（顏士淨教授）的簡報第 80 張明寫預訓練權重是「現成的改進方向」。
- 但第 15 張寫「所有人都從零訓練」，有一點反向訊號。
- 資料 README 禁止拿資料「識別真實使用者」，所以不能拿外部野狐棋譜庫去對照。

## 1. 規則原文（沒登入能看到的部分）

兩場 AIdea 頁面的規則文字相同，只差標題、評分和 baseline。讀取日 2026-09-26，當天人數：競賽二 91 人／69 組，競賽一 88 人／63 組。來源：https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403 、https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688 。

- **外部資料、第三方程式、生成式 AI**：「若使用額外的自製資料或外部開源資料、生成式AI工具，包括但不限於網路上的第三方開源程式…須遵守中華民國著作權法。參賽隊伍應於比賽結束繳交的書面報告文件中提供相關來源、並說明參賽者基於外部資料對本次競賽結果的貢獻。如有爭議，主辦單位保有最終決定權。」全頁沒有出現「預訓練」「KataGo」「引擎」「集成」「算力」這些字。
- **不能人工改答案**：「參賽者所上傳的結果不得使用人為修正」。
- **隊伍之間**：「各隊伍間不可私下共享程式及特徵值，賽後報告應獨立撰寫」。
- **重現**：
  - 學生組得獎隊伍要「提交可重現該模型的原始程式碼與最終報告」。
  - 前 25% 的報告要包含「(5)各項參數之設定(包括訓練權重)」「(6)程式執行環境說明」「(7)是否使用額外資源並說明來源」。
  - 評審看「(1)報告完整性、(2)報告正確性、與(3)程式原創性」。
- **上傳**：「每日上傳至多 5 次，並以最後一次上傳之預測結果獲得之分數進行排名」。
- 規則沒有提到能不能利用測試題或候選池本身的結構。相關的限制只有下面兩處：
  - 教學 README 寫候選池「Public 與 private 兩池互斥，不可混用」，每份 400 題、每題 5–20 局；每個池 400 人、每人 100 局。來源：https://github.com/AILAB-NDHU/AICup-2026-Tutorial （repo 內附副本 `third_party/aicup2026_tutorial/README.md`）。
  - 簡報第 73 張：「五個 player_id 不能重複,且不能跨池混用」。
- **資料 README**：「本資料集僅供 2026 AI CUP 競賽使用。棋譜已匿名化，不得用於識別真實使用者。」沒寫資料來源是哪個平台。來源：https://github.com/AILAB-NDHU/aicup2026-go-dataset （2026-09-26 讀取）。
- **官方課程簡報**：https://www.aicup.tw/_files/ugd/7fbdbf_dee0824b36cb4dd2a4af2497347a04f7.pptx （2026-09-26 下載，SHA-256 `ff2b6033…42ae4`）
  - 第 15 張「為什麼拿掉預訓練模型」：「所有人都從零訓練,比的是方法設計,不是誰拿到更好的預訓練權重」。
  - 第 80 張（EP5，講者顏士淨 NDHU AI Lab）：「兩個子題目前都沒用 pretrained 權重,這是現成的改進方向」。
  - 第 87 張：「Cross-entropy 跟 Rank 評分(差 1 級得 e⁻¹)不同 — 試 ordinal regression 或自訂 loss」。
  - 出題者自己估的效益，都是估計值，沒有實測：
    - D4 對稱擴充：Rank +3~5%、Player +2~4%（第 81 張）
    - 加氣與提子特徵：Rank +4%、Player +2%（第 82 張）
    - Set 聚合：「再拿 2~4 個百分點」（第 85 張）
    - 集成 +1~5%，「不集成幾乎拿不到前段名次」（第 84 張）
  - 第 95 張：「效益百分比都是相對估計值…務必自己做消融實驗驗證」。
- 已查核過、這次沒有變的：`docs/REGISTRATION.md`；`_research-2026-09-23/evidence.md` 第 41–46 行（時程、獎狀條件、計分、生成式 AI 揭露）。
- 還看不到、要登入才看得到的：討論區（有沒有人問過 KataGo）、「資料」分頁的授權說明。

## 2. 前人做法：棋力

- **官方 baseline 的結構**（repo 副本 `third_party/aicup2026_tutorial/utils.py`、`rank-prediction.ipynb`）：
  - 輸入是最後 250 個盤面，拆成黑、白兩層，共 500 層，直接疊在第一層卷積。
  - **沒有給模型「目標玩家執哪一色」**。
  - 整題的答案是每盤 softmax 的**算術平均**：`avg = np.concatenate(probs).mean(0)`。
- **Chen, Shih, Wu, ICLR 2025「Strength Estimator (SE)」**（2025-02-24）https://arxiv.org/abs/2502.17109 ，細節見 https://ar5iv.labs.arxiv.org/html/2502.17109
  - 做法：用 Bradley-Terry 模型比較「同一個局面、不同等級的人下的手」。
  - 資料：野狐，3–5k 到 9d 共 11 組，每組 4.5 萬盤。
  - 整題用幾何平均合成一個分數。
  - 結果：15 盤正確率超過 80%，前人方法 100 盤只有 49%。
  - 網路是 20 block、256 channel，訓練花了 242 GPU 小時（RTX A5000）。
  - 論文沒寫玩家有沒有和訓練集分開。
  - 程式和權重：https://rlg.iis.sinica.edu.tw/papers/strength-estimator/ ，頁面沒寫授權；也不涵蓋我們的 D、C 兩級。
- **Kuboki, Ogawa, Hsueh, 顏士淨, Ikeda（arXiv 2505.00279，2025-05-01）**https://arxiv.org/html/2505.00279v1
  - 作者群包含本賽出題者。
  - 特徵：SE 分數，加上 KataGo HumanSL 在 10 個等級下「實際落子的機率」的幾何平均（用 preaz_ 設定），再加上輸棋分數損失；用 LightGBM 訓練。
  - 「同一位玩家」抽樣的結果，以及換算成本題指標：

| 方法 | 盤數 | 完全猜中 | 差一級以內 | 換算本題指標（取機率最大） |
|---|---|---|---|---|
| Kuboki（SE＋HumanSL＋損失） | 5 | 58.1% | 95.3% | 0.718 |
| Kuboki | 10 | 61.0% | 96.2% | 0.740 |
| Kuboki | 15 | 63.2% | 96.6% | 0.755 |
| Chen SE（只用人類棋譜從頭訓練） | 5 | 51.6% | 93.1% | 0.669 |
| Chen SE | 10 | 59.4% | 96.0% | 0.729 |

  - 對照：隨機抽不同玩家的棋時，n=15 可到 89.4%；換成同一位玩家就掉很多，因為個人風格讓相鄰等級重疊。
  - 消融（隨機抽樣、n=20）：全部特徵 91.7%；拿掉 HumanSL 機率 86.4%；拿掉 SE 分數 87.7%；拿掉損失特徵反而是 91.9%。**損失特徵幾乎沒貢獻**。
- **Moudřík & Neruda 2016**：CNN 把玩家分強、中、弱三級，單一局面 71.5%、整盤 77.9%。https://researchgate.net/publication/309559342 。同作者的特徵加堆疊：https://arxiv.org/abs/1512.08969 （2015-12）、https://arxiv.org/pdf/1512.09254 。
- **西洋棋**
  - Maia（KDD 2020）：9 個分等級的模型。https://www.cs.toronto.edu/~ashton/pubs/maia-kdd2020.pdf
  - Maia-2（NeurIPS 2024）：一個網路加上「依棋力調整的注意力模組」。https://arxiv.org/abs/2409.20553 。這是「從頭訓練等級條件化落子模型」可以參考的架構。
  - Kaggle Finding Elo（2014）：冠軍用 Stockfish 評估差的統計量，加上局面特徵。https://www.kaggle.com/c/finding-elo/discussion/13008 。這條只看到搜尋摘要，頁面本身抓不到，**未查核**。
- **序數迴歸、校準、聚合**
  - CORN 比 CORAL 好：https://arxiv.org/abs/2111.08851 （Pattern Analysis and Applications 2023）。
  - SORD soft label 在 Image Aesthetics 資料集上類別正確率約 +2%：https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html
  - Temperature scaling：https://arxiv.org/abs/1706.04599 （ICML 2017）。
  - 注意力式 MIL：https://arxiv.org/abs/1802.04712 （ICML 2018）。
  - **反向證據**：Kittler 等人 1998 的實驗和敏感度分析，都是相加規則（sum rule）比相乘規則（product rule）好，因為相加對估計誤差比較穩。https://people.sabanciuniv.edu/berrin/cs512/reading/kittler-combining.pdf 。所以「改用 log 平均」要實測，不能直接假設會變好。
- **往屆**：AI CUP 2023 秋季賽也是東華出題，508 隊，題目是 10 級／1 段的落子預測，加三類棋風辨識。https://tbrain.trendmicro.com.tw/Competitions/Details/29 、https://github.com/AILAB-NDHU/AIcupTutorial 。找不到公開的冠軍做法。

## 3. 前人做法：辨識玩家

- **西洋棋 behavioral stylometry（NeurIPS 2021）**https://ar5iv.labs.arxiv.org/html/2208.01366
  - 做法：GE2E loss，逐手特徵加 Transformer 聚合，玩家向量取平均（centroid）。
  - 400 位候選、各 100 盤查詢和參考、去掉前 15 手：P@1 0.953。
  - 保留開局時更高：2,844 位候選下 0.982，去掉前 15 手是 0.860。
  - 盤數少時會急降；論文說大約 50 盤開始飽和。
  - 沒看過的玩家（0.86）和看過的（0.853）差不多。
- **個人化模型（KDD 2022）**https://www.cs.toronto.edu/~ashton/pubs/maia-individual-kdd2022.pdf
  - 400 位候選、10 盤：全部手數 0.86；從第 10 手起只剩 0.47；從第 30 手起 0.11。**開局最能認出人**。
- **圍棋（野狐，200 人）**：Lin 等人 https://arxiv.org/html/2408.06051v2
  - **只看每盤前 10 手**、查詢 10 盤：45.5–46.5%；25 盤：70.5–75.5%。
  - 看整盤、100 盤：最高 97%。
- **Loss 的選擇**
  - GE2E 比 TE2E 的 EER 降超過 10%：https://arxiv.org/abs/1710.10467
  - 另一方面，嚴格比較後，各種 metric learning loss 的差異很小：https://link.springer.com/chapter/10.1007/978-3-030-58595-2_41 （ECCV 2020）。
- **同一池內的轉導式修正**
  - CSLS 在 Procrustes 對齊上，P@1 從 77.4 升到 81.4（en-es）、從 74.9 升到 81.1（en-fr）：https://ar5iv.labs.arxiv.org/html/1710.04087
  - k-reciprocal 重排：rank-1 從 72.54% 升到 74.85%：https://arxiv.org/abs/1701.08398

## 4. KataGo：成本與授權

- **HumanSL 模型**：v1.15.0（2024-07-19）釋出 `b18c384nbt-humanv0`，有 `.bin.gz` 和 PyTorch `.ckpt` 兩種格式。https://github.com/lightvector/KataGo/releases/tag/v1.15.0 、https://katagotraining.org/extra_networks/
  - 可設定的等級是 `rank_20k`…`rank_9d`（也有 `preaz_`），每次查詢可用 `overrideSettings` 換；要加 `includePolicy:true` 才會回傳 `humanPolicy`。https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md
  - 等級是網路的輸入，所以**每換一個等級就要多算一次網路**。
- **授權**
  - 程式碼是 MIT：https://github.com/lightvector/KataGo/blob/master/LICENSE
  - 網路權重是 MIT 式授權，涵蓋 extra_networks 裡的「KataGo Human SL Network」：https://katagotraining.org/network_license/
- **訓練資料來源**：官方沒寫，但 metadata 的來源代碼有 OGS、KGS、FOX、TYGEM、GOGOD、GO4GO。https://github.com/lightvector/KataGo/blob/master/python/katago/game/sgfmetadata.py
  - 如果本賽資料也來自野狐（沒有證實），HumanSL 可能看過同一批棋，本機驗證分數會偏樂觀。
- **速度（公開實測，b18c384nbt）**
  - RTX 4070 + TensorRT：2,627 nnEvals/s（2023-06-08，https://lifein19x19.com/viewtopic.php?f=18&t=19196 ）；調整執行緒後約 6,500 visits/s（2023-07，https://forums.online-go.com/t/katago-speeds-of-different-hardwares/48463 ）。
  - RTX 5070 + TensorRT：約 3,262 visits/s（2025-05-15，https://songyp.com/blog/katago-workstation-build-and-bench ）。
  - **找不到 A100、G4 的實測數字**。
  - 舊版 KataGo 的 TensorRT 在 Blackwell（RTX 5090）上會初始化失敗：https://github.com/lightvector/KataGo/issues/1041 （2025-03）。G4 也是 Blackwell，所以在 G4 上走 PyTorch 讀 `.ckpt` 比較穩。
- **每小時能處理幾盤（估算，沒實測）**
  - 每盤約 110 手是目標玩家下的（手數中位數 213–235，見 `docs/DATA.md`），乘上 10 個等級，約 1,100 次網路運算。
  - 每秒 3k 次 → 約 9,800 盤／小時；每秒 10k → 約 33k 盤／小時；每秒 30k → 約 98k 盤／小時。
  - 競賽二大約要處理 6.5 萬盤：測試約 9.6k、本機驗證 19,156、堆疊模型的訓練題約 3.6 萬。大約是 G4 **0.7–6.6 小時，約 6–57 個運算單元**，範圍很寬。
  - 所以要先做 15 分鐘的試跑（約 2 個運算單元）量實際速度。
  - 如果要算輸棋分數損失（例如每個局面 100 visits），每盤要約 22k 次運算，每小時只能處理幾百盤；而且消融顯示它沒貢獻，不建議做。

## 5. 對本題指標的證據與成本排序

| # | 做法 | 證據 | 成本 |
|---|---|---|---|
| R1 | 下一次 Colab 先做「只推論」：輸出內部驗證和本機驗證的逐盤 logits，再在本機比較「機率平均」和「log 平均＋依盤數調的 temperature」，最後用期望得分規則 | 前人都用幾何平均；Kittler 是反例；出題者估 +2–4 分（未實測） | 約 2 單元；CPU 端很小 |
| R2 | 逐手「各等級落子機率」：(a) KataGo HumanSL 特徵＋小的堆疊模型；(b) 從頭訓練一個 10 個輸出頭的等級條件化落子網路 | 同一玩家設定下換算約 0.67–0.75（見第 2 節） | (a) 試跑約 2 單元，之後約 6–57 單元，還要處理 Colab 環境；(b) 新 notebook 加審查，G4 約 2–4 小時（估計）。SE 原文的大網路要 242 GPU 小時，我們付不起，只能用小網路，效果沒人驗證過 |
| R3 | 改進 baseline：D4 擴充、訓練更久、加氣／提子特徵、soft label 或 CORN、集成 | 只有出題者的估計和 SORD 的 +2% | 每次重訓 G4 約 0.7–2 小時 |
| P1 | CSLS 相似度正規化（k 先固定，例如 10，寫進事前登記） | 在其他領域 P@1 +3–7 分 | 只用 CPU、用已存的 embedding，打一次分 |
| P2 | 開局特徵：前 10–20 手，用 8 種對稱轉成標準形後做 n-gram 相似度，再和 embedding 分數合併 | 西洋棋、圍棋的研究都顯示開局最能認出人 | 本機 CPU，在內部分割上測 |
| P3 | 訓練更久（loss 到第 5 個 epoch 還在降）、PK 取樣、換 GE2E 或 softmax-proxy loss | GE2E 的 EER −10%；但 ECCV 2020 說各 loss 差異小 | G4 約 1–1.5 小時 |
| P4 | 用競賽二的模型推測每位候選人的等級，把等級差兩級以上的候選人往後排 | 沒有文獻 | 候選人 4 萬盤的推論，約數分鐘 |

- 「一題對一人」的指派，要等 11/04 確認測試集結構後才能用（`docs/DATA.md` 已經這樣規定）。
- 不建議做：
  - 拿外部野狐棋譜庫去比對：違反「不得用於識別真實使用者」。
  - 輸棋分數損失特徵：成本高，而且沒有貢獻。

## 還不知道的

- 本賽資料來自哪個平台。
- 論文裡的玩家有沒有和訓練集分開。
- G4 上實際每秒能跑多少次網路運算。
- 討論區有沒有官方回覆過能不能用 KataGo。

HARD RULES 全程遵守：沒有開啟任何 `*answers*.csv`，沒有跑 `score.py`，沒有寫入 repo。暫存檔都在 scratchpad：`C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\`，包括 `comp1.html.txt`、`comp2.html.txt`、`res_slides.txt`、`res_kdd22.txt`、`res_neurips21.txt`。這個資料夾裡的其他檔案原本就在，不是這次產生的。