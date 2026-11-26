# prereg06 探索：rules（2026-10-02）

> 這份由 workflow `wf_85ecc07b-a64` 的探索代理寫成。子代理不能直接寫 .md，主 session 照它回傳的結構化結果（同資料夾 `result.json`）原文排版存檔，沒有改內容。數字都是設計階段、只用訓練玩家（fit／inner），不是驗證集結果。

## 摘要

1. 兩場的正式規則（AIdea「規則」分頁，兩場一字不差）沒有禁止外部資料、預訓練權重或 KataGo。唯一相關的是第 3 條：可以用外部開源資料、第三方開源程式和生成式 AI，但要遵守著作權法，並在賽後報告寫出來源和「對結果的貢獻」。
2. 簡報第 15 張「所有人都從零訓練」不是規則。標題是「為什麼拿掉預訓練模型」，講的是今年的官方教學不再附模型檔（2023 那屆的教學附了 3 個 .h5；第 17 張寫「本版不含預訓練模型」）。同一份簡報第 80 張把 pretrained 權重列為「現成的改進方向」。
3. 用 KataGo（引擎或 HumanSL 權重）當特徵：判定是規則允許、必須揭露，被取消資格的風險低。出題單位主持人顏士淨是 arXiv 2505.00279 的共同作者，那篇就用 KataGo HumanSL 估計圍棋棋力。KataGo 的程式和權重都是 MIT 類授權。HumanSL 程式碼裡的來源代碼沒有幽玄之間，所以看過測試棋的疑慮低（這是推論）。
4. 真正的風險在報告審查：最終名次由 Private LB 加報告審查一起核定，評分看完整性、正確性和程式原創性。如果用 KataGo，「原創性」那一項會有中度不確定，所以揭露、消融（有和沒有 KataGo 各多少分）、可重現的程式都是必要的。不揭露反而變成高風險。
5. Colab 不是圍棋的規定（那是 C1 的規定）。得獎隊伍要交可重現的原始碼；Private LB 前 25% 且高於 baseline 的隊伍要交 7 項報告，其中包含訓練權重和額外資源。
6. 一人兩隊：兩場同時參加有明文允許（簡報第 6、95 張）。但「隊伍間不可私下共享程式及特徵值」用在同一個人的兩隊會怎樣，原文沒寫。兩場要各自呼叫 KataGo，不要把一場的模型輸出拿給另一場當特徵。
7. 外部棋譜可以用，但資料 README 禁止用來識別真實玩家，所以不能拿外部的幽玄之間棋譜去對照測試題。
8. 看不到的部分：AIdea 的討論、資料分頁要登入；報名系統 info/38、39 要登入；報告檢核系統 10/2 回 504。
9. 子代理不能寫 REPORT.md（工具擋下），完整報告內容放在這份輸出的 verdicts 和 open_questions。腳本和抽取結果已存在 results/prereg06/explore/rules/。

## 判定與證據

### (a) 外部資料

- 判定：允許，但要揭露（明文）。限制：不能用來識別真實玩家；得獎時要繳交用到的外部資料集
- 證據：AIdea 規則第 3 條（兩場相同）原文：「若使用額外的自製資料或外部開源資料、生成式AI工具」→「須遵守中華民國著作權法」→要「說明參賽者基於外部資料對本次競賽結果的貢獻」；另有「如有爭議，主辦單位保有最終決定權」。時程表「上傳報告」列：要交「自行製作或使用之外部資料集（如有）」。資料 README「使用範圍」：「不得用於識別真實使用者」。URL：https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403 （競賽一 dc0b799e-… 文字相同）；https://github.com/AILAB-NDHU/aicup2026-go-dataset 。2026-10-02 抓取，SHA 記在 results/prereg06/explore/rules/fetch_log.json

### (b) 預訓練模型／權重（例如 KataGo HumanSL）

- 判定：規則沒有點名，屬於模糊地帶但偏向允許。它落在第 3 條「外部開源資料／第三方開源程式」的範圍內，用了就要揭露。簡報第 15 張只代表出題者的期待，不是規則
- 證據：規則全文中英兩版都沒有出現「預訓練」「pretrained」「KataGo」「引擎」。規則第 4 條要求報告寫「(5)各項參數之設定(包括訓練權重)說明」和「(7)是否使用額外資源並說明來源」。對照姊妹賽 C1（https://www.aidea-web.tw/topic/3c8ec48b-0dff-4b14-9b5e-d91bbb98c699）：它明寫可用「預訓練模型」，圍棋頁沒有這種逐項列舉。簡報第 80 張：「這是現成的改進方向」

### (c) 外部圍棋引擎（KataGo）當特徵抽取器

- 判定：允許但要揭露。規則沒點名 KataGo，但它屬於規則明文列出的「網路上的第三方開源程式」，而且有 MIT 類授權，符合「已取得合法授權」
- 證據：規則第 3 條原文「包括但不限於網路上的第三方開源程式」。規則第 5 條要求作品「均屬參賽隊伍之原創或已取得合法授權」。KataGo 程式授權在 https://github.com/lightvector/KataGo/blob/master/LICENSE （MIT 類）；權重授權在 https://katagotraining.org/network_license/ ，明寫涵蓋「KataGo Human SL Network」

### (d) 生成式 AI

- 判定：允許，但要在書面報告揭露來源和貢獻（明文）
- 證據：規則第 3 條直接列出「生成式AI工具」。範本在 D:\AI-Portfolio\competitions\_drafts\disclosure-template.md 第 D.1 節

### (e) 重現（Colab）與報告

- 判定：Colab：規則沒提（那是 C1 的規定）。得獎隊伍必須交可重現的原始碼。Private LB 前 25% 且高於 baseline 的隊伍（最多 30 隊）必須交 7 項報告。報告審查會影響獎狀和最終名次，截止後不能補交
- 證據：「領獎資格與方式」：「提交可重現該模型的原始程式碼與最終報告」，以及「以私有版排行榜…成績及報告審查結果為核定依據」；評量標準是「(1)報告完整性、(2)報告正確性、與(3)程式原創性」。規則第 4 條列出 7 項：成員、演算法、前處理原始碼、訓練原始碼、參數（含訓練權重）、執行環境、額外資源。go.aicup.tw/competition 的「提醒您」第 2 點寫：排名、報告、程式碼都是「核定最終名次之重要依據」。C1 頁面才有「Google Colaboratory 中成功執行」。10/2 的人數：競賽二 144 人／98 組，競賽一 119 人／80 組

### (f) 隊伍間共享、一人兩隊

- 判定：兩場同時參加：允許（簡報明寫）。同一場用多個帳號：禁止。同一個人的兩隊之間共用程式或特徵：原文沒寫，屬於模糊地帶
- 證據：簡報第 6 張「兩個子題可獨立報名,也可同時參加」，第 95 張「也可以都做」。規則第 2 條「不得註冊多個帳號參加同一項競賽」。規則第 8 條「各隊伍間不可私下共享程式及特徵值，賽後報告應獨立撰寫」；第 9 條寫，報告或程式「雷同」會影響所有涉入隊伍的評審成績。報名系統的使用者條款（go.aicup.tw/static/docs/使用者條款.docx）只講個資，沒有參賽規則

### 簡報第 15 張「所有人都從零訓練」是規則還是說明

- 判定：這是在說明官方教學和 baseline 的改版，不是規則
- 證據：pptx 的 SHA-256 是 ff2b6033…842ae4，和 9/26 那次相同，共 96 張，沒有講者備註。第 15 張標題「Update · 為什麼拿掉預訓練模型」，副標「沒有 checkpoint 可以載入」。第 17 張寫「本版不含預訓練模型」；第 18、39、56、58、75 張把 baseline 標成「無預訓練」。2023 那屆的教學 README（AILAB-NDHU/AIcupTutorial）寫 baseline 模型「shared in this repository」。第 80 張寫「兩個子題目前都沒用 pretrained 權重,這是現成的改進方向」。全份簡報中「外部」「KataGo」「AI 工具」「生成式」都是 0 次。「規則」出現在第 2、10、19–23、27、31、37 張，講的全是圍棋規則或檔名規則。逐張結果在 slides_hits.json

### 用 KataGo HumanSL 或引擎當特徵的風險

- 判定：違規、被取消資格：低。電子獎狀：揭露完整的話低到中，不揭露就是高。原創性評分：中。觀感（第 15 張）：低到中。測試資料外洩疑慮：低。重現：中（工程風險）。一人兩隊：低到中（兩場要各自呼叫）
- 證據：KataGo v1.15.0（2024-07-19）的 release 說明 HumanSL「trained on a large number of human games」，檔案 b18c384nbt-humanv0.bin.gz 是 99,066,230 bytes，.ckpt 是 323,752,318 bytes（GitHub API 的數字，沒有下載）。python/katago/game/sgfmetadata.py 的來源代碼只有 OGS、KGS、FOX、TYGEM、GOGOD、GO4GO；本賽資料來自幽玄之間（docs/DATA.md）。arXiv 2505.00279 的作者含 Shi-Jim Yen，第 4.2 節寫「we employed KataGo HumanSL for Go」

## 機會

### 競賽二：KataGo HumanSL 逐手各等級落子機率當特徵，再和現有的 Q02 融合（comp2，信心 low）

- 預期增益：估計，未實測：過閘門的話比現有方法高 +0.05 到 +0.15。出處是 proposal2.md 第 31 行，從 Kuboki 等人的論文在「同一位玩家」設定下換算成本題指標，得到 0.718–0.755；現有 Q02 本機 0.6236
- 依據：規則允許，要揭露（本次查核）。前人：arXiv 2505.00279（出題者是共同作者）的消融顯示，拿掉 HumanSL 影響最大（understand_priorArt.md 第 84 行）。但那篇用野狐資料、分 11 組，不能直接當預測值
- 工作量：約 6 天；Colab 3 次
- 使用者要做：規則面不需要使用者做任何事。選做：登入 AIdea，只讀兩場的「討論」分頁，查有沒有官方回覆過預訓練或 KataGo，回報截圖。之後要逐句審報告裡的揭露段落
- 主要風險：「程式原創性」可能被扣分，報告審查又會影響名次核定。HumanSL 權重 99 MB 到 324 MB，Colab 環境和後端相容性有風險（舊版 TensorRT 在 Blackwell 上有問題）。必須在 11/03 前連測試日工具一起做完。消融數字必須來自實測

### 競賽一：用 KataGo 網路的中間特徵或 HumanSL 落子機率當風格特徵，和 Q04 的開局 token 合併（comp1，信心 low）

- 預期增益：未知，沒有證據。Q04 本機已經 0.8830，剩下的空間小
- 依據：規則允許，要揭露。沒有找到前人在玩家辨識上用 HumanSL 的實測
- 工作量：約 4 天；Colab 2 次
- 使用者要做：同上：選做只讀討論區；之後審報告
- 主要風險：效益不明。兩場共用同一份抽好的特徵檔，可能被看成「共享特徵值」，所以要在兩場的程式裡各自呼叫

### 外部人類棋譜擴充訓練（非本賽來源）（both，信心 low）

- 預期增益：未知，沒有實測
- 依據：規則第 3 條允許外部開源資料（要揭露並繳交）。但資料 README 禁止識別真實使用者，也不能拿幽玄之間的棋譜去比對測試題
- 工作量：約 5 天；Colab 2 次
- 使用者要做：如果要採用，要先確認外部資料來源本身的授權；這可以由 AI 唯讀查核
- 主要風險：來源授權。分佈和幽玄之間不同。得獎時要交出外部資料集。被認定成在識別玩家的風險

## 未解問題

- REPORT.md 沒有寫成：子代理寫 .md 報告檔被工具擋下（「Subagents should return findings as text」）。完整判定、揭露清單和風險表都在這份輸出裡；要放檔案的話，由上層 session 寫進 results/prereg06/explore/rules/REPORT.md
- AIdea 的「討論」「資料」分頁沒登入是灰色、點不開（10/2 用內建瀏覽器試過）。官方有沒有在討論區回答過能不能用預訓練或 KataGo，只有使用者登入後只讀才看得到
- 「程式原創性」在報告審查裡占多少、怎麼評，以及社會人士隊伍用了外部權重會不會在名次核定時被調整，原文都沒寫
- 競賽報告檢核系統 https://aicupreport.csie.ncu.edu.tw 在 10/2 兩次都是 504 或逾時。報告格式和上傳頁的要求還不知道
- HumanSL 完整的訓練資料清單官方沒公布。這次只根據程式碼裡的來源代碼（沒有幽玄之間）推論沒有外洩，無法完全排除
- 第 80 張的「pretrained 權重」是指外部權重，還是自己先預訓練再微調，原文沒講清楚
- EP1–EP5 的 YouTube 影片口述沒有看，簡報以外有沒有補充規定不知道
- 依 9/23 的決定不寄信、不填官方 Q&A 表單，所以沒有官方背書。規則寫「若需要官方正式回覆，務必透過表單提出」

## 寫出的檔案

- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\rules\fetch_sources.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\rules\fetch_log.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\rules\extract_slides.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\rules\slides_hits.json`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\rules\extract_docx.py`
