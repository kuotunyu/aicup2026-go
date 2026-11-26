# prereg06 探索：integration（2026-10-02）

> 這份由 workflow `wf_85ecc07b-a64` 的探索代理寫成。子代理不能直接寫 .md，主 session 照它回傳的結構化結果（同資料夾 `result.json`）原文排版存檔，沒有改內容。數字都是設計階段、只用訓練玩家（fit／inner），不是驗證集結果。

## 摘要

結論：06 可以接進測試日，不動 R20／04／Q04，但要先補三件事，11/03 23:59 前合併。
1. 打分機會：競賽二剩 2 次（用了 02、04；03 沒送分），競賽一剩 3 次（用了 02）。建議 06 每場最多用 1 次，競賽二留最後 1 次給 split v2 或查 bug 重跑。
2. 06 要和現在的首選比：競賽二的 H6R 對照組用 04（Q02，驗證 0.6236），競賽一用 Q04（0.8830）。增益要約 +0.006–0.012 以上才有一半機會被採用（推估）。
3. official_log.py 現在不認得 06：實際呼叫的結果是 results/test06 的檔會被當成 R20、標成「prereg02 R20」。04 符合條件後，非 04 的競賽二檔在 prepare 會被拒。一定要先「只插入新行」擴充它，而且沒有 06 紀錄時輸出要逐位元組不變（TagVersusNew 測試）。
4. 04 的測試日路徑有 21 個 git 檔，現在沒有任何指紋或測試保護。06 動工前要先替它們做指紋和 tag。
5. 最後一版的規則：PREREG_06 自己寫 5.3 式的規則，順序是 06 → 04 → R20 → R10，06 要做成 Q02＋新東西。PREREG_04 加 rev 4，只加一列和一節，指向 06。不要寫 PREREG_02 rev 9，TESTDAY.md 也不動，照 04 取代 U5 的前例。
6. 防誤改閘門：寫檔前後各跑一次，都是 41 passed。H6R 打分後會一直有 1 項失敗（帳被釘在 25 列，打一次分多 2 列）。這是 TESTDAY_04 允許的，但 CLAUDE.md 的說法要改，另加 test_p06_hygiene。
7. 時程（要 Colab 訓練的）：10/12 PREREG_06，10/16–10/18 使用者跑 Colab（避開 10/19–10/27 玉山），10/21 打分（最晚 10/27），10/30 合併（硬截止 11/03）。只用 CPU 的：PREREG_06 在 10/14–10/16。
8. 測試日：競賽二 06 排在 04 apply 之後（11/08–11/10），K6' 建議 11/19。只用 CPU 的競賽一 06 可以 11/05–11/06 做。
9. 使用者多做的事：只用 CPU 的每場只多 1 次上傳。新 GPU 模型要 10 月多 1–2 次 Colab、測試日多 1 次 Colab 和 1 次上傳，最壞再多 1 次重跑。
（REPORT.md 被工具擋下沒有寫成檔案，報告內容都在下面各欄；事實整理的腳本和 JSON 已寫入 integration/。）

## 判定與證據

### 1. 驗證集打分預算（每場剩幾次、在什麼條件下能用）

- 判定：競賽二：整季 4 次新方法，已用 2 次（02 H2R、04 H4R），03 未送分、機會保留，剩 2 次。競賽一：已用 1 次（02 H2P），剩 3 次。建議 PREREG_06 寫死：競賽二 06 最多用 1 次，就是 03 沒用掉的那一次，第 4 次照原文留給 split v2 或查 bug 重跑；競賽一 06 最多用 1 次。H6R 的 A 側用 results/prereg04/rank/pred_B.csv（Q02），H6P 用 results/prereg02/player/player_val_top5.csv（Q04），這樣才有理由在最後一版換掉它們。
- 證據：規則原文：docs/PREREG_02_OPENING_FUSION.md 第 3 節第 2 點（第 160 行）：「每場到 11/24 最多打 4 次新方法（02、03、04 各一次，另留一次給 split v2 或查 bug 重跑）。查 bug 後重跑也算進預算」。CLAUDE.md 寫成「PREREG_02 第 2 節」是筆誤，實際是第 3 節。01 是基準不算：docs/ORAL_3MIN.md 第 147 行。03 未送分、機會保留：PREREG_03 第 2.4 節、第 11 節 (c)；results/prereg03/RESULTS.md。results/prereg04/RESULTS.md：「競賽二整季 4 次，用掉第 2 次（02、04），還剩 2 次」。競賽一：PREREG_03 第 6 節、PREREG_04 第 6 節都寫「競賽一的整季預算維持『已用 1 次』」。新登記可用自己的預算：PREREG_04 開頭範圍 (B)。integration_facts.py 實際數 results/score_ledger.csv：25 列資料，每次比較寫 2 列；used rank=2、player=1；left rank=2、player=3。帳上 A 側的值：Q02 的分等級平均 0.6236017652948923、每題平均 0.6245054776753989；Q04 0.8829546403931485。能打分的條件：VALIDATION 紀律 1／2／6／8；frozen_config commit 之後才產生驗證預測（PREREG_02 第 3 節、PREREG_04 第 2.4、2.6 節）；過送分閘門（PREREG_04 第 2.5 節）；寫死的單行指令；對照組重讀差 ≤1e-12（PREREG_03 第 3 節、PREREG_04 第 3 節）；採用要分等級平均差的 95% 下界 >0，而且每題平均差的點估計不 <0。查 bug 重跑也算預算，H6R 一出 bug 就會吃掉最後一次。採用機率（推估，results/prereg05/explore/u3_timeline_numbers.json）：差值的標準誤在 0.003–0.006 時，真實增益約 +0.0059–+0.0118 才有 50% 機會過；H4R 的實際區間是 [+0.0206, +0.0481]。

### 2a. PREREG_06 要有的內容與流程（照 04）

- 判定：照 PREREG_04 的骨架寫：開頭（驗證規則、前幾份結果、design 只用 fit／inner、這次做哪一場；兩場都做就分兩節、工具分開）、修訂紀錄（rev 2 在 check 前，rev 3 測試日程序在 K5' 前）、0 為什麼（含挑選的自由度）、1 來源（定義寫死、釘 SHA；訓練要寫配方、指定 checkpoint、seed、GPU、AI Pro 帳號、權重 SHA 和本機備份；外部模型要寫 SHA、授權、來源和規則依據）、2 合併與選擇（第一個組合要逐位重現 Q02，inner OOF 0.6101211982700248，差 ≤1e-12；選法用 0.003 容許；閘門分混色 inner 和單色假題；新來源跟執色有關就要 Δκ 檢查）、3 寫死的 H6R／H6P 指令、採用條件、對照組重讀、合理性門檻、預算那一句、4 截止點 K3'／K5'／K6'、5 測試日（保護 R20 和 04 路徑、要做的工具、5.3 式的最後一版規則）、6 另一場不交換輸出、7 刻意不做的、8 揭露、9 順序。交件模型選甲。建議競賽二的 06＝Q02＋新東西，後備鏈才乾淨。流程：PREREG_06 rev 1、design、04 路徑指紋和 tag p04-testday-pre06、PREREG_04 rev 4 → 程式、審查（rev 2）→（Colab）→ check → selfcheck → tune → commit frozen_config → apply／decide／check_submission → H6R → RESULTS 由別的代理獨立核對 → 分支 p06-testday：先記 R20 和 04 的假測試基準輸出 → 規格 → 實作 → 三個角度審查 → rev 3 → 演練（R20、04 逐位元組不變、06 端到端、負面演練、C10_SLOW=2、notebook CPU 冒煙）→ 閘門輪 → 合併。
- 證據：PREREG_04 第 0–11 節的結構、第 2.3 節 selfcheck 的數字、第 9 節順序、第 11 節第 13 條演練清單；results/test04/design/spec.md 第 0–10 節；results/test04/design/r20_fake_golden.json 的做法。04 實際花的時間（integration_facts.json 的 elapsed_hours，由 git commit 時間算）：rev 1 到 H4R 4.8 小時；測試日擴充從分支點到合併 32.13 小時（設計 10.16、實作 13.66、兩輪閘門修正 8.31）；03 從實作到使用者交回 Colab 結果 43.96 小時。

### 2b. 一定不能改的檔、防誤改閘門指令

- 判定：不能改：R20 路徑 39 個指紋檔（official_log.py 只准插入新行）；這些 pattern 底下的新程式或設計檔；results/prereg01–04/**（不改、不新增、沒有 bytecode，對 6ec5e68 不能有差異）；results/prereg05/**（只有 CLAUDE.md 規定）；tools/p02_*.py、p03_rank_fuse、p04_opp、p04_rank_fuse、c10data、decide（也不能新增 tools/p02_*.py）；score.py、make_val.py、prereg01–04 的產生器和 notebook；PREREG_01–03、score_ledger、COLAB_RUN（舊的每一行保留；COLAB_RUN 只准在檔尾加一節，標題不能以「# 測試集 04（test04）」開頭）；PREREG_04、.gitattributes、.gitignore 只准加行；VALIDATION.md 和 split_v1；data/local 永遠不開；Drive 的 C10/results、C10_bundle_v1.zip、C10_test_v1.zip。04 測試日路徑 21 個 git 檔（tools/t04_rank.py、t04_fake_tests.py、colab/build_test04_rank_nb.py、test04_rank.ipynb、make_test04_smoke_tree.py、docs/TESTDAY_04.md、tests/test_t04_* 5 個、results/test04/design/* 9 個、notebook_diff_rank.txt），加上 data/colab_upload/test04、data/prereg04/counts_fit.npz：現在沒有測試保護，要新做指紋。閘門：C10_T04_GATE=1 .venv/Scripts/python.exe -m pytest -q -rs tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py。H6R／H6P 打分後會固定有 1 項失敗（test_ledger_and_prereg_rows_pinned），TESTDAY_04 第 1 步允許；要逐列寫進 RESULTS，改 CLAUDE.md 的說法，並新增 tests/test_p06_hygiene.py（用 C10_T06_GATE=1 啟動，釘住 04 路徑指紋、帳的列數 25＋06 的列、PREREG_06 只准加行、results/prereg06 凍結）。
- 證據：tests/test_p04_hygiene.py（指紋 SHA 常數、只准加行用子序列比對、FROZEN_RESULT_DIRS、FrozenTools）；tests/test_t04_hygiene.py（BASE 6ec5e68、SAME_AS_BASE、LINES_KEPT、LEDGER_ROWS_AT_BASE=25 只在 C10_T04_GATE／C10_K5 時跑、R20PathNoNewFiles 的 pattern 判斷、COLAB_RUN 的標題規則、FrozenPrereg04、NoBytecodeInTest04、official_log 只用標準函式庫）；tests/test_t02_hygiene.py（Pins 14 個檔、答案檔字樣掃描）。git ls-files 數出 21 個 04 檔；git tag -l 只有 r20-testday-pre04。integration_facts.py 的 name_checks：t06_*、p06_*、build_test06_*、test06_*.ipynb、docs/TESTDAY_06.md、results/test06/*、results/prereg06/design/* 都 OK；tools/t02_rank06.py、colab/test06_from_test02.ipynb、results/test02/design/* 會讓 R20PathNoNewFiles 失敗；tools/p02_rank_fuse06.py 會讓 FrozenTools 失敗；results/prereg04/* 會讓 FrozenPrereg04 和 FrozenResults 失敗。閘門 10/2 實跑：寫檔前 41 passed、33 subtests、1.14 秒；寫入 integration/ 之後 41 passed、33 subtests、1.17 秒。

### 3. 最後一版的規則要怎麼改（PREREG_04 5.3、TESTDAY U5、TESTDAY_04 第 7 步）

- 判定：上傳本身照 TESTDAY「之後的上傳」(b)（之後的登記依驗證規則採用的方法）就允許，要改的是「誰是最後一版」。三處，都只能加：(1) PREREG_06 第 5.3 節是規則本體。競賽二：有一筆 06 符合就是最終。符合要全部成立：H6R 採用（06 對 Q02）、K6' 前有一次成功的正式執行並 commit、T1 是 PASS 或 NORMALIZED、06 用到的 04 輸入現在符合 PREREG_04 5.3 而且沒變、06 自己的接縫和檢查都過、報告和交件檔從 prepare 之後沒變、最多一次占次數的重跑。否則照 PREREG_04 5.3（04 → R20 → R10）；不交沒打過分的組合。競賽一：06 符合就是最終，否則照 U5（Q04 的每池程序）。翻盤只能靠 PREREG_06 的修訂，指出程式錯誤，而且時間晚於被推翻的紀錄。照 PREREG_04 第 11 節第 12 條取代 U5 的前例，不改 TESTDAY.md，也不寫 PREREG_02 rev 9（會多一列 rev，04 的閘門會把它當 R20 側變動；而且 rev 9 是留給 R20 修 bug 的）。(2) PREREG_04 rev 4：只加一列和一節，寫「PREREG_06 符合時取代 04」，其他都不變。副作用是 04 之後修 bug 要用 rev 5；TESTDAY_04 第 21 行「下一版是 rev 4」會過時。它不能被拿來推翻 04，因為 p04_revision 要求那一列的 commit 晚於被推翻的 04 紀錄。(3) official_log.py 只插入新行：check_reports 開頭加 is_t06_file 分支，06 的報告名稱要不同（t06_apply_report.json、t06_submit_report.json）；cmd_prepare 加 06 的方法名稱，prepare_rule_t04 遇到 06 直接返回；cmd_final 加 06 的規則，有符合的 06 時拒絕改選 04／R20；t04_status 遇到最終是符合的 06 不算違反；public 同名檔的警告。沒有 06 紀錄時輸出要逐位元組不變，06 的 K6' 倒數放在 t06 工具。約束照舊：規則決定、人不挑；public 只抓格式錯誤和下降 ≥0.05；最後一次上傳 11/23 23:59 前；每次上傳要使用者說 OK；R20 → 04 → 06 的順序；每天 5 次（競賽二合計）。
- 證據：docs/TESTDAY.md 第 227、229 行（U5「兩場都是 prereg02」）；PREREG_02 第 9 節第 12 條；VALIDATION 紀律 4（public 400 題標準誤最高約 0.025）；PREREG_04 第 5.3 節「不交沒在驗證集打過分的其他組合」、第 11 節第 12 條（取代 U5、翻盤要 rev ≥4 而且時間要晚）、第 20 條；TESTDAY_04 第 7 步。integration_facts.py 實際呼叫：is_t04_file('rank','results/test06/rank/submission_rank.csv') 是 False，需要的報告是 apply_test_report.json，方法標籤是「prereg02 R20 rule B (甲)」；競賽一的 results/test06/player 檔要 submit_report.json，標籤是「prereg02 per-pool self-retrieval + guard (甲)」。tools/official_log.py 第 350–412 行（method_for、check_reports）、第 467–498 行（p04_revision 的時間規則）、第 859–878 行（prepare_rule_t04 會拒非 04 檔）、第 881–908 行（final_rule_t04）、第 911–939 行（t04_status 的違反判定）。tests/test_t04_official_log.py 第 410–431 行（TagVersusNew：status 只能多最後一行 K6，dict 拿掉 prereg04 後要相同）。

### 4. 時程與使用者動作

- 判定：要 Colab 訓練的 06（B／D／E 型）：10/09（五）決定；10/12（一）commit PREREG_06 rev 1、PREREG_04 rev 4、04 指紋和 tag；10/15（四）實作和審查；10/16（五）–10/18（日）使用者跑 Colab，最晚 10/24，但要避開 10/19–10/27；10/21（三）H6R（K3'），最晚 10/27（二）；10/30（五）測試日擴充合併，硬截止 11/03（二）23:59；10/31–11/02 選做彩排；11/01 把 06 的測試日代價列進使用者的 11 月決定；11/04 test02 不變；11/05–11/07 test04 不變（apply 11/15 前，K6 11/16）。競賽二 06 在 11/08（日）–11/10（二），apply 11/18（三）前，K6' 建議 11/19（四）23:59，保守版是 11/16；登記的重跑只用 G4，最晚 11/18 中午。競賽一 06：C 型 11/05–11/06，最晚 11/21（六）。11/22、11/23 跑 status，最後一次上傳 ≤11/23 23:59。只用 CPU 的 06（A／C 型）：PREREG_06 10/14–10/16（最晚 10/23），其他相同。使用者動作：基準是測試日 3 次 Colab、3 次上傳。A／C 型每場只多 1 次上傳加 1 次說 OK，10 月和 11 月都不用多跑 Colab。B 型 10 月多 1–2 次訓練（可以同時跑），測試日多 1 次 Colab、1 次上傳，加放 notebook、下載 zip；最壞再多 1 次重跑、2 次 Drive 改名、刪 1 個舊 zip。D 型同 B，但只能用 G4。E 型的時間要先實測。
- 證據：integration_facts.json 的 weekdays：10/09 五、10/12 一、10/16 五、10/18 日、10/21 三、10/27 二、10/30 五、11/03 二、11/04 三、11/08 日、11/10 二、11/16 一、11/18 三、11/19 四、11/21 六、11/23 一、11/24 二。gaps_days：10/27→11/03 是 7 天；11/19→11/23 是 4 天。04 的測試日擴充實際 32.13 小時；03 等使用者交回 Colab 結果 43.96 小時。玉山最擠的一段 10/19–10/27、11/04–11/27：D:/AI-Portfolio/competitions/TIMELINE.md。10/1 彩排在 G4 上：test02 競賽二 110 秒、競賽一 784.1 秒、test04 179 秒（results/test04/design/build_review.md 第 9.2–9.4 節）。03 兩臂約 43／46 分鐘（CLAUDE.md）。官方測試盤數：競賽二 10,000 盤；競賽一每池 40,000 候選加 5,000 題目，共兩池（docs/DATA.md，integration_facts.json 的 official_test_design）。11/03 不能延：TESTDAY「11/04 以前要就緒」要求完整測試全綠、git 乾淨。每次 Colab 要使用者 30–60 分鐘注意力，是任務說明給的數字；每次上傳約 10 分鐘是推估。

### 5. 實務風險

- 判定：(a) GPU：到目前為止每次都拿到 G4，11 月不保證。同型 GPU 容許差 1e-4、不同型 1e-3；登記的重跑只用 G4；競賽一只用 G4，11/20 才改 A100，會卡住 D 型 06。06 的 notebook 不要和 test02_player 同時跑。AI Pro 帳號沒有背景執行。(b) Drive：06 的結果放 C10/results/test06/<子資料夾>，子資料夾名稱不要用 rank、player、sw，讓 test02、test04 的解壓工具一定拒收；權重放 C10/results/prereg06，並備份到本機 data/prereg06_weights；只能有一份 C10_test_v1.zip，06 要核對它的 MANIFEST；每次搬檔後先截圖確認沒有 (1)；第二次彩排要改名、要新的 zip。(c) 帳號：權重只在 AI Pro 帳號；Claude 的 workflow 不能跨對話續跑，每一段都要 commit。(d) 防誤改測試：新檔放錯位置（檔名含 test02、放進 results/test02/design 或 results/prereg04、新增 tools/p02_*）；打分後固定 1 項失敗；official_log 多印一行 TagVersusNew 就壞；03 的測試曾在真實結果 commit 後壞掉；import 凍結資料夾裡的程式會寫出 __pycache__；測試日當天 main 必須乾淨（TESTDAY_04 第 1 步）。(e) 04 路徑沒有指紋。(f) CPU：C10_SLOW=2 要 27–72 分鐘、L2 冒煙 50–100 分鐘，要挑別條線沒在吃 CPU 的時段。(g) 11 月使用者很忙。(h) 驗證集量不到的條件式 06（像 05 的 D2）要在 PREREG_06 和 PREREG_04 rev 4 明文開例外，風險比較高。
- 證據：CLAUDE.md（prereg01–03 都在 G4、10/1 彩排三份都在 G4；AI Pro 不是 Pro+）；build_review.md 第 9.1 節（Drive 放錯過）、第 9.5 節（第二次彩排要新樹）；PREREG_02 第 9 節第 2 條、PREREG_02 rev 7 第 2 條；PREREG_04 第 11 節第 4、5、9、15 條，第 10.5 節（03 的測試壞掉），第 11.9 節 ab（時間）；COLAB_RUN test04「開始前」第 1 步；TESTDAY_04 第 1 步；integration_facts.json 的 name_checks、p04_test_day_path_files；tests/test_t04_official_log.py 第 410–431 行；competitions/TIMELINE.md「最擠的三段」；results/prereg05/explore/FINDINGS.md（D2 原本計畫的 PREREG_04 rev 4）。

## 機會

### A：競賽二只用 CPU 的 06（Q02＋新的 CPU 來源，接法照 O4）（comp2，信心 medium）

- 預期增益：看競賽二提分子題的結果，這份不估。要在 H6R 被採用，真實增益大約要 +0.006–+0.012（推估，標準誤假設 0.003–0.006）
- 依據：04 的前例：只用 CPU，rev 1 到 H4R 4.8 小時，測試日擴充 32.13 小時（git 時間）；results/prereg05/explore/u3_timeline_numbers.json 的採用機率
- 工作量：約 4 天；Colab 0 次
- 使用者要做：10 月不用做事；測試日在 04 之後多 1 次上傳（訊息 A 說 OK、訊息 B 上傳），最壞在 11/23 前多 1 次重傳
- 主要風險：增益太小就花掉打分機會換不到採用；06 依賴 04 的測試日條件（Δκ、重播失敗率）；official_log 要插入新行，而且不能讓 TagVersusNew 壞掉；打分後閘門固定 1 項失敗

### B：競賽二用 Colab 新訓練 GPU 模型的 06（像 03），測試日多一份推論 notebook（comp2，信心 low）

- 預期增益：未知。03 的前例只有 +0.0014 到 +0.0036（inner），沒過 +0.010 的閘門
- 依據：results/prereg03/RESULTS.md；03 從實作到交回 Colab 結果 43.96 小時；test04 notebook 在 G4 上 179 秒
- 工作量：約 8 天；Colab 3 次
- 使用者要做：10/16–10/18 跑 1–2 次 Colab 訓練（每次約 45 分鐘，可以同時跑）；選做彩排 1 次（10/31–11/02）；測試日多 1 次 Colab、放 notebook、下載 zip、1 次上傳；最壞再多 1 次重跑（2 次 Drive 改名、刪 1 個舊 zip）
- 主要風險：G4 拿不到、Drive 放錯、權重只在 AI Pro 帳號；10/19–10/27 和 11 月使用者很忙；03 的經驗顯示新模型對整題分數幫助小；新 notebook 要建、冒煙、審查，不能手改

### C：競賽一只用 CPU 的 06（讀 test02 已 commit 的每池 embedding 和候選檔）（comp1，信心 medium）

- 預期增益：看競賽一提分子題的結果。上限是 1 − 0.8830 = 0.117
- 依據：results/score_ledger.csv 的 Q04 0.8829546403931485；tools/t02_player.py 讀 results/test02/player/<pool>/dumps（cand_games.npz、query_games.npz）
- 工作量：約 4 天；Colab 0 次
- 使用者要做：10 月不用做事；11/05–11/06 在 test02 競賽一之後多 1 次上傳（加 1 次說 OK）
- 主要風險：官方怎麼切題目和候選，本機量不到（T3b／防護層）；official_log 對競賽一沒有最後一版規則，要插入新行，06 的報告不能叫 submit_report.json；PREREG_06 要在競賽一取代 U5

### D：競賽一用 Colab 新訓練 embedding 模型的 06（comp1，信心 low）

- 預期增益：未知
- 依據：PREREG_02 第 5 節把它列為 03 的候選，但沒做；目前沒有可以比的數字
- 工作量：約 8 天；Colab 2 次
- 使用者要做：10 月 1 次 Colab 訓練；測試日 1 次 Colab（只能用 G4，拿不到就等）和 1 次上傳；最壞再多 1 次重跑
- 主要風險：競賽一只能用 G4，11/20 才改 A100，K6' 會很趕；競賽一的 test02 本身就可能延到 11/20

### E：外部引擎的特徵（例如 KataGo／HumanSL），兩場都可能用（both，信心 low）

- 預期增益：未知，要等規則和 KataGo 子題
- 依據：PREREG_02 第 3 節、PREREG_04 第 7 節是自己訂的禁止，75bcb35 改成要重新評估；量很大：競賽二 inner 73,556 盤＋驗證 19,156 盤＋測試 10,000 盤，競賽一每池 40,000＋5,000 盤共兩池，速度沒有實測
- 工作量：約 10 天；Colab 4 次
- 使用者要做：10 月至少 1 次很長的 Colab（算 inner 和驗證的特徵）；測試日每場 1 次 Colab 和 1 次上傳；權重和 notebook 放在 AI Pro 帳號的 Drive
- 主要風險：規則要揭露外部資源和貢獻；Colab 時間不知道；權重檔大（本機不下載超過 50 MB 的檔，要在 Colab 裡下載）；10 月時間不夠

## 未解問題

- REPORT.md 被工具擋下（子代理不能寫報告檔），報告內容在本結果的各欄。要不要由主 session 照原文存到 results/prereg06/explore/integration/REPORT.md？
- 要不要另寫 PREREG_04 rev 4（指向 06），還是只靠 PREREG_06 取代（照 04 取代 U5 的前例）？rev 4 的代價是之後修 04 的 bug 要用 rev 5，TESTDAY_04 第 21 行會過時。
- K6'（06 在官方測試集上要成功並 commit 的期限）定 11/19，還是和 K6 一樣 11/16？11/19 讓 04 延誤時 06 還有時間；11/16 比較單純，但 06 比較可能來不及。
- 競賽二的第 4 次打分機會：PREREG_06 要寫死「保留給 split v2 或查 bug 重跑」，還是可以給 07？CLAUDE.md 現在寫「剩下的留給 split v2 或查 bug」，第幾節也寫錯（是 PREREG_02 第 3 節）。
- 06 如果只需要讀 test02 彩排的輸出，能不能沿用 10/1 的彩排樹和 Drive 上 old_rehearsal 裡的 zip（9fc4de29…），不做完整的第二次彩排？要在設計時實測。
- 11/01 使用者決定 11 月投入多少時，要不要接受 B、D、E 型多出來的 Colab 和可能的重跑（會和玉山每晚上傳、C1 衝刺撞在一起）？
- H6R 打分後，CLAUDE.md、AGENTS.md 裡「閘門要全過、沒有被略過」這句要改成什麼？建議：只准 test_ledger_and_prereg_rows_pinned 失敗，多出來的正好是 06 的 2 列，另由 tests/test_p06_hygiene.py 釘住。

## 寫出的檔案

- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\integration\integration_facts.py`
- `D:\AI-Portfolio\competitions\aicup2026-go\results\prereg06\explore\integration\integration_facts.json`
