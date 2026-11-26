# PREREG_02 rev 9：官方測試檔「結尾虛手」的正規化（T1.4）

這份是 `docs/PREREG_02_OPENING_FUSION.md`（以下稱 PREREG_02）的第 9 版修訂。PREREG_02 本身被 `tests/test_t07_hygiene.py`、`test_t08_hygiene.py`、`test_t09_hygiene.py`、`test_g11_hygiene.py` 整份凍結（和 tag 比對，連加行都不行），在它後面加 `| rev 9 |` 列會讓防誤改閘門出現不允許的失敗，所以 rev 9 寫成這份獨立檔。PREREG_02 第 9 節第 6 條、第 10 節第 5–6 條、第 12 節照舊有效；本檔只加一條例外，不改任何門檻。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 9 | （本檔第一次 commit） | 2026-10-17，官方測試集 11/04 11:00 才開放，**沒有任何測試資料**；先 commit 本檔，之後才寫程式（`tools/trailing_pass.py`、`tests/test_trailing_pass*.py`）。主 session 10/17 決定、使用者同意 |
| rev 10 | （本列的 commit） | 2026-10-18，**仍然沒有任何測試資料**。實作（`544966e` 之後的 `eabbaa5`、`cba9705`）和登記的差異、獨立審查的修正、scratch 演練結果（第 8 節）。不改第 2–3 節的規則；第 8 節第 1、3、4 條是登記沒寫清楚、影響測試日操作的地方 |

## 1. 為什麼

- T1.4（PREREG_02 第 9 節第 6 條、rev 6 第 6 條）把一盤裡**任何位置**的虛手（`B[]`、`B[tt]`、`W[]`、`W[tt]`）都算違規；一個檔超過 1% 的棋違規就是 FAIL_OPENING，競賽二退 R10、競賽一的 06／10 不用（退 Q04）。
- 訓練集完全沒有虛手（`docs/DATA.md`「格式」：主辦方清過）。官方測試檔如果保留了「數目結束」的棋最後那一兩手虛手，會被誤判 FAIL_OPENING，兩場同時退到備案。
- T1.4 要抓的是「開局統計失準」（擺子、讓子、不是 19 路、不是黑先、不交替）。最後一個落子之後的虛手不影響任何特徵：`tools/p02_moves.py` 的 `MOVE_RE` 本來就跳過虛手，開局特徵只看前 10–12 手。拿掉它們，測試棋就和訓練棋的格式一樣。

## 2. 定義

- **落子**：`B`／`W` 的值是兩個 a–s 字母（`t02_checks.py` 的 `POINT_RE`）。
- **虛手**：`B`／`W` 的值是空字串或 `tt`（和 T1.4 的 `sgf_check` 同一個定義）。
- **結尾虛手盤**：一盤棋（題目檔 `num_games` 以內的 `sgf_i`、候選檔的 `sgf_content`，即 `t02_checks.file_games` 列出的棋）同時符合：
  1. 原樣用凍結的 `t02_checks.sgf_check` 檢查：解析沒有錯誤，T1.4 的原因**只有** `pass` 一項（沒有擺子／讓子屬性、SZ 是 19 或沒寫、至少一手、黑先、嚴格交替）。
  2. 文字是「最後一個落子節點（`;B[xx]` 或 `;W[xx]`）」之後緊接 **1 個以上**虛手節點，再接 `)` 結束。每個虛手節點只能是 `;`＋`B`或`W`＋`[]`或`[tt]`（允許和 T1.4 相同位置的空白），不能帶其他屬性（例如 `;B[]C[...]` 不算）。單獨一個結尾虛手也算。
  3. 拿掉這些虛手節點之後，再用 `sgf_check` 檢查：沒有任何 T1.4 原因，而且落子序列和原本完全相同。
- **拿掉的方法（逐位元組最小修改）**：只刪掉「第一個結尾虛手節點的 `;`」到「最後一個虛手節點的 `]`」這一段位元組。檔案的其他位元組（BOM、行尾、引號、欄位順序、其他格、其他棋）一律不動。格子帶跳脫的引號（`""`）、檔案不是乾淨的 CSV（`csv` 讀不出和逐字掃描相同的格子、有寬度不對的列）時不改，那盤照原規則。

## 3. 什麼時候套用（每一場各自判斷）

競賽 c（rank 或 player）同時符合下面四條才套用：

1. 第 1 步（`docs/TESTDAY.md`）對 AIdea 原始檔跑的完整 T1（real mode、`stage: raw`、`complete: true`），已 commit，verdict 是 **FAIL_OPENING**。
2. 那份 T1 裡 T1.1–T1.3、T1.5 都沒有 FAIL；T1.4 至少一個檔 FAIL。
3. c 的檔案裡至少有一盤結尾虛手盤。
4. 把 c 的每個檔的**所有**結尾虛手盤都拿掉虛手後，用**沒改過的** `t02_checks.t1_check`（T1.1–T1.5 全部）檢查副本，結果是 **PASS 或 NORMALIZED**。

- 套用時 c 的兩池（public、private）所有檔都處理，不只 FAIL 的那個檔；另一場的檔逐位元組照抄（除非它自己也符合這四條）。
- **不在例外裡的，照原規則**：
  - 中盤虛手（虛手後面還有落子）、和其他 T1.4 原因同時出現的虛手、帶其他屬性的虛手節點：那盤**不改**，仍然算違規。
  - 其他 T1.4 原因（擺子、讓子、非 19 路、白先、不交替、沒有落子）、T1.5 不過、FAIL_STRUCT。
  - 拿掉結尾虛手後仍然 FAIL_OPENING（例如中盤虛手超過 1%）：副本不用，照 PREREG_02 第 9 節第 6 條走 FAIL_OPENING（競賽二 R10、競賽一照常每池程序、06／10 不用）。
  - 原始 T1 是 PASS 或 NORMALIZED（違規 ≤ 1%）：什麼都不改，即使有少量結尾虛手。
- 1% 的容忍度不變：副本裡剩下的違規（例如少量中盤虛手）≤ 1% 時副本 PASS，照常套用。

## 4. 怎麼做（11/04 的操作；細節 `docs/TESTDAY_02_PASS.md`）

1. 照 `docs/TESTDAY.md` 第 1 步 ingest、兩場都跑完整 `t1`，**commit** 兩份 `t1.json`（FAIL_OPENING 的那份也照實 commit）。
2. 任一場 FAIL_OPENING 時，**在第 2 步 `build` 之前**跑 `PY tools/trailing_pass.py prepare`：
   - 原始檔：`data/test/raw` 的 6 個檔和 `SOURCE.json` 逐位元組複製到 `data/test_pass/original/`，設成唯讀，之後不動。
   - 副本：`data/test_pass/stripped/` 只放 6 個官方檔名（套用的那場是拿掉結尾虛手的副本，其他逐位元組照抄）。
   - 紀錄：`results/test02/checks/trailing_pass/prepare.json`（每個檔前後的 SHA-256 和位元組數、盤數、拿掉虛手的盤數、拿掉的虛手節點數、中盤虛手盤數、虛手加其他原因的盤數、每一盤改了哪裡；兩場的判定 apply／not_needed／not_applicable 和理由；副本用 `t1_check` 預先算出的 T1 verdict），以及原始 `t1.json` 的備份 `raw_t1_<comp>.json`。
3. 判定 **apply**：commit 紀錄 → 用沒改過的 `t02_test_bundle.py ingest --src data/test_pass/stripped --rerun --reason "PREREG_02 rev 9 ..."` 把副本放進 `data/test/raw` → **兩場都**重跑 `t02_checks.py t1 --rerun --reason "PREREG_02 rev 9 ..."` → `PY tools/trailing_pass.py verify`（核對 `data/test/raw`＝副本、`SOURCE.json` 指向副本、原始檔沒變、兩份 `t1.json` 記的 SHA＝副本、套用那場的 verdict＝預算的 PASS／NORMALIZED、另一場 verdict 不變）→ commit。之後照 `docs/TESTDAY.md` 第 2 步起原樣走（`build` 不加 `--allow-fail-opening`）。
4. 判定 **not_applicable**：照原規則走 FAIL_OPENING（`build --allow-fail-opening --reason ...`、R10、06／10 不用），`prepare.json` 留著當紀錄。**not_needed**（沒有一場 FAIL_OPENING）：不用跑本工具。
5. **順序**：`prepare` 只能在第 2 步 `build` 之前（`data/test/MANIFEST.json`、`results/test02/checks/built/` 都還不存在）。如果已經用 FAIL_OPENING 的結果 build 或往下做了，**停下回報總部**，不自己倒回去重做。

套用之後，副本就是之後每一步的「原始檔」：`build`、Colab 測試包、T2–T5、04／06／07／08／09／10／14、`official_log.py` 都照原本的方式讀 `data/test/raw`（經由 `t1.json` 記的 SHA 綁定）和 `data/test`。AIdea 的原始檔留在 `data/test_pass/original/`（唯讀），對應關係在 `prepare.json`、`verify.json`。

## 5. 為什麼不用改任何凍結檔（10/17 讀程式確認）

- `t02_test_bundle.py build` 只核對 `t1.json` 記的檔案 SHA-256＝`data/test/raw` 的檔；`t02_checks.py run` 核對 MANIFEST 的 `source_sha256`＝`t1.json` 記的 SHA；`t02_rank.py`、`t02_player.py`、`t06_player.py`、`t10_player.py`、`t14_copy.py`、`official_log.py prepare` 要求 `t1.json` 完整、`stage: raw`、real mode、verdict PASS／NORMALIZED（有的另外記 `t1.json` 的 SHA）。沒有任何工具讀 `SOURCE.json` 的內容、核對 AIdea 原始檔的 SHA 或 `t1.json` 的 `rerun_reason`。
- `ingest` 只拒絕本工具組自己的紀錄檔（MANIFEST、SOURCE、FAKE_TREE）；`data/test_pass/stripped/` 只放 6 個 CSV。
- 新工具和測試的檔名避開凍結的 glob（`tools/t0*_*.py`、`tools/p0*`、`tests/test_t0[2-9]_*.py`、`tests/test_p0[1-9]*.py`）；測試日的紀錄寫在 `results/test02/checks/trailing_pass/`（`R20PathNoNewFiles` 允許 `results/test02/` 底下的測試日輸出，`design/`、`null/` 除外）。

## 6. 合併前的驗收（結果記在 rev 10）

- 單元測試：每種虛手寫法、中盤虛手、虛手加其他原因、帶屬性的虛手節點、引號格、BOM／CRLF 保留、只改指定位元組。
- 演練（scratch，只用訓練玩家，`t02_fake_tests.py from-train --shape official`）：在 rank 和 player 的檔各取約 30% 的棋加上 1 或 2 個結尾虛手 → 原始 T1 FAIL_OPENING → `prepare`（apply）→ `ingest --rerun` → `t1 --rerun` 為 PASS → `verify` 通過；副本和沒加虛手的檔**逐位元組相同**，`build` 產生的 6 個測試檔也逐位元組相同。對照組：超過 1% 的棋有中盤虛手時，判定 not_applicable、T1 仍是 FAIL_OPENING。
- 獨立審查、防誤改閘門（只有 4 項允許的失敗）。

## 7. 揭露

- 這條例外是 10/17 看過訓練資料（0 個虛手）、**沒有任何測試資料**時定的，不改任何門檻；只決定「最後一個落子之後的虛手」不算開局失準。
- 測試日實際拿掉了幾盤、幾個虛手，寫進 RESULTS 和報告。

## 8. rev 10：實作與演練（2026-10-18，沒有任何測試資料）

**和登記的差異、補充（影響測試日操作的是第 1、3、4 條）**

1. **commit 的順序**：real mode 的 `verify` 要求 `prepare` 的紀錄和**重跑後的兩份 `t1.json` 都已 commit 且乾淨**。所以順序是：`prepare` → commit 紀錄 → `ingest --rerun` → 兩場 `t1 --rerun` → commit → `verify` → commit（`docs/TESTDAY_02_PASS.md` 照這個寫）。`prepare` 另外要求 `tools/trailing_pass.py`、`tools/t02_checks.py` 和本檔已 commit 且乾淨。
2. **不改的棋分得更細**（第 2、3 節「不在例外裡的」不變，只是紀錄分類）：`mid_game_pass`（最後一個落子之前有虛手）、`pass_not_trailing_other`（虛手都在最後一個落子之後，但不是緊接 `)` 的純虛手節點：帶其他屬性、分支、沒有 `)`）、`pass_with_other_reason`、`pass_parse_error`、`trailing_unusual_cell`（格子帶跳脫引號）。這些都不改，仍然算 T1.4 違規。結尾虛手節點的顏色不限（例如 `;W[dp];W[]` 也拿掉），第 2 節本來就沒限制，這裡寫明。
3. **副本的 T1 預算不含驗證棋譜防護**：`prepare` 用 `t1_check` 預算副本的 verdict，不讀 `data/local`，所以算不到 real mode `t1` 的驗證棋譜防護（一個檔一半以上是本機驗證棋譜就拒絕）。`t1 --rerun` 拒絕、或 verdict 和預算不同：**停下回報總部**，不再改檔。
4. **`verify` 通過才能 build**：下游工具不知道 rev 9（凍結檔不改），這個順序只靠操作頁；`verify` 也會從唯讀的原始檔用同一條規則重新推出副本，核對 SHA-256（不只相信 `prepare.json`）。
5. 原始 T1 是 FAIL_STRUCT 的那一場標 `not_applicable`（理由寫「停下、另寫修訂」，照 PREREG_02 第 9 節第 6 條），不是 `not_needed`。
6. `prepare --rerun` 不刪原始檔：`data/test_pass/original/` 和目前的 `data/test/raw` 完全相同才沿用，不同就拒絕（請人工移開並回報）。判定 not_applicable 時不留空的 `data/test_pass/`。
7. 紀錄另外寫每一盤拿掉的位置（格子內的字元位置、檔案內的位元組位置）；空檔或沒有表頭的檔記為 `scan_problem`、不改（審查發現原本會讓 `prepare` 整個中斷）。

**演練（scratch，`W:` 對應的 scratchpad；只用訓練玩家；紀錄 `results/trailing_pass/rehearsal_summary.json`，腳本 `results/trailing_pass/rehearse.py`）**

- 假檔：`t02_fake_tests.py from-train --shape official`（競賽一 fit 玩家全黑、競賽二 inner 玩家每題單色；6 個檔共 100,000 盤）。沒加虛手的這組當對照 A：兩場原始 T1 PASS，`prepare` 結束碼 2（not_needed），沒有寫任何資料夾。
- **B（6 個檔各約 30% 的棋加 1 或 2 個結尾虛手）**：加了虛手的 30,301 盤（共 45,454 個虛手節點）；原始 T1 兩場 FAIL_OPENING（各檔 T1.4 違規比例 0.300–0.308）→ `prepare` 兩場 apply、預算 PASS → 拿掉虛手的 6 個副本和 A 的檔**逐位元組相同** → 沒改過的 `ingest --rerun`、兩場 `t1 --rerun` 都是 PASS → `verify` PASSED → `build --kind rehearsal` 產生的 6 個測試檔和 A 的**逐位元組相同**、重驗 T1 PASS；`t02_checks.py run --only T4,T5` 的輸出（去掉 provenance）兩場都和 A 相同。
- **C（對照：競賽二另有中盤虛手）**：競賽二兩檔各有 156、163 盤中盤虛手（另有約 30% 結尾虛手），競賽一只有結尾虛手。拿掉結尾虛手後競賽二的 T1.4 比例仍是 0.0312、0.0326 → **not_applicable**，競賽二的檔沒改、T1 仍是 FAIL_OPENING（照原規則退 R10）；競賽一 apply → PASS，4 個檔和 A 逐位元組相同；`verify` PASSED。
- **D（對照：只有中盤虛手）**：競賽二 133、148 盤中盤虛手 → not_applicable（沒有結尾虛手盤），結束碼 2，沒有寫任何資料夾。
- 下游：測試日之後的步驟都只讀 `data/test/raw`（經 `t1.json` 綁定）和 `build` 產生的 `data/test`；B 的這些檔和沒有虛手的 A 逐位元組相同，所以 Colab、04／06／07／08／09／10／14 拿到的輸入完全相同。需要 GPU 結果的步驟沒有在這次演練跑；`C10_SLOW=1` 的 t06、t10 演練是用驗證衍生的樹、只跑 T1.1–T1.3，不經過 T1.4，只當回歸檢查。
- 單元測試 `tests/test_trailing_pass.py`（合成的官方形狀檔，不用訓練或驗證資料）、`tests/test_trailing_pass_hygiene.py`（加進 AGENTS.md 的閘門）。

**獨立審查**（另一個代理，只讀）：沒有阻擋項目。重要 3 項：① AGENTS.md 要指到操作頁（總部做）；② 演練和 rev 10 還沒做（本節）；③ 空檔會讓 `prepare` 中斷（第 7 條修好）。小問題 9 項：第 1、3、4、5、6、7 條和第 2 條（中盤虛手分開計數、結尾顏色寫明）、登記只准加行（閘門測試）都已處理；「副本預算不含驗證棋譜防護」「verify 前 build 沒有程式擋」照第 3、4 條寫明，不改凍結檔。
