# PREREG_02 rev 11：官方測試檔「開頭欄位」的正規化（HD／CA／RE 等）

這份是 `docs/PREREG_02_OPENING_FUSION.md`（PREREG_02）的第 11 版修訂。PREREG_02 被 t07／t08／t09／g11 的閘門整份凍結，所以和 rev 9（`docs/PREREG_02_REV9_TRAILING_PASS.md`）一樣寫成獨立檔；不改任何門檻、不改任何凍結檔。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 11 | （本檔第一次 commit） | 2026-11-04 13:00（測試集 11:00 已開放；本檔 commit 時 repo 裡還**沒有** `data/test/`、`results/test02/checks/`，也就是還沒 ingest 任何官方測試檔）；先 commit 本檔，之後才寫程式（`tools/root_props.py`、`tests/test_root_props*.py`）。C10 主 session 11/04 決定 |

## 1. 為什麼

- 官方 11/02 發布訓練資料 v1.1（GitHub `AILAB-NDHU/aicup2026-go-dataset` release `v1.1`，`aicup2026_go_training_v1.1.tar.xz`，SHA-256 `e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e`）：release notes 寫「每盤棋譜的 `sgf_content` 補上讓子資訊 `HD[n]`（例：`(;SZ[19]HD[2];B[pd];…`）」，其他（列順序、代號、等級、執色、棋步）和 v1.0 相同。官方 `sample_train_A.csv`（11/04 讀）每盤開頭是 `(;SZ[19]HD[0];B[pd];W[pq];…`（20 盤：HD[0] 18 盤、HD[1] 2 盤，沒有 CA／RE）。官方 Q&A（10/22）說開頭保留 CA／HD／RE。
- 本機訓練資料是 v1.0（開頭只有 `SZ[19]`），所有凍結的模型、特徵和接縫都建在 v1.0 的格式上；`data/training` 和 v1.0 壓縮檔不換。
- 11/04 靜態稽核：KataGo 那條鏈讀棋譜的 `results/prereg07/explore/pilot_humansl/hs_lib.py` 的 `parse_moves` 要求整盤是 `(;SZ[19];B[..];W[..]...)`（`FAST_SGF.fullmatch`），開頭多一個欄位就丟 `HSError`。測試檔每盤都帶 HD 時，07（HS）、08（P1）、09、19、競賽一 B′（P1）的每一盤都會讀取失敗，照各自登記的失敗率規則「不採用」、悄悄退回保底。其他路徑（`p02_moves.MOVE_RE`、sgfmill、T2 鍵、14／21 的鍵、rev 9）忽略開頭欄位。
- 所以在 ingest 之後、build 之前，把訓練 v1.0 沒有的開頭欄位拿掉，讓下游每一個工具看到的格式和訓練時完全相同。HD 的資訊（讓子數）因此不給模型用；模型本來就沒看過它（v1.0 的讓子棋同樣沒有標記）。

## 2. 定義

- **棋**：和 T1.4 同一組（`t02_checks.file_games`：題目檔 `num_games` 以內的 `sgf_i`、候選檔每列的 `sgf_content`）。
- **根節點**：棋的文字開頭（可有空白）`(`、`;` 之後，到下一個 `;`、`(` 或 `)` 之前的屬性，用凍結的 `t02_checks.TOKEN_RE`／`VAL_RE`（T1.4 的詞元規則）切。
- **保留的屬性**：`SZ`、`B`、`W`，和 T1.4 的擺子／讓子屬性 `AB`、`AW`、`AE`、`HA`、`PL`（它們是 T1.4 違規，照原規則算，不拿掉）。
- **拿掉的屬性**：根節點裡其他所有屬性（例如 `HD`、`CA`、`RE`、`KM`、`RU`、`GM`、`FF`、`PB`、`PW`、`DT`、`C`）。根節點以外的節點一律不動（落子節點帶的屬性照原樣，下游照原樣處理）。
- **拿掉的方法（逐位元組最小修改）**：每個拿掉的屬性刪「識別字第一個字元」到「最後一個值的 `]` 和緊接的空白」這一段；其他位元組（BOM、行尾、引號、欄位順序、其他格、其他棋、所有落子）一律不動。例：`(;SZ[19]HD[2]CA[UTF-8]RE[W+R];B[pd]…` → `(;SZ[19];B[pd]…`；`(;CA[UTF-8]SZ[19]HD[0];B[pd]…` → `(;SZ[19];B[pd]…`。
- **每盤的自我檢查**（不過就不改那盤，記為 `not_edited`）：凍結的 `sgf_check` 解析沒有錯誤；格子的原始文字和值相同（沒有跳脫引號）；改後的 `sgf_props` 序列＝改前去掉被拿掉的屬性；改後 `sgf_check` 的落子序列、T1.4 原因、`setup`、`sz`、`pass` 和改前完全相同。
- 檔案不是乾淨的 CSV（和 rev 9 同一套掃描：`csv` 讀不出和逐字掃描相同的格子、寬度不對、空檔）時整個檔不改，記 `scan_problem`。

## 3. 什麼時候套用（每一場各自判斷）

1. 兩場的完整 T1（real mode、`stage: raw`、`complete: true`）都已 commit，記的 SHA-256＝目前的 `data/test/raw`；還沒 build（`data/test/MANIFEST.json`、`results/test02/checks/built/` 不存在）。
2. **和 rev 9 的順序**：任一場 T1 是 FAIL_OPENING 時，先照 `docs/TESTDAY_02_PASS.md` 跑完 rev 9（`trailing_pass.py prepare`；套用時一路做到 `verify` PASSED 並 commit），再跑本工具。工具會擋：有 FAIL_OPENING 但沒有 rev 9 的 `prepare.json`、或 `data/test/raw` 是 rev 9 副本但沒有 PASSED 的 `verify.json`。rev 9 只看棋的結尾，開頭欄位不影響它的判斷；本工具只看開頭，結尾虛手不影響本工具。所以兩個都要時，先 rev 9 再 rev 11，結果和反過來做相同（演練檢查），重跑本工具不會再改（冪等）。
3. 那一場有至少一盤要拿掉開頭欄位 → 候選；沒有 → `not_needed`（什麼都不改）。T1 是 FAIL_STRUCT → `not_applicable`（停下回報總部）。
4. 候選場的副本用**沒改過的** `t02_checks.t1_check` 預算：verdict 要等於目前的 verdict，而且 T1.4（每檔的盤數、違規盤數）和 T1.5（每檔的盤數、S99 盤數）逐檔相同；不同 → `not_applicable`（停下回報總部，不改檔）。
- 套用時那一場兩池所有檔都處理；另一場的檔逐位元組照抄。
- 不改 T1 的 verdict 和任何門檻：FAIL_OPENING 仍是 FAIL_OPENING（照原規則退），PASS／NORMALIZED 不變。

## 4. 怎麼做（操作頁 `docs/TESTDAY_02_ROOTPROPS.md`）

（操作頁已改名為 `docs/TESTDAY_22_ROOTPROPS.md`，見第 7 節。）

1. （rev 9 的部分做完、commit 之後）`PY tools/root_props.py prepare`：原檔（目前 `data/test/raw` 的 6 個檔和 `SOURCE.json`）逐位元組複製到 `data/test_rootprops/original/`（唯讀）；副本放 `data/test_rootprops/normalised/`（6 個官方檔名）；紀錄 `results/test02/checks/root_props/prepare.json`（每檔前後 SHA-256、盤數、改了幾盤、拿掉的屬性和次數、HD 值的分布、每盤改了哪裡、沒改的盤和理由、改前改後符合 `(;SZ[19];B..;W..)` 純格式的盤數）和改前兩份 `t1.json` 的備份 `pre_t1_<comp>.json`。結束碼 0 套用、2 不需要、1 拒絕。
2. 套用時：commit 紀錄 → 沒改過的 `t02_test_bundle.py ingest --src data/test_rootprops/normalised --rerun --reason "PREREG_02 rev 11 ..."` → 兩場 `t02_checks.py t1 --rerun --reason "PREREG_02 rev 11 ..."` → commit → `PY tools/root_props.py verify`（`data/test/raw`＝副本、`SOURCE.json` 只指向副本、原檔沒變、從原檔用同一條規則重新推出副本、兩份 `t1.json` 記的 SHA＝副本、verdict＝預算＝改前、`data/test/raw` 再掃一次沒有要拿掉的開頭欄位）→ commit。之後照 `docs/TESTDAY.md` 第 2 步起原樣走。
3. `verify` PASSED 才能 build（下游工具不知道 rev 11，順序只靠操作頁）。已經 build 了：停下回報總部。

## 5. 驗收（合併前；結果寫在本檔後面加的列或節）

- 單元測試：各種開頭（HD 單獨、HD＋CA＋RE、欄位在 SZ 前、空白、帶擺子屬性、根節點以外的屬性、跳脫引號、BOM／CRLF）、冪等、只改指定位元組。
- 演練（scratch，只用訓練玩家，`t02_fake_tests.py from-train --shape official`）：A＝沒有開頭欄位；H＝每盤加 `HD[n]`（0／1／2–9 的比例照 Q&A），部分再加 CA／RE；HP＝H 再加約 30% 結尾虛手。H、HP 正規化（HP 先 rev 9）後的 6 個檔和 A（HP 對照 A 加虛手後的 rev 9 副本＝A）**逐位元組相同**，build 出的檔也相同；H 沒正規化時直接比較下游解析函式（MOVE_RE、開局詞元、風格、sgfmill、T2 鍵、KataGo 的 `hs_lib.parse_moves`）的輸出。
- 獨立審查、防誤改閘門（只有 4 項允許的失敗）。

## 6. 揭露

- 這條是在打開任何官方測試檔之前、根據官方 v1.1 release notes、sample 檔和 Q&A 定的；不改任何門檻。拿掉了幾盤、哪些屬性、HD 分布，寫進 RESULTS 和報告。HD（讓子數）是官方提供的資訊，我們選擇不用，理由是模型都用 v1.0（沒有 HD）訓練。

## 7. rev 12：實作、改名與演練（2026-11-04，追加）

- **改名**：本檔第一次 commit 是 `41830d0`（檔名 `docs/PREREG_02_REV11_ROOT_PROPS.md`，早於程式 `c27a47a`）；因為 `tests/test_t19_hygiene.py` 把 `docs/PREREG_0*`、`docs/TESTDAY_0*` 整批凍結，新檔放在那裡會讓閘門出現不允許的失敗，所以改名為 `docs/PREREG_22_ROOT_PROPS.md`（操作頁 `docs/TESTDAY_22_ROOTPROPS.md`），內容照舊只加不改。編號 22 只是檔名，規則仍是「PREREG_02 rev 11」，工具和紀錄裡的理由字串都寫 rev 11。
- **結束碼**（補第 4 節第 1 條）：0 套用、2 不需要（什麼都沒改）、3 至少一場不適用（印 `STOP`：不要 ingest，停下回報總部）、1 拒絕。
- **補充規則**（審查後加，都是更保守的方向）：根節點的識別字不是全大寫（例如 `Sz`）時那盤不改；每個刪除範圍必須落在根節點內；rev 9 的紀錄必須是目前這批檔的（未套用時 `prepare.json` 記的原檔 SHA＝目前的檔；套用時目前的檔＝它的副本、`verify.json` PASSED 而且對應這份 `prepare.json`），不然拒絕；`verify` 另外檢查：沒有 build、工具／`t02_checks.py`／`trailing_pass.py` 和 prepare 時相同、`pre_t1_<comp>.json` 未變且改後 T1 和改前逐項相同、套用那場的 `t1.json` 是以含「rev 11」的理由重跑的。
- **11/04 13:15 主 session 對真實測試檔的觀察**（本 session 沒有讀測試檔）：6 個檔每盤開頭都是 `(;SZ[19]HD[n];`，n 只有 0 或 1，沒有 CA／RE 或其他欄位。本工具在這種輸入上等於「只拿掉根節點的 `HD[n]`」。
- **演練結果**（scratch `W:\reh`，fit／inner 訓練玩家的假樹 `t02_fake_tests.py from-train --shape official`，6 檔共 100,000 盤；紀錄 `results/root_props/rehearsal_summary.json`、`rehearsal_h01.json`，腳本 `results/root_props/rehearse.py`、`rehearse_h01.py`）：
  - H01（真實測試檔的格式：每盤 `(;SZ[19]HD[n];`，n＝0／1，約 20% 是 1）：原始 T1 兩場 PASS，和沒有 HD 的 A 除了 `other_properties_record_only` 以外完全相同；KataGo 的 `hs_lib.parse_moves` 100,000 盤**全部失敗**（A 全部成功）；`prepare` 兩場 apply，每盤只拿掉 `HD`，副本和 A **逐位元組相同**；ingest／t1 重跑 PASS、`verify` PASSED、build 出的 6 個檔和 A 逐位元組相同。
  - H（HD 0–9、部分 CA／RE、部分 CA 在 SZ 前）：同上全部相同，T4／T5 輸出和 A 相同；沒正規化時直接比較 p02_moves、開局詞元、開局字串、p04 編碼、p06 風格、T2 鍵、T1.4 事實、sgfmill 兩種特徵（每 50 盤一盤）：差異 0；KataGo 解析 0／100,000。
  - HP（H 加約 30% 結尾虛手）：兩場 FAIL_OPENING → 本工具拒絕（先 rev 9）→ rev 9 套用，副本＝H → rev 9 verify 前本工具拒絕 → verify 後本工具套用，副本＝A，build＝A；反過來做（先 rev 11 再 rev 9）也得到 A；重跑本工具不改任何東西。
- **獨立審查**（另一個代理，只讀）：沒有阻擋項目；重要 4 項（結束碼 3、rev 9 紀錄過期、verify 擋已 build、verify 讀 `pre_t1`）和多數小項已修，見上面的補充規則。沒改：`(; SZ[19] HD[3] ;` 這種帶空白的根節點改後仍不是純格式（由 `plain form` 計數和 WARNING 擋，真實檔沒有這種寫法）。
