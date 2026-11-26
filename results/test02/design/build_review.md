# test02 測試日工具組：整合與審查紀錄

- 規格：`results/test02/design/spec.md`（登記正本是 `docs/PREREG_02_OPENING_FUSION.md` 第 9 節 rev 5）。
- 第 1 節由整合者寫（2026-09-26，HEAD `5a5572e`，還沒有 commit）。第 2 節是三個唯讀代理的獨立審查，第 3 節是修正者的修正紀錄與重跑結果。

## 1. 整合紀錄

**結論**：spec 第 8 節整合步驟 1–6 全部做完、全部通過。

- 釘住的數字都重現：T3 虛無的 6 組數字、R20 自我檢查差 0、兩場的接縫逐位元組相同、5.2 的所有標準值。
- 兩份 notebook 在 CPU 冒煙測試都跑得完。
- 負面情境都在正確的地方停下。
- 整合時改了 1 個真問題：競賽二 notebook 在接縫沒過之後重跑，會沿用舊的逐盤輸出，所以登記的「重跑一次」其實沒有重算。現在會先停下，要求把結果資料夾改名。
- 凍結檔一個 byte 都沒動，也沒有 commit。

### 1.1 環境

- 本機 Windows 11，`.venv\Scripts\python.exe`（Python 3.12.13，torch 2.14.0+cpu），全程 CPU。
- 沒有開過任何答案檔，沒有跑 `tools/score.py`。
- 暫存都在 `…\scratchpad\testkit\INT\`，以下簡寫成 `INT`。
  - `…` = `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1`
- 凍結路徑的 `git status --porcelain` 是空的。凍結路徑包括：
  - `tools/p02_*.py`、`c10data.py`、`decide.py`、`score.py`、`make_val.py`
  - `colab/build_prereg0*`、`colab/prereg0*`
  - `results/prereg01`、`results/prereg02`、`results/score_ledger.csv`
  - `docs/VALIDATION.md`、`docs/PREREG_0*`、`data/local`
- 時間是牆鐘時間。步驟 3、5、6 有一部分同時在跑，互相搶 CPU，所以單獨跑會比較快。

### 1.2 步驟與結果

以下 `PY` = `.venv\Scripts\python.exe`，都在 repo 根目錄執行。

**步驟 1：完整測試**

- 指令：`PY -m pytest tests -q`
- 結果：359 passed、1 skipped，165 秒。
  - skip 的是 `test_t02_rank` 的真實資料重擬合，要設 `C10_SLOW`。
- 所有修改完成後的最後一輪：`C10_SLOW=2 PY -m pytest tests -v`
  - **378 passed、0 skipped**，557 秒。
  - 包含兩棵乾跑樹、R20／R10 的真實重擬合，以及新的 `tests/test_t02_dryrun.py`（18 項）。

**步驟 2：兩份 notebook 各重建兩次**

- 指令：`build_test02_rank_nb.py --out-nb/--out-diff` 和 `build_test02_player_nb.py --out-dir`，輸出到 `INT\nb\`。
- 競賽一：repo 裡的版本和兩次重建逐位元組相同。
  - notebook `0a789355…`、diff `b8ca6903…`，原始碼 SHA `ac2c19db…`。
- 競賽二（實作者交來的版本）：notebook `1cd814a7…`、diff `ca7070d5…`，三份相同。
- 競賽二改版後（見 1.3 F1），又重建兩次，逐位元組相同：
  - notebook `fbf87539fe37c4fb760554b1753f019666db69a2b6289200a30239e3c8a661e0`
  - diff `d377f3407c0240e6fb2bddd1e6b9e4362cb0f57bb3435d15533daa12982cbcff`
  - `notebook_sha256`（原始碼規則）`7f4138bc…`
  - `test_t02_nb_rank.py` 和 `test_t02_hygiene.py` 共 29 passed。

**步驟 3：T3 虛無分布**

- 指令：`PY tools/t02_checks.py precompute --dry-run --out INT\null_dry`
- 時間：12 分 15 秒（工具自己記 735 秒，當時和其他工作同時跑）。
- 結果：`pins OK`，和 spec 4.1 完全相同。

| | raw | canon |
|---|---|---|
| q99 | 0.1004267242871777 | 0.08085695648061325 |
| 詞彙數 | 146,421 | 144,097 |
| 虛無玩家 | 1,985 | 1,985 |
| inner 留出玩家 | 226 | 226 |
| 虛無超過 q99 | 19 | 19 |
| 留出超過 q99 | 1 | 3 |
| IDF SHA-256 | `2523d83c…` | `0214853f…` |

- 和 I3 寫在 repo 的 `results/test02/null/` 比：
  - 兩個 npz 逐位元組相同（raw `4eb9cf05…`、canon `5312a18c…`）。
  - `t3_null.json`、`train_refs.json` 除了 provenance 和計時欄位，全部相同。
- 所以 repo 裡的檔保留 I3 用 real mode 產生的版本，沒有覆寫。它記的是 `tool_committed_clean: false`、HEAD `5a5572e`。

**步驟 4：R10 備案參數**

- 先把 I1 的兩個檔備份到 `INT\fallback_I1\`。
- `PY tools/t02_rank.py fallback-fit --combo R20 --rerun --reason "…"`
  - 2.4 秒；最大差 0（容許 1e-12）。
  - refit loss 1.294020404166363，和 tune 記的相同。
- `PY tools/t02_rank.py fallback-fit --combo R10 --rerun --reason "…"`
  - 2.9 秒。
  - a = 1.2235994658708127，γ = 0.43313317341817686。
  - inner OOF 0.5686757203001245，擬合 25,400 題。
- 兩個檔和 I1 的版本只差 `created`、`seconds`、`rerun`，參數完全相同。
- 之後 `t02_rank.py` 沒再改過，所以這兩個檔仍然有效。

**步驟 5：`tests/test_t02_dryrun.py`（新檔）與乾跑**

每個工具都用命令列呼叫，和測試日相同。樹建在暫存目錄；設了 `C10_DRYRUN_DIR` 就建在那裡，而且必須在 repo 外。

`C10_SLOW=1`：`FromValPipeline`，13 項，約 2–3 分鐘。依序跑：

1. `from-val`
2. `t02_checks t1 --dry-run --only T1.1,T1.2,T1.3`：兩場都是 PASS。
3. 競賽二 `opening`、`apply`；另外用一份 verdict 改成 FAIL_OPENING 的 t1 副本走一次 `--fallback R10`。
4. 競賽一 `run` 三池、`submit`。
5. `t02_checks t6`。

結果都和 5.2 相同：

- **競賽二接縫**
  - `seam_val/rank_val_probs.csv` = `a8370043…`，`pred_B.csv` = `863a6258…`，都和 prereg02 逐位元組相同。
  - 最大差 0.0，規則 B 不同 0 題。
  - 開局分數的接縫部分（ll、o、has、n_terms）和 `opening_val.npz` 逐位元組相同。
- **競賽二測試檔**
  - `rank_test_probs.csv` 的 800 列：前綴換掉後，和 prereg02 驗證的第 1–800 列逐位元組相同。
  - `submission_rank.csv` 的標籤和 `pred_B.csv` 相同；`check --official` 沒有問題。
  - SHA-256 `d9b9801b3df85c75e5932ce228eebbeb8a76c43b40d2efd02700ce8807490d97`（只對這套假 id 成立）。
  - public、private 對調後，`--official` 會報錯。
- **R10 路徑**
  - `apply_test_report.json` 記 `fallback_used: true`、組合 R10、參數等於 `fallback_R10.json`。
  - 接縫仍是 R20，和 prereg02 逐位元組相同。
  - 交件檔 `--official` 通過；機率和 R20 版不同。
  - T1 是 PASS 時加 `--fallback R10` 會被拒絕（exit 1）。
- **競賽一接縫**
  - dumps 和 prereg02 逐陣列相同，24 個平均完全相同。
  - Q04 = 0.8997649863787647，閘門 +0.4991241719699286，第 1 層。
  - sr_med raw 0.22665843779803582、canon 0.22122531121305591（等於凍結值）。
  - `top5.csv` = `69ca74a6…`，400/400 相同。
- **競賽一 public**
  - Q04 0.9021469023127683，Q08 0.8982233054119719。
  - 閘門 +0.49217940296670204，第 1 層，ρ raw 1.0126024400509277。
  - 400/400 相同。
- **競賽一 private**
  - Q04 0.8982118617937633，Q08 0.8970547760408695。
  - 閘門 +0.4934357493963071，第 1 層，ρ raw 1.0039671134814359。
  - 400/400 相同。
- **競賽一交件**
  - 800 列，LF、沒有 BOM，`--official` 通過。
  - SHA-256 `b5d7473e1011c299f509a5ac269d60471068fc3f6cfbe1430ddafcd50ee9b8b4`（只對這套假 id 成立）。
- **T6**
  - 讀到 `apply_report.json` 的 `sr_top1_distinct_per_round`，交件中不同的第 1 名兩池都是 357。
  - 每輪 public 352–364、private 350–368，落在訓練模擬的 347–367 附近。
- **防護**
  - 在這棵驗證衍生的樹上跑內容檢查（`run --only T4`）會被拒絕，而且不寫任何檔。
  - `apply`、`submit` 沒加 `--rerun --reason` 時拒絕覆寫。

`C10_SLOW=2`：另外跑 `FromTrainChecks`，5 項，約 6 分鐘。依序跑：

1. `from-train`
2. 完整 T1
3. POS 兩池跑 `run --comp player`，同時寫出 T2 索引快取
4. `run --comp rank --only T2,T4,T5 --index-part fit`
5. NEG：跑 `t1`，再跑 `run --only T3,T4`

結果：

- **T1**：兩場都是 PASS（complete）。NEG 是 FAIL_STRUCT（設計如此）。
- **T2(a)**：POS 的四個檔都 FAIL。
  - 題目 4,997／4,959 盤全部是複製，門檻 10。
  - 候選 40,000／40,000 盤全部是複製，門檻 40。
  - (b)(c)(d) 都是 0。
- **競賽二題目對 fit 索引**：720,307 盤、5,767 人，複製 0，PASS。
- **T3**：POS 兩池 raw／canon 都是 400/400（需要 40），判定重疊。NEG raw 0、canon 2（需要 23），不判定。參考集重建和 committed null 完全相同。
- **T4**
  - 黑方比例：競賽一 0.5019–0.5113，競賽二 0.4899／0.4978。
  - 第一手 TVD 最大 0.0471，都在虛無最大值內。
  - 右上比例 0.78–0.84，沒有旋轉旗標。
- **T5**：平均 12.26／12.37，未滿 10 的比例 0.335／0.330，χ² 20.1／13.4。

**步驟 6：兩份 notebook 的 CPU 冒煙**

驅動樹 `INT\smoke\drive\results\prereg0{1,2}_smoke` 是從 `scratchpad\p02final\fix\{rank,player}\smoke\drive\results\` 複製來的，約 1 GB。

- 那條鏈在 9/26 08:06–08:22 跑。開跑前，`hash1.txt`、`hash2.txt` 記的兩份 prereg02 notebook 的 SHA 就是 commit 版 `9fb1f9d0…`、`738c1dd8…`。
- spec 2.4 寫的 `p02final\smoke\` 是 07:49 用較舊的 notebook 跑的（commit `aa1c981` 之前），而且競賽二的 config.json 沒記 notebook，所以沒有用它。

測試集 zip 照測試日的順序做：

1. `t02_fake_tests.py from-val --root INT\smoke\fake --zip-kind smoke`
2. `t02_test_bundle.py ingest --src INT\smoke\fake\data\test --root INT\smoke\tt`
3. `t02_checks.py t1 --smoke --only T1.1,T1.2,T1.3`：兩場都是 PASS。
4. `t02_test_bundle.py build --kind smoke --root INT\smoke\tt --zip-out INT\smoke\drive\C10_test_v1.zip`
   - 15.8 秒，重驗是真的呼叫 `t02_checks t1`，結果 PASS。
   - zip `30afde4a…`，MANIFEST `f29ad01a…`。

notebook（`PY colab/run_local_smoke.py colab/test02_{rank,player}.ipynb --scratch INT\smoke`）：

- **競賽二**：最後版本（`fbf87539…`）OK，1 分 44 秒。
  - 接縫和 prereg02_smoke 逐位元組相同（C0 最大差 0、logits 最大差 0、規則 B 不同 0 題、容許 1 題）。
  - 狀態 completed。
  - 改版前的兩個版本也各跑過一次，都 OK。
- **競賽一**：OK，1 分 39 秒。
  - 重現核對 6/6，重心差 0.0。
  - `seam_p02` 逐位元組相同。

本機工具（`--smoke --results-dir …\test02_smoke --tests-dir INT\smoke\tt\data\test --checks-dir INT\smoke\tt\results\test02\checks`）：

- **競賽二**
  - `opening` 18.8 秒，接縫部分和 `opening_val.npz` 逐位元組相同。
  - `apply` 1.1 秒。本機接縫在 smoke 只記錄：參照是真的 prereg02，但冒煙權重是 epoch 2，所以記為 FAILED，這是預期的。
  - `check_submission` 用一般模式檢查 smoke 子集題目檔，OK。
- **競賽一**
  - `--seam-ref` 用 `p02final\fix\player\tools_out` 的 4 個檔，加上 prereg02_smoke 的 dumps（組在 `INT\smoke\p02ref_player`，這樣能逐陣列比）。
  - seam_val 接縫通過：dumps 逐位元組相同、平均完全相同、Top-5 6/6 逐位元組相同。
  - public、private、`submit` 都完成，12 列。

結果 zip 的交接：

- 兩個 notebook 自己打包的 zip（競賽二 22 個成員、競賽一 18 個），用 `t02_test_bundle.py unpack --root INT\smoke\tt --results-dir INT\smoke\unpacked` 都能解開，狀態 completed。
- 在解開的樹上重跑本機工具，4 個輸出和直接讀 Drive 樹時逐位元組相同：
  - `submission_rank.csv`
  - `seam_val/rank_val_probs.csv`
  - `submission_player.csv`
  - `seam_val/top5.csv`
- `t02_checks t6 --smoke` 也跑得完。

負面情境（`INT\neg_smoke.py`，每個情境只改一樣東西，結束後還原）：

| 情境 | 競賽二 | 競賽一 |
|---|---|---|
| 寫進 prereg02_smoke／prereg01_smoke（執行 setup 格後呼叫 `guard_write`，含 `..` 路徑） | 4 條都擋下（PermissionError）；結果目錄和 `prereg02_smoke_other` 放行 | 4 條都擋下（RuntimeError）；同樣放行 |
| `_INPUT.json` 過期（public 題目檔改 1 byte，MANIFEST 跟著改） | preflight 停下：「public 已有逐盤輸出，但當時的題目檔和這次不同」 | preflight 停下：「有上一次用不同輸入檔算的 dump」 |
| 少一個測試檔 | 停下，列出缺的檔 | preflight 停下 |
| MANIFEST 的 SHA 不符 | 停下 | preflight 停下 |
| SMOKE 時 MANIFEST 種類是 official | 停下 | preflight 停下 |
| 上一次的接縫沒過（`seam_check.json` passed = false，本次新增） | 停下，要求把結果資料夾改名成 `rank_failed_1` | 不適用：每池開頭就刪掉 dump 重算 |
| 權重被改過（加 1 byte，之後連 mtime 一起還原） | 停下，指出 `epoch_2.pth` | preflight 停下，指出 `best_model.pth` |
| 全部還原後重跑 | 完成，8 秒（已完成的 dump 跳過） | 完成，84 秒（重算） |

### 1.3 整合時的修改

- **F1 `colab/build_test02_rank_nb.py`**（I4 提出的問題）
  - 問題：`dump_is_done` 會沿用已完成的 dump。接縫沒過之後在同一個 Drive 目錄重跑，比的還是同一份輸出，登記第 9 節第 2 條的「換一個執行階段重跑一次」就沒有重算。
  - 修法：preflight 加 ④-0。上一次的 `seam_check.json` 不是 passed 時停下，要求先把 `C10/results/test02/rank` 改名成 `rank_failed_1`（同一層），舊輸出也因此保留下來查原因。
  - 同時改了第一格「會停下來的情況」和 `T02_CHANGES` 的 preflight 那一列，並重建 notebook 和 diff（值見步驟 2）。
  - 冒煙測試的正面情境和負面情境都重跑過。
- **F2 `docs/TESTDAY.md` 第 4 步**：重跑前的處理改成同一個做法（改名成 `rank_failed_1`），並寫明 notebook 會強制這一步；也寫明競賽一每次都整份重算。
- **F3 `docs/COLAB_RUN.md` 測試集一節**：表格「中途斷線」那一列原本寫「已算好的部分會跳過」，改成「競賽二已算好的組會跳過；競賽一每個池都重算」。只改這一格，git diff 仍然只有新增的行。
- **F4 `tests/test_t02_dryrun.py`**：新檔，見步驟 5。
- `tools/t02_*.py`、`tools/check_submission.py`、`tools/official_log.py`、`colab/build_test02_player_nb.py` 都沒有改。
- `results/test02/rank/fallback_*.json` 是整合者用 `--rerun` 重跑的，參數和 I1 的相同。

### 1.4 需要寫進 PREREG rev 6 的偏離

**會碰到登記文字的**（rev 6 要逐條寫）：

1. **R10 備案的接縫（I1）**：加 `--fallback R10` 時，本機接縫仍然用 R20 在驗證題上算；只有 public、private 改用 R10。這是把第 9 節第 2 條和第 6 條放在一起的解讀，程序本身沒有改。
2. **競賽一本機接縫的判定（I2 a）**：照登記的條件（Q04、閘門過、第 1 層、Top-5 ≥ 398/400），另外加兩條：
   - 只用開局的 Q22、Q23 平均和 sr 中位數必須完全相同。
   - seam_val 的 dumps 和 prereg02 逐陣列相同時，24 個平均和 Top-5 檔也必須完全相同。
   - spec 3.3 把「平均相同」列為無條件的條件，但登記沒有這一條，所以沒有照 spec。
3. **`--no-cnn` 的接縫（I2 b）**：只核對上面那兩條開局部分；rev 5 沒寫到。
4. **`--no-cnn` 時閘門沒過（I2 c）**：沒有 Q00 可以退，所以保留選到的組合照樣交件。第 9 節第 4 條寫的是「改用 Q00」。
5. **T2「雜湊前 12 碼」（I3）**：實作是 SHA-1 digest 的前 12 bytes（96 bits），和產生虛無計數時相同，不是 12 個十六進位字元。
6. **T1 的行尾（I3）**：整份 CRLF 或整份 LF 都算 PASS；混用才要求 `crlf_to_lf`；單獨的 CR 算 FAIL_STRUCT。
7. **T1.4 的細節（I3）**：
   - 缺 SZ 只記錄（SGF 預設 19）；有 SZ 但不是 19 才算違規。
   - 解析錯誤和其他屬性只記錄。
   - (b) 用 MOVE_RE 的落子序列判斷，另外抓 pass。
8. **T2 (b)–(d) 的計數與分母（I3）**：
   - (b) 的分母是題目檔的盤數。
   - (c) 的複製數是在同檔另一題也出現的盤數，同色、異色都算進門檻，另外分開記。
   - (d) 以 private 那一側的盤數當分母；競賽一的每一側都算題目加候選。
9. **T4 的 TVD（I3）**：用第一手的座標分布，因為虛無最大值 0.048／0.073 是用座標算的；對稱類別的 TVD 另外記錄。spec 4.5 寫的是類別，登記只寫「第一手分布」。
10. **競賽二接縫沒過後的重跑（整合者 F1）**：登記的「換一個執行階段重跑一次」落實成「先把結果資料夾改名、再從頭算」。notebook 會強制這一步。

**只是實作說明**（rev 6 的修訂表記一句即可，不改登記內容）：

- **notebook 格子順序（I1）**：
  - 競賽二的 preflight 放在 GPU 關卡之後，照 spec「依序停下」的順序，不照 2.1 的清單順序。
  - 設定格要先讀 zip 的 MANIFEST 決定結果目錄，所以缺 zip 或種類不對會在第一格就停。
- **`notebook_sha256` 兩場的算法不同（I1、I2 f）**：
  - 競賽二：各格原始碼的 json，值換成 64 個 0。
  - 競賽一：`[cell_type, source]` 的緊湊 json，值換成佔位字串。
  - 兩隊分開，本來就不共用 helper。
  - 本機 real mode 下，競賽二只警告，競賽一會拒絕；Colab 跑完後不能再重建競賽一的 notebook。
- **比 spec 嚴的檢查**：
  - real mode 固定 `--checks-dir`，並要求凍結工具在 git 裡乾淨（I1）。
  - real mode 的 `--out-dir` 必須在 `results/test02` 裡（I2）。
  - 測試日的 T3 另外核對詞彙、`vocab_sha256`、整條 null S2、IDF 向量和 fit 參考集（I3）。
  - `official_log` 的 prepare／uploaded 多了幾項檢查（I4）。
- **`build` 接受 FAIL_OPENING（I4）**：要加 `--allow-fail-opening --reason`。spec 3.1 只寫 PASS／NORMALIZED 可以打包，但第 9 節第 6 條在 FAIL_OPENING 時要照樣做下去，所以一定要有 zip。
- **`ingest`（I4）**：同名、內容也相同的重複檔會接受，只有內容不同才拒絕。
- **競賽一 notebook 的測試檔（I2 e）**：只解開、只雜湊競賽一的 4 個檔（兩隊分開）。
- **T3 在競賽二（I3）**：只算 T3q，判定是 RECORD_ONLY。
- **spec 2.4 的冒煙來源（整合者）**：
  - 冒煙用的驅動樹來自 `p02final\fix`（commit 版 notebook），不是 spec 寫的 `p02final\smoke`。
  - 測試集 zip 走 ingest → t1 → `build --kind smoke`，和測試日同一條路；`from-val --zip-out` 也能產生 zip。

### 1.5 還沒驗到的地方與風險

1. **real mode 沒有端到端跑過。**
   - 沒跑過的包括：各工具的「已 commit 且乾淨」檢查、MANIFEST 種類必須是 official、`t1` 的 real 路徑、`official_log.py prepare`、`unpack` 到 `results/test02`。這些要有 commit 才能跑，現在只有單元測試（有 mock）。
   - 建議：兩個 commit 之後，在一份一次性的 clone（`core.autocrlf=false`）裡，用 from-val 的檔跑一次 real mode 的 opening／apply／run／submit。或者至少做 5.3 的彩排。
2. **CRLF 會讓 pin 失效。**
   - 競賽二工具對 prereg02 的 `frozen_config.json`、`rank_val_probs.csv`、`pred_B.csv`、`opening_val.npz` 用原始位元組核對 SHA。
   - 這台機器 `core.autocrlf=true`、repo 沒有 `.gitattributes`。在新的 clone，或對這些檔做 `git checkout`／`stash`／`reset`，json／csv 會變成 CRLF，競賽二就會拒跑。競賽一那邊會把 CRLF 折成 LF，不受影響。
   - 現在工作目錄裡是 LF，所以沒事。要不要加 `.gitattributes` 由總部決定。
3. **Colab 還沒用 test02 notebook 跑過。**
   - 真 GPU 上接縫的容許差、G4 的時間和運算單元（競賽二約 10 分鐘、約 2 單元；競賽一 15–20 分鐘、約 3 單元）都是推估，還沒實測。
   - 建議 10/28–11/02 做彩排。
4. **`docs/COLAB_RUN.md` 第 3 行的開頭提示還寫「現在要跑的是事前登記 02」。** 測試日前要由總部改（I4 的範圍外）。
5. **競賽二的本機接縫只在乾跑驗過。** 冒煙沒有 R20 的參照（冒煙權重是 epoch 2），所以本機接縫在冒煙只記錄。它的逐位元組相同是在 from-val 乾跑（commit 的 prereg02 dumps）上證明的；真的 Colab 輸出要到彩排或測試日才驗得到。
6. **provenance 記的是 commit 前的狀態。**
   - `results/test02/null/*` 記 `tool_committed_clean: false`（HEAD `5a5572e`）；`fallback_*.json` 記 `t02_rank.py_tracked: false`。
   - 數字不受影響：虛無的 npz 在整合者重跑時逐位元組相同。
   - 審查後如果改了 `t02_rank.py`（fallback-fit 的部分）或 `t02_checks.py`（precompute 的部分），commit 前要用 `--rerun --reason` 重產。
7. **`test_t02_dryrun.py` 的標準值依賴假 id 的格式**（自我檢索雜湊 `player_id|game_id`）。`t02_fake_tests.py` 的 id 格式改了，就要重寫這些值。
8. **T3 的限制（登記過的）**：候選裡訓練玩家只占約 20% 以下時偵測不到。

### 1.6 暫存

整合用的：`INT\`，約 3.5 GB（其中 `smoke\` 2.5 GB）。

- `smoke\`：冒煙驅動樹和負面情境的 log；可重建。
- `dry\`、`dry_final\`：乾跑樹。
- `null_dry\`、`nb\`、`fallback_I1\`、`tmp\`
- `neg_smoke.py`
- log：`pytest_step1.log`、`precompute_dry.log`、`dryrun_slow1.log`、`dryrun_slow2_train.log`、`final_full.log`

commit 之後可以刪：

- `INT\`（`neg_smoke.py` 和 log 可留）
- `testkit\I1\`、`I2player\`、`I3\`、`I4infra\`、`specB\fake_player`、`C\cache_train.pkl`（spec 8 第 9 步）

## 2. 獨立審查（2026-09-26，三個唯讀審查代理）

三個角度各自審查 1.2–1.4 交出的版本，repo 都沒有改（只在各自的 `testkit\review-*\` 放副本做實測）：

- **保真與接縫（fidelity）**：數字核心逐行比對過，和 prereg02 驗證過的一致（`apply_set`／`cmd_opening`／`fallback-fit` 對 `p02_rank_fuse`，`select`／`apply` 對 `p02_player_fuse`）；獨立重跑 `C10_SLOW=1` 的 from-val 乾跑 13 passed。找到 2 個 blocker、1 個 major、4 個 minor。
- **登記規則（rules）**：凍結檔、兩隊分開、T1–T6 的門檻與後果、交件組裝、接縫容許差、GPU 關卡都符合登記。找到 2 個 blocker、1 個 major、5 個 minor。
- **使用者操作與 Colab（user-colab）**：釘住的 SHA 和 Drive 上的檔一致，時間推估合理。找到 1 個 blocker、2 個 major、4 個 minor。

三份共 22 條，重複的合併後是 15 件事。編號 R1–R22 照審查交來的順序（fidelity 1–7、rules 8–15、user-colab 16–22）；每件事用 U 編號，第 3 節照 U 編號記處理結果。

| U | 嚴重度 | 內容 | 來源 | 修前核對 |
|---|---|---|---|---|
| U1 | blocker | `t02_test_bundle.py build --kind official` 重驗 T1.1–T1.3 時用 `--out results/test02/checks`、等 `<comp>/t1_built.json`；`t02_checks.py` real mode 對 `data/test` 只接受 `results/test02/checks/built`、只寫 `<out>/<comp>/t1.json`。測試日第 2 步一定失敗；單元測試把這一步 mock 掉了。 | R1、R9 | 讀程式確認：`recheck_target` 官方分支 vs `t02_checks.REAL_T1` 第二組、`resolve_paths` 的「writes only to」 |
| U2 | blocker | 競賽一 T1 是 FAIL_OPENING 時，notebook preflight 和 `t02_player.py run`／`submit` 都拒跑。登記第 9 節第 6 條：「競賽一照常每池自我檢索和防護」。spec 3.0／3.3 寫只收 PASS／NORMALIZED，和登記衝突。 | R2、R8、R16 | 讀程式確認：`T1_OK`、`check_t1`、產生器第 520 行 |
| U3 | major | 競賽二 notebook 中斷後接續時，已完成的組（含 seam_val）被 `dump_is_done` 沿用，public／private 可能在別的 GPU／torch／cuDNN 算；接縫只看 seam_val 的完成標記，本機也不比對。COLAB_RUN 還允許接續時改 A100。 | R3、R10、R17 | 讀程式確認：SEAM 格只讀 seam_val 的 `_DONE_`；FINAL 只記錄；`t02_rank.py` 沒有任何比對 |
| U4 | minor | 本機接縫直接用 notebook `seam_check.json` 的 `tol`，沒核對 1e-4／1e-3 規則。 | R4 | 確認 |
| U5 | minor | real mode 下 Colab 跑的 notebook 不是已 commit 的版本時，競賽二只警告（競賽一會拒絕）。 | R5、R11、R18 的 ④ | 確認 |
| U6 | minor | `t02_rank.py` 用原始位元組的 SHA 釘 prereg02 的 `frozen_config.json`、`rank_val_probs.csv`、`pred_B.csv`；`core.autocrlf=true`、沒有 `.gitattributes`，重新 checkout 會變 CRLF 而誤拒；`byte_identical` 也會誤判。 | R6、R15 | 確認（`git ls-files --eol` 三個檔目前是 i/lf w/lf） |
| U7 | minor | 競賽一本機接縫另加的「完全相同」條件（開局平均、sr 中位數、逐位元組相同時全部相同）會因 `.venv` 的 numpy／scipy 漂移而不過；provenance 沒記 scipy。 | R7 | 確認 |
| U8 | minor | real mode 可以用 `--fallback-file` 指定任何檔，也不檢查 `fallback_R10.json` 是不是 11/04 11:00 前 commit 的。 | R12 | 確認 |
| U9 | minor | `official_log.py prepare` 沒把交件檔和工具報告綁在一起（T1 verdict、接縫、real mode、報告記的 SHA）。 | R13 | 確認 |
| U10 | minor | 本機接縫沒過時的重跑流程不完整：notebook 自己不知道本機接縫沒過（不改名就沿用舊 dump）；已 commit 的解包檔讓 `unpack` 拒絕第二次的 zip，文件沒寫怎麼處理。 | R14 | 確認（`unpack` 對既有目標檔一律拒絕、沒有 `--rerun`） |
| U11 | major | 彩排後正式日的 COLAB_RUN 第 2、4、5、7 步沒考慮彩排留下的東西：同名舊 notebook 容易從「最近」開到；漏收拾時 Drive 有兩個 `C10_test_v1.zip`。 | R18 | 確認（文件文字） |
| U12 | minor | 競賽一 notebook 的測試包錯誤訊息叫使用者「重新上傳」，和 COLAB_RUN「不要自己重傳」矛盾。 | R19 | 確認 |
| U13 | minor | `test02_player_results.zip` 約 130 MB（推估），Drive 下載會跳病毒掃描提示，文件沒寫。 | R20 | 大小是審查的推估（prereg02 一池 43,003,790 bytes × 3），沒實測 |
| U14 | minor | COLAB_RUN 第 3 行還指向「事前登記 02」。 | R21 | 確認 |
| U15 | minor | COLAB_RUN 第 14 步叫競賽一照第 10 步處理「無法連線到 GPU 後端」，第 10 步是改選 A100，和第 13 步「不換 GPU」矛盾。 | R22 | 確認 |

## 3. 修正紀錄（2026-09-26，修正者；還沒有 commit）

**結論**：15 件都修了（U7 選「至少」那一版，見下），沒有拒絕或延後的。修完重跑第 1 節步驟 1–6，全部通過；凍結路徑 `git status --porcelain` 仍是空的。暫存在 `…\scratchpad\testkit\FIX\`（以下簡寫 `FIX`）。

### 3.1 逐件處理

| U | 狀態 | 改了什麼 |
|---|---|---|
| U1 | fixed | `recheck_target` 對所有 kind 都回傳 `(<checks>/built, <checks>/built/<comp>/t1.json)`，official 正好是 `t02_checks` 的 REAL_T1 第二組。改 docstring、TESTDAY 第 2 步的路徑和 commit 清單（`results/test02/checks/built/`）。新測試 `BuildWithRealChecks.test_official_recheck_uses_the_real_t1_pair`：不 mock `t02_checks`，攔下 `run_t1_recheck` 組出的指令，用 `t02_checks.build_parser()` 解析後交給 `resolve_paths(…, "real", REAL_T1)`，確認得到 `(data/test, results/test02/checks/built)`（舊程式在第一個斷言就不過）。不真的在 real mode 執行，以免在 repo 寫檔。 |
| U2 | fixed | `t02_player.py`：`T1_ALLOWED = T1_OK + ("FAIL_OPENING",)`，`check_t1` 只對 FAIL_STRUCT 和缺檔停；t1 紀錄多了 `fail_opening`、`rule`，照舊寫進 `pool_select.json`、`apply_report.json`、`submit_report.json`；`run` 印一行說明。產生器第 520 行改成接受 `('PASS', 'NORMALIZED', 'FAIL_OPENING')`，`T02_CHANGES` 的測試包那一列同步改寫；config.json 的 `test_bundle.t1` 原本就記 MANIFEST 的 t1。測試：`test_t02_player.Guards.test_fail_opening_keeps_the_per_pool_procedure`（public、private、submit 照常，報告記 verdict）、`test_t02_dryrun` 的 from-val 乾跑多跑一次 FAIL_OPENING 的 seam_val／public／private／submit（交件檔和 PASS 時逐位元組相同，`b5d7473e…`）、CPU 冒煙負面情境（MANIFEST 的 t1.player 改 FAIL_OPENING：notebook 跑完、config 記 FAIL_OPENING；改 FAIL_STRUCT：preflight 停下）。 |
| U3 | fixed | notebook SEAM 格：讀三組 `_DONE_val_p02e9_d4.json` 的 `(gpu_name, torch, cudnn)`，public／private 和 seam_val 不同就在 `problems` 記一條、`passed = False`，照 seam_failed 打包停下；下一次執行被 F1（④-0）擋住，要求先改名成 `rank_failed_1` 再整份重算。`seam_check.json` 多記 `dump_env`、`dump_env_same`；停下的訊息會列出原因。`t02_rank.py apply` 在讀任何 dump 前比對三組標記（`check_dump_env`），不同就拒絕、什麼都不寫。notebook 第一格「會停下來的情況」「中斷後怎麼接續」和 COLAB_RUN「中途斷線」那一列改寫成：接續要選中斷前的同一種 GPU，拿不到就先告訴 Claude。測試：`test_t02_rank.Refusals.test_dumps_from_another_environment_are_refused`（4 種改法）、`test_t02_dryrun.test_rank_dumps_from_another_environment_refused`（審查者的重現：from-val 樹把 public 的標記改成 A100／9.9.9，`apply` exit 1、輸出目錄只有 `opening_test.npz`）、CPU 冒煙負面情境（改 public 標記後重跑：接縫沒過、status seam_failed；再跑一次停在 F1 要求改名）。 |
| U4 | fixed | `t02_rank.seam_tol`：本機依 seam_val 和 `results/prereg02/rank/dumps/` 兩份 `_DONE_` 標記的 `gpu_name` 算容許差（相同且不是 None → 1e-4，否則 1e-3；和 notebook 的 `_same_gpu` 同一條規則），要求 `seam_check.json` 的 `tol` 等於它，不等就拒絕；本機接縫用自己算的值。smoke 例外（notebook 比的是 prereg02_smoke，本機沒有它的標記；本機接縫在 smoke 本來就只記錄）。`seam_local.json` 和 `apply_test_report.json` 記 `tol_check`。測試：`test_seam_tolerance_follows_the_registered_rule`。 |
| U5 | fixed | 新函式 `check_notebook_identity`：real mode 下 notebook_sha256 和已 commit 的 `colab/test02_rank.ipynb` 不同（或 repo 裡沒有這份 notebook）就拒絕；rehearsal 仍只警告。算法是對各格原始碼算，程式沒改時重建不會變，所以不會誤傷。測試：`test_notebook_identity_real_mode_refuses`。 |
| U6 | fixed | `pinned()` 改用 `pin_sha256`：`.json`、`.csv` 把 CRLF 折成 LF 再算（`rf.sha256_text`），`.npz` 仍是原始位元組；檔案現在是 LF，釘選值不用改（四個 pin 用兩種算法都相同，實測過）。`seam_compare` 的 `byte_identical_*` 也改成比對 CRLF 折回 LF 的內容。測試：`test_pins_fold_crlf_for_text_files`。沒有加 `.gitattributes`（仍由總部決定，見 3.4）。 |
| U7 | fixed（選「至少」那一版） | 登記條件之外的「完全相同」條件仍算進 passed（1.4 第 2 條的偏離不變，沒有改成只記錄，免得改動判定規則）。另外：provenance 和 `seam_local.json` 多記 numpy、scipy、Python 版本（`environment`）；`seam_local.json` 新增 `failed_implementation_additions`，只有這幾條沒過時 `run` 會印出「和 Colab 無關，先查本機程式或 .venv 的 numpy／scipy」；TESTDAY 第 6 步寫明這種情況不要先請使用者重跑。 |
| U8 | fixed | real mode 拒收 `--fallback-file`（在讀任何東西之前）。`load_fallback` 在 real mode 用 `git log -1 --format="%H %cI"` 查 `fallback_R10.json` 最後一次 commit，必須早於 `2026-11-04T11:00:00+08:00`，commit hash 和時間記進 `apply_test_report.json` 的 `fallback.file.commit_check.last_commit`。測試：`test_real_mode_uses_only_the_committed_file_from_before_the_test_data`（mock `git`：11:00 整拒絕、10:59:59 接受、查不到拒絕）。 |
| U9 | fixed | `official_log.py prepare` 在 `check_submission --official` 之後呼叫 `check_reports`：競賽二 `apply_test_report.json`（mode real、本機接縫過、check_submission official 沒有錯誤、`outputs_sha256` 等於這個檔）；競賽一 `submit_report.json`（mode real、official 且 passed、`output.sha256` 相同）；兩場 `seam_val/seam_local.json` passed 且 mode real；`results/test02/checks/<comp>/t1.json` 是 PASS／NORMALIZED，FAIL_OPENING 時競賽二必須 `fallback_used`、競賽一照常；FAIL_STRUCT 拒絕。這些檔都要已 commit 且乾淨，SHA-256 和 T1 verdict 記進上傳紀錄（`reports_sha256`、`t1_verdict`、`fallback_used`），請使用者確認的文字多一行「檢查：T1 …；接縫都通過」。報告以交件檔所在資料夾為準。測試：`OfficialLog.test_prepare_is_bound_to_passed_checks_and_seams`（11 種拒絕、FAIL_OPENING 的兩種放行）。 |
| U10 | fixed | TESTDAY 第 4 步新增「重跑一次」小節：① 已 unpack 過的話，用一行 Python 把 `results/test02/<comp>/` 全部（留下競賽二的 `fallback_*.json`）和 `unpack_<comp>.json` 移到 `results/test02/failed/<comp>_N/`、舊 zip 移到 `data/test02_failed/`，然後 commit（這一行在 `FIX\archive_test\` 的假樹上實測過）；② 使用者把 Drive 的 `rank` 改名成 `rank_failed_1`（競賽二一定要，競賽一建議）；③ 重跑；④ unpack 後確認新的 seam_val 完成標記 `time` 晚於封存那份，確定真的重算。第 5、6 步改成指向這一節。兩個工具的本機接縫沒過時，錯誤訊息都指向這一節。 |
| U11 | fixed | COLAB_RUN：第 2 步列出「第一次跑」和「做過彩排」兩種應該看到的清單；第 4 步「已有 `old_prereg02` 就跳過第 4、5 步」；第 5 步的彩排收拾改成後備檢查（同名檔只能一份）；第 7、12 步改成從 Drive 的 `C10` 資料夾開（右鍵 →「開啟工具」→「Google Colaboratory」，備援是「開啟筆記本」點進 `C10`），並核對第一格印的種類（競賽二「種類 official」、競賽一「kind official」）；新增第 21 步：彩排一跑完就把 3 個彩排檔收進 `old_rehearsal`。TESTDAY 彩排一節同步改成「彩排一結束就收拾」。real mode 的 notebook 身分核對見 U5。 |
| U12 | fixed | 競賽一 `_ZIP_HINT` 改成：不要自己重傳、解壓或改名，看 `C10` 裡是不是剛好一個 `C10_test_v1.zip`，截圖給 Claude。競賽二的 `TEST_HELP`（叫人再按一次「全部執行」）沒改：COLAB_RUN 允許「重新全部執行」。CPU 冒煙的「少一個測試檔」情境確認新字樣出現。 |
| U13 | fixed | COLAB_RUN 第 19 步加一行：競賽一的 zip 約 130 MB（推估），Drive 可能跳「無法掃描這個檔案是否含有病毒」之類的提示，按「仍要下載」。 |
| U14 | fixed | COLAB_RUN 第 3 行改成指向「測試集（test02）」一節（彩排和正式都用），兩節事前登記只留作紀錄。這是 COLAB_RUN 相對 HEAD 唯一改動的舊行，其餘仍是新增。 |
| U15 | fixed | COLAB_RUN 第 14 步寫明唯一例外：競賽一遇到「無法連線到 GPU 後端」「無法指派後端…」不改 A100，照第 13 步處理。只改測試集一節（事前登記 02 一節的同一行是紀錄，不動）。 |

### 3.2 改動的檔案

- 程式：`tools/t02_test_bundle.py`（U1）、`tools/t02_player.py`（U2、U7、U10 的訊息）、`tools/t02_rank.py`（U3–U6、U8、U10 的訊息）、`tools/official_log.py`（U9）、`colab/build_test02_player_nb.py`（U2、U12）、`colab/build_test02_rank_nb.py`（U3）。
- 重新產生：`colab/test02_rank.ipynb`、`colab/test02_player.ipynb`、`results/test02/notebook_diff_{rank,player}.txt`；`results/test02/rank/fallback_R10.json`、`fallback_selfcheck_R20.json`（`--rerun`，因為 `t02_rank.py` 的 SHA 變了；參數完全相同，差別只有 `created`、`seconds`、`rerun`、`tool.t02_rank.py_sha256`）。
- 文件：`docs/TESTDAY.md`（U1、U2、U3、U7、U9、U10、U11）、`docs/COLAB_RUN.md`（U3、U11、U13、U14、U15）。
- 測試：`tests/test_t02_infra.py`（U1、U9；OfficialLog 的測試樹多了工具報告）、`tests/test_t02_player.py`（U2）、`tests/test_t02_rank.py`（U3–U6、U8；合成樹的 `_DONE_` 標記多了 gpu_name／torch／cudnn，並補一份 prereg02 的標記）、`tests/test_t02_dryrun.py`（U2、U3）。新增 10 項測試。
- 沒有改：`tools/t02_checks.py`、`tools/t02_fake_tests.py`、`tools/check_submission.py`、`.gitignore`、`results/test02/null/*`、spec 與設計筆記；任何凍結檔。

### 3.3 重跑第 1 節步驟 1–6

`PY` = `.venv\Scripts\python.exe`（Python 3.12.13，torch 2.14.0+cpu），全程 CPU。沒有開任何答案檔、沒跑 `tools/score.py`。

| 步驟 | 指令 | 結果 |
|---|---|---|
| 1 | `PY -m pytest tests -q` | 367 passed、21 skipped（要 `C10_SLOW` 的乾跑），117 秒 |
| 2 | 兩個產生器各建兩次到 `FIX\nb1\`、`FIX\nb2\`，再寫進 repo | 三份逐位元組相同。競賽二 notebook `53337a7acc396bdc1f94b3cbdc7dd0ce859766511099e8004697ff0d6a487b05`、diff `9826ab965f697e813f2e18f30618c4ac798426540591aa46eb03257acc35f3bc`、`notebook_sha256` `dc7526be…`；競賽一 notebook `5c62cf5e1e4a36d989f0dc1479bdbdc0bc08f8d135001cfa15d754e4416e9918`、diff `0c61290d891f8d89ae937e1ba3e3794842094242eddfc23b89fe669262579e28`、原始碼 SHA `fc4a2df1…`。`test_t02_nb_*` 和 `test_t02_hygiene` 44 passed |
| 3 | `PY tools/t02_checks.py precompute --dry-run --out FIX\null_dry` | `pins OK`，539 秒（和其他工作同時跑）。6 組釘選數字、IDF SHA（`2523d83c…`、`0214853f…`）都和 1.2 相同；兩個 npz 和 repo 的 `results/test02/null/` 逐位元組相同（`4eb9cf05…`、`5312a18c…`），所以 repo 的檔不動 |
| 4 | `PY tools/t02_rank.py fallback-fit --combo R20 --rerun --reason …`，再 `--combo R10` | R20 最大差 0（容許 1e-12）；R10 a 1.223599、γ 0.433133、inner OOF 0.5687，擬合 25,400 題；參數和修正前的檔完全相同 |
| 5 | `C10_SLOW=2 C10_KEEP=1 C10_DRYRUN_DIR=FIX\dryrun PY -m pytest tests -q` | **388 passed、0 skipped**，624 秒。from-val 乾跑的標準值全部照舊（接縫 `a8370043…`、`863a6258…`、`69ca74a6…`，交件 `d9b9801b…`、`b5d7473e…`）；新加的 FAIL_OPENING 競賽一路徑和混用環境的拒絕都通過；from-train 的 5 項照舊 |
| 6 | CPU 冒煙，樹在 `FIX\smoke\`（驅動樹從 `INT\smoke\drive\results\prereg0{1,2}_smoke` 複製） | 見下 |

步驟 6 細節：

- 測試包照測試日的順序做：`from-val --zip-kind smoke` → `ingest` → `t1 --smoke --only T1.1,T1.2,T1.3`（兩場 PASS）→ `build --kind smoke`（12.7 秒，重驗寫在 `tt\results\test02\checks\built\<comp>\t1.json`）。zip `052dc4156ab0…`、MANIFEST `968895bce161…`。
- notebook：競賽二 OK（1 分 52 秒），接縫逐位元組相同、C0 最大差 0、`dump_env_same` true；競賽一 OK（1 分 34 秒），重現核對 6/6、重心差 0。
- 本機工具（`--smoke`）：競賽二 `opening` 接縫部分和 `opening_val.npz` 逐位元組相同，`apply` 完成（本機接縫在 smoke 照舊只記錄 FAILED，原因同 1.2）；競賽一接縫通過，public、private、`submit` 完成（12 列）。
- 兩個結果 zip 用 `unpack --root FIX\smoke\tt` 解到 `FIX\smoke\unpacked`，重跑本機工具，`submission_rank.csv`、`seam_val/rank_val_probs.csv`、`submission_player.csv`、`seam_val/top5.csv` 和直接讀 Drive 樹時逐位元組相同。
- 負面情境（`FIX\neg_smoke.py`，由 `INT\neg_smoke.py` 加上新情境）：兩場都「ALL AS EXPECTED」。

| 情境 | 競賽二 | 競賽一 |
|---|---|---|
| 1.2 原有的 6–7 個情境（寫進唯讀目錄、`_INPUT.json` 過期、少一個測試檔、MANIFEST SHA 不符、SMOKE 時種類 official、上一次接縫沒過、權重被改） | 全部照舊停下 | 全部照舊停下；少一個測試檔時的訊息是新的「不要自己重傳」 |
| **新**：接續時 public 的 dump 標記是別的 GPU／torch（U3） | 接縫沒過（`dump_env_same` false、status seam_failed），訊息列出三組環境；下一次執行停在 F1，要求改名 `rank_failed_1` | 不適用（每池重算） |
| **新**：MANIFEST 的 t1.player 是 FAIL_OPENING（U2） | 不適用 | 跑完，config 的 `test_bundle.t1` 記 FAIL_OPENING |
| **新**：MANIFEST 的 t1.player 是 FAIL_STRUCT | 不適用 | preflight 停下 |
| 全部還原後重跑 | 完成（已完成的 dump 跳過） | 完成（重算） |

### 3.4 需要寫進 PREREG rev 6 的（這一輪新增）

接在 1.4 後面。

**會碰到登記文字的**：

11. **接縫涵蓋的「每次執行」（U3）**：第 9 節第 2 條「每次…執行…重算一次」落實成「三組逐盤輸出必須是同一個執行環境算的（完成標記的 GPU 型號、torch、cuDNN 相同）」。中斷後在同一個環境接續，可以沿用已完成的組；環境不同就算接縫沒過，照 1.4 第 10 條改名後整份重算。notebook 和本機都檢查。
12. **重跑一次的程序（U10）**：1.4 第 10 條延伸到「本機接縫沒過」：先把本機的解包檔和輸出封存到 `results/test02/failed/<comp>_N/` 並 commit，Drive 資料夾改名，重跑後確認完成標記的時間比封存那份晚。

**只是實作說明，或 spec 的更正**（修訂表記一句即可）：

- **spec 3.0、3.3 的 T1 條件更正（U2）**：競賽一的 `run`、`submit` 和 notebook preflight 接受 FAIL_OPENING（照登記第 9 節第 6 條；spec 寫只收 PASS／NORMALIZED 是錯的）。FAIL_STRUCT 照舊停下。
- **spec 3.1 第 4 點的重驗輸出位置（U1）**：`build` 的 T1.1–T1.3 重驗一律寫 `<checks>/built/<comp>/t1.json`，real mode 是 `results/test02/checks/built/`。
- **容許差由本機重算（U4）**：本機依完成標記算 1e-4／1e-3，必須等於 notebook 記的；smoke 例外。
- **競賽二 real mode 拒絕不是已 commit 的 notebook（U5）**：取代 1.4「本機 real mode 下，競賽二只警告」那一句；rehearsal 仍只警告。
- **競賽二的 pin 對 CRLF 不敏感（U6）**：`.json`、`.csv` 的 SHA 先把 CRLF 折回 LF；釘選值不變。
- **R10 備案只用 11/04 前 commit 的檔（U8）**：real mode 不收 `--fallback-file`，並核對最後一次 commit 早於 2026-11-04 11:00 +08:00（第 9 節第 6 條「參數在 11/04 前算好並 commit」的機械化）。
- **上傳前綁定報告（U9）**：`prepare` 核對工具報告、接縫、T1 verdict，並把報告的 SHA-256 記進 `official_scores.json`（第 9 節第 12 條「檢查和接縫都過」的機械化）。
- **競賽一本機接縫的環境紀錄（U7）**：判定不變（1.4 第 2 條），多記 numpy／scipy 版本和「只有實作附加條件沒過」的欄位。

### 3.5 還沒驗到的地方與風險（取代 1.5 的第 2、4 項，其餘照舊）

1. **real mode 仍沒有端到端跑過**（1.5 第 1 項）。這一輪新增、只有單元測試或 mock 蓋到的 real 路徑：`build --kind official` 的重驗（用 `t02_checks` 自己的路徑規則驗過，沒有實際在 repo 執行）、`official_log prepare` 綁定報告、競賽二 real mode 拒絕別版 notebook、`fallback_R10.json` 的 commit 時間。建議照 1.5 第 1 項，兩個 commit 之後在一次性 clone 裡跑一次 real mode，包括 `build --kind official`。
2. **CRLF**（取代 1.5 第 2 項）：`t02_rank.py` 已不受影響；競賽一原本就不受影響。還沒解決的是：兩個 notebook 產生器對 `colab/prereg02_*.ipynb` 用原始位元組核對 SHA（`test_t02_hygiene` 的 pin 測試也是），重新 checkout 成 CRLF 時重建 notebook 和這個測試會失敗。只影響重建，不影響測試日的工具。要不要加 `.gitattributes`（例如 `*.ipynb -text`、`results/** -text`）仍由總部決定。
3. **Colab 還沒用 test02 notebook 跑過**（1.5 第 3 項照舊）。新增的環境比對在真 GPU 上的行為（`cudnn` 的值、同一種 GPU 接續時是否相同）要到彩排才看得到。
4. **競賽二接續時換 GPU 會多花一次重跑**：這是 U3 的設計（寧可整份重算，也不讓測試輸出沒被接縫涵蓋）。COLAB_RUN 已請使用者接續時選同一種 GPU。
5. **Drive／Colab 畫面字樣是憑記憶寫的**：「開啟工具」→「Google Colaboratory」、病毒掃描提示和「仍要下載」按鈕沒有對照目前的 zh-TW 介面核對；COLAB_RUN 都附了備援說法。`test02_player_results.zip` 約 130 MB 是推估。
6. **競賽一本機接縫的「完全相同」附加條件仍算進判定**（U7 只加了紀錄和訊息）。`.venv` 的 numpy／scipy 在 11/04 前被升級，競賽一會停到寫完修訂為止；現在看得出是這個原因。
7. **`fallback_*.json` 的 provenance** 記的是這一輪的 `t02_rank.py` SHA（`347fee37…`，還沒 commit）。commit 前再改 `t02_rank.py` 的話，要照 1.5 第 6 項用 `--rerun --reason` 重產。
8. **TESTDAY 的封存指令**只在 `FIX\archive_test\` 的假樹上實測過。

### 3.6 暫存

`FIX\`（約 3.1 GB，其中 `smoke\` 2.5 GB、`dryrun\` 0.55 GB）：`smoke\`（冒煙驅動樹、負面情境 log）、`dryrun\`（`C10_SLOW=2` 的兩棵乾跑樹）、`null_dry\`、`nb1\`、`nb2\`、`fallback_before\`（修正前的兩個 fallback 檔）、`archive_test\`、`patch_*.py`（這一輪的修改腳本）、`neg_smoke.py`、log（`pytest_step1.log`、`precompute_dry.log`、`final_full.log`）。commit 之後可以刪（`neg_smoke.py` 和 log 可留）。

## 4. real mode 端到端與使用者文件審查後的修正（2026-09-26，第二輪修正者；還沒有 commit）

兩個審查代理的結果：real mode 端到端（拋棄式 worktree，照 TESTDAY 跑完）和使用者文件走查（COLAB_RUN「測試集（test02）」一節、TESTDAY）。共 18 件，逐件核對程式或文件後都成立：major 3 件、minor 15 件。

**結論**：

- 3 件 major 都修了，程式和文件兩邊都改。
- 15 件 minor 修了 13 件，另 2 件部分處理（第 6、14 件，見 4.1）。
- 沒有改產生器，所以 notebook、notebook diff 不用重建，也不用 CPU 冒煙。
- 凍結檔、`results/test02/null/*`、`results/test02/rank/fallback_*.json`、PREREG 都沒動。
- 暫存在 `…\scratchpad\fix2\`（以下簡寫 `FIX2`）。

### 4.1 逐件處理

| # | 來源、嚴重度 | 核實 | 處理 |
|---|---|---|---|
| 1 | realmode，major | 成立。`official_log.check_reports` 只看 T1 的 `verdict`；`t02_test_bundle.load_t1`、`t02_rank.t1_status`、`t02_player.check_t1` 都不看 `complete`；`t02_checks.py t1` 在 real mode 對 `data/test/raw` 接受 `--only`。 | 五處都改。① `t02_checks.py t1` 在 real mode 對 `data/test/raw` 拒絕 `--only`；built 的重驗照舊可以帶 `--only`。② `build --kind official`、`t02_rank.py`、`t02_player.py` 的 real mode、`official_log.py prepare` 都要求 raw 的 `t1.json` 符合 `complete: true`、`stage: raw`、provenance `mode: real`。③ `t02_checks.py run` 在 real mode 另外要求 `stage: raw`。④ U1 的前提「T2–T5 已記錄」由 `prepare` 強制：`summary.json` 已 commit，T2–T5 四項都在，細節檔已 commit，而且 SHA-256 等於 summary 記的。verdict 不擋上傳。真的跑不了時，用 `--checks-missing-reason TEXT` 放行，原因記進上傳紀錄並印在確認段。T2–T5 的 verdict 和檔案 SHA-256 記進上傳紀錄的 `t2_t5`、`reports_sha256`。 |
| 2 | realmode，major | 成立。`common_checks` 的 `require_committed_clean` 只包含 `SELF` 和 check_submission；`check_notebook_identity` 是拿工作目錄的 notebook 比。 | `colab/test02_rank.ipynb` 加進 real mode 的 committed-clean 清單，和競賽一 `real_mode_checks` 的做法一致。notebook 已 commit 且乾淨後，工作目錄的內容就等於 HEAD，所以不另外讀 HEAD blob。 |
| 3 | realmode，minor | 成立。 | 競賽二的交件檔是用 R10 備案做的時候，預設方法改成 `prereg02 R10 fallback rule B (甲, T1 FAIL_OPENING)`。`--method` 寫到另一個組合（R20／R10，整詞比對）就拒絕。 |
| 4 | realmode，minor | 成立（依程式碼）。 | ① `ingest` 寫進 repo 的 `data/test/raw` 時，來源裡只要有本工具組自己的紀錄就拒絕：測試包的 `MANIFEST.json`（zip 成員或檔案，任何種類）、`SOURCE.json`、`FAKE_TREE.json`（只打開這三種檔名，而且要解析得出本工具的 format）。寫到 scratch 時只印 note，因為冒煙流程會 ingest from-val 樹（見 3.3 步驟 6）。`SOURCE.json` 多記 `own_records_in_sources`。② real mode 的內容檢查（`t1` 的 T1.4／T1.5、`run`）改為照樣計算驗證棋譜的 SHA-1 命中，逐檔記進 `provenance.validation_hash_guard`。一個檔有一半以上的棋是驗證棋譜就拒絕（那是驗證衍生的樹）；少量命中只警告並記錄，因為官方檔可能真的含訓練集的棋，T2 就是在抓這個。本機沒有 `data/local` 時記成 `available: false`，照常進行。和登記第 10 節第 16 條的文字不一致，見 4.4。 |
| 5 | realmode，minor | 成立。 | `prepare` 要求 `tools/check_submission.py`、`tools/official_log.py` 都已 commit 且乾淨。 |
| 6 | realmode，minor | 成立。在 scratch repo 實測（autocrlf=true）：刪檔後 `git checkout --` 寫出 CRLF，`git status` 仍乾淨；`git -c core.autocrlf=false checkout -- <path>` 在狀態乾淨時也會寫回 LF。 | 部分處理。`prepare` 遇到 SHA 不符、而且把 CRLF 折回 LF 後就等於報告記的 SHA 時，錯誤訊息會附上 `git -c core.autocrlf=false checkout -- <path>`；`uploaded` 的還原提示也改成同一行。**沒有加 `.gitattributes`**：不在這一輪允許改的檔案內。建議 `results/test02/** -text`、`results/official_scores.json -text`，由總部決定（同 3.5 第 2 項）。 |
| 7 | userdocs，major | 成立。兩份 notebook 在紅字前都印「要再跑請重新「全部執行」」；競賽一沒有類似 ④-0 的擋；`run_status` 和 `t02_player` 只看最後一次。另外查到：`env.json` 只有 GPU、torch 等欄位，沒有能分辨「同一個執行階段」的欄位（例如 hostname、開機 id），所以同一個執行階段無法從紀錄直接驗出。 | 文件和工具兩邊都改。① COLAB_RUN 第 16 步和「出狀況時」第 1 列寫明：那一行不算指示，出現紅字後一律不自己重跑。能自己處理的只剩 GPU、雲端硬碟授權、帳號三種。② `unpack` 印出並記錄 `run_log.json` 每一次執行的狀態（`runs`、`earlier_failed_runs`）。最後一次之前如果有 `seam_failed`、`repro_failed`、`readonly_changed`、`final_check_failed`：解到 repo 的結果資料夾就拒絕，解到 scratch 只警告。這種情況代表沒改 Drive 資料夾名稱就又跑了一次。`preflight_failed`、`interrupted`、`running` 不算。③ `t02_rank.py` 的 `load_notebook`、`t02_player.py` 的 `check_notebook`，在 real mode 也做同樣的拒絕。④ TESTDAY 第 4 節：每次都看 unpack 印的執行紀錄；被拒時把那次算作第 N 次嘗試、那次的輸出不用，照「重跑一次」處理。notebook 沒改（要改就得動產生器，見 4.5）。 |
| 8 | userdocs，minor | 成立。 | T0 的指令區塊第一行先 `os.makedirs('results/test02/checks', exist_ok=True)`，並加一句說明。 |
| 9 | userdocs，minor | 成立。 | 引言改成「Claude 說可以跑了之後才用；正式在 11/04 11:00 後，彩排在 10/28–11/02」；「見最後一段」改成「見第 21 步之後的「上傳交件檔」」。 |
| 10 | userdocs，minor | 成立（`c10-player02-07` 印的是 `player_results.zip`；這一格要和 prereg02 逐位元組相同，不能改）。 | COLAB_RUN 第 16 步和出狀況時那一列寫明：訊息裡的 `player_results.zip` 就是 `results`→`test02` 裡的 `test02_player_results.zip`；不要下載 `prereg01`、`prereg02` 資料夾裡同名的舊檔。 |
| 11 | userdocs，minor | 成立。 | 種類檢查移到第 11、15 步（和 `device` 那一行一起看）。第 7、12 步改成「先不要按全部執行」。第 13、15 步改成「從第 12 步開檔重試」。 |
| 12 | userdocs，minor | 成立。 | 「重跑一次」第 1 點改成每次都封存。封存指令加了三件事：沒 unpack 過（`results/test02/<comp>/` 不存在）也能跑；把 zip 的 SHA-256 和原因寫進 `failed/<comp>_N/attempt.json`，這樣一定有東西可以 commit；目錄已存在就停下，不覆蓋。在 `FIX2\archive_test\` 的假樹上實測三種情況：競賽二已 unpack、競賽一沒 unpack、同一場第 2 次，都照預期。第 4 節另寫明：遇到 ` (1)` 檔名時先確認、改名或用 `--zip` 指定；重跑前先刪「下載」資料夾裡的舊 zip。第 4 點補上沒 unpack 過時怎麼確認真的重算了（看 unpack 印的執行紀錄）。 |
| 13 | userdocs，minor | 成立。 | 彩排一節：① 修訂改寫成「rev 7」，彩排記錄寫進 RESULTS。② 預期值拆成 G4（逐位元組相同）和 A100（容許差 1e-3、規則 B 差 ≤ 4）兩種。③ 補上複製檔案並核對 SHA-256 的指令；zip 的 SHA 要等於 `<S>\FAKE_TREE.json` 的 `zip_sha256`。④ 收尾時請使用者刪掉「下載」裡彩排的兩個 zip；Drive 上的 `test02_rehearsal` 不動。另外寫明彩排的 `t1 --only` 指令不要照抄到正式那天。第 13 行「rev 6 起」改成「下一版是 rev 7」。 |
| 14 | userdocs，minor | 成立（`build_test02_rank_nb.py` 第 504–510 行逐一核對 6 個檔的 SHA-256）。 | **登記不改**，列在 4.4。notebook 不用改。 |
| 15 | userdocs，minor | 成立。 | ① 第 1 節改成「照當天畫面找下載處（可能在「資料」分頁）」。② 訊息 B 寫比賽全名和 AIdea 比賽頁網址（出自 `docs/REGISTRATION.md`「來源」；那是比賽頁，不是隊伍網址），並提醒隊號大小和場次相反、同一帳號還有 C1。③ 上傳頁要求的檔名或格式不同時先停下，照 D5 處理。 |
| 16 | userdocs，minor | 成立。 | TESTDAY 第 4 節新增「競賽一拿不到 G4」：競賽二先上傳；請使用者回報每次試的時間和畫面；滿一天後只在「繼續等」和「換 GPU」之間選。換 GPU 要先寫 rev 7、不放寬核對，改產生器並走完整的重建審查流程。建議最晚 11/20 決定（這是程序上的建議，見 4.4）。COLAB_RUN 第 13 步補上：隔天單獨開競賽一時先查運算單元（至少 5 個）。 |
| 17 | userdocs，minor | 成立。 | COLAB_RUN 第 3 行的「上面兩節」改成「下面先出現的「在 Colab 跑事前登記 01」「事前登記 02」兩節（在測試集一節之前）」。這一行在測試集一節之外，但它就是指向測試集一節的那行指引。 |
| 18 | userdocs，minor | 成立。 | 表格加上 A100：時間寫「沒實測，推估比 G4 慢（A100 只配 12 顆 CPU，G4 配 48 顆）」，沒有編數字。運算單元寫「A100 每小時約 5.4 個（第一節引用的第二手實測），跑滿 1 小時也在第 8 步的 10 個以內」。 |

### 4.2 改動的檔案

- 程式：
  - `tools/official_log.py`：第 1、3、5、6 件。
  - `tools/t02_checks.py`：第 1、4 件。
  - `tools/t02_test_bundle.py`：第 1、4、7 件。
  - `tools/t02_rank.py`：第 1、2、7 件。
  - `tools/t02_player.py`：第 1、7 件。
  - 以上五個檔的 docstring 都同步更新。
- 文件：
  - `docs/TESTDAY.md`：第 1、7、8、12、13、15、16 件，另有第 2 件的一句。
  - `docs/COLAB_RUN.md`：第 7、9、10、11、13、16、17、18 件。除了第 3 行，其餘改動都在測試集一節內。
- 測試：新增 16 項。
  - `tests/test_t02_infra.py`：`OfficialLog` 的測試樹改成完整 raw T1，並補上 T2–T5 的 summary。新增：
    - `test_prepare_needs_the_full_raw_t1`
    - `test_prepare_needs_t2_to_t5_recorded`
    - `test_rank_method_follows_the_fallback`
    - `test_prepare_needs_its_own_tools_committed`
    - `test_crlf_checkout_gets_a_restore_hint`
    - `Unpack.test_earlier_failed_run_in_the_same_folder`（repo 路徑指向暫存資料夾，測 real 的結果資料夾）
    - `Ingest.test_own_bundles_refused_by_the_real_ingest`
    - `OfficialBuildNeedsFullT1.test_load_t1`
  - `tests/test_t02_checks.py`：新增 `RealModeT1` 類別，3 項。模組路徑指向暫存資料夾、mock git。
    - raw 的 `--only` 被拒，完整 T1 寫出 `complete`、`stage raw`、mode real；built 的重驗照舊。
    - 驗證衍生的樹被拒；少量命中只記錄；沒有 hashes 時記成 unavailable。
    - 比例門檻。
  - `tests/test_t02_rank.py`：新增 3 項。
    - real mode 的 committed-clean 清單包含 notebook。
    - real mode 的 T1 必須是完整 raw T1。
    - 前面有失敗的執行時拒絕；`running` 不算。
  - `tests/test_t02_player.py`：新增 2 項（完整 raw T1；前面有失敗的執行）。
- 沒有改：
  - 兩個產生器、兩份 notebook、`results/test02/notebook_diff_*.txt`。
  - `tools/check_submission.py`、`tools/t02_fake_tests.py`。
  - `results/test02/null/*`、`results/test02/rank/fallback_*.json`。這兩個檔記的 `t02_rank.py` SHA 是修改前的值（3.5 第 7 項）。擬合的程式沒有改，沒有任何程式核對這個欄位，所以沒有重產。
  - `docs/PREREG_02_OPENING_FUSION.md`、spec、設計筆記、`.gitattributes`、任何凍結檔。

### 4.3 驗證

| 項目 | 結果 |
|---|---|
| `PY -m pytest -q tests` | 383 passed、21 skipped（要 `C10_SLOW` 的乾跑），65 subtests passed，99 秒。修正前是 367 passed。 |
| `C10_SLOW=2 PY -m pytest -q tests/test_t02_dryrun.py` | **20 passed、0 skipped**，526 秒（from-val 管線含 FAIL_OPENING／R10 路徑，以及 from-train 的完整 T1、T2–T5；乾跑模式不受 real mode 新規則影響，標準值照舊） |
| real mode 拒絕（在 repo 直接跑，不寫任何檔） | `PY tools/t02_checks.py t1 --comp rank --only T1.1,T1.2,T1.3`：exit 1，訊息為「--only is refused here」。`data/test` 不存在，`git status` 只有這一輪改的 11 個檔。 |
| TESTDAY 封存指令 | `FIX2\archive_test.sh` 在假樹上跑三種情況，都照預期：`failed/<comp>_N/` 裡有 `attempt.json`；zip 移到 `data/test02_failed/…_N.zip`；fallback 檔留在原處。 |
| git 的 CRLF 行為 | `FIX2\gitcrlf\`（autocrlf=true 的 scratch repo）：`git checkout --` 寫出 CRLF；`git -c core.autocrlf=false checkout --` 在狀態乾淨時也會寫回 LF。 |
| 凍結檔 | `git status --porcelain` 沒有列出 `tools/p02_*`、`c10data.py`、`decide.py`、`score.py`、`colab/*prereg0*`、`results/prereg0*`、`results/score_ledger.csv`、`docs/VALIDATION.md`、`data/**`。`test_t02_hygiene` 通過。 |

沒做的：這一輪沒有在拋棄式 worktree 重跑 real mode 端到端。新加的 real 路徑都用單元測試蓋到：module 路徑指向暫存 repo 並 mock git，或直接呼叫函式。要注意的是，real mode 現在會拒絕在驗證衍生樹上跑完整 T1，而完整 T1 又是 build 的前提，所以之後的 real mode 演練在 build 之前要用 FIXTURE 的 `t1.json`（手寫 `complete`、`stage`、`mode`）。那份檔只能放在拋棄式 worktree 裡。

### 4.4 需要寫進 PREREG rev 7 的（這一輪沒有改登記文字）

**會碰到登記文字的**：

1. **第 10 節第 12 條更正**：「兩份 notebook 各自只雜湊自己那場的測試檔」不對。實際上競賽一 notebook 只解開、只雜湊自己的 4 個檔；競賽二 notebook 核對全部 6 個檔的 SHA-256 和 MANIFEST，但只推論自己的 2 個題目檔。
2. **第 10 節第 16 條在 real mode 的做法**：登記寫「內容檢查遇到含本機驗證棋譜的檔一律拒絕執行」。現在的做法是：非 real 模式照舊，命中一盤就拒絕。real mode 逐檔記錄命中數，一個檔有一半以上是驗證棋譜才拒絕，因為官方檔可能真的含訓練集的棋，拒絕就會卡住整條流程。修正前 real mode 完全沒有這道檢查。
3. **「重跑一次」的機械化（第 9 節第 2 條、第 10 節第 10 條）**：`run_log.json` 裡最後一次之前有 `seam_failed`、`repro_failed`、`readonly_changed`、`final_check_failed` 的，那次執行的輸出不用。這種情況代表在同一個 Drive 資料夾重跑，不是登記的重跑，算作一次嘗試。`preflight_failed`、`interrupted` 之後的重跑允許。這條會影響哪些輸出能用，建議列在 rev 7 的「會影響測試日判定」。已知限制：紀錄裡沒有能分辨「同一個執行階段」的欄位，所以「換一個執行階段」只能靠 Drive 改名，加上文件要求先刪執行階段再開新的。
4. **競賽一拿不到 G4（第 9 節第 10 條）**：TESTDAY 新寫的決定程序是：滿一天後只在「繼續等」和「先寫修訂再換 GPU（重現核對不放寬）」之間選。「建議最晚 11/20 改走換 GPU」只是程序建議；要有約束力，就在 11/04 前寫進 rev 7。

**只是實作說明**（修訂表記一句即可）：

5. 上傳前的 T1 必須是 raw 的完整 T1（`complete`、`stage raw`、mode real）。`build --kind official`、`t02_rank`、`t02_player`、`prepare` 都檢查；real mode 的 `t1` 在 `data/test/raw` 不收 `--only`。這是第 9 節第 6 條的機械化。
6. `prepare` 要求 T2–T5 已記錄；verdict 不擋上傳，有 `--checks-missing-reason` 可以放行。這是 TESTDAY U1 和第 9 節第 12 條「檢查…都過」的解讀：只要求 T2–T5 跑過，不要求它們通過，因為第 9 節第 7–9 條規定它們不擋甲。另外要求 `check_submission.py`、`official_log.py` 已 commit 且乾淨；R10 備案的交件檔記成 R10 的方法。
7. real 的 `ingest` 拒絕本工具組自己產生的測試包、假樹、ingest 紀錄。
8. 競賽二 real mode 要求 `colab/test02_rank.ipynb` 已 commit 且乾淨。這一條把第 10 節第 13 條落實到競賽二。

### 4.5 還沒驗到的地方與風險（3.5 照舊，另加）

1. **notebook 本身仍會印「要再跑請重新「全部執行」」**，競賽一也仍然沒有類似 ④-0 的擋。目前做到的是：本機工具會拒絕在同一個資料夾重跑出來的結果，COLAB_RUN 也寫明不要重跑。但使用者如果照做了，會多花約 3 個運算單元和一次來回。要在 notebook 擋下來（競賽一 preflight 看到上一次的 run_log 有失敗就停，或把那一行改寫），就得改產生器、重建、冒煙、獨立審查。建議 11/04 前做，或在 rev 7 決定不做。
2. **CRLF**：`.gitattributes` 仍然沒加（第 6 件）。現在工具遇到 CRLF 會拒絕，並給出還原指令，方向是安全的。
3. **驗證棋譜的 50% 門檻**是這一輪訂的，沒有在官方資料上看過。官方檔如果大量取自訓練集，而其中剛好一半以上是我們的驗證玩家，real mode 就會拒絕內容檢查。依驗證集占訓練玩家的比例，這種情況幾乎不可能發生；真的發生時要寫修訂，不能繞過。
4. **`--checks-missing-reason`** 是逃生口。TESTDAY 寫明只在 T2–T5 真的跑不了時用，並寫進 RESULTS；工具不判斷理由是否正當。

### 4.6 暫存

`FIX2\`：
- 這一輪的修改腳本 `patch_*.py`、`patchlib.py`
- `archive_cmd.txt`、`archive_test.sh` 和 `archive_test\`（封存指令的假樹）
- `gitcrlf\`（CRLF 實驗）
- `slow_dryrun.log`

都可以刪。

## 5. 官方資料結構（PREREG_02 rev 8）：建置、審查與修正（2026-09-28）

起因：AIdea「資料」分頁的官方說明（`docs/DATA.md` 最後一節）：競賽一的候選和題目只收執黑棋譜、`color_1…color_20` 一律是 B；兩場的題目都是 5／10／15／20 盤各 100 題；競賽二同一題只有一種執色。9/28 的模擬與程式檢查在 `results/test02/design/official_structure/README.md`。流程：一個建置代理（I-T02）→ 一個唯讀審查代理（和 prereg03 rev 3 一起審）→ 一個修正者。**仍然沒有任何測試資料**；沒有 commit，沒讀答案檔，沒跑 `score.py`，沒用 GPU，凍結檔沒動；內容檢查只在訓練玩家做的假檔上跑。暫存在 `…\scratchpad\osfix\`（`dryrun\`、`precompute_*\`、`tests\`：建置；`review\`：審查；`fix\`：修正）。

### 5.1 建置（I-T02）

- `tools/t02_checks.py`：
  - T1.3 新增白名單轉換 `clear_colour_after_num_games_if_sgf_empty`（`num_games` 之後、`sgf_i` 空或只有空白、`color_i` 剛好是 B/W/b/w → 清掉 color，T1 判 NORMALIZED；`sgf_i` 有內容、color 是別的值仍是 FAIL_STRUCT）。官方執色設計只記錄在 `official_colour_design_record_only`。
  - T3b：`b_c` 剛好是 1 的池改用只含 fit 玩家黑棋的參考集；虛無分布、pins（`T3B_PINS`）、`precompute --part all|base|t3b`；該池原本 T3 的數字放 `registered_T3_record_only`。
  - T2(a) 分方（`same_side_copies_record_only`／`opposite_side_copies_record_only`／執色不明），門檻仍算全部複製；索引快取 v2 多存執色，舊快取自動重建。
  - T4 整個檔全黑時混色虛無範圍標 `not applicable`；T5 多 `official_design_match`。
  - real mode 的 `run` 要 7 個虛無檔都已 commit 且乾淨。
- `tools/t02_test_bundle.py`：白名單和 `normalize_bytes` 加入新轉換；`quick_structure` 維持嚴格。
- `tools/t02_fake_tests.py`：`from-train --shape official --slot-colours empty|filled`。
- 測試：`tests/test_t02_checks.py`、`test_t02_infra.py`、`test_t02_dryrun.py`（C10_SLOW=2 多一組 `FromTrainOfficialShape`）。
- `docs/TESTDAY.md`：11/04 前就緒清單的 T3b 一行、第 1 步表格和下一段、第 3 步的 rev 8 補充。
- 新檔：`results/test02/null/t3b_null.json`、`t3b_null_raw.npz`、`t3b_null_canon.npz`；rev 5 的 4 個虛無檔逐位元組不變。

### 5.2 審查（唯讀；結論：沒有 blocker、major）

審查者在 scratch 用 `precompute --part all --dry-run` 重跑（921 秒），4 個 npz 的 SHA-256 和 repo 相同；`t3b_null.json` 的數字和建議的 rev 8 條文逐項相符；T1.3 新正規化不會蓋掉真正的結構錯誤；T2(a) 分方、官方形狀假檔都對。和 test02 有關的 minor 3 件：

| # | 內容 |
|---|---|
| 2 | T3b 只在 `b_c` 剛好是 1 時啟用。官方池只要混進一盤非 B（資料瑕疵或說明不準），那一池就退回原本的 T3；原本的 T3 在幾乎全黑的 profile 上幾乎沒有偵測力（真陽性 raw 0.1253、canon 0.1359，π＝0.5 時判定重疊的機率 0），卻照樣報 PASS。 |
| 3 | rev 8 之後還是舊的文件：`docs/TESTDAY.md:13`（「下一版是 rev 7」）、`spec.md:279、589–590`（白名單少新轉換、「i > n: both empty」）、`checks.md:28`、`docs/DATA.md:107`（「9/28 起的修訂」）。 |
| 4 | rev 8 條文「df 只算參考集裡的黑棋」可能被讀成盤數；程式算的是「黑棋 fit 棋裡出現過該 token 的 fit 玩家數」。 |

另記：`checks.md` 的 T1.2 要求 `pub_`／`priv_` 前綴，AIdea 的說明沒提。修正者查過：前綴出自官方教學 notebook（tutorial `359e06f`，`DATA.md` 第 84 行也記了），不是自己推的，不改。

### 5.3 修正（FIXER）

| # | 處理 |
|---|---|
| 2 | 選「最少」的做法，不改 T3b 的啟用條件（改成 b_c ≥ 0.99 要另算虛無分布、另外登記）。新常數 `T3_LOW_POWER_SHARE = 0.9`、函式 `t3_low_power(b_c, variant)`：留在原本 T3、而且 max(b_c, 1 − b_c) ≥ 0.9 的池（含整池白棋，T3b 只處理黑棋）標 `low_power`。每池新增 `low_power`、`conclusion`（`overlap`／`no overlap declared`／`not determinable (low power)`／`not computed`）；`t3.json` 新增 `low_power_pools`、`low_power_rule`。verdict 順序：ERROR → NOT_EVALUABLE（T1 FAIL_OPENING）→ OVERLAP（任一池判定重疊，含 low_power 的池）→ **LOW_POWER**（沒有判定重疊、但有 low_power 的池）→ PASS。`CONSEQUENCE[("T3","LOW_POWER")]`：不寫「沒有重疊」、記「無法判斷（偵測力不足）」，和 PASS 一樣不擋上傳、不觸發乙。`T3_RULE`、`T3B_RULE`、`T3B_DEFINITIONS["when"]` 同步。0.9 是事先定的界線，沒有在 0.9–1 之間模擬偵測力。`official_log.py` 對 T2–T5 的 verdict 不設白名單，新值可以直接記。 |
| 3 | **沒改**：這四處不在修正者可以改的檔案內，而且 TESTDAY 第 13 行要填 rev 8 的 commit hash。確切的替換文字交給主 session，和 rev 8 同一個 commit 一起改（見 5.6）。`docs/TESTDAY.md` 裡屬於 T3 的段落（第 3 步 rev 8 補充）有改，見下。 |
| 4 | `T3B_DEFINITIONS["weights"]` 改成「df_black＝黑棋 fit 棋裡出現過該 token 的 fit 玩家數（R_b 的列；以玩家為單位，同 T3 的 df；不是盤數）」。rev 8 條文的建議文字同步改。 |

`docs/TESTDAY.md` 第 3 步 rev 8 補充的 T3 一點，下面加一行：近單色、留在原本 T3 的池標 `low_power`，沒判定重疊時 verdict 是 `LOW_POWER`，RESULTS 寫「無法判斷（偵測力不足）」，不寫「沒有重疊」。

因為 `tools/t02_checks.py` 改了（`T3B_DEFINITIONS`、`T3B_RULE` 會寫進 `t3b_null.json`），用 real mode 重算一次 T3b 的 3 個檔：`PY tools/t02_checks.py precompute --part t3b --rerun --reason "..."`（見 5.4）。

新增測試：
- `tests/test_t02_checks.py` `T3bBlackOnly.test_near_single_colour_pool_left_on_t3_is_low_power`：`t3_low_power` 的邊界（0.9、360/400、1 − 0.1、0.1、0、0.89、0.11、T3b、None）；10 位新玩家各 95 黑 5 白 → T3、low_power、沒判定重疊 → LOW_POWER（FAIL_OPENING 時 NOT_EVALUABLE）；85 黑 → 不標；整池白棋 → 標 low_power 但判定重疊仍是 OVERLAP；全黑池用 T3b，不標。
- `tests/test_t02_dryrun.py`：兩個 NEG 池（混色樹、官方形狀樹）加斷言 `low_power` 為假、`conclusion` 是 `no overlap declared`、`low_power_pools` 空。

### 5.4 驗證（修正後）

| 項目 | 結果 |
|---|---|
| `PY -m pytest -q tests/test_t02_checks.py tests/test_t02_infra.py tests/test_t02_hygiene.py` | 118 passed（115 秒；建置時 117，加新測試 1 項） |
| `PY -m pytest -q tests/test_p03_rank_fuse.py` | 45 passed（見 `results/prereg03/design/build_review.md` 第 5 節） |
| `PY -m pytest -q tests`（完整，`fix\tests\full_suite.log`） | **478 passed、29 skipped**，413 秒。建置時是 474 passed、2 errors（`test_p03_nb` RealFit 的 MemoryError，當時 precompute 同時在跑）；這次各步驟依序跑，沒有錯誤。多出的 4 項是這次新增的測試 |
| `C10_SLOW=2 PY -m pytest -v tests/test_t02_dryrun.py`（`C10_DRYRUN_DIR=fix\dryrun`，`fix\dryrun.log`） | **27 passed**，1,226 秒；用的是最終版工具和重算後的 `t3b_null.json`（`t3.json` 的 provenance 記 SHA `a8cf3834…`）。FromValPipeline 的 prereg02 位元組接縫（`test_rank_seam_byte_identical`、`test_opening_seam_bitwise`）通過；混色樹 POS 兩池 OVERLAP、NEG 池 PASS（b_c 0.493，`low_power` 假）；官方形狀樹 T1 留空 PASS／填入 NORMALIZED、POS 兩池 T3b 400/400 判定重疊、NEG 池 T3b（raw 1、canon 0，門檻 12）PASS、`low_power` 假 |
| `PY tools/t02_checks.py precompute --part t3b --rerun --reason ...`（real，`fix\precompute_t3b\run.log`） | pins OK，339 秒。`t3b_null_raw.npz`、`t3b_null_canon.npz` 和第一次逐位元組相同；`t3b_null.json` 只有 `definitions.weights`／`definitions.when`／`rule.pools_left_on_T3` 的文字和 provenance 不同，provenance 的工具 SHA 等於最終版 `tools/t02_checks.py`（`a6fb40f9…`）。rev 5 的 4 個虛無檔 SHA 不變 |
| 凍結檔 | `git status` 沒有列出 `tools/p02_*`、`c10data.py`、`decide.py`、`score.py`、`colab/**`、`results/prereg0[12]/**`、`results/score_ledger.csv`、`docs/VALIDATION.md`、`docs/PREREG_0*`、`data/**`；`test_t02_hygiene` 通過 |
| pyflakes（anaconda 版）檢查改過的 8 個檔 | 只有 `t3b_orient_precompute` 一個沒用到的區域變數（見 5.6 第 5 項） |

### 5.5 新的 pins（修正後）

- `T3B_PINS`（沒變）：參考集 5,767 人、黑棋 360,581 盤、`sha256_sorted_player_ids` `9940feed…`；raw S2_q99 0.11054835109258787、詞彙 146,421、虛無 1,133、holdout 112、虛無超門檻 11、holdout 超門檻 0；canon S2_q99 0.08269886422035472、詞彙 144,097、其餘同 raw。IDF SHA-256：raw `660f7803…`、canon `ab64b00f…`。
- 檔案 SHA-256（位元組）：`t3b_null.json` `a8cf38347b326b024a464b84b193846a9ce1c0205df3be072c95f7b107f5e58c`（第一次是 `67c1ec36…`，只差文字和 provenance）、`t3b_null_raw.npz` `e42318ca89298213175920b0600bc93ec9ed0707d6ea7a063100f7f49c988db7`、`t3b_null_canon.npz` `46589ca3a1d0fadbed9c5831959fbd3f237ade1b18842dbcf340e64c434279fc`（兩個 npz 和第一次相同）。
- rev 5 的 4 個虛無檔（不變）：`t3_null.json` `1950e9a8…`、`t3_null_raw.npz` `4eb9cf05…`、`t3_null_canon.npz` `5312a18c…`、`train_refs.json` `380a0a7f…`。
- 工具：`tools/t02_checks.py` SHA-256（CRLF 折成 LF）`a6fb40f99e7af19bfe4eef3f5cbb1d0049a8f0c14a7cc95cddfac73884f99cc8`；`tools/p03_rank_fuse.py` `77abd7479d26f0f647a258b4a6d5d2e517808b4c187184f14ff599d70b8aca2f`。
- `t3b_null.json` 裡的性質數字（rev 8 條文引用）和第一次相同：假陽性 raw、canon 各 0.97%、raw 或 canon 1.50%、inner 0/112；真陽性 raw 0.3718、canon 0.4144；判定重疊的機率 π＝0／0.2／0.3／0.4／0.5／1.0 為 0／0.0124／0.4484／0.965／0.9996／1.0；原本的 T3 套在全黑 profile：真陽性 0.1253／0.1359，π＝0.5 時 0。

### 5.6 交給主 session

- commit 要一起做：`tools/t02_checks.py`（含 `T3B_PINS`）、T3b 的 3 個虛無檔、PREREG_02 rev 8 條文、TESTDAY。real mode 的 `run` 缺這些檔就不跑。之後不要再改 `tools/t02_checks.py`，否則 `t3b_null.json` 記的工具 SHA 會和 commit 的不同（不影響判定，但要重算才一致）。
- 審查第 3 件的替換文字（和 rev 8 同一個 commit）：
  - `docs/TESTDAY.md` 第 13 行：「下一版是 rev 7；rev 6 已在 `9d64e54` 登記」→「下一版是 rev 9；rev 8 已在 `<rev 8 的 commit>` 登記」。第 239 行同理：「（或下一版修訂 rev 7），不在已登記的 rev 6 後面補字」→「（或下一版修訂 rev 9），不在已登記的 rev 8 後面補字」。
  - `results/test02/design/spec.md` 第 279 行白名單後面加「, `clear_colour_after_num_games_if_sgf_empty` (PREREG_02 rev 8)」；第 589–590 行後面加一行「- rev 8: for i > n, a colour letter B/W/b/w with an empty or whitespace-only `sgf_i` is normalizable (`clear_colour_after_num_games_if_sgf_empty`); `sgf_i` with content after n stays `FAIL_STRUCT`.」，原文保留。
  - `results/test02/design/checks.md` 第 28 行後面加同一句（設計紀錄原文保留）。
  - `docs/DATA.md` 第 107 行「（9/28 起的修訂處理這點）」→「（PREREG_02 rev 8 第 12 節第 1 條處理這點：清掉那格 color，T1 判 NORMALIZED）」。
- 還在的風險：
  1. 0.9 的界線沒有在 0.9–1 之間模擬；它只決定寫「無法判斷」還是「沒有重疊」，不影響任何上傳或模型。
  2. T3b 的偵測力比混色 T3 略低（π＝0.3 時 0.448 對 0.539）；約 20% 以下的重疊仍偵測不到。
  3. T3q（題目 profile，只記錄）仍用混色參考集，在全黑題目上偵測力一樣偏低。
  4. `data/cache/t2_train_index.npz` 舊格式會自動重建，第一次多花約 150 秒。
  5. `tools/t02_checks.py` 在 `t3b_orient_precompute` 有一個沒用到的區域變數 `fit`（pyflakes），建置時留下的，不影響結果；為了不再改工具（改了要重算 T3b 虛無檔），沒動。
