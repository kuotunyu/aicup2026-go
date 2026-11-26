# test04 測試日擴充：整合與審查紀錄

- 規格：`results/test04/design/spec.md`。登記正本是 `docs/PREREG_04_OPPONENT.md` 第 5 節和 rev 2 第 10 節；rev 3 草稿在 `rev3_draft.md`。
- 第 1 節由整合者寫（2026-09-30）。當時分支 `p04-testday`，HEAD `9f63acd`，三位實作者（I1、I2、I3）交來的檔都還沒 commit。
- 第 2 節以後留給三個角度的獨立審查和修正。

## 1. 整合紀錄

**結論**：spec 第 10.6 節交給整合者的步驟 1–4、7、8 全部做完，全部通過。第 5（審查）、6（rev 3 定稿）、9（合併）不在這一輪。

- **完整測試**（修正後的最後一輪）：一般 719 passed、65 skipped；`C10_SLOW=2` 781 passed、3 skipped。3 個 skip 都有正當理由，其中 1 個另外在 L2 之後補跑通過。
- **R20 路徑一個位元組都沒變**：
  - 一次性 clone（main `6ec5e68`、合併結果；修正後的合併結果另建一次）先後用同一個暫存路徑各跑一次 R20 的 from-val 演練。
  - 確定性的輸出彼此相同，也等於 `r20_fake_golden.json`：所有 CSV、正規化後的 `decide_summary.json`、開局陣列。其餘的檔只差在時間和路徑。
  - 對 tag `r20-testday-pre04` 的 39 個指紋檔做 diff：只有 `tools/official_log.py` 多了 576 行，沒有刪任何一行。
  - `official_log` 的 tag 版對新版差異測試通過。
- **04 的 dry run 對得回 prereg04**：數字全部逐位相同，負面演練也全過。
- **notebook 冒煙**：
  - L2b（隨機權重替身）通過。
  - L2（prereg01→02→03 兩臂的真冒煙，接 test02、test04 notebook 和本機工具）也通過：test04 notebook 在 CPU 上和 03 冒煙的 dump 逐位元組相同。
- **整合時改了 2 個真問題**，都只影響冒煙模式（`--smoke`），見 1.3：
  - `opp --smoke` 對 public、private 會拒絕。
  - `apply --smoke` 會因為冒煙權重的 Δκ 判成「04 不用」，走不到交件檔。
  - 這兩個在 L2 才第一次拿真的 notebook 輸出去跑才發現，I1 的 dry run 和單元測試都沒有涵蓋冒煙模式。
- 凍結檔一個位元組都沒動，repo 裡沒有 commit、stash、checkout。沒有開答案檔，沒有跑 `tools/score.py`，沒有用 GPU。

### 1.1 環境

- 本機 Windows 11（`Windows-11-10.0.26200`），全程 CPU。
- `.venv\Scripts\python.exe`：Python 3.12.13、numpy 2.5.3、scipy 1.18.1、zlib 1.3.1、torch 2.14.0+cpu、pytest 9.1.1。
- 每個工作最多 6 個 worker 或執行緒（`OMP_NUM_THREADS=6`、`MKL_NUM_THREADS=6`、`OPENBLAS_NUM_THREADS=6`）。一次只跑一件重的工作。
- 開跑前跑過 `handoff.py`：玉山那條線有一個 Python 在跑，平均約 1 核，其他線閒置。
- 暫存都在 `…\scratchpad\p04build\integrator\`，以下簡寫成 `INT`。
  - `…` = `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-solar\b3b12887-5b96-4ead-ab34-78eb6f69fc09`

### 1.2 步驟與結果

以下 `PY` = `.venv\Scripts\python.exe`，都在 repo 根目錄執行；時間是牆鐘時間（+08:00）。

**步驟 1：完整測試**（`PY -m pytest -q tests`）

- 第一次（整合修正前，05:00–05:07）：718 passed、65 skipped、295 subtests，400.5 秒。
  - 65 個 skip：62 個是要設 `C10_SLOW` 的慢速項目，另外 3 個見步驟 2。步驟 2 在 `C10_SLOW=2` 下只剩那 3 個。
- 整合修正後的最後一輪：719 passed、65 skipped（見步驟 8 後面的「修正後重跑」）。

**步驟 2：`C10_SLOW=2` 完整測試**

- 指令：`C10_SLOW=2 C10_KEEP=1 C10_DRYRUN_DIR=INT\slow2_tree PY -m pytest -q -rs tests`
- 第一次（修正前，05:08–05:35）：**780 passed、3 skipped**、358 subtests，1,648.1 秒（27 分半）。
- 3 個 skip：
  1. `test_p03_nb.py` 的 `SmokeDumps`：要設 `C10_P03_SMOKE_RESULTS`。L2 做完後另外指到 L2 的 03 冒煙輸出跑過，1 passed。
  2. `test_p03_rank_fuse.py` 的真實 apply：03 的正式 `frozen_config.json` 已 commit，設計上不重跑。
  3. `test_t04_dryrun.py` 的「R20 第一次上傳前跑的 notebook」：只能在正式模式用 mock 做（spec 8.3 已這樣寫），由 `test_t04_rank.py` 的 RealMode 測試涵蓋。
- 04 的 dry run（`INT\slow2_tree\t04dry`）的主要數字：
  - `opp --dry-run --workers 6`：89.4 秒。
    - seam_val 的 `o4_seam_val.npz` 是 `30d312c0…`，等於 `val_o4.npz`。
    - 重播失敗：seam_val 0/19,156、public 0/4,787、private 0/4,815。
    - seam_val 的 O4 Δκ 是 `0.1050265798301427`，逐位相同。
  - `apply --dry-run`：5.1 秒。
    - 本機接縫通過：最大差 0.0、規則 B 差 0 題、三個檔都逐位元組相同。
    - seam_val 三個 Δκ（O4、SWA、SWB）＝ `0.1050265798301427`、`0.03631175839656169`、`0.034356617152967346`。
    - 假 public：`0.09297820220912692`、`0.03212728346184879`、`0.02784129798577018`。
    - 假 private：`0.09999333037362917`、`0.04181334094598075`、`0.0363944712752194`。
    - `check_submission --official` 0 個錯誤。
    - 和 R20 規則 B 不同 179 題（等於 prereg02 和 prereg04 在第 1–800 題的差異題數）。
  - R20 部分等於 golden。t04 的每一步前後，test02 的快照都沒變。
  - 負面演練每一項的結束碼都照 spec 8.3（測試逐項斷言；主演練的紀錄 `t04_dryrun_log.json` 有 31 次 `t04_rank.py` 呼叫，其中 21 次是預期的非 0 結束碼）。
    - 封存 seam 失敗的那一次，`attempt.json` 記 `failure_class: notebook_seam`（`classes_found` 還有 `notebook_run_log`），和 spec 8.3 一致。
    - I3 回報的「記成 `notebook_run_log`」是較早版本的行為，現在已經不是。整合時把測試的斷言收緊成只接受 `notebook_seam`（見 1.3 F3）。
- 修正後重跑：781 passed、3 skipped（見步驟 8 後面的「修正後重跑」）。

**步驟 3：R20 逐位元組不變（spec 8.1 第 3 點）**

兩個一次性 clone，都在 `INT\clones\`（`core.autocrlf=false`）。跑的時候 `data\` 用 junction 指到 repo 的 `data\`，只讀。

- `main_6ec5e68`：main 在 `6ec5e68`（detached）。
- `merged`：分支 `9f63acd` 加上 repo 工作樹的 17 個新檔或改動檔，在 clone 裡 commit 成 `3fb46fd`（腳本 `INT\make_merged_clone.py`）。
  - 加入時用 `core.autocrlf=input`，和 repo 自己 commit 時一樣把 CRLF 轉成 LF。受影響的只有 `docs/COLAB_RUN.md`：repo 工作樹是 CRLF。
  - 另外 16 個檔和 repo 工作樹逐位元組相同。

兩邊依序跑（`INT\step3.sh`）。環境變數 `C10_SLOW=1 C10_KEEP=1 C10_DRYRUN_DIR=INT\r20dry`，指令 `PY -m pytest -q tests/test_t02_dryrun.py -k FromValPipeline -p no:cacheprovider`。跑完一邊就把 `INT\r20dry\fromval` 搬到 `INT\step3_out\<clone>_fromval`，再跑另一邊。

| clone | HEAD | 結果 | 時間 |
|---|---|---|---|
| `main_6ec5e68` | `6ec5e68` | 15 passed、12 deselected | 190 秒（05:36:01–05:39:12） |
| `merged` | `3fb46fd` | 15 passed、12 deselected | 192 秒（05:39:12–05:42:24） |

兩個 clone 跑之前、跑之後的 `git status --porcelain` 都是空的。比對（`INT\compare_r20.py` → `INT\step3_out\compare.json`）：

- **兩棵樹都等於 golden**：
  - 7 個 CSV 的 SHA-256：`submission_rank.csv` = `d9b9801b…`；seam 是 `a8370043…`、`863a6258…`。
  - 兩份 `decide_summary.json` 的正規化 SHA（刪掉 `input` 鍵）。
  - `opening_test.npz` 的 9 個陣列（dtype、shape、SHA）。
- **兩棵樹彼此比**：共 128 個檔，82 個逐位元組相同。
  - 所有 CSV 都相同，含競賽一的輸出、R10 備案、`rank_mixed`。
  - 兩份 `decide_summary.json` 整檔也相同：兩邊用的是同一個絕對路徑。
  - 不同的 46 個檔都只差在建立時間、`git_head`、耗時、clone 的絕對路徑，以及這些含時間戳記的檔的 SHA（例如 MANIFEST、t1.json、測試 zip）。
  - 3 個 `opening_test.npz` 只差在 `meta`，陣列都相同。
- **`git diff r20-testday-pre04 -- <39 個指紋檔>`**：
  - repo 工作樹：只有 `tools/official_log.py`，576 行新增、0 行刪除。
  - merged clone 的 commit（`git diff r20-testday-pre04 HEAD`）：同樣結果。
  - 39 個檔都在 `git ls-files` 裡。
- **`official_log` 的差異測試**（spec 8.1 第 4 點），在 merged clone 裡跑：
  - `PY -m pytest -q tests/test_t04_official_log.py -k TagVersusNew`：3 passed、38 subtests，1.9 秒。
  - 同一個 clone 裡，四個 hygiene 檔加上 `test_t02_infra` 的 OfficialLog：49 passed、38 subtests。
- 這個 clone 是在整合修正（1.3）之前建的。修正後另建 `merged2` 重跑，結果見步驟 8 後面的「修正後重跑」。

**步驟 4：notebook 的 CPU 冒煙（spec 8.5）**

L1（`tests/test_t04_nb.py`）已包含在步驟 1、2 裡。

*L2b*（`INT\L2b_chain.sh`，scratch `INT\L2b`，05:43–05:57，共 13.9 分鐘；spec 估 12–15 分鐘）：

| 步驟 | 指令 | 結果 |
|---|---|---|
| 建替身樹 | `PY colab/make_test04_smoke_tree.py --scratch INT\L2b --threads 6` | OK，161 秒；其中 03 參照 dump 124.6 秒 |
| notebook | `PY colab/run_local_smoke.py colab/test04_rank.ipynb --scratch INT\L2b` | OK，378 秒；14 個程式格全部 ok |
| 核對 | `PY colab/make_test04_smoke_tree.py --scratch INT\L2b --check --threads 6` | `ok: true`，295 秒；其中轉接重算 293.4 秒 |

- **接縫**：通過，tol 1e-4，`bitwise_equal` true。
  - 兩臂 C0（s0、swap）最大差都是 0.0；8 個對稱的 logits 差、swap 差、E 差都是 0。
  - d4 和 swap 的檔案 SHA 都等於參照。
- `run_log` 是 `completed`。
- **zip**：`3797adcd…`，348,921 bytes，40 個成員全部在 `sw/` 底下，必要成員 40 個都在。
- seam_val 的 6 個 npz 等於替身參照。
- **轉接**：public、private 用 P03 自己的 `dump_val_p03` 重算，5 個 npz 都逐位元組相同（160／175 盤）。三組的 `questions_path` 都是那一組的題目檔。
- **notebook 各段耗時**：seam_val 123 秒、public 115 秒、private 135 秒，`total_this_run` 375.9 秒。
- **替身**：權重 A `07ce4f96…`、B `e3969d78…`。冒煙測試包 zip `570a1059…`、MANIFEST `886e16d1…`（含建立時間，每次建都不同）。

*L2*（`INT\L2_chain.sh`，scratch `INT\L2`，同一個 scratch 從頭接到尾，05:57–06:47）：

| 步驟 | 指令（`S` = `INT\L2`） | 結果 |
|---|---|---|
| prereg01 | `PY colab/run_local_smoke.py colab/prereg01_rank.ipynb --scratch S --fresh` | OK，377 秒 |
| prereg02 | 同上，`prereg02_rank.ipynb` | OK，746 秒 |
| prereg03 A | 同上，`prereg03_rank_a.ipynb` | OK，523 秒 |
| prereg03 B | 同上，`prereg03_rank_b.ipynb` | OK，600 秒 |
| 冒煙測試包 | `t02_fake_tests.py from-train --shape official --slot-colours filled` → `t02_checks.py t1 --dry-run --only T1.1,T1.2,T1.3`（兩場）→ `t02_test_bundle.py build --kind smoke`，zip 複製成 `S\drive\C10_test_v1.zip` | OK，65 秒；zip `8b67a95f…`、MANIFEST `dffcfb11…` |
| test02 notebook | `PY colab/run_local_smoke.py colab/test02_rank.ipynb --scratch S` | OK，170 秒；接縫和 prereg02_smoke 逐位元組相同 |
| test02 本機 | `t02_rank.py opening --smoke`，再 `apply --smoke`（`--results-dir S\drive\results\test02_smoke --tests-dir S\fake\built\data\test --checks-dir S\fake\results\test02\checks --out-dir S\t02out\rank`） | 兩個都 exit 0，28 秒＋1 秒。開局的 seam 部分和 `opening_val.npz` 逐位元組相同（19,156 盤）。本機接縫記為 FAILED，只記錄：冒煙權重不是 R20 的權重，是預期的 |
| test04 notebook | `PY colab/run_local_smoke.py colab/test04_rank.ipynb --scratch S` | OK，336 秒 |
| t04 本機 | `t04_rank.py unpack --smoke --zip S\drive\results\test04_smoke\test04_rank_results.zip`，再 `opp --smoke --workers 6`、`apply --smoke`、`status --smoke`（DIRS：`--results-dir S\r4 --t02-results-dir S\drive\results\test02_smoke --t02-out-dir S\t02out\rank --tests-dir S\fake\built\data\test --checks-dir S\fake\results\test02\checks`） | 見下面 |

test04 notebook 用的參照是這條鏈 03 冒煙真的產出，不是隨機權重：

- 03 冒煙的 `final.pth`：A `ec62f8ba…`、B `2e88f8c9…`。
- 接縫：通過，tol 1e-4，`bitwise_equal` true。兩臂 C0 最大差 0.0；d4、swap 的檔案 SHA 都等於 03 冒煙的 dump。
- `run_log` 是 `completed`。`test02_record` 讀到真的 test02 冒煙 `config.json`（`run_log_status: completed`），測試包 MANIFEST 和 test02 相同（`dffcfb11…`）。
- zip `2542fd9f…`，355,328 bytes，40 個成員。
- 各組耗時：seam_val 98 秒、public 111 秒、private 124 秒，合計 334.9 秒。

**t04 本機工具在真的 notebook 輸出上**（這是 I1 讀、I2 寫的格式第一次實際接起來）：

- `unpack --smoke`：exit 0，2 秒，40 個檔。印出執行紀錄 `run 1: completed`。
- `opp --smoke`：**第一次 exit 2**，錯誤「question file has 5000 slots, val_keys.npz 160」。修正 F1 後：
  - exit 0，6 秒。
  - 重播失敗 seam_val 0/142、public 0/160、private 0/175。
  - O4 Δκ：public `0.08166747229699296`、private `0.08307640512198267`（冒煙，只記錄）。
  - seam_val 不是逐位元組相同。冒煙只有 142 盤，這是預期的，只記錄。
- `apply --smoke`：**第一次 exit 3**，private 的 SWA Δκ 是 `-0.0001345756232435565`：冒煙權重只訓練 70 步，Δκ 的正負沒有意義。修正 F2 後：
  - exit 0，2 秒，`usable` true。
  - notebook 接縫 true，`same_as_repo_notebook` true。
  - 本機 Q02 接縫只記錄：FAILED，最大差 0.388、規則 B 差 9 題。冒煙權重對上真的 prereg04，是預期的。
  - `smoke_record_only.dkappa_not_positive` = `["private.SWA: -0.0001345756232435565"]`。
  - 交件檔 `4872f24a…`，`check_submission`（冒煙子集的題目檔）0 個錯誤。和 R20 冒煙交件檔的規則 B 不同 21／24 題：兩邊都是冒煙權重，只記錄。
- `status --smoke`：exit 0。有印出 K6 倒數。
- 第一次（修正前）的 `opp/`、`rank/` 輸出留在 `INT\L2\first_attempt_outputs\` 備查。

另外：`C10_P03_SMOKE_RESULTS=INT\L2\drive\results\prereg03_smoke PY -m pytest -q tests/test_p03_nb.py -k SmokeDumps`：1 passed。這是步驟 2 的 skip 1。

**重跑 L2 時**（K5 閘門第 1 輪補，見第 6 節）：開跑前確認可用記憶體 ≥ 8 GB、沒有別的重 CPU 工作；時間抓 50–100 分鐘。某一步沒有 Python traceback 就結束（Git Bash 的 exit 139、Windows 的 0xC0000005）時，把那本 notebook 在 `S\drive\results\<prereg>_smoke\<臂>` 和 `S\work\*pass_ckpt*_<臂>_smoke` 的輸出移開，從那一步重跑（鏈腳本的第一個參數），重跑要記進紀錄。9/30 晚間的 gate 在 prereg03 A 臂遇過一次，重跑後通過、權重相同（6.4 L2）。

**跨次數比對**（K5 閘門第 2 輪補，見第 7 節）：prereg01、prereg02 的 CPU 冒煙權重每次跑都不一樣，所以上面 test02 冒煙輸出、test02 本機接縫、test04 本機 Q02 接縫（0.388／9 題）、test04 冒煙交件檔 `4872f24a…`、「和 R20 規則 B 不同 21 題」都只記錄這一次，**不是參照值**，重跑 L2 時不同不算退步。跨次數要相同的只有 prereg03 兩臂的 `final.pth`（A `ec62f8ba…`、B `2e88f8c9…`）、`opp` 的三個 O4 檔和 public、private 的 O4 Δκ；notebook 接縫（`bitwise_equal`）在同一條鏈內比。三次的 SHA 列在 7.1 第 4 條。

**步驟 7：上傳副本**

- `data/colab_upload/test04/` 裡只有 `test04_rank.ipynb` 一個檔，144,043 bytes。
- 它和 `colab/test04_rank.ipynb` 的 SHA-256 都是 `d6e5dfa94ec111cdc8cb0abfca7d5267373a56116ec406a46279784512389120`。
- notebook 還沒 commit，現在比的是工作樹。`colab/*.ipynb` 是 `-text`，commit 不會改位元組；commit 後照 TESTDAY_04 第 3 步那一行再核對一次。

**步驟 8：計數檔**

- `data/prereg04/counts_fit.npz`：SHA-256 `f6c6debe90639585763b9812c235743a6b7f89d8db1c6422ece7bccd550a08a5`，178,537,934 bytes。
- 它等於 `results/prereg04/opp/count_report.json` 的 `counts_sha256`。
- 備份 `data/prereg04/design_scratch/counts_fit.npz` 的 SHA 也相同。

**修正後重跑**（F1–F3 之後；程式和測試在這之後都沒有再改）

- 步驟 1：`PY -m pytest -q -rs tests` → **719 passed、65 skipped**、295 subtests，375.3 秒（06:48–06:54）。
  - 多出的 1 個是新測試 `test_smoke_records_dkappa_only`。
  - 65 個 skip：62 個要 `C10_SLOW`；另外 3 個是步驟 2 那 3 個。
- 步驟 2：`C10_SLOW=2 C10_KEEP=1 C10_DRYRUN_DIR=INT\slow2_final_tree PY -m pytest -q -rs tests` → **781 passed、3 skipped**、358 subtests，1,631.2 秒（06:54–07:21）。
  - 3 個 skip 和第一次相同。
  - 收緊過的封存斷言（F3）也在這一輪通過。
- 步驟 3：用修正後的工作樹另建 clone `merged2`（`9f63acd` 加上 18 個檔，含本檔的草稿；clone 內 commit `48f117b`）。
  - 用同一個 `C10_DRYRUN_DIR` 跑，15 passed，188 秒（07:22–07:25）。前後 `git status` 都是空的。
  - 和 `main_6ec5e68` 那棵樹比：
    - 128 個檔裡 82 個相同。不同的 46 個和第一次比對完全是同一組檔，都只差在時間、路徑、`git_head`。
    - CSV 全部相同。
    - `decide_summary.json` 整檔相同。
    - golden 的三塊都相同。
  - `git diff r20-testday-pre04 HEAD -- <39 個檔>`：只有 `tools/official_log.py` 576 行新增、0 行刪除。
  - `TagVersusNew`：3 passed。
- L2：F1、F2 改完後，在同一個 scratch 重跑 `opp`、`apply`、`status --smoke`，結果就是上面步驟 4 寫的。notebook 和 L2b 不受這兩個修正影響，沒有重跑。
- repo 的 `git status`：只有本輪的 18 個新檔或改動檔（17 個實作者的檔加上本檔）。凍結檔沒有出現在清單裡。

### 1.3 整合時的修改

只改了新檔，凍結檔都沒動。

- **F1 `tools/t04_rank.py` `cmd_opp`（冒煙模式）**
  - 問題：`--smoke` 時只有 seam_val 把題目檔的 slot 截到冒煙 key 的長度。public、private 拿整份題目檔（各 5,000 盤）去對 test02 冒煙的 key（前 12 題），`po.slots_match_keys` 一定拒絕（exit 2）。所以 spec 8.5 L2 的「`t04_rank.py … --smoke`」原本走不完。
  - 修法：`--smoke` 時三組都截到 key 的長度。冒煙 notebook 本來就只算每組的前幾題，`apply` 的 `fuse_set` 也核對同一件事。截完之後，`slots_match_keys` 仍然逐列核對題號、slot、執色。
  - 正式、彩排、dry run 的行為不變。
- **F2 `tools/t04_rank.py` `opp_decision`、`apply_decision`（冒煙模式）**
  - 問題：`--smoke` 時 6 個 Δκ 的正負照樣判定。冒煙權重只訓練 70 步，每組只有 12 題，正負是雜訊。L2 的 private SWA 是負的，`apply` 就以 exit 3 結束，交件檔、`check_submission` 這一段在冒煙中從來沒被執行到。
  - 修法：`--smoke` 時 Δκ 的正負只記錄。`opp` 的 O4 Δκ 仍寫進報告；`apply` 報告多一個 `smoke_record_only` 欄，列出不大於 0 的值。重播失敗率在冒煙仍然判定。這和 spec 5.3「冒煙：全部只記錄」、test02 的冒煙做法一致。
  - 正式、彩排、dry run 完全不變。新增 `tests/test_t04_rank.py::Rules::test_smoke_records_dkappa_only`：冒煙下只記錄；非冒煙照舊判 3；冒煙的重播失敗照舊擋。
- **F3 `tests/test_t04_dryrun.py`**：封存演練的斷言從「`notebook_run_log` 或 `notebook_seam`」收緊成只接受 `notebook_seam`（spec 8.3）。現在的程式本來就記 `notebook_seam`，所以只是測試變嚴，沒有行為改變。
- 沒有改 notebook、產生器、`official_log.py`、文件。上傳副本不必換。
- 改完後的 SHA-256（LF）：`tools/t04_rank.py` `166e6de5…`、`tests/test_t04_rank.py` `dc6d98e0…`、`tests/test_t04_dryrun.py` `f86eebc8…`。其餘 14 個實作者的檔和交來時相同，例如 notebook `d6e5dfa9…`、`official_log.py` `4affe57f…`、golden `36c8900b…`。

### 1.4 要寫進 rev 3 的偏離（三位實作者和整合者的彙整）

都不改登記的科學內容：來源、參數、組合、門檻、判定規則都照 rev 1、rev 2。

**會影響測試日判定或程序的**（rev 3 要逐條寫清楚）：

1. **`apply` 遇到 `opp` 判本機漂移時給 2（I1）**：spec 3.3 寫一律 3。現在：`opp` 報告是本機漂移（4）時，`apply` 拒絕（2），要先查好再 `opp --rerun`；只有登記的「04 不用」才給 3。這樣本機漂移不會被誤當成登記的判定。
2. **`apply` 在所有模式都拒絕「前面有失敗的執行」（I1、I3）**：spec 3.3 第 3 點寫「（real）」，8.3 寫「apply 拒絕」。照 8.3。
3. **seam_val 上先核對 O4 Δκ 和 `opp` 相同（不同給 4），再判本機接縫（沒過給 1）（I1）**：避免本機問題用掉登記的那一次重跑。spec 沒定先後。
4. **交件檔寫出後 `check_submission` 沒過（I1）**：spec 沒定義。現在給 4，stage `check_submission`，刪掉 `submission_rank.csv`，寫 `attempt_verdict.json`。
5. **`unpack` 多的核對（I1）**：
   - 狀態 completed 但 `seam_check.passed` 不是 true → 1。
   - `config.format` 必須是 `c10-test04 rank notebook config v1`，否則 2（`apply` 也核對）。
   - 成員副檔名只收 `.json .npz .csv .txt .log`。
   - 缺成員的檢查放在紀錄檢查之後，所以缺成員的接縫失敗 zip 仍然給 1。
6. **`archive` 判類別的優先順序（I1）**：`bug_revision` > `before_r20_u1` > `notebook_seam` > `notebook_run_log` > 其餘。找到的類別都記在 `classes_found`。seam_check 沒過、而且 run_log 是 seam_failed 時記 `notebook_seam`（spec 8.3；整合時已確認，測試已收緊）。
7. **`archive` 也能在非正式模式跑（I1、I3）**：`--dry-run` 等模式加 `--results-dir`，zip 搬到 `R4/failed_zips/attempt_N/`，8.3 的封存演練才能全部用 CLI 做。正式模式照 spec 搬到 `data/test04_failed/`。只要 `R4/failed` 裡有封存，`unpack` 在任何模式都做新舊核對和重跑次數核對；「R20 先」仍然只在正式模式。
8. **notebook 的 run_log 狀態（I2）**：
   - 開始時記 `running`；下一次執行時，還是 `running`、沒有 `end` 的舊執行改記 `interrupted`。
   - preflight 停下記 `preflight_failed`。
   - GPU 關卡停下時維持 `running`，之後會被改成 `interrupted`。
   - 結尾格失敗時記 `final_check_failed`，照樣打包後停下。
9. **notebook 比 spec 嚴的檢查（I2）**：
   - 正式執行時，preflight 第 2 點也核對參照標記的 torch（`2.11.0+cu128`）和 cuDNN（`91900`）。
   - 各組格子要求 dump 狀態是 `written`，不接續。
   - 記下 `net_state_sha256`（只記錄）。
10. **`official_log` 比 spec 9.2 多的（I3）**：
    - seam_val 必須 0 盤重播失敗（第 7 節）。
    - 沒有 `attempt.json`，或紀錄沒寫 `counts_toward_rerun: false` 的封存，一律算進重跑次數（保守的一邊，和 I1 的 `is_counted` 相同）。
    - `final` 裡「這一筆還沒上傳」那一行只在 log 有 04 紀錄時印，R20 的輸出因此逐位元組不變。
    - 還沒設最終選擇時，`status` 印「最終選擇要設成 04」，不印違反那一行。
    - 已經帶有核對過的修訂的 R20／R10 最終選擇，不算違反。
    - OK 請求自己算和 R20 的規則 B 差異題數，也記報告裡的值。
    - 新選項 `--not-04-reason`（`prepare`、`final`）：要引用「PREREG_04 rev N」且 N ≥ 4；該列已 commit，而且晚於被推翻的 04 紀錄。
11. **`RANK_COMBOS` 不擴充（I3）**：rev 3 草稿第 14 條已寫，偏離 `results/prereg04/design/plan.md` 第 7 節。

**只是實作說明**（rev 3 的修訂表記一句即可）：

12. **本機重算接縫的來源 notebook（I1）**：`c0_of`、`question_spans` 都從 `colab/test04_rank.ipynb` 取出（兩者在整本 notebook 裡各只有一個定義）。非正式模式下檔案不存在時，改用 `colab/test02_rank.ipynb`（`c0_of` 相同），記在 `source_notebook`；正式模式缺檔就拒絕。
13. **答案檔字樣（I1）**：用 `"ans"+"wer"` 比對整個絕對路徑，範圍比 spec 的 `"ans"+"wers"` 大。
14. **報告欄位（I1）**：
    - `opp_report.json` 多 `exit_code`、`verdict_kind`，重跑規則靠它們判斷。
    - `sets.anomalies` 是有異常的盤數。
    - 失敗率用整數算 `200*failures <= slots`，和 `failures/slots <= 0.005` 等價。
    - `inputs_bound` 的鍵在所有模式都用正式模式的 repo 路徑，另外有 `inputs_bound_paths` 對到實際檔案，共 33 項。
    - `questions_without_parsed_game` 每組列出題號。
15. **冒煙模式（I1＋整合者 F1、F2）**：
    - C5 的權重改用 test02 冒煙 config 記的 SHA，兩臂權重用 sw 的 config。
    - notebook 接縫、本機 Q02 接縫、seam_val 逐位元組核對、6 個 Δκ 的正負都只記錄；重播失敗率照判。
    - `opp` 三組都截到冒煙 key 的長度。
    - spec 8.5 L2 的順序要寫成 `unpack` → `opp` → `apply --smoke`（I1 的待辦 3；spec 原文只寫 `unpack/apply`）。
16. **notebook 的產生器（I2）**：
    - `bfloat16` 只准出現在沿用的 `state_dict_sha256` 裡，只用來判斷 dtype；整本沒有 autocast。spec 4.2 和 10.2 在這點上互相衝突，產生器和測試都用 ast 核對。
    - `ARMS`、`DUMP_NAMES`、`W_LABEL` 改在 preflight 定義，dump_tools 格只 assert 相同。
    - 拿掉 T02 的 `dump_questions`：test04 用不到。
    - `guard_write`、`record_readonly_after` 只改註解，用「去掉註解後 token 相同」核對。
    - diff 檔另附對照 T02 原格的 diff。
    - 產生器多一個 `--upload-copy`。
17. **`config.sets` 欄位的意思（I2）**：
    - `questions` 是標籤，不是絕對路徑；絕對路徑在 `config.dumps.<組>.questions_path`。
    - `rows` 是處理的題數（冒煙 12）；`questions_count` 是題目檔總題數；盤數在 `dumps.<組>.rows`。
18. **`make_test04_smoke_tree.py`（I2）**：
    - build 前先跑 `t02_checks.py t1 --dry-run --only T1.1,T1.2,T1.3`（兩場）。
    - 另寫一份標 `stand_in` 的 test02 冒煙紀錄，讓 preflight 第 5 點在 L2b 也有東西核對。
    - 核對功能做成同一個檔的 `--check`。
19. **golden 的算法（I3）**：`opening_test.npz` 逐陣列比 `numpy.ascontiguousarray(a).tobytes()` 的 SHA-256，另記 dtype、shape。npz 成員的 zip 時間固定，所以整檔不同只會來自 `meta`（建立時間、絕對路徑）。
    - golden 的指令多了 `-p no:cacheprovider` 和執行緒上限。
    - 步驟 3 另外在三個 clone 各跑一次，都等於 golden，所以確定性也驗過了。
20. **spec 8.1 第 1 點的描述過時（I2、I3、整合者）**：它寫 `results/test04/design/` 還沒追蹤，實際上已在 `9f63acd` commit。所以 golden 跑之前、之後的 `git status` 都是全空。
21. **時間（整合者）**：spec 8.5 的推估要更新。
    - L2b 實測 13.9 分鐘（I2 在有其他工作同時跑時量到 18.8 分鐘）。
    - L2 的 notebook 鏈約 47 分鐘（spec 估 65 分鐘），連本機工具約 50 分鐘。
    - `C10_SLOW=2` 完整測試約 27 分半。

### 1.5 還沒驗到的地方與風險

1. **正式模式沒有用 commit 過的程式端到端跑過**（I1 的待辦 2）。
   - 現在只有 mock 的單元測試涵蓋：R20 的 U1、K6、notebook 身分、新舊核對、`archive` 各類別、`require_committed`、`official_log prepare` 認 04。
   - 建議：rev 3 和程式 commit 之後、合併之前，照 test02 `build_review` 第 4 節的做法，在一次性 clone 裡用 from-val 的假樹實跑一次正式模式：`opp` → `unpack` → `apply` → `official_log prepare`／`uploaded`／`final`／`status`。
   - 注意：那個 clone 的 `data` 不能整個是指到 repo 的 junction，因為正式模式會寫 `data/test`、`data/test04_failed`。要把 `data` 建成真的資料夾，裡面只有 `local`、`training`、`prereg04`、`colab_bundle` 各自用 junction 指過去。
2. **test04 notebook 從沒在真 GPU 上跑過**。CPU 冒煙在 L2b、L2 都逐位元組相同；G4、A100 上的 C0 差距要靠 spec 8.6 的 Colab 彩排（要使用者同意，建議 10/15–10/22）。
3. **`opp` 判定的先後**（I1 的待辦 1，留給審查決定）。
   - spec 3.2 明寫「先看 3 再看 4」，實作照做。
   - 風險：seam_val 陣列不同（表示本機程式有問題），同時 public 或 private 的 O4 Δκ ≤ 0，會判成登記的 3；之後只能靠 `--bug-revision`（指出程式錯誤的修訂）翻盤。
   - 這種情形 seam_val 不同本身就是程式錯誤的證據，所以翻盤的路是有的。改成「陣列不同時 4 優先」也說得通。這是規則的選擇，整合者沒有改。
4. **notebook 接縫的 tol 規則兩邊寫法不同**（只在理論上有差）：notebook 取兩臂 tol 的較大者；本機重算要求兩臂 tol 相同，不同就判不一致（1）。兩臂的 03 參照 dump 都在同一型 GPU（`NVIDIA RTX PRO 6000 Blackwell Server Edition`、torch `2.11.0+cu128`、cuDNN 91900），所以實際上不會發生。
5. **L2 的 03 冒煙 `final.pth` 和 9/27 的紀錄不同**（A `ec62f8ba…` 對 `8a026f9e…`，B `2e88f8c9…` 對 `17728538…`）。原因沒查。test04 的接縫比的是同一條鏈的 dump，逐位元組相同，所以不影響本次的結論。
6. **步驟 7 要在 commit 後再核對一次**：notebook 還沒 commit；重建 notebook 時要用 `--upload-copy` 重寫上傳副本。
7. **主流程還欠的**：`CLAUDE.md` 第 80 行改指 `docs/TESTDAY_04.md`，並寫「U5 對競賽二由 5.3 取代」；PORTFOLIO_MAP 的 C10 那一行；rev 3 定稿（依本節 1.4 更新 `rev3_draft.md`）；`r20_fake_golden.json` 和本檔 commit。
8. **clone 的 `data` junction 已拆掉**：三個 clone 的 `data\` 原本是指到 repo `data\` 的 junction，跑完後已用 `cmd /c rmdir` 只拆掉連結（repo 的 `data\` 核對過還在，`counts_fit.npz` 大小不變）。審查時要在 clone 裡重跑，就再建 junction，跑完同樣先拆掉；**不能**對還掛著 junction 的 clone 遞迴刪除，否則會刪到真的資料。

### 1.6 暫存

`INT\`（合計約 12 GB），都可重建：

- `clones\main_6ec5e68`、`clones\merged`、`clones\merged2`：各約 0.5 GB；`data` junction 已拆（1.5 第 8 點）。`clones\merged_record.json`、`merged2_record.json` 記了每個複製檔的 SHA。
- `step3_out\`：三棵 R20 樹、`compare.json`、`compare_main_vs_merged2.json`、pytest log、`times.txt`，約 1 GB。
- `slow2_tree\`、`slow2_final_tree\`：`C10_SLOW=2` 的樹，各約 1.8 GB。
- `L2b\`（1.5 GB）、`L2\`（4.4 GB）：冒煙樹。`L2\first_attempt_outputs\` 是修正前的 `opp`、`rank` 輸出。
- log：`step1_pytest.log`、`step2_pytest.log`、`final1_pytest.log`、`final2_pytest.log`、`L2b_logs\`、`L2_logs\`，以及各個 `*times.txt`。
- 腳本：`compare_r20.py`、`make_merged_clone.py`、`step3.sh`、`L2b_chain.sh`、`L2_chain.sh`。
- 合併後可以刪整個 `INT\`（junction 已拆；刪之前再確認一次沒有 junction）。I1、I2、I3 各自的 `p04build\*` 也可以刪；golden 的來源樹 `p04build\golden_tree` 留到合併後。

## 2. review round 1 and fixes

- 修正者（round 1），2026-09-30 上午。分支 `p04-testday`，HEAD `9f63acd`。本輪的修改都沒有 commit：repo 裡沒有 commit、stash、checkout。
- 三個角度的審查（correctness、registration、user-colab）共 19 項：6 項 major，13 項 minor，沒有 blocker。其中 2 組是同一個問題從兩個角度看，合併後是 17 條。
- 每一條都先對照程式核實，**全部屬實**，全部修正（fixed）。沒有駁回，也沒有延後。
- 凍結檔都沒動：
  - `tools/official_log.py` 只改 04 自己加的那一段。對 tag `r20-testday-pre04` 仍然只有加行：598 行新增、0 行刪除。
  - `rev3_draft.md`、`spec.md`、`understand_*.md` 沒改。要寫進 rev 3 的條目在 2.3。
- 沒有開答案檔，沒有跑 `tools/score.py`，沒有用 GPU；每個工作最多 6 個執行緒。
- 暫存都在 `…\scratchpad\p04build\fixer1\`，以下簡寫成 `FX`（`…` 同 1.1）。

### 2.1 每一條的處理

1. **`archive` 核對不了 R20 U1 時，把這次記成 `before_r20_u1`，繞過「只准重跑一次」**（correctness major、registration major）→ **fixed**
   - `cmd_archive` 呼叫 `r20_first_upload` 拋錯時，直接拒絕，結束碼 2，什麼都不搬。訊息 `MSG_R20_UNCHECKED`：先 commit `results/official_scores.json` 和 test02 的 `apply_test_report.json`，再 archive。
   - 只有 R20 U1 核對成功、而且這次執行真的在 U1 之前開始（或有完成標記早於 U1）時，才記 `before_r20_u1`。
   - 「已有 1 個占次數的封存 → 拒絕」不再因為 `before_r20_u1` 而跳過。
   - 測試 `RealMode.test_archive_refuses_while_r20_u1_cannot_be_checked`：
     - log 不乾淨 → 2；`sw/`、zip 都留在原處，沒有產生 `failed/`。
     - log 乾淨 → 記成 `apply_exit_1`，占次數。
     - 接著第二次接縫失敗，在三種情況都被拒，沒有 `attempt_2`：log 乾淨、log 不乾淨、U1 記在這次執行之後。
2. **`unpack` 先做接縫核對、後做新舊核對，所以已封存的接縫失敗 zip 再解一次會得到 1**（correctness major）→ **fixed**
   - 這兩項移到所有紀錄核對之前：
     - 新舊核對：zip 的 SHA、開始時間、seam_val 完成標記的時間。
     - 占次數的封存已有 2 個。
   - `tests/test_t04_rank.py` 原本寫死「還是 1」的那一行，改成期待 2 和「不是重算的結果」，並核對沒有解開任何檔。
   - `tests/test_t04_dryrun.py::test_seam_failure_is_archived_as_a_counted_attempt` 加上：同一個 zip 再解一次 → 2，沒有 `sw/`、也沒有 `unpack_sw.json`。
3. **未預期的例外以 Python 預設的 1 結束，和「接縫類失敗」撞碼**（correctness minor）→ **fixed**
   - `main` 最後接住所有 `Exception`：先印 traceback，再印「unexpected internal error … This is not a seam failure: do not archive; stop and check the local tool」，結束碼 2。
   - 沒有另設新的結束碼。TESTDAY_04 結束碼表的 2 那一列補一句說明。
   - `unpack` 在讀任何紀錄之前，先用 `testzip()` 查 CRC；壞了就給 2（「damaged member」）。
   - `zjson` 也接住 `BadZipFile`、`zlib.error`。
   - 解壓到一半出錯時，刪掉已經解出的檔。
   - 審查者的 `exp_exit1.py` 重跑：exit 2，訊息是 `damaged member 'sw/run_log.json'`。
   - 測試：`test_damaged_member_is_a_refusal`、`test_unexpected_error_is_exit_2`。
4. **seam_val 漂移時，`opp` 仍把 public／private 的 O4 Δκ ≤ 0 判成登記的 3**（correctness minor、registration minor）→ **fixed**
   - `opp_decision` 改成：只有 seam_val 的檔案逐位元組相同、而且 seam_val 的 Δκ 等於登記值時，O4 Δκ ≤ 0 才列為登記的理由。
   - 漂移時給結束碼 4。這時的 Δκ 只記在報告的新欄位 `o4_dkappa_not_judged_while_drift`，stderr 也會印。
   - 重播失敗 > 0.5% 照舊是 3。
   - 整合者 1.5 第 3 點留給審查決定的規則，就此定案；rev 3 第 7 條要改寫（2.3 第 4 條）。
   - `Rules.test_opp_decision` 加的案例：
     - 三種漂移（檔案不同、Δκ 不同、兩者都不同）× public Δκ < 0 → 4，沒有登記理由。
     - 漂移加上 public 重播失敗 26/5000 → 3，登記理由裡沒有 Δκ。
5. **`archive` 只記 R4 裡 zip 的 SHA；`unpack --zip` 指到別處時，新舊核對少一項**（correctness minor）→ **fixed**
   - `archive` 另讀 `unpack_sw.json` 的 `zip.sha256`。它和 R4 裡的 zip 都不同時，以 `unpacked:<檔名>` 併進 `attempt.json` 的 `zips`，另外記 `unpacked_zip`。
   - `freshness` 和 `apply` 的 `check_unpack_record` 本來就讀 `zips` 的全部值，所以不用改。
   - 測試 `test_archive_records_the_unpacked_zip`：zip 放在 R4 外面解開 → `apply` 結束碼 1 → 封存（`zips` 只有 `unpacked:test04_rank_results.zip`）→ 同一個 zip 再解 → 2。
6. **登記內的 R20 側變動之後，TESTDAY_04 第 1 步的防誤改閘門永遠過不了**（registration major）→ **fixed**
   - 改的是 `test_t04_hygiene.py` 和文件。`test_p04_hygiene.py` 在 main 上，不改。
   - `test_t04_hygiene.py` 的 `SameAsBranchPoint`：
     - 只有 `tools/score.py`、`make_val.py`、`colab/build_prereg0*_*.py`、`colab/prereg0*_*.ipynb` 仍要求和 6ec5e68 逐位元組相同。
     - 打分紀錄帳、`docs/PREREG_01–03*.md`、`docs/COLAB_RUN.md` 改成「6ec5e68 的每一行照順序還在」。新測試是 `test_ledger_docs_colab_run_keep_every_line`，用的是和 PREREG_04 同一個子序列規則。
     - 原本的「COLAB_RUN 只在最後加一節」，改成「恰好有一節 `# 測試集 04（test04）`，而且 6ec5e68 沒有這一節」。
   - TESTDAY_04 第 1 步：
     - 多一行指令，列出 39 個指紋檔裡內容變了的檔。Git Bash 和 PowerShell 各跑過一次，現在都印 `['tools/official_log.py']`。
     - 閘門的例外寫明：失敗的只有指紋核對，而且多出來的檔正好是一份已 commit 的 PREREG_02 修訂列出的檔（rev 9 起，或 rev 7 第 2 條的 A100 改版）。用 `git diff --stat r20-testday-pre04 -- <檔>` 核對，記進 RESULTS。
     - 只動到檔名含 `player` 的檔時，04 照常做。
     - 動到 R20 路徑的檔時，照 rev 3 第 19 條停下，直到 PREREG_04 修訂 commit。
   - 原本的「D5」一節擴大成「R20 側的已登記變動」，涵蓋 D5、PREREG_02 rev 9 起、rev 7 第 2 條的 A100 改版、打分紀錄帳新增列，也寫明 R20 修 bug 重跑 test02 之後 04 怎麼辦（第 11 條）。
   - 審查者的 clone 實驗沒有重做。新規則是子序列規則，「PREREG_02 只加一列」一定通過；剩下的 `test_p04_hygiene` 失敗，正是新閘門允許、並要求逐檔核對的情況。
7. **正式模式下 `opp`／`apply` 的「R20 U1 先」前提，沒有任何測試走到**（registration major）→ **fixed**
   - 新測試 `RealMode.test_opp_and_apply_need_r20_first_upload`。
   - 只有 test02 自己的核對換成 dry-run 版：`check_test02` 包一層，因為它的正式模式要讀 repo 裡真的 `results/test02`。其餘都是正式模式的程式：frozen config 的 git 核對、凍結工具、`.venv`、K6、`require_committed`（mock）。
   - log 沒有任何上傳紀錄，或只有 `prepared` → `opp`、`apply` 都是 2，訊息含 `MSG_NO_R20_U1`，沒有寫出 `opp/`、`rank/`。
   - 有 `uploaded` 的紀錄時：
     - `opp` 給 0，報告記 `test02.r20_first_upload_id` 1、`checks.r20_first_upload.id` 1。
     - `apply` 通過這一關，停在下一個前提（「run t04_rank.py unpack first」）。
   - 突變核對（`FX\mut_t04.py`，在記憶體裡換掉模組函式）：把 `r20_first_upload` 換成永遠成立（等同突變 T21），這個測試失敗。
8. **`status` 在「舊修訂＋新的合格 04」時不報違反**（registration minor）→ **fixed**
   - `t04_status` 改成：最終選擇是 R20／R10 時，要同時成立兩件事才不算違反：
     - 現在每一筆符合條件的 04，都在那份修訂記的 `eligible_04_entries` 裡；
     - 修訂的 commit 時間晚於那一筆的 `prepared_at`。
   - 違反那一行改成「沒有引用修訂，或引用的修訂沒有涵蓋它」；TESTDAY_04 第 7 步跟著改。
   - 測試 `T04Status.test_revision_does_not_cover_a_later_04`，照審查者的情境：
     - 設了 rev 4 → 不違反。
     - rev 4 之後才 prepare 的 #3 → 違反，而且 `final` 引用同一份修訂時被拒。
   - 突變核對（`FX\mut_ol.py`）：把 `t04_status` 換回舊寫法，這個測試失敗。
9. **tag 版對新版的差異測試沒有 R10 流程**（registration minor）→ **fixed**
   - 新測試 `TagVersusNew.test_r10_flow_identical`，用 T1 FAIL_OPENING、`fallback_used` true 的 R10 檔。
   - 流程依序是：
     1. prepare（U1）；
     2. 一個寫 R20 的 `--method`，要被拒；
     3. uploaded → public；
     4. 競賽一跑一輪；
     5. 再一次 prepare → uploaded → public；
     6. final；
     7. status 兩次。
   - 旁邊有 04 的樹、沒有 04 的樹，各跑一次。
   - 結果：tag 版和新版的 log 位元組、stdout、回傳值都相同，status 只多 K6 那一行。兩筆 R10 紀錄的方法都是 `FALLBACK_METHOD`。
   - 突變核對：在模組上把 `FALLBACK_METHOD` 換成另一個字串（等同審查者的突變 H）。新測試失敗；原本的 R20 測試仍然通過，證實原本抓不到。
10. **「缺 `attempt.json` 或缺 `counts_toward_rerun`，就算占次數」沒有測試**（registration minor）→ **fixed**
    - 新測試 `UnpackDrills.test_archive_without_a_record_counts`，兩種情況：`failed/attempt_1/` 沒有 `attempt.json`；或 `attempt.json` 沒有 `counts_toward_rerun`。
    - 兩種情況下，第二次接縫類封存都被拒（「登記只准重跑一次」），`unpack` 的 `attempt_number` 都是 2。
    - 突變核對：把 `is_counted` 換成審查者的 T12，這個測試失敗。
11. **「輸入已變更，重跑 apply」的提示，照做一定被拒**（registration minor）→ **fixed**（二選一選了改提示，不另開重做的路）
    - `official_log.py` 04 段的提示改用常數 `T04_REDO_HINT`：「輸入已變更：這筆 04 不再符合 PREREG_04 5.3，最終 R20（docs/TESTDAY_04.md 第 7 步）；要重做 04 須先寫並 commit PREREG_04 修訂」。
    - 還沒 commit 的情況另寫：「先 commit；內容若已改變：…」。
    - TESTDAY_04「R20 側的已登記變動」也寫了同一件事。rev 3 第 20 條要補一句（2.3 第 6 條）。
12. **「登記的重跑只在 G4」沒有任何地方強制**（user-colab major）→ **fixed**，照審查建議做了三處。
    - **notebook（產生器）**：
      - preflight 算完 `PRIOR_SW`、`ATTEMPT` 之後檢查：`RESULT_ROOT` 底下有任何 `sw_failed_*` 資料夾，而這次的 GPU 不是 G4，就 `preflight_stop`。這時還沒開始算，不算一次執行。
      - 每次封存都要使用者把資料夾改成 `sw_failed_*`，所以這條也涵蓋重跑中途斷線、`apply_exit_1` 的封存。
      - 判斷寫成可以單獨測試的函式 `rerun_gpu_problem`。SMOKE 不查。
      - GPU 關卡的紅字加上「Claude 說這次是重跑時只能選 G4」。第一格的說明和 CHANGE_ROWS 也跟著改。
    - **`unpack`**：已有 1 個占次數的封存時（冒煙以外），zip 裡每個完成標記的 `gpu_name` 都要是 G4，否則給 2（`MSG_RERUN_G4`）。這一項排在紀錄核對之前；`unpack_sw.json` 記 `rerun_gpu_check`。
    - **`official_log`**：報告的 `attempt.number` ≥ 2 時，`notebook.gpu` 和 `notebook.env` 裡每個標記都要是 G4（`T04_RERUN_GPU`）。否則 prepare 拒絕，這筆紀錄也不符合 5.3。
    - 測試：
      - `test_t04_nb.py::Preflight::test_rerun_only_on_g4`
      - `test_t04_rank.py::UnpackDrills::test_rerun_must_be_on_g4`
      - `test_t04_official_log.py::T04Prepare::test_rerun_attempt_must_be_on_g4`
    - 突變核對：把 `rerun_gpu_problems` 換成永遠沒問題，`unpack` 的測試失敗。
    - 比登記嚴的地方（2.3 第 3 條揭露）：notebook 看資料夾名稱判斷，所以 `before_r20_u1`、`bug_revision` 的封存之後也只收 G4。TESTDAY_04 本來就寫這兩種照「重跑一次」的做法、只選 G4，所以文件和工具一致。
13. **`unpack --dry-run --allow-failed` 對正式的 zip 不能用**（user-colab minor）→ **fixed**
    - `--allow-failed` 本來就只准用在 dry-run、repo 外的資料夾。現在連模式旗標不符也改成警告。
    - 測試 `test_allow_failed_inspects_a_real_zip`：
      - `rehearsal false`、`kind official`、接縫失敗的 zip → 0，記兩個警告。
      - 不加 `--allow-failed` 時：接縫失敗仍是 1；只有旗標不符仍是 2。
    - TESTDAY_04 那一行補一句說明。
14. **第二次彩排前沒有清 Drive**（user-colab minor）→ **fixed**（文件）
    - TESTDAY_04 彩排一節加第 5 點，COLAB_RUN 彩排小節加第一點：第二次彩排前，一則訊息請使用者把 Drive `C10/results` 裡的 `test02_rehearsal`、`test04_rehearsal` 改名成 `…_1`，改名前後各截一張圖，Claude 看過才給步驟。
    - 本機的 `results/test04_rehearsal/` 先移走；收拾時改用 `old_rehearsal_2`。
15. **競賽一在等 G4 時不准跑 test04，會賠掉 04**（user-colab minor）→ **fixed**（文件和 notebook 第一格）
    - 規則改成：只有開著的執行階段才擋，只是在等 G4 不算。
    - G4 難拿時，先跑 test04 的第一次（可以用 A100）。test04 跑的時候，不要同時去試競賽一。
    - 改了 TESTDAY_04 第 3 步第 3 點、COLAB_RUN「開始前」第 1 點、產生器第一格的第 2 點。
16. **斷線時先改名、後刪執行階段**（user-colab minor）→ **fixed**（文件）
    - COLAB_RUN 的 D1–D4 和 TESTDAY_04「斷線」都改成這個順序：
      1. 截圖回報；
      2. 先刪執行階段，等 1–2 分鐘；
      3. 改名，前後截圖，等 Claude；
      4. 開新的執行階段，整份重跑。
17. **第 3 步把整節交給使用者，但裡面第 13 步要刪「下載」裡的檔**（user-colab minor）→ **fixed**（二選一選了拆成另一則訊息）
    - TESTDAY_04 第 3 步加第 4 點：Claude 先另發一則訊息，請使用者刪「下載」裡舊的 zip，等回報才給 notebook 步驟。
    - COLAB_RUN 第 13 步改成只看不刪：還有舊 zip 就停下截圖。

**改了哪些檔**（都不是凍結檔；SHA-256 都是 CRLF 轉成 LF 後算的前 12 碼）

| 檔 | SHA-256 |
|---|---|
| `tools/t04_rank.py` | `3e16744a4c8a` |
| `tools/official_log.py`（只改 04 段） | `d2524476d1ca` |
| `colab/build_test04_rank_nb.py` | `c3417666eefd` |
| `colab/test04_rank.ipynb`（重建） | `dfde753b093a` |
| `results/test04/notebook_diff_rank.txt`（重建） | `a854f165e140` |
| `data/colab_upload/test04/test04_rank.ipynb`（用 `--upload-copy` 重寫；資料夾裡只有這一個檔） | `dfde753b093a` |
| `tests/test_t04_rank.py` | `fbc1602eafd5` |
| `tests/test_t04_official_log.py` | `3bc7269812bc` |
| `tests/test_t04_nb.py` | `0c66c5f5a0b7` |
| `tests/test_t04_hygiene.py` | `f48b06a31fb5` |
| `tests/test_t04_dryrun.py` | `bbd8c7f999b7` |
| `docs/TESTDAY_04.md` | `a63753ad62a5` |
| `docs/COLAB_RUN.md`（只改「測試集 04」一節，對 HEAD 仍然只有加行） | `2967394d2453` |
| 本檔 | — |

- notebook 完整的 SHA-256 是 `dfde753b093abfad1a41a6893842c9b14d5bbcace212e435b9cb5c3a48400294`，照 notebook 自己的規則算的 `notebook_sha256` 是 `a27ef9c8…`。
- 1.2 步驟 7 記的舊值 `d6e5dfa9…` 已經不適用。commit 後照 TESTDAY_04 第 3 步那一行再核對一次。

### 2.2 驗證

以下 `PY` = `.venv\Scripts\python.exe`，都在 repo 根目錄執行；執行緒上限 6。開跑前看過：玉山那條線有一個 Python 在跑，其他線閒置。

**完整測試**（`PY -m pytest -q -rs tests`，08:24–08:31）

- 732 passed、65 skipped、342 subtests，401.3 秒。
- 比整合後的 719 多 13 個，就是本輪新加的測試。
- 65 個 skip 和 1.2 相同：62 個要設 `C10_SLOW`，另外 3 個見 1.2。
- 唯一的 warning 是 `test_p03_nb` 的 torch `pin_memory` 提示，和本輪無關。

**`C10_SLOW=2`**（08:35–08:57）

- 指令：`C10_SLOW=2 C10_KEEP=1 C10_DRYRUN_DIR=FX\slow2_tree PY -m pytest -q -rs`，範圍是 `tests/test_t04_*.py` 加 `tests/test_t02_dryrun.py`。
- 240 passed、1 skipped、340 subtests，1,352.8 秒。
- 1 個 skip 是 `test_t04_dryrun.py` 的「R20 第一次上傳前跑的 notebook」：只能在正式模式用 mock 做，已由 `RealMode` 涵蓋。
- 04 的 dry run、負面演練（含第 2 條新加的「封存的接縫失敗 zip 再解 → 2」）、R20 的 from-val 和 from-train 演練全部通過。

**突變核對**（在記憶體裡換掉模組的函式或常數，不寫 repo）

- `FX\mut_t04.py` 三項，每一項都讓對應的新測試失敗；不換時都通過：
  - T21：R20 U1 永遠成立。
  - T12：缺紀錄的封存不算占次數。
  - 拿掉 G4 核對。
- `FX\mut_ol.py` 兩項：
  - 舊的 `t04_status`：新測試失敗。
  - `FALLBACK_METHOD` 被重新定義：新的 R10 測試失敗，舊的 R20 測試仍然通過。
- 審查者的 `exp_exit1.py` 重跑：exit 2，印 damaged member。

**R20 逐位元組不變**（spec 8.1 第 3 點；因為 `official_log.py` 改了才重做）

- 一次性 clone `FX\clones\merged3`：`9f63acd` 加上 repo 工作樹的 18 個檔（`INT\make_merged_clone.py`），clone 內 commit 是 `e45d7604`。
  - 建 clone 之後，TESTDAY_04 第 3 步的標題又改了一行字。文件不影響 R20 的演練，所以沒有重建。
  - `data\` 用 junction 指到 repo，只讀；跑完已經用 `cmd /c rmdir` 拆掉。之後核對過：repo 的 `counts_fit.npz` 還在，大小不變。
  - 腳本是 `FX\step3.sh`。
- `C10_DRYRUN_DIR` 用 1.2 的同一個路徑（`INT\r20dry`），跑完搬到 `FX\step3_out\merged3_fromval`。
- `PY -m pytest -q tests/test_t02_dryrun.py -k FromValPipeline -p no:cacheprovider`：15 passed，188 秒。跑之前、之後的 `git status` 都是空的。
- 和 1.2 的 `main_6ec5e68` 樹比（`INT\compare_r20.py` → `FX\step3_out\compare_main_vs_merged3.json`）：
  - 兩棵樹都等於 golden：7 個 CSV、正規化後的 `decide_summary.json`、`opening_test.npz` 的陣列。
  - 128 個檔裡 82 個逐位元組相同，兩份 `decide_summary.json` 整檔也相同。
  - 不同的 46 個檔，和 1.2 的 `merged2` 比對是同一組檔，只差在時間、耗時、路徑、`git_head`，以及含時間戳記的檔的 SHA。
- 在同一個 clone 裡另外跑：
  - `git diff --numstat r20-testday-pre04 HEAD -- <39 個指紋檔>`：只有 `tools/official_log.py`，598 行新增、0 行刪除。
  - `TagVersusNew`：4 passed、66 subtests，含新的 R10 流程。
  - 三個 hygiene 檔（p04、t02、t04）：39 passed。

**notebook 冒煙 L2b**（因為產生器改了才重做；腳本 `FX\L2b_chain.sh`，scratch `FX\L2b`）

09:01–09:11，共 9.7 分鐘。

| 步驟 | 指令 | 結果 |
|---|---|---|
| 建替身樹 | `PY colab/make_test04_smoke_tree.py --scratch FX\L2b --threads 6` | OK，113 秒 |
| notebook | `PY colab/run_local_smoke.py colab/test04_rank.ipynb --scratch FX\L2b` | OK，279 秒；14 個程式格全部 ok |
| 核對 | `PY colab/make_test04_smoke_tree.py --scratch FX\L2b --check --threads 6` | `ok: true`，`problems` 是空的，190 秒 |

- 接縫：通過，tol 1e-4，`bitwise_equal` true，`keys_equal`、`weights_ok`、`env_same` 都是 true。
- `run_log` 是 `completed`。
- zip：`6d1be45b…`，348,987 bytes，40 個成員全部在 `sw/` 底下，必要成員 40 個都在。
- 轉接：public、private 用 P03 自己的 `dump_val_p03` 重算，5 個 npz 都逐位元組相同（private 175 盤）。
- 新的重跑 G4 核對在 SMOKE 下照設計不執行；它的判斷由 `test_t04_nb.py::Preflight::test_rerun_only_on_g4` 涵蓋。

### 2.3 要寫進 rev 3 的條目

本輪新增的條目。都不改來源、參數、組合、門檻；貼法照 `rev3_draft.md` 開頭的規則，由主 session 在 Rev3 階段併進草稿。

1. **第 9 條（封存）**：
   - `archive` 核對 R20 第一次上傳時失敗（例如 log 或 test02 報告沒 commit）→ 拒絕封存，不當成「R20 上傳前跑的」。
   - 只有真的在 U1 之前開始的執行，才記成不占次數的 `before_r20_u1`。
   - 已有 1 個占次數的封存時，接縫類失敗一律拒絕封存；`before_r20_u1` 也不例外。
2. **第 9 條（unpack 的新舊核對）**：
   - 新舊核對排在所有紀錄核對之前：zip 的 SHA、執行開始時間、seam_val 完成標記的時間。所以已封存的 zip（包括接縫失敗的那一份）再解一次一律是「不是重算的結果」（2），不是 1。
   - 封存時另記 `unpack` 讀的那一份 zip 的 SHA，不管它放在哪裡。
3. **第 4、9 條（重跑只在 G4，由工具強制）**：
   - notebook：有 `sw_failed_*` 資料夾時，preflight 只收 G4。這時還沒開始算，不算一次執行。
   - `unpack`：有 1 個占次數的封存時，15 個完成標記都要是 G4，否則 2。
   - `official_log`：attempt ≥ 2 的報告，notebook 的 GPU 和每個標記都要是 G4。
   - **揭露**：notebook 看資料夾名稱，所以 `before_r20_u1`、`bug_revision` 的封存之後也只收 G4，比第 9 條字面嚴。TESTDAY_04 對這兩種本來就照「重跑一次」只選 G4。
4. **第 7 條（取代「`opp` 算出 O4 Δκ ≤ 0 就是登記的判定」的前提）**：
   - `opp` 的 public／private O4 Δκ ≤ 0（或 NaN、沒有子題），只在 seam_val 的 O4 檔案逐位元組相同、而且 seam_val 的 Δκ 等於登記值時，才是登記的「04 不用」（3）。
   - seam_val 漂移時，Δκ 只記錄（`o4_dkappa_not_judged_while_drift`），結束碼 4。查好本機後重跑 `opp` 才判定。
   - 重播失敗 > 0.5% 照舊是 3。
   - 這一條也回答了整合者 1.5 第 3 點。
5. **第 19 條擴大**：範圍從 D5 擴大到「任何依已 commit 的 PREREG_02 修訂而改到的 R20 側檔案」，包括 D5、rev 9 起、rev 7 第 2 條競賽一改 A100，也包括打分紀錄帳新增列。
   - TESTDAY_04 第 1 步的閘門：失敗的只有 `test_p04_hygiene` 的指紋核對，而且多出來的檔正好是那份修訂列出的檔。逐檔記進 RESULTS。
   - 只改競賽一的檔（檔名含 `player`）：不需要 PREREG_04 修訂，04 照常做。
   - 改到 R20 路徑的檔：照原第 19 條，04 停下直到 PREREG_04 修訂。原文「直到修訂 commit 就恢復」要改成照這條閘門判定，不是「全過」。
   - `test_t04_hygiene.py` 對 PREREG_01–03、打分紀錄帳、`docs/COLAB_RUN.md`，改成「6ec5e68 的每一行照順序還在」。
6. **第 20 條補一句**：
   - R20 依 PREREG_02 修 bug、重跑 test02 之後，已做的 04 因為輸入變了，不再符合 5.3，最終 R20。
   - `opp`／`apply` 的重跑只接受 PREREG_04 rev ≥ 4，由那份修訂自己說明為什麼可以重做；不能直接引用 PREREG_02 的修訂。
   - `official_log` 的提示已照這一條改寫。
7. **第 12 條（`status`）**：R20／R10 為最終、而且帶著修訂時，`status` 會重新核對：
   - 現在每一筆符合條件的 04，都要在那份修訂涵蓋的清單裡；
   - 修訂的 commit 要晚於那一筆的 prepare。
   - 不成立就印「違反 5.3」。這和 `final` 用的是同一個時間規則。
8. **只是實作說明**：
   - 未預期的內部錯誤一律給 2，並印 traceback；1 只給判定過的接縫類失敗。
   - zip 的 CRC 壞了 → 2。
   - `unpack` 失敗時不留半套 `sw/`。
   - `--allow-failed`（只准用在 dry-run、repo 外）把模式旗標也改成警告，只供檢視。
   - `opp_report.json` 多一個欄位 `o4_dkappa_not_judged_while_drift`；`attempt.json` 多一個欄位 `unpacked_zip`；`unpack_sw.json` 多一個欄位 `rerun_gpu_check`。
9. **程序（只改文件）**：
   - 斷線時先刪執行階段、等 1–2 分鐘，再改名。
   - 競賽一只有「開著的執行階段」才擋 test04。
   - 刪「下載」裡的舊 zip，自成一則訊息。
   - 第二次彩排前，先改名 Drive 上的彩排資料夾。
10. **第 13 條的負面演練清單**：「舊 zip 再解一次」要寫明也包括「封存過的接縫失敗 zip → 2」。另外加上「重跑不是 G4 → 2」和「archive 時 log 沒 commit → 2」。

### 2.4 還沒驗到的地方與風險

1. **正式模式仍然沒有用已 commit 的程式端到端跑過**（1.5 第 1 點，這一輪沒有做）。
   - `RealMode` 的 mock 測試比之前多涵蓋了三條路：`opp` 的正式模式程式（到寫出報告）、`apply` 的 R20 關卡、`archive` 的 R20 核對。
   - 建議照 1.5 第 1 點，在 rev 3 commit 之後、合併之前，在一次性 clone 裡實跑一次。演練順序：
     1. 先在沒有 R20 U1 時跑 `opp`，要求 exit 2；
     2. 讓 archive 時的 log 不乾淨，要求 exit 2；
     3. 用 A100 標記的 zip 做重跑，要求 exit 2。
2. **notebook 的 G4 規則從沒在 Colab 上走過**。SMOKE 刻意不查這一條，彩排也只會走第一次（沒有 `sw_failed_*`）。目前只有函式本身和格子文字的測試。
3. **GPU 名稱的比對方式不一致**：
   - 本機 `unpack` 和 `official_log` 用完整名稱 `NVIDIA RTX PRO 6000 Blackwell Server Edition` 比；notebook 用子字串 `RTX PRO 6000` 比。
   - 如果 Colab 回報的 G4 名稱變了，會發生兩件事：notebook 接受，本機拒絕（2）；接縫的容許差也會變成「不同型 GPU」（1e-3）。這和 1.5 第 4 點是同一類問題。
   - 真遇到時停下查，要改規則得另寫 rev 4。
4. **`test_p04_hygiene.py` 在任何 R20 側的已登記變動之後都會一直失敗**。它在 main 上、指紋固定，這一輪沒改。這時的閘門是 TESTDAY_04 第 1 步寫的人工核對，不是測試，Claude 要照那一列逐檔核對。
5. **`apply` 本身不再核對重跑的 GPU**。它只依靠 `unpack` 的紀錄和 `official_log` 的重算：`sw/` 在 `unpack` 之後被改，`apply` 不會發現，要到 `prepare` 才擋。
6. **notebook 改了，上傳副本的 SHA 變成 `dfde753b…`**。commit 後要照 TESTDAY_04 第 3 步那一行再核對一次。
7. **L2 沒有重跑**，只重跑了 L2b。這一輪 notebook 改的是 preflight 裡 SMOKE 不跑的分支和說明文字，03→test02→test04 的串接沒有變。
8. **repo 工作樹的 18 個新檔或改動檔都還沒 commit**（整合者的 17 個加上本檔；本輪只改其中的檔，沒有新增檔）。`FX\clones\merged3` 的 `data` junction 已拆，可以整個刪掉；`FX\slow2_tree`、`FX\L2b` 可以重建。

## 3. review round 2 and fixes

- 修正者（round 2），2026-09-30 上午。分支 `p04-testday`，HEAD `9f63acd`。本輪的修改都沒有 commit：repo 裡沒有 commit、stash、checkout。
- 三個角度的第二輪審查共 16 條：3 條 major（correctness 1、registration 1、user-colab 1），13 條 minor，沒有 blocker。其中 2 組是同一個問題從兩個角度看（G4 名稱的比法；official_log 讀 `notebook.env` 的形狀），合併後是 14 條。
- 每一條都先對照程式核實，**全部屬實**，全部修正（fixed）。沒有駁回。registration 的「R20 U1 先後」那一條裡「notebook 記的 `attempt`、`prior_sw_folders` 和本機封存次數不一致就 2」這一小點，改成只記錄加警告（理由在 3.1 第 7 條），其餘照建議做。
- 凍結檔都沒動：
  - `tools/official_log.py` 只改 04 自己加的那一段。對 tag `r20-testday-pre04` 仍然只有加行：652 行新增、0 行刪除（tag 版的 595 行照順序都在）。
  - `rev3_draft.md`、`spec.md`、`understand_*.md` 沒改。要寫進 rev 3 的條目在 3.3。
- 沒有開答案檔，沒有跑 `tools/score.py`，沒有用 GPU；每個工作最多 6 個執行緒；重的工作一次只跑一件。
- 暫存都在 `…\scratchpad\p04build\fixer2\`，以下簡寫成 `FX2`（`…` 同 1.1）。

### 3.1 每一條的處理

1. **重播失敗時，登記判定路徑上的 κ 遮罩沒有測試**（correctness major）→ **fixed**（測試）
   - 新測試 `OppDrills.test_small_replay_failures_mask_and_apply_consistency`：public 注入 3 盤、private 注入 2 盤重播失敗（都挑可解析的盤）。
     - `opp` 結束碼 0，失敗數 `{seam_val 0, public 3, private 2}`。
     - 報告的 `o4_dkappa_early` 等於測試裡用「`parsed & ~replay_failed`」手算的 `val_side_kappa`，而且和不遮罩的值、和 0 失敗時的值都不同（證明遮罩真的有作用）。
     - 接著 `apply` 結束碼 0，`colour_check` 的 O4 Δκ 和 `opp` 記的逐位相同，`o4_replay_failed` 是 3／2。
   - 負面演練（同一個測試，三個子測試）：把 `opp_report.json` 的 public、private（`o4_dkappa_early`）或 seam_val（`dkappa_O4`）加 1e-15 → `apply` 結束碼 4，stage `o4_consistency`，reason 分別是 `public_o4_dkappa`、`private_o4_dkappa`、`seam_val_o4_dkappa`；沒有 `t04_apply_report.json`、交件檔、`rank_test_probs.csv`、`pred_B.csv`。
   - 審查者的突變 MC、MA 現在都會讓測試失敗（3.2）。
2. **tol 取大值、notebook 接縫的 swap C0 重算、o4 npz 在 opp 後被改，三條沒有測試**（correctness minor）→ **fixed**（測試）
   - `ApplyDrills.test_local_seam_tol_is_the_larger_of_test02_and_test04`：
     - test02 報告的 tol 設 1e-3、test04 是 1e-4 → `seam_local.json` 的 `tol_check.tol` 和 `tol` 都是 1e-3。
     - 再把 seam_val 第 1 題的 C5 logits（第 0 類）加 0.002，本機接縫的最大差約 5.6e-4（在 1e-4 和 1e-3 之間）：test02 tol 1e-3 時結束碼 0、接縫通過；test02 tol 1e-4 時結束碼 1（`local_seam`）。
   - `ApplyDrills.test_notebook_seam_recompute_swap_disagrees`：seam_val 的 `val_p03b_s0_swap.npz` 前 5 列加 `linspace(0, 0.5)`（標記同步更新）→ 結束碼 1，stage `notebook_seam_recompute`（在本機接縫之前），B 臂 `c0_swap_max_abs_diff` > tol、`c0_s0_max_abs_diff` = 0，A 臂通過。
     - 審查者寫「加 0.5」：每一類加同一個數 softmax 不變，所以改用 d4 演練同樣的 `linspace`。
   - `ApplyDrills.test_o4_file_changed_after_opp`：`opp` 之後改 `o4_public.npz` 一個值 → 結束碼 2，訊息含「o4_public.npz differs from the report」，沒有寫 `rank/`。
3. **unpack、archive 用工作樹的 notebook 判「不是已 commit 的 notebook」**（correctness minor）→ **fixed**
   - 正式模式 `unpack` 的前提多一項：`colab/test04_rank.ipynb` 已 commit 且乾淨（`require_committed`），不是就 2。
   - 正式模式 `archive` 在判 `not_committed_notebook` 之前也做同一個核對，不是就 2。
   - `notebook_identity`：repo 的 notebook 自己不一致（`NOTEBOOK_SHA256` 那一行和重算值不同，或找不到那一行）時，正式模式拒絕（2，訊息「is not self-consistent … a local problem, not a Colab one」）；其他模式只記錄、回傳「不知道」。`apply` 原本就有 `require_committed`，現在也吃到自身一致的核對。
   - 新測試 `RealMode.test_repo_notebook_committed_and_self_consistent`：`unpack` 有核對 notebook 已 commit；notebook 沒 commit → `unpack` 2、`archive` 2；只在工作樹加一格（`NOTEBOOK_SHA256` 過時）→ `unpack` 2（以前是 1）、`archive` 2（以前會封存成占次數的 `not_committed_notebook`）；都沒有搬任何檔。
4. **notebook 用子字串判 G4，本機用完整名稱**（correctness minor＝user-colab 的「G4 名稱比法」那一條）→ **fixed**（產生器、notebook 重建）
   - `rerun_gpu_problem(prior_sw, gpu_model, g4_name)` 改成 `gpu_model != g4_name`，呼叫時傳 `P03_REF_ENV_PIN['gpu_name']`（產生器從 `results/prereg03` 的完成標記讀出並核對過的完整名稱）。三處現在是同一個判定。
   - 名稱含 `RTX PRO 6000` 但不是完整名稱時，停下訊息改成「這次的 GPU 名稱和 03 記的 G4 名稱不同，本機會拒收……截圖給 Claude，不要再執行」；其他 GPU 照舊「隔一段時間再選 G4 試」。
   - `derive_pins` 用 ast 讀 `tools/t04_rank.py` 的 `G4_GPU` 和 `tools/official_log.py` 的 `T04_RERUN_GPU`，assert 兩者都等於 `SPEC_REF_ENV['gpu_name']` 和 03 的釘選值，以後不會各自漂移（兩個檔只讀、不 import）。
   - `test_rerun_only_on_g4` 多 3 個名稱變體的子測試（全名加字尾、`NVIDIA RTX PRO 6000`、`RTX PRO 6000 Blackwell`），都要停，訊息含全名。
5. **official_log 的「每個標記都是 G4」讀錯形狀，逐標記那一段是死碼**（correctness minor＝registration 的「official_log 的 G4 核對讀錯形狀」那一條）→ **fixed**
   - `t04_report_problems`：`notebook.env` 是 `t04_rank` 寫的扁平 `{gpu_name, torch, cudnn}`（15 個標記的同一個環境，`env_same` 已在必須為 true 的清單裡）時讀 `env.gpu_name`；巢狀形狀照舊也讀；**沒有 env 時不算已證明是 G4**。
   - `tests/test_t04_official_log.py` 的 fixture（`report04`）改成 `t04_rank` 實際寫的扁平 env。`test_rerun_attempt_must_be_on_g4` 有 4 個子測試：A100 config、扁平 env 是 A100、沒有 env、巢狀 env 有一個 A100。
   - `tests/test_t04_rank.py` 新測試 `DryRunPipeline.test_notebook_env_is_the_shape_official_log_reads`：直接拿 dry run 真的寫出的 `t04_apply_report.json`，確認 `notebook.env` 是這三個鍵，再改成 attempt 2 丟給 `official_log.t04_report_problems`：G4 沒問題；env 改成 A100、config 改成 A100、拿掉 env，各報一次「not only on G4」；attempt 1 時都不報。兩個模組的形狀從此由同一個測試綁住。
6. **`archive --bug-revision N` 不核對修訂和這次失敗的先後，同一份修訂可以一再引用**（registration major）→ **fixed**
   - `t04_rank.py cmd_archive`：有 `--bug-revision N` 時
     - 較早的 `attempt.json` 已經引用同一個 N → 拒絕（2）：「one revision exchanges one uncounted rerun; a new error needs a new revision」。
     - 這一次的時間一律加進要推翻的清單（`attempt_times`）：算出輸出的執行的開始時間（見第 7 條）、每個完成標記的 `time`、`unpack_sw.json` 的 `created`、`attempt_verdict.json` 的 `created`。rev N 的那一列必須在這些時間之後才 commit，否則 2。一個時間都找不到也拒絕。
     - 這些時間記在 `attempt.json` 的 `bug_revision.overturns`。
   - `official_log.py t04_archive_problems`（`prepare`／`final`／`status` 都會重算）：
     - 每一個有 bug revision 的封存（或類別是 `bug_revision`）：N ≥ 4、HEAD 有那一列、`git_row_added` 的時間晚於 `attempt.json` 記的每個執行開始、`done_times`、`bug_revision.overturns` 的時間；沒有任何時間也算不符合。
     - 兩個封存引用同一個 N → 不符合。
     - `counts_toward_rerun` 是 false 的封存，類別只能是 `before_r20_u1` 或 `bug_revision`。
   - 測試：
     - `UnpackDrills.test_bug_revision_after_this_failure_and_once`：rev 4 在 `attempt_verdict.json` 之前 commit → 2；之後 → 封存成 `bug_revision`、不占次數，`overturns` 記了執行開始、完成標記、`unpack_sw.json`、`attempt_verdict.json`；下一次失敗再引用 rev 4 → 2；改引用 rev 5 → 可以。
     - `T04Prepare.test_bug_revision_archives`（7 個子測試加正面）：修訂早於封存的執行、等於標記時間、同一個 rev 被兩個封存引用、HEAD 沒有那一列、類別是 `bug_revision` 卻沒有修訂、不占次數卻是 `notebook_seam`、封存沒有任何時間 → 都拒絕；兩個不同修訂各一個封存加一個占次數的封存 → 可以。
   - 審查者的情境（rev 4 在 10/01 commit，早於三次失敗）現在第一次就被拒。
7. **R20 U1 的先後用 `run_log.runs[0]`，那一筆可能只是 preflight 或 GPU 關卡的停止**（registration minor）→ **fixed**
   - 新函式 `producing_runs`：最後一次執行，加上它之前狀態不是 `preflight_failed`、`interrupted`、`running` 的執行。`r20_before_problems`（`unpack` 和 `archive` 共用）逐一核對這些執行的開始時間和每個完成標記，都要晚於 R20 的第一次上傳。
   - 新測試 `RealMode.test_a_stop_before_r20_u1_is_not_the_run`（第 1 次執行是 `preflight_failed` 或 `interrupted`，都在 U1 之前）：
     - 第 2 次執行在 U1 之後接縫失敗 → `unpack` 1，`archive` 記 `notebook_seam`、占次數、`before_r20_u1_problems` 是空的（以前會記成不占次數的 `before_r20_u1`）。
     - 第 2 次執行在 U1 之後完成 → `unpack` 0（以前被拒 2）。
     - 對照：第 2 次執行本身在 U1 之前完成 → 仍然 2，訊息只提 run 2。
   - 審查者另外建議「notebook 記的 `attempt`、`prior_sw_folders` 和本機封存次數不一致就 2」：**改成只記錄加警告**。理由：`apply` exit 1 的封存在 Drive 上的資料夾本身接縫通過，notebook 的 `ATTEMPT` 本來就可能比本機少；Drive 改名打錯一個數字時，拒收會浪費一次正確算完的執行。現在 `unpack_sw.json` 多 `drive_archives`（Drive 上的 `sw_failed_*` 對本機封存換算的名稱），不一致時印 WARNING；TESTDAY_04 第 4 步寫了看到警告時先請使用者截圖核對、對不上就停下。測試 `Rules.test_drive_archives_record_only`。
8. **防誤改測試放寬後，打分紀錄帳新增一列、PREREG_02 新增一列修訂都不會失敗**（registration minor）→ **fixed**
   - `test_t04_hygiene.py` 新測試 `test_ledger_and_prereg_rows_pinned`：
     - 打分紀錄帳、PREREG_01–03 要已追蹤而且 `git status` 乾淨。
     - 打分紀錄帳的資料列（csv 解析）要剛好是 6ec5e68 的 25 列；多出來的列印在失敗訊息裡。
     - PREREG_01–03 的 `| rev N |` 列要和 6ec5e68 相同；多出來的列印在失敗訊息裡。
     - `C10_K5=1`（K5 合併前、在分支上）：這 4 個檔和 6ec5e68 逐位元組相同（CRLF 折成 LF）。
   - 原本的子序列測試留著。`docs/COLAB_RUN.md` 在這個分支上有改（test04 一節），所以不在「乾淨」的要求裡，仍照子序列規則和「只有一節 test04」。
   - TESTDAY_04 第 1 步閘門的例外改寫：失敗的只能是指紋核對和／或這個新測試；多出來的每一列都要已 commit、對得上一份已 commit 的登記，逐列記進 RESULTS，不會自動算過。K5 核對表多一項 `C10_K5=1`。「R20 側的已登記變動」一節同步改寫。
   - 在一次性 clone 裡重做審查者的實驗（3.2）：沒 commit 的新增列、已 commit 的新增列，都讓這個測試失敗。
9. **幾道縱深防護沒有測試（R13、R14、R15、R22、O8）**（registration minor）→ **fixed**（測試）
   - `UnpackDrills.test_two_counted_archives_refuse_unpack`：已有 2 個占次數的封存 → `unpack` 2（「登記只准重跑一次」），沒有 `sw/`、沒有 `unpack_sw.json`。
   - `DirsAndModes.test_allow_failed_only_in_dry_run_outside_the_repo`：正式模式、`--rehearsal` 用 repo 裡的彩排資料夾（測試把 `REHEARSAL_R4` 指到暫存）→ 都是 2，訊息「--allow-failed is for inspection only」，什麼都沒寫。
   - `ApplyDrills.test_unpack_record_must_match_failed`（3 個子測試）：多了一個封存資料夾、`attempt_number` 被改、解開的 zip 後來被封存 → 都是 2（「does not match results/test04/failed」）。
   - `T04Prepare.test_archive_without_counts_toward_rerun_counts`：`attempt.json` 沒有 `counts_toward_rerun` 的封存算占次數；再加一個占次數的封存 → `prepare` 拒絕（「2 counted reruns archived」）。
10. **notebook 的 G4 關卡，停下那一行沒有測試走到**（user-colab major）→ **fixed**（測試）
    - 新測試 `Preflight.test_rerun_gpu_stop_is_taken`：從 `preflight_t04` 格切出 `RERUN_GPU_PROBLEM =` 到 `# ⑦` 那一段（先確認裡面沒有 `write_json`、`_INPUT`），用 `SMOKE=False` 和一個會丟 `SystemExit` 的 `preflight_stop` 執行：A100 重跑、查不到 GPU 名稱、G4 名稱變體 → 都停，訊息含「只能用 G4」；G4 重跑、第一次用 A100、SMOKE → 都不停。
    - 審查者的突變（`if RERUN_GPU_PROBLEM:` 改成 `if False:`）和「改回子字串判定」的突變，現在都會讓測試失敗（3.2）。
11. **notebook 用子字串判 G4**（user-colab minor）→ 和第 4 條同一件，**fixed**。
12. **notebook 的四段文字和 COLAB_RUN 矛盾**（user-colab minor）→ **fixed**（產生器、notebook 重建）
    - 第一格「中途斷線時」改成①截圖、②**先**刪執行階段並等 1–2 分鐘、③再改名（zip 也改）並截圖、④開新的執行階段，每一步等 Claude 回覆，並指向 COLAB_RUN「中途斷線時」。
    - ⑥-3（斷線後不接續）的紅字改成同一個順序。⑥-1（上一次接縫沒過）的紅字也改成先刪執行階段、Claude 說要重跑才改名、等回覆才開新的執行階段（審查沒點名，順手一致）。
    - ⑧ Drive 空間不足：改成「不要自己刪檔、也不要清垃圾桶；中斷連線並刪除執行階段，截圖給 Claude」。
    - 「沒有 GPU」的紅字（`build_official_2` 的 `rep()`）補上「Claude 說這次是重跑時只能選 G4，灰色就隔一段時間再試」。
    - CHANGE_ROWS 的設定、官方 2、preflight 三列跟著改。
13. **test04 第 6 步少了「不要從『最近』清單開」，開到舊版會用掉唯一的重跑**（user-colab minor）→ **fixed**（文件、notebook）
    - COLAB_RUN test04 第 6 步照 test02 第 7 步補上「不要從 Colab 的『最近』清單開」和選單裡沒有「Google Colaboratory」時的替代開法。
    - notebook 第一個程式格最後多印一行「notebook 版本：<`NOTEBOOK_SHA256` 前 12 碼>」。第一格的「怎麼跑」第 4 步、COLAB_RUN 第 10 步請使用者和 Claude 給的 12 碼對照，不同就中斷連線並刪除執行階段、截圖；「出狀況時」表多兩列（版本不同、G4 名稱變體）。
    - TESTDAY_04 第 3 步：Claude 用一行指令從已 commit 的 notebook 取出這 12 碼（目前 `0e7178be2955`，Git Bash 跑過），寫進給使用者的訊息；版本不同時不算一次執行（還沒有完成標記照 preflight 停下處理，已經有就照「斷線」）。
    - TESTDAY_04 彩排一節最後補：彩排後改了產生器時，另發一則訊息請使用者把 `old_rehearsal` 裡的 `test04_rank.ipynb` 改名成 `test04_rank_OLD.ipynb`。
14. **TESTDAY_04 通用結束碼表的 3 跳過第 7 步；apply 3 在 K6／FAIL_OPENING 時沒有報告可 commit**（user-colab minor）→ **fixed**（文件）
    - 通用表的 3 改成「commit 寫出的報告（K6、FAIL_OPENING 時工具不寫檔，stderr 記進 RESULTS）；最終 R20（FAIL_OPENING 時 R10）→ 第 7、8 步；不重跑」。
    - 第 5 步 `apply` 的 3 拆成兩列：Δκ／重播失敗 → commit `t04_apply_report.json`；T1 FAIL_OPENING、K6、`opp` 已判不用 → 工具不寫檔，stderr 記進 RESULTS。已對照程式：`cmd_apply` 只在 1、4 時寫 `attempt_verdict.json`，K6（`check_k6`）、FAIL_OPENING（`t1_first`）、`opp_no_04` 都在寫報告之前以 3 結束。
15. **COLAB_RUN 檔頭第 3 行的導引過時**（user-colab minor）→ **fixed**（文件）
    - 在原第 3 行之前加一段：「11/05 起競賽二的 04 照最後一節『測試集 04（test04）』；11/04 的 test02（含彩排）照『測試集（test02）』。下面這段 9/28 的說明是當時的紀錄（03 已在 9/28 跑完）」。原來每一行都在，`test_t04_hygiene` 通過；對 HEAD 仍然只有加行（129 行新增、0 行刪除），檔案維持 CRLF。
- 另外（不是審查意見）：`tests/test_t04_dryrun.py` 最後那個 skip 的說明文字改成新的 R20 U1 規則，沒有行為改變。

**改了哪些檔**（都不是凍結檔；SHA-256 前 12 碼，CRLF 折成 LF 後算）

| 檔 | SHA-256 |
|---|---|
| `tools/t04_rank.py` | `d9180ccae993` |
| `tools/official_log.py`（只改 04 段；對 tag 只有加行） | `ee0ace46b4b6` |
| `colab/build_test04_rank_nb.py` | `295f57a33bb1` |
| `colab/test04_rank.ipynb`（重建） | `a6513f41d008` |
| `results/test04/notebook_diff_rank.txt`（重建） | `587593d426c0` |
| `data/colab_upload/test04/test04_rank.ipynb`（`--upload-copy` 重寫；資料夾裡只有這一個檔） | `a6513f41d008` |
| `tests/test_t04_rank.py` | `6db2d7b8af23` |
| `tests/test_t04_official_log.py` | `8109b09cd3e9` |
| `tests/test_t04_nb.py` | `d9609900ca5a` |
| `tests/test_t04_hygiene.py` | `ea4953607718` |
| `tests/test_t04_dryrun.py`（只改 skip 說明） | `81cb12ec16a6` |
| `docs/TESTDAY_04.md` | `d09813e8c7f0` |
| `docs/COLAB_RUN.md`（檔頭加一段、test04 一節；工作樹 CRLF，原始位元組 `9c98b4fd5857`） | `f8811682bbf9` |
| 本檔 | — |

- notebook 完整的 SHA-256 是 `a6513f41d008277dd12088e2f548a70ec6fa7f371eca71064e49fc01546c0678`，照 notebook 自己的規則算的 `notebook_sha256` 是 `0e7178be29559e8a4bd9f30e40bba97d0c7da4d1393b8d7b856b7fc62a2468a7`（使用者對照的 12 碼 `0e7178be2955`）。2.1 記的 `dfde753b…`／`a27ef9c8…` 已經不適用。
- 產生器再跑一次，notebook 和 diff 逐位元組相同（`test_rebuild_is_byte_identical`）。

### 3.2 驗證

以下 `PY` = `.venv\Scripts\python.exe`，在 repo 根目錄執行；執行緒上限 6，重的工作一次一件。開跑前看過 `handoff.py`：玉山那條線忙碌，C1、自駕車閒置。

**完整測試**（`PY -m pytest -q -rs tests`）

- 第一次：747 passed、65 skipped、374 subtests，646 秒。比 2.2 的 732 多 15 個，就是本輪新加的測試（`Rules.test_drive_archives_record_only` 是這一次收集之後才加的，不在這個數字裡）。
- 最後一次（全部修改之後，12:35–13:19）：**748 passed、65 skipped**、374 subtests，2,598 秒（43 分：同時段別條線的 pytest 也在跑）。多出的 1 個就是 `Rules.test_drive_archives_record_only`。唯一的 warning 仍是 `test_p03_nb` 的 torch `pin_memory` 提示。中間一次加了 30 分鐘上限的完整測試在 30% 左右被上限砍掉（沒有失敗），不算。
- 65 個 skip 和 1.2 相同：62 個要設 `C10_SLOW`，另外 3 個見 1.2。

**`C10_SLOW=2`**（11:05–11:41，`FX2\verify_chain.sh`）

- 指令：`C10_SLOW=2 C10_KEEP=1 C10_DRYRUN_DIR=FX2\slow2_tree PY -m pytest -q -rs`，範圍是 `tests/test_t04_*.py` 加 `tests/test_t02_dryrun.py`。
- **256 passed、1 skipped**、372 subtests，2,177 秒（36 分；比 2.2 的 22 分半慢：同時段別條線也在用 CPU，測試也多了 16 個）。
- 1 個 skip 是 `test_t04_dryrun.py` 的「R20 第一次上傳前跑的 notebook」：只能在正式模式用 mock 做，由 `RealMode` 涵蓋（本輪另加 `test_a_stop_before_r20_u1_is_not_the_run`）。
- 前後 repo 的 `git status` 都是同樣的 18 項。

**突變核對**（一次性 clone `FX2\clones\merged4`，腳本 `FX2\mutate3.py`、`FX2\mut_chain.sh`，紀錄 `FX2\mut_out\mutations.log`；repo 沒有被寫）

- clone：`9f63acd` 加上 repo 工作樹的 18 個檔（`INT\make_merged_clone.py`），clone 內 commit `a7f84aa`。`data\` 用 junction 指到 repo，只讀；跑完已用 `cmd /c rmdir` 拆掉，核對過 clone 裡已經沒有 `data`。
- 基準（不改程式）：`test_t04_rank.py`、`test_t04_official_log.py`、`test_t04_hygiene.py` 141 passed、1 skipped。
- **23 個突變全部被抓到**（每個跑對應的測試檔，`-x`；合計約 32 分）：
  - 審查者的：MA（拿掉 public／private 的 O4 一致性）、MB（max 改 min）、MC（opp 的遮罩只用 parsed）、MD（拿掉 o4 npz 的 SHA 核對）、ML（swap C0 設成 0）、R13（`>= 2` 改 `>= 3`）、R14、R15、R22、O8。
  - 本輪修正本身：unpack 不核對 notebook 已 commit、不核對 notebook 自身一致、archive 不核對 notebook 已 commit、bug revision 可重複引用、bug revision 不比這次的時間、R20 先後改回 `runs[0]`、official_log 的修訂時間／重複引用／不占次數類別三項、official_log 忽略扁平 env（`test_t04_official_log.py` 和 `test_t04_rank.py` 的形狀測試各抓一次）。
  - 審查者的兩個 hygiene 實驗（沒 commit 地在打分紀錄帳加一列；在 PREREG_02 加 `| rev 9 |`）。
- 同一個 clone 裡另外把「打分紀錄帳多一列、PREREG_02 多 rev 9」**commit** 進去再跑 `test_ledger_and_prereg_rows_pinned`：失敗，訊息列出多出來的那一列。之後 `git reset --hard` 回 `a7f84aa`，`git status` 是空的。
- notebook 這一側（`FX2\mut_nb.py`，只在記憶體裡改 notebook JSON）：`if RERUN_GPU_PROBLEM:` 改成 `if False:` → `test_rerun_gpu_stop_is_taken` 失敗；判定改回子字串 → `test_rerun_gpu_stop_is_taken`、`test_rerun_only_on_g4` 失敗。repo 的 `git status` 前後都是 18 項。

**R20 逐位元組不變**（spec 8.1 第 3 點；因為 `official_log.py` 改了才重做，同一個 clone `merged4`）

- `C10_DRYRUN_DIR` 用 1.2 的同一個路徑（`INT\r20dry`），跑完搬到 `FX2\verify_out\merged4_fromval`。
- `PY -m pytest -q tests/test_t02_dryrun.py -k FromValPipeline -p no:cacheprovider`：15 passed，201 秒。跑之前、之後的 `git status` 都是空的。
- 和 1.2 的 `main_6ec5e68` 樹比（`INT\compare_r20.py` → `FX2\verify_out\compare_main_vs_merged4.json`）：
  - 兩棵樹都等於 golden：7 個 CSV、正規化後的 `decide_summary.json`、`opening_test.npz` 的陣列。
  - 128 個檔裡 82 個逐位元組相同，`decide_summary.json` 整檔相同。不同的 46 個和 2.2 的 `merged3` 是同一組檔，只差在時間、耗時、路徑、`git_head`，以及含時間戳記的檔的 SHA。
- 同一個 clone 裡：
  - `git diff --numstat r20-testday-pre04 HEAD -- <39 個指紋檔>`：只有 `tools/official_log.py`，652 行新增、0 行刪除。
  - `TagVersusNew`：4 passed、66 subtests。
  - 三個 hygiene 檔（p04、t02、t04）：40 passed；`C10_K5=1` 的 `test_t04_hygiene.py`：20 passed。

**notebook 冒煙 L2b**（因為產生器改了才重做；scratch `FX2\L2b`，11:45–12:04）

| 步驟 | 指令 | 結果 |
|---|---|---|
| 建替身樹 | `PY colab/make_test04_smoke_tree.py --scratch FX2\L2b --threads 6` | OK，150 秒 |
| notebook | `PY colab/run_local_smoke.py colab/test04_rank.ipynb --scratch FX2\L2b` | OK，430 秒 |
| 核對 | `PY colab/make_test04_smoke_tree.py --scratch FX2\L2b --check --threads 6` | `ok: true`，`problems` 是空的，551 秒 |

- 接縫：通過，tol 1e-4，`bitwise_equal` true；兩臂 C0（s0、swap）最大差都是 0.0。`run_log` 是 `completed`。
- zip：`74041c79…`，349,211 bytes，40 個成員全部在 `sw/` 底下，必要成員 40 個都在。
- 轉接：public、private 用 P03 自己的 `dump_val_p03` 重算，5 個 npz 都逐位元組相同（private 175 盤）。
- 第一個程式格最後一行印 `notebook 版本：0e7178be2955`，等於 TESTDAY_04 第 3 步那一行指令從 repo 取出的 12 碼。
- 重跑 G4 核對在 SMOKE 下照設計不執行；它的停下由 `test_rerun_gpu_stop_is_taken` 直接執行那一段涵蓋。

### 3.3 要寫進 rev 3 的條目

本輪新增的條目。都不改來源、參數、組合、門檻；貼法照 `rev3_draft.md` 開頭的規則，由主 session 在 Rev3 階段併進草稿。

1. **第 1 條（R20 第一次上傳在前）**：「執行開始時間」指算出輸出的那一次執行（最後一次執行，加上它之前狀態不是 `preflight_failed`、`interrupted`、`running` 的執行），和每個完成標記的時間。同一個 Drive 資料夾在 U1 之前只停在 preflight 或 GPU 關卡的那一次不算。`unpack` 和 `archive` 用同一個函式。
2. **第 4、9 條（重跑只在 G4）**：三處（notebook preflight ⑥-5、`t04_rank.py unpack`、`official_log.py`）一律用完整名稱 `NVIDIA RTX PRO 6000 Blackwell Server Edition` 比；產生器核對三處的常數和 03 完成標記的釘選值相同。名稱含 `RTX PRO 6000` 但不是完整名稱時，notebook 也在 preflight 停下（請使用者截圖）。`official_log` 讀報告實際寫的扁平 `notebook.env`；沒有 env 視為沒證明是 G4。
3. **第 9 條（「不是已 commit 的 notebook」）**：只對照已 commit、乾淨、而且自身一致（`NOTEBOOK_SHA256` 那一行等於重算值）的 `colab/test04_rank.ipynb`。repo 這一份不符合時是本機問題：`unpack`、`archive`、`apply` 都拒絕（2），不判成占次數的 1。
4. **第 20 條補**：
   - `archive --bug-revision N`：rev N 必須在這一次失敗之後才 commit（晚於算出輸出的執行的開始、每個完成標記、`unpack_sw.json`、`attempt_verdict.json`），不只晚於它推翻的紀錄。
   - 一份修訂只換一次不占次數的重跑：較早的封存已經引用同一個 N 就拒絕；新的錯誤要寫 rev N+1。
   - `official_log` 在 `prepare`／`final`／`status` 重算：有修訂的封存，那一列在 HEAD、commit 晚於封存記的時間、沒有兩個封存引用同一個 N；不占次數的封存只可能是 `before_r20_u1` 或 `bug_revision`。
5. **第 19 條（閘門）**：`test_t04_hygiene.py` 另要求打分紀錄帳和 PREREG_01–03 已 commit、乾淨，打分紀錄帳 25 列、`| rev N |` 列和 6ec5e68 相同。多出來的列不自動算過：TESTDAY_04 第 1 步逐列核對它的 commit 和所屬登記，記進 RESULTS。K5 合併前在分支上跑 `C10_K5=1`：這些檔和 6ec5e68 逐位元組相同。這一條取代 2.3 第 5 條最後一點「只要求每一行照順序還在」的說法。
6. **第 13 條的負面演練清單**：補上本輪新加的——已有 2 個占次數的封存 → `unpack` 2；`--allow-failed` 用在正式模式或 repo 裡的彩排資料夾 → 2；`unpack_sw.json` 和 `failed/` 對不上 → `apply` 2；`o4_<set>.npz` 在 `opp` 之後被改 → 2；`opp` 記的 O4 Δκ 差 1 bit → `apply` 4（`o4_consistency`）；swap C0 重算不一致 → 1（`notebook_seam_recompute`）；tol 取兩者較大；repo 的 notebook 沒 commit 或不一致 → 2；`--bug-revision` 早於這次失敗或重複引用 → 2；R20 之前只停在 preflight 的那一次 → 不算。
7. **只是實作說明**：
   - `unpack_sw.json` 多一欄 `drive_archives`：notebook 看到的 Drive `sw_failed_*` 資料夾對本機封存，只記錄，不一致時印 WARNING（不擋的理由見 3.1 第 7 條）。
   - notebook 第一個程式格最後印「notebook 版本」（`NOTEBOOK_SHA256` 前 12 碼），給使用者和 Claude 給的 12 碼對照。
   - `attempt.json` 的 `bug_revision.overturns` 包含這一次失敗自己的時間。
8. **程序（只改文件和 notebook 的文字）**：
   - COLAB_RUN test04 第 6 步：不要從 Colab 的「最近」清單開，附替代開法；第 10 步對照 12 碼。
   - 彩排後改了產生器：請使用者把 `old_rehearsal` 裡的舊 `test04_rank.ipynb` 改名成 `test04_rank_OLD.ipynb`。
   - notebook 的斷線、上一次接縫沒過兩則紅字改成「先刪執行階段、等 1–2 分鐘、再改名」；Drive 空間不足改成「不自己刪檔或清垃圾桶，截圖給 Claude」；「沒有 GPU」的紅字補「重跑只能選 G4」。
   - 結束碼 3 一律接第 7、8 步（設 `final`、確認最後一次上傳、11/22–23 的 `status`）；`apply` 以 K6、FAIL_OPENING 或 `opp` 已判不用結束時不寫檔，stderr 記進 RESULTS。

### 3.4 還沒驗到的地方與風險

1. **正式模式仍然沒有用已 commit 的程式端到端跑過**（1.5 第 1 點、2.4 第 1 點，這一輪也沒有做）。`RealMode` 的 mock 測試又多涵蓋了 R20 U1 前的 preflight 停止、notebook 的 commit／自身一致核對。建議照 2.4 第 1 點在 rev 3 commit 之後、合併之前實跑一次，另加兩個演練：只在工作樹改 notebook 時 `unpack` 要 2；`archive --bug-revision` 引用一份早於這次失敗的修訂要 2。
2. **notebook 的 G4 關卡從沒在 Colab 上走過**。現在有直接執行那一段的測試，但 G4 回報的名稱只在 03 的完成標記裡看過一次。彩排時請記下 `device` 那一行的完整名稱；和 `NVIDIA RTX PRO 6000 Blackwell Server Edition` 不同時，第一次執行本機只會把接縫容許差放寬到 1e-3（照 spec），但登記的重跑會在 preflight 停下，要在 11/04 前決定要不要寫 rev 4。
3. **Drive 上 `sw_failed_*` 的名稱只記錄、不擋**（3.1 第 7 條）。使用者改名打錯編號時，本機只印警告，要靠 TESTDAY_04 第 4 步的人工核對。
4. **`official_log` 不能自己驗證 `before_r20_u1` 的封存真的在 R20 上傳前**：它只核對不占次數的封存類別是 `before_r20_u1` 或 `bug_revision`。`before_r20_u1` 的判定只在 `t04_rank.py archive`（正式模式、已 commit 的 log）。
5. **打分紀錄帳釘在 25 列**：11 月若依另一份登記再打分，`test_t04_hygiene.py` 會一直失敗，第 1 步的閘門變成人工逐列核對（和指紋清單同一種做法）。
6. **notebook 改了，上傳副本的 SHA 變成 `a6513f41…`，使用者對照的 12 碼是 `0e7178be2955`**。commit 後照 TESTDAY_04 第 3 步那兩行再核對一次（12 碼那一行已在 Git Bash 跑過）。
7. **L2 沒有重跑**，只重跑了 L2b。這一輪 notebook 改的是 preflight 裡 SMOKE 不跑的分支、紅字和第一格最後多印的一行，03→test02→test04 的串接沒有變。
8. **12 碼對照的時間點**：使用者在第 10 步看到版本時，dump 可能已經開始算。開錯版本時中斷不算一次執行，但已經有完成標記時要照「斷線」改名 `sw_interrupted_N`，多一輪 Drive 操作。
9. **repo 工作樹的 18 個新檔或改動檔都還沒 commit**（本輪只改其中的檔，沒有新增檔）。`FX2\clones\merged4` 的 `data` junction 已拆，可以整個刪掉；`FX2\slow2_tree`、`FX2\L2b`、`FX2\verify_out\merged4_fromval` 可以重建。

## 4. review round 3 and fixes

- 修正者（round 3），2026-09-30 下午。分支 `p04-testday`，HEAD `9f63acd`。本輪的修改都沒有 commit：repo 裡沒有 commit、stash、checkout。
- 三個角度的第三輪審查共 12 條：2 條 major（registration 1、user-colab 1），10 條 minor，沒有 blocker。其中 3 條是同一件事（不占次數的封存在 Drive 上也叫 `sw_failed_N`：correctness 第 2 條、registration 第 2 條、user-colab 第 3 條），合併後是 10 件。
- 每一條都先對照程式核實，**全部屬實**，全部修正（fixed）。有 1 個小點沒照審查者的建議做：不占次數的封存之後要不要也只收 G4（4.1 第 2 條）。這裡選了和本機工具、`official_log` 一致、也和 rev 3 草稿字面一致的做法，較嚴的另一種讀法列進 4.3，請主 session 在 Rev3 階段決定。
- 凍結檔都沒動：
  - `tools/official_log.py` 本輪沒改（LF SHA 仍是 round 2 的 `ee0ace46b4b6`；對 tag `r20-testday-pre04` 仍然只有 652 行新增、0 行刪除）。所以不必重做 spec 8.1 第 3 點的 R20 clone 比對。
  - `rev3_draft.md`、`spec.md`、`understand_*.md` 沒改。要寫進 rev 3 的條目在 4.3。
- 沒有開答案檔，沒有跑 `tools/score.py`，沒有用 GPU；每個工作最多 6 個執行緒，重的工作一次只跑一件（開跑前看過 `handoff.py`：自駕車、玉山兩條線忙碌）。
- 暫存都在 `…\scratchpad\p04build\fixer3\`，以下簡寫成 `FX3`。

### 4.1 每一條的處理

1. **`--bug-revision N` 在這台電腦一定當掉**（registration major）→ **fixed**
   - 核實：`t04.git_show_head('docs/PREREG_04_OPPONENT.md')` 在 repo 實跑，reader thread 印 `UnicodeDecodeError: 'cp950' …`，函式丟 `AttributeError: 'NoneType' object has no attribute 'strip'`（`locale.getpreferredencoding(False)` 是 cp950）。
   - 修正：`tools/t04_rank.py` 的 `git()` 不再借用 `t02_rank.git`（凍結，text 模式用系統編碼），改成自己跑 `subprocess.run(..., capture_output=True)`，stdout 用 UTF-8 解碼（和 `official_log.git_head_text` 同一個做法）；git 跑不起來時回 `(None, 訊息)`，stdout 是空的也不會當掉。`git_show_head` 只接受字串。`archive`、`opp --rerun`、`apply --rerun` 的 `--bug-revision` 都走同一個 `check_bug_revision`，一次修好三條路。
   - 新測試 `RealGit`（`tests/test_t04_rank.py`，只有唯讀的 git 指令，不換假函式；PREREG_04 在工作樹有改動時才把 `require_committed` 換掉）：
     - `git_show_head` 讀到的中文 PREREG_04 含 `| rev 2 |` 和「事前登記」；不存在的檔回 None，`git()` 回 `(非 0, "")`。
     - `git_first_commit_with(..., "| rev 2 |")` 回 40 碼 commit 和可解析的時間；`| rev 99 |` 回 `(None, None)`。
     - `check_bug_revision(99, …)` → 2，訊息「has no line starting '| rev 99 |'」。
     - 審查者的指令 `archive --dry-run --bug-revision 99` → 2，同一則訊息，沒有「unexpected internal error」、沒有 traceback，什麼都沒搬。
   - 正面路徑（成功的那一條）另在一次性 clone `FX3\clones\m5` 實跑：在 clone 的 PREREG_04 加一列 `| rev 4 |` 並 commit，`check_bug_revision(4, [早於它的時間])` 回傳 commit `fadf7909…`、時間和那一列；要推翻的時間晚於修訂 → 2。跑完 `reset --hard` 回原 commit，`git status` 是空的。
2. **不占次數的封存在 Drive 上也叫 `sw_failed_N`**（correctness 第 2 條＝registration 第 2 條＝user-colab 第 3 條）→ **fixed**
   - 核實：`archive` 不論類別都印「`sw` 改名成 `sw_failed_{n}`」；notebook ⑥-4 把所有接縫失敗的 `sw_*` 都算進「2 個就停」和 `ATTEMPT`，⑥-5 只要有 `sw_failed_*` 就只收 G4。本機 `unpack` 和 `official_log` 只算占次數的封存（`attempt_number = 占次數的封存數 + 1`，G4 只在 `≥ 2` 時要求）。所以「`bug_revision` 封存（不占次數）＋再一次占次數的接縫失敗」之後，notebook 會擋掉登記仍允許的重跑；`before_r20_u1` 之後的那一次，notebook 比本機嚴（逼成只收 G4）。
   - 修正（審查者建議的做法 (a)，改名）：
     - `t04_rank.py archive`：占次數的封存 → `sw_failed_N`／`test04_rank_results_failed_N.zip`；不占次數的（`before_r20_u1`、`bug_revision`）→ `sw_exempt_N`／`test04_rank_results_exempt_N.zip`。印「`Drive: sw -> …`」、正式模式的「給使用者」那一行，和「下一次執行：登記的重跑，只用 G4」或「不占重跑次數…GPU 照第一次的規則：G4 優先、A100 備用」。`attempt.json` 多一欄 `drive_rename`。
     - notebook ⑥-4：`SEAM_FAILED_FOLDERS`＝`sw_failed_*` 一律算（也包括 Drive 上接縫本身通過、本機 `apply` 才判失敗的那種，和本機算法相同），`sw_exempt_*` 不算，其他改過名而且自己有接縫類失敗紀錄的也算（安全的一邊）。`ATTEMPT` 跟著算。
     - notebook ⑥-5：判定不變（有 `sw_failed_*` 才只收 G4），註解寫明 `sw_exempt_*` 之後照第一次的規則。三處（notebook、`unpack`、`official_log`）現在都是「有一個占次數的封存，這次才是登記的重跑」。
     - `unpack` 的 `drive_archives` 對照：Drive 上的 `sw_failed_*`、`sw_exempt_*` 對本機依 `counts_toward_rerun` 換算的名稱。
   - **沒照做的一小點**：user-colab 第 3 條建議「有 `sw_void_*` 時也只收 G4」（registration 第 2 條說兩種都可以，照揭露的做法決定）。這裡沒有這樣做，理由：
     - rev 3 草稿第 1 條（R20 上傳前跑的）寫「不算第 9 條那一次重跑」，第 20 條（程式錯誤）寫「這不算第 9 條那一次重跑」，第 9 條寫「第一次在 G4 或 A100 都一樣」、「第 9 條的重跑只在 G4」；本機 `unpack` 和 `official_log` 也是這樣判。
     - 改成較嚴的讀法要同時改 `unpack` 和 `official_log.py`（R20 側只准加行的檔），而且會讓 04 在拿不到 G4 時更早出局。
     - 這兩種讀法的差別在第 1 條「照第 9 條的做法再跑一次」的「做法」包不包括 GPU。列進 4.3 第 2 條，由主 session 決定；要改成較嚴的，三處要一起改。
   - 測試：
     - `test_t04_nb.py`：`Preflight.test_exempt_archives_do_not_count_and_failed_ones_always_do`（`sw_exempt_1`＋`sw_failed_2` 都接縫失敗 → 不停、`ATTEMPT` 2；在非 SMOKE 下直接執行 ⑥-5：只有 `sw_exempt_1` 的 A100 → 不停，加上 `sw_failed_2` → 停；再加一個接縫通過的 `sw_failed_3` → 「登記只准重跑一次」）；`test_rerun_only_on_g4` 多兩個判斷。
     - `test_t04_rank.py`：`test_archive_classes`（`bug_revision` → `sw_exempt_1`，下一個占次數的 → `sw_failed_2`）、`RealMode.test_before_r20_first_upload`（正式模式印「`sw` 改名成 `sw_exempt_1`」和「G4 優先、A100 備用」）、`test_unpack_real_ok_then_freshness`（印「登記的重跑，只用 G4」）、`Rules.test_drive_archives_record_only`（兩種名稱的對照）。
3. **A100 第一次執行的 tol 沒有測試**（correctness 第 1 條）→ **fixed**（測試）
   - 新測試 `ApplyDrills.test_first_run_on_a100`：15 個完成標記都改成 A100、`seam_check.json` 的 tol 和兩臂的 tol 都是 1e-3 → `apply` 結束碼 0；`seam_local.json` 的 `tol_check.sw` 是 `recomputed_tol` 1e-3、`notebook_tol` 1e-3、`same_gpu` false、`seam_dump_gpu` A100；test02 那邊 1e-4，`tol` 取較大者 1e-3，接縫通過。同樣的標記、notebook 仍記 1e-4 → 1（`notebook_seam_recompute`、`recompute_mismatch`，訊息「tol: notebook 0.0001, from the markers 0.001」）。
   - 審查者的突變 MX1（`seam_tol_sw` 永遠回 1e-4）現在會讓這個測試失敗（4.2）。
4. **凍結的結果資料夾只看 `git status`，已 commit 的修改測不到**（registration 第 3 條）→ **fixed**（測試）
   - `test_t04_hygiene.py` 新測試 `FrozenPrereg04.test_frozen_results_equal_to_base`：`git diff --name-only 6ec5e68 -- results/prereg01 … results/prereg04` 要是空的（工作樹對 6ec5e68 比內容，已 commit 和還沒 commit 的改動都抓得到），而且這四個資料夾沒有未追蹤的檔。原本的 `git status` 檢查留著。
   - 核對：repo 現在 `git diff --name-only 6ec5e68 HEAD -- results/prereg0*` 是空的，不會誤報。
   - 在 clone 重做審查者的實驗：commit 一行進 `results/prereg03/RESULTS.md`、commit 一個改動進 `results/prereg04/rank/check_report.json`，這個測試都失敗（4.2）。
5. **打分紀錄帳的列數釘死，會讓 R20 凍結的「全綠」就緒閘門變紅**（registration 第 4 條）→ **fixed**（審查者的做法 (a)）
   - `test_ledger_and_prereg_rows_pinned` 只在 `C10_T04_GATE=1` 或 `C10_K5=1` 時跑，其他時候 skip（說明寫在 skip 訊息裡）。子序列規則（`test_ledger_docs_colab_run_keep_every_line`）照舊每次都跑。
   - `docs/TESTDAY_04.md`：每一步的防誤改測試和第 1 步的指令都改成 `C10_T04_GATE=1 PY -m pytest -q …`（第 1 步加 `-rs`），閘門表那一列加「`-rs` 沒有列出這個測試被略過」；「R20 側的已登記變動」一節補一句說明。K5 核對表原本就是 `C10_K5=1`。
   - 在 clone 重做審查者的實驗：commit 一列打分紀錄帳之後，一般的 `pytest tests/test_t04_hygiene.py` 通過（20 passed、1 skipped），`C10_T04_GATE=1` 時這個測試失敗（4.2）。
   - 沒有改 repo 的 `CLAUDE.md`（不在本輪可改的檔裡；改成 (a) 之後 R20 的閘門不會再被它弄紅，就不需要那段註明）。
6. **版本 12 碼的處理和登記、工具不一致，而且核對發生在開始寫和算之後**（user-colab major）→ **fixed**（審查者的做法 (b)，另加 (a) 的邊界）
   - notebook 設定格：在 `NOTEBOOK_SHA256` 那一行之後、任何寫檔之前（在 `write_json`、`save_run_log()`、`save_config()` 定義和呼叫之前；之前只建了空的結果目錄），Colab 上（正式和彩排；SMOKE、本機不問）用 `input()` 印 12 碼，請使用者和 Claude 給的 12 碼對照。輸入 `y`（前後空白、大小寫不拘）才往下；其他（包括 `n`、空白、`yes`、`EOFError`）就 `SystemExit`：「沒有確認…還沒寫任何紀錄、還沒開始算，不算一次執行」，請使用者刪執行階段、截圖。最後原本那行印版本改成只記錄。
   - 所以開錯版本時 notebook 不會寫進 `sw`、也不會算；TESTDAY_04 第 3 步原本「已經有完成標記時照斷線改名」的特例刪掉。剩下的邊界照登記寫明：使用者對不同的 12 碼仍輸入 `y` 而跑完 → 照常下載，`unpack` 判 `not_committed_notebook`（占次數，rev 3 第 9 條不變）。按下全部執行 1 分鐘內沒有輸入框（沒有這個輸入框的舊版；目前還沒給過使用者）→ 馬上刪執行階段，還沒有完成標記就照 GPU 關卡停下處理，有就照斷線。
   - 文件：notebook 第一格「怎麼跑」第 4 步、「會停下來的情況」；COLAB_RUN 第 9 步（新的「版本確認」兩點）、第 10 步、「出狀況時」一列；TESTDAY_04 第 3 步。CHANGE_ROWS 的設定一列。
   - 新測試 `VersionPrompt`（`test_t04_nb.py`）：用 ast 列出設定格在輸入框之前的每一個呼叫（頂層敘述，不算函式定義裡的），沒有任何寫檔的呼叫（只有建空資料夾的 `makedirs`），而且 `input(` 在 `def write_json(`、`RUN_LOG_PATH = `、`save_run_log()`、`save_config()` 之前；直接執行那一段：`y`、` Y ` 通過，`n`、空白、`yes`、`EOFError` 停下（訊息含「沒有確認」「不算一次執行」「中斷連線並刪除執行階段」），本機和 SMOKE 不問。
7. **改名成 `sw_interrupted_K` 的資料夾其實是接縫失敗時，文件沒有路可走**（user-colab 第 1 條）→ **fixed**
   - `t04_rank.py archive` 新選項 `--drive-folder sw|sw_interrupted_K`（只給 `archive`；其他寫法、其他指令都是 2）：印出的改名和 `attempt.json` 的 `drive_rename` 指向那個資料夾和它旁邊的 `test04_rank_results_interrupted_K.zip`。
   - `docs/TESTDAY_04.md`「斷線」新增一小節「notebook 說 sw_interrupted_K 不是斷線，是接縫失敗」，一步一則訊息：① 請使用者下載寫明檔名的 `test04_rank_results_interrupted_K.zip`（明寫是 COLAB_RUN 第 14 步的例外）；沒有 zip 時改下載那個資料夾的 `run_log.json`、`seam_check.json`、`config.json`，Claude 放進 `results/test04/sw/`；② `archive --reason "…" --drive-folder sw_interrupted_K`，commit；③ 照 `archive` 印的完整名稱改名、截圖，再照「重跑一次」第 3–5 點。
   - notebook ⑥-4 那則紅字補「也不要改名」「Claude 照 docs/TESTDAY_04.md「斷線」一節處理」；COLAB_RUN「出狀況時」加一列。
   - 新測試 `UnpackDrills.test_archive_a_folder_renamed_as_a_disconnection`：只有 `test04_rank_results_interrupted_2.zip` 當證據 → 封存成 `notebook_seam`、占次數，改名 `sw_interrupted_2 → sw_failed_1`、zip 也對；只有三個 JSON 放在 `sw/` → 同樣可以封存；`--drive-folder` 寫錯（`sw_failed_1`、`sw_interrupted_0`、`sw_x`）→ 2、什麼都沒搬；`status --drive-folder` → 2。
8. **已經跑完的資料夾再按一次「全部執行」，notebook 叫使用者當斷線改名**（user-colab 第 2 條）→ **fixed**
   - notebook ⑥-3 之前新增 ⑥-3a：之前最後一次有結果的執行（不算 `preflight_failed`、`interrupted`、`running`）是 `completed`、`seam_check.json` 通過、`RESULT_ROOT/test04_rank_results.zip` 存在 → 另一則訊息停下：「這個資料夾已經跑完了（第 N 次執行 completed，接縫通過），不用再跑，也不要改名…照 docs/COLAB_RUN.md 第 13–15 步下載 …/test04_rank_results.zip」。沒有 zip 時照舊是斷線的訊息。
   - `flush_drive` 那一行（「要再跑請重新「全部執行」」）是 test02 逐字沿用的函式（產生器核對逐字相同），不改；改在文件：notebook 第一格第 5 步、COLAB_RUN 第 11 步加「跑完後不要再按全部執行」，TESTDAY_04「斷線」加「最後一次執行是 completed 的資料夾不是斷線」，COLAB_RUN「出狀況時」加一列。
   - 新測試 `SeamAndFinalCells.test_run_all_again_after_completion`：完整跑到 `final_check`，再從設定格跑到 preflight → 停下，訊息含「已經跑完了」「不要改名」和 zip 的路徑，不含 `sw_interrupted`；`run_log` 是 `completed, preflight_failed`；zip 的 SHA 沒變；再按一次仍是同一則；刪掉 zip 之後回到「斷線後不接續」。
9. **COLAB_RUN「出狀況時」兩列對同一則紅字給相反的處理**（user-colab 第 4 條）→ **fixed**
   - notebook：G4 名稱變體的訊息改用另一個開頭「G4 的名稱和 03 記的不同（這次是 …，03 記的是 …）」，一般的仍以「這是登記的重跑」開頭。
   - COLAB_RUN：兩列都改成「紅字開頭是…」，名稱變體那一列排在前面。
   - `test_rerun_only_on_g4` 核對兩種開頭。
10. **重跑（R3）和彩排漏掉「不睡眠」「競賽一沒在跑」「12 碼」**（user-colab 第 5 條）→ **fixed**（文件）
    - COLAB_RUN R3：先照第 1 步確認競賽一沒有開著的執行階段、照第 4 步設成不睡眠，再從第 6 步重做；GPU 照 Claude 說的；第 9 步對照這次訊息的 12 碼。R1 改成照 Claude 寫的完整名稱改名（`sw_failed_N` 或 `sw_exempt_N`），不只給 N。
    - TESTDAY_04「重跑一次」第 2 點：訊息寫兩個完整的新名稱；第 4 點：附 12 碼，先照第 3 步第 1、3 點再核對上傳副本和競賽一。「斷線」第 4 點也附 12 碼。
    - TESTDAY_04 彩排第 3 點：給使用者步驟前先核對上傳副本 SHA、取出 12 碼寫進步驟，確認競賽一沒有開著的執行階段。

**改了哪些檔**（都不是凍結檔；SHA-256 前 12 碼，CRLF 折成 LF 後算）

| 檔 | SHA-256 |
|---|---|
| `tools/t04_rank.py`（`git()` 改 UTF-8；`archive` 的 Drive 名稱、`--drive-folder`；`drive_archives_record`） | `61fcb9168374` |
| `colab/build_test04_rank_nb.py`（版本輸入框、⑥-1 訊息、⑥-3a、⑥-4、⑥-5 訊息開頭、第一格文字、CHANGE_ROWS） | `affab5c2575b` |
| `colab/test04_rank.ipynb`（重建） | `0b329c357ae5` |
| `results/test04/notebook_diff_rank.txt`（重建） | `3d19a1f732f9` |
| `data/colab_upload/test04/test04_rank.ipynb`（`--upload-copy` 重寫；資料夾裡只有這一個檔） | `0b329c357ae5` |
| `tests/test_t04_rank.py` | `bfd0f722b28c` |
| `tests/test_t04_nb.py` | `3bc4a31b5ab0` |
| `tests/test_t04_hygiene.py` | `9e94a224bc7b` |
| `docs/TESTDAY_04.md` | `4a90dddc9a57` |
| `docs/COLAB_RUN.md`（只改 test04 一節；工作樹 CRLF，原始位元組 `971cd1bc2c2e`；對 HEAD 仍然只有加行：134 行新增、0 行刪除） | `f0675ee04f90` |
| 本檔 | — |

- 沒改：`tools/official_log.py`（`ee0ace46b4b6`）、`tests/test_t04_official_log.py`（`8109b09cd3e9`）、`tests/test_t04_dryrun.py`（`81cb12ec16a6`）、`tools/t04_fake_tests.py`、`colab/make_test04_smoke_tree.py`。

- notebook 完整的 SHA-256 是 `0b329c357ae5db582e3d0159d0015df70430034369446242563ffbaf62f2e097`，`notebook_sha256`（notebook 自己的規則）是 `ef5be6e00a4f4b1d73263e48d37719b1c67c3346e6a1016811959b16bf44cd08`，**使用者對照的 12 碼是 `ef5be6e00a4f`**（TESTDAY_04 第 3 步那一行在 Git Bash 跑過，印的就是它）。3.1 記的 `a6513f41…`／`0e7178be2955` 已經不適用。
- 產生器再跑一次，notebook 和 diff 逐位元組相同（`test_rebuild_is_byte_identical`）；上傳副本 `data\colab_upload\test04\` 裡只有這一個檔，SHA 等於 notebook。

### 4.2 驗證

以下 `PY` = `.venv\Scripts\python.exe`，在 repo 根目錄執行；執行緒上限 6，重的工作一次一件。

**完整測試**（`PY -m pytest -q -rs -p no:cacheprovider tests`，14:35–14:48，`FX3\verify_chain.sh`）

- **759 passed、66 skipped**、388 subtests，780 秒。比 3.2 的 748 多 11 個：本輪新加 12 個測試，其中 `test_ledger_and_prereg_rows_pinned` 改成只在閘門時跑，所以一般的完整測試多 1 個 skip。唯一的 warning 仍是 `test_p03_nb` 的 torch `pin_memory` 提示。
- 66 個 skip：62 個要設 `C10_SLOW`，3 個和 1.2 相同，1 個是上面那一個（skip 訊息「the 04 gate only (C10_T04_GATE=1, …) or the K5 merge (C10_K5=1)」）。

**閘門和 K5**

- `C10_T04_GATE=1 PY -m pytest -q -rs tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py`：41 passed，沒有 skip。
- `C10_K5=1 PY -m pytest -q tests/test_t04_hygiene.py`：21 passed。

**`C10_SLOW=2`**（14:48–15:29）

- 指令：`C10_SLOW=2 C10_KEEP=1 C10_DRYRUN_DIR=FX3\slow2_tree PY -m pytest -q -rs`，範圍是 `tests/test_t04_*.py` 加 `tests/test_t02_dryrun.py`。
- **267 passed、2 skipped**、386 subtests，2,458 秒（41 分；同時段自駕車、玉山兩條線也在用 CPU）。
- 2 個 skip：`test_t04_dryrun.py` 的「R20 第一次上傳前跑的 notebook」（和 3.2 相同，由 `RealMode` 涵蓋），和上面那個只在閘門跑的測試。
- 前後 repo 的 `git status` 都是同樣的 18 項。

**突變核對**（一次性 clone `FX3\clones\m5`：`9f63acd` 加上 repo 工作樹的 18 個檔，`integrator\make_merged_clone.py` 建，clone 內 commit `2461a34`；腳本 `FX3\mutate4.py`，紀錄 `FX3\mutations4.jsonl`；repo 沒有被寫）

- `data\` 用 junction 指到 repo，只讀；跑完已用 `cmd /c rmdir` 拆掉，核對過 clone 裡已經沒有 `data`。
- **12 個突變全部被抓到**（每個跑對應的測試，`-x`；改到產生器的先在 clone 裡重建 notebook）：
  - MX1 `seam_tol_sw` 永遠 1e-4（審查者的）→ `test_first_run_on_a100` 失敗。
  - `git()` 改回 `t02_rank.git` → `RealGit` 失敗。
  - `archive` 一律給 `sw_failed` → `test_archive_classes` 失敗；`drive_archives_record` 的本機名稱一律 `sw_failed` → `test_drive_archives_record_only` 失敗；`--drive-folder` 被忽略 → `test_archive_a_folder_renamed_as_a_disconnection` 失敗。
  - notebook ⑥-4 改回「所有接縫失敗的 `sw_*`」、或 `sw_failed_*` 只在有接縫紀錄時才算 → `test_exempt_archives_…` 失敗；⑥-5 對 `sw_exempt_*` 也要求 G4 → 失敗。
  - 拿掉 ⑥-3a → `test_run_all_again_after_completion` 失敗。
  - 版本輸入框改成永遠不問、或什麼都接受 → `VersionPrompt` 失敗。
  - G4 名稱變體的訊息改回「這是登記的重跑」開頭 → `test_rerun_only_on_g4` 失敗。
- 基準（不改程式）：上面這些測試在 clone 裡都通過（`test_t04_nb` 5 passed、`test_t04_rank` 9 passed）。
- 審查者的兩個 hygiene 實驗（commit 之後才測）：
  - commit 一行進 `results/prereg03/RESULTS.md`、commit 一個改動進 `results/prereg04/rank/check_report.json` → `test_frozen_results_equal_to_base` 各失敗一次。
  - commit 一列打分紀錄帳 → 一般的 `pytest tests/test_t04_hygiene.py` 通過（20 passed、1 skipped），`C10_T04_GATE=1` 時 `test_ledger_and_prereg_rows_pinned` 失敗。
  - 每次之後 `git reset --hard` 回 `2461a34`，`git status` 是空的。
- 真的 git、成功的那一條（4.1 第 1 條）：clone 裡 commit 一列 `| rev 4 |`，`check_bug_revision(4, …)` 成功，回傳那一列和 commit `fadf7909…`；之後同樣 `reset --hard`。

**R20 逐位元組不變**（spec 8.1 第 3 點）

- 不需要重做：本輪 R20 側（39 個指紋檔，含 `tools/official_log.py`）一個位元組都沒改。3.2 的比對結果仍然適用。

**notebook 冒煙 L2b**（因為產生器改了才重做；`FX3\L2b`，15:29–15:52）

| 步驟 | 指令 | 結果 |
|---|---|---|
| 建替身樹 | `PY colab/make_test04_smoke_tree.py --scratch FX3\L2b --threads 6` | OK，384 秒 |
| notebook | `PY colab/run_local_smoke.py colab/test04_rank.ipynb --scratch FX3\L2b` | OK，578 秒 |
| 核對 | `PY colab/make_test04_smoke_tree.py --scratch FX3\L2b --check --threads 6` | `ok: true`，`problems` 是空的，422 秒 |

- 接縫：通過，tol 1e-4，`bitwise_equal` true；兩臂 C0（s0、swap）最大差都是 0.0。`run_log` 是 `completed`（1 次執行）；`config.json` 的 `attempt` 1、`prior_sw_folders` 空的、`notebook_sha256` 前 12 碼 `ef5be6e00a4f`。
- zip：`b22545b8…`，349,456 bytes，40 個成員全部在 `sw/` 底下。
- 第一個程式格印 `notebook 版本：ef5be6e00a4f`（SMOKE 不問輸入框、不加「已對照」字樣）。
- 版本輸入框、重跑 G4 核對、⑥-3a 在 SMOKE 下都不會走到或不會停：分別由 `VersionPrompt`、`test_rerun_gpu_stop_is_taken`／`test_exempt_archives_…`、`test_run_all_again_after_completion` 直接執行那一段涵蓋。

### 4.3 要寫進 rev 3 的條目

本輪新增的條目。都不改來源、參數、組合、門檻；貼法照 `rev3_draft.md` 開頭的規則，由主 session 在 Rev3 階段併進草稿。

1. **第 9 條「不算一次執行」補一種**：notebook 開頭的版本確認沒通過（輸入框沒有輸入 `y`）：停在第一個程式格，還沒寫任何紀錄、還沒算，和 GPU 關卡停下同類，同一個 Drive 資料夾照常可用。對不同的 12 碼仍輸入 `y` 而跑完的，照原文判「跑的不是已 commit 的 notebook」（占次數）。這一條取代 3.3 第 8 條、3.4 第 8 條關於「開錯版本已有完成標記時照斷線改名」的說法。
2. **第 1、9、20 條（Drive 名稱和誰算重跑）**：
   - `archive` 依類別給 Drive 名稱：占次數的封存 `sw_failed_N`（zip `test04_rank_results_failed_N.zip`），不占次數的（`before_r20_u1`、`bug_revision`）`sw_exempt_N`（zip `test04_rank_results_exempt_N.zip`）。
   - notebook 的「只准重跑一次」上限和 `ATTEMPT`：`sw_failed_*` 一律算（包括 Drive 上接縫本身通過、本機 `apply` 才判失敗的），`sw_exempt_*` 不算，其他改過名而且自己有接縫類失敗紀錄的也算。
   - 「登記的重跑只在 G4」只在有一個占次數的封存（Drive 上是 `sw_failed_*`）時適用；三處（notebook、`unpack`、`official_log`）同一個判定。第 1 條「照第 9 條的做法再跑一次」讀成：封存和改名照第 9 條，GPU 照第一次的規則（G4 優先、A100 備用），因為那一次不是第 9 條那一次重跑；第 20 條同。
   - **請主 session 決定**：另一種讀法是「不占次數的封存之後也只用 G4」（比較嚴，user-colab 審查者建議；可以擋住「接縫在 G4 失敗、因 R20 上傳前跑而不占次數、下一次改用容許差較寬的 A100」這種情況）。要採用的話，notebook ⑥-5、`t04_rank.py unpack`（`g4_rec` 的條件）、`official_log.py`（`attempt.number` 的條件，只能加行）三處要一起改，並在 rev 3 寫明。
3. **第 9 條「斷線」補兩種**：
   - 改名成 `sw_interrupted_K` 的資料夾裡已有接縫類失敗紀錄（notebook preflight 會停下說「不是斷線」）：照接縫失敗處理。證據是 `test04_rank_results_interrupted_K.zip`，沒有 zip 時是那個資料夾的 `run_log.json`、`seam_check.json`、`config.json`；`archive --drive-folder sw_interrupted_K`，Drive 上把那個資料夾改名成 `archive` 印的名稱。
   - 最後一次有結果的執行是 `completed`、接縫通過、結果 zip 在的資料夾不是斷線（跑完後又按了全部執行）：notebook 用另一則訊息停下，不改名，直接用那份 zip。
4. **第 19 條（閘門）**：`test_t04_hygiene.py` 的打分紀錄帳列數和 PREREG_01–03 `| rev N |` 列核對（`test_ledger_and_prereg_rows_pinned`）只在 `C10_T04_GATE=1`（`docs/TESTDAY_04.md` 的閘門指令）或 `C10_K5=1` 時跑。一般的完整測試（`docs/TESTDAY.md` 11/04 前的「全綠」就緒清單）略過它，所以別的登記依規打分多出來的列不會讓 R20 的就緒閘門變紅；04 的閘門照樣逐列核對、不自動算過。這一條修改 3.3 第 5 條「`test_t04_hygiene.py` 另要求…」為無條件的說法。另外：`results/prereg01`–`prereg04` 對 6ec5e68 不能有任何差異，已 commit 的也算（`git diff 6ec5e68`），也不能有未追蹤的檔。
5. **第 13 條的演練清單補**：`--bug-revision` 的核對用真的 git 讀中文的 PREREG_04（`t04_rank.py` 用 UTF-8 解碼；以前在 cp950 的電腦一定以內部錯誤結束）；第一次在 A100 跑 → test04 的 tol 1e-3、`apply` 0，notebook 記 1e-4 → 1；`results/prereg01`–`04` 已 commit 的改動 → hygiene 失敗；`sw_exempt_*` 不算次數、`sw_failed_*` 一律算；跑完後再按全部執行 → 另一則訊息、不改名；版本確認在任何寫入之前；`archive --drive-folder` 寫錯 → 2。
6. **只是實作說明**：
   - `attempt.json` 多一欄 `drive_rename`（原資料夾和 zip、新名稱）；`archive` 印 `Drive: … -> …` 和下一次的 GPU 規則。
   - `unpack_sw.json` 的 `drive_archives` 對照包含 `sw_exempt_*`。
   - notebook 設定格的版本輸入框（只在 Colab 上）；最後一行的版本只記錄。
   - G4 名稱變體的停下訊息開頭改成「G4 的名稱和 03 記的不同」。
7. **程序（只改文件）**：COLAB_RUN R1 照 Claude 寫的完整名稱改名；R3 先確認競賽一沒在跑、電腦不睡眠，再對照這次的 12 碼；彩排也要附 12 碼、先核對上傳副本 SHA；第 11 步「跑完後不要再按全部執行」；「出狀況時」各列用紅字開頭分辨。

### 4.4 還沒驗到的地方與風險

1. **版本輸入框從沒在 Colab 上跑過**。`input()` 在 Colab「全部執行」時會在格子下面跳出輸入框、等使用者輸入，這是 Colab 的一般行為，但這份 notebook 還沒在真的 Colab 上走過這一段（本機冒煙是 SMOKE，不問）。彩排時請使用者注意：跳出輸入框、輸入 `y` 之後才往下跑；彩排結果記下輸入框出現的時間點。
2. **上傳副本和 12 碼又變了**：`data\colab_upload\test04\test04_rank.ipynb` 現在是 `0b329c35…`，使用者對照的 12 碼是 `ef5be6e00a4f`。主 session 若已經請使用者把較早的副本放上 Drive（例如為了彩排），要換成這一份並重新核對 SHA；commit 後照 TESTDAY_04 第 3 步那兩行再核對一次。
3. **正式模式端到端**：round 3 correctness 審查者在一次性 clone 用 round 2 的程式實跑過正式模式（`opp`、`unpack`、`apply`、封存、G4 重跑、`official_log prepare`）。本輪改了 `archive`（Drive 名稱、`--drive-folder`）和 `git()`，都有 dry-run／RealMode 的測試，`--bug-revision` 的成功路徑也在 clone 用真的 git 跑過；但沒有再完整重跑一次正式模式的串接。建議在 rev 3 commit 之後、合併之前照 correctness 審查者的腳本（`review3-correctness\real_*.log` 那一套）再跑一次，另加：`archive --bug-revision 4`（clone 裡先 commit 一列 rev 4）正式模式結束碼 0、印 `sw_exempt_N`。
4. **G4 規則在不占次數的封存之後的讀法**（4.3 第 2 條）：現在三處都是「只有占次數的封存之後才只收 G4」。主 session 若選較嚴的讀法，三處要一起改（`official_log.py` 只能加行）。
5. **Drive 名稱仍然只記錄、不擋**：使用者把不占次數的封存改成 `sw_failed_N`，notebook 會多算一次、也會要求 G4（比較嚴，不會交錯檔）；反過來把占次數的封存改成 `sw_exempt_N`，notebook 會允許 A100，本機 `unpack` 會以「登記的重跑只在 G4」拒收（2），浪費一次執行。防護是 TESTDAY_04「重跑一次」第 2 點：訊息寫兩個完整的新名稱，Claude 看過改名前後的截圖才往下做；`unpack` 對不上時印 WARNING。
6. **`test_ledger_and_prereg_rows_pinned` 現在要靠 `C10_T04_GATE=1`**：少打這個變數時它會被略過而不是失敗。TESTDAY_04 第 1 步的指令加了 `-rs`，閘門表要求確認它沒有出現在 skip 清單裡。
7. **⑥-3a 靠結果 zip 還在**：使用者已經把 `test04_rank_results.zip` 改名或刪掉時，notebook 會回到「斷線後不接續」的訊息。TESTDAY_04「斷線」寫了：最後一次執行是 `completed` 的資料夾不是斷線，Claude 先看 `run_log`／截圖再決定。
8. **L2 沒有重跑**，只重跑了 L2b。本輪 notebook 改的是設定格的輸入框（SMOKE 不走）、preflight 的 ⑥ 和訊息文字，03→test02→test04 的串接沒有變。
9. **repo 工作樹的 18 個新檔或改動檔都還沒 commit**（本輪只改其中的檔，沒有新增檔）。`FX3\clones\m5` 的 `data` junction 已拆，可以整個刪掉；`FX3\slow2_tree`、`FX3\L2b` 可以重建。

## 5. rev 3 check（定稿對程式的核對與更正）

- 輸入：核對者對 `rev3_final.md`、`docs/TESTDAY_04.md` 和程式列出的 13 條不一致（都是 minor）。逐條對 `tools/t04_rank.py`、`tools/official_log.py`、`colab/test04_rank.ipynb`、`tests/test_t04_official_log.py` 核對，13 條都屬實。
- 原則：文字錯的改 `rev3_final.md`（`rev3_draft.md` 的內文由它重建）和 `docs/TESTDAY_04.md`；程式和登記的意思不合的，只改 `tools/t04_rank.py`，並加測試。來源、組合、參數、閘門、H4R、截止點、門檻都沒動，沒有改變任何驗證集的數字。

### 5.1 每一條的處理

| # | 核對者說的 | 核對結果 | 處理 |
|---|---|---|---|
| 1 | R20 先的核對排在接縫類核對之後，R20 上傳前跑、又有接縫類紀錄的 zip 給 1 不是 2 | 屬實；`cmd_unpack` 的順序就是規格 3.1 的順序；`archive` 對這種 zip 一律先判 `before_r20_u1` | **改文字**：第 1 條寫明「先查到接縫類失敗時給 1，一樣不用，`archive` 一律先判 `before_r20_u1`，做法相同」；TESTDAY_04 第 4 步結束碼 1 那一列和「R20 上傳前跑的」那一列。程式不動 |
| 2 | `check_submission` 沒過時只刪 `submission_rank.csv`，`pred_B.csv`（和被拒的交件檔逐位元組相同）等四個檔留下 | 屬實 | **改程式**：刪掉五個測試集輸出（新常數 `TEST_OUTPUTS`），`attempt_verdict.json` 的 `details.removed` 記刪了哪些；`seam_val/` 照留。第 8 條、11.6 d、TESTDAY_04 第 5 步 |
| 3 | `--bug-revision` 不解除「R20 核對不了」和「repo 的 notebook 沒 commit／不一致」的拒絕，第 9 條讀起來像全部都能解除 | 屬實（兩者直接 `refuse()`） | **改文字**：第 9 條拆成「修訂可以解除的三種」和「修訂不解除的三種」；TESTDAY_04「重跑一次」第 1 點 |
| 4 | 計數檔和 `count_report.json` 不同：`opp` 給 2 不是 4；`apply` 請使用者「重跑 opp」，但 `opp` 有登記的判定之後要修訂才能重跑 | 屬實；另外計數檔不見時 `opp` 以「unexpected internal error」（2）結束 | **改程式**：`load_tables` 先查計數檔在不在，`po.registered_counts` 的錯誤轉成 2 並附還原說明（`MSG_COUNTS_RESTORE`：同 SHA 的 `design_scratch` 副本、第 17 條、TESTDAY_04 第 1 步）；`check_opp_report` 單獨核對計數檔（2，同一則說明），其他不符的訊息改成「從 git 還原再跑 apply；`opp` 有登記的判定之後重跑要修訂；test02 依 PREREG_02 重跑會讓 04 停下」。第 2、9、17 條、11.6 k、TESTDAY_04 第 2、5 步 |
| 5 | `status` 的回傳值多一個 `rank.prereg04` | 屬實（`TagVersusNew.compare` 比對前先拿掉） | **改文字**：第 13 條 |
| 6 | `final` 的「這一筆還沒上傳」在給了 `--not-04-reason` 時也印 | 屬實；另外 log 沒有 04 紀錄卻給了 `--not-04-reason` 時，判定結果那一行的最終選擇印成 `None` | **改文字**：11.8 x（兩行的條件，和 `None` 只是顯示）。`official_log.py` 不動（只准加行，不是這次可以改的 t04 新檔） |
| 7 | 「final 改 R20…」只在沒有任何一筆 04 符合時印 | 屬實（`redo_r20 = not ok_ids and (last_bad or final_bad)`） | **改文字**：第 12 條 `status` 那一點，另寫還有別的 04 符合時印什麼 |
| 8 | 第 22 條少列兩種讀法 | 屬實：notebook 接縫以 C0 判（`seam_check.passed` 只看 key 和兩個 C0 的最大差）；遮罩 `parsed & ~replay_failed` 也用在 O4／T4 的 κ 子題 | **改文字**：第 22 條加兩點 |
| 9 | 開頭「都只是寫具體或更嚴」不對 | 屬實：不占次數的封存之後可以多出判接縫的執行；第 19 條讓 5.1 的硬規則測試在 PREREG_02 修訂下失敗時 04 照常做 | **改文字**：開頭改成「寫具體、更嚴，或在第 22 條揭露的讀法」；第 22 條加這兩點，和「以上讀法都不動來源、組合、參數、閘門、H4R、截止點或門檻」 |
| 10 | K6 的「MANIFEST `kind official`」沒有工具讀 | 屬實；已 commit 的 notebook 非 SMOKE 只收 `official`／`rehearsal`，`REHEARSAL = TEST_KIND == 'rehearsal'`，所以正式模式要求的 `smoke false`、`rehearsal false` 就等於 `kind official` | **改文字**：第 11 條寫成旗標加「和本機 MANIFEST、test02 的測試包相同」，並寫明工具不另外讀 `kind`。不加程式核對：第 13 條的正式模式演練用 `t04_fake_tests.py` 的假輸出（config 記 `kind rehearsal`），加核對會連帶改演練的做法；登記的科學不變 |
| 11 | Drive 剩餘空間查不到時只印提醒 | 屬實，和 `colab/test02_rank.ipynb` 同一段 | **改文字**：第 4 條 preflight（「查不到時只印提醒、照常往下，同 test02」）。notebook 不動 |
| 12 | `opp` 是本機漂移（4）時 `archive` 也用「04 已經不用」拒絕 | 屬實；而且帶 `--bug-revision` 時這個拒絕會被解除，漂移報告的時間還被當成「被推翻的紀錄」 | **改程式**：漂移（`verdict_kind == "drift"`）用另一則訊息立刻拒絕（`MSG_OPP_DRIFT_ARCHIVE`，修訂不解除，要先重跑 `opp`）；登記的判定照舊可由修訂解除（訊息改成「opp's registered decision is 'no 04'」）。第 9 條、11.6 h、TESTDAY_04「重跑一次」第 1 點 |
| 13 | TESTDAY_04 第 2、5 步的表 | 屬實：`.venv` 的 4 在 `common_prereqs` 停下，比率表的 4 也在寫檔之前，都沒有報告；`apply` 的 4 也包括 `check_submission` 沒過 | **改文件**：第 2 步結束碼 4、2 兩列；第 5 步結束碼 4、2 兩列和「輸出」那一段 |

### 5.2 改了哪些檔

- `tools/t04_rank.py`：`61fcb9168374` → `36842148f2a5`（LF SHA-256 前 12 碼）。新增 `MSG_COUNTS_RESTORE`、`MSG_OPP_DRIFT_ARCHIVE`、`TEST_OUTPUTS`；改 `load_tables`、`check_opp_report`、`Apply.go` 的 `check_submission` 段、`cmd_archive` 的 `opp` 段。
- `tests/test_t04_rank.py`：新增 `Rules.test_load_tables_counts_refusal`、`ApplyDrills.test_counts_file_changed_after_opp`、`ApplyDrills.test_check_submission_failure_leaves_no_test_output`；`UnpackDrills.test_archive_refusals` 加「`opp` 漂移，有、沒有修訂都拒絕，什麼都沒搬」。
- `results/test04/design/rev3_final.md`：上面各條，和開頭的程式版本（`t04_rank.py` 的 SHA、「`build_review.md` 第 5 節那一版」）。`rev3_draft.md` 的內文由 `rev3_final.md` 的兩部分重建（逐字相同，只留草稿自己的開頭，「定稿依據」那一行多寫第 5 節）。
- `docs/TESTDAY_04.md`：第 2、4、5 步的表，「重跑一次」第 1 點，第 5 步的輸出說明。
- 沒動：`tools/official_log.py`（`ee0ace46b4b6`）、`colab/build_test04_rank_nb.py`（`affab5c2575b`）、`colab/test04_rank.ipynb`（`0b329c357ae5`，上傳副本和 12 碼 `ef5be6e00a4f` 都不用換）、`spec.md`、`understand_*.md`、所有凍結檔。

### 5.3 驗證

`PY` = `.venv\Scripts\python.exe`，在 repo 根目錄執行，`OMP_NUM_THREADS=6 MKL_NUM_THREADS=6`。

- 本輪相關的測試（`PY -m pytest -q -p no:cacheprovider tests/test_t04_rank.py -k "counts or check_submission_failure or archive_refusals or load_tables or o4_file_changed or opp_says or swa_equal or venv_drift or apply_rerun or archive_classes"`）：**11 passed**、7 subtests，827 秒（同時段別的線也在用 CPU）。
- 完整測試（`PY -m pytest -q -rs -p no:cacheprovider tests`，16:54–17:27）：**762 passed、66 skipped**、390 subtests，1,892 秒（同時段別的線也在用 CPU，比 4.2 的 780 秒慢）。比 4.2 的 759 多 3 個（本輪新加的 3 個測試），subtests 多 2 個（計數檔測試的兩種情況）。66 個 skip 和 4.2 相同；唯一的 warning 仍是 `test_p03_nb` 的 torch `pin_memory` 提示。
- 閘門（`C10_T04_GATE=1 PY -m pytest -q -rs tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py`）：**41 passed**、33 subtests，沒有 skip。
- 真的 `load_tables()`（單元測試換掉的那一個）的成功路徑另外直接跑一次：比率表 SHA 等於 `TABLES_SHA`、計數檔 `f6c6debe9063`、`counts_sha_now()` 等於 `COUNTS_SHA`，33 秒。`apply` 的 `check_opp_report`（計數檔的核對改了順序）在完整測試的 `DryRunPipeline` 裡走過成功路徑。
- 沒有重跑 `C10_SLOW=2`、L2b、L2：本輪沒有動 notebook、產生器和 `official_log.py`；`t04_rank.py` 在成功路徑上只多了計數檔的存在核對和錯誤轉換（上一點），其餘改動都在拒絕或停下的路徑上。

### 5.4 還沒驗到的地方

1. 第 13 條的正式模式演練（一次性 clone、已 commit 的程式）還沒跑；本輪改的三處（`archive` 對 `opp` 漂移的拒絕、計數檔的訊息、`check_submission` 沒過時刪檔）只有 dry-run 的測試。照 4.4 第 3 點，在 rev 3 和程式 commit 之後、合併之前一起跑。
2. 11.8 x 記的 `final` 顯示 `None`（沒有 04 紀錄卻給 `--not-04-reason`）沒有修：只是顯示，而且 `official_log.py` 只准加行。測試日不會這樣下指令（TESTDAY_04 第 7 步只在有 04 紀錄時用 `--not-04-reason`）。
3. repo 工作樹仍未 commit（同 4.4 第 9 點）；本輪沒有新增檔。

## 6. K5 gate round 1 and fixes

- 修正者（K5 閘門第 1 輪），2026-09-30 晚間。分支 `p04-testday`，起點 HEAD `62ca557`（main 仍是 `6ec5e68`）。本節和下面的修改一起 commit 在 `p04-testday`（commit 編號見那個 commit 本身；沒有動 main、沒有 amend）。
- 輸入：gate 四個檢查。`fixes` 沒過（1 條 major），`realmode`、`r20-suites`、`L2` 通過；另有 8 條 minor，其中 2 條是同一件事（`archive` 印的下一次 GPU 規則），合併後 7 件。
- 每一條都先對照程式核實。程式的問題 3 件（1 major、2 minor）都屬實、都修了；1 件（CRLF）屬實，但 gate 建議的做法會出錯，改用逐檔的做法；另外 3 件是檢查者的環境（記憶體不足、CPU 冒煙的原生崩潰、負載造成的時間），程式不改，只補文件和紀錄。
- 來源、組合、參數、閘門、H4R、截止點、門檻都沒動，沒有改變任何驗證集的數字。PREREG_04 只改第 11 節裡和程式不一致、或寫得有歧義的程序文字；第 0–10 節和 rev 3 那一列（`(commit)` 還沒填）都沒動。
- 凍結檔都沒動：`tools/official_log.py` 仍是 `ee0ace46b4b6`（對 tag 仍然只有 652 行新增、0 行刪除）；notebook 和產生器沒改，所以不用重建，上傳副本和 12 碼 `ef5be6e00a4f` 都不用換。
- 沒有開答案檔，沒有跑 `tools/score.py`，沒有用 GPU；執行緒上限 6；開跑前看過可用記憶體（20 GB）和 CPU 負載（16%），同時段只跑一件重工作。
- 暫存都在 `…\scratchpad\p04gate\fixer1\`，以下簡寫成 `FX4`；gate 自己的紀錄在 `…\scratchpad\p04gate\{fixes,realmode,r20,L2}\`。

### 6.1 每一條的處理

| # | 檢查、嚴重度 | 檢查者說的 | 核對結果 | 處理 |
|---|---|---|---|---|
| 1 | fixes，major | `freshness` 只看 `run_log.runs[0].start`。TESTDAY_04「notebook 說 sw_interrupted_K 不是斷線」的流程裡，新的 `sw` 在封存前已經有一次 preflight 停下（`preflight_failed`），登記的重跑又在同一個 `sw` 跑，所以 `runs[0]` 一定早於封存，重跑一定被 `unpack` 以 2「這不是重算的結果」拒收；TESTDAY_04 對這個 2 的做法（刪 zip、等新的 zip）是死路 | 屬實。notebook 的 `preflight_stop` 把那一次記成 `preflight_failed` 存回同一份 `run_log.json`；`freshness` 用的是 `runs[0]`，R20 先後用的 `producing_runs` 不同。本輪另外用舊的 `freshness` 跑新加的測試（`FX4\old_freshness_check.py`）：失敗，訊息正是「`run_log.runs[0].start 2026-11-06T09:40:00+08:00 is not after the archive attempt_1 (2026-11-06T10:00:00+08:00)`」 | **改程式**：`freshness` 改成「算出輸出的執行」（`producing_runs`，第 1 條的定義）的每一次開始時間都要晚於最近一次封存；沒有任何一次執行時也是 2。`unpack_sw.json` 的 `freshness` 多記 `producing_runs` 和 `rule`，`runs0_start` 只記錄。zip 的 SHA-256 和 seam_val 完成標記時間的核對照舊，所以舊 zip 仍然擋得住（最後一次執行一定早於封存）。**改文件**：PREREG_04 第 9 條、11.6 g；TESTDAY_04 第 4 步結束碼 2 那一列、「重跑一次」第 5 點、「不是斷線」一節第 3 步。**測試**：`RealMode.test_rerun_after_an_interrupted_folder_archive`（`archive --drive-folder sw_interrupted_1` → 0；同一個 `sw` 的重跑 `[1 preflight_failed 09:40, 2 completed 11:00]`、封存 10:00 → 0，印「這是登記的重跑」、G4 核對適用；算出輸出的執行早於封存的兩種 → 2） |
| 2 | fixes minor、realmode minor 1（同一件事） | 已有 1 個占次數的封存，之後又做一個不占次數的封存（`bug_revision`）：`archive` 印「GPU 照第一次的規則：G4 優先、A100 備用」，但 notebook、`unpack`、`official_log` 三處都只收 G4；PREREG_04 第 1、20、22 條和 TESTDAY_04 第 173、182 行也寫成無條件的「照第一次的規則」 | 屬實（那一行只看這一次的 `counts`，沒看之前的 `counted`）。三處工具彼此一致，錯的只是印出來的那一行和文件 | **改程式**：下一次執行的說明改看「這次封存之後還有沒有占次數的封存」（`counts or counted`）：有就印「不占重跑次數（…），但已有 1 個占次數的封存 […]（Drive 上有 sw_failed_*）：下一次仍是登記的重跑，只用 G4」；沒有才印第一次的規則。`attempt.json` 多一欄 `next_run`（`registered_rerun_g4_only`、文字、規則），dry-run 也寫。**改文件**：PREREG_04 第 1、20、22 條、11.6 h；TESTDAY_04 第 4 步「R20 上傳前跑的」那一列、「重跑一次」第 1、7 點。**測試**：`RealMode.test_next_run_after_an_uncounted_archive_that_follows_a_counted_one`（占次數的 `attempt_1` 之後 `archive --bug-revision 4` → 0、`sw_exempt_2`，印「下一次仍是登記的重跑，只用 G4」、不印「A100 備用」；接著 A100 的重跑 zip → `unpack` 2「登記的重跑只在 G4」） |
| 3 | realmode minor 2 | 登記的重跑又是接縫類失敗時，`unpack`／`apply` 仍印「照 docs/TESTDAY_04.md「重跑一次」先 archive」，而 `archive` 一定拒絕 | 屬實（`unpack` 在紀錄核對前已算好 `counted`；`apply` 的 `check_unpack_record` 也回傳 `counted`） | **改程式**：新常數 `MSG_SEAM_LAST`（「這是登記的重跑（已有 1 個占次數的封存），沒有第三次：…不 archive，照「重跑一次」第 6 點停下查清楚，修訂前不上傳 04；只有先 commit 指出程式錯誤的 PREREG_04 修訂時，才 archive --bug-revision N」）和 `seam_msg(ctx)`；`unpack`、`apply` 在紀錄核對前把 `counted` 放進 `ctx["counted_archives"]`，原本 13 處的 `MSG_SEAM` 和本機接縫沒過的訊息都改用它。第一次的訊息不變。所有模式都一樣（`archive` 的拒絕也不分模式）。**改文件**：11.6 f；TESTDAY_04 第 4、5 步結束碼 1 那一列、「重跑一次」第 6 點。**測試**：`UnpackDrills.test_seam_failure_of_the_registered_rerun_has_no_third_run`（第一次 → 1、`MSG_SEAM`；有占次數的封存 → 1、`MSG_SEAM_LAST`、沒有「先 archive」、`archive` 真的拒絕；`apply` 的 notebook 接縫沒過 → 1，`attempt_verdict.json` 的 `message` 也是它） |
| 4 | realmode minor 3 | 這台電腦 `core.autocrlf=true`，重新 checkout 的 `results/prereg02` 會變成 CRLF，`test_t02_hygiene.py` 的 `Pins` 比原始 SHA，第 1 步的閘門會假性變紅；建議 `.gitattributes` 加 `results/prereg02/** -text` | 問題屬實，但建議的做法會出錯：`git ls-files --eol` 顯示 `results/prereg01`–`04` 在這個工作樹有 67 個 `w/crlf`、5 個 `w/mixed` 的文字檔（index 都是 LF），整個資料夾設 `-text` 會讓它們變成已修改，`test_frozen_result_trees_unchanged`、`git status` 的閘門反而會紅 | **只加行，逐檔**：`.gitattributes` 對 `test_t02_hygiene.py` 的 `PINS` 裡 7 個文字檔（`results/prereg02/rank/{frozen_config.json, rank_val_probs.csv, pred_B.csv}`、`results/prereg02/player/{frozen_config.json, guard_calibration.json, sr_report.json, player_val_top5.csv}`）設 `-text`；兩本 `colab/*.ipynb` 本來就是 `-text`，npz 是二進位。這 7 個檔在工作樹本來就是 LF，`git status` 仍然是空的。其他閘門測試比的是 CRLF 折成 LF 的 SHA，不受影響。**改文件**：PREREG_04 第 18 條；TESTDAY_04 第 1 步「防誤改測試」那一列加一種非凍結檔被改的失敗（只有 `Pins` 失敗而且 `git status` 空：先 `git -c core.autocrlf=false checkout -- results/prereg02` 再跑、記進 RESULTS）。驗證見 6.3 |
| 5 | r20-suites minor 1 | 第一次 `C10_SLOW=2` 完整測試有 8 個 MemoryError | 檢查者的環境（同時段另一個 gate 代理在跑 L2，別條線也在跑，可用虛擬記憶體最低到 5.7 GB）；記憶體恢復後這 8 個 passed，第二次完整測試 exit 0。程式不改，也不改成記憶體不足就 skip | **改文件**：spec 8.4（單獨跑、開跑前確認可用虛擬記憶體至少約 12 GB、出現 MemoryError 時先重跑失敗的、再重跑一次完整的，兩次都記）、spec 10.6 第 2 點；PREREG_04 11.9 ab。紀錄在 6.4 |
| 6 | r20-suites minor 2 | `build_review.md` 沒有已 commit 的 `62ca557` 上的完整 `C10_SLOW=2`、R20 兩個 clone 比對的紀錄（K5 核對表那一項沒有依據） | 屬實（第 2–5 節的 `C10_SLOW=2` 只跑 `tests/test_t04_*.py` 加 `test_t02_dryrun.py`，R20 clone 比的是 clone 裡 commit 的工作樹版本） | **補紀錄**：6.4 記下 gate 在 `62ca557` 上的結果 |
| 7 | L2 minor 1 | L2 程序沒寫 notebook 步驟在原生層當掉（exit 139、沒有 traceback）時怎麼接 | 檢查者的環境（CPU 冒煙時可用記憶體 5.3 GB、別的工作在跑）；移開那一臂的輸出重跑後通過，權重 `ec62f8ba…` 和之前、和 integrator 9/30 早上的 L2 相同。測試日的路徑（Colab、GPU）不經過這裡 | **改文件**：spec 8.5 L2 那一點（開跑前記憶體 ≥ 8 GB、沒有別的重 CPU 工作；崩潰時移開哪兩個資料夾、從那一步重跑、記進紀錄）；本檔 1.2 L2 紀錄後面同樣一段 |
| 8 | L2 minor 2 | L2 的時間只有安靜時的數字 | 屬實（gate 的步驟時間 84 分鐘，含一次崩潰重跑的牆鐘 98 分鐘） | **改文件**：PREREG_04 11.9 ab（L2 約 50–100 分鐘，視同時段負載而定；`C10_SLOW=2` 安靜時 27 分半、有別的工作時 56–72 分鐘）；spec 8.5 L2、8.6 的重跑規劃 |

### 6.2 改了哪些檔

- `tools/t04_rank.py`：`36842148f2a5` → `801b08820ccc`（LF SHA-256 前 12 碼）。改 `freshness`；新增 `MSG_SEAM_LAST`、`seam_msg`；`cmd_unpack`、`Apply.go` 設 `ctx["counted_archives"]`；13 處 `MSG_SEAM` 和本機接縫的訊息改用 `seam_msg(ctx)`；`cmd_archive` 的下一次執行說明和 `attempt.json` 的 `next_run`；檔頭說明。
- `tests/test_t04_rank.py`：`73a62273e07f` → `8afc9a031119`。新增 6.1 第 1、2、3 條的三個測試。
- `.gitattributes`：只加 11 行（4 行註解、7 行 `-text`），既有行沒動（`test_gitattributes_gitignore_only_gain_lines` 核對）。
- `docs/PREREG_04_OPPONENT.md`：只改第 11 節：第 1、9、18、20、22 條，11.6 f、g、h，11.9 ab。
- `results/test04/design/rev3_final.md`：第 2 部分同樣的改動（和 PREREG_04 的第 11 節逐字相同，本輪核對過），開頭加一行說明；`rev3_draft.md` 不再更新。
- `docs/TESTDAY_04.md`：第 1 步閘門表、第 4 步結束碼 1、2 的三列、「重跑一次」第 1、5、6、7 點、「不是斷線」一節第 3 步、第 5 步結束碼 1 那一列。
- `results/test04/design/spec.md`：8.4、8.5 L2、8.6、10.6 第 2 點。
- `results/test04/design/build_review.md`：1.2 L2 紀錄後的一段、本節。
- 沒動：`tools/official_log.py`（`ee0ace46b4b6`）、`colab/build_test04_rank_nb.py`（`affab5c2575b`）、`colab/test04_rank.ipynb`（`0b329c357ae5`；上傳副本 `data\colab_upload\test04\test04_rank.ipynb` 的 SHA-256 仍然相等，資料夾裡只有這一個檔）、`tools/t04_fake_tests.py`、`colab/make_test04_smoke_tree.py`、`docs/COLAB_RUN.md`、所有凍結檔。

### 6.3 驗證

`PY` = `.venv\Scripts\python.exe`，在 repo 根目錄執行，`OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6`。開跑前可用虛擬記憶體 20 GB、CPU 負載 16%，同時段沒有別的重工作。

- 本輪相關的測試（`PY -m pytest -q -p no:cacheprovider tests/test_t04_rank.py -k "registered_rerun_has_no_third_run or interrupted_folder_archive or uncounted_archive_that_follows or renamed_as_a_disconnection or unpack_real_ok_then_freshness or before_r20_first_upload or a_stop_before_r20 or seam_failed or second_counted or archive_classes or local_seam_fails"`）：**12 passed**、10 subtests，124 秒。
- 回歸核對：
  - 把 `62ca557` 的 `freshness` 換回去跑新的 `test_rerun_after_an_interrupted_folder_archive`（`FX4\old_freshness_check.py`）：**失敗**，`unpack` 結束碼 2，訊息「`run_log.runs[0].start 2026-11-06T09:40:00+08:00 is not after the archive attempt_1 (2026-11-06T10:00:00+08:00)`」。新測試抓得到 gate 找到的問題。
  - gate 的 `fixes\fresh_exp.py`（直接呼叫 `freshness`）在新程式上：problems `[]`，`producing_runs` 只有 run 2；對照組（新的資料夾）也是 `[]`。
- 完整測試（`PY -m pytest -q -rs -p no:cacheprovider tests`，21:07–21:28）：**765 passed、66 skipped**、392 subtests，1,249 秒，exit 0。比 5.3 的 762 多 3 個（本輪新加的 3 個測試），subtests 多 2 個（新測試的兩種情況）。66 個 skip 和 5.3 相同；唯一的 warning 仍是 `pin_memory`。
- 閘門：
  - `C10_K5=1 PY -m pytest -q -rs -p no:cacheprovider tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py`：**41 passed**、37 subtests。
  - `C10_T04_GATE=1`，同三個檔：**41 passed**、33 subtests，沒有 skip。
  - `C10_K5=1 PY -m pytest -q -rs -p no:cacheprovider tests/test_t04_hygiene.py`：**21 passed**。
- `git status --porcelain`：只有本輪改的 8 個檔（commit 前）；`results/prereg0*` 沒有出現（加了 `-text` 的 7 個檔在工作樹本來就是 LF）。
- `C10_SLOW=1 PY -m pytest -q -rs -p no:cacheprovider tests/test_t04_dryrun.py tests/test_t04_rank.py`（`C10_DRYRUN_DIR=FX4\slow1_tree`；04 的端到端 dry run，含封存和登記的重跑）：**124 passed、1 skipped**、154 subtests，825 秒，exit 0。skip 是 `t04_dryrun` 的「R20 U1 之前」（要正式模式，`test_t04_rank.py` 的 `RealMode` 涵蓋）。
- `.gitattributes` 的效果（`FX4\crlf_clone.sh`，紀錄 `FX4\crlf_clone.log`）：
  - clone1：用這台電腦預設的 `core.autocrlf=true` 從 repo clone `62ca557`，7 個釘選檔是 `w/crlf`；`C10_T04_GATE=1 … tests/test_t02_hygiene.py -k test_pinned_files` **1 failed**（重現 realmode 017）。
  - 在 clone1 commit 本輪的 `.gitattributes`，再從 clone1 重新 clone 成 clone2（新的 checkout，同樣 `autocrlf=true`）：7 個檔是 `w/lf attr/-text`，其他 prereg02 文字檔照舊是 `w/crlf`，`git status` 0 行；三個閘門測試（`C10_T04_GATE=1`）**41 passed**、33 subtests，跑完 `git status` 仍是 0 行。

### 6.4 K5 閘門第 1 輪在 `62ca557` 上的結果（K5 核對表第 2 項的依據）

gate 的四個檢查都在已 commit 的 `62ca557` 上跑（2026-09-30 17:40–20:50），repo 的 `git status` 前後都是空的，`data\` 的清單前後相同；沒有開答案檔，沒有跑 `tools/score.py`，沒有用 GPU，執行緒上限 6。紀錄在 `…\scratchpad\p04gate\`。本輪（6.1–6.3）的修改在這之後，所以下面是修改前那一版的結果；哪些要在本輪的 commit 上重跑，見 6.5。

**r20-suites**（`p04gate\r20`）：
- R20 逐位元組核對（spec 8.1 第 3 點）：兩個一次性 clone（`git clone -c core.autocrlf=false --no-checkout`，再 `checkout --detach`）：`main_6ec5e68`（`6ec5e689`）和 `branch_62ca557`（`62ca5576`），`data` 用 junction。依序跑 `tests/test_t02_dryrun.py -k FromValPipeline`（`C10_SLOW=1 C10_KEEP=1`，`C10_DRYRUN_DIR` 相同）：main 15 passed（524 秒），branch 15 passed（633 秒）。
  - 兩棵樹都等於 `r20_fake_golden.json`：7 個 CSV（`submission_rank.csv` `d9b9801b…`，seam `a8370043…`、`863a6258…`）、兩份 `decide_summary.json` 的正規化 SHA（`a14374bd…`、`b1b3f3a0…`）、`opening_test.npz` 的 9 個陣列。
  - 兩棵樹彼此比：128 個檔，82 個逐位元組相同，沒有只在一邊的檔；46 個不同的檔只差時間、`runtime_s`／`seconds`、`git_head`、路徑，和含時間戳記的檔（MANIFEST、測試 zip、`t1.json`、`apply_report`）的 SHA，和 fixer2 那次 main 對 merged4 的比對是同一組檔。
- 39 個指紋檔：`git diff --numstat r20-testday-pre04 62ca557 -- <39 檔>` 只有 `652 0 tools/official_log.py`（tag 物件 `f718601` → `cdec913`）；tag 版 594 行照順序都在新版的 1,246 行裡；其餘 38 個檔的 LF SHA 等於指紋；golden 的 13 個 blob 在 `62ca557`、`6ec5e68` 都不變。
- 完整測試：762 passed、66 skipped、390 subtests，1,726 秒，exit 0（和 5.3 相同）。
- 閘門：`C10_K5=1` 三個 hygiene 檔 41 passed（37 subtests）；`C10_T04_GATE=1` 41 passed（33 subtests）；兩個一起設 41 passed；`C10_K5=1` 只跑 `test_t04_hygiene.py` 21 passed。
- `C10_SLOW=2` 完整測試：
  - 第一次（18:33–19:29，3,384 秒）：8 failed、816 passed、4 skipped，8 個全是 MemoryError（`test_p04_opp.py` 的 `TestRealSelftest` 在 `c10data.py:114`，之後 `test_p04_rank_fuse` 五個在 `p02_rank_pseudo.py:101`、`test_t02_checks` 兩個）。同一時段另一個 gate 代理在跑 L2，別條線也在跑，可用虛擬記憶體最低 5.7 GB。
  - 記憶體恢復後：那 8 個單獨重跑 8 passed（119 秒）；它們所在的三個檔 127 passed（158 秒）。
  - 第二次完整測試（每 30 秒記一次記憶體）：**824 passed、4 skipped**、453 subtests，4,301 秒，exit 0。4 個 skip：P03 冒煙、prereg03 真 apply、`t04_dryrun` 的「R20 U1 之前」（要正式模式）、閘門專用。
- 上傳副本與計數檔：`data/colab_upload/test04/` 只有 `test04_rank.ipynb`，SHA-256 `0b329c357ae5db582e3d0159d0015df70430034369446242563ffbaf62f2e097`，等於工作樹和 `git show 62ca557:colab/test04_rank.ipynb`；12 碼 `ef5be6e00a4f`。`data/prereg04/counts_fit.npz` `f6c6debe90639585763b9812c235743a6b7f89d8db1c6422ece7bccd550a08a5`（178,537,934 bytes），等於 `count_report.json` 的 `counts_sha256`。

**realmode**（第 13 條最後一點的正式模式演練，`p04gate\realmode`；一次性 clone `git clone -b p04-testday`，在 `62ca557`，clone 裡的 commit 作者都是 kuotunyu，共 24 筆）：
- 86 個步驟，85 個符合預期（`realmode\summary.txt`）。唯一不符合的是 017：預設 `core.autocrlf=true` 的 clone 讓 `test_t02_hygiene` 的 `Pins` 失敗；把 repo 工作樹的位元組複製進 clone 後同一個閘門 41 passed（019）。這就是 6.1 第 4 條。
- 主鏈：`opp` 0（seam_val 的 O4 檔 `30d312c02c35`、Δκ `0.1050265798301427`）；R20 上傳前跑的 zip → `unpack` 2 → `archive` 0（`before_r20_u1`、`sw_exempt_1`、「G4 優先、A100 備用」）；接縫失敗 → `unpack` 1 → `archive` 0（`notebook_seam`、`sw_failed_2`、「登記的重跑，只用 G4」）；舊 zip、帶「(1)」的舊 zip、A100 的重跑、只在工作樹改的 notebook、錯版 notebook 之後再 archive，都照預期被拒；G4 的重跑 `unpack` 0（`attempt_number` 2、`drive_archives` 相等）→ `apply` 0（本機接縫逐位元組相同，六個 Δκ 都 > 0，交件檔 `a42c09fa99f4`，和 R20 規則 B 不同 179 題）。
- `official_log`：`prepare`／`uploaded`／`final`／`status` 照 PREREG_04 5.3 判定；沒有修訂、修訂列不存在、引用 rev 3、只有文字、修訂列只在工作樹的 `final --id 1` 都被拒；commit 一列 rev 4 之後才接受。`archive --bug-revision 3` 2、`--bug-revision 4` 0（`sw_exempt_N`）；同一個 rev 再用、早於失敗的 rev 都被拒。
- 這一輪找到的另外兩件（6.1 第 2、3 條）是訊息和文件的問題，三處工具的判定彼此一致，沒有交錯檔的風險。

**L2**（`p04gate\L2`；一次性 clone `git clone --no-hardlinks -c core.autocrlf=false --branch p04-testday`，HEAD `62ca557669538aae57e5e7460b003dfdc96bb9c7`）：
- 步驟時間：p01 1,376 秒、p02 1,442 秒、p03a 763 秒（另有 739 秒是崩潰的那一次）、p03b 585 秒、測試包 33 秒、t02nb 199 秒、t02open 36 秒、t02apply 1 秒、t04nb 577 秒、`unpack` 3 秒、`opp` 10 秒、`apply` 3 秒、`status` 2 秒；步驟合計 84 分鐘，牆鐘 17:42–19:20（98 分鐘，含一次重跑）。
- p03a 第一次以 exit 139 結束（沒有 traceback，停在 cell 22 final dumps，訓練已完成，`final.pth` `ec62f8ba…`）。把 `rank_a` 和 `work\prereg03_pass_ckpt_a_smoke` 移到 `L2\crash1\` 後從 p03a 重跑：24 格都 ok，`final.pth` 仍是 `ec62f8ba92e7…`（和 integrator 9/30 早上的 L2 相同）；p03b `2e88f8c9fa0b…`。
- test04 notebook：印出「notebook 版本：ef5be6e00a4f」；preflight 通過；`run_log` completed；接縫通過，兩臂 C0 最大差 0.0，d4、swap 逐位元組相同；zip `b27122c787c2…`，40 個成員全在。
- 本機工具：`unpack` 0（40 個檔、`notebook_identity.same` true）；`opp` 0（重播失敗 0/142、0/160、0/175；public、private 的 O4 Δκ `0.08166747229699296`、`0.08307640512198267`，和 integrator 那次相同）；`apply --smoke` 0（交件檔 `e19ccf7e3d10…`，本機 Q02 接縫只記錄：最大差 0.387、規則 B 差 9 題）；`status` 0。
- 另外的只讀核對：`make_test04_smoke_tree.py --check` `ok: true`（seam_val 6 個 npz 等於 03 冒煙的 dump；public 160 盤、private 175 盤的轉接用 P03 自己的 `dump_val_p03` 重算，逐位元組相同）；`test_p03_nb.py -k SmokeDumps` 1 passed。
- （K5 閘門第 2 輪補）這一輪的交件檔 `e19ccf7e3d10…`、本機 Q02 接縫 0.387／9 題，和 1.2 的 `4872f24a…`、0.388 不同，不是退步：prereg01、prereg02 的 CPU 冒煙權重每次跑都不一樣，下游跟著變，只記錄。跨次數相同的是 prereg03 兩臂的 `final.pth`、O4 檔和 O4 Δκ（見 1.2「跨次數比對」、7.1 第 4 條）。

**fixes**（`p04gate\fixes`）：round 3 的修正逐條對程式核實，都屬實、沒有退步（`--bug-revision` 在 cp950 的這台電腦用真的 git 讀中文 PREREG_04；版本 12 碼和輸入框的位置；`sw_failed`／`sw_exempt` 三處一致，對 `[exempt]`、`[exempt, counted]`、`[counted, exempt]` 三種順序推演結果相同）。找到 6.1 第 1 條（major）和第 2 條。

### 6.5 還沒驗到的地方、要重跑的 gate 檢查

1. **要在本輪的 commit 上重跑的**：
   - `fixes`：核對 6.1 第 1–3 條的修正（特別是「不是斷線」流程之後的重跑，`unpack` 要是 0；封存前的 preflight 停下不算，算出輸出的執行早於封存仍是 2）。
   - `realmode`：本輪改了 `unpack` 的新舊核對、`archive` 的下一次執行說明和 `attempt.json`，都在正式模式的封存和重跑路徑上。重跑時預設 `autocrlf=true` 的 clone 應該在 017 就通過（不用再把 repo 的位元組複製進去）；建議另加 6.1 第 1 條的流程（`archive --drive-folder sw_interrupted_K`，再解一份 `run_log` 開頭有封存前 `preflight_failed` 的重跑 zip → 0）和第 2 條（占次數的封存之後 `archive --bug-revision N` 印「下一次仍是登記的重跑，只用 G4」）。
   - `r20-suites`：spec 10.6 第 5 點「修正後重跑 1–3」。完整測試和 `C10_SLOW=2` 要在這個 commit 上各跑一次（`C10_SLOW=2` 照 spec 8.4 單獨跑、先確認記憶體），K5 核對表第 2 項才有已 commit 的依據。R20 兩個 clone 的比對照做：本輪沒碰 R20 的 39 個檔，`.gitattributes` 只影響 7 個 `results/prereg02` 檔的 checkout（`core.autocrlf=false` 的 clone 本來就是 LF），預期和 6.4 相同。
2. **不用重跑 `L2`**：notebook、產生器、`t04_fake_tests.py`、`make_test04_smoke_tree.py` 都沒改；`t04_rank.py` 在 L2 走的成功路徑沒有變（沒有封存時 `freshness` 在比對之前就回傳，`seam_msg` 只在結束碼 1 用到，`archive` 不在 L2 裡）。L2b 同理。
3. `C10_SLOW=2` 在本輪沒有跑（只跑了 `C10_SLOW=1` 的 04 兩個檔，見 6.3），留給上一點的 `r20-suites`。
4. `rev3_draft.md` 沒有跟著改（6.2）；正本是 `docs/PREREG_04_OPPONENT.md` 第 11 節，`rev3_final.md` 第 2 部分和它逐字相同。rev 3 那一列的 `(commit)` 仍待填。
5. 6.1 第 5、7、8 條是檢查者的環境，本輪只改文件，沒有辦法在程式裡防；spec 8.4、8.5 的做法（單獨跑、先看記憶體、崩潰時怎麼接）要靠之後跑的人照做。
6. 暫存：`FX4\crlf_clone1`、`FX4\crlf_clone2`（沒有 junction）、`FX4\slow1_tree` 都可以整個刪掉；`FX4\old_t04_rank.py` 是 `62ca557` 的 `t04_rank.py`，只給回歸核對用。
7. **待辦：notebook 第一格的一句說明**（K5 閘門第 2 輪 `fixes` minor，見 7.1 第 2 條）。`colab/build_test04_rank_nb.py` 第 391 行「會停下來的情況」preflight 那一點寫「重跑（有 `sw_failed_*` 資料夾）卻不是 G4（`sw_exempt_*` 是不占次數的封存，之後照第一次的規則）」，讀起來像任何 `sw_exempt_*` 之後都照第一次的規則；同一檔第 799 行 ⑥-5 的註解（「sw_exempt_*（不占次數的封存）之後照第一次的規則（G4 優先、A100 備用）」）也是同一種寫法。程式（`rerun_gpu_problem`：只要有 `sw_failed_*` 就只收 G4）、`unpack`、`official_log` 三處判定一致，只有這兩處文字不精確。**不為這一句重建 notebook**（會換上傳副本和 12 碼 `ef5be6e00a4f`）。下次因為別的原因重建時（例如彩排後），把兩處括號都改成「只有 `sw_exempt_*`、沒有 `sw_failed_*` 時照第一次的規則」，和第 313 行（CHANGE_ROWS）一致；重建後照 TESTDAY_04 第 3 步換上傳副本、跑 `tests/test_t04_nb.py` 和 L2b。

## 7. K5 gate round 2 and fixes

- 修正者（K5 閘門第 2 輪），2026-10-01 凌晨。分支 `p04-testday`，起點 HEAD `62794a3`（第 6 節那個 commit；main 仍是 `6ec5e68`）。本節和下面的修改一起 commit 在 `p04-testday`（commit 編號見那個 commit 本身；沒有動 main、沒有 amend）。
- 輸入：gate 第 2 輪在 `62794a3` 上交來的 4 條 minor（`fixes` 2 條、`realmode` 1 條、`L2` 1 條），沒有 major。
- 4 條都先對照程式或紀錄核實，都屬實：
  - 1 條只加測試，程式不用改。
  - 1 條改一則拒絕訊息並加測試，PREREG_04 第 11 節補一句。
  - 1 條照 gate 的建議不重建 notebook，只記待辦（6.5 第 7 點）。
  - 1 條只改文件。核對時另外發現 test04 本機 Q02 接縫的數字也會跟著變：gate 寫的「兩輪都相同」只到小數 3 位。
- 結束碼和判定都沒變，只改一則結束碼 2 的訊息文字。來源、組合、參數、閘門、H4R、截止點、門檻都沒動，沒有改變任何驗證集的數字。
- 凍結檔都沒動：
  - `tools/official_log.py` 仍是 `ee0ace46b4b6`。
  - notebook、產生器沒改（`0b329c357ae5`、`affab5c2575b`）。上傳副本 `data\colab_upload\test04\test04_rank.ipynb` 的 SHA-256 仍然相等（資料夾裡只有這一個檔），12 碼 `ef5be6e00a4f` 不用換。
- 沒有開答案檔，沒有跑 `tools/score.py`，沒有用 GPU；執行緒上限 6。開跑前可用虛擬記憶體 23.7 GB、CPU 負載 25%，同時段只跑一件重工作。
- 暫存都在 `…\scratchpad\p04gate\fixer2\`，以下簡寫成 `FX5`。

### 7.1 每一條的處理

| # | 檢查、嚴重度 | 檢查者說的 | 核對結果 | 處理 |
|---|---|---|---|---|
| 1 | fixes，minor | 第 6 節第 3 條把 14 處結束碼 1 的訊息都改用 `seam_msg(ctx)`，但測試只涵蓋 `unpack` 的 `fail1` 和 `apply` 的 notebook 接縫。本機 Q02 接縫沒過（`local_seam_failed`）在登記的重跑要印 `MSG_SEAM_LAST`，這一處沒有測試；它有自己的 `first` 參數，最容易被改回去。`check_notebook`、`check_dumps`、`notebook_seam_recompute` 的其餘各處個別改回去也不會被抓到 | 屬實：gate 的突變 M7（這一處永遠回傳 `first`）跑 `-k "local_seam or registered_rerun_has_no_third_run"` 是 3 passed（`p04gate\fixes2\mutations.json`）。14 個 `seam_fail` 呼叫裡，有行為測試的只有上面兩處 | **程式不改**。**加兩個測試**：<br>1. `ApplyDrills.test_local_seam_fails_on_the_registered_rerun`，兩種情況：<br>- 第一次：結束碼 1。輸出和 `attempt_verdict.json` 的 `message` 都含「走 docs/TESTDAY_04.md「重跑一次」」，不含 `MSG_SEAM_LAST`。<br>- 登記的重跑（`failed/attempt_1/attempt.json` 占次數；`unpack_sw.json` 改成 `attempt_number` 2、`failed_attempts_seen ["attempt_1"]`）：結束碼 1，兩處都含 `MSG_SEAM_LAST`、不含第一次的文字。<br>- 兩種情況接著都跑 `apply --rerun --reason`，見第 3 條。<br>2. `ApplyDrills.test_every_exit_1_message_goes_through_seam_msg`（用 ast 核對）：每個 `seam_fail` 呼叫的訊息都經過 `seam_msg(ctx, …)`，至少 14 個；`MSG_SEAM` 只在 `seam_msg` 的預設值被讀。<br>突變驗證見 7.3 |
| 2 | fixes，minor | notebook 第一格「會停下來的情況」preflight 那一點的括號「（`sw_exempt_*` 是不占次數的封存，之後照第一次的規則）」讀起來像任何 `sw_exempt_*` 之後都照第一次的規則 | 屬實（產生器第 391 行）。另外第 799 行 ⑥-5 的註解也是同一種寫法。程式（`rerun_gpu_problem`：只要有 `sw_failed_*` 就只收 G4）、`unpack`、`official_log` 三處判定一致；第 313 行（CHANGE_ROWS）的寫法是對的 | 照 gate 的建議**不重建**（會換上傳副本和 12 碼）。6.5 第 7 點記待辦：兩處都改，下次因為別的原因重建時一起做 |
| 3 | realmode，minor | 登記的重跑在 `apply` 以 1 結束之後，`apply --rerun` 的拒絕（2，這部分正確）仍寫「接縫類失敗要 archive 後重跑 notebook」；這時 `archive` 一定拒絕，也沒有第三次，和同一次 `apply` 印的 `MSG_SEAM_LAST` 矛盾 | 屬實：cloneC 012（1，`MSG_SEAM_LAST`）→ 014（`archive` 2「登記只准重跑一次」）→ 015（`apply --rerun` 2，舊訊息），紀錄在 `p04gate\realmode\logs_cloneC\` | **改程式**（`Apply.rerun_gate` 的這一支）：<br>- 先用 `list_attempts`／`is_counted` 算占次數的封存。<br>- 有的話，訊息改成「不是重跑 apply（counted archives [...]）。」加上 `MSG_SEAM_LAST`。<br>- 沒有的話，訊息逐字不變。<br>- 結束碼仍是 2，在寫任何檔之前；`--bug-revision` 的路徑不變。<br>**改文件**：<br>- PREREG_04 第 9 條「上一次 `apply` 是 1 時」那一句，加「那一次已經是登記的重跑…時沒有第三次，照上面「還是沒過」停下，不封存」。<br>- 11.6 f 加一句。<br>- `rev3_final.md` 第 2 部分同步改，和第 11 節逐字相同（本輪核對過），開頭加一句說明。<br>- TESTDAY_04 不用改：第 5 步結束碼 1 那一列已寫「登記的重跑照第 6 點停下、不 archive」，沒有叫人重跑 `apply`。<br>**測試**：第 1 條新測試的後半段。第一次：拒絕含「archive 後重跑 notebook」、不含 `MSG_SEAM_LAST`。登記的重跑：含 `MSG_SEAM_LAST` 和 `attempt_1`、不含舊句。兩種情況的 `attempt_verdict.json` 都沒被動到 |
| 4 | L2，minor | 紀錄沒寫出 prereg01、prereg02 的 CPU 冒煙權重每次跑都不一樣；下游的 SHA 和數字寫得像參照值，下次重跑 L2 的人拿去比會誤判成退步 | 屬實。本輪重算三次 L2（integrator、gate 第 1 輪、gate 第 2 輪）的檔：見下面「第 4 條的核對」 | **只改文件**：<br>- 1.2 L2 紀錄後加「跨次數比對」一段。<br>- 6.4 L2 加一點。<br>- spec 8.5 L2 加一點。<br>比「跨次數」時只比 prereg03 兩臂的 `final.pth`、O4 三檔、O4 Δκ；notebook 接縫在同一條鏈內比。程式不改 |

**第 4 條的核對**（三次依序是 integrator `INT\L2`、gate 第 1 輪 `p04gate\L2\S`、gate 第 2 輪 `p04gate\L2\r2\S`；SHA-256 前 12 碼）：

- **每次都不同**：
  - prereg01 `rank/ckpt/epoch_1.pth`：`31a06ec8a615`、`4f15c6b5556b`、`46b56b6c6c80`。
  - prereg02 `rank/ckpt/epoch_2.pth`：`2164cb81b279`、`e745f9076e9b`、`e76e83db29ea`。
  - test04 冒煙交件檔：`4872f24a0539`、`e19ccf7e3d10`、`3866a99d7614`。
  - 和 R20 規則 B 不同的題數：21（第 1 輪）、22（第 2 輪）。
  - t02 本機接縫：第 1 輪 0.561／12 題，第 2 輪 0.558／11 題。
  - **test04 本機 Q02 接縫**（gate 沒列出來的）：`max_abs_diff` 三次是 `0.3875497069433108`、`0.3866357256824817`、`0.38728173083386414`，seam_val 的 `rank_val_probs.csv` 三次都不同。規則 B 差都是 9 題，`pred_B.csv` 三次都是 `689e6b24e662`，只是剛好相同。它用的是 test02 冒煙的 C5，所以也跟著變。
- **每次都相同**：
  - prereg03 的 `final.pth`：A `ec62f8ba92e7`、B `2e88f8c9fa0b`，三次都一樣。
  - O4 三檔三次都一樣：`o4_seam_val.npz` `5e1ebd7e7b40`、`o4_public.npz` `5ef4cfe094d6`、`o4_private.npz` `08607a69d527`。
  - O4 Δκ：public `0.08166747229699296`、private `0.08307640512198267`，三次都一樣。
- **`t04_apply_report.json` 的 `inputs_sha256`**（gate 兩輪比）：36 項裡 25 項相同，11 項不同：
  - test02 那一側：3 個 `val_p02e9_d4`、`opening_test.npz`、`t1.json`、test02 的 `apply_test_report.json` 和 rank `config.json`。
  - 本來就含時間戳記的：`sw/config.json`、`sw/seam_check.json`、`unpack_sw.json`、`opp_report.json`。

### 7.2 改了哪些檔

- `tools/t04_rank.py`：`801b08820ccc` → `f212879e5aaf`（LF SHA-256 前 12 碼）。只改 `Apply.rerun_gate` 裡「上一次 `apply` 是 1」那一則拒絕訊息（第 3 條）。
- `tests/test_t04_rank.py`：`8afc9a031119` → `92f5cba6ed2c`。新增第 1 條的兩個測試。
- `docs/PREREG_04_OPPONENT.md`：只改第 11 節的第 9 條（「`apply` 的重跑也拒絕」那一句）和 11.6 f；第 0–10 節和 rev 3 那一列（`(commit)` 還沒填）都沒動。
- `results/test04/design/rev3_final.md`：第 2 部分同樣兩處，開頭一句說明。
- `results/test04/design/spec.md`：8.5 L2 加一點（跨次數比什麼）。
- `results/test04/design/build_review.md`：1.2 L2「跨次數比對」、6.4 L2 一點、6.5 第 7 點、本節。
- 沒動：
  - `tools/official_log.py`、`colab/build_test04_rank_nb.py`、`colab/test04_rank.ipynb` 和上傳副本。
  - `tools/t04_fake_tests.py`、`colab/make_test04_smoke_tree.py`。
  - `docs/TESTDAY_04.md`、`docs/COLAB_RUN.md`、`.gitattributes`。
  - 所有凍結檔。

### 7.3 驗證

`PY` = `.venv\Scripts\python.exe`，在 repo 根目錄執行，`OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6`。

- 本輪相關的測試：`PY -m pytest -q -p no:cacheprovider tests/test_t04_rank.py -k "local_seam or registered_rerun_has_no_third_run or every_exit_1 or apply_rerun_rules"`，**6 passed**、2 subtests，63 秒。
- **突變核對**（`FX5\mutate.py`，只在一次性 clone `FX5\clone` 裡改檔）：
  - clone 的做法：`git clone --no-hardlinks -c core.autocrlf=false --branch p04-testday`，在 `62794a3`，再複製本輪的兩個檔；`data` 用 junction。
  - 每個突變只跑它該被抓到的測試（00:37–00:43）。跑完 clone 的 `t04_rank.py` 已還原（`git status` 只有複製進去的兩個檔）。
  - 紀錄檔 `FX5\mutate.log`、`FX5\mutations.json` 和腳本、clone 在 00:54 左右隨整個 scratchpad 一起不見了（見 7.5）。下表照當時的輸出抄。
  - 結果：

| 突變 | 改了什麼 | 結果 |
|---|---|---|
| M0 | 對照組，不改 | 兩個新測試都 passed |
| M7 | gate 的那一個：本機接縫的訊息永遠是第一次的文字 | 兩個新測試**都失敗**。gate 第 2 輪時是 3 passed |
| M8 | 第 3 條改回去：`rerun_gate` 不看占次數的封存 | 本機接縫的新測試**失敗** |
| M9 | `check_notebook` 的 `resumed` 那一處改回 `MSG_SEAM` | ast 測試**失敗** |
| M10 | `check_dumps` 的 `dump contents` 那一處改成寫死的字串 | ast 測試**失敗** |
| M11 | `notebook_seam_recompute` 那一處改回 `MSG_SEAM` | ast 測試**失敗** |

- PREREG_04 第 11 節和 `rev3_final.md` 第 2 部分逐字相同：用腳本改兩個檔的同樣兩處，改完核對是 True；改之前 `62794a3` 的兩份也相同。
- 完整測試（`PY -m pytest -q -rs -p no:cacheprovider tests`，00:56–01:09）：**767 passed、66 skipped**、394 subtests，732 秒，exit 0。
  - 比 6.3 的 765 多 2 個（本輪新加的 2 個測試），subtests 多 2 個（新測試的兩種情況）。
  - 66 個 skip 和 6.3 相同；唯一的 warning 仍是 `pin_memory`。
  - 00:45 開始的第一次完整測試在 00:54 左右失去紀錄檔（7.5），用 PID 停掉它的行程樹後重跑，數字是重跑的。
- 閘門：
  - `C10_K5=1 PY -m pytest -q -rs -p no:cacheprovider tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py`：**41 passed**、37 subtests，沒有 skip。
  - `C10_T04_GATE=1`，同三個檔：**41 passed**、33 subtests。
- R20 的 39 個指紋檔：本輪改的 6 個檔都不在指紋清單裡；`tools/official_log.py` 對 tag 仍然只有 652 行新增；`colab/`、`.gitattributes`、`docs/TESTDAY_04.md`、`docs/COLAB_RUN.md` 對 `62794a3` 沒有差異。
- 上傳副本和計數檔：`data\colab_upload\test04\` 只有 `test04_rank.ipynb`，SHA-256 `0b329c357ae5…`，等於 `colab/test04_rank.ipynb`。`data/prereg04/counts_fit.npz` 和 `design_scratch` 的副本都是 `f6c6debe9063…`。
- `git status --porcelain`（commit 前）：只有本輪改的 6 個檔。
- 沒有重跑：
  - `C10_SLOW=1`／`C10_SLOW=2`：本輪的兩個測試不是慢速的；程式只改一則在寫檔之前的拒絕訊息。
  - L2b、L2：notebook、產生器、冒煙替身都沒改，`t04_rank.py` 在 L2 走的成功路徑沒有變。

### 7.4 對 R20 和正式模式的影響、要重跑的 gate 檢查

- **R20 的位元組**：不受影響。39 個指紋檔、`official_log.py`、`.gitattributes`、notebook 都沒動；`t04_rank.py` 不在 R20 的路徑上。
- **正式模式的鏈**：判定和結束碼都不變。
  - `t04_rank.py` 只改 `apply --rerun` 在「上一次 `apply` 是 1」時的拒絕文字：只在已有占次數的封存時換成 `MSG_SEAM_LAST`，仍是 2，在寫任何檔之前。
  - `unpack`、`opp`、`archive`、`status` 和 `apply` 的成功路徑都沒有變。
  - PREREG_04 第 11 節的兩處只補寫程式本來就這樣做的事。
- **要重跑的 gate 檢查**：
  - `fixes`：核對本節第 1、3 條。可以在一次性 clone 裡重做 M7、M8（本輪的突變紀錄已經不見，見 7.5）。
  - `realmode`：只要重跑 cloneC 的 012 → 014 → 015 那一段。預期 015 仍是 2，訊息改成「不是重跑 apply（counted archives ['attempt_1']）。」加上 `MSG_SEAM_LAST`；其他步驟不受影響。
  - `r20-suites`：R20 兩個 clone 的比對不用重跑（39 個檔不變）。K5 核對表第 2 項如果要「已 commit 的那一版」上的完整測試，就在本輪的 commit 上重跑一次完整測試（本輪在工作樹上跑過，上面 767 passed）；`C10_SLOW=2` 照 6.5 第 1 點。
  - `L2`：不用重跑（第 4 條只改文件）。

### 7.5 暫存被刪（要知道的事）

- 00:54 左右，本 session 的整個 scratchpad（`C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-solar\b3b12887-5b96-4ead-ab34-78eb6f69fc09\` 底下，包括 `scratchpad\` 和 `tasks\`）被刪掉了，不是本輪的指令刪的。`tasks\` 在 00:54–00:55 被重建。
- 一起不見的有：
  - `p04build\`（第 1–5 節的 `INT`、`FX3` 等）。
  - `p04gate\`（第 6 節的 `FX4` 和 gate 兩輪的紀錄）。
  - 本輪 `FX5` 的 clone、突變紀錄和第一次完整測試的紀錄。
- 本檔引用的那些暫存路徑，現在都只剩本檔的文字紀錄。
- `FX5\clone\data` 是指到 repo `data\` 的 junction，刪除沒有跟著 junction 進去。刪除後核對過：
  - `data\` 還有 49 個檔、4.0 GB。
  - `data/prereg04/counts_fit.npz`（和 `design_scratch` 的副本）的 SHA-256 仍是 `f6c6debe9063…`。
  - 上傳副本仍是 `0b329c357ae5…`。
  - `data\` 裡沒有 9/30 23:00 以後改過的檔。
- 本輪之後重建了 `FX5`，只放重跑的完整測試紀錄 `FX5\full.log`。

## 8. 合併前的最後處理（主 session，2026-10-01）

- **notebook 說明兩處**（7.1 第 2 條、6.5 第 7 點的待辦）：趁使用者還沒把 test04 notebook 放上 Drive、12 碼還沒給出去時一起改，不留到彩排之後（那時要請使用者在 Drive 改舊檔名）。
  - `colab/build_test04_rank_nb.py` 第 391 行（第一格「會停下來的情況」）和 ⑥-5 的註解，改成「只有 `sw_exempt_*`、沒有 `sw_failed_*` 時才照第一次的規則」。程式判定沒變（`rerun_gpu_problem` 照舊：有 `sw_failed_*` 就只收 G4）。
  - 重建：`PY colab/build_test04_rank_nb.py --upload-copy`。notebook 和上傳副本 SHA-256 都是 `9dcbf8953c3c…`，`NOTEBOOK_SHA256` 是 `efc0d03e1ce0…`，**使用者要對照的 12 碼改成 `efc0d03e1ce0`**（取代 4.1 的 `ef5be6e00a4f`）。notebook 的差異只有這兩處文字和 `NOTEBOOK_SHA256` 那一行。
  - `PY -m pytest -q tests/test_t04_nb.py`：60 passed、20 subtests，80 秒。
  - L2b（scratch `…\scratchpad\L2b_final`，6 執行緒）：建替身樹 355 秒 → notebook 約 10 分鐘，`completed` → `--check` 403 秒，`ok: true`、`problems` 空。接縫 `passed`、`bitwise_equal`，兩臂 C0 最大差 0.0；zip 40／40 個成員，都在 `sw/` 底下；`config.notebook_sha256` 等於新值。
- **rev 3 的 commit 編號**：修訂紀錄表那一列填成 `62ca557`，並寫明第 11 節在合併前、任何測試資料之前依 K5 驗證修正於 `62794a3`、`5bc8774`。
- **彩排用的假測試集重建**（7.5 的暫存被刪）：9/30 給使用者的 test02 彩排 zip `90f5baa276af…` 的樹不見了。`t02_fake_tests.py from-val` 重建出來的 zip 是 `9fc4de297ca9…`，和舊的只差 `tests/MANIFEST.json` 的 `created`、`tool` 兩個紀錄欄位，6 個題目檔逐位元組相同。使用者還沒上傳，所以上傳副本 `data/colab_upload/test02/C10_test_v1.zip` 換成新的（SHA 等於新樹的 `FAKE_TREE.json`），兩場的 T1.1–T1.3 都 PASS。整棵樹另外備份在 `data/rehearsal_backup/tree_20261001`（323 MB，不進 git）：暫存區再被清掉時複製回去，不要重新 `from-val`（MANIFEST 的時間會變，和 Drive 上的 zip 對不上）。

## 9. Colab 彩排（2026-10-01，對話 `5c8715ae`，AI Pro 帳號）

第一次照 `docs/TESTDAY.md`、`docs/TESTDAY_04.md` 的彩排段落做（和文件不同的地方列在 9.7）：test02 兩份 → 同一份彩排 zip 跑 test04 → 三份都完才收拾。彩排不上傳、輸出不 commit。TESTDAY 的原文是「結果記在 RESULTS（或下一版修訂 rev 9），不在已登記的 rev 8 後面補字」；test02 還沒有 RESULTS，`results/test02/design/build_review.md` 在 R20 的指紋裡（`results/prereg04/design/r20_path_fingerprint.json`）不能改，所以和 test04 一起記在本節。**待辦**：11/04 寫 test02 的 RESULTS 時加一行指到本節（記在 `CLAUDE.md`「接下來」）。假測試集是本機驗證題換前綴（`FAKE_TREE.json` 的 `from-val`），下面的分數和 Δκ 都是驗證資料上的數字，**不是測試集的證據**，只用來確認流程逐位元組接得起來。

本節經 4 個代理獨立核對（workflow `wf_2650de97-20f`：三個分段核對數字、一個核對有沒有漏掉文件要求的紀錄和步驟）。它們重算了所有 SHA、位元組比對和前綴對應，數字都對得上；指出的 1 個進位錯、標示錯和漏記的項目，都已照 9.6 最後一點改進本節。

### 9.1 開跑前

- **`<S>`**：用 `CLAUDE.md` 寫的固定路徑 `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-solar\b3b12887-5b96-4ead-ab34-78eb6f69fc09\scratchpad\rehearsal`（不是 TESTDAY 說的「這個 session 的 scratch」：`FAKE_TREE.json` 和兩份 `t1.json` 記著這個絕對路徑）。換帳號後舊對話的整個暫存資料夾又被清掉，照 7.5／8 的備份從 `data/rehearsal_backup/tree_20261001` 複製回去：52 個檔逐一比 SHA-256 相同，樹裡的 `<S>\data\colab_bundle\C10_test_v1.zip` 是 `9fc4de297ca9…`（等於 `FAKE_TREE.json` 的 `zip_sha256`；repo 的 `data/colab_bundle/` 沒有這個檔）。沒有重新 `from-val`；T1.1–T1.3 沿用備份裡的 `t1.json`。
- **上傳副本**：`data/colab_upload/test02/` 三個檔：`test02_rank.ipynb` `53337a7acc39…`、`test02_player.ipynb` `5c62cf5e1e4a…`（都等於已 commit 的版本，最後一次改在 `9d64e54`），zip `9fc4de297ca9…`；`data/colab_upload/test04/test04_rank.ipynb` `9dcbf8953c3c…` 等於已 commit 的版本，12 碼 `efc0d03e1ce0`（給使用者對照）。
- **Drive 修正**（使用者截圖，跑之前）：9/30 的步驟做完時，`test02_*.ipynb` 被放進了 `old_prereg02`，`C10_test_v1.zip` 不在 `C10`，`prereg02_*`、`prereg03_*` 四份舊 notebook 還在 `C10`（「事前登記 03」第 19 步沒做過）。逐步搬成：`old_prereg02` 放 `prereg02_*`、新建 `old_prereg03` 放 `prereg03_rank_{a,b}.ipynb`、`test02_*` 搬回 `C10`、從上傳副本補傳 zip。搬好後 `C10` 是 4 個資料夾＋4 個檔；搜尋 `C10_test_v1` 只有 `C10` 裡一份（32.9 MB，修改時間 01:12＝10/1 重建的那一份）。
- **運算單元**：開始前「可用量」180.93（頁面寫「Colab Pro」）。使用者看到後說運算單元完全不用擔心，快用完時再看要開新訂閱或加值；Claude 因此決定之後不再問、也不再把它當成停下來的條件，跑完沒有再量（偏離見 9.7 第 3 點）。
- **03 兩臂權重的本機備份**（9/30 使用者同意）：使用者從 Drive 下載的兩個檔，SHA-256 和登記的全文相同：`final.pth` `790b767c5425…`＝A 臂、`final (1).pth` `8e88f6ec6cba…`＝B 臂；已搬到 `data/prereg03_weights/rank_a/final.pth`、`rank_b/final.pth`（不進 git），「下載」裡沒有留下。

### 9.2 test02 競賽二（`test02_rank.ipynb`）

- **Colab**：G4（`NVIDIA RTX PRO 6000 Blackwell Server Edition`，48 vCPU），Python 3.13.15、torch `2.11.0+cu128`、CUDA 12.8、cuDNN `91900`。執行 1 次，`completed`（`run_log` 06:29:44Z–06:31:34Z，110 秒），`total_this_run` 132.9 秒（從第一格的 `RUN_START` 算起，含掛載 Drive；`dump_seam_val` 50.3 秒、public／private 各約 13 秒）。notebook 的接縫 `seam_check_passed`、`seam_bitwise_equal` 都是 true，只讀檔沒變（`readonly_unchanged`），三組題數 1,589／400／400、棋譜 19,156／4,787／4,815，`games_unparsed` 都是 0。結果 zip 8,660,357 bytes，`e0be860c8e8a…`。
- **本機**（`.venv` Python 3.12.13）：
  - `unpack`：22 個檔，執行紀錄只有 1 次 `completed`。
  - `opening`：seam 部分和 `opening_val.npz` 逐位元組相同（19,156 個 slot），public 4,787、private 4,815，52 秒（數 fit 棋譜 720,307 盤、109,652 個 context）。
  - `apply`：接縫最大 |dP| 0（容許 1e-4，同 GPU）、規則 B 不同 0 題，機率和 `pred_B` 都逐位元組相同 → PASSED；閘門 selected − R00 = 0.0893（> 0.01）；`submission_rank.csv` `d9b9801b3df8…`，`check_submission` official OK。
  - spec 5.2 的另一半（拿到 G4 時）：`local_rank/rank_test_probs.csv` 的 800 列，id 之後的文字和 `results/prereg02/rank/rank_val_probs.csv` 第 1–800 列逐列相同（`pub_q_0001…0400`↔`val_q_0001…0400`、`priv_q_0401…0800`↔`val_q_0401…0800`，標頭也相同）；`local_rank/seam_val/rank_val_probs.csv` 和 prereg02 的那一份逐位元組相同；`submission_rank.csv` 的 800 個標籤在前綴對應下等於 `results/prereg02/rank/pred_B.csv` 第 1–800 列（0 列不同）。
- **判定**：符合 `docs/TESTDAY.md` 彩排段落和 spec 5.2 的預期（G4：本機接縫逐位元組相同、測試機率和交件標籤在前綴對應下等於驗證的）。

### 9.3 test02 競賽一（`test02_player.ipynb`）

- **Colab**：G4，版本同 9.2。和競賽二同時跑（開始 06:30:51Z），`total_s` 784.1 秒：三個池的 dump 各約 256 秒，每個池裡約 227 秒是候選 embedding（`timings.json` 的 `embed_candidates_s` 226.7 每個池都覆寫，留下的是最後一個池 seam_val 的值，已含在它的 dump 時間裡）。最後沒有紅字。`repro_check`：400 題 Top-5 全部相同、centroid 逐位元組相同（和 prereg01 同 GPU、同 torch）；`seam_p02`：候選與題目的 dump 和 prereg02 的逐位元組相同。結果 zip 128,953,174 bytes（notebook 印 129.0 MB），`96be49c6e7f7…`。
- **本機**：
  - `unpack`：18 個檔，執行紀錄只有 1 次 `completed`。
  - `seam_val`：選 Q04（0.899765；Q00 0.400641，閘門 +0.499124），第 1 層（ρ raw 1.0068、canon 0.9888），接縫 PASSED（8 項全 true）；`local_player/seam_val/top5.csv` 和 `results/prereg02/player/player_val_top5.csv` 逐位元組相同（`cmp`）。
  - `public`：Q04 0.902147 對 Q08 0.898223，第 1 層（ρ raw 1.0126）。`private`：Q04 0.898212 對 Q08 0.897055，第 1 層（ρ raw 1.0040）。三個池都 0 題 fallback，各約 20 秒。public、private 的 `top5.csv` 在前綴對應下（`pub_pq_`／`priv_pq_`→`val_pq_`、`pub_p_`／`priv_p_`→`val_p_`）和 `player_val_top5.csv` 逐字相同，各 400 列。
  - `submit`：800 列（public 400、private 400），`b5d7473e1011…`；`submit_report.json` 的 `check_submission` 是 official、passed、0 個錯誤。
- **判定**：和 `docs/TESTDAY.md` 彩排段落、spec 5.2 寫下的每一個數字都相同。

### 9.4 test04（`test04_rank.ipynb`）

- **Colab**：test02 兩份的結果 zip 都放好後才開跑；給使用者的步驟第 1 點要先確認 `test02_player` 的執行階段已刪（TESTDAY_04 第 3 步第 3 點），使用者沒有另外回報（見 9.7 第 6 點）。第一次的規則，拿到 G4，`sw/config.json` 的 GPU 完整名稱是 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，和 03 完成標記的名稱相同（`seam_check` 的 `same_gpu` true，容許差 1e-4）；版本和 TESTDAY_04 寫的參照（`2.11.0+cu128`、`91900`）相同，`env_same` true（12 個 logits 標記和 3 個 key 標記一致）。執行 1 次，`completed`（`run_log` 07:40:44Z–07:43:43Z，179 秒）；`total_this_run` 229.4 秒是從第一格的 `RUN_START` 算起，含掛載 Drive 和使用者在版本輸入框前的等待（`dump_seam_val` 98.5 秒、public／private 各約 25 秒）。跑完表示版本輸入框的回答是 `y`（不分大小寫、前後空白；不是 `y` 會停下）；跑的確實是已 commit 的 notebook，由 `unpack_sw.json` 的 `notebook_identity`（`run_notebook_sha256` 等於 repo 的 `efc0d03e1ce0…`）證實。
- **兩臂接縫**（`sw/seam_check.json`）：`passed`、`bitwise_equal` 都是 true；A、B 兩臂的 C0 最大差 s0 0.0、swap 0.0，`bitwise_equal_d4`、`bitwise_equal_swap` 都是 true，8 種對稱的 logits 最大差全是 0.0。結果 zip 19,358,436 bytes，`cacd3bdefd83…`。
- **本機**（輸出在 `results/test04_rehearsal/`，gitignore）：
  - `unpack`：40 個檔解到 `sw/`，attempt 1，執行紀錄只有 1 次 `completed`。
  - `opp`（6 個 worker，107 秒）：seam_val 19,156、public 4,787、private 4,815 個 slot 重播失敗都是 0；seam_val 的 O4 檔和 `results/prereg04/opp/val_o4.npz` 逐位元組相同（`30d312c02c35…`），seam_val Δκ 0.1050265798301427 等於登記值；public、private 的 O4 Δκ 0.0930（741 個子題）、0.1000（751 個子題），是判定用的數（PREREG_04 rev 3 第 11 節第 7 條：兩個都 > 0 才是「04 可能可用」，最終以 `apply` 的 6 個數為準）；結束碼 0。
  - `apply`：Q02 接縫最大 |dP| 0、規則 B 不同 0 題，機率、`pred_B`、`pred_A` 三項逐位元組相同 → PASSED。執色檢查的 6 個數都 > 0，`usable` true：public O4 0.0930／SWA 0.0321／SWB 0.0278（741 個子題），private O4 0.1000／SWA 0.0418／SWB 0.0364（751 個子題）；public、private 的 O4 和 `opp` 的逐位相同。seam_val Δκ O4 0.10503（等於 `opp`、等於登記值）；SWA 0.03631、SWB 0.03436 只記錄（和 prereg04 `check_report.json` 的記錄值差 0）。`submission_rank.csv` `a42c09fa99f4…`（等於 `pred_B.csv`），`check_submission` official OK；規則 B 和 R20 彩排交件檔不同 179 題（只記錄）。結束碼 0。
- **判定**：符合 `docs/TESTDAY_04.md` 彩排段落（同 spec 8.6）的預期（G4：兩臂 `bitwise_equal`、本機 Q02 接縫逐位元組相同）。最大差是 0，不到 1e-5，不用討論 rev 4。

### 9.5 收拾

- **Drive**（使用者截圖）：`C10` 裡新建 `old_rehearsal`，放 `test02_rank.ipynb`、`test02_player.ipynb`、`test04_rank.ipynb`、`C10_test_v1.zip`；`C10` 剩 `old_prereg01`、`old_prereg02`、`old_prereg03`、`old_rehearsal`、`results` 五個資料夾和 `C10_bundle_v1.zip`，等於 COLAB_RUN「測試集（test02）」第 2 步「做過彩排」的樣子（加上 `old_prereg03`）。`results/test02_rehearsal`、`results/test04_rehearsal` 留在 Drive 不動。Colab 存檔時改了三份 notebook 的修改時間（它們已收進 `old_rehearsal`）。
- **本機**：
  - 刪掉 `data/colab_upload/test02/` 裡的三個彩排副本（正式那天 TESTDAY 第 2 步會重新複製）；`data/colab_upload/test04/` 保留（正式那天用同一份）。
  - 三個結果 zip 都是使用者一放進 `results\test02\`、`results\test04\`，就在 `unpack` 之前移到 `<S>\downloads\`（TESTDAY 的「立刻」），同時複製到 `data/rehearsal_backup/results_20261001/`。transcript 的時間：rank 06:37:51Z 放好、06:38:05Z 移走；player 06:51:11Z／06:51:23Z；test04 07:59:58Z／08:00:12Z。`results/test02/`、`results/test04/` 裡沒有留下 zip。
  - 跑完後，`<S>` 裡的 `rehearsal`、`local_rank`、`local_player`、`results` 和 repo 的 `results/test04_rehearsal/`，都另外複製到 `data/rehearsal_backup/run1_20261001/`（166 個檔，370,561,779 bytes；其中 `test04_rehearsal/` 56 個檔）。
  - 使用者電腦的「下載」、桌面、文件、OneDrive 裡沒有彩排的結果 zip（Claude 查過），所以沒有請使用者刪檔。
  - `results/test04_rehearsal/` 留在 repo 裡（gitignore）。**要做第二次彩排時**（TESTDAY_04 彩排第 5 點）：
    - 先請使用者把 Drive 的兩個 `*_rehearsal` 改名成 `*_rehearsal_1`，收拾時收進 `old_rehearsal_2`。
    - 本機的 `results/test04_rehearsal/` 刪掉，不要照文件移到「上一次的 `<S>`」：那裡是暫存區，已經被清掉兩次；第一次的內容已經備份在 `data/rehearsal_backup/run1_20261001/test04_rehearsal/`。
    - 新的 `<S>` 不能沿用這棵樹。`FAKE_TREE.json` 的 `zip` 和兩份 `t1.json` 的 `tests_dir` 都寫死了第一次的 `<S>`，所以第二次要在新的 `<S>` 重新 `t02_fake_tests.py from-val`、重跑兩場的 T1.1–T1.3，會得到新的 zip SHA，Drive 上要重傳。`CLAUDE.md`「固定路徑、不要重新 from-val」只適用第一次的樹。
- **沒有做的事**：沒有讀 `data/local/` 的答案檔、沒有跑 `score.py`、沒有打分、沒有上傳 AIdea；凍結檔沒有改（防誤改閘門見 9.6）。

### 9.6 驗證

- 彩排的本機步驟都做完之後，`git status --porcelain` 是空的（HEAD `3eed22a`），`C10_T04_GATE=1 PY -m pytest -q -rs tests/test_p04_hygiene.py tests/test_t02_hygiene.py tests/test_t04_hygiene.py`：41 passed、33 subtests passed，沒有列出被略過的測試。這次只在之後跑，開工前沒跑（9.7 第 1 點）。
- 4 個代理的獨立核對（workflow `wf_2650de97-20f`，唯讀、沒有碰 `data/local`）重算了本節所有的 SHA、大小、位元組比對和前綴對應：
  - 9.2 的 opening 逐 key 位元組比對；競賽二交件檔對 `pred_B.csv`；競賽一 public／private 的 Top-5；`sr_med` 和凍結值相同。
  - test04 的 40 個成員對 zip；`o4_seam_val.npz` 對 `val_o4.npz`；`rank/seam_val/{rank_val_probs,pred_B,pred_A}.csv` 對 `results/prereg04/rank/`。
  - 截圖類的事實和 session 的截圖檔逐張核對。
- 代理指出、已改進本節的：private O4 Δκ 進位（0.0999 → 0.1000）；public／private 的 O4 Δκ 不是「只記錄」，是判定用；漏記 `apply` 的 6 個 Δκ；預期出自 TESTDAY_04 不是本檔；`total_this_run` 含掛載和等待；`embed_candidates_s` 是覆寫值；「輸入 y」的推論範圍；spec 5.2 的三項（交件標籤、public／private Top-5、競賽一 check_submission）；zip 的路徑寫法；運算單元的轉述；第二次彩排要新樹；`results/test04_rehearsal` 的備份；以及 9.7 列的偏離。

### 9.7 和文件不同的地方（都不影響結果，記下來）

1. **防誤改閘門只在之後跑**：TESTDAY_04 要求每次開工先跑。這次在彩排的本機步驟都做完之後才跑（全過）。彩排的輸出只寫到 `<S>` 和 gitignore 的 `results/test04_rehearsal/`，開工時 `git status` 是乾淨的（HEAD `3eed22a`，只多了一個 `CLAUDE.md` 交接 commit），所以沒有凍結檔被改的機會；正式那天照文件先跑。
2. **`<S>` 用固定路徑**，不是 TESTDAY 說的「這個 session 的 scratch」（9.1；原因是樹裡寫死了路徑）。
3. **運算單元沒有記用量**：TESTDAY_04 彩排段落和 spec 8.6 要記「實際時間和運算單元」。時間記了，運算單元只有開始前的 180.93。使用者說不用擔心之後，Claude 決定不再問（9.1）。所以 COLAB_RUN 表格裡的運算單元（G4 約 2／約 3／約 0.7）仍是推估、沒有實測。另外，同一個帳號同時在跑 C1 的 Colab，用量本來也分不開（第 7 點）。
4. **給使用者的步驟省掉的檢查**：
   - test02 第 8 步（可用量至少 10）照做了。
   - test04 第 7 步（可用量至少 3）沒給：使用者說不用擔心之後就省掉。
   - test04「開始前」第 3 步（`C10` 內容、`results/prereg03` 有 `rank_a`、`rank_b`、還沒有 `test04_rank.ipynb`）沒有另外請使用者截圖，因為：`C10` 的內容在 test02 開跑前剛截圖確認過，test02 不改 `C10` 的檔案；使用者 06:36Z 左右剛從 `results/prereg03/rank_{a,b}/ckpt/` 下載過兩個 `final.pth`；`test04_rank.ipynb` 是這次才第一次上傳。
   - **正式那天**：COLAB_RUN 的運算單元門檻（test02 第 8 步 10 個、競賽一隔天 5 個、test04 第 7 步 3 個）照使用者的偏好只當成參考，不當成停下來的條件。COLAB_RUN 本身不改；`CLAUDE.md` 已記這個偏好。Drive 的檢查照常做。
5. **競賽二的 zip 提早下載**：COLAB_RUN test02 第 19 步寫「兩份都做完第 18 步後」才下載。這次 rank 跑完後，就請使用者在競賽一還在跑的時候先下載 rank 的 zip（和兩個 `final.pth`）。只是下載，沒有動 Drive 上的檔，競賽一照常跑完。
6. **刪執行階段沒有另外確認**：
   - 每次都在步驟裡寫了「先看檔名，再中斷連線並刪除執行階段」。
   - 但 rank 和 player 那兩次，使用者只回「放好了」；test04 跑完時使用者問「Colab 可以關掉了嗎」，Claude 回覆要先確認刪了執行階段再關，使用者沒有再回報。
   - 刪了沒有，Claude 不知道。沒刪的話，Colab 會在閒置一段時間後自己收回執行階段，不影響任何結果。
7. **同一帳號同時有別條線的 Colab**：C1 的 `c1_v2b_smoke.ipynb`（L4）在 `test02_player` 和 test04 的那段時間也在同一個帳號上跑（使用者 06:47Z 誤貼了它的截圖）。上面的 Colab 時間不是在閒置帳號上量的；Drive 也是同一個帳號，但 C1 的 notebook 不讀寫 `C10/`。

### 9.8 以前留下的開放項目

- **3.4 第 2 點（G4 回報的名稱）**：test04 的 `device` 那一行沒有請使用者另外抄，但 `sw/config.json` 記的完整名稱是 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，和 03 完成標記相同（`same_gpu` true），test02 兩份的 `env.json` 也是同一個名稱。**這一點關閉**：名稱沒變，不用 rev 4。
- **4.4 第 1 點（版本輸入框在真的 Colab 上）**：輸入框在真的 Colab 上走過一次。使用者回答後才往下跑，跑完，`config.notebook_sha256` 等於已 commit 的版本。輸入框出現的時間點沒有請使用者記錄，只知道 `total_this_run` 比 `run_log` 的開始到結束多 50 秒（含掛載和等待）。**部分關閉**。
- **2.4 第 2 點、3.4 第 2 點後半（登記的重跑只收 G4）**：彩排只走第一次，**還沒在 Colab 上走過**，維持開放。
