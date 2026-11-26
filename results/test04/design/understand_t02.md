# U1：test02 競賽二管線的重用面（給 test04 設計用）

- 寫於 2026-09-30，分支 `p04-testday`（從 main `6ec5e68` 開出，工作樹乾淨），唯讀閱讀，repo 沒有改動。
- 讀過：`tools/t02_rank.py`、`colab/build_test02_rank_nb.py`（和它產生的 `colab/test02_rank.ipynb`）、`tools/t02_test_bundle.py`、`tools/t02_fake_tests.py`、`tests/test_t02_dryrun.py`、`tests/test_t02_nb_rank.py`、`colab/run_local_smoke.py`、`tools/check_submission.py`、`results/test02/design/{spec.md,code.md,build_review.md}`。另外為了第 3 節，讀了 `tools/p02_rank_fuse.py`、`p03_rank_fuse.py`、`p04_rank_fuse.py` 的相關函式，以及 `tests/test_p04_hygiene.py`、`results/prereg04/design/r20_path_fingerprint.json`。
- 在 scratch 跑了一個探針（CPU，只讀 repo，沒開答案檔、沒打分）：`...\scratchpad\p04testday\U1\q02_replay.py`。結果寫在第 3.3 節。

## 0. 結論

1. **t02 的程式可以直接 import 來用，不必改。** 通用的輔助函式都在模組層，簽名穩定，例如 `seam_compare`、`load_notebook`、`check_test_files`、`check_set_inputs`、`t1_status`、`load_opening_test`、`common_checks`、`write_probs`、`require_committed_clean`、`notebook_source_sha256`、`pin_sha256`。
   - 寫死 R20 或 `p02e9_d4` 的部分不能拿來用：`apply_set`、`check_dump_env`、`seam_tol`、`dump_marker`、`notebook_paths`、`check_notebook_identity`、`resolve`、`check_out_dir`、`apply_outputs`。
   - `apply_set` 只回傳 R20 的機率 P，不回傳 m、E 項，所以 Q02 要用 p02／p03／p04 的凍結函式重新組。探針證明這樣組出來和 `results/prereg04/rank/rank_val_probs.csv` 逐位元組相同。
2. **`t02_test_bundle.py unpack` 不改就沒辦法收 test04 的 zip，所以 `t04_rank.py` 要有自己的 `unpack`。** 擋住的地方有三處：
   - argparse 的 `--comp` 只收 `rank`、`player`（`t02_test_bundle.py:1056`）。
   - 程式內用 `COMP_FILES[args.comp]` 查表（`:997`）。
   - real 模式的結果目錄只能是 `results/test02` 或 `results/test02_rehearsal`（`:948-952`）。
   - 新的 `unpack` 可以重用這些 helper：`tb._member_ok`、`run_status`、`run_history`、`earlier_failures`、`verify_manifest`、`check_writable`、`sha256_file`、`write_json`、`now_iso`、`tool_record`。另外也可以用 `p03_rank_fuse.member_problem`，它多了副檔名白名單。
3. **測試日之後，C5 的測試 dump 和 `opening_test.npz` 都在 `results/test02/rank/`**，格式見第 2 節。test04 只讀這些檔，不寫。
4. **假測試演練可以做到逐位元組的證明。** 演練順序是：
   - 照原樣跑 t02 的 from-val 管線，R20 的輸出要等於 main 上記錄過的標準值（`d9b9801b…`）。
   - 用 prereg03 的 d4、s0_swap dump 按同一套假 id 切片、換前綴，當成 test04 notebook 的輸出。
   - 跑 t04 的 opp 和 apply。
   - 探針已證明：Q02 在驗證題第 1–400 題（當 public）、第 401–800 題（當 private）上，每一列都等於 `results/prereg04/rank/rank_val_probs.csv` 對應的列（只換前綴）；全部 1,589 列也逐位元組相同。三個來源在兩個假測試集上的 Δκ 都 > 0。

## 1. 重用面

### 1.1 `tools/t02_rank.py`：可以直接 import 的部分

import 方式一律寫 `import t02_rank as t2`，不要用 `from t02_rank import …`。原因是 `tests/test_t02_rank.py:188-198` 用 `setattr(t02, …)` 把模組常數換成合成資料：函式執行時才讀模組的全域變數，用 `from` 在 import 當下就把值綁死了，測試換不到。

import 的副作用只有兩個：
- 把 `tools/` 插進 `sys.path`（`:81`）。
- import `c10data`、`check_submission`、`decide`、`p02_rank_fuse as rf`、`p02_rank_opening as ro`、`p02_rank_pseudo as rp`（`:82-87`）。

沒有 player 模組，隊伍分離可以維持。實測 `t02_rank`、`t02_test_bundle`、`t02_fake_tests`、`p04_rank_fuse`、`p04_opp` 在同一個行程裡 import 不會互相衝突。

**通用 helper**（和 R20 無關，可以直接用）：

| 名稱（行號） | 簽名 | 作用、t04 的用法 |
|---|---|---|
| `sha256_file` (142) | `(path)` | 原始位元組的 SHA-256（轉呼叫 `rp.sha256_file`） |
| `sha256_text` (146) | `(path)` | 先把 CRLF 折成 LF 再算，工具檔用這個 |
| `pin_sha256` (301) | `(path)` | `.json`、`.csv` 折 LF，其他檔（`.npz`）用原始位元組。t04 釘 prereg04 的 csv、json 時用它，不受 autocrlf 影響 |
| `same_path` (155)、`is_inside` (159) | `(a,b)`、`(path, root)` | 路徑比較 |
| `refuse_needle` (166) | `(path)` | 檔名含驗證答案檔字樣就丟 `T02Error` |
| `read_json` (173)、`write_json` (181) | | `write_json` 轉呼叫 `rp.write_json`：先寫暫存檔再改名。它**不擋任何目錄**，t04 要自己擋 |
| `git` (185)、`git_head` (193)、`repo_rel` (198)、`shown` (202)、`tracked` (207) | | git 查詢 |
| `require_committed_clean` (214) | `(paths, what)` | 每個檔都存在、在 repo 內、被 git 追蹤、和 HEAD 相同，否則丟例外。回傳 `{"files","rule","git_head"}` |
| `environment` (232)、`tool_record` (238) | `()` | 記錄環境。`tool_record` 記的是 **t02_rank.py** 的 SHA，t04 要另寫自己的版本 |
| `notebook_source_sha256` (250) | `(path)` → `(記錄值, 重算值)` | 任何含 `NOTEBOOK_SHA256 = '<64 hex>'` 一行的 notebook 都能用：把記錄值換成 64 個 0，再對 `json.dumps([各格原始碼], ensure_ascii=False)` 算 SHA-256。test04 的產生器沿用同一規則，t04 就能直接呼叫 |
| `remove_outputs` (319) | `(paths, a)` | 輸出已存在時，沒有 `a.rerun`、`a.reason` 就拒絕；有的話先記下舊檔 SHA 再刪 |
| `atomic_copy` (331) | `(src, dst)` | |
| `load_notebook` (375) | `(ctx)` → `(cfg, log, seam)` | 讀 `ctx["rank_dir"]` 底下的 `config.json`、`run_log.json`、`seam_check.json`。檢查項目：<br>• 頂層和最後一次 run 的 status 都是 `completed`<br>• real 模式下，之前的 run 沒有 `FAILED_RUN_STATUSES`<br>• `seam.passed is True`，而且 `tol` 是數值<br>• `smoke`、`rehearsal` 旗標符合模式（real `(False, False)`；dry-run 和 rehearsal `(False, True)`；smoke `(True, 任意)`）<br>test04 notebook 照同樣的 schema 寫這三個檔，就能直接用。錯誤訊息引用的是 PREREG_02，只是字面問題 |
| `check_test_files` (406) | `(ctx, cfg)` | `ctx["tests_dir"]/MANIFEST.json` 的 SHA 要等於 `cfg["test_bundle"]["manifest_sha256"]`；兩個 rank 檔要等於 `cfg["test_bundle"]["files"][name]`。test04 讀同一份 `C10_test_v1.zip`，就能直接用 |
| `question_files` (429) | `(ctx)` | `{"seam_val": SEAM_QUESTIONS, "public": T/rank_prediction_test_public.csv, "private": T/…_private.csv}` |
| `check_set_inputs` (434) | `(ctx, cfg, qfiles)` | 每組的 `<rank_dir>/<set>/dumps/_INPUT.json` 要有 `set == s`，`sha256` 要等於題目檔；`cfg["sets"][s]["questions_sha256"]` 也要相同。只要 test04 用 `<root>/<set>/dumps/` 的版面，就能直接用 |
| `t1_status` (450) | `(ctx, command, fallback)` | 讀 `ctx["checks_dir"]/rank/t1.json`：<br>• real 模式要求 `complete`、`stage raw`、provenance `mode real`<br>• `command="apply"`、`fallback=False` 時只收 PASS 或 NORMALIZED；FAIL_OPENING 會丟例外，訊息提示改用 R10<br>t04 可以接住這個例外，改寫成「FAIL_OPENING：04 不上場（PREREG_04 5.3）」 |
| `load_frozen` (478) | `(require_git)` | 讀 prereg02 R20 的 `frozen_config.json` 並做全部檢查：SHA 釘選、basis、R20、b = Bβ、dump、opening build、凍結工具的 SHA |
| `check_b` (504)、`theta_of` (510) | | |
| `common_checks` (515) | `(ctx, command, fallback)` → `(fc, cfg, seam, qfiles, wsha, rec)` | **t04 驗證「test02 競賽二已完成並 commit」的最佳入口**，見下一小節 |
| `load_opening_test` (680) | `(path, qfiles)` → dict | 讀 test02 的 `opening_test.npz`，檢查：<br>• `meta.format == OPENING_FORMAT`<br>• `seam_bitwise_equal is True`<br>• `frozen_opening_build_id` 相符<br>• 三組的題目檔 SHA 都等於 `meta.inputs`<br>回傳的鍵：`set`、`question_id`、`slot`、`color`、`o`、`has`、`meta`（已解析成 dict）、`sha256` |
| `write_probs` (759) | `(path, rows)` | 寫 `rf.PROB_COLUMNS`（LF），再用 `decide.read_probs` 嚴格讀回一次 |
| `read_pred` (770)、`_lf_bytes` (775) | | |
| `seam_compare` (780) | `(mine_csv, mine_b, ref_csv, ref_b, tol, smoke)` | **通用**，可以拿 `results/prereg04/rank/rank_val_probs.csv`、`pred_B.csv` 當參照。判定：題目集合和順序相同；`n_games`、`n_parsed` 相同；最大 \|ΔP\| ≤ tol；規則 B 不同 ≤ `SEAM_MAX_B_DIFF`=4（smoke 等比例縮小，至少 1）。另外記錄 `byte_identical_probs`、`byte_identical_pred_B`（兩者都折 CRLF 後比較） |
| `level_summary` (901) | `(P, ok, preds)` | 只記錄用 |
| 常數 | | `TEST_FILES` (112)、`SETS` (113)、`T1_OK` (121)、`FAILED_RUN_STATUSES` (122)、`SEAM_MAX_B_DIFF`、`SEAM_QUESTIONS_FULL` (123)、`PARAM_TOL` (124)、`DUMP_ENV_KEYS` (812)、`SEAM_TOL_SAME_GPU`、`SEAM_TOL_OTHER_GPU` (813)、`R20`、`R10` (119-120)、`OPENING_FORMAT` (125) |
| 例外 | | `T02Error(RuntimeError)` (130)；`t2.ERRORS = (T02Error, rp.P02Error, decide.DecideError)` (134)。t04 的 `main` 要一併接住 |

**`common_checks` 在 t04 的用法**：

```python
ctx02 = t2.resolve(argparse.Namespace(dry_run=..., smoke=False, rehearsal=..., results_dir=..., tests_dir=...,
                                      out_dir=<t02 的輸出資料夾>, checks_dir=None))
fc, cfg, seam, qfiles, wsha, rec = t2.common_checks(ctx02, "apply", False)
t2.check_dump_env(ctx02)
```

- `resolve` 只做路徑判斷，不寫檔：real 模式把三個路徑固定在 `results/test02`、`data/test`、`results/test02/rank`；dry-run 要求輸出夾在 repo 外。
- `common_checks` 在 real 模式做的事：
  - `t02_rank.py`、`check_submission.py`、`colab/test02_rank.ipynb` 已 commit 且乾淨。
  - spec 2.3 列的 notebook 輸出都已 commit（`notebook_paths`，368）。
  - `load_notebook`、notebook 身分、`check_test_files`、`check_set_inputs`、`t1_status`，以及權重等於 R20 凍結的 `epoch_9.pth`。
- 這些就是 PREREG_04 5.2「前提：test02 競賽二已完成並 commit、T1 是 PASS／NORMALIZED」的機械化，t04 不用重寫。

### 1.2 `t02_rank.py`：寫死、不能重用的部分

| 名稱（行號） | 寫死了什麼 | t04 怎麼辦 |
|---|---|---|
| `P02_RANK`、`P02_DUMPS`、`SEAM_QUESTIONS`、`REAL_RESULTS`、`REAL_TESTS`、`REAL_OUT`、`FALLBACK_FILE` (91-100) | 路徑固定在 prereg02、test02 | 當常數讀可以，不能拿來當 t04 的輸出位置 |
| `PINS` (103-108)、`FROZEN_OPENING_BUILD_ID` (109) | prereg02 四個檔的 SHA | `pinned(path, name)` 只認這四個名稱。t04 對 prereg04 檔要用自己的釘選表加 `t2.pin_sha256` |
| `DUMP_NAME="p02e9_d4"` (114)、`SET_FILES` (117)、`NOTEBOOK_FILES` (115)、`notebook_paths` (368) | test02 的 dump 檔名 | 另寫 `notebook_paths04`，列 test04 notebook 的輸出 |
| `NOTEBOOK`（246，`colab/test02_rank.ipynb`）、`check_notebook_identity` (263) | 只比 test02 notebook | 用 `t2.notebook_source_sha256(<test04 notebook>)` 另寫。real 模式不符就拒絕，彩排只警告，照 t02 的語意 |
| `readonly_dirs` (285)、`check_out_dir` (290) | 只擋 prereg01、prereg02、data/local | t04 要另外擋 `results/prereg03`、`results/prereg04`、`results/test02`、`data/local`，並規定 real 模式只寫 `results/test04/**` |
| `resolve` (338) | real 模式固定在 test02 | 只用來組 test02 的 ctx（見 1.1）；t04 的路徑另寫 |
| `dump_marker` (816)、`check_dump_env` (820)、`seam_tol` (836) | 讀 `_DONE_val_p02e9_d4.json`；容許差以 `P02_DUMPS` 的 GPU 為參照 | `check_dump_env(ctx02)` 可以拿來核對 **C5**（test02）三組的環境；test04 的 dump 另寫同樣邏輯，參照換成 prereg03 的 marker |
| `apply_set` (696) | 只算 C5、R20（`rf.question_logits`），只回傳 `(rows, info, P, ok, qids)` | Q02 需要 m 和四個 E 項（OP、O4、SWA、SWB），不能用它。可以把它的**驗證步驟**照抄成 t04 自己的 set loader（第 2.4 節） |
| `apply_outputs` (856)、`cmd_*`、`main` | CLI | t04 不呼叫 |
| `load_fallback` (885)、`FALLBACK_COMMIT_BEFORE` (867) | R10 | FAIL_OPENING 時 04 不上場，t04 用不到 |

### 1.3 `tools/t02_test_bundle.py`：helper 可以用，`unpack` 不能擴充

- **能不能不改檔就加一個新的 comp（例如 `rank04`）？不能。**
  - argparse `choices=("rank", "player")`（1056）。
  - 就算直接呼叫 `tb.cmd_unpack(Namespace(comp="rank04", …))` 繞過 argparse，`for n in COMP_FILES[args.comp]`（997）也會 KeyError。
  - 用 `comp="rank"` 的話，成員必須在 `rank/` 底下（`_member_ok`，930-940）；real 目錄只收 `results/test02` 或 `results/test02_rehearsal`，`results/test04` 在 repo 內會被拒（948-952）。
  - 所以 **`tools/t04_rank.py` 要自己寫 `unpack`**。
- **可以直接 import 的**（stdlib，沒有副作用）：

| 名稱（行號） | 簽名 | 說明 |
|---|---|---|
| `_member_ok` (930) | `(name, comp)` | 通用：絕對路徑、`..`、反斜線、`:` 都擋；`parts[0] != comp` 也擋；檔名含答案檔字樣也擋。`comp` 可以傳任何字串，例如 `"sw"` |
| `run_status` (225)、`run_history` (236)、`earlier_failures` (245) | `(run_log)` | 狀態和歷史 |
| `verify_manifest` (657) | `(tests_dir, names=ALL_NAMES)` → `(man, sha)` | t04 要傳 `names=tb.COMP_FILES["rank"]` |
| `check_writable` (148)、`write_json` (191)、`write_bytes` (201) | | 只擋 prereg01、prereg02、data/local（`READONLY`，144），t04 要再加自己的擋 |
| `sha256_file` (159)、`sha256_text_file` (167)、`now_iso` (172)、`inside` (135)、`same_path` (140)、`refuse_forbidden` (125) | | |
| `quick_structure` (277) | `(tests_dir, comp)` | T1.1–T1.3 的子集 |
| 常數 | | `RANK_FILES` (65)、`COMP_FILES` (71)、`MANIFEST_NAME`、`MANIFEST_FORMAT="c10-test02 manifest v1"` (78-79)、`ZIP_NAME="C10_test_v1.zip"` (81)、`KINDS` (84)、`FAILED_RUN_STATUSES` (90)、`Q_PREFIX` (104) |
| 例外 | | **`BundleError` 繼承 `SystemExit`**（117）：`except Exception` 接不到，只能 `except tb.BundleError`，或讓它以 exit 1 結束 |

- **`t04_rank.py unpack` 建議照抄 `cmd_unpack`（943-1029）的每一項檢查：**
  1. 成員：`_member_ok(name, ROOT)`，或用 `p3.member_problem(name, ROOT + "/")`（`p03_rank_fuse.py:1233`，多一道副檔名白名單 `.json .csv .npz .txt .log`）；不可重複。
  2. 一定要有 `ROOT/run_log.json`、`ROOT/config.json`；`run_status == "completed"`，否則只能 `--allow-failed` 解到 repo 外。
  3. 解到 repo 內時，`earlier_failures` 有東西就拒絕。
  4. `config.test_bundle.manifest_sha256` 要等於 `verify_manifest(data/test, RANK_FILES)` 的 SHA；兩個 rank 檔的 SHA 要等於 MANIFEST。
  5. real 模式要求 `smoke is False`、`rehearsal is False`；另外核對 `config.notebook == "colab/test04_rank.ipynb"`（t02 沒有這一項，建議加）。
  6. 目標檔都不存在；先寫 `.tmp` 再改名；最後寫紀錄 `unpack_<ROOT>.json`。
- **建議讓 test04 zip 的成員根目錄不是 `rank/`**（例如 `sw/`，zip 名 `test04_rank_results.zip`）。這樣就算誤用 `t02_test_bundle.py unpack --comp rank`，`_member_ok` 會直接拒絕，test04 的檔不會被解進 `results/test02/rank/`。

  要注意，把 base 的 `pack_results`（`colab/test02_rank.ipynb` 第 7 格）照搬過去的話，成員根目錄等於 `basename(RESULT_DIR)`（`zf.write(p, os.path.relpath(p, os.path.dirname(RESULT_DIR)))`），所以 notebook 的 `RESULT_DIR` 要叫 `.../results/test04/sw`。這個決定交給 U2（notebook）和整合者。

### 1.4 `tools/t02_fake_tests.py`：helper 可以用，`make_rank_dumps` 不行

- `make_rank_dumps(local_dir, p02_rank, tests_dir, res_rank)`（227-275）寫死 `RANK_DUMPS=("val_keys","val_p02e9_d4")`（84）；最後一行 `return info, markers["val_p02e9_d4"]`（275）也寫死了名稱。就算用 monkeypatch 換掉 `RANK_DUMPS`，最後一行還是會 KeyError。
- `_records`（318-332）把 `config.json` 的 `notebook` 寫死成 `f"colab/test02_{comp}.ipynb"`（320）。
- 所以 test04 的替身樹要另寫 builder。建議放新檔 `tools/t04_fake_tests.py`，因為 `t02_fake_tests.py` 在指紋清單裡，不能改。
- 可以重用的 helper：

| 名稱（行號） | 簽名 | 說明 |
|---|---|---|
| `_npz` (188) | `(path)` | 讀成 dict，會擋答案檔 |
| `_save_npz` (193) | `(path, arrays)` | 先寫暫存檔再改名，並做 `tb.check_writable` |
| `_rekey_array` (201) | `(a, prefix)` | `val_…` 換成 `prefix…`；回傳 `np.array(list of str)`，dtype 依最長字串，例如 `pub_q_0001` 是 `<U10`，`priv_q_0001` 是 `<U11` |
| `_question_ids` (208) | `(path)` | 只讀 question_id 欄 |
| `_marker` (220) | `(base, path, rows)` | 複製原 marker，重算 `rows`、`bytes`、`sha256`、`time`，再加 `fake` |
| `from_val` (335) | `(root, zip_kind="rehearsal", zip_out=None, rerun=False, local_dir=…, p02_dir=…, log=print)` | 可以當函式呼叫；演練要證明「原樣的 t02」，所以仍建議走 CLI |
| 常數 | | `PREFIX` (81)、`RANK_SLICES={"public":(0,400),"private":(400,800)}` (82)、`FAKE_NOTE` (88) |
| 例外 | | `FakeError` 也繼承 `SystemExit`（95） |

- from-val 樹的版面（`from_val`，335-410）：
  - `S/data/test/<6 個官方檔名>`、`S/data/test/MANIFEST.json`（kind 用 `--zip-kind`，預設 rehearsal；t1 寫 PASS）。
  - `S/data/colab_bundle/C10_test_v1.zip`。
  - `S/results/test02/rank/{config.json,run_log.json,env.json,timings.json,readonly_before.json,readonly_after.json,seam_check.json}`。
  - `S/results/test02/rank/<set>/dumps/{val_keys.npz,val_p02e9_d4.npz,_DONE_val_keys.json,_DONE_val_p02e9_d4.json,_INPUT.json}`。
  - player 的對應檔案。
  - `S/FAKE_TREE.json`。
- rank 的 public 是驗證題第 1–400 題，private 是第 401–800 題；dump 是列切片（slot 0–4786、4787–9601），只把 `question_id` 換前綴，logits 逐位元組不變。
- `seam_check.json` 是假的：`passed True`、`tol 1e-4`、`bitwise_equal True`。`config.json` 是 `smoke False`、`rehearsal True`、`notebook_sha256 None`。

### 1.5 `tools/check_submission.py`

- `check(kind, sub_path, question_files, candidate_files=(), official=False)`（127）回傳問題清單，空的就是通過。
- `official=True` 時另外要求：
  - 檔名剛好是 `submission_rank.csv`（135-139）。所以 t04 的輸出也要叫 `submission_rank.csv`，只是放在 `results/test04/rank/`。
  - 剛好兩個題目檔，順序是 public 再 private（140-142）；各 400 題，前綴 `pub_`、`priv_`（`_official_questions`，75）。
  - 剛好 800 列，順序等於兩個題目檔接起來（233-243）。
  - 沒有 BOM，表頭剛好是 `question_id,pred_rank`，等級大寫。
- t04 的呼叫方式照 t02：`check_submission.check("rank", sub, [pub_q, priv_q], official=True)`（`t02_rank.py:1012`）。

### 1.6 `colab/build_test02_rank_nb.py`：test04 產生器可以沿用的模式

- **紀律**：
  - 底稿 SHA 先 assert（50、98）。
  - 格子依 `c10` metadata 找，不依位置（103-119）。
  - 只用 `rep()`（目標剛好一次，78-81）和 `cut()`（84-90）改。
  - `CHANGE_ROWS`（144-198）同時餵給第 0 格的表和 `config.json` 的 `changes_from_prereg02`；每個改動過的底稿格都要列在表上（988-993）。
  - 沿用的片段用 ast 核對和底稿逐字相同（1042-1057）。
  - 答案檔字樣只准出現在繼承來的那一行拒收（1000-1007）。
  - 三組的格子只有第一行不同（717、1059-1062）。
  - `NOTEBOOK_SHA256` 的規則見 979-983；重跑產生器輸出要逐位元組相同。
- **Drive 版面**：`MyDrive/C10/`（`DRIVE_DIR`；本機冒煙用 `C10_DRIVE_DIR`）
  - `C10_bundle_v1.zip`：seam_val 題目檔的來源，在 `prepare` 格解壓並逐檔核對。
  - `C10_test_v1.zip`：第一個程式格只讀 `tests/MANIFEST.json` 判斷種類（291-311）：正式或彩排用 `('official','rehearsal')`，SMOKE 用 `('smoke',)`，不符就停。
  - `results/test02|test02_rehearsal|test02_smoke/rank/`：輸出（314-316）。
  - 只讀根目錄 `READONLY_ROOTS={'prereg01','prereg02'}`（319-321）。`guard_write`（333）擋寫入；`readonly_fingerprint`（341）記每個檔的 `[bytes, mtime_ns]`。
- **GPU 關卡**（第 5 格，官方 2）：
  - 沒有 cuda 就停（`OFF2_NEW_GATE`，411-416）。
  - `GPU_MODELS_OK = ('RTX PRO 6000', 'A100')`，比對 `gpu_name_of(ENV_INFO)`。
- **preflight**（428-638）：
  1. 權重 SHA 釘死（448-464）。
  2. 接縫參照的 SHA 釘死，並記下 prereg02 的 GPU（466-482）。
  3. 解壓測試集 zip：只收 `tests/<6 名>` 和 `MANIFEST.json`，每檔的 SHA、大小要等於 MANIFEST（484-512）。
  4. `quick_check_questions`（515-561）。
  5. **④-0：上一次的 `seam_check.json` 不是 `passed True` 就停**（563-571）。訊息請使用者把 Drive 資料夾改名成 `rank_failed_N` 再重跑。
  6. **④：`_INPUT.json` 已存在但內容不同就停**；資料夾有檔卻沒有 `_INPUT.json` 也停（573-596）。
  7. Drive 空間 ≥ 0.5 GB（598-606）。
  8. 只讀目錄的開頭指紋：重跑時和第一次不同就停（608-620）。
  9. 寫 `config.json` 的 `test_bundle`：`zip_sha256`、`manifest_sha256`、`kind`、`t1`、`files{6 名: sha}`；`sets`：`questions_sha256`、`rows`（622-633）。
- **三組的格子**（717-738）：
  - `DUMP_DIR = RESULT_DIR/<SET>/dumps`；`write_keys('val_keys', …)`；`dump_questions(NET, path, 'p02e9_d4', 8, W_LABEL, WSHA)`。
  - 已完成就跳過：`dump_is_done` 看 marker 的 rows、syms、weights_sha256 和檔案 SHA。
- **接縫格**（743-859）：
  - `SEAM_KEYS` 四個陣列要完全相同。
  - C0 是原方向 softmax 在每題內的 float32 平均；最大差 ≤ tol，tol 看**算 seam_val dump 的 GPU** 是否和 prereg02 相同：相同 1e-4，不同 1e-3。
  - 規則 B 不同 ≤ 4 題。
  - 三組 marker 的 `(gpu_name, torch, cudnn)` 必須相同（817-824）。
  - 寫 `seam_check.json`，舊結果保存在 `earlier_runs`（836-840）。
  - 沒過就寫 `status='seam_failed'`、打包、flush、停（848-859）。
- **結尾**（864-933）：
  - 逐組核對 marker、SHA、列數、形狀、權重、NaN 位置、`_INPUT.json`、題數。
  - 寫 `readonly_after.json`；status 是 `completed` 或 `readonly_changed`。
  - `pack_results('test02_rank_results.zip')`，zip 放在 `RESULT_ROOT`，成員從 `rank/` 起；跳過 `.pth`、`.pt`、`.tmp`，和大於 100 MB 的檔。
  - 必要的檔都要在 zip 裡（921-927）；最後 `flush_drive()`。
- **提醒 U2**：
  - prereg03 的網路是 `GoRankResNet(in_channels=501)`，而且「先轉視角再做 D4，d4 和 s0_swap 一次算完」（`colab/build_prereg03_rank_nb.py:224,232,1145`）。
  - test02 的 `dump_questions`、`load_net` 只適用 prereg02 的網路。test04 的 dump 工具要從 prereg03 的 notebook 取，test02 只借 zip、preflight、接縫和收尾的結構。

### 1.7 CPU 冒煙和測試分級

- **`colab/run_local_smoke.py`**：
  - 設定 `C10_SMOKE=1`、`C10_DRIVE_DIR=<S>/drive`、`C10_WORK_DIR=<S>/work`（71）。
  - 只准用 CPU 版 torch（73-75）。
  - 同一個 namespace 逐格 `exec`（80-104）；Windows 上把 `Process`、`Pool` 換成 thread 版。
  - test02 的冒煙還需要：`drive/results/prereg01_smoke`、`prereg02_smoke`，以及 kind 是 `smoke` 的 `C10_test_v1.zip`（spec 2.4）。
  - test04 還要有 **`prereg03_smoke`（兩臂的冒煙權重和 dump）**。這些樹目前不在（舊的 scratch 已清掉），要重建。
- **`tests/test_t02_nb_rank.py`**：不看 `C10_SLOW`，不跑 notebook。內容是：
  - 重建後逐位元組相同（105-115）。
  - 片段和底稿相同（140-184）。
  - 答案檔字樣只出現一次（187-196）。
  - 釘選值等於 repo 裡的檔（199-221）。
  - 用 `exec_defs`（91-98）只執行新格裡的函式定義，拿合成資料測（284-451）。
- **`C10_SLOW` 分級**：
  - `tests/test_t02_dryrun.py`：`1` 或 `2` 跑 `FromValPipeline`（130）；`2` 另外跑 `FromTrainChecks`（420）和 `FromTrainOfficialShape`（531）。
  - `tests/test_t02_rank.py:759`：任何值都跑真實 R20 的自我核對。
  - `tests/test_p04_opp.py:722`：跑真實 selftest，約 2 分鐘。
  - 9/28 的紀錄：`C10_SLOW=2` 的 dryrun 約 526–624 秒。

### 1.8 凍結函式：t04 組 Q02 要用哪些（已用探針驗證）

- **C5**：
  - `rf.load_val_keys(ddir)`（`p02_rank_fuse.py:555`），回傳 `question_id`（list）、`slot`（list[int]）、`color`、`parsed`（bool）、`groups`（`val_groups`：同題必須相鄰，slot 從 1 起連續）、`n`、`sha256`。
  - `rf.load_dump(path, n, 8)`（405），附 `marker`（`read_marker`，419）。
  - `rf.check_val_logits`（572）。
  - 遮罩：`np.where(parsed[:,None,None], logits, 0).astype(np.float32)`。
  - `p3.combo_logprobs(("C5",), {"p02e9_d4": …})`（`p03_rank_fuse.py:464`）。探針確認它和 `rf.source_logprobs("C5", …)` 逐位元組相同。
- **SW**：
  - `p3.load_swap(path, n)`（1025）：float32 [n,10]，syms 必須是 1。
  - `rf.log_softmax(np.where(parsed[:,None], sw, 0).astype(np.float64))`，和 `p04_rank_fuse.val_arm`（955）相同。
  - A0、B0：`rf.log_softmax(np.where(parsed[:,None], d4[:,0,:], 0).astype(np.float64))`。
- **O4、T4**：`p4.side_vectors(z, path)`（554），接著 `np.where(parsed[:,None], o, 0)`，再 `rf.log_softmax`（同 `load_o4_val`，682-716）。
- **OP**：
  - 用 `(question_id, slot)` 對齊，並核對執色（同 `t02_rank.apply_set:725-738`、`p3.opening_val_aligned:961`）。
  - `mop = rf.pool_matrix([[i for i in rows if has[i]] …], n) @ o`。
- **合併**：
  - `S = rf.pool_matrix(cnn_groups, n)`（209）；`m = rf.pooled(S, lpC5, "log")`；`Es = [mop, pooled(S,lpO4), pooled(S,lpSWA), pooled(S,lpSWB)]`，順序是 `fc["combo"]["terms"]` = `['OP','O4','SWA','SWB']`。
  - `z,_ = p4.gen_logits(theta, m[ok], [E[ok]…], n[ok])`（227）；`P[ok] = rf.softmax(z)`。
  - 寫檔用 `repr(float)`；`n_parsed` 為 0 的題機率留空（同 `p04_rank_fuse.py:1797-1811`）。
- **Δκ**：
  - `p4.val_side_kappa(vk, lp_ref, lp_src)`（916），裡面用 `p3.val_colour_subquestions`（732）和 `kappa_questions`（541），不用標籤。
  - `vk` 要有 `groups`、`parsed`、`color`。

## 2. 測試日之後，C5 的測試 dump 和 `opening_test.npz` 在哪裡、長什麼樣子

### 2.1 目錄

照 `docs/TESTDAY.md` 第 4、5 步做完之後：

```
data/test/                                    (gitignore) T1 檢查過的檔（NORMALIZED 時是正規化副本）
  rank_prediction_test_public.csv, rank_prediction_test_private.csv, player_*（4 個）, MANIFEST.json
  raw/<6 個原始檔>, raw/SOURCE.json
results/test02/checks/rank/{t1.json, summary.json, t2..t5 細節}     T1 完整：complete、stage raw、mode real
results/test02/checks/built/rank/t1.json, results/test02/checks/bundle_build.json
results/test02/unpack_rank.json                                   unpack 的紀錄（runs、zip SHA）
results/test02/rank/                                              （全部 commit；.gitattributes 有 -text）
  config.json run_log.json env.json timings.json seam_check.json readonly_before.json readonly_after.json
  seam_val/dumps/ public/dumps/ private/dumps/
      val_keys.npz  val_p02e9_d4.npz  _DONE_val_keys.json  _DONE_val_p02e9_d4.json  _INPUT.json
  opening_test.npz  opening_test_report.json                      ← t02_rank.py opening
  seam_val/{rank_val_probs.csv, pred_A.csv, pred_B.csv, decide_summary.json, seam_local.json}  ← apply
  rank_test_probs.csv pred_A.csv pred_B.csv decide_summary.json submission_rank.csv apply_test_report.json
  fallback_R10.json fallback_selfcheck_R20.json                   （測試日前就已 commit，在指紋清單裡）
results/test02/failed/rank_N/                                     失敗時封存的那一次（TESTDAY 第 4 步）
```

- t04 **只讀** `results/test02/rank/` 目前這一份。`failed/` 底下的一律不讀。
- 找得到 `seam_val/seam_local.json` 而且 `passed`，才算 test02 已完成。

### 2.2 陣列版面（dtype 取自實際檔案和 notebook 原始碼）

**`<set>/dumps/val_keys.npz`**：由 `write_keys('val_keys', V['keys'])` 寫出。

| 鍵 | dtype、形狀 | 內容 |
|---|---|---|
| `question_id` | `<U{最長}` [N] | 驗證是 `<U10`（`val_q_0001`）。官方 id 格式不知道，只保證 `pub_`、`priv_` 開頭 |
| `slot` | `int32` [N] | 1..num_games |
| `color` | `<U1` [N] | 題目檔的 `color_i`（B 或 W） |
| `parsed` | `bool` [N] | 是否解析成功 |

- 列序是題目檔的列序，同一題內依 slot 排；每題的 slot 都在，包括解析失敗的。
- 驗證 seam_val 有 N = 19,156 盤、1,589 題；假樹的 public 有 4,787 盤、400 題，private 有 4,815 盤、400 題。

**`<set>/dumps/val_p02e9_d4.npz`**：
- `logits` 是 float32 [N, 8, 10]，解析失敗的列整列是 NaN。
- `weights_sha256` 是 0 維的 `<U64`，值為 `423581b4…`；`syms` 是 0 維 int64，值為 8。
- 等級欄的順序是 `c10data.LEVELS`：D C B A 1D 2D 3D 4D 5D 6D。

**`_DONE_<stem>.json`**：由 `write_dump` 寫出。
- 鍵：`file`、`rows`、`syms`、`bytes`、`sha256`、`weights`、`weights_sha256`、`seconds`、`time`、`run`、`gpu_name`、`torch`、`cudnn`、`device`。
- `val_keys` 那一份的 `syms`、`weights`、`weights_sha256` 都是 None。
- prereg02 和 prereg03 的 marker 都是 `NVIDIA RTX PRO 6000 Blackwell Server Edition`、`2.11.0+cu128`、`91900`。

**`_INPUT.json`**：`{"set": s, "question_file": "bundle local/rank_val_questions.csv" | "tests/<name>", "sha256": …}`。

**`config.json`**：
- 頂層有 `smoke`、`rehearsal`、`test_kind`、`notebook`（`colab/test02_rank.ipynb`）、`notebook_sha256`、`gpu`、`torch`、`cudnn`、`weights{path,sha256}`。
- `test_bundle{zip_sha256, manifest_sha256, kind, t1, files{6 名}}`、`sets{s:{questions_sha256, rows}}`、`seam_reference`、`status`。
- 每次執行的設定放在 `runs`。

**`run_log.json`**：`{"runs":[{run,start,end,status,…}], "resume_count", "status"}`，status 是 `completed`、`seam_failed` 或 `readonly_changed`。

**`seam_check.json`**：
- 判定用的鍵：`passed`、`tol`、`bitwise_equal`、`max_abs_c0_diff`、`max_abs_logit_diff`、`keys_equal{4}`、`rule_b_diff{questions,allowed,question_ids}`。
- 環境相關的鍵：`same_gpu_as_prereg02`、`dump_env`、`dump_env_same`、`seam_dump`、`prereg02_dump`、`earlier_runs`。

### 2.3 `opening_test.npz`（`t02_rank.py:655-671`）

| 鍵 | dtype、形狀 | 內容 |
|---|---|---|
| `set` | `<U8` [M] | `seam_val`、`public`、`private` 依序接起來 |
| `question_id` | `<U…` [M] | |
| `slot` | `int16` [M] | |
| `color` | `<U1` [M] | |
| `ll` | `float64` [M,10] | |
| `o` | `float64` [M,10] | `has` 為真時是 `log_softmax(ll)`，否則是 0 |
| `has` | `bool` [M] | |
| `n_terms` | `int16` [M] | |
| `n_moves` | `int16` [M] | |
| `meta` | 0 維 unicode | JSON 字串，內容見下 |

- 列序是 `ro.read_questions(qfile)` 的順序，也就是題目檔的列序、每題 slot 1..num_games，和 val_keys 相同。
- seam 部分的 `ll`、`o`、`has`、`n_terms` 和 `results/prereg02/rank/opening_val.npz`（SHA `75ffec64…`）逐位元組相同。
- `meta` 的內容：`format="c10-test02 rank opening_test v1"`、`params`、`build_id`、`frozen_opening_build_id="c0cb86b4…"`、`seam_bitwise_equal`、**`created`（時間戳，所以整個檔不能逐位元組比）**、`fit_games`、`inputs{s:{path,sha256,slots,questions}}`、`reference`、`tool_sha256`。
- t04 讀這個檔一律經過 `t2.load_opening_test(path, qfiles)`，用 `(question_id, slot)` 對齊並核對執色。

### 2.4 t04 讀 test02 的 C5：照抄 `apply_set` 的驗證步驟（697-719）

1. `vk = rf.load_val_keys(ddir)`。
2. `d = rf.load_dump(ddir/val_p02e9_d4.npz, vk["n"], 8)`，並核對 `d["weights_sha256"] == fc["dumps"]["p02e9_d4"]["weights_sha256"]`。
3. `rf.check_val_logits(d, vk["parsed"])`。
4. `questions = rf.read_question_meta(qfile)`：
   - `qids` 要等於題目檔的題序；smoke 只要求是開頭的幾題。
   - 每題的 slot 數和執色清單要等於題目檔。

### 2.5 SHA 來源

| 值 | 來源 |
|---|---|
| R20 `frozen_config.json` `44a16e09…`、`opening_val.npz` `75ffec64…`、`rank_val_probs.csv` `a8370043…`、`pred_B.csv` `863a6258…` | `t02_rank.PINS`（103-108）、`tests/test_t02_hygiene.py` |
| `epoch_9.pth` `423581b4…`、seam 參照 `val_keys.npz` `055d2941…`、`val_p02e9_d4.npz` `ec0b729d…`、prereg02 的 GPU | `build_test02_rank_nb.py:62-66`（產生時會從 repo 重讀並 assert） |
| test02 notebook：檔案 SHA `53337a7a…`；`NOTEBOOK_SHA256`（原始碼規則）`dc7526be…` | `r20_path_fingerprint.json`、notebook 第 1 格 |
| prereg03 兩臂權重：A `790b767c…`、B `8e88f6ec…` | `p04_rank_fuse.SW_WEIGHTS`（174）、prereg03 的 marker |
| SW 驗證 dump：A `b355c26b…`、B `0cadff4b…` | `p04_rank_fuse.SW_SHA256`（169-173） |
| prereg03 的 d4 驗證 dump：A `1e4e5957…`、B `bad2277d…` | prereg03 的 `_DONE_val_p03{a,b}_d4.json`，以及 `results/prereg03/rank/check_report.json` |
| prereg03 的 `val_keys.npz` | 兩臂都是 `055d2941…`，等於 prereg02 的 |
| prereg04：`rank_val_probs.csv` `3c7ab658…`、`pred_B.csv` `27b6d225…`、`pred_A.csv` `6a6bb3c2…`、`frozen_config.json` `ae728e9c…`（原始位元組）、`val_o4.npz` `30d312c0…`、counts `f6c6debe…`（`data/prereg04/counts_fit.npz`，gitignore） | 工作樹的檔，都是 LF；count 的數字取自 `frozen_config.json` 的 `source_sha256.O4` |
| 假樹的 R20 交件檔 `d9b9801b3df85c75e5932ce228eebbeb8a76c43b40d2efd02700ce8807490d97` | `results/test02/design/build_review.md:109`；前 12 碼 `tests/test_t02_dryrun.py:65` |

## 3. 假測試演練的設計（照原樣的 t02，再接 t04）

### 3.1 流程

以下的 `S` 是 repo 外的 scratch 資料夾。

1. `PY tools/t02_fake_tests.py from-val --root S`：照原樣呼叫。
2. `PY tools/t02_checks.py t1 --comp rank --dry-run --root S --only T1.1,T1.2,T1.3`：得到 `S/results/test02/checks/rank/t1.json`，verdict PASS。
3. R20 管線，照原樣呼叫：
   - `PY tools/t02_rank.py opening --dry-run --results-dir S/results/test02 --tests-dir S/data/test --out-dir S/out/rank`
   - `PY tools/t02_rank.py apply --dry-run …`（同一組參數）
4. **R20 的標準值斷言**，也就是 (a)。細節在 3.2。
5. **記快照**：`S/results/test02/**`、`S/out/rank/**`、`S/data/test/**` 每個檔的 SHA。
6. **建 test04 替身樹**（新工具，例如 `PY tools/t04_fake_tests.py from-val --root S`）。來源是 `results/prereg03/rank_{a,b}/dumps/`：

   ```
   S/results/test04/<ROOT>/<set>/dumps/
       val_keys.npz               seam_val：prereg03 A 臂的 val_keys.npz 逐位元組複製（= prereg02 的 055d2941…）
       val_p03a_d4.npz            float32 [N,8,10], weights_sha256 790b767c…, syms 8
       val_p03a_s0_swap.npz       float32 [N,10],  weights_sha256 790b767c…, view 's0_colour_swapped'
       val_p03b_d4.npz / val_p03b_s0_swap.npz   同上（8e88f6ec…）
       _DONE_<stem>.json          seam_val：複製 prereg03 的 marker；public、private：t02_fake_tests._marker 重算
       _INPUT.json                {"set", "question_file", "sha256"}，同 t02 的做法
   ```

   - public、private 的來源列範圍用和 `make_rank_dumps` 同一套算法：`bounds = 每題第一個 slot 的列號`，再取 `[bounds[lo], bounds[hi])`。
   - 題序要核對等於 `data/local/rank_val_questions.csv` 的 question_id 欄（只讀 id），也要等於 `S/data/test` 的題目檔換前綴後的 id。
   - `question_id` 用 `_rekey_array` 換前綴；`logits` 用切片，要驗證切片後的 `tobytes` 等於原檔對應的區段；`weights_sha256`、`syms`、`view` 原樣保留。
   - 另寫紀錄檔：
     - `config.json`：`smoke False`、`rehearsal True`、`notebook "colab/test04_rank.ipynb"`、`notebook_sha256 None`、`gpu`、`torch`、`cudnn` 取自 prereg03 的 marker、`weights{a:{path,sha256}, b:{…}}`。`test_bundle` **直接複製 `S/results/test02/rank/config.json` 的 `test_bundle`**，因為用的是同一份 zip。`sets` 另外寫。
     - `run_log.json`：`{"status":"completed","runs":[{"run":1,"status":"completed"}]}`。
     - `seam_check.json`：`{"passed":true,"tol":1e-4,"bitwise_equal":true,…}`。
     - `env.json`、`timings.json`、`readonly_before.json`、`readonly_after.json`。
     - 在 `FAKE_TREE04.json` 記下來源 SHA 和規則，並註明「驗證衍生，不跑內容統計」。
   - 這樣 `t2.load_notebook`、`t2.check_test_files`、`t2.check_set_inputs` 都能直接套用在這棵樹上。
7. **O4 重播**：`PY tools/t04_rank.py opp --dry-run --tests-dir S/data/test --out-dir S/out04/opp --workers ≤6`。
   - seam_val 的每個陣列（`question_id`、`slot`、`color`、`o4`、`t4`、`n_moves`、`anomalies`、`moves_parsed`、`replay_failed`）都要和 `results/prereg04/opp/val_o4.npz` 的 `tobytes` 相同。
   - public 的陣列要等於 `val_o4` 的 `[0:4787]` 區段（只換前綴），private 等於 `[4787:9602]` 區段。
   - 三組的重播失敗都是 0。
8. **apply**：`PY tools/t04_rank.py apply --dry-run`，參數有：
   - `--t02-results S/results/test02 --t02-out S/out/rank`：讀 `opening_test.npz`，並核對 test02 的 apply 報告。
   - `--results-dir S/results/test04 --opp-dir S/out04/opp --tests-dir S/data/test --checks-dir S/results/test02/checks --out-dir S/out04/rank`。
9. **斷言 04 的結果**，也就是 (b)，細節在 3.3；再核對第 5 步的快照完全沒變。
10. **負面情境**，細節在 3.4。

### 3.2 (a) R20 的 `submission_rank.csv` 和 main 逐位元組相同

- **對照的標準值**：`S/out/rank/submission_rank.csv` 的 SHA-256 = `d9b9801b3df85c75e5932ce228eebbeb8a76c43b40d2efd02700ce8807490d97`。這是 main 上 build_review 1.2 記錄的值；rev 8 之後的 `C10_SLOW=2` 也用前綴 `d9b9801b3df8` 驗過。另外還有：
  - `seam_val/rank_val_probs.csv` = `a8370043…`，`seam_val/pred_B.csv` = `863a6258…`。
  - `rank_test_probs.csv` 的 800 列等於 prereg02 驗證第 1–800 列換前綴（同 `test_t02_dryrun.py:247-257`）。
  - `check_submission.check(..., official=True) == []`；`apply_test_report.json` 的 `combo.id == "R20"`，`fallback_used` 是 False。
- **建議在寫任何 t04 程式之前先記標準值**：在目前的分支（等於 main `6ec5e68`）跑一次第 1–3 步，把每個確定性輸出的完整 SHA 寫進 `results/test04/design/r20_fake_golden.json` 並 commit。
  - 確定性的輸出有：`seam_val/{rank_val_probs,pred_A,pred_B}.csv`、`{rank_test_probs,pred_A,pred_B,submission_rank}.csv`；`opening_test.npz` 除了 `meta` 以外的每個陣列。
  - `decide_summary.json` 記了絕對路徑，`*_report.json`、`seam_local.json` 有時間戳，都不比。
  - 之後在分支上、合併的那一刻、合併後，每次都用 `tests/test_t04_dryrun.py` 比這份標準值。這就是 PREREG_04 5.1 說的「和合併前逐位元組相同」。
- **程式碼本身沒變的證明**（和標準值互補）：
  - `tests/test_p04_hygiene.py`：39 個指紋檔都沒變，而且指紋檔等於 tag `r20-testday-pre04`。
  - `tests/test_p02_hygiene.py`、`test_t02_hygiene.py`：凍結的 p02 工具、decide、c10data 都沒變。
  - 合併前再跑 `git diff main...p04-testday --stat`，確認沒有碰到任何指紋檔、`tools/p0*`、`decide.py`、`c10data.py`。
- **t04 沒有寫進 test02 的證明**：第 5 步的快照在 t04 跑完後完全相同。
  - 這一點要自己寫測試，因為 `test_p04_hygiene` 只查「指紋檔改了沒」，不查 `results/test02/` 底下有沒有**新檔**（見它的 docstring 第 3 點）。

### 3.3 (b) 04 的交件檔對得回 `results/prereg04/rank/pred_B.csv` 和 `rank_val_probs.csv`

**探針結果**（`q02_replay.py`，只用第 1.8 節列的 import 函式，沒有呼叫 `p04_rank_fuse.cmd_apply`）：

```
C5 via p3.combo_logprobs == rf.source_logprobs bytes: True
full 1589 rows byte-identical to prereg04 rank_val_probs.csv: True 1589
public rows (val 1-400 as pub_) equal: True | private rows (val 401-800 as priv_) equal: True
dkappa all {'O4': 0.10503, 'SWA': 0.03631, 'SWB': 0.03436}        ← 等於 RESULTS 記的 +0.1050 / +0.0363 / +0.0344
dkappa public {'O4': 0.09298, 'SWA': 0.03213, 'SWB': 0.02784}
dkappa private {'O4': 0.09999, 'SWA': 0.04181, 'SWB': 0.03639}
```

- 所以每一題的計算和其他題無關：pool matrix 的每一列只加總自己那題的 slot，而且順序固定；`gen_logits` 和 `softmax` 都是逐列算。
- 把 400 題當成一組單獨算，結果和放在 1,589 題裡算逐位元組相同。

**演練要斷言的事項**（t04 的輸出在 `S/out04/rank/`）：

1. `seam_val/rank_val_probs.csv` 的 SHA = `3c7ab658…`，`seam_val/pred_B.csv` = `27b6d225…`，`seam_val/pred_A.csv` = `6a6bb3c2…`。三者都和 prereg04 逐位元組相同。
   - `t2.seam_compare` 的結果要是 `passed`、`max_abs_diff 0.0`、`rule_b_differences 0`，`byte_identical_*` 都為真。
2. `rank_test_probs.csv` 有表頭加 800 列。第 i 列（i < 400）等於 `b"pub_" + ref[1+i][4:]`；第 400–799 列等於 `b"priv_" + ref[1+i][4:]`。ref 是 prereg04 的 `rank_val_probs.csv` 折 LF 後的各行。
3. `submission_rank.csv` 等於 `pred_B.csv` 的位元組；各列等於 prereg04 `pred_B.csv` 第 1–800 列換前綴；`check_submission.check("rank", sub, [pub, priv], official=True) == []`。
4. 合理性檢查（不用標籤）：假樹上 04 和 R20 交件檔不同的列數，要等於 prereg02 和 prereg04 的 `pred_B.csv` 在第 1–800 題裡不同的題數。
5. 執色檢查的記錄等於上面探針的 6 個 Δκ 值，而且都 > 0，所以 04 可用。第一次跑出綠燈後，把完整精度的值寫進測試當標準值。
6. O4 重播失敗三組都是 0（見 3.1 第 7 步）。

### 3.4 負面情境（快速的單元測試用合成資料，慢速的演練用假樹副本）

- test04 的 `val_keys` 和 test02 的不同，例如把一個 `parsed` 反轉：要拒絕（t04 不能自己遮罩）。
- test04 某組 marker 的 `gpu_name`、`torch`、`cudnn` 和它的 seam_val 不同：拒絕。
- `seam_check.json` 沒過，或 `run_log` 有更早失敗的 run：拒絕。在 real 模式驗。
- `S/results/test02/checks/rank/t1.json` 改成 FAIL_OPENING：要印「04 不上場」，不寫任何輸出。
- test02 的 `seam_local.json` 沒過，或 `apply_test_report.json` 的 `fallback_used` 為真：拒絕。
- 某一組 O4 重播失敗 > 0.5%（在副本裡把一盤的 SGF 弄壞）：「04 不用」。
- 把某個來源的 Δκ 弄成 ≤ 0（在副本裡把 SW 換成 A0）：「04 不用」。
- 輸出夾在 `results/prereg0x`、`results/test02`、`data/local` 底下：拒絕。
- 用 `t02_test_bundle.py unpack --comp rank` 解 test04 的 zip：因為成員根目錄不是 `rank/`，要被拒絕。
- 沒有 `--rerun --reason` 就覆寫：拒絕。

### 3.5 時間和資源

- 假樹約 0.55 GB。
- t02 的 from-val 管線約 3–4 分鐘，其中 opening 要用 `load_rows(part="fit")` 讀 fit 棋譜。
- O4 重播 seam_val 加兩組，共約 28,758 個 slot。依 inner 73,556 盤、8 worker 270 秒推算，6 個 worker 約 2 分鐘（推估）。重播前要先讀 178 MB 的 counts 檔、建表。
- 建議掛在 `C10_SLOW=1` 或 `2`，和 `FromValPipeline` 同級；`C10_SLOW=2` 的總時間約再多 5 分鐘（推估）。
- 暫存放在 `C10_DRYRUN_DIR`，或 repo 外的 tempfile。

## 4. 風險

1. **抄 t02 的程式會漂移。** 把 `apply_set` 的驗證步驟抄進 t04，或把 R20 的組裝重寫一遍，都可能和凍結程式不一致。
   - 能 import 的一律 import（1.1、1.8）。
   - 用 3.3 的逐位元組接縫把關：seam 要等於 prereg04 的 csv，測試列要等於換前綴的驗證列。
2. **模組常數被綁死。** 寫成 `from t02_rank import X`，測試就 patch 不到；另外，t02 的 `T02Error` 是 `RuntimeError`，但 `BundleError`、`FakeError` 繼承 `SystemExit`，`except Exception` 接不到。
3. **t02 的 `unpack` 不能收 test04 的 zip**，要另寫。成員根目錄如果仍是 `rank/`，誤用 t02 unpack 時，只有在 test02 已經 unpack 過的情況下才會因為「目標檔已存在」被擋。建議 test04 用不同的成員根目錄（1.3）。
4. **C5 和 SW 來自兩份 notebook、兩次執行。** 各自的環境一致性只能在各自的 notebook 裡保證。
   - 04 本機接縫的容許差怎麼組合，要在修訂裡寫明。建議：test02 的 seam_val dump 的 GPU 等於 prereg02 的，**而且** test04 兩臂的 seam_val dump 的 GPU 等於 prereg03 的，才用 1e-4，否則 1e-3。
   - 兩份 notebook 用不同 GPU 算是合法的，不能要求它們相同。
5. **test04 的 `parsed` 可能和 test02 不同。** prereg03 的特徵是 501 層、不同的轉換，雖然在驗證集上 `val_keys` 相同（`055d2941…`），官方測試棋譜未必。
   - 不同時，t04 必須停下，不能自己遮罩；PREREG_04 的 SW 結構檢查要求 NaN 列等於 C5 的。
6. **T1 是 FAIL_OPENING 時，`t2.t1_status` 的錯誤訊息提示「改用 R10」。** t04 要改寫成「04 不上場」，以免誤導使用者或 Claude。
7. **`.gitattributes` 沒有 `results/test04/** -text`。** 這台機器 `core.autocrlf=true`，commit 的 csv、json 在新的 checkout 可能變成 CRLF，釘選和逐位元組比較就會失敗。
   - 建議只新增一行；`.gitattributes` 不在指紋清單裡。
   - t04 核對 prereg04 的 csv、json 時用 `t2.pin_sha256`（折 LF）。
8. **`official_log.py` 的預設方法名稱是 R20**（`DEFAULT_METHOD`，63）。`check_reports`（310-358）讀的是交件檔同一個資料夾裡的 `apply_test_report.json` 和 `seam_val/seam_local.json`。
   - t04 的報告要用相同的欄位名：`mode`、`seam_local.passed`、`check_submission.{mode,errors}`、`outputs_sha256["submission_rank.csv"]`、`fallback_used`，才能直接套用。
   - 方法名稱要用附加式修改，讓它認得 04；否則上傳紀錄會把 04 記成 R20。`--method` 只擋提到 R10 的字串。
9. **`data/prereg04/counts_fit.npz` 被 gitignore**（178 MB，SHA `f6c6debe…`）。測試日前它一定要在；不見了要重算，8 個 worker 約 34 分鐘。O4 重播也需要 `results/prereg04/opp/count_report.json` 等 commit 過的紀錄。
10. **`test_p04_hygiene` 查不到 `results/test02/` 底下的新檔**，t04 誤寫進去不會被抓到；要靠 t04 自己的輸出守衛和 3.2 的快照測試。另外它會查 `results/prereg03`、`prereg04/design` 有沒有新檔或 bytecode，t04 和它的測試不能在那裡留下任何東西。
11. **test04 notebook 的 CPU 冒煙需要 `prereg03_smoke` 樹**（兩臂的冒煙權重和 dump），目前不存在，要先重建（成本看 prereg03 產生器的冒煙設定）。這是 U2 的範圍，但會影響時程。
12. **標準值和假 id 綁在一起。** `d9b9801b…` 和 04 假交件檔的 SHA 只對 `t02_fake_tests` 的 id 規則成立；`t02_fake_tests.py` 已凍結，所以不會變。
    - `opening_test.npz`、`decide_summary.json` 和各報告有時間戳或絕對路徑，不能整檔比。
13. **官方結構是每題單一執色**，Δκ 的子題就是整題。假樹用的是驗證題，有混色，所以演練只能證明程式路徑正確，不能預測官方檔上 Δκ 的正負。
14. **test02 重跑過（`failed/rank_N`）時，t04 只能用目前 `results/test02/rank/` 的那一份。** 用 `t2.common_checks` 的 real 模式可以擋掉未 commit 或不一致的狀態。
15. **時程**：K5 是 11/03 23:59。演練每跑一輪 t02 加 t04 約 6–10 分鐘（推估）；本機 CPU 最多 6 個 worker，而且不能和其他線的重工作同時跑。
