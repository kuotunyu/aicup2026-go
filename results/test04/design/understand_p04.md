# U2 調查：p04 工具（O4 算在測試題上、Q02 融合）— 給 `tools/t04_rank.py` 的依據

- 讀過：`tools/p04_opp.py`、`tools/p04_rank_fuse.py`、`tests/test_p04_{opp,rank_fuse,hygiene}.py`、`results/prereg04/{opp,rank}/*.json`、`results/prereg04/build_review.md`、`results/prereg04/RESULTS.md`、`docs/PREREG_04_OPPONENT.md` 第 1、2、5、10 節，以及它們 import 的 p02／p03 函式和 `tools/t02_rank.py` 可重用的部分。
- repo 只讀。實測放在 scratch `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-solar\b3b12887-5b96-4ead-ab34-78eb6f69fc09\scratchpad\p04testday\U2\`：
  - `o4_recompute.py` → `o4_recompute.json`、`val_o4_recomputed.npz`、`val_o4_from_question_file.npz`
  - `fuse_recompute.py` → `fuse_recompute.json`、`rank_val_probs_recomputed.csv`
  - `kappa_noise.py` → `kappa_noise.json`
- 沒開任何答案檔、沒跑 `score.py`、沒用 GPU、worker ≤ 6；`results/prereg04/**` 沒被寫入（git status 乾淨）。
- 環境：`.venv` Python 3.12.13、numpy 2.5.3、scipy 1.18.1、zlib 1.3.1，和 `val_report.json` 記的相同。

## 結論

1. **O4／T4 完全可以靠 import 算，不必改任何 p04 檔。** `p04_opp` 的 `registered_counts → load_counts → build_tables → tables_record → read_question_slots → slots_match_keys → val_scores → save_npz` 都不綁驗證集。只有 CLI 的 `cmd_val` 綁死題目檔、題數、輸出位置和「只跑一次」，t04 不呼叫它。
2. **seam_val 的 O4 可以做到整個 `.npz` 檔逐位元組相同（9/30 實測）。** 照 `cmd_val` 的陣列順序和 dtype 重算，SHA-256 是 `30d312c0…`，等於已 commit 的 `results/prereg04/opp/val_o4.npz`，也等於 `val_report.json` 的 `output_sha256`。npz 裡每個 zip 成員的時間戳固定是 1980-01-01，不是寫入時間，所以登記的「逐位元組相同」照字面就做得到，不必降級成「陣列相同」。
3. **融合配方已驗證。** 只用可 import 的函式重組 `cmd_apply` 的數值核心，吃已 commit 的驗證輸入：
   - 重算的 `rank_val_probs.csv` 和 `results/prereg04/rank/rank_val_probs.csv` 逐位元組相同（`3c7ab658…`）。
   - 最大機率差 0，規則 B 和 `pred_B.csv` 差 0 題。
4. **執色檢查有現成函式。** `p04_rank_fuse.val_side_kappa(vk, lp_ref, lp_src)` 用在官方單色題上時，每題剛好是一個子題，回傳的 `dkappa` 就是登記的每池 Δκ。驗證集單色子題的值：O4 +0.1050、SWA +0.0363、SWB +0.0344。重算到 16 位數，和 apply／check 報告相同。
5. **CPU 時間。**
   - seam_val 19,156 盤：6 個 worker 77 秒；登記那次 8 個 worker 58 秒。
   - public＋private 約 10,000 盤：估 40 秒左右。
   - 融合：< 5 秒。
   - `t04 opp` 全部約 2–2.5 分鐘。
6. **修訂必須決定的三件事：**
   - (a) 測試集允許重播失敗 ≤ 0.5%，但 p04 只處理「0 盤失敗」。失敗盤怎麼處理要寫明，建議從 E_O4 和 O4／T4 的 κ 排除，見第 6 節決定 1。
   - (b) 開局項不能用 `p04_rank_fuse.val_question_inputs`，因為它的 build_id 綁在 prereg02 的 `opening_val.npz`。測試題要用 test02 的 `opening_test.npz`。
   - (c) 每池只有 400 題，Δκ 有抽樣雜訊。6 個檢查合計「其實可以用、卻誤判不用 04」的機率粗估 ≤ 2%。

## 1. t04 可以 import 的函式，以及寫死的驗證集假設

### 1.1 O4／T4：`tools/p04_opp.py`

| 函式（行） | 簽名、回傳與注意事項 | 綁驗證集？ |
|---|---|---|
| `registered_counts(c, counts_arg)`（:1096） | 參數 `c` 在函式裡沒有用到，呼叫 `registered_counts(None, None)` 即可。會要求 `results/prereg04/opp/count_report.json` 已追蹤且工作區乾淨、`status == "ok"`、`preview is False`，並核對 `data/prereg04/counts_fit.npz` 的 SHA-256 等於報告的 `counts_sha256`（`f6c6debe…`）。回傳 `(COUNTS_REAL, rec)` | 只綁登記的計數，正是 t04 要的 |
| `load_counts(path)`（:566） | 回傳 `{C_t, A_t, C_o, A_o}`，各為 int64 `(10, 557928)`；形狀或型別不對就 `P04Error` | 否 |
| `build_tables(counts)`（:579） | 回傳 `(tabs, info)`。`tabs` 是 float64 `[2, 557928, 10]`，index 0＝目標方的表（T4，t_choice_a20），1＝對手方的表（O4，o_choice_a20）。`info = {"t": {"r0", "moves"}, "o": {...}}` | 否 |
| `tables_record(tabs)`（:594） | 回傳 `{"tables_sha256", "tables_array"}`，SHA 取 C 順序 float64 位元組。必須等於 `frozen_config.json` 的 `source_sha256.O4.tables_sha256`，也就是 `f380e0d720edd4429e4858ab87a3e1a5efbbd9de84982c002ab214f19f120de5`（`val_report.json` 記的也是這個） | 否 |
| `read_question_slots(path)`（:1268） | 回傳 `[(question_id, slot, colour, sgf)]`，slot 是 1..num_games，依檔案順序。讀檔用 `utf-8-sig` 和 `csv.field_size_limit(10**8)`，只讀 `question_id`、`num_games`、`color_i`、`sgf_i`。`color_i.strip()` 不是 `B`／`W` 就 `P04Error`；路徑含 answer 字樣就拒絕 | 否，任何題目檔都能讀 |
| `slots_match_keys(slots, vk)`（:1284） | 逐列比對 `(question_id, slot, colour)` 和 `vk`，不同就 `P04Error`。只用 `vk["n"]` 和這三個陣列 | 否 |
| `val_scores(slots, tabs, workers=8, tmp_root=None)`（:1294） | 目標執色取題目檔的 `color_i`。回傳 dict：<br>・`o4`、`t4`：float64 `[N,10]`<br>・`n_moves`：int64 `[N,2,3]`，[目標方, 對手方] × 3 個階段，只算異常之前的手<br>・`anomalies`：int64 `[N]`<br>・`moves_parsed`：int64 `[N]`，MOVE_RE 抓到的全部手數<br>・`replay_failed`：bool `[N]`，定義是 `(moves_parsed == 0) \| (anomalies > 0)`（:1304） | 否 |
| `score_games(games, tabs, workers, tmp_root, chunk=200)`（:639） | `games = [(row, uint16 moves, 1 或 2)]`，列號要是 0..N-1。用 spawn 的 Pool；`tabs` 先存成 `tmp_root` 底下的 `tables.npy`（89 MB）再給 worker mmap，跑完刪除 | 否 |
| `score_game(m, tc, tab)`（:610）、`side_vectors(ll)`（:663） | 單盤計算，`ll` 形狀 `[2,3,10]`；`(t4, o4) = (ll[:,0].sum(1), ll[:,1].sum(1))` | 否 |
| `encode_one(sgf)`（:365）、`target_colour_value(col)`（:371） | MOVE_RE 編碼成 uint16；`'B'→1`、`'W'→2` | 否 |
| `save_npz(path, arrays)`（:823） | `np.savez_compressed` 先寫 tmp，再 `os.replace`；拒絕含 answer 字樣的路徑 | 否 |
| `sha256_file`（:676）、`refuse_answer_path`（:669）、`environment()`（:685）、`tool_state()`（:691）、`require_own_committed()`（:728）、`git_tracked_clean()`（:736）、`require_frozen_config()`（:757） | 來源紀錄和前提檢查：<br>・`tool_state()` 核對 p02_*、c10data、decide、p03_rank_fuse 都等於 HEAD，並給出 `p04_opp.py` 的 LF SHA，這個值要等於 `frozen_config.json` 的 `tool.files_sha256["p04_opp.py"]`（`b594e0ef…`）。<br>・`require_frozen_config()` 在分支上照樣會過（frozen_config 已 commit、選中的組合含 O4），可以當 t04 的前提，並把 SHA `ae728e9c…` 記進報告 | 否 |

- **import 的代價**：`import p04_opp` 約 1 秒，其中約 0.47 秒是 import 時執行的 `_canon_table()`。它會一併 import `p03_rank_fuse` 和全部 p02 模組。
- spawn 的每個 worker 都會重新 import 一次，所以 t04 腳本一定要有 `if __name__ == "__main__":`。

**寫死的驗證集假設**（t04 都要繞開，不呼叫這些）：
- **`cmd_val`**（:1309–1352）：
  - 題目檔固定為 `QUESTIONS = data/local/rank_val_questions.csv`（:100），key 固定為 `P02_DUMPS/val_keys.npz`（:101）。
  - 題數與 slot 數固定 `VAL_QUESTIONS, VAL_SLOTS = 1589, 19156`（:125），檢查在 :1323–1325。
  - 輸出只准寫 `results/prereg04/opp/val_o4.npz`，或 repo 外的預覽（`resolve_out` :836–856）。
  - 正式模式下檔案已存在就拒絕（:1316–1317）；`replay_failures > 0` 就是 `status: stop`（:1346–1347）。
- **其他指令**：`cmd_dupcheck` 同樣綁 `VAL_*`（:1440–1442）；`cmd_inner` 綁 `INNER_GAMES = 73556`（:124）；`main()` 只有 selftest、count、inner、dupcheck、val 五個指令。
- **答案檔的拒絕**：`refuse_answer_path`（:669–673）只要絕對路徑轉小寫後含 answer 字樣就拒絕。t04 也要有一樣的拒絕，而且原始碼裡不要直接寫出那個字，照 `tests/test_p04_hygiene.py` 的 `NEEDLE = "ans" + "wers"` 寫法。hygiene 的 `NoLabelFileName` 只掃 `tools/p04_*.py`，所以 t04 要有自己的測試。

### 1.2 融合：`tools/p04_rank_fuse.py`，以及它用到的 p02／p03 函式

**可以直接 import 的：**
- **`pf.gen_logits(th, m, Es, n)`（:227）**：`t = e^a·(n/12)^γ`；`u = m + B @ beta`，再依序加上 `w[j]·Es[j]`；回傳 `(t[:,None]*u, t)`。這是從設計版 fuse.py 逐字移植的，`GenCalibration.test_port_is_verbatim` 有核對。
- **常數**：
  - `pf.TERMS = ("OP","O4","SWA","SWB")`（:183）
  - `pf.PAIRS = (("O4","T4"),("SWA","A0"),("SWB","B0"))`（:182）
  - `pf.COMBOS`（:184）、`pf.B = basis()`（:179）
  - `pf.SW_SHA256`、`pf.SW_WEIGHTS`（:169–175）
- **`pf.val_side_kappa(vk, vlp_ref, vlp_src)`（:916）**：見第 4 節。
- **`pf.side_vectors(z, path)`（:554）**：從 npz 讀 `o4`／`t4`，檢查是 float64 `[N,10]`。注意它和 `p04_opp.side_vectors` 是不同的函式。
- **`pf.frozen_tools_check()`（:354）、`pf.tool_state()`（:371）**：工具的 SHA 紀錄。
- **p02 的 key 與 dump：**
  - `rf.load_val_keys(ddir)`（`p02_rank_fuse.py`:555）回傳 `{question_id, slot, color, parsed, path, sha256, n, groups}`。
    - `slot` 是 int 的 list；檔案裡存的是 int32。
    - `groups = [(qid, [row...])]`，要求同一題連續、slot 從 1..k 依序（`val_groups` :540）。
  - `rf.load_dump(path, n, 8)`（:405）：`logits` float32 `[n,8,10]`、`weights_sha256`、`syms`。
  - `rf.check_val_logits(d, parsed)`（:572）：已解析的列要是有限值，未解析的列要全是 NaN。
- **p02 的計算函式：**
  - `p3.combo_logprobs(("C5",), {"p02e9_d4": logits})`（`p03_rank_fuse.py`:464）和 `rf.source_logprobs("C5", …)`（p02:195）逐元素相同（實測）。
  - `rf.pool_matrix(groups, n_cols)`（:209）；`rf.pooled(S, lp, "log")`（:220）就是 `S @ lp`。
  - `rf.log_softmax`（:169）、`rf.softmax`（:175）、`rf.rule_b_np`（:179）、`rf.PROB_COLUMNS`（:102）、`rf.read_question_meta`（:1065）。
- **p03：**
  - `p3.load_swap(path, n)`（:1025）接受 float32 logits `[n,10]` 或 `[n,1,10]`；有 `syms` 欄位時必須是 1。prereg03 的 swap dump 成員是 `logits`、`weights_sha256`、`view="s0_colour_swapped"`。
  - `p3.kappa_questions(lp, groups)`（:541）、`p3.val_colour_subquestions(vk)`（:732）、`p3.question_boot_mean`（:744）。
- **decide.py**：`decide.read_probs`（:113）、`decide.rule_b`（:101）、`decide.run(in_path, out_dir)`（:242，寫出 pred_A、pred_B、decide_summary）、`decide.FALLBACK = "D"`（:68）。
- **`tools/t02_rank.py`**（在指紋裡，只 import、不改）：
  - `seam_compare`（:780）是通用的：用 `decide.read_probs` 讀兩份、比題號順序、算 max|dP|，規則 B 最多差 `SEAM_MAX_B_DIFF = 4` 題（:123），並回報 `byte_identical_*`。
  - `load_opening_test`（:680）、`t1_status`（:450）、`check_test_files`（:406）、`check_set_inputs`（:434）、`question_files`（:429）。
  - `dump_marker`、`check_dump_env`、`seam_tol`（:816–855）；`write_probs`（:759）。
  - `apply_set`（:696）不能整個拿來用，因為它套的是 R20 的 `question_logits`。但其中「C5 載入」（:698–704）、「題目檔對齊」（:706–717）、「開局對齊」（:725–738）三段，就是 t04 要照抄的寫法。

**寫死的、t04 不能用的：**
- **`load_o4_val(c, vk)`（:682）**：
  - 路徑固定是 `c.opp_dir/val_o4.npz` 加上 `val_report.json`；經 `opp_report` 要求 status ok、不是預覽、有 tables_sha256。
  - `replay_failed` 只要有一個是 True 就回報 problem，apply 會停下（:1764–1765）。
  - 遮罩只看 `vk["parsed"]`。
  - 測試集允許 ≤ 0.5% 失敗，所以不能直接用。
- **`val_c5(c)`（:950）**：它呼叫 `p3.load_val_side(c, …)`（p03:939），從 `c.p02_dumps` 讀。用一個假 ctx 指到 test02 各組的 dumps 也行，但照 t02 `apply_set` 的四行寫法（`rf.load_dump`、`check_val_logits`、遮罩、`source_logprobs`）比較直接。
- **`val_arm(c, x, vk)`（:955）**：
  - 檔名固定是 `val_p03{x}_d4.npz`（要有 8 個對稱）和 `val_p03{x}_s0_swap.npz`，還會和 `c.p02_dumps/val_keys.npz` 比對。
  - test04 notebook 如果只算對稱 0 的正確執色，這個函式就讀不進 A0／B0。
  - 建議正確執色的 logits 存成和 swap 檔相同的格式，這樣可以用 `p3.load_swap` 讀。
- **`val_question_inputs(c, vk, build_id)`（:970）**：它呼叫 `p3.opening_val_aligned`（p03:961），固定讀 `c.p02_local/opening_val.npz`，而且 build_id 必須等於凍結的 `c0cb86b4…`。test02 的 `opening_test.npz` 有自己的 build_id（t02 :623–628），所以測試題不能用這個函式。
- **`cmd_apply`（:1699）**：輸出只准寫 `results/prereg04/rank`；題目檔必須等於 val_keys。
- **`resolve`（:400）**：`READONLY` 清單（:139–141）包括 `results/test02`、`results/prereg04/opp`、`data`、`docs`、`colab`、`tools`。

## 2. seam_val O4 逐位元組相同的做法（實測）

步驟和 `cmd_val`（:1318–1334）同一順序：

```python
cpath, crec = po.registered_counts(None, None)                  # counts f6c6debe…
counts = po.load_counts(cpath); tabs, info = po.build_tables(counts); del counts
assert po.tables_record(tabs)["tables_sha256"] == fc["source_sha256"]["O4"]["tables_sha256"]   # f380e0d7…
vk = rf.load_val_keys(<seam_val 的 dumps 資料夾>)                 # prereg02 或 test02 的 seam_val，陣列相同
slots = po.read_question_slots(<data/local/rank_val_questions.csv>)   # SHA 73e51bbf…（val_report questions_file）
po.slots_match_keys(slots, vk)
res = po.val_scores(slots, tabs, workers=W, tmp_root=<repo 外>)
arrays = {"question_id": np.array(vk["question_id"]),           # <U10
          "slot": np.array(vk["slot"], dtype=np.int64),         # val_keys.npz 裡是 int32，這裡要是 int64
          "color": np.array(vk["color"]),                       # <U1
          "o4": res["o4"], "t4": res["t4"],                     # float64 [19156,10]
          "n_moves": res["n_moves"],                            # int64 [19156,2,3]
          "anomalies": res["anomalies"], "moves_parsed": res["moves_parsed"],   # int64 [19156]
          "replay_failed": res["replay_failed"]}                # bool [19156]
po.save_npz(out, arrays)        # npz 的成員順序就是這個 dict 的順序
assert sha256(out) == "30d312c02c357f6761567099003885be5e08635b6e6c0b49d71baac137cd9cd4"
```

實測結果（`o4_recompute.json`）：
- **檔案相同**：重算檔的 SHA-256 等於已 commit 的 `val_o4.npz`，也等於 `val_report.json` 的 `output_sha256`。9 個成員的順序、dtype、shape、位元組都相同。
- **key 陣列的來源不影響結果**：三個 key 陣列改用題目檔自己的字串組（`[s[0] for s in slots]` 等），SHA 仍相同。t04 不一定要讀 val_keys 也能重現。
- **zip 標頭**：9 個成員都是 `date_time (1980,1,1,0,0,0)`、deflate（8）、`external_attr 25165824`。時間戳不是寫入時間，所以檔案位元組可以重現。p04_opp 的 docstring（:823–824）也這樣說。
- **每盤結果和批次無關**：第 5000–6999 列單獨重算（列號從 0 重編、`chunk=37`、3 個 worker），`o4`、`t4` 的位元組和 `n_moves` 全部相同。
  - 原因：`score_game` 逐盤獨立，`ll` 依列號放回；3 個階段的加總順序固定。
  - 所以把 seam_val、public、private 串成一次 `val_scores` 呼叫、再切開也可以，還能省下 spawn 的開銷。
- 重播失敗 0 盤；目標方 2,143,537 手、對手方 2,143,537 手，和 `val_report.json` 相同。

**「逐位元組」建議在修訂裡怎麼寫：**
- 照字面，以**檔案的 SHA-256**（`30d312c0…`）判定。逐陣列比對只當診斷紀錄。
- 理由：已經實測做得到；時間戳固定；位元組只依賴同一個 `.venv`（Python zipfile、zlib 1.3.1 的 deflate、numpy 2.5.3 的 .npy header）和同一台機器的 CPU（`np.log` 等 SIMD 路徑）。
- 修訂要寫明分流：
  - 檔案 SHA 不同，但 9 個陣列全同：代表環境變了（例如 numpy 或 zlib 升級）。處理是還原 `.venv` 再跑，不是放寬比對。
  - 陣列不同：停下，查程式。

## 3. 融合配方

### 3.1 已驗證：用已 commit 的輸入重算 `rank_val_probs.csv`

`fuse_recompute.py` 只用第 1.2 節列的函式，重現 `cmd_apply`（:1699–1813）的數值核心：
- C5：`rf.load_dump` → 遮罩 → `p3.combo_logprobs`。
- O4：`rf.log_softmax(np.where(parsed[:,None], o4, 0.0))`，對應 :1767。
- SW：`rf.log_softmax(np.where(parsed[:,None], sw, 0.0).astype(np.float64))`，對應 :965。
- OP：`p3.opening_val_aligned(c, vk, fc["opening_build_id"])` → `mop`，對應 :976。
- 題目層：`S = rf.pool_matrix(cnn, n)`、`m = rf.pooled(S, lp5, "log")`，`Es` 依 `fc["combo"]["terms"]` 的順序，再 `pf.gen_logits` → `rf.softmax`。
- 寫檔：`repr(float(x))`、`lineterminator="\n"`。

結果：
- CSV 的 SHA-256 是 `3c7ab658dc558ce8c6aefdd7b0747de1a60de575d151e9eee96ca6775f442e5c`，和 `results/prereg04/rank/rank_val_probs.csv` 相同。
- `max |dP| = 0`；規則 B 和 `pred_B.csv`（`27b6d225…`）差 0 題；1,589 題，每題都有可解析的棋。

### 3.2 套到每一組題目（seam_val、public、private）

1. **凍結參數**：`fc` 是 `results/prereg04/rank/frozen_config.json`（`ae728e9c…`）。
   - 照 `cmd_apply` :1712–1734 檢查：`format`；smoke、preview、design_input 都是 false；`basis == pf.B`（:1718）；`combo` 是 Q02 而且 `terms == ["OP","O4","SWA","SWB"]`（:1727）；`B @ beta ≈ b`（atol 1e-12）；`theta` 從第 7 個起等於 `beta`（:1731–1732）。
   - `theta` 共 15 個數：`[a, γ, w_OP, w_O4, w_SWA, w_SWB, β1..β9]`，前幾個是 `[-0.00286820…, 0.44924503…, 0.36530334…, 1.18796173…, 1.39118944…, 0.96440332…, …]`。
2. **key**：`vk = rf.load_val_keys(results/test02/rank/<set>/dumps)`。
   - 題目檔對齊照 `cmd_apply` :1775–1787 或 t02 `apply_set` :706–717。
   - `rf.read_question_meta` 讀到的題號、順序、盤數、每盤執色，都要等於 vk。
3. **C5**：
   - 讀 `rf.load_dump(<dumps>/val_p02e9_d4.npz, n, 8)`。
   - `weights_sha256` 要等於 `fc["source_sha256"]["C5"]["weights_sha256"]`（`423581b4…`），也就是 prereg02 frozen_config 裡 `dumps.p02e9_d4` 記的值。
   - `rf.check_val_logits`。
   - `logits = np.where(parsed[:,None,None], d["logits"], 0.0).astype(np.float32)`。
   - `lp5 = p3.combo_logprobs(("C5",), {"p02e9_d4": logits})`。
4. **題目層**：
   - `cnn = [[i for i in rows if parsed[i]] for _, rows in groups]`
   - `S = rf.pool_matrix(cnn, n_rows)`
   - `n = len(cnn[q])`（float64）
   - `m = rf.pooled(S, lp5, "log")`
   - `ok = n > 0`
5. **OP**：
   - 讀 test02 的 `results/test02/rank/opening_test.npz`。`t02_rank.load_opening_test(path, qfiles)` 會核對 format、`seam_bitwise_equal`、build id、三組題目檔的 SHA。
   - 取 `set == <set>` 的列，依 `(question_id, slot)` 對齊，執色也要相同（t02 :725–737）。得到 `o`（float64 `[n,10]`）和 `has`（bool）。
   - `mop = rf.pool_matrix([[i for i in rows if has[i]] for _, rows in groups], n_rows) @ o`。
   - seam_val 那一段在 t02 已驗證和 `opening_val.npz` 逐位元相同，所以 mop 也相同。
6. **O4**：用 t04 opp 算出的該組 `o4`，列順序和 vk 相同。
   - 遮罩見第 6 節決定 1；0 盤失敗時遮罩就是 `parsed`。
   - `lpO4 = rf.log_softmax(np.where(mask[:,None], o4, 0.0))`
   - `E_O4 = rf.pooled(S_O4, lpO4, "log")`；遮罩等於 `parsed` 時 `S_O4 = S`。
7. **SWA／SWB**：用 test04 notebook 算出的該組對調 logits。
   - `p3.load_swap(path, n)` 讀入。權重 SHA 要等於 `pf.SW_WEIGHTS`：A 臂 `790b767c…`、B 臂 `8e88f6ec…`。
   - NaN 列要和 C5 的未解析列相同。
   - `lp = rf.log_softmax(np.where(parsed[:,None], sw, 0.0).astype(np.float64))`。先在 float32 上遮罩、再轉 float64，順序和 `val_arm` :965 相同。
   - `E = rf.pooled(S, lp, "log")`。
8. **合併**：
   - `Es = [E_OP, E_O4, E_SWA, E_SWB]`，順序照 `fc["combo"]["terms"]`。
   - `z, _ = pf.gen_logits(theta, m[ok], [E[ok] for E in Es], n[ok])`
   - `P[ok] = rf.softmax(z)`，其餘列是 NaN。
9. **寫檔**：
   - 表頭 `rf.PROB_COLUMNS`；每列 `[q, len(rows), int(n[k])] + [repr(float(x)) for x in P[k]]`，沒有可解析棋的題填 10 格空字串。
   - `csv.writer(lineterminator="\n")`、utf-8、先寫 tmp 再 rename。
   - 寫完用 `decide.read_probs` 嚴格讀一次；也可以直接用 `t02_rank.write_probs(path, rows)`，它兩件事都做。
10. **seam**：
    - seam_val 的 CSV 先跑 `decide.run` 產生 pred_B。
    - 再跑 `t02_rank.seam_compare(mine_csv, mine_B, results/prereg04/rank/rank_val_probs.csv, results/prereg04/rank/pred_B.csv, tol, smoke=False)`。
    - 通過的條件：題號和順序相同、`(n_games, n_parsed)` 相同、max|dP| ≤ tol、規則 B 最多差 4 題。
    - 如果 test02 的 seam_val C5 dump 和 test04 的 seam_val SW dump 都和原本的 dump 逐位元組相同，預期整份 CSV 也逐位元組相同（`3c7ab658…`）。這一條建議只記錄，照 t02 的 warning 寫法。
11. **測試集**：
    - public 的列在前、private 在後，寫成 `rank_test_probs.csv`。
    - `decide.run` 產生 pred_B；`submission_rank.csv` 是 `pred_B.csv` 的位元組複本。
    - `check_submission.check("rank", sub, [pub_q, priv_q], official=True)`，寫法見 t02 :985–1012。
    - n = 0 的題由 decide 填 `"D"`，和官方 notebook 一致。

**tol 要在修訂裡定**：
- 本機融合接縫同時用到 test02 的 C5（它的 GPU 要和 prereg02 dump 的 GPU 比）和 test04 的 SW（要和 prereg03 dump 的 GPU 比）。
- prereg02 和 prereg03 的 val dump 都是 `NVIDIA RTX PRO 6000 Blackwell Server Edition`、torch 2.11.0+cu128、cuDNN 91900（見各自的 `_DONE_*.json`）。
- 建議：兩個 seam_val dump 的 GPU 都和各自原本的 dump 相同，才用 1e-4；否則用 1e-3。

## 4. 測試日的執色檢查

### 4.1 定義與要用的函式

- **κ**（PREREG_03 2.2 第 4 條；實作在 `p3.kappa_questions`，p03:541–558）：
  - 題內每一盤 g 算 `Σ_k softmax(mean_{h≠g} l_h)_k · l_{g,k}`，再取題內平均；少於 2 盤記 NaN。
  - `l` 是每盤 10 維的 log_softmax。
- **Δκ**＝κ(目標方參照) − κ(對手方來源)，各題平均。三組配對是 `pf.PAIRS`：O4 對 T4、SWA 對 A0、SWB 對 B0（PREREG_04 2.2 第 2 條）。
- **官方題每題單色**（DATA.md「官方資料說明」、PREREG_03 rev 3 第 11 節）。
  - `pf.val_side_kappa(vk, lp_ref, lp_src)` 用 `p3.val_colour_subquestions`，依 `color_i` 把每題分成黑、白子題，≥ 2 盤的子題才算。
  - 官方題每題剛好一個子題，所以回傳的 `dkappa` 就是那一池的 Δκ（每題等權的平均）。
  - `by_colour`（黑題、白題）和 `question_bootstrap_ci95_recorded_only` 只記錄（PREREG_03 11(c) 最後一點）。
- **判定規則**（PREREG_04 5.2）：
  - public、private 分開算。O4、SWA、SWB 三個來源 × 兩池，共 6 個數。
  - 6 個全部 > 0 才用 04。
  - 任一個 ≤ 0、是 NaN，或沒有可以算的題，就不用 04，交 R20。
- **各來源的輸入**（遮罩後的每盤 log_softmax，和第 3.2 節相同）：
  - O4／T4：同一次 t04 opp 算出的 `o4`、`t4`。
  - SWA／SWB：test04 notebook 的對調 logits。
  - A0／B0：test04 notebook 的正確執色、對稱 0 logits。驗證時用的是 `val_p03{x}_d4[:, 0, :]`（:966）。
- **計算**：`kv = pf.val_side_kappa(vk_pool, lp_ref, lp_src)`，判定 `kv["dkappa"] > 0`。
  - O4 有失敗盤時，傳入 `dict(vk, parsed=mask_O4)`。`val_colour_subquestions` 只看 `vk["groups"]`、`vk["parsed"]`、`vk["color"]`。

### 4.2 驗證集上的數字（重算結果和報告逐位相同）

| 來源／參照 | 驗證單色子題 Δκ（2,986 個：黑 1,492、白 1,494） | 黑 | 白 | 出處 |
|---|---|---|---|---|
| O4／T4 | +0.1050265798301427（參照 −2.6211、來源 −2.7261） | +0.0898 | +0.1202 | `rank/apply_report.json` `val_colour_check.O4` |
| SWA／A0 | +0.03631175839656169 | +0.0440 | +0.0287 | `rank/check_report.json` `sources.SWA.colour.val_sub_questions`；`frozen_config.json` `single_colour.kappa_reference.SWA` |
| SWB／B0 | +0.034356617152967346 | +0.0376 | +0.0311 | 同上，SWB |

- inner 單色假題（120,544 題，SHA `45d0f14a…`，在 `frozen_config.json` 的 `single_colour.kappa_reference.*.inner`）：
  - O4：+0.0970 [+0.0759, +0.1177]；黑 +0.1023、白 +0.0918
  - SWA：+0.0274 [+0.0220, +0.0325]
  - SWB：+0.0249 [+0.0206, +0.0302]
- `fuse_recompute.json` 用兩種方法都重算出上表的 3 個值：`val_side_kappa`，以及直接用 `kappa_questions` 對子題求差。
- **建議 t04 在 seam_val 上也跑 `val_side_kappa` 並記錄**：
  - O4 應該完全等於 0.1050265798301427，因為 O4 逐位元組相同、遮罩也相同。
  - SWA、SWB 在 GPU 容許差內應接近 0.0363、0.0344。

### 4.3 每池 400 題的雜訊（只記錄；用驗證題內容，沒有答案）

- `kappa_noise.py`：驗證單色子題（2,986 個，平均 6.4 盤）每題 Δκ 的標準差，O4 0.80、SWA 0.26、SWB 0.25。
  - 換算成 400 題的標準誤：0.040、0.013、0.012；z 值 2.6、2.8、2.8。
  - 只看 ≥ 5 盤的子題（2,021 個）：z 值 3.0、3.3、2.9。
- 官方題 5–20 盤、平均 12.5 盤，雜訊應該更小。
- 粗估：每一個檢查「其實是正、因為抽樣才 ≤ 0」的機率約 0.1–0.5%，6 個合計不超過約 2%。
- 這只會造成「誤判不用 04、改交 R20」，不會讓錯的 04 上場。寫進修訂當已知限制即可，不改規則。

## 5. CPU 時間估計

| 步驟 | 實測或估計 |
|---|---|
| `import p04_opp` | 約 1.0 秒（實測） |
| `load_counts`＋`build_tables`＋`tables_record` | 1.7＋0.2＋0.1 秒（實測，檔案在 OS 快取時） |
| 讀題目檔＋`slots_match_keys`（19,156 slot） | 1.4 秒（實測） |
| seam_val `val_scores`（19,156 盤） | 6 個 worker 77.0 秒（實測，機器同時有其他工作）；登記那次 8 個 worker 全程 58.1 秒（`val_report.json` `seconds`） |
| 單一行程的單盤成本 | 約 0.0122 秒／盤（實測 200 盤） |
| public＋private | 400＋400 題 × 平均 12.5 盤＝10,000 盤。6 個 worker 約 40 秒（依 seam 的每盤牆鐘時間比例），8 個約 30 秒 |
| 三組串成一次 `val_scores`（29,156 盤） | 6 個 worker 約 2 分鐘，8 個約 1.5 分鐘 |
| 融合＋κ＋decide＋check_submission | < 5 秒（`fuse_recompute` 1,589 題 0.9 秒） |

- 記憶體：rate table 在主行程占 89 MB，tmp 裡的 `tables.npy` 另占 89 MB，由各 worker mmap 共用頁快取；counts 只在建表時暫用約 178 MB。
- 結果和 worker 數無關，所以 worker 數可以當參數。登記的 val 用 8 個；這台機器共用時照規則用 ≤ 6 個。

## 6. 風險，以及修訂要定的事

1. **決定 1：重播失敗盤怎麼處理。**
   - p04 在驗證集要求 0 盤失敗（rev 2 10.2 j）；測試集只規定「任一組 > 0.5% 就不用 04」（5.2），≤ 0.5% 時失敗盤怎麼用沒有寫。
   - 失敗盤的 `o4`、`t4` 仍是有限值：沒有任何一手時是 0 向量；有異常時是停下前那一段的部分和。所以一定要用 `replay_failed` 明確遮掉，不能靠 NaN。
   - 建議：O4 的遮罩＝`parsed & ~replay_failed`，E_O4 和 O4／T4 的 κ 都只用遮罩內的盤。這符合 2.1「每盤可解析的棋取平均，一盤都沒有就記 0」。
   - 溫度項的 n 仍用 C5 的解析數，不改。
   - seam_val 上是 0 盤失敗，兩種做法完全相同，不影響逐位元組比對。
   - 失敗比例每組分開算，分母是該組的 slot 數：seam_val 19,156、public 約 5,000、private 約 5,000。0.5% 大約是 25 盤。
2. **0 盤可解析的題。**
   - C5 的 n = 0：該列留白，decide 填 `"D"`，和 t02、官方 notebook 相同。
   - C5 有解析，但 O4 的遮罩是空的：E_O4 是 0 向量（`pool_matrix` 的空列）。這題不計入 O4 的 κ（少於 2 盤）。
3. **SGF 的怪情況**（MOVE_RE 是 `;([BW])\[([a-s]{2})\]`，`p02_moves.py`:44）：
   - **pass**：`B[]`、`B[tt]` 會被跳過（`tests/test_p04_opp.py`:273）。階段只數編碼到的手；距離特徵看上一個真正的落子。fit 計數時也是這樣處理，所以前後一致。
   - **設定子**（AB／AW，讓子棋）：會被忽略，重播的盤面少了這些子。可能漏掉提子，之後有一手下在「以為還有子」的點上，就變成異常、算重播失敗；也可能沒有失敗，但特徵是錯的。
   - **分支**：多個變化會被串成一條，也可能造成異常。
   - **小寫的 `b[..]`**：抓不到，會變成「沒有任何一手」。
   - **SZ ≠ 19**：座標仍在 a–s 內，但棋盤是錯的。
   - 這些都由 T1.4 把關：任一檔超過 1% 的盤有設定子、SZ ≠ 19、pass、不交替、第一手是 W 或沒有任何一手，就判 FAIL_OPENING（`t02_checks.py`:499–549、:137）。FAIL_OPENING 時 04 不上場（`fc["fail_opening"]`）。
   - 對照：fit 720,307 盤和驗證 19,156 盤都是 0 異常。
4. **執色正規化和語意。**
   - T1 NORMALIZED 的 `upper_colour` 只改 `data/test` 裡的正規化副本。t04 必須讀 `data/test`，也就是 notebook 用的同一份檔：核對 notebook `config.json` 的 `test_bundle` SHA、各組 `_INPUT.json` 的 SHA，以及 MANIFEST（`t02_rank.check_test_files`、`check_set_inputs`）。
   - 不讀 `data/test/raw`。raw 若是小寫，`read_question_slots` 會直接報錯，是安全的失敗方向。
   - `color_i` 的語意（目標玩家的執色）和訓練相同是前提；Δκ 檢查就是在守這一點。
   - 前提檢查可以用 `t02_rank.t1_status(ctx, "apply", fallback=False)`：只接受 PASS／NORMALIZED，而且正式模式要求完整的 raw T1。
5. **計數檔不在 git 裡。**
   - `data/prereg04/counts_fit.npz`（178 MB，gitignored）是唯一的輸入表。
   - 本機另有一份相同 SHA 的 `data/prereg04/design_scratch/counts_fit.npz`（`scratch_manifest.json` 記的 SHA 也一樣）。
   - 兩份都不見的話，正式的 `count` 會因為 `count_report.json` 已存在而拒絕再跑。只能在 repo 外用預覽重算（8 個 worker 約 34 分鐘），SHA 要等於 `f6c6debe…`。
6. **環境與機器。**
   - 逐位元組相同依賴同一個 `.venv` 和同一台機器，見第 2 節。t04 報告要記 Python、numpy、scipy、zlib 的版本。
7. **tmp 放哪裡。**
   - `score_games` 會在 `tmp_root` 建暫存資料夾，放 89 MB 的 `tables.npy`，跑完刪除。
   - 行程被中斷時會留下檔案，所以 `tmp_root` 不要放在 `results/` 底下（尤其不能在凍結的 `results/prereg04/**`）。用系統 TEMP 或 scratch。
8. **test04 notebook 輸出格式要配合的地方**（給負責 notebook 的人）：
   - 對調 logits 用 `p3.load_swap` 的格式：float32 `[n,10]`，成員有 `weights_sha256`，若有 `syms` 則為 1。
   - 正確執色的對稱 0 logits 建議用同一種格式。`val_arm` 要求 8 個對稱的 d4 檔，不適用。
   - 每組都要有 key 檔，而且要和 test02 同一組的 `val_keys.npz` 陣列相同。
   - NaN 列要和 C5 的未解析列相同。
9. **輸出位置和檔名**（建議）：
   - `results/test04/rank/opp/{seam_val,public,private}_o4.npz`，每個檔的成員和順序都和 `val_o4.npz` 相同，seam_val 那份要逐位元組等於 `30d312c0…`；另加一份 `opp_report.json`。
   - `opp_report.json` 要記：`frozen_config.json` 的 SHA（`ae728e9c…`）、`p04_opp.py` 的 SHA（`b594e0ef…`）、`tables_sha256`（`f380e0d7…`）、counts 的 SHA，每組題目檔的 SHA、slot 數、失敗數和比例、異常數、沒有任何一手的 slot 數、手數，每組輸出的 SHA，以及環境。
   - 這樣 t04 apply 可以照 `val_report_binding`（:1677）的方式，對這三個 SHA 做綁定檢查。
   - 用字：public、private 的 `question_id` dtype 會是 `<U10`、`<U11`，不影響判定。
10. **凍結的資料夾。**
    - t04 只寫 `results/test04/`。
    - import `tools/` 底下的模組只會在 `tools/__pycache__` 寫 bytecode（gitignored，無妨）。不要 import `results/prereg04/design/evidence/**` 的腳本；如果要，先設 `sys.dont_write_bytecode = True`，否則 hygiene 的 bytecode 檢查會失敗。
    - `tests/test_p04_hygiene.py` 的 `FrozenResults` 只涵蓋 prereg01–03 和 `prereg04/design`。建議 t04 的新測試把 `results/prereg04/opp`、`results/prereg04/rank` 也納入「沒有修改、沒有新檔」的檢查。
11. **演練可以加的強斷言**（`t02_fake_tests.py from-val`）：
    - 假的 public／private 就是驗證題 1–400 和 401–800，只改了題號前綴，SGF 位元組相同。
    - 所以 t04 opp 在假 public、private 上的 `o4`、`t4`，應該和 `val_o4.npz` 對應列逐位元相同。
    - 如果假的 SW dump 也是 prereg03 val dump 的切片，融合機率應該和 `rank_val_probs.csv` 第 1–800 題逐位相同。
    - 另外，`from-train --shape official` 的樹是單色、inner 玩家的題（沒有 C5 dump），可以只跑 O4／T4 的 Δκ，當作單色官方結構的 CPU 演練。預期接近 inner 單色的 +0.097。inner 玩家的棋不在 fit 計數裡，所以沒有洩漏。
