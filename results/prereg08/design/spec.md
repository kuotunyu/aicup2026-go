## spec notes（給工具建置代理的實作契約）

### A. 紀律（同 PREREG_07 spec 第 0 節，換成 08 的路徑）

- **data/local**：只有 `p08_p1.py val`（正式模式、凍結之後）和 `p08_rank_fuse.py apply` 准讀，而且只讀 `rank_val_questions.csv` 的 `question_id`、`num_games`、`color_i`、`sgf_i`。路徑含標籤檔字樣一律拒絕（沿用 `p07_rank_fuse.refuse_label_path`）。絕不跑 `tools/score.py`。
- **import 順序**：讀完題目檔之後才 import `p1_lib`／`p1_extract`／`hs_lib`／`pilot_*`（`hs_lib` 會裝 data/local 防護）。`u1_lib` 只給 `check`、`selfcheck`、`tune` 延遲 import；`apply`、`p08_p1.py` 不得 import 它，也不得呼叫 `pilot_fuse.fit_mapping` 或 `p1_fuse.fit_mapping_grouped`。
- **暫存**只寫 `data/prereg08/tmp/`，不寫 `%TEMP%`（u1_lib 會拒絕 `AppData\Local`）。
- import 前先設 `sys.dont_write_bytecode=True` 和 `PYTHONDONTWRITEBYTECODE=1`；跑完 `results/prereg08/**`、`results/prereg07/**`、`results/prereg05/explore/u1`、`third_party/katago` 不能有 `__pycache__`。
- **模式**：real（全部已 commit 且乾淨）、`--precommit`（只第一次建置，記 SHA 取代 commit 要求，照 p07）、`--out-dir data/prereg08/tmp/<x>`（preview，任何後續步驟都不接受）。
- **輸出已存在**時拒絕；要重跑加 `--rerun --reason`。`tune` 的 rerun 要 `--revision N ≥ 2`，而且那個修訂的 commit 晚於舊的 tune。
- **結束碼**：0 完成；1 這一步說停（閘門沒過、status stop）；2 拒絕或錯誤。

### B. 釘住的值（工具內常數，import 前核對）

- **程式檔**（SHA-256，CRLF 折成 LF）：
  - `hs_lib.py` `d19b6a8615ef757891ff2d451fe5b96b19f31cc706a87862e49a7688d0bde97c`
  - `pilot_run.py` `b0d127fd1d5a83af5b17ec3df1dccc85d94b11c3479d33cac07448a3796a3936`
  - `pilot_fuse.py` `37fb2dff0f37032bf08acc7f27213cafd30533ab024973f92b5b28f41c6ac0c6`
  - `u1_lib.py` `225dd51cff961ae3608e003432388e0b6d6a053fb4dac26add5425c002dc3ff7`
  - `p1_lib.py` `062600168326f9eecd4ec1073e54881c475f8f276e376164eab20d373bbff511`
  - `p1_extract.py` `2ee8eb127150e672b62bcb30b70c4fac46b8cd6e32c7e4eb19d54f746a6e3f9c`
  - `p1_fuse.py` `ad356c8ca9f26e1e761677afa07b5aa369ca31e5318f61eb6a6cbda33b970953`
  - `p1_plan.json` `0cd46bf90c764cae3824f5906e72e13336ab7276ef5fa7edb2ec6831911b228d`
  - `p2_sets.py` `98bad3d85a32e80c065226b0cc8cd1065b423cdb39dce89829a54b0cbf381c8f`
  - `tools/p07_rank_fuse.py` `8e144e2fe14001af6592130611c1bfce3f49c7bae8208137d2bb3f38ed5b7142`、`tools/p07_hs.py` `b27ea371bdc1477558effb9ca86c540805dcced971d983226e15fb5e19b7b8dc`（08 會 import p07，所以也釘）。
- **資料（raw）**：
  - P1 `data/prereg08/p1/sets.json` `f0c494962788615d76080b5481de5b762d71f3ad37f8add55a09efef82acb7c7`
  - P2 `data/prereg08/p2/sets.json` `2b68f7ae27b2b6bbe00a3920065495d2a5d6b73fd7df33854f4bf99ae5390049`
  - 07 試跑 `plan.json` `8494e9afd79c06249bd81545263b86caef7b9e77f3cd631a1e354368cfe33946`
  - 權重 `84b23476…`（323,752,318 bytes）
  - `SOURCE.txt` `039d5a1c…`
  - `val_keys.npz` `055d2941…`；題目檔 `73e51bbf…`（同 p07 `QUESTIONS_SHA`）
- **07 的凍結檔**：
  - `results/prereg07/rank/frozen_config.json` `93a0ac29615260865d554ee61aec258c633820a87aeb7dfcf8047a5dfd200c0b`
  - `hs_map.npz`（raw）`a4334e79c490e4d8102dd5136ac08c0de1060315981f4ccc527ebffa1b84a3e4`
  - `rank_val_probs.csv` `6ae2db6025a903eb5c58af9418dc203c0b6e84dd664049f85dfa850e57dc28c1`
  - `pred_B.csv` `51d7dbf82d50c3848acadb6934069580bd3ade7e6ca71ad5d1023633a18a26fa`
  - `results/prereg07/hs/val_hs.npz`（raw）`ef2067e8cb633f521c205a09aa4638c79fef3ba0b4e90e6a54bdc046e5a5550f`
  - `results/prereg04/opp/val_o4.npz` `30d312c0…`
  - 04 的 `rank_val_probs.csv` `3c7ab658…`
- **其他常數**：
  - 單色題 SHA `45d0f14acd98786adfd7cae040edc551420f095aaf6118d52edb47ddf91a5622`，120,544 題。
  - SCALE＝4.0、GATE＝0.010、TOL＝1e-12。
  - VAL_FAIL_MAX＝95、P2_FAIL_MAX＝600、P2_CUTOFF＝2026-10-08T23:59:59+08:00。
  - RHO_FLOOR_VAL＝0.078、RHO_FLOOR_TEST＝0.155。

### C. `tools/p08_p1.py`（GPU 抽取，重用 p1_lib／p1_extract，不改它們）

- **`PY tools/p08_p1.py manifest [--precommit] [--rerun --reason T]`** → `results/prereg08/feat/inner_manifest.json`
  - 拒絕：P2 沒到 600／600 而且現在早於 P2_CUTOFF。
  - 對 P1 的 probe（150）、bank（60）、mono（307）和 P2 的 probe（600）塊（用 `p1_fuse.load_p1_set` 的格式檢查），記每塊的 SHA-256、盤數、失敗數、沒有目標手數、`ref` 和 sets.json 的那一段相同（`p1_extract.set_games`）、meta（precision、device、batch、channels_last、`meta_sha256`、`katago_commit`、`code_sha256`）。
  - 權重完整 SHA；`third_party/katago` 對 SOURCE.txt。
  - **`p2`**：`{complete, chunks, games, failed, used, reason}`，其中 `used = complete and failed ≤ 600`。
  - **disjoint**：probe 玩家和 inner、驗證玩家（`players_split.csv`）不相交；P2 和 P1 probe 的 (rank, game_id) 不重疊。
  - **duplicates**：probe（P1，P2 用時加 P2）和 bank／mono 的 SGF 落子序列（原方向）雜湊比對，列出重複的 probe ref（tune 會排除）。
  - **move_cross_check**：bank／mono 每盤 P1 的 `n_moves`、`counts[:,0]` 對 07 試跑 HS 塊的總手數與目標手數、對 `inner_o4.npz` 的 `n_moves` 合計與 `n_moves[:,0,:]` 合計；不符要 0。
  - 只讀 metadata、計數和 SHA，不算任何分數。
- **`GPY tools/p08_p1.py val`**（正式）→ `data/prereg08/val/chunk_<i>.npz`，最後寫 `results/prereg08/feat/{val_p1.npz, val_seam_feats.npz, val_report.json}`
  - **前提**（打開 data/local 之前全部核對）：
    - `results/prereg08/rank/frozen_config.json`（`score_pending_val=true`、非 preview）和 `p1_map.npz` 已 commit 且乾淨、SHA 對；
    - 設計紀錄（manifest、check、selfcheck、tune_report）已 commit 且等於 frozen 記的 SHA；
    - 工具等於 tune 記的；還沒有 `val_report.json`；
    - `nvidia-smi` 沒有 Type C 的程序；`.venv-gpu` 的 python、torch、cuDNN、GPU 等於 frozen 記的參照（RTX 4090）。
  - **順序**：讀題目檔（`p04_opp.read_question_slots`）和 `val_keys`（`p02_rank_fuse.load_val_keys`），`slots_match_keys`（1,589／19,156）→ 算驗證盤的落子序列雜湊，和 probe 比（只記錄）→ 這時才 import `p1_lib`、`p1_extract` → 用 `p1_extract.run_chunk(pool, ProbeScorer(fp16, cuda, 1024, channels_last), games, 1024*group_batches)` 每 200 盤一塊；目標色＝題目的執色。
  - **塊的欄位**：`run_chunk` 的全部欄位，加上 `format "c10-prereg08 val chunk v1"`、`ref`、`keys_sha256`、`questions_sha256`、`ckpt_sha256`（完整）、`torch`、`cudnn`、`gpu_name`、`driver`、`meta_sha256`、`tool_sha256`、`mode`、`created_utc`、`seconds`。
  - 原子寫入（tmp＋`os.replace`）；續跑時通過檢查的塊跳過；欄位或環境不同的塊拒絕、不覆寫。
  - STOP 檔 `data/prereg08/val/STOP` 或 `--stop-after-min M`：手上那一塊寫完後停。
  - **組裝**：
    - `val_p1.npz`：question_id、slot、color、counts[G,2]、n_moves、failed、`scal[G,2,2]`（logp、熵的平均）、ref；
    - `val_seam_feats.npz`：前 400 盤的 `means[400,2,770]` float32、counts、n_moves、failed、兩塊的 SHA；
    - `val_report.json`：每塊 SHA、`first_chunk_started_utc`、`frozen_config_sha256`、環境、速度、失敗數、`status ok|stop`。
  - 失敗盤 > 0 時結束碼 1。
- **smoke**：`val --smoke --questions Q.csv --out-dir data/prereg08/tmp/<x> [--max-games N] [--fake-scorer] [--workers W] [--ignore-other-gpu-jobs]`。題目檔在 data/local 就拒絕；`--fake-scorer` 是確定性的 CPU 假網路。
- **`PY tools/p08_p1.py status`**：塊數、STOP、速度、剩餘時間估計。唯讀。

### D. `tools/p08_rank_fuse.py`（CPU）

- **`check [--precommit|--out-dir]`** → `results/prereg08/rank/check_report.json`：登記第 2.2 節第 1–7 條；記 manifest 的 SHA 和 commit。
- **`selfcheck`** → `selfcheck.json`，全部 ≤ 1e-12：
  - Q02 的 OOF（eval、mono）和 `u1_oof.npz` 的 `eval_P_Q02`、`mono_P_Q02` 逐位元相同；Q02H rb 0.6242752597994481。
  - 只用 P1 probe 的 TO 翻譯模型（`p1_fuse.fit_mapping_grouped`，`p1_plan` 的格點，maxiter 2000）：λ 0.01、`cv["0.01"].logloss` 1.6790734173594255、`refit_loss` 1.585088334325938。
  - inner 的 lp 照 p1_fuse 的呼叫方式：bank、mono 各呼叫一次 `mp["logp"]`。
  - Q02H＋P1TO ×1：rb 0.675034627957959、diff 0.0507593681585109、ci95 [0.03701489044315354, 0.06537945687999477]、mono diff 0.04964601127852308、B 0.055909493750024186、W 0.04309679963025359。
  - ×4：rb 0.6767709161878595、diff 0.05249565638841136、ci95 [0.0386137095020332, 0.06756280199569697]、mono diff 0.05075085423026038。
  - 遮罩版 E_HS／E_P1 和 `pooled` 版逐位元相同。
- **`tune [--precommit|--out-dir] [--rerun --reason T --revision N]`** → `tune_report.json`；閘門過才另寫 `frozen_config.json`、`p1_map.npz`：
  1. probe 集合＝P1 probe（manifest 的 `p2.used` 為真時加 P2 probe），扣掉 manifest 列出的重複盤和不可用盤（failed 或 `counts[:,0]==0`）。y＝等級；玩家鍵＝`rank:player_id`（兩份 sets.json 的玩家鍵在聯集上分折）。
  2. `fit_mapping_grouped(PF, U, X_TO, y, players, plan λ 格點, maxiter=2000)` → 存成 `p1_map.npz`（format、mu、sd、W、b、lambda、cv JSON、probe_used、games、players、兩份 sets SHA、refit_loss），再從檔案讀回，核對 lp 和記憶體裡的逐位元相同（bank、mono 分開呼叫）。
  3. inner 的 lp_P1TO（bank、mono），乘 SCALE；組 E_HS、E_P1 的遮罩版；仿 `p07_rank_fuse.run_terms` 寫 run_terms，HS 和 P1TO 兩項都取遮罩版 E。
  4. OOF：Q02、Q02H、Q08。只記錄的：Q08 ×1、Q02＋P1TO ×4、R20＋P1TO ×4，以及（用 P2 時）只用 P1 probe 翻譯模型的 Q08。
  5. 閘門：G1、G2 用 `inner["boot"]`；G4 用 `p3.mono_diff_record`，要全部 120,544 題、黑白都 > 0。另記每題平均差、`se_val_record`（`p3.se_val_record`＋`p4.adoption`）。
  6. 方向參照 ρ_eval、ρ_mono：只用 softmax(E_P1) 的期望等級對 Q02 OOF 的期望等級（同 p07 `hs_rho`）。
  7. 閘門過了才 `p4.gen_fit_record` 重擬合，寫 frozen（內容見登記第 2.6 節，另記 `score_pending_val=true`），工具的 SHA 一起記。
- **`apply [--accept-data-failures TEXT] [--rerun --reason T]`** → `rank_val_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`check_submission.txt`、`val_lp_p1.npz`、`apply_report.json`：
  - 前提：照 p07 的 `apply_prerequisites`（換成 08 的檔），加上 `val_report.json` 的開始時間晚於 frozen 的 commit、每塊 SHA 對。
  - 順序：
    1. `p07_rank_fuse.q02_validation()`，接縫要過；
    2. `p07_rank_fuse.check_question_file`；
    3. import pilot_fuse；
    4. `p07_rank_fuse.apply_core(PF, q, val_hs, hs_map, fc07)`，它的 `csv_bytes` 要等於 `6ae2db60…`；
    5. 讀 val 塊（SHA 核對）→ X_TO → `p1_map` 的 lp；
    6. 遮罩 E_P1 ×4 → 用 frozen 的 theta 算 `fuse_probs` → csv；
    7. 子行程跑 `decide.py`、`check_submission.py`（帶 `--questions data/local/rank_val_questions.csv`，相對路徑）。
  - 手數交叉核對（對 `val_hs.npz` 和 `val_o4.npz`）、方向 ρ、失敗數；照 p07 的 `apply_decision` 決定 `score_allowed`。apply 一律不寫 `true`；稽核還沒 commit 時是 pending。
- **`apply --finalize`**：
  - `prescore_audit.json`（format `c10-prereg08 prescore_audit v1`，欄位 `program_errors_affecting_Q08: []`）已 commit、晚於 `apply_report.json`。
  - 它的 `outputs_sha256`（四個輸出檔）和 `inputs_sha256`（`val_p1.npz`、`p1_map.npz`、`val_hs.npz`）綁住 apply 的值，才寫 `score_allowed`。
- **不在工具裡**：`score_H8R.json`、`RESULTS.md`（主 session 照登記第 3 節）。

### E. 測試（全部只用 CPU、合成資料，絕不碰 data/local；不在 pytest 行程裡 import hs_lib／u1_lib／pilot_*／p1_lib，要用的路徑一律以子行程跑；每個模組加 autouse fixture，`builtins` 不得留 `_c10_hs_guard`、`_c10_u1_guard`；`tests/conftest.py` 不改）

- **`tests/test_p08_p1.py`**：
  - 用合成 SGF 做的題目檔（放 `data/prereg08/tmp/`），以子行程跑 `val --smoke --fake-scorer`：產生的塊、組裝出的三個檔的形狀和欄位對、續跑會跳過、STOP 會停、欄位不同的舊塊會被拒絕。
  - smoke 給 data/local 路徑會被拒絕；real 模式在 frozen 沒 commit 時會被拒絕，而且是在打開任何 data/local 之前（子行程裡追蹤 open）。
  - 題目檔是在 `p1_lib` 進 `sys.modules` 之前讀的。
  - `manifest` 的 P2 規則（沒齊而且早於截止 → 拒絕；齊了 → used；失敗盤 > 600 → 不用）是純函式，可以單測。
- **`tests/test_p08_rank_fuse.py`**：
  - 沒有不可用盤時，遮罩 E 等於 pooled；SCALE 只乘一次。
  - 合成資料上，`fuse_probs` 的 6 項等於 p04 的 gen 公式。
  - probe 聯集、玩家分折、重複盤排除是確定性的。
  - `gate_record` 的邊界：+0.010−1e-12 過、下界 0 不過、黑或白 ≤ 0 不過、題目不全不過。
  - `apply_decision` 的矩陣；稽核的綁定（未 commit、SHA 不符、commit 時間早）；`--finalize` 只在稽核之後；tune 重跑要修訂；preview 不被後續步驟接受；標籤檔路徑會被拒絕。
- **`tests/test_p08_hygiene.py`**（只用標準函式庫）：
  - 釘住檔的 SHA（B 節）。
  - `results/prereg08/p1/**` 從 `0ca8316` 起沒有改變也沒有新檔；`results/prereg08/p2/**` 從 `a709408` 起相同。
  - `docs/PREREG_08_STRENGTH_PROBE.md` 只准從 rev 1 的 commit 起加行。
  - 帳上 H7R 之後只有 0 或 2 列 `prereg08 H8R`。
  - `score_H8R.json` commit 之後，`results/prereg08/rank`、`results/prereg08/feat` 已 commit 的檔在工作樹裡不變。
  - `results/prereg08/**` 沒有 `__pycache__`。
  - `results/prereg07/**` 照 `test_p07_hygiene.py`（不重複）。
- **既有測試的最小修改**（和工具同一個 commit，H8R 之前）：
  - `tests/test_p06_player_hygiene.py`：讓 07 的列之後可以有 `prereg08 H8R` 的列。
  - `tests/test_p07_hygiene.py::test_prereg07_rows`：H6P 之後的列＝07 的 2 或 4 列，再接 0 或 2 列 `prereg08 H8R`；不再要求每一列都以 `prereg07` 開頭。
  - 其他行不動，審查時逐行比對 diff。
- **閘門**：CLAUDE.md 的指令加 `tests/test_p08_hygiene.py`（十個檔），允許的失敗不變（`test_t04_hygiene` 的帳列）。
