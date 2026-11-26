# PREREG_03 實作：建置、整合、審查紀錄（2026-09-27）

兩個代理寫程式（I1：notebook 產生器與兩份 notebook；I2：本機工具 `tools/p03_rank_fuse.py`）→ 一個代理整合（本節）。之後的審查和修正接在後面的節。

## 1. 整合報告（INTEGRATOR）

### 結論

- 兩份 notebook 能重建，兩次輸出逐位元組相同，也和工作目錄裡的檔相同。
- 全新的 CPU 冒煙跑完整條鏈：prereg01 → prereg02 → prereg03 A、B 臂 → 本機工具 unpack／check／selfcheck／tune／apply → `decide.py` → `check_submission.py`。
- 真實資料的 selfcheck 再跑一次，三項重現都是差 0.0。
- 只找到一個接縫問題（見 1.2），修在 `tools/p03_rank_fuse.py`，不影響正式模式。
- 沒有 commit，沒有改任何凍結檔，沒讀答案檔，沒跑 `score.py`，沒用 GPU。

以下 `S=C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03build\INT`，`P=.venv/Scripts/python.exe`（CPU 版 torch 2.14.0、Python 3.12.13），都從 repo 根目錄執行。

### 1.1 執行的指令與結果

**測試**

- 第一次：`P -m pytest -q tests` → 438 passed、22 skipped，423.5 秒。被略過的是要設環境變數的慢測試。
- 修正後重跑：見本節最後「最終測試」。
- `python -m pyflakes`（pyflakes 3.0.1）檢查 4 個 p03 檔：沒有警告。

**重建 notebook**

- `P colab/build_prereg03_rank_nb.py --out-dir S/build1`，以及 `P colab/build_prereg03_rank_nb.py D:\AI-Portfolio\competitions\aicup2026-go --out-dir S/build2`，各 0.5 秒。
- 4 個輸出兩次逐位元組相同，也和工作目錄的檔相同：

| 檔案 | sha256 |
|---|---|
| `colab/prereg03_rank_a.ipynb` | `d020c4bd2fbb79b33f1a3ff934fd8812a9bbdf0bf35cd211a09ae4044c84fdc4` |
| `colab/prereg03_rank_b.ipynb` | `6a9c5c984d8f50bfecb9f052044297f3c3c35297d9cd6695dabae3ae79067865` |
| `results/prereg03/design/nb_diff_a.txt` | `3fa7945c2afd4350d29c93c10c9d393660756c7f8047142e07ae28b051b89aa8` |
| `results/prereg03/design/nb_diff_b.txt` | `f6deb92db207ab65c5f4bf227bc2bd7025bb1a43194e47952173bb392e70c755` |

**全新冒煙**（`S/chain.sh`，同一個 scratch `S/smoke`，只有第一步加 `--fresh`；前兩步和 pytest 同時跑，所以比平常慢）

| 步驟 | 指令 | 結果 |
|---|---|---|
| prereg01 | `P colab/run_local_smoke.py colab/prereg01_rank.ipynb --scratch S/smoke --fresh` | OK，548 秒 |
| prereg02 | 同上，`prereg02_rank.ipynb`，不加 `--fresh` | OK，1,012 秒 |
| prereg03 A | 同上，`prereg03_rank_a.ipynb` | OK，813 秒 |
| prereg03 B | 同上，`prereg03_rank_b.ipynb` | OK，809 秒 |

- **A 臂**：`device cpu | GPU None | ARM A | workers 0 | … | T 70 | AMP off`。
  - 各段秒數：load_rows 31、分片 key 核對 5、第 1 輪 401、第 2 輪 73、最後輸出 275。
  - 第 1 輪 63 步、2,000 盤（執黑 960，等於預期）；第 2 輪 7 步，停在第 T＝70 步。
  - 學習率最後是 0.0；`training_complete` true。
- **B 臂**：冒煙 E_L 各級 112–159，每級目標 112。
  - 第 1 輪收下 1,144 盤、36 步（和預期相同）；第 2 輪收下 1,108 盤，讀到 1,088 盤時到了第 70 步。
  - 兩臂初始權重 SHA 相同（`ec405e63…`）。
- **兩臂共同**：
  - key 檔和 `prereg02_smoke` 的 key 檔相同（inner 400 列、val 142 列）。
  - 唯讀目錄前後相同；`run_log` 狀態是 completed。
  - zip 各有 20 個檔，都在 `rank_{a,b}/` 底下，沒有權重；大小 211,724／211,361 bytes。
- **CPU 上可重現**：`final.pth` 的 SHA 是 A `8a026f9e…`、B `177285…`，和 I1 另一次冒煙完全相同。兩臂 inner dump 的位元組也和 I1 那次相同。這證明資料順序、D4 抽法、各處種子都照規格固定。

**prereg02 本機工具**（`S/tools_chain.sh p02`，輸出 `S/p02local`）

| 指令 | 結果 | 秒 |
|---|---|---|
| `P tools/p02_rank_opening.py build --smoke --out-dir S/p02local` | OK | 70.8 |
| `P tools/p02_rank_fuse.py repro --smoke --p01-dir …/prereg01_smoke/rank --p02-dir …/prereg02_smoke/rank --out-dir S/p02local` | PASSED：12 題，差 1.63e-8，規則 B 差 0 題 | 15.0 |
| `P tools/p02_rank_pseudo.py --smoke … --min-games 2 --fit-min-games 3` | OK | 25.5 |
| `P tools/p02_rank_fuse.py tune --smoke …` | 選中 R18；冒煙的 epoch 9 低於 24.15%，所以 C4、C5 被拿掉，沒有 R20 | 4.8 |
| `… apply --smoke …`、`P tools/decide.py …` | OK | 1.1、0.2 |

**R20 對照組**（只在 scratch）

- 上一步的 p02 冒煙沒有 R20，p03 的 selfcheck 在冒煙時就只能記「n/a」。
- 所以另寫 `S/p02_tune_force_r20.py`：import 凍結的 `p02_rank_fuse`，只在這個程序裡把地板改成 0、選法固定成 R20，再用 p02 自己的 tune／apply 產生 R20 的 `tune_report`、`frozen_config`、`rank_val_probs.csv`、`pred_B.csv`（輸出 `S/p02local_r20`）。
- 結果：tune 12.0 秒（R20 OOF 0.3406），apply 1.2 秒。p02 檔案沒動。

**p03 本機工具**（`S/tools_chain.sh p03`）

- unpack：
  - `P tools/p03_rank_fuse.py unpack --arm a --smoke --dest S/p03dir/rank_a …/prereg03_rank_a_results.zip`：20 個檔，1.0 秒；B 同理，0.9 秒。
- 照登記門檻跑（`--min-acc 0`，不加新旗標）：
  - `check`：exit 1。C6A 停在執色接線：對調 9.5% 比正確 8.5% 高 1.0 個百分點。冒煙 inner 只有 400 盤，1 個百分點就是 4 盤，這是雜訊。
  - `selfcheck`：exit 1，因為 check 沒過。
  - 這兩步證明停止條件在真的 notebook 輸出上有作用，也暴露了 1.2 的接縫問題。
- 主鏈（`--min-acc 0 --smoke-record-colour`，R20 對照組 `S/p02local_r20`，輸出 `S/p03local`）：
  - `check`：OK，1.2 秒，能用的來源 C5、C6A、C6B；C6A 的接線停止照記在 `stops_recorded_only_smoke`。
  - `selfcheck`：PASSED，1.4 秒。四項都有實際比較：逐盤 log 機率逐元素相同；R20 的 OOF 紀錄和各折差 0.0；重擬合參數差 0.0（refit loss 1.965791360783639 兩邊相同）；12 題機率差 0.0，規則 B 和 `pred_B.csv` 都沒有不同。
  - `tune`：3.3 秒，P00–P06 全跑。選中 P01（0.3466；P00 0.3406）；G1 差 +0.0060 < 0.010，閘門沒過，`score_allowed` false。SE_val 0.0055；FAIL_OPENING 用 R10。
  - `apply`：15.3 秒，第一次讀題目檔比較慢。
  - `P tools/decide.py S/p03local/rank_val_probs.csv S/p03local`：OK，12 題，A≠B 2 題。
  - `P tools/check_submission.py rank S/p03local/pred_B.csv --questions S/smoke_rank_questions.csv`（題目檔前 12 題）：OK。
- 登記地板（24.15%）、用 p02 原本的冒煙鏈當對照（輸出 `S/p03local_floor`）：
  - `check`：兩臂 unusable，只剩 P00。
  - `selfcheck`：PASSED，三項比較在冒煙時是 n/a。
  - `tune`：選中 P00，記為「未送分」；沒給 R10 時 FAIL_OPENING 用無開局版。
  - `apply`、`decide`、`check_submission`：OK。

**獨立核對**

| 核對 | 做法 | 結果 |
|---|---|---|
| dump 逐列重算（`S/verify_p03_dumps.py`，648 秒） | 每臂抽 inner 40 列、驗證題 40 個 slot。從原始 SGF 重算：inner 經 `load_rows(part="inner")`，不經分片；執色取 CSV 的 `color` 欄和題目檔的 `color_i`。用 tutorial 的特徵、自己寫的 numpy 視角轉換與 D4，以官方 `GoRankResNet(501)` 的 fp32 NCHW forward（不加 hook）載入 `final.pth` | 8 種對稱和執色對調的最大差 4.0e-7；權重 SHA、key、執色、label、NaN 列都一致 |
| apply 逐題重算（`S/verify_apply.py`） | 只用 numpy 從 dump、`opening_val.npz` 和凍結參數重算 | P01 12 題最大差 6.8e-16；P00 5.8e-16 |
| `SmokeDumps` 測試 | `C10_P03_SMOKE_RESULTS=S/smoke/drive/results/prereg03_smoke` | passed |
| unpack 拒收 | 用 I1 故障測試留下的真 zip | `*_failed_nan.zip`、`*_failed_timestop.zip` 因檔名拒收；改名後因缺 history 和 dump 拒收；A 的 zip 用 `--arm b` 拒收；正式模式 unpack 冒煙 zip 拒收（`config.json smoke True`），`results/prereg03/` 沒有寫進任何東西 |

**真實資料**（預覽模式，只寫 scratch）

- `P tools/p03_rank_fuse.py selfcheck --out-dir S/selfcheck_real`：PASSED，21 秒。
  - P00 的 inner OOF 是 0.5854702521055817，和 R20 差 0.0，也等於登記值。
  - 全部擬合資料重擬合的參數和 `results/prereg02/rank/frozen_config.json` 最大差 0.0；refit loss 兩邊都是 1.294020404166363。
  - 驗證題 1,589 題和 `rank_val_probs.csv` 最大差 0.0，規則 B 和 `pred_B.csv` 都沒有不同。
  - 凍結工具和 prereg02 記的 SHA 相同；prereg02 的 8 個輸入檔和帳上相同，也都已 commit。
- `P tools/p03_rank_fuse.py check --out-dir S/check_real`：9 秒，status ok，兩臂 not_run，只剩 P00。C5 的 8 種對稱 accuracy 是 21.43–25.58%。

**git**

- `git status --porcelain` 對所有凍結路徑都是空的：`tools/p02_*`、`t02_*`、`official_log.py`、`check_submission.py`、`c10data.py`、`decide.py`、`score.py`、`make_val.py`、`colab/*prereg0[12]*`、`colab/*test02*`、`run_local_smoke.py`、`results/prereg0[12]`、`results/test02`、`score_ledger.csv`、`docs/VALIDATION.md`、`docs/PREREG_0[12]*`、`docs/TESTDAY.md`、`data/`。
- 追蹤中的檔只有 `docs/COLAB_RUN.md` 有改動（I1 加的一節）；其他都是新檔。

### 1.2 修正（只在 p03 檔）

1. **接縫：冒煙 dump 走不完 check**（`tools/p03_rank_fuse.py:320-322, 1099-1103, 1152, 1687`）。
   - 冒煙模型只訓練 70 步、inner 400 盤，執色接線（> 0.2 個百分點）和 κ 的判定都只是雜訊。這次 A 臂 8.5% 對 9.5% 就被擋下，後面的 selfcheck、tune、apply 都沒辦法在真的 notebook 輸出上測。
   - 改法：新增只限 `--smoke` 的旗標 `--smoke-record-colour`（沿用 I2 的 `--min-acc` 做法）。停止照算，寫進 `colour.stops_recorded_only_smoke`，`thresholds.colour_stops` 記「recorded only」，但不算停止。
   - 正式模式在 `resolve()` 就拒絕這個旗標，停止條件完全不變。
   - 說明也更新了（`tools/p03_rank_fuse.py:12-14`）。
2. **測試**（`tests/test_p03_rank_fuse.py:700-713`）：
   - 同一組執色接錯和 κ 的假資料，加這個旗標後 status ok，記下的停止和沒加旗標時相同。
   - 正式模式加這個旗標會被拒絕：`--out-dir` 指到 tmp，並確認沒有寫出任何檔。

### 1.3 要記進 PREREG_03 rev 2 的項目（科學內容不變；要在任何正式 prereg03 輸出產生前 commit）

**notebook（I1）**

1. **channels_last**：官方 `forward` 在 `bn_in2` 之後用 `view` 攤平，channels_last 的張量不能 `view`。
   - 改法：`bn_in2` 加 forward hook 轉回 NCHW；`network.py` 不改，`final.pth` 可以直接載入官方類別。
   - 驗證：數值和官方 NCHW 相差 < 1e-5（I1 的測試）；另外不加 hook 的官方 forward 重算 dump，差 ≤ 4e-7（1.1）。
2. **規格沒寫死、由實作定的細節**：
   - 輪次從 1 起算，所以 B 臂的雜湊字串是 `…|acc|1|…` 起。
   - 每一輪開始時 random／numpy／torch 的種子是 42·1000＋pass。
   - worker 的 shuffle 種子是 (42·1000＋pass)·1000＋worker；webdataset shuffle 的 initial 是 100。
   - warmup：λ(t)＝(t＋1)/1000；之後 cosine，第 T 步剛好是 0。
   - 建模型前 `torch.manual_seed(42)`，記錄初始權重 SHA，兩臂相同。
3. **登記沒列、另外加的停止條件**：
   - 訓練前把 train、inner 分片的 key 全讀一遍核對，Colab 上約多 1 分鐘。
   - 每一輪結束核對盤數、key 集合、執黑盤數，不符就停，新狀態 `pass_check_failed`。
   - 收尾時的 `final_check_failed` 本來就在登記 3.5 第 6 條。
4. **停下後不能重跑**：訓練開始後任何一種停止都會留下 `STOPPED.json`，之後「全部執行」一律在 preflight 停下；要重跑得先寫修訂。
5. **key 檔比對方式**：npz 的位元組含時間戳，所以比「內容摘要」（逐陣列的 dtype、形狀、位元組）。
   - 正式執行釘 `d53cb540…`／`05c8031a…`（產生器從 repo 的 prereg02 key 檔讀出）。
   - 冒煙時直接和 `prereg02_smoke` 的檔比。
6. **打包檔核對**：打包檔的檔案清單必須等於 prereg01／02 `env.json` 記的那一份；特徵指紋釘 `5d29cf51…`。
7. **帳號紀錄**：notebook 讀不到是哪個帳號，所以由使用者回報、Claude 記進 RESULTS（登記 1.4 本來就寫 RESULTS；plan 3.8 寫的是 run_log）。
8. **第 300 步的時間推估**從第 1 輪建 loader 開始計時，含 loader 啟動，偏保守。
9. **`history.csv`**：停在第 T 步的那一輪，`samples_expected` 是空值。

**本機工具（I2）**

10. **selfcheck「每一列都相同」的解讀**（prereg02 沒有存逐題的 OOF）：R20 OOF 紀錄的每一項和每一折（≤ 1e-12，計數完全相同）、逐盤 log 機率逐元素相等、驗證題逐題比對。
11. **κ 的 95% CI**：用 p02 的 `PlayerBootstrap` 抽樣（每級內重抽玩家、`default_rng(0)`、1,000 次），統計量用各題平均，和 κ 的定義一致。分等級平均版本只記錄。
12. **浮點容許**：G3 比照 G1 加 1e-12；SE_val 的 Var_l 用 ddof＝1。
13. **FAIL_OPENING 平手**：無開局版要嚴格高於 R10（加 1e-12）才選它，平手維持 R10。
14. **結構問題一律判為 stop**（exit 1，tune 拒絕），不是「不能用」；只有地板不過才是「不能用」。
15. **key 陣列比對**：比值和形狀，dtype 只記錄。
16. **unpack 紀錄檔**：`results/prereg03/unpack_rank_{a,b}.json`，記 zip 的 SHA-256。

**整合（本節）**

17. **冒煙專用的 `--min-acc`、`--smoke-record-colour`**：正式模式拒絕，登記的門檻不變。
18. **執色接線門檻的寫法**：登記寫「0.2 個百分點以上」，plan 和工具寫「> 0.2」。inner 73,556 盤時，accuracy 差一定是 1/735.56 個百分點的整數倍，不可能剛好等於 0.2（要 147.11 盤）；Δacc ≥ 0.5 也一樣（要 367.78 盤）。所以兩種寫法結果相同，rev 2 註明即可。

### 1.4 還在的風險

1. **只有 Colab 才走得到的路徑，這次都沒測**：
   - GPU 關卡、G4／A100 的 worker 數（16／32／10）、Linux fork 的多 worker loader（冒煙是 0 個 worker，I1 在同一程序內模擬多 worker 切分）。
   - bf16 autocast 加 channels_last hook 在 CUDA 上的組合（CPU 冒煙時 AMP 關）。
   - Drive 的 `flush_and_unmount`、第 300 步的時間推估。
   - 10/04 的正式執行是這些的第一次實測。出錯一律停下、留 `STOPPED.json`，不會悄悄產生錯的結果。
2. **bf16 的加速沒有實測**：推估的 52–115 分鐘靠「bf16 快 1.5–2 倍」。沒加速時，fp32 每步 151–154 ms 乘 33,768 步約 85–87 分鐘，仍在 150 分上限內；A100 較慢，上限是 180 分。
3. **B 臂的 loader**：`wds.select` 放在 `tarfile_to_samples` 之後，不收的盤不解碼特徵（符合登記），但分片仍要整份解壓讀過。B 約 24 輪，讀的量約是 A 的 2 倍。
   - G4 有 32 個 worker、48 vCPU，應該夠。A100 只有 10 個 worker、12 vCPU，B 可能被 loader 拖慢。
   - 第 300 步的推估會擋下，但那一臂就不算結果，要寫修訂重跑。
4. **執色語意只在合成資料上驗過**：冒煙模型沒學到東西，κ 和接線檢查在真的 notebook 輸出上只算出雜訊。真正的第一次判定是正式 check。
5. **正式模式的 key 摘要比對**只在產生器端和 prereg02 已 commit 的 key 檔核過；notebook 端的正式比對要到 Colab 才會發生。
   - 假如 Colab 的 pandas 或 numpy 改版，讓 `df_val` 的順序變了，會在訓練前停下（安全，只損失約 5 分鐘）。
6. **zip 放在 repo 內**：COLAB_RUN 第 18 步叫使用者把 zip 放進 `results\prereg03\`，但 `.gitignore` 還沒有 `results/prereg03/*.zip`，`git add results/prereg03` 會把 zip 一起加進去。
7. **冒煙的 R20 對照組靠 scratch 裡的 harness**（monkeypatch p02 的地板和選法），只供冒煙；正式模式的 selfcheck 直接比 prereg02 已 commit 的檔，這次也跑過了。

### 1.5 交給主 session（不在 p03 檔的範圍）

- `.gitignore` 加 `results/prereg03/*.zip`（I1、I2 都提過）。
- 建 `data/colab_upload/prereg03/`，只放兩份 notebook；COLAB_RUN 第 7 步會叫使用者從那裡拿。
- COLAB_RUN 的 test02 一節第 2 步列的 Drive 內容，之後要加上 `old_prereg03`（I1 提過）。
- PREREG_03 rev 2：寫進 1.3 第 1–18 項，並在 Colab 執行前 commit。

### 最終測試（修正後）

- `C10_SLOW=1 C10_P03_SMOKE_RESULTS=S/smoke/drive/results/prereg03_smoke P -m pytest -q -rs tests`：455 passed、5 skipped，638 秒。
  - 被略過的 5 項都在 `test_t02_dryrun.py`，要 `C10_SLOW=2`（約 12 分鐘）才跑。
  - prereg02、test02 的測試全部通過，包含 `C10_SLOW=1` 的 test02 dry run 和 R20／R10 真實重擬合。
- 之後 `test_p03_rank_fuse.py` 又改了最後一段斷言（正式模式那一行改用 tmp 的 `--out-dir`）。重跑 `P -m pytest -q tests/test_p03_rank_fuse.py tests/test_p02_hygiene.py tests/test_t02_hygiene.py`：48 passed，42 秒。
- pyflakes 沒有警告。
- 冒煙、工具鏈的腳本和紀錄留在 `S`：
  - 腳本：`chain.sh`、`tools_chain.sh`、`p02_tune_force_r20.py`、`verify_p03_dumps.py`、`verify_apply.py`
  - 紀錄：`smoke_p0*.log`、`tools_chain_p02.log`、`tools_chain_p03.log`、`pytest_full_*.log`、`verify_p03_dumps.json`

## 2. 獨立審查（三個角度，唯讀）

三位審查者都沒讀答案檔、沒跑 `score.py`、沒用 GPU、沒改 repo。

| 角度 | 結論 |
|---|---|
| 訓練正確性（training） | 沒有 blocker，找不到會悄悄訓錯的路徑。實際核對：視角轉換與執色層、執色的來源、B 臂的 E_L 與每一輪收下的盤數、學習率排程與接續、bf16 的範圍、channels_last hook、資料順序的種子、NaN 停止、逐盤輸出的格式、worker 記憶體、loader 速度。2 個 minor（Drive 垃圾桶、接續時的分片排列） |
| 登記一致性（registration） | 科學內容符合 PREREG_03 rev 1，沒有 blocker 或 major；整合報告 1.3 第 1–18 項都認可。另有 5 個 minor：本機工具沒執行「停下的臂不算結果」、停止的分類和 1.3 第 4 項的說法、preflight 查不到空間時只提醒、`answers` 字樣、格子順序 |
| 外洩與使用者步驟（leak-user） | 外洩面沒有問題（答案檔、只用 fit 訓練、inner 只監看和調參、驗證題的用法、只讀目錄、凍結檔）。使用者步驟 1 個 major（Pro+ 路線的 Drive 空間估太低）、5 個 minor（接續時的分片排列、開檔備案、上傳資料夾、`.gitignore`、test02 一節的資料夾清單） |

### 2.1 發現清單（去重後 11 項；修正者逐項對照程式確認）

| # | 來源 | 嚴重度 | 問題 | 確認方式 |
|---|---|---|---|---|
| R1 | training＋leak-user | **major**（leak-user）／minor（training） | Colab 刪掉或覆寫 Drive 上的檔會進垃圾桶、照樣占配額。原本每一輪都覆寫 `state.pt`、刪掉 `pass_{N-2}.pth`，A 約 12 輪、B 約 24 輪，最壞合計十幾 GB；preflight 只在每一臂開頭查一次 3 GB。Drive 寫滿時的錯誤不走 `stop_arm`，會在中途出紅字停下、浪費運算單元；10/03 的回報也沒收 Pro+ 的 Drive 空間 | 程式：原本訓練格存 `state.pt` 用 `.tmp`＋`os.replace`，每一輪 `os.remove` 刪 `pass_{N-2}.pth`；preflight 的 `MIN_DRIVE_FREE_BYTES = 3 * 10 ** 9`。冒煙實測 `state.pt` 316,564,437 bytes、`.pth` 105,585,979 bytes。「刪掉會進垃圾桶」是 Colab 已知行為，本機無法實測；覆寫的舊版會不會也進垃圾桶未確認，照最壞情況處理 |
| R2 | training＋leak-user | minor | 接續時的 recipe 比對沒涵蓋分片排列。Pro+ 路線上兩臂用不同種 GPU 時，B 第一次自己抽特徵（NPROC＝B 那台的 CPU 數），接續時卻先從 A 寫的 `prereg03/features_rank` 還原（A 那台的 NPROC），分片的檔名和內容排列都不同，資料順序因此改變，但 recipe 比對照樣通過 | 程式：官方 9 的還原順序 `(FEAT_BACKUP, P02_FEAT, P01_FEAT)`；分片檔名是 `{tag}-w{i}-{n}.tar.gz`，`extract_split` 用 `iloc[i::NPROC]` 切；recipe 只有 `features_sha256`，指紋不含 NPROC。AI Pro 路線兩臂都從 prereg01 的備份還原，不受影響 |
| R3 | registration | minor | 本機工具沒執行登記 1.3「任一停止，那一臂不算結果」：unpack 和 check 只看 `run_log` 最後一筆是不是 completed，同一個 Drive 資料夾停下過、又重跑完成的結果會被收下 | 改之前 `tools/p03_rank_fuse.py` 沒有任何 `stopped` 或停止狀態的判斷；test02 有先例（`t02_rank.py` 的 `FAILED_RUN_STATUSES`、PREREG_02 rev 7） |
| R4 | registration | minor | 停止的分類和登記文字不一致：「接續時 recipe 不同」和每一輪結束時的學習率核對都是 SystemExit、不留 `STOPPED.json`；整合報告 1.3 第 4 項「訓練開始後任何一種停止都會留下 STOPPED.json」這句不正確 | 程式：原本訓練格的 recipe 比對與學習率核對都直接 `raise SystemExit` |
| R5 | registration | minor | plan 3.2 第 1 格和 3.5 第 2 條把「Drive ≥ 3 GB」列為不過就停；`shutil.disk_usage` 拋 OSError 時 notebook 只提醒、照樣往下跑，這個偏離沒記錄 | 程式：原本 preflight 的 `except OSError` 只 `print` |
| R6 | registration | minor | plan 4.1 寫「p03 的工具和 notebook 都不含 `answers` 字串」；兩份 notebook 各有 1 處，是沿用 prereg02 準備格的拒收 assert | `grep -c`：notebook A、B 各 1，工具、產生器 0 |
| R7 | registration | minor | 格子順序和 plan 3.2 不同：沿用的準備格（掛 Drive、複製並解壓打包檔、解開訓練資料）在 GPU 關卡（官方 2）之前 | notebook 格序：setup → prepare → official_1 → environment → official_2 → preflight_p03 |
| R8 | leak-user | minor | prereg03 的開檔步驟沒有 test02 一節那個「選單裡沒有 Google Colaboratory」的備案 | COLAB_RUN 原本的第 8、12 步 |
| R9 | leak-user | minor | 第 7 步要打開的 `data\colab_upload\prereg03` 還不存在（主 session 已列待辦） | `ls data/colab_upload/`：只有 `C10`、`prereg02` |
| R10 | leak-user | minor | `.gitignore` 沒擋 `results/prereg03/*.zip`（主 session 已列待辦） | `git check-ignore` 沒有輸出 |
| R11 | leak-user | minor | test02 一節列的 Drive 內容沒有 `old_prereg03`（主 session 已列待辦） | COLAB_RUN test02 一節第 2 步、第 6 步 |

## 3. 修正紀錄（FIXER）

- 原則：科學內容（配方、兩臂的取樣、T、選法、閘門、打分）一律不動。偏離規格或登記文字的地方全部列在 3.3，要寫進 PREREG_03 rev 2。
- 沒有 commit，沒有改任何凍結檔，沒讀答案檔，沒跑 `score.py`，沒用 GPU。
- 以下 `S=C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03build\FIX`，`P=.venv/Scripts/python.exe`，都從 repo 根目錄執行。

### 3.1 逐項處理

| # | 狀態 | 做了什麼 |
|---|---|---|
| R1 | **已修** | ① `pass_N.pth` 改存在 Colab 本機磁碟 `WORK/prereg03_pass_ckpt_<臂>`（仍只留最後兩輪），不寫進 Drive：接續只用 `state.pt`、最後輸出只用 `final.pth`，沒有任何步驟讀 `pass_N.pth`。Drive 上每一輪只剩 `state.pt` 的覆寫（約 0.32 GB 進垃圾桶）：A 約 11 次、B 約 23 次。② 每一輪開始前查 Drive 剩餘空間，不到 2 GB 就先停下：狀態記 `drive_space_pause`、送出 Drive 的寫入，不寫 `STOPPED.json`、不打包 failed zip（不是登記的停止：這一輪還沒開始，上一輪的 `state.pt` 已存好）。紅字用中文教使用者到垃圾桶永久刪除 `state.pt` 後再「全部執行」，會從這一輪的開頭接續。③ preflight 的 3 GB 下限不變（登記值），訊息改寫、說明垃圾桶。④ COLAB_RUN：第 2 步多收 Pro+ 的 Drive 儲存空間截圖；第 4 步寫明各臂會用掉多少（A 約 4 GB，在 Pro+ 另加特徵備份 1.3 GB；B 約 8 GB），並先看垃圾桶；第 15 步和「出狀況時」加這種停下的處理方式。帳號分級表（登記 1.4）**沒改**，只註明 Drive 空間不影響帳號的決定 |
| R2 | **已修** | ① B 臂的還原來源只試 prereg02、prereg01 的唯讀備份，不用 A 臂寫的 `prereg03/features_rank`。B 臂要嘛和 A 臂用同一份唯讀備份（AI Pro），要嘛每次都自己重抽（Pro+；同一種 GPU 的 NPROC 相同）。② `scan_keys()` 另回傳每個分片裡 key 的順序；新函式 `shard_layout_sha256()` 算分片排列的 SHA-256：依檔名排序的每個分片和它裡面 key 的順序，沒有 key 的分片也算，和壓縮位元組、worker 數無關。訓練分片的那一個寫進 `shard_key_check` 和 recipe 的 `train_shard_layout_sha256`，接續時不同就不接續。③ 接續被拒時訊息分兩種：`GPU_CLASS` 不同、而且只差 `GPU_CLASS`／`WORKERS`／分片排列 → 換回中斷前那一種 GPU；其他 → 截圖給 Claude |
| R3 | **已修** | `tools/p03_rank_fuse.py` 新增 `STOP_STATUSES`＝(`time_stop`, `nan_stop`, `pass_check_failed`, `final_check_failed`) 和 `stop_record_problems()`。以下任一種：unpack 拒收、check 判 stop（訊息 “a stopped arm is not a result (PREREG_03 1.3): write a revision first”）。<br>• `run_log` 任何一筆是這些狀態<br>• `run_log` 頂層狀態是這些<br>• `config.json` 頂層有 `stopped`<br>• 有 `STOPPED.json`<br>`drive_space_pause` 和狀態是空的（計算前的拒絕、斷線）不算停止。新測試（`test_local_tool_knows_every_registered_stop`）核對 notebook 裡 `stop_arm('…')` 的狀態集合和工具的 `STOP_STATUSES` 完全相同 |
| R4 | **已修** | 每一輪結束時的學習率核對改走 `stop_arm('pass_check_failed', …)`：這一輪已經用那個學習率訓練過，照登記的停止處理，不再是可以重跑的 SystemExit。「接續時 recipe 不同」維持計算前拒絕（不寫 `STOPPED.json`、存好的狀態不動、絕不用不同的設定接續），並把分類照實寫進 rev 2（3.3 第 4 項改寫） |
| R5 | **已修** | preflight 查不到 Drive 剩餘空間時也停（SystemExit，中文說明：刪掉執行階段後再全部執行，還是一樣就截圖給 Claude）。屬於計算前的拒絕 |
| R6 | 不改程式，記進 rev 2 | 兩份 notebook 唯一含 `answers` 字樣的，是沿用 prereg02 準備格的拒收 assert（打包檔裡有答案檔就停），不讀任何答案檔。修正後仍各 1 處（`test_cells_compile_ids_and_token` 有斷言） |
| R7 | 不改程式，記進 rev 2 | 照實記錄格子順序；不重排（要重新冒煙和審查，而 COLAB_RUN 第 11 步本來就先看右下角的 GPU 名稱） |
| R8 | **已修** | COLAB_RUN 第 8 步加「選單裡沒有 Google Colaboratory」的備案（照 test02 一節的寫法，另提醒右上角頭像要是那個帳號）；第 12 步沿用 |
| R9 | **留給主 session** | 這次的凍結清單包含 `data/**`，所以沒有建 `data/colab_upload/prereg03/`。主 session 建資料夾、只放兩份 notebook，核對 SHA-256 等於 3.4 的新值（舊值 `d020c4bd…`、`6a9c5c98…` 已作廢） |
| R10 | **已修** | `.gitignore` 加 `results/prereg03/*.zip`（含 `*_results_failed.zip`、`… (1).zip`）。`git check-ignore` 確認擋住 zip，但不擋 `rank_a/config.json` 和 `unpack_rank_a.json` |
| R11 | **已修** | COLAB_RUN test02 一節第 2 步的兩種情況、第 6 步做完後的清單都加上 `old_prereg03`，並註明 prereg03 改在 Pro+ 跑時，AI Pro 的 `C10` 不會有它 |

### 3.2 改了哪些檔

| 檔案 | 改動 |
|---|---|
| `colab/build_prereg03_rank_nb.py` | 第 0 格說明、差異表 5 列；preflight ③；官方 9 的還原順序；官方 11 的 `_sample_key`／`scan_keys`／`_shard_name`／`shard_layout_sha256`；官方 12 的分片排列紀錄；訓練格的 `PASS_CKPT_DIR`、`MIN_PASS_FREE_BYTES`、`drive_free_bytes()`、`drive_space_pause()`、recipe 的 `train_shard_layout_sha256` 和 `RECIPE_GPU_KEYS`、接續被拒的訊息、學習率核對、`pass_N.pth` 的位置 |
| `colab/prereg03_rank_{a,b}.ipynb`、`results/prereg03/design/nb_diff_{a,b}.txt` | 由產生器重新產生（沒有手改） |
| `tools/p03_rank_fuse.py` | `STOP_STATUSES`、`STOPPED_FILE`、`stop_record_problems()`；`cmd_unpack` 和 `arm_structure` 呼叫它；說明文字 |
| `tests/test_p03_nb.py` | `scan_keys` 新的回傳值；`test_shard_layout_digest`（重抽但壓縮位元組不同 → 相同；另一種切法、分片內順序不同、多一個空分片 → 不同；和 worker 切法無關）；`ReviewFixes` 7 項 |
| `tests/test_p03_rank_fuse.py` | Unpack：4 種停止狀態重跑後完成、`config.json` 的 `stopped`、`STOPPED.json` 都拒收，暫停和計算前拒絕之後完成的照收；EndToEnd 結構測試加 3 種 |
| `docs/COLAB_RUN.md` | 第 2、4、8、12、14、15 步，「出狀況時」3 列，test02 一節的 3 份資料夾清單 |
| `.gitignore` | `results/prereg03/*.zip` |

### 3.3 要記進 PREREG_03 rev 2 的項目（科學內容不變；在任何正式 prereg03 輸出產生前 commit）

1.3 的第 1–18 項照舊，**第 4 項改寫**，另加第 19–27 項。

**4（改寫）停止分三類**

- (a) **登記的停止，這一臂不算結果**：
  - 會留 `STOPPED.json`、打包 `*_results_failed.zip`，之後的「全部執行」在 preflight 就停。
  - 狀態：`time_stop`、`nan_stop`、`pass_check_failed`（每一輪的盤數、key、執黑、學習率核對，或沒讀到資料）、`final_check_failed`。
  - 本機工具只要看到任何一次，就拒收那一臂（第 24 項）。
- (b) **計算前的拒絕，可以重跑**：
  - GPU 關卡；preflight（`STOPPED.json`、打包檔、Drive 空間，含查不到、只讀指紋）。
  - 打包檔、特徵指紋、外洩、執色與 D4、E_L、key 檔、分片 key 的檢查。
  - `state.pt` 讀不進來。
  - 接續時 recipe 不同：只差 GPU 種類（連帶 worker 數或分片排列）時換回中斷前那一種；其他差異由 Claude 判斷，需要時先寫修訂。
  - 這些都發生在這次執行做任何一步訓練之前，存好的狀態不動。登記 1.3 列的「GPU 不對、各項檢查不符、接續時配方不同 → 停」指的是「不在這種情況下計算或接續」，不是讓這一臂作廢。這一臂絕不用不同的設定接續。
- (c) **不是停止，從最後存好的一輪接續**：
  - `drive_space_pause`（第 20 項）。
  - 斷線，或執行中的例外（例如 worker 讀分片出錯、CUDA 錯誤）：還沒存的那一輪直接丟掉，等同斷線。
  - 同一個錯誤重複出現時截圖給 Claude。

**19.** `pass_N.pth` 只存在 Colab 的本機磁碟（仍只留最後兩輪），不寫進 Drive（plan 3.6 原寫在 `ckpt/`）。沒有任何步驟讀它；接續只用 `state.pt`，最後輸出只用 `final.pth`。

**20.** 每一輪開始前 Drive 剩不到 2 GB 就先停下，狀態 `drive_space_pause`。plan 沒有這一條，是運作上的規則：
- 清出空間後從那一輪的開頭接續，和斷線後接續相同。
- 冒煙實測：暫停再接續的 `final.pth`、6 個逐盤輸出，都和沒中斷的那次逐位元組／逐陣列相同。

**21.** preflight 查不到 Drive 剩餘空間時也停（照 plan 3.2、3.5 第 2 條；rev 1 的實作只提醒）。

**22.** 特徵還原與分片排列：
- B 臂只從 prereg02、prereg01 的唯讀備份還原，不用 A 臂寫的 prereg03 備份。plan 3.2 第 4 格只寫了「只有 A 寫備份，B 抽到本機」。
- 訓練分片排列的 SHA-256 寫進 recipe 設定（`train_shard_layout_sha256`），接續時不同就不接續。這讓 plan 3.6「接續後資料順序和沒中斷時相同」在 Pro+、兩臂不同 GPU 的情況下也成立。

**23.** 每一輪結束的學習率核對不符屬於 (a)，狀態 `pass_check_failed`。rev 1 的實作是可以重跑的 SystemExit。

**24.** 本機工具拒收停下過的臂（unpack 拒收、check 判 stop），比照 test02（PREREG_02 rev 7）：
- 依據：`run_log` 任何一筆，或頂層是 (a) 的狀態、`config.json` 有 `stopped`，或有 `STOPPED.json`。
- `drive_space_pause` 和狀態是空的不算。

**25.** plan 4.1「p03 的工具和 notebook 都不含 `answers` 字串」的例外：兩份 notebook 各有 1 處，是沿用 prereg02 準備格的拒收 assert（打包檔裡有答案檔就停），不讀任何答案檔。工具和產生器是 0 處。

**26.** 格子順序：GPU 關卡（官方 2）在沿用的準備格之後（掛 Drive、複製並解壓打包檔、解開訓練資料、記 env），仍在任何抽特徵和訓練之前。
- 拿到不對的 GPU 時會多花幾分鐘，所以 COLAB_RUN 第 11 步先看右下角的 GPU 名稱。
- plan 3.2 的順序是 preflight（GPU 關卡）在最前面。

**27.** COLAB_RUN 第 2 步（10/03 回報）多收 Pro+ 的 Drive 儲存空間截圖。登記 1.4 的帳號分級不變；空間只影響要不要先清垃圾桶、會不會中途暫停。

### 3.4 最後驗證

**重建**

- `P colab/build_prereg03_rank_nb.py --out-dir S/build1`，以及 `P colab/build_prereg03_rank_nb.py D:\AI-Portfolio\competitions\aicup2026-go --out-dir S/build2`。
- 4 個輸出兩次逐位元組相同，也和工作目錄的檔相同。A、B 兩份仍只差 ARM 一行。

| 檔案 | sha256 |
|---|---|
| `colab/prereg03_rank_a.ipynb` | `7f2d306348e4258883aa63011d43f68e74fb7ccc6a6ec75ee4d02814f2f57db0` |
| `colab/prereg03_rank_b.ipynb` | `ebc37f5b7d317df66233dab5f038b08d699555f9d28ba6b48dccb56cd64a09b7` |
| `results/prereg03/design/nb_diff_a.txt` | `7568d4743fb5f41d2b0e7cbf28a5eb4170f8d35cd46e4f4d2924265d278f2fc5` |
| `results/prereg03/design/nb_diff_b.txt` | `fcd7c68eb3a2782fcaaf2f6f56ba354d8a0c67345e97483369a50c13179af1ca` |

**測試**

- `C10_SLOW=1 P -m pytest -q -rs tests`：462 passed、6 skipped，1,943 秒（和冒煙同時跑，所以比平常慢）。
  - 略過的是 5 項要 `C10_SLOW=2` 的 test02 dry run，和 1 項要 `C10_P03_SMOKE_RESULTS` 的 `SmokeDumps`。
- 之後又加了 `test_local_tool_knows_every_registered_stop`，所以重跑 p03 相關的測試：
  - 指令：`C10_SLOW=1 C10_P03_SMOKE_RESULTS=S/smoke/drive/results/prereg03_smoke P -m pytest -q -rs tests/test_p03_nb.py tests/test_p03_rank_fuse.py tests/test_p02_hygiene.py tests/test_t02_hygiene.py`
  - 結果：81 passed、0 skipped，193 秒。含 `SmokeDumps`，也含真實 fit 的 `RealFit`。
- pyflakes 3.0.1 檢查 4 個 p03 檔：沒有警告。

**全新冒煙**（`S/chain1.sh`、`S/chain2_main.sh`；同一個 `S/smoke`，只有第一步加 `--fresh`；後半段和 pytest、暫停測試同時跑，所以比平常慢）

| 步驟 | 結果 |
|---|---|
| prereg01 | OK，764 秒 |
| prereg02 | OK，1,624 秒 |
| prereg03 A | OK，1,456 秒 |
| prereg03 B | OK，1,287 秒 |

- **和整合時的冒煙相同**（`S/compare_smoke.py` → `S/compare_smoke.json`）：
  - `final.pth` 的 SHA-256：A `8a026f9e…`、B `17728538…`，和整合時完全相同。
  - 6 個 dump 逐陣列相同；`history.csv` 的步數、盤數、執黑、`order_sha256`、`d4_sha256`、loss、inner accuracy 都相同。
  - 所以這次的改動沒有動到訓練本身。
- **R1**：
  - Drive 的 `ckpt/` 只剩 `final.pth`、`history.csv`、`state.pt`。
  - `pass_1.pth` 在本機的 `work/prereg03_pass_ckpt_{a,b}_smoke/`。
- **R2**：
  - 兩臂的 `train_shard_layout_sha256` 都是 `5acefc74…`，等於 `shard_key_check.train.layout_sha256`。
  - 分片：train 8／8、val 2／2 都讀到 key。

**暫停與接續**（`S/chain2_pause.sh`，在 `S/smoke_pause`，也就是跑完 prereg02 後的另一份拷貝）

- 測試用的 notebook（`S/make_pause_nb.py`）只改一行：設 `C10_TEST_PAUSE=1` 時，第 2 輪開始前回報 Drive 剩 0 bytes。只存在 scratch，不上傳。
- **第 1 次執行**（測試版）：
  - exit 1，紅字是「Drive 只剩 0.00 GB，第 2 輪還沒開始就先停下…」。
  - `run_log` 狀態 `drive_space_pause`，`paused_before_pass` 是 2。
  - `state.pt`：第 1 輪、第 63 步，還沒完成。
  - 沒有 `STOPPED.json`；Drive 的 `ckpt/` 只有 `history.csv`、`state.pt`。
- **第 2 次執行**（正式 notebook）：
  - OK，722 秒。從第 2 輪接續（`loaded_state`、`resumed_training`、`passes_trained [2]`），`run_log` 是 [`drive_space_pause`, `completed`]。
  - `final.pth` 的 SHA-256 等於沒中斷的那次（`8a026f9e…`），6 個 dump 逐陣列相同。

**本機工具**

- 新冒煙的工具鏈（`S/tools_chain.sh`，沿用整合時的腳本，只改 `S`）全部照預期：
  - prereg02 本機鏈 9 步、R20 對照組（scratch harness），都是 exit 0。
  - p03：
    - unpack a／b：OK。
    - 照登記門檻的 check：exit 1，停在執色接線（對調 9.5% 比正確 8.5%，冒煙雜訊，同整合時）；之後的 selfcheck 也是 exit 1。
    - 主鏈：check、selfcheck、tune、apply、`decide.py`、`check_submission.py` 都是 0。
      - selfcheck 4 項比較都 ok。
      - tune 選 P02（OOF 0.3365，P00 0.3083）；G1 過，G2 的 CI [−0.0029, 0.0670] 沒過，`score_allowed` false。
    - 登記地板的鏈：只剩 C5，選 P00，`score_allowed` false。
  - 分數和整合時不同：prereg01／02 的冒煙模型不能逐位元組重現（`inner_p02e9_d4` 兩次的 logits 最多差 0.82，PREREG_02 rev 4 第 7 條）。prereg03 兩臂可以。
- **另外加測**（`S/extra`）：
  - 接續過的 zip（`run_log` 是 [`drive_space_pause`, `completed`]）：unpack 照收；只放 A 臂、加冒煙旗標（`--min-acc 0 --smoke-record-colour`）時 check 是 status ok。
  - 把同一個 zip 改成下面 3 種，unpack 都拒收，也沒有解開任何檔：
    - `run_log` 第 1 筆改成 `time_stop`
    - `config.json` 加 `stopped`
    - 加 `STOPPED.json`

**真實資料**（預覽模式，只寫 scratch）

- `P tools/p03_rank_fuse.py selfcheck --out-dir S/selfcheck_real`：PASSED，20 秒。
  - P00 的 inner OOF 是 0.5854702521055817，和 R20 差 0.0。
  - 重擬合參數差 0.0，refit loss 兩邊都是 1.294020404166363。
  - 驗證題 1,589 題最大差 0.0；規則 B 和 `pred_B.csv` 都沒有不同。
- `P tools/p03_rank_fuse.py check --out-dir S/check_real`：status ok，兩臂 not_run，只剩 C5，5 秒。

**git**

- 凍結路徑的 `git status --porcelain` 是空的：`tools/p02_*`、`t02_*`、`official_log.py`、`check_submission.py`、`c10data.py`、`decide.py`、`score.py`、`make_val.py`、`colab/*prereg0[12]*`、`colab/*test02*`、`run_local_smoke.py`、`results/prereg0[12]`、`results/test02`、`score_ledger.csv`、`docs/VALIDATION.md`、`docs/PREREG_0[12]*`、`docs/TESTDAY.md`。
- `data/` 沒有動。
- 追蹤中的檔只有 `.gitignore`、`docs/COLAB_RUN.md` 有改動，其他都是新檔。

### 3.5 還在的風險

1. **暫停機制靠 `shutil.disk_usage` 回報 Drive 配額**，而且配額要算進垃圾桶、更新得夠快。這些只有在 Colab 上才驗證得到。
   - prereg02 在 Colab 記到的 Drive 剩餘空間是 191,142,047,744 bytes，和本機工作磁碟的 201.6 GB 不同，看起來是配額，但不算證明。
   - 如果回報的其實是本機磁碟，暫停永遠不會觸發，只剩「`pass_N.pth` 不進 Drive」（進垃圾桶的量 A 臂少約 1 GB、B 臂少約 2.2 GB）和 COLAB_RUN 第 4 步的空間說明。
2. **Drive 超過配額時，可能不是寫入當下就出錯**，而是背景上傳失敗。每一輪開始前留 2 GB 的餘裕就是為了避開這種情況；兩臂同時寫時，每一輪結束各約寫 0.8 GB。
3. **Pro+ 路線的 B 臂每次執行都重抽特徵**，包括接續：G4 約 6 分鐘，A100 只有 12 vCPU 會更久。改之前 B 也幾乎一定要重抽，因為 A 的備份來不及寫完；多出來的只有接續時那一次。
4. **暫停要使用者回來處理**。沒人處理時執行階段開著，一直扣運算單元；COLAB_RUN 已寫「一時沒辦法處理就先刪執行階段」。
5. **只有 Colab 才走得到的路徑，這次的改動也沒測**，同 1.4：多 worker 的 fork、bf16、Drive 的 `flush_and_unmount`。
   - 冒煙時 `scan_keys` 是 0 個 worker；多 worker 的切法由 `test_shard_layout_digest` 在同一個程序裡模擬。

### 3.6 交給主 session

- 建 `data/colab_upload/prereg03/`，只放兩份 notebook，SHA-256 要等於 3.4 的值。
- PREREG_03 rev 2：1.3 第 1–18 項，第 4 項照 3.3 改寫，再加第 19–27 項；在 Colab 執行前 commit。
- 這次改了產生器，照 CLAUDE.md 的規矩，交給使用者前再請一個代理獨立審查這次的修正（3.1–3.3）。
- 腳本和紀錄留在 `S`：
  - 腳本：`chain1.sh`、`chain2_main.sh`、`chain2_pause.sh`、`make_pause_nb.py`、`compare_smoke.py`、`tools_chain.sh`、`p02_tune_force_r20.py`
  - 紀錄：`smoke_*.log`、`tools_chain_p0{2,3}.log`、`pytest_full.log`、`pytest_p03.log`、`compare_smoke.json`、`extra/`

## 4. 第二輪審查與修正（FIXER，2026-09-27）

第二輪審查（workflow `wf_1e29e8c2-494`：程式 `review:code`、使用者步驟 `review:user`）沒有 blocker 或 major，共 14 個 minor。其中 PREREG_03 的三項文字（修訂列條號、第 10 節第 4 條「不能用」和工具的 stop、`pass_check_failed` 少了「沒有讀到任何資料」）主 session 已在 rev 2 第 10 節改好，這次沒動 PREREG_03，程式和文件照它的第 10 節寫。其餘 11 項都修了；另外照主 session 的指示，GPU 型號查不到時不再印 `None`。

- 原則同第 3 節：科學內容（配方、兩臂的取樣、T、選法、閘門、打分）不動。
- 沒有 commit，沒有改任何凍結檔，沒讀答案檔，沒跑 `score.py`，沒用 GPU。
- 以下 `S=C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03min\fix2`，`P=.venv/Scripts/python.exe`，都從 repo 根目錄執行。

### 4.1 逐項處理

| # | 來源 | 問題 | 狀態 | 做了什麼 |
|---|---|---|---|---|
| S1 | code | 接續時 `state.pt` 已記訓練完成，訓練格不再寫 `history.csv`；收尾只核對記憶體裡的 history，所以會打包出 `completed`、但 `history.csv` 停在前一輪的 zip，本機工具判 stop | **已修** | 訓練格新增 `HISTORY_PATH`、`history_file_problems()`：讀 Drive 上的 `ckpt/history.csv`（全部當字串讀），欄名、列數和 9 個欄位（pass、pass_complete、steps、samples、black、global_step、run、order_sha256、d4_sha256）都要等於記憶體裡的 history，最後一列要是第 T 步。收尾格在打包前先核對；不符就用 `save_history()` 照記憶體裡的重寫，再核對一次，還是不符才算收尾核對不符（`final_check_failed`）。結果記在 `outputs.history_file`（重寫前的問題、有沒有重寫、重寫後的問題） |
| S2 | code＋user | Drive 空間暫停時也會印「Drive 已寫入完成並卸載」：第 15 步第一點、第 16 步、「出狀況時」的粗體總則都會把使用者導向「刪執行階段、等 Claude」 | **已修** | 第 15 步改成照順序比對三點，暫停那一點放第一，寫明「這時沒有 zip，不要做第 16、17 步，不要刪執行階段」；第 16 步只在「完成。請下載…」或「這一臂不算結果」時做；粗體總則改成「表格有寫的照表格做」，並列出全部可以自己處理的情況；D 段加「不要一看到紅字就刪執行階段」 |
| S3 | code＋user | 暫停門檻 2 GB、preflight 門檻 3 GB，暫停後照訊息清到「變多」重跑，可能又在 preflight 停下；Pro+ 的 A 臂在 preflight 之後才寫約 1.3 GB 備份，開頭剩 3.0–3.35 GB 時會在第 1 輪前暫停、清垃圾桶也解不開 | **已修** | ① 兩個門檻統一成 `MIN_PASS_FREE_BYTES` 3 GB，定義在 preflight 格，訓練格沿用（不再自己定義）。② preflight 新增 `feature_backup_marker_ok()`：A 臂在三個還原來源（和官方 9 同樣順序）都沒有設定相符的完成標記時，`WILL_WRITE_FEATURE_BACKUP` 為真，開頭門檻是 3＋2＝5 GB（`FEATURE_BACKUP_ROOM_BYTES`）；備份約 1.3 GB（從冒煙分片推估真實資料 1.34 GB，prereg01 一節估 1–2 GB），寫完後第 1 輪前仍剩 3 GB 以上。③ 暫停和 preflight 的訊息都寫出要清到的確切數字（`gb_text()`：3 GB 或 5 GB）、怎麼算剩餘空間（「共 … GB」減「已使用 … GB」）、垃圾桶清空了還不夠時 C10 的東西都不要刪、改看 COLAB_RUN「Drive 空間不夠時」。④ `config.json` 的 preflight 記 `min_drive_free_bytes`、`min_pass_free_bytes`、`will_write_feature_backup` |
| S6 | code | COLAB_RUN 第 4 步說 AI Pro「還剩約 191 GB，不用擔心空間」，這是還沒證實的讀數 | **已修** | 拿掉這句。第 4 步改成兩個帳號都看左下角、自己算剩餘空間（AI Pro 至少 3 GB、Pro+ 至少 5 GB），並截一張圖在第 18 步回報時給 Claude，用來和 `config.json` 的 `drive_free_bytes` 對照（3.5 第 1 條的風險可以在第一次正式執行後核對） |
| U1 | user | 同 S2（粗體總則只列兩種例外；第 16 步的觸發條件） | **已修** | 同 S2 |
| U2 | user | 暫停後重跑要過 preflight 的 3 GB，暫停訊息只講 2 GB；表格沒有 preflight 那則紅字 | **已修** | 同 S3；表格的「Drive 只剩 … GB」一列涵蓋兩種訊息（暫停、開頭檢查），新增一節「Drive 空間不夠時」寫逐步做法和要清到的數字；「其他紅字」那一列註明清垃圾桶是唯一的例外 |
| U3 | user | 開跑那天沒有在指定帳號再看一次可用量 | **已修** | 第 8 步加一點：三件都對之後先「執行階段 → 查看資源」，A＋B 要 ≥ 50、只跑 A 要 ≥ 25，不夠就不要按「全部執行」，截圖告訴 Claude；第 12 步註明 B 臂不用再查（沒有重新編號，避免牽動其他步驟的引用） |
| U4 | user | 開頭引言「Claude 說可以跑了之後才用這一節」和第 1–3 步（先回報）矛盾 | **已修** | 引言改成「10/03 先做第 1–3 步，做完就停；第 4 步以後等 Claude 回覆寫明帳號和 A 或 A＋B 之後才做」；檔頭第 3 行同步；第 1–3 步和第 4 步起的小標題也註明 |
| U5 | user | Pro+ 路線沒處理 `C10` 或打包檔已存在；「找不到打包檔」那一列在 Pro+ 上禁止唯一的修法，而且實際畫面是英文 | **已修** | 第 5 步：已有 `C10` 就用它、不要再建；已有 `C10_bundle_v1.zip` 就不要再傳；跳出「上傳選項」按「取消」並截圖；傳完確認剛好一個、在 `C10` 裡。表格那一列改寫英文原文 `FileNotFoundError: [Errno 2] No such file or directory: '/content/drive/MyDrive/C10/C10_bundle_v1.zip'`（準備格 `zipfile.ZipFile(BUNDLE_ZIP_SRC)` 丟的），分 AI Pro、Pro+ 兩種做法；Pro+ 可以等上傳打勾、把檔拖進 `C10` 或照第 5 步上傳。打包檔「不同」另列一列（照 notebook 的訊息：不要重傳、截圖） |
| U6 | user | 授權步驟少了「Google Drive for desktop 是正常的」 | **已修** | 第 10 步補回 |
| U7 | user | 沒說明停下時 Colab 顯示的 SystemExit 英文框架、要截哪裡 | **已修** | 「出狀況時」開頭加一段：`An exception has occurred…`、`SystemExit: ` 加中文說明、最後的 `UserWarning: To exit: use 'exit'…` 不用理也不要照做；截圖要包含「SystemExit:」那一整段中文；其他英文錯誤截最後 10 行左右。第 15 步引用它 |
| U8 | user | 要使用者到垃圾桶找永遠不會出現的 `pass_…pth` | **已修** | 第 4 步改成「沒有要留的就清空垃圾桶；有要留的只刪 `state.pt`、`best_model.pth`、`epoch_` 開頭的 `.pth`」；notebook 的兩則訊息也改成 `state.pt`、`.pth` 結尾 |
| G1 | 主 session 指示 | 有 CUDA 但讀不到 GPU 型號時，GPU 關卡的紅字寫「這次拿到的 GPU 是 None」 | **已修** | 官方 2 新增 `GPU_MODEL_SHOWN`（有 CUDA、型號是 None 時為「查不到 GPU 型號」，CPU 冒煙仍是 None）。關卡的紅字改成「有 GPU，但查不到 GPU 型號，無法確認是 G4（RTX PRO 6000）或 A100，停在這裡…」（查得到型號時照舊「這次拿到的 GPU 是 X，不是…」）；`device` 那一行和 preflight 的通過訊息改印 `GPU_MODEL_SHOWN`。`state.pt`、`config.json` 仍記原始的 `GPU_MODEL_NOW` |
| — | code | PREREG_03 rev 2 修訂列條號、第 10 節第 1 條少一個原因、第 4 條「不能用」 | 主 session 已修 | 讀過 rev 2 第 10 節（第 13、167、172 行），程式和工具的行為與它一致，這次沒改 PREREG_03 |

### 4.2 改了哪些檔

| 檔案 | 改動 |
|---|---|
| `colab/build_prereg03_rank_nb.py` | 差異表 4 列（官方 2、preflight、訓練、結尾）；第 0 格說明（開始時的空間、跑完的樣子、會停下的情況）；官方 2 的 `GPU_MODEL_SHOWN` 和關卡紅字；preflight ③ 的門檻、`gb_text()`、`feature_backup_marker_ok()`、`WILL_WRITE_FEATURE_BACKUP`、訊息與紀錄；訓練格的 `HISTORY_PATH`、`HISTORY_COLS_CHECKED`、`history_file_problems()`、暫停訊息，拿掉訓練格自己的 2 GB 門檻；結尾格的 history 核對與重寫、`outputs.history_file` |
| `colab/prereg03_rank_{a,b}.ipynb`、`results/prereg03/design/nb_diff_{a,b}.txt` | 由產生器重新產生（沒有手改） |
| `data/colab_upload/prereg03/prereg03_rank_{a,b}.ipynb` | 換成新的兩份（`data/` 不進 git，位元組和 `colab/` 相同） |
| `tests/test_p03_nb.py` | `test_drive_space_pause` 改用 preflight 的 3 GB、核對訊息裡的確切數字；`test_preflight_stops_when_drive_space_unknown` 核對單一門檻；新類別 `SecondReviewFixes` 4 項：`history.csv` 核對（沒有檔、過期一輪、重寫後、某欄不同、沒到 T，含全是數字加 e 的 SHA 字串）、結尾「核對 → 重寫 → 再核對 → 打包」的順序、preflight 門檻 7 種情況（A 沒備份 5 GB；B；AI Pro 的 prereg01 備份；Pro+ 接續時 A 自己的備份；設定不同；標記壞掉；標記不是 dict）與訊息、GPU 關卡（查不到型號、L4、A100、CPU） |
| `docs/COLAB_RUN.md` | 檔頭第 3 行；「事前登記 03」一節：引言、第 1–5、7、8、10–12、15、16、18 步、D 段、「出狀況時」（SystemExit 說明、總則、表格重排與改寫）、新增「Drive 空間不夠時」 |
| `results/prereg03/design/build_review.md` | 本節 |

### 4.3 要記進 PREREG_03 rev 2 的項目（主 session 處理；在任何正式 prereg03 輸出產生前 commit）

1. **第 10 節第 1 條 (c)**：「每一輪開始前剩不到 2 GB」改成 **3 GB**（和 preflight 同一個門檻 `MIN_PASS_FREE_BYTES`）。
2. **preflight 的 Drive 門檻**（第 10 節第 7 條或另加一條；屬 (b) 計算前的拒絕）：登記的 3 GB 不變；A 臂這次要重抽特徵、把分片備份寫到 Drive 時（prereg03、prereg02、prereg01 的 `features_rank` 都沒有設定相符的完成標記），開頭要 **5 GB**（多留 2 GB 給約 1.3 GB 的備份，寫完後第 1 輪前仍有 3 GB）。比登記嚴，運作上的規則，不影響判定。
3. **收尾核對多一項**（實作說明；屬第 1 條 (a) 的 `final_check_failed`）：Drive 上的 `ckpt/history.csv` 要和記憶體裡的 history（這次訓練的，或從 `state.pt` 讀回的）逐列相同、記到第 T 步；不同就先照記憶體裡的重寫再核對，重寫後仍不同才是 `final_check_failed`。起因：最後一輪存好 `state.pt` 之後、寫完 `history.csv` 之前斷線，接續時訓練格不會再寫 `history.csv`，本機工具會把那個 zip 判 stop。
4. 只是實作說明，可不寫：GPU 型號查不到時紅字寫「查不到 GPU 型號」（第 10 節第 11 條 GPU 關卡）；COLAB_RUN 第 4 步在開跑時多收一張 Drive 儲存空間截圖（第 10 節第 9 條只寫 10/03 收 Pro+ 的），用來核對 `drive_free_bytes`。

### 4.4 最後驗證

**重建**

- `P colab/build_prereg03_rank_nb.py`（寫進 repo）、`P colab/build_prereg03_rank_nb.py --out-dir S/b1`、`P colab/build_prereg03_rank_nb.py D:\AI-Portfolio\competitions\aicup2026-go --out-dir S/b2`：4 個輸出三份逐位元組相同。A、B 兩份仍只差 ARM 一行（產生器的 assert）。
- `data/colab_upload/prereg03/` 的兩份和 `colab/` 的兩份 `cmp` 相同、SHA-256 相同。

| 檔案 | sha256 |
|---|---|
| `colab/prereg03_rank_a.ipynb`（＝`data/colab_upload/prereg03/prereg03_rank_a.ipynb`） | `85e0c2ae3269c68ad166310efb2525b2d48ed1279c9cf6eb6b428e59e8693557` |
| `colab/prereg03_rank_b.ipynb`（＝`data/colab_upload/prereg03/prereg03_rank_b.ipynb`） | `aa3669accceca23fc3ab2439bcab68204281bcec999cd1e00d43dd16825bd8b3` |
| `results/prereg03/design/nb_diff_a.txt` | `e4aa08270ce59918a9d2f845d1e532ade020bd637094bb3dcde774773f7a0e80` |
| `results/prereg03/design/nb_diff_b.txt` | `8fdd02e3b175c9bdcce51a5f73a322b2f9cfa34ead005ed2af23e66a3801c989` |

3.4 的值（`7f2d3063…`、`ebc37f5b…`、`7568d474…`、`fcd7c68e…`）已作廢。

**測試**

- `C10_SLOW=1 P -m pytest -q -rs tests`：467 passed、6 skipped，3,082 秒（和冒煙同時跑）。略過的是 5 項要 `C10_SLOW=2` 的 test02 dry run，和 1 項要 `C10_P03_SMOKE_RESULTS` 的 `SmokeDumps`。跑前跑後的 `git status --porcelain` 相同。
- `C10_P03_SMOKE_RESULTS=S/smoke/drive/results/prereg03_smoke P -m pytest -q tests/test_p03_nb.py -k SmokeDumps`：1 passed。
- pyflakes 3.0.1（系統 Python 3.10）檢查 4 個 p03 檔：沒有警告。

**全新冒煙**（`S/chain_main.sh`）

- prereg01、prereg02 的冒煙樹沿用第 3 節 `p03build\FIX\smoke` 的輸出（`S/make_base.sh` 用 `cp -a` 複製 `prereg01_smoke`、`prereg02_smoke`、打包檔、解好的訓練資料；不複製 prereg03 的任何輸出、本機分片、`pass_N.pth`）。所以 prereg03 兩臂從零開始，分片從 `prereg01_smoke` 的唯讀備份還原。
- A 臂：OK，1,836 秒；B 臂：OK，1,914 秒（和 pytest、其他冒煙同時跑）。
- 和第 3 節的冒煙相同（`S/compare_fix2.py` → `S/compare_fix2.json`）：`final.pth` A `8a026f9e…`、B `17728538…`；兩臂 6 個 dump 逐陣列相同；`history.csv` 的步數、盤數、執黑、`order_sha256`、`d4_sha256`、loss、inner accuracy 相同。所以這次沒有動到訓練本身。
- preflight：兩臂的門檻都是 3 GB（A 臂找到 `prereg01_smoke` 設定相符的備份標記，`will_write_feature_backup` false）。`outputs.history_file`：沒有重寫、沒有問題。zip 的 `history.csv` 2 列、最大 global_step 70。

**暫停與接續**（`S/chain_pause.sh`，在另一份全新的樹 `S/smoke_pause`）

- 測試用 notebook（`S/make_pause_nb.py`，同第 3 節，只在 scratch）：第 2 輪開始前回報 Drive 剩 0 bytes。
- 第 1 次執行：exit 1（預期），1,042 秒。紅字是「Drive 只剩 0.00 GB，第 2 輪還沒開始就先停下（每一輪開始前要剩 3 GB 以上）。這是暫停…這時沒有 zip，不用刪執行階段…等 Drive 剩 3 GB 以上…垃圾桶清空了還不到 3 GB：C10 裡的東西都不要刪…」。`run_log` 狀態 `drive_space_pause`、`paused_before_pass` 2；`state.pt` 第 1 輪、第 63 步、未完成；沒有 `STOPPED.json`；Drive 的 `ckpt/` 只有 `history.csv`、`state.pt`。
- 第 2 次執行（正式 notebook）：OK，1,118 秒。preflight 門檻 3 GB，從第 2 輪接續（`loaded_state`、`resumed_training`、`passes_trained [2]`），`run_log` 是 [`drive_space_pause`, `completed`]。`final.pth` 等於沒中斷的那次（`8a026f9e…`），6 個 dump 逐陣列相同。
- 本機工具 `unpack --arm a --smoke` 照收這個 zip（exit 0）。

**訓練完成後接續的 history 測試**（S1；`S/chain_hist.sh`、`S/make_stale.py`）

- 做法：拿 A 臂跑完的樹（`S/hist_base`，A 臂跑完時的快照）複製兩份，改成審查描述的狀態：`history.csv` 只留第 1 輪（最後一輪存好 `state.pt` 之後、寫完 `history.csv` 之前斷線）、那次執行的狀態改成 None（沒跑完）、刪掉舊的結果 zip。再分別用新舊 notebook 重跑。
- **新 notebook**（`85e0c2ae…`）：OK，224 秒。印「已訓練完 70 步（2 輪），這次不訓練」「ckpt/history.csv 和記憶體裡的 history 不同（ckpt/history.csv 是 1 列 21 欄，記憶體裡的 history 是 2 列 21 欄），照記憶體裡的重寫」；`outputs.history_file` 是 rewritten true、重寫後沒有問題。zip 的 `history.csv` 2 列、最大 global_step 70，`run_log` [None, `completed`]。本機工具 `unpack` exit 0，`check --smoke --min-acc 0 --smoke-record-colour`（只放 A 臂）exit 0，C6A usable。
- **舊 notebook**（第 3 節的 `7f2d3063…`，只當對照）：一樣 OK、打包出 `completed` 的 zip，但 zip 的 `history.csv` 只有 1 列。本機工具 `unpack` 照收，`check` exit 1：`arm a: history.csv largest global_step 63 != T 70 (final.pth must be step T)`。這重現了審查說的問題，也證明新版修好了。

**A 臂要寫分片備份的路徑**（S3；`S/chain_backup.sh`，另一份全新的樹 `S/smoke_backup`，只在這份樹裡拿掉 `prereg01_smoke/features_rank`，模擬 Pro+ 第一次跑）

- 第 1 次執行：OK，1,883 秒。preflight 印「Drive 剩 422.73 GB（門檻 5 GB）」，`will_write_feature_backup` true、`min_drive_free_bytes` 5,000,000,000；特徵 `extracted`，備份寫到 `prereg03_smoke/features_rank`（有 `_COMPLETE.json`）。`final.pth` 等於主冒煙的 A 臂（`8a026f9e…`）。
- 第 2 次執行（跑完後再按一次「全部執行」）：OK，54 秒。preflight 門檻回到 3 GB（A 臂自己的備份標記相符），特徵 `local_reused`，已訓練完、不訓練，重新打包。

**git**

- 凍結路徑的 `git status --porcelain` 是空的：`tools/p02_*`、`t02_*`、`official_log.py`、`check_submission.py`、`c10data.py`、`decide.py`、`score.py`、`colab/*prereg01*`、`colab/*prereg02*`、`colab/*test02*`、`run_local_smoke.py`、`results/prereg01`、`results/prereg02`、`results/test02`、`score_ledger.csv`、`docs/VALIDATION.md`、`docs/PREREG_01*`、`docs/PREREG_02*`、`docs/TESTDAY.md`。
- `docs/PREREG_03_RECIPE_V2.md` 只有主 session 的 rev 2 改動，這次沒碰。`data/` 只改了 `data/colab_upload/prereg03/` 的兩份 notebook。

### 4.5 還在的風險

1. **`shutil.disk_usage` 是不是 Drive 配額**：同 3.5 第 1 條，只能在 Colab 上驗證。COLAB_RUN 第 4 步現在會收一張開跑時的儲存空間截圖，第一次正式執行後可以和 `config.json` 的 `drive_free_bytes` 對照。
2. **分片備份的大小是推估**：1.34 GB 是從冒煙分片的每盤大小外推（真實資料每個分片 1,000 盤，tar 的額外負擔比冒煙小）。實際超過 2 GB 的話，A 臂在 Pro+ 第一次跑會在第 1 輪前暫停；這時垃圾桶是空的，要照「Drive 空間不夠時」第 4 點處理（刪自己的檔或找 Claude）。
3. **preflight 只看完成標記的設定**：標記相符、但分片檔壞掉時，官方 9 會重抽並重寫備份（`rmtree` 舊備份也會進垃圾桶），這時開頭只查了 3 GB。每一輪開始前的 3 GB 暫停會擋住，不會寫壞東西。
4. **暫停後在同一個執行階段重按「全部執行」**：`flush_and_unmount` 之後設定格會重新掛 Drive，可能再跳授權。COLAB_RUN 寫了「跳出授權就照第 10 步」，本機測不到。
5. **Colab 顯示 SystemExit 的英文字樣**（`An exception has occurred…`、`UserWarning: To exit…`）是 IPython 的標準行為，沒有在目前的 Colab 畫面上截圖核對。
6. **Pro+ 的 Drive 如果很小**（例如共 15 GB、已用很多），3 GB 的暫停門檻會讓兩臂同時跑時中途暫停好幾次，每次都要使用者回來清垃圾桶。第 4 步已寫「剩得越多，中途暫停的次數越少」；10/03 看到 Pro+ 的截圖時，Claude 可以據此建議先清空間或只跑一臂。
7. **冒煙和第 3 節一樣只在 CPU、0 個 worker 上跑**：GPU 關卡的新訊息、多 worker、bf16、Drive 卸載，都要到 Colab 才第一次實測（同 1.4、3.5）。

### 4.6 交給主 session

- PREREG_03 rev 2：照 4.3 第 1–3 項改第 10 節（第 4 項可選），在 Colab 執行前 commit。
- 這次改了產生器，照 CLAUDE.md 的規矩，交給使用者前再請一個代理獨立審查這次的修正（4.1–4.3）。
- 腳本和紀錄留在 `S`：
  - 腳本：`make_base.sh`、`chain_main.sh`、`chain_pause.sh`、`make_pause_nb.py`、`chain_hist.sh`、`make_stale.py`、`chain_backup.sh`、`compare_fix2.py`、`old_prereg03_rank_a_7f2d.ipynb`（對照用的舊 notebook）
  - 紀錄：`chain_*.status`、`smoke_p03{a,b}.log`、`smoke_pause_run{1,2}.log`、`hist_{new,old}_run.log`、`hist_{new,old}_{unpack,check}.log`、`hist_{new,old}_check/check_report.json`、`smoke_backup_run{1,2}.log`、`pytest_full.log`、`compare_fix2.json`（notebook 冒煙的紀錄是 cp950 編碼）

## 5. 官方單色結構（rev 3）：建置、審查與修正（2026-09-28）

起因：AIdea「資料」分頁的官方說明（`docs/DATA.md` 最後一節）寫明競賽二每題 5／10／15／20 盤、同一題只有一種執色；本登記的 inner 假題、本機驗證集和 2.2 第 4 條的 κ 檢查都是黑白混合。流程：一個建置代理（I-P03）→ 一個唯讀審查代理（和 test02 的 rev 8 一起審）→ 一個修正者（本節最後）。**使用者還沒跑 Colab，沒有任何 prereg03 的真實輸出或分數**；沒有 commit，沒讀答案檔，沒跑 `score.py`，沒用 GPU，凍結檔沒動。暫存在 `…\scratchpad\osfix\`（`I-P03\`：建置；`review\`：審查；`fix\p03\`：修正）。

### 5.1 建置（I-P03，只改 `tools/p03_rank_fuse.py`、`tests/test_p03_rank_fuse.py`）

- **單色 inner 假題（只記錄）**：`mono_questions`／`mono_set`／`mono_qset`，常數 `MONO_*`。對象是 `pseudo_inner.json` 的 bank 玩家（沿用原本的折）；每位玩家、每種執色的池依 `sha256('c10-prereg03|mono|order|{rank}|{game_id}')` 排序；第 0–31 輪，k∈{5,10,15,20} 且 k 不超過池的盤數，每題一個種子 `sha256('c10-prereg03|mono|q|{r}|{player_id}|{c}|{k}')`。`tune` 用各折 OOF 參數打這些題（先核對各折參數重算混色評估輪的機率和 OOF 逐位相同），記全體／黑／白／各 k，以及選定組合減 P00 的配對 PlayerBootstrap CI。
- **單色 κ 參照**：`kappa_single_colour`、`val_colour_subquestions`、`question_boot_mean`。inner 單色假題上的 Δκ 和 PlayerBootstrap CI；本機驗證題依題目檔的 `color_i`（經 `val_keys.npz`，不讀答案）拆成單色子題（≥2 盤）算 Δκ。單色條件＝Δacc ≥ 0.5 個百分點且 inner 單色 Δκ 95% 下界 > 0。寫在 `check_report.json` 各臂的 `colour.kappa_single_colour`，`frozen_config.json.single_colour.kappa_reference` 另存選定組合用到的 C6 來源那一份。
- 新的程序性拒絕：正式模式沒有任何單色假題；`tune` 讀到的 `check_report.json` 沒有單色紀錄或題目集 SHA 不同；各折參數重算不出 OOF 機率。
- 真實資料預覽（沒有任何一臂）：120,544 題、623 人、1,170 個（玩家、執色）組，黑 59,744、白 60,800，題目集 SHA `45d0f14acd98…`；驗證子題 2,986 個（黑 1,492、白 1,494），1,589 題都至少有一個。P00 的單色 OOF（scratch 預跑）0.5833（混色 0.5855），黑 0.5647、白 0.6017。
- selfcheck 在真實資料照舊重現 R20（P00 inner OOF 0.5854702521055817）。

### 5.2 審查（唯讀；結論：沒有 blocker、major）

和 p03 有關的是第 1 件（minor）：建議的 rev 3 第 11 節 (c) 第 2 種情況「單色條件成立、但驗證單色子題 Δκ ≤ 0 → 03 不在測試日使用」在 check／tune 時就看得出來，但條文又寫 H3R 不變、`score_allowed` 也不看單色紀錄，已知測試日用不上的 03 仍會花掉一次打分（整季 4 次，已用 1 次）。另記一件和這次改動無關的：`tool_state` 記的未 commit 檔名少了第一個字（`'ools/p03_rank_fuse.py'`）。

### 5.3 修正（FIXER）

| # | 處理 |
|---|---|
| 審查第 1 件 | 選了審查建議的 (a)：**不跑 H3R，打分機會保留**。新函式 `single_colour_score_block(cid, kappa_reference)`：選定組合用到的每個 C6 來源，`test_day_reference.single_colour_condition` 為真而 `val_sub_question_dkappa_positive` 不是真（≤ 0 或沒有子題），或者根本沒有單色參照，就列一條原因。`tune`：`score_allowed` ＝ 閘門過 且 不是 P00 且 沒有這種原因；原因寫在 `tune_report.json`、`frozen_config.json` 的 `score_blocked_single_colour`，並印出。`apply`：從 `frozen_config.json` 的 `single_colour.kappa_reference` 重算同一件事，併進 `score_blocked_reasons`，另記 `score_blocked_single_colour`。選法、閘門 G1–G3、參數、無開局版、FAIL_OPENING 的選擇、`rank_val_probs.csv` 都不受影響。docstring 和註解同步改（單色 inner 分數仍然只記錄；κ 參照的第 2 種情況只會擋 H3R）。 |
| 另記的一件 | `tool_state` 改用 `git(..., binary=True)` 讀 `git status --porcelain`，不再被 `strip()` 吃掉第一行開頭的空白；只影響紀錄，「有沒有未 commit」的判斷本來就對。`tools/p02_rank_fuse.py:483` 有同樣寫法，是凍結檔，不動。 |

新增測試（`tests/test_p03_rank_fuse.py`，共 45 項，原 33＋建置 9＋這次 3）：
- `SingleColour.test_single_colour_score_block`：條件真／假 × 驗證 Δκ 正／負／沒有、沒有參照、P00、P03 兩個來源、P04／P05 只看自己的來源。
- `ToolState.test_porcelain_paths_keep_their_first_letter`：假的 porcelain 輸出，第一個檔名完整。
- `EndToEnd.test_single_colour_block_withholds_h3r`：合成世界每個組合都是 1.0、一定選 P00，所以測試裡強制選 P01 並讓閘門過（測的是 `score_allowed`，不是選法）。先確認沒有擋、`score_allowed` 為真；把 `check_report.json` 的單色參照改成「條件真、驗證 Δκ −0.001」後重跑 `tune`：`combo`、`params`、`selection`、`gates`、`se_val`、`oof`、`no_opening`、`fail_opening` 都相同，`score_allowed` 變假、原因只列 C6A；`apply` 也是假、`rank_val_probs.csv` 逐位元組相同；把 frozen 的參照改回正值，`apply` 重算變真（證明 apply 是重算、不是照抄）；條件改成假時不擋。
- `test_single_colour_records_are_record_only` 加註：那個世界的單色條件成立且驗證 Δκ > 0，所以擋不會觸發，`score_allowed` 仍要逐位相同。

### 5.4 驗證（修正後）

| 項目 | 結果 |
|---|---|
| `PY -m pytest -q tests/test_p03_rank_fuse.py` | 45 passed（90 秒） |
| 真實資料 selfcheck 預覽（`fix\p03\selfcheck\`） | PASSED，P00 inner OOF 0.5854702521055817（＝R20）；和改前的 `I-P03\selfcheck_before\selfcheck.json` 逐欄比，只有 `created`、`seconds`、`tool.files_sha256`、`tool.dirty_or_untracked` 不同（後者現在正確寫成 `tools/p03_rank_fuse.py`） |
| 真實資料 check 預覽（`fix\p03\check\`） | status ok、可用來源 `['C5']`；`single_colour` 整段和審查時的預覽相同（120,544 題、SHA `45d0f14a…`；子題 2,986 個） |
| 完整測試與 C10_SLOW=2 演練 | 見 `results/test02/design/build_review.md` 第 5 節 |

### 5.5 交給主 session

- PREREG_03 rev 3 條文（第 11 節）照修正後的程式寫：H3R 的指令、採用條件、預算不變，但多一種不跑 H3R 的情況（(c) 第 2 種；也包含選定的 C6 來源沒有單色參照）。`frozen_config.json` 和 `apply_report.json` 的 `score_blocked_single_colour` 寫原因；第 9 節第 5 步「過閘門才跑第 3 節的指令」改讀 `apply_report.json` 的 `score_allowed`。
- 如果總部決定不採用 (c) 第 2 種情況（改成只記錄），要一起拿掉 `single_colour_score_block` 的呼叫和 3 項測試，否則程式和登記不一致。
- 參考數字（0.5833 等）是 scratch 預跑的；寫進 RESULTS 時改引 commit 後 `tune_report.json` 的數字。
