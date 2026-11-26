# 事前登記 11：競賽二 07／08 的 KataGo 步驟在 4090 不能用時改用 Colab GPU（草稿，等主 session 決定）

**白話版**：07（HS）和 08（P1）的 KataGo 計算原本只登記在本機 RTX 4090 上做，4090 壞掉或環境修不好，07／08 這一季就不用（PREREG_07 5.3、PREREG_08 5.3）。本登記加一條備援：同一份程式（逐位元組相同、SHA 核對）、同一份權重、同樣的精度和批次設定，改在 Colab 的 GPU 上算，算出的檔案格式和 4090 一模一樣，再交給**原封不動**的 07／08／09 本機工具（apply、official_log）判定。只有在客觀條件成立、而且條件已 commit 之後才能用；只能用事先演練過、接縫和容許差都過關的 GPU 型號；每一段只准一次 Colab 嘗試；不准兩邊都跑再挑。容許差一律不改。

- 對象：只有競賽二（TEAM_11248）的 07 HS（`tools/t07_rank.py hs`）和 08 P1（`tools/t08_rank.py p1`）。09 不碰 GPU，讀 08 的塊，不需要備援；R20、04 本來就在 Colab。競賽一不受影響。
- 代號：
  - **段**（stage）：`hs`（07）或 `p1`（08）。每段三組：seam（驗證題前 400 盤）、public、private。
  - **備援**：本登記的 Colab 路徑。**4090 路徑**：PREREG_07／08 原本的路徑。
  - **套件**（kit）：`data/test11/kit/C10_gpu_kit_v1.zip`，內容見第 1 節。
- 本登記寫：什麼時候可以用（第 2 節）、哪些 GPU 可以用（第 3 節）、照什麼順序（第 4 節）、判定（第 5 節，不變）、揭露（第 7 節）。
- **狀態**：草稿，在分支 `gpu-fallback`，沒有合併、沒有任何演練結果。標 【待演練】 的數字要等第 3 節的演練跑完、commit 之後才填。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | （草稿，未 commit 到 main） | 2026-10-10，在任何測試資料之前：程式（`colab_gpu/`、`tools/g11_gpu_fallback.py`、`tests/test_g11_*.py`）、本登記、`docs/TESTDAY_GPU_FALLBACK.md`、`docs/COLAB_GPU_FALLBACK.md`。PREREG_07／08／09 各加一行指向本文件（只加行）。Colab 演練還沒跑。 |
| rev 2 | `cbf8dd2` | 2026-10-10，在任何測試資料之前、A100 第一次演練之後：准用規則改成「seam 照釘住的容許差通過，而且 offgrid 每個差（× 1.0）≤ 現在的容許差」；2.5 × offgrid 改成只記錄；紀錄要帶 `rule_version: 2`，舊規則（rule_version 1）的紀錄一律不算、要重產（第 10 節）。容許差、套件（kit）、測試日判定都不變。 |

## 1. 來源（全部釘死）

1. **程式（套件裡的檔，逐位元組照抄工作樹，SHA 記在 `results/test11/kit_manifest.json`）**：
   - `third_party/katago/**`（KataGo commit `d91ea855…`，`SOURCE.txt` 核對，PREREG_07 1.2）；
   - `results/prereg07/explore/pilot_humansl/hs_lib.py`、`pilot_run.py`、`pilot_fuse.py`；`results/prereg08/p1/p1_lib.py`、`p1_extract.py`、`p1_fuse.py`；`results/prereg05/explore/u1/u1_lib.py`（它的 SHA 是 P1 程式紀錄的一部分，Colab 上不 import）；
   - `colab_gpu/gpu_fallback_core.py`（新檔：讀題目、照 `t07_rank.hs_run`／`t08_rank.p1_run` 的寫法算塊）。
   - Colab 每次開跑都重算每個檔的 SHA，釘住檔（PREREG_07 1.2、PREREG_08 1.2）另核 LF SHA；P1 的程式紀錄（`p1_lib.code_shas()`）必須等於 4090 驗證塊記的那一份。
2. **權重**：`data/katago/b18c384nbt-humanv0.ckpt`，323,752,318 bytes，SHA-256 `84b23476…`。套件裡原檔照放（不壓縮）；Colab 上每次開跑完整重算 SHA。
3. **每一手／每一局面怎麼算**：照 PREREG_07 1.3 和 PREREG_08 1.3，不改。精度 fp16 autocast、channels_last、每批 1,024、`cudnn.benchmark` 開；HS 每組 200 盤一塊、16 批一組；P1 200 盤一塊、4 批一組、中性 metadata（SHA `404bb709…`）。這些值從 `tools/p07_rank_fuse.py`、`tools/p08_rank_fuse.py` 抄進 `KIT_MANIFEST.json`，Colab 程式只讀那裡。
4. **塊的格式**：和 4090 的塊同樣的欄位、同樣的順序、同樣的格式字串（`c10-prereg07 pilot chunk v1`、`c10-test08 p1 chunk v1`），記同樣的金鑰檔／題目檔 SHA、權重 SHA、KataGo commit、hs_lib SHA 或 P1 程式紀錄、torch／cuDNN／GPU／驅動、`created_utc`、模式（測試日 `real`、演練 `rehearsal`）。`tool_sha256` 記 `gpu_fallback_core.py` 的 SHA（不是 t07／t08 的），所以看得出是哪一條路算的。
5. **Colab 環境**：`pip install torch==2.11.0+cu128`（PyTorch 的 cu128 索引，和 `.venv-gpu` 同一個版本字串）、`numpy==2.5.3`，notebook 第 2 格逐字核對 `torch.__version__ == "2.11.0+cu128"`、numpy `2.5.3`；Python、cuDNN、驅動、GPU 型號照實記錄。計算在子程序裡跑，每算完一塊就存回雲端硬碟；斷線後重按「全部執行」接著算。

## 2. 什麼時候可以用（觸發條件）

1. 備援只在下面其中一條**客觀條件**成立、而且用 `tools/g11_gpu_fallback.py trigger` 寫成 `results/test11/trigger_<段>.json` **並 commit 之後**才能用。工具會核對：測試日的輸入檔要在觸發紀錄 commit 之後才建得出來；每一塊的 `created_utc` 都要晚於觸發紀錄的 commit 時間。
   - `hardware_failure`：4090 不見或起不來（`nvidia-smi` 失敗、看不到 RTX 4090、機器無法開機），重開機後仍然一樣。
   - `gpu_exit2`：`t07_rank.py hs`／`t08_rank.py p1` 結束碼 2，原因是 GPU、CUDA 或權重（`CUDA is not available`、GPU 型號不對、`nvidia-smi` 失敗、權重 SHA 不對而且 `data/katago_backup/` 也不對），照 TESTDAY_07／08 的處理做過一次仍然一樣。**不算**：別的 GPU 工作佔著（等它結束）、舊塊、git 不乾淨、前提不符這類可以照手冊修好的 2。
   - `venv_gpu_unrestorable`：結束碼 4 的 `.venv-gpu` 版本不符（`venv_gpu`），照 TESTDAY_07 重裝釘住的版本後仍然不符。
   - `deadline_11_14`：到 **11/14 23:59:59（台灣時間）**，那一段在 4090 上還沒有三組都成功（三組的報告都已 commit）。工具在這個時間之前拒絕這個理由。
2. **不准兩邊都跑再挑**：
   - 觸發之後，那一段的 4090 路徑不再使用。4090 已寫出的部分塊，資料夾改名成 `<組>_4090_<時間>` 留著，不刪、不混用；已經 commit 的 4090 結果（某一組的 `<組>_hs.npz` 等）用 `git mv` 移到 `results/test11/superseded/<段>/`，再匯入 Colab 的那一組。
   - 4090 三組都已成功的段，不能觸發（工具拒絕）。
   - 4090 跑完、但 `apply` 的 GPU 接縫沒過（結束碼 4），**不是**觸發條件：照原登記處理（PREREG_07 5.3、PREREG_08 5.3）。
3. **每一段只准一次 Colab 嘗試**（一個 `run_id`）。續跑只准在同一個環境上（GPU 型號、torch、cuDNN、Python、numpy；程式在每一段第一次開跑時記下環境，之後不同就停，結束碼 4；驅動只記錄）。
   - **什麼算一次嘗試**：Colab 寫出那一段的 `ENV.json`（第一塊開算之前）就算。測試日的輸入檔帶著准用的環境名單（第 3 節），notebook 第 2 格和 Colab 程式在寫任何檔之前就核對；不在名單上就停（結束碼 2），**不算嘗試**，換准用的 GPU 再跑。
   - 匯入工具第一次看到某一段的 Colab 結果時，寫 `results/test11/attempt_<段>.json`（就算之後被拒絕也寫，馬上 commit）；之後出現別的 `run_id` 一律拒絕。這一次嘗試失敗（拒絕、環境不同、接縫或判定沒過），那一段這一季就不用，最終照原登記的順序退回（07 不用 → PREREG_04 5.3；08 不用 → PREREG_07 5.4；09 跟著 08）。
4. 07 用了備援、08 的 4090 卻好了：08 照原路徑在 4090 上算，不需要觸發。反過來也一樣。每一段各自判斷。

## 3. 哪些 GPU 可以用（演練，在任何測試資料之前）

1. **演練的題目**（驗證題，只讀題目內容和執色，題目檔 SHA 照 PREREG_07 1.7 核對）：
   - `seam`：驗證題 val_keys 順序的前 400 盤（列 0–399），分塊和 4090 驗證塊相同，等於測試日 GPU 接縫要比的那一批。
   - `offgrid`：列 4,787–5,786，共 1,000 盤（假測試樹 private 的前 1,000 盤；4,787 不是 200 的倍數，所以每一塊都跨在驗證塊的格線上，批次組合和 4090 不同）。這是 08／09 在 4090 上量容許差時用的同一批盤。
   - 兩段（hs、p1）都跑這兩組，共 4 次。
2. **比法**（`tools/g11_gpu_fallback.py rehearsal-compare`，比較函式全部從測試日工具原樣 import）：
   - seam：`t07_rank.gpu_seam_record`（07 的 GPU 接縫：每手 ≤ 0.05、每盤每 profile 平均 ≤ 0.002、lp_HS ≤ 0.02）、`t08_rank.gpu_seam_record`（標準化特徵 ≤ 0.05、lp_P1TO ≤ 0.04）、`t09_rank.gpu_seam_record`（42 維 EX ≤ 0.05、lp_P1TO_E ≤ 0.04）——**就是測試日 apply 的判定**，參照是 4090 的驗證塊。
   - offgrid：`t07_fake_tests.compare`、`t08_fake_tests.compare`、`t09_fake_tests.compare`（和 4090 驗證塊的同一批盤比）。
3. **可以用的條件**（每段各自判）：seam 的接縫全部通過，failed／counts／n_moves 全同，而且 offgrid 每一個量的 **2.5 × 實測差 ≤ 現在的容許差**：

   | 段 | 量 | 現在的容許差 | Colab 實測（offgrid） | 2.5 × 實測 |
   |---|---|---|---|---|
   | hs | 每盤每 profile 平均 log p | 0.002 | 【待演練】 | 【待演練】 |
   | hs | lp_HS | 0.02 | 【待演練】 | 【待演練】 |
   | p1 | 標準化特徵 | 0.05 | 【待演練】 | 【待演練】 |
   | p1 | lp_P1TO | 0.04 | 【待演練】 | 【待演練】 |
   | p1 | 標準化 EX（42 維） | 0.05 | 【待演練】 | 【待演練】 |
   | p1 | lp_P1TO_E | 0.04 | 【待演練】 | 【待演練】 |

   參考（4090 自己，不同批次組合）：lp_HS 0.0059、lp_P1TO 0.0160、標準化特徵 0.0186、lp_P1TO_E 0.0113、42 維 EX 0.0161——都滿足這一條。
4. **容許差不改**。主 session 原本的規則是「容許差＝max(現值, 2.5 × 實測)，只能由演練、在測試資料之前改」。但容許差寫在已釘住的 `tools/t07_rank.py`、`t08_rank.py`、`t09_rank.py` 和 `tools/official_log.py` 裡，改它就要動測試日路徑。所以本登記把那條規則落實成：**2.5 × 實測 ≤ 現值時容許差就是現值（max 取現值）；超過時，那一種 GPU 不准用**，而不是放寬容許差。
5. 演練結果寫成 `results/test11/rehearsal_<GPU 名>.json`（每段的接縫數字、offgrid 數字、判定、環境）並 commit。**准用的環境＝這個檔裡判定為 allowed 的（GPU 型號、torch、cuDNN、Python、numpy）組合**；驅動只記錄（Colab 會自己更新，不受我們控制）。測試日的輸入檔帶著這份名單，notebook 和 Colab 程式開算前核對；匯入時工具再逐塊核對。
6. **演練紀錄的最後一次 commit 必須早於 2026-11-04 11:00（台灣時間，測試資料公開）**，否則不算（工具核對 commit 時間）；之後不再新增准用的環境。
7. **要演練的 GPU**：Colab Pro+ 實際會拿到的 **A100、H100**，拿得到的話再加 **G4**，一種接一種跑（每種 GPU 的輸出放在雲端硬碟不同的資料夾，互不影響）。每一種各寫一份紀錄（A100 40 GB 和 80 GB 是不同的 GPU 名稱，分開演練、分開准用）。測試日拿到哪一種准用的就用哪一種。
8a. **准用紀錄只對同一份套件有效**：紀錄記下演練時的 kit_id；套件重建（程式或釘住檔有任何改變）之後，舊紀錄不再算，要重新演練。
8b. **防假資料的絆線**：RTX 4090 以外的 GPU，offgrid 的差全部正好是 0.0，視為不是那張 GPU 自己算的，拒絕寫紀錄。
8. **演練只收真正在 Colab GPU 上算的結果**：每一塊的 `tool_sha256` 必須等於套件裡 `gpu_fallback_core.py` 的 SHA、`seconds` > 0、用真的權重；總表不能是 fake-from-val、fake scorer 或 `--max-games`。演練用的 `fake-from-val`（本機假資料）拒絕演練和測試日的輸入檔。

## 4. 順序（測試日）

1. 觸發（第 2 節）→ `trigger` → commit。
2. 那一段原本的前提照樣要成立（`hs`：R20 第一次上傳已記錄、T1、H7R、K6-07……；`p1`：再加 07 的 apply 結束碼 0）：`tools/g11_gpu_fallback.py input --mode real --stage <段>` 經由**原封不動**的 `t07_rank.prereqs`／`t08_rank.prereqs` 檢查，再用 `t07_rank.set_slots` 讀三組題目，寫成輸入檔 `data/test11/upload/C10_gpu_input_<段>_test.zip` 和紀錄 `results/test11/inputs/<段>_test.json`（commit）。
3. 使用者把輸入檔上傳到雲端硬碟，跑 Colab（`docs/COLAB_GPU_FALLBACK.md`），下載結果 zip。
4. `import --set seam` → commit → `import --set public` → commit → `import --set private` → commit。每一組：重算每一塊的 SHA、照 `t07_rank.chunk_check`／`t08_rank.chunk_check` 原樣檢查、再查它們沒看的欄位、三組同一環境（以 seam 為準）、環境在准用名單、塊晚於觸發和 R20 的第一次上傳、只有一個 `run_id`；然後把塊複製到 `data/test07/hs/<組>/`（`data/test08/p1/<組>/`），用原封不動的 `t07_rank.assemble`／`t08_rank.assemble` 寫出 `<組>_hs.npz`、`hs_<組>_report.json`（08：`<組>_p1.npz`、`p1_<組>_report.json`），和 4090 路徑寫的同一種檔；另寫 `results/test11/import_<段>_<組>.json`。
5. 之後的步驟完全照原手冊：`t07_rank.py apply`（`docs/TESTDAY_07.md` 第 4 步起）、`t08_rank.py p1`／`apply`、`t09_rank.py apply`、`official_log.py`。
6. 截止（K6）不變：07／08／09 都是 11/19 23:59:59。Colab 的 hs 約 3–4 小時、p1 約 1 小時（以 4090 的速度估，【待演練】實測後更新）。

## 5. 判定（不變）

- `apply` 和 `official_log.py` 一字不改。它們讀的檔、核對的 SHA、接縫和容許差、本機接縫、手數交叉核對、失敗率、方向檢查、K6、上傳順序和最後一版，全部照 PREREG_07／08／09。
- 已核對（建置代理讀程式確認）：07／08 的 `apply` 只要求三組的報告記的環境（torch、cuDNN、GPU）彼此相同，不和 `frozen_config.json` 或 RTX 4090 比；`official_log.py` 只用報告裡的接縫數字重算條件，不看 GPU。09 的 `apply` 不看環境。所以 Colab 的塊在不改任何釘住檔的情況下就能被接受（`tests/test_g11_dryrun.py` 實際跑過）。
- 本登記**只取代**這幾句（照 PREREG_07 5.5 的前例，不改原文）：

  | 被取代的條文 | 本登記的規則 |
  |---|---|
  | PREREG_07 5.2 `hs` 前提「`.venv-gpu` 的 torch、cuDNN 和 GPU 型號等於 `frozen_config.json` 記的值」 | 觸發之後：環境＝准用名單裡的 Colab 環境（第 3 節），由 `g11 import` 核對 |
  | PREREG_07 5.3「只在本機 RTX 4090 上算」「不登記 Colab 備案」「4090 壞掉……這一季就不用 07」 | 第 2 節的條件成立時，改用本登記的備援；備援失敗才不用 07 |
  | PREREG_08 5.2 `p1` 的同一條前提、PREREG_08 5.3 | 同上（08） |
  | PREREG_09 | 不取代任何條文（09 只讀 08 的塊）；09 的 GPU 接縫照原樣比 08 的 seam 塊 |

## 6. 資料

- 輸入檔含驗證題前 400 盤的題目內容（seam；演練另含列 4,787–5,786），和測試日 public／private 的題目內容（SGF、執色、題號、slot）。**沒有任何答案**。放在使用者的雲端硬碟（和 test02、test04 的測試檔同一個帳號），只供這一場比賽。競賽資料在 Colab 上使用已有前例（test02、test04）。
- 結果 zip 只有塊（數字）和環境紀錄。

## 7. 揭露（用到時寫進 RESULTS 和報告）

- 「07 的 HS（或 08 的 P1 特徵）在本機 GPU 故障後，改在 Google Colab 的 〈GPU 型號〉 上計算（torch 〈版本〉、cuDNN 〈版本〉），程式和權重與本機相同（SHA 核對），格式相同；與本機的驗證題接縫差異為〈數字〉，在登記的容許差內。」
- 觸發的條件和證據（`trigger_<段>.json`）、演練紀錄、用掉的 Colab 嘗試次數（必為 1）。
- 沒用到備援時，報告不提（只在 HISTORY 記有這份登記）。

## 8. 刻意不做的

- 不改任何容許差、不改 07／08／09 的任何工具或判定、不改 official_log。
- 不登記 CPU 備案（PREREG_07 5.3 的理由不變）。
- 不准第二次 Colab 嘗試、不准換 GPU 續跑、不准 4090 和 Colab 的塊混在同一段。
- 不在測試資料公開之後新增准用的 GPU。

## 9. 程式和測試

- `colab_gpu/gpu_fallback_core.py`（Colab 上跑的全部程式；本機演練和冒煙測試也用它）、`colab_gpu/build_gpu_fallback_nb.py` → `colab_gpu/gpu_fallback.ipynb`。`colab/` 資料夾不動（`tests/test_t09_hygiene.py` 凍結）。
- `tools/g11_gpu_fallback.py`：`kit`、`input`、`trigger`、`import`、`rehearsal-compare`、`fake-from-val`（只給演練）、`status`。
- `tests/test_g11_core.py`、`tests/test_g11_tool.py`、`tests/test_g11_hygiene.py`（釘住檔對 tag `p08-testday-pre09` 和 main 都沒變、`colab/` 沒動、只加行的檔只加行）、`tests/test_g11_dryrun.py`（`C10_SLOW=1`：在 R20／04 的假測試樹上，把 4090 驗證塊的同一批盤寫成 Colab 格式的塊 → `g11 import` → 原封不動的 07／08／09 `apply`，輸出等於 `results/test07|08|09/design/` 的基準；另用 FakeScorer 證明 Colab 程式和 `t07_rank.py hs`／`t08_rank.py p1` 寫出的塊逐欄相同）。
- 結果寫在 `results/test11/`（`.gitattributes` 加 `results/test11/** -text`），大檔在 `data/test11/`。

## 10. rev 2：准用規則改成 offgrid × 1.0（2026-10-10，在任何測試資料之前）

決定者：主 session（使用者授權由 Claude 判斷方法，以最終排名為目標）。依據是第一次真正的 Colab 演練；本節寫在任何測試資料之前（測試資料 11/04 11:00 才公開）。

1. **改什麼**：第 3 節第 3 點「可以用的條件」改成：某一段（hs／p1）的某一個 GPU 環境可以用，當且僅當
   - seam 用**現在釘住的容許差**全部通過（07：每手 ≤ 0.05、每盤每 profile 平均 ≤ 0.002、lp_HS ≤ 0.02；08：標準化特徵 ≤ 0.05、lp_P1TO ≤ 0.04；09：42 維 EX ≤ 0.05、lp_P1TO_E ≤ 0.04），failed／counts／n_moves 全同；而且
   - offgrid 每一個量的最大差（**× 1.0**）≤ 現在的容許差。
   - 2.5 × offgrid 的值照樣算出來，**只記錄**（紀錄檔的 `record_only_offgrid_times_2_5`），不進判定。
2. **為什麼**：
   - 第一次演練（Colab Pro+，`NVIDIA A100-SXM4-40GB`，torch 2.11.0+cu128、cuDNN 91900、Python 3.13.15、numpy 2.5.3；紀錄 `results/test11/rehearsal_nvidia-a100-sxm4-40gb.json`，舊規則版 `873095f`）：hs、p1 的 seam 都通過；offgrid 的最大差 hs：每盤每 profile 平均 0.000679、lp_HS 0.00860；p1：標準化特徵 0.01736、lp_P1TO 0.01672、42 維 EX 0.01739、lp_P1TO_E 0.01249。舊規則的 2.5 倍只差一點沒過：lp_HS 0.0215 > 0.02、lp_P1TO 0.0418 > 0.04。
   - 舊規則連 4090 自己都過不了：4090 不同批次組合的 lp_P1TO 是 0.0160，2.5 倍＝0.040，剛好在線上（PREREG_08 第 14 節）。所以 2.5 倍規則在這裡等於任何 GPU 都不准用。
   - 真正擋在測試日的是 seam（同一批盤、同樣分塊，和 4090 驗證塊比，照釘住的容許差）。seam 沒過只會讓備援不用（照原登記退回），不會交出錯的檔。offgrid 只是用來看那種 GPU 在不同批次組合下的雜訊有沒有大到不合理；× 1.0 在容許差內就夠了。
3. **不變的**：所有容許差（在釘住的 `t07_rank.py`、`t08_rank.py`、`t09_rank.py`、`official_log.py` 裡）；套件（`colab_gpu/gpu_fallback_core.py` 和釘住檔，kit_id 不變，已做的演練仍有效）；觸發條件、一段一次嘗試、准用紀錄要帶目前的 kit_id、要在 11/04 11:00 前 commit、絆線、測試日的全部判定。
4. **紀錄的版本**：`rehearsal-compare` 寫的紀錄帶 `rule_version: 2`（整份和每一段的判定都有）。`allowed_environments` 只認 `rule_version` 2 的紀錄；舊規則產的紀錄（沒有 `rule_version` 或是 1）就算數字會過也不算，要用新版工具重跑 `rehearsal-compare` 重產（同一份 Colab 結果即可，不用重跑 Colab）。
5. **程式和測試**：`tools/g11_gpu_fallback.py`（`RULE_VERSION = 2`、`stage_verdict`、`record_allows`、紀錄的 `rule_version`）；`tests/test_g11_tool.py`（× 1.0 的判定、A100 的數字會過、2.5 倍只記錄、rule_version 1 或沒有版本的紀錄不算）。
6. **取代**：第 3 節第 3 點的條件、第 3 點表格的「2.5 × 實測」一欄（改成只記錄），以及第 3 節第 4 點「2.5 × 實測 ≤ 現值時容許差就是現值；超過時那一種 GPU 不准用」由本節第 1 點取代。不改原文。
