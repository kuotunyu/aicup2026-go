# 測試日操作手冊：競賽二 07／08 的 Colab GPU 備援（PREREG_11）

- 這份是 Claude 的指令順序。只在 `docs/PREREG_11_GPU_FALLBACK.md` 第 2 節的條件成立時用；沒觸發就完全照 `docs/TESTDAY_07.md`、`TESTDAY_08.md`。
- 使用者的 Colab 步驟在 `docs/COLAB_GPU_FALLBACK.md`。給使用者的訊息一次一段、不可逆的步驟（上傳、Drive 刪除或改名）一則訊息只給一件（AGENTS.md）。
- `PY`＝`.venv/Scripts/python.exe`；repo 根目錄、Git Bash、指令前加 `PYTHONDONTWRITEBYTECODE=1`；不設 OMP／OPENBLAS／MKL 執行緒變數。`G`＝`tools/g11_gpu_fallback.py`。
- 結束碼：0 完成；2 拒絕（照訊息修好前提）；3 登記的「不用」（由 07／08 的前提判定，會寫 `not_used.json`；commit，照原手冊退回）；4 環境不同或對不上（停下查）。
- 只 `git add results/test11 results/test07 results/test08`（看是哪一段），不要 `git add -A`。`data/test11/**` 不進 git。

## A. 演練（10 月，任何測試資料之前；只做一次，每多一種 GPU 多做一次）

1. 閘門（AGENTS.md 的十五個檔）＋ `PY -m pytest -q tests/test_g11_core.py tests/test_g11_tool.py tests/test_g11_hygiene.py`。
2. `PY $G kit` → `data/test11/kit/C10_gpu_kit_v1.zip`（約 330 MB）、`results/test11/kit_manifest.json`。`PY colab_gpu/build_gpu_fallback_nb.py --upload-copy` → `colab_gpu/gpu_fallback.ipynb` 和上傳副本。commit。
   - 程式（`colab_gpu/gpu_fallback_core.py` 或任何釘住檔）有改，就要重建套件和 notebook，舊的演練紀錄作廢。
3. `PY $G input --mode rehearsal` → `data/test11/upload/C10_gpu_input_rehearsal.zip`、`results/test11/inputs/rehearsal.json`。commit。
   - 這一步讀驗證題的**題目檔**（SHA 核對），不讀答案。
4. 請使用者照 `docs/COLAB_GPU_FALLBACK.md` 的「演練」做：A100、H100，拿得到再加 G4，一種接一種（每種 GPU 的輸出在雲端硬碟 `out/rehearsal_<型號>/`，結果 zip 是 `C10_gpu_results_rehearsal_<型號>.zip`）。每種約 40–60 分鐘（【待演練】）。
5. 每一份結果 zip 解到自己的 `data/test11/colab_results/rehearsal_<型號>/`。
6. `PY $G rehearsal-compare --from data/test11/colab_results/rehearsal_<GPU> --work-dir <scratchpad>/g11_reh`
   → `results/test11/rehearsal_<GPU 名>.json`。commit。看 `verdict`：
   - `hs` / `p1` 各自 `allowed: true` 才准用（PREREG_11 rev 2：seam 照釘住的容許差通過、offgrid 每個差 × 1.0 ≤ 容許差；2.5 倍只記錄；紀錄要有 `rule_version: 2`，舊版紀錄要用新工具重跑 `rehearsal-compare` 重產）。不准用就照實記錄，**不改容許差**。
   - 工具只收真正的 Colab 演練：每塊的 `tool_sha256` 等於套件 core 的 SHA、`seconds` > 0、真權重；fake-from-val／fake scorer／`--max-games` 的結果一律拒絕。
   - **每一份演練紀錄都要在 11/04 11:00 以前 commit**（之後 commit 或修改的不算，工具核對 commit 時間）。
   - 把實測數字填進 PREREG_11 第 3 節的表（只填【待演練】的格），commit。
7. 演練用過的 Colab 輸出（雲端硬碟 `C10/gpu_fallback/out/rehearsal_<GPU>/`，每種 GPU 一個）請使用者留著，不要刪，到 11/25 以後再清。

## B. 觸發（測試日，4090 出事時）

1. 確認是第 2 節的條件，不是可以照 TESTDAY_07／08 修好的 2：
   - `nvidia-smi` 失敗或看不到 4090，重開機後一樣 → `hardware_failure`；
   - `hs`／`p1` 結束碼 2，訊息是 CUDA、GPU 型號、權重 SHA（備份也不對）→ 照原手冊處理一次仍一樣 → `gpu_exit2`；
   - 結束碼 4 `venv_gpu`，重裝釘住版本仍不符 → `venv_gpu_unrestorable`；
   - 11/14 23:59:59 之後那一段在 4090 還沒三組都成功 → `deadline_11_14`。
2. 先告訴使用者發生什麼事、要改用 Colab（白話一段），然後：
   `PY $G trigger --stage hs --reason <理由> --evidence "<哪一行指令、結束碼、訊息、時間>"` → commit `results/test11/trigger_hs.json`（08 用 `--stage p1`）。
3. 4090 已經寫出的部分塊：把 `data/test07/hs/<組>/` 改名成 `<組>_4090_<時間>`（不刪）。4090 已 commit 的某組結果（`results/test07/hs/<組>_hs.npz`、`hs_<組>_report.json`）：`git mv` 到 `results/test11/superseded/hs/`，commit。08 同理（`data/test08/p1/`、`results/test08/p1/`）。

## C. 07 的 HS 改在 Colab

1. 前提和 4090 路徑相同：R20 的第一次上傳已記成 uploaded 並 commit、T1 PASS／NORMALIZED、git 乾淨。
2. `PY $G input --mode real --stage hs` → `data/test11/upload/C10_gpu_input_hs_test.zip`、`results/test11/inputs/hs_test.json`。commit。輸入檔帶著准用的環境名單（演練紀錄裡 allowed 的 GPU＋torch＋cuDNN＋Python＋numpy）；Colab 在寫任何東西之前核對，不符就停（不算嘗試）。告訴使用者要選哪一種 GPU（准用名單裡的）。
   - 結束碼 3：07 照登記不用（工具已寫 `results/test07/not_used.json`），commit，照 TESTDAY_07 的「07 不用」處理。
3. 請使用者照 `docs/COLAB_GPU_FALLBACK.md` 的「測試日」做（先把雲端硬碟裡舊的輸入檔移走，上傳新的輸入檔；GPU 選演練准用的型號）。約 3–4 小時（【待演練】）。
4. 結果 zip 解到 `data/test11/colab_results/hs_test/`。第一次 `import` 會寫 `results/test11/attempt_hs.json`（就算這次 import 被拒絕也會寫）：**馬上 commit**，之後別的 Colab run 一律拒絕。
5. 依序匯入，每組做完就 commit：
   ```
   PY $G import --stage hs --set seam --from data/test11/colab_results/hs_test
   git add results/test07/hs results/test11 && git commit -m "test11: hs seam imported (Colab fallback)"
   PY $G import --stage hs --set public --from data/test11/colab_results/hs_test      # 再 commit
   PY $G import --stage hs --set private --from data/test11/colab_results/hs_test     # 再 commit
   ```
   - 拒絕（2）的常見原因：觸發紀錄沒 commit、輸入紀錄沒 commit、環境不在准用名單、`data/test07/hs/<組>/` 還有 4090 的塊、某一塊早於觸發或 R20 上傳、塊和 Colab 的總表不符（重新下載）。
   - 4：環境和 seam 不同、塊的環境不同 → 停下查，不重跑 Colab（只准一次嘗試）。
6. 接著完全照 `docs/TESTDAY_07.md` 第 4 步起（`t07_rank.py apply`）。apply 的 GPU 接縫就是演練比過的那一個；沒過（4）照 TESTDAY_07 處理，07 這一季不用。

## D. 08 的 P1 改在 Colab

1. 前提：07 的 `apply` 結束碼 0 並 commit（07 判不用時 08 也不用）。
2. `PY $G input --mode real --stage p1` → `C10_gpu_input_p1_test.zip`、`results/test11/inputs/p1_test.json`。commit。
3. 使用者照 `docs/COLAB_GPU_FALLBACK.md` 的「測試日」做。約 1 小時（【待演練】）。
4. 結果 zip 解到 `data/test11/colab_results/p1_test/`，`import --stage p1 --set seam|public|private`，每組 commit。
5. 接著照 `docs/TESTDAY_08.md` 的 `apply` 起，再照 `docs/TESTDAY_09.md`。

## E. 檢查和收尾

- 每一步之後 `PY $G status`。
- 用到備援時，在 `results/prereg07/RESULTS.md`……**不改**（凍結）；改在報告草稿和 `results/test11/` 的紀錄裡寫揭露（PREREG_11 第 7 節）。
- `data/test11/` 到 11/25 以後都要留著。
