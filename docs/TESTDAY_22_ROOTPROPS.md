# 測試日：官方測試檔開頭有 HD／CA／RE 等欄位時（PREREG_02 rev 11）

- 登記：`docs/PREREG_22_ROOT_PROPS.md`。和 `docs/TESTDAY.md` 不一致時，開頭欄位的處理以本頁為準，其他照 `docs/TESTDAY.md`。
- 為什麼一定要做：KataGo 那條鏈（07 HS、08／09／19 的 P1、競賽一 B′）讀棋譜只接受 `(;SZ[19];B[..];W[..]…)`；開頭多了 `HD[n]`，每一盤都會讀取失敗，那幾層會照登記規則「不採用」、悄悄退回保底（結束碼照樣是 0）。其他工具忽略開頭欄位。拿掉之後，每個工具看到的格式和訓練 v1.0 一樣。
- `PY` = `.venv\Scripts\python.exe`（Git Bash 寫 `.venv/Scripts/python.exe`），在 repo 根目錄執行；作者 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，不加 Co-Authored-By。
- 什麼時候做：`docs/TESTDAY.md` 第 1 步兩場的 `t1` 都跑完並 commit 之後、第 2 步 `build` **之前**。T1 報告的 `other_properties_record_only` 出現 HD／CA／RE 等就要做；沒出現時跑一次 `prepare` 也無妨（回 2，什麼都不改）。

## 0. 先確認

- `data/test/MANIFEST.json`、`results/test02/checks/built/` 都**不存在**。已經存在：**停下回報總部**。
- 任一場 `t1` 是 FAIL_OPENING（`t1` 的結束碼 2）：**先**照 `docs/TESTDAY_02_PASS.md` 做完 rev 9（`trailing_pass.py prepare`；套用時一路做到 `verify` PASSED 並 commit；不套用時 commit `prepare.json`），再回來本頁。工具會擋錯的順序，也會擋過期的 rev 9 紀錄。rev 9 的 `prepare` 本身拒絕（結束碼 1）時：停下回報總部。

## 1. 判定

```
PY tools/root_props.py prepare
```

- 結束碼 0：至少一場**套用**。往下做第 2 步。
- 結束碼 2：不需要（沒有要拿掉的開頭欄位），什麼都沒改。照 `docs/TESTDAY.md` 第 2 步走。`results/test02/checks/root_props/` 照樣 commit。
- 結束碼 3（印 `STOP`）：至少一場不適用（T1 是 FAIL_STRUCT，或副本的 T1 和原本不同）。**不要 ingest**，commit 紀錄後停下回報總部（另一場即使已寫出副本也先不用）。
- 結束碼 1：拒絕，照訊息處理，不繞過。
- 印出的每檔一行：盤數、改了幾盤、各類計數（`unusual_cell`／`root_unparsed`／`parse_error`／`check_mismatch` 這幾類**沒改**）、拿掉的屬性和次數、HD 分布、`plain form a -> b`（符合 KataGo 純格式的盤數，改前 → 改後）。**改後的 plain form 應該等於盤數**；不等於、或有沒改的盤：把這幾行貼給總部（那些盤 KataGo 仍讀不了，是否影響 07／08／B′ 的失敗率門檻要看數量）。
- 寫出：`results/test02/checks/root_props/{prepare.json, pre_t1_rank.json, pre_t1_player.json}`；結束碼 0 時另外 `data/test_rootprops/original/`（目前 `data/test/raw` 的 6 個檔＋`SOURCE.json`，唯讀）和 `data/test_rootprops/normalised/`（6 個官方檔名）。

commit：`results/test02/checks/root_props/`（訊息寫哪一場套用、改了幾盤、拿掉哪些屬性）。

## 2. 把副本放進 data/test/raw，重跑兩場 T1

```
PY tools/t02_test_bundle.py ingest --src data/test_rootprops/normalised --rerun --reason "PREREG_02 rev 11: root properties removed (results/test02/checks/root_props/prepare.json)"
PY tools/t02_checks.py t1 --comp rank --rerun --reason "PREREG_02 rev 11: root properties removed (results/test02/checks/root_props/prepare.json)"
PY tools/t02_checks.py t1 --comp player --rerun --reason "PREREG_02 rev 11: root properties removed (results/test02/checks/root_props/prepare.json)"
```

- 兩場都重跑、不帶 `--only`。verdict 要和改前**相同**（rev 11 不改任何 T1 判定）；不同或 `t1` 拒絕：停下回報總部。

commit：`results/test02/checks/{rank,player}/`。

## 3. 核對

```
PY tools/root_props.py verify
```

- `PASSED` 才往下；`NOT PASSED` 照列出的問題處理（多半是漏了第 2 步某一行或沒 commit），處理完 `verify --rerun --reason "..."`。

commit：`results/test02/checks/root_props/verify.json`。

## 4. 之後

- 照 `docs/TESTDAY.md` 第 2 步起原樣走（`build` 的 `--allow-fail-opening` 照 T1 verdict 決定，rev 11 不影響）。之後所有工具都讀 `data/test/raw`（副本）和 `data/test`，不需要額外旗標。
- rev 9 也套用時：rev 9 的 `verify` 是在 rev 11 之前跑的；rev 11 之後 `data/test/raw` 不再等於 rev 9 的副本，**不要再重跑 `trailing_pass.py verify`**。對應關係：AIdea 原檔 `data/test_pass/original/` → rev 9 副本 `data/test_pass/stripped/`（＝`data/test_rootprops/original/`）→ rev 11 副本 `data/test_rootprops/normalised/`。
- RESULTS 和報告要寫：哪一場套用、每檔改了幾盤、拿掉哪些屬性、HD 分布（`prepare.json`），原檔與副本 SHA-256 對應（`verify.json` 的 `mapping`）；HD 是官方給的資訊，我們不用（模型用 v1.0 訓練）。
- `data/test_rootprops/` 和 `data/test/` 一樣留到 11/25 以後。

## 彩排（scratch；不 commit 輸出）

`--root S`（S 在 repo 外，路徑不要含 `AppData\Local`，用 subst 磁碟）：`ingest --src <假檔> --root S`、`t1 --dry-run --tests-dir S/data/test/raw --out S/results/test02/checks`、`root_props.py prepare --root S`、`verify --root S`。11/04 的演練：`results/root_props/rehearse.py`、`results/root_props/rehearsal_summary.json`、`results/root_props/audit.md`。
