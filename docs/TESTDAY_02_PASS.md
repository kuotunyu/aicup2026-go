# 測試日：T1.4 只因「結尾虛手」而 FAIL_OPENING 時（PREREG_02 rev 9）

- 登記：`docs/PREREG_02_REV9_TRAILING_PASS.md`（rev 9；rev 10 是實作與演練的補充）。和 `docs/TESTDAY.md` 不一致時，FAIL_OPENING 的處理以本頁和 rev 9 為準，其他照 `docs/TESTDAY.md`。
- `PY` = `.venv\Scripts\python.exe`（Git Bash 寫 `.venv/Scripts/python.exe`），在 repo 根目錄執行；作者 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，不加 Co-Authored-By。
- 什麼時候看本頁：`docs/TESTDAY.md` 第 1 步的 `t1` 有任一場回 2（FAIL_OPENING）。**第 2 步 `build` 之前**做完本頁。

## 0. 先確認還在第 1 步

- `data/test/MANIFEST.json`、`results/test02/checks/built/` 都**不存在**。已經存在（已經用 FAIL_OPENING 的結果 build 或往下做了）：**停下回報總部**，不要自己倒回去（rev 9 第 4 節第 5 條）。
- 兩場的 `t1` 都跑完了（完整 T1，不帶 `--only`），兩份 `results/test02/checks/{rank,player}/t1.json` 和 `summary.json` 都已 commit（FAIL_OPENING 也照實 commit，commit 訊息寫 verdict）。

## 1. 判定

```
PY tools/trailing_pass.py prepare
```

- 結束碼 0：至少一場**套用**（apply）。往下做第 2 步。
- 結束碼 2：沒有一場套用（not_needed／not_applicable）。照原規則：`docs/TESTDAY.md` 第 1 步表格的 FAIL_OPENING 那一列（`build --allow-fail-opening --reason ...`、競賽二 `--fallback R10`、競賽一 06／10 不用）。`results/test02/checks/trailing_pass/prepare.json` 和 `raw_t1_*.json` 照樣 commit（訊息寫「rev 9 not applicable: <理由>」）。
- 結束碼 1：拒絕，照訊息處理，不繞過。
- 印出的每場一行：原始 T1 verdict → 判定（理由）；副本預算的 T1 verdict。之後每個有結尾虛手的檔一行：盤數、拿掉虛手的盤數和虛手節點數、各類計數（`pass_not_trailing`＝中盤虛手或帶其他屬性，`pass_with_other_reason`＝虛手加其他 T1.4 原因，這兩類都不改）。這幾行貼給總部。
- 寫出的東西：一律寫 `results/test02/checks/trailing_pass/{prepare.json, raw_t1_rank.json, raw_t1_player.json}`；結束碼 0 時另外寫 `data/test_pass/original/`（AIdea 原始 6 個檔＋`SOURCE.json`，唯讀，之後不動）和 `data/test_pass/stripped/`（只放 6 個官方檔名；不套用那場的檔逐位元組照抄）。`data/test/raw` 這時還沒動。

commit：`results/test02/checks/trailing_pass/`（訊息寫哪一場套用、拿掉幾盤）。

## 2. 把副本放進 data/test/raw，重跑兩場 T1

```
PY tools/t02_test_bundle.py ingest --src data/test_pass/stripped --rerun --reason "PREREG_02 rev 9: trailing passes removed (results/test02/checks/trailing_pass/prepare.json)"
PY tools/t02_checks.py t1 --comp rank --rerun --reason "PREREG_02 rev 9: trailing passes removed (results/test02/checks/trailing_pass/prepare.json)"
PY tools/t02_checks.py t1 --comp player --rerun --reason "PREREG_02 rev 9: trailing passes removed (results/test02/checks/trailing_pass/prepare.json)"
```

- `ingest` 是沒改過的工具：只覆寫 `data/test/raw` 的 6 個檔和 `SOURCE.json`（`SOURCE.json` 會記來源是 `data/test_pass/stripped`）。
- **兩場都重跑** `t1`，不帶 `--only`。套用那場要回 0（PASS 或 NORMALIZED），而且要等於 `prepare` 預算的 verdict；另一場的 verdict 不變（它的檔逐位元組相同）。
- `t1` 拒絕執行（例如 real mode 的驗證棋譜防護：一個檔有一半以上是本機驗證棋譜；`prepare` 的預算不讀 `data/local`，所以算不到這一項）或 verdict 和預算不同：**停下回報總部**，不要再改檔。

commit：`results/test02/checks/{rank,player}/`（`t1.json`、`summary.json`）。

## 3. 核對

```
PY tools/trailing_pass.py verify
```

- 核對：`data/test/raw`＝副本、`SOURCE.json` 只指向副本、原始檔沒變、兩份 `t1.json` 記的 SHA-256＝副本、套用那場的 verdict＝預算、另一場不變。`PASSED` 才往下。
- `NOT PASSED`：照列出的問題處理（多半是漏了第 2 步某一行或沒 commit）；處理完用 `verify --rerun --reason "..."` 再跑。看不懂就停下回報總部。

commit：`results/test02/checks/trailing_pass/verify.json`。

## 4. 之後

- **`verify` 是 PASSED 才能 build**：下游工具不知道 rev 9（凍結檔不改），這個順序只靠本頁。
- 照 `docs/TESTDAY.md` 第 2 步起原樣走：`build --kind official`（**不加** `--allow-fail-opening`；套用的那場 T1 已經是 PASS／NORMALIZED）。之後所有工具（Colab 測試包、T2–T5、04／06／07／08／09／10／14、`official_log.py`）都照原本的方式讀 `data/test/raw`（副本）和 `data/test`，不需要任何額外旗標。
- 只有一場套用時，另一場仍照它自己的 T1 verdict 處理（例如另一場還是 FAIL_OPENING：那一場照原規則走 R10 或 06／10 不用；`build` 要加 `--allow-fail-opening --reason`，理由寫是哪一場）。
- RESULTS 和報告要寫：哪一場套用、每個檔拿掉幾盤／幾個虛手節點（`prepare.json`）、原始檔與副本的 SHA-256 對應（`verify.json` 的 `mapping`）。
- `data/test_pass/` 和 `data/test/` 一樣留到 11/25 以後。

## 彩排（11/04 前，scratch；不 commit 彩排的輸出）

`--root S` 的樹（S 在 repo 外）：`ingest --src <假檔資料夾> --root S`、`t1 --dry-run --tests-dir S/data/test/raw --out S/results/test02/checks`、`trailing_pass.py prepare --root S`、`verify --root S`。只能用訓練玩家做的假檔（`t02_fake_tests.py from-train`），不能用驗證衍生的樹跑 T1.4。10/17 的演練結果記在 rev 10。
