# 測試日操作手冊：競賽一 10（test10，06 之後）

- 這份是 Claude 的指令順序，只管競賽一的 10：**全黑的池子落在第 3 層（ρ < θ₂）時，改交 06 的第 1 層計算（10A），不交「只用 CNN」**（PREREG_10 第 1 節，A）。B（P1 的 LDA 項）只有在 H10P 或 H13P 採用之後才會動（`results/prereg10/RESULTS.md` 有 `H10P：B **採用**` 那一行，PREREG_10 rev 4 第 13 節；或 `results/prereg13/RESULTS.md` 有 `H13P：B′ **採用**` 那一行，PREREG_10 rev 5 第 16 節）。**2026-10-13 起 repo 的 B 由 PREREG_13 採用**（H10P 沒跑；`t10_player.py status` 印 `B: adopted via PREREG_13`）；沒採用時 `t10_player.py p1` 一律回結束碼 3「B 沒有採用」、不寫檔，10 只有 A，下面標「B」的步驟全部跳過。
- 依據：`docs/PREREG_10_PLAYER_GUARD_P1.md` 第 1、2、4 節、rev 3（第 12 節，A 的測試日修訂）、rev 4（第 13 節，B 的測試日修訂）和 rev 5（第 16 節，B 採用也可以由 PREREG_13 判定）。不一致時以登記為準。test02 照 `docs/TESTDAY.md`，06 照 `docs/TESTDAY_06.md`，那兩份不改。10 不用 Colab，使用者不必多跑 notebook；只有 B 採用時，`p1`（和 11/04 前的演練）用本機 4090。
- **競賽一的最終選擇**：`docs/TESTDAY_06.md` 第 9 步（PREREG_06 5.4）在 log 裡有 10 的紀錄時由 PREREG_10 第 4.4 節和本檔第 6 步取代：06-ON 判定 `kept` → 06-ON（不傳 10）；否則有符合條件的 10 → 10；都沒有 → 照 PREREG_06 5.4（06-OFF → Q04）。`PY tools/official_log.py status` 會印出規則指定哪一份。
- `PY` = `.venv/Scripts/python.exe`（Git Bash），一律在 repo 根目錄 `D:\AI-Portfolio\competitions\aicup2026-go` 執行，前面加 `PYTHONDONTWRITEBYTECODE=1`。
- **每一步做完就 commit，不論結束碼**：`git add results/test10 results/official_scores.json`，**不要 `git add -A`**。作者 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，不加 Co-Authored-By。10 在正式模式要求 git 乾淨（06 或競賽二的輸出沒 commit 時是結束碼 2：先 commit 那一步）。
- **結束碼 3 之後不重跑**。唯一的例外：test02 競賽一或 06 照各自手冊重跑過 → 10 從第 2 步（seam）重做，三步都加 `--rerun --reason`（PREREG_10 4.2）。除此之外要翻盤，先寫並 commit PREREG_10 的修訂（rev 4 起）。

## 每一步都適用的規則

- 不開、不讀、不雜湊 `data/local/` 裡的驗證標籤檔；`t10_player.py seam` 只經由 06 的程式讀 `data/local` 的兩份棋譜檔，工具自己擋其他路徑。不跑 `tools/score.py`。GPU 只給 B 的 `p1`／`p1-rehearse` 用（`.venv-gpu`），開跑前 `nvidia-smi`。
- 凍結檔一律不改：`tools/t0*`、`tools/p0[1-9]*`、`colab/**`、`docs/TESTDAY*.md`（本檔以外）、`docs/COLAB_RUN.md`、`results/prereg01`–`prereg09/**`、`results/test0x/**` 的 design 紀錄、`third_party/**`、`results/prereg10/explore/**`。10 只寫 `results/test10/player/`，不寫 `results/test02/`、`results/test06/`。`tools/official_log.py`、`.gitattributes`、`.gitignore`、`docs/PREREG_02／06／10` 只准加行。
- 每次開工先跑防誤改閘門（`AGENTS.md` 的 19 個檔），唯一允許的失敗同 `AGENTS.md`；再跑 `PY -m pytest -q tests/test_t10_official_log.py -k TagVersusNew`。
- 上傳：每一份檔、每一次上傳，都要使用者在對話裡對那一份說 OK。**訊息 A** 放檢查結果和 OK 請求；使用者說 OK 之後，**訊息 B** 自成一則、只放上傳步驟。同一時間只能有一則等 OK 的訊息，不分競賽。
- 四份交件檔同名（`submission_player.csv`）：Q04 在 `results\test02\player\`，06-OFF 在 `results\test06\player\off\`，06-ON 在 `results\test06\player\on\`，**10 在 `results\test10\player\`**。訊息 B 只給一個資料夾，寫明 SHA-256 前 12 碼和「不要從另外三個資料夾拖」；發之前先核對：

  ```powershell
  Get-FileHash -Algorithm SHA256 "results\test10\player\submission_player.csv" | Format-List Hash, Path
  ```

## `t10_player.py` 的結束碼

| 結束碼 | 意思 | 怎麼做 |
|---|---|---|
| 0 | 完成 | commit，做下一步 |
| 2 | 拒絕：前提不符（test02 或 06 沒做完、沒 commit 或在 10 之後變了；06 的 `t06_pool.json` 不是用現在的 test02 報告和 06 seam 做的；T4 沒跑或和 test02 的報告對不上；test02 報告的防護層級用凍結門檻重算對不上；git 不乾淨；輸出已存在；這一次 10 已在上傳紀錄裡還要 `--rerun`；K6-10 前已 commit 過 10 而 K6-10 已過） | 照訊息修好前提，再跑同一個指令。「unexpected internal error」（有 traceback）是工具本身出錯：停下查，不是登記的判定。重算對不上：停下查 test02 的報告，不自己改報告 |
| 3 | 登記的「不用」：T1 FAIL_OPENING、H6P 沒採用、06 的 seam 因 dumps 不同沒過、K6-10 已過而且之前沒 commit 過 10、K5-10 不成立、**10 不需要**（兩池都和這一季的 06-OFF 或 Q04 逐列相同）、B 沒有採用（`p1`） | commit `results/test10/player/not_used.json`；最終照 PREREG_06 5.4（第 6 步）；不重跑 |
| 4 | 本機漂移：10 的 seam 重算 06 的第 1 層計算和 06 的 seam 不同（B 採用時：或重算的驗證集 10B 和 `results/prereg10/player/player_val_top5.csv` 不同） | commit `t10_seam.json`；停下查本機，查好後 `--rerun --reason "<查到的原因>"`；要放寬只能寫 PREREG_10 修訂 |
| 5 | 只有 B 的 `p1`／`p1-rehearse`：STOP 檔讓出 GPU，做完手上那一塊就停（沒有判定） | 刪掉 STOP，同一行重跑就接續（不要加 `--rerun`） |

- **B 的任何問題都只讓 B 不用，不擋 A**（PREREG_10 rev 4 第 14 節）：B 沒採用、B 的檔對不上、K3-10／K3-13／K5-10（B）／K5-13／K6B-10 不成立、11/04 11:00 之後還沒有演練紀錄、GPU 接縫沒過、seam 重算的驗證集 10B 對不上、某一池的 P1 有問題——`p1` 回 3、seam／apply／submit 照常（那些池不加 B）。沒採用時不寫檔，其他記在 `results/test10/p1/b_not_used.json`（commit），10 照 A 繼續，不重跑。B 的問題不會讓 seam、apply、submit 回 2 或 4。

## 日期

- 11/04 只做 test02。11/05 起：06（`docs/TESTDAY_06.md`）做完、06-OFF（有的話）上傳之後才做 10；10 在 CPU 上約 1 分鐘。
- **K6-10 11/21（六）23:59:59**：seam、兩池 apply、submit 都以結束碼 0（或 submit 的 3「10 不需要」）跑完並 commit。過了以後 10 的步驟不再執行。
- **K5-10**（10/31 23:59:59，合併前就定了）：PREREG_10 rev 3 那一列在 10/31 前 commit。`official_log.py` 每次 prepare／final 都會查。
- 11/22、11/23 各跑一次 `PY tools/official_log.py status`。最後一次上傳在 11/23 23:59 前（截止 11/24 23:59）。競賽一每天最多 5 次（Q04、06、10 合計）。

## 第 1 步：閘門

全部核對，任何一項不對就停下：

```
PY tools/official_log.py status
PY tools/t06_player.py status
PY tools/t10_player.py status
git status --porcelain
PY -c "import json; [print(p, {k: json.load(open(f'results/test02/player/{p}/apply_report.json', encoding='utf-8'))[k] for k in ('guard_level', 'guard_reason', 'gate_passed', 'b_c_candidate_black_share', 'b_q_question_black_share', 'rho')}) for p in ('public', 'private')]"
PY -c "import json; print(json.load(open('results/test02/checks/player/t4.json', encoding='utf-8'))['pools'])"
```

- Q04 的第一次上傳已記成 `uploaded` 並 commit；06 的 `feats`（三池）、`seam`、兩池 `apply` 都已 commit（第二池的結束碼 3「兩池都退回 test02」也算跑完）；06 的 `submit` 已跑（結束碼 0 或 3），有 06-OFF 時 06-OFF 已上傳；06-ON 有產生時，否決已判定（`veto.json` 已 commit）。
- `results/test02/checks/player/t4.json` 存在並已 commit（`docs/TESTDAY.md` 第 3 步的 T4）。
- 兩池的 `guard_level` 都不是 3、或第 3 層但 `b_c`／`b_q` 不是剛好 1：A 不會成立，10 一定是「10 不需要」（結束碼 3）。照樣跑完第 2–4 步，留下紀錄。

## 第 2 步：seam

```
PYTHONDONTWRITEBYTECODE=1 PY tools/t10_player.py seam
git add results/test10 && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "test10: seam"
```

- 看 `results/test10/player/seam_val/t10_seam.json`：`passed` 為真，`a_rule.holds` 為假（驗證池在第 1 層、黑白混合），`criteria` 全真。結束碼 4：照上表。

## 第 2B 步：B 的 P1（只在 B 採用時；4090，約 5.5 小時，排夜間）

- 前提：**11/04 11:00 前**已跑過演練、`results/test10/p1/p1_rehearsal.json` 已 commit（見本檔最後「B 的演練」）；11/04 11:00 之後還沒有這份紀錄，`p1` 回 3（B 不用，記在 `b_not_used.json`），10 只有 A。
- 順序：競賽二的 07 HS（11/04 夜）和 08 P1（11/05）先跑（PREREG_10 4.3）；10 的第 2 步 seam 之後再跑這一步。**K6B-10 11/14（六）23:59:59** 前兩池的 P1 和 GPU 接縫都要 commit，過了就沒有 B。
- 每一行開跑前 `nvidia-smi` 看沒有別的計算程序（工具也會查）。長的那兩行用 git 裡的啟動器 `tools/t10_p1_launch.sh`（獨立背景程序，會等 GPU 空，紀錄 `data/test10/p1/launch_<池>.log`），不要綁在對話上：

```powershell
Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/d/AI-Portfolio/competitions/aicup2026-go/tools/t10_p1_launch.sh public" -WindowStyle Hidden
```

  （`private` 同一行換池名；`seam_val` 很短，可以直接跑下面的指令。）跑完看 log 最後的 `rc=`：不論 rc 都先 `git add results/test10` 再 commit（0：報告；3：B 不用，`b_not_used.json` 和沒過的 `gpu_seam.json`），5 是 STOP 讓出（刪 STOP 再啟動一次就接續），2 讀 log。B 沒用到的檔（例如沒過的 `gpu_seam.json`）不會綁進 10 的 `inputs_bound`，不會擋 A 的上傳。

```
nvidia-smi
PYTHONDONTWRITEBYTECODE=1 .venv-gpu/Scripts/python.exe tools/t10_player.py p1 --pool seam_val
git add results/test10 && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "test10: B GPU seam"
PYTHONDONTWRITEBYTECODE=1 .venv-gpu/Scripts/python.exe tools/t10_player.py p1 --pool public
git add results/test10 && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "test10: B P1 public"
PYTHONDONTWRITEBYTECODE=1 .venv-gpu/Scripts/python.exe tools/t10_player.py p1 --pool private
git add results/test10 && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "test10: B P1 private"
```

- `seam_val`（約 2 分鐘）：驗證池前 400 盤重抽，`results/test10/p1/gpu_seam.json` 的 `passed` 為真（投影後最大差 ≤ tol_P、失敗盤和手數相同）。沒過：結束碼 3，B 這一季不用，commit `gpu_seam.json` 和 `b_not_used.json`，直接做第 3 步（A）。
- `public`／`private`（各約 2.7 小時）：塊在 `data/test10/p1/<池>/`（不進 git，**到 11/25 以後都要留著**），報告 `results/test10/p1/<池>_p1_report.json`。要讓出 GPU（例如 C4 要用）：建 `data/test10/p1/<池>/STOP`，結束碼 5；之後刪掉 STOP、同一行重跑就接續。
- 已有的塊和這一次不同（例如換了 `.venv-gpu`）是結束碼 2，工具不會覆寫：停下查。

## 第 3 步：apply（兩池）

```
PYTHONDONTWRITEBYTECODE=1 PY tools/t10_player.py apply --pool public
PYTHONDONTWRITEBYTECODE=1 PY tools/t10_player.py apply --pool private
git add results/test10 && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "test10: apply public / private"
```

- 每池的 `t10_pool.json`：`source` 是 `10A`（A 成立、06 閘門對 R02′ 通過）、`06-OFF`（A 不成立、06 用了）或 `test02`（A 不成立而 06 退回，或 A 成立但 06 閘門沒過）。`a_rule.conditions` 記 A 的四個條件各自的值；`gate06_vs_r02prime` 記 sr(選中) − sr(R02′)，R02′ 是 test02 的 `effective`（不是 `applied`）。
- A 成立的池不開一對一（PREREG_10 1.2），所以有 A 的這一季不會有 06-ON。
- **B 採用時**：來源是 06-OFF 或 10A 的池，工具再算 10B（w*·z(T_P1)）。池閘門 sr(10B) − sr(選中) ≥ 0 而且 P1 失敗盤 ≤ 0.5%：來源改成 `10B`；否則維持原來的來源，`b.pool.reason` 寫理由。那一池的 P1 還沒做：這一池先維持 A（`b.pool.reason`＝`p1_not_done`，結束碼 0），做完第 2B 步後在 K6B-10 前用 `apply --pool <池> --rerun --reason "B P1 done"` 再跑一次才會加 B（要在 10 上傳之前）；K6B-10 已過、GPU 接縫沒過、seam 的驗證集 10B 沒過、或那一池的 P1 有問題：這一池不加 B（照常結束碼 0，記在 `b_not_used.json`）。

## 第 4 步：submit

```
PYTHONDONTWRITEBYTECODE=1 PY tools/t10_player.py submit
git add results/test10 && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "test10: submit"
```

- 結束碼 3「10 不需要」：commit `not_used.json`，跳到第 6 步（最終照 PREREG_06 5.4）。
- 結束碼 0：`results/test10/player/t10_submit_report.json` 記 `sources`、`reference`（06-OFF 或 Q04）、`top1_differences_vs_06_off`、`top1_differences_vs_q04`、`check_submission.passed`、`inputs_bound`。B 採用時另記 `b`（GPU 接縫、tol、`pools_10B`），`inputs_bound` 含 B 的檔；`official_log.py prepare` 對 10B 另查 K5-10（B）、K6B-10、池閘門（PREREG_10 rev 4 第 13 節第 10 條）；B 由 PREREG_13 採用時（`b.source`＝`PREREG_13`），`inputs_bound` 改綁 `results/prereg13/` 的 `RESULTS.md`、`score_H13P.json`、`view_report.json`（不要 `score_H10P.json`），另查 K5-13（PREREG_10 rev 5 第 16 節）。

## 第 5 步：上傳 10

- 順序（PREREG_10 4.4）：Q04（11/04）→ 06-OFF →（06-ON → 否決）→ 10。06-ON 判定 `kept` 時不傳 10（`prepare` 會拒絕）。兩池都退回 test02 時沒有 06 檔，順序是 Q04 → 10。
- `prepare`（訊息 A 之前；`status` 確認當天次數 < 5）：

```
PYTHONDONTWRITEBYTECODE=1 PY tools/official_log.py prepare --comp player --file results/test10/player/submission_player.csv --reason "prereg10 10 (PREREG_10 4.4)"
git add results/official_scores.json && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "official log: prepare 10"
```

- 訊息 A：照 `prepare` 印出的內容（10、各池來源、和 Q04／06-OFF 第 1 名不同的題數、檢查結果），請使用者對這一份說 OK。訊息 B（使用者 OK 之後，自成一則）：AIdea 競賽一（TEAM_11249）→ 上傳 `results\test10\player\submission_player.csv`（SHA-256 前 12 碼）→ 回報上傳時間和 public 分數。
- 記錄：

```
PY tools/official_log.py uploaded --id N --ok-at <OK 時間> --uploaded-at <上傳時間>
PY tools/official_log.py public --id N --score <分數>
git add results/official_scores.json && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "official log: 10 uploaded, public"
```

- `public` 印出「傳錯檔」警告（10 和 Q04／06-OFF 分數完全相同，但 public 第 1 名不同）：請使用者截 AIdea 的上傳紀錄核對，核對前不再上傳、不設最終選擇。10 比上一份低 0.05 以上：只查錯（id 對應、列順序、池配對、seam），查不到就維持第 4.4 節的結果，寫進報告；public 分數不進判定。

## 第 6 步：最終選擇（11/22–11/23）

```
PY tools/official_log.py status
PY tools/official_log.py final --comp player --id N --basis "PREREG_10 4.4"
```

- `status` 的「10（PREREG_10 4.4）」那段寫「最終應該是 …」：照它選；它寫「最後一次上傳不是 …」時，再上傳一次那一份（訊息 A／B 照常），讓它成為最後一次上傳。
- 改選其他的只能用 `--not-10-reason "PREREG_10 rev N: <具體的程式錯誤和它破壞的條件>"`，N ≥ 2，而且那一列要在被推翻的紀錄之後 commit（實際上是 rev 4 起）。

## test02 或 06 重跑了怎麼辦

- 06 照 `docs/TESTDAY_06.md` 從 `feats` 重做之後，10 從第 2 步重做：

```
PYTHONDONTWRITEBYTECODE=1 PY tools/t10_player.py seam --rerun --reason "06 redone after test02 re-run"
PYTHONDONTWRITEBYTECODE=1 PY tools/t10_player.py apply --pool public --rerun --reason "06 redone after test02 re-run"
PYTHONDONTWRITEBYTECODE=1 PY tools/t10_player.py apply --pool private --rerun --reason "06 redone after test02 re-run"
PYTHONDONTWRITEBYTECODE=1 PY tools/t10_player.py submit --rerun --reason "06 redone after test02 re-run"
```

- 已記進上傳紀錄的舊 10 會因綁定的輸入變了而不再符合 4.4（`status` 會列出原因），新的 10 要重新 prepare、OK、上傳。這一次 10 已在上傳紀錄裡而 test02／06 都沒變時，`--rerun` 會被拒絕（結束碼 2）。

## 演練

```
C10_SLOW=1 C10_DRYRUN_DIR=<repo 以外的資料夾> PYTHONDONTWRITEBYTECODE=1 PY -m pytest -q tests/test_t10_dryrun.py
```

- 三棵假樹（官方形狀、nov4 兩池、nov4 只在 private），只用 fit 玩家，約 15 分鐘。結果記在 `results/test10/design/build_review.md`。

## B 的演練（只在 B 採用時；**11/04 11:00 前**，4090 約 2 分鐘）

- 時機：H10P 或 H13P 採用之後。H10P 的路：`results/prereg10/RESULTS.md` 寫好那一行並 commit（兩者都在 K3-10 10/25 23:59 前）、本分支（`p10b-testday`）用 `git merge --no-ff` 合併回 main（K5-10（B）：10/31 23:59 前；看的是 main first-parent 上那個 merge commit 的時間，fast-forward 會讓時間變成分支上的 commit）。H13P 的路（PREREG_10 rev 5，**現在走這條**）：`results/prereg13/RESULTS.md` 的 `H13P：B′ **採用**` 和 `score_H13P.json` 在 K3-13（10/25 23:59）前 commit（10/13 已完成）、分支 `p13-testday` 用 `git merge --no-ff` 合併回 main（K5-13：10/31 23:59 前，看 main first-parent 上第一次出現 `| rev 5 |` 的 commit）；演練要在 K4-13（11/04 11:00）前 commit。演練時 `data/test` 還不能有官方測試檔。量出 m，tol_P＝max(0.05, 2.5 × m)，以 commit 進 git 的演練紀錄為準（PREREG_10 rev 4 第 14 節：不另寫進登記）。
- 背景啟動（會等 GPU 空，紀錄在 `data/test10/p1/launch_rehearse.log`）：

```powershell
Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/d/AI-Portfolio/competitions/aicup2026-go/tools/t10_p1_launch.sh rehearse" -WindowStyle Hidden
```

- 結束碼 0 之後**馬上 commit**（commit 時間要早於 11/04 11:00）：

```
git add results/test10/p1/p1_rehearsal.json && git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "test10: B rehearsal (tol_P)"
```

- B 的假演練（CPU，合成 P1，只用 fit 玩家和驗證池的棋譜檔）：`C10_SLOW=1 ... pytest -q tests/test_t10_dryrun.py -k BTree`，結果記在 `results/test10/design/build_review.md`。
- **B 採用之後會變紅的測試（tripwire）**：`tests/test_t10_player.py::Modes::test_p1_is_b_not_adopted`、`tests/test_t10_b.py::BState::test_real_repository_is_not_adopted_now`、`tests/test_t10_b.py::Cli::test_not_adopted_is_3_and_writes_nothing`（正式模式那幾行）、`tests/test_t10_dryrun.py::Nov4Tree::test_b_gpu_seam_is_b_not_adopted`。它們假設 repo 裡的 B 還沒採用。**PREREG_10 rev 5 已照修訂改寫**：現在檢查「repo 的 B 由 PREREG_13 採用」（`test_t10_player.py::Modes::test_b_adopted_via_prereg13`、`test_t10_b.py::BState::test_real_repository_b_via_prereg13`；另外兩個保留原名，正式模式改用 `status` 讀，不跑正式的 `p1`）。B 的假樹也走一次 PREREG_13 的路（`test_t10_dryrun.py::BTree::test_b_via_prereg13`，`tools/t10_b_fake.py fake-p13`）。A 的假樹（OfficialBaseline、Nov4Tree、MixTree）不帶 `--prereg10-dir`，讀的是不存在的資料夾，所以 B 採用後仍只跑 A。
