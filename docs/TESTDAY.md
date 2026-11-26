# 測試日操作手冊（test02，11/04 起）

- 這份是 Claude 的指令順序。使用者在 Colab 上的步驟在 `docs/COLAB_RUN.md` 最後一節「測試集（test02）」。
- 依據：`docs/PREREG_02_OPENING_FUSION.md` 第 9 節（rev 5，登記）；實作規格 `results/test02/design/spec.md`（3.6、第 6 節）。兩者不一致時以登記為準。
- `PY` = `.venv\Scripts\python.exe`（Git Bash 寫 `.venv/Scripts/python.exe`），一律在 repo 根目錄 `D:\AI-Portfolio\competitions\aicup2026-go` 執行。
- **每個編號段落做完就 commit**，作者 `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，不加 Co-Authored-By。`data/**` 和結果 zip 都在 gitignore 裡不 commit；它們的 SHA-256 記在會 commit 的 JSON（`t1.json`、`bundle_build.json`、`unpack_<comp>.json`、`official_scores.json`）。
- 本機實測（9/26 用 `t02_fake_tests.py from-val` 的假測試集跑的 dry run）：競賽二 `opening` 34 秒、`apply` 1.4 秒；競賽一每池 `run` 30–34 秒。收到兩個結果 zip 後，本機全部做完預計 15 分鐘內。

## 每一步都適用的規則

- 不開、不讀、不雜湊 `data/local/` 裡的驗證標籤檔；不跑 `tools/score.py`；不用 GPU。
- 凍結檔一律不改：`tools/p02_*.py`、`tools/c10data.py`、`tools/decide.py`、`colab/build_prereg02_*.py`、`colab/prereg02_*.ipynb`、`results/prereg01/**`、`results/prereg02/**`。`PY -m pytest -q tests/test_t02_hygiene.py` 會抓到誤改。
- 工具拒絕執行時照訊息處理，不繞過。要改程式或放寬門檻，先寫 PREREG 修訂（下一版是 rev 9；rev 8 已在 `af739a5` 登記，不在它後面補字）。
- 兩隊分開：競賽二的檔案不和競賽一的比；`t02_checks.py` 每次只跑一個 `--comp`；兩份報告分開寫。
- 上傳（第 6 節 U2）：每一份檔案、每一次上傳，都要使用者在對話裡對那一份說 OK，不沿用先前的同意。所有檢查結果放在較早的訊息，**上傳指示自成一則訊息**。

## 11/04 以前要就緒（整合者／總部）

- [ ] T3 虛無分布已算好並 commit：`PY tools/t02_checks.py precompute` → `results/test02/null/{t3_null.json, t3_null_raw.npz, t3_null_canon.npz, train_refs.json}`，數字和登記第 9 節第 8 條相同。
- [ ] 全黑候選池用的 T3b 虛無分布（登記 rev 8）已算好並 commit：`PY tools/t02_checks.py precompute --part t3b` → `results/test02/null/{t3b_null.json, t3b_null_raw.npz, t3b_null_canon.npz}`，數字和 `tools/t02_checks.py` 的 `T3B_PINS` 相同。`--part t3b` 不動上一行那 4 個檔；real mode 的 `run` 要 7 個檔都已 commit 且乾淨。
- [ ] R10 備案已算好並 commit：`PY tools/t02_rank.py fallback-fit --combo R20`（自我核對），再 `--combo R10` → `results/test02/rank/fallback_R10.json`、`fallback_selfcheck_R20.json`。
- [ ] 所有 test02 程式、notebook、產生器都已 commit，`git status` 乾淨（real mode 會檢查）。`PY -m pytest -q tests` 全綠。
- [ ] 選做：10/28–11/02 的 Colab 彩排（本文最後一節）。
- [ ] 做過彩排的話，`data\colab_upload\test02\` 裡是彩排的檔：第 2 步前整個刪掉。

## 0. T0：官方教學 repo 有沒有改（11/04 11:00 後）

```
PY -c "import os; os.makedirs('results/test02/checks', exist_ok=True)"
git clone https://github.com/AILAB-NDHU/AICup-2026-Tutorial "<scratch>\tutorial_1104"
git -C "<scratch>\tutorial_1104" log -1 --format="%H %cI"
git -C "<scratch>\tutorial_1104" diff --output="D:\AI-Portfolio\competitions\aicup2026-go\results\test02\checks\t0_tutorial_diff.txt" 359e06fcaf811f9e87179f12f71425dd18dbabd1 HEAD -- README.md rank-prediction.ipynb player-identification.ipynb
```

- 第一行先建 `results/test02/checks`：這一步在第 1 步的 `t1` 之前，資料夾還不存在，`git diff --output` 不會自己建資料夾（會以 `fatal: could not open … for writing` 失敗）。
- 基準 `359e06f` 記在 `third_party/aicup2026_tutorial/SOURCE.txt`。
- 只比對 `AICup-2026-Tutorial`。AIdea「範例程式」連到的 `AIcupTutorial` 是 2023 年那屆的教學（`docs/DATA.md` 官方資料說明一節），不用。
- 差異檔是空的：官方定義沒變。
- 檔名、欄位或交件格式有變（D5）：只改各程式的檔名表並重建，接縫核對仍要通過（登記第 9 節第 11 條），並寫 PREREG 修訂。檔名表在 `tools/t02_test_bundle.py`（`RANK_FILES`、`PLAYER_Q_FILES`、`PLAYER_C_FILES`）、`tools/check_submission.py`（`OFFICIAL_*`）、`tools/official_log.py`（`QUESTION_FILES`、`CANDIDATE_FILES`、`FILE_NAMES`）、`tools/t02_checks.py`、`tools/t02_rank.py`、`tools/t02_player.py` 和兩個 `colab/build_test02_*_nb.py`。
- commit：`results/test02/checks/t0_tutorial_diff.txt`（commit 訊息寫上 tutorial 的 HEAD）。

## 1. 收檔與 T1

使用者：登入 AIdea，在競賽二（TEAM_11248）和競賽一（TEAM_11249）各自的比賽頁找測試集的下載處（可能在「資料」分頁；以當天畫面為準，找不到就截圖給 Claude，不要猜），下載後告訴 Claude 下載位置。

```
PY tools/t02_test_bundle.py ingest --src "<下載的資料夾或 zip>" [--src "<另一場的資料夾或 zip>"]
PY tools/t02_checks.py t1 --comp rank
PY tools/t02_checks.py t1 --comp player
```

- `ingest` 在資料夾（含裡面的 zip）或 zip 裡找 6 個官方檔名，逐位元組複製到 `data/test/raw/`，寫 `SOURCE.json`。少一個、或同名檔內容不同就拒絕；有相近的檔名會列出來（可能是 D5）。
  - 來源裡有本工具組自己的紀錄也拒絕：測試包的 `MANIFEST.json`（zip 成員或檔案，任何種類）、`SOURCE.json`、`FAKE_TREE.json`。那是彩排／冒煙／假測試集（從本機驗證棋譜做的）或重新打包的 zip，不是 AIdea 的下載。`--src` 只給 AIdea 下載下來的檔或資料夾。
- `t1` 預設讀 `data/test/raw`，寫 `results/test02/checks/<comp>/t1.json`。結束碼：0 = PASS 或 NORMALIZED，2 = FAIL_OPENING，3 = FAIL_STRUCT。
  - 在 `data/test/raw` 一律跑完整的 T1.1–T1.5，**不帶 `--only`**（real mode 在這裡會拒絕 `--only`；彩排一節的 `t1` 指令帶 `--only`，不要照抄過來）。之後的 `build --kind official`、`t02_rank.py`、`t02_player.py`、`official_log.py prepare` 都只收 `complete: true`、`stage: raw`、provenance `mode: real` 的 `t1.json`。
  - 內容檢查（T1.4、T1.5，和第 3 步的 T2–T5）先數每個檔有幾盤是本機驗證棋譜（SGF 的 SHA-1 和驗證題目／候選檔比對，不讀標籤檔）。一個檔有一半以上是驗證棋譜就拒絕：那是驗證衍生的樹跑進了 `data/test`，照 `data/test/raw/SOURCE.json` 查來源，重新 ingest AIdea 的下載。少量命中照常進行（官方檔可能真的含訓練集的棋，T2 就是在抓這個），數字在 `t1.json`（和 `t2`–`t5.json`）的 `provenance.validation_hash_guard`，寫進 RESULTS。

| T1 結果 | 後果（登記第 9 節第 6 條） |
|---|---|
| PASS | 照常 |
| NORMALIZED | 第 2 步只套 `t1.json` 列出的白名單轉換；Colab 和本機都用正規化副本。rev 8 加了一種：`num_games` 之後的格子 sgf 是空的（或只有空白）、color 卻寫了 B/W/b/w（官方說明「color_1…color_20 一律是 B」）→ `clear_colour_after_num_games_if_sgf_empty` 把那格 color 清空，件數記在 `t1.json` T1.3 的 `counts`。`num_games` 之後的格子如果有棋譜，仍然是 FAIL_STRUCT |
| FAIL_OPENING | 第 2 步加 `--allow-fail-opening --reason "..."`；競賽二第 5 步改用 R10（`--fallback R10`）；競賽一照常每池程序（notebook 和 `t02_player.py` 都接受 FAIL_OPENING，報告記下 verdict）；T3 記「無法判斷」 |
| FAIL_STRUCT | **停**。先寫 PREREG 修訂，修訂前不做任何預測 |

- 官方的執色設計（競賽一全黑、競賽二每題單一執色）只記錄，不改判定（rev 8）：`t1.json` T1.3 每個檔有 `official_colour_design_record_only`，競賽一記有幾格不是 B，競賽二記有幾題混了兩種執色、`num_games` 之後填的執色和該題不同的格數。`matches_official_design` 是 false 時照常往下做，寫進 RESULTS。

commit：`results/test02/checks/{rank,player}/`（`t1.json`、`summary.json`）。

## 2. 打包給 Colab

```
PY tools/t02_test_bundle.py build --kind official
```

- 只收第 1 步完整 T1 的 `t1.json`（見上）。寫出 `data/test/<6 個檔>`（正規化副本，否則逐位元組複製）、`data/test/MANIFEST.json`；用 `t02_checks.py t1 --only T1.1,T1.2,T1.3` 重驗 `data/test`（寫 `results/test02/checks/built/<comp>/t1.json`，必須 PASS；這是 `t02_checks.py` real mode 對 `data/test` 唯一接受的輸出位置）；寫 `results/test02/checks/bundle_build.json` 和 `data/colab_bundle/C10_test_v1.zip`，印出 zip 的 SHA-256。
- 放到給使用者拖的資料夾（`data\colab_upload\test02\` 裡只能有這 3 個檔），並核對 SHA-256：

```
PY -c "import hashlib,os,shutil; d='data/colab_upload/test02'; os.makedirs(d, exist_ok=True); [shutil.copyfile(s, os.path.join(d, os.path.basename(s))) for s in ('colab/test02_rank.ipynb', 'colab/test02_player.ipynb', 'data/colab_bundle/C10_test_v1.zip')]; [print(hashlib.sha256(open(os.path.join(d, f), 'rb').read()).hexdigest(), f) for f in sorted(os.listdir(d))]"
```

  zip 的 SHA-256 要等於 `bundle_build.json` 的 `zip_sha256`；兩份 notebook 要等於 repo 裡已 commit 的 `colab/test02_*.ipynb`。
- commit：`results/test02/checks/bundle_build.json`、`results/test02/checks/built/`（`<comp>/t1.json`、`<comp>/summary.json`）。
- 把 `docs/COLAB_RUN.md`「測試集（test02）」一節的步驟給使用者。

## 3. Colab 在跑的時候：T2–T5

```
PY tools/t02_checks.py run --comp rank
PY tools/t02_checks.py run --comp player
```

- real mode 讀 `data/test`，要 `MANIFEST.json`（kind official）和原始檔的 `t1.json` 對得上。
- 後果（登記第 9 節第 7–9 條）都**不擋第一次上傳甲**：T2 不過 → 開 split v2、另寫登記，新登記前任何預測不得利用這些複製；T3 判定重疊 → 考慮乙、另寫登記；T4 的 |b_c − b_q| > 0.5 由 `t02_player.py` 在防護裡自動套第 3 層，T4 只回報；T5 只記錄。
- rev 8 的補充（都寫進 RESULTS）：
  - **T2(a)**：每個檔的複製另外分成 `same_side_copies_record_only`（訓練集那一列的執色和測試棋相同，可能是同一位玩家）和 `opposite_side_copies_record_only`（訓練集記的是對手那一方，不代表同一位玩家洩漏）。不過的門檻仍然算全部複製；兩個數字都報。
  - **T3**：候選池的黑棋比例 b_c 剛好是 1（官方競賽一全黑）時，那一池改用 T3b：參考集只用 fit 玩家的黑棋、門檻用 `t3b_null.json`，判定規則不變（raw 或 canon 有 ≥ 40 人超過門檻就判定重疊）。`t3.json` 的 `variants` 寫每池用了哪一種；該池原本 T3 的數字放在 `registered_T3_record_only`，只記錄。`reference_check` 會多出 `t3b_fit_reference`、`t3b_raw`、`t3b_canon`，要全部 passed。real mode 缺 T3b 的 3 個檔時 `run` 直接拒絕；重建出來和 commit 的對不上時 T3 是 ERROR（照 ERROR 的後果查原因；T3 的 verdict 不擋上傳）。
    - b_c 不是剛好 1、但 max(b_c, 1 − b_c) ≥ 0.9 的池（例如官方池混進幾盤白棋，或整池白棋）照原本的 T3 跑，但標 `low_power`：原本的 T3 在幾乎單色的 profile 上幾乎偵測不到重疊。這種池沒判定重疊時，該池 `conclusion` 是「not determinable (low power)」、T3 的 verdict 是 `LOW_POWER`，RESULTS 寫「無法判斷（偵測力不足）」，**不寫「沒有重疊」**；判定重疊時仍是 OVERLAP。`t3.json` 的 `low_power_pools` 列出這些池。後果和 PASS 一樣不擋上傳、不觸發乙。
  - **T4**：整個檔都是黑棋時，用混色池模擬出來的虛無範圍（黑棋比例、第一手 TVD 最大值、右上角比例）標成 `not applicable`，原本的值放在 `mixed_colour_null_ranges_record_only`。這是官方結構造成的，不是異常。
  - **T5**：每份題目檔多一個 `official_design_match`（5／10／15／20 盤各 100 題）。官方是固定設計，對 5–20 均勻分布的卡方不看（`chi2_note`）。
- 但 T2–T5 要**跑完並 commit**：第 8 步的 `prepare` 會檢查 `summary.json` 裡有 T2–T5 四項、各自的細節檔已 commit（verdict 是什麼都可以）。真的跑不了（例如程式出錯）才在 `prepare` 加 `--checks-missing-reason "..."`，原因會記進上傳紀錄、印在給使用者的確認段，也寫進 RESULTS。
- 內容檢查對本機驗證棋譜的處理同第 1 步。
- commit：`results/test02/checks/{rank,player}/`。

## 4. 收 Colab 結果（先 commit，再做任何本機步驟）

使用者把兩個 zip 放進 `results\test02\`（COLAB_RUN 第 20 步）。檔名帶 ` (1)`（使用者的「下載」資料夾裡還有同名的舊檔，例如彩排或上一次失敗的那份）：先確認它是這一次的（unpack 印的執行紀錄、`config.json` 的時間），再改名成 `test02_<comp>_results.zip`，或用 `--zip` 直接指定那個檔；不確定就請使用者刪掉「下載」裡的舊檔，從 Drive 重新下載。

```
PY tools/t02_test_bundle.py unpack --zip results/test02/test02_rank_results.zip --comp rank
PY tools/t02_test_bundle.py unpack --zip results/test02/test02_player_results.zip --comp player
```

- `unpack` 拒絕：絕對路徑或 `..` 的成員、不在 `<comp>/` 底下的成員、已存在的目標檔、`run_log` 狀態不是 `completed`、`config.test_bundle.manifest_sha256` 和本機 `data/test/MANIFEST.json` 不同、config 寫 smoke 或 rehearsal。
- **每次都看 `unpack` 印的執行紀錄**，例如 `run_log.json runs: 1 preflight_failed, 2 completed`（也記在 `unpack_<comp>.json` 的 `runs`）。
  - 最後一次之前有 `seam_failed`、`repro_failed`、`readonly_changed` 或 `final_check_failed`：`unpack` 拒絕（`t02_rank.py`、`t02_player.py` 的 real mode 也會拒絕）。這代表使用者在紅字之後、沒改 Drive 資料夾名稱就又按了「全部執行」（很可能在同一個執行階段），不是登記的「換一個執行階段重跑一次」（第 9 節第 2 條、rev 6 第 10 條）。把它算成第 N 次嘗試（記進 RESULTS）：這次的輸出不用、不上傳，照下面「重跑一次」封存，請使用者改 Drive 資料夾名稱、刪掉執行階段後開新的一個重跑。同一個資料夾裡的這次重跑不算登記的那一次；登記的重跑（改名、新的執行階段）還是沒過，就停下查清楚（同下一點的「重跑還是沒過」）。
  - 前面只有 `preflight_failed`（例如 GPU 關卡停下、重選 GPU）或 `interrupted`、`running`（斷線後接續）：允許，照常。
- commit：`results/test02/rank/`、`results/test02/player/`、`results/test02/unpack_{rank,player}.json`（zip 本身不 commit）。
- **狀態不是 completed**（`seam_failed`、`repro_failed`、`readonly_changed`，或競賽一的 `final_check_failed`、`preflight_failed`）：那次執行的測試輸出一律不用、不上傳（登記第 9 節第 2 條）。
  - 只為了查看：`PY tools/t02_test_bundle.py unpack --zip ... --comp rank --allow-failed --results-dir "<scratch>\failed_rank"`（`--allow-failed` 只能解到 repo 外）。
  - 換一個執行階段重跑一次，照下面「重跑一次」做。
  - 重跑還是沒過：停下查清楚；要放寬容許差，照 rev 4 第 8 條另寫修訂，修訂前不上傳。

### 重跑一次（notebook 的接縫沒過，或第 5、6 步的本機接縫沒過）

登記第 9 節第 2 條：那次執行的測試輸出一律不用，換一個執行階段重跑一次。重跑必須**真的重算**：競賽二的 notebook 會沿用已完成的 dump（完成標記對得上就跳過），不改名就等於拿同一份輸出再比一次。第 `N` 次重跑用 `_failed_N`。

1. **封存失敗的那一次**（每次都做，有沒有 unpack 過都一樣）。把這一場在 `results/test02/<comp>/` 的全部東西（競賽二的 `fallback_R10.json`、`fallback_selfcheck_R20.json` 留在原處）、`results/test02/unpack_<comp>.json` 移到 `results/test02/failed/<comp>_N/`；失敗那次的結果 zip（`results/test02/test02_<comp>_results.zip`，notebook 狀態不是 completed、unpack 拒絕時它也還在這裡）移到 `data/test02_failed/`（gitignore）；在 `results/test02/failed/<comp>_N/attempt.json` 記原因和 zip 的 SHA-256。然後 commit（訊息寫「<comp> attempt N archived: <原因>」）。不先搬走舊 zip 的話，第 3 點把新 zip 拖進同一個資料夾時 Windows 會問「取代或略過」：略過就留著舊的，取代就蓋掉失敗那次的證據。以競賽二第 1 次為例：

```
PY -c "import os,shutil,sys,json,hashlib; c,n,why=sys.argv[1],sys.argv[2],sys.argv[3]; r=os.path.join('results','test02'); src=os.path.join(r,c); dst=os.path.join(r,'failed',c+'_'+n); keep={'fallback_R10.json','fallback_selfcheck_R20.json'}; os.makedirs(dst); [shutil.move(os.path.join(src,f),os.path.join(dst,f)) for f in (sorted(os.listdir(src)) if os.path.isdir(src) else []) if f not in keep]; u=os.path.join(r,'unpack_'+c+'.json'); os.path.exists(u) and shutil.move(u,dst); z=os.path.join(r,'test02_'+c+'_results.zip'); zs=hashlib.sha256(open(z,'rb').read()).hexdigest() if os.path.exists(z) else None; fz='data/test02_failed/test02_'+c+'_results_'+n+'.zip'; os.makedirs(os.path.dirname(fz),exist_ok=True); zs and shutil.move(z,fz); json.dump({'comp':c,'attempt':int(n),'reason':why,'zip_sha256':zs,'zip_moved_to':fz if zs else None},open(os.path.join(dst,'attempt.json'),'w'),indent=1); print(dst, sorted(os.listdir(dst)))" rank 1 "<原因，例如 notebook seam_failed>"
git add -A results/test02
```

   競賽一把 `rank 1` 換成 `player 1`。`results/test02/failed/<comp>_N/` 已經存在時指令會停下（不覆蓋），N 加 1。
2. **使用者改 Drive 的資料夾名稱**（逐步寫給使用者）：Drive →`C10`→`results`→`test02`，在 `rank` 資料夾上按右鍵 →「重新命名」→ 改成 `rank_failed_1`。競賽二**一定要改**；競賽一也改成 `player_failed_1`（notebook 每池都重算，改名只是把舊輸出留著查原因）。notebook 看到上一次的 `seam_check.json` 沒過時，會在 preflight 停下、要求先改名；本機接縫沒過時 notebook 自己不知道，所以這一步一定要由 Claude 提醒。
3. 使用者照 COLAB_RUN 第 7–20 步重跑那一份（選和第一次相同的 GPU 規則；先「中斷連線並刪除執行階段」再開新的，不在原本的執行階段重按「全部執行」）。下載前先請使用者刪掉電腦「下載」資料夾裡上一次的 `test02_<comp>_results.zip`，否則新檔名會變成 ` (1)`。把新的 zip 放進 `results\test02\`。
4. 回到本步開頭 `unpack`，commit，再做第 5 或第 6 步。**確認真的重算了**：新的 `results/test02/rank/seam_val/dumps/_DONE_val_p02e9_d4.json` 的 `time` 要晚於封存那份（`results/test02/failed/rank_N/seam_val/dumps/`）；不是的話代表 Drive 沒改名，停下請使用者改名後再跑。封存裡沒有 dumps（notebook 那次就沒過、沒 unpack）時，看新 zip 的 unpack 印的執行紀錄：前面沒有失敗的執行（有的話 unpack 會拒絕），就代表 Drive 已改名、整份重算。

### 競賽一拿不到 G4（登記第 9 節第 10 條、rev 3 第 8 條）

登記：競賽一只用 G4，拿不到先等，等一天再由 Claude 決定。**rev 7 第 2 條把期限寫死**：到 11/20 23:59 仍沒有一次在 G4 上成功的競賽一測試執行，就改用 A100（大量 RAM 關）；重現核對和接縫標準不變，沒過就不交新的競賽一上傳檔。rev 7 已登記，所以下面 (b) 不用再寫新的修訂才能改 GPU，但改 notebook 的流程照舊。

1. 競賽二不受影響：照常做第 4、5 步，交件檔照第 8 步先上傳。
2. 使用者照 COLAB_RUN 第 13 步等。請使用者每次試完回報時間和畫面（G4 灰色、跳「無法連線到 GPU 後端」、右下角不是 G4），Claude 記進 RESULTS。
3. 從第一次試算起滿 24 小時還拿不到，Claude 決定，選項只有兩個：
   - (a) 繼續等，每天請使用者試幾次。上傳目標 11/04–11/05 只是目標；最後一次上傳的目標是 11/23 23:59，截止 11/24 23:59。建議最晚 11/20 改走 (b)，留 3 天給修訂、重建、重跑、上傳。
   - (b) 換成 A100（11/20 23:59 起是強制，之前 Claude 也可以提前改）。改法：競賽一的重現核對（和 prereg01 至少 398/400 題 Top-5 相同、重心最大差 ≤ 1e-4）和本機接縫（第 9 節第 2 條）都不放寬；notebook 的 GPU 關卡改在產生器 `colab/build_test02_player_nb.py`，照 CLAUDE.md「Colab 出錯時」的流程重建、冒煙、獨立審查、在修訂表記一筆，commit 之後才交給使用者。
4. 不要為了趕時間請使用者手改 notebook，也不要讓使用者自己選別的 GPU。

## 5. 競賽二（本機）

```
PY tools/t02_rank.py opening
PY tools/t02_rank.py apply
```

- T1 是 FAIL_OPENING 時，第二行改成 `PY tools/t02_rank.py apply --fallback R10 --reason "T1 FAIL_OPENING: <t1.json 的原因>"`。
- 輸出在 `results/test02/rank/`：`opening_test.npz`、`opening_test_report.json`、`seam_val/`（`rank_val_probs.csv`、`pred_B.csv`、`seam_local.json`）、`rank_test_probs.csv`、`pred_A.csv`、`pred_B.csv`、`decide_summary.json`、`submission_rank.csv`、`apply_test_report.json`。
- 本機接縫（`seam_val/seam_local.json`）必須過：每題機率和 prereg02 最大差 ≤ 容許差、規則 B 不同 ≤ 4 題；notebook 的 dump 逐位元組相同時，預期兩個檔也逐位元組相同。容許差由本機依完成標記重算（算 seam_val dump 的 GPU 和 prereg02 相同 1e-4，不同 1e-3），必須等於 notebook `seam_check.json` 記的。沒過就沒有測試輸出，照第 4 步的「重跑一次」處理。
- `apply` 另外會拒絕：三組 dump 的完成標記記的 GPU／torch／cuDNN 不相同（notebook 中斷後在別的執行環境接續）；Colab 跑的 notebook 不是已 commit 的 `colab/test02_rank.ipynb`（`colab/test02_rank.ipynb` 本身有沒 commit 的修改也拒絕）；real mode 給 `--fallback-file`，或 `fallback_R10.json` 最後一次 commit 不早於 11/04 11:00。前兩種照第 4 步的「重跑一次」處理。

## 6. 競賽一（本機）

```
PY tools/t02_player.py run --pool seam_val
PY tools/t02_player.py run --pool public
PY tools/t02_player.py run --pool private
PY tools/t02_player.py submit
```

- `seam_val` 必須過（選 Q04、閘門過、防護第 1 層、Top-5 和 prereg02 至少 398/400 題相同）；沒過時 public、private 會拒絕執行，照第 4 步的「重跑一次」處理。只有「完全相同」那幾條實作附加條件沒過（`seam_local.json` 的 `failed_implementation_additions`）時，它們和 Colab 無關：先查本機程式或 `.venv` 的 numpy／scipy 有沒有變（版本記在 `seam_local.json` 的 `environment`），不要先請使用者重跑。
- T1 是 FAIL_OPENING 時照常跑（登記第 9 節第 6 條），`pool_select.json`、`apply_report.json`、`submit_report.json` 會記下 verdict。
- 每池各自自我檢索、選組合、閘門；**測試時閘門沒過，那一池改用 Q00**（D1）。兩池可能不同，照實記錄。
- 候選不是每人剛好 100 盤：工具停下；另寫修訂，修訂前不交競賽一。
- `--no-cnn --reason TEXT` 只在 CNN 部分不能用時（登記的退路 {Q22, Q23}）。
- 輸出在 `results/test02/player/`：每池 `pool_select.json`、`top5.csv`、`apply_report.json`，`seam_val/seam_local.json`，`submission_player.csv`、`submit_report.json`。

## 7. T6（只記錄）

```
PY tools/t02_checks.py t6 --comp player
```

沒有官方明文就不用「一題對一人」（登記第 9 節第 9 條）。

## 8. commit，然後每一場各自上傳

先 commit：`results/test02/rank/`、`results/test02/player/`、`results/test02/checks/`。

**第一次上傳（U1）的前提**，全部成立就盡早上傳（目標 11/04–11/05；目的是抓格式錯誤，不是選模型）：

- T1 是 PASS 或 NORMALIZED（FAIL_OPENING 時競賽二走 R10），而且是第 1 步 raw 的完整 T1（`prepare` 會檢查 complete、stage、mode）。
- notebook 的接縫（`rank/seam_check.json`、`player/repro_check.json`）和本機的接縫（兩邊的 `seam_val/seam_local.json`）都過。
- `check_submission --official` 回 OK（`prepare` 會跑）。
- 交件檔已 commit（`prepare` 會檢查）。
- `prepare` 另外核對同一資料夾裡已 commit 的報告：競賽二 `apply_test_report.json`（real mode、本機接縫過、記的交件檔 SHA-256 等於這個檔），競賽一 `submit_report.json`（real mode、`check_submission` official 通過、SHA-256 相同），兩場的 `seam_val/seam_local.json` 通過，`checks/<comp>/t1.json` 是 PASS／NORMALIZED（FAIL_OPENING：競賽二要是 R10 做的，競賽一照常）。這些檔的 SHA-256 記進上傳紀錄。
- T2–T5 已跑並記錄（`checks/<comp>/summary.json`）：`prepare` 會檢查四項都在、細節檔已 commit，verdict 不擋上傳。跑不了時才加 `--checks-missing-reason "..."`（見第 3 步）。
- `tools/check_submission.py`、`tools/official_log.py` 已 commit 且乾淨（`prepare` 會檢查）。
- 方法欄：競賽二走 R10 時 `prepare` 預設寫 R10 的方法；`--method` 寫到另一個組合（R20／R10）會被拒絕。

一次一場。以競賽二為例：

```
PY tools/official_log.py prepare --comp rank --file results/test02/rank/submission_rank.csv --reason "first upload (U1): catch format errors early"
```

1. commit `results/official_scores.json`。
2. **訊息 A**（檢查結果＋請使用者確認）：貼上 `prepare` 印出的那段「請確認這一份檔案」（比賽、隊伍、檔案、列數、SHA-256 前 12 碼、登記編號），請使用者對這一份說 OK。
3. 使用者說 OK 之後，**訊息 B 只放上傳指示**：到 AIdea「競賽二：從圍棋棋譜預測玩家棋力等級」（TEAM_11248；比賽頁 https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403 ，出處 `docs/REGISTRATION.md`「來源」）的上傳頁，上傳 `D:\AI-Portfolio\competitions\aicup2026-go\results\test02\rank\submission_rank.csv`（SHA-256 前 12 碼 …），傳完回報時間和 public 分數（數字或截圖）。畫面上的按鈕名稱以當天看到的為準。
   - 訊息 B 一定寫比賽的完整名稱和比賽頁網址：同一個 AIdea 帳號還有 C1 太陽能板，而且競賽二的隊號（11248）比競賽一（11249）小，只寫「競賽二」或隊號容易點錯頁。
   - 上傳頁要求的檔名或格式（例如要 zip、別的檔名、別的欄位）和 `submission_*.csv` 不同：先停下，不改檔、不上傳，當作 D5（第 0 步：官方定義變動）處理。
4. 使用者回報後：

```
PY tools/official_log.py uploaded --id N --ok-at 2026-11-04T15:00 --uploaded-at 2026-11-04T15:05
PY tools/official_log.py public --id N --score 0.xxxx [--lb-rank R] [--note "..."]
```

   時間沒寫時區就當台灣時間（+08:00）；`--ok-at` 是使用者在對話裡說 OK 的時間。然後 commit。
5. 競賽一同樣做一次：`--comp player --file results/test02/player/submission_player.csv`，隊伍 TEAM_11249，比賽「競賽一：從圍棋棋譜集辨識玩家」（比賽頁 https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688 ）。

## 之後的上傳（U3–U6）

- `public` 會標 `investigate: true`：分數低於官方 baseline（競賽二 0.4631、競賽一 0.2845），或比同一場上一次低 0.05 以上。這時只查錯誤（ID 對應、列順序、池的配對、接縫、防護層級），**不因為分數換方法**。其他差異不算證據：public 400 題的標準誤最高約 0.025。
- 上傳被拒或格式錯誤：修 bug、所有檢查重跑、重新取得 OK。
- 之後只有三種情況能再上傳：(a) 修 bug；(b) 之後的登記（例如 03）依本機驗證規則採用的方法；(c) 把最終選擇重傳一次，讓它成為最後一次上傳。**不為了在 public 排行榜試變體而上傳。**
- 每天最多 5 次，**以最後一次上傳計分**。
- 最終選擇（U5）：現在兩場都是 prereg02（甲）。

```
PY tools/official_log.py final --comp rank --id N --basis "prereg02 (甲): local-validation primary metric and the registered adoption conditions"
PY tools/official_log.py status
```

  最後一次上傳在 11/23 23:59 前完成（11/24 留作緩衝），`status` 要顯示最後一次上傳的 SHA-256 等於最終選擇。之後不再上傳，除非那次上傳本身失敗。11/23 前告訴使用者最後要傳哪一份。
- 看過官方分數才做的任何選擇，都寫進報告（VALIDATION 紀律 4）。分數只引用 `results/official_scores.json`。

## 彩排（選做，10/28–11/02，需要使用者約 1 小時）

`<S>` = 這個 session 的 scratch 底下一個新資料夾，例如 `...\scratchpad\testkit\rehearsal`。彩排的東西都不上傳、不 commit；結果記在 RESULTS（或下一版修訂 rev 9），不在已登記的 rev 8 後面補字。

```
PY tools/t02_fake_tests.py from-val --root "<S>" --zip-kind rehearsal
PY tools/t02_checks.py t1 --comp rank --rehearsal --root "<S>" --only T1.1,T1.2,T1.3
PY tools/t02_checks.py t1 --comp player --rehearsal --root "<S>" --only T1.1,T1.2,T1.3
```

- 假測試集是本機驗證檔改前綴而來：**只准跑 T1.1–T1.3**，內容統計（T1.4–T5）永遠不在它上面跑。這兩行 `t1` 帶 `--only` 只適用彩排的樹；正式那天第 1 步的 `t1` 不帶 `--only`（real mode 也會拒絕）。
- 把彩排的 zip 和兩份 notebook 放進 `data\colab_upload\test02\`（裡面只能有這 3 個檔；有別的先刪），並核對 SHA-256：

```
PY -c "import hashlib,os,shutil,sys; d='data/colab_upload/test02'; os.makedirs(d, exist_ok=True); [shutil.copyfile(s, os.path.join(d, os.path.basename(s))) for s in ('colab/test02_rank.ipynb', 'colab/test02_player.ipynb', os.path.join(sys.argv[1], 'data', 'colab_bundle', 'C10_test_v1.zip'))]; [print(hashlib.sha256(open(os.path.join(d, f), 'rb').read()).hexdigest(), f) for f in sorted(os.listdir(d))]" "<S>"
```

  zip 的 SHA-256 要等於 `<S>\FAKE_TREE.json` 的 `zip_sha256`（kind rehearsal）；兩份 notebook 要等於 repo 裡已 commit 的 `colab/test02_*.ipynb`。
- 使用者照 COLAB_RUN「測試集（test02）」一節跑；結果在 Drive 的 `results/test02_rehearsal/`。使用者把 zip 放進 `results\test02\` 後，Claude 立刻移到 `<S>\downloads\`，別讓它們留到正式那天。
- 本機：

```
PY tools/t02_test_bundle.py unpack --zip "<S>\downloads\test02_rank_results.zip" --comp rank --root "<S>" --results-dir "<S>\rehearsal"
PY tools/t02_test_bundle.py unpack --zip "<S>\downloads\test02_player_results.zip" --comp player --root "<S>" --results-dir "<S>\rehearsal"
PY tools/t02_rank.py opening --rehearsal --results-dir "<S>\rehearsal" --tests-dir "<S>\data\test" --out-dir "<S>\local_rank" --checks-dir "<S>\results\test02\checks"
PY tools/t02_rank.py apply --rehearsal --results-dir "<S>\rehearsal" --tests-dir "<S>\data\test" --out-dir "<S>\local_rank" --checks-dir "<S>\results\test02\checks"
PY tools/t02_player.py run --pool seam_val --rehearsal --results-dir "<S>\rehearsal" --tests-dir "<S>\data\test" --out-dir "<S>\local_player" --checks-dir "<S>\results\test02\checks"
```

  再對 `--pool public`、`--pool private` 和 `submit` 做一次（同樣的旗標）。
- 預期（spec 5.2）：
  - 競賽二：拿到 G4 時本機接縫逐位元組相同、`rank_test_probs.csv` 在前綴對應下等於驗證的機率；拿到 A100 時（COLAB_RUN 第 9 步允許）只要求每題機率最大差 ≤ 1e-3、規則 B 不同 ≤ 4 題，不會逐位元組相同。
  - 競賽一（只用 G4）：`seam_val` 選 Q04、閘門 +0.4991、第 1 層、Top-5 和 `player_val_top5.csv` 逐位元組相同；public 選 Q04（0.902147 對 Q08 0.898223）、第 1 層（ρ raw 1.0126）；private 選 Q04（0.898212 對 Q08 0.897055）、第 1 層（ρ raw 1.0040）。
  - 這次的接縫是在真的 GPU 上算的；`unpack` 印的執行紀錄應該只有一次 `completed`（前面最多是 `preflight_failed`、`interrupted`）。
- **彩排一結束就收拾**，不要留到正式那天：
  - 請使用者做 COLAB_RUN 第 21 步：把 Drive `C10` 裡的 `test02_rank.ipynb`、`test02_player.ipynb`、`C10_test_v1.zip` 收進 `old_rehearsal`。同名的舊 notebook 留在 `C10` 或 Colab 的「最近」清單裡，正式那天容易開錯；兩個 `C10_test_v1.zip` 並存時 notebook 可能讀到彩排的那份。
  - 刪掉 `data\colab_upload\test02\` 裡的彩排檔。
  - 請使用者刪掉電腦「下載」資料夾裡彩排的 `test02_rank_results.zip`、`test02_player_results.zip`：留著的話，正式那天下載的檔名會變成 ` (1)`，容易拖錯（unpack 會擋，但浪費時間）。
  - Drive 的 `results/test02_rehearsal` 不用動；正式那天的結果在旁邊的 `results/test02`，COLAB_RUN 第 19 步寫明從 `test02` 下載。
  - 正式日 COLAB_RUN 第 2、5 步會再確認一次；第 7、12 步要使用者從 Drive 的 `C10` 資料夾開 notebook，並看第一格印的種類是 official。
  - 彩排後若改過產生器、重建了 notebook，Drive 上 `old_rehearsal` 裡的是舊版：本機兩個工具在 real mode 都會拒絕不是已 commit 版本的 notebook 跑出來的結果。
