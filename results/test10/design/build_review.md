# test10 建置紀錄（PREREG_10 A；分支 `p10-testday`）

## Codex 獨立自審（2026-10-11）

- **阻斷問題已修正**：PREREG_10 rev 3 列在解決衝突的 merge commit 首次出現；一般 `git log -S` 不檢查 merge，會使 real-mode K5-10 永遠查不到。`t10_player.py` 現在用 `git log --reverse -m -S`，仍同時要求該列在 HEAD，並以該 merge 的 committer time 和 K5-10 比較；`official_log.py` 的兩個 rev-3 查核也走 merge-aware helper。
- **真實 Git 回歸測試**：建立 main／feature 衝突、解決後以 merge commit 加入 rev 3；確認舊 lookup 找不到、merge-aware lookup 找到該 merge，且 `t10_player.k5_10_state` 通過。fixture 將 committer date 固定在 2026-10-20（K5-10 前）；另以 2026-11-01 的未固定日期先確認會失敗，再固定日期確認通過。
- **額外保護**：若任一 06 pool 記錄 `used06=true` 卻沒有 06-OFF submit report，10 會以結束碼 2 停下，避免誤把它當成可和 Q04 比較；`used06=false` 時維持既有 fallback。新增兩個直接回歸測試。
- **本次驗證**：A 專屬 pytest（`test_t10_player.py`、`test_t10_official_log.py`、`test_t10_hygiene.py`）57 passed／351 subtests；C10 規定 18-file gate 200 passed／527 subtests，僅一個明確允許的 T04 ledger baseline failure。C10_SLOW 演練既有記錄 17 passed。完整 repo 測試在未合併 scratchpad worktree 為 1496 passed／36 failed／203 skipped；失敗位於跨賽段 main pin、C:/D: 暫存路徑及已登記 T04 ledger 基線，不是 18-file gate 的額外失敗。
- **審查者**：Codex 另行檢視了差異、真實 merge 測試、K5 日期邊界、06-OFF 缺報告 guard 與既有 PREREG 限制；沒有讀 `data/local`、打分或上傳。

2026-10-10，Claude（`2cd01c0a` 派出的子代理）。只做 PREREG_10 的 A（全黑池第 3 層改交 06 的第 1 層計算）；B 沒有採用，掛鉤一律回結束碼 3。沒有測試資料、沒有打分、沒有讀 `data/local` 的答案、沒有用 GPU。所有數字來自下面列的實際執行。

## 做了什麼（commit 依序）

| commit | 內容 |
|---|---|
| `08dfd8e` | `tools/t10_player.py`（seam、apply、submit、status；p1＝B 沒採用）、`tools/t10_fake_tests.py`（nov4 假樹、負面演練的拷貝）、`tests/test_t10_player.py`、`.gitattributes` 加 `results/test10/** -text` |
| `c0801df` | `tools/official_log.py` 只加行（518 行；認得 `results/test10/player/`、prepare 前提、最後一版規則、`--not-10-reason`、K6-10、K5-10、status、public）、`tests/test_t10_official_log.py` |
| `bebcf29` | `docs/PREREG_10_PLAYER_GUARD_P1.md` 的測試日修訂（這個 commit 裡寫成 rev 2／第 11 節，合併 main 時（`1f43c01`）改為 rev 3／第 12 節；`tools/p10_p1.py` 先 commit 的揭露改指 main 的 rev 2 第 11 節第 1 條） |
| `5383443` | `tests/test_t10_dryrun.py`、`tests/test_t10_hygiene.py`、`docs/TESTDAY_10.md` |
| （本檔） | `results/test10/design/build_review.md` |

## 設計重點（細節見 PREREG_10 rev 3 第 12 節）

- **A 的判定**是純函式 `t10_player.a_conditions`：T1 PASS／NORMALIZED；test02 報告的 b_c、b_q 剛好 1，且 T4 的 `black_only_pool` 一致（不一致＝結束碼 2）；層級 3、test02 閘門通過、effective 有開局項、ρ < 凍結 θ₂，並用 `p02_player_fuse.guard_decision` 以報告的 ρ 和凍結門檻重算，applied／level／guard_reason 都要相同（不同＝結束碼 2）；06 可用（H6P、06 feats 三池、06 seam、這一池 `t06_pool.json` 是現在的）。
- **10A 的計算**直接 import `t06_player.compute_pool`（不改 06 任何一支），R02′＝test02 的 `effective`（不是 `applied`），閘門 sr(選中) − sr(R02′) ≥ 0。A 不成立的池＝06 的 `top5_off.csv`。
- **seam**：seam_val 上 A 必須不成立，並以同一段計算重算，Top-5 和 06 seam 的 `top5_off.csv`（也和 H6P 的 `player_val_top5.csv`）逐位元組相同。
- **submit**：和這一季的 06-OFF（沒有時 Q04）逐列相同＝結束碼 3「10 不需要」。
- t06 回的結束碼 3 轉成 10 自己的 `not_used.json`，從不寫進 `results/test06`。
- `official_log.py`：log 沒有 10 紀錄時和 tag `p09-testday-pre10` 逐位元組同行為（TagVersusNew，含 06 流程）；有 10 紀錄時最後一版＝06-ON kept → 06-ON，否則符合的 10，否則 PREREG_06 5.4。

## 演練（`C10_SLOW=1 tests/test_t10_dryrun.py`，假樹只用 fit 玩家，樹放 repo 外）

| 樹 | test02（ρ raw／canon、層級、b_q） | 06 | 10 |
|---|---|---|---|
| 官方形狀（`t06_fake_tests official`，同 baseline） | public 0.9934／1.0154 第 1 層；private 1.0062／1.0135 第 1 層；b_q 1 | 兩池 06（06-OFF、06-ON 都產生） | A 不成立（層級 1），兩池＝06-OFF 逐位元組；submit 結束碼 3「10 不需要」 |
| nov4（兩池；public 多一盤白棋） | public 0.7022／0.6985 第 3 層、b_q 0.999765；private 0.6761／0.6864 第 3 層、b_q 1 | 兩池都退回 test02（apply private、submit 結束碼 3） | public：A 不成立（不是全黑）→ test02 的列；private：A 成立，06 閘門 +0.0647（V2 對 Q04）→ 10A，和 test02 第 1 名不同 323 題；交件檔 `ecd483893442`，參照 Q04 |
| mix（只有 private nov4） | public 第 1 層；private 同上第 3 層 | public 06、private 退回；06-OFF 產生、沒有 06-ON | public＝06-OFF；private 10A；交件檔 `503aae053a17`，和 06-OFF 第 1 名不同 public 0、private 323 |

- nov4 的建法：每池 374 題換成新穎開局（前 4 手 8 種對稱取最小，不在自己的 100 盤候選裡），26 題（新穎棋 < 5 盤的玩家）保留原題。
- 三棵樹的 10 seam 都通過（A 不成立、和 06 seam 及 H6P 逐位元組相同）。
- 官方形狀樹的 Q04、06-OFF、06-ON 輸出和 `results/test10/design/baseline_pre10.json`（`competition1.official`）逐位元組相同。
- 負面演練（nov4 樹的拷貝）：T1 FAIL_OPENING → 3；06 seam 因 dumps 不同沒過 → 3；private 改成第 0 層（06 重跑）→ test02 的列；06 閘門對 R02′ 沒過（行程內把門檻設 1.0）→ test02 的列；guard_reason 被改 → 2；K6-10 已過 → 3（沒有 10）／2（已有 10）；`p1`（B 的 GPU 接縫演練）→ 3「B 沒有採用」；已記進上傳紀錄後 `submit`／`apply --rerun` → 2；b_q ≠ 1 是 nov4 樹的 public 池本身。
- 10 的報告（nov4、mix）複製進暫存 repo、改成 real mode 後，`official_log.t10_report_problems`／`t10_inputs_problems` 沒有問題。

## 測試結果（最後一輪，HEAD `5383443` ＋本檔）

- `tests/test_t10_player.py`、`tests/test_t10_official_log.py`、`tests/test_t10_hygiene.py`：54 passed（本檔 commit 前 `test_build_review_committed` 是唯一的失敗，就是等本檔）。
- `C10_SLOW=1 tests/test_t10_dryrun.py`：17 passed（約 17 分鐘）。
- 防誤改閘門（`AGENTS.md` 的 15 個檔＋`tests/test_t09_path_hygiene.py`，`C10_T04_GATE=1`）：186 passed、1 failed，失敗的只有允許的 `test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`，沒有被略過的測試。
- 既有的 official_log 測試 `tests/test_t06_official_log.py`、`tests/test_t09_official_log.py`、`tests/test_t09_path_hygiene.py`：68 passed（official_log 加行之後）。
- **baseline_pre10 的合併前比對**（本分支重跑）：`C10_SLOW=1 C10_KEEP=1 tests/test_t06_dryrun.py` 26 passed、`tests/test_t09_dryrun.py` 22 passed（46 subtests）；樹裡的檔和 `baseline_pre10.json` 列的 79 個 SHA-256（競賽一 33、競賽二 46）全部相同，0 個不同。

## 還沒做／要注意的

1. **B 沒做**：`p1`、GPU 接縫、tol_P、10B 都只是拒絕的掛鉤。B 過閘門時要另寫 rev 4 起的修訂並在 K5-10 前合併。
2. **編號改為 rev 3**（10/10 協調者指示）：main 上已有 B 的 rev 2（`4c2c15a`，第 11 節），本測試日修訂改為 rev 3、第 12 節；K5-10 用「`| rev 3 |` 這一列的 commit 時間 ≤ 10/31」判定（`t10_player.TESTDAY_REV_ROW`、`official_log.P10_TESTDAY_ROW`、`tests/test_t10_hygiene.py`）。rev 3 那一列的 commit 欄寫【待填】，之後補 hash。`tools/p10_p1.py` 先 commit 的揭露已在 rev 2 第 11 節第 1 條，第 12 節第 14 條只指過去。分支 commit `bebcf29` 原本寫成 rev 2，合併 main 時改號。
3. **合併前**：照 `baseline_pre10.json` 重跑 `tests/test_t06_dryrun.py`、`tests/test_t09_dryrun.py`，比對 SHA（本分支已跑，見下方）；合併後 `AGENTS.md` 的閘門加 `tests/test_t10_hygiene.py`。
4. main 的工作目錄裡 `results/test10/design/baseline_pre10.json` 是 CRLF；加了 `results/test10/** -text` 之後，合併時要刪掉那個檔再 `git checkout` 一次（內容不變，git 存的是 LF）。
5. 當時的獨立審查未完成；Codex 已於 2026-10-11 依 rev 3 第 8 節第 2 步完成自審，最新證據見本檔頂部。這是本機 self-review，不是外部代理審查。

## 合併 main 之後（10/10）

- `1f43c01`：把 main（`a8d99dd`：B 的工具和 PREREG_10 rev 2、PREREG_11／gpu-fallback、`tests/test_p10_hygiene.py`）合併進 `p10-testday`。衝突只在 `.gitattributes`（test10、test11 兩段都留）和 PREREG_10（main 的 rev 2 列和第 11 節原封不動，本測試日修訂改成 rev 3 列（commit 欄【待填】）＋第 12 節，PREREG_10 對 main 只有加行）。
- 合併後：17 個檔的防誤改閘門（`AGENTS.md`，含 `tests/test_p10_hygiene.py`）200 passed、1 failed，失敗的只有允許的 `test_ledger_and_prereg_rows_pinned`，沒有略過；`tests/test_t10_player.py`、`test_t10_official_log.py`、`test_t10_hygiene.py` 54 passed。合併只改了編號和註解，t10 的程式邏輯沒變，所以慢的演練（`test_t10_dryrun.py`、t06／t09 的基準比對）沒有重跑。

## B 的測試日擴充（PREREG_10 rev 4；分支 `p10b-testday`，2026-10-11）

Claude（`2cd01c0a` 派出的子代理）。B 還沒採用（驗證玩家的 P1、apply、稽核、H10P 都還沒做），所以這一版的 B 只在「B 採用」時才動；現在的 repo 上 `p1`、`p1-rehearse` 都回結束碼 3「B 沒有採用」、不寫檔，A 的行為不變。只用 CPU、假的 scorer；沒有測試資料、沒有打分、沒有讀 `data/local` 的答案、沒有用 GPU。數字都來自下面列的實際執行。

| commit | 內容 |
|---|---|
| `66b2ed4` | `tools/t10_b.py`（新）：`b_state`（從已 commit 的檔判斷 B 是否採用）、`p1 --pool seam_val|public|private`、`p1-rehearse`、GPU 接縫、每池的 10B 和池閘門、seam 重算驗證集 10B；`tools/t10_player.py` 接上（`B_ADOPTED` 常數拿掉）；`tools/t10_b_fake.py`（假的 `results/prereg10`）；`tools/official_log.py` 只加行（rev 3 的 `t10_report_problems` 後面加同名包裝＋B 規則）；PREREG_10 rev 4（第 13 節，tol_P【待演練】） |
| `51bab68` | `tests/test_t10_b.py`（快）、`tests/test_t10_dryrun.py` 的 `BTree`、`tests/test_t10_hygiene.py` 改成 B 的規則、`tests/test_t10_player.py`／`test_t10_official_log.py` 各改一個預期、`docs/TESTDAY_10.md` 的 B 步驟和演練 |
| （本檔） | 本節 |

### 演練（`C10_SLOW=1 tests/test_t10_dryrun.py -k BTree`，10 passed，8 分 39 秒）

- 樹：mix（public 第 1 層＝06-OFF；private nov4＝10A），假的 B＝真的凍結檔＋合成 P1（`--synthetic-p1`：SGF 落點統計，不跑 KataGo；驗證池 44,935 盤）＋`p10_apply.apply_core` 算的驗證集 10B（06 逐位元組重現）＋假的 `score_H10P.json`（+0.001 [−0.004, …]）。
- seam：A 不成立、06 照常，重算的驗證集 10B 和假樹的 `player_val_top5.csv` 逐位元組相同（400／400）。
- 演練：m＝0、tol_P＝0.05；GPU 接縫 400 盤、最大差 0、通過；P1：public 45,000 盤、private 44,290 盤，失敗 0。
- apply：合成 P1 在 w*＝16 下很差，兩池的池閘門都沒過（public −0.204310、private −0.247314），B 不加；交件檔 `503aae053a17`，和同一棵樹「B 沒採用」跑出的 10 逐位元組相同（也和 A 的 MixTree 相同）。
- 行程內把池閘門門檻改成 −1：兩池都變 10B，交件檔 `881ce5e980fd`（和 06-OFF 第 1 名不同 public 89、private 341），報告通過 `official_log.py` 的 rev 3＋B 規則（git 時間 mock 在 K6B-10、K5-10 之前），`inputs_bound` 有 B 的檔。
- 負面演練：B 沒採用 → `p1`／`p1-rehearse` 結束碼 3、不寫檔；演練紀錄綁的是別的 `val_seam_feats.npz` → 2；行程內讓 5 盤的抽取漂移 → GPU 接縫最大差 106.24 > 0.05、5 盤超過、結束碼 3、`b_not_used.json`，之後 `p1 --pool public` 回 3，apply 照 A（10A，理由 `gpu_seam_failed`），10 的 `not_used.json` 不寫；沒有演練紀錄 → 2（【待演練】）；11/04 11:00 之後演練 → 2；K6B-10 前 apply 但還沒有 P1 → 2；K6B-10 之後 `p1` → 3、apply 不加 B（`k6b_10`）；STOP → 5、刪掉後接續完成；`--fake-scorer --max-games 4` 走 `p1_extract.run_chunk`（4 盤都有目標手）。

### 測試

- `tests/test_t10_b.py`：20 passed（18 subtests）。
- `tests/test_t10_hygiene.py`、`test_t10_player.py`、`test_t10_official_log.py`：57＋（改預期後）全部通過；`test_t10_official_log.py` 的 TagVersusNew（log 沒有 10 紀錄時和 tag 同行為）通過。
- 19 個檔的防誤改閘門：224 passed、2 failed。① 允許的 `test_t04_hygiene.py::SameAsBranchPoint::test_ledger_and_prereg_rows_pinned`。② `test_g11_hygiene.py::InsertOnly::test_official_log_untouched`：它要求 `tools/official_log.py` 和**目前的 main** 逐位元組相同，所以任何還沒合併、動到 `official_log.py` 的分支都會紅；合併回 main 後就不會（A 的 `p10-testday` 當時也是這樣合併的）。沒有改這支測試，要不要改由主 session 決定。
- A 的三棵慢樹（OfficialBaseline、Nov4Tree、MixTree）在 B 的程式下重跑：17 passed（16 分 43 秒），OfficialBaseline 的 Q04／06-OFF／06-ON 仍和 `baseline_pre10.json` 逐位元組相同。

### 還沒做／要注意

1. **tol_P【待演練】**：要等 B 採用（H10P）、`val_seam_feats.npz` commit、4090 空出來，11/04 11:00 前用 `data/prereg10/t10_rehearsal/run.sh` 跑、馬上 commit。沒有它正式模式的 `p1` 一律拒絕。
2. **B 還沒採用**：`results/prereg10/RESULTS.md` 要在 H10P 後照 rev 4 第 13 節第 1 條寫 `H10P：B **採用**`（或 `**不採用**`）那一行，工具才認得。
3. **K5-10（B）**：rev 4 那一列要在 10/31 23:59 前 commit 並合併回 main；本分支沒有合併（照指示）。rev 4 列的 commit 欄是【待填】。
4. **和登記的差異**（都寫在 rev 4 第 13 節）：seam 的 10B 在 dumps 不是逐位元組相同時用 ≥ 398／400 列（第 4.2 節原寫逐位元組）；GPU 接縫的統計量定義寫死；K5-10 分 A（rev 3 列）和 B（rev 4 列）。
5. 真的 4090 路徑（`ProbeScorer`、checkpoint 完整 SHA、`nvidia-smi`）沒有實際跑過，只跑過 fake／synthetic；第一次真跑是演練。
6. worktree 注意：本機 `core.autocrlf=true`，新 worktree 會把 `results/**` 取成 CRLF，`p10_apply.apply_core` 用原始位元組比 `player_val_top5.csv` 會誤判「06 沒重現」；main 的工作目錄是 LF，不受影響。本 worktree 已改回 LF。

### 獨立審查後的修正（2026-10-12，`398b7c1` 和本檔的 commit；PREREG_10 rev 4 第 14 節）

總原則：B 的任何問題只讓 B 不用（`p1` 結束碼 3、記在 `b_not_used.json`），10 照 A 繼續；seam／apply／submit 不再因 B 回 2 或 4。

| 審查項 | 怎麼改 | 測試 |
|---|---|---|
| B1 official_log | B 的規則只在至少一池是 10B 時才查；B 採用但沒有 10B 池＝A 的報告 | `test_t10_b.py::OfficialLogB::test_adopted_without_10b_is_an_a_report`；`test_t10_official_log.py` 的 "b adopted" 改成 10B 池 |
| B1 演練 | 11/04 11:00 之後沒有可用的演練紀錄＝B 不用（3，`no_rehearsal`）；之前＝2 | `TolP::test_rehearsal_record`、`BTree::test_rehearsal_rules`、`test_gpu_seam_failed` |
| M1 | `--dry-run` 不給 `--prereg10-dir` 時讀不存在的 `<O10>_no_prereg10` | `BTree::test_a_dry_runs_never_see_b`；A 三棵樹照舊全過 |
| M2 | `b_state` 不拒絕：`score_H10P.json` 和 `RESULTS.md` 判定行要剛好一行且一致（接受項目符號、半形冒號、後接文字）；不一致在演練（測試日前）是 2、測試日是 B 不用 | `BState::test_mark_variants`、`test_inconsistent_is_b_not_used`、`test_require_adopted`、`BTree::test_inconsistent_b_files_degrade` |
| M3 | K3-10：`score_H10P.json` 最後 commit、`RESULTS.md`「**採用**」第一次 commit ≤ 10/25 23:59:59，否則 `k3_10` | `BState::test_k3_10` |
| M4 | 帳上 H10P 列數接受 2 或 4；seam 的驗證集 10B 沒過不再讓 seam 失敗（B 不用，`val_10b_seam_failed`）；apply 的 B 段任何例外都退回 A；apply 前 P1 還沒做＝`p1_not_done`（A，之後 `--rerun` 加 B） | `BTree::test_val_10b_seam_failed_is_b_only`、`test_k6b_and_apply_before_p1` |
| K5-10（B） | 改成 `| rev 4 |` 列在 main 的 first-parent 歷史第一次出現 ≤ 10/31（＝10/31 前合併回 main） | `OfficialLogB::test_k5_b_uses_main_first_parent` |
| 其他 | 正式模式 `p1-rehearse` 不收 `--now`；`TOOLS` 加 `t10_b.py`／`p10_fuse.py`／`p10_apply.py`；啟動器進 git（`tools/t10_p1_launch.sh`，`.gitattributes` 設 `eol=lf`；`data/prereg10/t10_rehearsal/run.sh` 已刪）；`.gitattributes` 加 `results/prereg06/player/player_val_top5.csv -text`；`TESTDAY_10.md` 閘門改 19 個檔、寫 tripwire 測試 | `test_t10_hygiene.py::Registration::test_part_b` |

- 測試（修正後）：`test_t10_b.py`、`test_t10_player.py`、`test_t10_official_log.py`、`test_t10_hygiene.py` 84 passed；`C10_SLOW=1 tests/test_t10_dryrun.py` 全部 30 passed（OfficialBaseline、Nov4Tree、MixTree、BTree 13 個，36 分 57 秒）；19 個檔的閘門 224 passed、2 failed（同上：允許的 T04 帳、合併前必紅的 `test_g11_hygiene.py::test_official_log_untouched`）。
- tripwire（B 採用後預期變紅，要照修訂改寫）：`test_t10_player.py::Modes::test_p1_is_b_not_adopted`、`test_t10_b.py::BState::test_real_repository_is_not_adopted_now`、`test_t10_b.py::Cli::test_not_adopted_is_3_and_writes_nothing`、`test_t10_dryrun.py::Nov4Tree::test_b_gpu_seam_is_b_not_adopted`。

## B 由 PREREG_13 採用（PREREG_10 rev 5；分支 `p13-testday`，2026-10-13）

- 先 commit PREREG_10 rev 5（`| rev 5 |` 一列＋第 16 節，只加行，`a1e7e14`），再改程式：`tools/t10_b.py`（`b_state`＝PREREG_10 的路原樣，回 `not_marked` 時才查 PREREG_13 的路；`K3_13`、`K5_13`、`B13_ADOPTED_MARK`、`H13P_LEDGER_PREFIX`、`H13P_LEDGER_ROWS`、`B13_REV_ROW`；K5 在 PREREG_13 的路另要 `| rev 5 |` 在 10/31 前進 main）、`tools/official_log.py`（只加行，rev 5 的同名包裝）、`tools/t10_b_fake.py`（`fake-p13`）、四個 tripwire 照 rev 5 改寫、`docs/TESTDAY_10.md`。`tools/t10_player.py` 沒改（PREREG_13 第 10 節沒列）。
- 稽核綁定：`apply --finalize` 會改寫 `view_report.json`，所以 `t10_b` 比的是 `audit.view_report_sha256_at_audit`＝`prescore_audit.json` 記的 SHA＝稽核 commit 時的 `view_report.json`（`git show`），而且現在的報告除了 finalize 寫的 5 個欄位外要和那一份完全相同。repo 實測：B 由 PREREG_13 採用（H13P +0.022896，下界 +0.010402 > −0.006575，n_view 263）。
- 獨立審查（另開的 Claude 子代理，沒有參與撰寫）：沒有 blocker；1 個 major（現在的 `view_report.json` 只靠 Top-5／輸入 SHA 綁到稽核，n_view、閘門、w*、選中組合沒綁）和 6 個 minor（dry-run 的 PREREG_13 資料夾改依模式而不是路徑；`k5_b_state` 自己算 `b_state` 時不拋例外；K5-13 的理由和訊息另列 `k5_13`、`require_adopted` 寫明來源；`status` 印 K5；正式模式拒收帶 `fake` 欄位的紀錄；`fake-p13` 拒絕 `--out` 等於來源或來源在 repo 裡）都已修，各有測試。
- 測試日的 tol_P 仍【待演練】：`tools/t10_p1_launch.sh rehearse`（4090 約 2 分鐘），K4-13＝11/04 11:00 前 commit `results/test10/p1/p1_rehearsal.json`。
