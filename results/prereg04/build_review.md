# prereg04 程式建置與審查紀錄

- 來源：workflow `c10-prereg04-build`（`wf_3fcc14fe-36f`，2026-09-29）。建置 2 個代理、整合 1 個、審查 3 個角度、修正 1 個；修正代理在 15:46 被使用者誤關，之後從中斷處續做。
- 原本寫在 scratch，這裡原樣收錄兩份，只加這段前言：
  - 第 1 部分：整合紀錄（原 `integrate/build_review_s1.md`）。
  - 第 2 部分：審查意見與修正紀錄（原 `fix/build_review_s2_s3.md`），其中第 5 節的 rev 2 文字已寫進 `docs/PREREG_04_OPPONENT.md` 第 10 節（`82027fa`）。
- 後續（不在下面兩份裡）：
  - 兩個過時的 p03 測試在 `3eab110` 另外修正，所以之後的完整測試不再需要 `--deselect`（PREREG_04 10.5）。
  - commit 前的完整測試：552 過、31 略過、0 失敗。
  - 正式執行（selftest → count → inner → dupcheck → check → selfcheck → tune → val → apply → H4R）的結果在 `results/prereg04/RESULTS.md`。

---

# 第 1 部分：整合紀錄

# prereg04 程式建置：整合檢查紀錄 s1（2026-09-29，integrator）

範圍：`tools/p04_opp.py`（I1）、`tools/p04_rank_fuse.py`（I2）、`tests/test_p04_{opp,rank_fuse,hygiene}.py`。
- 沒有在 repo 裡 commit／stash／reset／checkout／tag。
- 沒開答案檔，沒跑 `tools/score.py`。`tests/test_score.py` 只用合成資料，屬於要求的完整測試。
- 沒用 GPU，所有驗證題都沒有算 O4。
- 登記位置 `results/prereg04/{opp,rank}`、`data/prereg04/counts_fit.npz` 在 repo 裡都不存在。

## 結論

- 兩支工具在 scratch 的一次性 clone 裡用**正式模式**從頭跑過：count → inner → dupcheck → check → selfcheck → tune。
- 數字和設計階段逐位元相同：Q00 = R20、Q01、Q02 都重現；選到 Q02；G1–G4 都過；frozen 參數和 I2 用 design 輸入跑的結果完全相同。
- `val`、`apply` 在 frozen_config 還沒 commit、或沒有驗證題 O4 時都會拒絕。
- 整合時找到三個問題：
  - **一個 p04 小修**：p04_opp 的 frozen 工具核對，改成和 p04_rank_fuse 一樣涵蓋全部 `p02_*.py`。
  - **一個舊有測試的地雷（p04 之外）**：`tests/test_p03_rank_fuse.py` 有兩個測試，03 跑完後就失效。其中一個會真的執行 p03 的 real apply，改寫 `results/prereg03/rank/apply_report.json`。已還原成 HEAD 的位元組，要修。
  - **一個 clone 才會出現的換行問題**：跟工具無關，見第 7 節。

## 1. 測試

| 指令（repo，`.venv\Scripts\python.exe`） | 結果 | 時間 |
|---|---|---|
| `-m pytest -q tests`（整合前原封不動） | 533 過、2 失敗、30 略過 | 11 分 14 秒 |
| `-m pytest -q tests/test_p04_opp.py tests/test_p04_rank_fuse.py tests/test_p04_hygiene.py tests/test_p02_hygiene.py tests/test_t02_hygiene.py`（修改後） | 74 過、1 略過（`C10_SLOW` 的真實 selftest） | 1 分 59 秒 |
| `-m pytest -q tests --deselect <兩個 p03 失效測試>`（修改後，最後一次） | 534 過、30 略過、2 排除，0 失敗；跑完 `git status` 只有 5 個新檔 | 8 分 51 秒 |
| pyflakes（anaconda）5 個 p04 檔 | 乾淨 | |
| `ruff --select E,F,W --line-length 120` | 只有 p04_opp 的 5 行 E501（121–130 字）；凍結的 p03／p02 也有同類 | |

- 失敗的兩個是 `test_p03_rank_fuse.py::Unpack::test_refusals` 和 `::EndToEnd::test_real_mode_apply_needs_committed_frozen_config`，原因都不在 p04：
  - 前者斷言 `results/prereg03/rank_a` 不存在，但 9/28 已解包並 commit。
  - 後者斷言 real apply 會因 frozen_config 沒 commit 而拒絕，但它已在 e3fc40a 之前 commit，所以 **p03 的 real apply 真的跑了**：
    - 改寫了 `results/prereg03/rank/apply_report.json`，只動到 created、seconds、git_head。
    - 重寫了 `rank_val_probs.csv`，位元組相同，SHA `ec5d386d…` 和 HEAD 一樣。
  - 已用 `git show HEAD:… > 檔` 還原 apply_report.json（HEAD 是 i/lf w/lf，位元組相同），之後 `git status` 只剩 5 個新檔。
- **建議修法（p04 範圍外，交給使用者或下一步決定）**：
  - 第二個測試在 `results/prereg03/rank/frozen_config.json` 已被追蹤時 skip，或改到一次性 repo 裡測。
  - 第一個測試的最後一條斷言改成「real `rank_a` 的內容沒被動過」。
  - 修好之前，完整測試請帶上面的兩個 `--deselect`。

## 2. R20 路徑防誤改測試（在 scratch clone 改檔，repo 不動）

| 改動 | `test_p04_hygiene.py` |
|---|---|
| 不改 | 5 過 |
| `tools/t02_rank.py` 末尾加一行 | 失敗（test_pinned_files_unchanged） |
| `docs/TESTDAY.md` 加一行 | 失敗 |
| `results/test02/rank/fallback_R10.json` 改一個數字 | 失敗 |
| 同一檔刪掉 | 失敗 |
| `tools/official_log.py` 加一行（豁免） | 過 |
| `tools/t02_rank.py` 只改換行（LF↔CRLF） | 過（CRLF 折回 LF） |
| 在 `results/test02/rank/` 新增一個檔 | 過（新檔不算改動） |
| 凍結工具 `tools/p02_rank_fuse.py` 加一行 | 失敗（FrozenTools） |
| `tools/p04_rank_fuse.py` 寫進答案檔名 | 失敗（NoLabelFileName） |

每次改完都還原，clone 的 `git status` 恢復乾淨。

## 3. repo 裡的正式模式拒絕（目前 p04 檔未 commit）

| 指令 | 結果 |
|---|---|
| `p04_opp.py val` | exit 2：工具未 commit |
| `p04_opp.py val --out-dir <scratch>` | exit 2：frozen_config.json missing（預覽也一樣，沒有繞過方式；沒讀題目檔） |
| `p04_opp.py count`、`inner` | exit 2：工具未 commit |
| `p04_rank_fuse.py apply` | exit 1：frozen_config 沒 commit |
| `p04_rank_fuse.py tune` | exit 1：工具未 commit |
| `p04_rank_fuse.py selfcheck` | exit 1：check_report 沒 commit |
| `p04_rank_fuse.py check` | exit 1：p04_opp 的輸出沒 commit |

`results/prereg04/` 只有 `design/`，`data/prereg04/` 只有 `design_scratch/`。

## 4. scratch clone 的正式模式端到端（PREREG_04 第 9 節第 3–4 步，停在 `val` 之前）

- **clone 的位置和內容**：`...\scratchpad\p04build\integrate\clone\`。
  - 用 `git clone --no-hardlinks` 從 repo 複製，HEAD 是 28f5ac8。
  - 複製了 `data/training/*.csv`、`data/prereg04/design_scratch/`，以及 `data/local` 的三個非答案檔（`rank_val_questions.csv`、`player_val_questions.csv`、`player_val_candidates.csv`）。
  - 5 個新檔在 clone 裡 commit，這只是 clone 自己的 commit，repo 沒有 commit。
  - tracked 檔的位元組同步成 repo 工作區的版本（見第 7 節）。
- **跑完之後的清理**：
  - 已移除 clone 的 remote。
  - 已刪掉 clone 裡的 `rank_val_questions.csv`，這樣 clone 裡不可能誤跑 `val`。
  - clone 約 3.8 GB，含訓練資料副本，審完可以刪。

| 步驟（clone 內，`.venv` 同一個 Python 3.12） | 結果 | 時間 |
|---|---|---|
| `p04_opp.py selftest` | passed：150 盤提子 0 不符，379 個局面特徵 0 不符 | 3 分 16 秒 |
| `p04_opp.py count --workers 12` | ok，preview false。720,307 盤、目標 79,985,811 手、對手 79,985,352 手、0 異常、557,928 格；counts SHA `f6c6debe…` 等於 manifest；四個陣列和 design 相同；fit 編碼的 7 個陣列和 design 相同 | 36 分 41 秒（load 97 秒；和完整測試同時跑） |
| clone commit count_report、selftest | | |
| `p04_opp.py inner --workers 12` | ok。73,556 盤，o4／t4 和 `proxy_inner_all.npz` 差 0.0（逐位元），各階段和 n_moves 也相同，兩張表和 design 相同。輸出 SHA `100a8893…`，和 I1 在 scratch 的 `inner_full_final` 相同 | 4 分 10 秒 |
| `p04_opp.py dupcheck` | ok：驗證題 19,156 slot 在 fit 裡出現 0 個（原色、對調色都是）；0 slot 沒有手；inner 對 fit 0 盤 | 2 分 28 秒 |
| clone commit inner、dupcheck | | |
| `p04_rank_fuse.py check` | ok，可用來源 C5、OP、O4、SWA、SWB，三個組合都 ok。Δκ 見下表 | 19 秒 |
| clone commit check | | |
| `p04_rank_fuse.py selfcheck` | passed，細節見下表 | 27 秒 |
| clone commit selfcheck | | |
| `p04_opp.py val`（tune 之前） | exit 2：frozen_config missing | |
| `p04_rank_fuse.py apply`（tune 之前） | exit 1 | |
| `p04_rank_fuse.py tune` | 選 Q02，閘門全過，細節見下表 | 51 秒 |
| `p04_opp.py val`（frozen_config 未 commit） | exit 2：not committed | |
| `p04_rank_fuse.py apply`（frozen_config 未 commit） | exit 1 | |
| clone commit tune_report、frozen_config | | |
| `p04_rank_fuse.py tune` 再跑一次 | exit 1：frozen_config 已追蹤 | |
| `p04_rank_fuse.py apply`（沒有驗證題 O4） | exit 1：val_o4.npz missing。**`p04_opp.py val` 沒跑** | |
| `p04_opp.py inner`、`count`、`dupcheck` 再跑一次 | exit 2：輸出已存在，只跑一次 | |

check 的執色檢查（Δκ = 目標方參照 − 對手方來源）：

| 來源 | inner 單色 Δκ | 95% CI | 驗證單色子題 Δκ |
|---|---|---|---|
| O4 | +0.0970 | [0.0759, 0.1177] | 延到 apply |
| SWA | +0.0274 | [0.0220, 0.0325] | +0.03631 |
| SWB | +0.0249 | [0.0206, 0.0302] | +0.03436 |

- 以上和 I2 用 design 輸入跑的 check 完全相等。SWA、SWB 和 prereg03 check 紀錄的差都是 0.0。
- O4 的釘死數字都在 check 裡重算：counts 逐陣列等於 design，inner 和 design 差 0.0。

selfcheck 的各項：

| 項目 | 結果 |
|---|---|
| Q00 OOF | 0.5854702521055817，和 R20 差 0 |
| 重擬合參數 | 差 0.0 |
| 驗證題機率 | 差 0.0，規則 B、pred_B 都相同 |
| gen 路徑接縫 | 規則 B 相同；機率差 2.39e-14、fold 參數差 2.94e-14、重擬合參數差 1.39e-12（只記錄） |
| Q01 | 0.6042484782462081，差 0 |
| Q02 | 0.6101211982700248，差 0 |

tune 的結果：

| 組合 | inner OOF | 每題平均 | 單色 | se_val | 採用機率（全額／0.75 折／五折） |
|---|---|---|---|---|---|
| Q00 | 0.5854702521055817 | 0.5906 | 0.5833 | — | — |
| Q01 | 0.6042484782462081 | 0.6096 | 0.6014 | 0.00694 | 77% / 53% / 27% |
| Q02 | 0.6101211982700248 | 0.6153 | 0.6107 | 0.00738 | 92% / 71% / 39% |

- **選擇**：Q02。Q01 在 0.003 以內嗎？不在，差 0.0059。
- **閘門**：

| 閘門 | 結果 |
|---|---|
| G1 | +0.02465 |
| G2 | [0.0139, 0.0352] |
| G3 | +0.02479 |
| G4 | +0.0274 [0.0212, 0.0333]；黑 +0.0302、白 +0.0247 |

  全過，gates 物件和 design 輸入那次逐字相同。
- **只記錄的組合**：

| 組合 | inner OOF | 比 Q00 |
|---|---|---|
| O4 單獨 | 0.5050 | −0.0805 |
| R20＋SWA＋SWB | 0.5966 | +0.0111 |
| Q02＋C6A | 0.6096 | +0.0241 |
| R10 | 0.5687 | −0.0168（= fallback_R10.json） |
| Q01 無開局 | 0.5870 | |
| Q02 無開局 | 0.5975 | |

- **frozen_config**：
  - Q02，terms OP、O4、SWA、SWB。
  - w：OP 0.3653、O4 1.1880、SWA 1.3912、SWB 0.9644；a −0.00287；γ 0.4492。
  - params 和 folds 都和 design 輸入那次相同。
  - 不同的只有 check、created、design_input、inputs_sha256、preview、source_sha256、tool。
  - score_allowed 為 true，apply 還要檢查 O4 的驗證 Δκ。

以上都和 plan.md 第 5 節、PREREG_04 第 0、3 節的數字一致。

## 5. 修改（只動 p04 檔）

1. **`tools/p04_opp.py` 的 `tool_state()`**
   - 原本透過 `p3.frozen_tools_check()` 只核對 p02_rank_*、p02_moves、c10data、decide，外加 p03_rank_fuse。
   - `p02_player_fuse.py`、`p02_player_open.py` 沒有核對，和 docstring、p04_rank_fuse 的 `frozen_files()`（glob 全部 `p02_*.py`）不一致。
   - 現在全部 `tools/p02_*.py` 都要和 HEAD 相同，也列進 git status 的檢查範圍。
   - 另外把 `fn ={` 改成 `fn = {`，並 import glob。
   - count、inner、val 的計算路徑沒變。
2. **`tests/test_p04_opp.py`** 新增 `test_tool_state_covers_every_p02_tool`。

最終 SHA-256（LF）：

| 檔案 | SHA-256 |
|---|---|
| p04_opp.py | `059fadb4480224a9a7e5c1b08cd096e1949c4d50618849bb595bd9bac403f61c`（I1 回報的是 `5af3851f…`） |
| p04_rank_fuse.py | `0bb43046b978abdd5c0e69bcebbcbcebf295003fe4256f687a5d3181a7de754c` |
| test_p04_opp.py | `375746f206a90cfcb10b414c42b5bdb36c323dd2a939db89f0caa2c173d14bfe` |
| test_p04_rank_fuse.py | `4065a06a7f6b32e99f7af8d0859c0568827b3b68176c9db7e3ea3d98edee191a` |
| test_p04_hygiene.py | `f45f35138a4fe0ab1bf1bb7df48fdaac52d956bfa92f7a17edcb11eeffa35a52` |

clone 的 count_report 記的是舊版 SHA（`5af3851f…`），因為 count 開跑在修改之前。inner、check、selfcheck、tune 都是新版。

## 6. rev 2 候選（都不改科學內容；要在正式 check 之前 commit）

**p04_opp（I1）**
- a. **selftest**：用 fit 盤、雙方、379 個局面、150 盤提子，比 plan 的 ≥ 89 個局面嚴。
- b. **count**：stride／limit 只准預覽用，worker 數改成參數。
  - 整數加總和 worker 數無關：8 個和 12 個 worker 的 SHA 都是 `f6c6debe…`。
  - shard 暫存檔放在輸出資料夾底下。
  - 加了 int32 溢位檢查：最大總數 265,024,183。
- c. **只算登記的兩個變體**：o_choice_a20、t_choice_a20。
- d. **驗證題的「重播失敗」定義**：沒有解析到任何一手，或有一手下在已有子的點（重播在那裡停下）。
- e. **dupcheck 的定義**：
  - 1% 門檻算「原色或對調色任一種相同」的 slot。
  - 空落子序列不算重複。
  - 另外記錄 inner 對 fit 的結果，並核對題目檔的 slot 和 `val_keys.npz` 相同。
- f. **（整合時加）** 工具核對涵蓋全部 `p02_*.py`。

**p04_rank_fuse（I2）**
- g. **Q00 走 prereg02 自己的擬合路徑**（p03 的 run_oof、refit_all、val_probs）；Q01、Q02 走移植的 gen_fit。
  - 原因：gen 路徑的 Q00 重擬合參數和 R20 差 1.39e-12，超過登記的 1e-12；原因是加總順序不同。
  - selfcheck 另外要求 gen 路徑的 Q00 規則 B 完全相同、OOF 差 ≤ 1e-12；機率差 2.4e-14 只記錄。
  - 這是 2.3 所說「數值路徑差異」要寫的修訂。
- h. **額外的嚴格檢查**：
  - SW 權重 SHA 要等於 03 的 final.pth。
  - A0／B0 和 SW 的 dump 要等於 03 的 check 紀錄。
  - p04_opp 報告的 status 不是 ok、或 preview 為真，都當停止。
- i. **兩個 design 大檔不在本機時**：check 改用 p04_opp 報告裡自己記的比對結果。
- j. **O4 驗證題的 NaN 樣式**：C5 有解析的 slot 必須是有限值，且 `replay_failed` 全為否；C5 沒解析的 slot 不要求是 NaN。實際 19,156 個 slot 都有解析。
- k. **`--design-o4`**：只供預覽，real tune 和 apply 都拒絕。可以保留，當日後重驗的工具。

## 7. 風險與注意事項

1. **p03 的兩個失效測試**（第 1 節）：
   - 任何人照舊習慣跑完整測試，都會改寫 `results/prereg03/rank/apply_report.json`。
   - 下一次 commit 前要看 `git status`，別用 `git add -A` 把它帶進去；或先修測試。
2. **換行（只在新 clone 出現）**：
   - 系統層 `core.autocrlf=true`。新 clone 會把 LF 檔 checkout 成 CRLF，這時 `p3.p02_inputs` 用原始位元組比對 `pseudo_inner.json`，會拒絕。
   - 原本的 repo 工作區是 LF，所以不受影響。
   - 在 clone 裡用 repo 的位元組同步 241 個檔後，就全部通過。
   - 將來若在新 clone 或新 worktree 跑正式模式，要記得這點。
3. **正式 count 的時間**：
   - 12 個 worker、同時跑完整測試：36.7 分。
   - I1 用 8 個 worker 是 49 分，設計階段是 72 分。
4. **輸出只跑一次**：
   - 正式 inner、count、dupcheck 的輸出已存在時會拒絕。
   - 要重跑，得先刻意刪掉輸出，並記進修訂。
5. **commit 順序**：正式流程裡 p04_opp 的 selftest.json 也要一起 commit；check 本身不要求它。
6. **大檔**：
   - `results/prereg04/opp/inner_o4.npz` 約 12.2 MB，要進 git（check 要求它已追蹤）。
   - counts 約 178 MB，放在 `data/`，被 gitignore。

---

# 第 2 部分：審查意見與修正紀錄

# prereg04 程式建置：審查 s2／s3 的修正紀錄（2026-09-29，fixer；15:46 中斷後續做）

範圍：`tools/p04_opp.py`、`tools/p04_rank_fuse.py`、`tests/test_p04_{opp,rank_fuse,hygiene}.py`。這 5 個檔都還沒 commit。

守則（這次都有做到）：
- 沒有在 repo 裡 commit、stash、reset、checkout、tag。git 操作只在 scratch clone 裡做。
- 沒開答案檔，沒跑 `tools/score.py`，沒用 GPU。任何地方最多用 6 個 worker（CPU 和 C4 的 `methods.caw.run` 共用）。
- 凍結的檔一個都沒改：p02／p03／t02 工具、official_log、check_submission、c10data、decide、make_val、colab、docs、data、results/prereg01–03、results/test02、results/prereg04/design、打分紀錄帳。
  - 例外：刪掉 `results/prereg04/design/evidence/B_proxy/__pycache__/fuse.cpython-312.pyc`。這是本次建置的測試 12:49 寫進去的 bytecode 快取，被 gitignore，所以 `git status` 看不到。原因和修法見第 2 節 B。
- 正式的輸出位置（`results/prereg04/{opp,rank}`、`data/prereg04/counts_fit.npz`）在 repo 裡都不存在。所有真實資料的執行都在 scratch clone 裡。
- **驗證題的 O4 一次都沒算**：
  - clone 裡的題目檔在 dupcheck 跑完後立刻刪掉，之後沒有跑 `p04_opp.py val`。
  - frozen_config commit 之後，只呼叫過 `p04_opp.require_frozen_config`，確認正式 tune 的產物能過 val 的前提。這一步只讀 frozen_config，不算 O4。
- 沒有重跑 720k 盤的完整 count（照指示）。

## 結論

- 三個審查角度共 14 條意見，去掉重複是 12 件事。
  - 其中 11 件中斷前的 fixer 已經修好。續做時逐條對照目前的程式重新核對：意見都成立，修法也都正確。
  - 剩下一件是 `tests/test_p03_rank_fuse.py` 的兩個失效測試（表中第 10 條的 (1)(2)），在 p04 範圍外，沒動；p04 hygiene 已經加上保護。這次在 clone 實測了原因，見第 4 節。
- 續做時另外補了 3 件，第 2 節：
  - count 在啟動 worker 之前先做 int32 檢查。
  - 測試不再把 bytecode 寫進凍結的 design 資料夾，hygiene 也會抓這件事。
  - 新增兩個工具之間介面常數的一致性測試。
- **數字都沒變**。scratch clone 用目前的程式跑正式模式：selftest → count（6 worker 被擋；count 沿用整合 clone 的結果）→ inner → dupcheck → check → selfcheck → tune，再加 11 次預期會被拒絕的執行。
  - inner 輸出和整合 clone、設計階段逐位元相同。
  - check、selfcheck、tune、frozen_config 的數字和整合 clone 的 `d8fd1aa` 相同，差別只在新增的紀錄欄位。見第 3.2 節。
- 完整測試跑了一次（`-p no:cacheprovider`，排除兩個 p03 失效測試）：551 過、30 略過、2 排除，0 失敗。跑完後 repo 的 `git status` 只有 5 個新檔。
- 有幾項要寫進 PREREG_04 rev 2，和程式一起 commit，在正式 selftest／count 之前（所以也在 check 之前）。全文在第 5 節。

## 1. 逐條意見與處置

| # | 角度 | 問題（簡述） | 狀態 | 現在的程式 | 驗證 |
|---|---|---|---|---|---|
| 1 | correctness | int32 溢位檢查的推論不成立 | 已修 | `p04_opp.merge_shards` 第 518 行起（`int32_bound`、每個 shard 的「361 × 較多那一方的手數 < 2^32」加上「沒有負的部分和」）；續做時再加 `int32_preflight`，第 2 節 A | `test_int32_guard_in_merge`、`test_int32_preflight_before_the_workers`；clone 用 6 個 worker 在啟動前被擋 |
| 2 | correctness | docstring 說 seam 要求機率差 ≤1e-12，程式沒有 | 已修（改 docstring，程式不變） | `p04_rank_fuse.py` docstring 43–47 和 selfcheck 1281–1288 的 `required` 一致：規則 B 相同、OOF 分等級平均差 ≤1e-12；機率差、各折和重擬合參數差只記錄 | clone selfcheck：機率差 2.39e-14、各折 2.94e-14、重擬合 1.39e-12，`gen_path_Q00_seam` ok；rev 2 g |
| 3 | registration | dupcheck「執色對調」寫成把顏色位元翻轉 | 已修 | `fit_hashes`（hash → fit 各列目標執色的位元 OR）、`match_games(seqs, colours, h)`、`dup_record`：`slots_in_fit_same_target_colour`／`_other_target_colour`；inner 對 fit 也照設計版分 | `test_dupcheck_core`、`test_dupcheck_matches_design_script`（和設計版 dupcheck.py 的迴圈逐項相同）；clone：驗證 0／0／0，inner 0／0 |
| 4 | registration | selftest 不是任何後續步驟的前提 | 已修 | `p04_rank_fuse.selftest_status`，check 的第 4 項；`committed_inputs` 含 `selftest.json` | `PinnedNumbers.test_selftest`、`test_committed_inputs_include_selftest`；clone check 讀到已 commit 的 selftest（passed、0／0） |
| 5 | registration | frozen_config 沒記比率表的 SHA | 已修 | `p04_opp.tables_record`（inner、val 報告的 `tables_sha256`）→ tune 的 `source_sha256.O4.tables_sha256` → apply 的 `val_report_binding` | clone：`f380e0d7…`，另外用設計的 `counts_fit_tables.npy` 第 1、4 片獨立重算，相同 |
| 6 | registration | apply 沒核對 check_report、val_o4 的來源 | 已修 | `check_report_binding`（已 commit、未改動、SHA 等於 `check.check_report_sha256`、是通過的正式 check）；`val_report_binding`（frozen_config SHA、p04_opp.py SHA、tables_sha256） | `ApplyBindings` 三個測試；clone：check_report 被改一個位元組時 apply 停下 |
| 7 | registration | NaN 樣式的 docstring 和程式不一致 | 已修 | docstring 84–89、`load_o4_val` docstring：C5 有解析的列必須有限，沒解析的遮成 0，不要求 NaN | rev 2 j |
| 8 | registration | rank 端的順序防護沒有測試 | 已修 | `val_o4_order_guard`（ls-files 加上 git status 乾淨） | `OrderGuard` 三個測試（沒 commit、只 stage、有改動、git 失敗、smoke 豁免；tune 會停；check 記成 stop） |
| 9 | leak-r20（major） | 指紋 JSON 本身沒被釘死 | 已修 | `test_p04_hygiene.py`：常數 `FINGERPRINT_SHA_LF` 加上和 tag `r20-testday-pre04` 的 blob 比對；tag 不在時算失敗，不略過 | clone 突變測試：改 t02_rank.py 並同步更新指紋 → 失敗；刪 tag → 失敗 |
| 10 | leak-r20（major） | p03 失效測試會改寫凍結結果；hygiene 看不到 | (3) 已修；(1)(2) 在範圍外 | hygiene `FrozenResults.test_frozen_results_untouched`（prereg01–03、prereg04/design 有改動或新檔就失敗） | clone 突變：改 prereg03 的 apply_report、新增檔都會失敗；原因和建議見第 4 節 |
| 11 | leak-r20 | 同 6 | 已修 | 同 6 | 同 6 |
| 12 | leak-r20 | val 的前提只看 git 狀態不看內容；只 stage 也算 | 已修 | `require_frozen_config`：格式、smoke／preview／design_input 都要 False、選中的組合要含 O4；`git_tracked_clean` 和順序防護都把只 stage 當成沒 commit | `test_val_gate_in_a_scratch_repo`（7 種壞內容、只 stage）；clone：只 stage 時 val 被拒；正式 tune 的 frozen_config（Q02）通過內容檢查 |
| 13 | leak-r20 | 同 3 | 已修 | 同 3 | 同 3 |
| 14 | leak-r20 | official_log.py 整檔豁免 | 已修 | hygiene `test_official_log_additive_only`：tag 版的每一行都要照順序出現 | clone 突變：附加、中間插入 → 通過；刪一行、改一行 → 失敗 |

## 2. 續做時另外補的（不在審查意見裡）

**A. count 在啟動 worker 之前先做 int32 檢查**（`p04_opp.shard_moves`、`int32_preflight`，由 `count_tables` 呼叫）

- 原本的防護只在 `merge_shards` 裡，要等全部 worker 算完才會發現 worker 太少。
- 我用設計的 `moves_fit.npz` 算了每種 worker 數下最大的 shard 上界（`fix/shard_bound.py`）：

| worker 數 | 最大上界 | 佔 2^32 |
|---|---|---|
| 6 | 4,814,925,584 | 1.121（擋） |
| 7 | 4,128,758,805 | 0.961 |
| 8（登記） | 3,612,396,318 | 0.841 |

- 如果因為 CPU 和 C4 共用而把 worker 調成 6，舊版要跑 1 小時以上才在合併時停下；現在載入 fit（約 70 秒）之後、啟動 worker 之前就停下。
- 上界用編碼後的手數算，比實際計入的手數多或相同（重播遇到異常會停），所以這道檢查比合併時的檢查嚴或一樣嚴。合併時的檢查照樣保留。
- 驗證：
  - 單元測試：每個 worker 分到的手數和 `count_tables` 的統計逐一相同；上界等於 2^32 時在建立暫存資料夾之前就停下。
  - clone：`count --workers 6` 在 72 秒時以 exit 2 停下（最大 13,337,744 手，上界 4,814,925,584），沒有產生任何計數檔。
- 科學內容不變。寫進 rev 2 的 b。

**B. 測試把 bytecode 寫進凍結的 design 資料夾**

- `tests/test_p04_rank_fuse.py` 的 `design_fuse()` 用 importlib 載入設計版 `fuse.py`，沒有先設 `sys.dont_write_bytecode`。
  - fuse.py 自己雖然會設這個旗標，但 loader 在執行它之前就已經寫好快取。
  - 結果是 12:49 在 `results/prereg04/design/evidence/B_proxy/__pycache__/` 寫了 `fuse.cpython-312.pyc`。
  - 這個資料夾被 gitignore，所以 hygiene 的 `git status` 檢查看不到。
- 修法：
  - `design_fuse()` 載入前先設旗標，載入後還原（`test_p04_opp.design_module` 原本就這樣做）。
  - 刪掉那個 .pyc。
  - hygiene 新增 `test_no_bytecode_in_frozen_results`，凍結的結果資料夾裡有 `__pycache__`、`.pyc` 或 `.pyo` 就失敗。
- 驗證：
  - 不帶 `-B` 跑 `GenCalibration::test_port_is_verbatim`，design 資料夾沒有新檔。
  - clone 突變測試：放一個 .pyc 進去 → 失敗。
- 刪掉之後，repo 的 `git status --ignored -- results/prereg04` 是空的。

**C. 兩個工具之間介面常數的一致性測試**（`test_p04_rank_fuse.FrozenTools.test_interface_constants_equal_p04_opp`）

- 兩個工具透過檔名和格式字串銜接，這個測試要求兩邊寫的一致：
  - 格式字串：`p04_opp.FROZEN_CONFIG_FORMAT` 對 `p04_rank_fuse.FROZEN_FORMAT`，selftest 的格式。
  - 5 個輸出檔名、`OPP_DIR`、frozen_config 路徑、計數檔路徑。
  - 1% 門檻、slot 數、inner 盤數、工具檔名。
- 原因：p04_opp 的 val 前提會讀 tune 寫的格式字串，但端對端演練不會跑 val，這個銜接原本沒有被實際跑到。

## 3. 驗證

### 3.1 p04 三個測試檔

- 加上第 2 節的修改之後：74 過、1 略過（`C10_SLOW=1` 才跑的真實 selftest）。
- 另外補跑 `FrozenTools`、`TestCountAndScore`：11 過。

### 3.2 scratch clone 正式模式演練（`fix/e2e_s23b.sh` → `fix/e2e_s23b.log`，clone 分支 s23）

- clone commit `57389d6` 的 5 個檔和 repo 當時的版本逐位元組相同。
  - 之後 repo 只在 `tests/test_p04_rank_fuse.py` 多加了第 2 節 C 的測試，工具檔沒有再改。
  - `p04_opp.py` 的 SHA `b594e0ef…` 就是 clone frozen_config 記的那一份。

| 步驟 | 結果 |
|---|---|
| selftest（正式） | exit 0，72 秒：提子 150 盤 0 不符，特徵 379 個局面 0 不符 |
| `count --workers 6`（正式） | exit 2，72 秒：啟動前的 int32 檢查擋下，沒寫任何計數檔 |
| count | **沒重跑**：沿用整合 clone 的 `counts_fit.npz`（`f6c6debe…`，和設計相同）和 commit `1503c49` 的 `count_report.json`（舊版程式，計數邏輯相同，只有合併函式重構過） |
| inner（正式，6 worker） | exit 0，247 秒：10 個陣列和整合 clone 的 `inner_o4.npz` 逐位元相同；和 `proxy_inner_all.npz` 最大差 0、逐相位逐位元相同；兩張表等於設計的 `counts_fit_tables.npy` 第 1、4 片；`tables_sha256` `f380e0d7…`（獨立重算相同） |
| dupcheck（正式） | exit 0，77 秒：驗證 19,156 slot、1,589 題，同執色 0、對手執色 0、任一 0；inner 73,556 盤 0／0；fit 720,307 盤的落子序列都不同 |
| check（正式） | exit 0：可用 C5、OP、O4、SWA、SWB。Δκ：O4 +0.0970 [0.0759, 0.1177]；SWA +0.0274 [0.0220, 0.0325]，驗證 0.0363；SWB +0.0249 [0.0206, 0.0302]，驗證 0.0344 |
| selfcheck（正式） | exit 0，8 項全 ok：Q00 0.5854702521055817（R20）；Q01 0.6042484782462081、Q02 0.6101211982700248，和設計差 0 |
| tune（正式） | exit 0：選 Q02，四個閘門都過；G4 +0.0274 [0.0212, 0.0333]，黑 +0.0302、白 +0.0247 |

- 和整合 clone（`d8fd1aa`）比較（`fix/cmp_runs.py`）：
  - frozen 的參數、各折、閘門、選擇都相同；Q00／Q01／Q02 的 OOF 逐位元相同。
  - selfcheck.json 除了時間和環境外 0 個差異。
  - check_report、frozen_config 的差異只有這些：
    - dupcheck 的新欄位名稱；
    - 新加的 selftest 紀錄；
    - `tables_sha256`；
    - 報告記的 `p04_opp_sha256`／`frozen_config_sha256`；
    - 各報告的 SHA（因為報告內容多了欄位）。
- 拒絕情境（都照預期）：
  1. tune 之前跑 val → exit 2（frozen_config 不存在）。
  2. tune 之前跑 apply → exit 1。
  3. frozen_config 沒 commit 時跑 val → exit 2。
  4. frozen_config 沒 commit 時跑 apply → exit 1。
  5. frozen_config 只 stage 時跑 val → exit 2（「A  …」）。
  6. frozen_config 已 commit 後再跑 tune → exit 1。
  7. check_report 被改一個位元組時跑 apply → exit 1。
  8. 沒有驗證題 O4 時跑 apply → exit 1（在 check_report 的綁定檢查通過之後才停）。
  9. 已有輸出時再跑 inner、count、dupcheck → 各 exit 2。
- 最後 clone 的 `git status` 乾淨，題目檔已刪。

### 3.3 hygiene 突變測試（`fix/hygiene_mut2.sh` → `fix/hygiene_mut2.log`，在 clone）

- 基線 9 過。
- 以下情境都失敗（照預期）：
  - t02_rank.py 改一行並同步更新指紋。
  - t02_rank.py 改一行、指紋不動。
  - official_log.py 刪一行、改一行。
  - prereg03 的 apply_report 多一個位元組、prereg03 多一個新檔。
  - 刪掉 tag。
  - design 資料夾多一個 .pyc。
- 以下情境都通過（照預期）：
  - official_log.py 附加一行、在中間插入一行。
  - t02_rank.py 只把換行改成 CRLF。
  - 不帶 `-B` 載入設計版 fuse.py，之後 design 資料夾沒有 bytecode。
- 每個情境都還原了，最後 clone 乾淨。

### 3.4 兩個被排除的 p03 測試（`fix/p03_deselected.sh` → `fix/p03_deselected.log`）

見第 4 節。

### 3.5 完整測試（repo，一次）

指令：`.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests --deselect <兩個 p03 失效測試>`，紀錄在 `fix/pytest_full_resumed.log`。

- 結果：**551 過、30 略過、2 排除，0 失敗**，5 分 58 秒。
  - 比中斷前多 3 個測試：第 2 節 A、B、C 各一個。
  - 30 個略過都是要設 `C10_SLOW` 才跑的慢速測試。
- 這次沒帶 `-B`，所以會實際載入設計版 fuse.py。跑完後：
  - `results/` 底下只有原本就被 gitignore 的 6 個 zip，沒有 bytecode，證實第 2 節 B 修好了。
  - 再跑一次 hygiene：9 過，含 bytecode 檢查和凍結結果資料夾檢查。
- 跑之前和跑之後的 `git status` 相同，只有 5 個未追蹤的 p04 檔；HEAD 仍是 `28f5ac8`。

### 3.6 repo 狀態

- `git status`：只有 5 個未追蹤的 p04 檔。
- HEAD 仍是 `28f5ac8`。
- `results/` 底下沒有新的或被忽略的非預期檔案（只有原本就被 gitignore 的 6 個 results zip）。
- `results/prereg04` 只有 `design/`，`data/prereg04` 只有 `design_scratch/`。

## 4. 兩個被排除的 p03 測試

**失敗原因**

- **`Unpack::test_refusals`**（第 768 行）：
  - 它斷言 `results/prereg03/rank_a` 不存在，想確認被拒絕的 unpack 沒有寫進 repo。
  - 但 rank_a 在 `e55b019`（9/29 01:57，03 的結果解包）就已經 commit，共 20 個追蹤中的檔。
  - 測試本身沒有寫任何東西，失敗是因為斷言的前提過期了。
- **`EndToEnd::test_real_mode_apply_needs_committed_frozen_config`**（第 1308 行）：
  - 它的前提是「正式的 frozen_config 還沒 commit」。
  - `1218ba8`（9/29 02:00，03 的 tune）之後，frozen_config 已經 commit。所以測試裡的 `p03.main(["apply"])` 會真的執行 prereg03 的正式 apply，並且成功（exit 0）。
  - 這次 apply 會改寫 `results/prereg03/rank/apply_report.json`，差 7 行，都是時間、環境一類的欄位。`rank_val_probs.csv` 逐位元組不變。

**和 p04 檔無關**：clone 裡兩種狀態各跑一次，都是同樣兩個測試在同樣的行失敗，apply_report.json 也都被改寫（每次跑完都從 clone 的 HEAD 還原）：
- A：s23，p04 檔已 commit。
- B：`28f5ac8` detached，完全沒有 p04 檔。

**p04 檔 commit 之後會不會過**：不會，A 就是這個狀態。

**建議做法**（`tests/test_p03_rank_fuse.py` 在 p04 範圍外，由使用者決定；單獨一個 commit，不牽涉 PREREG_04）：
1. `test_real_mode_apply_needs_committed_frozen_config`：開頭加判斷，frozen_config 已追蹤時就略過，不要在真的 repo 上執行正式 apply。
   ```python
   rc, _ = p03.git(["ls-files", "--error-unmatch", "--", "results/prereg03/rank/frozen_config.json"])
   if rc == 0:
       self.skipTest("prereg03 frozen_config.json is committed (1218ba8): a real apply would rewrite the frozen results")
   ```
2. `Unpack::test_refusals` 第 768、771 行：兩個 `assertFalse(os.path.exists(... "rank_a"))` 改成「被拒絕的 unpack 沒有動到 repo」。
   ```python
   self.assertEqual(p03.git(["status", "--porcelain", "--untracked-files=all", "--", "results/prereg03"])[1], "")
   ```
3. 修好之前：
   - PREREG_04 第 5.1、9 節要求的完整測試都帶這兩個 `--deselect`。
   - 萬一又跑到第二個測試，p04 hygiene 的 `test_frozen_results_untouched` 會失敗，不會悄悄過關。
   - 注意：hygiene 照檔名排序在 test_p03 之後跑，所以同一輪就抓得到。

## 5. PREREG_04 rev 2 建議全文

和程式、測試一起 commit，在正式 selftest／count 之前。同一份另存在 `fix/rev2_draft.md`。

### 修訂紀錄表新增一列

| rev 2 | `（commit 時填）` | 實作偏離與補強，見第 10 節。2026-09-29 和 `tools/p04_*.py`、`tests/test_p04_*.py` 一起 commit，**在正式 `selftest`／`count` 之前**，所以也在 `check` 之前。當時仍然沒有任何 04 的正式輸出或分數，驗證題的 O4 沒算過。scratch clone 的正式模式演練結果：inner 輸出和 `proxy_inner_all.npz` 逐位元相同；Q00／Q01／Q02 的 OOF 和第 2.3 節的數字差 0。來源、格子和比率的定義、組合、選法、閘門、H4R、截止點都不變 |

### 新增第 10 節

#### 10. rev 2：實作和登記的差異（在正式 `selftest`、`count`、`check` 之前 commit）

- 下面各項**都不改科學內容**：來源、格子和比率的定義、組合 Q00–Q02、選法、閘門 G1–G4、H4R、截止點都照 rev 1。
- 分成兩類：
  - 照原規格做不到、改成最接近的做法：10.2 的 g、j。
  - 比 rev 1 更嚴的補強，或把 `plan.md` 第 5 節的要求落實成工具強制：其餘各項。

#### 10.1 `tools/p04_opp.py`

- **a. selftest**：
  - 用 fit 盤，目標方和對手方都測。
  - 提子邏輯：150 盤和 sgfmill 比。
  - 特徵：12 盤、每 7 個局面取一個，和逐點的慢速參考實作比。實測 379 個局面，比 plan 要求的 ≥ 89 個嚴。
  - `selftest.json` 和 `count_report.json` 一起 commit；`check` 會檢查它（見 l）。
- **b. count**：
  - `--stride`／`--limit` 只准預覽用。worker 數是參數，正式執行用 8 個。
  - 整數加總和 worker 數無關：8 個和 12 個 worker 算出的計數檔 SHA 都是 `f6c6debe…`。
  - worker 照設計版用 int32 累加，另外加兩道防護，任一不成立就停下，改用更多 worker：
    1. **啟動 worker 之前**：用編碼後的手數算每個 shard 的「361 × 較多那一方的手數」，要 < 2^32。
       - fit 上 6 個 worker 會被擋（4,814,925,584）。7 個以上才過（4,128,758,805）。
       - 登記的 8 個是 3,612,396,318，約 2^32 的 84%。
    2. **合併時**：用每個 shard 實際計入的手數再檢查一次同一個上界，而且任何部分和都不能是負數。
  - 兩道防護一起保證沒有 int32 溢位。
- **c. 只算登記的兩個變體**：o_choice_a20（O4）和 t_choice_a20（T4）。
- **d. 驗證題的「重播失敗」**：沒有解析到任何一手，或有一手下在已有子的點（重播在那裡停下）。
- **e. dupcheck**（照設計版 `evidence/B_proxy/dupcheck.py` 的語意）：
  - 比對整串落子：每一手的顏色和位置都相同。
  - 「原色」：fit 裡有同一串，而且那一列的目標執色和題目 slot 的 `color_i` 相同。
  - 「執色對調」：同一串，但 fit 那一列的目標執色不同，也就是同一盤從對手那邊看。
  - 1% 門檻算「任一種」的 slot 數。空的落子序列不算重複。
  - inner 對 fit 用同樣的分法，只記錄。
  - 題目檔的 slot 要和 `val_keys.npz` 逐列相同。
- **f. 工具核對**：涵蓋全部 `tools/p02_*.py`。
- **g′. val 的前提**（第 1.2、2.6 節「工具強制」的做法）：
  - `results/prereg04/rank/frozen_config.json` 已 commit，工作區沒有改動。只 stage 不算 commit。
  - 內容要是 tune 產生的正式檔：格式 `c10-prereg04 rank frozen_config v1`，smoke、preview、design_input 都是 false。
  - 選中的組合要含 O4。沒含 O4 時不需要、也不准算驗證題的 O4（第 2.6 節）。
- **h′. 比率表的 SHA**：inner、val 的報告都記 `tables_sha256`，也就是兩張表的位元組 SHA-256（[2, 557,928, 10]、float64、C 順序）。

#### 10.2 `tools/p04_rank_fuse.py`

- **g. Q00 的路徑**（依第 2.3 節「數值路徑差異」修訂）：
  - Q00 走 prereg02 自己的擬合路徑（p03 的 `run_oof`、`refit_all`、`val_probs`），所以和 R20 逐位元相同。Q01、Q02 走移植的 `gen_fit`。
  - 原因：gen 路徑的 Q00 重擬合參數和 R20 差 1.39e-12，超過登記的 1e-12。差異來自加總順序：(m＋Bβ)＋w·m_OP 對 (m＋w·m_OP)＋Bβ。
  - selfcheck 另外要求 gen 路徑的 Q00 和 prereg02 路徑一致：
    - 每一題 inner 評估題的規則 B 判定都相同；
    - OOF 分等級平均差 ≤ 1e-12。
  - 以下只記錄：機率差（實測 2.4e-14）、各折參數差（2.9e-14）、重擬合參數差（1.39e-12）。
- **h. 額外的嚴格檢查**：
  - SW 的權重 SHA 要等於 03 的 `final.pth`。
  - A0／B0 和 SW 的 dump 要等於 03 check 的紀錄。
  - p04_opp 的報告 status 不是 ok、或 preview 為真，都當停止。
- **i. 兩個 design 大檔不在本機時**：check 改用 p04_opp 報告裡自己記錄的比對結果。
- **j. 驗證題 O4 的 NaN 樣式**：
  - 對驗證題的 O4，這條**取代第 2.2 節第 1 條的「NaN 樣式和 C5 相同」**。
  - 要求：`replay_failed` 全為否；C5 有解析的 slot，O4／T4 都是有限值。
  - C5 沒解析的 slot 不要求是 NaN：遮成 0，而且 E_j 只平均有解析的盤，所以不會用到。
  - 本機 C5 在 19,156 個 slot 都有解析（`val_keys.npz` 的 parsed 全為真），實際上沒有差別。
- **k. `--design-o4`**：只供預覽，正式的 tune 和 apply 都拒絕。保留下來當日後重驗的工具。
- **l. selftest 是 check 的前提**（`plan.md` 第 5 節「提子 0 盤不符、特徵 0 個不符」）：
  - 正式 check 要求 `results/prereg04/opp/selftest.json` 已 commit，和其他 p04_opp 輸出一起核對。
  - 內容要求：passed 為真、沒有 problems、不是預覽、提子和特徵都是 0 個不符。不符就停下。
- **m. 比率表 SHA 凍結**：
  - tune 把 inner 報告的 `tables_sha256` 寫進 `frozen_config.json` 的 `source_sha256.O4.tables_sha256`。
  - apply 要求 val 報告的值和它相同。
- **n. apply 的輸入綁定**（正式模式，任一不符就停下）：
  - `check_report.json` 已 commit、沒有改動、SHA 等於 `frozen_config.json` 的 `check.check_report_sha256`，而且是通過的正式 check。
  - 選中含 O4 時，`val_report.json` 記的 frozen_config SHA 要等於現在已 commit 的那一份。
  - `val_report.json` 記的 `p04_opp.py` SHA 要等於 tune 記錄的那一份。
- **o. 順序防護**：`val_o4.npz` 存在時，`frozen_config.json` 必須已追蹤，而且工作區乾淨；只 stage 不算 commit。不符就停下（check、tune 都會檢查）。

#### 10.3 第 5.1 節的防誤改測試（`tests/test_p04_hygiene.py`）

- **p. 指紋本身釘死**：`r20_path_fingerprint.json`（CRLF 折成 LF）的 SHA-256 要同時等於：
  - 測試裡的常數 `72fa6c1248f24a6c3d9fb9d8f9f699237c769f9a968faf7eba31d42fd8b348da`；
  - 本機 tag `r20-testday-pre04` 裡的版本。tag 不在時測試失敗，不略過。
- **q. `official_log.py` 改成實際檢查「只准附加」**：
  - tag 版的每一行都要照原順序出現在新檔裡（用 LF 比對）。
  - 這取代 rev 1 寫的「除了 `official_log.py`」整檔豁免。第 5.1 節的規則本身不變。
- **r. 凍結的結果資料夾**：`results/prereg01`、`prereg02`、`prereg03`、`results/prereg04/design` 以下情況測試就失敗：
  - 有任何修改或新檔；
  - 有 bytecode（`__pycache__`、`.pyc`）。這些檔被 gitignore，`git status` 看不到，所以另外檢查。

#### 10.4 第 9 節第 3 步改成

3. `selftest`、`count`、`inner`、`dupcheck` → commit。`selftest.json` 和 `count_report.json` 一起 commit。`count` 用 8 個 worker（至少 7 個，見 10.1 b）。

#### 10.5 已知、在 04 範圍外的問題（不改登記，只記錄）

- `tests/test_p03_rank_fuse.py` 有兩個測試在 03 跑完後就失效：
  - `Unpack::test_refusals`：它斷言的 `results/prereg03/rank_a` 不存在，但這個資料夾在 `e55b019` 已經 commit。
  - `EndToEnd::test_real_mode_apply_needs_committed_frozen_config`：frozen_config 在 `1218ba8` 已經 commit，所以這個測試會真的執行 prereg03 的正式 apply，改寫 `results/prereg03/rank/apply_report.json`。
- 在 scratch clone 實測：`28f5ac8`（沒有 p04 檔）和 p04 檔已 commit 的狀態下，都是同樣兩個測試失敗，和 04 無關。
- 在使用者決定怎麼修之前，第 5.1 節和第 9 節要求的完整測試都要帶這兩個 `--deselect`。
- 萬一又跑到第二個測試，10.3 r 的檢查會失敗，不會悄悄過關。

## 6. 風險與交接

1. **正式的 count 至少要 7 個 worker，登記是 8 個**。CPU 和 C4 共用時可能比設計的 72 分鐘慢很多：12 個 worker 的那次在 41 分鐘時 worker 0 才做了 70%，後來被中斷。
   - 建議等 C4 的 `methods.caw.run` 跑完，或排晚上跑，用預設的 8 個。
   - 每個 worker 的記憶體約 90 MB（4 個 int32 陣列，每個 557,928×10）。
2. 新的 `merge_shards` 還沒在 720k 盤、8 個 worker 的真實 shard 上跑過完整合併。
   - 只做過單元測試，和暴力法逐位元相同；整合 clone 那次跑的是舊的合併程式。
   - 正式 count 會核對釘死的數字，也會逐陣列比對設計的計數，所以合併若有錯會停下，不會悄悄過關。
3. `p04_opp.py val` 和 `p04_rank_fuse.py apply` 依規定從沒在真實資料上跑過，第一次就是正式那次。
   - 已驗證過的：smoke 端對端（P04World）和單元測試都有涵蓋；正式 tune 產出的 frozen_config 能通過 val 的內容前提；兩邊的介面常數有測試。
4. 兩個 p03 測試沒修之前，完整測試一定要帶 `--deselect`，否則會改寫 prereg03 的 apply_report.json。
5. p04 工具裡有 4 行超過 120 字元（p04_opp.py 第 10、993、1107、1110 行），是之前就有的，只是排版問題，這次沒動，避免改了工具又要重跑演練。
