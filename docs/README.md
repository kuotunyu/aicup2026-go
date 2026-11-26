# docs/ 目錄：哪份文件做什麼

只是導覽。很多文件被防誤改測試釘住（指紋、只准加行），**不要搬動、改名或改內容**；要修正就照規則追加修訂。

## 測試日：照這個順序看（11/04 起）

**使用者先看** `TESTDAY_USER_CHECKLIST.md`：一頁清單（每次上傳的 5 步、兩場不能搞混的地方、貼給 Codex 的獨立核對指令）。

| 順序 | 文件 | 做什麼 | 誰 |
|---|---|---|---|
| 1 | `TESTDAY.md` | test02：下載、檢查、建測試檔、R20（競賽二）和 Q04（競賽一） | Claude |
| 1a | `TESTDAY_02_PASS.md` | T1 判 FAIL_OPENING 時，檢查是不是只有棋局最後的虛手 | Claude |
| 1b | `COLAB_RUN.md` | 使用者的 Colab 步驟（test02、test04） | 使用者 |
| 1c | `TESTDAY_14.md` | 競賽二：測試題和訓練棋譜完全相同時改等級（多半 0 題） | Claude |
| 2 | `TESTDAY_04.md` | 競賽二 04 | Claude＋使用者跑一份 Colab |
| 3 | `TESTDAY_07.md` | 競賽二 07（4090 HS，約 3–4 小時） | Claude |
| 4 | `TESTDAY_08.md` | 競賽二 08（4090 P1，約 0.6 小時） | Claude |
| 5 | `TESTDAY_09.md` | 競賽二 09（CPU；19 的第一個保底） | Claude |
| 5a | `TESTDAY_19.md` | 競賽二 19（CPU，**首選**，接在 09 之後） | Claude |
| 6 | `TESTDAY_06.md` | 競賽一 06（CPU，含 06-ON 開關） | Claude |
| 7 | `TESTDAY_10.md` | 競賽一 10（A＋B′，B′ 要 4090 約 5.5 小時） | Claude |
| 8 | `TESTDAY_21.md` | 競賽一：題目和候選人的棋完全相同時改第 1 名（多半 0 題；有的話最後上傳） | Claude |
| v2 | `TESTDAY_27_V2.md` | **11/11 起的 v2 測試集**：封存 v1、分支（競賽二 K／R0／RR）、ingest 的來源；其他照上面各頁 | Claude |
| 備援 | `TESTDAY_GPU_FALLBACK.md`／`COLAB_GPU_FALLBACK.md` | 4090 不能用時，07／08 改在 Colab 算 | Claude／使用者 |

注意：`TESTDAY_07.md`、`TESTDAY_08.md` 裡「沒有 Colab 備援」的寫法已過時，以 PREREG_11 和 `TESTDAY_GPU_FALLBACK.md` 為準（`AGENTS.md` 測試日一節）。

## 事前登記（PREREG）：每個方法的規則，先寫才做

| 文件 | 內容 |
|---|---|
| `PREREG_01_BASELINE.md` | 基準、決策規則 |
| `PREREG_02_OPENING_FUSION.md` | R20、Q04、T1–T6 測試日檢查 |
| `PREREG_02_REV9_TRAILING_PASS.md` | 棋局最後虛手的修補（PREREG_02 被整份凍結，所以另開一份） |
| `PREREG_03_RECIPE_V2.md` | 訓練配方 v2（不用） |
| `PREREG_04_OPPONENT.md` | 04 |
| `PREREG_06_PLAYER_STYLE.md` | 06 |
| `PREREG_07_HUMANSL.md` | 07 |
| `PREREG_08_STRENGTH_PROBE.md` | 08 |
| `PREREG_09_MAPPING_ENSEMBLE.md` | 09 |
| `PREREG_10_PLAYER_GUARD_P1.md` | 競賽一 A、B（rev 5 起 B 可由 PREREG_13 採用） |
| `PREREG_11_GPU_FALLBACK.md` | Colab GPU 備援 |
| `PREREG_13_PLAYER_B_BLACK.md` | B′ |
| `PREREG_14_RANK_EXACT_COPY.md` | 競賽二完全複製應變層 |
| `PREREG_19_SET_MODEL.md` | 19：整題訓練的 set 模型 |
| `PREREG_21_PLAYER_EXACT_COPY.md` | 競賽一完全複製應變層 |
| `PREREG_22_ROOT_PROPS.md`／`PREREG_25_SET_MORE_PLAYERS.md` | 測試檔開頭欄位（HD）／25 |
| `PREREG_27_TESTSET_V2.md` | 官方 11/06 更新測試集：v1 怎麼保存、v2 怎麼原樣重跑（不改方法） |

沒有 05、12、15、16、17、18、20 的登記：那些探索沒過門檻，只有 `results/prereg*/explore/PLAN.md`。

## 其他

- `VALIDATION.md`：驗證集切分和打分規則（每場 4 次）。
- `DATA.md`：資料的坑、官方資料說明。
- `REGISTRATION.md`：報名查核。
- `ORAL_3MIN.md`：3 分鐘口述稿（使用者練）。
- `HISTORY.md`：逐日經過（10/18 之前的細節都在這裡）。
