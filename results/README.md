# results/ 目錄：每個資料夾是什麼、結論是什麼

只是導覽，不是正本：數字以各資料夾的 `RESULTS.md`、JSON 和 `score_ledger.csv` 為準。這些資料夾多半被防誤改測試釘住，**不要搬動或改名**。

## 打分紀錄

- `score_ledger.csv`：驗證集每一次打分的紀錄帳（只增不刪）。競賽二 4 次用完（H2R、H4R、H7R、H8R），競賽一 4 次用 3 次（H2P、H6P、H13P）。

## 事前登記的結果（prereg）

| 資料夾 | 競賽 | 內容 | 結論 |
|---|---|---|---|
| `prereg01` | 兩場 | 基準和決策規則 | 競賽二採用規則 B（期望得分） |
| `prereg02` | 兩場 | 開局融合（R20、Q04） | 採用：競賽二 R20、競賽一 Q04 |
| `prereg03` | 競賽二 | 訓練配方 v2 | 不用（未送分） |
| `prereg04` | 競賽二 | 加「看對手」（04） | 採用 |
| `prereg05` | 競賽二 | 探索 | 10/2 收掉，沒寫登記 |
| `prereg06` | 競賽一 | 風格特徵＋CSLS（06） | 採用，**競賽一首選** |
| `prereg07` | 競賽二 | KataGo HumanSL（07） | 驗證集不採用；測試日仍是保底鏈的一環 |
| `prereg08` | 競賽二 | KataGo 中性探針（08） | 採用 |
| `prereg09` | 競賽二 | 6 模型集成（09） | 採用（只用 inner）；10/20 起是 19 的第一個保底 |
| `prereg10` | 競賽一 | A（開局防護）、B（KataGo 認人） | A 採用；B 在混色池沒過 |
| `prereg12` | 競賽二 | 微調 KataGo（探索） | 沒過，不用 |
| `prereg13` | 競賽一 | B′：B 在全黑視圖重判 | **採用**（+0.0229） |
| `prereg15` | 競賽二 | 等級分布調整、黑白分開（探索） | 沒過，不用 |
| `prereg16` | 競賽二 | 一致性／尾端特徵（探索） | 沒過，不用 |
| `prereg17` | 競賽二 | TabICLv2 第 7 成員（探索） | 沒過，不用 |
| `prereg18` | 競賽二 | 同一位玩家出現在多題時合併判斷（探索） | 沒過（認不出同一人），不用 |
| `prereg19` | 競賽二 | 19：整題訓練（set 模型） | **採用**（inner +0.0231），**競賽二首選**；`rank/` 是凍結檔和模型 |
| `prereg20` | 競賽一 | B′ 加一人配一題（模擬） | 沒過（06-ON 已近滿分），不用 |

PREREG_21（競賽一完全複製）的設計和模擬在 `test21`。

PREREG_11（GPU 備援）的結果在 `test11`；PREREG_14（完全複製）的設計和模擬在 `test14`。

## 測試日的設計、演練和（11/04 起的）正式輸出（test）

| 資料夾 | 對應手冊 | 內容 |
|---|---|---|
| `test02` | `docs/TESTDAY.md` | R20（競賽二）、Q04（競賽一）、T1–T6 檢查 |
| `test04`、`test04_rehearsal` | `docs/TESTDAY_04.md` | 04 的設計和 Colab 演練 |
| `test06` | `docs/TESTDAY_06.md` | 06 |
| `test07` | `docs/TESTDAY_07.md` | 07（4090 HS） |
| `test08` | `docs/TESTDAY_08.md` | 08（4090 P1） |
| `test09` | `docs/TESTDAY_09.md` | 09（CPU） |
| `test19` | `docs/TESTDAY_19.md` | 19（CPU，測試日產生） |
| `test21` | `docs/TESTDAY_21.md` | 競賽一完全複製應變層 |
| `test10` | `docs/TESTDAY_10.md` | 10（A＋B′）；`p1/p1_rehearsal.json` 是 10/13 的 4090 演練 |
| `test11` | `docs/TESTDAY_GPU_FALLBACK.md` | Colab GPU 備援演練（A100、G4 准用） |
| `test14` | `docs/TESTDAY_14.md` | 競賽二完全複製應變層 |
| `trailing_pass` | `docs/TESTDAY_02_PASS.md` | 棋局最後虛手的修補（PREREG_02 rev 9）演練 |

## 其他

- `split_v1`：驗證集切分（凍結）。
- `verify_20260924`：9/24 的資料核對。
- `bench_colab_20260925.json`、`colab_facts_20260925.json`、`data_inventory.json`：早期的 Colab 速度和資料盤點。
