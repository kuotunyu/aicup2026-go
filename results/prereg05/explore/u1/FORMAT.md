# U1 存檔格式（給之後的探索直接讀）

只用 inner 偽題（split v1 inner 玩家的 prereg02 eval 輪、PREREG_03 的單色偽題），沒有碰驗證集；全部是設計階段的數字，不是驗證結果。

## 怎麼產生、怎麼讀

- `u1_repro.py` 產生 `u1_oof.npz`、`u1_repro.json`（約 3 分鐘，CPU）。它用唯讀匯入 `tools/p04_rank_fuse.py`、`p03_rank_fuse.py`、`p02_rank_fuse.py` 的函式，照 PREREG_04 tune 的流程重跑 Q00／Q01／Q02 的 5 折 OOF。
- 重現檢查（`u1_repro.json` 的 `checks`）：三個組合的 OOF 分等級平均、每題平均、各級、各盤數分箱、預測分布、單色偽題分數、每折參數，和 `results/prereg04/rank/tune_report.json` 的差距全部是 0。輸入檔的 SHA-256 也和 tune 記錄的相同。
- 讀法：`import u1_lib as U; z = U.load_saved()`。`u1_lib` 一匯入就裝上防護：任何程式碼要開、列出或 stat `data/local` 底下的東西都會直接報錯。
- 這個 npz 有 66 MB，可以隨時用 `u1_repro.py` 重建，所以不一定要進 git。SHA-256 記在 `u1_repro.json` 的 `output`。

## `u1_oof.npz` 的陣列

等級的順序一律是 `levels` = D, C, B, A, 1D, 2D, 3D, 4D, 5D, 6D（索引 0–9）。機率都是 float64，每列加總為 1。

| 陣列 | 形狀 | 內容 |
|---|---|---|
| `eval_P_Q00`、`eval_P_Q01`、`eval_P_Q02` | [127200, 10] | 登記的 inner 評估題（prereg02 eval 輪 1000–1199，混色）的 OOF 機率。Q00＝R20，Q01＝R20＋O4，Q02＝R20＋O4＋SWA＋SWB（04 採用的組合） |
| `eval_y` | [127200] | 真實等級索引 |
| `eval_n` | [127200] | 這題的盤數（5–20） |
| `eval_player` | [127200] | 玩家索引，指向下面的 `player_*` 陣列（`pseudo_inner.json` 的玩家順序） |
| `eval_fold` | [127200] | 玩家的折（0–4）；第 f 折的機率來自用其他 4 折玩家的 fit 題擬合的參數 |
| `eval_round` | [127200] | 輪次（1000–1199） |
| `eval_n_black` | [127200] | 這題裡目標玩家執黑的盤數 |
| `eval_games` | [127200, 20] int32 | 這題的棋局在 `results/prereg02/rank/dumps/inner_keys.npz` 的列號，不足 20 盤的補 −1 |
| `mono_P_Q00`、`mono_P_Q01`、`mono_P_Q02` | [120544, 10] | PREREG_03 的單色 inner 偽題（每題一種執色，k＝5／10／15／20；SHA `45d0f14a…`）。每題用它玩家那一折的 OOF 參數算出來，和 tune 的單色紀錄完全相同 |
| `mono_y`、`mono_n`、`mono_player`、`mono_fold`、`mono_round`、`mono_games` | 同上 | 意思和 eval 的相同；`mono_round` 是 0–31 |
| `mono_colour` | [120544] | 'B' 或 'W'（目標玩家的執色） |
| `player_id`、`player_level`、`player_fold` | [636] | 636 位 inner 題庫玩家 |
| `player_bank_size` | [636] | 題庫裡的盤數（20 盤的有 508 人，其餘 5–19 盤） |
| `player_n_inner_games` | [636] | 這位玩家在 inner 的全部盤數（單色偽題從這裡抽） |
| `game_colour`、`game_level` | [73556] | inner 每盤的目標執色、等級（`inner_keys.npz` 的順序） |

## 其他輸出

- `u1_analyze.py` → `u1_analysis.json`（約 6 分鐘）：分 `eval`（登記的混色評估題）和 `mono`（單色偽題）兩部分。內容有 overall、per_level、confusion_Q02、by_n、by_n_bin、by_n_bank20_players_only、test_shape_proxy、by_colour_mix／by_colour、test_shape_cells（只有 mono）、calibration_Q02、decision_headroom、crossfit_adjustments（只有 eval）、player_level。
- `u1_probe_colour.py` → `u1_probe_colour.json`（約 2 分鐘）：執色偏差能撿回多少分（逐盤診斷、交叉擬合的分色偏移、把執色項加進 Q02 校準）。

## 注意

- eval 題是每位玩家從同一個最多 20 盤的題庫抽 200 次，所以同一位玩家的題高度相關。只對題目重抽的 bootstrap（qboot）信賴區間會太窄，判斷時用 PlayerBootstrap（pboot，PREREG_04 G2 用的那一種）。
- 題庫不到 20 盤的 128 位玩家只會出現在小盤數的題目，所以 eval 的「依盤數」要看 `by_n_bank20_players_only`（只用題庫 20 盤的玩家）才不會混到玩家差異。
- 官方題目是單色、5／10／15／20 盤各 100 題，形狀比較接近 `mono_*`。
- 要加新的合併項目時，fit 題和各來源的逐盤 log 機率用 `U.load_inputs()` 重建（約 45 秒）；`U.run_gen(d, terms)` 會照 Q02 的方式重跑 OOF。
