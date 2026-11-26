# prereg40 結果（競賽二：S40＝SET2 輸入 × S25 玩家，取代 Q25 的 SET 項）— 不用

- 訓練：CPU 5 個種子，45 分鐘（`train_record.json`，`cfa9f19`）；port check 機率差 ≤ 1.6e-6（logp 差約 11 是 float32 極小機率的已知現象，prereg23／25 相同）。
- 0b（`stage0b.json`，`afb9943`）：留出 NLL S40 0.97812 對 S25 0.98536，−0.0072（門檻 −0.005）→ 過。
- inner 判一次（`eval_report.json`）：接縫全逐位元相同（Q25 0.7456519871546352、單色 0.7482450052195216）。
  - **G1 Q40 − Q25 = −0.0014**（Q40 0.7442130688383675；門檻 +0.005）→ 沒過
  - G2 [−0.0078, +0.0047] → 沒過
  - G4 單色 +0.0017 [−0.0014, +0.0048]（黑 +0.0043、白 −0.0009）→ 沒過
- 結論：留出 NLL 變好，但融合後 inner 分數沒增加（和 prereg23 的 SET2 一樣：增益和既有項重疊）。競賽二最終仍是 25。
- 判定前有 3 次啟動在載入資料階段就停（沒有算任何分數、沒寫 eval_started）：① worktree 路徑含 `AppData\Local` 被 U1 防護拒絕 → 改用 subst `P:`；② worktree 的 `results/split_v1/players_split.csv` 是 LF，prereg02 記錄的是 CRLF 位元組 → 用 `core.autocrlf=true` 重新 checkout 該檔；③ git 狀態顯示 M → `update-index --refresh`。之後那次是唯一判定。
