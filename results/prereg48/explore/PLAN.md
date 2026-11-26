# prereg48 PLAN：競賽一新題的 CNN 向量改用 TTA-8（先於任何 prereg48 向量與分數）

日期 2026-11-15 晚。分支 `p48-tta`。只做競賽一；不碰驗證集答案資料夾，不讀 `data/prereg10/val`，不跑 `tools/score.py`，
不用打分機會，不用競賽二的任何東西。

## 0. 白話版

v2-6 的 CNN 向量（prereg43 c*＝種子 0、320,000 步、raw 權重）每盤只算一次（`tta1`）。prereg42 PLAN 早就列了 TTA-8
（8 種盤面對稱各算一次、每盤向量取平均），但一直沒做。這一輪只把 CNN 向量換成 TTA-8，其他（後端、P1 項）在 D 上重調，
和 prereg43 的 c* 同一套程序。不訓練。

## 1. 向量（4090，戰情室分配）

- `p43_run.py embed --ckpt data/prereg43/S0/ckpt/step_320000.pt --weights raw --tta 8`，輸出 `data/prereg48/emb_S0/`：
  先 D（store），**select commit 之後**才算 E（store）與 pub2／priv2（pool）。
- 接縫（向量層）：同一程式 `--tta 1 --limit 200` 要和 `data/prereg43/emb_S0/V2_step_320000_raw_tta1/store_D.npz` 的前 200 盤相同
  （確認換的只有 TTA）。那份 tta1 是在 Colab G4 算的、這次在 4090，所以條件是 count 全同、最大絕對差 ≤ 1e-3（fp16 跨 GPU 誤差），
  不是逐位元。

## 2. 挑選（D 池，10 池，種子 100–109）

- 候選只有一個：c48＝S0 raw TTA-8。照 prereg43 修訂 4 的程序：prereg38 `retune_structure`（只留 B4）在 D 重調後端，
  再加 prereg39 的 P1 項（joint／b4，wP ∈ {0.25…8, 16} × CSLS k ∈ {10, 20, 40}）→ R(c48)。
- 接縫：用 tta1 向量跑同一條程式，D 要逐位元重現 0.7467766213119217。
- **D 閘門**：R(c48) − 0.7467766213119217 ≥ +0.01 才算 E 向量、進 E；否則只記錄、不算 E 向量。`tune.json` 先 commit。

## 3. E 池判定（20 池，只判一次；E 池第 6 次使用）

對 v2-6：平均差 ≥ +0.01、池 bootstrap（10,000 次、種子 43）95% 下界 > 0、20 池至少 18 池較高。E 向量的檔案時間必須晚於
tune.json 的 commit。

## 4. 採用後

pub2／priv2 用 TTA-8 向量 → 兩池 top5（v2-6 接縫：`results/prereg43/apply/<pool>/top5.csv` 逐位元組相同）→ 原題 800 列沿用
→ 1,600 列 → 檢查 → 上傳 → `results/official_scores_v2.json`。

## 5. 資源

4090：D 約 1.3 小時、（過閘門才算）E 約 2.6 小時、兩池約 10 分鐘；workers 0；VRAM ≤ 7 GB；看到戰情室 stop_now 就停（下一份前）。
CPU 部分照戰情室的長期許可（≤ 2.5 GB、8 核內、可用 RAM < 9 GB 就等）。

## 修訂 1（2026-11-15 19:50，任何 prereg48 分數之前；看過向量層接縫的數字之後）

- 向量層接縫實測（4090 tta1 前 200 盤 對 Colab G4 tta1）：count 全同，最大絕對差 0.0321，但向量沒有正規化、範數中位數約 113，
  相對差 2.8e-4，逐盤餘弦最小 0.99999967。原本寫的「最大絕對差 ≤ 1e-3」是沒考慮向量尺度的錯誤門檻。
- 改成：count 全同、逐盤餘弦最小 ≥ 0.99999、最大絕對差 ÷ 範數中位數 ≤ 1e-3。實測都過。其他不變（後端會先把每盤向量 L2，所以這個差對分數幾乎沒有影響）。
