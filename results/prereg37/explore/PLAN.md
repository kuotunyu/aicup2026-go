# prereg37 PLAN：競賽一新題的度量 CNN，訓練拉長到 4 倍（先於任何數字）

日期 2026-11-11（v2-2 上傳後）。這份在訓練和任何 prereg37 數字之前 commit。

## 1. 為什麼

prereg36 的紀錄（`results/prereg36/explore/tune.json`）：只用 CNN 的 D 池分數隨步數一路上升
（20k 0.265、28k 0.348、36k 0.384、40k 0.384），到 OneCycle 學習率歸零才持平；結束時訓練 loss 5.0（3,685 類），
每盤棋平均只看過約 9 次。判斷是訓練步數不足。這一輪只改一件事：步數。

## 2. 改什麼、不改什麼

- **只改**：OneCycle 總步數 40,000 → **160,000**；存檔間隔 4,000 → **16,000**（10 個存檔）。
- **不改**：`results/prereg36/explore/p36_lib.py`／`p36_train.py`／`p36_embed.py` 的程式（由 `p37_run.py` 包一層，
  只換輸出資料夾 `data/prereg37/` 和上面兩個常數）；模型（10 blocks×128、256 維）、輸入、取樣、批次、lr 2e-3、wd 1e-4、
  種子 0、fp16、NCHW；切分 T／D／E（`results/prereg36/explore/split_record.json`）；棋步快取 `data/prereg36/cache`；
  D 池、E 池（`data/prereg36/terms_D|E`，prereg36 已建好，G 各項不重算）。
- 只用官方訓練棋譜；不用競賽二的任何特徵或輸出。

## 3. 挑選（D，10 池，種子 100–109）

和 prereg36 相同的規則：10 個存檔各算「只用 CNN、CSLS 10」的 D 池平均，取最高（同分取最早）；
再在 wE ∈ {0.5, 1, 2, 4, 8, 16} × CSLS k ∈ {10, 20, 40} 取 D 池平均最高的組合（第一個最大值）＝ **M37**。
（wE 加了 16：prereg36 選到 4，更強的 CNN 可能要更大的權重；其餘同 prereg36。）`tune.json` 先 commit 才判定。

## 4. 判定（E，20 池，種子 0–19；只做一次）

- 對照 **M36**＝prereg36 已採用、已上傳的組合（step_40000、wE 4、CSLS 20，`results/prereg36/explore/eval.json`）。
  接縫：重算 M36 的 E 池平均要和 eval.json 的 0.42408307516856947 逐位元相同，否則停。
- **採用 M37 的條件（全部成立）**：mean(M37 − M36) ≥ **+0.01**；池 bootstrap（10,000 次、種子 37）95% 下界 > 0；
  20 池至少 18 池 M37 > M36。
- 沒過：維持 M36（v2-2），只記錄。
- 揭露：E 池在 prereg36 用過一次（M36 對 G）。M36、M37 的存檔和權重都只在 D 上挑，E 沒參與任何挑選，
  所以這次比較對兩者仍公平；但這是 E 的第二次使用，報告要寫明。

## 5. 過了之後

`p36_embed --source pool`（public／private）→ 套用 M37 → 組 1,600 列（原題 800 列沿用 v1 06-ON）→ 格式檢查 →
上傳（v2-3），記在 `results/official_scores_v2.json`。公開分數不用來選版本。

## 6. 資源與中斷

4090＋4 workers，預計約 12 小時（11/11 傍晚 → 11/12 早上）；RAM 監看：可用 < 4 GB 就建 `data/prereg37/STOP`
（每 100 步檢查，存檔後結束）。接續：刪 STOP 後 `bash results/prereg37/explore/run_train.sh --workers 4 --resume`。
硬截止：11/15 23:59 前判定，否則不用。

## 修訂 1（2026-11-12 06:35，任何 E 池數字之前；D 池選擇 `54b6a71` 已 commit）

PLAN 寫於 prereg38 採用前。現在已上傳的系統是「M36＋prereg38 後端（B4）」（v2-3，E 池 0.4969569920097353），所以 M37 要真正被用，必須和後端一起判。本節追加第二個判定，原第 4 節的「M37 對 M36（兩者都無後端）」照原樣判一次、只記錄、不決定上傳。

1. **後端數值重調（只在 D）**：`p38_backend.py retune --structure results/prereg38/explore/tune_D_B4.json --emb-D data/prereg37/emb/step_160000/store_D.npz --base-tune results/prereg37/explore/tune.json --out results/prereg37/explore/tune_D_B4_M37.json`；結構固定 B4，只重選 λ、LDA 維數、wE、CSLS k 等數值（照 prereg38 的 `retune_structure`）。commit 這個檔之後才碰 E。
2. **判一次（E 20 池）**：新＝M37 向量＋重調後的後端；對照＝已上傳的 v2-3（M36 向量＋`tune_D_B4.json` 的後端）。接縫：對照在 E 的平均必須逐位元等於 0.4969569920097353。程式 `p37_backend_judge.py`（用 prereg38 的 `judge_numbers`，bootstrap 種子同 prereg38）。
3. **採用條件**（同 prereg38）：平均 ≥ +0.01、bootstrap 95% 下界 > 0、20 池至少 18 池較高。
4. **E 池使用次數**：prereg36 第 1 次、prereg38 第 2 次；本修訂的兩個判定（第 4 節的原判定與本節）在同一次載入中先後算，算作第 3 次使用。報告寫明。
5. 採用後：官方兩池的 M37 向量在 4090 上算，`p38_backend.py apply --emb <M37 pool> --tune tune_D_B4_M37.json --eval <本判定>`，原題 800 列沿用 v1 06-ON，組成 1,600 列上傳。
