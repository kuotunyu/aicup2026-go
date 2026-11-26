# 事前登記 03：recipe v2（競賽二）可行性與成本評估

## 結論

1. **做得到，而且不用重抽特徵。** 目標玩家視角、訓練時的 D4、cosine LR、bf16、neighbour label smoothing，全部可以在 GPU 上從現有的 500 層 uint8 分片算出來。執色用 `(rank, game_id)` 對回訓練檔的 `color` 欄，prereg01 的特徵備份可以直接還原（約 80 秒）。
2. **主候選 R3A**：501 層（翻成「自己／對手」，再加 1 層執色）、每筆隨機套一種 D4、warmup 加 cosine、12 個 epoch、bf16、固定用最後一個 epoch。
   - G4 約 50–94 分鐘，約 7–14 個運算單元。下限是假設 bf16 快 2 倍，上限是 bf16 完全沒加速。bf16 加速沒實測，notebook 開頭 300 步要實測並印出推估總時間。
   - 第二個候選 R3B（多加 label smoothing）可以同時跑，成本加倍。
3. **本機合併**：`p02_rank_fuse.py` 被 PREREG_02 §9.1（`docs/PREREG_02_OPENING_FUSION.md:242`）凍結，不能改。要新寫 `tools/p03_rank_fuse.py`，只 import p02 的數值函式，另外定義 C6（新模型，8 種對稱）和 C7（C5＋C6 等權）。約 0.5–1 天。
4. **測試日工具**：要新增 `test03_rank` notebook、`t03_rank.py`，並改 `official_log.py` 的方法標籤。約 3–4 天；測試日多跑一份 notebook，約 1.5–2 個運算單元。
5. **inner 診斷的三個發現**（只用 CPU、只讀訓練玩家）：
   - 目前的瓶頸是跨玩家的泛化，不是訓練不夠。
   - 長棋並沒有比較差，所以不必加長盤面歷史。
   - 只在輸出端加「依執色的偏移」沒有用。所以視角能帶來多少，只有實際訓練才知道。

## (1) 目標玩家視角

**現況**
- 特徵是絕對顏色：每個盤面 t 占 2 層，第 2t 層是黑子、第 2t+1 層是白子，取最後 250 個盤面（`third_party/aicup2026_tutorial/utils.py:53-64`，其中 57 是取最後 250、59-61 是黑白兩層）。
- 分片裡只存 `features.npy`、`label.txt`、`rank_idx.txt`、`game_name.txt`，沒有執色（`rank-prediction.ipynb` 第 8 格）。

**執色從哪裡來**
- 訓練：`load_rows` 回傳的 `color` 欄（`tools/c10data.py:27`），意思是這位玩家在這盤執黑或執白（`docs/DATA.md:43`）。notebook 會帶進 `df_train`／`df_val`（`colab/build_rank_nb.py:758-759`）。
  - 訓練時用分片的 `(LEVELS[label], game_name)` 查 `df_train` 建的字典，每步 256 次查表，成本可以忽略。
  - 不要用「game_id ≤ 50000 就是黑」這條規則（`DATA.md:49` 禁止拿 game_id 當特徵），一律查 `color` 欄。
- inner：`INNER_KEYS['color']`，靠 `INNER_INDEX` 對齊（`colab/build_prereg02_rank_nb.py:686-695`）。
- 本機驗證題和測試題：題目檔的 `color_i`（官方 README:94），`parse_val` 已經逐 slot 存進 `V['keys']['color']`（`build_prereg02_rank_nb.py:777-781, 788`）。
- 本機套用時已經核對 val_keys 的執色和題目檔一致（`tools/p02_rank_fuse.py:1128-1131`）。

**建議：翻色，再加 1 層執色**（官方競賽一的特徵就是這樣做，`utils.py:97-104`）。在 GPU 上：

```
x = x_u8.view(B,250,2,19,19); x = torch.where(isB[:,None,None,None,None], x, x.flip(2))
x = torch.cat([x.reshape(B,500,19,19), isB.view(B,1,1,1).expand(B,1,19,19).to(x.dtype)], 1)  # 501 層
```

- 第 2t 層＝目標玩家的子，第 2t+1 層＝對手的子，第 500 層＝1 代表目標玩家執黑。
- 補齊用的空盤不受影響。
- 模型用 `GoRankResNet(in_channels=501)`（`network.py:25`），不改 `network.py`，所以打包檔和特徵指紋都不變（`build_rank_nb.py:829-832`）。
- 翻色只動通道、D4 只動空間，兩者可以交換順序；執色層在 D4 下不變。

**inner 診斷（只用 CPU）**
- epoch 9 在原方向的逐盤 accuracy：黑方 0.2559、白方 0.2557。
- 依玩家分兩半做交叉：每種執色各自擬合一組 10 維偏移，NLL 是 1.9395；兩種執色共用一組是 1.9362。依執色分開的偏移沒有改善。
- 所以如果有收益，只會來自模型學到「依執色而不同的盤面特徵」，這要訓練後才看得到。

**建議的自我檢查**
- 最後一個 epoch 的 inner 逐盤 accuracy，用正確執色算必須高於把執色對調後算的值。這是在驗證執色有沒有接對；只多一次原方向的推論，約 13 秒。

## (2) 訓練時的 D4 資料增強

- 用現有的 `d4()`（`build_prereg02_rank_nb.py:550-555`，自我檢查在 558-562）先算出 [8, 361] 的座標排列表，每筆抽一種 s，用 `gather` 做轉換，並 assert 結果和 `d4(x, s)` 相同。
- 成本：每步一次純記憶體的 gather（約 0.1 GB）。實測 fp32 每步約 151–154 ms（2,814 步，每個 epoch 約 432 秒），gather 不到 1%。
- 抽 s 用固定種子的 GPU generator，種子設為 SEED＋epoch。

**這是最主要的槓桿。** prereg02 的接著訓練（`results/prereg02/rank/ckpt/history.csv`）：

| | epoch 5 | epoch 9 |
|---|---|---|
| 訓練 accuracy | 26.3% | 33.9% |
| inner accuracy | 25.15% | 25.58% |
| inner loss | 1.930 | 1.980 |

差距在擴大，是在記住 fit 玩家。但題目層級的分數仍然有進步：R02 0.5365 → R04 0.5634（`tools_run.log:16-18`），因為校準吸收了過度自信。

## (3) 更長的訓練、LR、label smoothing、混合精度

- **LR**：AdamW lr 1e-3、wd 0.01（官方預設，第 14 格）。前 1,000 步線性 warmup，接 cosine 每步降到 0，沿用 prereg02 的 LambdaLR 寫法（`build_prereg02_rank_nb.py:994-1000, 1136-1140`）。
  - 總步數 T 要用 `expected_train_steps()`（`:1019-1042`）依實際的 DataLoader worker 數算出來，存進 state 的設定裡。prereg02 寫死 2,814 步的 assert（`:979, 1064-1072`）要換掉，因為 worker 數會變。
- **epoch 數**：12 個（約 33.8k 步），固定用最後一個 epoch，inner 只當監看。
- **neighbour label smoothing（R3B）**：q = 0.9·e_y ＋ 0.05·(e_{y−1} ＋ e_{y+1})，兩端把 0.1 全給唯一的鄰居，用 `F.cross_entropy(logits, q)` 一行就能做到。PREREG_02 §5（`:189`）評過這類序數 loss「證據弱」，所以只當第二候選。
- **bf16**：autocast 加 channels_last，權重維持 fp32，不需要 GradScaler。CPU 冒煙測試時關掉 autocast。
  - 逐盤輸出一律用 fp32，接縫核對才穩。
  - 目前 cuDNN 已經用 TF32（`config.json` 的 `tf32.cudnn=True`）。推算實際吞吐：訓練約 90 TFLOP/s，推論約 101 TFLOP/s（每盤前向 17.87 GFLOPs，`bench_colab_20260925.json`）。所以 bf16 預期快 1.3–2 倍，沒實測。
- **loader 要一起改**：
  - 官方 `decode_sample` 在 worker 裡 `.float()`（第 11 格），一個 batch 185 MB。改成傳 uint8（46 MB），到 GPU 上再轉型。
  - DataLoader worker 從 4 個（`build_rank_nb.py:875`）加到 16 個，G4 有 48 vCPU。
  - 官方的 `shardshuffle=1` 其實等於分片完全不洗（webdataset 1.0.2 的 `shuffle(1)`）。改用 `detshuffle`、傳 seed，順便解決 PREREG_02 rev 4 第 7 條「不能重現」的問題（`:233`）。

## (4) 架構

- **建議主幹不動**（256 通道 × 20 個 block），只把第一層改成 501 通道，計算量多 0.2%。
- 過擬合的證據（上面的表）不支持加寬或加深。每多一個 block 約多 0.85 GFLOPs，也就是多 5% 時間；通道 256 改 320 約多 56%。
- 有餘裕的選項：把 flatten 加 FC 的頭（`network.py:33-37`，2166→722）換成 global average pooling 頭。對旋轉比較友善、參數少 1.5M，但要在 notebook 裡另寫類別，而且效果不明。不建議放進主候選。

## (5) 要不要重抽特徵

**不用。** (1)(2)(3) 都在 GPU 上從現有分片算出來。
- AI Pro 帳號：從 prereg01 的備份還原（`build_prereg02_rank_nb.py:416-422` 的做法），實測 80 秒。
- Pro+ 帳號：Drive 上沒有備份，要重抽，實測 355 秒，備份再 6 秒。

**加長盤面歷史沒有理由（inner 診斷）**
- 32.7% 的 inner 棋 ≥ 250 手，只有 0.02% ≥ 360 手。
- 目前模型（8 種對稱平均）依手數的 accuracy：

| 手數 | 100–149 | 150–199 | 200–249 | 250–299 | ≥300 |
|---|---|---|---|---|---|
| accuracy | .226 | .251 | .263 | .263 | .276 |

- 長棋沒有比較差，開局另外由開局 n-gram 補。
- 真的要重抽（例如改成 360 個盤面、加氣或最後一手的層）：約 6 分鐘、Drive 約多 1.3–1.9 GB、特徵指紋改變、loader 解碼約多 44%。

## (6) G4 時間與運算單元

**依據**
- epoch：436–446 秒，含 inner 監看（`results/prereg0{1,2}/rank/timings.json`）。
- 準備：preflight、打包檔、`load_rows`、還原、`pick_workers`、解析驗證題，合計約 2.9 分鐘。
- 最後的 8 種對稱輸出：145 秒（inner 那份 103.7 秒）。
- G4 每小時 8.71 單元（二手實測，`docs/COLAB_RUN.md:33`）。

| R3A（12 個 epoch） | 每個 epoch | 合計 | 運算單元 |
|---|---|---|---|
| bf16 快 2 倍 | 3.7 分 | 約 50 分 | 約 7.2 |
| bf16 快 1.5 倍 | 4.9 分 | 約 64 分 | 約 9.3 |
| bf16 沒加速 | 7.4 分 | 約 94 分 | 約 13.7 |

- 在 Pro+ 重抽特徵：每種情況再加約 5 分鐘、0.7 單元。
- 每斷線一次：約多 10 分鐘、1.5 單元。
- R3B 同時跑：單元加倍（約 14–29），時間不變（Pro+ 使用者回報可以同時開 2–3 個）。
- 建議預留：一個候選 20 單元，兩個候選 35 單元。

**接續**
- 每個 epoch 存一次 `state.pt`：模型、optimizer、scheduler、global_step、epoch、history、各 RNG 狀態，以及 recipe 設定（worker 數、T、loss、增強、AMP）。設定不同就停下，沿用 `build_prereg02_rank_nb.py:1076-1103` 的做法。
- 斷線最多損失一個 epoch，4–7 分鐘。
- `.pth` 只留最後兩個（每個約 105 MB）。

**帳號**
- AI Pro 沒有背景執行，跑完前分頁要一直開著、電腦不能睡眠（`COLAB_RUN.md:188`），而且運算單元額度不明。
- Pro+ 可以背景執行最長 24 小時，但有使用者回報跑 1.5–3 小時被斷線（`COLAB_RUN.md:49-50`）。
- 建議在 **Pro+** 訓練：從零訓練不需要 prereg01／02 的權重。使用者要先把 `C10_bundle_v1.zip`（247 MB）上傳到 Pro+ 的 Drive 一次。

## (7) 逐盤輸出要存什麼，才能沿用本機的合併工具

| 檔案 | 內容 | 要和什麼一致 |
|---|---|---|
| `inner_keys.npz` | rank、game_id、player_id、color、label | 和 prereg02 的逐陣列相同（比陣列不比 SHA：npz 含時間戳）。`build_qset` 靠 (rank, game_id) 對齊（`p02_rank_fuse.py:778-792`） |
| `inner_p03_d4.npz` | logits float32 [73556, 8, 10]、`weights_sha256`、`syms` | `load_dump` 的格式（`:405-417`） |
| `val_keys.npz` | question_id、slot、color、parsed | 和 prereg02 的逐陣列相同 |
| `val_p03_d4.npz` | logits [19156, 8, 10]，解析失敗的整列 NaN | `check_val_logits`（`:572-579`） |
| `_DONE_*.json` | gpu、torch、cudnn、權重 SHA | `write_dump` 的格式（`build_prereg02_rank_nb.py:595-615`） |

- 逐盤輸出先做視角轉換、再做 d4，用 fp32。
- 選配：每個 epoch 存一份原方向的 inner 輸出（每份 2.9 MB），只當監看。

**`tools/p03_rank_fuse.py` 怎麼寫**
- import p02 這些函式：`basis`、`log_softmax`、`softmax`、`rule_b_np`、`pool_matrix`、`pooled`、`QSet`、`question_logits`、`fit`、`predict`、`metrics`、`PlayerBootstrap`、`load_dump`、`load_val_keys`、`check_val_logits`、`build_qset`、`read_question_meta`（`p02_rank_fuse.py:159-400, 405, 555-579, 778, 1065`）。
- `SOURCES`／`DUMP_S`／`COMBOS`（`:103-155`）、`cmd_tune`（`:795`）和 `cmd_apply`（`:1077-1198`）寫死在 p02 的來源和路徑上，p03 自己寫一份 tune／apply。
- 來源：
  - C5：p02e9，8 種對稱。
  - C6：p03，8 種對稱。
  - C7：16 個方向的 log_softmax 等權平均。不加新參數，`fit` 可以原樣沿用。
- 開局分數直接沿用已 commit 的 `opening_inner.npz`／`opening_val.npz`；假題沿用 `pseudo_inner.json`。
- 自我檢查：重算 R20 的 inner 交叉驗證，必須正好等於 0.5855（`tools_run.log:34`）。
- 送分閘門比 R20 不比 R00，例如 ≥ +0.010。
- 本機 CPU 每個組合約 2 秒。

## (8) 測試日工具要擴充的地方

**寫死的地方**
- `build_test02_rank_nb.py:48, 62-66, 124-130`：底稿、權重 SHA、逐盤輸出名稱。
- `t02_rank.py:114, 119-120, 478-490`：`p02e9_d4`、R20／R10。
- `official_log.py:63-66, 297-306`：方法標籤只認 R20／R10。

**要新增**
1. `colab/build_test03_rank_nb.py` → `test03_rank.ipynb`：底稿是 prereg03 的 notebook；三組題目（接縫、public、private），權重 SHA 釘死，接縫比 prereg03 的驗證題輸出。約 1 天，含冒煙測試和獨立審查。
2. `tools/t03_rank.py`：
   - `apply`：讀 test02 的 C5 輸出和 test03 的 C6 輸出，核對兩邊 key 相同、題目檔 SHA 相同；沿用 `t02_rank.py opening` 產生的 `opening_test.npz`；本機接縫比 prereg03 的 `rank_val_probs.csv`。
   - `fallback-fit`：選定組合拿掉開局項的版本（R10 的對應），11/04 前算好並 commit。
   - 約 1–1.5 天。
3. `official_log.py`：加 p03 的方法標籤，並補 PREREG 修訂紀錄。約 0.25 天。
4. 用 `t02_fake_tests.py` 做 dry-run、改 `TESTDAY.md`／`COLAB_RUN.md`：約 0.5–1 天。

**測試日的影響**
- 選到 C7 時，test02（AI Pro）和 test03 兩份都要跑成功，接縫也各有一個。
- 權重放 Pro+ 的話：要嘛測試集 zip 兩個 Drive 都放一份，要嘛使用者把 105 MB 的權重複製到 AI Pro 的 Drive（SHA 釘死）。

**時程**
- 登記加 builder 約 10/05 完成，Colab 約 10/06–10/10，打分約 10/12。
- test03 工具在 10/31 前完成，10/28–11/02 彩排，趕得上 11/04。

## 風險

- bf16 加速沒實測。
- G4 可能拿不到：退到 A100 的話只有 12 vCPU，DataLoader worker 最多開到 10 個。
- cuDNN 本身不確定，結果不能逐位元組重現，報告要寫明。
- 兩個帳號分工讓測試日多一個步驟。
- 整季打分預算：每場 4 次，已用 1 次（`PREREG_02:159`）。

## 診斷腳本與輸出

只用 inner、只讀 repo、只用 CPU：
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\p03\taskA\colour_diag.py`
- `…\p03\taskA\colour_diag.json`
- `…\p03\taskA\length_diag.py`
- `…\p03\taskA\length_diag.json`