# prereg12 探索：KataGo HumanSL 微調試跑（競賽二）

2026-10-12 寫成，**在任何 GPU 執行之前**。只用 fit／inner 訓練玩家；不碰 `data/local`、不讀任何驗證題特徵（`data/prereg08/val`、`data/prereg10/val`、`results/prereg0x/feat/val_*`），不跑 `tools/score.py`。本檔的閘門在試跑前寫死，試跑後只判一次。標「估計」的數字都還沒實測，GPU 的實際速度由 `bench.json` 量。

## 0. 白話版

- 現在競賽二的首選 Q09＝Q02H＋4·P1TO_E：KataGo HumanSL 網路**不動**，只讀它最後一層的內部表示，每盤取平均，再用 6 個小模型翻成等級。inner 0.7114（單色 0.7145）。
- 這次問的是：**讓網路最後幾層自己學「等級」**（微調），能不能比「只讀、不動」多拿到明顯的分數。
- 試跑只花 4090 一晚（估計 5–7 小時，上限 12 小時）。過了閘門才值得寫 PREREG_12；沒過就收掉這條線。
- 競賽二已經沒有驗證集打分機會。就算過關，採用也只能像 09 一樣靠 inner 閘門，而且要新做一條測試日路徑。

## 1. 三個做法的比較和選擇

| 做法 | 要什麼 | GPU 成本 | 預期 | 結論 |
|---|---|---|---|---|
| (a) 解凍最後 k 個主幹 block＋新的等級 head，每盤彙整 | 重新前向所有訓練盤和 inner 盤 | 反傳只過 k 個 block | 讓表示本身往「等級」調 | **選這個** |
| (b) 主幹凍結，學「怎麼彙整每一手」（attention／learned pooling） | 每個局面的 384 維特徵 | 抽取成本同 (a) 的推論，訓練便宜 | 小 | 不做 |
| (c) LoRA adapter 掛在每個 block | 同 (a) | 反傳要過整個主幹 | 不確定 | 不做 |

- **(b) 不做的理由**：
  - 現有的塊只存每盤平均（`means [G, 2, 770]`），每個局面只有 `pos_logp`、`pos_entropy`（10/12 讀 `data/prereg08/p1/probe/chunk_00000.npz` 確認）。要做 (b) 還是得把 probe 和 inner 的每個局面重算一遍、存 384 維（約 1.5 KB／局面，數十 GB），GPU 成本和 (a) 的推論一樣。
  - 「比平均更聰明的彙整」已經被試過兩次都沒用：10/5 的 `hs_rich_fitmap_cv.py`（分盤面階段、SD、分位數都沒贏過平均）；09 的 `lowrank3_ex` 已經吃了分位數和分階段平均（EX 44 維），單獨只有 +0.0076。
- **(c) 不做的理由**：反傳要走過整個 18 個 block，10/5 的 CPU 實測同類做法（只學 metadata、梯度穿過凍結主幹）每步是推論的 3.05 倍；同樣的 GPU 時間只看得到約一半的棋。改「最後幾層的表示」這件事，(a) 已經直接做到。
- **(a) 的細節**：k＝4（共 18 個 block，CPU 實測確認），加上最後的 trunk norm，可訓練參數 6,026,920（實測，含 head）。反傳只過最後 4 個 block，估計每個局面的訓練成本約推論的 1.3–1.6 倍（估計）。

## 2. 模型（寫死在 `ft_lib.CONFIG`）

- 網路：`data/katago/b18c384nbt-humanv0.ckpt` 的 `model` 權重，經 `hs_lib.load_model`；中性 metadata（`p1_lib.neutral_table`，雙方等級未知），同 P1。
- 前向停在 `model.act_trunkfinal`（hook 丟例外，policy／value head 不跑）。整個 KataGo 網路一直是 eval 模式（norm 用原本的統計值），只有最後 4 個 block 和 `norm_trunkfinal` 的參數會更新。
- 新 head（每個局面 → 每盤）：
  - 每個局面：[實際落點的 384 維, 全盤平均 384 維] → LayerNorm → Linear(768, 256) → GELU。
  - 每盤：目標方各手的平均（256）、對手方各手的平均（256）、有沒有對手手（1）、目標方顏色 one-hot（2）→ Linear(515, 10) → 等級 logits。和 P1TO 一樣同時看兩方。
- 損失：每盤的 cross entropy（目標方的等級）。題目計分（完全猜中 1、差一級 1/e、其他 0）留給融合的規則 B 處理，和 08／09 相同。
- 訓練設定：
  - 每盤抽目標方 16 手＋對手方 16 手（不足就全取），每批 8 盤（256 個局面；bench 遇到 OOM 會自動減半）。
  - AdamW：head lr 1e-3（wd 1e-2），主幹 lr 2e-5（wd 0）。
  - 前 1,000 步只訓練 head（主幹 lr＝0），之後主幹 1,000 步線性暖身；兩者都 cosine 降到 10%。梯度裁剪 1.0。
  - fp16 autocast＋GradScaler（可訓練參數保持 fp32）。
  - 1 個 epoch；訓練時間上限 3.0 GPU 小時（`bench.json` 的速度 × 3 小時 × 0.9 決定總步數，硬上限 3.3 小時）。
  - 每 2,500 步在 holdout 上評一次，holdout 的每盤 log-loss 最低的那一版存成 `best.pt`（選版本只用 fit 玩家）。

## 3. 資料

- **訓練（只用 fit 玩家）**：`tools/c10data.load_rows(part="fit")`（720,307 盤、5,767 人）。
  - 每級把玩家打亂（`random.Random(20261012)`），前 5% 的玩家當 holdout，其餘當訓練。
  - 每級 round robin 輪流從每位玩家抽一盤，訓練 20,000 盤／級（共 200,000 盤）、holdout 300 盤／級（共 3,000 盤）。
  - 落子序列和任何 inner 盤相同的 fit 盤一律排除（記數）。
  - 輸出 `data/prereg12/sets.json`，SHA 和計數寫進本資料夾的 `sets_record.json`。
- **inner 評估盤**：P1 的 bank（11,813 盤）和 mono（61,356 盤），用 `p1_extract.set_games` 的同一份清單和順序，**每一手、兩方都算**。
  - 特徵化和 P1 逐位元相同（`test_ft.py` 驗證）；inner 的玩家是訓練玩家，不是驗證玩家。
- 快取（CPU 先做，GPU 時不用等特徵化）：`data/prereg12/cache/<set>/chunk_*.npz`，每個局面約 1.1 KB（22 個平面用 packbits 壓成 993 bytes）。

## 4. 步驟和估計

| 步驟 | 指令（`S=results/prereg12/explore/finetune`） | 估計 |
|---|---|---|
| 1 sets（CPU） | `$S/ft_pilot.py sets` | 約 1–2 分鐘（smoke 實測 69 秒，含讀全部 fit／inner） |
| 2 快取 train／holdout（CPU） | `cache --set train`、`--set holdout` | 640 萬＋10 萬個局面，16 workers 約 1–1.5 小時（估計；smoke 4 workers 約 470 局面/秒）；約 7 GB |
| 3 快取 bank／mono（CPU） | `cache --set bank`、`--set mono` | 1,610 萬個局面，約 3–4 小時（估計）；約 18 GB |
| 4 bench（GPU） | `bench --device cuda` | 約 5 分鐘；量推論和訓練的局面/秒、記憶體 |
| 5 train（GPU） | `train --device cuda` | ≤ 3.0 小時（硬上限 3.3）；照 1 個 epoch 25,000 步估約 1.5–2.5 小時 |
| 6 extract（GPU） | `extract --set bank`、`--set mono` | 1,610 萬個局面，估 1,300–1,700 局面/秒 → 2.6–3.4 小時 |
| 7 eval（CPU） | `$S/ft_eval.py prep-inner`、`eval` | 約 5 分鐘（smoke 實測：Inner 58 秒、全部約 3 分鐘） |

- **GPU 合計估計 5–7 小時**（≤ 12 小時預算）。估計的根據：P1 實測 1,050–1,190 局面/秒（含 CPU 餵資料，GPU 忙碌約 0.8）；07 bench 1,805 次/秒（fp16）；10/5 CPU 的訓練／推論比。都還沒在 4090 上量過。
- CPU 步驟可以在白天先跑（不用 GPU）：`run_pilot.sh cpu-only`。
- 發射器：`data/prereg12/run_pilot.sh`（轉呼叫本資料夾的 `run_pilot.sh`），用 PowerShell 獨立背景程序啟動：
  `Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" -ArgumentList "/d/AI-Portfolio/competitions/aicup2026-go/data/prereg12/run_pilot.sh" -WindowStyle Hidden`
  - 只在本資料夾已 commit 而且沒有本機修改時才跑。
  - GPU 空閒的條件：`hs_lib.other_compute_processes`（nvidia-smi 的 Type `C`）和 `nvidia-smi --query-compute-apps` 的 python 列都是空的；每 5 分鐘查一次。每個 GPU 步驟在任何 CUDA 呼叫之前自己再查一次，有別的運算程序就不開始（exit 3）。
  - 讓出 GPU：建 `data/prereg12/STOP`（做完手上這一塊／這一段就停）；刪掉再啟動就接著跑（塊、checkpoint、報告都會沿用）。CUDA OOM（例如別的工作搶記憶體）會等 5 分鐘從 checkpoint 接著跑，連續 3 次失敗才停。
  - 10/24 前 C4 優先：C4 要用就建 STOP。
- 紀錄：`data/prereg12/run_pilot.log`、`run_log.jsonl`；訓練 `data/prereg12/train/`（`bench.json`、`ckpt_last.pt`、`best.pt`、`train_done.json`）；inner 輸出 `data/prereg12/ft/<set>/`。

## 5. 試跑閘門（現在寫死，試跑後只判一次）

- **候選（唯一）**：Q09＋FT。融合項 OP、O4、SWA、SWB、HS、P1TO（＝4·lp_P1TO_E，09 凍結的 `p1e_map.npz`）、FT（＝4·lp_FT，題內可用盤的遮罩平均）。融合照 08／09：p04 的廣義校準、所有權重 (0, 4)、5 折玩家 OOF、規則 B。程式：`ft_eval.py` 呼叫 `tools/p09_rank_fuse.Inner`（import，不改）。
- **對照**：Q09（inner 0.7114420614163589、單色 0.7145431684624712）。
- **先過接縫，否則停**：Q02H、Q08、Q08 單色等於 `p09_rank_fuse.REF`；Q09、Q09 單色等於 `results/prereg09/rank/tune_report.json`；相差都 ≤ 1e-12。（10/12 smoke 已逐位元重現這五個數。）
- **可用盤**：FT 的可用盤（沒失敗、有目標手）要和 P1 的可用盤完全相同，否則停下查。
- **閘門（全部成立才算過）**：
  - G1：inner 評估假題的分等級平均差（Q09＋FT − Q09）≥ **+0.010**。
  - G2：同一差值的配對 PlayerBootstrap（每級內重抽玩家 1,000 次，`default_rng(0)`）95% 下界 > 0。
  - G4：單色假題（全部 120,544 題）的差：配對 PlayerBootstrap 95% 下界 > 0，而且黑題、白題的差各自 > 0。
- **只記錄、不能救候選**：Q09＋FT ×1、以 FT 取代 P1TO_E（Q02H＋4·FT）、FT 對 Q02 的方向 ρ、每折權重、holdout 曲線。
- **時間到的規則**：訓練被 3 小時上限截斷，照樣用 `best.pt` 評估，不延長、不加第二組設定。
- **沒過**：記錄後收掉這條線（這一季不再試別的 k、學習率或 LoRA）；Q09 照舊。
- **過了**：只代表「值得寫 PREREG_12」，不是採用。要做的事：
  - 寫 PREREG_12，登記一個候選和 inner-only 的採用規則（同 PREREG_09：競賽二的 4 次驗證集打分已用完）；偏離 PREREG_08 5.4「不交沒在驗證集打過分的組合」要在報告寫明。
  - 驗證題只做無標籤檢查（FT 抽取、方向 ρ、格式），步驟照 09 的 apply／獨立稽核。
  - 新的測試日路徑：4090 上用 `best.pt` 抽測試題（估計和 08 的 P1 同級，約 0.5–1 小時）、新工具和測試、指紋；要趕在 11/04 測試日前，否則不採用。
- **已知的偏差**：inner 已被 P1、08、09 重複使用，有約 0.003 的選擇偏差（09 的揭露）；+0.010 的門檻留了餘裕。smoke 用「Q09 加雜訊」當假 FT（沒有新資訊）得到 +0.0005、G2 下界 −0.0026，閘門不會被白白通過。

## 6. CPU smoke（10/12，沒有用 GPU；輸出在 `data/prereg12/smoke/`，不是結果）

- `ft_eval.py prep-inner`：73,169 列（＝`INNER_USABLE`），塊 SHA 對 08 的 manifest。
- `ft_eval.py eval --fake-ft --out-root data/prereg12/smoke`：五個接縫逐位元相同；Q09＋假 FT 0.71192（+0.0005），閘門不過（預期）。
- `test_ft.py`（`C10_FT_REAL=1`，7 個全過）：
  - `featurize_turns` 和 `p1_lib.featurize_game` 逐位元相同（全部手、抽樣手）；packbits 來回相同（numpy、torch）；
  - 真的 HumanSL 權重在 CPU fp32：`FTNet.trunk` 的落點＋全盤平均和 P1 的 `ProbeScorer` 特徵相差 ≤ 1e-5；共 18 個 block；
  - 一步訓練：只有最後 4 個 block、`norm_trunkfinal`、head 有梯度，其他參數一位元都沒變，KataGo 網路維持 eval 模式；
  - bank 快取的手數等於 P1 bank 塊；FT 輸出和快取對得上。
- 管線（`--tiny` 是 CPU 用的小替身網路，只驗流程）：sets → 快取 → bench → train（STOP 後從第 6 步接著跑到完）→ extract；真權重在 CPU 上也跑過 3 步 train 和 bank extract。
- 發射器 smoke：見 `smoke_report.json`。

## 7. 檔案

- `ft_lib.py`（共用）、`ft_pilot.py`（sets／cache／bench／train／extract／status／gpu-free）、`ft_eval.py`（prep-inner／eval）、`test_ft.py`、`run_pilot.sh`、`sets_record.json`（sets 之後）、`eval_report.json`（閘門，試跑後 commit）。
- 大檔都在 `data/prereg12/`（不進 git）。只 import、不改：`third_party/katago`、`hs_lib.py`、`p1_lib.py`、`p1_extract.py`、`tools/p09_*.py`、`tools/c10data.py`。
