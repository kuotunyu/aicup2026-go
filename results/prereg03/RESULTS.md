# 事前登記 03 結果

登記：`docs/PREREG_03_RECIPE_V2.md` rev 3（`af739a5`）。驗證規則：`docs/VALIDATION.md`（split v1 rev 2）。
數字來源：
- `rank/`：`tune_report.json`、`check_report.json`、`selfcheck.json`、`frozen_config.json`、`apply_report.json`、`decide_summary.json`、`check_submission.txt`。
- `rank_{a,b}/`：Colab 紀錄（`run_log`、`env`、`config`、`leak_check`、`timings`、`readonly_*`、`ckpt/history.csv`）。
- `unpack_rank_{a,b}.json`，以及 git log。

分數四捨五入到小數第 4 位；百分比、毫秒、分鐘照表中寫的位數；時間到秒。

## 結論

- **沒有過送分閘門，記為「未送分」：競賽二維持 prereg02 的 R20，本機驗證集沒有打分**。`score_ledger.csv` 沒有新增列；競賽二整季的打分次數仍是 4 次用了 1 次。
- 選法選中 **P01（R20＋A 臂）**。inner 5 折交叉驗證分等級平均 0.5869，比 P00（＝R20）的 0.5855 只高 +0.0014：
  - G1（≥ +0.010）沒過。
  - G2（配對 bootstrap 95% 下界 > 0）沒過：[−0.0053, +0.0081]。
  - G3（每題平均差 ≥ 0）過：+0.0039。
- 新模型和 R20 的比較：
  - **逐盤準確率**：兩臂的 inner 逐盤準確率 27.39%／27.30%，比 prereg02 epoch 9 的 25.58% 高。
  - **整題分數**（分等級平均）：
    - 只用 A 臂（P04）0.5854，和 R20 持平。
    - 只用 B 臂（P05）0.5890，比 R20 高 +0.0036。
    - 和 R20 合併的 P01、P02、P03 只比 R20 高 +0.0013 到 +0.0026。
    - 7 個組合裡最高的是 P06（兩臂合併、不含 R20）0.5891，也只比 R20 高 +0.0036，同樣到不了 G1 的 +0.010。
- 登記的其他判定：
  - 兩臂都可用（`check` status ok）。
  - `selfcheck` 通過：P00 和 R20 完全相同。記錄的差距是 0，登記容許 ≤ 1e-12。
  - rev 3 (c) 的單色規則沒有擋下送分：兩臂的單色條件都成立，驗證單色子題的 Δκ 也都 > 0（`score_blocked_single_colour` 是空的）。這次擋下送分的是閘門。
- **對測試日的意思**（PREREG_03 第 5 節）：
  - 03 不採用，測試日只照 `docs/TESTDAY.md` 跑 test02、上傳 R20；FAIL_OPENING 時用 R10，不用 `frozen_config.json` 裡 P01 的無開局版。
  - 不做 test03，C3、C4 截止點不適用；10/28–11/02 的彩排只有 test02。
- 留下來的東西：
  - 新來源的逐盤輸出（`rank_{a,b}/dumps/`）已 commit，照 PREREG_03 2.4 留給之後的登記。
  - 權重只在 AI Pro 帳號 Drive 的 `C10/results/prereg03/rank_{a,b}/ckpt/final.pth`，zip 裡沒有：
    - A 臂 SHA-256 `790b767c5425e262130d6c103c71e89238edb17f927fa48e2a2770db788e50ac`
    - B 臂 SHA-256 `8e88f6ec6cbaad5372e5d62607b99ece3d77dc7a20af86a9197553f1a6c59a42`
    - 這是運作上的保留，不要刪。

## Colab（`colab/prereg03_rank_{a,b}.ipynb`，AI Pro 帳號，G4）

| 項目 | A 臂 | B 臂 | 來源 |
|---|---|---|---|
| 帳號 | AI Pro（9/28 指定）。使用者沒有另外回報帳號，由這兩點確認：特徵從該帳號 Drive 的 prereg01 備份還原；唯讀指紋裡 prereg01 的 996 個檔，和 prereg02 在 AI Pro 上跑時記的完全相同 | 同左 | `run_log.json` 的 `features_source`、`readonly_before.json`、`../prereg02/rank/p01_readonly_before.json` |
| GPU／環境 | NVIDIA RTX PRO 6000 Blackwell Server Edition（G4）、48 vCPU、torch 2.11.0+cu128、cuDNN 91900 | 同左 | `env.json` |
| 執行時間（UTC） | 2026-09-28 15:40:36 – 16:23:13（2,557 秒）；`timings` total 2,575 秒 | 15:41:14 – 16:26:40（2,726 秒）；`timings` total 2,742 秒 | `run_log.json`、`timings.json` |
| 中斷、接續、停止 | 沒有：一次執行，`resume_count` 0，`status` completed | 同左 | `run_log.json` |
| 開始時 Drive 剩餘空間 | 189,972,316,160 bytes（約 190 GB；門檻 3 GB） | 189,972,299,776 bytes | `config.json` 的 `drive_free_bytes` |
| 訓練 | 33,768 步，第 12 輪跑到第 T 步停；bf16 每步平均 60.3 ms；第 300 步推估 37.9 分鐘（上限 150） | 33,768 步，第 24 輪跑到第 T 步停；每步平均 61.8 ms；第 300 步推估 42.7 分鐘 | `timings.json`、`ckpt/history.csv` |
| 最後一輪 inner loss／逐盤準確率 | 1.8374／27.39% | 1.8433／27.30% | `ckpt/history.csv` |
| 外洩檢查 | fit 5,767 人 720,307 盤、inner 640 人 73,556 盤，0 盤出現在本機驗證檔；fit、inner、驗證三組玩家互不重疊 | 同左 | `leak_check.json` |
| 唯讀檔 | 開始與結束時的指紋相同（prereg01 996 個、prereg02 54 個檔，`changed_files` 空） | 同左 | `readonly_{before,after}.json` |
| 結果 zip SHA-256 | `762bb9de…` | `d5daaca0…` | `unpack_rank_{a,b}.json` |

- **時間**：推估是 A 52–96 分、B 60–115 分（PREREG_03 1.4），實際都比下限短。推估假設 bf16 比 fp32 的實測每步 151–154 ms 快 1.5–2 倍（`design/plan.md`），實際每步約 60 ms。
- **運算單元**：1.4 原本要求開跑前看可用量 ≥ 50。9/28 改成由使用者自己補足額度（`1c49db7`），所以沒有記錄實際的可用量。這是運作上的選擇，不影響判定。
- **notebook 身分**：Colab 內無法得知 notebook 本身的 SHA，所以沒有記錄。以下間接證據確認用的是 `39ebccb` 的版本：
  - `config.json` 的差異說明逐字出現在已 commit 的 notebook 裡。
  - 兩份 notebook 從 `39ebccb` 之後沒改過。
  - 上傳副本的 SHA（`85e0c2ae…`／`aa3669ac…`）和 git 裡的相同。
- **B 臂的 inner 監看有兩次大幅波動**：
  - 逐盤準確率第 3 輪 12.10%（前一輪 17.98%）、第 11 輪 16.40%（前一輪 23.72%），下一輪就恢復（22.15%、25.22%），最後 27.30%。
  - 監看只記錄，指定用的是第 T 步的 `final.pth`，`check` 的地板線和執色檢查都通過。
  - A 臂只有第 2 輪小跌（20.52% → 19.37%）。

## 本機（`tools/p03_rank_fuse.py`，`.venv`：Python 3.12.13、numpy 2.5.3、scipy 1.18.1）

### check：新來源能不能用（PREREG_03 2.2）

| 來源 | inner 逐盤準確率 | 執色 Δacc | inner Δκ（95% CI） | 驗證 Δκ | 單色 inner Δκ（95% CI，只記錄） | 驗證單色子題 Δκ（只記錄） |
|---|---|---|---|---|---|---|
| C6A（A 臂） | 27.391% | +0.672 個百分點 | +0.0354 [+0.0263, +0.0449] | +0.0378 | +0.0274 [+0.0220, +0.0325] | +0.0363 |
| C6B（B 臂） | 27.302% | +0.623 個百分點 | +0.0272 [+0.0177, +0.0371] | +0.0290 | +0.0249 [+0.0206, +0.0302] | +0.0344 |

- 可用來源：C5（R20 的來源）、C6A、C6B。
- inner、驗證題的 key 和 prereg02 相同；權重 SHA-256 和 `config.json` 記的相同。

### selfcheck

- P00 的 inner OOF 分等級平均 0.5854702521055817，等於 R20。逐盤 log 機率、重擬合參數、驗證題機率也都和 prereg02 相同：記錄的差距都是 0，容許 ≤ 1e-12（`rank/selfcheck.json`）。

### tune：5 折交叉驗證（inner 評估輪 127,200 題，規則 B）

| 組合 | 來源 | 分等級平均 | 每題平均 |
|---|---|---|---|
| P00 | C5（＝R20） | 0.5855 | 0.5906 |
| **P01（選中）** | C5＋C6A | **0.5869** | 0.5944 |
| P02 | C5＋C6B | 0.5868 | 0.5929 |
| P03 | C5＋C6A＋C6B | 0.5881 | 0.5955 |
| P04 | C6A | 0.5854 | 0.5953 |
| P05 | C6B | 0.5890 | 0.5954 |
| P06 | C6A＋C6B | 0.5891 | 0.5980 |

- **選法**（PREREG_03 2.3）：分等級平均最高的是 P06（0.5891）。和它差 0.003 以內的有 P01、P02、P03、P05、P06，取清單中最前面的 P01。
- **閘門**（PREREG_03 2.4，只看選中的 P01）：

| 閘門 | 條件 | 結果 |
|---|---|---|
| G1 | P01 − P00 ≥ +0.010 | +0.0014，**沒過** |
| G2 | 配對 PlayerBootstrap（每級內重抽玩家 1,000 次）95% 下界 > 0 | [−0.0053, +0.0081]，**沒過** |
| G3 | 每題平均差 ≥ 0 | +0.0039，過 |

- **只記錄**（PREREG_03 2.4）：
  - 驗證集的推估標準誤 0.0051，採用機率 4.6%。
  - P01 − P00 的各級差值：D +0.0086、C −0.0133、B +0.0254、A +0.0088、1D −0.0117、2D −0.0164、3D +0.0089、4D −0.0223、5D +0.0139、6D +0.0123。
  - 盤數分箱（5–7／8–12／13–20）的預測分布：見 `rank/tune_report.json` 各組合的 `oof.per_n_bin`。
- **FAIL_OPENING 備案**（寫進 `frozen_config.json`，只在 03 被採用時才用得到）：P01 的無開局版 inner 0.5727，高於 R10 的 0.5687，所以選無開局版。
- **單色 inner 假題**（rev 3 (a)，只記錄；120,544 題，SHA-256 `45d0f14a…`）：
  - P00 0.5833（黑 0.5647、白 0.6017）；P01 0.5869（黑 0.5706、白 0.6030）。
  - P01 − P00：全體 +0.0036，95% CI [+0.0003, +0.0074]；黑 +0.0060 [+0.0014, +0.0104]；白 +0.0012 [−0.0031, +0.0055]。

### apply、decide、check_submission

- `apply` 用 P01 寫出驗證題機率（1,589 題，0 題沒有可解析的棋）。`apply_report.json` 的 `score_allowed` 是 false，原因只有 tune 的閘門沒過。
- `decide.py`：規則 A、B 不同的有 169 題。
- `check_submission.py`（帶題目檔）：`pred_B.csv`、`pred_A.csv` 都 OK（`rank/check_submission.txt`）。
- 更正：commit `93ce679` 的訊息寫 check_submission OK，但那一次沒給題目檔，實際結果是失敗（`no question files given`）。帶題目檔重跑的結果在 `dc87dd0`。這不影響任何判定。
- **沒有跑第 3 節的 H3R**。

## 注意事項與揭露（PREREG_03 第 8 節、rev 3 (e)）

- 選法、閘門、參數都在看到任何驗證分數之前寫死（`frozen_config.json` 在 `1218ba8` commit，早於 `apply`）。這次整個驗證集都沒有打分。
- 配方的選擇參考了只用 inner 的診斷和文獻，沒有在 inner 上調任何訓練超參數。
- B 臂的 200 盤上限、各級等量、T、warmup，是看過 fit 的中繼資料後寫死的。
- 開局的 M、α 和組合都是在 inner 上選的，inner 交叉驗證略為樂觀。
- 無開局版沒有在驗證集打分。
- 訓練不能逐位元組重現（cuDNN、bf16）。
- 單色紀錄（rev 3 (a)(b)）只記錄。rev 3 (c) 是判定規則，只可能讓 03 不上場、不打分；這次兩臂都沒有觸發。兩者都是看過 AIdea 官方資料說明後、跑 Colab 之前補的。
- 單色假題和混色假題抽自同一批 inner 玩家，兩者不獨立。
- 生成式 AI 協助的部分照規則揭露。

## 核對

- 2026-09-29，workflow `wf_1a536a83-909` 用 3 個獨立代理核對：
  - 數字逐一對來源檔。
  - 選法、閘門、單色規則、FAIL_OPENING 備案、`score_allowed` 從原始紀錄重算。
  - 程序對照 PREREG_03 第 9 節。
- 判定全部相符，沒有 blocker。依核對意見修正了：
  - 揭露補齊。
  - B 臂 loss 的四捨五入（1.8433）。
  - 結論的措辭。
  - 補上帳號、notebook 身分、Drive 空間、B 臂監看波動、權重 SHA、對測試日的意思。

## commit 順序

| commit | 內容 |
|---|---|
| `af739a5` | PREREG_03 rev 3（Colab 開跑前） |
| `1c49db7` | 帳號與跑法：AI Pro、A＋B |
| `e55b019` | 兩臂結果解開（調參前） |
| `25ab9e8` | check |
| `d12ccd0` | selfcheck |
| `1218ba8` | tune、`frozen_config.json`（apply 前） |
| `93ce679` | apply、decide |
| `dc87dd0` | check_submission（帶題目檔） |
