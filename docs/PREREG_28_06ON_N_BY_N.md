# 事前登記 28：06-ON 的尺寸條件放寬成 N×N（400 ≤ N ≤ 800，競賽一）

**白話版**：PREREG_06 5.3 的 06-ON（Hungarian 一題配一人）有一條字面條件「兩池都是 400 題、400 位候選」。官方 11/06 公告 11/11 重開競賽一、測試集是「原有＋新增」題目。v2 若仍是一題對一人、只是變大（例如 500×500），06-ON 會只因為字面被關掉（v1 上 06-ON 比 06-OFF 多 0.0113）。fit 玩家模擬（`results/prereg28/explore/`）證實凍結的偵測器和門檻在 N＝500–800 照樣可靠，所以本登記**只把那一條**改成「每池 N 題、N 位候選、400 ≤ N ≤ 800」。偵測器、門檻、其他條件、public 否決、上傳順序一字不改。**在 v2 格式公布（11/11 11:00）之前登記並合併。**

- 決定者：C10 主 session 派的代理（使用者 10/2、10/9、10/16 把對排名有幫助的方法決定交給 AI）。
- 只做競賽一（TEAM_11249）。

## 修訂紀錄

| 版本 | commit | 內容 |
|---|---|---|
| rev 1 | （本檔第一次 commit） | 2026-11-06，分支 `p28-explore`。commit 時：探索已判（PLAN `c7832dc`、eval `f2b1032`，通過）；`tools/t28_n.py`、`tests/test_t28_n.py` 還沒 commit；v2 還沒公布，repo 裡沒有任何 v2 檔。先 commit 本檔，之後才 commit 工具。 |
| rev 2 | （本列所在的 commit） | 2026-11-06，合併前：獨立審查的修正、演練的數字、補充限制（第 10 節）。v2 還沒公布。 |

之後的修訂只准加行。

## 0. 證據（`results/prereg28/explore/RESULTS.md`，數字引自 `eval_report.json`）

- 接縫：本探索的程式在 N＝400 和 v1 校準（`results/prereg06/design/switch_report.json`）逐值相同。
- G1：N＝500／600／700／800，V2 的 E0、PA 兩種來源，PREREG_06 2.6「能不能用」四條全過；非一對一 6 種結構 × 20 池誤開 0（v1 的校準率也是 0／120）。V3 每個 N 也都過。
- G2：一對一池照偵測器開關 − 06-OFF（CSLS10）：+0.01067、+0.01310、+0.01340、+0.01336，95% 下界都 > +0.0085。
- G3：「原有 400 題＋m 題新增」（新增題的玩家不在候選裡／原有玩家出第二題／各半；m＝40、100、200）180 池，V2、V3 六個鍵全部 0 次打開；r 最小 4.78（門檻 2.25）。

## 1. 改的是什麼（唯一的改動）

PREREG_06 5.3（和 rev 3 第 17 條 (c)）的「兩池都是 400 題、400 位候選（06 自己數的和 test02 記的都要是）」改成：

> **每一池**（public、private 各自）06 自己數的題數、候選數，和 test02 記的題數、候選數，四個數是同一個整數 N，而且 **400 ≤ N ≤ 800**。兩池的 N 可以不同。

- N＝400 時和原文完全相同；N 在範圍外（＜400 或 ＞800）、或題數 ≠ 候選數、或 06 和 test02 數的不同 → 這一條不成立，06-ON 不做（和原文相同）。
- **範圍為什麼到 800**：模擬需要每位候選 100 盤黑棋＋題目最多 20 盤，fit 玩家中只有 926 人夠；N＞800 沒有模擬過，照原文不開。
- 條件的名稱（`pools_400_by_400`）不改：`tools/official_log.py` 依名稱核對 06-ON 的報告，所以照舊用這個名稱，`detail` 寫明是 PREREG_28 的規則。

## 2. 不改的東西

- 偵測器（PREREG_06 2.6：r_forced_cost＝這一池 forced_cost ÷ 假題池第 0、1 輪的平均）和凍結門檻 r*_X（`results/prereg06/player/frozen_config.json`：V2 2.2515761243137895、V3 2.2678004868728294；V0 不能用）。不重新校準。
- 06-ON 的其他條件：private 選中的組合能用且偵測器打開、`h6p_secondary_not_negative`、T2(c)＝0 且 (a)(b)(d) 沒有不過、兩池都用 06 且在第 1 層、AIdea 頁面沒寫不是一對一（`--aidea-not-one-to-one`）、K7' 之前。
- 接縫（seam_val，本機驗證集，400×400）、public 否決（v2 的 public(06-ON) − public(06-OFF) < 0 → 再傳 06-OFF）、上傳順序（PREREG_27 第 4 節）、最後一版的規則（PREREG_06 5.4／PREREG_10 4.4／PREREG_21）。
- `tools/t06_player.py`、`tools/official_log.py` 和其他凍結檔一個位元組都不改。

## 3. 和其他登記的關係

- **PREREG_27 第 7 節**寫「rev 3 不准改 06-ON 的條件（包括 400 題、400 位候選的字面）」，第 4 節寫「v2 不是 400×400 時 06-ON 不做」。那兩句管的是**看到 v2 格式之後**的修訂。本登記是看到格式之前的獨立登記，**明文修訂那兩句，只限第 1 節的尺寸條件**：v2 是 N×N、400 ≤ N ≤ 800 時，06-ON 照 PREREG_06 5.3 的其他條件和順序做。PREREG_27 第 7 節的其他禁止（模型、特徵、自我檢索與閘門、CSLS、其他 06-ON 條件、否決規則、門檻……）照舊。
- **順序**：本登記和工具在 11/11 10:00 前合併。11/11 若格式和 v1 不同（例如每池 500 題），照 PREREG_27 第 7 節先寫 rev 3（只改結構常數：T1.2、test02、check_submission 等認得新的題數），T1 過了、test02 和 06 照常跑；本登記的尺寸條件在 06 的 `submit` 時才判。rev 3 不需要、也不准再碰 06-ON 的尺寸條件。
- **PREREG_10 4.4**（06-ON 沒產生、被否決……時傳 10）和 PREREG_27 10.5（B′ 的 P1 在 build 好就先跑）不變：06-ON 有沒有產生，現在多了「N×N、400–800」這種可能。
- PREREG_06 第 5.5 節的例外（public 只用來把開關關掉、一季最多一次 06-ON）照舊適用於 v2。

## 4. 實作（`tools/t28_n.py`；`t06_player.py` 是凍結的，不改）

- `PY tools/t28_n.py submit <和 t06_player.py submit 完全相同的參數>`：載入 `t06_player`，把它的 `on_conditions` 換成包裝版，再呼叫 `t06_player.main`。其他指令（feats／seam／apply／veto／status）照舊用 `tools/t06_player.py`；`t28_n.py` 只接受 `submit`。
- 包裝版：先跑原本的 `on_conditions`。`pools_400_by_400` 原本就成立（400×400）→ **整個結果原樣不動**（所以 N＝400 時寫出的檔、報告和直接跑 `t06_player.py submit` 逐位元組相同）。原本不成立 → 這一條換成第 1 節的規則，`detail` 記原本的結果（`literal_400_by_400`）、規則、每池四個數和 N、本登記和工具的路徑。其他條件原樣。
- `V3_USABLE_ALL_N = True`（探索的 `V3_usable_at_every_N`）：PLAN 第 5 節「V3 在某個 N 不能用 → N ≠ 400 選中 V3 的池不開」的分支保留在程式裡，但這一季不會觸發。
- real mode（沒有 `--dry-run`）時，`t28_n.py`、本檔、`results/prereg28/explore/eval_report.json` 都要已 commit 而且乾淨，否則結束碼 2。
- `official_log.py` 不改：它依名稱核對 `on_conditions` 全部為真、`on_produced` 為真，包裝版的報告照樣通過。

## 5. 測試日（v2；接在 `docs/TESTDAY_27_V2.md` 和 `docs/TESTDAY_06.md` 第 5 步之後）

- `docs/TESTDAY_06.md` 第 5 步的 `PY tools/t06_player.py submit --aidea-page-checked "…"` **改成** `PY tools/t28_n.py submit --aidea-page-checked "…"`（參數相同；400×400 時結果和原指令相同，所以一律用這一個）。
- `--aidea-page-checked` 要抄 v2 的頁面原文（題數、候選數、有沒有寫新增題的玩家在不在候選裡、同一玩家可不可以多題）。頁面寫了不是一題對一人、或 Claude 沒把握 → 照舊加 `--aidea-not-one-to-one`，06-ON 不產生。
- submit 之後讀 `results/test06/player/off/t06_submit_report.json` 的 `on_conditions.pools_400_by_400`：`detail.prereg28.pools.<池>.n` 就是那一池的 N；寫進 RESULTS。
- 06-ON 產生之後的上傳、否決、K7'、最後一版全部照 `docs/TESTDAY_06.md` 和 PREREG_27 第 4 節。

## 6. 防誤改和閘門

- `tests/test_t28_n.py`：尺寸規則（範圍內外、題數 ≠ 候選數、06 和 test02 不同、非整數）、400×400 時和原版逐值相同、其他條件不受影響、V3 分支、`official_log` 的條件名稱、`t06_player.py` 自 `testset-v1-final` 沒改、CLI 只接受 `submit` 且真的把包裝版裝進 `t06_player`。
- 不新增允許失敗：本登記沒有改任何凍結檔。閘門照 AGENTS.md（加跑 `tests/test_t28_n.py`）。

## 7. 演練

- PREREG_27 的假 v2（`tools/testset_v2.py fake`）：`same`（每池 400×400）和 `grow-player`（每池 500×500）；工具 `results/prereg28/rehearsal/rehearse_t28.py`，紀錄 `results/prereg28/rehearsal/rehearsal_t28.json`（數字見第 8 節）。題數、候選數照 06／test02 的數法從檔案數出來，其他條件設成成立，比較原版和包裝版。
- 完整的測試日鏈（T1 → test02 → 06）在這兩棵樹上**跑不了**：T1.4／T1.5 和 06 的接縫要 `data/local`（不開），test02 要 Colab 的 CNN；N＝500 時 T1.2 判 FAIL_STRUCT，要等 PREREG_27 rev 3（PREREG_27 10.2 已演練過）。偵測器本身在 N×N 的行為由第 0 節的模擬涵蓋。

## 8. 已知限制和風險（揭露）

- 模擬只用 fit 玩家；大 N 的池彼此重疊很多（N＝800 任兩個 seed 至少共用 684 人），門檻也是在同一批人上選的；CNN 樣本內、LDA 在大 N 較弱；N＞400 的等級比例和官方不同。
- N ≠ 400 時，「兩池都在第 1 層」要靠 test02 的防護在新 N 上判第 1 層（T3 的開局重疊有用到 400 的地方）；沒判第 1 層就不開（偏保守），本登記不改它。
- 像 v2 的結構只模擬了三種；其他非一對一結構靠標準結構的 G1、AIdea 頁面條件和 public 否決。
- 範圍外（N＜400、N＞800、N_q ≠ N_c）照原文不開。

## 9. 揭露（報告）

- 寫明：v2 公布前登記了 06-ON 尺寸條件的放寬（本登記），根據是 fit 玩家模擬；偵測器和門檻沒改；用到 06-ON 時，v2 每池的 N。

## 10. rev 2：審查、演練、補充（2026-11-06，合併前）

### 10.1 演練（`results/prereg28/rehearsal/rehearsal_t28.json`；第 7 節「數字見第 8 節」指的是這裡）

| 假 v2 | 每池（題／候選） | t06 原版 | t28 包裝版 | 和原版逐值相同 |
|---|---|---|---|---|
| `same` | public 400／400、private 400／400 | 06-ON 條件全成立 | 全成立 | 是（原樣不動） |
| `grow-player` | public 500／500、private 500／500 | 尺寸條件不成立 → 不做 | **全成立**（N＝500） | 否（只有尺寸那一條） |
| 兩棵樹各自 private 多 1 題（`q_extra`）或多 1 位候選（`c_extra`） | — | 不成立 | 不成立 | — |

### 10.2 獨立審查（另一個 Claude 子代理，唯讀，`4a136d7`）

沒有 blocker；2 major、6 minor。它核對過：包裝版在 400×400 回傳 t06 原本的結果；任何不在登記範圍的情形（非整數、06 和 test02 數的不同、一池在範圍內另一池不在、題數 ≠ 候選數）都不開；`cmd_submit` 用模組全域的 `on_conditions`，包裝真的生效；`N_OFFICIAL` 只用在 `on_conditions`；veto、status 不重算條件；`official_log.py` 只看每個條件名稱的 `ok`、`all_on_conditions`、`on_produced`；`t10_player.py` 只看 `on_produced`；偵測器（假題池用這一池自己的候選）對 N 沒有寫死的假設；第 0 節的數字和 `eval_report.json` 相符。

處理：

1. （major）PREREG_27 第 4、7 節原文仍寫「不是 400×400 就不做」：PREREG_27 加了第 11 節（只加行）指向本登記；`tools/testset_v2.py` 的說明字串不改（以 PREREG_27 第 11 節為準）。
2. （major）操作頁沒換：`docs/TESTDAY_27_V2.md` 第 6 節加一行（v2 上 submit 一律用 `t28_n.py`，不要再用 `t06_player.py submit`，因為它會照字面刪掉 `on/` 的 06-ON）。`docs/TESTDAY_06.md` 不改（在 06 的指紋裡）；本檔第 5 節和 TESTDAY_27_V2 那一行優先。AGENTS.md 加一行。
3. （minor）稽核：尺寸條件被換掉時，`detail.sha256_lf` 記 `tools/t28_n.py`、本檔、`eval_report.json` 的 SHA-256（CRLF 折成 LF）。
4. （minor）審查說 G3 的結構「題數 ≠ 候選數、尺寸規則本來就擋」：**不對**，G3 的池是 N 題＝N 位候選（m 位候選沒有題目），正是尺寸規則會放行的方形非一對一池；不改。
5. （minor，補限制）`sr06_rounds` 要每位候選剛好 100 盤（`strict=True`）。v2 的候選盤數若不是 100，06 的 apply 會停（結束碼 2），06（含 06-ON）這一季不用；PREREG_27 第 7 節不准改自我檢索，所以那種情形照各登記退回保底。
6. （minor）閘門：`tests/test_t28_n.py` 加進 AGENTS.md 的閘門說明（AGENTS.md 加的那一行）。
7. （minor）測試補上：一池 400 一池 500、`None` 的報告、400×400 的多種情形都和原版逐值相同、`official_log.t06_report_problems` 接受包裝版 N＝500 的條件且擋 N＝900、重複呼叫 `main` 不會包兩層、real mode 要求檔案已 commit。
