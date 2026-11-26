# TASK A：官方提交格式，對照 `check_submission.py` 與 `make_val.py`

**結論：** `check_submission.py` 已經能驗官方的「一個檔、pub＋priv 共 800 列」格式。欄名、標頭、等級拼法、每題剛好一次、缺題、top-5 不重複、不跨池，這幾項都查得到。還缺四件事，下一步是補上：

1. 沒有檢查列的順序。
2. 不給題目檔時會誤判為 OK。
3. 不會驗證傳進來的題目檔、候選檔真的是官方那一套（題數、前綴、兩池互斥）。
4. 參數格式錯時會直接丟出 traceback。

`make_val.py` 不能改，split v1 已凍結，改了會動到檔案雜湊。

這次沒有改 repo、沒有開 answers 檔。唯一跑過的 repo 測試是 `tests.test_tools.StrictCheck`（3 個，全部通過）。探針腳本都放在 scratch。

---

## 1. 官方提交檔（依據：tutorial commit 359e06f，9/24 抓取，`third_party/aicup2026_tutorial/SOURCE.txt`）

**每個子題只交一個檔，裡面 pub＋priv 共 800 列**
- README:186-194：每個子題只上傳一份；缺任何一份測試集的題目，整次提交無效。
- 檔名在 README:196-199 規定為 `submission_rank.csv`、`submission_player.csv`。兩份 notebook 的寫法也一樣：rank-prediction.ipynb:564，player-identification.ipynb:668。

**棋力預測（競賽二）**
- 欄位：`question_id,pred_rank`。寫法是 `pd.DataFrame(rows, columns=['question_id','pred_rank'])` 再 `to_csv(index=False)`（rank-prediction.ipynb:550-551）。
- 產出的檔案：有標頭、逗號分隔、沒有 index、沒有 BOM；在 Linux 上行尾是 LF。
- 等級要照 `LEVELS = ['D','C','B','A','1D','2D','3D','4D','5D','6D']` 的拼法（rank-prediction.ipynb:86；README:76 是對照表，D＝10~12k，依此類推）。
  - README:11 寫的「12k…6d」只是描述，不是可以填的值。
- 列的順序：`for split in ('public','private')`，每份依題目檔的列順序寫（rank-prediction.ipynb:522）。也就是 pub 400 列在前、priv 400 列在後。
  - README 說「串接起來」，但沒有明文說順序會影響計分。
- 一局都解析不了的題目，notebook 填 `LEVELS[0]`，也就是 `'D'`（ipynb:533）。其餘題目是各局 softmax 平均後取 argmax（ipynb:544）。

**玩家辨識（競賽一）**
- 欄位：`question_id,top_1..top_5`（player-identification.ipynb:654-656，`TOP_N=5` 在 :93）。
- 列的順序同上，pub 在前、priv 在後（:620、:644）。
- top-5 的取法：到重心 L2 距離做 `np.argsort`（:650），所以五個人天然不重複。
- 查詢一局都解析不了時，填 `names[:5]`，其中 `names = sorted(centroids)`（:627、:647）。
  - 只有至少一局解析得了的候選人會進 `names`。
- 規則（README:201、ipynb:557）：
  - `pub_`／`priv_` 前綴要原樣照抄。
  - 五個人選要不同。
  - 只能填該題所屬那一池的玩家。pub 題填了 `priv_p_` 的人，整次提交無效。
- 計分（README:242）：pub、priv 各自算，各自取 400 題的平均。

**官方 notebook 怎麼讀測試檔**
- 棋力預測（rank-prediction.ipynb:523-530）：
  - `pd.read_csv(tests/rank_prediction_test_{split}.csv)`。
  - `sgf_cols` 是所有以 `sgf_` 開頭的欄，依檔案欄位順序。
  - 每題只留 `pd.notna(v) and str(v).strip()` 的欄。
  - **不看 `num_games`，也不看 `color_*`。**
- 玩家辨識（player-identification.ipynb:623-641）：
  - 候選檔讀 `player_id`、`sgf_content`、`color`。
  - 題目檔的 `sgf_*` 和 `color_*` 依欄位順序兩兩 zip（:632-633、:638）。
  - 只用 sgf 欄判斷空值，也不看 `num_games`。
  - 執色先 `str(color).lower()` 才交給 parser（:577）。
- 題目檔欄位（README:89-94）：`question_id`（`pub_`／`priv_` 前綴，例：`pub_q_0001`）、`num_games`、`sgf_1..20`、`color_1..20`（值是 "B"/"W"）。每份 400 題，每題 5–20 局（README:85）。
- 候選檔欄位（README:96-105）：`game_id`（`pub_`／`priv_`）、`player_id`（`pub_p_`／`priv_p_`）、`sgf_content`、`color`。每池 400 人 × 100 局，兩池互斥（README:98）。
- **官方沒寫明的：**
  - 玩家辨識題的 question_id 用不用 `_q_` 中綴。
  - `game_id` 的後綴格式。
  - 是否一題對一位候選人（PREREG_02 的 T6）。
  - 候選檔裡各局的排列順序。
  - rank 題在各等級的分布。
- AIdea 規則頁（scratch 的 `comp1.html.txt:147,178`）只說答案要照格式上傳、結果「不得使用人為修正」，沒有給欄位細節。
- README:66 說正式定義以官方發佈為準，所以 11/04 應重新抓 tutorial repo，和 359e06f 做 diff。

## 2. `tools/check_submission.py` 對照官方

**已經驗到的**
- 標頭完全一致（:19、:48-50）。
- 等級值要正確拼法、區分大小寫（:17、:81）。
- 有 BOM 就報錯（:38-39）；CRLF 和 LF 都接受（:44）。
- 空值或前後有空白就報錯（:71）；欄數不對就報錯（:68）。
- 同一題出現兩次（:74）、不在題目檔的題（:77）、少了的題（:90-92）都會報錯。
- top-5 要剛好 5 個不同的人（:85）。
- 每一個人選都要在自己那份題目檔配對的候選池裡（:58-61、:87-89）。這比單看前綴還嚴格。

我用官方形狀的合成檔（400＋400 題、兩池各 400×100）跑過：
- 正確的檔、CRLF 的檔都通過。
- 標籤填 `12k` 會失敗。
- `--candidates` 前後對調時，800 列全部失敗，是安全的失敗方向。

**需要改的（探針已確認）**
1. **不檢查列的順序**：整份倒過來排的 800 列照樣回 `[]`。建議要求順序等於「各題目檔的列順序，依 `--questions` 給的順序串接」，也就是 pub 在前、priv 在後，和官方 notebook 一模一樣。
2. **不給題目檔會誤判 OK**：`check_submission.py rank hdr_only.csv`（不帶 `--questions`）的結果是 rc 0，印出 "OK … covers all 0 questions"。至少要有 1 份題目檔；官方模式要剛好 2 份。
3. **參數格式錯會 crash**：旗標之前出現位置參數時（:99-103，`cur` 是 None），會丟 `AttributeError` traceback。應改成印出用法、回 exit 1。
4. **沒有官方模式的結構檢查**，建議加 `--official`：
   - 剛好 2 份題目檔，各 400 列、question_id 不重複，第 1 份全是 `pub_`、第 2 份全是 `priv_`；總列數 800。
   - 候選檔各 40,000 列，400 個 `player_id` × 各 100 局；分別是 `pub_p_*`、`priv_p_*`，`game_id` 分別是 `pub_*`、`priv_*`。
   - 兩池沒有共用的 player_id 或 game_id。
   - 每個人選的前綴要和題目前綴一致，錯池時訊息更清楚。
   - 檔名的 basename 應為 `submission_rank.csv`／`submission_player.csv`（至少給警告）。
5. **錯誤訊息會誤導（小事）**：同一份題目檔裡重複的 question_id，:55-56 報的是 "appears in two question files"。
6. **測試還沒涵蓋上面幾項**：`tests/test_tools.py:97-141` 的 StrictCheck 已有 pub/priv 合成資料，但沒有測順序、零題目檔、400/800 題數、前綴一致。

**會碰到提交檔的其他程式（給接手的人）**
- `decide.py` 寫的是 `pred_A.csv`／`pred_B.csv`（:69、:222-227）：LF、欄位 `question_id,pred_rank`、順序照輸入的機率檔。
  - 官方需要的是單一份、pub 在前 priv 在後的 `submission_rank.csv`。
  - 缺題時的遞補值 `FALLBACK="D"`（:68）和官方 :533 一致。
- `p02_player_fuse.py apply` 一次只處理一組 candidates＋questions。預設是本機路徑（:1032-1034），輸出寫成 `player_val_top5.csv`（:970-976）。
  - 官方要依 PREREG_02 2.2 每池各跑一次（各自做 self-retrieval＋guard），再依 pub、priv 順序串成 `submission_player.csv`。
  - 缺題時的遞補（前 5 位排序後的候選人，:962-968）和官方一致。
- `p02_rank_fuse.py:1191` 的提示只帶一份 `--questions`。

## 3. 本機驗證檔和官方格式的差異

以 `make_val.py` 加結構探針確認；只讀了 questions 和 candidates，沒開 answers。

| 項目 | 本機（split v1） | 官方 |
|---|---|---|
| 棋力題 ID | `val_q_NNNN`（make_val.py:131） | `pub_q_0001`／`priv_…`（README:91） |
| 辨識題 ID | `val_pq_NNNN`（:168） | `pub_`／`priv_`，中綴沒寫 |
| 候選玩家 | `val_p_<16 hex>`（:156） | `pub_p_`／`priv_p_`＋雜湊（README:103） |
| 候選局 | `val_g_NNNNNN`，全域流水號（:165） | `pub_`／`priv_`，後綴沒寫（README:102） |
| 檔案數 | 棋力 1 份題目檔；辨識 1 份題目＋1 池 | 各 2 份（pub/priv），兩池互斥 |
| 棋力題數 | 1,589 題，自然分布（`results/split_v1/manifest.json`，例如 A 308、1D 87） | 400＋400，分布沒寫 |
| 辨識 | 400 題、400 人 × 剛好 100 局，一題對一人（:166-170） | 每池 400 題、400×100；一題對一人沒寫（T6） |
| 標頭 | 42 欄，順序和 README 表格完全相同；候選檔 `game_id,player_id,sgf_content,color` 也相同 | 同上，但官方沒保證欄位順序 |
| 空槽 | `""`（:69-72）；`num_games` 5–20，已填槽和 `num_games` 0 不一致 | README 說「只有前 `num_games` 個欄位有值」，notebook 用 notna＋strip 過濾，不信任 `num_games` |
| 執色 | 只有大寫 `B`/`W` | 寫 "B"/"W"；notebook 會先轉小寫 |
| 行尾 | CRLF（csv.writer 預設，:121-123、:142-145） | pandas 在 Linux 上應是 LF |
| 候選順序 | 依玩家排、同一人的局相鄰、玩家間照雜湊序（:151、:155-165） | 沒寫 |

**要一起注意的地方**
- 我們讀題目的程式都用 DictReader 依欄名讀，和欄位順序、行尾無關：
  - p02_rank_opening.py:147-165
  - p02_player_fuse.py:145-153
  - p02_rank_fuse.py:1065-1074
- `norm_colour` 只接受大寫 'B'/'W'（p02_rank_opening.py:69-72）。官方如果是小寫，會直接報錯停下，不會悄悄算錯。T1 要檢查這點。
- `p02_player_fuse.read_questions`（:145-153）照 `num_games` 讀 slot 1..n，但不檢查這些 slot 非空，也不檢查 n 之後的 slot 是空的。官方 notebook 反而完全不看 `num_games`。
  - 建議 T1 對每題 assert：非空 sgf 數 = `num_games`。prereg01 的 rank notebook 已有這個 assert（colab/build_rank_nb.py:1029-1030）。

## 4. 建議的演練方式（不動凍結檔）

在 scratch 用本機檔改前綴，做一套官方形狀的測試材料：
- 棋力：1,589 題的前 400 題當 `pub_q_`，接著 400 題當 `priv_q_`。
- 辨識：同一份題目＋池子，分別改成 `pub_`／`priv_` 兩套。內容相同也沒關係，目的是測「每池各跑一次、再串接、再 `check_submission --official`」這條流程接不接得起來。

## 產出的檔案（都在 scratch）
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\testkit\formatA\probe_check.py`
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\testkit\formatA\inspect_local.py`
- `C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go\943dc84e-3a07-4c6f-b22a-052eec94e7d1\scratchpad\player_nb.txt`、`rank_nb.txt`：兩份 notebook 的文字 dump