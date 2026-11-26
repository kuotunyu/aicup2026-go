# prereg31：訓練前的獨立審查紀錄（PLAN 第 6 節第 4 步）

2026-11-10，訓練開始之前 commit。審查者：主 session 另派的唯讀子代理（Claude），只看程式、單元測試和 smoke 的接縫紀錄，不看任何結果，沒有改任何檔。審查時分支 HEAD 是 `b20fa1f`。

## 結論

沒有 blocker。1 項 major 是操作面的（不是程式錯），5 項 minor；**程式一行都沒改**。

| 編號 | 等級 | 內容 | 處理 |
|---|---|---|---|
| F1 | major（操作） | `--resume` 要求和 `train_started.json` 同一個 commit；這個 worktree 還有 prereg32 的代理在寫檔，訓練中途有人 commit 就接不回去 | 訓練開始到 `train_record.json` 寫出之前，`p31-explore` 不 commit 任何東西；prereg32 的檔在訓練結束後才 commit |
| F2 | minor | 訓練之後再改 `p31_lib.py`／`p31_train.py`，eval 的接縫 M 會失敗 | 訓練後不動這兩個檔；eval 前把 `train_record.json`、`stage0b.json` commit 好 |
| F3 | minor | 啟動時若留著 STOP 檔，第一次啟動會被用掉 | 啟動前確認 STOP 不存在 |
| F4 | minor | 接縫 R 之後有一段 RAM 尖峰（兩份特徵各約 2 GB），可用 RAM 太少時監看會放 STOP | 可用 RAM 充裕才啟動；啟動後確認監看有第一筆紀錄 |
| F5 | minor | 啟動腳本：結束碼只寫在 log 的 `end rc=`；監看程序若一開始就死掉訓練會沒有保護；`run_eval.sh` 沒有 RAM 監看、不傳參數 | 不改；啟動後看 `ram_watch_log.txt` 有第一筆 |
| F6 | minor | 只有正式路徑會走到的分支（`stage0b`、非 smoke 的乾淨檢查、接縫 R 實跑、`--low-priority`）smoke 沒跑過，只靠讀程式和單元測試 | 審查沒找到錯；照計畫執行，出錯照 PLAN 的重跑規則 |

## 審查確認正確的部分（摘要）

- 輸入寬度 1,542 只在 `input_width` 裡生效，離開後還原成 1,540；接縫 R 的重訓在 1,540 下跑。
- 0b 的規則照 PLAN：平均差 ≤ −0.010（含等號）、同種子 10 對至少 8 對嚴格較低；對照來自已 commit 的 prereg25（種子 0–2）和 prereg29（種子 3–9）紀錄。
- 門檻 G1／G2／G4 走 `p23_lib.gates`（Q31 − Q29），G5 是對 Q25 的區間下界 > 0；判定檔先寫、只記錄的分組後寫。
- HD 的接合：訓練 344,946 列、inner 73,556 列，缺 0、不一致 0；HD 的 μ／σ 只用 A 角色的列算，存進每個模型檔。
- smoke 才有的動作（打亂 HD 欄、雜訊特徵）都在 `if smoke` 之下，不會進正式路徑。
- 訓練不讀 inner；留出玩家的標籤只用來挑每個種子的 checkpoint，和對照組相同。
- eval 只能判一次：要 `stage0b.json` 是 GO、指向同一份訓練紀錄、重算規則也是 GO。
- 單元測試 237 項通過（審查者自己跑一次）。

## 審查做不到的

正式路徑分支的實際執行、正式訓練的 RAM 尖峰和時間、eval 那個 commit 上接縫 1 的逐位元重現（只在 smoke 紀錄看過）、`p31_hd.py` 第 420–647 行（統計和防護；只看了已 commit 的輸出）。
