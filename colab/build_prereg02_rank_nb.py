"""Build colab/prereg02_rank.ipynb from the committed colab/prereg01_rank.ipynb and
write results/prereg02/notebook_diff_rank.txt (per-cell unified diff against the
prereg01 notebook, plus the notebook metadata).

Spec: docs/PREREG_02_OPENING_FUSION.md section 1.1 (the registration wins on
any scientific question) and results/prereg02/design/plan.md sections 3.0, 3.1,
3.4, 3.5 (cell list, dumps, repro checks, guard_write, LR schedule).

How the notebook is assembled:
  * the base is colab/prereg01_rank.ipynb, whose SHA-256 is asserted first;
  * base cells are picked by their `c10` cell metadata ({"new": tag} or
    {"official_cell": i}), never by position;
  * a reused cell is either copied verbatim or changed only through rep(), whose
    target string must occur exactly once;
  * every changed or rewritten base cell must be named in CHANGE_ROWS (the table in
    notebook cell 0 and `changes_from_prereg01` in config.json), and every row must
    point at a cell that really changed;
  * every code cell must compile, no cell may contain the forbidden token (the
    validation-answer file name part) except the prepare cell's unchanged refusal,
    and cell ids are c10-rank02-NN.
The output is byte-identical when the builder is re-run.

Run with the smoke venv python from anywhere:
    python colab/build_prereg02_rank_nb.py D:/AI-Portfolio/competitions/aicup2026-go
"""

import difflib
import hashlib
import json
import os
import sys

import nbformat

REPO = sys.argv[1]
BASE_REL = "colab/prereg01_rank.ipynb"
BASE_NB = os.path.join(REPO, *BASE_REL.split("/"))
BASE_SHA256 = "5f97d51978674bcc789493a47f278f186fb839f3f2d396aa2d218a6820e9744b"
OUT_REL = "colab/prereg02_rank.ipynb"
OUT_NB = os.path.join(REPO, *OUT_REL.split("/"))
OUT_DIFF = os.path.join(REPO, "results", "prereg02", "notebook_diff_rank.txt")

# The validation-answer token is spelled in two parts so that a repo-wide search for
# it finds only code that refuses such files.
FORBIDDEN_TOKEN = "answ" + "ers"

# prereg01 results as committed in the repo (results/prereg01/rank/, byte-identical to
# the members of results/prereg01/rank_results.zip, i.e. to the files on the Drive).
P01_EXPECTED_SHA256 = {
    "rank_val_probs.csv": "92cc750cad9eb114541047cf0c4863634ef4de0be1bac75f889edcee4823800c",
    "ckpt/history.csv": "d76272073a80efde206dc8cdac0d5f297aaeb23240f52ff05be35a673718cf35",
    "env.json": "2461a1daa8fef88a636716af8c39a82c48413e98c9eb2d0cccfe99ebe4d90b85",
}


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rep(src, old, new):
    n = src.count(old)
    assert n == 1, ("rep() target must occur exactly once", n, old[:120])
    return src.replace(old, new)


assert sha(BASE_NB) == BASE_SHA256, ("colab/prereg01_rank.ipynb changed", sha(BASE_NB))
base = json.load(open(BASE_NB, encoding="utf-8"))
BASE_CELLS = base["cells"]


def base_key(meta):
    return ("official", meta["official_cell"]) if "official_cell" in meta else ("new", meta["new"])


BASE_BY_KEY = {}
for _j, _c in enumerate(BASE_CELLS):
    _k = base_key(_c["metadata"]["c10"])
    assert _k not in BASE_BY_KEY, _k
    BASE_BY_KEY[_k] = _j


def base_src(key):
    return "".join(BASE_CELLS[BASE_BY_KEY[key]]["source"])


def key_label(key):
    return "底稿 %s（prereg01 cell %d）" % (
        "官方 %d" % key[1] if key[0] == "official" else key[1], BASE_BY_KEY[key])


# ---------------------------------------------------------------------------
# What changed relative to colab/prereg01_rank.ipynb. One list feeds the table in
# notebook cell 0 and `changes_from_prereg01` in config.json. Each row:
# (cell key or None for notebook-wide rows, label, text).
# ---------------------------------------------------------------------------

CHANGE_ROWS = [
    (None, "新增", "第一格 markdown：用途、步驟、會停下的情況、輸出清單、接續方式、這張差異表。"),
    (("new", "setup"), "設定",
     "結果目錄改成 `prereg02`（SMOKE 是 `prereg02_smoke`），分片備份 `FEAT_BACKUP` 跟著改到 `prereg02/features_rank`。"
     "新增只讀的 `P01_ROOT`／`P01_RANK`／`P01_CKPT`／`P01_FEAT`（`prereg01`，SMOKE 是 `prereg01_smoke`）、`guard_write()`"
     "（目的地落在 `P01_ROOT` 底下就停）、`p01_fingerprint()`（檔名、大小、mtime）。`write_json` 先過 `guard_write`。"
     "`SMOKE_CFG` 不動（分片備份的指紋含它，改了就還原不到 prereg01 的分片），本版的 SMOKE 設定另放 `SMOKE_CFG_P02`。"
     "`config.json` 的 `prereg` 改指 PREREG_02，另記底稿 notebook 的 SHA-256 和這張表。"
     "新增收尾開關 `FINISH_WITHOUT_TRAINING = False`（登記 1.2「第 9 個 epoch 沒跑完」的退路；只在 Claude 請你改時改成 True）。"),
    (None, "新增 preflight",
     "開跑前確認 prereg01 的 `epoch_1..5.pth`、`best_model.pth`、`state.pt`、`history.csv`、`rank_val_probs.csv`、`env.json` 都在，"
     "記 SHA-256；`best_model.pth` 必須等於 `epoch_5.pth`；正式執行時機率檔、`history.csv`、`env.json` 必須和本機 commit 的 prereg01 "
     "結果逐位元組相同，inner accuracy 目標是 25.1482%、每個 epoch 2,814 步。prereg01 沒有分片備份只警告（會重抽）。Drive 剩餘空間 "
     "≥ 3 GB。記下 prereg01 目錄的指紋 `p01_readonly_before.json`；重跑時和第一次記的不同就停。"),
    (("official", 2), "底稿官方 2",
     "`NUM_EPOCHS` 維持 prereg01 的 5（SMOKE 是 prereg01 冒煙的 1），用來比對 `state.pt`。新增 `EXTRA_EPOCHS`＝4"
     "（SMOKE 是 1）、`PEAK_LR`＝`LEARNING_RATE`＝1e-3、`FINAL_EPOCH`＝`NUM_EPOCHS`＋`EXTRA_EPOCHS`。`CKPT_DIR` 由設定格決定"
     "（`prereg02/rank/ckpt`），`makedirs` 前先過 `guard_write`。最後的 `print` 多印 GPU 型號和接著訓練的 epoch 數。"
     "正式執行沒有 GPU 就在這裡停下（收尾開關打開時例外，缺的逐盤輸出另由 dump 工具擋下）；有 GPU 但型號"
     "（環境紀錄格 nvidia-smi 查到的）不是 G4（RTX PRO 6000）或 A100 也停下（PREREG_02 第 7 節第 8 條）。"),
    (("new", "feature_backup_tools"), "底稿分片工具",
     "`restore_features(fp, src)` 改成可以指定來源，對來源只讀；還原到本機前 `guard_write(FEAT_DIR)`。"
     "`backup_features` 只寫到 `prereg02/features_rank`，`rmtree` 和每個複製目的地先過 `guard_write`。"),
    (("official", 9), "底稿官方 9",
     "依序試 `prereg02/features_rank`、`prereg01/features_rank` 還原分片；都不能用才重抽（抽完備份到 prereg02）。"
     "`rmtree(FEAT_DIR)` 前先過 `guard_write`。`run_log.json` 另記分片來源。"),
    (None, "新增 dump 工具",
     "`d4(x, s)`（s＝0 原方向；s≥4 先在 dim 3 鏡射，再在 dims (2, 3) 旋轉 s%4 次，格內自我檢查 8 種兩兩不同且封閉）；"
     "`load_net()`（讀進記憶體、算 SHA-256、從同一份位元組載入）；inner 用分片的 `__key__`＝`等級_game_id` 對回 `df_val` 的列，"
     "assert 每盤剛好一次；本機驗證題照題目檔列順序和 slot 1..num_games 解析，失敗的保留位置（logits 填 NaN），"
     "每題一個 batch（和 prereg01 推論相同）；每個檔先寫 `.tmp` 再改名，附 `_DONE_<檔名>.json`（另記算它的那次執行的 GPU 型號、"
     "torch、cuDNN），重跑時對得上就跳過。正式執行時要算 dump 卻沒有 GPU 就停。另有 `record_p01_after()`（結尾指紋）、"
     "`pack_results()`（打包，跳過權重、暫存檔和 ≥ 100 MB 的檔）、`flush_drive()`（Colab 上 `drive.flush_and_unmount()`）。"),
    (None, "新增 P1 輸出與重現核對",
     "prereg01 epoch 1–4 原方向、epoch 5 八種對稱的逐盤 logits（inner、本機驗證題）和兩個 key 檔。重現核對三項：① epoch 5 在 inner "
     "的逐盤 accuracy 和 `history.csv` 相差 ≤ 0.05 個百分點；② 每題 float32 平均機率和 prereg01 `rank_val_probs.csv` 最大差距 "
     "≤ 1e-4（算 `val_p01e5_d4` 那次執行的 GPU 型號和 prereg01 不同時 1e-3；正式執行沒記到 GPU 就不通過）；③ 規則 B 不同的題 ≤ 4"
     "（SMOKE 依題數等比例縮小）。任一項不過就停，不訓練；停下前記結尾指紋、打包 `prereg02/rank_results_repro_failed.zip`、"
     "送出 Drive 的寫入。"),
    (None, "新增接著訓練",
     "從 prereg01 `state.pt`（只讀、核對 SHA-256、設定、逐張量等於 `epoch_5.pth`）載入 model 和 AdamW 狀態，接著訓練 "
     "epoch 6–9：`LambdaLR` 每步 `max(0, 1 − t/T)`、T＝4×2,814，每個 epoch 開始 seed＝42＋epoch，其餘和 prereg01 相同。"
     "每個 epoch 存 `epoch_N.pth`、`state.pt`（含 scheduler、global_step、history、設定）、`history.csv`（inner 驗證只當監看）。"
     "有 prereg02 的 `state.pt` 就接續，設定不同就停。不產生 `best_model.pth`。訓練前用 `_count_w*.json` 和分片清單算出每個 "
     "epoch 的步數（webdataset 先依 worker 分分片），和 prereg01 的 2,814 步不同就停。收尾開關打開時不訓練，已完成的 epoch "
     "從 prereg02 的 `state.pt` 讀回。"),
    (None, "新增 P2 輸出", "用 `epoch_9.pth` 存 8 種對稱的逐盤 logits（inner、本機驗證題）；第 9 個 epoch 沒完成（包括收尾開關打開時）"
     "就跳過並記錄。"),
    (("new", "final_check"), "底稿結尾（改寫）",
     "改成核對所有 dump 的列數、形狀、NaN 位置和完成標記；記 `p01_readonly_after.json` 並和開頭比對（不同就在打包後停下）；"
     "補完 config／timings／run_log；打包 `prereg02/rank_results.zip`（跳過 `.pth`、`.pt`、`.tmp` 和 ≥ 100 MB 的檔；"
     "dump 一律 < 100 MB，否則在寫檔時就停）；最後 `drive.flush_and_unmount()`。"),
    (None, "拿掉",
     "prereg01 的設定紀錄與接續工具、官方 16（訓練迴圈）、18（訓練曲線）、20、21（推論）、所有官方 markdown 格。"
     "推論改由 dump 工具做，訓練改由接著訓練格做。"),
    (None, "metadata",
     "notebook metadata 的 Colab 設定拿掉預選的 A100 和「大量 RAM」（照第一格選 G4；拿不到 G4 時選 A100 標準版）。每格 cell id 是 "
     "`c10-rank02-NN`，cell metadata `c10` 記底稿的格號、底稿的 `c10` 和是否改動。"),
]
P02_CHANGES = ["[%s] %s" % (label, text) for _k, label, text in CHANGE_ROWS]


def _md_cell(s):
    return s.replace("|", "\\|")


CHANGE_TABLE = "\n".join("| %s | %s |" % (label, _md_cell(text)) for _k, label, text in CHANGE_ROWS)

# ---------------------------------------------------------------------------
# Notebook cell 0 (markdown)
# ---------------------------------------------------------------------------

MD_HEADER = r"""# C10 事前登記 02：競賽二（棋力預測）逐盤輸出與接著訓練（Colab 版）

**用途**：`docs/PREREG_02_OPENING_FUSION.md` 第 1.1 節的 Colab 部分，一次執行做完：

1. **只讀 prereg01**：讀 prereg01 的權重、`state.pt`、機率檔、分片備份。任何寫進 prereg01 目錄的動作都會被擋下，開頭和結尾比對 prereg01 目錄的檔案指紋。
2. **存逐盤輸出**：prereg01 epoch 1–4（原方向）、epoch 5（8 種對稱）在 inner（73,556 盤）和本機驗證題（19,156 盤）的 logits。
3. **重現核對**：三項都過才繼續，任一項不過就停下，不訓練。
4. **接著訓練**：從 prereg01 的 `state.pt` 接 4 個 epoch（編號 6–9），學習率從 1e-3 線性降到 0。指定的 checkpoint 是 `epoch_9.pth`，不另選 best。
5. **存 epoch 9 的逐盤輸出**：8 種對稱。

合併、校準、開局特徵、打分都在本機做，不在這裡。這份 notebook 不讀任何答案檔（打包檔裡本來就沒有）。

## 怎麼跑

0. 用**跑 prereg01 的同一個 Google 帳號**：prereg01 的權重和 `state.pt` 在它的 Drive（`MyDrive/C10/results/prereg01/`）。Drive 要剩 3 GB 以上。
1. 把 `prereg02_rank.ipynb` 放進 `MyDrive/C10/`。打包檔 `C10_bundle_v1.zip` 不用重傳。
2. 選單「執行階段 → 變更執行階段類型」：選 **G4 GPU**。G4 是灰色、點不了時，選 **A100 GPU**，「大量 RAM」關掉。不要選「CPU」：沒有 GPU 時，前面幾格就會停下。Colab 改派別的 GPU（不是 G4 或 A100）時也會停下。
3. 選單「執行階段 → 全部執行」。第一個程式格會要求存取 Google Drive，按允許。
4. 跑完後（G4 約 45 分鐘，推估），最後兩行是「Drive 已寫入完成並卸載」和「完成。請下載…」。這時 zip 已經在 Drive 上：先按「執行階段 → 中斷連線並刪除執行階段」（開著會一直扣運算單元），再下載 `MyDrive/C10/results/prereg02/rank_results.zip`，交給 Claude 的方式照 `docs/COLAB_RUN.md`。

詳細步驟以 `docs/COLAB_RUN.md` 的「事前登記 02」一節為準；和這裡不一致時，以那一節為準。

## 會停下來的情況（錯誤訊息裡都寫了處理方式）

- **開頭的檢查（preflight）**：找不到 prereg01 的檔、prereg01 的檔和本機記錄的不同、`best_model.pth` 不等於 `epoch_5.pth`、Drive 空間不足、prereg01 目錄和第一次執行時不同。這時還沒開始算，但執行階段開著就會一直扣運算單元：截圖後先按「執行階段 → 中斷連線並刪除執行階段」。
- **沒有 GPU，或拿到的 GPU 不是 G4／A100**：在設定 `DEVICE` 的那一格停下，還沒開始算。照訊息先刪除執行階段，換成 G4（灰色時 A100）再全部執行。
- **重現核對沒過**：不訓練。這時會先打包 `MyDrive/C10/results/prereg02/rank_results_repro_failed.zip`（含逐盤輸出，查原因要用），再停下。下載這個 zip，連同錯誤訊息的截圖給 Claude（寫入已經送出，可以先刪除執行階段）。
- **每個 epoch 的步數和 prereg01 不同**（分片是重抽的才可能發生）：在接著訓練前停下，逐盤輸出已存好。截圖給 Claude。
- **prereg02 的 `ckpt/state.pt` 讀不進來或設定不同**：照錯誤訊息把 `prereg02/rank/ckpt` 改名保留，再全部執行。
- **結尾發現 prereg01 目錄被改動**：結果照樣打包，但會停下來；截圖給 Claude。

## 結果寫到哪裡

`MyDrive/C10/results/prereg02/rank/`：

| 檔案 | 內容 |
|---|---|
| `dumps/inner_keys.npz` | inner 每盤一列：`rank`、`game_id`、`player_id`、`color`、`label`（0–9，D…6D）。所有 `inner_*.npz` 的列順序都和它相同 |
| `dumps/inner_<名稱>.npz` | `logits` float32 [盤數, 對稱數, 10]、`weights_sha256`、`syms`。名稱：`p01e1`–`p01e4`（原方向）、`p01e5_d4`、`p02e9_d4`（8 種對稱） |
| `dumps/val_keys.npz` | 本機驗證題每個 slot 一列：`question_id`、`slot`（從 1 起算）、`color`、`parsed`；題目檔的列順序，再依 slot 順序 |
| `dumps/val_<名稱>.npz` | 同上的 logits；解析失敗的 slot 整列是 NaN |
| `dumps/_DONE_<檔名>.json` | 每個 dump 的完成標記：列數、SHA-256、權重 SHA-256、秒數，以及算它的那次執行的 GPU 型號、torch、cuDNN 版本 |
| `repro_check.json` | 三項重現核對的數字和結果（② 的容許差依「算 `val_p01e5_d4` 那次執行的 GPU」決定） |
| `ckpt/history.csv` | epoch 6–9 的訓練紀錄和 inner 逐盤 accuracy（只當監看） |
| `ckpt/epoch_6..9.pth`、`ckpt/state.pt` | 權重和接續用的狀態（不打包；指定的 checkpoint 是 `epoch_9.pth`） |
| `p01_readonly_before.json`、`p01_readonly_after.json` | prereg01 目錄每個檔的大小和 mtime（開頭、結尾） |
| `config.json`、`env.json`、`run_log.json`、`timings.json`、`leak_check.json` | 設定、環境、每次執行的紀錄、各階段秒數、`load_rows` 的外洩檢查 |

`MyDrive/C10/results/prereg02/rank_results.zip`：上面除了權重以外的全部檔案，這個檔帶回本機。

## 中斷後怎麼接續

再按一次「全部執行」就好：

- 已完成的逐盤輸出靠完成標記和 SHA-256 跳過。
- 分片特徵依序從 prereg02、prereg01 的 Drive 備份還原；都不能用才重抽（約多 6 分鐘），抽完備份到 prereg02。
- 接著訓練從 prereg02 的 `ckpt/state.pt` 接續。斷在 epoch 中間時，那個 epoch 從頭重跑，學習率排程從已存的步數接續。
- 資料順序和 prereg01 一樣不固定（webdataset 的洗牌），接續過的執行無法逐位元重現；每個 epoch 開始時 `random`、`numpy`、`torch` 的 seed 設為 42＋epoch。
- 換了 GPU 接續時，已存的逐盤輸出不重算；重現核對 ② 的容許差看的是當初算它的那次執行的 GPU（記在完成標記），不會因為這次換了 GPU 而改變。

## 接著訓練做不完時（收尾開關）

**只有 Claude 請你這樣做時才用**（例如運算單元不夠、一直斷線，第 9 個 epoch 跑不完）：

1. 在第一個程式格（設定格）開頭附近找到 `FINISH_WITHOUT_TRAINING = False`，改成 `FINISH_WITHOUT_TRAINING = True`。
2. 選單「執行階段 → 全部執行」。

這樣不會再接著訓練：已存的逐盤輸出照樣核對，第 9 個 epoch 沒完成就不做 epoch 9 的逐盤輸出，結尾照樣打包 `rank_results.zip`。本機照登記（PREREG_02 1.2）拿掉用到 epoch 9 的組合。prereg01 的逐盤輸出都已存好時，執行階段可以選 CPU；有缺的話會停下，請你換回 GPU。

## 和 prereg01 notebook 的差異

底稿是已 commit 的 `colab/prereg01_rank.ipynb`（產生時核對 SHA-256）。沿用的格子沒列在下表的，和 prereg01 逐字相同；和官方 notebook 的差異照 prereg01 的差異表（`config.json` 的 `deviations_from_official`）。逐格的 unified diff 在 `results/prereg02/notebook_diff_rank.txt`。

| 格 | 改了什麼 |
|---|---|
__TABLE__

## 其他

- **本機冒煙測試**：`colab/run_local_smoke.py` 設定 `C10_SMOKE=1`，只用極少量資料在 CPU 上跑。先在同一個 `--scratch` 跑 prereg01 的冒煙測試（產生 `prereg01_smoke`），再跑這份；結果寫到 `results/prereg02_smoke/`。SMOKE 時 prereg01 只有 1 個 epoch，`p01e1`–`p01e5_d4` 都用它，接著訓練 1 個 epoch，`p02e9_d4` 用那個 epoch。
- **不在這裡打分**：答案檔不上傳 Colab。
""".replace("__TABLE__", CHANGE_TABLE)

# ---------------------------------------------------------------------------
# Changes to reused base cells (rep() targets must occur exactly once)
# ---------------------------------------------------------------------------

SETUP_OLD_DIRS = """# 結果目錄：正式和 SMOKE 分開，絕不混用
RESULT_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg01_smoke' if SMOKE else 'prereg01')
RESULT_DIR = os.path.join(RESULT_ROOT, 'rank')
RESULT_CKPT = os.path.join(RESULT_DIR, 'ckpt')
FEAT_BACKUP = os.path.join(RESULT_ROOT, 'features_rank')   # 分片特徵的 Drive 備份
os.makedirs(RESULT_CKPT, exist_ok=True)
"""
SETUP_NEW_DIRS = """# 結果目錄：正式和 SMOKE 分開，絕不混用。本版（prereg02）只寫 prereg02；prereg01 只讀
RESULT_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg02_smoke' if SMOKE else 'prereg02')
RESULT_DIR = os.path.join(RESULT_ROOT, 'rank')
RESULT_CKPT = os.path.join(RESULT_DIR, 'ckpt')
FEAT_BACKUP = os.path.join(RESULT_ROOT, 'features_rank')   # 本版自己的分片備份（只在重抽時寫）

# 本版：prereg01 的結果目錄（權重、state.pt、機率檔、分片備份）只讀
P01_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg01_smoke' if SMOKE else 'prereg01')
P01_RANK = os.path.join(P01_ROOT, 'rank')
P01_CKPT = os.path.join(P01_RANK, 'ckpt')
P01_FEAT = os.path.join(P01_ROOT, 'features_rank')


def _real(path):
    return os.path.normcase(os.path.realpath(path))


def _is_under(path, root):
    p, r = _real(path), _real(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def guard_write(path):
    # 本版：寫檔、torch.save、np.savez、copyfile 的目的地、rmtree、zip 一律先過這一關；落在 prereg01 底下就停
    if _is_under(path, P01_ROOT):
        raise PermissionError('本版不能寫入 prereg01 的目錄（只讀）：%s' % path)
    return path


def p01_fingerprint():
    # prereg01 目錄每個檔的 {相對路徑: [bytes, mtime_ns]}；開頭和結尾各記一次，結尾必須相同
    out = {}
    for root, dirs, files in os.walk(P01_ROOT):
        dirs.sort()
        for n in sorted(files):
            p = os.path.join(root, n)
            st = os.stat(p)
            out[os.path.relpath(p, P01_ROOT).replace(os.sep, '/')] = [st.st_size, st.st_mtime_ns]
    return out


# 本機工作目錄、prereg02 的結果目錄都不能和 prereg01 互相包含
# （沿用的準備格、抽特徵的 worker 只寫 WORK 底下，所以不另外過 guard_write）
for _p in (WORK, RESULT_ROOT):
    guard_write(_p)
    assert not _is_under(P01_ROOT, _p), (P01_ROOT, _p)
guard_write(RESULT_CKPT)
os.makedirs(RESULT_CKPT, exist_ok=True)
"""

SETUP_OLD_START = "RUN_START = time.time()\n"
SETUP_NEW_START = """RUN_START = time.time()

# 本版（prereg02）收尾開關：平常維持 False。只有 Claude 請你改的時候才改成 True（見第一格「接著訓練做不完時」）。
# True：不再接著訓練；已存的逐盤輸出照樣核對，第 9 個 epoch 沒完成就照登記不做 epoch 9 的逐盤輸出，結尾照樣打包。
FINISH_WITHOUT_TRAINING = False
"""

SETUP_OLD_DEV = "# 和官方 notebook 的全部差異（與第 1 格的表同一份）\n"
SETUP_NEW_DEV = """# 本版（prereg02）的 SMOKE 設定另外放；SMOKE_CFG 不動（分片備份的指紋含它，改了就還原不到 prereg01 的分片）
SMOKE_CFG_P02 = {
    'extra_epochs': 1,     # 接著訓練的 epoch 數（正式是 4）
}

# 本版和 prereg01 notebook 的差異（與第一格的表同一份）
P02_CHANGES = __P02_CHANGES__

# 和官方 notebook 的全部差異（prereg01 的差異表，沿用的格子照它）
""".replace("__P02_CHANGES__", json.dumps(P02_CHANGES, ensure_ascii=False, indent=4))

SETUP_OLD_WJ = """def write_json(path, obj):
    # 先寫暫存檔再改名，中斷時不會留下寫一半的 JSON；float 用 repr，不四捨五入
    tmp = path + '.tmp'
"""
SETUP_NEW_WJ = """def write_json(path, obj):
    # 先寫暫存檔再改名，中斷時不會留下寫一半的 JSON；float 用 repr，不四捨五入
    guard_write(path)   # 本版：寫進 prereg01 目錄的一律擋下
    tmp = path + '.tmp'
"""

SETUP_OLD_PREREG = "    'prereg': 'docs/PREREG_01_BASELINE.md rev 2（競賽二）',\n"
SETUP_NEW_PREREG = """    'prereg': 'docs/PREREG_02_OPENING_FUSION.md 第 1.1 節（競賽二）',
    'base_notebook': {'path': '__BASE_REL__', 'sha256': '__BASE_SHA__'},
    'changes_from_prereg01': P02_CHANGES,
    'smoke_cfg_p02': SMOKE_CFG_P02 if SMOKE else None,
    'p01_root': P01_ROOT,
""".replace("__BASE_REL__", BASE_REL).replace("__BASE_SHA__", BASE_SHA256)

SETUP_OLD_PRINT = "print('結果目錄', RESULT_DIR)"
SETUP_NEW_PRINT = "print('結果目錄', RESULT_DIR)\nprint('prereg01（只讀）', P01_ROOT)"

OFF2_OLD_DEVICE = "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
OFF2_NEW_DEVICE = """# 本版（prereg02）：NUM_EPOCHS 維持 prereg01 的值，用來比對 prereg01 的 state.pt；接著訓練 EXTRA_EPOCHS 個 epoch
EXTRA_EPOCHS = SMOKE_CFG_P02['extra_epochs'] if SMOKE else 4
PEAK_LR = LEARNING_RATE            # 1e-3，每步線性降到 0
FINAL_EPOCH = NUM_EPOCHS + EXTRA_EPOCHS   # 正式是 9：登記指定的 checkpoint 是 epoch_9.pth
if not SMOKE:
    assert NUM_EPOCHS == 5 and EXTRA_EPOCHS == 4 and PEAK_LR == 1e-3 and BATCH_SIZE == 256, 'prereg02 的設定被改動'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 本版：正式執行一定要有 GPU（逐盤輸出和接著訓練在 CPU 上要好幾天）。收尾開關打開時可以沒有 GPU，
# 但逐盤輸出必須都已存好（缺的話 dump 工具會停下）
if not SMOKE and DEVICE.type != 'cuda' and not FINISH_WITHOUT_TRAINING:
    raise SystemExit('沒有 GPU，停在這裡（還沒開始算）。處理方式：選單「執行階段 → 變更執行階段類型」選 G4 GPU'
                     '（灰色時選 A100 GPU，「大量 RAM」關掉）→ 儲存，再按一次「全部執行」。')
# 本版：GPU 型號只接受 G4（RTX PRO 6000）或 A100（PREREG_02 第 7 節第 8 條）。Colab 在選的 GPU 沒空時可能改派別的 GPU。
# 型號用環境紀錄格 nvidia-smi 查到的（不在這裡初始化 CUDA）
GPU_MODELS_OK = ('RTX PRO 6000', 'A100')
GPU_MODEL_NOW = gpu_name_of(ENV_INFO)
if not SMOKE and DEVICE.type == 'cuda' and not any(k in (GPU_MODEL_NOW or '') for k in GPU_MODELS_OK):
    raise SystemExit('這次拿到的 GPU 是 %s，不是 G4（RTX PRO 6000）或 A100，停在這裡（還沒開始算）。處理方式：'
                     '選單「執行階段 → 中斷連線並刪除執行階段」，再「執行階段 → 變更執行階段類型」選 G4 GPU'
                     '（灰色時選 A100 GPU，「大量 RAM」關掉）→ 儲存，再按一次「全部執行」。' % GPU_MODEL_NOW)
"""
OFF2_OLD_MKDIR = "os.makedirs(CKPT_DIR, exist_ok=True)\n"
OFF2_NEW_MKDIR = "guard_write(CKPT_DIR)\nos.makedirs(CKPT_DIR, exist_ok=True)\n"
OFF2_OLD_PRINT = "print('device', DEVICE, '| NPROC', NPROC, '| batch', BATCH_SIZE, '| epochs', NUM_EPOCHS)"
OFF2_NEW_PRINT = ("print('device', DEVICE, '| GPU', GPU_MODEL_NOW, '| NPROC', NPROC, '| batch', BATCH_SIZE,\n"
                  "      '| epochs', NUM_EPOCHS, '| extra epochs', EXTRA_EPOCHS, '| final epoch', FINAL_EPOCH)")

FT_OLD_BACKUP = """    # 全部複製並核對大小之後，才寫完成標記
    shutil.rmtree(FEAT_BACKUP, ignore_errors=True)
"""
FT_NEW_BACKUP = """    # 全部複製並核對大小之後，才寫完成標記。本版：只寫到 prereg02 的 FEAT_BACKUP
    guard_write(FEAT_BACKUP)
    shutil.rmtree(FEAT_BACKUP, ignore_errors=True)
"""
FT_OLD_DST = """        dst = os.path.join(FEAT_BACKUP, rel)
"""
FT_NEW_DST = """        dst = guard_write(os.path.join(FEAT_BACKUP, rel))
"""
FT_OLD_SIG = """def restore_features(fingerprint):
    # 回傳 {'counts':…, 'mode':…}；沒有可用的備份（沒有、設定不同、缺檔、壞檔）時回傳 None，呼叫端會重抽
"""
FT_NEW_SIG = """def restore_features(fingerprint, src):
    # 回傳 {'counts':…, 'mode':…, 'source':…}；沒有可用的備份（沒有、設定不同、缺檔、壞檔）時回傳 None，呼叫端會試下一個來源或重抽
    # 本版：備份來源 src 可以指定（prereg02 或 prereg01 的 features_rank），對來源只讀；只寫本機的 FEAT_DIR
    guard_write(FEAT_DIR)
"""
FT_OLD_MARKER = "        m = read_json(FEAT_MARKER)\n"
FT_NEW_MARKER = "        m = read_json(os.path.join(src, '_COMPLETE.json'))\n"
FT_OLD_NOBACKUP = "        print('Drive 上沒有完整的分片備份')\n"
FT_NEW_NOBACKUP = "        print('%s：沒有完整的分片備份' % src)\n"
FT_OLD_CHECKSRC = "        if _feature_files(FEAT_BACKUP) != want:\n"
FT_NEW_CHECKSRC = "        if _feature_files(src) != want:\n"
FT_OLD_COPY = "                shutil.copyfile(os.path.join(FEAT_BACKUP, rel), dst)\n"
FT_NEW_COPY = "                shutil.copyfile(os.path.join(src, rel), dst)\n"
FT_OLD_RET = """    print('分片：%s（%d 個檔）' % (mode, len(want)))
    return {'counts': counts, 'mode': mode}"""
FT_NEW_RET = """    print('分片：%s（%d 個檔，來源 %s）' % (mode, len(want), src))
    return {'counts': counts, 'mode': mode, 'source': src}"""

OFF9_OLD_RESTORE = "restored = restore_features(FEAT_FINGERPRINT)\n"
OFF9_NEW_RESTORE = """# 本版：先試 prereg02 自己的備份，再試 prereg01 的備份（只讀）；兩個都不能用才重抽
restored = None
for _src in (FEAT_BACKUP, P01_FEAT):
    restored = restore_features(FEAT_FINGERPRINT, _src)
    if restored is not None:
        break
"""
OFF9_OLD_RMTREE = "    shutil.rmtree(FEAT_DIR, ignore_errors=True)\n    val_counts = extract_split(df_val, 'val')\n"
OFF9_NEW_RMTREE = ("    print('prereg02 和 prereg01 都沒有可用的分片備份：重抽（約多 6 分鐘），抽完備份到 %s' % FEAT_BACKUP)\n"
                   "    guard_write(FEAT_DIR)\n"
                   "    shutil.rmtree(FEAT_DIR, ignore_errors=True)\n    val_counts = extract_split(df_val, 'val')\n")
OFF9_OLD_MODE = "    RUN['features'] = restored['mode']\n"
OFF9_NEW_MODE = "    RUN['features'] = restored['mode']\n    RUN['features_source'] = restored['source']\n"

# ---------------------------------------------------------------------------
# New code cells
# ---------------------------------------------------------------------------

PREFLIGHT = r"""# 本版新增（preflight）：開跑前確認 prereg01 的權重、state.pt、機率檔都在，記下 SHA-256 和 prereg01 目錄的指紋。
# 任何一項不符就用中文說明處理方式後停下，還沒開始算（但執行階段開著就會扣運算單元）。
import hashlib

t0 = time.time()
P01_EPOCHS = SMOKE_CFG['num_epochs'] if SMOKE else 5
# 正式執行：這幾個檔必須和本機 commit 的 prereg01 結果（results/prereg01/rank/）逐位元組相同
P01_EXPECTED_SHA256 = {} if SMOKE else __P01_EXPECTED__
P01_INNER_ACC_REGISTERED = 25.1482   # PREREG_02 1.1 第 2 點；SMOKE 不用
P01_STEPS_REGISTERED = 2814          # prereg01 每個 epoch 的步數；SMOKE 不用
MIN_DRIVE_FREE_BYTES = 3 * 10 ** 9
P01_HELP = ('處理方式：確認這個 Colab 用的是跑 prereg01 的同一個 Google 帳號，而且 Drive 上有 '
            'MyDrive/C10/results/prereg01/rank/ckpt/（epoch_1..5.pth、best_model.pth、state.pt、history.csv）'
            '和 rank/rank_val_probs.csv、rank/env.json。不要搬動、改名或修改 prereg01 的任何檔案。'
            '還是不對就把這段訊息截圖給 Claude。')


def sha256_path(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            d.update(chunk)
    return d.hexdigest()


def gpu_name_of(env_run):
    # env.json 一筆執行紀錄裡的 GPU 型號（nvidia-smi 的第一欄）；沒有 GPU 時是 None
    smi = (env_run.get('gpu') or {}).get('nvidia_smi')
    return smi[0].split(',')[0].strip() if smi else None


P01_NEED = (['ckpt/epoch_%d.pth' % e for e in range(1, P01_EPOCHS + 1)]
            + ['ckpt/best_model.pth', 'ckpt/state.pt', 'ckpt/history.csv', 'rank_val_probs.csv', 'env.json'])
_missing = [n for n in P01_NEED if not os.path.isfile(os.path.join(P01_RANK, n))]
if _missing:
    raise SystemExit('找不到 prereg01 的檔案：%s（在 %s 底下）。%s' % (_missing, P01_RANK, P01_HELP))
P01_SHA = {n: sha256_path(os.path.join(P01_RANK, n)) for n in P01_NEED}
if P01_SHA['ckpt/best_model.pth'] != P01_SHA['ckpt/epoch_%d.pth' % P01_EPOCHS]:
    raise SystemExit('prereg01 的 best_model.pth 和 epoch_%d.pth 不是同一份權重（登記寫的是第 %d 個 epoch）。%s'
                     % (P01_EPOCHS, P01_EPOCHS, P01_HELP))
_bad = sorted(n for n, v in P01_EXPECTED_SHA256.items() if P01_SHA[n] != v)
if _bad:
    raise SystemExit('Drive 上 prereg01 的 %s 和本機 commit 的 prereg01 結果不同（SHA-256 不符）。%s' % (_bad, P01_HELP))

with open(os.path.join(P01_CKPT, 'history.csv'), newline='', encoding='utf-8') as f:
    P01_HISTORY = list(csv.DictReader(f))
if [int(h['epoch']) for h in P01_HISTORY] != list(range(1, P01_EPOCHS + 1)):
    raise SystemExit('prereg01 的 history.csv 不是 epoch 1..%d。%s' % (P01_EPOCHS, P01_HELP))
P01_INNER_ACC = float(P01_HISTORY[-1]['val_acc'])   # 重現核對 ① 的目標（最後一個 epoch 的 inner 逐盤 accuracy）
P01_STEPS = int(P01_HISTORY[-1]['steps'])           # 學習率排程的每個 epoch 步數
if not SMOKE and (abs(P01_INNER_ACC - P01_INNER_ACC_REGISTERED) > 1e-3 or P01_STEPS != P01_STEPS_REGISTERED):
    raise SystemExit('prereg01 的 history.csv 和登記的數字不同（inner accuracy %.4f%%、每個 epoch %d 步；登記是 %.4f%%、%d 步）。%s'
                     % (P01_INNER_ACC, P01_STEPS, P01_INNER_ACC_REGISTERED, P01_STEPS_REGISTERED, P01_HELP))
_p01_env_run = read_json(os.path.join(P01_RANK, 'env.json'))['runs'][-1]
P01_GPU_NAME = gpu_name_of(_p01_env_run)    # 重現核對 ② 的 GPU 規則用（和寫出逐盤輸出那次執行的 GPU 比）
P01_TORCH, P01_CUDNN = _p01_env_run.get('torch'), _p01_env_run.get('cudnn')

P01_FEAT_COMPLETE = os.path.exists(os.path.join(P01_FEAT, '_COMPLETE.json'))
if not P01_FEAT_COMPLETE:
    print('注意：prereg01 沒有完整的分片備份（%s）。如果 prereg02 也沒有，後面會重抽特徵，約多 6 分鐘。' % P01_FEAT)

try:
    DRIVE_FREE_BYTES = shutil.disk_usage(DRIVE_DIR).free
except OSError as e:
    DRIVE_FREE_BYTES = None
    print('注意：查不到 Drive 剩餘空間（%s: %s），請自己確認還有 3 GB 以上' % (type(e).__name__, e))
if DRIVE_FREE_BYTES is not None and DRIVE_FREE_BYTES < MIN_DRIVE_FREE_BYTES:
    raise SystemExit('Drive 只剩 %.2f GB，這份 notebook 要 3 GB 以上（權重和狀態約 0.75 GB，重抽特徵的話再加約 1.3 GB）。'
                     '處理方式：清出空間（Drive 的垃圾桶也要清空）後再按一次「全部執行」。' % (DRIVE_FREE_BYTES / 1e9))

# prereg01 目錄的指紋（開頭）。重跑時必須和第一次執行記下的相同：不同代表 prereg01 在兩次執行之間被改過
P01_BEFORE = p01_fingerprint()
P01_BEFORE_PATH = os.path.join(RESULT_DIR, 'p01_readonly_before.json')
_before_log = read_runs(P01_BEFORE_PATH)
if _before_log['runs'] and _before_log['runs'][0]['files'] != P01_BEFORE:
    _first = _before_log['runs'][0]['files']
    _changed = sorted(k for k in set(_first) | set(P01_BEFORE) if _first.get(k) != P01_BEFORE.get(k))
    raise SystemExit('prereg01 的目錄和第一次執行時不同（%d 個檔：%s）。prereg01 必須只讀，先查清楚是誰改的。'
                     '處理方式：不要再執行，把這段訊息截圖給 Claude。' % (len(_changed), _changed[:10]))
_before_log['runs'].append({'run': RUN['run'], 'time': now_iso(), 'root': P01_ROOT, 'n_files': len(P01_BEFORE),
                            'total_bytes': sum(v[0] for v in P01_BEFORE.values()), 'files': P01_BEFORE})
write_json(P01_BEFORE_PATH, _before_log)

save_config(preflight={'p01_root': P01_ROOT, 'p01_sha256': P01_SHA, 'p01_expected_sha256': P01_EXPECTED_SHA256,
                       'p01_inner_acc': P01_INNER_ACC, 'p01_steps_per_epoch': P01_STEPS, 'p01_gpu_name': P01_GPU_NAME,
                       'p01_torch': P01_TORCH, 'p01_cudnn': P01_CUDNN,
                       'p01_features_backup_complete': P01_FEAT_COMPLETE, 'drive_free_bytes': DRIVE_FREE_BYTES,
                       'p01_files_fingerprinted': len(P01_BEFORE)})
mark_time('preflight', time.time() - t0)
print('preflight 通過：prereg01 的 %d 個檔都在，best_model.pth＝epoch_%d.pth，inner accuracy 目標 %.4f%%，每個 epoch %d 步，'
      'prereg01 的 GPU：%s' % (len(P01_NEED), P01_EPOCHS, P01_INNER_ACC, P01_STEPS, P01_GPU_NAME))
""".replace("__P01_EXPECTED__", json.dumps(P01_EXPECTED_SHA256, indent=4).replace("\n", "\n    "))

DUMP_TOOLS = r"""# 本版新增：逐盤輸出（dump）工具
# - inner：對內部驗證的分片跑一遍（和 validate() 一樣 batch 256、不打散），用分片的 __key__＝「等級_game_id」對回 df_val 的列
# - 本機驗證題：照題目檔的列順序和 slot 1..num_games 解析，失敗的保留位置（logits 填 NaN）；
#   每題一個 batch，和 prereg01 的推論相同，重現核對才對得上
# - 每個檔先寫 .tmp 再改名，附 _DONE_<檔名>.json；重跑時完成標記、SHA-256、權重 SHA-256 都對得上就跳過
import io
import math
import random
import zipfile

DUMP_DIR = os.path.join(RESULT_DIR, 'dumps')
guard_write(DUMP_DIR)
os.makedirs(DUMP_DIR, exist_ok=True)
DUMP_MAX_BYTES = 100 * 1024 * 1024   # 結果 zip 會跳過 ≥ 100 MB 的檔，所以每個 dump 都必須更小
GPU_BATCH = 64                         # 和 prereg01 推論相同
NUM_SYMS = 8
EXPECTED_INNER_GAMES = SMOKE_CFG['inner_games'] if SMOKE else 73556
if not VAL_SIZE == len(df_val) == EXPECTED_INNER_GAMES:
    raise SystemExit('內部驗證的盤數不對：VAL_SIZE %s、df_val %d、應該是 %d。請截圖給 Claude。'
                     % (VAL_SIZE, len(df_val), EXPECTED_INNER_GAMES))


def d4(x, s):
    # 8 種對稱：s=0 原方向；s>=4 先在 dim 3 鏡射，再在 dims (2, 3) 旋轉 s%4 次（x 是 [batch, 500, 19, 19]）
    if s == 0:
        return x
    y = torch.flip(x, dims=(3,)) if s >= 4 else x
    return torch.rot90(y, k=s % 4, dims=(2, 3)).contiguous()


# 自我檢查：8 種兩兩不同，而且任兩種接連做仍在這 8 種之內（封閉，是一個群）
_t = torch.arange(2 * 19 * 19, dtype=torch.float32).reshape(1, 2, 19, 19)
_imgs = [d4(_t, s) for s in range(NUM_SYMS)]
assert all(not torch.equal(_imgs[a], _imgs[b]) for a in range(NUM_SYMS) for b in range(a))
assert all(any(torch.equal(d4(_imgs[a], b), c) for c in _imgs) for a in range(NUM_SYMS) for b in range(NUM_SYMS))


def load_net(path, expect_sha=None):
    # 整個檔讀進記憶體、算 SHA-256，再從同一份位元組載入：記下的 SHA 就是實際用的權重
    with open(path, 'rb') as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    if expect_sha is not None and got != expect_sha:
        raise SystemExit('%s 的 SHA-256 和開頭記下的不同（執行中被改動？）。請把這段訊息截圖給 Claude。' % path)
    net = GoRankResNet(num_classes=NUM_CLASSES)
    net.load_state_dict(torch.load(io.BytesIO(raw), map_location='cpu'))
    net.eval().to(DEVICE)
    return net, got


def _dump_path(stem):
    return os.path.join(DUMP_DIR, stem + '.npz')


def _marker_path(stem):
    return os.path.join(DUMP_DIR, '_DONE_%s.json' % stem)


def dump_is_done(stem, rows, syms, weights_sha):
    m = read_json(_marker_path(stem))
    if not m or not os.path.exists(_dump_path(stem)):
        return False
    if (m.get('rows'), m.get('syms'), m.get('weights_sha256')) != (rows, syms, weights_sha):
        return False
    return sha256_path(_dump_path(stem)) == m.get('sha256')


def write_dump(stem, rows, syms, weights, weights_sha, seconds, arrays):
    # arrays：要存進 npz 的 {名稱: 陣列}；rows、syms、weights、weights_sha 記進完成標記
    # 舊的完成標記先拿掉 → 寫 .tmp → 檢查大小 → 改名 → 最後才寫完成標記
    path, marker = _dump_path(stem), _marker_path(stem)
    tmp = path + '.tmp'
    for p in (path, tmp, marker):
        guard_write(p)
    if os.path.exists(marker):
        os.remove(marker)
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    size = os.path.getsize(tmp)
    if size >= DUMP_MAX_BYTES:
        raise SystemExit('%s 有 %.1f MB，超過 100 MB（結果 zip 會漏掉它）。請截圖給 Claude。' % (stem, size / 1e6))
    os.replace(tmp, path)
    # gpu_name／torch／cudnn：算出這個 dump 的那次執行（重跑時可能換了 GPU；重現核對 ② 的 GPU 規則看這裡）
    write_json(marker, {'file': 'dumps/%s.npz' % stem, 'rows': rows, 'syms': syms, 'bytes': size,
                        'sha256': sha256_path(path), 'weights': weights, 'weights_sha256': weights_sha,
                        'seconds': seconds, 'time': now_iso(), 'run': RUN['run'],
                        'gpu_name': gpu_name_of(ENV_INFO), 'torch': torch.__version__, 'cudnn': ENV_INFO.get('cudnn'),
                        'device': str(DEVICE)})


def require_gpu_for(stem):
    # 正式執行不在 CPU 上算逐盤輸出（要好幾天）；收尾開關打開又沒有 GPU 時，缺的 dump 在這裡停下
    if not SMOKE and DEVICE.type != 'cuda':
        raise SystemExit('%s 還沒算好，但這個執行階段沒有 GPU，停在這裡。處理方式：選單「執行階段 → 變更執行階段類型」'
                         '選 G4 GPU（灰色時選 A100 GPU，「大量 RAM」關掉）→ 儲存，再按一次「全部執行」。' % stem)


def record_p01_after():
    # prereg01 目錄的指紋（結尾，或重現核對沒過時）：和這次開頭記的比，寫進 p01_readonly_after.json；回傳 (指紋, 不同的檔)
    after = p01_fingerprint()
    path = os.path.join(RESULT_DIR, 'p01_readonly_after.json')
    log = read_runs(path)
    changed = sorted(k for k in set(P01_BEFORE) | set(after) if P01_BEFORE.get(k) != after.get(k))
    log['runs'].append({'run': RUN['run'], 'time': now_iso(), 'root': P01_ROOT, 'n_files': len(after),
                        'total_bytes': sum(v[0] for v in after.values()), 'files': after,
                        'same_as_before': not changed, 'changed_files': changed})
    write_json(path, log)
    return after, changed


def pack_results(zip_name):
    # 結果目錄裡除了權重和暫存檔以外的檔案，打成一個 zip 放在結果目錄旁邊（prereg02/<zip_name>）；回傳 (路徑, 跳過的大檔)
    path = guard_write(os.path.join(os.path.dirname(RESULT_DIR), zip_name))
    tmp = guard_write(path + '.tmp')
    skipped = []
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(RESULT_DIR):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(root, f)
                if f.endswith(('.pth', '.pt', '.tmp')):
                    continue   # 權重與暫存檔不帶
                if os.path.getsize(p) > 100 * 1024 * 1024:
                    skipped.append(os.path.relpath(p, RESULT_DIR))
                    continue
                zf.write(p, os.path.relpath(p, os.path.dirname(RESULT_DIR)))
    os.replace(tmp, path)
    if skipped:
        print('注意：這些檔超過 100 MB，沒有打包：%s' % skipped)
    return path, skipped


def flush_drive():
    # Colab：下載 zip、刪除執行階段之前，先把 Drive 的寫入全部送出再卸載（權重、state.pt、zip 才確定都在 Drive 上）
    if IN_COLAB:
        drive.flush_and_unmount()
        print('Drive 已寫入完成並卸載；要再跑請重新「全部執行」')


def write_keys(stem, arrays):
    # key 檔：已有完整的就讀回來比對，必須完全相同（不同代表列順序變了，已存的 dump 會對錯盤）
    rows = len(next(iter(arrays.values())))
    if dump_is_done(stem, rows, None, None):
        with np.load(_dump_path(stem), allow_pickle=False) as z:
            same = sorted(z.files) == sorted(arrays) and all(
                z[k].dtype == v.dtype and np.array_equal(z[k], v) for k, v in arrays.items())
        if not same:
            raise SystemExit('%s 和這次算出的 key 不同，已存的逐盤輸出可能對錯盤。處理方式：不要刪檔，把這段訊息截圖給 Claude。'
                             % stem)
        print('  %s：已存在且相同，跳過' % stem)
        return 'skipped'
    t0 = time.time()
    write_dump(stem, rows, None, None, None, time.time() - t0, arrays)
    print('  %s：寫出 %d 列' % (stem, rows))
    return 'written'


# inner 的 key：df_val 的列順序（load_rows 的順序再用 SEED 打散，每次執行都相同）
INNER_KEYS = {
    'rank': np.array(df_val['rank'].tolist(), dtype=str),
    'game_id': np.array(df_val['game_id'].tolist(), dtype=str),
    'player_id': np.array(df_val['player_id'].tolist(), dtype=str),
    'color': np.array(df_val['color'].tolist(), dtype=str),
    'label': np.array([RANK_TO_ID[r] for r in df_val['rank']], dtype=np.int8),
}
INNER_INDEX = {(r, g): i for i, (r, g) in enumerate(zip(df_val['rank'], df_val['game_id']))}
assert len(INNER_INDEX) == len(df_val), '(等級, game_id) 在內部驗證裡不唯一'
assert set(INNER_KEYS['color'].tolist()) <= {'B', 'W'}


def decode_with_key(sample):
    return (torch.from_numpy(sample['features.npy']).float(),
            torch.tensor(int(sample['rank_idx.txt']), dtype=torch.long),
            sample['game_name.txt'], sample['__key__'])


# inner 專用的 loader：和 val_loader 相同的分片、worker 數、batch，多回傳分片的 __key__
KEY_LOADER = DataLoader(
    wds.WebDataset(val_shards, shardshuffle=0).decode().map(decode_with_key).batched(BATCH_SIZE, partial=True),
    batch_size=None, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def dump_inner(net, name, syms, weights, weights_sha):
    stem, n = 'inner_' + name, len(df_val)
    if dump_is_done(stem, n, syms, weights_sha):
        print('  %s：已完成，跳過' % stem)
        return 'skipped'
    require_gpu_for(stem)
    t0 = time.time()
    logits = np.full((n, syms, NUM_CLASSES), np.nan, dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    count = 0
    with torch.no_grad():
        for x, y, names, keys in KEY_LOADER:
            idx = []
            for lab, g, k in zip(y.tolist(), names, keys):
                if k != '%s_%s' % (LEVELS[lab], g):
                    raise SystemExit('分片的 __key__ %r 和（等級, game_id）＝（%s, %s）對不上。請截圖給 Claude。'
                                     % (k, LEVELS[lab], g))
                i = INNER_INDEX.get((LEVELS[lab], g))
                if i is None:
                    raise SystemExit('分片裡的 %s 不在這次的內部驗證（df_val）裡。請截圖給 Claude。' % k)
                idx.append(i)
            idx = np.asarray(idx, dtype=np.int64)
            if len(set(idx.tolist())) != len(idx) or seen[idx].any():
                raise SystemExit('內部驗證有盤出現兩次（%s）。請截圖給 Claude。' % stem)
            seen[idx] = True
            count += len(idx)
            xb = x.to(DEVICE, non_blocking=True)
            for s in range(syms):
                logits[idx, s] = net(d4(xb, s)).float().cpu().numpy()
    if count != n or not seen.all():
        raise SystemExit('%s：讀到 %d 盤，內部驗證是 %d 盤，有盤沒讀到。請截圖給 Claude。' % (stem, count, n))
    write_dump(stem, n, syms, weights, weights_sha, time.time() - t0,
               {'logits': logits, 'weights_sha256': np.array(weights_sha), 'syms': np.array(syms)})
    print('  %s：%d 盤 × %d 種方向，%.0f 秒' % (stem, n, syms, time.time() - t0), flush=True)
    return 'written'


_infer_parser = SGFParseRankPrediction()


def sgf_to_features(sgf_content):
    # 給 multiprocessing.Pool 用的頂層函式（和 prereg01 推論相同）
    feat, err = _infer_parser.features_from_sgf_content(sgf_content)
    return None if err is not None else feat


VAL = None   # parse_val() 的結果，整份 notebook 共用（只解析一次）


def parse_val():
    # 本機驗證題：題目檔的列順序、slot 1..num_games；SMOKE 時和 prereg01 冒煙一樣只取前幾題
    global VAL
    if VAL is not None:
        return VAL
    t0 = time.time()
    test = pd.read_csv(RANK_QUESTIONS, dtype={'question_id': str})
    if SMOKE:
        test = test.head(SMOKE_CFG['questions'])
    sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
    qids, slots, colors, sgfs, spans = [], [], [], [], []
    for _, r in test.iterrows():
        # 和 prereg01 一樣取非空的 sgf_ 欄；本版另外確認它們剛好是 sgf_1..sgf_num_games，slot 才對得回執色
        n = int(r['num_games'])
        cols = [c for c in sgf_cols if pd.notna(r[c]) and str(r[c]).strip()]
        assert cols == ['sgf_%d' % j for j in range(1, n + 1)], (r['question_id'], cols, n)
        spans.append((r['question_id'], len(sgfs), n))
        for j in range(1, n + 1):
            c = r['color_%d' % j]
            assert c in ('B', 'W'), (r['question_id'], j, c)
            qids.append(r['question_id'])
            slots.append(j)
            colors.append(c)
            sgfs.append(r['sgf_%d' % j])
    with Pool(NPROC) as pool:
        feats = pool.map(sgf_to_features, sgfs, chunksize=8)
    parsed = np.array([f is not None for f in feats], dtype=bool)
    VAL = {'spans': spans, 'feats': feats,
           'keys': {'question_id': np.array(qids, dtype=str), 'slot': np.array(slots, dtype=np.int32),
                    'color': np.array(colors, dtype=str), 'parsed': parsed}}
    mark_time('parse_val', time.time() - t0)
    print('本機驗證題：%d 題、%d 盤，解析失敗 %d 盤，%.0f 秒'
          % (len(spans), len(feats), int((~parsed).sum()), time.time() - t0), flush=True)
    return VAL


def dump_val(net, name, syms, weights, weights_sha):
    V = parse_val()
    stem, n = 'val_' + name, len(V['feats'])
    if dump_is_done(stem, n, syms, weights_sha):
        print('  %s：已完成，跳過' % stem)
        return 'skipped'
    require_gpu_for(stem)
    t0 = time.time()
    logits = np.full((n, syms, NUM_CLASSES), np.nan, dtype=np.float32)
    with torch.no_grad():
        for _qid, start, k in V['spans']:
            ok = [start + j for j in range(k) if V['feats'][start + j] is not None]
            if not ok:
                continue
            # 和 prereg01 相同：一題的棋疊成一個 batch（每批最多 GPU_BATCH 盤）
            x = torch.from_numpy(np.stack([V['feats'][g] for g in ok])).float()
            for i in range(0, len(x), GPU_BATCH):
                xb = x[i:i + GPU_BATCH].to(DEVICE)
                rows = ok[i:i + GPU_BATCH]
                for s in range(syms):
                    logits[rows, s] = net(d4(xb, s)).float().cpu().numpy()
    write_dump(stem, n, syms, weights, weights_sha, time.time() - t0,
               {'logits': logits, 'weights_sha256': np.array(weights_sha), 'syms': np.array(syms)})
    print('  %s：%d 盤 × %d 種方向，%.0f 秒' % (stem, n, syms, time.time() - t0), flush=True)
    return 'written'


_E1 = math.exp(-1)


def _argmax_first(values):
    # 最大值的位置；同分取前面的（和 numpy.argmax、tools/decide.py 相同）
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def rule_b(p):
    # 規則 B（和 tools/decide.py 的 rule_b 相同）：使 p_k + e^-1·(p_{k-1} + p_{k+1}) 最大的等級，兩端外側算 0
    n = len(p)
    return LEVELS[_argmax_first([p[k] + _E1 * ((p[k - 1] if k > 0 else 0.0) + (p[k + 1] if k < n - 1 else 0.0))
                                 for k in range(n)])]


print('dump 工具就緒：inner %d 盤（worker %d）' % (len(df_val), workers))
"""

P1 = r"""# 本版新增：P1 逐盤輸出（prereg01 epoch 1–5）與重現核對（PREREG_02 1.1 第 2、3 點）
# 三項核對任一項不過就停在這裡：不接著訓練、不產生新的 checkpoint
t0 = time.time()
write_keys('inner_keys', INNER_KEYS)
write_keys('val_keys', parse_val()['keys'])

P1_DUMPS = [('p01e1', 1, 1), ('p01e2', 2, 1), ('p01e3', 3, 1), ('p01e4', 4, 1), ('p01e5_d4', 5, NUM_SYMS)]
DUMP_LOG = {}
for _name, _ep, _syms in P1_DUMPS:
    # SMOKE：prereg01 冒煙只訓練 1 個 epoch，epoch 1–5 都用它（只測流程）；正式執行 NUM_EPOCHS＝5，照登記
    _rel = 'ckpt/epoch_%d.pth' % min(_ep, NUM_EPOCHS)
    _wsha, _weights = P01_SHA[_rel], 'prereg01/rank/' + _rel
    t1 = time.time()
    if (dump_is_done('inner_' + _name, len(df_val), _syms, _wsha)
            and dump_is_done('val_' + _name, len(parse_val()['feats']), _syms, _wsha)):
        _st = ['skipped', 'skipped']
        print('  %s：inner、本機驗證題都已完成，跳過' % _name)
    else:
        _net, _ = load_net(os.path.join(P01_RANK, _rel), _wsha)
        _st = [dump_inner(_net, _name, _syms, _weights, _wsha), dump_val(_net, _name, _syms, _weights, _wsha)]
        del _net
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    DUMP_LOG[_name] = {'weights': _weights, 'weights_sha256': _wsha, 'syms': _syms,
                       'inner': _st[0], 'val': _st[1], 'seconds': time.time() - t1}
save_config(dumps_p1=DUMP_LOG)
mark_time('p1_dumps', time.time() - t0)

# ---- 重現核對 ----
t0 = time.time()
V = parse_val()
_parsed = V['keys']['parsed']
with np.load(os.path.join(DUMP_DIR, 'inner_p01e5_d4.npz'), allow_pickle=False) as _z:
    _inner0 = _z['logits'][:, 0, :]
REPRO_INNER_ACC = float((_inner0.argmax(1) == INNER_KEYS['label']).mean() * 100)
with np.load(os.path.join(DUMP_DIR, 'val_p01e5_d4.npz'), allow_pickle=False) as _z:
    _val0 = _z['logits'][:, 0, :]
with open(os.path.join(P01_RANK, 'rank_val_probs.csv'), newline='', encoding='utf-8') as f:
    _p01_rows = list(csv.DictReader(f))

_count_problems = []
if [r['question_id'] for r in _p01_rows] != [q for q, _s, _k in V['spans']]:
    _count_problems.append('題目順序或題數和 prereg01 的 rank_val_probs.csv 不同（%d 對 %d 題）'
                           % (len(V['spans']), len(_p01_rows)))
_max_diff, _b_diff, _a_diff, _compared = 0.0, [], [], 0
if not _count_problems:
    for (_qid, _start, _k), _r in zip(V['spans'], _p01_rows):
        _ok = [_start + j for j in range(_k) if _parsed[_start + j]]
        if int(_r['n_games']) != _k or int(_r['n_parsed']) != len(_ok):
            _count_problems.append('%s：盤數 %d/%d，prereg01 是 %s/%s' % (_qid, len(_ok), _k, _r['n_parsed'], _r['n_games']))
            continue
        if not _ok:
            continue
        # C0：每盤 softmax、同題在 float32 下平均（和 prereg01 推論相同）
        _c0 = [float(v) for v in torch.from_numpy(_val0[_ok]).softmax(1).numpy().mean(0)]
        _old = [float(_r['p_' + lv]) for lv in LEVELS]
        _max_diff = max(_max_diff, max(abs(a - b) for a, b in zip(_c0, _old)))
        _compared += 1
        if rule_b(_c0) != rule_b(_old):
            _b_diff.append(_qid)
        if LEVELS[_argmax_first(_c0)] != LEVELS[_argmax_first(_old)]:
            _a_diff.append(_qid)

GPU_NAME = gpu_name_of(ENV_INFO)   # 這次執行的 GPU（只記錄）
# ② 的 GPU 規則看「寫出 val_p01e5_d4 的那次執行」的 GPU（記在它的完成標記）：斷線後換 GPU 接續，已存的 dump 會跳過，
# 規則不能跟著這次的 GPU 變。正式執行的 dump 一定在 GPU 上算（沒有 GPU 不算），所以沒記到 GPU 就判為不通過
_m_val5 = read_json(_marker_path('val_p01e5_d4')) or {}
DUMP_GPU_NAME = _m_val5.get('gpu_name')
_dump_gpu_ok = SMOKE or DUMP_GPU_NAME is not None
_same_gpu = _dump_gpu_ok and DUMP_GPU_NAME == P01_GPU_NAME
_val_tol = 1e-4 if _same_gpu else 1e-3
# 規則 B 不同的題：正式 ≤ 4 題（1,589 題）；SMOKE 依題數等比例縮小（12 題 → 0 題）
_b_allowed = 4 if not SMOKE else int(4 * len(V['spans']) / 1589)
REPRO = {
    'run': RUN['run'], 'time': now_iso(), 'smoke': SMOKE,
    'gpu_name': GPU_NAME, 'torch': torch.__version__, 'cudnn': ENV_INFO.get('cudnn'),
    'dump_gpu_name': DUMP_GPU_NAME, 'dump_torch': _m_val5.get('torch'), 'dump_cudnn': _m_val5.get('cudnn'),
    'dump_run': _m_val5.get('run'),
    'p01_gpu_name': P01_GPU_NAME, 'p01_torch': P01_TORCH, 'p01_cudnn': P01_CUDNN,
    'same_gpu_as_p01': _same_gpu,
    'same_gpu_rule': '寫出 val_p01e5_d4 的那次執行的 GPU（dumps/_DONE_val_p01e5_d4.json 的 gpu_name）和 prereg01 的 GPU 比',
    'dump_gpu': {'gpu_name': DUMP_GPU_NAME, 'pass': _dump_gpu_ok},
    'inner_acc': {'dump': 'inner_p01e5_d4 s=0', 'value_pct': REPRO_INNER_ACC, 'target_pct': P01_INNER_ACC,
                  'abs_diff_pp': abs(REPRO_INNER_ACC - P01_INNER_ACC), 'tolerance_pp': 0.05,
                  'games': int(len(_inner0)), 'pass': abs(REPRO_INNER_ACC - P01_INNER_ACC) <= 0.05},
    'val_c0_vs_p01': {'dump': 'val_p01e5_d4 s=0', 'reference': 'prereg01/rank/rank_val_probs.csv',
                      'questions_compared': _compared, 'max_abs_diff': _max_diff, 'tolerance': _val_tol,
                      'pass': not _count_problems and _max_diff <= _val_tol},
    'rule_b_diff': {'questions': len(_b_diff), 'allowed': _b_allowed, 'question_ids': _b_diff,
                    'pass': not _count_problems and len(_b_diff) <= _b_allowed},
    'rule_a_diff_info_only': {'questions': len(_a_diff), 'question_ids': _a_diff},
    'counts': {'problems': _count_problems[:20], 'n_problems': len(_count_problems), 'pass': not _count_problems},
}
REPRO['pass'] = all(REPRO[k]['pass'] for k in ('inner_acc', 'val_c0_vs_p01', 'rule_b_diff', 'counts', 'dump_gpu'))
_repro_path = os.path.join(RESULT_DIR, 'repro_check.json')
_prev = read_json(_repro_path)
REPRO['earlier_runs'] = ((_prev.get('earlier_runs') or []) + [{k: v for k, v in _prev.items() if k != 'earlier_runs'}]
                         if _prev else [])
write_json(_repro_path, REPRO)
save_config(repro_check={k: REPRO[k] for k in ('pass', 'gpu_name', 'dump_gpu_name', 'p01_gpu_name', 'same_gpu_as_p01',
                                                 'inner_acc', 'val_c0_vs_p01', 'rule_b_diff', 'counts')})
mark_time('repro_check', time.time() - t0)
print('重現核對：① inner accuracy %.4f%%（prereg01 %.4f%%，差 %.4f 個百分點，容許 0.05）%s'
      % (REPRO_INNER_ACC, P01_INNER_ACC, abs(REPRO_INNER_ACC - P01_INNER_ACC), '通過' if REPRO['inner_acc']['pass'] else '沒過'))
print('          ② 每題平均機率最大差距 %.3g（容許 %g；算 dump 的 GPU %s，prereg01 %s，這次 %s）%s'
      % (_max_diff, _val_tol, DUMP_GPU_NAME, P01_GPU_NAME, GPU_NAME, '通過' if REPRO['val_c0_vs_p01']['pass'] else '沒過'))
print('          ③ 規則 B 不同 %d 題（容許 %d）%s' % (len(_b_diff), _b_allowed, '通過' if REPRO['rule_b_diff']['pass'] else '沒過'))
if _count_problems:
    print('          盤數或題目對不上：', _count_problems[:5])
if not _dump_gpu_ok:
    print('          val_p01e5_d4 的完成標記沒有記到 GPU（正式執行的逐盤輸出必須在 GPU 上算）')
if not REPRO['pass']:
    # 停下前照樣比對 prereg01 指紋、打包（查原因要用逐盤輸出），並把 Drive 的寫入送出
    _after, _p01_changed = record_p01_after()
    RUN.update(end=now_iso(), status='repro_failed')
    save_run_log()
    _zip, _ = pack_results('rank_results_repro_failed.zip')
    if _p01_changed:
        print('另外：prereg01 的目錄在這次執行中被改動（%d 個檔：%s）' % (len(_p01_changed), _p01_changed[:10]))
    flush_drive()
    raise SystemExit('重現核對沒過，依 PREREG_02 停在這裡：不接著訓練、不產生新的 checkpoint。'
                     '處理方式：不要再執行；下載 %s，把這段訊息截圖，一起給 Claude。' % _zip)
print('重現核對都通過')
"""

TRAIN = r"""# 本版新增：從 prereg01 的 state.pt 接著訓練 EXTRA_EPOCHS 個 epoch（編號 NUM_EPOCHS+1..FINAL_EPOCH）
# 學習率每步線性降到 0；其餘（模型、特徵、batch、fp32、沒有資料增強）和 prereg01 相同。不產生 best_model.pth
# 收尾開關 FINISH_WITHOUT_TRAINING（設定格）打開時不訓練：已完成的 epoch 從 prereg02 的 state.pt 讀回（登記 1.2 的退路）
from torch.optim.lr_scheduler import LambdaLR

P01_STATE_REL = 'ckpt/state.pt'
P01_LAST_REL = 'ckpt/epoch_%d.pth' % NUM_EPOCHS
STATE_PATH = os.path.join(CKPT_DIR, 'state.pt')   # prereg02 自己的 state.pt
T_TOTAL = EXTRA_EPOCHS * P01_STEPS                 # 正式：4 × 2,814 = 11,256 步
if not SMOKE:
    assert P01_STEPS == 2814 and T_TOTAL == 11256, (P01_STEPS, T_TOTAL)
# prereg01 的 state.pt 必須是這個設定（和 prereg01 notebook 的 STATE_CONFIG 相同）
STATE_CONFIG_P01 = {'SEED': SEED, 'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE,
                    'NUM_EPOCHS': NUM_EPOCHS, 'VAL_SIZE': VAL_SIZE, 'features': FEAT_FINGERPRINT}
# prereg02 的 state.pt：設定和這次不同就不接續
STATE_CONFIG_P02 = {'SEED': SEED, 'BATCH_SIZE': BATCH_SIZE, 'PEAK_LR': PEAK_LR,
                    'SCHEDULE': 'LambdaLR: lr = PEAK_LR * max(0, 1 - t / T)；每次 optimizer.step() 後 scheduler.step()',
                    'EXTRA_EPOCHS': EXTRA_EPOCHS, 'T': T_TOTAL, 'P01_STEPS': P01_STEPS, 'NUM_EPOCHS': NUM_EPOCHS,
                    'VAL_SIZE': VAL_SIZE, 'features': FEAT_FINGERPRINT,
                    'init_state_sha256': P01_SHA[P01_STATE_REL], 'init_weights_sha256': P01_SHA[P01_LAST_REL]}
P02_STATE_RESTART = ('在 Drive 把 %s 整個資料夾改名保留（例如 ckpt_broken_日期），再按一次「全部執行」：'
                     '會從 prereg01 的 state.pt 重新接著訓練 %d 個 epoch，逐盤輸出不受影響。把這件事告訴 Claude。'
                     '不要只刪 state.pt，否則舊的 epoch_N.pth 會和新的混在一起。' % (CKPT_DIR, EXTRA_EPOCHS))


def lr_lambda(t):
    return max(0.0, 1.0 - t / T_TOTAL)


def expected_lr(t):
    return PEAK_LR * lr_lambda(t)


def save_state_p02(epoch_done):
    # 這個 epoch 的 epoch_N.pth 存好之後才存 state.pt；存好才算這個 epoch 完成
    tmp = guard_write(STATE_PATH + '.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'global_step': global_step, 'epoch': epoch_done, 'history': history,
                'config': STATE_CONFIG_P02}, tmp)
    os.replace(tmp, guard_write(STATE_PATH))


def _read_checked(path, expect_sha):
    with open(path, 'rb') as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != expect_sha:
        raise SystemExit('%s 的 SHA-256 和開頭記下的不同（執行中被改動？）。請把這段訊息截圖給 Claude。' % path)
    return raw


def expected_train_steps():
    # 每個 epoch 的步數可以事先算：webdataset 1.0.2 先依 DataLoader worker 在排序後的分片清單上 islice(w, None, workers)
    # 分配分片，再在各 worker 內洗牌；每個 worker 最後一批不滿也算一步（partial=True）。抽特徵的 worker w 依序寫出
    # train-w<w>-000000.tar.gz…，每片 PER_SHARD 盤、最後一片是餘數（_count_w<w>.json 的 written）。回傳 (步數或 None, 說明)
    sizes = {}
    for cpath in sorted(glob.glob(os.path.join(FEAT_DIR, 'train', '_count_w*.json'))):
        wid = int(os.path.basename(cpath)[len('_count_w'):-len('.json')])
        written = int(read_json(cpath)['written'])
        shards = sorted(glob.glob(os.path.join(FEAT_DIR, 'train', 'train-w%d-*.tar.gz' % wid)))
        if len(shards) != -(-written // PER_SHARD):
            return None, ('抽特徵的 worker %d 有 %d 片分片，照 %d 盤、每片 %d 盤應該是 %d 片'
                          % (wid, len(shards), written, PER_SHARD, -(-written // PER_SHARD)))
        for i, s in enumerate(shards):
            sizes[os.path.normpath(s)] = PER_SHARD if i < len(shards) - 1 else written - PER_SHARD * (len(shards) - 1)
    listed = [os.path.normpath(s) for s in train_shards]   # 訓練 loader 實際用的清單和順序
    if sorted(sizes) != sorted(listed):
        return None, '訓練分片清單（%d 片）和 _count_w*.json 推出的分片（%d 片）對不上' % (len(listed), len(sizes))
    if sum(sizes.values()) != train_counts['written']:
        return None, '分片盤數合計 %d，訓練實際寫入 %d' % (sum(sizes.values()), train_counts['written'])
    n_loaders = max(workers, 1)
    per_loader = [sum(sizes[listed[i]] for i in range(w, len(listed), n_loaders)) for w in range(n_loaders)]
    return sum(-(-n // BATCH_SIZE) for n in per_loader if n), {'loader_workers': workers,
                                                               'samples_per_loader_worker': per_loader,
                                                               'shards': len(listed)}


RUN['finish_without_training'] = FINISH_WITHOUT_TRAINING
if FINISH_WITHOUT_TRAINING:
    # 收尾開關：不再訓練（登記 1.2：第 9 個 epoch 沒跑完就只用 prereg01 的輸出）。已完成的 epoch 從 prereg02 的 state.pt 讀回
    t0 = time.time()
    history, global_step, INIT_FROM = [], 0, None
    if os.path.exists(STATE_PATH):
        try:
            _state = torch.load(STATE_PATH, map_location='cpu', weights_only=False)
            history, global_step = list(_state.get('history') or []), int(_state.get('global_step') or 0)
            INIT_FROM = 'prereg02/rank/ckpt/state.pt（只讀回 history）'
            del _state
        except Exception as e:
            print('注意：prereg02 的 ckpt/state.pt 讀不進來（%s: %s），已完成的 epoch 記為沒有' % (type(e).__name__, e))
    save_run_log()
    save_config(training={'finish_without_training': True, 'init_from': INIT_FROM,
                          'epochs_in_state': [h['epoch'] for h in history], 'global_step_in_state': global_step})
    print('收尾開關打開：不接著訓練。prereg02 的 state.pt 記錄已完成的 epoch：%s' % [h['epoch'] for h in history])
    mark_time('train_init', time.time() - t0)
else:
    # 每個 epoch 的步數必須和 prereg01 相同（登記的排程是 EXTRA_EPOCHS × P01_STEPS 步）：訓練前先算好，不同就停下
    _steps, _steps_info = expected_train_steps()
    save_config(train_steps_check={'expected_steps_per_epoch': _steps, 'p01_steps_per_epoch': P01_STEPS,
                                   'detail': _steps_info})
    if _steps != P01_STEPS:
        raise SystemExit('接著訓練每個 epoch 會跑 %s 步，和 prereg01 的 %d 步不同（%s）。登記的學習率排程是 %d × %d 步，'
                         '所以停在這裡，還沒開始訓練（逐盤輸出已存好）。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                         % (_steps, P01_STEPS, _steps_info, EXTRA_EPOCHS, P01_STEPS))
    print('每個 epoch %d 步，和 prereg01 相同（%s）' % (_steps, _steps_info))

    t0 = time.time()
    RUN['loaded_state'] = False
    if os.path.exists(STATE_PATH):
        # 接續：prereg02 自己的 state.pt（自己存的檔，含 history）
        try:
            state = torch.load(STATE_PATH, map_location=DEVICE, weights_only=False)
            _missing = [k for k in ('model', 'optimizer', 'scheduler', 'global_step', 'epoch', 'history', 'config')
                        if k not in state]
        except Exception as e:
            raise SystemExit('prereg02 的 ckpt/state.pt 讀不進來（%s: %s）。處理方式：%s'
                             % (type(e).__name__, e, P02_STATE_RESTART)) from e
        if _missing:
            raise SystemExit('prereg02 的 ckpt/state.pt 缺少 %s。處理方式：%s' % (_missing, P02_STATE_RESTART))
        if state['config'] != STATE_CONFIG_P02:
            _diff = sorted(k for k in set(state['config']) | set(STATE_CONFIG_P02)
                           if state['config'].get(k) != STATE_CONFIG_P02.get(k))
            raise SystemExit('prereg02 的 ckpt/state.pt 的設定和這次不同，不能接續。不同的項目：%s。處理方式：刻意改的話，%s'
                             '不是刻意改的，先查出哪裡變了再跑。' % (_diff, P02_STATE_RESTART))
        try:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler = LambdaLR(optimizer, lr_lambda)
            scheduler.load_state_dict(state['scheduler'])
        except Exception as e:
            raise SystemExit('prereg02 的 ckpt/state.pt 載不進模型、optimizer 或 scheduler（%s: %s）。處理方式：%s'
                             % (type(e).__name__, e, P02_STATE_RESTART)) from e
        start_epoch, global_step, history = state['epoch'], state['global_step'], state['history']
        INIT_FROM = 'prereg02/rank/ckpt/state.pt'
        RUN['loaded_state'] = True
        del state
    else:
        # 第一次：唯讀 prereg01 的 state.pt（第 5 個 epoch 結束，含 AdamW 狀態），核對 SHA-256、設定、權重
        state = torch.load(io.BytesIO(_read_checked(os.path.join(P01_RANK, P01_STATE_REL), P01_SHA[P01_STATE_REL])),
                           map_location=DEVICE, weights_only=False)   # prereg01 自己存的檔，含 history
        _problems = []
        if state.get('epoch') != NUM_EPOCHS:
            _problems.append('epoch 是 %s，應該是 %d' % (state.get('epoch'), NUM_EPOCHS))
        if state.get('config') != STATE_CONFIG_P01:
            _cfg = state.get('config') or {}
            _problems.append('設定不同：%s' % sorted(k for k in set(_cfg) | set(STATE_CONFIG_P01)
                                                    if _cfg.get(k) != STATE_CONFIG_P01.get(k)))
        if not state.get('history') or state['history'][-1].get('steps') != P01_STEPS:
            _problems.append('history 最後一個 epoch 的步數不是 %d' % P01_STEPS)
        if [g['lr'] for g in state['optimizer']['param_groups']] != [PEAK_LR]:
            _problems.append('optimizer 的學習率不是 %g' % PEAK_LR)
        _w_last = torch.load(io.BytesIO(_read_checked(os.path.join(P01_RANK, P01_LAST_REL), P01_SHA[P01_LAST_REL])),
                             map_location='cpu')
        if sorted(_w_last) != sorted(state['model']) or not all(
                torch.equal(state['model'][k].cpu(), _w_last[k]) for k in _w_last):
            _problems.append('state.pt 的權重和 %s 不是逐張量相同' % P01_LAST_REL)
        del _w_last
        if _problems:
            raise SystemExit('prereg01 的 state.pt 和登記的起點不符：%s。%s' % ('；'.join(_problems), P01_HELP))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler = LambdaLR(optimizer, lr_lambda)   # 還沒有 initial_lr，會設成現在的 1e-3
        start_epoch, global_step, history = NUM_EPOCHS, 0, []
        INIT_FROM = 'prereg01/rank/ckpt/state.pt'
        del state
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    # LambdaLR 建立時會把學習率設成 λ(0)；接續時照已存的步數重設，確保下一步的學習率正確
    for _g in optimizer.param_groups:
        _g['lr'] = _g['initial_lr'] * lr_lambda(global_step)
    assert scheduler.last_epoch == global_step, (scheduler.last_epoch, global_step)
    assert [g['initial_lr'] for g in optimizer.param_groups] == [PEAK_LR]
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr(global_step)) < 1e-15

    RUN['start_epoch'] = start_epoch
    if start_epoch < FINAL_EPOCH and RUN['loaded_state']:
        run_log['resume_count'] += 1
        RUN.update(resumed_training=True, resume_time=now_iso())
    save_run_log()
    save_config(training={
        'init_from': INIT_FROM, 'start_epoch': start_epoch, 'final_epoch': FINAL_EPOCH, 'global_step_at_start': global_step,
        'state_config': {k: v for k, v in STATE_CONFIG_P02.items() if k != 'features'},
        'optimizer': {'class': type(optimizer).__name__, **optimizer.defaults},
        'criterion': type(criterion).__name__, 'device': str(DEVICE),
        'device_name': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None,
        'tf32': {'cudnn': torch.backends.cudnn.allow_tf32, 'matmul': torch.backends.cuda.matmul.allow_tf32},
        'loader_workers': workers, 'seed_rule': 'random／numpy／torch 的 seed 在每個 epoch 開始時設為 SEED + epoch',
        'checkpoint_rule': '指定 epoch_%d.pth（最後一個 epoch），不另選 best；inner 驗證只當監看' % FINAL_EPOCH})
    save_config_summary(resumed=run_log['resume_count'] > 0, resume_count=run_log['resume_count'])
    if start_epoch >= FINAL_EPOCH:
        print('epoch %d–%d 都已訓練完（global_step %d），這次不訓練' % (NUM_EPOCHS + 1, FINAL_EPOCH, global_step))
    else:
        print('從 %s 開始：epoch %d → %d，global_step %d／%d，學習率 %.3g'
              % (INIT_FROM, start_epoch + 1, FINAL_EPOCH, global_step, T_TOTAL, optimizer.param_groups[0]['lr']))
    mark_time('train_init', time.time() - t0)

    model.train()
    for epoch in range(start_epoch, FINAL_EPOCH):
        t0 = time.time()
        random.seed(SEED + epoch + 1)
        np.random.seed(SEED + epoch + 1)
        torch.manual_seed(SEED + epoch + 1)
        total_loss, correct, total, steps = 0.0, 0, 0, 0
        lr_start = optimizer.param_groups[0]['lr']

        for inputs, labels, _ in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()      # 本版：每步更新學習率
            global_step += 1

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            steps += 1
            if steps % 500 == 0:
                print('  ep%d step %d  loss %.4f  acc %.2f%%  lr %.3g  %.1f 分'
                      % (epoch + 1, steps, total_loss / steps, 100 * correct / total,
                         optimizer.param_groups[0]['lr'], (time.time() - t0) / 60), flush=True)

        assert abs(optimizer.param_groups[0]['lr'] - expected_lr(global_step)) < 1e-12, 'learning-rate schedule drifted'
        train_acc = 100 * correct / total if total else 0.0
        val_loss, val_acc = validate(model, val_loader, criterion)   # inner 逐盤 accuracy：只當監看
        history.append({'epoch': epoch + 1, 'train_acc': train_acc, 'val_acc': val_acc,
                        'train_loss': total_loss / steps if steps else 0.0, 'val_loss': val_loss,
                        'steps': steps, 'samples': total, 'seconds': time.time() - t0, 'run': RUN['run'],
                        'lr_start': lr_start, 'lr_end': optimizer.param_groups[0]['lr'], 'global_step': global_step})
        print('Epoch %d/%d  train acc %.2f%%  |  inner loss %.4f acc %.2f%%  |  lr %.3g → %.3g  |  %.1f 分'
              % (epoch + 1, FINAL_EPOCH, train_acc, val_loss, val_acc, lr_start, optimizer.param_groups[0]['lr'],
                 (time.time() - t0) / 60), flush=True)

        _tmp = guard_write(os.path.join(CKPT_DIR, 'epoch_%d.pth.tmp' % (epoch + 1)))
        torch.save(model.state_dict(), _tmp)
        os.replace(_tmp, guard_write(os.path.join(CKPT_DIR, 'epoch_%d.pth' % (epoch + 1))))
        save_state_p02(epoch + 1)   # 最後才存；存好才算這個 epoch 完成
        pd.DataFrame(history).to_csv(guard_write(os.path.join(CKPT_DIR, 'history.csv')), index=False)
        RUN['epochs_trained'].append(epoch + 1)
        save_run_log()
        save_config(epochs_trained=list(RUN['epochs_trained']))
        save_config_summary(epochs_completed=epoch + 1, global_step=global_step)
        mark_time('epoch_%d' % (epoch + 1), time.time() - t0)

if history:
    pd.DataFrame(history).to_csv(guard_write(os.path.join(CKPT_DIR, 'history.csv')), index=False)
TRAIN_COMPLETE = ([h['epoch'] for h in history] == list(range(NUM_EPOCHS + 1, FINAL_EPOCH + 1))
                  and global_step == sum(h['steps'] for h in history))
save_config_summary(training_complete=TRAIN_COMPLETE, epochs_completed=history[-1]['epoch'] if history else NUM_EPOCHS,
                    trained_in_runs=sorted({h['run'] for h in history}), finish_without_training=FINISH_WITHOUT_TRAINING)
print('接著訓練%s：epoch %s，global_step %d%s'
      % ('完成' if TRAIN_COMPLETE else '沒有完成', [h['epoch'] for h in history], global_step,
         '（收尾開關打開，這次不訓練）' if FINISH_WITHOUT_TRAINING
         else '，最後的學習率 %.3g' % optimizer.param_groups[0]['lr']))
"""

P2 = r"""# 本版新增：P2 逐盤輸出（接著訓練的最後一個 epoch，正式是 epoch_9.pth，8 種對稱）
# 第 FINAL_EPOCH 個 epoch 沒完成就跳過並記錄（本機會照登記拿掉用到它的組合）
t0 = time.time()
P2_NAME = 'p02e9_d4'
_p2_path = os.path.join(CKPT_DIR, 'epoch_%d.pth' % FINAL_EPOCH)
if TRAIN_COMPLETE and os.path.exists(_p2_path):
    _net, _wsha = load_net(_p2_path)
    _weights = 'prereg02/rank/ckpt/epoch_%d.pth' % FINAL_EPOCH
    _st = [dump_inner(_net, P2_NAME, NUM_SYMS, _weights, _wsha), dump_val(_net, P2_NAME, NUM_SYMS, _weights, _wsha)]
    del _net
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    with np.load(os.path.join(DUMP_DIR, 'inner_%s.npz' % P2_NAME), allow_pickle=False) as _z:
        _acc9 = float((_z['logits'][:, 0, :].argmax(1) == INNER_KEYS['label']).mean() * 100)
    P2_RESULT = {'done': True, 'weights': _weights, 'weights_sha256': _wsha, 'inner': _st[0], 'val': _st[1],
                 'inner_acc_from_dump_pct': _acc9, 'inner_acc_from_history_pct': history[-1]['val_acc'],
                 'abs_diff_pp': abs(_acc9 - history[-1]['val_acc'])}
    if P2_RESULT['abs_diff_pp'] > 0.05:
        print('注意：epoch %d 的 inner accuracy，dump 算出 %.4f%%，訓練時的監看是 %.4f%%，差超過 0.05 個百分點。'
              '結果照樣打包，回報 Claude 時提一下。' % (FINAL_EPOCH, _acc9, history[-1]['val_acc']))
    print('P2 逐盤輸出完成：%s（epoch %d 的 inner accuracy %.4f%%）' % (P2_NAME, FINAL_EPOCH, _acc9))
else:
    P2_RESULT = {'done': False, 'reason': '第 %d 個 epoch 沒有完成（TRAIN_COMPLETE=%s，%s %s%s）'
                 % (FINAL_EPOCH, TRAIN_COMPLETE, _p2_path, '存在' if os.path.exists(_p2_path) else '不存在',
                    '；收尾開關打開' if FINISH_WITHOUT_TRAINING else '')}
    print('跳過 P2 逐盤輸出：%s' % P2_RESULT['reason'])
P2_RESULT['finish_without_training'] = FINISH_WITHOUT_TRAINING
save_config(dumps_p2=P2_RESULT)
mark_time('p2_dumps', time.time() - t0)
"""

FINAL = r"""# 本版改寫的結尾：核對所有逐盤輸出，比對 prereg01 目錄前後的指紋，補完 config／timings／run_log，
# 打包 rank_results.zip（不含權重），把 Drive 的寫入送出
import zipfile

t0 = time.time()
V = parse_val()
_parsed = V['keys']['parsed']
_expect = ['inner_keys', 'val_keys'] + ['%s_%s' % (p, n) for n, _e, _s in P1_DUMPS for p in ('inner', 'val')]
if P2_RESULT['done']:
    _expect += ['inner_' + P2_NAME, 'val_' + P2_NAME]
DUMP_CHECK = {}
for stem in _expect:
    _path, _m = _dump_path(stem), read_json(_marker_path(stem))
    _sha = sha256_path(_path) if os.path.exists(_path) else None
    assert _m is not None and _sha == _m['sha256'], ('完成標記不符', stem)
    _rows_expected = len(df_val) if stem.startswith('inner') else len(V['feats'])
    with np.load(_path, allow_pickle=False) as _z:
        if stem.endswith('_keys'):
            _rows = len(_z['game_id'] if stem.startswith('inner') else _z['question_id'])
            assert all(len(_z[k]) == _rows for k in _z.files), stem
        else:
            _L = _z['logits']
            _rows = _L.shape[0]
            assert _L.dtype == np.float32 and _L.ndim == 3 and _L.shape[1:] == (int(_z['syms']), NUM_CLASSES), stem
            assert str(_z['weights_sha256']) == _m['weights_sha256'], stem
            if stem.startswith('val_'):
                assert np.array_equal(np.isnan(_L).all(axis=(1, 2)), ~_parsed) and not np.isnan(_L[_parsed]).any(), stem
            else:
                assert not np.isnan(_L).any(), stem
    assert _rows == _rows_expected == _m['rows'], (stem, _rows, _rows_expected, _m['rows'])
    assert os.path.getsize(_path) < DUMP_MAX_BYTES, stem
    DUMP_CHECK[stem] = {'rows': _rows, 'bytes': os.path.getsize(_path), 'sha256': _sha,
                        'weights': _m['weights'], 'weights_sha256': _m['weights_sha256']}

# prereg01 目錄的指紋（結尾）：必須和這次開頭記的完全相同
P01_AFTER, _p01_changed = record_p01_after()

OUTPUTS = {
    'dumps': DUMP_CHECK,
    'inner_games': len(df_val), 'val_questions': len(V['spans']), 'val_games': len(V['feats']),
    'val_games_unparsed': int((~_parsed).sum()),
    'repro_check_pass': REPRO['pass'], 'training_complete': TRAIN_COMPLETE, 'p2_dumps_done': P2_RESULT['done'],
    'finish_without_training': FINISH_WITHOUT_TRAINING,
    'p01_readonly_unchanged': not _p01_changed, 'p01_files_fingerprinted': len(P01_AFTER),
    'history': history,
    'files': {n: os.path.getsize(os.path.join(RESULT_DIR, n)) for n in sorted(os.listdir(RESULT_DIR))
              if os.path.isfile(os.path.join(RESULT_DIR, n))},
}
RUN.update(end=now_iso(), status='completed' if not _p01_changed else 'p01_changed')
save_config(outputs=OUTPUTS, end=RUN['end'])
save_run_log()
mark_time('final_check', time.time() - t0)
mark_time('total_this_run', time.time() - RUN_START)
print(json.dumps({k: v for k, v in OUTPUTS.items() if k not in ('dumps', 'history')}, ensure_ascii=False, indent=1))
print('ckpt/', sorted(os.listdir(CKPT_DIR)))

# 結果打包：結果目錄裡除了權重和暫存檔以外的檔案，打成一個 zip 放在結果目錄旁邊
RESULTS_ZIP, _skipped_big = pack_results(os.path.basename(RESULT_DIR) + '_results.zip')
with zipfile.ZipFile(RESULTS_ZIP) as _zf:
    _names = set(_zf.namelist())
_missing_in_zip = [s for s in _expect if 'rank/dumps/%s.npz' % s not in _names]
assert not _missing_in_zip, ('dump 沒進 zip', _missing_in_zip)
_zip_mb = os.path.getsize(RESULTS_ZIP) / 1e6
# 下載 zip、刪除執行階段之前，先把 Drive 的寫入全部送出（epoch_9.pth、state.pt 之後還要用）
flush_drive()
if _p01_changed:
    raise SystemExit('prereg01 的目錄在這次執行中被改動（%d 個檔：%s）。結果已打包在 %s，但先不要用；'
                     '把這段訊息截圖給 Claude。' % (len(_p01_changed), _p01_changed[:10], RESULTS_ZIP))
print('完成。請下載這個檔回傳給 Claude（合併、校準、打分都在本機做）：%s（%.1f MB）' % (RESULTS_ZIP, _zip_mb))
"""

# ---------------------------------------------------------------------------
# Assemble. Each entry: (kind, key or tag, source)
#   kind "md_new" / "code_new": new cell; tag names it
#   kind "base": reused base cell (source may differ from the base only via rep())
#   kind "rewrite": base cell replaced entirely
# ---------------------------------------------------------------------------

_setup = base_src(("new", "setup"))
_setup = rep(_setup, SETUP_OLD_START, SETUP_NEW_START)
_setup = rep(_setup, SETUP_OLD_DIRS, SETUP_NEW_DIRS)
_setup = rep(_setup, SETUP_OLD_DEV, SETUP_NEW_DEV)
_setup = rep(_setup, SETUP_OLD_WJ, SETUP_NEW_WJ)
_setup = rep(_setup, SETUP_OLD_PREREG, SETUP_NEW_PREREG)
_setup = rep(_setup, SETUP_OLD_PRINT, SETUP_NEW_PRINT)

_off2 = base_src(("official", 2))
_off2 = rep(_off2, OFF2_OLD_DEVICE, OFF2_NEW_DEVICE)
_off2 = rep(_off2, OFF2_OLD_MKDIR, OFF2_NEW_MKDIR)
_off2 = rep(_off2, OFF2_OLD_PRINT, OFF2_NEW_PRINT)

_ft = base_src(("new", "feature_backup_tools"))
for _old, _new in ((FT_OLD_BACKUP, FT_NEW_BACKUP), (FT_OLD_DST, FT_NEW_DST), (FT_OLD_SIG, FT_NEW_SIG),
                   (FT_OLD_MARKER, FT_NEW_MARKER), (FT_OLD_NOBACKUP, FT_NEW_NOBACKUP),
                   (FT_OLD_CHECKSRC, FT_NEW_CHECKSRC), (FT_OLD_COPY, FT_NEW_COPY), (FT_OLD_RET, FT_NEW_RET)):
    _ft = rep(_ft, _old, _new)
assert "FEAT_MARKER" in _ft and "read_json(FEAT_MARKER)" not in _ft

_off9 = base_src(("official", 9))
_off9 = rep(_off9, OFF9_OLD_RESTORE, OFF9_NEW_RESTORE)
_off9 = rep(_off9, OFF9_OLD_RMTREE, OFF9_NEW_RMTREE)
_off9 = rep(_off9, OFF9_OLD_MODE, OFF9_NEW_MODE)

PLAN = [
    ("md_new", "header", MD_HEADER),
    ("base", ("new", "setup"), _setup),
    ("code_new", "preflight", PREFLIGHT),
    ("base", ("new", "prepare"), base_src(("new", "prepare"))),
    ("base", ("official", 1), base_src(("official", 1))),
    ("base", ("new", "environment"), base_src(("new", "environment"))),
    ("base", ("official", 2), _off2),
    ("base", ("official", 4), base_src(("official", 4))),
    ("base", ("official", 6), base_src(("official", 6))),
    ("base", ("official", 8), base_src(("official", 8))),
    ("base", ("new", "feature_backup_tools"), _ft),
    ("base", ("official", 9), _off9),
    ("base", ("official", 11), base_src(("official", 11))),
    ("base", ("official", 12), base_src(("official", 12))),
    ("base", ("official", 14), base_src(("official", 14))),
    ("base", ("official", 15), base_src(("official", 15))),
    ("code_new", "dump_tools", DUMP_TOOLS),
    ("code_new", "p1_dumps_and_repro", P1),
    ("code_new", "continued_training", TRAIN),
    ("code_new", "p2_dumps", P2),
    ("rewrite", ("new", "final_check"), FINAL),
]

nb = nbformat.v4.new_notebook()
nb.metadata = json.loads(json.dumps(base["metadata"]))
nb.metadata["colab"] = {"provenance": []}   # no preset GPU or high-RAM shape (as the player notebook): choose G4

_used = set()
for k, (kind, key, src) in enumerate(PLAN):
    src = src.rstrip("\n")
    if kind == "md_new":
        cell = nbformat.v4.new_markdown_cell(src)
    else:
        # every code cell must be plain Python (the smoke runner compiles whole cells too)
        compile(src, "<cell %d>" % k, "exec")
        cell = nbformat.v4.new_code_cell(src)
    cell["id"] = "c10-rank02-%02d" % k
    if kind in ("base", "rewrite"):
        j = BASE_BY_KEY[key]
        assert key not in _used, key
        _used.add(key)
        b = BASE_CELLS[j]
        assert b["cell_type"] == "code", key
        cell["metadata"]["c10"] = {"base_cell": j, "base_id": b["id"], "base": b["metadata"]["c10"],
                                   "changed": src != "".join(b["source"])}
        if kind == "rewrite":
            cell["metadata"]["c10"]["rewritten"] = True
            assert cell["metadata"]["c10"]["changed"], key
    else:
        cell["metadata"]["c10"] = {"new": key}
    nb.cells.append(cell)
nbformat.validate(nb)

# Integrity: changed base cells <-> CHANGE_ROWS; unchanged base cells are verbatim; ids unique;
# the forbidden token appears only in the unchanged prepare cell's refusal.
_changed = {tuple(c["metadata"]["c10"]["base"].items()) for c in nb.cells
            if "base_cell" in c["metadata"]["c10"] and c["metadata"]["c10"]["changed"]}
_changed_keys = {base_key(dict(t)) for t in _changed}
_named = {k for k, _l, _t in CHANGE_ROWS if k is not None}
assert _changed_keys == _named, ("changed but not named", _changed_keys - _named, "named but unchanged",
                                 _named - _changed_keys)
for c in nb.cells:
    m = c["metadata"]["c10"]
    if "base_cell" in m and not m["changed"]:
        assert c["source"] == "".join(BASE_CELLS[m["base_cell"]]["source"]), m
assert len({c["id"] for c in nb.cells}) == len(nb.cells)
_prep = base_src(("new", "prepare"))
_tok_allowed = _prep.count(FORBIDDEN_TOKEN)
assert _tok_allowed == 1 and ("'%s' in p for p in _all_files" % FORBIDDEN_TOKEN) in _prep
assert sum(c["source"].count(FORBIDDEN_TOKEN) for c in nb.cells) == _tok_allowed, "forbidden token in a cell"
_prep_out = [c for c in nb.cells if c["metadata"]["c10"].get("base", {}).get("new") == "prepare"]
assert len(_prep_out) == 1 and not _prep_out[0]["metadata"]["c10"]["changed"]

os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
with open(OUT_NB, "w", encoding="utf-8", newline="\n") as f:
    nbformat.write(nb, f)

# ---------------------------------------------------------------------------
# Per-cell unified diff against colab/prereg01_rank.ipynb, plus the notebook metadata
# ---------------------------------------------------------------------------

out = []
out.append("# prereg02 rank：colab/prereg01_rank.ipynb 與 colab/prereg02_rank.ipynb 的逐格 unified diff（程式格、markdown 格、notebook metadata）")
out.append("# 底稿：%s  sha256=%s" % (BASE_REL, sha(BASE_NB)))
out.append("# 本版：%s  sha256=%s" % (OUT_REL, sha(OUT_NB)))
out.append("# 產生方式：colab/build_prereg02_rank_nb.py；difflib.unified_diff，依 cell metadata c10.base_cell 配對底稿的格；"
           "新增的格對空白比。cell 編號從 0 起算。")
out.append("# 每一處改動的說明見本版第 0 格「和 prereg01 notebook 的差異」；沒列在表上的沿用格和底稿逐字相同。")
summary = []
body = []
for k, cell in enumerate(nb.cells):
    meta = cell["metadata"]["c10"]
    kind = "程式格" if cell["cell_type"] == "code" else "markdown"
    if "base_cell" in meta:
        j = meta["base_cell"]
        old_src = "".join(BASE_CELLS[j]["source"])
        bm = meta["base"]
        what = "官方 %d" % bm["official_cell"] if "official_cell" in bm else bm["new"]
        a = "prereg01_rank.ipynb cell %d（%s）" % (j, what)
        status = "改寫" if meta.get("rewritten") else ("改動" if meta["changed"] else "完全相同")
    else:
        old_src = ""
        a = "/dev/null（新增：%s）" % meta["new"]
        status = "新增"
    b = "prereg02_rank.ipynb cell %d" % k
    summary.append("#   本版 cell %2d（%s）← %s：%s" % (k, kind, a, status))
    diff = list(difflib.unified_diff(old_src.splitlines(), cell["source"].splitlines(),
                                     fromfile=a, tofile=b, lineterm=""))
    body.append("")
    body.append("=" * 100)
    body.append("本版 cell %d（%s）↔ %s（%s）" % (k, kind, a, status))
    body.append("=" * 100)
    body.extend(diff if diff else ["（完全相同）"])
_dropped = []
for j, c in enumerate(BASE_CELLS):
    if base_key(c["metadata"]["c10"]) not in _used:
        bm = c["metadata"]["c10"]
        _dropped.append("#   prereg01 cell %2d（%s，%s）" % (
            j, "程式格" if c["cell_type"] == "code" else "markdown",
            "官方 %d" % bm["official_cell"] if "official_cell" in bm else bm["new"]))
out.append("# 對照表：")
out.extend(summary)
out.append("# 沒有沿用的底稿格（%d 格）：" % len(_dropped))
out.extend(_dropped)
out.extend(body)
_meta_old = json.dumps(base["metadata"], ensure_ascii=False, indent=1, sort_keys=True).splitlines()
_meta_new = json.dumps(nb.metadata, ensure_ascii=False, indent=1, sort_keys=True).splitlines()
out.append("")
out.append("=" * 100)
out.append("notebook metadata")
out.append("=" * 100)
out.extend(list(difflib.unified_diff(_meta_old, _meta_new, fromfile="prereg01 metadata",
                                     tofile="本版 metadata", lineterm="")) or ["（完全相同）"])
os.makedirs(os.path.dirname(OUT_DIFF), exist_ok=True)
with open(OUT_DIFF, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print("wrote", OUT_NB, sha(OUT_NB))
print("wrote", OUT_DIFF, sha(OUT_DIFF))
print("cells", len(nb.cells), "| base cells changed",
      sorted("%s %s" % k for k in _changed_keys), "| dropped", len(_dropped))
