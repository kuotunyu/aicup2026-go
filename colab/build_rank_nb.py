"""Build colab/prereg01_rank.ipynb from the official rank-prediction.ipynb and
write results/prereg01/notebook_diff_rank.txt (per-cell unified diff of every
code and markdown cell, plus the notebook metadata).

Official cells that are not modified are copied verbatim (same source, same id).
Every official cell that is modified must be named in DEV_ROWS; the build stops
otherwise. The same DEV_ROWS feeds the markdown table (notebook cell 1) and the
`deviations_from_official` list written to config.json.

Run with the smoke venv python from anywhere:
    python build_rank_nb.py D:/AI-Portfolio/competitions/aicup2026-go
"""

import difflib
import hashlib
import json
import os
import sys

import nbformat

REPO = sys.argv[1]
OFFICIAL = os.path.join(REPO, "third_party", "aicup2026_tutorial", "rank-prediction.ipynb")
OUT_NB = os.path.join(REPO, "colab", "prereg01_rank.ipynb")
OUT_DIFF = os.path.join(REPO, "results", "prereg01", "notebook_diff_rank.txt")

off = json.load(open(OFFICIAL, encoding="utf-8"))
OFF = {i: "".join(c["source"]) for i, c in enumerate(off["cells"])}
OFF_TYPE = {i: c["cell_type"] for i, c in enumerate(off["cells"])}
OFF_ID = {i: c.get("id") for i, c in enumerate(off["cells"])}

# ---------------------------------------------------------------------------
# Markdown notes appended to official markdown cells (the official text is kept as is)
# ---------------------------------------------------------------------------

MD_NOTE = {
    0: "\n\n> **本版**：路徑由最前面的設定格決定，不用手改。Windows 的限制見「與官方 notebook 的差異」。",
    3: "\n\n> **本版（① ②）**：改用 `load_rows` 只讀訓練玩家。內部驗證是另外 640 位玩家（`part=\"inner\"`），不是從同一批棋隨機切；兩邊各自打散後才寫分片。",
    5: "\n\n> **本版**：抽好的分片會備份到 Drive，重跑時直接複製回來。",
    19: "\n\n> **本版（④）**：只預測本機驗證題 `rank_val_questions.csv`，沒有 public／private，也不產生上傳檔。每題另存 10 維平均機率。",
    22: "\n\n> **本版**：分數不在這裡算。答案檔不上傳 Colab；把 `rank_val_probs.csv` 帶回本機，依事前登記第 3 節打分。",
}

# ---------------------------------------------------------------------------
# 和官方的全部差異：notebook 第 1 格的表、config.json 的 deviations_from_official 共用這一份。
# 每列：(標記, 表上「官方 cell」欄的文字, 涵蓋的官方格編號, 改了什麼)
# 官方格只要有改，編號就必須出現在某一列（產生時會檢查）。
# ---------------------------------------------------------------------------

_MD_CELLS = sorted(MD_NOTE)
DEV_ROWS = [
    ("①", "4", [4],
     "改用 `c10data.load_rows(..., part=\"fit\")` 讀訓練資料（5,767 人、720,307 盤），`part=\"inner\"` 讀內部驗證"
     "（640 人、73,556 盤），各自轉成 DataFrame。`load_rows` 會 assert 人數、盤數，並確認沒有任何一盤出現在本機驗證檔。"
     "結果存 `leak_check.json`（每次執行各記一筆）。SMOKE 時等 `load_rows` 全部通過，才用固定 seed 縮小資料。"),
    ("②", "4", [4],
     "取代 `df.iloc[:VAL_SIZE]` 的切法：`df_train`、`df_val` 各自 `sample(frac=1.0, random_state=42)` 打散之後，才寫分片。"),
    ("③", "2、8、9", [2, 8, 9],
     "`VAL_SIZE` 改成內部驗證實際寫入的盤數：每個抽特徵的 worker 寫完後，把 `rows`、`written`、`bad` 寫進分片目錄的"
     " `_count_w<N>.json`；`extract_split` 加總、核對後回傳；cell 9 設定 `VAL_SIZE` 並印出。cell 2 先設成 `None`。"),
    ("④", "20、21", [20, 21],
     "`predict()` 改讀打包檔的 `local/rank_val_questions.csv`，拿掉 public／private 迴圈（內層程式因此少一層縮排，"
     "進度訊息不印 split 名稱）。"
     "每題另存 10 維平均機率到 `rank_val_probs.csv`（完整 float；沒有可解析棋譜的題 `n_parsed=0`、機率欄留空）。"
     "官方的 argmax 結果改存成 `pred_official_argmax.csv`（官方是 `./submission_rank.csv`）。`predict()` 因此多了題目檔、"
     "機率檔路徑和 `limit` 三個參數；SMOKE 時只推論前 12 題。"),
    ("營運性", "新增", [],
     "最前面兩格 markdown：這份 notebook 的用途與操作說明，以及這張差異表。"),
    ("營運性", "新增", [],
     "新增的程式格：設定（Colab 或本機、Drive 路徑、SMOKE 縮小設定、結果紀錄工具）、準備（解壓打包檔並逐檔核對 SHA-256、"
     "確認沒有答案檔、解開訓練資料、把打包檔的 `tools/`、`tutorial/` 加進 import 路徑）、環境紀錄（`env.json`）、"
     "分片備份工具、設定紀錄與接續工具、結尾核對，以及把結果目錄裡權重以外的檔案打成 `rank_results.zip`（放在結果目錄旁，方便下載）。"),
    ("營運性", "新增（準備格）", [],
     "缺 `sgfmill` 或 `webdataset` 時才用 pip 安裝，版本固定為 `sgfmill==1.1.1`、`webdataset==1.0.2`（本機冒煙測試用的版本）。"
     "官方 notebook 只列出需要的套件，不安裝也不固定版本。Colab 內建的 torch、pandas、numpy、matplotlib 不動，實際版本記在 `env.json`。"),
    ("營運性", "新增（設定格）", [],
     "在 Linux 上明確指定 multiprocessing 用 `fork`。官方的 worker 函式定義在 notebook 內，依賴 fork；Python 3.14 起 Linux "
     "預設改成 forkserver。3.13 以前本來就是 fork，行為不變。實際的啟動方式記在 `config.json`。"),
    ("營運性", "metadata", [],
     "notebook metadata 加上 Colab 設定（GPU、A100、高 RAM），其餘沿用官方。官方格沿用官方的 cell id，新增的格用 `c10-rank-NN`；"
     "每格的 cell metadata 加 `c10` 對照欄（對應的官方格、是否改動）。官方原本就沒有存輸出。"),
    ("營運性", "markdown " + "、".join(str(i) for i in _MD_CELLS), _MD_CELLS,
     "官方的說明原文不動，在後面附加一段「**本版**」說明：路徑由設定格決定（0）、① ② 的讀法（3）、分片會備份到 Drive（5）、"
     "④ 只預測本機驗證題（19）、分數不在這裡算（22）。"),
    ("營運性", "全部", [],
     "工作目錄切到 `WORK`（Colab 是 `/content/c10`），官方的相對路徑 `./training`、`./rank-features` 照用。"
     "webdataset 會把 Windows 路徑的 `C:` 當成網址格式，本機冒煙測試只能用相對路徑。"),
    ("營運性", "全部", [],
     "改過的格在開頭或行內加了說明註解（標 ①–④ 或「本版」）。"),
    ("營運性", "全部", [],
     "紀錄檔都是每次執行各記一筆、不覆蓋前幾次：`timings.json`（各階段秒數）、`run_log.json`（每次執行與接續）、"
     "`env.json`、`leak_check.json`、`config.json`（頂層是固定說明與跨執行的累計狀態，每次執行的實際設定在 `runs`）。"
     "JSON 和 CSV 的 float 都寫完整精度。原生 Windows 的正式執行會直接報錯，因為官方的多進程依賴 Linux 的 fork。"),
    ("營運性", "2", [2],
     "`CKPT_DIR` 改到 Drive 的結果目錄 `ckpt/`，拿掉 `TEST_DIR`。`NPROC` 從寫死的 8 改成 `os.cpu_count()`"
     "（官方註解寫「設成 CPU 核數」）。加上 SMOKE 縮小設定（只在 SMOKE 生效）。最後的 `print` 多印 `NPROC`、batch、epoch 數。"),
    ("營運性", "4", [4],
     "不印官方的「全量 N 局」和全量的等級分布；改成逐一印 `fit`、`inner` 的人數與盤數，等級分布只印訓練集（`df_train`）。"
     "顯示前 3 列時不印棋譜欄，不讓棋譜內容進 notebook 輸出。"),
    ("營運性", "6", [6],
     "示範用的第一盤改取 `df_train`（本版沒有合併的 `df`）。"),
    ("營運性", "9", [9],
     "分片特徵抽完後備份到 Drive 的 `features_rank/`，全部核對後才寫完成標記。重跑時有完整且設定相符的備份，就複製回來、"
     "核對 SHA-256，不重抽；備份缺檔、大小或 SHA-256 不符時改成重抽。重抽前先清掉不完整的舊分片。"),
    ("營運性", "11、12", [11, 12],
     "`pick_workers` 另外記下讀出的筆數（寫進 `config.json`）。本機冒煙測試的 worker 候選數是 0（Windows 的 spawn 無法 "
     "pickle notebook 內定義的函式）。多 worker 對不上時照官方退回 0，並用 0 個 worker 再數一次；還是對不上 `VAL_SIZE` "
     "就停下來，因為這表示 ③ 沒有成立。"),
    ("營運性", "16", [16],
     "每個 epoch 結束，把 model、optimizer、epoch、最佳分數、history 存成 `ckpt/state.pt`。重跑時從最後完成的 epoch 接續；"
     "`state.pt` 讀不進來、或設定和上次不同，就停下來並說明處理方式。另外記最佳 epoch 的編號，結束時一併印出。"
     "history 多記 loss、步數、樣本數、秒數和第幾次執行。官方的 `epoch_N.pth`、`best_model.pth`、`history.csv` 照樣存在 `ckpt/`。"),
    ("營運性", "16", [16],
     "`sync_best()` 把最佳 epoch 的 `epoch_N.pth` 複製成 `best_model.pth`：接續時做一次（中斷時可能留下寫一半的 "
     "`best_model.pth`），正常訓練結束時也做一次。複製來源和官方存的 `best_model.pth` 是同一個 epoch 結束時的同一份權重。"
     "找不到那個 `epoch_N.pth` 就停下來並說明處理方式。"),
    ("營運性", "16", [16],
     "每次執行記下有沒有讀 `state.pt`、是否接續訓練、實際訓練了哪些 epoch（`run_log.json`、`config.json` 的 `runs`），"
     "頂層的 `resumed` 表示是否曾經接續訓練。"),
    ("營運性", "18", [18],
     "訓練曲線另存 `ckpt/history.png`。非 Colab 時用不開視窗的 Agg 後端，不 `show()`，畫完就 `close()`。"),
    ("營運性", "20", [20],
     "`predict()` 讀題目檔時 `question_id` 以字串讀入（`dtype={'question_id': str}`），代號原樣照抄，不會被轉成數字。"
     "每題 assert 非空的 `sgf_` 欄數等於 `num_games`，題目檔格式不對就停。兩者都不改變預測。"),
    ("營運性", "21", [21],
     "推論用的權重、題目檔、題數寫進 `config.json`，秒數寫進 `timings.json`；輸出路徑指向 Drive 的結果目錄。"),
]


def _md_cell(s):
    return s.replace("|", "\\|")


MD_TABLE = "\n".join("| %s | %s | %s |" % (tag, cells, _md_cell(text)) for tag, cells, _idx, text in DEV_ROWS)
DEVIATIONS = ["[%s] 官方 cell %s：%s" % (tag, cells, text) if _idx else "[%s] %s：%s" % (tag, cells, text)
              for tag, cells, _idx, text in DEV_ROWS]

# ---------------------------------------------------------------------------
# New markdown cells
# ---------------------------------------------------------------------------

MD_HEADER = r"""# C10 事前登記 01：競賽二（棋力預測）官方 baseline（Colab 版）

**用途**：照官方教學 `rank-prediction.ipynb` 訓練 `GoRankResNet`，對本機驗證題 `rank_val_questions.csv`（1,589 題）輸出每題的 10 維平均機率。打分在本機做，用決策規則 A（官方 argmax）和 B（期望得分）。

**對應事前登記**：`docs/PREREG_01_BASELINE.md`（rev 2）第 1、2 節。和官方 notebook 的差異只有 ①–④ 和下一格列的營運性差異。驗證規則見 `docs/VALIDATION.md`（split v1 rev 2）。

## 怎麼跑

1. 把 `C10_bundle_v1.zip` 放進 Google Drive 的 `MyDrive/C10/`。這個打包檔只有訓練資料、切分表、題目檔和候選庫，**沒有答案檔**。
2. 選單「執行階段 → 變更執行階段類型」：硬體加速器選 **A100 GPU**，並開啟**高 RAM**。
3. 選單「執行階段 → 全部執行」。第一個程式格會要求授權掛載 Google Drive，按允許。

## 結果寫到哪裡

`MyDrive/C10/results/prereg01/rank/`：

| 檔案 | 內容 |
|---|---|
| `rank_val_probs.csv` | 每題的 10 維平均機率（完整 float）。欄位：`question_id,n_games,n_parsed,p_D,…,p_6D` |
| `pred_official_argmax.csv` | 官方做法（取機率最大的等級）的預測，供對照 |
| `leak_check.json` | `load_rows` 的人數、盤數 assert 和外洩檢查結果（每次執行各一筆） |
| `config.json` | 頂層：固定說明、和官方的差異、跨執行的累計狀態（`resumed`、最佳 epoch）。`runs`：每次執行各一筆的實際設定（NPROC、DataLoader worker 數、分片來源、是否接續、訓練了哪些 epoch、推論設定） |
| `env.json`、`timings.json`、`run_log.json` | 每次執行各一筆：執行環境、各階段秒數、執行與接續的紀錄 |
| `ckpt/` | `state.pt`（接續用）、官方的 `epoch_N.pth`、`best_model.pth`、`history.csv`，以及訓練曲線 `history.png` |

分片特徵另外備份在 `MyDrive/C10/results/prereg01/features_rank/`。

## 中斷後怎麼接續

重新連線後，再按一次「全部執行」就好：

- 換到新的 VM 時，打包檔和訓練資料會重新解開；同一個 VM 已經解開過就跳過。
- `load_rows` 的人數、盤數和外洩檢查每次都重跑。
- 分片特徵：Drive 上有完整備份，就複製回來並核對 SHA-256，不重抽。備份缺檔、大小或 SHA-256 不符時自動重抽，抽出的樣本和原本相同。
- 訓練：每個 epoch 結束都會存到 Drive，重跑時從最後一個完成的 epoch 接續。每次執行有沒有接續、訓練了哪些 epoch，記在 `run_log.json` 和 `config.json` 的 `runs`；`config.json` 頂層的 `resumed` 表示是否曾經從 `state.pt` 接續訓練（已經訓練完、只重跑推論的那次不算）。
- **接續時沒有保存亂數狀態**：接續之後，dropout 的亂數和資料順序都和不中斷時不同。這不違反事前登記（官方本來就沒有固定資料順序），但接續過的執行無法逐位元重現。
- 推論：每次都整份重算。

會停下來、要手動處理的情況（錯誤訊息裡也有同樣的說明）：

- **`ckpt/state.pt` 讀不進來**（檔案壞掉或內容不完整）：在 Drive 把 `ckpt/` 整個資料夾改名保留（例如 `ckpt_broken_日期`），再按一次「全部執行」。會從頭訓練 5 個 epoch，分片備份不受影響。把這件事記進結果檔。不要只刪 `state.pt`，否則舊的 `epoch_N.pth`、`best_model.pth` 會和新的混在一起。
- **`ckpt/state.pt` 的設定和這次不同**：錯誤訊息會列出哪幾項不同。刻意改的話，照上一點把 `ckpt/` 改名後從頭訓練；不是刻意改的，先查出哪裡變了再跑。
- **找不到最佳 epoch 的 `epoch_N.pth`**（`best_model.pth` 要從它複製）：先到 Google Drive 的垃圾桶找，放回 `ckpt/` 再按「全部執行」；找不到就照第一點從頭訓練。不能拿其他 epoch 或殘留的 `best_model.pth` 代替（`docs/VALIDATION.md` 紀律 8）。

## 其他

- **本機冒煙測試**：`colab/run_local_smoke.py` 會設定 `C10_SMOKE=1`，只用極少量資料在 CPU 上跑一遍。結果寫到 `results/prereg01_smoke/`，和正式結果分開。
- **打分不在這裡做**：答案檔不上傳 Colab。跑完後最後一格會在 `MyDrive/C10/results/prereg01/` 產生 `rank_results.zip`（`rank/` 資料夾裡權重以外的檔案），下載這一個檔帶回本機，解開到 `results/prereg01/`，再依事前登記第 3 節打分。只有 `best_model.pth` 的推論結果可以打分（`docs/VALIDATION.md` 紀律 8）。
"""

MD_DIFF = r"""## 與官方 notebook 的差異

依 `docs/PREREG_01_BASELINE.md`（rev 2），和官方 `rank-prediction.ipynb` 的差異只能是 ①–④，加上下表標「營運性」的項目。下表的「官方 cell」是官方 notebook 的格子編號（從 0 起算）。**沒有列在表上的官方格，和官方逐字相同。** 逐格的 unified diff（程式格、markdown 格和 notebook metadata）在 `results/prereg01/notebook_diff_rank.txt`。

| 標記 | 官方 cell | 改了什麼 |
|---|---|---|
__TABLE__

**沒有動的**：

- 模型：`GoRankResNet`，預設 20 個 residual block，SMOKE 也一樣。
- 特徵：最後 250 個盤面 × 黑白兩個平面。
- 超參數：batch 256、AdamW lr 1e-3、5 個 epoch、seed 42、每個分片 1,000 筆、shuffle buffer 1,000。
- 選 checkpoint：內部驗證逐盤 accuracy 最高的 epoch（官方 `validate()`）。
- 推論：每盤各自 softmax，同題取平均；整題沒有可解析棋譜時填 `D`。

**資料順序不固定**：官方 notebook 沒有固定 Python 的 `random`，webdataset 的洗牌種子從它取，sample 洗牌則用程序代號加時間當種子。所以即使沒有中斷，兩次執行的資料順序也不一樣；接續的執行同樣如此，而且接續時也沒有保存 dropout 的亂數狀態（見第一格）。本版沿用官方做法，沒有改。
""".replace("__TABLE__", MD_TABLE)

# ---------------------------------------------------------------------------
# New code cells
# ---------------------------------------------------------------------------

SETUP = r"""# 設定：判斷在 Colab 還是本機冒煙測試，決定路徑和結果目錄。Colab 和本機只有這一格的寫法不同。
import csv
import datetime
import json
import multiprocessing
import os
import shutil
import sys
import time

RUN_START = time.time()


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


IN_COLAB = 'google.colab' in sys.modules
SMOKE = os.environ.get('C10_SMOKE') == '1'   # Colab 上沒有這個環境變數，所以永遠是 False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = '/content/drive/MyDrive/C10'
    WORK = '/content/c10'
else:
    DRIVE_DIR = os.environ['C10_DRIVE_DIR']
    WORK = os.environ['C10_WORK_DIR']
    if os.name == 'nt' and not SMOKE:
        # 官方的多進程依賴 Linux 的 fork；原生 Windows 只能跑冒煙測試（測試器會把進程換成執行緒）
        raise RuntimeError('原生 Windows 不支援正式執行，請在 Colab 上跑')

# 官方的多進程（抽特徵的 Process、推論的 Pool、DataLoader 的 worker）用的是 notebook 內定義的函式，只有 fork 能傳給子進程。
# Python 3.14 起 Linux 預設改成 forkserver，所以明確指定 fork；3.13 以前本來就是 fork，行為不變。
if sys.platform.startswith('linux'):
    multiprocessing.set_start_method('fork', force=True)
MP_START_METHOD = multiprocessing.get_start_method()

# 官方 notebook 用相對路徑（./training、./rank-features），這裡一律在 WORK 底下執行。
# webdataset 會把 Windows 路徑的「C:」當成網址格式，所以本機冒煙測試也必須用相對路徑。
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

# 結果目錄：正式和 SMOKE 分開，絕不混用
RESULT_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg01_smoke' if SMOKE else 'prereg01')
RESULT_DIR = os.path.join(RESULT_ROOT, 'rank')
RESULT_CKPT = os.path.join(RESULT_DIR, 'ckpt')
FEAT_BACKUP = os.path.join(RESULT_ROOT, 'features_rank')   # 分片特徵的 Drive 備份
os.makedirs(RESULT_CKPT, exist_ok=True)

# SMOKE 的縮小方式：只在本機冒煙測試生效，正式執行時完全不用
SMOKE_CFG = {
    'fit_games': 2000,     # load_rows 全部通過之後，才用固定 seed 抽
    'inner_games': 400,
    'sample_seed': 42,
    'nproc': 2,            # 抽特徵和推論的進程數
    'per_shard': 300,      # 讓每個 worker 寫出好幾個分片，測到換檔
    'batch_size': 32,
    'num_epochs': 1,
    'questions': 12,       # 推論只做前 12 題
    'loader_workers': 0,   # 本機 Windows 的 spawn 無法 pickle notebook 內定義的函式
}

# 和官方 notebook 的全部差異（與第 1 格的表同一份）
DEVIATIONS = __DEVIATIONS__


def write_json(path, obj):
    # 先寫暫存檔再改名，中斷時不會留下寫一半的 JSON；float 用 repr，不四捨五入
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_runs(path, **init):
    # 每次執行各記一筆的紀錄檔（{'runs': [...], ...}），不覆蓋前幾次；舊格式（整份只有一次執行）當成一筆保留
    obj = read_json(path)
    if obj is None:
        return {'runs': [], **init}
    if not isinstance(obj, dict) or not isinstance(obj.get('runs'), list):
        return {'runs': [{'legacy': obj}], **init}
    return obj


# 每次「全部執行」算一次 run；接續次數和時間記在 run_log.json
RUN_LOG_PATH = os.path.join(RESULT_DIR, 'run_log.json')
run_log = read_runs(RUN_LOG_PATH, resume_count=0)
RUN = {'run': len(run_log['runs']) + 1, 'start': now_iso(), 'end': None, 'smoke': SMOKE,
       'in_colab': IN_COLAB, 'features': None, 'start_epoch': None, 'loaded_state': False,
       'resumed_training': False, 'epochs_trained': []}
run_log['runs'].append(RUN)


def save_run_log():
    write_json(RUN_LOG_PATH, run_log)


TIMINGS_PATH = os.path.join(RESULT_DIR, 'timings.json')
timings = read_runs(TIMINGS_PATH)
TIMING = {'run': RUN['run'], 'start': RUN['start'], 'seconds': {}}
timings['runs'].append(TIMING)


def mark_time(stage, seconds):
    TIMING['seconds'][stage] = seconds
    write_json(TIMINGS_PATH, timings)


# config.json：頂層放固定的說明和跨執行的累計狀態；每次執行的實際設定各記一筆在 runs，不覆蓋前幾次
CONFIG_PATH = os.path.join(RESULT_DIR, 'config.json')
_old_config = read_runs(CONFIG_PATH)
CONFIG = {k: v for k, v in _old_config.items() if k != 'runs'}
CONFIG.update({
    'layout': '頂層：固定說明與跨執行的累計狀態；runs：每次執行各一筆的實際設定',
    'smoke': SMOKE, 'smoke_cfg': SMOKE_CFG if SMOKE else None,
    'prereg': 'docs/PREREG_01_BASELINE.md rev 2（競賽二）',
    'official_notebook': 'third_party/aicup2026_tutorial/rank-prediction.ipynb',
    'deviations_from_official': DEVIATIONS,
    # resumed：是否曾有任何一次執行從 ckpt/state.pt 接續訓練。和 run_log.json 的 resume_count 同一個來源；
    # 已經訓練完、只重跑推論的那次不算
    'resumed': run_log['resume_count'] > 0, 'resume_count': run_log['resume_count'],
})
CONFIG['runs'] = _old_config['runs']
CONFIG_RUN = {'run': RUN['run'], 'start': RUN['start'], 'end': None, 'in_colab': IN_COLAB,
              'mp_start_method': MP_START_METHOD}
CONFIG['runs'].append(CONFIG_RUN)


def save_config(**kw):
    # 這次執行的實際設定：只更新 runs 裡這次的那一筆
    CONFIG_RUN.update(kw)
    write_json(CONFIG_PATH, CONFIG)


def save_config_summary(**kw):
    # 跨執行的累計狀態（是否曾接續、已完成幾個 epoch、最佳 epoch）：放頂層
    CONFIG.update(kw)
    write_json(CONFIG_PATH, CONFIG)


save_run_log()
save_config()
print('IN_COLAB', IN_COLAB, '| SMOKE', SMOKE, '| run', RUN['run'], '| multiprocessing', MP_START_METHOD)
print('WORK', WORK)
print('結果目錄', RESULT_DIR)
""".replace("__DEVIATIONS__", json.dumps(DEVIATIONS, ensure_ascii=False, indent=4))

PREP = r"""# 準備：缺套件才安裝、把打包檔複製到 WORK 解壓並逐檔核對 SHA-256、解開訓練資料
import hashlib
import importlib.util
import subprocess
import tarfile
import zipfile

t0 = time.time()
if IN_COLAB:
    # 版本固定成本機冒煙測試用的版本
    need = [pkg for mod, pkg in (('sgfmill', 'sgfmill==1.1.1'), ('webdataset', 'webdataset==1.0.2'))
            if importlib.util.find_spec(mod) is None]
    if need:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *need])

BUNDLE_NAME = 'C10_bundle_v1'
BUNDLE_ZIP_SRC = os.path.join(DRIVE_DIR, BUNDLE_NAME + '.zip')
BUNDLE_ZIP = os.path.join(WORK, BUNDLE_NAME + '.zip')
BUNDLE = os.path.join(WORK, BUNDLE_NAME)


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            d.update(chunk)
    return d.hexdigest()


def bundle_mismatches(manifest):
    return [k for k, v in manifest['files'].items()
            if not os.path.exists(os.path.join(BUNDLE, k)) or sha256_file(os.path.join(BUNDLE, k)) != v]


with zipfile.ZipFile(BUNDLE_ZIP_SRC) as z:
    MANIFEST = json.loads(z.read(BUNDLE_NAME + '/bundle_manifest.json'))

if read_json(os.path.join(BUNDLE, 'bundle_manifest.json')) == MANIFEST and not bundle_mismatches(MANIFEST):
    print('打包檔已解壓，SHA-256 全部相符，跳過解壓')
else:
    shutil.rmtree(BUNDLE, ignore_errors=True)
    shutil.copyfile(BUNDLE_ZIP_SRC, BUNDLE_ZIP)
    with zipfile.ZipFile(BUNDLE_ZIP) as z:
        z.extractall(WORK)
    bad = bundle_mismatches(MANIFEST)
    if bad:
        raise SystemExit('打包檔 SHA-256 不符：%s' % bad)
    print('打包檔解壓完成，%d 個檔案的 SHA-256 全部相符' % len(MANIFEST['files']))

# 紀律 7：Colab 上不能有答案檔
_all_files = [os.path.join(r, n) for r, _d, ns in os.walk(BUNDLE) for n in ns]
assert not any('answers' in p for p in _all_files), '打包檔裡有答案檔，停止'

# 訓練資料：tar.xz 解開後是 ./training/train_<等級>.csv（另有 macOS 的 ._train_*.csv，load_rows 不會讀）
ARCHIVE = os.path.join(BUNDLE, 'training', 'aicup2026_go_training.tar.xz')
ARCHIVE_SHA = MANIFEST['files']['training/aicup2026_go_training.tar.xz']
TRAIN_FILES = ['train_%s.csv' % lv for lv in ('D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D')]
_marker = os.path.join('training', '.c10_extracted.json')
if (read_json(_marker, {}).get('archive_sha256') == ARCHIVE_SHA
        and all(os.path.exists(os.path.join('training', n)) for n in TRAIN_FILES)):
    print('訓練資料已解開，跳過')
else:
    shutil.rmtree('training', ignore_errors=True)
    with tarfile.open(ARCHIVE, 'r:xz') as tf:
        if hasattr(tarfile, 'data_filter'):
            tf.extractall(WORK, filter='data')
        else:
            tf.extractall(WORK)
    missing = [n for n in TRAIN_FILES if not os.path.exists(os.path.join('training', n))]
    if missing:
        raise SystemExit('解開後缺檔：%s' % missing)
    write_json(_marker, {'archive_sha256': ARCHIVE_SHA,
                         'bytes': {n: os.path.getsize(os.path.join('training', n)) for n in TRAIN_FILES}})
    print('訓練資料解開完成')

# 打包檔裡的 c10data.py、官方 utils.py／network.py 加進 import 路徑
for _sub in ('tools', 'tutorial'):
    _p = os.path.join(BUNDLE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
LOCAL_DIR = os.path.join(BUNDLE, 'local')                      # 只有題目檔和候選庫
SPLIT_CSV = os.path.join(BUNDLE, 'split_v1', 'players_split.csv')
RANK_QUESTIONS = os.path.join(LOCAL_DIR, 'rank_val_questions.csv')
mark_time('prepare_bundle', time.time() - t0)
print('本機題目檔', sorted(os.listdir(LOCAL_DIR)))
"""

ENV = r"""# 環境紀錄：每次執行各記一筆到結果目錄的 env.json（接續時可能換到不同的機器）
import platform
from importlib import metadata


def _pkg_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _ram_bytes():
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    if os.path.exists('/proc/meminfo'):
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) * 1024
    return None


_gpu = None
if torch.cuda.is_available():
    # 用 nvidia-smi 查，不在這裡初始化 CUDA（官方要到建模型時才初始化，之後才 fork 抽特徵的進程）
    try:
        _smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                               '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().splitlines()
    except FileNotFoundError:
        _smi = None
    _gpu = {'nvidia_smi': _smi, 'device_count': torch.cuda.device_count()}
ENV_INFO = {
    'run': RUN['run'], 'start_time': RUN['start'], 'in_colab': IN_COLAB, 'smoke': SMOKE,
    'python': sys.version, 'platform': platform.platform(),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    'gpu': _gpu, 'cpu_count': os.cpu_count(), 'ram_bytes': _ram_bytes(),
    'multiprocessing_start_method': MP_START_METHOD,
    'packages': {n: _pkg_version(n) for n in ('pandas', 'numpy', 'webdataset', 'sgfmill', 'matplotlib')},
    'bundle_zip_bytes': os.path.getsize(BUNDLE_ZIP_SRC), 'bundle_manifest': MANIFEST,
}
ENV_PATH = os.path.join(RESULT_DIR, 'env.json')
_env_log = read_runs(ENV_PATH)
_env_log['runs'].append(ENV_INFO)
write_json(ENV_PATH, _env_log)
print(json.dumps({k: ENV_INFO[k] for k in ('torch', 'cuda', 'gpu', 'cpu_count', 'ram_bytes', 'packages')},
                 ensure_ascii=False))
"""

FEAT_TOOLS = r"""# 分片備份工具：抽好的分片備份到 Drive；重跑時有完整且設定相符的備份就複製回來，不重抽
FEAT_MARKER = os.path.join(FEAT_BACKUP, '_COMPLETE.json')


def _feature_files(root):
    # {'train/xxx.tar.gz': bytes, ...}，包含各 worker 回報筆數的 _count_w*.json
    out = {}
    for tag in ('train', 'val'):
        d = os.path.join(root, tag)
        if os.path.isdir(d):
            for n in sorted(os.listdir(d)):
                out['%s/%s' % (tag, n)] = os.path.getsize(os.path.join(d, n))
    return out


def backup_features(fingerprint, counts):
    # 全部複製並核對大小之後，才寫完成標記
    shutil.rmtree(FEAT_BACKUP, ignore_errors=True)
    files = _feature_files(FEAT_DIR)
    for rel in files:
        dst = os.path.join(FEAT_BACKUP, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(FEAT_DIR, rel), dst)
    if _feature_files(FEAT_BACKUP) != files:
        raise SystemExit('分片備份的檔案大小和本機不符')
    write_json(FEAT_MARKER, {
        'fingerprint': fingerprint, 'counts': counts, 'nproc_used': NPROC, 'time': now_iso(),
        'files': {rel: {'bytes': b, 'sha256': sha256_file(os.path.join(FEAT_DIR, rel))}
                  for rel, b in files.items()}})
    print('分片已備份到 %s（%d 個檔，%.2f GB）' % (FEAT_BACKUP, len(files), sum(files.values()) / 1e9))


def restore_features(fingerprint):
    # 回傳 {'counts':…, 'mode':…}；沒有可用的備份（沒有、設定不同、缺檔、壞檔）時回傳 None，呼叫端會重抽
    try:
        m = read_json(FEAT_MARKER)
        if m is not None:
            want = {rel: v['bytes'] for rel, v in m['files'].items()}
            counts, same = m['counts'], m['fingerprint'] == fingerprint
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print('Drive 上的分片完成標記讀不進來（%s: %s），改成重抽' % (type(e).__name__, e))
        return None
    if m is None:
        print('Drive 上沒有完整的分片備份')
        return None
    if not same:
        print('Drive 上的分片備份和這次的設定不同，不用')
        return None
    mode = 'local_reused'
    if _feature_files(FEAT_DIR) != want:
        if _feature_files(FEAT_BACKUP) != want:
            print('Drive 上的分片備份缺檔或大小不符，改成重抽')
            return None
        shutil.rmtree(FEAT_DIR, ignore_errors=True)
        try:
            for rel in want:
                dst = os.path.join(FEAT_DIR, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(FEAT_BACKUP, rel), dst)
        except OSError as e:
            print('從 Drive 複製分片失敗（%s: %s），改成重抽' % (type(e).__name__, e))
            shutil.rmtree(FEAT_DIR, ignore_errors=True)
            return None
        mode = 'restored_from_drive'
    try:
        bad = [rel for rel, v in m['files'].items() if sha256_file(os.path.join(FEAT_DIR, rel)) != v['sha256']]
    except (OSError, KeyError, TypeError) as e:
        bad = ['（核對失敗：%s: %s）' % (type(e).__name__, e)]
    if bad:
        print('分片的 SHA-256 和備份紀錄不符（%d 個檔），改成重抽' % len(bad))
        shutil.rmtree(FEAT_DIR, ignore_errors=True)
        return None
    print('分片：%s（%d 個檔）' % (mode, len(want)))
    return {'counts': counts, 'mode': mode}
"""

RECORD = r"""# 設定紀錄與接續工具：這次執行的實際設定寫進 config.json 的 runs；state.pt 存接續需要的一切
STATE_PATH = os.path.join(CKPT_DIR, 'state.pt')
# state.pt 裡的設定和這次不同就不接續（例如分片重抽過、超參數不同）
STATE_CONFIG = {'SEED': SEED, 'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE,
                'NUM_EPOCHS': NUM_EPOCHS, 'VAL_SIZE': VAL_SIZE, 'features': FEAT_FINGERPRINT}

save_config(
    hyperparameters={'SEED': SEED, 'BATCH_SIZE': BATCH_SIZE, 'LEARNING_RATE': LEARNING_RATE,
                     'NUM_EPOCHS': NUM_EPOCHS, 'NUM_CLASSES': NUM_CLASSES, 'PER_SHARD': PER_SHARD,
                     'shuffle_buffer': 1000, 'shardshuffle_train': 1},
    optimizer={'class': type(optimizer).__name__, **optimizer.defaults},
    criterion=type(criterion).__name__,
    model={'class': type(model).__name__, 'parameters': sum(p.numel() for p in model.parameters()),
           'network_py_sha256': MANIFEST['files']['tutorial/network.py']},
    feature={'parser': type(_parser).__name__, 'history_length': _parser.history_length,
             'utils_py_sha256': MANIFEST['files']['tutorial/utils.py']},
    device=str(DEVICE), device_name=torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None,
    nproc=NPROC, cpu_count=os.cpu_count(),
    loader_workers=workers, worker_checks=WORKER_CHECKS, val_size=VAL_SIZE,
    data={'fit_games_used': len(df_train), 'inner_games_used': len(df_val),
          'train_written': train_counts, 'val_written': val_counts},
    features_mode=RUN['features'],
    checkpoint_rule='best_model.pth＝內部驗證逐盤 accuracy 最高的 epoch（官方 validate()；同分取較早的 epoch）',
)

# 接續出問題時怎麼處理（第一格「中斷後怎麼接續」也寫了同樣的說明）
STATE_RESTART = ('在 Drive 把 %s 整個資料夾改名保留（例如 ckpt_broken_日期），再按一次「全部執行」，'
                 '會從頭訓練 %d 個 epoch（分片備份不受影響），並把這件事記進結果檔。'
                 '不要只刪 state.pt，否則舊的 epoch_N.pth、best_model.pth 會和新的混在一起。' % (CKPT_DIR, NUM_EPOCHS))
STATE_HELP = '處理方式：' + STATE_RESTART


def save_state(epoch_done):
    # 這個 epoch 的官方檔都存好之後才存 state.pt；存好才算這個 epoch 完成
    tmp = STATE_PATH + '.tmp'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch_done, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch,
                'history': history, 'config': STATE_CONFIG}, tmp)
    os.replace(tmp, STATE_PATH)


def load_state():
    # 讀 state.pt 並載入 model、optimizer；讀不進來、內容不完整或設定不同，就停下來說明怎麼處理
    try:
        state = torch.load(STATE_PATH, map_location=DEVICE, weights_only=False)   # 自己存的檔，含 history
        missing = [k for k in ('model', 'optimizer', 'epoch', 'best_val_acc', 'best_epoch', 'history', 'config')
                   if k not in state]
    except Exception as e:
        raise SystemExit('ckpt/state.pt 讀不進來（%s: %s）。%s' % (type(e).__name__, e, STATE_HELP)) from e
    if missing:
        raise SystemExit('ckpt/state.pt 缺少 %s。%s' % (missing, STATE_HELP))
    if state['config'] != STATE_CONFIG:
        diff = sorted(k for k in set(state['config']) | set(STATE_CONFIG)
                      if state['config'].get(k) != STATE_CONFIG.get(k))
        raise SystemExit('ckpt/state.pt 的設定和這次不同，不能接續。不同的項目：%s。'
                         '處理方式：刻意改的話，%s不是刻意改的，先查出哪裡變了再跑。' % (diff, STATE_RESTART))
    try:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
    except Exception as e:
        raise SystemExit('ckpt/state.pt 的權重載不進模型或 optimizer（%s: %s）。%s'
                         % (type(e).__name__, e, STATE_HELP)) from e
    return state


def sync_best():
    # best_model.pth 對齊 history 裡的最佳 epoch：接續時（中斷可能留下寫一半的 best_model.pth）和正常訓練結束時各做一次
    if best_epoch is None:
        return
    src = os.path.join(CKPT_DIR, 'epoch_%d.pth' % best_epoch)
    if not os.path.exists(src):
        raise SystemExit(
            '%s 不見了（最佳 epoch 是 %d），best_model.pth 無法對齊。處理方式：先到 Google Drive 的垃圾桶找 '
            'epoch_%d.pth，放回 %s 再按一次「全部執行」；找不到的話，從頭訓練：%s'
            '不能拿其他 epoch 或殘留的 best_model.pth 代替（docs/VALIDATION.md 紀律 8：只有指定的 checkpoint 可以打分）。'
            % (src, best_epoch, best_epoch, CKPT_DIR, STATE_RESTART))
    shutil.copyfile(src, os.path.join(CKPT_DIR, 'best_model.pth'))
"""

FINAL = r"""# 結尾：核對輸出，補完 config／timings／run_log，列出結果檔
_probs = pd.read_csv(PROBS_PATH, dtype={'question_id': str})
_pred = pd.read_csv(PRED_PATH, dtype={'question_id': str})
assert list(_probs.columns) == PROB_COLUMNS, list(_probs.columns)
assert len(_probs) == len(_pred) == N_PREDICTED, (len(_probs), len(_pred), N_PREDICTED)
assert (_probs['question_id'] == _pred['question_id']).all()
_ok = _probs['n_parsed'] > 0
_sums = _probs.loc[_ok, PROB_COLUMNS[3:]].sum(axis=1)
OUTPUTS = {
    'questions': int(len(_probs)),
    'questions_without_parsable_games': int((~_ok).sum()),   # 這些題 A、B 都填 D
    'games_total': int(_probs['n_games'].sum()), 'games_parsed': int(_probs['n_parsed'].sum()),
    'prob_row_sum_min': float(_sums.min()) if len(_sums) else None,
    'prob_row_sum_max': float(_sums.max()) if len(_sums) else None,
    'pred_distribution': {k: int(v) for k, v in _pred['pred_rank'].value_counts().items()},
    'files': {n: os.path.getsize(os.path.join(RESULT_DIR, n)) for n in sorted(os.listdir(RESULT_DIR))
              if os.path.isfile(os.path.join(RESULT_DIR, n))},
}
RUN['end'] = now_iso()
save_config(outputs=OUTPUTS, end=RUN['end'])
save_run_log()
mark_time('total_this_run', time.time() - RUN_START)
print(json.dumps(OUTPUTS, ensure_ascii=False, indent=1))
print('ckpt/', sorted(os.listdir(CKPT_DIR)))

# 結果打包（營運性）：結果目錄裡除了權重以外的檔案打成一個 zip，放在結果目錄旁邊，方便整包下載
import zipfile
RESULTS_ZIP = os.path.join(os.path.dirname(RESULT_DIR), os.path.basename(RESULT_DIR) + '_results.zip')
with zipfile.ZipFile(RESULTS_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as _z:
    for _root, _dirs, _files in os.walk(RESULT_DIR):
        for _f in sorted(_files):
            _p = os.path.join(_root, _f)
            if _f.endswith(('.pth', '.pt', '.tmp')) or os.path.getsize(_p) > 100 * 1024 * 1024:
                continue   # 權重與暫存檔不帶
            _z.write(_p, os.path.relpath(_p, os.path.dirname(RESULT_DIR)))
print('完成。請下載這個檔回傳給 Claude（打分在本機做）：%s（%.1f MB）'
      % (RESULTS_ZIP, os.path.getsize(RESULTS_ZIP) / 1e6))
"""

# ---------------------------------------------------------------------------
# Modified official code cells (keyed by official cell index)
# ---------------------------------------------------------------------------

MOD = {}

MOD[2] = r"""# 路徑：官方的相對路徑照用（工作目錄已切到 WORK）；checkpoint 改存到 Drive 的結果目錄，中斷後才能接續
TRAIN_DIR = './training'
FEAT_DIR = './rank-features'
CKPT_DIR = RESULT_CKPT

# 等級由弱到強。順序很重要：評分時的「相鄰一級」就是這個順序上的相鄰。
LEVELS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
RANK_TO_ID = {lv: i for i, lv in enumerate(LEVELS)}
ID_TO_RANK = {i: lv for lv, i in RANK_TO_ID.items()}

NPROC = os.cpu_count() # 特徵抽取的進程數，設成 CPU 核數
PER_SHARD = 1000       # 每個 webdataset 分片的樣本數
SEED = 42
VAL_SIZE = None        # ③ 第 1 節寫完分片後，改成內部驗證實際寫入的盤數

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
NUM_CLASSES = len(LEVELS)

if SMOKE:
    # 只在本機冒煙測試生效：縮小資料量與步數，邏輯不變（縮小方式見第一個設定格的 SMOKE_CFG）
    NPROC = SMOKE_CFG['nproc']
    PER_SHARD = SMOKE_CFG['per_shard']
    BATCH_SIZE = SMOKE_CFG['batch_size']
    NUM_EPOCHS = SMOKE_CFG['num_epochs']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs(CKPT_DIR, exist_ok=True)
print('device', DEVICE, '| NPROC', NPROC, '| batch', BATCH_SIZE, '| epochs', NUM_EPOCHS)
RANK_TO_ID
"""

MOD[4] = r"""# ① 只讀訓練玩家：經 c10data.load_rows 讀。它會 assert 人數、盤數，並確認沒有任何一盤出現在本機驗證檔，
#    不符就直接 raise。訓練用 part="fit"，內部驗證用 part="inner"（另外 640 位玩家，不是從同一批棋切）。
from c10data import EXPECTED, load_rows

t0 = time.time()
leak = {'run': RUN['run'], 'time': now_iso(), 'c10data_sha256': MANIFEST['files']['tools/c10data.py'],
        'local_files_checked': sorted(os.listdir(LOCAL_DIR)), 'parts': {}}
frames = {}
for part in ('fit', 'inner'):
    t1 = time.time()
    rows = load_rows(TRAIN_DIR, SPLIT_CSV, part=part, local_dir=LOCAL_DIR)
    frames[part] = pd.DataFrame(rows, columns=['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
    del rows
    got = {'players': int(frames[part]['player_id'].nunique()), 'games': int(len(frames[part]))}
    assert got == EXPECTED[part], (part, got, EXPECTED[part])
    leak['parts'][part] = {
        'players': got['players'], 'games': got['games'], 'expected': EXPECTED[part],
        'count_assert': '通過', 'leak_check': '外洩檢查通過（0 盤出現在本機驗證檔）',
        'games_by_rank': {lv: int((frames[part]['rank'] == lv).sum()) for lv in LEVELS},
        'seconds': time.time() - t1}
    print('%s：%s 人、%s 盤，人數盤數 assert 通過，外洩檢查通過'
          % (part, format(got['players'], ','), format(got['games'], ',')), flush=True)
assert not set(frames['fit']['player_id']) & set(frames['inner']['player_id'])
leak['fit_inner_player_overlap'] = 0
df_train, df_val = frames.pop('fit'), frames.pop('inner')

if SMOKE:
    # 只在本機冒煙測試：load_rows 全部通過之後，才用固定 seed 縮小
    df_train = df_train.sample(n=SMOKE_CFG['fit_games'], random_state=SMOKE_CFG['sample_seed'])
    df_val = df_val.sample(n=SMOKE_CFG['inner_games'], random_state=SMOKE_CFG['sample_seed'])
    leak['smoke_subsample'] = {'fit_games': len(df_train), 'inner_games': len(df_val),
                               'seed': SMOKE_CFG['sample_seed']}

# ② 讀進來時是照等級、再照黑白排好的，官方的分片又照列順序寫；各自打散後才寫分片，否則每個 batch 只有一個等級
df_train = df_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df_val = df_val.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
# 每次執行各記一筆，不覆蓋前幾次
_leak_path = os.path.join(RESULT_DIR, 'leak_check.json')
_leak_log = read_runs(_leak_path)
_leak_log['runs'].append(leak)
write_json(_leak_path, _leak_log)
mark_time('load_rows', time.time() - t0)

print('等級分布（訓練）', df_train['rank'].value_counts().to_dict())
print('切分：訓練 %s / 驗證 %s' % (format(len(df_train), ','), format(len(df_val), ',')))
df_train[['player_id', 'game_id', 'rank', 'color']].head(3)
"""

MOD[6] = OFF[6].replace("_demo = df.iloc[0]['sgf_content']", "_demo = df_train.iloc[0]['sgf_content']")
assert MOD[6] != OFF[6]

MOD[8] = OFF[8]
_old_tail = """    print('  [w%d] 寫入 %d 筆（解析失敗 %d）%.1f 分'
          % (wid, written, bad, (time.time() - t0) / 60), flush=True)
"""
_new_tail = """    print('  [w%d] 寫入 %d 筆（解析失敗 %d）%.1f 分'
          % (wid, written, bad, (time.time() - t0) / 60), flush=True)
    # ③ 把實際寫入的筆數交回主程式（worker 是另一個進程，用小檔案回報）
    with open('%s/_count_w%d.json' % (os.path.dirname(pattern), wid), 'w') as f:
        json.dump({'rows': len(df_part), 'written': written, 'bad': bad}, f)
"""
assert MOD[8].count(_old_tail) == 1
MOD[8] = MOD[8].replace(_old_tail, _new_tail)
_old_end = """    if avg < 10 * 1024:
        raise SystemExit('%s: 分片平均僅 %.0f bytes，判定為空分片' % (tag, avg))"""
_new_end = """    if avg < 10 * 1024:
        raise SystemExit('%s: 分片平均僅 %.0f bytes，判定為空分片' % (tag, avg))

    # ③ 加總各 worker 回報的實際寫入筆數
    counts = [read_json('%s/_count_w%d.json' % (outdir, i)) for i in range(NPROC)]
    got = {k: sum(c[k] for c in counts) for k in ('rows', 'written', 'bad')}
    assert got['rows'] == len(df_part) and got['written'] + got['bad'] == got['rows'], got
    print('%s: 實際寫入 %d 筆（解析失敗 %d）' % (tag, got['written'], got['bad']), flush=True)
    return got
"""
assert OFF[8].rstrip().endswith(_old_end)
MOD[8] = MOD[8].rstrip()[: -len(_old_end)] + _new_end

MOD[9] = r"""# 一次性作業。分片產出後重跑訓練可以直接從下一節開始。
# 本版：Drive 上有完整的分片備份就直接複製回來，不重抽；沒有（或缺檔、壞檔）的話先清掉不完整的舊分片再抽，抽完備份到 Drive
t0 = time.time()
FEAT_FINGERPRINT = {'bundle_files': MANIFEST['files'], 'smoke': SMOKE,
                    'smoke_cfg': SMOKE_CFG if SMOKE else None, 'seed': SEED, 'per_shard': PER_SHARD,
                    'fit_games': len(df_train), 'inner_games': len(df_val)}
restored = restore_features(FEAT_FINGERPRINT)
if restored is None:
    shutil.rmtree(FEAT_DIR, ignore_errors=True)
    val_counts = extract_split(df_val, 'val')
    train_counts = extract_split(df_train, 'train')
    mark_time('features_extract', time.time() - t0)
    t1 = time.time()
    backup_features(FEAT_FINGERPRINT, {'val': val_counts, 'train': train_counts})
    mark_time('features_backup', time.time() - t1)
    RUN['features'] = 'extracted'
else:
    val_counts, train_counts = restored['counts']['val'], restored['counts']['train']
    mark_time('features_restore', time.time() - t0)
    RUN['features'] = restored['mode']
save_run_log()

VAL_SIZE = val_counts['written']   # ③ 內部驗證實際寫入的盤數
print('VAL_SIZE=%d（內部驗證實際寫入）；訓練實際寫入 %d 筆' % (VAL_SIZE, train_counts['written']))
print('完成，總耗時 %.1f 分' % ((time.time() - t0) / 60))
"""

MOD[11] = OFF[11]
_old = """def pick_workers(val_shards, candidate=4):
    \"\"\"數一遍驗證集：筆數對得上才用多 worker，否則退回單執行緒。\"\"\"
    total = sum(len(y) for _x, y, _n in make_loader(val_shards, candidate, False))
"""
_new = """WORKER_CHECKS = []   # 每次數驗證集的結果，寫進 config.json


def pick_workers(val_shards, candidate=4):
    \"\"\"數一遍驗證集：筆數對得上才用多 worker，否則退回單執行緒。\"\"\"
    total = sum(len(y) for _x, y, _n in make_loader(val_shards, candidate, False))
    WORKER_CHECKS.append({'candidate': candidate, 'counted': total, 'expected': VAL_SIZE,
                          'matched': total == VAL_SIZE})
"""
assert MOD[11].count(_old) == 1
MOD[11] = MOD[11].replace(_old, _new)

MOD[12] = r"""train_shards = sorted(glob.glob(f'{FEAT_DIR}/train/*.tar.gz'))
val_shards = sorted(glob.glob(f'{FEAT_DIR}/val/*.tar.gz'))
print('分片：train %d / val %d' % (len(train_shards), len(val_shards)))

# 本機冒煙測試（Windows 的 spawn）一律 0 個 worker；Colab 照官方先試 4 個
WORKER_CANDIDATE = SMOKE_CFG['loader_workers'] if (SMOKE and not IN_COLAB) else 4
t0 = time.time()
workers = pick_workers(val_shards, WORKER_CANDIDATE)
if workers == 0 and WORKER_CANDIDATE > 0:
    pick_workers(val_shards, 0)   # 照官方退回 0 之後，用 0 個 worker 再數一次
if not WORKER_CHECKS[-1]['matched']:
    raise SystemExit('驗證分片讀出 %d 筆，VAL_SIZE 是 %d：③ 沒有成立'
                     % (WORKER_CHECKS[-1]['counted'], VAL_SIZE))
mark_time('pick_workers', time.time() - t0)
train_loader = make_loader(train_shards, workers, True)
val_loader = make_loader(val_shards, workers, False)

_x, _y, _n = next(iter(val_loader))
print('一個 batch:', _x.shape, _y[:8].tolist())
"""

MOD[16] = r"""# 訓練期間不能中斷 kernel。
# 本版：每個 epoch 結束都把接續需要的狀態存到 Drive 的 ckpt/state.pt；中斷後重新全部執行，會從最後完成的 epoch 接續
history = []
model.train()
best_val_acc = 0.0
best_epoch = None
start_epoch = 0

if os.path.exists(STATE_PATH):
    state = load_state()   # 讀不進來、內容不完整、設定不同，都會停下來並說明處理方式
    start_epoch, best_val_acc = state['epoch'], state['best_val_acc']
    best_epoch, history = state['best_epoch'], state['history']
    del state
    sync_best()
    RUN['loaded_state'] = True
    if start_epoch < NUM_EPOCHS:
        # 真的接著訓練才算接續；已經訓練完、只重跑推論的那次不算
        run_log['resume_count'] += 1
        RUN.update(resumed_training=True, resume_time=now_iso())
        print('從 epoch %d 接續（目前最佳 val acc %.2f%%，epoch %s）' % (start_epoch, best_val_acc, best_epoch))
    else:
        print('%d 個 epoch 都已訓練完（最佳 val acc %.2f%%，epoch %s），這次不訓練，只重跑後面的推論'
              % (start_epoch, best_val_acc, best_epoch))
RUN['start_epoch'] = start_epoch
save_run_log()
save_config(start_epoch=start_epoch, loaded_state=RUN['loaded_state'],
            resumed_from_state=RUN['resumed_training'], epochs_trained=[], trained_any_epoch=False)
save_config_summary(resumed=run_log['resume_count'] > 0, resume_count=run_log['resume_count'],
                    epochs_completed=start_epoch)

for epoch in range(start_epoch, NUM_EPOCHS):
    t0 = time.time()
    total_loss, correct, total, steps = 0.0, 0, 0, 0

    for inputs, labels, _ in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        steps += 1
        if steps % 500 == 0:
            print('  ep%d step %d  loss %.4f  acc %.2f%%  %.1f 分'
                  % (epoch + 1, steps, total_loss / steps,
                     100 * correct / total, (time.time() - t0) / 60), flush=True)

    train_acc = 100 * correct / total if total else 0.0
    val_loss, val_acc = validate(model, val_loader, criterion)
    history.append({'epoch': epoch + 1, 'train_acc': train_acc, 'val_acc': val_acc,
                    'train_loss': total_loss / steps if steps else 0.0, 'val_loss': val_loss,
                    'steps': steps, 'samples': total, 'seconds': time.time() - t0, 'run': RUN['run']})
    print('Epoch %d/%d  train acc %.2f%%  |  val loss %.4f acc %.2f%%  |  %.1f 分'
          % (epoch + 1, NUM_EPOCHS, train_acc, val_loss, val_acc,
             (time.time() - t0) / 60), flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), f'{CKPT_DIR}/best_model.pth')
        print('  新的最佳 val acc %.2f%% —— 已存 best_model.pth' % best_val_acc)
    torch.save(model.state_dict(), f'{CKPT_DIR}/epoch_%d.pth' % (epoch + 1))
    save_state(epoch + 1)   # 最後才存；存好才算這個 epoch 完成
    RUN['epochs_trained'].append(epoch + 1)
    save_run_log()
    save_config(epochs_trained=list(RUN['epochs_trained']), trained_any_epoch=True)
    save_config_summary(epochs_completed=epoch + 1, best_epoch=best_epoch, best_val_acc=best_val_acc)
    mark_time('epoch_%d' % (epoch + 1), time.time() - t0)

pd.DataFrame(history).to_csv(f'{CKPT_DIR}/history.csv', index=False)
sync_best()   # 正常結束時也對齊一次（best_model.pth 和 epoch_N.pth 是同一個 epoch 的同一份權重）
save_config_summary(epochs_completed=len(history), best_epoch=best_epoch, best_val_acc=best_val_acc,
                    trained_in_runs=sorted({h['run'] for h in history}))
print('訓練完成，最佳 val acc %.2f%%（epoch %s）' % (best_val_acc, best_epoch))
"""

MOD[18] = r"""import matplotlib
if not IN_COLAB:
    matplotlib.use('Agg')   # 本機冒煙測試不開視窗
import matplotlib.pyplot as plt

hist = pd.read_csv(f'{CKPT_DIR}/history.csv')
best = hist.loc[hist['val_acc'].idxmax()]

plt.figure(figsize=(9, 4))
plt.plot(hist['epoch'], hist['train_acc'], marker='o', ms=3, label='train acc')
plt.plot(hist['epoch'], hist['val_acc'], marker='o', ms=3, label='val acc')
plt.axvline(best['epoch'], ls='--', c='gray', lw=1)
plt.annotate('best %.2f%%' % best['val_acc'],
             xy=(best['epoch'], best['val_acc']),
             xytext=(best['epoch'] + 1.5, best['val_acc'] - 4),
             arrowprops=dict(arrowstyle='->', color='gray'))
plt.xlabel('epoch'); plt.ylabel('accuracy (%)')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f'{CKPT_DIR}/history.png', dpi=120)
if IN_COLAB:
    plt.show()
else:
    plt.close()
"""

MOD[20] = r"""GPU_BATCH = 64
_infer_parser = SGFParseRankPrediction()
# ④ 每題另存 10 維平均機率；欄位固定
PROB_COLUMNS = ['question_id', 'n_games', 'n_parsed'] + ['p_%s' % lv for lv in LEVELS]


def sgf_to_features(sgf_content):
    @@DQ@@給 multiprocessing.Pool 用的頂層函式。@@DQ@@
    feat, err = _infer_parser.features_from_sgf_content(sgf_content)
    return None if err is not None else feat


def predict(weights_path, questions_path, out_path, probs_path, limit=None):
    net = GoRankResNet(num_classes=NUM_CLASSES)
    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
    net.eval().to(DEVICE)

    rows, prob_rows = [], []
    with Pool(NPROC) as pool:
        # ④ 讀本機驗證題（格式和官方測試檔相同），沒有 public／private 迴圈；question_id 以字串讀入，原樣照抄
        test = pd.read_csv(questions_path, dtype={'question_id': str})
        if limit is not None:
            test = test.head(limit)
        sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
        t0 = time.time()

        for idx, r in test.iterrows():
            # num_games 之後的 sgf_ 欄位是空字串，要濾掉
            sgfs = [r[c] for c in sgf_cols
                    if pd.notna(r[c]) and str(r[c]).strip()]
            # 本版：非空的棋譜數必須等於 num_games，題目檔格式不對就停
            assert len(sgfs) == int(r['num_games']), (r['question_id'], len(sgfs), r['num_games'])
            feats = [f for f in pool.map(sgf_to_features, sgfs) if f is not None]
            if not feats:
                rows.append([r['question_id'], LEVELS[0]])
                prob_rows.append([r['question_id'], len(sgfs), 0] + [''] * NUM_CLASSES)
                continue

            x = torch.from_numpy(np.stack(feats)).float()
            probs = []
            with torch.no_grad():
                for i in range(0, len(x), GPU_BATCH):
                    out = net(x[i:i + GPU_BATCH].to(DEVICE))
                    probs.append(out.softmax(1).cpu().numpy())

            avg = np.concatenate(probs).mean(0)      # 同題多局的機率平均
            rows.append([r['question_id'], LEVELS[int(avg.argmax())]])
            # 完整 float（Python float 的 repr），不四捨五入；n_games、n_parsed 是 Python int
            prob_rows.append([r['question_id'], len(sgfs), len(feats)] + [repr(float(p)) for p in avg])
            if (idx + 1) % 100 == 0:
                print('  %d/%d  %.0f 秒' % (idx + 1, len(test), time.time() - t0), flush=True)
        print('  完成 %d 題' % len(test), flush=True)

    sub = pd.DataFrame(rows, columns=['question_id', 'pred_rank'])
    sub.to_csv(out_path, index=False)
    with open(probs_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(PROB_COLUMNS)
        w.writerows(prob_rows)
    print('寫出 %s（%d 列）' % (out_path, len(sub)))
    print('寫出 %s（%d 列）' % (probs_path, len(prob_rows)))
    print('預測分布:', sub['pred_rank'].value_counts().to_dict())
    return sub
""".replace("@@DQ@@", '"' * 3)

MOD[21] = r"""# 紀律 8：只用事前登記指定的 best_model.pth 推論
PRED_PATH = os.path.join(RESULT_DIR, 'pred_official_argmax.csv')
PROBS_PATH = os.path.join(RESULT_DIR, 'rank_val_probs.csv')
t0 = time.time()
submission = predict(f'{CKPT_DIR}/best_model.pth', RANK_QUESTIONS, PRED_PATH, PROBS_PATH,
                     limit=SMOKE_CFG['questions'] if SMOKE else None)
N_PREDICTED = len(submission)
mark_time('inference', time.time() - t0)
save_config(inference={'weights': 'ckpt/best_model.pth', 'questions_file': os.path.basename(RANK_QUESTIONS),
                       'questions_predicted': N_PREDICTED, 'GPU_BATCH': GPU_BATCH, 'NPROC': NPROC,
                       'aggregation': '每盤各自 softmax，同題取平均（float32，與官方相同）',
                       'no_parsable_games': 'pred_rank 填 D；rank_val_probs.csv 的 n_parsed=0、機率欄留空'})
submission.head()
"""

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

plan = [("md_new", MD_HEADER, "header"), ("md_new", MD_DIFF, "deviations"),
        ("code_new", SETUP, "setup"), ("code_new", PREP, "prepare")]
for i in range(len(off["cells"])):
    if i == 2:
        plan.append(("code_new", ENV, "environment"))
    if i == 9:
        plan.append(("code_new", FEAT_TOOLS, "feature_backup_tools"))
    if i == 16:
        plan.append(("code_new", RECORD, "config_and_resume_tools"))
    if OFF_TYPE[i] == "markdown":
        plan.append(("md_off", OFF[i].rstrip("\n") + MD_NOTE[i] if i in MD_NOTE else OFF[i], i))
    else:
        plan.append(("code_off", MOD.get(i, OFF[i]), i))
plan.append(("code_new", FINAL, "final_check"))

nb = nbformat.v4.new_notebook()
# 官方的 metadata 照抄，再加上 Colab 的設定
nb.metadata = json.loads(json.dumps(off["metadata"]))
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"gpuType": "A100", "machine_shape": "hm", "provenance": []}

for k, (kind, src, tag) in enumerate(plan):
    if kind.endswith("off"):
        # 官方格：沒改的話逐字照抄（不動結尾換行）
        if src != OFF[tag]:
            src = src.rstrip("\n")
    else:
        src = src.rstrip("\n")
    if kind.startswith("md"):
        cell = nbformat.v4.new_markdown_cell(src)
    else:
        # 每個程式格都必須是純 Python（冒煙測試器也是整格 compile，失敗就拒跑）
        compile(src, "<cell %d>" % k, "exec")
        cell = nbformat.v4.new_code_cell(src)
    if kind.endswith("off"):
        cell["id"] = OFF_ID[tag]
        cell["metadata"]["c10"] = {"official_cell": tag, "changed": src != OFF[tag]}
    else:
        cell["id"] = "c10-rank-%02d" % k
        cell["metadata"]["c10"] = {"new": tag}
    nb.cells.append(cell)
nbformat.validate(nb)

# 完整性檢查：官方格全部都在；有改的官方格一定出現在差異表；差異表點名的官方格一定有改
_paired = {c["metadata"]["c10"]["official_cell"]: c for c in nb.cells if "official_cell" in c["metadata"]["c10"]}
assert sorted(_paired) == sorted(OFF), sorted(set(OFF) - set(_paired))
_changed = {i for i, c in _paired.items() if c["metadata"]["c10"]["changed"]}
_named = {i for _t, _c, idx, _x in DEV_ROWS for i in idx}
assert _changed == _named, ("changed but not in DEV_ROWS", sorted(_changed - _named),
                            "in DEV_ROWS but unchanged", sorted(_named - _changed))
for i, c in _paired.items():
    if i not in _changed:
        assert c["source"] == OFF[i] and c["cell_type"] == OFF_TYPE[i], i

os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
with open(OUT_NB, "w", encoding="utf-8", newline="\n") as f:
    nbformat.write(nb, f)

# 寫出的檔案讀回來，未改動的官方格仍然逐字相同
_back = json.load(open(OUT_NB, encoding="utf-8"))
for c in _back["cells"]:
    i = c["metadata"]["c10"].get("official_cell")
    if i is not None and i not in _changed:
        assert "".join(c["source"]) == OFF[i] and c["id"] == OFF_ID[i], i


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# Per-cell unified diff of every cell, plus the notebook metadata
out = []
out.append("# prereg01 rank：官方 rank-prediction.ipynb 與 colab/prereg01_rank.ipynb 的逐格 unified diff（程式格、markdown 格、notebook metadata）")
out.append("# 官方：third_party/aicup2026_tutorial/rank-prediction.ipynb  sha256=%s" % sha(OFFICIAL))
out.append("# 本版：colab/prereg01_rank.ipynb  sha256=%s" % sha(OUT_NB))
out.append("# 產生方式：colab/build_rank_nb.py；difflib.unified_diff，依 cell metadata c10.official_cell 配對官方的格；"
           "新增的格對空白比。cell 編號從 0 起算。")
out.append("# 每一處差異的標記（①–④ 或營運性）見 notebook 第 1 格「與官方 notebook 的差異」；沒列在表上的官方格和官方逐字相同。")
summary = []
for k, cell in enumerate(nb.cells):
    meta = cell["metadata"]["c10"]
    new_src = cell["source"]
    kind = "程式格" if cell["cell_type"] == "code" else "markdown"
    if "official_cell" in meta:
        i = meta["official_cell"]
        old_src = OFF[i]
        a = "official/rank-prediction.ipynb cell %d" % i
        status = "改動" if meta["changed"] else "完全相同"
    else:
        old_src = ""
        a = "/dev/null（新增：%s）" % meta["new"]
        status = "新增"
    b = "colab/prereg01_rank.ipynb cell %d" % k
    summary.append("#   本版 cell %2d（%s）← %s：%s" % (k, kind, a, status))
    diff = list(difflib.unified_diff(old_src.splitlines(), new_src.splitlines(),
                                     fromfile=a, tofile=b, lineterm=""))
    out.append("")
    out.append("=" * 100)
    out.append("本版 cell %d（%s）↔ %s（%s）" % (k, kind, a, status))
    out.append("=" * 100)
    out.extend(diff if diff else ["（完全相同）"])

_meta_off = json.dumps(off["metadata"], ensure_ascii=False, indent=1, sort_keys=True).splitlines()
_meta_new = json.dumps(nb.metadata, ensure_ascii=False, indent=1, sort_keys=True).splitlines()
out.append("")
out.append("=" * 100)
out.append("notebook metadata")
out.append("=" * 100)
out.extend(list(difflib.unified_diff(_meta_off, _meta_new, fromfile="official metadata",
                                     tofile="本版 metadata", lineterm="")) or ["（完全相同）"])
out[5:5] = ["# 對照表："] + summary
os.makedirs(os.path.dirname(OUT_DIFF), exist_ok=True)
with open(OUT_DIFF, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print("wrote", OUT_NB, sha(OUT_NB))
print("wrote", OUT_DIFF, sha(OUT_DIFF))
print("cells", len(nb.cells), "| official cells changed", sorted(_changed), "| deviation rows", len(DEV_ROWS))
