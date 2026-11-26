"""Build colab/prereg01_player.ipynb from the official player-identification notebook.

Modified official code cells are built by exact string replacement on the official
source (each target must occur exactly once), so unchanged lines stay byte-identical.
Official cells kept as they are ("identical") are copied verbatim, layout included.
"""
import hashlib
import json
import os

import nbformat

ROOT = r"D:\AI-Portfolio\competitions\aicup2026-go"
OFFICIAL = os.path.join(ROOT, "third_party", "aicup2026_tutorial", "player-identification.ipynb")
OUT = os.path.join(ROOT, "colab", "prereg01_player.ipynb")

with open(OFFICIAL, encoding="utf-8") as f:
    off_nb = json.load(f)
OFF = ["".join(c["source"]) for c in off_nb["cells"]]
OFF_TYPE = [c["cell_type"] for c in off_nb["cells"]]


def file_sha256(rel):
    with open(os.path.join(ROOT, rel), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# 打包檔裡這三個檔必須和 repo 裡的原檔相同；notebook 會在 Colab 上逐檔比對
SOURCE_SHA256 = {
    "tools/c10data.py": file_sha256("tools/c10data.py"),
    "tutorial/utils.py": file_sha256("third_party/aicup2026_tutorial/utils.py"),
    "tutorial/network.py": file_sha256("third_party/aicup2026_tutorial/network.py"),
}

SOURCE_SHA256_SRC = ("{  # 產生本 notebook 時 repo 裡原檔的 SHA-256\n"
                     + "".join("    %r: %r,\n" % kv for kv in SOURCE_SHA256.items()) + "}")


def rep(src, old, new):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"replacement target occurs {n} times:\n{old}")
    return src.replace(old, new)


cells = []


def md(text, official=None, change="new"):
    meta = {"c10": {"official_cell": official, "change": change}}
    cells.append(nbformat.v4.new_markdown_cell(text, metadata=meta))


def code(src, official=None, change="new", tags=()):
    tags = list(tags) or (["營運性"] if official is None else [])
    meta = {"c10": {"official_cell": official, "change": change, "tags": tags}}
    cells.append(nbformat.v4.new_code_cell(src, metadata=meta))


def off_md(i, note=None):
    # 官方 markdown 格原文不動；有「本版註」時加在最前面
    assert OFF_TYPE[i] == "markdown", i
    text = OFF[i]
    if note:
        text = note + "\n\n" + text
    md(text, official=i, change="modified" if note else "identical")


def off_code(i):
    assert OFF_TYPE[i] == "code", i
    code(OFF[i], official=i, change="identical")


# ---------------------------------------------------------------------------
# 差異清單：markdown 和 config.json 共用同一份
DEVIATIONS = [
    ("⑤", "官方 cell 4：訓練資料改用 `c10data.load_rows(TRAIN_DIR, SPLIT_CSV, part=\"train\", local_dir=LOCAL_DIR)` 讀，只含 split v1 的訓練玩家（6,407 人）。load_rows 會 assert 人數與盤數，並確認沒有任何一盤出現在本機驗證檔。之後照官方做 `MIN_GAMES` 過濾。"),
    ("⑤", "新增外洩檢查格：寫完 `player_index.json` 後，讀 `players_split.csv` assert 索引裡沒有任何 `split=='val'` 的玩家（也沒有不在切分表裡的玩家），結果存 `leak_check.json`（含 load_rows 的 assert 結果）。"),
    ("⑥", "官方 cell 20–21：候選庫改讀打包檔的 `local/player_val_candidates.csv`，題目改讀 `local/player_val_questions.csv`，拿掉 public／private 迴圈。候選清單照玩家代號排序（官方的 `names = sorted(centroids)`，保留）。用 `ckpt/best_model.pth` 推論，輸出 `player_val_top5.csv`（question_id,top_1…top_5）到 Drive 結果目錄（官方寫到 `./submission_player.csv`）。"),
    ("營運性", "官方 cell 4 列順序：load_rows 照等級 D…6D 回傳，讀進來後把列順序排回官方 `sorted(glob('train_*.csv'))` 的檔案順序（1D…6D、A、B、C、D；同一等級內保持檔案裡的列順序）。這只讓特徵分片 `df.iloc[i::NPROC]` 用和官方同一個切法，而且只有 NPROC 相同時才是同一個切法；讀進來的玩家集合本來就和官方不同（只有訓練玩家），所以各分片的內容和 `player_index.json` 的玩家順序都不會和官方相同。"),
    ("營運性", "程式來源與 SHA-256：準備格把打包檔的 `tools/`（c10data.py）與 `tutorial/`（官方 utils.py、network.py 原檔）放到 sys.path 最前面。環境紀錄格確認 utils、network 是從打包檔 `tutorial/` 載入，官方 cell 4 確認 c10data 是從打包檔 `tools/` 載入；三個檔的 SHA-256 都要和打包檔 manifest、以及產生本 notebook 時 repo 裡的原檔（`third_party/aicup2026_tutorial/utils.py`、`network.py`，`tools/c10data.py`）相同，不符就停止。三個檔的 SHA-256 記在 env.json 的 `source_sha256`（c10data 的另記在 leak_check.json）。"),
    ("營運性", "新增 code 格：設定格（判斷 Colab、掛載 Drive、路徑、SMOKE 開關、fork）、準備格（缺 sgfmill／webdataset 才用 pip 安裝固定版本 1.1.1／1.0.2、解開打包檔並逐檔核對 SHA-256、確認沒有答案檔、解開訓練資料、設定 sys.path）、環境紀錄格（env.json，並開始這次執行的 run_log.json、timings.json）、設定紀錄格（config.json）、外洩檢查格（見 ⑤）、接續偵測格、收尾格。"),
    ("營運性", "設定格：Linux 上強制 multiprocessing 用 fork（`set_start_method('fork', force=True)`）。官方 notebook 依賴 fork（worker 函式定義在 notebook 內），Python 3.14 起 Linux 的預設改成 forkserver。不是 Colab、不是 SMOKE、又不能 fork 時直接停止。"),
    ("營運性", "官方 cell 2 路徑：訓練資料在 `/content/c10/training`，特徵與索引放執行階段本機磁碟 `/content/c10/`，checkpoint 放 Drive 結果目錄的 `ckpt/`；官方的 `TEST_DIR` 拿掉，改成本機題目檔與候選庫的路徑（見 ⑥）。"),
    ("營運性", "官方 cell 2 其他：`NPROC = os.cpu_count()`（官方寫死 8，註解是「設成 CPU 核數」）；新增 DataLoader worker 數 `NUM_WORKERS`；SMOKE 時改用第一個設定格的縮小值；`print('device', DEVICE)` 改成多印 NPROC 與 worker 數。"),
    ("營運性", "官方 cell 12：DataLoader 的 worker 數改用 `NUM_WORKERS`：正式執行 = NPROC（同官方）；SMOKE 且非 Colab 時為 0（Windows 的 spawn 無法 pickle notebook 內定義的類別）。`persistent_workers` 在 0 worker 時設 False（PyTorch 規定）。"),
    ("營運性", "接續（官方 cell 12、14、16 與新增的接續偵測格）：每個 epoch 結束另存 `ckpt/train_state.pt`（模型、optimizer、已完成的 epoch 數、最低 loss、best 的 epoch 與 `best_model.pth` 的 SHA-256、每個 epoch 權重檔的 SHA-256、歷程、每個 epoch 用的 NPROC 與 worker 數）。重跑時從最後完成的 epoch 接續，並把 `TripletDataset` 的 epoch 計數設成已完成的 epoch 數，讓抽樣種子和不中斷時相同。設定或索引內容和上次不同時拒絕接續。`run_log.json` 的 resumed／n_resumes 只計真的從 train_state 接著訓練的執行（已完成的 epoch 數 > 0，而且還有 epoch 要跑）；訓練已經跑完、只重跑推論的不算。"),
    ("營運性", "接續與抽樣：triplet 的抽樣序列取決於 DataLoader 的 worker 數（每個 worker 的種子與分到的量）和 `player_index.json` 的玩家順序（特徵分片數 = NPROC）。換到 CPU 核數不同的執行階段再接續，接續後各 epoch 抽到的 triplet 會和不中斷時不同。每次執行的 NPROC 與 worker 數記在 run_log.json，每個 epoch 用的記在 train_state.pt 與 run_log.json；接續時和上次不同會印出警告並記在 run_log.json。"),
    ("營運性", "官方 cell 16 存檔順序與 best 完整性：存檔順序改成先 `epoch_N.pth`（torch.save，同官方），需要時再 `best_model.pth`，最後 `train_state.pt`（寫完才算這個 epoch 完成）。`best_model.pth` 改成複製剛存好的 `epoch_N.pth`（官方是另外 torch.save 一次；權重相同，但 torch.save 會把檔名寫進檔案內部，兩個檔的位元組不同），所以 best 的 SHA-256 就是 `epoch_{best}.pth` 的 SHA-256。接續時和訓練結束時都核對 Drive 上的 `best_model.pth`：SHA-256 和 train_state 記的不同（或檔案不在）就從 `epoch_{best}.pth` 還原；`epoch_{best}.pth` 也不在或 SHA-256 不符時停止，並印出處理方式。每次核對與還原都記在 run_log.json。"),
    ("營運性", "特徵（約 15 GB、約 100 萬個小檔）放執行階段本機磁碟，不備份到 Drive。同一個執行階段內重跑時，完成標記（資料列指紋、NPROC、索引 SHA-256）相符就跳過抽取；否則清掉殘檔重抽。"),
    ("營運性", "耗時與執行紀錄：官方 cell 4、9、16、21 多記耗時（timings.json），cell 9、16 多記 run_log.json；官方 notebook 沒有這些紀錄。"),
    ("營運性", "官方 cell 18：loss 曲線另存 `loss_curve.png`；SMOKE 時 matplotlib 用 Agg。"),
    ("營運性", "官方 cell 20 訊息：拿掉 public／private 迴圈後，`print('=== %s ===' % split)` 拿掉，完成訊息 `'  %s 完成 %d 題' % (split, len(test))` 改成 `'  完成 %d 題' % len(test)`（沒有 split 名稱可印）。查詢沒有任何可解析棋譜的題，照官方填照代號排序的前 5 位候選，另外計數（predict_summary.json 的 n_fallback_questions）。"),
    ("營運性", "官方 cell 20–21：另存 `player_val_embeddings.npz`（候選重心、題目平均向量，完整 float32）與 `predict_summary.json`（權重與輸入輸出檔的 SHA-256、題數、盤數、fallback 題數），不影響 Top-5。"),
    ("營運性", "收尾格：記下這次執行完成（run_log.json、timings.json）並列出結果檔；把結果目錄裡權重以外的檔案打成 `player_results.zip`（放在結果目錄旁，方便下載）；Colab 上呼叫 `drive.flush_and_unmount()`，把 Drive 的寫入全部送出再卸載，之後要再跑必須重新「全部執行」（設定格會重新掛載）。"),
    ("營運性", "SMOKE 模式（只在本機冒煙測試，正式執行完全不生效）：load_rows 之後用 SEED 固定抽 40 位 ≥2 盤的玩家、每人最多 20 盤；NPROC 2、batch 8、1 epoch × 20 步；推論只用題目檔前 6 題（每題前 2 盤）、候選庫只用照代號排序的前 12 位（每人前 3 盤）；結果寫到 `results/prereg01_smoke/player/`。"),
    ("營運性", "markdown：官方 cell 0、19、22 前面各加一段「本版註」，官方原文不動；官方其餘 markdown 格逐字相同。另外新增 4 個 markdown 格（標題說明、這份差異清單、外洩檢查、接續偵測）。官方格之間的相對順序不變，但官方 cell 0 排在本版的標題、差異清單、設定格、準備格之後。"),
    ("營運性", "notebook metadata：沿用官方的 kernelspec、language_info，加上 Colab 設定（accelerator GPU、A100、高 RAM）與 `c10` 欄（這份差異清單）。官方格沿用官方的 cell id，新增的格用 `c10-player-NN`；每格的 cell metadata 加 `c10` 對照欄（對應的官方格、是否改動、分類）。官方原本就沒有存輸出。"),
    ("營運性", "結果檔 config.json、env.json、timings.json、run_log.json、leak_check.json、predict_summary.json、history.csv 的 float 都寫完整值（json 與 pandas 預設用 repr），不四捨五入。"),
]
DEV_STRINGS = ["[%s] %s" % (t, s) for t, s in DEVIATIONS]

# ---------------------------------------------------------------------------
md(r"""# C10 競賽一（玩家辨識）：官方 baseline 在本機驗證集的版本（事前登記 01）

**用途**：照官方教學 `player-identification.ipynb` 訓練 `GoPlayerResNet`（triplet loss），但只用 split v1 的訓練玩家，並對本機驗證題（400 題）產出 Top-5。**這裡不打分**：答案檔不上傳 Colab；跑完後最後一格會在 `MyDrive/C10/results/prereg01/` 產生 `player_results.zip`（`player/` 資料夾裡權重以外的檔案，含 `player_val_top5.csv`），下載這一個檔帶回本機，再用 `tools/score.py` 打分並記進打分紀錄帳。

**對應事前登記**：`docs/PREREG_01_BASELINE.md`（rev 2）第 4 節，登記差異 ⑤、⑥；驗證規則見 `docs/VALIDATION.md`（split v1 rev 2）。和官方 notebook 的全部差異列在下一格；逐格 diff 在 `results/prereg01/notebook_diff_player.txt`。

## 怎麼跑

1. 把 `C10_bundle_v1.zip` 放到 Google Drive 的 `MyDrive/C10/`。
2. 「執行階段 → 變更執行階段類型」：硬體加速器選 **A100 GPU**，並打開 **高 RAM**。
3. 「執行階段 → 全部執行」。第一格會跳出 Google Drive 授權，照畫面允許。
4. 跑到最後一格印出「完成」就結束。耗時會記在結果目錄的 `timings.json`（官方 notebook 沒寫執行時間）。

## 結果寫到哪裡

`MyDrive/C10/results/prereg01/player/`

| 檔案 | 內容 |
|---|---|
| `player_val_top5.csv` | 400 題的 Top-5（欄位 question_id,top_1…top_5），**要下載回本機打分的檔** |
| `ckpt/best_model.pth` | 訓練 loss 最低的 epoch（`epoch_N.pth` 的複本），**唯一可以打分的 checkpoint**（官方做法） |
| `ckpt/epoch_N.pth`、`ckpt/history.csv` | 官方照存的每個 epoch 權重與 loss 歷程（epoch 權重一律不打分） |
| `ckpt/train_state.pt` | 接續用的完整訓練狀態（含 best 的 epoch 與 SHA-256） |
| `leak_check.json` | load_rows 的 assert 結果、索引裡驗證玩家數（必須是 0） |
| `player_val_embeddings.npz`、`predict_summary.json` | 候選重心與題目平均向量、推論摘要 |
| `config.json`、`env.json`、`timings.json`、`run_log.json`、`loss_curve.png` | 設定、環境、耗時、每次執行與接續的紀錄 |

## 中斷後怎麼接續

重新「全部執行」即可，不用改任何設定。

- **訓練**：每個 epoch 結束依序存 `ckpt/epoch_N.pth`、（loss 創新低時）`ckpt/best_model.pth`、最後存 `ckpt/train_state.pt`（模型、optimizer、已完成的 epoch 數、最低 loss、best 的 epoch 與 SHA-256、歷程）。`train_state.pt` 寫完才算這個 epoch 完成；重跑時從最後完成的 epoch 接著訓練，沒跑完的那個 epoch 整個重跑。只有真的接著訓練才算一次接續（訓練已跑完、只重跑推論的不算），次數與時間記在 `run_log.json`。
- **best_model.pth 的完整性**：接續時和訓練結束時，都用 `train_state.pt` 記的 SHA-256 核對 Drive 上的 `best_model.pth`；不符（例如寫到一半被中斷）就從 `epoch_{best}.pth` 還原。`epoch_{best}.pth` 也不在時會停下來並印出處理方式。
- **換執行階段的注意**：triplet 的抽樣序列取決於 DataLoader 的 worker 數和 `player_index.json` 的玩家順序，兩者都由 CPU 核數（`NPROC = os.cpu_count()`）決定。**換了 CPU 核數不同的執行階段再接續，接續之後抽到的 triplet 會和不中斷時不同**（抽樣方法、玩家、超參數都一樣，只是抽到的組合不同）。每次執行的 NPROC 與 worker 數記在 `run_log.json`，每個 epoch 用的記在 `train_state.pt` 與 `run_log.json`；接續時和上次不同會印出警告。
- **特徵**：玩家辨識的特徵約 15 GB、約 100 萬個小檔，放在執行階段本機磁碟 `/content/c10/player-features/`，**不備份到 Drive**（小檔太多，Drive 讀寫太慢）。執行階段重置後本機磁碟會清空，重跑時會重新抽特徵；同一個執行階段內重跑，偵測到完成標記就跳過。
- **打包檔與訓練資料**：同一個執行階段內已解開且 SHA-256 相符就跳過。

## 本機冒煙測試

`colab/run_local_smoke.py` 會設 `C10_SMOKE=1`，只用極少量資料在 CPU 上走完全部流程，結果寫到 `results/prereg01_smoke/player/`，和正式結果分開。SMOKE 的縮小方式寫在第一個設定格，正式執行時完全不生效。""")

md("## 與官方 notebook 的差異\n\n"
   "格號指官方 notebook 的格號（從 0 起算，含 markdown 格）。登記的差異只有 ⑤、⑥（`docs/PREREG_01_BASELINE.md` 第 4 節），其餘都是營運性差異：只改執行環境、存檔與接續方式，"
   "**模型、特徵（17 層）、超參數（batch 100、lr 1e-4、margin 0.5、5 epoch × 15,000 步、seed 42）、checkpoint 選法都不動**。\n\n"
   + "\n".join("%d. **%s** %s" % (i, t, s) for i, (t, s) in enumerate(DEVIATIONS, 1)))

# --------------------------------------------------------------------------- 設定格
code(r"""# 設定格（營運性）：判斷執行環境、掛載 Google Drive、決定路徑與 SMOKE 的縮小方式
import multiprocessing
import os
import sys

IN_COLAB = 'google.colab' in sys.modules
# SMOKE 只給本機冒煙測試用（colab/run_local_smoke.py 會設 C10_SMOKE=1）；Colab 上永遠是 False
SMOKE = (not IN_COLAB) and os.environ.get('C10_SMOKE') == '1'

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = '/content/drive/MyDrive/C10'
    WORK = '/content/c10'
else:
    DRIVE_DIR = os.environ['C10_DRIVE_DIR']
    WORK = os.environ['C10_WORK_DIR']

# 正式結果與 SMOKE 結果分開放，絕不混用
RESULT_DIR = os.path.join(DRIVE_DIR, 'results', 'prereg01_smoke' if SMOKE else 'prereg01', 'player')
BUNDLE_ZIP = os.path.join(DRIVE_DIR, 'C10_bundle_v1.zip')
BUNDLE = os.path.join(WORK, 'C10_bundle_v1')          # 打包檔解開後的目錄
os.makedirs(WORK, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 官方 notebook 依賴 Linux 的 fork（worker 函式定義在 notebook 內）。
# Python 3.14 起 Linux 的預設改成 forkserver，所以在 Linux 上明確指定 fork。
if sys.platform.startswith('linux') and multiprocessing.get_start_method(allow_none=True) != 'fork':
    multiprocessing.set_start_method('fork', force=True)
if not IN_COLAB and not SMOKE and multiprocessing.get_start_method() != 'fork':
    raise SystemExit('正式執行需要 fork 啟動方式（Colab 或 Linux）；Windows／macOS 只能跑 SMOKE')

# SMOKE 的縮小方式：只縮資料量與步數，邏輯和正式一致；正式執行時下面這些值完全不會用到
SMOKE_NPROC = 2               # 特徵抽取與推論的平行數
SMOKE_N_PLAYERS = 40          # load_rows 之後，用 SEED 固定抽 40 位 ≥2 盤的玩家
SMOKE_GAMES_PER_PLAYER = 20   # 每人最多 20 盤（照資料列順序取前 20 盤）
SMOKE_BATCH_SIZE = 8
SMOKE_NUM_EPOCHS = 1
SMOKE_STEPS_PER_EPOCH = 20
SMOKE_LOG_INTERVAL = 5
SMOKE_N_QUESTIONS = 6         # 只用題目檔的前 6 題
SMOKE_Q_GAMES = 2             # 每題只用前 2 盤（CPU 推論太慢）
SMOKE_N_CAND_PLAYERS = 12     # 候選庫只用照玩家代號排序後的前 12 位
SMOKE_CAND_GAMES = 3          # 每位候選只用前 3 盤（CPU 推論太慢）

print('IN_COLAB', IN_COLAB, '| SMOKE', SMOKE, '| start method', multiprocessing.get_start_method())
print('DRIVE_DIR ', DRIVE_DIR)
print('WORK      ', WORK)
print('RESULT_DIR', RESULT_DIR)
""")

# --------------------------------------------------------------------------- 準備格
code(r"""# 準備格（營運性）：缺套件才安裝、解開打包檔並逐檔核對 SHA-256、解開訓練資料、設定 sys.path
import hashlib
import importlib
import importlib.util
import json
import shutil
import subprocess
import tarfile
import time
import zipfile

_t_prepare = time.time()

if IN_COLAB:
    # 版本固定成本機冒煙測試用的版本（和競賽二的 notebook 相同）
    _need = [pkg for mod, pkg in (('sgfmill', 'sgfmill==1.1.1'), ('webdataset', 'webdataset==1.0.2'))
             if importlib.util.find_spec(mod) is None]
    if _need:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + _need)
        importlib.invalidate_caches()


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            d.update(chunk)
    return d.hexdigest()


def bundle_ok():
    # 解開的目錄完整：manifest 和 Drive 上 zip 裡的一致，而且每個檔案的 SHA-256 都相符
    mpath = os.path.join(BUNDLE, 'bundle_manifest.json')
    if not os.path.exists(mpath):
        return False
    with zipfile.ZipFile(BUNDLE_ZIP) as z:
        zip_manifest = json.loads(z.read('C10_bundle_v1/bundle_manifest.json'))
    with open(mpath, encoding='utf-8') as f:
        if json.load(f) != zip_manifest:
            return False
    return all(os.path.exists(os.path.join(BUNDLE, k)) and sha256_file(os.path.join(BUNDLE, k)) == v
               for k, v in zip_manifest['files'].items())


if not os.path.exists(BUNDLE_ZIP):
    raise SystemExit('找不到 %s：請先把 C10_bundle_v1.zip 放到 Google Drive 的 MyDrive/C10/' % BUNDLE_ZIP)
if bundle_ok():
    print('打包檔已解開，逐檔 SHA-256 相符，跳過')
else:
    shutil.rmtree(BUNDLE, ignore_errors=True)
    _local_zip = os.path.join(WORK, os.path.basename(BUNDLE_ZIP))
    shutil.copyfile(BUNDLE_ZIP, _local_zip)
    with zipfile.ZipFile(_local_zip) as z:
        z.extractall(WORK)
    os.remove(_local_zip)
    if not bundle_ok():
        raise SystemExit('打包檔解開後 SHA-256 不符，停止')
    print('打包檔已解開，逐檔 SHA-256 相符')
with open(os.path.join(BUNDLE, 'bundle_manifest.json'), encoding='utf-8') as f:
    MANIFEST = json.load(f)
for k, v in MANIFEST['files'].items():
    print('  %s  %s' % (v[:16], k))
# 紀律 7：Colab 上不能有答案檔
assert not any('answers' in n for _r, _d, _ns in os.walk(BUNDLE) for n in _ns), '打包檔裡有答案檔，停止'

# 打包檔的 tools/（c10data）與 tutorial/（官方 utils、network 原檔）放到 sys.path 最前面
for _sub in ('tools', 'tutorial'):
    _p = os.path.join(BUNDLE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 訓練資料：解開成 WORK/training/train_<LEVEL>.csv（另有 macOS 的 ._train_*.csv，load_rows 不會讀）
# 完成標記和競賽二的 notebook 共用同一個檔名與格式，同一個執行階段跑兩份時不必重解
_archive = os.path.join(BUNDLE, 'training', 'aicup2026_go_training.tar.xz')
_archive_sha = MANIFEST['files']['training/aicup2026_go_training.tar.xz']
_train_dir = os.path.join(WORK, 'training')
_marker = os.path.join(_train_dir, '.c10_extracted.json')
_levels = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
_marker_sha = None
if os.path.exists(_marker):
    with open(_marker, encoding='utf-8') as f:
        _marker_sha = json.load(f).get('archive_sha256')
_have = (_marker_sha == _archive_sha
         and all(os.path.exists(os.path.join(_train_dir, 'train_%s.csv' % lv)) for lv in _levels))
if _have:
    print('訓練資料已解開，跳過')
else:
    shutil.rmtree(_train_dir, ignore_errors=True)
    with tarfile.open(_archive, 'r:xz') as tf:
        if hasattr(tarfile, 'data_filter'):
            tf.extractall(WORK, filter='data')
        else:
            tf.extractall(WORK)
    _absent = [lv for lv in _levels if not os.path.exists(os.path.join(_train_dir, 'train_%s.csv' % lv))]
    if _absent:
        raise SystemExit('訓練資料解開後缺檔：%s' % _absent)
    with open(_marker, 'w', encoding='utf-8') as f:
        json.dump({'archive_sha256': _archive_sha,
                   'bytes': {'train_%s.csv' % lv: os.path.getsize(os.path.join(_train_dir, 'train_%s.csv' % lv))
                             for lv in _levels}}, f)
    print('訓練資料已解開到', _train_dir)

PREPARE_S = time.time() - _t_prepare
print('準備完成 %.1f 秒' % PREPARE_S)
""")

# --------------------------------------------------------------------------- official 0 md -> after our header
off_md(0, note="> **本版註**：下面是官方 notebook 的原始說明。路徑已在上面的設定格設好，不用手改；"
                "Windows／macOS 只跑 SMOKE（本機冒煙測試），正式執行在 Colab。")

off_code(1)

# --------------------------------------------------------------------------- 環境紀錄格
code(r"""# 環境紀錄格（營運性）：版本、硬體、開始時間 → env.json；並開始這一次執行的 run_log／timings 紀錄
import datetime
import importlib.metadata
import platform

import network as _network_mod
import utils as _utils_mod

SOURCE_SHA256 = @@SOURCE_SHA256@@


def check_source(mod, key):
    # 確認模組是從打包檔載入，而且和打包檔 manifest、repo 原檔的 SHA-256 都相同；回傳 SHA-256
    _dir = os.path.abspath(os.path.join(BUNDLE, os.path.dirname(key)))
    if os.path.dirname(os.path.abspath(mod.__file__)) != _dir:
        raise SystemExit('%s 不是從打包檔 %s 載入：%s' % (mod.__name__, _dir, mod.__file__))
    _sha = sha256_file(mod.__file__)
    if _sha != MANIFEST['files'][key] or _sha != SOURCE_SHA256[key]:
        raise SystemExit('%s 的 SHA-256 %s 和打包檔 manifest（%s）或 repo 原檔（%s）不符'
                         % (key, _sha, MANIFEST['files'][key], SOURCE_SHA256[key]))
    return _sha


# 官方 cell 1 import 的 utils／network 必須是打包檔裡的官方原檔
SOURCE_CHECK = {'tutorial/utils.py': check_source(_utils_mod, 'tutorial/utils.py'),
                'tutorial/network.py': check_source(_network_mod, 'tutorial/network.py')}


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def save_json(name, obj):
    # 寫到結果目錄；json 的 float 用 repr，完整不四捨五入。先寫暫存檔再改名
    path = os.path.join(RESULT_DIR, name)
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(path + '.tmp', path)


def load_json(name, default):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def total_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    if os.path.exists('/proc/meminfo'):
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) * 1024 / 1e9
    return None


_cuda = torch.cuda.is_available()
ENV = {
    'started_utc': now_utc(),
    'in_colab': IN_COLAB,
    'smoke': SMOKE,
    'python': sys.version,
    'platform': platform.platform(),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version() if _cuda else None,
    'gpu': torch.cuda.get_device_name(0) if _cuda else None,
    'gpu_count': torch.cuda.device_count(),
    'gpu_mem_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if _cuda else None,
    'cpu_count': os.cpu_count(),
    'ram_gb': total_ram_gb(),
    'work_disk_free_gb': shutil.disk_usage(WORK).free / 1e9,
    'work_inodes_free': os.statvfs(WORK).f_favail if hasattr(os, 'statvfs') else None,
    'mp_start_method': multiprocessing.get_start_method(),
    'versions': {n: pkg_version(n) for n in ('pandas', 'numpy', 'webdataset', 'sgfmill', 'matplotlib')},
    'bundle_manifest_sha256': sha256_file(os.path.join(BUNDLE, 'bundle_manifest.json')),
    'bundle_split': MANIFEST.get('split'),
    'source_sha256': SOURCE_CHECK,           # c10data.py 在官方 cell 4 核對後補進來
}
save_json('env.json', ENV)

# run_log.json：每次執行一筆（開始時間、是否接續、每個 epoch 的步數與耗時、是否完成）
RUN_T0 = time.time()
RUN_LOG = load_json('run_log.json', {'runs': []})
for _r in RUN_LOG['runs']:
    if _r.get('status') == 'running':          # 上一次沒跑完就中斷了
        _r['status'] = 'interrupted'
RUN = {'run': len(RUN_LOG['runs']) + 1, 'started_utc': ENV['started_utc'], 'status': 'running',
       'smoke': SMOKE, 'gpu': ENV['gpu'], 'cpu_count': ENV['cpu_count']}
RUN_LOG['runs'].append(RUN)


def update_run(**kw):
    RUN.update(kw)
    RUN['seconds_so_far'] = time.time() - RUN_T0
    RUN_LOG['n_runs'] = len(RUN_LOG['runs'])
    RUN_LOG['n_resumes'] = sum(1 for r in RUN_LOG['runs'] if r.get('resumed'))
    save_json('run_log.json', RUN_LOG)


TIMINGS = load_json('timings.json', {'runs': {}})
TIMINGS['runs'][str(RUN['run'])] = {}


def record_time(stage, seconds):
    TIMINGS['runs'][str(RUN['run'])][stage] = seconds
    save_json('timings.json', TIMINGS)


update_run()
record_time('prepare_s', PREPARE_S)
print(json.dumps(ENV, ensure_ascii=False, indent=1))
print('這是第 %d 次執行' % RUN['run'])
""".replace("@@SOURCE_SHA256@@", SOURCE_SHA256_SRC))

# --------------------------------------------------------------------------- official 2 (settings)
s = OFF[2]
s = rep(s, """# 官方資料放置位置：training/ 與 tests/ 依官方發佈的結構擺放
TRAIN_DIR = './training'
TEST_DIR = './tests'
FEAT_DIR = './player-features'
INDEX_PATH = './player_index.json'
CKPT_DIR = './player-trained-models'
""", """# 官方資料放置位置：training/ 與 tests/ 依官方發佈的結構擺放
# （本版，營運性：訓練資料在 WORK/training；特徵與索引放執行階段本機磁碟；checkpoint 放 Drive 結果目錄）
TRAIN_DIR = os.path.join(WORK, 'training')
SPLIT_CSV = os.path.join(BUNDLE, 'split_v1', 'players_split.csv')
LOCAL_DIR = os.path.join(BUNDLE, 'local')     # 只有題目檔與候選庫，沒有答案
# ⑥ 題目與候選庫改讀本機驗證檔（取代官方的 TEST_DIR）
QUESTIONS_CSV = os.path.join(LOCAL_DIR, 'player_val_questions.csv')
CANDIDATES_CSV = os.path.join(LOCAL_DIR, 'player_val_candidates.csv')
FEAT_DIR = os.path.join(WORK, 'player-features')
INDEX_PATH = os.path.join(WORK, 'player_index.json')
CKPT_DIR = os.path.join(RESULT_DIR, 'ckpt')
""")
s = rep(s, "NPROC = 8              # 特徵抽取與 SGF 解析的進程數，設成 CPU 核數\n",
        "NPROC = os.cpu_count()  # 特徵抽取與 SGF 解析的進程數，設成 CPU 核數（官方寫死 8；營運性）\n")
s = rep(s, "TOP_N = 5              # 提交要求 Top-5\n", """TOP_N = 5              # 提交要求 Top-5

if SMOKE:
    # SMOKE 只縮小資料量與步數（數值在第一個設定格）；正式執行不會進來
    NPROC = SMOKE_NPROC
    BATCH_SIZE = SMOKE_BATCH_SIZE
    NUM_EPOCHS = SMOKE_NUM_EPOCHS
    STEPS_PER_EPOCH = SMOKE_STEPS_PER_EPOCH
    LOG_INTERVAL = SMOKE_LOG_INTERVAL

# DataLoader 的 worker 數：正式執行同官方（= NPROC）；SMOKE 且非 Colab 時為 0
# （Windows 的 spawn 無法 pickle notebook 內定義的類別；營運性）
NUM_WORKERS = 0 if (SMOKE and not IN_COLAB) else NPROC
""")
s = rep(s, "print('device', DEVICE)\n", "print('device', DEVICE, '| NPROC', NPROC, '| DataLoader workers', NUM_WORKERS)\n")
code(s, official=2, change="modified", tags=["營運性", "⑥"])

# --------------------------------------------------------------------------- 設定紀錄格
code("""# 設定紀錄格（營運性）：全部超參數、實際 NPROC／worker 數、是否 SMOKE、偏離官方的地方 → config.json
DEVIATIONS = [
""" + "".join("    %r,\n" % d for d in DEV_STRINGS) + """]
CONFIG = {
    'notebook': 'colab/prereg01_player.ipynb',
    'official_notebook': 'third_party/aicup2026_tutorial/player-identification.ipynb',
    'prereg': 'docs/PREREG_01_BASELINE.md rev 2，第 4 節（⑤ ⑥）',
    'split': MANIFEST.get('split'),
    'in_colab': IN_COLAB,
    'smoke': SMOKE,
    'hyperparameters': {
        'MIN_GAMES': MIN_GAMES, 'SEED': SEED, 'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE, 'MARGIN': MARGIN, 'NUM_EPOCHS': NUM_EPOCHS,
        'STEPS_PER_EPOCH': STEPS_PER_EPOCH, 'LOG_INTERVAL': LOG_INTERVAL, 'TOP_N': TOP_N,
        'model': 'GoPlayerResNet() 預設值：in_channels 17、num_channels 128、num_blocks 16、embed_dim 256',
        'features': 'SGFParsePlayerIdentification() 預設 history_length 8：8 層自己 + 8 層對手 + 1 層執色 = 17 層',
        'loss': 'TripletMarginLoss(margin=MARGIN, p=2, reduction=mean)',
        'optimizer': 'AdamW(lr=LEARNING_RATE)，其餘預設',
        'precision': 'CUDA 上 bf16 autocast（同官方）',
        'checkpoint_for_scoring': 'ckpt/best_model.pth（訓練 loss 最低，同官方）',
    },
    'official_values': {'NPROC': 8, 'MIN_GAMES': 2, 'SEED': 42, 'BATCH_SIZE': 100, 'LEARNING_RATE': 1e-4,
                        'MARGIN': 0.5, 'NUM_EPOCHS': 5, 'STEPS_PER_EPOCH': 15000},
    'nproc': NPROC,
    'num_workers': NUM_WORKERS,
    'persistent_workers': NUM_WORKERS > 0,
    'device': str(DEVICE),
    'mp_start_method': multiprocessing.get_start_method(),
    'smoke_settings': ({k: v for k, v in globals().items() if k.startswith('SMOKE_')} if SMOKE else None),
    'paths': {'DRIVE_DIR': DRIVE_DIR, 'WORK': WORK, 'RESULT_DIR': RESULT_DIR, 'TRAIN_DIR': TRAIN_DIR,
              'SPLIT_CSV': SPLIT_CSV, 'QUESTIONS_CSV': QUESTIONS_CSV, 'CANDIDATES_CSV': CANDIDATES_CSV,
              'FEAT_DIR': FEAT_DIR, 'INDEX_PATH': INDEX_PATH, 'CKPT_DIR': CKPT_DIR},
    'resumed': None,          # 接續偵測格會填
    'deviations': DEVIATIONS,
}
save_json('config.json', CONFIG)
update_run(nproc=NPROC, num_workers=NUM_WORKERS)
print('config.json 已寫入；偏離官方的地方 %d 項' % len(DEVIATIONS))
""", official=None, change="new", tags=["營運性"])

off_md(3)

# --------------------------------------------------------------------------- official 4 (⑤ load_rows)
s = OFF[4]
s = rep(s, """df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{TRAIN_DIR}/train_*.csv'))],
               ignore_index=True)
""", """# ⑤ 只讀訓練玩家：改用 c10data.load_rows(part="train")（取代官方讀全部 train_*.csv）。
#   load_rows 會 assert 人數與盤數，並確認沒有任何一盤出現在本機驗證檔（題目檔、候選庫）
import c10data
from c10data import load_rows

# c10data 必須從打包檔 tools/ 載入，SHA-256 和打包檔 manifest、repo 原檔都相同（check_source 在環境紀錄格；營運性）
SOURCE_CHECK['tools/c10data.py'] = check_source(c10data, 'tools/c10data.py')
save_json('env.json', ENV)                 # ENV['source_sha256'] 就是 SOURCE_CHECK，補上 c10data 後重寫

_t_load = time.time()
_rows = load_rows(TRAIN_DIR, SPLIT_CSV, part='train', local_dir=LOCAL_DIR)
LOAD_INFO = {
    'call': 'load_rows(TRAIN_DIR, SPLIT_CSV, part="train", local_dir=LOCAL_DIR)',
    'expected': c10data.EXPECTED['train'],
    'got': {'players': len({r['player_id'] for r in _rows}), 'games': len(_rows)},
    'count_assert': 'passed',
    'local_leak_check': 'passed: 回傳的訓練棋沒有任何一盤出現在 local/ 的題目檔與候選庫',
    'c10data_sha256': SOURCE_CHECK['tools/c10data.py'],
    'players_split_sha256': MANIFEST['files']['split_v1/players_split.csv'],
}
df = pd.DataFrame(_rows, columns=c10data.COLUMNS)
del _rows
# 營運性：load_rows 照等級 D…6D 回傳，這裡把列順序排回官方 sorted(glob('train_*.csv')) 的檔案順序
# （1D…6D、A、B、C、D；同一等級內保持檔案裡的列順序）。這只讓特徵分片 df.iloc[i::NPROC] 用官方的切法，
# 而且只有 NPROC 相同時切法才相同；玩家集合本來就和官方不同（只有訓練玩家），分片內容與索引順序不會和官方相同
_file_order = {lv: i for i, lv in enumerate(sorted(c10data.LEVELS, key=lambda lv: 'train_%s.csv' % lv))}
df = df.iloc[np.argsort(df['rank'].map(_file_order).to_numpy(), kind='stable')].reset_index(drop=True)
record_time('load_rows_s', time.time() - _t_load)
print('load_rows(part="train")：%s 位玩家 / %s 局，人數盤數 assert 與外洩檢查都通過（%.0f 秒）'
      % (format(LOAD_INFO['got']['players'], ','), format(LOAD_INFO['got']['games'], ','),
         time.time() - _t_load))

if SMOKE:
    # SMOKE：load_rows 之後才縮小。用 SEED 固定抽幾位 ≥2 盤的玩家，每人只留前幾盤
    _vc = df['player_id'].value_counts()
    _elig = sorted(_vc[_vc >= 2].index)
    _pick = set(np.random.default_rng(SEED).choice(_elig, SMOKE_N_PLAYERS, replace=False))
    df = (df[df['player_id'].isin(_pick)].groupby('player_id', sort=False)
          .head(SMOKE_GAMES_PER_PLAYER).reset_index(drop=True))
    LOAD_INFO['smoke_subset'] = {'players': int(df['player_id'].nunique()), 'games': len(df)}
    print('SMOKE：縮小成 %d 位玩家 / %d 局' % (df['player_id'].nunique(), len(df)))

# 以下照官方
""")
code(s, official=4, change="modified", tags=["⑤", "營運性"])

off_md(5)
off_code(6)
off_md(7)
off_code(8)

# --------------------------------------------------------------------------- official 9 (extract, with marker)
s = OFF[9]
s = rep(s, """# 一次性作業。特徵產出後重跑訓練可以直接從下一節開始。
t0 = time.time()
extract_all(df)
""", """# 一次性作業。特徵產出後重跑訓練可以直接從下一節開始。
# （本版，營運性：特徵放執行階段本機磁碟、不備份到 Drive。同一個執行階段內重跑時，
#   完成標記相符就跳過；否則清掉殘檔重抽）
_fp = hashlib.sha256()
for _k in zip(df['player_id'], df['rank'], df['game_id']):
    _fp.update(('%s|%s|%s\\n' % _k).encode('utf-8'))
FEAT_FINGERPRINT = {'rows_sha256': _fp.hexdigest(), 'n_rows': len(df), 'nproc': NPROC,
                    'min_games': MIN_GAMES, 'smoke': SMOKE}
FEAT_MARKER = os.path.join(WORK, 'player-features.done.json')


def _features_done():
    if not (os.path.exists(FEAT_MARKER) and os.path.exists(INDEX_PATH)):
        return False
    with open(FEAT_MARKER, encoding='utf-8') as f:
        m = json.load(f)
    return m.get('fingerprint') == FEAT_FINGERPRINT and m.get('index_sha256') == sha256_file(INDEX_PATH)


t0 = time.time()
FEAT_SKIPPED = _features_done()
if FEAT_SKIPPED:
    print('特徵已在本機磁碟且完成標記相符，跳過抽取')
else:
    shutil.rmtree(FEAT_DIR, ignore_errors=True)
    for _p in (INDEX_PATH, FEAT_MARKER):
        if os.path.exists(_p):
            os.remove(_p)
    extract_all(df)
    with open(FEAT_MARKER, 'w', encoding='utf-8') as f:
        json.dump({'fingerprint': FEAT_FINGERPRINT, 'index_sha256': sha256_file(INDEX_PATH),
                   'finished_utc': now_utc()}, f)
record_time('extract_s', time.time() - t0)
update_run(features_reextracted=not FEAT_SKIPPED)
""")
code(s, official=9, change="modified", tags=["營運性"])

# --------------------------------------------------------------------------- ⑤ leak check
md("### 本版新增：外洩檢查（⑤）\n\n`player_index.json` 是訓練時唯一的玩家來源。這裡讀 `players_split.csv`，"
   "確認索引裡沒有任何本機驗證玩家（`split=='val'`），也沒有不在切分表裡的玩家；結果連同 load_rows 的 assert 結果存進 `leak_check.json`。")
code(r"""# ⑤ 外洩檢查：player_index.json 裡不能有任何驗證玩家（讀 players_split.csv 判斷）→ leak_check.json
from c10data import read_split

_split = read_split(SPLIT_CSV)
with open(INDEX_PATH) as f:
    _index = json.load(f)
_val_in_index = sorted(p for p in _index if _split.get(p, {}).get('split') == 'val')
_unknown_in_index = sorted(p for p in _index if p not in _split)
_content = json.dumps({k: sorted(v) for k, v in _index.items()}, sort_keys=True)
LEAK_CHECK = {
    'checked_utc': now_utc(),
    'smoke': SMOKE,
    'load_rows': LOAD_INFO,
    'split_players': {s_: sum(1 for v in _split.values() if v['split'] == s_) for s_ in ('train', 'val')},
    'index_players': len(_index),
    'index_games': sum(len(v) for v in _index.values()),
    'index_sha256': sha256_file(INDEX_PATH),
    'index_content_sha256': hashlib.sha256(_content.encode('utf-8')).hexdigest(),
    'val_players_in_index': len(_val_in_index),
    'unknown_players_in_index': len(_unknown_in_index),
    'passed': not _val_in_index and not _unknown_in_index,
}
save_json('leak_check.json', LEAK_CHECK)
assert not _val_in_index, 'player_index.json 裡有 %d 位驗證玩家，例如 %s' % (len(_val_in_index), _val_in_index[:3])
assert not _unknown_in_index, 'player_index.json 裡有 %d 位不在切分表的玩家' % len(_unknown_in_index)
print('外洩檢查通過：索引 %s 位玩家 / %s 局，驗證玩家 0 位'
      % (format(LEAK_CHECK['index_players'], ','), format(LEAK_CHECK['index_games'], ',')))
""", official=None, change="new", tags=["⑤"])

off_md(10)
off_code(11)

# --------------------------------------------------------------------------- resume detection
md("### 本版新增：接續偵測（營運性）\n\n"
   "`ckpt/train_state.pt` 存在就從最後完成的 epoch 接著訓練。設定或索引內容和上次不同時拒絕接續，避免混用。"
   "接續前先用 `train_state.pt` 記的 SHA-256 核對 `ckpt/best_model.pth`，不符就從 `epoch_{best}.pth` 還原。"
   "只有真的接著訓練才算一次接續；訓練已跑完、只重跑推論的不算。")
code(r"""# 接續偵測（營運性）：ckpt/train_state.pt 存在就從最後完成的 epoch 接續；並核對 best_model.pth
STATE_PATH = os.path.join(CKPT_DIR, 'train_state.pt')
BEST_PATH = os.path.join(CKPT_DIR, 'best_model.pth')
TRAIN_FINGERPRINT = {'seed': SEED, 'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'margin': MARGIN,
                     'num_epochs': NUM_EPOCHS, 'steps_per_epoch': STEPS_PER_EPOCH,
                     'min_games': MIN_GAMES, 'smoke': SMOKE}
# 這次執行的抽樣環境：triplet 的抽樣序列取決於 DataLoader 的 worker 數，和 player_index.json 的玩家順序
# （特徵分片數 = NPROC）。換了 CPU 核數不同的執行階段再接續，接續之後抽到的 triplet 會和不中斷時不同
SAMPLING_ENV = {'nproc': NPROC, 'num_workers': NUM_WORKERS, 'cpu_count': os.cpu_count(),
                'index_sha256': LEAK_CHECK['index_sha256']}


def epoch_path(n):
    return os.path.join(CKPT_DIR, 'epoch_%d.pth' % n)


def sync_best(best_epoch, best_sha256, when):
    # 核對 Drive 上的 best_model.pth：SHA-256 要等於 train_state 記的值（best_model.pth 是 epoch_{best}.pth 的複本，
    # 兩者 SHA-256 相同）。不符或不在（例如複製到一半被中斷）就從 epoch_{best}.pth 還原；那個檔也不在或不符就停止
    checks = RUN.setdefault('best_checks', [])
    if best_epoch is None:
        checks.append({'when': when, 'result': 'no_best'})
        update_run(status='failed')
        raise SystemExit('%s：沒有任何 epoch 存成 best_model.pth（每個 epoch 的平均 loss 都沒有低於 inf，可能是 NaN），'
                         '不能推論；請看 ckpt/history.csv' % when)
    if os.path.exists(BEST_PATH) and sha256_file(BEST_PATH) == best_sha256:
        checks.append({'when': when, 'best_epoch': best_epoch, 'result': 'ok'})
        update_run()
        print('%s：best_model.pth 和 train_state.pt 記的 SHA-256 相符（epoch %d）' % (when, best_epoch))
        return
    src = epoch_path(best_epoch)
    reason = 'missing' if not os.path.exists(BEST_PATH) else 'sha256_mismatch'
    why = {'missing': '不存在', 'sha256_mismatch': '的 SHA-256 和 train_state.pt 記的不同'}[reason]
    if not os.path.exists(src) or sha256_file(src) != best_sha256:
        checks.append({'when': when, 'best_epoch': best_epoch, 'result': 'cannot_restore', 'reason': reason,
                       'source_missing': not os.path.exists(src)})
        update_run(status='failed')
        raise SystemExit(
            '%s：ckpt/best_model.pth %s，而用來還原的 ckpt/epoch_%d.pth %s，無法還原。\n'
            '處理方式（二選一）：\n'
            '  1. epoch_%d.pth 如果在 Google Drive 的垃圾桶裡，還原到 %s 之後重新「全部執行」；'
            '它的 SHA-256 必須是 %s。\n'
            '  2. 找不回來的話，把整個 %s 資料夾改名保留（例如 ckpt_broken），再重新「全部執行」，會從頭訓練；'
            '在結果檔寫明重訓的原因。不要只刪 best_model.pth 或 train_state.pt。'
            % (when, why, best_epoch, '也不存在' if not os.path.exists(src) else '的 SHA-256 也不符',
               best_epoch, CKPT_DIR, best_sha256, CKPT_DIR))
    shutil.copyfile(src, BEST_PATH + '.tmp')
    os.replace(BEST_PATH + '.tmp', BEST_PATH)
    if sha256_file(BEST_PATH) != best_sha256:
        update_run(status='failed')
        raise SystemExit('%s：從 epoch_%d.pth 還原後，best_model.pth 的 SHA-256 仍然不符，停止' % (when, best_epoch))
    checks.append({'when': when, 'best_epoch': best_epoch, 'result': 'restored', 'reason': reason,
                   'restored_from': os.path.basename(src), 'restored_utc': now_utc()})
    update_run()
    print('%s：best_model.pth %s，已從 epoch_%d.pth 還原（SHA-256 %s…）' % (when, why, best_epoch, best_sha256[:16]))


TRAIN_STATE = None
if os.path.exists(STATE_PATH):
    TRAIN_STATE = torch.load(STATE_PATH, map_location='cpu', weights_only=False)
    if 'best_epoch' not in TRAIN_STATE:
        raise SystemExit('ckpt/train_state.pt 是舊版格式（沒有 best 的 epoch 與 SHA-256），不能接續；'
                         '請把 %s 資料夾改名保留後重新「全部執行」' % CKPT_DIR)
    if TRAIN_STATE['fingerprint'] != TRAIN_FINGERPRINT:
        raise SystemExit('ckpt/train_state.pt 的設定和現在不同，不能接續：%s vs %s'
                         % (TRAIN_STATE['fingerprint'], TRAIN_FINGERPRINT))
    if TRAIN_STATE['index_content_sha256'] != LEAK_CHECK['index_content_sha256']:
        raise SystemExit('player_index.json 的內容和上次訓練時不同，不能接續')
START_EPOCH = TRAIN_STATE['epoch'] if TRAIN_STATE is not None else 0
# 只有真的從 train_state 接著訓練（已完成的 epoch 數 > 0，而且還有 epoch 要跑）才算接續；
# 訓練已經跑完、只重跑推論的不算
RESUMED = TRAIN_STATE is not None and 0 < START_EPOCH < NUM_EPOCHS
INFERENCE_ONLY = TRAIN_STATE is not None and START_EPOCH >= NUM_EPOCHS
EPOCH_SHA256 = dict(TRAIN_STATE['epoch_sha256']) if TRAIN_STATE is not None else {}   # 每個 epoch_N.pth 的 SHA-256
EPOCH_ENV = list(TRAIN_STATE['epoch_env']) if TRAIN_STATE is not None else []         # 每個 epoch 用的抽樣環境
_prev_env = EPOCH_ENV[-1] if EPOCH_ENV else None
SAMPLING_SAME_AS_SAVED = (all(_prev_env[k] == SAMPLING_ENV[k] for k in ('nproc', 'num_workers', 'index_sha256'))
                          if _prev_env is not None else None)


def save_train_state(epochs_done, best, best_epoch, best_sha256, history):
    # 接續用的完整狀態；先寫暫存檔再改名，寫到一半中斷不會壞掉舊的
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'epoch': epochs_done, 'best': best, 'best_epoch': best_epoch, 'best_sha256': best_sha256,
             'epoch_sha256': EPOCH_SHA256, 'epoch_env': EPOCH_ENV, 'history': history,
             'fingerprint': TRAIN_FINGERPRINT,
             'index_sha256': LEAK_CHECK['index_sha256'],
             'index_content_sha256': LEAK_CHECK['index_content_sha256'],
             'saved_utc': now_utc()}
    torch.save(state, STATE_PATH + '.tmp')
    os.replace(STATE_PATH + '.tmp', STATE_PATH)


def log_epoch(epoch_no, steps, loss, minutes):
    RUN.setdefault('epochs', []).append({'epoch': epoch_no, 'steps': steps, 'loss': loss, 'minutes': minutes,
                                         'nproc': NPROC, 'num_workers': NUM_WORKERS,
                                         'finished_utc': now_utc()})
    update_run(epochs_done=epoch_no)
    record_time('epoch_%d_minutes' % epoch_no, minutes)


update_run(train_state_found=TRAIN_STATE is not None, inference_only=INFERENCE_ONLY,
           start_epoch=START_EPOCH, sampling_env=SAMPLING_ENV,
           saved_sampling_env=_prev_env, sampling_env_same_as_saved=SAMPLING_SAME_AS_SAVED,
           index_sha256=LEAK_CHECK['index_sha256'],
           index_order_same_as_saved=(TRAIN_STATE['index_sha256'] == LEAK_CHECK['index_sha256']
                                      if TRAIN_STATE is not None else None))
if TRAIN_STATE is not None and TRAIN_STATE['best_epoch'] is not None:
    sync_best(TRAIN_STATE['best_epoch'], TRAIN_STATE['best_sha256'], '接續時')
# best 核對成功之後才把這次記成接續；sync_best 停下來的那次不算（避免 n_resumes 多算）
CONFIG.update(train_state_found=TRAIN_STATE is not None, resumed=RESUMED, inference_only=INFERENCE_ONLY,
              start_epoch=START_EPOCH)
save_json('config.json', CONFIG)
update_run(resumed=RESUMED)
if RESUMED:
    print('接續：已完成 %d/%d epoch，最低 loss %r（epoch %s，存於 %s）'
          % (START_EPOCH, NUM_EPOCHS, TRAIN_STATE['best'], TRAIN_STATE['best_epoch'], TRAIN_STATE['saved_utc']))
    if not SAMPLING_SAME_AS_SAVED:
        print('注意：這次的 NPROC／worker 數／索引順序 %s 和上次 %s 不同，接續之後抽到的 triplet 會和不中斷時不同'
              % (SAMPLING_ENV, _prev_env))
elif INFERENCE_ONLY:
    print('訓練已完成（%d/%d epoch），這次只重跑推論，不算接續' % (START_EPOCH, NUM_EPOCHS))
else:
    print('沒有 train_state.pt，從頭訓練')
""", official=None, change="new", tags=["營運性"])

# --------------------------------------------------------------------------- official 12 (loader)
s = OFF[12]
s = rep(s, """train_dataset = TripletDataset(index, STEPS_PER_EPOCH, BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NPROC,
                          pin_memory=True, persistent_workers=True, drop_last=True)
""", """train_dataset = TripletDataset(index, STEPS_PER_EPOCH, BATCH_SIZE)
# 接續時把 epoch 計數設成已完成的 epoch 數，種子序列和不中斷時相同（營運性；從頭跑時是 0，同官方）
train_dataset.epoch = START_EPOCH
# worker 數見設定格；persistent_workers 在 0 worker 時必須是 False（營運性）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                          pin_memory=True, persistent_workers=NUM_WORKERS > 0, drop_last=True)
""")
code(s, official=12, change="modified", tags=["營運性"])

off_md(13)

# --------------------------------------------------------------------------- official 14 (model)
s = OFF[14]
s = rep(s, """criterion = torch.nn.TripletMarginLoss(margin=MARGIN, p=2, reduction='mean')
""", """criterion = torch.nn.TripletMarginLoss(margin=MARGIN, p=2, reduction='mean')
# 接續：載入上次存的模型與 optimizer 狀態（營運性）
if TRAIN_STATE is not None:
    model.load_state_dict(TRAIN_STATE['model'])
    optimizer.load_state_dict(TRAIN_STATE['optimizer'])
    print('已載入 train_state.pt（完成 %d epoch）' % TRAIN_STATE['epoch'])
""")
code(s, official=14, change="modified", tags=["營運性"])

off_code(15)

# --------------------------------------------------------------------------- official 16 (train loop)
s = OFF[16]
s = rep(s, """# 訓練期間不能中斷 kernel。
history = []
model.train()
best = float('inf')

for epoch in range(NUM_EPOCHS):
""", """# 訓練期間不能中斷 kernel。
# （本版，營運性：每個 epoch 結束依序存 epoch_N.pth → 需要時 best_model.pth → train_state.pt；
#   中斷的話重新全部執行，從最後完成的 epoch 接續）
history = list(TRAIN_STATE['history']) if TRAIN_STATE is not None else []
model.train()
best = TRAIN_STATE['best'] if TRAIN_STATE is not None else float('inf')
best_epoch = TRAIN_STATE['best_epoch'] if TRAIN_STATE is not None else None      # 營運性
best_sha256 = TRAIN_STATE['best_sha256'] if TRAIN_STATE is not None else None    # 營運性
_t_train = time.time()

for epoch in range(START_EPOCH, NUM_EPOCHS):
""")
s = rep(s, """    if avg < best:
        best = avg
        torch.save(model.state_dict(), f'{CKPT_DIR}/best_model.pth')
        print('  新的最低 loss %.4f —— 已存 best_model.pth' % best)
    torch.save(model.state_dict(), f'{CKPT_DIR}/epoch_%d.pth' % (epoch + 1))
""", """    # 存檔順序（營運性）：先 epoch_N.pth（同官方），需要時再 best_model.pth，最後 train_state.pt
    torch.save(model.state_dict(), f'{CKPT_DIR}/epoch_%d.pth' % (epoch + 1))
    EPOCH_SHA256[epoch + 1] = sha256_file(epoch_path(epoch + 1))
    if avg < best:
        best = avg
        # best_model.pth 改成複製剛存好的 epoch_N.pth：權重相同，位元組也相同，才能用 SHA-256 核對（營運性）
        best_epoch, best_sha256 = epoch + 1, EPOCH_SHA256[epoch + 1]
        shutil.copyfile(epoch_path(best_epoch), BEST_PATH + '.tmp')
        os.replace(BEST_PATH + '.tmp', BEST_PATH)
        print('  新的最低 loss %.4f —— 已存 best_model.pth' % best)
    EPOCH_ENV.append(dict(SAMPLING_ENV, epoch=epoch + 1, run=RUN['run']))
    # 接續用的狀態最後才寫：寫完才算這個 epoch 完成（營運性）
    save_train_state(epoch + 1, best, best_epoch, best_sha256, history)
    log_epoch(epoch + 1, steps, avg, history[-1]['minutes'])
""")
s = rep(s, """pd.DataFrame(history).to_csv(f'{CKPT_DIR}/history.csv', index=False)
""", """pd.DataFrame(history).to_csv(f'{CKPT_DIR}/history.csv', index=False)
# 訓練結束時再核對一次 best_model.pth，不符就從 epoch_{best}.pth 還原（營運性）
sync_best(best_epoch, best_sha256, '訓練結束時')
record_time('train_s_this_run', time.time() - _t_train)
""")
code(s, official=16, change="modified", tags=["營運性"])

off_md(17)

# --------------------------------------------------------------------------- official 18 (plot)
s = OFF[18]
s = rep(s, """import matplotlib.pyplot as plt
""", """import matplotlib
if SMOKE:
    matplotlib.use('Agg')      # 本機冒煙測試沒有畫面（營運性）
import matplotlib.pyplot as plt
""")
s = rep(s, """plt.ylim(0, 0.55); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
""", """plt.ylim(0, 0.55); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'loss_curve.png'), dpi=100)   # 另存到結果目錄（營運性）
plt.show()
""")
code(s, official=18, change="modified", tags=["營運性"])

off_md(19, note="> **本版註（⑥）**：本機驗證只有一池：候選庫 `local/player_val_candidates.csv`（400 人 × 100 盤）、"
                 "題目 `local/player_val_questions.csv`（400 題，每題 5–20 盤），輸出 `player_val_top5.csv`。"
                 "下面官方說明裡的 public／private 兩池不適用。")

# --------------------------------------------------------------------------- official 20 (predict, ⑥)
s = OFF[20]
s = rep(s, """GPU_BATCH = 4096
CHUNK = 256
""", """GPU_BATCH = 4096
CHUNK = 256
EMB_PATH = os.path.join(RESULT_DIR, 'player_val_embeddings.npz')   # 營運性：另存向量
""")
old_predict = s[s.index("def predict(weights_path, out_path):"):]
new_predict = """def predict(weights_path, out_path):
    net = GoPlayerResNet()
    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
    net.eval().to(DEVICE)

    rows = []
    with Pool(NPROC) as pool:
        # ⑥ 本機只有一池：候選庫與題目都改讀打包檔的 local/，拿掉 public／private 迴圈
        cand = pd.read_csv(CANDIDATES_CSV)
        if SMOKE:
            # SMOKE：候選庫只用照玩家代號排序後的前幾位，每人只用前幾盤
            _keep = sorted(cand['player_id'].unique())[:SMOKE_N_CAND_PLAYERS]
            cand = (cand[cand['player_id'].isin(_keep)].groupby('player_id', sort=False)
                    .head(SMOKE_CAND_GAMES).reset_index(drop=True))
        centroids = mean_embeddings(
            net, pool, list(zip(cand['sgf_content'], cand['color'])),
            list(cand['player_id']), '候選')
        names = sorted(centroids)          # 候選清單照玩家代號排序，不依賴檔案的列順序
        C = np.stack([centroids[n] for n in names])
        print('  重心 %d 個' % len(names), flush=True)

        test = pd.read_csv(QUESTIONS_CSV)
        if SMOKE:
            test = test.head(SMOKE_N_QUESTIONS).reset_index(drop=True)
        sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
        col_cols = [c for c in test.columns if c.startswith('color_')]

        jobs, keys = [], []
        for _, r in test.iterrows():
            used = 0
            # num_games 之後的 sgf_ 欄位是空字串，要濾掉
            for sc, cc in zip(sgf_cols, col_cols):
                if SMOKE and used >= SMOKE_Q_GAMES:
                    break
                if pd.notna(r[sc]) and str(r[sc]).strip():
                    jobs.append((r[sc], r[cc]))
                    keys.append(r['question_id'])
                    used += 1
        queries = mean_embeddings(net, pool, jobs, keys, '查詢')

        n_fallback = 0
        for qid in test['question_id']:
            q = queries.get(qid)
            if q is None:
                rows.append([qid] + names[:TOP_N])
                n_fallback += 1
                continue
            d = np.linalg.norm(C - q, axis=1)
            rows.append([qid] + [names[i] for i in np.argsort(d)[:TOP_N]])
        print('  完成 %d 題' % len(test), flush=True)

    sub = pd.DataFrame(rows,
                       columns=['question_id'] + ['top_%d' % i
                                                  for i in range(1, TOP_N + 1)])
    sub.to_csv(out_path, index=False)
    print('寫出 %s（%d 列）' % (out_path, len(sub)))

    # 另存（營運性，不影響 Top-5）：候選重心與題目平均向量（完整 float32）、推論摘要
    q_ids = [q for q in test['question_id'] if q in queries]
    np.savez(EMB_PATH, candidate_ids=np.array(names), centroids=C,
             question_ids=np.array(q_ids),
             queries=(np.stack([queries[q] for q in q_ids]) if q_ids
                      else np.zeros((0, C.shape[1]), C.dtype)))
    save_json('predict_summary.json', {
        'finished_utc': now_utc(),
        'smoke': SMOKE,
        'weights': weights_path,
        'weights_sha256': sha256_file(weights_path),
        'output': out_path,
        'output_sha256': sha256_file(out_path),
        'questions_csv_sha256': sha256_file(QUESTIONS_CSV),
        'candidates_csv_sha256': sha256_file(CANDIDATES_CSV),
        'n_questions': len(test),
        'n_query_games': len(jobs),
        'n_questions_embedded': len(q_ids),
        'n_fallback_questions': n_fallback,
        'n_candidate_rows': len(cand),
        'n_candidate_players': int(cand['player_id'].nunique()),
        'n_centroids': len(names),
        'GPU_BATCH': GPU_BATCH,
        'CHUNK': CHUNK,
        'NPROC': NPROC,
    })
    return sub
"""
s = rep(s, old_predict, new_predict)
code(s, official=20, change="modified", tags=["⑥", "營運性"])

# --------------------------------------------------------------------------- official 21
s = OFF[21]
s = rep(s, """submission = predict(f'{CKPT_DIR}/best_model.pth', './submission_player.csv')
""", """# ⑥ 只用事前登記指定的 checkpoint：訓練 loss 最低的 best_model.pth
_t_pred = time.time()
submission = predict(f'{CKPT_DIR}/best_model.pth', os.path.join(RESULT_DIR, 'player_val_top5.csv'))
record_time('predict_s', time.time() - _t_pred)
""")
code(s, official=21, change="modified", tags=["⑥", "營運性"])

# --------------------------------------------------------------------------- final
code(r"""# 收尾（營運性）：記下這次執行完成；Colab 上把 Drive 的寫入全部送出再卸載
record_time('total_s', time.time() - RUN_T0)
update_run(status='completed', finished_utc=now_utc(), seconds=time.time() - RUN_T0)
print('完成。結果在', RESULT_DIR)
for _name in sorted(os.listdir(RESULT_DIR)):
    print('  ', _name)

# 結果打包：結果目錄裡除了權重以外的檔案打成一個 zip，放在結果目錄旁邊，方便整包下載
import zipfile
RESULTS_ZIP = os.path.join(os.path.dirname(RESULT_DIR), os.path.basename(RESULT_DIR) + '_results.zip')
with zipfile.ZipFile(RESULTS_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as _z:
    for _root, _dirs, _files in os.walk(RESULT_DIR):
        for _f in sorted(_files):
            _p = os.path.join(_root, _f)
            if _f.endswith(('.pth', '.pt', '.tmp')) or os.path.getsize(_p) > 100 * 1024 * 1024:
                continue   # 權重與暫存檔不帶
            _z.write(_p, os.path.relpath(_p, os.path.dirname(RESULT_DIR)))
print('請下載這個檔回傳給 Claude（打分在本機做）：%s（%.1f MB）'
      % (RESULTS_ZIP, os.path.getsize(RESULTS_ZIP) / 1e6))
if IN_COLAB:
    drive.flush_and_unmount()
    print('Drive 已寫入完成並卸載；要再跑請重新「全部執行」')
""", official=None, change="new", tags=["營運性"])

off_md(22, note="> **本版註**：下面的 0.2845 是官方在官方 public 測試集上的分數，不是本機驗證集的分數；"
                 "本機分數在本機用 `tools/score.py` 打分後記在 `results/score_ledger.csv`。")

# ---------------------------------------------------------------------------
# cell id 固定：官方格沿用官方的 id，新增的格用 c10-player-NN（重跑產生器時輸出逐位元組相同）
_k = 0
for c in cells:
    i = c.metadata["c10"]["official_cell"]
    if i is not None:
        c["id"] = off_nb["cells"][i]["id"]
    else:
        c["id"] = "c10-player-%02d" % _k
        _k += 1
nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata = json.loads(json.dumps(off_nb["metadata"]))     # 沿用官方的 kernelspec、language_info
nb.metadata.update({
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
})
nb.metadata["c10"] = {"official_notebook": "third_party/aicup2026_tutorial/player-identification.ipynb",
                      "deviations": DEV_STRINGS}
nbformat.validate(nb)
# 官方每一格（code 與 markdown）都恰好對應一次，而且相對順序和官方相同
mapped = [c.metadata["c10"]["official_cell"] for c in cells if c.metadata["c10"]["official_cell"] is not None]
assert mapped == list(range(len(OFF))), mapped
for c in cells:
    i = c.metadata["c10"]["official_cell"]
    if i is not None:
        assert c.cell_type == OFF_TYPE[i], (i, c.cell_type)
        # 標成 identical 的格必須和官方逐字相同（排版也不動）；標成 modified 的必須真的不同
        assert (c.source == OFF[i]) == (c.metadata["c10"]["change"] == "identical"), i
        assert c["id"] == off_nb["cells"][i]["id"], i
assert len({c["id"] for c in cells}) == len(cells)
# 每個 code 格都是純 Python（沒有 IPython magic，compile 得過；和 run_local_smoke.py 同一個判斷）、沒有答案檔
for j, c in enumerate(cells):
    if c.cell_type == "code":
        compile(c.source, "<cell %d>" % j, "exec")
        assert "answers.csv" not in c.source and "answers_" not in c.source
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(cells))
