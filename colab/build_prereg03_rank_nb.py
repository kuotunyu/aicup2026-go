"""Build colab/prereg03_rank_a.ipynb and colab/prereg03_rank_b.ipynb from the committed
colab/prereg02_rank.ipynb, and write results/prereg03/design/nb_diff_{a,b}.txt (per-cell unified diff
against the prereg02 notebook, plus the notebook metadata).

Spec: docs/PREREG_03_RECIPE_V2.md section 1 (the registration wins on any scientific question) and
results/prereg03/design/plan.md sections 3.1-3.8 (cells, recipe, arm B sampling, stops, resume, outputs).

How the notebooks are assembled (the discipline of colab/build_prereg02_rank_nb.py and build_test02_rank_nb.py):
  * the base is colab/prereg02_rank.ipynb, whose SHA-256 is asserted first; base cells are picked by their
    `c10` cell metadata, never by position;
  * a reused cell is copied verbatim or changed only through rep() (target exactly once; cut() finds a target
    between two markers that each occur exactly once); a "rewrite" replaces a base cell entirely;
  * every changed base cell is named in CHANGE_ROWS (the table in notebook cell 0 and `changes_from_prereg02`
    in config.json), and every named cell really changed;
  * segment identity: d4 and its self-check move verbatim from the base dump tools into the view/D4 cell;
    write_dump, dump_is_done, write_keys, pack_results, flush_drive, parse_val, sgf_to_features and load_rows'
    cell stay byte-equal to the base;
  * every code cell compiles; the inherited prepare-cell refusal line is the only line of the notebook that
    contains the validation label file token;
  * the A and B notebooks are identical except for the one line ARM = 'A' / ARM = 'B' (asserted on the
    written files);
  * pins are read from the repository at build time (prereg02 key files, prereg02 env.json bundle manifest)
    and asserted against the constants below.
The output is byte-identical when the builder is re-run.

    python colab/build_prereg03_rank_nb.py [REPO] [--out-dir DIR]

--out-dir writes the four outputs (prereg03_rank_a.ipynb, prereg03_rank_b.ipynb, nb_diff_a.txt,
nb_diff_b.txt) into DIR instead of the repository (the test uses it to check byte identity).
"""

import argparse
import ast
import difflib
import hashlib
import json
import os

import nbformat
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("repo", nargs="?", default=os.path.dirname(HERE))
ap.add_argument("--out-dir", help="write the notebooks and diffs here instead of the repository")
ARGS = ap.parse_args()
REPO = os.path.abspath(ARGS.repo)

BASE_REL = "colab/prereg02_rank.ipynb"
BASE_NB = os.path.join(REPO, *BASE_REL.split("/"))
BASE_SHA256 = "9fb1f9d06ba5ee5fbebfb6dfe022246888c93e7762e9a89e1adc99c9f1c12b0c"
OUT_REL = {"A": "colab/prereg03_rank_a.ipynb", "B": "colab/prereg03_rank_b.ipynb"}
DIFF_REL = {"A": "results/prereg03/design/nb_diff_a.txt", "B": "results/prereg03/design/nb_diff_b.txt"}
if ARGS.out_dir:
    OUT_NB = {k: os.path.join(ARGS.out_dir, os.path.basename(v)) for k, v in OUT_REL.items()}
    OUT_DIFF = {k: os.path.join(ARGS.out_dir, os.path.basename(v)) for k, v in DIFF_REL.items()}
else:
    OUT_NB = {k: os.path.join(REPO, *v.split("/")) for k, v in OUT_REL.items()}
    OUT_DIFF = {k: os.path.join(REPO, *v.split("/")) for k, v in DIFF_REL.items()}

# The validation label file token is spelled in two parts so that a repo-wide search for it finds only code
# that refuses such files.
FORBIDDEN_TOKEN = "answ" + "ers"
ARM_LINE = "ARM = '%s'"

# ---------------------------------------------------------------------------- registered constants
# PREREG_03 1.1 / plan 3.3-3.4 (written into the notebook; asserted here once more)
T_TOTAL = 33768
assert T_TOTAL == 12 * 2814
E_L_REGISTERED = {"D": 49949, "C": 49717, "B": 56575, "A": 64188, "1D": 35845, "2D": 45137,
                  "3D": 45024, "4D": 43441, "5D": 46263, "6D": 47407}
TARGET_PER_LEVEL = 35845
assert min(E_L_REGISTERED.values()) == TARGET_PER_LEVEL and 10 * TARGET_PER_LEVEL == 358450

# pins derived from the repository (asserted against these constants below)
SPEC_KEY_DIGEST = {"inner_keys": "d53cb54006141d216b0840b767401a32d9f9bbddb4e82826df077a7397b1e302",
                   "val_keys": "05c8031a9153c6e088b9553ccbd2cb950427a5ffd80d36bddc1347ddf1bef6a5"}
SPEC_BUNDLE_FILES = {
    "training/aicup2026_go_training.tar.xz": "959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b",
    "split_v1/players_split.csv": "69554875064ecbd83d790b3215c8ba8f9540f217ec4af3e4a8c43228ac2a3dfc",
    "tools/c10data.py": "2906d97fe41171dc955288d0a6e695fcef905417d18d94cb98f146aac74c6495",
    "tutorial/utils.py": "3bdf715c6211282be103a75275ce211fdead65703d230b21db5be6ac411c1e63",
    "tutorial/network.py": "794a0d9cc9a20984292a3746bcec6f6e106dac2ef14394cc0d142958693ec3b6",
    "local/rank_val_questions.csv": "73e51bbfd7b051e73cbb62af4b180c403d2422f122240c5e67e9c92303776db5",
    "local/player_val_questions.csv": "d893a559f175b6535dc0d5f749314408b110149462f6a78a1b8cf7e9332cd676",
    "local/player_val_candidates.csv": "7cd2897593ee9add20dce4ff5d4b0dcf8699b6d51a3f10950dc399a757dd3ad5",
}


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rep(src, old, new):
    n = src.count(old)
    assert n == 1, ("rep() target must occur exactly once", n, old[:120])
    return src.replace(old, new)


def cut(src, start, end):
    """The substring of src from the unique `start` marker up to (not including) the unique `end` marker."""
    assert src.count(start) == 1, ("cut() start must occur exactly once", src.count(start), start[:80])
    assert src.count(end) == 1, ("cut() end must occur exactly once", src.count(end), end[:80])
    i, j = src.index(start), src.index(end)
    assert i < j, (start[:40], end[:40])
    return src[i:j]


def keys_digest(arrays):
    """Same function as keys_digest() in the notebook's key-file cell."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(("%s|%s|%s|" % (k, a.dtype.str, a.shape)).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def json_sha256(obj):
    """Same function as json_sha256() in the notebook's setup cell."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------- base notebook
assert sha(BASE_NB) == BASE_SHA256, ("colab/prereg02_rank.ipynb changed", sha(BASE_NB))
base = json.load(open(BASE_NB, encoding="utf-8"))
BASE_CELLS = base["cells"]


def base_key(meta):
    """prereg02 cell metadata -> a readable key (setup, prepare, official_2, dump_tools, final_check, ...)."""
    if "new" in meta:
        return meta["new"]
    b = meta["base"]
    return "official_%d" % b["official_cell"] if "official_cell" in b else b["new"]


BASE_BY_KEY = {}
for _j, _c in enumerate(BASE_CELLS):
    _k = base_key(_c["metadata"]["c10"])
    assert _k not in BASE_BY_KEY, _k
    BASE_BY_KEY[_k] = _j


def base_src(key):
    return "".join(BASE_CELLS[BASE_BY_KEY[key]]["source"])


# ---------------------------------------------------------------------------- pins from the repository
_P02 = os.path.join(REPO, "results", "prereg02", "rank")
KEY_DIGEST = {}
for _stem in ("inner_keys", "val_keys"):
    with np.load(os.path.join(_P02, "dumps", _stem + ".npz"), allow_pickle=False) as _z:
        KEY_DIGEST[_stem] = keys_digest({k: _z[k] for k in _z.files})
assert KEY_DIGEST == SPEC_KEY_DIGEST, KEY_DIGEST
_ENV = json.load(open(os.path.join(_P02, "env.json"), encoding="utf-8"))
assert all(r["bundle_manifest"]["files"] == SPEC_BUNDLE_FILES for r in _ENV["runs"]), "prereg02 bundle manifest"
_ENV1 = json.load(open(os.path.join(REPO, "results", "prereg01", "rank", "env.json"), encoding="utf-8"))
assert all(r["bundle_manifest"]["files"] == SPEC_BUNDLE_FILES for r in _ENV1["runs"]), "prereg01 bundle manifest"
BUNDLE_FILES = SPEC_BUNDLE_FILES
# the real-run feature fingerprint (the prepare/official-9 cells build exactly this dict when SMOKE is False)
FEAT_FINGERPRINT_REAL = {"bundle_files": BUNDLE_FILES, "smoke": False, "smoke_cfg": None, "seed": 42,
                         "per_shard": 1000, "fit_games": 720307, "inner_games": 73556}
FEAT_FINGERPRINT_SHA256 = json_sha256(FEAT_FINGERPRINT_REAL)

# ---------------------------------------------------------------------------
# What changed relative to colab/prereg02_rank.ipynb. One list feeds the table in notebook cell 0 and
# `changes_from_prereg02` in config.json. Each row: (base cell key or None for notebook-wide rows, label, text).
# ---------------------------------------------------------------------------

CHANGE_ROWS = [
    (None, "新增", "第一格 markdown：用途、兩臂、步驟、會停下的情況、輸出清單、接續方式、這張差異表。A、B 兩份 notebook 共用。"),
    ("setup", "設定",
     "開頭加一行 `ARM`（A 臂的 notebook 是 `'A'`、B 臂是 `'B'`；兩份只差這一行）。結果目錄改成 `prereg03/rank_a` 或 `rank_b`（SMOKE 是 `prereg03_smoke`），"
     "分片備份 `FEAT_BACKUP` 改到 `prereg03/features_rank`（兩臂共用、只有 A 寫）。只讀目錄改成三個：`prereg01`、`prereg02`、`test02`"
     "（SMOKE 是 `_smoke`），`guard_write()` 三個都擋，`readonly_fingerprint()` 記檔名、大小、mtime。`sha256_path()`、`gpu_name_of()` "
     "從 prereg02 的 preflight 格搬來（本版的 preflight 在 GPU 關卡之後），另加 `json_sha256()`。拿掉收尾開關 `FINISH_WITHOUT_TRAINING` "
     "和 `SMOKE_CFG_P02`；本版的 SMOKE 設定另放 `SMOKE_CFG_P03`（`SMOKE_CFG` 不動，分片備份的指紋含它）。`RUN` 改記輪次（pass）和狀態。"
     "`config.json` 的 `prereg` 改指 PREREG_03，另記 ARM、只讀目錄、底稿 notebook 的 SHA-256 和這張表。"),
    ("official_2", "底稿官方 2",
     "拿掉 prereg02 的接著訓練設定，改成 recipe v2 的寫死值：`IN_CHANNELS`＝501、`PEAK_LR`＝1e-3、`WEIGHT_DECAY`＝0.01、"
     "`WARMUP_STEPS`＝1,000、`T_TOTAL`＝33,768、`TIME_CHECK_STEP`＝300、時間上限（G4 150 分、A100 180 分）、`SHUFFLE_BUFFER`＝1,000"
     "（SMOKE 只縮小步數和上限），正式執行 assert 這些值。沒有 GPU、或型號不是 G4／A100 就停（沒有收尾開關的例外）。"
     "新增 `GPU_CLASS`（G4／A100／cpu-smoke）、`LOADER_WORKERS`（G4：A 16、B 32；A100：兩臂各 10；SMOKE 0）、`AMP`（正式 bf16，SMOKE 關）。"
     "有 GPU 但查不到型號時，停下的訊息和 `device` 那一行寫「查不到 GPU 型號」（`GPU_MODEL_SHOWN`），不寫 None。"
     "最後的 `print` 印 `device … | GPU … | ARM … | workers … | vCPU … | RAM …`。"),
    (None, "新增 preflight",
     "在 GPU 關卡之後：這一臂有 `STOPPED.json`（之前在訓練後停下）就停；打包檔的檔案清單和 SHA-256 必須等於 prereg01／prereg02 "
     "用的那一份（產生器從 repo 讀出）；Drive 剩餘空間 ≥ 3 GB（`MIN_PASS_FREE_BYTES`，和每一輪開始前的門檻相同）；A 臂這次要重抽特徵、"
     "把分片備份寫到 Drive 時（三個還原來源都沒有設定相符的完成標記）另加 2 GB（備份約 1.3 GB，本機冒煙的分片推估 1.34 GB；prereg01 一節估 1–2 GB），也就是 ≥ 5 GB；查不到剩餘空間也停。"
     "記三個只讀目錄的指紋 `readonly_before.json`，重跑時和第一次記的不同就停。"),
    ("feature_backup_tools", "底稿分片工具", "只改註解：備份只寫到 prereg03 的 `FEAT_BACKUP`（只有 A 臂會呼叫），還原來源可以是 prereg03、prereg02 或 prereg01。"),
    ("official_9", "底稿官方 9",
     "正式執行時特徵指紋的 SHA-256 必須等於產生器算出的 prereg01 指紋（不同就停，不重抽）。A 臂依序試 `prereg03`、`prereg02`、`prereg01` "
     "的 `features_rank` 還原分片，B 臂只試 `prereg02`、`prereg01`（不用 A 臂寫的 prereg03 備份：B 臂要嘛和 A 臂用同一份唯讀備份，"
     "要嘛每次都自己重抽，接續前後的分片排列才相同）；都不能用才重抽，只有 A 臂把重抽的分片備份到 `prereg03/features_rank`，B 臂只抽到本機。"),
    (None, "新增視角與 D4",
     "`d4()` 和它的自我檢查從 dump 工具搬來（一字不改）；`D4_PERM`（[8, 361] 排列表）、`d4_gather()`（每筆一種對稱）、"
     "`to_target_view()`（每個盤面兩層改成「目標玩家的子／對手的子」，接上第 500 層「目標玩家執黑＝1」）、`model_input()`（訓練："
     "視角 → D4 gather → float32 → channels_last）、`dump_input()`（最後輸出：視角 → float32 → `d4()` → channels_last）。"
     "單元檢查：人工棋盤的視角轉換（執黑不變、執白每個盤面兩層對調、第 500 層是常數）、8 種對稱 `gather` 等於 `d4()`、"
     "執色層在 D4 下不變、訓練和最後輸出的輸入相同。"),
    (None, "新增執色表",
     "`COLOR_IS_BLACK`：鍵是分片的 `__key__`＝`等級_game_id`，值取 `df_train`／`df_val` 的 `color` 欄（不從 game_id 推）。"
     "檢查：鍵不重複、只有 B／W、fit 執黑筆數等於 `df_train` 算出的值、執黑比例在 0.50 ± 0.01（SMOKE ± 0.05）。"
     "另外用 `c10data.read_split` 確認 fit、inner、本機驗證三組玩家兩兩不相交，記進 `leak_check.json`。"),
    (None, "新增 B 臂取樣",
     "只有 B 臂生效：用 `df_train` 重算每級「每人最多 200 盤」的總盤數 E_L，必須和登記值完全相同；每盤收下的機率 "
     "q＝min(1, 200/n_p)×35,845/E_L；每一輪用 `sha256(\"c10-prereg03|acc|{pass}|{rank}|{game_id}\")` 前 16 個十六進位字 ÷ 2^64 < q "
     "決定收不收（先看 key，不收的不解碼）。SMOKE 的上限和目標值另用 `SMOKE_CFG_P03`。A 臂只印一行。"),
    ("official_11", "底稿官方 11（改寫）",
     "官方的 `decode_sample`／`make_loader`／`pick_workers` 換掉。worker 傳 uint8 特徵、label、執色（查執色表，查不到就停）和 key；"
     "`make_train_loader(pass, accepted)`：分片先排序、再用 `random.Random(42·1000＋pass)` 洗，依 worker 切（`islice(w, None, workers)`），"
     "B 臂先依 key 篩選，worker 內 shuffle buffer 1,000、rng 種子 (42·1000＋pass)·1000＋worker，批次在 worker 內組成（最後一批不滿也算一步）。"
     "`make_eval_loader()`：照分片順序、不打散。`scan_keys()`：只讀 key、不解碼特徵，另回傳每個分片裡 key 的順序；"
     "`shard_layout_sha256()`：分片排列（依檔名排序的每個分片和它裡面 key 的順序）的 SHA-256，和壓縮位元組無關。"
     "`iterate()`：把 worker 的執色錯誤轉成中文說明後停下。"),
    ("official_12", "底稿官方 12（改寫）",
     "worker 數用 recipe 設定、不照官方退回 0。訓練前把訓練、inner 分片的 key 各讀一遍：數量等於實際寫入的盤數、不重複、"
     "每盤都查得到執色、沒有解析失敗時和 fit／inner 的盤完全相同、執黑盤數相同，任一不符就停；另記兩組分片排列的 SHA-256"
     "（訓練分片的那一個寫進 recipe 設定，接續時必須相同）。建立 inner 監看用的 `val_loader`。"),
    ("official_14", "底稿官方 14",
     "`GoRankResNet(in_channels=501)`，其他和官方相同；建模型前 `torch.manual_seed(SEED)`（兩臂同一個初始化，並記下初始權重的 SHA-256）；"
     "模型用 channels_last（`channels_last()`：官方 forward 在 `bn_in2` 之後用 `view` 攤平，channels_last 的張量不能 `view`，"
     "所以用 forward hook 把 `bn_in2` 的輸出轉回 NCHW，數值不變、`network.py` 不改）；AdamW 明寫 `lr=PEAK_LR`、`weight_decay=0.01`。"),
    ("official_15", "底稿官方 15（改寫）",
     "`validate()`：inner 監看（只記錄），原方向、fp32（不開 autocast）、目標玩家視角（執色查執色表）。"),
    ("dump_tools", "底稿 dump 工具",
     "`d4()` 搬到視角與 D4 的格子；`load_net()` 用 501 層、`channels_last()`；inner 的 loader 改用 `val_loader`（已含 key 和執色）；"
     "`record_p01_after()` 換成三個只讀目錄的 `record_readonly_after()`；`dump_inner`／`dump_val` 換成 `dump_inner_p03`／`dump_val_p03`："
     "fp32、先視角再 D4，一次存 8 種對稱（`<名稱>_d4`，[盤數, 8, 10]）和「原方向、執色對調」（`<名稱>_s0_swap`，[盤數, 10]）；"
     "inner 的執色用 `INNER_KEYS` 的 color 欄並和執色表核對，驗證題的執色用題目檔的 `color_i`。拿掉用不到的規則 B 函式。"),
    (None, "新增 key 檔",
     "`inner_keys`、`val_keys` 的內容摘要（逐陣列的 dtype、形狀、位元組）必須和 prereg02 的 key 檔相同：正式執行比對產生器從 repo 讀出的值，"
     "SMOKE 比對 `prereg02_smoke` 的檔。不同就在訓練前停。"),
    (None, "新增訓練",
     "從零訓練到 T 步（以 global step 計，不管是不是一輪的中間）。每一輪開始時 random／numpy／torch 的種子設為 42·1000＋pass、"
     "D4 的 GPU generator 種子設為 42＋pass；B 臂先算這一輪收下的盤。`LambdaLR`：t＜1,000 時 (t＋1)/1,000，之後 "
     "0.5·(1＋cos(π·(t−1,000)/(T−1,000)))。bf16 autocast（SMOKE 關）、不用 GradScaler。每一輪開始前 Drive 剩不到 3 GB 就先停下"
     "（不是登記的停止：這一輪還沒開始，上一輪的 `state.pt` 已存好；狀態記 `drive_space_pause`，清出空間後「全部執行」從這一輪的開頭接續；"
     "Colab 刪掉或覆寫的檔會進 Drive 的垃圾桶，照樣占空間）。每一輪結束：核對盤數、收到的 key、執黑盤數"
     "（和訓練前讀到的分片 key 比）、學習率，inner 監看，存 `pass_N.pth`（只存在 Colab 的本機磁碟、只留最後兩輪，不寫進 Drive）、"
     "`state.pt`（含 recipe 設定、各 RNG 狀態；recipe 含訓練分片排列的 SHA-256）、"
     "`history.csv`（含資料順序和 D4 抽法的 SHA-256）。第 T 步存 `final.pth`。接續：有 `state.pt` 就從下一輪的開頭重跑，recipe 設定不同就停。"
     "停止：第 300 步推估的訓練時間超過上限、loss 非有限值、每一輪的核對（含學習率）不符 → 寫 `STOPPED.json`、打包 `*_results_failed.zip`、"
     "送出 Drive 的寫入。"),
    (None, "新增最後輸出",
     "只用第 T 步的 `final.pth`（核對 SHA-256）：inner 和本機驗證題的 8 種對稱、原方向執色對調，全部 fp32。"
     "inner 原方向的 accuracy 和最後一輪的監看值相差超過 0.05 個百分點時只提醒。"),
    ("final_check", "底稿結尾（改寫）",
     "核對 6 個 dump 的完成標記、SHA-256、列數、形狀、權重 SHA-256、NaN 位置和 key 檔摘要；Drive 上的 `history.csv` 和記憶體裡的 history"
     "（這次訓練的，或從 `state.pt` 讀回的）逐列相同而且記到第 T 步，不同就先照記憶體裡的重寫再核對（接續時 `state.pt` 已記訓練完成，"
     "訓練格不會再寫 `history.csv`）；三個只讀目錄前後指紋相同；"
     "不符就照停止條件處理（`*_results_failed.zip`）。通過才打包 `prereg03/prereg03_rank_<臂>_results.zip`（不含權重、< 100 MB），"
     "最後 `drive.flush_and_unmount()`。"),
    (None, "拿掉",
     "prereg02 的 preflight（prereg01 的檔案核對）、P1 逐盤輸出與重現核對、接著訓練、P2 逐盤輸出。"),
    (None, "metadata",
     "notebook metadata 和 prereg02 相同。每格 cell id 是 `c10-rank03-NN`，cell metadata `c10` 記底稿的格號、底稿的 `c10` 和是否改動。"
     "A、B 兩份的 cell id 和 metadata 相同。"),
]
P03_CHANGES = ["[%s] %s" % (label, text) for _k, label, text in CHANGE_ROWS]


def _md_cell(s):
    return s.replace("|", "\\|")


CHANGE_TABLE = "\n".join("| %s | %s |" % (label, _md_cell(text)) for _k, label, text in CHANGE_ROWS)

# ---------------------------------------------------------------------------
# Notebook cell 0 (markdown, identical in both notebooks)
# ---------------------------------------------------------------------------

MD_HEADER = r"""# C10 事前登記 03：競賽二（棋力預測）recipe v2 從零訓練（Colab 版；A、B 兩臂）

**這份是 A 臂還是 B 臂**：看第一個程式格的 `ARM = 'A'`／`ARM = 'B'`，或跑起來後 `device` 那一行印的 `ARM`。兩份 notebook 的程式完全相同，只差那一行；這一格說明兩份共用。

**用途**：`docs/PREREG_03_RECIPE_V2.md` 第 1 節的 Colab 部分，一次執行做完：

1. **只讀 prereg01、prereg02、test02**：任何寫進這三個目錄的動作都會被擋下，開頭和結尾比對它們的檔案指紋。
2. **準備**：核對打包檔、讀 fit（5,767 人、720,307 盤）和 inner（640 人、73,556 盤）並做外洩檢查；還原或重抽分片特徵（指紋必須和 prereg01 相同）。
3. **執色表與單元檢查**：執色一律取訓練資料的 `color` 欄；視角轉換用人工棋盤檢查；D4 的 `gather` 和 `d4()` 8 種都要相同；訓練和 inner 分片的每個 key 都要查得到執色。B 臂另外重算每級的 E_L，必須和登記值完全相同。
4. **從零訓練**：官方 `GoRankResNet`，輸入 501 層（每個盤面改成「目標玩家的子／對手的子」，再加 1 層「目標玩家執黑」）；D4 增強（只在訓練時）；AdamW（lr 1e-3、weight decay 0.01）；前 1,000 步 warmup、之後 cosine 降到 0；固定 **33,768 步**；bf16 autocast。A 臂每一輪用全部 fit 棋；B 臂每位玩家最多算 200 盤、各級等量。指定的 checkpoint 是第 33,768 步的 `final.pth`，不另選 best。
5. **存逐盤輸出**（fp32）：inner 和本機驗證題的 8 種對稱 logits，另存「原方向、執色對調」的 logits（只給本機的接線與 κ 檢查用）。

合併、校準、打分都在本機做，不在這裡。這份 notebook 不讀任何答案檔（打包檔裡本來就沒有）。

## 怎麼跑

詳細步驟以 `docs/COLAB_RUN.md` 的「事前登記 03」一節為準；和這裡不一致時，以那一節為準。

1. 把 `prereg03_rank_a.ipynb`、`prereg03_rank_b.ipynb` 放進 `MyDrive/C10/`（旁邊要有 `C10_bundle_v1.zip`）。Drive 開始時要剩 3 GB 以上（A 臂這次要重抽特徵、把分片備份寫到 Drive 時 5 GB 以上）；每一輪覆寫的 `state.pt`（約 0.32 GB）會進 Drive 的垃圾桶、照樣占空間，整個跑完 A 臂約用掉 4 GB（A 臂重抽特徵時另加備份約 1.3 GB）、B 臂約 8 GB（見 `docs/COLAB_RUN.md`）。
2. 選單「執行階段 → 變更執行階段類型」：選 **G4 GPU**；G4 是灰色、點不了時，選 **A100 GPU**，「大量 RAM」關掉。其他 GPU 或沒有 GPU 時，前面幾格就會停下。
3. 選單「執行階段 → 全部執行」。第一個程式格會要求存取 Google Drive，按允許。兩臂同時跑時，**B 要等 A 印出 `device` 那一行之後才按「全部執行」**。
4. 跑完（推估 A 約 1–1.5 小時、B 約 1–2 小時）最後兩行是「Drive 已寫入完成並卸載」和「完成。請下載…」。這時 zip 已經在 Drive 上：先按「執行階段 → 中斷連線並刪除執行階段」（開著會一直扣運算單元），再下載 `MyDrive/C10/results/prereg03/prereg03_rank_a_results.zip`（B 臂是 `prereg03_rank_b_results.zip`）。Drive 空間不夠而暫停時也會先印「Drive 已寫入完成並卸載」，但那時沒有 zip，不要刪執行階段（見下面最後一種停下）。

## 會停下來的情況（錯誤訊息裡都寫了處理方式）

- **沒有 GPU，或拿到的 GPU 不是 G4／A100（或查不到 GPU 型號）**：在設定 `DEVICE` 的那一格停下，還沒開始算。
- **開始訓練前的檢查**：打包檔、特徵指紋、外洩、Drive 空間（3 GB；A 臂要重抽特徵時 5 GB；查不到也停）、執色表、視角與 D4 的單元檢查、分片的 key、key 檔和 prereg02 不同、B 臂的 E_L 和登記值不同。任一不符就停，不訓練。
- **第 300 步推估的訓練時間**超過 150 分鐘（G4）或 180 分鐘（A100）：停下，這一臂不算結果。
- **loss 出現非有限值**（NaN、inf）：停下，這一臂不算結果。
- **接續時 recipe 設定不同**（例如中斷前是 G4、這次拿到 A100，或分片排列不同）：停下，不接續；存好的訓練狀態不會動到。
- **每一輪讀到的盤或學習率和登記的不同**、**收尾核對不符**：停下，這一臂不算結果。
- **某一輪開始前 Drive 剩不到 3 GB**：先暫停（這不是登記的停止，結果照樣算）。清到 Drive 剩 3 GB 以上，再按一次「全部執行」，會從那一輪的開頭接續。
- 會讓這一臂不算結果的停止（上面第 3、4、6 項）會先打包 `prereg03_rank_<臂>_results_failed.zip`、送出 Drive 的寫入，並留下 `STOPPED.json`：之後再按「全部執行」會直接停下。照登記，要先由 Claude 寫修訂才能再跑。其他停下（第 1、2、5、7 項）都發生在計算之前，或只少了還沒存的那一輪，存好的狀態不受影響。

## 結果寫到哪裡

`MyDrive/C10/results/prereg03/rank_a/`（B 臂是 `rank_b/`；下表的 `p03a` 在 B 臂是 `p03b`）：

| 檔案 | 內容 |
|---|---|
| `dumps/inner_keys.npz` | inner 每盤一列：`rank`、`game_id`、`player_id`、`color`、`label`（0–9，D…6D）；和 prereg02 的 key 檔逐陣列相同 |
| `dumps/inner_p03a_d4.npz` | `logits` float32 [73,556, 8, 10]、`weights_sha256`、`syms`（8） |
| `dumps/inner_p03a_s0_swap.npz` | 原方向、執色對調：`logits` float32 [73,556, 10]、`weights_sha256`、`view` |
| `dumps/val_keys.npz` | 本機驗證題每個 slot 一列：`question_id`、`slot`（從 1 起算）、`color`、`parsed`；和 prereg02 的相同 |
| `dumps/val_p03a_d4.npz`、`dumps/val_p03a_s0_swap.npz` | 同上，[19,156, 8, 10]、[19,156, 10]；解析失敗的 slot 整列是 NaN |
| `dumps/_DONE_<檔名>.json` | 每個 dump 的完成標記：列數、SHA-256、權重 SHA-256、秒數、GPU 型號、torch、cuDNN |
| `ckpt/history.csv` | 每一輪：步數、盤數、執黑盤數、loss、train accuracy、inner 監看、學習率、秒數、每步毫秒、資料順序和 D4 抽法的 SHA-256（B 臂另有每級收下的盤數） |
| `ckpt/final.pth`、`ckpt/state.pt` | 權重和接續用的狀態（不打包；指定的 checkpoint 是 `final.pth`）。每一輪的 `pass_N.pth` 只存在 Colab 的本機磁碟（最後兩輪），不寫進 Drive |
| `readonly_before.json`、`readonly_after.json` | 三個只讀目錄每個檔的大小和 mtime（開頭、結尾） |
| `config.json`、`env.json`、`run_log.json`、`timings.json`、`leak_check.json` | 設定（含 recipe、B 臂的 E_L）、環境、每次執行的紀錄、各階段秒數（含 bf16 每步毫秒、第 300 步的推估）、外洩檢查 |
| `STOPPED.json` | 只有訓練開始後停下時才有 |

`MyDrive/C10/results/prereg03/prereg03_rank_a_results.zip`（B 臂是 `…_b_…`）：上面除了權重以外的全部檔案，這個檔帶回本機。

## 中斷後怎麼接續

再按一次「全部執行」就好，**要選中斷前的同一種 GPU**：

- 每一輪結束時存 `ckpt/state.pt`；接續時從下一輪的開頭重跑，資料順序和 D4 的抽法都和沒中斷時相同，最多損失一輪。分片排列（每個分片裡 key 的順序）也寫進 recipe 設定，和中斷前不同就不接續。
- 已完成的逐盤輸出靠完成標記和 SHA-256 跳過。
- 分片特徵：A 臂依序從 prereg03（A 臂重抽時寫的）、prereg02、prereg01 的 Drive 備份還原；B 臂只試 prereg02、prereg01 的備份。都不能用才重抽（約多 6 分鐘；B 臂只抽到本機，不寫備份）。
- cuDNN 和 bf16 不保證逐位元組相同，所以接續過的執行不能逐位元組重現。

## 和 prereg02 notebook 的差異

底稿是已 commit 的 `colab/prereg02_rank.ipynb`（產生時核對 SHA-256）。沿用的格子沒列在下表的，和 prereg02 逐字相同。逐格的 unified diff 在 `results/prereg03/design/nb_diff_a.txt`、`nb_diff_b.txt`。

| 格 | 改了什麼 |
|---|---|
__TABLE__

## 其他

- **本機冒煙測試**：`colab/run_local_smoke.py` 設定 `C10_SMOKE=1`，只用極少量資料在 CPU 上跑，關掉 autocast。先在同一個 `--scratch` 跑 prereg01、prereg02 的冒煙測試（產生 `prereg01_smoke`、`prereg02_smoke`），再跑這兩份；結果寫到 `results/prereg03_smoke/rank_a`、`rank_b`。SMOKE 的步數、warmup、B 臂的上限和第 300 步的檢查見設定格的 `SMOKE_CFG_P03`。
- **不在這裡打分**：答案檔不上傳 Colab。
""".replace("__TABLE__", CHANGE_TABLE)

# ---------------------------------------------------------------------------
# setup (base + rep)
# ---------------------------------------------------------------------------

SETUP_NEW_START = """RUN_START = time.time()

# 本版（prereg03）：這份 notebook 是哪一臂。A、B 兩份 notebook 只差下面這一行，不要改
__ARM_LINE__
"""

SETUP_NEW_DIRS = """# 結果目錄：正式和 SMOKE 分開，絕不混用。本版（prereg03）只寫 prereg03/rank_a 或 rank_b（看 ARM）；
# prereg01、prereg02、test02 只讀
assert ARM in ('A', 'B'), ARM
ARM_TAG = ARM.lower()
RESULT_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg03_smoke' if SMOKE else 'prereg03')
RESULT_DIR = os.path.join(RESULT_ROOT, 'rank_' + ARM_TAG)
RESULT_CKPT = os.path.join(RESULT_DIR, 'ckpt')
FEAT_BACKUP = os.path.join(RESULT_ROOT, 'features_rank')   # 本版的分片備份（兩臂共用；只有 A 臂在重抽時寫）

# 本版：只讀的結果目錄。prereg01、prereg02 的分片備份可以拿來還原（只讀）；test02 不讀，只比對前後的指紋
READONLY_ROOTS = {_n: os.path.join(DRIVE_DIR, 'results', _n + ('_smoke' if SMOKE else ''))
                  for _n in ('prereg01', 'prereg02', 'test02')}
P01_FEAT = os.path.join(READONLY_ROOTS['prereg01'], 'features_rank')
P02_FEAT = os.path.join(READONLY_ROOTS['prereg02'], 'features_rank')
P02_DUMPS = os.path.join(READONLY_ROOTS['prereg02'], 'rank', 'dumps')   # SMOKE 時比對 key 檔用（只讀）


def _real(path):
    return os.path.normcase(os.path.realpath(path))


def _is_under(path, root):
    p, r = _real(path), _real(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def guard_write(path):
    # 本版：寫檔、torch.save、np.savez、copyfile 的目的地、rmtree、zip 一律先過這一關；落在任何一個只讀目錄底下就停
    for _name, _root in READONLY_ROOTS.items():
        if _is_under(path, _root):
            raise PermissionError('本版不能寫入 %s 的目錄（只讀）：%s' % (_name, path))
    return path


def readonly_fingerprint(root):
    # 只讀目錄每個檔的 {相對路徑: [bytes, mtime_ns]}；開頭和結尾各記一次，結尾必須相同（目錄不存在時是空的）
    out = {}
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for n in sorted(files):
            p = os.path.join(base, n)
            st = os.stat(p)
            out[os.path.relpath(p, root).replace(os.sep, '/')] = [st.st_size, st.st_mtime_ns]
    return out


# 本版：這兩個函式原本在 prereg02 的 preflight 格（本版的 preflight 在 GPU 關卡之後，官方 2 就要用 gpu_name_of）
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


def json_sha256(obj):
    # 本版：設定（dict）的 SHA-256：json.dumps(sort_keys=True, ensure_ascii=False) 的 UTF-8 位元組
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


# 本機工作目錄、prereg03 的結果目錄都不能和只讀目錄互相包含
# （沿用的準備格、抽特徵的 worker 只寫 WORK 底下，所以不另外過 guard_write）
for _p in (WORK, RESULT_ROOT):
    guard_write(_p)
    for _root in READONLY_ROOTS.values():
        assert not _is_under(_root, _p), (_root, _p)
guard_write(RESULT_CKPT)
os.makedirs(RESULT_CKPT, exist_ok=True)

"""

SETUP_NEW_DEV = """# 本版（prereg03）的 SMOKE 設定另外放；SMOKE_CFG 不動（分片備份的指紋含它，改了就還原不到 prereg01 的分片）
SMOKE_CFG_P03 = {
    'total_steps': 70,             # T（正式 33,768）：A 臂一輪 63 步，所以會跑到第 2 輪的中間
    'warmup': 10,                  # 正式 1,000
    'cap': 1,                      # B 臂每人上限（正式 200；冒煙的每人盤數很少，改成 1 才測得到上限）
    'time_check_step': 5,          # 推估訓練時間的步數（正式 300）
    'time_limit_minutes': 10 ** 6, # 冒煙不設時間上限（正式 G4 150、A100 180）
}

# 本版和 prereg02 notebook 的差異（與第一格的表同一份）
P03_CHANGES = __P03_CHANGES__

""".replace("__P03_CHANGES__", json.dumps(P03_CHANGES, ensure_ascii=False, indent=4))

SETUP_NEW_PREREG = """    'prereg': 'docs/PREREG_03_RECIPE_V2.md 第 1 節（競賽二；實作規格 results/prereg03/design/plan.md 第 3 節）',
    'arm': ARM,
    'base_notebook': {'path': '__BASE_REL__', 'sha256': '__BASE_SHA__'},
    'changes_from_prereg02': P03_CHANGES,
    'smoke_cfg_p03': SMOKE_CFG_P03 if SMOKE else None,
    'readonly_roots': READONLY_ROOTS,
""".replace("__BASE_REL__", BASE_REL).replace("__BASE_SHA__", BASE_SHA256)

_setup = base_src("setup")
_setup = rep(_setup, "import datetime\nimport json\n", "import datetime\nimport hashlib\nimport json\n")
_setup = rep(_setup, cut(_setup, "RUN_START = time.time()\n", "\n\ndef now_iso():"), SETUP_NEW_START)
_setup = rep(_setup, cut(_setup, "# 結果目錄：正式和 SMOKE 分開，絕不混用。", "# SMOKE 的縮小方式"), SETUP_NEW_DIRS)
_setup = rep(_setup, cut(_setup, "# 本版（prereg02）的 SMOKE 設定另外放", "# 和官方 notebook 的全部差異"), SETUP_NEW_DEV)
_setup = rep(_setup, "    guard_write(path)   # 本版：寫進 prereg01 目錄的一律擋下\n",
             "    guard_write(path)   # 本版：寫進只讀目錄的一律擋下\n")
_setup = rep(_setup, """RUN = {'run': len(run_log['runs']) + 1, 'start': now_iso(), 'end': None, 'smoke': SMOKE,
       'in_colab': IN_COLAB, 'features': None, 'start_epoch': None, 'loaded_state': False,
       'resumed_training': False, 'epochs_trained': []}
""", """RUN = {'run': len(run_log['runs']) + 1, 'start': now_iso(), 'end': None, 'smoke': SMOKE, 'arm': ARM,
       'in_colab': IN_COLAB, 'features': None, 'start_pass': None, 'loaded_state': False,
       'resumed_training': False, 'passes_trained': [], 'status': None}   # 本版：記輪次（pass）和這次執行的結束狀態
""")
_setup = rep(_setup, "run_log['runs'].append(RUN)\n",
             "run_log['runs'].append(RUN)\n"
             "run_log['status'] = None   # 本版：這次執行結束時才寫（completed，或停下時的狀態）\n")
_setup = rep(_setup, cut(_setup, "    'prereg': 'docs/PREREG_02_OPENING_FUSION.md", "    'official_notebook':"),
             SETUP_NEW_PREREG)
_setup = rep(_setup, "print('結果目錄', RESULT_DIR)\nprint('prereg01（只讀）', P01_ROOT)",
             "print('ARM', ARM, '| 結果目錄', RESULT_DIR)\n"
             "print('只讀：', ' | '.join('%s %s' % kv for kv in READONLY_ROOTS.items()))")
_setup_code = _setup.replace(SETUP_NEW_DEV, "")   # the change table names the removed switch; the code must not
assert "FINISH_WITHOUT_TRAINING" not in _setup_code and "P01_ROOT" not in _setup_code and "SMOKE_CFG_P02" not in _setup_code

# ---------------------------------------------------------------------------
# official 2 (base + rep): recipe constants, GPU gate without the finish switch, workers, AMP
# ---------------------------------------------------------------------------

OFF2_NEW_RECIPE = """# 本版（prereg03）：recipe v2 的寫死值（PREREG_03 1.1；plan 3.3）。SMOKE 只縮小步數、warmup 和時間檢查（SMOKE_CFG_P03）
IN_CHANNELS = 501                 # 250 個盤面 ×「目標玩家的子、對手的子」＋ 1 層「目標玩家執黑」
PEAK_LR = LEARNING_RATE           # 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
T_TOTAL = 33768                   # ＝ 12 × 2,814；和 worker 數、GPU 型號無關。第 T 步的權重是指定的 checkpoint
TIME_CHECK_STEP = 300             # 第 300 步推估總訓練時間
TIME_LIMIT_MINUTES = {'G4': 150, 'A100': 180}
SHUFFLE_BUFFER = 1000             # worker 內的 shuffle buffer（同官方）
if SMOKE:
    T_TOTAL = SMOKE_CFG_P03['total_steps']
    WARMUP_STEPS = SMOKE_CFG_P03['warmup']
    TIME_CHECK_STEP = SMOKE_CFG_P03['time_check_step']
    TIME_LIMIT_MINUTES = {'cpu-smoke': SMOKE_CFG_P03['time_limit_minutes']}
else:
    assert (IN_CHANNELS, PEAK_LR, WEIGHT_DECAY, WARMUP_STEPS, T_TOTAL, TIME_CHECK_STEP, BATCH_SIZE, SEED) == (
        501, 1e-3, 0.01, 1000, 33768, 300, 256, 42), 'prereg03 的設定被改動'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 本版：正式執行一定要有 GPU（訓練和逐盤輸出在 CPU 上要好幾天）
if not SMOKE and DEVICE.type != 'cuda':
    raise SystemExit('沒有 GPU，停在這裡（還沒開始算）。處理方式：選單「執行階段 → 變更執行階段類型」選 G4 GPU'
                     '（灰色時選 A100 GPU，「大量 RAM」關掉）→ 儲存，再按一次「全部執行」。')
"""

OFF2_NEW_CLASS = """# 本版（prereg03）：GPU 類別決定 DataLoader 的 worker 數（寫進 recipe 設定，接續時必須相同；plan 3.3）：
# G4 → A 臂 16、B 臂 32；A100 → 兩臂各 10；本機冒煙（CPU）→ 0（Windows 的 spawn 無法 pickle notebook 內定義的函式）
if DEVICE.type != 'cuda':
    GPU_CLASS = 'cpu-smoke'
elif 'RTX PRO 6000' in (GPU_MODEL_NOW or ''):
    GPU_CLASS = 'G4'
else:
    GPU_CLASS = 'A100'
LOADER_WORKERS = {'G4': {'A': 16, 'B': 32}, 'A100': {'A': 10, 'B': 10}, 'cpu-smoke': {'A': 0, 'B': 0}}[GPU_CLASS][ARM]
if SMOKE and not IN_COLAB:
    LOADER_WORKERS = SMOKE_CFG['loader_workers']
AMP = DEVICE.type == 'cuda' and not SMOKE     # 訓練用 bf16 autocast；監看和逐盤輸出一律 fp32；CPU 冒煙關掉
_ram = ENV_INFO.get('ram_bytes')

np.random.seed(SEED)
"""

OFF2_GPU_SHOWN = """GPU_MODEL_NOW = gpu_name_of(ENV_INFO)
# 本版（prereg03）：有 GPU 但查不到型號（nvidia-smi 沒有回報）時，訊息和 device 那一行寫「查不到 GPU 型號」，不寫 None
GPU_MODEL_SHOWN = GPU_MODEL_NOW or ('查不到 GPU 型號' if DEVICE.type == 'cuda' else None)
"""

_off2 = base_src("official_2")
_off2 = rep(_off2, cut(_off2, "# 本版（prereg02）：NUM_EPOCHS 維持", "# 本版：GPU 型號只接受 G4"), OFF2_NEW_RECIPE)
_off2 = rep(_off2, "GPU_MODEL_NOW = gpu_name_of(ENV_INFO)\n", OFF2_GPU_SHOWN)
_off2 = rep(_off2, "    raise SystemExit('這次拿到的 GPU 是 %s，不是 G4（RTX PRO 6000）或 A100，停在這裡（還沒開始算）。處理方式：'\n",
            "    raise SystemExit('%s，停在這裡（還沒開始算）。處理方式：'\n")
_off2 = rep(_off2, "'（灰色時選 A100 GPU，「大量 RAM」關掉）→ 儲存，再按一次「全部執行」。' % GPU_MODEL_NOW)\n",
            "'（灰色時選 A100 GPU，「大量 RAM」關掉）→ 儲存，再按一次「全部執行」。'\n"
            "                     % ('這次拿到的 GPU 是 %s，不是 G4（RTX PRO 6000）或 A100' % GPU_MODEL_NOW if GPU_MODEL_NOW else\n"
            "                        '有 GPU，但查不到 GPU 型號，無法確認是 G4（RTX PRO 6000）或 A100'))\n")
_off2 = rep(_off2, "np.random.seed(SEED)\n", OFF2_NEW_CLASS)
_off2 = rep(_off2, "print('device', DEVICE, '| GPU', GPU_MODEL_NOW, '| NPROC', NPROC, '| batch', BATCH_SIZE,\n"
                   "      '| epochs', NUM_EPOCHS, '| extra epochs', EXTRA_EPOCHS, '| final epoch', FINAL_EPOCH)",
            "print('device', DEVICE, '| GPU', GPU_MODEL_SHOWN, '| ARM', ARM, '| workers', LOADER_WORKERS, '| vCPU', os.cpu_count(),\n"
            "      '| RAM', '%.0f GB' % (_ram / 1e9) if _ram else None, '| NPROC', NPROC, '| batch', BATCH_SIZE, '| T', T_TOTAL,\n"
            "      '| AMP', 'bf16' if AMP else 'off')")
assert "FINISH_WITHOUT_TRAINING" not in _off2 and "EXTRA_EPOCHS" not in _off2

# ---------------------------------------------------------------------------
# preflight (new, after the GPU gate)
# ---------------------------------------------------------------------------

PREFLIGHT = r"""# 本版新增（preflight，在 GPU 關卡之後）：這一臂是否已停下、打包檔是不是登記用的那一份、Drive 空間、只讀目錄的指紋（開頭）。
# 任何一項不符就用中文說明處理方式後停下，還沒開始算（但執行階段開著就會扣運算單元）。
t0 = time.time()
STOPPED_PATH = os.path.join(RESULT_DIR, 'STOPPED.json')
# 產生器從 repo 讀出的釘選值：prereg01、prereg02 用的打包檔（results/prereg0{1,2}/rank/env.json 的 bundle_manifest）
BUNDLE_FILES_PIN = __BUNDLE_FILES__
# Drive 剩餘空間的門檻（登記：開始前 ≥ 3 GB）。每一輪開始前（訓練格）用同一個門檻，暫停後清到這個數字再「全部執行」就過得了這裡
MIN_PASS_FREE_BYTES = 3 * 10 ** 9
FEATURE_BACKUP_ROOM_BYTES = 2 * 10 ** 9   # A 臂重抽特徵時寫到 Drive 的分片備份（約 1.3 GB，最多估 2 GB）：開始前另外多留這麼多


def gb_text(n):
    # 給使用者看的 GB 數（3 GB、5 GB）
    return '%g GB' % (n / 1e9)


def feature_backup_marker_ok(src):
    # src（某個 features_rank）有沒有設定相符的分片完成標記。只看標記裡的設定（筆數在讀完資料後才知道，官方 9 還原時會再逐檔核對）
    try:
        fp = (read_json(os.path.join(src, '_COMPLETE.json')) or {}).get('fingerprint')
    except (OSError, ValueError, AttributeError):
        return False
    known = {'bundle_files': MANIFEST['files'], 'smoke': SMOKE, 'smoke_cfg': SMOKE_CFG if SMOKE else None,
             'seed': SEED, 'per_shard': PER_SHARD}
    return isinstance(fp, dict) and all(fp.get(k) == v for k, v in known.items())


# A 臂在三個還原來源（和官方 9 同一個順序）都沒有設定相符的備份時會重抽特徵，並把約 1.3 GB 的分片備份寫到 Drive
WILL_WRITE_FEATURE_BACKUP = ARM == 'A' and not any(feature_backup_marker_ok(_s) for _s in (FEAT_BACKUP, P02_FEAT, P01_FEAT))
MIN_DRIVE_FREE_BYTES = MIN_PASS_FREE_BYTES + (FEATURE_BACKUP_ROOM_BYTES if WILL_WRITE_FEATURE_BACKUP else 0)

# ① 訓練開始後停下過（STOPPED.json）：照登記這一臂不算結果，要先寫修訂，不能接續或直接重跑
_stopped = read_json(STOPPED_PATH)
if _stopped is not None:
    raise SystemExit('這一臂之前已經停下（狀態 %s，第 %s 次執行，%s）。照 PREREG_03，這一臂不能接續或直接重跑，要先由 Claude '
                     '寫修訂。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                     % (_stopped.get('status'), _stopped.get('run'), _stopped.get('time')))

# ② 打包檔：和 prereg01、prereg02 用的是同一份（準備格已逐檔核對 SHA-256 和 bundle_manifest.json 相符）
if MANIFEST['files'] != BUNDLE_FILES_PIN:
    raise SystemExit('Drive 上的 C10_bundle_v1.zip 和 prereg01／prereg02 用的那一份不同（檔案清單或 SHA-256 不同）。'
                     '處理方式：不要自己重傳，把這段訊息截圖給 Claude。')

# ③ Drive 剩餘空間（登記：開始前 ≥ 3 GB；查不到也停）。訓練中每一輪開始前另查一次（同樣是 3 GB，見訓練格）；
#    A 臂這次要寫分片備份時，開始前要 5 GB（備份寫完後，第 1 輪開始前還剩 3 GB 以上）
try:
    DRIVE_FREE_BYTES = shutil.disk_usage(DRIVE_DIR).free
except OSError as e:
    raise SystemExit('查不到 Drive 的剩餘空間（%s: %s），還沒開始算。處理方式：「執行階段 → 中斷連線並刪除執行階段」，再按一次「全部執行」；'
                     '還是一樣就把這段訊息截圖給 Claude。' % (type(e).__name__, e))
if DRIVE_FREE_BYTES < MIN_DRIVE_FREE_BYTES:
    raise SystemExit('Drive 只剩 %.2f GB，這份 notebook 開始前要剩 %s 以上（%s）。還沒開始算，不用刪執行階段。'
                     '處理方式：到 Drive 左邊的「垃圾桶」，垃圾桶裡沒有要留的東西就按「清空垃圾桶」，有要留的就只把名稱是 state.pt、'
                     '或 .pth 結尾的項目選起來 →「永久刪除」；等 Drive 剩 %s 以上（左下角「共 … GB」減「已使用 … GB」，'
                     '可能要幾分鐘才更新），再按一次「全部執行」。垃圾桶清空了還不到 %s：C10 裡的東西都不要刪，'
                     '照 Claude 給的步驟（docs/COLAB_RUN.md「Drive 空間不夠時」）處理。'
                     % (DRIVE_FREE_BYTES / 1e9, gb_text(MIN_DRIVE_FREE_BYTES),
                        'A 臂這次要重抽特徵，分片備份約 1.3 GB 會寫到 Drive，所以比平常的 %s 多留 %s'
                        % (gb_text(MIN_PASS_FREE_BYTES), gb_text(FEATURE_BACKUP_ROOM_BYTES)) if WILL_WRITE_FEATURE_BACKUP else
                        '每一輪開始前也要剩這麼多；每一輪覆寫的 state.pt 約 0.32 GB 會進垃圾桶、照樣占空間',
                        gb_text(MIN_DRIVE_FREE_BYTES), gb_text(MIN_DRIVE_FREE_BYTES)))

# ④ 只讀目錄的指紋（開頭）。重跑時必須和第一次執行記下的相同：不同代表 prereg01／prereg02／test02 在兩次執行之間被改過
READONLY_BEFORE = {_name: readonly_fingerprint(_root) for _name, _root in READONLY_ROOTS.items()}
_before_path = os.path.join(RESULT_DIR, 'readonly_before.json')
_before_log = read_runs(_before_path)
if _before_log['runs'] and _before_log['runs'][0]['files'] != READONLY_BEFORE:
    _first = _before_log['runs'][0]['files']
    _changed = {_k: sorted(f for f in set(_first.get(_k, {})) | set(READONLY_BEFORE[_k])
                           if _first.get(_k, {}).get(f) != READONLY_BEFORE[_k].get(f)) for _k in READONLY_BEFORE}
    raise SystemExit('只讀目錄和第一次執行時不同（%s）。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                     % {k: v[:5] for k, v in _changed.items() if v})
_before_log['runs'].append({'run': RUN['run'], 'time': now_iso(), 'roots': READONLY_ROOTS,
                            'n_files': {k: len(v) for k, v in READONLY_BEFORE.items()}, 'files': READONLY_BEFORE})
write_json(_before_path, _before_log)

save_config(preflight={'arm': ARM, 'gpu_class': GPU_CLASS, 'loader_workers': LOADER_WORKERS, 'amp': AMP,
                       'bundle_files_match_pin': True, 'drive_free_bytes': DRIVE_FREE_BYTES,
                       'min_drive_free_bytes': MIN_DRIVE_FREE_BYTES, 'min_pass_free_bytes': MIN_PASS_FREE_BYTES,
                       'will_write_feature_backup': WILL_WRITE_FEATURE_BACKUP,
                       'cpu_count': os.cpu_count(), 'ram_bytes': ENV_INFO.get('ram_bytes'),
                       'readonly_files_fingerprinted': {k: len(v) for k, v in READONLY_BEFORE.items()}})
mark_time('preflight', time.time() - t0)
print('preflight 通過：ARM %s，GPU %s（%s），DataLoader workers %d，打包檔相符，Drive 剩 %.2f GB（門檻 %s），只讀目錄 %s 個檔'
      % (ARM, GPU_MODEL_SHOWN, GPU_CLASS, LOADER_WORKERS, DRIVE_FREE_BYTES / 1e9, gb_text(MIN_DRIVE_FREE_BYTES),
         {k: len(v) for k, v in READONLY_BEFORE.items()}))
""".replace("__BUNDLE_FILES__", json.dumps(BUNDLE_FILES, indent=4).replace("\n", "\n    ").replace("\n    }", "\n}"))

# ---------------------------------------------------------------------------
# feature backup tools (base + rep: comments only) and official 9 (base + rep)
# ---------------------------------------------------------------------------

_ft = base_src("feature_backup_tools")
_ft = rep(_ft, "    # 全部複製並核對大小之後，才寫完成標記。本版：只寫到 prereg02 的 FEAT_BACKUP\n",
          "    # 全部複製並核對大小之後，才寫完成標記。本版（prereg03）：只寫到 prereg03 的 FEAT_BACKUP（只有 A 臂會呼叫）\n")
_ft = rep(_ft, "    # 本版：備份來源 src 可以指定（prereg02 或 prereg01 的 features_rank），對來源只讀；只寫本機的 FEAT_DIR\n",
          "    # 本版：備份來源 src 可以指定（prereg03、prereg02 或 prereg01 的 features_rank），對來源只讀；只寫本機的 FEAT_DIR\n")

OFF9_NEW_RESTORE = """# 本版（prereg03）：正式執行的特徵指紋必須和 prereg01 抽特徵時相同（產生器算出的 SHA-256；PREREG_03 plan 3.2 第 4 格）
FEAT_FINGERPRINT_SHA256_PIN = '__FP_SHA__'
if not SMOKE and json_sha256(FEAT_FINGERPRINT) != FEAT_FINGERPRINT_SHA256_PIN:
    raise SystemExit('特徵指紋和 prereg01 不同（%s…），還沒開始算。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                     % json_sha256(FEAT_FINGERPRINT)[:12])
# 本版：A 臂依序試 prereg03 自己的備份、prereg02、prereg01 的備份（後兩個只讀）；B 臂只試 prereg02、prereg01 的備份
# （prereg03 的備份是 A 臂用 A 臂那台機器的 NPROC 抽的，分片排列可能和 B 臂自己抽的不同；B 臂不用它，接續前後的分片排列才相同）。
# 都不能用才重抽
restored = None
for _src in ((FEAT_BACKUP, P02_FEAT, P01_FEAT) if ARM == 'A' else (P02_FEAT, P01_FEAT)):
""".replace("__FP_SHA__", FEAT_FINGERPRINT_SHA256)

_off9 = base_src("official_9")
_off9 = rep(_off9, """# 本版：先試 prereg02 自己的備份，再試 prereg01 的備份（只讀）；兩個都不能用才重抽
restored = None
for _src in (FEAT_BACKUP, P01_FEAT):
""", OFF9_NEW_RESTORE)
_off9 = rep(_off9, "    print('prereg02 和 prereg01 都沒有可用的分片備份：重抽（約多 6 分鐘），抽完備份到 %s' % FEAT_BACKUP)\n",
            "    print('%s都沒有可用的分片備份：重抽（約多 6 分鐘）%s'\n"
            "          % ('prereg03、prereg02、prereg01 ' if ARM == 'A' else 'prereg02、prereg01 ',\n"
            "             '，抽完備份到 %s' % FEAT_BACKUP if ARM == 'A' else '（B 臂只抽到本機，不寫備份）'))\n")
_off9 = rep(_off9, """    t1 = time.time()
    backup_features(FEAT_FINGERPRINT, {'val': val_counts, 'train': train_counts})
    mark_time('features_backup', time.time() - t1)
""", """    if ARM == 'A':   # 本版：只有 A 臂寫備份（兩臂同時跑時不會搶寫同一個資料夾）
        t1 = time.time()
        backup_features(FEAT_FINGERPRINT, {'val': val_counts, 'train': train_counts})
        mark_time('features_backup', time.time() - t1)
""")
_off9 = rep(_off9, "save_run_log()\n", "RUN['features_fingerprint_sha256'] = json_sha256(FEAT_FINGERPRINT)\nsave_run_log()\n")

# ---------------------------------------------------------------------------
# view / D4 (new; d4 and its self-check moved verbatim from the base dump tools)
# ---------------------------------------------------------------------------

_dt_base = base_src("dump_tools")
D4_SEGMENT = cut(_dt_base, "def d4(x, s):", "\n\n\ndef load_net(")
assert D4_SEGMENT.count("assert") == 2 and D4_SEGMENT.endswith("for b in range(NUM_SYMS))"), D4_SEGMENT[-80:]

VIEW_D4 = r"""# 本版新增：目標玩家視角與 D4 對稱（PREREG_03 1.1；plan 3.2 第 5 格、3.3）
# - 官方特徵是每個盤面「黑子、白子」兩層（250 個盤面、500 層，uint8）。to_target_view 改成「目標玩家的子、對手的子」
#   （第 2t 層、第 2t+1 層），再接第 500 層「目標玩家執黑＝1」，共 501 層。在 GPU 上從 uint8 轉
# - d4(x, s)：8 種對稱（從 prereg02 的 dump 工具搬來，一字不改）。d4_gather：訓練時用 [8, 361] 排列表，每筆各做一種
# - 訓練的輸入：視角 → D4（gather）→ float32 → channels_last；最後輸出：視角 → float32 → d4() → channels_last（兩者相同，下面檢查）
# - 執色層在 D4 下不變
NUM_SYMS = 8


__D4_SEGMENT__


D4_PERM = torch.stack([d4(torch.arange(361).view(1, 1, 19, 19), s).reshape(361) for s in range(NUM_SYMS)])  # [8, 361]
assert all(sorted(D4_PERM[s].tolist()) == list(range(361)) for s in range(NUM_SYMS))
D4_PERM_DEV = D4_PERM.to(DEVICE)


def d4_gather(x, s):
    # x：[B, C, 19, 19]（任何 dtype）；s：[B] 的 0..7（和 x 在同一個 device）。第 b 筆等於 d4(x[b:b+1], s[b])
    n, c = x.shape[0], x.shape[1]
    idx = D4_PERM_DEV[s][:, None, :].expand(n, c, 361)
    return x.reshape(n, c, 361).gather(2, idx).view(n, c, 19, 19)


def to_target_view(x, is_black):
    # x：[B, 500, 19, 19]（官方特徵，第 2t 層黑子、第 2t+1 層白子）；is_black：[B] bool，目標玩家執黑。回傳 [B, 501, 19, 19]
    n = x.shape[0]
    y = x.view(n, 250, 2, 19, 19)
    y = torch.where(is_black[:, None, None, None, None], y, y.flip(2)).reshape(n, 500, 19, 19)
    colour = is_black.to(x.dtype)[:, None, None, None].expand(n, 1, 19, 19)
    return torch.cat([y, colour], dim=1)


def model_input(x, is_black, s=None):
    # 訓練（s 是每筆的對稱）和 inner 監看（s=None，原方向）的輸入：uint8 → 視角 →（D4）→ float32 → channels_last
    y = to_target_view(x, is_black)
    if s is not None:
        y = d4_gather(y, s)
    return y.float().contiguous(memory_format=torch.channels_last)


def dump_input(x, is_black, s):
    # 最後輸出的輸入：uint8 → 視角 → float32 → d4(·, s) → channels_last（先做視角轉換，再做 D4）
    return d4(to_target_view(x, is_black).float(), s).contiguous(memory_format=torch.channels_last)


# 單元檢查（在 DEVICE 上；用自己的 generator，不動全域的 RNG）
_g = torch.Generator().manual_seed(0)
# ① 視角：人工棋盤，兩筆（執黑、執白），每個盤面的黑子、白子不重疊
_r = torch.randint(0, 3, (2, 250, 19, 19), generator=_g)
_x = torch.zeros(2, 500, 19, 19, dtype=torch.uint8)
_x[:, 0::2], _x[:, 1::2] = (_r == 1).to(torch.uint8), (_r == 2).to(torch.uint8)
_x = _x.to(DEVICE)
_isb = torch.tensor([True, False], device=DEVICE)
_v = to_target_view(_x, _isb)
assert _v.shape == (2, 501, 19, 19) and _v.dtype == torch.uint8
assert torch.equal(_v[0, :500], _x[0]), '執黑的樣本必須不變'
assert torch.equal(_v[1, 0:500:2], _x[1, 1:500:2]) and torch.equal(_v[1, 1:500:2], _x[1, 0:500:2]), '執白的樣本每個盤面兩層要對調'
assert bool((_v[0, 500] == 1).all()) and bool((_v[1, 500] == 0).all()), '第 500 層要是常數（執黑 1、執白 0）'
assert torch.equal(to_target_view(_x, ~_isb)[0, 0:500:2], _x[0, 1:500:2]), '執色對調要換成另一方的子'
# ② D4：8 種對稱 gather 都等於 d4()（uint8 和 float32 都測）；執色層在 D4 下不變
_y = torch.randint(0, 256, (NUM_SYMS, IN_CHANNELS, 19, 19), generator=_g).to(torch.uint8).to(DEVICE)
_s = torch.arange(NUM_SYMS, device=DEVICE)
_gy, _gyf = d4_gather(_y, _s), d4_gather(_y.float(), _s)
for _k in range(NUM_SYMS):
    assert torch.equal(_gy[_k:_k + 1], d4(_y[_k:_k + 1], _k)), ('D4 gather 和 d4() 不同', _k)
    assert torch.equal(_gyf[_k:_k + 1], d4(_y[_k:_k + 1].float(), _k)), ('D4 gather 和 d4() 不同（float32）', _k)
    _vk = d4_gather(_v, torch.full((2,), _k, device=DEVICE, dtype=torch.long))
    assert torch.equal(_vk[:, 500], _v[:, 500]), ('執色層在 D4 下要不變', _k)
    # ③ 訓練的輸入和最後輸出的輸入相同
    assert torch.equal(model_input(_x, _isb, torch.full((2,), _k, device=DEVICE, dtype=torch.long)),
                       dump_input(_x, _isb, _k)), ('訓練和最後輸出的輸入不同', _k)
assert torch.equal(model_input(_x, _isb), dump_input(_x, _isb, 0))
VIEW_D4_CHECK = {'view_black_unchanged': True, 'view_white_swapped': True, 'colour_plane_constant': True,
                 'd4_gather_equals_d4': NUM_SYMS, 'colour_plane_d4_invariant': True, 'train_input_equals_dump_input': True,
                 'device': str(DEVICE)}
save_config(view_d4_check=VIEW_D4_CHECK)
print('視角與 D4 的單元檢查通過（%s）：執黑不變、執白兩層對調、第 500 層是常數、8 種對稱 gather＝d4()' % DEVICE)
""".replace("__D4_SEGMENT__", D4_SEGMENT)

# ---------------------------------------------------------------------------
# colour table (new)
# ---------------------------------------------------------------------------

COLOUR_TABLE = r"""# 本版新增：執色表（PREREG_03 1.1「執色一律取訓練資料的 color 欄，不從 game_id 推」；docs/DATA.md 第 3 條；plan 3.2 第 5 格）
# 鍵是分片的 __key__＝「等級_game_id」（(等級, game_id) 在訓練資料裡唯一），值是目標玩家是否執黑。
# 分片的每個 key 都要查得到（涵蓋率 100%）：loader 的 decode 查不到就停，訓練前另外把所有分片的 key 讀一遍核對
from c10data import read_split

t0 = time.time()
COLOR_IS_BLACK = {}
_colour_problems = []
for _part, _df in (('fit', df_train), ('inner', df_val)):
    for _r, _gid, _c in zip(_df['rank'], _df['game_id'], _df['color']):
        _key = '%s_%s' % (_r, _gid)
        if _c not in ('B', 'W'):
            _colour_problems.append('%s 的 color 是 %r' % (_key, _c))
        elif _key in COLOR_IS_BLACK:
            _colour_problems.append('%s 出現兩次' % _key)
        COLOR_IS_BLACK[_key] = _c == 'B'
FIT_KEY_SET = frozenset('%s_%s' % kv for kv in zip(df_train['rank'], df_train['game_id']))
INNER_KEY_SET = frozenset('%s_%s' % kv for kv in zip(df_val['rank'], df_val['game_id']))
N_FIT_BLACK = int((df_train['color'] == 'B').sum())
N_INNER_BLACK = int((df_val['color'] == 'B').sum())
FIT_BLACK_RATIO = N_FIT_BLACK / len(df_train)
_ratio_tol = 0.05 if SMOKE else 0.01   # SMOKE 的 fit 只有 2,000 盤，比例的抽樣誤差約 ±0.011
if len(COLOR_IS_BLACK) != len(df_train) + len(df_val) or len(FIT_KEY_SET) != len(df_train) or FIT_KEY_SET & INNER_KEY_SET:
    _colour_problems.append('鍵的數量不對：%d 個鍵，fit %d 盤、inner %d 盤' % (len(COLOR_IS_BLACK), len(df_train), len(df_val)))
if sum(COLOR_IS_BLACK[k] for k in FIT_KEY_SET) != N_FIT_BLACK:
    _colour_problems.append('執色表的 fit 執黑筆數和 df_train 算出的 %d 不同' % N_FIT_BLACK)
if not abs(FIT_BLACK_RATIO - 0.5) <= _ratio_tol:
    _colour_problems.append('fit 的執黑比例 %.4f 不在 0.50 ± %.2f' % (FIT_BLACK_RATIO, _ratio_tol))

# 外洩：fit、inner、本機驗證三組玩家兩兩不相交（load_rows 已各自 assert；這裡用 split 檔把三組一起再核一次）
_split = read_split(SPLIT_CSV)
_val_players = {p for p, info in _split.items() if info['split'] != 'train'}
_fit_players, _inner_players = set(df_train['player_id']), set(df_val['player_id'])
THREE_WAY = {'fit_players': len(_fit_players), 'inner_players': len(_inner_players), 'validation_players': len(_val_players),
             'fit_inner': len(_fit_players & _inner_players), 'fit_validation': len(_fit_players & _val_players),
             'inner_validation': len(_inner_players & _val_players),
             'unknown_players': len((_fit_players | _inner_players) - set(_split))}
if THREE_WAY['fit_inner'] or THREE_WAY['fit_validation'] or THREE_WAY['inner_validation'] or THREE_WAY['unknown_players']:
    _colour_problems.append('fit、inner、本機驗證的玩家有重疊或不在 split 裡：%s' % THREE_WAY)
_leak_log['runs'][-1]['three_way_disjoint'] = THREE_WAY
write_json(_leak_path, _leak_log)

COLOUR_CHECK = {'keys': len(COLOR_IS_BLACK), 'fit_games': len(df_train), 'inner_games': len(df_val),
                'fit_black': N_FIT_BLACK, 'fit_black_ratio': FIT_BLACK_RATIO, 'ratio_tolerance': _ratio_tol,
                'inner_black': N_INNER_BLACK, 'inner_black_ratio': N_INNER_BLACK / len(df_val),
                'three_way_disjoint': THREE_WAY, 'problems': _colour_problems[:20]}
save_config(colour_check=COLOUR_CHECK)
if _colour_problems:
    raise SystemExit('執色表或外洩檢查沒過（%d 項）：%s。還沒開始訓練。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                     % (len(_colour_problems), '；'.join(_colour_problems[:5])))
mark_time('colour_table', time.time() - t0)
print('執色表：%d 盤（fit 執黑 %d，比例 %.4f；inner 執黑 %d），三組玩家兩兩不相交'
      % (len(COLOR_IS_BLACK), N_FIT_BLACK, FIT_BLACK_RATIO, N_INNER_BLACK))
"""

# ---------------------------------------------------------------------------
# arm B sampling (new)
# ---------------------------------------------------------------------------

ARM_B = r"""# 本版新增：B 臂的取樣（PREREG_03 1.2；plan 3.4）。A 臂只印一行，每一輪用全部 fit 棋
# - E_L：每級「每位玩家最多算 CAP 盤」之後的總盤數，用 df_train 重算，必須和登記值完全相同
# - 盤 g（玩家 p、等級 L）收下的機率 q_g ＝ min(1, CAP / n_p) × TARGET / E_L，n_p 是這位玩家在 fit 的盤數
# - 第 pass 輪：u ＝ int(sha256("c10-prereg03|acc|{pass}|{rank}|{game_id}") 的前 16 個十六進位字, 16) / 2^64，u < q_g 就收下
#   （先看 key 決定收不收，不收的不解碼）。每級每一輪的期望盤數是 TARGET
CAP_REGISTERED = 200
E_L_REGISTERED = __E_L__
TARGET_PER_LEVEL_REGISTERED = __TARGET__


def level_totals(players, ranks, cap):
    # 回傳 ({等級: E_L}, {玩家: n_p})；一位玩家只能屬於一個等級
    n_p, rank_of = {}, {}
    for p, r in zip(players, ranks):
        n_p[p] = n_p.get(p, 0) + 1
        if rank_of.setdefault(p, r) != r:
            raise ValueError('玩家 %s 同時出現在 %s 和 %s' % (p, rank_of[p], r))
    totals = {lv: 0 for lv in LEVELS}
    for p, n in n_p.items():
        totals[rank_of[p]] += min(n, cap)
    return totals, n_p


def acceptance_q(n, e_l, target, cap):
    return min(1.0, cap / n) * target / e_l


def acceptance_u(pass_no, rank, game_id):
    h = hashlib.sha256(('c10-prereg03|acc|%d|%s|%s' % (pass_no, rank, game_id)).encode('utf-8')).hexdigest()
    return int(h[:16], 16) / 2.0 ** 64


def accepted_keys(pass_no, rows):
    # rows：[(key, rank, game_id, q, 執黑)]。回傳 (收下的 key, 每級收下的盤數, 收下的執黑盤數)
    keep, by_level, black = set(), {lv: 0 for lv in LEVELS}, 0
    for key, r, gid, q, is_b in rows:
        if acceptance_u(pass_no, r, gid) < q:
            keep.add(key)
            by_level[r] += 1
            black += is_b
    return frozenset(keep), by_level, black


t0 = time.time()
if ARM == 'B':
    CAP_USED = SMOKE_CFG_P03['cap'] if SMOKE else CAP_REGISTERED
    E_L, N_P = level_totals(df_train['player_id'].tolist(), df_train['rank'].tolist(), CAP_USED)
    TARGET_PER_LEVEL = min(E_L.values()) if SMOKE else TARGET_PER_LEVEL_REGISTERED
    if not SMOKE and (E_L != E_L_REGISTERED or min(E_L.values()) != TARGET_PER_LEVEL):
        raise SystemExit('B 臂的 E_L 和登記值不同：算出 %s，登記 %s。還沒開始訓練。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                         % (E_L, E_L_REGISTERED))
    B_ROWS = [('%s_%s' % (r, gid), r, gid, acceptance_q(N_P[p], E_L[r], TARGET_PER_LEVEL, CAP_USED), c == 'B')
              for p, r, gid, c in zip(df_train['player_id'], df_train['rank'], df_train['game_id'], df_train['color'])]
    _expect = {lv: 0.0 for lv in LEVELS}
    for _key, _r, _gid, _q, _b in B_ROWS:
        _expect[_r] += _q
    assert all(abs(v - TARGET_PER_LEVEL) < 1e-6 for v in _expect.values()), _expect   # 每級期望盤數＝TARGET
    SAMPLING = {'arm': 'B', 'cap': CAP_USED, 'E_L': E_L, 'E_L_registered': None if SMOKE else E_L_REGISTERED,
                'target_per_level': TARGET_PER_LEVEL, 'expected_per_pass': TARGET_PER_LEVEL * len(LEVELS),
                'players': len(N_P), 'players_capped': sum(1 for n in N_P.values() if n > CAP_USED),
                'rule': 'q = min(1, CAP/n_p) * TARGET / E_L[rank]; accept iff int(sha256("c10-prereg03|acc|{pass}|{rank}|'
                        '{game_id}")[:16], 16) / 2**64 < q; pass numbers start at 1'}
    print('B 臂：E_L %s（和登記值相同%s），每人上限 %d 盤，每級每一輪期望 %d 盤，一輪期望 %d 盤'
          % (E_L, '；SMOKE 不比對' if SMOKE else '', CAP_USED, TARGET_PER_LEVEL, TARGET_PER_LEVEL * len(LEVELS)))
else:
    CAP_USED, E_L, N_P, TARGET_PER_LEVEL, B_ROWS = None, None, None, None, None
    SAMPLING = {'arm': 'A', 'rule': 'every pass uses every fit game'}
    print('A 臂：每一輪用全部 fit 棋（%d 盤），不做 B 臂的取樣' % len(df_train))
save_config(sampling=SAMPLING)
mark_time('sampling', time.time() - t0)
""".replace("__E_L__", json.dumps(E_L_REGISTERED)).replace("__TARGET__", str(TARGET_PER_LEVEL))

# ---------------------------------------------------------------------------
# official 11 (rewrite): loaders
# ---------------------------------------------------------------------------

OFF11 = r"""# 本版（prereg03）改寫：loader（plan 3.3）。官方的 decode_sample／make_loader／pick_workers 拿掉
# - worker 傳 uint8 特徵（到 GPU 才轉型）、label、執色（查執色表）和 key
# - 訓練：每一輪重建 loader。分片先排序，再用 random.Random(42·1000＋pass) 洗；依 worker 切（islice(w, None, workers)）；
#   B 臂先依 key 篩選（不收的不解碼）；worker 內 shuffle buffer 1,000，rng 種子 (42·1000＋pass)·1000＋worker；
#   批次在 worker 內組成，最後一批不滿也算一步（同官方）。資料順序只由 (42, pass, worker) 決定
# - inner 監看和最後輸出：照分片順序、不打散
import random


class ColourKeyError(RuntimeError):
    pass


def shard_order(shards, pass_no):
    order = sorted(shards)
    random.Random(SEED * 1000 + pass_no).shuffle(order)
    return order


def worker_seed(pass_no, worker_id):
    return (SEED * 1000 + pass_no) * 1000 + worker_id


def seeded_shuffle(pass_no):
    # webdataset 1.0.2 的 shuffle（buffer 1,000、initial 100），rng 在 worker 裡依 (pass, worker) 建立；
    # worker 編號和 split_by_worker 用同一個來源（wds.utils.pytorch_worker_info）
    def stage(src):
        rng = random.Random(worker_seed(pass_no, wds.utils.pytorch_worker_info()[2]))
        return wds.shuffle(SHUFFLE_BUFFER, rng=rng)(src)
    return stage


def decode_sample_p03(sample):
    key = sample['__key__']
    feat = sample['features.npy']
    label = int(sample['rank_idx.txt'])
    if key not in COLOR_IS_BLACK:
        raise ColourKeyError('C10_COLOUR_MISSING %s' % key)
    if key.split('_', 1)[0] != LEVELS[label] or feat.dtype != np.uint8 or feat.shape != (500, 19, 19):
        raise ColourKeyError('C10_KEY_MISMATCH %s（label %d，%s %s）' % (key, label, feat.dtype, feat.shape))
    return (torch.from_numpy(feat), torch.tensor(label, dtype=torch.long),
            torch.tensor(COLOR_IS_BLACK[key], dtype=torch.bool), key)


def _accept(accepted):
    def pred(sample):
        key = sample['__key__']
        if key not in COLOR_IS_BLACK:
            raise ColourKeyError('C10_COLOUR_MISSING %s' % key)
        return key in accepted
    return pred


def make_train_loader(pass_no, accepted=None):
    stages = [wds.SimpleShardList(shard_order(train_shards, pass_no)), wds.split_by_worker, wds.tarfile_to_samples()]
    if accepted is not None:
        stages.append(wds.select(_accept(accepted)))   # B 臂：先看 key 決定收不收，不收的不解碼
    stages += [seeded_shuffle(pass_no), wds.decode(), wds.map(decode_sample_p03), wds.batched(BATCH_SIZE, partial=True)]
    return DataLoader(wds.DataPipeline(*stages), batch_size=None, num_workers=LOADER_WORKERS, pin_memory=True)


def make_eval_loader(shards, persistent):
    ds = wds.DataPipeline(wds.SimpleShardList(sorted(shards)), wds.split_by_worker, wds.tarfile_to_samples(),
                          wds.decode(), wds.map(decode_sample_p03), wds.batched(BATCH_SIZE, partial=True))
    return DataLoader(ds, batch_size=None, num_workers=LOADER_WORKERS, pin_memory=True,
                      persistent_workers=persistent and LOADER_WORKERS > 0)


def _sample_key(sample):
    return (sample['__key__'], sample['__url__'])   # tuple：webdataset 的 batched 才會把它收成兩個 list


def _shard_name(url):
    return url.replace('\\', '/').rsplit('/', 1)[-1]


def scan_keys(shards):
    # 只讀分片裡每一盤的 key（解壓但不解碼特徵）。回傳 (照 worker 的順序串起來的 key, {分片檔名: 分片裡 key 的順序})。
    # 一個分片只由一個 worker 從頭讀到尾，所以每個分片裡的順序和 worker 數無關
    ds = wds.DataPipeline(wds.SimpleShardList(sorted(shards)), wds.split_by_worker, wds.tarfile_to_samples(),
                          wds.map(_sample_key), wds.batched(4096, partial=True))
    keys, by_shard = [], {}
    for batch in DataLoader(ds, batch_size=None, num_workers=LOADER_WORKERS):
        keys.extend(batch[0])
        for k, u in zip(batch[0], batch[1]):
            by_shard.setdefault(_shard_name(u), []).append(k)
    return keys, by_shard


def shard_layout_sha256(shards, by_shard):
    # 分片排列：依檔名排序的每個分片（沒有 key 的分片也算，它也占 shard_order 的一個位置），和它裡面 key 的順序。
    # 資料順序只由這個和 (42, pass, worker) 決定；和分片檔的壓縮位元組（tar 裡的時間戳）無關，所以同一種切法重抽也相同
    names = sorted(_shard_name(s) for s in shards)
    assert len(set(names)) == len(names) and set(by_shard) <= set(names), 'scan_keys 讀到的分片和清單不同'
    h = hashlib.sha256()
    for name in names:
        h.update(('%s\n%s\n\n' % (name, '\n'.join(by_shard.get(name, [])))).encode('utf-8'))
    return h.hexdigest()


def iterate(loader, what):
    # 讀 loader；worker 裡的執色或 key 錯誤轉成中文說明後停下（其他錯誤照原樣拋出）
    try:
        yield from loader
    except RuntimeError as e:
        if 'C10_COLOUR_MISSING' in str(e) or 'C10_KEY_MISMATCH' in str(e):
            raise SystemExit('%s：分片裡有一盤查不到執色表，或 key 和等級對不上（%s）。這表示分片和這次的 fit／inner 資料對不上。'
                             '處理方式：不要再執行，把這段訊息截圖給 Claude。' % (what, str(e).strip().splitlines()[-1][:300])) from e
        raise
"""

# ---------------------------------------------------------------------------
# official 12 (rewrite): shard lists, shard-key check, inner loader
# ---------------------------------------------------------------------------

OFF12 = r"""# 本版（prereg03）改寫：分片清單和訓練前的分片 key 核對。官方先試 4 個 worker、對不上就退回 0；本版的 worker 數是 recipe 設定
# （G4：A 16、B 32；A100：10），不退回。訓練、inner 分片的 key 各讀一遍（只讀 key、不解碼特徵）：數量等於實際寫入的盤數、
# 不重複、每盤都查得到執色（涵蓋率 100%）、都屬於這一組；沒有解析失敗時和 fit／inner 的盤完全相同、執黑盤數相同。
# 另記分片排列的 SHA-256（訓練分片的那一個寫進 recipe 設定：接續時分片排列不同，資料順序就和沒中斷時不同，所以不接續）
train_shards = sorted(glob.glob(f'{FEAT_DIR}/train/*.tar.gz'))
val_shards = sorted(glob.glob(f'{FEAT_DIR}/val/*.tar.gz'))
print('分片：train %d / val %d' % (len(train_shards), len(val_shards)))
if not train_shards or not val_shards:
    raise SystemExit('找不到分片（train %d、val %d 個）。請把這段訊息截圖給 Claude。' % (len(train_shards), len(val_shards)))

t0 = time.time()
SHARD_CHECK, _shard_problems = {}, []
for _tag, _shards, _counts, _want, _black_want in (('train', train_shards, train_counts, FIT_KEY_SET, N_FIT_BLACK),
                                                   ('val', val_shards, val_counts, INNER_KEY_SET, N_INNER_BLACK)):
    _keys, _by_shard = scan_keys(_shards)
    _ks = set(_keys)
    _rec = {'shards': len(_shards), 'keys': len(_keys), 'distinct': len(_ks), 'written': _counts['written'],
            'bad': _counts['bad'], 'missing_colour': sum(1 for k in _ks if k not in COLOR_IS_BLACK),
            'not_in_part': len(_ks - _want), 'part_games_not_in_shards': len(_want - _ks),
            'black': sum(COLOR_IS_BLACK.get(k, False) for k in _keys), 'black_expected': _black_want,
            'layout_sha256': shard_layout_sha256(_shards, _by_shard), 'shards_with_keys': len(_by_shard)}
    if _rec['keys'] != _rec['written'] or _rec['distinct'] != _rec['keys']:
        _shard_problems.append('%s：%d 個 key（%d 個不重複），實際寫入 %d 盤' % (_tag, _rec['keys'], _rec['distinct'], _rec['written']))
    if _rec['missing_colour'] or _rec['not_in_part']:
        _shard_problems.append('%s：%d 盤查不到執色，%d 盤不屬於這一組' % (_tag, _rec['missing_colour'], _rec['not_in_part']))
    if _counts['bad'] == 0 and (_rec['part_games_not_in_shards'] or _rec['black'] != _black_want):
        _shard_problems.append('%s：%d 盤不在分片裡，執黑 %d 盤（應為 %d）'
                               % (_tag, _rec['part_games_not_in_shards'], _rec['black'], _black_want))
    SHARD_CHECK[_tag] = _rec
    if _tag == 'train':
        WRITTEN_FIT = frozenset(_ks)   # 每一輪的盤數核對用
    del _keys, _ks, _by_shard
save_config(shard_key_check=SHARD_CHECK)
if _shard_problems:
    raise SystemExit('分片的 key 核對沒過：%s。分片和這次的 fit／inner 資料或執色表對不上，還沒開始訓練。'
                     '處理方式：不要再執行，把這段訊息截圖給 Claude。' % '；'.join(_shard_problems))
mark_time('shard_key_check', time.time() - t0)
print('分片的 key 核對通過（%.0f 秒）：%s' % (time.time() - t0, {k: (v['keys'], v['black']) for k, v in SHARD_CHECK.items()}))

# inner 監看和最後輸出用的 loader（照分片順序、不打散）；另外用一個臨時的 loader 印一個 batch
val_loader = make_eval_loader(val_shards, persistent=True)
_x, _y, _b, _k = next(iter(make_eval_loader(val_shards, persistent=False)))
print('一個 batch:', tuple(_x.shape), _x.dtype, _y[:8].tolist(), '執黑 %d/%d' % (int(_b.sum()), len(_k)), _k[0])
del _x, _y, _b, _k
"""

# ---------------------------------------------------------------------------
# official 14 (base + rep), official 15 (rewrite)
# ---------------------------------------------------------------------------

_off14 = base_src("official_14")
_off14 = rep(_off14, """model = GoRankResNet(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
""", """# 本版（prereg03）：輸入 501 層，其他和官方相同；建模型前重設種子（兩臂同一個初始化）；channels_last；AdamW 明寫 weight decay
def state_dict_sha256(net):
    d = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        t = v.detach().cpu().contiguous()
        d.update(('%s|%s|%s|' % (k, t.dtype, tuple(t.shape))).encode('utf-8'))
        d.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.view(torch.int16).numpy().tobytes())
    return d.hexdigest()


def _nchw_output(module, inputs, output):
    return output.contiguous()


def channels_last(net):
    # channels_last：卷積用 NHWC 排列（bf16 較快）。官方 forward 在 bn_in2 之後用 view 攤平，channels_last 的張量不能 view，
    # 所以在 bn_in2 的輸出轉回一般（NCHW）排列：只改記憶體排列，數值和攤平的順序都和官方相同；network.py 不改
    net.to(memory_format=torch.channels_last)
    net.bn_in2.register_forward_hook(_nchw_output)
    return net


torch.manual_seed(SEED)
model = GoRankResNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
INIT_WEIGHTS_SHA256 = state_dict_sha256(model)   # 兩臂必須相同（同一個 torch 版本時）
model = channels_last(model.to(DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
""")
_off14 = rep(_off14, "print('batch %d | lr %s | epochs %d' % (BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS))",
             "print('batch %d | lr %s | weight decay %s | T %d | warmup %d | 輸入 %d 層 | 初始權重 %s…'\n"
             "      % (BATCH_SIZE, PEAK_LR, WEIGHT_DECAY, T_TOTAL, WARMUP_STEPS, IN_CHANNELS, INIT_WEIGHTS_SHA256[:12]))\n"
             "save_config(model={'class': 'GoRankResNet', 'in_channels': IN_CHANNELS, 'num_classes': NUM_CLASSES,\n"
             "                   'parameters': sum(p.numel() for p in model.parameters()),\n"
             "                   'init_weights_sha256': INIT_WEIGHTS_SHA256,\n"
             "                   'memory_format': 'channels_last; bn_in2 output hooked back to NCHW before the official view()'})")

OFF15 = r"""def validate(model, loader, criterion):
    # 本版（prereg03）改寫：inner 監看（只記錄）。原方向、fp32（不開 autocast）、目標玩家視角（執色查執色表）
    model.eval()
    loss_sum, correct, total, batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for x, labels, is_black, _keys in iterate(loader, 'inner 監看'):
            inputs = model_input(x.to(DEVICE, non_blocking=True), is_black.to(DEVICE, non_blocking=True))
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss_sum += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batches += 1
    model.train()
    return (loss_sum / batches if batches else 0.0,
            100 * correct / total if total else 0.0)
"""

# ---------------------------------------------------------------------------
# dump tools (base + rep)
# ---------------------------------------------------------------------------

DT_NEW_HEAD = """# 本版新增：逐盤輸出（dump）工具（prereg03：沿用 prereg02 的工具；d4() 搬到視角與 D4 的格子）
# - inner：對內部驗證的分片跑一遍（val_loader，不打散），用分片的 __key__＝「等級_game_id」對回 df_val 的列
# - 本機驗證題：照題目檔的列順序和 slot 1..num_games 解析，失敗的保留位置（logits 填 NaN）；每題一個 batch（同 prereg02）
# - 本版：一律 fp32、先做視角轉換再做 D4；8 種對稱（<名稱>_d4）和「原方向、執色對調」（<名稱>_s0_swap）一起算
# - 每個檔先寫 .tmp 再改名，附 _DONE_<檔名>.json；重跑時完成標記、SHA-256、權重 SHA-256 都對得上就跳過
"""

DT_NEW_RECORD = """def record_readonly_after():
    # 本版：三個只讀目錄（prereg01、prereg02、test02）的指紋（結尾，或停下時）：和這次開頭記的比，寫進 readonly_after.json；
    # 回傳 (指紋, {目錄: 不同的檔})
    after = {name: readonly_fingerprint(root) for name, root in READONLY_ROOTS.items()}
    path = os.path.join(RESULT_DIR, 'readonly_after.json')
    log = read_runs(path)
    changed = {name: sorted(k for k in set(READONLY_BEFORE[name]) | set(after[name])
                            if READONLY_BEFORE[name].get(k) != after[name].get(k)) for name in after}
    changed = {k: v for k, v in changed.items() if v}
    log['runs'].append({'run': RUN['run'], 'time': now_iso(), 'roots': READONLY_ROOTS, 'files': after,
                        'n_files': {k: len(v) for k, v in after.items()},
                        'same_as_before': not changed, 'changed_files': changed})
    write_json(path, log)
    return after, changed


"""

DT_NEW_DUMP_INNER = r"""def dump_inner_p03(net, name, weights, weights_sha):
    # 本版：inner 的 8 種對稱（inner_<名稱>_d4，[盤數, 8, 10]）和原方向、執色對調（inner_<名稱>_s0_swap，[盤數, 10]），讀一次分片。
    # 執色用 INNER_KEYS 的 color 欄（df_val 的 color），並和執色表（loader 傳來的）核對
    stem, stem_sw, n = 'inner_%s_d4' % name, 'inner_%s_s0_swap' % name, len(df_val)
    if dump_is_done(stem, n, NUM_SYMS, weights_sha) and dump_is_done(stem_sw, n, 1, weights_sha):
        print('  %s、%s：已完成，跳過' % (stem, stem_sw))
        return 'skipped'
    require_gpu_for(stem)
    t0 = time.time()
    logits = np.full((n, NUM_SYMS, NUM_CLASSES), np.nan, dtype=np.float32)
    swap = np.full((n, NUM_CLASSES), np.nan, dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    count = 0
    with torch.no_grad():
        for x, y, is_black_table, keys in iterate(KEY_LOADER, stem):
            idx = []
            for lab, k in zip(y.tolist(), keys):
                r, g = k.split('_', 1)
                if r != LEVELS[lab]:
                    raise SystemExit('分片的 __key__ %r 和等級 %s 對不上。請截圖給 Claude。' % (k, LEVELS[lab]))
                i = INNER_INDEX.get((r, g))
                if i is None:
                    raise SystemExit('分片裡的 %s 不在這次的內部驗證（df_val）裡。請截圖給 Claude。' % k)
                idx.append(i)
            idx = np.asarray(idx, dtype=np.int64)
            if len(set(idx.tolist())) != len(idx) or seen[idx].any():
                raise SystemExit('內部驗證有盤出現兩次（%s）。請截圖給 Claude。' % stem)
            is_b = INNER_KEYS['color'][idx] == 'B'
            if not np.array_equal(is_b, is_black_table.numpy()):
                raise SystemExit('INNER_KEYS 的 color 和執色表不同（%s）。請截圖給 Claude。' % stem)
            seen[idx] = True
            count += len(idx)
            xb = x.to(DEVICE, non_blocking=True)
            ib = torch.from_numpy(is_b).to(DEVICE)
            for s in range(NUM_SYMS):
                logits[idx, s] = net(dump_input(xb, ib, s)).float().cpu().numpy()
            swap[idx] = net(dump_input(xb, ~ib, 0)).float().cpu().numpy()
    if count != n or not seen.all():
        raise SystemExit('%s：讀到 %d 盤，內部驗證是 %d 盤，有盤沒讀到。請截圖給 Claude。' % (stem, count, n))
    secs = time.time() - t0
    write_dump(stem, n, NUM_SYMS, weights, weights_sha, secs,
               {'logits': logits, 'weights_sha256': np.array(weights_sha), 'syms': np.array(NUM_SYMS)})
    write_dump(stem_sw, n, 1, weights, weights_sha, secs,
               {'logits': swap, 'weights_sha256': np.array(weights_sha), 'view': np.array('s0_colour_swapped')})
    print('  %s：%d 盤 × %d 種方向＋執色對調，%.0f 秒' % (stem, n, NUM_SYMS, secs), flush=True)
    return 'written'


"""

DT_NEW_DUMP_VAL = r"""def dump_val_p03(net, name, weights, weights_sha):
    # 本版：本機驗證題的 8 種對稱（val_<名稱>_d4）和原方向、執色對調（val_<名稱>_s0_swap）。執色用題目檔的 color_i；
    # 每題一個 batch（每批最多 GPU_BATCH 盤，同 prereg02），解析失敗的 slot 整列 NaN
    V = parse_val()
    stem, stem_sw, n = 'val_%s_d4' % name, 'val_%s_s0_swap' % name, len(V['feats'])
    if dump_is_done(stem, n, NUM_SYMS, weights_sha) and dump_is_done(stem_sw, n, 1, weights_sha):
        print('  %s、%s：已完成，跳過' % (stem, stem_sw))
        return 'skipped'
    require_gpu_for(stem)
    t0 = time.time()
    logits = np.full((n, NUM_SYMS, NUM_CLASSES), np.nan, dtype=np.float32)
    swap = np.full((n, NUM_CLASSES), np.nan, dtype=np.float32)
    colour_b = V['keys']['color'] == 'B'
    with torch.no_grad():
        for _qid, start, k in V['spans']:
            ok = [start + j for j in range(k) if V['feats'][start + j] is not None]
            if not ok:
                continue
            x = torch.from_numpy(np.stack([V['feats'][g] for g in ok]))   # uint8，到 GPU 才轉型
            for i in range(0, len(x), GPU_BATCH):
                rows = ok[i:i + GPU_BATCH]
                xb = x[i:i + GPU_BATCH].to(DEVICE)
                ib = torch.from_numpy(colour_b[rows]).to(DEVICE)
                for s in range(NUM_SYMS):
                    logits[rows, s] = net(dump_input(xb, ib, s)).float().cpu().numpy()
                swap[rows] = net(dump_input(xb, ~ib, 0)).float().cpu().numpy()
    secs = time.time() - t0
    write_dump(stem, n, NUM_SYMS, weights, weights_sha, secs,
               {'logits': logits, 'weights_sha256': np.array(weights_sha), 'syms': np.array(NUM_SYMS)})
    write_dump(stem_sw, n, 1, weights, weights_sha, secs,
               {'logits': swap, 'weights_sha256': np.array(weights_sha), 'view': np.array('s0_colour_swapped')})
    print('  %s：%d 盤 × %d 種方向＋執色對調，%.0f 秒' % (stem, n, NUM_SYMS, secs), flush=True)
    return 'written'


"""

_dt = _dt_base
_dt = rep(_dt, cut(_dt, "# 本版新增：逐盤輸出（dump）工具\n", "import io\n"), DT_NEW_HEAD)
_dt = rep(_dt, "GPU_BATCH = 64                         # 和 prereg01 推論相同\nNUM_SYMS = 8\n",
          "GPU_BATCH = 64                         # 和 prereg01 推論相同（NUM_SYMS 在視角與 D4 的格子）\n")
_dt = rep(_dt, D4_SEGMENT, "# 本版：d4() 和它的自我檢查搬到視角與 D4 的格子（一字不改）")
_dt = rep(_dt, "    net = GoRankResNet(num_classes=NUM_CLASSES)\n",
          "    net = GoRankResNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)   # 本版：501 層\n")
_dt = rep(_dt, "    net.eval().to(DEVICE)\n", "    channels_last(net.eval().to(DEVICE))   # 本版：channels_last（和訓練時相同）\n")
_dt = rep(_dt, "    # 正式執行不在 CPU 上算逐盤輸出（要好幾天）；收尾開關打開又沒有 GPU 時，缺的 dump 在這裡停下\n",
          "    # 正式執行不在 CPU 上算逐盤輸出（要好幾天）\n")
_dt = rep(_dt, cut(_dt, "def record_p01_after():", "def pack_results(zip_name):"), DT_NEW_RECORD)
_dt = rep(_dt, "    # 結果目錄裡除了權重和暫存檔以外的檔案，打成一個 zip 放在結果目錄旁邊（prereg02/<zip_name>）；回傳 (路徑, 跳過的大檔)\n",
          "    # 結果目錄裡除了權重和暫存檔以外的檔案，打成一個 zip 放在結果目錄旁邊（prereg03/<zip_name>）；回傳 (路徑, 跳過的大檔)\n")
_dt = rep(_dt, cut(_dt, "def decode_with_key(sample):", "def dump_inner(net, name, syms, weights, weights_sha):"),
          "# 本版：inner 的 loader 用 val_loader（make_eval_loader：照分片順序，已含 __key__ 和執色）\n"
          "KEY_LOADER = val_loader\n\n\n")
_dt = rep(_dt, cut(_dt, "def dump_inner(net, name, syms, weights, weights_sha):", "_infer_parser = SGFParseRankPrediction()"),
          DT_NEW_DUMP_INNER)
_dt = rep(_dt, cut(_dt, "def dump_val(net, name, syms, weights, weights_sha):", "print('dump 工具就緒"),
          DT_NEW_DUMP_VAL)
_dt = rep(_dt, "print('dump 工具就緒：inner %d 盤（worker %d）' % (len(df_val), workers))",
          "print('dump 工具就緒：inner %d 盤（worker %d）' % (len(df_val), LOADER_WORKERS))")
assert "rule_b" not in _dt and "decode_with_key" not in _dt and "P01_" not in _dt and "def d4(" not in _dt

# ---------------------------------------------------------------------------
# key files (new)
# ---------------------------------------------------------------------------

KEYS_CELL = r"""# 本版新增：key 檔（inner_keys、val_keys；plan 3.7）。內容必須和 prereg02 的 key 檔逐陣列相同：正式執行比對產生器從 repo
# 讀出的內容摘要，SMOKE 比對 prereg02_smoke 的檔（只讀）。不同就在訓練前停下（本機合併時各盤會對錯）
t0 = time.time()
KEY_DIGEST_PIN = __KEY_DIGEST__


def keys_digest(arrays):
    # 每個陣列依名稱排序：「名稱|dtype|形狀|」加上陣列的位元組
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(('%s|%s|%s|' % (k, a.dtype.str, a.shape)).encode('utf-8'))
        h.update(a.tobytes())
    return h.hexdigest()


KEYS_CHECK = {}
for _stem, _arrays in (('inner_keys', INNER_KEYS), ('val_keys', parse_val()['keys'])):
    _got = keys_digest(_arrays)
    if SMOKE:
        _ref = os.path.join(P02_DUMPS, _stem + '.npz')
        if os.path.exists(_ref):
            with np.load(_ref, allow_pickle=False) as _z:
                _want, _src = keys_digest({k: _z[k] for k in _z.files}), 'prereg02_smoke/rank/dumps/%s.npz' % _stem
        else:
            _want, _src = None, '（SMOKE：沒有 prereg02_smoke 的 key 檔，不比對）'
    else:
        _want, _src = KEY_DIGEST_PIN[_stem], 'results/prereg02/rank/dumps/%s.npz（產生器讀出）' % _stem
    KEYS_CHECK[_stem] = {'digest': _got, 'reference_digest': _want, 'reference': _src,
                         'rows': len(next(iter(_arrays.values()))), 'same': _want is None or _got == _want}
save_config(keys_check=KEYS_CHECK)
if not all(v['same'] for v in KEYS_CHECK.values()):
    raise SystemExit('key 檔和 prereg02 的不同（%s），本機合併時各盤會對錯。還沒開始訓練。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                     % [k for k, v in KEYS_CHECK.items() if not v['same']])
write_keys('inner_keys', INNER_KEYS)
write_keys('val_keys', parse_val()['keys'])
mark_time('keys', time.time() - t0)
print('key 檔和 prereg02 相同：%s' % {k: v['rows'] for k, v in KEYS_CHECK.items()})
""".replace("__KEY_DIGEST__", json.dumps(KEY_DIGEST, indent=4))

# ---------------------------------------------------------------------------
# training (new)
# ---------------------------------------------------------------------------

TRAIN = r"""# 本版新增：從零訓練（PREREG_03 1.1–1.3；plan 3.3–3.6）。以 global step 計，到 T 步就停（不管是不是一輪的中間）
# - 每一輪（pass，從 1 起算）：先查 Drive 剩餘空間；random／numpy／torch 的種子設為 42·1000＋pass，D4 的 generator 種子設為 42＋pass；
#   B 臂先算這一輪收下的盤；建 loader → 訓練 → 核對這一輪的盤和學習率 → inner 監看（只記錄）→ 存 pass_N.pth（本機）、state.pt、history.csv
# - 第 T 步：存 final.pth（指定的 checkpoint，不挑 best）和 state.pt
# - 接續：有 state.pt 就從下一輪的開頭重跑（資料順序、D4 的抽法和沒中斷時相同）；recipe 設定（含訓練分片排列）不同就停
# - 停止（登記 3.5）：第 300 步推估的訓練時間超過上限、loss 非有限值、這一輪讀到的盤或學習率和登記的不同
#   → 留下 STOPPED.json、打包 *_results_failed.zip、送出 Drive 的寫入，這一臂不算結果
# - 不是登記的停止：一輪開始前 Drive 剩不到 MIN_PASS_FREE_BYTES（3 GB，和 preflight 的門檻相同）→ 狀態記 drive_space_pause、
#   送出 Drive 的寫入後停下；清到 3 GB 以上後從這一輪的開頭接續
#   （Colab 刪掉或覆寫的檔會進 Drive 的垃圾桶、照樣占空間；每一輪覆寫的 state.pt 約 0.32 GB）
from torch.optim.lr_scheduler import LambdaLR

STATE_PATH = os.path.join(CKPT_DIR, 'state.pt')
FINAL_PATH = os.path.join(CKPT_DIR, 'final.pth')
HISTORY_PATH = os.path.join(CKPT_DIR, 'history.csv')
# 每一輪的 pass_N.pth 只存在 Colab 的本機磁碟（接續只用 state.pt、最後輸出只用 final.pth；不寫進 Drive，免得刪掉的檔塞滿垃圾桶）
PASS_CKPT_DIR = guard_write(os.path.join(WORK, 'prereg03_pass_ckpt_' + ARM_TAG + ('_smoke' if SMOKE else '')))
os.makedirs(PASS_CKPT_DIR, exist_ok=True)
# 一輪開始前 Drive 至少要剩 MIN_PASS_FREE_BYTES（preflight 格定義，3 GB：state.pt 的暫存檔、final.pth、逐盤輸出、zip，兩臂同時寫也夠）
RESULTS_ZIP_NAME = 'prereg03_rank_%s_results.zip' % ARM_TAG
FAILED_ZIP_NAME = 'prereg03_rank_%s_results_failed.zip' % ARM_TAG
STATE_RESTART = ('在 Drive 把 %s 整個資料夾改名保留（例如 ckpt_broken_日期），再按一次「全部執行」：會從第 1 步重新訓練。'
                 '先把這件事告訴 Claude。不要只刪 state.pt。' % CKPT_DIR)


def lr_lambda(t):
    # t＝已做的 optimizer.step() 次數：前 WARMUP_STEPS 步線性升到 1，之後 cosine 降到 0（第 T 步是 0）
    if t < WARMUP_STEPS:
        return (t + 1) / WARMUP_STEPS
    return 0.5 * (1.0 + math.cos(math.pi * min(t - WARMUP_STEPS, T_TOTAL - WARMUP_STEPS) / (T_TOTAL - WARMUP_STEPS)))


def expected_lr(t):
    return PEAK_LR * lr_lambda(t)


RECIPE = {
    'format': 'c10-prereg03 recipe v1', 'ARM': ARM, 'SEED': SEED, 'IN_CHANNELS': IN_CHANNELS, 'BATCH_SIZE': BATCH_SIZE,
    'T': T_TOTAL, 'WARMUP': WARMUP_STEPS, 'LR': PEAK_LR, 'WD': WEIGHT_DECAY,
    'optimizer': 'AdamW(betas=(0.9, 0.999), eps=1e-8)', 'loss': 'CrossEntropyLoss (no label smoothing)',
    'schedule': 'LambdaLR: lambda(t) = (t+1)/WARMUP if t < WARMUP else 0.5*(1+cos(pi*(t-WARMUP)/(T-WARMUP))); '
                't = optimizer steps done; scheduler.step() after every optimizer.step()',
    'AMP': 'bf16 autocast for the training forward/loss only, no GradScaler' if AMP else 'off',
    'channels_last': True, 'TF32': 'torch defaults (as prereg02)',
    'augment': 'D4 in training only: s ~ U{0..7} per sample, torch.randint on DEVICE with a generator seeded SEED+pass; '
               '[8, 361] permutation gather; colour plane invariant',
    'data_order': 'shards sorted, then random.Random(SEED*1000+pass).shuffle; split_by_worker (islice w::workers); '
                  'per-worker shuffle buffer %d (initial 100) with random.Random((SEED*1000+pass)*1000+worker); '
                  'batched in the worker, partial last batch is a step; pass numbers start at 1' % SHUFFLE_BUFFER,
    'pass_seeds': 'random / numpy / torch seeded SEED*1000+pass at every pass start',
    'CAP': CAP_USED, 'E_L': E_L, 'TARGET_PER_LEVEL': TARGET_PER_LEVEL,
    'acceptance': SAMPLING['rule'], 'WORKERS': LOADER_WORKERS, 'GPU_CLASS': GPU_CLASS,
    'features_sha256': json_sha256(FEAT_FINGERPRINT), 'fit_games': len(df_train), 'train_written': train_counts['written'],
    'train_shard_layout_sha256': SHARD_CHECK['train']['layout_sha256'],
    'smoke_cfg_p03': SMOKE_CFG_P03 if SMOKE else None,
}
RECIPE_GPU_KEYS = {'GPU_CLASS', 'WORKERS', 'train_shard_layout_sha256'}   # 換了 GPU 種類時會一起變的項目（B 臂重抽時的分片排列）


HISTORY_COLS_CHECKED = ('pass', 'pass_complete', 'steps', 'samples', 'black', 'global_step', 'run', 'order_sha256', 'd4_sha256')


def save_history():
    if history:
        pd.DataFrame(history).to_csv(guard_write(HISTORY_PATH), index=False)


def history_file_problems():
    # 收尾用：Drive 上的 ckpt/history.csv 要和記憶體裡的 history（這次訓練的，或從 state.pt 讀回的）逐列相同，而且記到第 T 步。
    # 最後一輪存好 state.pt 之後、寫完 history.csv 之前斷線的話，接續時 state.pt 已記訓練完成，訓練格不會再寫 history.csv
    if not os.path.exists(HISTORY_PATH):
        return ['ckpt/history.csv 不見了']
    try:
        got = pd.read_csv(HISTORY_PATH, dtype=str, keep_default_na=False)
    except Exception as e:
        return ['ckpt/history.csv 讀不進來（%s: %s）' % (type(e).__name__, e)]
    want = pd.DataFrame(history)
    if list(got.columns) != list(want.columns) or len(got) != len(want):
        return ['ckpt/history.csv 是 %d 列 %d 欄，記憶體裡的 history 是 %d 列 %d 欄'
                % (len(got), len(got.columns), len(want), len(want.columns))]
    probs = ['ckpt/history.csv 的 %s 欄和記憶體裡的 history 不同' % c for c in HISTORY_COLS_CHECKED
             if got[c].tolist() != want[c].astype(str).tolist()]
    if not len(got) or got['global_step'].tolist()[-1] != str(T_TOTAL):
        probs.append('ckpt/history.csv 沒有記到第 %d 步' % T_TOTAL)
    return probs


def save_state(pass_done, complete):
    # 先寫暫存檔再改名；這一輪（或 final.pth）存好之後才存 state.pt
    tmp = guard_write(STATE_PATH + '.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'global_step': global_step, 'pass': pass_done, 'complete': complete, 'history': history,
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else None,
                        'd4': D4_GEN.get_state()},
                'recipe': RECIPE, 'final_sha256': FINAL_SHA, 'init_weights_sha256': INIT_WEIGHTS_SHA256,
                'gpu_name': GPU_MODEL_NOW, 'run': RUN['run']}, tmp)
    os.replace(tmp, guard_write(STATE_PATH))


def drive_free_bytes():
    try:
        return shutil.disk_usage(DRIVE_DIR).free
    except OSError:
        return None


def drive_space_pause(pass_no, free):
    # 不是登記的停止：這一輪還沒開始（上一輪的 state.pt 已存好），不寫 STOPPED.json。記下狀態、送出 Drive 的寫入後停下
    RUN.update(end=now_iso(), status='drive_space_pause', drive_free_bytes=free, paused_before_pass=pass_no)
    run_log['status'] = 'drive_space_pause'
    save_run_log()
    flush_drive()
    raise SystemExit('Drive 只剩 %.2f GB，第 %d 輪還沒開始就先停下（每一輪開始前要剩 %s 以上）。這是暫停，不是登記的停止，結果照樣算；'
                     '這時沒有 zip，不用刪執行階段。原因多半是垃圾桶：Colab 每一輪覆寫的 state.pt（約 0.32 GB）會進 Drive 的垃圾桶，照樣占空間。'
                     '處理方式：到 Drive 左邊的「垃圾桶」，把名稱是 state.pt 的項目選起來 →「永久刪除」（垃圾桶裡沒有要留的東西時，'
                     '也可以直接「清空垃圾桶」）；等 Drive 剩 %s 以上（左下角「共 … GB」減「已使用 … GB」，可能要幾分鐘才更新），'
                     '再按一次「全部執行」：會從第 %d 輪的開頭接續，前面的訓練不會重來。垃圾桶清空了還不到 %s：C10 裡的東西都不要刪，'
                     '照 Claude 給的步驟（docs/COLAB_RUN.md「Drive 空間不夠時」）處理。另外把這段訊息截圖給 Claude。'
                     % (free / 1e9, pass_no, gb_text(MIN_PASS_FREE_BYTES), gb_text(MIN_PASS_FREE_BYTES), pass_no,
                        gb_text(MIN_PASS_FREE_BYTES)))


def stop_arm(status, message, detail):
    # 登記的停止條件：這一臂不算結果。留下 STOPPED.json（之後再「全部執行」會在 preflight 直接停下），
    # 比對只讀目錄的指紋、打包 *_results_failed.zip（查原因要用）、送出 Drive 的寫入，最後停下
    rec = {'status': status, 'detail': detail, 'run': RUN['run'], 'time': now_iso(), 'arm': ARM,
           'global_step': globals().get('global_step')}
    write_json(STOPPED_PATH, rec)
    save_history()
    _after, _changed = record_readonly_after()
    RUN.update(end=now_iso(), status=status)
    run_log['status'] = status
    save_run_log()
    save_config(stopped=rec, end=RUN['end'])
    save_config_summary(stopped=rec)
    _zip, _ = pack_results(FAILED_ZIP_NAME)
    if _changed:
        print('另外：只讀目錄在這次執行中被改動：%s' % {k: v[:5] for k, v in _changed.items()})
    flush_drive()
    raise SystemExit('%s。照 PREREG_03，這一臂不算結果（狀態 %s）。處理方式：不要再執行；下載 %s，把這段訊息截圖，一起給 Claude。'
                     % (message, status, _zip))


# ---- 建立 scheduler、D4 的 generator；有 state.pt 就接續 ----
t0 = time.time()
scheduler = LambdaLR(optimizer, lr_lambda)
D4_GEN = torch.Generator(device=DEVICE)
history, global_step, start_pass, FINAL_SHA, TRAIN_DONE = [], 0, 1, None, False
RUN['loaded_state'] = False
if os.path.exists(STATE_PATH):
    try:
        _state = torch.load(STATE_PATH, map_location='cpu', weights_only=False)   # 自己存的檔（含 history、RNG 狀態）
        _missing = [k for k in ('model', 'optimizer', 'scheduler', 'global_step', 'pass', 'complete', 'history', 'recipe')
                    if k not in _state]
    except Exception as e:
        raise SystemExit('ckpt/state.pt 讀不進來（%s: %s）。處理方式：%s' % (type(e).__name__, e, STATE_RESTART)) from e
    if _missing:
        raise SystemExit('ckpt/state.pt 缺少 %s。處理方式：%s' % (_missing, STATE_RESTART))
    if _state['recipe'] != RECIPE:
        # 還沒做任何一步：存好的狀態不動。登記 1.3「接續時配方設定不同 → 停」：不用不同的設定接續
        _diff = sorted(k for k in set(_state['recipe']) | set(RECIPE) if _state['recipe'].get(k) != RECIPE.get(k))
        if 'GPU_CLASS' in _diff and set(_diff) <= RECIPE_GPU_KEYS:
            _todo = ('這次拿到的 GPU 和中斷前不同（中斷前是 %s）：先「中斷連線並刪除執行階段」，選中斷前的同一種 GPU 再「全部執行」；'
                     '那一種拿不到時先不要跑，截圖給 Claude。' % _state['recipe'].get('GPU_CLASS'))
        else:
            _todo = '不要再執行，把這段訊息截圖給 Claude（存好的訓練狀態不會動到）。'
        raise SystemExit('ckpt/state.pt 的 recipe 設定和這次不同，不能接續（PREREG_03 plan 3.5 第 5 條）。不同的項目：%s（存的 → 這次：%s）。'
                         '處理方式：%s' % (_diff, {k: (_state['recipe'].get(k), RECIPE.get(k)) for k in _diff}, _todo))
    try:
        model.load_state_dict(_state['model'])
        optimizer.load_state_dict(_state['optimizer'])
        scheduler.load_state_dict(_state['scheduler'])
    except Exception as e:
        raise SystemExit('ckpt/state.pt 載不進模型、optimizer 或 scheduler（%s: %s）。處理方式：%s'
                         % (type(e).__name__, e, STATE_RESTART)) from e
    global_step, history, start_pass = int(_state['global_step']), list(_state['history']), int(_state['pass']) + 1
    TRAIN_DONE, FINAL_SHA = bool(_state['complete']), _state.get('final_sha256')
    RUN['loaded_state'] = True
    del _state
# LambdaLR 建立時把學習率設成 λ(0)；接續時照已做的步數重設，確保下一步的學習率正確
for _grp in optimizer.param_groups:
    _grp['lr'] = _grp['initial_lr'] * lr_lambda(global_step)
assert scheduler.last_epoch == global_step, (scheduler.last_epoch, global_step)
assert [g['initial_lr'] for g in optimizer.param_groups] == [PEAK_LR]
assert abs(optimizer.param_groups[0]['lr'] - expected_lr(global_step)) < 1e-15
if TRAIN_DONE and not (FINAL_SHA and os.path.exists(FINAL_PATH) and sha256_path(FINAL_PATH) == FINAL_SHA):
    raise SystemExit('state.pt 記錄已訓練完 %d 步，但 final.pth 不見了或和記錄的 SHA-256 不同。處理方式：不要再執行，'
                     '把這段訊息截圖給 Claude。' % global_step)
RUN['start_pass'] = start_pass
if RUN['loaded_state'] and not TRAIN_DONE:
    run_log['resume_count'] += 1
    RUN.update(resumed_training=True, resume_time=now_iso())
save_run_log()
save_config(training={'recipe': RECIPE, 'init_weights_sha256': INIT_WEIGHTS_SHA256, 'loaded_state': RUN['loaded_state'],
                      'start_pass': start_pass, 'global_step_at_start': global_step, 'train_done_at_start': TRAIN_DONE,
                      'optimizer': {'class': type(optimizer).__name__, **optimizer.defaults},
                      'device': str(DEVICE), 'gpu_name': GPU_MODEL_NOW,
                      'tf32': {'cudnn': torch.backends.cudnn.allow_tf32, 'matmul': torch.backends.cuda.matmul.allow_tf32},
                      'checkpoint_rule': '指定第 T 步的 final.pth，不另選 best；inner 監看只記錄'})
save_config_summary(arm=ARM, recipe=RECIPE, resumed=run_log['resume_count'] > 0, resume_count=run_log['resume_count'])
mark_time('train_init', time.time() - t0)
if TRAIN_DONE:
    print('已訓練完 %d 步（%d 輪），這次不訓練' % (global_step, len(history)))
else:
    print('從第 %d 輪開始（%s）：global_step %d／%d，學習率 %.3g'
          % (start_pass, '接續 state.pt' if RUN['loaded_state'] else '從零訓練', global_step, T_TOTAL,
             optimizer.param_groups[0]['lr']))

def _check_loss(loss_t, step_no):
    # 讀出一步的 loss（加進這一輪的合計）；非有限值就照登記停下（登記 3.5 第 4 條）
    global _loss_sum
    v = loss_t.item()
    if not math.isfinite(v):
        stop_arm('nan_stop', '第 %d 步（第 %d 輪）的 loss 是 %r（非有限值）' % (step_no, _p, v),
                 {'step': step_no, 'pass': _p, 'loss': repr(v)})
    _loss_sum += v


# A 臂每一輪要讀到的盤（沒有解析失敗時就是全部 fit）；B 臂每一輪另算
_fit_black_written = sum(COLOR_IS_BLACK[k] for k in WRITTEN_FIT)
TIME_ESTIMATE = None
RUN_FRESH = global_step == 0
TRAIN_T0 = time.time()
_p = start_pass
while not TRAIN_DONE:
    _free = drive_free_bytes()   # 這一輪開始前：Drive 不夠就先停下（不是登記的停止；上一輪的 state.pt 已存好）
    if _free is not None and _free < MIN_PASS_FREE_BYTES:
        drive_space_pause(_p, _free)
    t_pass = time.time()
    random.seed(SEED * 1000 + _p)
    np.random.seed(SEED * 1000 + _p)
    torch.manual_seed(SEED * 1000 + _p)
    D4_GEN.manual_seed(SEED + _p)
    if ARM == 'B':
        _accepted, _acc_by_level, _ = accepted_keys(_p, B_ROWS)
        _expect_keys = _accepted & WRITTEN_FIT
        _expect_black = sum(COLOR_IS_BLACK[k] for k in _expect_keys)
    else:
        _accepted, _acc_by_level, _expect_keys, _expect_black = None, None, WRITTEN_FIT, _fit_black_written
    _loader = make_train_loader(_p, _accepted)
    model.train()
    _order = hashlib.sha256()
    _d4_parts, _seen = [], set()
    _steps, _samples, _loss_sum = 0, 0, 0.0
    _correct = torch.zeros((), dtype=torch.long, device=DEVICE)
    _black = torch.zeros((), dtype=torch.long, device=DEVICE)
    _lr_start = optimizer.param_groups[0]['lr']
    _prev = None          # 上一步的 loss（晚一步才讀，GPU 不用每步等 CPU）
    _pass_complete = True
    for x, y, is_black, keys in iterate(_loader, '第 %d 輪的訓練分片' % _p):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        is_black = is_black.to(DEVICE, non_blocking=True)
        s = torch.randint(0, NUM_SYMS, (len(keys),), device=DEVICE, generator=D4_GEN)
        inputs = model_input(x, is_black, s)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=AMP):
            outputs = model(inputs)
            loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1
        _steps += 1
        _samples += len(keys)
        _correct += (outputs.detach().argmax(1) == y).sum()
        _black += is_black.sum()
        _d4_parts.append(s)
        for k in keys:
            _order.update(k.encode('utf-8'))
            _order.update(b'\n')
        _seen.update(keys)
        if _prev is not None:
            _check_loss(_prev, global_step - 1)
        _prev = loss.detach()
        if RUN_FRESH and global_step == TIME_CHECK_STEP:
            # 第 300 步：推估整個訓練的時間（從第 1 輪建 loader 算起，含 loader 啟動），超過上限就停（登記 3.5 第 3 條）
            _check_loss(_prev, global_step)
            _prev = None
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            _el = time.time() - TRAIN_T0
            TIME_ESTIMATE = {'step': global_step, 'elapsed_seconds': _el, 'ms_per_step': 1000 * _el / global_step,
                             'estimated_train_minutes': _el / global_step * T_TOTAL / 60,
                             'limit_minutes': TIME_LIMIT_MINUTES[GPU_CLASS], 'gpu_class': GPU_CLASS, 'amp': AMP}
            TIMING['time_estimate'] = TIME_ESTIMATE
            write_json(TIMINGS_PATH, timings)
            save_config(time_estimate=TIME_ESTIMATE)
            print('第 %d 步：每步 %.1f 毫秒，推估訓練 %.1f 分鐘（上限 %s 分鐘）'
                  % (global_step, TIME_ESTIMATE['ms_per_step'], TIME_ESTIMATE['estimated_train_minutes'],
                     TIME_ESTIMATE['limit_minutes']), flush=True)
            if TIME_ESTIMATE['estimated_train_minutes'] > TIME_ESTIMATE['limit_minutes']:
                stop_arm('time_stop', '第 %d 步推估的訓練時間 %.1f 分鐘，超過 %s 的上限 %s 分鐘'
                         % (global_step, TIME_ESTIMATE['estimated_train_minutes'], GPU_CLASS, TIME_ESTIMATE['limit_minutes']),
                         TIME_ESTIMATE)
        if _steps % 500 == 0:
            print('  第 %d 輪 step %d（global %d／%d）loss %.4f  lr %.3g  %.1f 分'
                  % (_p, _steps, global_step, T_TOTAL, _loss_sum / max(_steps - 1, 1), optimizer.param_groups[0]['lr'],
                     (time.time() - t_pass) / 60), flush=True)
        if global_step >= T_TOTAL:
            _pass_complete = False   # 到 T 步就停：這一輪可能還沒讀完（剛好讀完時下面會改回 True）
            break
    if _prev is not None:
        _check_loss(_prev, global_step)
    if not _pass_complete and len(_seen) == len(_expect_keys) and _seen == _expect_keys:
        _pass_complete = True
    t_train = time.time() - t_pass
    del _loader
    if _steps == 0:
        stop_arm('pass_check_failed', '第 %d 輪沒有讀到任何資料' % _p, {'pass': _p})
    _black_n, _correct_n = int(_black), int(_correct)
    _d4_sha = hashlib.sha256(torch.cat(_d4_parts).to(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    # 這一輪讀到的盤要和訓練前核對的分片 key 相同（跑完整輪時：數量、key、執黑盤數；到 T 步的那一輪：不能多出別的盤）
    _pass_problems = []
    if _pass_complete:
        if _samples != len(_expect_keys) or _seen != _expect_keys or _black_n != _expect_black:
            _pass_problems.append('讀到 %d 盤（%d 個不同的 key）、執黑 %d，應為 %d 盤、執黑 %d'
                                  % (_samples, len(_seen), _black_n, len(_expect_keys), _expect_black))
    elif _samples != len(_seen) or not _seen <= _expect_keys:
        _pass_problems.append('讀到重複的盤或不該收的盤（%d 盤、%d 個不同的 key）' % (_samples, len(_seen)))
    if abs(optimizer.param_groups[0]['lr'] - expected_lr(global_step)) > 1e-12:
        # 這一輪已用這個學習率訓練過：照登記的停止處理（這一臂不算結果），不是可以重跑的錯誤
        _pass_problems.append('學習率排程和登記的不同（第 %d 步 %r，應為 %r）'
                              % (global_step, optimizer.param_groups[0]['lr'], expected_lr(global_step)))
    if _pass_problems:
        stop_arm('pass_check_failed', '第 %d 輪的核對不符：%s' % (_p, '；'.join(_pass_problems)),
                 {'pass': _p, 'problems': _pass_problems})
    tm = time.time()
    val_loss, val_acc = validate(model, val_loader, criterion)   # inner 逐盤 accuracy：只記錄
    t_mon = time.time() - tm
    _row = {'pass': _p, 'pass_complete': _pass_complete, 'steps': _steps, 'samples': _samples,
            'samples_expected': len(_expect_keys) if _pass_complete else None, 'black': _black_n,
            'global_step': global_step, 'train_loss': _loss_sum / _steps, 'train_acc': 100 * _correct_n / _samples,
            'inner_loss': val_loss, 'inner_acc': val_acc, 'lr_start': _lr_start, 'lr_end': optimizer.param_groups[0]['lr'],
            'seconds_train': t_train, 'seconds_monitor': t_mon, 'ms_per_step': 1000 * t_train / _steps, 'run': RUN['run'],
            'accepted': len(_accepted) if _accepted is not None else None,
            'accepted_by_level': json.dumps(_acc_by_level) if _acc_by_level is not None else None,
            'order_sha256': _order.hexdigest(), 'd4_sha256': _d4_sha}
    history.append(_row)
    if global_step >= T_TOTAL:
        _tmp = guard_write(FINAL_PATH + '.tmp')
        torch.save(model.state_dict(), _tmp)
        os.replace(_tmp, guard_write(FINAL_PATH))
        FINAL_SHA, TRAIN_DONE = sha256_path(FINAL_PATH), True
        save_state(_p, True)
    else:
        _tmp = guard_write(os.path.join(PASS_CKPT_DIR, 'pass_%d.pth.tmp' % _p))   # 本機磁碟，不寫進 Drive
        torch.save(model.state_dict(), _tmp)
        os.replace(_tmp, guard_write(os.path.join(PASS_CKPT_DIR, 'pass_%d.pth' % _p)))
        save_state(_p, False)   # 最後才存；存好才算這一輪完成
        _old = os.path.join(PASS_CKPT_DIR, 'pass_%d.pth' % (_p - 2))
        if os.path.exists(_old):
            os.remove(guard_write(_old))   # 本機的 .pth 只留最後兩輪
    save_history()
    RUN['passes_trained'].append(_p)
    save_run_log()
    save_config_summary(passes_completed=_p, global_step=global_step, final_sha256=FINAL_SHA)
    TIMING.setdefault('ms_per_step', {})[str(_p)] = 1000 * t_train / _steps
    mark_time('pass_%d' % _p, time.time() - t_pass)
    print('第 %d 輪%s：%d 步（global %d／%d）、%d 盤（執黑 %d）| train loss %.4f acc %.2f%% | inner loss %.4f acc %.2f%% | '
          'lr %.3g → %.3g | 每步 %.1f 毫秒 | %.1f 分'
          % (_p, '' if _pass_complete else '（到 T 步停在這一輪中間）', _steps, global_step, T_TOTAL, _samples, _black_n,
             _row['train_loss'], _row['train_acc'], val_loss, val_acc, _lr_start, optimizer.param_groups[0]['lr'],
             _row['ms_per_step'], (time.time() - t_pass) / 60), flush=True)
    _p += 1

TRAIN_COMPLETE = (TRAIN_DONE and global_step == T_TOTAL and sum(h['steps'] for h in history) == T_TOTAL
                  and [h['pass'] for h in history] == list(range(1, len(history) + 1)))
save_config_summary(training_complete=TRAIN_COMPLETE, global_step=global_step, final_sha256=FINAL_SHA,
                    passes=len(history), trained_in_runs=sorted({h['run'] for h in history}))
print('訓練%s：%d 輪、global_step %d／%d，final.pth %s' % ('完成' if TRAIN_COMPLETE else '沒有完成', len(history), global_step,
                                                        T_TOTAL, (FINAL_SHA or '（沒有）')[:12]))
"""

# ---------------------------------------------------------------------------
# final dumps (new)
# ---------------------------------------------------------------------------

FINAL_DUMPS = r"""# 本版新增：最後輸出（PREREG_03 1.3；plan 3.2 第 9 格、3.7）。只用第 T 步的 final.pth（沒跑完 T 步的權重一律不用）；
# 全部 fp32、不開 autocast；先做視角轉換，再做 D4。inner 和本機驗證題各存 8 種對稱，和「原方向、執色對調」
t0 = time.time()
DUMP_NAME = 'p03' + ARM_TAG
W_LABEL = 'prereg03/rank_%s/ckpt/final.pth' % ARM_TAG
if not TRAIN_COMPLETE:
    raise SystemExit('訓練沒有完成 %d 步（global_step %d），不能做最後輸出。請把這段訊息截圖給 Claude。' % (T_TOTAL, global_step))
_net, _wsha = load_net(FINAL_PATH, FINAL_SHA)
DUMPS_FINAL = {'name': DUMP_NAME, 'weights': W_LABEL, 'weights_sha256': _wsha,
               'inner': dump_inner_p03(_net, DUMP_NAME, W_LABEL, _wsha),
               'val': dump_val_p03(_net, DUMP_NAME, W_LABEL, _wsha)}
del _net
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
with np.load(_dump_path('inner_%s_d4' % DUMP_NAME), allow_pickle=False) as _z:
    _acc0 = float((_z['logits'][:, 0, :].argmax(1) == INNER_KEYS['label']).mean() * 100)
with np.load(_dump_path('inner_%s_s0_swap' % DUMP_NAME), allow_pickle=False) as _z:
    _acc_swap = float((_z['logits'].argmax(1) == INNER_KEYS['label']).mean() * 100)
DUMPS_FINAL.update(inner_acc_s0_pct=_acc0, inner_acc_s0_colour_swapped_pct=_acc_swap,
                   inner_acc_from_history_pct=history[-1]['inner_acc'], abs_diff_pp=abs(_acc0 - history[-1]['inner_acc']))
if DUMPS_FINAL['abs_diff_pp'] > 0.05:
    print('注意：final.pth 的 inner accuracy，逐盤輸出算出 %.4f%%，最後一輪的監看是 %.4f%%，差超過 0.05 個百分點。'
          '結果照樣打包，回報 Claude 時提一下。' % (_acc0, history[-1]['inner_acc']))
save_config(dumps_final=DUMPS_FINAL)
mark_time('final_dumps', time.time() - t0)
print('最後輸出完成：%s（inner 原方向 accuracy %.4f%%，執色對調 %.4f%%，只記錄）' % (DUMP_NAME, _acc0, _acc_swap))
"""

# ---------------------------------------------------------------------------
# final check (rewrite)
# ---------------------------------------------------------------------------

FINAL = r"""# 本版改寫的結尾：核對逐盤輸出（完成標記、SHA-256、列數、形狀、權重 SHA-256、NaN 位置、key 檔摘要）、訓練跑完 T 步、
# Drive 上的 history.csv（和記憶體裡的相同）、三個只讀目錄前後的指紋；不符就照停止條件處理（*_results_failed.zip，這一臂不用）。通過才補完紀錄、打包
# prereg03_rank_<臂>_results.zip（不含權重、< 100 MB），最後把 Drive 的寫入送出
import zipfile

t0 = time.time()
V = parse_val()
_parsed = V['keys']['parsed']
_n_in, _n_val = len(df_val), len(V['feats'])
_expect = {'inner_keys': (_n_in, None), 'val_keys': (_n_val, None),
           'inner_%s_d4' % DUMP_NAME: (_n_in, NUM_SYMS), 'inner_%s_s0_swap' % DUMP_NAME: (_n_in, 1),
           'val_%s_d4' % DUMP_NAME: (_n_val, NUM_SYMS), 'val_%s_s0_swap' % DUMP_NAME: (_n_val, 1)}
_final_problems, DUMP_CHECK = [], {}
for _stem, (_rows, _syms) in _expect.items():
    _path, _m = _dump_path(_stem), read_json(_marker_path(_stem))
    if _m is None or not os.path.exists(_path):
        _final_problems.append('%s：檔案或完成標記不見了' % _stem)
        continue
    _sha = sha256_path(_path)
    if _sha != _m.get('sha256'):
        _final_problems.append('%s：SHA-256 和完成標記不同' % _stem)
    with np.load(_path, allow_pickle=False) as _z:
        if _syms is None:
            _arr = {k: _z[k] for k in _z.files}
            _got_rows = len(next(iter(_arr.values())))
            if any(len(v) != _got_rows for v in _arr.values()) or keys_digest(_arr) != KEYS_CHECK[_stem]['digest']:
                _final_problems.append('%s：內容和訓練前核對的不同' % _stem)
        else:
            _L = _z['logits']
            _got_rows = _L.shape[0]
            _shape = (_rows, _syms, NUM_CLASSES) if _syms > 1 else (_rows, NUM_CLASSES)
            if _L.dtype != np.float32 or _L.shape != _shape:
                _final_problems.append('%s：形狀 %s %s，應為 float32 %s' % (_stem, _L.dtype, _L.shape, _shape))
            if str(_z['weights_sha256']) != FINAL_SHA or _m.get('weights_sha256') != FINAL_SHA:
                _final_problems.append('%s：權重 SHA-256 不是 final.pth 的' % _stem)
            _flat = _L.reshape(len(_L), -1)
            _nan_rows, _finite_rows = np.isnan(_flat).all(1), np.isfinite(_flat).all(1)
            if _stem.startswith('val_'):
                if not (np.array_equal(_nan_rows, ~_parsed) and _finite_rows[_parsed].all()):
                    _final_problems.append('%s：NaN 的位置和解析失敗的 slot 不同' % _stem)
            elif not _finite_rows.all():
                _final_problems.append('%s：有非有限值' % _stem)
    if _got_rows != _rows or _m.get('rows') != _rows:
        _final_problems.append('%s：%s 列，應為 %d 列' % (_stem, _got_rows, _rows))
    if os.path.getsize(_path) >= DUMP_MAX_BYTES:
        _final_problems.append('%s：超過 100 MB' % _stem)
    DUMP_CHECK[_stem] = {'rows': _got_rows, 'bytes': os.path.getsize(_path), 'sha256': _sha,
                         'weights': _m.get('weights'), 'weights_sha256': _m.get('weights_sha256')}
# 本版：Drive 上的 ckpt/history.csv 要和記憶體裡的 history 相同、記到第 T 步（打包進 zip 的就是這個檔）。不同就照記憶體裡的重寫一次
# （例如最後一輪存好 state.pt 之後、寫完 history.csv 之前斷線，接續時訓練格不會再寫），重寫後再核對一次，還是不同才算收尾核對不符
_hist_before = history_file_problems()
if _hist_before:
    print('ckpt/history.csv 和記憶體裡的 history 不同（%s），照記憶體裡的重寫' % '；'.join(_hist_before))
    save_history()
_hist_after = history_file_problems()
_final_problems.extend(_hist_after)
HISTORY_FILE_CHECK = {'problems_before_rewrite': _hist_before, 'rewritten': bool(_hist_before), 'problems': _hist_after,
                      'rows': len(history)}
if not TRAIN_COMPLETE or history[-1]['global_step'] != T_TOTAL:
    _final_problems.append('history 沒有記到第 %d 步' % T_TOTAL)
if not os.path.exists(FINAL_PATH) or sha256_path(FINAL_PATH) != FINAL_SHA:
    _final_problems.append('final.pth 不見了或 SHA-256 和記錄的不同')

# 三個只讀目錄的指紋（結尾）：必須和這次開頭記的完全相同
READONLY_AFTER, _ro_changed = record_readonly_after()
if _ro_changed:
    _final_problems.append('只讀目錄在這次執行中被改動：%s' % {k: v[:5] for k, v in _ro_changed.items()})

OUTPUTS = {
    'arm': ARM, 'dumps': DUMP_CHECK, 'inner_games': _n_in, 'val_questions': len(V['spans']), 'val_games': _n_val,
    'val_games_unparsed': int((~_parsed).sum()), 'training_complete': TRAIN_COMPLETE, 'global_step': global_step,
    'passes': len(history), 'history_file': HISTORY_FILE_CHECK, 'final_sha256': FINAL_SHA, 'final_weights': W_LABEL,
    'readonly_unchanged': not _ro_changed, 'readonly_files_fingerprinted': {k: len(v) for k, v in READONLY_AFTER.items()},
    'problems': _final_problems,
    'files': {n: os.path.getsize(os.path.join(RESULT_DIR, n)) for n in sorted(os.listdir(RESULT_DIR))
              if os.path.isfile(os.path.join(RESULT_DIR, n))},
}
save_config(outputs=OUTPUTS)
if _final_problems:
    stop_arm('final_check_failed', '收尾核對不符（%d 項）：%s' % (len(_final_problems), '；'.join(_final_problems[:5])),
             {'problems': _final_problems})
RUN.update(end=now_iso(), status='completed')
run_log['status'] = 'completed'
save_config(end=RUN['end'])
save_run_log()
mark_time('final_check', time.time() - t0)
mark_time('total_this_run', time.time() - RUN_START)
print(json.dumps({k: v for k, v in OUTPUTS.items() if k not in ('dumps', 'files')}, ensure_ascii=False, indent=1))
print('ckpt/', sorted(os.listdir(CKPT_DIR)))

# 結果打包：結果目錄裡除了權重和暫存檔以外的檔案，打成一個 zip 放在結果目錄旁邊（prereg03/）
RESULTS_ZIP, _skipped_big = pack_results(RESULTS_ZIP_NAME)
with zipfile.ZipFile(RESULTS_ZIP) as _zf:
    _names = set(_zf.namelist())
_missing_in_zip = [s for s in _expect if 'rank_%s/dumps/%s.npz' % (ARM_TAG, s) not in _names]
_zip_bytes = os.path.getsize(RESULTS_ZIP)
if _missing_in_zip or _skipped_big or _zip_bytes >= 100 * 1024 * 1024:
    stop_arm('final_check_failed', '結果 zip 不完整或太大（缺 %s，跳過 %s，%.1f MB）' % (_missing_in_zip, _skipped_big, _zip_bytes / 1e6),
             {'missing_in_zip': _missing_in_zip, 'skipped_big': _skipped_big, 'zip_bytes': _zip_bytes})
# 下載 zip、刪除執行階段之前，先把 Drive 的寫入全部送出（final.pth、state.pt 之後還要用）
flush_drive()
print('完成。請下載這個檔回傳給 Claude（合併、校準、打分都在本機做）：%s（%.1f MB）' % (RESULTS_ZIP, _zip_bytes / 1e6))
"""

# ---------------------------------------------------------------------------
# Assemble. Each entry: (kind, key or tag, source)
#   "md_new" / "code_new": new cell (tag names it); "base": reused base cell (changed only via rep());
#   "rewrite": base cell replaced entirely
# ---------------------------------------------------------------------------

PLAN = [
    ("md_new", "header", MD_HEADER),
    ("base", "setup", _setup),
    ("base", "prepare", base_src("prepare")),
    ("base", "official_1", base_src("official_1")),
    ("base", "environment", base_src("environment")),
    ("base", "official_2", _off2),
    ("code_new", "preflight_p03", PREFLIGHT),
    ("base", "official_4", base_src("official_4")),
    ("base", "official_6", base_src("official_6")),
    ("base", "official_8", base_src("official_8")),
    ("base", "feature_backup_tools", _ft),
    ("base", "official_9", _off9),
    ("code_new", "view_d4", VIEW_D4),
    ("code_new", "colour_table", COLOUR_TABLE),
    ("code_new", "arm_b_sampling", ARM_B),
    ("rewrite", "official_11", OFF11),
    ("rewrite", "official_12", OFF12),
    ("base", "official_14", _off14),
    ("rewrite", "official_15", OFF15),
    ("base", "dump_tools", _dt),
    ("code_new", "key_files", KEYS_CELL),
    ("code_new", "training", TRAIN),
    ("code_new", "final_dumps", FINAL_DUMPS),
    ("rewrite", "final_check", FINAL),
]


def build(arm):
    nb = nbformat.v4.new_notebook()
    nb.metadata = json.loads(json.dumps(base["metadata"]))
    used = set()
    for k, (kind, key, src) in enumerate(PLAN):
        src = src.rstrip("\n")
        if key == "setup":
            src = src.replace("__ARM_LINE__", ARM_LINE % arm)
            assert src.count(ARM_LINE % arm) == 1 and "__ARM_LINE__" not in src
        if kind == "md_new":
            cell = nbformat.v4.new_markdown_cell(src)
        else:
            compile(src, "<cell %d>" % k, "exec")   # plain Python (the smoke runner compiles whole cells too)
            cell = nbformat.v4.new_code_cell(src)
        cell["id"] = "c10-rank03-%02d" % k
        if kind in ("base", "rewrite"):
            j = BASE_BY_KEY[key]
            assert key not in used, key
            used.add(key)
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
    return nb, used


def cell_key(c):
    """The key of a built cell: the base cell's key for reused/rewritten cells, the tag for new cells."""
    m = c["metadata"]["c10"]
    return base_key(m["base"]) if "base_cell" in m else m["new"]


NB = {}
for _arm in ("A", "B"):
    NB[_arm], _used = build(_arm)
USED = _used

# ---------------------------------------------------------------------------- integrity checks
nb = NB["A"]
_changed_keys = {cell_key(c) for c in nb.cells if "base_cell" in c["metadata"]["c10"] and c["metadata"]["c10"]["changed"]}
_named = {k for k, _l, _t in CHANGE_ROWS if k is not None}
assert _changed_keys == _named, ("changed but not named", _changed_keys - _named, "named but unchanged", _named - _changed_keys)
for c in nb.cells:
    m = c["metadata"]["c10"]
    if "base_cell" in m and not m["changed"]:
        assert c["source"] == "".join(BASE_CELLS[m["base_cell"]]["source"]), m
assert len({c["id"] for c in nb.cells}) == len(nb.cells)
# the forbidden token: only in the unchanged prepare cell's refusal line
_prep = base_src("prepare")
_tok_allowed = _prep.count(FORBIDDEN_TOKEN)
assert _tok_allowed == 1 and ("'%s' in p for p in _all_files" % FORBIDDEN_TOKEN) in _prep
for _arm in ("A", "B"):
    assert sum(c["source"].count(FORBIDDEN_TOKEN) for c in NB[_arm].cells) == _tok_allowed, "forbidden token in a cell"
    _prep_out = [c for c in NB[_arm].cells if cell_key(c) == "prepare"]
    assert len(_prep_out) == 1 and not _prep_out[0]["metadata"]["c10"]["changed"]
# segment identity: functions kept from the base dump tools are byte-equal (d4 moved verbatim into view_d4)


def _func_segments(src):
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}


_base_f = _func_segments(_dt_base)
_new_dt = _func_segments([c["source"] for c in nb.cells if cell_key(c) == "dump_tools"][0])
for _fn in ("load_net", "_dump_path", "_marker_path", "dump_is_done", "write_dump", "require_gpu_for", "pack_results",
            "flush_drive", "write_keys", "sgf_to_features", "parse_val"):
    if _fn in ("load_net", "require_gpu_for", "pack_results"):
        continue   # changed on purpose (501 layers / comments); compared below without the changed lines
    assert _new_dt[_fn] == _base_f[_fn], _fn
_view_src = [c["source"] for c in nb.cells if cell_key(c) == "view_d4"][0]
assert D4_SEGMENT in _view_src and _func_segments(_view_src)["d4"] == _base_f["d4"]
_load_rows_cell = [c["source"] for c in nb.cells if cell_key(c) == "official_4"]
assert len(_load_rows_cell) == 1 and _load_rows_cell[0] == base_src("official_4")

# ---------------------------------------------------------------------------- write the notebooks
for _arm in ("A", "B"):
    os.makedirs(os.path.dirname(OUT_NB[_arm]), exist_ok=True)
    with open(OUT_NB[_arm], "w", encoding="utf-8", newline="\n") as f:
        nbformat.write(NB[_arm], f)

# the two files differ in exactly one line: the ARM line
_la = open(OUT_NB["A"], encoding="utf-8").read().split("\n")
_lb = open(OUT_NB["B"], encoding="utf-8").read().split("\n")
assert len(_la) == len(_lb)
_dl = [i for i, (x, y) in enumerate(zip(_la, _lb)) if x != y]
assert len(_dl) == 1, _dl
assert (ARM_LINE % "A") in _la[_dl[0]] and (ARM_LINE % "B") in _lb[_dl[0]]
assert _la[_dl[0]].replace(ARM_LINE % "A", ARM_LINE % "B") == _lb[_dl[0]]

# ---------------------------------------------------------------------------- per-cell diffs
for _arm in ("A", "B"):
    nb = NB[_arm]
    out = []
    out.append("# prereg03 rank（%s 臂）：colab/prereg02_rank.ipynb 與 %s 的逐格 unified diff（程式格、markdown 格、notebook metadata）"
               % (_arm, OUT_REL[_arm]))
    out.append("# 底稿：%s  sha256=%s" % (BASE_REL, sha(BASE_NB)))
    out.append("# 本版：%s  sha256=%s" % (OUT_REL[_arm], sha(OUT_NB[_arm])))
    out.append("# 產生方式：colab/build_prereg03_rank_nb.py；difflib.unified_diff，依 cell metadata c10.base_cell 配對底稿的格；"
               "新增的格對空白比。cell 編號從 0 起算。")
    out.append("# 每一處改動的說明見本版第 0 格「和 prereg02 notebook 的差異」；沒列在表上的沿用格和底稿逐字相同。"
               "A、B 兩份只差設定格的 ARM 那一行。")
    summary, body = [], []
    for k, cell in enumerate(nb.cells):
        meta = cell["metadata"]["c10"]
        kind = "程式格" if cell["cell_type"] == "code" else "markdown"
        if "base_cell" in meta:
            j = meta["base_cell"]
            old_src = "".join(BASE_CELLS[j]["source"])
            what = base_key(meta["base"])
            a = "prereg02_rank.ipynb cell %d（%s）" % (j, what)
            status = "改寫" if meta.get("rewritten") else ("改動" if meta["changed"] else "完全相同")
        else:
            old_src = ""
            a = "/dev/null（新增：%s）" % meta["new"]
            status = "新增"
        b = "%s cell %d" % (os.path.basename(OUT_REL[_arm]), k)
        summary.append("#   本版 cell %2d（%s）← %s：%s" % (k, kind, a, status))
        diff = list(difflib.unified_diff(old_src.splitlines(), cell["source"].splitlines(), fromfile=a, tofile=b, lineterm=""))
        body.append("")
        body.append("=" * 100)
        body.append("本版 cell %d（%s）↔ %s（%s）" % (k, kind, a, status))
        body.append("=" * 100)
        body.extend(diff if diff else ["（完全相同）"])
    _dropped = []
    for j, c in enumerate(BASE_CELLS):
        if base_key(c["metadata"]["c10"]) not in USED:
            _dropped.append("#   prereg02 cell %2d（%s，%s）" % (j, "程式格" if c["cell_type"] == "code" else "markdown",
                                                           base_key(c["metadata"]["c10"])))
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
    out.extend(list(difflib.unified_diff(_meta_old, _meta_new, fromfile="prereg02 metadata", tofile="本版 metadata",
                                         lineterm="")) or ["（完全相同）"])
    os.makedirs(os.path.dirname(OUT_DIFF[_arm]), exist_ok=True)
    with open(OUT_DIFF[_arm], "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")

for _arm in ("A", "B"):
    print("wrote", OUT_NB[_arm], sha(OUT_NB[_arm]))
    print("wrote", OUT_DIFF[_arm], sha(OUT_DIFF[_arm]))
print("cells", len(NB["A"].cells), "| base cells changed", sorted(_changed_keys), "| dropped",
      sorted(base_key(c["metadata"]["c10"]) for c in BASE_CELLS if base_key(c["metadata"]["c10"]) not in USED))
print("pins: key digests", KEY_DIGEST, "| feature fingerprint", FEAT_FINGERPRINT_SHA256)
