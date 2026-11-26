"""Build colab/test02_rank.ipynb from the committed colab/prereg02_rank.ipynb and write
results/test02/notebook_diff_rank.txt (per-cell unified diff against the prereg02 notebook, plus the
notebook metadata).

Spec: results/test02/design/spec.md sections 2.0, 2.1 and 2.3; registration: docs/PREREG_02_OPENING_FUSION.md
section 9 (rev 5), which wins on any scientific question.

How the notebook is assembled (the same discipline as colab/build_prereg02_rank_nb.py):
  * the base is colab/prereg02_rank.ipynb, whose SHA-256 is asserted first; base cells are picked by
    their `c10` cell metadata, never by position;
  * a reused cell is copied verbatim or changed only through rep() (target exactly once; cut() finds a
    target between two markers that each occur exactly once);
  * every changed base cell is named in CHANGE_ROWS (the table in notebook cell 0 and
    `changes_from_prereg02` in config.json), and every named cell really changed;
  * segment identity: d4 and its self-check, load_net, sgf_to_features, dump_is_done, write_dump,
    write_keys (and the other kept dump tools), the question-parsing loop from `sgf_cols =` to `parsed =`
    and the per-question inference loop (`with torch.no_grad(): ...`) are byte-equal to the base cell
    (ast.get_source_segment / statement line spans);
  * every code cell compiles; the inherited prepare-cell refusal line is the only line of the notebook
    containing the validation label file token; the three set cells differ only in their first line;
  * pins are read from the repository at build time (results/prereg02/rank/frozen_config.json and the
    committed dumps/markers) and asserted against the values of spec 2.1;
  * config.json records notebook_sha256 = sha256 of json.dumps([cell sources], ensure_ascii=False) with
    that value replaced by 64 zeros (a notebook cannot hold the SHA-256 of its own file).
The output is byte-identical when the builder is re-run.

    python colab/build_test02_rank_nb.py [REPO] [--out-nb PATH] [--out-diff PATH]
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
ap.add_argument("--out-nb", help="default: <repo>/colab/test02_rank.ipynb")
ap.add_argument("--out-diff", help="default: <repo>/results/test02/notebook_diff_rank.txt")
ARGS = ap.parse_args()
REPO = os.path.abspath(ARGS.repo)

BASE_REL = "colab/prereg02_rank.ipynb"
BASE_NB = os.path.join(REPO, *BASE_REL.split("/"))
BASE_SHA256 = "9fb1f9d06ba5ee5fbebfb6dfe022246888c93e7762e9a89e1adc99c9f1c12b0c"
OUT_REL = "colab/test02_rank.ipynb"
OUT_NB = ARGS.out_nb or os.path.join(REPO, *OUT_REL.split("/"))
OUT_DIFF = ARGS.out_diff or os.path.join(REPO, "results", "test02", "notebook_diff_rank.txt")

# The validation label file token is spelled in two parts so that a repo-wide search for it finds only code
# that refuses such files.
FORBIDDEN_TOKEN = "answ" + "ers"
HASH_PLACEHOLDER = "0" * 64
HASH_LINE = "NOTEBOOK_SHA256 = '%s'"

# spec 2.1 pins; each is re-derived from the repository below and must agree
SPEC_WEIGHTS_SHA256 = "423581b4fddd5c2c160ebf691f4e86b9e00c4fe743d5cfb23df7891f4a2ca939"
SPEC_SEAM_REF_SHA256 = {"val_keys.npz": "055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c",
                        "val_p02e9_d4.npz": "ec0b729d805eb9f3458f7a497444b099b8ec47e5cec73703ae4b8bf9cf0a11b2"}
SPEC_P02_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
WEIGHTS_LABEL = "prereg02/rank/ckpt/epoch_9.pth"
OFFICIAL_TEST_NAMES = ["rank_prediction_test_public.csv", "rank_prediction_test_private.csv",
                       "player_identification_test_public.csv", "player_identification_test_private.csv",
                       "player_identification_candidates_public.csv", "player_identification_candidates_private.csv"]
SETS = ("seam_val", "public", "private")


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


def nb_source_sha256(sources):
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode("utf-8")).hexdigest()


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
_FC = json.load(open(os.path.join(_P02, "frozen_config.json"), encoding="utf-8"))
assert _FC["dumps"] == {"p02e9_d4": {"weights_sha256": SPEC_WEIGHTS_SHA256, "syms": 8}}, _FC["dumps"]
assert _FC["combo"]["id"] == "R20" and _FC["smoke"] is False
_MARK = json.load(open(os.path.join(_P02, "dumps", "_DONE_val_p02e9_d4.json"), encoding="utf-8"))
with np.load(os.path.join(_P02, "dumps", "val_p02e9_d4.npz"), allow_pickle=False) as _z:
    _NPZ_WSHA = str(_z["weights_sha256"])
assert _MARK["weights_sha256"] == _NPZ_WSHA == SPEC_WEIGHTS_SHA256, (_MARK["weights_sha256"], _NPZ_WSHA)
assert _MARK["weights"] == WEIGHTS_LABEL, _MARK["weights"]
assert _MARK["gpu_name"] == SPEC_P02_GPU, _MARK["gpu_name"]
SEAM_REF_SHA256 = {n: sha(os.path.join(_P02, "dumps", n)) for n in SPEC_SEAM_REF_SHA256}
assert SEAM_REF_SHA256 == SPEC_SEAM_REF_SHA256, SEAM_REF_SHA256
for _n in SEAM_REF_SHA256:
    _m = json.load(open(os.path.join(_P02, "dumps", "_DONE_%s.json" % _n[:-4]), encoding="utf-8"))
    assert _m["sha256"] == SEAM_REF_SHA256[_n], _n
WEIGHTS_SHA256, P02_GPU = SPEC_WEIGHTS_SHA256, SPEC_P02_GPU

# ---------------------------------------------------------------------------
# What changed relative to colab/prereg02_rank.ipynb. One list feeds the table in notebook cell 0 and
# `changes_from_prereg02` in config.json. Each row: (base cell key or None for new/notebook-wide rows, label, text).
# ---------------------------------------------------------------------------
CHANGE_ROWS = [
    (None, "新增", "第一格 markdown：用途、步驟、GPU、會停下的情況、輸出清單、交回方式、這張差異表。"),
    ("setup", "設定",
     "先讀 Drive 上 `C10_test_v1.zip` 的 `tests/MANIFEST.json` 決定種類（正式 `official`、彩排 `rehearsal`；SMOKE 只接受 `smoke`），"
     "找不到或種類不對就停。結果目錄改成 `test02`（彩排 `test02_rehearsal`，SMOKE `test02_smoke`）的 `rank/`，沒有 `ckpt/`、"
     "沒有分片備份。只讀目錄改成兩個：`prereg01`、`prereg02`（SMOKE 是 `_smoke`），`guard_write()` 兩個都擋，"
     "`readonly_fingerprint(root)` 取代 `p01_fingerprint()`。`sha256_path()`、`gpu_name_of()` 從 prereg02 的 preflight 格"
     "逐字搬到這裡（本版的 preflight 在 GPU 關卡之後）。拿掉收尾開關 `FINISH_WITHOUT_TRAINING` 和 `SMOKE_CFG_P02`；"
     "`P02_CHANGES` 換成 `T02_CHANGES`，另記這份 notebook 的 SHA-256（規則見 `NOTEBOOK_SHA256_RULE`）。"
     "`config.json` 的 `prereg` 改指第 9 節，記 `rehearsal`、測試集種類、底稿 SHA-256、這張表和只讀目錄。"
     "`run_log.json` 頂層加 `status`（這次執行開始時清空，結束時寫 completed／seam_failed／readonly_changed）。"),
    ("prepare", "準備",
     "拿掉解開訓練資料的整段（本版只推論，不讀訓練資料）；第一行註解跟著改。打包檔的 SHA-256 核對和拒收答案檔的那一行照舊。"),
    ("official_2", "底稿官方 2",
     "拿掉 `CKPT_DIR`、`guard_write(CKPT_DIR)`／`makedirs`、`EXTRA_EPOCHS`／`PEAK_LR`／`FINAL_EPOCH` 和它們的 assert。"
     "沒有 GPU 的關卡拿掉收尾開關的例外；型號關卡照舊（只接受 G4〔RTX PRO 6000〕或 A100）。最後的 `print` 只印裝置、GPU、NPROC。"),
    (None, "新增 preflight",
     "在 GPU 關卡之後：① 權重 `prereg02/rank/ckpt/epoch_9.pth` 的 SHA-256 等於登記的值（SMOKE：等於 prereg02_smoke 完成標記記的）；"
     "② 接縫參照 `prereg02/rank/dumps/val_keys.npz`、`val_p02e9_d4.npz` 的 SHA-256 釘死，prereg02 的 GPU 型號記下；"
     "③ 測試集 zip 解到 `WORK/tests/`，只收 `tests/` 底下的 6 個官方檔名和 `MANIFEST.json`，每個檔的 SHA-256、大小要等於 MANIFEST；"
     "④ 快速格式檢查（T1 的一部分：沒有 BOM、400 題、`question_id` 不重複且前綴是 `pub_`／`priv_`、`num_games` 是 5–20 的整數、"
     "`sgf_1..20`／`color_1..20` 欄都在、剛好前 `num_games` 格有棋、執色是 B/W、兩份題目不重疊）；"
     "⑤ 上一次執行的 `seam_check.json` 沒過就停（已完成的 dump 會被沿用，登記的「重跑一次」必須重算：先把結果資料夾改名保留）；"
     "每組的 `_INPUT.json`（題目檔 SHA-256）：已存在但不同就停（防止沿用換過題目檔的舊 dump）；"
     "⑥ Drive 剩餘空間 ≥ 0.5 GB；⑦ 兩個只讀目錄的開頭指紋 `readonly_before.json`，重跑時和第一次記的不同就停。"
     "`config.json` 頂層記 GPU、torch、cuDNN、權重、測試集 zip 與 MANIFEST 的 SHA-256、各組題目檔 SHA-256 與題數。"),
    ("dump_tools", "底稿 dump 工具",
     "拿掉 inner 的部分（`EXPECTED_INNER_GAMES`、`INNER_*`、`decode_with_key`、`KEY_LOADER`、`dump_inner`）和最後的 print。"
     "`DUMP_DIR` 不在這格建立，改由各組的格子在呼叫前指定（底稿的函式把它當全域變數讀，本身一字不改）。"
     "`parse_val()` 改成 `parse_questions(path)`（依題目檔路徑快取，`pd.read_csv(path, dtype={'question_id': str})`，"
     "SMOKE 照舊只取前 12 題）；`dump_val(net, …)` 改成 `dump_questions(net, path, …)`，檔名仍是 `val_` 開頭"
     "（在 test02 裡只代表「一組題目的逐盤輸出」）。`record_p01_after()` 改成兩個只讀目錄的 `record_readonly_after()`。"
     "`d4`、`load_net`、`sgf_to_features`、`dump_is_done`、`write_dump`、`write_keys`、`pack_results`、`flush_drive`、解析迴圈"
     "和逐題推論迴圈和底稿逐字相同（產生器用 ast 核對）。"),
    (None, "新增三組逐盤輸出",
     "`seam_val`（打包檔的本機驗證題 1,589 題）、`public`、`private` 各一格，內容相同、只有第一行不同："
     "`DUMP_DIR = <組>/dumps`，`write_keys('val_keys', …)`，`dump_questions(…, 'p02e9_d4', 8, …)`；權重只載入一次"
     "（`load_net(path, expect_sha=登記的 SHA-256)`）。"),
    (None, "新增接縫核對",
     "`seam_val` 的逐盤輸出和 Drive 上 prereg02 的比（登記第 9 節第 2 條）：key 陣列完全相同；原方向逐盤 softmax 的每題 float32 平均"
     "最大差 ≤ 1e-4（算這份 dump 的 GPU 型號和 prereg02 相同時）或 ≤ 1e-3（不同時）；規則 B 不同 ≤ 4 題（SMOKE 依題數等比例，至少 1）。"
     "另記是否逐位元組相同、8 種對稱的 logits 最大差、兩邊的 GPU／torch／cuDNN。三組 dump 的完成標記記的 GPU 型號、torch、"
     "cuDNN 必須和 `seam_val` 的相同（中斷後接續時，已完成的組會被沿用，可能是別的執行環境算的；不同就算接縫沒過，"
     "接縫才涵蓋產生測試輸出的環境）。寫 `seam_check.json`；沒過就記結尾指紋、"
     "`status = seam_failed`、打包、送出 Drive 的寫入後停下。"),
    ("final_check", "底稿結尾（改寫）",
     "核對三組 dump 的完成標記、SHA-256、列數、形狀、權重、NaN 位置和 `_INPUT.json`；記 `readonly_after.json` 並和開頭比對"
     "（不同就 `status = readonly_changed`，打包後停下）；補完 config／timings／run_log；打包 `test02/test02_rank_results.zip`"
     "（zip 內路徑從 `rank/` 起；跳過 `.pth`、`.pt`、`.tmp` 和 ≥ 100 MB 的檔），確認必要的檔都在 zip 裡；最後 `drive.flush_and_unmount()`。"),
    (None, "拿掉",
     "prereg02 的 preflight、官方 4、6、8、9、11、12、14、15（讀訓練資料、抽分片、loader、建模型、驗證）、分片備份工具、"
     "P1 輸出與重現核對、接著訓練、P2 輸出。本版不訓練、不讀訓練資料。"),
    (None, "metadata",
     "notebook metadata 沿用 prereg02。每格 cell id 是 `c10-rank-t02-NN`，cell metadata `c10` 記底稿的格號、底稿的 `c10` 和是否改動。"),
]
T02_CHANGES = ["[%s] %s" % (label, text) for _k, label, text in CHANGE_ROWS]


def _md_cell(s):
    return s.replace("|", "\\|")


CHANGE_TABLE = "\n".join("| %s | %s |" % (label, _md_cell(text)) for _k, label, text in CHANGE_ROWS)

# ---------------------------------------------------------------------------
# Notebook cell 0 (markdown)
# ---------------------------------------------------------------------------
MD_HEADER = r"""# C10 測試日（test02）：競賽二（棋力預測）逐盤輸出（Colab 版，只推論）

**用途**：`docs/PREREG_02_OPENING_FUSION.md` 第 9 節（rev 5）的 Colab 部分。用 prereg02 指定的權重 `epoch_9.pth`（不訓練、不調參），算三組題目的逐盤輸出（8 種對稱的 logits）：

1. `seam_val`：本機驗證題 1,589 題（接縫核對用，和 prereg02 當時算的比）。
2. `public`、`private`：官方測試題各 400 題。

合併、校準、開局特徵、規則 B 和交件檔都在本機做（`tools/t02_rank.py`），不在這裡。這份 notebook 不讀訓練資料，也不讀任何答案檔。

## 怎麼跑

0. 用**跑 prereg02 的同一個 Google 帳號**：權重和接縫參照在它的 Drive（`MyDrive/C10/results/prereg02/rank/`）。
1. `MyDrive/C10/` 裡要有：`C10_bundle_v1.zip`（原本那份，不用重傳）、Claude 給你的 `C10_test_v1.zip`（檔名不能改、不要解壓），和這份 `test02_rank.ipynb`。
2. 選單「執行階段 → 變更執行階段類型」：選 **G4 GPU**。G4 是灰色、點不了時，選 **A100 GPU**，「大量 RAM」關掉。沒有 GPU 或 Colab 改派別的 GPU 時，前面幾格就會停下。
3. 選單「執行階段 → 全部執行」。第一個程式格會要求存取 Google Drive，按允許。
4. 跑完後（G4 約 10 分鐘，推估），最後兩行是「Drive 已寫入完成並卸載」和「完成。請下載…」。先按「執行階段 → 中斷連線並刪除執行階段」，再下載 `MyDrive/C10/results/test02/test02_rank_results.zip`，把位置告訴 Claude。
5. 跑完前不要關分頁，電腦不要睡眠。

詳細步驟以 `docs/COLAB_RUN.md` 的「測試集（test02）」一節為準；和這裡不一致時，以那一節為準。

## 會停下來的情況（錯誤訊息裡都寫了處理方式）

依序：
- **找不到測試集 zip，或 zip 的種類不對**：在第一個程式格停下。
- **沒有 GPU，或 GPU 不是 G4／A100**：還沒開始算。先刪除執行階段，換成 G4（灰色時 A100）再全部執行。
- **權重或接縫參照的 SHA-256 不符、測試集 zip 的檔案和 MANIFEST 不符、題目檔格式不對、Drive 空間不足、只讀目錄和第一次執行時不同**：preflight 格停下，還沒開始算。
- **上一次執行的接縫核對沒過**（`seam_check.json` 記沒過）：停下。舊的逐盤輸出不能沿用，重跑必須重算：照 Claude 的指示先把 Drive 的 `C10/results/test02/rank` 資料夾改名成 `rank_failed_1`，再全部執行。
- **某一組已有逐盤輸出，但題目檔換過**（`_INPUT.json` 不同）：停下，不會沿用舊的輸出。
- **任何寫進 prereg01、prereg02 目錄的動作**：一律擋下。
- **接縫核對沒過**（包括接續後三組的逐盤輸出不是同一個 GPU 型號、torch、cuDNN 算的）：這次的測試輸出一律不用、不上傳。會先打包 zip 再停下。**不要自己重跑**，下載 zip、截圖給 Claude。
- **結尾發現 prereg01 或 prereg02 目錄被改動**：結果照樣打包，但會停下；截圖給 Claude。

出現紅字時都先截圖給 Claude，不要自己重跑。

## 結果寫到哪裡

`MyDrive/C10/results/test02/rank/`（彩排是 `test02_rehearsal/`）：

| 檔案 | 內容 |
|---|---|
| `<組>/dumps/val_keys.npz` | 每個 slot 一列：`question_id`、`slot`（從 1 起算）、`color`、`parsed`；題目檔的列順序，再依 slot 順序 |
| `<組>/dumps/val_p02e9_d4.npz` | `logits` float32 [盤數, 8, 10]、`weights_sha256`、`syms`；解析失敗的 slot 整列是 NaN |
| `<組>/dumps/_DONE_<檔名>.json` | 完成標記：列數、SHA-256、權重 SHA-256、秒數，以及算它的那次執行的 GPU 型號、torch、cuDNN |
| `<組>/dumps/_INPUT.json` | 這一組用的題目檔和它的 SHA-256 |
| `seam_check.json` | 接縫核對的數字和結果 |
| `readonly_before.json`、`readonly_after.json` | prereg01、prereg02 目錄每個檔的大小和 mtime（開頭、結尾） |
| `config.json`、`env.json`、`run_log.json`、`timings.json` | 設定、環境、每次執行的紀錄（`status`）、各階段秒數 |

`<組>` 是 `seam_val`、`public`、`private`。`MyDrive/C10/results/test02/test02_rank_results.zip`：上面全部檔案（zip 內路徑從 `rank/` 起），這個檔帶回本機。

## 中斷後怎麼接續

重新打開後，**選和中斷前同一種 GPU**（中斷前是 G4 就選 G4），再按一次「全部執行」：已完成的逐盤輸出靠完成標記、SHA-256 和 `_INPUT.json` 跳過；沒完成的重算。三組的逐盤輸出必須是同一個執行環境算的（GPU 型號、torch、cuDNN，記在各組的完成標記）：接續時換了 GPU（例如中斷前 G4、接續時 A100），或 Colab 換了 torch／cuDNN，接縫核對會判定沒過並停下，這時照 Claude 的指示先把 `C10/results/test02/rank` 改名成 `rank_failed_1`，再整份重算。中斷前那一種 GPU 拿不到時，先不要跑，告訴 Claude。

## 和 prereg02 notebook 的差異

底稿是已 commit 的 `colab/prereg02_rank.ipynb`（產生時核對 SHA-256）。沿用的格子沒列在下表的，和 prereg02 逐字相同；和官方 notebook、prereg01 notebook 的差異照 prereg02 的差異表。逐格的 unified diff 在 `results/test02/notebook_diff_rank.txt`。

| 格 | 改了什麼 |
|---|---|
__TABLE__

## 其他

- **本機冒煙測試**：`colab/run_local_smoke.py` 設定 `C10_SMOKE=1`，只用極少量資料在 CPU 上跑：要先有 `prereg01_smoke`、`prereg02_smoke` 和種類 `smoke` 的測試集 zip；每組只推論前 12 題，結果寫到 `results/test02_smoke/`。
- **不在這裡打分**：答案檔不上傳 Colab；測試集沒有答案。
""".replace("__TABLE__", CHANGE_TABLE)

# ---------------------------------------------------------------------------
# setup (rep)
# ---------------------------------------------------------------------------
_pre = base_src("preflight")
_pre_tree = ast.parse(_pre)
_pre_defs = {n.name: ast.get_source_segment(_pre, n) for n in _pre_tree.body if isinstance(n, ast.FunctionDef)}
MOVED_DEFS = _pre_defs["sha256_path"] + "\n\n\n" + _pre_defs["gpu_name_of"]

SETUP_NEW_DIRS = r"""# 本版（test02）：測試集 zip（Drive 上的 C10_test_v1.zip）。這裡只讀它的 tests/MANIFEST.json，決定這次是正式、彩排還是冒煙；
# 逐檔核對在 preflight 格
import zipfile

TEST_ZIP = os.path.join(DRIVE_DIR, 'C10_test_v1.zip')
TEST_MANIFEST_MEMBER = 'tests/MANIFEST.json'
TEST_MANIFEST_FORMAT = 'c10-test02 manifest v1'
OFFICIAL_TEST_NAMES = __NAMES__
TEST_FILES = {'public': 'rank_prediction_test_public.csv', 'private': 'rank_prediction_test_private.csv'}
TEST_HELP = ('處理方式：確認 MyDrive/C10/ 裡有 Claude 給你的 C10_test_v1.zip（檔名不能改、不要解壓），再按一次「全部執行」。'
             '還是不對就把這段訊息截圖給 Claude。')
if not os.path.isfile(TEST_ZIP):
    raise SystemExit('找不到測試集 zip：%s。%s' % (TEST_ZIP, TEST_HELP))
try:
    with zipfile.ZipFile(TEST_ZIP) as _z:
        TEST_MANIFEST = json.loads(_z.read(TEST_MANIFEST_MEMBER))
except (OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
    raise SystemExit('測試集 zip 讀不出 %s（%s: %s）。%s' % (TEST_MANIFEST_MEMBER, type(e).__name__, e, TEST_HELP))
TEST_KIND = TEST_MANIFEST.get('kind') if isinstance(TEST_MANIFEST, dict) else None
_KINDS_OK = ('smoke',) if SMOKE else ('official', 'rehearsal')
if not isinstance(TEST_MANIFEST, dict) or TEST_MANIFEST.get('format') != TEST_MANIFEST_FORMAT or TEST_KIND not in _KINDS_OK:
    raise SystemExit('測試集 zip 的 MANIFEST 不是這次能用的（格式 %r、種類 %r；這次只接受 %s）。%s'
                     % (TEST_MANIFEST.get('format') if isinstance(TEST_MANIFEST, dict) else None, TEST_KIND,
                        '、'.join(_KINDS_OK), TEST_HELP))
REHEARSAL = TEST_KIND == 'rehearsal'

# 結果目錄：正式、彩排、SMOKE 分開，絕不混用。本版（test02）只寫 test02（彩排 test02_rehearsal、SMOKE test02_smoke）
RESULT_ROOT = os.path.join(DRIVE_DIR, 'results',
                           'test02_smoke' if SMOKE else ('test02_rehearsal' if REHEARSAL else 'test02'))
RESULT_DIR = os.path.join(RESULT_ROOT, 'rank')

# 本版：prereg01、prereg02 的結果目錄只讀（prereg02 的 epoch_9.pth 是這次的權重，它的驗證題逐盤輸出是接縫參照）
P01_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg01_smoke' if SMOKE else 'prereg01')
P02_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg02_smoke' if SMOKE else 'prereg02')
READONLY_ROOTS = {'prereg01': P01_ROOT, 'prereg02': P02_ROOT}


def _real(path):
    return os.path.normcase(os.path.realpath(path))


def _is_under(path, root):
    p, r = _real(path), _real(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def guard_write(path):
    # 本版：寫檔、np.savez、zip、rmtree 的目的地一律先過這一關；落在 prereg01 或 prereg02 底下就停
    for _name, _root in READONLY_ROOTS.items():
        if _is_under(path, _root):
            raise PermissionError('本版不能寫入 %s 的目錄（只讀）：%s' % (_name, path))
    return path


def readonly_fingerprint(root):
    # 只讀目錄每個檔的 {相對路徑: [bytes, mtime_ns]}；開頭和結尾各記一次，結尾必須相同
    out = {}
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for n in sorted(files):
            p = os.path.join(base, n)
            st = os.stat(p)
            out[os.path.relpath(p, root).replace(os.sep, '/')] = [st.st_size, st.st_mtime_ns]
    return out


# 本版：這兩個函式原本在 prereg02 的 preflight 格（本版的 preflight 在 GPU 關卡之後，官方 2 就要用 gpu_name_of）
__MOVED__


# 本機工作目錄、test02 的結果目錄都不能和只讀目錄互相包含
for _p in (WORK, RESULT_ROOT):
    guard_write(_p)
    for _root in READONLY_ROOTS.values():
        assert not _is_under(_root, _p), (_root, _p)
guard_write(RESULT_DIR)
os.makedirs(RESULT_DIR, exist_ok=True)

""".replace("__NAMES__", json.dumps(OFFICIAL_TEST_NAMES)).replace("__MOVED__", MOVED_DEFS)

SETUP_NEW_DEV = r"""# 本版（test02）和 prereg02 notebook 的差異（與第一格的表同一份）
T02_CHANGES = __T02_CHANGES__

# 本版：這份 notebook 的 SHA-256（產生器寫入；規則見 NOTEBOOK_SHA256_RULE）
NOTEBOOK_REL = 'colab/test02_rank.ipynb'
__HASH_LINE__
NOTEBOOK_SHA256_RULE = ('sha256 of json.dumps([the source of every cell], ensure_ascii=False) with this value replaced by '
                        '64 zeros (colab/build_test02_rank_nb.py; tools/t02_rank.py can recompute it)')

""".replace("__T02_CHANGES__", json.dumps(T02_CHANGES, ensure_ascii=False, indent=4)).replace(
    "__HASH_LINE__", HASH_LINE % HASH_PLACEHOLDER)

SETUP_NEW_PREREG = """    'prereg': 'docs/PREREG_02_OPENING_FUSION.md 第 9 節（rev 5，測試日；競賽二）',
    'rehearsal': REHEARSAL, 'test_kind': TEST_KIND,
    'notebook': NOTEBOOK_REL, 'notebook_sha256': NOTEBOOK_SHA256, 'notebook_sha256_rule': NOTEBOOK_SHA256_RULE,
    'base_notebook': {'path': '__BASE_REL__', 'sha256': '__BASE_SHA__'},
    'changes_from_prereg02': T02_CHANGES,
    'readonly_roots': READONLY_ROOTS,
""".replace("__BASE_REL__", BASE_REL).replace("__BASE_SHA__", BASE_SHA256)

_setup = base_src("setup")
_setup = rep(_setup, cut(_setup, "RUN_START = time.time()\n", "\n\ndef now_iso():"), "RUN_START = time.time()\n")
_setup = rep(_setup, cut(_setup, "# 結果目錄：正式和 SMOKE 分開，絕不混用。", "# SMOKE 的縮小方式"), SETUP_NEW_DIRS)
_setup = rep(_setup, cut(_setup, "# 本版（prereg02）的 SMOKE 設定另外放", "# 和官方 notebook 的全部差異"), SETUP_NEW_DEV)
_setup = rep(_setup, cut(_setup, "    'prereg': 'docs/PREREG_02_OPENING_FUSION.md", "    'official_notebook':"), SETUP_NEW_PREREG)
_setup = rep(_setup, "run_log['runs'].append(RUN)\n",
             "run_log['runs'].append(RUN)\n"
             "run_log['status'] = None   # 本版：這次執行結束時才寫（completed／seam_failed／readonly_changed）\n")
_setup = rep(_setup, "print('prereg01（只讀）', P01_ROOT)",
             "print('只讀：prereg01', P01_ROOT, '| prereg02', P02_ROOT)\n"
             "print('測試集 zip：%s（種類 %s）' % (TEST_ZIP, TEST_KIND))")

# ---------------------------------------------------------------------------
# prepare (rep): no training data
# ---------------------------------------------------------------------------
_prep = base_src("prepare")
_prep = rep(_prep, "# 準備：缺套件才安裝、把打包檔複製到 WORK 解壓並逐檔核對 SHA-256、解開訓練資料\n",
            "# 準備：缺套件才安裝、把打包檔複製到 WORK 解壓並逐檔核對 SHA-256（本版不解開訓練資料）\n")
_prep = rep(_prep, cut(_prep, "# 訓練資料：tar.xz 解開後是", "# 打包檔裡的 c10data.py"),
            "# 本版（test02）：只推論，不解開訓練資料（底稿這裡解開 training/aicup2026_go_training.tar.xz 的整段拿掉）\n\n")

# ---------------------------------------------------------------------------
# official 2 (rep): no checkpoints, no training settings, GPU gate without the finish switch
# ---------------------------------------------------------------------------
OFF2_NEW_GATE = """DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 本版（test02）：正式執行一定要有 GPU（接縫核對要和 prereg02 在 GPU 上算的逐盤輸出比）
if not SMOKE and DEVICE.type != 'cuda':
    raise SystemExit('沒有 GPU，停在這裡（還沒開始算）。處理方式：選單「執行階段 → 變更執行階段類型」選 G4 GPU'
                     '（灰色時選 A100 GPU，「大量 RAM」關掉）→ 儲存，再按一次「全部執行」。')
"""
_off2 = base_src("official_2")
_off2 = rep(_off2, "CKPT_DIR = RESULT_CKPT\n", "# 本版（test02）：只推論，不存 checkpoint（拿掉 CKPT_DIR）\n")
_off2 = rep(_off2, cut(_off2, "# 本版（prereg02）：NUM_EPOCHS 維持", "# 本版：GPU 型號只接受 G4"), OFF2_NEW_GATE)
_off2 = rep(_off2, "guard_write(CKPT_DIR)\nos.makedirs(CKPT_DIR, exist_ok=True)\n", "")
_off2 = rep(_off2, "print('device', DEVICE, '| GPU', GPU_MODEL_NOW, '| NPROC', NPROC, '| batch', BATCH_SIZE,\n"
                   "      '| epochs', NUM_EPOCHS, '| extra epochs', EXTRA_EPOCHS, '| final epoch', FINAL_EPOCH)",
            "print('device', DEVICE, '| GPU', GPU_MODEL_NOW, '| NPROC', NPROC)")

# ---------------------------------------------------------------------------
# preflight (new)
# ---------------------------------------------------------------------------
PREFLIGHT = r"""# 本版新增（preflight，在 GPU 關卡之後）：權重、接縫參照、測試集 zip、題目檔格式、舊輸出、Drive 空間、只讀目錄的指紋。
# 任何一項不符就用中文說明處理方式後停下，還沒開始算（但執行階段開著就會扣運算單元）。
import io

t0 = time.time()
DUMP_NAME, DUMP_SYMS = 'p02e9_d4', 8
SETS = ('seam_val', 'public', 'private')
# 產生器從 repo 讀出的釘選值（results/prereg02/rank/frozen_config.json、dumps/ 的完成標記與檔案）
W_PIN = '__W_PIN__'
W_LABEL_PIN = '__W_LABEL__'
SEAM_REF_SHA256_PIN = __SEAM_REF__
P02_REF_GPU_PIN = '__P02_GPU__'
MIN_DRIVE_FREE_BYTES = 5 * 10 ** 8
N_TEST_QUESTIONS = 400
P02_RANK = os.path.join(P02_ROOT, 'rank')
P02_DUMPS = os.path.join(P02_RANK, 'dumps')
P02_HELP = ('處理方式：確認這個 Colab 用的是跑 prereg02 的同一個 Google 帳號，而且 Drive 上有 '
            'MyDrive/C10/results/prereg02/rank/ckpt/epoch_9.pth 和 rank/dumps/（val_keys.npz、val_p02e9_d4.npz 和它們的 _DONE_ 檔）。'
            '不要搬動、改名或修改 prereg02 的任何檔案。還是不對就把這段訊息截圖給 Claude。')

# ① 權重：prereg02 指定的 epoch_9.pth（SMOKE：prereg02_smoke 的完成標記記的那一份）
_m_ref = read_json(os.path.join(P02_DUMPS, '_DONE_val_%s.json' % DUMP_NAME))
if not _m_ref:
    raise SystemExit('找不到 prereg02 的 rank/dumps/_DONE_val_%s.json。%s' % (DUMP_NAME, P02_HELP))
if SMOKE:
    W_LABEL, WSHA = _m_ref['weights'], _m_ref['weights_sha256']
else:
    W_LABEL, WSHA = W_LABEL_PIN, W_PIN
    if (_m_ref.get('weights'), _m_ref.get('weights_sha256')) != (W_LABEL, WSHA):
        raise SystemExit('prereg02 的 rank/dumps/_DONE_val_%s.json 記的權重不是 %s。%s' % (DUMP_NAME, W_LABEL, P02_HELP))
if not W_LABEL.startswith('prereg02/'):
    raise SystemExit('prereg02 的完成標記記的權重路徑 %r 不在 prereg02 底下。請截圖給 Claude。' % W_LABEL)
W_PATH = os.path.join(P02_ROOT, *W_LABEL[len('prereg02/'):].split('/'))
if not os.path.isfile(W_PATH):
    raise SystemExit('找不到權重 %s。%s' % (W_PATH, P02_HELP))
if sha256_path(W_PATH) != WSHA:
    raise SystemExit('權重 %s 的 SHA-256 不是登記的 %s…。%s' % (W_PATH, WSHA[:12], P02_HELP))

# ② 接縫參照：prereg02 在 Drive 上的本機驗證題逐盤輸出（正式執行釘 SHA-256；SMOKE 核對完成標記）
SEAM_REF = {}
for _n in ('val_keys.npz', 'val_%s.npz' % DUMP_NAME):
    _p = os.path.join(P02_DUMPS, _n)
    if not os.path.isfile(_p):
        raise SystemExit('找不到接縫參照 %s。%s' % (_p, P02_HELP))
    SEAM_REF[_n] = sha256_path(_p)
    _want = ((read_json(os.path.join(P02_DUMPS, '_DONE_%s.json' % _n[:-4])) or {}).get('sha256') if SMOKE
             else SEAM_REF_SHA256_PIN[_n])
    if SEAM_REF[_n] != _want:
        raise SystemExit('接縫參照 %s 的 SHA-256 不符。%s' % (_p, P02_HELP))
with np.load(os.path.join(P02_DUMPS, 'val_%s.npz' % DUMP_NAME), allow_pickle=False) as _z:
    if str(_z['weights_sha256']) != WSHA:
        raise SystemExit('接縫參照 val_%s.npz 記的權重不是 %s…。%s' % (DUMP_NAME, WSHA[:12], P02_HELP))
P02_REF_GPU = _m_ref.get('gpu_name')   # 接縫核對的 GPU 規則：和算這次 seam_val dump 的 GPU 比
if not SMOKE and P02_REF_GPU != P02_REF_GPU_PIN:
    raise SystemExit('prereg02 的完成標記記的 GPU 是 %r，不是登記時的 %s。%s' % (P02_REF_GPU, P02_REF_GPU_PIN, P02_HELP))

# ③ 測試集 zip：只收 tests/ 底下的 6 個官方檔名和 MANIFEST.json，解到 WORK/tests；每個檔的 SHA-256、大小要等於 MANIFEST
TESTS_DIR = os.path.join(WORK, 'tests')
_allowed = {'tests/' + n for n in OFFICIAL_TEST_NAMES} | {TEST_MANIFEST_MEMBER}
shutil.rmtree(TESTS_DIR, ignore_errors=True)
os.makedirs(TESTS_DIR)
with zipfile.ZipFile(TEST_ZIP) as _z:
    _members = [i.filename for i in _z.infolist() if not i.filename.endswith('/')]
    _bad = [m for m in _members if m not in _allowed]
    _absent = sorted(_allowed - set(_members))
    if _bad or _absent or len(_members) != len(set(_members)):
        raise SystemExit('測試集 zip 的內容不對（多出：%s；缺少：%s）。%s' % (_bad[:5], _absent, TEST_HELP))
    for _m in _members:
        with open(os.path.join(TESTS_DIR, _m[len('tests/'):]), 'wb') as _f:
            _f.write(_z.read(_m))
TEST_ZIP_SHA = sha256_path(TEST_ZIP)
TEST_MANIFEST_SHA = sha256_path(os.path.join(TESTS_DIR, 'MANIFEST.json'))
if read_json(os.path.join(TESTS_DIR, 'MANIFEST.json')) != TEST_MANIFEST:
    raise SystemExit('測試集 zip 的 MANIFEST 和第一格讀到的不同。%s' % TEST_HELP)
_mf = TEST_MANIFEST.get('files') or {}
_problems = []
if sorted(_mf) != sorted(OFFICIAL_TEST_NAMES):
    _problems.append('MANIFEST 列的檔名不是 6 個官方檔名：%s' % sorted(_mf))
for _n in OFFICIAL_TEST_NAMES:
    _p = os.path.join(TESTS_DIR, _n)
    _e = _mf.get(_n) or {}
    if sha256_path(_p) != _e.get('sha256') or os.path.getsize(_p) != _e.get('bytes'):
        _problems.append('%s 的 SHA-256 或大小和 MANIFEST 不同' % _n)
if _problems:
    raise SystemExit('測試集 zip 核對沒過：%s。%s' % ('；'.join(_problems), TEST_HELP))


def quick_check_questions(path, prefix):
    # T1 的一部分（競賽二的題目檔）：沒有 BOM、400 題、question_id 不重複且前綴對、num_games 是 5–20 的整數、
    # sgf_1..20／color_1..20 欄都在、剛好前 num_games 格有棋、這幾格的執色是 B/W。回傳 (問題清單, question_id 清單)
    problems = []
    with open(path, 'rb') as f:
        raw = f.read()
    if raw.startswith(b'\xef\xbb\xbf'):
        problems.append('檔頭有 BOM')
    csv.field_size_limit(10 ** 8)
    rd = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    need = ['question_id', 'num_games'] + ['sgf_%d' % i for i in range(1, 21)] + ['color_%d' % i for i in range(1, 21)]
    missing = [c for c in need if c not in (rd.fieldnames or [])]
    if missing:
        return problems + ['缺欄位 %s' % missing[:5]], []
    ids = []
    for r in rd:
        q = r['question_id']
        ids.append(q)
        if not q.startswith(prefix):
            problems.append('%s 的前綴不是 %s' % (q, prefix))
        g = (r['num_games'] or '').strip()
        if not g.isdigit() or not 5 <= int(g) <= 20:
            problems.append('%s 的 num_games 是 %r' % (q, r['num_games']))
            continue
        n = int(g)
        filled = [i for i in range(1, 21) if (r['sgf_%d' % i] or '').strip()]
        if filled != list(range(1, n + 1)):
            problems.append('%s：有棋的格是 %s，num_games 是 %d' % (q, filled[:3], n))
        bad = [i for i in range(1, n + 1) if r['color_%d' % i] not in ('B', 'W')]
        if bad:
            problems.append('%s：第 %s 格的執色不是 B/W' % (q, bad[:3]))
    if len(ids) != N_TEST_QUESTIONS:
        problems.append('%d 題，應該是 %d 題' % (len(ids), N_TEST_QUESTIONS))
    if len(set(ids)) != len(ids):
        problems.append('question_id 有重複')
    return problems, ids


_ids = {}
for _set, _prefix in (('public', 'pub_'), ('private', 'priv_')):
    _pr, _ids[_set] = quick_check_questions(os.path.join(TESTS_DIR, TEST_FILES[_set]), _prefix)
    _problems += ['%s：%s' % (TEST_FILES[_set], x) for x in _pr[:10]]
if set(_ids['public']) & set(_ids['private']):
    _problems.append('public 和 private 有相同的 question_id')
if _problems:
    raise SystemExit('測試題目檔的格式檢查沒過（%d 項）：%s。處理方式：不要改檔，把這段訊息截圖給 Claude。'
                     % (len(_problems), '；'.join(_problems[:10])))

# ④-0 上一次執行的接縫核對沒過就停：已完成的逐盤輸出會被 dump_is_done 沿用，登記第 9 節第 2 條的
#     「換一個執行階段重跑一次」就變成拿同一份輸出再比一次。要重跑必須重算，所以先把整個結果資料夾改名保留。
_prev_seam = read_json(os.path.join(RESULT_DIR, 'seam_check.json'))
if _prev_seam is not None and _prev_seam.get('passed') is not True:
    raise SystemExit('上一次執行的接縫核對沒過（%s 記 passed = %r）。這個資料夾裡的逐盤輸出不能沿用，重跑必須重算。'
                     '處理方式：先把畫面截圖給 Claude；Claude 說要重跑時，在 Google 雲端硬碟把 %s 整個資料夾'
                     '改名成 rank_failed_1（同一層，數字照 Claude 給的），再按一次「全部執行」。'
                     % (os.path.join(RESULT_DIR, 'seam_check.json'), _prev_seam.get('passed'),
                        os.path.relpath(RESULT_DIR, os.path.dirname(DRIVE_DIR)).replace(os.sep, '/')))

# ④ 各組的題目檔與 _INPUT.json：已存在但不同就停（dump_is_done 只看列數，換過的題目檔會被當成已完成）
SET_QUESTIONS = {'seam_val': RANK_QUESTIONS, 'public': os.path.join(TESTS_DIR, TEST_FILES['public']),
                 'private': os.path.join(TESTS_DIR, TEST_FILES['private'])}
SET_INPUT, SET_ROWS = {}, {}
for _set in SETS:
    _q = SET_QUESTIONS[_set]
    with open(_q, newline='', encoding='utf-8-sig') as _f:
        _n_rows = sum(1 for _ in csv.DictReader(_f))
    SET_ROWS[_set] = min(_n_rows, SMOKE_CFG['questions']) if SMOKE else _n_rows
    _rec = {'set': _set, 'question_file': ('bundle local/' if _set == 'seam_val' else 'tests/') + os.path.basename(_q),
            'sha256': sha256_path(_q)}
    _dir = os.path.join(RESULT_DIR, _set, 'dumps')
    _ip = os.path.join(_dir, '_INPUT.json')
    _old = read_json(_ip)
    if _old is not None and _old != _rec:
        raise SystemExit('%s 已有逐盤輸出，但當時的題目檔和這次不同（%s）。不能沿用。處理方式：不要刪檔，把這段訊息截圖給 Claude。'
                         % (_set, _ip))
    if _old is None and os.path.isdir(_dir) and os.listdir(_dir):
        raise SystemExit('%s 有檔案但沒有 _INPUT.json，不知道是用哪個題目檔算的。處理方式：不要刪檔，把這段訊息截圖給 Claude。' % _dir)
    guard_write(_dir)
    os.makedirs(_dir, exist_ok=True)
    if _old is None:
        write_json(_ip, _rec)
    SET_INPUT[_set] = _rec

# ⑤ Drive 剩餘空間
try:
    DRIVE_FREE_BYTES = shutil.disk_usage(DRIVE_DIR).free
except OSError as e:
    DRIVE_FREE_BYTES = None
    print('注意：查不到 Drive 剩餘空間（%s: %s），請自己確認還有 0.5 GB 以上' % (type(e).__name__, e))
if DRIVE_FREE_BYTES is not None and DRIVE_FREE_BYTES < MIN_DRIVE_FREE_BYTES:
    raise SystemExit('Drive 只剩 %.2f GB，這份 notebook 要 0.5 GB 以上。處理方式：清出空間（Drive 的垃圾桶也要清空）後再按一次'
                     '「全部執行」。' % (DRIVE_FREE_BYTES / 1e9))

# ⑥ 只讀目錄的指紋（開頭）。重跑時必須和第一次執行記下的相同：不同代表 prereg01／prereg02 在兩次執行之間被改過
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

save_config(preflight={'weights': W_LABEL, 'weights_sha256': WSHA, 'seam_reference_sha256': SEAM_REF,
                       'p02_reference_gpu': P02_REF_GPU, 'test_zip_sha256': TEST_ZIP_SHA,
                       'test_manifest_sha256': TEST_MANIFEST_SHA, 'test_kind': TEST_KIND, 'sets': SET_INPUT,
                       'drive_free_bytes': DRIVE_FREE_BYTES,
                       'readonly_files_fingerprinted': {k: len(v) for k, v in READONLY_BEFORE.items()}})
save_config_summary(gpu=GPU_MODEL_NOW, torch=torch.__version__, cudnn=ENV_INFO.get('cudnn'),
                    weights={'path': W_LABEL, 'sha256': WSHA},
                    test_bundle={'zip_sha256': TEST_ZIP_SHA, 'manifest_sha256': TEST_MANIFEST_SHA,
                                 'kind': TEST_KIND, 't1': TEST_MANIFEST.get('t1'),
                                 'files': {n: _mf[n]['sha256'] for n in OFFICIAL_TEST_NAMES}},
                    sets={s: {'questions_sha256': SET_INPUT[s]['sha256'], 'rows': SET_ROWS[s]} for s in SETS},
                    seam_reference={'gpu_name': P02_REF_GPU, 'sha256': SEAM_REF})
mark_time('preflight', time.time() - t0)
print('preflight 通過：權重 %s（%s…），測試集 zip %s（%s…），題目 %s，prereg02 的 GPU：%s'
      % (W_LABEL, WSHA[:12], TEST_KIND, TEST_ZIP_SHA[:12], SET_ROWS, P02_REF_GPU))
""".replace("__W_PIN__", WEIGHTS_SHA256).replace("__W_LABEL__", WEIGHTS_LABEL).replace(
    "__SEAM_REF__", json.dumps(SEAM_REF_SHA256)).replace("__P02_GPU__", P02_GPU)

# ---------------------------------------------------------------------------
# dump tools (base + rep)
# ---------------------------------------------------------------------------
DT_NEW_HEAD = """# 本版新增：逐盤輸出（dump）工具（test02：沿用 prereg02 的工具，拿掉 inner 的部分）
# - 題目檔（本機驗證題或官方測試題）：照題目檔的列順序和 slot 1..num_games 解析，失敗的保留位置（logits 填 NaN）；
#   每題一個 batch，和 prereg01、prereg02 的推論相同，接縫核對才對得上
# - 每個檔先寫 .tmp 再改名，附 _DONE_<檔名>.json；重跑時完成標記、SHA-256、權重 SHA-256 都對得上就跳過
# - DUMP_DIR 由各組的格子在呼叫前指定（<組>/dumps）；底稿的函式把它當全域變數讀，本身一字不改
"""
DT_NEW_RECORD = """def record_readonly_after():
    # 本版：兩個只讀目錄（prereg01、prereg02）的指紋（結尾，或接縫核對沒過時）：和這次開頭記的比，寫進 readonly_after.json；
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
_dt = base_src("dump_tools")
_dt = rep(_dt, cut(_dt, "# 本版新增：逐盤輸出（dump）工具\n", "import io\n"), DT_NEW_HEAD)
_dt = rep(_dt, "DUMP_DIR = os.path.join(RESULT_DIR, 'dumps')\nguard_write(DUMP_DIR)\nos.makedirs(DUMP_DIR, exist_ok=True)\n",
          "DUMP_DIR = None                        # 本版：各組的格子在呼叫前指定（<組>/dumps）\n")
_dt = rep(_dt, cut(_dt, "\nEXPECTED_INNER_GAMES = ", "\n\n\ndef d4(x, s):"), "")
_dt = rep(_dt, cut(_dt, "def record_p01_after():", "def pack_results(zip_name):"), DT_NEW_RECORD)
_dt = rep(_dt, cut(_dt, "# inner 的 key：df_val 的列順序", "_infer_parser = SGFParseRankPrediction()"), "")
_dt = rep(_dt, "VAL = None   # parse_val() 的結果，整份 notebook 共用（只解析一次）\n",
          "QSETS = {}   # 本版：parse_questions() 的結果，依題目檔路徑快取（每個檔只解析一次；各組的格子用完就釋放）\n")
_dt = rep(_dt, """def parse_val():
    # 本機驗證題：題目檔的列順序、slot 1..num_games；SMOKE 時和 prereg01 冒煙一樣只取前幾題
    global VAL
    if VAL is not None:
        return VAL
    t0 = time.time()
    test = pd.read_csv(RANK_QUESTIONS, dtype={'question_id': str})
""", """def parse_questions(path):
    # 本版：題目檔（本機驗證題或官方測試題）：題目檔的列順序、slot 1..num_games；SMOKE 時和 prereg01 冒煙一樣只取前幾題
    if path in QSETS:
        return QSETS[path]
    t0 = time.time()
    test = pd.read_csv(path, dtype={'question_id': str})
""")
_dt = rep(_dt, """    VAL = {'spans': spans, 'feats': feats,
           'keys': {'question_id': np.array(qids, dtype=str), 'slot': np.array(slots, dtype=np.int32),
                    'color': np.array(colors, dtype=str), 'parsed': parsed}}
    mark_time('parse_val', time.time() - t0)
    print('本機驗證題：%d 題、%d 盤，解析失敗 %d 盤，%.0f 秒'
          % (len(spans), len(feats), int((~parsed).sum()), time.time() - t0), flush=True)
    return VAL
""", """    QSETS[path] = {'spans': spans, 'feats': feats,
                   'keys': {'question_id': np.array(qids, dtype=str), 'slot': np.array(slots, dtype=np.int32),
                            'color': np.array(colors, dtype=str), 'parsed': parsed}}
    mark_time('parse_questions %s' % os.path.basename(path), time.time() - t0)
    print('題目檔 %s：%d 題、%d 盤，解析失敗 %d 盤，%.0f 秒'
          % (os.path.basename(path), len(spans), len(feats), int((~parsed).sum()), time.time() - t0), flush=True)
    return QSETS[path]
""")
_dt = rep(_dt, "def dump_val(net, name, syms, weights, weights_sha):\n    V = parse_val()\n",
          "def dump_questions(net, path, name, syms, weights, weights_sha):\n"
          "    # 本版：底稿的 dump_val 改成指定題目檔；檔名仍是 'val_' + 名稱（在 test02 裡只代表「一組題目的逐盤輸出」），"
          "寫到目前的 DUMP_DIR\n"
          "    V = parse_questions(path)\n")
_dt = rep(_dt, "print('dump 工具就緒：inner %d 盤（worker %d）' % (len(df_val), workers))",
          "NET = None      # 本版：三組共用的網路，第一組的格子載入一次\n"
          "SET_LOG = {}    # 本版：每組的逐盤輸出紀錄（寫進 config.json）\n"
          "print('dump 工具就緒：權重 %s（SHA-256 %s…），%d 種方向' % (W_LABEL, WSHA[:12], NUM_SYMS))")

# ---------------------------------------------------------------------------
# the three set cells (new; identical bodies)
# ---------------------------------------------------------------------------
SET_FIRST_LINE = "SET = '%s'   # 本版新增：這一格處理的題目組（seam_val、public、private 三格只有這一行不同）"
SET_BODY = r"""# 一組題目的逐盤輸出（prereg02 的 epoch_9.pth，8 種對稱），寫到 <組>/dumps；已完成就跳過
t0 = time.time()
_path = SET_QUESTIONS[SET]
DUMP_DIR = guard_write(os.path.join(RESULT_DIR, SET, 'dumps'))   # 底稿的 dump 函式把 DUMP_DIR 當全域變數讀
os.makedirs(DUMP_DIR, exist_ok=True)
if read_json(os.path.join(DUMP_DIR, '_INPUT.json')) != SET_INPUT[SET]:
    raise SystemExit('%s 的 _INPUT.json 和 preflight 記的不同。請把這段訊息截圖給 Claude。' % SET)
if NET is None:
    NET, _ = load_net(W_PATH, expect_sha=WSHA)
_V = parse_questions(_path)
_st = [write_keys('val_keys', _V['keys']), dump_questions(NET, _path, DUMP_NAME, DUMP_SYMS, W_LABEL, WSHA)]
SET_LOG[SET] = {'questions': len(_V['spans']), 'games': len(_V['feats']),
                'games_unparsed': int((~_V['keys']['parsed']).sum()), 'keys': _st[0], 'dump': _st[1],
                'seconds': time.time() - t0}
del _V
QSETS.pop(_path, None)   # 解析結果用完就釋放記憶體
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
save_config(dumps=SET_LOG)
mark_time('dump_%s' % SET, time.time() - t0)
print('%s：%s' % (SET, SET_LOG[SET]))"""

# ---------------------------------------------------------------------------
# seam check (new)
# ---------------------------------------------------------------------------
SEAM = r"""# 本版新增：接縫核對（PREREG_02 第 9 節第 2 條）：這次重算的 seam_val 逐盤輸出，和 Drive 上 prereg02 的比
# 通過條件：key 陣列完全相同；原方向逐盤 softmax 的每題 float32 平均（C0）最大差 ≤ tol（算這份 dump 的 GPU 型號和
# prereg02 相同時 1e-4，不同時 1e-3）；規則 B（C0）不同的題 ≤ 4（SMOKE 依題數等比例，至少 1）
t0 = time.time()
SEAM_KEYS = ('question_id', 'slot', 'color', 'parsed')


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def question_spans(qids):
    # key 陣列的題目區段 [(question_id, 開始列, 盤數)]；同一題的 slot 必須相鄰
    out, seen = [], set()
    for i, q in enumerate(qids.tolist()):
        if out and out[-1][0] == q:
            out[-1][2] += 1
            continue
        assert q not in seen, ('同一題的 slot 不相鄰', q)
        seen.add(q)
        out.append([q, i, 1])
    return out


def c0_of(L, spans, parsed):
    # C0：每盤原方向 softmax、同題在 float32 下平均（和 prereg01 推論、prereg02 的重現核對相同）；沒有可解析的盤是 None
    out = []
    for _q, start, k in spans:
        ok = [start + j for j in range(k) if parsed[start + j]]
        out.append([float(v) for v in torch.from_numpy(L[ok, 0, :]).softmax(1).numpy().mean(0)] if ok else None)
    return out


def seam_compare(new, ref, same_gpu, smoke):
    # new、ref：{'keys': {...}, 'logits': 陣列}。回傳紀錄；'passed' 是登記的判定
    keys_equal = {k: bool(new['keys'][k].shape == ref['keys'][k].shape and np.array_equal(new['keys'][k], ref['keys'][k]))
                  for k in SEAM_KEYS}
    L1, L0 = new['logits'], ref['logits']
    rec = {'keys_equal': keys_equal, 'same_gpu_as_prereg02': bool(same_gpu), 'tol': 1e-4 if same_gpu else 1e-3,
           'bitwise_equal': bool(all(keys_equal.values()) and L1.dtype == L0.dtype and L1.shape == L0.shape
                                 and L1.tobytes() == L0.tobytes()),
           'max_abs_logit_diff': None, 'max_abs_c0_diff': None, 'problems': []}
    if not all(keys_equal.values()) or L1.shape != L0.shape:
        rec['problems'].append('key 陣列或 logits 的形狀和 prereg02 不同：%s' % keys_equal)
        rec['passed'] = False
        return rec
    parsed = new['keys']['parsed'].astype(bool)
    rec['max_abs_logit_diff'] = (float(np.max(np.abs(L1[parsed].astype(np.float64) - L0[parsed].astype(np.float64))))
                                 if parsed.any() else 0.0)
    spans = question_spans(new['keys']['question_id'])
    c1, c0 = c0_of(L1, spans, parsed), c0_of(L0, spans, parsed)
    diffs = [max(abs(a - b) for a, b in zip(x, y)) for x, y in zip(c1, c0) if x is not None]
    rec['max_abs_c0_diff'] = max(diffs) if diffs else 0.0
    b_diff = [s[0] for s, x, y in zip(spans, c1, c0) if x is not None and rule_b(x) != rule_b(y)]
    b_allowed = 4 if not smoke else max(1, math.ceil(4 * len(spans) / 1589))
    rec.update(questions=len(spans), games=int(len(parsed)), games_parsed=int(parsed.sum()),
               rule_b_diff={'questions': len(b_diff), 'allowed': b_allowed, 'question_ids': b_diff[:50]})
    rec['passed'] = bool(rec['max_abs_c0_diff'] <= rec['tol'] and len(b_diff) <= b_allowed)
    return rec


_new_dir = os.path.join(RESULT_DIR, 'seam_val', 'dumps')
_new = {'keys': _load_npz(os.path.join(_new_dir, 'val_keys.npz')),
        'logits': _load_npz(os.path.join(_new_dir, 'val_%s.npz' % DUMP_NAME))['logits']}
_ref = {'keys': _load_npz(os.path.join(P02_DUMPS, 'val_keys.npz')),
        'logits': _load_npz(os.path.join(P02_DUMPS, 'val_%s.npz' % DUMP_NAME))['logits']}
_m_new = read_json(os.path.join(_new_dir, '_DONE_val_%s.json' % DUMP_NAME)) or {}
SEAM_DUMP_GPU = _m_new.get('gpu_name')   # 算這份 seam_val dump 的那次執行的 GPU（接續時可能不是這次）
_same_gpu = SEAM_DUMP_GPU == P02_REF_GPU and (SMOKE or SEAM_DUMP_GPU is not None)
SEAM = seam_compare(_new, _ref, _same_gpu, SMOKE)
del _new, _ref
# 三組的逐盤輸出必須是同一個執行環境算的：中斷後接續時，已完成的組會被 dump_is_done 沿用，public、private 可能是
# 別次執行（別的 GPU、torch、cuDNN）算的，接縫就沒涵蓋產生測試輸出的環境（登記第 9 節第 2 條）。不同就算接縫沒過
SEAM_ENV_KEYS = ('gpu_name', 'torch', 'cudnn')
SEAM_DUMP_ENV = {_s: {_k: (read_json(os.path.join(RESULT_DIR, _s, 'dumps', '_DONE_val_%s.json' % DUMP_NAME)) or {}).get(_k)
                      for _k in SEAM_ENV_KEYS} for _s in SETS}
_env_mixed = [_s for _s in SETS if SEAM_DUMP_ENV[_s] != SEAM_DUMP_ENV['seam_val']]
if _env_mixed:
    SEAM['problems'].append('%s 的逐盤輸出和 seam_val 的不是同一個執行環境算的（GPU 型號、torch、cuDNN：%s）'
                            % ('、'.join(_env_mixed), SEAM_DUMP_ENV))
    SEAM['passed'] = False
SEAM.update({
    'what': 'test02 notebook seam (PREREG_02 section 9 item 2): seam_val dump vs prereg02 val_p02e9_d4.npz on Drive',
    'run': RUN['run'], 'time': now_iso(), 'smoke': SMOKE,
    'tol_rule': '1e-4 when the GPU that wrote the seam_val dump (its _DONE_ marker) is the prereg02 dump GPU, else 1e-3',
    'gpu_this_run': GPU_MODEL_NOW, 'torch_this_run': torch.__version__, 'cudnn_this_run': ENV_INFO.get('cudnn'),
    'seam_dump': {k: _m_new.get(k) for k in ('gpu_name', 'torch', 'cudnn', 'run', 'sha256', 'weights_sha256')},
    'prereg02_dump': {k: _m_ref.get(k) for k in ('gpu_name', 'torch', 'cudnn', 'run', 'sha256', 'weights_sha256')},
    'reference_sha256': SEAM_REF,
    'dump_env': SEAM_DUMP_ENV, 'dump_env_same': not _env_mixed,
    'dump_env_rule': 'the val_p02e9_d4 dumps of seam_val, public and private must carry the same gpu_name, torch and '
                     'cudnn in their _DONE markers (a resumed run reuses finished sets)'})
_seam_path = os.path.join(RESULT_DIR, 'seam_check.json')
_prev = read_json(_seam_path)
SEAM['earlier_runs'] = ((_prev.get('earlier_runs') or []) + [{k: v for k, v in _prev.items() if k != 'earlier_runs'}]
                        if _prev else [])
write_json(_seam_path, SEAM)
save_config(seam_check={k: SEAM.get(k) for k in ('passed', 'tol', 'bitwise_equal', 'max_abs_c0_diff',
                                                 'max_abs_logit_diff', 'rule_b_diff', 'same_gpu_as_prereg02',
                                                 'dump_env_same')})
mark_time('seam_check', time.time() - t0)
print('接縫核對：C0 最大差 %s（容許 %g；算 dump 的 GPU %s，prereg02 %s），規則 B 不同 %s，逐位元組相同 %s → %s'
      % (SEAM['max_abs_c0_diff'], SEAM['tol'], SEAM_DUMP_GPU, P02_REF_GPU, SEAM.get('rule_b_diff'),
         SEAM['bitwise_equal'], '通過' if SEAM['passed'] else '沒過'))
if not SEAM['passed']:
    # 停下前照樣比對只讀目錄的指紋、打包（查原因要用），並把 Drive 的寫入送出
    _after, _ro_changed = record_readonly_after()
    RUN.update(end=now_iso(), status='seam_failed')
    run_log['status'] = 'seam_failed'
    save_config_summary(status='seam_failed')
    save_run_log()
    _zip, _ = pack_results('test02_rank_results.zip')
    flush_drive()
    raise SystemExit('接縫核對沒過%s：這次執行的測試輸出一律不用、不上傳（PREREG_02 第 9 節第 2 條）。處理方式：不要自己重跑；'
                     '下載 %s，連同這段訊息的截圖一起給 Claude。'
                     % (('（%s）' % '；'.join(SEAM['problems'])) if SEAM['problems'] else '', _zip))"""

# ---------------------------------------------------------------------------
# final (rewrite of the base final_check)
# ---------------------------------------------------------------------------
FINAL = r"""# 本版改寫的結尾：核對三組的逐盤輸出，比對兩個只讀目錄前後的指紋，補完 config／timings／run_log，
# 打包 test02_rank_results.zip（不含權重），把 Drive 的寫入送出
t0 = time.time()
DUMP_CHECK = {}
for SET in SETS:
    _dir = os.path.join(RESULT_DIR, SET, 'dumps')
    _chk = {}
    for _stem in ('val_keys', 'val_' + DUMP_NAME):
        _path, _m = os.path.join(_dir, _stem + '.npz'), read_json(os.path.join(_dir, '_DONE_%s.json' % _stem))
        _sha = sha256_path(_path) if os.path.exists(_path) else None
        assert _m is not None and _sha == _m['sha256'], ('完成標記不符', SET, _stem)
        assert os.path.getsize(_path) < DUMP_MAX_BYTES, (SET, _stem)
        _chk[_stem] = {'rows': _m['rows'], 'bytes': os.path.getsize(_path), 'sha256': _sha,
                       'gpu_name': _m.get('gpu_name'), 'torch': _m.get('torch'), 'cudnn': _m.get('cudnn'),
                       'run': _m.get('run')}
    with np.load(os.path.join(_dir, 'val_keys.npz'), allow_pickle=False) as _z:
        _keys = {k: _z[k] for k in _z.files}
    _rows = len(_keys['question_id'])
    assert sorted(_keys) == ['color', 'parsed', 'question_id', 'slot'] and all(len(v) == _rows for v in _keys.values()), SET
    _parsed = _keys['parsed'].astype(bool)
    with np.load(os.path.join(_dir, 'val_%s.npz' % DUMP_NAME), allow_pickle=False) as _z:
        _L = _z['logits']
        assert _L.dtype == np.float32 and _L.shape == (_rows, DUMP_SYMS, NUM_CLASSES), (SET, _L.shape)
        assert str(_z['weights_sha256']) == WSHA and int(_z['syms']) == DUMP_SYMS, SET
        assert np.array_equal(np.isnan(_L).all(axis=(1, 2)), ~_parsed) and not np.isnan(_L[_parsed]).any(), SET
    assert _rows == _chk['val_keys']['rows'] == _chk['val_' + DUMP_NAME]['rows'], SET
    assert read_json(os.path.join(_dir, '_INPUT.json')) == SET_INPUT[SET], SET
    _qids = _keys['question_id'].tolist()
    DUMP_CHECK[SET] = {'questions': len(set(_qids)), 'games': _rows, 'games_unparsed': int((~_parsed).sum()),
                       'files': _chk, 'input': SET_INPUT[SET]}
    assert DUMP_CHECK[SET]['questions'] == SET_ROWS[SET], (SET, DUMP_CHECK[SET]['questions'], SET_ROWS[SET])
    del _keys, _L

# 只讀目錄的指紋（結尾）：必須和這次開頭記的完全相同
READONLY_AFTER, _ro_changed = record_readonly_after()
STATUS = 'completed' if not _ro_changed else 'readonly_changed'
OUTPUTS = {
    'dumps': DUMP_CHECK, 'seam_check_passed': SEAM['passed'], 'seam_bitwise_equal': SEAM['bitwise_equal'],
    'readonly_unchanged': not _ro_changed, 'readonly_changed_files': _ro_changed,
    'files': {n: os.path.getsize(os.path.join(RESULT_DIR, n)) for n in sorted(os.listdir(RESULT_DIR))
              if os.path.isfile(os.path.join(RESULT_DIR, n))},
}
RUN.update(end=now_iso(), status=STATUS)
run_log['status'] = STATUS
save_config(outputs=OUTPUTS, end=RUN['end'])
save_config_summary(status=STATUS)
save_run_log()
mark_time('final_check', time.time() - t0)
mark_time('total_this_run', time.time() - RUN_START)
print(json.dumps({k: v for k, v in OUTPUTS.items() if k != 'dumps'}, ensure_ascii=False, indent=1))
print(json.dumps({s: {k: v for k, v in d.items() if k in ('questions', 'games', 'games_unparsed')}
                  for s, d in DUMP_CHECK.items()}, ensure_ascii=False))

# 結果打包：結果目錄裡除了權重和暫存檔以外的檔案，打成 test02/test02_rank_results.zip（zip 內路徑從 rank/ 起）
RESULTS_ZIP, _skipped_big = pack_results('test02_rank_results.zip')
with zipfile.ZipFile(RESULTS_ZIP) as _zf:
    _names = set(_zf.namelist())
_need = (['rank/' + n for n in ('config.json', 'run_log.json', 'env.json', 'timings.json', 'seam_check.json',
                                'readonly_before.json', 'readonly_after.json')]
         + ['rank/%s/dumps/%s' % (s, n) for s in SETS
            for n in ('val_keys.npz', 'val_%s.npz' % DUMP_NAME, '_DONE_val_keys.json', '_DONE_val_%s.json' % DUMP_NAME,
                      '_INPUT.json')])
_missing_in_zip = [n for n in _need if n not in _names]
assert not _missing_in_zip and not _skipped_big, ('沒進 zip', _missing_in_zip, _skipped_big)
_zip_mb = os.path.getsize(RESULTS_ZIP) / 1e6
flush_drive()
if _ro_changed:
    raise SystemExit('prereg01 或 prereg02 的目錄在這次執行中被改動（%s）。結果已打包在 %s，但先不要用；'
                     '把這段訊息截圖給 Claude。' % ({k: v[:5] for k, v in _ro_changed.items()}, RESULTS_ZIP))
print('完成。請下載這個檔回傳給 Claude（合併、校準、交件檔都在本機做）：%s（%.1f MB）' % (RESULTS_ZIP, _zip_mb))"""

# ---------------------------------------------------------------------------
# Assemble. Each entry: (kind, key or tag, source)
# ---------------------------------------------------------------------------
PLAN = [
    ("md_new", "header", MD_HEADER),
    ("base", "setup", _setup),
    ("base", "prepare", _prep),
    ("base", "official_1", base_src("official_1")),
    ("base", "environment", base_src("environment")),
    ("base", "official_2", _off2),
    ("code_new", "preflight_t02", PREFLIGHT),
    ("base", "dump_tools", _dt),
] + [("code_new", "dump_set_" + s, SET_FIRST_LINE % s + "\n" + SET_BODY) for s in SETS] + [
    ("code_new", "seam_check", SEAM),
    ("rewrite", "final_check", FINAL),
]

nb = nbformat.v4.new_notebook()
nb.metadata = json.loads(json.dumps(base["metadata"]))
_used = set()
for k, (kind, key, src) in enumerate(PLAN):
    src = src.rstrip("\n")
    if kind == "md_new":
        cell = nbformat.v4.new_markdown_cell(src)
    else:
        compile(src, "<cell %d>" % k, "exec")   # plain Python only (the smoke runner compiles whole cells too)
        cell = nbformat.v4.new_code_cell(src)
    cell["id"] = "c10-rank-t02-%02d" % k
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

# notebook_sha256 (a notebook cannot hold the SHA-256 of its own file): see HASH_LINE and NOTEBOOK_SHA256_RULE
_hash_cells = [c for c in nb.cells if (HASH_LINE % HASH_PLACEHOLDER) in c["source"]]
assert len(_hash_cells) == 1
NB_SHA = nb_source_sha256([c["source"] for c in nb.cells])
_hash_cells[0]["source"] = rep(_hash_cells[0]["source"], HASH_LINE % HASH_PLACEHOLDER, HASH_LINE % NB_SHA)
nbformat.validate(nb)

# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------
# changed base cells <-> CHANGE_ROWS; unchanged base cells verbatim; ids unique
_changed_keys = {base_key(BASE_CELLS[c["metadata"]["c10"]["base_cell"]]["metadata"]["c10"]) for c in nb.cells
                 if "base_cell" in c["metadata"]["c10"] and c["metadata"]["c10"]["changed"]}
_named = {k for k, _l, _t in CHANGE_ROWS if k is not None}
assert _changed_keys == _named, ("changed but not named", _changed_keys - _named, "named but unchanged",
                                 _named - _changed_keys)
for c in nb.cells:
    m = c["metadata"]["c10"]
    if "base_cell" in m and not m["changed"]:
        assert c["source"] == "".join(BASE_CELLS[m["base_cell"]]["source"]), m
assert len({c["id"] for c in nb.cells}) == len(nb.cells)

# the forbidden token: exactly the inherited, unchanged refusal line of the prepare cell, nowhere else
_prep_base = base_src("prepare")
assert _prep_base.count(FORBIDDEN_TOKEN) == 1 and ("'%s' in p for p in _all_files" % FORBIDDEN_TOKEN) in _prep_base
_hits = [(i, ln) for i, c in enumerate(nb.cells) for ln in c["source"].splitlines() if FORBIDDEN_TOKEN in ln.lower()]
assert len(_hits) == 1 and _hits[0][1].strip().startswith("assert not any(") and \
    _hits[0][1] in _prep_base.splitlines(), _hits
_hit_meta = nb.cells[_hits[0][0]]["metadata"]["c10"]
assert base_key(BASE_CELLS[_hit_meta["base_cell"]]["metadata"]["c10"]) == "prepare", _hit_meta


# segment identity against the base dump_tools cell
def _defs(src):
    tree = ast.parse(src)
    return tree, {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _lines(src, first, last):
    return "".join(src.splitlines(keepends=True)[first.lineno - 1:last.end_lineno])


def _between_defs(src, tree, after, before):
    body = tree.body
    i = next(k for k, n in enumerate(body) if isinstance(n, ast.FunctionDef) and n.name == after)
    j = next(k for k, n in enumerate(body) if isinstance(n, ast.FunctionDef) and n.name == before)
    assert i + 1 < j, (after, before)
    return _lines(src, body[i + 1], body[j - 1])


def _assign_span(src, fn, first_target, last_target):
    def tgt(n):
        return isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id
    i = next(k for k, n in enumerate(fn.body) if tgt(n) == first_target)
    j = next(k for k, n in enumerate(fn.body) if tgt(n) == last_target)
    return _lines(src, fn.body[i], fn.body[j])


def _with_seg(src, fn):
    ws = [n for n in fn.body if isinstance(n, ast.With)]
    assert len(ws) == 1, fn.name
    return _lines(src, ws[0], ws[0])


_bsrc, _nsrc = base_src("dump_tools"), _dt
(_btree, _bdefs), (_ntree, _ndefs) = _defs(_bsrc), _defs(_nsrc)
SEGMENTS = {}
for _name in ("d4", "load_net", "sgf_to_features", "dump_is_done", "write_dump", "write_keys", "_dump_path",
              "_marker_path", "require_gpu_for", "pack_results", "flush_drive", "_argmax_first", "rule_b"):
    SEGMENTS[_name] = (ast.get_source_segment(_bsrc, _bdefs[_name]), ast.get_source_segment(_nsrc, _ndefs[_name]))
SEGMENTS["d4 self-check"] = (_between_defs(_bsrc, _btree, "d4", "load_net"),
                             _between_defs(_nsrc, _ntree, "d4", "load_net"))
SEGMENTS["parse loop sgf_cols= .. parsed="] = (_assign_span(_bsrc, _bdefs["parse_val"], "sgf_cols", "parsed"),
                                               _assign_span(_nsrc, _ndefs["parse_questions"], "sgf_cols", "parsed"))
SEGMENTS["per-question loop (with torch.no_grad())"] = (_with_seg(_bsrc, _bdefs["dump_val"]),
                                                        _with_seg(_nsrc, _ndefs["dump_questions"]))
for _name, (_a, _b) in SEGMENTS.items():
    assert _a and _a == _b, ("segment differs from the base", _name)
assert "for _qid, start, k in V['spans']:" in SEGMENTS["per-question loop (with torch.no_grad())"][1]
assert not {"parse_val", "dump_val", "dump_inner", "record_p01_after", "decode_with_key"} & set(_ndefs)

# the three set cells differ only in their first line
_set_cells = [c["source"] for c in nb.cells if c["metadata"]["c10"].get("new", "").startswith("dump_set_")]
assert len(_set_cells) == 3 and len({s.split("\n", 1)[1] for s in _set_cells}) == 1
assert [s.split("\n", 1)[0] for s in _set_cells] == [SET_FIRST_LINE % s for s in SETS]

# names from the training notebook must be gone from every code cell (ast names, not comments)
_GONE = {"df_val", "df_train", "FINISH_WITHOUT_TRAINING", "SMOKE_CFG_P02", "RESULT_CKPT", "CKPT_DIR", "P02_CHANGES",
         "parse_val", "dump_val", "dump_inner", "EXTRA_EPOCHS", "FINAL_EPOCH", "load_rows", "record_p01_after",
         "p01_fingerprint", "P01_BEFORE", "KEY_LOADER", "INNER_KEYS", "VAL"}
for c in nb.cells:
    if c["cell_type"] == "code":
        _names = {n.id for n in ast.walk(ast.parse(c["source"])) if isinstance(n, ast.Name)}
        _names |= {n.name for n in ast.walk(ast.parse(c["source"])) if isinstance(n, ast.FunctionDef)}
        assert not _names & _GONE, (c["id"], _names & _GONE)
assert NB_SHA == nb_source_sha256([c["source"].replace(HASH_LINE % NB_SHA, HASH_LINE % HASH_PLACEHOLDER)
                                   for c in nb.cells])

os.makedirs(os.path.dirname(os.path.abspath(OUT_NB)), exist_ok=True)
with open(OUT_NB, "w", encoding="utf-8", newline="\n") as f:
    nbformat.write(nb, f)

# ---------------------------------------------------------------------------
# Per-cell unified diff against colab/prereg02_rank.ipynb, plus the notebook metadata
# ---------------------------------------------------------------------------
out = []
out.append("# test02 rank：colab/prereg02_rank.ipynb 與 colab/test02_rank.ipynb 的逐格 unified diff（程式格、markdown 格、notebook metadata）")
out.append("# 底稿：%s  sha256=%s" % (BASE_REL, sha(BASE_NB)))
out.append("# 本版：%s  sha256=%s  notebook_sha256（config.json 的規則）=%s" % (OUT_REL, sha(OUT_NB), NB_SHA))
out.append("# 產生方式：colab/build_test02_rank_nb.py；difflib.unified_diff，依 cell metadata c10.base_cell 配對底稿的格；"
           "新增的格對空白比。cell 編號從 0 起算。")
out.append("# 每一處改動的說明見本版第 0 格「和 prereg02 notebook 的差異」；沒列在表上的沿用格和底稿逐字相同。")
out.append("# 產生器核對過和底稿逐字相同的片段：" + "、".join(SEGMENTS))
summary, body = [], []
for k, cell in enumerate(nb.cells):
    meta = cell["metadata"]["c10"]
    kind = "程式格" if cell["cell_type"] == "code" else "markdown"
    if "base_cell" in meta:
        j = meta["base_cell"]
        old_src = "".join(BASE_CELLS[j]["source"])
        a = "prereg02_rank.ipynb cell %d（%s）" % (j, base_key(BASE_CELLS[j]["metadata"]["c10"]))
        status = "改寫" if meta.get("rewritten") else ("改動" if meta["changed"] else "完全相同")
    else:
        old_src = ""
        a = "/dev/null（新增：%s）" % meta["new"]
        status = "新增"
    b = "test02_rank.ipynb cell %d" % k
    summary.append("#   本版 cell %2d（%s）← %s：%s" % (k, kind, a, status))
    diff = list(difflib.unified_diff(old_src.splitlines(), cell["source"].splitlines(),
                                     fromfile=a, tofile=b, lineterm=""))
    body.append("")
    body.append("=" * 100)
    body.append("本版 cell %d（%s）↔ %s（%s）" % (k, kind, a, status))
    body.append("=" * 100)
    body.extend(diff if diff else ["（完全相同）"])
_dropped = ["#   prereg02 cell %2d（%s，%s）" % (j, "程式格" if c["cell_type"] == "code" else "markdown",
                                               base_key(c["metadata"]["c10"]))
            for j, c in enumerate(BASE_CELLS) if base_key(c["metadata"]["c10"]) not in _used]
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
out.extend(list(difflib.unified_diff(_meta_old, _meta_new, fromfile="prereg02 metadata",
                                     tofile="本版 metadata", lineterm="")) or ["（完全相同）"])
os.makedirs(os.path.dirname(os.path.abspath(OUT_DIFF)), exist_ok=True)
with open(OUT_DIFF, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print("wrote", OUT_NB, sha(OUT_NB))
print("wrote", OUT_DIFF, sha(OUT_DIFF))
print("cells", len(nb.cells), "| base cells changed", sorted(_changed_keys), "| dropped", len(_dropped),
      "| notebook_sha256", NB_SHA)
