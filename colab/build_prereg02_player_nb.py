"""Build colab/prereg02_player.ipynb (competition 1, TEAM_11249; inference only) from the committed
colab/prereg01_player.ipynb, and write results/prereg02/notebook_diff_player.txt (per-cell unified diff
against prereg01_player.ipynb, plus the notebook metadata).

Spec: results/prereg02/design/plan.md 3.0 and 3.2; registration docs/PREREG_02_OPENING_FUSION.md 2.1.

* The base notebook is read from the repo and its SHA-256 is asserted (BASE_SHA256 below).
* Cells kept from the base are picked by their cell metadata `c10` (official_cell, or the id of a
  prereg01 new cell plus its first line) and changed only with rep(): exact string replacement whose
  target must occur exactly once, so every unchanged line stays byte-identical.
* All other cells are new code or markdown cells with ids `c10-player02-NN`.
* The expected SHA-256 of the prereg01 player outputs (weights, top-5, embeddings, predict summary)
  are computed here from the repo copy in results/prereg01/player/ (identical to the files on Drive)
  and written into the preflight cell; the notebook checks them on the real run.
* Every code cell must compile, the notebook must not mention the answer files except in the base
  prepare cell's refusal assert, cell ids are unique, and a rebuild gives byte-identical outputs.

Usage (repo root optional; defaults to the repo that contains this file):
    .venv/Scripts/python.exe colab/build_prereg02_player_nb.py [REPO_ROOT]
"""
import argparse
import difflib
import hashlib
import io
import json
import os
import tokenize

import nbformat

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("root", nargs="?", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                help="repo root (default: the repo that contains this script)")
ROOT = os.path.abspath(ap.parse_args().root)

BASE_REL = "colab/prereg01_player.ipynb"
BASE_SHA256 = "54c96f03871cbb24be419850b6915c222a9e1315083034928a56a87d55c15c53"
OFFICIAL_REL = "third_party/aicup2026_tutorial/player-identification.ipynb"
OUT_REL = "colab/prereg02_player.ipynb"
DIFF_REL = "results/prereg02/notebook_diff_player.txt"
P01_REPO_DIR = "results/prereg01/player"      # repo copy of MyDrive/C10/results/prereg01/player (frozen)


def path(rel):
    return os.path.join(ROOT, *rel.split("/"))


def file_sha256(rel):
    with open(path(rel), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rep(src, old, new):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"replacement target occurs {n} times:\n{old}")
    return src.replace(old, new)


def dict_src(d, comment):
    # a dict literal, one key per line, for pasting into a code cell
    return ("{  # %s\n" % comment + "".join("    %r: %r,\n" % kv for kv in d.items()) + "}")


# --------------------------------------------------------------------------- base notebook
with open(path(BASE_REL), "rb") as f:
    _raw = f.read()
if hashlib.sha256(_raw).hexdigest() != BASE_SHA256:
    raise SystemExit("%s changed: sha256 %s != %s" % (BASE_REL, hashlib.sha256(_raw).hexdigest(), BASE_SHA256))
base = json.loads(_raw.decode("utf-8"))
BASE_SRC = ["".join(c["source"]) for c in base["cells"]]


def base_official(i):
    # the base cell that carries official cell i
    hits = [j for j, c in enumerate(base["cells"]) if c["metadata"]["c10"]["official_cell"] == i]
    assert len(hits) == 1, (i, hits)
    return hits[0]


def base_new(cell_id, head):
    # a cell that prereg01 added (official_cell is None), identified by id and first line
    hits = [j for j, c in enumerate(base["cells"])
            if c["id"] == cell_id and c["metadata"]["c10"]["official_cell"] is None]
    assert len(hits) == 1, (cell_id, hits)
    assert BASE_SRC[hits[0]].startswith(head), (cell_id, BASE_SRC[hits[0]][:80])
    return hits[0]


# --------------------------------------------------------------------------- prereg01 facts from the repo
with open(path(P01_REPO_DIR + "/predict_summary.json"), encoding="utf-8") as f:
    P01_SUMMARY_REPO = json.load(f)
assert P01_SUMMARY_REPO["smoke"] is False
# SHA-256 of the prereg01 player files as they are on the Drive (= the members of
# results/prereg01/player_results.zip = the committed copies in results/prereg01/player/). Pinned here, as in
# build_prereg02_rank_nb.py, so that a CRLF checkout of the repo copies (core.autocrlf=true) can never change
# what the notebook expects; the repo copies are checked against them with CRLF folded to LF.
P01_EXPECTED_SHA256 = {
    "best_model.pth": "356eb4f65488925e2a552ea382979780c903ef515b85d329a3805db837b7773b",
    "player_val_top5.csv": "0f5761a11b9f68ab07b2bc91b9650236f85b8cbeff45d459753dbd24459d3ec7",
    "player_val_embeddings.npz": "c247b2653e73589975d558b17e96c4a1f38c3e4d66468d56ea6d02596e15752d",
    "predict_summary.json": "88b720242fef538a99327fdbdf839515c214a8492e2bf551f48b93c4c81eeb32",
}


def text_sha256(rel):
    with open(path(rel), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


assert P01_SUMMARY_REPO["weights_sha256"] == P01_EXPECTED_SHA256["best_model.pth"]
assert P01_SUMMARY_REPO["output_sha256"] == P01_EXPECTED_SHA256["player_val_top5.csv"]
assert text_sha256(P01_REPO_DIR + "/player_val_top5.csv") == P01_EXPECTED_SHA256["player_val_top5.csv"]
assert text_sha256(P01_REPO_DIR + "/predict_summary.json") == P01_EXPECTED_SHA256["predict_summary.json"]
assert file_sha256(P01_REPO_DIR + "/player_val_embeddings.npz") == P01_EXPECTED_SHA256["player_val_embeddings.npz"]
P01_EXPECTED_COUNTS = {k: P01_SUMMARY_REPO[k] for k in
                       ("n_questions", "n_query_games", "n_candidate_rows", "n_candidate_players",
                        "n_centroids", "n_questions_embedded", "n_fallback_questions")}
assert (P01_EXPECTED_COUNTS["n_questions"], P01_EXPECTED_COUNTS["n_candidate_rows"]) == (400, 40000)

# --------------------------------------------------------------------------- cells
cells = []          # (nbformat cell, base index or None)


def md(text):
    c = nbformat.v4.new_markdown_cell(text, metadata={"c10": {"base_cell": None, "official_cell": None,
                                                              "change": "new"}})
    cells.append((c, None))


def code(src, tags):
    c = nbformat.v4.new_code_cell(src, metadata={"c10": {"base_cell": None, "official_cell": None,
                                                         "change": "new", "tags": list(tags)}})
    cells.append((c, None))


def from_base(j, src, tags=None):
    b = base["cells"][j]
    change = "identical" if src == BASE_SRC[j] else "modified"
    meta = {"c10": {"base_cell": j, "base_id": b["id"], "official_cell": b["metadata"]["c10"]["official_cell"],
                    "change": change, "tags": list(tags) if tags is not None else list(b["metadata"]["c10"]["tags"])}}
    maker = nbformat.v4.new_code_cell if b["cell_type"] == "code" else nbformat.v4.new_markdown_cell
    cells.append((maker(src, metadata=meta), j))


# 和 prereg01 notebook 的全部差異：第 2 格、config.json、notebook metadata 共用這一份
DEVIATIONS = [
    ("登記 2.1", "只推論：不訓練、不讀訓練資料。底稿裡讀訓練資料、抽特徵、外洩檢查、DataLoader、接續偵測、建模型與訓練、"
                 "loss 曲線的格（官方 cell 3–18 與 prereg01 新增的外洩檢查、接續偵測）和官方 cell 21 都拿掉；官方 cell 20 "
                 "只留 `sgf_windows`、`embed`（和官方原文相同），拿掉 `mean_embeddings`、`predict` 與另存 "
                 "`player_val_embeddings.npz`。準備格不再解開訓練資料壓縮檔（打包檔仍逐檔核對 SHA-256，包括那個壓縮檔，"
                 "但不解開），也不 import `c10data`。底稿的官方 cell 0、19、22 等說明格不沿用，改寫在第 1 格。"),
    ("登記 2.1", "權重：只讀 prereg01 的 `results/prereg01/player/ckpt/best_model.pth`。官方 cell 2 的 `CKPT_DIR` 改指 "
                 "prereg01 的 ckpt 目錄，不建立資料夾；印「總訓練量」那行改成印權重路徑，印 `device` 那行多印 GPU 型號"
                 "（`ENV['gpu']`）。權重整檔讀進記憶體，SHA-256 "
                 "和 preflight 記下的相同才從同一份位元組載入；推論是 fp32，和 prereg01 的推論相同。"),
    ("登記 2.1", "每盤 embedding：新增 `game_embeddings()`。每盤的算法和 prereg01 的 `mean_embeddings` 相同"
                 "（`sgf_windows` → 一盤一次 `embed` → `e.sum(0)`、`len(e)`；CHUNK 256、GPU_BATCH 4096），只是不按人或題加總，"
                 "每盤各存一列：窗口 embedding 的加總（float32）與窗口數。候選 → `dumps/cand_games.npz`"
                 "（game_id、player_id、color、emb_sum [N,256]、n_windows；候選檔的列順序，40,000 列）；題目 → "
                 "`dumps/query_games.npz`（question_id、slot、color、emb_sum、n_windows、parsed；題目檔的列順序、每題 slot "
                 "由小到大，slot 是 `sgf_<slot>` 欄的編號，4,935 列）。解析失敗或沒有窗口的盤 emb_sum 全 0、n_windows 0"
                 "（題目另記 parsed=False），和 prereg01 一樣不算進重心。兩個檔另存 `weights_sha256`。"),
    ("登記 2.1", "重現核對：只用上面存的每盤 embedding，照 prereg01 的順序與精度（依列順序逐盤在 float32 下累加、"
                 "除以窗口數）重算依窗口數加權的候選重心與題目向量，照官方做法用 L2 距離取 Top-5，寫 "
                 "`player_val_top5_repro.csv`。通過條件：和 prereg01 的 `player_val_top5.csv` 相比，400 題中至少 398 題的 "
                 "top_1…top_5 完全相同（SMOKE：全部相同）；候選代號與順序和 prereg01 `player_val_embeddings.npz` 相同，"
                 "重心的最大差距 ≤ 1e-4。結果寫 `repro_check.json`（題目向量的差距只記錄，不當條件）。沒過的話收尾格照樣"
                 "打包，最後停下，run_log 的狀態記為 repro_failed。"),
    ("營運性", "prereg01 唯讀保護：設定格定義 `P01_ROOT`（`results/prereg01`，SMOKE 是 `prereg01_smoke`）和 "
               "`guard_write(path)`：realpath 落在 `P01_ROOT` 底下就停止。所有寫檔（`save_json`、npz、csv、zip）、"
               "建資料夾、刪檔、複製與解壓的目的地都先經過它。preflight 開頭和收尾格各記一次 `P01_ROOT` 底下每個檔的"
               "相對路徑、大小、mtime（`p01_readonly_before.json`、`p01_readonly_after.json`），兩者不同就停下。"),
    ("營運性", "preflight（新增）：確認 prereg01 的 `ckpt/best_model.pth`、`player_val_top5.csv`、"
               "`player_val_embeddings.npz`、`predict_summary.json` 都在，記下 SHA-256（config.json 的 `p01_inputs`）。"
               "`predict_summary.json` 記的權重與 Top-5 的 SHA-256、題目檔與候選檔的 SHA-256、smoke 旗標都要對得上；"
               "正式執行時四個檔的 SHA-256 和盤數題數還要和 repo 的 `results/prereg01/player/` 相同（產生本 notebook 時"
               "寫進來），而且必須有 GPU、GPU 型號是 G4（`ENV['gpu']` 含 RTX PRO 6000；登記第 7 節第 8 條）、"
               "Drive 剩 ≥ 0.5 GB。任何一項不符就用中文說明處理方式後停下，還沒開始推論。"),
    ("營運性", "結果目錄改成 `results/prereg02/player/`（SMOKE 是 `prereg02_smoke`），結果 zip 是 "
               "`results/prereg02/player_results.zip`；權重不複製。收尾格另外核對 dump 的列數與 SHA-256、每個輸出檔 "
               "< 100 MB（zip 會跳過 ≥ 100 MB 的檔），再比對 prereg01 前後指紋；有任何問題照樣打包，最後停下。"),
    ("營運性", "中斷後重新「全部執行」會整份重跑，沒有接續（G4 約 5 分鐘，推估）：dump 格先刪掉上一次的 dump、"
               "重現核對結果和結尾指紋，npz 先寫暫存檔再改名。run_log.json、timings.json 每次執行各記一筆。"),
    ("營運性", "SMOKE：讀同一個 `--scratch` 裡 prereg01 冒煙測試產生的 `results/prereg01_smoke/`，縮小方式和 "
               "prereg01 冒煙測試相同（題目檔前 6 題、每題前 2 盤；候選照代號排序的前 12 位、每人前 3 盤），所以重現核對"
               "在 SMOKE 也必須全部相同。設定格的 SMOKE 縮小值沿用底稿（訓練用的幾個值這裡用不到）。"),
    ("營運性", "notebook：沿用的格保留底稿的 cell id（官方格就是官方的 id），新增的格用 `c10-player02-NN`；每格的 "
               "metadata `c10` 記對應的底稿格、官方格、是否改動。notebook metadata 拿掉底稿 Colab 設定裡 A100、高 RAM "
               "的預設（照第一格選 G4；重現核對不因 GPU 放寬，所以拿不到 G4 時先等，不換 GPU），`c10` 欄改記底稿與這份差異清單。"),
]
DEV_STRINGS = ["[%s] %s" % (t, s) for t, s in DEVIATIONS]

# --------------------------------------------------------------------------- 0 title
md(r"""# C10 競賽一（玩家辨識）：prereg01 模型的每盤 embedding 與重現核對（事前登記 02，只推論）

**用途**：不訓練、不讀訓練資料。用 prereg01 訓練好的 `best_model.pth`（只讀），替本機驗證的候選庫（400 人 × 100 盤 = 40,000 盤）和題目（400 題、4,935 盤）每盤各存一筆「窗口 embedding 的加總」和窗口數，帶回本機做開局 token 融合與候選池自我檢索（`docs/PREREG_02_OPENING_FUSION.md` 第 2 節）。同一次執行也用這些每盤 embedding 照 prereg01 的做法重算 Top-5，確認和 prereg01 一致（重現核對）。**這裡不打分**，Colab 上也沒有答案檔。

**對應事前登記**：`docs/PREREG_02_OPENING_FUSION.md` 第 2.1 節；實作規格 `results/prereg02/design/plan.md` 3.0、3.2。底稿是 `colab/prereg01_player.ipynb`，和它的全部差異列在下一格；逐格 diff 在 `results/prereg02/notebook_diff_player.txt`。

## 怎麼跑

1. 用**跑 prereg01 的同一個 Google 帳號**：prereg01 的權重和結果在那個帳號的 Drive（`MyDrive/C10/results/prereg01/`）。打包檔 `C10_bundle_v1.zip` 沿用，不用重傳。
2. 把這份 `prereg02_player.ipynb` 放進 `MyDrive/C10/`，用 Colab 開啟。
3. 「執行階段 → 變更執行階段類型」選 **G4 GPU**（prereg01 就是在 G4 上跑的）。G4 是灰色、點不了時，**先不要換別的 GPU**：這份只要幾分鐘，隔一段時間再試。重現核對的容許差不因 GPU 放寬（登記 2.1），換成 L4 等別的 GPU 時重心可能差超過 1e-4 而沒過，沒過就要再用 G4 重跑。所以 Colab 改派別的 GPU 時，preflight 會停下。等了一天還是拿不到 G4，先告訴 Claude。
4. 「執行階段 → 全部執行」，照畫面允許 Google Drive 存取。G4 約 5–7 分鐘（推估，還沒實測）。
5. 最後一行印出「Drive 已寫入完成並卸載」後，zip 已經在 Drive 上：先按「執行階段 → 中斷連線並刪除執行階段」（開著會一直扣運算單元），再下載 `MyDrive/C10/results/prereg02/player_results.zip`，交給 Claude 的方式照 `docs/COLAB_RUN.md`。下面接著出現紅字時，zip 照樣下載，並把紅字截圖給 Claude。

詳細步驟以 `docs/COLAB_RUN.md` 的「事前登記 02」一節為準；和這裡不一致時，以那一節為準。

## 結果寫到哪裡

`MyDrive/C10/results/prereg02/player/`（整個資料夾會打包成上一層的 `player_results.zip`）

| 檔案 | 內容 |
|---|---|
| `dumps/cand_games.npz` | 候選每盤一列（候選檔的列順序）：game_id、player_id、color、emb_sum [40000,256]、n_windows |
| `dumps/query_games.npz` | 題目每盤一列（題目檔的列順序、slot 由小到大）：question_id、slot、color、emb_sum [4935,256]、n_windows、parsed |
| `repro_check.json`、`player_val_top5_repro.csv` | 重現核對：用每盤 embedding 重算的 Top-5 和重心，和 prereg01 比 |
| `p01_readonly_before.json`、`p01_readonly_after.json` | prereg01 目錄開頭與結尾的指紋（檔名、大小、mtime），必須相同 |
| `config.json`、`env.json`、`run_log.json`、`timings.json` | 設定（含 prereg01 輸入檔的 SHA-256）、環境、每次執行的紀錄、耗時 |

prereg01 的目錄 `MyDrive/C10/results/prereg01/` **只讀**：這份 notebook 任何寫入落在那裡都會被擋下。

## 中斷或出錯

- **斷線**：重新「全部執行」，整份重跑（沒有接續，G4 約 5 分鐘，推估）。
- **preflight 停下**：照印出的中文說明處理，最常見的是開錯 Google 帳號或拿到的不是 G4。這時還沒開始推論，但執行階段開著就會一直扣運算單元：截圖後先按「執行階段 → 中斷連線並刪除執行階段」。
- **重現核對沒過**：最後一格照樣產生 `player_results.zip`，然後停下並印出原因。下載 zip、截圖給 Claude。

## 本機冒煙測試

`colab/run_local_smoke.py` 會設 `C10_SMOKE=1`，讀同一個 `--scratch` 裡 prereg01 冒煙測試產生的 `results/prereg01_smoke/`（只讀），結果寫到 `results/prereg02_smoke/player/`。要先跑 prereg01 player 的冒煙測試，再用同一個 `--scratch` 跑這份。SMOKE 的縮小方式和 prereg01 冒煙測試相同，正式執行時完全不生效。""")

md("## 與 prereg01 notebook 的差異\n\n"
   "底稿是 `colab/prereg01_player.ipynb`（它和官方 notebook 的差異記在它自己的第 2 格）。"
   "登記的內容是 `docs/PREREG_02_OPENING_FUSION.md` 第 2.1 節，其餘是營運性差異。"
   "**模型、權重、特徵（17 層）、每盤的 embedding 算法、重心與 Top-5 的算法都和 prereg01 相同**，差別只在不訓練、"
   "每盤各存一筆，並多了唯讀保護、preflight 和重現核對。\n\n"
   + "\n".join("%d. **%s** %s" % (i, t, s) for i, (t, s) in enumerate(DEVIATIONS, 1)))

# --------------------------------------------------------------------------- setup (base new cell)
j = base_new("c10-player-02", "# 設定格（營運性）")
s = BASE_SRC[j]
s = rep(s, """# 正式結果與 SMOKE 結果分開放，絕不混用
RESULT_DIR = os.path.join(DRIVE_DIR, 'results', 'prereg01_smoke' if SMOKE else 'prereg01', 'player')
BUNDLE_ZIP = os.path.join(DRIVE_DIR, 'C10_bundle_v1.zip')
BUNDLE = os.path.join(WORK, 'C10_bundle_v1')          # 打包檔解開後的目錄
os.makedirs(WORK, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
""", """# 正式結果與 SMOKE 結果分開放，絕不混用（prereg02：結果寫到 prereg02／prereg02_smoke）
RESULT_DIR = os.path.join(DRIVE_DIR, 'results', 'prereg02_smoke' if SMOKE else 'prereg02', 'player')
BUNDLE_ZIP = os.path.join(DRIVE_DIR, 'C10_bundle_v1.zip')
BUNDLE = os.path.join(WORK, 'C10_bundle_v1')          # 打包檔解開後的目錄

# prereg01（只讀）：權重和 prereg01 的推論結果都在這裡；SMOKE 時讀本機冒煙測試產生的 prereg01_smoke
P01_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg01_smoke' if SMOKE else 'prereg01')
P01_RESULT_DIR = os.path.join(P01_ROOT, 'player')
P01_PLAYER_CKPT = os.path.join(P01_RESULT_DIR, 'ckpt')


def _inside(path, root):
    path = os.path.normcase(os.path.realpath(path))
    root = os.path.normcase(os.path.realpath(root))
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def guard_write(path):
    # 所有寫檔、建資料夾、刪檔、複製、解壓、打包的目的地都先經過這裡：落在 prereg01 底下就停止。
    # 回傳 path，可以直接包在呼叫裡
    if _inside(path, P01_ROOT):
        raise RuntimeError('拒絕寫入 prereg01 的目錄（只讀）：%s' % path)
    return path


os.makedirs(guard_write(WORK), exist_ok=True)
os.makedirs(guard_write(RESULT_DIR), exist_ok=True)
""")
s = rep(s, "print('RESULT_DIR', RESULT_DIR)\n",
        "print('RESULT_DIR', RESULT_DIR)\nprint('P01_ROOT  ', P01_ROOT, '（只讀）')\n")
from_base(j, s, tags=["營運性", "登記 2.1"])

# --------------------------------------------------------------------------- prepare (base new cell)
j = base_new("c10-player-03", "# 準備格（營運性）")
s = BASE_SRC[j]
s = rep(s, "# 準備格（營運性）：缺套件才安裝、解開打包檔並逐檔核對 SHA-256、解開訓練資料、設定 sys.path\n",
        "# 準備格（營運性）：缺套件才安裝、解開打包檔並逐檔核對 SHA-256、設定 sys.path"
        "（prereg02：不解開、不讀訓練資料）\n")
s = rep(s, "import tarfile\n", "")                  # only the removed training-data block used it
s = rep(s, "    shutil.rmtree(BUNDLE, ignore_errors=True)\n",
        "    shutil.rmtree(guard_write(BUNDLE), ignore_errors=True)\n")
s = rep(s, "    shutil.copyfile(BUNDLE_ZIP, _local_zip)\n",
        "    shutil.copyfile(BUNDLE_ZIP, guard_write(_local_zip))\n")
s = rep(s, "        z.extractall(WORK)\n", "        z.extractall(guard_write(WORK))\n")
s = rep(s, "    os.remove(_local_zip)\n", "    os.remove(guard_write(_local_zip))\n")
_a = s.index("# 訓練資料：解開成 WORK/training/")
_b = s.index("PREPARE_S = time.time() - _t_prepare")
s = rep(s, s[_a:_b], "# prereg02：只推論，不解開、不讀訓練資料（訓練資料壓縮檔只在上面和打包檔的其他檔一起核對 SHA-256）\n\n")
from_base(j, s, tags=["營運性", "登記 2.1"])
PREPARE_BASE = j

# --------------------------------------------------------------------------- official 1 (imports), identical
j = base_official(1)
from_base(j, BASE_SRC[j])

# --------------------------------------------------------------------------- environment (base new cell)
j = base_new("c10-player-04", "# 環境紀錄格（營運性）")
s = BASE_SRC[j]
s = rep(s, """    path = os.path.join(RESULT_DIR, name)
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
""", """    path = guard_write(os.path.join(RESULT_DIR, name))      # prereg02：先確認不在 prereg01 底下
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
""")
s = rep(s, "    'source_sha256': SOURCE_CHECK,           # c10data.py 在官方 cell 4 核對後補進來\n",
        "    'source_sha256': SOURCE_CHECK,           # prereg02 不讀訓練資料、不 import c10data，只核對 utils、network\n")
from_base(j, s)

# --------------------------------------------------------------------------- official 2 (settings)
j = base_official(2)
s = BASE_SRC[j]
s = rep(s, """# （本版，營運性：訓練資料在 WORK/training；特徵與索引放執行階段本機磁碟；checkpoint 放 Drive 結果目錄）
TRAIN_DIR = os.path.join(WORK, 'training')
SPLIT_CSV = os.path.join(BUNDLE, 'split_v1', 'players_split.csv')
LOCAL_DIR = os.path.join(BUNDLE, 'local')     # 只有題目檔與候選庫，沒有答案
""", """# （prereg02，只推論：不讀訓練資料，所以沒有 TRAIN_DIR、SPLIT_CSV、FEAT_DIR、INDEX_PATH；
#   checkpoint 讀 prereg01 的 ckpt 目錄，只讀）
LOCAL_DIR = os.path.join(BUNDLE, 'local')     # 只有題目檔與候選庫，沒有答案
""")
s = rep(s, """FEAT_DIR = os.path.join(WORK, 'player-features')
INDEX_PATH = os.path.join(WORK, 'player_index.json')
CKPT_DIR = os.path.join(RESULT_DIR, 'ckpt')
""", """CKPT_DIR = P01_PLAYER_CKPT                    # prereg01 的 ckpt 目錄（只讀；只用 best_model.pth）
""")
s = rep(s, "os.makedirs(CKPT_DIR, exist_ok=True)\n",
        "# prereg02：CKPT_DIR 是 prereg01 的目錄，只讀，這裡不建立資料夾\n")
s = rep(s, "print('device', DEVICE, '| NPROC', NPROC, '| DataLoader workers', NUM_WORKERS)\n",
        "print('device', DEVICE, '| GPU', ENV['gpu'], '| NPROC', NPROC, '| DataLoader workers', NUM_WORKERS)\n")
assert s.endswith("BATCH_SIZE, ',')))")     # official cells end without a newline
s = rep(s, """print('總訓練量 %s 步 = %s 個 triplet'
      % (format(NUM_EPOCHS * STEPS_PER_EPOCH, ','),
         format(NUM_EPOCHS * STEPS_PER_EPOCH * BATCH_SIZE, ',')))""",
        """print('只推論、不訓練；權重讀 %s（只讀）' % os.path.join(CKPT_DIR, 'best_model.pth'))""")
from_base(j, s, tags=["營運性", "登記 2.1"])

# --------------------------------------------------------------------------- config record (base new cell)
j = base_new("c10-player-05", "# 設定紀錄格（營運性）")
s = BASE_SRC[j]
s = rep(s, "# 設定紀錄格（營運性）：全部超參數、實際 NPROC／worker 數、是否 SMOKE、偏離官方的地方 → config.json\n",
        "# 設定紀錄格（營運性）：推論設定、重現核對的條件、實際 NPROC、是否 SMOKE、和 prereg01 notebook 的差異 → config.json\n"
        "# （prereg01 notebook 和官方的差異記在 colab/prereg01_player.ipynb 的 metadata，這裡不重抄）\n")
_a = s.index("DEVIATIONS = [\n")
_b = s.index("save_json('config.json', CONFIG)\n")
s = rep(s, s[_a:_b], "DEVIATIONS = [\n" + "".join("    %r,\n" % d for d in DEV_STRINGS) + "]\n" + r"""# 重現核對的條件（登記 2.1）：Top-5 完全相同的題至少 400 − 2 = 398 題（SMOKE：全部相同）；重心最大差距 ≤ 1e-4
REPRO_MAX_MISMATCH = 0 if SMOKE else 2
REPRO_CENTROID_TOL = 1e-4
CONFIG = {
    'notebook': 'colab/prereg02_player.ipynb',
    'base_notebook': 'colab/prereg01_player.ipynb',
    'base_notebook_sha256': '@@BASE_SHA256@@',
    'official_notebook': 'third_party/aicup2026_tutorial/player-identification.ipynb',
    'prereg': 'docs/PREREG_02_OPENING_FUSION.md 第 2.1 節（只推論、每盤 embedding、重現核對）；'
              '規格 results/prereg02/design/plan.md 3.0、3.2',
    'split': MANIFEST.get('split'),
    'in_colab': IN_COLAB,
    'smoke': SMOKE,
    'inference': {
        'weights': 'prereg01 的 ckpt/best_model.pth（只讀；SHA-256 見 p01_inputs）',
        'model': 'GoPlayerResNet() 預設值：in_channels 17、num_channels 128、num_blocks 16、embed_dim 256',
        'features': 'SGFParsePlayerIdentification() 預設 history_length 8：8 層自己 + 8 層對手 + 1 層執色 = 17 層',
        'precision': 'fp32（和 prereg01 的推論相同，沒有 autocast）',
        'per_game': '每盤 sgf_windows → 一盤一次 embed（GPU_BATCH 4096）→ emb_sum = e.sum(0)（float32）、n_windows = len(e)；'
                    '每次平行解析 CHUNK 256 盤（同 prereg01 的 mean_embeddings）',
        'centroid': '依窗口數加權：同一位候選（或同一題）的各盤 emb_sum 依列順序在 float32 下累加，再除以窗口數的和（同 prereg01）',
        'distance': 'L2（np.linalg.norm），np.argsort 取前 TOP_N；沒有任何可解析棋譜的題填照代號排序的前 TOP_N 位（同官方）',
        'TOP_N': TOP_N,
    },
    'repro_criteria': {
        'top5_max_mismatch': REPRO_MAX_MISMATCH,
        'centroid_max_abs_diff': REPRO_CENTROID_TOL,
        'rule': 'top_1…top_5 完全相同的題 ≥ 題數 − top5_max_mismatch；候選代號與順序相同；重心最大差距 ≤ centroid_max_abs_diff',
    },
    'nproc': NPROC,
    'device': str(DEVICE),
    'mp_start_method': multiprocessing.get_start_method(),
    'smoke_settings': ({k: v for k, v in globals().items() if k.startswith('SMOKE_')} if SMOKE else None),
    'paths': {'DRIVE_DIR': DRIVE_DIR, 'WORK': WORK, 'RESULT_DIR': RESULT_DIR, 'P01_ROOT': P01_ROOT,
              'P01_RESULT_DIR': P01_RESULT_DIR, 'CKPT_DIR': CKPT_DIR,
              'QUESTIONS_CSV': QUESTIONS_CSV, 'CANDIDATES_CSV': CANDIDATES_CSV},
    'p01_inputs': None,       # preflight 會填
    'deviations_from_base': DEVIATIONS,
}
""".replace("@@BASE_SHA256@@", BASE_SHA256))
s = rep(s, "print('config.json 已寫入；偏離官方的地方 %d 項' % len(DEVIATIONS))\n",
        "print('config.json 已寫入；和 prereg01 notebook 的差異 %d 項' % len(DEVIATIONS))\n")
from_base(j, s, tags=["營運性", "登記 2.1"])

# --------------------------------------------------------------------------- preflight (new)
md("### 本版新增：preflight 與 prereg01 唯讀指紋\n\n"
   "開始推論前確認 prereg01 的權重（`ckpt/best_model.pth`）、Top-5（`player_val_top5.csv`）、重心"
   "（`player_val_embeddings.npz`）和推論摘要（`predict_summary.json`）都在，而且 SHA-256 對得上；正式執行時還要和 repo "
   "記錄的相同。同時記下 prereg01 目錄底下每個檔的大小和 mtime（`p01_readonly_before.json`），收尾格會再記一次比對。"
   "任何一項不符就用中文說明處理方式後停下，還沒開始推論。")
code(r"""# preflight（本版新增）：prereg01 的權重與推論結果都在、SHA-256 對得上，才開始推論；並記下 prereg01 目錄開頭的指紋
_t_pre = time.time()
P01_FILES = {
    'best_model.pth': os.path.join(CKPT_DIR, 'best_model.pth'),
    'player_val_top5.csv': os.path.join(P01_RESULT_DIR, 'player_val_top5.csv'),
    'player_val_embeddings.npz': os.path.join(P01_RESULT_DIR, 'player_val_embeddings.npz'),
    'predict_summary.json': os.path.join(P01_RESULT_DIR, 'predict_summary.json'),
}
# 正式執行時必須和 repo 的 results/prereg01/player/ 相同（產生本 notebook 時從 repo 算出來寫進這裡）；SMOKE 不比
P01_EXPECTED_SHA256 = @@P01_EXPECTED_SHA256@@
P01_EXPECTED_COUNTS = @@P01_EXPECTED_COUNTS@@
MIN_DRIVE_FREE_GB = 0.5
GPU_MODEL_REQUIRED = 'RTX PRO 6000'   # G4；prereg01 也是在它上面跑的（登記第 7 節第 8 條）


def p01_fingerprint():
    # prereg01 目錄底下每個檔的相對路徑、大小、mtime（只看 stat，不讀內容）
    files = {}
    for _root, _dirs, _files in os.walk(P01_ROOT):
        _dirs.sort()
        for _f in sorted(_files):
            _p = os.path.join(_root, _f)
            _st = os.stat(_p)
            files[os.path.relpath(_p, P01_ROOT).replace(os.sep, '/')] = {'size': _st.st_size,
                                                                          'mtime_ns': _st.st_mtime_ns}
    return {'root': P01_ROOT, 'taken_utc': now_utc(), 'n_files': len(files), 'files': files}


def preflight_stop(msg):
    update_run(status='preflight_failed', preflight_error=msg)
    raise SystemExit('preflight 沒過，還沒開始推論：\n' + msg)


_ACCOUNT_HINT = ('處理方式：確認這個 Colab 用的是跑 prereg01 的同一個 Google 帳號（prereg01 的權重和結果在那個帳號的 '
                 'Drive），而且 MyDrive/C10/results/prereg01/ 沒有被改名、搬動或重跑過。不要自己改檔；'
                 '還是不對就截圖給 Claude。')
if not os.path.isdir(P01_ROOT):
    preflight_stop('找不到 prereg01 的結果目錄 %s。\n%s' % (P01_ROOT, _ACCOUNT_HINT))
P01_BEFORE = p01_fingerprint()
save_json('p01_readonly_before.json', P01_BEFORE)
print('prereg01 目錄指紋（開頭）：%d 個檔' % P01_BEFORE['n_files'])

_missing = [k for k, p in P01_FILES.items() if not os.path.exists(p)]
if _missing:
    preflight_stop('找不到 prereg01 的 %s（在 %s）。\n%s' % ('、'.join(_missing), P01_RESULT_DIR, _ACCOUNT_HINT))
P01_SHA256 = {k: sha256_file(p) for k, p in P01_FILES.items()}
with open(P01_FILES['predict_summary.json'], encoding='utf-8') as f:
    P01_SUMMARY = json.load(f)
P01_ENV = None
if os.path.exists(os.path.join(P01_RESULT_DIR, 'env.json')):
    with open(os.path.join(P01_RESULT_DIR, 'env.json'), encoding='utf-8') as f:
        P01_ENV = json.load(f)
QUESTIONS_SHA256 = sha256_file(QUESTIONS_CSV)
CANDIDATES_SHA256 = sha256_file(CANDIDATES_CSV)

_problems = []
_gpu_problems = []            # GPU 的問題不附帳號的處理方式
if P01_SUMMARY.get('smoke') != SMOKE:
    _problems.append('prereg01 的 predict_summary.json 記 smoke=%r，這次是 smoke=%r' % (P01_SUMMARY.get('smoke'), SMOKE))
if P01_SUMMARY.get('weights_sha256') != P01_SHA256['best_model.pth']:
    _problems.append('best_model.pth 的 SHA-256 %s 和 prereg01 推論時用的 %s 不同'
                     % (P01_SHA256['best_model.pth'], P01_SUMMARY.get('weights_sha256')))
if P01_SUMMARY.get('output_sha256') != P01_SHA256['player_val_top5.csv']:
    _problems.append('player_val_top5.csv 的 SHA-256 %s 和 prereg01 寫出時的 %s 不同'
                     % (P01_SHA256['player_val_top5.csv'], P01_SUMMARY.get('output_sha256')))
if P01_SUMMARY.get('questions_csv_sha256') != QUESTIONS_SHA256:
    _problems.append('題目檔的 SHA-256 和 prereg01 推論時用的不同')
if P01_SUMMARY.get('candidates_csv_sha256') != CANDIDATES_SHA256:
    _problems.append('候選檔的 SHA-256 和 prereg01 推論時用的不同')
if not SMOKE:
    for _k, _v in P01_EXPECTED_SHA256.items():
        if P01_SHA256[_k] != _v:
            _problems.append('%s 的 SHA-256 %s 和 repo 記錄的 %s 不同' % (_k, P01_SHA256[_k], _v))
    for _k, _v in P01_EXPECTED_COUNTS.items():
        if P01_SUMMARY.get(_k) != _v:
            _problems.append('predict_summary.json 的 %s = %r，repo 記錄的是 %r' % (_k, P01_SUMMARY.get(_k), _v))
    # GPU：只用 G4（RTX PRO 6000）。重現核對不因 GPU 放寬，拿不到 G4 時先等，不換 GPU（登記第 7 節第 8 條）。
    # Colab 在選的 GPU 沒空時可能改派別的 GPU，所以核對型號（環境紀錄格記的 ENV['gpu']）
    if not torch.cuda.is_available():
        _gpu_problems.append('沒有 GPU。處理方式：「執行階段 → 變更執行階段類型」選 G4 GPU → 儲存，再重新「全部執行」'
                             '（G4 是灰色時先不要換別的 GPU，隔一段時間再試，見第一格）')
    elif GPU_MODEL_REQUIRED not in (ENV['gpu'] or ''):
        _gpu_problems.append('這次拿到的 GPU 是 %s，不是 G4（RTX PRO 6000）。競賽一的重現核對不因 GPU 放寬，所以不在別的 GPU '
                             '上跑。處理方式：「執行階段 → 中斷連線並刪除執行階段」，再「執行階段 → 變更執行階段類型」選 G4 GPU '
                             '→ 儲存，重新「全部執行」；G4 是灰色時先不要換別的 GPU，隔一段時間再試，等了一天還拿不到再告訴 Claude'
                             % ENV['gpu'])
try:
    _free_gb = shutil.disk_usage(DRIVE_DIR).free / 1e9
except OSError as e:          # 查不到就只提醒（同 prereg02 rank 的 preflight）
    _free_gb = None
    print('注意：查不到 Drive 剩餘空間（%s: %s），請自己確認還有 %.1f GB 以上' % (type(e).__name__, e, MIN_DRIVE_FREE_GB))
if _free_gb is not None and _free_gb < MIN_DRIVE_FREE_GB:
    _problems.append('Google Drive 只剩 %.2f GB（至少要 %.1f GB）。處理方式：清出空間後重新「全部執行」'
                     % (_free_gb, MIN_DRIVE_FREE_GB))
if _problems or _gpu_problems:
    preflight_stop('\n'.join('  - ' + p for p in _problems + _gpu_problems)
                   + ('\n' + _ACCOUNT_HINT if _problems else ''))

CONFIG['p01_inputs'] = {
    'root': P01_ROOT,
    'files': {k: {'path': p, 'sha256': P01_SHA256[k], 'bytes': os.path.getsize(p)} for k, p in P01_FILES.items()},
    'checked_against_repo': not SMOKE,
    'predict_summary': P01_SUMMARY,
    'p01_gpu': P01_ENV.get('gpu') if P01_ENV else None,
    'p01_torch': P01_ENV.get('torch') if P01_ENV else None,
    'questions_csv_sha256': QUESTIONS_SHA256,
    'candidates_csv_sha256': CANDIDATES_SHA256,
}
save_json('config.json', CONFIG)
update_run(p01_sha256=P01_SHA256, drive_free_gb=_free_gb)
record_time('preflight_s', time.time() - _t_pre)
print('preflight 通過：prereg01 的 4 個檔都在，SHA-256 相符%s；Drive 剩 %s'
      % ('（含 repo 記錄）' if not SMOKE else '（SMOKE 不比 repo 記錄）',
         '%.1f GB' % _free_gb if _free_gb is not None else '（查不到）'))
for _k in P01_FILES:
    print('  %s  %s' % (P01_SHA256[_k][:16], _k))
""".replace("@@P01_EXPECTED_SHA256@@", dict_src(P01_EXPECTED_SHA256, "repo 的 results/prereg01/player/ 算出來的 SHA-256"))
     .replace("@@P01_EXPECTED_COUNTS@@", dict_src(P01_EXPECTED_COUNTS, "repo 的 predict_summary.json 記的題數、盤數")),
     tags=["營運性"])

# --------------------------------------------------------------------------- official 20: sgf_windows + embed only
j = base_official(20)
s = BASE_SRC[j]
s = rep(s, "EMB_PATH = os.path.join(RESULT_DIR, 'player_val_embeddings.npz')   # 營運性：另存向量\n",
        "# prereg02：只留 sgf_windows、embed（官方原文）；每盤的 embedding 由下一格的 game_embeddings() 存\n")
s = rep(s, s[s.index("\n\ndef mean_embeddings("):], "")
assert s.endswith("    return np.concatenate(outs) if outs else np.zeros((0, 256), np.float32)\n"), s[-200:]
from_base(j, s, tags=["登記 2.1"])

# --------------------------------------------------------------------------- per-game embeddings (new)
md("### 本版新增：每盤 embedding（登記 2.1）\n\n"
   "每盤的算法和 prereg01 的 `mean_embeddings` 相同（`sgf_windows` → 一盤一次 `embed` → 窗口 embedding 在 float32 下加總），"
   "只是不按人或題平均，而是每盤各存一列加總和窗口數。依窗口數加權的重心 ＝ 同一位候選各盤加總的和 ÷ 窗口數的和，"
   "所以本機可以重算 prereg01 的重心，也可以換成每盤先正規化的 E1。候選 40,000 盤、題目 4,935 盤。")
code(r"""# 每盤 embedding（本版新增，登記 2.1）：候選 40,000 盤 → dumps/cand_games.npz；題目 4,935 盤 → dumps/query_games.npz
import io

_t_dump = time.time()
DUMP_DIR = os.path.join(RESULT_DIR, 'dumps')
CAND_NPZ = os.path.join(DUMP_DIR, 'cand_games.npz')
QUERY_NPZ = os.path.join(DUMP_DIR, 'query_games.npz')
REPRO_JSON = os.path.join(RESULT_DIR, 'repro_check.json')
REPRO_CSV = os.path.join(RESULT_DIR, 'player_val_top5_repro.csv')
EMBED_DIM = 256

# 沒有接續：整份重跑。先刪掉上一次留下的 dump、重現核對結果與結尾指紋，避免新舊混用
for _p in (CAND_NPZ, QUERY_NPZ, CAND_NPZ + '.tmp', QUERY_NPZ + '.tmp', REPRO_JSON, REPRO_CSV,
           os.path.join(RESULT_DIR, 'p01_readonly_after.json')):
    if os.path.exists(_p):
        os.remove(guard_write(_p))
os.makedirs(guard_write(DUMP_DIR), exist_ok=True)


def game_embeddings(model, pool, jobs, label):
    # jobs：[(sgf, color)]。每盤一列：窗口 embedding 的加總（float32）與窗口數。
    # 每盤的算法和 prereg01 的 mean_embeddings 相同（每次平行解析 CHUNK 盤；一盤一次 embed；e.sum(0) 是 float32），
    # 只是不按 key 加總。sgf_windows 回傳 None（解析失敗或沒有窗口）的盤：加總全 0、窗口數 0，prereg01 也跳過這種盤
    emb_sum = np.zeros((len(jobs), EMBED_DIM), np.float32)
    n_windows = np.zeros(len(jobs), np.int32)
    t0 = time.time()
    for start in range(0, len(jobs), CHUNK):
        for g, arr in zip(range(start, start + CHUNK), pool.map(sgf_windows, jobs[start:start + CHUNK])):
            if arr is None:
                continue
            e = embed(model, arr)
            assert e.dtype == np.float32 and e.shape == (len(arr), EMBED_DIM), (e.dtype, e.shape)
            emb_sum[g] = e.sum(0)
            n_windows[g] = len(e)
        done = min(start + CHUNK, len(jobs))
        if done % (CHUNK * 20) == 0 or done == len(jobs):
            print('  %s %d/%d 局  %.0f 秒' % (label, done, len(jobs), time.time() - t0), flush=True)
    return emb_sum, n_windows


def save_npz(path, **arrays):
    # 先寫暫存檔再改名；np.savez 收到檔名會自己補 .npz，所以用檔案物件寫
    with open(guard_write(path + '.tmp'), 'wb') as f:
        np.savez(f, **arrays)
    os.replace(path + '.tmp', guard_write(path))
    return {'rows': int(len(arrays['n_windows'])), 'sha256': sha256_file(path), 'bytes': os.path.getsize(path),
            'n_parsed': int((arrays['n_windows'] > 0).sum()), 'n_windows_total': int(arrays['n_windows'].sum())}


# 權重：整檔讀進記憶體，SHA-256 和 preflight 記的相同才載入（載入的就是算過 SHA-256 的那份位元組）
with open(P01_FILES['best_model.pth'], 'rb') as f:
    _wbytes = f.read()
WEIGHTS_SHA256 = hashlib.sha256(_wbytes).hexdigest()
if WEIGHTS_SHA256 != P01_SHA256['best_model.pth']:
    raise SystemExit('best_model.pth 在 preflight 之後變了（SHA-256 %s），停止' % WEIGHTS_SHA256)
net = GoPlayerResNet()
net.load_state_dict(torch.load(io.BytesIO(_wbytes), map_location='cpu'))
net.eval().to(DEVICE)
del _wbytes

DUMP_INFO = {'weights_sha256': WEIGHTS_SHA256, 'GPU_BATCH': GPU_BATCH, 'CHUNK': CHUNK, 'NPROC': NPROC,
             'device': str(DEVICE), 'gpu': ENV['gpu']}
with Pool(NPROC) as pool:
    # 候選：和 prereg01 同一個檔、同一個列順序；SMOKE 的縮小方式也和 prereg01 相同
    CAND = pd.read_csv(CANDIDATES_CSV)
    if SMOKE:
        _keep = sorted(CAND['player_id'].unique())[:SMOKE_N_CAND_PLAYERS]
        CAND = (CAND[CAND['player_id'].isin(_keep)].groupby('player_id', sort=False)
                .head(SMOKE_CAND_GAMES).reset_index(drop=True))
    _t = time.time()
    CAND_SUM, CAND_NW = game_embeddings(net, pool, list(zip(CAND['sgf_content'], CAND['color'])), '候選')
    record_time('embed_candidates_s', time.time() - _t)
    DUMP_INFO['cand_games.npz'] = save_npz(
        CAND_NPZ, game_id=np.array(CAND['game_id'].astype(str).tolist()),
        player_id=np.array(CAND['player_id'].astype(str).tolist()),
        color=np.array(CAND['color'].astype(str).tolist()),
        emb_sum=CAND_SUM, n_windows=CAND_NW, weights_sha256=np.array(WEIGHTS_SHA256))
    print('寫出 %s（%d 列）' % (CAND_NPZ, len(CAND)))

    # 題目：題目檔的列順序；每題依 sgf_1、sgf_2… 的順序，只收非空的欄（同 prereg01）；slot 是欄位編號
    TEST = pd.read_csv(QUESTIONS_CSV)
    if SMOKE:
        TEST = TEST.head(SMOKE_N_QUESTIONS).reset_index(drop=True)
    sgf_cols = [c for c in TEST.columns if c.startswith('sgf_')]
    col_cols = [c for c in TEST.columns if c.startswith('color_')]
    jobs, Q_IDS, Q_SLOTS, Q_COLORS = [], [], [], []
    for _, r in TEST.iterrows():
        used = 0
        # num_games 之後的 sgf_ 欄位是空字串，要濾掉
        for sc, cc in zip(sgf_cols, col_cols):
            if SMOKE and used >= SMOKE_Q_GAMES:
                break
            if pd.notna(r[sc]) and str(r[sc]).strip():
                _slot = int(sc[len('sgf_'):])
                assert cc == 'color_%d' % _slot, (sc, cc)
                jobs.append((r[sc], r[cc]))
                Q_IDS.append(str(r['question_id']))
                Q_SLOTS.append(_slot)
                Q_COLORS.append(str(r[cc]))
                used += 1
    _t = time.time()
    Q_SUM, Q_NW = game_embeddings(net, pool, jobs, '查詢')
    record_time('embed_queries_s', time.time() - _t)
    DUMP_INFO['query_games.npz'] = save_npz(
        QUERY_NPZ, question_id=np.array(Q_IDS), slot=np.array(Q_SLOTS, np.int32), color=np.array(Q_COLORS),
        emb_sum=Q_SUM, n_windows=Q_NW, parsed=Q_NW > 0, weights_sha256=np.array(WEIGHTS_SHA256))
    print('寫出 %s（%d 列）' % (QUERY_NPZ, len(jobs)))

N_CAND_ROWS, N_QUERY_ROWS = len(CAND), len(jobs)
# 盤數和題數必須和 prereg01 推論時相同（SMOKE 時和 prereg01 冒煙測試相同）
assert (N_CAND_ROWS, N_QUERY_ROWS, len(TEST)) == (P01_SUMMARY['n_candidate_rows'], P01_SUMMARY['n_query_games'],
                                                  P01_SUMMARY['n_questions']), \
    ((N_CAND_ROWS, N_QUERY_ROWS, len(TEST)), P01_SUMMARY)
DUMP_INFO['seconds'] = time.time() - _t_dump
update_run(dumps=DUMP_INFO)
record_time('dump_total_s', DUMP_INFO['seconds'])
print('每盤 embedding 完成：候選 %d 盤（可解析 %d）、題目 %d 盤（可解析 %d），%.0f 秒'
      % (N_CAND_ROWS, DUMP_INFO['cand_games.npz']['n_parsed'], N_QUERY_ROWS,
         DUMP_INFO['query_games.npz']['n_parsed'], DUMP_INFO['seconds']))
""", tags=["登記 2.1"])

# --------------------------------------------------------------------------- reproduction check (new)
md("### 本版新增：重現核對（登記 2.1）\n\n"
   "只用上一格存的每盤 embedding，照 prereg01 的順序與精度重算依窗口數加權的重心和題目向量，照官方做法用 L2 距離取 Top-5"
   "（`player_val_top5_repro.csv`），再和 prereg01 比。通過條件：400 題中至少 398 題的 Top-5 完全相同（SMOKE：全部相同），"
   "候選代號與順序相同，重心最大差距 ≤ 1e-4。結果寫 `repro_check.json`。沒過的話收尾格照樣打包，最後停下。")
code(r"""# 重現核對（本版新增，登記 2.1）：用每盤 embedding 重算 prereg01 的重心與 Top-5（E0），和 prereg01 的輸出比
_t_repro = time.time()


def weighted_means(keys, emb_sum, n_windows):
    # 和 prereg01 mean_embeddings 相同的順序與精度：依列順序逐盤在 float32 下累加，再除以窗口數的和（Python int）
    sums, counts = {}, {}
    for k, e, n in zip(keys, emb_sum, n_windows):
        if n == 0:
            continue          # prereg01 也跳過 sgf_windows 回傳 None 的盤
        sums[k] = sums.get(k, 0) + e
        counts[k] = counts.get(k, 0) + int(n)
    return {k: sums[k] / counts[k] for k in sums}


centroids = weighted_means(CAND['player_id'].astype(str).tolist(), CAND_SUM, CAND_NW)
names = sorted(centroids)          # 候選清單照玩家代號排序（同官方）
C = np.stack([centroids[n] for n in names])
queries = weighted_means(Q_IDS, Q_SUM, Q_NW)
rows, n_fallback = [], 0
for qid in TEST['question_id'].astype(str):
    q = queries.get(qid)
    if q is None:
        rows.append([qid] + names[:TOP_N])
        n_fallback += 1
        continue
    d = np.linalg.norm(C - q, axis=1)
    rows.append([qid] + [names[i] for i in np.argsort(d)[:TOP_N]])
_cols = ['question_id'] + ['top_%d' % i for i in range(1, TOP_N + 1)]
REPRO_TOP5 = pd.DataFrame(rows, columns=_cols)
REPRO_TOP5.to_csv(guard_write(REPRO_CSV + '.tmp'), index=False)
os.replace(REPRO_CSV + '.tmp', guard_write(REPRO_CSV))

# 和 prereg01 的 Top-5 逐題比（top_1…top_5 全部相同、順序也相同才算相同）
P01_TOP5 = pd.read_csv(P01_FILES['player_val_top5.csv'], dtype=str, keep_default_na=False)
_same_qids = P01_TOP5['question_id'].tolist() == REPRO_TOP5['question_id'].tolist()
_mismatch = []
if _same_qids:
    for _a, _b in zip(P01_TOP5[_cols].values.tolist(), REPRO_TOP5[_cols].values.tolist()):
        if _a != _b:
            _mismatch.append({'question_id': _a[0], 'prereg01': _a[1:], 'repro': _b[1:]})

# 和 prereg01 的重心、題目向量比（候選代號與順序必須相同；題目向量的差距只記錄）
with np.load(P01_FILES['player_val_embeddings.npz']) as _z:
    _p01_cids = [str(x) for x in _z['candidate_ids']]
    _p01_C = _z['centroids']
    _p01_qids = [str(x) for x in _z['question_ids']]
    _p01_Q = _z['queries']
_same_cids = _p01_cids == names and _p01_C.shape == C.shape
_centroid_diff = (float(np.max(np.abs(C.astype(np.float64) - _p01_C.astype(np.float64)))) if _same_cids and len(C)
                  else None)
_q_ids_emb = [q for q in TEST['question_id'].astype(str) if q in queries]
Q = np.stack([queries[q] for q in _q_ids_emb]) if _q_ids_emb else np.zeros((0, EMBED_DIM), np.float32)
_same_qids_emb = _p01_qids == _q_ids_emb and _p01_Q.shape == Q.shape
_query_diff = (float(np.max(np.abs(Q.astype(np.float64) - _p01_Q.astype(np.float64)))) if _same_qids_emb and len(Q)
               else None)

_n_q = len(REPRO_TOP5)
_n_identical = _n_q - len(_mismatch) if _same_qids else 0
_criteria = {
    'question_ids_same_order': _same_qids,
    'questions_count_ok': SMOKE or _n_q == 400,
    'top5_identical_ok': _same_qids and len(_mismatch) <= REPRO_MAX_MISMATCH,
    'candidate_ids_same_order': _same_cids,
    'centroid_diff_ok': _centroid_diff is not None and _centroid_diff <= REPRO_CENTROID_TOL,
}
_p01_gpu = CONFIG['p01_inputs']['p01_gpu']
REPRO = {
    'checked_utc': now_utc(),
    'smoke': SMOKE,
    'method': 'E0：依窗口數加權的重心、L2 距離、np.argsort 取前 %d（同 prereg01）；只用 dumps/ 的每盤 embedding 重算' % TOP_N,
    'n_questions': _n_q,
    'n_top5_identical': _n_identical,
    'n_top5_mismatch': len(_mismatch) if _same_qids else None,
    'top5_max_mismatch_allowed': REPRO_MAX_MISMATCH,
    'mismatched_questions': _mismatch,
    'n_fallback_questions': n_fallback,
    'n_centroids': len(names),
    'centroid_max_abs_diff': _centroid_diff,
    'centroid_tol': REPRO_CENTROID_TOL,
    'centroids_bitwise_identical': bool(_same_cids and np.array_equal(C, _p01_C)),
    'query_max_abs_diff_info_only': _query_diff,
    'query_ids_same_order_info_only': _same_qids_emb,
    'gpu': ENV['gpu'],
    'p01_gpu': _p01_gpu,
    'same_gpu_as_p01': (ENV['gpu'] == _p01_gpu) if _p01_gpu is not None else None,
    'torch': ENV['torch'],
    'p01_torch': CONFIG['p01_inputs']['p01_torch'],
    'weights_sha256': WEIGHTS_SHA256,
    'p01_top5_sha256': P01_SHA256['player_val_top5.csv'],
    'p01_embeddings_sha256': P01_SHA256['player_val_embeddings.npz'],
    'repro_top5_sha256': sha256_file(REPRO_CSV),
    'dumps': {k: DUMP_INFO[k] for k in ('cand_games.npz', 'query_games.npz')},
    'criteria': _criteria,
    'passed': all(_criteria.values()),
}
save_json('repro_check.json', REPRO)
update_run(repro_passed=REPRO['passed'])
record_time('repro_s', time.time() - _t_repro)
print('重現核對：Top-5 完全相同 %d/%d 題（最多可以差 %d 題）；重心最大差距 %s（上限 %g）；題目向量最大差距 %s（只記錄）'
      % (_n_identical, _n_q, REPRO_MAX_MISMATCH, _centroid_diff, REPRO_CENTROID_TOL, _query_diff))
print('GPU：這次 %s，prereg01 %s' % (ENV['gpu'], _p01_gpu))
if REPRO['passed']:
    print('重現核對通過')
else:
    print('重現核對沒過：%s。收尾格照樣打包 player_results.zip，然後停下'
          % '、'.join(k for k, v in _criteria.items() if not v))
""", tags=["登記 2.1"])

# --------------------------------------------------------------------------- final (base new cell)
j = base_new("c10-player-10", "# 收尾（營運性）")
s = BASE_SRC[j]
s = rep(s, """# 收尾（營運性）：記下這次執行完成；Colab 上把 Drive 的寫入全部送出再卸載
record_time('total_s', time.time() - RUN_T0)
update_run(status='completed', finished_utc=now_utc(), seconds=time.time() - RUN_T0)
print('完成。結果在', RESULT_DIR)
""", r"""# 收尾（營運性）：核對輸出、比對 prereg01 前後指紋、記下這次執行；打包；Colab 上把 Drive 的寫入全部送出再卸載
# （prereg02：重現核對沒過、輸出不對或 prereg01 目錄有變動時，照樣打包 player_results.zip，最後停下）
FINAL_PROBLEMS = []
for _name, _path, _rows in (('cand_games.npz', CAND_NPZ, N_CAND_ROWS), ('query_games.npz', QUERY_NPZ, N_QUERY_ROWS)):
    if not os.path.exists(_path) or sha256_file(_path) != DUMP_INFO[_name]['sha256']:
        FINAL_PROBLEMS.append('dumps/%s 不在或 SHA-256 和寫出時不同' % _name)
        continue
    with np.load(_path) as _z:
        if _z['emb_sum'].shape != (_rows, EMBED_DIM) or any(len(_z[_k]) != _rows for _k in _z.files if _k != 'weights_sha256'):
            FINAL_PROBLEMS.append('dumps/%s 的列數不是 %d' % (_name, _rows))
for _root, _dirs, _files in os.walk(RESULT_DIR):
    for _f in _files:
        if os.path.getsize(os.path.join(_root, _f)) >= 100 * 1024 * 1024:
            FINAL_PROBLEMS.append('%s ≥ 100 MB，結果 zip 會跳過它' % os.path.relpath(os.path.join(_root, _f), RESULT_DIR))
P01_AFTER = p01_fingerprint()
save_json('p01_readonly_after.json', P01_AFTER)
_changed = sorted(k for k in set(P01_BEFORE['files']) | set(P01_AFTER['files'])
                  if P01_BEFORE['files'].get(k) != P01_AFTER['files'].get(k))
P01_UNCHANGED = not _changed
if not P01_UNCHANGED:
    FINAL_PROBLEMS.append('prereg01 目錄有 %d 個檔和開頭不同（例如 %s；見 p01_readonly_before/after.json）'
                          % (len(_changed), _changed[:3]))
if not REPRO['passed']:
    FINAL_PROBLEMS.append('重現核對沒過：%s（見 repro_check.json）'
                          % '、'.join(k for k, v in REPRO['criteria'].items() if not v))
FINAL_STATUS = ('completed' if not FINAL_PROBLEMS else 'repro_failed' if not REPRO['passed'] else 'final_check_failed')
record_time('total_s', time.time() - RUN_T0)
update_run(status=FINAL_STATUS, finished_utc=now_utc(), seconds=time.time() - RUN_T0,
           p01_unchanged=P01_UNCHANGED, p01_changed_files=_changed, final_problems=FINAL_PROBLEMS)
print(('完成。結果在' if not FINAL_PROBLEMS else '結束，但有問題（見最後的說明）。結果在'), RESULT_DIR)
""")
s = rep(s, "with zipfile.ZipFile(RESULTS_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as _z:\n",
        "with zipfile.ZipFile(guard_write(RESULTS_ZIP), 'w', compression=zipfile.ZIP_DEFLATED) as _z:\n")
s = rep(s, "print('請下載這個檔回傳給 Claude（打分在本機做）：%s（%.1f MB）'\n",
        "print('請下載這個檔回傳給 Claude（融合與打分在本機做）：%s（%.1f MB）'\n")
s = rep(s, "    print('Drive 已寫入完成並卸載；要再跑請重新「全部執行」')\n",
        """    print('Drive 已寫入完成並卸載；要再跑請重新「全部執行」')
if FINAL_PROBLEMS:
    raise SystemExit('這次執行沒有通過（狀態 %s），請下載 player_results.zip 並截圖給 Claude：\\n%s'
                     % (FINAL_STATUS, '\\n'.join('  - ' + p for p in FINAL_PROBLEMS)))
""")
from_base(j, s, tags=["營運性", "登記 2.1"])

# --------------------------------------------------------------------------- notebook
_k = 0
nb_cells = []
for c, j in cells:
    if j is not None:
        c["id"] = base["cells"][j]["id"]
    else:
        c["id"] = "c10-player02-%02d" % _k
        _k += 1
    nb_cells.append(c)
nb = nbformat.v4.new_notebook()
nb.cells = nb_cells
nb.metadata = json.loads(json.dumps(base["metadata"]))
nb.metadata["colab"] = {"provenance": []}          # 不預設 A100／高 RAM：照第一格選 G4（拿不到時先等，不換 GPU）
nb.metadata["c10"] = {"notebook": OUT_REL, "base_notebook": BASE_REL, "base_notebook_sha256": BASE_SHA256,
                      "official_notebook": OFFICIAL_REL,
                      "prereg": "docs/PREREG_02_OPENING_FUSION.md section 2.1",
                      "deviations_from_base": DEV_STRINGS}
nbformat.validate(nb)

# --------------------------------------------------------------------------- checks
assert len({c["id"] for c in nb.cells}) == len(nb.cells)
_used = [j for _c, j in cells if j is not None]
assert len(_used) == len(set(_used)) and _used == sorted(_used), _used      # base cells used once, base order kept
# refusal check: no cell may mention the answer files, except the base prepare cell's assert that refuses a bundle
# containing one (inherited unchanged from prereg01)
FORBIDDEN = "ans" + "wers"          # spelled in two parts: no p02 source contains the word
_refusal = [ln for ln in BASE_SRC[PREPARE_BASE].splitlines() if FORBIDDEN in ln]
assert len(_refusal) == 1 and _refusal[0].startswith("assert not any("), _refusal
for k, (c, j) in enumerate(cells):
    meta = c.metadata["c10"]
    if j is not None:
        assert c.cell_type == base["cells"][j]["cell_type"]
        assert (c.source == BASE_SRC[j]) == (meta["change"] == "identical"), k
    if c.cell_type == "code":
        compile(c.source, "<cell %d>" % k, "exec")         # plain Python only (run_local_smoke.py compiles each cell)
        assert "import Pool" not in c.source or j == base_official(1), k   # smoke swaps Pool after each cell
    rest = "\n".join(ln for ln in c.source.splitlines() if ln not in _refusal)
    assert FORBIDDEN not in rest, k
# the training code is gone: none of these names is used in any code cell (comments and strings may still
# mention what was removed), and nothing calls torch.save
_names = set()
for c in nb.cells:
    if c.cell_type == "code":
        _toks = [t for t in tokenize.generate_tokens(io.StringIO(c.source).readline)
                 if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)]
        _names |= {t.string for t in _toks if t.type == tokenize.NAME}
        for _a, _b, _d in zip(_toks, _toks[1:], _toks[2:]):
            assert (_a.string, _b.string, _d.string) != ("torch", ".", "save"), "torch.save"
for _word in ("load_rows", "c10data", "TRAIN_DIR", "SPLIT_CSV", "FEAT_DIR", "INDEX_PATH", "extract_all", "tarfile",
              "train_loader", "TripletDataset", "optimizer", "train_step", "mean_embeddings", "predict"):
    assert _word not in _names, _word

with open(path(OUT_REL), "w", encoding="utf-8", newline="\n") as f:
    nbformat.write(nb, f)
# read back: cells kept identical are still byte-identical to the base
with open(path(OUT_REL), encoding="utf-8") as f:
    _back = json.load(f)
for c in _back["cells"]:
    m = c["metadata"]["c10"]
    if m["change"] == "identical":
        assert "".join(c["source"]) == BASE_SRC[m["base_cell"]] and c["id"] == base["cells"][m["base_cell"]]["id"]

# --------------------------------------------------------------------------- per-cell diff vs prereg01_player.ipynb
out = []
out.append("# prereg02 player：colab/prereg01_player.ipynb（底稿）與 colab/prereg02_player.ipynb 的逐格 unified diff"
           "（程式格、markdown 格、notebook metadata）")
out.append("# 底稿：%s  sha256=%s" % (BASE_REL, BASE_SHA256))
out.append("# 本版：%s  sha256=%s" % (OUT_REL, file_sha256(OUT_REL)))
out.append("# 產生方式：colab/build_prereg02_player_nb.py；difflib.unified_diff，依 cell metadata c10.base_cell 配對底稿的格；"
           "新增的格對空白比。cell 編號從 0 起算。")
out.append("# 每一處差異的說明見 notebook 第 1 格「與 prereg01 notebook 的差異」（共 %d 項）；底稿沒有列在對照表的格都拿掉了。"
           % len(DEVIATIONS))
summary = []
body = []
for k, c in enumerate(nb.cells):
    meta = c.metadata["c10"]
    kind = "程式格" if c.cell_type == "code" else "markdown"
    j = meta["base_cell"]
    if j is not None:
        old_src = BASE_SRC[j]
        a = "prereg01_player.ipynb cell %d（id %s%s）" % (
            j, base["cells"][j]["id"],
            "，官方 cell %d" % meta["official_cell"] if meta["official_cell"] is not None else "")
        status = "完全相同" if meta["change"] == "identical" else "改動"
    else:
        old_src = ""
        a = "/dev/null"
        status = "新增"
    b = "prereg02_player.ipynb cell %d（id %s）" % (k, c["id"])
    tags = "、".join(meta.get("tags") or []) if c.cell_type == "code" else ""
    summary.append("#   本版 cell %2d（%s）← %s：%s%s" % (k, kind, a, status, "；分類：%s" % tags if tags else ""))
    diff = list(difflib.unified_diff(old_src.splitlines(), c.source.splitlines(), fromfile=a, tofile=b, lineterm=""))
    body.append("")
    body.append("=" * 100)
    body.append("本版 cell %d（%s）↔ %s（%s）" % (k, kind, a, status))
    body.append("=" * 100)
    body.extend(diff if diff else ["（完全相同）"])
dropped = ["#   底稿 cell %2d（%s，id %s%s）：%s" % (
    j, "程式格" if base["cells"][j]["cell_type"] == "code" else "markdown", base["cells"][j]["id"],
    "，官方 cell %d" % base["cells"][j]["metadata"]["c10"]["official_cell"]
    if base["cells"][j]["metadata"]["c10"]["official_cell"] is not None else "",
    BASE_SRC[j].splitlines()[0][:70]) for j in range(len(base["cells"])) if j not in _used]
out.append("# 對照表（本版每一格 → 底稿的格）：")
out.extend(summary)
out.append("# 拿掉的底稿格（%d 格）：" % len(dropped))
out.extend(dropped)
out.extend(body)
_meta_old = json.dumps(base["metadata"], ensure_ascii=False, indent=1, sort_keys=True).splitlines()
_meta_new = json.dumps(nb.metadata, ensure_ascii=False, indent=1, sort_keys=True).splitlines()
out.append("")
out.append("=" * 100)
out.append("notebook metadata")
out.append("=" * 100)
out.extend(list(difflib.unified_diff(_meta_old, _meta_new, fromfile="prereg01_player.ipynb metadata",
                                     tofile="prereg02_player.ipynb metadata", lineterm="")) or ["（完全相同）"])
os.makedirs(os.path.dirname(path(DIFF_REL)), exist_ok=True)
with open(path(DIFF_REL), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print("wrote", OUT_REL, file_sha256(OUT_REL))
print("wrote", DIFF_REL, file_sha256(DIFF_REL))
print("cells", len(nb.cells), "| from base", len(_used), "| new", len(nb.cells) - len(_used),
      "| base cells dropped", len(dropped), "| deviations", len(DEVIATIONS))
