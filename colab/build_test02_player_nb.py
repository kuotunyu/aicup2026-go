"""Build colab/test02_player.ipynb (competition 1, TEAM_11249; test day, inference only) from the committed
colab/prereg02_player.ipynb, and write results/test02/notebook_diff_player.txt (per-cell unified diff against
prereg02_player.ipynb, plus the notebook metadata).

Spec: results/test02/design/spec.md 2.0, 2.2 and 2.3; registration docs/PREREG_02_OPENING_FUSION.md section 9
(rev 5).

* The base notebook is read from the repo and its SHA-256 is asserted (BASE_SHA256 below).
* Cells kept from the base are picked by their cell id and changed only with rep(): exact string replacement
  whose target must occur exactly once, so every unchanged line stays byte-identical. The per-game embedding
  cell is emitted three times (public, private, seam_val); each copy is a one-line POOL preamble plus the same
  body, and the builder asserts the three bodies are identical text.
* Segment identity: official cell 20, the prepare / environment / official 1 / official 2 cells and the
  reproduction cell are kept verbatim; game_embeddings(), save_npz() and the `with Pool(NPROC) as pool:` block
  (the CAND and TEST loops) are extracted with ast.get_source_segment from the base cell and from every copy of
  the new cell and asserted byte-equal.
* The prereg02 pins (weights, dumps) are read from the repo at build time and asserted consistent.
* Every code cell must compile, the notebook names the label files only in the inherited prepare-cell refusal
  line, cell ids are unique, and a rebuild gives byte-identical outputs.
* config.json 'notebook_sha256' is the SHA-256 of the notebook's cell sources with that value blanked
  (notebook_source_sha256() below; t02_player.py recomputes it from the committed notebook).

Usage (repo root optional; defaults to the repo that contains this file):
    .venv/Scripts/python.exe colab/build_test02_player_nb.py [REPO_ROOT] [--out-dir DIR]
    --out-dir DIR writes test02_player.ipynb and notebook_diff_player.txt into DIR instead of the repo
    (the rebuild test compares them byte for byte with the committed files).
"""
import argparse
import ast
import difflib
import hashlib
import io
import json
import os
import re
import tokenize

import nbformat

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("root", nargs="?", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                help="repo root (default: the repo that contains this script)")
ap.add_argument("--out-dir", default=None, help="write the two outputs here instead of the repo")
ARGS = ap.parse_args()
ROOT = os.path.abspath(ARGS.root)

BASE_REL = "colab/prereg02_player.ipynb"
BASE_SHA256 = "738c1dd81aa8a93953e45019cf6563e9b2b44ffa7cf846763c51fcc8190a5c12"
OUT_REL = "colab/test02_player.ipynb"
DIFF_REL = "results/test02/notebook_diff_player.txt"
P02_REPO_DIR = "results/prereg02/player"          # repo copy of MyDrive/C10/results/prereg02/player (frozen)
NB_SHA_PLACEHOLDER = "@@NB_SOURCE_SHA256@@"
NB_SHA_RE = re.compile(r"NB_SOURCE_SHA256 = '([0-9a-f]{64})'")
# the six official test file names (tutorial 359e06f README:40-45); competition 1 uses the last four
TEST_NAMES = ["rank_prediction_test_public.csv", "rank_prediction_test_private.csv",
              "player_identification_test_public.csv", "player_identification_test_private.csv",
              "player_identification_candidates_public.csv", "player_identification_candidates_private.csv"]
POOL_ORDER = ("public", "private", "seam_val")      # the base dump cell deletes REPRO_*: seam_val runs last


def path(rel):
    return os.path.join(ROOT, *rel.split("/"))


def out_path(rel):
    return os.path.join(os.path.abspath(ARGS.out_dir), os.path.basename(rel)) if ARGS.out_dir else path(rel)


def file_sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def text_sha256(rel):
    with open(path(rel), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def rep(src, old, new):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"replacement target occurs {n} times:\n{old}")
    return src.replace(old, new)


def notebook_source_sha256(cells):
    """cells: [(cell_type, source)] with the NB_SOURCE_SHA256 value replaced by the placeholder."""
    blob = json.dumps([[t, s] for t, s in cells], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- base notebook
with open(path(BASE_REL), "rb") as f:
    _raw = f.read()
if hashlib.sha256(_raw).hexdigest() != BASE_SHA256:
    raise SystemExit("%s changed: sha256 %s != %s" % (BASE_REL, hashlib.sha256(_raw).hexdigest(), BASE_SHA256))
base = json.loads(_raw.decode("utf-8"))
BASE_SRC = ["".join(c["source"]) for c in base["cells"]]


def base_cell(cell_id, head):
    """The base cell with this id, checked by its first line."""
    hits = [j for j, c in enumerate(base["cells"]) if c["id"] == cell_id]
    assert len(hits) == 1, (cell_id, hits)
    assert BASE_SRC[hits[0]].startswith(head), (cell_id, BASE_SRC[hits[0]][:80])
    return hits[0]


# --------------------------------------------------------------------------- prereg02 pins from the repo
with open(path(P02_REPO_DIR + "/frozen_config.json"), encoding="utf-8") as f:
    P02_FROZEN = json.load(f)
with open(path(P02_REPO_DIR + "/run_log.json"), encoding="utf-8") as f:
    P02_RUN = json.load(f)["runs"][-1]
with open(path(P02_REPO_DIR + "/env.json"), encoding="utf-8") as f:
    P02_ENV = json.load(f)
WEIGHTS_SHA256 = "356eb4f65488925e2a552ea382979780c903ef515b85d329a3805db837b7773b"
P02_DUMP_SHA256 = {"cand_games.npz": "56ed03d12e9a79871a21b5f69dc4bd4bb8c8aa69b1544cb8ec799e80227cfdea",
                   "query_games.npz": "665d0888a93f6632eb19edac76abf996d5499cd27c683d9844b97cb274b6f689"}
assert P02_FROZEN["smoke"] is False and P02_FROZEN["selected"]["id"] == "Q04"
assert P02_FROZEN["inputs"]["cand_games_npz"]["weights_sha256"] == WEIGHTS_SHA256
assert P02_FROZEN["inputs"]["cand_games_npz"]["sha256"] == P02_DUMP_SHA256["cand_games.npz"]
assert P02_FROZEN["inputs"]["repro_local_json"]["query_games_npz_sha256"] == P02_DUMP_SHA256["query_games.npz"]
assert P02_RUN["status"] == "completed" and P02_RUN["dumps"]["weights_sha256"] == WEIGHTS_SHA256
for _k, _v in P02_DUMP_SHA256.items():
    assert P02_RUN["dumps"][_k]["sha256"] == _v, _k
    assert file_sha256(path(P02_REPO_DIR + "/dumps/" + _k)) == _v, _k
P02_GPU = P02_ENV["gpu"]
assert P02_GPU == "NVIDIA RTX PRO 6000 Blackwell Server Edition", P02_GPU

# --------------------------------------------------------------------------- cells
cells = []          # (nbformat cell, base index or None)


def md(text):
    c = nbformat.v4.new_markdown_cell(text, metadata={"c10": {"base_cell": None, "base_id": None,
                                                              "change": "new"}})
    cells.append((c, None, None))


def code(src, tags):
    c = nbformat.v4.new_code_cell(src, metadata={"c10": {"base_cell": None, "base_id": None, "change": "new",
                                                         "tags": list(tags)}})
    cells.append((c, None, None))


def from_base(j, src, tags=None, cid=None):
    b = base["cells"][j]
    change = "identical" if src == BASE_SRC[j] else "modified"
    meta = {"c10": {"base_cell": j, "base_id": b["id"],
                    "official_cell": b["metadata"]["c10"].get("official_cell"), "change": change}}
    if b["cell_type"] == "code":
        meta["c10"]["tags"] = list(tags) if tags is not None else list(b["metadata"]["c10"].get("tags") or [])
    maker = nbformat.v4.new_code_cell if b["cell_type"] == "code" else nbformat.v4.new_markdown_cell
    cells.append((maker(src, metadata=meta), j, cid))


# 和 prereg02 notebook 的全部差異：第 1 格、config.json（changes_from_prereg02）、notebook metadata 共用這一份
T02_CHANGES = [
    ("登記 9.1", "只推論，和 prereg02 相同：模型、權重（prereg01 的 `best_model.pth`，prereg02 用的同一份）、特徵、每盤 "
                 "embedding 的算法都不變。準備格、官方 cell 1、環境紀錄格、官方 cell 2、官方 cell 20、重現核對格和 prereg02 "
                 "逐位元組相同；每盤 embedding 格裡的 `game_embeddings()`、`save_npz()` 和候選／題目兩段迴圈（`with Pool(NPROC)` "
                 "整段）也逐位元組相同（產生本 notebook 時用 ast 逐段核對）。"),
    ("登記 9.4", "每個候選池各算一次每盤 embedding：同一格程式印三次，順序 public → private → seam_val（本機驗證池）。三格只差"
                 "第一行 `POOL = …; QUESTIONS_CSV, CANDIDATES_CSV = POOL_FILES[POOL]`，其餘逐字相同。輸出寫到 "
                 "`results/test02/player/<POOL>/dumps/`（cand_games.npz、query_games.npz，欄位和 prereg02 相同），另寫 "
                 "`_INPUT.json`（題目檔、候選檔的檔名與 SHA-256）。盤數和題數依 preflight 算好的 `EXPECTED_COUNTS[POOL]` 核對"
                 "（seam_val 仍是 prereg01 記的 40,000／4,935／400）。seam_val 放最後，因為 dump 格開頭會刪掉上一次的重現核對結果。"),
    ("登記 9.2", "接縫核對（notebook 這一層）：seam_val 用 prereg02 原封不動的重現核對格（和 prereg01 至少 398/400 題 Top-5 "
                 "相同、重心最大差 ≤ 1e-4，SMOKE 全部相同），沒過時狀態記 repro_failed、照樣打包後停下。新增 `seam_p02.json`，"
                 "只記錄 seam_val 的逐盤 embedding 和 prereg02 的 dumps 是否逐位元組相同（最大差、兩次的 GPU／torch／cuDNN）；"
                 "判定在本機（`tools/t02_player.py`）。"),
    ("營運性", "測試包：讀 `MyDrive/C10/C10_test_v1.zip`。成員只能是 `tests/MANIFEST.json` 和六個官方檔名，六個都要在；"
               "MANIFEST 的 format 要是 `c10-test02 manifest v1`，kind 正式執行要是 official 或 rehearsal（SMOKE 要是 smoke），"
               "T1 的競賽一結果要是 PASS、NORMALIZED 或 FAIL_OPENING（登記第 9 節第 6 條：T1.4／T1.5 不過時競賽一照常每池程序，"
               "結果記進 config.json 的 `test_bundle.t1`；FAIL_STRUCT 或沒有結果就停下）。只解開競賽一的四個檔和 MANIFEST.json 到 `WORK/tests/`，逐檔核對 "
               "SHA-256 和 MANIFEST 相同；競賽二的兩個檔只核對檔名在不在，不讀內容。任何一項不符就停下。"),
    ("營運性", "快速格式檢查（T1 的一部分）：兩份題目檔各 400 題、沒有 BOM、`question_id` 不重複且前綴是 pub_／priv_、"
               "`num_games` 是 5–20 的整數、`sgf_1..20` 和 `color_1..20` 欄都在、剛好前 `num_games` 格有棋、執色是 B／W；"
               "兩份候選檔各 40,000 列＝400 位 × 各剛好 100 盤、player_id 前綴 pub_p_／priv_p_、game_id 不重複且前綴 pub_／priv_、"
               "執色是 B／W；兩池不共用玩家和棋。不符就列出來後停下。"),
    ("營運性", "唯讀保護擴大到 prereg02：設定格多了 `P02_ROOT`（`results/prereg02`，SMOKE 是 `prereg02_smoke`），"
               "`guard_write` 同時擋 prereg01 和 prereg02。preflight 開頭和收尾格各記一次兩個目錄的指紋"
               "（`readonly_before.json`、`readonly_after.json`，取代 `p01_readonly_*.json`），有變動時狀態記 readonly_changed。"),
    ("營運性", "preflight 另外核對：prereg02 的 `dumps/cand_games.npz`、`query_games.npz` 的 SHA-256（正式執行時和 repo 相同；"
               "接縫的參照）、兩個檔記的 `weights_sha256` 等於 `best_model.pth` 的 SHA-256、權重是 prereg02 用的那一份；"
               "stale-dump guard：已經存在的 `<POOL>/dumps/_INPUT.json` 和這次的輸入檔不同就停下。GPU 仍然只接受 G4。"),
    ("營運性", "結果目錄改成 `results/test02/player/`（SMOKE：`test02_smoke`；測試包 kind 是 rehearsal：`test02_rehearsal`），"
               "結果 zip 固定叫 `test02_player_results.zip`，放在 `results/test02/`，成員從 `player/` 起算。`run_log.json` "
               "最上層另記 `status`（running、completed、repro_failed、readonly_changed、final_check_failed、preflight_failed），"
               "和最後一筆 run 的 status 相同。"),
    ("營運性", "config.json 的新欄位：`notebook_sha256`（本 notebook 各格原始碼的 SHA-256，算法見 `notebook_sha256_rule`）、"
               "`rehearsal`、`gpu`／`torch`／`cudnn`、`weights`、`test_bundle`（zip 與 MANIFEST 的 SHA-256、競賽一四個檔的 "
               "SHA-256）、`sets`（三個池的輸入檔 SHA-256 與列數）、`p02_inputs`、`changes_from_prereg02`（本表）。"),
    ("營運性", "notebook：沿用的格保留 prereg02 的 cell id（三份每盤 embedding 格加上 `-<POOL>`），新增的格用 "
               "`c10-player-t02-NN`；每格的 metadata `c10` 記對應的底稿格與是否改動。notebook metadata 的 `c10` 改記底稿和本表。"),
]
T02_STRINGS = ["[%s] %s" % (t, s) for t, s in T02_CHANGES]

# --------------------------------------------------------------------------- 0 title (new)
md(r"""# C10 競賽一（玩家辨識）：測試集的每盤 embedding 與接縫核對（事前登記 02 rev 5 第 9 節，只推論）

**用途**：11/04 起的測試日用。不訓練、不調參。用 prereg02 同一份權重（prereg01 的 `best_model.pth`，只讀）和同一段程式，替三個候選池各存一份每盤 embedding：官方測試集的 **public**、**private**，以及本機驗證池 **seam_val**。seam_val 用來做接縫核對：證明這次 Colab 算出來的東西和 prereg02 相同。帶回本機後，由 `tools/t02_player.py` 在每個池各做自我檢索、閘門、防護和融合，組成 `submission_player.csv`。**這裡不打分**，Colab 上也沒有答案檔。

**對應事前登記**：`docs/PREREG_02_OPENING_FUSION.md` 第 9 節（rev 5）；實作規格 `results/test02/design/spec.md` 2.0、2.2、2.3。底稿是 `colab/prereg02_player.ipynb`，差異列在本格最後；逐格 diff 在 `results/test02/notebook_diff_player.txt`。

## 怎麼跑

1. 用**跑 prereg01、prereg02 的同一個 Google 帳號**：權重和 prereg02 的結果在那個帳號的 Drive（`MyDrive/C10/results/prereg01/`、`results/prereg02/`）。打包檔 `C10_bundle_v1.zip` 沿用，不用重傳。
2. 把 Claude 給的 `C10_test_v1.zip` 放進 `MyDrive/C10/`（檔名不要改），這份 `test02_player.ipynb` 也放進 `MyDrive/C10/`，用 Colab 開啟。
3. 「執行階段 → 變更執行階段類型」選 **G4 GPU**。G4 是灰色、點不了時，**不要換別的 GPU**：隔一段時間再試；等了一天還是拿不到，告訴 Claude（登記第 9 節第 10 條）。
4. 「執行階段 → 全部執行」，照畫面允許 Google Drive 存取。G4 約 15–20 分鐘（推估，還沒實測）。跑完前不關分頁、電腦不睡眠。
5. 最後一行印出「Drive 已寫入完成並卸載」後，zip 已經在 Drive 上：先按「執行階段 → 中斷連線並刪除執行階段」（開著會一直扣運算單元），再下載 `MyDrive/C10/results/test02/test02_player_results.zip`（彩排時在 `results/test02_rehearsal/`），把位置告訴 Claude。下面接著出現紅字時，zip 照樣下載，並把紅字截圖給 Claude，**不要自己重跑**。

詳細步驟以 `docs/COLAB_RUN.md` 的「測試集（test02）」一節為準；和這裡不一致時，以那一節為準。

## 會停下來的情況（依序）

GPU 不是 G4；prereg01 的權重或 prereg02 的 dumps 對不上（開錯帳號、檔案被動過）；找不到測試包、測試包的成員或 SHA-256 和 MANIFEST 不符；測試檔的快速格式檢查沒過；上一次用不同測試檔算的 dump 還在；Drive 剩不到 0.5 GB；任何寫入落在 prereg01、prereg02 的目錄。這些都在推論前停下。推論後：重現核對沒過（狀態 repro_failed）或唯讀目錄有變動（readonly_changed），收尾格照樣打包再停下。

## 結果寫到哪裡

`MyDrive/C10/results/test02/player/`（整個資料夾打包成 `results/test02/test02_player_results.zip`）

| 檔案 | 內容 |
|---|---|
| `<POOL>/dumps/cand_games.npz` | 候選每盤一列（候選檔的列順序）：game_id、player_id、color、emb_sum、n_windows、weights_sha256；POOL 是 public、private、seam_val |
| `<POOL>/dumps/query_games.npz` | 題目每盤一列（題目檔的列順序、slot 由小到大）：question_id、slot、color、emb_sum、n_windows、parsed |
| `<POOL>/dumps/_INPUT.json` | 這個池用的題目檔、候選檔的檔名與 SHA-256 |
| `repro_check.json`、`player_val_top5_repro.csv` | seam_val 的重現核對（和 prereg01 比，登記 2.1 的條件） |
| `seam_p02.json` | seam_val 的逐盤 embedding 和 prereg02 是否逐位元組相同（只記錄） |
| `readonly_before.json`、`readonly_after.json` | prereg01、prereg02 目錄開頭與結尾的指紋，必須相同 |
| `config.json`、`env.json`、`run_log.json`、`timings.json` | 設定（含測試包、權重、三個池的輸入檔 SHA-256）、環境、每次執行的紀錄與狀態、耗時 |

prereg01、prereg02 的目錄**只讀**：這份 notebook 任何寫入落在那裡都會被擋下。

## 中斷或出錯

- **斷線**：重新「全部執行」，整份重跑（沒有接續）。
- **preflight 停下**：照印出的中文說明處理。這時還沒開始推論，但執行階段開著就會一直扣運算單元：截圖後先按「執行階段 → 中斷連線並刪除執行階段」。
- **重現核對沒過或狀態不是 completed**：最後一格照樣產生 zip，然後停下並印出原因。下載 zip、截圖給 Claude；這一次的測試輸出不會拿去上傳（登記第 9 節第 2 條）。

## 本機冒煙測試

`colab/run_local_smoke.py` 會設 `C10_SMOKE=1`，讀同一個 `--scratch` 裡的 `results/prereg01_smoke/`、`results/prereg02_smoke/`（只讀）和 kind 為 smoke 的 `C10_test_v1.zip`，結果寫到 `results/test02_smoke/player/`。SMOKE 的縮小方式和 prereg02 冒煙測試相同，正式執行時完全不生效。

## 與 prereg02 notebook 的差異

**模型、權重、特徵、每盤 embedding 的算法、重現核對都和 prereg02 相同**；差別是每個候選池各算一次、讀測試包、多了 prereg02 的唯讀保護、測試檔的快速格式檢查和 seam_p02 紀錄。

""" + "\n".join("%d. **%s** %s" % (i, t, s) for i, (t, s) in enumerate(T02_CHANGES, 1)))

# --------------------------------------------------------------------------- setup (base, rep)
j = base_cell("c10-player-02", "# 設定格（營運性）")
s = BASE_SRC[j]
s = rep(s, """# 正式結果與 SMOKE 結果分開放，絕不混用（prereg02：結果寫到 prereg02／prereg02_smoke）
RESULT_DIR = os.path.join(DRIVE_DIR, 'results', 'prereg02_smoke' if SMOKE else 'prereg02', 'player')
""", """# test02：輸出根目錄 results/test02；SMOKE 是 test02_smoke，測試包 MANIFEST 的 kind 是 rehearsal 時是 test02_rehearsal
# （正式、彩排、SMOKE 的結果分開放，絕不混用；kind 在 preflight 會再完整核對一次）
TEST_ZIP = os.path.join(DRIVE_DIR, 'C10_test_v1.zip')
TEST_KIND = None
if os.path.exists(TEST_ZIP):
    import json as _json
    import zipfile as _zipfile
    try:
        with _zipfile.ZipFile(TEST_ZIP) as _z:
            TEST_KIND = _json.loads(_z.read('tests/MANIFEST.json').decode('utf-8')).get('kind')
    except (KeyError, ValueError, OSError, _zipfile.BadZipFile) as _e:
        print('注意：讀不到測試包的 MANIFEST（%s: %s），preflight 會停下並說明' % (type(_e).__name__, _e))
REHEARSAL = (not SMOKE) and TEST_KIND == 'rehearsal'
RESULT_ROOT = os.path.join(DRIVE_DIR, 'results',
                           'test02_smoke' if SMOKE else 'test02_rehearsal' if REHEARSAL else 'test02')
RESULT_DIR = os.path.join(RESULT_ROOT, 'player')
# 競賽一的測試檔：每個池一份題目檔、一份候選檔（官方檔名，tutorial 359e06f README）
TEST_POOLS = {'public': ('player_identification_test_public.csv', 'player_identification_candidates_public.csv'),
              'private': ('player_identification_test_private.csv', 'player_identification_candidates_private.csv')}
""")
s = rep(s, """P01_PLAYER_CKPT = os.path.join(P01_RESULT_DIR, 'ckpt')
""", """P01_PLAYER_CKPT = os.path.join(P01_RESULT_DIR, 'ckpt')
# prereg02（只讀）：接縫核對的參照（逐盤 embedding）；SMOKE 時讀本機冒煙測試產生的 prereg02_smoke
P02_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg02_smoke' if SMOKE else 'prereg02')
P02_RESULT_DIR = os.path.join(P02_ROOT, 'player')
""")
s = rep(s, """def guard_write(path):
    # 所有寫檔、建資料夾、刪檔、複製、解壓、打包的目的地都先經過這裡：落在 prereg01 底下就停止。
    # 回傳 path，可以直接包在呼叫裡
    if _inside(path, P01_ROOT):
        raise RuntimeError('拒絕寫入 prereg01 的目錄（只讀）：%s' % path)
    return path
""", """def guard_write(path):
    # 所有寫檔、建資料夾、刪檔、複製、解壓、打包的目的地都先經過這裡：落在 prereg01 或 prereg02 底下就停止。
    # 回傳 path，可以直接包在呼叫裡
    for _ro, _label in ((P01_ROOT, 'prereg01'), (P02_ROOT, 'prereg02')):
        if _inside(path, _ro):
            raise RuntimeError('拒絕寫入 %s 的目錄（只讀）：%s' % (_label, path))
    return path
""")
s = rep(s, "print('RESULT_DIR', RESULT_DIR)\nprint('P01_ROOT  ', P01_ROOT, '（只讀）')\n",
        "print('RESULT_DIR', RESULT_DIR, '| REHEARSAL', REHEARSAL)\nprint('P01_ROOT  ', P01_ROOT, '（只讀）')\n"
        "print('P02_ROOT  ', P02_ROOT, '（只讀）')\nprint('TEST_ZIP  ', TEST_ZIP, '| kind', TEST_KIND)\n")
from_base(j, s, tags=["營運性", "登記 9.4"])

# --------------------------------------------------------------------------- prepare, official 1, environment, official 2: identical
PREPARE_BASE = base_cell("c10-player-03", "# 準備格（營運性）")
from_base(PREPARE_BASE, BASE_SRC[PREPARE_BASE])
OFFICIAL1_BASE = base_cell("c27134cb", "import glob")
from_base(OFFICIAL1_BASE, BASE_SRC[OFFICIAL1_BASE])
ENV_BASE = base_cell("c10-player-04", "# 環境紀錄格（營運性）")
from_base(ENV_BASE, BASE_SRC[ENV_BASE])
OFFICIAL2_BASE = base_cell("7bfd8e59", "# 官方資料放置位置")
from_base(OFFICIAL2_BASE, BASE_SRC[OFFICIAL2_BASE])

# --------------------------------------------------------------------------- config record (base, rep)
j = base_cell("c10-player-05", "# 設定紀錄格（營運性）")
s = BASE_SRC[j]
s = rep(s, "# 設定紀錄格（營運性）：推論設定、重現核對的條件、實際 NPROC、是否 SMOKE、和 prereg01 notebook 的差異 → config.json\n"
           "# （prereg01 notebook 和官方的差異記在 colab/prereg01_player.ipynb 的 metadata，這裡不重抄）\n",
        "# 設定紀錄格（營運性）：推論設定、重現核對的條件、實際 NPROC、是否 SMOKE／彩排、和 prereg02 notebook 的差異 → config.json\n"
        "# （prereg02 notebook 和 prereg01 的差異記在 colab/prereg02_player.ipynb 的 metadata，這裡不重抄）\n")
_a = s.index("DEVIATIONS = [\n")
_b = s.index("save_json('config.json', CONFIG)\n")
s = rep(s, s[_a:_b], "T02_CHANGES = [\n" + "".join("    %r,\n" % d for d in T02_STRINGS) + "]\n" + r"""# 重現核對的條件（登記 2.1，seam_val 沿用）：Top-5 完全相同的題至少 400 − 2 = 398 題（SMOKE：全部相同）；重心最大差距 ≤ 1e-4
REPRO_MAX_MISMATCH = 0 if SMOKE else 2
REPRO_CENTROID_TOL = 1e-4
# 本 notebook 各格原始碼的 SHA-256（產生時算好；算法見 notebook_sha256_rule，本機 t02_player.py 用 repo 的 notebook 重算比對）
NB_SOURCE_SHA256 = '@@NB_SOURCE_SHA256@@'
CONFIG = {
    'notebook': 'colab/test02_player.ipynb',
    'notebook_sha256': NB_SOURCE_SHA256,
    'notebook_sha256_rule': ("sha256 of json.dumps([[cell_type, source], ...], ensure_ascii=False, separators=(',', ':')) "
                             "over all cells, where the 64-hex value assigned to NB_SOURCE_SHA256 in this cell is "
                             "replaced by the placeholder @@NB_SOURCE_SHA256@@"),
    'base_notebook': 'colab/prereg02_player.ipynb',
    'base_notebook_sha256': '@@BASE_SHA256@@',
    'official_notebook': 'third_party/aicup2026_tutorial/player-identification.ipynb',
    'prereg': 'docs/PREREG_02_OPENING_FUSION.md 第 9 節（rev 5：測試日只推論、每個候選池各算一次、接縫核對）；'
              '規格 results/test02/design/spec.md 2.0、2.2、2.3',
    'split': MANIFEST.get('split'),
    'in_colab': IN_COLAB,
    'smoke': SMOKE,
    'rehearsal': REHEARSAL,
    'gpu': ENV['gpu'],
    'torch': ENV['torch'],
    'cudnn': ENV['cudnn'],
    'inference': {
        'weights': 'prereg01 的 ckpt/best_model.pth（只讀；prereg02 用的同一份，SHA-256 見 weights）',
        'model': 'GoPlayerResNet() 預設值：in_channels 17、num_channels 128、num_blocks 16、embed_dim 256',
        'features': 'SGFParsePlayerIdentification() 預設 history_length 8：8 層自己 + 8 層對手 + 1 層執色 = 17 層',
        'precision': 'fp32（和 prereg01、prereg02 的推論相同，沒有 autocast）',
        'per_game': '每盤 sgf_windows → 一盤一次 embed（GPU_BATCH 4096）→ emb_sum = e.sum(0)（float32）、n_windows = len(e)；'
                    '每次平行解析 CHUNK 256 盤（同 prereg02）',
        'pools': 'public、private（官方測試集）與 seam_val（本機驗證池，接縫核對），依序各算一次',
        'TOP_N': TOP_N,
    },
    'repro_criteria': {
        'top5_max_mismatch': REPRO_MAX_MISMATCH,
        'centroid_max_abs_diff': REPRO_CENTROID_TOL,
        'rule': 'seam_val：top_1…top_5 完全相同的題 ≥ 題數 − top5_max_mismatch；候選代號與順序相同；重心最大差距 ≤ centroid_max_abs_diff',
    },
    'nproc': NPROC,
    'device': str(DEVICE),
    'mp_start_method': multiprocessing.get_start_method(),
    'smoke_settings': ({k: v for k, v in globals().items() if k.startswith('SMOKE_')} if SMOKE else None),
    'paths': {'DRIVE_DIR': DRIVE_DIR, 'WORK': WORK, 'RESULT_ROOT': RESULT_ROOT, 'RESULT_DIR': RESULT_DIR,
              'P01_ROOT': P01_ROOT, 'P02_ROOT': P02_ROOT, 'CKPT_DIR': CKPT_DIR, 'TEST_ZIP': TEST_ZIP,
              'SEAM_QUESTIONS_CSV': QUESTIONS_CSV, 'SEAM_CANDIDATES_CSV': CANDIDATES_CSV},
    'weights': None,          # preflight 會填 {path, sha256}
    'test_bundle': None,      # preflight 會填
    'sets': None,             # preflight 會填
    'p01_inputs': None,       # preflight 會填
    'p02_inputs': None,       # preflight 會填
    'changes_from_prereg02': T02_CHANGES,
}
""".replace("@@BASE_SHA256@@", BASE_SHA256))
s = rep(s, "print('config.json 已寫入；和 prereg01 notebook 的差異 %d 項' % len(DEVIATIONS))\n",
        "print('config.json 已寫入；和 prereg02 notebook 的差異 %d 項' % len(T02_CHANGES))\n")
from_base(j, s, tags=["營運性", "登記 9.2", "登記 9.4"])

# --------------------------------------------------------------------------- preflight (base, rep)
j = base_cell("c10-player02-02", "### 本版新增：preflight")
from_base(j, "### preflight、唯讀指紋、測試包與快速格式檢查\n\n"
             "開始推論前確認 prereg01 的權重與推論結果、prereg02 的 dumps 都在，而且 SHA-256 對得上（正式執行時還要和 repo "
             "記錄的相同）；解開測試包裡競賽一的四個檔並逐檔核對 SHA-256；做快速格式檢查；確認沒有上一次用不同輸入檔留下的 dump。"
             "同時記下 prereg01、prereg02 目錄底下每個檔的大小和 mtime（`readonly_before.json`），收尾格會再記一次比對。"
             "任何一項不符就用中文說明處理方式後停下，還沒開始推論。")
j = base_cell("c10-player02-03", "# preflight（本版新增）")
s = BASE_SRC[j]
s = rep(s, "# preflight（本版新增）：prereg01 的權重與推論結果都在、SHA-256 對得上，才開始推論；並記下 prereg01 目錄開頭的指紋\n",
        "# preflight（test02）：prereg01 的權重與推論結果、prereg02 的 dumps、測試包都對得上，快速格式檢查也過，才開始推論；\n"
        "# 並記下 prereg01、prereg02 目錄開頭的指紋\n"
        "import csv\nimport io\nfrom collections import Counter\n\n")
s = rep(s, """def p01_fingerprint():
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
    raise SystemExit('preflight 沒過，還沒開始推論：\\n' + msg)
""", """def ro_fingerprint(root):
    # 唯讀目錄底下每個檔的相對路徑、大小、mtime（只看 stat，不讀內容）
    files = {}
    for _root, _dirs, _files in os.walk(root):
        _dirs.sort()
        for _f in sorted(_files):
            _p = os.path.join(_root, _f)
            _st = os.stat(_p)
            files[os.path.relpath(_p, root).replace(os.sep, '/')] = {'size': _st.st_size,
                                                                      'mtime_ns': _st.st_mtime_ns}
    return {'root': root, 'taken_utc': now_utc(), 'n_files': len(files), 'files': files}


def readonly_fingerprints():
    # test02：prereg01、prereg02 兩個唯讀目錄的指紋
    return {'prereg01': ro_fingerprint(P01_ROOT), 'prereg02': ro_fingerprint(P02_ROOT)}


def set_status(status, **kw):
    # test02：run_log.json 最上層也記這次執行的狀態（本機工具讀它）；每筆 run 的 status 照舊
    RUN_LOG['status'] = status
    update_run(status=status, **kw)


def preflight_stop(msg):
    set_status('preflight_failed', preflight_error=msg)
    raise SystemExit('preflight 沒過，還沒開始推論：\\n' + msg)


set_status('running')
""")
s = rep(s, """_ACCOUNT_HINT = ('處理方式：確認這個 Colab 用的是跑 prereg01 的同一個 Google 帳號（prereg01 的權重和結果在那個帳號的 '
                 'Drive），而且 MyDrive/C10/results/prereg01/ 沒有被改名、搬動或重跑過。不要自己改檔；'
                 '還是不對就截圖給 Claude。')
if not os.path.isdir(P01_ROOT):
    preflight_stop('找不到 prereg01 的結果目錄 %s。\\n%s' % (P01_ROOT, _ACCOUNT_HINT))
P01_BEFORE = p01_fingerprint()
save_json('p01_readonly_before.json', P01_BEFORE)
print('prereg01 目錄指紋（開頭）：%d 個檔' % P01_BEFORE['n_files'])
""", """_ACCOUNT_HINT = ('處理方式：確認這個 Colab 用的是跑 prereg01、prereg02 的同一個 Google 帳號（權重和結果在那個帳號的 '
                 'Drive），而且 MyDrive/C10/results/prereg01/、results/prereg02/ 沒有被改名、搬動或重跑過。不要自己改檔；'
                 '還是不對就截圖給 Claude。')
for _ro, _label in ((P01_ROOT, 'prereg01'), (P02_ROOT, 'prereg02')):
    if not os.path.isdir(_ro):
        preflight_stop('找不到 %s 的結果目錄 %s。\\n%s' % (_label, _ro, _ACCOUNT_HINT))
READONLY_BEFORE = readonly_fingerprints()
save_json('readonly_before.json', READONLY_BEFORE)
print('唯讀目錄指紋（開頭）：prereg01 %d 個檔、prereg02 %d 個檔'
      % (READONLY_BEFORE['prereg01']['n_files'], READONLY_BEFORE['prereg02']['n_files']))
""")
TEST02_PREFLIGHT = r"""# ---- test02：prereg02 的 dumps（接縫的參照）與權重
WEIGHTS_SHA256_PIN = '@@WEIGHTS_SHA256@@'   # prereg02 frozen_config.json 記的權重（= prereg01 的 best_model.pth）
P02_DUMP_SHA256 = @@P02_DUMP_SHA256@@
P02_DUMPS = {k: os.path.join(P02_RESULT_DIR, 'dumps', k) for k in ('cand_games.npz', 'query_games.npz')}
_t02 = []
if not SMOKE and P01_SHA256['best_model.pth'] != WEIGHTS_SHA256_PIN:
    _t02.append('best_model.pth 的 SHA-256 %s 不是 prereg02 用的 %s' % (P01_SHA256['best_model.pth'], WEIGHTS_SHA256_PIN))
P02_SHA256 = {}
for _k, _p in P02_DUMPS.items():
    if not os.path.exists(_p):
        _t02.append('找不到 prereg02 的 dumps/%s（在 %s）' % (_k, P02_RESULT_DIR))
        continue
    P02_SHA256[_k] = sha256_file(_p)
    if not SMOKE and P02_SHA256[_k] != P02_DUMP_SHA256[_k]:
        _t02.append('prereg02 的 dumps/%s 的 SHA-256 %s 和 repo 記錄的 %s 不同' % (_k, P02_SHA256[_k], P02_DUMP_SHA256[_k]))
    with np.load(_p) as _z:
        _w = str(_z['weights_sha256']) if 'weights_sha256' in _z.files else None
    if _w != P01_SHA256['best_model.pth']:
        _t02.append('prereg02 的 dumps/%s 記的權重 %s 和 best_model.pth 的 %s 不同' % (_k, _w, P01_SHA256['best_model.pth']))
P02_ENV = None
if os.path.exists(os.path.join(P02_RESULT_DIR, 'env.json')):
    with open(os.path.join(P02_RESULT_DIR, 'env.json'), encoding='utf-8') as f:
        P02_ENV = json.load(f)
if _t02:
    preflight_stop('\n'.join('  - ' + p for p in _t02) + '\n' + _ACCOUNT_HINT)

# ---- test02：測試包 C10_test_v1.zip（成員、MANIFEST、SHA-256）；只解開競賽一的四個檔
TEST_NAMES = @@TEST_NAMES@@
PLAYER_NAMES = [n for _pool in ('public', 'private') for n in TEST_POOLS[_pool]]
TESTS_DIR = os.path.join(WORK, 'tests')
_ZIP_HINT = ('處理方式：不要自己重傳、解壓或改名。看一下 MyDrive/C10/ 裡是不是剛好一個 C10_test_v1.zip'
             '（不是 C10_test_v1 (1).zip，也不是解壓出來的資料夾），把這段訊息截圖給 Claude。')
if not os.path.exists(TEST_ZIP):
    preflight_stop('找不到測試包 %s。\n%s' % (TEST_ZIP, _ZIP_HINT))
TEST_ZIP_SHA256 = sha256_file(TEST_ZIP)
_zp = []
with zipfile.ZipFile(TEST_ZIP) as _z:
    _names = _z.namelist()
    _allowed = {'tests/MANIFEST.json'} | {'tests/' + n for n in TEST_NAMES}
    _odd = sorted(n for n in set(_names) if n not in _allowed)
    if _odd:
        _zp.append('測試包裡有不該有的成員：%s' % _odd[:5])
    if len(_names) != len(set(_names)):
        _zp.append('測試包裡有重複的成員')
    _missing = sorted(n for n in _allowed if n not in _names)
    if _missing:
        _zp.append('測試包缺少：%s' % _missing)
    TEST_MANIFEST_BYTES = _z.read('tests/MANIFEST.json') if 'tests/MANIFEST.json' in _names else b''
if _zp:
    preflight_stop('\n'.join('  - ' + p for p in _zp) + '\n' + _ZIP_HINT)
try:
    TEST_MANIFEST = json.loads(TEST_MANIFEST_BYTES.decode('utf-8'))
except ValueError as e:
    preflight_stop('測試包的 MANIFEST.json 讀不懂（%s）。\n%s' % (e, _ZIP_HINT))
_kinds = ('smoke',) if SMOKE else ('official', 'rehearsal')
if TEST_MANIFEST.get('format') != 'c10-test02 manifest v1':
    _zp.append('MANIFEST 的 format 是 %r，不是 c10-test02 manifest v1' % TEST_MANIFEST.get('format'))
if TEST_MANIFEST.get('kind') not in _kinds:
    _zp.append('MANIFEST 的 kind 是 %r（這次只接受 %s）' % (TEST_MANIFEST.get('kind'), '、'.join(_kinds)))
if (TEST_MANIFEST.get('kind') == 'rehearsal') != REHEARSAL:
    _zp.append('MANIFEST 的 kind %r 和設定格判斷的不一致（REHEARSAL=%r）' % (TEST_MANIFEST.get('kind'), REHEARSAL))
_mfiles = TEST_MANIFEST.get('files') or {}
if sorted(_mfiles) != sorted(TEST_NAMES):
    _zp.append('MANIFEST 記的檔案 %s 不是官方的六個檔名' % sorted(_mfiles))
# 登記第 9 節第 6 條：FAIL_OPENING（T1.4／T1.5 不過）時競賽一照常每池程序；FAIL_STRUCT 或沒有結果就停
if (TEST_MANIFEST.get('t1') or {}).get('player') not in ('PASS', 'NORMALIZED', 'FAIL_OPENING'):
    _zp.append('MANIFEST 記的競賽一 T1 結果是 %r（要是 PASS、NORMALIZED 或 FAIL_OPENING）' % (TEST_MANIFEST.get('t1') or {}).get('player'))
if _zp:
    preflight_stop('\n'.join('  - ' + p for p in _zp) + '\n' + _ZIP_HINT)
# 競賽二的兩個檔只核對檔名在不在（上面），不解開、不讀內容
if os.path.isdir(TESTS_DIR):
    shutil.rmtree(guard_write(TESTS_DIR))
with zipfile.ZipFile(TEST_ZIP) as _z:
    for _n in ['tests/MANIFEST.json'] + ['tests/' + n for n in PLAYER_NAMES]:
        _z.extract(_n, guard_write(WORK))
TEST_SHA256 = {}
for _n in PLAYER_NAMES:
    _p = os.path.join(TESTS_DIR, _n)
    TEST_SHA256[_n] = sha256_file(_p)
    if TEST_SHA256[_n] != (_mfiles.get(_n) or {}).get('sha256'):
        _zp.append('%s 的 SHA-256 %s 和 MANIFEST 記的 %s 不同' % (_n, TEST_SHA256[_n], (_mfiles.get(_n) or {}).get('sha256')))
    if 'bytes' in (_mfiles.get(_n) or {}) and os.path.getsize(_p) != _mfiles[_n]['bytes']:
        _zp.append('%s 有 %d bytes，MANIFEST 記 %s' % (_n, os.path.getsize(_p), _mfiles[_n]['bytes']))
with open(os.path.join(TESTS_DIR, 'MANIFEST.json'), 'rb') as f:
    _mb = f.read()
TEST_MANIFEST_SHA256 = hashlib.sha256(_mb).hexdigest()
if _mb != TEST_MANIFEST_BYTES:
    _zp.append('解開後的 MANIFEST.json 和 zip 裡的不同')
if _zp:
    preflight_stop('\n'.join('  - ' + p for p in _zp) + '\n' + _ZIP_HINT)

# ---- test02：快速格式檢查（T1 的一部分）
csv.field_size_limit(10 ** 9)
QCOLS = ['question_id', 'num_games'] + ['sgf_%d' % i for i in range(1, 21)] + ['color_%d' % i for i in range(1, 21)]
CCOLS = ['game_id', 'player_id', 'sgf_content', 'color']
PREFIXES = {'public': ('pub_', 'pub_p_', 'pub_'), 'private': ('priv_', 'priv_p_', 'priv_')}   # 題目、玩家、棋


def read_test_csv(path):
    with open(path, 'rb') as f:
        raw = f.read()
    rd = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    rows = list(rd)
    return raw.startswith(b'\xef\xbb\xbf'), list(rd.fieldnames or []), rows


def quick_questions(name, prefix, problems):
    bom, cols, rows = read_test_csv(os.path.join(TESTS_DIR, name))
    if bom:
        problems.append('%s：開頭有 BOM' % name)
    miss = [c for c in QCOLS if c not in cols]
    if miss:
        problems.append('%s：缺欄位 %s' % (name, miss[:5]))
        return None
    if len(rows) != 400:
        problems.append('%s：%d 題，不是 400 題' % (name, len(rows)))
    ids = [r['question_id'] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append('%s：question_id 有重複' % name)
    _bad = [q for q in ids if not q.startswith(prefix)]
    if _bad:
        problems.append('%s：%d 題的 question_id 不是 %s 開頭，例如 %s' % (name, len(_bad), prefix, _bad[:3]))
    bad, num_games = [], []
    for r in rows:
        try:
            n = int(r['num_games'])
        except (TypeError, ValueError):
            bad.append('%s 的 num_games=%r' % (r['question_id'], r['num_games']))
            continue
        if not 5 <= n <= 20:
            bad.append('%s 的 num_games=%d' % (r['question_id'], n))
            continue
        filled = [i for i in range(1, 21) if (r['sgf_%d' % i] or '').strip()]
        if filled != list(range(1, n + 1)):
            bad.append('%s 有棋的是第 %s 格，num_games=%d' % (r['question_id'], filled, n))
        if any(r['color_%d' % i] not in ('B', 'W') for i in range(1, n + 1)):
            bad.append('%s 的執色不是 B／W' % r['question_id'])
        num_games.append(n)
    if bad:
        problems.append('%s：%d 題格式不對，例如 %s' % (name, len(bad), bad[:3]))
    return {'rows': len(rows), 'num_games': num_games}


def quick_candidates(name, pprefix, gprefix, problems):
    bom, cols, rows = read_test_csv(os.path.join(TESTS_DIR, name))
    if bom:
        problems.append('%s：開頭有 BOM' % name)
    miss = [c for c in CCOLS if c not in cols]
    if miss:
        problems.append('%s：缺欄位 %s' % (name, miss))
        return None
    if len(rows) != 40000:
        problems.append('%s：%d 列，不是 40,000 列' % (name, len(rows)))
    per = Counter(r['player_id'] for r in rows)
    if len(per) != 400 or set(per.values()) != {100}:
        problems.append('%s：%d 位候選，每人盤數 %s（要 400 位 × 各剛好 100 盤；登記第 9 節第 4 條：不是的話停下）'
                        % (name, len(per), sorted(set(per.values()))[:6]))
    _bad = [p for p in per if not p.startswith(pprefix)]
    if _bad:
        problems.append('%s：%d 位候選的 player_id 不是 %s 開頭，例如 %s' % (name, len(_bad), pprefix, _bad[:3]))
    gids = [r['game_id'] for r in rows]
    if len(set(gids)) != len(gids):
        problems.append('%s：game_id 有重複' % name)
    _bad = [g for g in gids if not g.startswith(gprefix)]
    if _bad:
        problems.append('%s：%d 盤的 game_id 不是 %s 開頭，例如 %s' % (name, len(_bad), gprefix, _bad[:3]))
    _bad = [r['game_id'] for r in rows if r['color'] not in ('B', 'W')]
    if _bad:
        problems.append('%s：%d 盤的執色不是 B／W，例如 %s' % (name, len(_bad), _bad[:3]))
    return {'rows': len(rows), 'players': set(per), 'games': set(gids), 'per_player': per}


_qp = []
QUICK = {}
for _pool, (_qn, _cn) in TEST_POOLS.items():
    QUICK[_pool] = {'q': quick_questions(_qn, PREFIXES[_pool][0], _qp),
                    'c': quick_candidates(_cn, PREFIXES[_pool][1], PREFIXES[_pool][2], _qp)}
if QUICK['public']['c'] and QUICK['private']['c']:
    if QUICK['public']['c']['players'] & QUICK['private']['c']['players']:
        _qp.append('public 和 private 的候選有共用的 player_id')
    if QUICK['public']['c']['games'] & QUICK['private']['c']['games']:
        _qp.append('public 和 private 的候選有共用的 game_id')
if _qp:
    preflight_stop('測試檔的快速格式檢查沒過（這時還沒開始推論）：\n' + '\n'.join('  - ' + p for p in _qp)
                   + '\n處理方式：不要自己改檔，截圖給 Claude。')


def smoke_counts(qinfo, cinfo):
    # SMOKE 的縮小方式和 dump 格相同：候選照代號排序的前 SMOKE_N_CAND_PLAYERS 位、每人前 SMOKE_CAND_GAMES 盤；
    # 題目檔前 SMOKE_N_QUESTIONS 題、每題前 SMOKE_Q_GAMES 盤
    keep = sorted(cinfo['per_player'])[:SMOKE_N_CAND_PLAYERS]
    return (sum(min(cinfo['per_player'][p], SMOKE_CAND_GAMES) for p in keep),
            sum(min(n, SMOKE_Q_GAMES) for n in qinfo['num_games'][:SMOKE_N_QUESTIONS]),
            min(qinfo['rows'], SMOKE_N_QUESTIONS))


# 每個池的 (候選盤數, 題目盤數, 題數)；dump 格照這個核對。seam_val 沿用 prereg02（= prereg01 推論時的數字）
EXPECTED_COUNTS = {'seam_val': (P01_SUMMARY['n_candidate_rows'], P01_SUMMARY['n_query_games'], P01_SUMMARY['n_questions'])}
for _pool in ('public', 'private'):
    _qi, _ci = QUICK[_pool]['q'], QUICK[_pool]['c']
    EXPECTED_COUNTS[_pool] = (smoke_counts(_qi, _ci) if SMOKE else
                              (_ci['rows'], sum(_qi['num_games']), _qi['rows']))
POOL_ORDER = ('public', 'private', 'seam_val')      # seam_val 最後：dump 格開頭會刪掉上一次的重現核對結果
POOL_FILES = {'public': tuple(os.path.join(TESTS_DIR, n) for n in TEST_POOLS['public']),
              'private': tuple(os.path.join(TESTS_DIR, n) for n in TEST_POOLS['private']),
              'seam_val': (QUESTIONS_CSV, CANDIDATES_CSV)}
DUMP_INFO_BY_POOL = {}


# ---- test02：stale-dump guard（上一次用不同輸入檔算的 dump 還在就停下）
def input_record(pool):
    _q, _c = POOL_FILES[pool]
    return {'set': pool, 'question_file': os.path.basename(_q), 'sha256': sha256_file(_q),
            'candidates_file': os.path.basename(_c), 'candidates_sha256': sha256_file(_c)}


INPUT_RECORDS = {_pool: input_record(_pool) for _pool in POOL_ORDER}
_stale = []
for _pool, _rec in INPUT_RECORDS.items():
    _p = os.path.join(RESULT_DIR, _pool, 'dumps', '_INPUT.json')
    if os.path.exists(_p):
        with open(_p, encoding='utf-8') as f:
            _old = json.load(f)
        if _old != _rec:
            _stale.append('%s：上一次的題目檔 %s…、候選檔 %s…；這一次 %s…、%s…'
                          % (_pool, str(_old.get('sha256'))[:12], str(_old.get('candidates_sha256'))[:12],
                             _rec['sha256'][:12], _rec['candidates_sha256'][:12]))
if _stale:
    preflight_stop('%s 裡有上一次用不同輸入檔算的 dump：\n%s\n處理方式：不要自己刪檔，截圖給 Claude。'
                   % (RESULT_DIR, '\n'.join('  - ' + x for x in _stale)))


def write_input_json(pool):
    # dump 格呼叫：記下這個池用的題目檔與候選檔（下一次執行的 stale-dump guard 用）
    _p = guard_write(os.path.join(RESULT_DIR, pool, 'dumps', '_INPUT.json'))
    with open(_p + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(INPUT_RECORDS[pool], f, ensure_ascii=False, indent=1)
    os.replace(_p + '.tmp', _p)


"""
s = rep(s, "CONFIG['p01_inputs'] = {\n", TEST02_PREFLIGHT.replace("@@WEIGHTS_SHA256@@", WEIGHTS_SHA256)
        .replace("@@P02_DUMP_SHA256@@", repr(P02_DUMP_SHA256)).replace("@@TEST_NAMES@@", repr(TEST_NAMES))
        + "CONFIG['p01_inputs'] = {\n")
s = rep(s, """save_json('config.json', CONFIG)
update_run(p01_sha256=P01_SHA256, drive_free_gb=_free_gb)
""", """CONFIG['weights'] = {'path': os.path.relpath(P01_FILES['best_model.pth'], os.path.join(DRIVE_DIR, 'results')).replace(os.sep, '/'),
                     'sha256': P01_SHA256['best_model.pth']}
CONFIG['p02_inputs'] = {
    'root': P02_ROOT,
    'dumps': {k: {'path': p, 'sha256': P02_SHA256[k]} for k, p in P02_DUMPS.items()},
    'checked_against_repo': not SMOKE,
    'p02_gpu': P02_ENV.get('gpu') if P02_ENV else None,
    'p02_torch': P02_ENV.get('torch') if P02_ENV else None,
    'p02_cudnn': P02_ENV.get('cudnn') if P02_ENV else None,
}
CONFIG['test_bundle'] = {
    'zip': TEST_ZIP, 'zip_sha256': TEST_ZIP_SHA256, 'manifest_sha256': TEST_MANIFEST_SHA256,
    'kind': TEST_MANIFEST.get('kind'), 'created': TEST_MANIFEST.get('created'), 't1': TEST_MANIFEST.get('t1'),
    'files': TEST_SHA256,
    'names_checked_only': [n for n in TEST_NAMES if n not in PLAYER_NAMES],
}
_seam_rows = len(pd.read_csv(QUESTIONS_CSV, usecols=['question_id']))
CONFIG['sets'] = {_pool: {'questions_file': _rec['question_file'], 'questions_sha256': _rec['sha256'],
                          'candidates_file': _rec['candidates_file'], 'candidates_sha256': _rec['candidates_sha256'],
                          'rows': (QUICK[_pool]['q']['rows'] if _pool in QUICK else _seam_rows),
                          'expected_dump_counts': dict(zip(('candidate_games', 'query_games', 'questions'),
                                                           EXPECTED_COUNTS[_pool]))}
                  for _pool, _rec in INPUT_RECORDS.items()}
save_json('config.json', CONFIG)
update_run(p01_sha256=P01_SHA256, p02_dump_sha256=P02_SHA256, test_zip_sha256=TEST_ZIP_SHA256, drive_free_gb=_free_gb)
""")
s = rep(s, """for _k in P01_FILES:
    print('  %s  %s' % (P01_SHA256[_k][:16], _k))
""", """for _k in P01_FILES:
    print('  %s  %s' % (P01_SHA256[_k][:16], _k))
for _k in P02_DUMPS:
    print('  %s  prereg02 dumps/%s' % (P02_SHA256[_k][:16], _k))
print('測試包 %s（kind %s）：競賽一四個檔 SHA-256 和 MANIFEST 相同，快速格式檢查通過' % (TEST_ZIP_SHA256[:16], TEST_MANIFEST.get('kind')))
for _pool in POOL_ORDER:
    print('  %-8s 候選 %d 盤、題目 %d 盤、%d 題（dump 格照這個核對）' % ((_pool,) + tuple(EXPECTED_COUNTS[_pool])))
""")
from_base(j, s, tags=["營運性", "登記 9.2", "登記 9.4"])

# --------------------------------------------------------------------------- official 20: identical
OFFICIAL20_BASE = base_cell("bf5fcdb7", "GPU_BATCH = 4096")
from_base(OFFICIAL20_BASE, BASE_SRC[OFFICIAL20_BASE])

# --------------------------------------------------------------------------- per-game embeddings, three pools (base, rep)
j = base_cell("c10-player02-04", "### 本版新增：每盤 embedding")
from_base(j, "### 每盤 embedding：public、private、seam_val 各一格（登記第 9 節第 4 條）\n\n"
             "三格的程式完全相同，只差第一行的 `POOL`。每盤的算法和 prereg02 相同（`sgf_windows` → 一盤一次 `embed` → "
             "窗口 embedding 在 float32 下加總；`game_embeddings()` 和候選、題目兩段迴圈逐位元組相同）。輸出寫到 "
             "`results/test02/player/<POOL>/dumps/`，另寫 `_INPUT.json`。盤數和題數照 preflight 算好的數字核對。"
             "seam_val（本機驗證池）放最後：這一格開頭會刪掉上一次的重現核對結果，重現核對格接著用 seam_val 的結果。")
DUMP_BASE = base_cell("c10-player02-05", "# 每盤 embedding（本版新增，登記 2.1）")
s = BASE_SRC[DUMP_BASE]
s = rep(s, "# 每盤 embedding（本版新增，登記 2.1）：候選 40,000 盤 → dumps/cand_games.npz；題目 4,935 盤 → dumps/query_games.npz\n",
        "# 每盤 embedding（test02，登記第 9 節第 4 條）：POOL 這個池的候選 → <POOL>/dumps/cand_games.npz；"
        "題目 → <POOL>/dumps/query_games.npz\n")
s = rep(s, "DUMP_DIR = os.path.join(RESULT_DIR, 'dumps')\n", "DUMP_DIR = os.path.join(RESULT_DIR, POOL, 'dumps')\n")
s = rep(s, """for _p in (CAND_NPZ, QUERY_NPZ, CAND_NPZ + '.tmp', QUERY_NPZ + '.tmp', REPRO_JSON, REPRO_CSV,
           os.path.join(RESULT_DIR, 'p01_readonly_after.json')):
    if os.path.exists(_p):
        os.remove(guard_write(_p))
os.makedirs(guard_write(DUMP_DIR), exist_ok=True)
""", """for _p in (CAND_NPZ, QUERY_NPZ, CAND_NPZ + '.tmp', QUERY_NPZ + '.tmp', REPRO_JSON, REPRO_CSV,
           os.path.join(RESULT_DIR, 'readonly_after.json'), os.path.join(RESULT_DIR, 'seam_p02.json'),
           os.path.join(DUMP_DIR, '_INPUT.json')):
    if os.path.exists(_p):
        os.remove(guard_write(_p))
os.makedirs(guard_write(DUMP_DIR), exist_ok=True)
write_input_json(POOL)          # test02：這個池用的題目檔與候選檔（檔名與 SHA-256）
DUMP_INFO_BY_POOL.pop(POOL, None)
""")
s = rep(s, """N_CAND_ROWS, N_QUERY_ROWS = len(CAND), len(jobs)
# 盤數和題數必須和 prereg01 推論時相同（SMOKE 時和 prereg01 冒煙測試相同）
assert (N_CAND_ROWS, N_QUERY_ROWS, len(TEST)) == (P01_SUMMARY['n_candidate_rows'], P01_SUMMARY['n_query_games'],
                                                  P01_SUMMARY['n_questions']), \\
    ((N_CAND_ROWS, N_QUERY_ROWS, len(TEST)), P01_SUMMARY)
DUMP_INFO['seconds'] = time.time() - _t_dump
update_run(dumps=DUMP_INFO)
record_time('dump_total_s', DUMP_INFO['seconds'])
print('每盤 embedding 完成：候選 %d 盤（可解析 %d）、題目 %d 盤（可解析 %d），%.0f 秒'
      % (N_CAND_ROWS, DUMP_INFO['cand_games.npz']['n_parsed'], N_QUERY_ROWS,
         DUMP_INFO['query_games.npz']['n_parsed'], DUMP_INFO['seconds']))
""", """N_CAND_ROWS, N_QUERY_ROWS = len(CAND), len(jobs)
# 盤數和題數必須和 preflight 算好的相同（seam_val：和 prereg01 推論時相同；SMOKE 時照同樣的縮小方式）
assert (N_CAND_ROWS, N_QUERY_ROWS, len(TEST)) == tuple(EXPECTED_COUNTS[POOL]), \\
    (POOL, (N_CAND_ROWS, N_QUERY_ROWS, len(TEST)), EXPECTED_COUNTS[POOL])
DUMP_INFO['seconds'] = time.time() - _t_dump
DUMP_INFO_BY_POOL[POOL] = DUMP_INFO
update_run(dumps=DUMP_INFO_BY_POOL)
record_time(POOL + '_dump_total_s', DUMP_INFO['seconds'])
print('%s 每盤 embedding 完成：候選 %d 盤（可解析 %d）、題目 %d 盤（可解析 %d），%.0f 秒'
      % (POOL, N_CAND_ROWS, DUMP_INFO['cand_games.npz']['n_parsed'], N_QUERY_ROWS,
         DUMP_INFO['query_games.npz']['n_parsed'], DUMP_INFO['seconds']))
""")
DUMP_BODY = s
DUMP_PREAMBLE = "POOL = %r; QUESTIONS_CSV, CANDIDATES_CSV = POOL_FILES[POOL]  # test02：這一格處理的候選池\n"
for _pool in POOL_ORDER:
    from_base(DUMP_BASE, DUMP_PREAMBLE % _pool + DUMP_BODY, tags=["登記 9.4"], cid="c10-player02-05-" + _pool)

# --------------------------------------------------------------------------- reproduction check: identical (seam_val)
REPRO_MD = base_cell("c10-player02-06", "### 本版新增：重現核對")
from_base(REPRO_MD, BASE_SRC[REPRO_MD])
REPRO_BASE = base_cell("c10-player02-07", "# 重現核對（本版新增，登記 2.1）")
from_base(REPRO_BASE, BASE_SRC[REPRO_BASE])

# --------------------------------------------------------------------------- seam_p02 (new, record only)
md("### seam_p02：seam_val 的逐盤 embedding 和 prereg02 是否逐位元組相同（只記錄）\n\n"
   "把 seam_val 的 `cand_games.npz`、`query_games.npz` 和 prereg02 的 dumps 逐個陣列比，記下是否逐位元組相同、"
   "emb_sum 的最大差，以及兩次執行的 GPU、torch、cuDNN。這一格只記錄，不會停下；接縫的判定在本機"
   "（`tools/t02_player.py run --pool seam_val`，登記第 9 節第 2 條）。")
code(r"""# seam_p02（test02 新增，只記錄）：seam_val 的逐盤 embedding 和 prereg02 的 dumps 逐個陣列比
_t_seam = time.time()
SEAM_P02 = {'checked_utc': now_utc(), 'smoke': SMOKE, 'reference_dir': os.path.join(P02_RESULT_DIR, 'dumps'),
            'files': {}}
_all_equal, _max_diff = True, 0.0
for _name in ('cand_games.npz', 'query_games.npz'):
    _a = os.path.join(RESULT_DIR, 'seam_val', 'dumps', _name)
    _b = os.path.join(P02_RESULT_DIR, 'dumps', _name)
    with np.load(_a) as _za, np.load(_b) as _zb:
        _keys = sorted(set(_za.files) | set(_zb.files))
        _eq = {_k: bool(_k in _za.files and _k in _zb.files and _za[_k].dtype == _zb[_k].dtype
                        and _za[_k].shape == _zb[_k].shape and np.array_equal(_za[_k], _zb[_k])) for _k in _keys}
        _same_shape = _za['emb_sum'].shape == _zb['emb_sum'].shape
        _d = (float(np.max(np.abs(_za['emb_sum'].astype(np.float64) - _zb['emb_sum'].astype(np.float64))))
              if _same_shape and _za['emb_sum'].size else None)
    SEAM_P02['files'][_name] = {'arrays_equal': _eq, 'bitwise_equal': all(_eq.values()), 'max_abs_emb_diff': _d,
                                'sha256': sha256_file(_a), 'reference_sha256': sha256_file(_b)}
    _all_equal = _all_equal and all(_eq.values())
    _max_diff = None if (_max_diff is None or _d is None) else max(_max_diff, _d)
SEAM_P02.update({
    'bitwise_equal': _all_equal,
    'max_abs_emb_diff': _max_diff,
    'gpu': ENV['gpu'], 'torch': ENV['torch'], 'cudnn': ENV['cudnn'],
    'p02_gpu': P02_ENV.get('gpu') if P02_ENV else None,
    'p02_torch': P02_ENV.get('torch') if P02_ENV else None,
    'p02_cudnn': P02_ENV.get('cudnn') if P02_ENV else None,
    'rule': '只記錄；接縫的判定在本機 tools/t02_player.py（登記第 9 節第 2 條）',
})
save_json('seam_p02.json', SEAM_P02)
update_run(seam_p02_bitwise_equal=_all_equal, seam_p02_max_abs_emb_diff=_max_diff)
record_time('seam_p02_s', time.time() - _t_seam)
print('seam_p02：seam_val 的逐盤 embedding 和 prereg02 %s（emb_sum 最大差 %s）；GPU 這次 %s，prereg02 %s'
      % ('逐位元組相同' if _all_equal else '不完全相同', _max_diff, ENV['gpu'], SEAM_P02['p02_gpu']))
""", tags=["登記 9.2"])

# --------------------------------------------------------------------------- final (base, rep)
j = base_cell("c10-player-10", "# 收尾（營運性）")
s = BASE_SRC[j]
s = rep(s, """# 收尾（營運性）：核對輸出、比對 prereg01 前後指紋、記下這次執行；打包；Colab 上把 Drive 的寫入全部送出再卸載
# （prereg02：重現核對沒過、輸出不對或 prereg01 目錄有變動時，照樣打包 player_results.zip，最後停下）
FINAL_PROBLEMS = []
for _name, _path, _rows in (('cand_games.npz', CAND_NPZ, N_CAND_ROWS), ('query_games.npz', QUERY_NPZ, N_QUERY_ROWS)):
    if not os.path.exists(_path) or sha256_file(_path) != DUMP_INFO[_name]['sha256']:
        FINAL_PROBLEMS.append('dumps/%s 不在或 SHA-256 和寫出時不同' % _name)
        continue
    with np.load(_path) as _z:
        if _z['emb_sum'].shape != (_rows, EMBED_DIM) or any(len(_z[_k]) != _rows for _k in _z.files if _k != 'weights_sha256'):
            FINAL_PROBLEMS.append('dumps/%s 的列數不是 %d' % (_name, _rows))
""", """# 收尾（營運性）：核對三個池的輸出、比對 prereg01／prereg02 前後指紋、記下這次執行；打包；Colab 上把 Drive 的寫入全部送出再卸載
# （test02：重現核對沒過、輸出不對或唯讀目錄有變動時，照樣打包 test02_player_results.zip，最後停下）
FINAL_PROBLEMS = []
for _pool in POOL_ORDER:
    if _pool not in DUMP_INFO_BY_POOL:
        FINAL_PROBLEMS.append('%s 的每盤 embedding 格沒有跑完' % _pool)
        continue
    for _name, _k in (('cand_games.npz', 0), ('query_games.npz', 1)):
        _path = os.path.join(RESULT_DIR, _pool, 'dumps', _name)
        _rows = EXPECTED_COUNTS[_pool][_k]
        if not os.path.exists(_path) or sha256_file(_path) != DUMP_INFO_BY_POOL[_pool][_name]['sha256']:
            FINAL_PROBLEMS.append('%s/dumps/%s 不在或 SHA-256 和寫出時不同' % (_pool, _name))
            continue
        with np.load(_path) as _z:
            if _z['emb_sum'].shape != (_rows, EMBED_DIM) or any(len(_z[_kk]) != _rows for _kk in _z.files if _kk != 'weights_sha256'):
                FINAL_PROBLEMS.append('%s/dumps/%s 的列數不是 %d' % (_pool, _name, _rows))
    _ip = os.path.join(RESULT_DIR, _pool, 'dumps', '_INPUT.json')
    if not os.path.exists(_ip):
        FINAL_PROBLEMS.append('%s/dumps/_INPUT.json 不在' % _pool)
    else:
        with open(_ip, encoding='utf-8') as f:
            if json.load(f) != INPUT_RECORDS[_pool]:
                FINAL_PROBLEMS.append('%s/dumps/_INPUT.json 和這次的輸入檔不同' % _pool)
""")
s = rep(s, """P01_AFTER = p01_fingerprint()
save_json('p01_readonly_after.json', P01_AFTER)
_changed = sorted(k for k in set(P01_BEFORE['files']) | set(P01_AFTER['files'])
                  if P01_BEFORE['files'].get(k) != P01_AFTER['files'].get(k))
P01_UNCHANGED = not _changed
if not P01_UNCHANGED:
    FINAL_PROBLEMS.append('prereg01 目錄有 %d 個檔和開頭不同（例如 %s；見 p01_readonly_before/after.json）'
                          % (len(_changed), _changed[:3]))
""", """READONLY_AFTER = readonly_fingerprints()
save_json('readonly_after.json', READONLY_AFTER)
_changed = sorted('%s/%s' % (_r, _k) for _r in ('prereg01', 'prereg02')
                  for _k in set(READONLY_BEFORE[_r]['files']) | set(READONLY_AFTER[_r]['files'])
                  if READONLY_BEFORE[_r]['files'].get(_k) != READONLY_AFTER[_r]['files'].get(_k))
READONLY_UNCHANGED = not _changed
if not READONLY_UNCHANGED:
    FINAL_PROBLEMS.append('唯讀目錄有 %d 個檔和開頭不同（例如 %s；見 readonly_before/after.json）'
                          % (len(_changed), _changed[:3]))
""")
s = rep(s, """FINAL_STATUS = ('completed' if not FINAL_PROBLEMS else 'repro_failed' if not REPRO['passed'] else 'final_check_failed')
record_time('total_s', time.time() - RUN_T0)
update_run(status=FINAL_STATUS, finished_utc=now_utc(), seconds=time.time() - RUN_T0,
           p01_unchanged=P01_UNCHANGED, p01_changed_files=_changed, final_problems=FINAL_PROBLEMS)
""", """FINAL_STATUS = ('completed' if not FINAL_PROBLEMS else 'readonly_changed' if not READONLY_UNCHANGED
                else 'repro_failed' if not REPRO['passed'] else 'final_check_failed')
record_time('total_s', time.time() - RUN_T0)
set_status(FINAL_STATUS, finished_utc=now_utc(), seconds=time.time() - RUN_T0,
           readonly_unchanged=READONLY_UNCHANGED, readonly_changed_files=_changed, final_problems=FINAL_PROBLEMS)
""")
s = rep(s, "RESULTS_ZIP = os.path.join(os.path.dirname(RESULT_DIR), os.path.basename(RESULT_DIR) + '_results.zip')\n",
        "RESULTS_ZIP = os.path.join(RESULT_ROOT, 'test02_player_results.zip')      # test02：固定檔名；成員從 player/ 起算\n")
s = rep(s, "# 結果打包：結果目錄裡除了權重以外的檔案打成一個 zip，放在結果目錄旁邊，方便整包下載\n",
        "# 結果打包：結果目錄裡除了權重以外的檔案打成一個 zip，放在 results/test02（彩排、SMOKE 是各自的根目錄），方便整包下載\n")
s = rep(s, "print('請下載這個檔回傳給 Claude（融合與打分在本機做）：%s（%.1f MB）'\n",
        "print('請下載這個檔回傳給 Claude（每池的選擇、融合與交件檔在本機做）：%s（%.1f MB）'\n")
s = rep(s, "    raise SystemExit('這次執行沒有通過（狀態 %s），請下載 player_results.zip 並截圖給 Claude：\\n%s'\n",
        "    raise SystemExit('這次執行沒有通過（狀態 %s），請下載 test02_player_results.zip 並截圖給 Claude：\\n%s'\n")
from_base(j, s, tags=["營運性", "登記 9.2"])
FINAL_BASE = j

# --------------------------------------------------------------------------- notebook
_k = 0
nb_cells = []
for c, j, cid in cells:
    if cid is not None:
        c["id"] = cid
    elif j is not None:
        c["id"] = base["cells"][j]["id"]
    else:
        c["id"] = "c10-player-t02-%02d" % _k
        _k += 1
    nb_cells.append(c)
_cfg = [c for c in nb_cells if c.cell_type == "code" and "NB_SOURCE_SHA256 = '%s'" % NB_SHA_PLACEHOLDER in c.source]
assert len(_cfg) == 1
NB_SOURCE_SHA256 = notebook_source_sha256([(c.cell_type, c.source) for c in nb_cells])
_cfg[0].source = _cfg[0].source.replace("NB_SOURCE_SHA256 = '%s'" % NB_SHA_PLACEHOLDER,
                                        "NB_SOURCE_SHA256 = '%s'" % NB_SOURCE_SHA256)
nb = nbformat.v4.new_notebook()
nb.cells = nb_cells
nb.metadata = json.loads(json.dumps(base["metadata"]))
nb.metadata["colab"] = {"provenance": []}          # 不預設 A100／高 RAM：照第一格選 G4（拿不到時先等，不換 GPU）
nb.metadata["c10"] = {"notebook": OUT_REL, "base_notebook": BASE_REL, "base_notebook_sha256": BASE_SHA256,
                      "prereg": "docs/PREREG_02_OPENING_FUSION.md section 9 (rev 5)",
                      "spec": "results/test02/design/spec.md 2.0, 2.2, 2.3",
                      "notebook_source_sha256": NB_SOURCE_SHA256,
                      "changes_from_prereg02": T02_STRINGS}
nbformat.validate(nb)

# --------------------------------------------------------------------------- checks
assert len({c["id"] for c in nb.cells}) == len(nb.cells)
_used = [j for _c, j, _cid in cells if j is not None]
assert _used == sorted(_used), _used                                   # base order kept
assert [j for j in _used if _used.count(j) > 1] == [DUMP_BASE] * 3, _used   # only the dump cell is repeated
# the three per-pool dump cells: preamble + identical body
_dump_cells = [c for c, j, _cid in cells if j == DUMP_BASE]
assert len(_dump_cells) == 3
_bodies = [c.source.split("\n", 1) for c in _dump_cells]
assert [b[0] for b in _bodies] == [(DUMP_PREAMBLE % p).rstrip("\n") for p in POOL_ORDER], [b[0] for b in _bodies]
assert len({b[1] for b in _bodies}) == 1 and _bodies[0][1] == DUMP_BODY
# cells kept verbatim
for _j in (PREPARE_BASE, OFFICIAL1_BASE, ENV_BASE, OFFICIAL2_BASE, OFFICIAL20_BASE, REPRO_MD, REPRO_BASE):
    _hits = [c for c, j, _cid in cells if j == _j]
    assert len(_hits) == 1 and _hits[0].source == BASE_SRC[_j] and _hits[0].metadata["c10"]["change"] == "identical", _j


# segment identity (spec 2.0): extracted with ast.get_source_segment from the base cell and every new copy
def segments(src):
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("game_embeddings", "save_npz"):
            out[node.name] = ast.get_source_segment(src, node)
        if isinstance(node, ast.With) and ast.get_source_segment(src, node.items[0].context_expr) == "Pool(NPROC)":
            out["with Pool(NPROC) (CAND and TEST loops)"] = ast.get_source_segment(src, node)
        if isinstance(node, ast.With) and "open(P01_FILES['best_model.pth'], 'rb')" in ast.get_source_segment(src, node):
            out["weights read"] = ast.get_source_segment(src, node)
    return out


_base_seg = segments(BASE_SRC[DUMP_BASE])
assert set(_base_seg) == {"game_embeddings", "save_npz", "with Pool(NPROC) (CAND and TEST loops)", "weights read"}, \
    set(_base_seg)
for c in _dump_cells:
    assert segments(c.source) == _base_seg
SEGMENTS_CHECKED = sorted(_base_seg) + ["official cell 20 (whole cell)", "reproduction cell (whole cell)"]

# the label-file refusal: exactly the inherited prepare-cell line, nowhere else
FORBIDDEN = "ans" + "wers"          # spelled in two parts: no test02 source contains the word
_refusal = [ln for ln in BASE_SRC[PREPARE_BASE].splitlines() if FORBIDDEN in ln]
assert len(_refusal) == 1 and _refusal[0].startswith("assert not any("), _refusal
_hits = [(k, ln) for k, c in enumerate(nb.cells) for ln in c.source.splitlines() if FORBIDDEN in ln.lower()]
assert len(_hits) == 1 and _hits[0][1] == _refusal[0], _hits
for k, (c, j, _cid) in enumerate(cells):
    meta = c.metadata["c10"]
    if j is not None:
        assert c.cell_type == base["cells"][j]["cell_type"]
        assert (c.source == BASE_SRC[j]) == (meta["change"] == "identical"), k
    if c.cell_type == "code":
        compile(c.source, "<cell %d>" % k, "exec")         # plain Python only (run_local_smoke.py compiles each cell)
        assert "import Pool" not in c.source or j == OFFICIAL1_BASE, k   # smoke swaps Pool after each cell
# no training: none of these names is used in any code cell, and nothing calls torch.save
_names = set()
for c in nb.cells:
    if c.cell_type == "code":
        _toks = [t for t in tokenize.generate_tokens(io.StringIO(c.source).readline)
                 if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)]
        _names |= {t.string for t in _toks if t.type == tokenize.NAME}
        for _a, _b, _d in zip(_toks, _toks[1:], _toks[2:]):
            assert (_a.string, _b.string, _d.string) != ("torch", ".", "save"), "torch.save"
for _word in ("load_rows", "c10data", "TRAIN_DIR", "SPLIT_CSV", "FEAT_DIR", "INDEX_PATH", "extract_all", "tarfile",
              "train_loader", "TripletDataset", "optimizer", "train_step", "mean_embeddings", "predict",
              "decide", "p02_rank_fuse"):
    assert _word not in _names, _word
# competition 2's test files are never opened: they appear only in TEST_NAMES (name check)
for c in nb.cells:
    if c.cell_type == "code":
        for ln in c.source.splitlines():
            if "rank_prediction_test_" in ln and not ln.startswith("TEST_NAMES = ["):
                raise AssertionError("competition-2 test file named outside TEST_NAMES: " + ln)
_pre = [c for c, j, _cid in cells if c.cell_type == "code" and "TEST_NAMES = [" in c.source]
assert len(_pre) == 1

OUT_NB = out_path(OUT_REL)
OUT_DIFF = out_path(DIFF_REL)
os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
with open(OUT_NB, "w", encoding="utf-8", newline="\n") as f:
    nbformat.write(nb, f)
# read back: identical cells are still byte-identical to the base; the source SHA recomputes
with open(OUT_NB, encoding="utf-8") as f:
    _back = json.load(f)
for c in _back["cells"]:
    m = c["metadata"]["c10"]
    if m["change"] == "identical":
        assert "".join(c["source"]) == BASE_SRC[m["base_cell"]] and c["id"] == base["cells"][m["base_cell"]]["id"]
_srcs = [(c["cell_type"], "".join(c["source"])) for c in _back["cells"]]
_found = [NB_SHA_RE.findall(s) for _t, s in _srcs]
assert sum(len(x) for x in _found) == 1
assert notebook_source_sha256([(t, NB_SHA_RE.sub("NB_SOURCE_SHA256 = '%s'" % NB_SHA_PLACEHOLDER, s))
                               for t, s in _srcs]) == NB_SOURCE_SHA256
with open(OUT_NB, "rb") as f:
    assert f.read().lower().count(FORBIDDEN.encode()) == 1

# --------------------------------------------------------------------------- per-cell diff vs prereg02_player.ipynb
out = []
out.append("# test02 player：colab/prereg02_player.ipynb（底稿）與 colab/test02_player.ipynb 的逐格 unified diff"
           "（程式格、markdown 格、notebook metadata）")
out.append("# 底稿：%s  sha256=%s" % (BASE_REL, BASE_SHA256))
out.append("# 本版：%s  sha256=%s  notebook_source_sha256=%s" % (OUT_REL, file_sha256(OUT_NB), NB_SOURCE_SHA256))
out.append("# 產生方式：colab/build_test02_player_nb.py；difflib.unified_diff，依 cell metadata c10.base_cell 配對底稿的格；"
           "新增的格對空白比。cell 編號從 0 起算。")
out.append("# 每一處差異的說明見 notebook 第 0 格「與 prereg02 notebook 的差異」（共 %d 項）；底稿沒有列在對照表的格都拿掉了。"
           % len(T02_CHANGES))
out.append("# 逐段核對為相同（ast.get_source_segment）：%s" % "；".join(SEGMENTS_CHECKED))
summary = []
body = []
for k, c in enumerate(nb.cells):
    meta = c.metadata["c10"]
    kind = "程式格" if c.cell_type == "code" else "markdown"
    j = meta["base_cell"]
    if j is not None:
        old_src = BASE_SRC[j]
        a = "prereg02_player.ipynb cell %d（id %s）" % (j, base["cells"][j]["id"])
        status = "完全相同" if meta["change"] == "identical" else "改動"
    else:
        old_src = ""
        a = "/dev/null"
        status = "新增"
    b = "test02_player.ipynb cell %d（id %s）" % (k, c["id"])
    tags = "、".join(meta.get("tags") or []) if c.cell_type == "code" else ""
    summary.append("#   本版 cell %2d（%s）← %s：%s%s" % (k, kind, a, status, "；分類：%s" % tags if tags else ""))
    diff = list(difflib.unified_diff(old_src.splitlines(), c.source.splitlines(), fromfile=a, tofile=b, lineterm=""))
    body.append("")
    body.append("=" * 100)
    body.append("本版 cell %d（%s）↔ %s（%s）" % (k, kind, a, status))
    body.append("=" * 100)
    body.extend(diff if diff else ["（完全相同）"])
dropped = ["#   底稿 cell %2d（%s，id %s）：%s" % (
    j, "程式格" if base["cells"][j]["cell_type"] == "code" else "markdown", base["cells"][j]["id"],
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
out.extend(list(difflib.unified_diff(_meta_old, _meta_new, fromfile="prereg02_player.ipynb metadata",
                                     tofile="test02_player.ipynb metadata", lineterm="")) or ["（完全相同）"])
os.makedirs(os.path.dirname(OUT_DIFF), exist_ok=True)
with open(OUT_DIFF, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print("wrote", OUT_NB, file_sha256(OUT_NB))
print("wrote", OUT_DIFF, file_sha256(OUT_DIFF))
print("cells", len(nb.cells), "| from base", len(_used), "| new", len(nb.cells) - len(_used),
      "| base cells dropped", len(dropped), "| changes", len(T02_CHANGES), "| source sha256", NB_SOURCE_SHA256)
