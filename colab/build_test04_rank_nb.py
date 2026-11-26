"""Build colab/test04_rank.ipynb from two committed notebooks and write results/test04/notebook_diff_rank.txt
(per-cell unified diff of every cell against its own base cell, plus the notebook metadata).

Spec: results/test04/design/spec.md sections 2.2, 2.3 and 4; registration: docs/PREREG_04_OPPONENT.md rev 3
section 11 item 4 (the registration wins on any scientific question).

Standard library only. It imports no colab/build_* generator (they write files when imported) and nothing from
tools/p0* (tools/p04_rank_fuse.py is read as text and parsed with ast, never imported).

How the notebook is assembled (the discipline of colab/build_test02_rank_nb.py and build_prereg03_rank_nb.py):
  * two bases, both read from the committed notebook JSON after their SHA-256 is asserted:
      T02 = colab/test02_rank.ipynb (the test-day skeleton; in the R20 fingerprint, read only),
      P03 = colab/prereg03_rank_a.ipynb (model and dump code; colab/prereg03_rank_b.ipynb is asserted to differ
            from it in the ARM line only, so taking A is the same as taking B);
  * base cells are picked by their `c10` cell metadata (cell_key), never by position;
  * a reused cell is copied verbatim or changed only through rep() (target exactly once) and cut() (two markers
    that each occur exactly once); every changed base cell is named in CHANGE_ROWS (the table in cell 0 and
    `changes_from_bases` in config.json), and every named cell really changed;
  * segment identity (ast.get_source_segment; the build stops on any difference): the view_d4 cell equals P03's;
    load_net, dump_val_p03, channels_last, _nchw_output, state_dict_sha256 equal P03's; write_dump, dump_is_done,
    write_keys, require_gpu_for, pack_results, flush_drive, sgf_to_features, parse_questions,
    quick_check_questions, question_spans, c0_of, _argmax_first, rule_b, readonly_fingerprint, _is_under (and the
    other kept helpers) equal T02's; guard_write and record_readonly_after equal T02's up to comments (token
    stream without comments); the prepare cell equals T02's; the three set cells differ only in their first line;
  * pins are read from the repository at build time and asserted against the constants below (spec 4.1 table);
  * every code cell compiles; the inherited prepare-cell refusal line is the only line of the notebook that holds
    the validation label file token;
  * config.json records notebook_sha256 = sha256 of json.dumps([cell sources], ensure_ascii=False) with that
    value replaced by 64 zeros (the rule of test02; tools/t02_rank.py notebook_source_sha256 recomputes it).
The notebook JSON is written in nbformat v4.5 layout (json.dumps(sort_keys=True, indent=1, ensure_ascii=False),
sources split into lines), byte-identical to what nbformat.write produces. Re-running the builder gives
byte-identical outputs.

    python colab/build_test04_rank_nb.py [REPO] [--out-nb PATH] [--out-diff PATH] [--upload-copy]

--upload-copy also writes the byte copy data/colab_upload/test04/test04_rank.ipynb (not in git; the folder must
hold only this file).
"""

import argparse
import ast
import builtins
import difflib
import hashlib
import io
import json
import os
import struct
import sys
import tokenize
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

T02_REL = "colab/test02_rank.ipynb"
T02_SHA256 = "53337a7acc396bdc1f94b3cbdc7dd0ce859766511099e8004697ff0d6a487b05"
P03_REL = "colab/prereg03_rank_a.ipynb"
P03_SHA256 = "85e0c2ae3269c68ad166310efb2525b2d48ed1279c9cf6eb6b428e59e8693557"
P03B_REL = "colab/prereg03_rank_b.ipynb"
P03B_SHA256 = "aa3669accceca23fc3ab2439bcab68204281bcec999cd1e00d43dd16825bd8b3"
OUT_REL = "colab/test04_rank.ipynb"
DIFF_REL = "results/test04/notebook_diff_rank.txt"
UPLOAD_REL = "data/colab_upload/test04/test04_rank.ipynb"

# The validation label file token is spelled in two parts so that a repo-wide search for it finds only code
# that refuses such files.
FORBIDDEN_TOKEN = "answ" + "ers"
HASH_PLACEHOLDER = "0" * 64
HASH_LINE = "NOTEBOOK_SHA256 = '%s'"
ARMS = ("a", "b")
DUMP_NAMES = {"a": "p03a", "b": "p03b"}
SETS = ("seam_val", "public", "private")

# ---------------------------------------------------------------------------- spec 4.1 pins
# each is re-derived from the repository in derive_pins() and must agree
SPEC_W_PIN = {"a": "790b767c5425e262130d6c103c71e89238edb17f927fa48e2a2770db788e50ac",
              "b": "8e88f6ec6cbaad5372e5d62607b99ece3d77dc7a20af86a9197553f1a6c59a42"}
SPEC_W_LABEL = {x: "prereg03/rank_%s/ckpt/final.pth" % x for x in ARMS}
SPEC_SEAM_REF = {
    "a": {"val_keys.npz": "055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c",
          "val_p03a_d4.npz": "1e4e5957442596358b087e5f4e69c1fcf9f22f2c7933cc2049066bfab3075ecd",
          "val_p03a_s0_swap.npz": "b355c26bf68069feb0e48ee95c46bd211c2bdbc6b5a48416c8408a210aec3848"},
    "b": {"val_keys.npz": "055d29416bb79050128eea1e6944eb58376a751dc86780eee927bb94aca7ec8c",
          "val_p03b_d4.npz": "bad2277dd20bf832c02213be781768ba9d8a304922693bee383a28aa3531a627",
          "val_p03b_s0_swap.npz": "0cadff4bae36b2a41f02fddf4366a2081d9912ede4a3063c8b50fd28a2647527"}}
SPEC_REF_ENV = {"gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "torch": "2.11.0+cu128", "cudnn": 91900}
SPEC_TF32 = {"cudnn": True, "matmul": False}
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
SPEC_SEAM_VAL_QUESTIONS_SHA256 = "73e51bbfd7b051e73cbb62af4b180c403d2422f122240c5e67e9c92303776db5"

# segment identity lists (spec 4.1)
P03_DUMP_TOOLS_SEGMENTS = ("load_net", "dump_val_p03")
P03_MODEL_SEGMENTS = ("state_dict_sha256", "_nchw_output", "channels_last")
T02_SETUP_SEGMENTS = ("_real", "_is_under", "readonly_fingerprint", "sha256_path", "gpu_name_of", "write_json",
                      "read_json", "read_runs")
T02_SETUP_CODE_SEGMENTS = ("guard_write",)            # comments may differ, code identical
T02_DUMP_TOOLS_SEGMENTS = ("_dump_path", "_marker_path", "dump_is_done", "write_dump", "require_gpu_for",
                           "pack_results", "flush_drive", "write_keys", "sgf_to_features", "parse_questions",
                           "_argmax_first", "rule_b")
T02_DUMP_TOOLS_CODE_SEGMENTS = ("record_readonly_after",)
T02_PREFLIGHT_SEGMENTS = ("quick_check_questions",)
T02_SEAM_SEGMENTS = ("_load_npz", "question_spans", "c0_of")

# names of the training notebooks that must not appear in any code cell (ast names)
GONE_NAMES = {"df_val", "df_train", "optimizer", "T_TOTAL", "make_train_loader", "load_rows", "CKPT_DIR",
              "RESULT_CKPT", "FINISH_WITHOUT_TRAINING", "dump_questions", "dump_inner_p03", "dump_inner", "NET",
              "VAL", "INNER_KEYS", "KEY_LOADER", "T02_CHANGES", "P02_REF_GPU", "P02_DUMPS", "DUMP_SYMS",
              "DUMP_NAME", "ARM", "ARM_TAG", "train_loader", "val_loader"}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    with open(p, "rb") as f:
        return sha_bytes(f.read())


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


def base_key(meta):
    """prereg02-notebook cell metadata -> a readable key (setup, prepare, official_2, dump_tools, ...)."""
    if "new" in meta:
        return meta["new"]
    b = meta["base"]
    return "official_%d" % b["official_cell"] if "official_cell" in b else b["new"]


def cell_key(cell):
    """The key of a cell of T02 or P03 (build_prereg03_rank_nb.py cell_key): the base cell's key for reused or
    rewritten cells, the tag for new cells."""
    m = cell["metadata"]["c10"]
    return base_key(m["base"]) if "base_cell" in m else m["new"]


def t04_key(cell):
    """The key of a cell of the built test04 notebook: the base cell's key (in its own notebook) or the new tag."""
    m = cell["metadata"]["c10"]
    return cell_key({"metadata": {"c10": m["base"]}}) if "base_cell" in m else m["new"]


def src_of(cell):
    return "".join(cell["source"])


def func_segments(src):
    """{name: source segment} of the top-level function definitions of a cell."""
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def code_tokens(segment):
    """The token stream of a source segment without comments and blank-line tokens (code identity up to comments)."""
    out = []
    for t in tokenize.generate_tokens(io.StringIO(segment).readline):
        if t.type in (tokenize.COMMENT, tokenize.NL):
            continue
        out.append((t.type, t.string))
    return out


def npy_scalar(npz_path, name):
    """A 0-d array of an .npz written by np.savez (no numpy): '<U..' -> str, '<i8' -> int."""
    with zipfile.ZipFile(npz_path) as z:
        raw = z.read(name + ".npy")
    assert raw[:6] == b"\x93NUMPY", (npz_path, name)
    if raw[6] == 1:
        hl, off = struct.unpack("<H", raw[8:10])[0], 10
    else:
        hl, off = struct.unpack("<I", raw[8:12])[0], 12
    header = ast.literal_eval(raw[off:off + hl].decode("latin1"))
    assert header["shape"] == () and not header["fortran_order"], (npz_path, name, header)
    data, descr = raw[off + hl:], header["descr"]
    if descr.startswith("<U"):
        n = int(descr[2:])
        return data[:4 * n].decode("utf-32-le").rstrip("\x00")
    if descr == "<i8":
        return struct.unpack("<q", data[:8])[0]
    raise AssertionError(("unexpected dtype", npz_path, name, descr))


def literal_assign(path, name):
    """The literal value assigned to a top-level name in a Python file, read with ast (the file is not imported)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    hits = [n.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    assert len(hits) == 1, (path, name, len(hits))
    return ast.literal_eval(hits[0])


def read_json_file(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def derive_pins(repo):
    """Spec 4.1: every pin read from the repository, then compared with the constants above."""
    p03 = os.path.join(repo, "results", "prereg03")
    fuse = os.path.join(repo, "tools", "p04_rank_fuse.py")
    sw_weights, sw_sha = literal_assign(fuse, "SW_WEIGHTS"), literal_assign(fuse, "SW_SHA256")
    with open(os.path.join(repo, "docs", "PREREG_04_OPPONENT.md"), encoding="utf-8") as f:
        prereg04_lines = f.read().splitlines()
    w_pin, w_label, seam_ref, ref_env, tf32 = {}, {}, {}, {}, {}
    for x in ARMS:
        arm_dir = os.path.join(p03, "rank_" + x)
        dumps = os.path.join(arm_dir, "dumps")
        cfg = read_json_file(os.path.join(arm_dir, "config.json"))
        tf32[x] = cfg["runs"][0]["training"]["tf32"]
        n = DUMP_NAMES[x]
        m_d4 = read_json_file(os.path.join(dumps, "_DONE_val_%s_d4.json" % n))
        m_sw = read_json_file(os.path.join(dumps, "_DONE_val_%s_s0_swap.json" % n))
        wsha = cfg["final_sha256"]
        assert m_d4["weights_sha256"] == m_sw["weights_sha256"] == wsha == sw_weights[x], (x, wsha)
        for stem in ("val_%s_d4" % n, "val_%s_s0_swap" % n):
            assert npy_scalar(os.path.join(dumps, stem + ".npz"), "weights_sha256") == wsha, stem
        assert npy_scalar(os.path.join(dumps, "val_%s_d4.npz" % n), "syms") == 8
        assert npy_scalar(os.path.join(dumps, "val_%s_s0_swap.npz" % n), "view") == "s0_colour_swapped"
        assert m_d4["weights"] == m_sw["weights"], x
        w_pin[x], w_label[x] = wsha, m_d4["weights"]
        seam_ref[x] = {}
        for fname in ("val_keys.npz", "val_%s_d4.npz" % n, "val_%s_s0_swap.npz" % n):
            got = sha(os.path.join(dumps, fname))
            assert read_json_file(os.path.join(dumps, "_DONE_%s.json" % fname[:-4]))["sha256"] == got, (x, fname)
            seam_ref[x][fname] = got
        swap_name = "val_%s_s0_swap.npz" % n
        assert seam_ref[x][swap_name] == sw_sha[x]["val"], (x, "p04_rank_fuse.SW_SHA256")
        row = "| `rank_%s/dumps/%s` | `%s` |" % (x, swap_name, seam_ref[x][swap_name])
        assert prereg04_lines.count(row) == 1, ("PREREG_04 1.3 table row", row)
        ref_env[x] = {k: m_d4[k] for k in ("gpu_name", "torch", "cudnn")}
        assert {k: m_sw[k] for k in ("gpu_name", "torch", "cudnn")} == ref_env[x], x
    assert seam_ref["a"]["val_keys.npz"] == seam_ref["b"]["val_keys.npz"]
    assert ref_env["a"] == ref_env["b"], ref_env
    assert tf32["a"] == tf32["b"], tf32
    env = read_json_file(os.path.join(repo, "results", "prereg02", "rank", "env.json"))
    bundle = env["runs"][0]["bundle_manifest"]["files"]
    assert all(r["bundle_manifest"]["files"] == bundle for r in env["runs"]), "prereg02 bundle manifest"
    pins = {"W_PIN": w_pin, "W_LABEL_PIN": w_label, "SEAM_REF_SHA256_PIN": seam_ref, "P03_REF_ENV_PIN": ref_env["a"],
            "TF32_P03_PIN": tf32["a"], "BUNDLE_FILES_PIN": bundle}
    want = {"W_PIN": SPEC_W_PIN, "W_LABEL_PIN": SPEC_W_LABEL, "SEAM_REF_SHA256_PIN": SPEC_SEAM_REF,
            "P03_REF_ENV_PIN": SPEC_REF_ENV, "TF32_P03_PIN": SPEC_TF32, "BUNDLE_FILES_PIN": SPEC_BUNDLE_FILES}
    for k in pins:
        assert pins[k] == want[k], ("pin differs from the spec value", k, pins[k])
    assert bundle["local/rank_val_questions.csv"] == SPEC_SEAM_VAL_QUESTIONS_SHA256
    # review round 2: the rerun-on-G4 rule is one full-name comparison in the notebook (P03_REF_ENV_PIN), in
    # tools/t04_rank.py unpack (G4_GPU) and in tools/official_log.py (T04_RERUN_GPU); read with ast, never imported
    for rel, name in (("tools/t04_rank.py", "G4_GPU"), ("tools/official_log.py", "T04_RERUN_GPU")):
        got = literal_assign(os.path.join(repo, *rel.split("/")), name)
        assert got == SPEC_REF_ENV["gpu_name"] == pins["P03_REF_ENV_PIN"]["gpu_name"], (rel, name, got)
    return pins


# ---------------------------------------------------------------------------
# What changed relative to the two bases. One list feeds the table in notebook cell 0 and `changes_from_bases`
# in config.json. Each row: ((base notebook, base cell key) or None for new/notebook-wide rows, label, text).
# ---------------------------------------------------------------------------
CHANGE_ROWS = [
    (None, "新增", "第一格 markdown：用途、步驟、帳號、GPU、會停下的情況、輸出清單、斷線時怎麼辦（不接續）、這張差異表。"),
    (("test02", "setup"), "設定（test02）",
     "測試集 zip 照 test02（同一份 `C10_test_v1.zip`，MANIFEST 的種類：正式 `official`／彩排 `rehearsal`，SMOKE 只收 `smoke`）。"
     "結果目錄改成 `test04`（彩排 `test04_rehearsal`，SMOKE `test04_smoke`）的 `sw/`（不是 `rank/`：test02 的解壓工具一定拒收這份 zip）。"
     "只讀目錄改成四個：`prereg01`、`prereg02`、`prereg03`，以及 test02 的 `rank` 子資料夾（正式 `test02/rank`、彩排 "
     "`test02_rehearsal/rank`、SMOKE `test02_smoke/rank`）；不放整個 `test02`，因為競賽一的 `test02_player` 會寫 `test02/player/`。"
     "`guard_write()` 只改註解（四個都擋）。`T02_CHANGES` 換成 `T04_CHANGES`，`NOTEBOOK_REL` 改成 `colab/test04_rank.ipynb`，"
     "另記這份 notebook 的 SHA-256（規則見 `NOTEBOOK_SHA256_RULE`）。`config.json` 加 `format`、`kind`，`prereg` 改指 "
     "PREREG_04 rev 3 第 11 節，`base_notebooks` 記兩個底稿的 SHA-256。`run_log.json` 的 `status` 開頭記 `running`，"
     "之前還是 `running` 的執行（停在 GPU 關卡或中途斷線）改記 `interrupted`。在寫任何紀錄之前（`NOTEBOOK_SHA256` 那一行之後），"
     "Colab 上（正式和彩排；SMOKE、本機不問）用輸入框請使用者核對 `notebook 版本`（`NOTEBOOK_SHA256` 前 12 碼）和 Claude 給的 "
     "12 碼，不是 `y` 就停下（還沒寫任何紀錄，不算一次執行）；最後多印一行 `notebook 版本`。"),
    (("test02", "official_2"), "底稿官方 2（test02）",
     "加 `IN_CHANNELS = 501`（03 兩臂的輸入；正式執行 assert）。GPU 關卡照舊，只接受 G4（RTX PRO 6000）或 A100；有 GPU 但查不到型號時，"
     "停下的訊息和 `device` 那一行寫「查不到 GPU 型號」（`GPU_MODEL_SHOWN`，照 prereg03）。最後的 `print` 多印輸入層數。"
     "「沒有 GPU」和「GPU 型號不對」兩則停下訊息都補上「Claude 說這次是重跑時只能選 G4」。"),
    (None, "新增 preflight",
     "在 GPU 關卡之後，任一項不符就記 `preflight_failed` 並停下（還沒開始算，不算一次執行）："
     "① 兩臂權重 `prereg03/rank_{a,b}/ckpt/final.pth` 的 SHA-256 等於登記的值（SMOKE：等於 prereg03_smoke 完成標記記的）；"
     "② 接縫參照（兩臂的 `val_keys.npz`、`val_p03x_d4.npz`、`val_p03x_s0_swap.npz`）的 SHA-256 釘死，npz 記的權重等於①，"
     "03 的 GPU、torch、cuDNN 釘死；③ 打包檔的檔案清單和 SHA-256 等於 prereg01／02／03 用的那一份；"
     "④ 測試集 zip 照 test02 核對（7 個成員的 SHA-256 和大小、快速格式檢查）；"
     "⑤ test02 已完成、用的是同一份測試集（讀 test02 的 `rank/config.json`、`run_log.json`；SMOKE 有才核對）；"
     "⑥ 上一次執行的接縫沒過、這個資料夾之前有接縫類失敗的執行、這個資料夾已經跑完（最後一次有結果的執行是 `completed`、"
     "接縫通過、結果 zip 在：另一則訊息，請使用者下載那份 zip、不要改名）、`sw` 裡已有完成標記（斷線後不接續）、"
     "改名成 `sw_interrupted_*` 的資料夾其實是接縫失敗、占次數的封存已有 2 個（`sw_failed_*` 一律算，`sw_exempt_*` 不算，"
     "其他改過名而且自己有接縫類失敗紀錄的也算；和本機 `t04_rank.py` 同一個算法）：都停下；"
     "有改名成 `sw_failed_*` 的資料夾（重跑）而這次的 GPU 名稱不是 03 完成標記記的 G4 完整名稱：停下（登記的重跑只用 G4，"
     "和本機 `t04_rank.py`、`official_log.py` 同一個完整名稱判定；只有 `sw_exempt_*` 時照第一次的規則；名稱含 RTX PRO 6000 "
     "但不是完整名稱時，訊息開頭是「G4 的名稱和 03 記的不同」；SMOKE 不查）；"
     "⑦ 各組的 `_INPUT.json`（題目檔 SHA-256）已存在但不同就停；⑧ Drive 剩餘空間 ≥ 0.5 GB（不夠時請使用者截圖給 Claude，"
     "不自己刪檔或清垃圾桶）；"
     "⑨ TF32：記下這次的預設值，和 03 記的（卷積開、矩陣乘法關）不同才改成 03 的值，`cudnn.benchmark` 關掉；"
     "⑩ 四個只讀目錄的開頭指紋。`config.json` 頂層記 GPU、torch、cuDNN、兩臂權重、接縫參照、測試集、test02 的紀錄、各組題目檔、TF32。"),
    (None, "沿用 prereg03",
     "視角與 D4 的格子（`NUM_SYMS`、`d4`、`to_target_view`、`dump_input` 和單元檢查）和 prereg03 一字不改。"),
    (None, "新增模型工具",
     "從 prereg03 官方 14 格逐字取出 `state_dict_sha256`、`_nchw_output`、`channels_last` 三個函式；不建模型、不建 optimizer。"),
    (("test02", "dump_tools"), "底稿 dump 工具（test02）",
     "拿掉 `NUM_SYMS`、`d4` 和它的自我檢查（在視角與 D4 的格子）；`load_net` 換成 prereg03 的（501 層、channels_last，一字不改）；"
     "拿掉 test02 的 `dump_questions`，加入 prereg03 的 `dump_val_p03`（整個函式一字不改：每題一個 batch、8 種對稱加一次整題執色對調、"
     "fp32、不開 autocast），再加轉接 `QUESTIONS_PATH` 和 `parse_val()`（讀目前這一組的題目檔；`parse_questions` 依路徑快取，"
     "兩臂共用一份解析結果）。`NET` 改成 `NETS`（兩臂），`record_readonly_after()` 只改註解；其餘函式和 test02 逐字相同。"),
    (None, "新增三組逐盤輸出",
     "`seam_val`（打包檔的本機驗證題 1,589 題）、`public`、`private` 各一格，內容相同、只有第一行不同：先指定 `DUMP_DIR`、"
     "`QUESTIONS_PATH`，兩臂的權重各載入一次，題目檔只解析一次，`write_keys('val_keys', …)`，再依序算 A、B 兩臂的 "
     "`dump_val_p03`；呼叫後核對兩臂完成標記的列數，`SET_LOG` 記這一組的題目檔路徑。"),
    (None, "新增接縫核對",
     "`seam_val` 的逐盤輸出和 Drive 上 03 兩臂的比（PREREG_04 rev 3 第 11 節第 5 條）：每一臂 key 陣列完全相同；原方向正確執色"
     "（d4 的第 0 個對稱）和整題執色對調兩種 logits，每題可解析盤的 softmax float32 平均（C0，和 test02 同一個量）最大差 ≤ tol"
     "（算這份 dump 的 GPU 和 03 的相同時 1e-4，不同時 1e-3）；12 個完成標記和 npz 記的權重依臂相同；三組 × 兩臂的 12 個 logits "
     "標記加 3 個 key 標記是同一個執行環境（GPU、torch、cuDNN）。逐位元組是否相同、8 種對稱和對調的 logits 最大差、E（每題 "
     "float64 log_softmax 平均）最大差、規則 B 不同題數只記錄。寫 `seam_check.json`；沒過就記結尾指紋、`status = seam_failed`、"
     "打包、送出 Drive 的寫入後停下。"),
    (("test02", "final_check"), "底稿結尾（test02，改寫）",
     "改成兩臂：每組核對 `val_keys` 和 4 個 logits 檔的完成標記、SHA-256、大小、形狀（d4 float32 [盤數, 8, 10]、對調 float32 "
     "[盤數, 10]）、權重、NaN 位置、`_INPUT.json`、題數，而且都是這次執行算的；有問題就 `status = final_check_failed`。"
     "記 `readonly_after.json` 並和開頭比對（不同就 `readonly_changed`）；補完 config／timings／run_log；打包 "
     "`test04/test04_rank_results.zip`（zip 內路徑從 `sw/` 起；跳過 `.pth`、`.pt`、`.tmp` 和 ≥ 100 MB 的檔），確認必要的檔都在；"
     "最後 `drive.flush_and_unmount()`。"),
    (None, "拿掉",
     "test02 的 preflight、三組格子、接縫格（換成本版的新格）；prereg03 的讀訓練資料、特徵、loader、建模型、訓練、inner 和 key 檔相關的格子"
     "都沒有用。本版不訓練、不讀訓練資料。"),
    (None, "metadata",
     "notebook metadata 沿用 test02。每格 cell id 是 `c10-rank-t04-NN`；cell metadata `c10` 記底稿（`base_nb` 是 test02 或 "
     "prereg03）、底稿的格號與 `c10`、是否改動，新格記 `new`。"),
]
T04_CHANGES = ["[%s] %s" % (label, text) for _k, label, text in CHANGE_ROWS]


def _md_cell(s):
    return s.replace("|", "\\|")


CHANGE_TABLE = "\n".join("| %s | %s |" % (label, _md_cell(text)) for _k, label, text in CHANGE_ROWS)

# ---------------------------------------------------------------------------
# Notebook cell 0 (markdown)
# ---------------------------------------------------------------------------
MD_HEADER = r"""# C10 測試日（test04）：競賽二 04（03 兩臂讀對手那一方）

**用途**：`docs/PREREG_04_OPPONENT.md` rev 3 第 11 節第 4 條的 Colab 部分（只推論，不訓練、不調參）。用事前登記 03 的兩臂權重（`prereg03/rank_a/ckpt/final.pth`、`rank_b/ckpt/final.pth`，SHA-256 釘死），對三組題目算兩種逐盤 logits：原方向正確執色（8 種對稱，本機只用第 0 個，只為了執色檢查）和整題執色對調（合併項 SWA、SWB）：

1. `seam_val`：本機驗證題 1,589 題（接縫核對用，和 03 當時算的比）。
2. `public`、`private`：官方測試題各 400 題（和 11/04 test02 同一份 `C10_test_v1.zip`）。

合併、執色檢查、規則 B 和交件檔都在本機做（`tools/t04_rank.py`），不在這裡。這份 notebook 不讀訓練資料，也不讀任何答案檔。

## 怎麼跑

0. 只用 **AI Pro 帳號**（Drive、Colab、授權三處都要是它）：03 兩臂的權重只在它的 Drive（`MyDrive/C10/results/prereg03/`）。用錯帳號時 preflight 會找不到權重而停下。
1. `MyDrive/C10/` 裡要有：`C10_bundle_v1.zip`（原本那份）、11/04 test02 用的那一份 `C10_test_v1.zip`（檔名不能改、不要解壓、不要重傳），和這份 `test04_rank.ipynb`。`C10/results/prereg03` 的任何檔都不要動。
2. test02 要先跑完（notebook 會讀 `C10/results/test02/rank/` 的紀錄核對）。競賽一的 `test02_player` 有開著的執行階段時，先不要跑這份（只是在等 G4、沒有開執行階段，不算）。
3. 選單「執行階段 → 變更執行階段類型」：選 **G4 GPU**。G4 是灰色、點不了時，選 **A100 GPU**，「大量 RAM」關掉。**Claude 說「重跑」時只用 G4**。沒有 GPU、或 Colab 改派別的 GPU 時，前面幾格就會停下。
4. 選單「執行階段 → 全部執行」。第一個程式格會要求存取 Google Drive，按允許。接著第一個程式格會跳出一個輸入框，問「notebook 版本：」和 12 碼：和 Claude 在訊息裡給的 12 碼對照，一樣就輸入 `y` 再按 Enter；不一樣（開到舊的或彩排的 notebook）就輸入 `n` 再按 Enter，notebook 會停下（還沒寫任何紀錄，不算一次執行），再「執行階段 → 中斷連線並刪除執行階段」，截圖給 Claude。按下全部執行 1 分鐘內沒有跳出輸入框，就是開到舊的 notebook：馬上「中斷連線並刪除執行階段」，截圖給 Claude。
5. 跑完後（G4 約 4–5 分鐘，推估），最後兩行是「Drive 已寫入完成並卸載」和「完成。請下載…」。先按「執行階段 → 中斷連線並刪除執行階段」，再下載 `MyDrive/C10/results/test04/test04_rank_results.zip`（還沒看到就等幾分鐘，不要下載別的檔名）。**跑完後不要再按「全部執行」**（「Drive 已寫入完成並卸載」那一行後面的「要再跑請重新全部執行」不適用）；不小心按了，notebook 會停下說「這個資料夾已經跑完了」，照它說的下載同一份 zip，不要改名。
6. 跑完前不要關分頁，電腦不要睡眠。

詳細步驟以 `docs/COLAB_RUN.md` 的「測試集 04（test04）」一節為準；和這裡不一致時，以那一節為準。

## 會停下來的情況（錯誤訊息裡都寫了處理方式）

依序：
- **找不到測試集 zip，或 zip 的種類不對**：在第一個程式格停下。
- **notebook 版本沒有確認**（輸入框沒有輸入 `y`）：在第一個程式格停下，還沒寫任何紀錄。
- **沒有 GPU、查不到 GPU 型號，或 GPU 不是 G4／A100**：還沒開始算。先刪除執行階段，換成 G4（灰色時 A100；Claude 說重跑時只能 G4）再全部執行。
- **preflight**（還沒開始算，不算一次執行）：兩臂權重或接縫參照找不到、SHA-256 不符（多半是帳號不對）；打包檔、測試集 zip 和登記的不同；test02 還沒跑完或用的測試集不同；上一次執行的接縫核對沒過；這個資料夾已經跑完（不用再跑，也不要改名）；`sw` 裡已有做到一半的輸出（斷線後不接續）；改名成 `sw_interrupted_*` 的資料夾其實是接縫失敗；占次數的封存（`sw_failed_*`）已經有 2 個；重跑（有 `sw_failed_*` 資料夾）卻不是 G4（`sw_exempt_*` 是不占次數的封存；只有 `sw_exempt_*`、沒有 `sw_failed_*` 時才照第一次的規則）；Drive 空間不足；TF32 設不成 03 的值；只讀目錄和第一次執行時不同。
- **任何寫進 prereg01、prereg02、prereg03、test02/rank 目錄的動作**：一律擋下。
- **接縫核對沒過**：這次執行的 04 測試輸出一律不用、不上傳；R20 不受影響。會先打包 zip 再停下。**不要自己重跑**，下載 zip、截圖給 Claude。
- **結尾核對沒過，或結尾發現只讀目錄被改動**：結果照樣打包，但會停下；截圖給 Claude。

出現紅字時都先截圖給 Claude，不要自己重跑。

## 結果寫到哪裡

`MyDrive/C10/results/test04/sw/`（彩排是 `test04_rehearsal/sw/`）：

| 檔案 | 內容 |
|---|---|
| `<組>/dumps/val_keys.npz` | 每個 slot 一列：`question_id`、`slot`（從 1 起算）、`color`、`parsed`；題目檔的列順序，再依 slot 順序 |
| `<組>/dumps/val_p03a_d4.npz`、`val_p03b_d4.npz` | `logits` float32 [盤數, 8, 10]（原方向正確執色，8 種對稱）、`weights_sha256`、`syms`；解析失敗的 slot 整列是 NaN |
| `<組>/dumps/val_p03a_s0_swap.npz`、`val_p03b_s0_swap.npz` | `logits` float32 [盤數, 10]（原方向、整題執色對調）、`weights_sha256`、`view` |
| `<組>/dumps/_DONE_<檔名>.json` | 完成標記：列數、SHA-256、權重 SHA-256、秒數，以及算它的那次執行的 GPU 型號、torch、cuDNN |
| `<組>/dumps/_INPUT.json` | 這一組用的題目檔和它的 SHA-256 |
| `seam_check.json` | 接縫核對的數字和結果（兩臂分開記） |
| `readonly_before.json`、`readonly_after.json` | 四個只讀目錄每個檔的大小和 mtime（開頭、結尾） |
| `config.json`、`env.json`、`run_log.json`、`timings.json` | 設定、環境、每次執行的紀錄（`status`）、各階段秒數 |

`<組>` 是 `seam_val`、`public`、`private`。`MyDrive/C10/results/test04/test04_rank_results.zip`：上面全部檔案（zip 內路徑從 `sw/` 起），這個檔帶回本機。

## 中途斷線時（不接續）

**不要接著按「全部執行」**。已經算到一半的輸出一律不沿用，整份重算（約 5 分鐘）。照 `docs/COLAB_RUN.md`「中途斷線時」一步一步做，每一步做完都等 Claude 回覆：① 截圖給 Claude，Claude 會給編號 N；② **先**「執行階段 → 中斷連線並刪除執行階段」，等 1–2 分鐘（畫面寫斷線時，Colab 那台機器可能還在寫 Drive）；③ 再在 Google 雲端硬碟把 `C10/results/test04/sw` 改名成 `sw_interrupted_N`（同一層有 `test04_rank_results.zip` 的話也改名成 `test04_rank_results_interrupted_N.zip`），改名前後各截一張圖；④ 開新的執行階段，從「怎麼跑」第 3 步重做；GPU 規則照被打斷的那一次。notebook 看到 `sw` 裡有做到一半的輸出會自己停下。只停在 GPU 關卡或 preflight（還沒有任何完成標記）時，處理好之後在同一個資料夾再按一次「全部執行」就可以。

## 和底稿 notebook 的差異

兩個底稿都是已 commit 的 notebook（產生時核對 SHA-256）：`colab/test02_rank.ipynb`（測試日的骨架）和 `colab/prereg03_rank_a.ipynb`（模型與逐盤輸出；A、B 兩份只差 `ARM` 那一行）。沿用的格子沒列在下表的，和底稿逐字相同。逐格的 unified diff 在 `results/test04/notebook_diff_rank.txt`。

| 格 | 改了什麼 |
|---|---|
__TABLE__

## 其他

- **本機冒煙測試**：`colab/make_test04_smoke_tree.py` 先做 Drive 替身樹（`prereg03_smoke` 的隨機權重和它的驗證題逐盤輸出、種類 `smoke` 的測試集 zip），再用 `colab/run_local_smoke.py` 設定 `C10_SMOKE=1` 在 CPU 上跑：每組只推論前 12 題，結果寫到 `results/test04_smoke/`。
- **不在這裡打分**：答案檔不上傳 Colab；測試集沒有答案。
""".replace("__TABLE__", CHANGE_TABLE)

# ---------------------------------------------------------------------------
# setup (T02 + rep)
# ---------------------------------------------------------------------------
SETUP_NEW_DIRS = r"""# 結果目錄：正式、彩排、SMOKE 分開，絕不混用。本版（test04）只寫 test04（彩排 test04_rehearsal、SMOKE test04_smoke）的 sw/
# （不是 rank/：test02 的解壓工具一定拒收這份 zip；PREREG_04 rev 3 第 11 節第 15 條）
RESULT_ROOT = os.path.join(DRIVE_DIR, 'results',
                           'test04_smoke' if SMOKE else ('test04_rehearsal' if REHEARSAL else 'test04'))
RESULT_DIR = os.path.join(RESULT_ROOT, 'sw')

# 本版：只讀的結果目錄。prereg01、prereg02；prereg03（兩臂的 final.pth 是這次的權重，驗證題逐盤輸出是接縫參照）；
# test02 只放 rank 子資料夾（preflight 讀它的 config.json、run_log.json 核對 test02 已完成）。不放整個 test02：
# 競賽一的 test02_player 會一直寫 test02/player/，同時跑時會被誤判成只讀目錄被改動
P01_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg01_smoke' if SMOKE else 'prereg01')
P02_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg02_smoke' if SMOKE else 'prereg02')
P03_ROOT = os.path.join(DRIVE_DIR, 'results', 'prereg03_smoke' if SMOKE else 'prereg03')
T02_RANK = os.path.join(DRIVE_DIR, 'results', 'test02_smoke' if SMOKE else ('test02_rehearsal' if REHEARSAL else 'test02'),
                        'rank')
READONLY_ROOTS = {'prereg01': P01_ROOT, 'prereg02': P02_ROOT, 'prereg03': P03_ROOT, 'test02_rank': T02_RANK}"""

SETUP_NEW_DEV = r"""# 本版（test04）和兩個底稿 notebook（colab/test02_rank.ipynb、colab/prereg03_rank_a.ipynb）的差異（與第一格的表同一份）
T04_CHANGES = __T04_CHANGES__

# 本版：這份 notebook 的 SHA-256（產生器寫入；規則見 NOTEBOOK_SHA256_RULE）
NOTEBOOK_REL = 'colab/test04_rank.ipynb'
__HASH_LINE__
NOTEBOOK_SHA256_RULE = ('sha256 of json.dumps([the source of every cell], ensure_ascii=False) with this value replaced by '
                        '64 zeros (colab/build_test04_rank_nb.py; tools/t02_rank.py notebook_source_sha256 recomputes it)')

# 本版（review round 3）：寫任何紀錄、算任何東西之前，先請使用者核對 notebook 版本（NOTEBOOK_SHA256 前 12 碼，和 Claude 在
# 訊息裡給的 12 碼比；docs/COLAB_RUN.md 第 9 步）。只在 Colab 上問（正式和彩排；SMOKE、本機不問）。不是 y 就停下：還沒寫任何
# 紀錄、還沒開始算，不算一次執行（上面只建了空的結果目錄）；錯的版本因此不會寫進 sw，也不會算
NOTEBOOK_VERSION = NOTEBOOK_SHA256[:12]
if IN_COLAB and not SMOKE:
    try:
        _answer = input('notebook 版本：%s。和 Claude 在訊息裡給的 12 碼一樣嗎？一樣就輸入 y 再按 Enter；'
                        '不一樣就輸入 n 再按 Enter：' % NOTEBOOK_VERSION)
    except EOFError:
        _answer = ''
    if str(_answer).strip().lower() != 'y':
        raise SystemExit('notebook 版本 %s 沒有確認（輸入的是 %r），停在這裡：還沒寫任何紀錄、還沒開始算，不算一次執行。'
                         '處理方式：「執行階段」→「中斷連線並刪除執行階段」，把這段訊息截圖給 Claude，等 Claude 回覆'
                         '（版本不同多半是開到舊的 notebook：之後照 docs/COLAB_RUN.md 第 6 步從 Drive 的 C10 重開）。'
                         % (NOTEBOOK_VERSION, _answer))

""".replace("__T04_CHANGES__", json.dumps(T04_CHANGES, ensure_ascii=False, indent=4)).replace(
    "__HASH_LINE__", HASH_LINE % HASH_PLACEHOLDER)

SETUP_NEW_PREREG = """    'format': 'c10-test04 rank notebook config v1',
    'prereg': 'docs/PREREG_04_OPPONENT.md rev 3 section 11',
    'rehearsal': REHEARSAL, 'test_kind': TEST_KIND, 'kind': TEST_KIND,
    'notebook': NOTEBOOK_REL, 'notebook_sha256': NOTEBOOK_SHA256, 'notebook_sha256_rule': NOTEBOOK_SHA256_RULE,
    'base_notebooks': {'__T02_REL__': '__T02_SHA__', '__P03_REL__': '__P03_SHA__'},
    'changes_from_bases': T04_CHANGES,
    'readonly_roots': READONLY_ROOTS,
""".replace("__T02_REL__", T02_REL).replace("__T02_SHA__", T02_SHA256).replace(
    "__P03_REL__", P03_REL).replace("__P03_SHA__", P03_SHA256)

SETUP_NEW_STATUS = """run_log['runs'].append(RUN)
# 本版（test04）：這次執行開始時記 running，結束時寫 completed／seam_failed／readonly_changed／final_check_failed，
# preflight 停下時寫 preflight_failed。之前還是 running 的執行（停在 GPU 關卡或中途斷線）改記 interrupted（不算一次執行）
for _r in run_log['runs'][:-1]:
    if isinstance(_r, dict) and _r.get('status', 'running') == 'running' and _r.get('end') is None:
        _r.update(status='interrupted', marked_interrupted_by_run=RUN['run'])
RUN['status'] = 'running'
run_log['status'] = 'running'
"""


def build_setup(t02):
    s = t02["setup"]
    s = rep(s, "# 本版（test02）：測試集 zip（Drive 上的 C10_test_v1.zip）。",
            "# 本版（test04，和 test02 同一份）：測試集 zip（Drive 上的 C10_test_v1.zip）。")
    s = rep(s, cut(s, "# 結果目錄：正式、彩排、SMOKE 分開，絕不混用。", "\n\n\ndef _real(path):"), SETUP_NEW_DIRS)
    s = rep(s, "    # 本版：寫檔、np.savez、zip、rmtree 的目的地一律先過這一關；落在 prereg01 或 prereg02 底下就停\n",
            "    # 本版：寫檔、np.savez、zip、rmtree 的目的地一律先過這一關；落在任何一個只讀目錄（READONLY_ROOTS）底下就停\n")
    s = rep(s, "# 本機工作目錄、test02 的結果目錄都不能和只讀目錄互相包含\n",
            "# 本機工作目錄、test04 的結果目錄都不能和只讀目錄互相包含\n")
    s = rep(s, cut(s, "# 本版（test02）和 prereg02 notebook 的差異（與第一格的表同一份）\n", "# 和官方 notebook 的全部差異"),
            SETUP_NEW_DEV)
    s = rep(s, cut(s, "    'prereg': 'docs/PREREG_02_OPENING_FUSION.md", "    'official_notebook':"), SETUP_NEW_PREREG)
    s = rep(s, "run_log['runs'].append(RUN)\n"
               "run_log['status'] = None   # 本版：這次執行結束時才寫（completed／seam_failed／readonly_changed）\n",
            SETUP_NEW_STATUS)
    s = rep(s, "print('只讀：prereg01', P01_ROOT, '| prereg02', P02_ROOT)",
            "print('只讀：', ' | '.join('%s %s' % _kv for _kv in READONLY_ROOTS.items()))")
    # review round 2: a version the user can compare with the 12 characters Claude gives (docs/COLAB_RUN.md step 10);
    # review round 3: the comparison itself is the input() prompt before any write (SETUP_NEW_DEV); this line records it
    s = rep(s, "print('測試集 zip：%s（種類 %s）' % (TEST_ZIP, TEST_KIND))",
            "print('測試集 zip：%s（種類 %s）' % (TEST_ZIP, TEST_KIND))\n"
            "print('notebook 版本：%s%s' % (NOTEBOOK_VERSION, '（開頭已和 Claude 給的 12 碼對照）' if IN_COLAB and not SMOKE\n"
            "                              else ''))")
    assert s.count("RESULT_DIR = os.path.join(RESULT_ROOT, 'sw')\n") == 1   # the T02_CHANGES name: GONE_NAMES (ast)
    return s


# ---------------------------------------------------------------------------
# official 2 (T02 + rep): 501 input layers, GPU model shown
# ---------------------------------------------------------------------------
def build_official_2(t02):
    s = t02["official_2"]
    s = rep(s, "NUM_CLASSES = len(LEVELS)\n",
            "NUM_CLASSES = len(LEVELS)\n"
            "# 本版（test04）：03 兩臂的輸入是 501 層（250 個盤面 ×「目標玩家的子、對手的子」＋ 1 層「目標玩家執黑」）\n"
            "IN_CHANNELS = 501\n"
            "if not SMOKE:\n"
            "    assert (IN_CHANNELS, NUM_CLASSES, SEED) == (501, 10, 42), 'test04 的設定被改動'\n")
    s = rep(s, "# 本版（test02）：正式執行一定要有 GPU（接縫核對要和 prereg02 在 GPU 上算的逐盤輸出比）\n",
            "# 本版（test04）：正式執行一定要有 GPU（接縫核對要和 03 在 GPU 上算的逐盤輸出比）\n")
    s = rep(s, "GPU_MODEL_NOW = gpu_name_of(ENV_INFO)\n",
            "GPU_MODEL_NOW = gpu_name_of(ENV_INFO)\n"
            "# 本版（照 prereg03）：有 GPU 但查不到型號（nvidia-smi 沒有回報）時，訊息和 device 那一行寫「查不到 GPU 型號」，不寫 None\n"
            "GPU_MODEL_SHOWN = GPU_MODEL_NOW or ('查不到 GPU 型號' if DEVICE.type == 'cuda' else None)\n")
    s = rep(s, "    raise SystemExit('這次拿到的 GPU 是 %s，不是 G4（RTX PRO 6000）或 A100，停在這裡（還沒開始算）。處理方式：'\n",
            "    raise SystemExit('%s，停在這裡（還沒開始算）。處理方式：'\n")
    s = rep(s, "'（灰色時選 A100 GPU，「大量 RAM」關掉）→ 儲存，再按一次「全部執行」。' % GPU_MODEL_NOW)\n",
            "'（灰色時選 A100 GPU，「大量 RAM」關掉；Claude 說這次是重跑時只能選 G4，灰色就隔一段時間再試）→ 儲存，'\n"
            "                     '再按一次「全部執行」。'\n"
            "                     % ('這次拿到的 GPU 是 %s，不是 G4（RTX PRO 6000）或 A100' % GPU_MODEL_NOW if GPU_MODEL_NOW else\n"
            "                        '有 GPU，但查不到 GPU 型號，無法確認是 G4（RTX PRO 6000）或 A100'))\n")
    s = rep(s, "print('device', DEVICE, '| GPU', GPU_MODEL_NOW, '| NPROC', NPROC)",
            "print('device', DEVICE, '| GPU', GPU_MODEL_SHOWN, '| NPROC', NPROC, '| 輸入', IN_CHANNELS, '層')")
    # review round 2: the no-GPU message names the rerun rule too (a rerun only on G4)
    s = rep(s, "'（灰色時選 A100 GPU，「大量 RAM」關掉）→ 儲存，再按一次「全部執行」。')\n",
            "'（灰色時選 A100 GPU，「大量 RAM」關掉；Claude 說這次是重跑時只能選 G4，灰色就隔一段時間再試）→ 儲存，'\n"
            "                     '再按一次「全部執行」。')\n")
    return s


# ---------------------------------------------------------------------------
# preflight (new)
# ---------------------------------------------------------------------------
PREFLIGHT = r"""# 本版新增（preflight，在 GPU 關卡之後；PREREG_04 rev 3 第 11 節第 4、9 條）：兩臂權重、接縫參照、打包檔、測試集 zip、
# test02 已完成、上一次的執行（接縫、重跑次數、不接續）、各組題目檔、Drive 空間、TF32、只讀目錄的指紋。
# 任何一項不符就記 preflight_failed、用中文說明處理方式後停下，還沒開始算（不算一次執行；但執行階段開著就會扣運算單元）
import io

t0 = time.time()
ARMS = ('a', 'b')
DUMP_NAMES = {'a': 'p03a', 'b': 'p03b'}
SETS = ('seam_val', 'public', 'private')
# 產生器從 repo 讀出的釘選值（results/prereg03/rank_{a,b}/config.json、dumps/ 的完成標記與檔案、results/prereg02/rank/env.json；
# tools/p04_rank_fuse.py 的 SW_WEIGHTS、SW_SHA256；PREREG_04 1.3）
W_PIN = __W_PIN__
W_LABEL_PIN = __W_LABEL_PIN__
SEAM_REF_SHA256_PIN = __SEAM_REF__
P03_REF_ENV_PIN = __REF_ENV__
TF32_P03_PIN = __TF32__
BUNDLE_FILES_PIN = __BUNDLE__
MIN_DRIVE_FREE_BYTES = 5 * 10 ** 8
N_TEST_QUESTIONS = 400
SEAM_FAILED_STATUSES = ('seam_failed', 'readonly_changed', 'final_check_failed')   # 接縫類失敗（登記的「重跑一次」）
P03_HELP = ('處理方式：確認這個 Colab 用的是 AI Pro 帳號（Drive、Colab、授權三處都要是它），而且 Drive 上有 '
            'MyDrive/C10/results/prereg03/rank_a/ckpt/final.pth、rank_b/ckpt/final.pth 和 rank_a/dumps/、rank_b/dumps/'
            '（val_keys.npz、val_p03a_d4.npz、val_p03a_s0_swap.npz 等和它們的 _DONE_ 檔）。不要搬動、改名或修改 '
            'C10/results/prereg03 的任何檔案。還是不對就把這段訊息截圖給 Claude。')
RESULT_REL = os.path.relpath(RESULT_DIR, os.path.dirname(DRIVE_DIR)).replace(os.sep, '/')


def preflight_stop(message):
    # 停下前記下 preflight_failed（還沒開始算，不算一次執行），再用中文說明停下
    RUN.update(end=now_iso(), status='preflight_failed')
    run_log['status'] = 'preflight_failed'
    save_config_summary(status='preflight_failed')
    save_run_log()
    raise SystemExit(message)


# ① 兩臂的權重：03 的 final.pth（SMOKE：prereg03_smoke 的完成標記記的那一份）
P03_DUMPS = {_x: os.path.join(P03_ROOT, 'rank_' + _x, 'dumps') for _x in ARMS}
P03_REF_MARKER, W_LABEL, WSHA, W_PATH = {}, {}, {}, {}
for _x in ARMS:
    _m = read_json(os.path.join(P03_DUMPS[_x], '_DONE_val_%s_d4.json' % DUMP_NAMES[_x]))
    if not _m:
        preflight_stop('找不到 prereg03 的 rank_%s/dumps/_DONE_val_%s_d4.json。%s' % (_x, DUMP_NAMES[_x], P03_HELP))
    P03_REF_MARKER[_x] = _m
    if SMOKE:
        W_LABEL[_x], WSHA[_x] = _m.get('weights'), _m.get('weights_sha256')
    else:
        W_LABEL[_x], WSHA[_x] = W_LABEL_PIN[_x], W_PIN[_x]
        if (_m.get('weights'), _m.get('weights_sha256')) != (W_LABEL[_x], WSHA[_x]):
            preflight_stop('prereg03 的 rank_%s/dumps/_DONE_val_%s_d4.json 記的權重不是 %s（%s…）。%s'
                           % (_x, DUMP_NAMES[_x], W_LABEL[_x], WSHA[_x][:12], P03_HELP))
    if W_LABEL[_x] != W_LABEL_PIN[_x] or not WSHA[_x]:
        preflight_stop('prereg03 的完成標記記的權重是 %r（%r），不是 %s。請截圖給 Claude。'
                       % (W_LABEL[_x], WSHA[_x], W_LABEL_PIN[_x]))
    W_PATH[_x] = os.path.join(P03_ROOT, *W_LABEL[_x][len('prereg03/'):].split('/'))
    if not os.path.isfile(W_PATH[_x]):
        preflight_stop('找不到 %s 臂的權重 %s。%s' % (_x.upper(), W_PATH[_x], P03_HELP))
    if sha256_path(W_PATH[_x]) != WSHA[_x]:
        preflight_stop('%s 臂的權重 %s 的 SHA-256 不是登記的 %s…。%s' % (_x.upper(), W_PATH[_x], WSHA[_x][:12], P03_HELP))

# ② 接縫參照：03 兩臂在 Drive 上的本機驗證題逐盤輸出（正式執行釘 SHA-256；SMOKE 核對完成標記）、npz 記的權重、03 的執行環境
SEAM_REF, P03_REF_ENV = {}, {}
for _x in ARMS:
    SEAM_REF[_x] = {}
    for _n in ('val_keys.npz', 'val_%s_d4.npz' % DUMP_NAMES[_x], 'val_%s_s0_swap.npz' % DUMP_NAMES[_x]):
        _p = os.path.join(P03_DUMPS[_x], _n)
        if not os.path.isfile(_p):
            preflight_stop('找不到接縫參照 %s。%s' % (_p, P03_HELP))
        SEAM_REF[_x][_n] = sha256_path(_p)
        _want = ((read_json(os.path.join(P03_DUMPS[_x], '_DONE_%s.json' % _n[:-4])) or {}).get('sha256') if SMOKE
                 else SEAM_REF_SHA256_PIN[_x][_n])
        if SEAM_REF[_x][_n] != _want:
            preflight_stop('接縫參照 %s 的 SHA-256 不符。%s' % (_p, P03_HELP))
    for _n in ('val_%s_d4.npz' % DUMP_NAMES[_x], 'val_%s_s0_swap.npz' % DUMP_NAMES[_x]):
        with np.load(os.path.join(P03_DUMPS[_x], _n), allow_pickle=False) as _z:
            if str(_z['weights_sha256']) != WSHA[_x]:
                preflight_stop('接縫參照 %s 記的權重不是 %s…。%s' % (_n, WSHA[_x][:12], P03_HELP))
    P03_REF_ENV[_x] = {_k: P03_REF_MARKER[_x].get(_k) for _k in ('gpu_name', 'torch', 'cudnn')}
    if not SMOKE and P03_REF_ENV[_x] != P03_REF_ENV_PIN:
        preflight_stop('03 的 rank_%s 完成標記記的執行環境是 %s，不是登記時的 %s。%s' % (_x, P03_REF_ENV[_x], P03_REF_ENV_PIN, P03_HELP))
P03_REF_GPU = P03_REF_ENV['a']['gpu_name']   # 接縫的容許差：和算這次 seam_val dump 的 GPU 比

# ③ 打包檔：和 prereg01、prereg02、prereg03 用的是同一份（準備格已逐檔核對 SHA-256 和 bundle_manifest.json 相符）
if MANIFEST['files'] != BUNDLE_FILES_PIN:
    preflight_stop('Drive 上的 C10_bundle_v1.zip 和 prereg01／prereg02／prereg03 用的那一份不同（檔案清單或 SHA-256 不同）。'
                   '處理方式：不要自己重傳，把這段訊息截圖給 Claude。')

# ④ 測試集 zip（照 test02）：只收 tests/ 底下的 6 個官方檔名和 MANIFEST.json，解到 WORK/tests；每個檔的 SHA-256、大小要等於
#    MANIFEST；快速格式檢查（T1 的一部分）
TESTS_DIR = os.path.join(WORK, 'tests')
_allowed = {'tests/' + n for n in OFFICIAL_TEST_NAMES} | {TEST_MANIFEST_MEMBER}
shutil.rmtree(TESTS_DIR, ignore_errors=True)
os.makedirs(TESTS_DIR)
with zipfile.ZipFile(TEST_ZIP) as _z:
    _members = [i.filename for i in _z.infolist() if not i.filename.endswith('/')]
    _bad = [m for m in _members if m not in _allowed]
    _absent = sorted(_allowed - set(_members))
    if _bad or _absent or len(_members) != len(set(_members)):
        preflight_stop('測試集 zip 的內容不對（多出：%s；缺少：%s）。%s' % (_bad[:5], _absent, TEST_HELP))
    for _m in _members:
        with open(os.path.join(TESTS_DIR, _m[len('tests/'):]), 'wb') as _f:
            _f.write(_z.read(_m))
TEST_ZIP_SHA = sha256_path(TEST_ZIP)
TEST_MANIFEST_SHA = sha256_path(os.path.join(TESTS_DIR, 'MANIFEST.json'))
if read_json(os.path.join(TESTS_DIR, 'MANIFEST.json')) != TEST_MANIFEST:
    preflight_stop('測試集 zip 的 MANIFEST 和第一格讀到的不同。%s' % TEST_HELP)
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
    preflight_stop('測試集 zip 核對沒過：%s。%s' % ('；'.join(_problems), TEST_HELP))


__QUICK_CHECK__


_ids = {}
for _set, _prefix in (('public', 'pub_'), ('private', 'priv_')):
    _pr, _ids[_set] = quick_check_questions(os.path.join(TESTS_DIR, TEST_FILES[_set]), _prefix)
    _problems += ['%s：%s' % (TEST_FILES[_set], x) for x in _pr[:10]]
if set(_ids['public']) & set(_ids['private']):
    _problems.append('public 和 private 有相同的 question_id')
if _problems:
    preflight_stop('測試題目檔的格式檢查沒過（%d 項）：%s。處理方式：不要改檔，把這段訊息截圖給 Claude。'
                   % (len(_problems), '；'.join(_problems[:10])))


# ⑤ test02 已完成、用的是同一份測試集（讀 test02 的 rank/config.json、run_log.json；只讀）。正式和彩排一定要有；SMOKE 有才核對
def test02_state(rank_dir, manifest_sha, required):
    # 回傳 (紀錄, 問題或 None)
    cfg_path, log_path = os.path.join(rank_dir, 'config.json'), os.path.join(rank_dir, 'run_log.json')
    cfg, log = read_json(cfg_path), read_json(log_path)
    rec = {'config_sha256': sha256_path(cfg_path) if cfg is not None else None,
           'manifest_sha256': ((cfg.get('test_bundle') or {}).get('manifest_sha256') if isinstance(cfg, dict) else None),
           'run_log_status': log.get('status') if isinstance(log, dict) else None}
    if cfg is None and log is None and not required:
        return rec, None
    if cfg is None or log is None:
        return rec, 'test02 的 config.json 或 run_log.json 不在 %s' % rank_dir
    if rec['run_log_status'] != 'completed':
        return rec, 'test02 的 run_log.json 狀態是 %r，不是 completed' % (rec['run_log_status'],)
    if rec['manifest_sha256'] != manifest_sha:
        return rec, ('test02 用的測試集 MANIFEST（%s…）和這次的（%s…）不同'
                     % (str(rec['manifest_sha256'])[:12], manifest_sha[:12]))
    return rec, None


TEST02_RECORD, _t02_problem = test02_state(T02_RANK, TEST_MANIFEST_SHA, required=not SMOKE)
if _t02_problem:
    preflight_stop('%s。處理方式：先做完 test02；C10 裡只能有 11/04 那一份 C10_test_v1.zip（不要重傳、不要換）。'
                   '還是不對就把這段訊息截圖給 Claude。' % _t02_problem)


# ⑥ 上一次的執行、重跑次數、不接續（本機 t04_rank.py unpack／archive 是最終把關，這裡提早擋）
def seam_failure_of(folder):
    # 一個 sw 類資料夾的接縫類失敗紀錄：seam_check.json 不是 passed，或 run_log 有 seam_failed／readonly_changed／final_check_failed
    reasons = []
    sc = read_json(os.path.join(folder, 'seam_check.json'))
    if sc is not None and (not isinstance(sc, dict) or sc.get('passed') is not True):
        reasons.append('seam_check.json passed = %r' % (sc.get('passed') if isinstance(sc, dict) else sc,))
    rl = read_json(os.path.join(folder, 'run_log.json'))
    if isinstance(rl, dict):
        st = [r.get('status') for r in (rl.get('runs') or []) if isinstance(r, dict)] + [rl.get('status')]
        reasons += ['run_log %s' % s for s in sorted({s for s in st if s in SEAM_FAILED_STATUSES})]
    return reasons


# ⑥-1 上一次執行的接縫核對沒過就停：這個資料夾的輸出不能沿用，登記的「重跑一次」必須在新的資料夾整份重算
_prev_seam = read_json(os.path.join(RESULT_DIR, 'seam_check.json'))
if _prev_seam is not None and _prev_seam.get('passed') is not True:
    preflight_stop('上一次執行的接縫核對沒過（%s 記 passed = %r）。這個資料夾裡的逐盤輸出不能沿用，重跑必須重算。'
                   '處理方式：先把畫面截圖給 Claude，再按「執行階段」→「中斷連線並刪除執行階段」；Claude 說要重跑時，'
                   '在 Google 雲端硬碟把 %s 整個資料夾改名成 Claude 給的名稱（同一層，通常是 sw_failed_N），同一層的 '
                   'test04_rank_results.zip 也照 Claude 給的名稱改名（通常是 test04_rank_results_failed_N.zip），'
                   '等 Claude 回覆後開新的執行階段（GPU 照 Claude 說的），再按「全部執行」。'
                   % (os.path.join(RESULT_DIR, 'seam_check.json'), _prev_seam.get('passed'), RESULT_REL))
# ⑥-2 這個資料夾之前有接縫類失敗的執行（同一個資料夾又跑了一次，不是登記的重跑）
_prev_bad = [(r.get('run'), r.get('status')) for r in run_log['runs'][:-1]
             if isinstance(r, dict) and r.get('status') in SEAM_FAILED_STATUSES]
if _prev_bad:
    preflight_stop('%s 之前有接縫類失敗的執行（%s），這個資料夾不能再用。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                   % (RESULT_REL, _prev_bad))
# ⑥-3 不接續：sw 裡已經有任何完成標記（上一次斷線或已經跑過）就停
_done_now = []
if os.path.isdir(RESULT_DIR):
    for _s in sorted(os.listdir(RESULT_DIR)):
        _d = os.path.join(RESULT_DIR, _s, 'dumps')
        if os.path.isdir(_d):
            _done_now += ['%s/%s' % (_s, _f) for _f in sorted(os.listdir(_d)) if _f.startswith('_DONE_')]
# ⑥-3a（review round 3）這個資料夾已經跑完：之前最後一次有結果的執行是 completed、接縫通過、結果 zip 也在。不是斷線，
#      不改名，直接下載那份 zip（跑完後又按了一次「全部執行」）
_prev_result_runs = [r for r in run_log['runs'][:-1]
                     if isinstance(r, dict) and r.get('status') not in ('preflight_failed', 'interrupted', 'running')]
RESULTS_ZIP_DONE = os.path.join(RESULT_ROOT, 'test04_rank_results.zip')
if (_done_now and _prev_result_runs and _prev_result_runs[-1].get('status') == 'completed'
        and _prev_seam is not None and _prev_seam.get('passed') is True and os.path.isfile(RESULTS_ZIP_DONE)):
    preflight_stop('這個資料夾已經跑完了（第 %s 次執行 completed，接縫通過），不用再跑，也不要改名。處理方式：'
                   '「執行階段」→「中斷連線並刪除執行階段」，照 docs/COLAB_RUN.md 第 13–15 步下載 %s，'
                   '把這段訊息截圖給 Claude。'
                   % (_prev_result_runs[-1].get('run'), os.path.relpath(RESULTS_ZIP_DONE, os.path.dirname(DRIVE_DIR))
                      .replace(os.sep, '/')))
if _done_now:
    preflight_stop('斷線後不接續：%s 裡已經有逐盤輸出的完成標記（%s）。處理方式（照 docs/COLAB_RUN.md「中途斷線時」，'
                   '一步做完等 Claude 回覆再做下一步）：先截圖給 Claude；「執行階段」→「中斷連線並刪除執行階段」，等 1–2 分鐘；'
                   '再在 Google 雲端硬碟把 %s 改名成 sw_interrupted_N（N 照 Claude 給的；同一層有 test04_rank_results.zip '
                   '的話也改名成 test04_rank_results_interrupted_N.zip），改名前後截圖；最後開新的執行階段，再按「全部執行」整份重算。'
                   % (RESULT_REL, _done_now[:3], RESULT_REL))
# ⑥-4 RESULT_ROOT 底下改過名的 sw_* 資料夾。review round 3：和本機 t04_rank.py 同一個算法——sw_failed_*（本機 archive 判
#     占次數的封存，包括 Drive 上接縫本身通過、本機 apply 才判失敗的那種）一律算一次；sw_exempt_*（本機判不占次數：R20 上傳前
#     跑的、依 PREREG_04 修訂的程式錯誤）不算；其他改過名、自己有接縫類失敗紀錄的資料夾也算（安全的一邊）
PRIOR_SW = {}
if os.path.isdir(RESULT_ROOT):
    for _d in sorted(os.listdir(RESULT_ROOT)):
        if _d.startswith('sw_') and os.path.isdir(os.path.join(RESULT_ROOT, _d)):
            PRIOR_SW[_d] = seam_failure_of(os.path.join(RESULT_ROOT, _d))
_interrupted_but_failed = [_d for _d, _r in PRIOR_SW.items() if _d.startswith('sw_interrupted_') and _r]
if _interrupted_but_failed:
    preflight_stop('%s 不是斷線，是接縫失敗（%s）。處理方式：不要再執行，也不要改名，把這段訊息截圖給 Claude'
                   '（Claude 照 docs/TESTDAY_04.md「斷線」一節處理）。'
                   % ('、'.join(_interrupted_but_failed), {_d: PRIOR_SW[_d] for _d in _interrupted_but_failed}))
SEAM_FAILED_FOLDERS = sorted(_d for _d, _r in PRIOR_SW.items()
                             if not _d.startswith('sw_exempt_') and (_d.startswith('sw_failed_') or _r))
if len(SEAM_FAILED_FOLDERS) >= 2:
    preflight_stop('接縫失敗的資料夾已經有 %d 個（%s）：登記只准重跑一次。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                   % (len(SEAM_FAILED_FOLDERS), '、'.join(SEAM_FAILED_FOLDERS)))
ATTEMPT = len(SEAM_FAILED_FOLDERS) + 1   # 1：第一次；2：登記的重跑


# ⑥-5 登記的重跑只用 G4（PREREG_04 rev 3 第 4、9 條）：RESULT_ROOT 底下有改名成 sw_failed_* 的資料夾（占次數的封存）時，這次
#     就是重跑（或重跑中途斷線後再跑），GPU 一定要是 G4。只有 sw_exempt_*（不占次數的封存）、沒有 sw_failed_* 時照第一次的規則（G4 優先、A100 備用），
#     和本機 t04_rank.py unpack、official_log.py 相同（都是「有一個占次數的封存才是登記的重跑」）。SMOKE（本機 CPU）不查。
#     在任何計算之前停下，不算一次執行。G4 用完整名稱比（03 完成標記記的、P03_REF_ENV_PIN 的那一個）
def rerun_gpu_problem(prior_sw, gpu_model, g4_name):
    # 回傳停下的訊息；不必停下時回傳 None
    failed = sorted(_d for _d in prior_sw if _d.startswith('sw_failed_'))
    if failed and gpu_model != g4_name:
        if 'RTX PRO 6000' in (gpu_model or ''):
            # review round 3：開頭不同（docs/COLAB_RUN.md「出狀況時」用開頭分辨這兩種訊息）
            return ('G4 的名稱和 03 記的不同（這次是 %s，03 記的是 %s）。這是登記的重跑（%s 已改名封存），只能用 G4，'
                    '本機會拒收這次的輸出，停在這裡（還沒開始算，不算一次執行）。處理方式：「執行階段」→「中斷連線並刪除執行階段」，'
                    '把這段訊息截圖給 Claude，不要再執行。' % (gpu_model, g4_name, '、'.join(failed)))
        return ('這是登記的重跑（%s 已改名封存），只能用 G4（%s）；這次拿到的是 %s，停在這裡（還沒開始算，不算一次執行）。'
                '處理方式：「執行階段」→「中斷連線並刪除執行階段」，隔一段時間再選 G4 GPU 試（最晚 11/15 中午），'
                '還是拿不到就告訴 Claude。不要改選 A100 或其他 GPU。'
                % ('、'.join(failed), g4_name, gpu_model or '查不到型號的 GPU'))
    return None


RERUN_GPU_PROBLEM = None if SMOKE else rerun_gpu_problem(PRIOR_SW, GPU_MODEL_NOW, P03_REF_ENV_PIN['gpu_name'])
if RERUN_GPU_PROBLEM:
    preflight_stop(RERUN_GPU_PROBLEM)

# ⑦ 各組的題目檔與 _INPUT.json：已存在但不同就停（照 test02）
SET_QUESTIONS = {'seam_val': RANK_QUESTIONS, 'public': os.path.join(TESTS_DIR, TEST_FILES['public']),
                 'private': os.path.join(TESTS_DIR, TEST_FILES['private'])}
SET_INPUT, SET_ROWS, SET_QCOUNT = {}, {}, {}
for _set in SETS:
    _q = SET_QUESTIONS[_set]
    with open(_q, newline='', encoding='utf-8-sig') as _f:
        _n_rows = sum(1 for _ in csv.DictReader(_f))
    SET_QCOUNT[_set] = _n_rows
    SET_ROWS[_set] = min(_n_rows, SMOKE_CFG['questions']) if SMOKE else _n_rows
    _rec = {'set': _set, 'question_file': ('bundle local/' if _set == 'seam_val' else 'tests/') + os.path.basename(_q),
            'sha256': sha256_path(_q)}
    _dir = os.path.join(RESULT_DIR, _set, 'dumps')
    _ip = os.path.join(_dir, '_INPUT.json')
    _old = read_json(_ip)
    if _old is not None and _old != _rec:
        preflight_stop('%s 已有 _INPUT.json，但當時的題目檔和這次不同（%s）。處理方式：不要刪檔，把這段訊息截圖給 Claude。'
                       % (_set, _ip))
    if _old is None and os.path.isdir(_dir) and os.listdir(_dir):
        preflight_stop('%s 有檔案但沒有 _INPUT.json，不知道是用哪個題目檔算的。處理方式：不要刪檔，把這段訊息截圖給 Claude。' % _dir)
    guard_write(_dir)
    os.makedirs(_dir, exist_ok=True)
    if _old is None:
        write_json(_ip, _rec)
    SET_INPUT[_set] = _rec

# ⑧ Drive 剩餘空間
try:
    DRIVE_FREE_BYTES = shutil.disk_usage(DRIVE_DIR).free
except OSError as e:
    DRIVE_FREE_BYTES = None
    print('注意：查不到 Drive 剩餘空間（%s: %s），請自己確認還有 0.5 GB 以上' % (type(e).__name__, e))
if DRIVE_FREE_BYTES is not None and DRIVE_FREE_BYTES < MIN_DRIVE_FREE_BYTES:
    preflight_stop('Drive 只剩 %.2f GB，這份 notebook 要 0.5 GB 以上。處理方式：不要自己刪檔、也不要清垃圾桶；'
                   '「執行階段」→「中斷連線並刪除執行階段」，把這段訊息截圖給 Claude，等 Claude 回覆。'
                   % (DRIVE_FREE_BYTES / 1e9))


# ⑨ TF32：先記下這個執行階段的預設值；和 03 記的（卷積 TF32 開、矩陣乘法關）相同就不動，不同才用舊的設定方式改成 03 的值
#    （改不成就停）；cudnn.benchmark 是 True 時改成 False。預設值和實際用的值都記進 config.json
def tf32_flags():
    return {'cudnn': bool(torch.backends.cudnn.allow_tf32), 'matmul': bool(torch.backends.cuda.matmul.allow_tf32)}


def cudnn_conv_fp32_precision():
    _conv = getattr(torch.backends.cudnn, 'conv', None)
    return getattr(_conv, 'fp32_precision', None) if _conv is not None else None


TF32 = {'recorded_03': dict(TF32_P03_PIN), 'default': tf32_flags(), 'benchmark_default': bool(torch.backends.cudnn.benchmark),
        'deterministic_default': bool(torch.backends.cudnn.deterministic),
        'cudnn_conv_fp32_precision_default': cudnn_conv_fp32_precision()}
try:
    if TF32['default'] != TF32_P03_PIN:
        torch.backends.cudnn.allow_tf32 = TF32_P03_PIN['cudnn']
        torch.backends.cuda.matmul.allow_tf32 = TF32_P03_PIN['matmul']
    if torch.backends.cudnn.benchmark:
        torch.backends.cudnn.benchmark = False
except (RuntimeError, AttributeError) as e:
    preflight_stop('TF32 設不成 03 記的值（%s: %s）。處理方式：不要改 notebook，把這段訊息截圖給 Claude。' % (type(e).__name__, e))
TF32.update(used=tf32_flags(), benchmark_used=bool(torch.backends.cudnn.benchmark),
            deterministic_used=bool(torch.backends.cudnn.deterministic),
            cudnn_conv_fp32_precision_used=cudnn_conv_fp32_precision())
TF32['changed'] = bool(TF32['used'] != TF32['default'] or TF32['benchmark_used'] != TF32['benchmark_default'])
if TF32['used'] != TF32_P03_PIN or TF32['benchmark_used']:
    preflight_stop('TF32 設不成 03 記的值（現在 %s，benchmark %s；03 是 %s）。處理方式：不要改 notebook，把這段訊息截圖給 Claude。'
                   % (TF32['used'], TF32['benchmark_used'], TF32_P03_PIN))

# ⑩ 只讀目錄的指紋（開頭）。重跑時必須和第一次執行記下的相同：不同代表只讀目錄在兩次執行之間被改過
READONLY_BEFORE = {_name: readonly_fingerprint(_root) for _name, _root in READONLY_ROOTS.items()}
_before_path = os.path.join(RESULT_DIR, 'readonly_before.json')
_before_log = read_runs(_before_path)
if _before_log['runs'] and _before_log['runs'][0]['files'] != READONLY_BEFORE:
    _first = _before_log['runs'][0]['files']
    _changed = {_k: sorted(f for f in set(_first.get(_k, {})) | set(READONLY_BEFORE[_k])
                           if _first.get(_k, {}).get(f) != READONLY_BEFORE[_k].get(f)) for _k in READONLY_BEFORE}
    preflight_stop('只讀目錄和第一次執行時不同（%s）。處理方式：不要再執行，把這段訊息截圖給 Claude。'
                   % {k: v[:5] for k, v in _changed.items() if v})
_before_log['runs'].append({'run': RUN['run'], 'time': now_iso(), 'roots': READONLY_ROOTS,
                            'n_files': {k: len(v) for k, v in READONLY_BEFORE.items()}, 'files': READONLY_BEFORE})
write_json(_before_path, _before_log)

save_config(preflight={'weights': {_x: {'label': W_LABEL[_x], 'sha256': WSHA[_x]} for _x in ARMS},
                       'seam_reference_sha256': SEAM_REF, 'p03_reference_env': P03_REF_ENV,
                       'test_zip_sha256': TEST_ZIP_SHA, 'test_manifest_sha256': TEST_MANIFEST_SHA, 'test_kind': TEST_KIND,
                       'sets': SET_INPUT, 'test02_record': TEST02_RECORD, 'prior_sw_folders': PRIOR_SW, 'attempt': ATTEMPT,
                       'tf32': TF32, 'drive_free_bytes': DRIVE_FREE_BYTES,
                       'readonly_files_fingerprinted': {k: len(v) for k, v in READONLY_BEFORE.items()}})
save_config_summary(gpu=GPU_MODEL_NOW, torch=torch.__version__, cudnn=ENV_INFO.get('cudnn'), device=str(DEVICE),
                    weights={_x: {'label': W_LABEL[_x], 'sha256': WSHA[_x]} for _x in ARMS},
                    seam_reference={'a': SEAM_REF['a'], 'b': SEAM_REF['b'], 'gpu': P03_REF_GPU, 'env': P03_REF_ENV},
                    test_bundle={'zip_sha256': TEST_ZIP_SHA, 'manifest_sha256': TEST_MANIFEST_SHA,
                                 'kind': TEST_KIND, 't1': TEST_MANIFEST.get('t1'),
                                 'files': {n: _mf[n]['sha256'] for n in OFFICIAL_TEST_NAMES}},
                    test02_record=TEST02_RECORD,
                    sets={s: {'questions': SET_INPUT[s]['question_file'], 'questions_sha256': SET_INPUT[s]['sha256'],
                              'questions_count': SET_QCOUNT[s], 'rows': SET_ROWS[s]} for s in SETS},
                    tf32=TF32, attempt=ATTEMPT, prior_sw_folders=PRIOR_SW)
mark_time('preflight', time.time() - t0)
print('preflight 通過：權重 A %s…、B %s…，測試集 zip %s（%s…），題目 %s，03 的 GPU：%s，TF32 %s（%s），第 %d 次'
      % (WSHA['a'][:12], WSHA['b'][:12], TEST_KIND, TEST_ZIP_SHA[:12], SET_ROWS, P03_REF_GPU, TF32['used'],
         '改過' if TF32['changed'] else '預設值', ATTEMPT))"""

# ---------------------------------------------------------------------------
# model tools (new; three functions verbatim from P03's official 14)
# ---------------------------------------------------------------------------
MODEL_TOOLS = r"""# 本版新增：模型工具（從 prereg03 的官方 14 格逐字取出三個函式；不建模型、不建 optimizer）
# - channels_last：卷積用 NHWC 排列，bn_in2 的輸出轉回 NCHW（和 03 載入 final.pth 算逐盤輸出時相同）
# - state_dict_sha256：載入後的權重摘要（只記錄）
__MODEL_DEFS__


print('模型工具就緒：channels_last、state_dict_sha256')"""

# ---------------------------------------------------------------------------
# dump tools (T02 + rep)
# ---------------------------------------------------------------------------
DT_NEW_HEAD = """# 本版新增：逐盤輸出（dump）工具（test04：以 test02 的工具為底；load_net 和 dump_val_p03 從 prereg03 逐字搬來）
# - 題目檔（本機驗證題或官方測試題）：照題目檔的列順序和 slot 1..num_games 解析，失敗的保留位置（logits 填 NaN）；
#   每題一個 batch（每批最多 GPU_BATCH 盤），和 03 算驗證題 dump 時相同，接縫核對才對得上
# - 一律 fp32、不開 autocast；先做視角轉換再做 D4（dump_input 在視角與 D4 的格子）；8 種對稱（val_p03x_d4）和
#   「原方向、整題執色對調」（val_p03x_s0_swap）一起算（03 的 dump_val_p03 一字不改）
# - DUMP_DIR、QUESTIONS_PATH 由各組的格子在呼叫前指定；底稿的函式把它們當全域變數讀（QUESTIONS_PATH 經 parse_val()）
# - 每個檔先寫 .tmp 再改名，附 _DONE_<檔名>.json
"""
DT_NEW_RECORD = """def record_readonly_after():
    # 本版：四個只讀目錄（prereg01、prereg02、prereg03、test02 的 rank）的指紋（結尾，或接縫核對沒過時）：和這次開頭記的比，
    # 寫進 readonly_after.json；回傳 (指紋, {目錄: 不同的檔})
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
DT_ADAPTER = """QUESTIONS_PATH = None   # 本版：各組的格子在呼叫前指定（同 DUMP_DIR）；03 的 dump_val_p03 一字不改地呼叫 parse_val()


def parse_val():
    # 本版（轉接）：03 的 dump_val_p03 讀 parse_val()；這裡改成讀目前這一組的題目檔 QUESTIONS_PATH
    # （parse_questions 依路徑快取，所以兩臂拿到的是同一份解析結果）
    return parse_questions(QUESTIONS_PATH)


__DUMP_VAL_P03__


"""
DT_NEW_TAIL = """NETS = {}               # 本版：兩臂的網路（各約 105 MB），第一組的格子各載入一次，三組共用
NET_STATE_SHA256 = {}   # 本版：載入後的 state_dict 摘要（只記錄）
SET_LOG = {}            # 本版：每組的逐盤輸出紀錄（寫進 config.json）
# 本版：兩臂的名稱、完成標記的權重標籤和 03 算驗證題 dump 時相同（npz 和完成標記的權重欄位才和 03 一模一樣）
assert ARMS == ('a', 'b') and DUMP_NAMES == {'a': 'p03a', 'b': 'p03b'}, (ARMS, DUMP_NAMES)
assert W_LABEL == {_x: 'prereg03/rank_%s/ckpt/final.pth' % _x for _x in ARMS}, W_LABEL
print('dump 工具就緒：%s；%d 種方向＋整題執色對調'
      % ('；'.join('%s 臂 %s（SHA-256 %s…）' % (_x.upper(), W_LABEL[_x], WSHA[_x][:12]) for _x in ARMS), NUM_SYMS))"""


def build_dump_tools(t02, p03_dt_defs):
    s = t02["dump_tools"]
    s = rep(s, cut(s, "# 本版新增：逐盤輸出（dump）工具（test02", "import io\n"), DT_NEW_HEAD)
    s = rep(s, "GPU_BATCH = 64                         # 和 prereg01 推論相同\nNUM_SYMS = 8\n",
            "GPU_BATCH = 64                         # 和 prereg01 推論相同（NUM_SYMS 在視角與 D4 的格子）\n")
    s = rep(s, cut(s, "def d4(x, s):", "\n\n\ndef load_net("),
            "# 本版：d4() 和它的自我檢查在視角與 D4 的格子（prereg03 的那一格，一字不改）")
    s = rep(s, func_segments(s)["load_net"], p03_dt_defs["load_net"])
    s = rep(s, cut(s, "def record_readonly_after():", "def pack_results(zip_name):"), DT_NEW_RECORD)
    s = rep(s, cut(s, "def dump_questions(net, path, name, syms, weights, weights_sha):", "_E1 = math.exp(-1)"),
            DT_ADAPTER.replace("__DUMP_VAL_P03__", p03_dt_defs["dump_val_p03"]))
    s = rep(s, "NET = None      # 本版：三組共用的網路，第一組的格子載入一次\n"
               "SET_LOG = {}    # 本版：每組的逐盤輸出紀錄（寫進 config.json）\n"
               "print('dump 工具就緒：權重 %s（SHA-256 %s…），%d 種方向' % (W_LABEL, WSHA[:12], NUM_SYMS))", DT_NEW_TAIL)
    return s


# ---------------------------------------------------------------------------
# the three set cells (new; identical bodies)
# ---------------------------------------------------------------------------
SET_FIRST_LINE = "SET = '%s'   # 本版新增：這一格處理的題目組（seam_val、public、private 三格只有這一行不同）"
SET_BODY = r"""# 一組題目的逐盤輸出（03 兩臂的 final.pth：8 種對稱＋整題執色對調），寫到 <組>/dumps
t0 = time.time()
_path = SET_QUESTIONS[SET]
DUMP_DIR = guard_write(os.path.join(RESULT_DIR, SET, 'dumps'))   # 底稿的 dump 函式把 DUMP_DIR 當全域變數讀
os.makedirs(DUMP_DIR, exist_ok=True)
if read_json(os.path.join(DUMP_DIR, '_INPUT.json')) != SET_INPUT[SET]:
    raise SystemExit('%s 的 _INPUT.json 和 preflight 記的不同。請把這段訊息截圖給 Claude。' % SET)
QUESTIONS_PATH = _path   # 03 的 dump_val_p03 經 parse_val() 讀這一組的題目檔
for _x in ARMS:
    if _x not in NETS:   # 兩臂的權重各只載入一次
        NETS[_x], _ = load_net(W_PATH[_x], expect_sha=WSHA[_x])
        NET_STATE_SHA256[_x] = state_dict_sha256(NETS[_x])
_V = parse_questions(_path)   # 一組只解析一次，兩臂共用
assert parse_val() is _V, 'QUESTIONS_PATH 轉接錯了：dump_val_p03 讀到的不是這一組的題目檔'
_st = {'keys': write_keys('val_keys', _V['keys'])}
_done_rows = {}
for _x in ARMS:
    _st[_x] = dump_val_p03(NETS[_x], DUMP_NAMES[_x], W_LABEL[_x], WSHA[_x])   # 03 的函式，一字不改
    _done_rows[_x] = {_v: (read_json(os.path.join(DUMP_DIR, '_DONE_val_%s_%s.json' % (DUMP_NAMES[_x], _v))) or {}).get('rows')
                      for _v in ('d4', 's0_swap')}
    if _done_rows[_x] != {'d4': len(_V['feats']), 's0_swap': len(_V['feats'])}:
        raise SystemExit('%s：%s 臂的完成標記列數 %s 不是 %d。請把這段訊息截圖給 Claude。'
                         % (SET, _x.upper(), _done_rows[_x], len(_V['feats'])))
if set(_st.values()) != {'written'}:
    raise SystemExit('%s：有逐盤輸出沒有在這次執行重算（%s；斷線後不接續）。請把這段訊息截圖給 Claude。' % (SET, _st))
SET_LOG[SET] = {'questions_path': QUESTIONS_PATH, 'questions': len(_V['spans']), 'rows': len(_V['feats']),
                'games_unparsed': int((~_V['keys']['parsed']).sum()), 'status': _st, 'done_rows': _done_rows,
                'net_state_sha256': dict(NET_STATE_SHA256), 'seconds': time.time() - t0}
del _V
QSETS.pop(_path, None)   # 解析結果用完就釋放記憶體
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
save_config(dumps=SET_LOG)
save_config_summary(dumps=SET_LOG)
mark_time('dump_%s' % SET, time.time() - t0)
print('%s：%s' % (SET, {k: v for k, v in SET_LOG[SET].items() if k != 'net_state_sha256'}))"""

# ---------------------------------------------------------------------------
# seam check (new)
# ---------------------------------------------------------------------------
SEAM = r"""# 本版新增：notebook 接縫（PREREG_04 rev 3 第 11 節第 5 條）：這次重算的 seam_val 逐盤輸出，和 Drive 上 03 兩臂的比
# 每一臂：key 陣列完全相同；原方向正確執色（d4 的第 0 個對稱）和整題執色對調兩種 logits 的 C0（每題可解析盤的 softmax
# float32 平均，和 test02 同一個量）最大差 ≤ tol（算這份 seam_val dump 的 GPU 和 03 的 dump 相同時 1e-4，不同時 1e-3）；
# 權重相同（12 個完成標記和 npz 依臂）。三組 × 兩臂 × {d4, 對調} 的 12 個完成標記加 3 個 key 的完成標記必須是同一個執行環境。
# 逐位元組是否相同、logits 最大差（8 種對稱和對調）、E（每題 float64 log_softmax 平均）最大差、規則 B 不同題數只記錄
t0 = time.time()
SEAM_KEYS = ('question_id', 'slot', 'color', 'parsed')
SEAM_ENV_KEYS = ('gpu_name', 'torch', 'cudnn')


__T02_SEAM_DEFS__


def e_of(L, spans, parsed):
    # E：每盤 float64 log_softmax、同題平均（合併公式用的量；只記錄）；L 是 [盤數, 10]；沒有可解析的盤是 None
    out = []
    for _q, start, k in spans:
        ok = [start + j for j in range(k) if parsed[start + j]]
        if not ok:
            out.append(None)
            continue
        z = L[ok].astype(np.float64)
        m = z.max(1, keepdims=True)
        out.append((z - (m + np.log(np.exp(z - m).sum(1, keepdims=True)))).mean(0))
    return out


def max_diff(xs, ys):
    # 兩串每題向量（None 跳過）的逐元素最大差
    d = [float(np.max(np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64))))
         for x, y in zip(xs, ys) if x is not None and y is not None]
    return max(d) if d else 0.0


def seam_tol(new_gpu, ref_gpu):
    # 容許差：算這份 seam_val dump 的 GPU（完成標記）和 03 的 dump 的 GPU 相同（而且不是 None；SMOKE 例外）→ 1e-4，否則 1e-3
    same = new_gpu == ref_gpu and (SMOKE or new_gpu is not None)
    return bool(same), (1e-4 if same else 1e-3)


def seam_arm(new, ref, tol):
    # 一臂的接縫。new、ref：{'keys': {四個 key 陣列}, 'd4': [盤數, 8, 10], 'swap': [盤數, 10]}。回傳紀錄；
    # 'passed' 看 key、C0（s0）、C0（對調）（權重另外核對）
    keys_equal = {k: bool(new['keys'][k].shape == ref['keys'][k].shape and np.array_equal(new['keys'][k], ref['keys'][k]))
                  for k in SEAM_KEYS}
    D1, D0, S1, S0 = new['d4'], ref['d4'], new['swap'], ref['swap']
    rec = {'keys_equal': keys_equal, 'tol': tol, 'c0_s0_max_abs_diff': None, 'c0_swap_max_abs_diff': None,
           'bitwise_equal_d4': bool(D1.dtype == D0.dtype and D1.shape == D0.shape and D1.tobytes() == D0.tobytes()),
           'bitwise_equal_swap': bool(S1.dtype == S0.dtype and S1.shape == S0.shape and S1.tobytes() == S0.tobytes()),
           'max_abs_logit_diff_per_sym': None, 'max_abs_logit_diff_swap': None, 'max_abs_e_diff_s0': None,
           'max_abs_e_diff_swap': None, 'rule_b_differences_s0_record_only': None, 'problems': []}
    n = len(new['keys']['parsed'])
    if not all(keys_equal.values()) or D1.shape != D0.shape or S1.shape != S0.shape or D1.ndim != 3 or S1.ndim != 2 \
            or D1.shape[0] != n or S1.shape[0] != n:
        rec['problems'].append('key 陣列或 logits 的形狀和 03 不同：%s，d4 %s／%s，對調 %s／%s'
                               % (keys_equal, D1.shape, D0.shape, S1.shape, S0.shape))
        rec['passed'] = False
        return rec
    parsed = new['keys']['parsed'].astype(bool)
    spans = question_spans(new['keys']['question_id'])
    c1, c0 = c0_of(D1, spans, parsed), c0_of(D0, spans, parsed)
    w1, w0 = c0_of(S1[:, None, :], spans, parsed), c0_of(S0[:, None, :], spans, parsed)
    rec['c0_s0_max_abs_diff'] = max_diff(c1, c0)
    rec['c0_swap_max_abs_diff'] = max_diff(w1, w0)
    rec['max_abs_logit_diff_per_sym'] = [float(np.max(np.abs(D1[parsed, s].astype(np.float64) - D0[parsed, s].astype(np.float64))))
                                         if parsed.any() else 0.0 for s in range(D1.shape[1])]
    rec['max_abs_logit_diff_swap'] = (float(np.max(np.abs(S1[parsed].astype(np.float64) - S0[parsed].astype(np.float64))))
                                      if parsed.any() else 0.0)
    rec['max_abs_e_diff_s0'] = max_diff(e_of(D1[:, 0, :], spans, parsed), e_of(D0[:, 0, :], spans, parsed))
    rec['max_abs_e_diff_swap'] = max_diff(e_of(S1, spans, parsed), e_of(S0, spans, parsed))
    rec['rule_b_differences_s0_record_only'] = sum(1 for x, y in zip(c1, c0) if x is not None and rule_b(x) != rule_b(y))
    rec.update(questions=len(spans), games=int(n), games_parsed=int(parsed.sum()))
    for _name, _v in (('C0（原方向正確執色）', rec['c0_s0_max_abs_diff']), ('C0（整題執色對調）', rec['c0_swap_max_abs_diff'])):
        if not _v <= tol:
            rec['problems'].append('%s 最大差 %s 超過容許 %g' % (_name, _v, tol))
    rec['passed'] = bool(rec['c0_s0_max_abs_diff'] <= tol and rec['c0_swap_max_abs_diff'] <= tol)
    return rec


def dump_env(markers):
    # 完成標記 {名稱: dict} 的執行環境（GPU 型號、torch、cuDNN）；全部都在、而且全部相同才算同一個環境
    env = {n: {k: (m or {}).get(k) for k in SEAM_ENV_KEYS} for n, m in markers.items()}
    vals = {tuple(v[k] for k in SEAM_ENV_KEYS) for v in env.values()}
    return env, bool(markers) and len(vals) == 1 and all(m is not None for m in markers.values())


def weights_match(markers, npz_weights, wsha):
    # 這一臂的完成標記和 npz 記的 weights_sha256 都要等於這一臂的權重；回傳 (是否相符, 不符的名稱)
    bad = sorted([n for n, m in markers.items() if (m or {}).get('weights_sha256') != wsha]
                 + ['%s (npz)' % n for n, w in npz_weights.items() if w != wsha])
    return not bad, bad


_new_dir = os.path.join(RESULT_DIR, 'seam_val', 'dumps')
_new_keys = _load_npz(os.path.join(_new_dir, 'val_keys.npz'))
SEAM_MARKERS = {'%s/%s' % (_s, _stem): read_json(os.path.join(RESULT_DIR, _s, 'dumps', '_DONE_%s.json' % _stem))
                for _s in SETS for _stem in ['val_keys'] + ['val_%s_%s' % (DUMP_NAMES[_x], _v) for _x in ARMS
                                                           for _v in ('d4', 's0_swap')]}
SEAM_ENV, SEAM_ENV_SAME = dump_env(SEAM_MARKERS)
SEAM_ARMS = {}
for _x in ARMS:
    _n = DUMP_NAMES[_x]
    _m_new = SEAM_MARKERS['seam_val/val_%s_d4' % _n] or {}
    _same_gpu, _tol = seam_tol(_m_new.get('gpu_name'), P03_REF_MARKER[_x].get('gpu_name'))
    _new = {'keys': _new_keys, 'd4': _load_npz(os.path.join(_new_dir, 'val_%s_d4.npz' % _n))['logits'],
            'swap': _load_npz(os.path.join(_new_dir, 'val_%s_s0_swap.npz' % _n))['logits']}
    _ref = {'keys': _load_npz(os.path.join(P03_DUMPS[_x], 'val_keys.npz')),
            'd4': _load_npz(os.path.join(P03_DUMPS[_x], 'val_%s_d4.npz' % _n))['logits'],
            'swap': _load_npz(os.path.join(P03_DUMPS[_x], 'val_%s_s0_swap.npz' % _n))['logits']}
    _rec = seam_arm(_new, _ref, _tol)
    del _new, _ref
    _wm = {_k: _v for _k, _v in SEAM_MARKERS.items() if _k.split('/')[1] in ('val_%s_d4' % _n, 'val_%s_s0_swap' % _n)}
    _wn = {}
    for _s in SETS:
        for _v in ('d4', 's0_swap'):
            with np.load(os.path.join(RESULT_DIR, _s, 'dumps', 'val_%s_%s.npz' % (_n, _v)), allow_pickle=False) as _z:
                _wn['%s/val_%s_%s' % (_s, _n, _v)] = str(_z['weights_sha256'])
    _rec['weights_ok'], _rec['weights_mismatch'] = weights_match(_wm, _wn, WSHA[_x])
    if not _rec['weights_ok']:
        _rec['problems'].append('%s 臂的權重不符：%s' % (_x.upper(), _rec['weights_mismatch']))
    _rec.update(same_gpu=_same_gpu, seam_dump_gpu=_m_new.get('gpu_name'), reference_gpu=P03_REF_MARKER[_x].get('gpu_name'),
                seam_dump={k: _m_new.get(k) for k in ('gpu_name', 'torch', 'cudnn', 'run', 'sha256', 'weights_sha256')},
                reference_dump={k: P03_REF_MARKER[_x].get(k) for k in ('gpu_name', 'torch', 'cudnn', 'run', 'sha256',
                                                                        'weights_sha256')},
                file_sha_equal_d4=sha256_path(os.path.join(_new_dir, 'val_%s_d4.npz' % _n)) == SEAM_REF[_x]['val_%s_d4.npz' % _n],
                file_sha_equal_swap=(sha256_path(os.path.join(_new_dir, 'val_%s_s0_swap.npz' % _n))
                                     == SEAM_REF[_x]['val_%s_s0_swap.npz' % _n]))
    _rec['passed'] = bool(_rec['passed'] and _rec['weights_ok'])
    SEAM_ARMS[_x] = _rec
del _new_keys
SEAM = {
    'what': 'test04 notebook seam (PREREG_04 rev 3 section 11 item 5): seam_val dumps of both prereg03 arms vs '
            'the prereg03 val dumps on Drive',
    'passed': bool(all(SEAM_ARMS[_x]['passed'] for _x in ARMS) and SEAM_ENV_SAME),
    'tol': max(SEAM_ARMS[_x]['tol'] for _x in ARMS),
    'tol_rule': '1e-4 when the GPU that wrote the seam_val val_p03x_d4 dump (its _DONE_ marker) is the prereg03 dump GPU '
                '(and not None, except in SMOKE), else 1e-3; per arm, both arms equal when the environment is the same',
    'seam_dump_gpu': SEAM_ARMS['a']['seam_dump_gpu'], 'reference_gpu': P03_REF_GPU,
    'same_gpu': bool(all(SEAM_ARMS[_x]['same_gpu'] for _x in ARMS)),
    'keys_equal': bool(all(all(SEAM_ARMS[_x]['keys_equal'].values()) for _x in ARMS)),
    'weights_ok': bool(all(SEAM_ARMS[_x]['weights_ok'] for _x in ARMS)),
    'env_same': bool(SEAM_ENV_SAME), 'env': SEAM_ENV,
    'env_rule': 'the 12 logits markers (3 sets x 2 arms x {d4, s0_swap}) and the 3 key markers carry the same gpu_name, '
                'torch and cudnn',
    'arms': SEAM_ARMS,
    'bitwise_equal': bool(all(SEAM_ARMS[_x]['bitwise_equal_d4'] and SEAM_ARMS[_x]['bitwise_equal_swap'] for _x in ARMS)),
    'problems': [p for _x in ARMS for p in SEAM_ARMS[_x]['problems']]
                + ([] if SEAM_ENV_SAME else ['12 個 logits 和 3 個 key 的完成標記不是同一個執行環境算的（GPU、torch、cuDNN）']),
    'run': RUN['run'], 'created': now_iso(), 'smoke': SMOKE,
    'gpu_this_run': GPU_MODEL_NOW, 'torch_this_run': torch.__version__, 'cudnn_this_run': ENV_INFO.get('cudnn'),
    'reference_sha256': SEAM_REF, 'reference_env': P03_REF_ENV,
}
_seam_path = os.path.join(RESULT_DIR, 'seam_check.json')
_prev = read_json(_seam_path)
SEAM['earlier_runs'] = ((_prev.get('earlier_runs') or []) + [{k: v for k, v in _prev.items() if k != 'earlier_runs'}]
                        if _prev else [])
write_json(_seam_path, SEAM)
save_config(seam_check={k: SEAM.get(k) for k in ('passed', 'tol', 'bitwise_equal', 'same_gpu', 'keys_equal', 'weights_ok',
                                                 'env_same')})
mark_time('seam_check', time.time() - t0)
for _x in ARMS:
    print('接縫 %s 臂：C0 最大差 原方向 %s、對調 %s（容許 %g），逐位元組相同 d4 %s／對調 %s，規則 B 不同 %s 題（只記錄）→ %s'
          % (_x.upper(), SEAM_ARMS[_x]['c0_s0_max_abs_diff'], SEAM_ARMS[_x]['c0_swap_max_abs_diff'], SEAM_ARMS[_x]['tol'],
             SEAM_ARMS[_x]['bitwise_equal_d4'], SEAM_ARMS[_x]['bitwise_equal_swap'],
             SEAM_ARMS[_x]['rule_b_differences_s0_record_only'], '通過' if SEAM_ARMS[_x]['passed'] else '沒過'))
print('接縫核對：算 dump 的 GPU %s，03 的 GPU %s，同一個執行環境 %s → %s'
      % (SEAM['seam_dump_gpu'], P03_REF_GPU, SEAM_ENV_SAME, '通過' if SEAM['passed'] else '沒過'))
if not SEAM['passed']:
    # 停下前照樣比對只讀目錄的指紋、打包（查原因要用），並把 Drive 的寫入送出
    _after, _ro_changed = record_readonly_after()
    RUN.update(end=now_iso(), status='seam_failed')
    run_log['status'] = 'seam_failed'
    save_config_summary(status='seam_failed')
    save_run_log()
    _zip, _ = pack_results('test04_rank_results.zip')
    flush_drive()
    raise SystemExit('接縫核對沒過%s：這次執行的 04 測試輸出一律不用、不上傳；R20 不受影響。不要自己重跑，下載 %s、截圖給 Claude。'
                     % (('（%s）' % '；'.join(SEAM['problems'][:6])) if SEAM['problems'] else '', _zip))"""

# ---------------------------------------------------------------------------
# final (rewrite of the T02 final_check: two arms)
# ---------------------------------------------------------------------------
FINAL = r"""# 本版改寫的結尾（test02 的結尾改成兩臂）：核對三組的 key 和 4 個 logits 檔（完成標記、SHA-256、大小、形狀、權重、NaN 位置、
# _INPUT.json、題數，而且都是這次執行算的），比對四個只讀目錄前後的指紋，補完 config／timings／run_log，
# 打包 test04_rank_results.zip（不含權重），把 Drive 的寫入送出
t0 = time.time()
DUMP_CHECK, FINAL_PROBLEMS = {}, []
_LOGIT_STEMS = {'val_%s_%s' % (DUMP_NAMES[_x], _v): (_x, _v) for _x in ARMS for _v in ('d4', 's0_swap')}
for SET in SETS:
    _dir = os.path.join(RESULT_DIR, SET, 'dumps')
    _chk, _set_problems = {}, []
    for _stem in ['val_keys'] + list(_LOGIT_STEMS):
        _path, _m = os.path.join(_dir, _stem + '.npz'), read_json(os.path.join(_dir, '_DONE_%s.json' % _stem))
        if _m is None or not os.path.exists(_path):
            _set_problems.append('%s：檔案或完成標記不見了' % _stem)
            continue
        _sha, _size = sha256_path(_path), os.path.getsize(_path)
        if _sha != _m.get('sha256') or _size != _m.get('bytes'):
            _set_problems.append('%s：SHA-256 或大小和完成標記不同' % _stem)
        if _size >= DUMP_MAX_BYTES:
            _set_problems.append('%s：%d bytes，超過 100 MB' % (_stem, _size))
        if _m.get('run') != RUN['run']:
            _set_problems.append('%s：不是這次執行算的（完成標記記第 %s 次；斷線後不接續）' % (_stem, _m.get('run')))
        _chk[_stem] = {'rows': _m.get('rows'), 'syms': _m.get('syms'), 'bytes': _size, 'sha256': _sha,
                       'weights_sha256': _m.get('weights_sha256'), 'gpu_name': _m.get('gpu_name'),
                       'torch': _m.get('torch'), 'cudnn': _m.get('cudnn'), 'run': _m.get('run')}
    if not _set_problems:
        with np.load(os.path.join(_dir, 'val_keys.npz'), allow_pickle=False) as _z:
            _keys = {k: _z[k] for k in _z.files}
        _rows = len(_keys.get('question_id', []))
        if sorted(_keys) != ['color', 'parsed', 'question_id', 'slot'] or not all(len(v) == _rows for v in _keys.values()):
            _set_problems.append('val_keys 的欄位或長度不對：%s' % sorted(_keys))
        else:
            _parsed = _keys['parsed'].astype(bool)
            if _chk['val_keys']['rows'] != _rows:
                _set_problems.append('val_keys：完成標記的列數 %s 不是 %d' % (_chk['val_keys']['rows'], _rows))
            for _stem, (_x, _v) in _LOGIT_STEMS.items():
                with np.load(os.path.join(_dir, _stem + '.npz'), allow_pickle=False) as _z:
                    _L = _z['logits']
                    _shape = (_rows, NUM_SYMS, NUM_CLASSES) if _v == 'd4' else (_rows, NUM_CLASSES)
                    if _L.dtype != np.float32 or _L.shape != _shape:
                        _set_problems.append('%s：logits 是 %s %s，應該是 float32 %s' % (_stem, _L.dtype, _L.shape, _shape))
                        continue
                    if str(_z['weights_sha256']) != WSHA[_x]:
                        _set_problems.append('%s：npz 記的權重不是 %s 臂的' % (_stem, _x.upper()))
                    if _v == 'd4' and int(_z['syms']) != NUM_SYMS:
                        _set_problems.append('%s：syms 是 %s' % (_stem, _z['syms']))
                    if _v == 's0_swap' and str(_z['view']) != 's0_colour_swapped':
                        _set_problems.append('%s：view 是 %s' % (_stem, _z['view']))
                    _nan = np.isnan(_L).reshape(_rows, -1)
                    if not (np.array_equal(_nan.all(axis=1), ~_parsed) and not _nan[_parsed].any()):
                        _set_problems.append('%s：NaN 的列和解析失敗的盤對不上' % _stem)
                if (_chk[_stem]['rows'], _chk[_stem]['syms'], _chk[_stem]['weights_sha256']) != \
                        (_rows, NUM_SYMS if _v == 'd4' else 1, WSHA[_x]):
                    _set_problems.append('%s：完成標記的列數、syms 或權重不對' % _stem)
            if len(set(_keys['question_id'].tolist())) != SET_ROWS[SET]:
                _set_problems.append('%d 題，應該是 %d 題' % (len(set(_keys['question_id'].tolist())), SET_ROWS[SET]))
            DUMP_CHECK[SET] = {'questions': len(set(_keys['question_id'].tolist())), 'games': _rows,
                               'games_unparsed': int((~_parsed).sum())}
        del _keys
    if read_json(os.path.join(_dir, '_INPUT.json')) != SET_INPUT[SET]:
        _set_problems.append('_INPUT.json 和 preflight 記的不同')
    DUMP_CHECK.setdefault(SET, {}).update(files=_chk, input=SET_INPUT[SET], problems=_set_problems)
    FINAL_PROBLEMS += ['%s：%s' % (SET, p) for p in _set_problems]

# 只讀目錄的指紋（結尾）：必須和這次開頭記的完全相同
READONLY_AFTER, _ro_changed = record_readonly_after()
STATUS = 'final_check_failed' if FINAL_PROBLEMS else ('readonly_changed' if _ro_changed else 'completed')
OUTPUTS = {
    'dumps': DUMP_CHECK, 'final_problems': FINAL_PROBLEMS, 'seam_check_passed': SEAM['passed'],
    'seam_bitwise_equal': SEAM['bitwise_equal'], 'readonly_unchanged': not _ro_changed,
    'readonly_changed_files': _ro_changed,
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

# 結果打包：結果目錄裡除了權重和暫存檔以外的檔案，打成 test04/test04_rank_results.zip（zip 內路徑從 sw/ 起）
RESULTS_ZIP, _skipped_big = pack_results('test04_rank_results.zip')
with zipfile.ZipFile(RESULTS_ZIP) as _zf:
    _names = set(_zf.namelist())
_top = os.path.basename(RESULT_DIR)
_need = (['%s/%s' % (_top, n) for n in ('config.json', 'run_log.json', 'env.json', 'timings.json', 'seam_check.json',
                                        'readonly_before.json', 'readonly_after.json')]
         + ['%s/%s/dumps/%s' % (_top, s, n) for s in SETS
            for n in ['val_keys.npz', '_DONE_val_keys.json', '_INPUT.json']
            + ['%s.npz' % _stem for _stem in _LOGIT_STEMS] + ['_DONE_%s.json' % _stem for _stem in _LOGIT_STEMS]])
_missing_in_zip = [n for n in _need if n not in _names]
if (_missing_in_zip or _skipped_big) and STATUS == 'completed':
    STATUS = 'final_check_failed'
    FINAL_PROBLEMS.append('zip 裡缺 %s，或有檔案太大沒有打包 %s' % (_missing_in_zip[:5], _skipped_big))
    RUN.update(status=STATUS)
    run_log['status'] = STATUS
    save_config(outputs=dict(OUTPUTS, final_problems=FINAL_PROBLEMS))
    save_config_summary(status=STATUS)
    save_run_log()
    RESULTS_ZIP, _skipped_big = pack_results('test04_rank_results.zip')
_zip_mb = os.path.getsize(RESULTS_ZIP) / 1e6
flush_drive()
if STATUS != 'completed':
    raise SystemExit('結尾核對沒過（%s）：結果已打包在 %s，但先不要用。處理方式：不要自己重跑，把這段訊息截圖給 Claude。'
                     % ('；'.join(FINAL_PROBLEMS[:6]) if FINAL_PROBLEMS else
                        '只讀目錄在這次執行中被改動：%s' % {k: v[:5] for k, v in _ro_changed.items()}, RESULTS_ZIP))
print('完成。請下載 test04_rank_results.zip 回傳給 Claude（合併、執色檢查、交件檔都在本機做）：%s（%.1f MB）'
      % (RESULTS_ZIP, _zip_mb))"""


# ---------------------------------------------------------------------------
# helpers for the notebook JSON (nbformat v4.5 layout without nbformat)
# ---------------------------------------------------------------------------
def new_cell(kind, source, cell_id, c10):
    lines = source.splitlines(True)
    if kind == "markdown":
        return {"cell_type": "markdown", "id": cell_id, "metadata": {"c10": c10}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "id": cell_id, "metadata": {"c10": c10}, "outputs": [],
            "source": lines}


def notebook_text(nb):
    return json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def load_bases(repo):
    t02_path, p03_path, p03b_path = (os.path.join(repo, *r.split("/")) for r in (T02_REL, P03_REL, P03B_REL))
    for path, want in ((t02_path, T02_SHA256), (p03_path, P03_SHA256), (p03b_path, P03B_SHA256)):
        with open(path, "rb") as f:
            raw = f.read()
        assert b"\r\n" not in raw and sha_bytes(raw) == want, ("base notebook changed", path, sha_bytes(raw))
    # the prereg03 A and B notebooks differ in the ARM line only (the check of build_prereg03_rank_nb.py)
    with open(p03_path, encoding="utf-8") as f:
        la = f.read().split("\n")
    with open(p03b_path, encoding="utf-8") as f:
        lb = f.read().split("\n")
    diff = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    assert len(la) == len(lb) and len(diff) == 1, diff
    assert "ARM = 'A'" in la[diff[0]] and la[diff[0]].replace("ARM = 'A'", "ARM = 'B'") == lb[diff[0]]
    bases = {}
    for tag, path in (("test02", t02_path), ("prereg03", p03_path)):
        nb = read_json_file(path)
        by_key = {}
        for j, c in enumerate(nb["cells"]):
            k = cell_key(c)
            assert k not in by_key, (tag, k)
            by_key[k] = j
        bases[tag] = {"nb": nb, "by_key": by_key, "rel": T02_REL if tag == "test02" else P03_REL,
                      "src": {k: src_of(nb["cells"][j]) for k, j in by_key.items()}}
    return bases


def build(repo):
    """Assemble the notebook. Returns a dict with the notebook text, the diff text and what was checked."""
    pins = derive_pins(repo)
    bases = load_bases(repo)
    t02, p03 = bases["test02"]["src"], bases["prereg03"]["src"]
    p03_dt_defs = func_segments(p03["dump_tools"])
    p03_off14_defs = func_segments(p03["official_14"])
    t02_pre_defs = func_segments(t02["preflight_t02"])
    t02_seam_defs = func_segments(t02["seam_check"])

    preflight = PREFLIGHT
    for key, val in (("__W_PIN__", pins["W_PIN"]), ("__W_LABEL_PIN__", pins["W_LABEL_PIN"]),
                     ("__SEAM_REF__", pins["SEAM_REF_SHA256_PIN"]), ("__REF_ENV__", pins["P03_REF_ENV_PIN"]),
                     ("__BUNDLE__", pins["BUNDLE_FILES_PIN"])):
        preflight = rep(preflight, key, json.dumps(val, indent=4))
    preflight = rep(preflight, "__TF32__", "{'cudnn': %r, 'matmul': %r}" % (pins["TF32_P03_PIN"]["cudnn"],
                                                                            pins["TF32_P03_PIN"]["matmul"]))
    preflight = rep(preflight, "__QUICK_CHECK__", t02_pre_defs["quick_check_questions"])
    model_tools = rep(MODEL_TOOLS, "__MODEL_DEFS__", "\n\n\n".join(p03_off14_defs[n] for n in P03_MODEL_SEGMENTS))
    seam = rep(SEAM, "__T02_SEAM_DEFS__", "\n\n\n".join(t02_seam_defs[n] for n in T02_SEAM_SEGMENTS))

    plan = [
        ("md_new", "header", None, MD_HEADER),
        ("base", "setup", "test02", build_setup(t02)),
        ("base", "prepare", "test02", t02["prepare"]),
        ("base", "official_1", "test02", t02["official_1"]),
        ("base", "environment", "test02", t02["environment"]),
        ("base", "official_2", "test02", build_official_2(t02)),
        ("code_new", "preflight_t04", None, preflight),
        ("base", "view_d4", "prereg03", p03["view_d4"]),
        ("code_new", "model_tools", None, model_tools),
        ("base", "dump_tools", "test02", build_dump_tools(t02, p03_dt_defs)),
    ] + [("code_new", "dump_set_" + s, None, SET_FIRST_LINE % s + "\n" + SET_BODY) for s in SETS] + [
        ("code_new", "seam_check", None, seam),
        ("rewrite", "final_check", "test02", FINAL),
    ]

    base_t02 = bases["test02"]["nb"]
    nb = {"cells": [], "metadata": json.loads(json.dumps(base_t02["metadata"])), "nbformat": 4, "nbformat_minor": 5}
    used = set()
    for k, (kind, key, bname, src) in enumerate(plan):
        src = src.rstrip("\n")
        if kind != "md_new":
            compile(src, "<cell %d>" % k, "exec")   # plain Python only (the smoke runner compiles whole cells too)
        cell_id = "c10-rank-t04-%02d" % k
        if kind in ("base", "rewrite"):
            assert (bname, key) not in used, key
            used.add((bname, key))
            j = bases[bname]["by_key"][key]
            b = bases[bname]["nb"]["cells"][j]
            assert b["cell_type"] == "code", key
            c10 = {"base_nb": bname, "base_cell": j, "base_id": b["id"], "base": b["metadata"]["c10"],
                   "changed": src != src_of(b)}
            if kind == "rewrite":
                c10["rewritten"] = True
                assert c10["changed"], key
            nb["cells"].append(new_cell("code", src, cell_id, c10))
        else:
            nb["cells"].append(new_cell("markdown" if kind == "md_new" else "code", src, cell_id, {"new": key}))

    # notebook_sha256 (a notebook cannot hold the SHA-256 of its own file): see HASH_LINE and NOTEBOOK_SHA256_RULE
    sources = [src_of(c) for c in nb["cells"]]
    hash_cells = [i for i, s in enumerate(sources) if (HASH_LINE % HASH_PLACEHOLDER) in s]
    assert len(hash_cells) == 1
    nb_sha = nb_source_sha256(sources)
    i = hash_cells[0]
    nb["cells"][i]["source"] = rep(sources[i], HASH_LINE % HASH_PLACEHOLDER, HASH_LINE % nb_sha).splitlines(True)
    sources = [src_of(c) for c in nb["cells"]]
    assert nb_sha == nb_source_sha256([s.replace(HASH_LINE % nb_sha, HASH_LINE % HASH_PLACEHOLDER) for s in sources])
    assert sum(s.count("NOTEBOOK_SHA256 = '") for s in sources) == 1

    segments = check_notebook(nb, bases)
    text = notebook_text(nb)
    return {"nb": nb, "text": text, "nb_sha": nb_sha, "segments": segments, "bases": bases, "used": used,
            "pins": pins}


def check_notebook(nb, bases):
    """The build-time integrity checks (spec 4.1). Returns the list of verified segments for the diff header."""
    cells = nb["cells"]
    t02, p03 = bases["test02"]["src"], bases["prereg03"]["src"]

    def out(key):
        hits = [src_of(c) for c in cells if t04_key(c) == key]
        assert len(hits) == 1, (key, len(hits))
        return hits[0]

    # changed base cells <-> CHANGE_ROWS; unchanged base cells verbatim; ids unique
    changed = {(c["metadata"]["c10"]["base_nb"], t04_key(c)) for c in cells
               if "base_cell" in c["metadata"]["c10"] and c["metadata"]["c10"]["changed"]}
    named = {k for k, _l, _t in CHANGE_ROWS if k is not None}
    assert changed == named, ("changed but not named", changed - named, "named but unchanged", named - changed)
    for c in cells:
        m = c["metadata"]["c10"]
        if "base_cell" in m and not m["changed"]:
            assert src_of(c) == src_of(bases[m["base_nb"]]["nb"]["cells"][m["base_cell"]]), m
        assert c.get("outputs", []) == []
    assert len({c["id"] for c in cells}) == len(cells)

    # the forbidden token: exactly the inherited, unchanged refusal line of the prepare cell, nowhere else
    prep = t02["prepare"]
    assert prep.count(FORBIDDEN_TOKEN) == 1 and ("'%s' in p for p in _all_files" % FORBIDDEN_TOKEN) in prep
    hits = [(i, ln) for i, c in enumerate(cells) for ln in src_of(c).splitlines() if FORBIDDEN_TOKEN in ln.lower()]
    assert len(hits) == 1 and hits[0][1].strip().startswith("assert not any(") and hits[0][1] in prep.splitlines(), hits
    assert t04_key(cells[hits[0][0]]) == "prepare" and not cells[hits[0][0]]["metadata"]["c10"]["changed"]
    assert notebook_text(nb).lower().count(FORBIDDEN_TOKEN) == 1

    # segment identity
    segs = []
    new_setup, new_dt = func_segments(out("setup")), func_segments(out("dump_tools"))
    new_pre, new_seam, new_mt = func_segments(out("preflight_t04")), func_segments(out("seam_check")), \
        func_segments(out("model_tools"))
    checks = [("prereg03 dump_tools", func_segments(p03["dump_tools"]), new_dt, P03_DUMP_TOOLS_SEGMENTS, True),
              ("prereg03 official_14", func_segments(p03["official_14"]), new_mt, P03_MODEL_SEGMENTS, True),
              ("test02 setup", func_segments(t02["setup"]), new_setup, T02_SETUP_SEGMENTS, True),
              ("test02 setup", func_segments(t02["setup"]), new_setup, T02_SETUP_CODE_SEGMENTS, False),
              ("test02 dump_tools", func_segments(t02["dump_tools"]), new_dt, T02_DUMP_TOOLS_SEGMENTS, True),
              ("test02 dump_tools", func_segments(t02["dump_tools"]), new_dt, T02_DUMP_TOOLS_CODE_SEGMENTS, False),
              ("test02 preflight_t02", func_segments(t02["preflight_t02"]), new_pre, T02_PREFLIGHT_SEGMENTS, True),
              ("test02 seam_check", func_segments(t02["seam_check"]), new_seam, T02_SEAM_SEGMENTS, True)]
    for where, base_defs, new_defs, names, exact in checks:
        for name in names:
            a, b = base_defs[name], new_defs[name]
            if exact:
                assert a and a == b, ("segment differs from the base", where, name)
            else:
                assert a and code_tokens(a) == code_tokens(b), ("code differs from the base", where, name)
            segs.append("%s（%s%s）" % (name, where, "" if exact else "，只有註解不同"))
    assert out("view_d4") == p03["view_d4"]
    segs.append("view_d4 整格（prereg03）")
    assert out("prepare") == t02["prepare"]
    segs.append("prepare 整格（test02）")
    # the functions this version dropped are gone; the kept ones are where the spec puts them
    assert "d4" in func_segments(out("view_d4")) and "d4" not in new_dt and "dump_questions" not in new_dt
    assert "parse_val" in new_dt and "dump_val_p03" in new_dt and "parse_questions" in new_dt
    assert "for _qid, start, k in V['spans']:" in new_dt["dump_val_p03"]

    # the three set cells differ only in their first line
    set_cells = [src_of(c) for c in cells if c["metadata"]["c10"].get("new", "").startswith("dump_set_")]
    assert len(set_cells) == 3 and len({s.split("\n", 1)[1] for s in set_cells}) == 1
    assert [s.split("\n", 1)[0] for s in set_cells] == [SET_FIRST_LINE % s for s in SETS]
    assert "assert parse_val() is _V" in set_cells[0] and "'questions_path': QUESTIONS_PATH" in set_cells[0]

    # no training, no autocast; bfloat16 only inside the verbatim state_dict_sha256 (hashing a dtype, not inference)
    code = [src_of(c) for c in cells if c["cell_type"] == "code"]
    attrs = [(n.value.id if isinstance(n.value, ast.Name) else None, n.attr)
             for s in code for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Attribute)]
    assert not [a for a in attrs if a[1] == "autocast" or a == ("tarfile", "open")], "autocast / tarfile.open in code"
    assert [a for a in attrs if a[1] == "bfloat16"] == [("torch", "bfloat16")]
    assert ast.dump(ast.parse(new_mt["state_dict_sha256"])).count("attr='bfloat16'") == 1
    bound, loaded = set(dir(builtins)) | {"drive"}, set()
    for s in code:
        tree = ast.parse(s)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert not names & GONE_NAMES, names & GONE_NAMES
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                (bound if isinstance(n.ctx, (ast.Store, ast.Del)) else loaded).add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                bound.add(n.name)
            elif isinstance(n, ast.arg):
                bound.add(n.arg)
            elif isinstance(n, ast.alias):
                bound.add((n.asname or n.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                bound.add(n.name)
    assert not loaded - bound, sorted(loaded - bound)
    return segs


def diff_text(built, out_nb_sha):
    nb, bases = built["nb"], built["bases"]
    out = []
    out.append("# test04 rank：colab/test04_rank.ipynb 和兩個底稿 notebook 的逐格 unified diff（程式格、markdown 格、notebook metadata）")
    out.append("# 底稿：%s  sha256=%s" % (T02_REL, T02_SHA256))
    out.append("# 底稿：%s  sha256=%s（%s 只差 ARM 那一行，產生器核對過）" % (P03_REL, P03_SHA256, P03B_REL))
    out.append("# 本版：%s  sha256=%s  notebook_sha256（config.json 的規則）=%s" % (OUT_REL, out_nb_sha, built["nb_sha"]))
    out.append("# 產生方式：colab/build_test04_rank_nb.py；difflib.unified_diff，沿用的格依 cell metadata c10（base_nb、base_cell）"
               "對它自己的底稿格；新增的格對空白比，段落照 test02 的新增格另附對照 diff。cell 編號從 0 起算。")
    out.append("# 每一處改動的說明見本版第 0 格「和底稿 notebook 的差異」；沒列在表上的沿用格和底稿逐字相同。")
    out.append("# 產生器核對過和底稿逐字相同的片段：" + "、".join(built["segments"]))
    modeled_on = {"preflight_t04": "preflight_t02", "dump_set_seam_val": "dump_set_seam_val",
                  "dump_set_public": "dump_set_public", "dump_set_private": "dump_set_private",
                  "seam_check": "seam_check"}
    summary, body = [], []
    for k, cell in enumerate(nb["cells"]):
        meta = cell["metadata"]["c10"]
        kind = "程式格" if cell["cell_type"] == "code" else "markdown"
        new_src = src_of(cell)
        extra = None
        if "base_cell" in meta:
            bnb = bases[meta["base_nb"]]
            j = meta["base_cell"]
            old_src = src_of(bnb["nb"]["cells"][j])
            a = "%s cell %d（%s）" % (os.path.basename(bnb["rel"]), j, t04_key(cell))
            status = "改寫" if meta.get("rewritten") else ("改動" if meta["changed"] else "完全相同")
        else:
            old_src = ""
            a = "/dev/null（新增：%s）" % meta["new"]
            status = "新增"
            if meta["new"] in modeled_on:
                tb = bases["test02"]
                j = tb["by_key"][modeled_on[meta["new"]]]
                extra = (src_of(tb["nb"]["cells"][j]),
                         "%s cell %d（%s；段落對照，不是底稿）" % (os.path.basename(tb["rel"]), j, modeled_on[meta["new"]]))
        b = "test04_rank.ipynb cell %d" % k
        summary.append("#   本版 cell %2d（%s）← %s：%s" % (k, kind, a, status))
        diff = list(difflib.unified_diff(old_src.splitlines(), new_src.splitlines(), fromfile=a, tofile=b, lineterm=""))
        body.append("")
        body.append("=" * 100)
        body.append("本版 cell %d（%s）↔ %s（%s）" % (k, kind, a, status))
        body.append("=" * 100)
        body.extend(diff if diff else ["（完全相同）"])
        if extra:
            body.append("-" * 100)
            body.append("對照：%s" % extra[1])
            body.append("-" * 100)
            body.extend(list(difflib.unified_diff(extra[0].splitlines(), new_src.splitlines(), fromfile=extra[1],
                                                  tofile=b, lineterm="")) or ["（完全相同）"])
    out.append("# 對照表：")
    out.extend(summary)
    for tag in ("test02", "prereg03"):
        bnb = bases[tag]
        dropped = ["#   %s cell %2d（%s，%s）" % (os.path.basename(bnb["rel"]), j,
                                               "程式格" if c["cell_type"] == "code" else "markdown", cell_key(c))
                   for j, c in enumerate(bnb["nb"]["cells"]) if (tag, cell_key(c)) not in built["used"]]
        out.append("# 沒有沿用的 %s 格（%d 格）：" % (os.path.basename(bnb["rel"]), len(dropped)))
        out.extend(dropped)
    out.extend(body)
    meta_old = json.dumps(bases["test02"]["nb"]["metadata"], ensure_ascii=False, indent=1, sort_keys=True).splitlines()
    meta_new = json.dumps(nb["metadata"], ensure_ascii=False, indent=1, sort_keys=True).splitlines()
    out.append("")
    out.append("=" * 100)
    out.append("notebook metadata（對 test02）")
    out.append("=" * 100)
    out.extend(list(difflib.unified_diff(meta_old, meta_new, fromfile="test02 metadata", tofile="本版 metadata",
                                         lineterm="")) or ["（完全相同）"])
    text = "\n".join(out) + "\n"
    assert text.lower().count(FORBIDDEN_TOKEN) == 0
    return text


def write_bytes(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", nargs="?", default=os.path.dirname(HERE))
    ap.add_argument("--out-nb", help="default: <repo>/colab/test04_rank.ipynb")
    ap.add_argument("--out-diff", help="default: <repo>/results/test04/notebook_diff_rank.txt")
    ap.add_argument("--upload-copy", action="store_true",
                    help="also write the byte copy <repo>/data/colab_upload/test04/test04_rank.ipynb")
    args = ap.parse_args(argv)
    repo = os.path.abspath(args.repo)
    out_nb = args.out_nb or os.path.join(repo, *OUT_REL.split("/"))
    out_diff = args.out_diff or os.path.join(repo, *DIFF_REL.split("/"))
    built = build(repo)
    nb_bytes = built["text"].encode("utf-8")
    write_bytes(out_nb, nb_bytes)
    diff_bytes = diff_text(built, sha_bytes(nb_bytes)).encode("utf-8")
    write_bytes(out_diff, diff_bytes)
    print("wrote", out_nb, sha_bytes(nb_bytes))
    print("wrote", out_diff, sha_bytes(diff_bytes))
    if args.upload_copy:
        up = os.path.join(repo, *UPLOAD_REL.split("/"))
        write_bytes(up, nb_bytes)
        others = sorted(set(os.listdir(os.path.dirname(up))) - {os.path.basename(up)})
        assert not others, ("the upload folder must hold only test04_rank.ipynb", others)
        print("wrote", up, sha(up))
    print("cells", len(built["nb"]["cells"]), "| notebook_sha256", built["nb_sha"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
