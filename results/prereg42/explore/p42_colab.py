"""prereg42: the Colab kit and notebooks (PLAN.md 3). Standard library only.

    python results/prereg42/explore/p42_colab.py kit           # -> data/prereg42/colab/C10_p42_kit.zip + kit_manifest.json
    python results/prereg42/explore/p42_colab.py notebooks     # -> results/prereg42/colab/p42_<V>.ipynb (kit SHA embedded)
    python results/prereg42/explore/p42_colab.py notebooks --check     # exit 1 if the committed notebooks differ
    python results/prereg42/explore/p42_colab.py v3b [--check]         # V3b notebook (PLAN.md revision 3), see below

V3b (PLAN.md revision 3) reuses the kit unchanged (zip SHA in kit_manifest.json): its notebook is the committed
p42_V3.ipynb with the variant renamed and one extra cell after the kit cell that writes the new p42_lib.py,
p42_train.py, p42_embed.py (embedded in the notebook, base64) over the kit's copies and checks their SHA-256 against
results/prereg42/colab/overlay_manifest.json. Nothing besides the notebook has to go to Drive.

Kit = the code the Colab run needs, at its repo paths (tools/c10data.py, results/prereg36/explore/p36_lib.py,
p36_train.py, p36_embed.py, split_record.json, results/prereg42/explore/p42_lib.py, p42_train.py, p42_embed.py) and
the data (data/prereg36/cache/{codes.npy, off.npy, meta.npz, cache.json} = the move cache of the official training
games of the fit players; data/prereg06/feats/meta.npz = the player levels the prereg36 split needs). No validation
answers, no competition-2 data, no checkpoints. The manifest (results/prereg42/colab/kit_manifest.json, committed)
lists the SHA-256 of every file; the notebook checks the zip and every file before it runs.

Drive layout (the account that runs Colab; the main session uploads): MyDrive/C10/p42/C10_p42_kit.zip and the
notebook p42_<V>.ipynb. The run writes MyDrive/C10/p42/<V>/ (checkpoints every 8,000 steps, logs, train record),
MyDrive/C10/p42/emb/<tag>/store_D.npz, and at the end MyDrive/C10/p42/C10_p42_<V>_results.zip (final checkpoint,
train record, logs, D vectors) for the download. "Run all" again after a disconnect resumes (--auto-resume;
finished D vectors are skipped). Runtime: A100 (or any CUDA GPU); the preinstalled torch is used and recorded.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
KIT_DIR = os.path.join(REPO, "data", "prereg42", "colab")
KIT_ZIP = os.path.join(KIT_DIR, "C10_p42_kit.zip")
NB_DIR = os.path.join(REPO, "results", "prereg42", "colab")
MANIFEST = os.path.join(NB_DIR, "kit_manifest.json")
DRIVE = "/content/drive/MyDrive/C10/p42"
WORK = "/content/c10"
FILES = ["tools/c10data.py",
         "results/prereg36/explore/p36_lib.py", "results/prereg36/explore/p36_train.py",
         "results/prereg36/explore/p36_embed.py", "results/prereg36/explore/split_record.json",
         "results/prereg42/explore/p42_lib.py", "results/prereg42/explore/p42_train.py",
         "results/prereg42/explore/p42_embed.py",
         "data/prereg36/cache/cache.json", "data/prereg36/cache/codes.npy", "data/prereg36/cache/off.npy",
         "data/prereg36/cache/meta.npz", "data/prereg06/feats/meta.npz"]
VARIANTS = ("V1", "V2", "V3", "V4")
TX = ("V3",)
OVERLAY = ["results/prereg42/explore/p42_lib.py", "results/prereg42/explore/p42_train.py",
           "results/prereg42/explore/p42_embed.py"]
OV_MANIFEST = os.path.join(NB_DIR, "overlay_manifest.json")


def sha(path, lf=False):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if lf:
            h.update(f.read().replace(b"\r\n", b"\n"))
        else:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    return h.hexdigest()


def is_text(rel):
    return rel.endswith((".py", ".json"))


def cmd_kit(_a):
    os.makedirs(KIT_DIR, exist_ok=True)
    os.makedirs(NB_DIR, exist_ok=True)
    files = {}
    tmp = KIT_ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for rel in FILES:
            src = os.path.join(REPO, rel)
            if is_text(rel):
                with open(src, "rb") as f:
                    data = f.read().replace(b"\r\n", b"\n")
                info = zipfile.ZipInfo(rel, date_time=(2026, 11, 12, 0, 0, 0))
                z.writestr(info, data)
                files[rel] = hashlib.sha256(data).hexdigest()
            else:
                info = zipfile.ZipInfo(rel, date_time=(2026, 11, 12, 0, 0, 0))
                with open(src, "rb") as f, z.open(info, "w", force_zip64=True) as w:
                    for b in iter(lambda: f.read(1 << 20), b""):
                        w.write(b)
                files[rel] = sha(src)
    os.replace(tmp, KIT_ZIP)
    man = {"format": "c10-prereg42 colab kit v1", "zip": os.path.basename(KIT_ZIP), "zip_sha256": sha(KIT_ZIP),
           "zip_bytes": os.path.getsize(KIT_ZIP), "files": files}
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    print(json.dumps({k: man[k] for k in ("zip", "zip_sha256", "zip_bytes")}, indent=1))
    return 0


MD = """# C10 prereg42：訓練配方短跑 __V__（Colab）

**不用改任何程式**：執行階段選 A100（或其他 GPU），按「執行階段 → 全部執行」。

1. 掛上雲端硬碟，核對 `我的雲端硬碟/C10/p42/C10_p42_kit.zip` 的 SHA-256，解開並逐檔核對。
2. 訓練 __V__ 40,000 步（約 3 小時；每 8,000 步存到 `C10/p42/__V__/`；斷線後重按「全部執行」會接著跑）。
3. 算 D 玩家的向量（raw、EMA）存到 `C10/p42/emb/`。
4. 打包 `C10/p42/C10_p42___V___results.zip`，下載回傳。

任何一格出現紅字或 `STOP`，停下來，截最後 20 行。"""

CELL_SETUP = """# 1. 雲端硬碟、GPU、套件（不用改）
import hashlib, json, os, subprocess, sys, zipfile
from google.colab import drive
drive.mount('/content/drive')
DRIVE, WORK, VARIANT = __DRIVE__, __WORK__, __VARIANT__
KIT_SHA, FILES = __KIT_SHA__, __FILES__
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
print(subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout)
tv = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__, torch.cuda.is_available())'], capture_output=True, text=True).stdout.strip()
print('torch', tv)   # in a subprocess: the notebook itself never holds the GPU (p42_train refuses to share it)
assert tv.endswith('True'), 'STOP: 沒有 GPU，請在「變更執行階段類型」選 GPU'
kit = os.path.join(DRIVE, 'C10_p42_kit.zip')
assert os.path.exists(kit), 'STOP: 找不到 ' + kit
ok_flag = os.path.join(WORK, '.kit_ok')
if not (os.path.exists(ok_flag) and open(ok_flag).read() == KIT_SHA):
    assert sha(kit) == KIT_SHA, 'STOP: 套件檔的 SHA-256 不對'
    with zipfile.ZipFile(kit) as z:
        z.extractall(WORK)
    for rel, h in FILES.items():
        assert sha(os.path.join(WORK, rel)) == h, 'STOP: 檔案不對 ' + rel
    open(ok_flag, 'w').write(KIT_SHA)
print('kit ok', len(FILES), 'files')
os.makedirs(os.path.join(DRIVE, VARIANT), exist_ok=True)"""

CELL_TRAIN = """# 2. 訓練（不用改；斷線後重按「全部執行」會從最後的存檔接著跑）
out = os.path.join(DRIVE, VARIANT)
rec = os.path.join(out, 'train_record_%s.json' % VARIANT)
workers = str(max(2, min(12, (os.cpu_count() or 4) - 1)))
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
def run(args, log):
    with open(log, 'a') as lf:
        p = subprocess.Popen([sys.executable, '-u'] + args, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end='')
            lf.write(line); lf.flush()
        return p.wait()
done = os.path.exists(rec) and json.load(open(rec)).get('finished')
if not done:
    if not any(f.startswith('step_') for f in os.listdir(os.path.join(out, 'ckpt'))) if os.path.isdir(os.path.join(out, 'ckpt')) else True:
        rc = run(['results/prereg42/explore/p42_train.py', '--variant', VARIANT, '--workers', workers, '--bench', '300', '--out', out], os.path.join(out, 'bench_log.txt'))
        print('bench exit', rc)
    rc = run(['results/prereg42/explore/p42_train.py', '--variant', VARIANT, '--workers', workers, '--auto-resume', '--out', out, '--record', rec, '--min-free-ram-mb', '2048'], os.path.join(out, 'train_log.txt'))
    assert rc == 0, 'STOP: 訓練結束碼 %d（4＝記憶體不足或 STOP 檔，重按「全部執行」會接續）' % rc
print('training finished')"""

CELL_EMBED = """# 3. D 玩家的向量（不用改）
ck = os.path.join(out, 'ckpt', 'step_40000.pt')
emb_root = os.path.join(DRIVE, 'emb')
jobs = __JOBS__
for args in jobs:
    rc = run(['results/prereg42/explore/p42_embed.py', '--ckpt', ck, '--players', 'D', '--workers', workers, '--emb-root', emb_root] + args, os.path.join(out, 'embed_log.txt'))
    assert rc == 0, 'STOP: 向量結束碼 %d' % rc
print('embeddings finished')"""

CELL_ZIP = """# 4. 打包回傳（不用改）
res = os.path.join(DRIVE, 'C10_p42_%s_results.zip' % VARIANT)
with zipfile.ZipFile(res + '.tmp', 'w', zipfile.ZIP_STORED) as z:
    for name in ('train_record_%s.json' % VARIANT, 'train_log.txt', 'train_log.jsonl', 'bench_log.txt', 'embed_log.txt'):
        p = os.path.join(out, name)
        if os.path.exists(p):
            z.write(p, '%s/%s' % (VARIANT, name))
    z.write(ck, '%s/ckpt/step_40000.pt' % VARIANT)
    for d in sorted(os.listdir(emb_root)):
        if d.startswith(VARIANT + '_'):
            z.write(os.path.join(emb_root, d, 'store_D.npz'), 'emb/%s/store_D.npz' % d)
os.replace(res + '.tmp', res)
drive.flush_and_unmount()
print('完成。請下載 Drive 的 C10/p42/' + os.path.basename(res) + ' 回傳給 Claude（%.0f MB）' % (os.path.getsize(res) / 1e6) if os.path.exists(res) else '完成')"""


def lines(src):
    out = src.split("\n")
    return [x + "\n" for x in out[:-1]] + [out[-1]]


def notebook(variant, man):
    jobs = [["--weights", w] + (["--mode", m] if variant in TX else []) for w in ("raw", "ema")
            for m in (("full", "chunk24") if variant in TX else (None,))]
    jobs = [[x for x in j if x is not None] for j in jobs]
    rep = {"__DRIVE__": repr(DRIVE), "__WORK__": repr(WORK), "__VARIANT__": repr(variant),
           "__KIT_SHA__": repr(man["zip_sha256"]), "__FILES__": json.dumps(man["files"], sort_keys=True),
           "__JOBS__": json.dumps(jobs)}
    cells = []
    for i, (kind, src) in enumerate((("markdown", MD.replace("__V__", variant)), ("code", CELL_SETUP),
                                     ("code", CELL_TRAIN), ("code", CELL_EMBED), ("code", CELL_ZIP))):
        for k, v in rep.items():
            src = src.replace(k, v)
        c = {"cell_type": kind, "id": "p42-%s-%d" % (variant.lower(), i), "metadata": {}, "source": lines(src)}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        cells.append(c)
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def cmd_notebooks(a):
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    bad = 0
    for v in VARIANTS:
        path = os.path.join(NB_DIR, "p42_%s.ipynb" % v)
        text = notebook(v, man)
        if a.check:
            with open(path, encoding="utf-8", newline="") as f:
                if f.read().replace("\r\n", "\n") != text:
                    print("differs: %s" % path)
                    bad = 1
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote %s" % path)
    return bad


CELL_OVERLAY = """# 1b. V3b 的新版程式（PLAN 修訂 3）：覆蓋套件裡的 p42_lib.py、p42_train.py、p42_embed.py，逐檔核雜湊（不用改）
import base64
OVERLAY = __OVERLAY__
for rel, (h, b64) in OVERLAY.items():
    data = base64.b64decode(b64)
    assert hashlib.sha256(data).hexdigest() == h, 'STOP: 內嵌的檔案不對 ' + rel
    with open(os.path.join(WORK, rel), 'wb') as f:
        f.write(data)
    assert sha(os.path.join(WORK, rel)) == h, 'STOP: 檔案不對 ' + rel
print('overlay ok', len(OVERLAY), 'files')"""


def v3b_notebook():
    """(notebook text, overlay manifest) for V3b: p42_V3.ipynb with VARIANT 'V3b' + the overlay cell."""
    import base64
    files, emb = {}, {}
    for rel in OVERLAY:
        with open(os.path.join(REPO, rel), "rb") as f:
            data = f.read().replace(b"\r\n", b"\n")
        h = hashlib.sha256(data).hexdigest()
        files[rel] = h
        emb[rel] = [h, base64.b64encode(data).decode("ascii")]
    with open(MANIFEST, encoding="utf-8") as f:
        kit = json.load(f)
    man = {"format": "c10-prereg42 colab overlay v1", "variant": "V3b", "plan": "PLAN.md revision 3",
           "kit_zip_sha256": kit["zip_sha256"], "files": files,
           "replaces_kit_files": {rel: kit["files"][rel] for rel in OVERLAY}}
    with open(os.path.join(NB_DIR, "p42_V3.ipynb"), encoding="utf-8") as f:
        nb = json.load(f)
    cells = []
    for c in nb["cells"]:
        c = json.loads(json.dumps(c))
        src = "".join(c["source"])
        if (c.get("id") or "").startswith("p42-v3-"):
            c["id"] = c["id"].replace("p42-v3-", "p42-v3b-")
        if c["cell_type"] == "markdown":
            src = src.replace("訓練配方短跑 V3（Colab）", "訓練配方短跑 V3b（Colab；PLAN 修訂 3）")
            src = src.replace("訓練 V3 40,000 步", "（1b 格覆蓋 3 個新版程式檔）訓練 V3b 40,000 步")
            src = src.replace("`C10/p42/V3/`", "`C10/p42/V3b/`").replace("C10_p42_V3_results.zip", "C10_p42_V3b_results.zip")
        else:
            n = src.count("VARIANT = '/content/drive/MyDrive/C10/p42', '/content/c10', 'V3'")
            src = src.replace("VARIANT = '/content/drive/MyDrive/C10/p42', '/content/c10', 'V3'",
                              "VARIANT = '/content/drive/MyDrive/C10/p42', '/content/c10', 'V3b'")
            assert "'V3'" not in src, "V3 left in a cell"
        c["source"] = lines(src)
        cells.append(c)
        if c.get("id") == "p42-v3b-1":
            assert n == 1
            ov = CELL_OVERLAY.replace("__OVERLAY__", json.dumps(emb, sort_keys=True))
            cells.append({"cell_type": "code", "id": "p42-v3b-1b", "metadata": {}, "source": lines(ov),
                          "execution_count": None, "outputs": []})
    assert [c.get("id") for c in cells].count("p42-v3b-1b") == 1
    for c in cells:
        assert "V3_" not in "".join(c["source"]) and "/V3/" not in "".join(c["source"])
    nb["cells"] = cells
    text = json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    return text, json.dumps(man, indent=1, sort_keys=True) + "\n"


def cmd_v3b(a):
    text, man = v3b_notebook()
    path = os.path.join(NB_DIR, "p42_V3b.ipynb")
    if a.check:
        bad = 0
        for p_, want in ((path, text), (OV_MANIFEST, man)):
            with open(p_, encoding="utf-8", newline="") as f:
                if f.read().replace("\r\n", "\n") != want:
                    print("differs: %s" % p_)
                    bad = 1
        return bad
    for p_, want in ((path, text), (OV_MANIFEST, man)):
        with open(p_, "w", encoding="utf-8", newline="\n") as f:
            f.write(want)
        print("wrote %s" % p_)
    print(man)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("kit")
    n = sub.add_parser("notebooks")
    n.add_argument("--check", action="store_true")
    b = sub.add_parser("v3b")
    b.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    return {"kit": cmd_kit, "notebooks": cmd_notebooks, "v3b": cmd_v3b}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
