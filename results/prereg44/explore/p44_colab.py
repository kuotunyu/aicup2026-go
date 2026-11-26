"""prereg44: the Colab kit, its Drive parts and the notebook (PLAN.md 3.3, 3.4). Standard library only.

    python results/prereg44/explore/p44_colab.py kit        # data/prereg44/colab/C10_p44_kit.zip + results/prereg44/colab/kit_manifest.json
    python results/prereg44/explore/p44_colab.py parts      # data/prereg44/colab/upload/kit_parts/kit.partNN (7,000,000 bytes)
                                                            #   + results/prereg44/colab/kit_parts_sha.txt
    python results/prereg44/explore/p44_colab.py notebook [--check]   # results/prereg44/colab/p44_train.ipynb (+ upload copy)
    python results/prereg44/explore/p44_colab.py upload-list          # data/prereg44/colab/upload/UPLOAD_LIST.txt

Kit = the code (tools/c10data.py, results/prereg36/explore/p36_lib.py - competition-1 *code* only, results/prereg44/
explore/p44_lib.py, p44_train.py, p44_infer.py) and the three prereg44 move caches (train = A players, hold = S25's
held-out question games, inner = the inner games): official competition-2 training games only. No validation answers,
no competition-1 weight / vector / cache, no checkpoints.

Drive (the account that runs Colab; the main session uploads, this script never touches Drive): MyDrive/C10/p44/
kit_parts/kit.part00.. and MyDrive/C10/p44/p44_train.ipynb. Cell 0 joins the parts into MyDrive/C10/p44/C10_p44_kit.zip
(SHA checked). The run writes MyDrive/C10/p44/run/ (checkpoints every 20,000 steps, logs, train record),
MyDrive/C10/p44/logits/s<step>_<raw|ema>/{hold,inner}.npz, and C10_p44_results.zip (records, logs, logits, the four
candidate checkpoints) for the download. "Run all" again after a disconnect resumes.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
KIT_DIR = os.path.join(REPO, "data", "prereg44", "colab")
KIT_ZIP = os.path.join(KIT_DIR, "C10_p44_kit.zip")
UPLOAD = os.path.join(KIT_DIR, "upload")
PARTS = os.path.join(UPLOAD, "kit_parts")
NB_DIR = os.path.join(REPO, "results", "prereg44", "colab")
MANIFEST = os.path.join(NB_DIR, "kit_manifest.json")
PARTS_SHA = os.path.join(NB_DIR, "kit_parts_sha.txt")
NB = os.path.join(NB_DIR, "p44_train.ipynb")
PART_BYTES = 7000000
DRIVE = "/content/drive/MyDrive/C10/p44"
WORK = "/content/c10"
CACHE_FILES = ("cache.json", "codes.npy", "off.npy", "meta.npz")
FILES = (["tools/c10data.py", "results/prereg36/explore/p36_lib.py", "results/prereg44/explore/p44_lib.py",
          "results/prereg44/explore/p44_train.py", "results/prereg44/explore/p44_infer.py"]
         + ["data/prereg44/cache/%s/%s" % (p, f) for p in ("train", "hold", "inner") for f in CACHE_FILES])
CANDIDATE_STEPS = (140000, 160000, 180000, 200000)


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


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def cmd_kit(_a):
    os.makedirs(KIT_DIR, exist_ok=True)
    files = {}
    tmp = KIT_ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in FILES:
            src = os.path.join(REPO, rel)
            info = zipfile.ZipInfo(rel, date_time=(2026, 11, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            if is_text(rel):
                with open(src, "rb") as f:
                    data = f.read().replace(b"\r\n", b"\n")
                z.writestr(info, data)
                files[rel] = hashlib.sha256(data).hexdigest()
            else:
                with open(src, "rb") as f, z.open(info, "w", force_zip64=True) as w:
                    for b in iter(lambda: f.read(1 << 20), b""):
                        w.write(b)
                files[rel] = sha(src)
    os.replace(tmp, KIT_ZIP)
    man = {"format": "c10-prereg44 colab kit v1", "zip": os.path.basename(KIT_ZIP), "zip_sha256": sha(KIT_ZIP),
           "zip_bytes": os.path.getsize(KIT_ZIP), "files": files}
    write_text(MANIFEST, json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: man[k] for k in ("zip", "zip_sha256", "zip_bytes")}, indent=1))
    return 0


def cmd_parts(_a):
    man = json.load(open(MANIFEST, encoding="utf-8"))
    if sha(KIT_ZIP) != man["zip_sha256"]:
        raise SystemExit("the kit zip differs from kit_manifest.json")
    os.makedirs(PARTS, exist_ok=True)
    for p in glob.glob(os.path.join(PARTS, "kit.part*")):
        os.remove(p)
    lines = []
    with open(KIT_ZIP, "rb") as f:
        i = 0
        while True:
            b = f.read(PART_BYTES)
            if not b:
                break
            name = "kit.part%02d" % i
            with open(os.path.join(PARTS, name), "wb") as o:
                o.write(b)
            lines.append("%s %s %d" % (name, hashlib.sha256(b).hexdigest(), len(b)))
            i += 1
    write_text(PARTS_SHA, "\n".join(lines) + "\n")
    print("%d parts" % len(lines))
    return 0


MD = """# C10 prereg44：競賽二 等級 CNN（CNN44）長訓練（Colab）

**不用改任何程式**：在「變更執行階段類型」選 GPU（G4；拿不到時 A100 也可），按「執行階段 → 全部執行」。

0. 把 `我的雲端硬碟/C10/p44/kit_parts/` 的分割檔接回 `C10_p44_kit.zip`（雜湊核對）。
1. 掛上雲端硬碟、核對 GPU、解開套件並逐檔核對。
2. 第一次先 bench 300 步，再訓練 200,000 步（G4 約 12–13 小時；每 20,000 步存到 `C10/p44/run/ckpt/`；斷線後重按「全部執行」會從最後的存檔接著跑）。
3. 8 個候選版本（第 140k／160k／180k／200k 步 × 原始／EMA）算留出題棋和 inner 棋的每盤 logits，存到 `C10/p44/logits/`。
4. 打包 `C10/p44/C10_p44_results.zip`，下載回傳。

任何一格出現紅字或 `STOP`，停下來，截最後 20 行。"""

CELL_JOIN = """# 0. 把 Drive 上 C10/p44/kit_parts/ 的 __N__ 份接回 C10_p44_kit.zip（已存在且雜湊對就略過）
import hashlib, os, glob, shutil
from google.colab import drive
drive.mount('/content/drive')
D = __DRIVE__
KIT = os.path.join(D, 'C10_p44_kit.zip')
SHA = __KIT_SHA__
PART_SHA = __PART_SHA__
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()
if not (os.path.exists(KIT) and sha(KIT) == SHA):
    parts = sorted(glob.glob(os.path.join(D, 'kit_parts', 'kit.part*')))
    names = [os.path.basename(p) for p in parts]
    assert names == sorted(PART_SHA), 'STOP: kit_parts 的檔名不對（有 %d 份，要 %d 份；注意有沒有「 (1)」）' % (len(parts), len(PART_SHA))
    tmp = '/content/C10_p44_kit.zip'
    with open(tmp, 'wb') as o:
        for p in parts:
            with open(p, 'rb') as f:
                b = f.read()
            assert hashlib.sha256(b).hexdigest() == PART_SHA[os.path.basename(p)], 'STOP: 分割檔雜湊不對 ' + p
            o.write(b)
    assert sha(tmp) == SHA, 'STOP: 接回的 zip 雜湊不對'
    shutil.copyfile(tmp, KIT)
    assert sha(KIT) == SHA, 'STOP: 複製到 Drive 後雜湊不對'
print('kit OK', SHA[:12])"""

CELL_SETUP = """# 1. 雲端硬碟、GPU、套件（不用改）
import hashlib, json, os, subprocess, sys, zipfile
from google.colab import drive
drive.mount('/content/drive')
DRIVE, WORK = __DRIVE__, __WORK__
KIT_SHA, FILES = __KIT_SHA__, __FILES__
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
print(subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout)
tv = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__, torch.backends.cudnn.version(), torch.cuda.is_available())'], capture_output=True, text=True).stdout.strip()
print('torch', tv)   # in a subprocess: the notebook itself never holds the GPU (p44_train refuses to share it)
assert tv.endswith('True'), 'STOP: 沒有 GPU，請在「變更執行階段類型」選 GPU'
kit = os.path.join(DRIVE, 'C10_p44_kit.zip')
ok_flag = os.path.join(WORK, '.kit_ok')
if not (os.path.exists(ok_flag) and open(ok_flag).read() == KIT_SHA):
    assert sha(kit) == KIT_SHA, 'STOP: 套件檔的 SHA-256 不對'
    with zipfile.ZipFile(kit) as z:
        z.extractall(WORK)
    for rel, h in FILES.items():
        assert sha(os.path.join(WORK, rel)) == h, 'STOP: 檔案不對 ' + rel
    open(ok_flag, 'w').write(KIT_SHA)
print('kit ok', len(FILES), 'files')
RUN = os.path.join(DRIVE, 'run')
os.makedirs(RUN, exist_ok=True)"""

CELL_TRAIN = """# 2. 訓練（不用改；斷線後重按「全部執行」會從最後的存檔接著跑）
rec = os.path.join(RUN, 'train_record.json')
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
    ck = os.path.join(RUN, 'ckpt')
    if not (os.path.isdir(ck) and any(f.startswith('step_') for f in os.listdir(ck))):
        rc = run(['results/prereg44/explore/p44_train.py', '--workers', workers, '--bench', '300'], os.path.join(RUN, 'bench_log.txt'))
        print('bench exit', rc)
    rc = run(['results/prereg44/explore/p44_train.py', '--workers', workers, '--auto-resume', '--out', RUN, '--record', rec, '--min-free-ram-mb', '2048'], os.path.join(RUN, 'train_log.txt'))
    assert rc == 0, 'STOP: 訓練結束碼 %d（4＝記憶體不足或 STOP 檔，重按「全部執行」會接續）' % rc
print('training finished')"""

CELL_INFER = """# 3. 8 個候選版本的每盤 logits：留出題的棋（hold）和 inner 的棋（不用改；已完成的會略過）
LOG = os.path.join(DRIVE, 'logits')
rc = run(['results/prereg44/explore/p44_infer.py', 'candidates', '--ckpt-dir', os.path.join(RUN, 'ckpt'), '--parts', 'hold', 'inner', '--workers', workers, '--out-root', LOG], os.path.join(RUN, 'infer_log.txt'))
assert rc == 0, 'STOP: logits 結束碼 %d' % rc
print('logits finished')"""

CELL_ZIP = """# 4. 打包回傳（不用改）
res = os.path.join(DRIVE, 'C10_p44_results.zip')
with zipfile.ZipFile(res + '.tmp', 'w', zipfile.ZIP_STORED) as z:
    for name in ('train_record.json', 'train_log.txt', 'train_log.jsonl', 'bench_log.txt', 'infer_log.txt'):
        p = os.path.join(RUN, name)
        if os.path.exists(p):
            z.write(p, 'run/' + name)
    for st in __STEPS__:
        z.write(os.path.join(RUN, 'ckpt', 'step_%06d.pt' % st), 'run/ckpt/step_%06d.pt' % st)
    for d in sorted(os.listdir(LOG)):
        for part in ('hold', 'inner'):
            z.write(os.path.join(LOG, d, part + '.npz'), 'logits/%s/%s.npz' % (d, part))
os.replace(res + '.tmp', res)
drive.flush_and_unmount()
print('完成。請下載 Drive 的 C10/p44/C10_p44_results.zip 回傳給 Claude（%.0f MB）' % (os.path.getsize(res) / 1e6))"""


def lines(src):
    out = src.split("\n")
    return [x + "\n" for x in out[:-1]] + [out[-1]]


def notebook_text():
    man = json.load(open(MANIFEST, encoding="utf-8"))
    part_sha = {}
    for ln in open(PARTS_SHA, encoding="utf-8").read().split("\n"):
        if ln.strip():
            n, h, _b = ln.split()
            part_sha[n] = h
    rep = {"__DRIVE__": repr(DRIVE), "__WORK__": repr(WORK), "__KIT_SHA__": repr(man["zip_sha256"]),
           "__FILES__": json.dumps(man["files"], sort_keys=True), "__PART_SHA__": json.dumps(part_sha, sort_keys=True),
           "__N__": str(len(part_sha)), "__STEPS__": repr(CANDIDATE_STEPS)}
    cells = []
    for i, (kind, src) in enumerate((("markdown", MD), ("code", CELL_JOIN), ("code", CELL_SETUP),
                                     ("code", CELL_TRAIN), ("code", CELL_INFER), ("code", CELL_ZIP))):
        for k, v in rep.items():
            src = src.replace(k, v)
        c = {"cell_type": kind, "id": "p44-%d" % i, "metadata": {}, "source": lines(src)}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        cells.append(c)
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "G4", "machine_shape": "hm"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def cmd_notebook(a):
    text = notebook_text()
    if a.check:
        with open(NB, encoding="utf-8", newline="") as f:
            same = f.read().replace("\r\n", "\n") == text
        print("same" if same else "differs: %s" % NB)
        return 0 if same else 1
    write_text(NB, text)
    write_text(os.path.join(UPLOAD, "p44_train.ipynb"), text)
    print("wrote %s and the upload copy" % NB)
    return 0


def cmd_upload_list(_a):
    rows = []
    for p in sorted(glob.glob(os.path.join(PARTS, "kit.part*"))):
        rows.append(("kit_parts/" + os.path.basename(p), sha(p), os.path.getsize(p), "MyDrive/C10/p44/kit_parts/"))
    nbp = os.path.join(UPLOAD, "p44_train.ipynb")
    rows.append(("p44_train.ipynb", sha(nbp), os.path.getsize(nbp), "MyDrive/C10/p44/"))
    man = json.load(open(MANIFEST, encoding="utf-8"))
    text = ["# prereg44 Colab upload list (local folder data/prereg44/colab/upload/)",
            "# joined kit: C10_p44_kit.zip sha256 %s, %d bytes (cell 0 of the notebook rebuilds it on Drive)"
            % (man["zip_sha256"], man["zip_bytes"]),
            "# file  sha256  bytes  drive_destination"]
    text += ["%s  %s  %d  %s" % r for r in rows]
    write_text(os.path.join(UPLOAD, "UPLOAD_LIST.txt"), "\n".join(text) + "\n")
    print("%d files" % len(rows))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("kit")
    sub.add_parser("parts")
    n = sub.add_parser("notebook")
    n.add_argument("--check", action="store_true")
    sub.add_parser("upload-list")
    a = ap.parse_args(argv)
    return {"kit": cmd_kit, "parts": cmd_parts, "notebook": cmd_notebook, "upload-list": cmd_upload_list}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
