"""Build colab_gpu/gpu_fallback.ipynb, the Colab notebook of the registered GPU fallback (docs/PREREG_11_GPU_FALLBACK.md;
the user's steps docs/COLAB_GPU_FALLBACK.md).

    python colab_gpu/build_gpu_fallback_nb.py [--check] [--upload-copy]

The notebook does no computing of its own: it mounts Drive, checks the kit zip against the SHA-256 recorded in
results/test11/kit_manifest.json (embedded at build time), installs the pinned torch / numpy, unpacks the kit and the
one input zip, verifies them (colab_gpu/gpu_fallback_core.py verify: every file, the full checkpoint SHA-256) and runs
`gpu_fallback_core.py run` in a subprocess (resumable; every chunk is written to Drive as it finishes), then zips the
output folder on Drive for the download. Re-running "Run all" after a disconnect resumes.

Standard library only; the notebook JSON is written in nbformat v4.5 layout (json.dumps(sort_keys=True, indent=1,
ensure_ascii=False), sources split into lines, fixed cell ids), byte-identical on every build. --check compares the
committed notebook with a fresh build (exit 1 when they differ). --upload-copy also writes the byte copy
data/test11/upload/gpu_fallback.ipynb (not in git).
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NB = os.path.join(HERE, "gpu_fallback.ipynb")
KIT_RECORD = os.path.join(REPO, "results", "test11", "kit_manifest.json")
UPLOAD = os.path.join(REPO, "data", "test11", "upload", "gpu_fallback.ipynb")
TORCH_PIN = "2.11.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_WANT = "2.11.0+cu128"
NUMPY_PIN = "2.5.3"
DRIVE_DIR = "/content/drive/MyDrive/C10/gpu_fallback"


def kit_record():
    with open(KIT_RECORD, encoding="utf-8") as f:
        rec = json.load(f)
    if not rec.get("zip"):
        raise SystemExit("results/test11/kit_manifest.json has no zip record: run tools/g11_gpu_fallback.py kit first")
    return rec


MD_INTRO = """# C10 競賽二：KataGo GPU 備援（PREREG_11）

這份 notebook 只在 `docs/COLAB_GPU_FALLBACK.md` 叫你跑的時候用。**不用改任何程式**：照文件選好 GPU，按「執行階段 → 全部執行」。

它做的事：
1. 掛上 Google 雲端硬碟，找 `我的雲端硬碟/C10/gpu_fallback/` 裡的套件檔 `C10_gpu_kit_v1.zip` 和**唯一一個** `C10_gpu_input_*.zip`。
2. 安裝固定版本的 torch／numpy，解開套件、逐檔核對 SHA-256（含 KataGo 權重）。
3. 在 GPU 上算（每算完一塊就存回雲端硬碟；斷線後重按「全部執行」會接著算）。
4. 全部算完後，在 `C10/gpu_fallback/` 產生 `C10_gpu_results_<名稱>.zip`（演練時名稱含 GPU 型號），下載給 Claude。

任何一格出現紅字或 `STOP`，就停下來，把最後 20 行截圖給 Claude。"""

CELL_SETUP = """# 1. 雲端硬碟、GPU、兩個 zip（不用改）
import glob, hashlib, json, os, subprocess, sys, time
DRIVE_DIR = __DRIVE_DIR__
KIT_ZIP_NAME = __KIT_NAME__
KIT_ZIP_SHA256 = __KIT_SHA__
KIT_ID = __KIT_ID__
TORCH_PIN, TORCH_INDEX, TORCH_WANT, NUMPY_PIN = __TORCH_PIN__, __TORCH_INDEX__, __TORCH_WANT__, __NUMPY_PIN__
WORK = '/content/c10'

def stop(msg):
    raise RuntimeError('STOP: ' + msg)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

from google.colab import drive
drive.mount('/content/drive')
smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True)
if smi.returncode != 0 or not smi.stdout.strip():
    stop('這個執行階段沒有 GPU：執行階段 → 變更執行階段類型 → 選文件指定的 GPU，再全部執行')
print('GPU:', smi.stdout.strip())
if not os.path.isdir(DRIVE_DIR):
    stop(f'雲端硬碟裡沒有 {DRIVE_DIR}（照文件 A 段上傳）')
kit_zip = os.path.join(DRIVE_DIR, KIT_ZIP_NAME)
inputs = sorted(glob.glob(os.path.join(DRIVE_DIR, 'C10_gpu_input_*.zip')))
if not os.path.exists(kit_zip):
    stop(f'找不到 {KIT_ZIP_NAME}')
if len(inputs) != 1:
    stop(f'C10/gpu_fallback 裡要正好一個 C10_gpu_input_*.zip，現在是 {[os.path.basename(x) for x in inputs]}')
INPUT_ZIP = inputs[0]
t0 = time.time()
got = sha256_file(kit_zip)
if got != KIT_ZIP_SHA256:
    stop(f'{KIT_ZIP_NAME} 的 SHA-256 是 {got[:12]}…，應該是 {KIT_ZIP_SHA256[:12]}…（重新上傳，或問 Claude）')
INPUT_SHA256 = sha256_file(INPUT_ZIP)
import zipfile
INPUT = json.loads(zipfile.ZipFile(INPUT_ZIP).read('c10input/INPUT_MANIFEST.json').decode('utf-8'))
print(f'套件 OK（{time.time() - t0:.0f} 秒）；輸入檔 {os.path.basename(INPUT_ZIP)} sha256 {INPUT_SHA256}')
print('輸入：', INPUT['input_id'], INPUT['mode'], [(r['stage'], r['set']) for r in INPUT['runs']])"""

CELL_INSTALL = """# 2. 安裝固定版本的 torch / numpy（不用改；約 3–6 分鐘）
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'torch=={TORCH_WANT}', '--index-url', TORCH_INDEX],
                   capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
r2 = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'numpy=={NUMPY_PIN}'], capture_output=True, text=True)
print(r2.stdout[-800:], r2.stderr[-800:])
probe = subprocess.run([sys.executable, '-c', 'import json, sys, torch, numpy; print(json.dumps({"python": sys.version.split()[0], '
                        '"torch": torch.__version__, "numpy": numpy.__version__, "cuda": torch.cuda.is_available(), '
                        '"cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'],
                       capture_output=True, text=True)
print(probe.stdout, probe.stderr[-1500:])
if probe.returncode != 0:
    stop('torch 無法載入')
ENV = json.loads(probe.stdout.strip().splitlines()[-1])
if ENV['torch'] != TORCH_WANT or ENV['numpy'] != NUMPY_PIN or not ENV['cuda']:
    stop(f'版本不對：{ENV}（要 torch {TORCH_WANT}、numpy {NUMPY_PIN}、CUDA 可用）')
KEY = {k: str(ENV[k]) for k in ('torch', 'cudnn', 'gpu', 'python', 'numpy')}
GPU_SLUG = ''.join(c if c.isalnum() else '-' for c in ENV['gpu'].lower()).strip('-')
if INPUT['mode'] == 'real':      # PREREG_11 3: only an environment a committed rehearsal allowed (checked before anything is written)
    for st in dict.fromkeys(r['stage'] for r in INPUT['runs']):
        ok = (INPUT.get('allowed_environments') or {}).get(st) or []
        if KEY not in ok:
            stop(f'這個 GPU／版本 {KEY} 不在演練准用名單 {ok}。還沒寫任何東西、這一次不算嘗試：'
                 '刪掉執行階段，換 Claude 指定的 GPU 再全部執行，或截圖給 Claude')
print('環境 OK：', ENV, '（准用）' if INPUT['mode'] == 'real' else '')"""

CELL_UNPACK = """# 3. 解開套件和輸入檔，逐檔核對（不用改；約 1–2 分鐘）
import shutil, zipfile
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
for z in (kit_zip, INPUT_ZIP):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(WORK)
KIT, INP = os.path.join(WORK, 'c10kit'), os.path.join(WORK, 'c10input')
with open(os.path.join(KIT, 'KIT_MANIFEST.json'), encoding='utf-8') as f:
    if json.load(f)['kit_id'] != KIT_ID:
        stop('套件版本不對')
with open(os.path.join(INP, 'INPUT_MANIFEST.json'), encoding='utf-8') as f:
    if json.load(f) != INPUT:
        stop('輸入檔內容和第 1 格讀到的不同')
CORE = os.path.join(KIT, 'colab_gpu', 'gpu_fallback_core.py')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
v = subprocess.run([sys.executable, CORE, 'verify', '--kit', KIT, '--input', INP], capture_output=True, text=True, env=env)
print(v.stdout[-3000:], v.stderr[-3000:])
if v.returncode != 0:
    stop('套件或輸入檔核對失敗')
# rehearsal: one output folder per GPU type, so A100, H100 and G4 can be rehearsed one after another
OUT = os.path.join(DRIVE_DIR, 'out', INPUT['input_id'] + (f'_{GPU_SLUG}' if INPUT['mode'] == 'rehearsal' else ''))
print('結果會寫到：', OUT)"""

CELL_RUN = """# 4. 在 GPU 上算（不用改；可以續跑。進度每塊印一行，含 eta_min＝剩幾分鐘）
os.makedirs(OUT, exist_ok=True)
workers = max(1, min(20, (os.cpu_count() or 2) - 2))
cmd = [sys.executable, CORE, 'run', '--kit', KIT, '--input', INP, '--out', OUT, '--workers', str(workers),
       '--tmp', '/content/c10tmp']
print(' '.join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
for line in p.stdout:
    print(line, end='', flush=True)
RUN_CODE = p.wait()
print('exit', RUN_CODE)
if RUN_CODE == 5:
    stop('依要求暫停（STOP 檔或時間到）；要繼續就再按一次全部執行')
if RUN_CODE == 4:
    stop('執行環境和這一段第一次開跑時不同（GPU／torch／cuDNN／Python／numpy）。不要換 GPU 續跑，截圖給 Claude')
if RUN_CODE != 0:
    stop(f'計算失敗（exit {RUN_CODE}），截最後 20 行給 Claude')"""

CELL_PACK = """# 5. 打包結果（不用改）
import zipfile
res_zip = os.path.join(DRIVE_DIR, f"C10_gpu_results_{os.path.basename(OUT)}.zip")
tmp_zip = '/content/c10_results.zip'
with zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_STORED) as zf:
    for d, _dirs, files in os.walk(OUT):
        for f in sorted(files):
            if '.tmp' in f:
                continue
            p_ = os.path.join(d, f)
            zf.write(p_, os.path.relpath(p_, OUT))
shutil.copyfile(tmp_zip, res_zip)
print('完成：', res_zip, f'{os.path.getsize(res_zip) / 1e6:.1f} MB', 'sha256', sha256_file(res_zip))
print('下載這個檔給 Claude（在左邊雲端硬碟清單 C10 → gpu_fallback 裡按右鍵 → 下載），然後照文件刪執行階段。')"""


def lines(src):
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build(rec):
    sub = {"__DRIVE_DIR__": repr(DRIVE_DIR), "__KIT_NAME__": repr(rec["zip"]["file"]),
           "__KIT_SHA__": repr(rec["zip"]["sha256"]), "__KIT_ID__": repr(rec["kit_id"]),
           "__TORCH_PIN__": repr(TORCH_PIN), "__TORCH_INDEX__": repr(TORCH_INDEX), "__TORCH_WANT__": repr(TORCH_WANT),
           "__NUMPY_PIN__": repr(NUMPY_PIN)}
    setup = CELL_SETUP
    for k, v in sub.items():
        assert setup.count(k) == 1, k
        setup = setup.replace(k, v)
    cells = [("md", "g11-intro", MD_INTRO), ("code", "g11-setup", setup), ("code", "g11-install", CELL_INSTALL),
             ("code", "g11-unpack", CELL_UNPACK), ("code", "g11-run", CELL_RUN), ("code", "g11-pack", CELL_PACK)]
    out = []
    for kind, cid, src in cells:
        if kind == "code":
            compile(src, cid, "exec")
            out.append({"cell_type": "code", "execution_count": None, "id": cid, "metadata": {"c10": cid},
                        "outputs": [], "source": lines(src)})
        else:
            out.append({"cell_type": "markdown", "id": cid, "metadata": {"c10": cid}, "source": lines(src)})
    nb = {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                     "language_info": {"name": "python"},
                                     "c10": {"kit_id": rec["kit_id"], "kit_zip_sha256": rec["zip"]["sha256"],
                                             "registration": "docs/PREREG_11_GPU_FALLBACK.md"}},
          "nbformat": 4, "nbformat_minor": 5}
    return (json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--upload-copy", action="store_true")
    a = ap.parse_args(argv)
    b = build(kit_record())
    if a.check:
        with open(NB, "rb") as f:
            same = f.read() == b
        print("notebook equals a fresh build" if same else "notebook differs from a fresh build")
        return 0 if same else 1
    with open(NB, "wb") as f:
        f.write(b)
    print(f"{NB}: sha256 {hashlib.sha256(b).hexdigest()}")
    if a.upload_copy:
        os.makedirs(os.path.dirname(UPLOAD), exist_ok=True)
        with open(UPLOAD, "wb") as f:
            f.write(b)
        print(f"upload copy: {UPLOAD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
