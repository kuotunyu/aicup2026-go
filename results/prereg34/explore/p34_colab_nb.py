"""prereg34: build p34_ogs_p1.ipynb, the Colab notebook that runs the same P1 extraction (p34_extract.py run, i.e.
results/prereg08/p1/p1_extract.py unchanged) on the OGS games when the 4090 is busy (PLAN.md 3).

    .venv/Scripts/python.exe results/prereg34/explore/p34_colab_nb.py [--check] [--upload-copy]

Reuses the registered GPU fallback kit (PREREG_11; colab_gpu/, tools/g11_gpu_fallback.py kit): C10_gpu_kit_v1.zip, already
in the user's Drive at C10/gpu_fallback/, holds the pinned P1 code, the vendored KataGo code and the HumanSL checkpoint
(the v1.15.0 release asset named in hs_lib.CKPT_URL, SHA-256 pinned). Nothing is downloaded: the notebook checks the kit
zip against results/test11/kit_manifest.json and the small p34 bundle (p34_extract.py, sets.json, the OGS games file,
the seam copy; data/prereg34/colab/upload/C10_p34_bundle.zip, built by `p34_extract.py colab-bundle`) against
results/prereg34/explore/colab_bundle.json, both embedded at build time.

Cells: Drive + GPU + the two zips -> environment (prints `pip list | grep -iE '^torch |nvidia-cudnn'`, refuses unless
torch is a cu128 build: runtime version 2026.07) -> unpack + gpu_fallback_core verify (full checkpoint SHA-256) ->
seam G34 on this GPU (the 400 P3 probe games, compared with the 4090 chunks; stops unless it passes) -> the OGS games
(resumable, chunks written to Drive as they finish; the stage environment is locked in out/ENV.json) -> results zip.

Standard library only; the notebook JSON is deterministic (sort_keys, indent 1, fixed cell ids).
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
NB = os.path.join(HERE, "p34_ogs_p1.ipynb")
KIT_RECORD = os.path.join(REPO, "results", "test11", "kit_manifest.json")
BUNDLE_RECORD = os.path.join(HERE, "colab_bundle.json")
UPLOAD = os.path.join(REPO, "data", "prereg34", "colab", "upload", "p34_ogs_p1.ipynb")
KIT_DRIVE = "/content/drive/MyDrive/C10/gpu_fallback"
P34_DRIVE = "/content/drive/MyDrive/C10/p34"
TORCH_TAG = "+cu128"

MD_INTRO = """# C10 prereg34：OGS 棋譜的 P1 特徵（Colab）

**不用改任何程式。** 開之前先在「執行階段 → 變更執行階段類型」選 GPU（Claude 指定的型號）和**執行階段版本 2026.07**，再按「執行階段 → 全部執行」。

它做的事：
1. 掛上雲端硬碟，核對 `C10/gpu_fallback/C10_gpu_kit_v1.zip`（10 月演練時上傳的套件，含 KataGo 權重）和 `C10/p34/C10_p34_bundle.zip` 的 SHA-256。
2. 印出 `pip list | grep -iE '^torch |nvidia-cudnn'`；torch 不是 cu128 版就停（表示執行階段版本沒選 2026.07）。
3. 先用 400 盤舊棋（接縫 G34）和家裡 4090 的結果比對，沒過就停。
4. 再算 OGS 棋譜（每 200 盤存一次到雲端硬碟；斷線後重按「全部執行」會接著算）。
5. 全部算完，在 `C10/p34/` 產生 `C10_p34_results_<GPU>.zip`，下載給 Claude。

任何一格出現紅字或 `STOP`，就停下來，把最後 20 行截圖給 Claude。"""

CELL_SETUP = """# 1. 雲端硬碟、GPU、兩個 zip（不用改）
import glob, hashlib, json, os, shutil, subprocess, sys, time, zipfile
KIT_ZIP = __KIT_ZIP__
KIT_ZIP_SHA256 = __KIT_SHA__
KIT_ID = __KIT_ID__
BUNDLE_ZIP = __BUNDLE_ZIP__
BUNDLE_SHA256 = __BUNDLE_SHA__
P34_DRIVE = __P34_DRIVE__
TORCH_TAG = __TORCH_TAG__
WORK = '/content/c10'
OUT = os.path.join(P34_DRIVE, 'out')

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
    stop('這個執行階段沒有 GPU：執行階段 → 變更執行階段類型 → 選 Claude 指定的 GPU，再全部執行')
print('GPU:', smi.stdout.strip())
if not os.path.exists(BUNDLE_ZIP):
    # uploaded in pieces (<10 MB each): join them locally; the SHA-256 check below guards the result
    parts = sorted(glob.glob(BUNDLE_ZIP + '.part*'))
    if parts:
        joined = '/content/' + os.path.basename(BUNDLE_ZIP)
        with open(joined, 'wb') as fo:
            for p in parts:
                with open(p, 'rb') as fi:
                    shutil.copyfileobj(fi, fo)
        print('bundle 由', len(parts), '段接回')
        BUNDLE_ZIP = joined
for z in (KIT_ZIP, BUNDLE_ZIP):
    if not os.path.exists(z):
        stop(f'找不到 {z}')
t0 = time.time()
if sha256_file(KIT_ZIP) != KIT_ZIP_SHA256:
    stop('C10_gpu_kit_v1.zip 的 SHA-256 不對（不要重新上傳，截圖給 Claude）')
if sha256_file(BUNDLE_ZIP) != BUNDLE_SHA256:
    stop('C10_p34_bundle.zip 的 SHA-256 不對（重新上傳，或問 Claude）')
print(f'兩個 zip OK（{time.time() - t0:.0f} 秒）')"""

CELL_ENV = """# 2. 執行環境：torch 一定要是 cu128 版（執行階段版本 2026.07）（不用改）
pl = subprocess.run("pip list 2>/dev/null | grep -iE '^torch |nvidia-cudnn'", shell=True, capture_output=True, text=True)
print(pl.stdout)
probe = subprocess.run([sys.executable, '-c', 'import json, sys, torch, numpy; print(json.dumps({"python": sys.version.split()[0], '
                        '"torch": torch.__version__, "numpy": numpy.__version__, "cuda": torch.cuda.is_available(), '
                        '"cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'],
                       capture_output=True, text=True)
print(probe.stdout, probe.stderr[-1500:])
if probe.returncode != 0:
    stop('torch 無法載入')
ENV = json.loads(probe.stdout.strip().splitlines()[-1])
torch_lines = [l for l in pl.stdout.splitlines() if l.lower().startswith('torch ')]
if not ENV['torch'].endswith(TORCH_TAG) or not torch_lines or TORCH_TAG not in torch_lines[0]:
    stop(f'torch 是 {ENV["torch"]}，不是 cu128 版：中斷連線並刪除執行階段 → 變更執行階段類型 → 執行階段版本選 2026.07 → 全部執行')
if not ENV['cuda']:
    stop('CUDA 不能用')
KEY = {k: str(ENV[k]) for k in ('torch', 'cudnn', 'gpu', 'python', 'numpy')}
GPU_SLUG = ''.join(c if c.isalnum() else '-' for c in ENV['gpu'].lower()).strip('-')
print('環境 OK：', KEY)"""

CELL_UNPACK = """# 3. 解開套件和 p34 bundle，逐檔核對（不用改；約 1–2 分鐘）
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
for z in (KIT_ZIP, BUNDLE_ZIP):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(WORK)
KIT = os.path.join(WORK, 'c10kit')
with open(os.path.join(KIT, 'KIT_MANIFEST.json'), encoding='utf-8') as f:
    if json.load(f)['kit_id'] != KIT_ID:
        stop('套件版本不對')
with open(os.path.join(KIT, 'P34_BUNDLE_MANIFEST.json'), encoding='utf-8') as f:
    BM = json.load(f)
if BM['kit_id'] != KIT_ID or BM['kit_zip_sha256'] != KIT_ZIP_SHA256:
    stop('p34 bundle 是給另一份套件做的')
bad = [rel for rel, want in BM['files'].items() if sha256_file(os.path.join(KIT, *rel.split('/'))) != want]
if bad:
    stop(f'p34 bundle 檔案不符：{bad}')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONUTF8'):
    env.pop(k, None)
v = subprocess.run([sys.executable, os.path.join(KIT, 'colab_gpu', 'gpu_fallback_core.py'), 'verify', '--kit', KIT],
                   capture_output=True, text=True, env=env)
print(v.stdout[-2000:], v.stderr[-2000:])
if v.returncode != 0:
    stop('套件核對失敗（含 KataGo 權重的 SHA-256）')
os.makedirs(OUT, exist_ok=True)
env_path = os.path.join(OUT, 'ENV.json')
if os.path.exists(env_path):          # resume only on the same GPU / torch / cuDNN / python / numpy
    with open(env_path, encoding='utf-8') as f:
        first = json.load(f)
    if first['key'] != KEY or first['bundle_sha256'] != BUNDLE_SHA256:
        stop(f'和第一次開跑的環境不同：{first["key"]} vs {KEY}。不要換 GPU 續跑，截圖給 Claude')
else:
    with open(env_path, 'w', encoding='utf-8') as f:
        json.dump({'key': KEY, 'environment': ENV, 'bundle_sha256': BUNDLE_SHA256, 'kit_id': KIT_ID,
                   'gpu_line': smi.stdout.strip(), 'pip_list': pl.stdout, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, f, indent=1)
WORKERS = max(1, min(20, (os.cpu_count() or 2) - 2))
print('核對 OK；結果寫到', OUT, '；workers', WORKERS)

def run_p34(args, log_name):
    cmd = [sys.executable, os.path.join(KIT, 'results', 'prereg34', 'explore', 'p34_extract.py')] + args
    print(' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=KIT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    with open(os.path.join(OUT, log_name), 'a', encoding='utf-8') as lf:
        for line in p.stdout:
            print(line, end='', flush=True)
            lf.write(line)
    return p.wait()"""

CELL_SEAM = """# 4. 接縫 G34：400 盤舊棋在這個 GPU 上重算，和家裡 4090 的結果比（不用改；約 5 分鐘）
rc = run_p34(['run', '--set', 'probe', '--sets', 'data/prereg34/p1/seam_sets.json', '--out-dir', os.path.join(OUT, 'seam'),
              '--workers', str(WORKERS)], 'seam_log.txt')
if rc != 0:
    stop(f'接縫的計算失敗（exit {rc}），截最後 20 行給 Claude')
rc = run_p34(['seam-colab', '--new-dir', os.path.join(OUT, 'seam', 'probe'),
              '--ref', os.path.join(KIT, 'data', 'prereg34', 'p1', 'seam_ref.npz'),
              '--out', os.path.join(OUT, 'seam_colab.json')], 'seam_log.txt')
if rc != 0:
    stop('接縫 G34 沒過：不算 OGS。截圖給 Claude')
print('接縫 G34 過了')"""

CELL_RUN = """# 5. OGS 棋譜（不用改；可以續跑。每 200 盤印一行 chunk done，eta_min＝剩幾分鐘）
rc = run_p34(['run', '--set', 'probe', '--sets', 'data/prereg34/p1/sets.json', '--out-dir', os.path.join(OUT, 'ogs'),
              '--workers', str(WORKERS)], 'ogs_log.txt')
if rc != 0:
    stop(f'計算失敗（exit {rc}），截最後 20 行給 Claude')
n_want = (BM['ogs_games'] + 199) // 200
have = sorted(f for f in os.listdir(os.path.join(OUT, 'ogs', 'probe')) if f.startswith('chunk_') and f.endswith('.npz') and '.tmp' not in f)
if len(have) != n_want:
    stop(f'只有 {len(have)}／{n_want} 塊（可能是 STOP 檔）；要繼續就再按一次全部執行')
print('OGS 全部算完：', len(have), '塊')"""

CELL_PACK = """# 6. 打包結果（不用改）
res_zip = os.path.join(P34_DRIVE, f'C10_p34_results_{GPU_SLUG}.zip')
tmp_zip = '/content/c10_p34_results.zip'
with zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_STORED) as zf:
    for d, _dirs, files in os.walk(OUT):
        for f in sorted(files):
            if '.tmp' in f:
                continue
            p_ = os.path.join(d, f)
            zf.write(p_, os.path.relpath(p_, OUT))
shutil.copyfile(tmp_zip, res_zip)
print('完成：', res_zip, f'{os.path.getsize(res_zip) / 1e6:.1f} MB', 'sha256', sha256_file(res_zip))
print('下載這個檔給 Claude（雲端硬碟 C10 → p34 裡按右鍵 → 下載），然後「執行階段 → 中斷連線並刪除執行階段」。')"""


def lines(src):
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build(kit_rec, bundle_rec):
    sub = {"__KIT_ZIP__": repr(f"{KIT_DRIVE}/{kit_rec['zip']['file']}"), "__KIT_SHA__": repr(kit_rec["zip"]["sha256"]),
           "__KIT_ID__": repr(kit_rec["kit_id"]),
           "__BUNDLE_ZIP__": repr(f"{P34_DRIVE}/{bundle_rec['zip']['file']}"),
           "__BUNDLE_SHA__": repr(bundle_rec["zip"]["sha256"]), "__P34_DRIVE__": repr(P34_DRIVE),
           "__TORCH_TAG__": repr(TORCH_TAG)}
    if bundle_rec["manifest"]["kit_id"] != kit_rec["kit_id"] \
            or bundle_rec["manifest"]["kit_zip_sha256"] != kit_rec["zip"]["sha256"]:
        raise SystemExit("colab_bundle.json was built for another kit")
    setup = CELL_SETUP
    for k, v in sub.items():
        assert setup.count(k) == 1, k
        setup = setup.replace(k, v)
    cells = [("md", "p34-intro", MD_INTRO), ("code", "p34-setup", setup), ("code", "p34-env", CELL_ENV),
             ("code", "p34-unpack", CELL_UNPACK), ("code", "p34-seam", CELL_SEAM), ("code", "p34-run", CELL_RUN),
             ("code", "p34-pack", CELL_PACK)]
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
                                     "c10": {"kit_id": kit_rec["kit_id"], "kit_zip_sha256": kit_rec["zip"]["sha256"],
                                             "bundle_zip_sha256": bundle_rec["zip"]["sha256"],
                                             "plan": "results/prereg34/explore/PLAN.md 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    return (json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def read(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: {what}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--upload-copy", action="store_true")
    a = ap.parse_args(argv)
    b = build(read(KIT_RECORD, "the g11 kit record"),
              read(BUNDLE_RECORD, "run `p34_extract.py colab-bundle` first"))
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
