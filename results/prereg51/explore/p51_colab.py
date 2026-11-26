"""prereg51 arm B on Colab G4: overlay zip (prereg43/49/50/51 wrappers + the T49 cache) split into parts for Drive,
and the notebook p51_B.ipynb (prereg42 kit from C10/p42 as prereg43 did, overlay from C10/p51/overlay_parts).

    PY results/prereg51/explore/p51_colab.py build      # -> data/prereg51/colab/{C10_p51_overlay.zip, parts/}, notebook
"""
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "data", "prereg51", "colab")
ZIP = os.path.join(OUT, "C10_p51_overlay.zip")
PARTS = os.path.join(OUT, "parts")
NB = os.path.join(REPO, "results", "prereg51", "colab", "p51_B.ipynb")
SRC_NB = os.path.join(REPO, "results", "prereg43", "colab", "p43_S0.ipynb")
PART_BYTES = 7 << 20   # base64 must stay under the Drive connector's 10 MB request limit
FILES = ["results/prereg43/explore/p43_run.py", "results/prereg49/explore/p49_lib.py",
         "results/prereg49/explore/p49_run.py", "results/prereg50/explore/p50_run.py",
         "results/prereg51/explore/p51_run.py",
         "data/prereg49/cache/cache.json", "data/prereg49/cache/codes.npy", "data/prereg49/cache/off.npy",
         "data/prereg49/cache/meta.npz"]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def lines(src):
    out = src.split("\n")
    return [x + "\n" for x in out[:-1]] + [out[-1]]


def build_zip():
    os.makedirs(PARTS, exist_ok=True)
    files = {}
    with zipfile.ZipFile(ZIP + ".tmp", "w", zipfile.ZIP_STORED) as z:
        for rel in FILES:
            info = zipfile.ZipInfo(rel, date_time=(2026, 11, 18, 0, 0, 0))
            with open(os.path.join(REPO, rel), "rb") as f:
                data = f.read()
            if rel.endswith((".py", ".json")):
                data = data.replace(b"\r\n", b"\n")
            z.writestr(info, data)
            files[rel] = sha_bytes(data)
    os.replace(ZIP + ".tmp", ZIP)
    for p in os.listdir(PARTS):
        os.remove(os.path.join(PARTS, p))
    parts = []
    with open(ZIP, "rb") as f:
        i = 0
        for b in iter(lambda: f.read(PART_BYTES), b""):
            name = "overlay.part%03d" % i
            with open(os.path.join(PARTS, name), "wb") as o:
                o.write(b)
            parts.append([name, sha_bytes(b)])
            i += 1
    return files, sha(ZIP), parts


CELL_OVERLAY = """# 1b. prereg51 的程式和 T49 快取：從 C10/p51/overlay_parts/ 接回 overlay zip、核雜湊、解到工作目錄（不用改）
import glob, shutil
P51 = '/content/drive/MyDrive/C10/p51'
OSHA, NPARTS, OFILES = __OSHA__, __NPARTS__, __OFILES__
flag = os.path.join(WORK, '.p51_ok')
if not (os.path.exists(flag) and open(flag).read() == OSHA):
    parts = sorted(glob.glob(os.path.join(P51, 'overlay_parts', 'overlay.part*')))
    assert len(parts) == NPARTS, 'STOP: overlay_parts 只有 %d 份' % len(parts)
    tmp = '/content/C10_p51_overlay.zip'
    with open(tmp, 'wb') as o:
        for p in parts:
            with open(p, 'rb') as f:
                o.write(f.read())
    assert sha(tmp) == OSHA, 'STOP: 接回的 overlay zip 雜湊不對'
    with zipfile.ZipFile(tmp) as z:
        z.extractall(WORK)
    for rel, h in OFILES.items():
        assert sha(os.path.join(WORK, rel)) == h, 'STOP: 檔案不對 ' + rel
    open(flag, 'w').write(OSHA)
print('overlay ok', OSHA[:12])"""

CELL_SIZE = """# 2a. 測速選大小（PLAN 第 3 節：B15C128 預估 > 34 小時就用 B12C128；結果存在 Drive，斷線重跑不再測）
out = os.path.join(P51, 'B')
os.makedirs(out, exist_ok=True)
workers = str(max(2, min(12, (os.cpu_count() or 4) - 1)))
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
def run(args, log):
    with open(log, 'a') as lf:
        p = subprocess.Popen([sys.executable, '-u'] + args, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines_ = []
        for line in p.stdout:
            print(line, end='')
            lf.write(line); lf.flush(); lines_.append(line)
        return p.wait(), lines_
size_f = os.path.join(out, 'size_B.json')
if not os.path.exists(size_f):
    rc, ls = run(['results/prereg51/explore/p51_run.py', 'bench', '--arm', 'B', '--size', 'B15C128', '--steps-timed', '300', '--workers', workers], os.path.join(out, 'bench_log.txt'))
    assert rc == 0, 'STOP: 測速結束碼 %d' % rc
    res = json.loads([x for x in ls if '"steps_per_s"' in x][-1])
    hours = 320000 / res['steps_per_s'] / 3600
    choice = {'bench_B15C128': res, 'hours_B15C128': round(hours, 2), 'size': 'B15C128' if hours <= 34 else 'B12C128',
              'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()}
    json.dump(choice, open(size_f, 'w'), indent=1)
SIZE = json.load(open(size_f))['size']
if os.path.exists(os.path.join(out, 'force_B12C128.txt')):   # 戰情室看 units/h 後要求改小（PLAN 第 3 節）
    assert not os.path.isdir(os.path.join(out, 'ckpt')) or SIZE == 'B12C128', 'STOP: 已用 B15C128 存檔，改大小前先把 C10/p51/B/ckpt 改名'
    SIZE = 'B12C128'
print('size', SIZE, json.load(open(size_f))['hours_B15C128'], 'h for B15C128')"""

CELL_TRAIN = """# 2b. 訓練 B 臂 320,000 步（每 16,000 步存到 C10/p51/B/；斷線後重按「全部執行」會從最後的存檔接著跑）
rec = os.path.join(out, 'train_record_B.json')
done = os.path.exists(rec) and json.load(open(rec)).get('finished')
if not done:
    rc, _ = run(['results/prereg51/explore/p51_run.py', 'train', '--arm', 'B', '--size', SIZE, '--workers', workers, '--auto-resume', '--out', out, '--record', rec, '--min-free-ram-mb', '2048', '--log-every', '1000'], os.path.join(out, 'train_log.txt'))
    assert rc == 0, 'STOP: 訓練結束碼 %d（4＝記憶體不足或 STOP 檔，重按「全部執行」會接續）' % rc
print('training finished')"""

CELL_PACK = """# 3. 打包（不用改）
ck = os.path.join(out, 'ckpt', 'step_320000.pt')
res = os.path.join(P51, 'C10_p51_B_results.zip')
with zipfile.ZipFile(res + '.tmp', 'w', zipfile.ZIP_STORED) as z:
    for name in ('size_B.json', 'bench_log.txt', 'train_record_B.json', 'train_log.txt', 'train_log.jsonl'):
        p = os.path.join(out, name)
        if os.path.exists(p):
            z.write(p, 'B/%s' % name)
    z.write(ck, 'B/ckpt/step_320000.pt')
os.replace(res + '.tmp', res)
drive.flush_and_unmount()
print('完成', os.path.basename(res))"""

MD = """# C10 prereg51 B 臂：加大主幹 320,000 步（Colab G4）

執行階段選 G4（RTX PRO 6000），按「執行階段 → 全部執行」。不用改任何程式。

0–1. 從 `C10/p42/kit_parts/` 接回 prereg42 的套件（同 prereg43）。
1b. 從 `C10/p51/overlay_parts/` 接回 prereg51 的程式和 T49 快取。
2a. 測 10 分鐘速度選大小；2b. 訓練（斷線後重按「全部執行」會接續）。
3. 打包 `C10/p51/C10_p51_B_results.zip`。

任何一格出現紅字或 `STOP`，停下來，截最後 20 行。"""


def build_nb(files, osha, nparts):
    src = json.load(open(SRC_NB, encoding="utf-8"))
    cells = src["cells"]
    md = {"cell_type": "markdown", "metadata": {}, "source": lines(MD)}
    kit0, kit1 = cells[1], cells[2]

    def code(s):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(s)}
    ov = CELL_OVERLAY.replace("__OSHA__", repr(osha)).replace("__NPARTS__", str(nparts)) \
        .replace("__OFILES__", json.dumps(files, sort_keys=True))
    nb = dict(src, cells=[md, kit0, kit1, code(ov), code(CELL_SIZE), code(CELL_TRAIN), code(CELL_PACK)])
    os.makedirs(os.path.dirname(NB), exist_ok=True)
    with open(NB, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


def main():
    if sys.argv[1:] != ["build"]:
        raise SystemExit("usage: p51_colab.py build")
    files, osha, parts = build_zip()
    build_nb(files, osha, len(parts))
    man = {"format": "c10-prereg51 colab overlay v1", "zip_sha256": osha, "zip_bytes": os.path.getsize(ZIP),
           "files": files, "parts": parts, "notebook_sha256": sha(NB)}
    with open(os.path.join(REPO, "results", "prereg51", "colab", "overlay_manifest.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    print(json.dumps({"zip_sha256": osha, "zip_bytes": man["zip_bytes"], "parts": len(parts)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
