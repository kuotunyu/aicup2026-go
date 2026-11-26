"""prereg62 on Colab G4: (a) overlay2 zip (prereg60 / prereg61 code + the c49 checkpoint) split into 7 MB parts for
Drive C10/p62/overlay2_parts, (b) the notebooks p62_S1 / p62_S2 / p62_X.ipynb derived from prereg55's p55_S.ipynb
(same prereg42 kit and prereg51 overlay parts on Drive), each embedding p62_run.py (hash-checked).

    PY results/prereg62/explore/p62_colab.py build   # -> data/prereg62/colab/{C10_p62_overlay2.zip, parts/},
                                                     #    results/prereg62/colab/{p62_*.ipynb, overlay2_parts_sha.txt,
                                                     #    overlay2_manifest.json}

What the kit and prereg51's overlay already hold (checked against p55_S.ipynb's hash lists, not assumed):
kit = prereg36 / prereg42 code, tools/c10data.py, the prereg36 cache; overlay = p43_run, p49_lib, p49_run, p50_run,
p51_run and the prereg49 (T49) cache. Missing for p61_run / p60_run: the files of FILES below + p62_run.py.
"""
import base64
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(REPO, "data", "prereg62", "colab")
ZIP = os.path.join(OUT, "C10_p62_overlay2.zip")
PARTS = os.path.join(OUT, "parts")
COLAB = os.path.join(REPO, "results", "prereg62", "colab")
SRC_NB = os.path.join(REPO, "results", "prereg55", "colab", "p55_S.ipynb")
RUN = "results/prereg62/explore/p62_run.py"
PART_BYTES = 7 << 20   # base64 must stay under the Drive connector's 10 MB request limit
FILES = ["results/prereg60/explore/p60_lib.py", "results/prereg60/explore/p60_run.py",
         "results/prereg61/explore/p61_run.py", "data/prereg49/S0/ckpt/step_320000.pt"]
C49_SHA = "c091d3af1797a19ce98dc0a670eea211ceca2de4b58ee4a3c62b73b854748e12"
HOURS = {"S1": 10, "S2": 10, "X": 21}      # not measured: steps / 5.4 steps/s (the 17-plane network on G4)
TITLE = {"S1": "prereg61 配方換種子 1，200,000 步", "S2": "prereg61 配方換種子 2，200,000 步",
         "X": "prereg61 配方拉長到 400,000 步（種子 0）"}


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


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(s)}


def build_zip():
    os.makedirs(PARTS, exist_ok=True)
    files = {}
    with zipfile.ZipFile(ZIP + ".tmp", "w", zipfile.ZIP_STORED) as z:
        for rel in FILES:
            info = zipfile.ZipInfo(rel, date_time=(2026, 11, 21, 0, 0, 0))
            with open(os.path.join(REPO, rel), "rb") as f:
                data = f.read()
            if rel.endswith((".py", ".json")):
                data = data.replace(b"\r\n", b"\n")
            z.writestr(info, data)
            files[rel] = sha_bytes(data)
    if files[FILES[-1]] != C49_SHA:
        raise SystemExit("data/prereg49/S0/ckpt/step_320000.pt is not c49")
    os.replace(ZIP + ".tmp", ZIP)
    for p in os.listdir(PARTS):
        os.remove(os.path.join(PARTS, p))
    parts = []
    with open(ZIP, "rb") as f:
        i = 0
        for b in iter(lambda: f.read(PART_BYTES), b""):
            name = "overlay2.part%03d" % i
            with open(os.path.join(PARTS, name), "wb") as o:
                o.write(b)
            parts.append([name, sha_bytes(b), len(b)])
            i += 1
    return files, sha(ZIP), parts


MD = """# C10 prereg62 __ARM__：__TITLE__（Colab G4）

執行階段選 G4（RTX PRO 6000；執行階段版本 2026.07），按「執行階段 → 全部執行」。不用改任何程式。
套件共用：C10/p42 kit、C10/p51/overlay_parts、C10/p62/overlay2_parts（prereg60／61 的程式和 c49 存檔）。
輸出在 C10/p62/__ARM__/，每 5,000 步存檔到 C10/p62/__ARM__/ckpt/，斷線後重按「全部執行」會從最後的存檔接續。
開跑後幾分鐘，訓練那一格每 500 步會印一行，裡面的 `steps_per_s` 就是速度（預估約 5，全程約 __HOURS__ 小時；沒實測過）。
練完打包 C10/p62/C10_p62___ARM___results.zip。任何一格出現紅字或 STOP，停下來，截最後 20 行。"""

CELL_OVERLAY2 = """# 1c. prereg60／61 的程式和 c49 存檔：從 C10/p62/overlay2_parts/ 接回 overlay2 zip、核雜湊、解到工作目錄（不用改）
P62 = '/content/drive/MyDrive/C10/p62'
O2SHA, N2PARTS, O2FILES = __OSHA__, __NPARTS__, __OFILES__
flag = os.path.join(WORK, '.p62_ok')
if not (os.path.exists(flag) and open(flag).read() == O2SHA):
    parts = sorted(glob.glob(os.path.join(P62, 'overlay2_parts', 'overlay2.part*')))
    assert len(parts) == N2PARTS, 'STOP: overlay2_parts 只有 %d 份' % len(parts)
    tmp = '/content/C10_p62_overlay2.zip'
    with open(tmp, 'wb') as o:
        for p in parts:
            with open(p, 'rb') as f:
                o.write(f.read())
    assert sha(tmp) == O2SHA, 'STOP: 接回的 overlay2 zip 雜湊不對'
    with zipfile.ZipFile(tmp) as z:
        z.extractall(WORK)
    for rel, h in O2FILES.items():
        assert sha(os.path.join(WORK, rel)) == h, 'STOP: 檔案不對 ' + rel
    open(flag, 'w').write(O2SHA)
print('overlay2 ok', O2SHA[:12])"""

CELL_RUN = """# 1d. prereg62 的包裝程式 p62_run.py（--arm 決定種子和步數）；核雜湊後寫進工作目錄（不用改）
import base64
ARM, LAST = '__ARM__', __LAST__
H, B64 = __H__, __B64__
data = base64.b64decode(B64)
assert hashlib.sha256(data).hexdigest() == H, 'STOP: 內嵌的 p62_run.py 不對'
os.makedirs(os.path.join(WORK, 'results/prereg62/explore'), exist_ok=True)
with open(os.path.join(WORK, '__RUN__'), 'wb') as f:
    f.write(data)
out = os.path.join(P62, ARM)
os.makedirs(out, exist_ok=True)
print('p62_run ok', H[:12], 'arm', ARM)"""

CELL_VER = """# 1e. 版本（只印出來；在子程序裡查，筆記本本身不佔 GPU）
print(subprocess.run([sys.executable, '-c', 'import torch; print("torch", torch.__version__, "cuDNN", torch.backends.cudnn.version(), torch.cuda.get_device_name(0))'], capture_output=True, text=True).stdout.strip())
print(subprocess.run('pip list 2>/dev/null | grep -iE "^torch |nvidia-cudnn|^numpy "', shell=True, capture_output=True, text=True).stdout)"""

CELL_TRAIN = """# 2. 訓練 __ARM__（約 __HOURS__ 小時；每 5,000 步存到 C10/p62/__ARM__/ckpt/；斷線後重按「全部執行」會接續）
import threading, time
workers = str(max(2, min(12, (os.cpu_count() or 4) - 1)))
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
def run(args, log, jl):
    with open(log, 'a') as lf:
        p = subprocess.Popen([sys.executable, '-u'] + args, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        def tail():   # 每分鐘把訓練紀錄的最新一行印出來（steps_per_s＝速度）
            seen = None
            while p.poll() is None:
                time.sleep(60)
                try:
                    with open(jl) as f:
                        last = f.read().strip().split('\\n')[-1]
                except OSError:
                    continue
                if last and last != seen:
                    seen = last
                    print(last, flush=True)
        threading.Thread(target=tail, daemon=True).start()
        for line in p.stdout:
            print(line, end='')
            lf.write(line); lf.flush()
        return p.wait()
rec = os.path.join(out, 'train_record_%s.json' % ARM)
for attempt in range(6):
    if os.path.exists(rec) and json.load(open(rec)).get('finished'):
        break
    assert not os.path.exists(os.path.join(out, 'STOP')), 'STOP: C10/p62/%s/STOP 存在，拿掉後再按全部執行' % ARM
    rc = run(['__RUN__', 'train', '--arm', ARM, '--workers', workers, '--auto-resume', '--out', out, '--record', rec, '--min-free-ram-mb', '2048', '--log-every', '500'], os.path.join(out, 'train_log.txt'), os.path.join(out, 'train_log.jsonl'))
    assert rc in (0, 4), 'STOP: 訓練結束碼 %d' % rc
    if rc == 4:
        print('結束碼 4（記憶體不足或 STOP 檔）：60 秒後從存檔接續，第 %d 次' % (attempt + 1))
        time.sleep(60)
assert os.path.exists(rec) and json.load(open(rec)).get('finished'), 'STOP: 還沒練完，重按「全部執行」會接續'
print('training finished')"""

CELL_PACK = """# 3. 打包（不用改）
ck = os.path.join(out, 'ckpt', 'step_%06d.pt' % LAST)
res = os.path.join(P62, 'C10_p62_%s_results.zip' % ARM)
with zipfile.ZipFile(res + '.tmp', 'w', zipfile.ZIP_STORED) as z:
    for name in ('train_record_%s.json' % ARM, 'train_log.txt', 'train_log.jsonl'):
        p = os.path.join(out, name)
        if os.path.exists(p):
            z.write(p, '%s/%s' % (ARM, name))
    z.write(ck, '%s/ckpt/step_%06d.pt' % (ARM, LAST))
os.replace(res + '.tmp', res)
drive.flush_and_unmount()
print('完成', os.path.basename(res))"""


def build_nb(arm, files, osha, nparts, run_bytes):
    import p62_run
    src = json.load(open(SRC_NB, encoding="utf-8"))
    c = src["cells"]
    h = sha_bytes(run_bytes)

    def fill(s):
        return s.replace("__ARM__", arm).replace("__TITLE__", TITLE[arm]).replace("__HOURS__", str(HOURS[arm])) \
            .replace("__LAST__", str(p62_run.ARMS[arm]["steps"])).replace("__RUN__", RUN)
    ov = CELL_OVERLAY2.replace("__OSHA__", repr(osha)).replace("__NPARTS__", str(nparts)) \
        .replace("__OFILES__", json.dumps(files, sort_keys=True))
    rc = fill(CELL_RUN).replace("__H__", repr(h)).replace("__B64__", repr(base64.b64encode(run_bytes).decode("ascii")))
    md = {"cell_type": "markdown", "metadata": {}, "source": lines(fill(MD))}
    cells = [md, c[1], c[2], c[3], code(ov), code(rc), code(CELL_VER), code(fill(CELL_TRAIN)), code(CELL_PACK)]
    nb = os.path.join(COLAB, "p62_%s.ipynb" % arm)
    with open(nb, "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(src, cells=cells), f, ensure_ascii=False, indent=1)
        f.write("\n")
    return nb


def main():
    if sys.argv[1:] != ["build"]:
        raise SystemExit("usage: p62_colab.py build")
    import p62_run
    os.makedirs(COLAB, exist_ok=True)
    files, osha, parts = build_zip()
    with open(os.path.join(REPO, RUN), "rb") as f:
        run_bytes = f.read().replace(b"\r\n", b"\n")
    nbs = {arm: sha(build_nb(arm, files, osha, len(parts), run_bytes)) for arm in sorted(p62_run.ARMS)}
    with open(os.path.join(COLAB, "overlay2_parts_sha.txt"), "w", encoding="utf-8", newline="\n") as f:
        for name, h, n in parts:
            f.write("%s  %s  %d\n" % (h, name, n))
    man = {"format": "c10-prereg62 colab overlay2 v1", "zip_sha256": osha, "zip_bytes": os.path.getsize(ZIP),
           "files": files, "parts": parts, "p62_run_sha256": sha_bytes(run_bytes), "notebooks_sha256": nbs}
    with open(os.path.join(COLAB, "overlay2_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    print(json.dumps({"zip_sha256": osha, "zip_bytes": man["zip_bytes"], "parts": len(parts), "notebooks": nbs}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
