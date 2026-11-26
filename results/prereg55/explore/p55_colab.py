"""prereg55: notebook p55_S.ipynb derived from prereg51's p51_B.ipynb (same kit + overlay parts on Drive) plus a cell
that writes p55_run.py (hash-checked). Output on Drive: C10/p55/S/.

    PY results/prereg55/explore/p55_colab.py
"""
import base64
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "results", "prereg51", "colab", "p51_B.ipynb")
NB = os.path.join(REPO, "results", "prereg55", "colab", "p55_S.ipynb")
RUN = "results/prereg55/explore/p55_run.py"


def lines(src):
    out = src.split("\n")
    return [x + "\n" for x in out[:-1]] + [out[-1]]


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(s)}


MD = """# C10 prereg55：c49 配方換種子 1，320,000 步（Colab G4）

執行階段選 G4（RTX PRO 6000；執行階段版本 2026.07），按「執行階段 → 全部執行」。不用改任何程式。
套件和 prereg51 B 臂共用（C10/p42 kit、C10/p51/overlay_parts）；輸出在 C10/p55/S/，每 5,000 步存檔，斷線後重按「全部執行」會接續。
練完打包 C10/p55/C10_p55_S_results.zip。任何一格出現紅字或 STOP，停下來，截最後 20 行。"""

CELL_RUN = """# 1c. prereg55 的包裝程式 p55_run.py（只把種子改成 1）；核雜湊後寫進工作目錄（不用改）
import base64
H, B64 = __H__, __B64__
data = base64.b64decode(B64)
assert hashlib.sha256(data).hexdigest() == H, 'STOP: 內嵌的 p55_run.py 不對'
os.makedirs(os.path.join(WORK, 'results/prereg55/explore'), exist_ok=True)
with open(os.path.join(WORK, '__RUN__'), 'wb') as f:
    f.write(data)
P55 = '/content/drive/MyDrive/C10/p55'
out = os.path.join(P55, 'S')
os.makedirs(out, exist_ok=True)
print('p55_run ok', H[:12])"""

CELL_TRAIN = """# 2. 訓練種子 1、320,000 步（約 20 小時；每 5,000 步存到 C10/p55/S/ckpt/；斷線後重按「全部執行」會接續）
workers = str(max(2, min(12, (os.cpu_count() or 4) - 1)))
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
def run(args, log):
    with open(log, 'a') as lf:
        p = subprocess.Popen([sys.executable, '-u'] + args, cwd=WORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end='')
            lf.write(line); lf.flush()
        return p.wait()
rec = os.path.join(out, 'train_record_S.json')
done = os.path.exists(rec) and json.load(open(rec)).get('finished')
if not done:
    rc = run(['__RUN__', 'train', '--arm', 'S', '--workers', workers, '--auto-resume', '--out', out, '--record', rec, '--min-free-ram-mb', '2048', '--log-every', '1000'], os.path.join(out, 'train_log.txt'))
    assert rc == 0, 'STOP: 訓練結束碼 %d（4＝記憶體不足或 STOP 檔，重按「全部執行」會接續）' % rc
print('training finished')"""

CELL_PACK = """# 3. 打包（不用改）
ck = os.path.join(out, 'ckpt', 'step_320000.pt')
res = os.path.join(P55, 'C10_p55_S_results.zip')
with zipfile.ZipFile(res + '.tmp', 'w', zipfile.ZIP_STORED) as z:
    for name in ('train_record_S.json', 'train_log.txt', 'train_log.jsonl'):
        p = os.path.join(out, name)
        if os.path.exists(p):
            z.write(p, 'S/%s' % name)
    z.write(ck, 'S/ckpt/step_320000.pt')
os.replace(res + '.tmp', res)
drive.flush_and_unmount()
print('完成', os.path.basename(res))"""


def main():
    src = json.load(open(SRC, encoding="utf-8"))
    c = src["cells"]
    with open(os.path.join(REPO, RUN), "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    h = hashlib.sha256(data).hexdigest()
    run_cell = CELL_RUN.replace("__H__", repr(h)).replace("__B64__", repr(base64.b64encode(data).decode("ascii"))) \
        .replace("__RUN__", RUN)
    md = {"cell_type": "markdown", "metadata": {}, "source": lines(MD)}
    cells = [md, c[1], c[2], c[3], code(run_cell), code(CELL_TRAIN.replace("__RUN__", RUN)), code(CELL_PACK)]
    os.makedirs(os.path.dirname(NB), exist_ok=True)
    with open(NB, "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(src, cells=cells), f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps({"notebook": NB, "p55_run_sha256": h}))


if __name__ == "__main__":
    main()
