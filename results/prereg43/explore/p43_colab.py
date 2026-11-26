"""prereg43: Colab notebooks p43_S0.ipynb / p43_S1.ipynb, derived from prereg42's p42_V2.ipynb (same kit on Drive,
C10/p42/C10_p42_kit.zip) + an overlay cell that writes p43_run.py (hash-checked). Outputs go to C10/p43/.

    PY results/prereg43/explore/p43_colab.py [--check]
"""
import argparse
import base64
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC_NB = os.path.join(REPO, "results", "prereg42", "colab", "p42_V2.ipynb")
NB_DIR = os.path.join(REPO, "results", "prereg43", "colab")
RUN = "results/prereg43/explore/p43_run.py"
STEP = "step_320000.pt"


def lines(src):
    out = src.split("\n")
    return [x + "\n" for x in out[:-1]] + [out[-1]]


def overlay():
    with open(os.path.join(REPO, RUN), "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest(), base64.b64encode(data).decode("ascii")


OVERLAY_CELL = """# 1b. prereg43 的包裝程式 p43_run.py（只設 320,000 步、每 32,000 步存檔、種子）；核雜湊後寫進工作目錄（不用改）
import base64
H, B64 = __H__, __B64__
data = base64.b64decode(B64)
assert hashlib.sha256(data).hexdigest() == H, 'STOP: 內嵌的 p43_run.py 不對'
os.makedirs(os.path.join(WORK, 'results/prereg43/explore'), exist_ok=True)
with open(os.path.join(WORK, '__RUN__'), 'wb') as f:
    f.write(data)
assert sha(os.path.join(WORK, '__RUN__')) == H
P43 = '/content/drive/MyDrive/C10/p43'
os.makedirs(P43, exist_ok=True)
print('overlay ok', H[:12])"""

TRAIN_CELL = """# 2. 訓練 V2 320,000 步、種子 __SEED__（約 20 小時；每 32,000 步存到 C10/p43/S__SEED__/；斷線後重按「全部執行」接續）
out = os.path.join(P43, 'S__SEED__')
os.makedirs(out, exist_ok=True)
rec = os.path.join(out, 'train_record_S__SEED__.json')
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
    rc = run(['__RUN__', 'train', '--seed', '__SEED__', '--workers', workers, '--auto-resume', '--out', out, '--record', rec, '--min-free-ram-mb', '2048', '--log-every', '1000'], os.path.join(out, 'train_log.txt'))
    assert rc == 0, 'STOP: 訓練結束碼 %d（4＝記憶體不足或 STOP 檔，重按「全部執行」會接續）' % rc
print('training finished')"""

EMBED_CELL = """# 3. D 玩家的向量（最後存檔的 raw、EMA；不用改）
ck = os.path.join(out, 'ckpt', '__STEP__')
emb_root = os.path.join(P43, 'emb_S__SEED__')
for w in ('raw', 'ema'):
    if os.path.exists(os.path.join(emb_root, 'V2_step_320000_%s_tta1' % w, 'store_D.npz')):
        continue
    rc = run(['__RUN__', 'embed', '--ckpt', ck, '--players', 'D', '--workers', workers, '--emb-root', emb_root, '--weights', w], os.path.join(out, 'embed_log.txt'))
    assert rc == 0, 'STOP: 向量結束碼 %d' % rc
print('embeddings finished')"""

ZIP_CELL = """# 4. 打包（不用改）
res = os.path.join(P43, 'C10_p43_S__SEED___results.zip')
with zipfile.ZipFile(res + '.tmp', 'w', zipfile.ZIP_STORED) as z:
    for name in ('train_record_S__SEED__.json', 'train_log.txt', 'train_log.jsonl', 'embed_log.txt'):
        p = os.path.join(out, name)
        if os.path.exists(p):
            z.write(p, 'S__SEED__/%s' % name)
    z.write(ck, 'S__SEED__/ckpt/__STEP__')
    for d in sorted(os.listdir(emb_root)):
        z.write(os.path.join(emb_root, d, 'store_D.npz'), 'emb_S__SEED__/%s/store_D.npz' % d)
os.replace(res + '.tmp', res)
drive.flush_and_unmount()
print('完成', os.path.basename(res))"""

MD = """# C10 prereg43：V2 主幹 320,000 步、種子 __SEED__（Colab G4）

執行階段選 G4（RTX PRO 6000），按「執行階段 → 全部執行」。套件沿用 `C10/p42/C10_p42_kit.zip`（prereg42），
1b 格寫入 prereg43 的包裝程式；輸出在 `C10/p43/`。斷線後重按「全部執行」會從最後存檔接續。"""


def notebook(seed):
    with open(SRC_NB, encoding="utf-8") as f:
        nb = json.load(f)
    h, b64 = overlay()
    rep = {"__H__": repr(h), "__B64__": repr(b64), "__RUN__": RUN, "__SEED__": str(seed), "__STEP__": STEP}

    def sub(s):
        for k, v in rep.items():
            s = s.replace(k, v)
        return s

    src = {c.get("id"): c for c in nb["cells"]}
    cells = [{"cell_type": "markdown", "id": "p43-s%d-md" % seed, "metadata": {}, "source": lines(sub(MD))}]
    join = [c for c in nb["cells"] if c.get("id") is None]
    assert len(join) == 1
    for cid, c in (("p43-s%d-0" % seed, join[0]), ("p43-s%d-1" % seed, src["p42-v2-1"])):
        cells.append({"cell_type": "code", "id": cid, "metadata": {}, "source": c["source"],
                      "execution_count": None, "outputs": []})
    assert "VARIANT = '/content/drive/MyDrive/C10/p42', '/content/c10', 'V2'" in "".join(cells[-1]["source"])
    for i, body in (("1b", OVERLAY_CELL), ("2", TRAIN_CELL), ("3", EMBED_CELL), ("4", ZIP_CELL)):
        cells.append({"cell_type": "code", "id": "p43-s%d-%s" % (seed, i), "metadata": {}, "source": lines(sub(body)),
                      "execution_count": None, "outputs": []})
    nb["cells"] = cells
    nb["metadata"]["colab"] = {"gpuType": "G4", "machine_shape": "hm"}
    return json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n", h


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    bad = 0
    for s in (0, 1):
        text, h = notebook(s)
        path = os.path.join(NB_DIR, "p43_S%d.ipynb" % s)
        if a.check:
            with open(path, encoding="utf-8", newline="") as f:
                if f.read().replace("\r\n", "\n") != text:
                    print("differs: %s" % path)
                    bad = 1
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote %s (p43_run.py sha %s)" % (path, h[:12]))
    return bad


if __name__ == "__main__":
    sys.exit(main())
