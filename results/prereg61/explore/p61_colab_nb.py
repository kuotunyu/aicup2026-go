"""Builds results/prereg61/colab/p61_resume.ipynb from prereg62's p62_S1.ipynb (prereg61 revision 1)."""
import base64, hashlib, json, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src = json.load(open(os.path.join(REPO, "results/prereg62/colab/p62_S1.ipynb"), encoding="utf-8"))
cells = src["cells"]
def lines(s):
    o = s.split("\n"); return [x + "\n" for x in o[:-1]] + [o[-1]]
def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(s)}
w = open(os.path.join(REPO, "results/prereg61/explore/p61_colab_run.py"), "rb").read().replace(b"\r\n", b"\n")
wh = hashlib.sha256(w).hexdigest()
parts = sorted(os.listdir(os.path.join(REPO, "data/prereg61/colab/parts")))
ck = os.path.join(REPO, "data/prereg61/S0/ckpt/step_102500.pt")
ckh = hashlib.sha256(open(ck, "rb").read()).hexdigest()
md = {"cell_type": "markdown", "metadata": {}, "source": lines(
"""# C10 prereg61 接續：從 step 102,500 練到 200,000（Colab G4）

執行階段選 G4（執行階段版本 2026.07），按「執行階段 → 全部執行」。不用改任何程式。
套件共用：C10/p42 kit、C10/p51/overlay_parts、C10/p62/overlay2_parts；接續存檔在 C10/p62/P61_resume_parts（10 份）。
輸出在 C10/p62/P61/，每 5,000 步存檔，斷線後重按「全部執行」會接續。練完打包 C10/p62/C10_p62_P61_results.zip。
任何一格出現紅字或 STOP，停下來，截最後 20 行。""")}
c5 = "".join(cells[5]["source"])
c5 = c5.replace("ARM, LAST = 'S1', 200000", "ARM, LAST = 'P61', 200000")
c5 += """
# prereg61 revision 1: the P61 arm wrapper
WH, WB64 = %r, %r
wd = base64.b64decode(WB64)
assert hashlib.sha256(wd).hexdigest() == WH, 'STOP: p61_colab_run.py 不對'
os.makedirs(os.path.join(WORK, 'results/prereg61/explore'), exist_ok=True)
with open(os.path.join(WORK, 'results/prereg61/explore/p61_colab_run.py'), 'wb') as f:
    f.write(wd)
print('p61_colab_run ok', WH[:12])""" % (wh, base64.b64encode(w).decode())
resume = """# 1f. 接續存檔 step_102500（4090 練到這裡）：out/ckpt 沒有任何存檔時，從 C10/p62/P61_resume_parts 接回並核雜湊（不用改）
CKH, NCK = %r, %d
ckdir = os.path.join(out, 'ckpt')
os.makedirs(ckdir, exist_ok=True)
if not glob.glob(os.path.join(ckdir, 'step_*.pt')):
    ps = sorted(glob.glob(os.path.join(P62, 'P61_resume_parts', 'ck102500.part*')))
    assert len(ps) == NCK, 'STOP: P61_resume_parts 只有 %%d 份' %% len(ps)
    tmp = '/content/step_102500.pt'
    with open(tmp, 'wb') as o:
        for p in ps:
            with open(p, 'rb') as f:
                o.write(f.read())
    assert sha(tmp) == CKH, 'STOP: 接回的 step_102500.pt 雜湊不對'
    shutil.copy(tmp, os.path.join(ckdir, 'step_102500.pt'))
print('resume checkpoints:', sorted(os.path.basename(p) for p in glob.glob(os.path.join(ckdir, 'step_*.pt')))[-3:])""" % (ckh, len(parts))
c7 = "".join(cells[7]["source"]).replace("['results/prereg62/explore/p62_run.py', 'train', '--arm', ARM,", "['results/prereg61/explore/p61_colab_run.py', 'train',")
assert "p61_colab_run.py" in c7
new = [md] + cells[1:5] + [code(c5), cells[6], code(resume), code(c7), cells[8]]
src = dict(src, cells=new)
pre = "".join(cells[1]["source"]) + "".join(cells[2]["source"]) + "".join(cells[3]["source"])
assert "import" in pre
os.makedirs(os.path.join(REPO, "results/prereg61/colab"), exist_ok=True)
out = os.path.join(REPO, "results/prereg61/colab/p61_resume.ipynb")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(src, f, ensure_ascii=False, indent=1); f.write("\n")
print(out, hashlib.sha256(open(out, "rb").read()).hexdigest(), "ckpt", ckh[:12], "parts", len(parts))
