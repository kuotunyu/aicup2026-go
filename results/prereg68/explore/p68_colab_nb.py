"""Builds results/prereg68/colab/p68_X2.ipynb and p68_X3.ipynb from prereg62's p62_X.ipynb."""
import base64, hashlib, json, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src = json.load(open(os.path.join(REPO, "results/prereg62/colab/p62_X.ipynb"), encoding="utf-8"))
cells = src["cells"]
def lines(s):
    o = s.split("\n"); return [x + "\n" for x in o[:-1]] + [o[-1]]
def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(s)}
w = open(os.path.join(REPO, "results/prereg68/explore/p68_colab_run.py"), "rb").read().replace(b"\r\n", b"\n")
wh = hashlib.sha256(w).hexdigest()
for arm, seed in (("X2", 1), ("X3", 2)):
    md = {"cell_type": "markdown", "metadata": {}, "source": lines(
"""# C10 prereg68 %s：prereg61 配方、種子 %d、400,000 步（Colab G4）

執行階段選 G4（執行階段版本 2026.07），按「執行階段 → 全部執行」。不用改任何程式。
套件共用：C10/p42 kit、C10/p51/overlay_parts、C10/p62/overlay2_parts。
輸出在 C10/p62/%s/，每 5,000 步存檔，斷線後重按「全部執行」會接續（約 22 小時；24 小時上限前若斷線，重開執行階段再全部執行）。
練完打包 C10/p62/C10_p62_%s_results.zip。任何一格出現紅字或 STOP，停下來，截最後 20 行。""" % (arm, seed, arm, arm))}
    c5 = "".join(cells[5]["source"]).replace("ARM, LAST = 'X', 400000", "ARM, LAST = '%s', 400000" % arm)
    assert "'%s'" % arm in c5
    c5 += """
# prereg68 wrapper (adds arms X2 / X3)
WH, WB64 = %r, %r
wd = base64.b64decode(WB64)
assert hashlib.sha256(wd).hexdigest() == WH, 'STOP: p68_colab_run.py 不對'
os.makedirs(os.path.join(WORK, 'results/prereg68/explore'), exist_ok=True)
with open(os.path.join(WORK, 'results/prereg68/explore/p68_colab_run.py'), 'wb') as f:
    f.write(wd)
print('p68_colab_run ok', WH[:12])""" % (wh, base64.b64encode(w).decode())
    c7 = "".join(cells[7]["source"]).replace("['results/prereg62/explore/p62_run.py', 'train', '--arm', ARM,",
                                              "['results/prereg68/explore/p68_colab_run.py', 'train', '--arm', ARM,")
    assert "p68_colab_run.py" in c7
    nb = dict(src, cells=[md] + cells[1:5] + [code(c5), cells[6], code(c7), cells[8]])
    out = os.path.join(REPO, "results/prereg68/colab/p68_%s.ipynb" % arm)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1); f.write("\n")
    print(arm, hashlib.sha256(open(out, "rb").read()).hexdigest())
