"""Cell-by-cell unified diff: official player-identification.ipynb vs colab/prereg01_player.ipynb.

Covers code and markdown cells. The deviation list printed at the top is read from the
notebook's own metadata (the same list as its "與官方 notebook 的差異" cell and config.json).
"""
import difflib
import hashlib
import json
import os

ROOT = r"D:\AI-Portfolio\competitions\aicup2026-go"
OFFICIAL = "third_party/aicup2026_tutorial/player-identification.ipynb"
OURS = "colab/prereg01_player.ipynb"
OUT = os.path.join(ROOT, "results", "prereg01", "notebook_diff_player.txt")
KIND = {"code": "code", "markdown": "md"}


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def sha(rel):
    with open(os.path.join(ROOT, rel), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def lines(src):
    text = "".join(src) if isinstance(src, list) else src
    if text and not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


off = load(OFFICIAL)
ours = load(OURS)
deviations = ours["metadata"]["c10"]["deviations"]

out = []
out.append("官方 notebook 與 C10 版本的逐格 unified diff（事前登記 01，競賽一）\n")
out.append("官方：%s  sha256 %s\n" % (OFFICIAL, sha(OFFICIAL)))
out.append("本版：%s  sha256 %s\n" % (OURS, sha(OURS)))
out.append("格號從 0 起算，含 markdown 格。code 格與 markdown 格都比。\n")
out.append("每格標記：identical＝和官方逐字相同（排版也相同）；modified＝改自官方；new＝本版新增（整格都是 +）。\n")
out.append("差異分類：⑤ ⑥＝事前登記的差異；營運性＝只改執行環境、存檔與接續，不改模型、特徵、超參數。\n")

out.append("\n差異清單（和 notebook 的「與官方 notebook 的差異」格、config.json 同一份，共 %d 項）：\n" % len(deviations))
out.extend("  %d. %s\n" % (n, d) for n, d in enumerate(deviations, 1))

summary = []
body = []
seen = []
counts = {}
for j, c in enumerate(ours["cells"]):
    meta = c["metadata"]["c10"]
    i = meta["official_cell"]
    ctype = KIND[c["cell_type"]]
    tags = "、".join(meta.get("tags") or []) or "-"
    if i is None:
        a, fa, kind = [], "/dev/null", "new"
    else:
        seen.append(i)
        assert off["cells"][i]["cell_type"] == c["cell_type"], (i, j)
        a, fa = lines(off["cells"][i]["source"]), "官方 cell %d" % i
        kind = "identical" if a == lines(c["source"]) else "modified"
        assert kind == meta["change"], (j, kind, meta["change"])
    counts[(ctype, kind)] = counts.get((ctype, kind), 0) + 1
    b = lines(c["source"])
    fb = "本版 cell %d" % j
    src_label = "官方 cell %d" % i if i is not None else "（新增）"
    extra = "  分類：%s" % tags if ctype == "code" else ""
    summary.append(("  本版 cell %-3d %-4s ← %-12s %-9s%s" % (j, ctype, src_label, kind, extra)).rstrip() + "\n")
    body.append("\n" + "=" * 78 + "\n")
    body.append("本版 cell %d（%s）← %s ｜ %s%s\n" % (j, ctype, fa, kind,
                                                  " ｜ 分類：%s" % tags if ctype == "code" else ""))
    body.append("=" * 78 + "\n")
    if kind == "identical":
        body.append("（和官方逐字相同）\n")
        continue
    diff = list(difflib.unified_diff(a, b, fromfile=fa, tofile=fb, n=3))
    body.extend(d if d.endswith("\n") else d + "\n" for d in diff)

out.append("\n對照表（本版每一格 → 官方格）：\n")
out.extend(summary)
out.append("小計：%s\n" % "；".join("%s %s %d 格" % (t, k, n) for (t, k), n in sorted(counts.items())))
n_off = len(off["cells"])
if seen == list(range(n_off)):
    out.append("官方的 %d 格（code %d、markdown %d）在本版都恰好對應一次，相對順序和官方相同，沒有拿掉任何一格。\n"
               % (n_off, sum(1 for c in off["cells"] if c["cell_type"] == "code"),
                  sum(1 for c in off["cells"] if c["cell_type"] == "markdown")))
else:
    missing = sorted(set(range(n_off)) - set(seen))
    out.append("注意：官方格對應不完整或順序不同。拿掉的格：%s；對應順序：%s\n" % (missing, seen))
out.extend(body)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(out)
print("wrote", OUT, sum(1 for _ in out))
