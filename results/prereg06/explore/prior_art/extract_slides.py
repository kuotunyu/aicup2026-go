"""Extract per-slide text from the organizer's 2026 course slides (.pptx).

Usage (Git Bash, repo root):
  .venv/Scripts/python.exe results/prereg06/explore/prior_art/extract_slides.py <slides.pptx> <out.txt>

The .pptx is the public course deck linked from aicup.tw
(https://www.aicup.tw/_files/ugd/7fbdbf_dee0824b36cb4dd2a4af2497347a04f7.pptx).
It is not stored in the repo; download it to the scratchpad first.
Only the standard library is used (zipfile + xml).
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

NS_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def slide_texts(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        names.sort(key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
        for n in names:
            root = ET.fromstring(z.read(n))
            paras = []
            for p in root.iter(NS_A + "p"):
                t = "".join(x.text or "" for x in p.iter(NS_A + "t")).strip()
                if t:
                    paras.append(t)
            num = int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1))
            yield num, paras


def main():
    src, out = sys.argv[1], sys.argv[2]
    with open(out, "w", encoding="utf-8") as f:
        for num, paras in slide_texts(src):
            f.write(f"=== slide {num}\n")
            for t in paras:
                f.write(t + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
