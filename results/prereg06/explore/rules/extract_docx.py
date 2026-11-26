"""prereg06 explore / rules: plain-text extract of the AI CUP registration-system
user terms (.docx), downloaded by fetch_sources.py into the scratchpad.

Usage (Git Bash, repo root):
    .venv/Scripts/python.exe results/prereg06/explore/rules/extract_docx.py
Writes <scratch>/goaicup_terms.txt and prints it.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

SCRATCH = Path(
    r"C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go"
    r"\a8d08794-7564-42a1-933e-30b4ef8b21ce\scratchpad\rules"
)
SRC = SCRATCH / "goaicup_terms_docx.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def main() -> int:
    data = SRC.read_bytes()
    root = ET.fromstring(zipfile.ZipFile(SRC).read("word/document.xml"))
    paras = []
    for p in root.iter(f"{W}p"):
        t = "".join((n.text or "") for n in p.iter(f"{W}t")).strip()
        if t:
            paras.append(t)
    out = SCRATCH / "goaicup_terms.txt"
    out.write_text("\n".join(f"[{i}] {t}" for i, t in enumerate(paras, 1)), encoding="utf-8")
    print("sha256", hashlib.sha256(data).hexdigest(), "paragraphs", len(paras))
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
