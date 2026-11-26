"""prereg06 explore / rules: extract text from the official course slides (.pptx).

Reads the pptx downloaded by fetch_sources.py (scratchpad), writes:
  - <scratch>/slides_all.txt        every slide's text + notes, numbered
  - slides_hits.json (next to this script)  slides 15 and 80, every slide
    matching the keyword list, with slide text and notes, plus
    the neighbouring slides' titles for context.

Usage (Git Bash, repo root):
    .venv/Scripts/python.exe results/prereg06/explore/rules/extract_slides.py
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
SCRATCH = Path(
    r"C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go"
    r"\a8d08794-7564-42a1-933e-30b4ef8b21ce\scratchpad\rules"
)
PPTX = SCRATCH / "slides_pptx.pptx"

NS_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
NS_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
NS_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
NS_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"

KEYWORDS = [
    "預訓練", "pretrain", "pre-train", "Pretrain", "從零", "從頭", "外部", "KataGo", "katago",
    "AI 工具", "AI工具", "生成式", "規則", "規定", "ChatGPT", "LLM", "Colab", "重現", "報告",
    "引擎", "Leela", "HumanSL", "human", "額外", "禁止", "不可", "不能", "允許", "可以用",
    "公平", "權重", "共享", "同時參加", "原創",
]


def para_texts(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    out = []
    for p in root.iter(f"{NS_A}p"):
        t = "".join((n.text or "") for n in p.iter(f"{NS_A}t"))
        if t.strip():
            out.append(t.strip())
    return out


def slide_order(z: zipfile.ZipFile) -> list[str]:
    pres = ET.fromstring(z.read("ppt/presentation.xml"))
    rels = ET.fromstring(z.read("ppt/_rels/presentation.xml.rels"))
    rid2target = {r.get("Id"): r.get("Target") for r in rels.iter(f"{NS_REL}Relationship")}
    order = []
    for sid in pres.iter(f"{NS_P}sldId"):
        tgt = rid2target[sid.get(f"{NS_R}id")]
        order.append("ppt/" + tgt.lstrip("/").replace("ppt/", "") if not tgt.startswith("/") else tgt.lstrip("/"))
    return order


def notes_for(z: zipfile.ZipFile, slide_path: str) -> list[str]:
    rel_path = slide_path.replace("slides/", "slides/_rels/") + ".rels"
    if rel_path not in z.namelist():
        return []
    rels = ET.fromstring(z.read(rel_path))
    for r in rels.iter(f"{NS_REL}Relationship"):
        if r.get("Type", "").endswith("/notesSlide"):
            tgt = r.get("Target")
            npath = "ppt/notesSlides/" + Path(tgt).name
            if npath in z.namelist():
                return [t for t in para_texts(z.read(npath)) if not re.fullmatch(r"\d+", t)]
    return []


def main() -> int:
    data = PPTX.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    z = zipfile.ZipFile(PPTX)
    order = slide_order(z)
    slides = []
    for i, sp in enumerate(order, start=1):
        slides.append({"n": i, "path": sp, "text": para_texts(z.read(sp)), "notes": notes_for(z, sp)})

    with (SCRATCH / "slides_all.txt").open("w", encoding="utf-8") as f:
        for s in slides:
            f.write(f"===== slide {s['n']} ({s['path']}) =====\n")
            f.write("\n".join(s["text"]) + "\n")
            if s["notes"]:
                f.write("--- notes ---\n" + "\n".join(s["notes"]) + "\n")

    hits = {}
    for s in slides:
        blob = "\n".join(s["text"] + s["notes"])
        kws = sorted({k for k in KEYWORDS if k.lower() in blob.lower()})
        if kws or s["n"] in (6, 15, 80, 95):
            prev_t = slides[s["n"] - 2]["text"][:1] if s["n"] > 1 else []
            next_t = slides[s["n"]]["text"][:1] if s["n"] < len(slides) else []
            hits[s["n"]] = {
                "keywords": kws,
                "text": s["text"],
                "notes": s["notes"],
                "prev_title": prev_t,
                "next_title": next_t,
            }
    out = {
        "pptx_sha256": sha,
        "pptx_bytes": len(data),
        "n_slides": len(slides),
        "titles": {s["n"]: (s["text"][0] if s["text"] else "") for s in slides},
        "hits": hits,
    }
    (HERE / "slides_hits.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("sha256", sha, "slides", len(slides), "hit slides", sorted(hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
