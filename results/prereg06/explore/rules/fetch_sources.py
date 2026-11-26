"""prereg06 explore / rules: fetch public rule sources (read-only, no login).

Downloads raw pages into the session scratchpad (never the repo), records
HTTP status, size and SHA-256 in fetch_log.json next to this script, and
writes plain-text extracts (scratchpad) for grepping.

Usage (Git Bash, repo root):
    .venv/Scripts/python.exe results/prereg06/explore/rules/fetch_sources.py
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path(
    r"C:\Users\USER\AppData\Local\Temp\claude\D--AI-Portfolio-competitions-aicup2026-go"
    r"\a8d08794-7564-42a1-933e-30b4ef8b21ce\scratchpad\rules"
)
SCRATCH.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "aidea_comp2": "https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403",
    "aidea_comp1": "https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688",
    "aidea_faq": "https://www.aidea-web.tw/faq",
    "goaicup_38": "https://go.aicup.tw/competition/info/38",
    "goaicup_39": "https://go.aicup.tw/competition/info/39",
    "goaicup_list": "https://go.aicup.tw/competition/",
    "slides_pptx": "https://www.aicup.tw/_files/ugd/7fbdbf_dee0824b36cb4dd2a4af2497347a04f7.pptx",
    "tutorial_readme": "https://raw.githubusercontent.com/AILAB-NDHU/AICup-2026-Tutorial/main/README.md",
    "tutorial_readme_en": "https://raw.githubusercontent.com/AILAB-NDHU/AICup-2026-Tutorial/main/README.en.md",
    "dataset_readme": "https://raw.githubusercontent.com/AILAB-NDHU/aicup2026-go-dataset/main/README.md",
    "tutorial_commits": "https://api.github.com/repos/AILAB-NDHU/AICup-2026-Tutorial/commits?per_page=5",
    "dataset_commits": "https://api.github.com/repos/AILAB-NDHU/aicup2026-go-dataset/commits?per_page=5",
    # second pass (added after the first run)
    "aidea_comp2_en": "https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403?lang=en",
    "aidea_comp1_en": "https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688?lang=en",
    "aicup_2026_page": "https://www.aicup.tw/ai-cup-2026-competition",
    "aicup_report_system": "https://aicupreport.csie.ncu.edu.tw",
    "aidea_moe2026": "https://www.aidea-web.tw/moe2026",
    "tutorial_2023_readme": "https://raw.githubusercontent.com/AILAB-NDHU/AIcupTutorial/main/README.md",
    # third pass: AI CUP registration-system user terms (linked from the go.aicup.tw login box)
    "goaicup_terms_docx": "https://go.aicup.tw/static/docs/%E4%BD%BF%E7%94%A8%E8%80%85%E6%A2%9D%E6%AC%BE.docx",
    # fourth pass: KataGo HumanSL facts (licence, training-data sources) for the risk section
    "katago_sgfmetadata": "https://raw.githubusercontent.com/lightvector/KataGo/master/python/katago/game/sgfmetadata.py",
    "katago_license": "https://raw.githubusercontent.com/lightvector/KataGo/master/LICENSE",
    "katago_network_license": "https://katagotraining.org/network_license/",
    "katago_release_1150": "https://api.github.com/repos/lightvector/KataGo/releases/tags/v1.15.0",
    # fifth pass: sister AI CUP 2026 competition on AIdea (solar panels), for wording contrast only
    "aidea_solar": "https://www.aidea-web.tw/topic/3c8ec48b-0dff-4b14-9b5e-d91bbb98c699",
}

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
MAX_BYTES = 50 * 1024 * 1024


def fetch(url: str) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "zh-TW,zh;q=0.9"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise RuntimeError("over 50 MB, aborted")
            return r.status, data, r.geturl()
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        return e.code, e.read()[:200000], url


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>|</tr>|</div>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    lines = [re.sub(r"[ \t\u3000]+", " ", ln).strip() for ln in raw.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def main() -> int:
    only = set(sys.argv[1:])
    log_path = HERE / "fetch_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    for key, url in SOURCES.items():
        if only and key not in only:
            continue
        t0 = time.time()
        try:
            status, data, final = fetch(url)
        except Exception as e:  # noqa: BLE001
            log[key] = {"url": url, "error": repr(e), "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            print(key, "ERROR", e)
            continue
        if url.endswith(".pptx"):
            ext = ".pptx"
        elif url.endswith(".docx"):
            ext = ".docx"
        elif "api.github.com" in url:
            ext = ".json"
        elif url.endswith(".md"):
            ext = ".md"
        else:
            ext = ".html"
        out = SCRATCH / f"{key}{ext}"
        out.write_bytes(data)
        entry = {
            "url": url,
            "final_url": final,
            "status": status,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "saved_to": str(out),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seconds": round(time.time() - t0, 2),
        }
        if ext == ".html":
            txt = html_to_text(data.decode("utf-8", "replace"))
            tpath = SCRATCH / f"{key}.txt"
            tpath.write_text(txt, encoding="utf-8")
            entry["text_chars"] = len(txt)
            entry["text_saved_to"] = str(tpath)
        log[key] = entry
        print(key, status, len(data), entry.get("text_chars", ""))
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
