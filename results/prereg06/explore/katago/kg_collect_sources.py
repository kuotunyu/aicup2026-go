"""prereg06 explore / katago: record the public facts REPORT.md relies on (GitHub API reads only, no downloads of
weights or binaries). Writes results/prereg06/explore/katago/kg_sources.json.

Usage (repo root, Git Bash):
  .venv/Scripts/python.exe results/prereg06/explore/katago/kg_collect_sources.py
"""

import datetime
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.github.com/repos/lightvector/KataGo"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "c10-prereg06-katago", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    out = {"fetched_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "releases": {}}
    for tag in ["v1.15.0", "v1.15.3", "v1.16.0", "v1.17.2", "v1.18.0", "v1.18.1", "v1.18.2"]:
        rel = get(f"{API}/releases/tags/{tag}")
        keep = []
        for a in rel["assets"]:
            n = a["name"]
            if ("humanv" in n or n.endswith(".ckpt") or ("linux" in n and "bs" not in n)
                    or ("eigenavx2-windows" in n and "bs" not in n)):
                keep.append({"name": n, "bytes": a["size"], "url": a["browser_download_url"]})
        out["releases"][tag] = {"published_at": rel["published_at"], "name": rel["name"], "assets": keep}
    tag = get(f"{API}/git/refs/tags/v1.15.3")
    out["v1.15.3_commit"] = tag["object"]["sha"]
    out["issues"] = {}
    for n in [1009, 1030, 1041, 1074, 1077, 1154]:
        i = get(f"{API}/issues/{n}")
        out["issues"][str(n)] = {"title": i["title"], "state": i["state"], "created_at": i["created_at"],
                                 "url": i["html_url"]}
    with open(os.path.join(HERE, "kg_sources.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for tag, r in out["releases"].items():
        print(tag, r["published_at"], len(r["assets"]), "assets kept")
    print("v1.15.3 commit", out["v1.15.3_commit"])


if __name__ == "__main__":
    main()
