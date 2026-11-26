"""PREREG_04 rev 1: fingerprint of the R20 test-day path (git-tracked files only).
SHA-256 of each file's bytes with CRLF folded to LF, so a CRLF re-checkout does not change it.
Run from the repo root: .venv/Scripts/python.exe results/prereg04/design/evidence/rev1/r20_path_fingerprint.py
"""
import fnmatch, hashlib, json, subprocess, sys
PATTERNS = ["tools/t02_*.py", "tools/check_submission.py", "tools/official_log.py",
            "colab/*test02*", "results/test02/*", "docs/TESTDAY.md"]
files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
sel = sorted(f for f in files if f and any(fnmatch.fnmatch(f, p) or (p.endswith("/*") and f.startswith(p[:-1])) for p in PATTERNS))
out = {"what": "PREREG_04 rev 1 section 5.1: R20 test-day path fingerprint (CRLF folded to LF). tests/test_p04_hygiene.py fails if any file except tools/official_log.py changes.",
       "head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
       "patterns": PATTERNS, "n_files": len(sel),
       "files": {f: hashlib.sha256(open(f, "rb").read().replace(b"\r\n", b"\n")).hexdigest() for f in sel}}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/prereg04/design/r20_path_fingerprint.json", "w", encoding="utf-8"), indent=1)
print(out["n_files"], "files; head", out["head"])
