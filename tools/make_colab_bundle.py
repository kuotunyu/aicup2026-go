"""Pack everything a Colab run needs into one zip (never the validation answers).

Usage:
    python tools/make_colab_bundle.py data/colab_bundle/C10_bundle_v1.zip

Layout inside the zip (root folder C10_bundle_v1/):
    training/aicup2026_go_training.tar.xz   official archive, SHA-256 checked
    split_v1/players_split.csv
    tools/c10data.py
    tutorial/utils.py, tutorial/network.py  unmodified official helpers
    local/rank_val_questions.csv, local/player_val_questions.csv,
    local/player_val_candidates.csv          questions and candidates only
    bundle_manifest.json                     SHA-256 of every file above
"""

import hashlib
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_SHA256 = "959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b"
FILES = {
    "training/aicup2026_go_training.tar.xz": "data/aicup2026_go_training.tar.xz",
    "split_v1/players_split.csv": "results/split_v1/players_split.csv",
    "tools/c10data.py": "tools/c10data.py",
    "tutorial/utils.py": "third_party/aicup2026_tutorial/utils.py",
    "tutorial/network.py": "third_party/aicup2026_tutorial/network.py",
    "local/rank_val_questions.csv": "data/local/rank_val_questions.csv",
    "local/player_val_questions.csv": "data/local/player_val_questions.csv",
    "local/player_val_candidates.csv": "data/local/player_val_candidates.csv",
}
FORBIDDEN_SUBSTRINGS = ("answers",)


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def main(out_zip):
    for inside in FILES:
        if any(s in inside for s in FORBIDDEN_SUBSTRINGS):
            raise SystemExit(f"refusing to pack {inside}: answer files never go to Colab")
    manifest = {}
    for inside, src in FILES.items():
        path = os.path.join(ROOT, src)
        manifest[inside] = sha256(path)
    if manifest["training/aicup2026_go_training.tar.xz"] != ARCHIVE_SHA256:
        raise SystemExit("training archive does not match the official SHA-256")
    split_manifest = json.load(open(os.path.join(ROOT, "results/split_v1/manifest.json"), encoding="utf-8"))
    for name in ("rank_val_questions.csv", "player_val_questions.csv", "player_val_candidates.csv",
                 "players_split.csv"):
        inside = next(k for k in FILES if k.endswith(name))
        if manifest[inside] != split_manifest["output_sha256"][name]:
            raise SystemExit(f"{name} differs from results/split_v1/manifest.json; rebuild split v1 first")

    os.makedirs(os.path.dirname(os.path.abspath(out_zip)), exist_ok=True)
    top = "C10_bundle_v1"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for inside, src in FILES.items():
            method = zipfile.ZIP_STORED if inside.endswith(".xz") else zipfile.ZIP_DEFLATED
            z.write(os.path.join(ROOT, src), f"{top}/{inside}", compress_type=method)
        z.writestr(f"{top}/bundle_manifest.json",
                   json.dumps({"files": manifest, "split": "split_v1 rev 2"}, indent=1))
    print(json.dumps({"zip": out_zip, "bytes": os.path.getsize(out_zip), "zip_sha256": sha256(out_zip),
                      "files": manifest}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
