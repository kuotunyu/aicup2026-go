"""Build the layer-1 delivery package in data/delivery/ (C10, AI CUP 2026 Go). Nothing here computes a prediction.

    python repro/make_delivery.py code       # C10_code.zip (git archive of HEAD, raw git bytes) + C10_code_manifest.txt
    python repro/make_delivery.py bundle     # C10_code.bundle (git bundle of HEAD with its history) + C10_code_bundle_manifest.txt
    python repro/make_delivery.py features   # C10_features_layer1_comp{1,2}.zip (ZIP_STORED) + C10_features_manifest.json
    python repro/make_delivery.py sums       # C10_delivery_SHA256SUMS.txt + a copy of repro/colab_layer1.ipynb
    python repro/make_delivery.py public <PUBLIC_REPO>   # C10_public_features_comp{1,2}.zip (no official file),
                                             # C10_features_layer1_ingit.zip + its manifest,
                                             # repro/inputs_public.json, C10_public_SHA256SUMS.txt
    python repro/make_delivery.py split <FILE>...        # 7 MB parts + C10_public_parts_manifest.json (for upload)

code      `git -c core.autocrlf=false archive --format=zip HEAD`: the files as git stores them (LF), whatever the
          line endings of this Windows working copy. Every member is compared with its git blob. The archive is
          refused when the working tree has uncommitted changes to tracked files.
bundle    `git bundle create ... HEAD <branch>`: what the Colab notebook clones. The frozen competition-2 tools refuse
          to run unless their inputs are committed and clean in a git working tree (t25_rank.require_committed), so
          a plain unzip of C10_code.zip cannot run them; C10_code.zip is for reading.
features  the files listed in repro/inputs_comp1.json / inputs_comp2.json that are NOT tracked by git (all of them
          are under data/), at their repository paths. The listed files that ARE tracked come with C10_code.zip
          (git bytes). Each file is hashed while it is written and must equal the SHA-256 in the list.
          C10_features_manifest.json lists all the files of both lists with the bytes / SHA-256 as delivered and
          which zip carries them. Any path under data/local stops the build.
public    for the public repository (a single-commit snapshot without the per-game / per-question files and the
          model files). Official competition files may not be redistributed, so nothing written here holds one:
          C10_public_features_comp{1,2}.zip = the files of C10_features_layer1_comp{1,2}.zip minus the 10 official
          test CSVs and the 6 files made from them (OFFICIAL / DERIVED below); C10_features_layer1_ingit.zip = the
          layer-1 inputs that ARE tracked here but are left out of the public repository, plus repro/ref_lda_L.npz
          (ZIP_STORED, repository paths, git bytes of HEAD). Every packaged file is scanned for SGF move text.
          repro/inputs_public.json lists every layer-1 input with bytes / SHA-256 and where it comes from (a
          package, the public repository, 'official' = downloaded from AIdea by the user, 'derived' = rebuilt from
          the official files); repro/stage_inputs.py reads it. The older C10_features_layer1_comp{1,2}.zip hold
          the official test files and are for private use only.
split     cuts files of data/delivery/ into 7 MB parts under data/delivery/public_parts/parts/ and writes
          C10_public_parts_manifest.json there (the format repro/colab_public.ipynb reassembles).
data/local is never opened, listed or hashed.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "delivery")
CODE_ZIP = "C10_code.zip"
CODE_BUNDLE = "C10_code.bundle"
FEAT_ZIP = {"comp1": "C10_features_layer1_comp1.zip", "comp2": "C10_features_layer1_comp2.zip"}
LISTS = {"comp1": "repro/inputs_comp1.json", "comp2": "repro/inputs_comp2.json"}
# official game records must not be in the code zip (reported, never removed here)
SUSPECT_EXT = (".sgf", ".tar.xz", ".pt", ".pth")
SUSPECT_PREFIX = ("data/training", "data/test", "data/local", "data/prereg")


def git(*args, binary=False):
    r = subprocess.run(["git", "-C", ROOT] + list(args), capture_output=True)
    if r.returncode != 0:
        sys.exit(f"ERROR: git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tracked():
    return set(git("-c", "core.quotepath=false", "ls-files", "-z").split("\0")) - {""}


def blob_shas():
    """path -> (bytes, sha256) of every blob at HEAD, read through one `git cat-file --batch`."""
    names = [n for n in git("-c", "core.quotepath=false", "ls-tree", "-r", "-z", "--name-only", "HEAD").split("\0") if n]
    p = subprocess.Popen(["git", "-C", ROOT, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = {}
    for n in names:
        p.stdin.write(f"HEAD:{n}\n".encode("utf-8"))
        p.stdin.flush()
        head = p.stdout.readline().split()
        size = int(head[2])
        data = p.stdout.read(size)
        p.stdout.read(1)
        out[n] = (size, hashlib.sha256(data).hexdigest())
    p.stdin.close()
    p.wait()
    return out


def cmd_code():
    os.makedirs(OUT, exist_ok=True)
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        sys.exit("ERROR: tracked files have uncommitted changes; commit first\n" + dirty)
    head = git("rev-parse", "HEAD")
    dst = os.path.join(OUT, CODE_ZIP)
    git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", dst, "HEAD")
    blobs = blob_shas()
    bad, suspects, n, total = [], [], 0, 0
    with zipfile.ZipFile(dst) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        for i in members:
            data = z.read(i)
            n, total = n + 1, total + len(data)
            if blobs.get(i.filename, (None, None))[1] != hashlib.sha256(data).hexdigest():
                bad.append(i.filename)
            low = i.filename.lower()
            if low.endswith(SUSPECT_EXT) or low.startswith(SUSPECT_PREFIX):
                suspects.append(i.filename)
            elif low.endswith(".csv") and b"sgf" in data[:200].lower():
                suspects.append(i.filename + "  (csv with an sgf column)")
    missing = sorted(set(blobs) - {i.filename for i in members})
    if bad or missing:
        sys.exit(f"ERROR: archive differs from the git blobs: different {bad[:10]}, missing {missing[:10]}")
    lines = [f"C10_code.zip: git -c core.autocrlf=false archive --format=zip HEAD",
             f"commit   {head}", f"branch   {git('rev-parse', '--abbrev-ref', 'HEAD')}",
             f"files    {n}", f"bytes    {total} (uncompressed)", f"zip      {os.path.getsize(dst)} bytes",
             f"sha256   {sha_file(dst)}",
             f"check    every member equals its git blob at HEAD (SHA-256): {n} of {len(blobs)}",
             f"check    .sgf / weights / data/training, data/test*, data/local, data/prereg* / csv with an sgf column: "
             f"{len(suspects)} found"] + [f"  SUSPECT {s}" for s in suspects]
    with open(os.path.join(OUT, "C10_code_manifest.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_bundle():
    os.makedirs(OUT, exist_ok=True)
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        sys.exit("ERROR: tracked files have uncommitted changes; commit first\n" + dirty)
    head, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    dst = os.path.join(OUT, CODE_BUNDLE)
    if os.path.exists(dst):
        os.remove(dst)
    git("bundle", "create", dst, "HEAD", branch)
    git("bundle", "verify", dst)
    lines = [f"{CODE_BUNDLE}: git bundle create {CODE_BUNDLE} HEAD {branch}", f"commit   {head}", f"branch   {branch}",
             f"commits  {git('rev-list', '--count', 'HEAD')}", f"files    {len(tracked())} at HEAD",
             f"bundle   {os.path.getsize(dst)} bytes", f"sha256   {sha_file(dst)}",
             "use      git -c core.autocrlf=false clone C10_code.bundle <folder>   (then unzip the feature zips into it)"]
    with open(os.path.join(OUT, "C10_code_bundle_manifest.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_features():
    os.makedirs(OUT, exist_ok=True)
    tr, blobs, manifest = tracked(), None, {}
    for comp, lst in LISTS.items():
        with open(os.path.join(ROOT, lst), encoding="utf-8") as f:
            d = json.load(f)
        if d.get("data_local_opened_not_hashed"):
            sys.exit(f"ERROR: {lst} records data/local reads; stop")
        files = d["files"]
        local = [x["path"] for x in files if x["path"].replace("\\", "/").lower().startswith("data/local")
                 or "/local/" in x["path"].lower()]
        if local:
            sys.exit(f"ERROR: {lst} lists paths under data/local: {local[:5]}; stop")
        dst = os.path.join(OUT, FEAT_ZIP[comp])
        n = total = 0
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for x in files:
                rel = x["path"]
                rec = manifest.setdefault(rel, {"path": rel, "in": []})
                if rel in tr:                                  # comes with the code zip, git bytes
                    if blobs is None:
                        blobs = blob_shas()
                    size, sha = blobs[rel]
                    rec.update(bytes=size, sha256=sha)
                    if "C10_code.zip" not in rec["in"]:
                        rec["in"] += ["C10_code.zip", "C10_code.bundle"]
                    if sha != x["sha256"]:                     # CRLF working copy on Windows when the list was made
                        with open(os.path.join(ROOT, *rel.split("/")), "rb") as g:
                            lf = hashlib.sha256(g.read().replace(b"\r\n", b"\n")).hexdigest()
                        if lf != sha:
                            sys.exit(f"ERROR: {rel}: working copy differs from HEAD by more than line endings")
                        rec["windows_working_copy_sha256"] = x["sha256"]
                    continue
                src = os.path.join(ROOT, *rel.split("/"))
                h, size = hashlib.sha256(), 0
                zi = zipfile.ZipInfo(rel, date_time=(2026, 11, 25, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = 0o644 << 16
                zi.file_size = os.path.getsize(src)
                with open(src, "rb") as g, z.open(zi, "w", force_zip64=True) as w:
                    for b in iter(lambda: g.read(1 << 22), b""):
                        h.update(b)
                        w.write(b)
                        size += len(b)
                if h.hexdigest() != x["sha256"] or size != x["bytes"]:
                    sys.exit(f"ERROR: {rel} changed since {lst} was made (sha256 / bytes differ)")
                rec.update(bytes=size, sha256=x["sha256"])
                rec["in"].append(FEAT_ZIP[comp])
                n, total = n + 1, total + size
        print(f"{FEAT_ZIP[comp]}: {n} files, {total:,} bytes stored, zip {os.path.getsize(dst):,} bytes", flush=True)
    rows = [manifest[k] for k in sorted(manifest)]
    out = {"note": "every file of repro/inputs_comp1.json + inputs_comp2.json, with the bytes / SHA-256 as delivered "
                   "and the zip(s) that carry it; files tracked by git come with C10_code.bundle / C10_code.zip (git bytes, LF)",
           "files": rows, "n_files": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
           "zips": {name: {"bytes": os.path.getsize(os.path.join(OUT, name)), "sha256": sha_file(os.path.join(OUT, name)),
                           "n_files": sum(1 for r in rows if name in r["in"])} for name in FEAT_ZIP.values()}}
    with open(os.path.join(OUT, "C10_features_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["zips"], indent=1))


def cmd_sums():
    shutil.copyfile(os.path.join(ROOT, "repro", "colab_layer1.ipynb"), os.path.join(OUT, "colab_layer1.ipynb"))
    names = [CODE_BUNDLE, CODE_ZIP] + list(FEAT_ZIP.values()) + ["C10_features_manifest.json", "C10_code_manifest.txt",
                                                                 "C10_code_bundle_manifest.txt", "colab_layer1.ipynb"]
    with open(os.path.join(OUT, "C10_delivery_SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        for n in names:
            p = os.path.join(OUT, n)
            line = f"{sha_file(p)}  {n}  {os.path.getsize(p)}"
            f.write(line + "\n")
            print(line)


SUPP_ZIP = "C10_features_layer1_ingit.zip"
SUPP_EXTRA = ["repro/ref_lda_L.npz"]          # needed by --tolerant; a model file, so not in the public repository
PUB_ZIP = {"comp1": "C10_public_features_comp1.zip", "comp2": "C10_public_features_comp2.zip"}
PART = 7 * 1024 * 1024
# The official test files: never packaged for the public. The reviewer downloads them from AIdea (two downloads).
OFFICIAL_DOWNLOADS = {
    "player_v2": {"competition": "1 (player identification), v2 test set published 2026-11-11",
                  "aidea_zip": "aicup2026_player_identification_v2_tests_20261108_fix_readme.zip",
                  "local_zip": "data/test_v2_download/aicup2026_player_identification_v2_tests_20261108_fix_readme.zip"},
    "rank": {"competition": "2 (rank prediction)",
             "aidea_zip": "aicup2026_rank_prediction_tests_20261030.zip",
             "local_zip": "data/test_v2_download/aicup2026_rank_prediction_tests_20261030.zip"},
}
OFFICIAL = {f"data/test_v2_download/x/tests/player_identification_{k}_{p}.csv": "player_v2"
            for k in ("candidates", "candidates2", "test", "test2") for p in ("public", "private")}
OFFICIAL.update({f"data/test_v2_download/rank/rank_prediction_test_{p}.csv": "rank" for p in ("public", "private")})
# Files made from the official ones (byte copies under another name, the root HD[n] removed, a header-only file):
# never packaged either; repro/stage_inputs.py rebuilds them from the reviewer's official files and checks the SHA-256.
DERIVED = {}
for _p in ("public", "private"):
    _src = f"data/test_v2_download/rank/rank_prediction_test_{_p}.csv"
    DERIVED[f"data/test/rank_prediction_test_{_p}.csv"] = {"how": "strip_root_hd", "of": _src}
    DERIVED[f"data/prereg56/src_rank/player_identification_test2_{_p}.csv"] = {"how": "copy", "of": _src}
    DERIVED[f"data/prereg56/src_rank/player_identification_candidates2_{_p}.csv"] = {"how": "header_only"}
NEVER_PREFIX = ("data/test_v2_download/", "data/test/", "data/test_rootprops/", "data/training", "data/local")
SGF_RUN = None


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cmd_public():
    """The packages for the public repository (no official file inside) + repro/inputs_public.json."""
    import re
    global SGF_RUN
    SGF_RUN = re.compile(rb";[BW]\[[a-s]{2}\];[BW]\[[a-s]{2}\];[BW]\[[a-s]{2}\]")   # three consecutive SGF moves
    if len(sys.argv) < 3:
        sys.exit("usage: make_delivery.py public <PUBLIC_REPO>")
    pub = os.path.abspath(sys.argv[2])
    r = subprocess.run(["git", "-C", pub, "-c", "core.quotepath=false", "ls-files", "-s", "-z"], capture_output=True)
    if r.returncode != 0:
        sys.exit("ERROR: git ls-files in the public repository: " + r.stderr.decode("utf-8", "replace"))
    pub_blob = {}
    for e in r.stdout.decode("utf-8").split("\0"):
        if e:
            meta, name = e.split("\t", 1)
            pub_blob[name] = meta.split()[1]
    here_blob = {}
    for e in git("-c", "core.quotepath=false", "ls-tree", "-r", "-z", "HEAD").split("\0"):
        if e:
            meta, name = e.split("\t", 1)
            here_blob[name] = meta.split()[2]
    with open(os.path.join(OUT, "C10_features_manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    comp_of = {v: k for k, v in FEAT_ZIP.items()}
    blobs = blob_shas()
    rows, supp, packs = [], [], {"comp1": [], "comp2": []}
    for rec in man["files"]:
        rel = rec["path"]
        if rel.lower().startswith("data/local"):
            sys.exit(f"ERROR: {rel}: data/local in the manifest; stop")
        base = {"path": rel, "bytes": rec["bytes"], "sha256": rec["sha256"]}
        zips = [z for z in rec["in"] if z in FEAT_ZIP.values()]
        if rel in OFFICIAL:
            src = os.path.join(ROOT, *rel.split("/"))
            if sha_file(src) != rec["sha256"]:
                sys.exit(f"ERROR: {rel} changed")
            rows.append(dict(base, **{"from": "official", "md5": md5_file(src), "name": os.path.basename(rel),
                                      "download": OFFICIAL[rel]}))
        elif rel in DERIVED:
            rows.append(dict(base, **{"from": "derived"}, **DERIVED[rel]))
        elif zips:
            if rel.startswith(NEVER_PREFIX):
                sys.exit(f"ERROR: {rel}: an official-data path that is neither in OFFICIAL nor in DERIVED; stop")
            packs[comp_of[zips[0]]].append(rel)
            rows.append(dict(base, **{"from": PUB_ZIP[comp_of[zips[0]]]}))
        else:
            if blobs[rel] != (rec["bytes"], rec["sha256"]):
                sys.exit(f"ERROR: {rel}: HEAD blob differs from C10_features_manifest.json")
            if rel in pub_blob:
                if pub_blob[rel] != here_blob[rel]:
                    sys.exit(f"ERROR: {rel}: the public repository's copy differs from HEAD")
                rows.append(dict(base, **{"from": "repository"}))
            else:
                supp.append(rel)
                rows.append(dict(base, **{"from": SUPP_ZIP}))
    missing = sorted((set(OFFICIAL) | set(DERIVED)) - {r["path"] for r in rows})
    if missing:
        sys.exit(f"ERROR: expected official / derived inputs not in the manifest: {missing}")
    for rel in SUPP_EXTRA:
        if rel in pub_blob:
            sys.exit(f"ERROR: {rel} is in the public repository")
        size, sha = blobs[rel]
        supp.append(rel)
        rows.append({"path": rel, "bytes": size, "sha256": sha, "from": SUPP_ZIP, "only_for": "--tolerant"})
    by = {r["path"]: r for r in rows}

    def scan(data, tail, rel, hits):
        if SGF_RUN.search(tail + data):
            hits.append(rel)
        return data[-64:]

    zrecs, sgf_hits = {}, []
    # ---- the two feature packages: files that are not tracked by git, read from this working tree
    for comp, rels in packs.items():
        dst = os.path.join(OUT, PUB_ZIP[comp])
        total = 0
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for rel in rels:
                src = os.path.join(ROOT, *rel.split("/"))
                h, size, tail, hit = hashlib.sha256(), 0, b"", []
                zi = zipfile.ZipInfo(rel, date_time=(2026, 11, 26, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = 0o644 << 16
                zi.file_size = os.path.getsize(src)
                with open(src, "rb") as g, z.open(zi, "w", force_zip64=True) as w:
                    for b in iter(lambda: g.read(1 << 22), b""):
                        h.update(b)
                        w.write(b)
                        size += len(b)
                        if not hit:
                            tail = scan(b, tail, rel, hit)
                sgf_hits += hit
                if h.hexdigest() != by[rel]["sha256"] or size != by[rel]["bytes"]:
                    sys.exit(f"ERROR: {rel} changed since the manifest was made (sha256 / bytes differ)")
                total += size
        zrecs[PUB_ZIP[comp]] = {"bytes": os.path.getsize(dst), "sha256": sha_file(dst), "n_files": len(rels),
                                "stored_bytes": total}
        print(f"{PUB_ZIP[comp]}: {len(rels)} files, {total:,} bytes stored", flush=True)
    # ---- the supplement: tracked here, left out of the public repository; git bytes of HEAD
    dst = os.path.join(OUT, SUPP_ZIP)
    p = subprocess.Popen(["git", "-C", ROOT, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    total = 0
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for rel in sorted(supp):
            p.stdin.write(f"HEAD:{rel}\n".encode("utf-8"))
            p.stdin.flush()
            size = int(p.stdout.readline().split()[2])
            data = p.stdout.read(size)
            p.stdout.read(1)
            if hashlib.sha256(data).hexdigest() != by[rel]["sha256"] or size != by[rel]["bytes"]:
                sys.exit(f"ERROR: {rel}: HEAD blob differs from the manifest")
            if rel.startswith(NEVER_PREFIX):
                sys.exit(f"ERROR: {rel}: official-data path in the supplement; stop")
            scan(data, b"", rel, sgf_hits)
            zi = zipfile.ZipInfo(rel, date_time=(2026, 11, 26, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
            total += size
    p.stdin.close()
    p.wait()
    zrecs[SUPP_ZIP] = {"bytes": os.path.getsize(dst), "sha256": sha_file(dst), "n_files": len(supp), "stored_bytes": total}
    if sgf_hits:
        sys.exit(f"ERROR: SGF move text (three consecutive moves) found in packaged files: {sgf_hits[:10]}; "
                 f"the packages were written but must not be published")
    # ---- check: no member of any public package is an official / derived file or under an official-data folder
    for name in zrecs:
        with zipfile.ZipFile(os.path.join(OUT, name)) as z:
            bad = [i.filename for i in z.infolist() if i.filename in OFFICIAL or i.filename in DERIVED
                   or i.filename.startswith(NEVER_PREFIX) or i.filename.lower().endswith((".sgf", ".tar.xz"))]
        if bad:
            sys.exit(f"ERROR: {name} holds official files: {bad[:5]}")
    ext = {}
    for rel in supp:
        e = os.path.splitext(rel)[1] or "(none)"
        ext[e] = ext.get(e, 0) + 1
    smf = {"note": "layer-1 inputs that the private development repository tracks but the public repository leaves "
                   "out (per-game / per-question outputs and model files), plus repro/ref_lda_L.npz (--tolerant); "
                   "repository paths, git bytes (LF) of the commit below",
           "commit": git("rev-parse", "HEAD"), "zip": {SUPP_ZIP: zrecs[SUPP_ZIP]}, "by_extension": ext,
           "files": [by[rel] for rel in sorted(supp)]}
    with open(os.path.join(OUT, "C10_features_layer1_ingit_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(smf, f, indent=1)
    downloads = {}
    for key, d in OFFICIAL_DOWNLOADS.items():
        rec = {"competition": d["competition"], "aidea_zip": d["aidea_zip"],
               "files": sorted(os.path.basename(p_) for p_, k in OFFICIAL.items() if k == key)}
        lz = os.path.join(ROOT, *d["local_zip"].split("/"))
        if os.path.isfile(lz):                                  # the zip as downloaded from AIdea on 2026-11-11
            rec.update(zip_bytes=os.path.getsize(lz), zip_md5=md5_file(lz), zip_sha256=sha_file(lz))
            with zipfile.ZipFile(lz) as z:
                inside = {os.path.basename(i.filename): hashlib.sha256(z.read(i)).hexdigest()
                          for i in z.infolist() if not i.is_dir()}
            for p_, k in OFFICIAL.items():
                if k == key and inside.get(os.path.basename(p_)) != by[p_]["sha256"]:
                    sys.exit(f"ERROR: {os.path.basename(p_)} inside {d['aidea_zip']} is not the file the run read")
            rec["zip_members_checked"] = True
        downloads[key] = rec
    zips = {name: {k: v[k] for k in ("bytes", "sha256", "n_files")} for name, v in zrecs.items()}
    rows.sort(key=lambda r_: r_["path"])
    kinds = list(zips) + ["repository", "official", "derived"]
    pubj = {"note": "every input of the two layer-1 entry points (repro/inputs_comp1.json + inputs_comp2.json) and "
                    "repro/ref_lda_L.npz: bytes / SHA-256 and where the file comes from. 'official': the official "
                    "test files, NOT distributed here, to be downloaded from AIdea by the user (name, SHA-256, MD5); "
                    "'derived': rebuilt from the official files by repro/stage_inputs.py; a zip name: one of the "
                    "three packages of the cloud folder (none holds an official file); 'repository': this repository.",
            "zips": zips, "official_downloads": downloads, "n_files": len(rows),
            "total_bytes": sum(r_["bytes"] for r_ in rows),
            "from": {k: sum(1 for r_ in rows if r_["from"] == k) for k in kinds}, "files": rows}
    with open(os.path.join(ROOT, "repro", "inputs_public.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(pubj, f, indent=1)
    with open(os.path.join(OUT, "C10_public_SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        for n in list(zrecs) + ["C10_features_layer1_ingit_manifest.json"]:
            q = os.path.join(OUT, n)
            f.write(f"{sha_file(q)}  {n}  {os.path.getsize(q)}\n")
    print(json.dumps({"zips": zrecs, "supplement_by_extension": ext, "inputs_public": pubj["from"],
                      "n_files": pubj["n_files"], "official_downloads": downloads}, indent=1))


def cmd_split():
    """7 MB parts of the named files of data/delivery/ + C10_public_parts_manifest.json."""
    dst = os.path.join(OUT, "public_parts")
    os.makedirs(os.path.join(dst, "parts"), exist_ok=True)
    mp = os.path.join(dst, "C10_public_parts_manifest.json")
    man = {}
    if os.path.isfile(mp):
        with open(mp, encoding="utf-8") as f:
            man = json.load(f)
    for name in sys.argv[2:]:
        full, parts = hashlib.sha256(), []
        with open(os.path.join(OUT, name), "rb") as f:
            for i, b in enumerate(iter(lambda: f.read(PART), b"")):
                pn = f"{name}.part{i:04d}"
                with open(os.path.join(dst, "parts", pn), "wb") as w:
                    w.write(b)
                full.update(b)
                parts.append([pn, len(b), hashlib.sha256(b).hexdigest()])
        man[name] = {"bytes": sum(x[1] for x in parts), "sha256": full.hexdigest(), "parts": parts}
        print(f"{name}: {len(parts)} parts, sha256 {full.hexdigest()}", flush=True)
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)


if __name__ == "__main__":
    {"code": cmd_code, "bundle": cmd_bundle, "features": cmd_features, "sums": cmd_sums,
     "public": cmd_public, "split": cmd_split}[sys.argv[1] if len(sys.argv) > 1 else ""]()
