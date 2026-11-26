"""Public repository only: put the layer-1 inputs in place and commit the ones the frozen tools need committed.

    python repro/stage_inputs.py --zips-dir DIR --official-dir DIR2 [--root REPO] [--skip-zip-sha] [--check-only]

The public repository does not hold the official game records, the files computed per game / per question from
them, or the model files. Layer 1 (repro/comp1_from_features.py, repro/comp2_from_features.py) needs them at their
repository paths. They come from two places:

A. --official-dir: the OFFICIAL TEST FILES, which you download yourself from the two AIdea competition pages
   ("資料" tab; a participant account is needed). They may not be redistributed, so no package of this project
   holds them. Put the two downloaded zips, or the CSVs extracted from them, anywhere under one folder:

       aicup2026_player_identification_v2_tests_20261108_fix_readme.zip   (competition 1, v2 test set)
           player_identification_{candidates,candidates2,test,test2}_{public,private}.csv     8 files
       aicup2026_rank_prediction_tests_20261030.zip                       (competition 2)
           rank_prediction_test_{public,private}.csv                                          2 files

   The 10 CSVs are found by file name (in the folder, its subfolders, or inside any .zip there) and accepted only
   when bytes and SHA-256 equal repro/inputs_public.json (which also records their MD5).

B. --zips-dir: three packages from the cloud folder named in the top-level README (none holds an official file):

       C10_public_features_comp1.zip    897 files under data/     (4.6 GB)
       C10_public_features_comp2.zip     56 files under data/
       C10_features_layer1_ingit.zip     67 files under results/ and repro/ref_lda_L.npz

What this script does, in order (nothing here computes a prediction):

1. reads repro/inputs_public.json (every layer-1 input: path, bytes, SHA-256, and where it comes from);
2. checks the 54 inputs that are part of the repository itself;
3. official files: copies the 10 CSVs to data/test_v2_download/..., then rebuilds the 6 files that the original run
   had made from them and checks each SHA-256:
     data/prereg56/src_rank/player_identification_test2_<pool>.csv        byte copy of rank_prediction_test_<pool>.csv
     data/prereg56/src_rank/player_identification_candidates2_<pool>.csv  the header line only
     data/test/rank_prediction_test_<pool>.csv     the same file with the root HD[n] of every game removed
                                                   (what tools/root_props.py did on the test day, PREREG_22)
4. checks each package's size and SHA-256 (--skip-zip-sha: sizes only; every member is hashed in step 5 anyway);
5. extracts each package into the repository root, hashing every member while it is written; a member that is
   not in the list, or whose bytes / SHA-256 differ from the list, stops the run. Nothing under data/local is ever
   written or read;
6. `git add -f` + `git commit` of the 67 files of C10_features_layer1_ingit.zip, in THIS clone only.
   Why: the frozen competition-2 tools (tools/t25_rank.py require_committed, called by tools/t69_rank.py even in its
   rehearsal mode) refuse to run unless these inputs are tracked by git and unchanged against HEAD. The private
   development repository tracks them; the public one leaves them out, so they are committed locally instead.
   The files under data/ (the official files among them) stay untracked: data/ is in .gitignore.
   DO NOT PUSH this commit: it holds files derived from the official game records.
7. checks that the working tree has no changes to tracked files (`git status --porcelain --untracked-files=no`).

Run it again any time: files already in place with the right SHA-256 are not rewritten, and nothing is committed
when there is nothing new. --check-only does steps 1, 2 and a SHA-256 check of the files already in place, and
changes nothing.

The clone must keep git's bytes: `git clone -c core.autocrlf=false ...` (this script sets core.autocrlf=false in
the clone's own configuration and stops if a tracked input was checked out with other line endings).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
LIST = "repro/inputs_public.json"
SUPP = "C10_features_layer1_ingit.zip"
COMMIT_MESSAGE = "restore layer-1 inputs (local only; do not push)"
# identity of the local commit (fixed, so that no personal git identity is needed or recorded)
FALLBACK_IDENT = ["-c", "user.name=layer1-repro", "-c", "user.email=layer1-repro@localhost.invalid"]
HEADER_ONLY = b"player_id,game_id,sgf_content,color\n"
ROOT_HD = re.compile(rb"(\(;SZ\[19\])HD\[\d\]")


def die(msg):
    sys.exit("ERROR: " + msg)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(root, *args, check=True, ident=False):
    cmd = ["git", "-C", root] + (FALLBACK_IDENT if ident else []) + list(args)
    r = subprocess.run(cmd, capture_output=True)
    out = (r.stdout or b"").decode("utf-8", "replace").strip()
    if check and r.returncode != 0:
        die(f"git {' '.join(args[:3])}: {(r.stderr or b'').decode('utf-8', 'replace').strip()}")
    return r.returncode, out


def in_local(path):
    parts = [x.lower() for x in path.replace("\\", "/").split("/")]
    return any(a == "data" and b == "local" for a, b in zip(parts, parts[1:]))


def safe(rel):
    return not (rel.startswith(("/", "\\")) or ".." in rel.split("/") or ":" in rel or in_local(rel))


def in_place(root, r):
    p = os.path.join(root, *r["path"].split("/"))
    return os.path.isfile(p) and os.path.getsize(p) == r["bytes"] and sha_file(p) == r["sha256"]


def put(root, r, data):
    """Write one input (bytes already checked against the list)."""
    if len(data) != r["bytes"] or sha_bytes(data) != r["sha256"]:
        die(f"{r['path']}: bytes / SHA-256 differ from {LIST}")
    dst = os.path.join(root, *r["path"].split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".stage_tmp", "wb") as w:
        w.write(data)
    os.replace(dst + ".stage_tmp", dst)


def find_official(odir, rows):
    """{name: bytes} of the official CSVs found under odir (plain files or zip members), each equal to the list."""
    want = {r["name"]: r for r in rows}
    found, seen = {}, {n: [] for n in want}

    def offer(name, where, size, read):
        r = want.get(name)
        if r is None or name in found:
            return
        if size != r["bytes"]:
            seen[name].append(f"{where} ({size:,} bytes, expected {r['bytes']:,})")
            return
        data = read()
        if sha_bytes(data) != r["sha256"]:
            seen[name].append(f"{where} (SHA-256 {sha_bytes(data)[:16]}..., MD5 {hashlib.md5(data).hexdigest()})")
            return
        found[name] = data
        print(f"official: {name}  <- {where}  (SHA-256 OK)", flush=True)

    for base, dirs, names in os.walk(odir):
        dirs.sort()
        if in_local(base):
            die(f"{base}: a data/local folder is never read")
        for n in sorted(names):
            p = os.path.join(base, n)
            if n in want:
                offer(n, p, os.path.getsize(p), lambda p=p: open(p, "rb").read())
            elif n.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(p) as z:
                        for i in z.infolist():
                            if not i.is_dir():
                                offer(os.path.basename(i.filename), f"{p}!{i.filename}", i.file_size,
                                      lambda z=z, i=i: z.read(i))
                except zipfile.BadZipFile:
                    print(f"official: {p} is not a readable zip, skipped", flush=True)
    miss = [n for n in want if n not in found]
    if miss:
        lines = []
        for n in miss:
            r = want[n]
            lines.append(f"  {n}: expected {r['bytes']:,} bytes, SHA-256 {r['sha256']}, MD5 {r['md5']}"
                         + ("".join(f"\n      found but different: {s}" for s in seen[n])))
        die(f"{len(miss)} official test file(s) not found under {odir} (download them from the AIdea competition "
            f"pages, see the top of this script):\n" + "\n".join(lines))
    return found


def stage_official(root, files, odir):
    rows = [r for r in files.values() if r["from"] == "official"]
    der = [r for r in files.values() if r["from"] == "derived"]
    if all(in_place(root, r) for r in rows + der):
        print(f"official test files: {len(rows)} + {len(der)} rebuilt files already in place", flush=True)
        return
    if not odir:
        die("--official-dir is required: the folder with the official test files downloaded from AIdea")
    odir = os.path.abspath(odir)
    if not os.path.isdir(odir):
        die(f"{odir} is not a folder")
    found = find_official(odir, rows)
    by_path = {}
    for r in rows:
        by_path[r["path"]] = found[r["name"]]
        if not in_place(root, r):
            put(root, r, found[r["name"]])
    for r in der:
        if r["how"] == "header_only":
            data = HEADER_ONLY
        elif r["how"] == "copy":
            data = by_path[r["of"]]
        elif r["how"] == "strip_root_hd":
            data = ROOT_HD.sub(rb"\1", by_path[r["of"]])
        else:
            die(f"{r['path']}: unknown rule {r['how']}")
        if not in_place(root, r):
            put(root, r, data)
    print(f"official test files: {len(rows)} copied, {len(der)} rebuilt from them, all equal {LIST}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zips-dir", help="folder with the three packages of the cloud folder")
    ap.add_argument("--official-dir", help="folder with the official test files downloaded from AIdea (the two zips "
                                           "or the 10 CSVs, anywhere under it)")
    ap.add_argument("--root", help="repository root (default: the parent of repro/)")
    ap.add_argument("--skip-zip-sha", action="store_true", help="do not hash the whole zips (members are still hashed)")
    ap.add_argument("--check-only", action="store_true", help="verify what is in place; write and commit nothing")
    a = ap.parse_args()
    root = os.path.abspath(a.root or os.path.dirname(HERE))
    t0 = time.time()
    rc, top = git(root, "rev-parse", "--show-toplevel", check=False)
    if rc != 0 or os.path.normcase(os.path.abspath(top)) != os.path.normcase(root):
        die(f"{root} is not the top of a git working tree (clone the repository; a plain unzip cannot run "
            f"competition 2's tools)")
    with open(os.path.join(root, *LIST.split("/")), encoding="utf-8") as f:
        man = json.load(f)
    files = {r["path"]: r for r in man["files"]}
    bad = [p for p in files if not safe(p)]
    if bad:
        die(f"{LIST} lists unsafe paths: {bad[:5]}")
    zips = man["zips"]

    # ---- 2. inputs that are part of the repository (checked first: cheap, and tells a wrong clone at once)
    rc, out = git(root, "-c", "core.quotepath=false", "ls-files", "-z")
    tracked = set(out.split("\0")) - {""}
    repo_rows = [r for r in files.values() if r["from"] == "repository"]
    wrong = []
    for r in repo_rows:
        p = os.path.join(root, *r["path"].split("/"))
        if r["path"] not in tracked or not os.path.isfile(p):
            wrong.append((r["path"], "not in the repository"))
        elif sha_file(p) != r["sha256"]:
            wrong.append((r["path"], "bytes differ (line endings? clone with -c core.autocrlf=false)"))
    if wrong:
        die(f"{len(wrong)} repository inputs differ from {LIST}: {wrong[:5]}")
    print(f"repository inputs: {len(repo_rows)} files equal {LIST}", flush=True)

    if a.check_only:
        miss, diff = [], []
        for r in files.values():
            if r["from"] == "repository":
                continue
            p = os.path.join(root, *r["path"].split("/"))
            if not os.path.isfile(p):
                miss.append(r["path"])
            elif os.path.getsize(p) != r["bytes"] or sha_file(p) != r["sha256"]:
                diff.append(r["path"])
        print(f"check only: {len(files)} inputs listed; missing {len(miss)}, different {len(diff)}")
        for x in (miss + diff)[:20]:
            print("  ", x)
        return 0 if not (miss or diff) else 1

    if not a.zips_dir:
        die("--zips-dir is required")
    zdir = os.path.abspath(a.zips_dir)
    git(root, "config", "core.autocrlf", "false")

    # ---- 3. the official test files (user-supplied) and the files rebuilt from them
    stage_official(root, files, a.official_dir)

    # ---- 4. the packages
    for name, z in zips.items():
        p = os.path.join(zdir, name)
        if not os.path.isfile(p):
            die(f"{p} not found (download the three packages of the cloud folder into one folder)")
        if os.path.getsize(p) != z["bytes"]:
            die(f"{name}: {os.path.getsize(p):,} bytes, expected {z['bytes']:,} (incomplete download?)")
        if not a.skip_zip_sha:
            got = sha_file(p)
            if got != z["sha256"]:
                die(f"{name}: SHA-256 {got}, expected {z['sha256']}")
        print(f"{name}: {z['bytes']:,} bytes" + ("" if a.skip_zip_sha else ", SHA-256 OK"), flush=True)

    # ---- 5. extract, hashing every member
    written = kept = 0
    for name in zips:
        with zipfile.ZipFile(os.path.join(zdir, name)) as z:
            members = [i for i in z.infolist() if not i.is_dir()]
            want = {p for p, r in files.items() if r["from"] == name}
            got = {i.filename for i in members}
            if got != want:
                die(f"{name}: members differ from {LIST} (extra {sorted(got - want)[:3]}, missing {sorted(want - got)[:3]})")
            for i in members:
                rel = i.filename
                r = files[rel]
                if in_place(root, r):
                    kept += 1
                    continue
                dst = os.path.join(root, *rel.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                h, size, tmp = hashlib.sha256(), 0, dst + ".stage_tmp"
                with z.open(i) as src, open(tmp, "wb") as w:
                    for b in iter(lambda: src.read(1 << 22), b""):
                        h.update(b)
                        w.write(b)
                        size += len(b)
                if size != r["bytes"] or h.hexdigest() != r["sha256"]:
                    os.remove(tmp)
                    die(f"{name}: {rel} differs from {LIST} (bytes / SHA-256)")
                os.replace(tmp, dst)
                written += 1
        print(f"{name}: {len(members)} files in place", flush=True)
    print(f"extracted: wrote {written}, already in place {kept}", flush=True)
    if os.path.exists(os.path.join(root, "data", "local")):
        die("data/local exists in this tree; it must not (the delivery never contains it)")

    # ---- 6. commit the files the frozen tools need committed (this clone only)
    supp = sorted(p for p, r in files.items() if r["from"] == SUPP)
    for k in range(0, len(supp), 50):
        git(root, "add", "-f", "--", *supp[k:k + 50])
    rc, staged = git(root, "diff", "--cached", "--name-only", "-z")
    staged = [x for x in staged.split("\0") if x]
    other = sorted(set(staged) - set(supp))
    if other:
        die(f"other files are staged in this clone: {other[:5]}; unstage them first")
    if staged:
        git(root, "commit", "-q", "-m", COMMIT_MESSAGE, ident=True)
        print(f"committed {len(staged)} files locally: {git(root, 'rev-parse', '--short', 'HEAD')[1]} "
              f"\"{COMMIT_MESSAGE}\"", flush=True)
    else:
        print("nothing new to commit (the inputs are already committed in this clone)", flush=True)
    rc, out = git(root, "-c", "core.quotepath=false", "ls-files", "-z")
    tracked = set(out.split("\0"))
    miss = [p for p in supp if p not in tracked]
    if miss:
        die(f"not tracked after the commit: {miss[:5]}")
    leaked = sorted(p for p, r in files.items() if r["from"] in ("official", "derived") and p in tracked)
    if leaked:
        die(f"official test files are tracked by git in this clone: {leaked[:5]}")

    # ---- 7. clean tree, everything in place
    rc, st = git(root, "status", "--porcelain", "--untracked-files=no")
    if st:
        die("tracked files differ from HEAD:\n" + "\n".join(st.splitlines()[:10]))
    absent = [p for p in files if not os.path.isfile(os.path.join(root, *p.split("/")))]
    if absent:
        die(f"{len(absent)} inputs are missing: {absent[:5]}")
    print(f"working tree clean; {len(files)} layer-1 inputs staged in {time.time() - t0:.0f} s")
    print("Do NOT push the local commit. Next: repro/comp2_from_features.py and repro/comp1_from_features.py "
          "(--out outside the repository).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
