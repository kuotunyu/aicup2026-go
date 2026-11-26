"""Shared helpers of the repro/ entry points (C10, AI CUP 2026 Go). Nothing here computes a prediction."""
import hashlib
import json
import os
import sys
import tempfile

CR, LF = bytes([13]), bytes([10])


def repo_root(arg=None):
    """Repository root: --root, else $C10_ROOT, else the parent of repro/."""
    root = arg or os.environ.get("C10_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.abspath(root)
    for need in ("tools", "results", "data"):
        if not os.path.isdir(os.path.join(root, need)):
            sys.exit(f"ERROR: {root} is not the C10 repository root (no {need}/); pass --root or set C10_ROOT")
    return root


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_lf(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(CR + LF, LF)).hexdigest()


def same_lf(a, b):
    """Equal after folding CRLF to LF (a git working copy of a text file may be CRLF on Windows)."""
    return sha256_lf(a) == sha256_lf(b)


def out_dir(arg, root, name):
    """Output folder; must be outside the repository (the frozen tools refuse a rehearsal folder inside it)."""
    d = os.path.abspath(arg) if arg else tempfile.mkdtemp(prefix=f"c10_repro_{name}_")
    try:
        inside = os.path.commonpath([os.path.normcase(d), os.path.normcase(root)]) == os.path.normcase(root)
    except ValueError:                                        # different drives (Windows)
        inside = False
    if inside:
        sys.exit(f"ERROR: --out {d} is inside the repository; choose a folder outside it")
    os.makedirs(d, exist_ok=True)
    return d


def verdict(label, got_path, want_sha, ref_path=None):
    """PASS = the produced file's raw SHA-256 is the uploaded file's."""
    got = sha256(got_path)
    ok = got == want_sha
    print(f"{label}: {got_path}", f"  sha256   {got}", f"  expected {want_sha}", sep=os.linesep, flush=True)
    if ref_path and os.path.exists(ref_path):
        print(f"  stored copy {ref_path}: {'same bytes' if sha256(ref_path) == got else 'DIFFERENT bytes'}")
    print(f"  {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok and ref_path and os.path.exists(ref_path):
        d = rows_diff(got_path, ref_path)
        print(f"  rows different from the uploaded file: {d['rows_different']} "
              f"(produced {d['rows_produced']} rows with the header, uploaded {d['rows_uploaded']})")
        for i, x, y in d["first_differences"]:
            print(f"    row {i}: produced {x}  uploaded {y}")
    return ok


def rows_diff(got_path, ref_path, first=10):
    """Row-by-row comparison of two CSV files (CRLF folded)."""
    import csv
    import io

    def rows(p):
        with open(p, "rb") as f:
            return list(csv.reader(io.StringIO(f.read().replace(CR + LF, LF).decode("utf-8"))))
    a, b = rows(got_path), rows(ref_path)
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {"rows_different": len(diff) + abs(len(a) - len(b)), "rows_produced": len(a), "rows_uploaded": len(b),
            "first_differences": diff[:first]}


def final_record(got_path, want_sha, ref_path):
    """What tolerant_report.json keeps about the final CSV."""
    if not os.path.isfile(got_path):
        return {"produced": False, "expected_sha256": want_sha, "pass": False}
    got = sha256(got_path)
    rec = {"produced": True, "sha256": got, "expected_sha256": want_sha, "pass": got == want_sha}
    if got != want_sha and os.path.exists(ref_path):
        rec.update(rows_diff(got_path, ref_path))
    return rec


def blas_one_thread_posix():
    """Linux / macOS only: numpy's bundled OpenBLAS limited to 1 thread at run time (no environment variable).
    On Windows the registered competition-1 programs do this themselves (p38_backend.polite -> blas_threads(1),
    which looks for the Windows DLL only and is a no-op elsewhere); this is the same call on the .so, so that the
    reproduction runs with the thread count of the original run. Returns the library used, or None."""
    if os.name == "nt":
        return None
    import ctypes
    import glob
    import numpy as np
    pats = [os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs", "*openblas*"),
            os.path.join(os.path.dirname(np.__file__), ".dylibs", "*openblas*")]
    for lib in sorted(x for pat in pats for x in glob.glob(pat)):
        try:
            dll = ctypes.CDLL(lib)
        except OSError:
            continue
        for name in ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads",
                     "openblas_set_num_threads64_", "openblas_set_num_threads"):
            f = getattr(dll, name, None)
            if f is not None:
                f(ctypes.c_int(1))
                return os.path.basename(lib) + ":" + name
    return None


# ---------------------------------------------------------------- optional: record which files the run reads
_SEEN = {}


def trace_start(root):
    """Record every file under the repository opened for reading from now on (sys audit hook; nothing is changed)."""
    rootn = os.path.normcase(os.path.abspath(root)) + os.sep

    def hook(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
        if any(c in mode for c in "wax+"):
            return
        try:
            p = os.path.abspath(os.fsdecode(args[0]))
        except (TypeError, ValueError):
            return
        if os.path.normcase(p).startswith(rootn):
            _SEEN.setdefault(os.path.normcase(p), p)

    sys.addaudithook(hook)


def trace_write(root, out_json):
    """Write the recorded input files (relative path, bytes, SHA-256). Program files (.py), the virtual
    environments and .git are left out; anything under data/local is never hashed (and none is expected)."""
    rows, skipped = [], []
    for p in sorted(_SEEN.values()):
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        if rel.startswith((".venv", ".git/")) or rel.endswith((".py", ".pyc")) or not os.path.isfile(p):
            continue
        if rel.startswith("data/local"):
            skipped.append(rel)
            continue
        rows.append({"path": rel, "bytes": os.path.getsize(p), "sha256": sha256(p)})
    with open(out_json, "w", encoding="utf-8", newline="") as f:
        json.dump({"files": rows, "total_bytes": sum(r["bytes"] for r in rows),
                   "data_local_opened_not_hashed": skipped}, f, indent=1)
    print(f"inputs: {len(rows)} files, {sum(r['bytes'] for r in rows):,} bytes -> {out_json}")
