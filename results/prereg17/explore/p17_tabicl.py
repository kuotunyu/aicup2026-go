"""prereg17 exploration (competition 2): TabICLv2 in-context predictions for the inner games (PLAN.md 2.2).

Runs ONLY in the separate environment C:/Users/USER/venvs/c10-tabicl with `python -I` (never .venv / .venv-gpu):
    PYTHONDONTWRITEBYTECODE=1 C:/Users/USER/venvs/c10-tabicl/Scripts/python.exe -I \
        results/prereg17/explore/p17_tabicl.py run --feat-dir data/prereg17/feat --out-dir data/prereg17/tabicl
    (smoke: --feat-dir data/prereg17/smoke/feat --out-dir data/prereg17/smoke/tabicl)
Normally started through run_p17_tabicl.sh (detached; STOP file; retries while the GPU is busy).

Fits TabICLClassifier once per process on (Z_ctx, y_ctx) and predicts Z_inner in fixed chunks of 2,000 rows; a chunk
file that exists and matches the inputs is kept (resume). A file named STOP in the output folder stops the run
between chunks. Exit codes: 0 all chunks done; 3 stopped by STOP; 4 GPU busy (nothing computed); 6 the driver
differs (nothing computed); 1 refused or error (SystemExit with a message).

Module top level imports nothing but the standard library (the .venv tests read the constants below); numpy, torch
and tabicl are imported inside run().
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import datetime

sys.dont_write_bytecode = True

CHUNK = 2000
L = 10
DRIVER = "616.92"
BUSY_NAMES = ("python", "katago")
MAX_USED_MIB = 4000      # desktop apps hold about 2 GB on this WDDM card; more means another GPU job
CKPT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
TABICL_VERSION = "2.2.0"
TORCH_VERSION = "2.11.0+cu128"
TABICL_PARAMS = {"n_estimators": 8, "norm_methods": None, "feat_shuffle_method": "latin",
                 "class_shuffle_method": "shift", "outlier_threshold": 4.0, "softmax_temperature": 0.9,
                 "average_logits": True, "support_many_classes": True, "batch_size": 8, "kv_cache": False,
                 "allow_auto_download": False, "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
                 "device": "cuda", "use_amp": "auto", "use_fa3": "auto", "offload_mode": "auto", "random_state": 0}
INPUTS = ("Z_ctx.npy", "y_ctx.npy", "Z_inner.npy")
CHUNK_FORMAT = "c10-prereg17 tabicl chunk v1"
RUN_FORMAT = "c10-prereg17 tabicl run v1"


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def gpu_state():
    """-> (driver version, GPU name, [compute processes named like a GPU job], memory used MiB)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    first = (q.stdout.strip().splitlines() or [""])[0]
    parts = [x.strip() for x in first.split(",")]
    driver = parts[0] if parts else ""
    name = parts[1] if len(parts) > 1 else ""
    try:
        used = int(float(parts[2]))
    except (IndexError, ValueError):
        used = None
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True)
    lines = [x.strip() for x in a.stdout.splitlines() if x.strip()]
    busy = [x for x in lines if any(b in x.lower() for b in BUSY_NAMES) and str(os.getpid()) != x.split(",")[0].strip()]
    return driver, name, busy, used


def chunk_bounds(n, size=CHUNK):
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def chunk_path(out, s):
    return os.path.join(out, f"chunk_{s // CHUNK:05d}.npz")


def chunk_ok(path, s, e, inputs_sha):
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as z:
            return (str(z["format"]) == CHUNK_FORMAT and int(z["start"]) == s and int(z["end"]) == e
                    and z["prob"].shape == (e - s, L) and json.loads(str(z["inputs_sha256"])) == inputs_sha)
    except Exception:  # noqa: BLE001 (a broken partial file is recomputed)
        return False


def versions():
    from importlib import metadata
    dists = sorted((d.metadata["Name"], d.version) for d in metadata.distributions())
    return {"python": sys.version.split()[0], "packages": {n: v for n, v in dists}}


def run(a):
    t0 = time.time()
    feat, out = os.path.abspath(a.feat_dir), os.path.abspath(a.out_dir)
    for p in (feat, out, a.ckpt):
        if "data/local" in os.path.abspath(p).replace("\\", "/").lower():
            raise SystemExit("refused path")
    os.makedirs(out, exist_ok=True)
    driver, gname, busy, used = gpu_state()
    if driver != DRIVER:
        print(f"driver {driver!r} != {DRIVER}: not running", flush=True)
        return 6
    if busy or used is None or used > MAX_USED_MIB:
        print(f"GPU busy (compute processes {busy}, memory used {used} MiB, limit {MAX_USED_MIB})", flush=True)
        return 4
    ver = versions()
    if ver["packages"].get("tabicl") != TABICL_VERSION or ver["packages"].get("torch") != TORCH_VERSION:
        raise SystemExit(f"tabicl / torch versions differ: {ver['packages'].get('tabicl')}, "
                         f"{ver['packages'].get('torch')}")
    ck_sha = sha_file(a.ckpt)
    if ck_sha != CKPT_SHA256:
        raise SystemExit("checkpoint SHA-256 differs from PLAN.md 1")
    with open(os.path.join(feat, "feat_record.json"), encoding="utf-8") as f:
        frec = json.load(f)
    inputs_sha = {k: sha_file(os.path.join(feat, k)) for k in INPUTS}
    if any(frec["files_sha256"][k] != inputs_sha[k] for k in INPUTS):
        raise SystemExit("inputs differ from feat_record.json")
    inputs_sha["ckpt"] = ck_sha
    import numpy as np
    import torch
    from tabicl import TabICLClassifier
    Zc = np.load(os.path.join(feat, "Z_ctx.npy"))
    yc = np.load(os.path.join(feat, "y_ctx.npy"))
    Zi = np.load(os.path.join(feat, "Z_inner.npy"), mmap_mode="r")
    bounds = chunk_bounds(len(Zi))
    todo = [(s, e) for s, e in bounds if not chunk_ok(chunk_path(out, s), s, e, inputs_sha)]
    rec_path = os.path.join(out, "run_record.json")
    prior = {}
    if os.path.exists(rec_path):
        with open(rec_path, encoding="utf-8") as f:
            prior = json.load(f)
    sessions = list(prior.get("sessions") or [])
    sess = {"started": now_iso(), "pid": os.getpid(), "chunks_todo": len(todo), "chunks_done": 0, "gpu": gname,
            "driver": driver, "memory_used_mib_at_start": used, "mem_get_info": []}
    print(f"{len(todo)} of {len(bounds)} chunks to compute", flush=True)
    status = "done"
    classes = None
    if todo:
        clf = TabICLClassifier(model_path=a.ckpt, **TABICL_PARAMS)
        tf = time.time()
        sess["mem_get_info"].append(["fit"] + [int(x) for x in torch.cuda.mem_get_info()])
        clf.fit(Zc, yc)
        sess["fit_seconds"] = round(time.time() - tf, 1)
        classes = [int(x) for x in clf.classes_]
        if classes != list(range(L)):
            raise SystemExit(f"classes_ {classes} != 0..9")
        for s, e in todo:
            if os.path.exists(os.path.join(out, "STOP")):
                status = "stopped"
                break
            tc = time.time()
            sess["mem_get_info"].append([s] + [int(x) for x in torch.cuda.mem_get_info()])
            P = np.asarray(clf.predict_proba(np.asarray(Zi[s:e])))
            p = chunk_path(out, s)
            tmp = p + ".tmp.npz"
            np.savez(tmp, format=np.array(CHUNK_FORMAT), start=np.array(s), end=np.array(e), prob=P,
                     inputs_sha256=np.array(json.dumps(inputs_sha, sort_keys=True)),
                     seconds=np.array(time.time() - tc), created=np.array(now_iso()))
            os.replace(tmp, p)
            sess["chunks_done"] += 1
            print(f"chunk [{s}, {e}) {time.time() - tc:.1f}s (peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)",
                  flush=True)
    sess["seconds"] = round(time.time() - t0, 1)
    sess["status"] = status
    sessions.append(sess)
    missing = [s for s, e in bounds if not chunk_ok(chunk_path(out, s), s, e, inputs_sha)]
    rec = {"format": RUN_FORMAT, "updated": now_iso(), "feat_dir": feat, "out_dir": out, "ckpt": a.ckpt,
           "ckpt_sha256": ck_sha, "inputs_sha256": inputs_sha, "params": TABICL_PARAMS, "chunk": CHUNK,
           "rows": int(len(Zi)), "context_rows": int(len(Zc)), "chunks": len(bounds), "missing_chunks": len(missing),
           "classes": classes if classes is not None else prior.get("classes"),
           "complete": not missing, "versions": ver, "sessions": sessions,
           "gpu_seconds_total": round(sum(x.get("seconds", 0) for x in sessions), 1),
           "chunk_sha256": {os.path.basename(chunk_path(out, s)): sha_file(chunk_path(out, s))
                            for s, e in bounds if s not in missing}}
    write_json(rec_path, rec)
    print(f"-> {rec_path}: complete {rec['complete']}, {sess['seconds']}s this session", flush=True)
    return 0 if rec["complete"] else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--feat-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                                  "data", "prereg17", "downloads", "hf",
                                                  "tabicl-classifier-v2-20260212.ckpt"))
    a = ap.parse_args(argv)
    a.ckpt = os.path.abspath(a.ckpt)
    return {"run": run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
