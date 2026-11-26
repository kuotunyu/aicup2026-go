"""Run a Colab notebook locally on CPU in SMOKE mode (tiny data), cell by cell.

Usage (from the repo root, with the smoke virtualenv's python):
    python colab/run_local_smoke.py colab/prereg01_rank.ipynb --scratch <dir>

What it does:
  * creates <dir>/drive (a stand-in for MyDrive/C10) holding the bundle zip,
    and <dir>/work (a stand-in for /content/c10);
  * sets C10_SMOKE=1, C10_DRIVE_DIR, C10_WORK_DIR so the notebook takes its
    local code path (the notebook itself unzips and checks the bundle);
  * executes every code cell in one namespace and stops at the first error;
  * on Windows, replaces multiprocessing Process/Pool with thread-based
    stand-ins after each cell, because notebook-defined workers cannot be
    pickled under "spawn" (Colab uses Linux "fork", so nothing changes there).
Never uses the GPU: run it with a CPU-only torch.
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time
import traceback
from multiprocessing.dummy import Pool as ThreadPool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ThreadProcess:
    def __init__(self, target=None, args=(), kwargs=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}
        self.exitcode = None
        self._thread = threading.Thread(target=self._run)

    def _run(self):
        try:
            self._target(*self._args, **self._kwargs)
            self.exitcode = 0
        except BaseException:
            traceback.print_exc()
            self.exitcode = 1

    def start(self):
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--bundle", default=os.path.join(ROOT, "data/colab_bundle/C10_bundle_v1.zip"))
    ap.add_argument("--fresh", action="store_true", help="delete work/drive dirs first")
    a = ap.parse_args()

    drive = os.path.join(a.scratch, "drive")
    work = os.path.join(a.scratch, "work")
    if a.fresh:
        shutil.rmtree(drive, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(drive, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    target = os.path.join(drive, os.path.basename(a.bundle))
    if not os.path.exists(target):
        shutil.copyfile(a.bundle, target)
    os.environ.update({"C10_SMOKE": "1", "C10_DRIVE_DIR": drive, "C10_WORK_DIR": work})

    import torch
    if torch.cuda.is_available():
        raise SystemExit("smoke runs must use a CPU-only torch (the 4090 is reserved)")

    with open(a.notebook, encoding="utf-8") as f:
        nb = json.load(f)
    ns = {"__name__": "__main__"}
    log = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # IPython magics (!cmd, %magic) are not Python, so compile() rejects
        # them; a '%' that merely starts a continuation line is plain Python.
        try:
            code = compile(src, f"<cell {i}>", "exec")
        except SyntaxError as e:
            raise SystemExit(f"cell {i}: not plain Python (IPython magic?): {e}")
        t0 = time.time()
        print(f"\n===== cell {i} =====", flush=True)
        try:
            exec(code, ns)
        except BaseException:
            traceback.print_exc()
            log.append({"cell": i, "status": "error", "seconds": round(time.time() - t0, 1)})
            print(json.dumps({"result": "FAILED", "cells": log}, indent=1))
            sys.exit(1)
        if os.name == "nt":
            if "Process" in ns:
                ns["Process"] = ThreadProcess
            if "Pool" in ns:
                ns["Pool"] = ThreadPool
        log.append({"cell": i, "status": "ok", "seconds": round(time.time() - t0, 1)})
    print(json.dumps({"result": "OK", "cells": log}, indent=1))


if __name__ == "__main__":
    main()
