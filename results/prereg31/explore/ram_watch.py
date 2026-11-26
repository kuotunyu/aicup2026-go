"""prereg31: RAM watch for the S31 training (PLAN.md revision 1 item 2: available RAM < 4 GB -> STOP file).

    python -I results/prereg31/explore/ram_watch.py --log data/prereg31/train_log.txt --stop data/prereg31/STOP
        --out data/prereg31/ram_watch.jsonl --run-id <id>         (started by run_train.sh next to the training)
Every 10 s: available physical RAM (kernel32 GlobalMemoryStatusEx; standard library only). Below 4 GB available it
creates the STOP file once: p31_train then stops at the next held-out evaluation or before the next seed. To go on,
delete the STOP file by hand and start run_train.sh --resume (same commit and code). Ends when the log has the
launcher line "end ... run <id>", or after 48 h. Reads no data and imports nothing from the repository.
"""
import argparse
import ctypes
import json
import os
import sys
import time

LIMIT_GB = 4.0
EVERY_S = 10
MAX_HOURS = 48


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_bytes():
    """Available physical RAM in bytes (Windows)."""
    st = MemoryStatusEx()
    st.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(st)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(st.ullAvailPhys)


def must_stop(avail, limit_gb=LIMIT_GB):
    return bool(avail < limit_gb * 2 ** 30)


def run_ended(log_path, run_id):
    """True when the launcher wrote its end line for this run."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            return any(ln.startswith("end ") and ln.rstrip().endswith("run " + run_id) for ln in f)
    except OSError:
        return False


def watch(log_path, stop_path, out_path, run_id, limit_gb=LIMIT_GB, every=EVERY_S, max_hours=MAX_HOURS,
          avail=available_bytes, sleep=time.sleep):
    min_av, stopped, t0, n = None, False, time.time(), 0
    while True:
        av = avail()
        min_av = av if min_av is None else min(min_av, av)
        if must_stop(av, limit_gb) and not stopped:
            os.makedirs(os.path.dirname(os.path.abspath(stop_path)), exist_ok=True)
            with open(stop_path, "w", encoding="utf-8") as f:
                f.write("ram_watch: available RAM %.2f GB < %.1f GB at %s (run %s)\n"
                        % (av / 2 ** 30, limit_gb, time.strftime("%Y-%m-%dT%H:%M:%S"), run_id))
            stopped = True
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": run_id, "avail_gb": round(av / 2 ** 30, 2),
               "min_avail_gb": round(min_av / 2 ** 30, 2), "limit_gb": limit_gb, "stop_written": stopped}
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if run_ended(log_path, run_id) or time.time() - t0 > max_hours * 3600:
            return {"samples": n, "min_avail_gb": rec["min_avail_gb"], "stop_written": stopped}
        sleep(every)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--stop", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(watch(a.log, a.stop, a.out, a.run_id)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
