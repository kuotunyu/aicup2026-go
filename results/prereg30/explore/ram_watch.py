"""prereg30: RAM watch for the GPU pilot (scheduler rule 11/09: available RAM < 3 GB -> create the STOP file).

    .venv/Scripts/python.exe results/prereg30/explore/ram_watch.py   (repository root; started detached next to run_pilot.sh)
Every 10 s: available RAM, and the summed RSS of the p30_extract process and its children. Below 3 GB available it
creates data/prereg30/pilot/STOP (the run ends after the chunk in progress). Ends when run.log has 'launcher end'.
Writes data/prereg30/pilot/ram_watch.jsonl (min available, peak RSS). Reads no data.
"""
import json
import os
import time

import psutil

D = os.path.join("data", "prereg30", "pilot")
LOG = os.path.join(D, "run.log")
OUT = os.path.join(D, "ram_watch.jsonl")
LIMIT = 3 * 2 ** 30


def pilot_rss():
    tot = 0
    for p in psutil.process_iter(["cmdline"]):
        try:
            if any("p30_extract.py" in c for c in (p.info["cmdline"] or [])):
                tot += p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return tot


def main():
    min_av, peak, stopped, t0 = None, 0, False, time.time()
    while True:
        av = psutil.virtual_memory().available
        rss = pilot_rss()
        min_av = av if min_av is None else min(min_av, av)
        peak = max(peak, rss)
        if av < LIMIT and not stopped:
            open(os.path.join(D, "STOP"), "w").close()
            stopped = True
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "avail_gb": round(av / 2 ** 30, 2),
               "pilot_rss_gb": round(rss / 2 ** 30, 2), "min_avail_gb": round(min_av / 2 ** 30, 2),
               "peak_pilot_rss_gb": round(peak / 2 ** 30, 2), "stop_written": stopped}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        try:
            done = "launcher end" in open(LOG, encoding="utf-8", errors="replace").read()
        except OSError:
            done = False
        if done or time.time() - t0 > 4 * 3600:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
