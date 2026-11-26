"""prereg62: local CPU dry run of a notebook's rebuild cells and of p62_run.py in the rebuilt work dir (no GPU, no
Drive, no score). Proves that kit + prereg51 overlay + overlay2 + the embedded p62_run.py are all the files needed.

    GPUPY results/prereg62/explore/p62_dryrun.py --work <empty dir outside the repo> [--arm S1]

The notebook's own cell sources are executed (cells 2-5: kit, overlay, overlay2, wrapper) with only the Colab paths
replaced by local ones (kit zip data/prereg42/colab/upload, parts data/prereg51/colab/parts and
data/prereg62/colab/parts) and the Drive-mount / GPU-query lines dropped; every cell is compiled. Then, in the work
dir: 2 smoke steps on the CPU, auto-resume to step 3, an unreadable newest checkpoint set aside, and 1 smoke step
of arm X. Prints one JSON line. The caller deletes the work dir.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DROP = ("from google.colab", "drive.mount", "print(subprocess.run(['nvidia-smi'", "tv = ", "print('torch', tv)",
        "assert tv.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--arm", default="S1")
    a = ap.parse_args()
    top = os.path.abspath(a.work)
    inside = os.path.normcase(top + os.sep).startswith(os.path.normcase(REPO + os.sep))
    if inside or (os.path.exists(top) and os.listdir(top)):
        raise SystemExit("--work must be an empty directory outside the repo")
    work, fake = os.path.join(top, "c10"), os.path.join(top, "drive")
    for d in ("p42", "p51", "p62"):
        os.makedirs(os.path.join(fake, d))
    nb = json.load(open(os.path.join(REPO, "results", "prereg62", "colab", "p62_%s.ipynb" % a.arm), encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"]]
    for i, s in enumerate(cells[1:], 1):
        compile(s, "cell%d" % i, "exec")
    rep = [("'/content/drive/MyDrive/C10/p42'", repr(os.path.join(fake, "p42"))),
           ("'/content/drive/MyDrive/C10/p51'", repr(os.path.join(fake, "p51"))),
           ("'/content/drive/MyDrive/C10/p62'", repr(os.path.join(fake, "p62"))),
           ("'/content/c10'", repr(work)),
           ("os.path.join(DRIVE, 'C10_p42_kit.zip')",
            repr(os.path.join(REPO, "data", "prereg42", "colab", "upload", "C10_p42_kit.zip"))),
           ("os.path.join(P51, 'overlay_parts', 'overlay.part*')",
            repr(os.path.join(REPO, "data", "prereg51", "colab", "parts", "overlay.part*"))),
           ("os.path.join(P62, 'overlay2_parts', 'overlay2.part*')",
            repr(os.path.join(REPO, "data", "prereg62", "colab", "parts", "overlay2.part*"))),
           ("'/content/C10_p51_overlay.zip'", repr(os.path.join(top, "o1.zip"))),
           ("'/content/C10_p62_overlay2.zip'", repr(os.path.join(top, "o2.zip")))]
    ns = {}
    for i in (2, 3, 4, 5):
        s = "\n".join(x for x in cells[i].split("\n") if not x.startswith(DROP))
        for old, new in rep:
            s = s.replace(old, new)
        if "/content" in s:
            raise SystemExit("cell %d still has a Colab path" % i)
        exec(compile(s, "cell%d" % i, "exec"), ns)
    for z in ("o1.zip", "o2.zip"):
        os.remove(os.path.join(top, z))
    n_files = sum(len(f) for _r, _d, f in os.walk(work))

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    run = "results/prereg62/explore/p62_run.py"

    def go(arm, *extra):
        p = subprocess.run([sys.executable, "-u", run, "train", "--arm", arm, "--smoke", "--device", "cpu",
                            "--workers", "0", "--min-free-ram-mb", "2048", "--log-every", "1"] + list(extra),
                           cwd=work, env=env, capture_output=True, text=True)
        print(p.stdout[-1500:], p.stderr[-1500:], flush=True)
        if p.returncode != 0:
            raise SystemExit("smoke failed (%s %s): rc %d" % (arm, extra, p.returncode))

    def names(arm):
        return sorted(os.path.basename(p) for p in
                      glob.glob(os.path.join(work, "data", "prereg62", arm, "smoke", "S0", "ckpt", "*")))

    res = {"arm": a.arm, "files_in_work_dir": n_files}
    go(a.arm, "--steps", "2")
    res["after_2_steps"] = names(a.arm)
    go(a.arm, "--steps", "3", "--auto-resume")
    res["after_resume_to_3"] = names(a.arm)
    sm = os.path.join(work, "data", "prereg62", a.arm, "smoke", "S0")
    with open(os.path.join(sm, "ckpt", "step_000004.pt"), "wb") as f:
        f.write(b"truncated")
    go(a.arm, "--steps", "3", "--auto-resume")
    res["after_unreadable_newest"] = names(a.arm)
    rec = json.load(open(os.path.join(sm, "train_record.json"), encoding="utf-8"))
    c = rec["config"]["p60"]
    res["record"] = {k: c[k] for k in ("seed", "total_steps", "lr", "warmup", "first_batch", "n_planes", "n_classes")}
    res["record"]["steps"] = rec["steps"]
    res["code_files"] = sorted(rec["code_sha256"])
    res["lr_log"] = [json.loads(x)["lr"] for x in open(os.path.join(sm, "train_log.jsonl"), encoding="utf-8")]
    go("X", "--steps", "1")
    rx = json.load(open(os.path.join(work, "data", "prereg62", "X", "smoke", "S0", "train_record.json"),
                        encoding="utf-8"))["config"]["p60"]
    res["record_X"] = {k: rx[k] for k in ("seed", "total_steps", "lr", "warmup", "first_batch")}
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
